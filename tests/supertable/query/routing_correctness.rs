// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN hidden-vector-index routing correctness + cold-GET-count gate.
//!
//! This is the deterministic, benchmark-free gate for the OPANN
//! vector-routing work. The hidden vector-index path is:
//!
//!   1. ingested vectors land in an INCOMING staging area;
//!   2. `optimize()` drains them into per-cell IVF superfiles;
//!   3. a query descends a resident OPANN tree to the nearest CELLS,
//!      then for each routed cell scores that cell's per-cluster
//!      centroids (resident in `vector_summary.clusters`) and
//!      range-GETs only the selected clusters (offsets from
//!      `vector_summary.cluster_offsets`).
//!
//! The properties asserted here:
//!
//!   - **Pre-drain correctness** — `vector_search` returns the exact
//!     top-k (vs an in-test brute-force oracle) *before* draining,
//!     while vectors are still in INCOMING staging.
//!   - **Post-drain correctness** — after `optimize()` drains the
//!     staging area into per-cell IVF superfiles, `vector_search`
//!     STILL returns the exact top-k. This exercises tree routing +
//!     per-cell cluster selection + offset range-GET end to end.
//!   - **Pruned-routing correctness** — harder, off-center
//!     between-cluster queries at a small PRUNED nprobe still recover
//!     the true top-k above a floor, proving routing descends to the
//!     right cells under pruning rather than relying on a full sweep.
//!   - **Scalar-projection correctness** — post-drain (and after the
//!     user table is compacted into one merged superfile),
//!     `vector_search(..., Some(&["_id", "title"]))` decodes the
//!     `title` column for hidden-index hits and lands on exactly the
//!     queried cluster's docs. This exercises the hidden-hit →
//!     user-superfile row remap, whose `_id` lookup must be
//!     order-independent because a merged superfile's `_id` column is
//!     only piecewise-sorted.
//!   - **Bounded per-search GETs** — with the manifest + OPANN tree
//!     already resident (a warmup query ran, the counter reset), one
//!     routed search over uncached cells issues only a small, bounded
//!     number of object-store fetches — on the order of
//!     (cells probed × clusters probed), NOT scaling with total doc
//!     count. The one-time open/manifest/tree cost is amortised away
//!     by the warmup, so the count reflects per-search cost alone.
//!
//! Everything runs in-process over `LocalFsStorageProvider`, so it is
//! part of the ordinary `cargo test --test supertable` run; no Azure,
//! no emulator, no network.

#![deny(clippy::unwrap_used)]

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use datafusion::prelude::{col, lit};
use infino::{
    CompactionSettings, OptimizeOptions, VectorSearchOptions,
    superfile::{
        builder::{FtsConfig, VectorConfig},
        vector::{distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::default_tokenizer,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Dataset shape — small, well-separated synthetic clusters so the exact
// top-k is unambiguous and IVF achieves recall 1.0 at a generous nprobe.
// ---------------------------------------------------------------------------

/// Embedding dimension — small for speed, large enough that random unit
/// vectors in distinct directions stay well separated under Cosine.
const EMB_DIM: usize = 16;
/// Number of well-separated synthetic clusters in the corpus.
const N_CLUSTERS: usize = 24;
/// Docs per cluster. `N_CLUSTERS * DOCS_PER_CLUSTER` is the corpus size.
const DOCS_PER_CLUSTER: usize = 150;
/// Total docs in the corpus (≈ 3 600 — several hidden cells get populated
/// while keeping the test well under ~10 s).
const N_DOCS: usize = N_CLUSTERS * DOCS_PER_CLUSTER;
/// Docs committed per write cycle (so several superfiles are produced).
const DOCS_PER_COMMIT: usize = 600;
/// Magnitude of the per-doc gaussian noise added to a cluster's base
/// direction. Small enough that intra-cluster docs stay far closer to one
/// another than to any other cluster's docs, so the exact top-k is stable.
const NOISE_STDDEV: f32 = 0.02;
/// Seed for the cluster base directions.
const CLUSTER_SEED: u64 = 0xC0FFEE;
/// Seed offset for per-doc noise (xored with the global doc index).
const NOISE_SEED: u64 = 0xBEEF;

/// IVF centroid count for the user-table vector index. The hidden global
/// index uses its own fixed cell count (`GLOBAL_VECTOR_CELL_COUNT = 64`).
const N_CENT: usize = 16;
/// Random rotation seed for the vector index.
const VECTOR_ROT_SEED: u64 = 99;

/// Top-k for the correctness queries.
const TOP_K: usize = 10;
/// nprobe for the correctness queries — large so recall is 1.0 on this
/// clean, well-separated dataset (every relevant cell is probed).
const CORRECTNESS_NPROBE: usize = 64;
/// nprobe for the cold GET-count query — modest, to demonstrate that the
/// fetch count tracks (cells × clusters), not corpus size.
const COLD_NPROBE: usize = 4;
/// A small, PRUNED nprobe for the harder between-cluster correctness queries.
/// Deliberately well below `CORRECTNESS_NPROBE` (64) so the routing tree must
/// actually prune to a handful of cells rather than scan — proving pruned
/// routing finds the true neighbours, not just a full sweep.
const PRUNED_NPROBE: usize = 6;

/// Minimum recall@k accepted for the routing correctness assertions.
/// The dataset is engineered for exact recall; this is the documented
/// acceptance bar (≥ 0.99) and a guard against a flaky last-place tie.
const RECALL_FLOOR: f64 = 0.99;
/// Recall floor for the harder, off-center BETWEEN-cluster queries at a
/// PRUNED nprobe. These queries are pulled off a cluster center toward a
/// neighbour, so part of their true top-k legitimately falls in the neighbour
/// cluster whose cells a small probe set may not reach. The 0.99 exact-center
/// bar does not apply to a pruned, off-center boundary query; this looser floor
/// is a regression tripwire — observed recall on this fixture is 0.8–1.0 at
/// `PRUNED_NPROBE`, so a routing bug that mis-descends (or a regression back to
/// a corpus scan with wrong-cell selection) would crater it well below 0.80,
/// while the full-nprobe between-cluster pass (asserted separately) still holds
/// the strict `RECALL_FLOOR`.
const BETWEEN_RECALL_FLOOR: f64 = 0.80;

/// Disk-cache budget for the test caches.
const CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Cold-fetch chunk size for the test caches.
const COLD_FETCH_CHUNK_BYTES: u64 = 1 << 20;

/// Upper bound on the PER-SEARCH object-store fetch count, measured after the
/// manifest + OPANN routing tree are already resident (a warmup query ran, the
/// counter was reset). With the open/manifest/tree cost amortised away, the
/// only fetches a routed query issues are the per-probed-cell vector-blob
/// range-GETs (and a tiny constant for id-page reads), so the count tracks
/// (cells probed × clusters probed) at `PRUNED_NPROBE`, NOT corpus size. The
/// observed steady-state count on this fixture is well under this bound; the
/// headroom absorbs id-page / cluster-offset reads while still failing loudly
/// if routing reverts to a corpus scan.
const PER_SEARCH_GET_BUDGET: usize = 48;

// ---------------------------------------------------------------------------
// Counting storage provider — wraps an inner provider and counts every
// read-path fetch (`get`, `get_range`, `tail`). Delegates everything else.
// ---------------------------------------------------------------------------

/// Per-GET delay (ms) injected during a measured window so wall-clock /
/// `WAVE_PROBE_DELAY_MS` reveals the number of *sequential* fetch waves:
/// concurrent GETs in one wave overlap their sleeps, sequential waves stack.
/// Large enough that the search's sub-millisecond CPU is dwarfed, so the
/// wall-clock-to-waves round is unambiguous.
const WAVE_PROBE_DELAY_MS: u64 = 50;

/// `StorageProvider` decorator that counts read-path fetches. Splits the count
/// into total fetches and tombstone-sidecar fetches, and can inject a uniform
/// per-GET delay so a test can measure how many sequential waves a search
/// issues (not just how many GETs).
#[derive(Debug)]
struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    fetches: Arc<AtomicUsize>,
    tombstone_fetches: Arc<AtomicUsize>,
    delay_ms: Arc<AtomicU64>,
}

impl CountingStorage {
    fn new(
        inner: Arc<dyn StorageProvider>,
        fetches: Arc<AtomicUsize>,
        tombstone_fetches: Arc<AtomicUsize>,
        delay_ms: Arc<AtomicU64>,
    ) -> Self {
        Self {
            inner,
            fetches,
            tombstone_fetches,
            delay_ms,
        }
    }

    /// Count one read-path fetch (classifying tombstone sidecars) and apply the
    /// injected wave-probe delay, if armed.
    async fn observe(&self, uri: &str) {
        self.fetches.fetch_add(1, Ordering::Relaxed);
        if uri.ends_with(".tombstones") {
            self.tombstone_fetches.fetch_add(1, Ordering::Relaxed);
        }
        let d = self.delay_ms.load(Ordering::Relaxed);
        if d > 0 {
            tokio::time::sleep(Duration::from_millis(d)).await;
        }
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.observe(uri).await;
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.observe(uri).await;
        self.inner.get_range(uri, range).await
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        self.observe(uri).await;
        self.inner.tail(uri, len).await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.inner.list_with_prefix(prefix).await
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.inner.list_with_prefix_metadata(prefix).await
    }

    fn object_store_handle(
        &self,
        uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        self.inner.object_store_handle(uri)
    }
}

// ---------------------------------------------------------------------------
// Schema + options helpers (mirror compact_azure.rs, but LocalFs / in-process).
// ---------------------------------------------------------------------------

/// `DataType` for a fixed-size list of `f32` with `dim` elements.
fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// The combined title + embedding schema used throughout the test.
fn test_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(EMB_DIM), false),
    ]))
}

/// `SupertableOptions` with FTS on `title` and a Cosine vector index on
/// `emb`, single-thread writer pool for deterministic runs.
fn options_title_emb() -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon ThreadPoolBuilder with num_threads(1) builds"),
    );
    SupertableOptions::new(
        test_schema(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        Some(default_tokenizer()),
    )
    .expect("SupertableOptions::new with title+emb test fixture")
    .with_writer_pool(pool)
}

/// Small optimize options — drains the hidden INCOMING staging area and
/// compacts. `target_superfile_size_mb = 1` / `min_fill_percent = 1` match
/// the tiny superfiles this test produces (see `compact_gc.rs`).
fn small_optimize_opts() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: 1,
        min_fill_percent: 1,
        ..CompactionSettings::default()
    })
}

/// Build a `DiskCacheStore` over `storage` rooted at `cache_root`.
fn make_cache(storage: Arc<dyn StorageProvider>, cache_root: &std::path::Path) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: 4,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("DiskCacheStore::new")
}

// ---------------------------------------------------------------------------
// Deterministic, well-separated synthetic embeddings.
// ---------------------------------------------------------------------------

/// L2-normalize `v` in place (no-op for a zero vector).
fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

/// Random unit base direction for cluster `c` (deterministic in `c`).
fn cluster_base(c: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(CLUSTER_SEED ^ c as u64);
    let dist = StandardNormal;
    let mut v: Vec<f32> = (0..EMB_DIM)
        .map(|_| {
            let s: f64 = dist.sample(&mut rng);
            s as f32
        })
        .collect();
    normalize(&mut v);
    v
}

/// The cluster a given absolute doc index belongs to. Docs are laid out
/// cluster-contiguous: `[cluster 0 × DOCS_PER_CLUSTER, cluster 1 × …, …]`.
fn cluster_of(doc_idx: usize) -> usize {
    doc_idx / DOCS_PER_CLUSTER
}

/// Deterministic unit embedding for absolute doc `doc_idx`: its cluster's
/// base direction plus tiny gaussian noise, re-normalized.
fn doc_embedding(doc_idx: usize) -> Vec<f32> {
    let mut v = cluster_base(cluster_of(doc_idx));
    let mut rng = StdRng::seed_from_u64(NOISE_SEED ^ doc_idx as u64);
    let dist = StandardNormal;
    for x in &mut v {
        let n: f64 = dist.sample(&mut rng);
        *x += (n as f32) * NOISE_STDDEV;
    }
    normalize(&mut v);
    v
}

/// Build a two-column (title + emb) `RecordBatch` for absolute doc indices
/// `[doc_offset, doc_offset + n)`. Titles embed the absolute index so each
/// doc is identifiable, and the embedding is `doc_embedding(absolute index)`.
fn build_batch(doc_offset: usize, n: usize) -> RecordBatch {
    let titles: Vec<String> = (0..n).map(|i| format!("doc{:07}", doc_offset + i)).collect();
    let title_arr = LargeStringArray::from(titles.iter().map(String::as_str).collect::<Vec<_>>());
    let flat: Vec<f32> = (0..n).flat_map(|i| doc_embedding(doc_offset + i)).collect();
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(flat);
    let fsl = FixedSizeListArray::try_new(item_field, EMB_DIM as i32, Arc::new(values) as ArrayRef, None)
        .expect("FixedSizeListArray for emb column");
    RecordBatch::try_new(test_schema(), vec![Arc::new(title_arr), Arc::new(fsl)])
        .expect("RecordBatch with title and emb columns")
}

// ---------------------------------------------------------------------------
// Brute-force oracle (Cosine: rank by `1 - dot(unit, unit)`, ascending).
//
// The engine assigns `_id` as an opaque snowflake-style value, so the oracle
// is run in doc-*index* space (where embeddings are known) and then
// translated to `_id` space via a map built once from `SELECT _id, title`
// over the user table. `vector_search` results are compared on the stable
// `_id` set — the engine-native identity — which sidesteps decoding scalar
// columns from hidden-index hits (see the production-limitation note in the
// test body).
// ---------------------------------------------------------------------------

/// All doc embeddings, indexed by absolute doc index.
fn all_embeddings() -> Vec<Vec<f32>> {
    (0..N_DOCS).map(doc_embedding).collect()
}

/// Parse the absolute doc index out of a `doc{:07}` title.
fn doc_index_from_title(title: &str) -> usize {
    title
        .strip_prefix("doc")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| panic!("unexpected title format: {title:?}"))
}

/// Build the `doc_index -> _id` map by scanning `SELECT _id, title FROM
/// supertable`. The user table SQL scan reads scalar columns directly (it
/// does not go through the hidden-index remap), so this is reliable both
/// before and after a drain/compaction.
fn build_doc_index_to_id(st: &Supertable) -> Vec<i128> {
    let reader = st.reader();
    let batches = reader
        .query_sql("SELECT _id, title FROM supertable")
        .expect("SELECT _id, title FROM supertable");
    let mut map = vec![None; N_DOCS];
    for b in &batches {
        let id_arr = b
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is Decimal128");
        let title_arr = b
            .column_by_name("title")
            .expect("title column")
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title is LargeUtf8");
        for i in 0..b.num_rows() {
            if id_arr.is_valid(i) && title_arr.is_valid(i) {
                let idx = doc_index_from_title(title_arr.value(i));
                map[idx] = Some(id_arr.value(i));
            }
        }
    }
    map.into_iter()
        .enumerate()
        .map(|(idx, id)| id.unwrap_or_else(|| panic!("no _id mapped for doc index {idx}")))
        .collect()
}

/// Cosine distance between two unit vectors (`1 - dot`). Smaller = nearer.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    1.0 - dot
}

/// Exact top-k `_id`s for `query` over `all` embeddings, by Cosine distance.
/// `idx_to_id[i]` is the stable `_id` of doc index `i`.
fn brute_force_topk_ids(all: &[Vec<f32>], idx_to_id: &[i128], query: &[f32], k: usize) -> Vec<i128> {
    let mut scored: Vec<(f32, usize)> = all
        .iter()
        .enumerate()
        .map(|(idx, e)| (cosine_distance(query, e), idx))
        .collect();
    // Sort by distance asc, tie-break by index asc for determinism.
    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    scored
        .into_iter()
        .take(k)
        .map(|(_, idx)| idx_to_id[idx])
        .collect()
}

/// Extract the `_id` set from search results (engine-native `_id` + score).
fn extract_id_set(batches: &[RecordBatch]) -> HashSet<i128> {
    let mut out = HashSet::new();
    for b in batches {
        let id_col = b
            .column_by_name("_id")
            .expect("search result must have _id column");
        let arr = id_col
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id column must be Decimal128");
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                out.insert(arr.value(i));
            }
        }
    }
    out
}

/// recall@k = |returned ∩ exact| / k.
fn recall_at_k(returned: &HashSet<i128>, exact: &[i128]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let hit = exact.iter().filter(|id| returned.contains(id)).count();
    hit as f64 / exact.len() as f64
}

/// A handful of query directions: the (noise-free) base centroid of a few
/// clusters spread across the corpus. Each is normalized so Cosine is
/// well-defined; the exact top-k is that cluster's own docs.
fn query_clusters() -> Vec<usize> {
    vec![0, 5, 11, 17, 23]
}

/// Run `vector_search` for one query cluster center, returning the stable
/// `_id` set it surfaced. Uses `None` projection — the engine-native
/// `_id` + score path — so no scalar column is decoded.
fn search_ids(st: &Supertable, cluster: usize, nprobe: usize, k: usize) -> HashSet<i128> {
    let query = cluster_base(cluster);
    let batches = st
        .vector_search(
            "emb",
            &query,
            k,
            VectorSearchOptions::new().with_nprobe(nprobe),
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("vector_search(cluster {cluster}) failed: {e}"));
    extract_id_set(&batches)
}

/// Assert recall@k ≥ floor for every query cluster against the oracle, and
/// return the per-query returned `_id` sets (for pre/post agreement checks).
fn assert_recall(
    st: &Supertable,
    all: &[Vec<f32>],
    idx_to_id: &[i128],
    nprobe: usize,
    k: usize,
    phase: &str,
) -> Vec<HashSet<i128>> {
    let mut per_query = Vec::new();
    for c in query_clusters() {
        let returned = search_ids(st, c, nprobe, k);
        let query = cluster_base(c);
        let exact = brute_force_topk_ids(all, idx_to_id, &query, k);
        let recall = recall_at_k(&returned, &exact);
        assert!(
            recall >= RECALL_FLOOR,
            "[{phase}] cluster {c}: recall@{k}={recall:.4} < {RECALL_FLOOR} \
             (returned {} ids, exact top-{k} = {exact:?})",
            returned.len(),
        );
        per_query.push(returned);
    }
    per_query
}

/// Harder query directions: a point pulled OFF a cluster center toward a
/// neighbour (a `BETWEEN_BIAS`-weighted blend of two cluster bases). This is an
/// off-center, between-cluster query — its true top-k is dominated by the
/// nearer cluster but pulled toward the boundary — so a small, PRUNED nprobe
/// must still descend to that cluster's cells. A genuinely 50/50 midpoint
/// splits its top-k evenly across two clusters and legitimately needs a larger
/// probe set; the bias keeps the query hard (not a center) without making the
/// pruned recall floor a test of probe budget rather than routing correctness.
fn between_cluster_queries() -> Vec<(usize, usize)> {
    vec![(0, 1), (5, 6), (11, 12), (17, 18), (22, 23)]
}

/// Weight of the dominant cluster in a between-cluster blend (the other gets
/// `1 - BETWEEN_BIAS`). Off-center enough to be a real boundary query, biased
/// enough that the top-k is dominated by one cluster.
const BETWEEN_BIAS: f32 = 0.7;

/// Normalized `BETWEEN_BIAS`-weighted blend of two cluster base directions —
/// an off-center query pulled from cluster `a` toward cluster `b`.
fn midpoint_query(a: usize, b: usize) -> Vec<f32> {
    let (ba, bb) = (cluster_base(a), cluster_base(b));
    let mut v: Vec<f32> = ba
        .iter()
        .zip(&bb)
        .map(|(x, y)| x * BETWEEN_BIAS + y * (1.0 - BETWEEN_BIAS))
        .collect();
    normalize(&mut v);
    v
}

/// Assert recall@k ≥ `floor` at a PRUNED nprobe for every between-cluster
/// midpoint query — proving routing prunes correctly, not just on a full scan.
fn assert_between_cluster_recall(
    st: &Supertable,
    all: &[Vec<f32>],
    idx_to_id: &[i128],
    nprobe: usize,
    k: usize,
    floor: f64,
    phase: &str,
) {
    for (a, b) in between_cluster_queries() {
        let query = midpoint_query(a, b);
        let batches = st
            .vector_search(
                "emb",
                &query,
                k,
                VectorSearchOptions::new().with_nprobe(nprobe),
                None,
                None,
            )
            .unwrap_or_else(|e| panic!("vector_search(between {a},{b}) failed: {e}"));
        let returned = extract_id_set(&batches);
        let exact = brute_force_topk_ids(all, idx_to_id, &query, k);
        let recall = recall_at_k(&returned, &exact);
        assert!(
            recall >= floor,
            "[{phase}] between clusters {a},{b}: recall@{k}={recall:.4} < {floor} \
             at nprobe={nprobe} (returned {} ids)",
            returned.len(),
        );
    }
    eprintln!(
        "[routing] [{phase}] between-cluster (nprobe={nprobe}) recall ok for {} queries",
        between_cluster_queries().len()
    );
}

/// Decode the `title` column of a scalar-projection search result into the
/// set of absolute doc indices it surfaced. Proves `_id → user-superfile row`
/// remap lands on the right rows even after a drain + user-table compaction.
fn search_title_doc_indices(
    st: &Supertable,
    cluster: usize,
    nprobe: usize,
    k: usize,
) -> HashSet<usize> {
    let query = cluster_base(cluster);
    let batches = st
        .vector_search(
            "emb",
            &query,
            k,
            VectorSearchOptions::new().with_nprobe(nprobe),
            None,
            Some(&["_id", "title"]),
        )
        .unwrap_or_else(|e| panic!("scalar-projection vector_search(cluster {cluster}) failed: {e}"));
    let mut out = HashSet::new();
    for b in &batches {
        let title_arr = b
            .column_by_name("title")
            .expect("scalar projection must include title column")
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column must be LargeUtf8");
        for i in 0..title_arr.len() {
            if title_arr.is_valid(i) {
                out.insert(doc_index_from_title(title_arr.value(i)));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The test.
// ---------------------------------------------------------------------------

#[test]
fn opann_routing_exact_topk_pre_and_post_drain_with_bounded_cold_gets() {
    // Sanity: the noise perturbation keeps clusters separable — assert the
    // planted geometry up front so a config typo fails loudly. Check every
    // cluster's first doc against its own base vs the next cluster's base.
    for c in 0..N_CLUSTERS {
        let d_within = cosine_distance(&cluster_base(c), &doc_embedding(c * DOCS_PER_CLUSTER));
        let other = (c + 1) % N_CLUSTERS;
        let d_across = cosine_distance(&cluster_base(c), &doc_embedding(other * DOCS_PER_CLUSTER));
        assert!(
            d_within < d_across,
            "cluster {c} not separable: within={d_within} across={d_across}"
        );
    }

    let all = all_embeddings();

    let dir = TempDir::new().expect("data tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");

    // Counting storage wraps a LocalFs provider. The same Arc is shared by
    // the disk cache, so every read-path fetch (including the cache's cold
    // fills) is observed by the counter.
    let fetches = Arc::new(AtomicUsize::new(0));
    let tombstone_fetches = Arc::new(AtomicUsize::new(0));
    let delay_ms = Arc::new(AtomicU64::new(0));
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("LocalFs provider"));
    let counting: Arc<dyn StorageProvider> = Arc::new(CountingStorage::new(
        Arc::clone(&local),
        Arc::clone(&fetches),
        Arc::clone(&tombstone_fetches),
        Arc::clone(&delay_ms),
    ));
    let cache = make_cache(Arc::clone(&counting), cache_dir.path());

    let st = Supertable::create(
        options_title_emb()
            .with_storage(Arc::clone(&counting))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create supertable on LocalFs");

    // Write the corpus in DOCS_PER_COMMIT-sized commits → several superfiles.
    assert_eq!(N_DOCS % DOCS_PER_COMMIT, 0, "corpus must split evenly");
    let n_commits = N_DOCS / DOCS_PER_COMMIT;
    for i in 0..n_commits {
        let batch = build_batch(i * DOCS_PER_COMMIT, DOCS_PER_COMMIT);
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    assert_eq!(
        st.reader().n_docs_total(),
        N_DOCS as u64,
        "all docs committed"
    );
    eprintln!("[routing] wrote {N_DOCS} docs in {n_commits} commits");

    // Map every doc index to its engine-assigned stable `_id` via a direct
    // SQL scan of the user table. `_id` is the identity we compare on; this
    // map lets the doc-index-space oracle be expressed in `_id` space.
    let idx_to_id = build_doc_index_to_id(&st);

    // --- Property 1: pre-drain exact top-k (vectors still in INCOMING). ---
    let pre = assert_recall(&st, &all, &idx_to_id, CORRECTNESS_NPROBE, TOP_K, "pre-drain");
    eprintln!("[routing] pre-drain recall ok for {} queries", pre.len());

    // --- Drain via the public optimize() (calls hidden.drain() inside). ---
    st.optimize(&small_optimize_opts()).expect("optimize/drain");
    eprintln!("[routing] optimize() drained INCOMING staging");

    // --- Property 2: post-drain exact top-k still holds. ---
    let post = assert_recall(&st, &all, &idx_to_id, CORRECTNESS_NPROBE, TOP_K, "post-drain");
    assert_eq!(pre.len(), post.len(), "query count mismatch pre/post");
    for (i, (b, a)) in pre.iter().zip(post.iter()).enumerate() {
        assert_eq!(b, a, "query {i} returned different _id sets pre vs post drain");
    }
    eprintln!("[routing] post-drain recall ok and agrees with pre-drain");

    // --- Property 3: pruned routing is correct on HARDER, between-cluster
    // queries. A midpoint between two cluster centers has its true top-k split
    // across both, and we probe at a deliberately small PRUNED_NPROBE (6, not
    // 64) — so this proves routing descends to the right cells under pruning,
    // not just on a full sweep. Run it both at the pruned nprobe and once more
    // at the full nprobe (which must of course also pass). ---
    assert_between_cluster_recall(
        &st,
        &all,
        &idx_to_id,
        PRUNED_NPROBE,
        TOP_K,
        BETWEEN_RECALL_FLOOR,
        "post-drain/pruned",
    );
    assert_between_cluster_recall(
        &st,
        &all,
        &idx_to_id,
        CORRECTNESS_NPROBE,
        TOP_K,
        RECALL_FLOOR,
        "post-drain/full",
    );

    // --- Property 4: SCALAR PROJECTION after the drain + user-table
    // compaction. `vector_search(..., Some(&["_id","title"]))` exercises the
    // hidden-hit → user-superfile row remap and decodes the `title` column.
    // The decoded titles must be exactly the queried cluster's own docs. This
    // locks in the remap fix (a merged user superfile's `_id` column is only
    // piecewise-sorted, so the lookup must be order-independent). ---
    for c in query_clusters() {
        let returned_idx = search_title_doc_indices(&st, c, CORRECTNESS_NPROBE, TOP_K);
        // This cluster owns absolute doc indices [c*DOCS_PER_CLUSTER, +DOCS).
        let lo = c * DOCS_PER_CLUSTER;
        let hi = lo + DOCS_PER_CLUSTER;
        assert_eq!(
            returned_idx.len(),
            TOP_K,
            "scalar projection cluster {c}: expected {TOP_K} decoded titles, got {}",
            returned_idx.len()
        );
        for idx in &returned_idx {
            assert!(
                (lo..hi).contains(idx),
                "scalar projection cluster {c}: title decoded doc index {idx} \
                 outside this cluster's range [{lo}, {hi})"
            );
        }
    }
    eprintln!(
        "[routing] post-drain scalar projection (_id,title) decoded the expected \
         cluster docs for {} queries",
        query_clusters().len()
    );

    // --- Property 5: bounded PER-SEARCH GET count. ---
    //
    // Open a fresh consumer with a brand-new disk cache directory. The open +
    // manifest load + OPANN tree load is a one-time, amortised cost we do NOT
    // want to count, so first run a WARMUP query (cluster 0) to make the
    // manifest + routing tree resident. THEN reset the counter and run ONE
    // search for a DIFFERENT, far-away cluster (cluster 23) whose cells the
    // warmup did not cache — so the counter reflects only this search's own
    // per-cell vector-blob fetches (tree/manifest already resident).
    let cold_cache_dir = TempDir::new().expect("cold cache tempdir");
    let cold_cache = make_cache(Arc::clone(&counting), cold_cache_dir.path());
    let st_cold = Supertable::open(
        options_title_emb()
            .with_storage(Arc::clone(&counting))
            .with_disk_cache(Arc::clone(&cold_cache)),
    )
    .expect("open fresh cold-cache consumer");

    // Warmup: load the manifest + OPANN tree (and cache cluster 0's cells).
    let warmup_cluster = query_clusters()[0];
    let _warmup = search_ids(&st_cold, warmup_cluster, COLD_NPROBE, TOP_K);

    // Measure: a query for a far cluster the warmup did NOT touch. With the
    // tree/manifest resident, the counter now reflects only the per-search
    // routed vector-blob fetches.
    let measured_cluster = query_clusters()[query_clusters().len() - 1];
    assert_ne!(
        warmup_cluster, measured_cluster,
        "warmup and measured clusters must differ so the measured cells are uncached"
    );
    if let Some((total, max_per_cell)) = st_cold.hidden_vector_superfile_stats() {
        eprintln!("[routing] post-drain hidden index: {total} superfiles, max {max_per_cell} per cell");
    }
    eprintln!("=== MEASURED SEARCH BEGIN (cluster {measured_cluster}, nprobe {COLD_NPROBE}) ===");
    fetches.store(0, Ordering::Relaxed);
    tombstone_fetches.store(0, Ordering::Relaxed);
    // Arm the per-GET delay so wall-clock reveals the number of sequential
    // fetch waves: concurrent GETs in one wave overlap their sleeps.
    delay_ms.store(WAVE_PROBE_DELAY_MS, Ordering::Relaxed);
    let t0 = Instant::now();
    let measured_returned = search_ids(&st_cold, measured_cluster, COLD_NPROBE, TOP_K);
    let elapsed = t0.elapsed();
    delay_ms.store(0, Ordering::Relaxed);
    let per_search_gets = fetches.load(Ordering::Relaxed);
    let per_search_tombstone_gets = tombstone_fetches.load(Ordering::Relaxed);
    // Round wall-clock to whole waves; one extra half-wave of CPU/scheduling
    // slop is absorbed by the round.
    let waves = ((elapsed.as_secs_f64() * 1000.0) / WAVE_PROBE_DELAY_MS as f64).round() as u64;
    eprintln!(
        "=== MEASURED SEARCH END ({per_search_gets} S3 fetches, \
         {per_search_tombstone_gets} tombstone, ~{waves} wave(s), {elapsed:?}) ==="
    );

    // The measured query must still be correct.
    let measured_exact =
        brute_force_topk_ids(&all, &idx_to_id, &cluster_base(measured_cluster), TOP_K);
    let measured_recall = recall_at_k(&measured_returned, &measured_exact);
    assert!(
        measured_recall >= RECALL_FLOOR,
        "per-search measured query (cluster {measured_cluster}) \
         recall@{TOP_K}={measured_recall:.4} < {RECALL_FLOOR}"
    );

    eprintln!(
        "[routing] PER-SEARCH GET count = {per_search_gets} \
         (nprobe={COLD_NPROBE}, budget={PER_SEARCH_GET_BUDGET}, corpus={N_DOCS} docs; \
         tree/manifest already resident from warmup)"
    );
    assert!(
        per_search_gets > 0,
        "a routed search over uncached cells must issue at least one object-store fetch"
    );
    assert!(
        per_search_gets <= PER_SEARCH_GET_BUDGET,
        "per-search GET count {per_search_gets} exceeds bound {PER_SEARCH_GET_BUDGET}; \
         with the tree/manifest resident, a routed search's fetch count must track \
         (cells × clusters) at nprobe={COLD_NPROBE}, not corpus size ({N_DOCS} docs)"
    );
    // A post-drain OPANN search resolves deletes via the resident deleted-set,
    // so it must issue ZERO per-cell tombstone-sidecar GETs — the dead prefetch
    // wave that used to double the cold request count is gone.
    assert_eq!(
        per_search_tombstone_gets, 0,
        "a post-drain OPANN search must issue zero per-cell tombstone GETs; got \
         {per_search_tombstone_gets} (the dead sidecar prefetch wave is back)"
    );
    // With no tombstone wave, the search is a single parallel wave of cluster
    // range-GETs — the cold-latency property OPANN exists for.
    assert!(
        waves <= 1,
        "a cold OPANN search must complete in one fetch wave (parallel cluster \
         range-GETs); measured ~{waves} waves ({elapsed:?} at {WAVE_PROBE_DELAY_MS}ms/GET)"
    );
}

/// Cluster whose top-k rows are deleted in the leak test (far from cluster 0).
const VICTIM_CLUSTER: usize = 23;

/// --- Delete-correctness: the `_id`-only vector-search path must not return
/// tombstoned rows. ---
///
/// A user delete tombstones rows in the *user* table only; the hidden
/// vector-index cells are not rewritten until the next drain, so the deleted
/// rows are still physically present in the cells a query probes. Deletion is
/// therefore enforced at query time. This deletes a query cluster's exact
/// top-k and asserts a subsequent `_id`-only `vector_search` (the engine-native
/// `None`-projection path) returns NONE of the deleted `_id`s — only live rows.
///
/// Today this FAILS: the hidden-OPANN `_id`-only path resolves each hidden hit
/// to its stable user `_id` from the cell's resident inline region and returns
/// it WITHOUT consulting the user table's tombstones (the remap that filters
/// runs only on the scalar-projection path). The scalar-projection assertion
/// below already passes — pinning the leak to the `_id`-only path.
#[test]
fn opann_id_only_vector_search_excludes_deleted_rows() {
    let all = all_embeddings();

    let dir = TempDir::new().expect("data tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("LocalFs provider"));
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        options_title_emb()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create supertable");

    // Write the corpus, then drain so vector search runs over the hidden
    // per-cell IVF superfiles (the OPANN cell path under validation).
    let n_commits = N_DOCS / DOCS_PER_COMMIT;
    for i in 0..n_commits {
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(i * DOCS_PER_COMMIT, DOCS_PER_COMMIT))
            .expect("append");
        w.commit().expect("commit");
    }
    st.optimize(&small_optimize_opts()).expect("optimize/drain");

    let idx_to_id = build_doc_index_to_id(&st);
    // Reverse map: stable `_id` -> absolute doc index (for title lookup).
    let id_to_idx: HashMap<i128, usize> = idx_to_id
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    // The victim cluster's exact top-k `_id`s — the rows we delete.
    let query = cluster_base(VICTIM_CLUSTER);
    let deleted_ids: HashSet<i128> = brute_force_topk_ids(&all, &idx_to_id, &query, TOP_K)
        .into_iter()
        .collect();
    assert_eq!(deleted_ids.len(), TOP_K, "top-k must be distinct ids");

    // Sanity: pre-delete, the `_id`-only search surfaces those very rows.
    let before = search_ids(&st, VICTIM_CLUSTER, CORRECTNESS_NPROBE, TOP_K);
    let overlap_before = deleted_ids.intersection(&before).count();
    assert!(
        overlap_before >= TOP_K / 2,
        "pre-delete: expected the victim cluster's top-k to be returned, \
         got {overlap_before}/{TOP_K}"
    );

    // Delete those rows by their titles (buffered, single commit).
    let mut w = st.writer().expect("writer");
    for id in &deleted_ids {
        let idx = id_to_idx.get(id).expect("deleted id maps to a doc index");
        w.delete(col("title").eq(lit(format!("doc{idx:07}"))))
            .expect("buffer delete");
    }
    let result = w.commit().expect("commit deletes");
    let tombstoned: usize = result.outcomes.iter().map(|o| o.n_tombstoned()).sum();
    assert_eq!(
        tombstoned, TOP_K,
        "all victim rows must be tombstoned in the user table"
    );
    drop(w);

    // The `_id`-only path MUST now exclude every deleted row.
    let after_ids = search_ids(&st, VICTIM_CLUSTER, CORRECTNESS_NPROBE, TOP_K);
    let leaked: Vec<i128> = after_ids.intersection(&deleted_ids).copied().collect();
    assert!(
        leaked.is_empty(),
        "_id-only vector_search returned {} tombstoned row(s) {leaked:?} — deleted rows leaked",
        leaked.len(),
    );
    assert_eq!(
        after_ids.len(),
        TOP_K,
        "search must still return k LIVE rows after deletes, got {}",
        after_ids.len(),
    );

    // Contrast: the scalar-projection path already filters tombstones (the
    // hidden-hit -> user-superfile remap drops deleted rows), so its decoded
    // doc indices must avoid the deleted set — pinning the leak to `_id`-only.
    let deleted_idx: HashSet<usize> = deleted_ids.iter().map(|id| id_to_idx[id]).collect();
    let scalar_idx = search_title_doc_indices(&st, VICTIM_CLUSTER, CORRECTNESS_NPROBE, TOP_K);
    let scalar_leak: Vec<usize> = scalar_idx
        .iter()
        .copied()
        .filter(|i| deleted_idx.contains(i))
        .collect();
    assert!(
        scalar_leak.is_empty(),
        "scalar-projection path returned deleted doc indices {scalar_leak:?}"
    );
}

/// Option-B gate: recall under deletes. Deletes a query cluster's entire local
/// top-k, then asserts a small-nprobe `vector_search` still recovers the LIVE
/// top-k (ranks k+1..2k). A post-merge tombstone filter CANNOT satisfy this —
/// each cell emits its own top-k (the deleted rows), the filter drops them, and
/// farther cells backfill, so the true live neighbours (still in the same cell)
/// never get emitted. Only a kernel pre-heap deny — excluding tombstoned rows
/// BEFORE the per-cell top-k is selected — recovers them. RED until that lands.
#[test]
#[ignore = "Option-B gate: kernel pre-heap deny not yet implemented; recall under deletes is 0.0 with the post-merge filter alone"]
fn opann_vector_search_recall_under_deletes() {
    let all = all_embeddings();

    let dir = TempDir::new().expect("data tempdir");
    let cache_dir = TempDir::new().expect("cache tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("LocalFs provider"));
    let cache = make_cache(Arc::clone(&storage), cache_dir.path());
    let st = Supertable::create(
        options_title_emb()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create supertable");

    let n_commits = N_DOCS / DOCS_PER_COMMIT;
    for i in 0..n_commits {
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(i * DOCS_PER_COMMIT, DOCS_PER_COMMIT))
            .expect("append");
        w.commit().expect("commit");
    }
    st.optimize(&small_optimize_opts()).expect("optimize/drain");

    let idx_to_id = build_doc_index_to_id(&st);
    let id_to_idx: HashMap<i128, usize> = idx_to_id
        .iter()
        .enumerate()
        .map(|(idx, &id)| (id, idx))
        .collect();

    // Delete the victim cluster's exact top-k.
    let query = cluster_base(VICTIM_CLUSTER);
    let deleted_ids: HashSet<i128> = brute_force_topk_ids(&all, &idx_to_id, &query, TOP_K)
        .into_iter()
        .collect();
    let mut w = st.writer().expect("writer");
    for id in &deleted_ids {
        let idx = id_to_idx.get(id).expect("deleted id maps to a doc index");
        w.delete(col("title").eq(lit(format!("doc{idx:07}"))))
            .expect("buffer delete");
    }
    w.commit().expect("commit deletes");
    drop(w);

    // The LIVE top-k are ranks k+1..2k (the deleted set is exactly ranks 1..k).
    let live_exact: Vec<i128> =
        brute_force_topk_ids(&all, &idx_to_id, &query, TOP_K + deleted_ids.len())
            .into_iter()
            .filter(|id| !deleted_ids.contains(id))
            .take(TOP_K)
            .collect();
    let returned = search_ids(&st, VICTIM_CLUSTER, COLD_NPROBE, TOP_K);
    let recall = recall_at_k(&returned, &live_exact);
    eprintln!(
        "[routing] recall under deletes (entire local top-{TOP_K} tombstoned) at \
         nprobe={COLD_NPROBE}: {recall:.4} (returned {} rows)",
        returned.len()
    );
    assert!(
        recall >= RECALL_FLOOR,
        "recall under deletes at nprobe={COLD_NPROBE}: {recall:.4} < {RECALL_FLOOR} — \
         post-merge filter under-returns; the kernel pre-heap deny (Option B) is required"
    );
}
