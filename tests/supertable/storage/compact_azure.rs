// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Compaction integration test on Azure Blob Storage.
//!
//! Creates 20 small superfiles, runs BM25, vector, and SQL queries to
//! establish a result baseline, compacts the 20 files into exactly 2 via
//! two sequential jobs (10 files each), re-runs the queries, and verifies
//! the results are identical.  A fresh `Supertable::open` on the same
//! storage is opened last to confirm the new-reader path also matches.
//!
//! Two backends are tested:
//!
//! - **Azurite** (`INFINO_TEST_AZURE=1`) — local emulator, isolated container
//!   per run, cleaned up on success.
//! - **Real Azure** (`INFINO_TEST_REAL_AZURE=1`) — real account, prefixed
//!   subtree inside a pre-existing container, cleaned up on success.
//!
//! ## Two-job guarantee
//!
//! The compaction selection algorithm (`select()`) does first-fit
//! bin-packing sorted by file size asc.  For exactly 10 files to land in
//! each of the two bins, the target size must satisfy:
//!
//! ```text
//! sum(10 smallest) ≤ target_bytes < sum(11 smallest)
//! ```
//!
//! After the 20 superfiles are written, their actual sizes are read from
//! the manifest and the target is computed dynamically (ceiling of
//! `sum(10 smallest)` in whole MiB).  For the split to be clean, each
//! superfile must be at least ~1 MiB, which is the case with
//! `DOCS_PER_COMMIT = 2_000` on the Zipfian FTS corpus.
//!
//! ## Running against Azurite
//!
//! ```text
//! docker run -d --rm -p 10000:10000 \
//!   mcr.microsoft.com/azure-storage/azurite azurite-blob --blobHost 0.0.0.0
//!
//! INFINO_TEST_AZURE=1 cargo test --test supertable --features test-helpers \
//!   storage::compact_azure
//! ```
//!
//! ## Running against real Azure
//!
//! ```text
//! export INFINO_TEST_REAL_AZURE=1
//! export AZURE_STORAGE_ACCOUNT_NAME=<account>
//! export AZURE_STORAGE_ACCOUNT_KEY=<key>
//! export AZURE_STORAGE_CONTAINER_NAME=<container>   # must already exist
//! export INFINO_TEST_REAL_AZURE_PREFIX=infino-ci     # optional; default "infino-real-azure-compact"
//!
//! cargo test --test supertable --features test-helpers \
//!   storage::compact_azure::compact_real_azure_two_jobs_results_preserved
//! ```

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::VectorSearchOptions;
use infino::config::{
    CompactionSettings, Config, StorageBackend, StorageColdFetchMode, StorageSettings,
    SupertableSettings, ThreadCount,
};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::supertable::Supertable;
use infino::supertable::SupertableOptions;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{AzureStorageProvider, StorageProvider};
use infino::test_helpers::default_tokenizer;
use infino_bench_utils::corpus::generate_text_corpus;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use tempfile::TempDir;

use super::azure_helpers::{delete_emulator_container, ensure_emulator_container};

/// Docs committed in each of the 20 write cycles.  At 2 000 docs the
/// Zipfian FTS corpus produces superfiles of roughly 1–2 MiB, which is
/// large enough for the MiB-granularity target calculation to guarantee
/// a clean 10+10 split.
const DOCS_PER_COMMIT: usize = 2_000;
/// Number of commits (= number of superfiles before compaction).
const N_COMMITS: usize = 20;
/// Deterministic RNG seed for `generate_text_corpus`.
const CORPUS_SEED: u64 = 42;
/// BM25 top-k for verification queries.
const BM25_K: usize = 10;
/// 1 MiB in bytes — matches the granularity of `CompactionSettings::target_superfile_size_mb`.
const MIB: u64 = 1024 * 1024;
/// Expected number of superfiles after compaction (one per job).
const N_COMPACTED_FILES: usize = 2;
/// Number of source files each compaction job merges.
const FILES_PER_JOB: usize = N_COMMITS / N_COMPACTED_FILES;
/// Embedding dimension — small for test speed.
const EMB_DIM: usize = 16;
/// Number of IVF centroids in the vector index.
const N_CENT: usize = 8;
/// Random rotation seed for the vector index.
const VECTOR_ROT_SEED: u64 = 99;
/// Seed used to generate per-doc unit vectors.
const VECTOR_CORPUS_SEED: u64 = 7777;
/// Vector search top-k — intentionally 1.
///
/// A query vector equal to a doc's exact embedding has Cosine distance 0 to
/// that doc, so it always ranks #1 in its cluster regardless of the IVF
/// structure built before or after compaction.  Using k=1 makes every probe
/// a "singleton" in the same sense as the BM25 doc-ID queries: the returned
/// `_id` is stable no matter how superfiles are merged.
///
/// Using k>1 is unsafe here: the 1-bit RaBitQ shortlist selects
/// `k × rerank_mult` candidates per cluster before Fp32 reranking.  With
/// 250 docs/cluster (pre-compact) vs 2 500 docs/cluster (post-compact), the
/// shortlist cutoff changes and the exact top-5 set can shift even with
/// exhaustive nprobe — causing spurious test failures unrelated to any
/// correctness regression.
const VECTOR_K: usize = 1;
/// nprobe = N_CENT → all IVF clusters are probed, guaranteeing the queried
/// doc (distance 0 from its own embedding) is always found.
const VECTOR_NPROBE: usize = N_CENT;
/// Absolute doc indices whose exact embeddings serve as vector query probes.
const VECTOR_PROBE_DOCS: &[usize] = &[0, 5_000, 15_000, 25_000, 35_000];
/// Singleton doc-ID tokens used as SQL FTS TVF queries.
const SQL_FTS_QUERIES: &[&str] = &["doc0001234", "doc0019999", "doc0039999"];
/// Absolute doc indices whose embeddings serve as SQL vector TVF query probes.
const SQL_VECTOR_PROBE_DOCS: &[usize] = &[7_500, 32_500];

/// BM25 verification queries.
///
/// Each `"doc{:07}"` token is injected by `generate_text_corpus` as a
/// per-doc unique identifier, so it appears in exactly one document.
/// Singleton queries are fully deterministic across compaction: the
/// returned `_id` set is always a single element regardless of how
/// superfiles are merged or how global IDF statistics shift.  Using
/// only singletons avoids spurious failures from BM25 tie-breaking
/// changes that can flip the last-place entry in a top-k ranking when
/// the IDF of a high-frequency term shifts slightly after a merge.
const QUERIES: &[&str] = &[
    "doc0001234", // singleton in first 10 K docs
    "doc0009999", // singleton near the end of the first commit batch
    "doc0019999", // singleton mid-corpus
    "doc0029999", // singleton in third quarter
    "doc0039999", // singleton near the end of the corpus
];

/// Build a `DiskCacheStore` backed by `storage` with cache files rooted at
/// `cache_root`.  Used by both the Azurite and real-Azure tests to avoid
/// re-fetching superfile bytes on every query.
fn make_cache(
    storage: Arc<dyn StorageProvider>,
    cache_root: &std::path::Path,
) -> Arc<DiskCacheStore> {
    let cfg = DiskCacheConfig {
        cache_root: cache_root.to_path_buf(),
        disk_budget_bytes: 1 << 30,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: 4,
        cold_fetch_chunk_bytes: 1 << 20,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("DiskCacheStore::new")
}

/// `Config` for the real-Azure compaction tests.  Drives `apply_config` so
/// the storage provider, disk cache, and thread pools are all wired from one
/// place, matching the production `connect(uri)` path.
fn azure_compact_config(container: &str, prefix: &str, cache_root: &std::path::Path) -> Config {
    Config {
        supertable: SupertableSettings {
            writer_threads: ThreadCount::Fixed(1),
            ..SupertableSettings::default()
        },
        storage: StorageSettings {
            backend: StorageBackend::Azure,
            bucket: Some(container.to_string()),
            prefix: prefix.to_string(),
            disk_cache_root: Some(cache_root.to_path_buf()),
            disk_budget_bytes: 1 << 30,
            cold_fetch_mode: StorageColdFetchMode::LazyForegroundWithBackgroundFill,
            cold_fetch_streams: 8,
            cold_fetch_chunk_bytes: 8 << 20,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            ..StorageSettings::default()
        },
        compaction: CompactionSettings::default(),
    }
}

/// Returns the `DataType` for a fixed-size list of `f32` with `dim` elements.
fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// `SupertableOptions` for the combined title + embedding schema used in
/// these tests. Configures FTS on `title` and a vector index on `emb`,
/// with a single-thread rayon writer pool for deterministic runs.
fn options_title_emb() -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("rayon ThreadPoolBuilder with num_threads(1) builds"),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(EMB_DIM), false),
    ]));
    SupertableOptions::new(
        schema,
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
    .expect("SupertableOptions::new with title+emb test fixture args")
    .with_writer_pool(pool)
}

/// Generate a deterministic unit vector for `doc_idx`.
///
/// Draws `EMB_DIM` samples from a standard-normal distribution seeded
/// with `VECTOR_CORPUS_SEED ^ doc_idx`, casts them to `f32`, then
/// L2-normalizes the result so Cosine similarity is well-defined.
fn doc_embedding(doc_idx: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(VECTOR_CORPUS_SEED ^ doc_idx as u64);
    let dist = StandardNormal;
    let mut v: Vec<f32> = (0..EMB_DIM)
        .map(|_| {
            let s: f64 = dist.sample(&mut rng);
            s as f32
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Build a two-column `RecordBatch` (title + emb) for the given titles.
///
/// `doc_offset` is the absolute index of the first title so that
/// `doc_embedding(doc_offset + i)` is used for each row, keeping
/// embeddings deterministic per absolute document position.
fn build_batch(titles: &[&str], doc_offset: usize) -> RecordBatch {
    let title_arr = LargeStringArray::from(titles.to_vec());
    let flat: Vec<f32> = (0..titles.len())
        .flat_map(|i| doc_embedding(doc_offset + i))
        .collect();
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(flat);
    let fsl = FixedSizeListArray::try_new(
        item_field,
        EMB_DIM as i32,
        Arc::new(values) as ArrayRef,
        None,
    )
    .expect("FixedSizeListArray for emb column");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(EMB_DIM), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)])
        .expect("RecordBatch with title and emb columns")
}

/// Format a float slice as a comma-separated string for SQL TVF queries.
fn vec_to_csv(v: &[f32]) -> String {
    v.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

/// Run every query in `QUERIES` against a pinned snapshot of `st`.
/// Returns one sorted `Vec<i128>` of `_id` values per query.
fn run_bm25_queries(st: &Supertable) -> Vec<Vec<i128>> {
    let reader = st.reader();
    QUERIES
        .iter()
        .map(|q| {
            let batches = reader
                .bm25_search("title", q, BM25_K, BoolMode::Or, None)
                .unwrap_or_else(|e| panic!("bm25_search({q:?}) failed: {e}"));
            extract_sorted_ids(&batches)
        })
        .collect()
}

/// Run one `vector_search` call per entry in `VECTOR_PROBE_DOCS`.
/// Returns sorted `_id` sets per query.
fn run_vector_queries(st: &Supertable) -> Vec<Vec<i128>> {
    let reader = st.reader();
    VECTOR_PROBE_DOCS
        .iter()
        .map(|&idx| {
            let query = doc_embedding(idx);
            let batches = reader
                .vector_search(
                    "emb",
                    &query,
                    VECTOR_K,
                    VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                    None,
                    None,
                )
                .unwrap_or_else(|e| panic!("vector_search(doc {idx}) failed: {e}"));
            extract_sorted_ids(&batches)
        })
        .collect()
}

/// Run SQL FTS and SQL vector TVF queries against a pinned snapshot.
/// Returns sorted `_id` sets for each query (FTS queries first, then
/// vector queries), in the order of `SQL_FTS_QUERIES` followed by
/// `SQL_VECTOR_PROBE_DOCS`.
fn run_sql_queries(st: &Supertable) -> Vec<Vec<i128>> {
    let reader = st.reader();
    let mut results = Vec::new();
    for q in SQL_FTS_QUERIES {
        let sql = format!("SELECT _id FROM bm25_search('title', '{q}', 1)");
        let batches = reader
            .query_sql(&sql)
            .unwrap_or_else(|e| panic!("SQL bm25_search({q:?}) failed: {e}"));
        results.push(extract_sorted_ids(&batches));
    }
    for &idx in SQL_VECTOR_PROBE_DOCS {
        let csv = vec_to_csv(&doc_embedding(idx));
        let sql = format!("SELECT _id FROM vector_search('emb', '{csv}', 1)");
        let batches = reader
            .query_sql(&sql)
            .unwrap_or_else(|e| panic!("SQL vector_search(doc {idx}) failed: {e}"));
        results.push(extract_sorted_ids(&batches));
    }
    results
}

/// Assert that `before` and `after` contain the same `_id` sets for every
/// query, with a context label for diagnostic messages.
fn assert_query_results_match(before: &[Vec<i128>], after: &[Vec<i128>], context: &str) {
    assert_eq!(
        before.len(),
        after.len(),
        "[{context}] result-set count mismatch"
    );
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        assert_eq!(b, a, "[{context}] query {i} returned different _id sets");
    }
}

/// Extract and sort the `_id` column values from a search result.
fn extract_sorted_ids(batches: &[RecordBatch]) -> Vec<i128> {
    let mut ids = Vec::new();
    for b in batches {
        let col = b
            .column_by_name("_id")
            .expect("search result must have _id column");
        let arr = col
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id column must be Decimal128");
        for i in 0..arr.len() {
            if arr.is_valid(i) {
                ids.push(arr.value(i));
            }
        }
    }
    ids.sort_unstable();
    ids
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compact_azure_two_jobs_results_preserved() {
    if std::env::var("INFINO_TEST_AZURE").is_err() {
        eprintln!(
            "compact_azure_two_jobs_results_preserved: skipped \
             (set INFINO_TEST_AZURE=1 to enable)"
        );
        return;
    }

    // Fresh container per run: put_atomic is create-only and the pointer
    // lives at the container root, so a reused container from a prior run
    // would collide on the second run.
    let container = format!("infino-compact-{}", uuid::Uuid::new_v4());
    ensure_emulator_container(&container).await;
    eprintln!("[compact_azure] container {container} ready");

    // Generate the full corpus up front; it is sliced into DOCS_PER_COMMIT
    // chunks and committed one at a time to produce N_COMMITS superfiles.
    let corpus = generate_text_corpus(N_COMMITS * DOCS_PER_COMMIT, CORPUS_SEED);

    let cache_dir = TempDir::new().expect("Azurite cache tempdir");
    let writer_storage: Arc<dyn StorageProvider> = Arc::new(
        AzureStorageProvider::new_with_emulator(&container).expect("azure provider for writer"),
    );
    let cache = make_cache(Arc::clone(&writer_storage), cache_dir.path());
    let st = Supertable::create(
        options_title_emb()
            .with_storage(Arc::clone(&writer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("create supertable on Azure");

    // Write N_COMMITS superfiles, each with DOCS_PER_COMMIT documents.
    for i in 0..N_COMMITS {
        let chunk = &corpus[i * DOCS_PER_COMMIT..(i + 1) * DOCS_PER_COMMIT];
        let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let batch = build_batch(&refs, i * DOCS_PER_COMMIT);
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }

    let reader_pre = st.reader();
    assert_eq!(
        reader_pre.n_superfiles(),
        N_COMMITS,
        "expected {N_COMMITS} superfiles before compaction"
    );
    assert_eq!(
        reader_pre.n_docs_total(),
        (N_COMMITS * DOCS_PER_COMMIT) as u64,
        "pre-compact doc count mismatch"
    );
    eprintln!(
        "[compact_azure] {N_COMMITS} superfiles written, {} docs total",
        reader_pre.n_docs_total()
    );

    // Capture query results before any compaction.
    let pre_bm25 = run_bm25_queries(&st);
    let pre_vector = run_vector_queries(&st);
    let pre_sql = run_sql_queries(&st);
    eprintln!("[compact_azure] pre-compact query results captured");

    // Derive target_superfile_size_mb so that exactly FILES_PER_JOB files
    // land in each compaction job.
    //
    // The first-fit packing loop adds files (sorted size asc) until the
    // next file would overflow `target_bytes`.  For 10 identical-sized
    // files S to fit but not 11:
    //
    //   10·S ≤ target_bytes < 11·S
    //
    // Setting target_bytes = ⌈ sum(10 smallest) / MiB ⌉ × MiB guarantees
    // the left inequality.  The assertion below verifies the right one —
    // it fails fast when superfiles are unexpectedly small and would pack
    // more than FILES_PER_JOB into a single bin.
    let mut sizes: Vec<u64> = reader_pre
        .manifest()
        .superfiles
        .iter()
        .map(|e| {
            e.subsection_offsets
                .as_ref()
                .map(|o| o.total_size)
                .unwrap_or(0)
        })
        .collect();
    sizes.sort_unstable();
    let first_n_sum: u64 = sizes.iter().take(FILES_PER_JOB).sum();
    let next_file_size: u64 = sizes[FILES_PER_JOB];
    let target_mib = first_n_sum.div_ceil(MIB);
    let target_bytes = target_mib * MIB;
    assert!(
        target_bytes < first_n_sum + next_file_size,
        "superfile sizes are too uniform for a clean {FILES_PER_JOB}+{FILES_PER_JOB} split: \
         first_{FILES_PER_JOB}_sum={first_n_sum} B, \
         next_file={next_file_size} B, \
         target_bytes={target_bytes} B — \
         increase DOCS_PER_COMMIT so each superfile exceeds 1 MiB"
    );
    eprintln!(
        "[compact_azure] target_mib={target_mib} \
         (first_{FILES_PER_JOB}_sum={first_n_sum} B, next_file={next_file_size} B)"
    );

    // Two compaction jobs run sequentially inside compact():
    //   job 0 — merges files 0..FILES_PER_JOB into one compacted superfile
    //   job 1 — merges files FILES_PER_JOB..N_COMMITS into one compacted superfile
    // Both jobs operate on the single default partition (n_buckets=1).
    let cfg = CompactionSettings {
        target_superfile_size_mb: target_mib,
        min_fill_percent: 1,
        ..CompactionSettings::default()
    };
    st.compact(&cfg).expect("compact");
    eprintln!("[compact_azure] compact() done");

    let reader_post = st.reader();
    assert_eq!(
        reader_post.n_superfiles(),
        N_COMPACTED_FILES,
        "expected {N_COMPACTED_FILES} superfiles after compaction (one per job)"
    );
    assert_eq!(
        reader_post.n_docs_total(),
        (N_COMMITS * DOCS_PER_COMMIT) as u64,
        "doc count must be preserved across compaction"
    );
    eprintln!(
        "[compact_azure] post-compact: {} superfiles, {} docs",
        reader_post.n_superfiles(),
        reader_post.n_docs_total()
    );

    // Same-handle post-compact queries must return the same document IDs.
    let post_bm25 = run_bm25_queries(&st);
    let post_vector = run_vector_queries(&st);
    let post_sql = run_sql_queries(&st);
    assert_query_results_match(&pre_bm25, &post_bm25, "same-handle post-compact BM25");
    assert_query_results_match(&pre_vector, &post_vector, "same-handle post-compact vector");
    assert_query_results_match(&pre_sql, &post_sql, "same-handle post-compact SQL");
    eprintln!("[compact_azure] same-handle post-compact queries match");

    // Open a completely fresh Supertable on the same Azure container and
    // verify it also sees the compacted state with identical query results.
    // Reuse the same cache directory — the fresh reader can read any
    // superfile pages the writer's queries have already populated.
    let reader_storage: Arc<dyn StorageProvider> = Arc::new(
        AzureStorageProvider::new_with_emulator(&container)
            .expect("azure provider for fresh reader"),
    );
    let reader_cache = make_cache(Arc::clone(&reader_storage), cache_dir.path());
    let st2 = Supertable::open(
        options_title_emb()
            .with_storage(Arc::clone(&reader_storage))
            .with_disk_cache(reader_cache),
    )
    .expect("open fresh supertable from Azure");
    let fresh_bm25 = run_bm25_queries(&st2);
    let fresh_vector = run_vector_queries(&st2);
    let fresh_sql = run_sql_queries(&st2);
    assert_query_results_match(&pre_bm25, &fresh_bm25, "fresh-open reader BM25");
    assert_query_results_match(&pre_vector, &fresh_vector, "fresh-open reader vector");
    assert_query_results_match(&pre_sql, &fresh_sql, "fresh-open reader SQL");
    eprintln!("[compact_azure] fresh-open reader queries match — test passed");

    delete_emulator_container(&container).await;
    eprintln!("[compact_azure] container {container} deleted");
}

/// Compact two jobs on a real Azure Blob Storage account.
///
/// ## Required environment variables
///
/// | Variable | Purpose |
/// |---|---|
/// | `INFINO_TEST_REAL_AZURE=1` | enables the test (skipped otherwise) |
/// | `AZURE_STORAGE_ACCOUNT_NAME` | storage account name |
/// | `AZURE_STORAGE_ACCOUNT_KEY` | 64-byte Base64 storage account key |
/// | `AZURE_STORAGE_CONTAINER_NAME` | container that **must already exist** |
///
/// ## Optional
///
/// | Variable | Default | Purpose |
/// |---|---|---|
/// | `INFINO_TEST_REAL_AZURE_PREFIX` | `infino-real-azure-compact` | root prefix for test isolation |
///
/// A UUID sub-prefix is appended to the root so concurrent runs don't collide.
/// All blobs under the prefix are deleted on success; the container itself is
/// never touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn compact_real_azure_two_jobs_results_preserved() {
    if std::env::var("INFINO_TEST_REAL_AZURE").ok().as_deref() != Some("1") {
        eprintln!(
            "compact_real_azure_two_jobs_results_preserved: skipped \
             (set INFINO_TEST_REAL_AZURE=1 and AZURE_STORAGE_CONTAINER_NAME to enable)"
        );
        return;
    }

    let container = match std::env::var("AZURE_STORAGE_CONTAINER_NAME") {
        Ok(c) => c,
        Err(_) => {
            eprintln!(
                "compact_real_azure_two_jobs_results_preserved: skipped \
                 (missing AZURE_STORAGE_CONTAINER_NAME)"
            );
            return;
        }
    };
    let prefix_root = std::env::var("INFINO_TEST_REAL_AZURE_PREFIX")
        .unwrap_or_else(|_| "infino-real-azure-compact".to_string());
    let prefix = format!("{}/{}", prefix_root.trim_matches('/'), uuid::Uuid::new_v4());

    eprintln!("[real-azure-compact] container={container} prefix={prefix}");

    let cache_dir = TempDir::new().expect("real Azure cache tempdir");
    let cfg = azure_compact_config(&container, &prefix, cache_dir.path());

    // All work happens inside this async block so that errors still reach
    // the cleanup code below rather than panicking before cleanup runs.
    let result: Result<(), String> = async {
        let corpus = generate_text_corpus(N_COMMITS * DOCS_PER_COMMIT, CORPUS_SEED);

        let st = Supertable::create(
            options_title_emb()
                .apply_config(&cfg)
                .map_err(|e| format!("apply Azure config: {e}"))?,
        )
        .map_err(|e| format!("create supertable: {e}"))?;

        for i in 0..N_COMMITS {
            let chunk = &corpus[i * DOCS_PER_COMMIT..(i + 1) * DOCS_PER_COMMIT];
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let batch = build_batch(&refs, i * DOCS_PER_COMMIT);
            let mut w = st.writer().map_err(|e| format!("writer: {e}"))?;
            w.append(&batch).map_err(|e| format!("append: {e}"))?;
            w.commit().map_err(|e| format!("commit: {e}"))?;
        }

        let reader_pre = st.reader();
        if reader_pre.n_superfiles() != N_COMMITS {
            return Err(format!(
                "expected {N_COMMITS} superfiles before compaction, got {}",
                reader_pre.n_superfiles()
            ));
        }
        if reader_pre.n_docs_total() != (N_COMMITS * DOCS_PER_COMMIT) as u64 {
            return Err(format!(
                "pre-compact doc count: expected {}, got {}",
                N_COMMITS * DOCS_PER_COMMIT,
                reader_pre.n_docs_total()
            ));
        }
        eprintln!(
            "[real-azure-compact] {N_COMMITS} superfiles, {} docs written",
            reader_pre.n_docs_total()
        );

        let pre_bm25 = run_bm25_queries(&st);
        let pre_vector = run_vector_queries(&st);
        let pre_sql = run_sql_queries(&st);
        eprintln!("[real-azure-compact] pre-compact queries captured");

        // Compute target_mib for a clean FILES_PER_JOB + FILES_PER_JOB split.
        let mut sizes: Vec<u64> = reader_pre
            .manifest()
            .superfiles
            .iter()
            .map(|e| {
                e.subsection_offsets
                    .as_ref()
                    .map(|o| o.total_size)
                    .unwrap_or(0)
            })
            .collect();
        sizes.sort_unstable();
        let first_n_sum: u64 = sizes.iter().take(FILES_PER_JOB).sum();
        let next_file_size: u64 = sizes[FILES_PER_JOB];
        let target_mib = first_n_sum.div_ceil(MIB);
        let target_bytes = target_mib * MIB;
        if target_bytes >= first_n_sum + next_file_size {
            return Err(format!(
                "superfile sizes too uniform for a clean \
                 {FILES_PER_JOB}+{FILES_PER_JOB} split: \
                 first_{FILES_PER_JOB}_sum={first_n_sum} B, \
                 next_file={next_file_size} B, target_bytes={target_bytes} B"
            ));
        }
        eprintln!(
            "[real-azure-compact] target_mib={target_mib} \
             (first_{FILES_PER_JOB}_sum={first_n_sum} B, next_file={next_file_size} B)"
        );

        let compact_cfg = CompactionSettings {
            target_superfile_size_mb: target_mib,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        st.compact(&compact_cfg)
            .map_err(|e| format!("compact: {e}"))?;
        eprintln!("[real-azure-compact] compact() done");

        let reader_post = st.reader();
        if reader_post.n_superfiles() != N_COMPACTED_FILES {
            return Err(format!(
                "expected {N_COMPACTED_FILES} superfiles after compaction, got {}",
                reader_post.n_superfiles()
            ));
        }
        if reader_post.n_docs_total() != (N_COMMITS * DOCS_PER_COMMIT) as u64 {
            return Err(format!(
                "post-compact doc count: expected {}, got {}",
                N_COMMITS * DOCS_PER_COMMIT,
                reader_post.n_docs_total()
            ));
        }
        eprintln!(
            "[real-azure-compact] post-compact: {} superfiles, {} docs",
            reader_post.n_superfiles(),
            reader_post.n_docs_total()
        );

        let post_bm25 = run_bm25_queries(&st);
        let post_vector = run_vector_queries(&st);
        let post_sql = run_sql_queries(&st);
        assert_query_results_match(&pre_bm25, &post_bm25, "same-handle post-compact BM25");
        assert_query_results_match(&pre_vector, &post_vector, "same-handle post-compact vector");
        assert_query_results_match(&pre_sql, &post_sql, "same-handle post-compact SQL");
        eprintln!("[real-azure-compact] same-handle post-compact queries match");

        // Fresh reader: same config (same prefix + shared cache dir) so it
        // benefits from any pages already cold-fetched during the write/query
        // phase.  `apply_config` creates a new AzureStorageProvider + a new
        // DiskCacheStore over the same on-disk cache root.
        let st2 = Supertable::open(
            options_title_emb()
                .apply_config(&cfg)
                .map_err(|e| format!("apply Azure config for fresh reader: {e}"))?,
        )
        .map_err(|e| format!("open fresh supertable: {e}"))?;
        let fresh_bm25 = run_bm25_queries(&st2);
        let fresh_vector = run_vector_queries(&st2);
        let fresh_sql = run_sql_queries(&st2);
        assert_query_results_match(&pre_bm25, &fresh_bm25, "fresh-open reader BM25");
        assert_query_results_match(&pre_vector, &fresh_vector, "fresh-open reader vector");
        assert_query_results_match(&pre_sql, &fresh_sql, "fresh-open reader SQL");
        eprintln!("[real-azure-compact] fresh-open reader queries match — test passed");

        Ok(())
    }
    .await;

    // Cleanup: list and delete every blob written under the prefix.  A
    // no-prefix provider is used so the full object-store paths returned by
    // list_with_prefix can be passed directly to delete().
    let cleanup_storage =
        AzureStorageProvider::new(&container).expect("real Azure cleanup provider");
    let all_keys = cleanup_storage
        .list_with_prefix(&prefix)
        .await
        .unwrap_or_default();
    eprintln!(
        "[real-azure-compact] cleanup: deleting {} keys under prefix={prefix}",
        all_keys.len()
    );
    for key in &all_keys {
        let _ = cleanup_storage.delete(key).await;
    }

    result.expect("real Azure compact test failed");
}
