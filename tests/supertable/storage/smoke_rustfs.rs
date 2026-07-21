// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through a local RustFS HTTPS daemon.
//!
//! Uses the lazy shared [`rustfs_server::session`] via [`rustfs_server::open_test_fixture`].
//! The daemon starts on first S3 use; tests do not create or tear down the session.
//!
//! Also covers the connection-budget OverBudget e2es: cold vector / SQL / hybrid
//! refusal and the shared multi-superfile budget, previously run against s3s-fs.
//!
//! ## Gating
//!
//! Runs by default. Set `INFINO_TEST_DISABLE_RUSTFS=1` to skip on offline hosts or
//! platforms without auto-download (`INFINO_RUSTFS_BIN` overrides).

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    InfinoError, VectorSearchOptions,
    config::{Config, MemorySettings, StorageBackend, StorageSettings},
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{
        Supertable,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options, lazy_foreground_disk_cache},
};
use infino_bench_utils::rustfs_server;
use tempfile::TempDir;

/// Vector index shape for the RustFS TVF smoke fixture.
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
const EMB_DIM: usize = 16;
const EXPECTED_N_DOCS: u64 = 8;
const BM25_TOP_K: usize = 10;
/// Single-thread writer pool for budget e2es: one commit → one superfile.
/// A multi-thread pool shards `BUDGET_N_ROWS` across many small superfiles;
/// open-range prefetch then swallows each into a resident reader and the
/// cold-fetch budget gate never fires.
const BUDGET_WRITER_POOL_THREADS: usize = 1;
/// Vector-search top-k and nprobe for the over-budget e2es.
const VECTOR_SEARCH_K: usize = 3;
const VECTOR_NPROBE: usize = 4;
/// Connection memory budget for the over-budget e2e: 1 byte. The 90% gate
/// floors to 0, so the first cold cluster-block fetch is refused.
const TINY_BUDGET_BYTES: u64 = 1;
/// Row count for the over-budget e2e fixture. Must be large enough that IVF
/// cluster blocks are a genuine cold object-store fetch under the SQL TVF's
/// default (fine-first) probe shape — not swallowed by the lazy reader's
/// open-range / parquet-tail overlay. At 4K–16K rows the default path stays
/// warm (peak 0); 64K pushes the probed codes outside that overlay.
const BUDGET_N_ROWS: usize = 65_536;
/// Expected peak reservation band for a measured cold vector search over
/// `BUDGET_N_ROWS` (dim 16, `n_cent` 4, Sq8, default or `nprobe` 4). Assert a
/// band around the observed ~156 KB fetch: tight enough to prove it's the
/// real cluster fetch, loose enough to survive minor codec / layout drift.
const CONTROL_PEAK_LOW_BYTES: usize = 120_000;
const CONTROL_PEAK_HIGH_BYTES: usize = 200_000;
/// Bounded budget set generously above one cold fetch (~156 KB); 90% gate is
/// 900 KB. Proves an enforcing budget admits under-budget work.
const AMPLE_BUDGET_BYTES: u64 = 1_000_000;
const AMPLE_BUDGET_GATE_BYTES: usize = 900_000;
/// Shared multi-superfile budget: admits one ~156 KB fetch but not two
/// concurrent ones (90% gate = 180 KB).
const SHARED_BUDGET_BYTES: u64 = 200_000;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_rustfs_https() {
    if !rustfs_server::begin_rustfs_test("supertable_smoke_via_rustfs_https") {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open test fixture");
    let storage = Arc::clone(&fixture.storage);

    let probe_bytes = Bytes::from_static(b"hello-rustfs-smoke");
    storage
        .put_atomic("probe/hello.txt", probe_bytes.clone())
        .await
        .expect("probe put_atomic");
    let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
    assert_eq!(got, probe_bytes, "probe round-trip mismatch");

    storage
        .put_atomic("probe/cas.txt", Bytes::from_static(b"v1"))
        .await
        .expect("seed cas object");
    let (_, meta) = storage.get("probe/cas.txt").await.expect("read cas object");
    let etag = meta.etag.expect("etag after put_atomic");
    storage
        .put_if_match("probe/cas.txt", Bytes::from_static(b"v2"), Some(&etag))
        .await
        .expect("put_if_match with current etag");
    let stale = etag;
    let err = storage
        .put_if_match("probe/cas.txt", Bytes::from_static(b"v3"), Some(&stale))
        .await
        .expect_err("stale etag must fail");
    assert!(
        matches!(err, StorageError::PreconditionFailed { .. }),
        "expected PreconditionFailed, got {err:?}"
    );

    {
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        {
            let mut w = producer.writer().expect("writer");
            w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
                .expect("append");
            w.commit().expect("first commit via RustFS");
        }
        {
            let mut w = producer.writer().expect("writer for second commit");
            w.append(&build_title_batch(&["echo foxtrot"]))
                .expect("second append");
            w.commit().expect("second commit via RustFS (If-Match OCC)");
        }
        assert_eq!(producer.manifest_id(), 2);
    }

    let consumer = Supertable::open(default_supertable_options().with_storage(storage))
        .expect("open from RustFS");
    assert_eq!(consumer.manifest_id(), 2);
    assert_eq!(consumer.reader().n_docs_total(), 3);

    eprintln!("[rustfs-smoke] smoke done bucket={}", fixture.bucket);
}

/// Bucket lease with cleanup (same path as `tiers.rs` / `cargo bench`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rustfs_session_unique_bucket_lease_matches_bench_lifecycle() {
    if !rustfs_server::begin_rustfs_test(
        "rustfs_session_unique_bucket_lease_matches_bench_lifecycle",
    ) {
        return;
    }

    const PROBE_KEY: &str = "probe/session-lease.txt";
    let probe_bytes = Bytes::from_static(b"session-lease-probe");

    let bucket_name = {
        let lease = tokio::task::spawn_blocking(|| {
            rustfs_server::session().and_then(|session| session.open_unique_bucket(""))
        })
        .await
        .expect("spawn_blocking join")
        .expect("open_unique_bucket on shared session");

        eprintln!("[rustfs-session-smoke] leased bucket={}", lease.bucket);

        let bucket_name = lease.bucket.clone();
        lease
            .storage
            .put_atomic(PROBE_KEY, probe_bytes.clone())
            .await
            .expect("probe put_atomic via session lease");
        let (got, _) = lease
            .storage
            .get(PROBE_KEY)
            .await
            .expect("probe get via session lease");
        assert_eq!(
            got, probe_bytes,
            "session lease storage round-trip mismatch"
        );

        rustfs_server::release_lease(lease).await;
        bucket_name
    };

    let second_bucket = {
        let lease = tokio::task::spawn_blocking(|| {
            rustfs_server::session().and_then(|session| session.open_unique_bucket(""))
        })
        .await
        .expect("spawn_blocking join")
        .expect("second open_unique_bucket after first lease dropped");
        assert_ne!(
            lease.bucket, bucket_name,
            "each open_unique_bucket call must allocate a fresh bucket name"
        );
        lease
            .storage
            .put_atomic("probe/second-lease.txt", Bytes::from_static(b"ok"))
            .await
            .expect("second lease must reach the shared session daemon");
        let name = lease.bucket.clone();
        rustfs_server::release_lease(lease).await;
        name
    };
    let _ = second_bucket;

    let recreated = tokio::task::spawn_blocking(move || {
        rustfs_server::session().and_then(|session| session.open_bucket(&bucket_name, "", true))
    })
    .await
    .expect("spawn_blocking join")
    .expect("recreate bucket after lease cleanup");
    let err = recreated
        .storage
        .get(PROBE_KEY)
        .await
        .expect_err("cleaned-up bucket must not retain the probe object");
    assert!(
        matches!(err, StorageError::NotFound { .. }),
        "expected NotFound after lease cleanup; got {err:?}"
    );
    rustfs_server::release_lease(recreated).await;

    eprintln!("[rustfs-session-smoke] session lease + cleanup OK");
}

/// The shared session daemon must outlive an individual test fixture drop.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rustfs_session_survives_test_fixture_drop() {
    if !rustfs_server::begin_rustfs_test("rustfs_session_survives_test_fixture_drop") {
        return;
    }

    const PROBE_KEY: &str = "probe/keepalive.txt";
    let probe_bytes = Bytes::from_static(b"keepalive-probe");

    let storage = {
        let fixture = rustfs_server::open_test_fixture_async("")
            .await
            .expect("open test fixture");
        fixture
            .storage
            .put_atomic(PROBE_KEY, probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        Arc::clone(&fixture.storage)
    };

    let (got, _) = storage
        .get(PROBE_KEY)
        .await
        .expect("session daemon must stay up after fixture drop");
    assert_eq!(got, probe_bytes);
}

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
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
}

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

fn rustfs_vector_options(dim: usize) -> infino::supertable::SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    infino::supertable::SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: VECTOR_N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: infino::superfile::vector::distance::Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
            provided_centroids: None,
        }],
        Some(infino::test_helpers::default_tokenizer()),
    )
    .expect("rustfs TVF test options")
}

/// Options for budget e2es: same schema as [`rustfs_vector_options`], but a
/// single-thread writer pool so each commit lands as one large superfile.
fn rustfs_budget_options(dim: usize) -> infino::supertable::SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(BUDGET_WRITER_POOL_THREADS)
            .build()
            .expect("single-thread budget writer pool"),
    );
    rustfs_vector_options(dim).with_writer_pool(pool)
}

fn rustfs_vector_batch(dim: usize) -> RecordBatch {
    let titles = LargeStringArray::from(vec![
        "alpha vector one",
        "alpha vector two",
        "bravo vector three",
        "charlie vector four",
        "delta vector five",
        "echo vector six",
        "foxtrot vector seven",
        "golf vector eight",
    ]);
    let mut flat = Vec::with_capacity(titles.len() * dim);
    for row in 0..titles.len() {
        for d in 0..dim {
            flat.push(if d == row % dim { 1.0 } else { 0.0 });
        }
    }
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(flat);
    let vectors = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
        .expect("vectors");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(vectors)]).expect("batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_tvfs_through_query_sql_via_rustfs() {
    if !rustfs_server::begin_rustfs_test("supertable_tvfs_through_query_sql_via_rustfs") {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open test fixture for TVF smoke");
    let dim = EMB_DIM;
    assert!(dim > 0, "embedding dimension must be positive");
    eprintln!("[rustfs-smoke-tvf] bucket={}", fixture.bucket);

    let storage = Arc::clone(&fixture.storage);

    {
        let producer =
            Supertable::create(rustfs_vector_options(dim).with_storage(Arc::clone(&storage)))
                .expect("create tvf producer");
        let mut w = producer.writer().expect("tvf producer writer");
        w.append(&rustfs_vector_batch(dim))
            .expect("append unified vector+FTS batch");
        w.commit().expect("tvf producer commit via RustFS");
        assert_eq!(producer.manifest_id(), 1);
    }

    let consumer_storage = Arc::clone(&storage);
    let cache_dir = TempDir::new().expect("tvf cache tempdir");
    let cache = make_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        rustfs_vector_options(dim)
            .with_storage(consumer_storage)
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via RustFS (tvf consumer)");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), EXPECTED_N_DOCS);

    let pre = cache.stats();

    let q: Vec<f32> = (0..dim)
        .map(|i| if i == 0 { 1.0f32 } else { 0.0f32 })
        .collect();
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");

    fn count_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    let bm25 = consumer
        .reader()
        .query_sql(&format!(
            "SELECT _id FROM bm25_search('title', 'alpha', {BM25_TOP_K})"
        ))
        .expect("bm25_search via query_sql over RustFS");
    assert!(
        count_rows(&bm25) >= 2,
        "bm25_search('alpha') should return >=2 docs over RustFS; got {}",
        count_rows(&bm25)
    );

    let vec_sql = format!("SELECT _id FROM vector_search('emb', '{q_csv}', 3)");
    let vector = consumer
        .reader()
        .query_sql(&vec_sql)
        .expect("vector_search via query_sql over RustFS");
    assert!(
        count_rows(&vector) >= 1,
        "vector_search returned no rows over RustFS"
    );

    let hybrid_sql =
        format!("SELECT _id FROM hybrid_search('title', 'alpha', 'emb', '{q_csv}', 5)");
    let hybrid = consumer
        .reader()
        .query_sql(&hybrid_sql)
        .expect("hybrid_search via query_sql over RustFS");
    let hyb_rows = count_rows(&hybrid);
    assert!(
        hyb_rows > 0 && hyb_rows <= 5,
        "hybrid_search rows in (0, 5]; got {hyb_rows}"
    );

    let post = cache.stats();
    assert!(
        post.n_cold_fetches > pre.n_cold_fetches,
        "TVF queries must cold-fetch through RustFS; pre={} post={}",
        pre.n_cold_fetches,
        post.n_cold_fetches
    );

    eprintln!(
        "[rustfs-smoke-tvf] bm25 / vector / hybrid via query_sql OK; \
         n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );
}

/// A `Config` that carries only a connection memory budget; storage backend is
/// `None` so `apply_config` leaves storage / disk cache unattached (the caller
/// wires those explicitly afterward).
fn budget_only_config(connection_budget_bytes: u64) -> Config {
    Config {
        storage: StorageSettings {
            backend: StorageBackend::None,
            ..StorageSettings::default()
        },
        memory: MemorySettings {
            connection_budget_bytes,
        },
        ..Config::default()
    }
}

/// Open a fresh consumer against `storage` with a lazy-foreground disk cache
/// (so vector reads stay cold / non-resident) and `connection_budget_bytes` as
/// the connection budget (`0` = measured). Returns the handle plus the cache's
/// `TempDir` guard.
fn open_budget_consumer(
    dim: usize,
    storage: &Arc<dyn StorageProvider>,
    connection_budget_bytes: u64,
) -> (Supertable, TempDir) {
    let cache_dir = TempDir::new().expect("budget consumer cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(storage), cache_dir.path());
    let consumer = Supertable::open(
        rustfs_budget_options(dim)
            .apply_config(&budget_only_config(connection_budget_bytes))
            .expect("apply budget config to consumer options")
            .with_storage(Arc::clone(storage))
            .with_disk_cache(cache),
    )
    .expect("Supertable::open via RustFS (budget consumer)");
    (consumer, cache_dir)
}

/// Larger vector+FTS batch for the over-budget e2e: `BUDGET_N_ROWS` rows so
/// IVF cluster blocks carry real bytes. One-hot embeddings at `row % dim`;
/// titles carry the row index (and the word `budget` for hybrid BM25).
fn budget_vector_batch(dim: usize, n_rows: usize) -> RecordBatch {
    let titles = LargeStringArray::from(
        (0..n_rows)
            .map(|i| format!("budget vector row {i}"))
            .collect::<Vec<_>>(),
    );
    let mut flat = Vec::with_capacity(n_rows * dim);
    for row in 0..n_rows {
        for d in 0..dim {
            flat.push(if d == row % dim { 1.0 } else { 0.0 });
        }
    }
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let values = Float32Array::from(flat);
    let vectors = FixedSizeListArray::try_new(item_field, dim as i32, Arc::new(values), None)
        .expect("fixed-size vector array");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(vectors)]).expect("batch")
}

/// Cold vector search under a tiny per-connection budget is refused with
/// `InfinoError::OverBudget`. A measured control and an ample bounded budget
/// then run the identical cold query to completion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_vector_search_over_budget_via_rustfs() {
    if !rustfs_server::begin_rustfs_test("supertable_cold_vector_search_over_budget_via_rustfs") {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open budget fixture");
    let dim = EMB_DIM;
    let storage = Arc::clone(&fixture.storage);
    eprintln!("[rustfs-budget] bucket={}", fixture.bucket);

    {
        let producer =
            Supertable::create(rustfs_budget_options(dim).with_storage(Arc::clone(&storage)))
                .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via RustFS");
        assert_eq!(producer.manifest_id(), 1);
    }

    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), BUDGET_N_ROWS as u64);

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let result = consumer.vector_search(
        "emb",
        &q,
        VECTOR_SEARCH_K,
        VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
        None,
        None,
    );

    match result {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[rustfs-budget] cold vector search refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(hits) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold vector search returned {} batch(es)",
            hits.len()
        ),
    }

    let bounded_budget = consumer.options().connection_budget();
    eprintln!(
        "[rustfs-budget] bounded budget: denials={} peak={} B",
        bounded_budget.denials(),
        bounded_budget.peak()
    );
    assert!(
        bounded_budget.denials() >= 1,
        "bounded budget must record >=1 denial; got {}",
        bounded_budget.denials()
    );
    assert_eq!(
        bounded_budget.peak(),
        0,
        "a refused cold fetch commits nothing, so peak must stay 0"
    );

    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_hits = control
        .vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("measured cold vector search should run to completion");
    let control_rows: usize = control_hits.iter().map(|b| b.num_rows()).sum();
    assert!(
        control_rows >= 1,
        "measured cold vector search returned no rows over RustFS"
    );
    let control_budget = control.options().connection_budget();
    eprintln!(
        "[rustfs-budget] measured control: rows={control_rows} denials={} peak={} B",
        control_budget.denials(),
        control_budget.peak()
    );
    assert_eq!(
        control_budget.denials(),
        0,
        "measured budget must never deny"
    );
    let control_peak = control_budget.peak();
    assert!(
        (CONTROL_PEAK_LOW_BYTES..=CONTROL_PEAK_HIGH_BYTES).contains(&control_peak),
        "measured cold vector search peak {control_peak} B outside expected \
         [{CONTROL_PEAK_LOW_BYTES}, {CONTROL_PEAK_HIGH_BYTES}] band; \
         a peak near 0 means the budget was never exercised on the query path"
    );

    let (ample, _ample_guard) = open_budget_consumer(dim, &storage, AMPLE_BUDGET_BYTES);
    let ample_budget = ample.options().connection_budget();
    assert_eq!(
        ample_budget.limit(),
        Some(AMPLE_BUDGET_GATE_BYTES),
        "ample budget must be bounded (an enforced gate), not measured"
    );
    let ample_hits = ample
        .vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("under-budget cold vector search should run under a bounded budget");
    let ample_rows: usize = ample_hits.iter().map(|b| b.num_rows()).sum();
    let ample_peak = ample_budget.peak();
    eprintln!(
        "[rustfs-budget] bounded-ample: rows={ample_rows} denials={} peak={ample_peak} B",
        ample_budget.denials()
    );
    assert!(
        ample_rows >= 1,
        "bounded-ample cold vector search returned no rows"
    );
    assert_eq!(
        ample_budget.denials(),
        0,
        "an under-budget query must not be denied by a bounded budget"
    );
    assert!(
        (CONTROL_PEAK_LOW_BYTES..=CONTROL_PEAK_HIGH_BYTES).contains(&ample_peak),
        "bounded-ample peak {ample_peak} B outside expected \
         [{CONTROL_PEAK_LOW_BYTES}, {CONTROL_PEAK_HIGH_BYTES}] band"
    );
}

/// Same cold-fetch budget refusal through the `vector_search` SQL table
/// function. Pins that `OverBudget` survives the DataFusion error boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_vector_search_over_budget_via_sql_rustfs() {
    if !rustfs_server::begin_rustfs_test("supertable_cold_vector_search_over_budget_via_sql_rustfs")
    {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open budget SQL fixture");
    let dim = EMB_DIM;
    let storage = Arc::clone(&fixture.storage);

    {
        let producer =
            Supertable::create(rustfs_budget_options(dim).with_storage(Arc::clone(&storage)))
                .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via RustFS");
    }

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("SELECT _id FROM vector_search('emb', '{q_csv}', {VECTOR_SEARCH_K})");

    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    let result = consumer.reader().query_sql(&sql).map_err(InfinoError::from);
    match result {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[rustfs-budget-sql] cold vector search in SQL refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(batches) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold vector search in SQL returned {} batch(es)",
            batches.len()
        ),
    }
    let bounded_budget = consumer.options().connection_budget();
    assert!(
        bounded_budget.denials() >= 1,
        "bounded budget must record >=1 denial; got {}",
        bounded_budget.denials()
    );

    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_rows: usize = control
        .reader()
        .query_sql(&sql)
        .expect("measured cold vector search in SQL should run to completion")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert!(
        control_rows >= 1,
        "measured cold vector search in SQL returned no rows over RustFS"
    );
    assert_eq!(
        control.options().connection_budget().denials(),
        0,
        "measured budget must never deny"
    );
}

/// Same cold-fetch budget refusal through `hybrid_search` (BM25 + vector RRF).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_cold_hybrid_search_over_budget_via_sql_rustfs() {
    if !rustfs_server::begin_rustfs_test("supertable_cold_hybrid_search_over_budget_via_sql_rustfs")
    {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open hybrid budget fixture");
    let dim = EMB_DIM;
    let storage = Arc::clone(&fixture.storage);

    {
        let producer =
            Supertable::create(rustfs_budget_options(dim).with_storage(Arc::clone(&storage)))
                .expect("create budget producer");
        let mut w = producer.writer().expect("budget producer writer");
        w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
            .expect("append large vector+FTS batch");
        w.commit().expect("budget producer commit via RustFS");
    }

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let q_csv = q
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!(
        "SELECT _id FROM hybrid_search('title', 'budget', 'emb', '{q_csv}', {VECTOR_SEARCH_K})"
    );

    let (consumer, _cache_guard) = open_budget_consumer(dim, &storage, TINY_BUDGET_BYTES);
    match consumer.reader().query_sql(&sql).map_err(InfinoError::from) {
        Err(InfinoError::OverBudget(msg)) => {
            eprintln!("[rustfs-budget-sql] cold hybrid search refused as OverBudget: {msg}");
        }
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(batches) => panic!(
            "expected InfinoError::OverBudget under a {TINY_BUDGET_BYTES}-byte budget; \
             cold hybrid search returned {} batch(es)",
            batches.len()
        ),
    }
    assert!(
        consumer.options().connection_budget().denials() >= 1,
        "bounded budget must record >=1 denial"
    );

    let (control, _control_cache_guard) = open_budget_consumer(dim, &storage, 0);
    let control_rows: usize = control
        .reader()
        .query_sql(&sql)
        .expect("measured cold hybrid search should run to completion")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert!(
        control_rows >= 1,
        "measured cold hybrid search returned no rows over RustFS"
    );
    assert_eq!(
        control.options().connection_budget().denials(),
        0,
        "measured budget must never deny"
    );
}

/// The per-connection budget is shared across the multi-superfile fan-out.
/// Two commits → two ~156 KB cold fetches; a budget that fits one but not two
/// refuses the concurrent fan-out as `OverBudget`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_vector_budget_is_shared_across_superfiles_via_rustfs() {
    if !rustfs_server::begin_rustfs_test(
        "supertable_vector_budget_is_shared_across_superfiles_via_rustfs",
    ) {
        return;
    }

    let fixture = rustfs_server::open_test_fixture_async("")
        .await
        .expect("open shared-budget fixture");
    let dim = EMB_DIM;
    let storage = Arc::clone(&fixture.storage);
    eprintln!("[rustfs-shared] bucket={}", fixture.bucket);

    {
        let producer =
            Supertable::create(rustfs_budget_options(dim).with_storage(Arc::clone(&storage)))
                .expect("create shared-budget producer");
        for commit in 0..2 {
            let mut w = producer.writer().expect("shared-budget producer writer");
            w.append(&budget_vector_batch(dim, BUDGET_N_ROWS))
                .expect("append large vector+FTS batch");
            w.commit()
                .expect("shared-budget producer commit via RustFS");
            assert_eq!(producer.manifest_id(), commit + 1);
        }
    }

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
    let search = |table: &Supertable| {
        table.vector_search(
            "emb",
            &q,
            VECTOR_SEARCH_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
    };

    let (measured, _measured_guard) = open_budget_consumer(dim, &storage, 0);
    assert_eq!(measured.reader().n_superfiles(), 2);
    assert_eq!(measured.reader().n_docs_total(), (BUDGET_N_ROWS as u64) * 2);
    let measured_hits = search(&measured).expect("measured search over two superfiles runs");
    let measured_rows: usize = measured_hits.iter().map(|b| b.num_rows()).sum();
    let measured_peak = measured.options().connection_budget().peak();
    eprintln!("[rustfs-shared] measured: rows={measured_rows} peak={measured_peak} B");
    assert!(
        measured_rows >= 1,
        "measured two-superfile search returned no rows"
    );
    assert!(
        measured_peak > CONTROL_PEAK_HIGH_BYTES,
        "peak {measured_peak} B should exceed one superfile's fetch \
         ({CONTROL_PEAK_HIGH_BYTES} B): the two fetches must sum on one budget"
    );

    let (bounded, _bounded_guard) = open_budget_consumer(dim, &storage, SHARED_BUDGET_BYTES);
    let result = search(&bounded);
    let bounded_budget = bounded.options().connection_budget();
    eprintln!(
        "[rustfs-shared] bounded: denials={} peak={} B result={}",
        bounded_budget.denials(),
        bounded_budget.peak(),
        if result.is_ok() { "ok" } else { "over-budget" }
    );
    match result {
        Err(InfinoError::OverBudget(_)) => {}
        Err(other) => panic!("expected InfinoError::OverBudget, got {other:?}"),
        Ok(hits) => panic!(
            "two concurrent {CONTROL_PEAK_HIGH_BYTES}-B fetches must cross a \
             {SHARED_BUDGET_BYTES}-B budget; got {} batch(es)",
            hits.len()
        ),
    }
    assert!(
        bounded_budget.denials() >= 1,
        "the shared budget must record the crossing"
    );
}
