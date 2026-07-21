// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through a local RustFS HTTPS daemon.
//!
//! Uses the lazy shared [`rustfs_server::session`] via [`rustfs_server::open_test_fixture`].
//! The daemon starts on first S3 use; tests do not create or tear down the session.
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
    superfile::builder::{FtsConfig, VectorConfig},
    supertable::{
        Supertable,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use infino_bench_utils::rustfs_server;
use tempfile::TempDir;

/// Vector index shape for the RustFS TVF smoke fixture.
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
const EMB_DIM: usize = 16;
const EXPECTED_N_DOCS: u64 = 8;
const BM25_TOP_K: usize = 10;

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
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("first commit via RustFS");
        w.append(&build_title_batch(&["echo foxtrot"]))
            .expect("second append");
        w.commit().expect("second commit via RustFS (If-Match OCC)");
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
