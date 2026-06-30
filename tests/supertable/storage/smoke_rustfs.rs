// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable smoke through a local RustFS HTTPS daemon.
//!
//! Spawns RustFS via [`infino_bench_utils::rustfs_server`], points
//! `S3StorageProvider` at it, and runs storage + supertable commit
//! probes including conditional `If-Match` PUTs and search TVFs via
//! `query_sql`.
//!
//! ## Gating
//!
//! `INFINO_TEST_RUSTFS=1` — spawning RustFS downloads or locates a
//! ~95 MiB binary on first run, so the default `cargo test` skips it.
//!
//! ```text
//! INFINO_TEST_RUSTFS=1 cargo test --test supertable storage::smoke_rustfs
//! ```

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

const TEST_BUCKET: &str = "infino-rustfs-smoke";
/// Vector index shape for the RustFS TVF smoke fixture.
const VECTOR_N_CENT: usize = 4;
const VECTOR_ROT_SEED: u64 = 17;
const EMB_DIM: usize = 16;
const EXPECTED_N_DOCS: u64 = 8;
const BM25_TOP_K: usize = 10;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_rustfs_https() {
    if std::env::var("INFINO_TEST_RUSTFS").is_err() {
        eprintln!(
            "supertable_smoke_via_rustfs_https: skipped (set INFINO_TEST_RUSTFS=1 to enable)"
        );
        return;
    }

    let handle = tokio::task::spawn_blocking(|| rustfs_server::spawn_rustfs(TEST_BUCKET))
        .await
        .expect("spawn_blocking join")
        .expect("spawn rustfs");
    eprintln!(
        "[rustfs-smoke] spawned on {} bucket={TEST_BUCKET}",
        handle.endpoint
    );

    let storage: Arc<dyn StorageProvider> =
        rustfs_server::rustfs_s3_provider(&handle, "").expect("rustfs provider");

    // Probe round-trip before the writer path.
    let probe_bytes = Bytes::from_static(b"hello-rustfs-smoke");
    storage
        .put_atomic("probe/hello.txt", probe_bytes.clone())
        .await
        .expect("probe put_atomic");
    let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
    assert_eq!(got, probe_bytes, "probe round-trip mismatch");

    // Conditional PUT: success with matching etag, failure with stale etag.
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

    // Multi-commit supertable path (OCC pointer uses If-Match under the hood).
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

    eprintln!("[rustfs-smoke] smoke done");
}

/// Regression: [`IngestResult`](infino_bench_utils::ingest::supertable::IngestResult) must
/// retain the RustFS child through warm/cold search. Dropping the handle while the storage
/// `Arc` is still live reproduces the connection-refused failure seen before the keepalive fix.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn rustfs_keepalive_survives_fixture_drop() {
    if std::env::var("INFINO_TEST_RUSTFS").is_err() {
        eprintln!(
            "rustfs_keepalive_survives_fixture_drop: skipped (set INFINO_TEST_RUSTFS=1 to enable)"
        );
        return;
    }

    const PROBE_KEY: &str = "probe/keepalive.txt";
    let probe_bytes = Bytes::from_static(b"keepalive-probe");

    struct IngestLike {
        storage: Arc<dyn StorageProvider>,
        _keepalive: rustfs_server::RustFsHandle,
    }

    let fixture = {
        let handle = tokio::task::spawn_blocking(|| rustfs_server::spawn_rustfs(TEST_BUCKET))
            .await
            .expect("spawn_blocking join")
            .expect("spawn rustfs");
        let storage = rustfs_server::rustfs_s3_provider(&handle, "").expect("rustfs provider");
        storage
            .put_atomic(PROBE_KEY, probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        IngestLike {
            storage,
            _keepalive: handle,
        }
    };

    let storage = Arc::clone(&fixture.storage);
    let (got, _) = storage
        .get(PROBE_KEY)
        .await
        .expect("get with keepalive held");
    assert_eq!(
        got, probe_bytes,
        "storage must stay reachable while handle lives"
    );

    drop(fixture);
    assert!(
        storage.get(PROBE_KEY).await.is_err(),
        "dropping the RustFS handle must tear down the daemon"
    );
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
        }],
        vec![VectorConfig {
            column: "emb".into(),
            dim,
            n_cent: VECTOR_N_CENT,
            rot_seed: VECTOR_ROT_SEED,
            metric: infino::superfile::vector::distance::Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8ResidualEpsilon,
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

/// Search TVFs (`bm25_search`, `vector_search`, `hybrid_search`) through
/// `query_sql` against a RustFS-backed supertable with disk cache cold-fetch.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_tvfs_through_query_sql_via_rustfs() {
    if std::env::var("INFINO_TEST_RUSTFS").is_err() {
        eprintln!(
            "supertable_tvfs_through_query_sql_via_rustfs: skipped \
             (set INFINO_TEST_RUSTFS=1 to enable)"
        );
        return;
    }

    const TVF_BUCKET: &str = "infino-rustfs-smoke-tvf";
    let _handle = tokio::task::spawn_blocking(|| rustfs_server::spawn_rustfs(TVF_BUCKET))
        .await
        .expect("spawn_blocking join")
        .expect("spawn rustfs for TVF smoke");
    let dim = EMB_DIM;
    assert!(dim > 0, "embedding dimension must be positive");
    eprintln!(
        "[rustfs-smoke-tvf] spawned on {} bucket={TVF_BUCKET}",
        _handle.endpoint
    );

    let storage: Arc<dyn StorageProvider> =
        rustfs_server::rustfs_s3_provider(&_handle, "").expect("rustfs TVF provider");

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

    let mut q = vec![0.0f32; dim];
    q[0] = 1.0;
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
