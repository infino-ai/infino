//! 003 M16 — supertable smoke through the S3 wire protocol.
//!
//! Stands up an in-process s3s-fs server on a random port,
//! points `S3StorageProvider` at it, and runs a small
//! commit + open + query cycle. Validates the "real cloud
//! path" end-to-end: every storage call (head / get /
//! get_range / put_atomic / put_if_match / delete) goes
//! through the full S3 HTTP wire protocol; nothing
//! short-circuits to the local filesystem.
//!
//! ## Gating
//!
//! The test is gated on `INFINO_TEST_S3=1`. Without the env
//! var, the test exits as a no-op early (printing a brief
//! "skipped" line). Reason: spawning an in-process HTTP
//! server has cost (~50 ms per test invocation) and pulls
//! in s3s + s3s-fs dev-dependencies on the test binary's
//! compile path. The default `cargo test` run skips it.
//!
//! Invocation:
//!
//! ```text
//! INFINO_TEST_S3=1 cargo test --test supertable_smoke_s3
//! ```
//!
//! ## What's verified
//!
//! - `Supertable::create + writer.commit` against the S3
//!   wire path (superfiles + manifest part + manifest list +
//!   pointer all PUT via HTTP).
//! - `Supertable::open` from a fresh handle recovers the
//!   pre-commit state (manifest_id, n_superfiles, n_docs_total).
//! - Reader query via `query_sql` routes through the
//!   `DiskCacheStore` (cold-fetch via HTTP get_range from
//!   the s3s-fs server).
//!
//! ## What's NOT verified
//!
//! - AWS-specific quirks: virtual-hosted-style requests,
//!   AWS-Sig-V4 authentication corner cases, regional
//!   endpoints. The smoke test uses path-style (forced) +
//!   a fixed dummy credential pair. Real-AWS validation
//!   requires AWS credentials + a test bucket; out of scope
//!   for an in-process smoke.
//! - Concurrent writers (M11's OCC retry is exercised
//!   end-to-end in `tests/supertable_concurrent_processes.rs`
//!   against LocalFS; the S3 path uses S3 CAS natively, no
//!   read-then-overwrite window, so behavior is identical
//!   modulo wire latency).

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::config::{
    Config, StorageBackend, StorageColdFetchMode, StorageSettings, SupertableSettings,
};
use infino::superfile::builder::{FtsConfig, VectorConfig};
use infino::supertable::Supertable;
use infino::supertable::query::VectorSearchOptions;
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{S3StorageProvider, StorageProvider};
use infino::test_helpers::{build_title_batch, default_supertable_options};
use s3s::auth::SimpleAuth;
use s3s::service::S3ServiceBuilder;
use s3s_fs::FileSystem;
use tempfile::TempDir;
use tokio::net::TcpListener;

const TEST_BUCKET: &str = "infino-m16-smoke";
const TEST_REGION: &str = "us-east-1";
const TEST_ACCESS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const TEST_SECRET_KEY: &str = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";

/// Spawn s3s-fs on a random port. Returns the bound
/// address + the tempdir guard (must stay alive for the
/// test's lifetime — drop unlinks the bucket data).
async fn spawn_s3s_fs() -> (SocketAddr, TempDir) {
    let fs_root = TempDir::new().expect("s3s-fs root tempdir");
    // s3s-fs treats top-level dirs as buckets. Pre-create
    // the bucket dir so put_atomic on a key inside it
    // doesn't 404 the bucket itself.
    std::fs::create_dir_all(fs_root.path().join(TEST_BUCKET)).expect("create bucket dir");

    let fs_backend = FileSystem::new(fs_root.path()).expect("s3s-fs FileSystem");
    // Configure auth so s3s accepts the SigV4-signed
    // requests object_store sends. Without `set_auth`, s3s
    // responds 501 "no authentication provider" to any
    // signed request.
    let service = {
        let mut b = S3ServiceBuilder::new(fs_backend);
        b.set_auth(SimpleAuth::from_single(TEST_ACCESS_KEY, TEST_SECRET_KEY));
        b.build()
    };
    // S3Service derives Clone (internally Arc<Inner>); clones
    // share the underlying service handle.

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");

    tokio::spawn(async move {
        use hyper_util::rt::{TokioExecutor, TokioIo};
        use hyper_util::server::conn::auto::Builder as ConnBuilder;
        let http = ConnBuilder::new(TokioExecutor::new());
        loop {
            let (stream, _peer) = match listener.accept().await {
                Ok(t) => t,
                Err(_) => break,
            };
            let service = service.clone();
            let http = http.clone();
            tokio::spawn(async move {
                let _ = http.serve_connection(TokioIo::new(stream), service).await;
            });
        }
    });

    (addr, fs_root)
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

fn real_s3_options(dim: usize) -> infino::supertable::SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread writer pool"),
    );
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
            n_cent: 4,
            rot_seed: 17,
            metric: infino::superfile::vector::distance::Metric::Cosine,
            rerank_codec: infino::superfile::vector::rerank_codec::RerankCodec::Sq8Residual,
        }],
        Some(infino::test_helpers::default_tokenizer()),
    )
    .expect("real S3 test options")
    .with_writer_pool(pool)
}

fn real_s3_config(bucket: &str, prefix: &str, cache_root: &std::path::Path) -> Config {
    Config {
        supertable: SupertableSettings::default(),
        storage: StorageSettings {
            backend: StorageBackend::S3,
            bucket: Some(bucket.to_string()),
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
    }
}

fn real_s3_batch(dim: usize) -> RecordBatch {
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
    let vectors = FixedSizeListArray::try_new(
        item_field,
        dim as i32,
        Arc::new(values) as Arc<dyn Array>,
        None,
    )
    .expect("fixed-size vector array");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", fixed_list_f32(dim), false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(vectors)]).expect("batch")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_smoke_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_smoke_via_s3_wire_protocol: skipped (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    eprintln!("[m16] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Quick provider-level smoke before invoking the full
    // writer path — isolates "the S3 provider works at all"
    // from "the writer + cache stack works on top".
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for probe"),
        );
        let probe_bytes = bytes::Bytes::from_static(b"hello-m16");
        storage
            .put_atomic("probe/hello.txt", probe_bytes.clone())
            .await
            .expect("probe put_atomic");
        let (got, _) = storage.get("probe/hello.txt").await.expect("probe get");
        assert_eq!(got, probe_bytes, "probe round-trip mismatch");
        eprintln!("[m16] probe round-trip OK (PUT + GET via S3 wire)");
    }

    // Producer: writes through the S3 wire protocol.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for producer"),
        );
        let producer =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let mut w = producer.writer().expect("producer writer");
        w.append(&build_title_batch(&["alpha bravo", "charlie delta"]))
            .expect("append");
        w.commit().expect("producer commit via S3");
        assert_eq!(producer.manifest_id(), 1);
        eprintln!(
            "[m16] producer commit OK; manifest_id={}",
            producer.manifest_id()
        );
    }

    // Consumer: opens via the same S3 endpoint + a disk
    // cache. Reads should route through the cache → S3
    // get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        S3StorageProvider::new_with_endpoint(
            &endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("s3 provider for consumer"),
    );
    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = make_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        default_supertable_options()
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via S3");

    assert_eq!(consumer.manifest_id(), 1, "recovered manifest_id mismatch");
    assert_eq!(
        consumer.reader().n_docs_total(),
        2,
        "recovered n_docs_total mismatch"
    );
    eprintln!(
        "[m16] consumer open OK; manifest_id={} n_superfiles={} n_docs_total={}",
        consumer.manifest_id(),
        consumer.reader().n_superfiles(),
        consumer.reader().n_docs_total()
    );

    // SQL query through cache. First query cold-fetches via
    // S3; n_cold_fetches grows.
    let pre = cache.stats();
    assert_eq!(pre.n_cold_fetches, 0);
    let batches = consumer
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("query_sql via S3");
    assert_eq!(batches.len(), 1);
    let post = cache.stats();
    assert!(
        post.n_cold_fetches >= 1,
        "first query must cold-fetch through S3; got n_cold_fetches={}",
        post.n_cold_fetches
    );
    eprintln!(
        "[m16] cold-fetch via S3 OK; n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );

    eprintln!("[m16] smoke done");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn supertable_real_s3_lazy_vector_and_fts_round_trip() {
    if std::env::var("INFINO_TEST_REAL_S3").ok().as_deref() != Some("1") {
        eprintln!(
            "supertable_real_s3_lazy_vector_and_fts_round_trip: skipped \
             (set INFINO_TEST_REAL_S3=1 and INFINO_TEST_REAL_S3_BUCKET to enable)"
        );
        return;
    }

    let bucket = match std::env::var("INFINO_TEST_REAL_S3_BUCKET")
        .or_else(|_| std::env::var("INFINO_TEST_S3_BUCKET"))
    {
        Ok(bucket) => bucket,
        Err(_) => {
            eprintln!(
                "supertable_real_s3_lazy_vector_and_fts_round_trip: skipped \
                 (missing INFINO_TEST_REAL_S3_BUCKET)"
            );
            return;
        }
    };
    let prefix_root = std::env::var("INFINO_TEST_REAL_S3_PREFIX")
        .unwrap_or_else(|_| "infino-real-s3-integration".to_string());
    let prefix = format!("{}/{}", prefix_root.trim_matches('/'), uuid::Uuid::new_v4());

    eprintln!("[real-s3] bucket={bucket} prefix={prefix}");

    let cache_dir = TempDir::new().expect("real S3 cache tempdir");
    let cfg = real_s3_config(&bucket, &prefix, cache_dir.path());
    let result = async {
        let dim = 16;
        {
            let producer = Supertable::create(
                real_s3_options(dim)
                    .apply_config(&cfg)
                    .map_err(|e| format!("apply S3 config to producer options: {e}"))?,
            )
            .map_err(|e| format!("create unified supertable on real S3: {e}"))?;
            let mut writer = producer
                .writer()
                .map_err(|e| format!("real S3 producer writer: {e}"))?;
            writer
                .append(&real_s3_batch(dim))
                .map_err(|e| format!("append unified vector+FTS batch: {e}"))?;
            writer
                .commit()
                .map_err(|e| format!("commit unified supertable to real S3: {e}"))?;
            if producer.manifest_id() != 1 {
                return Err(format!(
                    "producer manifest_id mismatch: got {}",
                    producer.manifest_id()
                ));
            }
            eprintln!(
                "[real-s3] producer commit OK; manifest_id={}",
                producer.manifest_id()
            );
        }

        let consumer = Supertable::open(
            real_s3_options(dim)
                .apply_config(&cfg)
                .map_err(|e| format!("apply S3 config to consumer options: {e}"))?,
        )
        .map_err(|e| format!("open unified supertable from real S3: {e}"))?;

        if consumer.manifest_id() != 1 {
            return Err(format!(
                "recovered manifest id mismatch: got {}",
                consumer.manifest_id()
            ));
        }
        if consumer.reader().n_docs_total() != 8 {
            return Err(format!(
                "recovered doc count mismatch: got {}",
                consumer.reader().n_docs_total()
            ));
        }

        let bm25_hits = consumer
            .bm25_search(
                "title",
                "alpha",
                10,
                infino::superfile::fts::reader::BoolMode::Or,
            )
            .map_err(|e| format!("cold BM25 over real S3: {e}"))?;
        if bm25_hits.is_empty() {
            return Err("real S3 cold BM25 did not find alpha docs".to_string());
        }

        let mut query = vec![0.0f32; dim];
        query[0] = 1.0;
        let vector_hits = consumer
            .vector_search("emb", &query, 3, VectorSearchOptions::new().with_nprobe(4))
            .map_err(|e| format!("cold vector search over real S3: {e}"))?;
        if vector_hits.is_empty() {
            return Err("real S3 cold vector search returned no hits".to_string());
        }

        let cache = consumer
            .options()
            .disk_cache
            .as_ref()
            .ok_or_else(|| "S3 config did not attach disk cache".to_string())?;
        let stats = cache.stats();
        if stats.n_cold_fetches < 1 {
            return Err(format!(
                "real S3 reads did not hydrate through lazy disk cache; stats={stats:?}"
            ));
        }
        eprintln!(
            "[real-s3] cold lazy cache OK; n_cold_fetches={} cache_bytes={}",
            stats.n_cold_fetches, stats.current_bytes
        );

        let reader = consumer.reader();
        let manifest = reader.manifest();
        let list = manifest
            .list
            .as_ref()
            .ok_or_else(|| "real S3 open did not recover persisted manifest list".to_string())?;
        let mut cleanup_keys = vec![
            "_supertable/current".to_string(),
            infino::supertable::manifest::commit::list_uri(consumer.manifest_id()),
        ];
        cleanup_keys.extend(list.parts.iter().map(|p| p.uri.clone()));
        cleanup_keys.extend(
            manifest
                .superfiles
                .iter()
                .map(|entry| entry.uri.storage_path()),
        );

        Ok::<Vec<String>, String>(cleanup_keys)
    }
    .await;
    let cleanup_storage = S3StorageProvider::new_with_prefix(&bucket, &prefix)
        .expect("real S3 cleanup provider from AWS env");
    if let Ok(keys) = &result {
        for key in keys {
            let _ = cleanup_storage.delete(key).await;
        }
    } else {
        let _ = cleanup_storage.delete("_supertable/current").await;
    }
    eprintln!("[real-s3] cleanup OK; deleted keys under prefix={prefix}");
    result.expect("real S3 integration failed");
}

/// TVF lane over the S3 wire protocol: exercises
/// `bm25_search`, `vector_search`, and `hybrid_search`
/// end-to-end through `query_sql` (DataFusion plan -> custom
/// `TableProvider` -> custom exec -> kernel -> resolve to
/// `_id`) against an S3-backed supertable. The existing
/// `supertable_smoke_via_s3_wire_protocol` covers
/// `SELECT COUNT(*)` (provider scan path); this one covers the
/// search TVFs, which is where the retrieval engine actually
/// earns its keep on object storage.
///
/// Asserts `cache.stats().n_cold_fetches` grew across the
/// three queries — proves the TVF reads went through the
/// s3s-fs server (HTTP get_range), not a local short-circuit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn supertable_tvfs_through_query_sql_via_s3_wire_protocol() {
    if std::env::var("INFINO_TEST_S3").is_err() {
        eprintln!(
            "supertable_tvfs_through_query_sql_via_s3_wire_protocol: skipped \
             (set INFINO_TEST_S3=1 to enable)"
        );
        return;
    }

    let (addr, _fs_root_guard) = spawn_s3s_fs().await;
    let endpoint = format!("http://{}", addr);
    let dim = 16;
    eprintln!("[m16-tvf] s3s-fs spawned on {endpoint} bucket={TEST_BUCKET}");

    // Producer: writes a title (FTS) + emb (vector) batch
    // through the S3 wire protocol.
    {
        let storage: Arc<dyn StorageProvider> = Arc::new(
            S3StorageProvider::new_with_endpoint(
                &endpoint,
                TEST_BUCKET,
                TEST_ACCESS_KEY,
                TEST_SECRET_KEY,
                TEST_REGION,
            )
            .expect("s3 provider for tvf producer"),
        );
        let producer = Supertable::create(real_s3_options(dim).with_storage(Arc::clone(&storage)))
            .expect("create tvf producer");
        let mut w = producer.writer().expect("tvf producer writer");
        w.append(&real_s3_batch(dim))
            .expect("append unified vector+FTS batch");
        w.commit().expect("tvf producer commit via S3");
        assert_eq!(producer.manifest_id(), 1);
    }

    // Consumer: opens via the same S3 endpoint + a disk
    // cache. TVF reads cold-fetch through HTTP get_range.
    let consumer_storage: Arc<dyn StorageProvider> = Arc::new(
        S3StorageProvider::new_with_endpoint(
            &endpoint,
            TEST_BUCKET,
            TEST_ACCESS_KEY,
            TEST_SECRET_KEY,
            TEST_REGION,
        )
        .expect("s3 provider for tvf consumer"),
    );
    let cache_dir = TempDir::new().expect("tvf cache tempdir");
    let cache = make_cache(Arc::clone(&consumer_storage), cache_dir.path());

    let consumer = Supertable::open(
        real_s3_options(dim)
            .with_storage(Arc::clone(&consumer_storage))
            .with_disk_cache(Arc::clone(&cache)),
    )
    .expect("Supertable::open via S3 (tvf consumer)");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_docs_total(), 8);

    let pre = cache.stats();

    // One-hot query vector at dim 0. `real_s3_batch` row 0
    // has emb[0]=1.0, so doc 0 is the closest vector match.
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

    // 1. bm25_search through query_sql. The corpus has "alpha"
    //    in exactly two titles ("alpha vector one", "alpha
    //    vector two"), so the TVF must return >= 2 rows.
    let bm25 = consumer
        .query_sql("SELECT _id FROM bm25_search('title', 'alpha', 10)")
        .expect("bm25_search via query_sql over S3");
    assert!(
        count_rows(&bm25) >= 2,
        "bm25_search('alpha') should return >=2 docs over S3; got {}",
        count_rows(&bm25)
    );

    // 2. vector_search through query_sql. k=3.
    let vec_sql = format!("SELECT _id FROM vector_search('emb', '{q_csv}', 3)");
    let vector = consumer
        .query_sql(&vec_sql)
        .expect("vector_search via query_sql over S3");
    assert!(
        count_rows(&vector) >= 1,
        "vector_search returned no rows over S3"
    );

    // 3. hybrid_search through query_sql. RRF fusion over the
    //    same two retrievers; k=5.
    let hybrid_sql =
        format!("SELECT _id FROM hybrid_search('title', 'alpha', 'emb', '{q_csv}', 5)");
    let hybrid = consumer
        .query_sql(&hybrid_sql)
        .expect("hybrid_search via query_sql over S3");
    let hyb_rows = count_rows(&hybrid);
    assert!(
        hyb_rows > 0 && hyb_rows <= 5,
        "hybrid_search rows in (0, 5]; got {hyb_rows}"
    );

    // 4. Cold-fetch counter grew -> confirms TVF reads went
    //    through the S3 wire path, not a local short-circuit.
    let post = cache.stats();
    assert!(
        post.n_cold_fetches > pre.n_cold_fetches,
        "TVF queries must cold-fetch through S3; pre={} post={}",
        pre.n_cold_fetches,
        post.n_cold_fetches
    );

    eprintln!(
        "[m16-tvf] bm25 / vector / hybrid via query_sql over S3 OK; \
         n_cold_fetches={} cache_bytes={}",
        post.n_cold_fetches, post.current_bytes
    );
}
