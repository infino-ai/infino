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

use infino::supertable::Supertable;
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
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    DiskCacheStore::new(storage, cfg, pinned).expect("cache")
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
    .await
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
