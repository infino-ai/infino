// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `ColdFetchMode::LazyForegroundWithBackgroundFill`
//! integration. The cold path returns a lazy reader
//! immediately (paying only the cold-open byte budget
//! against object storage), a background task waits for the
//! foreground lazy reader to release, downloads the full
//! superfile to NVMe + mmaps it, and **any subsequent
//! `reader(uri)` call returns the mmap-backed reader** — the
//! corresponding search issues zero S3 GETs.
//!
//! These tests cover:
//!
//! - The cold-foreground reader is functional immediately
//!   (FTS queries return correct results) without waiting for
//!   the background superfile fill.
//! - The warm-path zero-S3-GET invariant: after the
//!   background promotion completes, a second `reader(uri)`
//!   plus a search resolves entirely from mmap — the counting
//!   storage proxy observes zero additional `get_range`
//!   calls.
//! - Concurrent cold readers on the same URI coalesce
//!   correctly: every caller observes the same lazy reader
//!   (no duplicate background fills, no extra cold-fetch
//!   coordinator activity).

#![deny(clippy::unwrap_used)]

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    superfile::{
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    supertable::{
        SuperfileUri,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{decimal128_ids, default_tokenizer},
};
use tempfile::TempDir;

// ============================================================
// Counting proxy — captures the S3-GET budget under test.
// ============================================================

#[derive(Debug)]
struct CountingProxy {
    inner: Arc<dyn StorageProvider>,
    get_range_calls: AtomicUsize,
    get_range_bytes: AtomicUsize,
    head_calls: AtomicUsize,
}

impl CountingProxy {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            get_range_calls: AtomicUsize::new(0),
            get_range_bytes: AtomicUsize::new(0),
            head_calls: AtomicUsize::new(0),
        })
    }
    fn calls(&self) -> usize {
        self.get_range_calls.load(Ordering::Acquire)
    }
    fn bytes(&self) -> usize {
        self.get_range_bytes.load(Ordering::Acquire)
    }
    #[allow(dead_code)]
    fn heads(&self) -> usize {
        self.head_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl StorageProvider for CountingProxy {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.head_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.get_range_calls.fetch_add(1, Ordering::AcqRel);
        let b = self.inner.get_range(uri, range).await?;
        self.get_range_bytes.fetch_add(b.len(), Ordering::AcqRel);
        Ok(b)
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        e: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_if_match(uri, bytes, e).await
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
}

// ============================================================
// Fixtures.
// ============================================================

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Disk-cache budget (1 GiB) for the lazy-foreground tests.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams.
const COLD_FETCH_STREAMS: usize = 4;
/// Cold-fetch chunk size for the lazy-foreground path.
const LAZY_COLD_FETCH_CHUNK_BYTES: u64 = 256;
/// BM25 top-k for the lazy-foreground searches.
const FTS_TOP_K: usize = 10;
/// Smaller top-k for cold/warm path searches.
const FTS_TOP_K_SMALL: usize = 5;
/// Hold a reader before dropping to order the background fill.
const FOREGROUND_HOLD_SLEEP_MS: u64 = 100;
/// Mmap-promotion wait timeout.
const MMAP_PROMOTION_TIMEOUT_SECS: u64 = 5;
/// Concurrent lazy readers for the coalescing test.
const LAZY_CONCURRENT_READER_COUNT: usize = 16;

fn build_fts_only_bytes() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    let ids = decimal128_ids(vec![1u64, 2, 3, 4]);
    let titles = LargeStringArray::from(vec![
        "alpha bravo special",
        "charlie delta",
        "echo special foxtrot",
        "gamma hotel",
    ]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
    b.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

async fn seed(storage: &dyn StorageProvider, uri: SuperfileUri, bytes: Bytes) {
    let path = uri.storage_path();
    storage.put_atomic(&path, bytes).await.expect("seed");
}

fn fresh_cache(storage: Arc<dyn StorageProvider>) -> (TempDir, Arc<DiskCacheStore>) {
    let dir = TempDir::new().expect("tempdir");
    let cfg = DiskCacheConfig {
        cache_root: dir.path().to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: LAZY_COLD_FETCH_CHUNK_BYTES,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: false,
        ..Default::default()
    };
    let store = DiskCacheStore::new_unpinned(storage, cfg).expect("store");
    (dir, store)
}

async fn wait_for_mmap_promotion(
    cache: &Arc<DiskCacheStore>,
    uri: SuperfileUri,
    timeout: Duration,
) {
    cache
        .wait_until_mmap_promoted(&uri, timeout)
        .await
        .expect("mmap promotion");
}

// ============================================================
// Tests.
// ============================================================

/// cold reader from
/// `LazyForegroundWithBackgroundFill` is functional
/// immediately. The reader is FTS-queryable without waiting
/// for the background superfile fill; the warm-promotion
/// guarantee is a separate, additive property tested below.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lazy_foreground_cold_reader_is_queryable_immediately() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_fts_only_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(Arc::clone(&proxy) as Arc<dyn StorageProvider>);

    let reader = cache.reader(&uri).await.expect("cold reader");
    // Running an FTS search against the lazy reader works
    // without any extra wait — the source-driven path fetches
    // the FTS subsection on demand.
    let fts = reader.fts().expect("fts");
    let hits = fts
        .search("title", &["special"], FTS_TOP_K, BoolMode::Or)
        .await
        .expect("bm25");
    assert_eq!(hits.len(), 2, "two docs contain 'special'");
}

/// The background full-superfile fill must not compete with a
/// foreground lazy reader. Holding the reader across a short
/// delay should not issue cache-fill range GETs; promotion begins
/// only after the reader is dropped.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lazy_background_fill_waits_for_foreground_reader_drop() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_fts_only_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(Arc::clone(&proxy) as Arc<dyn StorageProvider>);

    let reader = cache.reader(&uri).await.expect("cold reader");
    let calls_after_open = proxy.calls();
    tokio::time::sleep(Duration::from_millis(FOREGROUND_HOLD_SLEEP_MS)).await;
    assert_eq!(
        proxy.calls(),
        calls_after_open,
        "background fill should wait while foreground lazy reader is held"
    );

    drop(reader);
    wait_for_mmap_promotion(
        &cache,
        uri,
        Duration::from_secs(MMAP_PROMOTION_TIMEOUT_SECS),
    )
    .await;
    assert!(
        proxy.calls() > calls_after_open,
        "background fill should begin after foreground reader drops"
    );
}

/// **the** invariant: after the background
/// promotion completes, a second `reader(uri)` call returns
/// the mmap-backed reader and the corresponding search
/// resolves entirely from mmap — the counting storage proxy
/// observes **zero** additional `get_range` calls.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lazy_foreground_warm_search_after_promotion_issues_zero_s3_gets() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_fts_only_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(Arc::clone(&proxy) as Arc<dyn StorageProvider>);

    // 1. Cold reader — returns immediately with a lazy reader.
    //    Run a search through it so the foreground path
    //    actually exercises the per-query range GETs.
    {
        let r_cold = cache.reader(&uri).await.expect("cold reader");
        let fts = r_cold.fts().expect("fts");
        let _ = fts
            .search("title", &["alpha"], FTS_TOP_K_SMALL, BoolMode::Or)
            .await
            .expect("cold bm25");
    }
    let calls_after_cold = proxy.calls();
    assert!(
        calls_after_cold > 0,
        "cold lazy reader + search must issue at least one S3 get_range; got 0"
    );

    // 2. Wait for the background promotion to complete. The
    //    fill is spawned via `tokio::spawn` from the cold
    //    foreground; once it finishes, the cache entry has
    //    been atomically swapped to the mmap-backed reader.
    wait_for_mmap_promotion(
        &cache,
        uri,
        Duration::from_secs(MMAP_PROMOTION_TIMEOUT_SECS),
    )
    .await;

    // 3. Warm reader + search — must hit the promoted
    //    mmap-backed entry and issue zero additional
    //    `get_range` calls.
    let calls_before_warm = proxy.calls();
    {
        let r_warm = cache.reader(&uri).await.expect("warm reader");
        let fts = r_warm.fts().expect("fts");
        let hits = fts
            .search("title", &["special"], FTS_TOP_K_SMALL, BoolMode::Or)
            .await
            .expect("warm bm25");
        assert_eq!(hits.len(), 2, "warm search must return correct results");
    }
    let calls_after_warm = proxy.calls();
    assert_eq!(
        calls_after_warm,
        calls_before_warm,
        "warm-path zero-S3-GET invariant violated: warm reader + search \
         issued {} additional get_range calls (cold={}, before_warm={}, \
         after_warm={}); all bytes should have come from mmap",
        calls_after_warm - calls_before_warm,
        calls_after_cold,
        calls_before_warm,
        calls_after_warm,
    );
}

/// concurrent cold readers on the same URI
/// coalesce through the `OnceCell` coordinator. All callers
/// observe the same lazy reader; the background promotion
/// runs exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lazy_foreground_concurrent_cold_readers_coalesce_to_one_promotion() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_fts_only_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(Arc::clone(&proxy) as Arc<dyn StorageProvider>);

    // 16 concurrent cold readers on the same URI — all should
    // coalesce to a single background promotion.
    let mut joins = Vec::with_capacity(LAZY_CONCURRENT_READER_COUNT);
    for _ in 0..LAZY_CONCURRENT_READER_COUNT {
        let cache = Arc::clone(&cache);
        joins.push(tokio::spawn(async move { cache.reader(&uri).await }));
    }
    for h in joins {
        let _ = h.await.expect("join").expect("reader");
    }

    let stats = cache.stats();
    assert_eq!(
        stats.n_cold_fetches, 1,
        "16 concurrent cold readers must coalesce to exactly one \
         cold-fetch coordinator; got {}",
        stats.n_cold_fetches,
    );
}

/// the cold-path bandwidth profile is **2× per
/// cold miss** (per-query ranges + background full-superfile
/// download). This test documents that property by asserting
/// the total `get_range` bytes are at least `superfile_size`
/// (the background fill) plus the cold foreground's per-query
/// range — i.e., that the background fill actually runs.
/// Counter-balances `hybrid_bandwidth_per_cold_miss_equals_superfile_size`'s
/// 1× invariant for the hybrid mode.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lazy_foreground_total_bandwidth_includes_background_fill() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_fts_only_bytes();
    let superfile_size = bytes.len();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(Arc::clone(&proxy) as Arc<dyn StorageProvider>);

    // Background fill yields to a held foreground lazy reader (per-URI
    // quiescence); drop it so the fill can proceed and promote to mmap.
    let r = cache.reader(&uri).await.expect("cold");
    drop(r);
    wait_for_mmap_promotion(
        &cache,
        uri,
        Duration::from_secs(MMAP_PROMOTION_TIMEOUT_SECS),
    )
    .await;

    let total_bytes = proxy.bytes();
    assert!(
        total_bytes >= superfile_size,
        "background fill must download the full superfile ({} bytes); \
         counting proxy observed {} bytes total",
        superfile_size,
        total_bytes,
    );
}
