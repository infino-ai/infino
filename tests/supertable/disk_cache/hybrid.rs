// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hybrid cold-fetch (`ColdFetchMode::HybridWithPrefetch`).
//!
//! Builds on the disk-cache infrastructure with the
//! foreground-broadcast-then-finalize architecture:
//!
//! - Range-GETs run in parallel and feed two consumers: the
//!   foreground reader (assembled in-memory) and a
//!   fire-and-forget pwrite + mmap pipeline running in the
//!   background.
//! - The foreground returns when all range-fetches finish;
//!   pwrites + fsync + rename + mmap + cache registration
//!   finalize in a separate task that outlives this method.
//! - Bandwidth per cold miss = 1× superfile size (one set of
//!   `get_range` calls serves both foreground and cache fill).
//! - Concurrent cold readers on the same URI coalesce to a
//!   single coordinator (single fetch fan-out).
//! - `ColdFetchMode::RangeOnly` bypasses the cache entirely:
//!   `DiskCacheStore::reader` rejects it; callers construct
//!   `StorageRangeSource` + `open_lazy` directly.

#![deny(clippy::unwrap_used)]

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
    supertable::{
        SuperfileUri,
        reader_cache::{
            ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy, disk::DiskCacheError,
        },
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{decimal128_ids, default_tokenizer},
};
use tempfile::TempDir;

// ============================================================
// Counting proxy — measures bandwidth-per-cold-miss.
// ============================================================

#[derive(Debug)]
struct CountingProxy {
    inner: Arc<dyn StorageProvider>,
    get_range_calls: AtomicUsize,
    get_range_bytes: AtomicUsize,
}

impl CountingProxy {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            get_range_calls: AtomicUsize::new(0),
            get_range_bytes: AtomicUsize::new(0),
        })
    }
    fn calls(&self) -> usize {
        self.get_range_calls.load(Ordering::Acquire)
    }
    fn bytes(&self) -> usize {
        self.get_range_bytes.load(Ordering::Acquire)
    }
}

#[async_trait]
impl StorageProvider for CountingProxy {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
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
/// Disk-cache budget (1 GiB) for the hybrid-cache tests.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams.
const COLD_FETCH_STREAMS: usize = 4;
/// Small cold-fetch chunk to force multi-range fetches.
const COLD_FETCH_CHUNK_BYTES_SMALL: u64 = 64;
/// Sleep allowing the background finalizer to run between assertions.
const BACKGROUND_FINALIZER_SLEEP_MS: u64 = 50;
/// Concurrent hybrid readers for the coalescing test.
const HYBRID_CONCURRENT_READER_COUNT: usize = 50;

fn build_test_bytes() -> Bytes {
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
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("builder");
    let ids = decimal128_ids(vec![1u64, 2, 3]);
    let titles = LargeStringArray::from(vec!["alpha bravo", "charlie delta", "echo foxtrot"]);
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)]).expect("batch");
    b.add_batch(&batch, &[]).expect("add_batch");
    Bytes::from(b.finish().expect("finish"))
}

async fn seed(storage: &dyn StorageProvider, uri: SuperfileUri, bytes: Bytes) {
    let path = uri.storage_path();
    storage.put_atomic(&path, bytes).await.expect("seed");
}

fn fresh_cache(
    storage: Arc<dyn StorageProvider>,
    mode: ColdFetchMode,
) -> (TempDir, Arc<DiskCacheStore>) {
    let dir = TempDir::new().expect("tempdir");
    let cfg = DiskCacheConfig {
        cache_root: dir.path().to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: mode,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES_SMALL,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let store = DiskCacheStore::new_unpinned(storage, cfg).expect("store");
    (dir, store)
}

// ============================================================
// Tests.
// ============================================================

#[tokio::test]
async fn hybrid_mode_is_default() {
    // Just construct via Default and read the mode back.
    let cfg = DiskCacheConfig::default();
    assert_eq!(cfg.cold_fetch_mode, ColdFetchMode::HybridWithPrefetch);
}

#[tokio::test]
async fn hybrid_reader_returns_working_superfile_reader() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let (_d, cache) = fresh_cache(local, ColdFetchMode::HybridWithPrefetch);
    let reader = cache.reader(&uri).await.expect("reader");
    // Sanity: in-memory-bytes-backed reader serves FTS terms.
    let fts = reader.fts().expect("fts");
    let terms = fts.iter_column_terms("title").expect("iter terms");
    assert!(terms.iter().any(|t| t.as_slice() == b"alpha"));
}

#[tokio::test]
async fn hybrid_bandwidth_per_cold_miss_equals_superfile_size() {
    // The "1× bandwidth per cold miss" invariant —
    // the same range responses serve both foreground and
    // cache fill; no re-fetching.
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let superfile_size = bytes.len();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(
        Arc::clone(&proxy) as Arc<dyn StorageProvider>,
        ColdFetchMode::HybridWithPrefetch,
    );

    let _r = cache.reader(&uri).await.expect("cold reader");
    // Wait for the background finalizer to complete so we
    // measure total bandwidth (including any post-foreground
    // work — we expect zero post-foreground get_range bytes).
    tokio::time::sleep(std::time::Duration::from_millis(
        BACKGROUND_FINALIZER_SLEEP_MS,
    ))
    .await;

    let bytes_fetched = proxy.bytes();
    assert_eq!(
        bytes_fetched, superfile_size,
        "1× bandwidth invariant: total get_range bytes ({}) must equal superfile size ({}); \
         any excess indicates re-fetching between foreground and cache fill",
        bytes_fetched, superfile_size
    );
}

#[tokio::test]
async fn hybrid_warm_hit_issues_zero_range_fetches() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(
        Arc::clone(&proxy) as Arc<dyn StorageProvider>,
        ColdFetchMode::HybridWithPrefetch,
    );

    let _r = cache.reader(&uri).await.expect("cold");
    let calls_after_cold = proxy.calls();
    // Let the background finalize complete so subsequent calls
    // see the mmap-backed cached entry.
    tokio::time::sleep(std::time::Duration::from_millis(
        BACKGROUND_FINALIZER_SLEEP_MS,
    ))
    .await;
    let _r2 = cache.reader(&uri).await.expect("warm");
    assert_eq!(
        proxy.calls(),
        calls_after_cold,
        "warm hit must issue zero additional get_range calls"
    );
}

#[tokio::test]
async fn hybrid_concurrent_readers_coalesce_to_one_fetch_fan_out() {
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let bytes = build_test_bytes();
    let superfile_size = bytes.len();
    let uri = SuperfileUri::new_v4();
    seed(&*local, uri, bytes).await;

    let proxy = CountingProxy::new(local);
    let (_d, cache) = fresh_cache(
        Arc::clone(&proxy) as Arc<dyn StorageProvider>,
        ColdFetchMode::HybridWithPrefetch,
    );

    let mut joins = Vec::with_capacity(HYBRID_CONCURRENT_READER_COUNT);
    for _ in 0..HYBRID_CONCURRENT_READER_COUNT {
        let cache = Arc::clone(&cache);
        joins.push(tokio::spawn(async move { cache.reader(&uri).await }));
    }
    for h in joins {
        let _ = h.await.expect("join").expect("reader");
    }
    // Bandwidth still equals one superfile size — coalescing
    // ensured exactly one fetch fan-out served all 50.
    tokio::time::sleep(std::time::Duration::from_millis(
        BACKGROUND_FINALIZER_SLEEP_MS,
    ))
    .await;
    assert_eq!(
        proxy.bytes(),
        superfile_size,
        "50 concurrent cold readers must trigger exactly one fan-out"
    );
    let stats = cache.stats();
    assert_eq!(stats.n_cold_fetches, 1);
}

#[tokio::test]
async fn range_only_mode_bypasses_disk_cache() {
    // `ColdFetchMode::RangeOnly` is the stateless path —
    // `DiskCacheStore::reader` rejects it because the cache
    // isn't the right entry point (callers should use
    // `StorageRangeSource` + `SuperfileReader::open_lazy`
    // directly).
    //
    // Note: Currently, setting a cache directory with RangeOnly is
    // rejected at construction time. A future addition that relaxes this to
    // support RangeOnly with cache fallback will need to update
    // this test to assert reader rejection or admission rules.
    let store_dir = TempDir::new().expect("storage");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));

    let cfg = DiskCacheConfig {
        cache_root: store_dir.path().to_path_buf(),
        cold_fetch_mode: ColdFetchMode::RangeOnly,
        ..Default::default()
    };
    let err = DiskCacheStore::new_unpinned(local, cfg)
        .expect_err("RangeOnly mode must be rejected during cache construction");
    assert!(
        matches!(err, DiskCacheError::Config(_)),
        "expected DiskCacheError::Config, got {err:?}"
    );
}
