// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-cache layer with parallel cold fetch.
//!
//! Builds a tiny real superfile via `SuperfileBuilder`, puts
//! it into a `LocalFsStorageProvider`, wraps that in a
//! `CountingProxy` (so we can assert on `get_range` /
//! `head` call counts), and exercises `DiskCacheStore`
//! through the invariants it promises:
//!
//! - cold miss triggers cold-fetch (range-GETs to assemble
//!   the superfile file)
//! - warm hit issues zero `get_range` calls
//! - 100 concurrent cold readers on the same URI coalesce to
//!   exactly one fetch fan-out
//! - reader returns a working `SuperfileReader` (validates
//!   the mmap → `Bytes::from_owner` → `SuperfileReader::open`
//!   path)
//! - eviction respects the pinned-set callback
//! - over-budget with everything pinned surfaces
//!   `DiskCacheError::BudgetExceeded`
//! - reservation-race correctness: N concurrent cold-misses
//!   on distinct URIs whose total exceeds budget preserve
//!   `current_bytes ≤ disk_budget_bytes` invariantly

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use bytes::Bytes;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use infino::supertable::SuperfileUri;
use infino::supertable::reader_cache::disk::DiskCacheError;
use infino::supertable::reader_cache::{DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::storage::{
    LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider,
};
use infino::test_helpers::{decimal128_ids, default_tokenizer};
use tempfile::TempDir;

// ============================================================
// Counting proxy over any StorageProvider.
// ============================================================

#[derive(Debug)]
struct CountingProxy {
    inner: Arc<dyn StorageProvider>,
    head_calls: AtomicUsize,
    get_calls: AtomicUsize,
    get_range_calls: AtomicUsize,
}

impl CountingProxy {
    fn new(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            head_calls: AtomicUsize::new(0),
            get_calls: AtomicUsize::new(0),
            get_range_calls: AtomicUsize::new(0),
        })
    }

    fn get_range_count(&self) -> usize {
        self.get_range_calls.load(Ordering::Acquire)
    }

    fn head_count(&self) -> usize {
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
        self.get_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.get_range_calls.fetch_add(1, Ordering::AcqRel);
        self.inner.get_range(uri, range).await
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

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Generous disk-cache budget (1 GiB) used by the non-eviction tests.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Parallel cold-fetch streams for the test cache.
const COLD_FETCH_STREAMS: usize = 4;
/// Small cold-fetch chunk to force multiple ranges on tiny payloads.
const COLD_FETCH_CHUNK_BYTES_SMALL: u64 = 64;
/// Concurrent cold readers for the coalescing stress test.
const CONCURRENT_COLD_READER_COUNT: usize = 100;
/// Distinct URIs for the reservation-race test.
const RESERVATION_RACE_URI_COUNT: u32 = 8;
/// Sleep between LRU touches to make access ordering deterministic.
const LRU_TOUCH_SLEEP_MS: u64 = 1;

// ============================================================
// Tiny superfile fixture.
// ============================================================

fn build_test_superfile_bytes() -> Bytes {
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

async fn seed_superfile(storage: &dyn StorageProvider, uri: SuperfileUri, bytes: Bytes) {
    let path = uri.storage_path();
    storage.put_atomic(&path, bytes).await.expect("seed put");
}

fn fresh_cache_with_storage(
    storage: Arc<dyn StorageProvider>,
    budget_bytes: u64,
) -> (TempDir, Arc<DiskCacheStore>) {
    let cache_dir = TempDir::new().expect("tempdir");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        disk_budget_bytes: budget_bytes,
        cold_fetch_mode: infino::supertable::reader_cache::ColdFetchMode::HybridWithPrefetch,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES_SMALL, // force multiple ranges on tiny payload
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let store = DiskCacheStore::new_unpinned(storage, cfg).expect("store");
    (cache_dir, store)
}

// ============================================================
// Tests.
// ============================================================

#[tokio::test]
async fn cold_miss_triggers_range_fetches_warm_hit_does_not() {
    let store_dir = TempDir::new().expect("storage tempdir");
    let local = Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let proxy = CountingProxy::new(local);

    let uri = SuperfileUri::new_v4();
    let bytes = build_test_superfile_bytes();
    seed_superfile(&*proxy, uri, bytes.clone()).await;

    let (_cdir, cache) = fresh_cache_with_storage(
        Arc::clone(&proxy) as Arc<dyn StorageProvider>,
        DISK_CACHE_BUDGET_BYTES,
    );

    // Cold miss: at least one head + ≥1 range fetch.
    let _r = cache.reader(&uri).await.expect("cold reader");
    let head_after_cold = proxy.head_count();
    let range_after_cold = proxy.get_range_count();
    assert!(head_after_cold >= 1, "cold miss must HEAD the object");
    assert!(
        range_after_cold >= 1,
        "cold miss must issue at least one get_range; got {range_after_cold}"
    );

    // Warm hit: zero additional get_range calls.
    let _r2 = cache.reader(&uri).await.expect("warm reader");
    assert_eq!(
        proxy.get_range_count(),
        range_after_cold,
        "warm hit must not re-fetch (range count unchanged)"
    );
    // Stats reflect one cold fetch + one entry.
    let stats = cache.stats();
    assert_eq!(stats.n_entries, 1);
    assert_eq!(stats.n_cold_fetches, 1);
}

#[tokio::test]
async fn concurrent_cold_readers_coalesce_to_one_fetch() {
    let store_dir = TempDir::new().expect("storage tempdir");
    let local = Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));
    let proxy = CountingProxy::new(local);

    let uri = SuperfileUri::new_v4();
    let bytes = build_test_superfile_bytes();
    seed_superfile(&*proxy, uri, bytes).await;

    let (_cdir, cache) = fresh_cache_with_storage(
        Arc::clone(&proxy) as Arc<dyn StorageProvider>,
        DISK_CACHE_BUDGET_BYTES,
    );

    // Spawn 100 concurrent readers; OnceCell-coalescing
    // should produce exactly ONE cold fetch.
    let mut joins = Vec::with_capacity(CONCURRENT_COLD_READER_COUNT);
    for _ in 0..CONCURRENT_COLD_READER_COUNT {
        let cache = Arc::clone(&cache);
        joins.push(tokio::spawn(async move { cache.reader(&uri).await }));
    }
    for h in joins {
        let _ = h.await.expect("join").expect("reader ok");
    }

    let stats = cache.stats();
    assert_eq!(
        stats.n_cold_fetches, 1,
        "100 concurrent cold readers must coalesce to 1 cold fetch; got {}",
        stats.n_cold_fetches
    );
    assert_eq!(proxy.head_count(), 1, "one HEAD per coalesced cold miss");
}

#[tokio::test]
async fn reader_returns_working_superfile_reader() {
    // Validates the mmap → Bytes::from_owner →
    // SuperfileReader::open path produces a reader that
    // actually serves queries.
    let store_dir = TempDir::new().expect("storage tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));

    let uri = SuperfileUri::new_v4();
    let bytes = build_test_superfile_bytes();
    seed_superfile(&*local, uri, bytes).await;

    let (_cdir, cache) = fresh_cache_with_storage(Arc::clone(&local), DISK_CACHE_BUDGET_BYTES);
    let reader = cache.reader(&uri).await.expect("reader");

    // Sanity: the mmap-backed reader exposes an FTS reader
    // and the indexed terms include our planted token.
    let fts = reader.fts().expect("fts reader");
    let title_terms = fts.iter_column_terms("title").expect("iter terms");
    assert!(
        title_terms.iter().any(|t| t.as_slice() == b"alpha"),
        "mmap-backed reader must expose the planted FTS term"
    );
}

#[tokio::test]
async fn eviction_respects_pinned_set() {
    // Two superfiles, budget tight enough that only one fits.
    // Pin superfile A; ask for B; expect A to survive
    // (BudgetExceeded surfaces because the policy can't evict
    // A and B alone exceeds budget when A is held).
    let store_dir = TempDir::new().expect("storage tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));

    let uri_a = SuperfileUri::new_v4();
    let uri_b = SuperfileUri::new_v4();
    let bytes = build_test_superfile_bytes();
    let size = bytes.len() as u64;
    seed_superfile(&*local, uri_a, bytes.clone()).await;
    seed_superfile(&*local, uri_b, bytes).await;

    // Pinned-fn pins exactly URI A.
    let pinned: Arc<dyn Fn() -> HashSet<SuperfileUri> + Send + Sync> = Arc::new(move || {
        let mut s = HashSet::new();
        s.insert(uri_a);
        s
    });

    let cache_dir = TempDir::new().expect("cache tempdir");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        // Budget tight: fits exactly one superfile, not two.
        cold_fetch_mode: infino::supertable::reader_cache::ColdFetchMode::HybridWithPrefetch,
        disk_budget_bytes: size + (size / 2),
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES_SMALL,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let cache = DiskCacheStore::new(Arc::clone(&local), cfg, pinned).expect("cache");

    // Load A first (cold miss; reserves `size`).
    let _ra = cache.reader(&uri_a).await.expect("a");
    // Try B — needs another `size` worth, but A is pinned;
    // eviction can't free anything → BudgetExceeded.
    let err = cache.reader(&uri_b).await.expect_err("b must fail");
    assert!(
        matches!(err, DiskCacheError::BudgetExceeded),
        "expected BudgetExceeded, got {err:?}"
    );
    // A is still cached + the budget tracker reflects A only.
    let stats = cache.stats();
    assert_eq!(stats.n_entries, 1);
    assert_eq!(stats.current_bytes, size);
}

#[tokio::test]
async fn lru_evicts_oldest_unpinned_when_budget_pressure_hits() {
    // Three superfiles, budget for two. Touch A then B (B is
    // newer). Touch C → forces eviction; A (older) should be
    // the victim.
    let store_dir = TempDir::new().expect("storage tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));

    let uri_a = SuperfileUri::new_v4();
    let uri_b = SuperfileUri::new_v4();
    let uri_c = SuperfileUri::new_v4();
    let bytes = build_test_superfile_bytes();
    let size = bytes.len() as u64;
    seed_superfile(&*local, uri_a, bytes.clone()).await;
    seed_superfile(&*local, uri_b, bytes.clone()).await;
    seed_superfile(&*local, uri_c, bytes).await;

    let cache_dir = TempDir::new().expect("cache");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        // Room for ~2 superfiles.
        cold_fetch_mode: infino::supertable::reader_cache::ColdFetchMode::HybridWithPrefetch,
        disk_budget_bytes: 2 * size + (size / 4),
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES_SMALL,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let cache = DiskCacheStore::new_unpinned(Arc::clone(&local), cfg).expect("cache");

    let _ra = cache.reader(&uri_a).await.expect("a");
    // Tiny sleep so B's last_access_us > A's. Avoids relying
    // on observation order alone.
    tokio::time::sleep(std::time::Duration::from_millis(LRU_TOUCH_SLEEP_MS)).await;
    let _rb = cache.reader(&uri_b).await.expect("b");
    tokio::time::sleep(std::time::Duration::from_millis(LRU_TOUCH_SLEEP_MS)).await;
    // Cold miss on C → eviction picks A (oldest unpinned).
    let _rc = cache.reader(&uri_c).await.expect("c");

    let stats = cache.stats();
    assert_eq!(stats.n_entries, 2, "still two entries after eviction");
    assert_eq!(stats.n_evictions, 1);
}

#[tokio::test]
async fn reservation_race_preserves_budget_invariant() {
    // N concurrent cold misses on distinct URIs whose total
    // size > budget. The CAS-loop reservation must guarantee
    // current_bytes ≤ budget at every observation, never
    // overshooting transiently.
    let store_dir = TempDir::new().expect("storage tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(store_dir.path()).expect("local"));

    let bytes = build_test_superfile_bytes();
    let size = bytes.len() as u64;
    // Seed 8 distinct URIs.
    let uris: Vec<SuperfileUri> = (0..RESERVATION_RACE_URI_COUNT)
        .map(|_| SuperfileUri::new_v4())
        .collect();
    for u in &uris {
        seed_superfile(&*local, *u, bytes.clone()).await;
    }

    let cache_dir = TempDir::new().expect("cache");
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        // Budget for ~3 superfiles. With 8 concurrent cold (clipped — see below).
        cold_fetch_mode: infino::supertable::reader_cache::ColdFetchMode::HybridWithPrefetch,
        // misses, eviction will fire repeatedly.
        disk_budget_bytes: 3 * size,
        cold_fetch_streams: COLD_FETCH_STREAMS,
        cold_fetch_chunk_bytes: COLD_FETCH_CHUNK_BYTES_SMALL,
        mmap_cold_threshold_secs: 0,
        mmap_sweep_interval_secs: 0,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let cache = DiskCacheStore::new_unpinned(Arc::clone(&local), cfg).expect("cache");

    // Spawn 8 concurrent readers. With budget for ~3 and
    // 8 distinct URIs, some readers may legitimately hit
    // BudgetExceeded (their reservation arrives when
    // in-flight reservations have eaten the budget and
    // nothing's yet committed to evict). The stated
    // invariant is `current_bytes ≤ budget` at every
    // observation — NOT that all readers succeed. We assert
    // only the invariant.
    let mut joins = Vec::with_capacity(uris.len());
    for u in &uris {
        let cache = Arc::clone(&cache);
        let u = *u;
        joins.push(tokio::spawn(async move { cache.reader(&u).await }));
    }
    let mut n_ok = 0usize;
    let mut n_budget_exceeded = 0usize;
    for h in joins {
        match h.await.expect("join") {
            Ok(_) => n_ok += 1,
            Err(DiskCacheError::BudgetExceeded) => n_budget_exceeded += 1,
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    let stats = cache.stats();
    assert!(
        stats.current_bytes <= stats.budget_bytes,
        "invariant violated: current_bytes={} budget={}",
        stats.current_bytes,
        stats.budget_bytes
    );
    assert_eq!(
        n_ok + n_budget_exceeded,
        uris.len(),
        "every reader must terminate with either Ok or BudgetExceeded"
    );
    // At least one reader committed (otherwise the cache is
    // empty + budget pressure makes no sense).
    assert!(
        n_ok >= 1,
        "expected at least 1 reader to succeed; got {n_ok} ok / {n_budget_exceeded} budget_exceeded"
    );
}
