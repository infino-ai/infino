// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN cold-search GET + fetch-wave measurement helpers.
//!
//! Shared by the routing correctness integration tests and the supertable
//! vector bench: wrap a `StorageProvider` to count read-path fetches and the
//! number of **fetch waves** — each round where the query engine issues
//! multiple range GETs concurrently (a `try_join_all` / tokio batch). Overlapping
//! in-flight GETs belong to the same wave; when every GET from the batch has
//! finished and the next batch starts, that is a new wave.

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::MultipartUpload;

use crate::supertable::storage::{ObjectMeta, StorageError, StorageProvider};

/// Atomic counters paired with [`CountingStorage`].
#[derive(Debug, Clone)]
pub struct OpannColdCounters {
    pub fetches: Arc<AtomicUsize>,
    pub tombstone_fetches: Arc<AtomicUsize>,
    pub waves: Arc<AtomicUsize>,
    pub in_flight: Arc<AtomicUsize>,
}

impl OpannColdCounters {
    pub fn new() -> Self {
        Self {
            fetches: Arc::new(AtomicUsize::new(0)),
            tombstone_fetches: Arc::new(AtomicUsize::new(0)),
            waves: Arc::new(AtomicUsize::new(0)),
            in_flight: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn reset(&self) {
        self.fetches.store(0, Ordering::Relaxed);
        self.tombstone_fetches.store(0, Ordering::Relaxed);
        self.waves.store(0, Ordering::Relaxed);
        self.in_flight.store(0, Ordering::Relaxed);
    }

    pub fn gets(&self) -> usize {
        self.fetches.load(Ordering::Relaxed)
    }

    pub fn tombstone_gets(&self) -> usize {
        self.tombstone_fetches.load(Ordering::Relaxed)
    }

    pub fn waves(&self) -> u64 {
        self.waves.load(Ordering::Relaxed) as u64
    }
}

impl Default for OpannColdCounters {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-search object-store fetch counts for one timed cold query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColdVectorSearchMeasure {
    pub gets: usize,
    pub tombstone_gets: usize,
    /// Fetch waves: rounds of parallel in-flight GETs (see [`CountingStorage`]).
    pub waves: u64,
    pub wall_ms: u64,
}

/// A cold search on a fresh cache must issue at least one read through the
/// counting storage wrapper. Zero GETs with a successful recall means the
/// measurement missed real fetches (in-process cache bleed, wrong storage
/// handle, etc.) and must fail loudly.
pub fn assert_cold_search_issued_gets(measure: ColdVectorSearchMeasure, label: &str) {
    assert!(
        measure.gets > 0,
        "{label}: cold search issued zero object-store GETs through the counting \
         wrapper (expected manifest, OPANN, and/or superfile range fetches)"
    );
}

/// Reset counters, run `search`, return GET/wave counts plus the search result.
pub fn measure_cold_search_timed<T>(
    counters: &OpannColdCounters,
    search: impl FnOnce() -> T,
) -> (ColdVectorSearchMeasure, T) {
    counters.reset();
    let t0 = Instant::now();
    let result = search();
    let wall_ms = t0.elapsed().as_millis() as u64;
    (
        ColdVectorSearchMeasure {
            gets: counters.gets(),
            tombstone_gets: counters.tombstone_gets(),
            waves: counters.waves(),
            wall_ms,
        },
        result,
    )
}

/// `StorageProvider` decorator that counts read-path fetches and fetch waves.
#[derive(Debug)]
pub struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    fetches: Arc<AtomicUsize>,
    tombstone_fetches: Arc<AtomicUsize>,
    waves: Arc<AtomicUsize>,
    in_flight: Arc<AtomicUsize>,
}

impl CountingStorage {
    pub fn wrap(inner: Arc<dyn StorageProvider>) -> (Arc<dyn StorageProvider>, OpannColdCounters) {
        let counters = OpannColdCounters::new();
        let wrapped = Arc::new(Self::new(
            inner,
            Arc::clone(&counters.fetches),
            Arc::clone(&counters.tombstone_fetches),
            Arc::clone(&counters.waves),
            Arc::clone(&counters.in_flight),
        ));
        (wrapped, counters)
    }

    pub fn new(
        inner: Arc<dyn StorageProvider>,
        fetches: Arc<AtomicUsize>,
        tombstone_fetches: Arc<AtomicUsize>,
        waves: Arc<AtomicUsize>,
        in_flight: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            inner,
            fetches,
            tombstone_fetches,
            waves,
            in_flight,
        }
    }

    fn begin_fetch(&self, uri: &str) {
        self.fetches.fetch_add(1, Ordering::Relaxed);
        if uri.ends_with(".tombstones") {
            self.tombstone_fetches.fetch_add(1, Ordering::Relaxed);
        }
        // New fetch wave when this GET starts and nothing else is in flight —
        // the first GET of a concurrent tokio batch. Later GETs in the same
        // batch overlap (in_flight > 0) and stay in the same wave.
        let prev = self.in_flight.fetch_add(1, Ordering::AcqRel);
        if prev == 0 {
            self.waves.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn end_fetch(&self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.begin_fetch(uri);
        let result = self.inner.get(uri).await;
        self.end_fetch();
        result
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.begin_fetch(uri);
        let result = self.inner.get_range(uri, range).await;
        self.end_fetch();
        result
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        self.begin_fetch(uri);
        let result = self.inner.tail(uri, len).await;
        self.end_fetch();
        result
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

    async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
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

    fn prefix_inner(&self) -> Option<Arc<dyn StorageProvider>> {
        self.inner
            .prefix_inner()
            .map(|inner| {
                Arc::new(CountingStorage::new(
                    inner,
                    Arc::clone(&self.fetches),
                    Arc::clone(&self.tombstone_fetches),
                    Arc::clone(&self.waves),
                    Arc::clone(&self.in_flight),
                )) as Arc<dyn StorageProvider>
            })
            .or_else(|| Some(Arc::clone(&self.inner) as Arc<dyn StorageProvider>))
    }
}
