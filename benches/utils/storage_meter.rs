// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Counts object-store `head`, `tail`, and `get_range` calls during a bench window.
//! Used by the cost model to price cold-query S3 requests.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;
use infino::storage::{ObjectMeta, StorageError, StorageProvider};

/// One cold open + search iteration's object-store footprint.
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectStoreMeter {
    pub head_count: u64,
    pub get_count: u64,
    pub get_bytes: u64,
}

struct MeterCounters {
    head_count: AtomicU64,
    get_count: AtomicU64,
    get_bytes: AtomicU64,
}

impl MeterCounters {
    fn snapshot(&self) -> ObjectStoreMeter {
        ObjectStoreMeter {
            head_count: self.head_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            get_bytes: self.get_bytes.load(Ordering::Relaxed),
        }
    }

    fn record_get(&self, bytes: u64) {
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Storage provider wrapper that meters read-path requests.
pub struct MeteredStorage {
    provider: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
}

struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
}

impl CountingStorage {
    fn new(inner: Arc<dyn StorageProvider>, counters: Arc<MeterCounters>) -> Self {
        Self { inner, counters }
    }
}

impl std::fmt::Debug for CountingStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingStorage").finish_non_exhaustive()
    }
}

pub fn wrap(storage: Arc<dyn StorageProvider>) -> MeteredStorage {
    let counters = Arc::new(MeterCounters {
        head_count: AtomicU64::new(0),
        get_count: AtomicU64::new(0),
        get_bytes: AtomicU64::new(0),
    });
    let provider: Arc<dyn StorageProvider> =
        Arc::new(CountingStorage::new(storage, Arc::clone(&counters)));
    MeteredStorage { provider, counters }
}

impl MeteredStorage {
    pub fn provider(&self) -> Arc<dyn StorageProvider> {
        Arc::clone(&self.provider)
    }

    pub fn snapshot(&self) -> ObjectStoreMeter {
        self.counters.snapshot()
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.counters.head_count.fetch_add(1, Ordering::Relaxed);
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let bytes = self.inner.get_range(uri, range).await?;
        self.counters.record_get(bytes.len() as u64);
        Ok(bytes)
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let (bytes, size) = self.inner.tail(uri, len).await?;
        self.counters.record_get(bytes.len() as u64);
        Ok((bytes, size))
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

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.inner.list_with_prefix(prefix).await
    }

    fn object_store_handle(
        &self,
        uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        self.inner.object_store_handle(uri)
    }
}
