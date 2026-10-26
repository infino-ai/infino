// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A [`StorageProvider`] wrapper whose reads take a while, and that counts
//! them.
//!
//! Each `get` and `get_range` sleeps before reaching the inner provider, so a
//! test sees reads issued together overlap, and can act while a read is in
//! flight: cancel a query mid-open, say, then check that no read starts after
//! it returned. `tail` and `get_if_none_match` are the trait's defaults built
//! from those, so a `tail` here is a HEAD plus a range read, whatever the
//! inner provider does natively. Writes, listing and deletes pass through
//! untouched.

use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    ops::Range,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::MultipartUpload;
use tokio::time::sleep;

use crate::{
    runtime_metrics::io::UsageMeter,
    storage::{ObjectMeta, StorageError, StorageProvider},
};

/// Slow, counting reads over `inner`.
pub struct SlowStorage {
    inner: Arc<dyn StorageProvider>,
    delay: Duration,
    reads: AtomicUsize,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
    /// The object each read started on, in order.
    read_uris: Mutex<Vec<String>>,
}

impl SlowStorage {
    /// Wrap `inner` so every read waits `delay` first.
    pub fn wrap(inner: Arc<dyn StorageProvider>, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner,
            delay,
            reads: AtomicUsize::new(0),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
            read_uris: Mutex::new(Vec::new()),
        })
    }

    /// Reads started so far, finished or not.
    pub fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    /// Reads started so far on an object whose path contains `fragment`.
    pub fn reads_of(&self, fragment: &str) -> usize {
        self.read_uris
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|uri| uri.contains(fragment))
            .count()
    }

    /// Reads started and not yet finished, counting one abandoned midway.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// The most reads ever in flight at once.
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }

    /// Count a read in for as long as the returned guard lives, so a read
    /// whose future is dropped midway still counts out.
    fn start_read(&self, uri: &str) -> InFlight<'_> {
        self.read_uris
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(uri.to_owned());
        self.reads.fetch_add(1, Ordering::SeqCst);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        InFlight(&self.in_flight)
    }
}

/// One read in flight; see [`SlowStorage::start_read`].
struct InFlight<'a>(&'a AtomicUsize);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl Debug for SlowStorage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("SlowStorage")
            .field("delay", &self.delay)
            .field("reads", &self.reads())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl StorageProvider for SlowStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let _read = self.start_read(uri);
        sleep(self.delay).await;
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let _read = self.start_read(uri);
        sleep(self.delay).await;
        self.inner.get_range(uri, range).await
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_overwrite(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
        self.inner.put_overwrite(uri, bytes).await
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

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.inner.list_with_prefix_metadata(prefix).await
    }

    // No `object_store_handle`: the raw handle would let reads bypass the
    // delay and the count.

    fn usage_meter(&self) -> Arc<UsageMeter> {
        self.inner.usage_meter()
    }
}
