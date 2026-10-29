// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A [`StorageProvider`] wrapper whose reads take a while, and that counts
//! them.
//!
//! Each `get` and `get_range` sleeps before reaching the inner provider, so a
//! test sees reads issued together overlap, and can act while a read is in
//! flight: cancel a query mid-open, say, then check that no read starts after
//! it returned. [`SlowStorage::hold_reads`] parks every read until
//! [`SlowStorage::release_reads`], for a test that must act before a read
//! finishes, however slow the runner. `tail` and `get_if_none_match` are the trait's defaults built
//! from those, so a `tail` here is a HEAD plus a range read, whatever the
//! inner provider does natively. Writes, listing and deletes pass through
//! untouched.

use std::{
    collections::HashMap,
    fmt::{Debug, Formatter, Result as FmtResult},
    mem,
    ops::Range,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use object_store::MultipartUpload;
use tokio::{sync::watch, time::sleep};

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
    /// Each read's start and finish, in order, by the object it read.
    read_log: Mutex<Vec<ReadEvent>>,
    /// Whether reads are parked before they reach the inner provider.
    held: watch::Sender<bool>,
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
            read_log: Mutex::new(Vec::new()),
            held: watch::Sender::new(false),
        })
    }

    /// Park reads, those in flight and those to come, until
    /// [`Self::release_reads`]. A parked read still counts as started.
    pub fn hold_reads(&self) {
        self.held.send_replace(true);
    }

    /// Let parked reads go on.
    pub fn release_reads(&self) {
        self.held.send_replace(false);
    }

    /// Reads started so far, finished or not.
    pub fn reads(&self) -> usize {
        self.reads.load(Ordering::SeqCst)
    }

    /// Reads started so far on an object whose path contains `fragment`.
    pub fn reads_of(&self, fragment: &str) -> usize {
        self.log()
            .iter()
            .filter(|event| event.started && event.uri.contains(fragment))
            .count()
    }

    /// The most objects whose path contains `fragment` that ever had a read
    /// in flight at once, however many reads each had.
    pub fn max_objects_in_flight_of(&self, fragment: &str) -> usize {
        let mut reading: HashMap<&str, usize> = HashMap::new();
        let mut most = 0;
        for event in self.log().iter().filter(|e| e.uri.contains(fragment)) {
            let reads = reading.entry(&event.uri).or_default();
            if event.started {
                *reads += 1;
            } else {
                *reads -= 1;
                if *reads == 0 {
                    reading.remove(event.uri.as_str());
                }
            }
            most = most.max(reading.len());
        }
        most
    }

    fn log(&self) -> MutexGuard<'_, Vec<ReadEvent>> {
        self.read_log.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Reads started and not yet finished, counting one abandoned midway.
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// The most reads ever in flight at once.
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }

    /// What a read waits for: the delay, then any hold.
    async fn wait(&self) {
        sleep(self.delay).await;
        let _ = self.held.subscribe().wait_for(|held| !held).await;
    }

    /// Count a read in for as long as the returned guard lives, so a read
    /// whose future is dropped midway still counts out.
    fn start_read(&self, uri: &str) -> InFlight<'_> {
        self.log().push(ReadEvent {
            uri: uri.to_owned(),
            started: true,
        });
        self.reads.fetch_add(1, Ordering::SeqCst);
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        InFlight {
            storage: self,
            uri: uri.to_owned(),
        }
    }
}

/// A read starting or finishing on `uri`.
struct ReadEvent {
    uri: String,
    started: bool,
}

/// One read in flight; see [`SlowStorage::start_read`].
struct InFlight<'a> {
    storage: &'a SlowStorage,
    uri: String,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.storage.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.storage.log().push(ReadEvent {
            uri: mem::take(&mut self.uri),
            started: false,
        });
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
        self.wait().await;
        self.inner.get(uri).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let _read = self.start_read(uri);
        self.wait().await;
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
