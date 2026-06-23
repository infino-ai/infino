// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Counts object-store request traffic during a bench window: the read path
//! (`head` / `tail` / `get_range`) and the write path (`put_atomic` /
//! `put_if_match`, multipart `put_part` bytes + completion, and `delete`).
//! Used by the cost model to price cold-query S3 requests and the
//! write-dominated drain / optimize maintenance passes.

use std::{
    collections::HashMap,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    get_meter,
    storage::{ObjectMeta, StorageError, StorageProvider},
};
use object_store::{MultipartUpload, PutPayload, PutResult, UploadPart};

/// One bench window's object-store footprint (read + write requests + bytes).
#[derive(Debug, Clone, Copy, Default)]
pub struct ObjectStoreMeter {
    pub head_count: u64,
    pub get_count: u64,
    pub get_bytes: u64,
    /// Successful object writes — `put_atomic` / `put_if_match` calls plus each
    /// completed multipart upload (counted once on `complete`).
    pub put_count: u64,
    /// Total bytes written, summed over single PUTs and multipart part uploads.
    pub put_bytes: u64,
    pub delete_count: u64,
}

/// Per-phase GET wave summary for one bench window.
#[derive(Debug, Clone, Default)]
pub struct RangeWaveSummary {
    /// Parallel GET batches (overlapping request intervals count as one wave).
    pub wave_batches: usize,
    /// GET count attributed to each tagged phase (`get_meter` phase ids).
    pub by_phase: Vec<(u8, u64, usize)>,
}

impl ObjectStoreMeter {
    /// Field-wise difference `self - earlier` — the traffic accrued between two
    /// snapshots, used to attribute cost to one bench phase (drain, optimize).
    pub fn delta(self, earlier: ObjectStoreMeter) -> ObjectStoreMeter {
        ObjectStoreMeter {
            head_count: self.head_count.saturating_sub(earlier.head_count),
            get_count: self.get_count.saturating_sub(earlier.get_count),
            get_bytes: self.get_bytes.saturating_sub(earlier.get_bytes),
            put_count: self.put_count.saturating_sub(earlier.put_count),
            put_bytes: self.put_bytes.saturating_sub(earlier.put_bytes),
            delete_count: self.delete_count.saturating_sub(earlier.delete_count),
        }
    }
}

#[derive(Debug, Clone)]
struct RangeGetEvent {
    start_us: u128,
    end_us: u128,
    phase: u8,
}

#[derive(Default)]
struct MeterCounters {
    head_count: AtomicU64,
    get_count: AtomicU64,
    get_bytes: AtomicU64,
    put_count: AtomicU64,
    put_bytes: AtomicU64,
    delete_count: AtomicU64,
    range_log: Mutex<Vec<RangeGetEvent>>,
}

impl MeterCounters {
    fn snapshot(&self) -> ObjectStoreMeter {
        ObjectStoreMeter {
            head_count: self.head_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            get_bytes: self.get_bytes.load(Ordering::Relaxed),
            put_count: self.put_count.load(Ordering::Relaxed),
            put_bytes: self.put_bytes.load(Ordering::Relaxed),
            delete_count: self.delete_count.load(Ordering::Relaxed),
        }
    }

    fn record_get(&self, origin: Instant, start: Instant, end: Instant, bytes: u64, phase: u8) {
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.range_log.lock().unwrap().push(RangeGetEvent {
            start_us: start.duration_since(origin).as_micros(),
            end_us: end.duration_since(origin).as_micros(),
            phase,
        });
    }

    fn range_wave_summary(&self) -> RangeWaveSummary {
        let log = self.range_log.lock().unwrap();
        if log.is_empty() {
            return RangeWaveSummary::default();
        }
        let wave_batches = count_get_waves(&log);
        let mut by_phase: Vec<(u8, u64, usize)> = Vec::new();
        for events in events_by_phase(&log) {
            let phase = events[0].phase;
            by_phase.push((phase, events.len() as u64, count_get_waves(&events)));
        }
        by_phase.sort_by_key(|(phase, _, _)| *phase);
        RangeWaveSummary {
            wave_batches,
            by_phase,
        }
    }

    /// Record a single-PUT write (one request + its byte payload).
    fn record_put(&self, bytes: u64) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Count parallel GET **waves**: requests whose intervals overlap belong to
/// the same wave (same definition as `unified_object_store` cold-path diag).
fn count_get_waves(events: &[RangeGetEvent]) -> usize {
    if events.is_empty() {
        return 0;
    }
    let mut sorted: Vec<&RangeGetEvent> = events.iter().collect();
    sorted.sort_unstable_by_key(|e| (e.start_us, e.end_us));
    let mut batches = 1usize;
    let mut batch_end = sorted[0].end_us;
    for event in sorted.iter().skip(1) {
        if event.start_us <= batch_end {
            batch_end = batch_end.max(event.end_us);
        } else {
            batches += 1;
            batch_end = event.end_us;
        }
    }
    batches
}

fn events_by_phase(log: &[RangeGetEvent]) -> Vec<Vec<RangeGetEvent>> {
    let mut out: HashMap<u8, Vec<RangeGetEvent>> = HashMap::new();
    for event in log {
        out.entry(event.phase).or_default().push(event.clone());
    }
    let mut keys: Vec<u8> = out.keys().copied().collect();
    keys.sort_unstable();
    keys.into_iter()
        .filter_map(|k| out.remove(&k))
        .collect()
}

/// Storage provider wrapper that meters read- and write-path requests.
pub struct MeteredStorage {
    provider: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
}

struct CountingStorage {
    inner: Arc<dyn StorageProvider>,
    counters: Arc<MeterCounters>,
    origin: Instant,
}

impl CountingStorage {
    fn new(
        inner: Arc<dyn StorageProvider>,
        counters: Arc<MeterCounters>,
        origin: Instant,
    ) -> Self {
        Self {
            inner,
            counters,
            origin,
        }
    }
}

impl std::fmt::Debug for CountingStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingStorage").finish_non_exhaustive()
    }
}

pub fn wrap(storage: Arc<dyn StorageProvider>) -> MeteredStorage {
    let counters = Arc::new(MeterCounters::default());
    let origin = Instant::now();
    let provider: Arc<dyn StorageProvider> = Arc::new(CountingStorage::new(
        storage,
        Arc::clone(&counters),
        origin,
    ));
    MeteredStorage { provider, counters }
}

impl MeteredStorage {
    pub fn provider(&self) -> Arc<dyn StorageProvider> {
        Arc::clone(&self.provider)
    }

    pub fn snapshot(&self) -> ObjectStoreMeter {
        self.counters.snapshot()
    }

    /// GET wave batches and per-phase breakdown for the current window.
    pub fn range_wave_summary(&self) -> RangeWaveSummary {
        self.counters.range_wave_summary()
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.counters.head_count.fetch_add(1, Ordering::Relaxed);
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let start = Instant::now();
        let phase = get_meter::get_phase();
        let out = self.inner.get(uri).await?;
        let end = Instant::now();
        self.counters.record_get(
            self.origin,
            start,
            end,
            out.0.len() as u64,
            phase,
        );
        Ok(out)
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let start = Instant::now();
        let phase = get_meter::get_phase();
        let bytes = self.inner.get_range(uri, range).await?;
        let end = Instant::now();
        self.counters.record_get(
            self.origin,
            start,
            end,
            bytes.len() as u64,
            phase,
        );
        Ok(bytes)
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let start = Instant::now();
        let phase = get_meter::get_phase();
        let (bytes, size) = self.inner.tail(uri, len).await?;
        let end = Instant::now();
        self.counters.record_get(
            self.origin,
            start,
            end,
            bytes.len() as u64,
            phase,
        );
        Ok((bytes, size))
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        let n = bytes.len() as u64;
        let result = self.inner.put_atomic(uri, bytes).await?;
        self.counters.record_put(n);
        Ok(result)
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let n = bytes.len() as u64;
        let result = self.inner.put_if_match(uri, bytes, expected_etag).await?;
        self.counters.record_put(n);
        Ok(result)
    }

    async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
        // Wrap the upload so part bytes and the completing request are metered;
        // the bytes stream through `put_part`, not this call's arguments.
        let inner = self.inner.put_multipart(uri).await?;
        Ok(Box::new(CountingMultipartUpload {
            inner,
            counters: Arc::clone(&self.counters),
        }))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await?;
        self.counters.delete_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
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

/// Wraps a multipart upload so each part's bytes are tallied as they stream
/// (the part payloads, not the `put_multipart` call, carry the bytes) and the
/// upload is counted as one logical PUT on `complete`.
struct CountingMultipartUpload {
    inner: Box<dyn MultipartUpload>,
    counters: Arc<MeterCounters>,
}

impl std::fmt::Debug for CountingMultipartUpload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CountingMultipartUpload")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl MultipartUpload for CountingMultipartUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.counters
            .put_bytes
            .fetch_add(data.content_length() as u64, Ordering::Relaxed);
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> object_store::Result<PutResult> {
        // Count the multipart upload as one PUT request once it commits.
        let result = self.inner.complete().await?;
        self.counters.put_count.fetch_add(1, Ordering::Relaxed);
        Ok(result)
    }

    async fn abort(&mut self) -> object_store::Result<()> {
        self.inner.abort().await
    }
}

#[cfg(test)]
mod tests {
    use infino::storage::LocalFsStorageProvider;
    use tempfile::TempDir;

    use super::*;

    /// The write path is metered: single PUTs count their request + bytes,
    /// multipart tallies each part's bytes and counts one PUT on completion,
    /// and deletes are counted.
    #[tokio::test]
    async fn meters_single_puts_multipart_and_deletes() {
        let dir = TempDir::new().expect("tempdir");
        let inner: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let meter = wrap(inner);
        let p = meter.provider();

        p.put_atomic("a.bin", Bytes::from_static(b"hello")) // 5 bytes
            .await
            .expect("put_atomic");
        p.put_if_match("b.bin", Bytes::from_static(b"world!!"), None) // 7 bytes
            .await
            .expect("put_if_match");

        let mut upload = p.put_multipart("c.bin").await.expect("put_multipart");
        upload
            .put_part(PutPayload::from_bytes(Bytes::from_static(b"chunk-0"))) // 7
            .await
            .expect("part 0");
        upload
            .put_part(PutPayload::from_bytes(Bytes::from_static(b"chunk-1!"))) // 8
            .await
            .expect("part 1");
        upload.complete().await.expect("complete");

        p.delete("a.bin").await.expect("delete");

        let m = meter.snapshot();
        assert_eq!(m.put_count, 3, "two single PUTs + one completed multipart");
        assert_eq!(
            m.put_bytes,
            5 + 7 + 7 + 8,
            "single-PUT bytes plus multipart part bytes"
        );
        assert_eq!(m.delete_count, 1);
    }
}
