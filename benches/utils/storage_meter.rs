// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Counts object-store requests (HEAD / GET / PUT, including multipart
//! parts) and byte volumes during a bench window. The cost model prices
//! each lifecycle phase (ingest, drain, cold open, per-query fetch) from
//! these measured counts — never from estimates.
//!
//! Reads issued through `StorageProvider::object_store_handle` are wrapped
//! in [`CountingObjectStore`] so parquet range GETs (row materialization,
//! cold `_id` resolution) land in the same meter as provider-level reads.

use std::{
    fmt,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use infino::storage::{ObjectMeta, StorageError, StorageProvider, io_counters::io_is_background};
use object_store::{
    CopyOptions, GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta as OsObjectMeta,
    ObjectStore, PutMultipartOptions, PutOptions, PutPayload, PutResult,
    Result as ObjectStoreResult, UploadPart, path::Path as ObjPath,
};

use crate::rss::fmt_bytes;

/// Path token of the hidden vector-index sibling's storage prefix
/// (`_infino_<uuid>_vector_index/...` under the table root). Requests whose
/// URI carries it belong to the hidden table; everything else is the user
/// table.
const HIDDEN_INDEX_PATH_TOKEN: &str = "_vector_index";
/// Path tokens of the manifest namespace on either table: the pointer
/// (`_supertable/current`), the list dir, the parts dir, and the slow-CAS
/// state prefix (routing blob + centroid section — manifest-published
/// routing state, fetched once per generation, never per-query data).
/// Matches `POINTER_PATH`, `MANIFEST_DIR`, `MANIFEST_PARTS_DIR`, and
/// `slow_vector_state::STORAGE_PREFIX` in `supertable`.
const MANIFEST_PATH_TOKENS: [&str; 4] = [
    "_supertable/",
    "manifest/",
    "manifest-parts/",
    "slow-vector-state/",
];

/// Which table + namespace a request URI belongs to. GETs are attributed
/// per class so a query's fan can be split into user vs hidden traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UriClass {
    UserData,
    UserManifest,
    HiddenData,
    HiddenManifest,
}

/// Number of [`UriClass`] variants (array-indexed counters).
pub const N_URI_CLASSES: usize = 4;

impl UriClass {
    pub fn of(uri: &str) -> Self {
        let hidden = uri.contains(HIDDEN_INDEX_PATH_TOKEN);
        let manifest = MANIFEST_PATH_TOKENS.iter().any(|t| uri.contains(t));
        match (hidden, manifest) {
            (true, true) => Self::HiddenManifest,
            (true, false) => Self::HiddenData,
            (false, true) => Self::UserManifest,
            (false, false) => Self::UserData,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::UserData => "user data",
            Self::UserManifest => "user manifest",
            Self::HiddenData => "hidden data",
            Self::HiddenManifest => "hidden manifest",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::UserData => 0,
            Self::UserManifest => 1,
            Self::HiddenData => 2,
            Self::HiddenManifest => 3,
        }
    }

    fn from_index(i: usize) -> Self {
        match i {
            0 => Self::UserData,
            1 => Self::UserManifest,
            2 => Self::HiddenData,
            _ => Self::HiddenManifest,
        }
    }
}

/// Per-[`UriClass`] GET counters inside one metering window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassIo {
    pub get_count: u64,
    pub get_bytes: u64,
}

/// One traced read request (GET / range GET / tail), captured while a
/// trace window is active — the per-request evidence behind a fan count.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub uri: String,
    /// Byte range for `get_range`; `None` for whole-object GET / tail.
    pub range: Option<(u64, u64)>,
    pub bytes: u64,
}

/// Request + byte counts observed in one metering window.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ObjectStoreMeter {
    pub head_count: u64,
    /// Foreground GETs (query-critical lazy/probe reads).
    pub get_count: u64,
    pub get_bytes: u64,
    /// Background cache-fill GETs (lazy→mmap promotion), tagged via
    /// [`infino::storage::io_counters::scope_background`]. Priced on a
    /// separate ledger line so they do not inflate per-query cold GETs.
    pub bg_get_count: u64,
    pub bg_get_bytes: u64,
    pub put_count: u64,
    pub put_bytes: u64,
    /// LIST requests (billed at the PUT/list rate on S3).
    pub list_count: u64,
    /// DELETE requests (free on S3; counted for completeness, not priced).
    pub delete_count: u64,
    /// GET counts attributed per URI class (indexed by `UriClass::index`).
    pub get_by_class: [ClassIo; N_URI_CLASSES],
}

impl ObjectStoreMeter {
    /// Counts accumulated since an `earlier` snapshot of the same meter —
    /// the per-phase delta the cost model prices.
    pub fn since(&self, earlier: &ObjectStoreMeter) -> ObjectStoreMeter {
        let mut get_by_class = [ClassIo::default(); N_URI_CLASSES];
        for (i, slot) in get_by_class.iter_mut().enumerate() {
            slot.get_count = self.get_by_class[i]
                .get_count
                .saturating_sub(earlier.get_by_class[i].get_count);
            slot.get_bytes = self.get_by_class[i]
                .get_bytes
                .saturating_sub(earlier.get_by_class[i].get_bytes);
        }
        ObjectStoreMeter {
            head_count: self.head_count.saturating_sub(earlier.head_count),
            get_count: self.get_count.saturating_sub(earlier.get_count),
            get_bytes: self.get_bytes.saturating_sub(earlier.get_bytes),
            bg_get_count: self.bg_get_count.saturating_sub(earlier.bg_get_count),
            bg_get_bytes: self.bg_get_bytes.saturating_sub(earlier.bg_get_bytes),
            put_count: self.put_count.saturating_sub(earlier.put_count),
            put_bytes: self.put_bytes.saturating_sub(earlier.put_bytes),
            list_count: self.list_count.saturating_sub(earlier.list_count),
            delete_count: self.delete_count.saturating_sub(earlier.delete_count),
            get_by_class,
        }
    }

    /// View of this window's background-fill GETs as a foreground-shaped
    /// meter so existing request/cost formatters can price the fill line.
    pub fn background_fill_meter(&self) -> ObjectStoreMeter {
        ObjectStoreMeter {
            get_count: self.bg_get_count,
            get_bytes: self.bg_get_bytes,
            ..Default::default()
        }
    }

    /// Merge background-fill GETs from two windows (e.g. cold + repeat).
    pub fn merge_background_fill(&self, other: &ObjectStoreMeter) -> ObjectStoreMeter {
        ObjectStoreMeter {
            get_count: self.bg_get_count.saturating_add(other.bg_get_count),
            get_bytes: self.bg_get_bytes.saturating_add(other.bg_get_bytes),
            ..Default::default()
        }
    }

    /// Read-class requests (HEAD + GET) — billed at the GET rate.
    pub fn read_requests(&self) -> u64 {
        self.head_count + self.get_count
    }

    /// GET count/bytes for one table + namespace class.
    pub fn class_io(&self, class: UriClass) -> ClassIo {
        self.get_by_class[class.index()]
    }

    /// "user data 40 GET (30.1 MiB) · hidden data 24 GET (6.9 MiB)" —
    /// non-zero classes only; "0 GET" when the window saw none.
    pub fn fmt_get_class_breakdown(&self) -> String {
        let parts: Vec<String> = self
            .get_by_class
            .iter()
            .enumerate()
            .filter(|(_, c)| c.get_count > 0)
            .map(|(i, c)| {
                format!(
                    "{} {} GET ({})",
                    UriClass::from_index(i).label(),
                    c.get_count,
                    fmt_bytes(c.get_bytes),
                )
            })
            .collect();
        if parts.is_empty() {
            "0 GET".into()
        } else {
            parts.join(" · ")
        }
    }
}

/// One cold consumer's metered windows, split at the phase boundaries the
/// cost model prices separately: the one-time table open; the first query
/// on the cold cache (the one-time metadata warmup — under the v1 open
/// discipline it hydrates the admit-window centroid regions, Sq8 meta,
/// and stable-id blocks alongside its probe); a **second, distinct**
/// query on the same consumer (the steady cold per-query fetch — the
/// warmup blocks are resident, so it pays only its own probe and any
/// newly-touched cells); and the first query repeated verbatim — a cache
/// fill-lag probe, *not* a steady-state warm number (steady-state warm is
/// metered separately on a cache-hot consumer).
#[derive(Debug, Clone, Copy)]
pub struct ColdStoreSplit {
    pub open: ObjectStoreMeter,
    pub first_query: ObjectStoreMeter,
    pub second_query: ObjectStoreMeter,
    pub repeat_query: ObjectStoreMeter,
}

#[derive(Default)]
struct MeterCounters {
    head_count: AtomicU64,
    get_count: AtomicU64,
    get_bytes: AtomicU64,
    bg_get_count: AtomicU64,
    bg_get_bytes: AtomicU64,
    put_count: AtomicU64,
    put_bytes: AtomicU64,
    list_count: AtomicU64,
    delete_count: AtomicU64,
    /// Per-[`UriClass`] GET request counts (indexed by `UriClass::index`).
    class_get_count: [AtomicU64; N_URI_CLASSES],
    /// Per-[`UriClass`] GET byte volumes (indexed by `UriClass::index`).
    class_get_bytes: [AtomicU64; N_URI_CLASSES],
    /// Read-request trace window: `Some(entries)` while a trace is active
    /// (every GET / range GET / tail appends), `None` otherwise. PUTs are
    /// never traced — the trace exists to explain a query's GET fan.
    trace: Mutex<Option<Vec<TraceEntry>>>,
}

impl MeterCounters {
    fn snapshot(&self) -> ObjectStoreMeter {
        let mut get_by_class = [ClassIo::default(); N_URI_CLASSES];
        for (i, slot) in get_by_class.iter_mut().enumerate() {
            slot.get_count = self.class_get_count[i].load(Ordering::Relaxed);
            slot.get_bytes = self.class_get_bytes[i].load(Ordering::Relaxed);
        }
        ObjectStoreMeter {
            head_count: self.head_count.load(Ordering::Relaxed),
            get_count: self.get_count.load(Ordering::Relaxed),
            get_bytes: self.get_bytes.load(Ordering::Relaxed),
            bg_get_count: self.bg_get_count.load(Ordering::Relaxed),
            bg_get_bytes: self.bg_get_bytes.load(Ordering::Relaxed),
            put_count: self.put_count.load(Ordering::Relaxed),
            put_bytes: self.put_bytes.load(Ordering::Relaxed),
            list_count: self.list_count.load(Ordering::Relaxed),
            delete_count: self.delete_count.load(Ordering::Relaxed),
            get_by_class,
        }
    }

    fn record_get(&self, uri: &str, range: Option<(u64, u64)>, bytes: u64) {
        // Background cache-fill ranges run under `io_counters::scope_background`
        // and land in `bg_get_*` so they do not inflate per-query cold GETs,
        // but still appear on a dedicated fill ledger line.
        if io_is_background() {
            self.bg_get_count.fetch_add(1, Ordering::Relaxed);
            self.bg_get_bytes.fetch_add(bytes, Ordering::Relaxed);
            return;
        }
        self.get_count.fetch_add(1, Ordering::Relaxed);
        self.get_bytes.fetch_add(bytes, Ordering::Relaxed);
        let class = UriClass::of(uri).index();
        self.class_get_count[class].fetch_add(1, Ordering::Relaxed);
        self.class_get_bytes[class].fetch_add(bytes, Ordering::Relaxed);
        let mut trace = self.trace.lock().expect("trace mutex poisoned");
        if let Some(entries) = trace.as_mut() {
            entries.push(TraceEntry {
                uri: uri.to_string(),
                range,
                bytes,
            });
        }
    }

    fn record_put(&self, bytes: u64) {
        self.put_count.fetch_add(1, Ordering::Relaxed);
        self.put_bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

/// Storage provider wrapper that meters request counts and byte volumes.
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

impl fmt::Debug for CountingStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingStorage").finish_non_exhaustive()
    }
}

pub fn wrap(storage: Arc<dyn StorageProvider>) -> MeteredStorage {
    let counters = Arc::new(MeterCounters::default());
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

    /// Start capturing every read request (GET / range GET / tail) into a
    /// trace window. Any previously captured, un-taken entries are dropped.
    pub fn start_trace(&self) {
        *self.counters.trace.lock().expect("trace mutex poisoned") = Some(Vec::new());
    }

    /// Stop tracing and return the captured entries in request order.
    pub fn take_trace(&self) -> Vec<TraceEntry> {
        self.counters
            .trace
            .lock()
            .expect("trace mutex poisoned")
            .take()
            .unwrap_or_default()
    }
}

/// Multipart-upload wrapper: each part is one billable PUT, and the
/// completion call is one more (matches S3 `UploadPart` +
/// `CompleteMultipartUpload` billing; the create call is counted by
/// [`CountingStorage::put_multipart`]). Aborts are failure-path cleanup
/// and are left uncounted.
struct CountingUpload {
    inner: Box<dyn MultipartUpload>,
    counters: Arc<MeterCounters>,
}

impl fmt::Debug for CountingUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingUpload").finish_non_exhaustive()
    }
}

#[async_trait]
impl MultipartUpload for CountingUpload {
    fn put_part(&mut self, data: PutPayload) -> UploadPart {
        self.counters.record_put(data.content_length() as u64);
        self.inner.put_part(data)
    }

    async fn complete(&mut self) -> ObjectStoreResult<PutResult> {
        self.counters.record_put(0);
        self.inner.complete().await
    }

    async fn abort(&mut self) -> ObjectStoreResult<()> {
        self.inner.abort().await
    }
}

#[async_trait]
impl StorageProvider for CountingStorage {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.counters.head_count.fetch_add(1, Ordering::Relaxed);
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        let (bytes, meta) = self.inner.get(uri).await?;
        self.counters.record_get(uri, None, bytes.len() as u64);
        Ok((bytes, meta))
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        let requested = (range.start, range.end);
        let bytes = self.inner.get_range(uri, range).await?;
        self.counters
            .record_get(uri, Some(requested), bytes.len() as u64);
        Ok(bytes)
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let (bytes, size) = self.inner.tail(uri, len).await?;
        self.counters.record_get(uri, None, bytes.len() as u64);
        Ok((bytes, size))
    }

    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.counters.record_put(bytes.len() as u64);
        self.inner.put_atomic(uri, bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.counters.record_put(bytes.len() as u64);
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        // The create call itself is a billable request.
        self.counters.record_put(0);
        let inner = self.inner.put_multipart(uri).await?;
        Ok(Box::new(CountingUpload {
            inner,
            counters: Arc::clone(&self.counters),
        }))
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.counters.delete_count.fetch_add(1, Ordering::Relaxed);
        self.inner.delete(uri).await
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        self.counters.list_count.fetch_add(1, Ordering::Relaxed);
        self.inner.list_with_prefix(prefix).await
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.counters.list_count.fetch_add(1, Ordering::Relaxed);
        self.inner.list_with_prefix_metadata(prefix).await
    }

    fn object_store_handle(&self, uri: &str) -> Option<(Arc<dyn ObjectStore>, ObjPath)> {
        let (inner, path) = self.inner.object_store_handle(uri)?;
        // Wrap the raw store so parquet range reads issued straight against the
        // object-store handle (row materialization via `take_rows_object_store`,
        // cold `_id` resolution) are metered too. Without this, those GETs
        // bypass the meter entirely — a real per-query blind spot.
        let counted: Arc<dyn ObjectStore> = Arc::new(CountingObjectStore {
            inner,
            counters: Arc::clone(&self.counters),
        });
        Some((counted, path))
    }
}

/// Counting wrapper around a raw [`ObjectStore`] handed out by
/// [`CountingStorage::object_store_handle`]. Only the read surface is
/// metered (`get_opts` / `get_ranges`); every other method delegates
/// unchanged. Reads land in the *same* [`MeterCounters`] as the
/// `StorageProvider`-level reads, so a query's full GET fan is captured
/// whether it flows through the provider or the parquet object-store path.
struct CountingObjectStore {
    inner: Arc<dyn ObjectStore>,
    counters: Arc<MeterCounters>,
}

impl fmt::Debug for CountingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountingObjectStore")
            .finish_non_exhaustive()
    }
}

impl fmt::Display for CountingObjectStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CountingObjectStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for CountingObjectStore {
    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> ObjectStoreResult<GetResult> {
        let is_head = options.head;
        let res = self.inner.get_opts(location, options).await?;
        if is_head {
            self.counters.head_count.fetch_add(1, Ordering::Relaxed);
        } else {
            // The resolved `range` gives the byte count without consuming the
            // (streamed) payload.
            let len = res.range.end.saturating_sub(res.range.start);
            self.counters.record_get(
                location.as_ref(),
                Some((res.range.start, res.range.end)),
                len,
            );
        }
        Ok(res)
    }

    async fn get_ranges(
        &self,
        location: &ObjPath,
        ranges: &[std::ops::Range<u64>],
    ) -> ObjectStoreResult<Vec<Bytes>> {
        // Delegate so the inner store's range coalescing is preserved, then
        // count one GET per requested range with its returned byte length.
        let out = self.inner.get_ranges(location, ranges).await?;
        for (r, b) in ranges.iter().zip(&out) {
            self.counters
                .record_get(location.as_ref(), Some((r.start, r.end)), b.len() as u64);
        }
        Ok(out)
    }

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> ObjectStoreResult<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> ObjectStoreResult<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    fn delete_stream(
        &self,
        locations: BoxStream<'static, ObjectStoreResult<ObjPath>>,
    ) -> BoxStream<'static, ObjectStoreResult<ObjPath>> {
        self.inner.delete_stream(locations)
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, ObjectStoreResult<OsObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&ObjPath>) -> ObjectStoreResult<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy_opts(
        &self,
        from: &ObjPath,
        to: &ObjPath,
        options: CopyOptions,
    ) -> ObjectStoreResult<()> {
        self.inner.copy_opts(from, to, options).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_subtracts_fieldwise_and_saturates() {
        let mut earlier = ObjectStoreMeter {
            head_count: 1,
            get_count: 10,
            get_bytes: 100,
            bg_get_count: 2,
            bg_get_bytes: 20,
            put_count: 5,
            put_bytes: 50,
            ..Default::default()
        };
        earlier.get_by_class[UriClass::HiddenData.index()] = ClassIo {
            get_count: 4,
            get_bytes: 40,
        };
        let mut later = ObjectStoreMeter {
            head_count: 1,
            get_count: 25,
            get_bytes: 400,
            bg_get_count: 5,
            bg_get_bytes: 80,
            put_count: 9,
            put_bytes: 90,
            ..Default::default()
        };
        later.get_by_class[UriClass::HiddenData.index()] = ClassIo {
            get_count: 9,
            get_bytes: 140,
        };
        let delta = later.since(&earlier);
        assert_eq!(delta.head_count, 0);
        assert_eq!(delta.get_count, 15);
        assert_eq!(delta.get_bytes, 300);
        assert_eq!(delta.bg_get_count, 3);
        assert_eq!(delta.bg_get_bytes, 60);
        assert_eq!(delta.put_count, 4);
        assert_eq!(delta.put_bytes, 40);
        assert_eq!(
            delta.get_by_class[UriClass::HiddenData.index()],
            ClassIo {
                get_count: 5,
                get_bytes: 100,
            }
        );
        // Windows never run backwards; saturate instead of wrapping if a
        // caller ever crosses snapshots.
        assert_eq!(earlier.since(&later).get_count, 0);
    }

    #[test]
    fn uri_class_covers_both_tables_and_namespaces() {
        // User table: superfile data + the three manifest dirs.
        assert_eq!(UriClass::of("superfiles/ab12.parquet"), UriClass::UserData);
        assert_eq!(UriClass::of("_supertable/current"), UriClass::UserManifest);
        assert_eq!(
            UriClass::of("manifest/manifest-1.avro.zst"),
            UriClass::UserManifest
        );
        assert_eq!(
            UriClass::of("manifest-parts/part-ab.avro.zst"),
            UriClass::UserManifest
        );
        // Hidden table: same shapes under the hidden prefix.
        let hidden = "_infino_0000-uuid_vector_index/";
        assert_eq!(
            UriClass::of(&format!("{hidden}superfiles/cd34.parquet")),
            UriClass::HiddenData
        );
        assert_eq!(
            UriClass::of(&format!("{hidden}_supertable/current")),
            UriClass::HiddenManifest
        );
        assert_eq!(
            UriClass::of(&format!("{hidden}manifest-parts/part-cd.avro.zst")),
            UriClass::HiddenManifest
        );
    }

    #[test]
    fn read_requests_sums_head_and_get() {
        let m = ObjectStoreMeter {
            head_count: 2,
            get_count: 3,
            ..Default::default()
        };
        assert_eq!(m.read_requests(), 5);
    }

    /// LIST and DELETE at the provider level are counted (LIST is billed; the
    /// count feeds the detailed I/O ledger either way).
    #[tokio::test]
    async fn list_and_delete_are_counted() {
        use infino::storage::LocalFsStorageProvider;

        let dir = tempfile::TempDir::new().expect("tempdir");
        let provider: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let meter = wrap(provider);
        let p = meter.provider();
        p.put_atomic("seg/a.bin", Bytes::from_static(b"a"))
            .await
            .expect("put");
        let before = meter.snapshot();
        let _ = p.list_with_prefix("seg").await.expect("list");
        p.delete("seg/a.bin").await.expect("delete");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.list_count, 1);
        assert_eq!(delta.delete_count, 1);
    }

    /// Regression: a read issued straight against the raw store from
    /// `object_store_handle` (the parquet row-materialization / cold `_id`
    /// path) must be metered, not silently bypass the wrapper.
    #[tokio::test]
    async fn object_store_handle_reads_are_metered() {
        use infino::storage::LocalFsStorageProvider;
        use object_store::ObjectStoreExt;

        let dir = tempfile::TempDir::new().expect("tempdir");
        let provider: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        provider
            .put_atomic("seg/x.bin", Bytes::from_static(b"0123456789"))
            .await
            .expect("put");

        let meter = wrap(provider);
        let before = meter.snapshot();
        let (store, path) = meter
            .provider()
            .object_store_handle("seg/x.bin")
            .expect("handle");
        let bytes = store.get_range(&path, 2..5).await.expect("range");
        assert_eq!(&bytes[..], b"234");

        let delta = meter.snapshot().since(&before);
        assert_eq!(
            delta.get_count, 1,
            "a read via object_store_handle must be metered, not bypassed"
        );
        assert_eq!(delta.get_bytes, 3);
    }
}
