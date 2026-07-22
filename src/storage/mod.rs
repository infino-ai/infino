// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Storage provider abstraction over object stores.
//!
//! Wraps the `object_store` crate with a narrower, supertable-
//! shaped interface exposing only the operations the supertable's
//! manifest + disk-cache layers consume:
//!
//! - `head` / `get` / `get_range` — read paths.
//! - `put_atomic` / `put_if_match` / `put_multipart` — write
//!   paths; `put_atomic` and `put_if_match` are the
//!   conditional-write primitives the manifest's OCC + the
//!   atomic-rename pointer commit ride on.
//! - `delete` — idempotent object removal.
//!
//! ## Retry contract
//!
//! Implementations inherit `object_store`'s internal bounded
//! retry of transient failures (5xx, connection-reset,
//! timeouts) under its `RetryConfig`. The `Result` returned by
//! a `StorageProvider` method therefore represents either a
//! *permanent* failure or a *transient failure that exhausted
//! the provider's retry budget*. Callers do **not** retry
//! transient errors themselves.
//!
//! The single exception is OCC on the manifest pointer:
//! [`StorageError::PreconditionFailed`] is a legitimate
//! contention signal. The supertable's commit loop catches it
//! specifically, re-reads the pointer to capture the winner's
//! state, and retries the commit on top of it.

use std::{fmt, ops::Range, sync::Arc, time::SystemTime};

use async_trait::async_trait;
use bytes::Bytes;
use thiserror::Error;

pub mod azure;
pub(crate) mod counting;
pub mod gcs;
pub mod local_fs;
pub(crate) mod options;
mod retry;
pub mod s3;

pub use azure::AzureStorageProvider;
pub use gcs::GcsStorageProvider;
pub use local_fs::LocalFsStorageProvider;
pub(crate) use options::StorageOptions;
pub use s3::S3StorageProvider;

use crate::runtime_metrics::io::UsageMeter;

/// Object metadata returned by HEAD, GET, and list operations.
///
/// `size` is the content length in bytes. `etag` is the backend's
/// opaque version token (S3 ETag, LocalFS mtime-derived); used by
/// [`StorageProvider::put_if_match`] for CAS-fenced writes.
/// `last_modified` is `UNIX_EPOCH` for providers that don't surface it.
#[derive(Debug, Clone)]
pub struct ObjectMeta {
    pub size: u64,
    pub etag: Option<String>,
    pub last_modified: SystemTime,
}

/// Errors surfaced by [`StorageProvider`] implementations.
///
/// Variants distinguish permanent failures from
/// transient-exhausted ones so callers can choose recovery
/// (typically none — retry is the provider's job).
#[derive(Debug, Error)]
pub enum StorageError {
    /// Object doesn't exist. Permanent. Returned by `head`,
    /// `get`, `get_range` against a missing URI. `delete` is
    /// idempotent — a missing target returns `Ok(())` rather
    /// than this variant.
    #[error("not found: {uri}")]
    NotFound { uri: String },

    /// Conditional write didn't satisfy precondition.
    ///
    /// Fires when `put_atomic` finds the target already exists
    /// (`If-None-Match: *` on S3, `O_EXCL` on LocalFS) or when
    /// `put_if_match` finds an ETag mismatch. The supertable's
    /// commit loop catches this on the pointer-CAS path and
    /// re-reads + retries; other callers surface it.
    #[error("precondition failed: {uri}")]
    PreconditionFailed { uri: String },

    /// Transient failure that the provider's internal retry
    /// loop already exhausted (e.g., persistent 5xx, repeated
    /// connection reset). Callers do **not** retry.
    #[error("transient error after retry: {uri} — {source}")]
    TransientExhausted {
        uri: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Permanent failure (auth, schema/region mismatch,
    /// corrupted response, malformed URI). Callers do **not**
    /// retry.
    #[error("permanent error: {uri} — {source}")]
    Permanent {
        uri: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}

/// I/O diagnostics that are not the usage ledger: timeline and phase spans.
/// Request/byte counts and background tagging live in
/// [`crate::runtime_metrics::io`] — import that module, not here.
///
/// [`take`] / [`snapshot`] read the **process-default** meter (providers
/// built outside a [`crate::catalog::Connection`]). Prefer
/// `provider.usage_meter().snapshot()` for connection-scoped windows.
pub mod io_counters {
    use std::{
        sync::{Mutex, OnceLock},
        time::Instant,
    };

    use crate::runtime_metrics::io::{UsageMeter, UsageSnapshot, io_is_background};

    /// `(fetches, bytes, hidden_fetches, hidden_bytes)` since the last call on
    /// the process-default meter; resets those get-family counters.
    pub fn take() -> (u64, u64, u64, u64) {
        UsageMeter::process_default().take_gets()
    }

    /// Snapshot of the process-default meter (not a connection meter).
    pub fn snapshot() -> UsageSnapshot {
        UsageMeter::process_default().snapshot()
    }

    // Per-fetch *timeline* — diagnostic for the cold-search critical path.
    // Fetch counts/bytes tell us breadth; they can't tell us whether the cold
    // floor is a *serial dependent chain* (each read gated on the prior's
    // offsets — gaps = network RTT) or *parallel breadth* (many overlapping
    // reads — wall-time = slowest single chain). This records each
    // object-store op's `[start, end)` relative to a shared epoch, so a
    // post-hoc dump shows overlap (parallel) vs back-to-back (serial) and the
    // implied concurrency `Σdur / wall`. Gated on `INFINO_IO_TIMELINE`; a
    // no-op (one relaxed env check) otherwise so the hot path is unaffected.

    /// One recorded object-store fetch on the timeline.
    #[derive(Clone)]
    pub struct FetchSpan {
        pub op: &'static str,
        pub uri: String,
        pub off: u64,
        pub len: u64,
        /// microseconds since the epoch (first recorded span / last reset).
        pub start_us: u64,
        pub end_us: u64,
        /// `true` if issued by a background cache-fill task (off the
        /// query-critical path), `false` for foreground query reads.
        pub background: bool,
    }

    static TIMELINE_ON: OnceLock<bool> = OnceLock::new();
    static PHASE_TRACE_ON: OnceLock<bool> = OnceLock::new();
    static EPOCH: Mutex<Option<Instant>> = Mutex::new(None);
    static SPANS: Mutex<Vec<FetchSpan>> = Mutex::new(Vec::new());

    /// Whether timeline capture is active (`diagnostics.io_timeline` in YAML).
    pub fn timeline_enabled() -> bool {
        *TIMELINE_ON.get_or_init(|| crate::config::global().diagnostics.io_timeline)
    }

    /// Whether CPU phase spans are recorded.
    ///
    /// True when the YAML `diagnostics.io_timeline` flag is on, or when the
    /// process sets `INFINO_TRACE_VECTOR_WARM_PHASES` (bench/diag opt-in;
    /// engine YAML is never overridden by env — this is a separate tracer).
    pub fn phase_enabled() -> bool {
        *PHASE_TRACE_ON.get_or_init(|| {
            timeline_enabled() || std::env::var_os("INFINO_TRACE_VECTOR_WARM_PHASES").is_some()
        })
    }

    /// Capture an op-start `Instant` *iff* the timeline is active; `None`
    /// disables recording for this op with zero overhead when off.
    pub fn timeline_start() -> Option<Instant> {
        if timeline_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Capture a phase-start `Instant` when [`phase_enabled`]; `None` otherwise.
    pub fn phase_start() -> Option<Instant> {
        if phase_enabled() {
            Some(Instant::now())
        } else {
            None
        }
    }

    /// Record a completed op. `start` is the value from [`timeline_start`];
    /// the end is stamped now. No-op when `start` is `None`.
    pub fn timeline_record(
        op: &'static str,
        uri: &str,
        off: u64,
        len: u64,
        start: Option<Instant>,
    ) {
        let Some(start) = start else { return };
        let epoch = {
            let mut e = match EPOCH.lock() {
                Ok(e) => e,
                Err(_) => return,
            };
            *e.get_or_insert(start)
        };
        let to_us = |t: Instant| t.saturating_duration_since(epoch).as_micros() as u64;
        if let Ok(mut spans) = SPANS.lock() {
            spans.push(FetchSpan {
                op,
                uri: uri.to_string(),
                off,
                len,
                start_us: to_us(start),
                end_us: to_us(Instant::now()),
                background: io_is_background(),
            });
        }
    }

    /// Drop all recorded spans AND reset the epoch — call right before the
    /// unit of work to profile (e.g. one cold query or one drain batch).
    pub fn timeline_reset() {
        if let Ok(mut spans) = SPANS.lock() {
            spans.clear();
        }
        // Re-arm the epoch off the next recorded span.
        if let Ok(mut e) = EPOCH.lock() {
            *e = None;
        }
    }

    /// Take all spans recorded since the last reset, sorted by start time.
    pub fn timeline_take() -> Vec<FetchSpan> {
        let mut out = SPANS
            .lock()
            .map(|mut s| std::mem::take(&mut *s))
            .unwrap_or_default();
        out.sort_by_key(|s| s.start_us);
        out
    }

    /// Coarse *phase* log — diagnostic for the non-I/O portion of a profiled
    /// unit of work (the gap between its measured wall and the GET-timeline
    /// wall). The timeline only sees object-store ops; this records named
    /// CPU/await spans so a post-hoc dump shows where the non-read time goes.
    /// Same gate (`INFINO_IO_TIMELINE`); ordered by insertion (caller-sequenced).
    static PHASES: Mutex<Vec<(&'static str, u64)>> = Mutex::new(Vec::new());

    /// Record `name` took `micros` µs. No-op unless [`phase_enabled`].
    pub fn phase_record(name: &'static str, micros: u64) {
        if !phase_enabled() {
            return;
        }
        if let Ok(mut p) = PHASES.lock() {
            p.push((name, micros));
        }
    }

    /// Time `f` and record it under `name` (returns `f`'s value).
    pub fn phase_timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
        if !phase_enabled() {
            return f();
        }
        let t = Instant::now();
        let out = f();
        phase_record(name, t.elapsed().as_micros() as u64);
        out
    }

    /// Async counterpart of [`phase_timed`]: await `fut` and record wall µs.
    pub async fn phase_timed_async<T>(name: &'static str, fut: impl Future<Output = T>) -> T {
        if !phase_enabled() {
            return fut.await;
        }
        let t = Instant::now();
        let out = fut.await;
        phase_record(name, t.elapsed().as_micros() as u64);
        out
    }

    /// Drop all recorded phases — call right before the unit of work to profile.
    pub fn phase_reset() {
        if let Ok(mut p) = PHASES.lock() {
            p.clear();
        }
    }

    /// Take all phases recorded since the last reset, in insertion order.
    pub fn phase_take() -> Vec<(&'static str, u64)> {
        PHASES
            .lock()
            .map(|mut p| std::mem::take(&mut *p))
            .unwrap_or_default()
    }

    /// Sum recorded phases by name (concurrent fan-out units may emit the
    /// same name more than once; callers usually want the sum).
    pub fn phase_take_summed() -> Vec<(&'static str, u64)> {
        let phases = phase_take();
        let mut by_name: std::collections::BTreeMap<&'static str, u64> =
            std::collections::BTreeMap::new();
        for (name, us) in phases {
            *by_name.entry(name).or_default() += us;
        }
        by_name.into_iter().collect()
    }
}

/// Storage backend abstraction.
///
/// Implementations wrap `object_store` crate types (or fakes
/// for tests) and expose the subset of operations the
/// supertable's persistence + disk-cache layers consume.
///
/// All methods are async. Implementations are `Send + Sync`
/// so `Arc<dyn StorageProvider>` can be shared across the
/// supertable: the manifest part loader, the disk cache
/// store, and the writer all hold clones of the *same* `Arc`.
///
/// ## CAS-token invariant
///
/// A provider's conditional-write token is a single opaque,
/// backend-defined value. The token surfaced in [`ObjectMeta::etag`]
/// by `head`/`get`, the token returned by `put_atomic` /
/// `put_if_match`, and the token accepted by `put_if_match`'s
/// `expected_etag` are all the **same kind** (S3/Azure: the HTTP ETag;
/// GCS: the object generation). Callers chain the *returned* token
/// into the next `put_if_match` without re-reading, so a provider that
/// returns a different token kind than it accepts silently breaks OCC.
/// The `cas_conformance` test helper enforces this against every
/// backend.
#[async_trait]
pub trait StorageProvider: Send + Sync + fmt::Debug {
    /// Cheap metadata lookup. Used by the cold-fetch
    /// coordinator to size the destination file before
    /// issuing range-GETs.
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError>;

    /// Read the entire object together with its metadata. The
    /// returned [`ObjectMeta`] reflects the exact version whose
    /// bytes are in the response — no HEAD-then-GET race window
    /// — so callers chaining CAS writes against this read can
    /// use `meta.etag` directly.
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError>;

    /// Conditional read — like [`Self::get`], but returns `Ok(None)`
    /// ("not modified") when the object's current etag equals `etag`,
    /// and the full body + metadata otherwise.
    ///
    /// The default implementation issues a plain [`Self::get`] and
    /// compares etags locally — correct on every backend, though it
    /// still transfers the body. HTTP backends (S3, Azure) override
    /// it with a native `If-None-Match` request so the "unchanged"
    /// case is a bodyless 304. Callers use this for high-frequency
    /// freshness probes of tiny objects (the manifest pointer).
    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        let (bytes, meta) = self.get(uri).await?;
        if meta.etag.as_deref() == Some(etag) {
            return Ok(None);
        }
        Ok(Some((bytes, meta)))
    }

    /// Range-fetch. `range.end` is exclusive.
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError>;

    /// Tail-fetch path: — fetch the last `len` bytes of `uri` AND
    /// return the total object size from the same response.
    ///
    /// Lets cold-open callers (parquet footer / format trailer
    /// readers) skip an upfront `head()` round-trip: a single
    /// suffix-range GET pulls the bytes and discloses the
    /// object size at once.
    ///
    /// Implementations backed by HTTP range-GETs (S3, GCS)
    /// should use `Range: bytes=-len` so the response's
    /// Content-Range header carries the total size. The
    /// default impl falls back to a `head()` + bounded
    /// `get_range()` pair (one HEAD + one GET = 2 RTTs) for
    /// providers that can't directly issue a suffix range.
    ///
    /// `len` is clamped to the object size: callers requesting
    /// more bytes than the object holds receive the whole
    /// object plus `size == object_size`.
    async fn tail(&self, uri: &str, len: u64) -> Result<(Bytes, u64), StorageError> {
        let meta = self.head(uri).await?;
        let len = len.min(meta.size);
        if len == 0 {
            return Ok((Bytes::new(), meta.size));
        }
        let start = meta.size - len;
        let bytes = self.get_range(uri, start..meta.size).await?;
        Ok((bytes, meta.size))
    }

    /// Atomic write — succeeds only if the target doesn't
    /// exist. Maps to `If-None-Match: *` on S3,
    /// `x-goog-if-generation-match: 0` on GCS, `O_EXCL` on
    /// LocalFS.
    ///
    /// Returns the new object's etag when the backend surfaces
    /// one (S3 always, LocalFs via mtime). `Ok(None)` is legal
    /// and means the write succeeded but no etag was reported;
    /// CAS-chained callers treat `None` as "create-only-if-
    /// absent" on the subsequent [`put_if_match`].
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError>;

    /// Conditional write — succeeds only if the target's
    /// current ETag matches `expected_etag`.
    ///
    /// Used for OCC on the manifest pointer: the supertable
    /// reads the current pointer (capturing its etag), builds
    /// the new manifest, then writes the new pointer with the
    /// old etag as precondition. A concurrent writer that
    /// commits between read and write causes
    /// `PreconditionFailed`, which the commit loop catches and
    /// retries.
    ///
    /// `None` expected etag means "create only if absent"
    /// (semantically identical to `put_atomic`); pass `Some`
    /// to update an existing object.
    ///
    /// Returns the new object's etag on success — same
    /// `Ok(None)` semantics as [`put_atomic`].
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError>;

    /// Streaming multipart upload — for superfiles larger than
    /// `SupertableOptions::put_multipart_threshold_bytes`
    /// (default 100 MB), the writer routes through this path
    /// instead of `put_atomic` to avoid buffering the whole
    /// superfile in RAM during commit.
    ///
    /// Returns the underlying `object_store::MultipartUpload`
    /// handle; callers drive it via its own `put_part` /
    /// `complete` / `abort` methods.
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError>;

    /// Delete an object. **Idempotent** — deleting a missing
    /// object returns `Ok(())`, not [`StorageError::NotFound`].
    async fn delete(&self, uri: &str) -> Result<(), StorageError>;

    /// List object URIs under `prefix`. Returns the full URI of
    /// every object whose path starts with `prefix` (caller is
    /// responsible for slash-aware boundary checks if they want
    /// to restrict to direct children).
    ///
    /// Used by the WAL recovery sweep to enumerate
    /// `wal/mutations/*.json`. Listing is a relatively heavy
    /// operation on object-store backends (it's a LIST call;
    /// pagination handled internally) so callers should not
    /// invoke this on the hot path — it's an open-time / sweep-
    /// time primitive.
    ///
    /// List objects under `prefix`, returning each key with its metadata.
    ///
    /// Default returns an empty list — test/mock providers that don't
    /// need listing can leave the default in place; production providers
    /// (LocalFs, S3, Azure, GCS) override.
    async fn list_with_prefix_metadata(
        &self,
        _prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        Ok(Vec::new())
    }

    /// List object keys under `prefix`. Derived from [`list_with_prefix_metadata`].
    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        Ok(self
            .list_with_prefix_metadata(prefix)
            .await?
            .into_iter()
            .map(|(key, _)| key)
            .collect())
    }

    /// Expose the underlying `object_store` handle plus the object
    /// key that `uri` maps to within it, when this provider is backed
    /// by a store DataFusion can range-GET directly.
    ///
    /// Used by the SQL scan and search-hit row resolution to hand
    /// DataFusion's `ParquetSource` the real object store so it issues
    /// async footer / row-group / page range GETs against object
    /// storage, instead of buffering whole superfiles into memory.
    ///
    /// `None` for providers without a native `object_store` handle
    /// (mocks / in-memory test doubles); those callers fall back to the
    /// whole-object read path.
    fn object_store_handle(
        &self,
        _uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        None
    }

    /// Connection-scoped (or process-default) usage ledger this provider
    /// records into. Benches and billing snapshot this meter — there is no
    /// second counter wrapper. Default is the process-default meter (mocks /
    /// ad-hoc providers); durable providers override with their injected Arc.
    fn usage_meter(&self) -> Arc<UsageMeter> {
        UsageMeter::process_default()
    }
}

/// Convert an object-store LIST result back into the provider-relative key
/// accepted by the rest of [`StorageProvider`]. Remote providers prepend their
/// configured root before listing; callers must not see or re-prefix it.
pub(crate) fn logical_list_key(provider_prefix: &str, location: &str) -> String {
    if provider_prefix.is_empty() {
        return location.to_owned();
    }
    let prefix = format!("{provider_prefix}/");
    location
        .strip_prefix(&prefix)
        .expect("listed object remains under provider prefix")
        .to_owned()
}

/// A wrapper that prepends a sub-prefix to every URI before delegating to an
/// inner `StorageProvider`. Used to give the hidden `VectorIndexSuperTable` its
/// own namespace under the user table's storage prefix.
#[derive(Debug)]
pub struct PrefixedStorageProvider {
    inner: Arc<dyn StorageProvider>,
    sub_prefix: String,
}

impl PrefixedStorageProvider {
    pub fn new(inner: Arc<dyn StorageProvider>, sub_prefix: impl Into<String>) -> Self {
        let mut sub = sub_prefix.into();
        if !sub.is_empty() && !sub.ends_with('/') {
            sub.push('/');
        }
        Self {
            inner,
            sub_prefix: sub,
        }
    }

    fn prefixed(&self, uri: &str) -> String {
        format!("{}{}", self.sub_prefix, uri)
    }
}

#[async_trait::async_trait]
impl StorageProvider for PrefixedStorageProvider {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(&self.prefixed(uri)).await
    }

    async fn get(&self, uri: &str) -> Result<(bytes::Bytes, ObjectMeta), StorageError> {
        // Prefixed URI is recorded by the inner provider; UriClass tags it Hidden*.
        self.inner.get(&self.prefixed(uri)).await
    }

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(bytes::Bytes, ObjectMeta)>, StorageError> {
        self.inner
            .get_if_none_match(&self.prefixed(uri), etag)
            .await
    }

    async fn get_range(
        &self,
        uri: &str,
        range: std::ops::Range<u64>,
    ) -> Result<bytes::Bytes, StorageError> {
        self.inner.get_range(&self.prefixed(uri), range).await
    }

    async fn tail(&self, uri: &str, len: u64) -> Result<(bytes::Bytes, u64), StorageError> {
        self.inner.tail(&self.prefixed(uri), len).await
    }

    async fn put_atomic(
        &self,
        uri: &str,
        bytes: bytes::Bytes,
    ) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(&self.prefixed(uri), bytes).await
    }

    async fn put_if_match(
        &self,
        uri: &str,
        bytes: bytes::Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        self.inner
            .put_if_match(&self.prefixed(uri), bytes, expected_etag)
            .await
    }

    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(&self.prefixed(uri)).await
    }

    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(&self.prefixed(uri)).await
    }

    async fn list_with_prefix(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let full = self.prefixed(prefix);
        let results = self.inner.list_with_prefix(&full).await?;
        let strip_len = self.sub_prefix.len();
        Ok(results
            .into_iter()
            .map(|s| s[strip_len..].to_owned())
            .collect())
    }

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        // Must mirror `list_with_prefix` (prepend sub-prefix, delegate, strip):
        // GC on the hidden vector index lists via this method, so without the
        // override the trait default returns an empty list and GC reclaims
        // nothing under the prefixed namespace.
        let full = self.prefixed(prefix);
        let results = self.inner.list_with_prefix_metadata(&full).await?;
        let strip_len = self.sub_prefix.len();
        Ok(results
            .into_iter()
            .map(|(key, meta)| (key[strip_len..].to_owned(), meta))
            .collect())
    }

    fn object_store_handle(
        &self,
        uri: &str,
    ) -> Option<(Arc<dyn object_store::ObjectStore>, object_store::path::Path)> {
        // Prefixed object key is classified Hidden* by UriClass on record.
        self.inner.object_store_handle(&self.prefixed(uri))
    }

    fn usage_meter(&self) -> Arc<UsageMeter> {
        self.inner.usage_meter()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, error::Error, ops::Range, sync::Mutex};

    use async_trait::async_trait;
    use bytes::Bytes;

    use super::*;

    /// Fixed etag the mock reports for any stored object.
    const MOCK_ETAG: &str = "mock-etag";

    #[test]
    fn logical_list_key_strips_exact_provider_root() {
        assert_eq!(
            logical_list_key("table/root", "table/root/data/segment.parquet"),
            "data/segment.parquet"
        );
        assert_eq!(
            logical_list_key("", "table/root/data/segment.parquet"),
            "table/root/data/segment.parquet"
        );
    }

    /// Minimal in-memory [`StorageProvider`] implementing only the
    /// required methods, leaving `tail`, `list_with_prefix`, and
    /// `object_store_handle` at their trait defaults — those default
    /// bodies are the code under test here, since every production
    /// provider overrides all three.
    #[derive(Debug, Default)]
    struct InMemoryMock {
        objects: Mutex<HashMap<String, Bytes>>,
    }

    impl InMemoryMock {
        fn with(uri: &str, bytes: &[u8]) -> Self {
            let mock = Self::default();
            mock.objects
                .lock()
                .expect("lock")
                .insert(uri.into(), Bytes::copy_from_slice(bytes));
            mock
        }
    }

    fn not_found(uri: &str) -> StorageError {
        StorageError::NotFound { uri: uri.into() }
    }

    #[async_trait]
    impl StorageProvider for InMemoryMock {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok(ObjectMeta {
                    size: b.len() as u64,
                    etag: Some(MOCK_ETAG.into()),
                    last_modified: SystemTime::UNIX_EPOCH,
                }),
                None => Err(not_found(uri)),
            }
        }

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok((
                    b.clone(),
                    ObjectMeta {
                        size: b.len() as u64,
                        etag: Some(MOCK_ETAG.into()),
                        last_modified: SystemTime::UNIX_EPOCH,
                    },
                )),
                None => Err(not_found(uri)),
            }
        }

        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            let map = self.objects.lock().expect("lock");
            match map.get(uri) {
                Some(b) => Ok(b.slice(range.start as usize..range.end as usize)),
                None => Err(not_found(uri)),
            }
        }

        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            let mut map = self.objects.lock().expect("lock");
            if map.contains_key(uri) {
                return Err(StorageError::PreconditionFailed { uri: uri.into() });
            }
            map.insert(uri.into(), bytes);
            Ok(Some(MOCK_ETAG.into()))
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            _expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            self.objects.lock().expect("lock").insert(uri.into(), bytes);
            Ok(Some(MOCK_ETAG.into()))
        }

        async fn put_multipart(
            &self,
            uri: &str,
        ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
            // The mock doesn't support streaming uploads; a permanent
            // error is enough to exercise the call path.
            let boxed: Box<dyn Error + Send + Sync> = "multipart unsupported".into();
            Err(StorageError::Permanent {
                uri: uri.into(),
                source: boxed,
            })
        }

        async fn delete(&self, uri: &str) -> Result<(), StorageError> {
            self.objects.lock().expect("lock").remove(uri);
            Ok(())
        }

        fn usage_meter(&self) -> Arc<UsageMeter> {
            UsageMeter::process_default()
        }
    }

    // ---- default `tail` body (LocalFs aside, this is the fallback) ----

    #[tokio::test]
    async fn default_tail_returns_trailing_bytes_and_size() {
        let mock = InMemoryMock::with("k", b"abcdefgh");
        let (bytes, size) = mock.tail("k", 3).await.expect("tail");
        assert_eq!(size, 8);
        assert_eq!(&bytes[..], b"fgh");
    }

    #[tokio::test]
    async fn default_tail_clamps_len_to_object_size() {
        let mock = InMemoryMock::with("k", b"abc");
        let (bytes, size) = mock.tail("k", 100).await.expect("tail over-long");
        assert_eq!(size, 3);
        assert_eq!(&bytes[..], b"abc", "len clamps to the whole object");
    }

    #[tokio::test]
    async fn default_tail_zero_len_returns_empty_with_size() {
        let mock = InMemoryMock::with("k", b"abc");
        let (bytes, size) = mock.tail("k", 0).await.expect("tail zero");
        assert_eq!(size, 3);
        assert!(bytes.is_empty(), "zero-len tail still discloses size");
    }

    #[tokio::test]
    async fn default_tail_propagates_not_found() {
        let mock = InMemoryMock::default();
        assert!(matches!(
            mock.tail("missing", 4).await,
            Err(StorageError::NotFound { .. })
        ));
    }

    // ---- default `list_with_prefix` + `object_store_handle` ----

    #[tokio::test]
    async fn default_list_with_prefix_is_empty() {
        let mock = InMemoryMock::with("a/b", b"x");
        assert!(
            mock.list_with_prefix("a/").await.expect("list").is_empty(),
            "the default list never enumerates objects",
        );
    }

    #[test]
    fn default_object_store_handle_is_none() {
        let mock = InMemoryMock::default();
        assert!(mock.object_store_handle("k").is_none());
    }

    // ---- exercise the required methods so the mock's own surface is
    //      covered too (and the byte ops behave as the trait specifies) ----

    #[tokio::test]
    async fn mock_byte_ops_round_trip() {
        let mock = InMemoryMock::default();

        // put_atomic creates; a second create hits the precondition.
        assert_eq!(
            mock.put_atomic("k", Bytes::from_static(b"hello"))
                .await
                .expect("put_atomic"),
            Some(MOCK_ETAG.to_string()),
        );
        assert!(matches!(
            mock.put_atomic("k", Bytes::from_static(b"x")).await,
            Err(StorageError::PreconditionFailed { .. })
        ));

        // head + get + get_range read it back.
        assert_eq!(mock.head("k").await.expect("head").size, 5);
        let (bytes, _) = mock.get("k").await.expect("get");
        assert_eq!(&bytes[..], b"hello");
        assert_eq!(&mock.get_range("k", 1..3).await.expect("range")[..], b"el");

        // put_if_match overwrites unconditionally for the mock.
        mock.put_if_match("k", Bytes::from_static(b"world!"), Some(MOCK_ETAG))
            .await
            .expect("put_if_match");
        assert_eq!(mock.head("k").await.expect("head2").size, 6);

        // delete is idempotent.
        mock.delete("k").await.expect("delete");
        mock.delete("k").await.expect("delete idempotent");
        assert!(matches!(
            mock.get("k").await,
            Err(StorageError::NotFound { .. })
        ));
        assert!(matches!(
            mock.head("missing").await,
            Err(StorageError::NotFound { .. })
        ));
        assert!(matches!(
            mock.get_range("missing", 0..1).await,
            Err(StorageError::NotFound { .. })
        ));
    }

    /// The default `get_if_none_match` issues a plain `get` and compares etags
    /// locally: a matching etag short-circuits to `None` ("not modified"), a
    /// different one returns the full body + metadata.
    #[tokio::test]
    async fn default_get_if_none_match_reports_modified_state() {
        let mock = InMemoryMock::with("k", b"payload");
        assert!(
            mock.get_if_none_match("k", MOCK_ETAG)
                .await
                .expect("conditional get")
                .is_none(),
            "a matching etag means not-modified",
        );
        let (bytes, meta) = mock
            .get_if_none_match("k", "stale-etag")
            .await
            .expect("conditional get")
            .expect("a mismatched etag returns the body");
        assert_eq!(&bytes[..], b"payload");
        assert_eq!(meta.etag.as_deref(), Some(MOCK_ETAG));
    }

    #[tokio::test]
    async fn mock_put_multipart_surfaces_permanent_error() {
        let mock = InMemoryMock::default();
        assert!(matches!(
            mock.put_multipart("k").await,
            Err(StorageError::Permanent { .. })
        ));
    }

    // ---- error Display + ObjectMeta derives ----

    #[test]
    fn storage_error_display_covers_every_variant() {
        let cases: [(StorageError, &str); 4] = [
            (StorageError::NotFound { uri: "u".into() }, "not found"),
            (
                StorageError::PreconditionFailed { uri: "u".into() },
                "precondition failed",
            ),
            (
                StorageError::TransientExhausted {
                    uri: "u".into(),
                    source: "boom".into(),
                },
                "transient",
            ),
            (
                StorageError::Permanent {
                    uri: "u".into(),
                    source: "boom".into(),
                },
                "permanent",
            ),
        ];
        for (err, needle) in cases {
            assert!(
                err.to_string().contains(needle),
                "{err:?} display should contain {needle:?}",
            );
        }
    }

    #[test]
    fn object_meta_is_clone_and_debug() {
        let meta = ObjectMeta {
            size: 7,
            etag: Some("e".into()),
            last_modified: SystemTime::UNIX_EPOCH,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.size, 7);
        assert_eq!(cloned.etag.as_deref(), Some("e"));
        assert!(format!("{meta:?}").contains("ObjectMeta"));
    }

    /// Each per-op recorder advances its counter; `snapshot` reads them all
    /// without resetting. Assertions are relative (before/after) because the
    /// counters are process-global and other tests may increment concurrently.
    #[test]
    fn usage_meter_record_and_snapshot_are_monotonic() {
        let meter = UsageMeter::new();
        let before = meter.snapshot();
        meter.record_get("seg/x", None, 100);
        meter.record_head();
        meter.record_put(50);
        meter.record_list();
        meter.record_delete();
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.get_bytes, 100);
        assert_eq!(delta.head_count, 1);
        assert_eq!(delta.put_count, 1);
        assert_eq!(delta.put_bytes, 50);
        assert_eq!(delta.list_count, 1);
        assert_eq!(delta.delete_count, 1);
    }

    /// Hidden-namespace reads must tag `record_hidden_get` on get / range /
    /// tail / object-store-handle paths (not only `get_range`).
    #[tokio::test]
    async fn prefixed_provider_tags_hidden_gets() {
        use object_store::ObjectStoreExt;

        use crate::storage::LocalFsStorageProvider;

        // Prefix must contain `_vector_index` so UriClass classifies as Hidden*.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let meter = UsageMeter::new();
        let inner = Arc::new(
            LocalFsStorageProvider::new_with_meter(dir.path(), Arc::clone(&meter))
                .expect("localfs"),
        );
        let prefixed = PrefixedStorageProvider::new(inner, "_infino_test_vector_index/");
        prefixed
            .put_atomic("seg/x.bin", Bytes::from_static(b"0123456789"))
            .await
            .expect("put");

        let before = meter.snapshot();
        let (got, _) = prefixed.get("seg/x.bin").await.expect("get");
        assert_eq!(got.as_ref(), b"0123456789");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.hidden_get_count(), 1);
        assert_eq!(delta.hidden_get_bytes(), 10);

        let before = meter.snapshot();
        let _ = prefixed.get_range("seg/x.bin", 0..4).await.expect("range");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.hidden_get_count(), 1);
        assert_eq!(delta.hidden_get_bytes(), 4);

        let before = meter.snapshot();
        let (tail, size) = prefixed.tail("seg/x.bin", 3).await.expect("tail");
        assert_eq!(size, 10);
        assert_eq!(tail.as_ref(), b"789");
        let delta = meter.snapshot().since(&before);
        assert!(delta.get_count >= 1);
        assert_eq!(delta.hidden_get_count(), 1);
        assert_eq!(delta.hidden_get_bytes(), 3);

        let before = meter.snapshot();
        let (store, path) = prefixed.object_store_handle("seg/x.bin").expect("handle");
        let _ = store.get(&path).await.expect("os get");
        let delta = meter.snapshot().since(&before);
        assert_eq!(delta.get_count, 1);
        assert_eq!(delta.hidden_get_count(), 1);
        assert_eq!(delta.hidden_get_bytes(), 10);
    }
}
