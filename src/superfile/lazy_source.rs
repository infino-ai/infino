// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`LazyByteSource`] — pulls byte ranges from an arbitrary
//! backing (mmap, network range-fetch, broadcast subscription)
//! so [`SuperfileReader::open_lazy`] can construct a reader
//! without materializing the full superfile up-front.
//!
//! The trait lives next to the `superfile` reader; concrete
//! impls live wherever the backing does. Errors propagate
//! the typed [`crate::storage::StorageError`] directly —
//! `storage` is a foundational module that both `superfile`
//! and `supertable` build on, so no layering inversion.
//!
//! ## What "lazy" means here
//!
//! [`SuperfileReader::open_lazy`] accepts a source instead
//! of bytes-in-hand. The caller no longer materializes the
//! superfile before calling; the source decides where the
//! bytes come from (mmap of a local file, range-fetched
//! object store, a coalescing broadcaster that fans one
//! fetch out to many subscribers).
//!
//! Opening reads only the metadata ranges the reader needs,
//! not the whole superfile: the Parquet footer, then the FTS
//! and vector section headers and directories. The inner
//! readers thread the source through their own lookups, so
//! queries fetch only the bytes they touch — `FtsReader`
//! fetches each query term's posting list on demand (the
//! postings region is never read in full), and `VectorReader`
//! fetches centroids and then only the probed clusters'
//! blocks. A source-opened [`SuperfileReader`] therefore does
//! not retain the full superfile; `parquet_bytes()` returns
//! `None`, and pass-through Parquet callers use the eager
//! `open` path instead.
//!
//! See [`SuperfileReader::open_lazy`].
//!
//! [`SuperfileReader::open_lazy`]: crate::superfile::reader::SuperfileReader::open_lazy

use std::{fmt, ops::Range, sync::Arc};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::try_join_all;

use crate::runtime_bridge::bridge_sync_to_async_send;

/// Source of byte ranges from an arbitrary backing.
///
/// Async because the non-trivial impls (object-store
/// range-fetch, broadcast subscription) are async. The
/// in-memory `Bytes`-backed impl is also async for trait
/// consistency (it just resolves immediately).
#[async_trait]
pub trait LazyByteSource: Send + Sync {
    /// Total size of the backing object, in bytes.
    fn size(&self) -> u64;

    /// Fetch a contiguous range of `len` bytes starting at
    /// `start`. The returned `Bytes` must equal what
    /// `&full_object[start..start+len]` would have returned.
    ///
    /// Out-of-bounds requests (start + len > size()) return
    /// [`LazyByteSourceError::OutOfBounds`]. Underlying
    /// storage failures propagate via
    /// [`LazyByteSourceError::Storage`].
    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError>;

    /// Best-effort sync access to a contiguous range without
    /// I/O. Implementations that always have the bytes
    /// resident in memory (e.g. [`BytesLazyByteSource`], a
    /// mmap'd file with the pages already faulted in) return
    /// `Some` zero-copy. Implementations backed by network
    /// fetches return `Some` only if the range happens to be
    /// in an in-process LRU cache, otherwise `None`.
    ///
    /// This method exists so the vector reader's sync
    /// `search()` path can stay sync on the in-memory
    /// source without spawning an async runtime. On an
    /// out-of-bounds range the implementation may return
    /// `None` (treated as "not available sync" by the
    /// caller, which then either falls back to the async
    /// `range` or surfaces an `OutOfBounds` error itself).
    ///
    /// The default impl returns `None`; in-memory and warm-
    /// cache sources override.
    fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
        None
    }

    /// Tail-fetch path: — fetch the last `len` bytes of the
    /// backing object AND surface the total object size.
    ///
    /// Lets the cold-open path (parquet footer, format
    /// trailer parsing) skip the upfront `head()` round-trip
    /// it would otherwise need to learn `size` before issuing
    /// a tail-relative range GET. Implementations backed by
    /// object stores should override to issue a native
    /// `bytes=-len` suffix range; implementations that
    /// already know their size return it trivially via the
    /// default impl below.
    ///
    /// Default impl: read `size()`, error if zero (the source
    /// genuinely doesn't know its size and hasn't been
    /// overridden to handle that case), else clamp `len` and
    /// fall through to `range(size - len, len)`.
    ///
    /// Returns `(bytes, total_size)`. `bytes.len() == len`
    /// when the object is at least `len` bytes; otherwise
    /// the returned slice covers the entire object and
    /// `total_size == bytes.len()`.
    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        let size = self.size();
        if size == 0 {
            return Err(LazyByteSourceError::OutOfBounds {
                start: 0,
                len,
                size: 0,
            });
        }
        let len = len.min(size);
        let bytes = self.range(size - len, len).await?;
        Ok((bytes, size))
    }
}

/// Errors surfaced by [`LazyByteSource`] implementations.
#[derive(Debug, thiserror::Error)]
pub enum LazyByteSourceError {
    /// Underlying storage / network failure.
    /// `#[from]`-convertible from
    /// [`crate::storage::StorageError`] so impls backed by
    /// the storage layer (range-fetch over an object store,
    /// LocalFS) propagate the typed error directly instead
    /// of stringifying it.
    #[error("lazy source storage: {0}")]
    Storage(#[from] crate::storage::StorageError),

    /// Caller requested a range outside `size()`.
    #[error("range out of bounds: start={start} len={len} size={size}")]
    OutOfBounds { start: u64, len: u64, size: u64 },

    /// The backing storage returned fewer bytes than the
    /// requested range without erroring (a clamped/partial
    /// range, e.g. an object_store GET that hit a truncated
    /// body or an object shorter than the cached size). The
    /// [`LazyByteSource::range`] contract requires the returned
    /// bytes to equal `full_object[start..start + len]`, so a
    /// read that cannot be completed surfaces here rather than
    /// being handed up truncated — a short buffer otherwise
    /// panics deep in a sub-reader's slice math.
    #[error("short read: start={start} requested={requested} got={got}")]
    ShortRead {
        start: u64,
        requested: u64,
        got: u64,
    },
}

/// Backing for source-aware superfile sub-readers.
///
/// `InMemory` is the eager path: callers already hold the complete
/// subsection and every access is a zero-copy [`Bytes::slice`].
/// `Lazy` is a range-fetching source: mmap, object storage, or a
/// foreground cold-fetch subscriber.
#[derive(Clone)]
pub(crate) enum Source {
    InMemory(Bytes),
    Lazy(Arc<dyn LazyByteSource>),
}

impl fmt::Debug for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InMemory(b) => f.debug_tuple("InMemory").field(&b.len()).finish(),
            Self::Lazy(_) => f.debug_struct("Lazy").finish_non_exhaustive(),
        }
    }
}

impl Source {
    /// Total backing size in bytes.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Lazy(s) => s.size() as usize,
        }
    }

    /// Best-effort sync fetch. Always succeeds for in-bounds
    /// `InMemory` ranges; `Lazy` sources may return `Some` for
    /// already-resident ranges.
    pub fn try_get_range_sync(&self, range: Range<usize>) -> Option<Bytes> {
        let start = range.start as u64;
        let len = range.len() as u64;
        match self {
            Self::InMemory(b) => {
                if range.end > b.len() {
                    return None;
                }
                Some(b.slice(range))
            }
            Self::Lazy(s) => s.try_get_range_sync(start, len),
        }
    }

    /// Sync range fetch with an internal async bridge for cold lazy
    /// misses. Hot in-memory and mmap-backed paths resolve through
    /// [`Self::try_get_range_sync`] and do not enter a runtime.
    pub fn get_range(&self, range: Range<usize>) -> Result<Bytes, LazyByteSourceError> {
        if let Some(bytes) = self.try_get_range_sync(range.clone()) {
            return Ok(bytes);
        }
        let Self::Lazy(s) = self else {
            return Err(LazyByteSourceError::OutOfBounds {
                start: range.start as u64,
                len: range.len() as u64,
                size: self.len() as u64,
            });
        };
        let start = range.start as u64;
        let len = range.len() as u64;
        // All three runtime contexts (multi-thread worker, rayon
        // reader-pool thread with no ambient runtime, or a
        // `current_thread` test runtime) are handled by the shared
        // bridge in `runtime_bridge`. Clone the `Arc` into the future
        // so it is `Send + 'static`.
        let src = Arc::clone(s);
        bridge_sync_to_async_send(async move { src.range(start, len).await })
    }

    /// Concurrent multi-range fetch. Sync-resident ranges are served
    /// immediately; cold lazy misses are dispatched under one async
    /// bridge and returned in input order.
    pub fn get_ranges_parallel(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<Vec<Bytes>, LazyByteSourceError> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<Option<Bytes>> = Vec::with_capacity(ranges.len());
        let mut pending: Vec<(usize, u64, u64)> = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            if let Some(b) = self.try_get_range_sync(r.clone()) {
                out.push(Some(b));
                continue;
            }
            if !matches!(self, Self::Lazy(_)) {
                return Err(LazyByteSourceError::OutOfBounds {
                    start: r.start as u64,
                    len: r.len() as u64,
                    size: self.len() as u64,
                });
            }
            pending.push((i, r.start as u64, r.len() as u64));
            out.push(None);
        }

        if !pending.is_empty() {
            let Self::Lazy(s) = self else {
                unreachable!("pending non-empty implies Source::Lazy");
            };
            let src = Arc::clone(s);
            let order: Vec<usize> = pending.iter().map(|(i, _, _)| *i).collect();
            let fut = async move {
                let futs = pending
                    .into_iter()
                    .map(|(_i, start, len)| {
                        let s = Arc::clone(&src);
                        async move { s.range(start, len).await }
                    })
                    .collect::<Vec<_>>();
                try_join_all(futs).await
            };
            // Shared bridge handles every runtime context; the future
            // owns its `Arc` clones so it is `Send + 'static`.
            let bytes: Vec<Bytes> = bridge_sync_to_async_send(fut)?;
            for (slot, b) in order.into_iter().zip(bytes) {
                out[slot] = Some(b);
            }
        }

        Ok(out
            .into_iter()
            .map(|b| b.expect("every slot filled by sync or async path"))
            .collect())
    }

    /// Async single-range fetch. Sync-resident ranges (in-memory,
    /// warm mmap, in-process cache hits) resolve with zero I/O;
    /// cold `Lazy` misses `await` the source's async `range`
    /// directly on the caller's runtime — no `block_on`, no
    /// throwaway runtime, so the object-store client's reqwest
    /// connections stay on the runtime that owns them.
    pub async fn range_async(&self, range: Range<usize>) -> Result<Bytes, LazyByteSourceError> {
        if let Some(bytes) = self.try_get_range_sync(range.clone()) {
            return Ok(bytes);
        }
        let Self::Lazy(s) = self else {
            return Err(LazyByteSourceError::OutOfBounds {
                start: range.start as u64,
                len: range.len() as u64,
                size: self.len() as u64,
            });
        };
        s.range(range.start as u64, range.len() as u64).await
    }

    /// Async concurrent multi-range fetch. Sync-resident ranges are
    /// served immediately; cold `Lazy` misses are dispatched as one
    /// `try_join_all` batch and `await`ed on the caller's runtime.
    /// Returns bytes in input order. This is the async sibling of
    /// [`Self::get_ranges_parallel`] and carries the same ordering
    /// and bounds-check contract without the sync→async bridge.
    pub async fn get_ranges_parallel_async(
        &self,
        ranges: &[Range<usize>],
    ) -> Result<Vec<Bytes>, LazyByteSourceError> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let mut out: Vec<Option<Bytes>> = Vec::with_capacity(ranges.len());
        let mut pending: Vec<(usize, u64, u64)> = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            if let Some(b) = self.try_get_range_sync(r.clone()) {
                out.push(Some(b));
                continue;
            }
            if !matches!(self, Self::Lazy(_)) {
                return Err(LazyByteSourceError::OutOfBounds {
                    start: r.start as u64,
                    len: r.len() as u64,
                    size: self.len() as u64,
                });
            }
            pending.push((i, r.start as u64, r.len() as u64));
            out.push(None);
        }

        if !pending.is_empty() {
            let Self::Lazy(s) = self else {
                unreachable!("pending non-empty implies Source::Lazy");
            };
            let order: Vec<usize> = pending.iter().map(|(i, _, _)| *i).collect();
            let futs = pending
                .into_iter()
                .map(|(_i, start, len)| {
                    let s = Arc::clone(s);
                    async move { s.range(start, len).await }
                })
                .collect::<Vec<_>>();
            let bytes = try_join_all(futs).await?;
            for (slot, b) in order.into_iter().zip(bytes) {
                out[slot] = Some(b);
            }
        }

        Ok(out
            .into_iter()
            .map(|b| b.expect("every slot filled by sync or async path"))
            .collect())
    }
}

/// In-memory `LazyByteSource` adapter — useful for tests and
/// for callers that already have the full superfile bytes.
#[derive(Debug, Clone)]
pub struct BytesLazyByteSource {
    bytes: Bytes,
}

impl BytesLazyByteSource {
    pub fn new(bytes: Bytes) -> Self {
        Self { bytes }
    }
}

#[async_trait]
impl LazyByteSource for BytesLazyByteSource {
    fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let total = self.bytes.len() as u64;
        if start.saturating_add(len) > total {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: total,
            });
        }
        let s = start as usize;
        let e = s + len as usize;
        Ok(self.bytes.slice(s..e))
    }

    /// In-memory bytes are always available without I/O.
    /// Returns a zero-copy `Bytes::slice` of the backing
    /// buffer (atomic refcount bump only, no allocation).
    /// `None` on out-of-bounds — the caller falls back to
    /// `range` for a typed error if it cares.
    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        let total = self.bytes.len() as u64;
        if start.saturating_add(len) > total {
            return None;
        }
        let s = start as usize;
        let e = s + len as usize;
        Some(self.bytes.slice(s..e))
    }
}

/// Lazy sub-range path: — sub-range view onto another [`LazyByteSource`].
///
/// `SuperfileReader::open_lazy` uses this to hand a sub-region
/// of the outer superfile (the vector subsection, the FTS
/// subsection) through to the inner readers without each
/// inner reader having to do absolute-offset arithmetic.
///
/// Every `range(start, len)` on the sub-source translates to
/// `inner.range(offset + start, len)`. `size()` is the
/// sub-region length (`len`), not the inner source's total.
/// `try_get_range_sync` shifts the offset the same way and
/// passes through to the inner.
///
/// Bounds: out-of-bounds requests (start + len > self.size())
/// surface as `OutOfBounds` errors with `size = self.size`,
/// not the inner's larger size — keeps caller-visible errors
/// scoped to the slice the caller actually sees.
pub(crate) struct LazySubSource {
    inner: Arc<dyn LazyByteSource>,
    /// Absolute offset of the sub-region's start inside the
    /// inner source.
    offset: u64,
    /// Length of the sub-region.
    len: u64,
}

impl LazySubSource {
    pub fn new(inner: Arc<dyn LazyByteSource>, offset: u64, len: u64) -> Self {
        debug_assert!(offset + len <= inner.size(), "sub-source overruns inner");
        Self { inner, offset, len }
    }
}

impl fmt::Debug for LazySubSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazySubSource")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

#[async_trait]
impl LazyByteSource for LazySubSource {
    fn size(&self) -> u64 {
        self.len
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if start.saturating_add(len) > self.len {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: self.len,
            });
        }
        self.inner.range(self.offset + start, len).await
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if start.saturating_add(len) > self.len {
            return None;
        }
        self.inner.try_get_range_sync(self.offset + start, len)
    }
}

/// Prefetched overlay: — overlay that serves pre-fetched byte ranges
/// from memory and falls through to an underlying
/// [`LazyByteSource`] on miss.
///
/// `VectorReader::open_lazy` uses this to install the 2–3
/// open-time range fetches it issues against the underlying
/// source, then hands the overlay to `open_with_source`. Every
/// per-region read inside `open_with_source` (sub-header,
/// codec_meta, etc.) lives inside one of the installed ranges
/// and resolves zero-copy via [`Self::try_get_range_sync`] —
/// no extra GETs against the underlying source.
///
/// Concurrency: `install` is non-mutating (`&self`) and uses
/// an interior `Vec` to allow staged construction during
/// `open_lazy`, then the overlay is wrapped in `Arc` and
/// shared by every subsequent read. The Vec is read-only after
/// the open completes, so the racy-read-during-install pattern
/// the lazy open path actually uses is single-threaded.
pub(crate) struct PrefetchedSource {
    inner: Arc<dyn LazyByteSource>,
    /// (absolute_start, bytes). One entry per pre-fetched
    /// range. Lookup walks the vec linearly — the open-time
    /// path installs ≤ 3 ranges per superfile so the linear
    /// scan is faster than a tree-keyed structure (cache-line
    /// hot, no allocation).
    prefetched: Vec<(u64, Bytes)>,
}

impl PrefetchedSource {
    pub fn new(inner: Arc<dyn LazyByteSource>) -> Self {
        Self {
            inner,
            prefetched: Vec::new(),
        }
    }

    /// Install a pre-fetched range. `start` is the absolute
    /// offset into the backing object; `bytes.len()` is the
    /// range length. Subsequent `try_get_range_sync` /
    /// `range` requests for any sub-range of
    /// `[start..start + bytes.len())` resolve from this
    /// buffer without touching the underlying source.
    pub fn install(&mut self, start: u64, bytes: Bytes) {
        self.prefetched.push((start, bytes));
    }

    /// Lookup helper — returns a zero-copy slice if any
    /// installed range covers the request.
    fn lookup(&self, start: u64, len: u64) -> Option<Bytes> {
        let req_end = start.checked_add(len)?;
        for (p_start, p_bytes) in &self.prefetched {
            let p_end = *p_start + p_bytes.len() as u64;
            if *p_start <= start && req_end <= p_end {
                let offset = (start - *p_start) as usize;
                return Some(p_bytes.slice(offset..offset + len as usize));
            }
        }
        None
    }
}

impl fmt::Debug for PrefetchedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PrefetchedSource")
            .field("size", &self.inner.size())
            .field("prefetched_count", &self.prefetched.len())
            .field(
                "prefetched_bytes",
                &self.prefetched.iter().map(|(_, b)| b.len()).sum::<usize>(),
            )
            .finish()
    }
}

#[async_trait]
impl LazyByteSource for PrefetchedSource {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if let Some(b) = self.lookup(start, len) {
            return Ok(b);
        }
        self.inner.range(start, len).await
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if let Some(b) = self.lookup(start, len) {
            return Some(b);
        }
        self.inner.try_get_range_sync(start, len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bytes_lazy_source_size_and_range() {
        let payload = Bytes::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let src = BytesLazyByteSource::new(payload.clone());
        assert_eq!(src.size(), payload.len() as u64);

        let slice = src.range(2, 4).await.expect("range");
        assert_eq!(slice.as_ref(), &payload[2..6]);
    }

    #[tokio::test]
    async fn bytes_lazy_source_out_of_bounds_surfaces_typed_error() {
        let src = BytesLazyByteSource::new(Bytes::from(vec![0u8; 4]));
        let err = src
            .range(2, 100)
            .await
            .expect_err("must reject out-of-bounds");
        assert!(
            matches!(err, LazyByteSourceError::OutOfBounds { .. }),
            "expected OutOfBounds, got {err:?}"
        );
    }

    /// Sync access on `BytesLazyByteSource` must always
    /// succeed for in-bounds ranges (it's in-memory backed)
    /// and must return a zero-copy slice of the source's
    /// underlying buffer.
    #[test]
    fn bytes_lazy_source_try_get_range_sync_returns_zero_copy_slice() {
        let payload = Bytes::from(vec![10u8, 20, 30, 40, 50, 60, 70, 80]);
        let src = BytesLazyByteSource::new(payload.clone());
        let got = src
            .try_get_range_sync(2, 4)
            .expect("in-bounds sync must succeed");
        assert_eq!(got.as_ref(), &payload[2..6]);
        // Zero-copy: the returned Bytes shares the same
        // allocation as the source (Bytes::slice does a
        // refcount bump, never copies). Compare the raw
        // backing pointers to assert that — Bytes::as_ptr()
        // points at the slice's first byte, so for
        // `slice(2..6)` it lands at `payload.as_ptr() + 2`.
        let expected_ptr = unsafe { payload.as_ptr().add(2) };
        assert_eq!(got.as_ptr(), expected_ptr);
    }

    #[test]
    fn bytes_lazy_source_try_get_range_sync_returns_none_out_of_bounds() {
        let src = BytesLazyByteSource::new(Bytes::from(vec![0u8; 4]));
        assert!(src.try_get_range_sync(2, 100).is_none());
        assert!(src.try_get_range_sync(100, 0).is_none());
    }

    /// Prefetched overlay: — overlay serves an installed range
    /// zero-copy via `try_get_range_sync` without calling
    /// into the underlying source.
    #[tokio::test]
    async fn prefetched_source_serves_installed_range_zero_copy() {
        let payload = Bytes::from(vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let inner = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let mut overlay = PrefetchedSource::new(inner);
        let installed = payload.slice(2..7);
        overlay.install(2, installed.clone());

        let got = overlay
            .try_get_range_sync(3, 3)
            .expect("installed range covers (3..6)");
        assert_eq!(got.as_ref(), &payload[3..6]);

        let async_got = overlay.range(3, 3).await.expect("async path also serves");
        assert_eq!(async_got.as_ref(), &payload[3..6]);
    }

    /// Lazy sub-range path: — sub-source slices into a parent's range.
    /// `range(start, len)` resolves against the parent at
    /// `offset + start`; `size()` returns the slice length.
    #[tokio::test]
    async fn lazy_sub_source_translates_offsets_and_reports_slice_size() {
        let payload = Bytes::from((0u8..32).collect::<Vec<_>>());
        let inner: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        // Slice the middle 16 bytes (offset 8, len 16).
        let sub = LazySubSource::new(Arc::clone(&inner), 8, 16);
        assert_eq!(sub.size(), 16, "sub-source size must equal slice length");

        let got = sub.range(0, 4).await.expect("range(0..4) in slice");
        assert_eq!(got.as_ref(), &payload[8..12]);
        let got = sub.range(12, 4).await.expect("range(12..16) in slice");
        assert_eq!(got.as_ref(), &payload[20..24]);

        let sync_got = sub.try_get_range_sync(2, 6).expect("sync in slice");
        assert_eq!(sync_got.as_ref(), &payload[10..16]);
    }

    /// Lazy sub-range path: — out-of-bounds requests against a sub-source
    /// surface with the slice's `size`, not the inner's larger
    /// size — keeps the caller-visible error scoped to the slice
    /// the caller actually addressed.
    #[tokio::test]
    async fn lazy_sub_source_out_of_bounds_uses_slice_size_in_error() {
        let payload = Bytes::from(vec![0u8; 32]);
        let inner: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload));
        let sub = LazySubSource::new(Arc::clone(&inner), 8, 16);
        let err = sub
            .range(10, 10)
            .await
            .expect_err("10+10 overruns the 16-byte slice");
        match err {
            LazyByteSourceError::OutOfBounds { start, len, size } => {
                assert_eq!(start, 10);
                assert_eq!(len, 10);
                assert_eq!(size, 16, "size must be the slice's, not the inner's");
            }
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
        assert!(sub.try_get_range_sync(10, 10).is_none());
    }

    /// Prefetched overlay: — counting source confirms a range request
    /// that hits the overlay never reaches the underlying source.
    #[tokio::test]
    async fn prefetched_source_overlay_hit_skips_underlying_range_call() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Debug)]
        struct CountingSource {
            inner: BytesLazyByteSource,
            range_calls: AtomicUsize,
        }

        #[async_trait]
        impl LazyByteSource for CountingSource {
            fn size(&self) -> u64 {
                self.inner.size()
            }
            async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
                self.range_calls.fetch_add(1, Ordering::SeqCst);
                self.inner.range(start, len).await
            }
            fn try_get_range_sync(&self, _: u64, _: u64) -> Option<Bytes> {
                None
            }
        }

        let payload = Bytes::from(vec![0u8, 1, 2, 3, 4, 5, 6, 7]);
        let counting = Arc::new(CountingSource {
            inner: BytesLazyByteSource::new(payload.clone()),
            range_calls: AtomicUsize::new(0),
        });
        let prefetched = counting.range(0, 4).await.expect("seed prefetch");
        assert_eq!(counting.range_calls.load(Ordering::SeqCst), 1);

        let mut overlay = PrefetchedSource::new(counting.clone());
        overlay.install(0, prefetched);

        let _ = overlay.range(1, 2).await.expect("overlay serves");
        let _ = overlay.range(0, 4).await.expect("overlay serves");
        assert_eq!(
            counting.range_calls.load(Ordering::SeqCst),
            1,
            "overlay hits must not bump the underlying range counter"
        );

        let _ = overlay.range(4, 4).await.expect("miss falls through");
        assert_eq!(
            counting.range_calls.load(Ordering::SeqCst),
            2,
            "an overlay miss must reach the underlying source exactly once"
        );
    }

    /// The `Debug` impls on `Source`, `LazySubSource`, and
    /// `PrefetchedSource` render without panicking and name their type.
    #[test]
    fn debug_impls_render_each_source_kind() {
        let payload = Bytes::from(vec![0u8; 16]);
        let in_mem = Source::InMemory(payload.clone());
        assert!(format!("{in_mem:?}").contains("InMemory"));

        let inner: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(payload.clone()));
        let lazy = Source::Lazy(Arc::clone(&inner));
        assert!(format!("{lazy:?}").contains("Lazy"));

        let sub = LazySubSource::new(Arc::clone(&inner), 4, 8);
        assert!(format!("{sub:?}").contains("LazySubSource"));

        let mut overlay = PrefetchedSource::new(Arc::clone(&inner));
        overlay.install(0, payload.slice(0..4));
        assert!(format!("{overlay:?}").contains("PrefetchedSource"));
    }

    /// The `Source` enum's sync + async fetch surface: in-memory happy
    /// paths, the empty-range fast path, and the out-of-bounds error
    /// branches that the non-`Source` tests above never reach.
    #[tokio::test]
    async fn source_enum_fetch_surface_and_bounds() {
        let payload = Bytes::from((0u8..16).collect::<Vec<_>>());
        let src = Source::InMemory(payload.clone());
        assert_eq!(src.len(), 16);
        assert_eq!(
            src.try_get_range_sync(2..6)
                .expect("in-bounds sync")
                .as_ref(),
            &payload[2..6]
        );
        assert_eq!(
            src.get_range(0..4).expect("get_range").as_ref(),
            &payload[0..4]
        );

        // Out-of-bounds on an in-memory source surfaces a typed error
        // (no lazy fallback).
        assert!(matches!(
            src.get_range(8..100),
            Err(LazyByteSourceError::OutOfBounds { .. })
        ));

        // Multi-range: empty fast path, happy path, and OOB.
        assert!(src.get_ranges_parallel(&[]).expect("empty").is_empty());
        let got = src.get_ranges_parallel(&[0..2, 4..8]).expect("multi-range");
        assert_eq!(got.len(), 2);
        assert!(matches!(
            src.get_ranges_parallel(&[0..2, 8..100]),
            Err(LazyByteSourceError::OutOfBounds { .. })
        ));

        // Async siblings: happy, empty, and OOB.
        assert_eq!(
            src.range_async(4..8).await.expect("range_async").as_ref(),
            &payload[4..8]
        );
        assert!(matches!(
            src.range_async(8..100).await,
            Err(LazyByteSourceError::OutOfBounds { .. })
        ));
        assert!(
            src.get_ranges_parallel_async(&[])
                .await
                .expect("empty async")
                .is_empty()
        );
        assert!(matches!(
            src.get_ranges_parallel_async(&[0..2, 8..100]).await,
            Err(LazyByteSourceError::OutOfBounds { .. })
        ));
    }
}
