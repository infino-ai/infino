//! [`LazyByteSource`] — pulls byte ranges from an arbitrary
//! backing (mmap, network range-fetch, broadcast subscription)
//! so [`SuperfileReader::open_lazy`] can construct a reader
//! without materializing the full segment up-front.
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
//! of bytes-in-hand. It asks the source for the full
//! segment (`source.get_range(0..size)`) and constructs the
//! same reader `open(bytes)` would. The caller no longer
//! materializes the segment before calling; the source
//! decides where the bytes come from (mmap of a local
//! file, range-fetched object store, a coalescing
//! broadcaster that fans one fetch out to many
//! subscribers).
//!
//! What it doesn't do *yet*: fetch only the bytes a
//! specific BM25 / vector query touches. The inner FTS
//! posting reader + vector cluster reader still take a
//! full materialized buffer, so each `open_lazy` call
//! still pulls the whole segment regardless of which
//! queries will run against the resulting reader. Making
//! per-query laziness work needs those inner readers to
//! thread the source through their own lookups — the trait
//! shape here is what makes that change source-compatible
//! when it lands.
//!
//! See [`SuperfileReader::open_lazy`].
//!
//! [`SuperfileReader::open_lazy`]: crate::superfile::reader::SuperfileReader::open_lazy

use async_trait::async_trait;
use bytes::Bytes;

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
}

/// In-memory `LazyByteSource` adapter — useful for tests and
/// for callers that already have the full segment bytes.
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
}
