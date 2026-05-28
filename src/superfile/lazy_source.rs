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
//! Per-query laziness — fetch only the bytes a specific
//! BM25 / vector query touches — is rolled out a subsystem
//! at a time. PR1 of plan 013 lands the vector half:
//! `VectorReader::open_with_source` parses just the per-
//! column sub-header + Sq8 codec-meta region on the lazy
//! path, instead of materializing whole subsections. The
//! supertable layer composes a parent `LazyByteSource`
//! (over the segment) with [`SubLazyByteSource`] to scope
//! the view to the vector blob's byte window before
//! handing it to `VectorReader::open_with_source` — so the
//! inner reader stays oblivious to where in the file its
//! blob lives. The FTS half follows in plan-013 PR4.
//!
//! See [`SuperfileReader::open_lazy`].
//!
//! [`SuperfileReader::open_lazy`]: crate::superfile::reader::SuperfileReader::open_lazy

use async_trait::async_trait;
use bytes::Bytes;
use std::ops::Range;
use std::sync::Arc;

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

/// Window over a parent [`LazyByteSource`]. `range(s, l)`
/// translates to `parent.range(offset + s, l)` and
/// `try_get_range_sync(s, l)` to the same on the sync path.
///
/// Used by `SuperfileReader::open_lazy_with` to hand each
/// sub-blob (FTS, vector) a [`LazyByteSource`] view scoped to
/// its byte region inside the parent segment, without the
/// sub-readers having to know about the supertable layer's
/// offset arithmetic. Composing this with a counting source
/// in tests also lets us assert per-blob byte / range budgets
/// directly.
///
/// Out-of-bounds inside the window surfaces as
/// [`LazyByteSourceError::OutOfBounds`] with the *window's*
/// size (`length`), not the parent's, so error messages
/// reflect the sub-reader's view.
#[derive(Clone)]
pub struct SubLazyByteSource {
    parent: Arc<dyn LazyByteSource>,
    offset: u64,
    length: u64,
}

impl std::fmt::Debug for SubLazyByteSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubLazyByteSource")
            .field("offset", &self.offset)
            .field("length", &self.length)
            .finish()
    }
}

impl SubLazyByteSource {
    /// Build a window `[offset, offset + length)` over
    /// `parent`. The window must fit inside the parent's
    /// `size()`; this is enforced at construction so callers
    /// see a typed error up-front instead of at the first
    /// `range()`.
    pub fn new(
        parent: Arc<dyn LazyByteSource>,
        offset: u64,
        length: u64,
    ) -> Result<Self, LazyByteSourceError> {
        let parent_size = parent.size();
        if offset.saturating_add(length) > parent_size {
            return Err(LazyByteSourceError::OutOfBounds {
                start: offset,
                len: length,
                size: parent_size,
            });
        }
        Ok(Self {
            parent,
            offset,
            length,
        })
    }
}

#[async_trait]
impl LazyByteSource for SubLazyByteSource {
    fn size(&self) -> u64 {
        self.length
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if start.checked_add(len).is_none_or(|end| end > self.length) {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: self.length,
            });
        }
        self.parent.range(self.offset + start, len).await
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if start.checked_add(len).is_none_or(|end| end > self.length) {
            return None;
        }
        self.parent.try_get_range_sync(self.offset + start, len)
    }
}

/// Polymorphic byte source for the structural decode path of
/// every reader in `superfile::*` (vector, FTS, supertable
/// top-level). Two variants:
///
/// - [`Source::InMemory`] — caller already materialised the
///   bytes. Every fetch is a zero-copy [`Bytes::slice`].
///   Sync `try_get_range_sync` always succeeds for in-bounds
///   ranges; `get_range` resolves on the same fast path.
/// - [`Source::Lazy`] — a [`LazyByteSource`] behind an
///   `Arc`. Production impls today: mmap-backed
///   [`BytesLazyByteSource`] (zero-copy sync via
///   `try_get_range_sync`), `StorageRangeSource` (cold
///   range-GETs against an object store), or a future
///   broadcast / coalescing source.
///
/// Both variants expose **sync-only** byte access — every
/// public surface in `src/` is sync. The async
/// [`LazyByteSource::range`] is hidden behind
/// [`Source::get_range`]'s `block_in_place + Handle::block_on`
/// / one-shot `current_thread` `Runtime` bridge, the same
/// pattern the supertable's `superfile_reader` uses for its
/// disk-cache fetch path.
///
/// Hot-path callers (`Source::InMemory`, mmap-backed
/// `BytesLazyByteSource`) never hit the bridge — both
/// override [`LazyByteSource::try_get_range_sync`] to
/// return zero-copy slices, so [`Source::get_range`]
/// resolves on the sync fast path.
///
/// `Source: Clone` so `Arc`-shared instances can be handed
/// to multiple readers / supertable segments without
/// forking the underlying state. `Lazy` clones the `Arc`;
/// `InMemory` clones the `Bytes` (refcount bump only, no
/// allocation).
///
/// Lives in `superfile::lazy_source` (cross-reader) rather
/// than under any one reader because it's the *shared*
/// open-time + search-time byte-fetch abstraction across
/// the whole superfile stack. Vector landed it first
/// (013 prerequisite); FTS adopts it in 013 PR4.
#[derive(Clone)]
pub enum Source {
    InMemory(Bytes),
    Lazy(Arc<dyn LazyByteSource>),
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InMemory(b) => f.debug_tuple("InMemory").field(&b.len()).finish(),
            Self::Lazy(_) => f.debug_struct("Lazy").finish_non_exhaustive(),
        }
    }
}

impl Source {
    /// Total backing size in bytes — matches what a single
    /// `get_range(0..len())` would cover.
    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Lazy(s) => s.size() as usize,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sync best-effort fetch. Always succeeds on
    /// `Source::InMemory` (zero-copy `Bytes::slice`); on
    /// `Source::Lazy` returns `Some` only if the range is
    /// already resident in the source's in-process cache
    /// (warm disk cache, mmap-backed `BytesLazyByteSource`,
    /// etc.).
    ///
    /// Returns `None` for out-of-bounds ranges so callers
    /// can distinguish "not available sync" from a hard
    /// error; callers that need a typed error should fall
    /// through to [`Self::get_range`].
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

    /// Sync range fetch with internal async bridging on
    /// cold `Source::Lazy` misses.
    ///
    /// Fast path: [`Self::try_get_range_sync`] (zero-copy
    /// `Bytes::slice` on `InMemory`; same on
    /// `BytesLazyByteSource` / mmap-backed sources). This
    /// covers every production caller today and every
    /// hot-path read at default open
    /// (`Source::Lazy(BytesLazyByteSource over
    /// Bytes::from_owner(mmap))`).
    ///
    /// Cold path (`Source::Lazy` and `try_get_range_sync`
    /// returned `None`): bridge to the source's
    /// `async fn range(...)` via
    /// `block_in_place + Handle::block_on` when there's an
    /// ambient tokio runtime, or build a throwaway
    /// `current_thread` `Runtime` when there isn't. Same
    /// pattern the supertable's `superfile_reader` uses for
    /// its sync disk-cache fetch path. The runtime-build
    /// cost on the no-ambient fallback is ≈ 1 ms —
    /// negligible vs the network round-trip the source is
    /// about to issue. In production the supertable always
    /// has an ambient runtime, so the no-ambient branch
    /// fires only in standalone tests / scripts.
    ///
    /// Convention: every public surface in `src/` stays
    /// sync, async is hidden behind well-defined bridge
    /// points. `Source::get_range` is one of those bridge
    /// points.
    pub fn get_range(&self, range: Range<usize>) -> Result<Bytes, LazyByteSourceError> {
        if let Some(bytes) = self.try_get_range_sync(range.clone()) {
            return Ok(bytes);
        }
        let Self::Lazy(s) = self else {
            // `Source::InMemory` always satisfies `try_get_range_sync`
            // for in-bounds ranges. Reaching this arm means the
            // request was out of bounds.
            return Err(LazyByteSourceError::OutOfBounds {
                start: range.start as u64,
                len: range.len() as u64,
                size: self.len() as u64,
            });
        };
        let start = range.start as u64;
        let len = range.len() as u64;
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(s.range(start, len))),
            Err(_) => {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| {
                        LazyByteSourceError::Storage(crate::storage::StorageError::Permanent {
                            uri: "lazy-source://superfile".to_string(),
                            source: Box::new(std::io::Error::other(format!(
                                "tokio runtime build for lazy source fetch: {e}"
                            ))),
                        })
                    })?;
                rt.block_on(s.range(start, len))
            }
        }
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

    /// Counting wrapper used by SubLazyByteSource tests + by
    /// downstream tests that assert per-blob byte / range
    /// budgets. Records every `range` / `try_get_range_sync`
    /// invocation as (start, len).
    struct CountingLazySource {
        inner: BytesLazyByteSource,
        ranges: std::sync::Mutex<Vec<(u64, u64)>>,
        sync_ranges: std::sync::Mutex<Vec<(u64, u64)>>,
    }

    impl CountingLazySource {
        fn new(bytes: Bytes) -> Self {
            Self {
                inner: BytesLazyByteSource::new(bytes),
                ranges: Default::default(),
                sync_ranges: Default::default(),
            }
        }
        fn ranges(&self) -> Vec<(u64, u64)> {
            self.ranges.lock().expect("ranges mutex poisoned").clone()
        }
        fn sync_ranges(&self) -> Vec<(u64, u64)> {
            self.sync_ranges
                .lock()
                .expect("sync_ranges mutex poisoned")
                .clone()
        }
    }

    #[async_trait]
    impl LazyByteSource for CountingLazySource {
        fn size(&self) -> u64 {
            self.inner.size()
        }
        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.ranges
                .lock()
                .expect("ranges mutex poisoned")
                .push((start, len));
            self.inner.range(start, len).await
        }
        fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
            self.sync_ranges
                .lock()
                .expect("sync_ranges mutex poisoned")
                .push((start, len));
            self.inner.try_get_range_sync(start, len)
        }
    }

    #[tokio::test]
    async fn sub_lazy_byte_source_translates_offsets_into_parent() {
        // Parent payload: 0..32, sub window [10..22) → length 12.
        let payload = Bytes::from((0u8..32).collect::<Vec<u8>>());
        let counting = Arc::new(CountingLazySource::new(payload.clone()));
        let sub = SubLazyByteSource::new(Arc::clone(&counting) as Arc<dyn LazyByteSource>, 10, 12)
            .expect("sub window in bounds");
        assert_eq!(sub.size(), 12);

        // range(0, 4) -> parent.range(10, 4) -> bytes 10..14
        let r0 = sub.range(0, 4).await.expect("sub range start");
        assert_eq!(r0.as_ref(), &payload[10..14]);
        // range(8, 4) -> parent.range(18, 4) -> bytes 18..22
        let r1 = sub.range(8, 4).await.expect("sub range end");
        assert_eq!(r1.as_ref(), &payload[18..22]);

        // Calls reach the parent at the translated offsets, not the
        // sub-relative ones.
        let recorded = counting.ranges();
        assert_eq!(recorded, vec![(10, 4), (18, 4)]);

        // sync path mirrors the async one — offsets are
        // translated into the parent's coordinate space, not the
        // sub-relative ones.
        let s0 = sub.try_get_range_sync(2, 6).expect("sub sync");
        assert_eq!(s0.as_ref(), &payload[12..18]);
        assert_eq!(counting.sync_ranges(), vec![(12, 6)]);
    }

    #[tokio::test]
    async fn sub_lazy_byte_source_window_oob_rejected_at_ctor() {
        let parent = Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0u8; 16])))
            as Arc<dyn LazyByteSource>;
        let err = SubLazyByteSource::new(parent, 10, 100).expect_err("window exceeds parent");
        match err {
            LazyByteSourceError::OutOfBounds { start, len, size } => {
                assert_eq!((start, len, size), (10, 100, 16));
            }
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn sub_lazy_byte_source_range_oob_uses_window_size() {
        // Sub window is [4..12); a request that fits in the parent
        // but pokes past the window must error with size = 8 (the
        // window), not 16 (the parent).
        let parent = Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0u8; 16])))
            as Arc<dyn LazyByteSource>;
        let sub = SubLazyByteSource::new(parent, 4, 8).expect("sub in bounds");
        let err = sub.range(6, 4).await.expect_err("sub oob");
        match err {
            LazyByteSourceError::OutOfBounds { start, len, size } => {
                assert_eq!((start, len, size), (6, 4, 8));
            }
            other => panic!("expected OutOfBounds, got {other:?}"),
        }
        // sync path same shape.
        assert!(sub.try_get_range_sync(6, 4).is_none());
    }
}
