//! Supertable-side [`LazyByteSource`] implementations.
//!
//! The superfile crate owns the trait
//! ([`crate::superfile::LazyByteSource`]). The supertable
//! crate owns the impls that bridge to the storage layer:
//!
//! - [`StorageRangeSource`] wraps an
//!   `Arc<dyn StorageProvider>` so per-query callers can run
//!   `SuperfileReader::open_lazy` against any storage
//!   backend. This is the `ColdFetchMode::RangeOnly` path —
//!   stateless callers that don't want to materialize the
//!   segment in the disk cache.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;

use crate::storage::{StorageError, StorageProvider};
use crate::superfile::{LazyByteSource, LazyByteSourceError};

/// `LazyByteSource` over a `StorageProvider::get_range`.
///
/// Each call to [`range`] issues a fresh `get_range` against
/// the storage backend. Use this for stateless / RangeOnly
/// callers; for steady-state hot reads the disk-cache store
/// is the right path.
///
/// ## Size discovery
///
/// `size` is an `AtomicU64` rather than a plain
/// `u64` so the source can be constructed *without* an
/// up-front HEAD round-trip. The first call to [`tail`] (used
/// by cold-open callers like `read_parquet_metadata_lazy`)
/// issues a suffix-range GET, learns the size from the
/// response, and patches the atomic. Subsequent calls see
/// the cached value via `size()`.
///
/// When the size *is* known up-front (the disk-cache layer
/// already HEAD'd, or a sync test passes a known length),
/// [`Self::with_known_size`] populates the atomic at
/// construction so `range()` can still bounds-check.
///
/// [`range`]: LazyByteSource::range
/// [`tail`]: LazyByteSource::tail
#[derive(Debug)]
pub struct StorageRangeSource {
    storage: Arc<dyn StorageProvider>,
    /// Storage-side URI of the object (e.g.
    /// `data/seg-<uuid>.sf.parquet`).
    uri: String,
    /// Cached total size. `0` means "not yet known". Set
    /// either at construction ([`Self::with_known_size`] /
    /// [`Self::new`]) or lazily on the first [`tail`] call.
    ///
    /// [`tail`]: LazyByteSource::tail
    size: AtomicU64,
}

impl StorageRangeSource {
    /// Construct + cache the object's total size. One HEAD
    /// round-trip up-front; subsequent `range` calls each do
    /// their own GET-range.
    pub async fn new(
        storage: Arc<dyn StorageProvider>,
        uri: impl Into<String>,
    ) -> Result<Self, StorageError> {
        let uri: String = uri.into();
        let meta = storage.head(&uri).await?;
        Ok(Self {
            storage,
            uri,
            size: AtomicU64::new(meta.size),
        })
    }

    /// Construct with a caller-provided size. Used by
    /// `DiskCacheStore::cold_fetch_lazy` when the cache layer
    /// has already issued a HEAD (legacy path; M5 callers
    /// prefer [`Self::with_unknown_size`] to skip the HEAD
    /// entirely).
    pub fn with_known_size(
        storage: Arc<dyn StorageProvider>,
        uri: impl Into<String>,
        size: u64,
    ) -> Self {
        Self {
            storage,
            uri: uri.into(),
            size: AtomicU64::new(size),
        }
    }

    /// construct without an up-front size.
    ///
    /// The size is discovered lazily on the first
    /// [`LazyByteSource::tail`] call (which uses a native
    /// suffix-range GET that returns size in the response).
    /// Callers that rely on `size()` being non-zero before
    /// any I/O happens must use [`Self::new`] or
    /// [`Self::with_known_size`] instead.
    ///
    /// Cold-open is the canonical caller: it starts with a
    /// parquet-footer `tail()` call which both fetches the
    /// bytes and patches the size in one round-trip,
    /// saving an entire HEAD vs. [`Self::new`].
    pub fn with_unknown_size(storage: Arc<dyn StorageProvider>, uri: impl Into<String>) -> Self {
        Self {
            storage,
            uri: uri.into(),
            size: AtomicU64::new(0),
        }
    }

    /// Storage URI this source pulls from. Useful for tests
    /// and observability.
    pub fn uri(&self) -> &str {
        &self.uri
    }
}

#[async_trait]
impl LazyByteSource for StorageRangeSource {
    fn size(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        let known = self.size.load(Ordering::Acquire);
        // Only bounds-check when the size is known. With
        // `with_unknown_size`, the first range call may
        // legitimately precede the discovery `tail()`; we
        // trust the underlying storage to surface OOB as a
        // typed `StorageError`.
        if known > 0 && start.saturating_add(len) > known {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: known,
            });
        }
        let range = start..(start + len);
        // `StorageError` -> `LazyByteSourceError::Storage`
        // via the `#[from]` impl — typed propagation, no
        // stringification.
        Ok(self.storage.get_range(&self.uri, range).await?)
    }

    /// single-RTT tail fetch.
    ///
    /// Routes through `StorageProvider::tail`, which on S3
    /// uses a native suffix-range GET so the response carries
    /// both the bytes AND the total object size. The first
    /// `tail()` call on a [`Self::with_unknown_size`] source
    /// patches the cached size atomic, so subsequent
    /// `range()` callers get the same bounds-checking
    /// behavior as if the source had been constructed with
    /// a known size.
    async fn tail(&self, len: u64) -> Result<(Bytes, u64), LazyByteSourceError> {
        let (bytes, total) = self.storage.tail(&self.uri, len).await?;
        // Patch the size atomic if this was the first call
        // against an `with_unknown_size` source. Use
        // `store(Release)` rather than CAS — concurrent
        // `tail` calls would all observe the same total, so
        // a last-writer-wins store is correct.
        self.size.store(total, Ordering::Release);
        Ok((bytes, total))
    }
}
