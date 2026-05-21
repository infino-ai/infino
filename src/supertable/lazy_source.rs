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
/// [`range`]: LazyByteSource::range
#[derive(Debug)]
pub struct StorageRangeSource {
    storage: Arc<dyn StorageProvider>,
    /// Storage-side URI of the object (e.g.
    /// `data/seg-<uuid>.sf`).
    uri: String,
    /// Cached total size — retrieved once at construction
    /// via `storage.head(uri)` so `LazyByteSource::size()`
    /// can be sync.
    size: u64,
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
            size: meta.size,
        })
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
        self.size
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if start.saturating_add(len) > self.size {
            return Err(LazyByteSourceError::OutOfBounds {
                start,
                len,
                size: self.size,
            });
        }
        let range = start..(start + len);
        // `StorageError` -> `LazyByteSourceError::Storage`
        // via the `#[from]` impl — typed propagation, no
        // stringification.
        Ok(self.storage.get_range(&self.uri, range).await?)
    }
}
