// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! In-process cache of parsed `SuperfileReader`s keyed by
//! [`SuperfileUri`].
//!
//! The supertable's manifest carries metadata only — superfile id,
//! summary stats, FTS bloom, etc. The actual parsed superfile
//! readers live here, behind the [`SuperfileReaderCache`] trait,
//! owned by the supertable inner state and shared across reader
//! threads via `Arc<dyn SuperfileReaderCache>`. This split keeps
//! manifest snapshots cheap (a few KB each) while letting hot
//! queries reuse one parsed reader per superfile across threads.
//!
//! ## Implementations
//!
//! - [`InMemoryReaderCache`] — `Mutex<HashMap<...>>`-backed.
//!   Holds every superfile's bytes resident in RAM for the
//!   supertable's lifetime; no eviction. Fits the in-memory
//!   supertable shape — process restart loses the data, by
//!   design at this layer.
//! - [`DiskCacheStore`] (in [`disk`]) — mmap-backed L2 cache
//!   that pages superfiles in from an underlying storage provider
//!   on miss, evicts via [`LruPolicy`] when over budget. Used
//!   when the supertable is configured with object-store
//!   durability and wants a bounded RAM footprint.
//!
//! Both implementations return `Arc<SuperfileReader>`; they
//! differ in their backing and in their API shape (the in-memory
//! cache exposes the sync trait below; `DiskCacheStore` exposes
//! its own async surface because cold-fetch goes through tokio).
//!
//! Raw byte I/O against durable storage (S3, local FS) lives one
//! layer below in [`crate::storage::StorageProvider`]; the
//! `DiskCacheStore` is built on top of it.

pub mod block_source;
pub mod config;
pub mod disk;
pub mod in_memory;
mod kept_lazy;

use std::sync::Arc;

use bytes::Bytes;
pub use config::{CacheEvictionPolicy, ColdFetchMode, DiskCacheConfig, LruPolicy};
pub(crate) use disk::RunScratch;
pub use disk::{CacheStats, DiskCacheError, DiskCacheStore, OpenTier, ReadIntent};
pub use in_memory::InMemoryReaderCache;
use thiserror::Error;

use super::manifest::SuperfileUri;
use crate::superfile::{ReadError, SuperfileReader};

/// Maps a `SuperfileUri` to a `SuperfileReader`. Owned by the
/// supertable's inner state; shared across all readers via
/// `Arc<dyn SuperfileReaderCache>`.
///
/// All methods take `&self`. Implementations are responsible for
/// their own internal synchronization (the in-memory variant uses
/// `Mutex`; the disk-backed variant uses a concurrent map plus
/// per-URI `OnceCell`s for cold-fetch coalescing). Callers don't
/// acquire any lock externally.
pub trait SuperfileReaderCache: Send + Sync {
    /// Get a reader for `uri`. Implementations cache internally;
    /// the returned `Arc` is shared with concurrent callers
    /// asking for the same URI.
    ///
    /// Returns [`ReaderCacheError::NotFound`] if the URI was
    /// never registered with [`SuperfileReaderCache::insert`].
    fn reader(&self, uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, ReaderCacheError>;

    /// Insert a new superfile's bytes under `uri`. Called once per
    /// superfile by the writer, at commit time.
    ///
    /// Idempotent: re-inserting the same `uri` is a no-op (the
    /// caller's contract is that a `SuperfileUri` always names the
    /// same bytes — superfiles are immutable). Implementations
    /// may skip the bytes parse on the second call entirely.
    ///
    /// Returns [`ReaderCacheError::OpenFailed`] if the bytes
    /// don't parse as a valid superfile.
    fn insert(&self, uri: SuperfileUri, bytes: Bytes) -> Result<(), ReaderCacheError>;

    /// Approximate resident byte count, summed across every
    /// cached superfile. Used by tests + observability that need to
    /// confirm RAM bounds match expectations.
    fn resident_bytes(&self) -> usize;

    /// Drop a superfile's cached bytes once it's no longer referenced
    /// by the manifest (e.g. merged away by compaction).
    ///
    /// Safe under concurrency: a caller already holding an
    /// `Arc<SuperfileReader>` keeps it alive regardless. A later
    /// `reader()` for this `uri` just misses and falls back to
    /// storage, which still has the bytes until `gc()` reclaims them.
    ///
    /// Default no-op — bounded caches (e.g. LRU) don't need this.
    fn remove(&self, _uri: &SuperfileUri) {}

    /// The lazy reader [`Self::keep_lazy`] kept for `uri`, marking it used
    /// and weighing it again, since queries load into it. Default: none is
    /// kept.
    fn lazy(&self, _uri: &SuperfileUri) -> Option<Arc<SuperfileReader>> {
        None
    }

    /// Keep `reader`, a lazy reader over range reads of `uri` in durable
    /// storage, for later opens of `uri` on this table, and return the
    /// reader kept for it: the first one kept wins, so opens racing on one
    /// superfile all share a reader. The reader reads through the storage
    /// handle it was opened with, so only a reader opened with this table's
    /// own storage belongs here. Default: keeps nothing and returns `reader`.
    fn keep_lazy(&self, _uri: SuperfileUri, reader: Arc<SuperfileReader>) -> Arc<SuperfileReader> {
        reader
    }

    /// Charge what [`Self::keep_lazy`] keeps to `budget`, the memory budget
    /// of the connection that opened the table, and let it take that back
    /// when a reservation does not fit. Called when a table opens with its
    /// options. Default: nothing is kept, so nothing is charged.
    fn attach_memory_budget(&self, _budget: &Arc<crate::memory::ConnectionMemoryBudget>) {}
}

/// Error type for [`SuperfileReaderCache`] operations.
#[derive(Debug, Error)]
pub enum ReaderCacheError {
    #[error("superfile uri {uri:?} not found in cache")]
    NotFound { uri: SuperfileUri },

    #[error("failed to open superfile bytes: {source}")]
    OpenFailed {
        #[source]
        source: ReadError,
    },
}

#[cfg(test)]
mod trait_default_tests {
    use super::*;

    /// The trait's default `remove` is a documented safe no-op: a cache
    /// with its own eviction may ignore removal hints entirely, and the
    /// reader contract (miss -> NotFound) is untouched by the call.
    struct IgnoresRemoval;

    impl SuperfileReaderCache for IgnoresRemoval {
        fn reader(&self, uri: &SuperfileUri) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
            Err(ReaderCacheError::NotFound { uri: *uri })
        }
        fn insert(&self, _uri: SuperfileUri, _bytes: Bytes) -> Result<(), ReaderCacheError> {
            Ok(())
        }
        fn resident_bytes(&self) -> usize {
            0
        }
    }

    #[test]
    fn default_remove_is_a_safe_no_op() {
        let cache = IgnoresRemoval;
        let uri = SuperfileUri::new_v4();
        // An implementation may accept an insert and still answer misses —
        // the trait promises nothing about where the bytes went, only that
        // `remove` is safe to call either way.
        cache
            .insert(uri, Bytes::from_static(b"not a superfile"))
            .expect("this implementation accepts any bytes");
        cache.remove(&uri);
        assert!(
            matches!(cache.reader(&uri), Err(ReaderCacheError::NotFound { .. })),
            "remove neither panics nor changes the miss contract"
        );
        assert_eq!(cache.resident_bytes(), 0);
    }
}
