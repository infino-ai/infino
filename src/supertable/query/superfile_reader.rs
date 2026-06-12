// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Tiered superfile-bytes lookup.
//!
//! [`superfile_reader`] is the single accessor the query paths
//! (`bm25_search`, `vector_search`, `query_sql`) use to turn a
//! `SuperfileUri` into an `Arc<SuperfileReader>`. The policy:
//!
//!   1. **In-memory tier first.** If `store.reader(uri)`
//!      succeeds — i.e., this process's writer recently
//!      published the superfile and the bytes are still in
//!      `InMemoryReaderCache` — return that reader. Fast
//!      path; no syscalls.
//!   2. **Disk cache fallback.** Miss in the in-memory tier
//!      AND a `DiskCacheStore` is attached →
//!      `DiskCacheStore::reader(uri)` (`await`ed directly).
//!      The cache itself handles cold-fetch from object
//!      storage, pwrite to the local cache directory, and
//!      mmap.
//!   3. **No cache.** Miss in the in-memory tier and no
//!      cache attached → surface the in-memory tier's
//!      `ReaderCacheError::NotFound`. The in-process-only
//!      path; supports callers without storage attached.
//!
//! The accessor is `async`: the query paths
//! (`SupertableReader::vector_search` / `bm25_search` /
//! `query_sql`) are themselves async and run on the owning
//! tokio runtime, so the cold object-store fetch the cache
//! issues is driven by that runtime's reactor — no sync
//! bridge, no throwaway `current_thread` runtime, and
//! object-store retries fire correctly.

use std::sync::Arc;

use crate::storage::StorageProvider;
use crate::superfile::SuperfileReader;
use crate::supertable::manifest::{SubsectionOffsets, SuperfileUri};
use crate::supertable::reader_cache::DiskCacheStore;
use crate::supertable::reader_cache::disk::DiskCacheError;
use crate::supertable::reader_cache::{ReaderCacheError, SuperfileReaderCache};

/// Look up `uri`'s `SuperfileReader`, preferring the in-
/// memory tier and falling back to the disk cache when
/// configured. See the module-level docs for the precise
/// policy.
///
/// `offsets` is an optional pre-known layout hint
/// pulled from the manifest's [`SubsectionOffsets`]. When `Some`
/// the disk-cache cold-fetch path fires the parquet-footer,
/// vector subsection, and FTS subsection GETs **in parallel**
/// (1 RTT cold open) instead of doing the parquet footer first
/// and the subsection fetches second (2 RTTs). `None` falls back
/// to the 2-RTT path — same shape, slower.
pub async fn superfile_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    storage: Option<&Arc<dyn StorageProvider>>,
    uri: &SuperfileUri,
    offsets: Option<&SubsectionOffsets>,
) -> Result<Arc<SuperfileReader>, ReaderCacheError> {
    // 1. In-memory tier.
    match store.reader(uri) {
        Ok(r) => return Ok(r),
        Err(ReaderCacheError::NotFound { .. }) => {
            // Fall through to the cache.
        }
        Err(other) => return Err(other),
    }

    // 2. Disk cache fallback (when attached).
    if let Some(cache) = disk_cache {
        match cache.reader_with_hints(uri, offsets).await {
            Ok(reader) => return Ok(reader),
            // Cache can't admit this superfile (e.g. it's larger than the
            // whole budget). Stream it directly via range GETs instead
            // of failing the query.
            Err(DiskCacheError::BudgetExceeded) => {
                return cache
                    .open_range_only(uri, offsets)
                    .await
                    .map_err(cache_open_failed);
            }
            Err(e) => return Err(cache_open_failed(e)),
        }
    }

    // 3. Storage-only fallback. This covers reopened LocalFs/S3
    // handles configured with durable storage but no disk cache.
    // It is intentionally whole-object: callers who need bounded
    // memory attach `DiskCacheStore`, which uses lazy/range opens.
    if let Some(storage) = storage {
        let path = uri.storage_path();
        let (bytes, _) = storage
            .get(&path)
            .await
            .map_err(|e| ReaderCacheError::OpenFailed {
                source: crate::superfile::ReadError::Io(std::io::Error::other(format!(
                    "storage fetch {path}: {e}"
                ))),
            })?;
        let reader = SuperfileReader::open(bytes)
            .map_err(|source| ReaderCacheError::OpenFailed { source })?;
        return Ok(Arc::new(reader));
    }

    Err(ReaderCacheError::NotFound { uri: *uri })
}

fn cache_open_failed(e: DiskCacheError) -> ReaderCacheError {
    ReaderCacheError::OpenFailed {
        source: crate::superfile::ReadError::Io(std::io::Error::other(format!(
            "disk cache fetch: {e}"
        ))),
    }
}
