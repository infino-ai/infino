//! Tiered segment-bytes lookup.
//!
//! [`superfile_reader`] is the single accessor the query paths
//! (`bm25_search`, `vector_search`, `query_sql`) use to turn a
//! `SuperfileUri` into an `Arc<SuperfileReader>`. The policy:
//!
//!   1. **In-memory tier first.** If `store.reader(uri)`
//!      succeeds — i.e., this process's writer recently
//!      published the segment and the bytes are still in
//!      `InMemoryReaderCache` — return that reader. Fast
//!      path; no syscalls.
//!   2. **Disk cache fallback.** Miss in the in-memory tier
//!      AND a `DiskCacheStore` is attached →
//!      `DiskCacheStore::reader(uri)` (async). Sync-bridged
//!      via `block_in_place + Handle::block_on` when there's
//!      an ambient tokio runtime; falls through to building
//!      a dedicated one via [`tokio::runtime::Runtime::new`]
//!      otherwise. The cache itself handles cold-fetch from
//!      object storage, pwrite to the local cache directory,
//!      and mmap.
//!   3. **No cache.** Miss in the in-memory tier and no
//!      cache attached → surface the in-memory tier's
//!      `ReaderCacheError::NotFound`. The in-process-only
//!      path; supports callers without storage attached.
//!
//! The accessor is sync to keep call sites in the query
//! paths (which are themselves sync from
//! `SupertableReader::bm25_search` etc.) unchanged. The
//! async-to-sync bridge mirrors the writer's
//! `persist_commit` pattern.

use std::sync::Arc;

use crate::superfile::SuperfileReader;
use crate::supertable::manifest::SuperfileUri;
use crate::supertable::reader_cache::DiskCacheStore;
use crate::supertable::reader_cache::{ReaderCacheError, SuperfileReaderCache};

/// Look up `uri`'s `SuperfileReader`, preferring the in-
/// memory tier and falling back to the disk cache when
/// configured. See the module-level docs for the precise
/// policy.
pub fn superfile_reader(
    store: &Arc<dyn SuperfileReaderCache>,
    disk_cache: Option<&Arc<DiskCacheStore>>,
    uri: &SuperfileUri,
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
    let cache = match disk_cache {
        Some(c) => Arc::clone(c),
        None => return Err(ReaderCacheError::NotFound { uri: *uri }),
    };

    let uri_copy = *uri;
    let result = match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            // Ambient tokio runtime — block_in_place + block_on
            // is the same pattern the writer's commit path uses
            // for its sync→async bridge.
            tokio::task::block_in_place(|| handle.block_on(cache.reader(&uri_copy)))
        }
        Err(_) => {
            // No ambient runtime. Build a tiny one just for
            // this fetch. Cold path; the overhead (≈ 1 ms to
            // create a current_thread Runtime) is negligible
            // compared to the parallel range-fetch the cache
            // is about to issue. Falls out of scope at end of
            // statement.
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    return Err(ReaderCacheError::OpenFailed {
                        source: crate::superfile::ReadError::Io(std::io::Error::other(format!(
                            "tokio runtime build for disk cache fetch: {e}"
                        ))),
                    });
                }
            };
            rt.block_on(cache.reader(&uri_copy))
        }
    };

    result.map_err(|e| ReaderCacheError::OpenFailed {
        source: crate::superfile::ReadError::Io(std::io::Error::other(format!(
            "disk cache fetch: {e}"
        ))),
    })
}
