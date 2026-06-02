//! Per-process reader-side cache of per-superfile tombstone
//! bitmaps.
//!
//! ## Why a cache
//!
//! The reader's per-superfile filter has to know which doc-ids are
//! tombstoned before it can drop them from result sets. The
//! source of truth is `superfiles/<superfile_id>.tombstones` on
//! object storage. Hitting storage on every query would dominate
//! the hot path; the cache holds a [`RoaringBitmap`] per superfile
//! and refreshes it on a coarse TTL so steady-state cost is
//! ~30 ns per superfile per query (a DashMap lookup + an
//! `is_empty` check).
//!
//! ## Freshness model
//!
//! Each cache entry carries a `last_checked: Instant`. On lookup:
//!
//! - If the entry exists AND `now - last_checked < refresh_ttl`,
//!   return the cached bitmap directly. Hot path; no I/O.
//! - Otherwise, refresh from storage. On the refresh:
//!     - 404 → cache as `{etag: None, bitmap: empty}`.
//!     - 200 → cache as `{etag: Some(...), bitmap: parsed}`.
//!
//! Stale tombstones are an eventual-consistency concern, not a
//! correctness one: a query against a freshly-tombstoned row may
//! return it once before the next refresh window closes.
//!
//! ## Writer-side invalidation
//!
//! When this process's own tombstone-phase writer CAS-PUTs a
//! sidecar, it calls [`SidecarCache::invalidate`] so the next
//! query sees the new bitmap immediately. Other processes pick
//! up the change on their next refresh.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::runtime_bridge::bridge_sync_to_async;
use crate::supertable::wal::WalStore;

/// Default refresh interval — 1 second. Bounds how stale the
/// cache's view can be on a query path that didn't write its
/// own tombstones. Tuned to amortize the post-TTL refresh's
/// extra storage GET across enough queries that the steady-
/// state per-query cost stays inside the hot-path budget.
pub const DEFAULT_REFRESH_TTL: Duration = Duration::from_secs(1);

/// Typed failures from cache refresh. The cache's hot path is
/// infallible; this only surfaces when a TTL miss / first-miss
/// refresh has to hit storage and fails.
#[derive(Debug, thiserror::Error)]
pub enum SidecarCacheError {
    /// Underlying storage failed (network blip, throttling, codec
    /// error). The cache leaves the previous entry (if any)
    /// untouched so a subsequent retry has a clean shot.
    #[error("tombstone sidecar refresh failed for {superfile_id}: {message}")]
    RefreshFailed { superfile_id: Uuid, message: String },
}

/// Per-process tombstone-sidecar cache. Owned by `SupertableInner`
/// when storage is attached; absent otherwise (in-memory-only
/// supertables have no sidecars to cache).
///
/// Cheap to `Arc`-share across the query paths. The
/// [`DashMap`] sharding makes per-superfile lookups
/// concurrency-safe without a per-cache lock.
#[derive(Debug)]
pub struct SidecarCache {
    inner: DashMap<Uuid, CachedSidecar>,
    refresh_ttl: Duration,
    wal_store: WalStore,
}

/// One cached entry. `etag` is the storage-layer etag returned
/// on the last successful GET; reserved for the eventual
/// conditional-GET optimization that turns warm-but-stale TTL
/// misses into 304s instead of full body fetches.
///
/// `bitmap` is `Arc`-wrapped so the cache can hand out the
/// shared snapshot without cloning the bytes on every read.
#[derive(Debug, Clone)]
struct CachedSidecar {
    #[allow(dead_code)]
    etag: Option<String>,
    bitmap: Arc<RoaringBitmap>,
    last_checked: Instant,
}

impl SidecarCache {
    /// Construct a cache backed by the supplied [`WalStore`].
    /// `refresh_ttl` bounds how stale the cache's view can be;
    /// pass [`DEFAULT_REFRESH_TTL`] unless you have a specific
    /// reason to deviate.
    pub fn new(wal_store: WalStore, refresh_ttl: Duration) -> Self {
        Self {
            inner: DashMap::new(),
            refresh_ttl,
            wal_store,
        }
    }

    /// Return the current bitmap for `superfile_id`. Hot path:
    /// O(1) DashMap lookup + a TTL check. Cold path:
    /// sync-bridges to the async storage GET via the same
    /// `block_in_place + block_on` pattern the rest of the
    /// query layer uses; falls through to a fresh
    /// `current_thread` runtime when called from outside any
    /// tokio context (e.g., a rayon worker).
    ///
    /// `now` is hoisted to the caller so a per-query
    /// `Instant::now()` is amortized across every per-superfile
    /// lookup in that query.
    pub fn bitmap_for(
        &self,
        superfile_id: Uuid,
        now: Instant,
    ) -> Result<Arc<RoaringBitmap>, SidecarCacheError> {
        // Hot path: cached and within the freshness window.
        if let Some(entry) = self.inner.get(&superfile_id)
            && now.duration_since(entry.last_checked) < self.refresh_ttl
        {
            return Ok(Arc::clone(&entry.bitmap));
        }

        // Refresh from storage.
        self.refresh(superfile_id)
    }

    /// Drop any cached entry for `superfile_id`. Called by this
    /// process's tombstone-phase writer after a successful sidecar
    /// CAS-PUT so the next query in the same process sees the
    /// freshly-landed bit immediately, without waiting for the
    /// TTL window to close.
    pub fn invalidate(&self, superfile_id: Uuid) {
        self.inner.remove(&superfile_id);
    }

    /// Drop every cached entry. Useful for tests; also for any
    /// future code path that wants to force a wholesale refresh.
    #[cfg(test)]
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Number of cached entries. Exposed for tests and for the
    /// overhead bench to confirm the cache reaches the expected
    /// shape (e.g., one entry per superfile post-warmup).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Refresh `superfile_id` from storage. Caches the new
    /// bitmap + etag (or the "known 404" sentinel) and returns
    /// the bitmap.
    fn refresh(&self, superfile_id: Uuid) -> Result<Arc<RoaringBitmap>, SidecarCacheError> {
        let wal_store = self.wal_store.clone();
        let result =
            bridge_sync_to_async(async move { wal_store.get_tombstones(superfile_id).await });

        let (bitmap, etag) = match result {
            Ok(Some((sidecar, etag))) => (Arc::new(sidecar.bitmap), Some(etag)),
            Ok(None) => (Arc::new(RoaringBitmap::new()), None),
            Err(e) => {
                return Err(SidecarCacheError::RefreshFailed {
                    superfile_id,
                    message: format!("{e}"),
                });
            }
        };

        self.inner.insert(
            superfile_id,
            CachedSidecar {
                etag,
                bitmap: Arc::clone(&bitmap),
                last_checked: Instant::now(),
            },
        );

        Ok(bitmap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use crate::supertable::wal::tombstones_codec::TombstonesSidecar;
    use tempfile::TempDir;

    fn fixture() -> (TempDir, WalStore, SidecarCache) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = SidecarCache::new(ws.clone(), DEFAULT_REFRESH_TTL);
        (dir, ws, cache)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn first_lookup_against_absent_sidecar_returns_empty_bitmap() {
        let (_dir, _ws, cache) = fixture();
        let now = Instant::now();
        let bitmap = cache
            .bitmap_for(Uuid::from_u128(0xAB), now)
            .expect("lookup");
        assert!(bitmap.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lookup_reflects_persisted_sidecar() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xCAFE);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(3);
        bitmap.insert(5);
        let sidecar = TombstonesSidecar { seal: None, bitmap };
        ws.put_tombstones(sf_id, None, &sidecar).await.expect("put");

        let cached = cache.bitmap_for(sf_id, Instant::now()).expect("lookup");
        let collected: Vec<u32> = cached.iter().collect();
        assert_eq!(collected, vec![1u32, 3, 5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn second_lookup_within_ttl_skips_refresh() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xDEAD);
        // First lookup primes the cache as "known 404".
        let now = Instant::now();
        let _ = cache.bitmap_for(sf_id, now).expect("warm");

        // Now write a sidecar; without invalidation the cache
        // should still return the original empty view within
        // the TTL.
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(42);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");

        let cached = cache.bitmap_for(sf_id, now).expect("warm read");
        assert!(cached.is_empty(), "cache must hold the pre-write view");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn invalidate_forces_next_lookup_to_refresh() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xBEEF);
        let now = Instant::now();
        let _ = cache.bitmap_for(sf_id, now).expect("warm");

        // Write a sidecar then invalidate.
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(7);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.invalidate(sf_id);

        let cached = cache.bitmap_for(sf_id, now).expect("re-read");
        let collected: Vec<u32> = cached.iter().collect();
        assert_eq!(collected, vec![7u32]);
    }

    #[test]
    fn cache_is_empty_on_construction() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(storage);
        let cache = SidecarCache::new(ws, DEFAULT_REFRESH_TTL);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }
}
