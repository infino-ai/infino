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
///
/// Applies only to *present* sidecars (a segment that actually
/// has tombstones); absent/empty sidecars use
/// [`DEFAULT_NEGATIVE_TTL`].
pub const DEFAULT_REFRESH_TTL: Duration = Duration::from_secs(1);

/// Negative/empty-view refresh interval — much longer than the
/// positive TTL. The overwhelmingly common case is a segment
/// with **no** tombstones at all (no sidecar on storage → a 404,
/// cached as an empty bitmap). Re-GETting that 404 on the 1 s
/// positive TTL turns every steady-state query into one
/// object-store round trip *per segment*, which at high segment
/// counts (a wide supertable fan-out) dominates the hot path —
/// the serial post-search tombstone sweep becomes seconds.
///
/// A sidecar only appears when some process deletes a row in that
/// segment. This process invalidates its own deletes synchronously
/// (see [`SidecarCache::invalidate`]), so the only staleness this
/// TTL governs is a *cross-process* delete — already an
/// eventual-consistency concern, not a correctness one. Holding
/// the negative view far longer than the positive one keeps the
/// hot path GET-free in the no-deletes steady state while bounding
/// cross-process delete visibility to this window.
pub const DEFAULT_NEGATIVE_TTL: Duration = Duration::from_secs(60);

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
    negative_ttl: Duration,
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
            negative_ttl: DEFAULT_NEGATIVE_TTL,
            wal_store,
        }
    }

    /// The freshness window for a given cached view: the short
    /// positive TTL for present (non-empty) sidecars, the long
    /// negative TTL for the absent/empty "no tombstones" view.
    fn ttl_for_empty(&self, is_empty: bool) -> Duration {
        if is_empty {
            self.negative_ttl
        } else {
            self.refresh_ttl
        }
    }

    /// `true` when `superfile_id` has no cached view or its view is
    /// past the applicable freshness window.
    fn needs_refresh(&self, superfile_id: Uuid, now: Instant) -> bool {
        match self.inner.get(&superfile_id) {
            Some(entry) => {
                now.duration_since(entry.last_checked)
                    >= self.ttl_for_empty(entry.bitmap.is_empty())
            }
            None => true,
        }
    }

    /// Concurrently refresh every id whose cached view is missing or
    /// stale, so a subsequent per-segment [`Self::bitmap_for`] sweep
    /// is all cache hits.
    ///
    /// This is the hot-path entry point for a wide fan-out: it
    /// replaces N *serial* blocking storage GETs (one per segment,
    /// each a sync→async bridge) with a single *concurrent* batch
    /// whose wall cost is ≈ one round trip rather than N. Ids that
    /// are already fresh are skipped, so in the no-deletes steady
    /// state (every view negative and within [`DEFAULT_NEGATIVE_TTL`])
    /// this issues zero GETs. A per-id refresh error is left for the
    /// later [`Self::bitmap_for`] call to surface; the batch never
    /// fails as a whole.
    pub async fn prefetch(&self, superfile_ids: &[Uuid], now: Instant) {
        let stale: Vec<Uuid> = superfile_ids
            .iter()
            .copied()
            .filter(|id| self.needs_refresh(*id, now))
            .collect();
        if stale.is_empty() {
            return;
        }
        let fetches = stale.into_iter().map(|id| {
            let wal_store = self.wal_store.clone();
            async move { (id, wal_store.get_tombstones(id).await) }
        });
        let results = futures::future::join_all(fetches).await;
        for (id, result) in results {
            let (bitmap, etag) = match result {
                Ok(Some((sidecar, etag))) => (Arc::new(sidecar.bitmap), Some(etag)),
                Ok(None) => (Arc::new(RoaringBitmap::new()), None),
                // Leave any prior entry untouched; the serial
                // bitmap_for fallback re-attempts and surfaces the
                // error if this id is actually consulted.
                Err(_) => continue,
            };
            self.inner.insert(
                id,
                CachedSidecar {
                    etag,
                    bitmap,
                    last_checked: now,
                },
            );
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
        // Hot path: cached and within the freshness window (long
        // for the absent/empty "no tombstones" view, short for a
        // present sidecar).
        if let Some(entry) = self.inner.get(&superfile_id)
            && now.duration_since(entry.last_checked) < self.ttl_for_empty(entry.bitmap.is_empty())
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prefetch_populates_all_ids_in_one_batch() {
        let (_dir, ws, cache) = fixture();
        // One present sidecar, the rest absent (404).
        let present = Uuid::from_u128(0x01);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(9);
        ws.put_tombstones(present, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        let ids: Vec<Uuid> = std::iter::once(present)
            .chain((2..32u128).map(Uuid::from_u128))
            .collect();

        let now = Instant::now();
        cache.prefetch(&ids, now).await;

        // Every id is now cached, so a follow-up sweep is GET-free.
        assert_eq!(cache.len(), ids.len());
        for &id in &ids {
            assert!(!cache.needs_refresh(id, now), "id {id} should be fresh");
        }
        assert_eq!(
            cache
                .bitmap_for(present, now)
                .expect("present")
                .iter()
                .collect::<Vec<_>>(),
            vec![9u32]
        );
        assert!(cache.bitmap_for(ids[1], now).expect("absent").is_empty());
    }

    #[test]
    fn empty_view_uses_long_negative_ttl() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(storage);
        // A 1 ms positive TTL would make a present sidecar instantly
        // stale, but an absent/empty view must ride the much longer
        // negative TTL so the no-deletes hot path stays GET-free.
        let cache = SidecarCache::new(ws, Duration::from_millis(1));
        assert_eq!(cache.ttl_for_empty(true), DEFAULT_NEGATIVE_TTL);
        assert_eq!(cache.ttl_for_empty(false), Duration::from_millis(1));
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
