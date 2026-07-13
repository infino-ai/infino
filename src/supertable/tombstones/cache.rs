// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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
//! and validates it against the manifest's tombstone-seq map so
//! steady-state cost is ~30 ns per superfile per query (a DashMap
//! lookup + a seq compare).
//!
//! ## Freshness model
//!
//! The manifest's `tombstone_seqs` map is the single freshness
//! authority: the mutation pipeline stamps `superfile_id →
//! manifest_id` into the manifest right after its sidecar CAS-PUTs
//! land, so the map names exactly which sidecars exist and which
//! version of each a reader should hold. The cache keeps the view
//! current via [`SidecarCache::reconcile`], called wherever a new
//! manifest is swapped in. On lookup:
//!
//! - Superfile absent from the map → **no sidecar exists**; return
//!   the shared empty bitmap. No entry, no storage GET, ever.
//! - Cached entry's seq matches the map → return it. Hot path;
//!   no I/O and no TTL — an unchanged sidecar is fresh forever.
//! - Otherwise (no entry, or the map moved past it) → refresh from
//!   storage and record the map's seq on the new entry.
//!
//! Cross-process delete visibility is therefore bounded by manifest
//! freshness (the read-consistency window), not by a cache TTL.
//!
//! ## Seal freshness
//!
//! Sealing (compaction stamping a [`SealRecord`] onto a sidecar)
//! changes sidecar content without going through the tombstone
//! phase, so it does not bump the seq. The seal-consuming lookups
//! ([`SidecarCache::seal_for`], [`SidecarCache::sidecar_for`])
//! additionally bound entry age by [`DEFAULT_SEAL_TTL`] so a
//! cross-process seal is observed within that window.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use arc_swap::ArcSwap;
use dashmap::DashMap;
use futures::future::join_all;
use roaring::RoaringBitmap;
use uuid::Uuid;

use crate::{
    runtime_bridge::bridge_sync_to_async,
    supertable::wal::{SealRecord, WalStore},
};

/// Freshness bound for seal-consuming lookups — 1 second. A
/// cached entry whose seq matches the manifest is fresh forever
/// for bitmap reads, but a seal can land without a seq bump, so
/// [`SidecarCache::seal_for`] / [`SidecarCache::sidecar_for`]
/// re-GET entries older than this. Coarse enough to amortize
/// across compaction's selection sweeps.
pub const DEFAULT_SEAL_TTL: Duration = Duration::from_secs(1);

/// Reader-side snapshot of the manifest's per-superfile tombstone
/// versions. Built from
/// [`ManifestSnapshot::get_tombstone_seqs`](crate::supertable::manifest::ManifestSnapshot::get_tombstone_seqs)
/// wherever a new manifest is swapped in, and pushed into the
/// cache via [`SidecarCache::reconcile`].
#[derive(Debug, Default)]
pub struct TombstoneSeqView {
    /// `manifest_id` of the manifest this view was taken from.
    /// [`SidecarCache::reconcile`] is forward-only on this field,
    /// so a racing stale refresh can never regress the view.
    pub manifest_id: u64,
    /// `superfile_id →` seq of that superfile's sidecar (the
    /// `manifest_id` of the commit whose tombstone phase last
    /// changed it). Absence means the superfile has no sidecar.
    pub seqs: BTreeMap<Uuid, u64>,
}

/// Typed failures from cache refresh. The cache's hot path is
/// infallible; this only surfaces when a seq-mismatch / first-miss
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
    seal_ttl: Duration,
    seq_view: ArcSwap<TombstoneSeqView>,
    /// Shared "no tombstones" bitmap handed out for superfiles
    /// absent from the seq map, so the by-far-common case
    /// allocates nothing.
    empty_bitmap: Arc<RoaringBitmap>,
    wal_store: WalStore,
}

/// One cached entry. `etag` is the storage-layer etag returned
/// on the last successful GET; reserved for the eventual
/// conditional-GET optimization that turns seq-stale refreshes
/// into 304s instead of full body fetches.
///
/// `bitmap` is `Arc`-wrapped so the cache can hand out the
/// shared snapshot without cloning the bytes on every read.
/// `seal` is cached to enable compaction selection to check
/// sealed status without a storage roundtrip.
#[derive(Debug, Clone)]
struct CachedSidecar {
    #[allow(dead_code)]
    etag: Option<String>,
    bitmap: Arc<RoaringBitmap>,
    seal: Option<SealRecord>,
    /// The seq-view value this entry was fetched under. The entry
    /// is bitmap-fresh exactly while this matches the current
    /// view's seq for the superfile.
    seq: u64,
    /// When the entry was fetched — bounds seal staleness (see
    /// [`DEFAULT_SEAL_TTL`]); bitmap reads ignore it.
    fetched_at: Instant,
}

/// Which freshness rule a lookup applies (see the module docs).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Freshness {
    /// Seq match alone proves freshness — bitmap-only consumers.
    Bitmap,
    /// Seq match plus a [`DEFAULT_SEAL_TTL`] age bound — seal
    /// consumers, because seals land without a seq bump.
    Seal,
}

impl SidecarCache {
    /// Construct a cache backed by the supplied [`WalStore`],
    /// born with `initial_view` (the tombstone-seq view of the
    /// manifest the owning supertable handle was opened with).
    pub fn new(
        wal_store: WalStore,
        seal_ttl: Duration,
        initial_view: Arc<TombstoneSeqView>,
    ) -> Self {
        Self {
            inner: DashMap::new(),
            seal_ttl,
            seq_view: ArcSwap::new(initial_view),
            empty_bitmap: Arc::new(RoaringBitmap::new()),
            wal_store,
        }
    }

    /// Install a newer seq view. Forward-only on
    /// `view.manifest_id` — a concurrent older view (e.g. a stale
    /// refresh racing the writer's own stamp) is dropped, so the
    /// cache's freshness authority never moves backwards.
    pub fn reconcile(&self, view: Arc<TombstoneSeqView>) {
        loop {
            let current = self.seq_view.load();
            if view.manifest_id <= current.manifest_id {
                return;
            }
            let prev = self.seq_view.compare_and_swap(&*current, Arc::clone(&view));
            if Arc::ptr_eq(&prev, &current) {
                return;
            }
        }
    }

    /// `manifest_id` of the currently-installed seq view. Callers
    /// use this to skip building a view the cache would drop.
    pub fn view_manifest_id(&self) -> u64 {
        self.seq_view.load().manifest_id
    }

    /// Concurrently refresh every id whose cached view is missing or
    /// seq-stale, so a subsequent per-superfile [`Self::bitmap_for`]
    /// sweep is all cache hits.
    ///
    /// This is the hot-path entry point for a wide fan-out: it
    /// replaces N *serial* blocking storage GETs (one per superfile,
    /// each a sync→async bridge) with a single *concurrent* batch
    /// whose wall cost is ≈ one round trip rather than N. Ids absent
    /// from the seq map have no sidecar and ids whose entry matches
    /// the map are already fresh, so in the no-new-deletes steady
    /// state this issues zero GETs. A per-id refresh error is left
    /// for the later [`Self::bitmap_for`] call to surface; the batch
    /// never fails as a whole.
    pub async fn prefetch(&self, superfile_ids: &[Uuid], now: Instant) {
        let view = self.seq_view.load();
        let stale: Vec<(Uuid, u64)> = superfile_ids
            .iter()
            .filter_map(|id| {
                let expected = view.seqs.get(id).copied()?;
                match self.inner.get(id) {
                    Some(entry) if entry.seq == expected => None,
                    _ => Some((*id, expected)),
                }
            })
            .collect();
        if stale.is_empty() {
            return;
        }
        let fetches = stale.into_iter().map(|(id, expected)| {
            let wal_store = self.wal_store.clone();
            async move { (id, expected, wal_store.get_tombstones(id).await) }
        });
        let results = join_all(fetches).await;
        for (id, expected, result) in results {
            let (bitmap, seal, etag) = match result {
                Ok(Some((sidecar, etag))) => (Arc::new(sidecar.bitmap), sidecar.seal, Some(etag)),
                Ok(None) => (Arc::clone(&self.empty_bitmap), None, None),
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
                    seal,
                    seq: expected,
                    fetched_at: now,
                },
            );
        }
    }

    /// Fetch bitmap and seal for `superfile_id` from cache or storage.
    /// Hot path: O(1) seq-map lookup (+ DashMap lookup when a sidecar
    /// exists). Cold path: sync-bridges to the async storage GET.
    /// `now` is hoisted to the caller so a per-query `Instant::now()`
    /// is amortized across every per-superfile lookup in that query.
    fn fetch_sidecar(
        &self,
        superfile_id: Uuid,
        now: Instant,
        freshness: Freshness,
    ) -> Result<(Arc<RoaringBitmap>, Option<SealRecord>), SidecarCacheError> {
        let view = self.seq_view.load();
        let Some(expected) = view.seqs.get(&superfile_id).copied() else {
            // The seq map is authoritative for existence: absent
            // means no sidecar, so there is nothing to fetch.
            return Ok((Arc::clone(&self.empty_bitmap), None));
        };
        if let Some(entry) = self.inner.get(&superfile_id) {
            let seq_fresh = entry.seq == expected;
            let fresh = match freshness {
                Freshness::Bitmap => seq_fresh,
                Freshness::Seal => {
                    seq_fresh && now.duration_since(entry.fetched_at) < self.seal_ttl
                }
            };
            if fresh {
                return Ok((Arc::clone(&entry.bitmap), entry.seal.clone()));
            }
        }

        // Cold path: refresh from storage.
        self.refresh_and_return_sidecar(superfile_id, expected)
    }

    /// Return the current bitmap for `superfile_id`. Hot path:
    /// O(1) lookups + a seq compare. Cold path: sync-bridges to
    /// the async storage GET via the same `block_in_place +
    /// block_on` pattern the rest of the query layer uses; falls
    /// through to a fresh `current_thread` runtime when called
    /// from outside any tokio context (e.g., a rayon worker).
    ///
    /// `now` is hoisted to the caller so a per-query
    /// `Instant::now()` is amortized across every per-superfile
    /// lookup in that query.
    pub fn bitmap_for(
        &self,
        superfile_id: Uuid,
        now: Instant,
    ) -> Result<Arc<RoaringBitmap>, SidecarCacheError> {
        self.fetch_sidecar(superfile_id, now, Freshness::Bitmap)
            .map(|(bitmap, _)| bitmap)
    }

    /// Return the seal record for `superfile_id` if present. Compaction
    /// selection uses this to check sealed status without a storage roundtrip.
    pub fn seal_for(
        &self,
        superfile_id: Uuid,
        now: Instant,
    ) -> Result<Option<SealRecord>, SidecarCacheError> {
        self.fetch_sidecar(superfile_id, now, Freshness::Seal)
            .map(|(_, seal)| seal)
    }

    /// Return both the bitmap and seal for `superfile_id`. Compaction
    /// merge operations use this to fetch complete sidecar state in one
    /// cache lookup.
    pub fn sidecar_for(
        &self,
        superfile_id: Uuid,
        now: Instant,
    ) -> Result<(Arc<RoaringBitmap>, Option<SealRecord>), SidecarCacheError> {
        self.fetch_sidecar(superfile_id, now, Freshness::Seal)
    }

    /// Drop every cached entry. Useful for tests; also for any
    /// future code path that wants to force a wholesale refresh.
    #[cfg(test)]
    pub fn clear(&self) {
        self.inner.clear();
    }

    /// Number of cached entries. Exposed for tests and for the
    /// overhead bench to confirm the cache reaches the expected
    /// shape (e.g., one entry per tombstoned superfile post-warmup).
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `true` when the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Refresh from storage and return both bitmap and seal.
    /// Used by [`Self::fetch_sidecar`] to avoid redundant refresh logic.
    fn refresh_and_return_sidecar(
        &self,
        superfile_id: Uuid,
        expected_seq: u64,
    ) -> Result<(Arc<RoaringBitmap>, Option<SealRecord>), SidecarCacheError> {
        let wal_store = self.wal_store.clone();
        let result =
            bridge_sync_to_async(async move { wal_store.get_tombstones(superfile_id).await });

        let (bitmap, seal, etag) = match result {
            Ok(Some((sidecar, etag))) => (Arc::new(sidecar.bitmap), sidecar.seal, Some(etag)),
            Ok(None) => (Arc::clone(&self.empty_bitmap), None, None),
            Err(e) => {
                return Err(SidecarCacheError::RefreshFailed {
                    superfile_id,
                    message: format!("{e}"),
                });
            }
        };

        let entry = CachedSidecar {
            etag,
            bitmap: Arc::clone(&bitmap),
            seal: seal.clone(),
            seq: expected_seq,
            fetched_at: Instant::now(),
        };

        self.inner.insert(superfile_id, entry);

        Ok((bitmap, seal))
    }
}

#[cfg(test)]
mod tests {
    use std::iter::once;

    use chrono::Utc;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::wal::tombstones_codec::TombstonesSidecar,
    };

    fn fixture() -> (TempDir, WalStore, SidecarCache) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        let cache = SidecarCache::new(
            ws.clone(),
            DEFAULT_SEAL_TTL,
            Arc::new(TombstoneSeqView::default()),
        );
        (dir, ws, cache)
    }

    /// A view at `manifest_id` mapping each id to `seq`.
    fn view(manifest_id: u64, entries: &[(Uuid, u64)]) -> Arc<TombstoneSeqView> {
        Arc::new(TombstoneSeqView {
            manifest_id,
            seqs: entries.iter().copied().collect(),
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unmapped_superfile_is_empty_with_no_entry_and_no_get() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xAB);
        // Even a sidecar physically present on storage is invisible
        // while the seq map doesn't name it — the map is the
        // authority for existence, so no GET is issued at all.
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(11);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");

        let cached = cache.bitmap_for(sf_id, Instant::now()).expect("lookup");
        assert!(cached.is_empty());
        assert_eq!(cache.len(), 0, "no entry is materialized for unmapped ids");
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
        cache.reconcile(view(1, &[(sf_id, 1)]));

        let cached = cache.bitmap_for(sf_id, Instant::now()).expect("lookup");
        let collected: Vec<u32> = cached.iter().collect();
        assert_eq!(collected, vec![1u32, 3, 5]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn matching_seq_serves_cached_view_without_refresh() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xDEAD);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));
        let now = Instant::now();
        let first = cache.bitmap_for(sf_id, now).expect("warm");
        assert_eq!(first.iter().collect::<Vec<_>>(), vec![1u32]);

        // Overwrite the sidecar on storage without bumping the seq:
        // the cache must keep serving the seq-fresh view.
        let (_, etag) = ws
            .get_tombstones(sf_id)
            .await
            .expect("read")
            .expect("present");
        let mut bumped = RoaringBitmap::new();
        bumped.insert(1);
        bumped.insert(2);
        ws.put_tombstones(
            sf_id,
            Some(&etag),
            &TombstonesSidecar {
                seal: None,
                bitmap: bumped,
            },
        )
        .await
        .expect("overwrite");

        let cached = cache.bitmap_for(sf_id, now).expect("cached read");
        assert_eq!(
            cached.iter().collect::<Vec<_>>(),
            vec![1u32],
            "cache must hold the seq-fresh view"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seq_bump_forces_next_lookup_to_refresh() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xBEEF);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(7);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));
        let now = Instant::now();
        let first = cache.bitmap_for(sf_id, now).expect("warm");
        assert_eq!(first.iter().collect::<Vec<_>>(), vec![7u32]);

        // A new delete lands: sidecar grows and the stamp bumps the
        // seq. Reconciling the newer view must force a refetch even
        // though zero wall-clock time has passed.
        let (_, etag) = ws
            .get_tombstones(sf_id)
            .await
            .expect("read")
            .expect("present");
        let mut grown = RoaringBitmap::new();
        grown.insert(7);
        grown.insert(9);
        ws.put_tombstones(
            sf_id,
            Some(&etag),
            &TombstonesSidecar {
                seal: None,
                bitmap: grown,
            },
        )
        .await
        .expect("grow");
        cache.reconcile(view(2, &[(sf_id, 2)]));

        let cached = cache.bitmap_for(sf_id, now).expect("re-read");
        assert_eq!(cached.iter().collect::<Vec<_>>(), vec![7u32, 9]);
    }

    #[test]
    fn reconcile_is_forward_only() {
        let (_dir, _ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0x77);
        cache.reconcile(view(5, &[(sf_id, 5)]));
        assert_eq!(cache.view_manifest_id(), 5);

        // A stale view (racing refresh) must not regress the authority.
        cache.reconcile(view(3, &[]));
        assert_eq!(cache.view_manifest_id(), 5);

        cache.reconcile(view(6, &[(sf_id, 6)]));
        assert_eq!(cache.view_manifest_id(), 6);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn prefetch_populates_mapped_ids_in_one_batch() {
        let (_dir, ws, cache) = fixture();
        // One superfile with a sidecar (mapped); the rest unmapped.
        let present = Uuid::from_u128(0x01);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(9);
        ws.put_tombstones(present, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(present, 1)]));
        let ids: Vec<Uuid> = once(present)
            .chain((2..32u128).map(Uuid::from_u128))
            .collect();

        let now = Instant::now();
        cache.prefetch(&ids, now).await;

        // Only the mapped id materializes an entry; the rest are
        // authoritatively empty with zero GETs.
        assert_eq!(cache.len(), 1);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_for_returns_cached_seal() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xFFFF);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(42);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));

        let now = Instant::now();
        let seal = cache.seal_for(sf_id, now).expect("lookup");
        assert!(seal.is_none(), "initially unsealed");

        // Within the seal TTL, subsequent calls are GET-free.
        let seal_2 = cache.seal_for(sf_id, now).expect("cached");
        assert!(seal_2.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_lookup_past_ttl_observes_new_seal_without_seq_bump() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let ws = WalStore::new(Arc::clone(&storage));
        // Zero seal TTL: every seal_for consults storage, modeling
        // "the TTL window has closed".
        let cache = SidecarCache::new(
            ws.clone(),
            Duration::ZERO,
            Arc::new(TombstoneSeqView::default()),
        );
        let sf_id = Uuid::from_u128(0x5EA1);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(3);
        ws.put_tombstones(
            sf_id,
            None,
            &TombstonesSidecar {
                seal: None,
                bitmap: bitmap.clone(),
            },
        )
        .await
        .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));

        let now = Instant::now();
        assert!(cache.seal_for(sf_id, now).expect("unsealed").is_none());
        // Bitmap reads stay seq-fresh — no GET — even at zero TTL.
        assert_eq!(
            cache
                .bitmap_for(sf_id, now)
                .expect("bitmap")
                .iter()
                .collect::<Vec<_>>(),
            vec![3u32]
        );

        // A compactor seals the sidecar without a seq bump; the next
        // seal read (past the TTL) must observe it.
        let (sidecar, etag) = ws
            .get_tombstones(sf_id)
            .await
            .expect("read")
            .expect("present");
        let sealed = TombstonesSidecar {
            seal: Some(SealRecord {
                compaction_id: Uuid::from_u128(0xC0),
                sealed_at: Utc::now(),
            }),
            bitmap: sidecar.bitmap,
        };
        ws.put_tombstones(sf_id, Some(&etag), &sealed)
            .await
            .expect("seal");

        let seal = cache.seal_for(sf_id, now).expect("sealed");
        assert!(seal.is_some(), "seal observed once the TTL window closed");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sidecar_for_returns_both_bitmap_and_seal() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0xABCD);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(2);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));

        let now = Instant::now();
        let (cached_bitmap, seal) = cache.sidecar_for(sf_id, now).expect("lookup");
        let collected: Vec<u32> = cached_bitmap.iter().collect();
        assert_eq!(collected, vec![1u32, 2]);
        assert!(seal.is_none());
    }

    #[test]
    fn cache_is_empty_on_construction() {
        let (_dir, _ws, cache) = fixture();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    /// `clear` drops every cached entry, returning the cache to its
    /// empty state after a lookup has populated it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn clear_empties_the_cache() {
        let (_dir, ws, cache) = fixture();
        let sf_id = Uuid::from_u128(0x1234);
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        ws.put_tombstones(sf_id, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("put");
        cache.reconcile(view(1, &[(sf_id, 1)]));
        let _ = cache.bitmap_for(sf_id, Instant::now()).expect("lookup");
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty(), "clear drops all entries");
        assert_eq!(cache.len(), 0);
    }
}
