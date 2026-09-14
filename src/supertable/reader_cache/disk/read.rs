// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The public reader API: hand back a [`SuperfileReader`] for a URI, serving it
//! from memory or disk when it is cached and cold-fetching it when it is not.

use std::sync::{Arc, atomic::Ordering};
#[cfg(any(test, feature = "test-helpers"))]
use std::time::{Duration, Instant};

use tokio::sync::OnceCell;

#[cfg(any(test, feature = "test-helpers"))]
use crate::supertable::reader_cache::disk::fetch::PromotionWaitGuard;
use crate::{
    storage::StorageProvider,
    superfile::{
        LazyByteSource,
        reader::{OpenOptions, SuperfileReader},
    },
    supertable::{
        StorageRangeSource,
        manifest::{SubsectionOffsets, SuperfileUri},
        reader_cache::disk::*,
    },
};

impl DiskCacheStore {
    /// The reader every query (FTS, SQL, vector) opens a superfile through. Serves the best local
    /// copy now and, for FTS/SQL, warms a full local mmap in the background. See
    /// [`Self::reader_tiered`] for the memory/disk/source lookup.
    ///
    /// `offsets` is the manifest's record of where the parquet footer and the vector and FTS blobs
    /// sit inside the file. With it, a cold miss fetches all three open ranges in one parallel round
    /// trip instead of two; without it (`None`) the open still works, just one round trip slower.
    /// `intent` is the read policy: [`ReadIntent::Warm`] for FTS/SQL (serve now, warm a full mmap
    /// in the background), [`ReadIntent::Stream`] for vector search (block cache only, never
    /// promote).
    ///
    /// If the file cannot be admitted at all ([`DiskCacheError::BudgetExceeded`], typically a single
    /// superfile larger than the whole budget), this degrades to [`Self::open_range_only`], an
    /// uncached streaming reader, so the query still runs instead of failing. [`ReadIntent::Load`]
    /// must not come through here: a compaction read can never accept that degrade, so it uses
    /// [`Self::reader_synchronous_with_storage`], which fails loudly instead.
    pub async fn open_for_query(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        intent: ReadIntent,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        debug_assert_ne!(
            intent,
            ReadIntent::Load,
            "Load reads go through reader_synchronous_with_storage; they must never degrade"
        );

        match self.reader_tiered(uri, intent, offsets, storage).await {
            // Nothing local and the file cannot be admitted: stream it uncached rather than fail.
            Err(DiskCacheError::BudgetExceeded) => {
                self.open_range_only(uri, offsets, storage).await
            }
            served => served,
        }
    }

    /// Open a streaming reader straight against object storage, bypassing the cache entirely: no
    /// budget reservation, no background fill, no entry inserted into `cached`. The query still
    /// succeeds by issuing range GETs for only the bytes it touches.
    ///
    /// This is the [`DiskCacheError::BudgetExceeded`] fallback that [`Self::open_for_query`] takes,
    /// and it deliberately does not go through [`Self::reader_tiered`]: by the time it runs, the
    /// tiers have all been walked and tier 4 refused to admit the file (typically a single superfile
    /// larger than the whole cache budget). Re-walking them would re-check a file just proved to be
    /// neither local nor admittable. Nor can it be a tier of its own: the tiers find or admit a
    /// [`CachedEntry`], and this path has nothing to cache and so nothing to evict. It is the escape
    /// hatch for when caching is impossible, not a fifth place to look.
    async fn open_range_only(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let fetch_storage = self.resolve_storage(storage);
        let storage_uri = Self::storage_path(uri);

        let range_src: Arc<dyn LazyByteSource> = match offsets {
            Some(o) if o.total_size > 0 => Arc::new(StorageRangeSource::with_known_size(
                fetch_storage,
                storage_uri,
                o.total_size,
            )),
            _ => Arc::new(StorageRangeSource::with_unknown_size(
                fetch_storage,
                storage_uri,
            )),
        };

        // Range-only is also a lazy reader over object storage. A full CRC
        // scan here would turn a fallback path meant to issue targeted
        // ranges into a whole-superfile read.
        let reader =
            SuperfileReader::open_lazy_with(range_src, OpenOptions { verify_crc: false }).await?;

        Ok(Arc::new(reader))
    }

    /// The reader compaction opens its input through. Blocks until the whole file is local and
    /// mmapped, then serves; a lazy cache hit is not enough for a rewrite, so it re-fetches a full
    /// copy. A miss is fetched through `fetch_storage` rather than the cache's own `self.storage`:
    /// the hidden vector-index's superfiles live behind a prefixed storage provider the shared
    /// (user-keyed) cache cannot resolve on its own.
    pub async fn reader_synchronous_with_storage(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        self.reader_tiered(uri, ReadIntent::Load, None, Some(&fetch_storage))
            .await
    }

    /// The reader lookup, read top to bottom as four tiers, cheapest first: whole file in memory,
    /// whole file on local disk, a lazy reader already open, then the object store. Each tier says
    /// exactly what it is trying to find. `intent` is the only policy: it decides which local copies
    /// count (a lazy reader is a hit for a query but not for [`ReadIntent::Load`], which rewrites the
    /// whole file) and which fetch shape a miss uses. Nothing else branches on the caller.
    ///
    /// # Tier cascade
    ///
    /// A read falls through the tiers until one holds the file. The first hit serves and returns; a
    /// higher tier is always cheaper, so we never pay a lower tier's cost when a higher one hits.
    ///
    /// ```text
    ///                 reader_tiered(uri, intent)
    ///                            │
    ///                            ▼
    ///   Tier 1  whole_file_in_memory        ── hit ─►  serve                  0 GETs
    ///   memory  (Mapped or Buffered)                   (already in process)
    ///                            │ miss
    ///                            ▼
    ///           [Load only] drop_lazy_entry     a lazy handle cannot serve a
    ///           so it shadows nothing below      whole-file rewrite; re-fetch
    ///                            │
    ///                            ▼
    ///   Tier 2  fetch_from_disk_cache        ── hit ─►  mmap + serve          0 GETs
    ///   disk    (whole file on local NVMe)             (from a prior run, or
    ///                            │ miss                  a lazy fill finished)
    ///                            ▼
    ///   Tier 3  open_lazy_reader             ── hit ─►  serve                 rides the
    ///   lazy    (Paged handle, query only)             (share its block cache) open stream
    ///                            │ miss
    ///                            ▼
    ///   Tier 4  fetch_from_source_coalesced ────────►  admit + serve         cold GETs
    ///   source  (one task per URI single-flights)     (nothing was local)
    /// ```
    async fn reader_tiered(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        // Tier 1, memory: the whole file is already mmapped or buffered in this process. No I/O.
        if let Some(entry) = self.whole_file_in_memory(uri) {
            return Ok(self.serve(uri, &entry, intent, storage));
        }

        // A Load rewrites the whole file, so a lazy handle is no use to it: drop it here so it can
        // neither shadow the disk check below nor be served as the tier-3 hit. (majorly for compaction)
        if intent == ReadIntent::Load {
            self.drop_lazy_entry(uri);
        }

        // Tier 2, disk cache: the whole file is already on local disk (a prior run, or a lazy reader
        // that has since been fully filled). Mmap it, zero object-store GETs.
        if let Some(entry) = self
            .fetch_from_disk_cache(uri, offsets.map(|o| o.total_size))
            .await?
        {
            return Ok(self.serve(uri, &entry, intent, storage));
        }

        // Tier 3, open lazy reader: a Paged handle is already streaming this file. No full copy is
        // local, but a query can ride the existing block cache instead of opening a second stream.
        if intent != ReadIntent::Load
            && let Some(entry) = self.open_lazy_reader(uri)
        {
            return Ok(self.serve(uri, &entry, intent, storage));
        }

        // Tier 4, source: nothing local. Fetch it from the object store, coalescing concurrent
        // callers so only one does the work.
        let entry = self
            .fetch_from_source_coalesced(uri, intent, offsets, storage)
            .await?;

        Ok(self.serve(uri, &entry, intent, storage))
    }

    /// Tier 1: a cached entry that holds the whole file locally (mmapped or buffered), so it serves
    /// any read with no object-store GETs. A lazy ([`Residency::Paged`]) entry does not qualify and
    /// falls through to the disk check.
    fn whole_file_in_memory(&self, uri: &SuperfileUri) -> Option<Arc<CachedEntry>> {
        let entry = self.cached.get(uri)?;

        entry.has_whole_file().then(|| Arc::clone(&entry))
    }

    /// Tier 3: a cached lazy ([`Residency::Paged`]) entry, already streaming this file from the
    /// source over a shared block cache. Reached only after tier 2 confirms no full copy is on disk.
    fn open_lazy_reader(&self, uri: &SuperfileUri) -> Option<Arc<CachedEntry>> {
        // Explicitly Paged, not "whatever tier 1 did not take": a future residency kind must not
        // be served here by accident.
        self.cached
            .get(uri)
            .filter(|e| e.block_source().is_some())
            .map(|e| Arc::clone(&e))
    }

    /// Remove a cached lazy entry and give its budget back. Used when [`ReadIntent::Load`] finds a
    /// lazy entry it cannot use and must re-fetch a full copy.
    fn drop_lazy_entry(&self, uri: &SuperfileUri) {
        if let Some((_, removed)) = self.cached.remove(uri) {
            self.release_entry_accounting(&removed);
        }
        self.coordinators.remove(uri);
    }

    /// Hand the caller a reader: bump the LRU timestamp, and for [`ReadIntent::Warm`] keep the
    /// background fill going toward a full mmap. Every tier hands its hit through here.
    fn serve(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        entry: &Arc<CachedEntry>,
        intent: ReadIntent,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Arc<SuperfileReader> {
        entry.last_access_us.store(self.now_us(), Ordering::Release);
        if intent == ReadIntent::Warm {
            self.maybe_spawn_background_fill(uri, entry, storage);
        }

        Arc::clone(&entry.reader)
    }

    /// Tier 4: fetch the file from the object store, but only once even if many readers ask for the
    /// same URI at the same time. The first caller runs the fetch; everyone else waits on the shared
    /// `OnceCell` and gets that one result, so N concurrent misses cost one download, not N.
    ///
    /// If the fetch fails, the failure is not cached: throw the cell away and let this caller try
    /// again on its own, so one error does not make every later reader of the URI fail too.
    ///
    /// The cell lives only for the fetch. Once the entry is admitted, tiers 1 and 3 serve later
    /// callers, so this is where the cell is dropped for every fetch shape. Leaving it behind would
    /// keep a strong reference that outlives eviction and quietly serves an evicted entry.
    async fn fetch_from_source_coalesced(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();

        match cell
            .get_or_init(|| self.fetch_from_source(uri, intent, offsets, storage))
            .await
        {
            Ok(entry) => {
                // The entry is admitted, so tiers 1 and 3 serve later callers; the cell's
                // single-flight job is done. Dropping it also drops the strong reference it holds,
                // so eviction can actually free the entry instead of leaving a ghost this cell
                // would keep serving.
                self.coordinators.remove(uri);
                Ok(Arc::clone(entry))
            }
            Err(_) => {
                self.coordinators.remove(uri);
                self.fetch_from_source(uri, intent, offsets, storage).await
            }
        }
    }

    // Test and bench helpers. Compiled only for tests and the `test-helpers` feature, never into
    // the shipped library.

    /// Test and bench shorthand: a [`ReadIntent::Warm`] walk of [`Self::reader_tiered`] with no
    /// manifest offsets and the cache's own storage. Unlike [`Self::open_for_query`] it does not
    /// degrade to a range-only reader on [`DiskCacheError::BudgetExceeded`], so tests can assert on
    /// that error surfacing.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn reader(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        self.reader_tiered(uri, ReadIntent::Warm, None, None).await
    }

    /// Test shorthand for [`Self::reader_synchronous_with_storage`] using the cache's own storage:
    /// a [`ReadIntent::Load`] read that ends with the whole file mmapped.
    #[cfg(test)]
    pub(crate) async fn reader_synchronous(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let storage = Arc::clone(&self.storage);
        self.reader_synchronous_with_storage(uri, storage).await
    }

    /// Block until the background fill has swapped in the mmap-backed reader, or fail after
    /// `timeout`.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn wait_until_mmap_promoted(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        timeout: Duration,
    ) -> Result<(), DiskCacheError> {
        let _guard = PromotionWaitGuard::new(&self.n_promotion_waiters);
        let start = Instant::now();
        while start.elapsed() < timeout {
            if self.is_mmap_promoted(uri) {
                return Ok(());
            }
            tokio::time::sleep(MMAP_PROMOTION_POLL_INTERVAL).await;
        }
        Err(DiskCacheError::SuperfileOpen(format!(
            "superfile {uri:?} not mmap-promoted within {timeout:?}"
        )))
    }

    /// Block until no cache entry has a background fill still in flight
    /// (fill spawned, not yet mmap-promoted), or fail after `timeout`.
    ///
    /// Scoped to work the caller's own opens actually caused: entries that
    /// never spawned a fill (vector opens) and superfiles never opened at
    /// all are not waited on. Registering as a promotion waiter releases
    /// fills that are politely waiting on a held foreground reader.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn wait_until_fills_settled(
        self: &Arc<Self>,
        timeout: Duration,
    ) -> Result<(), DiskCacheError> {
        let _guard = PromotionWaitGuard::new(&self.n_promotion_waiters);
        let start = Instant::now();
        loop {
            let pending = self.cached.iter().any(|entry| {
                // A Paged entry whose fill latched but hasn't promoted to Mapped yet.
                entry
                    .value()
                    .fill_spawned()
                    .is_some_and(|f| f.load(Ordering::Acquire))
            });
            if !pending {
                return Ok(());
            }
            if start.elapsed() >= timeout {
                return Err(DiskCacheError::SuperfileOpen(format!(
                    "background fills not settled within {timeout:?}"
                )));
            }
            tokio::time::sleep(MMAP_PROMOTION_POLL_INTERVAL).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::Ordering},
        time::Duration,
    };

    use tempfile::TempDir;

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::LazyByteSource,
        supertable::{
            manifest::{SubsectionOffsets, SuperfileUri},
            reader_cache::{
                config::{ColdFetchMode, DiskCacheConfig},
                disk::{read::*, test_support::*},
            },
        },
    };

    // The tiers must never re-fetch data that is already local, for any read intent, and a lazy
    // entry must not shadow a full file that has landed on disk. Measured via the source-fetch
    // counter (`n_cold_fetches`) and the disk-reuse counter (`n_disk_reuses`).
    async fn read_delta(
        store: &Arc<DiskCacheStore>,
        intent: ReadIntent,
        uri: &SuperfileUri,
    ) -> (u64, u64) {
        let before = store.stats();
        match intent {
            ReadIntent::Stream | ReadIntent::Warm => {
                store.open_for_query(uri, None, None, intent).await
            }
            ReadIntent::Load => store.reader_synchronous(uri).await,
        }
        .expect("read");
        let after = store.stats();
        (
            after.n_cold_fetches - before.n_cold_fetches,
            after.n_disk_reuses - before.n_disk_reuses,
        )
    }

    #[tokio::test]
    async fn tiers_never_refetch_local_data() {
        for intent in [ReadIntent::Stream, ReadIntent::Warm, ReadIntent::Load] {
            // MEMORY: a mmapped entry already cached.
            let (_d, store) = test_store();
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, tiny_superfile_bytes()).await;
            store.reader_synchronous(&uri).await.expect("populate");
            let (cold, _) = read_delta(&store, intent, &uri).await;
            assert_eq!(cold, 0, "{intent:?}: memory hit must not fetch from source");

            // MEM-LAZY: a lazy entry cached, then the full file lands on disk from a sibling. The
            // read must use the disk file, not keep fetching from source (the shadow bug).
            let (_dl, store_l) = test_store();
            let uri_l = SuperfileUri::new_v4();
            put_superfile(&store_l, &uri_l, tiny_superfile_bytes()).await;
            store_l
                .open_for_query(&uri_l, None, None, ReadIntent::Stream)
                .await
                .expect("lazy open");
            let writer = reopen_store(&store_l, |_| {});
            writer
                .reader_synchronous(&uri_l)
                .await
                .expect("sibling writes the full file");
            let (cold_l, _) = read_delta(&store_l, intent, &uri_l).await;
            assert_eq!(
                cold_l, 0,
                "{intent:?}: a lazy entry must not shadow the on-disk file"
            );
            // This holds for Stream too. "Stream never promotes" means it never downloads the
            // whole file to get there; a whole file already sitting on local disk is free, and
            // mmapping it beats keeping the lazy reader's range GETs alive.
            assert!(
                store_l.is_mmap_promoted(&uri_l),
                "{intent:?}: the read must promote to the on-disk file"
            );

            // DISK: a finished file on disk, fresh in-memory map.
            let (_d2, store_a) = test_store();
            let uri2 = SuperfileUri::new_v4();
            put_superfile(&store_a, &uri2, tiny_superfile_bytes()).await;
            store_a.reader_synchronous(&uri2).await.expect("disk file");
            let store_b = reopen_store(&store_a, |_| {});
            let (cold2, reuse2) = read_delta(&store_b, intent, &uri2).await;
            assert_eq!(cold2, 0, "{intent:?}: disk hit must not fetch from source");
            assert_eq!(
                reuse2, 1,
                "{intent:?}: disk hit must reuse the on-disk file"
            );

            // SOURCE: nothing local.
            let (_d3, store_c) = test_store();
            let uri3 = SuperfileUri::new_v4();
            put_superfile(&store_c, &uri3, tiny_superfile_bytes()).await;
            let (cold3, _) = read_delta(&store_c, intent, &uri3).await;
            assert_eq!(cold3, 1, "{intent:?}: a cold miss must fetch from source");
        }
    }

    #[tokio::test]
    async fn is_mmap_promoted_false_for_unknown_uri() {
        let (_dir, store) = test_store();
        assert!(!store.is_mmap_promoted(&SuperfileUri::new_v4()));
    }

    #[tokio::test]
    async fn reader_synchronous_cold_then_warm_hit() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        let _r = store.reader_synchronous(&uri).await.expect("cold");
        let s = store.stats();
        assert_eq!(s.n_cold_fetches, 1);
        assert_eq!(s.n_entries, 1);
        assert_eq!(s.current_bytes, size);
        // mmap-backed after the synchronous fetch.
        assert!(store.is_mmap_promoted(&uri));

        // Second call is a warm cache hit (no new cold fetch).
        let _r2 = store.reader_synchronous(&uri).await.expect("warm");
        assert_eq!(store.stats().n_cold_fetches, 1);
    }

    #[tokio::test]
    async fn reader_synchronous_missing_object_errors() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // Nothing put at the storage path → head() fails.
        let err = store.reader_synchronous(&uri).await.expect_err("no object");
        let _ = format!("{err}");
        // Coordinator removed so a later (successful) put can proceed.
        assert!(store.coordinators.is_empty());

        // And it does: the failed attempt did not poison the URI.
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("read succeeds once the object exists");
        assert!(store.coordinators.is_empty());
    }

    // ----- cold fetch: hybrid path (default mode) -----

    /// Compaction opens hidden superfiles through
    /// `reader_synchronous_with_storage`, which must return an eager reader
    /// even when the cache currently holds a lazy entry from query fan-out.
    #[tokio::test]
    async fn reader_synchronous_with_storage_upgrades_lazy_hidden_entry() {
        use crate::storage::{LocalFsStorageProvider, PrefixedStorageProvider};

        let dir = TempDir::new().expect("tempdir");
        let user_storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("user root"));
        let hidden_root = dir.path().join("hidden_prefix");
        std::fs::create_dir_all(&hidden_root).expect("hidden root");
        let hidden_storage: Arc<dyn StorageProvider> = Arc::new(PrefixedStorageProvider::new(
            Arc::clone(&user_storage),
            "hidden_prefix",
        ));

        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&user_storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("cache");

        let uri = SuperfileUri::new_v4();
        hidden_storage
            .put_atomic(&uri.storage_path(), tiny_superfile_bytes())
            .await
            .expect("put at hidden prefix");

        // Query path admission: lazy reader with no resident parquet bytes.
        let lazy = cache
            .open_for_query(&uri, None, Some(&hidden_storage), ReadIntent::Warm)
            .await
            .expect("lazy cold fetch via caller storage");
        assert!(
            lazy.parquet_bytes().is_none(),
            "lazy mode should not materialize full parquet bytes"
        );

        // Compaction path must force an eager reopen via caller storage.
        let eager = cache
            .reader_synchronous_with_storage(&uri, Arc::clone(&hidden_storage))
            .await
            .expect("synchronous compaction open");
        assert!(
            eager.parquet_bytes().is_some(),
            "compaction input must have resident parquet bytes"
        );
        let batch = eager
            .get_record_batch(None)
            .expect("compaction should read full RecordBatch");
        assert_eq!(batch.num_rows(), 1);
    }

    // ----- RangeOnly mode rejects + open_range_only bypass -----

    #[test]
    fn reader_range_only_mode_is_rejected() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let cfg = DiskCacheConfig {
            cache_root: dir.path().join("cache"),
            cold_fetch_mode: ColdFetchMode::RangeOnly,
            ..Default::default()
        };
        let err = DiskCacheStore::new_unpinned(storage, cfg)
            .expect_err("range_only + disk cache must be rejected");
        assert!(matches!(err, DiskCacheError::Config(_)));
    }

    #[tokio::test]
    async fn open_range_only_unknown_size_reads_directly() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        // offsets = None → unknown-size StorageRangeSource.
        let r = store
            .open_range_only(&uri, None, None)
            .await
            .expect("range open");
        assert_eq!(r.n_docs(), 1);
        // Bypasses the cache entirely — nothing admitted.
        assert_eq!(store.stats().n_entries, 0);
        assert_eq!(store.stats().current_bytes, 0);
    }

    #[tokio::test]
    async fn open_range_only_known_size_reads_directly() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        let offsets = SubsectionOffsets {
            total_size: total,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        };
        let r = store
            .open_range_only(&uri, Some(&offsets), None)
            .await
            .expect("known-size range open");
        assert_eq!(r.n_docs(), 1);
    }

    // ----- lazy-foreground-with-background-fill mode -----

    #[tokio::test]
    async fn wait_until_mmap_promoted_times_out_for_unpromoted() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // Never fetched → never promoted → times out.
        let err = store
            .wait_until_mmap_promoted(&uri, Duration::from_millis(30))
            .await
            .expect_err("must time out");
        assert!(matches!(err, DiskCacheError::SuperfileOpen(_)));
        // Guard restored the waiter counter.
        assert_eq!(store.n_promotion_waiters.load(Ordering::Acquire), 0);
    }

    // ----- admission and coordinator lifetime -----

    /// Tier 3 as a serving tier: a second query for a URI whose lazy reader is already open must
    /// ride that reader (same block cache), not open a second stream from the source.
    #[tokio::test]
    async fn second_stream_open_shares_the_open_lazy_reader() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let first = store
            .open_for_query(&uri, None, None, ReadIntent::Stream)
            .await
            .expect("first lazy open");
        let second = store
            .open_for_query(&uri, None, None, ReadIntent::Stream)
            .await
            .expect("second open rides the first");

        assert!(
            Arc::ptr_eq(&first, &second),
            "tier 3 must hand back the already-open reader"
        );
        assert_eq!(store.stats().n_cold_fetches, 1, "no second source fetch");
        assert_eq!(store.stats().n_entries, 1);
    }

    /// A lazy admission must never displace a whole-file entry, and whatever an admission drops
    /// has its budget released. Drives `cold_fetch_lazy` straight at a URI that is already
    /// mmapped, the shape a compaction read racing a query open produces.
    #[tokio::test]
    async fn lazy_admission_never_replaces_a_whole_file_entry() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("whole file mmapped");
        assert_eq!(store.stats().current_bytes, size);

        let served = store
            .cold_fetch_lazy(&uri, None, store.resolve_storage(None), false)
            .await
            .expect("late lazy admission");

        assert!(
            served.has_whole_file(),
            "the caller must be handed the whole-file copy, not the lazy one"
        );
        assert!(
            store.is_mmap_promoted(&uri),
            "the mmapped entry must still be current"
        );
        assert_eq!(store.stats().n_entries, 1);
        assert_eq!(
            store.stats().current_bytes,
            size,
            "the dropped lazy entry must leave no bytes charged"
        );
    }

    /// Evicting a vector-opened (Stream) entry must actually free it: its bytes come off the
    /// budget, and the next read is a real source fetch that re-admits the entry, not a stale copy
    /// served from a coordinator that outlived the eviction.
    #[tokio::test]
    async fn evicting_a_stream_entry_frees_it_and_the_next_read_refetches() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let reader = store
            .open_for_query(&uri, None, None, ReadIntent::Stream)
            .await
            .expect("lazy open");
        // Fill one block through the entry's block source so SourceOwned bytes are charged. (A
        // lazy reader has no eager parquet bytes to read wholesale.) Clone the source out first:
        // holding the map guard across the await could deadlock against a fill's eviction.
        let source = Arc::clone(
            store
                .cached
                .get(&uri)
                .expect("entry cached")
                .block_source()
                .expect("a Stream entry is Paged"),
        );
        source.range(0, 64).await.expect("range read fills a block");
        drop(source);
        assert!(
            store.stats().current_bytes > 0,
            "filled blocks are charged to the budget"
        );
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert_eq!(store.stats().n_entries, 1);
        drop(reader);

        // Ask eviction to free a single byte. The policy is all-or-nothing (it returns no victims
        // if the eligible entries cannot cover the request), so a tiny request is what selects our
        // one entry.
        store
            .evict_at_least(1)
            .await
            .expect("the one cached entry is evictable");
        assert_eq!(store.stats().n_entries, 0, "the entry left the cache");
        assert_eq!(
            store.stats().current_bytes,
            0,
            "its bytes came off the budget, so nothing still holds the entry"
        );

        let _again = store
            .open_for_query(&uri, None, None, ReadIntent::Stream)
            .await
            .expect("reopen after eviction");
        assert_eq!(
            store.stats().n_cold_fetches,
            2,
            "an evicted entry is fetched again, not resurrected from a stale coordinator"
        );
        assert_eq!(store.stats().n_entries, 1, "and re-admitted to the cache");
    }

    // ----- eviction + budget -----
}
