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

/// Shared walks a caller joins before walking alone: the first, plus one retry when the first
/// failed or handed a `Load` a lazy entry.
const COALESCED_FETCH_ATTEMPTS: usize = 2;

impl DiskCacheStore {
    /// The reader every query (FTS, SQL, vector) opens a superfile through. The lookup is
    /// [`Self::reader_tiered`]; `intent` is explained on [`ReadIntent`].
    ///
    /// `storage_key` is the object key (the manifest entry's `storage_path()`). It is passed beside
    /// `uri` because a source-named superfile's key is not derived from its uuid, while the cache
    /// stays keyed by `uri`. With `offsets` (the manifest's footer and blob layout) a cold open
    /// fetches what it needs in one round trip.
    ///
    /// If the file cannot be admitted ([`DiskCacheError::BudgetExceeded`], usually one superfile
    /// larger than the whole budget), the query streams it uncached through
    /// [`Self::open_range_only`] instead of failing. [`ReadIntent::Load`] is rejected: a whole-file
    /// read cannot degrade like that, so it uses [`Self::reader_synchronous_with_storage`].
    pub async fn open_for_query(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        intent: ReadIntent,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if intent == ReadIntent::Load {
            return Err(DiskCacheError::SuperfileOpen(
                "Load reads go through reader_synchronous_with_storage; they must never degrade"
                    .into(),
            ));
        }

        match self
            .reader_tiered(uri, storage_key, intent, offsets, storage)
            .await
        {
            // Nothing local and the file cannot be admitted: stream it uncached rather than fail.
            Err(DiskCacheError::BudgetExceeded) => {
                self.open_range_only(storage_key, offsets, storage).await
            }
            served => served,
        }
    }

    /// A streaming reader straight over object storage: no budget, no fill, no cache entry, so it
    /// needs only the object key. [`Self::open_for_query`] falls back to it on
    /// [`DiskCacheError::BudgetExceeded`], after the tiers have shown the file is neither local nor
    /// admittable, so it does not walk them again.
    async fn open_range_only(
        self: &Arc<Self>,
        storage_key: &str,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let fetch_storage = self.resolve_storage(storage);
        let storage_uri = storage_key.to_owned();

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

        // No CRC check: it reads the whole file, and this path reads only the ranges touched.
        let reader =
            SuperfileReader::open_lazy_with(range_src, OpenOptions { verify_crc: false }).await?;

        Ok(Arc::new(reader))
    }

    /// The reader compaction opens its inputs through, as a [`ReadIntent::Load`]: it returns once
    /// the whole file is local and mmapped. A miss fetches through `fetch_storage`, not
    /// `self.storage`, because hidden vector-index files live behind a prefixed provider the shared
    /// cache cannot resolve.
    pub async fn reader_synchronous_with_storage(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        self.reader_tiered(
            uri,
            storage_key,
            ReadIntent::Load,
            None,
            Some(&fetch_storage),
        )
        .await
    }

    /// The lookup: four tiers, cheapest first, and the first hit serves. `intent` is the only
    /// policy (see [`ReadIntent`]). `storage_key` is used only to fetch from the object store; the
    /// local tiers are keyed by `uri`.
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
    ///   ┌─ one walk per URI at a time; concurrent callers share its result ──────────────────┐
    ///   │ Tier 2  fetch_from_disk_cache      ── hit ─►  mmap + serve          0 GETs         │
    ///   │ disk    (whole file on local NVMe)           (from a prior run, or                 │
    ///   │                          │ miss               a lazy fill finished)                │
    ///   │                          ▼                                                         │
    ///   │ Tier 3  open_lazy_reader           ── hit ─►  serve                 rides the      │
    ///   │ lazy    (Paged handle, query only)           (share its block cache) open stream   │
    ///   │                          │ miss                                                    │
    ///   │                          ▼                                                         │
    ///   │ Tier 4  fetch_from_source          ────────►  admit + serve         cold GETs      │
    ///   │ source  (nothing was local)                                                        │
    ///   └────────────────────────────────────────────────────────────────────────────────────┘
    /// ```
    async fn reader_tiered(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        // Tier 1, memory: the whole file is already mmapped or buffered in this process. No I/O.
        if let Some(entry) = self.whole_file_in_memory(uri) {
            return Ok(self.serve(uri, storage_key, &entry, intent, storage));
        }

        // A Load needs the whole file, so a lazy entry is no use to it. Drop it and free its budget
        // before fetching a whole copy.
        if intent == ReadIntent::Load {
            self.drop_lazy_entry(uri);
        }

        // Tiers 2 to 4, one walk per URI: concurrent callers share it, so N misses cost one stat,
        // one mmap or one download.
        let entry = self
            .fetch_local_or_source_coalesced(uri, storage_key, intent, offsets, storage)
            .await?;

        Ok(self.serve(uri, storage_key, &entry, intent, storage))
    }

    /// Tier 1: a cached entry holding the whole file (mmapped or buffered). It serves any read
    /// with no GETs. A lazy entry does not count.
    fn whole_file_in_memory(&self, uri: &SuperfileUri) -> Option<Arc<CachedEntry>> {
        let entry = self.cached.get(uri)?;

        entry.has_whole_file().then(|| Arc::clone(&entry))
    }

    /// Tier 3: a cached lazy ([`Residency::Paged`]) entry. A query shares its block cache instead
    /// of opening a second stream.
    fn open_lazy_reader(&self, uri: &SuperfileUri) -> Option<Arc<CachedEntry>> {
        // Match Paged by name, so a new residency kind is never served here by accident.
        self.cached
            .get(uri)
            .filter(|e| matches!(e.residency, Residency::Paged { .. }))
            .map(|e| Arc::clone(&e))
    }

    /// Remove a lazy entry and release its budget, for a [`ReadIntent::Load`]. A whole file that
    /// landed since the memory check is kept.
    ///
    /// Leaves any coordinator alone: it belongs to a fetch in flight, and removing it would send
    /// later callers into a second download. A Load that joins that fetch still ends with a whole
    /// file (see [`Self::fetch_local_or_source_coalesced`]).
    fn drop_lazy_entry(&self, uri: &SuperfileUri) {
        if let Some((_, removed)) = self
            .cached
            .remove_if(uri, |_, entry| !entry.has_whole_file())
        {
            self.release_entry_accounting(&removed);
        }
    }

    /// Every hit is served through here: bump the entry's LRU time and, for [`ReadIntent::Warm`],
    /// start its background fill if it has not started yet.
    fn serve(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        entry: &Arc<CachedEntry>,
        intent: ReadIntent,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Arc<SuperfileReader> {
        entry.last_access_us.store(self.now_us(), Ordering::Release);
        if intent == ReadIntent::Warm {
            self.maybe_spawn_background_fill(uri, storage_key, entry, storage);
        }

        Arc::clone(&entry.reader)
    }

    /// Tiers 2 to 4 for one caller, uncoalesced. [`Self::fetch_local_or_source_coalesced`] wraps
    /// it in the per-URI single flight.
    async fn fetch_local_or_source(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        // A fill may have installed the whole file since the memory check. Use its entry, not the
        // file on disk: a fill that left the vector blob on the block cache wrote a file with a
        // hole, and only the live entry can serve it.
        if let Some(entry) = self.whole_file_in_memory(uri) {
            return Ok(entry);
        }

        // Tier 2, disk: the whole file is on local disk (a prior run, or a finished fill). Mmap
        // it, no GETs.
        if let Some(entry) = self
            .fetch_from_disk_cache(uri, offsets.map(|o| o.total_size))
            .await?
        {
            return Ok(entry);
        }

        // Tier 3, lazy reader: a query rides its block cache. Not for a Load, which needs the
        // whole file.
        if intent != ReadIntent::Load
            && let Some(entry) = self.open_lazy_reader(uri)
        {
            return Ok(entry);
        }

        // Tier 4, object store: nothing is local.
        self.fetch_from_source(uri, storage_key, intent, offsets, storage)
            .await
    }

    /// Tiers 2 to 4 as one walk per URI. The first caller runs [`Self::fetch_local_or_source`] and
    /// concurrent callers await the same `OnceCell`, so N misses cost one stat, mmap or download.
    /// Probing disk inside the walk also keeps a whole file from being admitted twice.
    ///
    /// The returned entry satisfies `intent`. The cell is not keyed by intent, so a `Load` can join
    /// a query's walk and get a lazy entry; it then walks again rather than serve a partial file.
    /// A failure is not cached either: each waiter walks again once, still coalesced, and after a
    /// second failure walks alone, so one bad fetch cannot poison later readers.
    ///
    /// The cell lives only for the walk and is dropped here on every outcome. Left behind, it would
    /// keep an evicted entry alive and keep serving it.
    async fn fetch_local_or_source_coalesced(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        for _ in 0..COALESCED_FETCH_ATTEMPTS {
            let cell = self
                .coordinators
                .entry(*uri)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone();

            let result = cell
                .get_or_init(|| {
                    self.fetch_local_or_source(uri, storage_key, intent, offsets, storage)
                })
                .await;

            // Remove only our own cell. A newer one belongs to a retry already in flight.
            self.coordinators
                .remove_if(uri, |_, live| Arc::ptr_eq(live, &cell));

            match result {
                // Joined a query's lazy walk. A Load needs the whole file, so walk again.
                Ok(entry) if intent == ReadIntent::Load && !entry.has_whole_file() => continue,
                Ok(entry) => return Ok(Arc::clone(entry)),
                Err(_) => continue,
            }
        }

        self.fetch_local_or_source(uri, storage_key, intent, offsets, storage)
            .await
    }

    // Test and bench helpers. Compiled only for tests and the `test-helpers` feature, never into
    // the shipped library.

    /// Test and bench shorthand: a [`ReadIntent::Warm`] read through the cache's own storage, at
    /// the key derived from `uri`. Unlike [`Self::open_for_query`] it returns
    /// [`DiskCacheError::BudgetExceeded`] instead of degrading, so tests can assert on it.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn reader(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        self.reader_tiered(uri, &uri.storage_path(), ReadIntent::Warm, None, None)
            .await
    }

    /// Test shorthand for [`Self::reader_synchronous_with_storage`] using the cache's own storage:
    /// a [`ReadIntent::Load`] read that ends with the whole file mmapped.
    #[cfg(test)]
    pub(crate) async fn reader_synchronous(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let storage = Arc::clone(&self.storage);
        self.reader_synchronous_with_storage(uri, &uri.storage_path(), storage)
            .await
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

    use futures::future::join_all;
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

    /// Readers that miss on one cold URI at the same time; enough to overlap on the fetch, few
    /// enough to stay cheap.
    const CONCURRENT_MISSES: usize = 8;

    /// `(source fetches, disk reuses)` caused by one read of `uri` with `intent`.
    async fn read_delta(
        store: &Arc<DiskCacheStore>,
        intent: ReadIntent,
        uri: &SuperfileUri,
    ) -> (u64, u64) {
        let before = store.stats();
        match intent {
            ReadIntent::Stream | ReadIntent::Warm => {
                store
                    .open_for_query(uri, &uri.storage_path(), None, None, intent)
                    .await
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

    /// No tier re-fetches data that is already local, for any intent.
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

            // MEM-LAZY: a lazy entry cached, then a sibling writes the full file to disk. The read
            // must use the disk file, not keep fetching from source.
            let (_dl, store_l) = test_store();
            let uri_l = SuperfileUri::new_v4();
            put_superfile(&store_l, &uri_l, tiny_superfile_bytes()).await;
            store_l
                .open_for_query(
                    &uri_l,
                    &uri_l.storage_path(),
                    None,
                    None,
                    ReadIntent::Stream,
                )
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
            // Stream too: it never downloads the whole file, but one already on disk is free.
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

            // Every store's budget still matches what it holds.
            for s in [&store, &store_l, &store_b, &store_c] {
                s.assert_budget_consistent();
            }
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
        assert!(
            store.coordinators.is_empty(),
            "the fetch's cell is dropped once the entry is admitted"
        );
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
            .open_for_query(
                &uri,
                &uri.storage_path(),
                None,
                Some(&hidden_storage),
                ReadIntent::Warm,
            )
            .await
            .expect("lazy cold fetch via caller storage");
        assert!(
            lazy.parquet_bytes().is_none(),
            "lazy mode should not materialize full parquet bytes"
        );

        // Compaction path must force an eager reopen via caller storage.
        let eager = cache
            .reader_synchronous_with_storage(&uri, &uri.storage_path(), Arc::clone(&hidden_storage))
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
            .open_range_only(&uri.storage_path(), None, None)
            .await
            .expect("range open");
        assert_eq!(r.n_docs(), 1);
        // Bypasses the cache: nothing admitted.
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
            .open_range_only(&uri.storage_path(), Some(&offsets), None)
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

    /// `Load` reads have their own entry point; `open_for_query` must refuse them rather than let a
    /// whole-file read degrade to a range-only reader on a budget miss.
    #[tokio::test]
    async fn open_for_query_rejects_load() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let err = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Load)
            .await
            .expect_err("Load must be rejected");
        assert!(matches!(err, DiskCacheError::SuperfileOpen(_)));
        assert_eq!(store.stats().n_cold_fetches, 0, "rejected before any fetch");
    }

    /// The coordinator is keyed by URI, not intent, so a `Load` can join a query's in-flight lazy
    /// fetch and be handed a `Paged` entry. It must then fetch the whole file itself. Simulated by
    /// parking a lazy entry in a coordinator cell exactly as a finished lazy fetch leaves it.
    #[tokio::test]
    async fn load_that_joins_a_lazy_fetch_still_gets_the_whole_file() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // A lazy open, then lift its entry out of the map and into a cell, as if that fetch were
        // the one this Load joins.
        store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("lazy open");
        let (_, paged) = store.cached.remove(&uri).expect("lazy entry cached");
        assert!(!paged.has_whole_file());
        store
            .coordinators
            .insert(uri, Arc::new(OnceCell::new_with(Some(Ok(paged)))));

        let reader = store
            .reader_synchronous(&uri)
            .await
            .expect("Load through a joined lazy cell");

        assert!(
            reader.parquet_bytes().is_some(),
            "compaction gets whole-file bytes"
        );
        assert!(
            store.is_mmap_promoted(&uri),
            "the whole file replaced the lazy entry"
        );
        assert_eq!(
            store.stats().n_cold_fetches,
            2,
            "one lazy fetch, then the Load's own"
        );
        assert!(store.coordinators.is_empty(), "the joined cell was dropped");
        store.assert_budget_consistent();
    }

    // ----- admission and coordinator lifetime -----

    /// The single flight: readers missing on one URI at once cost one source fetch, and all come
    /// back holding the same reader.
    #[tokio::test]
    async fn concurrent_cold_misses_share_one_fetch() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let readers = join_all((0..CONCURRENT_MISSES).map(|_| {
            let store = Arc::clone(&store);
            async move {
                store
                    .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
                    .await
                    .expect("coalesced open")
            }
        }))
        .await;

        assert!(
            readers.iter().all(|r| Arc::ptr_eq(r, &readers[0])),
            "every caller holds the one reader"
        );
        assert_eq!(store.stats().n_cold_fetches, 1, "one fetch for all of them");
        assert_eq!(store.stats().n_entries, 1);
        assert!(
            store.coordinators.is_empty(),
            "the shared cell is dropped once the walk settles"
        );
        store.assert_budget_consistent();
    }

    /// A failed shared walk is not cached: the next caller retries and the failed cell is gone.
    /// Simulated by parking a failed cell exactly as a failed walk leaves it for its waiters.
    #[tokio::test]
    async fn a_failed_shared_walk_does_not_poison_the_next_caller() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        store.coordinators.insert(
            uri,
            Arc::new(OnceCell::new_with(Some(Err(
                DiskCacheError::BudgetExceeded,
            )))),
        );

        let reader = store
            .reader_synchronous(&uri)
            .await
            .expect("the retry fetches for real");

        assert_eq!(reader.n_docs(), 1);
        assert_eq!(
            store.stats().n_cold_fetches,
            1,
            "the retry is the only fetch"
        );
        assert!(store.coordinators.is_empty(), "the failed cell is gone");
        store.assert_budget_consistent();
    }

    /// A Load drops the lazy entry it cannot use, and only that: a whole file that landed since
    /// the memory check is a hit to keep, not something to re-download.
    #[tokio::test]
    async fn load_drops_a_lazy_entry_and_never_a_whole_file() {
        let (_dir, store) = test_store();
        let whole = SuperfileUri::new_v4();
        let lazy = SuperfileUri::new_v4();
        put_superfile(&store, &whole, tiny_superfile_bytes()).await;
        put_superfile(&store, &lazy, tiny_superfile_bytes()).await;
        store
            .reader_synchronous(&whole)
            .await
            .expect("whole file mmapped");
        store
            .open_for_query(&lazy, &lazy.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("lazy open");
        let charged = store.stats().current_bytes;
        let lazy_size = store
            .cached
            .get(&lazy)
            .expect("lazy entry")
            .size_bytes
            .load(Ordering::Acquire);

        store.drop_lazy_entry(&whole);
        store.drop_lazy_entry(&lazy);

        assert!(store.is_mmap_promoted(&whole), "the whole file stays");
        assert!(!store.is_cached(&lazy), "the lazy entry is gone");
        assert_eq!(
            store.stats().current_bytes,
            charged - lazy_size,
            "only the lazy entry's charge came back"
        );
        store.assert_budget_consistent();
    }

    /// Tier 3 as a serving tier: a second query for a URI whose lazy reader is already open must
    /// ride that reader (same block cache), not open a second stream from the source.
    #[tokio::test]
    async fn second_stream_open_shares_the_open_lazy_reader() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let first = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("first lazy open");
        let second = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
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
            .cold_fetch_lazy(
                &uri,
                &uri.storage_path(),
                None,
                store.resolve_storage(None),
                false,
            )
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
        store.assert_budget_consistent();
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
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("lazy open");
        // Fill one block so the entry has SourceOwned bytes charged. Clone the source out first:
        // holding the map guard across the await could deadlock against an eviction.
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

        // Ask eviction for one byte. It is all-or-nothing, so a tiny request picks our one entry.
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
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("reopen after eviction");
        assert_eq!(
            store.stats().n_cold_fetches,
            2,
            "an evicted entry is fetched again, not resurrected from a stale coordinator"
        );
        assert_eq!(store.stats().n_entries, 1, "and re-admitted to the cache");
        store.assert_budget_consistent();
    }

    // ----- eviction + budget -----
}
