// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The public reader API: hand back a [`SuperfileReader`] for a URI, serving it
//! from memory or disk when it is cached and cold-fetching it when it is not.

use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};

use tokio::sync::OnceCell;

use crate::{
    storage::StorageProvider,
    superfile::{
        LazyByteSource,
        reader::{OpenOptions, SuperfileReader},
    },
    supertable::{
        StorageRangeSource,
        manifest::{SubsectionOffsets, SuperfileUri},
        reader_cache::{
            config::ColdFetchMode,
            disk::{fetch::PromotionWaitGuard, *},
        },
    },
};

impl DiskCacheStore {
    /// Hot path. Cached → cloned `Arc<SuperfileReader>`; cold
    /// → coalesced cold-fetch coordinator. Dispatches by
    /// `config.cold_fetch_mode`:
    ///
    /// - [`ColdFetchMode::LazyForegroundWithBackgroundFill`] (default):
    ///   foreground returns a lazy reader over a `StorageRangeSource`
    ///   that pays only the per-query range budget; a background task
    ///   downloads the full superfile to NVMe and swaps in the mmap'd
    ///   entry, so subsequent (warm) queries are resident. Minimizes
    ///   cold-query p50 on object-storage-native deployments.
    /// - [`ColdFetchMode::HybridWithPrefetch`]:
    ///   parallel range-GETs feed the foreground reader (built
    ///   from in-memory bytes) and a fire-and-forget cache fill
    ///   (mmap'd, registered on completion). Foreground returns
    ///   when range-fetches finish; pwrites + mmap + cache
    ///   registration finalize in the background.
    /// - [`ColdFetchMode::RangeOnly`]: callers should construct
    ///   a `StorageRangeSource` + `SuperfileReader::open_lazy`
    ///   directly — `DiskCacheStore::reader` rejects this mode
    ///   because the disk-cache layer isn't the right entry
    ///   point — `RangeOnly` bypasses the cache by design.
    pub async fn reader(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        // Default allows fill — same as FTS/SQL. Vector search must call
        // [`Self::reader_with_hints`] with `allow_background_fill = false`.
        self.reader_with_hints(uri, None, None, true).await
    }

    /// like [`Self::reader`] but takes a precomputed
    /// [`SubsectionOffsets`] hint (sourced from the manifest's
    /// [`crate::supertable::manifest::SuperfileEntry::subsection_offsets`]).
    /// On a cold miss in the
    /// `LazyForegroundWithBackgroundFill` mode the hint lets the
    /// cold-fetch path fire the parquet-footer, vector subsection,
    /// and FTS subsection GETs **in parallel** (1 RTT cold open)
    /// instead of doing the parquet footer first and the
    /// subsection fetches second (2 RTTs).
    ///
    /// `allow_background_fill` is the modality gate: FTS/SQL pass `true`
    /// so parquet/FTS bytes can promote to mmap (vector blob skipped);
    /// vector search passes `false` and retains only the block cache.
    ///
    /// `None` falls back to the 2-RTT shape — same shape,
    /// slower. The other cold-fetch modes (`HybridWithPrefetch`,
    /// `RangeOnly`) ignore the hint today.
    pub async fn reader_with_hints(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        allow_background_fill: bool,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        match self.config.cold_fetch_mode {
            ColdFetchMode::HybridWithPrefetch => self.reader_hybrid(uri, storage).await,
            ColdFetchMode::RangeOnly => Err(DiskCacheError::SuperfileOpen(
                "ColdFetchMode::RangeOnly bypasses the disk cache; \
                 construct StorageRangeSource + open_lazy directly"
                    .into(),
            )),
            ColdFetchMode::LazyForegroundWithBackgroundFill => {
                self.reader_lazy_with_bg_fill_hinted(uri, offsets, storage, allow_background_fill)
                    .await
            }
        }
    }

    /// Open a streaming, RangeOnly reader directly against object
    /// storage, bypassing the disk cache entirely: no budget
    /// reservation, no background fill, no entry inserted into
    /// `cached`.
    ///
    /// Used as the [`DiskCacheError::BudgetExceeded`] fallback —
    /// e.g. a single superfile larger than the whole cache budget.
    /// The query still succeeds by issuing range GETs for only the
    /// bytes the reader touches; nothing is admitted, so there's
    /// nothing to evict.
    pub async fn open_range_only(
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

    /// Strictly-cached cold-fetch path — waits for all pwrites
    /// + fsync + mmap before returning. Public for integration
    /// tests that want this deterministic behavior; the
    /// production reader path uses `reader_hybrid`.
    pub async fn reader_synchronous(
        self: &Arc<Self>,
        uri: &SuperfileUri,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        let storage = Arc::clone(&self.storage);
        self.reader_synchronous_with_storage(uri, storage).await
    }

    /// Like [`Self::reader_synchronous`], but fetches a cache miss through
    /// `fetch_storage` instead of the cache's own `self.storage`. Needed for
    /// the hidden vector-index, whose superfiles live behind a prefixed storage
    /// provider that the shared (user-keyed) cache's `self.storage` can't
    /// resolve — without this the cold-fetch reads the wrong path. On a cache
    /// hit it returns the resident mmap-backed reader regardless of storage.
    pub async fn reader_synchronous_with_storage(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            if entry.mmap.is_some() {
                entry.last_access_us.store(self.now_us(), Ordering::Release);
                return Ok(Arc::clone(&entry.reader));
            }
            drop(entry);
            if let Some((_, removed)) = self.cached.remove(uri) {
                self.release_entry_accounting(&removed);
            }
            self.coordinators.remove(uri);
            let replacement = self.cold_fetch(uri, Arc::clone(&fetch_storage)).await?;
            return Ok(Arc::clone(&replacement.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let result = cell
            .get_or_init(|| async { self.cold_fetch(uri, Arc::clone(&fetch_storage)).await })
            .await;
        match result {
            Ok(entry) => {
                self.coordinators.remove(uri);
                Ok(Arc::clone(&entry.reader))
            }
            Err(_e) => {
                self.coordinators.remove(uri);
                Err(self
                    .cold_fetch(uri, Arc::clone(&fetch_storage))
                    .await
                    .err()
                    .unwrap_or(DiskCacheError::SuperfileOpen("cold fetch error".into())))
            }
        }
    }

    /// Hybrid cold-fetch. Range-fetches feed the foreground
    /// reader from in-memory bytes; pwrites + mmap + cache
    /// registration run as a background task that outlives
    /// this method's return.
    pub(crate) async fn reader_hybrid(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
            return Ok(Arc::clone(&entry.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        // OnceCell value: `Result<Arc<CachedEntry>, ...>` but we
        // only need the reader part for the foreground response.
        // The coordinator builds a CachedEntry whose `reader` is
        // the in-memory-backed `Arc<SuperfileReader>`; the
        // background task replaces the entry in `cached` with a
        // mmap-backed reader once the disk file is finalized.
        let result = cell
            .get_or_init(|| async {
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_hybrid(uri, fetch_storage).await
            })
            .await;
        match result {
            Ok(entry) => Ok(Arc::clone(&entry.reader)),
            Err(DiskCacheError::BudgetExceeded) => {
                self.coordinators.remove(uri);
                Err(DiskCacheError::BudgetExceeded)
            }
            Err(_) => {
                // Only the retry path needs the resolved storage handle; the Ok
                // and BudgetExceeded arms skip the clone.
                self.coordinators.remove(uri);
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_hybrid(uri, fetch_storage)
                    .await
                    .map(|entry| Arc::clone(&entry.reader))
            }
        }
    }

    /// Block until the background fill has swapped in the
    /// mmap-backed reader, or fail after `timeout`.
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
    pub async fn wait_until_fills_settled(
        self: &Arc<Self>,
        timeout: Duration,
    ) -> Result<(), DiskCacheError> {
        let _guard = PromotionWaitGuard::new(&self.n_promotion_waiters);
        let start = Instant::now();
        loop {
            let pending = self.cached.iter().any(|entry| {
                entry.value().fill_spawned.load(Ordering::Acquire) && entry.value().mmap.is_none()
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
        supertable::{
            manifest::{SubsectionOffsets, SuperfileUri},
            reader_cache::{
                config::{ColdFetchMode, DiskCacheConfig},
                disk::{read::*, test_support::*},
            },
        },
    };

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
            .reader_with_hints(&uri, None, Some(&hidden_storage), true)
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

    // ----- eviction + budget -----
}
