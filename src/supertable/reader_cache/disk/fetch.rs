// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cold fetch: pull a superfile from object storage into the cache. The three
//! fetch shapes (whole, hybrid, lazy), the byte pumps that download it, and the
//! finalize step that lands it on disk as an mmap-backed entry.

use std::{
    fs,
    io::{ErrorKind, SeekFrom},
    mem,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
};

use async_trait::async_trait;
use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use futures::{
    future::try_join_all,
    stream::{FuturesUnordered, StreamExt},
};
use memmap2::Mmap;
use tokio::{
    io::{AsyncSeekExt, AsyncWriteExt},
    sync::oneshot,
    task::{JoinHandle, JoinSet, spawn_blocking},
};
use tracing::{Instrument, debug_span};

use crate::{
    config::global as global_config,
    runtime_bridge::carry_span,
    runtime_metrics::io::scope_background,
    storage::{StorageError, StorageProvider},
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazyByteSourceError, PrefetchedSource, ReadError,
        format::{footer, kv},
        reader::{DEFAULT_TAIL_SPECULATIVE_BYTES, OpenOptions, SuperfileReader},
    },
    supertable::{
        StorageRangeSource,
        manifest::{SubsectionOffsets, SuperfileUri},
        reader_cache::{
            block_source::{BlockCachedSource, SourceLayout},
            config::ColdFetchMode,
            disk::{
                fill_wait::{FillWait, FillWaiters, PromotionDefer},
                sources::mmap_readonly_with_handle,
                *,
            },
        },
    },
    utils::trace::{OpOrigin, detached},
};

/// A hybrid fill's temp file, deleted on drop until the finalizer takes it over
/// ([`Self::claim`]). Pwrites still in flight write into the unlinked file and free it as they
/// finish.
struct UnclaimedTmp(Option<PathBuf>);

impl UnclaimedTmp {
    fn claim(mut self) {
        self.0 = None;
    }
}

impl Drop for UnclaimedTmp {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = fs::remove_file(path);
        }
    }
}

/// A lazy open's block sources, until its entry is admitted ([`Self::admitted`]). A source that
/// adopts a `.blocks` file left on disk takes over the charge of the file's record, and gives it
/// back when it drops. Dropped unadmitted (an open that fails, is cancelled, or loses its slot to a
/// whole file), a source hands the file back to a record instead, so the file stays charged and
/// eviction can still reach it.
struct UnadmittedBlockSources<'a> {
    store: &'a DiskCacheStore,
    uri: SuperfileUri,
    sources: Vec<Arc<BlockCachedSource>>,
}

impl UnadmittedBlockSources<'_> {
    fn admitted(mut self) {
        self.sources.clear();
    }
}

impl Drop for UnadmittedBlockSources<'_> {
    fn drop(&mut self) {
        for source in &self.sources {
            self.store.retain_unadmitted_block_file(&self.uri, source);
        }
    }
}

impl DiskCacheStore {
    /// Mmap a cache file and open it as a [`Residency::Mapped`] entry. The caller owns the budget
    /// and the map insert. The reader and the entry share one `Arc<Mmap>`, so the idle-page sweep's
    /// `madvise` reaches the mapping the reader uses.
    pub(crate) fn open_cached_entry(
        &self,
        path: &Path,
        size: u64,
        verify_crc: bool,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let (mmap, bytes) = mmap_readonly_with_handle(path).map_err(DiskCacheError::Io)?;
        let reader = SuperfileReader::open_with(bytes, OpenOptions { verify_crc })?;
        Ok(self.build_mmap_entry(Arc::new(reader), mmap, size, None))
    }

    /// [`Self::open_cached_entry`] for a copy at [`Self::hole_path`]: opened the way its fill left
    /// it, with the vector blob read through a block cache over the superfile's `.blocks` file, and
    /// charged its size minus the hole. The hole's range comes from the copy's own footer.
    pub(crate) async fn open_holed_entry(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        size: u64,
        storage_key: &str,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let (mmap, bytes) =
            mmap_readonly_with_handle(&self.hole_path(uri)).map_err(DiskCacheError::Io)?;

        let kv_map = footer::read_kv_metadata(&bytes).map_err(ReadError::Footer)?;

        let hole = vector_blob_range_in(&kv_map).ok_or_else(|| {
            DiskCacheError::SuperfileOpen("holed cache copy has no vector blob".into())
        })?;

        let remote: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
            fetch_storage,
            storage_key.to_owned(),
            size,
        ));

        // This source charges the vector blocks it holds, so the entry is charged only for the
        // bytes the copy holds. It serves only vector bytes, so every GET is aligned.
        let vector_source = BlockCachedSource::new_with_accounting(
            remote,
            Arc::downgrade(self),
            *uri,
            self.blocks_path(uri),
            true,
            SourceLayout::ALL_ALIGNED,
        );

        self.holed_mmap_entry(mmap, bytes, hole, vector_source, size)
            .await
    }

    /// A [`Residency::Mapped`] entry for a local copy missing its vector blob: reads inside `hole`
    /// go to `vector_source`, the rest to `bytes`. Parquet is installed resident, so sync decodes
    /// (take, id scans) run off the local bytes; there is no CRC check, since the vector bytes are
    /// not local to check. The hole is not on disk, so when `vector_source` charges its own blocks
    /// the entry is charged `size` minus the hole; otherwise the hole's blocks ride on `size`.
    async fn holed_mmap_entry(
        &self,
        mmap: Arc<Mmap>,
        bytes: Bytes,
        (hole_start, hole_len): (u64, u64),
        vector_source: Arc<BlockCachedSource>,
        size: u64,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
            local: Arc::new(BytesLazyByteSource::new(bytes.clone())),
            hole_start,
            hole_len,
            fallback: Arc::clone(&vector_source),
        });
        let mut reader =
            SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false }).await?;
        reader.install_resident_parquet(bytes, (hole_start, hole_len))?;

        let charge = if vector_source.owns_accounting() {
            size.saturating_sub(hole_len)
        } else {
            size
        };
        Ok(self.build_mmap_entry(Arc::new(reader), mmap, charge, Some(vector_source)))
    }

    /// Build a [`Residency::Mapped`] entry, always `Eager`. `charge` is the file size, or less when
    /// the vector blob stays on a block source that charges its own blocks. `vector_source` is
    /// `Some` only when the vector blob was left out of the mmap.
    pub(crate) fn build_mmap_entry(
        &self,
        reader: Arc<SuperfileReader>,
        mmap: Arc<Mmap>,
        charge: u64,
        vector_source: Option<Arc<BlockCachedSource>>,
    ) -> Arc<CachedEntry> {
        Arc::new(CachedEntry {
            reader,
            residency: Residency::Mapped {
                mmap,
                vector_source,
            },
            size_bytes: Arc::new(AtomicU64::new(charge)),
            accounting: EntryAccounting::Eager,
            last_access_us: AtomicU64::new(self.now_us()),
        })
    }

    /// Swap the whole file a background fill (or the hybrid finalizer) downloaded into the
    /// file's cached entry, when `may_replace` allows it. `entry` is already mmapped from
    /// `tmp_path`. Never into an empty slot: an evicted file stays evicted.
    ///
    /// The new entry is paid for by what the old one reserved, then `own_reservation`, then free
    /// space; what is left over is given back. Installed, the tempfile is renamed to
    /// `final_path`. Not installed, it is deleted and `own_reservation` is given back.
    fn install_promoted_entry(
        &self,
        uri: SuperfileUri,
        entry: Arc<CachedEntry>,
        tmp_path: &Path,
        final_path: &Path,
        may_replace: impl Fn(&CachedEntry) -> bool,
        own_reservation: u64,
    ) -> Option<Arc<CachedEntry>> {
        let charge = entry.size_bytes.load(Ordering::Acquire);

        // Decide under the entry's lock, touch the files after it.
        let (installed, to_release) = match self.cached.entry(uri) {
            // Read under the entry's lock, so a cancel and an install never cross.
            Entry::Occupied(mut occupied) if may_replace(occupied.get()) => {
                // A Stream-opened entry's block source keeps reserving its own blocks.
                let held = own_reservation + occupied.get().whole_file_charge();
                if held >= charge || self.try_reserve_without_evicting(charge - held) {
                    let replaced = mem::replace(occupied.get_mut(), Arc::clone(&entry));
                    // Leave the lock before the old entry drops: its last reference may fsync a
                    // block index.
                    drop(occupied);
                    drop(replaced);
                    (true, held.saturating_sub(charge))
                } else {
                    (false, own_reservation)
                }
            }
            // No entry, someone else's whole file, or an entry this download may not replace.
            Entry::Occupied(_) | Entry::Vacant(_) => (false, own_reservation),
        };

        if installed {
            self.move_into_cache(uri, tmp_path, final_path);
        } else {
            let _ = fs::remove_file(tmp_path);
        }
        if to_release > 0 {
            self.release_block_bytes(to_release);
        }
        installed.then_some(entry)
    }

    /// Whether `fill`'s download is still wanted: its store is still in use, and the download may
    /// replace whatever the file has cached (see [`BackgroundFill::may_replace`]). The landing
    /// checks again under the lock; this check only saves work.
    fn fill_can_install(self: &Arc<Self>, uri: &SuperfileUri, fill: &BackgroundFill) -> bool {
        !background_store_abandoned(self)
            && fill.may_replace(self.cached.get(uri).as_deref().map(Arc::as_ref))
    }

    /// What `uri`'s cached entry reserved for the whole file (see
    /// [`CachedEntry::whole_file_charge`]), or `None` with no entry.
    fn entry_whole_file_charge(&self, uri: &SuperfileUri) -> Option<u64> {
        self.cached.get(uri).map(|entry| entry.whole_file_charge())
    }

    /// Whether a query holds `uri`'s reader. The file's background download waits while one does.
    pub(super) fn file_in_use(&self, uri: &SuperfileUri) -> bool {
        self.cached
            .get(uri)
            .is_some_and(|entry| entry.held_by_query())
    }

    /// Remove `uri`'s entry when `removable` says so, and cancel its download, both under the
    /// entry's lock. The download is unregistered, so the next Warm read can start a new one; the
    /// old one stops at its next check and never lands.
    /// - a download finishing at the same time either lands first and is removed with the entry,
    ///   or sees the cancel;
    /// - a Warm read starting a download at the same time finds the entry gone (see
    ///   [`Self::maybe_spawn_background_fill`]).
    ///
    /// With no entry, the download is cancelled all the same: the idle cap may have dropped it.
    pub(super) fn remove_entry_cancelling_fill(
        &self,
        uri: &SuperfileUri,
        removable: impl Fn(&CachedEntry) -> bool,
    ) -> Option<Arc<CachedEntry>> {
        let slot = self.cached.entry(*uri);
        if let Some((_, fill)) = self.fills.remove(uri) {
            fill.cancelled.store(true, Ordering::Release);
        }
        match slot {
            Entry::Occupied(entry) if removable(entry.get()) => Some(entry.remove()),
            Entry::Occupied(_) | Entry::Vacant(_) => None,
        }
    }

    /// The one place a background download reserves disk: grow `fill`'s reservation to `bytes`.
    /// With `make_room`, evict to find the room, but never `uri` itself, which would cancel this
    /// download. Without it, use only free space, and return `Ok(false)` when there is not enough.
    ///
    /// Each caller reads the file's entry once and decides both from that one read: with the
    /// entry cached, a query has the file open and the download may evict; with none, it may not.
    async fn reserve_for_fill(
        &self,
        uri: &SuperfileUri,
        fill: &BackgroundFill,
        bytes: u64,
        make_room: bool,
    ) -> Result<bool, DiskCacheError> {
        let short = bytes.saturating_sub(fill.reserved.load(Ordering::Acquire));
        if short == 0 {
            return Ok(true);
        }
        if make_room {
            self.reserve_manual_keeping(short, uri).await?;
        } else if !self.try_reserve_without_evicting(short) {
            return Ok(false);
        }
        fill.reserved.fetch_add(short, Ordering::AcqRel);
        Ok(true)
    }

    /// Rename an installed fill's tempfile to the cache name, after the shard lock is released.
    /// An eviction in between leaves the slot empty, and the renamed file would then belong to no
    /// entry and no budget, so it is deleted. If the rename fails, the tempfile is deleted; the
    /// entry keeps serving from its mmap.
    fn move_into_cache(&self, uri: SuperfileUri, tmp_path: &Path, final_path: &Path) {
        if let Err(error) = fs::rename(tmp_path, final_path) {
            tracing::warn!(target: "infino::cache", uri = %uri.0, %error, "fill: rename into the cache failed");
            let _ = fs::remove_file(tmp_path);
            return;
        }
        // Delete only when the slot is empty: any entry in it may be serving this file, since
        // tier 2 adopts whatever sits under the cache name.
        if !self.cached.contains_key(&uri) {
            let _ = fs::remove_file(final_path);
        }
    }

    /// Hybrid cold-fetch. Returns the foreground reader
    /// (in-memory-bytes-backed) as soon as range-fetches
    /// complete; spawns a background task to fsync + rename +
    /// mmap + register the cache entry. Subsequent callers on
    /// the same URI either see the in-flight OnceCell (same
    /// foreground reader) or, once finalize completes, hit
    /// the mmap-backed cache entry.
    pub(crate) async fn cold_fetch_hybrid(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = storage_key.to_owned();
        let head = fetch_storage.head(&storage_uri).await?;

        let size = head.size;
        // Guarded: any `?` below hands the bytes back. Committed once the entry is admitted; from
        // then on the entry's own accounting carries them.
        let reservation = self.reserve(size).await?;

        let tmp = self.tmp_path(uri);
        let final_path = self.cache_path(uri);

        // 1. Parallel range-GETs. Each task: get_range →
        //    save Bytes for foreground assembly + spawn a
        //    fire-and-forget pwrite.
        let n_streams = self.config.cold_fetch_streams.max(1) as u64;
        let chunk_size = self
            .config
            .cold_fetch_chunk_bytes
            .max(size.div_ceil(n_streams));
        let n_chunks = if size == 0 {
            0
        } else {
            size.div_ceil(chunk_size)
        };

        // Deleted unless the entry is admitted, when the finalizer takes the file over: an open that
        // errors or is dropped midway (a cancelled query) leaves no full-size download on disk
        // that nothing charges. Created here rather than on the blocking pool, so no await falls
        // between the file appearing and its guard: a cancel there would leave the file unguarded.
        let file = fs::File::create(&tmp)?;
        let unclaimed_tmp = UnclaimedTmp(Some(tmp.clone()));
        file.set_len(size)?;
        let file = Arc::new(tokio::sync::Mutex::new(tokio::fs::File::from_std(file)));

        // Per-chunk slot for the foreground buffer assembly.
        let chunks: Arc<tokio::sync::Mutex<Vec<Option<(u64, Bytes)>>>> =
            Arc::new(tokio::sync::Mutex::new(vec![None; n_chunks as usize]));

        // A `JoinSet`, so dropping this open (an error, or a cancelled query) aborts the GETs still
        // in flight rather than leaving them to download the rest of the file.
        let mut fetches = JoinSet::new();
        let mut write_handles = Vec::with_capacity(n_chunks as usize);

        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(&fetch_storage);
            let file = Arc::clone(&file);
            let chunks = Arc::clone(&chunks);
            let uri_s = storage_uri.clone();

            // Spawn the fetch task. It captures a Sender for
            // its pwrite handle so the outer task can join
            // pwrites separately from fetches.
            let (write_tx, write_rx) = oneshot::channel::<JoinHandle<Result<(), DiskCacheError>>>();
            write_handles.push(write_rx);

            fetches.spawn(
                async move {
                    let bytes = storage.get_range(&uri_s, start..end).await?;
                    // Save Bytes for the foreground.
                    {
                        let mut guard = chunks.lock().await;
                        guard[i as usize] = Some((start, bytes.clone()));
                    }
                    // Spawn the pwrite as a fire-and-forget task.
                    // Its JoinHandle goes to the background
                    // finalizer (via oneshot) so the foreground
                    // doesn't wait for it.
                    let pwrite_handle = tokio::spawn(
                        async move {
                            let mut guard = file.lock().await;
                            guard.seek(SeekFrom::Start(start)).await?;
                            guard.write_all(&bytes).await?;
                            Ok::<(), DiskCacheError>(())
                        }
                        .in_current_span(),
                    );
                    let _ = write_tx.send(pwrite_handle);
                    Ok::<(), DiskCacheError>(())
                }
                .in_current_span(),
            );
        }

        // 2. Await all fetches (NOT pwrites). Foreground bytes
        //    are now complete.
        while let Some(fetched) = fetches.join_next().await {
            fetched.map_err(|e| DiskCacheError::SuperfileOpen(format!("fetch join: {e}")))??;
        }

        // 3. Assemble the in-memory buffer for the foreground.
        let buffer = {
            let chunks_guard = chunks.lock().await;
            let mut buf = vec![0u8; size as usize];
            for (start, bytes) in chunks_guard.iter().flatten() {
                let s = *start as usize;
                let e = s + bytes.len();
                buf[s..e].copy_from_slice(bytes);
            }
            buf
        };
        let foreground_bytes = Bytes::from(buffer);
        // Not a landing: this buffer answers the query in flight and the
        // durable copy is verified by the finalizer below, so checking here
        // would scan the same bytes twice on the cold path. The buffer
        // arrived under TLS, which is the integrity guarantee the read path
        // rests on.
        let foreground_reader = SuperfileReader::open_with(
            foreground_bytes,
            OpenOptions {
                verify_crc: self.config.crc_verification.verifies_on_open(),
            },
        )?;
        let foreground_reader = Arc::new(foreground_reader);

        // 4. Construct a CachedEntry with the foreground
        //    reader. Multiple foreground callers waiting on
        //    the coordinator's OnceCell each get an Arc clone
        //    of this reader. Once the background finalizer
        //    completes, the same `cached` slot gets replaced
        //    by a mmap-backed reader; from that point on,
        //    cache hits serve the mmap reader instead.
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&foreground_reader),
            // Hybrid foreground: the whole file is in a heap buffer; the finalizer mmaps it later.
            residency: Residency::Buffered,
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        // Admit the entry so later readers hit tier 1 instead of re-entering the coordinator.
        let entry = self.admit_entry(*uri, entry);
        reservation.commit();
        unclaimed_tmp.claim();

        // 5. Spawn the background finalizer: wait for pwrites,
        //    fsync, rename, mmap, and atomically replace the
        //    cached entry with a mmap-backed reader. On error,
        //    release the manual reservation back to the pool.
        let store = Arc::clone(self);
        // The finalizer may only replace the buffered entry it is promoting; this is how it tells.
        let owner = Arc::downgrade(&foreground_reader);
        let uri_owned = *uri;
        let tmp_owned = tmp.clone();
        let final_owned = final_path.clone();
        let file_owned = Arc::clone(&file);
        // Detached: the finalizer outlives the fetch that spawned it, so it
        // gets a root of its own that merely follows from the caller. See
        // `trace::detached` for why inheriting the caller's span here would
        // corrupt that span's reported duration.
        let finalize_span = detached(debug_span!(
            "cache.finalize_fill",
            uri = %uri_owned.0,
            bytes = size,
            origin = OpOrigin::Maintenance.as_str(),
        ));
        tokio::spawn(
            async move {
                let _ = finalize_to_mmap(
                    store,
                    uri_owned,
                    tmp_owned,
                    final_owned,
                    file_owned,
                    write_handles,
                    size,
                    owner,
                )
                .await;
            }
            .instrument(finalize_span),
        );

        Ok(entry)
    }

    /// Tier 4: fetch from the object store, the one place a fetch shape is chosen. A
    /// [`ReadIntent::Load`] downloads the whole file and mmaps it. `Warm` and `Stream` follow the
    /// configured cold-fetch mode; only `Warm` lets a lazy open start a background fill.
    pub(crate) async fn fetch_from_source(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        intent: ReadIntent,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let fetch_storage = self.resolve_storage(storage);

        if intent == ReadIntent::Load {
            return self.cold_fetch(uri, storage_key, fetch_storage).await;
        }
        match self.config.cold_fetch_mode {
            ColdFetchMode::HybridWithPrefetch => {
                self.cold_fetch_hybrid(uri, storage_key, fetch_storage)
                    .await
            }
            ColdFetchMode::RangeOnly => Err(DiskCacheError::SuperfileOpen(
                "ColdFetchMode::RangeOnly bypasses the disk cache; \
                 construct StorageRangeSource + open_lazy directly"
                    .into(),
            )),
            ColdFetchMode::LazyForegroundWithBackgroundFill => {
                let allow_background_fill = intent == ReadIntent::Warm;
                self.cold_fetch_lazy(
                    uri,
                    storage_key,
                    offsets,
                    fetch_storage,
                    allow_background_fill,
                )
                .await
            }
        }
    }

    /// Download `[start, start + len)` of a lazily opened `uri` into its block cache in a few
    /// parallel GETs, so a scan of that range reads from disk. Meant for a scan that reads most of
    /// the range, such as a pass over the FTS section, whose reads would otherwise each be a GET.
    /// Does nothing when `uri` is not open lazily: a whole file needs no prefetch.
    pub(crate) async fn prefetch_range(
        &self,
        uri: &SuperfileUri,
        start: u64,
        len: u64,
    ) -> Result<(), LazyByteSourceError> {
        let block_source = match self.cached.get(uri) {
            Some(entry) => match &entry.residency {
                Residency::Paged { block_source, .. } => Arc::clone(block_source),
                _ => return Ok(()),
            },
            None => return Ok(()),
        };
        block_source
            .prefetch(
                start,
                len,
                self.config.cold_fetch_chunk_bytes,
                self.config.cold_fetch_streams,
            )
            .await
    }

    /// Start the background download of `uri` from its lazy `entry`, unless the file already has
    /// one (see [`BackgroundFill`]). Only `Warm` reads call this. The vector blob, if any, is left
    /// out of the download and keeps coming from the block cache.
    pub(crate) fn maybe_spawn_background_fill(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        entry: &CachedEntry,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) {
        if skip_background_fill() || entry.has_whole_file() {
            return;
        }
        // Most Warm reads find the download already there, so look under a read lock first.
        let now_us = self.now_us();
        if self
            .fills
            .get(uri)
            .is_some_and(|fill| !fill.may_retry(now_us))
        {
            return;
        }
        // A whole copy already on disk (an earlier download landed it while this entry was being
        // opened) is what the next open maps: nothing to download.
        if self.cache_path(uri).exists() || self.hole_path(uri).exists() {
            return;
        }
        let fill = Arc::<BackgroundFill>::default();
        match self.fills.entry(*uri) {
            Entry::Occupied(current) if !current.get().may_retry(now_us) => return,
            slot => drop(slot.insert(Arc::clone(&fill))),
        }
        // Built before the task is spawned, so the download is unregistered even if the task never
        // runs.
        let exit = FillExit::new(self, *uri, fill);
        // The file may have been evicted since `entry` was looked up. Eviction cancels under the
        // entry's lock (see [`Self::remove_entry_cancelling_fill`]): either it cancelled this
        // download, or the entry is gone here.
        if !self.cached.contains_key(uri) {
            return;
        }
        // A Stream-opened (SourceOwned) entry charges only its filled blocks, so `size_bytes` is
        // not the file size. Take the size from the block source.
        let size = entry
            .block_source()
            .map(|bs| bs.size())
            .filter(|&s| s > 0)
            .unwrap_or_else(|| entry.size_bytes.load(Ordering::Acquire));
        // The vector blob stays in the block cache, so the download writes only the rest.
        let skip_vec = vector_blob_range(&entry.reader);

        // The defer window opens now, not when the task is scheduled, so a queued task cannot
        // extend it.
        let defer = PromotionDefer::start(self.config.promotion_defer_timeout);
        let uri_owned = *uri;
        let storage_uri_owned = storage_key.to_owned();
        let fetch_storage = self.resolve_storage(storage);
        // Detached, and deliberately long-lived: the fill waits for the
        // foreground's lazy readers to release before it downloads. Parenting
        // it to the query that happened to trigger it would bill the query for
        // every second of that wait.
        let fill_span = detached(debug_span!(
            "cache.background_fill",
            uri = %uri_owned.0,
            bytes = size,
            origin = OpOrigin::Maintenance.as_str(),
        ));
        tokio::spawn(
            async move {
                let result = lazy_background_fill(
                    &exit,
                    storage_uri_owned,
                    size,
                    fetch_storage,
                    skip_vec,
                    defer,
                )
                .await;
                match result {
                    Ok(()) => {}
                    // The download's own file is gone: its cache directory was removed (a
                    // dropped store's, say). Like a cancel, nothing to retry.
                    Err(DiskCacheError::Io(error)) if error.kind() == ErrorKind::NotFound => {}
                    // The entry still serves reads. A read after `FAILED_FILL_RETRY_AFTER` tries
                    // again.
                    Err(error) => {
                        tracing::warn!(target: "infino::cache", uri = %uri_owned.0, %error, "fill: download failed");
                        exit.failed();
                    }
                }
            }
            .instrument(fill_span),
        );
    }

    /// Lazy cold-fetch path. Foreground builds a reader via
    /// `SuperfileReader::open_lazy_with(StorageRangeSource)`;
    /// the file's background download (see [`BackgroundFill`]) waits for
    /// foreground lazy readers to release, then downloads the full
    /// superfile and swaps it into the entry.
    ///
    /// If `offsets` is present, the lazy source starts with a known
    /// superfile size and an optional open-batch overlay:
    ///   - with a complete `open_blob`: zero superfile-object GETs at
    ///     open time, because the manifest-part fetch already carried
    ///     the bytes.
    ///   - otherwise: the parquet tail and whichever vector / FTS open
    ///     ranges the blob lacks are fetched in one parallel batch.
    ///
    /// If `offsets` is absent, the source starts with unknown size and
    /// discovers it through the first suffix-tail fetch.
    ///
    /// A Warm open (`allow_background_fill`) reserves the whole file for its
    /// background download. When the budget cannot hold it, the open goes on
    /// as a Stream open does: the entry stays lazy, charges only the blocks
    /// it reads, and later reads of the file find it in the cache. Opening
    /// the file again for each read would cost every read the footer and
    /// column metadata over again, hundreds of small range reads a file.
    pub(crate) async fn cold_fetch_lazy(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        offsets: Option<&SubsectionOffsets>,
        fetch_storage: Arc<dyn StorageProvider>,
        allow_background_fill: bool,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = storage_key.to_owned();

        // The whole file reserved for a background download, so the entry is charged its full
        // size up front; without it the block source charges what it fills. A guard, committed
        // only once the entry is admitted: an open that errors or is dropped midway (a cancelled
        // query) gives the bytes back instead of charging the budget for an entry nobody holds.
        // The block sources are guarded until then too, for a `.blocks` file they adopt.
        let mut unadmitted = UnadmittedBlockSources {
            store: self,
            uri: *uri,
            sources: Vec::new(),
        };
        let (lazy_reader, size, whole_file, block_source) = if let Some(offsets) = offsets {
            let total_size = offsets.total_size;

            let whole_file = self
                .reserve_whole_file(allow_background_fill, total_size)
                .await;
            let eager = whole_file.is_some();

            // Build the lazy source with the size baked in (no HEAD or suffix
            // discovery), then overlay the open-time byte ranges.
            let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
                Arc::clone(&fetch_storage),
                storage_uri.clone(),
                total_size,
            ));
            let block_source = BlockCachedSource::new_with_accounting(
                inner,
                Arc::downgrade(self),
                *uri,
                self.blocks_path(uri),
                !eager,
                SourceLayout {
                    fts_hole: offsets.fts,
                    aligned_region: offsets.vec,
                },
            );
            unadmitted.sources.push(Arc::clone(&block_source));
            let block_source_arc = Arc::clone(&block_source);
            let overlay = open_batch_overlay(
                Arc::clone(&fetch_storage),
                storage_uri.clone(),
                offsets,
                block_source,
            )
            .await?;
            let source: Arc<dyn LazyByteSource> = Arc::new(overlay);

            // Every internal read inside `open_lazy_with` (parquet tail,
            // vec subsection head, fts subsection) hits the overlay sync
            // when the open batch is present. Lazy opens intentionally
            // skip full CRC scans: verifying every subsection would force
            // whole-superfile range reads, defeating the lazy/open-batch
            // path. Eager cache promotion can still verify when it
            // materializes the full superfile.
            let lazy_reader = SuperfileReader::open_lazy_with(
                Arc::clone(&source),
                OpenOptions { verify_crc: false },
            )
            .await?;
            (lazy_reader, total_size, whole_file, block_source_arc)
        } else {
            // Unknown-size path: avoid the cold-open HEAD round-trip.
            // The first `tail()` inside `open_lazy_with` is a native
            // suffix-range GET that returns both footer bytes and total
            // object size, then patches the source's size atomic. The size
            // is known only after the open, so a Warm open reserves after
            // it; when the budget cannot hold the file, the source built to
            // ride on that reservation is dropped and the file is opened
            // once more as a Stream open, charging its own blocks (the
            // blocks the first open wrote are adopted when their index has
            // landed, fetched again otherwise).
            let opened = self
                .open_lazy_unknown_size(
                    uri,
                    &storage_uri,
                    &fetch_storage,
                    !allow_background_fill,
                    &mut unadmitted,
                )
                .await?;
            let whole_file = self
                .reserve_whole_file(allow_background_fill, opened.1)
                .await;

            let (lazy_reader, size, block_source) = if allow_background_fill && whole_file.is_none()
            {
                drop(opened);
                self.open_lazy_unknown_size(
                    uri,
                    &storage_uri,
                    &fetch_storage,
                    true,
                    &mut unadmitted,
                )
                .await?
            } else {
                opened
            };
            (lazy_reader, size, whole_file, block_source)
        };

        let eager = whole_file.is_some();

        // A Stream open, or a Warm open the budget could not hold whole, charges only the blocks
        // it fills (SourceOwned), so a fanout over many superfiles does not reserve them all in
        // full and evict live peers. A Warm open the budget holds will download the whole file,
        // and has reserved the full size above.
        let lazy_reader = Arc::new(lazy_reader);
        let (size_bytes, accounting) = if eager {
            (Arc::new(AtomicU64::new(size)), EntryAccounting::Eager)
        } else {
            (
                block_source.filled_bytes_handle(),
                EntryAccounting::SourceOwned,
            )
        };
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&lazy_reader),
            residency: Residency::Paged {
                block_source: Arc::clone(&block_source),
            },
            size_bytes,
            accounting,
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);

        // Admission may return an existing whole file instead; serve what it returns. Either way
        // admission takes the reserved bytes over: the new entry holds them, or admission
        // releases them when a whole file already holds the slot.
        let admitted = self.admit_entry(*uri, Arc::clone(&entry));
        if let Some(reservation) = whole_file {
            reservation.commit();
        }
        if Arc::ptr_eq(&admitted, &entry) {
            unadmitted.admitted();
        }

        Ok(admitted)
    }

    /// A lazy open of `storage_uri` whose size is not known up front: the first `tail()` inside
    /// `open_lazy_with` is a suffix-range GET that returns the footer bytes and the object's size.
    /// `owns_accounting` is the block source's (see [`BlockCachedSource::new_with_accounting`]),
    /// and the source joins `unadmitted`. Returns the reader, the size the open discovered and the
    /// block source it reads through.
    async fn open_lazy_unknown_size(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_uri: &str,
        fetch_storage: &Arc<dyn StorageProvider>,
        owns_accounting: bool,
        unadmitted: &mut UnadmittedBlockSources<'_>,
    ) -> Result<(SuperfileReader, u64, Arc<BlockCachedSource>), DiskCacheError> {
        let range_src: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_unknown_size(
            Arc::clone(fetch_storage),
            storage_uri.to_owned(),
        ));
        let block_source = BlockCachedSource::new_with_accounting(
            range_src,
            Arc::downgrade(self),
            *uri,
            self.blocks_path(uri),
            owns_accounting,
            // No manifest hints here, so the layout is unknown.
            SourceLayout::ALL_ALIGNED,
        );
        unadmitted.sources.push(Arc::clone(&block_source));
        let source: Arc<dyn LazyByteSource> = Arc::clone(&block_source) as Arc<dyn LazyByteSource>;
        let lazy_reader =
            SuperfileReader::open_lazy_with(Arc::clone(&source), OpenOptions { verify_crc: false })
                .await?;
        let size = source.size();
        Ok((lazy_reader, size, block_source))
    }

    /// Download the whole file to local disk and admit it as a mapped entry. What a `Load` miss
    /// uses.
    pub(crate) async fn cold_fetch(
        &self,
        uri: &SuperfileUri,
        storage_key: &str,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = storage_key.to_owned();
        let head = fetch_storage.head(&storage_uri).await?;
        let size = head.size;

        // Reserve budget (CAS-loop with eviction on miss).
        let reservation = self.reserve(size).await?;

        // Pump bytes from storage to a sparse destination.
        let tmp = self.tmp_path(uri);
        let final_path = self.cache_path(uri);
        self.cold_fetch_to_disk(&fetch_storage, &storage_uri, &tmp, size)
            .await?;

        // A landing: the whole superfile has just been written to local disk,
        // the one copy nothing else guards. Verify it once here rather than
        // on every open of it hereafter.
        //
        // Before the promote, not after, and that order is the whole of it.
        // A file that fails here must never reach the name a later read — or
        // a later process scanning this directory — would adopt, because
        // adoption does not re-check it. The two other fill paths verify
        // their tempfile for the same reason; the mmap survives the rename,
        // since it is the same inode under a new name.
        let (mmap, bytes) = mmap_readonly_with_handle(&tmp).map_err(DiskCacheError::Io)?;
        let reader = match SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: self.config.crc_verification.verifies_on_landing(),
            },
        ) {
            Ok(reader) => reader,
            Err(error) => {
                let _ = fs::remove_file(&tmp);
                return Err(error.into());
            }
        };
        tokio::fs::rename(&tmp, &final_path).await?;
        let entry = self.build_mmap_entry(Arc::new(reader), mmap, size, None);

        let entry = self.admit_entry(*uri, entry);
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);

        reservation.commit();

        Ok(entry)
    }

    /// Fetch `size` bytes from `storage_uri` into `dest_path` via parallel range-GETs. Each chunk
    /// is its own spawned task capped by a semaphore; positioned (`pwrite`) writes go straight to
    /// the file with no shared lock. Foreground path: a query is waiting on the whole file.
    pub(crate) async fn cold_fetch_to_disk(
        &self,
        fetch_storage: &Arc<dyn StorageProvider>,
        storage_uri: &str,
        dest_path: &Path,
        size: u64,
    ) -> Result<(), DiskCacheError> {
        let n_streams = self.config.cold_fetch_streams.max(1);
        // Fixed chunk size — do NOT scale with `size`. Peak
        // in-flight memory is `n_streams × chunk_size`
        // regardless of superfile size, because the per-fill
        // semaphore below caps concurrent chunks at `n_streams`.
        let chunk_size = self.config.cold_fetch_chunk_bytes.max(1);

        // Preallocate the destination as a plain `std::fs::File`
        // so chunk writers can use positioned (`pwrite`) writes
        // off the async reactor without a shared file lock.
        let file = {
            let f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(dest_path)?;
            f.set_len(size)?;
            Arc::new(f)
        };

        let n_chunks = if size == 0 {
            0
        } else {
            size.div_ceil(chunk_size)
        };
        // Per-fill concurrency cap: at most `n_streams` chunks
        // hold their fetched `Bytes` resident at once.
        let stream_sem = Arc::new(tokio::sync::Semaphore::new(n_streams));
        let mut joins = Vec::with_capacity(n_chunks as usize);
        for i in 0..n_chunks {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(size);
            let storage = Arc::clone(fetch_storage);
            let file = Arc::clone(&file);
            let uri = storage_uri.to_string();
            let stream_sem = Arc::clone(&stream_sem);
            joins.push(tokio::spawn(
                async move {
                    let _permit = stream_sem.acquire_owned().await.map_err(|e| {
                        DiskCacheError::SuperfileOpen(format!("stream semaphore closed: {e}"))
                    })?;
                    fetch_and_pwrite(&storage, &uri, &file, start, end, false).await
                }
                .in_current_span(),
            ));
        }
        for h in joins {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("join error: {e}")))??;
        }
        spawn_blocking(carry_span(move || file.sync_all()))
            .await
            .map_err(|e| DiskCacheError::SuperfileOpen(format!("fsync join: {e}")))??;
        Ok(())
    }
}

pub(crate) struct PromotionWaitGuard<'a>(&'a AtomicU64);

impl<'a> PromotionWaitGuard<'a> {
    pub(crate) fn new(counter: &'a AtomicU64) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for PromotionWaitGuard<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Background finalizer for the hybrid cold-fetch. Awaits all pwrites, fsyncs the tempfile, mmaps
/// it, and swaps the buffered entry for the mmap-backed one (the install renames the file into the
/// cache). On failure, drops the buffered entry and its budget, and the tempfile.
async fn finalize_to_mmap(
    store: Arc<DiskCacheStore>,
    uri: SuperfileUri,
    tmp_path: PathBuf,
    final_path: PathBuf,
    file: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    pwrite_handles: Vec<oneshot::Receiver<JoinHandle<Result<(), DiskCacheError>>>>,
    size: u64,
    owner: Weak<SuperfileReader>,
) -> Result<(), DiskCacheError> {
    let res: Result<(), DiskCacheError> = async {
        // 1. Resolve every pwrite handle through its oneshot,
        //    then await the underlying join.
        for recv in pwrite_handles {
            let handle = recv
                .await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("pwrite handle: {e}")))?;
            handle
                .await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("pwrite join: {e}")))??;
        }
        // 2. fsync + drop the file before rename.
        {
            let mut guard = file.lock().await;
            guard.flush().await?;
            guard.sync_all().await?;
        }
        drop(file);
        // A landing: the finalizer has just completed the file on disk.
        let (mmap, bytes) = mmap_readonly_with_handle(&tmp_path)?;
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: store.config.crc_verification.verifies_on_landing(),
            },
        )?;
        // Swap in over the buffered entry, which already carries the whole-size charge. The
        // finalizer reserved nothing of its own.
        let entry = store.build_mmap_entry(Arc::new(reader), mmap, size, None);
        store.install_promoted_entry(
            uri,
            entry,
            &tmp_path,
            &final_path,
            // The entry it promotes, or any lazy one; never someone else's whole file.
            |current| is_mine(current, &owner) || !current.has_whole_file(),
            0,
        );
        Ok::<(), DiskCacheError>(())
    }
    .await;
    if res.is_err() {
        // Drop only the entry this finalizer was promoting, and its partial download.
        if let Some((_, mine)) = store
            .cached
            .remove_if(&uri, |_, entry| is_mine(entry, &owner))
        {
            store.release_entry_accounting(&mine);
        }
        let _ = fs::remove_file(&tmp_path);
    }
    res
}

/// The `(offset, len)` ranges of `wanted` that no entry of `blob`
/// covers whole — what the open wave still has to fetch when the
/// manifest inlined only part of the open batch.
fn uncovered_ranges(blob: &[(u64, Vec<u8>)], wanted: &[(u64, u64)]) -> Vec<(u64, u64)> {
    wanted
        .iter()
        .copied()
        .filter(|&(off, len)| {
            len > 0
                && !blob
                    .iter()
                    .any(|(start, bytes)| *start <= off && off + len <= *start + bytes.len() as u64)
        })
        .collect()
}

/// The open batch of a lazy open over `inner`, a range source of the object
/// `offsets` describes, in an overlay in front of it. Whatever the manifest
/// part carried in its open blob rides straight in; the rest, the parquet
/// tail and any open range the writer left out of the blob because copying
/// it into every manifest read would cost more than the round trip (a large
/// vector index, or the term dictionary an older writer recorded as an open
/// range), is fetched over the wire in one parallel wave. A complete blob
/// means zero open-time GETs against the object. Older or incomplete hints
/// fall back to fixed headers; the readers then discover the rest. Shared by
/// the disk cache's lazy open and the cacheless fallback, so both open a
/// superfile with the same round trips.
pub(crate) async fn open_batch_overlay(
    storage: Arc<dyn StorageProvider>,
    storage_uri: String,
    offsets: &SubsectionOffsets,
    inner: Arc<dyn LazyByteSource>,
) -> Result<PrefetchedSource, StorageError> {
    let total_size = offsets.total_size;
    // Match `SuperfileReader::open_lazy_with`'s parquet tail
    // speculation length so the overlay covers the entire
    // upcoming `source.tail()` call.
    let parquet_tail_len = DEFAULT_TAIL_SPECULATIVE_BYTES.min(total_size);
    let parquet_tail_start = total_size.saturating_sub(parquet_tail_len);

    // Seed the inner lazy readers with exact open-time metadata
    // when the manifest carries it. Older/incomplete hints fall
    // back to fixed headers; the readers then discover the rest.
    let vec_ranges = if !offsets.vec_open_ranges.is_empty() {
        offsets.vec_open_ranges.clone()
    } else {
        match offsets.vec {
            Some((off, len)) if len > 0 => {
                vec![(off, VECTOR_OPEN_HEADER_FALLBACK_BYTES.min(len))]
            }
            _ => Vec::new(),
        }
    };
    let fts_ranges = if !offsets.fts_open_ranges.is_empty() {
        offsets.fts_open_ranges.clone()
    } else {
        match offsets.fts {
            Some((off, len)) if len > 0 => {
                vec![(off, FTS_OPEN_HEADER_FALLBACK_BYTES.min(len))]
            }
            _ => Vec::new(),
        }
    };
    let mut overlay = PrefetchedSource::new(inner);
    for (off, bytes) in &offsets.open_blob {
        overlay.install(*off, Bytes::copy_from_slice(bytes));
    }
    let tail_range = (parquet_tail_start, parquet_tail_len);
    let tail_missing =
        parquet_tail_len > 0 && !uncovered_ranges(&offsets.open_blob, &[tail_range]).is_empty();
    let vec_missing = uncovered_ranges(&offsets.open_blob, &vec_ranges);
    let fts_missing = uncovered_ranges(&offsets.open_blob, &fts_ranges);
    if tail_missing || !vec_missing.is_empty() || !fts_missing.is_empty() {
        let storage_for_parquet = Arc::clone(&storage);
        let storage_for_vec = Arc::clone(&storage);
        let storage_for_fts = Arc::clone(&storage);
        let parquet_uri = storage_uri.clone();
        let vec_uri = storage_uri.clone();
        let fts_uri = storage_uri.clone();

        let parquet_fut = async move {
            if !tail_missing {
                return Ok::<_, StorageError>(Bytes::new());
            }
            storage_for_parquet
                .get_range(&parquet_uri, parquet_tail_start..total_size)
                .await
        };
        let vec_fut = async move { fetch_hint_ranges(storage_for_vec, vec_uri, vec_missing).await };
        let fts_fut = async move { fetch_hint_ranges(storage_for_fts, fts_uri, fts_missing).await };

        let (parquet_bytes, vec_pre, fts_pre) = futures::try_join!(parquet_fut, vec_fut, fts_fut)?;
        if !parquet_bytes.is_empty() {
            overlay.install(parquet_tail_start, parquet_bytes);
        }
        for (off, bytes) in vec_pre {
            overlay.install(off, bytes);
        }
        for (off, bytes) in fts_pre {
            overlay.install(off, bytes);
        }
    }
    Ok(overlay)
}

async fn fetch_hint_ranges(
    storage: Arc<dyn StorageProvider>,
    storage_uri: String,
    ranges: Vec<(u64, u64)>,
) -> Result<Vec<(u64, Bytes)>, StorageError> {
    try_join_all(
        ranges
            .into_iter()
            .filter(|&(_, len)| len > 0)
            .map(|(off, len)| {
                let storage = Arc::clone(&storage);
                let storage_uri = storage_uri.clone();
                async move {
                    let bytes = storage.get_range(&storage_uri, off..off + len).await?;
                    Ok::<_, StorageError>((off, bytes))
                }
            }),
    )
    .await
}

fn background_store_abandoned(store: &Arc<DiskCacheStore>) -> bool {
    Arc::strong_count(store) == 1
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundFillOutcome {
    Complete,
    Paused,
    Abandoned,
}

/// Whether a background download must stop now: `Abandoned` when its store was dropped or it was
/// cancelled, `Paused` when a query holds its file inside the defer window.
fn download_interrupted(
    store: &Arc<DiskCacheStore>,
    uri: &SuperfileUri,
    fill: &BackgroundFill,
    defer: PromotionDefer,
) -> Option<BackgroundFillOutcome> {
    if background_store_abandoned(store) || fill.is_cancelled() {
        Some(BackgroundFillOutcome::Abandoned)
    } else if defer.yields_to(store, uri) {
        Some(BackgroundFillOutcome::Paused)
    } else {
        None
    }
}

/// Fetch one byte range and pwrite it into `file` at `range_start`. The positioned write runs on
/// the blocking pool, off the async reactor. `background` tags the GET so meters attribute it to
/// fill rather than to a foreground query. Shared by both byte pumps.
async fn fetch_and_pwrite(
    storage: &Arc<dyn StorageProvider>,
    uri: &str,
    file: &Arc<fs::File>,
    range_start: u64,
    range_end: u64,
    background: bool,
) -> Result<(), DiskCacheError> {
    let get = storage.get_range(uri, range_start..range_end);
    let bytes = if background {
        scope_background(get).await?
    } else {
        get.await?
    };
    let file = Arc::clone(file);
    spawn_blocking(carry_span(move || file.write_all_at(&bytes, range_start)))
        .await
        .map_err(|error| DiskCacheError::SuperfileOpen(format!("write join: {error}")))??;
    Ok(())
}

async fn cold_fetch_to_disk_cancelable(
    store: &Arc<DiskCacheStore>,
    uri: &SuperfileUri,
    fill: &BackgroundFill,
    fetch_storage: &Arc<dyn StorageProvider>,
    storage_uri: &str,
    dest_path: &Path,
    size: u64,
    filled: &mut Vec<bool>,
    skip_vec: Option<(u64, u64)>,
    defer: PromotionDefer,
) -> Result<BackgroundFillOutcome, DiskCacheError> {
    let n_streams = store.config.cold_fetch_streams.max(1);
    let chunk_size = store.config.cold_fetch_chunk_bytes.max(1);
    let n_chunks = if size == 0 {
        0
    } else {
        size.div_ceil(chunk_size)
    };
    // `filled` is the resume cursor, owned by the caller across pause/resume:
    // an entry is `true` once its chunk is durably written. On the first
    // attempt it is empty; size it and truncate the destination. On a resume
    // (a same-URI reader paused the previous attempt) it carries the
    // already-written chunks, so the fetch skips them instead of
    // re-downloading the whole object from byte 0.
    let first_attempt = filled.len() != n_chunks as usize;
    if first_attempt {
        filled.clear();
        filled.resize(n_chunks as usize, false);
    }
    let file = {
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true);
        if first_attempt {
            opts.truncate(true);
        }
        let file = opts.open(dest_path)?;
        if first_attempt {
            file.set_len(size)?;
        }
        Arc::new(file)
    };

    let mut next_chunk = 0u64;
    let mut in_flight = FuturesUnordered::new();

    // Bound memory by `n_streams × chunk_size` and stop promptly when the
    // short-lived cache that requested this background fill is dropped.
    loop {
        while next_chunk < n_chunks && in_flight.len() < n_streams {
            // Skip chunks a prior attempt already wrote (resume cursor).
            if filled[next_chunk as usize] {
                next_chunk += 1;
                continue;
            }
            if let Some(stop) = download_interrupted(store, uri, fill, defer) {
                return Ok(stop);
            }
            let chunk_idx = next_chunk;
            let start = chunk_idx * chunk_size;
            let end = (start + chunk_size).min(size);
            // Vector blob stays on the block cache: leave those bytes sparse
            // in the fill file (no GET). Parquet + FTS ranges still download.
            let fetch_ranges = chunk_fetch_ranges(start, end, skip_vec);
            if fetch_ranges.is_empty() {
                filled[chunk_idx as usize] = true;
                next_chunk += 1;
                continue;
            }
            let storage = Arc::clone(fetch_storage);
            let file = Arc::clone(&file);
            let key = storage_uri.to_string();
            // Fill GETs are tagged background (see `fetch_and_pwrite`) so query-window meters
            // attribute only foreground lazy/probe GETs to the cold query cost.
            in_flight.push(async move {
                for (range_start, range_end) in fetch_ranges {
                    fetch_and_pwrite(&storage, &key, &file, range_start, range_end, true).await?;
                }
                Ok::<u64, DiskCacheError>(chunk_idx)
            });
            next_chunk += 1;
        }

        let foreground = foreground_notify().notified();
        tokio::pin!(foreground);
        let _ = foreground.as_mut().enable();
        if let Some(stop) = download_interrupted(store, uri, fill, defer) {
            return Ok(stop);
        }
        tokio::select! {
            biased;
            _ = &mut foreground => {
                // A query started: re-check same-URI hold. Unrelated fills
                // (file not in use) fall through and keep downloading.
                if let Some(stop) = download_interrupted(store, uri, fill, defer) {
                    return Ok(stop);
                }
            }
            result = in_flight.next() => match result {
                // Mark the chunk durable only once its write completes, so a
                // pause mid-flight re-fetches just the unfinished chunks.
                Some(result) => filled[result? as usize] = true,
                None => break,
            }
        }
        if background_store_abandoned(store) {
            return Ok(BackgroundFillOutcome::Abandoned);
        }
    }

    if let Some(stop) = download_interrupted(store, uri, fill, defer) {
        return Ok(stop);
    }
    spawn_blocking(carry_span(move || file.sync_all()))
        .await
        .map_err(|error| DiskCacheError::SuperfileOpen(format!("fsync join: {error}")))??;
    Ok(BackgroundFillOutcome::Complete)
}

/// Is `entry` the one the hybrid finalizer is promoting? Same reader, by pointer.
fn is_mine(entry: &CachedEntry, owner: &Weak<SuperfileReader>) -> bool {
    owner
        .upgrade()
        .is_some_and(|mine| Arc::ptr_eq(&entry.reader, &mine))
}

/// Diagnostic gate for measuring lazy foreground reads without promotion,
/// from `diagnostics.disable_background_fill` (YAML-only; no env override).
pub(crate) fn skip_background_fill() -> bool {
    global_config().diagnostics.disable_background_fill
}

/// Ends a background download however it ends, even when its task never ran: unregisters it so
/// a later read can start a new one, gives back what it reserved, and deletes the partial file. A
/// failed download stays registered (see [`BackgroundFill::may_retry`]).
struct FillExit {
    store: Weak<DiskCacheStore>,
    uri: SuperfileUri,
    fill: Arc<BackgroundFill>,
    tmp: PathBuf,
}

impl FillExit {
    /// Count `fill`, just registered for `uri`, as running, and take over its ending.
    fn new(store: &Arc<DiskCacheStore>, uri: SuperfileUri, fill: Arc<BackgroundFill>) -> Self {
        store.fills_running.fetch_add(1, Ordering::AcqRel);
        Self {
            store: Arc::downgrade(store),
            uri,
            fill,
            tmp: store.tmp_path(&uri),
        }
    }

    /// Mark the download failed: it stays registered, so the file is not downloaded again before
    /// [`FAILED_FILL_RETRY_AFTER`].
    fn failed(&self) {
        if let Some(store) = self.store.upgrade() {
            let _ = self.fill.failed_at_us.set(store.now_us());
        }
    }
}

impl Drop for FillExit {
    fn drop(&mut self) {
        // A dropped store took the reservation with it.
        if let Some(store) = self.store.upgrade() {
            if self.fill.failed_at_us.get().is_none() {
                // Only this download: a cancel unregisters it at once, and a newer one for the same
                // file must stay.
                store
                    .fills
                    .remove_if(&self.uri, |_, fill| Arc::ptr_eq(fill, &self.fill));
            }
            self.fill.downloading.store(false, Ordering::Release);
            store.release_block_bytes(self.fill.reserved.swap(0, Ordering::AcqRel));
            store.fills_running.fetch_sub(1, Ordering::AcqRel);
        }
        let _ = fs::remove_file(&self.tmp);
    }
}

/// The background download of a file (see [`BackgroundFill`]), step by step:
/// 1. wait until no query holds the file, or `defer` runs out;
/// 2. wait for a download slot;
/// 3. reserve what the file's entry has not reserved already: the file minus its vector blob when
///    `skip_vec` leaves the blob in the block cache (see [`HoleFallbackSource`]);
/// 4. download, pausing while a query reads the file inside `defer`;
/// 5. swap the file into its lazy entry, or, with no entry, leave it on disk for the next read.
async fn lazy_background_fill(
    exit: &FillExit,
    storage_uri: String,
    size: u64,
    fetch_storage: Arc<dyn StorageProvider>,
    skip_vec: Option<(u64, u64)>,
    defer: PromotionDefer,
) -> Result<(), DiskCacheError> {
    let FillExit {
        store: weak_store,
        uri,
        fill,
        tmp,
    } = exit;

    // Hold the store only weakly while waiting: a queue can hold a download for every file a scan
    // touched, and must not keep a dropped store alive.
    let Some(store) = FillWaiters::park(weak_store, uri, fill, defer, FillWait::Release).await
    else {
        return Ok(());
    };
    if !store.fill_can_install(uri, fill) {
        return Ok(());
    }
    let semaphore = Arc::clone(&store.prefetch_semaphore);
    drop(store);
    let _prefetch_permit = semaphore.acquire_owned().await.map_err(|error| {
        DiskCacheError::SuperfileOpen(format!("prefetch semaphore closed: {error}"))
    })?;
    let Some(store) = weak_store.upgrade() else {
        return Ok(());
    };
    // Cancelled while it waited, or a whole copy landed: reserve nothing.
    if !store.fill_can_install(uri, fill) {
        return Ok(());
    }
    // From here the idle cap keeps the file's entry, so the entry's Warm reservation covers the
    // download (see `reclaimable_lazy_entry`). With no entry, nobody has the file open: the
    // download reserves the whole file it will leave on disk, from free space only.
    // Set the flag before reading the entry: the cap checks it under the entry's lock.
    fill.downloading.store(true, Ordering::Release);
    let reserved = match store.entry_whole_file_charge(uri) {
        Some(held) => {
            // The vector blob stays in the block cache, so the download writes only the rest.
            let file_bytes = skip_vec.map_or(size, |(_, vec_len)| size.saturating_sub(vec_len));
            store
                .reserve_for_fill(uri, fill, file_bytes.saturating_sub(held), true)
                .await?
        }
        None => store.reserve_for_fill(uri, fill, size, false).await?,
    };
    if !reserved {
        return Ok(());
    }
    drop(store);
    // Resume cursor: chunks durably written so far, preserved across
    // pause/resume so a same-URI reader interrupting the fill costs only
    // the unfinished chunks rather than a re-download of the whole object.
    let mut filled: Vec<bool> = Vec::new();
    let store = loop {
        let Some(store) =
            FillWaiters::park(weak_store, uri, fill, defer, FillWait::Quiescence).await
        else {
            return Ok(());
        };
        // Cancelled, or a whole copy is cached: the download would be thrown away.
        if !store.fill_can_install(uri, fill) {
            return Ok(());
        }
        let outcome = cold_fetch_to_disk_cancelable(
            &store,
            uri,
            fill,
            &fetch_storage,
            &storage_uri,
            tmp,
            size,
            &mut filled,
            skip_vec,
            defer,
        )
        .await?;
        match outcome {
            BackgroundFillOutcome::Complete => break store,
            // Keep the partial `tmp` and the `filled` cursor: the next attempt resumes from the
            // first unwritten chunk. The store is dropped here, so it is not held while parked.
            BackgroundFillOutcome::Paused => {}
            BackgroundFillOutcome::Abandoned => return Ok(()),
        }
    };

    // Cancelled, or a whole copy landed during the download: nothing to install.
    if !store.fill_can_install(uri, fill) {
        return Ok(());
    }

    // A copy missing its vector blob goes under its own name, so that nothing, a restart included,
    // can take it for a complete file.
    let holed = skip_vec.is_some();
    let final_path = if holed {
        store.hole_path(uri)
    } else {
        store.cache_path(uri)
    };

    // No entry holds the file: leave the download on disk, and the next read opens it from there.
    let Some(block_source) = store
        .cached
        .get(uri)
        .and_then(|entry| entry.block_source().cloned())
    else {
        // Check the bytes as an install would: a file under the cache name has always passed.
        if !holed && store.config.crc_verification.verifies_on_landing() {
            let (_, mapped) = mmap_readonly_with_handle(tmp)?;
            SuperfileReader::open_with(mapped, OpenOptions { verify_crc: true })?;
        }
        // Nobody has the file open, so use only free space. With none, the download failed, and
        // a later read may try again after the retry wait.
        if !store.reserve_for_fill(uri, fill, size, false).await? {
            return Err(DiskCacheError::BudgetExceeded);
        }
        let reserved = fill.reserved.swap(0, Ordering::AcqRel);
        store.land_on_disk(uri, fill, tmp, &final_path, holed, size, reserved);
        return Ok(());
    };

    // The file's lazy entry is cached: open the download and swap it in.
    let (mmap_arc, mapped) = mmap_readonly_with_handle(tmp)?;
    let entry = match skip_vec {
        // Keep the entry's block cache for the vector blob, so the vector ranges already read stay
        // local.
        Some(hole) => {
            store
                .holed_mmap_entry(mmap_arc, mapped, hole, block_source, size)
                .await?
        }
        None => {
            let reader = SuperfileReader::open_with(
                mapped,
                OpenOptions {
                    verify_crc: store.config.crc_verification.verifies_on_landing(),
                },
            )?;
            store.build_mmap_entry(Arc::new(reader), mmap_arc, size, None)
        }
    };

    // The new entry may cost more than the download reserved (a holed copy whose block source
    // reserves nothing for itself costs the whole file). What the old entry reserved goes to the
    // install, so top up only the rest; the download is done, so make room for it.
    let charge = entry.size_bytes.load(Ordering::Acquire);
    let held_by_entry = store.entry_whole_file_charge(uri).unwrap_or(0);
    store
        .reserve_for_fill(uri, fill, charge.saturating_sub(held_by_entry), true)
        .await?;
    // The install keeps the reservation for the new entry or gives it back; the exit then has
    // nothing left to give back.
    let own_reservation = fill.reserved.swap(0, Ordering::AcqRel);
    let installed = store.install_promoted_entry(
        *uri,
        entry,
        tmp,
        &final_path,
        |current| fill.may_replace(Some(current)),
        own_reservation,
    );
    // With no vector hole the mmap serves every range, and the block file is dead weight.
    if installed.is_some() && !holed {
        store.drop_block_file(uri);
    }
    Ok(())
}

/// Absolute `(offset, length)` of the vector blob from Parquet KV metadata.
fn vector_blob_range(reader: &SuperfileReader) -> Option<(u64, u64)> {
    vector_blob_range_in(&footer::extract_kv_map(&reader.parquet_metadata()).ok()?)
}

fn vector_blob_range_in(kv_map: &footer::KvMap) -> Option<(u64, u64)> {
    let off: u64 = kv_map.get(kv::VEC_OFFSET)?.parse().ok()?;
    let len: u64 = kv_map.get(kv::VEC_LENGTH)?.parse().ok()?;
    (len > 0).then_some((off, len))
}

/// Sub-ranges of `[start, end)` that are outside an optional skip hole.
///
/// Empty means the whole chunk lies inside the hole (no GET).
fn chunk_fetch_ranges(start: u64, end: u64, skip: Option<(u64, u64)>) -> Vec<(u64, u64)> {
    debug_assert!(start <= end);
    let Some((hole_start, hole_len)) = skip else {
        return vec![(start, end)];
    };
    if hole_len == 0 || start == end {
        return vec![(start, end)];
    }
    let hole_end = hole_start.saturating_add(hole_len);
    if end <= hole_start || start >= hole_end {
        return vec![(start, end)];
    }
    let mut out = Vec::with_capacity(2);
    if start < hole_start {
        out.push((start, hole_start.min(end)));
    }
    if end > hole_end {
        out.push((hole_end.max(start), end));
    }
    out
}

/// Local mmap/bytes source with a hole that falls through to another source.
///
/// Used after background fill excludes the vector blob: parquet + FTS come
/// from the filled mmap; vector ranges keep using the block cache.
struct HoleFallbackSource {
    local: Arc<dyn LazyByteSource>,
    hole_start: u64,
    hole_len: u64,
    fallback: Arc<BlockCachedSource>,
}

impl HoleFallbackSource {
    fn hole_end(&self) -> u64 {
        self.hole_start.saturating_add(self.hole_len)
    }

    fn overlaps_hole(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        end > self.hole_start && start < self.hole_end()
    }

    fn fully_in_hole(&self, start: u64, len: u64) -> bool {
        let end = start.saturating_add(len);
        start >= self.hole_start && end <= self.hole_end()
    }
}

#[async_trait]
impl LazyByteSource for HoleFallbackSource {
    fn size(&self) -> u64 {
        self.local.size()
    }

    async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
        if len == 0 {
            return Ok(Bytes::new());
        }
        if !self.overlaps_hole(start, len) {
            return self.local.range(start, len).await;
        }
        if self.fully_in_hole(start, len) {
            return self.fallback.range(start, len).await;
        }
        // Spanning request: stitch local and fallback pieces in order.
        let end = start + len;
        let hole_end = self.hole_end();
        let mut pieces = Vec::with_capacity(3);
        let mut cursor = start;
        if cursor < self.hole_start {
            let piece_end = self.hole_start.min(end);
            pieces.push(self.local.range(cursor, piece_end - cursor).await?);
            cursor = piece_end;
        }
        if cursor < end && cursor < hole_end {
            let piece_end = hole_end.min(end);
            pieces.push(self.fallback.range(cursor, piece_end - cursor).await?);
            cursor = piece_end;
        }
        if cursor < end {
            pieces.push(self.local.range(cursor, end - cursor).await?);
        }
        if pieces.len() == 1 {
            return Ok(pieces.pop().expect("one piece"));
        }
        let mut out = Vec::with_capacity(len as usize);
        for piece in pieces {
            out.extend_from_slice(&piece);
        }
        Ok(Bytes::from(out))
    }

    fn is_local(&self, start: u64, len: u64) -> bool {
        if len == 0 {
            return true;
        }
        let end = start.saturating_add(len);
        let hole_start = self.hole_start.clamp(start, end);
        let hole_end = self.hole_end().clamp(start, end);
        // The parts before and after the hole come from the local copy.
        (hole_start == start || self.local.is_local(start, hole_start - start))
            && (hole_end == hole_start || self.fallback.is_local(hole_start, hole_end - hole_start))
            && (hole_end == end || self.local.is_local(hole_end, end - hole_end))
    }

    fn note_read(&self, start: u64, len: u64) {
        self.local.note_read(start, len);
        self.fallback.note_read(start, len);
    }

    fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
        if len == 0 {
            return Some(Bytes::new());
        }
        if !self.overlaps_hole(start, len) {
            return self.local.try_get_range_sync(start, len);
        }
        if self.fully_in_hole(start, len) {
            return self.fallback.try_get_range_sync(start, len);
        }
        // Spanning sync reads are rare; force the async path.
        None
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs,
        future::poll_fn,
        os::unix::fs::MetadataExt,
        path::PathBuf,
        sync::{
            Arc,
            atomic::{AtomicU64, AtomicUsize, Ordering},
        },
        thread,
        time::{Duration, Instant},
    };

    use bytes::Bytes;
    use futures::FutureExt;
    use tempfile::TempDir;
    use tokio::{
        runtime::Builder,
        spawn,
        task::yield_now,
        time::{sleep, timeout},
    };

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            LazyByteSource,
            reader::{OpenOptions, SuperfileReader},
        },
        supertable::{
            StorageRangeSource,
            manifest::{SubsectionOffsets, SuperfileUri},
            reader_cache::{
                OpenTier,
                block_source::{
                    BlockCachedSource, CACHE_BLOCK_BYTES, SourceLayout, indexed_filled_bytes,
                },
                config::{ColdFetchMode, DiskCacheConfig},
                disk::{
                    CachedEntry, EntryAccounting, Residency, fetch::*,
                    sources::mmap_readonly_with_handle, test_support::*,
                },
            },
        },
        test_helpers::fault_storage::{FaultOp, FaultStorage},
    };

    /// Fills parked behind one held reader in the polling test.
    const PARKED_FILLS: usize = 64;

    /// How long the short-lived runtime in the runtime-shutdown test lives.
    const SHORT_RUNTIME_LIFE: Duration = Duration::from_millis(50);

    /// How long a test watches a parked fill: many ticks of the store's ticker.
    const HELD_FOR: Duration = Duration::from_millis(200);

    /// A defer window short enough for a test to wait out.
    const SHORT_WINDOW: Duration = Duration::from_millis(100);

    /// Chunk size of the download in the reopen test: small enough that the tiny fixture takes
    /// several, so its first chunk's GET stands apart from every other read.
    const DOWNLOAD_CHUNK_BYTES: u64 = 256;

    /// Bytes a fill test reserves, and the size of the file it pretends to fill.
    const RESERVED_BYTES: u64 = 4096;

    /// Poll cadence while a test waits for a background fill to finish.
    const FILL_POLL_INTERVAL: Duration = Duration::from_millis(10);

    /// How long a read that must finish gets before the test calls it stuck.
    const READ_DEADLINE: Duration = Duration::from_secs(5);

    /// Bytes the unknown-size refusal test reads from a block no open touched.
    const REFUSAL_READ_BYTES: u64 = 4096;

    /// A store over recording storage holding one vector superfile, with a Warm open whose fill is
    /// parked at the only fill permit. Returns the held reader, the permit and the vector hole.
    async fn store_with_a_parked_fill() -> (
        TempDir,
        Arc<DiskCacheStore>,
        Arc<RecordingStorage>,
        SuperfileUri,
        Arc<SuperfileReader>,
        tokio::sync::OwnedSemaphorePermit,
        (u64, u64),
    ) {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let storage: Arc<dyn StorageProvider> = Arc::clone(&recording) as Arc<dyn StorageProvider>;
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                // The fill runs alongside the held reader, and parks at its permit until released.
                promotion_defer_timeout: Duration::ZERO,
                prefetch_concurrency: 1,
                ..Default::default()
            },
        )
        .expect("store");
        let uri = SuperfileUri::new_v4();
        storage
            .put_atomic(&uri.storage_path(), tiny_vector_superfile_bytes())
            .await
            .expect("put");
        let permit = hold_fill_slot(&store).await;
        let held = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("warm open starts the fill");
        let hole = vector_blob_range(&held).expect("vector blob");
        // With no defer window the fill's first wait lets it go at once, so the first time this
        // single-threaded test yields, the fill runs up to the permit and waits there.
        assert_eq!(
            store.fills_running.load(Ordering::Acquire),
            1,
            "the fill is running"
        );
        sleep(FILL_POLL_INTERVAL).await;
        (dir, store, recording, uri, held, permit, hole)
    }

    /// Wait for the parked fill to run and exit.
    async fn wait_for_the_fill_to_exit(store: &Arc<DiskCacheStore>) {
        timeout(PROMOTE_TIMEOUT, async {
            while store.fills_running.load(Ordering::Acquire) > 0 {
                sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("the fill exits");
    }

    fn overlaps(range: &std::ops::Range<u64>, (off, len): (u64, u64)) -> bool {
        range.start < off + len && off < range.end
    }

    /// A fill whose entry was evicted before it started downloading gives up without a single GET.
    #[tokio::test]
    async fn fill_whose_entry_was_evicted_does_not_download() {
        let (_dir, store, recording, uri, held, permit, _) = store_with_a_parked_fill().await;
        store.evict_at_least(1).await.expect("evict the entry");

        let from = recording.n_ranges();
        drop(permit);
        wait_for_the_fill_to_exit(&store).await;
        drop(held);

        assert!(
            recording.ranges_since(from).is_empty(),
            "no GET after the eviction"
        );
        assert!(!store.is_cached(&uri), "nothing is reinstated");
        assert!(
            !store.cache_path(&uri).exists(),
            "no file under the cache name"
        );
        store.assert_budget_consistent();
    }

    /// A fill that finds someone else's whole file in its slot gives up without a single GET, and
    /// leaves that file alone.
    #[tokio::test]
    async fn fill_over_a_whole_file_does_not_download() {
        let (_dir, store, recording, uri, held, permit, _) = store_with_a_parked_fill().await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("a Load puts the whole file in the slot");

        let from = recording.n_ranges();
        drop(permit);
        wait_for_the_fill_to_exit(&store).await;
        drop(held);

        assert!(
            recording.ranges_since(from).is_empty(),
            "no GET once the whole file is there"
        );
        assert!(store.is_mmap_promoted(&uri), "the Load's whole file stays");
        assert!(store.cache_path(&uri).exists());
        store.assert_budget_consistent();
    }

    /// A fill whose entry is evicted while it downloads skips the promotion. The download never
    /// reads the vector blob, so any GET inside it would come from the promotion.
    #[tokio::test]
    async fn fill_whose_entry_was_evicted_mid_download_skips_the_promotion() {
        let (_dir, store, recording, uri, held, permit, hole) = store_with_a_parked_fill().await;

        let from = recording.n_ranges();
        recording.pause_next_read();
        drop(permit);
        until_a_read_is_paused(&recording).await;
        store
            .evict_at_least(1)
            .await
            .expect("evict the entry mid-download");
        recording.resume();
        wait_for_the_fill_to_exit(&store).await;
        drop(held);

        let in_hole: Vec<_> = recording
            .ranges_since(from)
            .into_iter()
            .filter(|r| overlaps(r, hole))
            .collect();
        assert!(
            in_hole.is_empty(),
            "promotion reads reached storage: {in_hole:?}"
        );
        assert!(!store.is_cached(&uri), "nothing is reinstated");
        assert!(
            !store.cache_path(&uri).exists(),
            "no file under the cache name"
        );
        assert_eq!(leftover_tempfiles(&store), 0, "the download is deleted");
        store.assert_budget_consistent();
    }

    /// A store over a recording storage for the fill-lifecycle tests: one download slot, no defer
    /// window, downloads of several small chunks, and `budget` bytes of disk (`None`: the default).
    fn recording_store(
        budget: Option<u64>,
    ) -> (TempDir, Arc<DiskCacheStore>, Arc<RecordingStorage>) {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let defaults = DiskCacheConfig::default();
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&recording) as Arc<dyn StorageProvider>,
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                promotion_defer_timeout: Duration::ZERO,
                prefetch_concurrency: 1,
                cold_fetch_streams: 1,
                cold_fetch_chunk_bytes: DOWNLOAD_CHUNK_BYTES,
                disk_budget_bytes: budget.unwrap_or(defaults.disk_budget_bytes),
                ..defaults
            },
        )
        .expect("store");
        (dir, store, recording)
    }

    /// Put a small non-vector superfile in storage, larger than one download chunk.
    async fn put_tiny_superfile(recording: &RecordingStorage) -> (SuperfileUri, u64) {
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        assert!(
            bytes.len() as u64 > DOWNLOAD_CHUNK_BYTES,
            "a download of several chunks"
        );
        let size = bytes.len() as u64;
        recording
            .put_atomic(&uri.storage_path(), bytes)
            .await
            .expect("put");
        (uri, size)
    }

    /// Open `uri` Warm, which starts its download, and let go of the reader.
    async fn warm_open(store: &Arc<DiskCacheStore>, uri: &SuperfileUri) {
        drop(
            store
                .open_for_query(uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("a Warm open"),
        );
    }

    /// Push the idle-entry cap down to one entry and run it, with a second lazy entry so the
    /// table is over the cap. Returns that second entry's file.
    fn run_the_cap_over(store: &Arc<DiskCacheStore>) -> SuperfileUri {
        let other = SuperfileUri::new_v4();
        store.install_block_entry_for_test(other, dummy_block_source(store, other));
        store.set_lazy_entry_cap(1);
        store.release_unheld_lazy_entries_over_cap();
        other
    }

    /// Take `store`'s only download slot, so a download waits for it until the permit drops.
    async fn hold_fill_slot(store: &Arc<DiskCacheStore>) -> tokio::sync::OwnedSemaphorePermit {
        Arc::clone(&store.prefetch_semaphore)
            .acquire_owned()
            .await
            .expect("fill permit")
    }

    /// The landing rule a live download installs under (see [`BackgroundFill::may_replace`]).
    fn a_download_may_replace(current: &CachedEntry) -> bool {
        BackgroundFill::default().may_replace(Some(current))
    }

    /// The download registered for `uri`.
    fn fill_of(store: &DiskCacheStore, uri: &SuperfileUri) -> Arc<BackgroundFill> {
        Arc::clone(
            store
                .fills
                .get(uri)
                .expect("a download is registered")
                .value(),
        )
    }

    /// Land a finished download of the tiny superfile for `uri` on disk, as `fill`, reserving its
    /// size first as a download does. Returns the size.
    fn land_tiny_file(
        store: &DiskCacheStore,
        uri: &SuperfileUri,
        fill: &BackgroundFill,
        holed: bool,
    ) -> u64 {
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        let tmp = store.tmp_path(uri);
        fs::write(&tmp, &bytes).expect("write the download");
        assert!(store.try_reserve_without_evicting(size));
        let final_path = if holed {
            store.hole_path(uri)
        } else {
            store.cache_path(uri)
        };
        store.land_on_disk(uri, fill, &tmp, &final_path, holed, size, size);
        size
    }

    /// Ranges of the first download chunk fetched since `from`, one per download of a file.
    fn first_chunks_since(recording: &RecordingStorage, from: usize) -> usize {
        recording
            .ranges_since(from)
            .into_iter()
            .filter(|r| r.start == 0 && r.end == DOWNLOAD_CHUNK_BYTES)
            .count()
    }

    /// A file's download outlives its entry.
    /// - The idle-entry cap drops the entry while the download waits for its slot.
    /// - A query opens the file again: the reopen starts no second download.
    /// - The one download lands in the reopened entry.
    #[tokio::test]
    async fn a_download_survives_its_entry_being_dropped_and_the_file_reopened() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        let fill = fill_of(&store, &uri);

        run_the_cap_over(&store);
        assert!(!store.is_cached(&uri), "the cap drops the idle entry");
        store.assert_budget_consistent();

        let from = recording.n_ranges();
        let reopened = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("a query opens the file again");
        assert_eq!(store.fills.len(), 1, "the reopen starts no second download");
        assert!(Arc::ptr_eq(&fill_of(&store, &uri), &fill));

        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download finishes");
        drop(reopened);

        assert!(
            store.is_mmap_promoted(&uri),
            "installed into the reopened entry"
        );
        assert_eq!(first_chunks_since(&recording, from), 1, "one download");
        assert_eq!(leftover_tempfiles(&store), 0);
        store.assert_budget_consistent();
    }

    /// While a download runs, the idle cap keeps its file's entry, so the file stays reserved while
    /// it is written to disk.
    #[tokio::test]
    async fn a_downloading_files_entry_stays_charged_over_the_cap() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, size) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        // Hold the download at its first chunk.
        recording.pause_next_read();
        drop(permit);
        until_a_read_is_paused(&recording).await;

        run_the_cap_over(&store);
        assert!(store.is_cached(&uri), "the downloading file's entry stays");
        assert!(
            store.stats().current_bytes >= size,
            "the file stays charged while it downloads"
        );

        recording.resume();
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download finishes");
        assert!(store.is_mmap_promoted(&uri));
        store.assert_budget_consistent();
    }

    /// A table with more files than the idle-entry cap, read twice: each file is downloaded once,
    /// however often the cap drops and reopens its entry.
    #[tokio::test]
    async fn each_file_is_downloaded_once_over_the_cap() {
        const FILES: usize = 4;
        let (_dir, store, recording) = recording_store(None);
        store.set_lazy_entry_cap(1);
        let mut uris = Vec::new();
        for _ in 0..FILES {
            uris.push(put_tiny_superfile(&recording).await.0);
        }
        let from = recording.n_ranges();
        let permit = hold_fill_slot(&store).await;
        for _ in 0..2 {
            for uri in &uris {
                warm_open(&store, uri).await;
                store.release_unheld_lazy_entries_over_cap();
            }
        }
        assert_eq!(store.fills.len(), FILES, "one download per file");
        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the downloads finish");

        assert_eq!(
            first_chunks_since(&recording, from),
            FILES,
            "each file once"
        );
        store.assert_budget_consistent();

        // Every file is now local: in its entry, or on disk for the next read.
        let after = recording.n_ranges();
        for uri in &uris {
            warm_open(&store, uri).await;
        }
        assert_eq!(recording.n_ranges(), after, "a third pass makes no GET");
    }

    /// A failed download stays registered, so the next Warm read does not download the file again;
    /// only a read after `FAILED_FILL_RETRY_AFTER` may.
    #[tokio::test]
    async fn a_failed_download_is_not_started_again_by_the_next_read() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        let fill = fill_of(&store, &uri);
        // Every GET of the download now fails.
        recording.delete(&uri.storage_path()).await.expect("delete");
        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download ends");
        let failed_at_us = *fill.failed_at_us.get().expect("the download failed");

        let from = recording.n_ranges();
        let entry = Arc::clone(&store.cached.get(&uri).expect("still lazy"));
        store.maybe_spawn_background_fill(&uri, &uri.storage_path(), &entry, None);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("nothing runs");
        assert_eq!(recording.n_ranges(), from, "no second download");
        assert!(Arc::ptr_eq(&fill_of(&store, &uri), &fill));

        let retry_after_us = FAILED_FILL_RETRY_AFTER.as_micros() as u64;
        assert!(!fill.may_retry(failed_at_us + retry_after_us - 1));
        assert!(fill.may_retry(failed_at_us + retry_after_us));
    }

    /// A download whose cache directory is removed under it (a dropped store's, say) ends like a
    /// cancel, not a failure.
    ///
    /// Setup: the download waits for its slot, and the cache directory is removed meanwhile, so it
    /// cannot create its file.
    ///
    /// This test pins that the download is not marked failed and does not stay registered.
    #[tokio::test]
    async fn a_download_whose_cache_directory_is_removed_is_not_a_failure() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        let fill = fill_of(&store, &uri);
        fs::remove_dir_all(&store.config.cache_root).expect("remove the cache directory");
        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download ends");

        assert!(fill.failed_at_us.get().is_none(), "not failed");
        assert!(store.fills.get(&uri).is_none(), "not registered");
    }

    /// A download whose task never runs (its runtime shut down first) still unregisters, or the
    /// file would never download again.
    #[test]
    fn a_fill_dropped_before_it_runs_leaves_no_registration() {
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let entry = Arc::clone(&store.cached.get(&uri).expect("entry"));
        // A future that is ready at once never lets the spawned fill run.
        runtime.block_on(async {
            store.maybe_spawn_background_fill(&uri, &uri.storage_path(), &entry, None);
        });
        assert_eq!(store.fills.len(), 1, "registered, not yet run");
        drop(runtime);
        assert!(store.fills.is_empty(), "the dropped task unregistered it");
        assert_eq!(store.fills_running.load(Ordering::Acquire), 0);
    }

    /// A download for a file nobody has open takes only free space. With none, it ends with no GET
    /// and evicts nothing.
    #[tokio::test]
    async fn a_download_for_a_file_nobody_has_open_does_not_evict() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        let other = run_the_cap_over(&store);
        assert!(!store.is_cached(&uri), "the cap drops the idle entry");
        // Leave no free space.
        let free = store.disk_budget_bytes() - store.stats().current_bytes;
        assert!(store.try_reserve_without_evicting(free));

        let from = recording.n_ranges();
        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download ends");
        assert_eq!(recording.n_ranges(), from, "no GET");
        assert!(store.is_cached(&other), "nothing evicted");
        assert!(
            store.fills.is_empty(),
            "ended, not failed: a later read may start it"
        );
        store.release_block_bytes(free);
        store.assert_budget_consistent();
    }

    /// A download with no entry to land in is opened and checked before it goes under the cache
    /// name, as an install would check it: the next read opens that file without checking.
    #[tokio::test]
    async fn a_corrupt_download_does_not_land() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        run_the_cap_over(&store);
        assert!(!store.is_cached(&uri), "the cap drops the idle entry");
        // The object now arrives with a broken footer.
        let mut corrupt = tiny_superfile_bytes().to_vec();
        let end = corrupt.len();
        corrupt[end - CORRUPT_FOOTER_BYTES..end].fill(0);
        recording
            .put_overwrite(&uri.storage_path(), Bytes::from(corrupt))
            .await
            .expect("overwrite");

        drop(permit);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download ends");
        assert!(!store.cache_path(&uri).exists(), "nothing lands");
        let fill = fill_of(&store, &uri);
        assert!(fill.failed_at_us.get().is_some(), "it failed");
        assert_eq!(leftover_tempfiles(&store), 0);
        store.assert_budget_consistent();
    }

    /// A download landing on disk replaces an older copy under the other name, so no file is left
    /// that nothing counts.
    #[tokio::test]
    async fn a_landed_download_replaces_a_copy_under_the_other_name() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let size = tiny_superfile_bytes().len() as u64;
        for holed in [true, false] {
            land_tiny_file(&store, &uri, &BackgroundFill::default(), holed);
        }
        assert!(store.cache_path(&uri).exists());
        assert!(
            !store.hole_path(&uri).exists(),
            "the older holed copy is gone"
        );
        assert_eq!(store.stats().current_bytes, size);
        store.assert_budget_consistent();
    }

    /// A cancelled download never lands: the cancel and the landing take the same lock.
    #[tokio::test]
    async fn a_cancelled_download_does_not_land() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        land_tiny_file(&store, &uri, &cancelled_fill(), false);
        assert!(!store.cache_path(&uri).exists());
        assert_eq!(leftover_tempfiles(&store), 0);
        assert_eq!(
            store.stats().current_bytes,
            0,
            "its reservation is given back"
        );
        store.assert_budget_consistent();
    }

    /// A lazy entry opened while an earlier download was landing the file starts no second
    /// download: the copy on disk is what the next open maps.
    #[tokio::test]
    async fn a_file_already_on_disk_is_not_downloaded_again() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        land_tiny_file(&store, &uri, &BackgroundFill::default(), false);
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let entry = Arc::clone(&store.cached.get(&uri).expect("the lazy entry"));

        store.maybe_spawn_background_fill(&uri, &uri.storage_path(), &entry, None);
        assert!(store.fills.is_empty(), "no second download");
    }

    /// Eviction for a download never takes the file the download is for: not its entry, not a copy
    /// on disk, not its block file.
    #[tokio::test]
    async fn eviction_keeps_the_file_it_makes_room_for() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        land_tiny_file(&store, &uri, &BackgroundFill::default(), false);

        assert!(store.evict_except(1, Some(&uri)).await.is_err());
        assert!(store.cache_path(&uri).exists(), "kept");
        store.evict_at_least(1).await.expect("evict without keep");
        assert!(!store.cache_path(&uri).exists(), "evicted");
        store.assert_budget_consistent();
    }

    /// A download never evicts its own file to make room, which would cancel it.
    /// - The file's entry holds the only bytes eviction could free.
    /// - The budget is one byte short of what the download needs.
    ///
    /// So the download fails, and the entry stays.
    #[tokio::test]
    async fn a_download_never_evicts_its_own_file() {
        const ENTRY_BYTES: u64 = 1;
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = RESERVED_BYTES - 1;
            cfg.prefetch_concurrency = 1;
        });
        let uri = SuperfileUri::new_v4();
        let reader = SuperfileReader::open(tiny_superfile_bytes()).expect("tiny superfile opens");
        assert!(store.try_reserve_without_evicting(ENTRY_BYTES));
        store.cached.insert(
            uri,
            Arc::new(CachedEntry {
                reader: Arc::new(reader),
                residency: Residency::Paged {
                    block_source: dummy_block_source(&store, uri),
                },
                size_bytes: Arc::new(AtomicU64::new(ENTRY_BYTES)),
                accounting: EntryAccounting::Eager,
                last_access_us: AtomicU64::new(0),
            }),
        );
        let exit = FillExit::new(&store, uri, Arc::default());
        let result = lazy_background_fill(
            &exit,
            uri.storage_path(),
            RESERVED_BYTES,
            Arc::clone(&store.storage),
            None,
            PromotionDefer::start(Duration::ZERO),
        )
        .await;
        drop(exit);
        assert!(
            matches!(result, Err(DiskCacheError::BudgetExceeded)),
            "no room: it fails"
        );
        assert!(store.is_cached(&uri), "the file's entry stays");
        store.assert_budget_consistent();
    }

    /// An erased file's waiting download is cancelled: GC deletes the object next, and the local
    /// copy must never outlive it.
    #[tokio::test]
    async fn an_erased_files_download_is_cancelled() {
        let (_dir, store, recording, uri, held, permit, _hole) = store_with_a_parked_fill().await;
        drop(held);
        assert!(store.erase_superfile_local_copy(&uri));
        let from = recording.n_ranges();
        drop(permit);
        wait_for_the_fill_to_exit(&store).await;
        assert_eq!(recording.n_ranges(), from, "no GET");
        assert!(!store.is_cached(&uri));
        assert_eq!(leftover_tempfiles(&store), 0);
        store.assert_budget_consistent();
    }

    /// A cancelled download's exit unregisters only itself, not the newer download a later Warm
    /// read registered for the same file.
    #[tokio::test]
    async fn a_cancelled_downloads_exit_keeps_the_newer_one() {
        let (_dir, store, recording) = recording_store(None);
        let (uri, _) = put_tiny_superfile(&recording).await;
        let permit = hold_fill_slot(&store).await;
        warm_open(&store, &uri).await;
        recording.pause_next_read();
        drop(permit);
        until_a_read_is_paused(&recording).await;
        let first = fill_of(&store, &uri);

        store.evict_at_least(1).await.expect("evict the file");
        assert!(first.is_cancelled());
        warm_open(&store, &uri).await;
        let second = fill_of(&store, &uri);
        assert!(!Arc::ptr_eq(&first, &second));

        // The first download wakes, sees its cancel and exits; the second, held at its first GET,
        // keeps its registration.
        recording.pause_next_read();
        recording.resume();
        timeout(PROMOTE_TIMEOUT, async {
            while store.fills_running.load(Ordering::Acquire) > 1 {
                sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("the first download exits");
        assert!(
            Arc::ptr_eq(
                store
                    .fills
                    .get(&uri)
                    .expect("the newer one is registered")
                    .value(),
                &second
            ),
            "the newer one stays"
        );
        recording.resume();
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the second download finishes");
        assert!(store.is_mmap_promoted(&uri), "and installs");
        store.assert_budget_consistent();
    }

    /// How much smaller than the file the surplus test's promoted charge is: any amount works, it
    /// only has to leave a surplus in the fill's full-size reservation.
    const SURPLUS_TEST_BYTES: u64 = 100;

    /// How many bytes at the end of a file the corrupt-download test zeroes: enough to break the
    /// footer, which every open reads.
    const CORRUPT_FOOTER_BYTES: usize = 16;

    /// However a fill ends, its exit unregisters it, gives back what it still holds reserved and
    /// deletes its partial download, and leaves the file's entry alone: the entry serves reads
    /// without it.
    #[tokio::test]
    async fn a_fill_that_ends_gives_back_its_reservation_and_its_download() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let before = store.stats().current_bytes;
        let fill: Arc<BackgroundFill> = Arc::default();
        store.fills.insert(uri, Arc::clone(&fill));
        assert!(store.try_reserve_without_evicting(RESERVED_BYTES));
        fill.reserved.store(RESERVED_BYTES, Ordering::Release);
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, b"partial-download-bytes").expect("seed tmp scratch file");

        drop(FillExit {
            store: Arc::downgrade(&store),
            uri,
            fill,
            tmp: tmp.clone(),
        });

        assert!(store.fills.is_empty(), "unregistered");
        assert_eq!(
            store.stats().current_bytes,
            before,
            "its reservation came back"
        );
        assert!(!tmp.exists(), "its partial download is deleted");
        assert!(store.is_cached(&uri), "the file's entry stays");
        store.assert_budget_consistent();
    }

    // ----- warm insert path (insert_warm + cold-free path) -----

    #[tokio::test]
    async fn reader_hybrid_cold_then_stays_lazy_without_full_promotion() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // reader() dispatches by config default.
        let r = store.reader(&uri).await.expect("cold hybrid");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert_eq!(store.stats().n_entries, 1);

        // Warm path remains lazy/block-backed by design (no full-file barrier).
        assert!(!store.is_mmap_promoted(&uri));

        // Warm hit reuses cached reader; no extra cold fetch.
        let _r2 = store.reader(&uri).await.expect("warm");
        assert_eq!(store.stats().n_cold_fetches, 1);
    }

    #[tokio::test]
    async fn reader_hybrid_empty_object_zero_chunks() {
        // size == 0 takes the n_chunks == 0 branch in cold_fetch_hybrid;
        // the empty buffer fails to parse as a superfile, surfacing an
        // open error rather than a cache entry.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, Bytes::new()).await;
        let err = store.reader(&uri).await.expect_err("empty not a superfile");
        let _ = format!("{err}");
    }

    /// A hybrid fetch that fails after reserving must give the reservation back, or every failed
    /// open shrinks the budget for good.
    #[tokio::test]
    async fn hybrid_fetch_failure_leaves_the_ledger_balanced() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::HybridWithPrefetch;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, Bytes::from_static(b"not a superfile")).await;

        store
            .reader(&uri)
            .await
            .expect_err("garbage does not open as a superfile");

        assert_eq!(
            store.stats().current_bytes,
            0,
            "the failed fetch's reservation came back"
        );
        assert_eq!(store.stats().n_entries, 0);
        store.assert_budget_consistent();
    }

    /// A finished fill's download: `bytes` in a fresh fill tempfile, mmapped and opened the way the
    /// fill opens it, charged `size`.
    fn finished_fill(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
        bytes: &Bytes,
        size: u64,
    ) -> (PathBuf, Arc<CachedEntry>) {
        let tmp = store.tmp_path(uri);
        std::fs::write(&tmp, bytes.as_ref()).expect("write the fill's tempfile");
        let (mmap, mapped) = mmap_readonly_with_handle(&tmp).expect("mmap");
        let reader = Arc::new(
            SuperfileReader::open_with(mapped, OpenOptions { verify_crc: false }).expect("open"),
        );
        (tmp, store.build_mmap_entry(reader, mmap, size, None))
    }

    /// A fill whose file was evicted.
    fn cancelled_fill() -> BackgroundFill {
        let fill = BackgroundFill::default();
        fill.cancelled.store(true, Ordering::Release);
        fill
    }

    /// Temp files left in the cache root: fetch and fill tempfiles alike.
    fn leftover_tempfiles(store: &Arc<DiskCacheStore>) -> usize {
        std::fs::read_dir(&store.config.cache_root)
            .expect("cache root")
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| SuperfileUri::from_cache_tmp_filename(n).is_some())
            })
            .count()
    }

    /// Evicted mid-fill: the fill is cancelled, so nothing is installed, the tempfile goes, nothing
    /// lands under the cache name, and a fill with no reservation of its own touches no budget.
    #[tokio::test]
    async fn fill_finishing_on_an_evicted_slot_deletes_its_tempfile() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        let (tmp, entry) = finished_fill(&store, &uri, &bytes, size);
        let final_path = store.cache_path(&uri);

        let before = store.current_bytes.load(Ordering::Acquire);
        let fill = cancelled_fill();
        let installed = store.install_promoted_entry(
            uri,
            entry,
            &tmp,
            &final_path,
            |current| fill.may_replace(Some(current)),
            0,
        );

        assert!(installed.is_none(), "an evicted file is not reinstated");
        assert!(!tmp.exists(), "the tempfile is deleted");
        assert!(!final_path.exists(), "nothing lands under the cache name");
        assert_eq!(store.current_bytes.load(Ordering::Acquire), before);
        assert_eq!(store.stats().n_entries, 0);
        store.assert_budget_consistent();
    }

    /// A fill that reserved for itself (the Stream-then-Warm shape) and finds its file evicted must
    /// give that reservation back, or every evict-during-fill shrinks the budget for good.
    #[tokio::test]
    async fn fill_finishing_on_an_evicted_slot_releases_its_own_reservation() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        let (tmp, entry) = finished_fill(&store, &uri, &bytes, size);

        let before = store.current_bytes.load(Ordering::Acquire);
        store
            .reserve_manual(size)
            .await
            .expect("the fill's own reservation");
        let fill = cancelled_fill();
        let installed = store.install_promoted_entry(
            uri,
            entry,
            &tmp,
            &store.cache_path(&uri),
            |current| fill.may_replace(Some(current)),
            size,
        );

        assert!(installed.is_none(), "an evicted file is not reinstated");
        assert_eq!(
            store.current_bytes.load(Ordering::Acquire),
            before,
            "the fill's reservation came back"
        );
        store.assert_budget_consistent();
    }

    /// Someone else's whole file beats a finishing fill: it keeps its entry, its charge and its
    /// file on disk, byte for byte (the fill never renames over it), and the fill gives back only
    /// what it reserved for itself.
    #[tokio::test]
    async fn fill_finishing_over_a_whole_file_keeps_it_and_its_file() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        seed_cache_file(&store, &uri, &bytes);
        let stranger = store
            .fetch_from_disk_cache(&uri, &uri.storage_path(), None, ReadIntent::Warm, None)
            .await
            .expect("disk probe")
            .expect("whole file admitted");

        store
            .reserve_manual(size)
            .await
            .expect("the fill's own reservation");

        let charged_before = store.stats().current_bytes;

        // Different bytes in the fill's download, so an overwrite would show.
        let other = tiny_vector_superfile_bytes();
        let (tmp, promoted) = finished_fill(&store, &uri, &other, other.len() as u64);
        let final_path = store.cache_path(&uri);
        let installed = store.install_promoted_entry(
            uri,
            promoted,
            &tmp,
            &final_path,
            a_download_may_replace,
            size,
        );

        assert!(installed.is_none(), "a redundant download is not installed");
        assert!(Arc::ptr_eq(
            &store.cached.get(&uri).expect("still cached").reader,
            &stranger.reader
        ));
        assert_eq!(
            std::fs::read(&final_path).expect("the stranger's file"),
            bytes.as_ref(),
            "the stranger's file on disk is untouched"
        );
        assert!(!tmp.exists(), "the fill's tempfile is deleted");
        assert_eq!(
            store.stats().current_bytes,
            charged_before - size,
            "only the fill's own reservation is released"
        );
        store.assert_budget_consistent();
    }

    /// A finished local copy beats range reads: a fill installs over the file's lazy entry, takes
    /// over that entry's full charge, and only then moves its file under the cache name.
    #[tokio::test]
    async fn fill_installs_over_the_files_lazy_entry() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes.clone()).await;
        store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("a Warm open: lazy, charged in full");
        assert_eq!(store.stats().current_bytes, size);

        let (tmp, promoted) = finished_fill(&store, &uri, &bytes, size);
        let final_path = store.cache_path(&uri);
        let installed = store.install_promoted_entry(
            uri,
            promoted,
            &tmp,
            &final_path,
            a_download_may_replace,
            0,
        );

        assert!(
            installed.is_some(),
            "the finished copy replaces the lazy one"
        );
        assert!(store.is_mmap_promoted(&uri));
        assert!(
            final_path.exists() && !tmp.exists(),
            "renamed into the cache"
        );
        assert_eq!(
            store.stats().current_bytes,
            size,
            "the lazy entry's charge now backs the mmap"
        );
        store.assert_budget_consistent();
    }

    /// A lazy entry charged per block leaves no full charge to inherit. With room in the budget the
    /// fill tops up and installs; without room the download is thrown away, leaving no file.
    #[tokio::test]
    async fn fill_over_a_per_block_lazy_entry_installs_only_with_room() {
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        for (budget, expect_installed) in [(u64::MAX, true), (size - 1, false)] {
            let (_dir, store) = test_store_with(|cfg| cfg.disk_budget_bytes = budget);
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, bytes.clone()).await;
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
                .await
                .expect("a Stream open: lazy, charged per block");

            let (tmp, promoted) = finished_fill(&store, &uri, &bytes, size);
            let final_path = store.cache_path(&uri);
            let installed = store.install_promoted_entry(
                uri,
                promoted,
                &tmp,
                &final_path,
                a_download_may_replace,
                0,
            );

            assert_eq!(installed.is_some(), expect_installed, "budget {budget}");
            assert_eq!(store.is_mmap_promoted(&uri), expect_installed);
            assert_eq!(final_path.exists(), expect_installed);
            assert!(!tmp.exists(), "the tempfile never outlives the install");
            store.assert_budget_consistent();
        }
    }

    /// A finished download is never installed into an empty slot; the fill leaves it on disk
    /// instead.
    #[tokio::test]
    async fn fill_with_no_entry_is_not_installed() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let (tmp, promoted) = finished_fill(&store, &uri, &bytes, bytes.len() as u64);
        let installed = store.install_promoted_entry(
            uri,
            promoted,
            &tmp,
            &store.cache_path(&uri),
            a_download_may_replace,
            0,
        );
        assert!(installed.is_none());
        assert!(!store.is_cached(&uri));
        assert!(!tmp.exists(), "the tempfile never outlives the install");
        store.assert_budget_consistent();
    }

    /// A download left on disk is charged its size once, from its reservation, also over an older
    /// record for the same file.
    #[tokio::test]
    async fn a_landed_download_is_charged_once() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        for reserved in [size, size + SURPLUS_TEST_BYTES] {
            let tmp = store.tmp_path(&uri);
            fs::write(&tmp, &bytes).expect("write the download");
            assert!(store.try_reserve_without_evicting(reserved));
            store.land_on_disk(
                &uri,
                &BackgroundFill::default(),
                &tmp,
                &store.cache_path(&uri),
                false,
                size,
                reserved,
            );
            assert!(!tmp.exists());
            assert!(store.cache_path(&uri).exists());
            assert_eq!(store.stats().current_bytes, size, "reserved {reserved}");
            store.assert_budget_consistent();
        }
    }

    /// The whole race, end to end: a Warm fill runs while a query holds the reader, the file is
    /// evicted, which cancels the fill, and a query reopens the file lazily. A Warm reopen starts
    /// the file's download again and that one wins the slot; a Stream reopen leaves it lazy. Either
    /// way the setup is complete (no stray tempfiles), the ledger honest, and the next read must not
    /// fetch or throw anything away. The fixture has vectors, so the copy lands under the hole name.
    #[tokio::test]
    async fn an_evicted_file_downloads_again_only_on_a_warm_reopen() {
        for reopen in [ReadIntent::Stream, ReadIntent::Warm] {
            let (_dir, store) = test_store_with(|cfg| {
                cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
                // The fill runs alongside the held query, as it does once the window closes.
                cfg.promotion_defer_timeout = Duration::ZERO;
                cfg.prefetch_concurrency = 1;
            });
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
            // Hold the only fill permit, so the fill parks right before its download until the
            // eviction and the reopen have happened.
            let permit = hold_fill_slot(&store).await;

            let held = store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open starts the fill");
            store.evict_at_least(1).await.expect("evict mid-fill");
            let reopened = store
                .open_for_query(&uri, &uri.storage_path(), None, None, reopen)
                .await
                .expect("a query reopens the file");
            drop(permit);
            let warm = reopen == ReadIntent::Warm;

            if warm {
                assert!(
                    poll_until_mmap_promoted(&store, &uri, PROMOTE_TIMEOUT).await,
                    "the new fill wins the slot"
                );
            }
            drop(held);
            drop(reopened);
            store
                .wait_until_fills_settled(PROMOTE_TIMEOUT)
                .await
                .expect("fills settle");

            assert_eq!(
                store.hole_path(&uri).exists(),
                warm,
                "{reopen:?}: file in the cache only after a Warm reopen"
            );
            assert_eq!(
                leftover_tempfiles(&store),
                0,
                "{reopen:?}: no stray tempfiles"
            );
            // The reopened query's block write may still be queued: its run is charged
            // before it lands.
            store.block_writes_settled().await;
            store.assert_budget_consistent();

            let fetches = store.stats().n_cold_fetches;
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
                .await
                .expect("next read");
            assert_eq!(
                store.stats().n_cold_fetches,
                fetches,
                "{reopen:?}: the next read is served locally"
            );
            assert_eq!(
                store.hole_path(&uri).exists(),
                warm,
                "{reopen:?}: nothing local was thrown away"
            );
            assert!(
                store.blocks_path(&uri).exists(),
                "{reopen:?}: nothing local was thrown away"
            );
        }
    }

    /// A promotion that keeps the vector blob on a block source charging its own blocks is charged
    /// the file minus its hole: the hole is not on disk, and its blocks are already charged.
    #[tokio::test]
    async fn vector_hole_promotion_charges_the_file_minus_its_hole() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_vector_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        let reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("stream open");
        let (vec_off, vec_len) = vector_blob_range(&reader).expect("vector blob");
        let source = Arc::clone(
            store
                .cached
                .get(&uri)
                .expect("lazy entry")
                .block_source()
                .expect("paged"),
        );
        source
            .range(vec_off, vec_len)
            .await
            .expect("a query touches the vector blocks");
        store.block_writes_settled().await;
        let blocks = source.filled_bytes_handle().load(Ordering::Acquire);
        drop(source);
        drop(reader);

        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open starts the fill"),
        );
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");

        assert_eq!(
            store.stats().current_bytes,
            size - vec_len + blocks,
            "mmap charged without the hole, blocks charged once"
        );
        store.assert_budget_consistent();
    }

    /// A fill that reserved more than its entry is charged (the Stream-then-Warm shape) gives the
    /// surplus back when it installs over a per-block lazy entry.
    #[tokio::test]
    async fn fill_returns_the_surplus_of_its_own_reservation() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes.clone()).await;
        store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("a Stream open: lazy, charged per block");
        store
            .reserve_manual(size)
            .await
            .expect("the fill's own reservation");

        let charge = size - SURPLUS_TEST_BYTES;
        let (tmp, promoted) = finished_fill(&store, &uri, &bytes, charge);
        let installed = store.install_promoted_entry(
            uri,
            promoted,
            &tmp,
            &store.cache_path(&uri),
            a_download_may_replace,
            size,
        );

        assert!(installed.is_some());
        assert_eq!(
            store.stats().current_bytes,
            charge,
            "the surplus came back, the lazy entry's blocks went with it"
        );
        store.assert_budget_consistent();
    }

    /// An eviction that lands between the install and the rename leaves the slot empty; the file
    /// the rename then puts under the cache name belongs to nobody and must go.
    #[tokio::test]
    async fn file_renamed_after_an_eviction_is_removed() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, tiny_superfile_bytes().as_ref()).expect("the fill's tempfile");
        let final_path = store.cache_path(&uri);

        // The slot is already empty, as if eviction ran right after the install.
        store.move_into_cache(uri, &tmp, &final_path);

        assert!(
            !tmp.exists() && !final_path.exists(),
            "no orphan left behind"
        );
    }

    /// A rename that fails leaves the installed entry serving from its mmap and deletes the
    /// tempfile, which would otherwise sit uncharged until a restart reclaims it.
    #[tokio::test]
    async fn a_failed_rename_keeps_the_entry_and_drops_the_tempfile() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes.clone()).await;
        store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("a Warm open: lazy, charged in full");
        // A directory under the cache name makes the rename fail.
        let final_path = store.cache_path(&uri);
        std::fs::create_dir(&final_path).expect("block the cache name");

        let (tmp, promoted) = finished_fill(&store, &uri, &bytes, size);
        let installed = store.install_promoted_entry(
            uri,
            promoted,
            &tmp,
            &final_path,
            a_download_may_replace,
            0,
        );

        let installed = installed.expect("the install itself succeeds");
        assert_eq!(installed.reader.n_docs(), 1, "served from the mmap");
        assert!(!tmp.exists(), "the tempfile is deleted");
        store.assert_budget_consistent();
    }

    /// The hybrid finalizer takes the same path: install first, then rename, so the finished file
    /// ends up under the cache name with no tempfile left over.
    #[tokio::test]
    async fn hybrid_finalizer_lands_its_file_under_the_cache_name() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::HybridWithPrefetch;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        drop(store.reader(&uri).await.expect("hybrid cold fetch"));
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("the finalizer promotes");

        assert!(store.cache_path(&uri).exists(), "file under the cache name");
        assert_eq!(leftover_tempfiles(&store), 0, "no stray tempfiles");
        store.assert_budget_consistent();
    }

    /// A fill that will keep the vector blob on its entry's block source reserves only what the
    /// file will hold, not the file plus a hole it never writes.
    #[tokio::test]
    async fn vector_hole_fill_reserves_the_file_minus_its_hole() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.prefetch_concurrency = 1;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_vector_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        let reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("stream open");
        let (_, vec_len) = vector_blob_range(&reader).expect("vector blob");
        drop(reader);

        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open starts the fill"),
        );
        let fill = fill_of(&store, &uri);
        let most_reserved = timeout(PROMOTE_TIMEOUT, async {
            let mut most = 0;
            while !store.fills.is_empty() {
                most = most.max(fill.reserved.load(Ordering::Acquire));
                yield_now().await;
            }
            most
        })
        .await
        .expect("the fill finishes");

        assert!(store.is_mmap_promoted(&uri));
        assert_eq!(most_reserved, size - vec_len, "the hole is never reserved");
        store.assert_budget_consistent();
    }

    #[test]
    fn every_download_gets_its_own_tempfile() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let (a, b) = (store.tmp_path(&uri), store.tmp_path(&uri));
        assert_ne!(
            a, b,
            "two downloads of one superfile never share a tempfile"
        );
        for path in [&a, &b] {
            let name = path.file_name().and_then(|n| n.to_str()).expect("name");
            assert_eq!(
                SuperfileUri::from_cache_tmp_filename(name),
                Some(uri),
                "the scan still recognises it"
            );
        }
    }

    /// Two stores over one directory, as two connections to one table build, number their
    /// downloads apart: the first download of one superfile in each never shares a tempfile.
    #[test]
    fn two_stores_over_one_directory_never_share_a_tempfile() {
        let (_dir, store) = test_store();
        let other = reopen_store(&store, |_| {});
        let uri = SuperfileUri::new_v4();
        assert_ne!(store.tmp_path(&uri), other.tmp_path(&uri));
    }

    #[tokio::test]
    async fn cold_fetch_uses_caller_storage_not_cache_embedded_storage() {
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
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("cache");

        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        hidden_storage
            .put_atomic(&uri.storage_path(), bytes.clone())
            .await
            .expect("put at hidden prefix");

        let reader = cache
            .open_for_query(
                &uri,
                &uri.storage_path(),
                None,
                Some(&hidden_storage),
                ReadIntent::Warm,
            )
            .await
            .expect("cold fetch via caller storage");
        assert_eq!(reader.n_docs(), 1);
        assert_eq!(cache.stats().n_cold_fetches, 1);
    }

    /// Bench default (`LazyForegroundWithBackgroundFill`): the lazy inner
    /// `StorageRangeSource` must honor the caller's prefixed storage, not
    /// the cache's embedded user-root provider.
    #[tokio::test]
    async fn lazy_cold_fetch_uses_caller_storage_not_cache_embedded_storage() {
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
        let bytes = tiny_superfile_bytes();
        hidden_storage
            .put_atomic(&uri.storage_path(), bytes.clone())
            .await
            .expect("put at hidden prefix");

        let reader = cache
            .open_for_query(
                &uri,
                &uri.storage_path(),
                None,
                Some(&hidden_storage),
                ReadIntent::Warm,
            )
            .await
            .expect("lazy cold fetch via caller storage");
        assert_eq!(reader.n_docs(), 1);
        assert_eq!(cache.stats().n_cold_fetches, 1);
    }

    #[tokio::test]
    async fn reader_lazy_unknown_size_promotes_after_release() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // open_for_query(None) → unknown-size lazy cold fetch.
        let r = store.reader(&uri).await.expect("lazy cold");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);

        // Releasing the foreground reader permits the full-file background
        // fill to replace the lazy entry with an mmap-backed reader.
        drop(r);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("background promotion");
        let r2 = store.reader(&uri).await.expect("warm mmap");
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert!(store.is_mmap_promoted(&uri));
        assert!(r2.parquet_bytes().is_some());
    }

    /// A held foreground reader must not defer the background fill forever.
    /// Once the promotion window closes the fill runs alongside the query,
    /// so the superfile stops serving reads from a heap-resident lazy reader
    /// for the life of the process.
    ///
    /// Polls promotion directly: `wait_until_mmap_promoted` registers a
    /// promotion waiter, which is itself an override of the deferral, so a
    /// test using it could not tell whether the window did the work.
    #[tokio::test]
    async fn lazy_fill_promotes_while_a_foreground_reader_is_still_held() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.promotion_defer_timeout = Duration::ZERO;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // Never dropped for the length of the assertion: this stands in for a
        // superfile under continuous query load.
        let held = store.reader(&uri).await.expect("lazy cold");
        assert_eq!(held.n_docs(), 1);

        assert!(
            poll_until_mmap_promoted(&store, &uri, PROMOTE_TIMEOUT).await,
            "closed promotion window must let the fill run under load"
        );
        assert_eq!(
            store.stats().n_cold_fetches,
            1,
            "promotion re-reads the object once, not per query"
        );
        drop(held);
    }

    /// With the window left open the fill still yields: a held reader keeps
    /// the superfile lazy, and promotion happens once the reader releases.
    #[tokio::test]
    async fn unbounded_promotion_defer_keeps_yielding_to_a_held_reader() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let held = store.reader(&uri).await.expect("lazy cold");
        assert!(
            !poll_until_mmap_promoted(&store, &uri, DEFER_OBSERVATION).await,
            "an open window must keep deferring to the held reader"
        );

        drop(held);
        assert!(
            poll_until_mmap_promoted(&store, &uri, PROMOTE_TIMEOUT).await,
            "releasing the reader must let the deferred fill finish"
        );
    }

    /// A store with a lazy entry for a file, and that entry's reader as a query holds it.
    /// Installing the entry directly starts no background fill of the store's own, so a test sees
    /// only the fills it parks.
    fn store_with_held_file() -> (
        TempDir,
        Arc<DiskCacheStore>,
        SuperfileUri,
        Arc<SuperfileReader>,
    ) {
        let (dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let query = Arc::clone(&store.cached.get(&uri).expect("cached").reader);
        (dir, store, uri, query)
    }

    /// Wait until `n` fills are parked on `store`.
    async fn until_parked(store: &DiskCacheStore, n: usize) {
        timeout(PROMOTE_TIMEOUT, async {
            while store.fill_waiters.parked() != n {
                sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("fills parked");
    }

    /// A fill parked behind a held reader is not polled again until the reader is free, however
    /// many fills wait: they cost the runtime no wake-ups of their own. Each one then goes after
    /// the reader is released, and the list empties.
    #[tokio::test]
    async fn waiting_fills_are_not_polled_while_the_reader_is_held() {
        let (_dir, store, uri, query) = store_with_held_file();
        let mut waits = Vec::new();
        for _ in 0..PARKED_FILLS {
            let polls = Arc::new(AtomicUsize::new(0));
            let counted = Arc::clone(&polls);
            let store = Arc::downgrade(&store);
            let mut wait = Box::pin(async move {
                FillWaiters::park(
                    &store,
                    &uri,
                    &Arc::default(),
                    PromotionDefer::Forever,
                    FillWait::Quiescence,
                )
                .await
            });
            let task = spawn(poll_fn(move |cx| {
                counted.fetch_add(1, Ordering::Relaxed);
                wait.as_mut().poll(cx)
            }));
            waits.push((polls, task));
        }
        until_parked(&store, PARKED_FILLS).await;

        sleep(HELD_FOR).await;
        for (polls, _) in &waits {
            assert_eq!(
                polls.load(Ordering::Relaxed),
                1,
                "a parked fill is polled once, then not again while the reader is held"
            );
        }

        drop(query);
        for (_, task) in waits {
            timeout(PROMOTE_TIMEOUT, task)
                .await
                .expect("a released reader wakes the fill")
                .expect("join");
        }
        assert_eq!(
            store.fill_waiters.parked(),
            0,
            "every woken fill left the list"
        );
    }

    /// A fill can park from a short-lived runtime, and that one starts the store's ticker. The
    /// ticker must not die with that runtime, or every fill parked after it would wait for good.
    #[tokio::test]
    async fn fills_parked_after_another_runtime_shuts_down_still_wake() {
        let (_dir, store, uri, query) = store_with_held_file();

        let first_store = Arc::downgrade(&store);
        thread::spawn(move || {
            let runtime = Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("runtime");
            // The timer is built inside `block_on`: it needs the runtime's clock.
            let parked = runtime.block_on(async {
                timeout(
                    SHORT_RUNTIME_LIFE,
                    FillWaiters::park(
                        &first_store,
                        &uri,
                        &Arc::default(),
                        PromotionDefer::Forever,
                        FillWait::Quiescence,
                    ),
                )
                .await
            });
            assert!(
                parked.is_err(),
                "the reader is held, so the fill stays parked"
            );
        })
        .join()
        .expect("short-lived runtime");

        let weak_store = Arc::downgrade(&store);
        let wait = spawn(async move {
            FillWaiters::park(
                &weak_store,
                &uri,
                &Arc::default(),
                PromotionDefer::Forever,
                FillWait::Quiescence,
            )
            .await
        });
        until_parked(&store, 1).await;
        drop(query);
        timeout(PROMOTE_TIMEOUT, wait)
            .await
            .expect("the ticker outlives the runtime that started it")
            .expect("join");
    }

    /// Dropping the store wakes the fills parked on it, rather than leaving them parked forever.
    #[tokio::test]
    async fn dropping_the_store_wakes_its_parked_fills() {
        let (_dir, store, uri, _query) = store_with_held_file();
        let weak_store = Arc::downgrade(&store);
        let wait = spawn(async move {
            FillWaiters::park(
                &weak_store,
                &uri,
                &Arc::default(),
                PromotionDefer::Forever,
                FillWait::Release,
            )
            .await
        });
        until_parked(&store, 1).await;
        drop(store);
        timeout(PROMOTE_TIMEOUT, wait)
            .await
            .expect("a dropped store wakes the fill")
            .expect("join");
    }

    /// A parked fill goes once its defer window closes, even though the query still holds the
    /// reader: a file under steady load must not stay lazy for good.
    #[tokio::test]
    async fn a_closed_defer_window_wakes_a_parked_fill() {
        let (_dir, store, uri, _query) = store_with_held_file();
        let start = Instant::now();
        let defer = PromotionDefer::start(SHORT_WINDOW);
        let weak_store = Arc::downgrade(&store);
        timeout(
            PROMOTE_TIMEOUT,
            FillWaiters::park(
                &weak_store,
                &uri,
                &Arc::default(),
                defer,
                FillWait::Quiescence,
            ),
        )
        .await
        .expect("the closed window wakes the fill");
        assert!(
            start.elapsed() >= SHORT_WINDOW,
            "the fill waits out the window first"
        );
    }

    /// A fill cancelled while it waits (its file evicted) is woken, though a query still holds
    /// the file, rather than left parked.
    #[tokio::test]
    async fn a_cancelled_fill_is_woken() {
        let (_dir, store, uri, _query) = store_with_held_file();
        let weak_store = Arc::downgrade(&store);
        let fill: Arc<BackgroundFill> = Arc::default();
        let parked = Arc::clone(&fill);
        let wait = spawn(async move {
            FillWaiters::park(
                &weak_store,
                &uri,
                &parked,
                PromotionDefer::Forever,
                FillWait::Release,
            )
            .await
        });
        until_parked(&store, 1).await;
        fill.cancelled.store(true, Ordering::Release);
        timeout(PROMOTE_TIMEOUT, wait)
            .await
            .expect("the cancellation wakes the fill")
            .expect("join");
    }

    /// A fill already cancelled does not park at all: it returns at once, without waiting two
    /// ticks for the file to look free.
    #[test]
    fn a_cancelled_fill_does_not_park() {
        let (_dir, store, uri, _query) = store_with_held_file();
        assert!(
            FillWaiters::park(
                &Arc::downgrade(&store),
                &uri,
                &Arc::new(cancelled_fill()),
                PromotionDefer::Forever,
                FillWait::Quiescence,
            )
            .now_or_never()
            .is_some(),
            "a cancelled fill goes without parking"
        );
        assert_eq!(store.fill_waiters.parked(), 0);
    }

    /// A caller waiting for promotion lets a fill past its first wait at once, but not past the
    /// wait before a download: the download would pause for the held reader again straight away.
    #[tokio::test]
    async fn a_promotion_waiter_releases_only_the_first_wait() {
        let (_dir, store, uri, query) = store_with_held_file();
        store.n_promotion_waiters.fetch_add(1, Ordering::AcqRel);
        let weak_store = Arc::downgrade(&store);
        timeout(
            PROMOTE_TIMEOUT,
            FillWaiters::park(
                &weak_store,
                &uri,
                &Arc::default(),
                PromotionDefer::Forever,
                FillWait::Release,
            ),
        )
        .await
        .expect("a promotion waiter releases the first wait");

        let wait = spawn(async move {
            FillWaiters::park(
                &weak_store,
                &uri,
                &Arc::default(),
                PromotionDefer::Forever,
                FillWait::Quiescence,
            )
            .await
        });
        until_parked(&store, 1).await;
        sleep(HELD_FOR).await;
        assert!(
            !wait.is_finished(),
            "the wait before a download ignores promotion waiters"
        );
        store.n_promotion_waiters.fetch_sub(1, Ordering::AcqRel);
        drop(query);
        timeout(PROMOTE_TIMEOUT, wait)
            .await
            .expect("a released reader wakes the fill")
            .expect("join");
    }

    /// A download cancelled while it waits for a slot reserves nothing once it gets one.
    /// - The budget has no room for the download, and the only entry is the file's own.
    /// - So a download that reserved before checking its cancel would fail.
    ///
    /// This test pins the check after the slot.
    #[tokio::test]
    async fn a_fill_cancelled_while_queued_reserves_nothing() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = RESERVED_BYTES - 1;
            cfg.prefetch_concurrency = 1;
        });
        let uri = SuperfileUri::new_v4();
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let before = store.current_bytes.load(Ordering::Acquire);
        let permit = hold_fill_slot(&store).await;
        let fill: Arc<BackgroundFill> = Arc::default();
        store.fills.insert(uri, Arc::clone(&fill));
        let exit = FillExit::new(&store, uri, Arc::clone(&fill));
        let storage = Arc::clone(&store.storage);
        let task = spawn(async move {
            let result = lazy_background_fill(
                &exit,
                uri.storage_path(),
                RESERVED_BYTES,
                storage,
                None,
                // A closed window: the first wait lets it go at once, straight to the slot.
                PromotionDefer::start(Duration::ZERO),
            )
            .await;
            drop(exit);
            result
        });
        // One yield runs the fill up to the slot it waits for.
        sleep(FILL_POLL_INTERVAL).await;
        // Cancel with the entry left in place.
        store.remove_entry_cancelling_fill(&uri, |_| false);
        drop(permit);
        timeout(PROMOTE_TIMEOUT, task)
            .await
            .expect("the fill ends")
            .expect("join")
            .expect("a cancelled fill reserves nothing, so it cannot fail to");
        assert_eq!(
            store.current_bytes.load(Ordering::Acquire),
            before,
            "nothing stays charged"
        );
        assert!(store.is_cached(&uri), "and evicted nothing");
    }

    /// A fill waiting to download holds only a weak reference to its store, so dropping the store
    /// ends the fill instead of the fill keeping the whole store alive while a query holds the
    /// reader.
    #[tokio::test]
    async fn a_waiting_fill_does_not_keep_a_dropped_store_alive() {
        let (_dir, store, uri, _query) = store_with_held_file();
        // Let the fill past its first wait, so it parks in the wait before a download.
        store.n_promotion_waiters.fetch_add(1, Ordering::AcqRel);
        let exit = FillExit::new(&store, uri, Arc::default());
        let storage = Arc::clone(&store.storage);
        let fill = spawn(async move {
            lazy_background_fill(
                &exit,
                uri.storage_path(),
                RESERVED_BYTES,
                storage,
                None,
                PromotionDefer::Forever,
            )
            .await
        });
        until_parked(&store, 1).await;
        drop(store);
        timeout(PROMOTE_TIMEOUT, fill)
            .await
            .expect("the fill ends once its store is dropped")
            .expect("join")
            .expect("fill");
    }

    #[test]
    fn uncovered_ranges_are_the_open_ranges_the_blob_does_not_hold_whole() {
        let blob = vec![(100u64, vec![0u8; 50]), (1_000, vec![0u8; 10])];
        let wanted = [
            (100u64, 50u64),
            (110, 20),
            (90, 20),
            (140, 20),
            (1_000, 10),
            (2_000, 5),
            (3_000, 0),
        ];
        assert_eq!(
            uncovered_ranges(&blob, &wanted),
            vec![(90, 20), (140, 20), (2_000, 5)],
            "whole-inside ranges are covered; straddling, outside and only empty ranges are not"
        );
        assert!(uncovered_ranges(&[], &[]).is_empty());
        assert_eq!(uncovered_ranges(&[], &[(0, 1)]), vec![(0, 1)]);
    }

    #[tokio::test]
    async fn reader_lazy_with_hints_known_size_promotes_after_release() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        // Known size, no open_blob → fetches the open batch over the
        // wire (parquet tail + vec + fts ranges) using the fallback
        // header lengths derived from `vec`/`fts` hints.
        let offsets = SubsectionOffsets {
            total_size: total,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        };
        let r = store
            .open_for_query(
                &uri,
                &uri.storage_path(),
                Some(&offsets),
                None,
                ReadIntent::Warm,
            )
            .await
            .expect("lazy hinted cold");
        assert_eq!(r.n_docs(), 1);
        assert_eq!(store.stats().n_cold_fetches, 1);
        drop(r);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("background promotion");
        let r2 = store
            .open_for_query(
                &uri,
                &uri.storage_path(),
                Some(&offsets),
                None,
                ReadIntent::Warm,
            )
            .await
            .expect("warm hinted mmap");
        assert_eq!(store.stats().n_cold_fetches, 1);
        assert!(store.is_mmap_promoted(&uri));
        assert!(r2.parquet_bytes().is_some());
    }

    /// A Warm open of a file the budget cannot hold whole stays a cache entry: lazy, charging
    /// only the blocks it reads, and served to the next open without a second cold fetch. Both
    /// open paths: the size known before the open from the manifest's offsets, and not.
    #[tokio::test]
    async fn a_warm_open_past_the_budget_keeps_a_lazy_entry_for_the_next_open() {
        let bytes = tiny_superfile_bytes();
        let hinted = SubsectionOffsets {
            total_size: bytes.len() as u64,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        };
        for (path, offsets) in [("unknown size", None), ("known size", Some(&hinted))] {
            let (_dir, store) = test_store_with(|cfg| {
                cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
                cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
            });
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, bytes.clone()).await;

            let (first, tier) = store
                .open_for_query_tiered(&uri, &uri.storage_path(), offsets, None, ReadIntent::Warm)
                .await
                .expect("served past the budget");
            assert_eq!(first.n_docs(), 1, "{path}");
            assert_eq!(
                tier,
                OpenTier::Source,
                "{path}: opened against the object store"
            );
            assert!(store.is_cached(&uri), "{path}: kept as a lazy entry");
            assert!(
                !store.is_mmap_promoted(&uri),
                "{path}: nothing downloaded whole"
            );

            let (second, tier) = store
                .open_for_query_tiered(&uri, &uri.storage_path(), offsets, None, ReadIntent::Warm)
                .await
                .expect("served again");
            assert_eq!(
                tier,
                OpenTier::Lazy,
                "{path}: the next open rides the entry"
            );
            assert!(Arc::ptr_eq(&first, &second), "{path}: the same reader");
            assert_eq!(
                store.stats().n_cold_fetches,
                1,
                "{path}: one open for both reads"
            );
            assert!(
                store.stats().current_bytes <= TEST_TINY_BUDGET_BYTES,
                "{path}: the budget still holds"
            );
            store.assert_budget_consistent();
        }
    }

    /// The unknown-size Warm open the budget refuses, step by step: the first open learns the
    /// size on a block source that rides on a whole-file reserve and so charges nothing, the
    /// reserve fails, that source is dropped, and the file is opened once more as a Stream open
    /// whose block source charges what it fills. The entry is that second source's - charged by
    /// its blocks, nothing on disk outside the charge - so a block a later read fills is charged
    /// as it lands, and the next open rides the entry. The budget holds blocks, never the whole
    /// file.
    #[tokio::test]
    async fn an_unknown_size_warm_open_past_the_budget_charges_its_own_blocks() {
        let bytes = multi_block_superfile_bytes();
        let size = bytes.len() as u64;
        assert!(
            size > 2 * CACHE_BLOCK_BYTES,
            "the fixture spans several blocks"
        );
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.disk_budget_bytes = size - 1;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, bytes.clone()).await;

        let (first, tier) = store
            .open_for_query_tiered(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("served past the budget");
        assert_eq!(tier, OpenTier::Source, "opened against the object store");
        store.block_writes_settled().await;

        let entry = store
            .cached
            .get(&uri)
            .map(|e| Arc::clone(e.value()))
            .expect("an entry");
        assert_eq!(
            entry.accounting,
            EntryAccounting::SourceOwned,
            "the entry is charged by its blocks, not by a whole-file reserve"
        );
        let source = entry
            .block_source()
            .cloned()
            .expect("a lazy entry and its block source");
        assert!(
            source.owns_accounting(),
            "the entry's source is the second open's, which charges its own blocks"
        );
        // The entry is not in the cache while the opens run, so both block sources read
        // straight from storage and write nothing: no block file, nothing to charge.
        let opened = source.filled_bytes_handle().load(Ordering::Acquire);
        assert_eq!(opened, 0, "the opens leave nothing filled");
        assert_eq!(
            store.stats().current_bytes,
            0,
            "and nothing charged before the first read"
        );

        // A block no open read: the entry's source fetches it and charges it.
        let (start, len) = (CACHE_BLOCK_BYTES, REFUSAL_READ_BYTES);
        let want = bytes.slice(start as usize..(start + len) as usize);
        assert!(
            source.range(start, len).await.expect("read") == want,
            "the read returns the file's bytes"
        );
        store.block_writes_settled().await;
        let filled = source.filled_bytes_handle().load(Ordering::Acquire);
        assert!(filled > opened, "the read filled a block");
        assert_eq!(
            store.stats().current_bytes,
            filled,
            "the block the read filled is charged, by the entry's source"
        );
        let mut idx_path = store.blocks_path(&uri).into_os_string();
        idx_path.push(".idx");
        if let Ok(idx) = fs::read(PathBuf::from(idx_path)) {
            let ino = fs::metadata(store.blocks_path(&uri))
                .expect("block file")
                .ino();
            let (_, on_disk) = indexed_filled_bytes(size, &idx, ino).expect("a valid block index");
            assert!(
                on_disk <= filled,
                "no block on disk outside the charge ({on_disk} > {filled})"
            );
        }
        store.assert_budget_consistent();

        let (second, tier) = store
            .open_for_query_tiered(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("served again");
        assert_eq!(tier, OpenTier::Lazy, "the next open rides the entry");
        assert!(Arc::ptr_eq(&first, &second), "the same reader");
        assert_eq!(store.stats().n_cold_fetches, 1, "one cold fetch for both");
    }

    #[tokio::test]
    async fn vector_open_skips_fill_fts_open_starts_it() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // Vector modality: block-cache only — no background fill.
        let vector_reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("vector lazy open");
        drop(vector_reader);
        sleep(FOREGROUND_GUARD_HOLD).await;
        assert!(
            !store.is_mmap_promoted(&uri),
            "vector open must not spawn background fill"
        );

        // FTS/SQL modality on the same URI starts fill after the fact.
        let fts_reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("fts lazy open");
        drop(fts_reader);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("FTS open must start background fill");
        assert!(store.is_mmap_promoted(&uri));
    }

    /// A vector-opened entry that later gets an FTS open promotes to a Mapped entry that KEEPS its
    /// block source as the vector hole: parquet and FTS serve from the mmap, the vector blob stays
    /// on the block cache. Pins the `Residency::Mapped { vector_source: Some }` path and its
    /// `block_source()` accessor arm, which the scalar fixtures never reach.
    #[tokio::test]
    async fn vector_open_promotes_to_mapped_keeping_the_hole() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;

        // Vector modality: lazy, block-cache only, no fill.
        let vector_reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("vector open");
        drop(vector_reader);

        // FTS modality starts the fill, which promotes while leaving the vector blob sparse.
        let fts_reader = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("fts open");
        drop(fts_reader);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");

        let entry = store.cached.get(&uri).expect("entry cached");
        assert!(entry.is_mapped(), "promoted to a Mapped entry");
        assert!(
            entry.block_source().is_some(),
            "the vector blob stays on the block cache as the retained hole",
        );
    }

    /// A promotion that left the vector blob on the block cache must survive a restart. The next
    /// process reuses the local copy and its `.blocks`, with no GET; before, it took the copy for a
    /// complete file, failed to open it (the hole reads as zeros where the vector header should
    /// be), deleted both and fetched the superfile again.
    #[tokio::test]
    async fn vector_hole_promotion_is_reused_after_a_restart() {
        for intent in [ReadIntent::Stream, ReadIntent::Warm] {
            let dir = TempDir::new().expect("tempdir");
            let local: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
            let recording = RecordingStorage::over(local);
            let store = DiskCacheStore::new_unpinned(
                Arc::clone(&recording) as Arc<dyn StorageProvider>,
                DiskCacheConfig {
                    cache_root: dir.path().join("cache"),
                    cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                    mmap_cold_threshold_secs: 0,
                    ..Default::default()
                },
            )
            .expect("store");
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;

            // An FTS or SQL read warms the file; the fill leaves the vector blob out of it.
            drop(
                store
                    .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                    .await
                    .expect("warm open"),
            );
            store
                .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
                .await
                .expect("promote");
            store.block_writes_settled().await;
            assert!(
                store.hole_path(&uri).exists() && !store.cache_path(&uri).exists(),
                "a copy missing its vector blob is kept under its own name"
            );

            // Restart: a fresh store over the same directory, and the old one gone.
            let restarted = reopen_store(&store, |cfg| {
                cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            });
            drop(store);

            let before = restarted.stats();
            let gets_before = recording.n_ranges();
            restarted
                .open_for_query(&uri, &uri.storage_path(), None, None, intent)
                .await
                .expect("read after the restart");
            let after = restarted.stats();
            assert_eq!(
                after.n_cold_fetches - before.n_cold_fetches,
                0,
                "{intent:?}: the restart must not fetch the superfile again"
            );
            assert_eq!(
                after.n_disk_reuses - before.n_disk_reuses,
                1,
                "{intent:?}: the restart must reuse the local copy"
            );
            assert_eq!(
                recording.n_ranges(),
                gets_before,
                "{intent:?}: the vector header comes from .blocks, not object storage"
            );
            assert!(
                restarted.hole_path(&uri).exists() && restarted.blocks_path(&uri).exists(),
                "{intent:?}: the copy and its .blocks are still on disk"
            );
            restarted.assert_budget_consistent();
        }
    }

    /// A holed copy whose `.blocks` is gone (evicted under budget pressure, or dropped by the scan
    /// for a broken index) is still reused after a restart: the vector header is fetched by range,
    /// not the whole superfile again.
    #[tokio::test]
    async fn holed_copy_without_its_blocks_is_still_reused_after_a_restart() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open"),
        );
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");

        let restarted = reopen_store(&store, |cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        drop(store);
        restarted.drop_block_file(&uri);

        let before = restarted.stats();
        restarted
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("read after the restart");
        let after = restarted.stats();
        assert_eq!(after.n_cold_fetches, before.n_cold_fetches, "no refetch");
        assert_eq!(after.n_disk_reuses - before.n_disk_reuses, 1, "copy reused");
        assert!(restarted.hole_path(&uri).exists(), "copy still on disk");
        restarted.assert_budget_consistent();
    }

    /// Evicting a promoted copy with a hole deletes it under its own name, together with the
    /// `.blocks` that held its vector ranges.
    #[tokio::test]
    async fn evicting_a_holed_copy_deletes_it_and_its_blocks() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open"),
        );
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");
        assert!(store.hole_path(&uri).exists(), "promoted with a hole");

        store.evict_at_least(1).await.expect("evict");

        assert!(!store.hole_path(&uri).exists(), "the holed copy is deleted");
        assert!(!store.blocks_path(&uri).exists(), "and so is its .blocks");
        store.assert_budget_consistent();
    }

    /// A query holds its lazy reader while the background fill promotes the file. The reader's
    /// block source is no longer the entry's, so it cannot fill; a block it never read must come
    /// from the promoted local copy, not from object storage.
    #[tokio::test]
    async fn a_replaced_block_source_reads_the_promoted_copy() {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&recording) as Arc<dyn StorageProvider>,
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                // The fill goes ahead while the query still holds its reader.
                promotion_defer_timeout: Duration::ZERO,
                ..Default::default()
            },
        )
        .expect("store");
        let uri = SuperfileUri::new_v4();
        let bytes = multi_block_superfile_bytes();
        assert!(
            bytes.len() as u64 > 2 * CACHE_BLOCK_BYTES,
            "the fixture spans several blocks"
        );
        put_superfile(&store, &uri, bytes.clone()).await;

        let held = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("warm open");
        let stale = store
            .cached
            .get(&uri)
            .and_then(|entry| entry.block_source().cloned())
            .expect("a lazy entry and its block source");
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");

        // A block in the middle of the file: no open read touched it.
        let (start, len) = (CACHE_BLOCK_BYTES, 4096);
        let want = bytes.slice(start as usize..(start + len) as usize);
        let gets = recording.n_ranges();
        assert!(
            stale.range(start, len).await.expect("read") == want,
            "the read returns the file's bytes"
        );
        assert_eq!(recording.n_ranges(), gets, "the read made no GET");
        drop(held);
    }

    /// Every read a replaced source cannot serve from its blocks comes from the whole-file copy:
    /// an exact passthrough range (how FTS subsection reads go) as much as a block it never filled.
    #[tokio::test]
    async fn every_bypassed_read_of_a_replaced_source_reads_the_whole_file_copy() {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let storage = Arc::clone(&recording) as Arc<dyn StorageProvider>;
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                ..Default::default()
            },
        )
        .expect("store");
        let uri = SuperfileUri::new_v4();
        let bytes = multi_block_superfile_bytes();
        put_superfile(&store, &uri, bytes.clone()).await;

        let passthrough = (CACHE_BLOCK_BYTES / 2, 4096);
        let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
            storage,
            uri.storage_path(),
            bytes.len() as u64,
        ));
        let source = BlockCachedSource::new_with_accounting(
            inner,
            Arc::downgrade(&store),
            uri,
            store.blocks_path(&uri),
            true,
            SourceLayout {
                fts_hole: Some(passthrough),
                ..SourceLayout::default()
            },
        );
        // The whole file lands in the cache; this source is not the entry's.
        store
            .insert_warm(&uri, bytes.clone())
            .await
            .expect("whole-file copy");

        let gets = recording.n_ranges();
        for (start, len) in [passthrough, (CACHE_BLOCK_BYTES, 4096)] {
            let want = bytes.slice(start as usize..(start + len) as usize);
            assert!(
                source.range(start, len).await.expect("read") == want,
                "range {start}+{len} returns the file's bytes"
            );
            assert!(
                source.range_exact(start, len).await.expect("exact read") == want,
                "an exact range {start}+{len} returns the file's bytes"
            );
        }
        assert_eq!(recording.n_ranges(), gets, "no read went to object storage");
    }

    /// A copy with a vector hole is not used by a replaced source: its hole is zeros on disk. A
    /// read of the hole through the replaced source goes to object storage and returns the real
    /// vector bytes, never the zeros.
    #[tokio::test]
    async fn a_replaced_source_never_reads_a_holed_copys_zeros() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_vector_superfile_bytes();
        put_superfile(&store, &uri, bytes.clone()).await;
        let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
            Arc::clone(&store.storage),
            uri.storage_path(),
            bytes.len() as u64,
        ));
        // Its own block file, so its blocks start empty and every read takes the replaced path;
        // sharing the entry's would adopt the blocks the open already filled.
        let replaced = BlockCachedSource::new_with_accounting(
            inner,
            Arc::downgrade(&store),
            uri,
            store.config.cache_root.join("replaced.blocks"),
            true,
            SourceLayout::ALL_ALIGNED,
        );
        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open"),
        );
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");
        assert!(
            store
                .cached
                .get(&uri)
                .is_some_and(|entry| entry.block_source().is_some()),
            "promoted with the vector blob left out of the file"
        );

        let (start, len) = {
            let reader = SuperfileReader::open(bytes.clone()).expect("open original");
            vector_blob_range(&reader).expect("vector range")
        };
        let want = bytes.slice(start as usize..(start + len) as usize);
        assert!(
            replaced.range(start, len).await.expect("read the hole") == want,
            "the hole reads as the real vector bytes"
        );
    }

    /// A holed copy's own block source that cannot fill (budget full, the file pinned so nothing
    /// can be evicted) still finishes a read of the hole, from object storage. It must not ask the
    /// cache for the whole file: the holed copy's hole fallback is this same source.
    #[tokio::test]
    async fn a_holed_copys_own_source_that_cannot_fill_still_finishes_a_read() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_vector_superfile_bytes();
        put_superfile(&store, &uri, bytes.clone()).await;
        let (hole_start, hole_len) = {
            let reader = SuperfileReader::open(bytes.clone()).expect("open original");
            vector_blob_range(&reader).expect("vector range")
        };

        // Nothing can be charged and nothing evicted, so the source can never fill a block.
        store.budget_bytes.store(0, Ordering::Release);
        store.set_pinned_fn(Arc::new(move || HashSet::from([uri])));

        // The holed copy as a fill leaves it: the vector range zeroed on disk, served by `source`.
        let mut holed = bytes.to_vec();
        holed[hole_start as usize..(hole_start + hole_len) as usize].fill(0);
        fs::create_dir_all(&store.config.cache_root).expect("cache root");
        fs::write(store.hole_path(&uri), &holed).expect("holed copy");
        let (mmap, local) = mmap_readonly_with_handle(&store.hole_path(&uri)).expect("mmap");
        let inner: Arc<dyn LazyByteSource> = Arc::new(StorageRangeSource::with_known_size(
            Arc::clone(&store.storage),
            uri.storage_path(),
            bytes.len() as u64,
        ));
        let source = BlockCachedSource::new_with_accounting(
            inner,
            Arc::downgrade(&store),
            uri,
            store.blocks_path(&uri),
            true,
            SourceLayout::ALL_ALIGNED,
        );
        let entry = store
            .holed_mmap_entry(
                mmap,
                local,
                (hole_start, hole_len),
                Arc::clone(&source),
                bytes.len() as u64,
            )
            .await
            .expect("holed entry");
        store.admit_entry(uri, entry);

        let read = timeout(READ_DEADLINE, source.range(hole_start, hole_len))
            .await
            .expect("the read finishes")
            .expect("read the hole");
        let want = bytes.slice(hole_start as usize..(hole_start + hole_len) as usize);
        assert!(read == want, "the hole reads as the real vector bytes");
    }

    #[tokio::test]
    async fn background_fill_waits_for_same_uri_reader() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let reader = store.reader(&uri).await.expect("lazy cold");
        let _foreground = ForegroundQueryGuard::enter();
        sleep(FOREGROUND_GUARD_HOLD).await;
        assert!(
            !store.is_mmap_promoted(&uri),
            "background promotion must yield while this URI's lazy reader is held"
        );

        drop(reader);
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promotion resumes after the URI reader is released");
    }

    #[tokio::test]
    async fn reader_for_one_uri_does_not_pause_another_uri_fill() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.prefetch_concurrency = 2;
        });
        let held_uri = SuperfileUri::new_v4();
        let fill_uri = SuperfileUri::new_v4();
        put_superfile(&store, &held_uri, tiny_superfile_bytes()).await;
        put_superfile(&store, &fill_uri, tiny_superfile_bytes()).await;

        let held_reader = store.reader(&held_uri).await.expect("held lazy reader");
        let _fill_reader = store.reader(&fill_uri).await.expect("fill lazy reader");
        drop(_fill_reader);
        let _foreground = ForegroundQueryGuard::enter();
        store
            .wait_until_mmap_promoted(&fill_uri, PROMOTE_TIMEOUT)
            .await
            .expect("unrelated URI fill must proceed while another URI is held");
        assert!(
            !store.is_mmap_promoted(&held_uri),
            "held URI must still wait for its own reader release"
        );
        drop(held_reader);
    }

    /// `HoleFallbackSource` serves ranges outside the vector hole from the
    /// local (filled) bytes and ranges inside the hole from the fallback block
    /// cache, stitching a spanning read from both halves. Covers the geometry
    /// helpers (`overlaps_hole` / `fully_in_hole` / `hole_end`) alongside
    /// `size`, `range`, and the `try_get_range_sync` fast path.
    #[tokio::test]
    async fn hole_fallback_source_routes_local_and_fallback_by_hole() {
        use crate::superfile::lazy_source::BytesLazyByteSource;

        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        // local = the "filled" bytes (0xAA); the fallback's inner = the block
        // cache side (0xBB) that serves the excluded vector hole.
        let local: Arc<dyn LazyByteSource> =
            Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0xAAu8; 100])));
        let remote: Arc<dyn LazyByteSource> =
            Arc::new(BytesLazyByteSource::new(Bytes::from(vec![0xBBu8; 100])));
        let fallback = BlockCachedSource::new_pre_reserved(
            remote,
            Arc::downgrade(&store),
            uri,
            store.blocks_path(&uri),
            SourceLayout::ALL_ALIGNED,
        );
        let hfs = HoleFallbackSource {
            local,
            hole_start: 40,
            hole_len: 20,
            fallback,
        };

        assert_eq!(hfs.size(), 100, "size reflects the local (full) source");

        // Wholly before the hole → local bytes.
        assert_eq!(
            &hfs.range(0, 10).await.expect("pre-hole")[..],
            &[0xAAu8; 10]
        );
        // Wholly inside the hole → fallback bytes.
        assert_eq!(
            &hfs.range(40, 20).await.expect("in-hole")[..],
            &[0xBBu8; 20]
        );
        // Spanning: local[30..40] + fallback[40..60] + local[60..70].
        let mut want = vec![0xAAu8; 10];
        want.extend_from_slice(&[0xBBu8; 20]);
        want.extend_from_slice(&[0xAAu8; 10]);
        assert_eq!(
            &hfs.range(30, 40).await.expect("spanning")[..],
            &want[..],
            "spanning read stitches local + fallback + local in order",
        );

        // Sync fast path: a read outside the hole resolves locally; a spanning
        // read returns None to force the async path.
        assert_eq!(
            hfs.try_get_range_sync(0, 10).as_deref(),
            Some(&[0xAAu8; 10][..]),
            "sync read outside the hole comes from local",
        );
        assert!(
            hfs.try_get_range_sync(30, 40).is_none(),
            "spanning sync read forces the async path",
        );
    }

    #[test]
    fn chunk_fetch_ranges_skips_vector_hole() {
        assert_eq!(
            chunk_fetch_ranges(0, 100, None),
            vec![(0, 100)],
            "no hole ⇒ full chunk"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((100, 50))),
            vec![(0, 100)],
            "hole after chunk ⇒ full chunk"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((0, 100))),
            Vec::<(u64, u64)>::new(),
            "chunk fully inside hole ⇒ no GET"
        );
        assert_eq!(
            chunk_fetch_ranges(50, 150, Some((0, 200))),
            Vec::<(u64, u64)>::new(),
            "chunk fully inside larger hole ⇒ no GET"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((40, 20))),
            vec![(0, 40), (60, 100)],
            "hole splits chunk into two fetch ranges"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((80, 40))),
            vec![(0, 80)],
            "hole overlapping chunk end ⇒ leading fetch only"
        );
        assert_eq!(
            chunk_fetch_ranges(0, 100, Some((0, 40))),
            vec![(40, 100)],
            "hole overlapping chunk start ⇒ trailing fetch only"
        );
    }

    #[tokio::test]
    async fn same_uri_reader_pauses_in_flight_background_ranges() {
        let (dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_streams = 1;
            cfg.cold_fetch_chunk_bytes = 1;
        });
        let uri = SuperfileUri::new_v4();
        let storage_uri = uri.storage_path();
        store
            .storage
            .put_atomic(&storage_uri, Bytes::from(vec![7u8; PREEMPT_TEST_BYTES]))
            .await
            .expect("put background-fill payload");
        let destination = dir.path().join("preempt.tmp");
        let fill_store = Arc::clone(&store);
        let fill_storage = Arc::clone(&store.storage);
        let fill_destination = destination.clone();
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let signal = Arc::downgrade(&store.cached.get(&uri).expect("cached").reader);
        let fill = spawn(async move {
            let mut filled = Vec::new();
            let outcome = cold_fetch_to_disk_cancelable(
                &fill_store,
                &uri,
                &BackgroundFill::default(),
                &fill_storage,
                &storage_uri,
                &fill_destination,
                PREEMPT_TEST_BYTES as u64,
                &mut filled,
                None,
                // This test is about the yield itself, so keep the window
                // open for its whole duration.
                PromotionDefer::Forever,
            )
            .await;
            (outcome, filled)
        });

        timeout(PROMOTE_TIMEOUT, async {
            while !destination.exists() {
                yield_now().await;
            }
        })
        .await
        .expect("background fill started");
        // A query holding the file's reader is the per-URI pause.
        let foreground = signal.upgrade().expect("the entry holds the reader");
        let _ = ForegroundQueryGuard::enter();
        let (outcome, filled) = fill.await.expect("background task joined");
        let outcome = outcome.expect("background fill returned an outcome");
        assert_eq!(outcome, BackgroundFillOutcome::Paused);
        // Resume cursor is sized to the object's chunk count and preserved
        // across the pause so a later attempt resumes rather than restarting.
        assert_eq!(filled.len(), PREEMPT_TEST_BYTES);
        assert!(
            filled.iter().any(|&done| !done),
            "a same-URI pause must leave unfinished chunks for the resume"
        );
        drop(foreground);
    }

    // ----- wait_until_mmap_promoted timeout path -----

    // ----- an open dropped midway, as a cancelled query drops it -----

    /// Offsets for a file of `total_size` with no open blob, so a known-size open fetches its open
    /// batch over the wire and has a read to be stopped in.
    fn offsets_without_open_blob(total_size: u64) -> SubsectionOffsets {
        SubsectionOffsets {
            total_size,
            vec: None,
            fts: None,
            vec_open_ranges: Vec::new(),
            fts_open_ranges: Vec::new(),
            open_blob: Vec::new(),
        }
    }

    /// A store over recording storage with background fill on, and the recording.
    fn store_over_recording() -> (TempDir, Arc<DiskCacheStore>, Arc<RecordingStorage>) {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&recording) as Arc<dyn StorageProvider>,
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("store");
        (dir, store, recording)
    }

    /// Wait until `recording` holds a read paused.
    async fn until_a_read_is_paused(recording: &RecordingStorage) {
        timeout(READ_DEADLINE, async {
            while !recording.is_paused() {
                sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("a read starts");
    }

    /// A known-size Warm open reserves the whole file for its background download before it
    /// fetches the open batch. Dropped during that fetch, the reservation comes back: nothing was
    /// admitted to own it, and a charge left behind would hold budget for no file until restart.
    #[tokio::test]
    async fn a_lazy_open_dropped_mid_read_gives_back_its_whole_file_reservation() {
        let (_dir, store, recording) = store_over_recording();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;

        recording.pause_next_read();
        let opening = spawn({
            let store = Arc::clone(&store);
            async move {
                let offsets = offsets_without_open_blob(total);
                store
                    .cold_fetch_lazy(
                        &uri,
                        &uri.storage_path(),
                        Some(&offsets),
                        store.resolve_storage(None),
                        true,
                    )
                    .await
                    .map(|_| ())
            }
        });
        until_a_read_is_paused(&recording).await;
        assert_eq!(
            store.stats().current_bytes,
            total,
            "the open holds the whole file reserved while it reads"
        );

        opening.abort();
        assert!(opening.await.expect_err("aborted").is_cancelled());
        assert_eq!(store.stats().current_bytes, 0, "the reservation came back");
        assert!(!store.is_cached(&uri), "nothing was admitted");
        store.assert_budget_consistent();
    }

    /// A lazy open that adopts a `.blocks` file left on disk takes over the charge the file's
    /// record carried. Dropped before its entry is admitted (a cancelled query), it hands the file
    /// back to a record, so the file stays charged and eviction can still reach it.
    #[tokio::test]
    async fn a_lazy_open_dropped_after_adopting_a_block_file_keeps_the_file_charged() {
        let (_dir, store, recording) = store_over_recording();
        let uri = SuperfileUri::new_v4();
        let bytes = multi_block_vector_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        // The open reads only the tail and the vector header, neither of them in the last block.
        let last = ((total - 1) / CACHE_BLOCK_BYTES) as u32;
        let seeded = seed_valid_block_file(&store, &uri, total, &[last]);
        let restarted = reopen_store(&store, |cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        drop(store);
        assert_eq!(
            restarted.stats().current_bytes,
            seeded,
            "the restart counted the file"
        );

        // With no size hints the open reads the tail first, then the vector header through its
        // block source, which adopts the file before it reads.
        recording.pause_read_after(1);
        let opening = spawn({
            let restarted = Arc::clone(&restarted);
            async move {
                restarted
                    .cold_fetch_lazy(
                        &uri,
                        &uri.storage_path(),
                        None,
                        restarted.resolve_storage(None),
                        false,
                    )
                    .await
                    .map(|_| ())
            }
        });
        until_a_read_is_paused(&recording).await;
        opening.abort();
        assert!(opening.await.expect_err("aborted").is_cancelled());

        assert!(!restarted.is_cached(&uri), "nothing was admitted");
        assert_eq!(
            restarted.stats().current_bytes,
            seeded,
            "the file is still charged"
        );
        restarted.assert_budget_consistent();
        restarted.evict_at_least(seeded).await.expect("evict");
        assert!(
            !restarted.blocks_path(&uri).exists(),
            "eviction reaches the file"
        );
        assert_eq!(restarted.stats().current_bytes, 0);
    }

    /// The same reservation comes back when a read of the open batch fails, not only when the open
    /// is dropped.
    #[tokio::test]
    async fn a_lazy_open_whose_open_batch_read_fails_gives_back_its_reservation() {
        let (dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let total = bytes.len() as u64;
        put_superfile(&store, &uri, bytes).await;
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let faults = FaultStorage::wrap(local);
        faults.fail(FaultOp::GetRange, &uri.0.to_string(), 1);

        let offsets = offsets_without_open_blob(total);
        let opened = store
            .cold_fetch_lazy(
                &uri,
                &uri.storage_path(),
                Some(&offsets),
                Arc::clone(&faults) as Arc<dyn StorageProvider>,
                true,
            )
            .await;
        assert!(opened.is_err(), "the failed read fails the open");
        assert_eq!(faults.fired(), 1, "the armed read is the one that failed");
        assert_eq!(store.stats().current_bytes, 0, "the reservation came back");
        store.assert_budget_consistent();
    }

    /// A restart counts a holed copy found on disk and reuses it on the next open, which reads the
    /// copy's vector header through the block cache. Dropped during that read, the reuse gives the
    /// restart's charge back and leaves no record, so nothing can count the copy twice; the next
    /// open still finds the copy and reuses it, charged once.
    #[tokio::test]
    async fn a_holed_copy_reuse_dropped_mid_read_gives_back_its_charge() {
        let (_dir, store, recording) = store_over_recording();
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open"),
        );
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");
        store.block_writes_settled().await;
        let restarted = reopen_store(&store, |cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        drop(store);
        // Without its blocks the copy's vector header is fetched by range.
        restarted.drop_block_file(&uri);
        let counted = restarted.stats().current_bytes;
        assert!(counted > 0, "the restart counted the copy");

        recording.pause_next_read();
        let reusing = spawn({
            let restarted = Arc::clone(&restarted);
            async move {
                restarted
                    .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
                    .await
                    .map(|_| ())
            }
        });
        until_a_read_is_paused(&recording).await;
        reusing.abort();
        assert!(reusing.await.expect_err("aborted").is_cancelled());
        assert_eq!(restarted.stats().current_bytes, 0, "the charge came back");
        assert!(!restarted.unindexed.contains_key(&uri), "no record left");
        restarted.assert_budget_consistent();

        let before = restarted.stats();
        restarted
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("read after the dropped one");
        let after = restarted.stats();
        assert_eq!(after.n_cold_fetches, before.n_cold_fetches, "no refetch");
        assert_eq!(after.n_disk_reuses - before.n_disk_reuses, 1, "copy reused");
        restarted.assert_budget_consistent();
    }

    /// Download streams and chunk size for the dropped hybrid-fill test: small chunks, so the
    /// file is split across several downloads and some land while one is held.
    const HYBRID_TEST_STREAMS: usize = 4;
    const HYBRID_TEST_CHUNK_BYTES: u64 = 256;

    /// A hybrid open dropped while its chunk downloads run, as a cancelled query drops it, leaves
    /// nothing behind: its reservation comes back, and the temp file the downloads were filling is
    /// deleted rather than left on disk at full size where nothing charges it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_hybrid_open_dropped_mid_download_leaves_no_temp_file() {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let recording = RecordingStorage::over(local);
        let store = DiskCacheStore::new_unpinned(
            Arc::clone(&recording) as Arc<dyn StorageProvider>,
            DiskCacheConfig {
                cache_root: dir.path().join("cache"),
                cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
                cold_fetch_streams: HYBRID_TEST_STREAMS,
                cold_fetch_chunk_bytes: HYBRID_TEST_CHUNK_BYTES,
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            },
        )
        .expect("store");
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // Every chunk download holds the storage; once they are all gone the count is back here.
        let holders_at_rest = Arc::strong_count(&recording);
        recording.pause_next_read();
        let opening = spawn({
            let store = Arc::clone(&store);
            async move {
                store
                    .cold_fetch_hybrid(&uri, &uri.storage_path(), store.resolve_storage(None))
                    .await
                    .map(|_| ())
            }
        });
        until_a_read_is_paused(&recording).await;
        assert_eq!(leftover_tempfiles(&store), 1, "the download is under way");

        opening.abort();
        assert!(opening.await.expect_err("aborted").is_cancelled());
        timeout(READ_DEADLINE, async {
            while Arc::strong_count(&recording) > holders_at_rest {
                sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("the held download is aborted, not left running");
        assert_eq!(leftover_tempfiles(&store), 0, "the temp file is deleted");
        assert_eq!(store.stats().current_bytes, 0, "the reservation came back");
        store.assert_budget_consistent();
    }
}
