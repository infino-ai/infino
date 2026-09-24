// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cold fetch: pull a superfile from object storage into the cache. The three
//! fetch shapes (whole, hybrid, lazy), the byte pumps that download it, and the
//! finalize step that lands it on disk as an mmap-backed entry.

use std::{
    fs,
    io::SeekFrom,
    mem,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
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
    task::{JoinHandle, spawn_blocking},
};
use tracing::{Instrument, debug_span};

use crate::{
    config::global as global_config,
    runtime_bridge::carry_span,
    runtime_metrics::io::scope_background,
    storage::{StorageError, StorageProvider},
    superfile::{
        BytesLazyByteSource, LazyByteSource, LazyByteSourceError, PrefetchedSource,
        format::{footer, kv},
        reader::{OpenOptions, SuperfileReader},
    },
    supertable::{
        StorageRangeSource,
        manifest::{SubsectionOffsets, SuperfileUri},
        reader_cache::{
            block_source::BlockCachedSource,
            config::ColdFetchMode,
            disk::{sources::mmap_readonly_with_handle, *},
        },
    },
    utils::trace::{OpOrigin, detached},
};

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

    /// Install the whole file a background fill (or the hybrid finalizer) produced. `entry` is
    /// already mmapped from `tmp_path`.
    ///
    /// It replaces the entry it started from (`owner` is that entry's reader) or any other lazy
    /// entry, never someone else's whole file (see [`fill_may_replace`]). Its charge is covered by
    /// what the replaced entry held (if charged in full), then `own_reservation`, then spare
    /// budget; if that is not enough, the fill yields rather than leave bytes uncharged.
    ///
    /// Installed, the tempfile is renamed to `final_path`; the mmap follows the file. Otherwise the
    /// tempfile is deleted and only `own_reservation` is released.
    pub(crate) fn install_promoted_entry(
        &self,
        uri: SuperfileUri,
        entry: Arc<CachedEntry>,
        tmp_path: &Path,
        final_path: &Path,
        owner: &Weak<SuperfileReader>,
        own_reservation: Option<u64>,
    ) -> Option<Arc<CachedEntry>> {
        let charge = entry.size_bytes.load(Ordering::Acquire);
        let own = own_reservation.unwrap_or(0);

        // Decide under the shard lock, touch the files after it.
        let (installed, to_release) = match self.cached.entry(uri) {
            Entry::Occupied(mut occupied) if fill_may_replace(occupied.get(), owner) => {
                let current = occupied.get();
                let inherited = match current.accounting {
                    EntryAccounting::Eager => current.size_bytes.load(Ordering::Acquire),
                    // Its block source keeps charging its blocks until it drops.
                    EntryAccounting::SourceOwned => 0,
                };
                let held = own + inherited;
                let covered = held >= charge || self.try_reserve_without_evicting(charge - held);
                if covered {
                    let replaced = mem::replace(occupied.get_mut(), Arc::clone(&entry));
                    // Leave the shard lock before the old entry drops: its last reference may
                    // fsync a block index.
                    drop(occupied);
                    drop(replaced);
                    (true, held.saturating_sub(charge))
                } else {
                    (false, own)
                }
            }
            // Someone else's whole file, or evicted mid-fill: the download was wasted.
            Entry::Occupied(_) | Entry::Vacant(_) => (false, own),
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

    /// Whether a finished fill could still install over what the slot holds now. The install
    /// re-checks under the shard lock; this only spares the fill work it would throw away.
    fn fill_can_install(&self, uri: &SuperfileUri, owner: &Weak<SuperfileReader>) -> bool {
        self.cached
            .get(uri)
            .is_some_and(|entry| fill_may_replace(&entry, owner))
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

        let file = tokio::fs::File::create(&tmp).await?;
        file.set_len(size).await?;
        let file = Arc::new(tokio::sync::Mutex::new(file));

        // Per-chunk slot for the foreground buffer assembly.
        let chunks: Arc<tokio::sync::Mutex<Vec<Option<(u64, Bytes)>>>> =
            Arc::new(tokio::sync::Mutex::new(vec![None; n_chunks as usize]));

        let mut fetch_handles = Vec::with_capacity(n_chunks as usize);
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

            fetch_handles.push(tokio::spawn(
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
            ));
        }

        // 2. Await all fetches (NOT pwrites). Foreground bytes
        //    are now complete.
        for h in fetch_handles {
            h.await
                .map_err(|e| DiskCacheError::SuperfileOpen(format!("fetch join: {e}")))??;
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
        let foreground_reader = SuperfileReader::open_with(
            foreground_bytes,
            OpenOptions {
                verify_crc: self.config.verify_crc_on_open,
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

    /// Start the background fill for a lazy entry, at most once. Only `Warm` reads call this. The
    /// vector blob, if any, is left out of the download and keeps coming from the block cache.
    pub(crate) fn maybe_spawn_background_fill(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        entry: &CachedEntry,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) {
        if skip_background_fill() || entry.is_mapped() {
            return;
        }
        // Only a Paged entry can start a fill, and its latch makes that happen at most once.
        let Some(fill_spawned) = entry.fill_spawned() else {
            return;
        };
        if fill_spawned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        // A Stream-opened (SourceOwned) entry charges only its filled blocks, so `size_bytes` is
        // not the file size and nothing is reserved for the download yet. Take the size from the
        // block source and reserve below.
        let size = entry
            .block_source()
            .map(|bs| bs.size())
            .filter(|&s| s > 0)
            .unwrap_or_else(|| entry.size_bytes.load(Ordering::Acquire));
        let needs_reserve = matches!(entry.accounting, EntryAccounting::SourceOwned);
        let skip_vec = vector_blob_range(&entry.reader);

        // The vector blob stays on this entry's block source, which charges its own blocks, so
        // reserve only what the file will hold.
        let reservation = match skip_vec {
            Some((_, vec_len)) if needs_reserve => size.saturating_sub(vec_len),
            _ => size,
        };

        // The defer window opens now, not when the task is scheduled, so a queued task cannot
        // extend it.
        let defer = PromotionDefer::start(self.config.promotion_defer_timeout);
        let store = Arc::downgrade(self);
        let reader = Arc::downgrade(&entry.reader);
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
                if needs_reserve {
                    let Some(s) = store.upgrade() else {
                        return;
                    };
                    if s.reserve_manual(reservation).await.is_err() {
                        return;
                    }
                }
                let _ = lazy_background_fill(
                    store,
                    reader,
                    uri_owned,
                    storage_uri_owned,
                    size,
                    needs_reserve.then_some(reservation),
                    fetch_storage,
                    skip_vec,
                    defer,
                )
                .await;
            }
            .instrument(fill_span),
        );
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
                    && !blob.iter().any(|(start, bytes)| {
                        *start <= off && off + len <= *start + bytes.len() as u64
                    })
            })
            .collect()
    }

    /// Lazy cold-fetch path. Foreground builds a reader via
    /// `SuperfileReader::open_lazy_with(StorageRangeSource)`;
    /// background task waits for foreground lazy readers to release,
    /// then downloads the full superfile to NVMe, mmaps it, and replaces
    /// the cache entry.
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
    pub(crate) async fn cold_fetch_lazy(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        offsets: Option<&SubsectionOffsets>,
        fetch_storage: Arc<dyn StorageProvider>,
        allow_background_fill: bool,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = storage_key.to_owned();
        let block_source_arc: Arc<BlockCachedSource>;
        let (lazy_reader, size) = if let Some(offsets) = offsets {
            let total_size = offsets.total_size;

            // Match `SuperfileReader::open_lazy_with`'s parquet tail
            // speculation length so the overlay covers the entire
            // upcoming `source.tail()` call.
            let parquet_tail_len = PARQUET_TAIL_SPEC_BYTES.min(total_size);
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
                !allow_background_fill,
                // FTS subsection reads bypass block rounding (exact ranges);
                // see the `passthrough` field docs.
                offsets.fts,
            );
            block_source_arc = Arc::clone(&block_source);
            let mut overlay = PrefetchedSource::new(block_source);

            // Whatever the manifest part carried rides straight into the
            // overlay; the rest of the open batch — the parquet tail and
            // any open range the writer left out of the blob because
            // copying it into every manifest read would cost more than
            // the round trip (a large superfile's term dictionary) — is
            // fetched over the wire in one parallel wave. A complete blob
            // means zero open-time GETs against the superfile object.
            for (off, bytes) in &offsets.open_blob {
                overlay.install(*off, Bytes::copy_from_slice(bytes));
            }
            let tail_range = (parquet_tail_start, parquet_tail_len);
            let tail_missing = parquet_tail_len > 0
                && !Self::uncovered_ranges(&offsets.open_blob, &[tail_range]).is_empty();
            let vec_missing = Self::uncovered_ranges(&offsets.open_blob, &vec_ranges);
            let fts_missing = Self::uncovered_ranges(&offsets.open_blob, &fts_ranges);
            if tail_missing || !vec_missing.is_empty() || !fts_missing.is_empty() {
                let storage_for_parquet = Arc::clone(&fetch_storage);
                let storage_for_vec = Arc::clone(&fetch_storage);
                let storage_for_fts = Arc::clone(&fetch_storage);
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
                let vec_fut =
                    async move { fetch_hint_ranges(storage_for_vec, vec_uri, vec_missing).await };
                let fts_fut =
                    async move { fetch_hint_ranges(storage_for_fts, fts_uri, fts_missing).await };

                let (parquet_bytes, vec_pre, fts_pre) =
                    futures::try_join!(parquet_fut, vec_fut, fts_fut)?;
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
            (lazy_reader, total_size)
        } else {
            // Unknown-size path: avoid the cold-open HEAD round-trip.
            // The first `tail()` inside `open_lazy_with` is a native
            // suffix-range GET that returns both footer bytes and total
            // object size, then patches the source's size atomic.
            let range_src: Arc<dyn LazyByteSource> =
                Arc::new(StorageRangeSource::with_unknown_size(
                    Arc::clone(&fetch_storage),
                    storage_uri.clone(),
                ));
            let block_source = BlockCachedSource::new_with_accounting(
                range_src,
                Arc::downgrade(self),
                *uri,
                self.blocks_path(uri),
                !allow_background_fill,
                // No manifest hints here, so the FTS subsection is unknown.
                None,
            );
            block_source_arc = Arc::clone(&block_source);
            let source: Arc<dyn LazyByteSource> = block_source;
            let lazy_reader = SuperfileReader::open_lazy_with(
                Arc::clone(&source),
                OpenOptions { verify_crc: false },
            )
            .await?;
            let size = source.size();
            (lazy_reader, size)
        };

        // A Stream open charges only the blocks it fills (SourceOwned), so a fanout over many
        // superfiles does not reserve them all in full and evict live peers. A Warm open will
        // download the whole file, so it reserves the full size now.
        if allow_background_fill {
            self.reserve_manual(size).await?;
        }

        let lazy_reader = Arc::new(lazy_reader);
        let (size_bytes, accounting) = if allow_background_fill {
            (Arc::new(AtomicU64::new(size)), EntryAccounting::Eager)
        } else {
            (
                block_source_arc.filled_bytes_handle(),
                EntryAccounting::SourceOwned,
            )
        };
        let entry = Arc::new(CachedEntry {
            reader: Arc::clone(&lazy_reader),
            // Unlatched: a later Warm read starts the fill, see `maybe_spawn_background_fill`.
            residency: Residency::Paged {
                block_source: block_source_arc,
                fill_spawned: AtomicBool::new(false),
            },
            size_bytes,
            accounting,
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        // Admission may return an existing whole file instead; serve what it returns.
        Ok(self.admit_entry(*uri, entry))
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

        // Promote to final path + open as mmap.
        tokio::fs::rename(&tmp, &final_path).await?;
        let (mmap, bytes) = mmap_readonly_with_handle(&final_path).map_err(DiskCacheError::Io)?;
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: self.config.verify_crc_on_open,
            },
        )?;
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
        let (mmap, bytes) = mmap_readonly_with_handle(&tmp_path)?;
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: store.config.verify_crc_on_open,
            },
        )?;
        // Swap in over the buffered entry, which already carries the whole-size charge. The
        // finalizer reserved nothing of its own.
        let entry = store.build_mmap_entry(Arc::new(reader), mmap, size, None);
        store.install_promoted_entry(uri, entry, &tmp_path, &final_path, &owner, None);
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

/// How long a background fill keeps yielding to foreground queries holding
/// the same superfile's lazy reader.
///
/// Yielding keeps the fill from competing for I/O with the query that
/// triggered it. Left unbounded, a superfile under continuous load never
/// goes idle, so the fill never runs and the reader stays lazy for the life
/// of the process — holding its term dictionary and doc-lengths tail on the
/// heap, and serving reads as per-block range fetches instead of slices of
/// an mmap. Once the window closes the fill proceeds alongside the running
/// query: it writes a temp file and swaps the cache entry, so an in-flight
/// query keeps reading through the `Arc` it already holds.
#[derive(Clone, Copy, Debug)]
enum PromotionDefer {
    /// Yield for as long as the reader stays busy.
    Forever,
    /// Yield until this instant, then fill regardless.
    Until(Instant),
}

impl PromotionDefer {
    /// Open a window of `timeout` starting now. A `timeout` that overflows
    /// the clock (notably [`Duration::MAX`]) yields indefinitely.
    fn start(timeout: Duration) -> Self {
        match Instant::now().checked_add(timeout) {
            Some(deadline) => Self::Until(deadline),
            None => Self::Forever,
        }
    }

    /// Whether the window is still open.
    fn open(self) -> bool {
        match self {
            Self::Forever => true,
            Self::Until(deadline) => Instant::now() < deadline,
        }
    }

    /// Whether the fill should still yield to `reader` being held by a
    /// caller other than the cache entry.
    fn yields_to(self, reader: &Weak<SuperfileReader>) -> bool {
        self.open() && reader_blocks_background_fill(reader)
    }
}

async fn wait_for_lazy_foreground_release(
    store: &Weak<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
    defer: PromotionDefer,
) -> Option<Arc<DiskCacheStore>> {
    loop {
        if store.strong_count() == 0 || reader.strong_count() == 0 {
            return None;
        }
        if let Some(strong) = store.upgrade()
            && strong.n_promotion_waiters.load(Ordering::Acquire) > 0
        {
            return Some(strong);
        }
        if !defer.open() {
            // Window closed: fill alongside the running query rather than
            // leave this superfile lazy for the life of the process.
            return store.upgrade();
        }
        if reader.strong_count() <= 1 {
            // `strong_count == 1` also occurs briefly while a caller is
            // acquiring the cache entry, so re-check after one scheduler turn.
            tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
            if reader.strong_count() <= 1 {
                return store.upgrade();
            }
            continue;
        }
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
    }
}

/// Wait until this URI's lazy reader is held only by the cache entry.
/// Unrelated table/URI fills are not gated here — only this reader's
/// strong-count. A grace re-check covers the open→query handoff.
async fn wait_for_reader_quiescence(
    store: &Arc<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
    defer: PromotionDefer,
) -> bool {
    loop {
        while defer.yields_to(reader) {
            if background_store_abandoned(store) {
                return false;
            }
            tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
        }
        if reader.strong_count() == 0 {
            return false;
        }
        tokio::time::sleep(STORE_UPGRADE_RETRY_INTERVAL).await;
        if reader.strong_count() == 0 {
            return false;
        }
        if !defer.yields_to(reader) {
            return !background_store_abandoned(store);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackgroundFillOutcome {
    Complete,
    Paused,
    Abandoned,
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
    reader: &Weak<SuperfileReader>,
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
            if background_store_abandoned(store) {
                return Ok(BackgroundFillOutcome::Abandoned);
            }
            if reader.strong_count() == 0 {
                return Ok(BackgroundFillOutcome::Abandoned);
            }
            if defer.yields_to(reader) {
                return Ok(BackgroundFillOutcome::Paused);
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
            let uri = storage_uri.to_string();
            // Fill GETs are tagged background (see `fetch_and_pwrite`) so query-window meters
            // attribute only foreground lazy/probe GETs to the cold query cost.
            in_flight.push(async move {
                for (range_start, range_end) in fetch_ranges {
                    fetch_and_pwrite(&storage, &uri, &file, range_start, range_end, true).await?;
                }
                Ok::<u64, DiskCacheError>(chunk_idx)
            });
            next_chunk += 1;
        }

        let foreground = foreground_notify().notified();
        tokio::pin!(foreground);
        let _ = foreground.as_mut().enable();
        if reader.strong_count() == 0 {
            return Ok(BackgroundFillOutcome::Abandoned);
        }
        if defer.yields_to(reader) {
            return Ok(BackgroundFillOutcome::Paused);
        }
        tokio::select! {
            biased;
            _ = &mut foreground => {
                // A query started: re-check same-URI hold. Unrelated fills
                // (strong_count == 1) fall through and keep downloading.
                if reader.strong_count() == 0 {
                    return Ok(BackgroundFillOutcome::Abandoned);
                }
                if defer.yields_to(reader) {
                    return Ok(BackgroundFillOutcome::Paused);
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

    if background_store_abandoned(store) {
        return Ok(BackgroundFillOutcome::Abandoned);
    }
    if reader.strong_count() == 0 {
        return Ok(BackgroundFillOutcome::Abandoned);
    }
    if defer.yields_to(reader) {
        return Ok(BackgroundFillOutcome::Paused);
    }
    spawn_blocking(carry_span(move || file.sync_all()))
        .await
        .map_err(|error| DiskCacheError::SuperfileOpen(format!("fsync join: {error}")))??;
    Ok(BackgroundFillOutcome::Complete)
}

/// Is `entry` the one this fill started from? Same reader, by pointer.
fn is_mine(entry: &CachedEntry, owner: &Weak<SuperfileReader>) -> bool {
    owner
        .upgrade()
        .is_some_and(|mine| Arc::ptr_eq(&entry.reader, &mine))
}

/// A finished fill may replace its own entry or a lazy one, never someone else's whole file.
fn fill_may_replace(entry: &CachedEntry, owner: &Weak<SuperfileReader>) -> bool {
    is_mine(entry, owner) || !entry.has_whole_file()
}

/// Undo a fill that did not finish: drop the entry it started from if still there, release the
/// fill's own reservation, delete the partial download. Leaves the coordinator alone: it belongs
/// to whatever fetch is in flight, not to the fill.
fn rollback_lazy_background_fill(
    store: &Arc<DiskCacheStore>,
    uri: &SuperfileUri,
    tmp: &Path,
    owner: &Weak<SuperfileReader>,
    own_reservation: Option<u64>,
) {
    if let Some((_, mine)) = store
        .cached
        .remove_if(uri, |_, entry| is_mine(entry, owner))
    {
        store.release_entry_accounting(&mine);
    }
    if let Some(bytes) = own_reservation {
        store.release_block_bytes(bytes);
    }
    let _ = fs::remove_file(tmp);
}

/// Diagnostic gate for measuring lazy foreground reads without promotion,
/// from `diagnostics.disable_background_fill` (YAML-only; no env override).
pub(crate) fn skip_background_fill() -> bool {
    global_config().diagnostics.disable_background_fill
}

/// Download the whole file for a lazy entry and install it as a mapped one. With `skip_vec`, the
/// vector blob is left out of the file and read through the block cache instead (see
/// [`HoleFallbackSource`]).
async fn lazy_background_fill(
    store: Weak<DiskCacheStore>,
    reader: Weak<SuperfileReader>,
    uri: SuperfileUri,
    storage_uri: String,
    size: u64,
    mut own_reservation: Option<u64>,
    fetch_storage: Arc<dyn StorageProvider>,
    skip_vec: Option<(u64, u64)>,
    defer: PromotionDefer,
) -> Result<(), DiskCacheError> {
    let Some(store) = wait_for_lazy_foreground_release(&store, &reader, defer).await else {
        return Ok(());
    };
    let tmp = store.tmp_path(&uri);
    let final_path = store.cache_path(&uri);

    if background_store_abandoned(&store) {
        rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);
        return Ok(());
    }

    let _prefetch_permit = match Arc::clone(&store.prefetch_semaphore).acquire_owned().await {
        Ok(permit) => permit,
        Err(error) => {
            rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);
            return Err(DiskCacheError::SuperfileOpen(format!(
                "prefetch semaphore closed: {error}"
            )));
        }
    };
    // Resume cursor: chunks durably written so far, preserved across
    // pause/resume so a same-URI reader interrupting the fill costs only
    // the unfinished chunks rather than a re-download of the whole object.
    let mut filled: Vec<bool> = Vec::new();
    loop {
        if !wait_for_reader_quiescence(&store, &reader, defer).await {
            rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);

            return Ok(());
        }
        // Evicted, or someone else's whole file is there: the download would be thrown away.
        if !store.fill_can_install(&uri, &reader) {
            rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);
            return Ok(());
        }
        match cold_fetch_to_disk_cancelable(
            &store,
            &reader,
            &fetch_storage,
            &storage_uri,
            &tmp,
            size,
            &mut filled,
            skip_vec,
            defer,
        )
        .await?
        {
            BackgroundFillOutcome::Complete => break,
            // Keep the partial `tmp` and the `filled` cursor: the next attempt
            // resumes from the first unwritten chunk.
            BackgroundFillOutcome::Paused => {}
            BackgroundFillOutcome::Abandoned => {
                rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);
                return Ok(());
            }
        }
    }

    let result: Result<(), DiskCacheError> = async {
        if background_store_abandoned(&store) {
            return Ok(());
        }

        // The slot changed during the download. Skip the promotion: its vector-header reads would
        // go uncached to object storage, only for the install to yield.
        if !store.fill_can_install(&uri, &reader) {
            rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation.take());
            return Ok(());
        }

        // Mmap the tempfile itself; the install renames it into the cache only if it wins the slot.
        let (mmap_arc, bytes) = mmap_readonly_with_handle(&tmp)?;

        let (promoted_reader, vector_source) = match skip_vec {
            Some((hole_start, hole_len)) => {
                // Keep the live block cache, so the vector ranges the cold query read stay local.
                // None only if an eviction raced the check above: start a fresh one for the hole.
                let block_source = store
                    .cached
                    .get(&uri)
                    .and_then(|entry| entry.block_source().cloned())
                    .unwrap_or_else(|| {
                        let remote: Arc<dyn LazyByteSource> =
                            Arc::new(StorageRangeSource::with_known_size(
                                Arc::clone(&fetch_storage),
                                storage_uri.clone(),
                                size,
                            ));
                        // Serves only the vector hole; FTS bytes come from the mmap.
                        BlockCachedSource::new_pre_reserved(
                            remote,
                            Arc::downgrade(&store),
                            uri,
                            store.blocks_path(&uri),
                            None,
                        )
                    });
                let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
                    local: Arc::new(BytesLazyByteSource::new(bytes.clone())),
                    hole_start,
                    hole_len,
                    fallback: Arc::clone(&block_source),
                });
                let mut reader =
                    SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false })
                        .await?;
                // Sync parquet decodes (take, id scans) run off the mmap.
                reader.install_resident_parquet(bytes)?;
                (reader, Some(block_source))
            }
            None => {
                let reader = SuperfileReader::open_with(
                    bytes,
                    OpenOptions {
                        verify_crc: store.config.verify_crc_on_open,
                    },
                )?;
                (reader, None)
            }
        };

        let block_source_retained = vector_source.is_some();
        // The hole is not on disk. When its block source charges its own blocks, the mmap is
        // charged only for what the file holds; otherwise the hole's blocks ride on this charge.
        let charge = match (&vector_source, skip_vec) {
            (Some(source), Some((_, vec_len))) if source.owns_accounting() => {
                size.saturating_sub(vec_len)
            }
            _ => size,
        };
        let entry =
            store.build_mmap_entry(Arc::new(promoted_reader), mmap_arc, charge, vector_source);
        // Installed with no vector hole, the mmap serves every range and the block file is dead
        // weight. Not installed, the block file stays with whoever holds the slot.
        let installed =
            store.install_promoted_entry(uri, entry, &tmp, &final_path, &reader, own_reservation);
        // The install settled the fill's reservation either way: it backs the new entry, or it
        // was released.
        own_reservation = None;
        if installed.is_some() && !block_source_retained {
            store.drop_block_file(&uri);
        }
        Ok(())
    }
    .await;

    if result.is_err() || background_store_abandoned(&store) {
        // `own_reservation` is already `None` once the install ran, so this releases nothing twice.
        rollback_lazy_background_fill(&store, &uri, &tmp, &reader, own_reservation);
    }
    result
}

/// Absolute `(offset, length)` of the vector blob from Parquet KV metadata.
fn vector_blob_range(reader: &SuperfileReader) -> Option<(u64, u64)> {
    let kv_map = footer::extract_kv_map(reader.parquet_metadata()).ok()?;
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
        path::PathBuf,
        sync::{Arc, atomic::Ordering},
        time::Duration,
    };

    use bytes::Bytes;
    use tempfile::TempDir;
    use tokio::{spawn, task::yield_now, time::timeout};

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::reader::{OpenOptions, SuperfileReader},
        supertable::{
            manifest::{SubsectionOffsets, SuperfileUri},
            reader_cache::{
                block_source::BlockCachedSource,
                config::{ColdFetchMode, DiskCacheConfig},
                disk::{fetch::*, test_support::*},
            },
        },
    };

    /// Poll cadence while a test waits for a background fill to finish.
    const FILL_POLL_INTERVAL: Duration = Duration::from_millis(10);

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
        let permit = Arc::clone(&store.prefetch_semaphore)
            .acquire_owned()
            .await
            .expect("fill permit");
        let held = store
            .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
            .await
            .expect("warm open starts the fill");
        let hole = vector_blob_range(&held).expect("vector blob");
        // Parked at the permit, the fill task holds the store alive.
        timeout(PROMOTE_TIMEOUT, async {
            while Arc::strong_count(&store) == 1 {
                tokio::time::sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("the fill starts and parks");
        (dir, store, recording, uri, held, permit, hole)
    }

    /// Wait for the parked fill to run and exit. It holds the store alive while it runs.
    async fn wait_for_the_fill_to_exit(store: &Arc<DiskCacheStore>) {
        timeout(PROMOTE_TIMEOUT, async {
            while Arc::strong_count(store) > 1 {
                tokio::time::sleep(FILL_POLL_INTERVAL).await;
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
        timeout(PROMOTE_TIMEOUT, async {
            while !recording.is_paused() {
                tokio::time::sleep(FILL_POLL_INTERVAL).await;
            }
        })
        .await
        .expect("the download starts");
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

    /// How much smaller than the file the surplus test's promoted charge is: any amount works, it
    /// only has to leave a surplus in the fill's full-size reservation.
    const SURPLUS_TEST_BYTES: u64 = 100;

    /// `rollback_lazy_background_fill` drops the entry the fill was promoting and deletes its
    /// partial tempfile.
    #[tokio::test]
    async fn rollback_lazy_background_fill_evicts_entry_and_tmp() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();

        // Seed a cache entry the way a lazy fill would, plus a leftover tmp
        // scratch file for the partial download.
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        let owner = Arc::downgrade(&store.cached.get(&uri).expect("cached").reader);
        assert!(
            store.is_cached(&uri),
            "entry must be cached before rollback"
        );
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, b"partial-download-bytes").expect("seed tmp scratch file");
        assert!(tmp.exists(), "tmp scratch file must exist before rollback");

        rollback_lazy_background_fill(&store, &uri, &tmp, &owner, None);

        assert!(
            !store.is_cached(&uri),
            "cached entry must be gone after rollback"
        );
        assert!(
            !tmp.exists(),
            "tmp scratch file must be deleted after rollback"
        );
        store.assert_budget_consistent();
    }

    /// Rollback undoes only the entry the fill was promoting. A whole file someone else admitted
    /// in the meantime is not the fill's to remove, and its charge is not the fill's to release.
    #[tokio::test]
    async fn rollback_leaves_an_entry_the_fill_did_not_create() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        seed_cache_file(&store, &uri, &tiny_superfile_bytes());
        let stranger = store
            .fetch_from_disk_cache(&uri, None)
            .await
            .expect("disk probe")
            .expect("whole file admitted");
        let charged = store.stats().current_bytes;
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, b"partial-download-bytes").expect("seed tmp scratch file");

        // A fill spawned for some long-gone lazy entry gives up.
        let not_mine: Weak<SuperfileReader> = Weak::new();
        rollback_lazy_background_fill(&store, &uri, &tmp, &not_mine, None);

        assert!(
            Arc::ptr_eq(
                &store.cached.get(&uri).expect("still cached").reader,
                &stranger.reader
            ),
            "the stranger stays"
        );
        assert_eq!(
            store.stats().current_bytes,
            charged,
            "its charge is untouched"
        );
        assert!(
            !tmp.exists(),
            "the fill's own scratch file is still cleaned up"
        );
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

    /// Evicted mid-fill: the slot is empty, so nothing is installed, the tempfile goes, nothing
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
        let installed =
            store.install_promoted_entry(uri, entry, &tmp, &final_path, &Weak::new(), None);

        assert!(installed.is_none(), "vacant slot must not reinstate");
        assert!(!tmp.exists(), "the tempfile is deleted");
        assert!(!final_path.exists(), "nothing lands under the cache name");
        assert_eq!(store.current_bytes.load(Ordering::Acquire), before);
        assert_eq!(store.stats().n_entries, 0);
        store.assert_budget_consistent();
    }

    /// A fill that reserved for itself (the Stream-then-Warm shape) and finds its slot evicted must
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
        let installed = store.install_promoted_entry(
            uri,
            entry,
            &tmp,
            &store.cache_path(&uri),
            &Weak::new(),
            Some(size),
        );

        assert!(installed.is_none(), "vacant slot must not reinstate");
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
            .fetch_from_disk_cache(&uri, None)
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
            &Weak::new(),
            Some(size),
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

    /// A finished local copy beats range reads: a fill installs over a lazy entry it did not start
    /// from (its own was evicted and a query reopened the file), inheriting that entry's full
    /// charge, and only then moves its file under the cache name.
    #[tokio::test]
    async fn fill_beats_a_lazy_entry_it_did_not_start_from() {
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
        let installed =
            store.install_promoted_entry(uri, promoted, &tmp, &final_path, &Weak::new(), None);

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
    /// fill tops up and installs; without room it yields, leaving no file behind.
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
            let installed =
                store.install_promoted_entry(uri, promoted, &tmp, &final_path, &Weak::new(), None);

            assert_eq!(installed.is_some(), expect_installed, "budget {budget}");
            assert_eq!(store.is_mmap_promoted(&uri), expect_installed);
            assert_eq!(final_path.exists(), expect_installed);
            assert!(!tmp.exists(), "the tempfile never outlives the install");
            store.assert_budget_consistent();
        }
    }

    /// The whole race, end to end: a Warm fill runs while a query holds the reader, the entry is
    /// evicted, a query reopens the file lazily, and the fill finishes. The fill must win the slot,
    /// leave a complete setup (file under the cache name, no stray tempfiles), keep the ledger
    /// honest, and the next read must not fetch or throw anything away.
    #[tokio::test]
    async fn fill_that_outlives_an_eviction_still_promotes() {
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
            let permit = Arc::clone(&store.prefetch_semaphore)
                .acquire_owned()
                .await
                .expect("fill permit");

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

            assert!(
                poll_until_mmap_promoted(&store, &uri, PROMOTE_TIMEOUT).await,
                "{reopen:?}: the finished fill wins the slot"
            );
            drop(held);
            drop(reopened);
            store
                .wait_until_fills_settled(PROMOTE_TIMEOUT)
                .await
                .expect("fills settle");

            assert!(
                store.cache_path(&uri).exists(),
                "{reopen:?}: file in the cache"
            );
            assert_eq!(
                leftover_tempfiles(&store),
                0,
                "{reopen:?}: no stray tempfiles"
            );
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
            assert!(
                store.cache_path(&uri).exists() && store.blocks_path(&uri).exists(),
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
            &Weak::new(),
            Some(size),
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
        let installed =
            store.install_promoted_entry(uri, promoted, &tmp, &final_path, &Weak::new(), None);

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
    /// file will hold, for the whole download, not the file plus a hole it never writes.
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
        let charged_by_blocks = store.stats().current_bytes;

        // Park the fill right after it reserves, before its download.
        let permit = Arc::clone(&store.prefetch_semaphore)
            .acquire_owned()
            .await
            .expect("fill permit");
        drop(
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, ReadIntent::Warm)
                .await
                .expect("warm open starts the fill"),
        );
        let reserved = timeout(PROMOTE_TIMEOUT, async {
            loop {
                let now = store.stats().current_bytes;
                if now > charged_by_blocks {
                    return now - charged_by_blocks;
                }
                yield_now().await;
            }
        })
        .await
        .expect("the fill reserves");
        drop(permit);

        assert_eq!(reserved, size - vec_len, "the hole is never reserved");
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

    /// The window is a deadline, and `Duration::MAX` is the "never stop
    /// yielding" sentinel rather than an overflow panic.
    #[test]
    fn promotion_defer_window_opens_and_closes() {
        assert!(
            !PromotionDefer::start(Duration::ZERO).open(),
            "a zero window is closed on arrival"
        );
        assert!(
            PromotionDefer::start(Duration::from_secs(3600)).open(),
            "a long window is open"
        );
        assert!(
            matches!(
                PromotionDefer::start(Duration::MAX),
                PromotionDefer::Forever
            ),
            "Duration::MAX overflows the clock and means unbounded yielding"
        );
        assert!(PromotionDefer::Forever.open());
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
            DiskCacheStore::uncovered_ranges(&blob, &wanted),
            vec![(90, 20), (140, 20), (2_000, 5)],
            "whole-inside ranges are covered; straddling, outside and only empty ranges are not"
        );
        assert!(DiskCacheStore::uncovered_ranges(&[], &[]).is_empty());
        assert_eq!(
            DiskCacheStore::uncovered_ranges(&[], &[(0, 1)]),
            vec![(0, 1)]
        );
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
        tokio::time::sleep(FOREGROUND_GUARD_HOLD).await;
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

    #[tokio::test]
    async fn background_fill_waits_for_same_uri_reader() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        let reader = store.reader(&uri).await.expect("lazy cold");
        let _foreground = ForegroundQueryGuard::enter();
        tokio::time::sleep(FOREGROUND_GUARD_HOLD).await;
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
            None,
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
        let signal_reader = Arc::new(
            SuperfileReader::open(tiny_superfile_bytes()).expect("foreground signal reader"),
        );
        let signal_weak = Arc::downgrade(&signal_reader);
        let fill = spawn(async move {
            let mut filled = Vec::new();
            let outcome = cold_fetch_to_disk_cancelable(
                &fill_store,
                &signal_weak,
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
        // Holding the signal reader (strong_count > 1) is the per-URI pause.
        let foreground = Arc::clone(&signal_reader);
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
}
