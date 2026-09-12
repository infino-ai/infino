// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Cold fetch: pull a superfile from object storage into the cache. The three
//! fetch shapes (whole, hybrid, lazy), the byte pumps that download it, and the
//! finalize step that lands it on disk as an mmap-backed entry.

use std::{
    fs,
    io::SeekFrom,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
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
    sync::{OnceCell, oneshot},
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
            disk::{sources::mmap_readonly_with_handle, *},
        },
    },
    utils::trace::{OpOrigin, detached},
};

impl DiskCacheStore {
    /// mmap a cache file and open it as a [`SuperfileReader`], building the
    /// `CachedEntry`. Shared by the warm-insert path and the open-time index
    /// rebuild ([`Self::restore_from_cache_root`]); the caller owns budget
    /// accounting and the `cached`-map insert. The reader's bytes and
    /// `CachedEntry.mmap` share one `Arc<Mmap>` so a later `MADV_DONTNEED`
    /// sweep touches the same mapping.
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

    /// Build a promoted [`Residency::Mapped`] entry: the single place the mmap shape is written,
    /// always `Eager` (the whole superfile size is store-reserved). `vector_source` is `Some` only
    /// when the vector blob was left out of the mmap and still comes from the block cache.
    pub(crate) fn build_mmap_entry(
        &self,
        reader: Arc<SuperfileReader>,
        mmap: Arc<Mmap>,
        size: u64,
        vector_source: Option<Arc<BlockCachedSource>>,
    ) -> Arc<CachedEntry> {
        Arc::new(CachedEntry {
            reader,
            residency: Residency::Mapped {
                mmap,
                vector_source,
            },
            size_bytes: Arc::new(AtomicU64::new(size)),
            accounting: EntryAccounting::Eager,
            last_access_us: AtomicU64::new(self.now_us()),
        })
    }

    /// Install a promoted entry, honoring the reinstate-only-if-present gate that keeps the budget
    /// balanced under a racing eviction.
    ///
    /// `Fresh` inserts unconditionally: the foreground cold-fetch owns the slot it just reserved.
    /// `ReplaceIfPresent` is for a background finalizer that may have been evicted while it ran: it
    /// replaces an occupied slot, but on a vacant slot it drops the just-renamed file and returns
    /// `None` (eviction already released the reservation, so reinstating the entry would leak it).
    pub(crate) fn install_promoted_entry(
        &self,
        uri: SuperfileUri,
        entry: Arc<CachedEntry>,
        final_path: &Path,
        mode: InstallMode,
    ) -> Option<Arc<CachedEntry>> {
        match mode {
            InstallMode::Fresh => {
                self.cached.insert(uri, Arc::clone(&entry));
                Some(entry)
            }
            InstallMode::ReplaceIfPresent => match self.cached.entry(uri) {
                Entry::Occupied(mut occ) => {
                    *occ.get_mut() = Arc::clone(&entry);
                    Some(entry)
                }
                Entry::Vacant(_) => {
                    let _ = fs::remove_file(final_path);
                    None
                }
            },
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

        // A finished cache file may already be on disk; use it before fetching.
        if let Some(entry) = self.try_reuse_cached_file(uri, None).await? {
            return Ok(entry);
        }

        let head = fetch_storage.head(&storage_uri).await?;
        let size = head.size;
        // Don't use the borrow-lifetimed Reservation guard
        // because it would tie the future to `&self` and block
        // the `tokio::spawn` of the background finalizer. We
        // reserve manually here; the background task either
        // commits (cache filled) or rolls back via fetch_sub.
        self.reserve_manual(size).await?;
        let reserved_bytes = size;
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
        // Register entry in the cache so subsequent reader()
        // calls hit cache rather than re-entering the
        // coordinator.
        self.cached.insert(*uri, Arc::clone(&entry));

        // 5. Spawn the background finalizer: wait for pwrites,
        //    fsync, rename, mmap, and atomically replace the
        //    cached entry with a mmap-backed reader. On error,
        //    release the manual reservation back to the pool.
        let store = Arc::clone(self);
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
                    reserved_bytes,
                )
                .await;
            }
            .instrument(finalize_span),
        );

        Ok(entry)
    }

    /// lazy-foreground cold-fetch coordinator.
    /// Returns immediately with a
    /// [`SuperfileReader::open_lazy`]-built reader over a
    /// [`crate::supertable::StorageRangeSource`]; spawns a
    /// background task that waits for foreground lazy readers
    /// to release before fetching the full superfile, mmap'ing
    /// it, and replacing the cached entry. Subsequent
    /// `reader(uri)` calls return the mmap-backed reader (zero
    /// S3 GETs for any subsequent search).
    /// lazy cold-fetch coordinator. When `offsets` is `Some`,
    /// the cold open uses manifest-provided size/open-batch hints;
    /// when `None`, it falls back to unknown-size suffix-tail
    /// discovery.
    pub(crate) async fn reader_lazy_with_bg_fill_hinted(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        offsets: Option<&SubsectionOffsets>,
        storage: Option<&Arc<dyn StorageProvider>>,
        allow_background_fill: bool,
    ) -> Result<Arc<SuperfileReader>, DiskCacheError> {
        if let Some(entry) = self.cached.get(uri) {
            entry.last_access_us.store(self.now_us(), Ordering::Release);
            if allow_background_fill {
                self.maybe_spawn_background_fill(uri, storage_key, &entry, storage);
            }
            return Ok(Arc::clone(&entry.reader));
        }
        let cell = self
            .coordinators
            .entry(*uri)
            .or_insert_with(|| Arc::new(OnceCell::new()))
            .clone();
        let result = cell
            .get_or_init(|| async {
                let fetch_storage = self.resolve_storage(storage);
                self.cold_fetch_lazy(
                    uri,
                    storage_key,
                    offsets,
                    fetch_storage,
                    allow_background_fill,
                )
                .await
            })
            .await;
        let fetch_storage = self.resolve_storage(storage);
        match result {
            Ok(entry) => {
                if allow_background_fill {
                    self.maybe_spawn_background_fill(uri, storage_key, entry, storage);
                }
                Ok(Arc::clone(&entry.reader))
            }
            Err(_e) => {
                self.coordinators.remove(uri);
                match self
                    .cold_fetch_lazy(
                        uri,
                        storage_key,
                        offsets,
                        fetch_storage,
                        allow_background_fill,
                    )
                    .await
                {
                    Ok(entry) => {
                        if allow_background_fill {
                            self.maybe_spawn_background_fill(uri, storage_key, &entry, storage);
                        }
                        Ok(Arc::clone(&entry.reader))
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Start parquet/FTS background fill once per URI when an FTS/SQL open
    /// asks for it. Vector opens never call this — they keep block-cache
    /// retention only. Fill skips the vector blob range.
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
        // A source-owned (vector-opened) entry accounts its live filled bytes,
        // so `size_bytes` is NOT the object size. Take the full size from the
        // block source, and reserve it here — the vector open never did, so the
        // promotion's mmap needs budget reserved before it downloads.
        let size = entry
            .block_source()
            .map(|bs| bs.size())
            .filter(|&s| s > 0)
            .unwrap_or_else(|| entry.size_bytes.load(Ordering::Acquire));
        let needs_reserve = matches!(entry.accounting, EntryAccounting::SourceOwned);
        let skip_vec = vector_blob_range(&entry.reader);
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
                    if s.reserve_manual(size).await.is_err() {
                        return;
                    }
                }
                let _ = lazy_background_fill(
                    store,
                    reader,
                    uri_owned,
                    storage_uri_owned,
                    size,
                    size,
                    fetch_storage,
                    skip_vec,
                )
                .await;
            }
            .instrument(fill_span),
        );
    }

    /// Lazy cold-fetch path. Foreground builds a reader via
    /// `SuperfileReader::open_lazy_with(StorageRangeSource)`;
    /// background task waits for foreground lazy readers to release,
    /// then downloads the full superfile to NVMe, mmaps it, and replaces
    /// the cache entry.
    ///
    /// If `offsets` is present, the lazy source starts with a known
    /// superfile size and an optional open-batch overlay:
    ///   - with `open_blob`: zero superfile-object GETs at open time,
    ///     because manifest-part fetch already carried the bytes.
    ///   - without `open_blob`: parquet tail + vector + FTS open ranges
    ///     are fetched in one parallel batch.
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

        // A finished cache file may already be on disk; use it before fetching.
        if let Some(entry) = self
            .try_reuse_cached_file(uri, offsets.map(|o| o.total_size))
            .await?
        {
            return Ok(entry);
        }

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

            if !offsets.open_blob.is_empty() {
                // The open-batch bytes (parquet tail + vector + FTS open
                // ranges) already rode in with the manifest part GET that
                // `cold_open` performed. Install them straight into the
                // overlay: ZERO open-time GETs against the superfile object.
                for (off, bytes) in &offsets.open_blob {
                    overlay.install(*off, Bytes::copy_from_slice(bytes));
                }
            } else {
                // Fallback when no captured open blob is present:
                // fetch the open batch over the wire
                // (parquet tail + vec + fts ranges in parallel, 1 RTT).
                let storage_for_parquet = Arc::clone(&fetch_storage);
                let storage_for_vec = Arc::clone(&fetch_storage);
                let storage_for_fts = Arc::clone(&fetch_storage);
                let parquet_uri = storage_uri.clone();
                let vec_uri = storage_uri.clone();
                let fts_uri = storage_uri.clone();

                let parquet_fut = async move {
                    let end = total_size;
                    let start = parquet_tail_start;
                    if end == start {
                        return Ok::<_, StorageError>(Bytes::new());
                    }
                    storage_for_parquet
                        .get_range(&parquet_uri, start..end)
                        .await
                };
                let vec_fut =
                    async move { fetch_hint_ranges(storage_for_vec, vec_uri, vec_ranges).await };
                let fts_fut =
                    async move { fetch_hint_ranges(storage_for_fts, fts_uri, fts_ranges).await };

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

        // Vector opens keep their blob sparse and never mmap-promote, so they
        // account their live filled bytes (source-owned) instead of reserving
        // the whole superfile up front. Otherwise a fanout touching many
        // superfiles over-reserves and evicts live peers. FTS/SQL opens promote
        // to a full mmap, so they keep the eager full-size reservation.
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
            // Fill is modality-gated via [`Self::maybe_spawn_background_fill`] after the open
            // returns, so it starts false here and vector opens never flip it.
            residency: Residency::Paged {
                block_source: block_source_arc,
                fill_spawned: AtomicBool::new(false),
            },
            size_bytes,
            accounting,
            last_access_us: AtomicU64::new(self.now_us()),
        });
        self.n_cold_fetches.fetch_add(1, Ordering::AcqRel);
        self.cached.insert(*uri, Arc::clone(&entry));

        Ok(entry)
    }

    /// Run the cold-fetch coordinator for `uri`. Reserves
    /// budget, fetches, mmap's, registers in `cached`.
    pub(crate) async fn cold_fetch(
        &self,
        uri: &SuperfileUri,
        storage_key: &str,
        fetch_storage: Arc<dyn StorageProvider>,
    ) -> Result<Arc<CachedEntry>, DiskCacheError> {
        let storage_uri = storage_key.to_owned();

        // A finished cache file may already be on disk; use it before fetching.
        if let Some(entry) = self.try_reuse_cached_file(uri, None).await? {
            return Ok(entry);
        }

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
        self.install_promoted_entry(*uri, Arc::clone(&entry), &final_path, InstallMode::Fresh);
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

/// Background finalizer for the hybrid cold-fetch. Awaits
/// all pwrites, fsyncs + renames the destination file, mmaps
/// it, and atomically replaces the cache entry with a
/// mmap-backed reader. On failure, releases the disk
/// reservation back to the pool and removes the entry.
async fn finalize_to_mmap(
    store: Arc<DiskCacheStore>,
    uri: SuperfileUri,
    tmp_path: PathBuf,
    final_path: PathBuf,
    file: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    pwrite_handles: Vec<oneshot::Receiver<JoinHandle<Result<(), DiskCacheError>>>>,
    size: u64,
    reserved_bytes: u64,
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
        tokio::fs::rename(&tmp_path, &final_path).await?;
        let (mmap, bytes) = mmap_readonly_with_handle(&final_path)?;
        let reader = SuperfileReader::open_with(
            bytes,
            OpenOptions {
                verify_crc: store.config.verify_crc_on_open,
            },
        )?;
        // Replace the in-memory-backed entry with the mmap-backed one. `ReplaceIfPresent` drops
        // the file instead of reinstating when a racing reservation evicted the slot mid-finalize
        // (reinstating an evicted, already-released entry would leak its reservation).
        let entry = store.build_mmap_entry(Arc::new(reader), mmap, size, None);
        store.install_promoted_entry(uri, entry, &final_path, InstallMode::ReplaceIfPresent);
        store.coordinators.remove(&uri);
        Ok::<(), DiskCacheError>(())
    }
    .await;
    if res.is_err() {
        // Rollback. Use the same atomic gate as eviction
        // (`cached.remove(uri).is_some()`) so we don't double-
        // decrement when a racing eviction already removed
        // this entry + released its bytes.
        if let Some((_, entry)) = store.cached.remove(&uri) {
            store.release_entry_accounting(&entry);
        }
        store.coordinators.remove(&uri);
    }
    // `reserved_bytes` parameter is retained for future use
    // (e.g., observability counters); the bytes accounting is
    // entirely driven by `cached.remove` gating now.
    let _ = reserved_bytes;
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

async fn wait_for_lazy_foreground_release(
    store: &Weak<DiskCacheStore>,
    reader: &Weak<SuperfileReader>,
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
) -> bool {
    loop {
        while reader_blocks_background_fill(reader) {
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
        if !reader_blocks_background_fill(reader) {
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
            if reader_blocks_background_fill(reader) {
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
        if reader_blocks_background_fill(reader) {
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
                if reader_blocks_background_fill(reader) {
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
    if reader_blocks_background_fill(reader) {
        return Ok(BackgroundFillOutcome::Paused);
    }
    spawn_blocking(carry_span(move || file.sync_all()))
        .await
        .map_err(|error| DiskCacheError::SuperfileOpen(format!("fsync join: {error}")))??;
    Ok(BackgroundFillOutcome::Complete)
}

fn rollback_lazy_background_fill(store: &Arc<DiskCacheStore>, uri: &SuperfileUri, tmp: &Path) {
    if let Some((_, entry)) = store.cached.remove(uri) {
        store.release_entry_accounting(&entry);
    }
    store.coordinators.remove(uri);
    let _ = fs::remove_file(tmp);
}

/// Diagnostic gate for measuring lazy foreground reads without promotion,
/// from `diagnostics.disable_background_fill` (YAML-only; no env override).
pub(crate) fn skip_background_fill() -> bool {
    global_config().diagnostics.disable_background_fill
}

/// Promote one released lazy reader to an mmap-backed cache entry.
///
/// When `skip_vec` is set, the fill file leaves the vector blob sparse and
/// promotion opens a hybrid reader: mmap for parquet/FTS, the preserved
/// block-cache source for vector ranges.
async fn lazy_background_fill(
    store: Weak<DiskCacheStore>,
    reader: Weak<SuperfileReader>,
    uri: SuperfileUri,
    storage_uri: String,
    size: u64,
    reserved_bytes: u64,
    fetch_storage: Arc<dyn StorageProvider>,
    skip_vec: Option<(u64, u64)>,
) -> Result<(), DiskCacheError> {
    let Some(store) = wait_for_lazy_foreground_release(&store, &reader).await else {
        return Ok(());
    };
    let tmp = store.tmp_path(&uri);
    let final_path = store.cache_path(&uri);

    if background_store_abandoned(&store) {
        rollback_lazy_background_fill(&store, &uri, &tmp);
        let _ = reserved_bytes;
        return Ok(());
    }

    let _prefetch_permit = match Arc::clone(&store.prefetch_semaphore).acquire_owned().await {
        Ok(permit) => permit,
        Err(error) => {
            rollback_lazy_background_fill(&store, &uri, &tmp);
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
        if !wait_for_reader_quiescence(&store, &reader).await {
            rollback_lazy_background_fill(&store, &uri, &tmp);
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
        )
        .await?
        {
            BackgroundFillOutcome::Complete => break,
            // Keep the partial `tmp` and the `filled` cursor: the next attempt
            // resumes from the first unwritten chunk.
            BackgroundFillOutcome::Paused => {}
            BackgroundFillOutcome::Abandoned => {
                rollback_lazy_background_fill(&store, &uri, &tmp);
                return Ok(());
            }
        }
    }

    let result: Result<(), DiskCacheError> = async {
        if background_store_abandoned(&store) {
            return Ok(());
        }

        tokio::fs::rename(&tmp, &final_path).await?;
        let (mmap_arc, bytes) = mmap_readonly_with_handle(&final_path)?;

        // Reuse the live block-cache source when excluding the vector blob so
        // touched vector ranges from the cold query stay local after promote.
        let prior_block = store
            .cached
            .get(&uri)
            .and_then(|entry| entry.block_source().cloned());
        let (promoted_reader, vector_source) = match (skip_vec, prior_block) {
            (Some((vec_off, vec_len)), Some(block_source)) => {
                let local: Arc<dyn LazyByteSource> =
                    Arc::new(BytesLazyByteSource::new(bytes.clone()));
                let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
                    local,
                    hole_start: vec_off,
                    hole_len: vec_len,
                    fallback: Arc::clone(&block_source),
                });
                let mut reader =
                    SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false })
                        .await?;
                // Sync parquet decodes (take / id scans) run off the mmap;
                // the sparse vector region stays behind the hole source.
                reader.install_resident_parquet(bytes)?;
                (reader, Some(block_source))
            }
            (Some((vec_off, vec_len)), None) => {
                // Evicted mid-fill: fresh block cache over storage for the hole.
                let remote: Arc<dyn LazyByteSource> =
                    Arc::new(StorageRangeSource::with_known_size(
                        Arc::clone(&fetch_storage),
                        storage_uri.clone(),
                        size,
                    ));
                let block_source = BlockCachedSource::new_pre_reserved(
                    remote,
                    Arc::downgrade(&store),
                    uri,
                    store.blocks_path(&uri),
                    // Serves only the promoted reader's vector hole; FTS
                    // bytes come from the mmap.
                    None,
                );
                let local: Arc<dyn LazyByteSource> =
                    Arc::new(BytesLazyByteSource::new(bytes.clone()));
                let source: Arc<dyn LazyByteSource> = Arc::new(HoleFallbackSource {
                    local,
                    hole_start: vec_off,
                    hole_len: vec_len,
                    fallback: Arc::clone(&block_source),
                });
                let mut reader =
                    SuperfileReader::open_lazy_with(source, OpenOptions { verify_crc: false })
                        .await?;
                // Sync parquet decodes (take / id scans) run off the mmap;
                // the sparse vector region stays behind the hole source.
                reader.install_resident_parquet(bytes)?;
                (reader, Some(block_source))
            }
            (None, _) => {
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
        let entry =
            store.build_mmap_entry(Arc::new(promoted_reader), mmap_arc, size, vector_source);
        // Installed (still present) + no retained block source -> the promoted mmap serves every
        // range, so the sparse block sidecar is dead weight; drop it. Evicted mid-fill -> the
        // install dropped the file and we leave the sidecar to normal reclaim.
        if store
            .install_promoted_entry(uri, entry, &final_path, InstallMode::ReplaceIfPresent)
            .is_some()
            && !block_source_retained
        {
            store.drop_block_file(&uri);
        }
        store.coordinators.remove(&uri);
        Ok(())
    }
    .await;

    if result.is_err() || background_store_abandoned(&store) {
        rollback_lazy_background_fill(&store, &uri, &tmp);
        let _ = fs::remove_file(&tmp);
    }
    let _ = reserved_bytes;
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
    use std::sync::{Arc, atomic::Ordering};

    use bytes::Bytes;
    use tempfile::TempDir;
    use tokio::{spawn, task::yield_now, time::timeout};

    use crate::{
        storage::StorageProvider,
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

    /// `rollback_lazy_background_fill` undoes an in-flight promotion: it drops
    /// the cache entry, forgets the coordinator, and deletes the tmp scratch
    /// file left by the partial download.
    #[tokio::test]
    async fn rollback_lazy_background_fill_evicts_entry_and_tmp() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();

        // Seed a cache entry the way a lazy fill would, plus a leftover tmp
        // scratch file for the partial download.
        store.install_block_entry_for_test(uri, dummy_block_source(&store, uri));
        assert!(
            store.is_cached(&uri),
            "entry must be cached before rollback"
        );
        let tmp = store.tmp_path(&uri);
        std::fs::write(&tmp, b"partial-download-bytes").expect("seed tmp scratch file");
        assert!(tmp.exists(), "tmp scratch file must exist before rollback");

        rollback_lazy_background_fill(&store, &uri, &tmp);

        assert!(
            !store.is_cached(&uri),
            "cached entry must be gone after rollback"
        );
        assert!(
            !tmp.exists(),
            "tmp scratch file must be deleted after rollback"
        );
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

    #[tokio::test]
    async fn install_promoted_entry_on_evicted_slot_drops_file_and_skips_insert() {
        // The background-finalize gate: if the slot was evicted while the finalizer ran,
        // ReplaceIfPresent must drop the just-renamed file and NOT reinsert. Reinserting an
        // already-released entry would leak its reservation against the budget.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();

        // Stage a real, openable cache file at the final path.
        let final_path = store.cache_path(&uri);
        if let Some(parent) = final_path.parent() {
            fs::create_dir_all(parent).expect("cache dir");
        }
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        fs::write(&final_path, &bytes).expect("write cache file");

        let (mmap, mapped) = mmap_readonly_with_handle(&final_path).expect("mmap");
        let reader = Arc::new(
            SuperfileReader::open_with(mapped, OpenOptions { verify_crc: false }).expect("open"),
        );
        let entry = store.build_mmap_entry(reader, mmap, size, None);

        // `cached` is empty (slot evicted): ReplaceIfPresent returns None, removes the file, and
        // touches no budget.
        let before = store.current_bytes.load(Ordering::Acquire);
        let installed =
            store.install_promoted_entry(uri, entry, &final_path, InstallMode::ReplaceIfPresent);
        assert!(installed.is_none(), "vacant slot must not reinstate");
        assert!(!final_path.exists(), "file dropped on evicted slot");
        assert_eq!(store.current_bytes.load(Ordering::Acquire), before);
        assert_eq!(store.stats().n_entries, 0);
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
            .reader_with_hints(&uri, &uri.storage_path(), None, Some(&hidden_storage), true)
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
            .reader_with_hints(&uri, &uri.storage_path(), None, Some(&hidden_storage), true)
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

        // reader_with_hints(None) → unknown-size lazy cold fetch.
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
            .reader_with_hints(&uri, &uri.storage_path(), Some(&offsets), None, true)
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
            .reader_with_hints(&uri, &uri.storage_path(), Some(&offsets), None, true)
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
            .reader_with_hints(&uri, &uri.storage_path(), None, None, false)
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
            .reader_with_hints(&uri, &uri.storage_path(), None, None, true)
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
            .reader_with_hints(&uri, &uri.storage_path(), None, None, false)
            .await
            .expect("vector open");
        drop(vector_reader);

        // FTS modality starts the fill, which promotes while leaving the vector blob sparse.
        let fts_reader = store
            .reader_with_hints(&uri, &uri.storage_path(), None, None, true)
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
