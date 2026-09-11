// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The on-disk cache directory: scan it at open, reuse a superfile already
//! sitting there, the per-URI path helpers, and drop a superfile's local copy.

use std::{
    fs,
    path::Path,
    sync::{Arc, atomic::Ordering},
    time::SystemTime,
};

use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use tokio::io::AsyncWriteExt;

use crate::supertable::{
    manifest::SuperfileUri,
    reader_cache::{block_source::indexed_filled_bytes, disk::*},
};

impl DiskCacheStore {
    /// Insert already-in-hand bytes into the cache without
    /// round-tripping through storage. Used by the writer to
    /// pre-populate the cache with the superfiles it just
    /// published, so the producer's next query on its own
    /// superfiles skips the cold-fetch wall-time hit (parallel
    /// range-fetch + pwrite + mmap, ~50-150 ms per superfile on
    /// the laptop bench).
    ///
    /// Idempotent: if `uri` is already in the cache,
    /// returns `Ok(())` without re-writing. Failure modes:
    /// - [`DiskCacheError::BudgetExceeded`] if the byte
    ///   count won't fit even after eviction.
    /// - [`DiskCacheError::Io`] for filesystem failures
    ///   (cache dir not writable, disk full, etc.).
    /// - [`DiskCacheError::SuperfileOpen`] if the bytes
    ///   don't parse as a valid superfile (programmer error
    ///   — the writer must hand over the same bytes it
    ///   wrote to storage).
    ///
    /// Cold-fetch semantics: does **not** increment
    /// `n_cold_fetches` (this is a warm insert, not a
    /// storage round-trip). Increments `n_entries` and
    /// `current_bytes` exactly as the cold-fetch path does.
    pub async fn insert_warm(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        bytes: Bytes,
    ) -> Result<(), DiskCacheError> {
        // Idempotent: already-cached URIs are a no-op. The
        // writer may call this for superfiles a prior commit
        // already published (e.g., an OCC retry where the
        // same UUID superfile got re-inserted into the cache).
        if self.cached.contains_key(uri) {
            return Ok(());
        }

        let size = bytes.len() as u64;

        // Reserve budget (CAS-loop with eviction on miss).
        // Use `reserve_manual` so a panic between this and
        // the DashMap insert doesn't double-decrement on
        // unwind — `reserve_manual` keeps the bytes
        // reserved; we manually roll back on the rare error
        // path below.
        self.reserve_manual(size).await?;

        // Roll back the reservation on any error past this
        // point. Wrap the rest in a closure-shape so `?`
        // works while we still get to undo current_bytes
        // on failure.
        let result: Result<Arc<CachedEntry>, DiskCacheError> = async {
            let tmp = self.tmp_path(uri);
            let final_path = self.cache_path(uri);

            // Write the bytes to a tmp file, then atomically rename into place.
            // No fsync: the disk cache is a reconstructible mirror of bytes that
            // are already durable in object storage, so a crash losing an
            // unflushed cache file just cold-fetches on the next open — and
            // `restore_from_cache_root` CRC-verifies on-disk files at open,
            // dropping any torn one. Skipping the fsync keeps the committer's
            // warm-fill off the synchronous disk-flush path.
            {
                let mut file = tokio::fs::File::create(&tmp).await?;
                file.write_all(&bytes).await?;
                file.flush().await?;
            }
            tokio::fs::rename(&tmp, &final_path).await?;

            // mmap the freshly-written file + open it as a superfile reader.
            // Skip CRC: the committer just built these bytes in memory and they
            // are known-valid (CRC'd at build, already opened as a reader for
            // summary extraction) — re-scanning here is redundant. Files read
            // back from a PRIOR run take the verifying path via
            // `restore_from_cache_root`.
            self.open_cached_entry(&final_path, size, false)
        }
        .await;

        let entry = match result {
            Ok(e) => e,
            Err(e) => {
                // Roll back the reservation; leave any tmp
                // file behind for next-run cleanup (the
                // write may have partially succeeded).
                self.current_bytes.fetch_sub(size, Ordering::Release);
                return Err(e);
            }
        };

        // Final commit: install into the cache map. If a
        // concurrent caller raced us to the same URI (e.g.,
        // a cold-fetch landed first), prefer the
        // already-present entry — release our reservation
        // for the duplicate bytes.
        match self.cached.entry(*uri) {
            Entry::Vacant(v) => {
                v.insert(entry);
            }
            Entry::Occupied(_) => {
                // Lost the race; release our reservation +
                // unlink the just-written file (or leave it
                // — the existing entry mmaps a different
                // file on disk).
                self.current_bytes.fetch_sub(size, Ordering::Release);
                let _ = fs::remove_file(self.cache_path(uri));
            }
        }
        Ok(())
    }

    // ----- internals -----

    /// List `cache_root` and record each superfile's size and mtime in `unindexed`, adding the total
    /// to `current_bytes`. Only a stat per file: files are opened later, by the reads that need them
    /// ([`Self::try_reuse_cached_file`]).
    ///
    /// Deletes leftovers that can never be used: orphaned `.blocks` sidecars, zero-length files,
    /// and `.tmp` files older than [`TMP_RECLAIM_AGE`] (a fresh one belongs to a sibling process's
    /// in-flight fetch). On a scan error the map stays empty and reads just cold-fetch.
    pub(crate) fn scan_cache_root(&self) {
        let dir = match fs::read_dir(&self.config.cache_root) {
            Ok(d) => d,
            Err(_) => return,
        };

        let mut total: u64 = 0;
        for entry in dir.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if name.contains(BLOCKS_IDX_TMP_INFIX) {
                let stale = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .is_some_and(|age| age >= TMP_RECLAIM_AGE);
                if stale {
                    let _ = fs::remove_file(&path);
                }
                continue;
            }
            if name.ends_with(BLOCKS_IDX_SUFFIX) {
                continue;
            }
            if let Some(body) = name.strip_suffix(BLOCKS_FILE_SUFFIX) {
                if let Some(uri) = SuperfileUri::from_cache_filename(body) {
                    match self.scan_block_file(&path, &uri) {
                        Some(bytes) => total += bytes,
                        None => self.drop_block_file(&uri),
                    }
                }
                continue;
            }

            if SuperfileUri::from_cache_tmp_filename(name).is_some() {
                // A fresh tempfile belongs to a sibling's in-flight fetch; only a stale one is a
                // crashed fetch's leftover. See [`TMP_RECLAIM_AGE`].
                let stale = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| SystemTime::now().duration_since(t).ok())
                    .is_some_and(|age| age >= TMP_RECLAIM_AGE);

                if stale {
                    let _ = fs::remove_file(&path);
                }

                continue;
            }

            let Some(uri) = SuperfileUri::from_cache_filename(name) else {
                continue; // Foreign file: leave it alone.
            };

            let Ok(meta) = entry.metadata() else { continue };
            let size = meta.len();

            if size == 0 {
                let _ = fs::remove_file(&path);
                continue;
            }

            self.unindexed.insert(
                uri,
                UnindexedFile {
                    size_bytes: size,
                    mtime_us: file_mtime_us(&meta),
                },
            );

            total += size;
        }

        self.current_bytes.fetch_add(total, Ordering::Release);
        self.trim_cold_to_budget();
    }

    pub(crate) fn scan_block_file(&self, path: &Path, uri: &SuperfileUri) -> Option<u64> {
        let meta = fs::metadata(path).ok()?;
        let size = meta.len();
        if size == 0 {
            return None;
        }
        let idx_bytes = fs::read(self.blocks_idx_path(uri)).ok()?;
        let (_, filled) = indexed_filled_bytes(size, &idx_bytes)?;
        if filled == 0 {
            return None;
        }
        self.block_files.insert(
            *uri,
            UnindexedFile {
                size_bytes: filled,
                mtime_us: file_mtime_us(&meta),
            },
        );
        Some(filled)
    }

    /// Serve `uri` from an existing cache file instead of fetching from storage. Runs before every
    /// cold fetch and checks the filesystem directly, so it also finds files written after this
    /// store opened.
    ///
    /// Returns `Ok(None)` when there is no usable file: missing, wrong size, or it fails to open.
    /// Unusable files are deleted so the cold fetch writes a fresh one. A file can also vanish
    /// between the stat and the open (GC deletes cache copies); that too is just `Ok(None)`.
    ///
    /// `expected_size` is optional: pass it when the caller already has the size (the lazy path gets
    /// it from the manifest), never fetch one. A truncated file fails to open anyway, since the
    /// footer sits at the end.
    pub(crate) async fn try_reuse_cached_file(
        &self,
        uri: &SuperfileUri,
        expected_size: Option<u64>,
    ) -> Result<Option<Arc<CachedEntry>>, DiskCacheError> {
        let path = self.cache_path(uri);
        let Ok(meta) = fs::metadata(&path) else {
            // No whole-file copy; decrement a vanished counted one but leave any block cache intact.
            if let Some((_, file)) = self.unindexed.remove(uri) {
                self.current_bytes
                    .fetch_sub(file.size_bytes, Ordering::Release);
            }
            return Ok(None);
        };

        let size = meta.len();
        if size == 0 || expected_size.is_some_and(|want| want != size) {
            self.discard_cache_file(uri);
            return Ok(None);
        }

        // The scan already counted this file's bytes. A file it never saw (written later by another
        // process) is charged like a fetch.
        let counted = self.unindexed.remove(uri).is_some();
        let reservation = if counted {
            None
        } else {
            Some(self.reserve(size).await?)
        };

        match self.open_cached_entry(&path, size, self.config.verify_crc_on_open) {
            Ok(entry) => {
                // Two racing reuses of one URI can both insert; the second insert must free the
                // first one's bytes.
                if let Some(replaced) = self.cached.insert(*uri, Arc::clone(&entry)) {
                    self.release_entry_accounting(&replaced);
                }

                if let Some(r) = reservation {
                    r.commit();
                }

                self.n_disk_reuses.fetch_add(1, Ordering::AcqRel);

                Ok(Some(entry))
            }
            Err(_) => {
                // A dropped reservation frees itself; scan-counted bytes are given back by hand.
                if counted {
                    self.current_bytes.fetch_sub(size, Ordering::Release);
                }
                let _ = fs::remove_file(&path);
                self.drop_block_file(uri);
                Ok(None)
            }
        }
    }

    /// Delete an unusable cache file and give back the bytes the scan counted for it. Subtracts the
    /// record's own size, so the accounting balances even when the file on disk no longer matches
    /// what the scan saw (or is gone entirely).
    pub(crate) fn discard_cache_file(&self, uri: &SuperfileUri) {
        if let Some((_, file)) = self.unindexed.remove(uri) {
            self.current_bytes
                .fetch_sub(file.size_bytes, Ordering::Release);
        }

        let _ = fs::remove_file(self.cache_path(uri));
        self.drop_block_file(uri);
    }

    pub(crate) fn drop_block_file(&self, uri: &SuperfileUri) {
        if let Some((_, file)) = self.block_files.remove(uri) {
            self.current_bytes
                .fetch_sub(file.size_bytes, Ordering::Release);
        }
        let _ = fs::remove_file(self.blocks_path(uri));
        let _ = fs::remove_file(self.blocks_idx_path(uri));
    }

    pub(crate) fn release_scanned_block_file(&self, uri: &SuperfileUri) {
        if let Some((_, prior)) = self.block_files.remove(uri) {
            self.current_bytes
                .fetch_sub(prior.size_bytes, Ordering::Release);
        }
    }

    /// Account bytes for an adopted block cache whose data is already on disk.
    /// Bytes a restart scan already counted for this file are released first (so
    /// a scanned-then-adopted file nets to zero), then any excess over budget is
    /// trimmed from cold candidates rather than left for a reservation that a
    /// read-only table never issues.
    pub(crate) fn account_adopted_bytes(&self, uri: &SuperfileUri, bytes: u64) {
        self.release_scanned_block_file(uri);
        self.current_bytes.fetch_add(bytes, Ordering::Release);
        self.trim_cold_to_budget();
    }

    /// Erase every local trace of a superfile: its in-memory index entry, the promoted file, the
    /// sparse block sidecar, and the per-URI fetch coordinator.
    ///
    /// GC calls this immediately before deleting the superfile from storage, so the local copy never
    /// outlives its source. Returns whether an index entry was present, which is what `n_gc_drops`
    /// counts.
    pub(crate) fn erase_superfile_local_copy(&self, uri: &SuperfileUri) -> bool {
        let present = if let Some((_, entry)) = self.cached.remove(uri) {
            self.release_entry_accounting(&entry);
            true
        } else {
            false
        };

        // Give back bytes counted for a never-opened file.
        if let Some((_, file)) = self.unindexed.remove(uri) {
            self.current_bytes
                .fetch_sub(file.size_bytes, Ordering::Release);
        }

        self.coordinators.remove(uri);
        let _ = fs::remove_file(self.cache_path(uri));
        self.drop_block_file(uri);
        if present {
            self.n_gc_drops.fetch_add(1, Ordering::AcqRel);
        }

        present
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use tempfile::TempDir;

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            manifest::SuperfileUri,
            reader_cache::{
                block_source::CACHE_BLOCK_BYTES,
                config::DiskCacheConfig,
                disk::{store_files::*, test_support::*},
            },
        },
    };

    #[tokio::test]
    async fn insert_warm_caches_and_serves_reader() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        store.insert_warm(&uri, bytes).await.expect("insert_warm");

        // Entry is mmap-backed, counted, and warm inserts don't bump
        // the cold-fetch counter.
        assert!(store.is_mmap_promoted(&uri));
        let s = store.stats();
        assert_eq!(s.n_entries, 1);
        assert_eq!(s.current_bytes, size);
        assert_eq!(s.n_cold_fetches, 0);
        assert_eq!(store.current_mmap_size_bytes(), size);

        // The cache file landed on disk.
        assert!(store.cache_path(&uri).is_file());

        // reader() hits the cache (still no cold fetch).
        let _r = store.reader(&uri).await.expect("reader");
        assert_eq!(store.stats().n_cold_fetches, 0);
    }

    #[tokio::test]
    async fn insert_warm_is_idempotent() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("first");
        let before = store.stats().current_bytes;
        // Second insert with the same URI is a no-op.
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("second");
        assert_eq!(store.stats().current_bytes, before);
        assert_eq!(store.stats().n_entries, 1);
    }

    #[tokio::test]
    async fn insert_warm_rejects_unparseable_bytes() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, Bytes::from_static(b"not a superfile"))
            .await
            .expect_err("garbage must fail to open");
        // Reservation rolled back on the error path.
        assert_eq!(store.stats().current_bytes, 0);
        assert_eq!(store.stats().n_entries, 0);
        // Surfaced as a typed open/read error.
        let _ = format!("{err}");
        let _ = format!("{err:?}");
    }

    #[tokio::test]
    async fn insert_warm_budget_exceeded_when_too_big() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = 4; // smaller than any real superfile
        });
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect_err("must exceed budget");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));
        assert_eq!(store.stats().current_bytes, 0);
    }

    // ----- engine-managed (auto-sized) budget reconciliation -----

    #[tokio::test]
    async fn erase_superfile_local_copy_removes_entry_file_and_accounting() {
        // GC drop-through: a dropped URI leaves no entry, no promoted file, no
        // block sidecar, and balanced byte accounting.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("insert_warm");
        assert!(store.cache_path(&uri).is_file());
        assert_eq!(store.stats().n_entries, 1);

        assert!(store.erase_superfile_local_copy(&uri), "entry was present");
        let s = store.stats();
        assert_eq!(s.n_entries, 0);
        assert_eq!(s.current_bytes, 0, "accounting released");
        assert_eq!(s.n_gc_drops, 1);
        assert!(!store.cache_path(&uri).exists(), "promoted file unlinked");
        assert!(!store.blocks_path(&uri).exists(), "block sidecar unlinked");

        // A second drop of the same URI is a no-op, and a never-cached URI
        // reports absent — only real drops count.
        assert!(!store.erase_superfile_local_copy(&uri));
        assert!(!store.erase_superfile_local_copy(&SuperfileUri::new_v4()));
        assert_eq!(store.stats().n_gc_drops, 1);
    }

    #[tokio::test]
    async fn erase_superfile_local_copy_keeps_a_held_reader_alive() {
        // GC can drop a URI a query is reading. The mapping survives: unlink
        // removes the directory entry, not the inode, so the held `Arc<Mmap>`
        // keeps faulting valid pages (this is why eviction has always been
        // safe under live readers).
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("insert_warm");
        let held = store.reader(&uri).await.expect("reader");
        let n_docs = held.n_docs();

        assert!(store.erase_superfile_local_copy(&uri));
        assert!(!store.cache_path(&uri).exists(), "file unlinked");

        // Same reader, after the drop: still serving its own mapping.
        assert_eq!(held.n_docs(), n_docs, "held reader still reads its mmap");

        // A refetch of the same URI writes a fresh inode via tmp + rename, so
        // it cannot zero the bytes the held reader is still mapping.
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("refetch after drop");
        assert_eq!(
            held.n_docs(),
            n_docs,
            "refetch left the held mapping intact"
        );
    }

    #[tokio::test]
    async fn erase_superfile_local_copy_leaves_source_owned_accounting_to_its_owner() {
        // A block-backed entry's bytes are accounted by the block source, not
        // the entry (`EntryAccounting::SourceOwned`). Dropping the entry must
        // not decrement for it — the source's own release does that, and a
        // second decrement here would underflow `current_bytes`.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let block_source = dummy_block_source(&store, uri);
        block_source
            .filled_bytes_handle()
            .store(4096, Ordering::Release);
        store.install_block_entry_for_test(uri, block_source);
        assert_eq!(store.stats().current_bytes, 0, "source owns these bytes");

        assert!(store.erase_superfile_local_copy(&uri), "entry was present");
        let s = store.stats();
        assert_eq!(s.n_entries, 0);
        assert_eq!(s.current_bytes, 0, "no decrement, no underflow");
        assert_eq!(s.n_gc_drops, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn erase_superfile_local_copy_concurrent_with_a_fill_keeps_accounting_consistent() {
        // A fill and an erase race on one URI, 200 interleavings. Whichever wins, the
        // byte count must equal what is actually cached, and a fill that lost its file
        // to the erase must release what it reserved. Drift either way corrupts the
        // budget: phantom bytes starve the cache, missing ones let it overrun.
        let (_dir, store) = test_store();
        let size = tiny_superfile_bytes().len() as u64;

        for _ in 0..RACE_ITERATIONS {
            let uri = SuperfileUri::new_v4();
            let filler = Arc::clone(&store);
            let eraser = Arc::clone(&store);
            let fill =
                tokio::spawn(async move { filler.insert_warm(&uri, tiny_superfile_bytes()).await });
            let erase = tokio::spawn(async move { eraser.erase_superfile_local_copy(&uri) });
            let (fill_res, erase_res) = tokio::join!(fill, erase);
            let filled = fill_res.expect("fill task").is_ok();
            erase_res.expect("erase task");

            let s = store.stats();
            let expected = if s.n_entries == 1 { size } else { 0 };
            assert_eq!(
                s.current_bytes, expected,
                "bytes must match entry presence (entries={}, filled={filled})",
                s.n_entries
            );

            if !filled {
                assert_eq!(s.n_entries, 0, "a failed fill leaves no entry behind");
                assert_eq!(
                    s.current_bytes, 0,
                    "a failed fill rolls its reservation back"
                );
            }

            // Leave a clean slate for the next iteration.
            store.erase_superfile_local_copy(&uri);
            assert_eq!(store.stats().current_bytes, 0);
        }
    }

    #[tokio::test]
    async fn erase_superfile_local_copy_unlinks_file_left_without_an_entry() {
        // A cache file can exist with no map entry (crash between rename and
        // insert); erase_superfile_local_copy still unlinks it so the orphan cannot outlive its
        // storage object.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        fs::write(store.cache_path(&uri), b"stale bytes").expect("write orphan");

        assert!(!store.erase_superfile_local_copy(&uri), "no entry to drop");
        assert!(
            !store.cache_path(&uri).exists(),
            "orphan file still unlinked"
        );
        assert_eq!(store.stats().n_gc_drops, 0);
    }

    #[tokio::test]
    async fn erase_superfile_local_copy_unlinks_block_index() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        fs::write(store.blocks_path(&uri), b"blocks").expect("blocks");
        fs::write(store.blocks_idx_path(&uri), b"idx").expect("idx");

        store.erase_superfile_local_copy(&uri);

        assert!(!store.blocks_path(&uri).exists());
        assert!(!store.blocks_idx_path(&uri).exists());
    }

    #[tokio::test]
    async fn restart_scan_preserves_block_cache_and_index() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        seed_valid_block_file(&store, &uri, 2 * CACHE_BLOCK_BYTES, &[0, 1]);

        let reopened = reopen_store(&store, |_| {});

        assert!(
            reopened.blocks_path(&uri).exists(),
            "restart keeps the .blocks file for adoption"
        );
        assert!(
            reopened.blocks_idx_path(&uri).exists(),
            "restart keeps the .blocks.idx sidecar"
        );
    }

    #[tokio::test]
    async fn restart_scan_deletes_a_block_file_with_an_invalid_index() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        fs::write(store.blocks_path(&uri), b"sparse block bytes").expect("blocks");
        fs::write(store.blocks_idx_path(&uri), b"not a valid index").expect("idx");

        let reopened = reopen_store(&store, |_| {});

        assert!(
            !reopened.blocks_path(&uri).exists(),
            "an un-adoptable block file is reclaimed, not leaked"
        );
        assert!(!reopened.blocks_idx_path(&uri).exists());
        assert_eq!(reopened.stats().current_bytes, 0);
    }

    #[tokio::test]
    async fn restart_scan_sweeps_stale_index_tempfiles_and_spares_fresh_ones() {
        let (_dir, store) = test_store();
        let idx = store.blocks_idx_path(&SuperfileUri::new_v4());
        let stale = idx.with_extension("idx.tmp.4242.0");
        let fresh = idx.with_extension("idx.tmp.4242.1");
        let now = SystemTime::now();
        for (path, mtime) in [(&stale, now - TMP_RECLAIM_AGE * 2), (&fresh, now)] {
            fs::write(path, b"partial").expect("seed tmp");
            fs::File::options()
                .write(true)
                .open(path)
                .expect("open tmp")
                .set_modified(mtime)
                .expect("set mtime");
        }

        let _opened = reopen_store(&store, |_| {});
        assert!(!stale.exists(), "stale index tempfile reclaimed");
        assert!(fresh.exists(), "fresh index tempfile left for its owner");
    }

    #[tokio::test]
    async fn restart_scan_counts_retained_block_files() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let size = 3 * CACHE_BLOCK_BYTES + 1000;
        let filled = seed_valid_block_file(&store, &uri, size, &[0, 1, 3]);

        let reopened = reopen_store(&store, |_| {});
        assert_eq!(
            reopened.stats().current_bytes,
            filled,
            "scan counts the retained block bytes against the budget"
        );
    }

    #[tokio::test]
    async fn scan_reclaims_a_block_file_without_an_index() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let f = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(store.blocks_path(&uri))
            .expect("blocks");
        f.set_len(4 * CACHE_BLOCK_BYTES).expect("set_len");

        let reopened = reopen_store(&store, |_| {});
        assert_eq!(reopened.stats().current_bytes, 0);
        assert!(
            !reopened.blocks_path(&uri).exists(),
            "a block file with no index cannot be adopted, so it is reclaimed"
        );
    }

    #[tokio::test]
    async fn a_non_owning_adopt_releases_the_scanned_block_bytes() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let filled = seed_valid_block_file(&store, &uri, 2 * CACHE_BLOCK_BYTES, &[0, 1]);

        let reopened = reopen_store(&store, |_| {});
        assert_eq!(reopened.stats().current_bytes, filled);

        reopened.release_scanned_block_file(&uri);
        assert_eq!(
            reopened.stats().current_bytes,
            0,
            "a promotion-path adopt drops the scan-counted bytes instead of double-counting"
        );
    }

    #[tokio::test]
    async fn adopting_a_scanned_block_file_does_not_double_count() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let size = 2 * CACHE_BLOCK_BYTES + 500;
        let filled = seed_valid_block_file(&store, &uri, size, &[0, 1]);

        let reopened = reopen_store(&store, |_| {});
        assert_eq!(reopened.stats().current_bytes, filled);

        reopened.account_adopted_bytes(&uri, filled);
        assert_eq!(
            reopened.stats().current_bytes,
            filled,
            "adopt claims the scanned bytes rather than adding them again"
        );
    }

    #[tokio::test]
    async fn open_does_not_index_the_whole_cache_directory() {
        // Opening a table must not pay for files no query has asked for. The budget still has to
        // know the directory, so the stat pass runs; only the mmap + footer parse + CRC is deferred.
        let (_dir, store) = test_store();
        let bytes = tiny_superfile_bytes();
        let mut total = 0;
        for _ in 0..SEEDED_CACHE_FILES {
            total += seed_cache_file(&store, &SuperfileUri::new_v4(), &bytes);
        }

        // Second store over the same cache_root: this is the open under test.
        let opened = reopen_store(&store, |_| {});

        let s = opened.stats();
        assert_eq!(
            s.n_entries, 0,
            "open indexes nothing: {SEEDED_CACHE_FILES} files on disk, none opened"
        );
        assert_eq!(
            s.current_bytes, total,
            "budget still knows the directory from the stat pass"
        );
    }

    #[tokio::test]
    async fn read_reuses_a_cache_file_the_index_never_saw() {
        // A file that appears after open (another process wrote it, or open deliberately skipped
        // it) must still be served from local bytes.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();

        // Present in BOTH places, so the fallback path also succeeds and the only difference
        // between pass and fail is where the bytes came from.
        put_superfile(&store, &uri, bytes.clone()).await;
        seed_cache_file(&store, &uri, &bytes);

        let _reader = store.reader(&uri).await.expect("reader");

        let s = store.stats();
        assert_eq!(
            s.n_cold_fetches, 0,
            "served from the local file, no storage round-trip"
        );
        assert_eq!(s.n_disk_reuses, 1);
        assert_eq!(s.n_entries, 1, "the reused file is now indexed");
        assert_eq!(
            s.current_bytes,
            bytes.len() as u64,
            "a file the open-time scan never saw is charged like a fetch"
        );
    }

    #[tokio::test]
    async fn second_store_reuses_on_read_not_on_open() {
        // Cross-process reuse under a lazy open: a fresh store over a warm cache_root starts empty
        // and pays nothing until a read asks, and that read costs no source operations.
        let dir = TempDir::new().expect("tempdir");
        let cache_root = dir.path().join("cache");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;

        {
            let cfg = DiskCacheConfig {
                cache_root: cache_root.clone(),
                mmap_cold_threshold_secs: 0,
                ..Default::default()
            };
            let first = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg).expect("store1");
            first.insert_warm(&uri, bytes).await.expect("insert_warm");
            assert!(first.cache_path(&uri).is_file());
        }

        let cfg2 = DiskCacheConfig {
            cache_root: cache_root.clone(),
            mmap_cold_threshold_secs: 0,
            ..Default::default()
        };
        let second = DiskCacheStore::new_unpinned(Arc::clone(&storage), cfg2).expect("store2");

        let at_open = second.stats();
        assert_eq!(at_open.n_entries, 0, "nothing opened at construction");
        assert_eq!(
            at_open.current_bytes, size,
            "the file's bytes are counted against the budget"
        );

        let _reader = second.reader(&uri).await.expect("reader from disk");
        let after = second.stats();
        assert_eq!(after.n_cold_fetches, 0, "reuse, not a re-fetch");
        assert_eq!(after.n_disk_reuses, 1);
        assert_eq!(after.n_entries, 1);
        assert_eq!(
            after.current_bytes, size,
            "reuse moves ownership, not bytes"
        );
    }

    #[tokio::test]
    async fn unusable_cache_files_fall_through_to_source() {
        // Every rejection path lands in the same place: unlink the bad file and let the read
        // cold-fetch a clean copy. Nothing is ever served from a file that failed to open.
        let bytes = tiny_superfile_bytes();
        for (label, seeded) in [
            ("zero length", Bytes::new()),
            ("not a superfile", Bytes::from_static(b"garbage bytes")),
        ] {
            let (_dir, store) = test_store();
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, bytes.clone()).await;
            seed_cache_file(&store, &uri, &seeded);

            let reader = store.reader(&uri).await;
            assert!(reader.is_ok(), "{label}: falls through and serves");
            assert_eq!(
                store.stats().n_cold_fetches,
                1,
                "{label}: came from storage, not the bad local file"
            );
        }
    }

    #[tokio::test]
    async fn scan_reclaims_stale_tmp_files_and_spares_fresh_ones() {
        // A stale tempfile can only be a crashed fetch's leftover, but a fresh one belongs to a
        // sibling process's in-flight fetch: deleting it would fail that fetch's rename.
        let (_dir, store) = test_store();
        let bytes = tiny_superfile_bytes();
        let stale = SuperfileUri::new_v4();
        let fresh = SuperfileUri::new_v4();
        let skewed = SuperfileUri::new_v4();
        let now = SystemTime::now();
        for (uri, mtime) in [
            (stale, now - TMP_RECLAIM_AGE * 2),
            (fresh, now),
            // A future mtime (clock skew) must read as not-stale, never as reclaimable.
            (skewed, now + TMP_RECLAIM_AGE),
        ] {
            let path = store.tmp_path(&uri);
            fs::write(&path, bytes.as_ref()).expect("seed tmp");
            fs::File::options()
                .write(true)
                .open(&path)
                .expect("open tmp")
                .set_modified(mtime)
                .expect("set mtime");
        }

        let opened = reopen_store(&store, |_| {});

        assert!(!opened.tmp_path(&stale).exists(), "stale tmp reclaimed");
        assert!(
            opened.tmp_path(&fresh).exists(),
            "fresh tmp left for its owner"
        );
        assert!(
            opened.tmp_path(&skewed).exists(),
            "future mtime spared under clock skew"
        );
        assert_eq!(
            opened.stats().current_bytes,
            0,
            "tmp files never count against the budget"
        );
    }

    #[tokio::test]
    async fn sibling_fetch_completes_after_a_scan_and_its_file_is_adopted() {
        // The interleaving the age gate protects: another process's fetch is in flight while we
        // open, its rename lands after our scan, and our read adopts the finished file.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        fs::write(store.tmp_path(&uri), bytes.as_ref()).expect("sibling's in-flight tmp");

        let opened = reopen_store(&store, |_| {});
        assert!(
            opened.tmp_path(&uri).exists(),
            "scan spared the in-flight tmp"
        );

        // The sibling finishes: fsync'd bytes, atomic rename to the final name.
        fs::rename(opened.tmp_path(&uri), opened.cache_path(&uri)).expect("sibling renames");

        let _r = opened
            .reader(&uri)
            .await
            .expect("read adopts the finished file");
        let stats = opened.stats();
        assert_eq!(stats.n_disk_reuses, 1);
        assert_eq!(stats.n_cold_fetches, 0, "no storage round-trip");
    }

    #[tokio::test]
    async fn externally_deleted_counted_file_stops_counting_on_first_touch() {
        // A file the scan counted can be deleted outside the engine. The first read that misses it
        // must drop its record: left counted, it double-counts against the fetched replacement and
        // a later eviction of the phantom would unlink the replacement's live file.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        let size = bytes.len() as u64;
        put_superfile(&store, &uri, bytes.clone()).await;
        seed_cache_file(&store, &uri, &bytes);

        let opened = reopen_store(&store, |_| {});
        assert_eq!(opened.stats().current_bytes, size, "scan counted the file");
        fs::remove_file(opened.cache_path(&uri)).expect("external deletion");

        // The synchronous path renames the replacement into place before returning, so the
        // file's presence is deterministic to assert.
        let _r = opened
            .reader_synchronous(&uri)
            .await
            .expect("cold fetch replaces it");
        let s = opened.stats();
        assert_eq!(s.n_cold_fetches, 1);
        assert_eq!(
            s.current_bytes, size,
            "no phantom bytes: the record died with the file"
        );
        assert!(opened.cache_path(&uri).is_file(), "replacement landed");

        // The phantom is gone, so nothing can unlink the replacement out from under its entry.
        assert_eq!(
            opened.evict_unindexed(u64::MAX),
            0,
            "no unindexed leftovers"
        );
        assert!(opened.cache_path(&uri).is_file());
    }

    #[tokio::test]
    async fn reuse_rejects_a_file_of_the_wrong_size() {
        // Only the lazy path knows the expected size; when it disagrees, the file must be unlinked
        // and the caller sent to source, never served.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let bytes = tiny_superfile_bytes();
        seed_cache_file(&store, &uri, &bytes);

        let wrong = bytes.len() as u64 + 1;
        let reused = store
            .try_reuse_cached_file(&uri, Some(wrong))
            .await
            .expect("reuse probe");
        assert!(reused.is_none(), "size mismatch is a miss, not a serve");
        assert!(
            !store.cache_path(&uri).exists(),
            "mismatched file unlinked so the fetch lands on a fresh inode"
        );
        assert_eq!(store.stats().n_disk_reuses, 0, "a rejection is not a reuse");
    }

    // ----- cold fetch: synchronous path -----
}
