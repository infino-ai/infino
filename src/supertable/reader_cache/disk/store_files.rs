// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The on-disk cache directory: scan it at open, reuse a superfile already
//! sitting there, the per-URI path helpers, and drop a superfile's local copy.

use std::{
    fs,
    os::unix::fs::MetadataExt,
    path::Path,
    sync::{Arc, atomic::Ordering},
    time::SystemTime,
};

use bytes::Bytes;
use dashmap::mapref::entry::Entry;
use tokio::io::AsyncWriteExt;

use crate::{
    storage::StorageProvider,
    supertable::{
        manifest::SuperfileUri,
        reader_cache::{block_source::indexed_filled_bytes, disk::*},
    },
};

/// Whether adopting an on-disk cache file verifies its checksums.
///
/// A query does not: the file was verified by whichever fill wrote it, nothing
/// is being written now, and checking it would mean reading every byte of
/// every cached superfile at process start — the scan this policy exists to
/// remove, and one the engines this matches default to off.
///
/// A [`ReadIntent::Load`] does, because a Load is a compaction or reindex
/// input. Those read every byte of the file anyway, so the checksum is
/// marginal, and what they do with it is the reason: the merge writes a *new*
/// superfile with fresh, valid checksums over whatever it read, and once the
/// inputs are reclaimed that output is the only copy left. A bad input
/// adopted here is therefore not a wrong answer that a refetch repairs, it is
/// the good copy being replaced by a bad one — which is the one failure the
/// "recovery is a refetch" argument cannot survive.
///
/// `never` still means never, at every boundary.
fn adoption_verifies(config: &DiskCacheConfig, intent: ReadIntent) -> bool {
    match intent {
        ReadIntent::Load => config.crc_verification.verifies_on_landing(),
        ReadIntent::Warm | ReadIntent::Stream => config.crc_verification.verifies_on_open(),
    }
}

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

            // Write the bytes to a tmp file, fsync, then atomically rename into
            // place. The fsync is what makes the file on disk match the bytes
            // written: nothing reads it back to check. An earlier version
            // skipped it and leaned on a later run re-verifying every cache
            // file at open to drop a torn one, but the default policy no
            // longer does that scan, so a crash between the write and the
            // flush would leave a torn file that the next run adopts. The
            // other two fill paths already `sync_all` for the same reason.
            {
                let mut file = tokio::fs::File::create(&tmp).await?;
                file.write_all(&bytes).await?;
                file.flush().await?;
                file.sync_all().await?;
            }
            tokio::fs::rename(&tmp, &final_path).await?;

            // mmap the freshly-written file + open it as a superfile reader.
            // Skip CRC: the committer just built these bytes in memory and they
            // are known-valid (CRC'd at build, already opened as a reader for
            // summary extraction) — re-scanning here is redundant, and the
            // fsync above is what carries that validity to disk.
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
    /// ([`Self::fetch_from_disk_cache`]).
    ///
    /// Deletes leftovers that can never be used: orphaned `.blocks` sidecars, zero-length files,
    /// `.tmp` files older than [`TMP_RECLAIM_AGE`] (a fresh one belongs to a sibling process's
    /// in-flight fetch), and a holed copy ([`Self::hole_path`]) that a complete copy of the same
    /// superfile makes redundant. On a scan error the map stays empty and reads just cold-fetch.
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

            let Some((uri, holed)) = SuperfileUri::from_cache_filename(name)
                .map(|uri| (uri, false))
                .or_else(|| SuperfileUri::from_cache_hole_filename(name).map(|uri| (uri, true)))
            else {
                continue; // Foreign file: leave it alone.
            };

            // A complete copy wins over one missing its vector blob.
            if holed && self.cache_path(&uri).exists() {
                let _ = fs::remove_file(&path);
                continue;
            }

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
        let (_, filled) = indexed_filled_bytes(size, &idx_bytes, meta.ino())?;
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
        tracing::info!(target: "infino::cache", uri = %uri.0, filled, "cache build: adopted .blocks on scan");
        Some(filled)
    }

    /// Serve `uri` from an existing cache file instead of fetching from storage. Runs before every
    /// cold fetch and checks the filesystem directly, so it also finds files written after this
    /// store opened.
    ///
    /// A copy under [`Self::hole_path`] is reused too: it is opened the way its fill left it, with
    /// the vector blob read through the block cache, and charged for what it holds.
    ///
    /// Returns `Ok(None)` when there is no usable file: missing, wrong size, or it fails to open.
    /// Unusable files are deleted so the cold fetch writes a fresh one. A file can also vanish
    /// between the stat and the open (GC deletes cache copies); that too is just `Ok(None)`.
    ///
    /// `expected_size` is optional: pass it when the caller already has the size (the lazy path gets
    /// it from the manifest), never fetch one. A truncated file fails to open anyway, since the
    /// footer sits at the end.
    pub(crate) async fn fetch_from_disk_cache(
        self: &Arc<Self>,
        uri: &SuperfileUri,
        storage_key: &str,
        expected_size: Option<u64>,
        intent: ReadIntent,
        storage: Option<&Arc<dyn StorageProvider>>,
    ) -> Result<Option<Arc<CachedEntry>>, DiskCacheError> {
        let found = [(self.cache_path(uri), false), (self.hole_path(uri), true)]
            .into_iter()
            .find_map(|(path, holed)| fs::metadata(&path).ok().map(|meta| (path, meta, holed)));

        let Some((path, meta, holed)) = found else {
            // No whole-file copy; decrement a vanished counted one but leave any block cache intact.
            if let Some((_, file)) = self.unindexed.remove(uri) {
                self.current_bytes
                    .fetch_sub(file.size_bytes, Ordering::Release);
            }
            return Ok(None);
        };

        let size = meta.len();
        if size == 0 || expected_size.is_some_and(|want| want != size) {
            tracing::warn!(
                target: "infino::cache",
                uri = %uri.0,
                on_disk_bytes = size,
                manifest_bytes = expected_size,
                "cache file size zero or does not match manifest offsets; discarding + re-fetching"
            );
            self.discard_cache_file(uri);
            return Ok(None);
        }

        // The scan already counted this file's bytes, and the open takes that charge over; a file
        // it never saw (written later by another process) is charged like a fetch. Either way the
        // charge is given back unless an entry is admitted. An open that fails, or is dropped
        // midway (a cancelled query, while a holed copy reads its vector header), leaves the file
        // on disk uncounted until its next open or a restart's scan counts it again. Putting the
        // scan's record back instead could count it twice, were a concurrent open admitting the
        // same file meanwhile.
        let reservation = match self.unindexed.remove(uri) {
            Some(_) => self.adopt_charge(size),
            None => self.reserve(size).await?,
        };

        let opened = if holed {
            self.open_holed_entry(uri, size, storage_key, self.resolve_storage(storage))
                .await
        } else {
            self.open_cached_entry(&path, size, adoption_verifies(&self.config, intent))
        };

        match opened {
            Ok(entry) => {
                // Counted or reserved at the file's length; a holed copy is charged less, since
                // its hole is not on disk.
                let uncharged = size - entry.size_bytes.load(Ordering::Acquire);

                // Two racing reuses of one URI can both admit; admission frees the loser's bytes.
                let entry = self.admit_entry(*uri, entry);

                reservation.commit();

                self.current_bytes.fetch_sub(uncharged, Ordering::Release);

                self.n_disk_reuses.fetch_add(1, Ordering::AcqRel);

                Ok(Some(entry))
            }
            Err(e) => {
                // The reservation, dropped here, gives the charge back.
                drop(reservation);

                tracing::warn!(target: "infino::cache", uri = %uri.0, err = ?e, "evict: cache file failed to open, discarding + re-fetching");

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

        tracing::info!(target: "infino::cache", uri = %uri.0, "evict: discard whole .sf.parquet (unusable/size-mismatch)");
        self.remove_local_copy(uri);
        self.drop_block_file(uri);
    }

    /// Delete the local copy of `uri`, under whichever name it has.
    pub(crate) fn remove_local_copy(&self, uri: &SuperfileUri) {
        let _ = fs::remove_file(self.cache_path(uri));
        let _ = fs::remove_file(self.hole_path(uri));
    }

    pub(crate) fn drop_block_file(&self, uri: &SuperfileUri) {
        if let Some((_, file)) = self.block_files.remove(uri) {
            self.current_bytes
                .fetch_sub(file.size_bytes, Ordering::Release);
        }
        tracing::info!(target: "infino::cache", uri = %uri.0, "evict: dropped .blocks + .blocks.idx (unusable/scan)");
        let _ = fs::remove_file(self.blocks_path(uri));
        let _ = fs::remove_file(self.blocks_idx_path(uri));
    }

    /// Keep charging the `.blocks` file of a dropped lazy entry's `source`. The source releases its
    /// own bytes on drop but leaves the file for a later open to adopt; untracked, the file would
    /// sit on disk outside the budget, where eviction cannot reach it. Skipped when the path no
    /// longer names the source's file: a newer generation replaced it, and a record left for the
    /// old file would later be evicted and unlink the newer one.
    pub(crate) fn retain_block_file(
        &self,
        uri: &SuperfileUri,
        source: &BlockCachedSource,
        now_us: u64,
    ) {
        let filled = source.filled_bytes_handle().load(Ordering::Acquire);
        if filled == 0 || !source.path_names_its_file() {
            return;
        }
        self.current_bytes.fetch_add(filled, Ordering::Release);
        let file = UnindexedFile {
            size_bytes: filled,
            mtime_us: now_us,
        };
        if let Some(prior) = self.block_files.insert(*uri, file) {
            self.current_bytes
                .fetch_sub(prior.size_bytes, Ordering::Release);
        }
    }

    /// [`Self::retain_block_file`] for a block source dropped before its entry was admitted (an
    /// open that failed, was cancelled, or lost its slot to a whole file). Such a source writes no
    /// blocks, so what it holds is a file it adopted, whose record it took over. Skipped when a
    /// lazy entry holds `uri`, whose own source owns the path.
    pub(crate) fn retain_unadmitted_block_file(
        &self,
        uri: &SuperfileUri,
        source: &BlockCachedSource,
    ) {
        let lazy_entry_owns_path = self
            .cached
            .get(uri)
            .is_some_and(|entry| entry.block_source().is_some());
        if !lazy_entry_owns_path {
            self.retain_block_file(uri, source, self.now_us());
        }
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
        tracing::info!(target: "infino::cache", uri = %uri.0, present, "evict: erased local copy (object-store GC drop)");
        self.remove_local_copy(uri);
        self.drop_block_file(uri);
        if present {
            self.n_gc_drops.fetch_add(1, Ordering::AcqRel);
        }

        present
    }
}

#[cfg(test)]
mod landing_tests {
    use std::{
        fs::{read, write},
        path::Path,
    };

    use bytes::Bytes;

    use super::super::test_support::{
        put_superfile, reopen_store, test_store_with, tiny_vector_superfile_bytes,
    };
    use crate::{
        config::CrcVerification, superfile::format::footer::read_kv_metadata,
        supertable::manifest::SuperfileUri,
    };

    /// Bytes back from the vector blob's end to flip. Near the end rather
    /// than near the start so the flip lands in data covered by a checksum
    /// rather than in a header or an offset table, which a structural check
    /// would reject whatever the policy said.
    const CORRUPT_INTO_VEC: usize = 16;

    /// `bytes` with one byte of its vector blob flipped, which is a region
    /// the superfile's own checksums cover.
    ///
    /// The region matters. Most of the file is Parquet, which these checksums
    /// do not cover, so a flip at an arbitrary offset would go unnoticed and
    /// prove nothing about the policy. The fixture is chosen for the same
    /// reason: a superfile built with no FTS and no vector config carries no
    /// embedded blob, and therefore nothing for verification to check.
    fn corrupt_the_vector_blob(bytes: impl AsRef<[u8]>) -> Bytes {
        let mut bytes = bytes.as_ref().to_vec();
        let kv = read_kv_metadata(&bytes).expect("the fixture carries its footer");
        let vec_off: usize = kv["inf.vec.offset"].parse().expect("vector blob offset");
        let vec_len: usize = kv["inf.vec.length"].parse().expect("vector blob length");
        bytes[vec_off + vec_len - CORRUPT_INTO_VEC] ^= 0xFF;
        Bytes::from(bytes)
    }

    /// Flip the same byte in a copy already on the cache disk.
    fn corrupt_the_cached_copy(path: &Path) {
        let bytes = read(path).expect("the landed copy is on disk");
        write(path, corrupt_the_vector_blob(bytes)).expect("write the corrupted copy back");
    }

    /// Bytes that arrive corrupted are caught as they land, and the entry is
    /// not admitted.
    ///
    /// This is the true-positive half of the landing check. The cache file is
    /// written from these bytes and opened, so a flip anywhere the superfile's
    /// checksums cover has to surface here or nowhere: nothing downstream
    /// re-reads the file to check it.
    #[tokio::test]
    async fn a_corrupt_download_is_refused_as_it_lands() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(
            &store,
            &uri,
            corrupt_the_vector_blob(tiny_vector_superfile_bytes()),
        )
        .await;

        store
            .reader_synchronous(&uri)
            .await
            .expect_err("corrupt bytes must not be admitted to the cache");

        // And nothing is left behind under the name a later read, or a later
        // process scanning this directory, would adopt. Adoption does not
        // re-check the file, so a rejected download promoted anyway would be
        // laundered into the cache by the very next attempt.
        assert!(
            !store.cache_path(&uri).is_file(),
            "a download that failed verification must not keep the cache name"
        );
    }

    /// With verification off the same bytes are admitted, which is what makes
    /// the test above a statement about the policy rather than about some
    /// other check happening to fire.
    #[tokio::test]
    async fn never_admits_the_corrupt_download_it_does_not_check() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Never;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(
            &store,
            &uri,
            corrupt_the_vector_blob(tiny_vector_superfile_bytes()),
        )
        .await;

        store
            .reader_synchronous(&uri)
            .await
            .expect("never checks, so the corruption goes unnoticed");
    }

    /// Healthy bytes land and serve.
    ///
    /// The half that matters more. Verification that rejected good data would
    /// turn a readable table into an unreadable one, which is a worse failure
    /// than the gap it closes, and it would show up as exactly this test
    /// failing.
    #[tokio::test]
    async fn a_healthy_download_lands_and_serves() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;

        store
            .reader_synchronous(&uri)
            .await
            .expect("healthy bytes must land and serve");
    }

    /// A copy a previous run left on disk is adopted without being read back,
    /// even when it is corrupt.
    ///
    /// Deliberate, and the exposure this policy accepts by name. Adoption
    /// writes nothing, so there is no landing to check at; verifying here
    /// would mean reading every byte of every cached superfile at process
    /// start, which is the scan the policy removes. The engines this matches
    /// default that scan to off and carry the same exposure.
    #[tokio::test]
    async fn a_query_adopts_a_copy_left_by_a_previous_run_unread() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("a Load read lands the whole file on disk");

        corrupt_the_cached_copy(&store.cache_path(&uri));

        let restarted = reopen_store(&store, |cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        drop(store);
        restarted.scan_cache_root();

        let before = restarted.stats();
        // A query's intent, not a compaction's: this is the read whose cost
        // the policy is protecting.
        let _ = restarted.reader(&uri).await;
        let after = restarted.stats();
        assert_eq!(
            after.n_cold_fetches - before.n_cold_fetches,
            0,
            "auto does not look on the query path, and so must not refetch"
        );
    }

    /// A compaction input is verified even where a query would not be, and a
    /// corrupt copy is refetched instead of merged.
    ///
    /// This is the one failure the rest of the design cannot absorb.
    /// Compaction reads its inputs from this cache and writes a new superfile
    /// with fresh, valid checksums over whatever it read; once the inputs are
    /// reclaimed, that output is the only copy. A bad input adopted here is
    /// therefore not a wrong answer a refetch repairs, it is the good copy in
    /// the object store being replaced by a bad one.
    #[tokio::test]
    async fn a_compaction_input_is_verified_where_a_query_would_not_be() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("a Load read lands the whole file on disk");

        corrupt_the_cached_copy(&store.cache_path(&uri));

        let restarted = reopen_store(&store, |cfg| {
            cfg.crc_verification = CrcVerification::Auto;
        });
        drop(store);
        restarted.scan_cache_root();

        let before = restarted.stats();
        let reader = restarted
            .reader_synchronous(&uri)
            .await
            .expect("a corrupt input must not fail the compaction, it must be refetched");
        let after = restarted.stats();

        assert_eq!(
            after.n_disk_reuses - before.n_disk_reuses,
            0,
            "the corrupt copy must not be handed to a merge"
        );
        assert_eq!(
            after.n_cold_fetches - before.n_cold_fetches,
            1,
            "it must come from the authoritative copy instead"
        );
        drop(reader);
    }

    /// `never` still means never, including for a compaction input. An
    /// operator who turned verification off must not have it reappear on a
    /// path this change touched.
    #[tokio::test]
    async fn never_does_not_verify_a_compaction_input_either() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Never;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("a Load read lands the whole file on disk");

        corrupt_the_cached_copy(&store.cache_path(&uri));

        let restarted = reopen_store(&store, |cfg| {
            cfg.crc_verification = CrcVerification::Never;
        });
        drop(store);
        restarted.scan_cache_root();

        let before = restarted.stats();
        let _ = restarted.reader_synchronous(&uri).await;
        let after = restarted.stats();
        assert_eq!(
            after.n_cold_fetches - before.n_cold_fetches,
            0,
            "never must not look, on any path"
        );
    }

    /// `always` does look, and a corrupt copy left by a previous run is
    /// discarded and refetched, so the read still returns.
    ///
    /// Two things asserted together on purpose. Detecting the corruption is
    /// useless if the query then fails, and recovering is useless if the
    /// corruption went unnoticed; what a caller needs is the right answer
    /// without knowing anything happened.
    #[tokio::test]
    async fn always_rescans_a_copy_left_by_a_previous_run() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.crc_verification = CrcVerification::Always;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        store
            .reader_synchronous(&uri)
            .await
            .expect("a Load read lands the whole file on disk");

        corrupt_the_cached_copy(&store.cache_path(&uri));

        let restarted = reopen_store(&store, |cfg| {
            cfg.crc_verification = CrcVerification::Always;
        });
        drop(store);
        restarted.scan_cache_root();

        let before = restarted.stats();
        let reader = restarted
            .reader_synchronous(&uri)
            .await
            .expect("a corrupt cached copy must not fail the read");
        let after = restarted.stats();

        assert_eq!(
            after.n_disk_reuses - before.n_disk_reuses,
            0,
            "the corrupt copy must not be adopted"
        );
        assert_eq!(
            after.n_cold_fetches - before.n_cold_fetches,
            1,
            "it must be refetched from the authoritative copy instead"
        );
        drop(reader);
    }

    /// A healthy copy from a previous run is adopted under every policy, and
    /// adopting it costs no fetch. The `always` arm is the one that could
    /// regress: it reads the file back, and a false positive there would turn
    /// a restart into a full refetch of the cache.
    #[tokio::test]
    async fn a_healthy_copy_is_reused_after_a_restart_under_every_policy() {
        for policy in [
            CrcVerification::Auto,
            CrcVerification::Always,
            CrcVerification::Never,
        ] {
            let (_dir, store) = test_store_with(|cfg| {
                cfg.crc_verification = policy;
            });
            let uri = SuperfileUri::new_v4();
            put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
            store
                .reader_synchronous(&uri)
                .await
                .expect("a Load read lands the whole file on disk");

            let restarted = reopen_store(&store, |cfg| {
                cfg.crc_verification = policy;
            });
            drop(store);
            restarted.scan_cache_root();

            let before = restarted.stats();
            restarted
                .reader_synchronous(&uri)
                .await
                .expect("restart reuses the copy");
            let after = restarted.stats();

            assert_eq!(
                after.n_disk_reuses - before.n_disk_reuses,
                1,
                "{policy:?}: the healthy copy must be adopted, not refetched"
            );
            assert_eq!(
                after.n_cold_fetches - before.n_cold_fetches,
                0,
                "{policy:?}: and adopting it must cost no fetch"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use roaring::RoaringBitmap;
    use tempfile::TempDir;

    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            manifest::SuperfileUri,
            reader_cache::{
                block_source::{CACHE_BLOCK_BYTES, serialize_index},
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
        // storage object. Under either name: a complete copy or a holed one.
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        fs::write(store.cache_path(&uri), b"stale bytes").expect("write orphan");
        fs::write(store.hole_path(&uri), b"stale bytes").expect("write holed orphan");

        assert!(!store.erase_superfile_local_copy(&uri), "no entry to drop");
        assert!(
            !store.cache_path(&uri).exists() && !store.hole_path(&uri).exists(),
            "orphan files still unlinked"
        );
        assert_eq!(store.stats().n_gc_drops, 0);
    }

    /// The open-time scan counts a holed copy like any cache file, and drops it when a complete
    /// copy of the same superfile is there too.
    #[tokio::test]
    async fn restart_scan_keeps_one_local_copy_per_superfile() {
        let (_dir, store) = test_store();
        let bytes = tiny_superfile_bytes();
        let (both, holed) = (SuperfileUri::new_v4(), SuperfileUri::new_v4());
        let complete = seed_cache_file(&store, &both, &bytes);
        fs::write(store.hole_path(&both), &bytes).expect("stale holed copy");
        fs::write(store.hole_path(&holed), &bytes).expect("holed copy");

        let opened = reopen_store(&store, |_| {});

        assert!(opened.cache_path(&both).exists(), "the complete copy stays");
        assert!(
            !opened.hole_path(&both).exists(),
            "and wins over the holed one"
        );
        assert!(
            opened.hole_path(&holed).exists(),
            "a lone holed copy is kept"
        );
        assert_eq!(
            opened.stats().current_bytes,
            complete + bytes.len() as u64,
            "both kept copies are counted"
        );
        opened.assert_budget_consistent();
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

    /// An index that names another file (left by a generation that lost a rename
    /// race to a newer file at the path) is not trusted at restart: both files
    /// go and nothing is charged.
    #[tokio::test]
    async fn restart_scan_deletes_a_block_file_whose_index_names_another_file() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        seed_valid_block_file(&store, &uri, 2 * CACHE_BLOCK_BYTES, &[0]);
        let ino = fs::metadata(store.blocks_path(&uri)).expect("blocks").ino();
        let stale = serialize_index(&RoaringBitmap::from_iter([0]), ino + 1);
        fs::write(store.blocks_idx_path(&uri), stale).expect("idx");

        let reopened = reopen_store(&store, |_| {});

        assert!(!reopened.blocks_path(&uri).exists());
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

    /// A block file an older build indexed is deleted by the restart scan and
    /// not counted, whatever block size the index names: such an index names no
    /// file, so it may describe an earlier file at the path.
    #[tokio::test]
    async fn restart_scan_deletes_a_block_file_an_older_build_indexed() {
        for block in OLDER_BUILD_BLOCK_BYTES {
            let (_dir, store) = test_store();
            let uri = SuperfileUri::new_v4();
            fs::File::create(store.blocks_path(&uri))
                .expect("blocks")
                .set_len(2 * block)
                .expect("set_len");
            fs::write(store.blocks_idx_path(&uri), unstamped_index(block, &[0, 1])).expect("idx");

            let reopened = reopen_store(&store, |_| {});
            assert!(!reopened.blocks_path(&uri).exists(), "{block}-byte blocks");
            assert!(!reopened.blocks_idx_path(&uri).exists());
            assert_eq!(reopened.stats().current_bytes, 0);
        }
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
        // cold-fetch a clean copy. Nothing is ever served from a file that failed to open, under
        // either name.
        let bytes = tiny_superfile_bytes();
        for holed in [false, true] {
            for (label, seeded) in [
                ("zero length", Bytes::new()),
                ("not a superfile", Bytes::from_static(b"garbage bytes")),
            ] {
                let (_dir, store) = test_store();
                let uri = SuperfileUri::new_v4();
                put_superfile(&store, &uri, bytes.clone()).await;
                let bad = if holed {
                    store.hole_path(&uri)
                } else {
                    store.cache_path(&uri)
                };
                fs::create_dir_all(&store.config.cache_root).expect("cache root");
                fs::write(&bad, &seeded).expect("seed bad copy");

                let reader = store.reader(&uri).await;
                assert!(
                    reader.is_ok(),
                    "{label}, holed {holed}: falls through and serves"
                );
                assert_eq!(
                    store.stats().n_cold_fetches,
                    1,
                    "{label}, holed {holed}: came from storage, not the bad local file"
                );
                assert!(
                    !bad.exists(),
                    "{label}, holed {holed}: the bad copy is deleted"
                );
            }
        }
    }

    #[tokio::test]
    async fn scan_reclaims_stale_tmp_files_and_spares_fresh_ones() {
        // A stale tempfile can only be a crashed fetch's leftover, but a fresh one belongs to a
        // sibling process's in-flight fetch: deleting it would fail that fetch's rename.
        let (_dir, store) = test_store();
        let bytes = tiny_superfile_bytes();
        let now = SystemTime::now();
        let stale = store.tmp_path(&SuperfileUri::new_v4());
        let fresh = store.tmp_path(&SuperfileUri::new_v4());
        let skewed = store.tmp_path(&SuperfileUri::new_v4());
        // The unnumbered name older builds wrote is reclaimed by the same rule.
        let legacy = store
            .config
            .cache_root
            .join(format!("{}.tmp", SuperfileUri::new_v4().cache_filename()));
        for (path, mtime) in [
            (&stale, now - TMP_RECLAIM_AGE * 2),
            (&fresh, now),
            // A future mtime (clock skew) must read as not-stale, never as reclaimable.
            (&skewed, now + TMP_RECLAIM_AGE),
            (&legacy, now - TMP_RECLAIM_AGE * 2),
        ] {
            fs::write(path, bytes.as_ref()).expect("seed tmp");
            fs::File::options()
                .write(true)
                .open(path)
                .expect("open tmp")
                .set_modified(mtime)
                .expect("set mtime");
        }

        let opened = reopen_store(&store, |_| {});

        assert!(!stale.exists(), "stale tmp reclaimed");
        assert!(fresh.exists(), "fresh tmp left for its owner");
        assert!(skewed.exists(), "future mtime spared under clock skew");
        assert!(
            !legacy.exists(),
            "a stale unnumbered tmp from an older build is reclaimed"
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
        let sibling_tmp = store.tmp_path(&uri);
        fs::write(&sibling_tmp, bytes.as_ref()).expect("sibling's in-flight tmp");

        let opened = reopen_store(&store, |_| {});
        assert!(sibling_tmp.exists(), "scan spared the in-flight tmp");

        // The sibling finishes: fsync'd bytes, atomic rename to the final name.
        fs::rename(&sibling_tmp, opened.cache_path(&uri)).expect("sibling renames");

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
            .fetch_from_disk_cache(
                &uri,
                &uri.storage_path(),
                Some(wrong),
                ReadIntent::Warm,
                None,
            )
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
