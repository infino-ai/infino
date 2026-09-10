// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-budget accounting: reserve space before a fetch, evict cold entries to
//! make room, and the idle-page sweeps that trim resident memory.

use std::{
    fs,
    sync::{Arc, atomic::Ordering},
};

use memmap2::{Mmap, UncheckedAdvice};

use crate::supertable::{
    manifest::SuperfileUri,
    reader_cache::{config::EvictionCandidate, disk::*},
};

impl DiskCacheStore {
    /// Run one pass of the `MADV_DONTNEED` sweep against
    /// currently-cached entries. Each entry with
    /// `now - last_access_us > mmap_cold_threshold_secs * 1e6`
    /// gets `madvise(MADV_DONTNEED)` on its mmap; pages
    /// re-fault on next read (cheap on SSD-backed page cache).
    ///
    /// Exposed for explicit invocation from tests so they
    /// don't have to sleep for the sweep cadence. The
    /// background thread calls this on each tick.
    ///
    /// Iteration safety: snapshots `(uri, mmap_arc,
    /// last_access)` tuples into a Vec, drops the DashMap
    /// iterator (releasing shard guards), then `madvise`s.
    /// Holding shard guards through `madvise` would block
    /// eviction during the sweep — `madvise` on a multi-GB
    /// mmap can take milliseconds.
    pub fn sweep_once(&self) -> u64 {
        let threshold_us = self
            .config
            .mmap_cold_threshold_secs
            .saturating_mul(1_000_000);
        let now_us = self.now_us();
        // Snapshot: clone the Arc<Mmap> + last-access into an
        // owned Vec, then drop the iterator.
        let snapshot: Vec<(SuperfileUri, Arc<Mmap>, u64)> = self
            .cached
            .iter()
            .filter_map(|e| {
                let mmap = e.value().mmap.clone()?;
                let last = e.value().last_access_us.load(Ordering::Acquire);
                Some((*e.key(), mmap, last))
            })
            .collect();
        let mut n_advised = 0u64;
        for (_uri, mmap, last_access) in snapshot {
            let idle = now_us.saturating_sub(last_access);
            if idle >= threshold_us {
                // `MADV_DONTNEED` lives on `UncheckedAdvice` in
                // memmap2 because it's unsafe for *writable*
                // mappings (pages truly freed → re-reads see
                // zero-filled). For our **read-only** mappings
                // it's safe: dropped pages re-fault from the
                // backing file on next access. The cache files
                // are immutable once written + we never write
                // to the mmap, so the read-back is bit-identical.
                //
                // Errors are non-fatal — typically platform
                // limitations on macOS/BSD; we just skip.
                //
                // SAFETY: the mmap is read-only and the backing
                // file is immutable for the lifetime of this
                // mapping; pages dropped by `MADV_DONTNEED`
                // re-fault from disk on next read.
                let _ = unsafe { mmap.unchecked_advise(UncheckedAdvice::DontNeed) };
                n_advised += 1;
            }
        }
        if n_advised > 0 {
            self.n_madvise_calls.fetch_add(n_advised, Ordering::AcqRel);
        }
        n_advised
    }

    /// Sum of mmap virtual sizes across all cached entries
    /// with an active mapping. This is the **upper bound**
    /// on the cache's resident memory — actual RSS is some
    /// subset (only pages that have been faulted in and not
    /// yet `madvise(MADV_DONTNEED)`'d by a sweep). Used by
    /// [`crate::supertable::Supertable::stats`] to
    /// report `mmap_resident_bytes` and to drive the
    /// budget-aware sweep in [`Self::sweep_for_budget`].
    pub fn current_mmap_size_bytes(&self) -> u64 {
        self.cached
            .iter()
            .filter_map(|e| e.value().mmap.as_ref().map(|m| m.len() as u64))
            .sum()
    }

    /// drop mmap pages until the cache's working set
    /// is back under `budget_bytes`. No-op if already under
    /// budget. Returns the number of entries that received
    /// `madvise(MADV_DONTNEED)`.
    ///
    /// Policy: iterate entries by ascending `last_access_us`
    /// (oldest first); `madvise` each one until the
    /// projected residency drops below the budget. Entries
    /// stay in the cache map — pages re-fault from the
    /// backing file on next access. The on-disk cache and
    /// `disk_budget_bytes` are unchanged; only the RSS
    /// footprint is affected.
    ///
    /// Pinned URIs are NOT skipped here: pinning protects
    /// against EVICTION (entry removal + file unlink), not
    /// against page reclaim. A pinned entry whose pages
    /// have been madvise'd re-faults on next access and
    /// behaves correctly; the cost is one re-fault per
    /// re-touched page.
    pub fn sweep_for_budget(&self, budget_bytes: u64) -> u64 {
        let mut total = self.current_mmap_size_bytes();
        if total <= budget_bytes {
            return 0;
        }
        // Snapshot candidates: (uri, mmap_arc, last_access,
        // size). Drop the iterator before madvise calls so
        // we don't hold shard guards across the syscall.
        let mut candidates: Vec<(SuperfileUri, Arc<Mmap>, u64, u64)> = self
            .cached
            .iter()
            .filter_map(|e| {
                let mmap = e.value().mmap.clone()?;
                Some((
                    *e.key(),
                    mmap,
                    e.value().last_access_us.load(Ordering::Acquire),
                    e.value().size_bytes.load(Ordering::Acquire),
                ))
            })
            .collect();
        // Oldest-first.
        candidates.sort_by_key(|(_, _, last, _)| *last);

        let mut n_advised = 0u64;
        for (_uri, mmap, _last, size) in candidates {
            if total <= budget_bytes {
                break;
            }
            // SAFETY: the mmap is read-only and the backing
            // file is immutable for the mapping's lifetime;
            // pages dropped by MADV_DONTNEED re-fault from
            // disk on next read. Identical safety argument
            // to the `sweep_once` path; see that fn for the
            // full discussion.
            let _ = unsafe { mmap.unchecked_advise(UncheckedAdvice::DontNeed) };
            self.n_madvise_calls.fetch_add(1, Ordering::AcqRel);
            total = total.saturating_sub(size);
            n_advised += 1;
        }
        n_advised
    }

    /// Evict never-opened files (whole-file copies, then retained block files) to
    /// bring `current_bytes` back under budget. Only cold candidates are freed;
    /// live cached entries are left to the async reservation path. Runs on paths
    /// that add already-on-disk bytes without a reservation (restart scan, block
    /// adoption), so a read-only reopen still self-corrects to budget.
    pub(crate) fn trim_cold_to_budget(&self) {
        let budget = self.disk_budget_bytes();
        let cur = self.current_bytes.load(Ordering::Acquire);
        if cur <= budget {
            return;
        }
        let over = cur - budget;
        let freed = self.evict_unindexed(over);
        if freed < over {
            self.evict_block_files(over - freed);
        }
    }

    /// Delete never-opened files, oldest first, until `bytes_needed` is freed. Returns the bytes
    /// freed. These go before live entries: nothing holds them, so deleting one is safe and cheap.
    pub(crate) fn evict_unindexed(&self, bytes_needed: u64) -> u64 {
        // Usually empty after warmup; skip the snapshot allocation.
        if self.unindexed.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(SuperfileUri, u64, u64)> = self
            .unindexed
            .iter()
            .map(|e| (*e.key(), e.value().size_bytes, e.value().mtime_us))
            .collect();

        candidates.sort_by_key(|(_, _, mtime_us)| *mtime_us);

        let mut freed: u64 = 0;
        for (uri, size, _) in candidates {
            if freed >= bytes_needed {
                break;
            }

            // Only the winner of the remove deletes and subtracts, so two concurrent evictions
            // cannot double-count.
            if self.unindexed.remove(&uri).is_some() {
                let _ = fs::remove_file(self.cache_path(&uri));
                self.drop_block_file(&uri);
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed += size;
            }
        }

        freed
    }

    pub(crate) fn evict_block_files(&self, bytes_needed: u64) -> u64 {
        if self.block_files.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(SuperfileUri, u64, u64)> = self
            .block_files
            .iter()
            .map(|e| (*e.key(), e.value().size_bytes, e.value().mtime_us))
            .collect();

        candidates.sort_by_key(|(_, _, mtime_us)| *mtime_us);

        let mut freed: u64 = 0;
        for (uri, size, _) in candidates {
            if freed >= bytes_needed {
                break;
            }

            if self.block_files.remove(&uri).is_some() {
                let _ = fs::remove_file(self.blocks_path(&uri));
                let _ = fs::remove_file(self.blocks_idx_path(&uri));
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed += size;
            }
        }

        freed
    }

    /// Same as [`Self::reserve`] but returns just the
    /// reserved-bytes count instead of a borrow-lifetimed
    /// guard. Caller is responsible for either committing
    /// (no-op — the bytes stay reserved as part of a cached
    /// entry) or rolling back via
    /// `self.current_bytes.fetch_sub(bytes, Release)` on
    /// failure. Used by the hybrid cold-fetch path where the
    /// reservation outlives the borrow on `&self` via a
    /// `tokio::spawn`-ed background finalizer.
    pub(crate) async fn reserve_manual(&self, bytes: u64) -> Result<(), DiskCacheError> {
        loop {
            let budget = self.disk_budget_bytes();
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= budget {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + bytes, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(());
                }
                continue;
            }
            let needed = (cur + bytes).saturating_sub(budget);
            self.evict_at_least(needed).await?;
        }
    }

    /// Reserve bytes for block-cache growth.
    pub(crate) async fn reserve_block_bytes(&self, bytes: u64) -> Result<(), DiskCacheError> {
        self.reserve_manual(bytes).await
    }

    /// Release previously reserved block-cache bytes.
    pub(crate) fn release_block_bytes(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Release);
    }

    /// True when `token` still identifies the live lazy entry for `uri`.
    pub(crate) fn lazy_block_entry_is_current(&self, uri: &SuperfileUri, token: &Arc<()>) -> bool {
        self.cached
            .get(uri)
            .and_then(|entry| {
                entry
                    .block_token
                    .as_ref()
                    .map(|current| Arc::ptr_eq(current, token))
            })
            .unwrap_or(false)
    }

    /// Release accounting for one removed cache entry.
    pub(crate) fn release_entry_accounting(&self, entry: &CachedEntry) {
        if entry.accounting == EntryAccounting::Eager {
            self.current_bytes
                .fetch_sub(entry.size_bytes.load(Ordering::Acquire), Ordering::Release);
        }
    }

    #[cfg(test)]
    pub(crate) fn install_block_entry_for_test(
        &self,
        uri: SuperfileUri,
        filled: Arc<AtomicU64>,
        block_token: Arc<()>,
    ) {
        let reader = SuperfileReader::open(
            crate::supertable::reader_cache::disk::test_support::tiny_superfile_bytes(),
        )
        .expect("tiny superfile opens");
        self.cached.insert(
            uri,
            Arc::new(CachedEntry {
                reader: Arc::new(reader),
                mmap: None,
                size_bytes: filled,
                accounting: EntryAccounting::SourceOwned,
                block_token: Some(block_token),
                block_source: None,
                fill_spawned: AtomicBool::new(false),
                last_access_us: AtomicU64::new(self.now_us()),
            }),
        );
    }

    #[cfg(test)]
    pub(crate) fn remove_block_entry_for_test(&self, uri: &SuperfileUri) {
        let _ = self.cached.remove(uri);
    }

    /// Reserve `bytes` of disk budget via CAS-loop on
    /// `current_bytes`. On budget pressure runs eviction;
    /// retries until either reserved or `BudgetExceeded`.
    pub(crate) async fn reserve(&self, bytes: u64) -> Result<Reservation<'_>, DiskCacheError> {
        loop {
            let budget = self.disk_budget_bytes();
            let cur = self.current_bytes.load(Ordering::Acquire);
            if cur + bytes <= budget {
                if self
                    .current_bytes
                    .compare_exchange_weak(cur, cur + bytes, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    return Ok(Reservation {
                        store: self,
                        bytes,
                        committed: false,
                    });
                }
                // Lost the race; another reservation slipped
                // in. Re-read and retry — most of the time
                // there's still room.
                continue;
            }
            // Over budget — try eviction. If eviction frees
            // enough, the next loop iteration's CAS will
            // succeed.
            let needed = (cur + bytes).saturating_sub(budget);
            self.evict_at_least(needed).await?;
        }
    }

    /// Drive the eviction policy until either `bytes_needed`
    /// is freed or no eligible victims remain (→
    /// `BudgetExceeded`).
    pub(crate) async fn evict_at_least(&self, bytes_needed: u64) -> Result<(), DiskCacheError> {
        // Never-opened files go first; freeing one cannot hurt a live reader.
        let freed = self.evict_unindexed(bytes_needed);
        if freed >= bytes_needed {
            return Ok(());
        }
        let freed = freed + self.evict_block_files(bytes_needed - freed);
        if freed >= bytes_needed {
            return Ok(());
        }
        let bytes_needed = bytes_needed - freed;

        // Clone the current pinned_fn out of the mutex
        // before invoking it — the closure itself may
        // acquire other locks (e.g., the supertable's
        // manifest ArcSwap), and holding the cache's
        // pinned_fn mutex across that call invites
        // deadlocks.
        let pinned_fn = {
            let g = self.pinned_fn.lock().expect("pinned_fn mutex poisoned");
            Arc::clone(&g)
        };
        let pinned = pinned_fn();
        let candidates: Vec<EvictionCandidate> = self
            .cached
            .iter()
            .map(|e| EvictionCandidate {
                uri: *e.key(),
                size_bytes: e.value().size_bytes.load(Ordering::Acquire),
                last_access_us: e.value().last_access_us.load(Ordering::Acquire),
            })
            .collect();
        let victims = self
            .config
            .eviction
            .select_for_eviction(&candidates, &pinned, bytes_needed);
        if victims.is_empty() {
            // Error only when nothing at all was freed; unindexed files may have covered part of
            // the request.
            if freed > 0 {
                return Ok(());
            }
            return Err(DiskCacheError::BudgetExceeded);
        }
        for uri in victims {
            // Atomic gate against concurrent eviction: only
            // the caller that wins `DashMap::remove` runs
            // unlink + decrement. Without this gate, two
            // reservations evicting the same victim could
            // double-decrement current_bytes.
            if let Some((_, entry)) = self.cached.remove(&uri) {
                let path = self.cache_path(&uri);
                let _ = fs::remove_file(&path);
                let _ = fs::remove_file(self.blocks_path(&uri));
                let _ = fs::remove_file(self.blocks_idx_path(&uri));
                self.release_entry_accounting(&entry);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
            }
        }
        Ok(())
    }
}

/// RAII guard for a disk-budget reservation. Drop without
/// `commit()` releases the reserved bytes back to the pool —
/// the caller's reservation never lands.
pub(crate) struct Reservation<'a> {
    store: &'a DiskCacheStore,
    bytes: u64,
    committed: bool,
}

impl<'a> Reservation<'a> {
    pub(crate) fn commit(mut self) {
        self.committed = true;
    }
}

impl<'a> Drop for Reservation<'a> {
    fn drop(&mut self) {
        if !self.committed {
            self.store
                .current_bytes
                .fetch_sub(self.bytes, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::supertable::{
        manifest::SuperfileUri,
        reader_cache::{
            block_source::CACHE_BLOCK_BYTES,
            disk::{budget::*, test_support::*},
        },
    };

    #[tokio::test]
    async fn auto_budget_is_raised_and_admits_previously_oversized_entry() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
        });
        store.mark_budget_auto_sized();
        // Undersized: the tiny superfile cannot be admitted.
        let uri = SuperfileUri::new_v4();
        let err = store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect_err("undersized budget must reject");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));

        // Reconcile raises the auto-sized budget; the same insert succeeds.
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_RAISED_FLOOR_BYTES);
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("raised budget admits the entry");

        // Raise-only: a smaller floor later never lowers the budget.
        store.reconcile_budget_floor(TEST_TINY_BUDGET_BYTES, TEST_TINY_BUDGET_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_RAISED_FLOOR_BYTES);
    }

    #[tokio::test]
    async fn explicit_budget_is_never_changed_by_reconcile() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
        });
        // No mark_budget_auto_sized(): the budget is explicit. Reconcile
        // must warn (once) but leave the budget verbatim.
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        store.reconcile_budget_floor(TEST_RAISED_FLOOR_BYTES, TEST_RAISED_FLOOR_BYTES);
        assert_eq!(store.disk_budget_bytes(), TEST_TINY_BUDGET_BYTES);
        assert_eq!(store.stats().budget_bytes, TEST_TINY_BUDGET_BYTES);
    }

    #[tokio::test]
    async fn retained_block_files_are_evicted_under_budget_pressure() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        let size = 3 * CACHE_BLOCK_BYTES;
        let filled = seed_valid_block_file(&store, &uri, size, &[0, 1, 2]);

        let reopened = reopen_store(&store, |cfg| cfg.disk_budget_bytes = filled);
        assert_eq!(reopened.stats().current_bytes, filled);

        reopened.evict_at_least(filled).await.expect("evict");
        assert_eq!(
            reopened.stats().current_bytes,
            0,
            "the retained block file is evicted to reclaim budget"
        );
        assert!(!reopened.blocks_path(&uri).exists());
        assert!(!reopened.blocks_idx_path(&uri).exists());
    }

    #[tokio::test]
    async fn reopen_trims_retained_block_files_to_budget() {
        let (_dir, store) = test_store();
        let mut total = 0;
        for _ in 0..4 {
            let uri = SuperfileUri::new_v4();
            total += seed_valid_block_file(&store, &uri, 3 * CACHE_BLOCK_BYTES, &[0, 1, 2]);
        }

        let budget = total / 2;
        let reopened = reopen_store(&store, |cfg| cfg.disk_budget_bytes = budget);
        assert!(
            reopened.stats().current_bytes <= budget,
            "a read-only reopen trims retained blocks to budget at scan, current={} budget={budget}",
            reopened.stats().current_bytes
        );
    }

    // ----- lazy open: cost scales with the working set, not the directory -----

    #[tokio::test]
    async fn over_budget_directory_still_serves_the_first_read() {
        // Seeding the budget from the directory while the index is empty is only safe if eviction
        // can reclaim files no entry owns. Without that, `reserve` finds a full budget and zero
        // eviction candidates, and the first read fails with BudgetExceeded instead of serving.
        let bytes = tiny_superfile_bytes();
        let one = bytes.len() as u64;
        let (_dir, store) = test_store_with(|cfg| {
            // Room for a single superfile, so a directory of several is over budget.
            cfg.disk_budget_bytes = one;
        });
        for _ in 0..SEEDED_CACHE_FILES {
            seed_cache_file(&store, &SuperfileUri::new_v4(), &bytes);
        }

        let wanted = SuperfileUri::new_v4();
        put_superfile(&store, &wanted, bytes.clone()).await;

        let opened = reopen_store(&store, |cfg| cfg.disk_budget_bytes = one);

        let reader = opened.reader(&wanted).await;
        assert!(
            reader.is_ok(),
            "first read on an over-budget directory must serve, got {:?}",
            reader.err()
        );
        assert!(
            opened.stats().current_bytes <= one,
            "eviction brought the directory back under budget"
        );
    }

    #[tokio::test]
    async fn eviction_takes_unindexed_files_before_live_entries() {
        // Budget pressure with both kinds present: the files no read has opened go first, so an
        // entry a query just adopted keeps its mapping.
        let bytes = tiny_superfile_bytes();
        let one = bytes.len() as u64;
        let (_dir, store) = test_store();
        let adopted = SuperfileUri::new_v4();
        seed_cache_file(&store, &adopted, &bytes);
        for _ in 0..2 {
            seed_cache_file(&store, &SuperfileUri::new_v4(), &bytes);
        }

        // Fresh store over the same root: three unindexed files, a budget with no slack.
        let opened = reopen_store(&store, |cfg| cfg.disk_budget_bytes = 3 * one);
        let _held = opened.reader(&adopted).await.expect("adopt one file");

        // A warm insert needs one file's worth of room; eviction must find it among the two
        // never-opened files, not under the adopted entry.
        opened
            .insert_warm(&SuperfileUri::new_v4(), bytes.clone())
            .await
            .expect("insert under pressure");

        let s = opened.stats();
        assert!(
            opened.cached.contains_key(&adopted),
            "adopted entry survives"
        );
        assert!(
            opened.cache_path(&adopted).is_file(),
            "its file survives too"
        );
        assert_eq!(s.n_evictions, 1, "one unindexed file made the room");
        assert!(s.current_bytes <= 3 * one, "back under budget");
    }

    #[tokio::test]
    async fn cold_fetch_evicts_lru_when_over_budget() {
        // Budget fits ~1.5 entries, forcing eviction of the older one
        // when the second cold fetch reserves.
        let one = tiny_superfile_bytes();
        let entry_size = one.len() as u64;
        let (_dir, store) = test_store_with(move |cfg| {
            cfg.disk_budget_bytes = entry_size + entry_size / 2;
        });

        let uri_a = SuperfileUri::new_v4();
        let uri_b = SuperfileUri::new_v4();
        put_superfile(&store, &uri_a, tiny_superfile_bytes()).await;
        put_superfile(&store, &uri_b, tiny_superfile_bytes()).await;

        store.reader_synchronous(&uri_a).await.expect("a");
        store.reader_synchronous(&uri_b).await.expect("b");

        // a was the LRU victim; b is resident.
        assert_eq!(store.stats().n_evictions, 1);
        assert!(store.cached.contains_key(&uri_b));
        assert!(!store.cached.contains_key(&uri_a));
        // a's cache file was unlinked.
        assert!(!store.cache_path(&uri_a).exists());
        assert_eq!(store.stats().current_bytes, entry_size);
    }

    #[tokio::test]
    async fn cold_fetch_budget_exceeded_with_all_pinned() {
        let one = tiny_superfile_bytes();
        let entry_size = one.len() as u64;
        let (_dir, store) = test_store_with(move |cfg| {
            cfg.disk_budget_bytes = entry_size + entry_size / 2;
        });

        let uri_a = SuperfileUri::new_v4();
        let uri_b = SuperfileUri::new_v4();
        put_superfile(&store, &uri_a, tiny_superfile_bytes()).await;
        put_superfile(&store, &uri_b, tiny_superfile_bytes()).await;

        // First fetch lands.
        store.reader_synchronous(&uri_a).await.expect("a");
        // Pin everything so eviction finds no victims.
        store.set_pinned_fn(Arc::new(move || {
            let mut s = HashSet::new();
            s.insert(uri_a);
            s
        }));
        let err = store
            .reader_synchronous(&uri_b)
            .await
            .expect_err("no eligible victims");
        assert!(matches!(err, DiskCacheError::BudgetExceeded));
        // a stays put; budget unchanged.
        assert!(store.cached.contains_key(&uri_a));
    }

    // ----- sweep_once / sweep_for_budget / madvise counters -----

    #[tokio::test]
    async fn sweep_once_advises_idle_mmap_entries() {
        // threshold 0 means every entry is immediately "idle".
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        let advised = store.sweep_once();
        assert_eq!(advised, 1);
        assert_eq!(store.stats().n_madvise_calls, 1);
        // A second sweep advises again (counter accumulates).
        assert_eq!(store.sweep_once(), 1);
        assert_eq!(store.stats().n_madvise_calls, 2);
    }

    #[tokio::test]
    async fn sweep_once_skips_when_threshold_not_reached() {
        // Large threshold → nothing is idle, so no madvise.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.mmap_cold_threshold_secs = 1_000_000;
        });
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        assert_eq!(store.sweep_once(), 0);
        assert_eq!(store.stats().n_madvise_calls, 0);
    }

    #[tokio::test]
    async fn sweep_for_budget_noop_under_budget() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        // budget far above resident size → no madvise.
        assert_eq!(store.sweep_for_budget(u64::MAX), 0);
        assert_eq!(store.stats().n_madvise_calls, 0);
    }

    #[tokio::test]
    async fn sweep_for_budget_reclaims_oldest_first() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        store
            .insert_warm(&uri, tiny_superfile_bytes())
            .await
            .expect("warm");
        let resident = store.current_mmap_size_bytes();
        assert!(resident > 0);
        // budget 0 forces every entry to be advised.
        let advised = store.sweep_for_budget(0);
        assert_eq!(advised, 1);
        assert_eq!(store.stats().n_madvise_calls, 1);
    }

    #[tokio::test]
    async fn current_mmap_size_bytes_zero_when_empty() {
        let (_dir, store) = test_store();
        assert_eq!(store.current_mmap_size_bytes(), 0);
    }

    // ----- error type conversions / Debug -----
}
