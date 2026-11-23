// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Disk-budget accounting: reserve space before a fetch, evict cold entries to
//! make room, and the idle-page sweeps that trim resident memory.

#[cfg(test)]
use std::time::Duration;
use std::{
    fs, mem,
    path::Path,
    sync::{Arc, atomic::Ordering},
};

use dashmap::mapref::entry::Entry;
use memmap2::{Mmap, UncheckedAdvice};
use tokio::sync::OwnedSemaphorePermit;
#[cfg(test)]
use tokio::time::timeout;

#[cfg(test)]
use crate::{
    memory::ConnectionMemoryBudget,
    supertable::reader_cache::{
        block_source::BlockCachedSource,
        disk::test_support::{cache_entry, tiny_superfile_bytes},
    },
};
use crate::{
    superfile::SuperfileReader,
    supertable::{
        manifest::SuperfileUri,
        reader_cache::{config::EvictionCandidate, disk::*},
    },
};

/// How long a test waits for queued block writes to finish, before failing instead of hanging.
#[cfg(test)]
const BLOCK_WRITES_SETTLE_DEADLINE: Duration = Duration::from_secs(10);

/// Cached files the budget of a temp-file charge test holds exactly.
#[cfg(test)]
const SCRATCH_TEST_BUDGET_FILES: u64 = 4;

/// Bytes the parked temp-file charge test charges.
#[cfg(test)]
const SCRATCH_TEST_PARKED_BYTES: u64 = 1024;

/// The metadata cache's own limit in the test of its charge on the disk budget: large, so
/// only the disk budget decides what it holds.
#[cfg(test)]
const METADATA_TEST_BUDGET_BYTES: u64 = 1 << 30;

/// Share of the disk budget a table's temp files beside the cache may be charged, in divisor
/// form (`budget / this`): the cache keeps at least the other half.
const SCRATCH_SHARE_DIVISOR: u64 = 2;

/// Room [`DiskCacheStore::try_reserve_ahead`] reserved for a download ahead of reads, given back
/// on drop once the download ends.
pub(crate) struct AheadRoom {
    store: Arc<DiskCacheStore>,
    bytes: u64,
}

impl Drop for AheadRoom {
    fn drop(&mut self) {
        self.store
            .ahead_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

impl DiskCacheStore {
    /// Whether dropping `uri`'s `entry` would actually release memory: it must be a
    /// lazy entry (no mmap — its cost is anonymous heap that `madvise` cannot
    /// reclaim) and the cache must be its last holder (a reader a query still
    /// has in hand keeps the heap alive regardless, and re-opening costs that
    /// query a round trip). An entry whose block source has unfinished writes is
    /// kept too (see `BlockCachedSource::has_unfinished_writes`), and so is one
    /// whose file is downloading: its Warm reservation covers the download. That
    /// keeps at most one extra entry per download slot.
    pub(super) fn reclaimable_lazy_entry(&self, uri: &SuperfileUri, entry: &CachedEntry) -> bool {
        entry.mmap().is_none()
            && !entry.held_by_query()
            && entry
                .block_source()
                .is_none_or(|source| !source.has_unfinished_writes())
            && !self
                .fills
                .get(uri)
                .is_some_and(|fill| fill.downloading.load(Ordering::Acquire))
    }

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
                let mmap = e.value().mmap().cloned()?;
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
        self.release_idle_paged_entries(threshold_us, now_us);
        n_advised
    }

    /// Drop idle entries that hold no mmap: lazy (`Residency::Paged`) ones,
    /// and `Buffered` ones a hybrid fetch has not finalized yet.
    ///
    /// Their cost is anonymous heap, which `madvise` cannot touch, so the
    /// mmap sweep above sees nothing to do for them and their memory
    /// accumulates with every superfile a process has ever opened.
    /// Dropping the entry releases it: the cache holds the last `Arc`,
    /// and the next reader for that URI re-opens. Its budget is released
    /// with it, since a Warm lazy or buffered entry is charged its full
    /// size at open. A background download still waiting for its slot goes on
    /// without the entry.
    ///
    /// Only entries the cache alone holds are dropped. A reader a query
    /// is still using has a live `Arc`, so removing it from the map would
    /// free nothing and cost the next query a re-open; those are left
    /// alone and revisited on the next tick. Promotion is the better
    /// outcome for a busy entry and is what the background fill does, so
    /// in practice this reclaims the ones promotion never reached.
    fn release_idle_paged_entries(&self, threshold_us: u64, now_us: u64) {
        let idle: Vec<SuperfileUri> = self
            .cached
            .iter()
            .filter(|e| {
                self.reclaimable_lazy_entry(e.key(), e.value())
                    && now_us.saturating_sub(e.value().last_access_us.load(Ordering::Acquire))
                        >= threshold_us
            })
            .map(|e| *e.key())
            .collect();
        for uri in idle {
            self.release_lazy_entry(&uri, now_us);
        }
    }

    /// Set how many unheld lazy entries a cold open keeps: the table's
    /// `superfiles_in_flight`.
    pub(crate) fn set_lazy_entry_cap(&self, cap: usize) {
        self.lazy_entry_cap.store(cap.max(1), Ordering::Relaxed);
    }

    /// Drop the oldest lazy entries no query holds while there are more than
    /// the table's queries work on at once (its `superfiles_in_flight`, by
    /// default sized from the host's memory), without waiting for the idle
    /// sweep.
    ///
    /// A query that touches thousands of superfiles opens a lazy reader for
    /// each, and each finished one stays in the map until the next sweep
    /// tick: its parsed footer and page index, several MB of heap on a large
    /// file, times every superfile the query read. Capping the unheld ones
    /// bounds that by the cap rather than by the table. Returns the released readers, for the
    /// caller to free where it chooses.
    pub(crate) fn release_unheld_lazy_entries_over_cap(&self) -> Vec<Arc<SuperfileReader>> {
        let cap = self.lazy_entry_cap.load(Ordering::Relaxed);
        if self.cached.len() <= cap {
            return Vec::new();
        }
        let mut unheld: Vec<(u64, SuperfileUri)> = self
            .cached
            .iter()
            .filter(|e| self.reclaimable_lazy_entry(e.key(), e.value()))
            .map(|e| (e.value().last_access_us.load(Ordering::Acquire), *e.key()))
            .collect();
        let Some(over) = unheld.len().checked_sub(cap) else {
            return Vec::new();
        };
        unheld.sort_unstable();
        let now_us = self.now_us();
        unheld
            .into_iter()
            .take(over)
            .filter_map(|(_, uri)| self.release_lazy_entry(&uri, now_us))
            .collect()
    }

    /// Remove `uri`'s entry if it is still a lazy one only the cache holds,
    /// releasing its budget and keeping its block file. Returns its reader, for
    /// the caller to free where it chooses.
    pub(super) fn release_lazy_entry(
        &self,
        uri: &SuperfileUri,
        now_us: u64,
    ) -> Option<Arc<SuperfileReader>> {
        // Re-check under the shard guard: a query may have taken a
        // reference since the snapshot, in which case dropping the
        // entry would free nothing and cost that query a re-open.
        if let Some((_, removed)) = self
            .cached
            .remove_if(uri, |uri, entry| self.reclaimable_lazy_entry(uri, entry))
        {
            self.release_entry_accounting(&removed);
            if let Some(source) = removed.block_source() {
                self.retain_block_file(uri, source, now_us);
            }
            return Some(Arc::clone(&removed.reader));
        }
        None
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
            .filter_map(|e| e.value().mmap().map(|m| m.len() as u64))
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
                let mmap = e.value().mmap().cloned()?;
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
        let freed = self.evict_unindexed(over, None);
        if freed < over {
            self.evict_block_files(over - freed, None);
        }
    }

    /// Delete never-opened files, oldest first, until `bytes_needed` is freed. Returns the bytes
    /// freed. These go before live entries: nothing holds them, so deleting one is safe and cheap.
    pub(crate) fn evict_unindexed(&self, bytes_needed: u64, keep: Option<&SuperfileUri>) -> u64 {
        // Usually empty after warmup; skip the snapshot allocation.
        if self.unindexed.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(SuperfileUri, u64, u64)> = self
            .unindexed
            .iter()
            .filter(|e| Some(e.key()) != keep)
            .map(|e| (*e.key(), e.value().size_bytes, e.value().mtime_us))
            .collect();

        candidates.sort_by_key(|(_, _, mtime_us)| *mtime_us);

        let mut freed: u64 = 0;
        for (uri, size, mtime_us) in candidates {
            if freed >= bytes_needed {
                break;
            }

            // Only the winner of the remove deletes and subtracts, so two concurrent evictions
            // cannot double-count.
            if self.unindexed.remove(&uri).is_some() {
                tracing::info!(
                    target: "infino::cache",
                    uri = %uri.0,
                    size,
                    cached_at_us = self.cached_at_us(&uri),
                    mtime_us,
                    "evict: whole .sf.parquet (budget)"
                );
                self.remove_local_copy(&uri);
                self.drop_block_file(&uri);
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed += size;
            }
        }

        freed
    }

    pub(crate) fn evict_block_files(&self, bytes_needed: u64, keep: Option<&SuperfileUri>) -> u64 {
        if self.block_files.is_empty() {
            return 0;
        }

        let mut candidates: Vec<(SuperfileUri, u64, u64)> = self
            .block_files
            .iter()
            .filter(|e| Some(e.key()) != keep)
            .map(|e| (*e.key(), e.value().size_bytes, e.value().mtime_us))
            .collect();

        candidates.sort_by_key(|(_, _, mtime_us)| *mtime_us);

        let mut freed: u64 = 0;
        for (uri, size, mtime_us) in candidates {
            if freed >= bytes_needed {
                break;
            }

            if self.block_files.remove(&uri).is_some() {
                tracing::info!(
                    target: "infino::cache",
                    uri = %uri.0,
                    size,
                    cached_at_us = created_at_us(&self.blocks_path(&uri)),
                    mtime_us,
                    "evict: .blocks (budget)"
                );
                let _ = fs::remove_file(self.blocks_path(&uri));
                let _ = fs::remove_file(self.blocks_idx_path(&uri));
                self.current_bytes.fetch_sub(size, Ordering::Release);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
                freed += size;
            }
        }

        freed
    }

    /// Reserve `bytes`, evicting as needed, with no guard: the bytes stay reserved until the caller
    /// releases them ([`Self::release_block_bytes`]) or hands them to an admitted entry. For a
    /// reservation that outlives a borrow of `self`, such as a background fill's.
    pub(crate) async fn reserve_manual(&self, bytes: u64) -> Result<(), DiskCacheError> {
        self.reserve_manual_except(bytes, None).await
    }

    /// [`Self::reserve_manual`] for a background download: never evicts `keep`, the file it makes
    /// room for, since evicting it would cancel the download.
    pub(super) async fn reserve_manual_keeping(
        &self,
        bytes: u64,
        keep: &SuperfileUri,
    ) -> Result<(), DiskCacheError> {
        self.reserve_manual_except(bytes, Some(keep)).await
    }

    async fn reserve_manual_except(
        &self,
        bytes: u64,
        keep: Option<&SuperfileUri>,
    ) -> Result<(), DiskCacheError> {
        self.reserve_evicting(bytes, keep, 0).await
    }

    /// Reserve `bytes`, evicting cache files as needed and never `keep`. `claimed` is how many
    /// of the bytes are already counted among the charges eviction cannot free (a temp-file
    /// charge claims its share before it reserves). Fails before evicting anything when those
    /// charges, the table's metadata and temp files, leave no room for the bytes: evicting
    /// every cache file would not make it.
    async fn reserve_evicting(
        &self,
        bytes: u64,
        keep: Option<&SuperfileUri>,
        claimed: u64,
    ) -> Result<(), DiskCacheError> {
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
            let held = self.unevictable_bytes().saturating_sub(claimed);
            if held.saturating_add(bytes) > budget {
                return Err(DiskCacheError::BudgetExceeded);
            }
            let needed = (cur + bytes).saturating_sub(budget);
            self.evict_except(needed, keep).await?;
        }
    }

    /// Bytes charged to the budget that eviction cannot free: the table's metadata and its
    /// temp files.
    fn unevictable_bytes(&self) -> u64 {
        self.metadata_bytes
            .load(Ordering::Acquire)
            .saturating_add(self.scratch_bytes.load(Ordering::Acquire))
    }

    /// Charge `bytes` the table's metadata cache is about to write beside this cache, evicting
    /// cache files to make room. Metadata is never evicted to make room for cache files: every
    /// query reads it. Fails, charging nothing, when eviction cannot free the room.
    pub(crate) async fn charge_metadata(&self, bytes: u64) -> Result<(), DiskCacheError> {
        self.reserve_manual(bytes).await?;
        self.metadata_bytes.fetch_add(bytes, Ordering::AcqRel);
        Ok(())
    }

    /// Charge `bytes` of metadata already on disk when the table opens. They take the disk
    /// either way, like the cache files found at startup, so they are counted at once; cache
    /// files no read has opened go now, and the next reservation evicts the rest back under
    /// the budget.
    pub(crate) fn adopt_metadata(&self, bytes: u64) {
        self.current_bytes.fetch_add(bytes, Ordering::Release);
        self.metadata_bytes.fetch_add(bytes, Ordering::AcqRel);
        self.trim_cold_to_budget();
    }

    /// Give back `bytes` charged for metadata that left the disk.
    pub(crate) fn release_metadata(&self, bytes: u64) {
        self.metadata_bytes.fetch_sub(bytes, Ordering::AcqRel);
        self.current_bytes.fetch_sub(bytes, Ordering::Release);
    }

    /// The most bytes this cache's budget lets its table's temp files be charged.
    pub(crate) fn scratch_cap(&self) -> u64 {
        self.disk_budget_bytes() / SCRATCH_SHARE_DIVISOR
    }

    /// Bytes temp files may still be charged before they hold their whole share of the budget.
    pub(crate) fn scratch_room(&self) -> u64 {
        self.scratch_cap()
            .saturating_sub(self.scratch_bytes.load(Ordering::Acquire))
    }

    /// Bytes of the budget nothing is charged for: what can still be written beside the cache
    /// before a reservation has to evict.
    pub(crate) fn unused_budget_bytes(&self) -> u64 {
        self.disk_budget_bytes()
            .saturating_sub(self.current_bytes.load(Ordering::Acquire))
    }

    /// An empty charge on this cache's budget, for temp files beside it to grow into.
    pub(crate) fn scratch_charge(self: &Arc<Self>) -> ScratchCharge {
        ScratchCharge {
            store: Arc::clone(self),
            bytes: 0,
        }
    }

    /// The charge parked for the temp files in `folder` (see [`ScratchCharge::park`]); empty
    /// when none is.
    pub(crate) fn take_parked_scratch(self: &Arc<Self>, folder: &Path) -> ScratchCharge {
        ScratchCharge {
            store: Arc::clone(self),
            bytes: self
                .parked_scratch
                .remove(folder)
                .map_or(0, |(_, bytes)| bytes),
        }
    }

    /// Reserve a whole file of `bytes` for a Warm open's background download. `None` when the open
    /// allows no download, or when the budget cannot hold the file: eviction ran out of victims,
    /// the one error it raises (`BudgetExceeded`). Without it the open stays lazy and charges only
    /// the blocks it reads. The guard gives the bytes back unless committed.
    pub(crate) async fn reserve_whole_file(
        &self,
        allow_background_fill: bool,
        bytes: u64,
    ) -> Option<Reservation<'_>> {
        if !allow_background_fill {
            return None;
        }
        self.reserve(bytes).await.ok()
    }

    /// Reserve `bytes` only if the budget has room now, never evicting. For a caller that cannot
    /// wait, such as an install under a shard lock.
    pub(crate) fn try_reserve_without_evicting(&self, bytes: u64) -> bool {
        let budget = self.disk_budget_bytes();
        self.current_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                cur.checked_add(bytes).filter(|&next| next <= budget)
            })
            .is_ok()
    }

    /// Release previously reserved block-cache bytes.
    pub(crate) fn release_block_bytes(&self, bytes: u64) {
        self.current_bytes.fetch_sub(bytes, Ordering::Release);
    }

    /// Reserve room for `bytes` downloaded ahead of reads, held until the download ends. `None`
    /// when the unwritten runs plus the downloads ahead still in flight would pass
    /// [`AHEAD_WRITE_BEHIND_PERCENT`] of [`WRITE_BEHIND_CAP_BYTES`]. Counting the ones in flight
    /// stops many downloads, each checked alone, from overrunning the cap when they all land.
    pub(crate) fn try_reserve_ahead(self: &Arc<Self>, bytes: u64) -> Option<AheadRoom> {
        let share = WRITE_BEHIND_CAP_BYTES * AHEAD_WRITE_BEHIND_PERCENT / 100;
        let queued = self.write_behind_bytes.load(Ordering::Acquire);
        self.ahead_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |ahead| {
                let next = ahead.checked_add(bytes)?;
                (queued.saturating_add(next) <= share).then_some(next)
            })
            .ok()?;
        Some(AheadRoom {
            store: Arc::clone(self),
            bytes,
        })
    }

    /// Takes `bytes` of the cap on unwritten runs, or `false` if that would go over
    /// [`WRITE_BEHIND_CAP_BYTES`]. A run bigger than the whole cap is still let in when nothing else
    /// is waiting, so a big read can be cached at all.
    pub(crate) fn try_admit_write_behind(&self, bytes: u64) -> bool {
        self.write_behind_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
                let next = cur.checked_add(bytes)?;
                (cur == 0 || next <= WRITE_BEHIND_CAP_BYTES).then_some(next)
            })
            .is_ok()
    }

    /// Takes `bytes` of the cap even past it, for a run whose caller waits for the write: the caller
    /// holds those bytes until then either way.
    pub(crate) fn admit_write_behind(&self, bytes: u64) {
        self.write_behind_bytes.fetch_add(bytes, Ordering::AcqRel);
    }

    /// Waits for one of the store's [`BLOCK_WRITERS`] slots. Waiting holds no thread.
    pub(crate) async fn block_writer(&self) -> OwnedSemaphorePermit {
        Arc::clone(&self.block_writers)
            .acquire_owned()
            .await
            .expect("invariant: the block-writer semaphore is never closed")
    }

    /// Gives back what admitting a run took, once the run is written or dropped.
    pub(crate) fn release_write_behind(&self, bytes: u64) {
        self.write_behind_bytes.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// True when `token` still identifies the live block source for `uri`.
    pub(crate) fn lazy_block_entry_is_current(&self, uri: &SuperfileUri, token: &Arc<()>) -> bool {
        self.cached
            .get(uri)
            .and_then(|entry| entry.block_source().map(|source| source.owns_token(token)))
            .unwrap_or(false)
    }

    /// Release accounting for one removed cache entry.
    pub(crate) fn release_entry_accounting(&self, entry: &CachedEntry) {
        if entry.accounting == EntryAccounting::Eager {
            self.current_bytes
                .fetch_sub(entry.size_bytes.load(Ordering::Acquire), Ordering::Release);
        }
    }

    /// Put `entry` in the cache under `uri`. Every admission goes through here, so when admissions
    /// race on a URI (a compaction read and a query open, two disk reuses) two rules hold:
    ///
    /// 1. A lazy entry never replaces a whole file. It is dropped, its budget released, and the
    ///    whole file is returned instead.
    /// 2. Whatever is replaced has its budget released, so a lost race leaves no bytes charged.
    ///
    /// Returns the entry now in the slot. Callers serve that, not what they passed in.
    pub(crate) fn admit_entry(
        &self,
        uri: SuperfileUri,
        entry: Arc<CachedEntry>,
    ) -> Arc<CachedEntry> {
        match self.cached.entry(uri) {
            Entry::Occupied(mut occupied) => {
                if !entry.has_whole_file() && occupied.get().has_whole_file() {
                    let existing = Arc::clone(occupied.get());
                    drop(occupied);
                    self.release_entry_accounting(&entry);
                    return existing;
                }

                let replaced = mem::replace(occupied.get_mut(), Arc::clone(&entry));
                // Leave the shard lock before `replaced` drops: its last reference may fsync a
                // block index or munmap, and every URI in the shard would wait on that.
                drop(occupied);
                self.release_entry_accounting(&replaced);
                entry
            }
            Entry::Vacant(slot) => {
                slot.insert(Arc::clone(&entry));
                entry
            }
        }
    }

    /// Reserve `bytes`, evicting as needed, or fail with `BudgetExceeded`. The guard releases the
    /// bytes on drop unless committed.
    pub(crate) async fn reserve(&self, bytes: u64) -> Result<Reservation<'_>, DiskCacheError> {
        self.reserve_manual(bytes).await?;
        Ok(Reservation {
            store: self,
            bytes,
            committed: false,
        })
    }

    /// Take over `bytes` already charged (a file the startup scan counted), so they are given back
    /// unless committed, as a fresh reservation's are.
    pub(crate) fn adopt_charge(&self, bytes: u64) -> Reservation<'_> {
        Reservation {
            store: self,
            bytes,
            committed: false,
        }
    }

    /// Like [`Self::reserve`], but only from free space: `None` instead of evicting.
    pub(crate) fn try_reserve(&self, bytes: u64) -> Option<Reservation<'_>> {
        self.try_reserve_without_evicting(bytes)
            .then(|| Reservation {
                store: self,
                bytes,
                committed: false,
            })
    }

    /// [`Self::evict_except`] with nothing kept.
    #[cfg(test)]
    pub(crate) async fn evict_at_least(&self, bytes_needed: u64) -> Result<(), DiskCacheError> {
        self.evict_except(bytes_needed, None).await
    }

    /// Drive the eviction policy until either `bytes_needed` is freed or no eligible victims remain
    /// (→ `BudgetExceeded`), taking nothing of `keep`: its entry, copy or block file.
    pub(super) async fn evict_except(
        &self,
        bytes_needed: u64,
        keep: Option<&SuperfileUri>,
    ) -> Result<(), DiskCacheError> {
        // Never-opened files go first; freeing one cannot hurt a live reader.
        let freed = self.evict_unindexed(bytes_needed, keep);
        if freed >= bytes_needed {
            return Ok(());
        }
        let freed = freed + self.evict_block_files(bytes_needed - freed, keep);
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
        let mut pinned = pinned_fn();
        pinned.extend(keep.copied());
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
            // the caller that wins the removal runs
            // unlink + decrement. Without this gate, two
            // reservations evicting the same victim could
            // double-decrement current_bytes. The removal
            // also cancels the file's download.
            if let Some(entry) = self.remove_entry_cancelling_fill(&uri, |_| true) {
                tracing::info!(
                    target: "infino::cache",
                    uri = %uri.0,
                    cached_at_us = self.cached_at_us(&uri),
                    last_used_us = unix_us(self.store_time(entry.last_access_us.load(Ordering::Acquire))),
                    "evict: live cached entry (budget pressure)"
                );

                // A leftover coordinator would keep the evicted entry, and its budget, alive.
                self.coordinators.remove(&uri);

                self.remove_local_copy(&uri);
                let _ = fs::remove_file(self.blocks_path(&uri));
                let _ = fs::remove_file(self.blocks_idx_path(&uri));
                self.release_entry_accounting(&entry);
                self.n_evictions.fetch_add(1, Ordering::AcqRel);
            }
        }
        Ok(())
    }

    // Test helpers. Compiled only for tests, never into the shipped library.

    /// The memory budget background downloads charge, if any.
    #[cfg(test)]
    pub(crate) fn fill_memory_budget(&self) -> Option<&Arc<ConnectionMemoryBudget>> {
        self.config.fill_memory_budget.as_ref()
    }

    /// Counts a queued block-file write until the returned ticket drops; see
    /// [`Self::block_writes_settled`].
    #[cfg(test)]
    pub(crate) fn block_write_started(self: &Arc<Self>) -> BlockWriteTicket {
        self.block_writes.fetch_add(1, Ordering::AcqRel);
        BlockWriteTicket(Arc::clone(self))
    }

    /// Bytes of fetched runs not written yet.
    #[cfg(test)]
    pub(crate) fn write_behind_in_use(&self) -> u64 {
        self.write_behind_bytes.load(Ordering::Acquire)
    }

    /// Waits until every queued block-file write, and the index write after it, is done, so a
    /// test can check the result.
    #[cfg(test)]
    pub(crate) async fn block_writes_settled(&self) {
        let settled = async {
            loop {
                let idle = self.block_writes_idle.notified();
                tokio::pin!(idle);
                idle.as_mut().enable();
                if self.block_writes.load(Ordering::Acquire) == 0 {
                    return;
                }
                idle.await;
            }
        };
        timeout(BLOCK_WRITES_SETTLE_DEADLINE, settled)
            .await
            .expect("queued block writes finish in time");
    }

    /// The budget ledger at rest: every charged byte is backed by a live entry, a scanned cache
    /// file, a scanned block file, a background fill's reservation, a temp-file charge or the
    /// table's metadata, and nothing else. Holds only with no fetch or fill running (see
    /// `wait_until_fills_settled`).
    #[cfg(test)]
    pub(crate) fn assert_budget_consistent(&self) {
        let entries: u64 = self.cached.iter().map(|e| e.value().charged_bytes()).sum();
        let unindexed: u64 = self.unindexed.iter().map(|f| f.value().size_bytes).sum();
        let block_files: u64 = self.block_files.iter().map(|f| f.value().size_bytes).sum();
        let fills: u64 = self
            .fills
            .iter()
            .map(|f| f.value().reserved.load(Ordering::Acquire))
            .sum();
        let scratch = self.scratch_bytes.load(Ordering::Acquire);
        let metadata = self.metadata_bytes.load(Ordering::Acquire);
        assert_eq!(
            self.current_bytes.load(Ordering::Acquire),
            entries + unindexed + block_files + fills + scratch + metadata,
            "budget ledger drifted from what the cache holds (entries {entries}, \
             unindexed {unindexed}, block files {block_files}, fills {fills}, \
             temp files {scratch}, metadata {metadata})"
        );
        let parked: u64 = self.parked_scratch.iter().map(|p| *p.value()).sum();
        assert!(
            parked <= scratch,
            "parked temp-file bytes {parked} exceed the {scratch} charged"
        );
    }

    #[cfg(test)]
    pub(crate) fn install_block_entry_for_test(
        &self,
        uri: SuperfileUri,
        block_source: Arc<BlockCachedSource>,
    ) {
        let reader = SuperfileReader::open(tiny_superfile_bytes()).expect("tiny superfile opens");
        let size_bytes = block_source.filled_bytes_handle();
        let residency = Residency::Paged { block_source };
        let accounting = EntryAccounting::SourceOwned;
        let now_us = self.now_us();
        cache_entry(
            self,
            uri,
            Arc::new(reader),
            residency,
            size_bytes,
            accounting,
            now_us,
        );
    }

    #[cfg(test)]
    pub(crate) fn remove_block_entry_for_test(&self, uri: &SuperfileUri) {
        let _ = self.cached.remove(uri);
    }
}

/// One queued block-file write, counted by [`DiskCacheStore::block_writes_settled`] until it
/// drops.
#[cfg(test)]
pub(crate) struct BlockWriteTicket(Arc<DiskCacheStore>);

#[cfg(test)]
impl Drop for BlockWriteTicket {
    fn drop(&mut self) {
        if self.0.block_writes.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.0.block_writes_idle.notify_waiters();
        }
    }
}

/// How [`ScratchCharge::try_grow`] went.
pub(crate) enum ScratchRoom {
    /// The bytes are charged.
    Charged,
    /// No room right now: other temp files hold the share, or eviction could not free enough.
    /// A later call may succeed.
    NoRoom(DiskCacheError),
    /// The budget could not hold the charge even with no other temp files charged, so waiting
    /// never helps.
    NeverFits(DiskCacheError),
}

/// Bytes charged to a cache's budget for temp files beside it. Drop gives them back.
///
/// The charge is part of the one budget the cache keeps to: growing it evicts cache files,
/// least recently used first. All of a cache's charges together stay within
/// `budget / SCRATCH_SHARE_DIVISOR`.
pub(crate) struct ScratchCharge {
    store: Arc<DiskCacheStore>,
    bytes: u64,
}

impl ScratchCharge {
    /// Charge `more` bytes, evicting cache files to make room. Fails with
    /// [`DiskCacheError::ScratchFull`], evicting nothing, when temp files would pass their share
    /// of the budget, and with [`DiskCacheError::BudgetExceeded`] when eviction cannot free the
    /// room. A failed call leaves the charge as it was.
    pub(crate) async fn grow(&mut self, more: u64) -> Result<(), DiskCacheError> {
        let store = &self.store;
        let cap = store.scratch_cap();
        // The share is claimed before anything is evicted, so of two charges that would pass it
        // together only one evicts.
        store
            .scratch_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |charged| {
                charged.checked_add(more).filter(|&next| next <= cap)
            })
            .map_err(|charged| DiskCacheError::ScratchFull {
                needed: more,
                charged,
                cap,
            })?;
        if let Err(error) = store.reserve_evicting(more, None, more).await {
            store.scratch_bytes.fetch_sub(more, Ordering::AcqRel);
            return Err(error);
        }
        self.bytes += more;
        Ok(())
    }

    /// [`Self::grow`], telling a charge with no room right now from one the budget could never
    /// hold. Either way the charge is left as it was.
    pub(crate) async fn try_grow(&mut self, more: u64) -> ScratchRoom {
        match self.grow(more).await {
            Ok(()) => ScratchRoom::Charged,
            Err(error) if self.could_hold(more) => ScratchRoom::NoRoom(error),
            Err(error) => ScratchRoom::NeverFits(error),
        }
    }

    /// Charge `bytes` of a temp file already written, without evicting and without waiting:
    /// for a commit's build output, which must not be held up. The bytes take the disk either
    /// way, so they are counted at once and the next reservation evicts cache files back
    /// under the budget; files no read has opened go now. `false`, charging nothing, when temp
    /// files would pass their share of the budget.
    pub(crate) fn adopt(&mut self, bytes: u64) -> bool {
        let store = &self.store;
        let cap = store.scratch_cap();
        let counted =
            store
                .scratch_bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |charged| {
                    charged.checked_add(bytes).filter(|&next| next <= cap)
                });
        if counted.is_err() {
            return false;
        }
        store.current_bytes.fetch_add(bytes, Ordering::Release);
        self.bytes += bytes;
        store.trim_cold_to_budget();
        true
    }

    /// Bytes the cache's temp files may still be charged before they hold their whole share
    /// of the budget.
    pub(crate) fn room(&self) -> u64 {
        self.store.scratch_room()
    }

    /// Give back whatever is charged past `bytes`.
    pub(crate) fn shrink_to(&mut self, bytes: u64) {
        let Some(less) = self.bytes.checked_sub(bytes) else {
            return;
        };
        self.store.scratch_bytes.fetch_sub(less, Ordering::AcqRel);
        self.store.current_bytes.fetch_sub(less, Ordering::Release);
        self.bytes -= less;
    }

    /// The bytes charged now.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Whether the budget could hold this charge grown by `more` with no other temp files
    /// charged: within the temp files' share, and beside the table's metadata, which
    /// eviction cannot free.
    pub(crate) fn could_hold(&self, more: u64) -> bool {
        let store = &self.store;
        let grown = self.bytes.saturating_add(more);
        grown <= store.scratch_cap()
            && store
                .metadata_bytes
                .load(Ordering::Acquire)
                .saturating_add(grown)
                <= store.disk_budget_bytes()
    }

    /// Leave the bytes charged for temp files that stay in `folder` with nothing running that
    /// owns them, until [`DiskCacheStore::take_parked_scratch`] takes them over.
    pub(crate) fn park(mut self, folder: &Path) {
        let bytes = mem::take(&mut self.bytes);
        if bytes > 0 {
            *self
                .store
                .parked_scratch
                .entry(folder.to_path_buf())
                .or_default() += bytes;
        }
    }
}

impl Drop for ScratchCharge {
    fn drop(&mut self) {
        self.shrink_to(0);
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
    use std::{
        collections::HashMap,
        fmt::Debug,
        fs::{self, OpenOptions},
        io::Write,
        path::Path,
        sync::{Arc, Mutex},
        thread,
        time::{Duration, SystemTime},
    };

    use bytes::Bytes;
    use tempfile::tempdir;
    use tracing::{
        Event, Subscriber,
        callsite::rebuild_interest_cache,
        field::{Field, Visit},
        subscriber::{set_default, set_global_default},
    };
    use tracing_subscriber::{Layer, layer::Context, prelude::*, registry};

    use crate::{
        superfile::LazyByteSource,
        supertable::{
            manifest::{SuperfileUri, disk_cache::MetadataDiskCache, part::ContentHash},
            reader_cache::{
                block_source::CACHE_BLOCK_BYTES,
                config::ColdFetchMode,
                disk::{budget::*, created_at_us, file_mtime_us, test_support::*, unix_us},
            },
        },
    };

    /// How far a store-clock time turned into a wall-clock time may land from the moment it names:
    /// a little late by a pause between the store's two clock reads, either way by a wall clock
    /// slewed since.
    const CLOCK_SLACK_US: u64 = 1_000;

    /// Long enough for a later write to move a file's mtime past its creation time.
    const MTIME_STEP: Duration = Duration::from_millis(20);

    /// The `evict:` lines logged, each as its message and its integer fields by name.
    #[derive(Clone, Default)]
    struct EvictionLog(Arc<Mutex<Vec<(String, HashMap<&'static str, u64>)>>>);

    impl EvictionLog {
        /// The integer fields of the one line logged with `message`.
        fn fields(&self, message: &str) -> HashMap<&'static str, u64> {
            let lines = self.0.lock().expect("eviction log lock");
            let mut matching = lines.iter().filter(|(logged, _)| logged == message);
            let (_, fields) = matching
                .next()
                .unwrap_or_else(|| panic!("no {message:?} line"));
            assert!(matching.next().is_none(), "one {message:?} line");
            fields.clone()
        }
    }

    #[derive(Default)]
    struct LineFields {
        message: String,
        numbers: HashMap<&'static str, u64>,
    }

    impl Visit for LineFields {
        fn record_u64(&mut self, field: &Field, value: u64) {
            self.numbers.insert(field.name(), value);
        }
        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            if field.name() == "message" {
                self.message = format!("{value:?}");
            }
        }
    }

    impl<S: Subscriber> Layer<S> for EvictionLog {
        fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
            let mut line = LineFields::default();
            event.record(&mut line);
            if line.message.starts_with("evict:") {
                self.0
                    .lock()
                    .expect("eviction log lock")
                    .push((line.message, line.numbers));
            }
        }
    }

    fn mtime_us(path: &Path) -> u64 {
        file_mtime_us(&fs::metadata(path).expect("file on disk"))
    }

    /// A file's created time stays put when it is written again and renamed, as a cache file's is
    /// when a download lands. Without a birth time there is none, rather than another time.
    #[test]
    fn a_file_keeps_its_created_time_through_a_later_write_and_a_rename() {
        let dir = tempdir().expect("tempdir");
        let (tmp, kept) = (dir.path().join("copy.tmp"), dir.path().join("copy"));
        fs::write(&tmp, b"first").expect("create");
        let created = created_at_us(&tmp);

        thread::sleep(MTIME_STEP);
        OpenOptions::new()
            .append(true)
            .open(&tmp)
            .and_then(|mut file| file.write_all(b" second"))
            .expect("write again");
        fs::rename(&tmp, &kept).expect("rename");

        if fs::metadata(&kept).expect("kept").created().is_ok() {
            let created = created.expect("a birth time");
            assert_eq!(
                created_at_us(&kept),
                Some(created),
                "the birth time survives"
            );
            assert!(
                created < mtime_us(&kept),
                "the later write moved only the mtime"
            );
        } else {
            assert_eq!(created_at_us(&kept), None, "no other time stands in");
        }
    }

    /// Each budget eviction logs when its file was cached and when it was last used: a file never
    /// opened, a `.blocks` record, and a live entry adopted from disk after a restart, whose cached
    /// time is still the one its file was written at.
    #[tokio::test]
    async fn budget_evictions_log_when_each_file_was_cached_and_last_used() {
        // A process-wide registry keeps the eviction lines enabled whichever test thread meets
        // them first. Another test may have installed it already.
        let _ = set_global_default(registry());
        let log = EvictionLog::default();
        let _logging = set_default(registry().with(log.clone()));
        rebuild_interest_cache();

        let (_dir, store) = test_store();
        let bytes = tiny_superfile_bytes();
        let (unopened, adopted, blocks) = (
            SuperfileUri::new_v4(),
            SuperfileUri::new_v4(),
            SuperfileUri::new_v4(),
        );
        seed_cache_file(&store, &unopened, &bytes);
        seed_cache_file(&store, &adopted, &bytes);
        seed_valid_block_file(&store, &blocks, 3 * CACHE_BLOCK_BYTES, &[0, 1, 2]);
        // A later write moves the mtime, so each line's two times differ wherever the filesystem
        // keeps a birth time.
        for path in [
            store.cache_path(&unopened),
            store.cache_path(&adopted),
            store.blocks_path(&blocks),
        ] {
            OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_modified(SystemTime::now() + MTIME_STEP))
                .expect("move the mtime");
        }
        // Read before eviction deletes the files.
        let times = |path: &Path| (created_at_us(path), mtime_us(path));
        let unopened_times = times(&store.cache_path(&unopened));
        let blocks_times = times(&store.blocks_path(&blocks));
        let adopted_created = created_at_us(&store.cache_path(&adopted));

        // A restart: the records and the entry below all come from the files on disk.
        let reopened = reopen_store(&store, |_| {});
        let before_open = unix_us(SystemTime::now());
        reopened
            .reader(&adopted)
            .await
            .expect("adopt the whole copy");

        reopened.evict_unindexed(u64::MAX, None);
        reopened.evict_block_files(u64::MAX, None);
        reopened
            .evict_at_least(bytes.len() as u64)
            .await
            .expect("evict the live entry");

        let whole = log.fields("evict: whole .sf.parquet (budget)");
        assert_eq!(whole.get("cached_at_us").copied(), unopened_times.0);
        assert_eq!(whole.get("mtime_us"), Some(&unopened_times.1));

        let block_file = log.fields("evict: .blocks (budget)");
        assert_eq!(block_file.get("cached_at_us").copied(), blocks_times.0);
        assert_eq!(block_file.get("mtime_us"), Some(&blocks_times.1));

        let live = log.fields("evict: live cached entry (budget pressure)");
        assert_eq!(
            live.get("cached_at_us").copied(),
            adopted_created,
            "adopting the file after a restart keeps when it was cached"
        );
        let last_used = live["last_used_us"];
        let evicted = unix_us(SystemTime::now());
        assert!(
            (before_open - CLOCK_SLACK_US..=evicted + CLOCK_SLACK_US).contains(&last_used),
            "last used at the open: {last_used} not within {before_open}..={evicted}"
        );
    }

    /// A file with more than one local copy was cached when its oldest copy was created, since the
    /// eviction that logs it deletes them all.
    #[tokio::test]
    async fn a_file_was_cached_when_its_oldest_copy_was_created() {
        let (_dir, store) = test_store();
        let uri = SuperfileUri::new_v4();
        seed_valid_block_file(&store, &uri, CACHE_BLOCK_BYTES, &[0]);
        thread::sleep(MTIME_STEP);
        seed_cache_file(&store, &uri, &tiny_superfile_bytes());

        let oldest = created_at_us(&store.blocks_path(&uri));
        assert_eq!(store.cached_at_us(&uri), oldest);
        if oldest.is_some() {
            assert!(
                oldest < created_at_us(&store.cache_path(&uri)),
                "two distinct copies"
            );
        }
    }

    #[tokio::test]
    async fn downloads_ahead_of_reads_get_half_the_unwritten_runs_cap() {
        // Downloads ahead may fill half the write queue's cap, counting the ones
        // still downloading; the other half is left for reads' own downloads.
        let (_dir, store) = test_store_with(|_| {});
        let half = WRITE_BEHIND_CAP_BYTES * AHEAD_WRITE_BEHIND_PERCENT / 100;
        assert!(store.try_reserve_ahead(half + 1).is_none());
        // Queued runs and downloads in flight both count.
        store.admit_write_behind(half / 2);
        let first = store.try_reserve_ahead(half / 4).expect("room");
        assert!(store.try_reserve_ahead(half / 4 + 1).is_none());
        let second = store.try_reserve_ahead(half / 4).expect("the last of it");
        assert!(store.try_reserve_ahead(1).is_none());
        // A download that ends gives its room back.
        drop(first);
        assert!(store.try_reserve_ahead(half / 4).is_some());
        drop(second);
        store.release_write_behind(half / 2);
        assert!(store.try_reserve_ahead(half).is_some());
    }

    /// A lazy reader's cost is anonymous heap, which `madvise` cannot
    /// reclaim, so the idle sweep drops the entry instead. Only when the
    /// cache is the last holder: a reader a query still has in hand must
    /// survive, or the drop frees nothing and costs that query a re-open.
    #[tokio::test]
    async fn idle_sweep_releases_unheld_lazy_readers_and_spares_held_ones() {
        // threshold 0 makes every entry immediately idle.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.mmap_cold_threshold_secs = 0;
            // Keep the entry lazy for the length of the test; promotion is
            // the other way its memory goes away and would mask this one.
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;

        // Held by a "query": the sweep must leave it alone.
        let held = store.reader(&uri).await.expect("lazy cold");
        store.sweep_once();
        assert!(
            store.is_cached(&uri),
            "a lazy reader a caller still holds must survive the sweep"
        );

        // Released: the cache is now the only holder, so the sweep reclaims it.
        drop(held);
        store.sweep_once();
        assert!(
            !store.is_cached(&uri),
            "an idle lazy reader must be dropped — its heap is not madvise-able"
        );

        // And it is re-openable afterwards.
        let again = store.reader(&uri).await.expect("re-open after release");
        assert_eq!(again.n_docs(), 1);
    }

    /// The sweep gives back the budget of every entry it drops. A Warm lazy entry is charged its
    /// full size up front, so dropping it without releasing that charge leaves phantom bytes, and
    /// eviction then deletes real files to make room for them. The file's download outlives the
    /// entry: it lands on disk, charged once, and the next read opens it from there.
    #[tokio::test]
    async fn idle_sweep_releases_the_budget_of_what_it_drops() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.mmap_cold_threshold_secs = 0;
            cfg.promotion_defer_timeout = Duration::MAX;
            cfg.prefetch_concurrency = 1;
        });
        let warm = SuperfileUri::new_v4();
        let stream = SuperfileUri::new_v4();
        for uri in [&warm, &stream] {
            put_superfile(&store, uri, tiny_superfile_bytes()).await;
        }
        // Hold the only download slot until the sweep: a download past its slot keeps its file's
        // entry, and on a slow run the Warm open's download gets there before the sweep.
        let slot = hold_fill_slot(&store).await;
        store.reader(&warm).await.expect("warm lazy open");
        store
            .open_for_query(
                &stream,
                &stream.storage_path(),
                None,
                None,
                ReadIntent::Stream,
            )
            .await
            .expect("stream lazy open");
        assert!(
            store.stats().current_bytes > 0,
            "the warm open charged its full size"
        );

        let size = tiny_superfile_bytes().len() as u64;
        store.sweep_once();

        assert!(
            !store.is_cached(&warm) && !store.is_cached(&stream),
            "both idle lazy entries are dropped"
        );
        store.assert_budget_consistent();
        assert_eq!(
            store.stats().current_bytes,
            0,
            "nothing is charged once both entries are gone"
        );

        drop(slot);
        store
            .wait_until_fills_settled(PROMOTE_TIMEOUT)
            .await
            .expect("the download finishes");
        assert!(store.cache_path(&warm).exists(), "on disk");
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, size, "charged once");

        let reuses = store.stats().n_disk_reuses;
        store.reader(&warm).await.expect("read it again");
        assert_eq!(store.stats().n_disk_reuses, reuses + 1, "opened from disk");
        assert!(store.is_mmap_promoted(&warm));
        store.assert_budget_consistent();
    }

    /// Opens `uri` lazily for streaming and returns its block source.
    async fn open_lazily(
        store: &Arc<DiskCacheStore>,
        uri: &SuperfileUri,
    ) -> Arc<BlockCachedSource> {
        store
            .open_for_query(uri, &uri.storage_path(), None, None, ReadIntent::Stream)
            .await
            .expect("stream lazy open");
        store
            .cached
            .get(uri)
            .and_then(|entry| entry.block_source().map(Arc::clone))
            .expect("a lazy entry")
    }

    /// Opens `uri` lazily for streaming and reads its first byte, so one block is filled.
    async fn open_and_fill_a_block(store: &Arc<DiskCacheStore>, uri: &SuperfileUri) {
        let source = open_lazily(store, uri).await;
        source.range(0, 1).await.expect("block read");
        store.block_writes_settled().await;
    }

    /// A Stream lazy entry with one block filled, released while its reader is still alive, as
    /// one waiting on the freeing thread is.
    ///  - the block file is counted once, by its record, while the reader lives;
    ///  - freeing the reader later gives back nothing more.
    #[tokio::test]
    async fn a_released_reader_freed_later_leaves_its_block_file_counted_once() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        open_and_fill_a_block(&store, &uri).await;
        let filled = store
            .cached
            .get(&uri)
            .and_then(|entry| entry.block_source().map(|s| s.filled_bytes_handle()))
            .expect("a lazy entry")
            .load(Ordering::Acquire);
        assert!(filled > 0, "a block is filled");

        let queued = store.release_lazy_entry(&uri, store.now_us());
        assert!(queued.is_some(), "the entry is released");
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, filled, "counted once");

        drop(queued);
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, filled, "still counted once");
    }

    /// A Stream lazy entry with one block filled, released while its source lives, and a run that
    /// lands on that source after the release (a read-ahead GET that finished as its statement
    /// ended), leaving its bytes reserved in the store and filled in the source.
    ///  - the record counts the blocks filled at release;
    ///  - dropping the source gives the late run back, so the cache counts only the record.
    #[tokio::test]
    async fn a_run_landing_after_release_is_given_back_when_its_source_drops() {
        /// Bytes the late run lands.
        const LATE_BYTES: u64 = 4096;
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        open_and_fill_a_block(&store, &uri).await;
        let source = store
            .cached
            .get(&uri)
            .and_then(|entry| entry.block_source().cloned())
            .expect("a lazy entry");
        let filled = source.filled_bytes_handle().load(Ordering::Acquire);

        let queued = store.release_lazy_entry(&uri, store.now_us());
        assert!(queued.is_some(), "the entry is released");
        store.current_bytes.fetch_add(LATE_BYTES, Ordering::Release);
        source
            .filled_bytes_handle()
            .fetch_add(LATE_BYTES, Ordering::AcqRel);

        drop(queued);
        drop(source);
        assert_eq!(
            store.stats().current_bytes,
            filled,
            "only the record stays counted"
        );
    }

    /// A lazy entry with a block write still queued.
    ///  - the sweep keeps it: its block file would be counted before the write's bytes are.
    ///  - once the write is done, the sweep drops it and counts the whole file.
    #[tokio::test]
    async fn idle_sweep_waits_for_a_queued_block_write() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.mmap_cold_threshold_secs = 0;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        let source = open_lazily(&store, &uri).await;
        let stall = source.stall_writes();
        source.range(0, 1).await.expect("block read");

        store.sweep_once();
        assert!(store.is_cached(&uri), "the entry waits for its write");

        drop(stall);
        store.block_writes_settled().await;
        let filled = source.filled_bytes_handle().load(Ordering::Acquire);
        assert!(filled > 0, "the write landed");
        drop(source);
        store.sweep_once();
        assert!(!store.is_cached(&uri), "then the idle entry is dropped");
        store.assert_budget_consistent();
        assert_eq!(
            store.stats().current_bytes,
            filled,
            "and its file is charged"
        );
    }

    #[tokio::test]
    async fn idle_sweep_waits_for_a_queued_index_flush() {
        // Block 0 is written, and its index flush waits for a writer.
        //  - the sweep keeps the entry: a reopen would fetch the block again.
        //  - once the flush is done, the sweep drops it.
        // This test pins the sweep waiting for the flush.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.mmap_cold_threshold_secs = 0;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        let source = open_lazily(&store, &uri).await;
        let stall = source.stall_queued_flushes();
        source.range(0, 1).await.expect("block read");
        wait_until(
            || stall.waiting() == 1 && !source.has_runs_in_memory(),
            "the write lands and queues its flush",
        )
        .await;

        store.sweep_once();
        assert!(store.is_cached(&uri), "the entry waits for its index flush");

        drop(stall);
        store.block_writes_settled().await;
        drop(source);
        store.sweep_once();
        assert!(!store.is_cached(&uri), "then the idle entry is dropped");
        store.assert_budget_consistent();
    }

    /// A run bigger than the whole cap is let in when nothing else is waiting; anything more then
    /// waits for room.
    #[tokio::test]
    async fn an_oversized_run_is_admitted_alone() {
        let (_dir, store) = test_store_with(|_| {});
        assert!(store.try_admit_write_behind(WRITE_BEHIND_CAP_BYTES + 1));
        assert!(
            !store.try_admit_write_behind(1),
            "nothing more fits beside it"
        );
        store.release_write_behind(WRITE_BEHIND_CAP_BYTES + 1);
        assert!(store.try_admit_write_behind(1));
        assert!(
            !store.try_admit_write_behind(WRITE_BEHIND_CAP_BYTES),
            "with another run waiting, the cap holds"
        );
    }

    /// A lazy entry the sweep drops leaves its `.blocks` file on disk for a later open to adopt.
    /// The file must stay charged, or the disk holds bytes the budget cannot see or evict.
    #[tokio::test]
    async fn idle_sweep_keeps_charging_the_block_file_it_leaves() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
            cfg.mmap_cold_threshold_secs = 0;
            cfg.promotion_defer_timeout = Duration::MAX;
        });
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        open_and_fill_a_block(&store, &uri).await;
        let filled = store.stats().current_bytes;
        assert!(filled > 0, "the read filled a block");

        store.sweep_once();
        assert!(!store.is_cached(&uri), "the idle entry is dropped");
        assert!(
            store.blocks_path(&uri).exists(),
            "its file stays for adoption"
        );
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, filled, "and stays charged");

        // A reopen adopts the file without charging it twice.
        open_and_fill_a_block(&store, &uri).await;
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, filled);

        store.sweep_once();
        store.evict_at_least(filled).await.expect("evict");
        assert!(
            !store.blocks_path(&uri).exists(),
            "eviction reclaims the file"
        );
        store.assert_budget_consistent();
        assert_eq!(store.stats().current_bytes, 0);
    }

    /// Caches `n` tiny superfiles in `store` and returns their URIs, oldest first.
    async fn cache_tiny_superfiles(store: &Arc<DiskCacheStore>, n: usize) -> Vec<SuperfileUri> {
        let mut uris = Vec::with_capacity(n);
        for _ in 0..n {
            let uri = SuperfileUri::new_v4();
            store
                .insert_warm(&uri, tiny_superfile_bytes())
                .await
                .expect("cached");
            uris.push(uri);
        }
        uris
    }

    #[tokio::test]
    async fn temp_files_take_their_room_from_the_cache_within_one_budget() {
        let file = tiny_superfile_bytes().len() as u64;
        // Room for exactly four cached files, so temp files may take two files' worth.
        let (_dir, store) =
            test_store_with(|cfg| cfg.disk_budget_bytes = SCRATCH_TEST_BUDGET_FILES * file);
        let uris = cache_tiny_superfiles(&store, SCRATCH_TEST_BUDGET_FILES as usize).await;
        assert_eq!(
            store.stats().current_bytes,
            SCRATCH_TEST_BUDGET_FILES * file
        );

        // One file's worth of temp files evicts the least recently used cached file.
        let mut charge = store.scratch_charge();
        charge.grow(file).await.expect("room made");
        let stats = store.stats();
        assert_eq!(stats.scratch_bytes, file);
        assert_eq!(stats.current_bytes, stats.budget_bytes, "one budget");
        assert_eq!(stats.n_evictions, 1);
        assert!(!store.is_cached(&uris[0]), "the oldest file went");
        assert!(store.is_cached(&uris[1]));
        store.assert_budget_consistent();

        // Dropping the charge gives its room back.
        drop(charge);
        let stats = store.stats();
        assert_eq!(stats.scratch_bytes, 0);
        assert_eq!(stats.current_bytes, stats.budget_bytes - file);
        store.assert_budget_consistent();
    }

    #[tokio::test]
    async fn a_temp_file_charge_past_its_share_fails_and_evicts_nothing() {
        let file = tiny_superfile_bytes().len() as u64;
        let (_dir, store) =
            test_store_with(|cfg| cfg.disk_budget_bytes = SCRATCH_TEST_BUDGET_FILES * file);
        let uris = cache_tiny_superfiles(&store, SCRATCH_TEST_BUDGET_FILES as usize).await;
        let cap = store.stats().budget_bytes / SCRATCH_SHARE_DIVISOR;

        // More than the share in one step: refused before anything is evicted.
        let mut charge = store.scratch_charge();
        let err = charge.grow(cap + 1).await.expect_err("past the share");
        assert!(matches!(err, DiskCacheError::ScratchFull { .. }), "{err}");
        assert_eq!(charge.bytes(), 0);
        assert_eq!(store.stats().n_evictions, 0);
        assert!(uris.iter().all(|uri| store.is_cached(uri)));
        store.assert_budget_consistent();

        // The share is counted across charges: a second one cannot take what the first holds.
        charge.grow(cap).await.expect("the whole share");
        let evictions = store.stats().n_evictions;
        let mut second = store.scratch_charge();
        let err = second.grow(1).await.expect_err("the share is taken");
        assert!(matches!(err, DiskCacheError::ScratchFull { .. }), "{err}");
        assert_eq!(store.stats().n_evictions, evictions, "nothing more evicted");
        assert_eq!(store.stats().scratch_bytes, cap);
        store.assert_budget_consistent();

        drop(second);
        drop(charge);
        assert_eq!(store.stats().scratch_bytes, 0);
        store.assert_budget_consistent();
    }

    /// A charge within its share that the budget cannot make room for gives the share back, so
    /// the next charge finds it free, and tells "no room now" from "can never fit".
    #[tokio::test]
    async fn a_temp_file_charge_the_budget_cannot_hold_gives_its_share_back() {
        let (_dir, store) = test_store_with(|_| {});
        let budget = store.stats().budget_bytes;
        let cap = budget / SCRATCH_SHARE_DIVISOR;
        // The whole budget is reserved with nothing to evict.
        store.reserve_manual(budget).await.expect("an empty cache");

        let mut charge = store.scratch_charge();
        let err = charge.grow(1).await.expect_err("nothing to evict");
        assert!(matches!(err, DiskCacheError::BudgetExceeded), "{err}");
        assert_eq!(charge.bytes(), 0);
        assert_eq!(
            store.stats().scratch_bytes,
            0,
            "the claimed share is given back"
        );
        assert!(matches!(charge.try_grow(1).await, ScratchRoom::NoRoom(_)));
        assert!(matches!(
            charge.try_grow(cap + 1).await,
            ScratchRoom::NeverFits(_)
        ));

        store.release_block_bytes(budget);
        assert!(matches!(charge.try_grow(cap).await, ScratchRoom::Charged));
        assert_eq!(store.stats().scratch_bytes, cap);
        store.assert_budget_consistent();
    }

    /// A table's metadata cache charges what it holds to the disk cache's budget: a write
    /// evicts cache files to make room, a cache file admitted later evicts other cache files
    /// and never the metadata, and what leaves the metadata cache is given back.
    #[tokio::test]
    async fn a_table_s_metadata_counts_against_its_disk_budget() {
        let file = tiny_superfile_bytes().len() as u64;
        let (dir, store) =
            test_store_with(|cfg| cfg.disk_budget_bytes = SCRATCH_TEST_BUDGET_FILES * file);
        let uris = cache_tiny_superfiles(&store, SCRATCH_TEST_BUDGET_FILES as usize).await;
        let metadata_root = dir.path().join("metadata");
        let open = || {
            MetadataDiskCache::new(
                metadata_root.clone(),
                METADATA_TEST_BUDGET_BYTES,
                Some(Arc::clone(&store)),
            )
            .expect("metadata cache")
        };
        let metadata = open();
        let object = tiny_superfile_bytes();
        let hash = ContentHash::of(&object);

        // The budget is full of cache files: the metadata takes one's room.
        assert!(metadata.put(hash, object.clone()).await);
        assert_eq!(store.stats().metadata_bytes, file);
        assert_eq!(store.stats().n_evictions, 1);
        store.assert_budget_consistent();

        // A cache file admitted again takes another cache file's room.
        let evicted = uris
            .iter()
            .find(|uri| !store.is_cached(uri))
            .expect("one cache file made room");
        store
            .insert_warm(evicted, tiny_superfile_bytes())
            .await
            .expect("cached again");
        assert!(metadata.contains(&hash), "the metadata stays");
        assert_eq!(store.stats().metadata_bytes, file);
        store.assert_budget_consistent();

        // An object the disk budget cannot hold beside the metadata is not cached, and
        // nothing is evicted for it: no eviction could make the room.
        let evictions = store.stats().n_evictions;
        let too_big = Bytes::from(vec![0u8; (SCRATCH_TEST_BUDGET_FILES * file) as usize]);
        assert!(!metadata.put(ContentHash::of(&too_big), too_big).await);
        assert_eq!(store.stats().metadata_bytes, file);
        assert_eq!(store.stats().n_evictions, evictions);
        store.assert_budget_consistent();

        // The hidden index's metadata cache charges the same budget.
        let hidden = metadata.for_hidden_index().expect("hidden index cache");
        let hidden_object = Bytes::from(vec![1u8; file as usize]);
        assert!(
            hidden
                .put(ContentHash::of(&hidden_object), hidden_object)
                .await
        );
        assert_eq!(store.stats().metadata_bytes, 2 * file);
        drop(hidden);
        assert_eq!(store.stats().metadata_bytes, file);

        // A cache opened over the same folder charges what is on disk.
        drop(metadata);
        assert_eq!(store.stats().metadata_bytes, 0);
        let metadata = open();
        assert_eq!(store.stats().metadata_bytes, file);
        metadata.remove(&hash);
        assert_eq!(store.stats().metadata_bytes, 0);
        store.assert_budget_consistent();
    }

    /// A metadata cache that is full swaps an old object for the new one within its own
    /// limit, and the table's cache files are left alone.
    #[tokio::test]
    async fn a_full_metadata_cache_makes_its_own_room() {
        let file = tiny_superfile_bytes().len() as u64;
        let (dir, store) =
            test_store_with(|cfg| cfg.disk_budget_bytes = SCRATCH_TEST_BUDGET_FILES * file);
        // A metadata cache with room for one object, holding one.
        let metadata =
            MetadataDiskCache::new(dir.path().join("metadata"), file, Some(Arc::clone(&store)))
                .expect("metadata cache");
        let first = tiny_superfile_bytes();
        assert!(metadata.put(ContentHash::of(&first), first.clone()).await);
        // Cache files fill the rest of the budget.
        let uris = cache_tiny_superfiles(&store, SCRATCH_TEST_BUDGET_FILES as usize - 1).await;
        let evictions = store.stats().n_evictions;

        let second = Bytes::from(vec![2u8; file as usize]);
        let second_hash = ContentHash::of(&second);
        assert!(metadata.put(second_hash, second).await);
        assert!(metadata.contains(&second_hash));
        assert!(
            !metadata.contains(&ContentHash::of(&first)),
            "the old one went"
        );
        assert_eq!(store.stats().metadata_bytes, file);
        assert_eq!(
            store.stats().n_evictions,
            evictions,
            "no cache file was evicted"
        );
        assert!(uris.iter().all(|uri| store.is_cached(uri)));
        store.assert_budget_consistent();
    }

    /// Temp files cannot push metadata out, so a temp-file charge that only fits by doing so
    /// can never fit, and is told apart from one with no room right now.
    #[tokio::test]
    async fn a_temp_file_charge_counts_the_metadata_it_cannot_evict() {
        let file = tiny_superfile_bytes().len() as u64;
        let (dir, store) =
            test_store_with(|cfg| cfg.disk_budget_bytes = SCRATCH_TEST_BUDGET_FILES * file);
        let metadata = MetadataDiskCache::new(
            dir.path().join("metadata"),
            METADATA_TEST_BUDGET_BYTES,
            Some(Arc::clone(&store)),
        )
        .expect("metadata cache");
        // Metadata holds three quarters of the budget.
        for fill in 0..SCRATCH_TEST_BUDGET_FILES - 1 {
            let object = Bytes::from(vec![fill as u8; file as usize]);
            assert!(metadata.put(ContentHash::of(&object), object).await);
        }
        let mut charge = store.scratch_charge();
        // Within the temp files' share, but not beside the metadata.
        let share = store.stats().budget_bytes / SCRATCH_SHARE_DIVISOR;
        assert!(!charge.could_hold(share));
        assert!(matches!(
            charge.try_grow(share).await,
            ScratchRoom::NeverFits(_)
        ));
        assert!(matches!(charge.try_grow(file).await, ScratchRoom::Charged));
        assert_eq!(store.stats().scratch_bytes, file);
        store.assert_budget_consistent();
    }

    #[tokio::test]
    async fn a_parked_temp_file_charge_stays_counted_until_it_is_taken_over() {
        let (dir, store) = test_store_with(|_| {});
        let folder = dir.path().join("spill");
        let mut charge = store.scratch_charge();
        charge.grow(SCRATCH_TEST_PARKED_BYTES).await.expect("room");

        // Parked, the bytes stay charged with no guard holding them.
        charge.park(&folder);
        assert_eq!(store.stats().scratch_bytes, SCRATCH_TEST_PARKED_BYTES);
        store.assert_budget_consistent();

        // Taken over, they are one charge again; a second take finds nothing.
        let mut taken = store.take_parked_scratch(&folder);
        assert_eq!(taken.bytes(), SCRATCH_TEST_PARKED_BYTES);
        assert_eq!(store.take_parked_scratch(&folder).bytes(), 0);
        assert_eq!(store.stats().scratch_bytes, SCRATCH_TEST_PARKED_BYTES);

        // Shrinking gives back the difference, dropping the rest.
        taken.shrink_to(1);
        assert_eq!(store.stats().scratch_bytes, 1);
        store.assert_budget_consistent();
        drop(taken);
        assert_eq!(store.stats().current_bytes, 0);
        store.assert_budget_consistent();
    }

    #[tokio::test]
    async fn a_footprint_over_the_budget_leaves_the_budget_as_set() {
        let (_dir, store) = test_store_with(|cfg| {
            cfg.disk_budget_bytes = TEST_TINY_BUDGET_BYTES;
        });
        // Warned about (once), never changed.
        store.warn_if_budget_below(TEST_LARGE_FOOTPRINT_BYTES);
        store.warn_if_budget_below(TEST_LARGE_FOOTPRINT_BYTES);
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
