// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The decoded file metadata the reader cache keeps between queries, held to one process budget.
//!
//! A cached file's reader keeps its decoded footer and page index after a query ends, and the disk
//! budget does not count that memory. So:
//! - whatever keeps decoded bytes holds a [`HeapReservation`], which gives them back when it drops;
//! - past the budget, and only while the rest of the process leaves the metadata less room than the
//!   budget, the files no query holds leave the cache, oldest first. A file whole on local disk
//!   keeps its bytes there, so its next open maps it again with no download.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::{
    any::Any,
    sync::{
        Arc, Mutex, OnceLock, PoisonError, Weak,
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use dashmap::mapref::entry::Entry;
use tracing::debug;

use crate::{
    memory::{SAMPLE_EVERY, process_limit},
    runtime_metrics::rss::{status_anon_rss_bytes, total_memory_bytes},
    superfile::SuperfileReader,
    supertable::{
        manifest::SuperfileUri,
        reader_cache::disk::{CachedEntry, DiskCacheStore, EntryAccounting, UnindexedFile},
    },
};

/// Share of the process memory limit (else of the machine's memory) the cache may keep as decoded
/// metadata. The rest is for the queries themselves.
const HEAP_BUDGET_PERCENT: u64 = 25;

/// Batches of released readers waiting to be freed: past this many, the caller frees its own, so
/// a burst of statements cannot pile up memory that is gone from the cache but not yet freed.
const FREE_QUEUE_BATCHES: usize = 2;

/// The freeing thread's name, as a stack dump shows it.
const FREE_THREAD_NAME: &str = "infino-cache-free";

/// Free `released` on one long-lived thread rather than the caller's: dropping thousands of
/// decoded footers and page indexes takes seconds, and no query should wait on another's. The
/// cache has already let them go and given back their count. Freed here when the thread cannot
/// start or its queue is full; whatever is queued at exit is left to the process's end.
pub(crate) fn free_elsewhere<T: Any + Send>(released: Vec<T>) {
    static QUEUE: OnceLock<Option<SyncSender<Box<dyn Any + Send>>>> = OnceLock::new();
    if released.is_empty() {
        return;
    }
    let queue = QUEUE.get_or_init(|| {
        let (queue, batches) = sync_channel::<Box<dyn Any + Send>>(FREE_QUEUE_BATCHES);
        thread::Builder::new()
            .name(FREE_THREAD_NAME.into())
            .spawn(move || batches.iter().for_each(drop))
            .ok()
            .map(|_| queue)
    });
    if let Some(queue) = queue {
        // Full or gone: the batch comes back and is freed below.
        let _ = queue.try_send(Box::new(released));
    }
}

/// [`anon_resident`]'s mark for a reading not taken yet.
const NEVER_READ: u64 = u64::MAX;

/// The process's budget. Each test builds its own, so tests never share a total.
static PROCESS: HeapBudget = HeapBudget::new();

/// [`HEAP_BUDGET_PERCENT`] of the process memory limit, else of the machine's memory. `None` when
/// neither is known: then nothing is dropped.
fn heap_budget_bytes() -> Option<u64> {
    memory_limit().map(|bytes| bytes / 100 * HEAP_BUDGET_PERCENT)
}

/// The process memory limit, else the machine's memory.
fn memory_limit() -> Option<u64> {
    static HOST: OnceLock<Option<u64>> = OnceLock::new();
    process_limit().or_else(|| *HOST.get_or_init(total_memory_bytes))
}

/// Whether the rest of the process leaves cached metadata less room than its `budget`: anonymous
/// resident memory above the limit minus the budget. A reading that cannot be taken counts as yes.
fn short_of_memory(budget: u64) -> bool {
    let Some(limit) = memory_limit() else {
        return true;
    };
    anon_resident().is_none_or(|anon| anon > limit.saturating_sub(budget))
}

/// The process's anonymous resident bytes, read again at most once per [`SAMPLE_EVERY`], so a
/// caller asking per file costs two atomic loads, and the answer is as fresh as the limit check's.
fn anon_resident() -> Option<u64> {
    static START: OnceLock<Instant> = OnceLock::new();
    static READ_AT_US: AtomicU64 = AtomicU64::new(NEVER_READ);
    static ANON: AtomicU64 = AtomicU64::new(0);
    let now = START.get_or_init(Instant::now).elapsed().as_micros() as u64;
    let read_at = READ_AT_US.load(Ordering::Acquire);
    if read_at != NEVER_READ && now.saturating_sub(read_at) < SAMPLE_EVERY.as_micros() as u64 {
        return Some(ANON.load(Ordering::Acquire));
    }
    let anon = status_anon_rss_bytes()?;
    ANON.store(anon, Ordering::Release);
    READ_AT_US.store(now, Ordering::Release);
    Some(anon)
}

/// Decoded bytes counted in a budget until this drops. Kept next to what it pays for, so the bytes
/// come back however that ends.
#[derive(Debug)]
#[must_use = "the bytes go back as soon as the reservation drops"]
pub(crate) struct HeapReservation {
    budget: &'static HeapBudget,
    bytes: u64,
}

impl Drop for HeapReservation {
    fn drop(&mut self) {
        // Saturating: each reservation gives back once, but a miscount must not wrap.
        let _ = self
            .budget
            .charged
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |charged| {
                Some(charged.saturating_sub(self.bytes))
            });
    }
}

/// A total of reserved bytes, and the caches whose files may leave to keep it under budget.
#[derive(Debug)]
pub(crate) struct HeapBudget {
    charged: AtomicU64,
    caches: Mutex<Vec<Weak<DiskCacheStore>>>,
    /// A test's own budget, in place of the process's.
    #[cfg(test)]
    fixed: Option<Option<u64>>,
    /// A test's own answer to whether the process is short of memory.
    #[cfg(test)]
    short: AtomicBool,
}

impl HeapBudget {
    const fn new() -> Self {
        Self {
            charged: AtomicU64::new(0),
            caches: Mutex::new(Vec::new()),
            #[cfg(test)]
            fixed: None,
            #[cfg(test)]
            short: AtomicBool::new(true),
        }
    }

    /// The process's budget.
    pub(crate) fn process() -> &'static Self {
        &PROCESS
    }

    /// A budget of its own for one test, `None` for no budget.
    #[cfg(test)]
    pub(crate) fn with_budget(budget: Option<u64>) -> &'static Self {
        Box::leak(Box::new(Self {
            fixed: Some(budget),
            ..Self::new()
        }))
    }

    fn budget(&self) -> Option<u64> {
        #[cfg(test)]
        if let Some(budget) = self.fixed {
            return budget;
        }
        heap_budget_bytes()
    }

    /// Make this test budget see the process as short of memory, or not.
    #[cfg(test)]
    pub(crate) fn set_short_of_memory(&self, short: bool) {
        self.short.store(short, Ordering::Release);
    }

    /// Whether the process is short of memory for cached metadata: anonymous resident memory
    /// above the limit minus the budget. With room to spare, keep what is decoded, as before.
    pub(crate) fn short_of_memory(&self) -> bool {
        #[cfg(test)]
        if self.fixed.is_some() {
            return self.short.load(Ordering::Acquire);
        }
        self.budget().is_some_and(short_of_memory)
    }

    /// The bytes reserved now.
    #[cfg(test)]
    pub(crate) fn charged(&self) -> u64 {
        self.charged.load(Ordering::Acquire)
    }

    /// Reserve `bytes` for a part kept anyway. Drops nothing now: the next
    /// [`Self::release_over_budget`] does, once queries let go of their files.
    pub(crate) fn reserve(&'static self, bytes: u64) -> HeapReservation {
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        HeapReservation {
            budget: self,
            bytes,
        }
    }

    /// Drop the oldest cached files no query holds until the total is back under budget. Does
    /// nothing under it, with no budget, or while the process has room to spare: then the cache
    /// keeps what it holds, as before.
    pub(crate) fn release_over_budget(&self) {
        let Some(budget) = self.budget() else { return };
        let charged = self.charged.load(Ordering::Acquire);
        if charged <= budget || !self.short_of_memory() {
            return;
        }
        let caches: Vec<Arc<DiskCacheStore>> = {
            let mut list = self.caches.lock().unwrap_or_else(PoisonError::into_inner);
            list.retain(|cache| cache.strong_count() > 0);
            list.iter().filter_map(Weak::upgrade).collect()
        };
        let mut oldest: Vec<(u64, usize, SuperfileUri)> = Vec::new();
        for (at, cache) in caches.iter().enumerate() {
            for entry in cache.cached.iter() {
                if entry.value().heap.get().is_some()
                    && cache.heap_reclaimable(entry.key(), entry.value())
                {
                    let last = entry.value().last_access_us.load(Ordering::Acquire);
                    oldest.push((last, at, *entry.key()));
                }
            }
        }
        oldest.sort_unstable();
        let mut released = Vec::new();
        for (_, at, uri) in oldest {
            if self.charged.load(Ordering::Acquire) <= budget {
                break;
            }
            released.extend(caches[at].release_for_heap(&uri));
        }
        // The lazy files past the cap would otherwise go at the next query's first open, which
        // would pay for freeing them.
        for cache in &caches {
            cache.release_unheld_lazy_entries_over_cap();
        }
        free_elsewhere(released);
        debug!(
            target: "infino::cache",
            budget,
            before = charged,
            after = self.charged.load(Ordering::Acquire),
            "decoded metadata over budget: dropped the oldest unheld files"
        );
    }

    /// Let this budget drop `cache`'s files.
    fn register(&self, cache: &Arc<DiskCacheStore>) {
        let mut list = self.caches.lock().unwrap_or_else(PoisonError::into_inner);
        list.retain(|cache| cache.strong_count() > 0);
        list.push(Arc::downgrade(cache));
    }
}

impl DiskCacheStore {
    /// Let the process budget drop this cache's files.
    pub(crate) fn register_heap(self: &Arc<Self>) {
        HeapBudget::process().register(self);
    }

    /// The budget this cache's files count against: the process's, or a test's own.
    pub(crate) fn heap_budget(&self) -> &'static HeapBudget {
        #[cfg(test)]
        if let Some(budget) = self.heap_budget.get() {
            return budget;
        }
        HeapBudget::process()
    }

    /// Count this cache's files against `budget` instead, before any is counted.
    #[cfg(test)]
    pub(crate) fn use_heap_budget(self: &Arc<Self>, budget: &'static HeapBudget) {
        self.heap_budget
            .set(budget)
            .expect("a cache's budget is set once");
        budget.register(self);
    }

    /// Count the footer and page index `uri`'s reader keeps against the budget, once, so a warm
    /// open measures nothing. The count goes back when the file leaves the cache.
    pub(crate) fn charge_heap_once(&self, uri: &SuperfileUri, bytes: impl FnOnce() -> u64) {
        let Some(entry) = self.cached.get(uri).map(|entry| Arc::clone(entry.value())) else {
            return;
        };
        entry
            .heap
            .get_or_init(|| self.heap_budget().reserve(bytes()));
    }

    /// Whether `entry` may leave for the budget: no query holds it, and nothing is still writing or
    /// downloading it. A mapped copy missing its vector blob stays: its blob is still filling in.
    fn heap_reclaimable(&self, uri: &SuperfileUri, entry: &CachedEntry) -> bool {
        match entry.mmap() {
            None => self.reclaimable_lazy_entry(uri, entry),
            Some(_) => {
                !entry.held_by_query()
                    && entry.accounting == EntryAccounting::Eager
                    && entry.block_source().is_none()
                    && !self
                        .fills
                        .get(uri)
                        .is_some_and(|fill| fill.downloading.load(Ordering::Acquire))
            }
        }
    }

    /// Drop `uri` from the cache for the budget. A lazy file goes as it does past the lazy cap. A
    /// mapped file keeps its bytes on disk, counted in the disk budget and recorded as a restart
    /// would find it, so the next open maps it again.
    fn release_for_heap(&self, uri: &SuperfileUri) -> Option<Arc<SuperfileReader>> {
        let slot = self.cached.entry(*uri);
        let Entry::Occupied(slot) = slot else {
            return None;
        };
        if slot.get().mmap().is_none() {
            drop(slot);
            return self.release_lazy_entry(uri, self.now_us());
        }
        if !self.heap_reclaimable(uri, slot.get()) {
            return None;
        }
        // Recorded under the entry's lock, as a landed download is, so an open cannot find the
        // file on disk with neither an entry nor a record and count it twice. Stamped now, in the
        // records' clock: it was used more recently than a file a restart found.
        let file = UnindexedFile {
            size_bytes: slot.get().size_bytes.load(Ordering::Acquire),
            mtime_us: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |since| since.as_micros() as u64),
        };
        self.unindexed.insert(*uri, file);
        Some(Arc::clone(&slot.remove().reader))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, OnceLock,
            atomic::{AtomicU64, Ordering},
            mpsc::{Receiver, Sender, channel},
        },
        thread,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    use crate::{
        superfile::SuperfileReader,
        supertable::{
            manifest::SuperfileUri,
            reader_cache::{
                ColdFetchMode, ReadIntent,
                disk::{
                    CachedEntry, DiskCacheStore, EntryAccounting, Residency,
                    heap::{FREE_THREAD_NAME, HeapBudget, free_elsewhere},
                    test_support::*,
                },
            },
        },
    };

    /// Bytes each test entry counts against the budget.
    const ENTRY_BYTES: u64 = 100;

    /// How long a test waits for the freeing thread before it fails.
    const FREE_DEADLINE: Duration = Duration::from_secs(10);

    /// A value whose drop reports the thread it ran on, once `go` lets it finish.
    struct Freed {
        go: Receiver<()>,
        ran_on: Sender<Option<String>>,
    }

    impl Drop for Freed {
        fn drop(&mut self) {
            let _ = self.go.recv_timeout(FREE_DEADLINE);
            let _ = self
                .ran_on
                .send(thread::current().name().map(str::to_string));
        }
    }

    /// Cache a lazy entry for a new file, last used at `last_access_us`, counting [`ENTRY_BYTES`].
    fn charged_entry(store: &Arc<DiskCacheStore>, last_access_us: u64) -> SuperfileUri {
        let uri = SuperfileUri::new_v4();
        let reader = SuperfileReader::open(tiny_superfile_bytes()).expect("tiny superfile opens");
        store.cached.insert(
            uri,
            Arc::new(CachedEntry {
                heap: OnceLock::new(),
                reader: Arc::new(reader),
                residency: Residency::Paged {
                    block_source: dummy_block_source(store, uri),
                },
                size_bytes: Arc::new(AtomicU64::new(0)),
                accounting: EntryAccounting::SourceOwned,
                last_access_us: AtomicU64::new(last_access_us),
            }),
        );
        store.charge_heap_once(&uri, || ENTRY_BYTES);
        uri
    }

    /// A cache with a budget of its own.
    fn store_with_budget(budget: Option<u64>) -> (tempfile::TempDir, Arc<DiskCacheStore>) {
        let (dir, store) = test_store();
        store.use_heap_budget(HeapBudget::with_budget(budget));
        (dir, store)
    }

    #[test]
    fn a_reservation_counts_until_it_drops() {
        // A reservation of 100 bytes, then dropped.
        // This test pins that the total follows what is kept.
        let budget = HeapBudget::with_budget(Some(0));
        let kept = budget.reserve(ENTRY_BYTES);
        assert_eq!(budget.charged(), ENTRY_BYTES);
        drop(kept);
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn the_pass_drops_the_oldest_files_no_query_holds_until_under_budget() {
        // Three counted files in a budget of two; the oldest is held by a query.
        //  - the pass drops the oldest unheld file, not the held one or the newest;
        //  - the dropped file's count comes back with it.
        // This test pins the drop order and that a held file never goes.
        let (_dir, store) = store_with_budget(Some(2 * ENTRY_BYTES));
        let budget = store.heap_budget();
        let held = charged_entry(&store, 1);
        let oldest_unheld = charged_entry(&store, 2);
        let newest = charged_entry(&store, 3);
        let query = store
            .cached
            .get(&held)
            .map(|e| Arc::clone(&e.value().reader));
        budget.release_over_budget();
        assert!(store.is_cached(&held), "a held file stays");
        assert!(
            !store.is_cached(&oldest_unheld),
            "the oldest unheld file goes"
        );
        assert!(store.is_cached(&newest));
        assert_eq!(budget.charged(), 2 * ENTRY_BYTES);
        drop(query);
    }

    #[test]
    fn a_held_file_goes_on_the_pass_after_its_last_holder_drops() {
        // One counted file over a budget of 0, held by a query.
        //  - the pass at that moment keeps it;
        //  - once the query drops its reader, the next pass drops it and its count.
        // This test pins that the end-of-statement pass frees what the statement held.
        let (_dir, store) = store_with_budget(Some(0));
        let budget = store.heap_budget();
        let uri = charged_entry(&store, 1);
        let query = store
            .cached
            .get(&uri)
            .map(|e| Arc::clone(&e.value().reader));
        budget.release_over_budget();
        assert!(store.is_cached(&uri), "held: kept");
        assert_eq!(budget.charged(), ENTRY_BYTES);
        drop(query);
        budget.release_over_budget();
        assert!(!store.is_cached(&uri), "unheld: dropped");
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn with_room_to_spare_nothing_is_dropped() {
        // A file over a budget of 0, unheld, while the process is not short of memory.
        //  - the pass keeps it, as the cache did before the budget;
        //  - once the process is short of memory, the next pass drops it.
        // This test pins that a table over the budget behaves as before until memory is short.
        let (_dir, store) = store_with_budget(Some(0));
        let budget = store.heap_budget();
        let uri = charged_entry(&store, 1);
        budget.set_short_of_memory(false);
        budget.release_over_budget();
        assert!(store.is_cached(&uri), "room to spare: kept");
        budget.set_short_of_memory(true);
        budget.release_over_budget();
        assert!(!store.is_cached(&uri), "short of memory: dropped");
    }

    #[test]
    fn released_readers_are_freed_on_the_freeing_thread_not_the_callers() {
        // A released value whose drop blocks until the test lets it go.
        //  - handing it over returns at once, while its drop still waits;
        //  - the drop then runs on the freeing thread.
        // This test pins that no query waits for freeing what the cache let go.
        let (go, wait) = channel();
        let (ran_on, report) = channel();
        free_elsewhere(vec![Freed { go: wait, ran_on }]);
        assert!(
            report.try_recv().is_err(),
            "the caller did not run the drop"
        );
        go.send(()).expect("let it go");
        let name = report.recv_timeout(FREE_DEADLINE).expect("dropped");
        assert_eq!(name.as_deref(), Some(FREE_THREAD_NAME));
    }

    #[test]
    fn the_pass_also_lets_go_of_lazy_files_past_the_cap() {
        // Three unheld lazy files counted over a budget of two, and a lazy cap of one.
        // This test pins that the pass leaves no files past the cap for the next query's first
        // open to free.
        let (_dir, store) = store_with_budget(Some(2 * ENTRY_BYTES));
        store.set_lazy_entry_cap(1);
        let files: Vec<_> = (1..=3).map(|at| charged_entry(&store, at)).collect();
        store.heap_budget().release_over_budget();
        let kept = files.iter().filter(|uri| store.is_cached(uri)).count();
        assert_eq!(kept, 1, "only the cap's worth stays");
    }

    #[test]
    fn with_no_budget_nothing_is_dropped() {
        // Counted files and no budget (no memory limit, no readable machine memory).
        // This test pins that the cache then behaves as before.
        let (_dir, store) = store_with_budget(None);
        let uri = charged_entry(&store, 1);
        store.heap_budget().release_over_budget();
        assert!(store.is_cached(&uri));
    }

    #[test]
    fn a_file_is_charged_once_and_its_charge_goes_with_it() {
        // One file counted twice, then dropped by a path that knows nothing of the budget (the
        // lazy cap's release).
        // This test pins that a warm open never counts twice and every way out refunds.
        let (_dir, store) = store_with_budget(None);
        let budget = store.heap_budget();
        let uri = charged_entry(&store, 1);
        store.charge_heap_once(&uri, || unreachable!("charged already: nothing measured"));
        assert_eq!(budget.charged(), ENTRY_BYTES);
        store.release_lazy_entry(&uri, store.now_us());
        assert!(!store.is_cached(&uri));
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_mapped_file_keeps_its_file_and_disk_charge_when_it_leaves() {
        // A whole file mapped from local disk, counted, over a budget of 0.
        //  - the pass drops it from the cache with its count, and keeps its file on disk, still
        //    counted in the disk budget;
        //  - its disk record is stamped in the records' clock (time since the unix epoch);
        //  - the next open maps it again from disk, with no download.
        // This test pins that dropping metadata never costs a download or skews disk eviction.
        let (_dir, store) = store_with_budget(Some(0));
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        let (key, storage) = (uri.storage_path(), Arc::clone(&store.storage));
        let load = || store.reader_synchronous_with_storage(&uri, &key, Arc::clone(&storage));
        load().await.expect("load");
        store.charge_heap_once(&uri, || ENTRY_BYTES);
        let disk_bytes = store.current_bytes.load(Ordering::Acquire);
        let before = SystemTime::now().duration_since(UNIX_EPOCH).expect("clock");
        store.heap_budget().release_over_budget();
        assert!(!store.is_cached(&uri), "the file leaves the cache");
        assert_eq!(store.heap_budget().charged(), 0);
        assert!(store.cache_path(&uri).exists(), "its bytes stay on disk");
        assert_eq!(store.current_bytes.load(Ordering::Acquire), disk_bytes);
        let stamped = store.unindexed.get(&uri).expect("recorded").mtime_us;
        assert!(
            stamped >= before.as_micros() as u64,
            "stamped in the records' clock"
        );
        store.assert_budget_consistent();
        let fetches = store.stats().n_cold_fetches;
        load().await.expect("reopen");
        assert_eq!(
            store.stats().n_cold_fetches,
            fetches,
            "mapped again, no download"
        );
        store.assert_budget_consistent();
    }

    #[tokio::test]
    async fn a_mapped_copy_missing_its_vector_blob_never_leaves() {
        // A vector file promoted to a mapped copy that leaves its vector blob on the block cache,
        // counted, over a budget of 0.
        // This test pins that the pass keeps it: its blob is still filling in, and its disk
        // record could not describe the copy.
        let (_dir, store) = test_store_with(|cfg| {
            cfg.cold_fetch_mode = ColdFetchMode::LazyForegroundWithBackgroundFill;
        });
        store.use_heap_budget(HeapBudget::with_budget(Some(0)));
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_vector_superfile_bytes()).await;
        for intent in [ReadIntent::Stream, ReadIntent::Warm] {
            store
                .open_for_query(&uri, &uri.storage_path(), None, None, intent)
                .await
                .expect("open");
        }
        store
            .wait_until_mmap_promoted(&uri, PROMOTE_TIMEOUT)
            .await
            .expect("promote");
        assert!(
            store
                .cached
                .get(&uri)
                .is_some_and(|e| e.block_source().is_some())
        );
        store.charge_heap_once(&uri, || ENTRY_BYTES);
        store.heap_budget().release_over_budget();
        assert!(
            store.is_cached(&uri),
            "a copy missing its vector blob stays"
        );
        assert_eq!(store.heap_budget().charged(), ENTRY_BYTES);
    }
}
