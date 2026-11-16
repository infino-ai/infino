// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The decoded file metadata the reader cache keeps between queries, held to one process budget.
//!
//! A cached file's reader keeps its decoded footer and page index after a query ends, and the disk
//! budget does not count that memory. So:
//! - each counted cache entry holds a [`HeapReservation`], which gives its bytes back when it drops;
//! - past the budget, and only while the rest of the process leaves the metadata less room than the
//!   budget, the files read lazily or held whole in memory that no query holds leave the cache,
//!   oldest first, keeping their downloaded blocks on disk. A file mapped whole from local disk is
//!   not counted, and this budget never drops it.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::{
    any::Any,
    cmp::Reverse,
    sync::{
        Arc, Mutex, OnceLock, PoisonError, Weak,
        atomic::{AtomicU64, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
};

use tracing::debug;

use crate::{
    memory::process_limit,
    runtime_metrics::rss::{status_anon_rss_bytes, total_memory_bytes},
    superfile::SuperfileReader,
    supertable::{manifest::SuperfileUri, reader_cache::disk::DiskCacheStore},
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
/// resident memory above `limit` minus the budget. A reading that cannot be taken counts as yes.
fn short_of_memory(limit: u64, budget: u64) -> bool {
    status_anon_rss_bytes().is_none_or(|anon| anon > limit.saturating_sub(budget))
}

/// Decoded bytes counted in a budget until this drops. Kept next to what it pays for, so the bytes
/// come back however that ends.
#[derive(Debug)]
#[must_use = "the bytes go back as soon as the reservation drops"]
pub(crate) struct HeapReservation {
    budget: &'static HeapBudget,
    bytes: AtomicU64,
}

impl HeapReservation {
    /// Count `more` bytes in this reservation, kept until it drops.
    pub(crate) fn add(&self, more: u64) {
        self.budget.charged.fetch_add(more, Ordering::AcqRel);
        self.bytes.fetch_add(more, Ordering::AcqRel);
    }
}

impl Drop for HeapReservation {
    fn drop(&mut self) {
        // Saturating: each reservation gives back once, but a miscount must not wrap.
        let _ = self
            .budget
            .charged
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |charged| {
                Some(charged.saturating_sub(self.bytes.load(Ordering::Acquire)))
            });
    }
}

/// A total of reserved bytes, and the caches whose files may leave to keep it under budget.
#[derive(Debug)]
pub(crate) struct HeapBudget {
    /// Bytes all live reservations hold.
    charged: AtomicU64,
    /// The caches whose files this budget may drop; dead ones are pruned when the list is touched.
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
        memory_limit()
            .zip(self.budget())
            .is_some_and(|(limit, budget)| short_of_memory(limit, budget))
    }

    /// The bytes reserved now.
    #[cfg(test)]
    pub(crate) fn charged(&self) -> u64 {
        self.charged.load(Ordering::Acquire)
    }

    /// Count `bytes` of decoded metadata until the reservation drops. Drops no file now: the next
    /// [`Self::release_over_budget`] does, once queries let go of their files.
    pub(crate) fn reserve(&'static self, bytes: u64) -> HeapReservation {
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        HeapReservation {
            budget: self,
            bytes: AtomicU64::new(bytes),
        }
    }

    /// Drop the oldest counted files no query holds until the total is back under budget, then each
    /// cache's unheld lazy files past its cap. Does nothing under budget, with no budget, or while
    /// the process has room to spare: then the cache keeps what it holds, as before.
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
        // Each cache stamps accesses on its own clock, started when it was, so files are ranked by
        // how long each has been idle, not by their stamps.
        let mut oldest: Vec<(Reverse<u64>, usize, SuperfileUri)> = Vec::new();
        for (at, cache) in caches.iter().enumerate() {
            let now_us = cache.now_us();
            for entry in cache.cached.iter() {
                if entry.value().heap.get().is_some()
                    && cache.reclaimable_lazy_entry(entry.key(), entry.value())
                {
                    let last = entry.value().last_access_us.load(Ordering::Acquire);
                    oldest.push((Reverse(now_us.saturating_sub(last)), at, *entry.key()));
                }
            }
        }
        oldest.sort_unstable();
        let mut released = Vec::new();
        for (_, at, uri) in oldest {
            if self.charged.load(Ordering::Acquire) <= budget {
                break;
            }
            released.extend(caches[at].release_lazy_entry(&uri, caches[at].now_us()));
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

    /// Count `bytes` more of what `reader` keeps decoded against the budget, until `uri`'s file
    /// leaves the cache. Counted only while the cache still holds that reader for `uri`, so an
    /// evicted file's reader never counts against its replacement. A file mapped whole from local
    /// disk is not counted: this budget never drops it.
    pub(crate) fn charge_heap(
        &self,
        uri: &SuperfileUri,
        reader: &Arc<SuperfileReader>,
        bytes: u64,
    ) {
        let Some(entry) = self.cached.get(uri).map(|entry| Arc::clone(entry.value())) else {
            return;
        };
        if entry.mmap().is_some() || !Arc::ptr_eq(&entry.reader, reader) {
            return;
        }
        entry
            .heap
            .get_or_init(|| self.heap_budget().reserve(0))
            .add(bytes);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{
            Arc, OnceLock,
            atomic::AtomicU64,
            mpsc::{Receiver, Sender, channel},
        },
        thread,
        time::Duration,
    };

    use crate::{
        superfile::SuperfileReader,
        supertable::{
            manifest::SuperfileUri,
            reader_cache::disk::{
                CachedEntry, DiskCacheStore, EntryAccounting, Residency,
                heap::{FREE_THREAD_NAME, HeapBudget, free_elsewhere},
                test_support::*,
            },
        },
    };

    /// Bytes each test entry counts against the budget.
    const ENTRY_BYTES: u64 = 100;

    /// How long a test waits for the freeing thread before it fails.
    const FREE_DEADLINE: Duration = Duration::from_secs(10);

    /// How far apart a test starts two caches' clocks and stamps their files.
    const CLOCK_STEP: Duration = Duration::from_millis(10);

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
        let reader =
            Arc::new(SuperfileReader::open(tiny_superfile_bytes()).expect("tiny superfile opens"));
        store.cached.insert(
            uri,
            Arc::new(CachedEntry {
                heap: OnceLock::new(),
                reader: Arc::clone(&reader),
                residency: Residency::Paged {
                    block_source: dummy_block_source(store, uri),
                },
                size_bytes: Arc::new(AtomicU64::new(0)),
                accounting: EntryAccounting::SourceOwned,
                last_access_us: AtomicU64::new(last_access_us),
            }),
        );
        store.charge_heap(&uri, &reader, ENTRY_BYTES);
        uri
    }

    /// The reader the cache holds for `uri`.
    fn cached_reader(store: &DiskCacheStore, uri: &SuperfileUri) -> Arc<SuperfileReader> {
        store
            .cached
            .get(uri)
            .map(|entry| Arc::clone(&entry.reader))
            .expect("cached")
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
    fn files_of_two_caches_go_by_idle_time_not_by_their_caches_clocks() {
        // Two caches over one budget of one file. The older cache's file was last used one step
        // ago; the newer cache, started later, stamps its file now with a smaller number.
        //  - the pass drops the older cache's file, idle longer, and keeps the one just used.
        // This test pins that caches started at different times are ranked on one scale.
        let budget = HeapBudget::with_budget(Some(ENTRY_BYTES));
        let (_older_dir, older) = test_store();
        older.use_heap_budget(budget);
        thread::sleep(2 * CLOCK_STEP);
        let idle = charged_entry(&older, older.now_us());
        let (_newer_dir, newer) = test_store();
        newer.use_heap_budget(budget);
        thread::sleep(CLOCK_STEP);
        let hot = charged_entry(&newer, newer.now_us());
        budget.release_over_budget();
        assert!(!older.is_cached(&idle), "the file idle longer goes");
        assert!(newer.is_cached(&hot), "the file just used stays");
        assert_eq!(budget.charged(), ENTRY_BYTES);
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
    fn a_files_count_goes_with_it_whichever_way_it_leaves() {
        // One file counted in two parts, then dropped by a path that knows nothing of the budget
        // (the lazy cap's release).
        // This test pins that a count grows with what the reader keeps and every way out refunds.
        let (_dir, store) = store_with_budget(None);
        let budget = store.heap_budget();
        let uri = charged_entry(&store, 1);
        store.charge_heap(&uri, &cached_reader(&store, &uri), ENTRY_BYTES);
        assert_eq!(budget.charged(), 2 * ENTRY_BYTES);
        store.release_lazy_entry(&uri, store.now_us());
        assert!(!store.is_cached(&uri));
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn a_reader_the_cache_no_longer_holds_is_not_counted() {
        // A file counted once by its cached reader, then counted again through another reader of
        // the same file, as a statement that opened it before an eviction and a reopen would.
        // This test pins that only the reader the cache holds counts, so the file counts once.
        let (_dir, store) = store_with_budget(None);
        let uri = charged_entry(&store, 1);
        let evicted =
            Arc::new(SuperfileReader::open(tiny_superfile_bytes()).expect("tiny superfile opens"));
        store.charge_heap(&uri, &evicted, ENTRY_BYTES);
        assert_eq!(store.heap_budget().charged(), ENTRY_BYTES, "counted once");
    }

    #[tokio::test]
    async fn a_mapped_file_is_not_counted_or_dropped() {
        // A whole file mapped from local disk, over a budget of 0, the process short of memory.
        //  - counting it counts nothing;
        //  - the pass keeps it, and the disk budget still matches what is on disk.
        // This test pins that the budget covers only files that can leave.
        let (_dir, store) = store_with_budget(Some(0));
        let uri = SuperfileUri::new_v4();
        put_superfile(&store, &uri, tiny_superfile_bytes()).await;
        let (key, storage) = (uri.storage_path(), Arc::clone(&store.storage));
        store
            .reader_synchronous_with_storage(&uri, &key, storage)
            .await
            .expect("load");
        assert!(
            store.cached.get(&uri).is_some_and(|e| e.mmap().is_some()),
            "mapped"
        );
        store.charge_heap(&uri, &cached_reader(&store, &uri), ENTRY_BYTES);
        assert_eq!(store.heap_budget().charged(), 0, "not counted");
        store.heap_budget().release_over_budget();
        assert!(store.is_cached(&uri), "kept");
        store.assert_budget_consistent();
    }
}
