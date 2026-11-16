// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The decoded file metadata the reader cache keeps between queries, held to one process budget.
//!
//! A cached file's reader keeps its decoded footer and page index after a query ends, and the disk
//! budget does not count that memory. So:
//! - each counted cache entry holds a [`HeapReservation`], which gives its bytes back when it drops;
//! - past the budget, and only while the rest of the process leaves the metadata less room than the
//!   budget, the files read lazily or held whole in memory that no query holds leave the cache,
//!   oldest first, until they fit what queries leave of the budget (never under a quarter of it),
//!   keeping their downloaded blocks on disk; when held files fill it, tables' cached SQL sessions
//!   go too. A file mapped whole from local disk is not counted, and this budget never drops it.

#[cfg(test)]
use std::sync::atomic::AtomicBool;
use std::{
    any::Any,
    cmp::Reverse,
    ptr,
    sync::{
        Arc, Mutex, OnceLock, PoisonError, TryLockError, Weak,
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

/// When held files fill the budget, the unheld files kept are a quarter of it: the files small
/// statements just used stay, rather than leave and reopen at every statement end.
const UNHELD_FLOOR_DIVISOR: u64 = 4;

/// Batches of released readers waiting to be freed: past this many, the caller frees its own, so
/// a burst of statements cannot pile up memory that is gone from the cache but not yet freed.
const FREE_QUEUE_BATCHES: usize = 2;

/// The freeing thread's name, as a stack dump shows it.
const FREE_THREAD_NAME: &str = "infino-cache-free";

/// Batches of released readers, freed on one long-lived thread rather than the caller's: dropping
/// thousands of decoded footers and page indexes takes seconds, and no query should wait on
/// another's. The cache has already let them go and given back their count.
struct FreeQueue {
    /// The thread's end of the queue, started on first use; `None` if the thread could not start.
    sender: OnceLock<Option<SyncSender<Box<dyn Any + Send>>>>,
}

impl FreeQueue {
    const fn new() -> Self {
        Self {
            sender: OnceLock::new(),
        }
    }

    /// Free `released` on the queue's thread. When the thread cannot start, or
    /// [`FREE_QUEUE_BATCHES`] batches are already waiting, they are freed here and the caller waits
    /// for it. Whatever is queued at exit is left to the process's end.
    fn free<T: Any + Send>(&self, released: Vec<T>) {
        if released.is_empty() {
            return;
        }
        let sender = self.sender.get_or_init(|| {
            let (sender, batches) = sync_channel::<Box<dyn Any + Send>>(FREE_QUEUE_BATCHES);
            thread::Builder::new()
                .name(FREE_THREAD_NAME.into())
                .spawn(move || batches.iter().for_each(drop))
                .ok()
                .map(|_| sender)
        });
        if let Some(sender) = sender {
            // Full or gone: the batch comes back and is freed below.
            let _ = sender.try_send(Box::new(released));
        }
    }
}

/// The process's freeing thread.
static FREE: FreeQueue = FreeQueue::new();

/// Free `released` on the process's freeing thread (see [`FreeQueue::free`]).
pub(crate) fn free_elsewhere<T: Any + Send>(released: Vec<T>) {
    FREE.free(released);
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

/// Whether `anon` resident bytes leave cached metadata less room than its `budget`: above `limit`
/// minus the budget. A reading that cannot be taken counts as no, as for the SQL memory limit.
fn leaves_too_little_room(anon: Option<u64>, limit: u64, budget: u64) -> bool {
    anon.is_some_and(|anon| anon > limit.saturating_sub(budget))
}

/// The bytes of unheld files to keep when running statements hold `held` bytes of `share`: what is
/// left of it, but never less than 1/[`UNHELD_FLOOR_DIVISOR`] of it.
fn unheld_target(share: u64, held: u64) -> u64 {
    (share / UNHELD_FLOOR_DIVISOR).max(share.saturating_sub(held))
}

/// What one walk over the caches found: the unheld files counted against a budget, oldest first
/// (idle time, cache, file, bytes), and the bytes held and unheld.
struct Scan {
    oldest: Vec<(Reverse<u64>, usize, SuperfileUri, u64)>,
    unheld: u64,
    held: u64,
}

/// What keeps files between statements and can let them go: a table's cached SQL session.
pub(crate) trait CachedSessions: Send + Sync {
    /// Drop the cached sessions, outside any lock, so the files they prepared can leave.
    fn drop_cached_sessions(&self);
}

/// Decoded bytes counted in a budget until this drops. Kept next to what it pays for, so the bytes
/// come back however that ends.
#[derive(Debug)]
#[must_use = "the bytes go back as soon as the reservation drops"]
pub(crate) struct HeapReservation {
    budget: &'static HeapBudget,
    bytes: AtomicU64,
    /// A doc-id map's, also counted in the budget's maps total.
    map: bool,
}

impl HeapReservation {
    /// The bytes counted now.
    fn bytes(&self) -> u64 {
        self.bytes.load(Ordering::Acquire)
    }

    /// Whether these bytes count against `budget`.
    fn counts_against(&self, budget: &HeapBudget) -> bool {
        ptr::eq(self.budget, budget)
    }

    /// Count `more` bytes in this reservation, kept until it drops.
    pub(crate) fn add(&self, more: u64) {
        self.budget.charged.fetch_add(more, Ordering::AcqRel);
        if self.map {
            self.budget.maps.fetch_add(more, Ordering::AcqRel);
        }
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
        if self.map {
            let _ = self
                .budget
                .maps
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |maps| {
                    Some(maps.saturating_sub(self.bytes.load(Ordering::Acquire)))
                });
        }
    }
}

/// A total of reserved bytes, and the caches whose files may leave to keep it under budget.
#[derive(Debug)]
pub(crate) struct HeapBudget {
    /// Bytes all live reservations hold.
    charged: AtomicU64,
    /// The part of `charged` that is full-text doc-id maps. Counted apart because SQL metadata may
    /// sit above the budget while memory is to spare, so the shared total can't bound map keeps.
    maps: AtomicU64,
    /// The caches whose files this budget may drop; dead ones are pruned when the list is touched.
    caches: Mutex<Vec<Weak<DiskCacheStore>>>,
    /// The tables whose cached sessions hold files between statements; pruned like `caches`.
    sessions: Mutex<Vec<Weak<dyn CachedSessions>>>,
    /// Held by the running pass: one at a time, so two statements ending together do not both drop
    /// files for the same excess.
    pass: Mutex<()>,
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
            maps: AtomicU64::new(0),
            caches: Mutex::new(Vec::new()),
            sessions: Mutex::new(Vec::new()),
            pass: Mutex::new(()),
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
            .is_some_and(|(limit, budget)| {
                leaves_too_little_room(status_anon_rss_bytes(), limit, budget)
            })
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
            map: false,
        }
    }

    /// Reserve `bytes` for a doc-id map a query can do without: `None` while the process is short
    /// of memory, or when kept maps would pass the budget. Refusing is one compare-and-swap; it
    /// drops nothing.
    pub(crate) fn try_reserve_map(&'static self, bytes: u64) -> Option<HeapReservation> {
        if self.short_of_memory() {
            return None;
        }
        let budget = self.budget().unwrap_or(u64::MAX);
        self.maps
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |maps| {
                maps.checked_add(bytes).filter(|&total| total <= budget)
            })
            .ok()?;
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        Some(HeapReservation {
            budget: self,
            bytes: AtomicU64::new(bytes),
            map: true,
        })
    }

    /// Reserve `bytes` for a doc-id map kept whatever the budget says. It still counts in the maps
    /// total, so it leaves less room for maps a query can do without.
    pub(crate) fn reserve_map(&'static self, bytes: u64) -> HeapReservation {
        self.maps.fetch_add(bytes, Ordering::AcqRel);
        self.charged.fetch_add(bytes, Ordering::AcqRel);
        HeapReservation {
            budget: self,
            bytes: AtomicU64::new(bytes),
            map: true,
        }
    }

    /// Drop the oldest unheld files counted here until they are back under their target (see
    /// [`unheld_target`]), then each cache's unheld lazy files past its cap. When held files alone
    /// fill the budget, the tables' cached sessions are dropped first so theirs can leave too. Does
    /// nothing under budget, with no budget, while the process has room to spare, or while another
    /// pass runs: then the cache keeps what it holds, as before.
    pub(crate) fn release_over_budget(&self) {
        let Some(budget) = self.budget() else { return };
        let charged = self.charged.load(Ordering::Acquire);
        if charged <= budget || !self.short_of_memory() {
            return;
        }
        // One pass at a time. A statement that ends while one runs leaves its files to the next
        // statement end, so with no later statement they stay until the idle sweep.
        let _pass = match self.pass.try_lock() {
            Ok(pass) => pass,
            Err(TryLockError::Poisoned(pass)) => pass.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        let released = self.drop_unheld_over_target(budget);
        free_elsewhere(released);
        debug!(
            target: "infino::cache",
            budget,
            before = charged,
            after = self.charged.load(Ordering::Acquire),
            "decoded metadata over budget: dropped the oldest unheld files"
        );
    }

    /// One pass of [`Self::release_over_budget`]: the readers it let go, for the caller to free.
    fn drop_unheld_over_target(&self, budget: u64) -> Vec<Arc<SuperfileReader>> {
        let caches: Vec<Arc<DiskCacheStore>> = {
            let mut list = self.caches.lock().unwrap_or_else(PoisonError::into_inner);
            list.retain(|cache| cache.strong_count() > 0);
            list.iter().filter_map(Weak::upgrade).collect()
        };
        let mut scan = self.scan(&caches);
        // Held files fill the budget: drop the tables' cached sessions, rebuilt on their next
        // statement, and look again.
        if scan.held > budget {
            drop(scan);
            self.drop_cached_sessions();
            scan = self.scan(&caches);
        }
        let target = unheld_target(budget, scan.held);
        let mut unheld = scan.unheld;
        let mut released = Vec::new();
        for (_, at, uri, bytes) in scan.oldest {
            if unheld <= target {
                break;
            }
            if let Some(reader) = caches[at].release_lazy_entry(&uri, caches[at].now_us()) {
                released.push(reader);
                unheld -= bytes;
            }
        }
        // The lazy files past the cap would otherwise go at the next query's first open, which
        // would pay for freeing them.
        for cache in &caches {
            released.extend(cache.release_unheld_lazy_entries_over_cap());
        }
        released
    }

    /// The unheld files counted here, oldest first, with the bytes held and unheld.
    fn scan(&self, caches: &[Arc<DiskCacheStore>]) -> Scan {
        // Each cache stamps accesses on its own clock, started when it was, so files are ranked by
        // how long each has been idle, not by their stamps.
        let mut oldest = Vec::new();
        let mut unheld = 0;
        for (at, cache) in caches.iter().enumerate() {
            let now_us = cache.now_us();
            for entry in cache.cached.iter() {
                let Some(heap) = entry
                    .value()
                    .heap
                    .get()
                    .filter(|heap| heap.counts_against(self))
                else {
                    continue;
                };
                if cache.reclaimable_lazy_entry(entry.key(), entry.value()) {
                    let last = entry.value().last_access_us.load(Ordering::Acquire);
                    let idle = Reverse(now_us.saturating_sub(last));
                    oldest.push((idle, at, *entry.key(), heap.bytes()));
                    unheld += heap.bytes();
                }
            }
        }
        oldest.sort_unstable();
        let held = self.charged.load(Ordering::Acquire).saturating_sub(unheld);
        Scan {
            oldest,
            unheld,
            held,
        }
    }

    /// Let every registered table drop its cached sessions.
    fn drop_cached_sessions(&self) {
        let sessions: Vec<Arc<dyn CachedSessions>> = {
            let mut list = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
            list.retain(|sessions| sessions.strong_count() > 0);
            list.iter().filter_map(Weak::upgrade).collect()
        };
        for table in sessions {
            table.drop_cached_sessions();
        }
    }

    /// Let this budget drop `sessions`' cached sessions when held files fill it.
    pub(crate) fn register_sessions(&self, sessions: Weak<dyn CachedSessions>) {
        let mut list = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        list.retain(|sessions| sessions.strong_count() > 0);
        list.push(sessions);
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
        mem,
        sync::{
            Arc, Barrier, Mutex,
            atomic::{AtomicU64, Ordering},
            mpsc::{Receiver, Sender, channel},
        },
        thread,
        time::Duration,
    };

    use tempfile::TempDir;

    use crate::{
        superfile::SuperfileReader,
        supertable::{
            manifest::SuperfileUri,
            reader_cache::disk::{
                DiskCacheStore, EntryAccounting, Residency,
                heap::{
                    CachedSessions, FREE_THREAD_NAME, FreeQueue, HeapBudget, HeapReservation,
                    UNHELD_FLOOR_DIVISOR, leaves_too_little_room,
                },
                test_support::*,
            },
        },
    };

    /// Bytes each test entry counts against the budget.
    const ENTRY_BYTES: u64 = 100;
    /// Map keeps issued at once in the burst test.
    const BURST_KEEPS: u64 = 200;
    /// Map keeps the burst test's budget has room for.
    const BURST_ROOM: u64 = 80;

    /// How long a test waits for a pass that should return at once before it fails.
    const PASS_DEADLINE: Duration = Duration::from_secs(10);

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
        let residency = Residency::Paged {
            block_source: dummy_block_source(store, uri),
        };
        let size_bytes = Arc::new(AtomicU64::new(0));
        let accounting = EntryAccounting::SourceOwned;
        let held = Arc::clone(&reader);
        cache_entry(
            store,
            uri,
            held,
            residency,
            size_bytes,
            accounting,
            last_access_us,
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
    fn store_with_budget(budget: Option<u64>) -> (TempDir, Arc<DiskCacheStore>) {
        let (dir, store) = test_store();
        store.use_heap_budget(HeapBudget::with_budget(budget));
        (dir, store)
    }

    #[test]
    fn a_burst_of_map_keeps_gets_exactly_the_room_there_is() {
        // A budget with room for 80 maps of 100 bytes, not short of memory, and 200 threads asking
        // for a map at the same moment.
        //  - exactly 80 get one; the maps total and the overall total are 80 maps' worth;
        //  - dropping them gives both totals back.
        // This test pins that a fan-out of keeps can't pass the budget between its checks.
        let budget = HeapBudget::with_budget(Some(BURST_ROOM * ENTRY_BYTES));
        budget.set_short_of_memory(false);
        let start = Arc::new(Barrier::new(BURST_KEEPS as usize));
        let keeps: Vec<_> = (0..BURST_KEEPS)
            .map(|_| {
                let start = Arc::clone(&start);
                thread::spawn(move || {
                    start.wait();
                    budget.try_reserve_map(ENTRY_BYTES)
                })
            })
            .collect();
        let kept: Vec<HeapReservation> = keeps
            .into_iter()
            .filter_map(|keep| keep.join().expect("keep thread"))
            .collect();
        assert_eq!(kept.len() as u64, BURST_ROOM);
        assert_eq!(
            budget.maps.load(Ordering::Acquire),
            BURST_ROOM * ENTRY_BYTES
        );
        assert_eq!(budget.charged(), BURST_ROOM * ENTRY_BYTES);
        drop(kept);
        assert_eq!(budget.maps.load(Ordering::Acquire), 0);
        assert_eq!(budget.charged(), 0);
    }

    #[test]
    fn short_of_memory_refuses_a_map_with_room_to_spare() {
        // A budget with no maps kept, first with memory to spare, then short of it.
        //  - with memory to spare a map is kept;
        //  - short of memory the next is refused, though the maps total is far under the budget;
        //  - a map kept regardless still counts, refused or not.
        // This test pins that the process's memory, not only the maps total, decides a keep.
        let budget = HeapBudget::with_budget(Some(BURST_ROOM * ENTRY_BYTES));
        budget.set_short_of_memory(false);
        let first = budget.try_reserve_map(ENTRY_BYTES).expect("room to spare");
        budget.set_short_of_memory(true);
        assert!(budget.try_reserve_map(ENTRY_BYTES).is_none());
        let regardless = budget.reserve_map(ENTRY_BYTES);
        assert_eq!(budget.maps.load(Ordering::Acquire), 2 * ENTRY_BYTES);
        drop((first, regardless));
        assert_eq!(budget.maps.load(Ordering::Acquire), 0);
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
        // This test pins that no query waits for freeing what the cache let go. Its own queue, so
        // other tests' batches cannot fill it and make the caller free this one.
        static QUEUE: FreeQueue = FreeQueue::new();
        let (go, wait) = channel();
        let (ran_on, report) = channel();
        QUEUE.free(vec![Freed { go: wait, ran_on }]);
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

    /// Budget, in entries, for the cases below: a small stand-in for a large table's budget.
    const CASE_BUDGET_ENTRIES: u64 = 8;

    /// A cache of its own counting against `budget_entries` entries' bytes, short of memory.
    fn store_of(budget_entries: u64) -> (TempDir, Arc<DiskCacheStore>) {
        store_with_budget(Some(budget_entries * ENTRY_BYTES))
    }

    /// `n` counted files in `store`, the first last used earliest.
    fn entries(store: &Arc<DiskCacheStore>, n: u64, first_stamp: u64) -> Vec<SuperfileUri> {
        (0..n)
            .map(|at| charged_entry(store, first_stamp + at))
            .collect()
    }

    /// Readers held for `uris`, as a running query holds them.
    fn hold(store: &DiskCacheStore, uris: &[SuperfileUri]) -> Vec<Arc<SuperfileReader>> {
        uris.iter().map(|uri| cached_reader(store, uri)).collect()
    }

    /// How many of `uris` are still cached.
    fn cached(store: &DiskCacheStore, uris: &[SuperfileUri]) -> usize {
        uris.iter().filter(|uri| store.is_cached(uri)).count()
    }

    #[test]
    fn an_unreadable_memory_reading_counts_as_room_to_spare() {
        // Readings of none, above and below the limit minus the budget.
        // This test pins that a reading that cannot be taken counts as room to spare, as the SQL
        // memory limit is off then too, and that the others count as short and not short.
        assert!(
            !leaves_too_little_room(None, 100, 25),
            "unreadable: room to spare"
        );
        assert!(
            leaves_too_little_room(Some(80), 100, 25),
            "above limit minus budget"
        );
        assert!(!leaves_too_little_room(Some(70), 100, 25), "below it");
    }

    #[test]
    fn with_nothing_held_the_pass_drops_the_oldest_down_to_the_budget() {
        // A big scan has ended: 30 unheld files over a budget of 8, none held.
        //  - what stays counted is the budget, the files used last.
        // This test pins that with nothing held the target is the whole budget.
        let (_dir, store) = store_of(CASE_BUDGET_ENTRIES);
        let uris = entries(&store, 30, 1);
        store.heap_budget().release_over_budget();
        assert_eq!(
            store.heap_budget().charged(),
            CASE_BUDGET_ENTRIES * ENTRY_BYTES
        );
        assert_eq!(cached(&store, &uris[22..]), 8, "the 8 used last stay");
    }

    #[test]
    fn with_held_files_over_the_budget_the_files_just_used_stay() {
        // A long scan holding 3x the budget while small statements end: 24 held and 6 unheld over a
        // budget of 8.
        //  - the unheld files kept are a quarter of the budget, the ones used last;
        //  - the held ones all stay.
        // This test pins the floor: neither every unheld file leaves, nor none does.
        let (_dir, store) = store_of(CASE_BUDGET_ENTRIES);
        let held_uris = entries(&store, 24, 1);
        let _scan = hold(&store, &held_uris);
        let unheld = entries(&store, 6, 100);
        store.heap_budget().release_over_budget();
        let floor = (CASE_BUDGET_ENTRIES / UNHELD_FLOOR_DIVISOR) as usize;
        assert_eq!(
            cached(&store, &unheld),
            floor,
            "a quarter of the budget stays"
        );
        assert!(
            unheld[6 - floor..].iter().all(|uri| store.is_cached(uri)),
            "the newest stay"
        );
        assert_eq!(cached(&store, &held_uris), 24, "held files stay");
    }

    /// A table's cached session holding readers, as a cached SQL context holds its files.
    struct HeldSession(Mutex<Vec<Arc<SuperfileReader>>>);

    impl CachedSessions for HeldSession {
        fn drop_cached_sessions(&self) {
            let readers = mem::take(&mut *self.0.lock().expect("session"));
            drop(readers);
        }
    }

    #[test]
    fn a_cached_session_holding_the_budget_is_dropped_and_its_files_go() {
        // A table's cached context holding 2x the budget on a table with no new commits: 16 files
        // held only by a registered session, over a budget of 8.
        //  - the pass drops the session;
        //  - then its files are unheld and go down to the budget, the ones used last staying.
        // This test pins that memory held between statements by a cached session can come back.
        let (_dir, store) = store_of(CASE_BUDGET_ENTRIES);
        let uris = entries(&store, 16, 1);
        let session: Arc<dyn CachedSessions> =
            Arc::new(HeldSession(Mutex::new(hold(&store, &uris))));
        store
            .heap_budget()
            .register_sessions(Arc::downgrade(&session));
        store.heap_budget().release_over_budget();
        assert_eq!(cached(&store, &uris), 8, "down to the budget");
        assert_eq!(cached(&store, &uris[8..]), 8, "the newest stay");
    }

    #[test]
    fn files_of_many_small_caches_go_oldest_first_down_to_the_budget() {
        // Many small tables each under budget: 4 caches of 3 files over a budget of 8, none held.
        //  - what stays counted is the budget, 8 of the 12 files.
        // This test pins that the pass works across every cache counted against one budget.
        let budget = HeapBudget::with_budget(Some(CASE_BUDGET_ENTRIES * ENTRY_BYTES));
        let mut stores = Vec::new();
        let mut uris = Vec::new();
        for _ in 0..4 {
            let (dir, store) = test_store();
            store.use_heap_budget(budget);
            uris.extend(entries(&store, 3, 1));
            stores.push((dir, store));
        }
        budget.release_over_budget();
        assert_eq!(budget.charged(), CASE_BUDGET_ENTRIES * ENTRY_BYTES);
        let left: usize = stores.iter().map(|(_, store)| cached(store, &uris)).sum();
        assert_eq!(left, 8);
    }

    #[test]
    fn a_pass_already_running_makes_another_return_at_once() {
        // Five unheld files over a budget of one, while a pass is running and then after it ends.
        //  - while it runs, another statement end drops nothing;
        //  - after it, a pass drops files down to the budget.
        // This test pins that two statements ending together do not both drop files for the same
        // excess. Its deadline bounds the second pass returning, which only try-locks, so load
        // cannot slow it.
        let (_dir, store) = store_of(1);
        let uris = entries(&store, 5, 1);
        let budget = store.heap_budget();
        let running = budget.pass.lock().expect("pass");
        let (done, returned) = channel();
        thread::spawn(move || {
            budget.release_over_budget();
            let _ = done.send(());
        });
        assert!(
            returned.recv_timeout(PASS_DEADLINE).is_ok(),
            "another pass returns at once, not after the running one"
        );
        assert_eq!(cached(&store, &uris), 5, "the running pass decides");
        drop(running);
        budget.release_over_budget();
        assert_eq!(cached(&store, &uris), 1);
    }

    #[test]
    fn a_pass_counts_only_the_files_counted_against_its_own_budget() {
        // A cache whose files count against one budget, also registered with a second budget that
        // is over its own total.
        // This test pins that a budget never drops files another budget counts.
        let (_dir, store) = store_of(100);
        let uris = entries(&store, 3, 1);
        let other = HeapBudget::with_budget(Some(0));
        other.register(&store);
        let _over = other.reserve(ENTRY_BYTES);
        other.release_over_budget();
        assert_eq!(cached(&store, &uris), 3, "not this budget's files");
    }

    #[test]
    fn a_pass_frees_the_files_past_the_cap_with_its_own() {
        // Three unheld files under budget and a lazy cap of one.
        // This test pins that the cap's releases join the pass's one batch, so a busy freeing
        // thread sees one batch per pass, not one per cache.
        let (_dir, store) = store_of(100);
        let _uris = entries(&store, 3, 1);
        store.set_lazy_entry_cap(1);
        let released = store
            .heap_budget()
            .drop_unheld_over_target(100 * ENTRY_BYTES);
        assert_eq!(
            released.len(),
            2,
            "the two past the cap come back to the pass"
        );
    }
}
