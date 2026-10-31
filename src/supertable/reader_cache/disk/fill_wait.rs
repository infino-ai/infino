// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Background fills parked until a query lets go of their file, checked together on one timer per
//! store.
//!
//! A fill must not download while a query still holds the file's lazy reader, and nothing signals
//! when a query drops its handle: the only test is the reader's strong count. Polling it from each
//! fill wakes one task per file every tick, so a cold scan over thousands of files floods the
//! runtime the queries run on. Instead a waiting fill parks here, and one ticker task per store
//! checks every parked fill and wakes only those that may go.
//!
//! ```text
//!   fill A ──park──┐                         ┌──> wake A (reader free twice in a row)
//!   fill B ──park──┼──> [waiters] <── tick ──┤
//!   fill C ──park──┘     one task, every     └──> wake C (defer window closed)
//!                        RETRY_INTERVAL           B stays parked: a query holds its reader
//! ```
//!
//! A parked fill holds only a weak reference to its store, so dropping the store drops the
//! waiters and wakes every fill parked on it.

use std::{
    mem,
    sync::{Arc, Mutex, MutexGuard, Weak, atomic::Ordering},
    time::{Duration, Instant},
};

use tokio::{sync::oneshot, time::sleep};

use crate::{
    runtime_bridge::shared_io_runtime,
    superfile::reader::SuperfileReader,
    supertable::reader_cache::disk::{DiskCacheStore, reader_blocks_background_fill},
};

/// How often the ticker re-checks the parked fills. It also sets the grace a reader gets between
/// an open and its query: a fill goes only after two free ticks in a row, so 10 to 20 ms after the
/// reader is let go.
const RETRY_INTERVAL: Duration = Duration::from_millis(10);

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
pub(super) enum PromotionDefer {
    /// Yield for as long as the reader stays busy.
    Forever,
    /// Yield until this instant, then fill regardless.
    Until(Instant),
}

impl PromotionDefer {
    /// Open a window of `timeout` starting now. A `timeout` that overflows
    /// the clock (notably [`Duration::MAX`]) yields indefinitely.
    pub(super) fn start(timeout: Duration) -> Self {
        match Instant::now().checked_add(timeout) {
            Some(deadline) => Self::Until(deadline),
            None => Self::Forever,
        }
    }

    /// Whether the window is still open at `now`.
    fn open_at(self, now: Instant) -> bool {
        match self {
            Self::Forever => true,
            Self::Until(deadline) => now < deadline,
        }
    }

    /// Whether the fill should still yield to `reader` being held by a
    /// caller other than the cache entry.
    pub(super) fn yields_to(self, reader: &Weak<SuperfileReader>) -> bool {
        self.open_at(Instant::now()) && reader_blocks_background_fill(reader)
    }
}

/// Which of a fill's two waits is parking.
///
/// They differ only for callers waiting for a file to be promoted (`wait_until_mmap_promoted` and
/// `wait_until_fills_settled`, test and bench helpers; production registers none). Those let a
/// [`FillWait::Release`] go early, so the helper does not wait out the defer window. A
/// [`FillWait::Quiescence`] ignores them: the download itself still pauses for a held reader, so
/// letting it go would only pause it again at once, a busy loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FillWait {
    /// The first wait, before the fill takes a download slot.
    Release,
    /// The wait before each download attempt, the first one included.
    Quiescence,
}

/// The parked fills of one store, and whether its ticker is running.
#[derive(Default)]
pub(super) struct FillWaiters {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    parked: Vec<Parked>,
    ticker_running: bool,
}

struct Parked {
    reader: Weak<SuperfileReader>,
    defer: PromotionDefer,
    wait: FillWait,
    /// The reader looked free on the previous tick. `strong_count == 1` also occurs briefly while
    /// a caller is acquiring the cache entry, so a fill goes only after two free ticks in a row.
    free_last_tick: bool,
    wake: oneshot::Sender<()>,
}

impl Parked {
    /// Whether the fill may go without waiting for the reader to be free: the reader is gone, the
    /// defer window has closed, or a test helper waits for promotion (see [`FillWait`]).
    fn released(&self, now: Instant, promotion_waiters: bool) -> bool {
        self.reader.strong_count() == 0
            || (self.wait == FillWait::Release && promotion_waiters)
            || !self.defer.open_at(now)
    }

    /// One tick's verdict: whether the fill may go now. Besides [`Self::released`], a fill goes
    /// once its reader has been free on this tick and the one before.
    fn may_go(&mut self, now: Instant, promotion_waiters: bool) -> bool {
        if self.released(now, promotion_waiters) {
            return true;
        }
        let free = !reader_blocks_background_fill(&self.reader);
        let go = free && self.free_last_tick;
        self.free_last_tick = free;
        go
    }
}

impl FillWaiters {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("fill waiters poisoned")
    }

    /// Park until the fill for `reader` may go: the reader is free (or gone), the defer window has
    /// closed, or, for [`FillWait::Release`], a test helper waits for promotion. Then hand back the
    /// store, or `None` if it was dropped meanwhile. Says nothing about which of these happened:
    /// the caller checks the reader and the store itself.
    pub(super) async fn park(
        store: &Weak<DiskCacheStore>,
        reader: &Weak<SuperfileReader>,
        defer: PromotionDefer,
        wait: FillWait,
    ) -> Option<Arc<DiskCacheStore>> {
        // Drop the upgraded store before awaiting, so a parked fill never keeps its store alive.
        let (woken, start_ticker) = {
            let strong = store.upgrade()?;
            let (wake, woken) = oneshot::channel();
            let parked = Parked {
                reader: reader.clone(),
                defer,
                wait,
                free_last_tick: false,
                wake,
            };
            if parked.released(Instant::now(), promotion_waiters(&strong)) {
                return Some(strong);
            }
            let mut state = strong.fill_waiters.lock();
            state.parked.push(parked);
            let start_ticker = !state.ticker_running;
            state.ticker_running = true;
            (woken, start_ticker)
        };
        if start_ticker {
            // On the shared runtime, not the parking fill's: that one may be a short-lived
            // runtime, and a ticker that died with it would leave every later fill parked for good.
            shared_io_runtime().spawn(run_ticker(store.clone()));
        }
        // Err: the store, and with it the sender, was dropped; the upgrade below says so.
        let _ = woken.await;
        store.upgrade()
    }

    /// Take out the fills that may go and return their wake senders, forgetting the ones nobody
    /// waits for any more, and whether any are still parked. When none are, the ticker stops.
    ///
    /// The list is scanned outside the lock, so a fill parking meanwhile never waits on a long
    /// scan; it lands in the list left behind and is merged back after.
    fn tick(&self, promotion_waiters: bool) -> (Vec<oneshot::Sender<()>>, bool) {
        let mut parked = mem::take(&mut self.lock().parked);
        let now = Instant::now();
        let ready = parked
            .extract_if(.., |p| {
                p.wake.is_closed() || p.may_go(now, promotion_waiters)
            })
            .map(|p| p.wake)
            .collect();
        (ready, self.put_back(parked))
    }

    /// Return the fills a tick kept to the list, after any that parked during the scan. Returns
    /// whether any are parked, which is also whether the ticker keeps running.
    fn put_back(&self, mut kept: Vec<Parked>) -> bool {
        let mut state = self.lock();
        // When every fill left, keep the small list of newcomers and free the big one.
        if !kept.is_empty() {
            kept.append(&mut state.parked);
            state.parked = kept;
        }
        state.ticker_running = !state.parked.is_empty();
        state.ticker_running
    }

    /// Fills in the list, for tests. Reads 0 while a tick is scanning the list, so a test waits
    /// for a count to show up rather than asserting it.
    #[cfg(test)]
    pub(super) fn parked(&self) -> usize {
        self.lock().parked.len()
    }
}

/// One store's ticker. Holds the store only while it scans, so it never keeps a dropped store
/// alive.
async fn run_ticker(store: Weak<DiskCacheStore>) {
    loop {
        sleep(RETRY_INTERVAL).await;
        let Some(strong) = store.upgrade() else {
            return;
        };
        let (ready, more) = strong.fill_waiters.tick(promotion_waiters(&strong));
        // Let go of the store before waking anyone: a woken fill checks whether it holds the
        // store's last reference, and must not see ours.
        drop(strong);
        for wake in ready {
            let _ = wake.send(());
        }
        if !more {
            return;
        }
    }
}

/// Whether a test helper waits for some fill of `store` to promote its file.
fn promotion_waiters(store: &DiskCacheStore) -> bool {
    store.n_promotion_waiters.load(Ordering::Acquire) > 0
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Weak,
        time::{Duration, Instant},
    };

    use tokio::sync::oneshot;

    use crate::supertable::reader_cache::disk::fill_wait::{
        FillWait, FillWaiters, Parked, PromotionDefer,
    };

    /// A parked fill on no reader, with its receiver, so the sender reads as open.
    fn parked() -> (Parked, oneshot::Receiver<()>) {
        let (wake, woken) = oneshot::channel();
        let fill = Parked {
            reader: Weak::new(),
            defer: PromotionDefer::Forever,
            wait: FillWait::Quiescence,
            free_last_tick: false,
            wake,
        };
        (fill, woken)
    }

    /// A fill that parks while a tick scans lands in the list left behind; putting the kept fills
    /// back must keep it too, or it would never be woken.
    #[test]
    fn a_fill_parked_during_a_scan_stays_parked() {
        let waiters = FillWaiters::default();
        let (kept, _kept_rx) = parked();
        let (newcomer, _newcomer_rx) = parked();
        waiters.lock().parked.push(newcomer);
        assert!(waiters.put_back(vec![kept]), "fills are still parked");
        assert_eq!(waiters.parked(), 2, "the kept fill and the newcomer");

        let (newcomer, _rx) = parked();
        waiters.lock().parked.clear();
        waiters.lock().parked.push(newcomer);
        assert!(waiters.put_back(Vec::new()), "the newcomer is still parked");
        assert_eq!(waiters.parked(), 1);

        waiters.lock().parked.clear();
        assert!(
            !waiters.put_back(Vec::new()),
            "nothing parked stops the ticker"
        );
    }

    /// The window is a deadline, and `Duration::MAX` is the "never stop
    /// yielding" sentinel rather than an overflow panic.
    #[test]
    fn promotion_defer_window_opens_and_closes() {
        let zero = PromotionDefer::start(Duration::ZERO);
        assert!(
            !zero.open_at(Instant::now()),
            "a zero window is closed on arrival"
        );
        assert!(
            PromotionDefer::start(Duration::from_secs(3600)).open_at(Instant::now()),
            "a long window is open"
        );
        assert!(
            matches!(
                PromotionDefer::start(Duration::MAX),
                PromotionDefer::Forever
            ),
            "Duration::MAX overflows the clock and means unbounded yielding"
        );
        assert!(PromotionDefer::Forever.open_at(Instant::now()));
    }
}
