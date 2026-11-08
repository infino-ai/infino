// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-connection memory budget.
//!
//! Infino keeps data on object storage. To answer a query it pulls the
//! superfiles it needs onto local disk and memory-maps them, so most of the
//! memory a query touches is those mapped files — and the OS owns that: it
//! drops mapped pages when it needs the RAM back. What the OS *can't* reclaim
//! is the plain heap a query allocates on top — decode buffers, the vector
//! shortlist, SQL result batches, ingest buffers. Left unchecked, that heap is
//! what grows until the process is killed. This budget guards it.
//!
//! # What we track
//!
//! Only the allocations that grow with the data or the query, and a caller
//! reserves the bytes *before* it allocates them — so the budget can say no in
//! advance:
//!   - the vector shortlist fetched for a search,
//!   - the working state of the SQL operators that buffer — sorts, aggregates,
//!     joins, repartitions — which DataFusion reserves through the pool in
//!     `datafusion_pool`,
//!   - ingest buffers.
//!
//! # What we don't
//!
//! Everything already small or bounded by construction — a top-k heap holds k
//! rows, posting lists stream through, scratch is a fixed size. Reserving at
//! each of these would be noise, so we skip them and cover them with a small
//! headroom margin instead.
//!
//! That headroom is exactly why a bounded budget enforces against only **90%
//! of the value given in config** — the spare 10% absorbs all of these
//! untracked allocations. We trade exact accounting for far less plumbing, the
//! same bargain the query engines make.
//!
//! Nor the batches a SQL plan streams between its operators — a scan's decoded
//! pages, a filter's or projection's output, the rows an `unnest` repeats, the
//! result as it is collected. DataFusion reserves none of them, and a streaming
//! scan runs even at a zero gate. They are bounded per batch, not by the
//! budget, and a batch is only as small as its rows: one that repeats a long
//! text column many times over can be larger than the whole budget.
//!
//! # Process memory limit
//!
//! What bounds those batches is a process-wide limit on SQL (`resident`): a
//! running statement is refused once the process's anonymous resident memory
//! passes it. It reads the process, not the reservations, so it sees every
//! batch whichever operator allocated it. The limit is 90% of the process's
//! cgroup memory limit, or `memory.process_limit_bytes` in config; with
//! neither, there is none.
//!
//! # When the budget is full
//!
//! A reservation that would cross the limit first asks the holders of memory
//! that can be given back (see [`Reclaim`]) to give it back, and tries once
//! more. If it still doesn't fit, it fails with [`OverBudget`]. The caller
//! turns that into a normal query error — one query fails cleanly, instead of
//! the whole process running out of memory.
//!
//! A vector search's cold read (a download from object storage) waits instead,
//! for up to 30 seconds, while other cold reads on the connection hold memory
//! that will come back, and fails only if that doesn't make room (see
//! [`ConnectionMemoryBudget::admit_cold_read`]).
//!
//! # Two modes
//!
//! Fixed when the budget is built:
//!   - **measured** ([`ConnectionMemoryBudget::measured`], the default) — counts
//!     usage so we can see it, but never refuses. No change in behaviour.
//!   - **bounded** ([`ConnectionMemoryBudget::with_limit`]) — refuses once usage
//!     would cross the ceiling. The ceiling is 90% of the configured value,
//!     leaving the 10% headroom above.
//!
//! # How it fits together
//!
//! One budget is made per connection and shared by everything that connection
//! runs at once, so what's bounded is the running total of live reservations —
//! not any single allocation. Each reservation hands its bytes back when it is
//! dropped.
//!
//! ```text
//!
//!  query / ingest call sites
//!        │ try_reserve(n)               │ try_reserve(m)
//!        ▼                              ▼
//!   ┌──────────────┐             ┌──────────────┐
//!   │  Reservation │             │  Reservation │   each holds its bytes;
//!   │   size = n   │             │   size = m   │   Drop returns them
//!   └──────┬───────┘             └──────┬───────┘
//!          │ Arc                        │ Arc
//!          └──────────────┬─────────────┘
//!                         │
//!                         ▼
//!            ┌───────────────────────────────┐
//!            │     ConnectionMemoryBudget    │   one per connection, shared
//!            │      used:  AtomicUsize       │   across all its concurrent
//!            │     limit: Option<usize>      │   work
//!            └───────────────────────────────┘
//!
//!   bounded quantity  =  Σ live reservations  ≤  limit   (when a limit is set)
//! ```

// The call sites — the connection that owns the budget, and the query / ingest
// paths that reserve against it — arrive in the follow-up changes. Until then
// these items are exercised only by the tests below; this allow is removed as
// the call sites are wired in.
#![allow(dead_code)]

use std::{
    error::Error,
    fmt,
    pin::pin,
    sync::{
        Arc, Mutex, MutexGuard, PoisonError, Weak,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use tokio::{
    sync::Notify,
    time::{Instant, timeout_at},
};

mod datafusion_pool;
mod resident;

pub(crate) use datafusion_pool::budgeted_session_context;
pub(crate) use resident::{
    over_process_limit, process_limit, process_limit_exceeded, process_over_limit,
};

test_visible! {
    /// Replace this process's SQL memory limit — resolved from config or the
    /// cgroup on first use — with `limit` (`None`: no limit). Process-wide, so
    /// only a test binary of its own may call it.
    fn set_process_memory_limit(limit: Option<u64>) {
        resident::set_process_limit(limit)
    }
}

/// The fraction of a configured budget we actually enforce: gate at 9/10 and
/// leave the final 1/10 as headroom for allocations too small to track. Applied
/// once, in [`ConnectionMemoryBudget::with_limit`].
const ENFORCED_BUDGET_NUMERATOR: u128 = 9;
const ENFORCED_BUDGET_DENOMINATOR: u128 = 10;

/// Longest a cold read waits for room before it fails with [`OverBudget`].
/// Far longer than a cold read takes to finish, so a read still waiting this
/// long is being passed over by others, not waiting its turn.
const COLD_READ_MAX_WAIT: Duration = Duration::from_secs(30);

/// A connection's live memory budget: an atomic byte counter against an
/// optional ceiling. Cloned via `Arc` to every place the connection allocates,
/// so the bounded quantity is the sum of all live [`Reservation`]s.
///
/// One concrete type covers both modes — measured vs bounded is the optional
/// limit, not a different policy — so there is no trait. The one trait we do
/// implement is the query engine's memory-pool interface, in a thin adapter
/// that forwards to an `Arc<ConnectionMemoryBudget>`.
///
/// Must be `pub`: the `pub` vector-search functions take this type as an
/// argument. It is still not public API, because `lib.rs` makes the whole `memory`
/// module `pub` only under `test-helpers``, so a shipped build keeps this type crate-internal.
#[derive(Debug)]
pub struct ConnectionMemoryBudget {
    // Enforced ceiling in bytes, already reduced to the headroom gate.
    // `None` is measure-only: count usage, never refuse.
    limit: Option<usize>,
    // Bytes reserved across all live reservations. Every access is `Relaxed`: this is a pure
    // accounting counter, it guards no other memory, so no  stronger ordering is needed.
    used: AtomicUsize,
    // The largest memory `used` ever reached. Observability only, never gates. Exact
    // even under concurrency: each grow records its own atomic post-add value,
    // and `fetch_max` can't lose a maximum.
    peak_used: AtomicUsize,
    // Count of refused reservations (a count, not bytes); observability only,
    // never affects gating.
    denials: AtomicU64,
    // Holders that give memory back when a reservation doesn't fit.
    reclaimers: Mutex<Vec<Weak<dyn Reclaim>>>,
    // Bytes cold reads hold. Changed under the lock together with the
    // reservations it counts, so a refused read sees exactly the memory cold
    // reads can still give back.
    cold_read_bytes: Mutex<usize>,
    // Cold reads waiting for room, so a release wakes them only when there
    // are any: most releases pay one atomic load.
    cold_read_waiters: AtomicUsize,
    // Sent on every release while a cold read is waiting, to all of them.
    released: Notify,
}

/// Memory a holder can give back when the connection needs it for something
/// else: bytes it keeps only to save work later, like read-ahead's downloads,
/// which the scan can read again.
pub(crate) trait Reclaim: Send + Sync {
    /// Give back everything held, and hold nothing more.
    fn reclaim(&self);
}

impl ConnectionMemoryBudget {
    /// A measure-only budget: counts usage but never refuses. The default when
    /// no limit is configured.
    pub fn measured() -> Arc<Self> {
        Arc::new(Self {
            limit: None,
            used: AtomicUsize::new(0),
            peak_used: AtomicUsize::new(0),
            denials: AtomicU64::new(0),
            reclaimers: Mutex::new(Vec::new()),
            cold_read_bytes: Mutex::new(0),
            cold_read_waiters: AtomicUsize::new(0),
            released: Notify::new(),
        })
    }

    /// A bounded budget. `configured_bytes` is the operator-facing value; the
    /// enforced ceiling is 90% of it, leaving headroom for small untracked
    /// allocations.
    ///
    /// Expects `configured_bytes > 0`; use
    /// [`from_budget_bytes`](Self::from_budget_bytes) when `0` should mean
    /// measure-only. A value below ~10 bytes rounds down to a `0` ceiling that
    /// refuses everything, but that never happens at real (MB/GB) budgets.
    pub fn with_limit(configured_bytes: u64) -> Arc<Self> {
        debug_assert!(
            configured_bytes > 0,
            "with_limit expects a positive budget; 0 / unset means measured() at the call site"
        );

        let limit = (configured_bytes as u128 * ENFORCED_BUDGET_NUMERATOR
            / ENFORCED_BUDGET_DENOMINATOR) as usize;

        Arc::new(Self {
            limit: Some(limit),
            used: AtomicUsize::new(0),
            peak_used: AtomicUsize::new(0),
            denials: AtomicU64::new(0),
            reclaimers: Mutex::new(Vec::new()),
            cold_read_bytes: Mutex::new(0),
            cold_read_waiters: AtomicUsize::new(0),
            released: Notify::new(),
        })
    }

    /// Map a configured byte value to a budget: `0` is measure-only
    /// ([`measured`](Self::measured)), anything positive is bounded
    /// ([`with_limit`](Self::with_limit)). Both config sources (`ConnectOptions`
    /// and `config.yaml`) route through here, so "0 means measure-only" is
    /// defined in exactly one place.
    pub fn from_budget_bytes(bytes: u64) -> Arc<Self> {
        if bytes > 0 {
            Self::with_limit(bytes)
        } else {
            Self::measured()
        }
    }

    /// Reserve `n` bytes, returning a guard that frees them on drop. Fails with
    /// [`OverBudget`] if the reservation would cross the ceiling; a measured
    /// budget always succeeds.
    pub(crate) fn try_reserve(self: &Arc<Self>, n: usize) -> Result<Reservation, OverBudget> {
        self.try_grow(n)?;

        Ok(Reservation {
            budget: Arc::clone(self),
            size: n,
        })
    }

    /// Charge `n` bytes to the counter. If that would cross the ceiling, ask
    /// every [`Reclaim`] holder to give its memory back and try once more; on
    /// refusal the counter is left unchanged. Backs [`Self::try_reserve`],
    /// [`Reservation::try_grow`], and the DataFusion pool adapter's `try_grow`.
    pub(crate) fn try_grow(&self, n: usize) -> Result<(), OverBudget> {
        self.grow_within_limit(n)
            .or_else(|_| {
                self.reclaim();
                self.grow_within_limit(n)
            })
            .inspect_err(|_| {
                self.denials.fetch_add(1, Ordering::Relaxed);
            })
    }

    /// [`Self::try_reserve`] for a [`Reclaim`] holder: it never asks the
    /// others to give memory back, so one holder can't take another's.
    pub(crate) fn try_reserve_without_reclaim(
        self: &Arc<Self>,
        n: usize,
    ) -> Result<Reservation, OverBudget> {
        self.reserve_within_limit(n).inspect_err(|_| {
            self.denials.fetch_add(1, Ordering::Relaxed);
        })
    }

    /// Reserve `n` bytes if they fit under the ceiling as things stand: no
    /// reclaim, and a refusal is not counted as a denial.
    fn reserve_within_limit(self: &Arc<Self>, n: usize) -> Result<Reservation, OverBudget> {
        self.grow_within_limit(n)?;
        Ok(Reservation {
            budget: Arc::clone(self),
            size: n,
        })
    }

    /// Reserve `n` bytes for a cold read (a download from object storage),
    /// waiting for room instead of failing while other cold reads on this
    /// connection hold memory.
    ///
    /// A query reads many cells from many files at once, and every query on
    /// the connection shares this budget, so a read can be refused while
    /// other reads still hold memory that will come back in milliseconds.
    /// Then it waits for one of them to let go and tries again: reads that
    /// don't fit at once run in turn instead of failing their query.
    ///
    /// ```text
    ///   query A ── cell, cell ─┐
    ///   query B ── cell, cell ─┼─► fits? ─► reserve
    ///   query A ── cell ───────┘     │
    ///                                └─ no: wait for a cold read to let go,
    ///                                   then try again (every waiting read)
    /// ```
    ///
    /// A read waits only while a cold read holds memory, which is sure to come
    /// back, and only while its coming back could make room. Otherwise it fails with [`OverBudget`]: `n` is more than the
    /// ceiling minus the memory held outside cold reads, which no cold read
    /// gives back. It also fails once it has waited [`COLD_READ_MAX_WAIT`]: a
    /// read that keeps losing the room to others gives up rather than waiting
    /// for ever. A refused read asks [`Reclaim`] holders to give memory back,
    /// outside the lock, when it is first refused and once more before it gives
    /// up, and counts one denial.
    ///
    /// No caller waits while it holds a [`ColdReadPermit`]; one that did would
    /// at worst wait out the cap. Any release on the connection wakes the
    /// waiting reads, not only a cold read's.
    pub(crate) async fn admit_cold_read(
        self: &Arc<Self>,
        n: usize,
    ) -> Result<ColdReadPermit, OverBudget> {
        self.admit_cold_read_within(n, COLD_READ_MAX_WAIT).await
    }

    /// [`Self::admit_cold_read`] with the wait capped at `max_wait`.
    async fn admit_cold_read_within(
        self: &Arc<Self>,
        n: usize,
        max_wait: Duration,
    ) -> Result<ColdReadPermit, OverBudget> {
        let deadline = Instant::now() + max_wait;
        // Counted for the whole call, before the first try, so a release
        // between a refusal and the wait still wakes this read.
        let _waiting = ColdReadWaiting::new(&self.cold_read_waiters);
        let mut refused = false;
        loop {
            // Listen before trying, for the same reason.
            let mut released = pin!(self.released.notified());
            released.as_mut().enable();
            let wait = {
                let mut held = self.lock_cold_read_bytes();
                if let Ok(reservation) = self.reserve_within_limit(n) {
                    return Ok(self.hold(&mut held, reservation));
                }
                refused && self.may_wait(*held, n)
            };
            if wait && timeout_at(deadline, released).await.is_ok() {
                continue;
            }
            // First refusal, waiting can't make room, or the wait ran out:
            // ask the reclaim holders to give memory back (outside the lock)
            // and try again.
            self.reclaim();
            let mut held = self.lock_cold_read_bytes();
            let over = match self.reserve_within_limit(n) {
                Ok(reservation) => return Ok(self.hold(&mut held, reservation)),
                Err(over) => over,
            };
            if !refused {
                refused = true;
                self.denials.fetch_add(1, Ordering::Relaxed);
                if self.may_wait(*held, n) {
                    continue;
                }
            }
            return Err(over);
        }
    }

    /// Whether a refused read of `n` may wait: a cold read holds memory that
    /// will come back, and its coming back could make room.
    fn may_wait(&self, held: usize, n: usize) -> bool {
        held > 0
            && self.limit.is_none_or(|limit| {
                let outside = self.used().saturating_sub(held);
                n <= limit.saturating_sub(outside)
            })
    }

    /// Count `reservation` as a held cold read.
    fn hold(&self, held: &mut usize, reservation: Reservation) -> ColdReadPermit {
        *held += reservation.size;
        ColdReadPermit {
            reservation: Some(reservation),
        }
    }

    fn lock_cold_read_bytes(&self) -> MutexGuard<'_, usize> {
        self.cold_read_bytes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Ask `holder` to give its memory back whenever a reservation doesn't
    /// fit, for as long as it lives.
    pub(crate) fn register_reclaim(&self, holder: Weak<dyn Reclaim>) {
        let mut reclaimers = self
            .reclaimers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        reclaimers.retain(|r| r.strong_count() > 0);
        reclaimers.push(holder);
    }

    /// Ask every live holder to give its memory back. Called outside the
    /// registry's lock, since giving memory back drops reservations.
    fn reclaim(&self) {
        let holders: Vec<Arc<dyn Reclaim>> = self
            .reclaimers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter_map(Weak::upgrade)
            .collect();
        for holder in holders {
            holder.reclaim();
        }
    }

    /// Charge `n` bytes if that stays under the ceiling; a measured budget
    /// always does. On refusal the counter is left unchanged.
    fn grow_within_limit(&self, n: usize) -> Result<(), OverBudget> {
        match self.limit {
            // Unbounded
            None => {
                let prev = self.used.fetch_add(n, Ordering::Relaxed);
                self.record_peak(prev.saturating_add(n));
                Ok(())
            }

            Some(limit) => self
                .used
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                    used.checked_add(n).filter(|next| *next <= limit)
                })
                .map(|prev| self.record_peak(prev.saturating_add(n)))
                .map_err(|used| OverBudget {
                    requested: n,
                    used,
                    limit,
                }),
        }
    }

    /// Charge `n` bytes unconditionally, for a caller that has already committed
    /// the allocation and cannot fail — the query engine's infallible `grow`.
    pub(crate) fn grow_unchecked(&self, n: usize) {
        let prev = self.used.fetch_add(n, Ordering::Relaxed);
        self.record_peak(prev.saturating_add(n));
    }

    /// Raise the recorded peak to `used_now` if it's a new maximum.
    fn record_peak(&self, used_now: usize) {
        self.peak_used.fetch_max(used_now, Ordering::Relaxed);
    }

    /// Return `n` bytes to the budget.
    ///
    /// Never underflows: a [`Reservation`] only releases the exact bytes it
    /// added (its `size` grows solely through a successful [`Self::try_grow`]),
    /// so `used` is always at least `n` when this runs.
    pub(crate) fn release(&self, n: usize) {
        self.used.fetch_sub(n, Ordering::Relaxed);
        // Room came back, whoever held it: let waiting cold reads try again.
        if self.cold_read_waiters.load(Ordering::Acquire) > 0 {
            self.released.notify_waiters();
        }
    }

    /// Reserve `n` bytes for something held only to save work later, read
    /// ahead of its use, or `None` if that takes the budget past `percent` of
    /// its limit. Checked before and after reserving, so holders racing for
    /// the last of the share can't all get through. Never asks the others to
    /// give memory back.
    pub(crate) fn try_reserve_within_share(
        self: &Arc<Self>,
        n: usize,
        percent: usize,
    ) -> Option<Reservation> {
        if !self.fits_share(n, percent) {
            return None;
        }
        let reservation = self.try_reserve_without_reclaim(n).ok()?;
        self.fits_share(0, percent).then_some(reservation)
    }

    /// [`Self::try_reserve_within_share`] for a download from object storage that runs on its
    /// own, such as a cache fill: held as a cold read, so a refused cold read waits for it to be
    /// written and given back instead of failing. Never waits itself.
    pub(crate) fn try_admit_within_share(
        self: &Arc<Self>,
        n: usize,
        percent: usize,
    ) -> Option<ColdReadPermit> {
        let mut held = self.lock_cold_read_bytes();
        let reservation = self.try_reserve_within_share(n, percent)?;
        Some(self.hold(&mut held, reservation))
    }

    /// Whether `n` more bytes keep the budget within `percent` of its limit, as things stand.
    /// Always true for a measure-only budget.
    pub(crate) fn fits_share(&self, n: usize, percent: usize) -> bool {
        self.limit.is_none_or(|limit| {
            self.used().saturating_add(n).saturating_mul(100) <= limit.saturating_mul(percent)
        })
    }

    /// Bytes currently reserved.
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    /// Bytes still free under the ceiling, or `None` for a measure-only budget.
    pub(crate) fn remaining(&self) -> Option<usize> {
        self.limit.map(|limit| limit.saturating_sub(self.used()))
    }

    test_visible! {
        /// The largest [`used`](Self::used) ever reached. Only grows; never reset.
        /// Test-visible so integration tests can assert a query actually reserved
        /// against the budget: a peak above 0 means the budget was wired and exercised.
        fn peak(&self) -> usize {
            self.peak_used.load(Ordering::Relaxed)
        }
    }

    test_visible! {
        /// Bytes currently reserved. Test-visible companion to
        /// [`peak`](Self::peak): where peak proves the budget was
        /// exercised, `used_bytes` proves reservations were RELEASED —
        /// the property that separates operation-scoped holders from
        /// caches that pin bytes for their whole lifetime.
        fn used_bytes(&self) -> usize {
            self.used()
        }
    }

    test_visible! {
        /// The enforced ceiling, or `None` when measured.
        fn limit(&self) -> Option<usize> {
            self.limit
        }
    }

    test_visible! {
        /// Reservations refused so far. Test-visible so integration tests can
        /// assert the gate fired on an over-budget query.
        fn denials(&self) -> u64 {
            self.denials.load(Ordering::Relaxed)
        }
    }
}

impl fmt::Display for ConnectionMemoryBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.limit {
            Some(limit) => write!(
                f,
                "connection-memory-budget(used: {} B, limit: {limit} B)",
                self.used()
            ),
            None => write!(
                f,
                "connection-memory-budget(used: {} B, measured)",
                self.used()
            ),
        }
    }
}

/// A guard for bytes held against a [`ConnectionMemoryBudget`]; the bytes are
/// returned to the budget when it is dropped.
#[derive(Debug)]
pub(crate) struct Reservation {
    budget: Arc<ConnectionMemoryBudget>,
    // memory in bytes, which is to be reserved
    size: usize,
}

impl Reservation {
    // Grow the reservation as a buffer accumulates. On failure the reservation
    // keeps its current size.
    pub(crate) fn try_grow(&mut self, extra: usize) -> Result<(), OverBudget> {
        self.budget.try_grow(extra)?;

        self.size += extra;

        Ok(())
    }

    /// Bytes currently held.
    pub(crate) fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.release(self.size);
    }
}

/// A reservation was refused because it would exceed the connection's budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OverBudget {
    // Bytes the caller asked for.
    pub requested: usize,
    // Bytes already reserved when the request was made.
    pub used: usize,
    // The enforced ceiling.
    pub limit: usize,
}

impl fmt::Display for OverBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "over connection memory budget: requested {} B with {} B in use of a {} B limit",
            self.requested, self.used, self.limit
        )
    }
}

impl Error for OverBudget {}

/// Counts one cold read waiting for room for as long as it lives, including
/// when the read's query is cancelled mid-wait.
struct ColdReadWaiting<'a>(&'a AtomicUsize);

impl<'a> ColdReadWaiting<'a> {
    fn new(waiters: &'a AtomicUsize) -> Self {
        waiters.fetch_add(1, Ordering::AcqRel);
        Self(waiters)
    }
}

impl Drop for ColdReadWaiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// The bytes reserved for one cold read; they come back when it is dropped,
/// and every read waiting for room tries again.
#[derive(Debug)]
pub(crate) struct ColdReadPermit {
    // Always `Some` until dropped; taken in `drop` to release it under the lock.
    reservation: Option<Reservation>,
}

impl Drop for ColdReadPermit {
    fn drop(&mut self) {
        let Some(reservation) = self.reservation.take() else {
            return;
        };
        let budget = Arc::clone(&reservation.budget);
        {
            let mut held = budget.lock_cold_read_bytes();
            *held -= reservation.size;
            // Released under the lock, so a refused read never sees the count
            // without the bytes.
            drop(reservation);
        }
        // Releasing the reservation woke every waiting read: reads differ in
        // size, so the one a single wake picks may still not fit while a
        // smaller one would.
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use futures::FutureExt;
    use tokio::time::timeout;

    use super::*;

    const KB: u64 = 1024;

    #[test]
    fn measured_tracks_but_never_denies() {
        let budget = ConnectionMemoryBudget::measured();
        assert_eq!(budget.limit(), None);

        let huge = usize::MAX / 2;
        let r1 = budget.try_reserve(huge).expect("measured never denies");
        let r2 = budget.try_reserve(huge).expect("measured never denies");
        assert_eq!(budget.used(), huge * 2);
        assert_eq!(budget.denials(), 0);

        drop((r1, r2));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn peak_holds_the_largest_used() {
        let budget = ConnectionMemoryBudget::measured();
        let r1 = budget.try_reserve(1000).expect("ok");
        let r2 = budget.try_reserve(500).expect("ok");
        assert_eq!(budget.used(), 1500);
        assert_eq!(budget.peak(), 1500);

        // `used` falls back to 0, but the peak stays at the largest value seen.
        drop((r1, r2));
        assert_eq!(budget.used(), 0);
        assert_eq!(budget.peak(), 1500);

        // A smaller later reservation doesn't lower the peak.
        let r3 = budget.try_reserve(200).expect("ok");
        assert_eq!(budget.peak(), 1500);
        drop(r3);
    }

    #[test]
    fn with_limit_bakes_in_the_headroom_gate() {
        // 1000 configured -> gate at 900 (9/10).
        let budget = ConnectionMemoryBudget::with_limit(1000);
        assert_eq!(budget.limit(), Some(900));
    }

    #[test]
    fn from_budget_bytes_maps_zero_to_measured_and_positive_to_bounded() {
        // The config / ConnectOptions convention: 0 means measure-only.
        assert_eq!(ConnectionMemoryBudget::from_budget_bytes(0).limit(), None);
        // Positive -> bounded, with the same 90% gate as with_limit.
        assert_eq!(
            ConnectionMemoryBudget::from_budget_bytes(1000).limit(),
            Some(900)
        );
    }

    #[test]
    fn bounded_allows_up_to_the_gate_and_denies_past_it() {
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate = 900
        let held = budget.try_reserve(900).expect("exactly at the gate fits");
        assert_eq!(budget.used(), 900);
        // A bounded reservation records the peak too.
        assert_eq!(budget.peak(), 900);

        let err = budget
            .try_reserve(1)
            .expect_err("one byte over the gate is denied");
        assert_eq!(
            err,
            OverBudget {
                requested: 1,
                used: 900,
                limit: 900
            }
        );
        assert_eq!(budget.denials(), 1);
        // A denied reservation must not have changed the counter or the peak.
        assert_eq!(budget.used(), 900);
        assert_eq!(budget.peak(), 900);

        drop(held);
        assert_eq!(budget.used(), 0);
        // Space is free again after the drop.
        budget.try_reserve(900).expect("budget freed on drop");
    }

    #[test]
    fn reservation_drop_frees_exactly_what_it_held() {
        let budget = ConnectionMemoryBudget::with_limit(10 * KB); // gate = 9216
        {
            let _a = budget.try_reserve(4000).expect("fits");
            let _b = budget.try_reserve(4000).expect("fits");
            assert_eq!(budget.used(), 8000);
        }
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn reservation_grows_in_place_and_denies_when_full() {
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate = 900
        let mut r = budget.try_reserve(500).expect("fits");
        r.try_grow(400).expect("500 + 400 = 900 fits");
        assert_eq!(r.size(), 900);
        assert_eq!(budget.used(), 900);

        r.try_grow(1).expect_err("over the gate");
        // Failed grow leaves the reservation untouched.
        assert_eq!(r.size(), 900);
        assert_eq!(budget.used(), 900);
    }

    #[test]
    fn grow_unchecked_ignores_the_gate() {
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate = 900
        budget.grow_unchecked(5000); // well over the gate, but unconditional
        assert_eq!(budget.used(), 5000);
        assert_eq!(budget.peak(), 5000); // grow_unchecked records the peak too
        assert_eq!(budget.denials(), 0); // not a refusal path
        budget.release(5000);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn display_shows_mode_and_usage() {
        let bounded = ConnectionMemoryBudget::with_limit(1000); // gate = 900
        let _held = bounded.try_reserve(100).expect("fits");
        assert_eq!(
            format!("{bounded}"),
            "connection-memory-budget(used: 100 B, limit: 900 B)"
        );

        let measured = ConnectionMemoryBudget::measured();
        assert_eq!(
            format!("{measured}"),
            "connection-memory-budget(used: 0 B, measured)"
        );
    }

    /// The point of the atomics: many threads racing on a bounded budget must
    /// never collectively exceed the ceiling. Gate is 900 and each thread holds
    /// 100, so exactly nine win and the rest are refused — deterministically,
    /// because each winner's 100 is committed atomically before the next checks.
    #[test]
    fn concurrent_reservations_never_overcommit() {
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate = 900
        let threads = 32;
        let chunk = 100;

        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let budget = Arc::clone(&budget);
                thread::spawn(move || budget.try_reserve(chunk).ok())
            })
            .collect();

        // Hold every winning reservation until all threads finish, so peak
        // `used` is the sum of all successes at once.
        let held: Vec<_> = handles
            .into_iter()
            .filter_map(|h| h.join().expect("thread panicked"))
            .collect();

        assert_eq!(held.len(), 9, "exactly nine 100-byte holds fit under 900");
        assert_eq!(budget.used(), 900);
        assert_eq!(budget.denials() as usize, threads - 9);

        drop(held);
        assert_eq!(budget.used(), 0);
    }

    /// Reserve-and-release churn from many threads must balance back to zero —
    /// no leaked bytes, no `release` underflow.
    #[test]
    fn concurrent_churn_balances_back_to_zero() {
        let budget = ConnectionMemoryBudget::with_limit(10_000); // gate = 9000
        let handles: Vec<_> = (0..16)
            .map(|_| {
                let budget = Arc::clone(&budget);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        // Reserve then immediately drop -> release.
                        drop(budget.try_reserve(500));
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("thread panicked");
        }
        assert_eq!(
            budget.used(),
            0,
            "every reservation released; nothing leaked"
        );
    }

    // -- Giving memory back --------------------------------------------------

    /// The configured budget; the gate is 9/10 of it.
    const RECLAIM_BUDGET: u64 = 1000;
    /// What the gate lets through.
    const RECLAIM_GATE: usize = 900;
    /// What the holder keeps.
    const HELD: usize = 600;

    /// A holder of memory it gives back on request, counting the requests.
    pub(super) struct Holder {
        reservation: Mutex<Option<Reservation>>,
        asked: AtomicUsize,
    }

    impl Holder {
        /// A holder keeping `bytes` of `budget`, registered with it.
        pub(super) fn holding(budget: &Arc<ConnectionMemoryBudget>, bytes: usize) -> Arc<Self> {
            let holder = Arc::new(Self {
                reservation: Mutex::new(Some(budget.try_reserve(bytes).expect("fits"))),
                asked: AtomicUsize::new(0),
            });
            let weak: Weak<Holder> = Arc::downgrade(&holder);
            budget.register_reclaim(weak);
            holder
        }

        pub(super) fn asked(&self) -> usize {
            self.asked.load(Ordering::Relaxed)
        }
    }

    impl Reclaim for Holder {
        fn reclaim(&self) {
            self.asked.fetch_add(1, Ordering::Relaxed);
            self.reservation
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
        }
    }

    #[test]
    fn a_reservation_that_does_not_fit_takes_back_what_a_holder_keeps() {
        // The holder keeps 600 of 900; a 500 reservation fits only once it
        // gives them back.
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let holder = Holder::holding(&budget, HELD);
        let wanted = RECLAIM_GATE - HELD + 1;
        let reservation = budget.try_reserve(wanted).expect("fits after giving back");
        assert_eq!(holder.asked(), 1);
        assert_eq!(budget.used(), wanted);
        assert_eq!(
            budget.denials(),
            0,
            "a reservation that fit is not a denial"
        );
        drop(reservation);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn a_reservation_that_fits_asks_nobody() {
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let holder = Holder::holding(&budget, HELD);
        let _fits = budget.try_reserve(RECLAIM_GATE - HELD).expect("fits");
        assert_eq!(holder.asked(), 0);
        assert_eq!(budget.used(), RECLAIM_GATE);
    }

    #[test]
    fn a_reservation_too_big_even_after_giving_back_is_refused_as_before() {
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let holder = Holder::holding(&budget, HELD);
        let err = budget
            .try_reserve(RECLAIM_GATE + 1)
            .expect_err("bigger than the whole gate");
        assert_eq!(
            err,
            OverBudget {
                requested: RECLAIM_GATE + 1,
                used: 0,
                limit: RECLAIM_GATE
            }
        );
        assert_eq!(holder.asked(), 1);
        assert_eq!(budget.denials(), 1, "counted once, not once per try");
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn growing_a_reservation_takes_back_too() {
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let holder = Holder::holding(&budget, HELD);
        let mut reservation = budget.try_reserve(0).expect("empty");
        reservation
            .try_grow(RECLAIM_GATE - HELD + 1)
            .expect("fits after giving back");
        assert_eq!(holder.asked(), 1);
    }

    #[test]
    fn every_holder_gives_back() {
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let a = Holder::holding(&budget, HELD / 2);
        let b = Holder::holding(&budget, HELD / 2);
        let _wanted = budget
            .try_reserve(RECLAIM_GATE)
            .expect("fits after both give back");
        assert_eq!((a.asked(), b.asked()), (1, 1));
    }

    #[test]
    fn a_holders_own_reservation_never_takes_back() {
        // One holder must not take another's memory.
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        let holder = Holder::holding(&budget, HELD);
        budget
            .try_reserve_without_reclaim(RECLAIM_GATE - HELD + 1)
            .expect_err("does not fit");
        assert_eq!(holder.asked(), 0);
        assert_eq!(budget.used(), HELD);
        assert_eq!(budget.denials(), 1);
    }

    #[test]
    fn a_measured_budget_never_asks() {
        let budget = ConnectionMemoryBudget::measured();
        let holder = Holder::holding(&budget, HELD);
        let _huge = budget
            .try_reserve(usize::MAX / 2)
            .expect("measured never denies");
        assert_eq!(holder.asked(), 0);
    }

    #[test]
    fn a_dropped_holder_is_skipped_and_forgotten() {
        let budget = ConnectionMemoryBudget::with_limit(RECLAIM_BUDGET);
        drop(Holder::holding(&budget, HELD));
        budget
            .try_reserve(RECLAIM_GATE + 1)
            .expect_err("nothing to give back");
        // The next holder to register clears it out.
        let _holder = Holder::holding(&budget, HELD);
        let reclaimers = budget
            .reclaimers
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(reclaimers.len(), 1);
    }

    /// Budget for the cold-read tests, and the share of it held outside any
    /// cold read, as the rest of a query (hit lookup, operators) holds it.
    const SHARED_BUDGET: u64 = 100 * KB;

    /// Share of the limit a download may use in the share tests: room for a quarter of the
    /// ceiling but not that plus a tenth, so one admission fits and a second does not.
    const DOWNLOAD_SHARE_PERCENT: usize = 30;

    /// A budget whose ceiling is half held outside any cold read, the
    /// ceiling, and that outside reservation.
    fn half_held_budget() -> (Arc<ConnectionMemoryBudget>, usize, Reservation) {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let outside = budget.try_reserve(ceiling / 2).expect("fits");
        (budget, ceiling, outside)
    }

    /// A refused cold read waits for a held one to let go and then gets its
    /// memory, instead of failing its query. The two reads can belong to
    /// different queries: the budget, and so the wait, is the connection's.
    #[tokio::test]
    async fn a_refused_cold_read_waits_for_a_held_one() {
        let (budget, ceiling, _outside) = half_held_budget();
        let first = budget.admit_cold_read(ceiling / 4).await.expect("fits");
        let mut second = Box::pin(budget.admit_cold_read(ceiling / 3));
        assert!(
            (&mut second).now_or_never().is_none(),
            "a refused read must wait while another cold read holds memory"
        );
        drop(first);
        second
            .now_or_never()
            .expect("the read retries once the first one lets go")
            .expect("and then fits");
    }

    /// A download admitted within its share holds memory as a cold read: a refused cold read waits
    /// for it to come back instead of failing, and past the share the download gets nothing.
    #[tokio::test]
    async fn a_refused_cold_read_waits_for_a_download_admitted_within_its_share() {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let download = budget
            .try_admit_within_share(ceiling / 4, DOWNLOAD_SHARE_PERCENT)
            .expect("within the share");
        assert!(
            budget
                .try_admit_within_share(ceiling / 10, DOWNLOAD_SHARE_PERCENT)
                .is_none(),
            "past the share, a download gets nothing"
        );
        let _outside = budget.try_reserve(ceiling / 2).expect("fits");
        let mut read = Box::pin(budget.admit_cold_read(ceiling / 3));
        assert!(
            (&mut read).now_or_never().is_none(),
            "a refused read waits while a download holds memory"
        );
        drop(download);
        read.now_or_never()
            .expect("the read retries once the download lets go")
            .expect("and then fits");
    }

    /// Every waiting read retries when memory comes back, not just one.
    #[tokio::test]
    async fn every_refused_cold_read_retries_when_one_lets_go() {
        let (budget, ceiling, _outside) = half_held_budget();
        let first = budget.admit_cold_read(ceiling / 3).await.expect("fits");
        let mut second = Box::pin(budget.admit_cold_read(ceiling / 5));
        let mut third = Box::pin(budget.admit_cold_read(ceiling / 5));
        assert!((&mut second).now_or_never().is_none());
        assert!((&mut third).now_or_never().is_none());
        drop(first);
        // Hold both permits: dropping the second would wake the third on
        // its own and hide a wake-up that reached only one of them.
        let _second = second
            .now_or_never()
            .expect("the second read retries")
            .expect("and fits");
        let _third = third
            .now_or_never()
            .expect("the third read retries too")
            .expect("and fits");
    }

    /// With no cold read holding memory, waiting frees nothing: the read fails
    /// at once instead of hanging. Memory held outside cold reads doesn't
    /// count, since nothing says when it comes back.
    #[tokio::test]
    async fn a_cold_read_fails_when_no_cold_read_holds_memory() {
        let (budget, ceiling, _outside) = half_held_budget();
        budget
            .admit_cold_read(ceiling)
            .now_or_never()
            .expect("must not wait on reads that do not exist")
            .expect_err("the budget refuses");
        let first = budget.admit_cold_read(ceiling / 4).await.expect("fits");
        drop(first);
        budget
            .admit_cold_read(ceiling)
            .now_or_never()
            .expect("a finished read no longer counts")
            .expect_err("the budget refuses");
    }

    /// A read larger than the whole ceiling can never fit, so it fails at once
    /// rather than waiting for the reads that are held.
    #[tokio::test]
    async fn a_cold_read_larger_than_the_budget_fails_at_once() {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let _held = budget.admit_cold_read(ceiling / 4).await.expect("fits");
        budget
            .admit_cold_read(ceiling + 1)
            .now_or_never()
            .expect("must not wait for reads that can't make room")
            .expect_err("the budget refuses");
    }

    /// There is no queue: a read that fits goes in even while a larger one
    /// waits, so one query's waiting read never holds up another's.
    #[tokio::test]
    async fn a_cold_read_that_fits_goes_in_while_another_waits() {
        let (budget, ceiling, _outside) = half_held_budget();
        let _first = budget.admit_cold_read(ceiling / 4).await.expect("fits");
        let mut large = Box::pin(budget.admit_cold_read(ceiling / 3));
        assert!((&mut large).now_or_never().is_none(), "refused, waits");
        budget
            .admit_cold_read(ceiling / 10)
            .now_or_never()
            .expect("a read that fits doesn't wait behind the large one")
            .expect("and fits");
    }

    /// Wait cap for the cap test: short, so the test waits it out quickly.
    const TEST_WAIT_CAP: Duration = Duration::from_millis(20);

    /// How long the cap test gives the read to give up, far past its cap.
    const TEST_GIVE_UP_WITHIN: Duration = Duration::from_secs(5);

    /// A waiting read gives up at its cap even when no cold read ever lets
    /// go (a stalled download, a query no longer polled), instead of waiting
    /// for ever.
    #[tokio::test]
    async fn a_waiting_cold_read_gives_up_at_its_cap() {
        let (budget, ceiling, _outside) = half_held_budget();
        let _held = budget.admit_cold_read(ceiling / 4).await.expect("fits");
        let waiting = budget.admit_cold_read_within(ceiling / 3, TEST_WAIT_CAP);
        timeout(TEST_GIVE_UP_WITHIN, waiting)
            .await
            .expect("gives up at its cap, with no release to wake it")
            .expect_err("the budget refuses");
    }

    /// Room freed outside cold reads (an operator, read-ahead, a cache) wakes
    /// a waiting read too, so it doesn't sleep until some cold read finishes.
    #[tokio::test]
    async fn memory_freed_outside_cold_reads_wakes_a_waiting_one() {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let outside = budget.try_reserve(ceiling * 6 / 10).expect("fits");
        let _held = budget.admit_cold_read(ceiling / 10).await.expect("fits");
        let mut waiting = Box::pin(budget.admit_cold_read(ceiling * 35 / 100));
        assert!((&mut waiting).now_or_never().is_none(), "refused, waits");
        drop(outside);
        waiting
            .now_or_never()
            .expect("the outside release wakes it")
            .expect("and it fits");
    }

    /// Memory held outside cold reads doesn't come back when they finish, so
    /// a read that only fits without it fails at once rather than waiting for
    /// every cold read on the connection.
    #[tokio::test]
    async fn a_cold_read_fails_at_once_when_outside_memory_leaves_no_room() {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let _outside = budget.try_reserve(ceiling * 7 / 10).expect("fits");
        let _held = budget.admit_cold_read(ceiling / 10).await.expect("fits");
        budget
            .admit_cold_read(ceiling * 4 / 10)
            .now_or_never()
            .expect("must not wait for room the outside memory keeps")
            .expect_err("the budget refuses");
    }

    /// Holds a reservation it gives back when asked, and counts the asks.
    #[derive(Default)]
    struct HoldingReclaim {
        held: Mutex<Option<Reservation>>,
        asked: AtomicUsize,
    }

    impl Reclaim for HoldingReclaim {
        fn reclaim(&self) {
            self.asked.fetch_add(1, Ordering::Relaxed);
            drop(
                self.held
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .take(),
            );
        }
    }

    /// Before a waiting read gives up, it asks reclaim holders once more: a
    /// holder that took memory back while it waited (read-ahead resuming)
    /// gives it up, and the read fits.
    #[tokio::test]
    async fn a_waiting_cold_read_reclaims_once_more_before_giving_up() {
        let budget = ConnectionMemoryBudget::with_limit(SHARED_BUDGET);
        let ceiling = budget.limit().expect("bounded");
        let holder = Arc::new(HoldingReclaim::default());
        let weak: Weak<dyn Reclaim> = Arc::downgrade(&holder) as Weak<dyn Reclaim>;
        budget.register_reclaim(weak);
        let first = budget.admit_cold_read(ceiling / 2).await.expect("fits");
        let mut waiting = Box::pin(budget.admit_cold_read(ceiling * 6 / 10));
        assert!((&mut waiting).now_or_never().is_none(), "refused, waits");
        // The holder takes memory again while the read waits.
        *holder.held.lock().unwrap_or_else(PoisonError::into_inner) =
            Some(budget.try_reserve(ceiling * 45 / 100).expect("fits"));
        drop(first);
        let _admitted = waiting
            .now_or_never()
            .expect("tries again once the cold read lets go")
            .expect("and fits after asking the holder once more");
        assert_eq!(holder.asked.load(Ordering::Relaxed), 2);
    }

    /// A refused read asks reclaim holders to give memory back, and counts as
    /// a denial, once: its retries after a wait do neither, so a queue of
    /// waiting reads doesn't strip the other holders on every release.
    #[tokio::test]
    async fn a_waiting_cold_read_reclaims_and_counts_once() {
        let (budget, ceiling, _outside) = half_held_budget();
        let holder = Arc::new(HoldingReclaim::default());
        let weak: Weak<dyn Reclaim> = Arc::downgrade(&holder) as Weak<dyn Reclaim>;
        budget.register_reclaim(weak);
        let first = budget.admit_cold_read(ceiling / 3).await.expect("fits");
        let second = budget.admit_cold_read(ceiling / 10).await.expect("fits");
        let mut waiting = Box::pin(budget.admit_cold_read(ceiling / 4));
        assert!((&mut waiting).now_or_never().is_none(), "refused, waits");
        // Wakes the waiting read, which still doesn't fit.
        drop(second);
        assert!((&mut waiting).now_or_never().is_none(), "refused again");
        drop(first);
        let _admitted = waiting
            .now_or_never()
            .expect("retries once the first lets go")
            .expect("and fits");
        assert_eq!(holder.asked.load(Ordering::Relaxed), 1, "reclaimed once");
        assert_eq!(budget.denials(), 1, "one refused read, one denial");
    }
}
