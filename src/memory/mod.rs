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
//!   - SQL result / intermediate batches,
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
//! # When the budget is full
//!
//! A reservation that would cross the limit fails with [`OverBudget`]. The
//! caller turns that into a normal query error — one query fails cleanly,
//! instead of the whole process running out of memory.
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
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

mod datafusion_pool;

pub(crate) use datafusion_pool::budgeted_session_context;

/// The fraction of a configured budget we actually enforce: gate at 9/10 and
/// leave the final 1/10 as headroom for allocations too small to track. Applied
/// once, in [`ConnectionMemoryBudget::with_limit`].
const ENFORCED_BUDGET_NUMERATOR: u128 = 9;
const ENFORCED_BUDGET_DENOMINATOR: u128 = 10;

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

    /// Charge `n` bytes to the counter, refusing if that would cross the
    /// ceiling. On refusal the counter is left unchanged. Backs
    /// [`Self::try_reserve`], [`Reservation::try_grow`], and the DataFusion
    /// pool adapter's `try_grow`.
    pub(crate) fn try_grow(&self, n: usize) -> Result<(), OverBudget> {
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
                .map_err(|used| {
                    // Budget exceeded
                    self.denials.fetch_add(1, Ordering::Relaxed);

                    OverBudget {
                        requested: n,
                        used,
                        limit,
                    }
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
    }

    /// Bytes currently reserved.
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
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

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

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
}
