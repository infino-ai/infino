// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query execution work stats — deterministic counters of the physical
//! work one query performs, independent of cache temperature.
//!
//! Parallel to [`super::io`] (connection-scoped I/O ledger) and [`super::cpu`]
//! (process CPU): this module scopes to a **single query**. A caller wraps a
//! search in [`with_op_stats`]; the reader minted for that query picks the
//! collector up ([`current`]) and threads it through the fan-out, and each
//! kernel flushes its work counters into it. The same query against the same
//! table state reports the same numbers whether the cache was warm or cold —
//! these count what the plan *did*, not what the storage layer happened to
//! fetch (the [`super::io::UsageMeter`] ledger keeps counting actuals).
//!
//! With no collector installed the per-flush cost is one `Option` check;
//! counters accumulate locally in kernels and flush per superfile / work
//! unit, never inside scoring loops.

use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use serde::{Deserialize, Serialize};

/// Physical work one query performed. Every field is a plain count; the
/// struct is `#[non_exhaustive]` because counters land modality by modality
/// (FTS first; vector, SQL, planned-read, and CPU attribution follow).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpStats {
    /// FTS posting-list bytes resident for the query's clauses — the term
    /// metadata + skip tables + posting blocks the kernels index into
    /// (musts, shoulds, plain OR terms, and negation filters). Counted once
    /// per superfile even when ranged slices share a cursor set.
    pub fts_postings_bytes: u64,
}

/// Accumulates one query's work counters across its fan-out (tokio unit
/// tasks and rayon kernel waves both add through the same `Arc`).
#[derive(Debug, Default)]
pub struct OpStatsCollector {
    fts_postings_bytes: AtomicU64,
}

impl OpStatsCollector {
    /// Flush a kernel's posting-bytes tally (one add per superfile).
    pub(crate) fn add_fts_postings_bytes(&self, bytes: u64) {
        self.fts_postings_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// The counters accumulated so far.
    pub fn snapshot(&self) -> OpStats {
        OpStats {
            fts_postings_bytes: self.fts_postings_bytes.load(Ordering::Relaxed),
        }
    }
}

thread_local! {
    /// The collector queries minted on this thread attach to. Set only inside
    /// [`with_op_stats`]; the sync search entry points run on the caller's
    /// thread up to reader mint, which is where [`current`] is read — from
    /// there the collector travels as an explicit `Arc` through the fan-out,
    /// so spawned tasks and pool closures never consult this slot.
    static CURRENT: RefCell<Option<Arc<OpStatsCollector>>> = const { RefCell::new(None) };
}

/// Restores the previously-installed collector when the scope ends, so
/// nested [`with_op_stats`] scopes and panics both unwind cleanly.
struct ScopeGuard {
    previous: Option<Arc<OpStatsCollector>>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CURRENT.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// Run `f` with a fresh per-query collector installed for the current
/// thread, returning `f`'s result alongside the work stats every query
/// executed inside the scope accumulated.
///
/// Scopes nest: an inner scope shadows the outer one and restores it on
/// exit (including panic unwind), so an outer scope never absorbs an inner
/// query's counters.
pub fn with_op_stats<T>(f: impl FnOnce() -> T) -> (T, OpStats) {
    let collector = Arc::new(OpStatsCollector::default());
    let previous = CURRENT.with(|slot| slot.borrow_mut().replace(Arc::clone(&collector)));
    let _guard = ScopeGuard { previous };
    let value = f();
    (value, collector.snapshot())
}

/// The collector installed on this thread, if a [`with_op_stats`] scope is
/// active. Read at reader mint (the last point on the caller's thread before
/// the query fans out).
pub(crate) fn current() -> Option<Arc<OpStatsCollector>> {
    CURRENT.with(|slot| slot.borrow().clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_collects_and_clears() {
        assert!(current().is_none(), "no collector outside a scope");
        let (value, stats) = with_op_stats(|| {
            let collector = current().expect("collector installed inside the scope");
            collector.add_fts_postings_bytes(123);
            7u32
        });
        assert_eq!(value, 7);
        assert_eq!(stats.fts_postings_bytes, 123);
        assert!(current().is_none(), "scope uninstalls its collector");
    }

    #[test]
    fn nested_scopes_shadow_and_restore() {
        let (_, outer) = with_op_stats(|| {
            let outer_collector = current().expect("outer collector");
            outer_collector.add_fts_postings_bytes(1);
            let (_, inner) = with_op_stats(|| {
                current()
                    .expect("inner collector shadows outer")
                    .add_fts_postings_bytes(10);
            });
            assert_eq!(inner.fts_postings_bytes, 10);
            // The outer collector is restored and keeps accumulating.
            current()
                .expect("outer collector restored")
                .add_fts_postings_bytes(2);
        });
        assert_eq!(
            outer.fts_postings_bytes, 3,
            "outer scope never absorbs the inner query's counters"
        );
    }

    #[test]
    fn a_panicking_scope_still_restores_the_previous_collector() {
        let (_, outer) = with_op_stats(|| {
            let result = std::panic::catch_unwind(|| {
                let (_, _) = with_op_stats(|| panic!("kernel failure"));
            });
            assert!(result.is_err());
            current()
                .expect("outer collector survives the inner panic")
                .add_fts_postings_bytes(5);
        });
        assert_eq!(outer.fts_postings_bytes, 5);
    }
}
