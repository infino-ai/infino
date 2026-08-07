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
//! With no collector installed the per-flush cost is one `Option` check,
//! and the superfile-level kernel-CPU brackets gate their procfs reads on
//! [`metering_active`] (one relaxed atomic load); counters accumulate
//! locally in kernels and flush per superfile / work unit, never inside
//! scoring loops.
//!
//! ## Deliberate exclusions
//!
//! Reads whose cost amortizes across queries or would break the
//! warm/cold invariance are NOT counted, by design: superfile open I/O
//! (footers, subsection headers) and the reader-cached Parquet
//! metadata / page-index parse (paid once per reader lifetime, not per
//! query); tombstone sidecar prefetches and manifest freshness probes
//! (table-state / consistency-policy reads, not plan work); the hidden
//! index's per-generation fast state; and phase C's bookkeeping
//! refetches (cluster index + Sq8 meta re-reads on the warm-only rerank
//! wave — cold runs have no phase C, so pricing them would make the one
//! priced counter temperature-dependent); and the materialization takes'
//! streamed page ranges (a promoted resident reader decodes in place —
//! reader-cache state again; `rows_materialized` carries that leg
//! invariantly). Their expected cost is recovered through a consumer's
//! calibrated rates, never surcharged on the query that happened to pay
//! them.

use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use super::cpu;

/// Physical work one query performed. Every field except
/// [`Self::kernel_cpu_ns`] (measured time, varies run to run) and
/// [`Self::vector_rows_reranked`] (actual execution rows — the deferred
/// path reranks cold cells in place, so the count can shift with cache
/// temperature) is a deterministic plan count: same query, same table
/// state → same value, warm or cold. The struct is `#[non_exhaustive]`
/// because counters land modality by modality.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OpStats {
    /// FTS posting-list bytes resident for the query's clauses — the term
    /// metadata + skip tables + posting blocks the kernels index into
    /// (musts, shoulds, plain OR terms, and negation filters). Counted once
    /// per superfile even when ranged slices share a cursor set.
    pub fts_postings_bytes: u64,
    /// Vector cells the query's scans actually probed, summed across
    /// superfiles (a cell counts once per superfile scan that chose at
    /// least one of its clusters).
    pub vector_cells_scanned: u64,
    /// Quantized codes the cell scans estimated (Σ cluster row counts over
    /// every chosen cluster, warm and cold arms alike).
    pub vector_candidates_scanned: u64,
    /// Rows actually rescored at full precision, across every arm: the
    /// global-shortlist rerank (phase C of the deferred path), the scan's
    /// immediate cold-cell rerank, and the immediate probe paths
    /// (pre-drain user tables and filtered search). An execution
    /// diagnostic, not a plan count: on the deferred path cold cells
    /// rerank in place under a width-divided budget, so the total can
    /// shift with cache temperature. Never priced — the rerank leg's
    /// cost is CPU-dominated and carried by a consumer's process-level
    /// CPU accounting; rows are deliberately NOT folded into the
    /// request-shaped range counter below (survivor fetches coalesce
    /// into a handful of real requests, so one-range-per-row would
    /// inflate it far past request scale).
    pub vector_rows_reranked: u64,
    /// Byte-source ranges the plan requested, before coalescing and before
    /// the cache decides whether a request becomes a local read or a GET —
    /// the warm-equivalent request-work measure, kept REQUEST-SHAPED:
    /// its magnitudes stay commensurate with real object-store requests
    /// so a consumer can price it at a per-request rate. FTS: one per
    /// PFOR term posting range (two when a hint-less slot forces a
    /// header probe), one per phrase-member posting and position-run
    /// range, one per dictionary fetch a build performs, and one per
    /// df-header probe. Vector: per scanned cell, its cluster index plus
    /// one prefix/block range per chosen cluster (plus the lazy Sq8 meta
    /// table when the column stores one), plus one per stable-id region
    /// / `_id` remap read / hydrated-section cell read. Rerank rows are
    /// deliberately excluded — see [`Self::vector_rows_reranked`]. SQL:
    /// one per Parquet range the scan requests. Materialization takes are
    /// deliberately excluded (a promoted resident reader decodes in place
    /// while a lazy reader streams ranges — reader-cache state, so
    /// counting them would break the warm/cold invariance);
    /// [`Self::rows_materialized`] is that leg's invariant signal.
    pub planned_read_ranges: u64,
    /// Parquet **data-page** bytes SQL scans requested through the
    /// DataFusion store, independent of whether they were served from
    /// resident bytes or fetched. Footer and page-index reads never
    /// transit the store — they are open-time amortized state (see the
    /// module's exclusions) — so they are deliberately not in here.
    pub sql_page_bytes: u64,
    /// Rows decoded from stored columns to build results — the scalar
    /// projection decode (`resolve_columns`, including the `_id` stamping
    /// fallback it serves) plus, for SQL, the scan operators' output rows
    /// from DataFusion's own metrics. The id-score arithmetic fast path
    /// decodes nothing and counts nothing; remap/tombstone-internal id
    /// reads count planned ranges only, never rows.
    pub rows_materialized: u64,
    /// Nanoseconds of the query's bracketed synchronous kernel sections.
    /// Engine kernels use the thread-CPU clock; SQL operator sections use
    /// DataFusion's own `elapsed_compute` instrumentation (an `Instant`
    /// timer around synchronous poll work — approximately on-CPU for
    /// compute-bound operators, excluding async I/O waits). A refinement
    /// of — not a replacement for — a consumer's own process-level CPU
    /// accounting: fan-out glue and async awaits are outside the brackets.
    pub kernel_cpu_ns: u64,
}

/// Accumulates one query's work counters across its fan-out (tokio unit
/// tasks and rayon kernel waves both add through the same `Arc`).
#[derive(Debug, Default)]
pub(crate) struct OpStatsCollector {
    fts_postings_bytes: AtomicU64,
    vector_cells_scanned: AtomicU64,
    vector_candidates_scanned: AtomicU64,
    vector_rows_reranked: AtomicU64,
    planned_read_ranges: AtomicU64,
    sql_page_bytes: AtomicU64,
    rows_materialized: AtomicU64,
    kernel_cpu_ns: AtomicU64,
}

impl OpStatsCollector {
    /// Flush a kernel's posting-bytes tally (one add per superfile).
    pub(crate) fn add_fts_postings_bytes(&self, bytes: u64) {
        self.fts_postings_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Flush one superfile scan's cell + candidate tallies.
    pub(crate) fn add_vector_scan(&self, cells: u64, candidates: u64) {
        self.vector_cells_scanned
            .fetch_add(cells, Ordering::Relaxed);
        self.vector_candidates_scanned
            .fetch_add(candidates, Ordering::Relaxed);
    }

    /// Flush the global-shortlist rerank's row count.
    pub(crate) fn add_vector_rows_reranked(&self, rows: u64) {
        self.vector_rows_reranked.fetch_add(rows, Ordering::Relaxed);
    }

    /// Flush a kernel's planned byte-source range count.
    pub(crate) fn add_planned_read_ranges(&self, ranges: u64) {
        self.planned_read_ranges
            .fetch_add(ranges, Ordering::Relaxed);
    }

    /// Flush one SQL Parquet range request's byte length.
    pub(crate) fn add_sql_page_bytes(&self, bytes: u64) {
        self.sql_page_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Flush one resolve wave's decoded row count.
    pub(crate) fn add_rows_materialized(&self, rows: u64) {
        self.rows_materialized.fetch_add(rows, Ordering::Relaxed);
    }

    /// Flush one bracketed kernel section's on-CPU nanoseconds.
    pub(crate) fn add_kernel_cpu_ns(&self, ns: u64) {
        self.kernel_cpu_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// The counters accumulated so far.
    pub(crate) fn snapshot(&self) -> OpStats {
        OpStats {
            fts_postings_bytes: self.fts_postings_bytes.load(Ordering::Relaxed),
            vector_cells_scanned: self.vector_cells_scanned.load(Ordering::Relaxed),
            vector_candidates_scanned: self.vector_candidates_scanned.load(Ordering::Relaxed),
            vector_rows_reranked: self.vector_rows_reranked.load(Ordering::Relaxed),
            planned_read_ranges: self.planned_read_ranges.load(Ordering::Relaxed),
            sql_page_bytes: self.sql_page_bytes.load(Ordering::Relaxed),
            rows_materialized: self.rows_materialized.load(Ordering::Relaxed),
            kernel_cpu_ns: self.kernel_cpu_ns.load(Ordering::Relaxed),
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

/// Live [`with_op_stats`] scopes, process-wide. Superfile-layer kernel
/// brackets gate their procfs reads on this instead of a collector they
/// cannot see (the collector rides the supertable reader, and the kernel
/// may run on a different thread than the scope) — so an unmetered
/// process pays one relaxed load per bracket, never a schedstat read.
static ACTIVE_SCOPES: AtomicUsize = AtomicUsize::new(0);

/// `true` while any [`with_op_stats`] scope is live anywhere in the
/// process. A cross-thread gate, deliberately coarser than [`current`]:
/// a kernel polled off the scope's thread must still measure.
pub(crate) fn metering_active() -> bool {
    ACTIVE_SCOPES.load(Ordering::Relaxed) > 0
}

/// Decrements [`ACTIVE_SCOPES`] when a [`with_op_stats`] scope ends
/// (panic unwind included). [`suppressed`] never touches the counter:
/// suppression detaches the thread-local inside a scope that is still
/// running and still metering elsewhere.
struct ActiveScopeGuard;

impl Drop for ActiveScopeGuard {
    fn drop(&mut self) {
        ACTIVE_SCOPES.fetch_sub(1, Ordering::Relaxed);
    }
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
    ACTIVE_SCOPES.fetch_add(1, Ordering::Relaxed);
    let _active = ActiveScopeGuard;
    let value = f();
    (value, collector.snapshot())
}

/// The collector installed on this thread, if a [`with_op_stats`] scope is
/// active. Read at reader mint (the last point on the caller's thread before
/// the query fans out).
pub(crate) fn current() -> Option<Arc<OpStatsCollector>> {
    CURRENT.with(|slot| slot.borrow().clone())
}

/// Bracket one synchronous kernel section with the thread-CPU clock and
/// flush the on-CPU delta into `collector`. `f` must run entirely on the
/// calling thread (rayon pool closures and inline kernel branches do);
/// with no collector — or off Linux procfs — it is a plain call.
pub(crate) fn timed_kernel<T>(
    collector: &Option<Arc<OpStatsCollector>>,
    f: impl FnOnce() -> T,
) -> T {
    // Gate on the live-scope counter like the superfile-level brackets:
    // a reader held past its scope still carries a collector, and its
    // queries must not keep paying procfs reads into an Arc nobody
    // snapshots.
    if !metering_active() {
        return f();
    }
    let Some(stats) = collector else {
        return f();
    };
    let start = cpu::thread_cpu_ns();
    let value = f();
    stats.add_kernel_cpu_ns(cpu::thread_cpu_delta_ns(start));
    value
}

/// Bracket one synchronous section with the thread-CPU clock, gated on
/// [`metering_active`] so an unmetered process pays one relaxed load and
/// no procfs reads. For the superfile layers, which have no collector —
/// the ns travel back to the supertable as data.
pub(crate) fn timed_section<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let start = metering_active().then(cpu::thread_cpu_ns).flatten();
    let out = f();
    (out, cpu::thread_cpu_delta_ns(start))
}

/// Run `f` with NO collector installed, restoring the active scope after.
/// For constructing state that outlives the current query (e.g. a cached
/// `SessionContext`): anything capturing [`current`] inside `f` stays
/// detached, so a long-lived cache can never bill later queries into this
/// scope.
pub(crate) fn suppressed<T>(f: impl FnOnce() -> T) -> T {
    let previous = CURRENT.with(|slot| slot.borrow_mut().take());
    let _guard = ScopeGuard { previous };
    f()
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

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
            let result = catch_unwind(|| {
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
