// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-op execution work stats — deterministic counters of the physical
//! work one query **or one write** performs, independent of cache
//! temperature (reads) and of pool width / commit contention (writes).
//!
//! Parallel to [`super::io`] (connection-scoped I/O ledger) and [`super::cpu`]
//! (process CPU): this module scopes to a **single op**. A caller wraps a
//! search or a write in [`with_op_stats`]; the reader or writer minted for
//! that op picks the collector up ([`current`]) and threads it through the
//! fan-out, and each kernel flushes its work counters into it. The same query
//! against the same table state reports the same numbers whether the cache
//! was warm or cold — these count what the plan *did*, not what the storage
//! layer happened to fetch (the [`super::io::UsageMeter`] ledger keeps
//! counting actuals).
//!
//! One struct serves both directions deliberately: `update` and `delete`
//! resolve their predicate through a real reader, so a mutation's scan leg
//! reports read counters and its commit leg reports write counters — one
//! scope, one snapshot, both halves of the op's physical work. That scan
//! is the same work a `SELECT` would do and should be priced as read work
//! (exempting it would let `SELECT`-then-delete-by-id dodge the meter).
//!
//! ## Write-side determinism
//!
//! Two things vary on the write path through no fault of the caller: the
//! shard split follows the writer pool's width, and a contended commit
//! retries its publish. The write counters split accordingly:
//! [`OpStats::rows_written`], the three ingested-byte counters, and
//! [`OpStats::rows_tombstoned`] are functions of the batch, the predicate,
//! and the table state alone — one thread or sixteen, first-attempt commit
//! or third. [`OpStats::superfiles_written`],
//! [`OpStats::superfile_bytes_written`], and [`OpStats::fts_terms_indexed`]
//! are width-dependent execution observations, recorded for reconciliation
//! and never to be priced.
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
//!
//! Write-side exclusions, same reasoning: object-store PUT count/bytes
//! (OCC retries, multipart fan-out, and manifest-part rewrites are our
//! contention and our config, not the caller's — the
//! [`super::io::UsageMeter`] ledger keeps the actuals); the deferred
//! storage-reclaim sweep and ingest-triggered hidden-index maintenance
//! (detached tasks never pick up a collector, so they are excluded by
//! the mint discipline). Per-shard build CPU IS measured: the bracket
//! sits inside each shard's pool closure (`fanout_shards_metered`) and
//! the WAL update's build step, on the thread doing the work — a
//! calling-thread bracket would see nothing, since the caller blocks in
//! `pool.install` for the whole fan-out.

use std::{
    cell::{Cell, RefCell},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

use serde::{Deserialize, Serialize};

use super::cpu;

/// Physical work one op performed — read counters for its query legs,
/// write counters for its commit legs (a pure query leaves the write
/// block zero and vice versa; a mutation fills both). Every read field
/// except [`Self::kernel_cpu_ns`] (measured time, varies run to run) and
/// [`Self::vector_rows_reranked`] (actual execution rows — the deferred
/// path reranks cold cells in place, so the count can shift with cache
/// temperature) is a deterministic plan count: same query, same table
/// state → same value, warm or cold. The write block splits the same way
/// (see the module's write-side determinism note). The struct is
/// `#[non_exhaustive]` because counters land modality by modality.
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
    /// fallback it serves); for SQL, the scan operators' output rows from
    /// DataFusion's own metrics; and filter-side verify decodes
    /// (`exact_match` decodes one row per candidate to compare values,
    /// counted once whichever take served the batch). The id-score
    /// arithmetic fast path decodes nothing and counts nothing;
    /// remap/tombstone-internal id reads count planned ranges only, never
    /// rows.
    pub rows_materialized: u64,
    /// Nanoseconds of the query's bracketed synchronous kernel sections.
    /// One clock everywhere: the thread-CPU clock (schedstat). Engine
    /// kernels bracket their own sections; SQL scans are bracketed per
    /// poll by `MeteredExec`, which measures on whichever worker polls
    /// that partition. DataFusion's `elapsed_compute` is deliberately not
    /// used — it is wall time and omits Parquet decode — so a SQL query
    /// and a search query are priced on the same basis.
    ///
    /// Coverage is not total: orchestration between kernels and async
    /// awaits sit outside the brackets, so this reads lower than a
    /// process-level measurement of the same op. Measured on the standard
    /// vector bench, then closed toward completeness — the remaining gap
    /// is what a consumer must calibrate its reference against.
    pub kernel_cpu_ns: u64,

    // ---- Write-side work (see the module's write-side determinism
    // note; every counter below is a value the commit path already
    // computes, harvested rather than instrumented) ----
    /// Rows the op durably indexed: appended rows plus an update's
    /// replacement rows. Counted from the caller's batch at buffering
    /// time, before any fan-out — invariant to shard count and OCC
    /// retries.
    pub rows_written: u64,
    /// Arrow footprint of the op's scalar/text columns — the payload the
    /// commit encodes to Parquet. Deterministic (input-shaped).
    pub scalar_bytes_written: u64,
    /// f32 payload bytes of the op's vector columns (`rows × dim × 4`) —
    /// the input to rotation, k-means, and quantized encode.
    /// Deterministic (input-shaped).
    pub vector_bytes_written: u64,
    /// Arrow footprint of the FTS-indexed text columns, a subset of
    /// [`Self::scalar_bytes_written`], not additional payload: the
    /// tokenize → dictionary → postings build scales with it.
    /// Deterministic (input-shaped).
    pub fts_text_bytes_written: u64,
    /// Rows an update or delete tombstoned (sidecar bits set; excludes
    /// not-found ids). A function of the predicate and table state.
    pub rows_tombstoned: u64,
    /// New superfile objects this op published. Host-width dependent —
    /// the shard split follows the writer pool — so recorded for
    /// reconciliation, never to be priced.
    pub superfiles_written: u64,
    /// On-storage bytes of those superfiles (sealed bodies, excluding
    /// manifest parts and the pointer). Per-superfile fixed overhead
    /// scales with shard count, so width-dependent; recorded only.
    pub superfile_bytes_written: u64,
    /// Object-store PUTs this op's *plan* implies — the write-side twin
    /// of [`Self::planned_read_ranges`], and priceable for the same
    /// reason.
    ///
    /// Actual PUTs ([`Self::put_requests`]) are ours, not the caller's:
    /// they scale with how many shards the writer pool split the commit
    /// into and how many times a contended publish retried. This counts
    /// instead what the data itself requires, which neither varies with:
    ///
    /// - the objects the commit's *ingested payload* occupies at the
    ///   table's target superfile size, and
    /// - the manifest json + pointer that every commit must publish
    ///   ([`MANIFEST_PUTS_PER_COMMIT`]), counted once per successful
    ///   commit rather than once per OCC attempt.
    ///
    /// Manifest *parts* are excluded deliberately: their count follows
    /// the superfile count, so they carry the same width-dependence the
    /// priced legs exist to avoid.
    ///
    /// The object term is derived from the buffered input rather than the
    /// sealed output, and that distinction is load-bearing rather than
    /// cosmetic. Per-superfile overhead is not a fixed footer: every shard
    /// carries its own dictionary, FST and index headers, so on a corpus
    /// whose vocabulary is shared across rows the same input seals to
    /// roughly four times more bytes at pool width 16 than at width 1.
    /// Dividing that by the target would make an identical append plan
    /// more requests on a wider host — the exact width-dependence this
    /// counter exists to avoid.
    ///
    /// Requests are a real and material share of write cost — a PUT is
    /// 12.5x a GET in the bench cost model, and unlike a warm read's
    /// ranges they never resolve from residency, so a commit always pays
    /// them. A consumer that accounts for write cost without a request
    /// term would understate every write by that share.
    pub planned_write_requests: u64,
    /// Distinct FTS terms built, summed per column across this op's new
    /// superfiles. A term present in k shards counts k times, so
    /// width-dependent; recorded only.
    pub fts_terms_indexed: u64,
}

/// Accumulates one op's work counters across its fan-out (tokio unit
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
    rows_written: AtomicU64,
    scalar_bytes_written: AtomicU64,
    vector_bytes_written: AtomicU64,
    fts_text_bytes_written: AtomicU64,
    rows_tombstoned: AtomicU64,
    superfiles_written: AtomicU64,
    planned_write_requests: AtomicU64,
    superfile_bytes_written: AtomicU64,
    fts_terms_indexed: AtomicU64,
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
        // Unconditional by design. Much of what reaches here was measured
        // on a rayon worker and carried back as data — the resident decode
        // wave, the per-cell probes — and the poll-level bracket reads only
        // its own thread's clock, so it never contained those nanoseconds.
        // Suppressing them here would not de-duplicate, it would delete.
        // De-duplication happens where the measurement happens, in
        // [`timed_kernel`] and [`timed_section`], which run on the same
        // thread as the bracket that would otherwise cover them.
        self.kernel_cpu_ns.fetch_add(ns, Ordering::Relaxed);
    }

    /// Flush one buffered append's input shape — rows plus the three
    /// ingested-byte legs, computed together at the buffering site.
    pub(crate) fn add_ingested_write(&self, rows: u64, scalar: u64, vector: u64, fts_text: u64) {
        self.rows_written.fetch_add(rows, Ordering::Relaxed);
        self.scalar_bytes_written
            .fetch_add(scalar, Ordering::Relaxed);
        self.vector_bytes_written
            .fetch_add(vector, Ordering::Relaxed);
        self.fts_text_bytes_written
            .fetch_add(fts_text, Ordering::Relaxed);
    }

    /// Flush a mutation's tombstoned-row count.
    pub(crate) fn add_rows_tombstoned(&self, rows: u64) {
        self.rows_tombstoned.fetch_add(rows, Ordering::Relaxed);
    }

    /// Flush one committed publish batch's output shape (after the
    /// commit returns Ok, so a failed or retried publish never counts).
    pub(crate) fn add_commit_outputs(&self, superfiles: u64, bytes: u64, fts_terms: u64) {
        self.superfiles_written
            .fetch_add(superfiles, Ordering::Relaxed);
        self.superfile_bytes_written
            .fetch_add(bytes, Ordering::Relaxed);
        self.fts_terms_indexed
            .fetch_add(fts_terms, Ordering::Relaxed);
    }

    /// Flush the PUTs one committed publish *planned*: the data objects
    /// the caller derived from its input shape, plus the manifest json +
    /// pointer every manifest commit publishes. Called beside the other
    /// commit flushes, under the same post-Ok discipline, so a failed or
    /// retried publish never counts.
    ///
    /// Callers state their own object term because it is shape-specific:
    /// a buffered append plans `ceil(ingested payload / target object size)`;
    /// an update plans exactly 1 (its replacement rows land in the WAL's
    /// single preallocated superfile by design). A delete flushes nothing
    /// here — it commits no manifest, and its per-superfile tombstone
    /// CAS-writes (like the WAL state-doc writes of both mutations) stay
    /// recorded-only in `put_requests` for now: their count follows where
    /// the target rows live, a table-state quantity a later change can
    /// plan from the resolve — deliberately not guessed at today.
    pub(crate) fn add_planned_commit_requests(&self, data_objects: u64) {
        self.planned_write_requests.fetch_add(
            data_objects.saturating_add(MANIFEST_PUTS_PER_COMMIT),
            Ordering::Relaxed,
        );
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
            rows_written: self.rows_written.load(Ordering::Relaxed),
            scalar_bytes_written: self.scalar_bytes_written.load(Ordering::Relaxed),
            vector_bytes_written: self.vector_bytes_written.load(Ordering::Relaxed),
            fts_text_bytes_written: self.fts_text_bytes_written.load(Ordering::Relaxed),
            rows_tombstoned: self.rows_tombstoned.load(Ordering::Relaxed),
            superfiles_written: self.superfiles_written.load(Ordering::Relaxed),
            planned_write_requests: self.planned_write_requests.load(Ordering::Relaxed),
            superfile_bytes_written: self.superfile_bytes_written.load(Ordering::Relaxed),
            fts_terms_indexed: self.fts_terms_indexed.load(Ordering::Relaxed),
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

/// PUTs every successful commit publishes regardless of shape: the
/// manifest json and the pointer file, one each. Manifest *parts* are
/// excluded — their count follows the superfile count, which is
/// width-dependent (see [`OpStats::planned_write_requests`]).
const MANIFEST_PUTS_PER_COMMIT: u64 = 2;

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
    // An enclosing bracket on THIS thread already covers whatever `f`
    // does — DataFusion is pull-based, so a poll drives its subtree
    // synchronously on the polling thread. Measuring again here would
    // charge the same nanoseconds twice; it was roughly 59% of what a
    // vector query through the SQL TVF appeared to cost against the
    // identical direct call. Same-thread by construction, which is why
    // the check belongs here and not at the collector's sink.
    if outer_bracket_active() {
        return f();
    }
    let start = cpu::thread_cpu_ns();
    let value = f();
    stats.add_kernel_cpu_ns(cpu::thread_cpu_delta_ns(start));
    value
}

thread_local! {
    /// Depth of live outer CPU brackets on this thread.
    ///
    /// `MeteredExec` brackets a whole DataFusion poll, and DataFusion is
    /// pull-based, so that poll synchronously drives the operator subtree
    /// beneath it — including the search kernels, which bracket their own
    /// sections. Tokio runs one task at a time per thread, so a raised
    /// depth means exactly "this thread is already inside a bracket that
    /// covers whatever runs next", and the inner folds must stand down
    /// and let the outermost one report.
    ///
    /// This lives here rather than beside `MeteredExec` because the fold
    /// it guards is here: every CPU nanosecond reaches the collector
    /// through [`OpStatsCollector::add_kernel_cpu_ns`], whether it came
    /// from `timed_kernel` or from a `timed_section` value a caller
    /// folded by hand. Gating one choke point covers them all.
    static OUTER_BRACKET_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// True while an enclosing CPU bracket on this thread is already
/// measuring, so an inner fold would double-charge.
pub(crate) fn outer_bracket_active() -> bool {
    OUTER_BRACKET_DEPTH.with(|d| d.get()) > 0
}

/// Marks this thread as inside an outer CPU bracket until dropped.
///
/// A real counter, so the guard holds without a LIFO argument: overlapping
/// lifetimes each add one and remove one, and the depth is zero exactly
/// when no bracket is live. Decrements on drop, unwind included — a poll
/// that panicked out of a raised depth would otherwise leave the thread
/// permanently silenced, and every later query scheduled onto that worker
/// would fold nothing.
pub(crate) struct OuterBracketGuard;

impl OuterBracketGuard {
    pub(crate) fn enter() -> Self {
        OUTER_BRACKET_DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        Self
    }
}

impl Drop for OuterBracketGuard {
    fn drop(&mut self) {
        OUTER_BRACKET_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Bracket one synchronous section with the thread-CPU clock, gated on
/// [`metering_active`] so an unmetered process pays one relaxed load and
/// no procfs reads. For the superfile layers, which have no collector —
/// the ns travel back to the supertable as data.
pub(crate) fn timed_section<R>(f: impl FnOnce() -> R) -> (R, u64) {
    // Zero when an enclosing bracket on this thread already covers the
    // section, so the caller's later fold adds nothing. A section running
    // on a rayon worker sees depth 0 — thread-locals are per-thread — and
    // reports its real time, which is exactly right: the poll-level
    // bracket never measured that worker's CPU.
    if outer_bracket_active() {
        return (f(), 0);
    }
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
    fn a_same_thread_measurement_stands_down_inside_an_outer_bracket() {
        // `MeteredExec` brackets a whole DataFusion poll, and that poll
        // drives the search kernels inline — they bracket their own
        // sections, so without this both measure the same nanoseconds. It
        // was roughly 59% of what a vector query through the SQL TVF
        // appeared to cost against the identical direct call.
        let (_, stats) = with_op_stats(|| {
            let collector = current();
            let mut ran = 0u32;
            let _outer = OuterBracketGuard::enter();
            timed_kernel(&collector, || ran += 1);
            let (_, section_ns) = timed_section(|| ran += 1);
            assert_eq!(ran, 2, "the work still runs; only the clock stands down");
            assert_eq!(
                section_ns, 0,
                "a section the enclosing bracket already covers reports \
                 nothing for its caller to fold"
            );
        });
        assert_eq!(
            stats.kernel_cpu_ns, 0,
            "the enclosing bracket is the one that reports this thread's CPU"
        );
    }

    #[test]
    fn a_carried_measurement_folds_even_under_an_outer_bracket() {
        // The counterpart, and the reason the check cannot live at the
        // collector's sink: nanoseconds measured on a rayon worker and
        // carried back as data were never inside the poll-level bracket,
        // which reads only its own thread's clock. Dropping them would not
        // de-duplicate, it would delete — measured at 0.25-1.16 ms per
        // query on the search TVF path.
        let (_, stats) = with_op_stats(|| {
            let collector = current().expect("collector installed");
            let _outer = OuterBracketGuard::enter();
            collector.add_kernel_cpu_ns(4_242);
        });
        assert_eq!(
            stats.kernel_cpu_ns, 4_242,
            "a value measured elsewhere must still reach the collector"
        );
    }

    #[test]
    fn an_outer_bracket_restores_its_depth_on_unwind() {
        // A poll that panicked out of a raised depth would silence every
        // later query scheduled onto that worker.
        let _ = catch_unwind(|| {
            let _outer = OuterBracketGuard::enter();
            panic!("poll blew up");
        });
        assert!(
            !outer_bracket_active(),
            "the guard must restore the depth even when the poll unwinds"
        );
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
