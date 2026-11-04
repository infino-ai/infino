// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Choosing the order a superfile stores its documents in.
//!
//! A posting list costs less to store and less to walk when the
//! documents carrying a term sit close together: the gaps between their
//! ids are smaller, and a block covers a narrower span so more of the
//! list can be skipped without decoding. Arrival order gives no such
//! grouping, and nothing about arrival order is worth preserving inside
//! the full-text index, because the index reaches its rows through the
//! doc-id map rather than by position.
//!
//! So compaction picks an order. This module computes it by recursive
//! graph bisection: split the documents in two, repeatedly move
//! documents across the split while doing so lowers the cost of the
//! terms they carry, then recurse into each half. Reading the leaves
//! left to right gives an order in which documents sharing terms are
//! neighbours.
//!
//! The cost being minimised is the standard one for this problem, a
//! stand-in for the bits a term's postings take once split across the
//! two halves:
//!
//! ```text
//! cost(half) = sum over terms of  deg * log2(size / (deg + 1))
//! ```
//!
//! where `deg` is how many documents in that half carry the term and
//! `size` is the half's document count. Moving a document changes only
//! the terms it carries, so a move's effect is summed over those terms
//! alone, which is what makes the iteration affordable.
//!
//! Nothing here reads or writes a blob. It takes term sets and returns
//! an order, so it can be tested against the cost it claims to lower.

use std::{
    fmt, mem,
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed},
    },
    time::Instant,
};

#[cfg(feature = "detailed-tracing")]
use rayon::current_thread_index;
use rayon::{current_num_threads, join, prelude::*};
use tracing::Span;

use crate::{
    runtime_metrics::cpu::{thread_cpu_delta_ns, thread_cpu_ns},
    superfile::fts::posting::BLOCK_LEN,
    utils::trace::detail_span,
};

/// The sign bit of an `f32`.
const SIGN_BIT: u32 = 0x8000_0000;

/// Documents below this per-partition size are left in the order they
/// already have. Splitting further costs more than the grouping is
/// worth, and the gaps inside a group this small are already short.
///
/// One posting block of documents (two sub-blocks): a group that fits a
/// block is already as close as the postings can place it, and splitting
/// below it was measured to cost time and index bytes (the numbers are
/// with the benches; the round loop below says why a leaf-sized round
/// stops paying).
const MIN_PARTITION: usize = BLOCK_LEN;

/// Reorder convergence ratio.
///
/// Swept from 0.0 (every split run to the round ceiling) to 0.5 on two corpora
/// of the same document count but very different text — 5.03M documents of
/// encyclopedia prose at 7.76 GiB, and the same count of web crawl at
/// 17.70 GiB, so documents average 2.3x longer:
///
/// ```text
///            encyclopedia              web crawl
///   ratio  compact   index vs best   compact   index vs best
///   0.00   158.7 s      +0.38%       253.5 s      best
///   0.02   144.7 s      +0.40%       245.4 s     +0.033%
///   0.05   141.6 s       best        248.2 s     +0.066%
///   0.10   139.8 s      +0.44%       235.0 s     +0.138%
///   0.20   136.0 s      +1.07%       230.9 s     +0.270%
///   0.50   128.3 s      +1.83%       223.7 s     +0.715%
/// ```
///
/// The *shape* depends on the corpus: the first has an interior optimum, where
/// reordering past it makes the index worse, and the second is monotonic, where
/// more reordering always helps a little. So there is no ratio that is optimal
/// everywhere, and nothing here should be read as claiming one.
///
/// What does hold on both is that the whole range is narrow — 1.8% and 0.7%
/// end to end — and this value lands within 0.07% of each corpus's own best
/// while running faster than doing every round on both. That is the reason it
/// is a fixed value rather than something calibrated per corpus, and not a
/// knob: the error from not knowing the text is far smaller than what
/// measuring it would cost, and a service cannot sweep a customer's data
/// before indexing it. What does adapt to the corpus is the bar this ratio is
/// taken against, which is measured from the data on every run.
///
/// A looser ratio buys optimize time with query time. At 0.20 the bisection
/// did 10–16% less work, and a 5M-document table's queries ran 1.6% slower
/// across every query shape measured, with its index 0.35% larger; at this
/// value they ran as before. Query time is what the reordering is for, so
/// the ratio stays here.
///
/// Index size stands in for layout quality throughout. The reordering exists
/// for query performance, which is not resolved at this scale.
const REORDER_CONVERGENCE: f32 = 0.05;

/// Ceiling on a split's move rounds. Equal to the fixed count the bisection
/// used before the convergence test existed, so setting [`REORDER_CONVERGENCE`]
/// to zero reproduces the old behaviour exactly. A backstop against a corpus
/// whose rounds keep paying, not the working limit — the convergence test is
/// what normally ends a split.
const REORDER_MAX_ROUNDS: usize = 20;

/// Fewest items one rayon task takes in the bisection's per-document and
/// per-term loops. The work per item is a few reads and an atomic store, so
/// a task that small is mostly the scheduler's: left to split down to one
/// item, a round on a wide pool spends more time in work-stealing and epoch
/// bookkeeping than in the loop itself.
const PAR_MIN_ITEMS: usize = 4096;

/// Documents whose gains [`rank_by_gain`] sums side by side, one
/// accumulator each: enough independent chains of adds to keep the adder
/// busy while each waits on its own previous add.
const GAIN_LANES: usize = 8;

/// How hard the bisection works a split, and when it decides a split has
/// stopped paying.
///
/// A diminishing-returns ratio is the working limit and the round count only a
/// backstop, because how much work a split is worth depends on how much
/// structure a corpus has: a round count tuned on one body of text does not
/// carry to another, a ratio does. The tests override both; nothing else does,
/// which is why these are constants rather than configuration.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BisectParams {
    /// Stop once a round's realised gain per document falls below this
    /// fraction of the run's reference gain per document, which [`Bisect`]
    /// takes from the root split's first round, or from the first completed
    /// level below it when the root's lowered nothing — never from the first
    /// round of the split being tested. `0.0` disables the test.
    pub(crate) convergence: f32,
    /// Ceiling on a split's move rounds whatever `convergence` says.
    pub(crate) max_rounds: usize,
}

/// A bisection in progress: the limits in force, plus the reference the
/// convergence test measures rounds against.
///
/// The reference is the gain per document of the first first round that
/// lowered the cost: the root's, since it is processed first, or, when the
/// root moved documents without lowering anything, the summed first rounds of
/// the first level below it that did. Either way the bar is taken from the
/// corpus rather than chosen. Measuring a round against it, rather than
/// against the first round of the split the round belongs to, is what stops
/// large partitions being cut short: a big partition banks most of its
/// available gain in round one, so a bar set by that round rejects later
/// rounds that are still worth more than anything happening deep in the tree.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bisect<'a> {
    params: BisectParams,
    /// `f64` bits of the reference gain per document; `0` until a round sets it.
    reference: &'a AtomicU64,
    /// Where splits report what they did.
    report: &'a RunReport,
    /// Parent for the spans splits open. Set explicitly because a rayon task
    /// stolen by another thread runs under whatever span that thread has open.
    parent: &'a Span,
}

impl Bisect<'_> {
    /// Whether an offer has set the bar yet.
    fn has_reference(&self) -> bool {
        self.reference.load(Relaxed) != 0
    }

    /// Whether a round that realised `gain` over `len` documents means this
    /// split has stopped paying.
    fn converged(&self, gain: f64, len: usize) -> bool {
        if self.params.convergence <= 0.0 {
            return false;
        }
        let reference = f64::from_bits(self.reference.load(Relaxed));
        // No reference yet means no offering level has lowered the cost;
        // there is nothing to measure against, so the ceiling is the only
        // limit.
        reference > 0.0
            && gain / (len.max(1) as f64) < f64::from(self.params.convergence) * reference
    }

    /// Offer `gain`, what a first round over `len` documents lowered the
    /// cost by, as the reference for the whole run; the first offer that
    /// lowered the cost sets it.
    ///
    /// Offers come only from where the order of events is fixed: the root,
    /// which runs alone, and each level the top of the bisection completes
    /// as a whole, whose splits' gains are summed in partition order. Below
    /// that several splits run at once, and letting them offer would hand
    /// the bar to whichever the thread pool happened to finish first and
    /// make the output depend on scheduling. A root can move documents and
    /// realise nothing — a corpus that repeats a document has the swap
    /// exchange two copies — so the levels under it offer in turn; a run
    /// no level of which lowers the cost has no bar and keeps every round.
    fn offer_reference(&self, gain: f64, len: usize) {
        let per_doc = gain / (len.max(1) as f64);
        if per_doc > 0.0 {
            let _ = self
                .reference
                .compare_exchange(0, per_doc.to_bits(), Relaxed, Relaxed);
        }
    }
}

impl Default for BisectParams {
    fn default() -> Self {
        Self {
            convergence: REORDER_CONVERGENCE,
            max_rounds: REORDER_MAX_ROUNDS,
        }
    }
}

/// Splits at least this large get a span of their own. Smaller ones only add
/// to their depth's totals in [`BisectReport`].
const SPLIT_SPAN_MIN_DOCS: usize = 1 << 20;

/// Parallel partitions at least this large get a span showing the thread they
/// ran on. Deep enough that a large run spreads over every core, shallow
/// enough that a 30M-document run opens a few hundred spans, not thousands.
const PARALLEL_SPAN_MIN_DOCS: usize = 1 << 18;

/// Bisections running in this process right now.
static RUNNING: AtomicUsize = AtomicUsize::new(0);

/// Whether splits measure what they did for [`BisectReport`]. Off in normal
/// builds, so the clocks and extra cost passes cost nothing there.
const REPORT: bool = cfg!(any(test, feature = "detailed-tracing"));

/// What one bisection did, by depth. Only for spans and tests: nothing here
/// changes the order.
#[derive(Debug, Default, Clone)]
pub(crate) struct BisectReport {
    /// The gain per document the convergence test measures rounds against.
    /// Zero when no offering level lowered the cost.
    pub(crate) reference: f64,
    /// Threads in the pool the bisection ran on.
    pub(crate) pool_threads: usize,
    /// Other bisections running in this process when this one started and
    /// when it ended. They share the pool with it.
    pub(crate) others_at_start: usize,
    pub(crate) others_at_end: usize,
    /// `levels[depth]` sums every split made at that depth.
    pub(crate) levels: Vec<LevelReport>,
}

impl BisectReport {
    /// All depths added together.
    pub(crate) fn total(&self) -> LevelReport {
        let mut total = LevelReport::default();
        for level in &self.levels {
            total.add(level);
        }
        total
    }
}

impl fmt::Display for BisectReport {
    /// One line per depth.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (depth, level) in self.levels.iter().enumerate() {
            if level.splits > 0 || level.localized > 0 {
                writeln!(f, "d{depth} {level}")?;
            }
        }
        Ok(())
    }
}

/// The splits made at one depth, added together.
#[derive(Debug, Default, Clone)]
pub(crate) struct LevelReport {
    pub(crate) splits: u64,
    /// Documents across those splits.
    pub(crate) docs: u64,
    pub(crate) rounds: u64,
    /// Document pairs swapped across the split.
    pub(crate) swaps: u64,
    /// Splits that ended on a round that moved nothing.
    pub(crate) settled: u64,
    /// Splits the convergence test ended.
    pub(crate) converged: u64,
    /// Splits still moving documents when they hit the round ceiling.
    pub(crate) capped: u64,
    /// Sum of the gains the rounds chose their swaps by, taken before any
    /// swap is applied.
    pub(crate) predicted_gain: f64,
    /// How much the cost really fell over those rounds.
    pub(crate) actual_gain: f64,
    /// Rounds that moved documents but did not lower the cost.
    pub(crate) wasted_rounds: u64,
    /// Partitions whose terms were renumbered here.
    pub(crate) localized: u64,
    pub(crate) localize_ns: u64,
    /// Time in the splits, added across threads.
    pub(crate) split_ns: u64,
    /// Splits large enough to run in parallel, and their time per step.
    pub(crate) parallel_splits: u64,
    pub(crate) count_ns: u64,
    pub(crate) gains_ns: u64,
    pub(crate) rank_ns: u64,
    pub(crate) swap_ns: u64,
    /// Thread CPU time in the count, gains and swap steps, which run on one
    /// thread. Far below their wall time means the thread was waiting for a
    /// core. Zero where there is no thread CPU clock.
    pub(crate) serial_cpu_ns: u64,
    /// Time a parallel split waited in `join` after both its halves were done,
    /// running other work it had picked up from the pool.
    pub(crate) held_ns: u64,
}

impl LevelReport {
    /// Count what a round that moved documents really lowered the cost by.
    fn record_actual(&mut self, gain: f64) {
        self.actual_gain += gain;
        if gain <= 0.0 {
            self.wasted_rounds += 1;
        }
    }

    fn add(&mut self, other: &LevelReport) {
        self.splits += other.splits;
        self.docs += other.docs;
        self.rounds += other.rounds;
        self.swaps += other.swaps;
        self.settled += other.settled;
        self.converged += other.converged;
        self.capped += other.capped;
        self.predicted_gain += other.predicted_gain;
        self.actual_gain += other.actual_gain;
        self.wasted_rounds += other.wasted_rounds;
        self.localized += other.localized;
        self.localize_ns += other.localize_ns;
        self.split_ns += other.split_ns;
        self.parallel_splits += other.parallel_splits;
        self.count_ns += other.count_ns;
        self.gains_ns += other.gains_ns;
        self.rank_ns += other.rank_ns;
        self.swap_ns += other.swap_ns;
        self.serial_cpu_ns += other.serial_cpu_ns;
        self.held_ns += other.held_ns;
    }
}

impl fmt::Display for LevelReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = |ns: u64| ns / 1_000_000;
        write!(
            f,
            "splits={} docs={} rounds={} swaps={} settled={} converged={} capped={} \
             predicted={:.0} actual={:.0} wasted_rounds={} localized={} localize_ms={} \
             split_ms={}",
            self.splits,
            self.docs,
            self.rounds,
            self.swaps,
            self.settled,
            self.converged,
            self.capped,
            self.predicted_gain,
            self.actual_gain,
            self.wasted_rounds,
            self.localized,
            ms(self.localize_ns),
            ms(self.split_ns),
        )?;
        if self.parallel_splits > 0 {
            write!(
                f,
                " parallel_splits={} count_ms={} gains_ms={} rank_ms={} swap_ms={} \
                 serial_cpu_ms={} held_ms={}",
                self.parallel_splits,
                ms(self.count_ns),
                ms(self.gains_ns),
                ms(self.rank_ns),
                ms(self.swap_ns),
                ms(self.serial_cpu_ns),
                ms(self.held_ns),
            )?;
        }
        Ok(())
    }
}

/// Collects the splits' reports while a bisection runs.
#[derive(Debug)]
struct RunReport {
    levels: Mutex<Vec<LevelReport>>,
    /// Parent for the split spans, since rayon tasks lose the current span.
    span: Span,
}

impl RunReport {
    fn new() -> Self {
        Self {
            levels: Mutex::new(Vec::new()),
            span: Span::current(),
        }
    }

    fn add(&self, depth: u32, level: &LevelReport) {
        let mut levels = self.levels.lock().unwrap_or_else(PoisonError::into_inner);
        let depth = depth as usize;
        if levels.len() <= depth {
            levels.resize(depth + 1, LevelReport::default());
        }
        levels[depth].add(level);
    }

    /// Add a state's reports, indexed by depth, under one lock.
    fn add_all(&self, levels: &[LevelReport]) {
        let mut all = self.levels.lock().unwrap_or_else(PoisonError::into_inner);
        if all.len() < levels.len() {
            all.resize(levels.len(), LevelReport::default());
        }
        for (into, level) in all.iter_mut().zip(levels) {
            into.add(level);
        }
    }

    fn finish(self, reference: f64) -> BisectReport {
        BisectReport {
            reference,
            pool_threads: current_num_threads(),
            others_at_start: 0,
            others_at_end: 0,
            levels: self
                .levels
                .into_inner()
                .unwrap_or_else(PoisonError::into_inner),
        }
    }
}

/// How a split's rounds ended.
#[derive(Debug, Clone, Copy)]
enum SplitEnd {
    /// A round moved nothing.
    Settled,
    /// The convergence test stopped it.
    Converged,
    /// It ran out of rounds while still moving documents.
    Capped,
}

/// Nanoseconds since `start`.
fn ns_since(start: Instant) -> u64 {
    start.elapsed().as_nanos() as u64
}

/// Recursion depth cap, so a pathological corpus cannot drive the
/// splitting arbitrarily deep. At this depth a partition is `2^-24` of
/// the corpus, far below [`MIN_PARTITION`] for any real one.
const MAX_DEPTH: u32 = 24;

/// Degrees below this read their cost from a table filled once per
/// split instead of calling [`term_cost`]. Nearly every degree is
/// below it, and at 256 KiB per side the tables stay in cache. Larger
/// degrees call [`term_cost`] directly, so the values are the same.
const COST_TABLE_LEN: usize = 1 << 16;

/// How many ranked documents per side a round orders before it starts
/// swapping. A round only reads down its two lists until a pair stops paying
/// for itself, and after the first round or two that prefix is tiny, so
/// ordering the whole partition every round is almost all waste. The block
/// doubles when a round consumes it, which costs one extra linear selection
/// per doubling and never changes which swaps happen.
const SWAP_BLOCK: usize = 64;

/// Partitions below this size are split on one thread, with one scratch
/// state for their whole subtree. Above it the two halves recurse in
/// parallel, and a split's gains and sorts run in parallel too. Every
/// partition is split the same way on any thread, so the order does not
/// depend on how the work is spread.
const PARALLEL_MIN_PARTITION: usize = 1 << 14;

/// A document's terms, as ids into a shared vocabulary, laid out one
/// document after another with an index of where each begins.
///
/// Compressed-sparse-row rather than a `Vec<Vec<u32>>`: the bisection
/// walks a document's terms repeatedly and allocates nothing to do it,
/// and the whole structure is two allocations however many documents
/// there are.
#[derive(Debug, Default, Clone)]
pub(crate) struct ForwardIndex {
    /// Term ids, documents laid end to end.
    terms: Vec<u32>,
    /// `starts[d]..starts[d + 1]` is document `d`'s slice of `terms`.
    /// One longer than the document count.
    starts: Vec<u32>,
    /// One past the largest term id, sizing the degree tables.
    n_terms: usize,
}

impl ForwardIndex {
    /// Build from a per-document term-id iterator. Ids need not be
    /// sorted or unique within a document; duplicates only weight a
    /// term more heavily, which is harmless here.
    pub(crate) fn from_docs<I, T>(docs: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u32]>,
    {
        let mut terms = Vec::new();
        let mut starts = vec![0u32];
        let mut n_terms = 0usize;
        for doc in docs {
            for &t in doc.as_ref() {
                terms.push(t);
                n_terms = n_terms.max(t as usize + 1);
            }
            starts.push(terms.len() as u32);
        }
        Self {
            terms,
            starts,
            n_terms,
        }
    }

    /// Documents in the index.
    pub(crate) fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    #[inline]
    fn doc(&self, d: u32) -> &[u32] {
        let lo = self.starts[d as usize] as usize;
        let hi = self.starts[d as usize + 1] as usize;
        &self.terms[lo..hi]
    }
}

/// The order to store documents in: `order[new_id]` is the index of the
/// document that should take `new_id`.
///
/// The identity when there is nothing to gain, which is a corpus too
/// small to split or one whose documents share no terms.
#[cfg(test)]
pub(crate) fn bisect_order(fwd: &ForwardIndex, params: BisectParams) -> Vec<u32> {
    bisect_order_reported(fwd, params).0
}

/// The order to store documents in, as `bisect_order` returns it, plus what
/// the bisection did to get there.
pub(crate) fn bisect_order_reported(
    fwd: &ForwardIndex,
    params: BisectParams,
) -> (Vec<u32>, BisectReport) {
    let others_at_start = RUNNING.fetch_add(1, Relaxed);
    let n = fwd.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    let report = RunReport::new();
    let reference = AtomicU64::new(0);
    if n > MIN_PARTITION {
        let pool = StatePool {
            n_terms: fwd.n_terms,
            free: Mutex::new(Vec::new()),
        };
        let run = Bisect {
            params,
            reference: &reference,
            report: &report,
            parent: &report.span,
        };
        split_levels(fwd, &mut order, &pool, run);
    }
    let reference = f64::from_bits(reference.load(Relaxed));
    let mut report = report.finish(reference);
    report.others_at_start = others_at_start;
    report.others_at_end = RUNNING.fetch_sub(1, Relaxed) - 1;
    (order, report)
}

/// Split the top of the tree a level at a time, then hand each partition to
/// [`split_parallel`].
///
/// While a level holds fewer partitions than the pool has threads, or no
/// level has yet set the run's bar, its splits run side by side and each
/// spreads its rounds across threads too. No subtree is queued until the
/// whole level is done, so a round waiting on its own parallel pieces can
/// only pick up another split of the same level, never a subtree many times
/// its size; and a completed level offers its realised gain as the bar in
/// partition order, the same on every run. Every partition goes through
/// exactly the steps [`split_parallel`] would take it through, so the order
/// is the same.
fn split_levels(fwd: &ForwardIndex, order: &mut [u32], pool: &StatePool, run: Bisect<'_>) {
    let width = rayon::current_num_threads();
    // Each partition travels with its distinct terms once a refine has
    // counted them; nothing has counted the root's.
    let mut level: Vec<(&mut [u32], Option<usize>)> = vec![(order, None)];
    let mut handed_on: Vec<(&mut [u32], u32, Option<usize>)> = Vec::new();
    let mut depth = 0u32;
    // A level at a time while the level is narrower than the pool, and for
    // as long as no level has set the bar: each completed level offers its
    // realised gain in partition order, so a run whose root moved nothing
    // still takes its bar from the first level that lowered the cost, and
    // takes the same bar on every run.
    while !level.is_empty() && (level.len() < width || !run.has_reference()) {
        // Only a partition `split_parallel` would refine and halve itself is
        // split here; any other takes its own path there. A small partition
        // is halved here too while no bar is set — the same serial refine
        // `BisectState::split` would give it — so the levels keep offering until
        // one lowers the cost.
        let wide = run.has_reference();
        let (here, rest): (Vec<_>, Vec<_>) = level.into_iter().partition(|(part, distinct)| {
            part.len() > MIN_PARTITION
                && depth < MAX_DEPTH
                && (part.len() >= PARALLEL_MIN_PARTITION || !wide)
                && !localizing_pays(fwd, *distinct)
        });
        handed_on.extend(
            rest.into_iter()
                .map(|(part, distinct)| (part, depth, distinct)),
        );
        let docs: usize = here.iter().map(|(part, _)| part.len()).sum();
        let split: Vec<(Refined, [&mut [u32]; 2])> = here
            .into_par_iter()
            .map(|(part, _)| {
                let mid = part.len() / 2;
                let parallel = part.len() >= PARALLEL_MIN_PARTITION;
                let refined = pool.with_state(|state| {
                    let refined = state.refine(fwd, part, mid, depth, parallel, run);
                    state.flush(run.report);
                    refined
                });
                let (left, right) = part.split_at_mut(mid);
                (refined, [left, right])
            })
            .collect();
        // The level's splits ran side by side, but their first rounds' gains
        // are summed here in partition order, so the bar they offer is the
        // same on every run.
        run.offer_reference(
            split
                .iter()
                .map(|(refined, _)| refined.first_round_gain)
                .sum(),
            docs,
        );
        level = split
            .into_iter()
            .flat_map(|(refined, halves)| halves.into_iter().zip(refined.distinct.map(Some)))
            .collect();
        depth += 1;
    }
    handed_on.extend(
        level
            .into_iter()
            .map(|(part, distinct)| (part, depth, distinct)),
    );
    handed_on
        .into_par_iter()
        .for_each(|(part, depth, distinct)| split_parallel(fwd, part, depth, pool, run, distinct));
}

/// Split `order` as [`BisectState::split`] does, running the two halves
/// of a large partition in parallel. `distinct` is the partition's distinct
/// terms, where its parent's refine counted them.
fn split_parallel(
    fwd: &ForwardIndex,
    order: &mut [u32],
    depth: u32,
    pool: &StatePool,
    run: Bisect<'_>,
    distinct: Option<usize>,
) {
    if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
        return;
    }
    if localizing_pays(fwd, distinct) {
        let started = REPORT.then(Instant::now);
        let local = pool.with_state(|state| state.localize(fwd, order));
        if let Some(started) = started {
            run.report.add(
                depth,
                &LevelReport {
                    localized: 1,
                    localize_ns: ns_since(started),
                    ..LevelReport::default()
                },
            );
        }
        let mut inner: Vec<u32> = (0..order.len() as u32).collect();
        let inner_pool = StatePool {
            n_terms: local.n_terms,
            free: Mutex::new(Vec::new()),
        };
        // Its terms fill the new table exactly, so it does not renumber
        // again before it splits.
        split_parallel(&local, &mut inner, depth, &inner_pool, run, None);
        apply_inner_order(order, &inner);
        return;
    }
    if order.len() < PARALLEL_MIN_PARTITION {
        pool.with_state(|state| {
            state.split(fwd, order, depth, run, distinct);
            state.flush(run.report);
        });
        return;
    }
    let span = match order.len() >= PARALLEL_SPAN_MIN_DOCS {
        true => detail_span!(
            "reorder_split_parallel",
            parent: run.parent,
            depth = depth,
            docs = order.len(),
            worker = current_thread_index(),
        ),
        false => Span::none(),
    };
    let run = match span.is_none() {
        true => run,
        false => Bisect {
            parent: &span,
            ..run
        },
    };
    let mid = order.len() / 2;
    // Only the root's rounds run their gains and sorts in parallel. Below it
    // the subtrees are already running side by side, and a round waiting on its
    // own parallel pieces would steal a whole queued subtree and finish it
    // before resuming, which put one split's ranking at 46 s instead of 0.3 s.
    // The root's first round offers the bar from inside `refine`.
    let [left_distinct, right_distinct] = pool.with_state(|state| {
        let refined = state.refine(fwd, order, mid, depth, depth == 0, run);
        state.flush(run.report);
        refined.distinct
    });
    let (left, right) = order.split_at_mut(mid);
    let (left_done, right_done) = join(
        || {
            split_parallel(fwd, left, depth + 1, pool, run, Some(left_distinct));
            REPORT.then(Instant::now)
        },
        || {
            split_parallel(fwd, right, depth + 1, pool, run, Some(right_distinct));
            REPORT.then(Instant::now)
        },
    );
    if let (Some(left_done), Some(right_done)) = (left_done, right_done) {
        run.report.add(
            depth,
            &LevelReport {
                held_ns: ns_since(left_done.max(right_done)),
                ..LevelReport::default()
            },
        );
    }
}

/// Whether renumbering this partition's terms would pay for itself.
///
/// The degree and move-gain tables are sized by the vocabulary they are
/// indexed against, and every round of every split below reads them at
/// random. Once the partition's own distinct terms fill no more than half the
/// current table, renumbering them into a dense range at least halves it, and
/// one pass over the partition buys every split below a table that sits closer
/// to cache.
///
/// The test is on the terms the partition actually carries, which its
/// parent's refine has already counted, so it costs nothing to make. A test on
/// postings instead, which bound the distinct terms from above, waits far
/// longer than it needs to: a few common terms make up most postings, so the
/// partitions in the middle of the recursion keep running against tables
/// sized for a much larger one. Halving at minimum also bounds how often it
/// fires on any root-to-leaf path, and nothing here depends on the corpus: it
/// is the relationship between a partition and its vocabulary that decides
/// whether the tables are oversized.
///
/// `distinct` is `None` where no refine has counted the partition: the root,
/// and a partition just renumbered, whose terms fill its new table and so
/// could never shrink it by half anyway.
fn localizing_pays(fwd: &ForwardIndex, distinct: Option<usize>) -> bool {
    // An empty table cannot shrink.
    fwd.n_terms > 0 && distinct.is_some_and(|distinct| distinct.saturating_mul(2) <= fwd.n_terms)
}

/// Apply a permutation computed over a partition's own positions back onto
/// the caller's slice.
fn apply_inner_order(order: &mut [u32], inner: &[u32]) {
    let was: Vec<u32> = order.to_vec();
    for (slot, &i) in order.iter_mut().zip(inner.iter()) {
        *slot = was[i as usize];
    }
}

/// What a refine leaves for the splits below it.
struct Refined {
    /// What its first round lowered the cost by: its offer towards the run's
    /// bar.
    first_round_gain: f64,
    /// The distinct terms each half carries, left then right.
    distinct: [usize; 2],
}

/// Scratch states shared by the parallel splits. A split takes one and
/// puts it back clean, so only as many exist as ever ran at once.
struct StatePool {
    n_terms: usize,
    free: Mutex<Vec<BisectState>>,
}

impl StatePool {
    fn with_state<R>(&self, f: impl FnOnce(&mut BisectState) -> R) -> R {
        // A split never panics while holding the lock, so a poisoned
        // list is still a valid one.
        let taken = self
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let mut state = taken.unwrap_or_else(|| BisectState::new(self.n_terms));
        let out = f(&mut state);
        self.free
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(state);
        out
    }
}

/// Scratch reused down the whole recursion, so a split allocates
/// nothing beyond what the first one needed.
struct BisectState {
    deg_left: Vec<u32>,
    deg_right: Vec<u32>,
    /// What moving any one left document lowers a term's cost by. It
    /// depends only on the term's degrees, so it is computed once per
    /// term per round rather than once per document carrying it.
    move_gain_left: Vec<f32>,
    /// The same for a document moving from the right.
    move_gain_right: Vec<f32>,
    /// `term_cost(d, left size)` for the current split, by `d`.
    cost_left: Vec<f32>,
    /// `term_cost(d, right size)` for the current split, by `d`.
    cost_right: Vec<f32>,
    /// Term ids whose degree entries a split touched, so the tables can
    /// be cleared in proportion to the partition rather than to the
    /// vocabulary.
    touched: Vec<u32>,
    /// Ranked documents for each side as [`rank_key`]s, reused by every round of every
    /// split this state serves. Ranking used to return a fresh `Vec` per
    /// side per round, which is two allocations per round per partition —
    /// millions of them over a corpus-sized bisection.
    left_gains: Vec<u64>,
    right_gains: Vec<u64>,
    /// Reports of the splits this state made, by depth, until
    /// [`Self::flush`] adds them to the run's. Saves taking the shared lock
    /// once per split.
    levels: Vec<LevelReport>,
}

impl BisectState {
    fn new(n_terms: usize) -> Self {
        Self {
            deg_left: vec![0u32; n_terms],
            deg_right: vec![0u32; n_terms],
            move_gain_left: vec![0f32; n_terms],
            move_gain_right: vec![0f32; n_terms],
            cost_left: Vec::new(),
            cost_right: Vec::new(),
            touched: Vec::new(),
            left_gains: Vec::new(),
            right_gains: Vec::new(),
            levels: Vec::new(),
        }
    }

    fn note(&mut self, depth: u32, level: &LevelReport) {
        let depth = depth as usize;
        if self.levels.len() <= depth {
            self.levels.resize(depth + 1, LevelReport::default());
        }
        self.levels[depth].add(level);
    }

    fn flush(&mut self, report: &RunReport) {
        if self.levels.is_empty() {
            return;
        }
        report.add_all(&self.levels);
        self.levels.clear();
    }

    /// Renumber the terms this partition carries into a dense range and
    /// return the partition as its own index, documents numbered by position.
    ///
    /// The degree and move-gain tables are indexed by term id, so against a
    /// shared vocabulary they stay as wide as the whole corpus however few
    /// documents a partition holds — megabytes walked at random by every
    /// round, missing cache on nearly every lookup. One pass over the
    /// partition's postings buys tables sized to the terms actually present,
    /// and every split below inherits them.
    ///
    /// Renumbering is a bijection, so every degree, gain and comparison is
    /// unchanged, and the terms within a document keep their order, so the
    /// gain sums add in the same sequence and to the same float.
    fn localize(&mut self, fwd: &ForwardIndex, order: &[u32]) -> ForwardIndex {
        // Outside a refine `deg_left` is all zeros — `clear_degrees` leaves it
        // that way and this runs in a refine's place — so it serves as the
        // global-to-local term map without a second table that size. Zero
        // means "not seen here"; a local id is held as `id + 1`.
        let mut terms: Vec<u32> = Vec::with_capacity(order.len());
        let mut starts: Vec<u32> = Vec::with_capacity(order.len() + 1);
        starts.push(0);
        let mut n_local = 0u32;
        for &d in order.iter() {
            for &t in fwd.doc(d) {
                let slot = &mut self.deg_left[t as usize];
                if *slot == 0 {
                    n_local += 1;
                    *slot = n_local;
                    self.touched.push(t);
                }
                terms.push(*slot - 1);
            }
            starts.push(terms.len() as u32);
        }
        for &t in &self.touched {
            self.deg_left[t as usize] = 0;
        }
        self.touched.clear();
        ForwardIndex {
            terms,
            starts,
            n_terms: n_local as usize,
        }
    }

    /// Split `order` and its halves down to the leaves. `distinct` is the
    /// partition's distinct terms, where its parent's refine counted them.
    fn split(
        &mut self,
        fwd: &ForwardIndex,
        order: &mut [u32],
        depth: u32,
        run: Bisect<'_>,
        distinct: Option<usize>,
    ) {
        if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
            return;
        }
        if localizing_pays(fwd, distinct) {
            let started = REPORT.then(Instant::now);
            let local = self.localize(fwd, order);
            if let Some(started) = started {
                self.note(
                    depth,
                    &LevelReport {
                        localized: 1,
                        localize_ns: ns_since(started),
                        ..LevelReport::default()
                    },
                );
            }
            let mut inner: Vec<u32> = (0..order.len() as u32).collect();
            let mut inner_state = BisectState::new(local.n_terms);
            // Its terms fill the new table exactly, so it does not renumber
            // again before it splits.
            inner_state.split(&local, &mut inner, depth, run, None);
            inner_state.flush(run.report);
            apply_inner_order(order, &inner);
            return;
        }
        let mid = order.len() / 2;
        // The root's first round offers the bar from inside `refine`.
        let [left_distinct, right_distinct] =
            self.refine(fwd, order, mid, depth, false, run).distinct;
        let (left, right) = order.split_at_mut(mid);
        self.split(fwd, left, depth + 1, run, Some(left_distinct));
        self.split(fwd, right, depth + 1, run, Some(right_distinct));
    }

    /// Move documents across the split while it lowers the cost, then
    /// leave the two halves in `order`. `parallel` spreads each round's
    /// gains and sorts across threads.
    fn refine(
        &mut self,
        fwd: &ForwardIndex,
        order: &mut [u32],
        mid: usize,
        depth: u32,
        parallel: bool,
        run: Bisect<'_>,
    ) -> Refined {
        let started = REPORT.then(Instant::now);
        let span = match order.len() >= SPLIT_SPAN_MIN_DOCS {
            true => detail_span!(
                "reorder_split",
                parent: run.parent,
                depth = depth,
                docs = order.len(),
                rounds = tracing::field::Empty,
                swaps = tracing::field::Empty,
                end = tracing::field::Empty,
                predicted_gain = tracing::field::Empty,
                actual_gain = tracing::field::Empty,
                count_ms = tracing::field::Empty,
                gains_ms = tracing::field::Empty,
                rank_ms = tracing::field::Empty,
                swap_ms = tracing::field::Empty,
                serial_cpu_ms = tracing::field::Empty,
            ),
            false => Span::none(),
        };
        let _entered = span.enter();
        // Per-step clocks only on the large splits, where they cost nothing
        // next to the step.
        let timed = parallel && REPORT;
        let mut level = LevelReport {
            splits: 1,
            docs: order.len() as u64,
            parallel_splits: u64::from(timed),
            ..LevelReport::default()
        };
        let mut end = SplitEnd::Capped;

        let step = timed.then(Instant::now);
        let cpu = timed.then(thread_cpu_ns).flatten();
        self.count_degrees(fwd, order, mid, parallel);
        if let Some(step) = step {
            level.count_ns += ns_since(step);
            level.serial_cpu_ns += thread_cpu_delta_ns(cpu);
        }
        let n_left = mid as f32;
        let n_right = (order.len() - mid) as f32;
        fill_cost_table(&mut self.cost_left, n_left);
        fill_cost_table(&mut self.cost_right, n_right);
        let mut cost_before = self.partition_cost(n_left, n_right, parallel);
        // What the split's first round lowered the cost by: its offer
        // towards the run's bar, which measures every later round against
        // what a first round over fresh halves is worth per document.
        let mut first_round_gain = 0.0f64;

        // Taken out of `self` so the ranking can borrow the move-gain
        // tables while writing them, and put back before returning so the
        // next split reuses the same allocations.
        let mut left_gains = mem::take(&mut self.left_gains);
        let mut right_gains = mem::take(&mut self.right_gains);
        let mut last_swaps = 0usize;
        // A round is judged by its realised gain: the cost before it less
        // the cost after, read off the degree tables the swaps just moved.
        // The pair gains the swap loop adds up predict that only while the
        // moves are few and far apart; where a partition is small the pairs
        // moved in one round share terms, and their summed gains run to
        // many times what the round actually buys, which kept the leaf
        // splits rounding long after they had stopped paying. The reading
        // costs a pass over the partition's terms, a small fraction of a
        // round's pass over its postings in a large partition and about
        // one round's worth in a leaf. A round that raised the cost is kept
        // as it is — its swaps stay — and ends the split like any round
        // under the bar.
        for round in 0..run.params.max_rounds {
            // A document's gain is what the cost drops by if it moves:
            // its terms get one rarer on this side and one commoner on
            // the other. Positive means the move is worth making.
            let moved = {
                let step = timed.then(Instant::now);
                let cpu = timed.then(thread_cpu_ns).flatten();
                self.compute_move_gains(n_left, n_right, parallel);
                if let Some(step) = step {
                    level.gains_ns += ns_since(step);
                    level.serial_cpu_ns += thread_cpu_delta_ns(cpu);
                }
                let step = timed.then(Instant::now);
                let (left, right) = order.split_at_mut(mid);
                rank_by_gain(fwd, left, &self.move_gain_left, parallel, &mut left_gains);
                rank_by_gain(
                    fwd,
                    right,
                    &self.move_gain_right,
                    parallel,
                    &mut right_gains,
                );

                // Swap in pairs so the halves keep their sizes. Both
                // lists are ordered by gain, so once a pair does not pay
                // for itself no later pair can either — which is why only
                // the leading `ready` entries need to be in order, and why
                // extending the block cannot change the outcome.
                let pairs = left_gains.len().min(right_gains.len());
                let mut ready = last_swaps.saturating_mul(2).max(SWAP_BLOCK).min(pairs);
                order_leading_gains(&mut left_gains, ready, parallel);
                order_leading_gains(&mut right_gains, ready, parallel);
                if let Some(step) = step {
                    level.rank_ns += ns_since(step);
                }

                let step = timed.then(Instant::now);
                let cpu = timed.then(thread_cpu_ns).flatten();
                let mut swaps = 0usize;
                let mut gain = 0.0f64;
                while swaps < pairs {
                    if swaps == ready {
                        let next = (ready * 2).min(pairs);
                        extend_ordered_prefix(&mut left_gains, ready, next, parallel);
                        extend_ordered_prefix(&mut right_gains, ready, next, parallel);
                        ready = next;
                    }
                    let (l, r) = (left_gains[swaps], right_gains[swaps]);
                    let pair_gain = key_gain(l) + key_gain(r);
                    if pair_gain <= 0.0 {
                        break;
                    }
                    swaps += 1;
                    gain += f64::from(pair_gain);
                    if parallel {
                        // Applied below, once the round knows its swaps.
                        continue;
                    }
                    let (li, ri) = (key_pos(l) as usize, key_pos(r) as usize);
                    let (ld, rd) = (left[li], right[ri]);
                    // The degree tables follow the documents across.
                    for &t in fwd.doc(ld) {
                        self.deg_left[t as usize] -= 1;
                        self.deg_right[t as usize] += 1;
                    }
                    for &t in fwd.doc(rd) {
                        self.deg_right[t as usize] -= 1;
                        self.deg_left[t as usize] += 1;
                    }
                    left[li] = rd;
                    right[ri] = ld;
                }
                if parallel && swaps > 0 {
                    // Which pairs swap depends only on the ranked gains, and
                    // every pair names its own two positions, so the swaps can
                    // land in any order. Degrees are counts, so adding them up
                    // in parallel ends at the same tables.
                    let (deg_left, deg_right) = (
                        as_atomic(&mut self.deg_left),
                        as_atomic(&mut self.deg_right),
                    );
                    let (lefts, rights) = (&left_gains[..swaps], &right_gains[..swaps]);
                    let (left_docs, right_docs) = (&*left, &*right);
                    lefts
                        .par_iter()
                        .zip(rights)
                        .with_min_len(PAR_MIN_ITEMS)
                        .for_each(|(&l, &r)| {
                            for &t in fwd.doc(left_docs[key_pos(l) as usize]) {
                                deg_left[t as usize].fetch_sub(1, Relaxed);
                                deg_right[t as usize].fetch_add(1, Relaxed);
                            }
                            for &t in fwd.doc(right_docs[key_pos(r) as usize]) {
                                deg_right[t as usize].fetch_sub(1, Relaxed);
                                deg_left[t as usize].fetch_add(1, Relaxed);
                            }
                        });
                    for (&l, &r) in lefts.iter().zip(rights) {
                        let (li, ri) = (key_pos(l) as usize, key_pos(r) as usize);
                        let (ld, rd) = (left[li], right[ri]);
                        left[li] = rd;
                        right[ri] = ld;
                    }
                }
                if let Some(step) = step {
                    level.swap_ns += ns_since(step);
                    level.serial_cpu_ns += thread_cpu_delta_ns(cpu);
                }
                if REPORT {
                    level.predicted_gain += gain;
                }
                swaps
            };
            last_swaps = moved;
            level.rounds += 1;
            if moved == 0 {
                end = SplitEnd::Settled;
                break;
            }
            level.swaps += moved as u64;
            let cost_after = self.partition_cost(n_left, n_right, parallel);
            let round_gain = cost_before - cost_after;
            cost_before = cost_after;
            if round == 0 {
                first_round_gain = round_gain;
                // The root offers at once, so its own later rounds are
                // measured against it too; the levels offer again below
                // for a root whose first round lowered nothing.
                if depth == 0 {
                    run.offer_reference(round_gain, order.len());
                }
            }
            if REPORT {
                level.record_actual(round_gain);
            }
            if round > 0 && run.converged(round_gain, order.len()) {
                // This round bought less per document than the corpus showed
                // was available, so the rounds still to come are not worth
                // their pass over the partition — and the work they would take
                // is better spent on a split that is still paying.
                end = SplitEnd::Converged;
                break;
            }
        }

        self.left_gains = left_gains;
        self.right_gains = right_gains;
        let refined = Refined {
            first_round_gain,
            distinct: self.distinct_by_side(),
        };
        self.clear_degrees();

        let Some(started) = started else {
            return refined;
        };
        match end {
            SplitEnd::Settled => level.settled = 1,
            SplitEnd::Converged => level.converged = 1,
            SplitEnd::Capped => level.capped = 1,
        }
        level.split_ns = ns_since(started);
        span.record("rounds", level.rounds);
        span.record("swaps", level.swaps);
        span.record("end", tracing::field::debug(end));
        span.record("predicted_gain", level.predicted_gain);
        span.record("actual_gain", level.actual_gain);
        span.record("count_ms", level.count_ns / 1_000_000);
        span.record("gains_ms", level.gains_ns / 1_000_000);
        span.record("rank_ms", level.rank_ns / 1_000_000);
        span.record("swap_ms", level.swap_ns / 1_000_000);
        span.record("serial_cpu_ms", level.serial_cpu_ns / 1_000_000);
        self.note(depth, &level);
        refined
    }

    /// The distinct terms on each side of the split, read off its degree
    /// tables: a pass over the terms the partition touches, not its postings.
    fn distinct_by_side(&self) -> [usize; 2] {
        self.touched.iter().fold([0, 0], |[left, right], &t| {
            [
                left + usize::from(self.deg_left[t as usize] > 0),
                right + usize::from(self.deg_right[t as usize] > 0),
            ]
        })
    }

    /// The cost of the current split, from its degree tables. A large split
    /// sums in parallel, a fixed chunk of terms at a time, each chunk in
    /// order and then the chunks in order, so the total is the same on any
    /// thread count.
    fn partition_cost(&self, n_left: f32, n_right: f32, parallel: bool) -> f64 {
        let cost_of = |terms: &[u32]| -> f64 {
            terms
                .iter()
                .map(|&t| self.term_split_cost(t as usize, n_left, n_right))
                .sum()
        };
        if !parallel {
            return cost_of(&self.touched);
        }
        let chunks: Vec<f64> = self
            .touched
            .par_chunks(PAR_MIN_ITEMS)
            .map(cost_of)
            .collect();
        chunks.into_iter().sum()
    }

    /// What term `t` costs across both halves of the current split.
    #[inline]
    fn term_split_cost(&self, t: usize, n_left: f32, n_right: f32) -> f64 {
        let left = Side {
            deg: self.deg_left[t],
            size: n_left,
            costs: &self.cost_left,
        };
        let right = Side {
            deg: self.deg_right[t],
            size: n_right,
            costs: &self.cost_right,
        };
        f64::from(left.cost(left.deg)) + f64::from(right.cost(right.deg))
    }

    /// Fill the move gains of every term in the partition from its
    /// current degrees. A side's gain is only read by documents on that
    /// side carrying the term, so a side with no such document is skipped.
    fn compute_move_gains(&mut self, n_left: f32, n_right: f32, parallel: bool) {
        if parallel {
            // Each term's gains read only its own degrees and write only its
            // own entries, so the terms can be done in any order.
            let Self {
                deg_left,
                deg_right,
                move_gain_left,
                move_gain_right,
                cost_left,
                cost_right,
                touched,
                ..
            } = self;
            let (gain_left, gain_right) = (as_atomic(move_gain_left), as_atomic(move_gain_right));
            let (deg_left, deg_right) = (&*deg_left, &*deg_right);
            let (cost_left, cost_right) = (cost_left.as_slice(), cost_right.as_slice());
            touched
                .par_iter()
                .with_min_len(PAR_MIN_ITEMS)
                .for_each(|&t| {
                    let t = t as usize;
                    let (dl, dr) = (deg_left[t], deg_right[t]);
                    let left = Side {
                        deg: dl,
                        size: n_left,
                        costs: cost_left,
                    };
                    let right = Side {
                        deg: dr,
                        size: n_right,
                        costs: cost_right,
                    };
                    if dl > 0 {
                        gain_left[t].store(move_gain(&left, &right).to_bits(), Relaxed);
                    }
                    if dr > 0 {
                        gain_right[t].store(move_gain(&right, &left).to_bits(), Relaxed);
                    }
                });
            return;
        }
        for &t in &self.touched {
            let t = t as usize;
            let (dl, dr) = (self.deg_left[t], self.deg_right[t]);
            let left = Side {
                deg: dl,
                size: n_left,
                costs: &self.cost_left,
            };
            let right = Side {
                deg: dr,
                size: n_right,
                costs: &self.cost_right,
            };
            if dl > 0 {
                self.move_gain_left[t] = move_gain(&left, &right);
            }
            if dr > 0 {
                self.move_gain_right[t] = move_gain(&right, &left);
            }
        }
    }

    /// Count each term's documents on either side of `mid` into the degree
    /// tables, and list the terms the partition touches.
    ///
    /// A large split counts in parallel. Degrees are counts, so adding them
    /// up from any thread ends at the same tables; the touched terms are
    /// then read off the tables in id order, so every later pass over them,
    /// and the sums it makes, runs the same whichever threads counted. The
    /// serial path lists a term as it first meets it, which costs nothing
    /// beyond the count itself; a pass over the whole vocabulary would.
    fn count_degrees(&mut self, fwd: &ForwardIndex, order: &[u32], mid: usize, parallel: bool) {
        if parallel {
            let Self {
                deg_left,
                deg_right,
                touched,
                ..
            } = self;
            let (left, right) = order.split_at(mid);
            {
                let (deg_left, deg_right) = (as_atomic(deg_left), as_atomic(deg_right));
                join(
                    || {
                        left.par_iter().with_min_len(PAR_MIN_ITEMS).for_each(|&d| {
                            for &t in fwd.doc(d) {
                                deg_left[t as usize].fetch_add(1, Relaxed);
                            }
                        })
                    },
                    || {
                        right.par_iter().with_min_len(PAR_MIN_ITEMS).for_each(|&d| {
                            for &t in fwd.doc(d) {
                                deg_right[t as usize].fetch_add(1, Relaxed);
                            }
                        })
                    },
                );
            }
            let (deg_left, deg_right) = (&*deg_left, &*deg_right);
            touched.clear();
            touched.par_extend(
                (0..deg_left.len() as u32)
                    .into_par_iter()
                    .with_min_len(PAR_MIN_ITEMS)
                    .filter(|&t| deg_left[t as usize] > 0 || deg_right[t as usize] > 0),
            );
            return;
        }
        for (i, &d) in order.iter().enumerate() {
            let side_left = i < mid;
            for &t in fwd.doc(d) {
                // Seen on neither side yet: first sight in this partition.
                if self.deg_left[t as usize] == 0 && self.deg_right[t as usize] == 0 {
                    self.touched.push(t);
                }
                let slot = match side_left {
                    true => &mut self.deg_left[t as usize],
                    false => &mut self.deg_right[t as usize],
                };
                *slot += 1;
            }
        }
    }

    fn clear_degrees(&mut self) {
        for &t in &self.touched {
            self.deg_left[t as usize] = 0;
            self.deg_right[t as usize] = 0;
        }
        self.touched.clear();
    }
}

/// The [`rank_key`] of every document in `half`: the sum of its terms' move
/// gains for that side, with its position within the half, written into `out`.
///
/// Takes the output buffer rather than returning one so a split's rounds
/// share two allocations instead of making two apiece.
fn rank_by_gain(
    fwd: &ForwardIndex,
    half: &[u32],
    move_gain: &[f32],
    parallel: bool,
    out: &mut Vec<u64>,
) {
    // Keys for the documents of one stretch of `half` starting at `first`.
    let keys_of = |first: usize, docs: &[u32], keys: &mut Vec<u64>| {
        for (j, group) in docs.chunks(GAIN_LANES).enumerate() {
            let gains = lane_gains(fwd, group, move_gain);
            let at = first + j * GAIN_LANES;
            keys.extend(
                gains[..group.len()]
                    .iter()
                    .enumerate()
                    .map(|(k, &gain)| rank_key(gain, (at + k) as u32)),
            );
        }
    };
    out.clear();
    match parallel {
        // Stretches of whole lane groups, collected in order, so the
        // ranking is the same whichever way it ran.
        true => {
            let stretch = PAR_MIN_ITEMS.next_multiple_of(GAIN_LANES);
            let parts: Vec<Vec<u64>> = half
                .par_chunks(stretch)
                .enumerate()
                .map(|(c, docs)| {
                    let mut keys = Vec::with_capacity(docs.len());
                    keys_of(c * stretch, docs, &mut keys);
                    keys
                })
                .collect();
            out.reserve(half.len());
            for part in parts {
                out.extend(part);
            }
        }
        false => keys_of(0, half, out),
    }
}

/// The gain of each document in `group` (at most [`GAIN_LANES`] of them):
/// the sum of its terms' `move_gain`, added from zero in the order the
/// document lists its terms, as a document-at-a-time loop adds them. The
/// documents are summed side by side, one accumulator each, so the adds of
/// different documents do not wait on one another: summed one document at a
/// time, every add waited on the one before it, and the gather spent its
/// time in that chain rather than in its loads.
fn lane_gains(fwd: &ForwardIndex, group: &[u32], move_gain: &[f32]) -> [f32; GAIN_LANES] {
    let mut terms: [&[u32]; GAIN_LANES] = [&[]; GAIN_LANES];
    for (slot, &d) in terms.iter_mut().zip(group) {
        *slot = fwd.doc(d);
    }
    let longest = terms.iter().map(|t| t.len()).max().unwrap_or(0);
    let mut gains = [0.0f32; GAIN_LANES];
    for k in 0..longest {
        for (gain, doc) in gains.iter_mut().zip(&terms) {
            if let Some(&t) = doc.get(k) {
                *gain += move_gain[t as usize];
            }
        }
    }
    gains
}

/// Extend the ordered prefix of `gains` from its first `ordered` entries to
/// its first `ready`. The first `ordered` are already the smallest keys, in
/// order, and every other entry sits behind them, so ordering the next ones
/// from the rest alone ends at the order a pass over the whole list would.
fn extend_ordered_prefix(gains: &mut [u64], ordered: usize, ready: usize, parallel: bool) {
    order_leading_gains(&mut gains[ordered..], ready - ordered, parallel);
}

/// Put the `want` highest-gain entries at the front of `gains`, in the
/// order a full sort would have put them; leave the rest unordered.
///
/// Every key holds a unique position, so the keys are distinct and the
/// leading `want` entries are the same whatever `want` was asked for before.
/// That is what lets a round grow its block without changing which documents
/// it swaps, and what makes an unstable sort deterministic here.
fn order_leading_gains(gains: &mut [u64], want: usize, parallel: bool) {
    if want >= gains.len() {
        match parallel {
            true => gains.par_sort_unstable(),
            false => gains.sort_unstable(),
        }
        return;
    }
    if want == 0 {
        return;
    }
    // Linear-time partition around the `want`-th entry, then order only
    // what sits in front of it.
    gains.select_nth_unstable(want);
    let head = &mut gains[..want];
    match parallel {
        true => head.par_sort_unstable(),
        false => head.sort_unstable(),
    }
}

/// Sort key for a ranked document: gain descending, then position ascending,
/// as one ascending `u64`. Gains order as [`f32::total_cmp`] orders them.
fn rank_key(gain: f32, pos: u32) -> u64 {
    let bits = gain.to_bits();
    // Maps the bits to a `u32` that ascends with the gain.
    let ascending = match bits & SIGN_BIT != 0 {
        true => !bits,
        false => bits ^ SIGN_BIT,
    };
    (u64::from(!ascending) << 32) | u64::from(pos)
}

/// The gain a [`rank_key`] was made from, bit for bit.
fn key_gain(key: u64) -> f32 {
    let ascending = !((key >> 32) as u32);
    f32::from_bits(match ascending & SIGN_BIT != 0 {
        true => ascending ^ SIGN_BIT,
        false => !ascending,
    })
}

/// The position a [`rank_key`] was made from.
fn key_pos(key: u64) -> u32 {
    key as u32
}

/// A table of 32-bit words viewed as atomics, so threads that each update
/// their own entries, or add to shared counts, can share it.
fn as_atomic<T: Word32>(table: &mut [T]) -> &[AtomicU32] {
    const { assert!(size_of::<T>() == 4 && align_of::<T>() == align_of::<AtomicU32>()) };
    // SAFETY: `T` is a plain 32-bit word with `AtomicU32`'s size and alignment,
    // and the exclusive borrow keeps every other access out while the view
    // lives.
    unsafe { &*(table as *mut [T] as *const [AtomicU32]) }
}

/// The word types [`as_atomic`] views.
trait Word32: Copy {}
impl Word32 for u32 {}
impl Word32 for f32 {}

/// One half of a split, as a term's move gain sees it.
struct Side<'a> {
    /// Documents in this half carrying the term.
    deg: u32,
    /// Documents in this half.
    size: f32,
    /// This half's cost table.
    costs: &'a [f32],
}

impl Side<'_> {
    #[inline]
    fn cost(&self, deg: u32) -> f32 {
        match self.costs.get(deg as usize) {
            Some(&c) => c,
            None => term_cost(deg, self.size),
        }
    }
}

/// What the cost drops by when one document carrying a term moves from
/// `here` to `there`.
#[inline]
fn move_gain(here: &Side, there: &Side) -> f32 {
    here.cost(here.deg) - here.cost(here.deg.saturating_sub(1)) + there.cost(there.deg)
        - there.cost(there.deg + 1)
}

/// Fill `table` with `term_cost(d, size)` for every degree a half of
/// `size` documents can reach, up to [`COST_TABLE_LEN`].
fn fill_cost_table(table: &mut Vec<f32>, size: f32) {
    // A degree reaches at most `size + 1`: the term's documents on this
    // side plus the one moving in.
    let len = (size as usize + 2).min(COST_TABLE_LEN);
    table.clear();
    table.extend((0..len as u32).map(|d| term_cost(d, size)));
}

/// What a term carried by `deg` of a half's `size` documents
/// contributes to that half's cost. Zero when nothing carries it, which
/// is also what keeps the logarithm in range.
#[inline]
fn term_cost(deg: u32, size: f32) -> f32 {
    match deg {
        0 => 0.0,
        d => d as f32 * (size / (d as f32 + 1.0)).max(1.0).log2(),
    }
}

/// The total cost of an order, for tests and for anything that wants to
/// see whether reordering paid. Lower is better grouped.
#[cfg(test)]
pub(crate) fn order_cost(fwd: &ForwardIndex, order: &[u32], window: usize) -> f64 {
    use std::collections::HashMap;

    // Cost the order in windows, which is what a posting block is: the
    // fewer distinct terms a window holds, the tighter its postings.
    let mut total = 0.0f64;
    for chunk in order.chunks(window.max(1)) {
        let mut deg: HashMap<u32, u32> = HashMap::new();
        for &d in chunk {
            for &t in fwd.doc(d) {
                *deg.entry(t).or_default() += 1;
            }
        }
        for (_, d) in deg {
            total += f64::from(term_cost(d, chunk.len() as f32));
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rayon::ThreadPoolBuilder;

    use super::*;

    /// Documents in the lane-gain test: several whole lane groups and a
    /// ragged last one; and enough for the parallel ranking to cut several
    /// stretches, the last one ragged too.
    const LANE_TEST_DOCS: [usize; 2] = [8 * GAIN_LANES + 3, 2 * PAR_MIN_ITEMS + GAIN_LANES + 5];
    /// Terms the lane-gain test's documents draw from.
    const LANE_TEST_TERMS: u32 = 1000;
    /// The longest document in the lane-gain test, in terms.
    const LANE_TEST_MAX_LEN: usize = 23;

    /// Summed side by side, each document's gain is the one a
    /// document-at-a-time loop adds up, bit for bit, whatever the lengths
    /// of the documents grouped with it — none, one term, or longer than
    /// any other in its group — and the ranking keys are the ones that loop
    /// makes, serial or parallel.
    #[test]
    fn lane_gains_add_each_document_as_one_at_a_time_does() {
        for n_docs in LANE_TEST_DOCS {
            let mut rng = StdRng::seed_from_u64(n_docs as u64);
            let docs: Vec<Vec<u32>> = (0..n_docs)
                .map(|d| {
                    let len = (d * 5) % LANE_TEST_MAX_LEN;
                    (0..len)
                        .map(|_| rng.random_range(0..LANE_TEST_TERMS))
                        .collect()
                })
                .collect();
            let fwd = ForwardIndex::from_docs(&docs);
            let move_gain: Vec<f32> = (0..LANE_TEST_TERMS)
                .map(|_| rng.random_range(-1.0e3f32..1.0e3))
                .collect();
            let one_at_a_time = |d: u32| {
                let mut gain = 0.0f32;
                for &t in fwd.doc(d) {
                    gain += move_gain[t as usize];
                }
                gain
            };
            let order: Vec<u32> = (0..n_docs as u32).rev().collect();
            for group in order.chunks(GAIN_LANES) {
                let gains = lane_gains(&fwd, group, &move_gain);
                for (gain, &d) in gains.iter().zip(group) {
                    assert_eq!(gain.to_bits(), one_at_a_time(d).to_bits(), "document {d}");
                }
            }
            let expected: Vec<u64> = order
                .iter()
                .enumerate()
                .map(|(i, &d)| rank_key(one_at_a_time(d), i as u32))
                .collect();
            for parallel in [false, true] {
                let mut keys = Vec::new();
                rank_by_gain(&fwd, &order, &move_gain, parallel, &mut keys);
                assert_eq!(keys, expected, "docs={n_docs} parallel={parallel}");
            }
        }
    }

    /// Entry counts the ordered-prefix test ranks: empty, one, around the
    /// first block, odd, and several doublings past it.
    const PREFIX_TEST_LENS: [usize; 9] = [
        0,
        1,
        2,
        SWAP_BLOCK - 1,
        SWAP_BLOCK,
        SWAP_BLOCK + 1,
        3 * SWAP_BLOCK + 7,
        1_001,
        PAR_MIN_ITEMS + 13,
    ];
    /// Distinct gains the ordered-prefix test draws from, few enough that
    /// many entries tie on the gain and are ordered by position.
    const PREFIX_TEST_GAINS: i32 = 40;

    /// The swap loop's way of growing its ordered prefix — order the first
    /// `ready` entries, then on each doubling order only the next ones from
    /// the entries behind the prefix — leaves every prefix it reaches as a
    /// full sort of the whole list would, from any first `ready` and at
    /// every size, odd and tiny ones included. Each step is checked before
    /// the next, as the swap loop reads it.
    #[test]
    fn a_doubled_prefix_orders_as_a_full_sort_does() {
        let mut rng = StdRng::seed_from_u64(11);
        for len in PREFIX_TEST_LENS {
            let keys: Vec<u64> = (0..len as u32)
                .map(|pos| {
                    let gain = rng.random_range(-PREFIX_TEST_GAINS..PREFIX_TEST_GAINS) as f32;
                    rank_key(gain, pos)
                })
                .collect();
            let mut sorted = keys.clone();
            sorted.sort_unstable();
            for first in [1, 2, 3, SWAP_BLOCK] {
                for parallel in [false, true] {
                    let mut gains = keys.clone();
                    let mut ready = first.min(len);
                    order_leading_gains(&mut gains, ready, parallel);
                    loop {
                        assert_eq!(
                            gains[..ready],
                            sorted[..ready],
                            "len={len} first={first} ready={ready} parallel={parallel}"
                        );
                        if ready == len {
                            break;
                        }
                        // The swap loop doubles once it has used the
                        // whole prefix, with the step it takes there.
                        let next = (ready * 2).min(len);
                        extend_ordered_prefix(&mut gains, ready, next, parallel);
                        ready = next;
                    }
                    let mut all = gains.clone();
                    all.sort_unstable();
                    assert_eq!(all, sorted, "the entries are a permutation");
                }
            }
        }
    }

    /// Every round, every split: the shape the oracle implements. The
    /// convergence test is a deliberate change of behaviour and is covered on
    /// its own, so the equivalence tests switch it off.
    fn exhaustive() -> BisectParams {
        BisectParams {
            convergence: 0.0,
            max_rounds: REF_MAX_ROUNDS,
        }
    }

    /// Documents drawn from `n_clusters` vocabularies, shuffled so the
    /// clusters are scattered through arrival order. A correct
    /// reordering pulls each cluster back together.
    fn clustered(n_docs: usize, n_clusters: usize, seed: u64) -> (ForwardIndex, Vec<usize>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut docs: Vec<Vec<u32>> = Vec::with_capacity(n_docs);
        let mut cluster_of = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            let c = rng.random_range(0..n_clusters);
            cluster_of.push(c);
            let base = (c * 50) as u32;
            let mut terms: Vec<u32> = (0..12).map(|_| base + rng.random_range(0..50u32)).collect();
            // A few terms every document shares, so the split cannot
            // simply read the cluster off a single term.
            terms.push(rng.random_range(0..3u32) + (n_clusters * 50) as u32);
            docs.push(terms);
        }
        (ForwardIndex::from_docs(&docs), cluster_of)
    }

    /// The same clustered corpus, but with term ids scattered through the
    /// bucket space a merge hashes its terms into instead of packed into a
    /// dense range.
    ///
    /// This is the shape the bisection is handed in production, and the one
    /// that makes renumbering pay just below the root rather than deep inside
    /// the recursion: the degree tables are sized by the bucket space, which a
    /// corpus's own terms come nowhere near filling, so the first halves a
    /// refine counts already carry under half of it. A dense-id corpus
    /// renumbers only once a partition carries under half its vocabulary.
    fn scattered_into_buckets(
        n_docs: usize,
        n_clusters: usize,
        seed: u64,
    ) -> (ForwardIndex, Vec<usize>) {
        // Collisions merely merge two terms, and the oracle is handed the same
        // index, so the comparison stays honest either way.
        let bucket = |id: u32| -> u32 {
            let h = u64::from(id).wrapping_mul(TERM_BUCKET_MIX);
            (((h >> 32) as u32) ^ (h as u32)) & (TERM_BUCKETS - 1)
        };
        let (dense, cluster_of) = clustered(n_docs, n_clusters, seed);
        let docs: Vec<Vec<u32>> = (0..dense.len() as u32)
            .map(|d| dense.doc(d).iter().map(|&t| bucket(t)).collect())
            .collect();
        (ForwardIndex::from_docs(&docs), cluster_of)
    }

    #[test]
    fn the_order_is_always_a_permutation() {
        for (n, clusters) in [
            (0usize, 1usize),
            (1, 1),
            (31, 2),
            (32, 2),
            (33, 2),
            (500, 4),
        ] {
            let (fwd, _) = clustered(n, clusters, 7);
            let order = bisect_order(&fwd, exhaustive());
            assert_eq!(order.len(), n, "n={n}");
            let mut seen = order.clone();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), n, "n={n}: every document exactly once");
            assert!(order.iter().all(|&d| (d as usize) < n), "n={n}: in range");
        }
    }

    /// The leaf is one posting block: a block of documents keeps its order,
    /// one document more is split.
    #[test]
    fn a_partition_is_split_only_above_a_posting_block() {
        let (block, _) = clustered(BLOCK_LEN, 2, 3);
        assert_eq!(
            bisect_order(&block, exhaustive()),
            (0..BLOCK_LEN as u32).collect::<Vec<_>>()
        );
        let (over, _) = clustered(BLOCK_LEN + 1, 2, 3);
        assert_ne!(
            bisect_order(&over, exhaustive()),
            (0..=BLOCK_LEN as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn documents_without_terms_are_handled() {
        let docs: Vec<Vec<u32>> = (0..200).map(|_| Vec::new()).collect();
        let fwd = ForwardIndex::from_docs(&docs);
        let order = bisect_order(&fwd, exhaustive());
        assert_eq!(order.len(), 200);
        let mut seen = order.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 200);
    }

    #[test]
    fn reordering_lowers_the_cost_it_sets_out_to_lower() {
        for seed in 0..4u64 {
            let (fwd, _) = clustered(2_000, 5, seed);
            let arrival: Vec<u32> = (0..fwd.len() as u32).collect();
            let reordered = bisect_order(&fwd, exhaustive());
            let before = order_cost(&fwd, &arrival, 128);
            let after = order_cost(&fwd, &reordered, 128);
            assert!(
                after < before,
                "seed {seed}: cost {after} did not improve on {before}"
            );
        }
    }

    #[test]
    fn documents_of_a_cluster_end_up_neighbours() {
        // The measure is how often consecutive documents in the order
        // come from the same cluster. Arrival order is random, so it
        // sits near 1/clusters; a working bisection is far above it.
        const CLUSTERS: usize = 5;
        let (fwd, cluster_of) = clustered(2_000, CLUSTERS, 11);
        let order = bisect_order(&fwd, exhaustive());
        let same = |o: &[u32]| {
            o.windows(2)
                .filter(|w| cluster_of[w[0] as usize] == cluster_of[w[1] as usize])
                .count() as f64
                / (o.len() - 1) as f64
        };
        let arrival: Vec<u32> = (0..fwd.len() as u32).collect();
        let before = same(&arrival);
        let after = same(&order);
        assert!(
            after > 0.75 && after > before * 2.0,
            "neighbours from the same cluster: {before} before, {after} after"
        );
    }

    #[test]
    fn the_order_is_deterministic() {
        let (fwd, _) = clustered(1_000, 3, 5);
        assert_eq!(
            bisect_order(&fwd, exhaustive()),
            bisect_order(&fwd, exhaustive())
        );

        // Again with the parameters a real merge runs with, on a corpus large
        // enough to reach the parallel arm. A thousand documents under the
        // exhaustive parameters exercise neither: the convergence test is off,
        // and the splitting never leaves one thread.
        let (fwd, _) = clustered(2 * PARALLEL_MIN_PARTITION, 9, 11);
        let first = bisect_order(&fwd, BisectParams::default());
        for run in 1..8 {
            assert_eq!(
                first,
                bisect_order(&fwd, BisectParams::default()),
                "run {run} of the parallel arm produced a different order"
            );
        }
    }

    /// Same inputs, same bytes — under the parameters a real merge runs with,
    /// on a corpus big enough to reach the parallel arm, and with a root that
    /// has nothing to move.
    ///
    /// The convergence test measures rounds against a reference taken from the
    /// data. Two halves sharing no terms are already apart, so the root's first
    /// round swaps nothing and offers no reference, which leaves its two
    /// children to run concurrently — the point where a reference offered by
    /// whichever finished first would make the whole run depend on a race.
    #[test]
    fn the_order_is_deterministic_when_the_root_has_nothing_to_move() {
        let half = 2 * PARALLEL_MIN_PARTITION;
        let mut docs: Vec<Vec<u32>> = Vec::with_capacity(2 * half);
        let mut rng = StdRng::seed_from_u64(29);
        for d in 0..2 * half {
            // Disjoint vocabularies, so no swap across the root's split can
            // lower the cost and the first round moves nothing. The halves are
            // deliberately unalike -- one densely shared, one nearly all
            // singletons -- so a first round of one is worth far more per
            // document than a first round of the other, and which of them sets
            // the bar decides what the rest of the run does.
            let terms: Vec<u32> = match d < half {
                true => (0..12).map(|_| rng.random_range(0..400u32)).collect(),
                false => (0..12)
                    .map(|_| 10_000 + rng.random_range(0..60_000u32))
                    .collect(),
            };
            docs.push(terms);
        }
        let fwd = ForwardIndex::from_docs(&docs);

        // A root that moves nothing offers no bar, so the level under it
        // does, from both halves' gains summed in partition order — not from
        // whichever half the thread pool happened to finish first, which
        // once made the result depend on scheduling. The same order on a
        // pool of one thread and on the test's pool is what shows it.
        let (parallel, report) = bisect_order_reported(&fwd, BisectParams::default());
        assert!(
            report.reference > 0.0,
            "the level under the root set the bar"
        );
        assert_eq!(report.levels[0].swaps, 0, "the root moved nothing");
        let serial = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool")
            .install(|| bisect_order(&fwd, BisectParams::default()));
        assert_eq!(
            parallel, serial,
            "the bar, and so the order, do not depend on the pool"
        );
    }

    #[test]
    fn a_cost_matches_term_cost_on_both_sides_of_the_table() {
        const SIZE: f32 = 200_000.0;
        let mut costs = Vec::new();
        fill_cost_table(&mut costs, SIZE);
        assert_eq!(costs.len(), COST_TABLE_LEN);
        let side = Side {
            deg: 0,
            size: SIZE,
            costs: &costs,
        };
        let cap = COST_TABLE_LEN as u32;
        for d in [0, 1, 2, cap - 1, cap, cap + 1, SIZE as u32 + 1] {
            assert_eq!(
                side.cost(d).to_bits(),
                term_cost(d, SIZE).to_bits(),
                "d={d}"
            );
        }
    }

    /// Documents in a corpus the serial splitter handles from the root.
    const DENSE_RENUMBER_DOCS: usize = 4_000;
    /// Enough clusters in that corpus that a partition deep in the recursion
    /// carries under half its vocabulary.
    const DENSE_RENUMBER_CLUSTERS: usize = 16;

    /// Terms per document in the uneven corpus, each drawn from its quarter's
    /// own vocabulary.
    const UNEVEN_TERMS_PER_DOC: usize = 12;
    /// Vocabulary of the narrow quarters of the uneven corpus.
    const UNEVEN_NARROW_TERMS: u32 = 200;
    /// Vocabulary of its one wide quarter: twenty times a narrow one, so the
    /// half holding it renumbers at a different depth from the others.
    const UNEVEN_WIDE_TERMS: u32 = 4_000;
    /// Documents in the serial uneven corpus, under the parallel splitter's
    /// threshold so the serial splitter takes it from the root.
    const SERIAL_UNEVEN_DOCS: usize = 4_000;
    /// Vocabulary of its narrow half.
    const SERIAL_NARROW_TERMS: u32 = 400;
    /// Vocabulary of its wide half, so the narrow half fills under half the
    /// whole table and the wide half does not.
    const SERIAL_WIDE_TERMS: u32 = 2_000;
    /// Renumberings the uneven corpus makes when every half renumbers on its
    /// own count.
    const UNEVEN_RENUMBERINGS: u64 = 4;

    /// Bucket space the merge hashes terms into, mirrored here so a test
    /// corpus has the sparse vocabulary a real one does.
    const TERM_BUCKETS: u32 = 1 << 22;

    /// Odd multiplier spreading term ids over [`TERM_BUCKETS`].
    const TERM_BUCKET_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

    /// The round ceiling the bisection used before the convergence test, so
    /// the oracle stays the algorithm the optimizations are measured against.
    const REF_MAX_ROUNDS: usize = 20;

    /// The unoptimized bisection, kept verbatim as an oracle.
    ///
    /// Every optimization in this module is required to leave the chosen
    /// order byte-for-byte unchanged — that is what lets them ship without
    /// re-arguing reorder quality. This is the thing they are checked
    /// against: a direct transcription of the straightforward algorithm,
    /// full sorts and full recomputation per round, no reuse, no
    /// renumbering. It is slow on purpose.
    struct RefState {
        deg_left: Vec<u32>,
        deg_right: Vec<u32>,
        move_gain_left: Vec<f32>,
        move_gain_right: Vec<f32>,
        cost_left: Vec<f32>,
        cost_right: Vec<f32>,
        touched: Vec<u32>,
    }

    impl RefState {
        fn new(n_terms: usize) -> Self {
            Self {
                deg_left: vec![0; n_terms],
                deg_right: vec![0; n_terms],
                move_gain_left: vec![0.0; n_terms],
                move_gain_right: vec![0.0; n_terms],
                cost_left: Vec::new(),
                cost_right: Vec::new(),
                touched: Vec::new(),
            }
        }

        fn split(&mut self, fwd: &ForwardIndex, order: &mut [u32], depth: u32) {
            if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
                return;
            }
            let mid = order.len() / 2;
            self.refine(fwd, order, mid);
            let (left, right) = order.split_at_mut(mid);
            self.split(fwd, left, depth + 1);
            self.split(fwd, right, depth + 1);
        }

        fn refine(&mut self, fwd: &ForwardIndex, order: &mut [u32], mid: usize) {
            for (i, &d) in order.iter().enumerate() {
                for &t in fwd.doc(d) {
                    if self.deg_left[t as usize] == 0 && self.deg_right[t as usize] == 0 {
                        self.touched.push(t);
                    }
                    match i < mid {
                        true => self.deg_left[t as usize] += 1,
                        false => self.deg_right[t as usize] += 1,
                    }
                }
            }
            let n_left = mid as f32;
            let n_right = (order.len() - mid) as f32;
            fill_cost_table(&mut self.cost_left, n_left);
            fill_cost_table(&mut self.cost_right, n_right);

            for _ in 0..REF_MAX_ROUNDS {
                for &t in &self.touched {
                    let t = t as usize;
                    let (dl, dr) = (self.deg_left[t], self.deg_right[t]);
                    let left = Side {
                        deg: dl,
                        size: n_left,
                        costs: &self.cost_left,
                    };
                    let right = Side {
                        deg: dr,
                        size: n_right,
                        costs: &self.cost_right,
                    };
                    if dl > 0 {
                        self.move_gain_left[t] = move_gain(&left, &right);
                    }
                    if dr > 0 {
                        self.move_gain_right[t] = move_gain(&right, &left);
                    }
                }
                let (left, right) = order.split_at_mut(mid);
                let gains = |half: &[u32], g: &[f32]| -> Vec<(f32, u32)> {
                    half.iter()
                        .enumerate()
                        .map(|(i, &d)| {
                            (
                                fwd.doc(d).iter().map(|&t| g[t as usize]).sum::<f32>(),
                                i as u32,
                            )
                        })
                        .collect()
                };
                let mut lg = gains(left, &self.move_gain_left);
                let mut rg = gains(right, &self.move_gain_right);
                let by = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                lg.sort_by(by);
                rg.sort_by(by);

                let mut swaps = 0usize;
                for (l, r) in lg.iter().zip(rg.iter()) {
                    if l.0 + r.0 <= 0.0 {
                        break;
                    }
                    swaps += 1;
                    let (li, ri) = (l.1 as usize, r.1 as usize);
                    let (ld, rd) = (left[li], right[ri]);
                    for &t in fwd.doc(ld) {
                        self.deg_left[t as usize] -= 1;
                        self.deg_right[t as usize] += 1;
                    }
                    for &t in fwd.doc(rd) {
                        self.deg_right[t as usize] -= 1;
                        self.deg_left[t as usize] += 1;
                    }
                    left[li] = rd;
                    right[ri] = ld;
                }
                if swaps == 0 {
                    break;
                }
            }
            for &t in &self.touched {
                self.deg_left[t as usize] = 0;
                self.deg_right[t as usize] = 0;
            }
            self.touched.clear();
        }
    }

    fn reference_order(fwd: &ForwardIndex) -> Vec<u32> {
        let mut order: Vec<u32> = (0..fwd.len() as u32).collect();
        if fwd.len() <= MIN_PARTITION {
            return order;
        }
        RefState::new(fwd.n_terms).split(fwd, &mut order, 0);
        order
    }

    /// The optimizations must not move a single document. Several shapes,
    /// including one past `PARALLEL_MIN_PARTITION` so the parallel arm and
    /// the whole recursion below it are both covered.
    #[test]
    fn the_order_matches_the_unoptimized_bisection() {
        for (docs, clusters, seed) in [
            (2_000usize, 7usize, 1u64),
            (5_000, 3, 2),
            (9_000, 40, 3),
            (PARALLEL_MIN_PARTITION + 1_500, 11, 4),
        ] {
            let (fwd, _) = clustered(docs, clusters, seed);
            assert_eq!(
                bisect_order(&fwd, exhaustive()),
                reference_order(&fwd),
                "order diverged at docs={docs} clusters={clusters} seed={seed}"
            );
        }
    }

    /// The convergence test must actually end splits early and must still
    /// leave the corpus better grouped than it found it — the point is to stop
    /// paying for rounds that have stopped earning, not to stop reordering.
    #[test]
    fn converging_early_still_lowers_the_cost() {
        let (fwd, _) = clustered(20_000, 9, 21);
        let arrival: Vec<u32> = (0..fwd.len() as u32).collect();
        let eager = bisect_order(
            &fwd,
            BisectParams {
                convergence: 0.5,
                max_rounds: REF_MAX_ROUNDS,
            },
        );
        let full = bisect_order(&fwd, exhaustive());

        // Stopping early is a different order from running every round; if it
        // were not, the test would be proving nothing.
        assert_ne!(eager, full, "convergence never fired");

        let cost = |o: &[u32]| order_cost(&fwd, o, 64);
        assert!(
            cost(&eager) < cost(&arrival),
            "early convergence left the order no better than arrival: {} vs {}",
            cost(&eager),
            cost(&arrival)
        );
    }

    /// Renumbering where the parallel splitter still owns the partition keeps
    /// the order the plain bisection produces. The root never renumbers: no
    /// refine has counted its terms; its halves do, as soon as they carry
    /// under half the bucket space.
    #[test]
    fn a_scattered_vocabulary_renumbers_below_the_root_and_keeps_the_order() {
        let (fwd, _) = scattered_into_buckets(3 * PARALLEL_MIN_PARTITION, 11, 5);
        let (order, report) = bisect_order_reported(&fwd, exhaustive());
        let localized = |depth: usize| report.levels.get(depth).map_or(0, |l| l.localized);
        assert_eq!(localized(0), 0, "the root has no count to renumber on");
        assert!(
            localized(1) > 0,
            "this corpus no longer renumbers just below the root, where the \
             parallel splitter owns the partition, so it has stopped covering \
             the path it exists for"
        );
        assert_eq!(order, reference_order(&fwd));
    }

    /// A dense vocabulary renumbers too, once the recursion has cut a
    /// partition down to under half of it, and keeps the order.
    #[test]
    fn a_dense_vocabulary_renumbers_once_its_halves_shrink_and_keeps_the_order() {
        let (fwd, _) = clustered(DENSE_RENUMBER_DOCS, DENSE_RENUMBER_CLUSTERS, 3);
        let (order, report) = bisect_order_reported(&fwd, exhaustive());
        assert!(
            report.levels.iter().any(|l| l.localized > 0),
            "this corpus never renumbers, so it has stopped covering the path"
        );
        assert_eq!(order, reference_order(&fwd));
    }

    /// A refine reports, for each half it leaves, exactly the distinct terms
    /// that half carries.
    #[test]
    fn a_refine_counts_the_distinct_terms_of_each_half() {
        for (n, clusters, seed, parallel) in [(64usize, 2usize, 1u64), (500, 4, 2), (2_000, 9, 3)]
            .into_iter()
            .flat_map(|(n, c, s)| [false, true].map(|parallel| (n, c, s, parallel)))
        {
            let (fwd, _) = clustered(n, clusters, seed);
            let mut order: Vec<u32> = (0..fwd.len() as u32).collect();
            let reference = AtomicU64::new(0);
            let report = RunReport::new();
            let run = Bisect {
                params: exhaustive(),
                reference: &reference,
                report: &report,
                parent: &report.span,
            };
            let mid = order.len() / 2;
            let refined =
                BisectState::new(fwd.n_terms).refine(&fwd, &mut order, mid, 0, parallel, run);
            let distinct = |half: &[u32]| {
                half.iter()
                    .flat_map(|&d| fwd.doc(d).iter().copied())
                    .collect::<HashSet<u32>>()
                    .len()
            };
            assert_eq!(
                refined.distinct,
                [distinct(&order[..mid]), distinct(&order[mid..])],
                "n={n} clusters={clusters} parallel={parallel}"
            );
        }
    }

    /// The trigger: only a counted partition, only into a table it fills at
    /// most half of, and never from an empty table.
    #[test]
    fn renumbering_pays_once_the_terms_fill_at_most_half_the_table() {
        let fwd = ForwardIndex::from_docs(&[vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]]);
        let half = fwd.n_terms / 2;
        assert!(!localizing_pays(&fwd, None), "nothing has counted it");
        assert!(localizing_pays(&fwd, Some(0)));
        assert!(localizing_pays(&fwd, Some(half)));
        assert!(!localizing_pays(&fwd, Some(half + 1)));
        assert!(!localizing_pays(&fwd, Some(fwd.n_terms)));
        let empty = ForwardIndex::from_docs(&[vec![], vec![]]);
        assert!(
            !localizing_pays(&empty, Some(0)),
            "an empty table cannot shrink"
        );
    }

    /// Each half renumbers on its own count, through every path that hands a
    /// count on: the level splitter, the parallel splitter's join, and the
    /// serial splitter.
    ///
    /// Four quarters with disjoint vocabularies, one far wider than the rest,
    /// so the halves differ: a count handed to the wrong half, or not handed
    /// on at all, renumbers different partitions and changes how many. On one
    /// thread the level splitter stops after the root and hands both halves to
    /// the parallel splitter, whose join then hands the quarters theirs.
    #[test]
    fn each_half_renumbers_on_its_own_count() {
        let quarter = PARALLEL_MIN_PARTITION;
        let mut rng = StdRng::seed_from_u64(17);
        let vocabularies = [
            (0, UNEVEN_NARROW_TERMS),
            (UNEVEN_NARROW_TERMS, UNEVEN_WIDE_TERMS),
            (UNEVEN_NARROW_TERMS + UNEVEN_WIDE_TERMS, UNEVEN_NARROW_TERMS),
            (
                2 * UNEVEN_NARROW_TERMS + UNEVEN_WIDE_TERMS,
                UNEVEN_NARROW_TERMS,
            ),
        ];
        let mut docs: Vec<Vec<u32>> = vocabularies
            .iter()
            .flat_map(|&(first, n)| {
                (0..quarter)
                    .map(|_| {
                        (0..UNEVEN_TERMS_PER_DOC)
                            .map(|_| first + rng.random_range(0..n))
                            .collect()
                    })
                    .collect::<Vec<Vec<u32>>>()
            })
            .collect();
        // One document out of place, so the root moves a pair and sets the
        // bar rather than leaving it to the levels below.
        docs.swap(0, 2 * quarter);
        let fwd = ForwardIndex::from_docs(&docs);
        let one = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool");
        let (order, report) = one.install(|| bisect_order_reported(&fwd, exhaustive()));
        let localized: Vec<u64> = report.levels.iter().map(|l| l.localized).collect();
        assert_eq!(
            localized.iter().sum::<u64>(),
            UNEVEN_RENUMBERINGS,
            "renumberings by depth: {localized:?}"
        );
        assert_eq!(order, reference_order(&fwd));
    }

    /// The serial splitter hands each half its own count: of two halves with
    /// disjoint vocabularies, only the narrow one fills under half the table.
    #[test]
    fn the_serial_splitter_hands_each_half_its_own_count() {
        let mut rng = StdRng::seed_from_u64(29);
        let half = SERIAL_UNEVEN_DOCS / 2;
        let docs: Vec<Vec<u32>> = (0..SERIAL_UNEVEN_DOCS)
            .map(|d| {
                let (first, n) = match d < half {
                    true => (0, SERIAL_NARROW_TERMS),
                    false => (SERIAL_NARROW_TERMS, SERIAL_WIDE_TERMS),
                };
                (0..UNEVEN_TERMS_PER_DOC)
                    .map(|_| first + rng.random_range(0..n))
                    .collect()
            })
            .collect();
        let fwd = ForwardIndex::from_docs(&docs);
        let mut order: Vec<u32> = (0..fwd.len() as u32).collect();
        let reference = AtomicU64::new(0);
        let report = RunReport::new();
        let run = Bisect {
            params: exhaustive(),
            reference: &reference,
            report: &report,
            parent: &report.span,
        };
        let mut state = BisectState::new(fwd.n_terms);
        state.split(&fwd, &mut order, 0, run, None);
        state.flush(run.report);
        let report = report.finish(0.0);
        let localized: Vec<u64> = report.levels.iter().map(|l| l.localized).collect();
        assert_eq!(
            localized.iter().sum::<u64>(),
            1,
            "renumberings by depth: {localized:?}"
        );
        assert_eq!(order, reference_order(&fwd));
    }

    /// Documents with no terms at all renumber into an empty table and still
    /// take the order the plain bisection gives them.
    #[test]
    fn a_half_without_terms_keeps_the_order() {
        let (dense, _) = clustered(4 * MIN_PARTITION, 2, 23);
        let docs: Vec<Vec<u32>> = (0..dense.len() as u32)
            .map(|d| dense.doc(d).to_vec())
            .chain((0..4 * MIN_PARTITION).map(|_| Vec::new()))
            .collect();
        let fwd = ForwardIndex::from_docs(&docs);
        let (order, report) = bisect_order_reported(&fwd, exhaustive());
        // Only the termless half shrinks the table by half; the clustered half
        // carries most of the vocabulary, and an empty table cannot shrink.
        let localized: Vec<u64> = report.levels.iter().map(|l| l.localized).collect();
        assert_eq!(
            localized.iter().sum::<u64>(),
            1,
            "renumberings by depth: {localized:?}"
        );
        assert_eq!(order, reference_order(&fwd));
    }

    #[test]
    fn the_parallel_order_matches_the_serial_one() {
        let (fwd, _) = clustered(3 * PARALLEL_MIN_PARTITION, 20, 13);
        let mut serial: Vec<u32> = (0..fwd.len() as u32).collect();
        let reference = AtomicU64::new(0);
        let report = RunReport::new();
        let run = Bisect {
            params: exhaustive(),
            reference: &reference,
            report: &report,
            parent: &report.span,
        };
        BisectState::new(fwd.n_terms).split(&fwd, &mut serial, 0, run, None);
        assert_eq!(bisect_order(&fwd, exhaustive()), serial);
    }

    /// Gains that order unusually under [`f32::total_cmp`].
    const EDGE_GAINS: [f32; 12] = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::MIN_POSITIVE / 2.0,
        -f32::MIN_POSITIVE / 2.0,
        f32::MAX,
        f32::MIN,
    ];

    /// Random gains per seed in the rank-key property test.
    const RANK_KEY_DOCS: usize = 2_000;

    /// Seeds the rank-key property test runs over.
    const RANK_KEY_SEEDS: u64 = 50;

    /// Sorting by [`rank_key`] puts documents in the order the comparison
    /// on `(gain, position)` does, and the key gives back both exactly.
    #[test]
    fn the_rank_key_orders_as_the_gain_comparison_does() {
        let by = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
        for seed in 0..RANK_KEY_SEEDS {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut ranked: Vec<(f32, u32)> = (0..RANK_KEY_DOCS)
                .map(|i| {
                    let gain = match rng.random_range(0..4) {
                        // Ties, and the edge values against each other.
                        0 => EDGE_GAINS[rng.random_range(0..EDGE_GAINS.len())],
                        1 => rng.random_range(-4..4) as f32,
                        2 => f32::from_bits(rng.random()),
                        _ => rng.random_range(-1.0e3f32..1.0e3),
                    };
                    (gain, i as u32)
                })
                .collect();
            let mut keys: Vec<u64> = ranked.iter().map(|&(g, i)| rank_key(g, i)).collect();
            ranked.sort_by(by);
            keys.sort_unstable();
            let decoded: Vec<(u32, u32)> = keys
                .iter()
                .map(|&k| (key_gain(k).to_bits(), key_pos(k)))
                .collect();
            let expected: Vec<(u32, u32)> = ranked.iter().map(|&(g, i)| (g.to_bits(), i)).collect();
            assert_eq!(decoded, expected, "seed {seed}");
        }
    }

    /// The report accounts for every split once, and records the cost really
    /// falling on a corpus with structure to find.
    #[test]
    fn the_report_counts_every_split_once() {
        let (fwd, _) = clustered(3 * PARALLEL_MIN_PARTITION, 20, 13);
        let (order, report) = bisect_order_reported(&fwd, BisectParams::default());
        assert_eq!(order, bisect_order(&fwd, BisectParams::default()));

        assert_eq!(report.levels[0].splits, 1, "one root split");
        assert!(report.reference > 0.0, "the root moved documents");
        for (depth, level) in report.levels.iter().enumerate() {
            assert_eq!(
                level.settled + level.converged + level.capped,
                level.splits,
                "every split at depth {depth} ends exactly one way"
            );
        }
        let total = report.total();
        assert!(total.swaps > 0);
        assert!(total.actual_gain > 0.0, "the cost fell: {total}");
        assert!(total.predicted_gain > 0.0);
        // Under the default bar some splits end on their realised gain
        // before the round ceiling.
        assert!(
            report.levels.iter().any(|level| level.converged > 0),
            "some split ended on the bar"
        );
        assert!(
            total.rounds < total.splits * REORDER_MAX_ROUNDS as u64,
            "not every split ran to the ceiling: {total}"
        );
    }

    /// A corpus that repeats its documents: the root swaps copies of one
    /// document for copies of another, round after round, and lowers the
    /// cost by nothing, so the bar comes from the first level under it that
    /// does with its first rounds, and the splits below end on it instead
    /// of at the ceiling.
    #[test]
    fn the_bar_comes_from_the_first_level_that_lowers_the_cost() {
        /// Wide enough for the top of the bisection to run level by level.
        const N_DOCS: u32 = 40_000;
        /// Documents of four topics of forty terms each, six terms per
        /// document on a stride that makes the documents repeat every
        /// forty ids, every fifth document carrying two extra terms and all
        /// of them one common term.
        fn repeating_doc(id: u32) -> Vec<u32> {
            let base = (id % 4) * 40;
            let mut terms: Vec<u32> = (0..6).map(|step| base + (id + step * 7) % 40).collect();
            if id.is_multiple_of(5) {
                terms.extend([160, 161]);
            }
            terms.push(162);
            terms
        }
        let docs: Vec<Vec<u32>> = (0..N_DOCS).map(repeating_doc).collect();
        let fwd = ForwardIndex::from_docs(&docs);
        let (_, report) = bisect_order_reported(&fwd, BisectParams::default());
        assert!(report.reference > 0.0, "a level lowered the cost: {report}");
        assert_eq!(
            report.levels[0].capped, 1,
            "the root realised nothing and ran to the ceiling: {report}"
        );
        let first = report
            .levels
            .iter()
            .position(|level| level.actual_gain > 0.0)
            .expect("a level lowered the cost");
        let (capped, splits) = report.levels[first + 1..]
            .iter()
            .fold((0, 0), |(c, n), level| (c + level.capped, n + level.splits));
        assert!(
            capped * 4 < splits,
            "with the bar set, the splits below end on it, not at the ceiling: {report}"
        );
    }
}
