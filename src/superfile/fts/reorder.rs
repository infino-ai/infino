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
    utils::trace::detail_span,
};

/// The sign bit of an `f32`.
const SIGN_BIT: u32 = 0x8000_0000;

/// Documents below this per-partition size are left in the order they
/// already have. Splitting further costs more than the grouping is
/// worth, and the gaps inside a group this small are already short.
const MIN_PARTITION: usize = 32;

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
/// Index size stands in for layout quality throughout. The reordering exists
/// for query performance, which is not resolved at this scale.
const REORDER_CONVERGENCE: f32 = 0.05;

/// Ceiling on a split's move rounds. Equal to the fixed count the bisection
/// used before the convergence test existed, so setting [`REORDER_CONVERGENCE`]
/// to zero reproduces the old behaviour exactly. A backstop against a corpus
/// whose rounds keep paying, not the working limit — the convergence test is
/// what normally ends a split.
const REORDER_MAX_ROUNDS: usize = 20;

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
    /// takes from the root split's first round — not from the first round of
    /// the split being tested. `0.0` disables the test.
    pub(crate) convergence: f32,
    /// Ceiling on a split's move rounds whatever `convergence` says.
    pub(crate) max_rounds: usize,
}

/// A bisection in progress: the limits in force, plus the reference the
/// convergence test measures rounds against.
///
/// The reference is the gain per document of the first split's first round —
/// the root's, since it is processed first — so the bar is taken from the
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
    /// Whether a round that realised `gain` over `len` documents means this
    /// split has stopped paying.
    fn converged(&self, gain: f64, len: usize) -> bool {
        if self.params.convergence <= 0.0 {
            return false;
        }
        let reference = f64::from_bits(self.reference.load(Relaxed));
        // No reference yet means the first split made no move at all; there is
        // nothing to measure against, so the ceiling is the only limit.
        reference > 0.0
            && gain / (len.max(1) as f64) < f64::from(self.params.convergence) * reference
    }

    /// Offer the root's first round as the reference for the whole run.
    ///
    /// Only the split at depth zero may offer. Any split can be first to
    /// finish a first round, but below the root several run at once, so
    /// letting them offer would hand the bar to whichever the thread pool
    /// happened to finish first and make the output depend on scheduling. The
    /// root runs alone, so taking the bar from it alone is what keeps a merge
    /// reproducible. A root that moves nothing offers nothing, and the run
    /// then has no bar and keeps every round.
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
    /// Zero when the root moved nothing.
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
/// While a level holds fewer partitions than the pool has threads, its splits
/// run side by side and each spreads its rounds across threads too. No subtree
/// is queued until the whole level is done, so a round waiting on its own
/// parallel pieces can only pick up another split of the same level, never a
/// subtree many times its size. Every partition goes through exactly the steps
/// [`split_parallel`] would take it through, so the order is the same.
fn split_levels(fwd: &ForwardIndex, order: &mut [u32], pool: &StatePool, run: Bisect<'_>) {
    let width = rayon::current_num_threads();
    let mut level: Vec<&mut [u32]> = vec![order];
    let mut handed_on: Vec<(&mut [u32], u32)> = Vec::new();
    let mut depth = 0u32;
    while !level.is_empty() && level.len() < width {
        // Only a partition `split_parallel` would refine and halve itself is
        // split here; any other takes its own path there.
        let (here, rest): (Vec<_>, Vec<_>) = level.into_iter().partition(|part| {
            part.len() > MIN_PARTITION
                && depth < MAX_DEPTH
                && part.len() >= PARALLEL_MIN_PARTITION
                && !localizing_pays(fwd, part)
        });
        handed_on.extend(rest.into_iter().map(|part| (part, depth)));
        level = here
            .into_par_iter()
            .flat_map_iter(|part| {
                let mid = part.len() / 2;
                pool.with_state(|state| {
                    state.refine(fwd, part, mid, depth, true, run);
                    state.flush(run.report);
                });
                let (left, right) = part.split_at_mut(mid);
                [left, right]
            })
            .collect();
        depth += 1;
    }
    handed_on.extend(level.into_iter().map(|part| (part, depth)));
    handed_on
        .into_par_iter()
        .for_each(|(part, depth)| split_parallel(fwd, part, depth, pool, run));
}

/// Split `order` as [`BisectState::split`] does, running the two halves
/// of a large partition in parallel.
fn split_parallel(
    fwd: &ForwardIndex,
    order: &mut [u32],
    depth: u32,
    pool: &StatePool,
    run: Bisect<'_>,
) {
    if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
        return;
    }
    if localizing_pays(fwd, order) {
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
        split_parallel(&local, &mut inner, depth, &inner_pool, run);
        apply_inner_order(order, &inner);
        return;
    }
    if order.len() < PARALLEL_MIN_PARTITION {
        // Both of `split`'s entry tests have just been made above, against
        // this same partition.
        pool.with_state(|state| {
            state.split_halves(fwd, order, depth, run);
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
    pool.with_state(|state| {
        state.refine(fwd, order, mid, depth, depth == 0, run);
        state.flush(run.report);
    });
    let (left, right) = order.split_at_mut(mid);
    let (left_done, right_done) = join(
        || {
            split_parallel(fwd, left, depth + 1, pool, run);
            REPORT.then(Instant::now)
        },
        || {
            split_parallel(fwd, right, depth + 1, pool, run);
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
/// indexed against, and a partition can carry no more distinct terms than it
/// has postings. So when the postings cannot fill half the current table,
/// renumbering is guaranteed to at least halve it, and one pass over the
/// partition buys every split below a table that sits closer to cache.
///
/// Expressing the trigger as "the vocabulary has shrunk by half" rather than
/// as a document count is what makes it travel between corpora: it fires on
/// the relationship between a partition and its vocabulary, which is the thing
/// that actually decides whether the tables are oversized. Halving at minimum
/// also bounds how often it can fire on any root-to-leaf path.
fn localizing_pays(fwd: &ForwardIndex, order: &[u32]) -> bool {
    // An empty table cannot shrink, and without this a partition carrying no
    // terms at all satisfies the test against its own renumbering forever.
    if fwd.n_terms == 0 {
        return false;
    }
    let postings: usize = order.iter().map(|&d| fwd.doc(d).len()).sum();
    postings.saturating_mul(2) <= fwd.n_terms
}

/// Apply a permutation computed over a partition's own positions back onto
/// the caller's slice.
fn apply_inner_order(order: &mut [u32], inner: &[u32]) {
    let was: Vec<u32> = order.to_vec();
    for (slot, &i) in order.iter_mut().zip(inner.iter()) {
        *slot = was[i as usize];
    }
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

    fn split(&mut self, fwd: &ForwardIndex, order: &mut [u32], depth: u32, run: Bisect<'_>) {
        if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
            return;
        }
        if localizing_pays(fwd, order) {
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
            inner_state.split(&local, &mut inner, depth, run);
            inner_state.flush(run.report);
            apply_inner_order(order, &inner);
            return;
        }
        self.split_halves(fwd, order, depth, run);
    }

    /// [`Self::split`] with the entry tests already made.
    ///
    /// Only for a caller that has just run them against this same partition:
    /// [`localizing_pays`] walks every posting in `order` to decide, so the
    /// handoff out of the parallel splitter would otherwise pay for that walk
    /// twice to reach the answer it already has.
    fn split_halves(&mut self, fwd: &ForwardIndex, order: &mut [u32], depth: u32, run: Bisect<'_>) {
        let mid = order.len() / 2;
        self.refine(fwd, order, mid, depth, false, run);
        let (left, right) = order.split_at_mut(mid);
        self.split(fwd, left, depth + 1, run);
        self.split(fwd, right, depth + 1, run);
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
    ) {
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
        self.count_degrees(fwd, order, mid);
        if let Some(step) = step {
            level.count_ns += ns_since(step);
            level.serial_cpu_ns += thread_cpu_delta_ns(cpu);
        }
        let n_left = mid as f32;
        let n_right = (order.len() - mid) as f32;
        fill_cost_table(&mut self.cost_left, n_left);
        fill_cost_table(&mut self.cost_right, n_right);
        // The cost before the last round that moved documents, until the next
        // gain pass measures what that round really bought.
        let mut cost_before: Option<f64> = None;

        // Taken out of `self` so the ranking can borrow the move-gain
        // tables while writing them, and put back before returning so the
        // next split reuses the same allocations.
        let mut left_gains = mem::take(&mut self.left_gains);
        let mut right_gains = mem::take(&mut self.right_gains);
        let mut last_swaps = 0usize;
        // A round's realised gain is the sum of the pair gains it accepted,
        // in the units of the cost this is minimising. The swap loop already
        // computes them, so the convergence test is a running total and a
        // comparison.
        for round in 0..run.params.max_rounds {
            // A document's gain is what the cost drops by if it moves:
            // its terms get one rarer on this side and one commoner on
            // the other. Positive means the move is worth making.
            let (moved, round_gain, round_cost) = {
                let cost = REPORT.then(|| self.partition_cost(n_left, n_right));
                if let (Some(before), Some(cost)) = (cost_before.take(), cost) {
                    level.record_actual(before - cost);
                }
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
                        ready = (ready * 2).min(pairs);
                        order_leading_gains(&mut left_gains, ready, parallel);
                        order_leading_gains(&mut right_gains, ready, parallel);
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
                    lefts.par_iter().zip(rights).for_each(|(&l, &r)| {
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
                (swaps, gain, cost)
            };
            last_swaps = moved;
            level.rounds += 1;
            if moved == 0 {
                end = SplitEnd::Settled;
                break;
            }
            level.swaps += moved as u64;
            level.predicted_gain += round_gain;
            cost_before = round_cost;
            if round == 0 {
                if depth == 0 {
                    run.offer_reference(round_gain, order.len());
                }
            } else if run.converged(round_gain, order.len()) {
                // This round bought less per document than the corpus showed
                // was available, so the rounds still to come are not worth
                // their pass over the partition — and the work they would take
                // is better spent on a split that is still paying.
                end = SplitEnd::Converged;
                break;
            }
        }
        if let Some(before) = cost_before {
            level.record_actual(before - self.partition_cost(n_left, n_right));
        }

        self.left_gains = left_gains;
        self.right_gains = right_gains;
        self.clear_degrees();

        let Some(started) = started else {
            return;
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
    }

    /// The cost of the current split, from its degree tables.
    fn partition_cost(&self, n_left: f32, n_right: f32) -> f64 {
        let mut cost = 0.0f64;
        for &t in &self.touched {
            let t = t as usize;
            cost += self.term_split_cost(t, n_left, n_right);
        }
        cost
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
            touched.par_iter().for_each(|&t| {
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

    fn count_degrees(&mut self, fwd: &ForwardIndex, order: &[u32], mid: usize) {
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
    let gain_of = |(i, &d): (usize, &u32)| {
        let mut gain = 0.0f32;
        for &t in fwd.doc(d) {
            gain += move_gain[t as usize];
        }
        rank_key(gain, i as u32)
    };
    out.clear();
    match parallel {
        // `par_extend` over an indexed iterator fills `out` in index
        // order, so the ranking is the same whichever way it ran.
        true => out.par_extend(half.par_iter().enumerate().map(gain_of)),
        false => out.extend(half.iter().enumerate().map(gain_of)),
    }
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
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::*;

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
    /// that makes renumbering pay at the top of the recursion rather than deep
    /// inside it: the degree tables are sized by the bucket space, which a
    /// corpus's own postings come nowhere near filling. A dense-id corpus
    /// cannot reach that path at all, because a partition large enough for the
    /// parallel splitter to still own it carries far more postings than such a
    /// vocabulary has terms.
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

    #[test]
    fn a_corpus_too_small_to_split_keeps_its_order() {
        let (fwd, _) = clustered(MIN_PARTITION, 2, 3);
        assert_eq!(
            bisect_order(&fwd, exhaustive()),
            (0..MIN_PARTITION as u32).collect::<Vec<_>>()
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

        // A root that moves nothing offers no reference, so there is no bar to
        // measure against and convergence is off for the whole run. The result
        // must therefore be exactly what disabling the test outright gives.
        // When a child could offer the reference instead, this held or not
        // depending on which child the thread pool happened to finish first.
        let disabled = BisectParams {
            convergence: 0.0,
            ..BisectParams::default()
        };
        assert_eq!(
            bisect_order(&fwd, BisectParams::default()),
            bisect_order(&fwd, disabled),
            "a split below the root set the convergence bar"
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

    /// Renumbering at the top of the recursion, where the parallel splitter
    /// still owns the partition, keeps the order the plain bisection produces.
    ///
    /// The dense-id cases cannot reach this: renumbering only fires for them
    /// once the recursion has cut partitions down to a few dozen documents, by
    /// which point the serial splitter has long since taken over.
    #[test]
    fn a_scattered_vocabulary_renumbers_at_the_top_and_keeps_the_order() {
        let (fwd, _) = scattered_into_buckets(3 * PARALLEL_MIN_PARTITION, 11, 5);
        let whole: Vec<u32> = (0..fwd.len() as u32).collect();
        assert!(
            localizing_pays(&fwd, &whole),
            "this corpus no longer renumbers at the root, so it has stopped \
             covering the path it exists for"
        );
        assert_eq!(bisect_order(&fwd, exhaustive()), reference_order(&fwd));
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
        BisectState::new(fwd.n_terms).split(&fwd, &mut serial, 0, run);
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
    }
}
