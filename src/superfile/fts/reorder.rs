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
    mem,
    sync::{Mutex, PoisonError},
};

use rayon::{join, prelude::*};

/// Documents below this per-partition size are left in the order they
/// already have. Splitting further costs more than the grouping is
/// worth, and the gaps inside a group this small are already short.
const MIN_PARTITION: usize = 32;

/// Move rounds per split. The first round captures nearly all of the
/// available gain and later ones taper sharply, so this trades a long
/// tail of tiny improvements for a bounded cost per level.
const MAX_ROUNDS: usize = 20;

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
pub(crate) fn bisect_order(fwd: &ForwardIndex) -> Vec<u32> {
    let n = fwd.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    if n <= MIN_PARTITION {
        return order;
    }
    let pool = StatePool {
        n_terms: fwd.n_terms,
        free: Mutex::new(Vec::new()),
    };
    split_parallel(fwd, &mut order, 0, &pool);
    order
}

/// Split `order` as [`BisectState::split`] does, running the two halves
/// of a large partition in parallel.
fn split_parallel(fwd: &ForwardIndex, order: &mut [u32], depth: u32, pool: &StatePool) {
    if order.len() < PARALLEL_MIN_PARTITION {
        pool.with_state(|state| state.split_localized(fwd, order, depth));
        return;
    }
    if depth >= MAX_DEPTH {
        return;
    }
    let mid = order.len() / 2;
    pool.with_state(|state| state.refine(fwd, order, mid, true));
    let (left, right) = order.split_at_mut(mid);
    join(
        || split_parallel(fwd, left, depth + 1, pool),
        || split_parallel(fwd, right, depth + 1, pool),
    );
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
    /// Ranked documents for each side, reused by every round of every
    /// split this state serves. Ranking used to return a fresh `Vec` per
    /// side per round, which is two allocations per round per partition —
    /// millions of them over a corpus-sized bisection.
    left_gains: Vec<(f32, u32)>,
    right_gains: Vec<(f32, u32)>,
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
        }
    }

    /// Split a partition small enough to own its vocabulary: renumber the
    /// terms it actually carries into a dense range and run the whole
    /// subtree against that instead of against the corpus vocabulary.
    ///
    /// The degree and move-gain tables are indexed by term id, so over the
    /// shared vocabulary they stay `n_terms` entries wide however few
    /// documents a partition holds — tens of megabytes that every round
    /// walks at random and misses cache on, even once a partition is down
    /// to a few thousand documents. Renumbering costs one pass over this
    /// partition's postings, and every split beneath it inherits tables
    /// sized to the terms actually present.
    ///
    /// Renumbering is a bijection, so every degree, gain and comparison is
    /// unchanged, and the terms within a document keep their order, so the
    /// gain sums add in the same sequence and to the same float.
    fn split_localized(&mut self, fwd: &ForwardIndex, order: &mut [u32], depth: u32) {
        if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
            return;
        }

        // Outside a refine `deg_left` is all zeros — `clear_degrees` leaves
        // it that way and this runs in a refine's place — so it serves as
        // the global-to-local term map without a second table that size.
        // Zero means "not seen here"; a local id is held as `id + 1`.
        let mut terms: Vec<u32> = Vec::with_capacity(order.len() * 8);
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

        let local = ForwardIndex {
            terms,
            starts,
            n_terms: n_local as usize,
        };
        // Positions within this partition, permuted by the subtree and then
        // applied to the caller's slice.
        let mut local_order: Vec<u32> = (0..order.len() as u32).collect();
        BisectState::new(local.n_terms).split(&local, &mut local_order, depth);

        let was: Vec<u32> = order.to_vec();
        for (slot, &l) in order.iter_mut().zip(local_order.iter()) {
            *slot = was[l as usize];
        }
    }

    fn split(&mut self, fwd: &ForwardIndex, order: &mut [u32], depth: u32) {
        if order.len() <= MIN_PARTITION || depth >= MAX_DEPTH {
            return;
        }
        let mid = order.len() / 2;
        self.refine(fwd, order, mid, false);
        let (left, right) = order.split_at_mut(mid);
        self.split(fwd, left, depth + 1);
        self.split(fwd, right, depth + 1);
    }

    /// Move documents across the split while it lowers the cost, then
    /// leave the two halves in `order`. `parallel` spreads each round's
    /// gains and sorts across threads.
    fn refine(&mut self, fwd: &ForwardIndex, order: &mut [u32], mid: usize, parallel: bool) {
        self.count_degrees(fwd, order, mid);
        let n_left = mid as f32;
        let n_right = (order.len() - mid) as f32;
        fill_cost_table(&mut self.cost_left, n_left);
        fill_cost_table(&mut self.cost_right, n_right);

        // Taken out of `self` so the ranking can borrow the move-gain
        // tables while writing them, and put back before returning so the
        // next split reuses the same allocations.
        let mut left_gains = mem::take(&mut self.left_gains);
        let mut right_gains = mem::take(&mut self.right_gains);
        let mut last_swaps = 0usize;

        for _ in 0..MAX_ROUNDS {
            // A document's gain is what the cost drops by if it moves:
            // its terms get one rarer on this side and one commoner on
            // the other. Positive means the move is worth making.
            let moved = {
                self.compute_move_gains(n_left, n_right);
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

                let mut swaps = 0usize;
                while swaps < pairs {
                    if swaps == ready {
                        ready = (ready * 2).min(pairs);
                        order_leading_gains(&mut left_gains, ready, parallel);
                        order_leading_gains(&mut right_gains, ready, parallel);
                    }
                    let (l, r) = (left_gains[swaps], right_gains[swaps]);
                    if l.0 + r.0 <= 0.0 {
                        break;
                    }
                    swaps += 1;
                    let (li, ri) = (l.1 as usize, r.1 as usize);
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
                swaps
            };
            last_swaps = moved;
            if moved == 0 {
                break;
            }
        }

        self.left_gains = left_gains;
        self.right_gains = right_gains;
        self.clear_degrees();
    }

    /// Fill the move gains of every term in the partition from its
    /// current degrees. A side's gain is only read by documents on that
    /// side carrying the term, so a side with no such document is skipped.
    fn compute_move_gains(&mut self, n_left: f32, n_right: f32) {
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

/// `(gain, position within the half)` for every document in `half`: the
/// sum of its terms' move gains for that side, written into `out`.
///
/// Takes the output buffer rather than returning one so a split's rounds
/// share two allocations instead of making two apiece.
fn rank_by_gain(
    fwd: &ForwardIndex,
    half: &[u32],
    move_gain: &[f32],
    parallel: bool,
    out: &mut Vec<(f32, u32)>,
) {
    let gain_of = |(i, &d): (usize, &u32)| {
        let mut gain = 0.0f32;
        for &t in fwd.doc(d) {
            gain += move_gain[t as usize];
        }
        (gain, i as u32)
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
/// Ties break by position, which is unique, so the comparison is a total
/// order and the leading `want` entries are the same whatever `want` was
/// asked for before. That is what lets a round grow its block without
/// changing which documents it swaps.
fn order_leading_gains(gains: &mut [(f32, u32)], want: usize, parallel: bool) {
    let by = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    if want >= gains.len() {
        match parallel {
            true => gains.par_sort_by(by),
            false => gains.sort_by(by),
        }
        return;
    }
    if want == 0 {
        return;
    }
    // Linear-time partition around the `want`-th entry, then order only
    // what sits in front of it.
    gains.select_nth_unstable_by(want, by);
    let head = &mut gains[..want];
    match parallel {
        true => head.par_sort_by(by),
        false => head.sort_by(by),
    }
}

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
            let order = bisect_order(&fwd);
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
            bisect_order(&fwd),
            (0..MIN_PARTITION as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn documents_without_terms_are_handled() {
        let docs: Vec<Vec<u32>> = (0..200).map(|_| Vec::new()).collect();
        let fwd = ForwardIndex::from_docs(&docs);
        let order = bisect_order(&fwd);
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
            let reordered = bisect_order(&fwd);
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
        let order = bisect_order(&fwd);
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
        assert_eq!(bisect_order(&fwd), bisect_order(&fwd));
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

    /// Today's bisection, kept verbatim as an oracle.
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

            for _ in 0..MAX_ROUNDS {
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
                bisect_order(&fwd),
                reference_order(&fwd),
                "order diverged at docs={docs} clusters={clusters} seed={seed}"
            );
        }
    }

    #[test]
    fn the_parallel_order_matches_the_serial_one() {
        let (fwd, _) = clustered(3 * PARALLEL_MIN_PARTITION, 20, 13);
        let mut serial: Vec<u32> = (0..fwd.len() as u32).collect();
        BisectState::new(fwd.n_terms).split(&fwd, &mut serial, 0);
        assert_eq!(bisect_order(&fwd), serial);
    }
}
