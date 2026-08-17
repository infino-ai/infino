// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS scoring/collection kernels on [`FtsReader`]: the disjunction
//! scorers (WAND+BMW, MaxScore+BMM, windowed, exhaustive) with their
//! dispatch, and the AND flat-merge intersection family. Split from the
//! reader `core` as its own `impl FtsReader` block.

use std::{cmp::Ordering, collections::BinaryHeap, slice::from_mut};

use super::{
    core::*,
    cursor::TermCursor,
    filter::ExcludeFilter,
    metadata::NormTable,
    sink::{
        AndSink, CollectSink, CountSink, MustShouldSink, ScoreSink, TopKEntry, drain_top_k_desc,
    },
};
use crate::superfile::{
    error::FtsError,
    fts::{bm25, posting::BLOCK_LEN},
};

/// Intersection cardinality by a rarest-driven membership walk: iterate the
/// term with the fewest blocks and count docs the others all contain. Each
/// membership probe is `TermCursor::contains`, which bit-tests a bitset
/// block with no decode — so a common (bitset) term's blocks are never
/// expanded. Used for `v4` blobs; the flat-merge stays for `v1`–`v3`, where
/// every block is PFOR and the sorted merge over decoded blocks is faster.
fn count_and_intersect_membership(mut cursors: Vec<TermCursor>) -> u64 {
    // Drive by the rarest term (fewest blocks) to minimise membership
    // probes. Ties don't matter; any driver yields the same count.
    let driver_idx = cursors
        .iter()
        .enumerate()
        .min_by_key(|(_, c)| c.block_count())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mut driver = cursors.swap_remove(driver_idx);
    let mut others = cursors;
    let mut n = 0u64;
    while !driver.is_exhausted() {
        let doc = driver.current_doc_id();
        if others.iter_mut().all(|o| o.contains(doc)) {
            n += 1;
        }
        driver.next();
    }
    n
}

/// Intersection cardinality via bitset AND: build each term's doc presence into
/// a doc-space bitset (byte-level — a dense term's bitset blocks are word-copied,
/// never expanded to doc ids), AND them together, and popcount. Word-parallel, so
/// its cost is `n_terms × max_doc/64` regardless of how many docs the rarest term
/// holds — the opposite of the rarest-driven membership walk, which iterates the
/// rarest term doc by doc. Wins when *every* term is common (dense), where the
/// membership driver is itself a long list. `max_doc` is the largest doc id
/// across the terms, so `acc`/`scratch` span every term's presence.
fn count_and_intersect_bitset(cursors: Vec<TermCursor>, max_doc: u32) -> u64 {
    let words = max_doc as usize / 64 + 1;
    let mut acc = vec![0u64; words];
    let mut scratch = vec![0u64; words];
    let mut decode = [0u32; BLOCK_LEN];
    let mut cursors = cursors.iter();
    // The first term seeds the accumulator; the rest AND their presence in.
    let Some(first) = cursors.next() else {
        return 0;
    };
    or_cursor_into_bitset(&mut acc, first, &mut decode);
    for c in cursors {
        scratch.iter_mut().for_each(|w| *w = 0);
        or_cursor_into_bitset(&mut scratch, c, &mut decode);
        for (a, s) in acc.iter_mut().zip(scratch.iter()) {
            *a &= *s;
        }
    }
    acc.iter().map(|w| w.count_ones() as u64).sum()
}

/// Block-Max-AND upper bound at the leader's doc, valid over
/// `[leader_doc, window_end]` where `window_end` is the smallest block boundary
/// across all cursors. Each non-leader is bounded by the single block that
/// *contains* `leader_doc`, not a max over every block under the leader's whole
/// (possibly wide) block — the looser range-max collapsed to a common term's
/// global max on rare∧common queries, so the skip rarely fired. Leader block
/// max/end are passed in (callers hold the leader split off for the flat-merge).
fn block_max_and_bound(
    leader_block_max: f32,
    leader_block_end: u32,
    others: &mut [TermCursor],
    leader_doc: u32,
) -> (f32, u32) {
    let mut ub = leader_block_max;
    let mut window_end = leader_block_end;
    for c in others.iter_mut() {
        c.shallow_advance_block_to(leader_doc);
        ub += c.inspect_block_max_bm25();
        window_end = window_end.min(c.inspect_block_last_doc_id());
    }
    (ub, window_end)
}

impl FtsReader {
    /// Multi-term OR via WAND + BlockMaxWAND.
    ///
    /// Algorithm: maintain a `TermCursor` per query term. Each
    /// iteration sorts cursors by current `doc_id`, computes the
    /// **WAND pivot** (smallest j such that the prefix-sum of
    /// term-level upper bounds exceeds the kth-best score), then
    /// applies the **BMW augmentation** (per-block UBs across the
    /// pivot prefix). If the pivot doc can't beat the threshold even
    /// with full per-block UBs, advance the leftmost cursor past the
    /// smallest block-end among the prefix; otherwise score the doc
    /// and advance.
    ///
    /// Reference: Ding & Suel, "Faster Top-k Document Retrieval Using
    /// Block-Max Indexes", SIGIR 2011.
    ///
    /// Result invariants: top-k by descending BM25 score, ties broken
    /// by ascending doc_id.
    ///
    /// Production path for small-`k`, **floor-free** 2-term ORs (see
    /// `dispatch_or_algo`), and the `search_with_algo_for_bench`
    /// entry point. Cursor construction is shared with the BMM path.
    ///
    /// Carries **no cross-segment floor and no exclude filter** — the
    /// dispatcher only routes here when both are absent (`floor_eff` is
    /// `NEG_INFINITY` and the query has no negation). Seeding WAND's pivot
    /// threshold from a finite floor was tried and reverted: it skipped
    /// blocks that still held qualifying docs at higher floors (caught by
    /// `wand_bmw_2term_no_floor_agrees_with_bmm` vs the floored BMM). When
    /// a floor is live, MaxScore handles it instead.
    pub(super) fn run_wand_bmw(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // `search_multi` passes `k = usize::MAX` to gather every
        // matching doc before weighting across columns; cap initial
        // capacity at n_docs (the upper bound on distinct doc_ids in
        // the heap) so we don't try to allocate `usize::MAX * size_of::<TopKEntry>()`.
        // The BinaryHeap grows on demand if needed.
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        // Reused index buffer to avoid per-iteration allocation.
        let mut idx: Vec<usize> = Vec::with_capacity(cursors.len());

        loop {
            // Drop exhausted cursors. Doing this in-place keeps idx
            // valid for the next iteration without re-allocation.
            cursors.retain(|c| !c.is_exhausted());
            if cursors.is_empty() {
                break;
            }

            // Sort cursor indices ascending by current doc_id.
            idx.clear();
            idx.extend(0..cursors.len());
            // Per-iteration WAND cursor reorder; pdqsort because
            // cursors hold distinct current doc_ids in the heap
            // state used by this scan.
            idx.sort_unstable_by_key(|&i| cursors[i].current_doc_id());

            // WAND pivot: smallest j such that the prefix-sum of
            // *term-level* upper bounds exceeds the threshold.
            let mut accum_term_ub: f32 = 0.0;
            let mut pivot_j: Option<usize> = None;
            for (j, &ci) in idx.iter().enumerate() {
                accum_term_ub += cursors[ci].term_max_bm25;
                if accum_term_ub > threshold {
                    pivot_j = Some(j);
                    break;
                }
            }

            let Some(mut pivot_j) = pivot_j else {
                // Sum of all remaining term UBs ≤ threshold: no
                // future doc can beat the heap. Done.
                break;
            };

            let pivot_doc = cursors[idx[pivot_j]].current_doc_id();

            // Extend the pivot prefix to include any cursors past
            // `pivot_j` that are also at `pivot_doc`. They contribute
            // to both the BMW upper-bound sum and the actual score,
            // so missing them under-counts the BMW UB and could
            // trigger an incorrect skip.
            while pivot_j + 1 < idx.len() && cursors[idx[pivot_j + 1]].current_doc_id() == pivot_doc
            {
                pivot_j += 1;
            }

            // BMW augmentation: sum of per-block upper bounds for the
            // block that would contain `pivot_doc` in each prefix
            // cursor. Lagging cursors' current decoded block is for
            // an earlier doc whose UB doesn't bound their
            // contribution at pivot_doc; `shallow_advance_block_to`
            // moves the lightweight inspect-block pointer to the
            // pivot-doc block without decoding, then
            // `inspect_block_max_bm25` reads that block's UB.
            let mut accum_block_ub: f32 = 0.0;
            for &ci in &idx[..=pivot_j] {
                cursors[ci].shallow_advance_block_to(pivot_doc);
                accum_block_ub += cursors[ci].inspect_block_max_bm25();
            }

            if accum_block_ub <= threshold {
                // No doc in [pivot_doc, smallest_pivot_block_end]
                // can beat the kth-best score. Advance the leftmost
                // cursor to the next interesting doc — either one
                // past the smallest pivot-block-end among the prefix,
                // or a suffix cursor's current doc if that's closer.
                // The suffix cap matters for recall: without it,
                // leftmost can leap multiple blocks past pivot_doc
                // and overshoot a doc one of the suffix cursors is
                // sitting at, leaving that doc with too few cursors
                // ever positioned on it to score correctly.
                let mut target = u32::MAX;
                for &ci in &idx[..=pivot_j] {
                    let last = cursors[ci].inspect_block_last_doc_id();
                    if last < target {
                        target = last;
                    }
                }
                let mut effective_target = target.saturating_add(1);
                for &ci in &idx[pivot_j + 1..] {
                    let d = cursors[ci].current_doc_id();
                    if d < effective_target {
                        effective_target = d;
                    }
                }
                cursors[idx[0]].skip_to(effective_target);
                continue;
            }

            // Align every lagging cursor in the pivot prefix to
            // `pivot_doc` so its contribution is included in this
            // doc's score. If any cursor's posting list doesn't
            // contain `pivot_doc` (the seek lands past it), abandon
            // this pivot — re-sort and re-pivot next iteration. This
            // is the WAND alignment step (Ding & Suel §3); without
            // it, lagging cursors that DO have pivot_doc in their
            // posting list get advanced past it on subsequent
            // iterations without ever contributing to its score,
            // producing under-counted scores and missing top-k hits.
            let mut aligned = true;
            for &ci in &idx[..=pivot_j] {
                if cursors[ci].current_doc_id() < pivot_doc {
                    cursors[ci].skip_to(pivot_doc);
                    if cursors[ci].current_doc_id() != pivot_doc {
                        aligned = false;
                        break;
                    }
                }
            }
            if !aligned {
                continue;
            }

            // All prefix cursors are at pivot_doc. Score it by summing
            // contributions from every cursor at pivot_doc (cursors
            // beyond the prefix may also be at pivot_doc — they
            // contribute too). SIMD-pack up to 4 cursors per scoring
            // call.
            let norm = dl_norm_k1.get(pivot_doc);
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in &cursors {
                if cursor.current_doc_id() == pivot_doc {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                }
            }
            if packed > 0 {
                score += bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Update heap.
            if heap.len() < k {
                heap.push(TopKEntry(score, pivot_doc));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if let Some(TopKEntry(min_score, _)) = heap.peek()
                && score > *min_score
            {
                heap.pop();
                heap.push(TopKEntry(score, pivot_doc));
                threshold = heap.peek().expect("non-empty").0;
            }

            // Advance every cursor at pivot_doc (the prefix, plus any
            // cursors past the prefix that happened to be at it).
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == pivot_doc {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR via Block-Max MaxScore (BMM).
    ///
    /// Algorithm sketch (Turtle & Flood 1995, Strohman & Croft 2007;
    /// the "Block-Max" augmentation per Petri & Moffat 2017):
    ///
    ///   1. Sort cursors in *descending* `term_max_bm25`.
    ///   2. Compute suffix sums: `partial_max[i] = sum_{j>=i} cursors[j].term_max_bm25`.
    ///   3. Partition into **essential** prefix `cursors[0..f]` and
    ///      **non-essential** suffix `cursors[f..n]` where
    ///      `f = min{ i : partial_max[i] <= threshold }`. A doc that
    ///      appears only in non-essential cursors has max-possible
    ///      score `partial_max[f] <= threshold` and can't make top-k.
    ///   4. Find next candidate doc as the smallest `current_doc_id`
    ///      among essential cursors. (Non-essential cursors are
    ///      skipped *to* the candidate, not iterated for new candidates.)
    ///   5. Apply BMW-style block-skip on the leftmost essential: if
    ///      `leftmost_block_ub + sum_other_term_ubs <= threshold`,
    ///      no doc in the leftmost's current block can beat top-k —
    ///      jump leftmost past its block.
    ///   6. Score: sum essential contributions, then run the
    ///      non-essential loop with **block-level** early termination
    ///      using `current_block_max_bm25` of the remaining cursors.
    ///   7. Update heap; recompute `f` from the new threshold; repeat.
    ///
    /// **When is BMM better than WAND+BMW?** When query terms have
    /// similar upper bounds (3+ same-rank Zipfian terms is the
    /// canonical case) — WAND's pivot moves around because no single
    /// cursor dominates, while MaxScore stably partitions essential
    /// vs non-essential. WAND wins when one term has much higher UB
    /// (rare + common); the partition collapses to a single
    /// essential cursor anyway and WAND's pivot is tighter.
    ///
    /// The router [`Self::dispatch_or_algo`] picks between
    /// the two using a UB-spread heuristic. Both algorithms share
    /// cursor construction via [`Self::build_term_cursors`] so the
    /// router doesn't pay for cursor work twice.
    pub(super) fn run_max_score_bmm(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        self.run_max_score_bmm_range(column_id, cursors, k, 0, u32::MAX, filter, floor_eff)
    }

    /// Multi-term AND via leapfrog intersection over the skip table.
    ///
    /// The smallest-df cursor is the leader: every matching doc must
    /// be in its posting list. For each leader doc, every other
    /// cursor runs `skip_to(candidate)` — a skip-table-driven jump
    /// that decodes at most one block per call (and zero if the
    /// target lies in the already-decoded block). If any cursor
    /// lands past the candidate, that doc isn't in the intersection;
    /// the candidate is bumped to the new high-water mark and the
    /// remaining cursors re-skip. When all cursors converge on the
    /// same doc, the BM25 contribution from each is summed.
    ///
    /// Cost is bounded by `min_df` leader steps × `n_terms` skip_to
    /// calls, with each skip_to a constant-or-O(log) skip-table walk.
    /// The old `run_and` did a full PFOR decode of every term's full
    /// posting list (dominated by the largest list, e.g. ~hundreds of
    /// K postings for a common Zipfian term) followed by a HashMap
    /// intersection — orders of magnitude more work than this when
    /// any term is rare.
    pub(super) fn run_and_intersect(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // Smallest-df cursor at index 0 = leader. The remaining order
        // doesn't matter for correctness but ascending-df reduces the
        // expected number of leapfrog bumps per candidate.
        cursors.sort_by_key(|c| c.block_count());

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut sink = ScoreSink {
            heap: &mut heap,
            k,
            filter,
            floor_eff,
        };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        Ok(drain_top_k_desc(heap))
    }

    /// Ranked must+should walk: the match set is the musts'
    /// intersection (driven by the same flat-merge as
    /// [`run_and_intersect`](Self::run_and_intersect), so the two
    /// always agree on which docs match), and each matching doc's
    /// score additionally collects every should term that lands on it.
    /// Shoulds never affect matching — a doc containing every must and
    /// no should still matches, with its must-only score.
    pub(super) fn run_must_should(
        &self,
        column_id: u32,
        mut must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        debug_assert!(
            !must_cursors.is_empty() && !should_cursors.is_empty(),
            "dispatch routes empty-side shapes to the AND/OR kernels"
        );
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        must_cursors.sort_by_key(|c| c.block_count());

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let should_ub = should_cursors.iter().map(|c| c.term_max_bm25).sum();
        let mut sink = MustShouldSink {
            heap: &mut heap,
            k,
            filter,
            floor_eff,
            shoulds: should_cursors,
            should_ub,
            dl_norm_k1,
        };
        self.and_flat_merge(&mut must_cursors, dl_norm_k1, &mut sink);
        Ok(drain_top_k_desc(heap))
    }

    /// Unranked multi-term AND: the matching doc ids in ascending order
    /// via the block flat-merge in [`and_flat_merge`](Self::and_flat_merge),
    /// with no BM25 scoring and no top-k heap. Because it shares that
    /// traversal with the ranked [`run_and_intersect`](Self::run_and_intersect),
    /// the two always agree on which docs match, and an unranked count
    /// over high-frequency terms costs the same posting-list work as the
    /// ranked search minus the scoring.
    pub(super) fn collect_and_intersect(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
    ) -> Vec<u32> {
        if cursors.is_empty() {
            return Vec::new();
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        cursors.sort_by_key(|c| c.block_count());
        let mut sink = CollectSink { out: Vec::new() };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        sink.out
    }

    /// Unranked multi-term AND **count**: the size of the intersection
    /// via the same flat-merge as [`collect_and_intersect`](Self::collect_and_intersect),
    /// but through a [`CountSink`] that tallies hits instead of
    /// collecting them — no `Vec<u32>` materialized.
    pub(super) fn count_and_intersect(&self, column_id: u32, mut cursors: Vec<TermCursor>) -> u64 {
        if cursors.is_empty() {
            return 0;
        }
        // On a v4 blob a common term's blocks may be bitset-encoded, where
        // decoding (set-bit expansion) is slower than the PFOR path the
        // flat-merge assumes.
        if self.has_bitset_blocks {
            // When even the *rarest* term is dense (covers ≥ 1/DIVISOR of the
            // corpus), the rarest-driven membership walk still iterates a long
            // list. AND the terms' presence bitsets word-at-a-time instead —
            // cost is independent of the terms' lengths. The two full-width
            // bitsets it allocates only pay off at this density, so a sparser
            // intersection keeps the membership probe.
            if cursors.len() >= 2 {
                let max_doc = cursors
                    .iter()
                    .filter_map(|c| c.blocks.last())
                    .map(|b| b.last_doc_id)
                    .max()
                    .unwrap_or(0);
                let min_df = cursors.iter().map(|c| c.df).min().unwrap_or(0);
                if min_df.saturating_mul(OR_COUNT_BITSET_DENSITY_DIVISOR) >= u64::from(max_doc) {
                    return count_and_intersect_bitset(cursors, max_doc);
                }
            }
            // Rarest term is sparse: drive by it and probe the rest by
            // membership — a bitset block answers with an O(1) bit-test, no
            // decode. See `count_and_intersect_membership`.
            return count_and_intersect_membership(cursors);
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        cursors.sort_by_key(|c| c.block_count());
        let mut sink = CountSink { n: 0 };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        sink.n
    }

    /// Dispatch to the 2-term specialization or the general `n >= 3`
    /// (and `n == 1`) flat-merge. The 2-term shape walks the two sorted
    /// `block_doc_ids` arrays with two index pointers instead of calling
    /// `skip_to` per leader doc — removing the function-call +
    /// within-block linear-scan overhead on the hottest AND case
    /// (rare ∧ common). The general path keeps the per-doc leapfrog,
    /// which amortizes well with the block-max pruning a scoring sink
    /// drives.
    pub(super) fn and_flat_merge<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        if cursors.len() == 2 {
            self.and_flat_merge_2term(cursors, dl_norm_k1, sink);
        } else {
            self.and_flat_merge_general(cursors, dl_norm_k1, sink);
        }
    }

    /// General `n >= 3`-term AND path. Same shape as the 2-term path:
    /// block-max pruning at the top, then a flat-merge over the
    /// leader's decoded `block_doc_ids` against each non-leader's
    /// decoded `block_doc_ids`. For each leader doc, every non-leader's
    /// `pos` is advanced with a tight `pos += 1` scan instead of
    /// `skip_to` — no function-call or within-block linear-scan
    /// overhead per leader doc, just integer comparisons over the
    /// already-decoded buffers. When any cursor exhausts its block,
    /// the outer loop crosses blocks via `next()` and re-aligns.
    pub(super) fn and_flat_merge_general<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        'outer: loop {
            if cursors[0].is_exhausted() {
                break;
            }

            // Block-Max-AND pruning (scoring sinks only; unranked `bar()` is
            // NEG_INFINITY). If the tight per-block-pair bound at the leader's
            // doc can't beat the bar, skip to the smallest block boundary
            // across all cursors (past which a bound may rise). See
            // `block_max_and_bound`.
            let bar = sink.bar();
            if bar > f32::NEG_INFINITY {
                let leader_doc = cursors[0].current_doc_id();
                let leader_block_max = cursors[0].current_block_max_bm25();
                let leader_block_end = cursors[0].current_block_last_doc_id();
                let (ub, window_end) = block_max_and_bound(
                    leader_block_max,
                    leader_block_end,
                    &mut cursors[1..],
                    leader_doc,
                );
                if ub <= bar {
                    cursors[0].skip_to(window_end.saturating_add(1));
                    continue;
                }
            }

            // Align every non-leader cursor to >= leader's current doc.
            // Largest landing-doc becomes the new alignment target if
            // any cursor jumped past leader. If any cursor crossed
            // leader's current block, restart the outer loop so pruning
            // re-fires on leader's new block; otherwise the flat-merge
            // proceeds in the current decoded blocks.
            let leader_doc = cursors[0].current_doc_id();
            let leader_block_end = cursors[0].current_block_last_doc_id();
            let mut max_other = leader_doc;
            let mut crossed_block = false;
            for c in cursors[1..].iter_mut() {
                c.skip_to(leader_doc);
                if c.is_exhausted() {
                    break 'outer;
                }
                let here = c.current_doc_id();
                if here > leader_block_end {
                    crossed_block = true;
                }
                if here > max_other {
                    max_other = here;
                }
            }
            if max_other > leader_doc {
                cursors[0].skip_to(max_other);
                if cursors[0].is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat-merge across decoded blocks. Split leader off so
            // both leader and others borrow mutably without overlap;
            // the inner loop reads each cursor's `block_doc_ids` and
            // updates its `pos` directly.
            let (leader_slice, others) = cursors.split_at_mut(1);
            let c0 = &mut leader_slice[0];
            let lb_n = c0.block_n;
            let mut i = c0.pos;
            while i < lb_n {
                let a = c0.block_doc_ids[i];

                // For each non-leader, walk its `pos` forward through
                // the decoded block until block_doc_ids[pos] >= a (or
                // the block exhausts). If any block exhausts, break
                // out to the outer loop's block-crossing step. If any
                // cursor lands above `a`, the leader doc isn't in the
                // intersection — advance leader only.
                let mut block_exhausted = false;
                let mut all_match = true;
                for o in others.iter_mut() {
                    while o.pos < o.block_n && o.block_doc_ids[o.pos] < a {
                        o.pos += 1;
                    }
                    if o.pos >= o.block_n {
                        block_exhausted = true;
                        break;
                    }
                    if o.block_doc_ids[o.pos] != a {
                        all_match = false;
                        break;
                    }
                }
                if block_exhausted {
                    break;
                }
                if all_match {
                    let score = if sink.needs_score() {
                        let norm = dl_norm_k1.get(a);
                        let mut score =
                            bm25::score_with_dl_norm_k1(c0.idf_x_k1p1, c0.block_tfs[i], norm);
                        for o in others.iter() {
                            score +=
                                bm25::score_with_dl_norm_k1(o.idf_x_k1p1, o.block_tfs[o.pos], norm);
                        }
                        score
                    } else {
                        0.0
                    };
                    sink.emit(a, score);
                    i += 1;
                    for o in others.iter_mut() {
                        o.pos += 1;
                    }
                } else {
                    i += 1;
                }
            }
            c0.pos = i;

            // Cross blocks for whichever cursors exhausted. The outer
            // loop's alignment step re-pulls everyone to the new leader
            // doc on the next iteration.
            if c0.pos >= c0.block_n {
                c0.next();
            }
            for o in others.iter_mut() {
                if o.pos >= o.block_n {
                    o.next();
                }
            }
        }
    }

    /// 2-term specialization. While both cursors share a doc-id region
    /// covered by their respective decoded blocks, do a flat
    /// sorted-merge over the two `block_doc_ids` arrays: no `skip_to`
    /// function calls per leader doc, no per-doc within-block linear
    /// scan — just two index pointers walking forward. When either
    /// block exhausts, the cursor crosses to its next block (decoding
    /// on demand) and the merge resumes.
    pub(super) fn and_flat_merge_2term<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        debug_assert_eq!(cursors.len(), 2);
        // Split into two simultaneous mutable refs so the inner loop
        // can read both cursors' decoded buffers and update both
        // positions without borrow-checker contortions.
        let (left, right) = cursors.split_at_mut(1);
        let c0 = &mut left[0];
        let c1 = &mut right[0];

        'outer: loop {
            if c0.is_exhausted() || c1.is_exhausted() {
                break;
            }

            // Block-Max-AND pruning (scoring sinks only). Bound `c1` by the
            // single block covering the leader doc; if it can't beat the bar,
            // skip to the nearer block boundary. See `block_max_and_bound`.
            let bar = sink.bar();
            if bar > f32::NEG_INFINITY {
                let leader_doc = c0.current_doc_id();
                let (ub, window_end) = block_max_and_bound(
                    c0.current_block_max_bm25(),
                    c0.current_block_last_doc_id(),
                    from_mut(c1),
                    leader_doc,
                );
                if ub <= bar {
                    c0.skip_to(window_end.saturating_add(1));
                    continue;
                }
            }

            // Align c1 with c0 at the current leader doc. After this
            // call both cursors are positioned on doc_ids >= leader.
            // If c1 jumped past the leader's current block we'll bump
            // the leader via the outer loop's next iteration.
            c1.skip_to(c0.current_doc_id());
            if c1.is_exhausted() {
                break 'outer;
            }
            // If c1 sits above c0's pos, pull c0 forward to align.
            // When that pull crosses c0's current block, restart the
            // outer loop so pruning re-fires on c0's new block;
            // otherwise fall through and let the flat-merge handle
            // the within-block divergence inline.
            if c1.current_doc_id() > c0.current_doc_id() {
                let crossed_block = c1.current_doc_id() > c0.current_block_last_doc_id();
                c0.skip_to(c1.current_doc_id());
                if c0.is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat sorted-merge within the overlap of the two decoded
            // blocks. Pre-load all locals; the borrow checker is
            // satisfied because c0/c1 are independently mutable refs.
            let lb_n = c0.block_n;
            let rb_n = c1.block_n;
            let mut i = c0.pos;
            let mut j = c1.pos;
            let c0_idf = c0.idf_x_k1p1;
            let c1_idf = c1.idf_x_k1p1;
            while i < lb_n && j < rb_n {
                let a = c0.block_doc_ids[i];
                let b = c1.block_doc_ids[j];
                if a < b {
                    i += 1;
                } else if a > b {
                    j += 1;
                } else {
                    let score = if sink.needs_score() {
                        let norm = dl_norm_k1.get(a);
                        bm25::score_with_dl_norm_k1(c0_idf, c0.block_tfs[i], norm)
                            + bm25::score_with_dl_norm_k1(c1_idf, c1.block_tfs[j], norm)
                    } else {
                        0.0
                    };
                    sink.emit(a, score);
                    i += 1;
                    j += 1;
                }
            }
            c0.pos = i;
            c1.pos = j;

            // Whichever cursor exhausted its block crosses to its next
            // block; the other holds. The outer loop re-checks
            // is_exhausted and re-aligns on the next iteration.
            if i >= lb_n {
                c0.next();
            }
            if j >= rb_n {
                c1.next();
            }
        }
    }

    /// MaxScore+BMM constrained to the doc_id half-open range
    /// `[doc_id_start, doc_id_end)`. Used by the supertable layer's
    /// intra-superfile parallel fan-out: when the reader pool has more
    /// threads than superfiles, each superfile is split into N sub-ranges
    /// and the per-sub-range searches run in parallel, each producing
    /// its own top-K heap that the caller merges.
    ///
    /// Setting `doc_id_start == 0` and `doc_id_end == u32::MAX`
    /// reproduces the un-ranged BMM walk byte-for-byte (the seek is
    /// a no-op and the upper-bound check trivially never fires).
    ///
    /// **Pruning trade**: each sub-range maintains an independent
    /// top-K heap + BMM threshold. The threshold tightens slower than
    /// in the un-ranged walk because each sub-range sees only `1/N`
    /// of the docs, so the per-sub-range BMW block-skip fires less
    /// aggressively. Net wall-time win comes from spreading the
    /// scoring work across more cores; the per-sub-range work loss
    /// from looser pruning is bounded by the bookkeeping path (and
    /// in practice ~10–20% of single-thread serial), well below the
    /// 2× cores-doubled headroom.
    pub(super) fn run_max_score_bmm_range(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // Sub-range seek: jump every cursor past any doc_id below
        // the lower bound. Cursors already past the bound stay where
        // they are; cursors whose entire posting list sits below the
        // bound become exhausted. The skip_to walks the skip-table
        // (cross-block) when needed, so we don't decode blocks we'll
        // never score.
        if doc_id_start > 0 {
            for cursor in &mut cursors {
                cursor.skip_to(doc_id_start);
            }
        }

        // Sort descending by term-max UB. Stability isn't required —
        // ties (equal `term_max_bm25` across terms) are rare and the
        // tie-break is arbitrary as long as the prefix-sum invariant
        // holds.
        cursors.sort_unstable_by(|a, b| {
            b.term_max_bm25
                .partial_cmp(&a.term_max_bm25)
                .unwrap_or(Ordering::Equal)
        });

        // Suffix sums of term_max_bm25. partial_max[0] = total UB,
        // partial_max[n] = 0. Monotonically decreasing.
        let n = cursors.len();
        let mut partial_max = vec![0.0_f32; n + 1];
        for i in (0..n).rev() {
            partial_max[i] = partial_max[i + 1] + cursors[i].term_max_bm25;
        }

        // Sized by this slice's own window: only docs inside it can be
        // ranked here, and a sliced fan-out would otherwise preallocate
        // one whole-superfile heap per slice.
        let initial_cap =
            top_k_initial_capacity(k, u64::from(self.n_docs), Some((doc_id_start, doc_id_end)));
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        // Seed the pruning threshold with the caller's floor: docs
        // strictly below it can never matter, so the MaxScore
        // machinery (essential boundary, block skips, heap admission)
        // starts from the floor instead of from zero. BM25 scores are
        // positive, so an unfloored run keeps the original 0.0 seed.
        let mut threshold: f32 = floor_eff.max(0.0);

        let recompute_f = |partial_max: &[f32], threshold: f32| -> usize {
            // Essential boundary: smallest f such that
            // partial_max[f] ≤ threshold. Linear scan from the front —
            // for typical N ≤ 8 query terms this is cheaper than a
            // binary search's branch-and-bound overhead.
            let mut f = 0;
            while f < partial_max.len() - 1 && partial_max[f] > threshold {
                f += 1;
            }
            f
        };
        // With a zero threshold only partial_max[n]=0 satisfies, so
        // f=n (all terms essential); a seeded floor can already shrink
        // the essential set before the first doc is scored.
        let mut f_essential: usize = recompute_f(&partial_max, threshold);

        // Total term-level UB. Used for the block-skip bound on
        // essential cursors below.
        let total_term_ub = partial_max[0];

        loop {
            // **f=1 block-batch fast path.** Once threshold rises
            // enough that only `cursors[0]` (highest term_max) is
            // essential, the candidate set is *exactly* `cursors[0]`'s
            // posting list. We can decode one of its blocks and
            // process every doc in the block inline — no per-doc
            // pivot search, no per-doc cursor sort. The outer loop's
            // overhead amortizes over ~128 docs per block instead of
            // 1 doc per iteration. This is the steady state for
            // dominator queries (wide-UB) and for similar-UB queries
            // after the heap fills with multi-term hits.
            if f_essential == 1 {
                if cursors[0].is_exhausted() || cursors[0].current_doc_id() >= doc_id_end {
                    break;
                }
                // Block-skip: if `block_max + sum_others_term_max`
                // can't beat threshold, skip the block.
                let block_ub = cursors[0].current_block_max_bm25()
                    + (total_term_ub - cursors[0].term_max_bm25);
                if block_ub <= threshold {
                    let end = cursors[0].current_block_last_doc_id();
                    cursors[0].skip_to(end.saturating_add(1));
                    continue;
                }

                let block_end = cursors[0].current_block_last_doc_id();
                let mut f_changed = false;
                // Per-doc UB tightening: bound this doc's max possible
                // score by `essential_score + sum_others_term_max`.
                // If even this can't beat the heap threshold, skip
                // the non-essential lookups + heap update entirely
                // — those are the dominant per-doc cost. Only docs
                // where the essential alone is "in striking distance"
                // pay the full lookup price.
                let others_term_ub = total_term_ub - cursors[0].term_max_bm25;
                while !cursors[0].is_exhausted()
                    && cursors[0].current_doc_id() <= block_end
                    && cursors[0].current_doc_id() < doc_id_end
                {
                    let candidate = cursors[0].current_doc_id();
                    // Drop docs excluded by a negated term (None = keep
                    // all): skip without scoring.
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(candidate)
                    {
                        cursors[0].next();
                        continue;
                    }
                    let norm = dl_norm_k1.get(candidate);
                    let essential_score = bm25::score_with_dl_norm_k1(
                        cursors[0].idf_x_k1p1,
                        cursors[0].current_tf(),
                        norm,
                    );
                    if essential_score + others_term_ub <= threshold {
                        // No combination of non-essential
                        // contributions at `candidate` can push it
                        // above threshold. Skip lookup + heap.
                        cursors[0].next();
                        continue;
                    }
                    // SIMD-pack non-essentials at `candidate`.
                    let mut idfs = [cursors[0].idf_x_k1p1, 0.0, 0.0, 0.0];
                    let mut tfs = [cursors[0].current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(1) {
                        cursor.skip_to(candidate);
                        if cursor.current_doc_id() == candidate {
                            idfs[packed] = cursor.idf_x_k1p1;
                            tfs[packed] = cursor.current_tf() as f32;
                            packed += 1;
                            if packed == 4 {
                                score += bm25::score_simd_x4(idfs, tfs, norm);
                                idfs = [0.0; 4];
                                tfs = [0.0; 4];
                                packed = 0;
                            }
                        }
                    }
                    if packed > 0 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                    }

                    if heap.len() < k {
                        heap.push(TopKEntry(score, candidate));
                        if heap.len() == k {
                            // max(): a seeded floor must never be
                            // lowered by a weaker local kth-best.
                            threshold = heap.peek().expect("non-empty").0.max(threshold);
                            let new_f = recompute_f(&partial_max, threshold);
                            if new_f != f_essential {
                                f_essential = new_f;
                                f_changed = true;
                            }
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, candidate));
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                        let new_f = recompute_f(&partial_max, threshold);
                        if new_f != f_essential {
                            f_essential = new_f;
                            f_changed = true;
                        }
                    }

                    cursors[0].next();

                    if f_changed {
                        break;
                    }
                }
                continue;
            }

            // Pick the next candidate doc: smallest current_doc_id
            // among essential cursors. (Non-essential cursors only
            // get probed via skip_to once we have a candidate.)
            // Specialized for f=2 (the most common steady state for
            // similar-UB queries) to avoid the iter loop overhead.
            let (candidate, leftmost_essential) = if f_essential == 2 {
                let d0 = cursors[0].current_doc_id();
                let d1 = cursors[1].current_doc_id();
                if d0 == u32::MAX && d1 == u32::MAX {
                    break;
                }
                if d0 <= d1 { (d0, 0) } else { (d1, 1) }
            } else {
                let mut candidate = u32::MAX;
                let mut leftmost_essential: usize = 0;
                for (i, cursor) in cursors.iter().take(f_essential).enumerate() {
                    let d = cursor.current_doc_id();
                    if d < candidate {
                        candidate = d;
                        leftmost_essential = i;
                    }
                }
                if candidate == u32::MAX {
                    break;
                }
                (candidate, leftmost_essential)
            };
            // Sub-range upper bound: every subsequent candidate is
            // monotonically increasing, so once we cross the bound
            // there's no work left for this sub-range.
            if candidate >= doc_id_end {
                break;
            }

            // **BMW-style block-skip on the leftmost essential.** Bound
            // the score of any doc in `leftmost_essential`'s current
            // block by `current_block_max + sum_of_other_term_UBs`. If
            // that bound can't beat the threshold, no doc in this
            // block can make top-k — skip the cursor past its current
            // block. This is what makes BMM competitive with WAND+BMW
            // on dominant-term queries; without it MaxScore scans
            // every doc in the dominant term's posting list.
            let leftmost_term_ub = cursors[leftmost_essential].term_max_bm25;
            let leftmost_block_ub = cursors[leftmost_essential].current_block_max_bm25();
            // others_ub = sum of OTHER cursors' term UBs (essential + non-essential).
            // We use term-level UBs for the others as a conservative bound; using
            // their per-block UBs would tighten further but require keeping them
            // synced with the candidate, which we only do lazily in the
            // non-essential probe below.
            let others_ub = total_term_ub - leftmost_term_ub;
            if leftmost_block_ub + others_ub <= threshold {
                let last_in_block = cursors[leftmost_essential].current_block_last_doc_id();
                cursors[leftmost_essential].skip_to(last_in_block.saturating_add(1));
                continue;
            }

            // Drop docs excluded by a negated term before scoring —
            // the non-essential probes below are the dominant per-doc
            // cost and an excluded doc can never enter the heap. The
            // essential-cursor advance after this block still runs, so
            // the walk progresses.
            let admitted = match filter.as_deref_mut() {
                Some(f) => f.admits(candidate),
                None => true,
            };
            if admitted {
                // Score essential contributions at the candidate doc.
                // SIMD-pack up to 4 cursors per scoring call. (Essential
                // scoring has no early-bail; non-essential scoring below
                // does, so it stays scalar to keep `score` always
                // up-to-date for the bail check.)
                let norm = dl_norm_k1.get(candidate);
                let mut score: f32 = 0.0;
                let mut idfs = [0.0_f32; 4];
                let mut tfs = [0.0_f32; 4];
                let mut packed = 0;
                for cursor in cursors.iter().take(f_essential) {
                    if cursor.current_doc_id() == candidate {
                        idfs[packed] = cursor.idf_x_k1p1;
                        tfs[packed] = cursor.current_tf() as f32;
                        packed += 1;
                        if packed == 4 {
                            score += bm25::score_simd_x4(idfs, tfs, norm);
                            idfs = [0.0; 4];
                            tfs = [0.0; 4];
                            packed = 0;
                        }
                    }
                }
                if packed > 0 {
                    score += bm25::score_simd_x4(idfs, tfs, norm);
                }

                // Per-doc UB tightening: bound the doc's max possible
                // score by `essential_score + sum_non_essentials_term_max`.
                // If even this can't beat threshold, skip the
                // non-essential probe + heap update entirely. This is
                // looser than the per-non-essential block_ub bound below
                // but spares the `skip_to` cursor advances themselves —
                // those are the dominant per-doc cost.
                let non_essentials_term_ub = partial_max[f_essential];
                if score + non_essentials_term_ub > threshold {
                    // Tighter pre-bail using non-essential block_max
                    // (which is tighter than term_max). Use shallow
                    // advance — moves the lightweight inspect-block
                    // pointer to candidate's block without decoding,
                    // amortized O(1). If even this tighter UB can't beat
                    // threshold, skip the deep skip_to pass entirely.
                    let mut remaining_block_ub: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(f_essential) {
                        cursor.shallow_advance_block_to(candidate);
                        remaining_block_ub += cursor.inspect_block_max_bm25();
                    }

                    if score + remaining_block_ub > threshold {
                        for cursor in cursors.iter_mut().skip(f_essential) {
                            let block_ub = cursor.inspect_block_max_bm25();
                            if score + remaining_block_ub <= threshold {
                                break;
                            }
                            cursor.skip_to(candidate);
                            if cursor.current_doc_id() == candidate {
                                score += bm25::score_with_dl_norm_k1(
                                    cursor.idf_x_k1p1,
                                    cursor.current_tf(),
                                    norm,
                                );
                            }
                            remaining_block_ub -= block_ub;
                        }
                    }
                }
                // (If essential score + remaining_block_ub already ≤ threshold,
                // we don't bother scoring non-essentials — the doc can't beat
                // the kth-best.)

                // Update heap. `threshold` is kept in sync with
                // heap.peek().0 every time we mutate the heap, so we can
                // gate the replace-or-skip decision against the local
                // f32 instead of paying for a heap.peek() per iter.
                // (max(): a seeded floor must never be lowered by a
                // weaker local kth-best.)
                if heap.len() < k {
                    heap.push(TopKEntry(score, candidate));
                    if heap.len() == k {
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                        f_essential = recompute_f(&partial_max, threshold);
                    }
                } else if score > threshold {
                    heap.pop();
                    heap.push(TopKEntry(score, candidate));
                    threshold = heap.peek().expect("non-empty").0.max(threshold);
                    f_essential = recompute_f(&partial_max, threshold);
                }
            }

            // Advance every essential cursor that was at the candidate
            // doc. (Non-essential cursors stay where skip_to landed
            // them; the next iteration's skip_to will move them as
            // needed for the next candidate.)
            for cursor in cursors.iter_mut().take(f_essential) {
                if cursor.current_doc_id() == candidate {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Windowed union scorer for multi-term OR — the fast path for
    /// uniform-upper-bound / common-term ORs, where MaxScore can't prune
    /// and degrades to scoring the whole union with per-doc f-way merge
    /// overhead.
    ///
    /// Walks the doc-id space one `OR_WINDOW`-doc window at a time. Within
    /// a window each cursor streams its postings **sequentially**,
    /// accumulating its BM25 contribution into `scores[doc - base]` and
    /// marking a presence bit — no per-doc min-scan across cursors, no
    /// heap touch during accumulation. The window is then drained in
    /// ascending doc order (bit-trick over the presence bitset) and each
    /// distinct matching doc is offered to the top-k heap once. Empty
    /// windows are skipped (the base jumps to the next live doc), so a
    /// sparse union costs only its non-empty windows.
    ///
    /// **Exact top-k:** same result set/order as [`Self::run_max_score_bmm`]
    /// — same heap-admission rule (`score > threshold`, floor-seeded), same
    /// `(score desc, doc asc)` tie-break, docs offered in ascending order.
    /// The one nuance is summation *order*: contributions are summed
    /// term-major here vs. per-doc-major in MaxScore, and f32 add is
    /// non-associative, so a score can differ by ≤1 ULP. Validated against
    /// the brute-force BM25 oracle; if a boundary tie ever flips, the
    /// accumulator would move to f64.
    ///
    /// Negation: the [`ExcludeFilter`] is applied at **drain** (globally
    /// ascending → satisfies its monotonic-feed contract), never during the
    /// term-major accumulation.
    pub(super) fn run_windowed_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // A top-0 request admits nothing. Guard here too (callers already
        // short-circuit) so the heap-admission `else if` below can never
        // run against an empty heap.
        if k == 0 {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        if doc_id_start > 0 {
            for c in &mut cursors {
                c.skip_to(doc_id_start);
            }
        }

        // This scan's own window, as in the MaxScore path. Un-ranged
        // callers pass `[0, u32::MAX)`, which the `n_docs` cap collapses
        // back to a whole-superfile scope.
        let initial_cap =
            top_k_initial_capacity(k, u64::from(self.n_docs), Some((doc_id_start, doc_id_end)));
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        // Floor-seeded threshold, identical to the MaxScore path.
        let mut threshold: f32 = floor_eff.max(0.0);

        // Per-window state, allocated once and reused across windows.
        // Cleared lazily during the drain (only touched slots), so reset
        // cost is proportional to matches, not to OR_WINDOW.
        let mut scores = vec![0.0f32; OR_WINDOW as usize];
        let mut present = [0u64; OR_WINDOW_WORDS];

        loop {
            // Next non-empty window: smallest current doc among live
            // cursors, aligned down to a window boundary. O(f) per window
            // (not per doc) — this replaces MaxScore's per-doc min-scan.
            let mut min_doc = u32::MAX;
            for c in &cursors {
                if !c.is_exhausted() {
                    min_doc = min_doc.min(c.current_doc_id());
                }
            }
            if min_doc == u32::MAX || min_doc >= doc_id_end {
                break;
            }
            let base = min_doc & !(OR_WINDOW - 1);
            // saturating: a doc id within OR_WINDOW of u32::MAX would
            // overflow `base + OR_WINDOW` (panic in debug; wrap in release,
            // which makes window_end < base → the accumulate loop stalls and
            // the outer loop spins). Saturate, then clamp to doc_id_end.
            let window_end = base.saturating_add(OR_WINDOW).min(doc_id_end);

            // Accumulate each cursor's contributions in [base, window_end).
            // Sequential walk per cursor; `d - base` is in range because
            // every live cursor sits at `>= min_doc >= base`.
            for c in &mut cursors {
                while !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d >= window_end {
                        break;
                    }
                    let pos = c.pos;
                    if pos + bm25::SCORE_SIMD_LANES <= c.block_n {
                        let doc_ids = [
                            c.block_doc_ids[pos],
                            c.block_doc_ids[pos + 1],
                            c.block_doc_ids[pos + 2],
                            c.block_doc_ids[pos + 3],
                        ];
                        if doc_ids[bm25::SCORE_SIMD_LANES - 1] < window_end {
                            let contributions = bm25::score_one_term_x4(
                                c.idf_x_k1p1,
                                [
                                    c.block_tfs[pos],
                                    c.block_tfs[pos + 1],
                                    c.block_tfs[pos + 2],
                                    c.block_tfs[pos + 3],
                                ],
                                [
                                    dl_norm_k1.get(doc_ids[0]),
                                    dl_norm_k1.get(doc_ids[1]),
                                    dl_norm_k1.get(doc_ids[2]),
                                    dl_norm_k1.get(doc_ids[3]),
                                ],
                            );
                            for lane in 0..bm25::SCORE_SIMD_LANES {
                                let local = (doc_ids[lane] - base) as usize;
                                scores[local] += contributions[lane];
                                present[local >> 6] |= 1u64 << (local & 63);
                            }
                            c.advance_by(bm25::SCORE_SIMD_LANES);
                            continue;
                        }
                    }
                    let local = (d - base) as usize;
                    scores[local] += bm25::score_with_dl_norm_k1(
                        c.idf_x_k1p1,
                        c.current_tf(),
                        dl_norm_k1.get(d),
                    );
                    present[local >> 6] |= 1u64 << (local & 63);
                    c.next();
                }
            }

            // Drain ascending; clear touched slots for reuse; apply
            // negation; offer to the heap.
            for (word_idx, word) in present.iter_mut().enumerate() {
                let mut bits = *word;
                *word = 0;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let local = (word_idx << 6) | b;
                    let score = scores[local];
                    scores[local] = 0.0;
                    let doc = base + local as u32;
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(doc)
                    {
                        continue;
                    }
                    if heap.len() < k {
                        heap.push(TopKEntry(score, doc));
                        if heap.len() == k {
                            threshold = heap.peek().expect("non-empty").0.max(threshold);
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, doc));
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                    }
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Exhaustive union walk for multi-term OR. No threshold-driven
    /// block skipping — every doc in the union of the cursor postings
    /// is scored and offered to the top-K heap.
    ///
    /// **Not on the production path.** `dispatch_or_algo` routes
    /// to MaxScore+BMM or the windowed union; this function is reachable
    /// only via `search_with_algo_for_bench(OrAlgo::Exhaustive)`. It exists
    /// because the supertable bench surfaced one specific shape where
    /// it narrowly wins, and we want the option available for future
    /// re-routing work without re-implementing it.
    ///
    /// **When this can beat BMM (measured at 10M × 8 superfiles)**:
    /// - **Prefix expansions over very-rare terms, in parallel mode.**
    ///   E.g., `term0009*` expanding to 10 terms at Zipfian rank
    ///   90–99 (df ≈ 0.1% each). On the supertable parallel bench,
    ///   exhaustive ran at 40.2 ms vs BMM's 54.0 ms — a 26% win. The
    ///   per-superfile work is tiny (∼12 K matching docs across 10
    ///   short cursors) so BMM's per-block bookkeeping
    ///   (`f_essential` recomputation, `shallow_advance_block_to`,
    ///   `inspect_block_max_bm25`) dominates over actual scoring
    ///   work.
    ///
    /// **When BMM is strictly better — measured regressions if we
    /// route to exhaustive**:
    /// - **Mid-rank uniform-UB queries.** Five terms at rank 50–54
    ///   (df ≈ 0.4% each): exhaustive serial 174 ms vs BMM 99 ms —
    ///   a **76% regression**. Three terms at rank 50–52: exhaustive
    ///   serial 93 ms vs BMM 61 ms — a **52% regression**. Enough
    ///   matching docs exist that BMM's skip-pruning actually fires
    ///   and amortizes its bookkeeping.
    /// - **Any dominant-term query.** BMM's `f_essential == 1` fast
    ///   path collapses to a block-batch loop on the dominant
    ///   cursor's postings — about as tight as exhaustive could be,
    ///   and with skip on top.
    /// - **Single-term queries.** Don't go through OR dispatch
    ///   anyway; `search_single_term_bmw` handles them.
    ///
    /// **Routing heuristic if revisited**: the obvious-looking
    /// `max(term_max_bm25) / sum(term_max_bm25) < 1.5/n_cursors`
    /// (uniform UB) **over-routes** because it admits mid-rank
    /// queries where BMM wins. A better rule would gate on
    /// *absolute* low total df **and** uniform UB — e.g.,
    /// `σdf < n_docs / 100 AND max_ub/sum_ub < 1.5/n_cursors`.
    /// Empirically that admits the prefix-of-rare-terms shape and
    /// excludes the mid-rank multi-term shapes. Not yet wired up:
    /// the single-query parallel win (26% on prefix) hasn't
    /// justified the routing-heuristic maintenance cost yet.
    ///
    /// Algorithm: classic k-way merge over `TermCursor`s. Each
    /// iteration finds the smallest current `doc_id` among live
    /// cursors, sums BM25 contributions from all cursors at that
    /// doc, advances those cursors, pushes into the top-K min-heap.
    ///
    /// Result invariants match [`Self::run_max_score_bmm`]: top-k by
    /// descending BM25 score, ties broken by ascending doc_id.
    pub(super) fn run_exhaustive_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        loop {
            // Find smallest current doc_id across all live cursors —
            // the next candidate to score. Exhausted cursors report
            // `u32::MAX`, which can't be smaller than any live cursor's
            // doc_id, so this terminates naturally when every cursor
            // has been drained.
            let mut candidate = u32::MAX;
            for cursor in &cursors {
                let d = cursor.current_doc_id();
                if d < candidate {
                    candidate = d;
                }
            }
            if candidate == u32::MAX {
                break;
            }

            // Score: sum BM25 from every cursor positioned at the
            // candidate doc. Pack up to 4 cursors per SIMD scoring
            // call, matching the BMM essential-scoring shape.
            let norm = dl_norm_k1.get(candidate);
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == candidate {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                    cursor.next();
                }
            }
            if packed > 0 {
                score += bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Top-K update. `threshold` mirrors `heap.peek().0` so
            // the replace-or-skip branch doesn't re-peek per iter.
            if heap.len() < k {
                heap.push(TopKEntry(score, candidate));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if score > threshold {
                heap.pop();
                heap.push(TopKEntry(score, candidate));
                threshold = heap.peek().expect("non-empty").0;
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR dispatch. Routes everything to MaxScore+BMM.
    ///
    /// **Routing decision (1M docs — head-to-head WAND+BMW vs MaxScore+BMM):**
    ///
    /// | Query shape                                 | WAND+BMW | MaxScore+BMM |
    /// |---|---|---|
    /// | two-term wide (rank 1 + 50)                 | 1.25 ms  | **0.28 ms**  |
    /// | three-term wide (rank 1 + 50 + 100)         | 17.2 ms  | 18.3 ms      |
    /// | three-term similar UBs (rank 50/51/52)      | 28.3 ms  | **24.7 ms**  |
    /// | five-term similar UBs (rank 50–54)          | 59.1 ms  | **55.1 ms**  |
    ///
    /// BMM wins on most shapes once we have:
    ///   1. A precomputed per-doc length-norm table (no per-call
    ///      `dl/avgdl` work in scoring).
    ///   2. SIMD x4 scoring of all aligned cursors per doc.
    ///   3. A block-batch fast path when only one cursor is essential
    ///      (`f_essential == 1`) — the steady state for wide-UB and
    ///      heap-warmed similar-UB queries.
    ///
    /// **Exhaustive union walk** ([`Self::run_exhaustive_union`]) is
    /// implemented and reachable via `search_with_algo_for_bench`,
    /// but the dispatcher does NOT route to it. Empirically it
    /// regressed mid-rank uniform-UB shapes by 50–80% — see
    /// `run_exhaustive_union`'s doc comment for the cost model and
    /// the one shape (prefix-of-very-rare-terms in parallel mode)
    /// where it narrowly wins. WAND+BMW remains in the codebase
    /// for the same reason — bench-harness comparison only.
    pub(super) fn dispatch_or_algo(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // Route on upper-bound *spread*, not term count: when no single
        // term dominates, MaxScore's essential set never shrinks and it
        // degrades to scoring the whole union with per-doc f-way merge
        // overhead — the windowed union scorer is dramatically faster
        // there. A dominant-term query stays on MaxScore, which prunes
        // hard (its block-skip / f→1 fast path); windowing would lose by
        // scoring every windowed doc.
        // A 2-term OR of one rare + one common term is a worst case for
        // MaxScore: it scores the common term's long posting list end to
        // end. WAND+BMW pivots on the rare (short) term and skips most of
        // the common term's list — a large win. For two comparable-length
        // common terms there is no short anchor to skip on, so WAND's
        // per-iteration cursor re-sort just adds overhead and MaxScore
        // wins; those fall through. Route 2-term ORs to WAND only when
        // (a) one list is much shorter than the other (df ratio,
        // `two_term_has_rare_anchor`); (b) k is small — at large k the
        // top-k threshold is too low for WAND to prune; (c) no negation —
        // `run_wand_bmw` applies no exclude filter; and (d) no
        // cross-segment floor (`floor_eff` unset) — seeding WAND's
        // threshold from a floor mis-prunes, so a live floor stays on
        // MaxScore.
        // A 2-term rare+common OR goes to WAND+BMW (it pivots on the rare
        // term); otherwise MaxScore by default, and the windowed scan only
        // where pruning is dead — see `route_or_to_windowed`. WAND takes no
        // filter or floor, so a negated / floored query skips it.
        let no_floor = floor_eff == f32::NEG_INFINITY;
        if cursors.len() == 2
            && k <= WAND_BMW_2TERM_MAX_K
            && filter.is_none()
            && no_floor
            && two_term_has_rare_anchor(&cursors)
        {
            self.run_wand_bmw(column_id, cursors, k)
        } else if route_or_to_windowed(&cursors, k) {
            self.run_windowed_union(column_id, cursors, k, filter, floor_eff, 0, u32::MAX)
        } else {
            self.run_max_score_bmm(column_id, cursors, k, filter, floor_eff)
        }
    }

    /// Bench/dev helper: force the multi-term OR path to use a specific
    /// algorithm regardless of the dispatcher's heuristic. Used by the
    /// superfile tier's per-algorithm probes
    /// (`benches/utils/superfile.rs`) to compare WAND+BMW, MaxScore+BMM,
    /// and the windowed union under identical inputs so the dispatch
    /// thresholds are validated against measured numbers every run.
    ///
    /// **Not part of the stable API** — production code should use
    /// `search`, which routes through `dispatch_or_algo`.
    #[doc(hidden)]
    pub async fn search_with_algo_for_bench(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        algo: OrAlgo,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let cursors = self
            .build_term_cursors(column_id, terms, None, false)
            .await?;
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        // Bench-only selector; never carries negation or a floor.
        match algo {
            OrAlgo::Bmm => self.run_max_score_bmm(column_id, cursors, k, None, f32::NEG_INFINITY),
            OrAlgo::WandBmw => self.run_wand_bmw(column_id, cursors, k),
            OrAlgo::Exhaustive => self.run_exhaustive_union(column_id, cursors, k),
            OrAlgo::Windowed => {
                self.run_windowed_union(column_id, cursors, k, None, f32::NEG_INFINITY, 0, u32::MAX)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder, reader::BoolMode, tokenize::AsciiLowerTokenizer,
    };

    #[tokio::test]
    async fn token_match_doc_set_matches_bm25_for_same_terms() {
        // token_match(Or) must return exactly the doc set bm25 ranks.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let mut bm25: Vec<u32> = r
            .search("body", &["rust", "java"], 10, BoolMode::Or)
            .await
            .expect("search")
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        bm25.sort_unstable();
        let boolean = r
            .token_match("body", &["rust", "java"], BoolMode::Or)
            .await
            .expect("boolean")
            .0;
        assert_eq!(bm25, boolean, "boolean Or doc set == bm25 doc set");
    }

    #[tokio::test]
    async fn exhaustive_and_bmm_agree_on_top_k() {
        // Build a larger blob so multi-term OR queries are
        // interesting (some docs have multiple terms, some have one).
        // Both algorithms must return identical top-K (descending
        // score, ascending doc_id tiebreak).
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        // 20 docs sprinkled with mixed term combinations.
        let docs = [
            "alpha",
            "beta",
            "gamma",
            "alpha beta",
            "alpha gamma",
            "beta gamma",
            "alpha beta gamma",
            "delta",
            "epsilon",
            "alpha delta",
            "beta epsilon",
            "gamma delta",
            "alpha beta delta",
            "alpha epsilon gamma",
            "delta epsilon",
            "alpha alpha alpha",
            "beta beta beta",
            "gamma gamma",
            "alpha beta gamma delta epsilon",
            "epsilon",
        ];
        for (i, text) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Three terms with similar UBs — the heuristic should pick
        // exhaustive for this shape, but we cross-check by calling
        // both paths directly via the bench harness.
        let terms: &[&str] = &["alpha", "beta", "gamma"];
        let bmm = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Bmm)
            .await
            .expect("bmm");
        let exh = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Exhaustive)
            .await
            .expect("exhaustive");
        assert_eq!(bmm.len(), exh.len(), "result length mismatch");
        for ((d_bmm, s_bmm), (d_exh, s_exh)) in bmm.iter().zip(exh.iter()) {
            assert_eq!(d_bmm, d_exh, "doc_id mismatch");
            assert!(
                (s_bmm - s_exh).abs() < 1e-4,
                "score mismatch: bmm={s_bmm} exhaustive={s_exh}"
            );
        }
    }

    #[tokio::test]
    async fn search_with_algo_wand_bmw_agrees_with_bmm() {
        // The historical WAND+BMW baseline must agree with the production
        // BMM path on the planted corpus.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        let docs = [
            "alpha beta",
            "alpha",
            "beta gamma",
            "alpha beta gamma",
            "gamma",
            "alpha gamma",
            "beta",
            "alpha beta gamma",
        ];
        for (i, t) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, t).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let terms: &[&str] = &["alpha", "beta", "gamma"];
        let bmm = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Bmm)
            .await
            .expect("bmm");
        let wand = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::WandBmw)
            .await
            .expect("wand");
        assert_eq!(bmm.len(), wand.len());
        for ((db, sb), (dw, sw)) in bmm.iter().zip(wand.iter()) {
            assert_eq!(db, dw, "doc_id mismatch");
            assert!((sb - sw).abs() < 1e-4, "score mismatch {sb} vs {sw}");
        }
    }

    #[tokio::test]
    async fn wand_bmw_exercises_block_skips_on_multi_block_lists() {
        // A corpus large enough that the common terms span several
        // 128-doc posting blocks, with five query terms of differing
        // document frequency and a handful of docs carrying all five.
        // Running WAND+BMW at a small k forces the pivot to move, the
        // block-upper-bound skip to fire, lagging cursors to re-align,
        // and the 4-wide SIMD scoring pack to be used on the
        // all-terms docs — then cross-checks the result against BMM.

        /// Total planted docs; well over several `BLOCK_LEN` (128) so
        /// the dense-term posting lists occupy multiple blocks.
        const N_DOCS: u32 = 400;
        /// Requested top-K — small, so the heap fills early and the
        /// score threshold starts pruning blocks.
        const K: usize = 5;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            // `alpha` in ~every doc, `beta` in ~half, `gamma` every
            // 5th, `delta` every 13th, `epsilon` every 29th — a
            // descending-df mix that makes the WAND pivot non-trivial.
            text.push_str("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 5 == 0 {
                text.push_str("gamma ");
            }
            if i % 13 == 0 {
                text.push_str("delta ");
            }
            if i % 29 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let terms: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon"];
        let wand = r
            .search_with_algo_for_bench("body", terms, K, OrAlgo::WandBmw)
            .await
            .expect("wand");
        let bmm = r
            .search_with_algo_for_bench("body", terms, K, OrAlgo::Bmm)
            .await
            .expect("bmm");
        assert_eq!(wand.len(), bmm.len(), "result length mismatch");
        assert_eq!(wand.len(), K, "expected a full top-K");
        for ((dw, sw), (db, sb)) in wand.iter().zip(bmm.iter()) {
            assert_eq!(dw, db, "doc_id mismatch wand={dw} bmm={db}");
            assert!((sw - sb).abs() < 1e-4, "score mismatch {sw} vs {sb}");
        }
    }

    #[tokio::test]
    async fn windowed_union_agrees_with_bmm() {
        // The windowed union scorer must return the identical top-k as
        // the production MaxScore+BMM path — across term counts, k values,
        // and the uniform-UB (common-term) shape it targets. N_DOCS spans
        // multiple windows (and many BLOCK_LEN=128 posting blocks), so the
        // walk exercises the multi-window path: base advancing to the next
        // window, empty-window skipping, and cross-window monotonicity —
        // not just a single window. Tied to OR_WINDOW so it keeps crossing
        // the boundary if the window size changes.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha zeta eta theta "); // ~every doc
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            if i % 7 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        let uniform_terms: &[&str] = &["zeta", "eta", "theta"];
        let uniform_cursors = r
            .build_term_cursors(col, uniform_terms, None, false)
            .await
            .expect("uniform cursors");
        // Phase 1 routing is k-gated: the common-heavy (equal-upper-bound)
        // shape stays on the pruning MaxScore path at small/mid k, and only
        // falls to the windowed scan at deep k (past the pruning cutoff),
        // where MaxScore can no longer prune it.
        for k in [1usize, 5, 16, OR_WINDOWED_UNIFORM_MAX_PRUNING_K] {
            assert!(
                !route_or_to_windowed(&uniform_cursors, k),
                "common-heavy OR at k={k} (<= cutoff) should route to MaxScore"
            );
        }
        for k in [OR_WINDOWED_UNIFORM_MAX_PRUNING_K + 1, 1000] {
            assert!(
                route_or_to_windowed(&uniform_cursors, k),
                "common-heavy OR at deep k={k} should route to the windowed scan"
            );
        }

        let shapes: &[&[&str]] = &[
            &["alpha", "beta"],
            &["alpha", "beta", "gamma"],
            &["beta", "gamma", "delta"], // no single dominator
            &["alpha", "beta", "gamma", "delta", "epsilon"],
            uniform_terms,
        ];
        for terms in shapes {
            for k in [1usize, 5, 50, 1000] {
                let bmm = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Bmm)
                    .await
                    .expect("bmm");
                let win = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Windowed)
                    .await
                    .expect("windowed");
                assert_eq!(bmm.len(), win.len(), "len mismatch {terms:?} k={k}");
                for ((db, sb), (dw, sw)) in bmm.iter().zip(win.iter()) {
                    assert_eq!(db, dw, "doc_id mismatch {terms:?} k={k}: bmm={db} win={dw}");
                    assert!(
                        (sb - sw).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sb} vs {sw}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn wand_bmw_2term_no_floor_agrees_with_bmm() {
        // The small-k 2-term production path (`run_wand_bmw`) must return
        // the identical top-k as MaxScore+BMM on the same inputs, across k.
        // It is only reached floor-free (the dispatcher routes to MaxScore
        // when a cross-segment floor is live), so both sides run unfloored
        // (`NEG_INFINITY`). Multi-window corpus so WAND exercises block
        // skips; `gamma` rarer than `beta` rarer than `alpha`.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        let shapes: &[&[&str]] = &[&["alpha", "beta"], &["beta", "gamma"], &["alpha", "gamma"]];
        for terms in shapes {
            for k in [1usize, 5, 50, 128] {
                let cw = r
                    .build_term_cursors(col, terms, None, false)
                    .await
                    .expect("cursors");
                let cb = r
                    .build_term_cursors(col, terms, None, false)
                    .await
                    .expect("cursors");
                let wand = r.run_wand_bmw(col, cw, k).expect("wand");
                let bmm = r
                    .run_max_score_bmm(col, cb, k, None, f32::NEG_INFINITY)
                    .expect("bmm");
                assert_eq!(wand.len(), bmm.len(), "len mismatch {terms:?} k={k}");
                for ((dw, sw), (db, sb)) in wand.iter().zip(bmm.iter()) {
                    assert_eq!(dw, db, "doc mismatch {terms:?} k={k}: {dw} vs {db}");
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sw} vs {sb}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn two_term_rare_anchor_gates_on_df_ratio() {
        // `df` is read onto the cursor, and the 2-term WAND router fires
        // only when one posting list is >= WAND_BMW_2TERM_DF_RATIO× shorter
        // than the other (a rare anchor), not when both terms are common.
        const N_DOCS: u32 = 4000;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("common "); // every doc
            if i % 2 == 0 {
                text.push_str("frequent "); // half the docs
            }
            if i % 200 == 0 {
                text.push_str("rare "); // ~1/200 of docs
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        // common (df≈N) + rare (df≈N/200): ratio 200 ≥ 16 → anchor.
        let anchored = r
            .build_term_cursors(col, &["common", "rare"], None, false)
            .await
            .expect("cursors");
        assert!(
            two_term_has_rare_anchor(&anchored),
            "rare+common should have a rare anchor"
        );
        // common (df≈N) + frequent (df≈N/2): ratio 2 < 16 → no anchor.
        let uniform = r
            .build_term_cursors(col, &["common", "frequent"], None, false)
            .await
            .expect("cursors");
        assert!(
            !two_term_has_rare_anchor(&uniform),
            "two common terms should not anchor"
        );
    }

    #[test]
    fn deep_k_dominant_union_reroutes_only_when_list_is_long() {
        // The deep-k reroute to the windowed scorer fires only when
        // pruning is dead (k reaches the rarer terms' combined df) AND the
        // dominant list is long enough to amortize the window setup.
        const LONG: u64 = 3_000_000; // dominant common term
        const RARE: u64 = 500; // rare second term
        let total = LONG + RARE;

        // Deep k (>= rest_df) over a long dominant list: reroute.
        assert!(
            or_reroute_by_df(LONG, total, 2, 1000),
            "deep k over a long dominant list should reroute to windowed"
        );
        // Shallow k (< rest_df): the rare term still fills the heap, pruning
        // is alive, stay on MaxScore.
        assert!(
            !or_reroute_by_df(LONG, total, 2, 100),
            "k below the rare term's df keeps pruning alive → MaxScore"
        );
        // Exact boundary k == rest_df: the heap needs one doc beyond the
        // rare term's list, so pruning is already dead → reroute (the test
        // is `>=`, so the boundary counts).
        assert!(
            or_reroute_by_df(LONG, total, 2, RARE as usize),
            "k exactly at rest_df should reroute"
        );
        // One below the boundary (k == rest_df - 1): rare term still fills
        // the heap, stay on MaxScore.
        assert!(
            !or_reroute_by_df(LONG, total, 2, RARE as usize - 1),
            "k just below rest_df keeps pruning alive → MaxScore"
        );
        // Long list but only one term: not an OR.
        assert!(
            !or_reroute_by_df(LONG, LONG, 1, 1000),
            "single term is not a union"
        );
        // Small union (dominant list below the floor): too little work to
        // amortize the window; stay on MaxScore even at deep k. This is the
        // case that regressed before the floor was added.
        let small = OR_WINDOWED_MIN_DOMINANT_DF - 1;
        assert!(
            !or_reroute_by_df(small, small + RARE, 2, 1_000_000),
            "a union below the dominant-df floor must not reroute"
        );
        // Exactly at the floor with deep k: reroute.
        assert!(
            or_reroute_by_df(
                OR_WINDOWED_MIN_DOMINANT_DF,
                OR_WINDOWED_MIN_DOMINANT_DF + RARE,
                2,
                1000
            ),
            "at the dominant-df floor a deep-k union reroutes"
        );
    }

    #[tokio::test]
    async fn windowed_union_negation_agrees_with_bmm() {
        // The windowed scorer applies the ExcludeFilter (negation) at
        // drain. Drive a negated query straight through run_windowed_union
        // and check it matches MaxScore+BMM with the same exclusion — BMM's
        // negation is the oracle-validated reference, so equality proves
        // the windowed filter arm. (Calls the scorers directly so the
        // windowed arm is exercised regardless of the production dispatch.)
        const N_DOCS: u32 = OR_WINDOW + 1000; // spans more than one window
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            if i % 7 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        // (positive terms, negated terms)
        let cases: &[(&[&str], &[&str])] = &[
            (&["alpha", "beta", "gamma"], &["delta"]),
            (&["beta", "gamma", "delta"], &["epsilon"]),
            (&["alpha", "beta", "gamma", "delta"], &["epsilon", "gamma"]),
        ];
        for (pos, neg) in cases {
            for k in [1usize, 5, 50] {
                let mut wf = ExcludeFilter::new(
                    r.build_term_cursors(col, neg, None, false)
                        .await
                        .expect("neg cursors"),
                );
                let win = r
                    .run_windowed_union(
                        col,
                        r.build_term_cursors(col, pos, None, false)
                            .await
                            .expect("pos cursors"),
                        k,
                        Some(&mut wf),
                        f32::NEG_INFINITY,
                        0,
                        u32::MAX,
                    )
                    .expect("windowed");
                let mut bf = ExcludeFilter::new(
                    r.build_term_cursors(col, neg, None, false)
                        .await
                        .expect("neg cursors"),
                );
                let bmm = r
                    .run_max_score_bmm(
                        col,
                        r.build_term_cursors(col, pos, None, false)
                            .await
                            .expect("pos cursors"),
                        k,
                        Some(&mut bf),
                        f32::NEG_INFINITY,
                    )
                    .expect("bmm");
                assert_eq!(win.len(), bmm.len(), "len {pos:?} -{neg:?} k={k}");
                for ((dw, sw), (db, sb)) in win.iter().zip(bmm.iter()) {
                    assert_eq!(
                        dw, db,
                        "doc mismatch {pos:?} -{neg:?} k={k}: win={dw} bmm={db}"
                    );
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score mismatch {pos:?} -{neg:?} k={k}: {sw} vs {sb}"
                    );
                }
            }
        }

        // Sanity: the filter is actually active — at a high k the negated
        // query must return strictly fewer docs than the positive-only one
        // (the negated term excludes a non-empty set).
        let pos: &[&str] = &["alpha", "beta", "gamma"];
        let neg: &[&str] = &["delta"];
        let unfiltered = r
            .run_windowed_union(
                col,
                r.build_term_cursors(col, pos, None, false)
                    .await
                    .expect("pos"),
                N_DOCS as usize,
                None,
                f32::NEG_INFINITY,
                0,
                u32::MAX,
            )
            .expect("unfiltered");
        let mut f = ExcludeFilter::new(
            r.build_term_cursors(col, neg, None, false)
                .await
                .expect("neg"),
        );
        let filtered = r
            .run_windowed_union(
                col,
                r.build_term_cursors(col, pos, None, false)
                    .await
                    .expect("pos"),
                N_DOCS as usize,
                Some(&mut f),
                f32::NEG_INFINITY,
                0,
                u32::MAX,
            )
            .expect("filtered");
        assert!(
            filtered.len() < unfiltered.len(),
            "negation should drop docs: filtered={} unfiltered={}",
            filtered.len(),
            unfiltered.len()
        );
    }

    #[tokio::test]
    async fn search_with_algo_empty_and_zero_k_short_circuit() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        assert!(
            r.search_with_algo_for_bench("body", &[], 5, OrAlgo::Bmm)
                .await
                .expect("empty")
                .is_empty()
        );
        assert!(
            r.search_with_algo_for_bench("body", &["rust"], 0, OrAlgo::Exhaustive)
                .await
                .expect("zero k")
                .is_empty()
        );
    }
}
