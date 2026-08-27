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

/// Left-pack control table: for each 8-bit survivor mask, the lane indices that
/// gather the set lanes to the front (for `permutevar8x32`). Unset trailing
/// slots are don't-cares (the store advances only by `popcount(mask)`).
#[cfg(target_arch = "x86_64")]
const fn build_left_pack() -> [[u8; 8]; 256] {
    let mut t = [[0u8; 8]; 256];
    let mut m = 0usize;
    while m < 256 {
        let mut out = 0usize;
        let mut b = 0usize;
        while b < 8 {
            if (m >> b) & 1 == 1 {
                t[m][out] = b as u8;
                out += 1;
            }
            b += 1;
        }
        m += 1;
    }
    t
}
#[cfg(target_arch = "x86_64")]
static LEFT_PACK: [[u8; 8]; 256] = build_left_pack();

/// Compact `(docs, scores)` in place to the entries with `score >= min_score`,
/// preserving order; returns the survivor count. Drops the candidates that can
/// no longer beat the top-k threshold before they reach the non-essential
/// completion, so the expensive per-term probe only touches survivors.
fn filter_survivors(docs: &mut [u32], scores: &mut [f32], min_score: f32) -> usize {
    debug_assert_eq!(docs.len(), scores.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is present (checked above); all loads/stores are
            // unaligned and bounded to `[0, n)`; the store window `[out, out+8)`
            // never precedes an unread source (out <= i, data is register-held
            // before the store), so the in-place left-pack cannot corrupt a
            // not-yet-read lane.
            return unsafe { filter_survivors_avx2(docs, scores, min_score) };
        }
    }
    let mut out = 0usize;
    for i in 0..docs.len() {
        let keep = scores[i] >= min_score;
        docs[out] = docs[i];
        scores[out] = scores[i];
        out += keep as usize;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn filter_survivors_avx2(docs: &mut [u32], scores: &mut [f32], min_score: f32) -> usize {
    use std::arch::x86_64::*;
    // SAFETY, per obligation:
    // - AVX2 intrinsics: reachable only through the dispatcher's
    //   `is_x86_feature_detected!("avx2")` guard, so the target feature the
    //   `#[target_feature]` attribute requires is present at runtime.
    // - `loadu`/`storeu` + `*ptr`: the unaligned variants impose no alignment
    //   precondition; `dptr`/`sptr` come from the same `n`-element slices, and
    //   every access is bounded below (see per-site notes).
    // - `LEFT_PACK[mask]`: `mask` is an 8-lane movemask, so `0..=255` — always
    //   in range of the 256-entry table; `_mm_loadl_epi64` reads its 8 bytes.
    unsafe {
        let n = docs.len();
        let thr = _mm256_set1_ps(min_score);
        let dptr = docs.as_mut_ptr();
        let sptr = scores.as_mut_ptr();
        let mut out = 0usize;
        let mut i = 0usize;
        while i + 8 <= n {
            // SAFETY: `i + 8 <= n` bounds the source span `[i, i+8)`; `out <= i`
            // holds every iteration (out grows by popcount ≤ 8, i by 8), so the
            // dest span `[out, out+8)` is in-bounds too. `vs`/`vd` are read into
            // registers before the store, so the in-place left-pack overwrite of
            // `[out, out+8)` cannot clobber a source lane not yet read.
            let vs = _mm256_loadu_ps(sptr.add(i));
            let vd = _mm256_loadu_si256(dptr.add(i) as *const __m256i);
            let mask = _mm256_movemask_ps(_mm256_cmp_ps(vs, thr, _CMP_GE_OQ)) as usize;
            let ctrl =
                _mm256_cvtepu8_epi32(_mm_loadl_epi64(LEFT_PACK[mask].as_ptr() as *const __m128i));
            _mm256_storeu_ps(sptr.add(out), _mm256_permutevar8x32_ps(vs, ctrl));
            _mm256_storeu_si256(
                dptr.add(out) as *mut __m256i,
                _mm256_permutevar8x32_epi32(vd, ctrl),
            );
            out += (mask as u32).count_ones() as usize;
            i += 8;
        }
        while i < n {
            // SAFETY: `i < n` and `out <= i` keep every `.add` within `[0, n)`.
            let keep = *sptr.add(i) >= min_score;
            *dptr.add(out) = *dptr.add(i);
            *sptr.add(out) = *sptr.add(i);
            out += keep as usize;
            i += 1;
        }
        out
    }
}

/// Add one non-essential term's contribution to `scores[i]` for every survivor
/// `docs[i]` the term contains, scoring matches in SIMD batches rather than one
/// at a time. `docs` must be ascending — `bitset_probe_tf` advances the cursor
/// monotonically. Only touches matched survivors.
fn score_noness_batched(c: &mut TermCursor, docs: &[u32], scores: &mut [f32], dl_norm: &NormTable) {
    const L: usize = bm25::SCORE_SIMD_LANES;
    let mut n = 0usize;
    let mut bidx = [0usize; L];
    let mut btf = [0u32; L];
    let mut bnorm = [0f32; L];
    for (i, &doc) in docs.iter().enumerate() {
        if let Some(tf) = c.bitset_probe_tf(doc) {
            bidx[n] = i;
            btf[n] = tf;
            bnorm[n] = dl_norm.get(doc);
            n += 1;
            if n == L {
                let contrib = bm25::score_one_term_x4(c.idf_x_k1p1, btf, bnorm);
                for l in 0..L {
                    scores[bidx[l]] += contrib[l];
                }
                n = 0;
            }
        }
    }
    for l in 0..n {
        scores[bidx[l]] += bm25::score_with_dl_norm_k1(c.idf_x_k1p1, btf[l], bnorm[l]);
    }
}

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

/// Route a ranked AND to the membership walk ([`FtsReader::and_membership_scored`])
/// instead of the block flat-merge. True only for v4 blobs — where a common
/// term's blocks are bitset-encoded, so [`TermCursor::contains`] is an O(1)
/// bit-test — with **≥3 terms** and a **very sparse rarest term**: driving the
/// rarest doc-by-doc is then cheap and bit-testing the common others beats
/// decoding their blocks to align them.
///
/// Two guards keep it off the shapes where it regressed the ranked tail:
/// - **≥3 terms**: a 2-term AND keeps the specialized `and_flat_merge_2term`
///   (two-pointer merge over the decoded blocks), which the membership walk was
///   losing to.
/// - **rarest < 1/`AND_MEMBERSHIP_RAREST_SPARSE_DIVISOR`**: the walk gives up the
///   flat-merge's heap-bar block skip, so it only pays when the rarest list is
///   genuinely short. A looser bound regressed p99 where the bar-skip was
///   working.
fn and_prefer_membership(has_bitset_blocks: bool, cursors: &[TermCursor]) -> bool {
    if !has_bitset_blocks || cursors.len() < 3 {
        return false;
    }
    let max_doc = cursors
        .iter()
        .filter_map(|c| c.blocks.last())
        .map(|b| b.last_doc_id)
        .max()
        .unwrap_or(0);
    let min_df = cursors.iter().map(|c| c.df).min().unwrap_or(0);
    min_df.saturating_mul(AND_MEMBERSHIP_RAREST_SPARSE_DIVISOR) < u64::from(max_doc)
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

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut sink = ScoreSink {
            heap: &mut heap,
            k,
            filter,
            floor_eff,
        };
        if and_prefer_membership(self.has_bitset_blocks, &cursors) {
            // Rarest-driven membership walk: bit-test the common terms instead
            // of decoding their blocks to align them (the flat-merge's dominant
            // cost on rare∧common). See `and_membership_scored`.
            self.and_membership_scored(cursors, dl_norm_k1, &mut sink);
        } else {
            // Smallest-df cursor at index 0 = leader. Ascending-df order reduces
            // the expected number of leapfrog bumps per candidate.
            cursors.sort_by_key(|c| c.block_count());
            self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        }
        Ok(drain_top_k_desc(heap))
    }

    /// Ranked AND via the same rarest-driven membership walk that the unranked
    /// count path uses ([`count_and_intersect_membership`]), but scoring the
    /// matches. Iterate the rarest term; for each of its docs, probe the others
    /// with [`TermCursor::contains`] — a **bitset bit-test with no block decode**
    /// — short-circuiting on the first miss. Only a doc present in *every* term
    /// is materialized (decoded + positioned via [`TermCursor::materialize_at`])
    /// and scored. This skips the flat-merge's per-leader-doc `skip_to` that
    /// fully decodes a common term's 128-doc block to read one doc — the profiled
    /// cost on n≥3-term AND with a common term. Score is `Σ` per-term BM25 at the
    /// doc, identical to the flat-merge; emitted through the generic sink, so the
    /// walk is written against any [`AndSink`]. Only the pure-AND path
    /// ([`run_and_intersect`](Self::run_and_intersect), a `ScoreSink`) routes here
    /// today; must+should keeps the flat-merge so its should-clause scoring stays
    /// in one place. Gated to the v4 bitset case with a sparse rarest term (see
    /// [`and_prefer_membership`]); the flat-merge stays for v1–v3 and the
    /// all-dense case.
    fn and_membership_scored<S: AndSink>(
        &self,
        mut cursors: Vec<TermCursor>,
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        let driver_idx = cursors
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.block_count())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mut driver = cursors.swap_remove(driver_idx);
        let mut others = cursors;
        while !driver.is_exhausted() {
            let doc = driver.current_doc_id();
            if others.iter_mut().all(|o| o.contains(doc)) {
                let score = if sink.needs_score() {
                    let norm = dl_norm_k1.get(doc);
                    let mut s =
                        bm25::score_with_dl_norm_k1(driver.idf_x_k1p1, driver.current_tf(), norm);
                    for o in others.iter_mut() {
                        o.materialize_at(doc);
                        s += bm25::score_with_dl_norm_k1(o.idf_x_k1p1, o.current_tf(), norm);
                    }
                    s
                } else {
                    0.0
                };
                sink.emit(doc, score);
            }
            driver.next();
        }
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
                    // Bound the non-essentials at `candidate` by each one's
                    // block-max for the block that *contains* it (via the
                    // monotonic `inspect_block` hint — amortized O(1), no
                    // decode), not by its global term-max. Far tighter for a
                    // common term, so the skip below fires on many more docs,
                    // dropping the non-essential `skip_to` + score + heap — the
                    // dominant per-doc cost.
                    let mut others_ub = 0.0f32;
                    for c in cursors.iter_mut().skip(1) {
                        c.shallow_advance_block_to(candidate);
                        others_ub += c.inspect_block_max_bm25();
                    }
                    if essential_score + others_ub <= threshold {
                        cursors[0].next();
                        continue;
                    }
                    // SIMD-pack non-essentials at `candidate`.
                    let mut idfs = [cursors[0].idf_x_k1p1, 0.0, 0.0, 0.0];
                    let mut tfs = [cursors[0].current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(1) {
                        if let Some(tf) = cursor.bitset_probe_tf(candidate) {
                            idfs[packed] = cursor.idf_x_k1p1;
                            tfs[packed] = tf as f32;
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
                            // Locate + score the non-essential without expanding
                            // a dense block's doc ids: a bit-test + popcount-rank
                            // tf read on a bitset block, decode-and-locate on a
                            // PACKED one.
                            if let Some(tf) = cursor.bitset_probe_tf(candidate) {
                                score += bm25::score_with_dl_norm_k1(cursor.idf_x_k1p1, tf, norm);
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

    /// Windowed MaxScore: one kernel that adapts to the running k-th
    /// threshold per window, so a query needs no a-priori choice between the
    /// windowed OR-sum and per-candidate pruning. Each window recomputes the
    /// essential / non-essential split from the live threshold (essentials =
    /// the highest-`term_max` prefix whose suffix-sum still exceeds
    /// threshold). Only essentials are accumulated into the window (SIMD
    /// OR-sum, as in [`Self::run_windowed_union`]); the non-essentials are
    /// completed per surviving candidate at drain. When the threshold is low
    /// (early scan, dense union) every term is essential and this is exactly
    /// the cheap windowed OR-sum; as the heap fills and the threshold rises
    /// the essential set shrinks and the kernel prunes like MaxScore — both
    /// regimes in one loop, decided per window, never per query.
    ///
    /// A doc reaches top-k only if it carries an essential term: a doc with
    /// only non-essential terms has max score `≤ partial_max[f_essential] ≤
    /// threshold`, so dropping it (never accumulated) is exact. Same top-k
    /// and `(score desc, doc asc)` order as [`Self::run_max_score_bmm`];
    /// oracle-gated.
    pub(super) fn run_windowed_maxscore(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
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

        // Descending by term-max UB, then suffix sums: `partial_max[f]` is the
        // total UB of terms `f..n`, monotonically decreasing, so the essential
        // boundary is the smallest `f` with `partial_max[f] ≤ threshold`.
        cursors.sort_unstable_by(|a, b| {
            b.term_max_bm25
                .partial_cmp(&a.term_max_bm25)
                .unwrap_or(Ordering::Equal)
        });
        let n = cursors.len();
        let mut partial_max = vec![0.0_f32; n + 1];
        for i in (0..n).rev() {
            partial_max[i] = partial_max[i + 1] + cursors[i].term_max_bm25;
        }
        let recompute_f = |partial_max: &[f32], threshold: f32| -> usize {
            let mut f = 0;
            while f < partial_max.len() - 1 && partial_max[f] > threshold {
                f += 1;
            }
            f
        };

        let initial_cap =
            top_k_initial_capacity(k, u64::from(self.n_docs), Some((doc_id_start, doc_id_end)));
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = floor_eff.max(0.0);
        let mut scores = vec![0.0f32; OR_WINDOW as usize];
        let mut present = [0u64; OR_WINDOW_WORDS];
        // Compacted per-window candidate list (doc + essential score), reused;
        // the drain fills it, the SIMD filter compacts it to survivors.
        let mut win_docs: Vec<u32> = Vec::with_capacity(OR_WINDOW as usize);
        let mut win_scores: Vec<f32> = Vec::with_capacity(OR_WINDOW as usize);
        // Sum of every term's UB — the reference for the essential-side
        // block-max skip below.
        let total_term_ub = partial_max[0];

        loop {
            // Continuous partition: recompute the essential set from the live
            // threshold. `threshold` only rises, so `f_essential` only shrinks.
            let f_essential = recompute_f(&partial_max, threshold);

            // f==1 fast path: a single dominant essential. Process its blocks
            // per-candidate (block-skip + non-essential completion) with no
            // window buffer — the windowed accumulate/drain has no advantage
            // when the candidate set is one term's postings, and this matches
            // MaxScore's f==1 path on small-k dominant queries.
            if f_essential == 1 {
                if cursors[0].is_exhausted() || cursors[0].current_doc_id() >= doc_id_end {
                    break;
                }
                // Skip the whole block if it cannot beat threshold.
                if heap.len() >= k {
                    let block_ub = cursors[0].current_block_max_bm25()
                        + (total_term_ub - cursors[0].term_max_bm25);
                    if block_ub <= threshold {
                        let end = cursors[0].current_block_last_doc_id();
                        cursors[0].skip_to(end.saturating_add(1));
                        continue;
                    }
                }
                let block_end = cursors[0].current_block_last_doc_id();
                let (ess, non_ess) = cursors.split_at_mut(1);
                let c0 = &mut ess[0];
                while !c0.is_exhausted()
                    && c0.current_doc_id() <= block_end
                    && c0.current_doc_id() < doc_id_end
                {
                    let candidate = c0.current_doc_id();
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(candidate)
                    {
                        c0.next();
                        continue;
                    }
                    let norm = dl_norm_k1.get(candidate);
                    let essential_score =
                        bm25::score_with_dl_norm_k1(c0.idf_x_k1p1, c0.current_tf(), norm);
                    // Bound the non-essentials at `candidate` by each one's
                    // block-max for the block that *contains* it (monotonic
                    // `shallow_advance` hint — amortized O(1), no decode), not by
                    // its global term-max. Far tighter for a common term, so the
                    // skip below fires on many more docs, dropping the completion
                    // probe + heap work — the dominant per-doc cost on a dense
                    // leader query.
                    let mut others_ub = 0.0f32;
                    for c in non_ess.iter_mut() {
                        c.shallow_advance_block_to(candidate);
                        others_ub += c.inspect_block_max_bm25();
                    }
                    if essential_score + others_ub <= threshold {
                        c0.next();
                        continue;
                    }
                    // Complete: probe each non-essential and SIMD-pack the
                    // matches (leader seeded as lane 0, so `score` is the full
                    // BM25 sum).
                    let mut idfs = [c0.idf_x_k1p1, 0.0, 0.0, 0.0];
                    let mut tfs = [c0.current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score = 0.0f32;
                    for c in non_ess.iter_mut() {
                        if let Some(tf) = c.bitset_probe_tf(candidate) {
                            idfs[packed] = c.idf_x_k1p1;
                            tfs[packed] = tf as f32;
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
                    let mut raised = false;
                    if heap.len() < k {
                        heap.push(TopKEntry(score, candidate));
                        if heap.len() == k {
                            threshold = heap.peek().expect("non-empty").0.max(threshold);
                            raised = true;
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, candidate));
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                        raised = true;
                    }
                    c0.next();
                    // Only when the threshold actually rose: if even this term
                    // plus every other term at its max can't beat it, no further
                    // doc can qualify at all. (Re-checking the partition per
                    // candidate — an O(n) scan — was the f==1 path's dominant
                    // cost on dense leader queries.)
                    if raised && total_term_ub <= threshold {
                        break;
                    }
                }
                continue;
            }

            // Candidates come from the essential terms only. In the same scan
            // find the nearest essential *block* boundary — the window ends
            // there so the threshold updates per-block and the essential set
            // collapses to the f==1 leader path early on a dense, small-k
            // query. `current_block_last_doc_id` reads block metadata only,
            // no decode.
            let mut min_doc = u32::MAX;
            let mut block_bound = doc_id_end;
            for c in cursors.iter().take(f_essential) {
                if !c.is_exhausted() {
                    min_doc = min_doc.min(c.current_doc_id());
                    block_bound = block_bound.min(c.current_block_last_doc_id().saturating_add(1));
                }
            }
            if min_doc == u32::MAX || min_doc >= doc_id_end {
                break;
            }
            // 64-align the base for the presence bitmask, then end the window at
            // the nearest essential block boundary (so the threshold updates
            // per-block and the essential set can collapse early), capped at the
            // score-buffer width.
            let base = min_doc & !63;
            let window_end = block_bound
                .min(base.saturating_add(OR_WINDOW))
                .min(doc_id_end);

            // Accumulate the essential terms' contributions into the window
            // (SIMD OR-sum; scalar tail). Identical to the windowed-union body,
            // restricted to essentials.
            // Block-max skip fires only once the heap is full (threshold is a
            // real k-th score); hoisted so the per-posting loop doesn't re-test.
            let prune = heap.len() >= k;
            for c in cursors.iter_mut().take(f_essential) {
                let mut checked_block = usize::MAX;
                while !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d >= window_end {
                        break;
                    }
                    // Block-max skip, checked once per block (not per posting):
                    // skip a whole block whose docs cannot beat threshold even
                    // carrying every other term at its term-max — the bound
                    // per-candidate MaxScore uses, applied to the essential
                    // accumulate so the dense OR-sum still prunes at small k.
                    // Never fires while the heap is filling, so a doc that must
                    // be admitted is never dropped.
                    if prune && c.current_block != checked_block {
                        checked_block = c.current_block;
                        let block_ub =
                            c.current_block_max_bm25() + (total_term_ub - c.term_max_bm25);
                        if block_ub <= threshold {
                            let last = c.current_block_last_doc_id();
                            c.skip_to(last.saturating_add(1));
                            continue;
                        }
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

            // Drain the presence bitmask into a compacted, ascending
            // `(doc, essential-score)` list (dropping negated docs), then run
            // the reference pipeline: SIMD-filter to competitive survivors,
            // complete the non-essentials over the survivors only, collect.
            let (_, non_ess) = cursors.split_at_mut(f_essential);
            let heap_full = heap.len() >= k;
            let words = ((window_end - base) as usize)
                .div_ceil(64)
                .min(OR_WINDOW_WORDS);
            win_docs.clear();
            win_scores.clear();
            for (word_idx, word) in present[..words].iter_mut().enumerate() {
                let mut bits = *word;
                *word = 0;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let local = (word_idx << 6) | b;
                    let doc = base + local as u32;
                    let score = std::mem::take(&mut scores[local]);
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(doc)
                    {
                        continue;
                    }
                    win_docs.push(doc);
                    win_scores.push(score);
                }
            }

            // Complete the non-essentials over the compacted survivors. Once
            // the heap is full, do it the reference way: strongest non-essential
            // first, re-filtering the survivor list (progressively tighter
            // budget) before each and batch-scoring matches — so weaker terms
            // touch a shrinking survivor set. While filling, every doc must be
            // admitted, so score all non-essentials over every candidate.
            if heap_full && !non_ess.is_empty() {
                let nn = non_ess.len();
                for jj in 0..nn {
                    // Max score still obtainable from non-essentials jj..nn.
                    let bar = threshold - partial_max[f_essential + jj];
                    let sv = filter_survivors(&mut win_docs, &mut win_scores, bar);
                    win_docs.truncate(sv);
                    win_scores.truncate(sv);
                    if win_docs.is_empty() {
                        break;
                    }
                    score_noness_batched(&mut non_ess[jj], &win_docs, &mut win_scores, dl_norm_k1);
                }
            } else {
                for c in non_ess.iter_mut() {
                    score_noness_batched(c, &win_docs, &mut win_scores, dl_norm_k1);
                }
            }

            // Collect the fully-scored candidates (ascending doc order).
            for (idx, &doc) in win_docs.iter().enumerate() {
                let score = win_scores[idx];
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

    /// Multi-term OR dispatch. Routes every union to the one windowed MaxScore
    /// kernel ([`Self::run_windowed_maxscore`]), except a 2-term rare+common OR,
    /// which goes to WAND+BMW ([`Self::run_wand_bmw`]) — it pivots on the rare
    /// term to skip the common term's long posting list, the one shape the
    /// windowed kernel can't beat. The rationale for that cutoff is inline below.
    ///
    /// The per-candidate MaxScore+BMM ([`Self::run_max_score_bmm`]), the plain
    /// windowed OR-sum ([`Self::run_windowed_union`]), and the exhaustive walk
    /// ([`Self::run_exhaustive_union`]) remain in the tree only as bench
    /// baselines and the correctness oracle, reachable via
    /// `search_with_algo_for_bench`; the dispatcher routes to none of them.
    pub(super) fn dispatch_or_algo(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // A 2-term OR of one rare + one common term is a worst case for a
        // union scan: it walks the common term's long posting list end to end.
        // WAND+BMW pivots on the rare (short) term and skips most of the common
        // term's list — a large win. For two comparable-length common terms
        // there is no short anchor to skip on, so WAND's per-iteration cursor
        // re-sort is pure overhead; those fall through. Route a 2-term OR to
        // WAND only when (a) one list is much shorter than the other (df ratio,
        // `two_term_has_rare_anchor`); (b) k is small — at large k the top-k
        // threshold is too low for WAND to prune; (c) no negation —
        // `run_wand_bmw` applies no exclude filter; and (d) no cross-segment
        // floor — seeding WAND's threshold from a floor mis-prunes.
        //
        // Everything else runs the one windowed MaxScore union kernel. Its
        // essential/non-essential partition is continuous in the running k-th
        // threshold: when a term dominates it degenerates to a per-candidate
        // leader scan (block-max pruning, the dominant-term fast path); when no
        // term dominates it stays a windowed SIMD OR-sum across the essential
        // block. One kernel therefore covers both the sparse-selective and the
        // uniform-common regimes with no cross-over routing to tune.
        let no_floor = floor_eff == f32::NEG_INFINITY;
        if cursors.len() == 2
            && k <= WAND_BMW_2TERM_MAX_K
            && filter.is_none()
            && no_floor
            && two_term_has_rare_anchor(&cursors)
        {
            self.run_wand_bmw(column_id, cursors, k)
        } else {
            self.run_windowed_maxscore(column_id, cursors, k, filter, floor_eff, 0, u32::MAX)
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
            OrAlgo::WindowedMaxscore => self.run_windowed_maxscore(
                column_id,
                cursors,
                k,
                None,
                f32::NEG_INFINITY,
                0,
                u32::MAX,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder,
        posting::{ENCODING_BITSET, ENCODING_OFF},
        reader::BoolMode,
        tokenize::AsciiLowerTokenizer,
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
    async fn bitset_probe_tf_reads_correct_tf_across_presence_words() {
        // Regression for the ranked-OR non-essential probe: `bitset_probe_tf`
        // locates a doc in a dense (bitset) block by bit-test and reads its
        // tf by popcount-rank — summing the popcounts of every 64-bit presence
        // word ahead of the doc's word, then the set bits below it in its own
        // word. The top-k oracle only exercised a 20-doc block, so every doc
        // sat in presence word 0 (bit < 64) and the cross-word accumulation
        // loop never ran. Plant a block where the probed docs sit at bit >= 64
        // with a tf that differs from the rest, so a wrong cross-word popcount
        // would land on the wrong tf.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        // `common` is present at doc 0 and docs 100..=165. First-doc 0 word-
        // aligns the bitset base to 0, so doc 100 -> bit 100 (presence word 1)
        // and doc 165 -> bit 165 (presence word 2); the 0->100 gap widens the
        // deltas enough that the block takes the bitset encoding, not PFOR. tf
        // is 1 everywhere except 3 at doc 100 and 2 at doc 165.
        for id in 0u32..=165 {
            let text = match id {
                0 => "common",
                100 => "common common common",
                165 => "common common",
                101..=164 => "common",
                _ => "filler", // 1..=99: present docs that don't carry `common`
            };
            b.add_doc(0, id, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let mut cursors = r
            .build_term_cursors(0, &["common"], None, false)
            .await
            .expect("build common cursor");
        let cursor = &mut cursors[0];

        // Guard the test's own premise: the block must be a bitset, else the
        // probe takes the PACKED fallback and the rank path under test is skipped.
        let blk = cursor.blocks[0];
        assert_eq!(
            cursor.bytes[blk.block_byte_offset + ENCODING_OFF],
            ENCODING_BITSET,
            "corpus must produce a bitset block for this test to be meaningful"
        );

        // Probe in ascending doc order (the cursor advances forward only).
        // doc 50 is absent; doc 100 needs word 0's popcount (rank 1); doc 165
        // needs words 0 and 1 summed (rank 66) — the multi-word path.
        assert_eq!(
            cursor.bitset_probe_tf(50),
            None,
            "doc 50 absent from bitset"
        );
        assert_eq!(
            cursor.bitset_probe_tf(100),
            Some(3),
            "tf at doc 100 (bit 100, presence word 1)"
        );
        assert_eq!(
            cursor.bitset_probe_tf(165),
            Some(2),
            "tf at doc 165 (bit 165, presence word 2)"
        );
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
        let shapes: &[&[&str]] = &[
            &["alpha", "beta"],
            &["alpha", "beta", "gamma"],
            &["beta", "gamma", "delta"], // no single dominator
            &["alpha", "beta", "gamma", "delta", "epsilon"],
            &["zeta", "eta", "theta"], // uniform-common (no dominator)
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
    async fn windowed_maxscore_agrees_with_bmm() {
        // The continuous-partition windowed MaxScore must return the identical
        // top-k as per-candidate MaxScore+BMM across query shapes and k. Small
        // k makes the threshold rise fast so the essential set shrinks
        // mid-query (exercising the non-essential completion path); large k
        // keeps every term essential (the pure windowed OR-sum path). The
        // multi-window corpus exercises the partition changing across windows.
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

        let shapes: &[&[&str]] = &[
            &["alpha", "beta"],
            &["alpha", "beta", "gamma"],
            &["beta", "gamma", "delta"],   // no single dominator
            &["epsilon", "beta", "alpha"], // rare essential + common non-essentials
            &["alpha", "beta", "gamma", "delta", "epsilon"],
            &["zeta", "eta", "theta"], // uniform-common
        ];
        for terms in shapes {
            for k in [1usize, 5, 16, 50, 1000] {
                let bmm = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Bmm)
                    .await
                    .expect("bmm");
                let wms = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::WindowedMaxscore)
                    .await
                    .expect("windowed-maxscore");
                assert_eq!(bmm.len(), wms.len(), "len mismatch {terms:?} k={k}");
                for ((db, sb), (dw, sw)) in bmm.iter().zip(wms.iter()) {
                    assert_eq!(db, dw, "doc_id mismatch {terms:?} k={k}: bmm={db} wms={dw}");
                    assert!(
                        (sb - sw).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sb} vs {sw}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn windowed_maxscore_negation_agrees_with_bmm() {
        // The windowed MaxScore applies the ExcludeFilter (negation) at drain,
        // same as the windowed union. Drive negated queries straight through
        // run_windowed_maxscore and check they match MaxScore+BMM with the same
        // exclusion (the oracle-validated reference).
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
                let wms = r
                    .run_windowed_maxscore(
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
                    .expect("windowed-maxscore");
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
                assert_eq!(wms.len(), bmm.len(), "len {pos:?} -{neg:?} k={k}");
                for ((dw, sw), (db, sb)) in wms.iter().zip(bmm.iter()) {
                    assert_eq!(dw, db, "doc {pos:?} -{neg:?} k={k}: wms={dw} bmm={db}");
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score {pos:?} -{neg:?} k={k}: {sw} vs {sb}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn windowed_maxscore_agrees_with_bmm_randomized() {
        // Deterministic fuzz (xorshift): many corpora with varied per-term
        // densities and random multi-term shapes across k. The continuous
        // partition must produce the identical top-k as MaxScore+BMM in every
        // case — this exercises regime combinations the fixed corpora miss.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let vocab = ["a", "b", "c", "d", "e", "f", "g", "h"];
        for trial in 0..12u32 {
            // Sometimes span more than one OR_WINDOW (4096 docs).
            let n_docs = 500 + (rng() % 5000) as u32;
            // Per-term inclusion probability (out of 8) — varied densities.
            let probs: Vec<u64> = (0..vocab.len()).map(|_| 1 + rng() % 8).collect();
            let tok = Arc::new(AsciiLowerTokenizer);
            let mut b = FtsBuilder::new(tok);
            b.register_column("body".into(), false).expect("register");
            for i in 0..n_docs {
                let mut text = String::new();
                for (t, term) in vocab.iter().enumerate() {
                    if rng() % 8 < probs[t] {
                        let tf = 1 + rng() % 3;
                        for _ in 0..tf {
                            text.push_str(term);
                            text.push(' ');
                        }
                    }
                }
                if text.is_empty() {
                    text.push('a');
                }
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
            let r = FtsReader::open(blob, json).expect("open");
            for _ in 0..4 {
                let nt = 2 + (rng() % 4) as usize;
                let mut terms: Vec<&str> = Vec::new();
                for _ in 0..nt {
                    let t = vocab[(rng() % vocab.len() as u64) as usize];
                    if !terms.contains(&t) {
                        terms.push(t);
                    }
                }
                if terms.len() < 2 {
                    continue;
                }
                for k in [1usize, 7, 100, 1000] {
                    let bmm = r
                        .search_with_algo_for_bench("body", &terms, k, OrAlgo::Bmm)
                        .await
                        .expect("bmm");
                    let wms = r
                        .search_with_algo_for_bench("body", &terms, k, OrAlgo::WindowedMaxscore)
                        .await
                        .expect("wms");
                    assert_eq!(bmm.len(), wms.len(), "trial {trial} {terms:?} k={k} len");
                    for ((db, sb), (dw, sw)) in bmm.iter().zip(wms.iter()) {
                        assert_eq!(db, dw, "trial {trial} {terms:?} k={k} doc {db} vs {dw}");
                        assert!(
                            (sb - sw).abs() < 1e-4,
                            "trial {trial} {terms:?} k={k} score {sb} vs {sw}"
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn windowed_maxscore_dense_common_leader_agrees_with_bmm() {
        // Regression for the f==1 leader path. A dense, common leader term
        // (present in most docs) with common non-essentials and small k is the
        // shape where the threshold rises high, the essential set collapses to
        // the single leader, and the per-candidate tight block-max bail plus the
        // early "no doc can qualify" termination carry the query — the path a
        // rare-leader corpus (the other tests) never stresses. Varying tf spreads
        // scores so each block's block-max UB differs from the global term-max,
        // so the tight bound actually prunes where the loose one would not. Must
        // still return the identical top-k as per-candidate MaxScore+BMM.
        const N_DOCS: u32 = OR_WINDOW * 2 + 313; // several blocks, > 1 window
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            // All four terms common (stopword-like); "not" is the rarest so it
            // becomes the essential leader. tf on "to" varies 1..=3 by doc.
            let mut text = String::new();
            for _ in 0..(1 + i % 3) {
                text.push_str("to ");
            }
            if i % 8 != 0 {
                text.push_str("be ");
            }
            if i % 4 != 0 {
                text.push_str("or ");
            }
            if i % 8 < 5 {
                text.push_str("not ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let terms = ["to", "be", "or", "not"];
        for k in [1usize, 5, 10, 50, 200] {
            let bmm = r
                .search_with_algo_for_bench("body", &terms, k, OrAlgo::Bmm)
                .await
                .expect("bmm");
            let wms = r
                .search_with_algo_for_bench("body", &terms, k, OrAlgo::WindowedMaxscore)
                .await
                .expect("wms");
            assert_eq!(bmm.len(), wms.len(), "len k={k}");
            for ((db, sb), (dw, sw)) in bmm.iter().zip(wms.iter()) {
                assert_eq!(db, dw, "doc mismatch k={k}: bmm={db} wms={dw}");
                assert!((sb - sw).abs() < 1e-4, "score mismatch k={k}: {sb} vs {sw}");
            }
        }
    }

    #[tokio::test]
    async fn windowed_maxscore_ranged_agrees_with_bmm_range() {
        // The ranged fan-out entry now runs this same kernel over a doc-id
        // sub-window. Driven over explicit sub-ranges (including a
        // window-boundary-crossing slice), it must match per-candidate
        // MaxScore+BMM restricted to the identical window — a sliced query
        // returns exactly the docs in its slice, scored identically.
        const N_DOCS: u32 = OR_WINDOW * 2 + 777;
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
            if i % 6 == 0 {
                text.push_str("delta ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        let terms: &[&str] = &["alpha", "beta", "gamma", "delta"];
        let ranges: &[(u32, u32)] = &[
            (0, N_DOCS),
            (1000, OR_WINDOW + 2000), // crosses a window boundary
            (OR_WINDOW, OR_WINDOW + 300),
        ];
        for &(lo, hi) in ranges {
            for k in [1usize, 10, 100] {
                let wms = r
                    .run_windowed_maxscore(
                        col,
                        r.build_term_cursors(col, terms, None, false)
                            .await
                            .expect("cursors"),
                        k,
                        None,
                        f32::NEG_INFINITY,
                        lo,
                        hi,
                    )
                    .expect("wms ranged");
                let bmm = r
                    .run_max_score_bmm_range(
                        col,
                        r.build_term_cursors(col, terms, None, false)
                            .await
                            .expect("cursors"),
                        k,
                        lo,
                        hi,
                        None,
                        f32::NEG_INFINITY,
                    )
                    .expect("bmm ranged");
                assert_eq!(wms.len(), bmm.len(), "len [{lo},{hi}) k={k}");
                for ((dw, sw), (db, sb)) in wms.iter().zip(bmm.iter()) {
                    assert_eq!(dw, db, "doc [{lo},{hi}) k={k}: wms={dw} bmm={db}");
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score [{lo},{hi}) k={k}: {sw} vs {sb}"
                    );
                }
            }
        }
    }

    #[test]
    fn filter_survivors_compacts_in_order() {
        // The competitive filter keeps entries scoring >= min_score, preserves
        // ascending order, keeps each doc paired with its score, and returns the
        // survivor count. The 40-element input covers both the SIMD 8-lane body
        // (on x86_64, via the runtime-dispatched AVX2 left-pack) and the scalar
        // tail; on other targets it exercises the scalar fallback end to end.
        let docs0: Vec<u32> = (0..40).collect();
        let scores0: Vec<f32> = (0..40).map(|i| i as f32 * 0.1).collect();
        let min_score = 1.25f32; // keeps i where 0.1*i >= 1.25, i.e. i >= 13
        let mut docs = docs0.clone();
        let mut scores = scores0.clone();
        let n = filter_survivors(&mut docs, &mut scores, min_score);
        let expect: Vec<u32> = docs0
            .iter()
            .copied()
            .filter(|&i| scores0[i as usize] >= min_score)
            .collect();
        assert_eq!(n, expect.len(), "survivor count");
        assert_eq!(&docs[..n], &expect[..], "docs compacted in order");
        for (di, &d) in docs[..n].iter().enumerate() {
            assert!(
                (scores[di] - d as f32 * 0.1).abs() < 1e-6,
                "score stayed paired with its doc"
            );
        }
        // Empty-survivor and all-survivor edges.
        let mut d2 = docs0.clone();
        let mut s2 = scores0.clone();
        assert_eq!(filter_survivors(&mut d2, &mut s2, f32::INFINITY), 0);
        let mut d3 = docs0.clone();
        let mut s3 = scores0.clone();
        assert_eq!(filter_survivors(&mut d3, &mut s3, f32::NEG_INFINITY), 40);
        assert_eq!(&d3[..], &docs0[..], "all-pass leaves order untouched");
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
