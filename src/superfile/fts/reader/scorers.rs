// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS scoring/collection kernels on [`FtsReader`]: the disjunction
//! scorers (WAND+BMW, MaxScore+BMM, windowed, exhaustive) with their
//! dispatch, and the AND flat-merge intersection family. Split from the
//! reader `core` as its own `impl FtsReader` block.

#[cfg(test)]
use std::cell::{Cell, RefCell};
#[cfg(target_arch = "x86_64")]
use std::ops::Range;
use std::{cmp::Ordering, collections::BinaryHeap, mem, slice::from_mut};

use wide::f32x8;

use super::{
    core::*,
    cursor::{CursorUse, TermCursor, seek_sorted},
    filter::ExcludeFilter,
    metadata::{NormTable, unscored_norm_table},
    search::seed_blocks_for,
    sink::{
        AndSink, BufferedScoreSink, CollectSink, Collector, MustShouldCountSink, MustShouldSink,
        ScoreCountSink, ScoreSink, ShouldBatch, TopKBuffer, TopKEntry, drain_top_k_desc,
        offer_top_k, replace_worst,
    },
};
use crate::superfile::{
    error::FtsError,
    fts::{
        bm25,
        posting::{BLOCK_LEN, SUB_BLOCK_LEN},
    },
    id_space::FtsDocId,
};

/// Left-pack control table: for each 8-bit survivor mask, the lane indices that
/// gather the set lanes to the front (for `permutevar8x32`). Unset trailing
/// slots are don't-cares (the store advances only by `popcount(mask)`).
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
    filter_survivors_scalar(docs, scores, min_score)
}

/// [`filter_survivors`] on any target, and where AVX2 is absent.
#[inline(always)]
fn filter_survivors_scalar(docs: &mut [u32], scores: &mut [f32], min_score: f32) -> usize {
    let mut out = 0usize;
    for i in 0..docs.len() {
        let keep = scores[i] >= min_score;
        docs[out] = docs[i];
        scores[out] = scores[i];
        out += keep as usize;
    }
    out
}

/// [`filter_survivors`] over a candidate list, cut to its survivors.
#[inline(always)]
fn retain_survivors(docs: &mut Vec<u32>, scores: &mut Vec<f32>, min_score: f32) {
    let n = filter_survivors(docs, scores, min_score);
    docs.truncate(n);
    scores.truncate(n);
}

/// Complete `docs`' scores with the non-essential `terms`, strongest first:
/// before each term, drop the docs that cannot beat `threshold` even with
/// that term and every later one at its bound (`suffix[i]` bounds `terms[i..]`),
/// so each weaker term is probed over a shrinking list. `tfs` is scratch.
#[inline(always)]
fn complete_survivors(
    terms: &mut [TermCursor],
    suffix: &[f32],
    threshold: f32,
    docs: &mut Vec<u32>,
    scores: &mut Vec<f32>,
    dl_norm: &NormTable,
    tfs: &mut Vec<u32>,
) {
    for (i, c) in terms.iter_mut().enumerate() {
        retain_survivors(docs, scores, threshold - suffix[i]);
        if docs.is_empty() {
            break;
        }
        score_noness_batched(c, docs, scores, dl_norm, tfs);
    }
}

/// [`filter_survivors`] eight entries per step: one compare of eight scores
/// against the bar, then each survivor's doc and score left-packed in place.
///
/// # Safety
///
/// The caller must have detected AVX2 at runtime. Given that, every load and
/// store lies inside the `n`-entry slices, and the in-place left-pack never
/// overwrites an entry before it is read (the write cursor never passes the
/// read cursor, and each group is held in registers before it is stored).
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

#[cfg(test)]
thread_local! {
    /// Every block bound the ranked-union walk computed, with the doc range
    /// it claims to cover. The walk drops a run of blocks when this bound
    /// cannot beat the bar, so the bound must never fall below a real score
    /// inside its range. That is a property of the bound alone: a bound a
    /// little too small drops only docs that would not have placed, leaving
    /// the returned top-k unchanged, so no comparison against a reference
    /// walk can see it. The accompanying test checks the claims directly.
    pub(super) static BOUND_CLAIMS: RefCell<Vec<BoundClaim>> =
        const { RefCell::new(Vec::new()) };
    /// How many times a ranked-union window promoted a non-essential term,
    /// so a test of promotion can show it ran rather than pass vacuously.
    pub(super) static PROMOTIONS: Cell<usize> = const { Cell::new(0) };
}

/// One bound and the `[from, to)` docs it is claimed to cover.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(super) struct BoundClaim {
    pub(super) from: u32,
    pub(super) to: u32,
    pub(super) bound: f32,
}

/// How many more docs a term must have in a union window than the
/// essentials it would be probed for, before the window demotes it from
/// accumulation to probing.
const DEMOTE_DENSITY: f32 = 8.0;

/// Add one non-essential term's contribution to `scores[i]` for every survivor
/// `docs[i]` the term contains. `docs` must be ascending: the probe advances
/// the cursor monotonically. `tfs` is scratch.
#[inline(always)]
pub(super) fn score_noness_batched(
    c: &mut TermCursor,
    docs: &[u32],
    scores: &mut [f32],
    dl_norm: &NormTable,
    tfs: &mut Vec<u32>,
) {
    tfs.clear();
    tfs.resize(docs.len(), 0);
    c.probe_tfs(docs, tfs);
    add_term_scores(c.idf_weight, docs, tfs, dl_norm, scores);
}

/// Add one term's BM25 at each `(docs[i], tfs[i])` to `scores[i]`, eight per
/// SIMD call. Every slot is scored and added without a branch on whether the
/// term holds the doc, which the predictor cannot learn: a tf of 0 scores
/// exactly 0, so it leaves the sum unchanged. The same bits as adding each
/// posting's score alone.
#[inline(always)]
fn add_term_scores(
    idf_weight: f32,
    docs: &[u32],
    tfs: &[u32],
    dl_norm: &NormTable,
    scores: &mut [f32],
) {
    debug_assert_eq!(docs.len(), tfs.len());
    let scores = &mut scores[..docs.len()];
    let mut i = 0usize;
    while i + 8 <= docs.len() {
        let dids: [u32; 8] = docs[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let tf8: [u32; 8] = tfs[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let add = bm25::score_one_term_x8(idf_weight, tf8, dl_norm.get8(dids));
        for (s, a) in scores[i..i + 8].iter_mut().zip(add) {
            *s += a;
        }
        i += 8;
    }
    for ((s, &d), &tf) in scores[i..].iter_mut().zip(&docs[i..]).zip(&tfs[i..]) {
        if tf != 0 {
            *s += bm25::score_with_dl_norm_k1(idf_weight, tf, dl_norm.get(d));
        }
    }
}

/// One term's BM25 at each `(docs[i], tfs[i])` into `out` (cleared first),
/// eight per SIMD call. A tf of 0 scores exactly 0. The same bits as scoring
/// each posting alone.
pub(super) fn score_postings(
    idf_weight: f32,
    docs: &[u32],
    tfs: &[u32],
    dl_norm: &NormTable,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(docs.len(), tfs.len());
    // Sized once and written in place, so a group of eight costs no
    // capacity check.
    out.clear();
    out.resize(docs.len(), 0.0);
    let groups = docs.chunks_exact(8).zip(tfs.chunks_exact(8));
    for ((d8, t8), o8) in groups.zip(out.chunks_exact_mut(8)) {
        let dids: [u32; 8] = d8.try_into().expect("invariant: an 8-wide chunk");
        let tf8: [u32; 8] = t8.try_into().expect("invariant: an 8-wide chunk");
        o8.copy_from_slice(&bm25::score_one_term_x8(
            idf_weight,
            tf8,
            dl_norm.get8(dids),
        ));
    }
    let i = docs.len() / 8 * 8;
    for ((&d, &tf), o) in docs[i..].iter().zip(&tfs[i..]).zip(&mut out[i..]) {
        *o = match tf {
            0 => 0.0,
            _ => bm25::score_with_dl_norm_k1(idf_weight, tf, dl_norm.get(d)),
        };
    }
}

/// The largest doc id any of `cursors` holds.
pub(super) fn cursors_max_doc(cursors: &[TermCursor]) -> u32 {
    cursors
        .iter()
        .filter_map(|c| match c.predecoded {
            true => c.block_doc_ids[..c.block_n].last().copied(),
            false => c.blocks.last().map(|b| b.last_doc_id),
        })
        .max()
        .unwrap_or(0)
}

/// What probing costs, in postings read: the rarest term's every document,
/// times each other term it is tested against, at
/// [`count_probe_postings`] a test.
/// The terms' document frequencies summed, saturating: a df comes from the
/// term's stored metadata, and a corrupt one must not wrap a plan's cost.
pub(super) fn df_total(cursors: &[TermCursor]) -> u64 {
    cursors.iter().fold(0u64, |t, c| t.saturating_add(c.df))
}

pub(super) fn probe_cost(driver_df: u64, probes: usize) -> u64 {
    driver_df
        .saturating_mul(probes as u64)
        .saturating_mul(COUNT_PROBE_POSTINGS)
}

/// What folding costs, in postings read: every term read once into a
/// doc-space bitset, plus each bitset's words written and combined.
pub(super) fn fold_cost(cursors: &[TermCursor], max_doc: u32) -> u64 {
    let words = u64::from(max_doc) / 64 + 1;
    let reads = df_total(cursors);
    words
        .saturating_mul(cursors.len() as u64)
        .saturating_add(reads)
}

/// Intersection cardinality by whichever plan reads less: probe the other
/// terms at each document of the rarest, or fold every term into a bitset
/// and AND them.
pub(super) fn count_and_planned(cursors: Vec<TermCursor>) -> u64 {
    if cursors.len() < 2 {
        return cursors.first().map_or(0, |c| c.df);
    }
    let max_doc = cursors_max_doc(&cursors);
    let min_df = cursors.iter().map(|c| c.df).min().unwrap_or(0);
    match fold_cost(&cursors, max_doc) < probe_cost(min_df, cursors.len() - 1) {
        true => count_excluding_fold(cursors, true, Vec::new(), max_doc),
        false => count_excluding_probe(cursors, true, Vec::new()),
    }
}

/// How many documents match the plain terms `pos` (all of them when `all`,
/// any of them otherwise) and none of `neg`, by whichever plan reads less:
/// walk the positive side and test each of its documents against the
/// excluded terms, or fold both sides into doc-space bitsets and count
/// `positive & !excluded` a word at a time.
pub(super) fn count_excluding(pos: Vec<TermCursor>, all: bool, neg: Vec<TermCursor>) -> u64 {
    if pos.is_empty() {
        return 0;
    }
    if neg.is_empty() {
        return match all {
            true => count_and_planned(pos),
            false => or_count_unranked(pos),
        };
    }
    let max_doc = cursors_max_doc(&pos).max(cursors_max_doc(&neg));
    let (candidates, probes) = match all {
        true => (
            pos.iter().map(|c| c.df).min().unwrap_or(0),
            pos.len() - 1 + neg.len(),
        ),
        false => (df_total(&pos), neg.len()),
    };
    let fold = fold_cost(&pos, max_doc).saturating_add(fold_cost(&neg, max_doc));
    match fold < probe_cost(candidates, probes) {
        true => count_excluding_fold(pos, all, neg, max_doc),
        false => count_excluding_probe(pos, all, neg),
    }
}

/// [`count_excluding`] by folding: positive and excluded sides each become
/// a doc-space bitset, then `popcount(positive & !excluded)`.
fn count_excluding_fold(
    pos: Vec<TermCursor>,
    all: bool,
    neg: Vec<TermCursor>,
    max_doc: u32,
) -> u64 {
    let words = max_doc as usize / 64 + 1;
    with_fold_bitsets(words, |positive, term| {
        match all {
            false => or_cursors_into_bitset(positive, &pos),
            // Each further term is folded into `term` and intersected in,
            // which leaves `term` clear for the next.
            true => {
                let mut scratch = [0u32; BLOCK_LEN];
                let (first, rest) = pos.split_first().expect("invariant: a positive term");
                or_cursor_into_bitset(positive, first, &mut scratch);
                for c in rest {
                    or_cursor_into_bitset(term, c, &mut scratch);
                    for (p, t) in positive.iter_mut().zip(term.iter_mut()) {
                        *p &= mem::take(t);
                    }
                }
            }
        }
        if neg.is_empty() {
            return popcount_and_clear(positive);
        }
        or_cursors_into_bitset(term, &neg);
        positive
            .iter_mut()
            .zip(term.iter_mut())
            .map(|(p, n)| u64::from((mem::take(p) & !mem::take(n)).count_ones()))
            .sum()
    })
}

/// [`count_excluding`] by probing: walk the positive side's documents (the
/// rarest term's, tested against the other positive terms, under `all`;
/// the union's frontier otherwise) and test each against the excluded
/// terms, which only ever move forward.
fn count_excluding_probe(mut pos: Vec<TermCursor>, all: bool, mut neg: Vec<TermCursor>) -> u64 {
    let mut n = 0u64;
    if all {
        let lead = pos
            .iter()
            .enumerate()
            .min_by_key(|(_, c)| c.df)
            .map_or(0, |(i, _)| i);
        let mut driver = pos.swap_remove(lead);
        pos.sort_by_key(|c| c.df);
        while !driver.is_exhausted() {
            let doc = driver.current_doc_id();
            if pos.iter_mut().all(|c| c.contains(doc)) && !neg.iter_mut().any(|c| c.contains(doc)) {
                n += 1;
            }
            driver.next();
        }
        return n;
    }
    while let Some(doc) = pos
        .iter()
        .filter(|c| !c.is_exhausted())
        .map(TermCursor::current_doc_id)
        .min()
    {
        if !neg.iter_mut().any(|c| c.contains(doc)) {
            n += 1;
        }
        for c in pos.iter_mut() {
            if !c.is_exhausted() && c.current_doc_id() == doc {
                c.next();
            }
        }
    }
    n
}

/// Block-Max-AND bound on the **non-leader** terms at the leader's doc, valid
/// over `[leader_doc, window_end]` where `window_end` is the smallest block
/// boundary across all cursors. Each non-leader is bounded by the single block
/// that *contains* `leader_doc`, not a max over every block under the leader's
/// whole (possibly wide) block — the looser range-max collapsed to a common
/// term's global max on rare∧common queries, so the skip rarely fired. The
/// leader's block end is passed in (callers hold the leader split off for the
/// flat-merge); its block max is added by the caller.
fn block_max_and_bound(
    leader_block_end: u32,
    others: &mut [TermCursor],
    leader_doc: u32,
) -> (f32, u32) {
    let mut others_ub = 0.0;
    let mut window_end = leader_block_end;
    for c in others.iter_mut() {
        c.shallow_advance_block_to(leader_doc);
        others_ub += c.inspect_block_max_bm25();
        window_end = window_end.min(c.inspect_block_last_doc_id());
    }
    (others_ub, window_end)
}

/// [`must_lead_should_walk`] with pruning off, for the exact-count search:
/// every doc of the one required term is scored and counted. A block at a
/// time: the required term's postings scored eight-wide, then each optional
/// term probed for the whole block and its contributions added in query
/// order (a missing term adds exactly 0), as the general counted walk adds
/// them, so the scores are the same bits. Only a doc that beats the k-th best
/// (or fills the heap) is offered.
fn must_lead_should_count(
    mut lead: TermCursor,
    mut shoulds: Vec<TermCursor>,
    k: usize,
    dl_norm_k1: &NormTable,
    heap: &mut BinaryHeap<TopKEntry>,
) -> u64 {
    let mut count = 0u64;
    let mut threshold = 0.0f32;
    let mut docs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    let mut scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut tfs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    while !lead.is_exhausted() {
        lead.hold_whole_block();
        let (from, to) = (lead.pos, lead.block_n);
        docs.clear();
        docs.extend_from_slice(&lead.block_doc_ids[from..to]);
        score_postings(
            lead.idf_weight,
            &docs,
            &lead.block_tfs[from..to],
            dl_norm_k1,
            &mut scores,
        );
        for c in &mut shoulds {
            score_noness_batched(c, &docs, &mut scores, dl_norm_k1, &mut tfs);
        }
        count += docs.len() as u64;
        for (&doc, &score) in docs.iter().zip(&scores) {
            offer_top_k(heap, k, &mut threshold, score, doc);
        }
        lead.advance_by(to - from);
    }
    count
}

/// Relative error, in units of `f32::EPSILON` per summed term, by which two
/// sums of the same non-negative terms in different orders can differ: each
/// addition rounds by at most half an ulp, on either sum.
const REASSOCIATION_ULPS_PER_TERM: f32 = 2.0;

/// The bar a bound is pruned against: `bar` lowered by the most a sum of
/// `terms` non-negative scores changes with the order it is added in. A
/// pruning test sums a doc's bounds in one order while its score is summed
/// in another, so a bound can round a few ulps under the score it covers;
/// pruning at this bar never drops a doc that beats `bar`, and admission
/// still compares the real score with the real bar.
#[inline(always)]
pub(super) fn pruning_bar(bar: f32, terms: usize) -> f32 {
    match bar.is_finite() {
        true => bar - bar.abs() * (REASSOCIATION_ULPS_PER_TERM * terms as f32 * f32::EPSILON),
        false => bar,
    }
}

/// The `k`-th best score over the docs `decode` yields block by block — it
/// fills the two buffers and returns how many docs, `None` when done —
/// scored eight at a time into `coll`, counting only docs above `floor`.
/// `NEG_INFINITY` when they hold fewer than `k` such docs: only a full `k`
/// bounds a walk's `k`-th from below.
pub(super) fn kth_scored<C: Collector>(
    mut coll: C,
    k: usize,
    floor: f32,
    idf_weight: f32,
    dl_norm: &NormTable,
    mut decode: impl FnMut(&mut [u32; BLOCK_LEN], &mut [u32; BLOCK_LEN]) -> Option<usize>,
) -> f32 {
    let (mut ids, mut tfs) = ([0u32; BLOCK_LEN], [0u32; BLOCK_LEN]);
    let mut scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut bar = f32::NEG_INFINITY;
    while let Some(n) = decode(&mut ids, &mut tfs) {
        score_postings(idf_weight, &ids[..n], &tfs[..n], dl_norm, &mut scores);
        for (&doc, &score) in ids[..n].iter().zip(&scores) {
            if score > floor {
                coll.offer(k, &mut bar, score, doc);
            }
        }
    }
    match coll.into_sorted().get(k.wrapping_sub(1)) {
        Some(&(_, kth)) => kth,
        None => f32::NEG_INFINITY,
    }
}

/// [`kth_scored`] with no floor, by scoring every doc and selecting the
/// `k`-th best once.
fn kth_selected(
    k: usize,
    idf_weight: f32,
    dl_norm: &NormTable,
    mut decode: impl FnMut(&mut [u32; BLOCK_LEN], &mut [u32; BLOCK_LEN]) -> Option<usize>,
) -> f32 {
    let (mut ids, mut tfs) = ([0u32; BLOCK_LEN], [0u32; BLOCK_LEN]);
    let mut block: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut scores: Vec<f32> = Vec::new();
    while let Some(n) = decode(&mut ids, &mut tfs) {
        score_postings(idf_weight, &ids[..n], &tfs[..n], dl_norm, &mut block);
        scores.extend_from_slice(&block);
    }
    match k.checked_sub(1).filter(|&i| i < scores.len()) {
        Some(i) => *scores.select_nth_unstable_by(i, |a, b| b.total_cmp(a)).1,
        None => f32::NEG_INFINITY,
    }
}

/// `k` from which a multi-term union starts its bar at [`union_seed`]: at a
/// smaller `k` the bar rises within the first windows anyway.
const UNION_SEED_MIN_K: usize = 64;

/// How many times the blocks that could hold `k` docs a union term is seeded
/// from: the best blocks often hold a few of the top docs each, so reading
/// only enough blocks for `k` docs leaves the seed's `k`-th well under the
/// final one, and the walk's bar starts low. Measured at `k` = 1000: four
/// times as many cut the walk most; more cost more to read than they save.
const UNION_SEED_DEPTH: usize = 4;

/// Block tables longer than this are not scanned for a seed: a term that
/// common scores too low per doc to raise the bar, and the scan is linear.
const UNION_SEED_MAX_BLOCKS: usize = 4096;

/// A lower bound on a union's `k`-th best score, before its walk starts: for
/// each term with at least `k` docs, the `k`-th best of its own scores over its
/// highest-bound blocks, the largest of these, one ulp down. A doc's union
/// score is at least its score for any one term, so `k` docs score at least
/// that; one ulp down keeps a doc tied at the bound. `NEG_INFINITY` when no
/// term qualifies.
///
/// A lone term (what a split leaves of a union) is seeded as the single-term
/// walk seeds one, from the [`seed_blocks_for`] best blocks however long its
/// list.
fn union_seed(cursors: &[TermCursor], k: usize, dl_norm: &NormTable) -> f32 {
    let lone = cursors.len() == 1;
    let mut seed = f32::NEG_INFINITY;
    let mut order: Vec<usize> = Vec::new();
    for c in cursors {
        let n_blocks = c.blocks.len();
        if c.predecoded || c.df < k as u64 || (!lone && n_blocks > UNION_SEED_MAX_BLOCKS) {
            continue;
        }
        // The highest-bound blocks — for a term of several,
        // [`UNION_SEED_DEPTH`] times enough to hold `k` docs (every block but
        // the last holds `BLOCK_LEN`) — selected rather than sorted.
        let need = match lone {
            true => seed_blocks_for(k),
            false => UNION_SEED_DEPTH * (k.div_ceil(SUB_BLOCK_LEN) + 1),
        }
        .min(n_blocks);
        order.clear();
        order.extend(0..n_blocks);
        order.select_nth_unstable_by(need - 1, |&a, &b| {
            c.blocks[b]
                .block_max_bm25
                .total_cmp(&c.blocks[a].block_max_bm25)
        });
        let mut blocks = order[..need].iter();
        let decode = |ids: &mut [u32; BLOCK_LEN], tfs: &mut [u32; BLOCK_LEN]| {
            blocks.next().map(|&b| c.decode_block_into(b, ids, tfs))
        };
        // A few blocks' scores are selected from in one pass, cheaper than
        // offering each to a collector; a lone term's many go through a top-k
        // buffer, which keeps only those that can still place. The blocks hold
        // at most `need` × `BLOCK_LEN` docs, which bounds the buffer however
        // large `k` is.
        let kth = match lone {
            true => {
                let cap = k.min(need.saturating_mul(BLOCK_LEN));
                let buffer = <TopKBuffer as Collector>::with_capacity(k, cap);
                kth_scored(buffer, k, f32::NEG_INFINITY, c.idf_weight, dl_norm, decode)
            }
            false => kth_selected(k, c.idf_weight, dl_norm, decode),
        };
        // `NEG_INFINITY` (fewer than `k` docs) stays `NEG_INFINITY`.
        seed = seed.max(kth.next_down());
    }
    seed
}

/// A non-essential union term whose estimated docs in a window are at most
/// this many times the essential terms' is accumulated in that window
/// rather than probed.
const PROMOTE_DENSITY: f32 = 0.5;

/// How many times fewer postings the optional terms must have over a
/// required block's range than the block itself, for the must-led walk to
/// collect theirs and score only the required docs among them.
const OPTIONAL_FIRST_RATIO: f32 = 2.0;

/// Words of the mask [`positions_held`] returns: one bit per lane of a
/// block.
const HELD_WORDS: usize = BLOCK_LEN / HELD_WORD_BITS;
/// Bits per word of that mask.
const HELD_WORD_BITS: usize = u64::BITS as usize;

/// Bit `p` set for each position `p` of `ids` (ascending, at most a
/// block) whose doc is in `docs` (ascending), as little-endian words.
fn positions_held(ids: &[u32], docs: &[u32]) -> [u64; HELD_WORDS] {
    debug_assert!(ids.len() <= BLOCK_LEN);
    // `docs` is the shorter list (the caller collects it only when it is):
    // walk it and search forward in `ids`, rather than stepping both.
    let (mut p, mut bits) = (0usize, [0u64; HELD_WORDS]);
    for &d in docs {
        p = seek_sorted::<true>(ids, p, d);
        if p >= ids.len() {
            break;
        }
        bits[p / HELD_WORD_BITS] |= u64::from(ids[p] == d) << (p % HELD_WORD_BITS);
    }
    bits
}

/// The tf at each of `docs` (ascending) in a term whose postings over their
/// range are `(term_docs, term_tfs)`, `0` where it holds none, into `out`.
fn tfs_at(docs: &[u32], term_docs: &[u32], term_tfs: &[u32], out: &mut Vec<u32>) {
    out.clear();
    let mut q = 0usize;
    for &d in docs {
        q += term_docs[q..].partition_point(|&x| x < d);
        out.push(match term_docs.get(q) {
            Some(&x) if x == d => term_tfs[q],
            _ => 0,
        });
    }
}

/// One required term with optional terms: the required term drives, one
/// block at a time, and the optional terms are only probed at its docs.
///
/// A required block is skipped whole when its own block maximum plus each
/// optional term's maximum **over that block's doc range** cannot beat the
/// k-th best. Bounding the optional terms by their maxima over the range
/// rather than over their whole lists is what lets a common optional term
/// stop holding every block open. Inside a block, each candidate is
/// rejected before any probe when its required score plus the optional
/// terms' block maxima at that doc cannot beat the bar, and abandoned
/// part-way through the probes once what is still unprobed cannot close the
/// gap, strongest optional term first.
///
/// Docs arrive in ascending order and a tie keeps the smaller doc, so a
/// candidate whose bound only equals the k-th best can never enter the heap:
/// pruning on `<=` drops nothing that would have been kept.
fn must_lead_should_walk<C: Collector>(
    mut lead: TermCursor,
    mut shoulds: Vec<TermCursor>,
    k: usize,
    mut filter: Option<&mut ExcludeFilter>,
    floor_eff: f32,
    dl_norm_k1: &NormTable,
    heap: &mut C,
) {
    // Probe the optional terms strongest first (the bound on what is still
    // unprobed then falls fastest), but add their scores in query order, as
    // the general must+should walk does, so a score is the same bits on
    // either path. The block batch probes in query order instead from
    // [`BUFFERED_TOPK_MIN_K`] (see there).
    let m = shoulds.len();
    let query_order = k >= BUFFERED_TOPK_MIN_K;
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_unstable_by(|&a, &b| {
        shoulds[b]
            .term_max_bm25
            .partial_cmp(&shoulds[a].term_max_bm25)
            .unwrap_or(Ordering::Equal)
    });
    debug_assert!(
        m <= MAX_SHOULDS_LED,
        "caller routes wider should lists elsewhere"
    );
    let mut contrib = [0.0_f32; MAX_SHOULDS_LED];
    let mut doc_ub = vec![0.0_f32; m];
    let mut doc_suffix = vec![0.0_f32; m + 1];
    // The bar a doc must beat: the collector's k-th best once it is full (a
    // heap keeps it exact, a buffer at its last cut, never above the true
    // one), else the floor (`NEG_INFINITY` when there is none, which prunes
    // nothing), lowered for pruning (see [`pruning_bar`]): bounds are summed
    // strongest term first and scores in query order.
    let mut kth = f32::NEG_INFINITY;
    let bar = |heap: &C, kth: f32| -> f32 {
        let bar = match heap.is_full(k) {
            true => kth.max(floor_eff),
            false => floor_eff,
        };
        pruning_bar(bar, m + 1)
    };
    let mut threshold = bar(heap, kth);
    // The block batch's scratch: the block's candidates, their required
    // scores, the docs still alive with their running sums, the optional
    // terms' tfs at them, and each optional term's maximum over the block
    // with its suffix sums in query order.
    let mut cand_docs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    let mut cand_scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut partial: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut alive: Vec<u16> = Vec::with_capacity(BLOCK_LEN);
    // The candidates still alive after an optional term, written apart from
    // `alive` so the compaction never reads a slot it may just have written.
    // A fixed array and a count in a register, as a `Vec`'s length would
    // sit in memory between a push and its truncate.
    let mut alive_next = [0u16; BLOCK_LEN];
    let mut contribs: Vec<f32> = Vec::with_capacity(m * BLOCK_LEN);
    let mut probe_docs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    let mut cand_tfs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    let mut cand_contribs: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    // Each optional term's postings over a block's range, for optional-first.
    let mut opt_docs: Vec<Vec<u32>> = vec![Vec::with_capacity(BLOCK_LEN); m];
    let mut opt_tfs: Vec<Vec<u32>> = vec![Vec::with_capacity(BLOCK_LEN); m];
    // The optional terms by their maximum over a block's range, ascending,
    // and which of them drive the block.
    let mut by_win: Vec<usize> = Vec::with_capacity(m);
    let mut driver: Vec<bool> = Vec::with_capacity(m);
    let mut win = vec![0.0_f32; m];
    let mut win_suffix = vec![0.0_f32; m + 1];
    while !lead.is_exhausted() {
        let start = lead.current_doc_id();
        let block_end = lead.current_block_last_doc_id();
        let pruning = threshold > f32::NEG_INFINITY;
        if pruning {
            for (w, c) in win.iter_mut().zip(shoulds.iter_mut()) {
                *w = c.block_max_in_range(start, block_end);
            }
            let window: f32 = win.iter().sum();
            if lead.current_block_max_bm25() + window <= threshold {
                lead.skip_to(block_end.saturating_add(1));
                continue;
            }
        }
        // The rest of the required block at once: score it, keep the docs
        // that can clear the bar with every optional term at its maximum over
        // the block, and probe the optional terms (strongest first below
        // [`BUFFERED_TOPK_MIN_K`], in query order from it), each in one
        // sorted pass over the docs still alive, dropping a doc once what it
        // has plus what is still unprobed cannot clear the bar. That is the
        // walk's own test against bounds never tighter than its per-doc ones
        // (each a block maximum, summed the same way) and a bar never higher
        // (the block's opening one), so it drops only docs the walk drops.
        // Contributions are summed in query order, as the walk sums them, a
        // missing term adding exactly 0, and every offer is checked as the
        // walk checks it, so the heap ends the same.
        if filter.is_none() && lead.holds_whole_block() {
            let (from, to) = (lead.pos, lead.block_n);
            // Optional first. When the required term's block maximum alone
            // cannot beat the bar, a doc of this block reaches the top-k only
            // with some optional term, so the candidates are the required
            // docs the optional terms hold. When those terms have far fewer
            // postings over the block's range, collect theirs and score only
            // the required docs among them, instead of scoring the whole block
            // and probing at every doc. A required-only doc scores at most the
            // block maximum, which cannot enter, so the heap ends the same.
            let lead_max = lead.current_block_max_bm25();
            if pruning && lead_max <= threshold {
                let range_end = lead.block_doc_ids[to - 1];
                // The drivers: the optional terms a doc needs. The weakest
                // over the block's range are left out while the required
                // maximum plus theirs still cannot beat the bar, so a doc
                // holding none of the drivers cannot enter. Only the drivers'
                // postings are collected; the others are probed at the docs
                // the drivers find.
                by_win.clear();
                by_win.extend(0..m);
                by_win.sort_unstable_by(|&a, &b| win[a].total_cmp(&win[b]));
                driver.clear();
                driver.resize(m, true);
                let mut reach = lead_max;
                for &si in &by_win {
                    if reach + win[si] > threshold {
                        break;
                    }
                    reach += win[si];
                    driver[si] = false;
                }
                let optional_docs: f32 = (0..m)
                    .filter(|&si| driver[si])
                    .map(|si| {
                        shoulds[si]
                            .block_max_and_density_in_range(start, range_end)
                            .1
                    })
                    .sum();
                if optional_docs * OPTIONAL_FIRST_RATIO < (to - from) as f32 {
                    let ids = &lead.block_doc_ids[from..to];
                    let mut held = [0u64; HELD_WORDS];
                    for (si, c) in shoulds.iter_mut().enumerate() {
                        opt_docs[si].clear();
                        opt_tfs[si].clear();
                        if driver[si] {
                            c.collect_range(start, range_end, &mut opt_docs[si], &mut opt_tfs[si]);
                            for (h, w) in held.iter_mut().zip(positions_held(ids, &opt_docs[si])) {
                                *h |= w;
                            }
                        }
                    }
                    cand_docs.clear();
                    cand_tfs.clear();
                    for (wi, &word) in held.iter().enumerate() {
                        let mut bits = word;
                        while bits != 0 {
                            let p = wi * HELD_WORD_BITS + bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            cand_docs.push(ids[p]);
                            cand_tfs.push(lead.block_tfs[from + p]);
                        }
                    }
                    score_postings(
                        lead.idf_weight,
                        &cand_docs,
                        &cand_tfs,
                        dl_norm_k1,
                        &mut cand_scores,
                    );
                    // Optional contributions in query order, as the walk adds them.
                    for si in 0..m {
                        let c = &mut shoulds[si];
                        match driver[si] {
                            true => {
                                tfs_at(&cand_docs, &opt_docs[si], &opt_tfs[si], &mut cand_tfs);
                                add_term_scores(
                                    c.idf_weight,
                                    &cand_docs,
                                    &cand_tfs,
                                    dl_norm_k1,
                                    &mut cand_scores,
                                );
                            }
                            false => score_noness_batched(
                                c,
                                &cand_docs,
                                &mut cand_scores,
                                dl_norm_k1,
                                &mut cand_tfs,
                            ),
                        }
                    }
                    for (&d, &score) in cand_docs.iter().zip(&cand_scores) {
                        if score > floor_eff {
                            heap.offer(k, &mut kth, score, d);
                            threshold = bar(heap, kth);
                        }
                    }
                    lead.advance_by(to - from);
                    continue;
                }
            }
            cand_docs.clear();
            cand_docs.extend_from_slice(&lead.block_doc_ids[from..to]);
            score_postings(
                lead.idf_weight,
                &cand_docs,
                &lead.block_tfs[from..to],
                dl_norm_k1,
                &mut cand_scores,
            );
            lead.advance_by(to - from);
            let n = cand_docs.len();
            if m == 1 {
                // One optional term, the rare-led pair: filter by its block
                // maximum, probe it once, offer; none of the per-term
                // bookkeeping below.
                let mut w = 0usize;
                for r in 0..n {
                    let (d, s) = (cand_docs[r], cand_scores[r]);
                    cand_docs[w] = d;
                    cand_scores[w] = s;
                    w += usize::from(!(pruning && s + win[0] <= threshold));
                }
                cand_docs.truncate(w);
                cand_scores.truncate(w);
                let should = &mut shoulds[0];
                cand_tfs.clear();
                cand_tfs.resize(w, 0);
                should.probe_tfs(&cand_docs, &mut cand_tfs);
                score_postings(
                    should.idf_weight,
                    &cand_docs,
                    &cand_tfs,
                    dl_norm_k1,
                    &mut cand_contribs,
                );
                for ((&d, &lead_score), &contrib) in
                    cand_docs.iter().zip(&cand_scores).zip(&cand_contribs)
                {
                    let score = lead_score + contrib;
                    if threshold > f32::NEG_INFINITY && score <= threshold {
                        continue;
                    }
                    if score > floor_eff {
                        heap.offer(k, &mut kth, score, d);
                        threshold = bar(heap, kth);
                    }
                }
                continue;
            }
            // At a large `k` the bar stays low and drops few docs: the optional
            // terms go in query order, where a running sum is the score itself.
            // At a smaller `k` strongest first collapses the bound on what is
            // still unprobed fastest and drops docs sooner; their contributions
            // are then kept per term and summed in query order at the end.
            // Either way `win_suffix[r]` bounds the terms from the `r`-th probed
            // on, and the scored docs end in `probe_docs` and `partial`.
            if pruning {
                win_suffix[m] = 0.0;
                for r in (0..m).rev() {
                    let term = if query_order { r } else { order[r] };
                    win_suffix[r] = win_suffix[r + 1] + win[term];
                }
            }
            if query_order {
                // Each term probed over the docs still alive and added into
                // their running sums in place: a sum is then the score itself,
                // the same bits the general walk adds, and the docs that cannot
                // reach the bar with the remaining terms at their block maxima
                // are dropped between terms.
                probe_docs.clear();
                probe_docs.extend_from_slice(&cand_docs);
                partial.clear();
                partial.extend_from_slice(&cand_scores);
                for si in 0..m {
                    if pruning {
                        retain_survivors(&mut probe_docs, &mut partial, threshold - win_suffix[si]);
                    }
                    if probe_docs.is_empty() {
                        break;
                    }
                    score_noness_batched(
                        &mut shoulds[si],
                        &probe_docs,
                        &mut partial,
                        dl_norm_k1,
                        &mut cand_tfs,
                    );
                }
            } else {
                alive.clear();
                for (i, &s) in cand_scores.iter().enumerate() {
                    alive.push(i as u16);
                    alive.truncate(
                        alive.len() - usize::from(pruning && s + win_suffix[0] <= threshold),
                    );
                }
                partial.clear();
                partial.extend_from_slice(&cand_scores);
                contribs.clear();
                contribs.resize(m * n, 0.0);
                for (rank, &si) in order.iter().enumerate() {
                    if alive.is_empty() {
                        break;
                    }
                    let c = &mut shoulds[si];
                    probe_docs.clear();
                    probe_docs.extend(alive.iter().map(|&i| cand_docs[usize::from(i)]));
                    cand_tfs.clear();
                    cand_tfs.resize(probe_docs.len(), 0);
                    c.probe_tfs(&probe_docs, &mut cand_tfs);
                    score_postings(
                        c.idf_weight,
                        &probe_docs,
                        &cand_tfs,
                        dl_norm_k1,
                        &mut cand_contribs,
                    );
                    let rest = win_suffix[rank + 1];
                    let mut w = 0usize;
                    for (&a, &x) in alive.iter().zip(&cand_contribs) {
                        let i = usize::from(a);
                        contribs[si * n + i] = x;
                        let p = partial[i] + x;
                        partial[i] = p;
                        alive_next[w] = a;
                        w += usize::from(!(pruning && p + rest <= threshold));
                    }
                    alive.clear();
                    alive.extend_from_slice(&alive_next[..w]);
                }
                // The survivors' scores re-summed in query order.
                probe_docs.clear();
                partial.clear();
                for &i in &alive {
                    let i = usize::from(i);
                    probe_docs.push(cand_docs[i]);
                    partial.push(
                        (0..m).fold(cand_scores[i], |score, si| score + contribs[si * n + i]),
                    );
                }
            }
            for (&d, &score) in probe_docs.iter().zip(&partial) {
                if threshold > f32::NEG_INFINITY && score <= threshold {
                    continue;
                }
                if score > floor_eff {
                    heap.offer(k, &mut kth, score, d);
                    threshold = bar(heap, kth);
                }
            }
            continue;
        }
        let mut have_bound = false;
        let mut bound_holds_to = 0u32;
        let mut others_ub = 0.0_f32;
        while !lead.is_exhausted() && lead.current_doc_id() <= block_end {
            let candidate = lead.current_doc_id();
            if let Some(f) = filter.as_deref_mut()
                && !f.admits(candidate)
            {
                lead.next();
                continue;
            }
            let norm = dl_norm_k1.get(candidate);
            let lead_score = bm25::score_with_dl_norm_k1(lead.idf_weight, lead.current_tf(), norm);
            let pruning = threshold > f32::NEG_INFINITY;
            if pruning {
                if !have_bound || candidate > bound_holds_to {
                    others_ub = 0.0;
                    bound_holds_to = u32::MAX;
                    for (rank, &si) in order.iter().enumerate() {
                        let c = &mut shoulds[si];
                        c.shallow_advance_block_to(candidate);
                        doc_ub[rank] = c.inspect_block_max_bm25();
                        others_ub += doc_ub[rank];
                        bound_holds_to = bound_holds_to.min(c.inspect_block_last_doc_id());
                    }
                    doc_suffix[m] = 0.0;
                    for i in (0..m).rev() {
                        doc_suffix[i] = doc_suffix[i + 1] + doc_ub[i];
                    }
                    have_bound = true;
                }
                if lead_score + others_ub <= threshold {
                    lead.next();
                    continue;
                }
            }
            let mut partial = lead_score;
            let mut abandoned = false;
            for (rank, &si) in order.iter().enumerate() {
                let c = &mut shoulds[si];
                // Every probed slot is written, hit or miss, so no clear is
                // needed; a doc's slots are read only when none was skipped.
                contrib[si] = match c.bitset_probe_tf(candidate) {
                    Some(tf) => bm25::score_with_dl_norm_k1(c.idf_weight, tf, norm),
                    None => 0.0,
                };
                partial += contrib[si];
                if pruning && partial + doc_suffix[rank + 1] <= threshold {
                    abandoned = true;
                    break;
                }
            }
            let mut score = lead_score;
            for &x in &contrib[..m] {
                score += x;
            }
            if !abandoned && score > floor_eff {
                heap.offer(k, &mut kth, score, candidate);
                threshold = bar(heap, kth);
            }
            lead.next();
        }
    }
}

/// The scored rarest-driven intersection, a driver block at a time: skip a
/// block whose driver maximum plus the other terms' maxima over its range
/// cannot beat the bar; otherwise score the block's docs in one batch, drop
/// those that cannot reach the bar with every other term at its window
/// maximum, and complete the others one term at a time over the survivors,
/// rarest first, a doc leaving at the first term it lacks and the survivors
/// re-filtered against what the terms still to come could add. Terms are
/// added driver first and then rarest first, the order the per-document walk
/// adds them in, so scores are the same bits.
fn and_membership_scored_batched<S: AndSink>(
    mut driver: TermCursor,
    mut others: Vec<TermCursor>,
    dl_norm_k1: &NormTable,
    sink: &mut S,
) {
    let m = others.len();
    let mut window = vec![0.0_f32; m];
    let mut suffix = vec![0.0_f32; m + 1];
    let mut docs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    let mut scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
    let mut tfs: Vec<u32> = Vec::with_capacity(BLOCK_LEN);
    while !driver.is_exhausted() {
        let bar = pruning_bar(sink.bar(), m + 1);
        let pruning = bar > f32::NEG_INFINITY;
        let start = driver.current_doc_id();
        let block_end = driver.current_block_last_doc_id();
        if pruning {
            for (w, c) in window.iter_mut().zip(others.iter_mut()) {
                *w = c.block_max_in_range(start, block_end);
            }
            suffix[m] = 0.0;
            for i in (0..m).rev() {
                suffix[i] = suffix[i + 1] + window[i];
            }
            if driver.current_block_max_bm25() + suffix[0] <= bar {
                driver.skip_to(block_end.saturating_add(1));
                continue;
            }
        }
        driver.hold_whole_block();
        let (from, to) = (driver.pos, driver.block_n);
        docs.clear();
        docs.extend_from_slice(&driver.block_doc_ids[from..to]);
        score_postings(
            driver.idf_weight,
            &docs,
            &driver.block_tfs[from..to],
            dl_norm_k1,
            &mut scores,
        );
        driver.advance_by(to - from);
        if pruning {
            retain_survivors(&mut docs, &mut scores, bar - suffix[0]);
        }
        for (i, o) in others.iter_mut().enumerate() {
            if docs.is_empty() {
                break;
            }
            tfs.resize(docs.len(), 0);
            o.probe_tfs(&docs, &mut tfs);
            // Score every candidate, eight per SIMD call, and compact without
            // a branch: whether a candidate holds the term is a coin flip the
            // predictor cannot learn. A missing term's tf of 0 scores exactly
            // 0, and the slot it writes is overwritten by the next keeper.
            let mut w = 0usize;
            let mut r = 0usize;
            while r + 8 <= docs.len() {
                let dids: [u32; 8] = docs[r..r + 8]
                    .try_into()
                    .expect("invariant: an 8-wide slice");
                let tf8: [u32; 8] = tfs[r..r + 8]
                    .try_into()
                    .expect("invariant: an 8-wide slice");
                let add = bm25::score_one_term_x8(o.idf_weight, tf8, dl_norm_k1.get8(dids));
                for l in 0..8 {
                    let total = scores[r + l] + add[l];
                    docs[w] = dids[l];
                    scores[w] = total;
                    w += usize::from(tf8[l] != 0);
                }
                r += 8;
            }
            for r in r..docs.len() {
                let (d, tf) = (docs[r], tfs[r]);
                if tf != 0 {
                    docs[w] = d;
                    scores[w] = scores[r]
                        + bm25::score_with_dl_norm_k1(o.idf_weight, tf, dl_norm_k1.get(d));
                    w += 1;
                }
            }
            docs.truncate(w);
            scores.truncate(w);
            if pruning && i + 1 < m {
                retain_survivors(&mut docs, &mut scores, bar - suffix[i + 1]);
            }
        }
        for (&d, &s) in docs.iter().zip(scores.iter()) {
            sink.emit(d, s);
        }
    }
}

/// Add `N` ascending documents' contributions into a window and mark them
/// present. Consecutive documents of a block usually share a presence word,
/// and `N` read-modify-writes of one word wait on each other's stores; when
/// the first and last share it, one combined write replaces them.
#[inline(always)]
fn mark_lanes<const N: usize>(
    scores: &mut [f32],
    present: &mut [u64],
    base: u32,
    doc_ids: [u32; N],
    contributions: [f32; N],
) {
    debug_assert!(doc_ids.iter().all(|&d| d - base < OR_WINDOW));
    // Sized views and in-window masks let the compiler drop the per-lane
    // bounds checks; the mask is a no-op for a doc inside the window.
    let scores = &mut scores[..OR_WINDOW as usize];
    let present = &mut present[..OR_WINDOW_WORDS];
    let local = doc_ids.map(|d| (d - base) as usize & (OR_WINDOW as usize - 1));
    for lane in 0..N {
        scores[local[lane]] += contributions[lane];
    }
    let (first, last) = (local[0] >> 6, local[N - 1] >> 6);
    if first == last {
        let mut bits = 0u64;
        for l in local {
            bits |= 1u64 << (l & 63);
        }
        present[first] |= bits;
    } else {
        for l in local {
            present[l >> 6] |= 1u64 << (l & 63);
        }
    }
}

/// Whether any of `scores` is above `threshold`, compared lane-wise eight at
/// a time so it vectorizes; a running `max` over floats does not, as
/// `f32::max` cannot be reassociated.
#[inline(always)]
fn any_beats(scores: &[f32], threshold: f32) -> bool {
    let mut hit = [0u32; 8];
    let mut chunks = scores.chunks_exact(8);
    for chunk in chunks.by_ref() {
        for (h, &x) in hit.iter_mut().zip(chunk) {
            *h |= u32::from(x > threshold);
        }
    }
    hit.iter().fold(0, |a, &h| a | h) != 0 || chunks.remainder().iter().any(|&x| x > threshold)
}

/// Whether any of one presence word's 64 window slots scores above
/// `threshold`: [`any_beats`] over a length fixed at compile time, so the
/// lanes unroll with no tail. Slots hold non-negative scores and the bar is
/// never negative, so this is exactly "the word's best slot beats the bar".
#[inline(always)]
fn word_beats(slots: &[f32], threshold: f32) -> bool {
    let slots: &[f32; 64] = slots
        .try_into()
        .expect("invariant: a presence word spans 64 slots");
    any_beats(slots, threshold)
}

/// Add `c`'s current posting into a union window whose slot 0 is doc
/// `base`, then move the cursor past it: the path for a lazily held block,
/// whose buffer shows one doc. The cursor's current doc must be at or past
/// `base` and before the window's end.
#[inline(always)]
fn accumulate_one(
    c: &mut TermCursor,
    base: u32,
    scores: &mut [f32],
    present: &mut [u64],
    dl_norm_k1: &NormTable,
) {
    let d = c.current_doc_id();
    let local = (d - base) as usize;
    scores[local] += bm25::score_with_dl_norm_k1(c.idf_weight, c.current_tf(), dl_norm_k1.get(d));
    present[local >> 6] |= 1u64 << (local & 63);
    c.next();
}

/// Add the rest of the cursor's block inside the window at once: the
/// postings before `window_end` found once, scored and marked a run at a
/// time with no per-posting bound or block checks, the cursor moved past
/// them once. Each posting gets the same operations as scored alone, so the
/// same bits. A lazily held block, whose buffer shows a single doc, adds
/// that one posting.
#[inline(always)]
fn accumulate_segment(
    c: &mut TermCursor,
    base: u32,
    window_end: u32,
    scores: &mut [f32],
    present: &mut [u64],
    dl_norm_k1: &NormTable,
) {
    if !c.holds_whole_block() {
        accumulate_one(c, base, scores, present, dl_norm_k1);
        return;
    }
    let (pos, n) = (c.pos, c.block_n);
    // The cursor's doc is inside the window, so at least it is added.
    let to = match c.block_doc_ids[n - 1] < window_end {
        true => n,
        false => pos + c.block_doc_ids[pos..n].partition_point(|&d| d < window_end),
    };
    let mut i = pos;
    #[cfg(target_arch = "x86_64")]
    if avx512_kernels() {
        if let Some((bit_base, bitset)) = c.whole_bitset() {
            let ids = &c.block_doc_ids[..n];
            // SAFETY: both features the kernel enables are present (checked
            // by `avx512_kernels`).
            let added = unsafe {
                accumulate_bitset_avx512(
                    c.idf_weight,
                    BitsetRun {
                        words: &c.bytes[bitset],
                        bit_base,
                        docs: ids[pos]..ids[to - 1] + 1,
                        tfs: &c.block_tfs[pos..to],
                    },
                    base,
                    scores,
                    present,
                    dl_norm_k1,
                )
            };
            debug_assert_eq!(added, to - pos, "one tf per doc of the bitset");
            c.advance_by(to - pos);
            return;
        }
        // SAFETY: as above.
        i += unsafe {
            accumulate_run_avx512(
                c.idf_weight,
                &c.block_doc_ids[pos..to],
                &c.block_tfs[pos..to],
                base,
                scores,
                present,
                dl_norm_k1,
            )
        };
    }
    let (ids, tfs) = (&c.block_doc_ids[..n], &c.block_tfs[..n]);
    while i + 8 <= to {
        let dids: [u32; 8] = ids[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let tf8: [u32; 8] = tfs[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let contributions = bm25::score_one_term_x8(c.idf_weight, tf8, dl_norm_k1.get8(dids));
        mark_lanes(scores, present, base, dids, contributions);
        i += 8;
    }
    for j in i..to {
        let d = ids[j];
        let local = (d - base) as usize;
        scores[local] += bm25::score_with_dl_norm_k1(c.idf_weight, tfs[j], dl_norm_k1.get(d));
        present[local >> 6] |= 1u64 << (local & 63);
    }
    c.advance_by(to - pos);
}

#[cfg(test)]
thread_local! {
    /// Run the union window's portable accumulate even where the AVX-512
    /// kernels could, so a test can compare the two paths' windows.
    pub(super) static PORTABLE_ACCUMULATE: Cell<bool> = const { Cell::new(false) };
}

/// Whether the union window's AVX-512 kernels run here: the CPU has both
/// features they enable.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn avx512_kernels() -> bool {
    #[cfg(test)]
    if PORTABLE_ACCUMULATE.with(Cell::get) {
        return false;
    }
    std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("avx2")
}

/// The aligned unit both AVX-512 window kernels add into: one 16-lane
/// register of scores. The window's width is a multiple of it.
const ACCUMULATE_TILE: usize = 16;
const _: () = assert!((OR_WINDOW as usize).is_multiple_of(ACCUMULATE_TILE));

/// The eight-wide loop of [`accumulate_segment`] over one in-window run of
/// a block's postings, returning how many it consumed (a multiple of eight).
/// Eight postings of a dense term usually span fewer than 16 documents; their
/// contributions are then expanded under a bit mask into the aligned tile
/// ([`ACCUMULATE_TILE`]) holding the first, and the next when they cross its
/// end, each added with one 16-wide read-modify-write in place of eight
/// scalar ones, and the same mask marks them present. Their norms' buckets are then 16
/// contiguous bytes too: one load, compressed under the mask to the eight
/// docs' buckets, and one gather from the decode table, in place of eight
/// dependent loads of each. Every slot still receives exactly its own
/// contribution (the others add `0.0` to a non-negative score), at the norm
/// [`NormTable::get`] gives, so the window holds the same bits as through
/// [`mark_lanes`].
///
/// # Safety
///
/// The caller must have detected both AVX-512F and AVX2 at runtime. Given
/// that, every access is bounds-checked before it is made: a tile's
/// read-modify-write only when the run's last doc is inside the window (a
/// tile is aligned to a divisor of the window's width, so every tile the run
/// touches is inside it), and a 16-byte bucket load only when it fits the
/// norm table; gather indices
/// are bytes into a 256-entry table. Ids that are not ascending give wrong
/// sums or a panic, never an out-of-bounds access.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn accumulate_run_avx512(
    idf_weight: f32,
    ids: &[u32],
    tfs: &[u32],
    base: u32,
    scores: &mut [f32],
    present: &mut [u64],
    dl_norm_k1: &NormTable,
) -> usize {
    use std::arch::x86_64::*;
    let scores = &mut scores[..OR_WINDOW as usize];
    let present = &mut present[..OR_WINDOW_WORDS];
    let (buckets, lut) = dl_norm_k1.parts();
    let mut i = 0usize;
    while i + 8 <= ids.len() {
        let dids: [u32; 8] = ids[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let tf8: [u32; 8] = tfs[i..i + 8]
            .try_into()
            .expect("invariant: an 8-wide slice");
        let first = (dids[0] - base) as usize;
        if dids[7] - dids[0] < 16
            && ((dids[7] - base) as usize) < OR_WINDOW as usize
            && dids[0] as usize + 16 <= buckets.len()
        {
            let mut mask = 0u32;
            for d in dids {
                mask |= 1 << (d - dids[0]);
            }
            // SAFETY: avx512f and avx2 are enabled on this function, which the
            // caller reaches only after detecting both at runtime (a virtual
            // machine may mask one off). The 16-byte bucket load covers
            // `[dids[0], dids[0] + 16)`, inside `buckets` by the check above.
            // The compress keeps the eight lanes the mask selects, in order, so
            // the low eight lanes are the eight docs' buckets, each below 256:
            // the gather reads only inside the 256-entry table.
            let norms = unsafe {
                let bytes = _mm_loadu_si128(buckets.as_ptr().add(dids[0] as usize).cast());
                let picked = _mm512_maskz_compress_epi32(mask as u16, _mm512_cvtepu8_epi32(bytes));
                let gathered =
                    _mm256_i32gather_ps::<4>(lut.as_ptr(), _mm512_castsi512_si256(picked));
                let mut norms = [0.0f32; 8];
                _mm256_storeu_ps(norms.as_mut_ptr(), gathered);
                norms
            };
            let contributions = bm25::score_one_term_x8(idf_weight, tf8, norms);
            // The eight docs land in the 16-slot tile holding the first and,
            // when they cross its end, the next: tiles aligned to 16 slots, so
            // the next run reads either a tile this one wrote whole, which the
            // store forwards, or one it did not touch, never a partial overlap
            // that would wait for the store to drain.
            let tile = first & !(ACCUMULATE_TILE - 1);
            let wide = mask << (first - tile);
            let (low, high) = (wide as u16, (wide >> ACCUMULATE_TILE) as u16);
            // SAFETY: as above for the target features. The last doc's slot is
            // inside the window by the check above, and tiles are aligned to a
            // divisor of the window's width, so each tile written — the first's,
            // and the next only when a doc falls in it — lies inside the window
            // slice; the loads and stores are unaligned forms. Each expand
            // reads only as many low source lanes as its mask has bits, at most
            // the eight the 256-bit load defined; the second takes the lanes
            // past the first tile's, moved down by the permute.
            unsafe {
                let src = _mm512_castps256_ps512(_mm256_loadu_ps(contributions.as_ptr()));
                let slot = scores.as_mut_ptr().add(tile);
                let add = _mm512_maskz_expand_ps(low, src);
                _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), add));
                if high != 0 {
                    let past = _mm512_set1_epi32(low.count_ones() as i32);
                    let lanes = _mm512_add_epi32(
                        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                        past,
                    );
                    let rest = _mm512_permutexvar_ps(lanes, src);
                    let add = _mm512_maskz_expand_ps(high, rest);
                    let slot = slot.add(ACCUMULATE_TILE);
                    _mm512_storeu_ps(slot, _mm512_add_ps(_mm512_loadu_ps(slot), add));
                }
            }
            let (word, bit) = (first >> 6, first & 63);
            present[word] |= u64::from(mask) << bit;
            // A mask starting past bit 48 may reach into the next word; when
            // it does, that word holds one of the run's docs, so it exists.
            let spill = match bit {
                0 => 0,
                _ => u64::from(mask) >> (64 - bit),
            };
            if spill != 0 {
                present[word + 1] |= spill;
            }
        } else {
            let contributions = bm25::score_one_term_x8(idf_weight, tf8, dl_norm_k1.get8(dids));
            mark_lanes(scores, present, base, dids, contributions);
        }
        i += 8;
    }
    i
}

/// The docs of a bitset block one window adds: bit `i` of `words` stands
/// for doc `bit_base + i`; the docs in `docs` are added, each with the next
/// of `tfs` in doc order.
#[cfg(target_arch = "x86_64")]
struct BitsetRun<'a> {
    words: &'a [u8],
    bit_base: u32,
    docs: Range<u32>,
    tfs: &'a [u32],
}

#[cfg(target_arch = "x86_64")]
impl BitsetRun<'_> {
    /// The run's docs among `[tile, tile + 16)`, as a mask of lanes.
    #[inline(always)]
    fn tile_mask(&self, tile: u32) -> u16 {
        let off = i64::from(tile) - i64::from(self.bit_base);
        let (at, shift) = (off.div_euclid(8), off.rem_euclid(8));
        // Inside the bitset, one four-byte read holds the tile's bits; at
        // its edges, the bytes past it read as empty.
        let bits = match usize::try_from(at)
            .ok()
            .and_then(|at| self.words.get(at..at + 4))
        {
            Some(four) => u32::from_le_bytes(four.try_into().expect("invariant: four bytes")),
            None => {
                let byte = |i: i64| match usize::try_from(i) {
                    Ok(i) => self.words.get(i).map_or(0, |&b| u32::from(b)),
                    Err(_) => 0,
                };
                byte(at) | byte(at + 1) << 8 | byte(at + 2) << 16
            }
        };
        let mut mask = (bits >> shift) as u16;
        // Lanes before the run's first doc or past its last are dropped.
        let width = ACCUMULATE_TILE as u32;
        let lo = self.docs.start.saturating_sub(tile).min(width);
        let hi = self.docs.end.saturating_sub(tile).min(width);
        mask &= ((1u32 << hi) - 1) as u16 & !(((1u32 << lo) - 1) as u16);
        mask
    }
}

/// [`accumulate_run_avx512`] for a block stored as a presence bitset, which
/// a common term's dense blocks are: the window is walked in its aligned
/// 16-slot tiles, and a tile's docs are read off the bitset as a lane mask
/// rather than rebuilt from expanded ids. A tile's tfs are expanded into
/// its lanes from the block's tfs in one load, its docs' length buckets are
/// 16 contiguous bytes, and its lanes are scored and added at once: lanes
/// without a doc score `0 / (0 + 1)`, exactly `0.0`, which leaves a
/// non-negative slot as it was. A doc's contribution takes the operations
/// [`bm25::score_one_term_x8`] gives it, so the window holds the same bits.
/// Returns how many docs it added.
///
/// # Safety
///
/// The caller must have detected AVX-512F and AVX2 at runtime. Every access
/// is bounds-checked: a tile is inside the window (it is aligned to a
/// divisor of the window's width and holds an in-window doc), the tf load
/// reads one tf per doc and is checked against `tfs`, the 16-byte bucket
/// load is taken only when it fits the norm table, and gather indices are
/// bytes into a 256-entry table.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx2")]
unsafe fn accumulate_bitset_avx512(
    idf_weight: f32,
    run: BitsetRun<'_>,
    base: u32,
    scores: &mut [f32],
    present: &mut [u64],
    dl_norm_k1: &NormTable,
) -> usize {
    use std::arch::x86_64::*;
    let scores = &mut scores[..OR_WINDOW as usize];
    let present = &mut present[..OR_WINDOW_WORDS];
    let (buckets, lut) = dl_norm_k1.parts();
    debug_assert!(run.docs.start >= base && run.docs.end <= base.saturating_add(OR_WINDOW));
    let first = (run.docs.start - base) as usize & !(ACCUMULATE_TILE - 1);
    let mut j = 0usize;
    for slot in (first..(run.docs.end - base) as usize).step_by(ACCUMULATE_TILE) {
        let tile = base + slot as u32;
        let mask = run.tile_mask(tile);
        if mask == 0 {
            continue;
        }
        let n = mask.count_ones() as usize;
        assert!(j + n <= run.tfs.len(), "a bitset doc without a tf");
        let mut bytes = [0u8; ACCUMULATE_TILE];
        match buckets.get(tile as usize..tile as usize + ACCUMULATE_TILE) {
            Some(tile_buckets) => bytes.copy_from_slice(tile_buckets),
            None => {
                for (lane, b) in bytes.iter_mut().enumerate() {
                    if mask >> lane & 1 == 1 {
                        *b = buckets[tile as usize + lane];
                    }
                }
            }
        }
        // SAFETY: the target features are enabled on this function (see the
        // contract above). The expand-load reads `n` consecutive tfs from
        // `tfs[j]`, inside `tfs` by the assert above. The gather reads only
        // the lanes `mask` selects, at byte indices into the 256-entry
        // table. The tile's 16 slots lie inside the window slice: `slot` is a
        // multiple of the tile width, which divides the window's, and below
        // it. The presence bits land in one word: a tile's slots start at a
        // multiple of 16 within it.
        unsafe {
            let tf = _mm512_cvtepi32_ps(_mm512_maskz_expandloadu_epi32(
                mask,
                run.tfs.as_ptr().add(j).cast(),
            ));
            let idx = _mm512_cvtepu8_epi32(_mm_loadu_si128(bytes.as_ptr().cast()));
            let norm =
                _mm512_mask_i32gather_ps::<4>(_mm512_set1_ps(1.0), mask, idx, lut.as_ptr().cast());
            let add = _mm512_div_ps(
                _mm512_mul_ps(_mm512_set1_ps(idf_weight), tf),
                _mm512_add_ps(tf, norm),
            );
            let at = scores.as_mut_ptr().add(slot);
            _mm512_storeu_ps(at, _mm512_add_ps(_mm512_loadu_ps(at), add));
        }
        present[slot >> 6] |= u64::from(mask) << (slot & 63);
        j += n;
    }
    j
}

/// Bitset words the negation fold of a counted single-term walk may clear
/// per doc of the term: a doc's probe is a search in a negated block, which
/// costs about what clearing a cache line of words does.
const NEGATION_FOLD_WORDS_PER_DOC: u64 = 8;

/// Most terms a union may have for its windows below [`BUFFERED_TOPK_MIN_K`]
/// to end at the nearest essential block boundary rather than the score
/// buffer's width. With two terms that boundary sits a block of one term's
/// postings away and a window has no non-essential term to bound, so ending
/// there is cheap and lets the bar rise per block on a small `k`. With more,
/// each window pays a bound per non-essential term and a drain, and the
/// nearest of several boundaries is close, so narrow windows cost more than
/// their pruning saves.
const BLOCK_WINDOW_MAX_TERMS: usize = 2;

/// Smallest `k` at which a top-k walk collects into a buffer rather than a
/// heap. It marks where a `k` is large in general: the bar rises slowly
/// there, so the choices that pay only for a fast-rising bar — a union window
/// ending at a block boundary ([`BLOCK_WINDOW_MAX_TERMS`]), the must-led
/// walk probing its strongest optional term first — stop below it.
pub(super) const BUFFERED_TOPK_MIN_K: usize = 512;

/// Most optional terms the required-term walk takes; a wider list keeps
/// the general must+should walk.
const MAX_SHOULDS_LED: usize = 16;

/// A rare-led walk's answer when it proved complete, else the floor the
/// general walk may start from: just under the walk's k-th best. Those `k`
/// documents are in the union, so the true k-th best is at least that score;
/// a document tied with it may still win on its id, so the floor sits one
/// step below. With fewer than `k` documents found there is no floor.
fn seed_or_hits(
    hits: Vec<(FtsDocId, f32)>,
    complete: bool,
    k: usize,
    rare: usize,
) -> Result<Vec<(FtsDocId, f32)>, RareLedFallback> {
    if complete {
        return Ok(hits);
    }
    match hits.get(k.wrapping_sub(1)) {
        Some(&(_, kth)) if k > 0 => Err(RareLedFallback::Seed(kth.next_down())),
        // Fewer than `k` found: the heap never filled, so nothing was pruned
        // and these are every rare-term document, fully scored.
        _ if k > 0 => Err(RareLedFallback::Split { rare, hits }),
        _ => Err(RareLedFallback::Walk),
    }
}

/// What a rare-led walk that cannot stand on its own hands the general walk.
pub(super) enum RareLedFallback {
    /// Start from this floor: just under the walk's k-th best.
    Seed(f32),
    /// The rare term (cursor `rare`) held fewer than `k` documents, all in
    /// `hits` with their full scores: the other terms can be ranked alone and
    /// merged in.
    Split {
        rare: usize,
        hits: Vec<(FtsDocId, f32)>,
    },
    /// Nothing to start from.
    Walk,
}

/// Tail of the two-term WAND once `weak` cannot reach the threshold alone:
/// every remaining candidate is a doc of `strong`, scored with `weak`'s tf
/// when the block bound allows and `weak` holds the doc. `weak` is only
/// probed, never stepped, so a stopword's bitset blocks are never expanded.
///
/// The heap need not be full: a candidate above the threshold is pushed
/// while the heap has room and replaces the worst entry once it holds `k`,
/// and the threshold follows the heap only from that point. The caller
/// today enters with a full heap (its threshold is set only then), but
/// the tail does not rely on it — a seeded threshold must not lose docs.
fn wand_two_term_tail(
    weak: &mut TermCursor,
    strong: &mut TermCursor,
    k: usize,
    heap: &mut BinaryHeap<TopKEntry>,
    threshold: &mut f32,
    dl_norm_k1: &NormTable,
) {
    while !strong.is_exhausted() {
        let doc = strong.current_doc_id();
        let norm = dl_norm_k1.get(doc);
        let own = bm25::score_with_dl_norm_k1(strong.idf_weight, strong.current_tf(), norm);
        weak.shallow_advance_block_to(doc);
        if own + weak.inspect_block_max_bm25() > *threshold {
            let mut score = own;
            if let Some(tf) = weak.bitset_probe_tf(doc) {
                score += bm25::score_with_dl_norm_k1(weak.idf_weight, tf, norm);
            }
            if score > *threshold {
                if heap.len() < k {
                    heap.push(TopKEntry(score, FtsDocId::new(doc)));
                } else {
                    replace_worst(heap, TopKEntry(score, FtsDocId::new(doc)));
                }
                if heap.len() == k {
                    *threshold = heap.peek().expect("non-empty").0;
                }
            }
        }
        strong.next();
    }
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
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1();

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
            if cursors.len() == 2 && threshold > 0.0 {
                // Two terms and one of them can no longer reach the
                // threshold on its own: from here the pivot is always the
                // other term's doc, so walk that term and only probe the
                // weak one. The generic loop below would re-sort the pair,
                // re-derive the pivot and step the weak cursor past every
                // scored doc, which for a stopword is one block expansion
                // per probe.
                let weak = usize::from(cursors[0].term_max_bm25 > cursors[1].term_max_bm25);
                if cursors[weak].term_max_bm25 <= threshold {
                    let (a, b) = cursors.split_at_mut(1);
                    let (weak_c, strong_c) = match weak {
                        0 => (&mut a[0], &mut b[0]),
                        _ => (&mut b[0], &mut a[0]),
                    };
                    wand_two_term_tail(weak_c, strong_c, k, &mut heap, &mut threshold, dl_norm_k1);
                    break;
                }
            }
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
                if accum_term_ub > pruning_bar(threshold, cursors.len()) {
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

            if accum_block_ub <= pruning_bar(threshold, cursors.len()) {
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
                    idfs[packed] = cursor.idf_weight;
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
                heap.push(TopKEntry(score, FtsDocId::new(pivot_doc)));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if heap.peek().is_some_and(|worst| score > worst.0) {
                // Replace the evicted entry in place: one sift instead of
                // the pop's sift-down plus the push's sift-up.
                replace_worst(&mut heap, TopKEntry(score, FtsDocId::new(pivot_doc)));
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
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
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
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        // At a large `k` most hits are admitted: collect into a buffer cut by
        // selection rather than sifting a heap per hit.
        if k >= BUFFERED_TOPK_MIN_K {
            let dl_norm_k1 = self.columns[column_id as usize].dl_norm_k1();
            let capacity = top_k_initial_capacity(k, u64::from(self.n_docs), None);
            let mut sink = BufferedScoreSink {
                top: TopKBuffer::with_capacity(k, capacity),
                k,
                filter,
                floor_eff,
                kth: floor_eff.max(0.0),
            };
            self.and_scored_walk(cursors, dl_norm_k1, &mut sink);
            return Ok(sink.top.into_sorted());
        }
        let (hits, ()) = self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
            let mut sink = ScoreSink {
                heap,
                k,
                filter,
                floor_eff,
            };
            self.and_scored_walk(cursors, dl_norm_k1, &mut sink);
        });
        Ok(hits)
    }

    /// [`Self::run_and_intersect`] with pruning off: every intersection
    /// doc is scored and counted. Returns the top-k and the exact count.
    pub(super) fn run_and_intersect_counted(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        if cursors.is_empty() {
            return Ok((Vec::new(), 0));
        }
        Ok(self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
            let mut sink = ScoreCountSink {
                heap,
                k,
                filter,
                count: 0,
            };
            self.and_scored_walk(cursors, dl_norm_k1, &mut sink);
            sink.count
        }))
    }

    /// The heap, norms and drain shared by the ranked AND walks. `walk`
    /// fills the heap and returns whatever else it tallied.
    fn scored_top_k<R>(
        &self,
        column_id: u32,
        k: usize,
        walk: impl FnOnce(&NormTable, &mut BinaryHeap<TopKEntry>) -> R,
    ) -> (Vec<(FtsDocId, f32)>, R) {
        let dl_norm_k1 = self.columns[column_id as usize].dl_norm_k1();
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let extra = walk(dl_norm_k1, &mut heap);
        (drain_top_k_desc(heap), extra)
    }

    /// The ranked AND walk for `cursors`, feeding every intersection doc
    /// to `sink`: the term with the fewest blocks drives a block at a time,
    /// and the others are probed over each block's survivors, fewest blocks
    /// first, which is also the order their scores are added in.
    fn and_scored_walk<S: AndSink>(
        &self,
        mut cursors: Vec<TermCursor>,
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        debug_assert!(sink.needs_score(), "an unranked AND takes the flat-merge");
        cursors.sort_by_key(|c| c.df);
        let mut others = cursors;
        let driver = others.remove(0);
        and_membership_scored_batched(driver, others, dl_norm_k1, sink);
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
        must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        debug_assert!(
            !must_cursors.is_empty() && !should_cursors.is_empty(),
            "dispatch routes empty-side shapes to the AND/OR kernels"
        );
        let dl_norm_k1 = self.columns[column_id as usize].dl_norm_k1();
        let cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        // At a large `k` most scored docs are admitted and a heap's sift per
        // admission dominates; the selection buffer pays an append instead.
        Ok(match k >= BUFFERED_TOPK_MIN_K {
            true => {
                let mut buf = TopKBuffer::with_capacity(k, cap);
                self.must_should_into(
                    must_cursors,
                    should_cursors,
                    k,
                    filter,
                    floor_eff,
                    dl_norm_k1,
                    &mut buf,
                );
                buf.into_sorted()
            }
            false => {
                let mut heap = <BinaryHeap<TopKEntry> as Collector>::with_capacity(k, cap);
                self.must_should_into(
                    must_cursors,
                    should_cursors,
                    k,
                    filter,
                    floor_eff,
                    dl_norm_k1,
                    &mut heap,
                );
                heap.into_sorted()
            }
        })
    }

    /// The ranked must+should walk into `heap`, whichever collector it is.
    #[allow(clippy::too_many_arguments)]
    fn must_should_into<C: Collector>(
        &self,
        mut must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        dl_norm_k1: &NormTable,
        heap: &mut C,
    ) {
        // The per-candidate walk wins where the bar prunes. At a large `k`
        // the bar stays low: with one optional term the walk's single batched
        // probe per block still wins, and with three or more its strongest-
        // first probing, which drops a doc once the rest cannot close the gap,
        // saves more probes than the general walk's one batch probe of every
        // term; with exactly two, the general walk is cheaper (measured).
        let optional = should_cursors.len();
        if must_cursors.len() == 1
            && optional <= MAX_SHOULDS_LED
            && (k <= WAND_BMW_2TERM_MAX_K || optional != 2)
        {
            let lead = must_cursors.pop().expect("one must cursor");
            must_lead_should_walk(lead, should_cursors, k, filter, floor_eff, dl_norm_k1, heap);
            return;
        }
        must_cursors.sort_by_key(|c| c.df);
        let should_ub = should_cursors.iter().map(|c| c.term_max_bm25).sum();
        let mut sink = MustShouldSink {
            heap,
            kth: f32::NEG_INFINITY,
            k,
            filter,
            floor_eff,
            shoulds: ShouldBatch::new(should_cursors),
            should_ub,
            dl_norm_k1,
        };
        self.and_flat_merge(&mut must_cursors, dl_norm_k1, &mut sink);
        sink.flush();
    }

    /// [`Self::run_must_should`] with pruning off: every must-intersection
    /// doc is scored and counted. Returns the top-k and the exact count.
    pub(super) fn run_must_should_counted(
        &self,
        column_id: u32,
        mut must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        debug_assert!(
            !must_cursors.is_empty() && !should_cursors.is_empty(),
            "dispatch routes empty-side shapes to the AND/OR kernels"
        );
        if must_cursors.len() == 1 && filter.is_none() {
            let lead = must_cursors.pop().expect("one must cursor");
            return Ok(self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
                must_lead_should_count(lead, should_cursors, k, dl_norm_k1, heap)
            }));
        }
        Ok(self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
            must_cursors.sort_by_key(|c| c.df);
            let mut sink = MustShouldCountSink {
                heap,
                k,
                filter,
                shoulds: ShouldBatch::new(should_cursors),
                dl_norm_k1,
                count: 0,
            };
            self.and_flat_merge(&mut must_cursors, dl_norm_k1, &mut sink);
            sink.flush();
            sink.count
        }))
    }

    /// Unranked multi-term AND: the matching doc ids in ascending order
    /// via the block flat-merge in [`and_flat_merge`](Self::and_flat_merge),
    /// with no BM25 scoring and no top-k heap. Because it shares that
    /// traversal with the ranked [`run_and_intersect`](Self::run_and_intersect),
    /// the two always agree on which docs match, and an unranked count
    /// over high-frequency terms costs the same posting-list work as the
    /// ranked search minus the scoring.
    pub(super) fn collect_and_intersect(&self, mut cursors: Vec<TermCursor>) -> Vec<FtsDocId> {
        if cursors.is_empty() {
            return Vec::new();
        }
        cursors.sort_by_key(|c| c.df);
        let mut sink = CollectSink { out: Vec::new() };
        // A collect never scores, so it needs none of the column's norms.
        self.and_flat_merge(&mut cursors, unscored_norm_table(), &mut sink);
        sink.out
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
            let bar = pruning_bar(sink.bar(), cursors.len() + sink.added_terms());
            if bar > f32::NEG_INFINITY {
                let leader_doc = cursors[0].current_doc_id();
                let leader_block_max = cursors[0].current_block_max_bm25();
                let leader_block_end = cursors[0].current_block_last_doc_id();
                let (others_ub, window_end) =
                    block_max_and_bound(leader_block_end, &mut cursors[1..], leader_doc);
                if leader_block_max + others_ub <= bar {
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
            let (ld, lt) = (&c0.block_doc_ids[..lb_n], &c0.block_tfs[..lb_n]);
            while i < lb_n {
                let a = ld[i];

                // For each non-leader, walk its `pos` forward through
                // the decoded block until block_doc_ids[pos] >= a (or
                // the block exhausts). If any block exhausts, break
                // out to the outer loop's block-crossing step. If any
                // cursor lands above `a`, the leader doc isn't in the
                // intersection — advance leader only.
                let mut block_exhausted = false;
                let mut all_match = true;
                for o in others.iter_mut() {
                    // Scan on a local index over the decoded prefix: the
                    // field form reloaded the buffer pointer and stored the
                    // position on every step, ~5 ns a doc across every
                    // posting of every non-leader.
                    let od = &o.block_doc_ids[..o.block_n];
                    let mut p = o.pos;
                    while p < od.len() && od[p] < a {
                        p += 1;
                    }
                    o.pos = p;
                    if p >= od.len() {
                        block_exhausted = true;
                        break;
                    }
                    if od[p] != a {
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
                        let mut score = bm25::score_with_dl_norm_k1(c0.idf_weight, lt[i], norm);
                        for o in others.iter() {
                            score +=
                                bm25::score_with_dl_norm_k1(o.idf_weight, o.block_tfs[o.pos], norm);
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
            let bar = pruning_bar(sink.bar(), 2 + sink.added_terms());
            if bar > f32::NEG_INFINITY {
                let leader_doc = c0.current_doc_id();
                let leader_block_max = c0.current_block_max_bm25();
                let (others_ub, window_end) =
                    block_max_and_bound(c0.current_block_last_doc_id(), from_mut(c1), leader_doc);
                if leader_block_max + others_ub <= bar {
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
            let c0_idf = c0.idf_weight;
            let c1_idf = c1.idf_weight;
            // Subslices sized to the decoded prefix: the loop's index
            // checks fold into `i < lb_n` / `j < rb_n` instead of a
            // compare against each Vec's length per load.
            let (ld, lt) = (&c0.block_doc_ids[..lb_n], &c0.block_tfs[..lb_n]);
            let (rd, rt) = (&c1.block_doc_ids[..rb_n], &c1.block_tfs[..rb_n]);
            while i < lb_n && j < rb_n {
                let a = ld[i];
                let b = rd[j];
                if a < b {
                    i += 1;
                } else if a > b {
                    j += 1;
                } else {
                    let score = if sink.needs_score() {
                        let norm = dl_norm_k1.get(a);
                        bm25::score_with_dl_norm_k1(c0_idf, lt[i], norm)
                            + bm25::score_with_dl_norm_k1(c1_idf, rt[j], norm)
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
    /// `[doc_id_start, doc_id_end)`. [`Self::run_max_score_bmm`] runs it over
    /// the whole range, which reproduces the un-ranged walk exactly (the seek
    /// is a no-op and the upper-bound check never fires); the ranged form is
    /// the oracle the ranged windowed kernel is checked against.
    pub(super) fn run_max_score_bmm_range(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1();

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
        // Bounds are tested against the threshold lowered for summing order
        // (see [`pruning_bar`]); offers compare real scores with it.
        let n_terms = cursors.len();
        let pb = |t: f32| pruning_bar(t, n_terms);

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
        let mut f_essential: usize = recompute_f(&partial_max, pb(threshold));

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
                if block_ub <= pb(threshold) {
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
                        cursors[0].idf_weight,
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
                    if essential_score + others_ub <= pb(threshold) {
                        cursors[0].next();
                        continue;
                    }
                    // SIMD-pack non-essentials at `candidate`.
                    let mut idfs = [cursors[0].idf_weight, 0.0, 0.0, 0.0];
                    let mut tfs = [cursors[0].current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(1) {
                        if let Some(tf) = cursor.bitset_probe_tf(candidate) {
                            idfs[packed] = cursor.idf_weight;
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

                    if offer_top_k(&mut heap, k, &mut threshold, score, candidate) {
                        let new_f = recompute_f(&partial_max, pb(threshold));
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
            let leftmost_block_ub = cursors[leftmost_essential].current_block_max_bm25();
            let last_in_block = cursors[leftmost_essential].current_block_last_doc_id();
            // Bound every OTHER cursor's contribution over the leftmost block's
            // doc range `[candidate, last_in_block]` by its per-block max across
            // that range (`block_max_in_range`, amortized O(1) via the monotonic
            // inspect-block hint), not by its global term-max. This is the
            // Block-Max-WAND pivot bound: far tighter than the term-level sum, so
            // the block skip fires on many more blocks. `block_max_in_range` is a
            // safe upper bound over the whole range (single-block lookups
            // underestimate when the range spans several cursor blocks), so the
            // skip stays correct.
            let mut others_ub = 0.0f32;
            for (i, c) in cursors.iter_mut().enumerate() {
                if i == leftmost_essential {
                    continue;
                }
                others_ub += c.block_max_in_range(candidate, last_in_block);
            }
            if leftmost_block_ub + others_ub <= pb(threshold) {
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
                        idfs[packed] = cursor.idf_weight;
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
                if score + non_essentials_term_ub > pb(threshold) {
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

                    if score + remaining_block_ub > pb(threshold) {
                        for cursor in cursors.iter_mut().skip(f_essential) {
                            let block_ub = cursor.inspect_block_max_bm25();
                            if score + remaining_block_ub <= pb(threshold) {
                                break;
                            }
                            // Locate + score the non-essential without expanding
                            // a dense block's doc ids: a bit-test + popcount-rank
                            // tf read on a bitset block, decode-and-locate on a
                            // PACKED one.
                            if let Some(tf) = cursor.bitset_probe_tf(candidate) {
                                score += bm25::score_with_dl_norm_k1(cursor.idf_weight, tf, norm);
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
                if offer_top_k(&mut heap, k, &mut threshold, score, candidate) {
                    f_essential = recompute_f(&partial_max, pb(threshold));
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
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        // A top-0 request admits nothing. Guard here too (callers already
        // short-circuit) so the heap-admission `else if` below can never
        // run against an empty heap.
        if k == 0 {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1();

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
                    accumulate_segment(c, base, window_end, &mut scores, &mut present, dl_norm_k1);
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
                    offer_top_k(&mut heap, k, &mut threshold, score, doc);
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Every match of a union scored and counted in one pass, for the
    /// exact-count search: the top `k` and the number of matching documents
    /// left after `filter`. Nothing is pruned: every match is scored, and
    /// only the comparison with the k-th best decides what the heap sees.
    ///
    /// A window of [`OR_WINDOW`] documents at a time: each term adds its
    /// scores into the window eight postings per SIMD call and marks
    /// presence, then the window is drained in ascending document order,
    /// counted a word at a time and offered to the heap only where a score
    /// beats the k-th best. A single term needs no window: each block is
    /// scored in place and its docs offered in order. Same admission rule and
    /// tie-break as [`Self::run_exhaustive_union`] (`score > threshold`,
    /// ascending docs), so the hits differ from it at most in summation order.
    pub(super) fn run_windowed_union_counted(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        let dl_norm_k1 = self.columns[column_id as usize].dl_norm_k1();
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;
        let mut count = 0u64;
        let offer =
            |heap: &mut BinaryHeap<TopKEntry>, threshold: &mut f32, doc: u32, score: f32| {
                if k == 0 {
                    return;
                }
                offer_top_k(heap, k, threshold, score, doc);
            };
        cursors.retain(|c| !c.is_exhausted());
        if let [c] = cursors.as_mut_slice() {
            // Every doc of the term is asked about, so the negated terms are
            // folded once into a bitset over its span and each doc is one bit
            // test, rather than a search in a negated block per doc. The fold
            // clears the span's words and reads every negated block, so it is
            // taken only where the term's own probes would cost more: it has
            // at least one doc per [`NEGATION_FOLD_WORDS_PER_DOC`] words, and
            // at least as many docs as there are negated blocks. A row-set
            // filter keeps its own test.
            let last = c.blocks.last().map_or(0, |b| b.last_doc_id);
            let words = last as usize / 64 + 1;
            let excluded: Option<Vec<u64>> = match filter.as_deref() {
                Some(f)
                    if f.allow.is_none()
                        && !f.cursors.is_empty()
                        && c.df.saturating_mul(NEGATION_FOLD_WORDS_PER_DOC) >= words as u64
                        && c.df >= f.cursors.iter().map(|nc| nc.blocks.len() as u64).sum() =>
                {
                    let mut ex = vec![0u64; words];
                    or_cursors_into_bitset(&mut ex, &f.cursors);
                    Some(ex)
                }
                _ => None,
            };
            let mut lane_scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
            while !c.is_exhausted() {
                c.hold_whole_block();
                let (from, to) = (c.pos, c.block_n);
                score_postings(
                    c.idf_weight,
                    &c.block_doc_ids[from..to],
                    &c.block_tfs[from..to],
                    dl_norm_k1,
                    &mut lane_scores,
                );
                let docs = c.block_doc_ids[from..to].iter();
                match filter.as_deref_mut() {
                    // Every doc counts. Once the heap is full, a block none of
                    // whose scores beats the k-th best is passed over whole,
                    // one lane-wise compare instead of an offer per doc.
                    None => {
                        count += (to - from) as u64;
                        if heap.len() < k || any_beats(&lane_scores, threshold) {
                            for (&doc, &score) in docs.zip(&lane_scores) {
                                offer(&mut heap, &mut threshold, doc, score);
                            }
                        }
                    }
                    Some(f) => {
                        for (&doc, &score) in docs.zip(&lane_scores) {
                            let admitted = match &excluded {
                                Some(ex) => (ex[(doc >> 6) as usize] >> (doc & 63)) & 1 == 0,
                                None => f.admits(doc),
                            };
                            if !admitted {
                                continue;
                            }
                            count += 1;
                            offer(&mut heap, &mut threshold, doc, score);
                        }
                    }
                }
                c.advance_by(to - from);
            }
            return Ok((drain_top_k_desc(heap), count));
        }
        let mut scores = vec![0.0f32; OR_WINDOW as usize];
        let mut present = [0u64; OR_WINDOW_WORDS];
        loop {
            let mut min_doc = u32::MAX;
            for c in &cursors {
                if !c.is_exhausted() {
                    min_doc = min_doc.min(c.current_doc_id());
                }
            }
            if min_doc == u32::MAX {
                break;
            }
            let base = min_doc & !(OR_WINDOW - 1);
            let window_end = base.saturating_add(OR_WINDOW);
            for c in &mut cursors {
                while !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d >= window_end {
                        break;
                    }
                    accumulate_segment(c, base, window_end, &mut scores, &mut present, dl_norm_k1);
                }
            }
            for (word_idx, word) in present.iter_mut().enumerate() {
                let mut bits = mem::take(word);
                if bits == 0 {
                    continue;
                }
                if filter.is_none() {
                    count += u64::from(bits.count_ones());
                    // A word whose best score cannot beat the k-th best holds
                    // no hit: count it (done above) and clear its slots
                    // without visiting a document.
                    let slots = &mut scores[word_idx << 6..(word_idx + 1) << 6];
                    if heap.len() >= k && !word_beats(slots, threshold) {
                        slots.fill(0.0);
                        continue;
                    }
                }
                while bits != 0 {
                    let local = (word_idx << 6) | bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let score = mem::take(&mut scores[local]);
                    let doc = base + local as u32;
                    if let Some(f) = filter.as_deref_mut() {
                        if !f.admits(doc) {
                            continue;
                        }
                        count += 1;
                    }
                    offer(&mut heap, &mut threshold, doc, score);
                }
            }
        }
        Ok((drain_top_k_desc(heap), count))
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
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        // At a large `k` with a low bar most survivors are admitted, and a
        // heap's sift per admission is a fifth of the kernel; a buffer cut by
        // selection costs an append instead. At a small `k` the bar must rise
        // on every admission, which only the heap does.
        match k >= BUFFERED_TOPK_MIN_K {
            true => self.run_windowed_maxscore_with::<TopKBuffer>(
                column_id,
                cursors,
                k,
                filter,
                floor_eff,
                doc_id_start,
                doc_id_end,
            ),
            false => self.run_windowed_maxscore_with::<BinaryHeap<TopKEntry>>(
                column_id,
                cursors,
                k,
                filter,
                floor_eff,
                doc_id_start,
                doc_id_end,
            ),
        }
    }

    fn run_windowed_maxscore_with<C: Collector>(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1();

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
        // Fixed by the query's shape, not by how many terms are live.
        let block_windows = n <= BLOCK_WINDOW_MAX_TERMS && k < BUFFERED_TOPK_MIN_K;
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
        let mut heap = C::with_capacity(k, initial_cap);
        let mut threshold: f32 = floor_eff.max(0.0);
        // A floor bars every doc at or under it, as a full collector's k-th
        // does, so pruning holds from the first window rather than once `k`
        // docs are in.
        let floored = floor_eff > f32::NEG_INFINITY;
        // Bounds are tested against the threshold lowered for summing order
        // (see [`pruning_bar`]); offers compare real scores with it.
        let pb = |t: f32| pruning_bar(t, n);
        let mut scores = vec![0.0f32; OR_WINDOW as usize];
        let mut present = [0u64; OR_WINDOW_WORDS];
        // Leader-block positions that survive the batch bar, reused per block.
        // A fixed array and a count held in a register: pushing to a `Vec`
        // and dropping the entry again keeps its length in memory, one
        // store-to-load chain per posting.
        // Eight slots of slack: a batch of eight writes all its slots and
        // keeps only the survivors.
        let mut survivors = [0u16; BLOCK_LEN + 8];
        let mut n_survivors: usize;
        let mut leader_scores = [0.0f32; BLOCK_LEN];
        // Compacted per-window candidate list (doc + essential score), reused;
        // the drain fills it, the SIMD filter compacts it to survivors.
        let mut win_docs: Vec<u32> = Vec::with_capacity(OR_WINDOW as usize);
        // The non-essentials' tfs at the window's survivors, reused.
        let mut noness_tfs: Vec<u32> = Vec::with_capacity(OR_WINDOW as usize);
        let mut win_scores: Vec<f32> = Vec::with_capacity(OR_WINDOW as usize);
        // Non-essential block-max bound for the single-essential path, and its
        // suffix sums, both rebuilt only when a candidate leaves the doc range
        // the bound was measured over.
        let mut noness_ub = vec![0.0_f32; n];
        let mut noness_suffix = vec![0.0_f32; n + 1];
        // Per-window bounds and doc estimates and their suffix sums, in the
        // same term-max order as `partial_max`.
        let mut win_ub = vec![0.0_f32; n];
        let mut win_docs_est = vec![0.0_f32; n];
        let mut partial_win = vec![0.0_f32; n + 1];
        // Sum of every term's UB — the reference for the essential-side
        // block-max skip below.
        let total_term_ub = partial_max[0];

        loop {
            // Continuous partition: recompute the essential set from the live
            // threshold. `threshold` only rises, so `f_essential` only shrinks.
            let f_essential = recompute_f(&partial_max, pb(threshold));

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
                let mut block_others_ub = f32::INFINITY;
                if floored || heap.is_full(k) {
                    // The other terms bounded over this block's own range,
                    // not their whole lists: a mid-frequency term's best
                    // document elsewhere must not hold this block open.
                    let start = cursors[0].current_doc_id();
                    let end = cursors[0].current_block_last_doc_id();
                    let (lead, others) = cursors.split_at_mut(1);
                    let others_ub: f32 = others
                        .iter_mut()
                        .map(|c| c.block_max_in_range(start, end))
                        .sum();
                    if lead[0].current_block_max_bm25() + others_ub <= pb(threshold) {
                        lead[0].skip_to(end.saturating_add(1));
                        continue;
                    }
                    block_others_ub = others_ub;
                }
                // Score the rest of the leader block at once, eight postings per
                // SIMD call, and keep the positions that can clear the bar with
                // the other terms at their bound over the whole block. That
                // bound is never tighter than the walk's per-candidate one and
                // the test is the walk's own expression, so every position
                // dropped here the walk would drop too; the walk then jumps from
                // survivor to survivor instead of scoring each posting behind a
                // branch.
                n_survivors = 0;
                let batched = block_others_ub.is_finite()
                    && filter.is_none()
                    && cursors[0].holds_whole_block();
                if batched {
                    let c = &cursors[0];
                    let (from, to) = (c.pos, c.block_n);
                    let mut i = from;
                    while i + 8 <= to {
                        let dids: [u32; 8] = c.block_doc_ids[i..i + 8]
                            .try_into()
                            .expect("invariant: an 8-wide slice");
                        let tfs: [u32; 8] = c.block_tfs[i..i + 8]
                            .try_into()
                            .expect("invariant: an 8-wide slice");
                        let lanes =
                            bm25::score_one_term_x8(c.idf_weight, tfs, dl_norm_k1.get8(dids));
                        leader_scores[i..i + 8].copy_from_slice(&lanes);
                        // One compare for the eight lanes, then the survivors'
                        // positions written left-packed from the mask: no
                        // lane's slot waits on the count of the lane before.
                        let keep = (f32x8::from(lanes) + f32x8::splat(block_others_ub))
                            .simd_gt(f32x8::splat(pb(threshold)))
                            .to_bitmask() as usize;
                        let pack = &LEFT_PACK[keep];
                        for (slot, &lane) in
                            survivors[n_survivors..n_survivors + 8].iter_mut().zip(pack)
                        {
                            *slot = (i + usize::from(lane)) as u16;
                        }
                        n_survivors += keep.count_ones() as usize;
                        i += 8;
                    }
                    let tail = c.block_doc_ids[i..to].iter().zip(&c.block_tfs[i..to]);
                    for (j, (slot, (&d, &tf))) in
                        (i..).zip(leader_scores[i..to].iter_mut().zip(tail))
                    {
                        let score =
                            bm25::score_with_dl_norm_k1(c.idf_weight, tf, dl_norm_k1.get(d));
                        *slot = score;
                        survivors[n_survivors] = j as u16;
                        n_survivors += usize::from(score + block_others_ub > pb(threshold));
                    }
                    // Complete the survivors the way a window drain does:
                    // strongest non-essential first, the survivors re-filtered
                    // against the bound of the clauses still unprobed, each
                    // term probed for the whole list in one sorted pass.
                    let start = cursors[0].current_doc_id();
                    let end = cursors[0].current_block_last_doc_id();
                    let (ess, non_ess) = cursors.split_at_mut(1);
                    let c0 = &mut ess[0];
                    win_docs.clear();
                    win_scores.clear();
                    for &at in &survivors[..n_survivors] {
                        let d = c0.block_doc_ids[usize::from(at)];
                        if d >= doc_id_end {
                            break;
                        }
                        win_docs.push(d);
                        win_scores.push(leader_scores[usize::from(at)]);
                    }
                    let m = non_ess.len();
                    noness_suffix[m] = 0.0;
                    for i in (0..m).rev() {
                        noness_suffix[i] =
                            noness_suffix[i + 1] + non_ess[i].block_max_in_range(start, end);
                    }
                    complete_survivors(
                        non_ess,
                        &noness_suffix,
                        pb(threshold),
                        &mut win_docs,
                        &mut win_scores,
                        dl_norm_k1,
                        &mut noness_tfs,
                    );
                    for (&doc, &score) in win_docs.iter().zip(&win_scores) {
                        if heap.offer(k, &mut threshold, score, doc)
                            && total_term_ub <= pb(threshold)
                        {
                            // Not even every term at its maximum can beat the
                            // bar now: no later doc can either.
                            break;
                        }
                    }
                    let rest = c0.block_n - c0.pos;
                    c0.advance_by(rest);
                    continue;
                }
                let block_end = cursors[0].current_block_last_doc_id();
                let (ess, non_ess) = cursors.split_at_mut(1);
                let c0 = &mut ess[0];
                // The non-essentials' bound changes only when a candidate
                // crosses one of their block boundaries, which for a rare
                // essential term is far less often than once a candidate.
                // Track the doc up to which the last measurement still holds
                // and reuse it until then, instead of walking every
                // non-essential per candidate.
                let mut others_ub = 0.0f32;
                let mut bound_holds_to = 0u32;
                let mut have_bound = false;
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
                        bm25::score_with_dl_norm_k1(c0.idf_weight, c0.current_tf(), norm);
                    // Bound the non-essentials at `candidate` by each one's
                    // block-max for the block that *contains* it (monotonic
                    // `shallow_advance` hint — amortized O(1), no decode), not by
                    // its global term-max. Far tighter for a common term, so the
                    // skip below fires on many more docs, dropping the completion
                    // probe + heap work — the dominant per-doc cost on a dense
                    // leader query.
                    if !have_bound || candidate > bound_holds_to {
                        others_ub = 0.0;
                        bound_holds_to = u32::MAX;
                        for (i, c) in non_ess.iter_mut().enumerate() {
                            c.shallow_advance_block_to(candidate);
                            let ub = c.inspect_block_max_bm25();
                            noness_ub[i] = ub;
                            others_ub += ub;
                            bound_holds_to = bound_holds_to.min(c.inspect_block_last_doc_id());
                        }
                        // Suffix sums over the same measurement: `noness_suffix[i]`
                        // bounds the clauses from `i` on, so a running score can be
                        // tested against what is still unprobed.
                        let m = non_ess.len();
                        noness_suffix[m] = 0.0;
                        for i in (0..m).rev() {
                            noness_suffix[i] = noness_suffix[i + 1] + noness_ub[i];
                        }
                        have_bound = true;
                    }
                    if essential_score + others_ub <= pb(threshold) {
                        c0.next();
                        continue;
                    }
                    // Complete: probe each non-essential and SIMD-pack the
                    // matches (leader seeded as lane 0, so `score` is the full
                    // BM25 sum).
                    // Probe the non-essentials in term-max order, strongest
                    // first, folding each contribution in as it is read and
                    // re-testing against `noness_suffix`, the bound on the
                    // clauses still unprobed. Resolving the largest uncertainty
                    // first collapses that bound fastest, so a doc that cannot
                    // reach the bar stops costing probes part-way rather than
                    // paying for every clause. Probing dominates this loop and
                    // scoring does not, which is why the running total is kept
                    // scalar rather than packed four at a time.
                    let mut score = essential_score;
                    let mut abandoned = false;
                    for (i, c) in non_ess.iter_mut().enumerate() {
                        if let Some(tf) = c.bitset_probe_tf(candidate) {
                            score += bm25::score_with_dl_norm_k1(c.idf_weight, tf, norm);
                        }
                        if score + noness_suffix[i + 1] <= pb(threshold) {
                            abandoned = true;
                            break;
                        }
                    }
                    if abandoned {
                        c0.next();
                        continue;
                    }
                    let raised = heap.offer(k, &mut threshold, score, candidate);
                    c0.next();
                    // Only when the threshold actually rose: if even this term
                    // plus every other term at its max can't beat it, no further
                    // doc can qualify at all. (Re-checking the partition per
                    // candidate — an O(n) scan — was the f==1 path's dominant
                    // cost on dense leader queries.)
                    if raised && total_term_ub <= pb(threshold) {
                        break;
                    }
                }
                continue;
            }

            // Candidates come from the essential terms only; the first two
            // essential docs place the window and tell whether one term holds
            // it alone. Where windows end at block boundaries, the same scan
            // finds the nearest essential one from block metadata, no decode.
            let mut min_doc = u32::MAX;
            let mut second_doc = u32::MAX;
            let mut block_bound = doc_id_end;
            for c in cursors.iter().take(f_essential) {
                if !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d < min_doc {
                        second_doc = min_doc;
                        min_doc = d;
                    } else if d < second_doc {
                        second_doc = d;
                    }
                    if block_windows {
                        block_bound =
                            block_bound.min(c.current_block_last_doc_id().saturating_add(1));
                    }
                }
            }
            if min_doc == u32::MAX || min_doc >= doc_id_end {
                break;
            }
            // 64-align the base for the presence bitmask and run the window to
            // the score buffer's width, or, for a small `k` over at most
            // [`BLOCK_WINDOW_MAX_TERMS`] terms, to the nearest essential
            // block boundary so the bar updates per block. The
            // bounds stay sound over the wider range, as each term's is the
            // largest of its block maxima across it.
            //
            // When every other essential term's next doc lies past the window,
            // the window holds one essential's docs alone: the others
            // contribute nothing here, so the leader's blocks are bounded by
            // the non-essentials only and skipped by their metadata. This is
            // the shape of a stopword unioned with a rare term at large k: the
            // rare term's postings are far apart and the stopword's blocks
            // between them mostly sit under the bar.
            let base = min_doc & !63;
            let cap_end = base.saturating_add(OR_WINDOW).min(doc_id_end);
            let sole = second_doc >= cap_end;
            let window_end = match sole || !block_windows {
                true => cap_end,
                false => block_bound.min(cap_end),
            };

            // Accumulate the essential terms' contributions into the window
            // (SIMD OR-sum; scalar tail). Identical to the windowed-union body,
            // restricted to essentials.
            // Block-max skip fires only once the bar is real (the collector's
            // k-th, or a floor); hoisted so the per-posting loop doesn't re-test.
            let prune = floored || heap.is_full(k);
            // Block-level partition for this window. A term whose block maxima
            // over the window cannot lift the weaker terms past the threshold
            // is non-essential *here*, even if its term max keeps it essential
            // overall: a stopword's dense blocks are then probed for the other
            // terms' candidates instead of scored doc by doc. Suffix sums over
            // the fixed term-max order keep the bound valid (block maxima never
            // exceed term maxima, so `f_win <= f_essential`). Only the weakest
            // essential term needs testing first: unless its window bound plus
            // the non-essentials' is under the threshold, no term can be
            // demoted. Demotion trades accumulating a term's docs (SIMD, a few
            // ns each) for a probe per remaining candidate (tens of ns), so a
            // term is demoted only when its estimated docs in the window
            // outnumber the remaining essentials' by `DEMOTE_DENSITY`. A window
            // whose total bound is under the threshold holds no competitive doc
            // and is skipped whole.
            //
            // Each term's bound and doc estimate over the window are read once,
            // before any cursor moves; the non-essentials' sum, the essential
            // block skip below and the demotion test all use them.
            let windowed = prune && !sole;
            let mut noness_win = partial_max[f_essential];
            let mut win_all_sum = 0.0_f32;
            if prune {
                let win_last = window_end.saturating_sub(1);
                for (i, c) in cursors.iter_mut().enumerate() {
                    let (ub, docs) = c.block_max_and_density_in_range(base, win_last);
                    win_ub[i] = ub;
                    win_docs_est[i] = docs;
                }
                noness_win = win_ub[f_essential..]
                    .iter()
                    .sum::<f32>()
                    .min(partial_max[f_essential]);
                if windowed {
                    win_all_sum = win_ub.iter().sum();
                }
            }
            let mut f_win = if sole {
                partial_win.copy_from_slice(&partial_max);
                f_essential
            } else if prune {
                let weakest = f_essential - 1;
                if win_ub[weakest] + noness_win > pb(threshold) {
                    partial_win.copy_from_slice(&partial_max);
                    f_essential
                } else {
                    partial_win[n] = 0.0;
                    for i in (0..n).rev() {
                        partial_win[i] = partial_win[i + 1] + win_ub[i];
                    }
                    let f = recompute_f(&partial_win, pb(threshold));
                    let kept: f32 = win_docs_est[..f].iter().sum();
                    let demoted: f32 = win_docs_est[f..f_essential].iter().sum();
                    if f == 0 || demoted >= DEMOTE_DENSITY * kept.max(1.0) {
                        f
                    } else {
                        partial_win.copy_from_slice(&partial_max);
                        f_essential
                    }
                }
            } else {
                f_essential
            };
            if f_win == 0 {
                for c in cursors.iter_mut().take(f_essential) {
                    c.skip_to(window_end);
                }
                continue;
            }
            // Promotion, the converse: a non-essential term with at most
            // [`PROMOTE_DENSITY`] times the essentials' docs in this window is
            // accumulated with them rather than probed at their candidates,
            // accumulating a posting costing a fraction of a probe. Terms stay
            // in term-max order, so a doc's score is summed in the same order
            // either way, and the suffix bounds of the terms still probed are
            // unchanged.
            if windowed && f_win == f_essential {
                let essential_docs: f32 = win_docs_est[..f_essential].iter().sum();
                while f_win < n
                    && !cursors[f_win].predecoded
                    && win_docs_est[f_win] <= PROMOTE_DENSITY * essential_docs
                {
                    // A probe may have left the cursor's buffers on another
                    // block: put it back into iteration at the window's first
                    // essential doc. Not at `base`, which is aligned down and
                    // may reach into the window before: its docs there were
                    // drained already, and a doc below `min_doc` holds no
                    // essential term, so it cannot beat the bar.
                    cursors[f_win].reseek(min_doc);
                    #[cfg(test)]
                    PROMOTIONS.with(|p| p.set(p.get() + 1));
                    f_win += 1;
                }
            }
            for (ci, c) in cursors.iter_mut().enumerate().take(f_win) {
                let mut checked_block = usize::MAX;
                while !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d >= window_end {
                        break;
                    }
                    // Block-max skip, checked once per block (not per posting):
                    // skip a whole block whose docs cannot beat the threshold
                    // with every other term at its bound below, so the dense
                    // OR-sum still prunes at small k. Never fires before the
                    // bar is real, so a doc that must be admitted is never
                    // dropped. A block that reaches past a sole-essential
                    // window is scored up to the window's end, not skipped.
                    if prune && c.current_block != checked_block {
                        checked_block = c.current_block;
                        // The tightest bound valid for the whole block: the
                        // non-essentials alone in a sole-essential window, the
                        // others' window maxima when the block ends inside the
                        // window, else every other term at its term max.
                        let global_ub = total_term_ub - c.term_max_bm25;
                        let inside = c.current_block_last_doc_id() < window_end;
                        let others_ub = match (sole, windowed && inside) {
                            (true, _) => noness_win,
                            (false, true) => (win_all_sum - win_ub[ci]).min(global_ub),
                            (false, false) => global_ub,
                        };
                        #[cfg(test)]
                        BOUND_CLAIMS.with(|v| {
                            // Past the window a sole-essential or windowed
                            // bound claims nothing: it counts no other term's
                            // postings out there.
                            let block_end = c.current_block_last_doc_id().saturating_add(1);
                            v.borrow_mut().push(BoundClaim {
                                from: c.current_doc_id(),
                                to: match sole || windowed {
                                    true => block_end.min(window_end),
                                    false => block_end,
                                },
                                bound: c.current_block_max_bm25() + others_ub,
                            })
                        });
                        if c.current_block_max_bm25() + others_ub <= pb(threshold)
                            && (!sole || inside)
                        {
                            c.skip_blocks_under(others_ub, pb(threshold), window_end);
                            continue;
                        }
                    }
                    accumulate_segment(c, base, window_end, &mut scores, &mut present, dl_norm_k1);
                }
            }

            // Drain the presence bitmask into a compacted, ascending
            // `(doc, essential-score)` list (dropping negated docs), then run
            // the reference pipeline: SIMD-filter to competitive survivors,
            // complete the non-essentials over the survivors only, collect.
            let (_, non_ess) = cursors.split_at_mut(f_win);
            let words = ((window_end - base) as usize)
                .div_ceil(64)
                .min(OR_WINDOW_WORDS);
            if non_ess.is_empty() {
                // Every term was accumulated, so a window score is final:
                // admit straight off the presence bitmask, in doc order, and
                // never build the candidate list. Most docs of a dense window
                // fail the bar, and this way each costs one compare.
                for (word_idx, word) in present[..words].iter_mut().enumerate() {
                    let mut bits = *word;
                    *word = 0;
                    if bits == 0 {
                        continue;
                    }
                    // A word whose best score cannot beat the k-th best holds no
                    // hit: clear its slots without visiting a document.
                    if floored || heap.is_full(k) {
                        let slots = &mut scores[word_idx << 6..(word_idx + 1) << 6];
                        if !word_beats(slots, threshold) {
                            slots.fill(0.0);
                            continue;
                        }
                    }
                    while bits != 0 {
                        let b = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let local = (word_idx << 6) | b;
                        let doc = base + local as u32;
                        let score = mem::take(&mut scores[local]);
                        if let Some(f) = filter.as_deref_mut()
                            && !f.admits(doc)
                        {
                            continue;
                        }
                        heap.offer(k, &mut threshold, score, doc);
                    }
                }
                continue;
            }
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
                    let score = mem::take(&mut scores[local]);
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(doc)
                    {
                        continue;
                    }
                    win_docs.push(doc);
                    win_scores.push(score);
                }
            }

            // Complete the non-essentials over the compacted survivors. Once a
            // bar is set (the collector full, or a floor), do it the reference
            // way: strongest non-essential first, re-filtering the survivor
            // list (progressively tighter budget) before each and
            // batch-scoring matches — so weaker terms touch a shrinking
            // survivor set. Before then every doc must be admitted, so score
            // all non-essentials over every candidate.
            if prune {
                complete_survivors(
                    non_ess,
                    &partial_win[f_win..],
                    pb(threshold),
                    &mut win_docs,
                    &mut win_scores,
                    dl_norm_k1,
                    &mut noness_tfs,
                );
            } else {
                for c in non_ess.iter_mut() {
                    score_noness_batched(
                        c,
                        &win_docs,
                        &mut win_scores,
                        dl_norm_k1,
                        &mut noness_tfs,
                    );
                }
            }

            // Terms demoted for this window were only probed; move them past
            // the window so every essential cursor stays on the window
            // frontier and no doc is accumulated twice.
            for c in cursors.iter_mut().take(f_essential).skip(f_win) {
                c.skip_to(window_end);
            }

            // Collect the fully-scored candidates (ascending doc order).
            for (idx, &doc) in win_docs.iter().enumerate() {
                let score = win_scores[idx];
                heap.offer(k, &mut threshold, score, doc);
            }
        }

        Ok(heap.into_sorted())
    }

    /// Exhaustive union walk for multi-term OR: a k-way merge over the
    /// cursors that scores every doc in the union and offers it to the top-k
    /// heap, with no threshold-driven skipping. Not on any query path; the
    /// bench selector (`OrAlgo::Exhaustive`) keeps it as the unpruned
    /// baseline the pruned kernels are measured against.
    ///
    /// Result invariants match [`Self::run_max_score_bmm`]: top-k by
    /// descending BM25 score, ties broken by ascending doc_id.
    pub(super) fn run_exhaustive_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = col_meta.dl_norm_k1();

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        loop {
            // The smallest current doc_id across the live cursors is the
            // next candidate; exhausted cursors report `u32::MAX`, so the
            // loop ends once every cursor is drained.
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
            // Score: sum BM25 from every cursor at the candidate, four
            // cursors per SIMD call, advancing each.
            let norm = dl_norm_k1.get(candidate);
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() != candidate {
                    continue;
                }
                idfs[packed] = cursor.idf_weight;
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
            if packed > 0 {
                score += bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Top-K update. `threshold` mirrors `heap.peek().0` so the
            // replace-or-skip branch doesn't re-peek per iteration.
            offer_top_k(&mut heap, k, &mut threshold, score, candidate);
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR dispatch. With no negation and no floor, a union with a
    /// rare term — much rarer than the others at a small `k`, or under `k`
    /// documents at any `k` — first tries the rare-led walks
    /// ([`Self::rare_led_pair`] for two terms, [`Self::rare_led`] for more):
    /// they rank the rare term's documents directly and either finish, hand
    /// back a seed for the bar, or hand the rest to [`Self::rare_split`]. A
    /// 2-term pair the rare-led walk declines goes to WAND+BMW
    /// ([`Self::run_wand_bmw`]) where its df ratio and `k` qualify it.
    /// Everything else runs the windowed MaxScore kernel
    /// ([`Self::run_windowed_maxscore`]), its bar started at the
    /// [`union_seed`] when nothing filters or floors it.
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
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
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
        // A term with fewer than `k` documents is split off at any `k`: the
        // rare-led walk scores each of its documents in full, the other terms
        // are ranked alone, and the two are merged (`rare_split`). The general
        // walk would instead score nearly every document of the other terms,
        // as a top `k` made mostly of one-term documents keeps the bar low.
        let rare_under_k = cursors.iter().any(|c| c.df < k as u64);
        let wand_pair = k <= WAND_BMW_2TERM_MAX_K && two_term_has_rare_anchor(&cursors);
        if cursors.len() == 2 && filter.is_none() && no_floor && (wand_pair || rare_under_k) {
            match self.rare_led_pair(column_id, &cursors, k) {
                Ok(hits) => Ok(hits),
                Err(RareLedFallback::Seed(seed)) => {
                    self.run_seeded_union(column_id, cursors, k, seed)
                }
                Err(RareLedFallback::Split { rare, hits }) => {
                    self.rare_split(column_id, cursors, rare, hits, k)
                }
                Err(RareLedFallback::Walk) if wand_pair => self.run_wand_bmw(column_id, cursors, k),
                Err(RareLedFallback::Walk) => {
                    self.run_seeded_union(column_id, cursors, k, f32::NEG_INFINITY)
                }
            }
        } else {
            let mut floor = floor_eff;
            if cursors.len() > 2
                && cursors.len() - 1 <= MAX_SHOULDS_LED
                && (k <= WAND_BMW_2TERM_MAX_K || rare_under_k)
                && filter.is_none()
                && no_floor
            {
                match self.rare_led(column_id, &cursors, k) {
                    Ok(hits) => return Ok(hits),
                    Err(RareLedFallback::Seed(seed)) => floor = seed,
                    Err(RareLedFallback::Split { rare, hits }) => {
                        return self.rare_split(column_id, cursors, rare, hits, k);
                    }
                    Err(RareLedFallback::Walk) => {}
                }
            }
            match filter.is_none() && no_floor {
                true => self.run_seeded_union(column_id, cursors, k, floor),
                false => {
                    self.run_windowed_maxscore(column_id, cursors, k, filter, floor, 0, u32::MAX)
                }
            }
        }
    }

    /// The windowed MaxScore walk of a union nothing filters, its bar
    /// started at the larger of `bar` (a rare-led seed, or `NEG_INFINITY`)
    /// and, from [`UNION_SEED_MIN_K`], the [`union_seed`]. Both are lower
    /// bounds on the true `k`-th best, so neither prunes a top-`k` doc.
    fn run_seeded_union(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        mut bar: f32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        if k >= UNION_SEED_MIN_K {
            let dl_norm_k1 = self.columns[column_id as usize].dl_norm_k1();
            bar = bar.max(union_seed(&cursors, k, dl_norm_k1));
        }
        self.run_windowed_maxscore(column_id, cursors, k, None, bar, 0, u32::MAX)
    }

    /// The top `k` of a union whose rare term (cursor `rare`) holds fewer than
    /// `k` documents, all of them already in `rare_hits` with full scores.
    /// The other terms are ranked alone to `k`, the rare documents dropped
    /// from that list and the two merged. A document without the rare term
    /// scores the same in both, summed in the same order (removing a term
    /// keeps the others' order). One that makes the final top `k` is in the
    /// others' top `k`: were `k` documents ahead of it there, each would still
    /// be ahead with its full score, the rare term only adding to it.
    fn rare_split(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        rare: usize,
        rare_hits: Vec<(FtsDocId, f32)>,
        k: usize,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        cursors.remove(rare);
        let rest = self.dispatch_or_algo(column_id, cursors, k, None, f32::NEG_INFINITY)?;
        let mut rare_ids: Vec<FtsDocId> = rare_hits.iter().map(|&(d, _)| d).collect();
        rare_ids.sort_unstable();
        let mut merged = rare_hits;
        merged.extend(
            rest.into_iter()
                .filter(|(d, _)| rare_ids.binary_search(d).is_err()),
        );
        merged.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        merged.truncate(k);
        Ok(merged)
    }

    /// [`Self::rare_led_pair`] for three or more terms: one term much rarer
    /// than every other, and first in the windowed kernel's order, leads,
    /// the others are probed at its documents, and the result stands when
    /// the k-th best is strictly above the sum of the others' maxima, which
    /// bounds any document without the rare term. The
    /// bound is a sum of bounds, rounded, so it may only equal such a
    /// document's score: strictly above, not at, is what rules a tie out.
    fn rare_led(
        &self,
        column_id: u32,
        cursors: &[TermCursor],
        k: usize,
    ) -> Result<Vec<(FtsDocId, f32)>, RareLedFallback> {
        // The terms in the order the windowed kernel sums them (descending
        // by term maximum, the same sort on the same input, so the same
        // permutation): the walk adds the lead first and the others in the
        // order given, so a score here is the kernel's bits. The rare term
        // must come first in that order to lead.
        let mut order: Vec<usize> = (0..cursors.len()).collect();
        order.sort_unstable_by(|&a, &b| {
            cursors[b]
                .term_max_bm25
                .partial_cmp(&cursors[a].term_max_bm25)
                .unwrap_or(Ordering::Equal)
        });
        let rare = &cursors[order[0]];
        if rare.df == 0 || rare.df > RARE_LED_MAX_DF {
            return Err(RareLedFallback::Walk);
        }
        let others: Vec<TermCursor> = order[1..].iter().map(|&i| cursors[i].clone()).collect();
        // A lead with fewer than `k` documents is cheap to walk however common
        // the others are; otherwise they must be much commoner to be worth it.
        if rare.df >= k as u64
            && others
                .iter()
                .any(|c| c.df < rare.df.saturating_mul(WAND_BMW_2TERM_DF_RATIO))
        {
            return Err(RareLedFallback::Walk);
        }
        let others_max: f32 = others.iter().map(|c| c.term_max_bm25).sum();
        let (hits, complete) = self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
            must_lead_should_walk(
                rare.clone(),
                others,
                k,
                None,
                f32::NEG_INFINITY,
                dl_norm_k1,
                heap,
            );
            heap.len() >= k && heap.peek().expect("full heap").0 > others_max
        });
        seed_or_hits(hits, complete, k, order[0])
    }

    /// A two-term OR of a rare term and a much commoner one, answered from
    /// the rare term's documents alone when that is provably the answer.
    ///
    /// Every document holding the rare term is ranked first, the rare term
    /// leading and the common one probed ([`must_lead_should_walk`]). A
    /// document without the rare term scores at most the common term's
    /// maximum, an upper bound strictly above any score it stands for, so
    /// once the k-th best reaches it no such document can enter the top k,
    /// not even on a tie, and the ranked rare-term documents are the answer.
    /// Otherwise an `Err` saying how the caller goes on: from a seed for the
    /// bar, by splitting off the rare term, or with the general walk.
    /// This skips the general walk's opening stretch, where the threshold is
    /// still below the common term's maximum and its documents are scored
    /// one by one until `k` rare-term documents have been seen.
    fn rare_led_pair(
        &self,
        column_id: u32,
        cursors: &[TermCursor],
        k: usize,
    ) -> Result<Vec<(FtsDocId, f32)>, RareLedFallback> {
        let rare_at = usize::from(cursors[0].df > cursors[1].df);
        let (rare, common) = (&cursors[rare_at], &cursors[1 - rare_at]);
        if rare.df > RARE_LED_MAX_DF {
            return Err(RareLedFallback::Walk);
        }
        let common_max = common.term_max_bm25;
        let (hits, complete) = self.scored_top_k(column_id, k, |dl_norm_k1, heap| {
            must_lead_should_walk(
                rare.clone(),
                vec![common.clone()],
                k,
                None,
                f32::NEG_INFINITY,
                dl_norm_k1,
                heap,
            );
            heap.len() >= k && heap.peek().expect("full heap").0 >= common_max
        });
        seed_or_hits(hits, complete, k, rare_at)
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
    /// Hits in the blob's own id space, deliberately untranslated: the
    /// benches this serves compare one algorithm against another over
    /// the same blob, so both sides are in the same space and the
    /// translation every other search entry point performs would only
    /// add work to the thing being measured.
    #[doc(hidden)]
    pub async fn search_with_algo_for_bench(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        algo: OrAlgo,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let cursors = self
            .build_term_cursors(
                column_id,
                terms,
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
            )
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
    use std::{collections::HashMap, sync::Arc};

    use bytes::Bytes;
    use rand::{RngExt, SeedableRng, rngs::StdRng};

    use super::{
        super::{search::seed, test_util::*},
        *,
    };
    use crate::superfile::{
        fts::{
            builder::FtsBuilder,
            posting::{ENCODING_BITSET, block_encoding},
            reader::BoolMode,
            short::SHORT_MAX_DF,
            tokenize::StandardTokenizer,
        },
        id_space::{FtsDocId, RowId},
    };

    /// The held mask is one bit per lane: docs at a word's last lane and
    /// the next word's first set the bits either side of the edge, out to
    /// the block's last lane, and a doc past the block sets nothing.
    #[test]
    fn positions_held_sets_one_bit_per_lane_across_word_edges() {
        let ids: Vec<u32> = (0..BLOCK_LEN as u32).map(|i| 3 * i).collect();
        let lanes = [0usize, 63, 64, 127, 128, BLOCK_LEN - 1];
        let mut docs: Vec<u32> = lanes.iter().map(|&p| ids[p]).collect();
        docs.insert(1, 1);
        docs.push(ids[BLOCK_LEN - 1] + 1);
        let mut want = [0u64; HELD_WORDS];
        for p in lanes {
            want[p / HELD_WORD_BITS] |= 1 << (p % HELD_WORD_BITS);
        }
        assert_eq!(positions_held(&ids, &docs), want);
    }

    #[tokio::test]
    async fn token_match_doc_set_matches_bm25_for_same_terms() {
        // token_match(Or) must return exactly the doc set bm25 ranks.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let mut bm25: Vec<RowId> = r
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
    async fn a_rare_term_with_fewer_docs_than_k_merges_with_the_others_alone() {
        // A rare word under `k` documents, every one of them also heavy in the
        // common words, so the common words ranked alone put the rare
        // documents at their top: the merge must drop those copies and still
        // fill `k`, answering exactly as the exhaustive walk does.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register column");
        let mut texts = Vec::new();
        for i in 0..3000u32 {
            texts.push(match i {
                _ if i % 150 == 7 => "kalakaua music music music city city".to_string(),
                _ if i % 3 == 0 => format!("music filler{} word", i % 17),
                _ if i % 5 == 0 => format!("city filler{} music", i % 13),
                _ => format!("filler{} other words here", i % 29),
            });
        }
        for (i, text) in texts.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        for terms in [&["music", "kalakaua"][..], &["music", "kalakaua", "city"]] {
            for k in [25usize, 60, 128] {
                let got: Vec<(u32, u32)> = r
                    .search("body", terms, k, BoolMode::Or)
                    .await
                    .expect("search")
                    .into_iter()
                    .map(|(d, s)| (d.get(), s.to_bits()))
                    .collect();
                let want: Vec<(u32, u32)> = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Exhaustive)
                    .await
                    .expect("exhaustive")
                    .into_iter()
                    .map(|(d, s)| (d.get(), s.to_bits()))
                    .collect();
                assert_eq!(got.len(), k.min(want.len()), "{terms:?} k {k}");
                let docs = |v: &[(u32, u32)]| v.iter().map(|&(d, _)| d).collect::<Vec<_>>();
                assert_eq!(docs(&got), docs(&want), "{terms:?} k {k}");
            }
        }
    }

    #[tokio::test]
    async fn exhaustive_and_bmm_agree_on_top_k() {
        // Build a larger blob so multi-term OR queries are
        // interesting (some docs have multiple terms, some have one).
        // Both algorithms must return identical top-K (descending
        // score, ascending doc_id tiebreak).
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false)
            .expect("register column");
        // `common` is present at doc 0 and docs 100..=300 — more than one
        // block, so it takes the long form (a single-block term would be
        // written in the short form, which has no bitset to probe). Its
        // first block holds docs 0 and 100..=226. First-doc 0 word-aligns
        // the bitset base to 0, so doc 100 -> bit 100 (presence word 1)
        // and doc 165 -> bit 165 (presence word 2); the 0->100 gap widens
        // the deltas enough that the block takes the bitset encoding, not
        // PFOR. tf is 1 everywhere except 3 at doc 100 and 2 at doc 165.
        for id in 0u32..=300 {
            let text = match id {
                0 => "common",
                100 => "common common common",
                165 => "common common",
                101..=300 => "common",
                _ => "filler", // 1..=99: present docs that don't carry `common`
            };
            b.add_doc(0, id, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let mut cursors = r
            .build_term_cursors(
                0,
                &["common"],
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
            )
            .await
            .expect("build common cursor");
        let cursor = &mut cursors[0];

        // Guard the test's own premise: the block must be a bitset, else the
        // probe takes the PACKED fallback and the rank path under test is skipped.
        let blk = cursor.blocks[0];
        assert_eq!(
            block_encoding(&cursor.bytes[blk.block_byte_offset..blk.block_byte_end]),
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        // posting blocks, with five query terms of differing
        // document frequency and a handful of docs carrying all five.
        // Running WAND+BMW at a small k forces the pivot to move, the
        // block-upper-bound skip to fire, lagging cursors to re-align,
        // and the 4-wide SIMD scoring pack to be used on the
        // all-terms docs — then cross-checks the result against BMM.

        /// Total planted docs: several `BLOCK_LEN`, so the dense-term
        /// posting lists occupy multiple blocks.
        const N_DOCS: u32 = 4 * BLOCK_LEN as u32;
        /// Requested top-K — small, so the heap fills early and the
        /// score threshold starts pruning blocks.
        const K: usize = 5;

        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
                    r.build_term_cursors(
                        col,
                        neg,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("neg cursors"),
                );
                let wms = r
                    .run_windowed_maxscore(
                        col,
                        r.build_term_cursors(
                            col,
                            pos,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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
                    r.build_term_cursors(
                        col,
                        neg,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("neg cursors"),
                );
                let bmm = r
                    .run_max_score_bmm(
                        col,
                        r.build_term_cursors(
                            col,
                            pos,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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
            // Sometimes span more than one OR_WINDOW.
            let n_docs = 500 + (rng() % u64::from(OR_WINDOW + OR_WINDOW / 4)) as u32;
            // Per-term inclusion probability (out of 8) — varied densities.
            let probs: Vec<u64> = (0..vocab.len()).map(|_| 1 + rng() % 8).collect();
            let mut b = FtsBuilder::new();
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
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
    async fn windowed_maxscore_window_partition_demotes_a_stopword() {
        // A stopword in every doc whose tf reaches 3 only in every eighth
        // block (2 elsewhere in those blocks, 1 in the rest) unioned with a
        // rare term that stops two thirds of the way through the corpus.
        // At k large enough that the threshold settles between the
        // stopword's tf-1 and tf-3 scores, the stopword stays essential by
        // its term max but its block maxima over most windows fall under the
        // threshold: those windows demote it to a probe-only term, and once
        // the rare term is exhausted the windows are skipped whole. Small k
        // (stopword non-essential outright) and huge k (heap never fills)
        // bracket the path. Every k must match per-candidate MaxScore+BMM.
        const N_DOCS: u32 = OR_WINDOW * 3 + 500;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let hot = (i / 128).is_multiple_of(8);
            let the_tf = match (hot, i % 4) {
                (true, 0) => 3,
                (true, _) => 2,
                (false, _) => 1,
            };
            let mut text = String::new();
            for _ in 0..the_tf {
                text.push_str("the ");
            }
            if i % 37 == 0 && i < OR_WINDOW * 2 {
                for _ in 0..=(i / 37 % 4) {
                    text.push_str("incredibles ");
                }
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        for terms in [&["the", "incredibles"], &["incredibles", "the"]] {
            for k in [
                1usize,
                10,
                128,
                300,
                600,
                1000,
                1500,
                3000,
                N_DOCS as usize + 1,
            ] {
                let bmm = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Bmm)
                    .await
                    .expect("bmm");
                let wms = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::WindowedMaxscore)
                    .await
                    .expect("wms");
                assert_eq!(bmm.len(), wms.len(), "len {terms:?} k={k}");
                for ((db, sb), (dw, sw)) in bmm.iter().zip(wms.iter()) {
                    assert_eq!(db, dw, "doc mismatch {terms:?} k={k}: bmm={db} wms={dw}");
                    assert!(
                        (sb - sw).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sb} vs {sw}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn windowed_maxscore_sole_essential_window_agrees_with_bmm() {
        // A stopword in four docs out of five, hot (tf 3) in every eighth
        // block, unioned with a term that occurs three times in the whole
        // corpus, each occurrence just past a window-cap boundary. At k large
        // enough that the threshold settles under the stopword's hot score,
        // both terms stay essential, yet between the rare term's postings the
        // stopword is the only essential with docs: those windows run to the
        // cap, the stopword's cold blocks are skipped by their metadata, and
        // the block that reaches past the cap is scored up to it, so the rare
        // term's doc right after the cap still collects the stopword's
        // contribution. With no non-essential term the window admits straight
        // off the presence bitmask. Every k must match per-candidate
        // MaxScore+BMM, including k = 1 (stopword non-essential outright) and
        // a k the heap never fills.
        const N_DOCS: u32 = OR_WINDOW * 3 + 500;
        /// Where in each window's span the rare term sits. The common term
        /// covers four docs in five, so its blocks end off the window cap and
        /// one of them straddles it; this offset puts the rare doc inside that
        /// block, past the cap.
        const RARE_OFFSET: u32 = 200;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let hot = (i / 128).is_multiple_of(8);
            let the_tf = match (hot, i % 4) {
                (true, 0) => 3,
                (true, _) => 2,
                (false, _) => 1,
            };
            let mut text = String::new();
            if i % 5 != 0 {
                for _ in 0..the_tf {
                    text.push_str("the ");
                }
            }
            // Just past a window cap, inside the common term's block that
            // straddles it: the block cannot be skipped on the sole-essential
            // bound, which counts no other essential term, because this doc
            // lies beyond the window that bound was taken over.
            if i % OR_WINDOW == RARE_OFFSET {
                text.push_str("incredibles ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        for terms in [&["the", "incredibles"], &["incredibles", "the"]] {
            for k in [
                1usize,
                3,
                10,
                50,
                128,
                200,
                600,
                1500,
                3000,
                N_DOCS as usize + 1,
            ] {
                let bmm = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Bmm)
                    .await
                    .expect("bmm");
                let wms = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::WindowedMaxscore)
                    .await
                    .expect("wms");
                assert_eq!(bmm.len(), wms.len(), "len {terms:?} k={k}");
                for ((db, sb), (dw, sw)) in bmm.iter().zip(wms.iter()) {
                    assert_eq!(db, dw, "doc mismatch {terms:?} k={k}: bmm={db} wms={dw}");
                    assert!(
                        (sb - sw).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sb} vs {sw}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn no_window_bound_falls_below_a_real_score_it_covers() {
        // The walk drops whole runs of blocks whose bound cannot beat the bar.
        // Comparing the returned ranking against a reference walk does not test
        // that bound: shrink it a little and the docs it wrongly drops are ones
        // that would not have placed anyway, so every ranking still matches,
        // and only a large cut reaches a doc that would have placed. Nor is it
        // enough to check that a dropped doc scored under the bar, because a
        // loose bound satisfies that with room to spare. So check the bound on
        // its own terms: over the docs it covers, it must never fall below a
        // real score.
        //
        // Randomised over several corpora, because the bound's slack depends on
        // how far a block's maximum sits above the docs inside it, and that is
        // what varying term frequency and document length control.
        const N_DOCS: u32 = OR_WINDOW * 3 + 500;
        const TRIALS: u64 = 6;
        const SCORE_EPS: f32 = 1e-4;
        /// In-block position of the planted document described below.
        const SATURATED_SLOT: u32 = 7;
        let mut checked_ranges = 0usize;
        let mut checked_docs = 0usize;
        for seed in 0..TRIALS {
            let mut rng = StdRng::seed_from_u64(seed);
            // The shape that drives the skip: a term in most documents whose
            // blocks differ sharply in their maximum, so most of them fall
            // under the bar, beside a term rare enough that whole windows hold
            // none of it. The periods and frequencies are drawn per trial,
            // since how far a block's maximum sits above the documents inside
            // it is exactly what decides the bound's slack.
            let hot_period = rng.random_range(4..12u32);
            let hot_tf = rng.random_range(3..6u32);
            let common_gap = rng.random_range(4..7u32);
            let rare_period = rng.random_range(1500..4200u32);
            let rare_offset = rng.random_range(0..1500u32);
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..N_DOCS {
                let hot = (i / BLOCK_LEN as u32).is_multiple_of(hot_period);
                let tf = match (hot, i % 4) {
                    (true, 0) => hot_tf,
                    (true, _) => 2,
                    (false, _) => 1,
                };
                let mut text = String::new();
                // One document per block carries the common term at its
                // highest frequency and the weak term, in the shortest
                // document in the corpus. Both terms therefore reach their
                // maximum weight on the same document, so the bound over that
                // block equals a real score instead of sitting above every one
                // of them, and any shrink of it is immediately wrong. It holds
                // neither of the other terms, so it does not stop a window
                // from being a single essential term's own.
                if i % BLOCK_LEN as u32 == SATURATED_SLOT {
                    for _ in 0..hot_tf {
                        text.push_str("alpha ");
                    }
                    text.push_str("delta");
                    b.add_doc(0, i, text.trim()).expect("add doc");
                    continue;
                }
                if !i.is_multiple_of(common_gap) {
                    for _ in 0..tf {
                        text.push_str("alpha ");
                    }
                }
                if i % rare_period == rare_offset % rare_period {
                    text.push_str("beta ");
                }
                if rng.random_range(0..50u32) == 0 {
                    text.push_str("gamma ");
                }
                // A term in nearly every document, so its weight is the
                // smallest and the rising bar makes it non-essential first.
                // Without one the sole-essential bound, which sums exactly the
                // non-essential weights, is zero in every window and a test
                // cannot tell a correct one from a shrunken one.
                if rng.random_range(0..10u32) != 0 {
                    text.push_str("delta ");
                }
                // Every other document is longer, so the planted one keeps the
                // shortest length and with it the highest weight per term.
                for f in 0..=rng.random_range(0..6u32) {
                    text.push_str(&format!("pad{f:02} "));
                }
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(blob, json).expect("open");
            for terms in [
                &["alpha", "beta"][..],
                &["beta", "alpha"][..],
                &["alpha", "beta", "gamma"][..],
                &["alpha", "beta", "delta"][..],
                &["alpha", "beta", "gamma", "delta"][..],
            ] {
                // The unpruned ranking: above the match count the heap never
                // fills, the bar stays at negative infinity and no bound is
                // consulted, so these are the true scores.
                let truth: HashMap<FtsDocId, f32> = r
                    .search_with_algo_for_bench("body", terms, N_DOCS as usize + 1, OrAlgo::Bmm)
                    .await
                    .expect("truth")
                    .into_iter()
                    .collect();
                for k in [1usize, 10, 100, 600, 1000, 3000] {
                    BOUND_CLAIMS.with(|v| v.borrow_mut().clear());
                    let _ = r
                        .search_with_algo_for_bench("body", terms, k, OrAlgo::WindowedMaxscore)
                        .await
                        .expect("wms");
                    let claims = BOUND_CLAIMS.with(|v| v.borrow().clone());
                    checked_ranges += claims.len();
                    for claim in claims {
                        for doc in claim.from..claim.to {
                            let Some(&score) = truth.get(&FtsDocId::new(doc)) else {
                                continue;
                            };
                            checked_docs += 1;
                            assert!(
                                score <= claim.bound + SCORE_EPS,
                                "seed {seed} {terms:?} k={k}: doc {doc} scores {score}, above \
                                 the {} bound covering docs {}..{}",
                                claim.bound,
                                claim.from,
                                claim.to
                            );
                        }
                    }
                }
            }
        }
        // A bound that is never exercised proves nothing, so fail loudly if a
        // future change stops the walk from dropping anything here.
        assert!(
            checked_ranges >= 20 && checked_docs >= 5_000,
            "the corpora stopped exercising the bound: {checked_ranges} ranges, {checked_docs} docs"
        );
    }

    #[tokio::test]
    async fn and_scored_walk_agrees_with_the_flat_merge() {
        // A rare driver whose per-doc score spans most of its idf inside a
        // single block, AND-ed with a dense (bitset) term and a mid-density
        // one. The window bound can only see the driver's block max, so what
        // rejects the driver's low-tf, long-document postings here is the
        // walk's survivor filter on each doc's own score. The flat merge walks
        // the same intersection with no such filter, so a doc dropped wrongly
        // surfaces as a top-k mismatch. The k ladder brackets a bar that bites
        // from the first window (k = 1) through one that never fills (k above
        // the 128 matching docs, leaving the filter shut).
        const N_DOCS: u32 = 128 * 40;
        const DRIVER_EVERY: u32 = 32;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            for _ in 0..(1 + i % 3) {
                text.push_str("the ");
            }
            if !i.is_multiple_of(5) {
                for _ in 0..(1 + i % 2) {
                    text.push_str("of ");
                }
            }
            if i.is_multiple_of(DRIVER_EVERY) {
                let nth = i / DRIVER_EVERY;
                for _ in 0..(1 + nth % 8) {
                    text.push_str("book ");
                }
                // Pad a slice of the driver's docs so the length norm, not
                // the tf alone, spreads their scores across one block.
                for _ in 0..(nth % 5) * 7 {
                    text.push_str("filler ");
                }
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        let norms = &r.columns[col as usize].dl_norm_k1();
        let terms = ["book", "the", "of"];
        let build = async || {
            r.build_term_cursors(
                col,
                &terms,
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
            )
            .await
            .expect("cursors")
        };
        for k in [1usize, 5, 10, 50, 200] {
            let mut walk_heap = BinaryHeap::new();
            let mut walk_sink = ScoreSink {
                heap: &mut walk_heap,
                k,
                filter: None,
                floor_eff: f32::NEG_INFINITY,
            };
            r.and_scored_walk(build().await, norms, &mut walk_sink);
            let walk = drain_top_k_desc(walk_heap);

            let mut merge_heap = BinaryHeap::new();
            let mut merge_sink = ScoreSink {
                heap: &mut merge_heap,
                k,
                filter: None,
                floor_eff: f32::NEG_INFINITY,
            };
            let mut merge_cursors = build().await;
            merge_cursors.sort_by_key(|c| c.df);
            r.and_flat_merge(&mut merge_cursors, norms, &mut merge_sink);
            let merge = drain_top_k_desc(merge_heap);

            assert_eq!(walk.len(), merge.len(), "hit count k={k}");
            for ((dw, sw), (dm, sm)) in walk.iter().zip(merge.iter()) {
                assert_eq!(dw, dm, "doc mismatch k={k}: walk={dw} merge={dm}");
                assert!((sw - sm).abs() < 1e-4, "score mismatch k={k}: {sw} vs {sm}");
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
                        r.build_term_cursors(
                            col,
                            terms,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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
                        r.build_term_cursors(
                            col,
                            terms,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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

    /// Docs in the window-kernel tests' norm table: three windows, less a
    /// few, so a tile of the last window reaches past the table's end.
    #[cfg(target_arch = "x86_64")]
    const KERNEL_DOCS: u32 = 3 * OR_WINDOW - 5;

    /// A reader of [`KERNEL_DOCS`] docs whose lengths spread across the norm
    /// table's buckets.
    #[cfg(target_arch = "x86_64")]
    fn kernel_norms_reader() -> FtsReader {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..KERNEL_DOCS {
            b.add_doc(0, i, &"w ".repeat(1 + (i as usize * 7919) % 300))
                .expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// The window `start` with each of `ids` (with `tfs`) scored alone and
    /// added, and their presence bits.
    #[cfg(target_arch = "x86_64")]
    fn scored_alone(
        mut start: Vec<f32>,
        (ids, tfs): (&[u32], &[u32]),
        idf: f32,
        base: u32,
        norms: &NormTable,
    ) -> (Vec<f32>, [u64; OR_WINDOW_WORDS]) {
        let mut present = [0u64; OR_WINDOW_WORDS];
        for (&d, &tf) in ids.iter().zip(tfs) {
            let local = (d - base) as usize;
            start[local] += bm25::score_with_dl_norm_k1(idf, tf, norms.get(d));
            present[local >> 6] |= 1u64 << (local & 63);
        }
        (start, present)
    }

    /// Two windows hold the same bits and presence.
    #[cfg(target_arch = "x86_64")]
    fn assert_same_window(got: (&[f32], &[u64]), want: (&[f32], &[u64]), trial: usize) {
        assert_eq!(got.1, want.1, "trial {trial}: presence");
        for (slot, (a, b)) in got.0.iter().zip(want.0).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "trial {trial}: slot {slot}");
        }
    }

    /// The AVX-512 window accumulate leaves the same bits in the window as
    /// the scalar one over the postings it consumes: dense runs (one or two
    /// aligned tiles) and sparse ones, anywhere in any window of the doc
    /// space, up to the window's last slot and the norm table's last doc,
    /// onto a window that already holds scores.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn accumulate_run_avx512_matches_scalar() {
        if !avx512_kernels() {
            return;
        }
        let r = kernel_norms_reader();
        let norms = r.columns[0].dl_norm_k1();
        let mut next = xorshift(0x9E37_79B9_7F4A_7C15);
        let mut at_window_end = 0;
        for trial in 0..600 {
            let base = OR_WINDOW * (next() % 3) as u32;
            let n = 8 + (next() % 120) as usize;
            let max_gap = [1u64, 2, 3, 40][trial % 4];
            // Half the runs start at the window's front, half close to its
            // end, where a run's last tiles are the window's last.
            let lead = (next() % 64) as u32;
            let mut d = match trial % 2 {
                0 => base + lead,
                _ => (base + OR_WINDOW).saturating_sub(lead + 8),
            };
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                if d >= base + OR_WINDOW || d >= KERNEL_DOCS {
                    break;
                }
                ids.push(d);
                d += 1 + (next() % max_gap) as u32;
            }
            let tfs: Vec<u32> = ids.iter().map(|_| 1 + (next() % 20) as u32).collect();
            let start: Vec<f32> = (0..OR_WINDOW)
                .map(|_| (next() % 1000) as f32 / 97.0)
                .collect();
            let mut got = start.clone();
            let mut got_present = [0u64; OR_WINDOW_WORDS];
            let idf = 1.7f32;
            // SAFETY: both features are present (checked above).
            let used = unsafe {
                accumulate_run_avx512(idf, &ids, &tfs, base, &mut got, &mut got_present, norms)
            };
            assert!(
                used <= ids.len() && used.is_multiple_of(8),
                "trial {trial}: used {used}"
            );
            if used > 0 && ids[used - 1] + ACCUMULATE_TILE as u32 > base + OR_WINDOW {
                at_window_end += 1;
            }
            let want = scored_alone(start, (&ids[..used], &tfs[..used]), idf, base, norms);
            assert_same_window((&got, &got_present), (&want.0, &want.1), trial);
        }
        assert!(at_window_end > 0, "some run reached the window's last tile");
    }

    /// The bitset kernel adds exactly what scoring each doc alone adds, and
    /// marks the same docs present: bitsets of any density whose first bit
    /// sits before the window, inside it, or anywhere relative to a tile,
    /// and whose last runs past the window's end or the norm table's,
    /// cut to the docs a window holds, onto a window that already holds
    /// scores.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn accumulate_bitset_avx512_matches_scalar() {
        if !avx512_kernels() {
            return;
        }
        let r = kernel_norms_reader();
        let norms = r.columns[0].dl_norm_k1();
        let mut next = xorshift(0x2545_F491_4F6C_DD1D);
        let (mut before_window, mut past_window, mut past_table) = (0, 0, 0);
        for trial in 0..600 {
            let base = OR_WINDOW * (next() % 3) as u32;
            let n_words = 1 + (next() % 40) as usize;
            let span = n_words as u32 * 64;
            // The block starts up to 512 docs before the window, or inside
            // it; some on a word boundary, most not.
            let lowest = base.saturating_sub(512);
            let highest = base + OR_WINDOW - 64;
            let mut bit_base = lowest + (next() % u64::from(highest - lowest)) as u32;
            if trial % 4 == 0 {
                bit_base &= !63;
            }
            // One doc in `odds` is absent, or (negative) one in `-odds` present.
            let odds = [2i64, 8, 64, -4][trial % 4];
            let words: Vec<u8> = (0..n_words * 8)
                .map(|_| {
                    (0..8).fold(0u8, |w, bit| {
                        let roll = next() % odds.unsigned_abs();
                        w | u8::from((roll != 0) == (odds > 0)) << bit
                    })
                })
                .collect();
            let all: Vec<u32> = (0..span)
                .filter(|&k| words[k as usize / 8] >> (k % 8) & 1 == 1)
                .map(|k| bit_base + k)
                .filter(|&d| d >= base && d < base + OR_WINDOW && d < KERNEL_DOCS)
                .collect();
            if all.is_empty() {
                continue;
            }
            before_window += usize::from(bit_base < base);
            past_window += usize::from(bit_base + span > base + OR_WINDOW);
            let last = all[all.len() - 1];
            past_table += usize::from((last & !15) + ACCUMULATE_TILE as u32 > KERNEL_DOCS);
            // A window takes the docs from the cursor's to the window's end.
            let from = (next() % all.len() as u64) as usize;
            let ids = &all[from..];
            let tfs: Vec<u32> = ids.iter().map(|_| 1 + (next() % 20) as u32).collect();
            let start: Vec<f32> = (0..OR_WINDOW)
                .map(|_| (next() % 1000) as f32 / 97.0)
                .collect();
            let mut got = start.clone();
            let mut got_present = [0u64; OR_WINDOW_WORDS];
            let idf = 1.7f32;
            let run = BitsetRun {
                words: &words,
                bit_base,
                docs: ids[0]..ids[ids.len() - 1] + 1,
                tfs: &tfs,
            };
            // SAFETY: both features are present (checked above).
            let added = unsafe {
                accumulate_bitset_avx512(idf, run, base, &mut got, &mut got_present, norms)
            };
            assert_eq!(added, ids.len(), "trial {trial}: docs added");
            let want = scored_alone(start, (ids, &tfs), idf, base, norms);
            assert_same_window((&got, &got_present), (&want.0, &want.1), trial);
        }
        assert!(before_window > 0, "some bitset began before the window");
        assert!(past_window > 0, "some bitset ran past the window");
        assert!(past_table > 0, "some tile reached past the norm table");
    }

    /// Pruning tests a bound summed in one order against a bar that a score
    /// summed in another order must beat; [`pruning_bar`] must cover the gap.
    /// Sums of the same non-negative terms in two orders stay above each
    /// other's pruning bar.
    #[test]
    fn pruning_bar_covers_summing_in_another_order() {
        let mut next = xorshift(0x9E37_79B9_7F4A_7C15);
        for _ in 0..20_000 {
            let n = 2 + (next() % (MAX_SHOULDS_LED as u64)) as usize;
            let terms: Vec<f32> = (0..n)
                .map(|_| (next() % 1_000_000) as f32 / 37_000.0 * (1 + next() % 3) as f32)
                .collect();
            let mut shuffled = terms.clone();
            for i in (1..n).rev() {
                shuffled.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            let a: f32 = terms.iter().sum();
            let b: f32 = shuffled.iter().fold(0.0, |acc, &t| acc + t);
            // A sum in one order is never under the pruning bar of the same
            // terms summed in another, so pruning never drops it.
            assert!(
                a > pruning_bar(b, n) && b > pruning_bar(a, n),
                "{terms:?}: {a} vs {b}"
            );
        }
    }

    /// The dispatched filter (AVX2 where present) and the scalar one keep the
    /// same entries in the same order, across lengths that end on and off an
    /// eight-lane boundary and bars that keep none, some and all.
    #[test]
    fn filter_survivors_dispatch_agrees_with_scalar() {
        let mut x: u32 = 0x9E37_79B9;
        for n in [0usize, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 128, 129] {
            let docs0: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
            let scores0: Vec<f32> = (0..n)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x % 1000) as f32 / 100.0
                })
                .collect();
            for min_score in [f32::NEG_INFINITY, 0.0, 2.5, 5.0, 9.99, f32::INFINITY] {
                let (mut d1, mut s1) = (docs0.clone(), scores0.clone());
                let (mut d2, mut s2) = (docs0.clone(), scores0.clone());
                let n1 = filter_survivors(&mut d1, &mut s1, min_score);
                let n2 = filter_survivors_scalar(&mut d2, &mut s2, min_score);
                assert_eq!(n1, n2, "count at n={n} bar={min_score}");
                assert_eq!(&d1[..n1], &d2[..n2], "docs at n={n} bar={min_score}");
                assert_eq!(&s1[..n1], &s2[..n2], "scores at n={n} bar={min_score}");
            }
        }
    }

    #[tokio::test]
    async fn wand_bmw_stopword_anchor_tail_agrees_with_bmm() {
        // A term in every doc (bitset blocks, tf 1..=3) unioned with a
        // rare term of varying tf: once the top-k fills, the stopword's
        // max score is below the threshold and the kernel takes the
        // probe-only tail. Its top-k must equal MaxScore+BMM for every k
        // that fills the heap early, and for k large enough that it does
        // not fill at all.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            for _ in 0..=(i % 3) {
                text.push_str("the ");
            }
            if i % 37 == 0 {
                for _ in 0..=(i / 37 % 4) {
                    text.push_str("rare ");
                }
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        for terms in [&["the", "rare"], &["rare", "the"]] {
            for k in [1usize, 5, 10, 50, 128, 400] {
                let cw = r
                    .build_term_cursors(
                        col,
                        terms,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("cursors");
                let cb = r
                    .build_term_cursors(
                        col,
                        terms,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("cursors");
                let wand = r.run_wand_bmw(col, cw, k).expect("wand");
                let bmm = r
                    .run_max_score_bmm(col, cb, k, None, f32::NEG_INFINITY)
                    .expect("bmm");
                assert_eq!(wand.len(), bmm.len(), "len mismatch {terms:?} k={k}");
                for ((dw, sw), (db, sb)) in wand.iter().zip(bmm.iter()) {
                    assert_eq!(dw, db, "doc mismatch {terms:?} k={k}: {dw} vs {db}");
                    assert!((sw - sb).abs() < 1e-4, "score mismatch {terms:?} k={k}");
                }
            }
        }
    }

    #[tokio::test]
    async fn wand_two_term_tail_fills_a_partial_heap() {
        // Entered with an empty heap and the threshold seeded at the weak
        // term's max score (a floor-seeded pivot would do this), the tail
        // must still push qualifying docs until the heap holds k and only
        // then evict, and end with the same top-k as MaxScore+BMM.
        const N_DOCS: u32 = OR_WINDOW + 700;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("the");
            if i % 37 == 0 {
                for _ in 0..=(i / 37 % 4) {
                    text.push_str(" rare");
                }
            }
            b.add_doc(0, i, &text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        let dl_norm_k1 = &r.columns[col as usize].dl_norm_k1();
        for k in [1usize, 5, 20] {
            let mut cursors = r
                .build_term_cursors(
                    col,
                    &["the", "rare"],
                    QueryIdf::ThisFile,
                    CursorUse::Score,
                    None,
                    None,
                )
                .await
                .expect("cursors");
            let (a, bb) = cursors.split_at_mut(1);
            let (weak, strong) = (&mut a[0], &mut bb[0]);
            assert!(weak.term_max_bm25 < strong.term_max_bm25);
            let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
            let mut threshold = weak.term_max_bm25;
            wand_two_term_tail(weak, strong, k, &mut heap, &mut threshold, dl_norm_k1);
            let tail = drain_top_k_desc(heap);
            let cb = r
                .build_term_cursors(
                    col,
                    &["the", "rare"],
                    QueryIdf::ThisFile,
                    CursorUse::Score,
                    None,
                    None,
                )
                .await
                .expect("cursors");
            let bmm = r
                .run_max_score_bmm(col, cb, k, None, f32::NEG_INFINITY)
                .expect("bmm");
            assert_eq!(tail.len(), k, "k={k}: the tail must fill the heap");
            for ((dt, st), (db, sb)) in tail.iter().zip(bmm.iter()) {
                assert_eq!(dt, db, "k={k}");
                assert!((st - sb).abs() < 1e-4, "k={k}");
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        let shapes: &[&[&str]] = &[&["alpha", "beta"], &["beta", "gamma"], &["alpha", "gamma"]];
        for terms in shapes {
            for k in [1usize, 5, 50, 128] {
                let cw = r
                    .build_term_cursors(
                        col,
                        terms,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("cursors");
                let cb = r
                    .build_term_cursors(
                        col,
                        terms,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        // common (df≈N) + rare (df≈N/200): ratio 200 ≥ 16 → anchor.
        let anchored = r
            .build_term_cursors(
                col,
                &["common", "rare"],
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
            )
            .await
            .expect("cursors");
        assert!(
            two_term_has_rare_anchor(&anchored),
            "rare+common should have a rare anchor"
        );
        // common (df≈N) + frequent (df≈N/2): ratio 2 < 16 → no anchor.
        let uniform = r
            .build_term_cursors(
                col,
                &["common", "frequent"],
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
            )
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
                    r.build_term_cursors(
                        col,
                        neg,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("neg cursors"),
                );
                let win = r
                    .run_windowed_union(
                        col,
                        r.build_term_cursors(
                            col,
                            pos,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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
                    r.build_term_cursors(
                        col,
                        neg,
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    )
                    .await
                    .expect("neg cursors"),
                );
                let bmm = r
                    .run_max_score_bmm(
                        col,
                        r.build_term_cursors(
                            col,
                            pos,
                            QueryIdf::ThisFile,
                            CursorUse::Score,
                            None,
                            None,
                        )
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
                r.build_term_cursors(col, pos, QueryIdf::ThisFile, CursorUse::Score, None, None)
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
            r.build_term_cursors(col, neg, QueryIdf::ThisFile, CursorUse::Score, None, None)
                .await
                .expect("neg"),
        );
        let filtered = r
            .run_windowed_union(
                col,
                r.build_term_cursors(col, pos, QueryIdf::ThisFile, CursorUse::Score, None, None)
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

    /// The exhaustive baseline scores four or more terms at one doc in packs
    /// of four and the rest alone; it must rank a five-term union as the
    /// production kernel does (the same docs, scores equal up to summation
    /// order).
    #[tokio::test]
    async fn exhaustive_union_of_five_terms_agrees_with_the_production_kernel() {
        let words = ["alpha", "beta", "gamma", "delta", "omega"];
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..2_000u32 {
            let text: Vec<&str> = words
                .iter()
                .enumerate()
                .filter(|&(w, _)| i % (w as u32 + 2) == 0 || i % 97 == 0)
                .map(|(_, &t)| t)
                .chain(["filler"])
                .collect();
            b.add_doc(0, i, &text.join(" ")).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        for k in [5usize, 50] {
            let got = r
                .search("body", &words, k, BoolMode::Or)
                .await
                .expect("search");
            let want = r
                .search_with_algo_for_bench("body", &words, k, OrAlgo::Exhaustive)
                .await
                .expect("exhaustive");
            assert_eq!(got.len(), want.len(), "k {k}");
            for ((dg, sg), (dw, sw)) in got.iter().zip(&want) {
                assert!((sg - sw).abs() < 1e-4, "k {k}: {dg:?} {sg} vs {dw:?} {sw}");
            }
        }
    }

    /// A reader over `n_docs` docs in which each `(word, per_mille)` lands in
    /// about that many docs per thousand, one to three times, each doc padded
    /// with up to 29 filler tokens; `seed` fixes the draw.
    fn random_corpus(seed: u64, n_docs: u32, words: &[(&str, u64)]) -> FtsReader {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        let mut x = seed;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for i in 0..n_docs {
            let mut text = String::new();
            for &(word, per_mille) in words {
                if next() % 1000 < per_mille {
                    for _ in 0..1 + next() % 3 {
                        text.push_str(word);
                        text.push(' ');
                    }
                }
            }
            text.push_str(&"pad ".repeat((next() % 30) as usize));
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// Assert that `search` ranks `terms` at `k` as the exhaustive walk does,
    /// doc for doc. Over one or two terms the scores match bit for bit, a sum
    /// of two being the same in either order; over more the kernels sum in
    /// different orders, so to within 1e-4.
    async fn assert_ranks_as_exhaustive(r: &FtsReader, terms: &[&str], k: usize) {
        let got = r
            .search("body", terms, k, BoolMode::Or)
            .await
            .expect("search");
        let want = r
            .search_with_algo_for_bench("body", terms, k, OrAlgo::Exhaustive)
            .await
            .expect("exhaustive");
        assert_eq!(got.len(), want.len(), "{terms:?} k {k}");
        for ((dg, sg), (dw, sw)) in got.iter().zip(&want) {
            let what = format!("{terms:?} k {k}: {} {sg} vs {} {sw}", dg.get(), dw.get());
            assert_eq!(dg.get(), dw.get(), "{what}");
            match terms.len() <= 2 {
                true => assert_eq!(sg.to_bits(), sw.to_bits(), "{what}"),
                false => assert!((sg - sw).abs() < 1e-4, "{what}"),
            }
        }
    }

    /// A union at a `k` the seed serves ranks as the unpruned exhaustive walk
    /// does: the seeded bar never drops a doc of the top `k`. One shape where
    /// the seed is tight — a common term with a less common one, so the
    /// union's `k`-th best is mostly the common term's own — and one with a
    /// stopword and overlapping terms, where it sits well below. At `k` =
    /// 2,000 the less common term holds fewer than `k` docs, so the pair is
    /// split instead and the lone common term is seeded.
    #[tokio::test]
    async fn seeded_large_k_union_agrees_with_the_exhaustive_walk() {
        let words = [("stop", 950), ("mid", 400), ("rare", 180), ("few", 8)];
        let r = random_corpus(0x2545_F491_4F6C_DD1D, 6_000, &words);
        for terms in [&["mid", "rare"][..], &["stop", "mid", "rare"]] {
            for k in [UNION_SEED_MIN_K, 1_000, 2_000] {
                assert_ranks_as_exhaustive(&r, terms, k).await;
            }
        }
    }

    /// A union whose rarest term is stored in short form, its whole list
    /// decoded up front with no block table, yet holds at least `k` docs: the
    /// seed passes over that term and seeds from the others, and the union
    /// ranks as the exhaustive walk does.
    #[tokio::test]
    async fn a_seed_passes_over_a_short_form_term() {
        let words = [("short", 33), ("mid", 133), ("wide", 167)];
        let r = random_corpus(0xD1B5_4A32_D192_ED03, 3_000, &words);
        let df = r.term_df("body", "short").await.expect("df").0;
        assert!(
            (UNION_SEED_MIN_K as u64..=SHORT_MAX_DF as u64).contains(&df),
            "`short` holds {df} docs"
        );
        for k in [UNION_SEED_MIN_K, df as usize] {
            assert_ranks_as_exhaustive(&r, &["short", "mid", "wide"], k).await;
        }
    }

    /// A ranked union promotes a non-essential term into a window where it
    /// holds few docs, reseeking it there after the windows before probed it,
    /// and ranks as the exhaustive walk does. `c` is in every doc of the
    /// first third, where it is probed, and in one doc in ten after, where
    /// it is accumulated; `a` and `b` are the essentials throughout.
    #[tokio::test]
    async fn a_promoted_term_ranks_as_the_exhaustive_walk() {
        const N_DOCS: u32 = 3 * OR_WINDOW + 500;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        let mut x: u64 = 0xA076_1D64_78BD_642F;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for i in 0..N_DOCS {
            let mut text = String::new();
            for word in ["a", "b"] {
                if next() % 10 < 3 {
                    text.push_str(&format!("{word} ").repeat(1 + (next() % 3) as usize));
                }
            }
            if i < N_DOCS / 3 || next() % 10 == 0 {
                text.push_str("c ");
            }
            text.push_str(&"pad ".repeat((next() % 30) as usize));
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        PROMOTIONS.with(|p| p.set(0));
        for k in [UNION_SEED_MIN_K, 100, 1_000] {
            assert_ranks_as_exhaustive(&r, &["a", "b", "c"], k).await;
        }
        assert!(PROMOTIONS.with(Cell::get) > 0, "no window promoted `c`");
    }

    /// A seed one place too high drops the `k`-th doc only where it is the
    /// best of its block and scores apart from the next better one. Here
    /// every doc of `x` scores apart (one length, a different tf each), and
    /// its tfs fall with the doc id, so each posting block holds the next
    /// `BLOCK_LEN` ranks: at `k` one past a block, the `k`-th doc heads the
    /// next. A union with a rarer term over its best docs, which splits and
    /// seeds `x` as a lone term, must rank exactly as the exhaustive walk
    /// does.
    #[tokio::test]
    async fn a_seed_over_distinct_scores_keeps_the_kth_doc() {
        /// Docs holding `x`, the `i`-th `X_DOCS - i` times; each doc's
        /// length is fixed so its tf alone orders the scores.
        const X_DOCS: u32 = 400;
        const DOC_LEN: u32 = 420;
        /// `y` is in `x`'s best docs, fewer than `k`: the split merges them
        /// back over `x` ranked alone, so the union's `k`-th is `x`'s own.
        const Y_DOCS: u32 = 30;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..3_000u32 {
            let x = X_DOCS.saturating_sub(i);
            let y = u32::from(i < Y_DOCS);
            let text = ["x "].repeat(x as usize).concat()
                + &["y "].repeat(y as usize).concat()
                + &["pad "].repeat((DOC_LEN - x - y) as usize).concat();
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        for k in [UNION_SEED_MIN_K, BLOCK_LEN + 1, 2 * BLOCK_LEN + 1] {
            assert_ranks_as_exhaustive(&r, &["x", "y"], k).await;
        }
    }

    /// The single-term walk's seed, over a list long enough to be seeded,
    /// keeps the `k`-th doc where a seed one place too high would lose it:
    /// `k - 1` docs score highest and alike, and the `k`-th scores apart and
    /// below them, alone in a block of the lowest scores, so that block's
    /// bound sits under the `(k-1)`-th. Every doc has one length, so tf alone
    /// orders the scores. Checked with the heap collector and the buffer.
    #[tokio::test]
    async fn a_single_term_seed_keeps_the_kth_doc_alone_in_its_block() {
        /// Docs past the single-term walk's seed floor, every one holding
        /// `x`, which a term in every doc takes in wide blocks.
        const N_DOCS: u32 = (seed::MIN_BLOCKS as u32 + 8) * BLOCK_LEN as u32;
        /// Tokens per doc.
        const DOC_LEN: usize = 4;
        for k in [UNION_SEED_MIN_K, BUFFERED_TOPK_MIN_K + 100] {
            // The best `k - 1` lead the list, inside the blocks the seed
            // reads; the `k`-th sits in the middle of the last block, so
            // neither its block nor its span of blocks holds one of them.
            let kth_doc = N_DOCS - BLOCK_LEN as u32 / 2;
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..N_DOCS {
                let tf = match i {
                    _ if i == kth_doc => 2,
                    _ if i < k as u32 - 1 => 3,
                    _ => 1,
                };
                let text = ["x "].repeat(tf).concat() + &["pad "].repeat(DOC_LEN - tf).concat();
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
            let layout = r
                .term_layout("body", "x")
                .await
                .expect("layout")
                .expect("x");
            assert!(layout.num_blocks >= seed::MIN_BLOCKS, "premise: {layout:?}");
            assert_ranks_as_exhaustive(&r, &["x"], k).await;
            let hits = r
                .search("body", &["x"], k, BoolMode::Or)
                .await
                .expect("search");
            assert_eq!(hits[k - 1].0.get(), kth_doc, "k {k}");
        }
    }

    /// A pair whose rarer term is too common for the rare-led walk yet under
    /// `k` documents: the walk declines it, and the pair runs the seeded
    /// windowed walk, which must rank as the exhaustive walk does, doc for
    /// doc.
    #[tokio::test]
    async fn a_pair_the_rare_led_walk_declines_runs_the_seeded_walk() {
        // `rarer` holds just more docs than the rare-led walk takes, and `k`
        // is past them; `commoner` holds more than `k`.
        let rarer_docs = RARE_LED_MAX_DF as u32 + 1_000;
        let k = rarer_docs as usize + 2_000;
        let n_docs = rarer_docs + 30_000;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..n_docs {
            let text = match (i < rarer_docs, i % 7 != 0) {
                (true, true) => format!(
                    "rarer commoner {}",
                    ["pad "].repeat((i % 5) as usize).concat()
                ),
                (true, false) => "rarer pad".to_string(),
                (false, _) => format!("commoner {}", ["pad "].repeat((i % 3) as usize).concat()),
            };
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_ranks_as_exhaustive(&r, &["rarer", "commoner"], k).await;
    }

    /// A union with a term holding fewer than `k` documents, at a `k` far
    /// past the small-`k` routes: the rare term is split off and its
    /// documents merged with the other terms ranked alone, which must rank as
    /// the exhaustive walk does, for a pair and for three terms.
    #[tokio::test]
    async fn a_union_with_a_term_under_k_documents_agrees_with_the_exhaustive_walk() {
        let words = [("stop", 900), ("mid", 300), ("rare", 8)];
        let r = random_corpus(0x9E37_79B9_7F4A_7C15, 6_000, &words);
        for terms in [&["rare", "stop"][..], &["rare", "mid", "stop"]] {
            for k in [WAND_BMW_2TERM_MAX_K + 1, 1_000] {
                assert_ranks_as_exhaustive(&r, terms, k).await;
            }
        }
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

    // ── Bound soundness ───────────────────────────────────────────────

    /// Proptest cases for the bound properties; each builds a reader.
    const BOUND_CASES: u32 = 32;
    /// Slack for the bound comparisons: the bounds are exact scores held
    /// as `f32` and the reference sums two `f32` terms, so only rounding
    /// separates them.
    const BOUND_EPS: f32 = 1e-4;
    /// Widest filler run planted per doc, so document lengths span the
    /// exact and the quantized regions of the stored length.
    const MAX_FILLER: u8 = 60;
    /// Rows in a bound corpus.
    const BOUND_DOCS: u32 = 900;

    /// A two-term corpus with per-doc term frequencies and lengths drawn
    /// from `seed`, indexed at a `(k1, b)` pair. Term `lead` is planted
    /// in every third doc from a random phase, `other` in a random
    /// subset of docs, so the two lists cross block boundaries at
    /// unrelated places.
    fn bound_reader(seed: u64, k1: f32, b: f32) -> (FtsReader, Vec<Option<u32>>, Vec<Option<u32>>) {
        let mut rng = seed | 1;
        let mut draw = |m: u64| -> u64 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % m
        };
        let phase = draw(3) as u32;
        let mut builder = FtsBuilder::new();
        builder
            .register_column_with_tokenizer(
                "body".into(),
                false,
                Arc::new(StandardTokenizer),
                bm25::Bm25Params::new(k1, b),
            )
            .expect("register");
        let mut lead_tf: Vec<Option<u32>> = Vec::with_capacity(BOUND_DOCS as usize);
        let mut other_tf: Vec<Option<u32>> = Vec::with_capacity(BOUND_DOCS as usize);
        let mut text = String::new();
        for doc in 0..BOUND_DOCS {
            text.clear();
            let lt = (doc % 3 == phase).then(|| 1 + draw(6) as u32);
            let ot = (draw(4) == 0).then(|| 1 + draw(9) as u32);
            for _ in 0..lt.unwrap_or(0) {
                text.push_str("lead ");
            }
            for _ in 0..ot.unwrap_or(0) {
                text.push_str("other ");
            }
            for f in 0..draw(u64::from(MAX_FILLER) + 1) {
                text.push_str(&format!("f{f} "));
            }
            if text.is_empty() {
                text.push_str("filler");
            }
            builder.add_doc(0, doc, text.trim_end()).expect("add doc");
            lead_tf.push(lt);
            other_tf.push(ot);
        }
        let json = format!(r#"[{{"name":"body","tokenizer":"standard","k1":{k1},"b":{b}}}]"#);
        let r =
            FtsReader::open(Bytes::from(builder.finish().expect("finish")), &json).expect("open");
        (r, lead_tf, other_tf)
    }

    /// `block_max_and_bound` is an upper bound on the AND score of every
    /// document in the window it declares: for each doc of the leader,
    /// the bound at that doc must be at least the true `lead + other`
    /// score of every doc carrying both terms in `[doc, window_end]`. A
    /// bound below a real score would let Block-Max-AND skip a block
    /// holding a top-k document, which is a silent recall loss the
    /// top-k oracles only catch when the skipped doc happens to be
    /// planted at the boundary. Lengths span both stored-length regions
    /// and the pair is non-standard, so the quantized norm and the
    /// bound rescale are both on the path.
    #[test]
    fn prop_block_max_and_bound_is_never_below_a_real_and_score() {
        use proptest::{
            prelude::*,
            test_runner::{Config, TestRunner},
        };
        let strategy = (any::<u64>(), 0.3f32..2.5, 0.0f32..=1.0);
        let mut runner = TestRunner::new(Config {
            cases: BOUND_CASES,
            ..Config::default()
        });
        runner
            .run(&strategy, |(seed, k1, b)| {
                let (r, lead_tf, other_tf) = bound_reader(seed, k1, b);
                let col = r.resolve_column_id("body").expect("col");
                let norms = &r.columns[col as usize].dl_norm_k1();
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .expect("runtime");
                let mut cursors = rt
                    .block_on(r.build_term_cursors(col, &["lead", "other"], QueryIdf::ThisFile, CursorUse::Score, None, None))
                    .expect("cursors");
                prop_assume!(cursors.len() == 2);
                let (lead_slice, others) = cursors.split_at_mut(1);
                let lead = &mut lead_slice[0];
                let score = |doc: u32, c: &TermCursor, tf: u32| -> f32 {
                    bm25::score_with_dl_norm_k1(c.idf_weight, tf, norms.get(doc))
                };
                while !lead.is_exhausted() {
                    let doc = lead.current_doc_id();
                    let lead_block_max = lead.current_block_max_bm25();
                    let (others_ub, window_end) =
                        block_max_and_bound(lead.current_block_last_doc_id(), others, doc);
                    let ub = lead_block_max + others_ub;
                    prop_assert!(window_end >= doc, "window end precedes the leader doc");
                    for d in doc..=window_end.min(BOUND_DOCS - 1) {
                        if let (Some(lt), Some(ot)) = (lead_tf[d as usize], other_tf[d as usize]) {
                            let real = score(d, lead, lt) + score(d, &others[0], ot);
                            prop_assert!(
                                real <= ub + BOUND_EPS,
                                "seed {seed} k1 {k1} b {b}: doc {d} scores {real} above the bound {ub} \
                                 declared at leader doc {doc} for window ..={window_end}"
                            );
                        }
                    }
                    lead.next();
                }
                Ok(())
            })
            .expect("bound property holds");
    }

    /// The two-term WAND tail returns exactly the top-k of `strong`'s
    /// docs by `strong + weak` score among those above the seeded
    /// threshold, ties broken by ascending doc id. It prunes with
    /// `weak`'s block bound at each candidate, so a bound below a real
    /// `weak` score would drop a qualifying doc from the heap: the
    /// reference here is a brute-force rescore of every candidate, so
    /// such a drop fails as a set mismatch.
    #[test]
    fn prop_wand_two_term_tail_matches_a_brute_force_top_k() {
        use proptest::{
            prelude::*,
            test_runner::{Config, TestRunner},
        };
        let strategy = (any::<u64>(), 0.3f32..2.5, 0.0f32..=1.0, 1usize..=25);
        let mut runner = TestRunner::new(Config {
            cases: BOUND_CASES,
            ..Config::default()
        });
        runner
            .run(&strategy, |(seed, k1, b, k)| {
                let (r, lead_tf, other_tf) = bound_reader(seed, k1, b);
                let col = r.resolve_column_id("body").expect("col");
                let norms = &r.columns[col as usize].dl_norm_k1();
                let rt = tokio::runtime::Builder::new_current_thread()
                    .build()
                    .expect("runtime");
                let mut cursors = rt
                    .block_on(r.build_term_cursors(
                        col,
                        &["lead", "other"],
                        QueryIdf::ThisFile,
                        CursorUse::Score,
                        None,
                        None,
                    ))
                    .expect("cursors");
                prop_assume!(cursors.len() == 2);
                // The tail's contract names the weaker term by its maximum.
                let lead_is_weak = cursors[0].term_max_bm25 < cursors[1].term_max_bm25;
                let (weak_tfs, strong_tfs) = if lead_is_weak {
                    (&lead_tf, &other_tf)
                } else {
                    (&other_tf, &lead_tf)
                };
                let (a, bb) = cursors.split_at_mut(1);
                let (weak, strong) = if lead_is_weak {
                    (&mut a[0], &mut bb[0])
                } else {
                    (&mut bb[0], &mut a[0])
                };
                let seeded = weak.term_max_bm25;
                let (weak_idf, strong_idf) = (weak.idf_weight, strong.idf_weight);
                let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
                let mut threshold = seeded;
                wand_two_term_tail(weak, strong, k, &mut heap, &mut threshold, norms);
                let got = drain_top_k_desc(heap);

                // Brute force over strong's docs, above the seeded threshold.
                let mut want: Vec<(u32, f32)> = (0..BOUND_DOCS)
                    .filter_map(|d| {
                        let st = strong_tfs[d as usize]?;
                        let norm = norms.get(d);
                        let mut s = bm25::score_with_dl_norm_k1(strong_idf, st, norm);
                        if let Some(wt) = weak_tfs[d as usize] {
                            s += bm25::score_with_dl_norm_k1(weak_idf, wt, norm);
                        }
                        (s > seeded).then_some((d, s))
                    })
                    .collect();
                want.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
                want.truncate(k);
                prop_assert_eq!(got.len(), want.len(), "seed {} k {}: hit count", seed, k);
                for ((gd, gs), (wd, ws)) in got.iter().zip(&want) {
                    prop_assert!(
                        (gs - ws).abs() <= BOUND_EPS,
                        "seed {seed} k {k}: score {gs} vs {ws} at docs {gd}/{wd}"
                    );
                    // Equal scores may legitimately swap docs only within a tie.
                    if (gs - ws).abs() > BOUND_EPS / 10.0 {
                        prop_assert_eq!(gd, wd, "seed {} k {}: doc order", seed, k);
                    }
                }
                Ok(())
            })
            .expect("tail property holds");
    }

    /// Every count plan against set arithmetic on the corpus: both ways of
    /// counting an intersection, both ways of counting with exclusions (all
    /// and any), the planned choice between them, and the two-term union by
    /// subtraction. Corpora mix a near-every-doc term (bitset blocks), mid and
    /// rare terms (packed, patched, short form) and single-doc terms (inline),
    /// so each kernel meets each block form.
    #[tokio::test]
    async fn count_plans_agree_with_set_arithmetic() {
        let mut rng = StdRng::seed_from_u64(0xC0_FFEE);
        let vocab = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"];
        for trial in 0..8 {
            let n_docs = 300 + rng.random_range(0..6_000usize);
            let odds: Vec<u32> = vocab
                .iter()
                .enumerate()
                .map(|(t, _)| match t {
                    0 => 950,
                    1 => 500,
                    _ => rng.random_range(1..300),
                })
                .collect();
            let docs: Vec<String> = (0..n_docs)
                .map(|d| {
                    let mut text = format!("doc{d}");
                    for (t, term) in vocab.iter().enumerate() {
                        if rng.random_range(0..1000) < odds[t] {
                            text.push(' ');
                            text.push_str(term);
                        }
                    }
                    text
                })
                .collect();
            let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            let (blob, json) = build_standard_blob(&refs);
            let r = FtsReader::open(blob, &json).expect("open");
            let holds = |term: &str| -> Vec<bool> {
                docs.iter()
                    .map(|d| d.split(' ').any(|w| w == term))
                    .collect()
            };
            let cursors = |terms: Vec<&'static str>| {
                let r = &r;
                async move {
                    r.build_term_cursors(
                        0,
                        &terms,
                        QueryIdf::ThisFile,
                        CursorUse::Count,
                        None,
                        None,
                    )
                    .await
                    .expect("cursors")
                }
            };
            for _ in 0..12 {
                let n_pos = rng.random_range(1..4usize);
                let n_neg = rng.random_range(0..3usize);
                let mut picked: Vec<&'static str> = Vec::new();
                while picked.len() < n_pos + n_neg {
                    let t = vocab[rng.random_range(0..vocab.len())];
                    if !picked.contains(&t) {
                        picked.push(t);
                    }
                }
                let (pos, neg) = picked.split_at(n_pos);
                let (pos, neg) = (pos.to_vec(), neg.to_vec());
                let pos_sets: Vec<Vec<bool>> = pos.iter().map(|t| holds(t)).collect();
                let neg_sets: Vec<Vec<bool>> = neg.iter().map(|t| holds(t)).collect();
                for all in [true, false] {
                    let want = (0..n_docs)
                        .filter(|&d| {
                            let p = match all {
                                true => pos_sets.iter().all(|s| s[d]),
                                false => pos_sets.iter().any(|s| s[d]),
                            };
                            p && !neg_sets.iter().any(|s| s[d])
                        })
                        .count() as u64;
                    let what = format!("trial {trial} pos {pos:?} neg {neg:?} all={all}");
                    let max_doc = n_docs as u32;
                    let fold = count_excluding_fold(
                        cursors(pos.clone()).await,
                        all,
                        cursors(neg.clone()).await,
                        max_doc,
                    );
                    assert_eq!(fold, want, "fold: {what}");
                    let probe = count_excluding_probe(
                        cursors(pos.clone()).await,
                        all,
                        cursors(neg.clone()).await,
                    );
                    assert_eq!(probe, want, "probe: {what}");
                    let planned = count_excluding(
                        cursors(pos.clone()).await,
                        all,
                        cursors(neg.clone()).await,
                    );
                    assert_eq!(planned, want, "planned: {what}");
                    if neg.is_empty() && pos.len() >= 2 {
                        let planned = match all {
                            true => count_and_planned(cursors(pos.clone()).await),
                            false => or_count_unranked(cursors(pos.clone()).await),
                        };
                        assert_eq!(planned, want, "no exclusions: {what}");
                    }
                }
            }
        }
    }

    /// A reader keeps long terms' block tables for reuse. A reused reader
    /// must answer exactly as a fresh one, including through a view with
    /// other BM25 parameters, whose bounds decode at another scale from the
    /// same term offsets. 40K documents put the common term past the
    /// cache's minimum table length.
    #[tokio::test]
    async fn reused_block_tables_answer_as_fresh_ones() {
        let mut x: u64 = 0x1234_5678_9ABC_DEF1;
        // Enough docs that the common term's block table is long enough
        // to be cached.
        let docs: Vec<String> = (0..80_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let mut t = String::from("doc");
                if x % 100 < 90 {
                    t.push_str(" common");
                }
                if x % 100 < 6 {
                    t.push_str(" rare");
                }
                if (x >> 8).is_multiple_of(7) {
                    t.push_str(" common common filler");
                }
                t
            })
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let (blob, json) = build_standard_blob(&refs);
        let open = || FtsReader::open(blob.clone(), &json).expect("open");
        let reused = open();
        let params = bm25::Bm25Params::new(0.9, 0.4);
        for k in [1usize, 10, 100] {
            for terms in [
                vec!["common", "rare"],
                vec!["common", "doc"],
                vec!["common"],
            ] {
                for _ in 0..2 {
                    let got = reused
                        .search("body", &terms, k, BoolMode::Or)
                        .await
                        .expect("reused");
                    let want = open()
                        .search("body", &terms, k, BoolMode::Or)
                        .await
                        .expect("fresh");
                    assert_eq!(got, want, "{terms:?} k={k}");
                    let got = reused
                        .with_bm25_override(params)
                        .search("body", &terms, k, BoolMode::Or)
                        .await
                        .expect("reused override");
                    let want = open()
                        .with_bm25_override(params)
                        .search("body", &terms, k, BoolMode::Or)
                        .await
                        .expect("fresh override");
                    assert_eq!(got, want, "override {terms:?} k={k}");
                }
            }
        }
        assert!(
            reused.block_tables.cached_blocks() > 0,
            "the common term's table was cached"
        );
    }
}
