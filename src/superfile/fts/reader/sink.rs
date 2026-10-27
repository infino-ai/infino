// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Top-k collection: the reversed-order [`TopKEntry`] heap element and
//! [`drain_top_k_desc`], plus the [`AndSink`] family parameterizing the
//! AND flat-merge over score-into-heap vs collect/count, and the shared
//! `and_heap_push`. `pub(super)` within `reader/`.

use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::atomic::{AtomicU32, Ordering as AtomicOrdering},
};

use super::{
    cursor::TermCursor, filter::ExcludeFilter, metadata::NormTable, scorers::score_noness_batched,
};
use crate::superfile::id_space::FtsDocId;

/// A monotonically rising score floor shared by every segment of one
/// query's fan-out, readable and raisable lock-free mid-walk.
///
/// The kth-best score of any one segment is a lower bound on the
/// kth-best of the whole query (the merged candidate set is a
/// superset), so each walk may publish its local kth as it improves
/// and every other walk may prune against the highest published value
/// — the same cross-segment floor the fan-out already applies at
/// segment start, made live. Pruning stays exact: walks skip only
/// work strictly below the floor, so score ties still surface and the
/// merged top-k is unchanged; only the amount of skipped work depends
/// on timing.
pub(crate) struct LiveFloor(AtomicU32);

impl LiveFloor {
    pub(crate) fn new() -> Self {
        Self(AtomicU32::new(f32::NEG_INFINITY.to_bits()))
    }

    /// The highest floor published so far (`NEG_INFINITY` until some
    /// segment has k scores).
    #[inline]
    pub(crate) fn load(&self) -> f32 {
        f32::from_bits(self.0.load(AtomicOrdering::Acquire))
    }

    /// Raise the floor to `candidate` if it exceeds the current value;
    /// lower values are ignored, so the floor is monotone under any
    /// interleaving of publishers.
    pub(crate) fn raise(&self, candidate: f32) {
        let mut current = self.0.load(AtomicOrdering::Relaxed);
        while candidate.total_cmp(&f32::from_bits(current)) == Ordering::Greater {
            match self.0.compare_exchange_weak(
                current,
                candidate.to_bits(),
                AtomicOrdering::Release,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return,
                Err(seen) => current = seen,
            }
        }
    }
}

/// Top-k min-heap entry `(score, doc_id)`, shared by every search
/// path (single-term BMW, WAND+BMW, MaxScore+BMM, exhaustive union,
/// AND intersection, and the `search_multi` combiner).
///
/// Ordering is **reversed** on purpose: smaller score is "greater",
/// so `BinaryHeap::peek()` returns the smallest-score entry. Once the
/// heap holds k entries, `peek()` is the current kth-best score — the
/// bar a new doc must beat (also the BMW/BMM pruning threshold).
/// Tie-break: larger doc_id is "greater", so on equal scores the
/// smaller doc_id survives in the heap.
#[derive(Debug, Copy, Clone)]
pub(in crate::superfile::fts) struct TopKEntry<I = FtsDocId>(
    pub(in crate::superfile::fts) f32,
    pub(in crate::superfile::fts) I,
);
/// An integer that orders as the score does: the sign bit flipped for a
/// non-negative score, every bit for a negative one. The top-k heaps compare
/// these rather than floats, a single integer compare per heap step; a
/// score is never NaN.
#[inline(always)]
fn score_key(score: f32) -> u32 {
    let bits = score.to_bits();
    match bits >> 31 {
        0 => bits | 0x8000_0000,
        _ => !bits,
    }
}

/// The score whose [`score_key`] is `key`.
#[inline(always)]
fn score_of_key(key: u32) -> f32 {
    f32::from_bits(match key >> 31 {
        1 => key & 0x7FFF_FFFF,
        _ => !key,
    })
}

/// `(score, doc)` packed so a larger integer is a better entry: the score's
/// key above, the doc id inverted below so that on a tied score the smaller
/// doc is the better one. Selecting on these is one integer compare a step.
#[inline(always)]
fn rank_key(score: f32, doc: u32) -> u64 {
    (u64::from(score_key(score)) << 32) | u64::from(!doc)
}

/// The `(score, doc)` a [`rank_key`] packs.
#[inline(always)]
fn unrank_key(key: u64) -> (f32, u32) {
    (score_of_key((key >> 32) as u32), !(key as u32))
}

impl<I: Eq> PartialEq for TopKEntry<I> {
    fn eq(&self, other: &Self) -> bool {
        score_key(self.0) == score_key(other.0) && self.1 == other.1
    }
}
impl<I: Eq> Eq for TopKEntry<I> {}
impl<I: Ord> PartialOrd for TopKEntry<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<I: Ord> Ord for TopKEntry<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Score is inverted (lower score = greater) so the max-heap's
        // peek is the worst kept entry; the doc-id leg must NOT be
        // inverted, so that among score-tied entries the LARGER doc id
        // is greater — i.e. peek — and is the one evicted when a
        // better doc arrives, keeping the smaller id.
        score_key(other.0)
            .cmp(&score_key(self.0))
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// What a `depth + 1` heap kept, as a term's head. `kept` receives the
/// documents in descending score, ascending id on a tie — a reader's
/// top-k order. The bound is the largest key among them; when more than
/// `depth` were kept, the last is popped as the tail, the first document
/// left out by its key, else the tail is zero. Returns `(bound, tail)`.
pub(in crate::superfile::fts) fn finish_head<I: Ord + Copy>(
    heap: &mut BinaryHeap<TopKEntry<I>>,
    depth: usize,
    kept: &mut Vec<(I, f32)>,
    key: impl Fn(I) -> f32,
) -> (f32, f32) {
    let complete = heap.len() <= depth;
    kept.clear();
    kept.extend(heap.drain().map(|TopKEntry(s, d)| (d, s)));
    kept.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let bound = kept.iter().map(|&(d, _)| key(d)).fold(0.0f32, f32::max);
    let tail = if complete {
        0.0
    } else {
        kept.pop().map(|(d, _)| key(d)).unwrap_or(0.0)
    };
    (bound, tail)
}

/// Drain a top-k min-heap into the public result order: descending
/// score, ascending doc_id on ties.
///
/// pdqsort: entries are unique by `(score, doc_id)` — every search
/// path offers each doc_id to its heap at most once — so an unstable
/// sort has no observable reorderings.
pub(super) fn drain_top_k_desc<I: Ord + Copy>(heap: BinaryHeap<TopKEntry<I>>) -> Vec<(I, f32)> {
    let mut out: Vec<(I, f32)> = heap.into_iter().map(|TopKEntry(s, d)| (d, s)).collect();
    // `total_cmp` orders a score that is never NaN exactly as `partial_cmp`
    // does, with an integer compare instead of an `Option` per step.
    out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Per-hit action for the multi-term AND flat-merge intersection.
///
/// The traversal in [`FtsReader::and_flat_merge_general`] /
/// [`FtsReader::and_flat_merge_2term`] — cursor alignment, block
/// crossing, and the in-block pointer walk — runs identically whether
/// the caller wants ranked hits or just the matching doc ids. Only the
/// action at each converged doc differs, so both go through one
/// traversal parameterized by this trait and cannot disagree on which
/// docs match.
///
/// [`ScoreSink`] computes BM25 and feeds a top-k heap (the ranked
/// search path); [`CollectSink`] records the doc id and computes no
/// score (the unranked `token_match` / count path). The traversal is
/// monomorphized per sink, so `needs_score()` folds to a constant: the
/// scorer compiles to a dedicated copy with scoring inlined, and the
/// collector's copy drops the scoring arithmetic as dead code.
pub(super) trait AndSink {
    /// Block-max pruning bar: docs whose block can't reach this score
    /// are skipped. Returning `NEG_INFINITY` (the default) disables
    /// pruning, which is what an unranked sink wants — it has no score
    /// threshold to prune against.
    fn bar(&self) -> f32 {
        f32::NEG_INFINITY
    }

    /// Whether the traversal should compute a hit's BM25 score. A sink
    /// that returns `false` skips all scoring arithmetic — what makes an
    /// unranked count over a large intersection cheaper than ranking it.
    fn needs_score(&self) -> bool;

    /// How many terms the sink adds to a doc's score beyond the walk's own,
    /// so the walk's pruning bar allows for summing them in another order
    /// (see `pruning_bar`).
    fn added_terms(&self) -> usize {
        0
    }

    /// Record one doc in the intersection. `score` is meaningful only
    /// when [`needs_score`](AndSink::needs_score) returns `true`;
    /// otherwise it is `0.0` and ignored.
    fn emit(&mut self, doc: u32, score: f32);
}

/// Ranked sink: floor-gates each hit and pushes it into the top-k heap.
pub(super) struct ScoreSink<'a> {
    pub(super) heap: &'a mut BinaryHeap<TopKEntry>,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) floor_eff: f32,
}

impl AndSink for ScoreSink<'_> {
    fn bar(&self) -> f32 {
        // kth-best once the heap fills, else the caller's seeded floor —
        // whichever is higher.
        if self.heap.len() >= self.k {
            self.heap
                .peek()
                .expect("heap len == k")
                .0
                .max(self.floor_eff)
        } else {
            self.floor_eff
        }
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, score: f32) {
        // Floor gate: strictly-below-floor docs are dead to the caller.
        if score > self.floor_eff {
            and_heap_push(self.heap, self.k, self.filter.as_deref_mut(), score, doc);
        }
    }
}

/// Ranked must+should sink: scores the must intersection like
/// [`ScoreSink`], then adds each should term's contribution for docs
/// it lands on before heap admission. The should cursors ride inside
/// the sink — the AND walk emits docs in ascending order, so each
/// should list is `skip_to`-streamed forward at most once per query,
/// exactly like [`ExcludeFilter`]'s negated cursors.
pub(super) struct MustShouldSink<'a, C: Collector> {
    pub(super) heap: &'a mut C,
    /// The collector's k-th best: exact in a heap, at its last cut in a
    /// buffer (never above the true one).
    pub(super) kth: f32,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) floor_eff: f32,
    pub(super) shoulds: ShouldBatch,
    /// Σ `term_max_bm25` over the should cursors — the most the
    /// shoulds can add to any single doc's score.
    pub(super) should_ub: f32,
    /// Per-doc BM25 length normalization for the column, for scoring
    /// the should terms at emitted docs.
    pub(super) dl_norm_k1: &'a NormTable,
}

impl<C: Collector> AndSink for MustShouldSink<'_, C> {
    fn added_terms(&self) -> usize {
        self.shoulds.len()
    }

    fn bar(&self) -> f32 {
        // The AND walk's block-max arithmetic bounds the MUST portion
        // of a doc's score only, so the pruning bar is lowered by the
        // most the shoulds could add: a must block that can't reach
        // (kth-best − should_ub) can't produce a top-k doc even with
        // every should matching at its maximum.
        let full_bar = match self.heap.is_full(self.k) {
            true => self.kth.max(self.floor_eff),
            false => self.floor_eff,
        };
        full_bar - self.should_ub
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, must_score: f32) {
        if self.shoulds.push(doc, must_score) {
            self.flush();
        }
    }
}

impl<C: Collector> MustShouldSink<'_, C> {
    /// Score the held docs' shoulds and offer them to the heap. The walk
    /// calls this once more after its last doc.
    pub(super) fn flush(&mut self) {
        let (docs, scores) = self.shoulds.score(self.dl_norm_k1);
        for (&doc, &score) in docs.iter().zip(scores) {
            // Floor gate on the FULL score — a must-only score below the
            // floor can still survive once its shoulds are added.
            if score <= self.floor_eff || (self.heap.is_full(self.k) && score <= self.kth) {
                continue;
            }
            // The negation is asked only of a doc that would be admitted, in
            // ascending doc order.
            if let Some(f) = self.filter.as_deref_mut()
                && !f.admits(doc)
            {
                continue;
            }
            self.heap.offer(self.k, &mut self.kth, score, doc);
        }
        self.shoulds.clear();
    }
}

/// How many must-intersection docs wait for their should scores.
const SHOULD_BATCH: usize = 128;

/// Must-intersection docs held until a batch is full, so each should term
/// is probed once for the batch ([`TermCursor::probe_tfs`]) rather than
/// stepped to every doc. Each doc adds its should scores in cursor order,
/// so the ranked and the counted must+should sinks produce the same bits.
/// The should cursors are only probed, never iterated.
pub(super) struct ShouldBatch {
    shoulds: Vec<TermCursor>,
    docs: Vec<u32>,
    scores: Vec<f32>,
    tfs: Vec<u32>,
}

impl ShouldBatch {
    /// The number of should terms.
    pub(super) fn len(&self) -> usize {
        self.shoulds.len()
    }

    pub(super) fn new(shoulds: Vec<TermCursor>) -> Self {
        Self {
            shoulds,
            docs: Vec::with_capacity(SHOULD_BATCH),
            scores: Vec::with_capacity(SHOULD_BATCH),
            tfs: Vec::with_capacity(SHOULD_BATCH),
        }
    }

    /// Hold `doc` with its must score; `true` once the batch is full.
    #[inline]
    fn push(&mut self, doc: u32, must_score: f32) -> bool {
        self.docs.push(doc);
        self.scores.push(must_score);
        self.docs.len() == SHOULD_BATCH
    }

    /// Add every should term's BM25 to each held doc's score; the held
    /// docs with their full scores, in ascending doc order.
    fn score(&mut self, dl_norm_k1: &NormTable) -> (&[u32], &[f32]) {
        for c in &mut self.shoulds {
            score_noness_batched(c, &self.docs, &mut self.scores, dl_norm_k1, &mut self.tfs);
        }
        (&self.docs, &self.scores)
    }

    fn clear(&mut self) {
        self.docs.clear();
        self.scores.clear();
    }
}

/// Ranked sink that scores every intersection doc and counts them. The
/// bar stays at `NEG_INFINITY`, so the walk prunes nothing; a negated
/// doc is neither scored nor counted.
pub(super) struct ScoreCountSink<'a> {
    pub(super) heap: &'a mut BinaryHeap<TopKEntry>,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) count: u64,
}

impl AndSink for ScoreCountSink<'_> {
    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, score: f32) {
        if let Some(f) = self.filter.as_deref_mut()
            && !f.admits(doc)
        {
            return;
        }
        self.count += 1;
        and_heap_push(self.heap, self.k, None, score, doc);
    }
}

/// [`ScoreSink`] collecting into a [`TopKBuffer`]: the same floor gate and
/// filter, an append per admitted hit instead of a sift, and a bar that rises
/// at each cut. For a large `k`, where most hits are admitted.
pub(super) struct BufferedScoreSink<'a> {
    pub(super) top: TopKBuffer,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) floor_eff: f32,
    /// The k-th best at the last cut, never below the floor.
    pub(super) kth: f32,
}

impl AndSink for BufferedScoreSink<'_> {
    fn bar(&self) -> f32 {
        match self.top.is_full(self.k) {
            true => self.kth.max(self.floor_eff),
            false => self.floor_eff,
        }
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, score: f32) {
        if score <= self.floor_eff {
            return;
        }
        if let Some(f) = self.filter.as_deref_mut()
            && !f.admits(doc)
        {
            return;
        }
        self.top.offer(self.k, &mut self.kth, score, doc);
    }
}

/// [`MustShouldSink`] without pruning: every must-intersection doc is
/// scored and counted, and the shoulds only add to its score.
pub(super) struct MustShouldCountSink<'a> {
    pub(super) heap: &'a mut BinaryHeap<TopKEntry>,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) shoulds: ShouldBatch,
    pub(super) dl_norm_k1: &'a NormTable,
    pub(super) count: u64,
}

impl AndSink for MustShouldCountSink<'_> {
    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, must_score: f32) {
        if let Some(f) = self.filter.as_deref_mut()
            && !f.admits(doc)
        {
            return;
        }
        self.count += 1;
        if self.shoulds.push(doc, must_score) {
            self.flush();
        }
    }
}

impl MustShouldCountSink<'_> {
    /// Score the held docs' shoulds and offer them to the heap. The walk
    /// calls this once more after its last doc.
    pub(super) fn flush(&mut self) {
        let (docs, scores) = self.shoulds.score(self.dl_norm_k1);
        for (&doc, &score) in docs.iter().zip(scores) {
            and_heap_push(self.heap, self.k, None, score, doc);
        }
        self.shoulds.clear();
    }
}

/// Unranked sink: collect the matching doc ids in ascending order, no
/// scoring, no top-k. Drives the `token_match` AND path through the
/// same optimized flat-merge the scorer uses.
pub(super) struct CollectSink {
    pub(super) out: Vec<FtsDocId>,
}

impl AndSink for CollectSink {
    fn needs_score(&self) -> bool {
        false
    }

    fn emit(&mut self, doc: u32, _score: f32) {
        self.out.push(FtsDocId::new(doc));
    }
}

/// Push `(score, doc_id)` into the top-k AND heap with the same
/// tie-break (asc doc_id) the OR paths use, so AND and OR rankings
/// agree on score-tied docs.
///
/// `filter` drops docs excluded by a negated (`-term`) clause before
/// they enter the heap; `None` admits everything.
#[inline]
pub(super) fn and_heap_push(
    heap: &mut BinaryHeap<TopKEntry>,
    k: usize,
    filter: Option<&mut ExcludeFilter>,
    score: f32,
    doc_id: u32,
) {
    let full = heap.len() >= k;
    if full
        && heap
            .peek()
            .is_some_and(|w| !(score > w.0 || (score == w.0 && doc_id < w.1.get())))
    {
        return;
    }
    // The negation is asked only of a doc that would enter the heap: one that
    // would not is dropped either way, and the docs asked still ascend.
    if let Some(f) = filter
        && !f.admits(doc_id)
    {
        return;
    }
    if !full {
        heap.push(TopKEntry(score, FtsDocId::new(doc_id)));
    } else if let Some(mut worst) = heap.peek_mut() {
        *worst = TopKEntry(score, FtsDocId::new(doc_id));
    }
}

/// Replace the heap's worst entry in place — one sift, where a `pop`
/// followed by a `push` costs a full sift-down and two sift-ups. The
/// caller has already decided `entry` belongs in the top-k and the heap
/// is non-empty (it holds `k`); unlike pop-then-push this does not insert
/// into an empty heap, so that precondition is asserted.
#[inline]
pub(super) fn replace_worst(heap: &mut BinaryHeap<TopKEntry>, entry: TopKEntry) {
    debug_assert!(!heap.is_empty(), "replace_worst on an empty heap");
    if let Some(mut worst) = heap.peek_mut() {
        *worst = entry;
    }
}

/// Offer `(score, doc)` to a top-`k` heap whose admission bar is
/// `threshold`: pushed while the heap fills, and once it is full kept only
/// when it beats the bar, in place of the worst entry. Whenever the heap is
/// full after the offer the bar becomes its worst score, but never drops
/// below where it was, so a floor the caller seeded is kept. Returns whether
/// the bar was set or rose, for callers that re-plan on a higher bar. A
/// `k` of zero admits nothing.
///
/// Most offers lose to a full heap's bar, so that test is inlined into every
/// caller and only an admission pays for a call.
#[inline(always)]
pub(super) fn offer_top_k(
    heap: &mut BinaryHeap<TopKEntry>,
    k: usize,
    threshold: &mut f32,
    score: f32,
    doc: u32,
) -> bool {
    if heap.len() >= k && score <= *threshold {
        return false;
    }
    admit_top_k(heap, k, threshold, score, doc)
}

/// The admitting half of [`offer_top_k`], out of line.
fn admit_top_k(
    heap: &mut BinaryHeap<TopKEntry>,
    k: usize,
    threshold: &mut f32,
    score: f32,
    doc: u32,
) -> bool {
    if k == 0 {
        return false;
    }
    if heap.len() < k {
        heap.push(TopKEntry(score, FtsDocId::new(doc)));
        if heap.len() < k {
            return false;
        }
    } else {
        debug_assert!(score > *threshold, "the caller rejected a score at the bar");
        replace_worst(heap, TopKEntry(score, FtsDocId::new(doc)));
    }
    *threshold = heap.peek().expect("a full heap").0.max(*threshold);
    true
}

/// A top-`k` collector a ranked kernel is generic over: the heap, which
/// raises the bar on every admission, or the buffer, which raises it in
/// steps and costs no sift per admission.
pub(super) trait Collector: Sized {
    fn with_capacity(k: usize, capacity: usize) -> Self;
    /// Whether `k` entries have been admitted, so the bar is a real k-th best.
    fn is_full(&self, k: usize) -> bool;
    /// Offer `(score, doc)`; see [`offer_top_k`] for the admission rule. Returns
    /// whether the bar was set or rose.
    fn offer(&mut self, k: usize, threshold: &mut f32, score: f32, doc: u32) -> bool;
    /// The best `k`, by descending score and ascending doc on a tie.
    fn into_sorted(self) -> Vec<(FtsDocId, f32)>;
}

impl Collector for BinaryHeap<TopKEntry> {
    fn with_capacity(_k: usize, capacity: usize) -> Self {
        BinaryHeap::with_capacity(capacity)
    }
    #[inline]
    fn is_full(&self, k: usize) -> bool {
        self.len() >= k
    }
    #[inline]
    fn offer(&mut self, k: usize, threshold: &mut f32, score: f32, doc: u32) -> bool {
        offer_top_k(self, k, threshold, score, doc)
    }
    fn into_sorted(self) -> Vec<(FtsDocId, f32)> {
        drain_top_k_desc(self)
    }
}

/// The best `k` entries offered, kept in an unordered buffer cut back to the
/// best `k` by selection whenever it outgrows them by [`TopKBuffer::slack`].
/// A heap pays a `log k`-step sift for every admission, which at a large `k`
/// with a low bar is most of them; the buffer pays an append, and a linear
/// selection per `slack` appends. Its bar is the k-th best at the last cut:
/// never above the true k-th best, so pruning against it stays safe, and
/// refreshed often enough (`slack` is `k / 2`) that it trails closely.
/// Entries are held as [`rank_key`]s.
pub(super) struct TopKBuffer {
    k: usize,
    buf: Vec<u64>,
    full: bool,
}

impl TopKBuffer {
    /// Appends between cuts once full.
    fn slack(k: usize) -> usize {
        (k / TOPK_BUFFER_SLACK_DIVISOR).max(1)
    }

    /// Keep the best `k` and return the k-th best's score.
    fn cut(&mut self, k: usize) -> f32 {
        if self.buf.len() > k {
            self.buf.select_nth_unstable_by(k - 1, |a, b| b.cmp(a));
            self.buf.truncate(k);
        }
        unrank_key(*self.buf.iter().min().expect("k > 0 entries")).0
    }
}

/// [`TopKBuffer`] cuts once it holds `k + k / this` entries.
const TOPK_BUFFER_SLACK_DIVISOR: usize = 2;

impl Collector for TopKBuffer {
    fn with_capacity(k: usize, capacity: usize) -> Self {
        Self {
            k,
            // `k` may stand for "every hit" (`usize::MAX`): size from the
            // caller's bounded estimate, never from `k` itself.
            buf: Vec::with_capacity(capacity),
            full: false,
        }
    }
    #[inline]
    fn is_full(&self, _k: usize) -> bool {
        self.full
    }
    #[inline]
    fn offer(&mut self, k: usize, threshold: &mut f32, score: f32, doc: u32) -> bool {
        // A score equal to the bar is kept: the bar may trail the true k-th
        // best, and a tie there with a smaller doc wins on the doc-id leg,
        // which the final selection decides.
        if k == 0 || (self.full && score < *threshold) {
            return false;
        }
        self.buf.push(rank_key(score, doc));
        let due = match self.full {
            false => self.buf.len() == k,
            true => self.buf.len() >= k.saturating_add(Self::slack(k)),
        };
        if !due {
            return false;
        }
        self.full = true;
        *threshold = self.cut(k).max(*threshold);
        true
    }
    fn into_sorted(mut self) -> Vec<(FtsDocId, f32)> {
        self.buf.sort_unstable_by(|a, b| b.cmp(a));
        self.buf.truncate(self.k);
        self.buf
            .into_iter()
            .map(|key| {
                let (score, doc) = unrank_key(key);
                (FtsDocId::new(doc), score)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_keys_round_trip_and_order_as_the_entries_do() {
        let scores = [
            0.0f32,
            -0.0,
            1e-30,
            0.5,
            1.0,
            7.25,
            1e30,
            -1.0,
            -1e-30,
            f32::INFINITY,
        ];
        let docs = [0u32, 1, 7, u32::MAX - 1, u32::MAX];
        for &s in &scores {
            for &d in &docs {
                let (s2, d2) = unrank_key(rank_key(s, d));
                assert_eq!((s2.to_bits(), d2), (s.to_bits(), d));
                for &t in &scores {
                    for &e in &docs {
                        // A larger key is a better entry: the smaller in
                        // `TopKEntry`'s reversed order.
                        let by_key = rank_key(t, e).cmp(&rank_key(s, d));
                        let by_entry = TopKEntry(s, d).cmp(&TopKEntry(t, e));
                        assert_eq!(by_key, by_entry, "({s}, {d}) vs ({t}, {e})");
                    }
                }
            }
        }
    }

    #[test]
    fn and_heap_push_keeps_the_smaller_doc_on_a_tied_score() {
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        and_heap_push(&mut heap, 2, None, 1.0, 5);
        and_heap_push(&mut heap, 2, None, 1.0, 9);
        // Same score, smaller doc: evicts doc 9.
        and_heap_push(&mut heap, 2, None, 1.0, 7);
        assert_eq!(
            drain_top_k_desc(heap.clone()),
            vec![(FtsDocId::new(5), 1.0), (FtsDocId::new(7), 1.0)]
        );
        // Same score, larger doc: no change.
        and_heap_push(&mut heap, 2, None, 1.0, 12);
        assert_eq!(
            drain_top_k_desc(heap.clone()),
            vec![(FtsDocId::new(5), 1.0), (FtsDocId::new(7), 1.0)]
        );
        // Better score: evicts the worst (doc 7, the larger of the tie).
        and_heap_push(&mut heap, 2, None, 2.0, 30);
        assert_eq!(
            drain_top_k_desc(heap.clone()),
            vec![(FtsDocId::new(30), 2.0), (FtsDocId::new(5), 1.0)]
        );
        // Worse score than the worst: no change.
        and_heap_push(&mut heap, 2, None, 0.5, 1);
        assert_eq!(
            drain_top_k_desc(heap),
            vec![(FtsDocId::new(30), 2.0), (FtsDocId::new(5), 1.0)]
        );
    }

    #[test]
    fn offer_top_k_fills_then_keeps_the_best_and_never_lowers_a_seeded_bar() {
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        let mut bar = 0.0f32;
        assert!(
            !offer_top_k(&mut heap, 2, &mut bar, 1.0, 1),
            "still filling"
        );
        assert!(
            offer_top_k(&mut heap, 2, &mut bar, 3.0, 2),
            "full: the bar is set"
        );
        assert_eq!(bar, 1.0);
        assert!(
            !offer_top_k(&mut heap, 2, &mut bar, 0.5, 3),
            "under the bar"
        );
        assert!(offer_top_k(&mut heap, 2, &mut bar, 2.0, 4), "beats the bar");
        assert_eq!(bar, 2.0);
        let mut docs: Vec<u32> = heap.iter().map(|e| e.1.get()).collect();
        docs.sort_unstable();
        assert_eq!(docs, [2, 4]);
        // A seeded floor above the heap's worst score is kept.
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        let mut bar = 5.0f32;
        assert!(offer_top_k(&mut heap, 1, &mut bar, 1.0, 1));
        assert_eq!(bar, 5.0);
        assert!(
            !offer_top_k(&mut heap, 0, &mut bar, 9.0, 2),
            "k = 0 admits nothing"
        );
    }

    #[test]
    fn score_key_orders_exactly_as_total_cmp() {
        let values = [
            0.0f32,
            -0.0,
            f32::MIN_POSITIVE,
            1e-30,
            0.25,
            1.0,
            1.0 + f32::EPSILON,
            3.5,
            7.25,
            1e30,
            f32::MAX,
            -1.0,
            -3.5,
            f32::MIN,
        ];
        for &a in &values {
            for &b in &values {
                assert_eq!(
                    score_key(a).cmp(&score_key(b)),
                    a.total_cmp(&b),
                    "score_key({a}) vs score_key({b})"
                );
            }
        }
        // Equal scores tie on the key, so the doc id decides: the heap keeps
        // the smaller one.
        let tied = |d: u32| TopKEntry(2.5f32, FtsDocId::new(d));
        let mut heap: BinaryHeap<TopKEntry> = [tied(9), tied(3)].into_iter().collect();
        assert_eq!(
            heap.peek().map(|e| e.1.get()),
            Some(9),
            "the larger id is evicted first"
        );
        heap.pop();
        assert_eq!(heap.peek().map(|e| e.1.get()), Some(3));
    }

    #[test]
    fn the_buffered_collector_keeps_what_the_heap_keeps() {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rng = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        // `usize::MAX` stands for "every hit" and must neither overflow nor cut.
        for k in [1usize, 2, 7, 64, 512, 1000, usize::MAX] {
            for floor in [0.0f32, 3.0] {
                let mut heap: BinaryHeap<TopKEntry> = Collector::with_capacity(k, 0);
                let mut buf = TopKBuffer::with_capacity(k, 0);
                let (mut bar_h, mut bar_b) = (floor, floor);
                // Ascending docs, as every kernel offers them, with heavy ties.
                for doc in 0..5 * k.min(1000) as u32 + 37 {
                    let score = rng(64) as f32 / 8.0;
                    heap.offer(k, &mut bar_h, score, doc);
                    buf.offer(k, &mut bar_b, score, doc);
                    assert!(bar_b <= bar_h, "the buffer's bar never passes the heap's");
                }
                let want = heap.into_sorted();
                let got = buf.into_sorted();
                // Entries under a seeded floor may sit in either (a floor only
                // prunes); the ones above it must agree exactly.
                let above = |v: &[(FtsDocId, f32)]| -> Vec<(u32, f32)> {
                    v.iter()
                        .filter(|e| e.1 > floor)
                        .map(|e| (e.0.get(), e.1))
                        .collect()
                };
                assert_eq!(above(&got), above(&want), "k={k} floor={floor}");
            }
        }
    }

    #[test]
    fn replace_worst_is_pop_then_push() {
        let mut a: BinaryHeap<TopKEntry> = [(1.0, 4), (2.0, 1), (3.0, 7)]
            .into_iter()
            .map(|(s, d)| TopKEntry(s, FtsDocId::new(d)))
            .collect();
        let mut b = a.clone();
        replace_worst(&mut a, TopKEntry(2.5, FtsDocId::new(9)));
        b.pop();
        b.push(TopKEntry(2.5, FtsDocId::new(9)));
        assert_eq!(drain_top_k_desc(a), drain_top_k_desc(b));
    }

    #[test]
    fn drain_top_k_desc_orders_descending_with_tiebreak() {
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        heap.push(TopKEntry(1.0, FtsDocId::new(4)));
        heap.push(TopKEntry(2.0, FtsDocId::new(1)));
        heap.push(TopKEntry(2.0, FtsDocId::new(0))); // tie with doc 1
        let out = drain_top_k_desc(heap);
        assert_eq!(
            out,
            vec![
                (FtsDocId::new(0), 2.0),
                (FtsDocId::new(1), 2.0),
                (FtsDocId::new(4), 1.0)
            ]
        );
    }
}
