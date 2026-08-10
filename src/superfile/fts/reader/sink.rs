// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Top-k collection: the reversed-order [`TopKEntry`] heap element and
//! [`drain_top_k_desc`], plus the [`AndSink`] family parameterizing the
//! AND flat-merge over score-into-heap vs collect/count, and the shared
//! `and_heap_push`. `pub(super)` within `reader/`.

use std::{cmp::Ordering, collections::BinaryHeap};

use super::{cursor::TermCursor, filter::ExcludeFilter, metadata::NormTable};
use crate::superfile::fts::bm25;

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
pub(super) struct TopKEntry(pub(super) f32, pub(super) u32);
impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for TopKEntry {}
impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Score is inverted (lower score = greater) so the max-heap's
        // peek is the worst kept entry; the doc-id leg must NOT be
        // inverted, so that among score-tied entries the LARGER doc id
        // is greater — i.e. peek — and is the one evicted when a
        // better doc arrives, keeping the smaller id.
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// Drain a top-k min-heap into the public result order: descending
/// score, ascending doc_id on ties.
///
/// pdqsort: entries are unique by `(score, doc_id)` — every search
/// path offers each doc_id to its heap at most once — so an unstable
/// sort has no observable reorderings.
pub(super) fn drain_top_k_desc(heap: BinaryHeap<TopKEntry>) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|TopKEntry(s, d)| (d, s)).collect();
    out.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
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
pub(super) struct MustShouldSink<'a> {
    pub(super) heap: &'a mut BinaryHeap<TopKEntry>,
    pub(super) k: usize,
    pub(super) filter: Option<&'a mut ExcludeFilter>,
    pub(super) floor_eff: f32,
    pub(super) shoulds: Vec<TermCursor>,
    /// Σ `term_max_bm25` over the should cursors — the most the
    /// shoulds can add to any single doc's score.
    pub(super) should_ub: f32,
    /// Per-doc BM25 length normalization for the column, for scoring
    /// the should terms at emitted docs.
    pub(super) dl_norm_k1: &'a NormTable,
}

impl AndSink for MustShouldSink<'_> {
    fn bar(&self) -> f32 {
        // The AND walk's block-max arithmetic bounds the MUST portion
        // of a doc's score only, so the pruning bar is lowered by the
        // most the shoulds could add: a must block that can't reach
        // (kth-best − should_ub) can't produce a top-k doc even with
        // every should matching at its maximum.
        let full_bar = if self.heap.len() >= self.k {
            self.heap
                .peek()
                .expect("heap len == k")
                .0
                .max(self.floor_eff)
        } else {
            self.floor_eff
        };
        full_bar - self.should_ub
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, must_score: f32) {
        let norm = self.dl_norm_k1.get(doc);
        let mut score = must_score;
        for c in &mut self.shoulds {
            c.skip_to(doc);
            if !c.is_exhausted() && c.current_doc_id() == doc {
                score += bm25::score_with_dl_norm_k1(c.idf_x_k1p1, c.current_tf(), norm);
            }
        }
        // Floor gate on the FULL score — a must-only score below the
        // floor can still survive once its shoulds are added.
        if score > self.floor_eff {
            and_heap_push(self.heap, self.k, self.filter.as_deref_mut(), score, doc);
        }
    }
}

/// Unranked sink: collect the matching doc ids in ascending order, no
/// scoring, no top-k. Drives the `token_match` AND path through the
/// same optimized flat-merge the scorer uses.
pub(super) struct CollectSink {
    pub(super) out: Vec<u32>,
}

impl AndSink for CollectSink {
    fn needs_score(&self) -> bool {
        false
    }

    fn emit(&mut self, doc: u32, _score: f32) {
        self.out.push(doc);
    }
}

/// Unranked counting sink: tally the intersection size without
/// materializing the ids. Drives the count path through the same
/// flat-merge as [`CollectSink`] but skips the `Vec<u32>` — for a
/// high-cardinality count that allocation (4 bytes/doc) is pure waste.
pub(super) struct CountSink {
    pub(super) n: u64,
}

impl AndSink for CountSink {
    fn needs_score(&self) -> bool {
        false
    }

    fn emit(&mut self, _doc: u32, _score: f32) {
        self.n += 1;
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
    if let Some(f) = filter
        && !f.admits(doc_id)
    {
        return;
    }
    if heap.len() < k {
        heap.push(TopKEntry(score, doc_id));
    } else if let Some(&worst) = heap.peek()
        && (score > worst.0 || (score == worst.0 && doc_id < worst.1))
    {
        heap.pop();
        heap.push(TopKEntry(score, doc_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drain_top_k_desc_orders_descending_with_tiebreak() {
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        heap.push(TopKEntry(1.0, 4));
        heap.push(TopKEntry(2.0, 1));
        heap.push(TopKEntry(2.0, 0)); // tie with doc 1
        let out = drain_top_k_desc(heap);
        assert_eq!(out, vec![(0, 2.0), (1, 2.0), (4, 1.0)]);
    }
}
