// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term groups: a [`GroupCursor`] scores several surface forms of one
//! word (`run`, `runs`, `running`, `ran`) as a single BM25 atom — what a
//! stemmed index would have stored for the one stem they share, computed
//! at query time over the unstemmed postings. It is the third
//! [`AnyCursor`](super::phrase::AnyCursor) variant beside the plain term
//! and the phrase, and rides the same atom walks. Scoped `pub(super)` to
//! the `reader/` module.

use super::cursor::TermCursor;
use crate::superfile::fts::bm25;

/// Doc-at-a-time cursor over a term group: the union of the members'
/// posting lists, scored as one term.
///
/// - **Match.** The group is present at a doc when any member is; the
///   cursor sits on the smallest current doc across its live members.
/// - **tf** at a doc is the sum of the members' term frequencies there —
///   the count a stemmed index would hold for the stem.
/// - **idf** is the idf of the member with the largest document
///   frequency, i.e. the smallest member idf. The commonest form sets the
///   group's rarity, so a rare inflection can no longer outscore the base
///   form and a doc holding two forms is credited for one term, not two.
///   Under table-wide statistics the caller passes the table-wide
///   commonest member's idf instead of the per-superfile minimum.
/// - **Upper bounds** for the pruning walks stay sound; the argument is
///   on [`Self::block_max_in_range`].
///
/// Only members present in the superfile are built into the cursor; the
/// caller drops absent members the way it drops absent single terms, and
/// a group with no member present is absent.
pub(super) struct GroupCursor {
    /// Member cursors (the head first, then its surface forms).
    pub(super) members: Vec<TermCursor>,
    /// The group's `idf × (K1 + 1)` — the scoring constant.
    idf_x_k1p1: f32,
    /// Group-scaled term-level upper bound (see type docs).
    term_max_bm25: f32,
    /// Smallest current doc across live members, or `u32::MAX` when every
    /// member is exhausted.
    current_doc: u32,
    /// Σ member tf at `current_doc`.
    current_tf: u32,
}

impl GroupCursor {
    /// Build from the present members' cursors (query order) and, under
    /// table-wide statistics, the group idf the caller derived from the
    /// members' table-wide document frequencies; `None` takes the
    /// smallest member idf in this superfile.
    pub(super) fn new(members: Vec<TermCursor>, global_idf: Option<f32>) -> Self {
        debug_assert!(
            !members.is_empty(),
            "a group with no present member is absent"
        );
        let local_min = members
            .iter()
            .map(TermCursor::idf)
            .fold(f32::INFINITY, f32::min);
        let idf = global_idf.unwrap_or(local_min);
        // Same construction as `block_max_in_range`, over the whole list:
        // each member's term-level bound with its own idf divided out,
        // summed, then scaled by the group idf.
        let term_max_bm25 = idf
            * members
                .iter()
                .map(TermCursor::term_max_tf_factor)
                .sum::<f32>();
        let mut cursor = Self {
            members,
            idf_x_k1p1: idf * (bm25::K1 + 1.0),
            term_max_bm25,
            current_doc: 0,
            current_tf: 0,
        };
        cursor.align();
        cursor
    }

    /// Re-derive `current_doc` / `current_tf` from the members' positions:
    /// the smallest live member doc and the summed tf of every member
    /// sitting on it.
    fn align(&mut self) {
        let mut doc = u32::MAX;
        for m in &self.members {
            if !m.is_exhausted() {
                doc = doc.min(m.current_doc_id());
            }
        }
        self.current_doc = doc;
        self.current_tf = match doc {
            u32::MAX => 0,
            _ => self
                .members
                .iter()
                .filter(|m| !m.is_exhausted() && m.current_doc_id() == doc)
                .map(TermCursor::current_tf)
                .sum(),
        };
    }

    #[inline]
    pub(super) fn is_exhausted(&self) -> bool {
        self.current_doc == u32::MAX
    }

    #[inline]
    pub(super) fn current_doc_id(&self) -> u32 {
        self.current_doc
    }

    /// Advance to the first doc ≥ `target` that holds any member.
    pub(super) fn skip_to(&mut self, target: u32) {
        if self.is_exhausted() || self.current_doc >= target {
            return;
        }
        for m in &mut self.members {
            m.skip_to(target);
        }
        self.align();
    }

    /// BM25 contribution at the current doc with the caller-supplied
    /// per-doc normalization: one saturation over the summed tf.
    #[inline]
    pub(super) fn score_current(&self, dl_norm_k1: f32) -> f32 {
        bm25::score_with_dl_norm_k1(self.idf_x_k1p1, self.current_tf, dl_norm_k1)
    }

    /// Group-level upper bound over any doc (see type docs).
    #[inline]
    pub(super) fn term_max_bm25(&self) -> f32 {
        self.term_max_bm25
    }

    /// Upper bound on the group's score at any doc in `[range_start,
    /// range_end]`: the group idf times the sum, over members, of each
    /// member's block-level bound with its own idf divided out.
    ///
    /// Why this is a bound. BM25's saturation `f(tf) = tf·(k1+1) / (tf +
    /// K)` (with `K = k1 · norm > 0`) is increasing and concave with
    /// `f(0) = 0`, so it is subadditive: `f(a + b) ≤ f(a) + f(b)`. At a doc
    /// `d` in the range with member frequencies `tf_i`, the group scores
    /// `idf_g · f(Σ tf_i) ≤ idf_g · Σ f(tf_i) = Σ (idf_g / idf_i) · [idf_i ·
    /// f(tf_i)]`, and each bracket is member `i`'s own BM25 score at `d`,
    /// which that member's stored block maximum over the range already
    /// bounds from above. A member absent from `d` has `tf_i = 0` and adds
    /// `f(0) = 0` on the left, so only members whose blocks cover the
    /// range contribute. Hence `score_g(d) ≤ idf_g · Σ_i bound_i / idf_i`.
    /// The phrase cursor's bound is the same construction with `min` in
    /// place of `Σ`: a phrase's tf is at most every member's tf, a
    /// group's tf is their sum.
    pub(super) fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        let idf = self.idf_x_k1p1 / (bm25::K1 + 1.0);
        idf * self
            .members
            .iter_mut()
            .map(|m| m.block_max_tf_factor_in_range(range_start, range_end))
            .sum::<f32>()
    }
}
