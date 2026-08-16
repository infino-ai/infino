// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Exact-phrase matching: a multi-term [`PhraseCursor`] over
//! [`PhraseMember`] atoms with positional verification, and the
//! [`AnyCursor`] enum that lets the atom walks treat a plain term and a
//! phrase uniformly. Drives both scored phrase search and phrase count.
//! Scoped `pub(super)` to the `reader/` module.

use bytes::Bytes;

use super::{
    cursor::{TermCursor, TermMeta},
    metadata::NormTable,
};
use crate::superfile::{
    ReadError,
    error::FtsError,
    fts::{
        bm25,
        positions::{decode_run, skip_run},
    },
};

/// One member term of a [`PhraseCursor`]: its posting cursor, its
/// fetched position runs, and a lazily-built per-block cache of each
/// pair's run offset.
pub(super) struct PhraseMember {
    pub(super) cursor: TermCursor,
    /// The term's complete position runs (empty for an inline df=1
    /// member, whose single position is `inline_position`).
    pub(super) positions: Bytes,
    /// The term's parsed metadata header, re-parsed from the cursor's
    /// own bytes at member build — the source of the per-block
    /// position-run offsets. `None` for an inline member (no postings
    /// bytes). Kept here, not on [`TermCursor`] or [`BlockMeta`]:
    /// plain term queries never touch positions, and their hot
    /// structures must not grow for the phrase path's benefit.
    pub(super) term_meta: Option<TermMeta>,
    /// The single position of an inline (df=1, tf=1) member — the
    /// inline FST value's slot carries it instead of a tf. `None` for
    /// PFOR members.
    pub(super) inline_position: Option<u32>,
    /// The member's bare idf (the cursor stores only `idf × (K1+1)`).
    pub(super) idf: f32,
    /// Byte offset of each decoded-block pair's run within
    /// `positions`, valid for `run_offsets_block`. Rebuilt on block
    /// crossings by one `skip_run` walk over the block's runs. Used by
    /// the `V1`/`V2` fallback decode (no sub-index).
    pub(super) run_offsets: Vec<u32>,
    /// Which block index `run_offsets` / the sub-index cache covers
    /// (`usize::MAX` = none).
    pub(super) run_offsets_block: usize,
    /// Sub-index decode (`V3`) cache: the last pair whose run offset was
    /// resolved in `run_offsets_block`, and that run's byte offset. Pairs
    /// are visited in ascending order within a block, so the next decode
    /// skips from `max(this cached pair, the sub-index checkpoint)` —
    /// dense reuse costs ~one `skip_run`, sparse access at most
    /// `POSITION_SUBINDEX_STRIDE - 1`. `usize::MAX` = nothing cached.
    pub(super) cached_pair: usize,
    pub(super) cached_run_offset: u32,
    /// Scratch for the member's decoded positions at the aligned doc.
    pub(super) pos_scratch: Vec<u32>,
}

/// Sentinel for [`PhraseMember::run_offsets_block`]: no block cached.
const NO_BLOCK_CACHED: usize = usize::MAX;

impl PhraseMember {
    /// The member's positions at its cursor's current doc, decoded
    /// into `pos_scratch`. The cursor must be positioned on a doc
    /// (not exhausted).
    pub(super) fn decode_current_positions(&mut self) -> Result<(), FtsError> {
        self.pos_scratch.clear();
        if let Some(p) = self.inline_position {
            self.pos_scratch.push(p);
            return Ok(());
        }
        let block = self.cursor.current_block;
        let pair = self.cursor.pos;

        // Fast path (VERSION_V3): the run-offset sub-index gives the
        // nearest checkpoint at or before `pair`. Start the skip from
        // whichever is closer to `pair` — that checkpoint, or the pair we
        // resolved last in this same block (pairs are visited ascending,
        // so the last one is `<= pair`). Dense reuse then costs ~one
        // `skip_run`; sparse access at most `STRIDE - 1`. Returns an owned
        // tuple, so no `term_meta` borrow is held across the decode below.
        let subindex = self
            .term_meta
            .as_ref()
            .expect("PFOR member has term meta")
            .positions_subindex_offset(self.cursor.bytes.as_ref(), block, pair);
        if let Some((checkpoint, runs_to_skip)) = subindex {
            let checkpoint_pair = pair - runs_to_skip;
            let (mut from_pair, mut at) = (checkpoint_pair, checkpoint as usize);
            if self.run_offsets_block == block
                && self.cached_pair >= checkpoint_pair
                && self.cached_pair <= pair
            {
                from_pair = self.cached_pair;
                at = self.cached_run_offset as usize;
            }
            for p in from_pair..pair {
                skip_run(&self.positions, &mut at, self.cursor.block_tfs[p]).ok_or_else(|| {
                    FtsError::Read(ReadError::MalformedVersion(
                        "position runs truncated within block".into(),
                    ))
                })?;
            }
            // Cache this pair's run start for the next (higher) pair.
            self.run_offsets_block = block;
            self.cached_pair = pair;
            self.cached_run_offset = at as u32;
            decode_run(
                &self.positions,
                &mut at,
                self.cursor.block_tfs[pair],
                &mut self.pos_scratch,
            )
            .ok_or_else(|| {
                FtsError::Read(ReadError::MalformedVersion(
                    "position run truncated or overflowing".into(),
                ))
            })?;
            return Ok(());
        }

        // Fallback (V1/V2, no sub-index): build the block's run offsets by
        // walking every run from the block's recorded first-run offset.
        if self.run_offsets_block != block {
            self.run_offsets.clear();
            let block_first = self
                .term_meta
                .as_ref()
                .expect("PFOR member has term meta")
                .positions_block_offset(self.cursor.bytes.as_ref(), block)
                as usize;
            let mut at = block_first;
            for i in 0..self.cursor.block_n {
                self.run_offsets.push(at as u32);
                skip_run(&self.positions, &mut at, self.cursor.block_tfs[i]).ok_or_else(|| {
                    FtsError::Read(ReadError::MalformedVersion(
                        "position runs truncated within block".into(),
                    ))
                })?;
            }
            self.run_offsets_block = block;
        }
        let mut at = self.run_offsets[pair] as usize;
        decode_run(
            &self.positions,
            &mut at,
            self.cursor.block_tfs[pair],
            &mut self.pos_scratch,
        )
        .ok_or_else(|| {
            FtsError::Read(ReadError::MalformedVersion(
                "position run truncated or overflowing".into(),
            ))
        })?;
        Ok(())
    }
}

/// Doc-at-a-time cursor over an exact phrase: the members'
/// intersection drives doc alignment, and a doc matches only when the
/// members' positions verify adjacency (member `i` at `p + i` for
/// some anchor `p`). Scores as one BM25 atom with `tf` = the number
/// of verified anchors and `idf` = Σ member idf. Exposes the same
/// notion of term- and block-level upper bounds as [`TermCursor`], so
/// the atom walks can prune with it:
/// `bound = phrase_idf × min_i(member_bound_i / idf_i)` — sound
/// because the phrase tf in any doc is ≤ every member's tf there and
/// the BM25 tf-factor is monotone in tf.
pub(super) struct PhraseCursor {
    pub(super) members: Vec<PhraseMember>,
    /// Member indices in ascending posting-list length (rarest first).
    /// The doc-alignment in [`Self::seek_match`] is a set intersection —
    /// order-independent — so it probes members rarest-first: the short
    /// lists drive the candidate doc and the long lists (a common word
    /// like "the") are only skip-confirmed last, once per candidate,
    /// instead of being re-skipped on every advance of a rare member.
    /// Positional verification still runs in query order (`members`
    /// order), which the phrase adjacency check requires.
    pub(super) align_order: Vec<usize>,
    /// Σ member idf × (K1 + 1) — the phrase's scoring constant.
    pub(super) idf_x_k1p1: f32,
    /// Phrase-scaled term-level upper bound (see type docs).
    pub(super) term_max_bm25: f32,
    /// Aligned-and-verified doc, or `u32::MAX` when exhausted.
    pub(super) current_doc: u32,
    /// Number of verified anchors at `current_doc`.
    pub(super) current_tf: u32,
    /// Reused across `verify_at_aligned` calls to hold the candidate
    /// phrase-start positions as they are filtered member by member —
    /// avoids a per-doc allocation on the hot verify path.
    pub(super) verify_scratch: Vec<u32>,
}

impl PhraseCursor {
    /// Build from member cursors (query order), their fetched
    /// position runs, and their positional metadata — `(term_meta,
    /// inline_position)` per member, exactly one of the two present —
    /// then seek to the first matching doc.
    pub(super) fn new(
        cursors: Vec<TermCursor>,
        positions: Vec<Bytes>,
        positional: Vec<(Option<TermMeta>, Option<u32>)>,
    ) -> Result<Self, FtsError> {
        debug_assert!(cursors.len() >= 2, "single-token phrases degrade to terms");
        debug_assert_eq!(cursors.len(), positions.len());
        debug_assert_eq!(cursors.len(), positional.len());
        let mut idf_sum = 0.0f32;
        let mut min_scaled_bound = f32::INFINITY;
        let members: Vec<PhraseMember> = cursors
            .into_iter()
            .zip(positions)
            .zip(positional)
            .map(|((cursor, positions), (term_meta, inline_position))| {
                let idf = cursor.idf_x_k1p1 / (bm25::K1 + 1.0);
                min_scaled_bound = min_scaled_bound.min(cursor.term_max_bm25 / idf);
                idf_sum += idf;
                PhraseMember {
                    cursor,
                    positions,
                    term_meta,
                    inline_position,
                    idf,
                    run_offsets: Vec::new(),
                    run_offsets_block: NO_BLOCK_CACHED,
                    cached_pair: NO_BLOCK_CACHED,
                    cached_run_offset: 0,
                    pos_scratch: Vec::new(),
                }
            })
            .collect();
        // Rarest-first probe order for alignment: fewest posting blocks
        // (shortest list) first. Query order is preserved in `members`.
        let mut align_order: Vec<usize> = (0..members.len()).collect();
        align_order.sort_by_key(|&i| members[i].cursor.block_count());
        let mut cursor = Self {
            idf_x_k1p1: idf_sum * (bm25::K1 + 1.0),
            term_max_bm25: idf_sum * min_scaled_bound,
            members,
            align_order,
            current_doc: 0,
            current_tf: 0,
            verify_scratch: Vec::new(),
        };
        cursor.seek_match_unranked(0)?;
        Ok(cursor)
    }

    #[inline]
    pub(super) fn is_exhausted(&self) -> bool {
        self.current_doc == u32::MAX
    }

    #[inline]
    pub(super) fn current_doc_id(&self) -> u32 {
        self.current_doc
    }

    /// Advance to the first verified phrase match at doc ≥ `target`.
    pub(super) fn skip_to(&mut self, target: u32) -> Result<(), FtsError> {
        if self.is_exhausted() || self.current_doc >= target {
            return Ok(());
        }
        self.seek_match_unranked(target)
    }

    /// [`Self::skip_to`] for ranked walks: additionally skips docs
    /// whose phrase contribution provably can't matter. `bar` is the
    /// most this atom may need to contribute (the walk's pruning bar
    /// minus every other atom's upper bound); a doc whose phrase
    /// score bound falls strictly below it is passed over without any
    /// position work — sound for top-k because the doc's total score
    /// then can't reach the bar, but NOT for match/count walks, which
    /// must keep using [`Self::skip_to`].
    pub(super) fn skip_to_pruned(
        &mut self,
        target: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        if self.is_exhausted() || self.current_doc >= target {
            return Ok(());
        }
        self.seek_match(target, bar, dl_norm_k1)
    }

    /// Approximate seek (the cheap half of a two-phase phrase): advance to the
    /// next doc ≥ `from` that contains **every member** — the members'
    /// doc-intersection — **without** verifying adjacency or decoding any
    /// positions. Drives off the rarest member (`align_order[0]`, the only one
    /// iterated) and confirms the rest by `contains` bit-test on their dense
    /// blocks, so a common word like "the" is never decoded just to align a
    /// doc. Sets `current_doc` to that doc (`u32::MAX` when exhausted) and
    /// leaves `current_tf` at 0 — the doc is a *candidate*, not yet a verified
    /// phrase match.
    ///
    /// An AND atom walk aligns on this approximation across all atoms (so a
    /// rare co-clause prunes the candidate set first) and only then asks
    /// [`Self::verify_at_aligned`] to decode positions on the survivors. On its
    /// own, `approx_seek` + a `verify_at_aligned` retry loop is exactly
    /// [`Self::seek_match_unranked`].
    pub(super) fn approx_seek(&mut self, mut from: u32) {
        let driver = self.align_order[0];
        'docs: loop {
            // Advance the rarest member; it alone drives the candidate doc.
            {
                let d = &mut self.members[driver].cursor;
                d.skip_to(from);
                if d.is_exhausted() {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return;
                }
            }
            let aligned = self.members[driver].cursor.current_doc_id();
            // Every other member must contain `aligned` — a bit-test on a
            // dense block, no decode. A miss advances the driver past it.
            for oi in 1..self.align_order.len() {
                let mi = self.align_order[oi];
                if !self.members[mi].cursor.contains(aligned) {
                    match aligned.checked_add(1) {
                        Some(next) => from = next,
                        None => {
                            self.current_doc = u32::MAX;
                            self.current_tf = 0;
                            return;
                        }
                    }
                    continue 'docs;
                }
            }
            self.current_doc = aligned;
            self.current_tf = 0;
            return;
        }
    }

    /// Unranked (match/count) doc alignment to the next *verified* phrase match
    /// ≥ `from`: the approximate member-intersection ([`Self::approx_seek`])
    /// followed by adjacency verification, retrying at the next candidate until
    /// one verifies or the members exhaust. The count-path twin of the ranked
    /// [`Self::seek_match`], which must keep decoding tfs for its score bar.
    pub(super) fn seek_match_unranked(&mut self, mut from: u32) -> Result<(), FtsError> {
        loop {
            self.approx_seek(from);
            if self.current_doc == u32::MAX {
                return Ok(());
            }
            let aligned = self.current_doc;
            // Verify adjacency; the probed members are decoded here, lazily.
            let tf = self.verify_at_aligned(aligned)?;
            if tf > 0 {
                self.current_tf = tf;
                return Ok(());
            }
            match aligned.checked_add(1) {
                Some(next) => from = next,
                None => {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return Ok(());
                }
            }
        }
    }

    /// Leapfrog the members to their next common doc ≥ `from`, verify
    /// adjacency there, and repeat until a match or exhaustion. When
    /// `bar` is finite, aligned docs are pre-screened without touching
    /// positions: the phrase tf can't exceed any member's tf, so the
    /// BM25 score at the members' minimum tf bounds the phrase's
    /// contribution, and a doc strictly below `bar` is skipped before
    /// the run decode. (`<`, not `<=`: a doc exactly at the bar can
    /// still displace the incumbent kth-best on the ascending-doc-id
    /// tie-break, so it must be verified.)
    pub(super) fn seek_match(
        &mut self,
        mut from: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        'docs: loop {
            // Align every member to the same doc ≥ `from`, probing
            // rarest-first so a common member is skip-confirmed last
            // rather than re-skipped on every rare-member advance.
            let mut aligned = from;
            let mut oi = 0usize;
            while oi < self.align_order.len() {
                let mi = self.align_order[oi];
                let c = &mut self.members[mi].cursor;
                c.skip_to(aligned);
                if c.is_exhausted() {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return Ok(());
                }
                let here = c.current_doc_id();
                if here > aligned {
                    // Restart alignment at the higher doc.
                    aligned = here;
                    oi = 0;
                    continue;
                }
                oi += 1;
            }

            if bar > f32::NEG_INFINITY {
                let min_tf = self
                    .members
                    .iter()
                    .map(|m| m.cursor.current_tf())
                    .min()
                    .expect("members >= 2");
                let ub =
                    bm25::score_with_dl_norm_k1(self.idf_x_k1p1, min_tf, dl_norm_k1.get(aligned));
                if ub < bar {
                    from = match aligned.checked_add(1) {
                        Some(next) => next,
                        None => {
                            self.current_doc = u32::MAX;
                            self.current_tf = 0;
                            return Ok(());
                        }
                    };
                    continue 'docs;
                }
            }

            // Verify adjacency at the aligned doc.
            let tf = self.verify_at_aligned(aligned)?;
            if tf > 0 {
                self.current_doc = aligned;
                self.current_tf = tf;
                return Ok(());
            }
            from = match aligned.checked_add(1) {
                Some(next) => next,
                None => {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return Ok(());
                }
            };
            continue 'docs;
        }
    }

    /// Count the phrase's anchors at the members' aligned doc: the
    /// first member's positions `p` where member `i` also has `p + i`
    /// for every `i`. Member position lists are ascending, so each
    /// probe is a binary search over a per-doc-tf-sized slice.
    pub(super) fn verify_at_aligned(&mut self, aligned: u32) -> Result<u32, FtsError> {
        // Staged, rarest-first, lazy-decode verification. A phrase match
        // starting at position `s` has member `j` at `s + j`, so any
        // member can seed the candidate starts: the rarest member (by
        // posting length — `align_order[0]`) seeds them, then each
        // remaining member filters the survivors *in rarest-first order*.
        //
        // Two wins over decode-all-then-probe-from-the-first-member:
        //   * The seed loop is as short as the rarest member's per-doc tf,
        //     not the (often common) query-first member's.
        //   * Decoding is lazy: a common member's positions — whose
        //     per-block run-offset walk is the real cost — are only
        //     decoded once some candidate survives every rarer member.
        //     On the huge co-occurrence sets a phrase with a common word
        //     produces, almost every doc is rejected by a rare member
        //     first, so the common members are never decoded there.
        //
        // `materialize_at` decodes each member's block only now: on the
        // unranked path a common member reached `aligned` by a `contains`
        // bit-test and its block is not yet decoded; on the ranked path it
        // was already decoded by `skip_to`, so this is a no-op there.
        let anchor = self.align_order[0];
        let anchor_off = anchor as u32;
        self.members[anchor].cursor.materialize_at(aligned);
        self.members[anchor].decode_current_positions()?;
        self.verify_scratch.clear();
        for &pa in &self.members[anchor].pos_scratch {
            if let Some(start) = pa.checked_sub(anchor_off) {
                self.verify_scratch.push(start);
            }
        }
        for oi in 1..self.align_order.len() {
            if self.verify_scratch.is_empty() {
                break;
            }
            let j = self.align_order[oi];
            self.members[j].cursor.materialize_at(aligned);
            self.members[j].decode_current_positions()?;
            let plist = &self.members[j].pos_scratch;
            let off = j as u32;
            // Compact the survivors in place: keep a start iff member `j`
            // holds `start + j`.
            let mut w = 0usize;
            for r in 0..self.verify_scratch.len() {
                let start = self.verify_scratch[r];
                let keep = start
                    .checked_add(off)
                    .is_some_and(|want| plist.binary_search(&want).is_ok());
                if keep {
                    self.verify_scratch[w] = start;
                    w += 1;
                }
            }
            self.verify_scratch.truncate(w);
        }
        // The surviving starts are this doc's phrase occurrences — its tf.
        // Store it so `score_current` scores the phrase after a two-phase
        // `verify_at` (the single-phase `skip_to_pruned` sets it itself).
        self.current_tf = self.verify_scratch.len() as u32;
        Ok(self.current_tf)
    }

    /// Score the phrase at its current doc with the caller-supplied
    /// per-doc BM25 normalization.
    #[inline]
    pub(super) fn score_current(&self, dl_norm_k1: f32) -> f32 {
        bm25::score_with_dl_norm_k1(self.idf_x_k1p1, self.current_tf, dl_norm_k1)
    }

    /// Phrase-scaled block-level upper bound over `[range_start,
    /// range_end]` — the block analog of `term_max_bm25`.
    pub(super) fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        let mut min_scaled = f32::INFINITY;
        for m in self.members.iter_mut() {
            let b = m.cursor.block_max_in_range(range_start, range_end);
            min_scaled = min_scaled.min(b / m.idf);
        }
        let idf_sum = self.idf_x_k1p1 / (bm25::K1 + 1.0);
        idf_sum * min_scaled
    }
}

/// A query atom's cursor: a plain term or an exact phrase. The atom
/// walks below are heterogeneous doc-at-a-time loops over this enum —
/// deliberately separate from the field-level optimized kernels
/// (flat-merge AND, MaxScore/BMM, windowed union), which keep serving
/// term-only queries unchanged. A query containing any phrase routes
/// here: correctness-first walks whose per-doc cost is dominated by
/// the phrase verification itself.
pub(super) enum AnyCursor {
    Term(TermCursor),
    Phrase(PhraseCursor),
}

impl AnyCursor {
    #[inline]
    pub(super) fn is_exhausted(&self) -> bool {
        match self {
            AnyCursor::Term(c) => c.is_exhausted(),
            AnyCursor::Phrase(c) => c.is_exhausted(),
        }
    }

    #[inline]
    pub(super) fn current_doc_id(&self) -> u32 {
        match self {
            AnyCursor::Term(c) => c.current_doc_id(),
            AnyCursor::Phrase(c) => c.current_doc_id(),
        }
    }

    /// Advance to the first (phrase: first *verified*) doc ≥ `target`.
    pub(super) fn skip_to(&mut self, target: u32) -> Result<(), FtsError> {
        match self {
            AnyCursor::Term(c) => {
                c.skip_to(target);
                Ok(())
            }
            AnyCursor::Phrase(c) => c.skip_to(target),
        }
    }

    /// Two-phase alignment for the AND walk: advance to the atom's next
    /// *candidate* doc ≥ `target` without paying for a phrase's positions. A
    /// term atom is exact (its doc *is* a match); a phrase atom advances to the
    /// next doc holding all its members ([`PhraseCursor::approx_seek`]), leaving
    /// adjacency for [`Self::verify_at`]. Pairs with [`Self::approx_current_doc`].
    pub(super) fn approx_skip_to(&mut self, target: u32) {
        match self {
            AnyCursor::Term(c) => c.skip_to(target),
            AnyCursor::Phrase(c) => c.approx_seek(target),
        }
    }

    /// The atom's current *candidate* doc (see [`Self::approx_skip_to`]): a term
    /// atom's decoded doc, or a phrase atom's member-aligned doc (`u32::MAX`
    /// when exhausted).
    pub(super) fn approx_current_doc(&self) -> u32 {
        match self {
            AnyCursor::Term(c) => c.current_doc_id(),
            AnyCursor::Phrase(c) => c.current_doc,
        }
    }

    /// Confirm the atom actually matches at `doc` — the doc its approximation
    /// has already reached. A term atom trivially matches; a phrase atom decodes
    /// positions and checks adjacency ([`PhraseCursor::verify_at_aligned`]). The
    /// expensive half of the two-phase split, run by the AND walk only on docs
    /// where every atom's approximation agrees, so a rare co-clause prunes the
    /// position work.
    pub(super) fn verify_at(&mut self, doc: u32) -> Result<bool, FtsError> {
        match self {
            AnyCursor::Term(_) => Ok(true),
            AnyCursor::Phrase(c) => Ok(c.verify_at_aligned(doc)? > 0),
        }
    }

    /// [`Self::skip_to`] with the ranked walks' pruning bar: a phrase
    /// atom skips docs it provably can't lift over the bar without
    /// doing any position work (see [`PhraseCursor::skip_to_pruned`]).
    /// Term atoms ignore the bar — their per-doc score costs nothing
    /// beyond the postings walk itself.
    pub(super) fn skip_to_pruned(
        &mut self,
        target: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        match self {
            AnyCursor::Term(c) => {
                c.skip_to(target);
                Ok(())
            }
            AnyCursor::Phrase(c) => c.skip_to_pruned(target, bar, dl_norm_k1),
        }
    }

    /// BM25 contribution at the cursor's current doc.
    #[inline]
    pub(super) fn score_current(&self, dl_norm_k1: f32) -> f32 {
        match self {
            AnyCursor::Term(c) => {
                bm25::score_with_dl_norm_k1(c.idf_x_k1p1, c.current_tf(), dl_norm_k1)
            }
            AnyCursor::Phrase(c) => c.score_current(dl_norm_k1),
        }
    }

    /// Atom-level score upper bound (any doc).
    #[inline]
    pub(super) fn term_max_bm25(&self) -> f32 {
        match self {
            AnyCursor::Term(c) => c.term_max_bm25,
            AnyCursor::Phrase(c) => c.term_max_bm25,
        }
    }

    /// Score upper bound over the doc range (see the cursors' docs).
    #[inline]
    pub(super) fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        match self {
            AnyCursor::Term(c) => c.block_max_in_range(range_start, range_end),
            AnyCursor::Phrase(c) => c.block_max_in_range(range_start, range_end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_util::*, *};
    use crate::superfile::fts::reader::{FtsReader, core::ClauseLists};

    fn phrase(terms: &[&str]) -> Vec<Vec<String>> {
        vec![terms.iter().map(|t| t.to_string()).collect()]
    }

    #[tokio::test]
    async fn phrase_matches_adjacent_in_order_only() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let phrases = phrase(&["new", "york"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 2, 4], "adjacency in order only");
        // Doc 4 has the phrase twice — highest tf, and with uniform
        // doc lengths in play its score must strictly exceed doc 0's
        // (same length, tf 1... doc 0 len 3, doc 4 len 4; tf=2 wins).
        assert_eq!(hits[0].0, 4, "double occurrence ranks first");
    }

    #[tokio::test]
    async fn phrase_composes_with_clauses() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let ny = phrase(&["new", "york"]);

        // Must-phrase + must-term: "the" only in doc 2.
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    musts: &["the"],
                    must_phrases: &ny,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("must phrase + term");
        assert_eq!(
            hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
            vec![2],
            "+\"new york\" +the"
        );

        // Negated phrase: haven-docs minus the phrase docs.
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    shoulds: &["haven"],
                    negative_phrases: &ny,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("negated phrase");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3], "haven docs don't contain the phrase");
    }

    #[tokio::test]
    async fn phrase_with_absent_member_matches_nothing() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let ghost = phrase(&["new", "zealand"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    must_phrases: &ghost,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("ghost phrase");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn phrase_on_positionless_column_is_typed_error() {
        use crate::superfile::fts::builder::FtsBuilder;
        let mut b = FtsBuilder::new(crate::test_helpers::default_tokenizer());
        b.register_column("title".into(), false).expect("register");
        b.add_doc(0, 0, "new york").expect("add doc");
        let blob = Bytes::from(b.finish().expect("finish"));
        let r =
            FtsReader::open(blob, r#"[{"name":"title","tokenizer":"ascii_lower"}]"#).expect("open");
        let phrases = phrase(&["new", "york"]);
        let err = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect_err("must be a typed error");
        assert!(matches!(err, FtsError::PositionsUnavailable { .. }));
    }
}
