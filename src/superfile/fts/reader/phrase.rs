// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Exact-phrase matching: a multi-term [`PhraseCursor`] over
//! [`PhraseMember`] atoms with positional verification, and the
//! [`AnyCursor`] enum that lets the atom walks treat a plain term and a
//! phrase uniformly. Drives both scored phrase search and phrase count.
//! Scoped `pub(super)` to the `reader/` module.

#[cfg(test)]
use std::cell::Cell;
use std::mem::take;

use bytes::Bytes;

use super::{
    cursor::{TermCursor, TermMeta},
    metadata::NormTable,
};
use crate::superfile::{
    ReadError,
    error::FtsError,
    fts::{bm25, positions::GroupIndex},
};

/// One distinct term of a [`PhraseCursor`] ([`distinct_members`]): its
/// posting cursor, its fetched position groups, and the current block's
/// group located for per-pair access.
pub(super) struct PhraseMember {
    pub(super) cursor: TermCursor,
    /// A second cursor over the same postings for the block-at-a-time
    /// alignment: it filters a whole block of the rarest member's docs
    /// ([`TermCursor::retain_contained_located`]) and records where this
    /// member holds each survivor, so the survivors' tfs and positions are
    /// read at those locations without moving `cursor`.
    pub(super) probe: TermCursor,
    /// The term's complete position groups (empty for an inline df=1
    /// member, whose single position is `inline_position`).
    pub(super) positions: Bytes,
    /// The term's parsed metadata header, re-parsed from the cursor's
    /// own bytes at member build — the source of the per-block
    /// position-group offsets. `None` for an inline member (no postings
    /// bytes). Kept here, not on [`TermCursor`] or [`BlockMeta`]:
    /// plain term queries never touch positions, and their hot
    /// structures must not grow for the phrase path's benefit.
    pub(super) term_meta: Option<TermMeta>,
    /// The single position of an inline (df=1, tf=1) member — the
    /// inline dictionary entry's slot carries it instead of a tf. `None` for
    /// PFOR members.
    pub(super) inline_position: Option<u32>,
    /// Scratch for the member's decoded positions at the aligned doc.
    pub(super) pos_scratch: Vec<u32>,
    /// The current block's position group located for per-run access,
    /// and which block it belongs to (`usize::MAX` = none). Reused
    /// across blocks, so a block crossing allocates nothing.
    pub(super) group_index: GroupIndex,
    pub(super) group_block: usize,
    /// Block `tfs_block`'s tfs, decoded without its doc ids, for reading
    /// candidates by located pair ([`Self::located_positions`]).
    tfs: Vec<u32>,
    tfs_block: usize,
}

/// Sentinel for [`PhraseMember::group_block`]: no block located.
const NO_BLOCK_CACHED: usize = usize::MAX;

impl PhraseMember {
    /// The member's positions at its cursor's current doc, decoded
    /// into `pos_scratch`. The cursor must be positioned on a doc
    /// (not exhausted) **with its whole block decoded**: `pos` is read as
    /// the pair index within the block and `block_tfs[..block_n]` as the
    /// block's tf run. A cursor that reached the doc by a skip into a
    /// bitset block holds only that one doc (`block_n == 1`, `pos == 0`)
    /// and would read the wrong run; `materialize_at` expands it first,
    /// which is why every caller goes through it.
    pub(super) fn decode_current_positions(&mut self) -> Result<(), FtsError> {
        if self.inline_position.is_none() {
            debug_assert_eq!(
                self.cursor.decoded_block, self.cursor.current_block,
                "positions need the whole block decoded; call materialize_at first"
            );
        }
        self.located_positions(self.cursor.current_block, self.cursor.pos)
    }
}

impl PhraseMember {
    /// Decode block `block`'s tfs into the member's cache, unless they are
    /// the ones it holds.
    fn ensure_tfs(&mut self, block: usize) {
        if self.tfs_block == block {
            return;
        }
        let c = &self.cursor;
        if !c.predecoded && c.decoded_block == block && c.tf_decoded_block == block {
            // The walk cursor already holds this block's tfs.
            self.tfs.clear();
            self.tfs.extend_from_slice(&c.block_tfs[..c.block_n]);
        } else {
            c.block_tfs_into(block, &mut self.tfs);
        }
        self.tfs_block = block;
    }

    /// The member's tf in its `pair`-th posting of block `block`, a location
    /// from [`TermCursor::retain_contained_located`].
    fn located_tf(&mut self, block: usize, pair: usize) -> u32 {
        if self.inline_position.is_some() {
            return 1;
        }
        self.ensure_tfs(block);
        self.tfs[pair]
    }

    /// The member's positions in its `pair`-th posting of block `block`,
    /// decoded into `pos_scratch`. Reads the block's tfs and position group
    /// once per block and the posting's run alone: the doc ids are never
    /// expanded, and no cursor is walked to the doc.
    fn located_positions(&mut self, block: usize, pair: usize) -> Result<(), FtsError> {
        self.pos_scratch.clear();
        if let Some(p) = self.inline_position {
            self.pos_scratch.push(p);
            return Ok(());
        }
        self.ensure_tfs(block);
        let term_meta = *self.term_meta.as_ref().expect("PFOR member has term meta");
        if self.group_block != block {
            let mut at =
                term_meta.positions_block_offset(self.cursor.bytes.as_ref(), block) as usize;
            self.group_index
                .locate(&self.positions, &mut at, &self.tfs)
                .ok_or_else(|| {
                    FtsError::Read(ReadError::Malformed(
                        "position group truncated or undecodable".into(),
                    ))
                })?;
            self.group_block = block;
        }
        self.group_index
            .run_positions(&self.positions, pair, self.tfs[pair], &mut self.pos_scratch)
            .ok_or_else(|| {
                FtsError::Read(ReadError::Malformed(
                    "position run truncated or overflowing".into(),
                ))
            })?;
        Ok(())
    }
}

/// The phrase's occurrences at one aligned doc, verified in stages,
/// rarest member first. A phrase starting at position `s` has query
/// position `q`'s term (member `member_at[q]`) at `s + offsets[q]`, so the
/// rarest query position (`order[0]`) seeds the candidate starts and each
/// other one, in `order`, keeps the starts it confirms, compacted in place
/// in `starts`. A member's positions are only read (`positions_of`, which
/// fills its `pos_scratch`) once some start survives every rarer member, so
/// on the many docs a rare member rejects the common members' runs are
/// never decoded. `order` lists the query positions of one member
/// consecutively, so its positions are read once. Returns the surviving
/// starts: the phrase's tf at the doc.
fn verify_members(
    members: &mut [PhraseMember],
    member_at: &[usize],
    order: &[usize],
    offsets: &[u32],
    starts: &mut Vec<u32>,
    mut positions_of: impl FnMut(&mut PhraseMember, usize) -> Result<(), FtsError>,
) -> Result<u32, FtsError> {
    let anchor = order[0];
    let mut read = member_at[anchor];
    positions_of(&mut members[read], read)?;
    starts.clear();
    starts.extend(
        members[read]
            .pos_scratch
            .iter()
            .filter_map(|&p| p.checked_sub(offsets[anchor])),
    );
    for &q in &order[1..] {
        if starts.is_empty() {
            break;
        }
        let m = member_at[q];
        if m != read {
            positions_of(&mut members[m], m)?;
            read = m;
        }
        let plist = &members[m].pos_scratch;
        let off = offsets[q];
        // Compacted in place without a branch on the outcome, which on a
        // common member is close to a coin flip; the search per start stays,
        // as it beat a merge on every lane measured.
        let mut w = 0usize;
        for r in 0..starts.len() {
            let s = starts[r];
            let found = s
                .checked_add(off)
                .is_some_and(|want| plist.binary_search(&want).is_ok());
            starts[w] = s;
            w += usize::from(found);
        }
        starts.truncate(w);
    }
    Ok(starts.len() as u32)
}

/// A phrase's distinct terms, in order of first appearance, and for each
/// of its query positions the index of its term among them. A repeated
/// term's postings are then fetched, aligned and decoded once.
pub(super) fn distinct_members(terms: &[String]) -> (Vec<&str>, Vec<usize>) {
    let mut distinct: Vec<&str> = Vec::with_capacity(terms.len());
    let member_at = terms
        .iter()
        .map(|t| match distinct.iter().position(|d| *d == t) {
            Some(m) => m,
            None => {
                distinct.push(t);
                distinct.len() - 1
            }
        })
        .collect();
    (distinct, member_at)
}

/// Compact `locs` to the entries at `kept` (ascending indices into it).
fn compact_locs(locs: &mut Vec<(u32, u32)>, kept: &[u32]) {
    for (i, &k) in kept.iter().enumerate() {
        locs[i] = locs[k as usize];
    }
    locs.truncate(kept.len());
}

/// Doc-at-a-time cursor over an exact phrase: the members'
/// intersection drives doc alignment, and a doc matches only when the
/// members' positions verify the phrase's spacing (member `i` at
/// `p + position_offsets[i]` for some anchor `p`). Scores as one BM25 atom with `tf` = the number
/// of verified anchors and `idf` = Σ member idf. Exposes the same
/// notion of term- and block-level upper bounds as [`TermCursor`], so
/// the atom walks can prune with it:
/// `bound = phrase_idf × min_i(member_bound_i / idf_i)` — sound
/// because the phrase tf in any doc is ≤ every member's tf there and
/// the BM25 tf-factor is monotone in tf.
pub(super) struct PhraseCursor {
    /// One per distinct term, in order of first appearance in the query.
    pub(super) members: Vec<PhraseMember>,
    /// For each query position, the index in `members` of its term: the
    /// identity unless the phrase repeats a term.
    member_at: Vec<usize>,
    /// Each query position's **token-position** offset from the phrase's
    /// first term, in query order — strictly ascending, first entry `0`.
    ///
    /// Named for the unit on purpose: a member's positions are addressed
    /// by **byte** offsets into its positions blob. Confusing the two
    /// would be a correctness bug, not a type error.
    ///
    /// Plain `0..n` for a phrase whose words were adjacent in the
    /// query, which is every phrase on a column with no analysis
    /// chain. A column whose stopword set removed a word from the
    /// middle of the phrase gets a gap here, so the verification asks
    /// for the members exactly as far apart as the same chain put them
    /// at index time — see `Phrase` in `fts::tokenize`.
    pub(super) position_offsets: Vec<u32>,
    /// Member indices in ascending posting-list length (rarest first).
    /// The doc-alignment in [`Self::seek_match`] is a set intersection —
    /// order-independent — so it probes members rarest-first: the short
    /// lists drive the candidate doc and the long lists (a common word
    /// like "the") are only skip-confirmed last, once per candidate,
    /// instead of being re-skipped on every advance of a rare member.
    pub(super) align_order: Vec<usize>,
    /// Query positions in the order [`verify_members`] checks them: their
    /// members' `align_order`, a repeated term's positions together.
    verify_order: Vec<usize>,
    /// Σ member idf — the phrase's scoring constant, and the factor a
    /// bound in the "scaled" form is multiplied back up by.
    pub(super) idf_weight: f32,
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
    /// Docs of the rarest member's current block that every other member
    /// contains, awaiting the ranked bound and adjacency verification;
    /// `cand_next` is the first not yet served. Filled a block at a time
    /// by [`Self::seek_match`].
    cands: Vec<u32>,
    cand_next: usize,
    /// Per member, each candidate's (block, pair) location in that
    /// member's postings, in step with `cands`.
    cand_locs: Vec<Vec<(u32, u32)>>,
    /// Scratch: which candidates a member's filter kept.
    kept: Vec<u32>,
    /// Where the next block refill starts: one past the last block
    /// batched. `None` once the driver's last block is batched.
    refill_from: Option<u32>,
    /// Plain terms the phrase is ANDed with ([`Self::filter_by`]): each
    /// filters the batch alongside the members, before any position work,
    /// so only docs holding every term are verified. Empty for a phrase on
    /// its own.
    filters: Vec<TermCursor>,
    /// The batch's sources in ascending posting-list length: index `i <
    /// members.len()` is member `i`, any other is filter `i - members.len()`.
    /// The first drives the candidates; the rest filter them, rarest first.
    /// With no filters it is `align_order`.
    batch_order: Vec<usize>,
    /// Scratch: a filter's locations, which no verification reads.
    filter_locs: Vec<(u32, u32)>,
}

/// The cursor batch source `s` reads from, in [`PhraseCursor::batch_order`]
/// numbering: a member's walk cursor (or its probe, when the member filters
/// rather than drives), else a filter term.
fn source_cursor<'a>(
    members: &'a mut [PhraseMember],
    filters: &'a mut [TermCursor],
    s: usize,
    drives: bool,
) -> &'a mut TermCursor {
    let n = members.len();
    match members.get_mut(s) {
        Some(m) if drives => &mut m.cursor,
        Some(m) => &mut m.probe,
        None => &mut filters[s - n],
    }
}

impl PhraseCursor {
    /// Build from one cursor per distinct member term (see
    /// [`distinct_members`]), their fetched position runs, and their
    /// positional metadata — `(term_meta, inline_position)` per member,
    /// exactly one of the two present — plus each query position's member
    /// (`member_at`) and offset, then seek to the first matching doc.
    pub(super) fn new(
        cursors: Vec<TermCursor>,
        positions: Vec<Bytes>,
        positional: Vec<(Option<TermMeta>, Option<u32>)>,
        member_at: Vec<usize>,
        offsets: Vec<u32>,
    ) -> Result<Self, FtsError> {
        debug_assert!(
            member_at.len() >= 2,
            "single-token phrases degrade to terms"
        );
        debug_assert_eq!(cursors.len(), positions.len());
        debug_assert_eq!(cursors.len(), positional.len());
        debug_assert_eq!(member_at.len(), offsets.len());
        debug_assert!(member_at.iter().all(|&m| m < cursors.len()));
        debug_assert_eq!(offsets.first(), Some(&0), "offsets are normalized");
        debug_assert!(
            offsets.windows(2).all(|w| w[0] < w[1]),
            "offsets are strictly ascending"
        );
        // The phrase's idf counts every query position, a repeated term
        // once per occurrence, summed in query order.
        let idf_sum = member_at
            .iter()
            .fold(0.0f32, |sum, &m| sum + cursors[m].idf_weight);
        let mut min_scaled_bound = f32::INFINITY;
        let members: Vec<PhraseMember> = cursors
            .into_iter()
            .zip(positions)
            .zip(positional)
            .map(|((cursor, positions), (term_meta, inline_position))| {
                // An unscored member (built to match, not rank) has no idf
                // and no bound to scale.
                if cursor.idf_weight > 0.0 {
                    min_scaled_bound =
                        min_scaled_bound.min(cursor.term_max_bm25 / cursor.idf_weight);
                }
                PhraseMember {
                    probe: cursor.clone(),
                    cursor,
                    positions,
                    term_meta,
                    inline_position,
                    pos_scratch: Vec::new(),
                    group_index: GroupIndex::default(),
                    group_block: NO_BLOCK_CACHED,
                    tfs: Vec::new(),
                    tfs_block: NO_BLOCK_CACHED,
                }
            })
            .collect();
        // Rarest-first probe order for alignment: the shortest list first.
        // Query order is preserved in `members`.
        let n_members = members.len();
        let mut align_order: Vec<usize> = (0..members.len()).collect();
        align_order.sort_by_key(|&i| members[i].cursor.df);
        let mut rank = vec![0usize; n_members];
        for (r, &m) in align_order.iter().enumerate() {
            rank[m] = r;
        }
        let mut verify_order: Vec<usize> = (0..member_at.len()).collect();
        verify_order.sort_by_key(|&q| rank[member_at[q]]);
        let batch_order = align_order.clone();
        let mut cursor = Self {
            idf_weight: idf_sum,
            // An unscored phrase has no idf, so no bound either.
            term_max_bm25: match idf_sum > 0.0 {
                true => idf_sum * min_scaled_bound,
                false => 0.0,
            },
            members,
            member_at,
            position_offsets: offsets,
            align_order,
            verify_order,
            current_doc: 0,
            current_tf: 0,
            verify_scratch: Vec::new(),
            cands: Vec::new(),
            cand_next: 0,
            cand_locs: vec![Vec::new(); n_members],
            kept: Vec::new(),
            refill_from: Some(0),
            filters: Vec::new(),
            batch_order,
            filter_locs: Vec::new(),
        };
        cursor.seek_match_unranked(0)?;
        Ok(cursor)
    }

    /// Narrow the phrase to the docs that also hold every one of `terms`,
    /// which then filter each batch with the members, and move to the first
    /// such match. The rarest of the members and the terms drives the
    /// batches, so a rare term confines position work to its own docs and a
    /// rare phrase is not slowed by a common term. Call once, before the
    /// cursor is walked.
    pub(super) fn filter_by(&mut self, terms: Vec<TermCursor>) -> Result<(), FtsError> {
        debug_assert!(self.filters.is_empty(), "filters are attached once");
        // No match lies before the phrase's first, unfiltered one.
        let first = self.current_doc;
        // Back to the start: the first batch moved the driver's cursor and
        // the other members' probes, and left the driver's probe and the
        // other members' cursors where they began.
        let driver = self.align_order[0];
        for (i, m) in self.members.iter_mut().enumerate() {
            match i == driver {
                true => m.cursor = m.probe.clone(),
                false => m.probe = m.cursor.clone(),
            }
        }
        let n = self.members.len();
        self.filters = terms;
        let lens: Vec<u64> = self
            .members
            .iter()
            .map(|m| m.cursor.df)
            .chain(self.filters.iter().map(|c| c.df))
            .collect();
        self.batch_order = (0..n + self.filters.len()).collect();
        self.batch_order.sort_by_key(|&s| lens[s]);
        self.cands.clear();
        self.cand_next = 0;
        self.refill_from = Some(0);
        if self.is_exhausted() {
            return Ok(());
        }
        self.seek_match_unranked(first)
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
        self.seek_match(target, bar, Some(dl_norm_k1))
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

    /// Unranked (match/count) alignment to the next *verified* phrase match
    /// ≥ `from`: the block-batched walk of [`Self::seek_match`] with no bar,
    /// so no tf is compared and no doc is pre-screened.
    pub(super) fn seek_match_unranked(&mut self, from: u32) -> Result<(), FtsError> {
        self.seek_match(from, f32::NEG_INFINITY, None)
    }

    /// Advance to the first verified phrase match at doc ≥ `from`, a block
    /// of the rarest source (member or filter term) at a time. That source's
    /// current block is the candidate list; every other source filters it in
    /// one pass ([`TermCursor::retain_contained_located`]: a rank on a dense
    /// block, one decode and a merge on a packed block), a member recording
    /// where it holds each survivor, and only the survivors are verified,
    /// each member's positions read at its recorded location. The candidates left
    /// over after a match are kept for the next call. When `bar` is
    /// finite, a block whose phrase bound is under the bar is skipped
    /// before it is filtered, and a survivor is pre-screened without
    /// touching positions: the phrase tf can't exceed any member's tf, so
    /// the BM25 score at the members' minimum tf bounds its contribution,
    /// and a doc strictly below `bar` is passed over. (`<`, not `<=`: a doc
    /// exactly at the bar can still displace the incumbent kth-best on the
    /// ascending-doc-id tie-break, so it must be verified.)
    pub(super) fn seek_match(
        &mut self,
        from: u32,
        bar: f32,
        dl_norm_k1: Option<&NormTable>,
    ) -> Result<(), FtsError> {
        debug_assert!(
            bar == f32::NEG_INFINITY || dl_norm_k1.is_some(),
            "a finite bar needs the norms"
        );
        let bar_norm = dl_norm_k1.filter(|_| bar > f32::NEG_INFINITY);
        while self.cand_next < self.cands.len() && self.cands[self.cand_next] < from {
            self.cand_next += 1;
        }
        let driver = self.batch_order[0];
        let n_members = self.members.len();
        loop {
            while self.cand_next < self.cands.len() {
                let c = self.cand_next;
                let s = self.cands[c];
                self.cand_next += 1;
                // Every member holds `s` at a known (block, pair): its tf and
                // positions are read there, with no cursor walked to it.
                if let Some(norm) = bar_norm {
                    let mut min_tf = u32::MAX;
                    for (m, locs) in self.members.iter_mut().zip(&self.cand_locs) {
                        let (b, p) = locs[c];
                        min_tf = min_tf.min(m.located_tf(b as usize, p as usize));
                    }
                    let ub = bm25::score_with_dl_norm_k1(self.idf_weight, min_tf, norm.get(s));
                    if ub < bar {
                        continue;
                    }
                }
                let tf = self.verify_located(c)?;
                if tf > 0 {
                    self.current_doc = s;
                    self.current_tf = tf;
                    return Ok(());
                }
            }

            // Refill from the driver's next block at or after `from`.
            let Some(refill) = self.refill_from else {
                self.current_doc = u32::MAX;
                self.current_tf = 0;
                return Ok(());
            };
            let start = refill.max(from);
            let d = source_cursor(&mut self.members, &mut self.filters, driver, true);
            d.skip_to(start);
            if d.is_exhausted() {
                self.refill_from = None;
                self.current_doc = u32::MAX;
                self.current_tf = 0;
                return Ok(());
            }
            // A skip into a dense block may hold just the one doc it landed
            // on; the batch needs the whole block.
            d.hold_whole_block();
            let first = d.current_doc_id();
            let block_last = d.current_block_last_doc_id();
            self.refill_from = block_last.checked_add(1);
            // Block-level bound: no doc of this block can score above the
            // smallest member block max over its range scaled to the
            // phrase idf. Under the bar, the block is skipped whole before
            // any of its docs is aligned or its positions read.
            if bar_norm.is_some() && self.block_max_in_range(first, block_last) < bar {
                continue;
            }
            let d = source_cursor(&mut self.members, &mut self.filters, driver, true);
            self.cands.clear();
            self.cands
                .extend_from_slice(&d.block_doc_ids[d.pos..d.block_n]);
            self.cand_next = 0;
            // A driving member's candidates sit at their own pair indices in
            // its decoded block; each other member's filter records where it
            // holds the survivors, and the earlier members' locations are
            // compacted to the same survivors. A filter term's locations are
            // not needed: it only removes candidates.
            let (b, pos0) = (d.current_block as u32, d.pos as u32);
            for locs in &mut self.cand_locs {
                locs.clear();
            }
            if driver < n_members {
                self.cand_locs[driver].extend((0..self.cands.len() as u32).map(|i| (b, pos0 + i)));
            }
            for oi in 1..self.batch_order.len() {
                if self.cands.is_empty() {
                    break;
                }
                let si = self.batch_order[oi];
                let mut locs = match si < n_members {
                    true => take(&mut self.cand_locs[si]),
                    false => take(&mut self.filter_locs),
                };
                source_cursor(&mut self.members, &mut self.filters, si, false)
                    .retain_contained_located(&mut self.cands, &mut locs, &mut self.kept);
                match si < n_members {
                    true => self.cand_locs[si] = locs,
                    false => self.filter_locs = locs,
                }
                for &pj in &self.batch_order[..oi] {
                    if pj < n_members {
                        compact_locs(&mut self.cand_locs[pj], &self.kept);
                    }
                }
            }
        }
    }

    /// [`Self::verify_at_aligned`] for candidate `c` of the current batch,
    /// reading each member's positions at its recorded location rather than
    /// through a walk cursor.
    fn verify_located(&mut self, c: usize) -> Result<u32, FtsError> {
        let locs = &self.cand_locs;
        self.current_tf = verify_members(
            &mut self.members,
            &self.member_at,
            &self.verify_order,
            &self.position_offsets,
            &mut self.verify_scratch,
            |m, j| {
                let (b, p) = locs[j][c];
                m.located_positions(b as usize, p as usize)
            },
        )?;
        Ok(self.current_tf)
    }

    /// Count the phrase's anchors at the members' aligned doc: the
    /// first member's positions `p` where member `i` also has `p + i`
    /// for every `i`. Member position lists are ascending, so each
    /// probe is a binary search over a per-doc-tf-sized slice.
    pub(super) fn verify_at_aligned(&mut self, aligned: u32) -> Result<u32, FtsError> {
        // Staged, rarest first: see [`verify_members`].
        //
        // `materialize_at` decodes each member's block only now: on the
        // unranked path a common member reached `aligned` by a `contains`
        // bit-test and its block is not yet decoded; on the ranked path a
        // `skip_to` into a bitset block holds just the one doc it landed on,
        // and `decode_current_positions` needs the whole block (its pair
        // index and tf run). Only a plain walk leaves the block decoded,
        // so these calls are not optional.
        self.current_tf = verify_members(
            &mut self.members,
            &self.member_at,
            &self.verify_order,
            &self.position_offsets,
            &mut self.verify_scratch,
            |m, _| {
                m.cursor.materialize_at(aligned);
                m.decode_current_positions()
            },
        )?;
        Ok(self.current_tf)
    }

    /// Score the phrase at its current doc with the caller-supplied
    /// per-doc BM25 normalization.
    #[inline]
    pub(super) fn score_current(&self, dl_norm_k1: f32) -> f32 {
        bm25::score_with_dl_norm_k1(self.idf_weight, self.current_tf, dl_norm_k1)
    }

    /// Phrase-scaled block-level upper bound over `[range_start,
    /// range_end]` — the block analog of `term_max_bm25`.
    pub(super) fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        // Each member's bound is divided by its idf, which an unscored
        // phrase does not have; only ranked walks ask for this bound.
        debug_assert!(
            self.idf_weight > 0.0,
            "an unscored phrase has no block bound"
        );
        let mut min_scaled = f32::INFINITY;
        for m in self.members.iter_mut() {
            let b = m.cursor.block_max_in_range(range_start, range_end);
            min_scaled = min_scaled.min(b / m.cursor.idf_weight);
        }
        self.idf_weight * min_scaled
    }
}

#[cfg(test)]
thread_local! {
    /// Test switch: keep every shape on the two-phase walk, the reference
    /// the batched walk is graded against bit for bit.
    static TWO_PHASE_ONLY: Cell<bool> = const { Cell::new(false) };
}

/// Align every atom of an AND by its approximation, from `target`: a term
/// atom is exact, a phrase atom advances only to a doc holding all its
/// members, without decoding positions. The first doc every approximation
/// agrees on, or `None` once an atom is exhausted.
pub(super) fn approx_align(atoms: &mut [AnyCursor], target: u32) -> Option<u32> {
    let mut aligned = target;
    let mut i = 0usize;
    while i < atoms.len() {
        let a = &mut atoms[i];
        a.approx_skip_to(aligned);
        if a.is_exhausted() {
            return None;
        }
        let here = a.approx_current_doc();
        if here > aligned {
            aligned = here;
            i = 0;
            continue;
        }
        i += 1;
    }
    Some(aligned)
}

/// Hand the AND of exactly one phrase and at least one plain term to the
/// phrase's batched walk: the terms become the phrase's filters
/// ([`PhraseCursor::filter_by`]), so the phrase atom alone walks the
/// intersection, verified, and the term atoms are left in place for scoring.
/// Returns the phrase atom's index, or `None` for any other shape, which
/// keeps the two-phase walk.
pub(super) fn batch_phrase_with_terms(atoms: &mut [AnyCursor]) -> Result<Option<usize>, FtsError> {
    #[cfg(test)]
    if TWO_PHASE_ONLY.with(Cell::get) {
        return Ok(None);
    }
    let mut phrases = atoms
        .iter()
        .enumerate()
        .filter(|(_, a)| matches!(a, AnyCursor::Phrase(_)))
        .map(|(i, _)| i);
    let (Some(at), None) = (phrases.next(), phrases.next()) else {
        return Ok(None);
    };
    let terms: Vec<TermCursor> = atoms
        .iter()
        .filter_map(|a| match a {
            AnyCursor::Term(c) => Some(c.clone()),
            AnyCursor::Phrase(_) => None,
        })
        .collect();
    match (terms.is_empty(), &mut atoms[at]) {
        (false, AnyCursor::Phrase(p)) => {
            p.filter_by(terms)?;
            Ok(Some(at))
        }
        _ => Ok(None),
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

    /// The most docs the atom can sit on: a term's document frequency, a
    /// phrase's rarest member's, since every doc a phrase verifies holds
    /// all of its members.
    pub(super) fn df_bound(&self) -> u64 {
        match self {
            AnyCursor::Term(c) => c.df,
            AnyCursor::Phrase(p) => p.members.iter().map(|m| m.cursor.df).min().unwrap_or(0),
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
                bm25::score_with_dl_norm_k1(c.idf_weight, c.current_tf(), dl_norm_k1)
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

    use std::{slice, sync::Arc};

    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use roaring::RoaringBitmap;

    use super::{super::test_util::*, *};
    use crate::{
        superfile::{
            fts::{
                builder::FtsBuilder,
                reader::{BoolMode, FtsReader, QueryIdf, core::ClauseLists, cursor::CursorUse},
                tokenize::Phrase,
            },
            id_space::{FtsDocId, RowId},
        },
        test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
    };

    /// [`TWO_PHASE_ONLY`] set while it lives, and cleared when it drops,
    /// so a failed assertion cannot leave it set for the next test on the
    /// thread.
    struct TwoPhaseOnly;

    impl TwoPhaseOnly {
        fn set(on: bool) -> Self {
            TWO_PHASE_ONLY.with(|c| c.set(on));
            Self
        }
    }

    impl Drop for TwoPhaseOnly {
        fn drop(&mut self) {
            TWO_PHASE_ONLY.with(|c| c.set(false));
        }
    }

    fn phrase(terms: &[&str]) -> Vec<Phrase<String>> {
        vec![Phrase::adjacent(
            terms.iter().map(|t| t.to_string()).collect(),
        )]
    }

    /// A multi-block positional term's groups are decoded whole and
    /// indexed by tf prefix sums. Plant a phrase in every one of 300 docs
    /// at varying in-block pair slots (three blocks, tf varying so the
    /// runs differ in length) and verify every doc through the phrase
    /// path.
    #[tokio::test]
    async fn grouped_positions_reach_every_pair_across_blocks() {
        use crate::superfile::fts::builder::FtsBuilder;
        let n_docs = 300u32;
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        for i in 0..n_docs {
            let text = format!(
                "{}{}alpha beta",
                "alpha ".repeat((i % 7) as usize),
                "filler ".repeat((i % 50) as usize)
            );
            b.add_doc(0, i, &text).expect("doc");
        }
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let phrases = phrase(&["alpha", "beta"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                n_docs as usize + 1,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            (0..n_docs).collect::<Vec<_>>(),
            "every doc holds the phrase"
        );
        // No doc holds "beta alpha".
        let reversed = phrase(&["beta", "alpha"]);
        let none = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &reversed,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        assert!(none.is_empty());
    }

    /// Position groups pick packed or LEB128 per block by size. Plant a
    /// term whose first block has small gaps (packed) and whose second
    /// block carries an outlier gap (LEB128 wins), plus a short-form
    /// member with tf > 1 (one packed group), and verify the phrase in
    /// every doc through both decode paths.
    #[tokio::test]
    async fn position_groups_verify_phrases_in_long_and_short_terms() {
        use crate::superfile::fts::{builder::FtsBuilder, posting::BLOCK_LEN};
        const N_DOCS: u32 = 200;
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        for d in 0..N_DOCS {
            let alpha: Vec<u32> = match d < BLOCK_LEN as u32 {
                true => vec![0, 5, 10],
                false => vec![0, 1 << 20],
            };
            let beta: Vec<u32> = alpha.iter().map(|p| p + 1).collect();
            b.add_prebuilt_term_posting(0, "alpha", FtsDocId::new(d), alpha.len() as u32, &alpha)
                .expect("alpha");
            b.add_prebuilt_term_posting(0, "beta", FtsDocId::new(d), beta.len() as u32, &beta)
                .expect("beta");
            if d < 3 {
                // Short-form member (df = 3) with two positions each.
                let rare = [beta[0] + 1, beta[0] + 7];
                b.add_prebuilt_term_posting(0, "rare", FtsDocId::new(d), 2, &rare)
                    .expect("rare");
            }
        }
        b.append_prebuilt_doc_lengths(0, &vec![(1 << 20) + 8; N_DOCS as usize]);
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");

        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrase(&["alpha", "beta"]),
                    ..ClauseLists::default()
                },
                N_DOCS as usize + 1,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..N_DOCS).collect::<Vec<_>>());

        let rare_hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrase(&["beta", "rare"]),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let mut ids: Vec<u32> = rare_hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1, 2], "short-form member's packed group");
    }

    /// Ranked phrase walks prune whole blocks of the rarest member whose
    /// block-level phrase bound is under the bar. Plant "the movement" in
    /// every 13th doc, twice per doc in every fifth run of eight blocks
    /// and once elsewhere, so a small k fills the heap from the doubled
    /// runs and the rarest member's blocks inside the single-occurrence
    /// runs fall under the bar; near-misses
    /// ("movement the") keep the members aligning on docs that fail to
    /// verify. Every k must match the unpruned walk's top-k.
    #[tokio::test]
    async fn ranked_phrase_block_pruning_agrees_with_the_unpruned_walk() {
        use crate::superfile::fts::{builder::FtsBuilder, posting::BLOCK_LEN};
        const N_DOCS: u32 = BLOCK_LEN as u32 * 40;
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        for d in 0..N_DOCS {
            let hot = (d / (8 * BLOCK_LEN as u32)).is_multiple_of(5);
            let text = match (d % 13, hot) {
                (0, true) => "the movement of the movement",
                (0, false) => "the movement of the people",
                (5, _) => "movement the people of",
                _ => "the people of the town",
            };
            b.add_doc(0, d, text).expect("add doc");
        }
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let phrases = phrase(&["the", "movement"]);
        let clauses = || ClauseLists {
            should_phrases: &phrases,
            ..ClauseLists::default()
        };
        // With k above the match count the heap never fills, so the bar
        // stays at negative infinity and nothing is pruned: the oracle.
        let mut all = r
            .search_excluding("title", clauses(), N_DOCS as usize + 1, f32::NEG_INFINITY)
            .await
            .expect("unpruned");
        assert_eq!(
            all.len(),
            (N_DOCS as usize).div_ceil(13),
            "every 13th doc matches"
        );
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for k in [1usize, 3, 10, 50, 100, 200] {
            let pruned = r
                .search_excluding("title", clauses(), k, f32::NEG_INFINITY)
                .await
                .expect("pruned");
            assert_eq!(pruned.len(), k, "k={k}");
            for (i, ((dp, sp), (da, sa))) in pruned.iter().zip(all.iter()).enumerate() {
                assert_eq!(
                    dp, da,
                    "doc mismatch k={k} rank {i}: pruned={dp} oracle={da}"
                );
                assert!(
                    (sp - sa).abs() < 1e-4,
                    "score mismatch k={k} rank {i}: {sp} vs {sa}"
                );
            }
        }
    }

    /// The block-batched ranked walk and the doc-at-a-time unranked walk
    /// must visit the same verified matches with the same tfs, from any
    /// mix of consecutive advances and skips. The rarest member's blocks
    /// each span dozens of the other members' blocks (a bitset one and a
    /// packed one), so a batch is filtered across many member blocks and
    /// its survivors are verified in blocks the probe has long left. With
    /// a finite bar the ranked walk must still yield every match at or
    /// above it.
    #[tokio::test]
    async fn batched_ranked_phrase_walk_matches_the_unranked_walk() {
        use rand::{RngExt, SeedableRng, rngs::StdRng};

        use crate::superfile::fts::{bm25, builder::FtsBuilder, posting::BLOCK_LEN};
        const N_DOCS: u32 = BLOCK_LEN as u32 * 60;
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        for d in 0..N_DOCS {
            let text = match (d % 53, d % 2, d % 3) {
                (0, 0, _) if d.is_multiple_of(5) => "the mid rare the mid rare",
                (0, 0, _) => "the mid rare",
                (0, _, _) => "rare the mid",
                (_, _, 0) => "the mid",
                _ => "the x",
            };
            b.add_doc(0, d, text).expect("add doc");
        }
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let phrases = phrase(&["the", "mid", "rare"]);
        let build = || async {
            let (mut atoms, _) = r
                .build_atom_cursors(0, &[], &phrases, QueryIdf::ThisFile, None, CursorUse::Score)
                .await
                .expect("atoms");
            match atoms.remove(0).expect("phrase present") {
                AnyCursor::Phrase(c) => c,
                AnyCursor::Term(_) => unreachable!("a phrase atom"),
            }
        };
        let dl_norm = r.columns[0].dl_norm_k1();

        // Every match, by both walks, consecutively.
        let mut unranked = build().await;
        let mut expected = Vec::new();
        while !unranked.is_exhausted() {
            expected.push((unranked.current_doc_id(), unranked.current_tf));
            unranked
                .skip_to(unranked.current_doc_id() + 1)
                .expect("skip");
        }
        assert_eq!(
            expected.len(),
            (N_DOCS as usize).div_ceil(106),
            "every 106th doc matches"
        );
        let mut ranked = build().await;
        let mut got = Vec::new();
        while !ranked.is_exhausted() {
            got.push((ranked.current_doc_id(), ranked.current_tf));
            ranked
                .skip_to_pruned(ranked.current_doc_id() + 1, f32::NEG_INFINITY, dl_norm)
                .expect("skip");
        }
        assert_eq!(got, expected, "consecutive ranked walk");

        // Random skips: both walks land on the same next match.
        let mut rng = StdRng::seed_from_u64(7);
        for trial in 0..20 {
            let mut a = build().await;
            let mut bb = build().await;
            let mut target = 0u32;
            while !a.is_exhausted() {
                assert_eq!(
                    (a.current_doc_id(), a.current_tf),
                    (bb.current_doc_id(), bb.current_tf),
                    "trial {trial} target {target}"
                );
                target = a.current_doc_id() + 1 + rng.random_range(0..700u32);
                a.skip_to(target).expect("skip");
                bb.skip_to_pruned(target, f32::NEG_INFINITY, dl_norm)
                    .expect("skip");
            }
            assert!(
                bb.is_exhausted(),
                "trial {trial}: ranked walk must exhaust too"
            );
        }

        // With a bar, the ranked walk yields exactly the matches whose
        // score is not below it (the tf-2 docs), in order.
        let bar_doc = expected
            .iter()
            .find(|(_, tf)| *tf == 2)
            .map(|(d, _)| *d)
            .expect("a tf-2 match");
        let mut pruned = build().await;
        let bar = bm25::score_with_dl_norm_k1(pruned.idf_weight, 2, dl_norm.get(bar_doc));
        let above: Vec<(u32, u32)> = expected
            .iter()
            .copied()
            .filter(|(d, tf)| {
                bm25::score_with_dl_norm_k1(pruned.idf_weight, *tf, dl_norm.get(*d)) >= bar
            })
            .collect();
        assert!(
            above.len() >= 5 && above.len() < expected.len(),
            "the bar splits the matches"
        );
        let mut got = Vec::new();
        pruned.skip_to_pruned(0, bar, dl_norm).expect("seek");
        // The constructor's initial seek is unranked; re-seek from 0 with the bar.
        if !pruned.is_exhausted() {
            let first = pruned.current_doc_id();
            if bm25::score_with_dl_norm_k1(pruned.idf_weight, pruned.current_tf, dl_norm.get(first))
                >= bar
            {
                got.push((first, pruned.current_tf));
            }
        }
        while !pruned.is_exhausted() {
            let next = pruned.current_doc_id() + 1;
            pruned.skip_to_pruned(next, bar, dl_norm).expect("skip");
            if !pruned.is_exhausted() {
                got.push((pruned.current_doc_id(), pruned.current_tf));
            }
        }
        assert_eq!(got, above, "ranked walk under a bar");
    }

    /// A block whose position runs are very long (600 occurrences per doc,
    /// 599 unit gaps then one huge one that becomes a stream exception)
    /// still packs as one group and verifies. Postings are planted
    /// directly so the fixture costs no tokenization.
    #[tokio::test]
    async fn a_block_with_very_long_runs_verifies_phrases() {
        use crate::superfile::fts::builder::FtsBuilder;
        const N_DOCS: u32 = 129;
        const TF: u32 = 600;
        const OUTLIER_GAP: u32 = 1 << 25;
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        let alpha: Vec<u32> = (0..TF)
            .map(|j| match j + 1 == TF {
                true => 2 * j + OUTLIER_GAP,
                false => 2 * j,
            })
            .collect();
        let filler: Vec<u32> = alpha.iter().map(|p| p + 1).collect();
        for d in 0..N_DOCS {
            b.add_prebuilt_term_posting(0, "alpha", FtsDocId::new(d), TF, &alpha)
                .expect("alpha");
            b.add_prebuilt_term_posting(0, "filler", FtsDocId::new(d), TF, &filler)
                .expect("filler");
        }
        b.append_prebuilt_doc_lengths(0, &vec![2 * TF + OUTLIER_GAP; N_DOCS as usize]);
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");

        let phrases = phrase(&["alpha", "filler"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                N_DOCS as usize + 1,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..N_DOCS).collect::<Vec<_>>());
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 2, 4], "adjacency in order only");
        // Doc 4 has the phrase twice — highest tf, and with uniform
        // doc lengths in play its score must strictly exceed doc 0's
        // (same length, tf 1... doc 0 len 3, doc 4 len 4; tf=2 wins).
        assert_eq!(hits[0].0, 4, "double occurrence ranks first");
    }

    /// An emoji is a token under `standard`, so it occupies a position
    /// and breaks adjacency: `"cat dog"` must not match `cat 🙂 dog`.
    #[tokio::test]
    async fn an_emoji_between_two_words_breaks_the_phrase() {
        use crate::superfile::fts::{builder::FtsBuilder, reader::BoolMode};
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        b.add_doc(0, 0, "cat 🙂 dog").expect("doc 0");
        b.add_doc(0, 1, "cat dog").expect("doc 1");
        b.add_doc(0, 2, "cat, dog").expect("doc 2");
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let phrases = phrase(&["cat", "dog"]);
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
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(
            ids,
            vec![1, 2],
            "punctuation takes no position; an emoji does"
        );
        // And the emoji itself is searchable.
        let emoji = r
            .search("title", &["🙂"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert_eq!(emoji.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![0]);
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
            hits.iter().map(|(d, _)| d.get()).collect::<Vec<_>>(),
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
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), false).expect("register");
        b.add_doc(0, 0, "new york").expect("add doc");
        let blob = Bytes::from(b.finish().expect("finish"));
        let r = FtsReader::open(
            blob,
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
        )
        .expect("open");
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

    // ── Cursor-level phrase seeks ─────────────────────────────────────

    /// Rows in the seek corpus: the common member spans a dozen blocks.
    const SEEK_DOCS: u32 = 3000;
    /// Rows holding the rare member. The first holds the words reversed;
    /// the other two hold the phrase, hundreds of docs (several of the
    /// common member's blocks) apart.
    const RARE_REVERSED: u32 = 5;
    const RARE_MATCH_FIRST: u32 = 700;
    const RARE_MATCH_SECOND: u32 = 1400;

    /// The one phrase cursor `terms` builds on `r`'s first column, for
    /// `purpose`.
    async fn phrase_cursor(r: &FtsReader, terms: &[&str], purpose: CursorUse) -> PhraseCursor {
        phrase_atom(r, &phrase(terms)[0], purpose).await
    }

    /// The one phrase cursor `p` builds on `r`'s first column, for
    /// `purpose`.
    async fn phrase_atom(r: &FtsReader, p: &Phrase<String>, purpose: CursorUse) -> PhraseCursor {
        let (atoms, _) = r
            .build_atom_cursors(
                0,
                &[],
                slice::from_ref(p),
                QueryIdf::ThisFile,
                None,
                purpose,
            )
            .await
            .expect("build atoms");
        match atoms.into_iter().next().flatten() {
            Some(AnyCursor::Phrase(pc)) => pc,
            _ => panic!("expected one phrase atom"),
        }
    }

    fn open_positional(docs: impl Iterator<Item = String>) -> FtsReader {
        let mut b = FtsBuilder::new();
        b.register_column("title".into(), true).expect("register");
        for (i, text) in docs.enumerate() {
            b.add_doc(0, i as u32, &text).expect("add doc");
        }
        let json =
            r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// Phrases whose matching occurrence lies deep in a common member's run,
    /// past any short prefix of it a reader might stop at.
    /// Docs repeat a common word up to ~60 times and hold a rarer word once
    /// or twice at a random depth. Every phrase's (doc, tf) pairs, scored,
    /// must equal a brute-force scan of the text, and the unscored walk must
    /// match the same docs.
    #[tokio::test]
    async fn phrases_deep_in_long_member_runs_match_a_text_scan() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut rng = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        let docs: Vec<Vec<&str>> = (0..700)
            .map(|_| {
                let len = 1 + rng(70) as usize;
                let mut d: Vec<&str> = (0..len)
                    .map(|_| if rng(10) < 7 { "w" } else { "z" })
                    .collect();
                if rng(5) == 0 {
                    for _ in 0..1 + rng(2) {
                        let at = rng(d.len() as u64 + 1) as usize;
                        d.insert(at, "m");
                    }
                }
                d
            })
            .collect();
        let r = open_positional(docs.iter().map(|d| d.join(" ")));
        let norms = r.columns[0].dl_norm_k1();
        let phrases: [&[&str]; 5] = [
            &["w", "m"],
            &["m", "w"],
            &["w", "w", "m"],
            &["m", "w", "w"],
            &["w", "m", "w"],
        ];
        for terms in phrases {
            let expected: Vec<(u32, u32)> = docs
                .iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    let tf = d.windows(terms.len()).filter(|w| *w == terms).count() as u32;
                    (tf > 0).then_some((i as u32, tf))
                })
                .collect();
            assert!(!expected.is_empty(), "{terms:?} occurs");
            let mut scored = phrase_cursor(&r, terms, CursorUse::Score).await;
            let mut seen = Vec::new();
            while !scored.is_exhausted() {
                seen.push((scored.current_doc_id(), scored.current_tf));
                let next = scored.current_doc_id() + 1;
                scored
                    .seek_match(next, f32::NEG_INFINITY, Some(norms))
                    .expect("seek");
            }
            assert_eq!(seen, expected, "scored {terms:?}");
            let mut matched = phrase_cursor(&r, terms, CursorUse::Match).await;
            let mut docs_seen = Vec::new();
            while !matched.is_exhausted() {
                docs_seen.push(matched.current_doc_id());
                let next = matched.current_doc_id() + 1;
                matched.skip_to(next).expect("skip");
            }
            let expected_docs: Vec<u32> = expected.iter().map(|&(d, _)| d).collect();
            assert_eq!(docs_seen, expected_docs, "unscored {terms:?}");
        }
    }

    /// A phrase built to match, not rank, carries no idf and a 0 bound:
    /// dividing its members' bounds by their 0 idf would make the bound NaN.
    #[tokio::test]
    async fn an_unscored_phrase_has_no_idf_and_a_zero_bound() {
        let r = open_positional(["x y", "y x", "x y z"].into_iter().map(String::from));
        let pc = phrase_cursor(&r, &["x", "y"], CursorUse::Match).await;
        assert_eq!(pc.idf_weight, 0.0);
        assert_eq!(pc.term_max_bm25, 0.0);
    }

    /// A phrase whose rare member drives the alignment and whose common
    /// member must be seeked several blocks forward per candidate: the
    /// cursor lands on each verified match and skips the reversed
    /// co-occurrence. A member positioned by its own block scan rather
    /// than a seek to the driver's doc would verify positions against
    /// the wrong block and either miss the match or report a false one.
    #[tokio::test]
    async fn seek_match_positions_a_far_member_by_block_seek() {
        let r = open_positional((0..SEEK_DOCS).map(|d| match d {
            RARE_REVERSED => "y x".to_string(),
            RARE_MATCH_FIRST | RARE_MATCH_SECOND => "x y".to_string(),
            _ => "x".to_string(),
        }));
        let norms = r.columns[0].dl_norm_k1();
        let mut pc = phrase_cursor(&r, &["x", "y"], CursorUse::Score).await;
        assert!(
            pc.members[0].cursor.block_count() > 8,
            "premise: the common member spans many blocks"
        );
        assert_eq!(pc.align_order[0], 1, "premise: the rare member drives");
        assert_eq!(
            pc.current_doc_id(),
            RARE_MATCH_FIRST,
            "construction seeks past the reversed doc"
        );
        assert_eq!(pc.current_tf, 1);
        pc.seek_match(RARE_MATCH_FIRST + 1, f32::NEG_INFINITY, Some(norms))
            .expect("seek");
        assert_eq!(pc.current_doc_id(), RARE_MATCH_SECOND);
        assert_eq!(pc.current_tf, 1);
        pc.skip_to(RARE_MATCH_SECOND + 1).expect("skip");
        assert!(pc.is_exhausted(), "no third match");
    }

    /// A repeated-word phrase counts every occurrence start, overlapping
    /// ones included: `"a a"` has two starts in `a a a`, one in `a a`,
    /// none in `a b a`. Counting non-overlapping runs would report one
    /// for the first doc and misrank it.
    #[tokio::test]
    async fn seek_match_counts_overlapping_starts_of_a_repeated_word() {
        let r = open_positional(
            ["a a a", "a b a", "a a", "b b", "a a a a"]
                .into_iter()
                .map(str::to_string),
        );
        let norms = r.columns[0].dl_norm_k1();
        let mut pc = phrase_cursor(&r, &["a", "a"], CursorUse::Score).await;
        let mut seen: Vec<(u32, u32)> = Vec::new();
        while !pc.is_exhausted() {
            seen.push((pc.current_doc_id(), pc.current_tf));
            let next = pc.current_doc_id() + 1;
            pc.seek_match(next, f32::NEG_INFINITY, Some(norms))
                .expect("seek");
        }
        assert_eq!(seen, vec![(0, 2), (2, 1), (4, 3)]);
    }

    /// A ranked walk under a finite bar pre-screens each candidate by the
    /// score at its members' smallest tf. An inline member (one doc, one
    /// occurrence) has no stored tf to read; it counts as one, so its doc
    /// clears a bar under that score and is verified.
    #[tokio::test]
    async fn a_finite_bar_reads_an_inline_member_as_tf_one() {
        let r = open_positional((0..300).map(|d| match d {
            150 => "x solo".to_string(),
            _ => "x".to_string(),
        }));
        let norms = r.columns[0].dl_norm_k1();
        let mut pc = phrase_cursor(&r, &["x", "solo"], CursorUse::Score).await;
        assert!(
            pc.members[1].inline_position.is_some(),
            "premise: an inline member"
        );
        // Construction verified the one candidate unranked; rewind the batch
        // so the ranked walk reads it again, now under a bar at half its
        // score.
        assert_eq!(
            (pc.current_doc_id(), pc.cands.as_slice()),
            (150, &[150][..])
        );
        let bar = pc.score_current(norms.get(150)) / 2.0;
        pc.cand_next = 0;
        pc.current_doc = 0;
        pc.seek_match(0, bar, Some(norms)).expect("seek");
        assert_eq!((pc.current_doc_id(), pc.current_tf), (150, 1));
    }

    /// A phrase that repeats a term reads one shared member at each of the
    /// term's offsets. Docs are drawn from a few words, mostly the repeated
    /// ones, with the full phrases, overlapping repeats and near-misses
    /// planted among them, across several blocks. For adjacent and gapped
    /// phrases, every walk must agree with a scan of the tokens: the (doc,
    /// tf) pairs of the scored and unscored walks, the match ids and count,
    /// and the ranked scores bit for bit, whose idf counts every query
    /// position, a repeated term once per occurrence.
    #[tokio::test]
    async fn repeated_terms_share_a_member_and_match_a_token_scan() {
        const N_DOCS: usize = 900;
        const PLANTED: [&str; 7] = [
            "to be or not to be",
            "to be to be to be",
            "be to be to",
            "to be or not to x",
            "not to be or to be",
            "to x be to be x to",
            "be or to be or x",
        ];
        const WORDS: [&str; 5] = ["to", "be", "to", "be", "or"];
        let mut next = xorshift(0x2545_F491_4F6C_DD1D);
        let mut rng = move |m: u64| next() % m;
        let docs: Vec<Vec<&str>> = (0..N_DOCS)
            .map(|_| {
                let mut d: Vec<&str> = (0..1 + rng(12))
                    .map(|_| match rng(4) {
                        0 => "not",
                        _ => WORDS[rng(WORDS.len() as u64) as usize],
                    })
                    .collect();
                if rng(3) == 0 {
                    let at = rng(d.len() as u64 + 1) as usize;
                    let planted = PLANTED[rng(PLANTED.len() as u64) as usize];
                    d.splice(at..at, planted.split(' '));
                }
                d
            })
            .collect();
        let r = open_positional(docs.iter().map(|d| d.join(" ")));
        let norms = r.columns[0].dl_norm_k1();
        let gapped = |terms: &[&str], offsets: &[u32]| Phrase {
            terms: terms.iter().map(|t| t.to_string()).collect(),
            offsets: offsets.to_vec(),
        };
        let phrases = [
            Phrase::adjacent(
                ["to", "be", "or", "not", "to", "be"]
                    .map(String::from)
                    .to_vec(),
            ),
            Phrase::adjacent(["to", "be", "to", "be"].map(String::from).to_vec()),
            Phrase::adjacent(["be", "to", "be"].map(String::from).to_vec()),
            Phrase::adjacent(["to", "to"].map(String::from).to_vec()),
            Phrase::adjacent(["be", "be", "be"].map(String::from).to_vec()),
            gapped(&["to", "be", "to"], &[0, 1, 3]),
            gapped(&["be", "to", "be", "x"], &[0, 2, 3, 5]),
            gapped(&["to", "to"], &[0, 2]),
        ];
        for p in &phrases {
            let label = format!("{:?} at {:?}", p.terms, p.offsets);
            let (distinct, _) = distinct_members(p);
            assert!(distinct.len() < p.len(), "premise: {label} repeats a term");
            let expected: Vec<(u32, u32)> = docs
                .iter()
                .enumerate()
                .filter_map(|(i, d)| {
                    let tf = (0..d.len())
                        .filter(|&s| {
                            p.iter()
                                .zip(p.offsets())
                                .all(|(t, &o)| d.get(s + o as usize).is_some_and(|w| w == t))
                        })
                        .count() as u32;
                    (tf > 0).then_some((i as u32, tf))
                })
                .collect();
            assert!(!expected.is_empty(), "{label} occurs");
            let expected_ids: Vec<u32> = expected.iter().map(|&(d, _)| d).collect();
            let one = slice::from_ref(p);

            let mut scored = phrase_atom(&r, p, CursorUse::Score).await;
            assert_eq!(scored.members.len(), distinct.len(), "one member per term");
            let mut seen = Vec::new();
            while !scored.is_exhausted() {
                seen.push((scored.current_doc_id(), scored.current_tf));
                let next = scored.current_doc_id() + 1;
                scored
                    .seek_match(next, f32::NEG_INFINITY, Some(norms))
                    .expect("seek");
            }
            assert_eq!(seen, expected, "scored walk, {label}");

            let mut matched = phrase_atom(&r, p, CursorUse::Match).await;
            let mut docs_seen = Vec::new();
            while !matched.is_exhausted() {
                docs_seen.push(matched.current_doc_id());
                let next = matched.current_doc_id() + 1;
                matched.skip_to(next).expect("skip");
            }
            assert_eq!(docs_seen, expected_ids, "unscored walk, {label}");

            let (ids, _) = r
                .atoms_match_ids("title", &[], one, BoolMode::And)
                .await
                .expect("match ids");
            let ids: Vec<u32> = ids.iter().map(|d| d.get()).collect();
            assert_eq!(ids, expected_ids, "match ids, {label}");
            let (count, _) = r
                .atoms_match_count("title", &[], one, BoolMode::And, &[], &[])
                .await
                .expect("count");
            assert_eq!(count, expected.len() as u64, "count, {label}");

            // The phrase idf, from each query position's own term cursor.
            let mut idf = 0.0f32;
            for t in p.iter() {
                let c = r
                    .build_term_cursors(0, &[t], QueryIdf::ThisFile, CursorUse::Score, None, None)
                    .await
                    .expect("term cursor");
                idf += c[0].idf_weight;
            }
            let mut ranked: Vec<(u32, u32)> = expected
                .iter()
                .map(|&(d, tf)| {
                    (
                        d,
                        bm25::score_with_dl_norm_k1(idf, tf, norms.get(d)).to_bits(),
                    )
                })
                .collect();
            ranked.sort_by(|a, b| {
                f32::from_bits(b.1)
                    .total_cmp(&f32::from_bits(a.1))
                    .then(a.0.cmp(&b.0))
            });
            for k in [1usize, 5, 40, expected.len() + 1] {
                let hits = r
                    .search_excluding(
                        "title",
                        ClauseLists {
                            should_phrases: one,
                            ..ClauseLists::default()
                        },
                        k,
                        f32::NEG_INFINITY,
                    )
                    .await
                    .expect("phrase search");
                let mut got: Vec<(u32, u32)> =
                    hits.iter().map(|(d, s)| (d.get(), s.to_bits())).collect();
                got.sort_by(|a, b| {
                    f32::from_bits(b.1)
                        .total_cmp(&f32::from_bits(a.1))
                        .then(a.0.cmp(&b.0))
                });
                let want = &ranked[..k.min(ranked.len())];
                assert_eq!(got, want, "ranked top-{k}, {label}");
            }
        }
    }

    /// A three-member phrase matches only where all three are adjacent in
    /// order: a doc missing the last member is never a candidate, a doc
    /// holding all three out of order is aligned but fails verification.
    #[tokio::test]
    async fn three_member_phrase_rejects_a_missing_or_misplaced_member() {
        let r = open_positional(
            ["p q r", "p q s", "p r q", "q r p q r", "p q"]
                .into_iter()
                .map(str::to_string),
        );
        let norms = r.columns[0].dl_norm_k1();
        let mut pc = phrase_cursor(&r, &["p", "q", "r"], CursorUse::Score).await;
        let mut seen: Vec<(u32, u32)> = Vec::new();
        while !pc.is_exhausted() {
            seen.push((pc.current_doc_id(), pc.current_tf));
            let next = pc.current_doc_id() + 1;
            pc.seek_match(next, f32::NEG_INFINITY, Some(norms))
                .expect("seek");
        }
        assert_eq!(seen, vec![(0, 1), (3, 1)]);
    }

    /// Docs in the phrase-and-terms corpus: dozens of blocks for the common
    /// words, several for the mid-frequency ones.
    const AND_CORPUS_DOCS: u32 = 6_000;

    /// The short-form term of the phrase-and-terms corpus sits in one doc in
    /// this many.
    const FEW_EVERY: u32 = 997;

    /// Brute-force score tolerance: the scorers share the formula and the
    /// inputs, so this absorbs only summation order and f32 rounding.
    const AND_SCORE_TOLERANCE: f32 = 1e-3;

    /// Docs a pushed-down row set admits: one in this many.
    const ALLOW_EVERY: u32 = 3;

    /// A doc of the phrase-and-terms corpus: chunks drawn independently and
    /// shuffled, so a planted phrase stays adjacent while everything else
    /// lands in a random order. Kept under sixteen tokens, so every length
    /// norm is stored exactly and the brute-force scores agree to rounding.
    fn and_corpus_doc(rng: &mut StdRng, i: u32) -> String {
        let mut chunks: Vec<&str> = Vec::new();
        // The common phrase, sometimes twice; its words, also often apart or
        // reversed, so many docs hold both members without the phrase.
        match rng.random_range(0..8u32) {
            0 | 1 => chunks.push("the who"),
            2 => chunks.push("the who the who"),
            3 => chunks.push("who the"),
            4 => chunks.extend(["the", "who"]),
            _ => {}
        }
        // The rare phrase, and its words reversed.
        match rng.random_range(0..300u32) {
            0..=1 => chunks.push("zephyr onyx"),
            2 => chunks.push("onyx zephyr"),
            _ => {}
        }
        let mut plant = |word: &'static str, one_in: u32| {
            if rng.random_range(0..one_in) == 0 {
                chunks.push(word);
            }
        };
        plant("uk", 150);
        plant("mid", 10);
        plant("neg", 7);
        plant("often", 2);
        plant("often", 3);
        // A term in one doc (inline) and one in a handful (short form).
        if i == AND_CORPUS_DOCS / 2 {
            chunks.push("solo the who");
        }
        if i.is_multiple_of(FEW_EVERY) {
            chunks.push("few the who");
        }
        let fill = rng.random_range(0..4usize);
        chunks.extend(["a", "b", "c", "d"].into_iter().take(fill));
        for j in (1..chunks.len()).rev() {
            chunks.swap(j, rng.random_range(0..=j));
        }
        chunks.join(" ")
    }

    /// Ranked hits as (row, score bits), for exact comparison.
    fn bits(hits: &[(RowId, f32)]) -> Vec<(u32, u32)> {
        hits.iter().map(|(d, s)| (d.get(), s.to_bits())).collect()
    }

    /// One phrase ANDed with plain terms walks the phrase's batched path.
    /// Over a corpus of thousands of docs, with the phrase common and the
    /// terms rare, the reverse, members present but not adjacent, inline
    /// and short-form terms, negations and an allow set, every result must
    /// match a brute-force scan of the text (count, ids, top-k within
    /// rounding) and the two-phase walk bit for bit (ids, scores, counts).
    #[tokio::test]
    async fn a_phrase_anded_with_terms_matches_brute_force_and_the_two_phase_walk() {
        let mut rng = StdRng::seed_from_u64(0x5EED_A11D);
        let texts: Vec<String> = (0..AND_CORPUS_DOCS)
            .map(|i| and_corpus_doc(&mut rng, i))
            .collect();
        let r = open_positional(texts.iter().cloned());
        let refs: Vec<(u64, &str)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i as u64, t.as_str()))
            .collect();
        let oracle = BruteForceBm25::index(&refs, default_tokenizer().as_ref());
        let allow: RoaringBitmap = (0..AND_CORPUS_DOCS).step_by(ALLOW_EVERY as usize).collect();
        let allow = Arc::new(allow);

        type Case = (
            &'static [&'static str],
            &'static [&'static str],
            &'static [&'static str],
            &'static [&'static str],
        );
        // (must terms, must phrase, negated terms, negated phrase)
        let cases: &[Case] = &[
            (&["uk"], &["the", "who"], &[], &[]),
            (&["often"], &["the", "who"], &[], &[]),
            (&["often"], &["zephyr", "onyx"], &[], &[]),
            (&["mid"], &["who", "the"], &[], &[]),
            (&["uk", "often"], &["the", "who"], &[], &[]),
            (&["often", "mid"], &["the", "who"], &[], &[]),
            (&["solo"], &["the", "who"], &[], &[]),
            (&["few"], &["the", "who"], &[], &[]),
            (&["few", "often"], &["the", "who"], &[], &[]),
            (&["often"], &["the", "who"], &["neg"], &[]),
            (
                &["mid", "often"],
                &["the", "who"],
                &["neg"],
                &["zephyr", "onyx"],
            ),
            (&["uk"], &["the", "who"], &["mid"], &[]),
            // A term that is also a member, and members named twice.
            (&["the"], &["the", "who"], &[], &[]),
            (&["often"], &["the", "who", "the"], &[], &[]),
            (&["mid"], &["who", "the", "who"], &["neg"], &[]),
        ];
        for &(terms, members, negs, neg_members) in cases {
            let label = format!("+{members:?} +{terms:?} -{negs:?} -{neg_members:?}");
            let musts: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
            let must_phrases = phrase(members);
            let negatives: Vec<String> = negs.iter().map(|t| t.to_string()).collect();
            let neg_phrases = match neg_members.is_empty() {
                true => Vec::new(),
                false => phrase(neg_members),
            };
            let want = oracle.top_k_atoms(
                &musts,
                &must_phrases,
                &[],
                &[],
                &negatives,
                &neg_phrases,
                AND_CORPUS_DOCS as usize,
            );
            assert!(!want.is_empty(), "{label}: premise: the query matches");
            let mut want_ids: Vec<u32> = want.iter().map(|&(d, _)| d as u32).collect();
            want_ids.sort_unstable();
            let want_allowed = want_ids.iter().filter(|&&d| allow.contains(d)).count() as u64;

            for two_phase in [false, true] {
                let _walk = TwoPhaseOnly::set(two_phase);
                let label = format!("{label} two_phase={two_phase}");
                let (count, _) = r
                    .atoms_match_count(
                        "title",
                        terms,
                        &must_phrases,
                        BoolMode::And,
                        negs,
                        &neg_phrases,
                    )
                    .await
                    .expect("count");
                assert_eq!(count, want.len() as u64, "{label}: unranked count");
                if negs.is_empty() && neg_members.is_empty() {
                    let (ids, _) = r
                        .atoms_match_ids("title", terms, &must_phrases, BoolMode::And)
                        .await
                        .expect("ids");
                    let ids: Vec<u32> = ids.iter().map(|d| d.get()).collect();
                    assert_eq!(ids, want_ids, "{label}: unranked ids");
                }
            }

            for allowed in [None, Some(allow.clone())] {
                let lists = || ClauseLists {
                    musts: terms,
                    negatives: negs,
                    must_phrases: &must_phrases,
                    negative_phrases: &neg_phrases,
                    allow: allowed.clone(),
                    ..ClauseLists::default()
                };
                let label = format!("{label} allow={}", allowed.is_some());
                for k in [1usize, 10, 100, AND_CORPUS_DOCS as usize] {
                    let mut runs = Vec::new();
                    for two_phase in [false, true] {
                        let _walk = TwoPhaseOnly::set(two_phase);
                        let ranked = r
                            .search_excluding("title", lists(), k, f32::NEG_INFINITY)
                            .await
                            .expect("ranked");
                        let (counted, count) = r
                            .search_excluding_with_count("title", lists(), k)
                            .await
                            .expect("exact count");
                        runs.push((bits(&ranked), bits(&counted), count));
                    }
                    assert_eq!(runs[0], runs[1], "{label} k={k}: batched vs two-phase");
                    let (ranked, counted, count) = &runs[0];
                    assert_eq!(ranked, counted, "{label} k={k}: ranked vs exact-count hits");
                    match allowed.is_some() {
                        true => {
                            assert_eq!(*count, want_allowed, "{label} k={k}: allowed count");
                            assert!(
                                ranked.iter().all(|&(d, _)| allow.contains(d)),
                                "{label} k={k}: a hit outside the allow set"
                            );
                        }
                        false => {
                            assert_eq!(*count, want.len() as u64, "{label} k={k}: exact count");
                            assert_eq!(ranked.len(), k.min(want.len()), "{label} k={k}: hits");
                            for (i, (&(_, got), &(_, w))) in ranked.iter().zip(&want).enumerate() {
                                assert!(
                                    (f32::from_bits(got) - w).abs() <= AND_SCORE_TOLERANCE,
                                    "{label} k={k}: rank {i} score vs brute force {w}"
                                );
                            }
                            if k >= want.len() {
                                let mut got: Vec<u32> = ranked.iter().map(|&(d, _)| d).collect();
                                got.sort_unstable();
                                assert_eq!(got, want_ids, "{label} k={k}: full result set");
                            }
                        }
                    }
                }
            }
        }

        // Batched musts scored with optional words, unfloored and floored at
        // the k-th best score, so the batched walk prunes against a finite
        // bar less what the other atoms could add.
        let shoulds = ["a", "b"];
        let should_words: Vec<String> = shoulds.iter().map(|t| t.to_string()).collect();
        for (terms, members) in [
            (&["often"][..], &["the", "who"][..]),
            (&["the"], &["the", "who", "the"]),
        ] {
            let label = format!("+{members:?} +{terms:?} {shoulds:?}");
            let musts: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
            let must_phrases = phrase(members);
            let want = oracle.top_k_atoms(
                &musts,
                &must_phrases,
                &should_words,
                &[],
                &[],
                &[],
                AND_CORPUS_DOCS as usize,
            );
            assert!(want.len() > 100, "{label}: premise: many matches");
            let lists = || ClauseLists {
                musts: terms,
                shoulds: &shoulds,
                must_phrases: &must_phrases,
                ..ClauseLists::default()
            };
            for k in [10usize, 100] {
                for floor in [f32::NEG_INFINITY, want[k - 1].1] {
                    let label = format!("{label} k={k} floor={floor}");
                    let mut runs = Vec::new();
                    for two_phase in [false, true] {
                        let _walk = TwoPhaseOnly::set(two_phase);
                        let ranked = r
                            .search_excluding("title", lists(), k, floor)
                            .await
                            .expect("ranked");
                        runs.push(bits(&ranked));
                    }
                    assert_eq!(runs[0], runs[1], "{label}: batched vs two-phase");
                    if floor == f32::NEG_INFINITY {
                        assert_eq!(runs[0].len(), k, "{label}: hits");
                        for (i, (&(_, got), &(_, w))) in runs[0].iter().zip(&want).enumerate() {
                            assert!(
                                (f32::from_bits(got) - w).abs() <= AND_SCORE_TOLERANCE,
                                "{label}: rank {i} score vs brute force {w}"
                            );
                        }
                    }
                }
            }
        }
    }
}
