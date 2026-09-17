// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Low-level FTS posting cursors: the parsed per-term header/skip table
//! ([`TermMeta`], [`BlockMeta`]) and the block-at-a-time [`TermCursor`]
//! the scorers, phrase walk, and count kernels drive. Scoped `pub(super)`
//! to the `reader/` module — never referenced outside the FTS layer.

use std::{ops::Range, sync::Arc};

use bytes::Bytes;

use super::{
    bounds::{BoundDecoder, StoredBound},
    core::{read_u32_le, read_u64_le},
    metadata::ColumnMeta,
};
use crate::superfile::{
    ReadError,
    error::FtsError,
    format::{
        self,
        fts::{
            BlockLayout, POSITION_SUBINDEX_ENTRIES_PER_BLOCK, POSITION_SUBINDEX_STRIDE, SkipLayout,
            U32_BYTES, U64_BYTES, coarse_slot, skip_entry, term_meta,
        },
    },
    fts::{
        bm25,
        builder::{TERM_META_POSITIONAL_SIZE, TERM_META_SIZE},
        posting::{
            BLOCK_LEN, BlockHeader, ENCODING_BITSET, block_encoding, decode_block,
            decode_block_doc_ids, decode_block_tfs,
        },
        short::decode_short,
    },
};

/// How a blob lays out the position run-offset sub-index each
/// positional long-form term carries between its skip table and its
/// blocks — decided by the blob version at open.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum SubindexKind {
    /// `V1`/`V2`: no sub-index; the phrase decode walks a block's runs
    /// from the block start. Also `V7`+, whose grouped positions are
    /// decoded whole per block and indexed by tf prefix sums.
    None,
    /// `V3`–`V6`: `u32` offsets absolute within the term's positions.
    Wide,
}

impl SubindexKind {
    /// Bytes one sub-index entry occupies (zero when there is none).
    pub(super) fn entry_bytes(self) -> usize {
        match self {
            Self::None => 0,
            Self::Wide => U32_BYTES,
        }
    }
}

/// Parsed per-(column, term) metadata header from the postings
/// region. The byte layout is documented once, on the writer side —
/// see [`TERM_META_SIZE`] in `builder.rs` — this struct is its
/// read-side mirror and must stay in sync with that doc.
///
/// [`TermMeta::parse`] is the single place that validates untrusted
/// offsets (the FST value points here) against the postings region:
/// both the fixed 20-byte header and the skip table it declares are
/// bounds-checked before any caller touches a byte. Both the
/// single-term BMW path and [`TermCursor::new`] go through here, so
/// the header layout is interpreted in exactly one spot.
#[derive(Debug, Copy, Clone)]
pub(super) struct TermMeta {
    /// Document frequency — number of docs containing the term.
    pub(super) df: u64,
    /// Number of PFOR blocks (= number of skip-table entries).
    pub(super) num_blocks: usize,
    /// Absolute offset (within the postings region) of the first
    /// skip-table entry: `metadata_offset + TERM_META_SIZE`.
    pub(super) skip_start: usize,
    /// This term's byte offset in the positions region (positional
    /// columns; zero otherwise).
    pub(super) positions_offset: u64,
    /// Byte length of this term's position runs (positional columns;
    /// zero otherwise).
    pub(super) positions_length: u32,
    /// Absolute offset (within the postings region) of this term's
    /// position run-offset sub-index — the block of
    /// `num_blocks × ENTRIES_PER_BLOCK` `u32`s sitting right after the
    /// skip table on a `VERSION_V3` positional term. `None` on
    /// `V1`/`V2` (no sub-index) and on positionless terms.
    pub(super) subindex_start: Option<usize>,
    /// The sub-index entry layout `subindex_start` points at.
    pub(super) subindex: SubindexKind,
    /// Absolute offset (within the postings region) of the coarse
    /// block-max table — `ceil(num_blocks / COARSE_BLOCK_MAX_SPAN)`
    /// 4-byte slots at the tail of the term region.
    pub(super) coarse_start: usize,
    /// Term-relative end of the last posting block: `postings_length`
    /// minus the coarse table's bytes. The blocks end here; the coarse
    /// table follows.
    pub(super) blocks_end_in_term: usize,
    /// Whether this term carries a coarse block-max table (V5 and
    /// later). `false` for V1–V4 — the ranked walk then skips the coarse
    /// level.
    pub(super) has_coarse: bool,
    /// A short-form term (`fts::short`, V7): one block, no header, skip
    /// table, sub-index or coarse table in its bytes. Only `df`,
    /// `num_blocks == 1` and the positions fields are meaningful; the
    /// per-block accessors that read the skip table must not be called.
    /// Its one block's position runs start at offset 0 of the term's
    /// positions bytes.
    pub(super) short: bool,
    /// Whether the term's positions are per-block **groups** behind a
    /// one-byte width header (`V7`) rather than bare LEB128 runs.
    pub(super) positions_grouped: bool,
    /// Which header the term's blocks carry.
    pub(super) block_layout: BlockLayout,
    /// How the skip table locates the blocks.
    pub(super) skip: SkipLayout,
    /// Bytes per skip entry under `skip` on this column.
    skip_entry_bytes: usize,
}

impl TermMeta {
    /// Parse + bounds-validate the header and its skip table.
    /// Returns `Err` (never panics) on a corrupt or malicious
    /// `metadata_offset` — the crate-wide "untrusted input yields
    /// `Err`, not a slice-index panic" rule.
    pub(super) fn parse(
        postings: &[u8],
        metadata_offset: usize,
        positional: bool,
        subindex: SubindexKind,
        stored: StoredBound,
        positions_grouped: bool,
    ) -> Result<Self, FtsError> {
        let has_coarse = stored.has_coarse();
        let skip = stored.skip_layout();
        let skip_entry_bytes = skip.entry_bytes(positional);
        // Positional columns carry the extended 32-byte header (the
        // term's positions offset + length after `num_blocks`); the
        // skip table starts after whichever stride applies. The
        // positions fields themselves are consumed by the phrase read
        // path, not here.
        let term_meta_size = match positional {
            true => TERM_META_POSITIONAL_SIZE,
            false => TERM_META_SIZE,
        };
        if metadata_offset + term_meta_size > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "term metadata offset out of postings region".into(),
            )));
        }
        let df = read_u32_le(
            &postings[metadata_offset + term_meta::DF_OFF
                ..metadata_offset + term_meta::DF_OFF + U32_BYTES],
        ) as u64;
        // bytes [4..12] = self-offset (redundant; u64); skip
        let postings_length = read_u32_le(
            &postings[metadata_offset + term_meta::POSTINGS_LENGTH_OFF
                ..metadata_offset + term_meta::POSTINGS_LENGTH_OFF + U32_BYTES],
        ) as usize;
        let num_blocks = read_u32_le(
            &postings[metadata_offset + term_meta::NUM_BLOCKS_OFF
                ..metadata_offset + term_meta::NUM_BLOCKS_OFF + U32_BYTES],
        ) as usize;

        let (positions_offset, positions_length) = match positional {
            true => (
                read_u64_le(
                    &postings[metadata_offset + term_meta::POSITIONS_OFFSET_OFF
                        ..metadata_offset + term_meta::POSITIONS_OFFSET_OFF + U64_BYTES],
                ),
                read_u32_le(
                    &postings[metadata_offset + term_meta::POSITIONS_LENGTH_OFF
                        ..metadata_offset + term_meta::POSITIONS_LENGTH_OFF + U32_BYTES],
                ),
            ),
            false => (0, 0),
        };

        // The last block's end offset comes straight from
        // `postings_length`; bound it now instead of slicing OOB later.
        if metadata_offset + postings_length > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "term postings length exceeds the fetched term range".into(),
            )));
        }
        let skip_start = metadata_offset + term_meta_size;
        let skip_end = skip_start + num_blocks * skip_entry_bytes;
        if skip_end > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "skip table runs past postings region".into(),
            )));
        }
        // v3 positional terms store a run-offset sub-index right after the
        // skip table: `num_blocks × ENTRIES_PER_BLOCK` u32s. Bound it now;
        // the blocks follow it (their offsets are read from the skip
        // table, which the writer already shifted past the sub-index).
        let subindex_start = match subindex {
            SubindexKind::Wide => {
                let subindex_end = skip_end
                    + num_blocks * POSITION_SUBINDEX_ENTRIES_PER_BLOCK * subindex.entry_bytes();
                if subindex_end > postings.len() {
                    return Err(FtsError::Read(ReadError::MalformedVersion(
                        "position sub-index runs past postings region".into(),
                    )));
                }
                Some(skip_end)
            }
            SubindexKind::None => None,
        };
        // Coarse block-max table (V5 and later): `ceil(num_blocks / span)`
        // slots at the tail of the term region, so the blocks end where it
        // begins. V1–V4 blobs have no such table — the blocks run to
        // `postings_length` and the ranked walk skips the coarse level.
        let coarse_size = match has_coarse {
            true => {
                num_blocks.div_ceil(format::fts::COARSE_BLOCK_MAX_SPAN) * skip.coarse_slot_bytes()
            }
            false => 0,
        };
        // The length layout reaches a block through its span's start
        // offset, so it exists only alongside the coarse table.
        if skip == SkipLayout::Length && !has_coarse {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "length-coded skip table without a coarse table".into(),
            )));
        }
        if coarse_size > postings_length {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "coarse block-max table larger than the term region".into(),
            )));
        }
        let blocks_end_in_term = postings_length - coarse_size;
        let coarse_start = metadata_offset + blocks_end_in_term;
        Ok(Self {
            df,
            num_blocks,
            skip_start,
            positions_offset,
            positions_length,
            subindex_start,
            subindex,
            coarse_start,
            blocks_end_in_term,
            has_coarse,
            short: false,
            positions_grouped,
            block_layout: stored.block_layout(),
            skip,
            skip_entry_bytes,
        })
    }

    /// The metadata a short-form term implies: it has no header to
    /// parse, so the phrase path builds this from the decoded body's
    /// `df`. Its position group is inline in the body (the member's
    /// `positions` bytes are that slice), so the region fields are zero.
    pub(super) fn for_short(df: u64) -> Self {
        Self {
            df,
            num_blocks: 1,
            skip_start: 0,
            positions_offset: 0,
            positions_length: 0,
            subindex_start: None,
            subindex: SubindexKind::None,
            coarse_start: 0,
            blocks_end_in_term: 0,
            has_coarse: false,
            short: true,
            positions_grouped: true,
            block_layout: BlockLayout::Compact,
            skip: SkipLayout::Length,
            skip_entry_bytes: 0,
        }
    }

    /// Raw coarse slot `g`, bounding every block in span `g` (blocks
    /// `[g*SPAN .. (g+1)*SPAN)`) once a [`super::bounds::BoundDecoder`]
    /// has interpreted it. Coarse entries exist only where `has_coarse`.
    #[inline]
    pub(super) fn coarse_slot(&self, postings: &[u8], g: usize) -> u32 {
        let at = self.coarse_start + g * self.skip.coarse_slot_bytes() + coarse_slot::BOUND_OFF;
        read_u32_le(&postings[at..at + U32_BYTES])
    }

    /// Term-relative byte offset of span `g`'s first block (length
    /// layout only, where the slot records it).
    #[inline]
    fn coarse_span_start(&self, postings: &[u8], g: usize) -> usize {
        debug_assert_eq!(self.skip, SkipLayout::Length);
        let at =
            self.coarse_start + g * self.skip.coarse_slot_bytes() + coarse_slot::SPAN_START_OFF;
        read_u32_le(&postings[at..at + U32_BYTES]) as usize
    }

    /// For a `VERSION_V3` positional term, the run offset of the nearest
    /// sub-index checkpoint at or before pair `pair_in_block` of block
    /// `block`, and the number of runs to skip from it to reach the pair.
    /// The offset is relative to the term's positions (like
    /// [`Self::positions_block_offset`]). `None` when there is no
    /// sub-index (`V1`/`V2`) — the caller falls back to the block-start
    /// walk. The skip is always `< POSITION_SUBINDEX_STRIDE`.
    #[inline]
    pub(super) fn positions_subindex_offset(
        &self,
        postings: &[u8],
        block: usize,
        pair_in_block: usize,
    ) -> Option<(u32, usize)> {
        let start = self.subindex_start?;
        let slot = pair_in_block / POSITION_SUBINDEX_STRIDE;
        let idx = block * POSITION_SUBINDEX_ENTRIES_PER_BLOCK + slot;
        let runs_to_skip = pair_in_block % POSITION_SUBINDEX_STRIDE;
        let checkpoint = match self.subindex {
            SubindexKind::Wide => {
                let at = start + idx * U32_BYTES;
                read_u32_le(&postings[at..at + U32_BYTES])
            }
            SubindexKind::None => return None,
        };
        Some((checkpoint, runs_to_skip))
    }

    /// Byte offset of skip entry `i`.
    #[inline]
    fn entry_off(&self, i: usize) -> usize {
        debug_assert!(!self.short, "a short-form term has no skip table");
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        self.skip_start + i * self.skip_entry_bytes
    }

    /// Skip-table entry `i` as `(last_doc_id, raw bound slot)`. The slot
    /// is what a [`super::bounds::BoundDecoder`] turns into the block's
    /// upper bound. Per-entry on purpose — the single-term BMW walk
    /// streams entries without materializing a `Vec`.
    #[inline]
    pub(super) fn skip_entry(&self, postings: &[u8], i: usize) -> (u32, u32) {
        let entry_off = self.entry_off(i);
        let last_doc_id = read_u32_le(
            &postings[entry_off + skip_entry::LAST_DOC_ID_OFF
                ..entry_off + skip_entry::LAST_DOC_ID_OFF + U32_BYTES],
        );
        let bound_at = entry_off + self.skip.bound_off();
        let bound_slot = read_u32_le(&postings[bound_at..bound_at + U32_BYTES]);
        (last_doc_id, bound_slot)
    }

    /// The last doc id of the block before `i` — what block `i`'s
    /// compact header derives its base from. `None` for the first block.
    #[inline]
    pub(super) fn prev_last_doc_id(&self, postings: &[u8], i: usize) -> Option<u32> {
        match i {
            0 => None,
            _ => {
                let entry_off = self.entry_off(i - 1);
                Some(read_u32_le(
                    &postings[entry_off + skip_entry::LAST_DOC_ID_OFF
                        ..entry_off + skip_entry::LAST_DOC_ID_OFF + U32_BYTES],
                ))
            }
        }
    }

    /// Block `i`'s encoded byte length (length layout).
    #[inline]
    fn block_len(&self, postings: &[u8], i: usize) -> usize {
        let at = self.entry_off(i) + skip_entry::BLOCK_LEN_OFF;
        u16::from_le_bytes([postings[at], postings[at + 1]]) as usize
    }

    /// Block `i`'s term-relative byte range. Under the length layout a
    /// sequential walk passes the previous block's `end` and pays one
    /// `u16` read; a random block sums the lengths from its span's
    /// recorded start, at most `COARSE_BLOCK_MAX_SPAN - 1` of them.
    #[inline]
    pub(super) fn block_range_in_term(
        &self,
        postings: &[u8],
        i: usize,
        prev_end: Option<usize>,
    ) -> Range<usize> {
        match self.skip {
            SkipLayout::Absolute => {
                let at = self.entry_off(i) + skip_entry::BLOCK_OFFSET_OFF;
                let start = read_u32_le(&postings[at..at + U32_BYTES]) as usize;
                let end = match i + 1 < self.num_blocks {
                    true => {
                        let next = self.entry_off(i + 1) + skip_entry::BLOCK_OFFSET_OFF;
                        read_u32_le(&postings[next..next + U32_BYTES]) as usize
                    }
                    // The coarse block-max table follows the last block,
                    // so the blocks end before it — not at `postings_length`.
                    false => self.blocks_end_in_term,
                };
                start..end
            }
            SkipLayout::Length => {
                let start = match prev_end {
                    Some(end) => end,
                    None => {
                        let span = format::fts::COARSE_BLOCK_MAX_SPAN;
                        let g = i / span;
                        let mut at = self.coarse_span_start(postings, g);
                        for j in g * span..i {
                            at += self.block_len(postings, j);
                        }
                        at
                    }
                };
                start..start + self.block_len(postings, i)
            }
        }
    }

    /// This block's position-group byte offset within the term's
    /// positions bytes (zero on a positionless column).
    #[inline]
    pub(super) fn positions_block_offset(&self, postings: &[u8], i: usize) -> u32 {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        if self.short {
            // One block whose runs are the whole of the term's positions.
            return 0;
        }
        if self.skip_entry_bytes <= self.skip.positions_off() {
            return 0; // length layout, positionless column: no field
        }
        let at = self.entry_off(i) + self.skip.positions_off();
        read_u32_le(&postings[at..at + U32_BYTES])
    }
}

/// Per-term per-block metadata, parsed once at `TermCursor` construction.
#[derive(Debug, Clone, Copy)]
pub(super) struct BlockMeta {
    /// Largest doc_id present in this block.
    pub(super) last_doc_id: u32,
    /// Absolute byte offset (within the FTS postings region) of this
    /// block's encoded bytes.
    pub(super) block_byte_offset: usize,
    /// Absolute byte offset of the first byte AFTER this block. For
    /// the last block of a term it's `metadata_offset + postings_length`.
    pub(super) block_byte_end: usize,
    /// Per-block BM25 upper bound, recovered from the skip table's
    /// fixed-point `max_bm25_x1000` field.
    pub(super) block_max_bm25: f32,
}

/// Per-query-term cursor used by [`FtsReader::run_max_score_bmm`]
/// (and by [`FtsReader::run_wand_bmw`] in the bench-only path).
///
/// State:
///   - `blocks`: parsed skip table — one entry per block, lets us
///     decide whether to decode a block before paying the cost.
///   - `current_block` + `pos`: where we are in the term's posting
///     list. `pos == block_n` is treated as "advance to next block".
///   - `block_doc_ids` / `block_tfs`: decoded buffers for the current
///     block, reused across blocks.
///
/// `current_doc_id() == u32::MAX` is the "exhausted" sentinel; the
/// WAND loop drops cursors that are exhausted at the top of each
/// iteration.
#[derive(Clone)]
pub(crate) struct TermCursor {
    /// The effective inverse document frequency this cursor scores
    /// with: the table-wide value when the query uses table-wide
    /// statistics, otherwise this superfile's own, with a repeated
    /// term's query-side frequency already folded in.
    ///
    /// It is the whole per-cursor constant of the score numerator —
    /// there is no separate `(k1 + 1)` factor to carry — so the hot
    /// inner loop is one multiply, one add and one divide.
    pub(super) idf_weight: f32,
    /// Maximum block-max-BM25 across all blocks. Used by the WAND
    /// pivot test (term-level upper bound).
    pub(super) term_max_bm25: f32,
    /// Document frequency of the term (postings list length). Used by
    /// the 2-term OR router to detect a rare anchor term (short list),
    /// where WAND+BMW can skip the other term's long list.
    pub(super) df: u64,
    /// Per-block metadata (the parsed skip table). Read-only after
    /// build and `Arc`-shared, so cloning a cursor for another doc-id
    /// sub-range costs the ~1 KiB decode buffers, never a re-parse.
    pub(super) blocks: Arc<[BlockMeta]>,
    /// Decoded buffers for the current block. Reused across decodes.
    pub(super) block_doc_ids: Vec<u32>,
    pub(super) block_tfs: Vec<u32>,
    /// Number of valid entries in the decoded block buffers (the
    /// last block may be partial).
    pub(super) block_n: usize,
    /// Index into `blocks` of the currently-decoded block. Equal to
    /// `blocks.len()` once exhausted.
    pub(super) current_block: usize,
    /// Position within the currently-decoded block. Always `<
    /// block_n` while not exhausted.
    pub(super) pos: usize,
    /// Index into `blocks` of the block being inspected by the BMW
    /// upper-bound check. Standard block-cursor split:
    /// `shallow_advance_block_to(pivot_doc)` updates this without
    /// decoding the block, so subsequent BMW UB lookups for
    /// monotonically-increasing pivot docs are amortized O(1). Always
    /// `>= current_block`; synced up whenever `current_block` is
    /// advanced.
    pub(super) inspect_block: usize,
    /// This term's own postings bytes — the metadata header (offset
    /// 0), skip table, and encoded blocks, fetched as a single
    /// contiguous range by [`FtsReader::fetch_term_postings`]. All
    /// `BlockMeta` byte offsets are relative to the start of this
    /// buffer. Empty for inline (df=1) cursors, which never decode.
    /// Mirrors the vector reader's per-probed-cluster buffers: the
    /// search hot loops index only the bytes this term touches, never
    /// the whole postings region.
    ///
    /// Deliberately carries NO positional state: term cursors are the
    /// hot per-query unit the multi-cursor kernels iterate over, and
    /// the positional extras matter only to phrase members —
    /// [`PhraseMember`] re-derives them from these bytes instead, so
    /// plain term queries never pay for them in cursor or block-meta
    /// footprint.
    pub(super) bytes: Bytes,
    /// True when this term's FST slot carried no postings-length hint,
    /// so the build probed the 20-byte header before fetching the body
    /// — two planned byte-source ranges instead of one.
    pub(super) header_probed: bool,
    /// Count-only cursor: `decode_current_block` skips the tf half of each
    /// block (see [`decode_block_doc_ids`]). Set by the unranked count
    /// kernels (union / intersection), which never read `block_tfs`;
    /// leaves `block_tfs` stale, so a `count_only` cursor must not be used
    /// for scoring.
    pub(super) count_only: bool,
    /// Which block index is currently decoded into `block_doc_ids`
    /// (`usize::MAX` = none). Lets [`Self::contains`] skip re-decoding a
    /// PACKED block it already holds while probing membership across a
    /// run of ascending target docs.
    pub(super) decoded_block: usize,
    /// Which block index has its tf array decoded into `block_tfs`
    /// (`usize::MAX` = none). Set whenever `block_tfs` is filled — by a full
    /// [`Self::decode_current_block`] (non-count) or by a tf-only decode in
    /// [`Self::bitset_probe_tf`], which reads a single doc's tf by rank
    /// without expanding the block's doc ids. Lets the probe reuse the
    /// decoded tfs across a run of candidates landing in the same block.
    pub(super) tf_decoded_block: usize,
    /// The whole posting list is already in `block_doc_ids[..block_n]` /
    /// `block_tfs[..block_n]` and `blocks` has exactly one entry with no
    /// bytes behind it: the df=1 inline form and the short form
    /// (`fts::short`). Such a cursor never decodes; the membership probes
    /// binary-search the buffer instead of reading a block encoding.
    pub(super) predecoded: bool,
    /// Which header the blocks carry (from the blob version).
    pub(super) layout: BlockLayout,
    /// The parsed header of the block it names — the membership probes
    /// visit one block many times and must not re-parse it per probe.
    header_cache: Option<(usize, BlockHeader)>,
}

impl TermCursor {
    /// Parse one term's metadata + skip table out of its own postings
    /// byte range and decode its first block. `term_bytes` starts at
    /// the term's 20-byte metadata header (offset 0) and runs to the
    /// end of its last block — the contiguous range
    /// [`FtsReader::fetch_term_postings`] fetched for this term.
    pub(super) fn new(
        term_bytes: Bytes,
        col: &ColumnMeta,
        stored: StoredBound,
        global_idf: Option<f32>,
        weight: u32,
        header_probed: bool,
        count_only: bool,
    ) -> Result<Self, FtsError> {
        let postings: &[u8] = term_bytes.as_ref();
        let metadata_offset = 0usize;

        // The plain-term cursor never decodes positions, so it needs no
        // sub-index (it reads block offsets straight from the skip table).
        // `has_coarse` tells it the last block ends before the coarse
        // table, not at `postings_length`.
        let term_meta = TermMeta::parse(
            postings,
            metadata_offset,
            col.positions,
            SubindexKind::None,
            stored,
            false,
        )?;
        let local_idf = bm25::idf(col.scored_doc_count(), term_meta.df);
        // Effective idf folds in the query-term-frequency `weight` (> 1
        // only for a deduplicated repeated term) on top of any global-idf
        // override. Every stored bound is decoded at this idf too, so the
        // bounds stay consistent with the scores computed from it.
        let idf = global_idf.unwrap_or(local_idf) * weight as f32;
        let bounds = BoundDecoder::new(stored, col, idf, local_idf);

        // Collect straight into the `Arc` allocation: `0..num_blocks` is
        // an exact-size iterator, so this writes each entry in place —
        // one allocation, no intermediate `Vec` + copy. The skip table
        // is ~a quarter of a long term's cursor-build bytes (one 32-byte
        // entry per 128-doc block), so the doubled write showed up on
        // common-term queries.
        let mut term_max_bm25: f32 = 0.0;
        let mut prev_end: Option<usize> = None;
        let blocks: Arc<[BlockMeta]> = (0..term_meta.num_blocks)
            .map(|i| {
                let (last_doc_id, raw) = term_meta.skip_entry(postings, i);
                let block_max_bm25 = bounds.bound(raw);
                term_max_bm25 = term_max_bm25.max(block_max_bm25);
                let range = term_meta.block_range_in_term(postings, i, prev_end);
                prev_end = Some(range.end);

                BlockMeta {
                    last_doc_id,
                    block_byte_offset: metadata_offset + range.start,
                    block_byte_end: metadata_offset + range.end,
                    block_max_bm25,
                }
            })
            .collect();

        let mut cursor = Self {
            idf_weight: idf,
            term_max_bm25,
            df: term_meta.df,
            blocks,
            block_doc_ids: vec![0u32; BLOCK_LEN],
            block_tfs: vec![0u32; BLOCK_LEN],
            block_n: 0,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: term_bytes,
            header_probed,
            count_only,
            decoded_block: usize::MAX,
            tf_decoded_block: usize::MAX,
            predecoded: false,
            layout: term_meta.block_layout,
            header_cache: None,
        };
        if !cursor.blocks.is_empty() {
            cursor.decode_current_block();
        }
        Ok(cursor)
    }

    /// Build a cursor from a short-form body (`fts::short`): decode the
    /// whole list — at most one block — into the cursor's buffers and
    /// synthesize its single block's metadata. The block's upper bound
    /// is computed here, at the query's own statistics, as the maximum
    /// per-doc score over the decoded postings: exact, and no stored
    /// slot to decode. `body` is kept as `bytes` so the work tallies
    /// count the range that was fetched.
    pub(super) fn new_short(
        body: Bytes,
        col: &ColumnMeta,
        global_idf: Option<f32>,
        weight: u32,
        header_probed: bool,
    ) -> Result<Self, FtsError> {
        let mut block_doc_ids = vec![0u32; BLOCK_LEN];
        let mut block_tfs = vec![0u32; BLOCK_LEN];
        let decoded = decode_short(
            body.as_ref(),
            col.positions,
            &mut block_doc_ids,
            &mut block_tfs,
        )
        .ok_or_else(|| {
            FtsError::Read(ReadError::MalformedVersion(
                "malformed short-form term body".into(),
            ))
        })?;
        let n = decoded.n;
        let local_idf = bm25::idf(col.scored_doc_count(), n as u64);
        let idf_weight = global_idf.unwrap_or(local_idf) * weight as f32;
        let block_max_bm25 = block_doc_ids[..n]
            .iter()
            .zip(&block_tfs[..n])
            .map(|(&d, &t)| bm25::score_with_dl_norm_k1(idf_weight, t, col.dl_norm_k1.get(d)))
            .fold(0.0f32, f32::max);
        let blocks: Arc<[BlockMeta]> = Arc::from([BlockMeta {
            last_doc_id: block_doc_ids[n - 1],
            block_byte_offset: 0,
            block_byte_end: 0,
            block_max_bm25,
        }]);
        Ok(Self {
            idf_weight,
            term_max_bm25: block_max_bm25,
            df: n as u64,
            blocks,
            block_doc_ids,
            block_tfs,
            block_n: n,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: body,
            header_probed,
            count_only: false,
            decoded_block: 0,
            tf_decoded_block: 0,
            predecoded: true,
            layout: BlockLayout::Compact,
            header_cache: None,
        })
    }

    /// Synthesize a cursor for a df=1 inline-encoded term. Skips the
    /// postings-region read entirely — the caller already has
    /// (doc_id, tf) from unpacking the FST value, and BMW upper bound
    /// for a 1-doc term equals that doc's actual BM25 score (only one
    /// doc means min_dl = dl and max_tf = tf, so the per-block UB
    /// formula collapses to the score itself). Computed at query time
    /// since there's no skip-table entry stored for inline terms.
    pub(super) fn new_inline(
        doc_id: u32,
        tf: u32,
        n_scored_docs: u64,
        dl_norm_k1: f32,
        global_idf: Option<f32>,
        weight: u32,
    ) -> Self {
        // Fold the qtf `weight` into the effective idf so the single-doc block-max
        // (computed below from `idf_weight`) scales together with the score.
        let idf_weight = global_idf.unwrap_or_else(|| bm25::idf(n_scored_docs, 1)) * weight as f32;
        let block_max_bm25 = bm25::score_with_dl_norm_k1(idf_weight, tf, dl_norm_k1);

        let blocks: Arc<[BlockMeta]> = Arc::from([BlockMeta {
            last_doc_id: doc_id,
            // No postings-region bytes back this cursor; the decoded
            // buffer is pre-filled below so `decode_current_block` is
            // never called against these offsets.
            block_byte_offset: 0,
            block_byte_end: 0,
            block_max_bm25,
        }]);

        let mut block_doc_ids = vec![0u32; BLOCK_LEN];
        let mut block_tfs = vec![0u32; BLOCK_LEN];
        block_doc_ids[0] = doc_id;
        block_tfs[0] = tf;

        Self {
            idf_weight,
            term_max_bm25: block_max_bm25,
            df: 1,
            blocks,
            block_doc_ids,
            block_tfs,
            block_n: 1,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: Bytes::new(),
            header_probed: false,
            // Inline cursors carry their single posting pre-decoded and
            // never call `decode_current_block`, so the flag is inert.
            count_only: false,
            decoded_block: 0,
            tf_decoded_block: 0,
            predecoded: true,
            layout: BlockLayout::Compact,
            header_cache: None,
        }
    }

    /// Block `b`'s parsed header. The compact layout derives the base
    /// doc id from the previous block's `last_doc_id`, which the skip
    /// table gave us at construction.
    #[inline]
    pub(super) fn block_header(&self, b: usize) -> BlockHeader {
        let block = self.blocks[b];
        let prev = match b {
            0 => None,
            _ => Some(self.blocks[b - 1].last_doc_id),
        };
        BlockHeader::parse(
            &self.bytes[block.block_byte_offset..block.block_byte_end],
            self.layout,
            prev,
        )
    }

    /// The current block's parsed header, cached across probes of the
    /// same block.
    #[inline]
    fn current_header(&mut self) -> BlockHeader {
        let b = self.current_block;
        if let Some((cached, hdr)) = self.header_cache
            && cached == b
        {
            return hdr;
        }
        let hdr = self.block_header(b);
        self.header_cache = Some((b, hdr));
        hdr
    }

    pub(super) fn decode_current_block(&mut self) {
        debug_assert!(!self.predecoded, "a pre-decoded cursor has no block bytes");
        let block = self.blocks[self.current_block];
        // Borrow in place rather than clone an owned `Bytes` (disjoint from the
        // `&mut self.block_*` decode targets, which are separate fields).
        let hdr = self.current_header();
        let bytes = &self.bytes[block.block_byte_offset..block.block_byte_end];
        // Count-only cursors skip the tf half of the block; the count
        // kernels never read `block_tfs`, so it is left stale.
        self.block_n = match self.count_only {
            true => decode_block_doc_ids(bytes, &hdr, &mut self.block_doc_ids),
            false => decode_block(bytes, &hdr, &mut self.block_doc_ids, &mut self.block_tfs),
        };
        self.pos = 0;
        self.decoded_block = self.current_block;
        // A non-count decode also fills `block_tfs` for this block, so the
        // tf-only probe can reuse it without re-decoding.
        if !self.count_only {
            self.tf_decoded_block = self.current_block;
        }
    }

    /// Membership probe: does this term contain `doc`? Advances the block
    /// cursor forward to the block that could hold `doc` (targets arrive
    /// ascending on the AND-count leapfrog) and, on a **bitset block**,
    /// answers with a single bit-test — no decode. A PACKED block is
    /// decoded once (cached via `decoded_block`) and binary-searched. Used
    /// only by the count leapfrog; it moves `current_block`, so a cursor
    /// probed with `contains` must not also be iterated.
    pub(super) fn contains(&mut self, doc: u32) -> bool {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < doc
        {
            self.current_block += 1;
        }
        if self.current_block >= self.blocks.len() {
            return false;
        }
        // Pre-decoded (inline or short-form) cursor: the whole list is in
        // the buffer, no block bytes to read.
        if self.predecoded {
            return self.block_doc_ids[..self.block_n]
                .binary_search(&doc)
                .is_ok();
        }
        let block = self.blocks[self.current_block];
        // Borrow the block's bytes in place — `self.bytes` is held for the
        // cursor's life, so a subslice needs no owned `Bytes` clone. A
        // per-probe `.slice()` here bumps and drops an atomic refcount on
        // every membership probe; over a long driver it was ~11% of the
        // intersection-count time (and wasted on the PACKED path, which
        // only reads the encoding byte before falling to the decode cache).
        let hdr = self.current_header();
        let raw = &self.bytes[block.block_byte_offset..block.block_byte_end];
        if hdr.encoding == ENCODING_BITSET {
            match Self::bitset_word(raw, &hdr, doc) {
                Some((bit, word, _)) => (word >> (bit % 64)) & 1 == 1,
                None => false,
            }
        } else {
            // Borrow of `raw` ends above; the decode needs `&mut self`.
            if self.decoded_block != self.current_block {
                self.decode_current_block();
            }
            self.block_doc_ids[..self.block_n]
                .binary_search(&doc)
                .is_ok()
        }
    }

    /// Materialize a `contains`-probed cursor at `doc`: ensure the current
    /// block is decoded and `pos` points at `doc`. A membership probe
    /// (`contains`) advances `current_block` but, on a **bitset block**,
    /// answers by bit-test without decoding — leaving `block_doc_ids`,
    /// `block_tfs`, and `pos` stale. The phrase position-verification path
    /// needs the fully decoded block; this decodes it (only when the current
    /// block isn't already decoded) and scans `pos` up to `doc`. Callers
    /// pass a `doc` a preceding `contains(doc)` confirmed is present, arriving
    /// in ascending order, so the forward `pos` scan always lands on it.
    pub(super) fn materialize_at(&mut self, doc: u32) {
        if self.decoded_block != self.current_block {
            self.decode_current_block();
        }
        while self.pos < self.block_n && self.block_doc_ids[self.pos] < doc {
            self.pos += 1;
        }
    }

    /// Ranked-OR non-essential membership probe returning the doc's tf
    /// **without expanding the block's doc ids**. On a dense (bitset) block it
    /// bit-tests presence and, on a hit, reads the one tf by popcount-rank into
    /// the tf array (decoded once per block) — never materializing the 128 doc
    /// ids, which is the dominant cost of the ranked-OR non-essential
    /// completion on common terms. On a PACKED block there is no rank shortcut
    /// (the doc ids must be decoded to locate the doc), so it falls back to
    /// `skip_to` + `current_tf`. Like [`Self::contains`] it advances
    /// `current_block`, so a cursor probed this way must not also be iterated.
    /// Rank of the doc at in-block position `bit` among a bitset block's presence
    /// bits — the count of set bits before `bit`, i.e. that doc's index into the
    /// block's doc-order tf array. `word` is the presence word already loaded at
    /// `bit`'s position; `bitset_end` is the end of the presence bitmap (start of
    /// the tf array). Shared by [`Self::bitset_probe_tf`] (which first checks the
    /// bit is set) and [`Self::tf_at_contained`] (which knows it is).
    /// Locate `doc` in a bitset block: its bit index within the presence
    /// bitset, the word holding it and where the bitset ends (the tf
    /// array's start). `None` when `doc` lies below the block's origin or
    /// past its last word — absent either way.
    #[inline]
    fn bitset_word(raw: &[u8], hdr: &BlockHeader, doc: u32) -> Option<(usize, u64, usize)> {
        if doc < hdr.base {
            return None;
        }
        let bit = (doc - hdr.base) as usize;
        let bitset_end = raw.len() - hdr.tfs_size();
        let word_at = hdr.payload() + (bit / 64) * 8;
        if word_at + 8 > bitset_end {
            return None;
        }
        let word = u64::from_le_bytes(raw[word_at..word_at + 8].try_into().expect("8 bytes"));
        Some((bit, word, bitset_end))
    }

    #[inline]
    fn bitset_tf_rank(raw: &[u8], payload: usize, bit: usize, word: u64, bitset_end: usize) -> u32 {
        let word_idx = bit / 64;
        let presence = &raw[payload..bitset_end];
        let mut rank: u32 = 0;
        for w in presence[..word_idx * 8].chunks_exact(8) {
            rank += u64::from_le_bytes(w.try_into().expect("8 bytes")).count_ones();
        }
        let below = if bit.is_multiple_of(64) {
            0u64
        } else {
            (1u64 << (bit % 64)) - 1
        };
        rank + (word & below).count_ones()
    }

    pub(super) fn bitset_probe_tf(&mut self, doc: u32) -> Option<u32> {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < doc
        {
            self.current_block += 1;
        }
        if self.current_block >= self.blocks.len() {
            return None;
        }
        // Pre-decoded (inline or short-form) cursor: locate in the buffer.
        if self.predecoded {
            return self.block_doc_ids[..self.block_n]
                .binary_search(&doc)
                .ok()
                .map(|i| self.block_tfs[i]);
        }
        let block = self.blocks[self.current_block];
        let hdr = self.current_header();
        if hdr.encoding != ENCODING_BITSET {
            // PACKED: no rank shortcut — decode + locate like the old path.
            self.skip_to(doc);
            return if self.current_doc_id() == doc {
                Some(self.current_tf())
            } else {
                None
            };
        }
        let raw = &self.bytes[block.block_byte_offset..block.block_byte_end];
        let (bit, word, bitset_end) = Self::bitset_word(raw, &hdr, doc)?;
        if (word >> (bit % 64)) & 1 == 0 {
            return None; // doc not present in this block
        }
        // Present: the r-th set bit (doc) maps to the r-th tf in doc order.
        let rank = Self::bitset_tf_rank(raw, hdr.payload(), bit, word, bitset_end);
        // Decode this block's tf array once (doc order), reused across a run of
        // candidates in the same block; the doc ids are never expanded. The
        // union and intersection kernels probe a dense block many times, so
        // one 128-lane unpack beats a bit-field read per probe (measured:
        // reading the single lane cost union 7% and intersection 5%).
        if self.tf_decoded_block != self.current_block {
            decode_block_tfs(raw, &hdr, &mut self.block_tfs);
            self.tf_decoded_block = self.current_block;
        }
        Some(self.block_tfs[rank as usize])
    }

    pub(super) fn is_exhausted(&self) -> bool {
        self.current_block >= self.blocks.len()
    }

    /// Block count, used as a cheap proxy for df when AND intersection
    /// picks the rarest cursor as the leader. Block count is an exact
    /// upper bound on df: a term's df is `(blocks - 1) * BLOCK_LEN +
    /// last_block_n`, so cursors compare in the same order by block
    /// count as they do by df. Inline cursors return 1.
    #[inline(always)]
    pub(super) fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline(always)]
    pub(super) fn current_doc_id(&self) -> u32 {
        if self.is_exhausted() {
            u32::MAX
        } else {
            // A live cursor always has `pos < block_n` — every mutator
            // (`next`, `advance_by`, `skip_to`, `advance_block`) restores it
            // or marks the cursor exhausted (see the `pos` field doc). The
            // extra `pos >= block_n` guard was dead on this hot walk
            // primitive; the debug tripwire fires if a future change breaks
            // the invariant.
            debug_assert!(self.pos < self.block_n);
            self.block_doc_ids[self.pos]
        }
    }

    #[inline(always)]
    pub(super) fn current_tf(&self) -> u32 {
        debug_assert!(!self.is_exhausted() && self.pos < self.block_n);
        self.block_tfs[self.pos]
    }

    #[inline(always)]
    pub(super) fn current_block_max_bm25(&self) -> f32 {
        if self.is_exhausted() {
            0.0
        } else {
            self.blocks[self.current_block].block_max_bm25
        }
    }

    /// Largest doc_id in the cursor's current block. Used by the BMW
    /// skip step to compute the smallest "next interesting doc_id"
    /// across the prefix.
    #[inline(always)]
    pub(super) fn current_block_last_doc_id(&self) -> u32 {
        if self.is_exhausted() {
            u32::MAX
        } else {
            self.blocks[self.current_block].last_doc_id
        }
    }

    /// Shallow-advance the inspect-block pointer to the block that
    /// would contain `target`. Does NOT decode and does NOT touch the
    /// doc cursor (`current_block`, `pos`, decoded buffers stay put);
    /// only the lightweight `inspect_block` index moves. Used by the
    /// BMW UB sum at `pivot_doc` for cursors whose current_doc lags
    /// pivot_doc — their relevant block-max is the block containing
    /// pivot_doc, not their current decoded block.
    ///
    /// Monotonically advances; calling this for monotonically-
    /// increasing `target` across WAND iterations gives amortized
    /// O(1) per call.
    pub(super) fn shallow_advance_block_to(&mut self, target: u32) {
        // Never let inspect_block fall behind current_block — once
        // the doc cursor has decoded past a block, that block's
        // metadata is no longer relevant.
        if self.inspect_block < self.current_block {
            self.inspect_block = self.current_block;
        }
        while self.inspect_block < self.blocks.len()
            && self.blocks[self.inspect_block].last_doc_id < target
        {
            self.inspect_block += 1;
        }
    }

    /// Maximum `block_max_bm25` across all blocks of this cursor whose
    /// doc-id range overlaps `[range_start, range_end]` (inclusive on
    /// both ends). Used by AND block-max pruning to compute a safe
    /// upper bound on this cursor's contribution across the leader's
    /// current block — a single-block lookup at one boundary
    /// underestimates when the leader's range spans multiple
    /// cursor blocks with varying block_max. Uses `inspect_block` as
    /// a hint pointer so monotonically-advancing leader ranges amortize
    /// to O(1) amortized per call.
    pub(super) fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        // Advance inspect_block to the first block whose last_doc_id
        // could intersect the range. shallow_advance_block_to lands on
        // the first block with last_doc_id >= range_start, which is
        // exactly the first block that can overlap the range.
        self.shallow_advance_block_to(range_start);
        let mut max: f32 = 0.0;
        let mut i = self.inspect_block;
        while i < self.blocks.len() {
            // Block i starts at the doc right after the previous block's
            // last_doc_id (or doc 0 if i == 0). Once block_start exceeds
            // range_end the rest of the blocks lie strictly past the
            // range; stop walking.
            let block_start = if i == 0 {
                0u32
            } else {
                self.blocks[i - 1].last_doc_id.saturating_add(1)
            };
            if block_start > range_end {
                break;
            }
            let m = self.blocks[i].block_max_bm25;
            if m > max {
                max = m;
            }
            i += 1;
        }
        max
    }

    /// Block-max-BM25 at the inspect-block pointer. Pair with
    /// `shallow_advance_block_to(pivot_doc)` to bound the cursor's
    /// contribution at pivot_doc.
    pub(super) fn inspect_block_max_bm25(&self) -> f32 {
        if self.inspect_block >= self.blocks.len() {
            0.0
        } else {
            self.blocks[self.inspect_block].block_max_bm25
        }
    }

    /// Last doc_id in the block at the inspect-block pointer. Used
    /// for the BMW skip target — the smallest "next interesting doc"
    /// across the prefix is one past the smallest such block-end.
    pub(super) fn inspect_block_last_doc_id(&self) -> u32 {
        if self.inspect_block >= self.blocks.len() {
            u32::MAX
        } else {
            self.blocks[self.inspect_block].last_doc_id
        }
    }

    /// Advance one position. Crosses block boundaries automatically;
    /// decodes the next block on demand.
    #[inline(always)]
    pub(super) fn next(&mut self) {
        if self.is_exhausted() {
            return;
        }
        self.pos += 1;
        if self.pos >= self.block_n {
            self.advance_block();
        }
    }

    /// Advance a known in-block batch, crossing to the next block when
    /// `count` consumes its remaining postings. Unlike [`Self::next`],
    /// callers must not start at or advance past the decoded block end.
    #[inline(always)]
    pub(super) fn advance_by(&mut self, count: usize) {
        debug_assert!(!self.is_exhausted());
        debug_assert!(count > 0 && self.pos + count <= self.block_n);
        self.pos += count;
        // The assertion above makes equality equivalent to `>=` here.
        if self.pos == self.block_n {
            self.advance_block();
        }
    }

    /// Move to and decode the next posting block, or mark the cursor
    /// exhausted when the current block is the last one.
    #[inline(always)]
    pub(super) fn advance_block(&mut self) {
        self.current_block += 1;
        if self.current_block > self.inspect_block {
            self.inspect_block = self.current_block;
        }
        if self.current_block < self.blocks.len() {
            self.decode_current_block();
        }
    }

    /// Skip forward so `current_doc_id() >= target`. Uses the skip
    /// table to skip whole blocks when the entire block precedes
    /// `target`. Common-case fast path (target lies within the
    /// already-decoded current block) is just an inlined `pos++`
    /// scan — no re-decode, no `is_exhausted` rechecks.
    #[inline(always)]
    pub(super) fn skip_to(&mut self, target: u32) {
        if self.is_exhausted() {
            return;
        }
        let cur_block = self.current_block;
        let cur_block_last = self.blocks[cur_block].last_doc_id;
        if cur_block_last >= target {
            // Fast path: target is in our currently-decoded block.
            // Just scan pos forward. The `current_doc_id() >= target`
            // guard from before is folded into this scan — if pos is
            // already at-or-past, the loop body doesn't execute.
            let n = self.block_n;
            while self.pos < n && self.block_doc_ids[self.pos] < target {
                self.pos += 1;
            }
            if self.pos < n {
                return;
            }
            // Walked off the end of the decoded block (rare under
            // skip-table invariants); fall through to cross-block.
        }
        self.skip_to_cross_block(target);
    }

    /// Cross-block path of `skip_to`: target is past the current
    /// decoded block. Advances `current_block` via the skip table,
    /// decodes the new block (only when crossing), and scans pos.
    /// Pulled out so the within-block fast path stays small enough
    /// to inline at every call site.
    #[cold]
    pub(super) fn skip_to_cross_block(&mut self, target: u32) {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < target
        {
            self.current_block += 1;
        }
        if self.current_block > self.inspect_block {
            self.inspect_block = self.current_block;
        }
        if self.is_exhausted() {
            return;
        }
        self.decode_current_block();
        while self.pos < self.block_n && self.block_doc_ids[self.pos] < target {
            self.pos += 1;
        }
        if self.pos >= self.block_n {
            self.current_block += 1;
            if self.current_block > self.inspect_block {
                self.inspect_block = self.current_block;
            }
            if self.current_block < self.blocks.len() {
                self.decode_current_block();
            }
        }
    }

    /// Tf for `doc` on a cursor a preceding [`Self::contains(doc)`] just confirmed
    /// present. `contains` already advanced `current_block` to `doc`'s block (and,
    /// on a PACKED block, decoded it), so this skips the block-advance and the
    /// presence bit-test that [`Self::bitset_probe_tf`] repeats, doing only the tf
    /// lookup: a popcount-rank into the tf array on a bitset block, or a binary
    /// search over the decoded doc ids on a PACKED one. Only valid immediately
    /// after `contains(doc)` returned `true` with no intervening advance.
    ///
    /// Kept at the end of the impl, past the doc-cursor hot methods
    /// (`skip_to`, `next`, `decode_current_block`, `current_doc_id`), so adding
    /// it doesn't shift their code offsets — those methods drive the flat-merge
    /// AND path, which is measurably sensitive to its own instruction layout.
    pub(super) fn tf_at_contained(&mut self, doc: u32) -> u32 {
        // Pre-decoded (inline or short-form) cursor: locate in the buffer.
        if self.predecoded {
            let pos = self.block_doc_ids[..self.block_n]
                .binary_search(&doc)
                .expect("contains(doc) confirmed presence");
            return self.block_tfs[pos];
        }
        let block = self.blocks[self.current_block];
        let hdr = self.current_header();
        let raw = &self.bytes[block.block_byte_offset..block.block_byte_end];
        if hdr.encoding == ENCODING_BITSET {
            let (bit, word, bitset_end) =
                Self::bitset_word(raw, &hdr, doc).expect("contains(doc) confirmed presence");
            let rank = Self::bitset_tf_rank(raw, hdr.payload(), bit, word, bitset_end);
            if self.tf_decoded_block != self.current_block {
                decode_block_tfs(raw, &hdr, &mut self.block_tfs);
                self.tf_decoded_block = self.current_block;
            }
            self.block_tfs[rank as usize]
        } else {
            // PACKED: `contains` decoded this block's doc ids and tfs. Locate doc.
            let pos = self.block_doc_ids[..self.block_n]
                .binary_search(&doc)
                .expect("contains(doc) confirmed presence");
            self.block_tfs[pos]
        }
    }

    /// Whether this term's postings are stored in the dense **bitset** encoding,
    /// sampled from the first block's encoding byte (a dense term's blocks are
    /// uniformly bitset). When true, [`Self::contains`] answers by an O(1)
    /// bit-test instead of a block decode — the signal a 2-term AND uses to
    /// decide the membership walk beats the flat-merge's block expansion.
    ///
    /// `#[cold]`: called once per query at dispatch, not in a per-doc loop —
    /// out-of-line so it stays clear of the hot doc-cursor methods' layout.
    #[cold]
    pub(super) fn is_bitset_dense(&self) -> bool {
        if self.predecoded {
            return false; // inline / short-form cursor: no block bytes
        }
        match self.blocks.first() {
            Some(block) => {
                block_encoding(&self.bytes[block.block_byte_offset..block.block_byte_end])
                    == ENCODING_BITSET
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rand_distr::{Distribution, LogNormal};

    use super::*;
    use crate::superfile::fts::{
        bm25, builder::FtsBuilder, reader::FtsReader, tokenize::AsciiLowerTokenizer,
    };

    /// The per-block BM25 upper bound stored in the skip table must be a
    /// valid upper bound over the *query-time* score of every document in
    /// that block. Query-time scoring reads each document's length from the
    /// byte-quantized norm table, which truncates the length downward — and
    /// a shorter length yields a *higher* BM25 score. If the stored block
    /// max is computed from the exact (un-truncated) length, it lands below
    /// the query score of a doc whose length quantizes down, and the
    /// block-max skip in the ranked-OR walk drops that doc from the top-k.
    ///
    /// This plants a term spanning several 128-doc blocks whose documents
    /// all have a length in the quantize-down region, then walks the term's
    /// cursor and asserts `block_max >= query_score` for every posting.
    /// Without the length-consistent block bound the assertion fires on the
    /// highest-tf doc in each block; a small-doc corpus (every length in the
    /// exact-quantization region) never exercises it.
    /// A corpus shaped like real text: log-normal lengths (median 89
    /// tokens, a long tail) and one common term drawn per token with the
    /// probability of a Zipf rank-1 word, so its frequency grows with the
    /// document and every block holds documents scoring within a hair of
    /// each other — the shape block-max pruning is most sensitive to.
    fn realistic_reader(n_docs: u32) -> FtsReader {
        let mut rng = StdRng::seed_from_u64(114);
        let lengths = LogNormal::new(4.4886, 1.55).expect("log-normal params");
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        let mut text = String::new();
        for doc_id in 0..n_docs {
            let len = (lengths.sample(&mut rng) as usize).clamp(1, 3000);
            text.clear();
            for _ in 0..len {
                if rng.random_bool(0.069) {
                    text.push_str("common ");
                } else {
                    text.push_str(&format!("f{} ", rng.random_range(0..5000u32)));
                }
            }
            b.add_doc(0, doc_id, text.trim_end()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    /// A positional and a positionless column, each with a term spanning
    /// many blocks whose gaps vary, so the length-coded skip table has
    /// to be right in both entry widths.
    fn two_column_reader() -> FtsReader {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("pos".into(), true).expect("register");
        b.register_column("flat".into(), false).expect("register");
        let mut text = String::new();
        for doc_id in 0..6000u32 {
            // Every doc has `common`; a stretch of docs carries it several
            // times, and every 500th doc is a long one.
            text.clear();
            let reps = if (doc_id / 128) % 3 == 0 { 3 } else { 1 };
            for r in 0..reps {
                text.push_str(&format!("filler{} common ", (doc_id + r) % 97));
            }
            if doc_id % 500 == 0 {
                for i in 0..300 {
                    text.push_str(&format!("w{i} "));
                }
            }
            b.add_doc(0, doc_id, text.trim_end()).expect("add pos");
            b.add_doc(1, doc_id, text.trim_end()).expect("add flat");
        }
        let json = r#"[{"name":"pos","tokenizer":"ascii_lower","positions":true},{"name":"flat","tokenizer":"ascii_lower"}]"#;
        FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open")
    }

    #[tokio::test]
    async fn length_coded_skip_entries_locate_every_block_by_either_route() {
        let view = two_column_reader();
        assert_eq!(view.bounds.skip_layout(), SkipLayout::Length);
        for (col, positional) in [(0u32, true), (1u32, false)] {
            let cursors = view
                .build_term_cursors(col, &["common"], None, false, None, None)
                .await
                .expect("cursors");
            let cursor = &cursors[0];
            assert!(cursor.blocks.len() > 40, "{} blocks", cursor.blocks.len());
            let postings: &[u8] = cursor.bytes.as_ref();
            let meta = TermMeta::parse(
                postings,
                0,
                positional,
                SubindexKind::None,
                view.bounds,
                view.positions_grouped,
            )
            .expect("meta");
            assert_eq!(meta.num_blocks, cursor.blocks.len());
            assert_eq!(
                meta.skip.entry_bytes(positional),
                if positional { 14 } else { 10 }
            );
            // Sequential accumulation and the random route (span start plus
            // summed lengths) agree with each other and with the cursor.
            let mut prev_end = None;
            let mut last_pos_off = 0u32;
            for i in 0..meta.num_blocks {
                let sequential = meta.block_range_in_term(postings, i, prev_end);
                let random = meta.block_range_in_term(postings, i, None);
                assert_eq!(sequential, random, "block {i}");
                assert_eq!(
                    sequential.start, cursor.blocks[i].block_byte_offset,
                    "block {i}"
                );
                assert_eq!(sequential.end, cursor.blocks[i].block_byte_end, "block {i}");
                prev_end = Some(sequential.end);
                let (last_doc, _) = meta.skip_entry(postings, i);
                assert_eq!(last_doc, cursor.blocks[i].last_doc_id);
                assert_eq!(
                    meta.prev_last_doc_id(postings, i),
                    (i > 0).then(|| cursor.blocks[i - 1].last_doc_id)
                );
                let pos_off = meta.positions_block_offset(postings, i);
                match positional {
                    true => {
                        assert!(i == 0 || pos_off > last_pos_off, "block {i} group offset");
                        last_pos_off = pos_off;
                    }
                    false => assert_eq!(pos_off, 0, "no positions field"),
                }
            }
            // The last block ends where the coarse table begins.
            assert_eq!(prev_end, Some(meta.blocks_end_in_term));
            // Every block decodes from its own range and its predecessor's
            // last doc, and the term's docs come out ascending across blocks.
            let mut prev_last: Option<u32> = None;
            let mut d = vec![0u32; BLOCK_LEN];
            for i in 0..meta.num_blocks {
                let range = meta.block_range_in_term(postings, i, None);
                let hdr =
                    BlockHeader::parse(&postings[range.clone()], meta.block_layout, prev_last);
                let n = decode_block_doc_ids(&postings[range], &hdr, &mut d);
                assert!(
                    prev_last.is_none_or(|p| p < d[0]),
                    "block {i} starts after its predecessor"
                );
                assert!(
                    d[..n].windows(2).all(|w| w[0] < w[1]),
                    "block {i} ascending"
                );
                assert_eq!(d[n - 1], cursor.blocks[i].last_doc_id, "block {i} last doc");
                prev_last = Some(d[n - 1]);
            }
        }
    }

    /// Per block of `common`: the bound its cursor decoded and the exact
    /// maximum score under `view`'s statistics, in block order.
    async fn block_bounds_and_maxima(view: &FtsReader) -> Vec<(f32, f32)> {
        let col = &view.columns[0];
        let mut cursors = view
            .build_term_cursors(0, &["common"], None, false, None, None)
            .await
            .expect("cursors");
        let cursor = cursors.first_mut().expect("term present");
        let idf = cursor.idf_weight;
        let blocks: Vec<(u32, f32)> = cursor
            .blocks
            .iter()
            .map(|b| (b.last_doc_id, b.block_max_bm25))
            .collect();
        let mut exact = vec![0.0f32; blocks.len()];
        let mut blk = 0usize;
        while !cursor.is_exhausted() {
            let doc = cursor.current_doc_id();
            while blk + 1 < blocks.len() && doc > blocks[blk].0 {
                blk += 1;
            }
            let score =
                bm25::score_with_dl_norm_k1(idf, cursor.current_tf(), col.dl_norm_k1.get(doc));
            exact[blk] = exact[blk].max(score);
            cursor.next();
        }
        blocks.iter().map(|b| b.1).zip(exact).collect()
    }

    /// At the statistics the file declares, every stored bound is the
    /// block's maximum to the last bit (plus the one-ULP guard), and a
    /// `k1`/`b` override — the only remaining reason a bound is scored
    /// away from what it was baked at — leaves it sound. Exactness on the
    /// default path is what keeps pruning intact: a common term's block
    /// maxima sit so close together that even a percent of looseness
    /// admits most of the blocks a tight bound would skip.
    #[tokio::test]
    async fn bounds_are_exact_at_the_declared_statistics_and_sound_under_an_override() {
        let reader = realistic_reader(6_000);
        assert_eq!(reader.columns[0].bound_scale, 1.0);
        let baked = block_bounds_and_maxima(&reader).await;
        assert!(baked.len() > 30, "need a long posting list to say anything");
        for (i, &(bound, max)) in baked.iter().enumerate() {
            assert!(
                bound == max.next_up(),
                "block {i}: bound {bound} is not the exact maximum {max} plus one ULP"
            );
        }

        let view = reader.with_bm25_override(bm25::Bm25Params::new(0.9, 0.4));
        assert!(
            view.columns[0].bound_scale > 1.0,
            "an override owes an inflation factor"
        );
        let overridden = block_bounds_and_maxima(&view).await;
        for (i, &(bound, max)) in overridden.iter().enumerate() {
            assert!(
                bound >= max,
                "override block {i}: bound {bound} below max {max}"
            );
        }
    }

    /// The declared average is a rounded fixed-point value. Bounds are
    /// baked at exactly that value and the norm table is built from it,
    /// so an average the fixed point cannot represent exactly still
    /// leaves every bound one ULP above its block's maximum.
    #[tokio::test]
    async fn bounds_stay_exact_when_the_average_is_not_exactly_representable() {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), false).expect("register");
        // 501 tokens over 500 documents: an average of 1.002.
        for doc in 0..500u32 {
            let text = if doc == 250 {
                "common common"
            } else {
                "common"
            };
            b.add_doc(0, doc, text).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let reader = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(reader.columns[0].avgdl(), bm25::stored_avgdl(1.002));
        for (i, (bound, max)) in block_bounds_and_maxima(&reader)
            .await
            .into_iter()
            .enumerate()
        {
            assert!(
                bound == max.next_up(),
                "block {i}: bound {bound} vs max {max}"
            );
        }
    }

    #[tokio::test]
    async fn block_max_bounds_query_time_score() {
        // A length that truncates under the one-byte length quantizer:
        // `dequantize_len(quantize_len(200)) == 192`, so a length-200 doc is
        // scored as if length 192 and scores *higher* than at its true
        // length.
        const DOC_LEN: usize = 200;
        assert!(
            bm25::dequantize_len(bm25::quantize_len(DOC_LEN as u32)) < DOC_LEN as u32,
            "corpus doc length must quantize downward to exercise the bound"
        );
        // The term under test lives in this many docs — enough to span
        // multiple 128-doc blocks so the block-max skip engages.
        const TERM_DOCS: u32 = 260;
        // Total corpus size. Kept well above `TERM_DOCS` so the term's IDF
        // is large enough that the quantization-induced score gap clears the
        // skip table's fixed-point rounding and the assertion is decisive.
        const N_DOCS: u32 = 1300;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        for doc_id in 0..N_DOCS {
            // Every doc is `DOC_LEN` tokens long (so `avgdl == DOC_LEN` and
            // every length quantizes down identically). The term docs carry
            // `common` with a term frequency of 1..=3 — a genuine per-block
            // spread of scores whose maximum is the highest-tf doc — padded
            // with a filler token; the rest are filler only.
            let common_tf = if doc_id < TERM_DOCS {
                1 + (doc_id % 3) as usize
            } else {
                0
            };
            let mut text = String::with_capacity(DOC_LEN * 5);
            for _ in 0..common_tf {
                text.push_str("common ");
            }
            for _ in 0..(DOC_LEN - common_tf) {
                text.push_str("pad ");
            }
            b.add_doc(0, doc_id, text.trim_end()).expect("add doc");
        }
        let bytes = Bytes::from(b.finish().expect("finish builder"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let reader = FtsReader::open(bytes, json).expect("open FtsReader");

        let mut cursors = reader
            .build_term_cursors(0, &["common"], None, false, None, None)
            .await
            .expect("build term cursors");
        let cursor = cursors.first_mut().expect("`common` present in dictionary");
        assert!(
            cursor.blocks.len() >= 2,
            "term must span multiple blocks so the block-max skip engages \
             (got {} block(s))",
            cursor.blocks.len()
        );
        let col_meta = &reader.columns[0];

        let mut checked = 0u32;
        while !cursor.is_exhausted() {
            let doc = cursor.current_doc_id();
            let tf = cursor.current_tf();
            let query_score =
                bm25::score_with_dl_norm_k1(cursor.idf_weight, tf, col_meta.dl_norm_k1.get(doc));
            let block_max = cursor.current_block_max_bm25();
            assert!(
                block_max >= query_score,
                "stored block max {block_max} < query-time score {query_score} for \
                 doc {doc} (tf={tf}): the per-block BM25 bound under-estimates a \
                 document in its own block, so the ranked-OR block-max skip can drop it",
            );
            checked += 1;
            cursor.next();
        }
        assert_eq!(
            checked, TERM_DOCS,
            "every posting for the term must be visited"
        );
    }
}
