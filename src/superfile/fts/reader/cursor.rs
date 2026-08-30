// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Low-level FTS posting cursors: the parsed per-term header/skip table
//! ([`TermMeta`], [`BlockMeta`]) and the block-at-a-time [`TermCursor`]
//! the scorers, phrase walk, and count kernels drive. Scoped `pub(super)`
//! to the `reader/` module — never referenced outside the FTS layer.

use std::sync::Arc;

use bytes::Bytes;

use super::core::{read_u32_le, read_u64_le};
use crate::superfile::{
    ReadError,
    error::FtsError,
    format::{
        self,
        fts::{
            POSITION_SUBINDEX_ENTRIES_PER_BLOCK, POSITION_SUBINDEX_STRIDE, U32_BYTES, U64_BYTES,
            skip_entry, term_meta,
        },
    },
    fts::{
        bm25,
        builder::{SKIP_ENTRY_SIZE, TERM_META_POSITIONAL_SIZE, TERM_META_SIZE},
        posting::{self, BLOCK_LEN, decode_block, decode_block_doc_ids},
    },
};

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
    /// Absolute offset (within the postings region) of the coarse
    /// block-max table — `ceil(num_blocks / COARSE_BLOCK_MAX_SPAN)`
    /// fixed-point `u32`s at the tail of the term region.
    pub(super) coarse_start: usize,
    /// Term-relative end of the last posting block: `postings_length`
    /// minus the coarse table's bytes. The blocks end here; the coarse
    /// table follows.
    pub(super) blocks_end_in_term: usize,
    /// Whether this term carries a coarse block-max table (V5 blobs).
    /// `false` for V1–V4 — the ranked walk then skips the coarse level.
    pub(super) has_coarse: bool,
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
        has_subindex: bool,
        has_coarse: bool,
    ) -> Result<Self, FtsError> {
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
        let skip_end = skip_start + num_blocks * SKIP_ENTRY_SIZE;
        if skip_end > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "skip table runs past postings region".into(),
            )));
        }
        // v3 positional terms store a run-offset sub-index right after the
        // skip table: `num_blocks × ENTRIES_PER_BLOCK` u32s. Bound it now;
        // the blocks follow it (their offsets are read from the skip
        // table, which the writer already shifted past the sub-index).
        let subindex_start = match has_subindex {
            true => {
                let subindex_end =
                    skip_end + num_blocks * POSITION_SUBINDEX_ENTRIES_PER_BLOCK * U32_BYTES;
                if subindex_end > postings.len() {
                    return Err(FtsError::Read(ReadError::MalformedVersion(
                        "position sub-index runs past postings region".into(),
                    )));
                }
                Some(skip_end)
            }
            false => None,
        };
        // Coarse block-max table (V5 only): `ceil(num_blocks / span)` u32s at
        // the tail of the term region, so the blocks end where it begins.
        // V1–V4 blobs have no such table — the blocks run to `postings_length`
        // and the ranked walk skips the coarse level.
        let coarse_size = match has_coarse {
            true => num_blocks.div_ceil(format::fts::COARSE_BLOCK_MAX_SPAN) * U32_BYTES,
            false => 0,
        };
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
            coarse_start,
            blocks_end_in_term,
            has_coarse,
        })
    }

    /// Decode a 4-byte block-max slot (a per-block skip entry's field or a
    /// coarse-table entry) into a guaranteed upper bound on the BM25 score.
    /// V5 stores the exact `f32` bits; legacy `V1`-`V4` store
    /// `ceil(max × scale)` as fixed-point. Both return a value at or above
    /// the true max:
    /// - V5 nudges the exact stored max up one `f32` ULP. The stored value
    ///   equals the reader's per-doc score for the block's max doc, so for
    ///   local scoring it is already an exact bound; the ULP guards the
    ///   cross-superfile idf-rescale multiply, whose f32 rounding could
    ///   otherwise dip a hair below a score-tied doc and drop tied hits.
    /// - Legacy adds one fixed-point step, covering both the `x1000 / scale`
    ///   division rounding and files written before the encode-side `ceil`.
    #[inline]
    fn decode_block_max(&self, raw: u32) -> f32 {
        if self.has_coarse {
            f32::from_bits(raw).next_up()
        } else {
            raw.saturating_add(1) as f32 / format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE
        }
    }

    /// Decode coarse block-max entry `g` into a guaranteed upper bound
    /// on the BM25 score of every block in span `g` (blocks
    /// `[g*SPAN .. (g+1)*SPAN)`). Coarse entries exist only on V5, where
    /// they are `f32` bits.
    #[inline]
    pub(super) fn coarse_entry(&self, postings: &[u8], g: usize) -> f32 {
        let at = self.coarse_start + g * U32_BYTES;
        self.decode_block_max(read_u32_le(&postings[at..at + U32_BYTES]))
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
        let at = start + idx * U32_BYTES;
        let checkpoint = read_u32_le(&postings[at..at + U32_BYTES]);
        let runs_to_skip = pair_in_block % POSITION_SUBINDEX_STRIDE;
        Some((checkpoint, runs_to_skip))
    }

    /// Decode skip-table entry `i` into `(last_doc_id,
    /// block_offset_in_term, block_max_bm25)`. `block_offset_in_term`
    /// is relative to the term's `metadata_offset`; `block_max_bm25`
    /// is recovered from the fixed-point `max_bm25_x1000` field. The
    /// reserved field (entry bytes 12..16) is ignored. Per-entry on
    /// purpose — the single-term BMW walk streams entries without
    /// materializing a `Vec`.
    #[inline]
    pub(super) fn skip_entry(&self, postings: &[u8], i: usize) -> (u32, usize, f32) {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        let entry_off = self.skip_start + i * SKIP_ENTRY_SIZE;
        let last_doc_id = read_u32_le(
            &postings[entry_off + skip_entry::LAST_DOC_ID_OFF
                ..entry_off + skip_entry::LAST_DOC_ID_OFF + U32_BYTES],
        );
        let block_offset = read_u32_le(
            &postings[entry_off + skip_entry::BLOCK_OFFSET_OFF
                ..entry_off + skip_entry::BLOCK_OFFSET_OFF + U32_BYTES],
        ) as usize;
        let block_max_raw = read_u32_le(
            &postings[entry_off + skip_entry::MAX_BM25_OFF
                ..entry_off + skip_entry::MAX_BM25_OFF + U32_BYTES],
        );
        // Decode to a guaranteed upper bound on the block's BM25 (V5 exact
        // f32 + one ULP; legacy fixed-point + one step). The upper-bound
        // guarantee matters for the cross-superfile floor: block-skip
        // compares `block_max <= floor`, and a bound that dips below a
        // score-tied block's true max would let a rising floor skip it,
        // dropping tied hits by completion order (nondeterministic top-k).
        (
            last_doc_id,
            block_offset,
            self.decode_block_max(block_max_raw),
        )
    }

    /// This block's position-run byte offset within the term's
    /// positions bytes — the skip entry's fourth field (zero on
    /// positionless columns, where it is the reserved slot).
    #[inline]
    pub(super) fn positions_block_offset(&self, postings: &[u8], i: usize) -> u32 {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        let entry_off = self.skip_start + i * SKIP_ENTRY_SIZE;
        read_u32_le(
            &postings[entry_off + skip_entry::POSITIONS_BLOCK_OFFSET_OFF
                ..entry_off + skip_entry::POSITIONS_BLOCK_OFFSET_OFF + U32_BYTES],
        )
    }

    /// End offset (relative to the term's `metadata_offset`) of block
    /// `i`'s bytes. Blocks are concatenated back-to-back, so each
    /// block ends where the next one's `block_offset` begins; the last
    /// block ends at `postings_length`.
    #[inline]
    pub(super) fn block_end_in_term(&self, postings: &[u8], i: usize) -> usize {
        if i + 1 < self.num_blocks {
            let next_off = self.skip_start + (i + 1) * SKIP_ENTRY_SIZE;
            read_u32_le(&postings[next_off + 4..next_off + 8]) as usize
        } else {
            // The coarse block-max table follows the last block, so the
            // blocks end before it — not at `postings_length`.
            self.blocks_end_in_term
        }
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
    /// Precomputed `idf * (K1 + 1)` — the score numerator's
    /// per-cursor constant. Computed once at cursor build so the
    /// hot inner loop fits one multiply + add + divide per call.
    /// (The bare `idf` value isn't kept on the cursor — every hot
    /// scoring path uses `score_with_dl_norm_k1` which takes
    /// `idf_x_k1p1` directly.)
    pub(super) idf_x_k1p1: f32,
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
}

impl TermCursor {
    /// Parse one term's metadata + skip table out of its own postings
    /// byte range and decode its first block. `term_bytes` starts at
    /// the term's 20-byte metadata header (offset 0) and runs to the
    /// end of its last block — the contiguous range
    /// [`FtsReader::fetch_term_postings`] fetched for this term.
    pub(super) fn new(
        term_bytes: Bytes,
        n_docs: u64,
        positional: bool,
        global_idf: Option<f32>,
        header_probed: bool,
        count_only: bool,
        has_coarse: bool,
    ) -> Result<Self, FtsError> {
        let postings: &[u8] = term_bytes.as_ref();
        let metadata_offset = 0usize;

        // The plain-term cursor never decodes positions, so it needs no
        // sub-index (it reads block offsets straight from the skip table).
        // `has_coarse` (V5) tells it the last block ends before the coarse
        // table, not at `postings_length`.
        let term_meta = TermMeta::parse(postings, metadata_offset, positional, false, has_coarse)?;
        let local_idf = bm25::idf(n_docs, term_meta.df);
        let idf = global_idf.unwrap_or(local_idf);
        // Stored per-block BMW upper bounds bake in the LOCAL idf. Only a
        // global-idf override needs to rescale them by global/local:
        // block_max = local_idf_x_k1p1 × (an idf-independent tf-factor),
        // so the linear rescale is exact and keeps the BMW skip UBs
        // consistent with the global-idf scores computed from
        // `idf_x_k1p1` below. `None` (the default per-superfile path, and
        // the case where a gathered global idf happens to equal the
        // local one) leaves the stored value untouched — the block loop
        // does no extra work, matching the per-superfile scorer exactly.
        let idf_rescale = match global_idf {
            Some(_) if local_idf > 0.0 && idf != local_idf => Some(idf / local_idf),
            _ => None,
        };

        // Collect straight into the `Arc` allocation: `0..num_blocks` is
        // an exact-size iterator, so this writes each entry in place —
        // one allocation, no intermediate `Vec` + copy. The skip table
        // is ~a quarter of a long term's cursor-build bytes (one 32-byte
        // entry per 128-doc block), so the doubled write showed up on
        // common-term queries.
        let mut term_max_bm25: f32 = 0.0;
        let blocks: Arc<[BlockMeta]> = (0..term_meta.num_blocks)
            .map(|i| {
                let (last_doc_id, block_offset_in_term, raw_block_max) =
                    term_meta.skip_entry(postings, i);
                let block_max_bm25 = match idf_rescale {
                    Some(ratio) => raw_block_max * ratio,
                    None => raw_block_max,
                };
                term_max_bm25 = term_max_bm25.max(block_max_bm25);

                BlockMeta {
                    last_doc_id,
                    block_byte_offset: metadata_offset + block_offset_in_term,
                    block_byte_end: metadata_offset + term_meta.block_end_in_term(postings, i),
                    block_max_bm25,
                }
            })
            .collect();

        let mut cursor = Self {
            idf_x_k1p1: idf * (bm25::K1 + 1.0),
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
        };
        if !cursor.blocks.is_empty() {
            cursor.decode_current_block();
        }
        Ok(cursor)
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
        n_docs: u64,
        dl_norm_k1: f32,
        global_idf: Option<f32>,
    ) -> Self {
        let idf = global_idf.unwrap_or_else(|| bm25::idf(n_docs, 1));
        let idf_x_k1p1 = idf * (bm25::K1 + 1.0);
        let block_max_bm25 = bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1);

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
            idf_x_k1p1,
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
        }
    }

    pub(super) fn decode_current_block(&mut self) {
        let block = self.blocks[self.current_block];
        // Borrow in place rather than clone an owned `Bytes` (disjoint from the
        // `&mut self.block_*` decode targets, which are separate fields).
        let bytes = &self.bytes[block.block_byte_offset..block.block_byte_end];
        // Count-only cursors skip the tf half of the block; the count
        // kernels never read `block_tfs`, so it is left stale.
        self.block_n = match self.count_only {
            true => decode_block_doc_ids(bytes, &mut self.block_doc_ids),
            false => decode_block(bytes, &mut self.block_doc_ids, &mut self.block_tfs),
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
        // Inline (df=1) cursor: single pre-decoded doc, no postings bytes.
        if self.bytes.is_empty() {
            return self.block_n > 0 && self.block_doc_ids[0] == doc;
        }
        let block = self.blocks[self.current_block];
        // Borrow the block's bytes in place — `self.bytes` is held for the
        // cursor's life, so a subslice needs no owned `Bytes` clone. A
        // per-probe `.slice()` here bumps and drops an atomic refcount on
        // every membership probe; over a long driver it was ~11% of the
        // intersection-count time (and wasted on the PACKED path, which
        // only reads the encoding byte before falling to the decode cache).
        let raw = &self.bytes[block.block_byte_offset..block.block_byte_end];
        if raw[posting::ENCODING_OFF] == posting::ENCODING_BITSET {
            let base = read_u32_le(&raw[4..8]);
            if doc < base {
                return false;
            }
            let bit = (doc - base) as usize;
            let tfs_size = BLOCK_LEN * raw[2] as usize / 8;
            let bitset_end = raw.len() - tfs_size;
            let word_at = posting::HEADER_SIZE + (bit / 64) * 8;
            if word_at + 8 > bitset_end {
                return false; // past this block's presence bits ⇒ absent
            }
            let word = u64::from_le_bytes(raw[word_at..word_at + 8].try_into().expect("8 bytes"));
            (word >> (bit % 64)) & 1 == 1
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
    pub(super) fn bitset_probe_tf(&mut self, doc: u32) -> Option<u32> {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < doc
        {
            self.current_block += 1;
        }
        if self.current_block >= self.blocks.len() {
            return None;
        }
        // Inline (df=1) cursor: single pre-decoded posting, no postings bytes.
        if self.bytes.is_empty() {
            if self.block_n > 0 && self.block_doc_ids[0] == doc {
                return Some(self.block_tfs[0]);
            }
            return None;
        }
        let block = self.blocks[self.current_block];
        let raw = &self.bytes[block.block_byte_offset..block.block_byte_end];
        if raw[posting::ENCODING_OFF] != posting::ENCODING_BITSET {
            // PACKED: no rank shortcut — decode + locate like the old path.
            self.skip_to(doc);
            return if self.current_doc_id() == doc {
                Some(self.current_tf())
            } else {
                None
            };
        }
        let base = read_u32_le(&raw[4..8]);
        if doc < base {
            return None;
        }
        let bit = (doc - base) as usize;
        let tf_bits = raw[2] as usize;
        let tfs_size = BLOCK_LEN * tf_bits / 8;
        let bitset_end = raw.len() - tfs_size;
        let word_idx = bit / 64;
        let word_at = posting::HEADER_SIZE + word_idx * 8;
        if word_at + 8 > bitset_end {
            return None; // past this block's presence bits ⇒ absent
        }
        let word = u64::from_le_bytes(raw[word_at..word_at + 8].try_into().expect("8 bytes"));
        if (word >> (bit % 64)) & 1 == 0 {
            return None; // doc not present in this block
        }
        // Present. `rank` = number of set bits before `bit` = popcount of the
        // whole presence words ahead of this one + popcount of this word below
        // `bit`. The r-th set bit (doc) maps to the r-th tf in doc order.
        let presence = &raw[posting::HEADER_SIZE..bitset_end];
        let mut rank: u32 = 0;
        for w in presence[..word_idx * 8].chunks_exact(8) {
            rank += u64::from_le_bytes(w.try_into().expect("8 bytes")).count_ones();
        }
        let below = if bit.is_multiple_of(64) {
            0u64
        } else {
            (1u64 << (bit % 64)) - 1
        };
        rank += (word & below).count_ones();
        // Decode this block's tf array once (doc order), reused across a run of
        // candidates in the same block; the doc ids are never expanded.
        if self.tf_decoded_block != self.current_block {
            let bytes = &self.bytes[block.block_byte_offset..block.block_byte_end];
            posting::decode_block_tfs(bytes, &mut self.block_tfs);
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
        if self.is_exhausted() || self.pos >= self.block_n {
            u32::MAX
        } else {
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
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

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
            .build_term_cursors(0, &["common"], None, false)
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
                bm25::score_with_dl_norm_k1(cursor.idf_x_k1p1, tf, col_meta.dl_norm_k1.get(doc));
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
