// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Format-spec primitives: magic byte sequences, version strings, KV
//! metadata key constants. Anything that defines what bytes go where in a
//! superfile lives here.

pub mod checksum;
pub mod footer;

/// 3-byte project magic shared by every section.
pub const PROJECT_MAGIC: &[u8; 3] = b"INF";

/// File-format version. Semver string. Bump major to break compatibility.
/// 1.1.0: `inf.fts.columns` entries may carry `"stored":false` (index-only
/// FTS columns, absent from the Parquet body); the field is emitted only
/// when false, and same-major readers default a missing field to true.
pub const FORMAT_VERSION: &str = "1.1.0";

/// CRC width in bytes (`u32` CRC-32C, little-endian) appended after a
/// directory or after a subsection's payload. Defined once so the
/// writer and reader arithmetic agree symbolically rather than via
/// duplicated `+ 4 /* CRC */` literals.
pub const CRC_BYTES: usize = 4;

/// FTS section magic bytes and constants.
pub mod fts {
    use crate::superfile::fts::posting::BLOCK_LEN;

    /// 8-byte magic at the start of the FTS blob: `INF` + `FTS` +
    /// `01`. The trailing `01` is a fixed part of the section
    /// identity, **not** a version — it never changes across blob
    /// versions (v2 blobs carry this same magic). The blob's version
    /// is the `u32` at [`hdr::VERSION_OFF`], and only that field.
    pub const MAGIC: &[u8; 8] = b"INFFTS01";
    /// Legacy blob version: the positionless layout with the 48-byte
    /// header. **Read-only** — files written before the positions
    /// region existed carry it and stay readable until support is
    /// explicitly dropped; new code always writes
    /// [`VERSION_V2`].
    pub const VERSION_V1_LEGACY: u32 = 1;

    /// The version new code writes: the header grows to
    /// [`HEADER_SIZE_V2`] with the positions-region offset at
    /// [`hdr::POSITIONS_OFFSET_OFF`], and a positions region —
    /// empty unless a column records positions — sits between the
    /// postings region and the doc-lengths directory. Readers accept
    /// both versions.
    pub const VERSION_V2: u32 = 2;

    /// The version new code writes when a column stores positions: same
    /// header and region layout as [`VERSION_V2`], but each positional
    /// term's region gains a **position run-offset sub-index** between its
    /// skip table and its posting blocks. The sub-index stores, every
    /// [`POSITION_SUBINDEX_STRIDE`] pairs within a block, the byte offset
    /// of that pair's position run (relative to the term's positions), so
    /// the reader reaches a pair's positions by skipping `< STRIDE` runs
    /// instead of walking every run from the block start. Readers accept
    /// `V1`/`V2`/`V3`; `V1`/`V2` files (no sub-index) stay readable
    /// unchanged, so existing indices need no reindex. A column *without*
    /// positions is written as [`VERSION_V2`] — the sub-index only exists
    /// where positions do.
    pub const VERSION_V3: u32 = 3;

    /// The version written when any posting block is stored in the **bitset
    /// encoding**: a dense block's doc ids are a presence bitset (header
    /// byte 3 = [`crate::superfile::fts::posting::ENCODING_BITSET`]) rather
    /// than PFOR deltas, so the union count OR's it in without decoding.
    /// Same header + region layout as [`VERSION_V2`]/[`VERSION_V3`]
    /// (positions region present iff positional; sub-index for positional
    /// terms as in `V3`) — `V4` adds only the per-block encoding choice,
    /// which is self-describing via the header byte. Readers accept
    /// `V1`–`V4`; `V1`–`V3` blobs carry only PACKED blocks and read
    /// unchanged, so existing indices need no reindex.
    pub const VERSION_V4: u32 = 4;

    /// The version new code writes: everything `V4` allows (positions region
    /// iff positional, `V3` sub-index for positional terms, bitset blocks
    /// self-describing per block) **plus** two block-max changes:
    ///
    /// 1. Each skip-table entry stores the per-block max BM25 as an **exact
    ///    little-endian `f32`** (the 4-byte slot that held `V1`–`V4`'s
    ///    `ceil`-quantised fixed-point `u32`). It equals the reader's per-doc
    ///    score for the block's max doc (same quantized-length scoring), so it
    ///    is an exact upper bound with none of the fixed-point `ceil` slack.
    /// 2. A per-term **coarse block-max table** at the tail of each PFOR
    ///    term's postings region — one `f32` per [`COARSE_BLOCK_MAX_SPAN`]
    ///    blocks, the span's max of the per-block maxes — giving the ranked
    ///    walk a second, coarser skip level.
    ///
    /// The header + region layout is otherwise identical to `V2`–`V4`.
    /// Readers accept `V1`–`V5` and gate the block-max decode on the version:
    /// `V1`–`V4` blobs decode the fixed-point `u32` (and carry no coarse
    /// table), so existing indices read unchanged and need no reindex.
    pub const VERSION_V5: u32 = 5;

    /// The version new code writes. Byte-for-byte the [`VERSION_V5`]
    /// layout; what changes is the **scale** of the stored bounds and the
    /// **average document length** they and the file's scoring are
    /// expressed at.
    ///
    /// `V1`–`V5` bounds are maxima of `idf · tf · (k1 + 1) / (tf + k1 ·
    /// norm)`. `V6` drops the `(k1 + 1)` factor, so a bound is a maximum
    /// of `idf · tf / (tf + k1 · norm)` — the same quantity the scorer
    /// now produces, and the one a BM25 implementation is conventionally
    /// expected to report. The factor was a constant multiplier on every
    /// score in a query, so it never changed a ranking; it did make
    /// every published score a fixed multiple of what the same `k1` and
    /// `b` produce elsewhere, which matters to anything reading the
    /// number rather than the order — a score threshold, a weighted
    /// fusion against vector distances, a comparison against another
    /// engine.
    ///
    /// The average a `V6` file declares in its doc-lengths directory is
    /// the one to score it at: the writer bakes it as the table-wide
    /// average over the documents that carry tokens, folding in every
    /// superfile committed before it, so a query needs no other value
    /// and the stored bounds — exact scores at that average — stay
    /// exact with nothing to inflate. A `V5` file declares its own
    /// row-count average, which the reader corrects on open.
    ///
    /// Readers accept `V1`–`V6`. An older blob's bounds are still exact
    /// upper bounds in their own scale, and the reader brings them into
    /// this one by folding `1 / (k1 + 1)` into the column's bound
    /// correction — so existing indices read unchanged, keep their
    /// pruning power, and need no reindex. Getting that gate wrong in
    /// the other direction (treating a `V6` blob as older) would divide
    /// a bound that is already correct and silently prune documents out
    /// of the top-k, which is why the scale is version-stamped rather
    /// than inferred.
    pub const VERSION_V6: u32 = 6;

    /// The version new code writes. Same header, regions, bound scale
    /// and declared average as [`VERSION_V6`]; what changes is how a
    /// **rare term** is laid out, and one bit of every dictionary value.
    ///
    /// A term whose whole posting list fits one block (`df <=
    /// BLOCK_LEN`) no longer pays the long-form fixed cost — the 20/32
    /// byte metadata header, a skip entry, a position sub-index row, a
    /// coarse slot and a block header, 92 bytes before the first posting
    /// on a positional column — nor the block codec's padding of a
    /// partial block to `BLOCK_LEN` lanes, which on a term with two docs
    /// far apart is hundreds of bytes for two doc ids. It is written in
    /// the **short form** instead (`fts::short`): a varint `df`, a
    /// tf-equals-one bitmap, the doc-id deltas as group-varint, the
    /// remaining tfs as varints and, on a positional column, the term's
    /// position offset and length. A few bytes per posting, no lane
    /// padding, nothing per block. On a Zipfian corpus the single-block
    /// terms are ~97% of the dictionary and were more than half of the
    /// postings region; they are read once and whole, so the reader
    /// decodes a short body into the same pre-filled single-block cursor
    /// the df=1 inline form already uses.
    ///
    /// The term dictionary is no longer an FST: it is sorted,
    /// front-coded term blocks behind a first-key index (`fts::dict`),
    /// whose entries carry the short/long form explicitly and the
    /// metadata offset as a delta — a third smaller than the FST for
    /// the same terms. Readers select the layout by this version;
    /// `V1`–`V6` blobs keep their FST and its packed values.
    /// Multi-block terms change in two fixed costs. A block's header is
    /// one 4-byte word (`posting::BlockLayout::Compact`): the base doc
    /// id is the previous block's last doc id, which the skip table
    /// already holds, and a patched block's exception counts ride in the
    /// word. A skip entry carries the block's byte length instead of its
    /// offset ([`SkipLayout::Length`]) and drops the positions field on
    /// a positionless column; each coarse slot gains its span's start
    /// offset so a random block is still reached in constant work.
    ///
    /// Readers accept `V1`–`V7`.
    pub const VERSION_V7: u32 = 7;

    /// Stride of the position run-offset sub-index ([`VERSION_V3`]): one
    /// stored offset per this many pairs within a posting block. A decode
    /// skips at most `STRIDE - 1` runs from the nearest sub-index entry.
    /// Divides evenly into the posting-block length so every block's
    /// sub-index has `ceil(pairs_in_block / STRIDE)` entries.
    pub const POSITION_SUBINDEX_STRIDE: usize = 16;

    /// Sub-index run-offset checkpoints stored per posting block
    /// ([`VERSION_V3`]): the whole-block entry count, one every
    /// [`POSITION_SUBINDEX_STRIDE`] pairs across a full posting block.
    /// Both the writer's per-term sub-index sizing and the reader's flat
    /// `block * ENTRIES + slot` indexing derive from this single value, so
    /// they stay in lockstep.
    pub const POSITION_SUBINDEX_ENTRIES_PER_BLOCK: usize = BLOCK_LEN / POSITION_SUBINDEX_STRIDE;

    /// Bytes per stored document length from [`VERSION_V7`]: a `u16`,
    /// saturating at [`DOC_LENGTH_STORED_MAX`], instead of the `u32`
    /// `V1`–`V6` stored. The scorer reads a one-byte bucket of the length
    /// and the directory carries the exact average, so nothing about
    /// scoring changes; a document past 65,535 tokens has its stored
    /// length (and so its stored bucket and bound) computed from the
    /// saturated value, consistently on the writer and the reader.
    pub const DOC_LENGTH_BYTES_V7: usize = 2;
    /// Largest per-document length a `V7` blob stores.
    pub const DOC_LENGTH_STORED_MAX: u32 = u16::MAX as u32;

    /// Fixed-point scale for the per-column average document length.
    /// The builder stores `round(avgdl × 1000)` in the doc-lengths
    /// directory as a `u32` (`avgdl_x1000`); the reader recovers the
    /// `f32` average length by dividing by this. Defined once so the
    /// write and read paths share one scale.
    pub const AVGDL_FIXED_POINT_SCALE: f32 = 1000.0;

    /// **Legacy** fixed-point scale for a posting block's max-BM25 upper
    /// bound, used only by `V1`–`V4` blobs. Those store `ceil(max_bm25 ×
    /// this)` as a `u32` in each skip-table entry; the reader recovers the
    /// bound by dividing by this and adding one step (a safety margin for
    /// files written before the encode-side `ceil`).
    ///
    /// `V5` no longer uses a fixed-point scale at all — it stores the block
    /// max as an **exact `f32`** (see [`VERSION_V5`]). The `ceil`-to-`u32`
    /// quantization at scale 1000 rounded the bound *up* by up to ~0.002,
    /// which for a low-idf term like "the" (BM25 ~0.1–0.3) is a large
    /// *relative* inflation that blocks skips a tight bound would allow.
    /// Storing the exact `f32` (same 4 bytes, matching the precision of the
    /// reader's per-doc scoring) removes that slack.
    pub const BLOCK_MAX_BM25_FIXED_POINT_SCALE: f32 = 1000.0;

    /// Offset of `avgdl_x1000` within a doc-lengths directory entry
    /// (`[0..4]` column id, `[4..12]` array offset, `[12..16]` this).
    pub const DOC_LENGTHS_ENTRY_AVGDL_OFF: usize = 12;

    /// Number of consecutive posting blocks summarised by one entry of
    /// a term's coarse block-max table (V5 and later). The table sits at
    /// the tail of a PFOR term's postings region: `ceil(num_blocks / this)`
    /// `f32`s, each the max of its span's per-block max BM25.
    ///
    /// It gives the ranked single-term walk a second, coarser skip level:
    /// when the running k-th-best score already dominates a whole span's
    /// upper bound, the walk jumps the span in one comparison instead of
    /// touching each block's skip entry. On a very long, heavily-skipped
    /// posting list (a common term at small k) the per-block skip scan is
    /// itself the dominant cost; the coarse level removes ~31/32 of it.
    /// The span is a coarse-max of already-`ceil`-quantised block bounds,
    /// so it stays a true upper bound and the top-k is unchanged.
    pub const COARSE_BLOCK_MAX_SPAN: usize = 32;

    /// Total FTS blob header size in bytes for [`VERSION_V1_LEGACY`] (no
    /// positions). The FST directory begins immediately after this
    /// fixed-size header.
    pub const HEADER_SIZE_V1_LEGACY: usize = 48;

    /// Header size for [`VERSION_V2`]: the v1 fields plus the
    /// trailing positions-region offset (`u64` at
    /// [`hdr::POSITIONS_OFFSET_OFF`]).
    pub const HEADER_SIZE_V2: usize = 56;

    /// Width of the 8-byte FTS magic field.
    pub const MAGIC_BYTES: usize = 8;
    /// Width of a little-endian `u32` header field.
    pub const U32_BYTES: usize = 4;
    /// Width of a little-endian `u64` header field.
    pub const U64_BYTES: usize = 8;

    /// FTS blob header field offsets (48-byte header):
    ///
    /// ```text
    /// [ 0.. 8] MAGIC
    /// [ 8..12] version (u32 LE)
    /// [12..16] n_columns (u32 LE)
    /// [16..20] n_docs (u32 LE)
    /// [20..24] n_terms_total (u32 LE)
    /// [24..32] fst_offset (u64 LE)
    /// [32..40] postings_offset (u64 LE)
    /// [40..48] doc_lengths_table_offset (u64 LE)
    /// ```
    pub mod hdr {
        /// `[8..12]` format version (`u32` LE).
        pub const VERSION_OFF: usize = 8;
        /// `[12..16]` column count (`u32` LE).
        pub const N_COLUMNS_OFF: usize = 12;
        /// `[16..20]` document count (`u32` LE).
        pub const N_DOCS_OFF: usize = 16;
        /// `[20..24]` total distinct `(column, term)` pairs (`u32` LE).
        pub const N_TERMS_OFF: usize = 20;
        /// `[24..32]` FST body offset (`u64` LE).
        pub const FST_OFFSET_OFF: usize = 24;
        /// `[32..40]` postings region offset (`u64` LE).
        pub const POSTINGS_OFFSET_OFF: usize = 32;
        /// `[40..48]` doc-lengths directory offset (`u64` LE).
        pub const DOC_LENGTHS_DIR_OFF: usize = 40;
        /// `[48..56]` positions region offset (`u64` LE).
        /// [`VERSION_V2`](super::VERSION_V2) headers
        /// only — a v1 header ends at
        /// [`HEADER_SIZE_V1_LEGACY`](super::HEADER_SIZE_V1_LEGACY). The region sits
        /// between the postings region and the doc-lengths directory
        /// so the lazy-open doc-lengths tail fetch stays small.
        pub const POSITIONS_OFFSET_OFF: usize = 48;
    }

    /// Per-term metadata header field offsets (relative to a term's
    /// `metadata_offset`):
    ///
    /// ```text
    /// [ 0.. 4] df (u32 LE)
    /// [ 4..12] self-offset (u64 LE, redundant)
    /// [12..16] postings_length (u32 LE)
    /// [16..20] num_blocks (u32 LE)
    /// ```
    ///
    /// Terms of a **positional column** carry an extended 32-byte
    /// header — the 20-byte layout above plus:
    ///
    /// ```text
    /// [20..28] positions_offset (u64 LE, absolute in the positions region)
    /// [28..32] positions_length (u32 LE)
    /// ```
    ///
    /// The column's positions flag (from `inf.fts.columns`) selects
    /// the stride; the two layouts never mix within one column.
    pub mod term_meta {
        /// `[0..4]` document frequency (`u32` LE).
        pub const DF_OFF: usize = 0;
        /// `[12..16]` total byte length of the term's postings (`u32` LE).
        pub const POSTINGS_LENGTH_OFF: usize = 12;
        /// `[16..20]` number of PFOR blocks / skip-table entries (`u32` LE).
        pub const NUM_BLOCKS_OFF: usize = 16;
        /// `[20..28]` absolute offset of this term's position bytes in
        /// the positions region (`u64` LE). Positional columns only.
        pub const POSITIONS_OFFSET_OFF: usize = 20;
        /// `[28..32]` byte length of this term's position bytes
        /// (`u32` LE). Positional columns only.
        pub const POSITIONS_LENGTH_OFF: usize = 28;
    }

    /// Which header a posting block carries — by blob version.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BlockLayout {
        /// `V1`–`V6`: the 8-byte header with the base doc id stored.
        Wide,
        /// `V7`+: the 4-byte header word; a packed or patched block's base
        /// doc id is the previous block's last doc id (zero for the first
        /// block), a bitset block's origin follows the word.
        Compact,
    }

    /// How the term dictionary lays its terms out — by blob version.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum DictLayout {
        /// `V1`–`V6`: one FST keyed `column <SEP> term`, values packed as
        /// `fts::fst_value` describes.
        Fst,
        /// `V7`+: front-coded term blocks behind a fixed-width first-key
        /// table (see `fts::dict`).
        Blocks,
    }

    /// Everything a blob version decides about how its regions are laid
    /// out — the one table the writer (by the era it writes) and the
    /// reader (by the version it opened) both consult, so the version
    /// ladder is spelled out once.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct BlobLayout {
        /// Each long-form term ends with a coarse block-max table, one
        /// slot per [`COARSE_BLOCK_MAX_SPAN`] blocks (`V5`+).
        pub coarse: bool,
        /// A single-block term is written in the short form (`V7`+).
        pub short_form: bool,
        /// Positions are per-block groups (`V7`+); a grouped blob carries
        /// no run-offset sub-index.
        pub grouped_positions: bool,
        /// A positional long-form term carries a run-offset sub-index
        /// between its skip table and its blocks (`V3`–`V6`).
        pub position_subindex: bool,
        /// Dense blocks may take the presence-bitset encoding (`V4`+).
        pub bitset_blocks: bool,
        pub block: BlockLayout,
        pub skip: SkipLayout,
        pub dict: DictLayout,
        /// Bytes per stored document length.
        pub doc_length_bytes: usize,
    }

    impl BlobLayout {
        /// The layout of blob `version`, or `None` for a version this
        /// crate does not know.
        pub fn for_version(version: u32) -> Option<Self> {
            let legacy = Self {
                coarse: false,
                short_form: false,
                grouped_positions: false,
                position_subindex: false,
                bitset_blocks: false,
                block: BlockLayout::Wide,
                skip: SkipLayout::Absolute,
                dict: DictLayout::Fst,
                doc_length_bytes: U32_BYTES,
            };
            Some(match version {
                VERSION_V1_LEGACY | VERSION_V2 => legacy,
                VERSION_V3 => Self {
                    position_subindex: true,
                    ..legacy
                },
                VERSION_V4 => Self {
                    position_subindex: true,
                    bitset_blocks: true,
                    ..legacy
                },
                VERSION_V5 | VERSION_V6 => Self {
                    coarse: true,
                    position_subindex: true,
                    bitset_blocks: true,
                    ..legacy
                },
                VERSION_V7 => Self {
                    coarse: true,
                    short_form: true,
                    grouped_positions: true,
                    position_subindex: false,
                    bitset_blocks: true,
                    block: BlockLayout::Compact,
                    skip: SkipLayout::Length,
                    dict: DictLayout::Blocks,
                    doc_length_bytes: DOC_LENGTH_BYTES_V7,
                },
                _ => return None,
            })
        }
    }

    /// How a term's skip table locates its blocks — by blob version.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SkipLayout {
        /// `V1`–`V6`: 16-byte entries carrying each block's absolute byte
        /// offset within the term; 4-byte coarse slots.
        Absolute,
        /// `V7`+: an entry carries the block's byte **length** (`u16`) in
        /// place of its offset — 10 bytes, 14 on a positional column —
        /// and each 8-byte coarse slot adds the offset of its span's
        /// first block. A sequential walk accumulates lengths; a random
        /// block is one slot read plus at most `COARSE_BLOCK_MAX_SPAN - 1`
        /// lengths summed.
        Length,
    }

    impl SkipLayout {
        /// Bytes one skip entry takes on a column with or without
        /// positions.
        pub fn entry_bytes(self, positional: bool) -> usize {
            match (self, positional) {
                (Self::Absolute, _) => 16,
                (Self::Length, true) => 14,
                (Self::Length, false) => 10,
            }
        }

        /// Bytes one coarse block-max slot takes.
        pub fn coarse_slot_bytes(self) -> usize {
            match self {
                Self::Absolute => U32_BYTES,
                Self::Length => 2 * U32_BYTES,
            }
        }

        /// Entry offset of the block-max bound (`u32` LE).
        pub fn bound_off(self) -> usize {
            match self {
                Self::Absolute => 8,
                Self::Length => 6,
            }
        }

        /// Entry offset of the block's position-group offset (`u32` LE,
        /// positional columns).
        pub fn positions_off(self) -> usize {
            match self {
                Self::Absolute => 12,
                Self::Length => 10,
            }
        }
    }

    /// Skip-table entry field offsets (relative to the entry start).
    ///
    /// [`SkipLayout::Absolute`]:
    ///
    /// ```text
    /// [ 0.. 4] last_doc_id (u32 LE)
    /// [ 4.. 8] block_offset (u32 LE, relative to term metadata start)
    /// [ 8..12] block-max bound (u32 LE)
    /// [12..16] positions_block_offset (u32 LE; positional columns)
    /// ```
    ///
    /// [`SkipLayout::Length`]:
    ///
    /// ```text
    /// [ 0.. 4] last_doc_id (u32 LE)
    /// [ 4.. 6] block_len (u16 LE, the block's encoded bytes)
    /// [ 6..10] block-max bound (u32 LE)
    /// [10..14] positions_block_offset (u32 LE; positional columns only)
    /// ```
    ///
    /// A block's offset under the length layout is the coarse slot's
    /// span start plus the lengths of the span's earlier blocks. The
    /// positions field records the byte offset of this block's position
    /// group, relative to the term's `positions_offset` — per-block
    /// random access into the term's position bytes, aligned with the
    /// doc blocks. An absolute-layout positionless column writes zero
    /// there (the field's reserved era); a length-layout one omits it.
    pub mod skip_entry {
        /// `[0..4]` largest doc-id in the block (`u32` LE), both layouts.
        pub const LAST_DOC_ID_OFF: usize = 0;
        /// `[4..8]` byte offset to the encoded block (`u32` LE), absolute
        /// layout.
        pub const BLOCK_OFFSET_OFF: usize = 4;
        /// `[4..6]` byte length of the encoded block (`u16` LE), length
        /// layout.
        pub const BLOCK_LEN_OFF: usize = 4;
    }

    /// Coarse slot field offsets (relative to the slot start): the
    /// span's block-max bound, and under [`SkipLayout::Length`] the byte
    /// offset (relative to term metadata start) of the span's first
    /// block.
    pub mod coarse_slot {
        /// `[0..4]` span bound (`u32` LE: `f32` bits from V5, fixed point
        /// before).
        pub const BOUND_OFF: usize = 0;
        /// `[4..8]` span start offset (`u32` LE), length layout only.
        pub const SPAN_START_OFF: usize = 4;
    }
}

/// Vector section magic bytes and constants.
pub mod vec {
    /// 8-byte magic at the start of the vector blob's outer header.
    pub const OUTER_MAGIC: &[u8; 8] = b"INFVEC01";
    /// 8-byte magic at the start of each per-column subsection.
    pub const SUB_MAGIC: &[u8; 8] = b"INFVECC1";
    /// Doc-id width in bytes (`u32` little-endian) stored after each
    /// per-cluster code block. A per-cluster block row is `code_bytes`
    /// of quantized code followed by [`DOC_ID_BYTES`] of doc-id, so the
    /// stride is `code_bytes + DOC_ID_BYTES`.
    pub const DOC_ID_BYTES: usize = 4;
    /// Width in bytes of an inline stable `_id` (an `i128` Snowflake value,
    /// little-endian). The materialized (hidden-cell) build inserts an
    /// `n_docs`-long region of these *between* the codec-meta region and the
    /// per-cluster blocks, indexed by `local_doc_id`, so an id+score query (and
    /// the drain) can read the stable `_id` straight from the cell blob instead
    /// of resolving it through a scalar `_id` column. Placing it before the
    /// per-cluster blocks keeps that (trailing) region the sole input to the
    /// reader's `n_docs` derivation; the region's presence and size are then
    /// self-describing from the offset gap `per_cluster_blocks_off −
    /// codec_meta_end` (`0` or `n_docs * STABLE_ID_BYTES`), needing no header
    /// flag. The streaming/merge builds emit no such region.
    pub const STABLE_ID_BYTES: usize = 16;
    /// Outer-blob version for the single-column IVF layout (one
    /// subsection directory entry per vector column). Written at
    /// bytes [8..12] of the outer header.
    pub const VERSION: u32 = 1;
    /// Outer-blob version for the multi-cell IVF layout: one logical
    /// vector column whose blob packs many complete cell-IVF
    /// subsections behind a cell directory of
    /// `(global_cell_id, subsection_off, subsection_len)`.
    pub const VERSION_MULTI_CELL: u32 = 2;

    /// subsection layout version stamped at
    /// bytes [8..12] of each per-column sub-header.
    ///
    /// On-disk shape:
    ///
    /// ```text
    /// [sub_header][summary_centroid][centroids][cluster_idx]
    ///   [codec_meta]                              ← open-time region
    ///   [per-cluster blocks: each = codes_chunk + doc_ids_chunk]
    ///   [full]                                    ← rerank column
    ///   [crc]
    /// ```
    ///
    /// Two wins land together because they ride on the same
    /// layout (no version skew to manage):
    ///
    /// 1. **Open-time region contiguous** at the head of the
    ///    subsection. One range fetch covers everything search
    ///    needs before picking a cluster (~1.5 MB at 1M × 384
    ///    sq8, ~16 MB at 10M × 1024 sq8).
    /// 2. **Per-cluster `codes + doc_ids` interleave.** One range
    ///    fetch per probed cluster covers both. Each block is
    ///    `count[c] * (code_bytes + 4)` bytes; the existing
    ///    `cluster_index[c] = (doc_off, count)` is enough to
    ///    address it (block byte offset =
    ///    `doc_off * (code_bytes + 4)`).
    ///
    /// Sub-header byte layout (56 bytes):
    ///
    /// ```text
    /// [ 0.. 8] SUB_MAGIC
    /// [ 8..12] SUBSECTION_VERSION
    /// [12..16] codec_meta_size (u32 LE) — 0 when no codec_meta
    ///                                     (Fp32 / RabitqOnly)
    /// [16..24] summary_centroid_offset (u64 LE)
    /// [24..32] reserved (u64)
    /// [32..40] centroids_off (u64 LE)
    /// [40..48] cluster_idx_off (u64 LE)
    /// [48..56] per_cluster_blocks_off (u64 LE)
    /// ```
    ///
    /// Derived offsets (computed by the reader at open):
    /// - `codec_meta_off = cluster_idx_off + n_cent * 8`
    ///   when `codec_meta_size > 0`, else unused.
    /// - `full_off = per_cluster_blocks_off + n_docs * (code_bytes + 4)`.
    /// - per-cluster block at byte offset
    ///   `per_cluster_blocks_off + doc_off[c] * (code_bytes + 4)`,
    ///   block size `count[c] * (code_bytes + 4)`.
    ///
    /// Only this version is accepted on read; a superfile stamped
    /// with any other value at this slot is rejected as malformed
    /// rather than carrying an alternate parse path.
    pub const SUBSECTION_VERSION: u32 = 2;

    /// Width of a little-endian `u32` field in the vector blob.
    pub const U32_BYTES: usize = 4;
    /// Width of a little-endian `u64` field in the vector blob.
    pub const U64_BYTES: usize = 8;
    /// Width of the 8-byte section/sub-section magic.
    pub const MAGIC_BYTES: usize = 8;

    /// Outer-header size: magic + version + n_columns/n_cells + n_docs +
    /// dir_offset. Same 32-byte shape for v1 and v2; the u32 at
    /// [`outer_hdr::N_COLUMNS_OFF`] is `n_columns` for v1 and
    /// `n_cells` for v2.
    pub const OUTER_HEADER_SIZE: usize = 32;
    /// Per-column subsection-directory entry size in bytes (v1).
    pub const DIR_ENTRY_SIZE: usize = 64;
    /// Per-cell directory entry size in bytes (v2 multi-cell):
    /// `global_cell_id (u32) + subsection_off (u64) + subsection_len (u64)
    /// + codec_id (u32)` = 24 bytes.
    pub const CELL_DIR_ENTRY_SIZE: usize = 24;
    /// Per-column sub-header size (inside each subsection).
    pub const SUB_HEADER_SIZE: usize = 56;

    /// On-disk `metric_id` discriminator for squared-L2 distance.
    pub const METRIC_ID_L2SQ: u32 = 0;
    /// On-disk `metric_id` discriminator for cosine distance.
    pub const METRIC_ID_COSINE: u32 = 1;
    /// On-disk `metric_id` discriminator for negated dot product.
    pub const METRIC_ID_NEGDOT: u32 = 2;

    /// Cluster-index entry size: `(doc_off: u32, count: u32)`.
    pub const CLUSTER_IDX_ENTRY_BYTES: usize = 8;
    /// Byte offset of the `count` field within a cluster-index entry
    /// (it is the second `u32` of the pair).
    pub const CLUSTER_IDX_COUNT_OFFSET: usize = 4;

    /// Outer-header field offsets (see the byte map above).
    pub mod outer_hdr {
        /// `[8..12]` outer-blob version (`u32` LE).
        pub const VERSION_OFF: usize = 8;
        /// `[12..16]` column count (v1) or cell count (v2) (`u32` LE).
        pub const N_COLUMNS_OFF: usize = 12;
        /// Alias for [`N_COLUMNS_OFF`] when reading a v2 multi-cell blob.
        pub const N_CELLS_OFF: usize = 12;
        /// `[16..24]` document count (`u64` LE).
        pub const N_DOCS_OFF: usize = 16;
        /// `[24..32]` directory byte offset (`u64` LE).
        pub const DIR_OFFSET_OFF: usize = 24;
    }

    /// Per-cell directory-entry field offsets (24-byte entry, v2).
    pub mod cell_dir_entry {
        /// `[+0..+4]` global cell id (`u32` LE).
        pub const CELL_ID_OFF: usize = 0;
        /// `[+4..+12]` subsection byte offset relative to blob start (`u64` LE).
        pub const SUBSECTION_OFF_OFF: usize = 4;
        /// `[+12..+20]` subsection byte length (`u64` LE).
        pub const SUBSECTION_LEN_OFF: usize = 12;
        /// `[+20..+24]` rerank codec (`u32` LE).
        pub const RESERVED_OFF: usize = 20;
        /// Semantic alias for [`RESERVED_OFF`] used by new readers/writers.
        pub const CODEC_ID_OFF: usize = RESERVED_OFF;
    }

    /// Per-column directory-entry field offsets (64-byte entry).
    pub mod dir_entry {
        /// `[+4..+8]` vector dimension (`u32` LE).
        pub const DIM_OFF: usize = 4;
        /// `[+8..+12]` IVF centroid count (`u32` LE).
        pub const N_CENT_OFF: usize = 8;
        /// `[+12..+16]` metric id (`u32` LE).
        pub const METRIC_ID_OFF: usize = 12;
        /// `[+16..+24]` rotation seed (`u64` LE).
        pub const ROT_SEED_OFF: usize = 16;
        /// `[+24..+32]` subsection byte offset (`u64` LE).
        pub const SUBSECTION_OFF_OFF: usize = 24;
        /// `[+32..+40]` subsection byte length (`u64` LE).
        pub const SUBSECTION_LEN_OFF: usize = 32;
        /// `[+40..+48]` absolute summary offset (`u64` LE).
        pub const SUMMARY_ABS_OFF: usize = 40;
        /// `[+52]` rerank-codec discriminator byte.
        pub const CODEC_ID_OFF: usize = 52;
        /// `[+56..+60]` codec-meta offset within the subsection (`u32` LE).
        pub const CODEC_META_OFF_OFF: usize = 56;
        /// `[+60..+64]` codec-meta size (`u32` LE).
        pub const CODEC_META_SIZE_OFF: usize = 60;
    }

    /// Per-column sub-header field offsets (56-byte header).
    pub mod sub_hdr {
        /// `[8..12]` subsection layout version (`u32` LE).
        pub const VERSION_OFF: usize = 8;
        /// `[12..16]` codec-meta size (`u32` LE).
        pub const CODEC_META_SIZE_OFF: usize = 12;
        /// `[16..24]` summary-centroid offset (`u64` LE).
        pub const SUMMARY_OFF_OFF: usize = 16;
        // `[24..32]` reserved (`u64`).
        /// `[32..40]` centroids offset (`u64` LE).
        pub const CENTROIDS_OFF_OFF: usize = 32;
        /// `[40..48]` cluster-index offset (`u64` LE).
        pub const CLUSTER_IDX_OFF_OFF: usize = 40;
        /// `[48..56]` per-cluster-blocks offset (`u64` LE).
        pub const PER_CLUSTER_BLOCKS_OFF_OFF: usize = 48;
    }
}

/// Parquet KV metadata keys, all prefixed `inf.` to match the project magic.
pub mod kv {
    /// Required: marker that this Parquet file is an infino superfile.
    /// Always `"infino-superfile"`.
    pub const FORMAT: &str = "inf.format";

    /// Required: format-version string (e.g. `"1.0.0"`).
    pub const FORMAT_VERSION: &str = "inf.format_version";

    /// Required: name of the schema column serving the `id` role.
    pub const ID_COLUMN: &str = "inf.id_column";

    /// Required: total document count in this superfile (string-encoded u64).
    pub const N_DOCS: &str = "inf.n_docs";

    /// Required: writer library + version + git commit (auto-populated at
    /// compile time via `build.rs`).
    pub const BUILDER: &str = "inf.builder";

    /// Present iff at least one FTS column: byte offset of the FTS blob.
    pub const FTS_OFFSET: &str = "inf.fts.offset";

    /// Present iff at least one FTS column: byte length of the FTS blob.
    pub const FTS_LENGTH: &str = "inf.fts.length";

    /// Present iff at least one FTS column: per-column FTS config JSON.
    pub const FTS_COLUMNS: &str = "inf.fts.columns";

    /// Present iff at least one vector index: byte offset of vector blob.
    pub const VEC_OFFSET: &str = "inf.vec.offset";

    /// Present iff at least one vector index: byte length of vector blob.
    pub const VEC_LENGTH: &str = "inf.vec.length";

    /// Present iff at least one vector index: per-index vector config JSON.
    pub const VEC_COLUMNS: &str = "inf.vec.columns";

    /// Present iff vector blob uses a non-default layout (`ivf` default).
    pub const VEC_LAYOUT: &str = "inf.vec.layout";

    /// Optional: JSON array of global cell ids packed into a multi-cell
    /// vector blob, in cell-directory order. Present when
    /// `inf.vec.layout = multi_cell_ivf`.
    pub const VEC_CELLS: &str = "inf.vec.cells";

    /// Optional: byte offset of the raw stable-id sidecar — a packed
    /// little-endian `i128` array, one entry per local doc id (Parquet row
    /// order), mirroring the `_id` column. Lets a hit → `_id` resolve read a
    /// fixed-width slice instead of decompressing the Parquet id pages.
    /// Absent on superfiles written before the sidecar existed; readers fall
    /// back to the Parquet id column when it is missing.
    pub const IDS_OFFSET: &str = "inf.ids.offset";

    /// Present iff the stable-id sidecar is present: its byte length
    /// (`16 * n_docs` for the raw layout; whatever the packed layout
    /// came to otherwise).
    pub const IDS_LENGTH: &str = "inf.ids.length";
    /// Optional, alongside the sidecar keys: names the sidecar's layout.
    /// Absent means the raw `i128` array; [`IDS_LAYOUT_PACKED`] means the
    /// frame-of-reference blocks of `superfile::ids`, which a reader
    /// decodes per doc at the same fixed cost for a fraction of the bytes.
    pub const IDS_LAYOUT: &str = "inf.ids.layout";
    /// The [`IDS_LAYOUT`] value for the packed sidecar.
    pub const IDS_LAYOUT_PACKED: &str = "packed";

    /// Sentinel value for the `inf.format` key.
    pub const FORMAT_VALUE: &str = "infino-superfile";

    /// All required-on-every-superfile keys (used for open-time validation).
    pub const REQUIRED: &[&str] = &[FORMAT, FORMAT_VERSION, ID_COLUMN, N_DOCS, BUILDER];

    /// All FTS-related keys (presence is all-or-none).
    pub const FTS_KEYS: &[&str] = &[FTS_OFFSET, FTS_LENGTH, FTS_COLUMNS];

    /// All vector-related keys (presence is all-or-none).
    pub const VEC_KEYS: &[&str] = &[VEC_OFFSET, VEC_LENGTH, VEC_COLUMNS];

    /// Stable-id sidecar keys (presence is all-or-none).
    pub const IDS_KEYS: &[&str] = &[IDS_OFFSET, IDS_LENGTH];

    /// All known keys (for diagnostics only).
    pub const ALL: &[&str] = &[
        FORMAT,
        FORMAT_VERSION,
        ID_COLUMN,
        N_DOCS,
        BUILDER,
        FTS_OFFSET,
        FTS_LENGTH,
        FTS_COLUMNS,
        VEC_OFFSET,
        VEC_LENGTH,
        VEC_COLUMNS,
        VEC_LAYOUT,
        VEC_CELLS,
        IDS_OFFSET,
        IDS_LENGTH,
    ];
}

/// Width of one stable-id sidecar entry (`inf.ids.*`): a `Decimal128` id
/// packed as a little-endian `i128`. The sidecar is `n_docs` of these, in
/// local doc id order.
pub(crate) const ID_SIDECAR_ENTRY_BYTES: usize = size_of::<i128>();

/// Reserved column-name prefix; user FTS column / vector index names must not
/// start with this string. Defensive — keeps the user's namespace and our
/// internal namespace separate even if we add more KV keys later.
pub const RESERVED_PREFIX: &str = "inf.";

/// Little-endian `u32` at `at`, `None` past the end of `bytes`.
#[inline]
pub(crate) fn u32_le_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
}

/// Little-endian `u64` at `at`, `None` past the end of `bytes`.
#[inline]
pub(crate) fn u64_le_at(bytes: &[u8], at: usize) -> Option<u64> {
    bytes
        .get(at..at + 8)
        .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
}

/// Reserved separator byte inside FST keys (`<column>\x1F<term>`). User
/// column names must not contain this byte. ASCII Unit Separator (U+001F)
/// is below every printable ASCII char so prefix iteration over a column's
/// terms works correctly via FST range scan.
pub const FST_SEPARATOR: u8 = 0x1F;

/// Parsed (major, minor, patch) representation of a semver string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    /// Parse a strict `MAJOR.MINOR.PATCH` semver string. No pre-release or
    /// build metadata accepted (we control this string ourselves; the
    /// strictness is the point).
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }
        Some(Version {
            major: parts[0].parse().ok()?,
            minor: parts[1].parse().ok()?,
            patch: parts[2].parse().ok()?,
        })
    }

    /// Reader policy: accept this superfile if its format-version's major
    /// matches our `FORMAT_VERSION`'s major. Minor/patch differences are
    /// forward-compatible by design (unknown KV keys ignored, unknown JSON
    /// fields ignored).
    pub fn is_compatible_with_current(&self) -> bool {
        let current =
            Version::parse(FORMAT_VERSION).expect("FORMAT_VERSION is a valid semver constant");
        self.major == current.major
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn blob_layout_table_follows_the_version_ladder() {
        use fts::{BlobLayout, BlockLayout, DictLayout, SkipLayout};
        let legacy = BlobLayout::for_version(fts::VERSION_V1_LEGACY).expect("v1");
        assert!(
            !legacy.coarse
                && !legacy.short_form
                && !legacy.grouped_positions
                && !legacy.position_subindex
                && !legacy.bitset_blocks
        );
        assert_eq!(legacy.block, BlockLayout::Wide);
        assert_eq!(legacy.skip, SkipLayout::Absolute);
        assert_eq!(legacy.dict, DictLayout::Fst);
        assert_eq!(legacy.doc_length_bytes, fts::U32_BYTES);
        assert_eq!(BlobLayout::for_version(fts::VERSION_V2), Some(legacy));
        let v3 = BlobLayout::for_version(fts::VERSION_V3).expect("v3");
        assert!(v3.position_subindex && !v3.bitset_blocks && !v3.coarse);
        let v4 = BlobLayout::for_version(fts::VERSION_V4).expect("v4");
        assert!(v4.position_subindex && v4.bitset_blocks && !v4.coarse);
        let v5 = BlobLayout::for_version(fts::VERSION_V5).expect("v5");
        assert!(v5.coarse && v5.position_subindex && v5.bitset_blocks && !v5.short_form);
        assert_eq!(BlobLayout::for_version(fts::VERSION_V6), Some(v5));
        assert_eq!(v5.block, BlockLayout::Wide);
        let v7 = BlobLayout::for_version(fts::VERSION_V7).expect("v7");
        assert!(v7.coarse && v7.short_form && v7.grouped_positions && v7.bitset_blocks);
        assert!(!v7.position_subindex, "grouped positions need no sub-index");
        assert_eq!(v7.block, BlockLayout::Compact);
        assert_eq!(v7.skip, SkipLayout::Length);
        assert_eq!(v7.dict, DictLayout::Blocks);
        assert_eq!(v7.doc_length_bytes, fts::DOC_LENGTH_BYTES_V7);
        assert_eq!(BlobLayout::for_version(fts::VERSION_V7 + 1), None);
        assert_eq!(BlobLayout::for_version(0), None);
    }

    #[test]
    fn skip_layouts_size_their_entries_and_slots() {
        use fts::SkipLayout;
        assert_eq!(SkipLayout::Absolute.entry_bytes(true), 16);
        assert_eq!(SkipLayout::Absolute.entry_bytes(false), 16);
        assert_eq!(SkipLayout::Length.entry_bytes(true), 14);
        assert_eq!(SkipLayout::Length.entry_bytes(false), 10);
        assert_eq!(SkipLayout::Absolute.coarse_slot_bytes(), 4);
        assert_eq!(SkipLayout::Length.coarse_slot_bytes(), 8);
        // The bound and positions fields sit right after what precedes them.
        assert_eq!(SkipLayout::Absolute.bound_off(), 8);
        assert_eq!(
            SkipLayout::Length.bound_off(),
            fts::skip_entry::BLOCK_LEN_OFF + 2
        );
        assert_eq!(SkipLayout::Absolute.positions_off(), 12);
        assert_eq!(
            SkipLayout::Length.positions_off(),
            SkipLayout::Length.bound_off() + 4
        );
    }

    #[test]
    fn little_endian_word_readers_stop_at_the_end() {
        let bytes = [1u8, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0];
        assert_eq!(u32_le_at(&bytes, 0), Some(1));
        assert_eq!(u32_le_at(&bytes, 4), Some(2));
        assert_eq!(u32_le_at(&bytes, 8), Some(3));
        assert_eq!(u32_le_at(&bytes, 9), None);
        assert_eq!(u64_le_at(&bytes, 0), Some(1 | (2 << 32)));
        assert_eq!(u64_le_at(&bytes, 4), Some(2 | (3 << 32)));
        assert_eq!(u64_le_at(&bytes, 5), None);
        assert_eq!(u32_le_at(&[], 0), None);
    }

    #[test]
    fn project_magic_is_three_bytes() {
        assert_eq!(PROJECT_MAGIC, b"INF");
        assert_eq!(PROJECT_MAGIC.len(), 3);
    }

    #[test]
    fn fts_magic_starts_with_project_magic() {
        assert_eq!(&fts::MAGIC[0..3], PROJECT_MAGIC);
        assert_eq!(fts::MAGIC, b"INFFTS01");
        assert_eq!(fts::MAGIC.len(), 8);
    }

    #[test]
    fn vec_outer_magic_starts_with_project_magic() {
        assert_eq!(&vec::OUTER_MAGIC[0..3], PROJECT_MAGIC);
        assert_eq!(vec::OUTER_MAGIC, b"INFVEC01");
    }

    #[test]
    fn vec_sub_magic_starts_with_project_magic() {
        assert_eq!(&vec::SUB_MAGIC[0..3], PROJECT_MAGIC);
        assert_eq!(vec::SUB_MAGIC, b"INFVECC1");
    }

    #[test]
    fn three_magics_are_distinct() {
        let m: HashSet<&[u8]> = [
            fts::MAGIC.as_slice(),
            vec::OUTER_MAGIC.as_slice(),
            vec::SUB_MAGIC.as_slice(),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            m.len(),
            3,
            "FTS / vec-outer / vec-sub magics must be distinct"
        );
    }

    #[test]
    fn all_kv_keys_have_inf_prefix() {
        for k in kv::ALL {
            assert!(
                k.starts_with(RESERVED_PREFIX),
                "KV key {k:?} should start with {RESERVED_PREFIX:?}"
            );
        }
    }

    #[test]
    fn all_kv_keys_are_unique() {
        let set: HashSet<&&str> = kv::ALL.iter().collect();
        assert_eq!(set.len(), kv::ALL.len(), "duplicate KV key in kv::ALL");
    }

    #[test]
    fn required_kv_keys_present_in_all() {
        for k in kv::REQUIRED {
            assert!(
                kv::ALL.contains(k),
                "required key {k:?} missing from kv::ALL"
            );
        }
    }

    #[test]
    fn fts_and_vec_key_groups_present_in_all() {
        for k in kv::FTS_KEYS {
            assert!(kv::ALL.contains(k));
        }
        for k in kv::VEC_KEYS {
            assert!(kv::ALL.contains(k));
        }
    }

    #[test]
    fn version_parses_strict_semver() {
        assert_eq!(
            Version::parse("1.0.0"),
            Some(Version {
                major: 1,
                minor: 0,
                patch: 0
            })
        );
        assert_eq!(
            Version::parse("12.34.567"),
            Some(Version {
                major: 12,
                minor: 34,
                patch: 567
            })
        );
    }

    #[test]
    fn version_rejects_malformed_strings() {
        // wrong number of parts
        assert_eq!(Version::parse(""), None);
        assert_eq!(Version::parse("1"), None);
        assert_eq!(Version::parse("1.0"), None);
        assert_eq!(Version::parse("1.0.0.0"), None);
        // non-numeric components
        assert_eq!(Version::parse("a.b.c"), None);
        assert_eq!(Version::parse("1.0.x"), None);
        // pre-release / build metadata not accepted
        assert_eq!(Version::parse("1.0.0-alpha"), None);
        assert_eq!(Version::parse("1.0.0+sha"), None);
        // negative numbers
        assert_eq!(Version::parse("-1.0.0"), None);
        // whitespace
        assert_eq!(Version::parse(" 1.0.0"), None);
        assert_eq!(Version::parse("1.0.0 "), None);
    }

    #[test]
    fn current_format_version_is_valid_semver() {
        assert!(Version::parse(FORMAT_VERSION).is_some());
    }

    #[test]
    fn version_compat_matches_on_major() {
        let v = Version::parse(FORMAT_VERSION).expect("parse Version");
        assert!(v.is_compatible_with_current());

        let v2 = Version {
            major: v.major,
            minor: v.minor + 99,
            patch: v.patch + 99,
        };
        assert!(
            v2.is_compatible_with_current(),
            "minor/patch drift is compatible"
        );

        let v3 = Version {
            major: v.major + 1,
            minor: 0,
            patch: 0,
        };
        assert!(
            !v3.is_compatible_with_current(),
            "major bump is incompatible"
        );
    }

    #[test]
    fn fst_separator_is_below_printable_ascii() {
        const _: () = assert!(FST_SEPARATOR < b' ');
        assert_eq!(FST_SEPARATOR, 0x1F);
    }

    #[test]
    fn format_value_sentinel_is_the_expected_string() {
        assert_eq!(kv::FORMAT_VALUE, "infino-superfile");
    }
}
