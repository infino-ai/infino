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
    /// 8-byte magic at the start of the FTS blob: `INF` + `FTS` +
    /// `01`. The trailing `01` is a fixed part of the section
    /// identity, **not** a version — it never changes across blob
    /// versions. The blob's version is the `u32` at
    /// [`hdr::VERSION_OFF`], and only that field.
    pub const MAGIC: &[u8; 8] = b"INFFTS01";

    /// The baseline blob version.
    ///
    /// Regions, in order: the [`HEADER_SIZE`]-byte header ([`hdr`]), the
    /// term dictionary, the postings region, the positions region (empty
    /// unless a column records positions) and the doc-lengths directory
    /// followed by each column's length array. Every region ends with a
    /// CRC-32C.
    ///
    /// The term dictionary is sorted, front-coded term blocks behind a
    /// first-key index (`utils::terms`). Each entry says which of three
    /// forms the term's postings take:
    ///
    /// - **inline**: a df=1 posting that fits the dictionary value, with
    ///   no bytes in the postings region;
    /// - **short** (`fts::short`): a term whose posting list fits one
    ///   block — a varint `df`, a tf-equals-one bitmap, the doc-id deltas
    ///   as group-varint, the remaining tfs as varints and, on a
    ///   positional column, the term's position group;
    /// - **long**: a metadata header ([`term_meta`]), a skip table
    ///   ([`skip_entry`]), the posting blocks and a coarse block-max
    ///   table ([`coarse_slot`]) at the tail.
    ///
    /// A posting block's header is one 4-byte word (`posting`); a packed
    /// or patched block's base doc id is the previous block's last doc id,
    /// which the skip table already holds, and a dense block may take the
    /// presence-bitset encoding. A skip entry carries the block's byte
    /// length and its max BM25 as exact `f32` bits — the scorer's own
    /// score for the block's best document — so a random block is one
    /// coarse slot plus at most `COARSE_BLOCK_MAX_SPAN - 1` lengths away.
    /// Positions are per-block groups, decoded whole and indexed by the
    /// block's tf prefix sums.
    ///
    /// Each column's doc-lengths directory entry declares the average
    /// document length the file is scored at: the table-wide average over
    /// the documents that carry tokens, folding in every superfile
    /// committed before it, so a query needs no other value and the
    /// stored bounds are exact. Lengths are stored as `u16`, saturating
    /// at [`DOC_LENGTH_STORED_MAX`].
    pub const VERSION_V7: u32 = 7;

    /// The version written when a superfile's documents are stored in
    /// the FTS blob under an ordering of their own. Byte for byte the
    /// [`VERSION_V7`] layout for every term, block, skip entry and
    /// dictionary value; what it adds is one region, the **doc-id map**.
    /// Its header is the [`VERSION_V7`] header unchanged: the map is the
    /// last region before the doc-lengths directory and its size follows
    /// from the document count, so where it begins is arithmetic and
    /// needs no field of its own.
    ///
    /// In a `V7` blob an FTS doc id *is* a Parquet row index, so postings
    /// are ordered by arrival. `V8` separates the two: postings are
    /// ordered by whatever grouping the writer chose, and the map gives
    /// the row a doc id belongs to, one `u32` per document followed by a
    /// CRC. Compaction uses this to place documents that share terms
    /// next to each other, which shortens posting deltas and narrows the
    /// id span a block covers, without moving a single Parquet row: row
    /// groups keep the statistics arrival order gave them, the vector
    /// blob keeps its own ordering, and the table stays time ordered.
    ///
    /// The map is the whole of the difference. The search kernels never
    /// see it, because they work in the blob's own id space throughout
    /// and the ids are translated once, on the way out. The region sits
    /// between the positions region and the doc-lengths directory, and
    /// the open-time tail fetch starts at the map rather than at the
    /// directory, so the two arrive in one range read instead of two.
    ///
    /// One thing a caller can observe changes, and it is not a bug: a
    /// top-k breaks equal scores by the blob's own doc id, so a `V8`
    /// blob and a `V7` blob of the same documents can name different
    /// rows among a group that scores identically. Both are correct
    /// rankings and every returned row carries the score its rank
    /// claims; which of several tied documents is chosen is not part of
    /// the ordering the scorer defines, and reordering is precisely a
    /// change to the id that breaks the tie.
    pub const VERSION_V8: u32 = 8;

    /// The oldest blob version this crate reads. A blob below it was
    /// written before [`VERSION_MIN_RELEASE`] and is refused on open with
    /// an error that says to reindex it with [`REPAIR_RELEASE`].
    pub const VERSION_MIN: u32 = VERSION_V7;

    /// The first infino release that writes [`VERSION_MIN`].
    pub const VERSION_MIN_RELEASE: &str = "0.8.4";

    /// The infino release whose `reindex` rewrites an older blob to
    /// [`VERSION_MIN`] and reports a table under a removed analyzer.
    pub const REPAIR_RELEASE: &str = "0.9.1";

    /// The blob version a file must carry to be current — what the
    /// staleness check compares against and what a migration plans from.
    ///
    /// Not the newest version: [`VERSION_V8`] sits above it and is written
    /// only where a compaction chooses an order, so deriving this from the
    /// highest known version would mark every plain file stale and have a
    /// reindex re-emit the same version forever.
    pub const VERSION_CURRENT: u32 = VERSION_V7;

    /// Bytes per stored document length: a `u16`, saturating at
    /// [`DOC_LENGTH_STORED_MAX`]. The scorer reads a one-byte bucket of
    /// the length and the directory carries the exact average; a document
    /// past 65,535 tokens has its stored length (and so its stored bucket
    /// and bound) computed from the saturated value, consistently on the
    /// writer and the reader.
    pub const DOC_LENGTH_BYTES: usize = 2;
    /// Largest per-document length a blob stores.
    pub const DOC_LENGTH_STORED_MAX: u32 = u16::MAX as u32;

    /// Fixed-point scale for the per-column average document length.
    /// The builder stores `round(avgdl × 1000)` in the doc-lengths
    /// directory as a `u32` (`avgdl_x1000`); the reader recovers the
    /// `f32` average length by dividing by this. Defined once so the
    /// write and read paths share one scale.
    pub const AVGDL_FIXED_POINT_SCALE: f32 = 1000.0;

    /// Offset of `avgdl_x1000` within a doc-lengths directory entry
    /// (`[0..4]` column id, `[4..12]` array offset, `[12..16]` this).
    pub const DOC_LENGTHS_ENTRY_AVGDL_OFF: usize = 12;

    /// Number of consecutive posting blocks summarised by one entry of
    /// a term's coarse block-max table. The table sits at the tail of a
    /// long-form term's postings region: `ceil(num_blocks / this)` slots,
    /// each the max of its span's per-block max BM25.
    ///
    /// It gives the ranked single-term walk a second, coarser skip level:
    /// when the running k-th-best score already dominates a whole span's
    /// upper bound, the walk jumps the span in one comparison instead of
    /// touching each block's skip entry. On a very long, heavily-skipped
    /// posting list (a common term at small k) the per-block skip scan is
    /// itself the dominant cost; the coarse level removes ~31/32 of it.
    /// The span bound is the max of exact block bounds, so it stays a
    /// true upper bound and the top-k is unchanged.
    pub const COARSE_BLOCK_MAX_SPAN: usize = 32;

    /// FTS blob header size in bytes: the fields in [`hdr`]. The term
    /// dictionary begins immediately after it.
    pub const HEADER_SIZE: usize = 56;

    /// Width of the 8-byte FTS magic field.
    pub const MAGIC_BYTES: usize = 8;
    /// Width of a little-endian `u32` header field.
    pub const U32_BYTES: usize = 4;
    /// Width of a little-endian `u64` header field.
    pub const U64_BYTES: usize = 8;

    /// FTS blob header field offsets ([`HEADER_SIZE`] bytes):
    ///
    /// ```text
    /// [ 0.. 8] MAGIC
    /// [ 8..12] version (u32 LE)
    /// [12..16] n_columns (u32 LE)
    /// [16..20] n_docs (u32 LE)
    /// [20..24] n_terms_total (u32 LE)
    /// [24..32] dictionary offset (u64 LE)
    /// [32..40] postings_offset (u64 LE)
    /// [40..48] doc_lengths_table_offset (u64 LE)
    /// [48..56] positions_offset (u64 LE)
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
        /// `[24..32]` term dictionary offset (`u64` LE).
        pub const DICT_OFFSET_OFF: usize = 24;
        /// `[32..40]` postings region offset (`u64` LE).
        pub const POSTINGS_OFFSET_OFF: usize = 32;
        /// `[40..48]` doc-lengths directory offset (`u64` LE).
        pub const DOC_LENGTHS_DIR_OFF: usize = 40;
        /// `[48..56]` positions region offset (`u64` LE). The region sits
        /// between the postings region and the doc-lengths directory so
        /// the lazy-open doc-lengths tail fetch stays small.
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

    /// Skip-table entry field offsets (relative to the entry start):
    ///
    /// ```text
    /// [ 0.. 4] last_doc_id (u32 LE)
    /// [ 4.. 6] block_len (u16 LE, the block's encoded bytes)
    /// [ 6..10] block-max bound (u32 LE, f32 bits)
    /// [10..14] positions_block_offset (u32 LE; positional columns only)
    /// ```
    ///
    /// A block's offset is its coarse slot's span start plus the lengths
    /// of the span's earlier blocks. The positions field records the
    /// byte offset of this block's position group, relative to the
    /// term's `positions_offset` — per-block random access into the
    /// term's position bytes, aligned with the doc blocks.
    pub mod skip_entry {
        /// `[0..4]` largest doc-id in the block (`u32` LE).
        pub const LAST_DOC_ID_OFF: usize = 0;
        /// `[4..6]` byte length of the encoded block (`u16` LE).
        pub const BLOCK_LEN_OFF: usize = 4;
        /// `[6..10]` the block-max bound (`u32` LE, `f32` bits).
        pub const BOUND_OFF: usize = 6;
        /// `[10..14]` the block's position-group offset (`u32` LE),
        /// positional columns only.
        pub const POSITIONS_OFF: usize = 10;
        /// Bytes one entry takes on a positionless column.
        pub const BYTES: usize = 10;
        /// Bytes one entry takes on a positional column.
        pub const POSITIONAL_BYTES: usize = 14;

        /// Bytes one entry takes on a column with or without positions.
        pub fn bytes(positional: bool) -> usize {
            match positional {
                true => POSITIONAL_BYTES,
                false => BYTES,
            }
        }
    }

    /// Coarse slot field offsets (relative to the slot start): the
    /// span's block-max bound and the byte offset (relative to term
    /// metadata start) of the span's first block.
    pub mod coarse_slot {
        /// `[0..4]` span bound (`u32` LE, `f32` bits).
        pub const BOUND_OFF: usize = 0;
        /// `[4..8]` span start offset (`u32` LE).
        pub const SPAN_START_OFF: usize = 4;
        /// Bytes one slot takes.
        pub const BYTES: usize = 8;
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
    /// Namespace every key in this module shares. A carried footer's
    /// keys are dropped by this prefix, so a key added here is covered
    /// without touching that path.
    pub const PREFIX: &str = "inf.";

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

    /// The table schema version the superfile was written under
    /// (string-encoded u32). Provenance only: a reader decides whether a
    /// file matches the table by comparing physical schemas, never by
    /// this number. Absent on superfiles written before it existed.
    pub const SCHEMA_ID: &str = "inf.schema_id";

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

    /// Keys locating a blob in the file. Only the splice that lays the
    /// file out writes them, since only it knows where each blob landed.
    pub const REGION_KEYS: [&str; 6] = [
        FTS_OFFSET, FTS_LENGTH, VEC_OFFSET, VEC_LENGTH, IDS_OFFSET, IDS_LENGTH,
    ];

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

pub(crate) use crate::utils::bytes::{u32_le_at, u64_le_at};
/// The key separator is owned by the term dictionary (`utils::terms`) and
/// re-exported here because the format layer validates column names
/// against it.
pub use crate::utils::terms::KEY_SEPARATOR;

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
    fn skip_entry_fields_tile_their_entries() {
        use fts::{coarse_slot, skip_entry};
        assert_eq!(
            skip_entry::LAST_DOC_ID_OFF + fts::U32_BYTES,
            skip_entry::BLOCK_LEN_OFF
        );
        assert_eq!(
            skip_entry::BLOCK_LEN_OFF + size_of::<u16>(),
            skip_entry::BOUND_OFF
        );
        assert_eq!(skip_entry::BOUND_OFF + fts::U32_BYTES, skip_entry::BYTES);
        assert_eq!(skip_entry::POSITIONS_OFF, skip_entry::BYTES);
        assert_eq!(
            skip_entry::POSITIONS_OFF + fts::U32_BYTES,
            skip_entry::POSITIONAL_BYTES
        );
        assert_eq!(skip_entry::bytes(false), skip_entry::BYTES);
        assert_eq!(skip_entry::bytes(true), skip_entry::POSITIONAL_BYTES);
        assert_eq!(
            coarse_slot::SPAN_START_OFF + fts::U32_BYTES,
            coarse_slot::BYTES
        );
    }

    #[test]
    fn the_current_version_is_readable_and_below_the_newest() {
        const {
            assert!(fts::VERSION_MIN <= fts::VERSION_CURRENT);
            assert!(fts::VERSION_CURRENT <= fts::VERSION_V8);
        }
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
    fn key_separator_is_below_printable_ascii() {
        const _: () = assert!(KEY_SEPARATOR < b' ');
        assert_eq!(KEY_SEPARATOR, 0x1F);
    }

    #[test]
    fn format_value_sentinel_is_the_expected_string() {
        assert_eq!(kv::FORMAT_VALUE, "infino-superfile");
    }
}
