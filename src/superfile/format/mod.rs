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
pub const FORMAT_VERSION: &str = "1.0.0";

/// CRC width in bytes (`u32` CRC-32C, little-endian) appended after a
/// directory or after a subsection's payload. Defined once so the
/// writer and reader arithmetic agree symbolically rather than via
/// duplicated `+ 4 /* CRC */` literals.
pub const CRC_BYTES: usize = 4;

/// FTS section magic bytes and constants.
pub mod fts {
    /// 8-byte magic at the start of the FTS blob: `INF` + `FTS` + version `01`.
    pub const MAGIC: &[u8; 8] = b"INFFTS01";
    /// Numeric version emitted in the blob header (redundant with magic
    /// suffix; future-proofing for a per-section version separate from
    /// section identity).
    pub const VERSION: u32 = 1;

    /// Fixed-point scale for the per-column average document length.
    /// The builder stores `round(avgdl × 1000)` in the doc-lengths
    /// directory as a `u32` (`avgdl_x1000`); the reader recovers the
    /// `f32` average length by dividing by this. Defined once so the
    /// write and read paths share one scale.
    pub const AVGDL_FIXED_POINT_SCALE: f32 = 1000.0;

    /// Fixed-point scale for a posting block's max-BM25 upper bound.
    /// The builder stores `round(max_bm25 × 1000)` in each skip-table
    /// entry (`max_bm25_x1000`); the reader recovers the `f32` bound
    /// by dividing by this. Drives WAND / block-max skip decisions, so
    /// write and read must agree on the scale.
    pub const BLOCK_MAX_BM25_FIXED_POINT_SCALE: f32 = 1000.0;

    /// Total FTS blob header size in bytes. The FST directory begins
    /// immediately after this fixed-size header.
    pub const HEADER_SIZE: usize = 48;

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
    pub mod term_meta {
        /// `[0..4]` document frequency (`u32` LE).
        pub const DF_OFF: usize = 0;
        /// `[12..16]` total byte length of the term's postings (`u32` LE).
        pub const POSTINGS_LENGTH_OFF: usize = 12;
        /// `[16..20]` number of PFOR blocks / skip-table entries (`u32` LE).
        pub const NUM_BLOCKS_OFF: usize = 16;
    }

    /// Skip-table entry field offsets (relative to the entry start;
    /// each entry is `SKIP_ENTRY_SIZE` bytes):
    ///
    /// ```text
    /// [ 0.. 4] last_doc_id (u32 LE)
    /// [ 4.. 8] block_offset (u32 LE, relative to term metadata start)
    /// [ 8..12] max_bm25_x1000 (u32 LE)
    /// [12..16] reserved (u32)
    /// ```
    pub mod skip_entry {
        /// `[0..4]` largest doc-id in the block (`u32` LE).
        pub const LAST_DOC_ID_OFF: usize = 0;
        /// `[4..8]` byte offset to the encoded PFOR block (`u32` LE).
        pub const BLOCK_OFFSET_OFF: usize = 4;
        /// `[8..12]` fixed-point block-max BM25 bound (`u32` LE).
        pub const MAX_BM25_OFF: usize = 8;
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
    /// Outer-blob version. Written at bytes [8..12] of the outer
    /// header. Bump on outer-blob-shape changes (currently 1).
    pub const VERSION: u32 = 1;

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
    /// [24..28] summary_radius_x100 (u32 LE)
    /// [28..32] reserved (u32)
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

    /// Outer-header size: magic + version + n_columns + n_docs +
    /// dir_offset.
    pub const OUTER_HEADER_SIZE: usize = 32;
    /// Per-column subsection-directory entry size in bytes.
    pub const DIR_ENTRY_SIZE: usize = 64;
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
        /// `[12..16]` column count (`u32` LE).
        pub const N_COLUMNS_OFF: usize = 12;
        /// `[16..24]` document count (`u64` LE).
        pub const N_DOCS_OFF: usize = 16;
        /// `[24..32]` directory byte offset (`u64` LE).
        pub const DIR_OFFSET_OFF: usize = 24;
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
        /// `[24..28]` summary radius ×100 (`u32` LE).
        pub const SUMMARY_RADIUS_X100_OFF: usize = 24;
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

    /// Present iff at least one vector column: byte offset of vector blob.
    pub const VEC_OFFSET: &str = "inf.vec.offset";

    /// Present iff at least one vector column: byte length of vector blob.
    pub const VEC_LENGTH: &str = "inf.vec.length";

    /// Present iff at least one vector column: per-column vector config JSON.
    pub const VEC_COLUMNS: &str = "inf.vec.columns";

    /// Sentinel value for the `inf.format` key.
    pub const FORMAT_VALUE: &str = "infino-superfile";

    /// All required-on-every-superfile keys (used for open-time validation).
    pub const REQUIRED: &[&str] = &[FORMAT, FORMAT_VERSION, ID_COLUMN, N_DOCS, BUILDER];

    /// All FTS-related keys (presence is all-or-none).
    pub const FTS_KEYS: &[&str] = &[FTS_OFFSET, FTS_LENGTH, FTS_COLUMNS];

    /// All vector-related keys (presence is all-or-none).
    pub const VEC_KEYS: &[&str] = &[VEC_OFFSET, VEC_LENGTH, VEC_COLUMNS];

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
    ];
}

/// Reserved column-name prefix; user FTS / vector column names must not
/// start with this string. Defensive — keeps the user's namespace and our
/// internal namespace separate even if we add more KV keys later.
pub const RESERVED_PREFIX: &str = "inf.";

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
