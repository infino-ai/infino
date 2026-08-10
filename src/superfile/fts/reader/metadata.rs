// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS index metadata + open configuration: the per-doc length-norm
//! table ([`NormTable`]), per-column metadata ([`ColumnMeta`]) and its
//! JSON config ([`FtsColumnConfig`]), and the reader [`OpenOptions`].

use std::{ops::Range, sync::Arc};

use serde::Deserialize;

use crate::superfile::fts::{bm25, tokenize::Tokenizer};

/// Per-doc BM25 length normalizer, quantized to one byte per doc.
///
/// The scorer needs `dl_norm_k1[doc] = K1·(1 - B + B·dl/avgdl)` for
/// every scored doc. Held as an `f32` per doc, that table is 4 bytes ×
/// n_docs — at multi-million-doc scale too large to stay cache-resident,
/// so each scored doc pays a scattered load from a table that overflows
/// cache. Instead the doc length is quantized to one byte
/// ([`bm25::quantize_len`]) and a 256-entry table decodes each bucket to
/// its norm value: the per-doc table is 4× smaller (one byte), and the
/// decode table is 1 KiB (L1-resident). A scored doc reads
/// `lut[bytes[doc]]` — one load from the small per-doc table plus one L1
/// lookup — instead of one load from a 4×-larger table.
#[derive(Debug, Clone)]
pub struct NormTable {
    /// Per-doc quantized length bucket. Empty for a column with no docs.
    bytes: Vec<u8>,
    /// Bucket → `K1·(1 - B + B·dequantize_len(bucket)/avgdl)`. A fixed
    /// 256-entry table, boxed so `ColumnMeta` stays pointer-sized (it is
    /// scanned by non-scoring paths — column lookup, listing) while the
    /// `u8` bucket index into a fixed-length array lets the compiler drop
    /// the bounds check in `get`.
    lut: Box<[f32; 256]>,
}

impl NormTable {
    /// Build from a column's per-doc lengths and average length. An
    /// `avgdl` of `0.0` (empty column) yields an empty table; it is
    /// never indexed because `search` short-circuits on empty columns.
    pub(super) fn new(doc_lengths: impl Iterator<Item = u32>, n_docs: usize, avgdl: f32) -> Self {
        if avgdl <= 0.0 {
            return Self::empty();
        }
        let inv_avgdl = 1.0_f32 / avgdl;
        // Fill the boxed table in place so the 256 f32s land on the heap
        // directly rather than being built on the stack and moved.
        let mut lut = Box::new([0.0_f32; 256]);
        for (b, slot) in lut.iter_mut().enumerate() {
            let dl = bm25::dequantize_len(b as u8) as f32;
            *slot = bm25::K1 * (1.0 - bm25::B + bm25::B * dl * inv_avgdl);
        }
        let mut bytes = Vec::with_capacity(n_docs);
        for dl in doc_lengths {
            bytes.push(bm25::quantize_len(dl));
        }
        Self { bytes, lut }
    }

    /// `dl_norm_k1` for a doc (length quantized): one per-doc byte load
    /// plus one L1 decode-table lookup. Hot path — keep it inlined.
    #[inline(always)]
    pub(super) fn get(&self, doc: u32) -> f32 {
        self.lut[self.bytes[doc as usize] as usize]
    }

    /// Number of docs in the table. Test-only: the query path indexes
    /// by doc id and never needs the count.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.bytes.len()
    }

    /// An empty table: `bytes` is empty, so `get` must never be called on
    /// it. For call sites that need a `&NormTable` but provably never index
    /// it — an unranked (`bar == NEG_INFINITY`) phrase seek, which does no
    /// scoring. The `lut` is a zeroed 256-entry table, allocated but never
    /// read.
    pub(super) fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            lut: Box::new([0.0; 256]),
        }
    }
}

/// Per-column metadata, indexed by column_id (declaration order).
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    /// Byte range into [`FtsReader::blob`] holding this column's
    /// `u32` doc-lengths array (4 bytes per doc, length × n_docs).
    pub doc_lengths_range: Range<usize>,
    /// Average doc length across this column. `0.0` if the column has
    /// no docs.
    pub avgdl: f32,
    /// Per-doc BM25 length normalizer, byte-quantized — see
    /// [`NormTable`]. Computed once per reader at `open` time from the
    /// column's on-disk doc-lengths array. The hot scoring loop reads
    /// `dl_norm_k1.get(d)` and multiplies-out to `idf · tf · (K1+1) /
    /// (tf + dl_norm_k1.get(d))`.
    pub dl_norm_k1: NormTable,
    /// Whether this column's index carries token positions (from
    /// `inf.fts.columns`); phrase queries require it.
    pub positions: bool,
    /// Tokenizer for this column, reconstructed at open time from the
    /// `tokenizer` name in `inf.fts.columns`. Query terms for this
    /// column must be tokenized with it to match how the column was
    /// indexed.
    pub tokenizer: Arc<dyn Tokenizer>,
}

/// JSON-deserialized form of one entry in `inf.fts.columns`. The KV
/// value is a JSON array of these, in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct FtsColumnConfig {
    pub name: String,
    /// The column's analyzer name: `"ascii_lower"` (the default) or
    /// `"standard"`. A missing field deserializes to `"ascii_lower"`
    /// for backward compatibility with files written before the
    /// analyzer name was recorded.
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
    /// Whether this column's index records token positions (phrase
    /// support). Files written before positions existed lack the
    /// field, which can only mean no positions — so a missing field
    /// deserializes to `false`.
    #[serde(default)]
    pub positions: bool,
}

pub(super) fn default_tokenizer() -> String {
    "ascii_lower".to_string()
}

/// Per-open knobs for [`FtsReader::open_with`]. Mirrors the
/// vector reader's `OpenOptions` so the superfile layer can
/// pass a single `verify_crc` flag through to both
/// sub-readers.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the four per-section CRC32C checks (FST,
    /// postings region, doc-lengths directory, per-column
    /// doc-lengths arrays). Defaults to `true`; flip to
    /// `false` only when the underlying storage already
    /// validates checksums (content-addressed object
    /// store, ZFS, etc.) to skip the scan on cold open.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

impl OpenOptions {
    pub fn for_object_store() -> Self {
        Self { verify_crc: false }
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder, reader::FtsReader, tokenize::AsciiLowerTokenizer,
    };

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn open_with_verify_crc_off_succeeds() {
        // The trusted-storage fast path skips the four CRC scans but must
        // still produce a fully usable reader.
        let (blob, json) = build_blob();
        let r = FtsReader::open_with(blob, &json, OpenOptions { verify_crc: false })
            .expect("open with crc off");
        assert_eq!(r.n_docs(), 3);
        assert_eq!(r.fts_columns().collect::<Vec<_>>(), vec!["body"]);
    }

    #[test]
    fn open_with_object_store_options_matches_crc_off() {
        // `for_object_store` is the named constructor for the crc-off
        // OpenOptions the lazy/object-store path uses.
        let opts = OpenOptions::for_object_store();
        assert!(!opts.verify_crc);
        let (blob, json) = build_blob();
        FtsReader::open_with(blob, &json, opts).expect("open object-store options");
    }

    #[test]
    fn default_open_options_verifies_crc() {
        assert!(OpenOptions::default().verify_crc);
    }

    #[test]
    fn default_tokenizer_helper_is_ascii_lower() {
        assert_eq!(default_tokenizer(), "ascii_lower");
    }

    #[test]
    fn fts_column_config_missing_tokenizer_defaults() {
        // A column JSON without the optional `tokenizer` field decodes to
        // the ascii_lower default (round-trips an old file written before
        // the field existed).
        let (blob, _) = build_blob();
        let json = r#"[{"name":"body"}]"#;
        let r = FtsReader::open(blob, json).expect("open with terse json");
        let cfg = r.fts_columns_config().next().expect("one column");
        assert_eq!(cfg.name, "body");
    }

    #[test]
    fn fts_columns_config_exposes_per_column_metadata() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let cols: Vec<&ColumnMeta> = r.fts_columns_config().collect();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].name, "body");
        // Three non-empty docs ⇒ a positive average doc length and a
        // populated per-doc normalization table.
        assert!(cols[0].avgdl > 0.0);
        assert_eq!(cols[0].dl_norm_k1.len(), 3);
    }

    #[test]
    fn norm_table_footprint_is_one_byte_per_doc() {
        // Memory guard: the resident length-norm table must stay at one
        // byte per doc (plus the fixed 256-entry decode LUT), not the
        // 4-byte-per-doc `f32` table it replaced. Build enough
        // varied-length docs that the per-doc term dominates the LUT.
        const N: u32 = 5_000;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        for d in 0..N {
            // Lengths cycle 1..=40 tokens so norms span many buckets and
            // the table isn't a degenerate single value.
            let words = (d % 40) + 1;
            let text: String = (0..words).map(|w| format!("t{}x{w} ", d % 97)).collect();
            b.add_doc(0, d, text.trim()).expect("add doc");
        }
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(bytes), json).expect("open");
        let nt = &r.columns[0].dl_norm_k1;

        let per_doc = nt.bytes.capacity(); // 1 byte/doc
        let lut = std::mem::size_of_val(&*nt.lut); // 256 * 4 = 1 KiB
        let m2_bytes = per_doc + lut;
        let f32_baseline = N as usize * std::mem::size_of::<f32>();

        assert_eq!(nt.bytes.len(), N as usize, "one bucket byte per doc");
        assert_eq!(nt.lut.len(), 256, "fixed 256-entry decode table");
        // The whole point: strictly smaller than the old f32 table, and
        // asymptotically 4× smaller (per-doc term is 1 byte vs 4).
        assert!(
            m2_bytes < f32_baseline,
            "norm table {m2_bytes} B not smaller than f32 baseline {f32_baseline} B"
        );
        assert_eq!(
            per_doc * 4,
            f32_baseline,
            "per-doc term is exactly 4× smaller"
        );
    }
}
