// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! On-disk byte attribution for an FTS blob: which region, which
//! document-frequency band, and which part of a term's layout (fixed
//! per-term and per-block overhead, lane padding, payload, positions)
//! the bytes went to. The tool that shows where an index's size comes
//! from, so a layout change is judged by what it did to the bytes and
//! not by the total alone. Walks the dictionary once; `O(terms)` and
//! meant for a report, never a query path.

use std::fmt;

use super::{
    core::{FtsReader, fetch_source_range, header_postings_length},
    cursor::{SubindexKind, TermMeta},
};
use crate::superfile::{
    ReadError,
    bits::width_of,
    error::FtsError,
    format::{self, FST_SEPARATOR, fts::POSITION_SUBINDEX_ENTRIES_PER_BLOCK},
    fts::{
        builder::{TERM_META_POSITIONAL_SIZE, TERM_META_SIZE},
        fst_value::FstValue,
        posting::{
            BLOCK_LEN, BlockHeader, ENCODING_PACKED, ENCODING_PATCHED, decode_block,
            patched_exception_ranges,
        },
        short::{decode_short, short_df},
    },
};

/// Upper edges of the document-frequency bands a term is filed under.
/// The tail bands are where a per-term fixed cost shows; the head bands
/// are where per-block and payload costs show.
const DF_BAND_EDGES: [u64; 7] = [1, 4, 16, 128, 1024, 16_384, 262_144];
const DF_BAND_LABELS: [&str; 8] = [
    "df=1",
    "df 2-4",
    "df 5-16",
    "df 17-128",
    "df 129-1k",
    "df 1k-16k",
    "df 16k-256k",
    "df >256k",
];
/// Bytes per mebibyte, for the report.
const BYTES_PER_MIB: f64 = 1024.0 * 1024.0;
/// Lanes a packed block always encodes, whether or not they hold docs.
const LANES: u64 = BLOCK_LEN as u64;

/// Byte attribution for one document-frequency band of one column.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DfBucket {
    pub label: &'static str,
    pub terms: u64,
    pub postings: u64,
    /// Term bytes in the dictionary keys (column prefix excluded).
    pub key_bytes: u64,
    /// Terms in the df=1 inline form (no postings bytes at all).
    pub inline_terms: u64,
    /// Terms in the short form and their body bytes.
    pub short_terms: u64,
    pub short_bytes: u64,
    /// Long-form fixed overhead: metadata headers, skip entries, position
    /// sub-index rows, coarse slots, block headers.
    pub meta_bytes: u64,
    pub skip_bytes: u64,
    pub subindex_bytes: u64,
    pub coarse_bytes: u64,
    pub block_header_bytes: u64,
    /// Long-form payload: packed doc-id lanes, packed tf lanes, bitset
    /// blocks (presence words + tfs).
    pub docid_bytes: u64,
    pub tf_bytes: u64,
    pub bitset_bytes: u64,
    /// Of the packed lanes, the bytes spent on lanes with no document
    /// behind them (a partial block padded to `BLOCK_LEN`).
    pub padding_bytes: u64,
    pub long_terms: u64,
    pub blocks: u64,
    pub partial_blocks: u64,
    /// Blocks in the patched encoding (narrow width plus exceptions).
    pub patched_blocks: u64,
    /// Patched blocks by their delta-exception count (`PATCH_HIST_EDGES`
    /// bands) and the bytes each band saves over plain packing.
    pub patched_by_exceptions: [u64; PATCH_HIST_BANDS],
    pub patched_saved_by_exceptions: [u64; PATCH_HIST_BANDS],
    /// Patched blocks by the bytes each saves (`PATCH_SAVING_EDGES`
    /// bands) and the bytes each band saves in all.
    pub patched_by_saving: [u64; PATCH_HIST_BANDS],
    pub patched_saved_by_saving: [u64; PATCH_HIST_BANDS],
    /// Patched blocks that are a term's partial last block, and the
    /// bytes those save — a walk meets at most one per term.
    pub patched_partial_blocks: u64,
    pub patched_partial_saved: u64,
    /// Bytes the band's patched blocks save over plain packing.
    pub patched_saved: u64,
    /// Position-run bytes this band's terms own in the positions region.
    pub positions_bytes: u64,
}

impl DfBucket {
    fn add(&mut self, o: &DfBucket) {
        self.terms += o.terms;
        self.postings += o.postings;
        self.key_bytes += o.key_bytes;
        self.inline_terms += o.inline_terms;
        self.short_terms += o.short_terms;
        self.short_bytes += o.short_bytes;
        self.meta_bytes += o.meta_bytes;
        self.skip_bytes += o.skip_bytes;
        self.subindex_bytes += o.subindex_bytes;
        self.coarse_bytes += o.coarse_bytes;
        self.block_header_bytes += o.block_header_bytes;
        self.docid_bytes += o.docid_bytes;
        self.tf_bytes += o.tf_bytes;
        self.bitset_bytes += o.bitset_bytes;
        self.padding_bytes += o.padding_bytes;
        self.long_terms += o.long_terms;
        self.blocks += o.blocks;
        self.partial_blocks += o.partial_blocks;
        self.patched_blocks += o.patched_blocks;
        for i in 0..PATCH_HIST_BANDS {
            self.patched_by_exceptions[i] += o.patched_by_exceptions[i];
            self.patched_saved_by_exceptions[i] += o.patched_saved_by_exceptions[i];
            self.patched_by_saving[i] += o.patched_by_saving[i];
            self.patched_saved_by_saving[i] += o.patched_saved_by_saving[i];
        }
        self.patched_partial_blocks += o.patched_partial_blocks;
        self.patched_partial_saved += o.patched_partial_saved;
        self.patched_saved += o.patched_saved;
        self.positions_bytes += o.positions_bytes;
    }

    /// Every byte this band's terms occupy in the postings region.
    pub fn postings_region_bytes(&self) -> u64 {
        self.short_bytes
            + self.meta_bytes
            + self.skip_bytes
            + self.subindex_bytes
            + self.coarse_bytes
            + self.block_header_bytes
            + self.docid_bytes
            + self.tf_bytes
            + self.bitset_bytes
    }

    /// Long-form bytes that are neither doc ids, tfs nor a bitset.
    pub fn fixed_overhead_bytes(&self) -> u64 {
        self.meta_bytes
            + self.skip_bytes
            + self.subindex_bytes
            + self.coarse_bytes
            + self.block_header_bytes
    }
}

/// One column's attribution: a bucket per df band plus the total.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnSizeBreakdown {
    pub column: String,
    pub positional: bool,
    pub buckets: Vec<DfBucket>,
    pub total: DfBucket,
    pub doc_lengths_bytes: u64,
}

/// Whole-blob attribution: the region sizes from the header plus a
/// per-column walk of every term.
#[derive(Debug, Clone, PartialEq)]
pub struct FtsSizeBreakdown {
    pub n_docs: u64,
    pub n_terms: u64,
    pub fst_bytes: u64,
    pub postings_region_bytes: u64,
    pub positions_region_bytes: u64,
    pub columns: Vec<ColumnSizeBreakdown>,
}

/// Bands of the patched-block histograms: upper edges (inclusive) of
/// the first four, everything larger in the fifth.
const PATCH_HIST_BANDS: usize = 5;
/// Delta-exception count edges.
const PATCH_HIST_EDGES: [u64; PATCH_HIST_BANDS - 1] = [1, 4, 8, 16];
/// Bytes-saved edges.
const PATCH_SAVING_EDGES: [u64; PATCH_HIST_BANDS - 1] = [16, 32, 64, 128];

fn hist_band(edges: &[u64; PATCH_HIST_BANDS - 1], v: u64) -> usize {
    edges.iter().position(|&e| v <= e).unwrap_or(edges.len())
}

fn band_of(df: u64) -> usize {
    DF_BAND_EDGES
        .iter()
        .position(|&edge| df <= edge)
        .unwrap_or(DF_BAND_EDGES.len())
}

impl FtsReader {
    /// Attribute every byte of the blob's dictionary, postings and
    /// positions regions to a column, a df band and a layout part.
    /// Walks the whole dictionary and fetches every term range; a
    /// report, not a query.
    pub fn size_breakdown(&self) -> Result<FtsSizeBreakdown, FtsError> {
        let fst_bytes = self.dict_bytes()?;
        let dict = self.open_dict(&fst_bytes)?;
        let mut columns = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            let positional = col.positions;
            let subindex = match positional {
                true => self.subindex,
                false => SubindexKind::None,
            };
            let mut prefix = col.name.as_bytes().to_vec();
            prefix.push(FST_SEPARATOR);
            let mut buckets: Vec<DfBucket> = DF_BAND_LABELS
                .iter()
                .map(|&label| DfBucket {
                    label,
                    ..DfBucket::default()
                })
                .collect();
            let mut d = [0u32; BLOCK_LEN];
            let mut t = [0u32; BLOCK_LEN];
            for (key, packed) in dict.iter_prefix(&prefix) {
                let key_bytes = (key.len() - prefix.len()) as u64;
                match packed {
                    FstValue::Inline { .. } => {
                        let b = &mut buckets[band_of(1)];
                        b.terms += 1;
                        b.postings += 1;
                        b.key_bytes += key_bytes;
                        b.inline_terms += 1;
                    }
                    FstValue::Pfor {
                        metadata_offset,
                        postings_length_hint,
                        short,
                    } => {
                        let start = self.postings_range.start + metadata_offset as usize;
                        let len = match postings_length_hint {
                            Some(l) => l as usize,
                            None => header_postings_length(
                                fetch_source_range(
                                    &self.source,
                                    start..start + TERM_META_SIZE,
                                    "fts/size header",
                                )?
                                .as_ref(),
                            )?,
                        };
                        let bytes =
                            fetch_source_range(&self.source, start..start + len, "fts/size term")?;
                        let tb = bytes.as_ref();
                        if short {
                            let df = u64::from(short_df(tb).ok_or_else(|| {
                                FtsError::Read(ReadError::MalformedVersion(
                                    "malformed short-form term body".into(),
                                ))
                            })?);
                            let decoded =
                                decode_short(tb, positional, &mut d, &mut t).ok_or_else(|| {
                                    FtsError::Read(ReadError::MalformedVersion(
                                        "malformed short-form term body".into(),
                                    ))
                                })?;
                            let b = &mut buckets[band_of(df)];
                            b.terms += 1;
                            b.postings += df;
                            b.key_bytes += key_bytes;
                            b.short_terms += 1;
                            b.short_bytes += len as u64;
                            // The group is inline in the body: part of
                            // `short_bytes`, reported under positions too.
                            b.positions_bytes += decoded
                                .positions_at
                                .map(|at| (len - at) as u64)
                                .unwrap_or(0);
                            continue;
                        }
                        let meta = TermMeta::parse(
                            tb,
                            0,
                            positional,
                            subindex,
                            self.bounds,
                            self.positions_grouped,
                        )?;
                        let nb = meta.num_blocks as u64;
                        let b = &mut buckets[band_of(meta.df)];
                        b.terms += 1;
                        b.postings += meta.df;
                        b.key_bytes += key_bytes;
                        b.long_terms += 1;
                        b.blocks += nb;
                        b.meta_bytes += match positional {
                            true => TERM_META_POSITIONAL_SIZE,
                            false => TERM_META_SIZE,
                        } as u64;
                        b.skip_bytes += nb * meta.skip.entry_bytes(positional) as u64;
                        b.subindex_bytes += nb
                            * (POSITION_SUBINDEX_ENTRIES_PER_BLOCK * subindex.entry_bytes()) as u64;
                        if meta.has_coarse {
                            b.coarse_bytes += nb
                                .div_ceil(format::fts::COARSE_BLOCK_MAX_SPAN as u64)
                                * meta.skip.coarse_slot_bytes() as u64;
                        }
                        b.positions_bytes += u64::from(meta.positions_length);
                        let mut prev_end: Option<usize> = None;
                        for i in 0..meta.num_blocks {
                            let range = meta.block_range_in_term(tb, i, prev_end);
                            prev_end = Some(range.end);
                            let block = &tb[range];
                            let hdr = BlockHeader::parse(
                                block,
                                meta.block_layout,
                                meta.prev_last_doc_id(tb, i),
                            );
                            let doc_count = hdr.count() as u64;
                            b.block_header_bytes += hdr.payload() as u64;
                            if doc_count < LANES {
                                b.partial_blocks += 1;
                            }
                            let delta_bits = u64::from(hdr.delta_bits);
                            let tf_bits = u64::from(hdr.tf_bits);
                            match hdr.encoding {
                                ENCODING_PACKED => {
                                    b.docid_bytes += LANES * delta_bits / 8;
                                    b.tf_bytes += LANES * tf_bits / 8;
                                    b.padding_bytes +=
                                        (LANES - doc_count) * (delta_bits + tf_bits) / 8;
                                }
                                ENCODING_PATCHED => {
                                    // Exceptions count with the stream they patch.
                                    let (delta_exc, tf_exc) = patched_exception_ranges(block, &hdr);
                                    b.patched_blocks += 1;
                                    // What plain packing would have cost: the
                                    // widths of the widest delta and tf.
                                    let n = decode_block(block, &hdr, &mut d, &mut t);
                                    let mut widest_delta = d[0].wrapping_sub(hdr.base);
                                    for w in d[..n].windows(2) {
                                        widest_delta = widest_delta.max(w[1] - w[0]);
                                    }
                                    let widest_tf = t[..n].iter().copied().max().unwrap_or(0);
                                    let plain = hdr.payload()
                                        + BLOCK_LEN * width_of(u64::from(widest_delta)) as usize
                                            / 8
                                        + BLOCK_LEN * width_of(u64::from(widest_tf)) as usize / 8;
                                    let saved = plain.saturating_sub(block.len()) as u64;
                                    let e = hist_band(&PATCH_HIST_EDGES, hdr.n_delta_exc() as u64);
                                    b.patched_by_exceptions[e] += 1;
                                    b.patched_saved_by_exceptions[e] += saved;
                                    let g = hist_band(&PATCH_SAVING_EDGES, saved);
                                    b.patched_by_saving[g] += 1;
                                    b.patched_saved_by_saving[g] += saved;
                                    b.patched_saved += saved;
                                    if doc_count < LANES {
                                        b.patched_partial_blocks += 1;
                                        b.patched_partial_saved += saved;
                                    }
                                    b.docid_bytes +=
                                        LANES * delta_bits / 8 + delta_exc.len() as u64;
                                    b.tf_bytes += LANES * tf_bits / 8 + tf_exc.len() as u64;
                                    b.padding_bytes +=
                                        (LANES - doc_count) * (delta_bits + tf_bits) / 8;
                                }
                                _ => {
                                    b.bitset_bytes +=
                                        (block.len() - hdr.payload()) as u64 - LANES * tf_bits / 8;
                                    b.tf_bytes += LANES * tf_bits / 8;
                                }
                            }
                        }
                    }
                }
            }
            let mut total = DfBucket {
                label: "total",
                ..DfBucket::default()
            };
            for b in &buckets {
                total.add(b);
            }
            columns.push(ColumnSizeBreakdown {
                column: col.name.clone(),
                positional,
                buckets,
                total,
                doc_lengths_bytes: col.doc_lengths_range.len() as u64,
            });
        }
        Ok(FtsSizeBreakdown {
            n_docs: u64::from(self.n_docs),
            n_terms: u64::from(self.n_terms_total),
            fst_bytes: self.fst_range.len() as u64,
            postings_region_bytes: self.postings_range.len() as u64,
            positions_region_bytes: self
                .positions_range
                .as_ref()
                .map(|r| r.len() as u64)
                .unwrap_or(0),
            columns,
        })
    }
}

fn mib(b: u64) -> f64 {
    b as f64 / BYTES_PER_MIB
}

impl fmt::Display for FtsSizeBreakdown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "fts blob: {} docs, {} terms", self.n_docs, self.n_terms)?;
        writeln!(
            f,
            "  fst        {:>12} B  {:>9.2} MiB",
            self.fst_bytes,
            mib(self.fst_bytes)
        )?;
        writeln!(
            f,
            "  postings   {:>12} B  {:>9.2} MiB",
            self.postings_region_bytes,
            mib(self.postings_region_bytes)
        )?;
        writeln!(
            f,
            "  positions  {:>12} B  {:>9.2} MiB",
            self.positions_region_bytes,
            mib(self.positions_region_bytes)
        )?;
        for c in &self.columns {
            writeln!(
                f,
                "column `{}` (positional={}, doc_lengths {:.2} MiB)",
                c.column,
                c.positional,
                mib(c.doc_lengths_bytes)
            )?;
            writeln!(
                f,
                "  {:<12} {:>9} {:>11} {:>7} {:>7} | {:>8} {:>8} | {:>7} {:>7} {:>7} {:>7} {:>7} | {:>8} {:>8} {:>8} {:>7} {:>8} | {:>8} | {:>8}",
                "band",
                "terms",
                "postings",
                "inline",
                "short",
                "shortMiB",
                "keyMiB",
                "metaMiB",
                "skipMiB",
                "subMiB",
                "crsMiB",
                "hdrMiB",
                "docidMiB",
                "tfMiB",
                "bitsMiB",
                "padMiB",
                "patchMiB",
                "regnMiB",
                "posMiB"
            )?;
            for b in c.buckets.iter().chain(std::iter::once(&c.total)) {
                if b.terms == 0 {
                    continue;
                }
                writeln!(
                    f,
                    "  {:<12} {:>9} {:>11} {:>7} {:>7} | {:>8.2} {:>8.2} | {:>7.2} {:>7.2} {:>7.2} {:>7.2} {:>7.2} | {:>8.2} {:>8.2} {:>8.2} {:>7.2} {:>8.2} | {:>8.2} | {:>8.2}",
                    b.label,
                    b.terms,
                    b.postings,
                    b.inline_terms,
                    b.short_terms,
                    mib(b.short_bytes),
                    mib(b.key_bytes),
                    mib(b.meta_bytes),
                    mib(b.skip_bytes),
                    mib(b.subindex_bytes),
                    mib(b.coarse_bytes),
                    mib(b.block_header_bytes),
                    mib(b.docid_bytes),
                    mib(b.tf_bytes),
                    mib(b.bitset_bytes),
                    mib(b.padding_bytes),
                    mib(b.patched_saved),
                    mib(b.postings_region_bytes()),
                    mib(b.positions_bytes),
                )?;
            }
            writeln!(
                f,
                "  long-form fixed overhead {:.2} MiB, lane padding {:.2} MiB, {} blocks ({} partial, {} patched)",
                mib(c.total.fixed_overhead_bytes()),
                mib(c.total.padding_bytes),
                c.total.blocks,
                c.total.partial_blocks,
                c.total.patched_blocks
            )?;
            let bands = |edges: &[u64; PATCH_HIST_BANDS - 1], counts: &[u64], saved: &[u64]| {
                let mut lo = 0u64;
                let mut parts = Vec::new();
                for i in 0..PATCH_HIST_BANDS {
                    let label = match edges.get(i) {
                        Some(&hi) => format!("{lo}-{hi}"),
                        None => format!(">{lo}"),
                    };
                    parts.push(format!(
                        "[{label}] {} blocks {:.2} MiB",
                        counts[i],
                        mib(saved[i])
                    ));
                    lo = edges.get(i).map(|&e| e + 1).unwrap_or(lo);
                }
                parts.join("; ")
            };
            writeln!(
                f,
                "  patched blocks by delta exceptions (count, bytes saved): {}",
                bands(
                    &PATCH_HIST_EDGES,
                    &c.total.patched_by_exceptions,
                    &c.total.patched_saved_by_exceptions
                )
            )?;
            writeln!(
                f,
                "  patched partial (last) blocks: {} saving {:.2} MiB; patched full blocks: {} saving {:.2} MiB",
                c.total.patched_partial_blocks,
                mib(c.total.patched_partial_saved),
                c.total.patched_blocks - c.total.patched_partial_blocks,
                mib(c.total.patched_saved_by_saving.iter().sum::<u64>()
                    - c.total.patched_partial_saved),
            )?;
            writeln!(
                f,
                "  patched blocks by bytes saved (count, bytes saved): {}",
                bands(
                    &PATCH_SAVING_EDGES,
                    &c.total.patched_by_saving,
                    &c.total.patched_saved_by_saving
                )
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::{builder::FtsBuilder, tokenize::AsciiLowerTokenizer};

    #[test]
    fn patched_form_is_granted_only_under_the_posting_cap() {
        // `common` is in 18,000 of 24,000 docs and `rare` in 400, both with
        // a 1,000-doc hole every 4,000 docs: the blocks straddling a hole
        // carry one large delta among small ones, exactly the shape the
        // patched form is for. `rare` sits under the cap and takes it;
        // `common` is over it and stays plain, the histograms tally.
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("text".into(), false).expect("register");
        for i in 0..24_000u32 {
            let in_hole = i % 4_000 >= 3_000;
            let mut text = String::new();
            match in_hole {
                true => text.push_str("hole"),
                false => {
                    text.push_str("common");
                    if i % 60 == 0 {
                        text.push_str(" rare");
                    }
                }
            }
            b.add_doc(0, i, &text).expect("doc");
        }
        let json = r#"[{"name":"text","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let s = r.size_breakdown().expect("breakdown");
        let c = &s.columns[0];
        let band = |label: &str| c.buckets.iter().find(|b| b.label == label).expect("band");
        let rare = band("df 129-1k");
        assert_eq!(rare.terms, 1);
        assert!(
            rare.patched_blocks >= 1,
            "rare term's gappy blocks are patched"
        );
        let common = band("df 16k-256k");
        assert_eq!(common.terms, 1);
        assert!(common.blocks > 100);
        assert_eq!(common.patched_blocks, 0, "over the cap: no patched block");
        // `hole` (6,000 docs, under the cap) straddles the common stretches
        // the same way; between them they are every patched block.
        let mid = band("df 1k-16k");
        assert_eq!(mid.terms, 1);
        let total = &c.total;
        assert_eq!(
            total.patched_blocks,
            rare.patched_blocks + mid.patched_blocks
        );
        assert_eq!(
            total.patched_by_exceptions.iter().sum::<u64>(),
            total.patched_blocks
        );
        assert_eq!(
            total.patched_by_saving.iter().sum::<u64>(),
            total.patched_blocks
        );
        assert_eq!(
            total.patched_saved_by_exceptions.iter().sum::<u64>(),
            total.patched_saved
        );
        assert_eq!(
            total.patched_saved_by_saving.iter().sum::<u64>(),
            total.patched_saved
        );
        assert!(total.patched_saved > 0);
        assert!(total.patched_partial_blocks <= total.patched_blocks);
        assert!(total.patched_partial_saved <= total.patched_saved);
        assert!(s.to_string().contains("patched partial (last) blocks"));
    }

    #[test]
    fn every_postings_byte_is_attributed_and_forms_are_told_apart() {
        let mut b = FtsBuilder::new(Arc::new(AsciiLowerTokenizer));
        b.register_column("body".into(), true).expect("register");
        // `only` inlines (df=1, tf=1); `pair` is short (df=2); `every`
        // spans two blocks (long form).
        for i in 0..(BLOCK_LEN as u32 + 5) {
            let text = match i {
                0 => "only pair every",
                1 => "pair every",
                _ => "every every",
            };
            b.add_doc(0, i, text).expect("doc");
        }
        let json = r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let s = r.size_breakdown().expect("breakdown");
        assert_eq!(s.columns.len(), 1);
        let c = &s.columns[0];
        assert_eq!(c.total.terms, 3);
        assert_eq!(c.total.inline_terms, 1);
        assert_eq!(c.total.short_terms, 1);
        assert_eq!(c.total.long_terms, 1);
        assert_eq!(c.total.blocks, 2);
        assert_eq!(c.total.partial_blocks, 1);
        // The whole postings region (the reader's range already excludes
        // the trailing CRC) is accounted for.
        assert_eq!(c.total.postings_region_bytes(), s.postings_region_bytes);
        assert!(s.to_string().contains("df 2-4"));
    }
}
