// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reading one term's frequency in a few documents of a blob without
//! reading its whole posting list: a table-level ranked query that has
//! bounded every other document away probes the few it still needs.
//!
//! A long term's bytes are its header, its skip table, its blocks and a
//! coarse table at the tail. A probe reads the header and skip table in
//! one range — sized from the term's document count, read again when the
//! header names more blocks — then only the blocks whose doc-id range
//! covers the documents asked for, and decodes those. A short term is one
//! small body read whole; an inline one is in its dictionary value.

use std::slice::from_ref;

use super::{core::*, cursor::TermMeta};
use crate::{
    superfile::{
        ReadError,
        error::FtsError,
        fts::{
            posting::{BLOCK_LEN, BlockHeader, block_lanes, decode_block},
            short::decode_short,
        },
    },
    utils::terms::DictEntry,
};

fn malformed(what: &str) -> FtsError {
    FtsError::Read(ReadError::Malformed(format!("term probe: {what}")))
}

impl FtsReader {
    /// The frequency of the term `entry` names in `column` — its dictionary
    /// value in this blob — in each of `docs`, blob document ids in
    /// ascending order; zero where the document does not hold it. `df` is
    /// the term's document count here, which sizes the first read.
    pub(crate) async fn probe_tfs(
        &self,
        column: &str,
        entry: DictEntry,
        df: u64,
        docs: &[u32],
    ) -> Result<Vec<u32>, FtsError> {
        debug_assert!(docs.windows(2).all(|w| w[0] < w[1]), "docs ascend");
        let column_id = self.resolve_column_id(column)?;
        let col = &self.columns[column_id as usize];
        match entry {
            DictEntry::Inline { doc_id, tf } => Ok(docs
                .iter()
                .map(|&d| if d == doc_id { col.inline_tf(tf) } else { 0 })
                .collect()),
            DictEntry::Pfor {
                metadata_offset,
                postings_length,
                short: true,
            } => {
                let body = self
                    .fetch_term_postings(&[(metadata_offset as usize, postings_length as usize)])
                    .await?
                    .pop()
                    .ok_or_else(|| malformed("short body"))?;
                let (mut ids, mut tfs) = (vec![0u32; BLOCK_LEN], vec![0u32; BLOCK_LEN]);
                let n = decode_short(&body, col.positions, &mut ids, &mut tfs)
                    .ok_or_else(|| malformed("short body"))?
                    .n;
                Ok(lookup(&ids[..n], &tfs[..n], docs))
            }
            DictEntry::Pfor {
                metadata_offset,
                postings_length,
                short: false,
            } => {
                self.probe_long(
                    col.positions,
                    col.wide_blocks,
                    metadata_offset as usize,
                    postings_length as usize,
                    df,
                    docs,
                )
                .await
            }
        }
    }

    /// [`Self::probe_tfs`] for a long-form term at `metadata_offset` in the
    /// postings region, `postings_length` bytes long.
    async fn probe_long(
        &self,
        positional: bool,
        wide: bool,
        metadata_offset: usize,
        postings_length: usize,
        df: u64,
        docs: &[u32],
    ) -> Result<Vec<u32>, FtsError> {
        let base = self.postings_range.start + metadata_offset;
        if metadata_offset + postings_length > self.postings_range.len() {
            return Err(malformed("term range past the postings region"));
        }
        let read = |len: usize| {
            let len = len.min(postings_length);
            let range = base..base + len;
            async move {
                self.source
                    .get_ranges_parallel_async(from_ref(&range))
                    .await
                    .map_err(|e| FtsError::RangeFetch {
                        what: "fts/postings term probe head",
                        source: e,
                    })?
                    .pop()
                    .ok_or_else(|| malformed("head read"))
            }
        };
        let guess = (df as usize).div_ceil(block_lanes(wide));
        let mut head = read(TermMeta::head_len(guess, positional)).await?;
        // The header may name more blocks than the count suggested: read
        // the whole table it names then.
        let named = TermMeta::header_num_blocks(&head).ok_or_else(|| malformed("header"))?;
        if TermMeta::head_len(named, positional) > head.len() {
            head = read(TermMeta::head_len(named, positional)).await?;
        }
        let (meta, _) = TermMeta::parse_head(&head, 0, positional)?;
        // Each document's block: the first whose last doc reaches it.
        let last_docs: Vec<u32> = (0..meta.num_blocks)
            .map(|i| meta.skip_entry(&head, i).0)
            .collect();
        let mut wanted: Vec<usize> = docs
            .iter()
            .filter_map(|&d| {
                let i = last_docs.partition_point(|&last| last < d);
                (i < meta.num_blocks).then_some(i)
            })
            .collect();
        wanted.dedup();
        // Blocks follow the skip table back to back.
        let skip_end = TermMeta::head_len(meta.num_blocks, positional);
        let mut starts = Vec::with_capacity(meta.num_blocks + 1);
        let mut at = skip_end;
        for i in 0..meta.num_blocks {
            starts.push(at);
            at += meta.block_len(&head, i);
        }
        starts.push(at);
        if at > postings_length {
            return Err(malformed("blocks past the term range"));
        }
        let ranges: Vec<_> = wanted
            .iter()
            .map(|&i| base + starts[i]..base + starts[i + 1])
            .collect();
        let blocks = self
            .source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: "fts/postings term probe blocks",
                source: e,
            })?;
        let mut out = vec![0u32; docs.len()];
        let (mut ids, mut tfs) = (vec![0u32; BLOCK_LEN], vec![0u32; BLOCK_LEN]);
        for (&i, bytes) in wanted.iter().zip(&blocks) {
            let prev = meta.prev_last_doc_id(&head, i);
            let hdr = BlockHeader::parse(bytes, prev, wide);
            let n = decode_block(bytes, &hdr, &mut ids, &mut tfs);
            let lo = match i {
                0 => 0,
                _ => last_docs[i - 1] + 1,
            };
            let hi = last_docs[i];
            for (slot, &d) in out.iter_mut().zip(docs) {
                if d >= lo && d <= hi {
                    *slot = lookup(&ids[..n], &tfs[..n], &[d])[0];
                }
            }
        }
        Ok(out)
    }
}

/// Each of `docs`' frequency among the ascending `ids` and their `tfs`;
/// zero where absent.
fn lookup(ids: &[u32], tfs: &[u32], docs: &[u32]) -> Vec<u32> {
    docs.iter()
        .map(|d| ids.binary_search(d).map_or(0, |i| tfs[i]))
        .collect()
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::superfile::fts::builder::FtsBuilder;

    /// Documents in the fixture blob: enough for a common term to span many
    /// posting blocks.
    const N_DOCS: u32 = 3_000;
    /// Filler occurrences cycle through this many counts, zero included.
    const FILLER_CYCLE: u32 = 11;
    /// Every this-many-th document holds `mid`.
    const MID_EVERY: u32 = 3;
    /// The documents holding `rare`.
    const RARE: [u32; 2] = [70, 1900];
    /// The document holding `hapax`.
    const HAPAX: u32 = 1234;

    fn text(i: u32) -> String {
        let mut text = String::from("common");
        for _ in 0..(i % FILLER_CYCLE) {
            text.push_str(" filler");
        }
        if i.is_multiple_of(MID_EVERY) {
            text.push_str(" mid");
        }
        if RARE.contains(&i) {
            text.push_str(" rare");
        }
        if i == HAPAX {
            text.push_str(" hapax");
        }
        text
    }

    /// The frequency of `term` in document `i`, by construction.
    fn truth(term: &str, i: u32) -> u32 {
        match term {
            "common" => 1,
            "filler" => i % FILLER_CYCLE,
            "mid" => u32::from(i.is_multiple_of(MID_EVERY)),
            "rare" => u32::from(RARE.contains(&i)),
            "hapax" => u32::from(i == HAPAX),
            _ => 0,
        }
    }

    /// Probed frequencies equal the documents' own, for every form a term's
    /// postings take — inline, short and long over many blocks — on a
    /// column with positions and one without, for documents that hold the
    /// term and documents that do not.
    #[tokio::test]
    async fn a_probe_reads_each_documents_own_frequency() {
        for positions in [false, true] {
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), positions)
                .expect("register");
            for i in 0..N_DOCS {
                b.add_doc(0, i, &text(i)).expect("add");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = format!(
                r#"[{{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":{positions}}}]"#
            );
            let r = FtsReader::open(blob, &json).expect("open");
            let dict_bytes = r.dict_bytes_async().await.expect("dict");
            let facts = r
                .term_index_facts_after(&dict_bytes, "body", None, 64, 1)
                .await
                .expect("facts");
            assert_eq!(facts.len(), 5);
            for (term, fact) in facts {
                let term = String::from_utf8(term).expect("utf8");
                let fact = fact.expect("a fact");
                // Every document, a stride through them, the few holders
                // and a single document past them all.
                let sets: Vec<Vec<u32>> = vec![
                    (0..N_DOCS).collect(),
                    (0..N_DOCS).step_by(97).collect(),
                    vec![RARE[0], HAPAX, RARE[1]],
                    vec![N_DOCS - 1],
                ];
                for docs in sets {
                    let got = r
                        .probe_tfs("body", fact.entry, fact.df, &docs)
                        .await
                        .expect("probe");
                    let want: Vec<u32> = docs.iter().map(|&d| truth(&term, d)).collect();
                    assert_eq!(got, want, "{term:?} positions={positions}");
                }
            }
        }
    }
}
