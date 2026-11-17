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

use super::{ColumnMeta, core::*, cursor::TermMeta};
use bytes::Bytes;

use crate::{
    superfile::{
        ReadError,
        error::FtsError,
        fts::{
            positions::GroupIndex,
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
        let blocks = self
            .covering_blocks(
                positional,
                wide,
                metadata_offset,
                postings_length,
                df,
                docs,
                false,
            )
            .await?;
        Ok(docs
            .iter()
            .map(|&d| {
                blocks
                    .iter()
                    .find(|b| b.covers(d))
                    .map_or(0, |b| lookup(&b.ids, &b.tfs, &[d])[0])
            })
            .collect())
    }

    /// The blocks of a long-form term covering `docs`, decoded, each with
    /// its position group when `with_positions`: the term's header and
    /// skip table in one read — sized from `df`, read again when the header
    /// names more blocks — then the covering blocks and groups together.
    #[allow(clippy::too_many_arguments)]
    async fn covering_blocks(
        &self,
        positional: bool,
        wide: bool,
        metadata_offset: usize,
        postings_length: usize,
        df: u64,
        docs: &[u32],
        with_positions: bool,
    ) -> Result<Vec<ProbedBlock>, FtsError> {
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
        let mut ranges: Vec<_> = wanted
            .iter()
            .map(|&i| base + starts[i]..base + starts[i + 1])
            .collect();
        if with_positions {
            // Each block's group runs from its offset to the next block's,
            // or to the end of the term's positions.
            let region = self.positions_range.start + meta.positions_offset as usize;
            let group_end = |i: usize| match i + 1 < meta.num_blocks {
                true => meta.positions_block_offset(&head, i + 1) as usize,
                false => meta.positions_length as usize,
            };
            for &i in &wanted {
                let from = region + meta.positions_block_offset(&head, i) as usize;
                let to = region + group_end(i);
                if from > to || to > self.positions_range.end {
                    return Err(malformed("position group past the positions region"));
                }
                ranges.push(from..to);
            }
        }
        let mut fetched = self
            .source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: "fts/postings term probe blocks",
                source: e,
            })?;
        let groups: Vec<Bytes> = match with_positions {
            true => fetched.split_off(wanted.len()),
            false => Vec::new(),
        };
        let (mut ids, mut tfs) = (vec![0u32; BLOCK_LEN], vec![0u32; BLOCK_LEN]);
        let mut out = Vec::with_capacity(wanted.len());
        for (k, (&i, bytes)) in wanted.iter().zip(&fetched).enumerate() {
            let prev = meta.prev_last_doc_id(&head, i);
            let hdr = BlockHeader::parse(bytes, prev, wide);
            let n = decode_block(bytes, &hdr, &mut ids, &mut tfs);
            out.push(ProbedBlock {
                lo: prev.map_or(0, |p| p + 1),
                hi: last_docs[i],
                ids: ids[..n].to_vec(),
                tfs: tfs[..n].to_vec(),
                group: groups.get(k).cloned(),
            });
        }
        Ok(out)
    }

    /// The number of in-order occurrences of a phrase in each of `docs`,
    /// blob document ids in ascending order, on a positional `column`: its
    /// members' dictionary values and document counts here, and each
    /// member's offset from the first. Each member's covering blocks and
    /// their position groups are read; a document lacking a member holds
    /// no occurrence.
    pub(crate) async fn probe_phrase_tfs(
        &self,
        column: &str,
        members: &[(DictEntry, u64)],
        offsets: &[u32],
        docs: &[u32],
    ) -> Result<Vec<u32>, FtsError> {
        debug_assert_eq!(members.len(), offsets.len());
        let column_id = self.resolve_column_id(column)?;
        let col = &self.columns[column_id as usize];
        if !col.positions {
            return Err(FtsError::PositionsUnavailable {
                column: column.to_owned(),
            });
        }
        let mut per_member: Vec<Vec<Vec<u32>>> = Vec::with_capacity(members.len());
        for &(entry, df) in members {
            per_member.push(self.member_positions(col, entry, df, docs).await?);
        }
        Ok((0..docs.len())
            .map(|d| {
                let runs: Vec<&[u32]> = per_member.iter().map(|m| m[d].as_slice()).collect();
                phrase_occurrences(&runs, offsets)
            })
            .collect())
    }

    /// One term's positions in each of `docs`; empty where absent.
    async fn member_positions(
        &self,
        col: &ColumnMeta,
        entry: DictEntry,
        df: u64,
        docs: &[u32],
    ) -> Result<Vec<Vec<u32>>, FtsError> {
        let mut out = vec![Vec::new(); docs.len()];
        let mut group = GroupIndex::default();
        match entry {
            // On a positional column the slot is the term's one position.
            DictEntry::Inline { doc_id, tf: slot } => {
                if let Ok(i) = docs.binary_search(&doc_id) {
                    out[i].push(slot);
                }
            }
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
                let decoded = decode_short(&body, true, &mut ids, &mut tfs)
                    .ok_or_else(|| malformed("short body"))?;
                let mut at = decoded
                    .positions_at
                    .ok_or_else(|| malformed("short positions"))?;
                group
                    .locate(&body, &mut at, &tfs[..decoded.n])
                    .ok_or_else(|| malformed("short position group"))?;
                for (slot, d) in out.iter_mut().zip(docs) {
                    if let Ok(p) = ids[..decoded.n].binary_search(d) {
                        group
                            .run_positions(&body, p, tfs[p], slot)
                            .ok_or_else(|| malformed("short position run"))?;
                    }
                }
            }
            DictEntry::Pfor {
                metadata_offset,
                postings_length,
                short: false,
            } => {
                let blocks = self
                    .covering_blocks(
                        true,
                        col.wide_blocks,
                        metadata_offset as usize,
                        postings_length as usize,
                        df,
                        docs,
                        true,
                    )
                    .await?;
                for block in &blocks {
                    let bytes = block.group.as_ref().ok_or_else(|| malformed("group"))?;
                    let mut at = 0usize;
                    group
                        .locate(bytes, &mut at, &block.tfs)
                        .ok_or_else(|| malformed("position group"))?;
                    for (slot, &d) in out.iter_mut().zip(docs) {
                        if !block.covers(d) {
                            continue;
                        }
                        if let Ok(p) = block.ids.binary_search(&d) {
                            group
                                .run_positions(bytes, p, block.tfs[p], slot)
                                .ok_or_else(|| malformed("position run"))?;
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}

/// One decoded posting block a probe read: the doc-id range it covers,
/// its documents and their frequencies, and its position group when read.
struct ProbedBlock {
    lo: u32,
    hi: u32,
    ids: Vec<u32>,
    tfs: Vec<u32>,
    group: Option<Bytes>,
}

impl ProbedBlock {
    fn covers(&self, doc: u32) -> bool {
        doc >= self.lo && doc <= self.hi
    }
}

/// The occurrences of a phrase in one document: the first member's
/// positions from which every member sits at its offset. `runs` are the
/// members' ascending positions there.
fn phrase_occurrences(runs: &[&[u32]], offsets: &[u32]) -> u32 {
    let Some(first) = runs.first() else {
        return 0;
    };
    first
        .iter()
        .filter(|&&p| {
            runs.iter().zip(offsets).skip(1).all(|(run, &off)| {
                p.checked_add(off)
                    .is_some_and(|want| run.binary_search(&want).is_ok())
            })
        })
        .count() as u32
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
    use std::{
        collections::{BTreeSet, HashMap},
        iter::repeat_n,
    };

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

    /// The tokens of phrase-fixture document `i`: "alpha beta" repeated a
    /// varying number of times, "beta alpha" once, a stretch of filler, and
    /// in two documents "rare beta", in one "hapax gamma".
    fn phrase_tokens(i: u32) -> Vec<&'static str> {
        let mut tokens = Vec::new();
        for _ in 0..(i % 4) {
            tokens.extend(["alpha", "beta"]);
        }
        tokens.extend(["beta", "alpha"]);
        tokens.extend(repeat_n("filler", (i % 5) as usize));
        if RARE.contains(&i) {
            tokens.extend(["rare", "beta"]);
        }
        if i == HAPAX {
            tokens.extend(["hapax", "gamma"]);
        }
        tokens.push("gamma");
        tokens
    }

    /// Probed phrase occurrences equal a brute-force count over the
    /// documents' tokens, for phrases whose members are long, short and
    /// inline, adjacent and with a gap.
    #[tokio::test]
    async fn a_phrase_probe_counts_each_documents_own_occurrences() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register");
        for i in 0..N_DOCS {
            b.add_doc(0, i, &phrase_tokens(i).join(" ")).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let dict_bytes = r.dict_bytes_async().await.expect("dict");
        let facts: HashMap<String, (DictEntry, u64)> = r
            .term_index_facts_after(&dict_bytes, "body", None, 64, 1)
            .await
            .expect("facts")
            .into_iter()
            .map(|(term, fact)| {
                let fact = fact.expect("a fact");
                (
                    String::from_utf8(term).expect("utf8"),
                    (fact.entry, fact.df),
                )
            })
            .collect();
        let phrases: [(&[&str], &[u32]); 5] = [
            (&["alpha", "beta"], &[0, 1]),
            (&["beta", "alpha"], &[0, 1]),
            (&["rare", "beta"], &[0, 1]),
            (&["hapax", "gamma"], &[0, 1]),
            (&["beta", "beta"], &[0, 2]),
        ];
        let docs: Vec<u32> = (0..N_DOCS)
            .step_by(7)
            .chain([RARE[1], HAPAX])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        for (terms, offsets) in phrases {
            let members: Vec<(DictEntry, u64)> = terms.iter().map(|t| facts[*t]).collect();
            let got = r
                .probe_phrase_tfs("body", &members, offsets, &docs)
                .await
                .expect("probe");
            let want: Vec<u32> = docs
                .iter()
                .map(|&d| {
                    let tokens = phrase_tokens(d);
                    (0..tokens.len())
                        .filter(|&p| {
                            terms
                                .iter()
                                .zip(offsets)
                                .all(|(t, &off)| tokens.get(p + off as usize) == Some(t))
                        })
                        .count() as u32
                })
                .collect();
            assert_eq!(got, want, "{terms:?}");
        }
    }
}
