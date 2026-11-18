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

use bytes::Bytes;
use futures::future::try_join_all;

use super::{ColumnMeta, core::*, cursor::TermMeta};
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
        if let DictEntry::Inline { doc_id, tf } = entry {
            return Ok(docs
                .iter()
                .map(|&d| if d == doc_id { col.inline_tf(tf) } else { 0 })
                .collect());
        }
        let blocks = self.blocks_for(col, entry, df, docs, false).await?;
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

    /// The blocks of a term in a `Pfor` dictionary value covering `docs`,
    /// decoded, each with its position group when `with_positions`: a
    /// short-form term's one body, or a long-form term's covering blocks.
    async fn blocks_for(
        &self,
        col: &ColumnMeta,
        entry: DictEntry,
        df: u64,
        docs: &[u32],
        with_positions: bool,
    ) -> Result<Vec<ProbedBlock>, FtsError> {
        let DictEntry::Pfor {
            metadata_offset,
            postings_length,
            short,
        } = entry
        else {
            return Err(malformed("an inline term has no blocks"));
        };
        let (offset, len) = (metadata_offset as usize, postings_length as usize);
        if !short {
            return self
                .covering_blocks(
                    col.positions,
                    col.wide_blocks,
                    offset,
                    len,
                    df,
                    docs,
                    with_positions,
                )
                .await;
        }
        let body = self
            .fetch_term_postings(&[(offset, len)])
            .await?
            .pop()
            .ok_or_else(|| malformed("short body"))?;
        let (mut ids, mut tfs) = (vec![0u32; BLOCK_LEN], vec![0u32; BLOCK_LEN]);
        let decoded = decode_short(&body, col.positions, &mut ids, &mut tfs)
            .ok_or_else(|| malformed("short body"))?;
        let group = match with_positions {
            false => None,
            true => Some(
                body.slice(
                    decoded
                        .positions_at
                        .ok_or_else(|| malformed("short positions"))?..,
                ),
            ),
        };
        Ok(vec![ProbedBlock {
            lo: 0,
            hi: u32::MAX,
            ids: ids[..decoded.n].to_vec(),
            tfs: tfs[..decoded.n].to_vec(),
            group,
        }])
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
        // The blocks lie back to back, as do their groups: a few requests
        // serve them all.
        let mut fetched = self
            .fetch_coalesced(&ranges, "fts/postings term probe blocks")
            .await?;
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
        let per_member = try_join_all(
            members
                .iter()
                .map(|&(entry, df)| self.member_positions(col, entry, df, docs)),
        )
        .await?;
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
        // On a positional column an inline slot is the term's one position.
        if let DictEntry::Inline { doc_id, tf: slot } = entry {
            if let Ok(i) = docs.binary_search(&doc_id) {
                out[i].push(slot);
            }
            return Ok(out);
        }
        let mut group = GroupIndex::default();
        for block in self.blocks_for(col, entry, df, docs, true).await? {
            let bytes = block
                .group
                .as_ref()
                .ok_or_else(|| malformed("position group"))?;
            group
                .locate(bytes, &mut 0, &block.tfs)
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
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use async_trait::async_trait;

    use super::*;
    use crate::superfile::{
        fts::{builder::FtsBuilder, reader::OpenOptions, reader::test_util::AsyncOnlySource},
        lazy_source::{LazyByteSource, LazyByteSourceError},
    };

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
    /// Requests a probe of one long term may take: its header and skip
    /// table, read again when the header names more blocks than its count
    /// suggested, then its covering blocks and their position groups, which
    /// lie back to back in two regions.
    const PROBE_MAX_REQUESTS: usize = 4;

    /// [`AsyncOnlySource`] counting the requests it serves.
    struct CountingSource {
        inner: AsyncOnlySource,
        requests: AtomicUsize,
    }

    #[async_trait]
    impl LazyByteSource for CountingSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.requests.fetch_add(1, Ordering::Relaxed);
            self.inner.range(start, len).await
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

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

    /// A probe of a long term over many blocks reads its covering blocks,
    /// and their position groups, in a few requests rather than one per
    /// block: on an object store each request is a round trip and a billed
    /// operation, and a phrase of common words probes hundreds of
    /// superfiles.
    #[tokio::test]
    async fn a_probe_reads_its_covering_blocks_in_a_few_requests() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register");
        for i in 0..N_DOCS {
            b.add_doc(0, i, &phrase_tokens(i).join(" ")).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let eager = FtsReader::open(blob.clone(), json).expect("open");
        let dict_bytes = eager.dict_bytes_async().await.expect("dict");
        let facts: HashMap<String, (DictEntry, u64)> = eager
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
        let source = Arc::new(CountingSource {
            inner: AsyncOnlySource(blob),
            requests: AtomicUsize::new(0),
        });
        let lazy = FtsReader::open_lazy(
            Arc::clone(&source) as Arc<dyn LazyByteSource>,
            json,
            OpenOptions::for_object_store(),
        )
        .await
        .expect("open_lazy");
        // A stride through every block of a term every document holds.
        let docs: Vec<u32> = (0..N_DOCS).step_by(97).collect();
        let (entry, df) = facts["gamma"];
        let blocks = df.div_ceil(BLOCK_LEN as u64) as usize;
        assert!(
            blocks > PROBE_MAX_REQUESTS * 2,
            "the term spans many blocks"
        );
        let before = source.requests.load(Ordering::Relaxed);
        let tfs = lazy
            .probe_tfs("body", entry, df, &docs)
            .await
            .expect("probe");
        assert!(tfs.iter().all(|&tf| tf > 0), "every document holds it");
        let term = source.requests.load(Ordering::Relaxed) - before;
        assert!(
            term <= PROBE_MAX_REQUESTS,
            "a term probe over {blocks} blocks took {term} requests"
        );
        let members = [facts["beta"], facts["alpha"]];
        let before = source.requests.load(Ordering::Relaxed);
        lazy.probe_phrase_tfs("body", &members, &[0, 1], &docs)
            .await
            .expect("phrase probe");
        let phrase = source.requests.load(Ordering::Relaxed) - before;
        assert!(
            phrase <= members.len() * PROBE_MAX_REQUESTS,
            "a phrase probe of {} members took {phrase} requests",
            members.len()
        );
    }
}
