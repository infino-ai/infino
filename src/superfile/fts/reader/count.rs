// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unranked FTS match/count kernels on [`FtsReader`]: the heterogeneous
//! atom walk plus the token/phrase match-id, match-count, and term-df
//! entry points (no BM25 scoring, no top-k). Its own `impl FtsReader`
//! block, split from the reader `core`.

use super::{
    core::*,
    filter::AtomExcludeFilter,
    options::BoolMode,
    phrase::AnyCursor,
    work::{MatchWork, atom_cursor_bytes, atom_planned_ranges},
};
use crate::{
    runtime_metrics::op_stats::timed_section,
    superfile::{
        ReadError,
        error::FtsError,
        format::fts::U32_BYTES,
        fts::{
            builder::TERM_META_SIZE,
            dict::{DictReader, make_key},
            fst_value::FstValue,
        },
    },
};

impl FtsReader {
    /// Unranked doc-at-a-time walk over heterogeneous atoms, calling
    /// `on_doc` for every matching doc in ascending order. `And` walks
    /// the atoms' intersection (a phrase atom's own verification is
    /// part of its cursor); `Or` walks their union. The shared spine
    /// of the phrase-aware `token_match` / `count` entries.
    pub(super) fn walk_atoms_match(
        &self,
        mut atoms: Vec<AnyCursor>,
        mode: BoolMode,
        mut filter: Option<AtomExcludeFilter>,
        mut on_doc: impl FnMut(u32),
    ) -> Result<(), FtsError> {
        match mode {
            BoolMode::Or => {
                while let Some(doc) = atoms
                    .iter()
                    .filter(|a| !a.is_exhausted())
                    .map(AnyCursor::current_doc_id)
                    .min()
                {
                    let admitted = match filter.as_mut() {
                        Some(f) => f.admits(doc)?,
                        None => true,
                    };
                    if admitted {
                        on_doc(doc);
                    }
                    let Some(next) = doc.checked_add(1) else {
                        break;
                    };
                    for a in atoms.iter_mut() {
                        if !a.is_exhausted() && a.current_doc_id() == doc {
                            a.skip_to(next)?;
                        }
                    }
                }
                Ok(())
            }
            BoolMode::And => {
                let mut target = 0u32;
                'docs: loop {
                    // Phase 1 — align every atom by its cheap *approximation*: a
                    // term atom is exact; a phrase atom advances only to a doc
                    // holding all its members, without decoding positions. So a
                    // rare co-clause (e.g. a term AND'd with a common-word
                    // phrase) prunes the candidate set here, before any phrase
                    // pays for adjacency.
                    let mut aligned = target;
                    let mut i = 0usize;
                    while i < atoms.len() {
                        let a = &mut atoms[i];
                        a.approx_skip_to(aligned);
                        if a.is_exhausted() {
                            break 'docs;
                        }
                        let here = a.approx_current_doc();
                        if here > aligned {
                            aligned = here;
                            i = 0;
                            continue;
                        }
                        i += 1;
                    }
                    // Phase 2 — every approximation agrees on `aligned`. Verify
                    // the phrase atoms' positions here only: the intersection is
                    // already down to the co-occurring docs, so position decode
                    // runs on that handful, not on a phrase's whole match set.
                    let mut matched = true;
                    for a in atoms.iter_mut() {
                        if !a.verify_at(aligned)? {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        let admitted = match filter.as_mut() {
                            Some(f) => f.admits(aligned)?,
                            None => true,
                        };
                        if admitted {
                            on_doc(aligned);
                        }
                    }
                    let Some(next) = aligned.checked_add(1) else {
                        break;
                    };
                    target = next;
                }
                Ok(())
            }
        }
    }

    /// Phrase-aware unranked match: the `local_doc_id`s matching the
    /// terms + phrases under `mode`, ascending — the atoms sibling of
    /// [`Self::token_match`], used whenever the match set contains a
    /// phrase. Under `And`, a missing atom empties the set.
    pub(crate) async fn atoms_match_ids(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Vec<String>],
        mode: BoolMode,
    ) -> Result<(Vec<u32>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        // Unranked: idf is irrelevant to the match set, so build local.
        let (built, dict_ranges) = self
            .build_atom_cursors(column_id, terms, phrases, None)
            .await?;
        let missing_and_atom = mode == BoolMode::And && built.iter().any(Option::is_none);
        let atoms: Vec<AnyCursor> = built.into_iter().flatten().collect();
        // The atoms that DID build cost their bytes even when a missing
        // AND atom empties the result — mirrors `prepare_clauses`.
        let mut work = MatchWork::for_atoms(&atoms);
        work.planned_ranges += dict_ranges;
        if missing_and_atom || atoms.is_empty() {
            return Ok((Vec::new(), work));
        }
        let (walk, walk_ns) = timed_section(|| {
            let mut out = Vec::new();
            self.walk_atoms_match(atoms, mode, None, |d| out.push(d))
                .map(|()| out)
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((walk?, work))
    }

    /// Phrase-aware unranked match **count** — the atoms sibling of
    /// [`Self::token_match_count`].
    pub(crate) async fn atoms_match_count(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Vec<String>],
        mode: BoolMode,
        neg_terms: &[&str],
        neg_phrases: &[Vec<String>],
    ) -> Result<(u64, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        // Unranked: idf is irrelevant to the match set, so build local.
        let (built, dict_ranges) = self
            .build_atom_cursors(column_id, terms, phrases, None)
            .await?;
        let missing_and_atom = mode == BoolMode::And && built.iter().any(Option::is_none);
        let atoms: Vec<AnyCursor> = built.into_iter().flatten().collect();
        let mut work = MatchWork::for_atoms(&atoms);
        work.planned_ranges += dict_ranges;
        if missing_and_atom || atoms.is_empty() {
            return Ok((0, work));
        }
        // Negated clauses become a skip-based exclusion gate, never a
        // materialized set: each surviving positive doc is `skip_to`-probed
        // against the negated cursors, so a common negated term's long list
        // is only partially decoded. Empty ⇒ `None`, the same walk as an
        // unnegated count.
        let mut filter = None;
        if !neg_terms.is_empty() || !neg_phrases.is_empty() {
            let (neg_built, neg_dict_ranges) = self
                .build_atom_cursors(column_id, neg_terms, neg_phrases, None)
                .await?;
            let neg_atoms: Vec<AnyCursor> = neg_built.into_iter().flatten().collect();
            // Count the negated clause's posting work the same way the
            // positive atoms above (and the scored path's `ExcludeFilter`)
            // are counted — planned posting bytes + ranges from cursor
            // metadata — so op_stats prices a negated count consistently.
            // (Like every skip/leapfrog path, this is a planned figure, not
            // the partial bytes the skip probe actually decodes.)
            work.postings_bytes += atom_cursor_bytes(&neg_atoms);
            work.planned_ranges += atom_planned_ranges(&neg_atoms) + neg_dict_ranges;
            if !neg_atoms.is_empty() {
                filter = Some(AtomExcludeFilter::new(neg_atoms));
            }
        }
        let (walk, walk_ns) = timed_section(|| {
            let mut n = 0u64;
            self.walk_atoms_match(atoms, mode, filter, |_| n += 1)
                .map(|()| n)
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((walk?, work))
    }

    /// Resolve a column name to its dense column_id, or
    /// `FtsError::UnknownColumn` if the column isn't FTS-indexed in
    /// this superfile. Shared by every public search entry point.
    pub(super) fn resolve_column_id(&self, column: &str) -> Result<u32, FtsError> {
        self.column_id_by_name
            .get(column)
            .copied()
            .ok_or_else(|| FtsError::UnknownColumn(column.to_string()))
    }

    /// Unranked token match over a **token list** — the no-scoring
    /// sibling of [`Self::search`]. `mode = And` returns the
    /// `local_doc_id`s present in *every* token's posting list
    /// (intersection); `mode = Or` returns those in *any* (union), in
    /// ascending doc-id order.
    ///
    /// Reuses the same [`build_term_cursors`](Self::build_term_cursors)
    /// the scored path uses, then walks the cursors —
    /// [`collect_and_intersect`](Self::collect_and_intersect) for `And`,
    /// [`or_merge_unranked`] for `Or` — with no BM25 scoring and no
    /// top-k heap, so nothing is ranked. Cursors traverse blocks in
    /// doc-id order, so the result is already ascending (no re-sort).
    pub async fn token_match(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
    ) -> Result<(Vec<u32>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((Vec::new(), MatchWork::default()));
        }
        let cursors = self
            .build_term_cursors(column_id, tokens, None, true)
            .await?;
        // Tallied before the mode branch: the cursors that DID build cost
        // their bytes even when a missing AND token empties the result.
        // +1: the build's dictionary fetch.
        let mut work = MatchWork::for_cursors(&cursors);
        work.planned_ranges += 1;
        let (docs, walk_ns) = timed_section(|| match mode {
            BoolMode::And => {
                // AND needs every token present; a missing token ⇒ empty
                // set. Otherwise intersect via the same optimized
                // block flat-merge the ranked scorer uses.
                if cursors.len() != tokens.len() {
                    return Vec::new();
                }
                self.collect_and_intersect(column_id, cursors)
            }
            BoolMode::Or => or_merge_unranked(cursors),
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((docs, work))
    }

    /// Unranked token-match **count** — the cardinality
    /// [`token_match`](Self::token_match) would return, without
    /// materializing the doc-id `Vec`. The AND path tallies through a
    /// [`CountSink`], the OR path counts the union walk; both skip the
    /// `Vec<u32>` so a high-cardinality count doesn't allocate one id
    /// per match.
    pub async fn token_match_count(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
    ) -> Result<(u64, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((0, MatchWork::default()));
        }
        let cursors = self
            .build_term_cursors(column_id, tokens, None, true)
            .await?;
        let mut work = MatchWork::for_cursors(&cursors);
        work.planned_ranges += 1;
        let (n, walk_ns) = timed_section(|| match mode {
            BoolMode::And => {
                if cursors.len() != tokens.len() {
                    return 0;
                }
                self.count_and_intersect(column_id, cursors)
            }
            BoolMode::Or => or_count_unranked(cursors),
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((n, work))
    }

    /// Document frequency for each of `tokens` in `column` — the number
    /// of docs containing each — in input order, read cheaply from the
    /// index **without** decoding posting lists.
    ///
    /// The whole set resolves against **one** FST parse and **one**
    /// coalesced header fetch, rather than one parse + one fetch per
    /// token: the dictionary is opened once, every token is classified
    /// by an in-memory FST lookup (absent → `0`; inline df=1 term → `1`;
    /// PFOR term → its `df`, the first 4 bytes of its 20-byte metadata
    /// header), and all the PFOR headers are pulled in a single batched
    /// [`Self::fetch_term_postings`] call (which coalesces adjacent
    /// ranges into a minimal set of parallel GETs). This matters on the
    /// global-statistics path, where a superfile is probed for every
    /// scored term of a query at once.
    pub async fn term_dfs(
        &self,
        column: &str,
        tokens: &[&str],
    ) -> Result<(Vec<u64>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((Vec::new(), MatchWork::default()));
        }
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let col_meta = &self.columns[column_id as usize];

        // First pass — pure in-memory FST lookups. Absent and inline
        // tokens get their df here; each PFOR token's header range is
        // collected for the single batched fetch below, remembering
        // which token slot it fills so results scatter back in order.
        let mut dfs = vec![0u64; tokens.len()];
        let mut header_ranges: Vec<(usize, Option<usize>)> = Vec::new();
        let mut pfor_slots: Vec<usize> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            let key = make_key(&col_meta.name, token);
            match dict.lookup(&key) {
                None => {}
                Some(packed) => match FstValue::unpack(packed) {
                    FstValue::Inline { .. } => dfs[i] = 1,
                    FstValue::Pfor {
                        metadata_offset, ..
                    } => {
                        header_ranges.push((metadata_offset as usize, Some(TERM_META_SIZE)));
                        pfor_slots.push(i);
                    }
                },
            }
        }

        // One coalesced fetch for every PFOR header; `df` is its first 4
        // bytes. Each header is one planned range (pre-coalesce), and its
        // bytes count as indexed work — the walk read them.
        // +1: the dictionary fetch that resolved the slots.
        let mut work = MatchWork {
            postings_bytes: 0,
            planned_ranges: 1,
            kernel_cpu_ns: 0,
        };
        if !header_ranges.is_empty() {
            let fetched = self.fetch_term_postings(&header_ranges).await?;
            work.planned_ranges += header_ranges.len() as u64;
            let (decoded, decode_ns) = timed_section(|| {
                for (fetched_idx, &slot) in pfor_slots.iter().enumerate() {
                    let header = fetched.get(fetched_idx).ok_or_else(|| {
                        FtsError::Read(ReadError::MalformedVersion(
                            "term_dfs: fetched fewer headers than requested".into(),
                        ))
                    })?;
                    work.postings_bytes += header.len() as u64;
                    let header_bytes = header.as_ref();
                    if header_bytes.len() < U32_BYTES {
                        return Err(FtsError::Read(ReadError::MalformedVersion(
                            "term_dfs: short postings header".into(),
                        )));
                    }
                    dfs[slot] = read_u32_le(&header_bytes[0..U32_BYTES]) as u64;
                }
                Ok(())
            });
            decoded?;
            work.kernel_cpu_ns += decode_ns;
        }
        Ok((dfs, work))
    }

    /// Document frequency of a single `token` in `column`. Thin wrapper
    /// over [`Self::term_dfs`]; see it for how `df` is read without
    /// decoding the posting list. Returns `0` if the token isn't in the
    /// column's dictionary. Used by the candidate planner to estimate a
    /// `WHERE` predicate's match count *ahead of* running `token_match`,
    /// so a predicate matching a large fraction of the superfile can
    /// fall back to a plain scan instead of a (losing) index pushdown.
    pub async fn term_df(&self, column: &str, token: &str) -> Result<(u64, MatchWork), FtsError> {
        let (mut dfs, work) = self.term_dfs(column, &[token]).await?;
        Ok((dfs.pop().unwrap_or(0), work))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder,
        posting::{self, ENCODING_BITSET, ENCODING_PACKED},
        tokenize::AsciiLowerTokenizer,
    };

    #[tokio::test]
    async fn token_match_or_unions_and_intersects_unranked() {
        // build_blob: doc0 "rust async runtime", doc1 "tokio is a rust
        // runtime", doc2 "java spring boot".
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");

        // Single token → its posting list, ascending.
        assert_eq!(
            r.token_match("body", &["rust"], BoolMode::Or)
                .await
                .expect("single")
                .0,
            vec![0, 1]
        );
        // OR = union (rust ∪ java).
        assert_eq!(
            r.token_match("body", &["rust", "java"], BoolMode::Or)
                .await
                .expect("or")
                .0,
            vec![0, 1, 2]
        );
        // AND = intersection (rust ∩ runtime).
        assert_eq!(
            r.token_match("body", &["rust", "runtime"], BoolMode::And)
                .await
                .expect("and")
                .0,
            vec![0, 1]
        );
        // AND with an absent token → empty.
        assert!(
            r.token_match("body", &["rust", "zzz"], BoolMode::And)
                .await
                .expect("and absent")
                .0
                .is_empty()
        );
        // OR ignores an absent token.
        assert_eq!(
            r.token_match("body", &["java", "zzz"], BoolMode::Or)
                .await
                .expect("or absent")
                .0,
            vec![2]
        );
        // Empty token list → empty.
        assert!(
            r.token_match("body", &[], BoolMode::And)
                .await
                .expect("empty")
                .0
                .is_empty()
        );
    }

    #[tokio::test]
    async fn token_match_count_matches_token_match_len() {
        // The counting path (CountSink for AND, or_count_unranked for OR)
        // must agree with token_match's materialized length on every
        // shape — single token, OR union, AND intersection, absent
        // tokens, and the empty list.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let cases: &[(&[&str], BoolMode)] = &[
            (&["rust"], BoolMode::Or),
            (&["rust", "java"], BoolMode::Or),
            (&["rust", "runtime"], BoolMode::And),
            (&["rust", "zzz"], BoolMode::And),
            (&["java", "zzz"], BoolMode::Or),
            (&[], BoolMode::And),
        ];
        for (tokens, mode) in cases {
            let len = r
                .token_match("body", tokens, *mode)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", tokens, *mode)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(count, len, "count vs len for {tokens:?} {mode:?}");
        }
    }

    #[tokio::test]
    async fn or_count_spans_multiple_windows() {
        // The windowed disjunction count must equal the union's true
        // cardinality when the doc-id space spans several OR_WINDOW
        // windows — exercising cross-window accumulation, the per-window
        // popcount + clear, and dedup of docs that match multiple terms
        // within one window. The naive ascending merge (token_match
        // length) is the reference. Tied to OR_WINDOW so it keeps crossing
        // the boundary if the window size changes.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha "); // every doc
            if i.is_multiple_of(2) {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let shapes: &[&[&str]] = &[
            &["alpha"],                           // every doc
            &["beta", "gamma"],                   // overlap on docs % 6
            &["alpha", "beta", "gamma", "delta"], // all overlapping
            &["gamma", "zzz_absent"],             // one absent term
        ];
        for terms in shapes {
            let merge_len = r
                .token_match("body", terms, BoolMode::Or)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", terms, BoolMode::Or)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(
                count, merge_len,
                "windowed count vs merge len for {terms:?}"
            );
        }
        // `alpha` is in every doc, so its union count is exactly N_DOCS —
        // pins the absolute multi-window cardinality, not just agreement
        // with the merge.
        assert_eq!(
            r.token_match_count("body", &["alpha"], BoolMode::Or)
                .await
                .expect("count")
                .0,
            N_DOCS as u64
        );
    }

    #[tokio::test]
    async fn and_count_matches_merge_on_dense_bitset_corpus() {
        // A dense corpus stores common terms as bitset blocks (v4). The
        // intersection count must agree with `token_match`'s flat-merge AND
        // length across both v4 intersection kernels: the bitset-AND
        // (word-parallel presence AND, when every term is dense enough to
        // trip the density gate) and the rarest-driven membership walk (when
        // a sparse term keeps the intersection below the gate). `token_match`
        // AND collects via the decode-based flat-merge, an independent
        // reference from either count kernel.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        const RARE_STRIDE: u32 = 371; // sparse ⇒ below the density gate
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha "); // every doc → dense
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            if i.is_multiple_of(RARE_STRIDE) {
                text.push_str("rare ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let shapes: &[&[&str]] = &[
            &["alpha", "beta"],                   // dense ∩ dense → bitset-AND
            &["alpha", "beta", "gamma"],          // 3 dense → bitset-AND
            &["alpha", "beta", "gamma", "delta"], // 4 dense → bitset-AND
            &["beta", "gamma"],                   // dense ∩ dense, neither anchor
            &["alpha", "rare"],                   // dense ∩ sparse → membership
            &["gamma", "zzz_absent"],             // absent term ⇒ empty
        ];
        for terms in shapes {
            let merge_len = r
                .token_match("body", terms, BoolMode::And)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", terms, BoolMode::And)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(count, merge_len, "AND count vs merge len for {terms:?}");
        }
        // Absolute pin: `alpha` is in every doc, so `alpha ∩ beta` is exactly
        // the docs where `beta` is present (i % 2 == 0) = ⌈N_DOCS / 2⌉.
        assert_eq!(
            r.token_match_count("body", &["alpha", "beta"], BoolMode::And)
                .await
                .expect("count")
                .0,
            u64::from(N_DOCS.div_ceil(2))
        );
    }

    #[tokio::test]
    async fn count_kernels_handle_a_mixed_encoding_term() {
        // A term that is dense early (near-consecutive doc ids ⇒ BITSET blocks)
        // and sparse later (strided ⇒ PACKED blocks) has *both* block encodings
        // in one posting list. The per-block encoding dispatch in the union
        // (`or_cursor_into_bitset`), the membership probe (`TermCursor::contains`
        // advancing across a bitset→packed boundary), and the doc-id decode must
        // each pick the right branch block by block. Uniform-stride corpora
        // never produce this, so drive it explicitly and cross-check every count
        // kernel against `token_match`'s independent flat-merge length.
        const DENSE_END: u32 = 256; // docs 0..256 hold `mix` every doc → BITSET
        const SPARSE_STRIDE: u32 = 30; // docs after that every 30th → PACKED
        const N_DOCS: u32 = 4200;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("common "); // every doc → dense partner
            if i % 2 == 0 {
                text.push_str("even "); // every other doc → dense, non-dominant
            }
            let mix = i < DENSE_END || (i - DENSE_END).is_multiple_of(SPARSE_STRIDE);
            if mix {
                text.push_str("mix ");
            }
            if i.is_multiple_of(37) {
                text.push_str("rareb "); // sparse ⇒ AND stays below the density gate
            }
            text.push_str(&format!("f{}", i % 50));
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Prove `mix` really has both encodings — else the test silently checks
        // nothing about the transition.
        let cursors = r
            .build_term_cursors(0, &["mix"], None, true)
            .await
            .expect("build cursors");
        let mix = &cursors[0];
        let (mut saw_bitset, mut saw_packed) = (false, false);
        for blk in mix.blocks.iter() {
            match mix.bytes.as_ref()[blk.block_byte_offset + posting::ENCODING_OFF] {
                ENCODING_BITSET => saw_bitset = true,
                ENCODING_PACKED => saw_packed = true,
                other => panic!("unexpected encoding byte {other}"),
            }
        }
        assert!(
            saw_bitset && saw_packed,
            "`mix` must carry both BITSET and PACKED blocks (bitset={saw_bitset}, packed={saw_packed})"
        );

        // Every count kernel over the mixed term must agree with the flat-merge.
        // - `mix` alone: doc-id decode across mixed blocks.
        // - `mix ∪ even`: dense union, neither dominant ⇒ `or_count_bitset` →
        //   `or_cursor_into_bitset` takes both the word-copy and scatter branches.
        // - `mix ∩ common`: `common` dominates → membership walk probes `mix`
        //   (`contains`) across the bitset→packed boundary.
        // - `mix ∩ rareb`: rarest is sparse ⇒ membership drives by `rareb` and
        //   probes `mix`, again crossing the encoding boundary.
        let cases: &[(&[&str], BoolMode)] = &[
            (&["mix"], BoolMode::Or),
            (&["mix", "even"], BoolMode::Or),
            (&["mix", "common"], BoolMode::And),
            (&["mix", "rareb"], BoolMode::And),
            (&["mix", "even", "common"], BoolMode::And),
        ];
        for (tokens, mode) in cases {
            let merge_len = r
                .token_match("body", tokens, *mode)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", tokens, *mode)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(
                count, merge_len,
                "mixed-encoding count for {tokens:?} {mode:?}"
            );
        }
    }

    #[tokio::test]
    async fn intersection_count_edges_across_the_density_gate() {
        // Dispatch edges in `count_and_intersect`: the density-gate boundary
        // (bitset-AND just above, membership just below `min_df*DIVISOR >=
        // max_doc`), a single-cursor AND, and inline (df=1) cursors in both an
        // intersection (`contains` inline branch, true and false) and a dense
        // union (`or_cursor_into_bitset` inline branch). All cross-checked
        // against `token_match`'s independent flat-merge length.
        const N_DOCS: u32 = 4096; // max doc id 4095
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("common"); // every doc
            text.push_str(if i.is_multiple_of(2) { " even" } else { " odd" });
            // `gatehi` in 256 docs (df*16 = 4096 ≥ 4095 ⇒ bitset-AND);
            // `gatelo` in 255 docs (df*16 = 4080 < 4095 ⇒ membership).
            if i.is_multiple_of(16) {
                text.push_str(" gatehi");
            }
            if i.is_multiple_of(16) && i != 0 {
                text.push_str(" gatelo");
            }
            if i == 100 {
                text.push_str(" inlinea inlineb"); // two df=1 terms on one doc
            }
            if i == 200 {
                text.push_str(" inlinec"); // a df=1 term on a different doc
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Pin the gate arithmetic: `gatehi` lands on/above the switch, `gatelo`
        // just below, so the two intersections below take different kernels.
        let max_doc = u64::from(N_DOCS - 1);
        let hi = r.term_df("body", "gatehi").await.expect("df").0;
        let lo = r.term_df("body", "gatelo").await.expect("df").0;
        assert!(
            hi.saturating_mul(OR_COUNT_BITSET_DENSITY_DIVISOR) >= max_doc,
            "gatehi df={hi} must sit on/above the density gate"
        );
        assert!(
            lo.saturating_mul(OR_COUNT_BITSET_DENSITY_DIVISOR) < max_doc,
            "gatelo df={lo} must sit below the density gate"
        );
        for token in ["inlinea", "inlineb", "inlinec"] {
            assert_eq!(
                r.term_df("body", token).await.expect("df").0,
                1,
                "{token} df"
            );
        }

        let and_cases: &[&[&str]] = &[
            &["common", "gatehi"],   // bitset-AND — just above the gate
            &["common", "gatelo"],   // membership — just below the gate
            &["common"],             // single-cursor AND
            &["inlinea", "inlineb"], // two df=1 on the same doc ⇒ contains inline true
            &["inlinea", "inlinec"], // df=1 on different docs ⇒ contains inline false
            &["inlinea", "common"],  // inline drives, bitset term probed
        ];
        for tokens in and_cases {
            let merge_len = r
                .token_match("body", tokens, BoolMode::And)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", tokens, BoolMode::And)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(count, merge_len, "AND edge count for {tokens:?}");
        }

        // Dense union containing an inline term ⇒ `or_count_bitset` →
        // the inline branch of `or_cursor_into_bitset`.
        let or_tokens: &[&str] = &["inlinea", "even", "odd"];
        let merge_len = r
            .token_match("body", or_tokens, BoolMode::Or)
            .await
            .expect("token_match")
            .0
            .len() as u64;
        let count = r
            .token_match_count("body", or_tokens, BoolMode::Or)
            .await
            .expect("token_match_count")
            .0;
        assert_eq!(count, merge_len, "OR-with-inline count");

        // Absolute pins on the inline intersections: same doc ⇒ 1, disjoint ⇒ 0.
        assert_eq!(
            r.token_match_count("body", &["inlinea", "inlineb"], BoolMode::And)
                .await
                .expect("count")
                .0,
            1
        );
        assert_eq!(
            r.token_match_count("body", &["inlinea", "inlinec"], BoolMode::And)
                .await
                .expect("count")
                .0,
            0
        );
    }

    #[tokio::test]
    async fn or_count_anchored_matches_merge_on_dominant_term() {
        // When one term's df dwarfs the rest, the OR count takes the
        // df-anchored path (`df(dominant) + |others \ dominant|`) instead
        // of walking the dominant list. It must return the identical
        // cardinality as the independent naive ascending merge
        // (`token_match` length). The corpus spans several OR_WINDOW
        // windows and — crucially — leaves gaps where the dominant term
        // is *absent* but a rare term is present, so the "doc not in the
        // anchor" branch is exercised, not just `df(dominant)`.
        const N_DOCS: u32 = OR_WINDOW * 2 + 137;
        const RARE_STRIDE: u32 = 250; // rare term hits ~1/250 docs
        const RAREB_STRIDE: u32 = 400;
        const HOLE_STRIDE: u32 = 300; // docs missing the dominant term
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            // `common` is in almost every doc (the dominant term) — but
            // not the holes, so the anchor genuinely lacks some docs the
            // rare terms supply.
            if !i.is_multiple_of(HOLE_STRIDE) {
                text.push_str("common ");
            }
            if i.is_multiple_of(RARE_STRIDE) {
                text.push_str("rare ");
            }
            if i.is_multiple_of(RAREB_STRIDE) {
                text.push_str("rareb ");
            }
            // Guarantee no empty doc (a hole that hits neither rare term).
            if text.is_empty() {
                text.push_str("filler ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // `common` alone is far more than 4× `rare` + `rareb` combined, so
        // these shapes drive the anchored path; `filler` never dominates.
        let shapes: &[&[&str]] = &[
            &["common", "rare"],
            &["common", "rare", "rareb"],
            &["common", "rareb", "zzz_absent"],
            &["common", "filler"],
        ];
        for terms in shapes {
            let merge_len = r
                .token_match("body", terms, BoolMode::Or)
                .await
                .expect("token_match")
                .0
                .len() as u64;
            let count = r
                .token_match_count("body", terms, BoolMode::Or)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(
                count, merge_len,
                "anchored count vs merge len for {terms:?}"
            );
        }
    }

    #[tokio::test]
    async fn term_df_reports_document_frequency() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // common → df 3 (PFOR header read), rust → df 2 (PFOR),
        // uniqzero → df 1 (inline FST value), absent → 0.
        assert_eq!(r.term_df("body", "common").await.expect("df").0, 3);
        assert_eq!(r.term_df("body", "rust").await.expect("df").0, 2);
        assert_eq!(r.term_df("body", "uniqzero").await.expect("df").0, 1);
        assert_eq!(r.term_df("body", "missing").await.expect("df").0, 0);
    }

    #[tokio::test]
    async fn term_df_unknown_column_errors() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let err = r.term_df("nope", "rust").await.expect_err("error");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }

    #[tokio::test]
    async fn term_dfs_matches_per_term_term_df() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Interleave the FST value kinds — PFOR (df>1), absent, inline
        // (df=1), PFOR, absent — so a slot-mapping bug in the batched
        // path (which fetches only the PFOR headers, then scatters the
        // results back) would surface as a mismatch here.
        let tokens = ["rust", "missing", "uniqzero", "common", "absent2"];
        let batched = r.term_dfs("body", &tokens).await.expect("term_dfs").0;
        // Element-wise identical to resolving each token on its own.
        let mut per_term = Vec::with_capacity(tokens.len());
        for t in tokens {
            per_term.push(r.term_df("body", t).await.expect("term_df").0);
        }
        assert_eq!(
            batched, per_term,
            "batched term_dfs must equal per-term term_df"
        );
        // …and matches the planted ground truth (common=3, rust=2,
        // uniqzero=1 inline, absent tokens=0).
        assert_eq!(batched, vec![2, 0, 1, 3, 0], "planted document frequencies");
        // Empty input short-circuits to empty output (no dict open, no fetch).
        assert!(r.term_dfs("body", &[]).await.expect("empty").0.is_empty());
    }
}
