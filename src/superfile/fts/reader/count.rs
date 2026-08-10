// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unranked FTS match/count kernels on [`FtsReader`]: the heterogeneous
//! atom walk plus the token/phrase match-id, match-count, and term-df
//! entry points (no BM25 scoring, no top-k). Its own `impl FtsReader`
//! block, split from the reader `core`.

use crate::runtime_metrics::op_stats::timed_section;
use crate::superfile::{
    ReadError,
    error::FtsError,
    format::fts::{U32_BYTES, term_meta},
    fts::{
        builder::TERM_META_SIZE,
        dict::{DictReader, make_key},
        fst_value::FstValue,
    },
};

use super::core::*;
use super::cursor::TermCursor;
use super::filter::AtomExcludeFilter;
use super::options::BoolMode;
use super::phrase::AnyCursor;
use super::work::{MatchWork, atom_cursor_bytes, atom_planned_ranges};

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
                    let mut aligned = target;
                    let mut i = 0usize;
                    while i < atoms.len() {
                        let a = &mut atoms[i];
                        a.skip_to(aligned)?;
                        if a.is_exhausted() {
                            break 'docs;
                        }
                        let here = a.current_doc_id();
                        if here > aligned {
                            aligned = here;
                            i = 0;
                            continue;
                        }
                        i += 1;
                    }
                    let admitted = match filter.as_mut() {
                        Some(f) => f.admits(aligned)?,
                        None => true,
                    };
                    if admitted {
                        on_doc(aligned);
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
        let cursors = self.build_term_cursors(column_id, tokens, None).await?;
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
        let cursors = self.build_term_cursors(column_id, tokens, None).await?;
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
