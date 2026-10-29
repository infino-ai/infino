// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unranked FTS match/count kernels on [`FtsReader`]: the heterogeneous
//! atom walk plus the token/phrase match-id, match-count, and term-df
//! entry points (no BM25 scoring, no top-k). Its own `impl FtsReader`
//! block, split from the reader `core`.

use std::{cmp::Reverse, collections::BinaryHeap, str::from_utf8};

#[cfg(any(test, feature = "test-helpers"))]
use super::cursor::TermMeta;
use super::{
    core::*,
    cursor::{CursorUse, TermCursor},
    filter::AtomExcludeFilter,
    metadata::NormTable,
    options::BoolMode,
    phrase::{AnyCursor, approx_align, batch_phrase_with_terms},
    scorers::{count_and_planned, count_excluding},
    search::FetchedTermMemo,
    sink::{TopKEntry, finish_head},
    work::{MatchWork, atom_cursor_bytes, atom_planned_ranges},
};
#[cfg(any(test, feature = "test-helpers"))]
use crate::superfile::fts::posting::{
    ENCODING_BITSET, ENCODING_PACKED, ENCODING_PATCHED, ENCODING_RUN,
};
use crate::{
    runtime_metrics::op_stats::timed_section,
    superfile::{
        ReadError,
        error::FtsError,
        format::fts::U32_BYTES,
        fts::{bm25, builder::TERM_META_SIZE, short::short_df, tokenize::Phrase},
        id_space::{DocMap, FtsDocId, RowId},
    },
    utils::terms::{DictEntry, make_key},
};

/// Planned dictionary reads for a cursor build over `tokens`: none when a
/// memo already resolves every token, one otherwise.
fn dictionary_fetches(tokens: &[&str], prefetched: Option<&FetchedTermMemo>) -> u64 {
    match prefetched.is_some_and(|m| m.covers(tokens)) {
        true => 0,
        false => 1,
    }
}

/// One term's entry in a table-level term index, as read from one
/// superfile: see [`FtsReader::term_index_facts_with`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TermIndexFact {
    /// Documents in this superfile containing the term.
    pub(crate) df: u64,
    /// The largest key — the length-normalized term frequency at this
    /// superfile's own average length and parameters, the factor a score
    /// multiplies the idf by — of a document of the term here.
    pub(crate) bound: f32,
    /// The term's dictionary entry: where its postings sit.
    pub(crate) entry: DictEntry,
    /// The term's best documents here, to the depth the caller asked for.
    pub(crate) head: TermHead,
}

/// What one term contributes to the term index's tier from this
/// superfile: its best documents by impact and what it holds back.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TermHead {
    /// `(row, term frequency, length bucket)` in descending impact, at
    /// most the depth asked for.
    pub(crate) docs: Vec<(u32, u32, u8)>,
    /// The length-normalized term frequency — `tf / (tf + norm)` at this
    /// file's average length and parameters — of the best document not in
    /// `docs`; `0.0` when `docs` is the term's whole posting list here.
    pub(crate) tail: f32,
}

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
        let atom_postings = atoms
            .iter()
            .map(AnyCursor::df_bound)
            .fold(0u64, u64::saturating_add);
        let sparse_union = matches!(mode, BoolMode::Or)
            && or_walk_prefers_heap(atoms.len(), atom_postings, u64::from(self.n_docs));
        match mode {
            BoolMode::Or if sparse_union => {
                // A sparse union over many atoms: a heap keyed by current
                // doc, so a step costs the log of the atom count rather
                // than the atom count (see `or_walk_unranked`). Atoms
                // sitting on the same doc pop consecutively and advance
                // together past it.
                let mut heap =
                    doc_heap(&atoms, |a| (!a.is_exhausted()).then(|| a.current_doc_id()));
                // The atoms sitting on the doc just popped, gathered before any
                // of them advances so their re-pushes never interleave with
                // the peeks. One buffer for the whole walk.
                let mut at_doc: Vec<usize> = Vec::new();
                while let Some(Reverse((doc, first))) = heap.pop() {
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
                    at_doc.clear();
                    at_doc.push(first);
                    while let Some(Reverse((d, i))) = heap.peek().copied() {
                        if d != doc {
                            break;
                        }
                        heap.pop();
                        at_doc.push(i);
                    }
                    for &i in &at_doc {
                        let a = &mut atoms[i];
                        a.skip_to(next)?;
                        if !a.is_exhausted() {
                            heap.push(Reverse((a.current_doc_id(), i)));
                        }
                    }
                }
                Ok(())
            }
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
                // One phrase ANDed with terms walks the intersection itself, a
                // block at a time, every doc it lands on verified.
                let batched = batch_phrase_with_terms(&mut atoms)?;
                let mut target = 0u32;
                'docs: loop {
                    let aligned = match batched {
                        Some(at) => {
                            let a = &mut atoms[at];
                            a.skip_to(target)?;
                            if a.is_exhausted() {
                                break 'docs;
                            }
                            a.current_doc_id()
                        }
                        // Phase 1 — every atom aligned by its cheap
                        // approximation, so a rare co-clause prunes the
                        // candidates before any phrase pays for adjacency.
                        None => match approx_align(&mut atoms, target) {
                            Some(aligned) => aligned,
                            None => break 'docs,
                        },
                    };
                    // Phase 2 — every approximation agrees on `aligned`. Verify
                    // the phrase atoms' positions here only: the intersection is
                    // already down to the co-occurring docs, so position decode
                    // runs on that handful, not on a phrase's whole match set.
                    // The batched walk verified its doc already.
                    let mut matched = true;
                    if batched.is_none() {
                        for a in atoms.iter_mut() {
                            if !a.verify_at(aligned)? {
                                matched = false;
                                break;
                            }
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
        phrases: &[Phrase<String>],
        mode: BoolMode,
    ) -> Result<(Vec<RowId>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        // Unranked: built to match, so no global idf and no norms; the
        // column's length array is never read.
        let (built, dict_ranges) = self
            .build_atom_cursors(
                column_id,
                terms,
                phrases,
                QueryIdf::ThisFile,
                None,
                CursorUse::Match,
            )
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
            self.walk_atoms_match(atoms, mode, None, |d| out.push(FtsDocId::new(d)))
                .map(|()| out)
        });
        work.kernel_cpu_ns = walk_ns;
        let docs = walk?;
        if !docs.is_empty() {
            self.ensure_doc_map().await?;
        }
        Ok((self.ids_to_rows(docs)?, work))
    }

    /// Phrase-aware unranked match **count** — the atoms sibling of
    /// [`Self::token_match_count`].
    pub(crate) async fn atoms_match_count(
        &self,
        column: &str,
        terms: &[&str],
        phrases: &[Phrase<String>],
        mode: BoolMode,
        neg_terms: &[&str],
        neg_phrases: &[Phrase<String>],
    ) -> Result<(u64, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        // Plain terms on both sides: count by probing or folding, whichever
        // reads less, instead of walking the atoms a document at a time.
        if phrases.is_empty() && neg_phrases.is_empty() && !neg_terms.is_empty() {
            let pos = self
                .build_term_cursors(
                    column_id,
                    terms,
                    QueryIdf::ThisFile,
                    CursorUse::Count,
                    None,
                    None,
                )
                .await?;
            let mut work = MatchWork::for_cursors(&pos);
            if mode == BoolMode::And && pos.len() != terms.len() {
                return Ok((0, work));
            }
            let neg = self
                .build_term_cursors(
                    column_id,
                    neg_terms,
                    QueryIdf::ThisFile,
                    CursorUse::Count,
                    None,
                    None,
                )
                .await?;
            work.postings_bytes += MatchWork::for_cursors(&neg).postings_bytes;
            let (n, walk_ns) = timed_section(|| count_excluding(pos, mode == BoolMode::And, neg));
            work.kernel_cpu_ns = walk_ns;
            return Ok((n, work));
        }
        // Unranked: built to match, so no global idf and no norms; the
        // column's length array is never read.
        let (built, dict_ranges) = self
            .build_atom_cursors(
                column_id,
                terms,
                phrases,
                QueryIdf::ThisFile,
                None,
                CursorUse::Match,
            )
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
                .build_atom_cursors(
                    column_id,
                    neg_terms,
                    neg_phrases,
                    QueryIdf::ThisFile,
                    None,
                    CursorUse::Match,
                )
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
    ) -> Result<(Vec<RowId>, MatchWork), FtsError> {
        self.token_match_prefetched(column, tokens, mode, None)
            .await
    }

    /// [`Self::token_match`] with the terms a caller has already resolved
    /// — from a table-level term index that knows where each term's
    /// postings sit — served from `prefetched`. When the memo covers every
    /// token the dictionary is never opened; a token it lacks resolves
    /// through the dictionary as usual.
    pub(crate) async fn token_match_prefetched(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(Vec<RowId>, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((Vec::new(), MatchWork::default()));
        }
        let cursors = self
            .build_term_cursors(
                column_id,
                tokens,
                QueryIdf::ThisFile,
                CursorUse::Count,
                None,
                prefetched,
            )
            .await?;
        // Tallied before the mode branch: the cursors that DID build cost
        // their bytes even when a missing AND token empties the result.
        // +1 for the dictionary fetch, when the build had to make one.
        let mut work = MatchWork::for_cursors(&cursors);
        work.planned_ranges += dictionary_fetches(tokens, prefetched);
        let (docs, walk_ns) = timed_section(|| match mode {
            BoolMode::And => {
                // AND needs every token present; a missing token ⇒ empty
                // set. Otherwise intersect via the same optimized
                // block flat-merge the ranked scorer uses.
                if cursors.len() != tokens.len() {
                    return Vec::new();
                }
                self.collect_and_intersect(cursors)
            }
            BoolMode::Or => or_merge_unranked(cursors)
                .into_iter()
                .map(FtsDocId::new)
                .collect(),
        });
        work.kernel_cpu_ns = walk_ns;
        if !docs.is_empty() {
            self.ensure_doc_map().await?;
        }
        Ok((self.ids_to_rows(docs)?, work))
    }

    /// The cursors an `Or` [`Self::token_match_prefetched`] walks over
    /// `tokens`, built once (the postings fetch and skip-table parse) for
    /// [`Self::token_match_range_prebuilt`] to walk one doc-id slice at a
    /// time, with what the build read. Slices of one superfile share the
    /// set; each walks a clone of its cursors.
    pub(crate) async fn match_cursor_set(
        &self,
        column: &str,
        tokens: &[&str],
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(OrCursorSet, MatchWork), FtsError> {
        let set = self
            .build_cursor_set(
                column,
                tokens,
                QueryIdf::ThisFile,
                CursorUse::Count,
                prefetched,
            )
            .await?;
        let work = MatchWork {
            postings_bytes: set.postings_bytes(),
            planned_ranges: set.planned_ranges() + dictionary_fetches(tokens, prefetched),
            kernel_cpu_ns: 0,
        };
        Ok((set, work))
    }

    /// The rows whose doc ids in `[start, end)` hold any of `set`'s terms,
    /// ascending: one slice of an `Or` [`Self::token_match_prefetched`].
    /// The slices of a superfile's doc-id space together match exactly
    /// what the whole walk matches; the range is a cut of the blob's ids
    /// and carries no row meaning, so each slice's rows are put in order
    /// on their own. The work holds the walk's time alone; the set's build
    /// counted its bytes.
    pub(crate) fn token_match_range_prebuilt(
        &self,
        set: &OrCursorSet,
        start: u32,
        end: u32,
    ) -> Result<(Vec<RowId>, MatchWork), FtsError> {
        let mut work = MatchWork::default();
        if set.cursors.is_empty() || start >= end {
            return Ok((Vec::new(), work));
        }
        let (docs, walk_ns) = timed_section(|| {
            or_merge_unranked_in(set.cursors.clone(), start, end)
                .into_iter()
                .map(FtsDocId::new)
                .collect::<Vec<_>>()
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((self.ids_to_rows(docs)?, work))
    }

    /// Unranked token-match **count** — the cardinality
    /// [`token_match`](Self::token_match) would return, without
    /// materializing the doc-id `Vec`. The AND path counts its
    /// intersection, the OR path its union walk; both skip the
    /// `Vec<u32>` so a high-cardinality count doesn't allocate one id
    /// per match.
    pub async fn token_match_count(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
    ) -> Result<(u64, MatchWork), FtsError> {
        self.token_match_count_prefetched(column, tokens, mode, None)
            .await
    }

    /// [`Self::token_match_count`] served from `prefetched`, as
    /// [`Self::token_match_prefetched`] is.
    pub(crate) async fn token_match_count_prefetched(
        &self,
        column: &str,
        tokens: &[&str],
        mode: BoolMode,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(u64, MatchWork), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((0, MatchWork::default()));
        }
        let cursors = self
            .build_term_cursors(
                column_id,
                tokens,
                QueryIdf::ThisFile,
                CursorUse::Count,
                None,
                prefetched,
            )
            .await?;
        let mut work = MatchWork::for_cursors(&cursors);
        work.planned_ranges += dictionary_fetches(tokens, prefetched);
        let (n, walk_ns) = timed_section(|| match mode {
            BoolMode::And => {
                if cursors.len() != tokens.len() {
                    return 0;
                }
                count_and_planned(cursors)
            }
            BoolMode::Or => or_count_unranked(cursors),
        });
        work.kernel_cpu_ns = walk_ns;
        Ok((n, work))
    }

    /// What a table-level term index records about each of `tokens` in
    /// `column`, in input order — `None` for an absent token: its `df`, the
    /// upper bound on the BM25 score it can reach in this superfile, and
    /// its dictionary entry. One dictionary resolution and one cursor build
    /// serve all three: a built cursor already carries the term's `df` and
    /// its maximum block score, so nothing is read twice.
    ///
    /// The bound is scored at this superfile's own idf and declared
    /// parameters, exactly as its stored block maxima are: for a long term
    /// the maximum over its skip entries, for a short or inline term the
    /// maximum over its postings. A query rescales it for a global idf the
    /// same way the in-superfile bounds are.
    #[cfg(test)]
    pub(crate) async fn term_index_facts(
        &self,
        column: &str,
        tokens: &[&str],
    ) -> Result<Vec<Option<TermIndexFact>>, FtsError> {
        self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let dict_bytes = self.dict_bytes_async().await?;
        self.term_index_facts_with(&dict_bytes, column, tokens, 0)
            .await
    }

    /// [`Self::term_index_facts`] over `dict_bytes`, this reader's dictionary
    /// already fetched by the caller.
    pub(crate) async fn term_index_facts_with(
        &self,
        dict_bytes: &[u8],
        column: &str,
        tokens: &[&str],
        head_depth: usize,
    ) -> Result<Vec<Option<TermIndexFact>>, FtsError> {
        self.debug_assert_own_dict(dict_bytes);
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let dict = Self::open_dict(dict_bytes)?;
        let col_meta = &self.columns[column_id as usize];
        let entries: Vec<Option<DictEntry>> = tokens
            .iter()
            .map(|token| dict.lookup(&make_key(&col_meta.name, token)))
            .collect();
        let cursors = self
            .build_term_cursors_opt(
                column_id,
                tokens,
                QueryIdf::ThisFile,
                CursorUse::Score,
                None,
                None,
                Some(dict_bytes),
            )
            .await?;
        // A head names its documents by row, so a term with postings here
        // needs the doc-id map; terms this file lacks read none of it.
        let doc_map = match cursors.iter().any(Option::is_some) {
            true => Some(self.ensure_doc_map().await?),
            false => None,
        };
        let norms = col_meta.dl_norm_k1();
        Ok(entries
            .into_iter()
            .zip(cursors)
            .map(|(entry, cursor)| {
                let mut cursor = cursor?;
                // A depth of zero is read as one: an empty head would read
                // as a term no document carries.
                let (head, bound) = self.head_of(&mut cursor, norms, doc_map?, head_depth.max(1));
                Some(TermIndexFact {
                    df: cursor.df,
                    bound,
                    entry: entry?,
                    head,
                })
            })
            .collect())
    }

    /// The best `depth` documents of a scored cursor by impact, with the
    /// best impact left out as the tail, and the largest key among them —
    /// the term's bound here. Blocks whose maximum cannot beat the running
    /// `depth`-th impact are skipped undecoded, as the single-term search
    /// skips them; the cursor is consumed.
    fn head_of(
        &self,
        cursor: &mut TermCursor,
        norms: &NormTable,
        doc_map: &DocMap,
        depth: usize,
    ) -> (TermHead, f32) {
        // Most terms have a document or two: the heap is sized by what the
        // term has, up to the depth and the tail.
        let mut heap: BinaryHeap<TopKEntry<(FtsDocId, u32)>> =
            BinaryHeap::with_capacity((depth + 1).min(cursor.df as usize + 1));
        while !cursor.is_exhausted() {
            if heap.len() > depth
                && let Some(TopKEntry(min, _)) = heap.peek()
                && cursor.current_block_max_bm25() <= *min
            {
                cursor.skip_blocks_under(0.0, *min, u32::MAX);
                continue;
            }
            let doc = cursor.current_doc_id();
            let tf = cursor.current_tf();
            let score = bm25::score_with_dl_norm_k1(cursor.idf_weight, tf, norms.get(doc));
            if heap.len() <= depth {
                heap.push(TopKEntry(score, (FtsDocId::new(doc), tf)));
            } else if let Some(mut worst) = heap.peek_mut()
                && score > worst.0
            {
                *worst = TopKEntry(score, (FtsDocId::new(doc), tf));
            }
            cursor.next();
        }
        // The key of a document, as the term index orders its tiers, at
        // the average length and parameters this file scores at.
        let key = |(id, tf): (FtsDocId, u32)| bm25::tf_factor(tf, norms.get(id.get()));
        let mut hits = Vec::new();
        let (bound, tail) = finish_head(&mut heap, depth, &mut hits, key);
        let buckets = norms.buckets();
        let docs = hits
            .into_iter()
            .map(|((id, tf), _)| (doc_map.row_of(id).get(), tf, buckets[id.get() as usize]))
            .collect();
        (TermHead { docs, tail }, bound)
    }

    /// Up to `limit` of `column`'s terms after `after` (from the first when
    /// `None`), in term order, each with what [`Self::term_index_facts_with`]
    /// records for it. Walks the dictionary with its values, avoiding a
    /// separate term lookup for each entry.A df=1 inline term gets its fact
    /// straight from its entry: its one posting's score is its bound, so no
    /// cursor is built. `dict_bytes` is this reader's dictionary, fetched
    /// once by the caller.
    pub(crate) async fn term_index_facts_after(
        &self,
        dict_bytes: &[u8],
        column: &str,
        after: Option<&[u8]>,
        limit: usize,
        head_depth: usize,
    ) -> Result<Vec<(Vec<u8>, Option<TermIndexFact>)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        // The walk starts at `after` itself, which the caller already has.
        let mut entries =
            self.column_terms_from(dict_bytes, column_id, after.unwrap_or(b""), limit + 1)?;
        if let Some(after) = after
            && entries
                .first()
                .is_some_and(|(term, _)| term.as_slice() == after)
        {
            entries.remove(0);
        }
        entries.truncate(limit);
        self.ensure_norms(column_id).await?;
        // Every term walked here has a document, and a head names it by row.
        let doc_map = self.ensure_doc_map().await?;
        let col_meta = &self.columns[column_id as usize];
        let mut out = Vec::with_capacity(entries.len());
        let mut with_postings: Vec<usize> = Vec::new();
        for (term, entry) in entries {
            let fact = match entry {
                DictEntry::Inline { doc_id, tf } => {
                    // The one posting is the whole list: its key is the
                    // bound, the head is that document, and nothing is left
                    // out.
                    let tf = col_meta.inline_tf(tf);
                    let bound = bm25::tf_factor(tf, col_meta.dl_norm_k1().get(doc_id));
                    let head = TermHead {
                        docs: vec![(
                            doc_map.row_of(FtsDocId::new(doc_id)).get(),
                            tf,
                            col_meta.dl_norm_k1().buckets()[doc_id as usize],
                        )],
                        tail: 0.0,
                    };
                    Some(TermIndexFact {
                        df: 1,
                        bound,
                        entry,
                        head,
                    })
                }
                DictEntry::Pfor { .. } => {
                    with_postings.push(out.len());
                    None
                }
            };
            out.push((term, fact));
        }
        // The rest need their postings read: the lookup path serves them.
        let facts = {
            let terms = with_postings
                .iter()
                .map(|&i| from_utf8(&out[i].0))
                .collect::<Result<Vec<&str>, _>>()
                .map_err(|_| FtsError::Read(ReadError::Malformed("non-utf8 term".into())))?;
            self.term_index_facts_with(dict_bytes, column, &terms, head_depth)
                .await?
        };
        for (i, fact) in with_postings.into_iter().zip(facts) {
            out[i].1 = fact;
        }
        Ok(out)
    }

    /// Document frequency for each of `tokens` in `column` — the number
    /// of docs containing each — in input order, read cheaply from the
    /// index **without** decoding posting lists.
    ///
    /// The whole set resolves against **one** dictionary parse and **one**
    /// coalesced header fetch, rather than one parse + one fetch per token: the
    /// dictionary is opened once, every token is classified by an in-memory
    /// dictionary lookup (absent → `0`; inline df=1 term → `1`; PFOR term → its
    /// `df`, the first 4 bytes of its 20-byte metadata header), and all the
    /// PFOR headers are pulled in a single batched
    /// [`Self::fetch_term_postings`] call (which coalesces adjacent ranges into
    /// a minimal set of parallel GETs). This matters on the global-statistics
    /// path, where a superfile is probed for every scored term of a query at
    /// once.
    pub async fn term_dfs(
        &self,
        column: &str,
        tokens: &[&str],
    ) -> Result<(Vec<u64>, MatchWork), FtsError> {
        self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((Vec::new(), MatchWork::default()));
        }
        let dict_bytes = self.dict_bytes_async().await?;
        self.term_dfs_with(&dict_bytes, column, tokens).await
    }

    /// [`Self::term_dfs`] over `dict_bytes`, this reader's dictionary already
    /// fetched by the caller.
    pub(crate) async fn term_dfs_with(
        &self,
        dict_bytes: &[u8],
        column: &str,
        tokens: &[&str],
    ) -> Result<(Vec<u64>, MatchWork), FtsError> {
        self.debug_assert_own_dict(dict_bytes);
        let column_id = self.resolve_column_id(column)?;
        if tokens.is_empty() {
            return Ok((Vec::new(), MatchWork::default()));
        }
        let dict = Self::open_dict(dict_bytes)?;
        let col_meta = &self.columns[column_id as usize];

        // First pass — pure in-memory dictionary lookups. Absent and inline
        // tokens get their df here; each PFOR token's header range is
        // collected for the single batched fetch below, remembering
        // which token slot it fills so results scatter back in order.
        let mut dfs = vec![0u64; tokens.len()];
        let mut header_ranges: Vec<(usize, usize)> = Vec::new();
        let mut pfor_slots: Vec<(usize, bool)> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            let key = make_key(&col_meta.name, token);
            match dict.lookup(&key) {
                None => {}
                Some(packed) => match packed {
                    DictEntry::Inline { .. } => dfs[i] = 1,
                    DictEntry::Pfor {
                        metadata_offset,
                        postings_length,
                        short,
                    } => {
                        // A long term's df heads its 20-byte header; a
                        // short body is at most a few hundred bytes and
                        // leads with its df, so fetch it whole.
                        let len = match short {
                            true => postings_length as usize,
                            false => TERM_META_SIZE,
                        };
                        header_ranges.push((metadata_offset as usize, len));
                        pfor_slots.push((i, short));
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
                for (fetched_idx, &(slot, short)) in pfor_slots.iter().enumerate() {
                    let header = fetched.get(fetched_idx).ok_or_else(|| {
                        FtsError::Read(ReadError::Malformed(
                            "term_dfs: fetched fewer headers than requested".into(),
                        ))
                    })?;
                    work.postings_bytes += header.len() as u64;
                    let header_bytes = header.as_ref();
                    if short {
                        dfs[slot] = u64::from(short_df(header_bytes).ok_or_else(|| {
                            FtsError::Read(ReadError::Malformed(
                                "term_dfs: short-form term body does not decode".into(),
                            ))
                        })?);
                        continue;
                    }
                    if header_bytes.len() < U32_BYTES {
                        return Err(FtsError::Read(ReadError::Malformed(
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

    /// How `term`'s postings are laid out in `column` of this file, or
    /// `None` if the term is absent. Benchmarks report this for every term
    /// their quality battery names, so the battery is known to reach every
    /// posting form (inline, short, packed, patched, bitset) rather than
    /// assumed to from the term's rank. Reads the term's postings once, the
    /// way a search would, and parses only the block headers.
    #[cfg(any(test, feature = "test-helpers"))]
    pub async fn term_layout(
        &self,
        column: &str,
        term: &str,
    ) -> Result<Option<TermLayout>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let col_meta = &self.columns[column_id as usize];
        let dict_bytes = self.dict_bytes_async().await?;
        let dict = Self::open_dict(&dict_bytes)?;
        let key = make_key(&col_meta.name, term);
        let Some(packed) = dict.lookup(&key) else {
            return Ok(None);
        };
        let (metadata_offset, postings_length, short) = match packed {
            DictEntry::Inline { .. } => {
                return Ok(Some(TermLayout {
                    df: 1,
                    inline: true,
                    ..TermLayout::default()
                }));
            }
            DictEntry::Pfor {
                metadata_offset,
                postings_length,
                short,
            } => (metadata_offset, postings_length, short),
        };
        let mut fetched = self
            .fetch_term_postings(&[(metadata_offset as usize, postings_length as usize)])
            .await?;
        let bytes = fetched.pop().expect("one fetched range for one PFOR term");
        if short {
            let df = short_df(bytes.as_ref()).ok_or_else(|| {
                FtsError::Read(ReadError::Malformed(
                    "term_layout: short-form term body does not decode".into(),
                ))
            })?;
            return Ok(Some(TermLayout {
                df: u64::from(df),
                short: true,
                num_blocks: 1,
                ..TermLayout::default()
            }));
        }
        let meta = TermMeta::parse(bytes.as_ref(), 0, col_meta.positions)?;
        let cursor = TermCursor::new(bytes, col_meta, TermIdf::Absent, 1, CursorUse::Count, None)?;
        let mut layout = TermLayout {
            df: meta.df,
            num_blocks: meta.num_blocks,
            has_coarse: meta.has_coarse,
            ..TermLayout::default()
        };
        for b in 0..cursor.block_count() {
            match cursor.block_header(b).encoding {
                ENCODING_PACKED => layout.packed_blocks += 1,
                ENCODING_PATCHED => layout.patched_blocks += 1,
                ENCODING_BITSET => layout.bitset_blocks += 1,
                ENCODING_RUN => layout.run_blocks += 1,
                other => {
                    return Err(FtsError::Read(ReadError::Malformed(format!(
                        "term_layout: block {b} carries unknown encoding {other}"
                    ))));
                }
            }
        }
        Ok(Some(layout))
    }
}

/// The posting form one term takes in one file — see
/// [`FtsReader::term_layout`].
#[cfg(any(test, feature = "test-helpers"))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TermLayout {
    /// Documents carrying the term in this file.
    pub df: u64,
    /// A df=1 term stored inline in the dictionary: no posting bytes.
    pub inline: bool,
    /// A short-form term (`df <= SUB_BLOCK_LEN`): one bodiless block.
    pub short: bool,
    /// Posting blocks (one for a short term, zero for an inline one).
    pub num_blocks: usize,
    /// Whether the term carries a coarse block-max table.
    pub has_coarse: bool,
    /// Long-form blocks by encoding.
    pub packed_blocks: usize,
    pub patched_blocks: usize,
    pub bitset_blocks: usize,
    /// Blocks of consecutive doc ids, stored with no doc payload.
    pub run_blocks: usize,
}

#[cfg(test)]
mod tests {

    use std::ops::Range;

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{
        builder::FtsBuilder,
        posting::{self, BLOCK_LEN, ENCODING_BITSET, ENCODING_PACKED, ENCODING_PATCHED},
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
        // The counting path (count_and_intersect for AND, or_count_unranked for OR)
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

    /// A union over many terms must equal the planted union exactly under
    /// every entry that walks it: the term-only match, its count, and the
    /// phrase-aware atom walk. Five shapes over one corpus of many doc-id
    /// windows cover each walk either side of its routing:
    ///
    /// * ~2000 single-doc terms plus two rarer shared ones, the shape an
    ///   `IN` list of ids produces: a sparse union, so the match and atom
    ///   walks take the heap, and so does the count, since the rescans of
    ///   every window outnumber its heap steps. The shared terms put
    ///   several cursors on one doc so the dedup is exercised.
    /// * ten mid-frequency terms: still sparse, so the match and atom walks
    ///   take the heap; the count keeps the windowed walk because its
    ///   postings outnumber the window rescans.
    /// * eight common terms in ~30% of the docs each: the average doc holds
    ///   more of them than the heap pays for, so the match and atom walks
    ///   keep the rescan, and the count takes its dense bitset.
    /// * exactly `OR_HEAP_MIN_TERMS` terms and one fewer, so the rescan
    ///   walks and the heap walks are checked against the same oracle.
    #[tokio::test]
    async fn many_term_unions_equal_the_planted_union() {
        /// Enough docs for dozens of doc-id windows, so the count's
        /// windowed walk pays a rescan per window rather than one in all.
        const N_DOCS: u32 = 200_000;
        /// Every this-many-th doc carries its own single-doc term.
        const OWN_EVERY: u32 = 100;
        /// Strides of the two shared terms, rarer than the own terms so the
        /// count's gate still sees a sparse union.
        const SHARED_EVERY: u32 = 997;
        const PAIR_EVERY: u32 = 1_009;
        /// Strides of the ten mid-frequency terms (a few hundred docs each).
        const MID_STRIDES: Range<u32> = 500..510;
        /// The common terms: `d{k}` sits on the docs whose `(i + k)` falls
        /// in the first `DENSE_ON` of every `DENSE_PERIOD`, ~30% each.
        const DENSE_TERMS: Range<u32> = 0..OR_HEAP_MIN_TERMS as u32;
        const DENSE_PERIOD: u32 = 10;
        const DENSE_ON: u32 = 3;
        let dense_has = |i: u32, k: u32| (i + k) % DENSE_PERIOD < DENSE_ON;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            if i.is_multiple_of(OWN_EVERY) {
                text.push_str(&format!("u{i}"));
            }
            if i.is_multiple_of(SHARED_EVERY) {
                text.push_str(" shared");
            }
            if i.is_multiple_of(PAIR_EVERY) {
                text.push_str(" pair");
            }
            for k in MID_STRIDES {
                if i.is_multiple_of(k) {
                    text.push_str(&format!(" m{k}"));
                }
            }
            for k in DENSE_TERMS {
                if dense_has(i, k) {
                    text.push_str(&format!(" d{k}"));
                }
            }
            if text.is_empty() {
                text.push_str("filler");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let mut owned: Vec<String> = (0..N_DOCS)
            .step_by(OWN_EVERY as usize)
            .map(|i| format!("u{i}"))
            .collect();
        owned.push("shared".into());
        owned.push("pair".into());
        owned.push("zzz_absent".into());
        let sparse: Vec<&str> = owned.iter().map(String::as_str).collect();
        let sparse_oracle: Vec<u32> = (0..N_DOCS)
            .filter(|i| {
                i.is_multiple_of(OWN_EVERY)
                    || i.is_multiple_of(SHARED_EVERY)
                    || i.is_multiple_of(PAIR_EVERY)
            })
            .collect();
        let mid_owned: Vec<String> = MID_STRIDES.map(|k| format!("m{k}")).collect();
        let mid: Vec<&str> = mid_owned.iter().map(String::as_str).collect();
        let mid_oracle: Vec<u32> = (0..N_DOCS)
            .filter(|i| MID_STRIDES.into_iter().any(|k| i.is_multiple_of(k)))
            .collect();
        let dense_owned: Vec<String> = DENSE_TERMS.map(|k| format!("d{k}")).collect();
        let dense: Vec<&str> = dense_owned.iter().map(String::as_str).collect();
        let dense_oracle: Vec<u32> = (0..N_DOCS)
            .filter(|&i| DENSE_TERMS.into_iter().any(|k| dense_has(i, k)))
            .collect();
        let own_oracle = |shape: &[&str]| -> Vec<u32> {
            (0..N_DOCS)
                .filter(|i| shape.contains(&format!("u{i}").as_str()))
                .collect()
        };
        let shapes: Vec<(&str, Vec<&str>, Vec<u32>)> = vec![
            ("many sparse and shared", sparse.clone(), sparse_oracle),
            ("ten mid-frequency", mid, mid_oracle),
            ("eight common", dense, dense_oracle),
            (
                "on the threshold",
                sparse[..OR_HEAP_MIN_TERMS].to_vec(),
                own_oracle(&sparse[..OR_HEAP_MIN_TERMS]),
            ),
            (
                "under the threshold",
                sparse[..OR_HEAP_MIN_TERMS - 1].to_vec(),
                own_oracle(&sparse[..OR_HEAP_MIN_TERMS - 1]),
            ),
        ];
        for (label, shape, oracle) in shapes {
            assert!(!oracle.is_empty(), "{label}: oracle must not be empty");
            let matched: Vec<u32> = r
                .token_match("body", &shape, BoolMode::Or)
                .await
                .expect("token_match")
                .0
                .into_iter()
                .map(|d| d.get())
                .collect();
            assert_eq!(matched, oracle, "token_match, {label}");
            let count = r
                .token_match_count("body", &shape, BoolMode::Or)
                .await
                .expect("token_match_count")
                .0;
            assert_eq!(count, oracle.len() as u64, "token_match_count, {label}");
            let atoms: Vec<u32> = r
                .atoms_match_ids("body", &shape, &[], BoolMode::Or)
                .await
                .expect("atoms_match_ids")
                .0
                .into_iter()
                .map(|d| d.get())
                .collect();
            assert_eq!(atoms, oracle, "atoms_match_ids, {label}");
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        // A dense corpus stores common terms as bitset blocks. The
        // intersection count must agree with `token_match`'s flat-merge AND
        // length across both count intersection kernels: the bitset-AND
        // (word-parallel presence AND, when every term is dense enough to
        // trip the density gate) and the rarest-driven membership walk (when
        // a sparse term keeps the intersection below the gate). `token_match`
        // AND collects via the decode-based flat-merge, an independent
        // reference from either count kernel.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        const RARE_STRIDE: u32 = 371; // sparse ⇒ below the density gate
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        // A term held by every early doc (a RUN block), then by most docs with
        // gaps (BITSET blocks), then sparsely (strided ⇒ PACKED blocks) has
        // all three block encodings in one posting list. The per-block encoding dispatch in the union
        // (`or_cursor_into_bitset`), the membership probe (`TermCursor::contains`
        // advancing across a bitset→packed boundary), and the doc-id decode must
        // each pick the right branch block by block. Uniform-stride corpora
        // never produce this, so drive it explicitly and cross-check every count
        // kernel against `token_match`'s independent flat-merge length.
        const RUN_END: u32 = BLOCK_LEN as u32; // the first block's docs all hold `mix` → RUN
        const DENSE_END: u32 = 4 * BLOCK_LEN as u32; // then all but every 11th → BITSET
        const SPARSE_STRIDE: u32 = 30; // docs after that every 30th → PACKED
        const N_DOCS: u32 = 4200;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("common "); // every doc → dense partner
            if i % 2 == 0 {
                text.push_str("even "); // every other doc → dense, non-dominant
            }
            let mix = i < RUN_END
                || (i < DENSE_END && i % 11 != 5)
                || (i >= DENSE_END && (i - DENSE_END).is_multiple_of(SPARSE_STRIDE));
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Prove `mix` really has both encodings — else the test silently checks
        // nothing about the transition.
        let cursors = r
            .build_term_cursors(
                0,
                &["mix"],
                QueryIdf::ThisFile,
                CursorUse::Count,
                None,
                None,
            )
            .await
            .expect("build cursors");
        let mix = &cursors[0];
        let (mut saw_bitset, mut saw_packed, mut saw_run) = (false, false, false);
        for blk in mix.blocks.iter() {
            match posting::block_encoding(
                &mix.bytes.as_ref()[blk.block_byte_offset..blk.block_byte_end],
            ) {
                ENCODING_BITSET => saw_bitset = true,
                ENCODING_RUN => saw_run = true,
                ENCODING_PACKED | ENCODING_PATCHED => saw_packed = true,
                other => panic!("unexpected encoding byte {other}"),
            }
        }
        assert!(
            saw_bitset && saw_packed && saw_run,
            "`mix` must carry RUN, BITSET and PACKED blocks \
             (run={saw_run}, bitset={saw_bitset}, packed={saw_packed})"
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
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
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
        // When one term's df dwarfs the rest, the OR count of three or more
        // terms takes the df-anchored path (`df(dominant) + |others \
        // dominant|`) instead of walking the dominant list, unless the terms
        // together are dense enough to fold into one bitset first. Both
        // corpora below must give the identical cardinality to the
        // independent naive ascending merge (`token_match` length).
        //
        // The dense corpus (`common` in nearly every doc) takes the bitset
        // fold. The sparse one (`common` in one doc in twenty, under the
        // fold's density) takes the anchored path, through both of its
        // window shapes: windows with a few rare docs probe the anchor per
        // doc, and the window holding the `burst` run folds it. Gaps where
        // `common` is absent but a rarer term is present exercise the "doc
        // not in the anchor" branch, not just `df(dominant)`.
        const RARE_STRIDE: u32 = 500;
        const RAREB_STRIDE: u32 = 800;
        const HOLE_STRIDE: u32 = 300; // dense corpus: docs missing `common`
        const SPARSE_COMMON_STRIDE: u32 = 20;
        const BURST_DOCS: u32 = 70; // past the anchored walk's fold cutoff
        let corpora: [(u32, bool); 2] = [(OR_WINDOW * 2 + 137, true), (OR_WINDOW * 8, false)];
        for (n_docs, dense) in corpora {
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..n_docs {
                let mut text = String::new();
                let common = match dense {
                    true => !i.is_multiple_of(HOLE_STRIDE),
                    false => i.is_multiple_of(SPARSE_COMMON_STRIDE),
                };
                if common {
                    text.push_str("common ");
                }
                if i.is_multiple_of(RARE_STRIDE) {
                    text.push_str("rare ");
                }
                if i.is_multiple_of(RAREB_STRIDE) {
                    text.push_str("rareb ");
                }
                if (OR_WINDOW + 1..OR_WINDOW + 1 + BURST_DOCS).contains(&i) {
                    text.push_str("burst ");
                }
                // Guarantee no empty doc.
                if text.is_empty() {
                    text.push_str("filler ");
                }
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(blob, json).expect("open");

            let shapes: &[&[&str]] = &[
                &["common", "rare"],
                &["common", "rare", "rareb"],
                &["common", "rare", "burst"],
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
                    "OR count vs merge len for {terms:?}, dense = {dense}"
                );
            }
        }
    }

    #[tokio::test]
    async fn term_df_reports_document_frequency() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // common → df 3 (PFOR header read), rust → df 2 (PFOR),
        // uniqzero → df 1 (inline dictionary entry), absent → 0.
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
        // Interleave the dictionary entry kinds — PFOR (df>1), absent, inline
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

    /// A column no row in this superfile filled is walked without
    /// computing an idf for it.
    ///
    /// Sparse columns miss whole superfiles — a compaction or a repair
    /// The head the term-index walk reports for a term is the engine's own
    /// top-m for that term, with the first document left out as the tail:
    /// the same documents, scores and order the ranked search gives.
    #[tokio::test]
    async fn term_heads_match_the_single_term_ranking() {
        const N_DOCS: u32 = 3_000;
        const DEPTH: usize = 7;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            // A common term in every doc with a varying length, a mid term
            // in every third doc, a rare term in two, a hapax in one.
            let mut text = String::from("common");
            for _ in 0..(i % 11) {
                text.push_str(" filler");
            }
            if i % 3 == 0 {
                text.push_str(" mid");
            }
            if i == 70 || i == 1900 {
                text.push_str(" rare");
            }
            if i == 1234 {
                text.push_str(" hapax");
            }
            b.add_doc(0, i, &text).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let dict_bytes = r.dict_bytes_async().await.expect("dict");
        let facts = r
            .term_index_facts_after(&dict_bytes, "body", None, 64, DEPTH)
            .await
            .expect("facts");
        assert_eq!(facts.len(), 5, "common, filler, hapax, mid, rare");
        for (term, fact) in facts {
            let term = std::str::from_utf8(&term).expect("utf8").to_owned();
            let fact = fact.expect("every term has a fact");
            let ranked = r
                .search_with_floor("body", &[&term], DEPTH + 1, BoolMode::Or, f32::NEG_INFINITY)
                .await
                .expect("ranked");
            // The head carries facts; scored with the file's own norms and
            // idf they are the ranking's scores, bit for bit.
            let norms = r.columns[0].dl_norm_k1();
            let idf = bm25::idf(u64::from(N_DOCS), fact.df);
            let got: Vec<(u32, f32)> = fact
                .head
                .docs
                .iter()
                .map(|&(row, tf, bucket)| {
                    assert_eq!(bucket, norms.buckets()[row as usize], "{term:?} row {row}");
                    (row, bm25::score_with_dl_norm_k1(idf, tf, norms.get(row)))
                })
                .collect();
            let want: Vec<(u32, f32)> = ranked
                .iter()
                .take(DEPTH)
                .map(|(row, s)| (row.get(), *s))
                .collect();
            assert_eq!(got, want, "head of {term:?} is the top-{DEPTH}");
            let df = fact.df as usize;
            assert_eq!(
                fact.head.tail == 0.0,
                df <= DEPTH,
                "{term:?}: df {df} leaves a tail only past the depth"
            );
            // The tail is the key of the first document left out: its score
            // over the idf, which the ranking only knows as a product.
            let want_tail = match ranked.get(DEPTH) {
                Some((_, s)) => *s,
                None => 0.0,
            };
            let tail_score = fact.head.tail * idf;
            assert!(
                (tail_score - want_tail).abs() <= want_tail.abs() * 1e-5,
                "tail of {term:?} is the first document left out: {tail_score} vs {want_tail}"
            );
            // The bound is the largest key of the term's documents: at
            // least every head document's, and the tail's.
            let head_max_key = fact
                .head
                .docs
                .iter()
                .map(|&(row, tf, _)| bm25::tf_factor(tf, norms.get(row)))
                .fold(0.0f32, f32::max);
            assert_eq!(
                fact.bound, head_max_key,
                "{term:?}: the bound is the best key"
            );
            assert!(
                fact.head.tail <= fact.bound,
                "{term:?}: the tail is under the bound"
            );
        }
    }

    /// that lands such a shard would otherwise ask for `df = 1` against
    /// `n_docs = 0`, which is not a ratio and trips the scorer's
    /// precondition.
    #[tokio::test]
    async fn a_column_with_no_scored_documents_walks_without_scoring() {
        const N_DOCS: u32 = 8;
        let mut b = FtsBuilder::new();
        b.register_column("filled".into(), false).expect("register");
        b.register_column("empty".into(), false).expect("register");
        for i in 0..N_DOCS {
            b.add_doc(0, i, &format!("common u{i}"))
                .expect("add filled");
            // Indexed for row alignment, but every cell is empty.
            b.add_doc(1, i, "").expect("add empty");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"filled","tokenizer":"standard","k1":1.2,"b":0.75},{"name":"empty","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let dict_bytes = r.dict_bytes_async().await.expect("dict");

        let walked = r
            .term_index_facts_after(&dict_bytes, "empty", None, 16, 0)
            .await
            .expect("walking an empty column must not panic");
        assert!(
            walked.is_empty(),
            "an empty column has no terms: {walked:?}"
        );

        // The filled column beside it still reports its terms.
        let filled = r
            .term_index_facts_after(&dict_bytes, "filled", None, 16, 0)
            .await
            .expect("walk");
        assert!(!filled.is_empty(), "the filled column still has terms");
    }

    /// The walk with values gives every term exactly the fact the lookup
    /// path gives it: inline (tf 1 and more), short and long terms, on a
    /// positional and a plain column, across chunk edges.
    #[tokio::test]
    async fn facts_from_the_walk_match_facts_from_lookups() {
        const N_DOCS: u32 = 600;
        const CHUNK: usize = 5;
        let mut b = FtsBuilder::new();
        b.register_column("pos".into(), true).expect("register");
        b.register_column("flat".into(), false).expect("register");
        for i in 0..N_DOCS {
            // `common` is long, `mid` short, `u{i}` inline with tf 1 and
            // `twice{i}` a df=1 term with tf 2. `u{i}` sits past position
            // 1, so its positional slot differs from its tf.
            let mut text = format!("common twice{i} twice{i} u{i}");
            if i.is_multiple_of(7) {
                text.push_str(" mid");
            }
            b.add_doc(0, i, &text).expect("add pos");
            b.add_doc(1, i, &text).expect("add flat");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"pos","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true},{"name":"flat","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let dict_bytes = r.dict_bytes_async().await.expect("dict");
        for column in ["pos", "flat"] {
            let mut walked = Vec::new();
            let mut after: Option<Vec<u8>> = None;
            loop {
                let chunk = r
                    .term_index_facts_after(&dict_bytes, column, after.as_deref(), CHUNK, 0)
                    .await
                    .expect("walk");
                let done = chunk.len() < CHUNK;
                after = chunk.last().map(|(t, _)| t.clone());
                walked.extend(chunk);
                if done {
                    break;
                }
            }
            let all = r.iter_column_terms(column).expect("terms");
            let names: Vec<&str> = all
                .iter()
                .map(|t| std::str::from_utf8(t).expect("utf8"))
                .collect();
            assert_eq!(
                walked.iter().map(|(t, _)| t.clone()).collect::<Vec<_>>(),
                all,
                "{column}: the walk visits every term once, in order"
            );
            let looked_up = r.term_index_facts(column, &names).await.expect("lookups");
            for ((term, walk), lookup) in walked.iter().zip(looked_up) {
                let (walk, lookup) = (walk.clone().expect("walk fact"), lookup.expect("fact"));
                let name = String::from_utf8_lossy(term);
                assert_eq!(walk.df, lookup.df, "{column}/{name}: df");
                assert_eq!(
                    walk.bound.to_bits(),
                    lookup.bound.to_bits(),
                    "{column}/{name}: bound"
                );
                assert_eq!(walk.entry, lookup.entry, "{column}/{name}: entry");
            }
        }
    }
}
