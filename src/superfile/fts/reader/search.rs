// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Scored FTS search entry points on [`FtsReader`]: the public
//! `search` / `search_with_floor` / pretokenized and doc-id-ranged
//! variants, the multi-column `search_multi`, the single-term
//! BlockMaxWAND fast path, and the atom (phrase-aware) scored walk.
//! Its own `impl FtsReader` block, split from the reader `core`.

use std::collections::{BinaryHeap, HashMap};

use super::{
    core::*,
    cursor::{TermCursor, TermMeta},
    filter::{AtomExcludeFilter, ExcludeFilter},
    options::BoolMode,
    phrase::AnyCursor,
    sink::{TopKEntry, and_heap_push, drain_top_k_desc},
    work::{
        MatchWork, atom_cursor_bytes, atom_planned_ranges, term_cursor_bytes, term_cursor_ranges,
    },
};
use crate::{
    runtime_metrics::{
        cpu::{thread_cpu_delta_ns, thread_cpu_ns},
        op_stats::{metering_active, timed_section},
    },
    superfile::{
        ReadError,
        error::FtsError,
        fts::{
            bm25,
            dict::{DictReader, make_key},
            fst_value::FstValue,
            posting::{BLOCK_LEN, decode_block},
        },
    },
};

impl FtsReader {
    /// Ranked search over heterogeneous atoms — the walk every
    /// phrase-bearing query takes. With musts, the match set is their
    /// intersection and shoulds are scoring-only (the clause model);
    /// with none, the shoulds' union matches. Docs excluded by
    /// `filter` never reach the heap; docs scoring strictly below
    /// `floor_eff` are dropped at admission.
    fn run_atoms_search(
        &self,
        column_id: u32,
        mut musts: Vec<AnyCursor>,
        mut shoulds: Vec<AnyCursor>,
        k: usize,
        mut filter: Option<AtomExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let dl_norm_k1 = &self.columns[column_id as usize].dl_norm_k1;
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);

        // Per-atom pruning slack: an atom only needs to contribute
        // more than the walk's bar minus what every *other* atom could
        // possibly add. Phrase atoms use it to skip position work on
        // docs that provably can't matter (`skip_to_pruned`).
        let atom_slack = |atoms: &[AnyCursor], extra_ub: f32| -> Vec<f32> {
            let total: f32 = atoms.iter().map(AnyCursor::term_max_bm25).sum();
            atoms
                .iter()
                .map(|a| total - a.term_max_bm25() + extra_ub)
                .collect()
        };

        if musts.is_empty() {
            // Union of shoulds, doc-at-a-time: score every atom
            // sitting on the frontier doc, then advance them past it.
            let others_ub = atom_slack(&shoulds, 0.0);
            while let Some(doc) = shoulds
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
                    let norm = dl_norm_k1.get(doc);
                    let score: f32 = shoulds
                        .iter()
                        .filter(|a| !a.is_exhausted() && a.current_doc_id() == doc)
                        .map(|a| a.score_current(norm))
                        .sum();
                    if score > floor_eff {
                        and_heap_push(&mut heap, k, None, score, doc);
                    }
                }
                let Some(next) = doc.checked_add(1) else {
                    break;
                };
                let bar = match heap.len() >= k {
                    true => heap.peek().expect("heap len == k").0.max(floor_eff),
                    false => floor_eff,
                };
                for (a, &others) in shoulds.iter_mut().zip(&others_ub) {
                    if !a.is_exhausted() && a.current_doc_id() == doc {
                        a.skip_to_pruned(next, bar - others, dl_norm_k1)?;
                    }
                }
            }
            return Ok(drain_top_k_desc(heap));
        }

        // Must-driven walk: leapfrog the musts to each common doc,
        // score musts + landing shoulds there.
        let should_ub: f32 = shoulds.iter().map(AnyCursor::term_max_bm25).sum();
        let must_others_ub = atom_slack(&musts, should_ub);
        let should_others_ub: Vec<f32> = {
            let must_ub_total: f32 = musts.iter().map(AnyCursor::term_max_bm25).sum();
            atom_slack(&shoulds, must_ub_total)
        };
        let mut target = 0u32;
        'docs: loop {
            let bar = match heap.len() >= k {
                true => heap.peek().expect("heap len == k").0.max(floor_eff),
                false => floor_eff,
            };
            let mut aligned = target;
            let mut i = 0usize;
            while i < musts.len() {
                let a = &mut musts[i];
                a.skip_to_pruned(aligned, bar - must_others_ub[i], dl_norm_k1)?;
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
            // Bar skip: the kth-best (or the seeded floor) minus the
            // most the shoulds could add bounds what the musts must
            // reach; a candidate whose must-side block bounds can't
            // get there is dead without scoring (and, for phrase
            // shoulds, without any position work). `>=`, not `>`: a
            // doc exactly at the bar can still displace the incumbent
            // kth-best on the ascending-doc-id tie-break.
            let scoring_needed = match bar > f32::NEG_INFINITY {
                true => {
                    let must_ub: f32 = musts
                        .iter_mut()
                        .map(|a| a.block_max_in_range(aligned, aligned))
                        .sum();
                    must_ub + should_ub >= bar
                }
                false => true,
            };
            let admitted = scoring_needed
                && match filter.as_mut() {
                    Some(f) => f.admits(aligned)?,
                    None => true,
                };
            if admitted {
                let norm = dl_norm_k1.get(aligned);
                let mut score: f32 = musts.iter().map(|a| a.score_current(norm)).sum();
                for (sh, &others) in shoulds.iter_mut().zip(&should_others_ub) {
                    sh.skip_to_pruned(aligned, bar - others, dl_norm_k1)?;
                    if !sh.is_exhausted() && sh.current_doc_id() == aligned {
                        score += sh.score_current(norm);
                    }
                }
                if score > floor_eff {
                    and_heap_push(&mut heap, k, None, score, aligned);
                }
            }
            let Some(next) = aligned.checked_add(1) else {
                break;
            };
            target = next;
        }
        Ok(drain_top_k_desc(heap))
    }

    /// Single-column BM25 search.
    ///
    /// `terms` are the *already-tokenized* query terms — caller-tokenized
    /// to match the column's tokenizer. The format currently uses one
    /// tokenizer for all columns, so callers can use the same tokenizer
    /// that was used for indexing.
    pub async fn search(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        self.search_with_floor(column, terms, k, mode, f32::NEG_INFINITY)
            .await
    }

    /// [`Self::search`] with an externally-supplied **score floor**:
    /// docs scoring **strictly below** `floor` can never appear in the
    /// caller's final result (e.g. a cross-segment top-k already holds
    /// k hits at or above it), so every pruning structure — BMW block
    /// skips, the MaxScore essential boundary, heap admission — starts
    /// from the floor instead of from empty. Docs scoring **equal to**
    /// `floor` are still returned (tie candidates survive), which keeps
    /// the caller's merged result identical to an unfloored run.
    /// `f32::NEG_INFINITY` disables the floor.
    pub async fn search_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // A flat term list under one mode is the degenerate clause
        // shape: `And` makes every term a must, `Or` a should.
        // `prepare_clauses` resolves the column and, on the `<= threshold`
        // pruning comparisons every kernel uses, seeds them with the
        // largest f32 strictly below `floor` ("strictly below floor is
        // dead, equal-to-floor survives") via `floor.next_down()`.
        let (musts, shoulds): (&[&str], &[&str]) = match mode {
            BoolMode::And => (terms, &[]),
            BoolMode::Or => (&[], terms),
        };
        let prep = self
            .prepare_clauses(
                column,
                ClauseLists {
                    musts,
                    shoulds,
                    ..ClauseLists::default()
                },
                k,
                floor,
            )
            .await?;
        self.run_prepared(prep)
    }

    /// [`Self::search`] that also returns the walk's work — posting
    /// bytes, planned ranges, and the bracketed kernel on-CPU ns
    /// (`prepare_clauses`' inline walks plus the `run_prepared`
    /// section), all carried on the one [`MatchWork`]. Prefix search
    /// reports through this so an expansion to thousands of terms
    /// carries its cost like any other query shape.
    pub(crate) async fn search_with_work(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        mode: BoolMode,
    ) -> Result<(Vec<(u32, f32)>, MatchWork), FtsError> {
        let (musts, shoulds): (&[&str], &[&str]) = match mode {
            BoolMode::And => (terms, &[]),
            BoolMode::Or => (&[], terms),
        };
        let prep = self
            .prepare_clauses(
                column,
                ClauseLists {
                    musts,
                    shoulds,
                    ..ClauseLists::default()
                },
                k,
                f32::NEG_INFINITY,
            )
            .await?;
        let mut work = MatchWork {
            postings_bytes: prep.postings_bytes(),
            planned_ranges: prep.planned_ranges(),
            kernel_cpu_ns: prep.inline_kernel_cpu_ns(),
        };
        let hits = match prep {
            PreparedClauses::Done { hits, .. } => hits,
            prep => {
                let (hits, run_ns) = timed_section(|| self.run_prepared(prep));
                work.kernel_cpu_ns += run_ns;
                hits?
            }
        };
        Ok((hits, work))
    }

    /// BM25 search over explicit clause lists, with negated terms
    /// excluded.
    ///
    /// `musts` all have to match (their intersection is the match
    /// set); `shoulds` are scoring-only — a matching should raises a
    /// doc's score but never adds or removes a match. With no musts,
    /// the shoulds' union is the match set (a plain OR query).
    /// `negatives` filter out any doc containing one of them,
    /// regardless of score. All lists are already tokenized; the
    /// default-operator resolution (bare token → must or should)
    /// happened at parse time via `ParsedQuery::into_clauses`.
    ///
    /// No musts and no shoulds → [`FtsError::NegationOnly`] (nothing
    /// to rank) when negatives exist, else an empty result.
    pub(crate) async fn search_excluding(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let prep = self.prepare_clauses(column, lists, k, floor).await?;
        self.run_prepared(prep)
    }

    /// I/O half of an un-ranged clause search: resolve the column,
    /// classify the query shape, and fetch every cursor
    /// [`Self::run_prepared`] needs to score. The single-atom shape
    /// finishes here since it's cheap; the phrase-atom shape also
    /// finishes here, but only because it isn't wired to the reader
    /// pool yet, not because it's cheap.
    pub(crate) async fn prepare_clauses(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
        floor: f32,
    ) -> Result<PreparedClauses, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if k == 0 {
            return Ok(PreparedClauses::Done {
                hits: Vec::new(),
                postings_bytes: 0,
                planned_ranges: 0,
                kernel_cpu_ns: 0,
            });
        }
        if lists.no_positive_atoms() {
            if lists.no_negative_atoms() {
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    postings_bytes: 0,
                    planned_ranges: 0,
                    kernel_cpu_ns: 0,
                });
            }
            return Err(FtsError::NegationOnly);
        }
        let floor_eff = floor.next_down();

        if lists.has_phrases() {
            // Phrase-bearing query: the heterogeneous atom walks.
            let (must_atoms, must_dict) = self
                .build_atom_cursors(column_id, lists.musts, lists.must_phrases, lists.global_idf)
                .await?;
            if must_atoms.iter().any(Option::is_none) {
                // A must atom can never match in this superfile. The
                // atoms that DID build still cost their bytes.
                let built: Vec<AnyCursor> = must_atoms.into_iter().flatten().collect();
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    postings_bytes: atom_cursor_bytes(&built),
                    planned_ranges: atom_planned_ranges(&built) + must_dict,
                    kernel_cpu_ns: 0,
                });
            }
            let must_atoms: Vec<AnyCursor> = must_atoms.into_iter().flatten().collect();
            let (should_built, should_dict) = self
                .build_atom_cursors(
                    column_id,
                    lists.shoulds,
                    lists.should_phrases,
                    lists.global_idf,
                )
                .await?;
            let should_atoms: Vec<AnyCursor> = should_built.into_iter().flatten().collect();
            // Negatives are a hard exclusion filter, not scored, so their
            // idf is irrelevant — always build them local.
            let (negative_built, negative_dict) = self
                .build_atom_cursors(column_id, lists.negatives, lists.negative_phrases, None)
                .await?;
            let negative_atoms: Vec<AnyCursor> = negative_built.into_iter().flatten().collect();
            let postings_bytes = atom_cursor_bytes(&must_atoms)
                + atom_cursor_bytes(&should_atoms)
                + atom_cursor_bytes(&negative_atoms);
            let planned_ranges = atom_planned_ranges(&must_atoms)
                + atom_planned_ranges(&should_atoms)
                + atom_planned_ranges(&negative_atoms)
                + must_dict
                + should_dict
                + negative_dict;
            let filter = match negative_atoms.is_empty() {
                true => None,
                false => Some(AtomExcludeFilter::new(negative_atoms)),
            };
            // The atom walk is the whole kernel for phrase shapes —
            // `run_prepared` sees only the finished `Done` — so bracket
            // its on-CPU time here (sync section, no awaits inside).
            // Gated: an unmetered process must not pay the procfs reads.
            let kernel_start = metering_active().then(thread_cpu_ns).flatten();
            let result =
                self.run_atoms_search(column_id, must_atoms, should_atoms, k, filter, floor_eff)?;
            return Ok(PreparedClauses::Done {
                hits: result,
                postings_bytes,
                planned_ranges,
                kernel_cpu_ns: thread_cpu_delta_ns(kernel_start),
            });
        }

        let neg_filter = match lists.negatives {
            [] => None,
            // Negatives are a hard exclusion filter, not scored, so their
            // idf is irrelevant — always build them with local stats.
            _ => Some(ExcludeFilter::new(
                self.build_term_cursors(column_id, lists.negatives, None, false)
                    .await?,
            )),
        };
        // FST-dictionary ranges the builds below request — one per
        // `build_term_cursors` call (the dictionary fetch is a real
        // byte-source range on every query, warm or cold).
        let mut dict_ranges = u64::from(neg_filter.is_some());

        // Single-atom fast path: BlockMaxWAND-driven block skipping.
        // One term scores identically whichever clause list it sits
        // in (a lone must and a lone should both rank that term's
        // postings), so both shapes take it. Skipped under global stats
        // — the bespoke single-term BMW does not take an idf override,
        // so route a lone term through the general cursor path (which
        // does) instead; correctness over the single-term micro-opt.
        if lists.global_idf.is_none() && lists.musts.len() + lists.shoulds.len() == 1 {
            let term = lists
                .musts
                .iter()
                .chain(lists.shoulds)
                .next()
                .expect("one atom");
            let mut filter = neg_filter;
            let filter_postings_bytes = filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
            let filter_ranges = filter.as_ref().map_or(0, ExcludeFilter::planned_ranges);
            let (result, term_work, kernel_cpu_ns) = self
                .search_single_term_bmw(column_id, term, k, filter.as_mut(), floor_eff)
                .await?;
            // +1: the BMW walk's own dictionary fetch.
            dict_ranges += 1;
            return Ok(PreparedClauses::Done {
                hits: result,
                postings_bytes: term_work.postings_bytes + filter_postings_bytes,
                planned_ranges: term_work.planned_ranges + filter_ranges + dict_ranges,
                kernel_cpu_ns,
            });
        }

        if lists.musts.is_empty() {
            let cursors = self
                .build_term_cursors(column_id, lists.shoulds, lists.global_idf, false)
                .await?;
            dict_ranges += 1;
            if cursors.is_empty() {
                let postings_bytes = neg_filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
                let planned_ranges =
                    neg_filter.as_ref().map_or(0, ExcludeFilter::planned_ranges) + dict_ranges;
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    postings_bytes,
                    planned_ranges,
                    kernel_cpu_ns: 0,
                });
            }
            return Ok(PreparedClauses::Or {
                column_id,
                cursors,
                filter: neg_filter,
                k,
                floor_eff,
                dict_ranges,
            });
        }
        // Build must cursors; if any must is missing, the
        // intersection is empty.
        let must_cursors = self
            .build_term_cursors(column_id, lists.musts, lists.global_idf, false)
            .await?;
        dict_ranges += 1;
        if must_cursors.len() != lists.musts.len() {
            let postings_bytes = term_cursor_bytes(&must_cursors)
                + neg_filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
            let planned_ranges = term_cursor_ranges(&must_cursors)
                + neg_filter.as_ref().map_or(0, ExcludeFilter::planned_ranges)
                + dict_ranges;
            return Ok(PreparedClauses::Done {
                hits: Vec::new(),
                postings_bytes,
                planned_ranges,
                kernel_cpu_ns: 0,
            });
        }
        if lists.shoulds.is_empty() {
            return Ok(PreparedClauses::Must {
                column_id,
                must_cursors,
                filter: neg_filter,
                k,
                floor_eff,
                dict_ranges,
            });
        }
        // Shoulds absent from this superfile contribute nothing;
        // when none survive, the walk is a plain must intersection.
        let should_cursors = self
            .build_term_cursors(column_id, lists.shoulds, lists.global_idf, false)
            .await?;
        dict_ranges += 1;
        if should_cursors.is_empty() {
            return Ok(PreparedClauses::Must {
                column_id,
                must_cursors,
                filter: neg_filter,
                k,
                floor_eff,
                dict_ranges,
            });
        }
        Ok(PreparedClauses::MustShould {
            column_id,
            must_cursors,
            should_cursors,
            filter: neg_filter,
            k,
            floor_eff,
            dict_ranges,
        })
    }

    /// CPU half paired with [`Self::prepare_clauses`] — scores the
    /// cursors it fetched. No I/O, so it can run on the reader pool.
    pub(crate) fn run_prepared(&self, prep: PreparedClauses) -> Result<Vec<(u32, f32)>, FtsError> {
        match prep {
            PreparedClauses::Done { hits, .. } => Ok(hits),
            PreparedClauses::Must {
                column_id,
                must_cursors,
                mut filter,
                dict_ranges: _,
                k,
                floor_eff,
            } => self.run_and_intersect(column_id, must_cursors, k, filter.as_mut(), floor_eff),
            PreparedClauses::MustShould {
                column_id,
                must_cursors,
                should_cursors,
                mut filter,
                k,
                floor_eff,
                dict_ranges: _,
            } => self.run_must_should(
                column_id,
                must_cursors,
                should_cursors,
                k,
                filter.as_mut(),
                floor_eff,
            ),
            PreparedClauses::Or {
                column_id,
                cursors,
                mut filter,
                k,
                floor_eff,
                dict_ranges: _,
            } => self.dispatch_or_algo(column_id, cursors, k, filter.as_mut(), floor_eff),
        }
    }

    /// Multi-term OR BM25 search constrained to a doc_id sub-range.
    ///
    /// Same scoring semantics as [`Self::search`] in `BoolMode::Or`
    /// for the multi-term case, but only docs whose id falls within
    /// `[doc_id_start, doc_id_end)` are eligible. Used by the
    /// supertable's intra-superfile parallel fan-out: when the reader
    /// pool has more threads than superfiles, each superfile is sliced
    /// into N equal-width doc-id sub-ranges and one task per
    /// sub-range runs here in parallel; the caller merges the
    /// per-sub-range top-K heaps.
    ///
    /// Returns `Ok(Vec::new())` for `terms.is_empty()`, `k == 0`, or
    /// a degenerate range (`doc_id_start >= doc_id_end`).
    ///
    /// Single-term inputs (`terms.len() == 1`) are NOT
    /// sub-range-optimized here — single-term queries already
    /// complete in microseconds via [`Self::search`]'s BMW path; the
    /// supertable layer should keep them on the un-ranged call. The
    /// implementation delegates to
    /// [`Self::run_max_score_bmm_range`] which seeks every cursor
    /// to `doc_id_start` and breaks the outer loop when the next
    /// candidate doc_id reaches `doc_id_end`.
    pub async fn search_or_range_pretokenized(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        self.search_or_range_pretokenized_with_floor(
            column,
            terms,
            k,
            doc_id_start,
            doc_id_end,
            f32::NEG_INFINITY,
            None,
        )
        .await
    }

    /// [`Self::search_or_range_pretokenized`] with a score floor — see
    /// [`Self::search_with_floor`] for the floor contract.
    pub async fn search_or_range_pretokenized_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
        global_idf: Option<&GlobalTermIdf>,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let set = self.build_or_cursor_set(column, terms, global_idf).await?;
        self.search_or_range_prebuilt(&set, k, doc_id_start, doc_id_end, floor)
    }

    /// Build the OR cursors for `terms` once — the postings fetch and
    /// skip-table parse — for reuse across doc-id sub-ranges via
    /// [`Self::search_or_range_prebuilt`]. An intra-superfile fan-out
    /// that builds per slice re-fetches every term's full posting bytes
    /// and re-parses its skip table per slice (measured at 1M as 2.5x
    /// cold bytes when slicing widened); clones of these cursors share
    /// `bytes` and the `Arc` skip table instead.
    ///
    /// `global_idf` is baked into the cursors here (see
    /// [`Self::build_term_cursors`]), so every sub-range sharing a set
    /// must want the same override — it does: one gather per query.
    pub(crate) async fn build_or_cursor_set(
        &self,
        column: &str,
        terms: &[&str],
        global_idf: Option<&GlobalTermIdf>,
    ) -> Result<OrCursorSet, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let cursors = if terms.is_empty() {
            Vec::new()
        } else {
            self.build_term_cursors(column_id, terms, global_idf, false)
                .await?
        };
        Ok(OrCursorSet { column_id, cursors })
    }

    /// Multi-term OR over `[doc_id_start, doc_id_end)` against prebuilt
    /// cursors — the ranged fan-out's per-slice call;
    /// [`Self::search_or_range_pretokenized_with_floor`] delegates here.
    /// The ranged path carries no negation in v1.
    ///
    /// Kernel choice goes through the same `route_or_to_windowed` seam as the
    /// single-shot path, so a query runs the same kernel whether or not the
    /// fan-out sliced it — hardcoding BMM here once caused an 11-24x
    /// post-compact broad-OR regression.
    pub(crate) fn search_or_range_prebuilt(
        &self,
        set: &OrCursorSet,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        if set.cursors.is_empty() || k == 0 || doc_id_start >= doc_id_end {
            return Ok(Vec::new());
        }
        let cursors = set.cursors.clone();
        if route_or_to_windowed(&cursors, k) {
            self.run_windowed_union(
                set.column_id,
                cursors,
                k,
                None,
                floor.next_down(),
                doc_id_start,
                doc_id_end,
            )
        } else {
            self.run_max_score_bmm_range(
                set.column_id,
                cursors,
                k,
                doc_id_start,
                doc_id_end,
                None,
                floor.next_down(),
            )
        }
    }

    /// Multi-column BM25 search (most_fields semantics): each
    /// `(column, weight)` runs an OR-mode search; per-column scores are
    /// multiplied by `weight` and summed across columns.
    pub async fn search_multi(
        &self,
        columns: &[(&str, f32)],
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // Tokenize the query with each column's configured tokenizer so
        // per-column analyzers are honored — a table may index different
        // columns with different analyzers.
        let mut combined: HashMap<u32, f32> = HashMap::new();
        for (col_name, weight) in columns {
            let col_id = self.resolve_column_id(col_name)?;
            let tok = &self.columns[col_id as usize].tokenizer;
            let term_strings: Vec<String> = tok.tokenize(query).collect();
            let term_refs: Vec<&str> = term_strings.iter().map(|s| s.as_str()).collect();
            let per_col = self.search(col_name, &term_refs, usize::MAX, mode).await?;
            for (doc_id, s) in per_col {
                *combined.entry(doc_id).or_insert(0.0) += s * weight;
            }
        }
        Ok(top_k(combined, k))
    }

    /// Single-term BM25 search with BlockMaxWAND-driven block skipping.
    ///
    /// Reads the per-(col, term) metadata + skip table, then iterates
    /// blocks in order. Maintains a top-k min-heap of `(score, doc_id)`.
    /// Once the heap is full (`heap.len() == k`), subsequent blocks
    /// whose skip-table `max_bm25` can't beat the heap's current
    /// minimum (= the current kth-best score) are skipped without
    /// decoding. Both the block bytes and the per-doc score loop are
    /// avoided.
    ///
    /// For uniform-dense lists where every block has similar
    /// `max_bm25`, BMW provides zero benefit. Its win shows up on
    /// posting lists with high score variance — e.g. very long lists
    /// where most blocks contain mid-relevance docs and the top-k is
    /// dominated by a few outliers.
    /// Returns `(hits, posting work, on-CPU ns of the scoring walk)` —
    /// the walk runs inside `prepare_clauses`, so its work and kernel
    /// time must travel with the result (single-term is the most common
    /// query shape; leaving it unbracketed would make `kernel_cpu_ns`
    /// incomparable across clause shapes). The work excludes the
    /// dictionary fetch — the caller counts it once per build.
    async fn search_single_term_bmw(
        &self,
        column_id: u32,
        term: &str,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<(Vec<(u32, f32)>, MatchWork, u64), FtsError> {
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let col_meta = &self.columns[column_id as usize];
        let key = make_key(&col_meta.name, term);
        let Some(packed) = dict.lookup(&key) else {
            return Ok((Vec::new(), MatchWork::default(), 0));
        };
        let (metadata_offset, postings_length) = match FstValue::unpack(packed) {
            FstValue::Inline { doc_id, tf } => {
                // df=1 inline path: no postings-region read, no
                // skip-table, no PFOR decode. The single doc's score
                // is the entire result for any k ≥ 1 (unless it sits
                // strictly below the caller's floor).
                //
                // On a positional column the slot carries the term's
                // single position, tf implied 1 (the builder only
                // inlines tf == 1 there) — score with the implied tf.
                let tf = match col_meta.positions {
                    true => 1,
                    false => tf,
                };
                let idf_t = bm25::idf(self.n_docs as u64, 1);
                let idf_x_k1p1 = idf_t * (bm25::K1 + 1.0);
                // Drop the lone match if a negated term excludes it.
                // The inline slot read no postings-region bytes; the
                // work-stats byte count for this path is genuinely zero.
                if let Some(f) = filter.as_deref_mut()
                    && !f.admits(doc_id)
                {
                    return Ok((Vec::new(), MatchWork::default(), 0));
                }
                let dl_norm_k1 = col_meta.dl_norm_k1.get(doc_id);
                let score = bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1);
                if score <= floor_eff {
                    return Ok((Vec::new(), MatchWork::default(), 0));
                }
                return Ok((vec![(doc_id, score)], MatchWork::default(), 0));
            }
            FstValue::Pfor {
                metadata_offset,
                postings_length_hint,
            } => (
                metadata_offset as usize,
                postings_length_hint.map(|len| len as usize),
            ),
        };
        // Fetch only this term's byte range (metadata header + skip
        // table + blocks). The returned buffer starts at the metadata
        // header, so the region-relative `metadata_offset` rebases to
        // 0 for all indexing below.
        let term_bytes = {
            let mut fetched = self
                .fetch_term_postings(&[(metadata_offset, postings_length)])
                .await?;
            fetched.pop().expect("one fetched range for one PFOR term")
        };
        let postings = term_bytes.as_ref();
        let metadata_offset = 0usize;

        // Everything below is the synchronous scoring walk (no awaits):
        // bracket it on this thread for the per-query kernel CPU stat.
        // Gated: an unmetered process must not pay the procfs reads on
        // the most common query shape.
        let kernel_start = metering_active().then(thread_cpu_ns).flatten();
        let term_meta = TermMeta::parse(postings, metadata_offset, col_meta.positions, false)?;

        let idf_t = bm25::idf(self.n_docs as u64, term_meta.df);
        let idf_x_k1p1 = idf_t * (bm25::K1 + 1.0);
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // Top-k min-heap; see `TopKEntry` for the reversed ordering
        // that makes `peek()` the current kth-best score.
        let mut heap: BinaryHeap<TopKEntry> =
            BinaryHeap::with_capacity(k.min(term_meta.num_blocks * BLOCK_LEN).max(1));
        let mut buf_d = vec![0u32; BLOCK_LEN];
        let mut buf_t = vec![0u32; BLOCK_LEN];

        for i in 0..term_meta.num_blocks {
            // last_doc_id (first tuple slot) is unused here — it serves
            // AND-merge seeks, which single-term never does.
            let (_, block_offset_in_term, block_max_bm25) = term_meta.skip_entry(postings, i);

            // Floor skip: nothing in this block can reach the caller's
            // floor — dead regardless of local heap state.
            if block_max_bm25 <= floor_eff {
                continue;
            }
            // BMW skip: heap full AND this block can't beat the kth-best.
            if heap.len() >= k
                && let Some(TopKEntry(min_score, _)) = heap.peek()
                && block_max_bm25 <= *min_score
            {
                continue;
            }

            // Locate the block's bytes.
            let block_end_in_term = term_meta.block_end_in_term(postings, i);
            let block_bytes = &postings
                [metadata_offset + block_offset_in_term..metadata_offset + block_end_in_term];

            //  Actual number of real docs in that block.
            let n = decode_block(block_bytes, &mut buf_d, &mut buf_t);

            for j in 0..n {
                let doc_id = buf_d[j];
                // Drop docs excluded by a negated term (None = keep all).
                if let Some(f) = filter.as_deref_mut()
                    && !f.admits(doc_id)
                {
                    continue;
                }
                let tf = buf_t[j];
                let score = bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1.get(doc_id));
                // Floor gate: strictly-below-floor docs are dead to the
                // caller; keeping them out also keeps the heap's min
                // (the BMW skip bar) honest.
                if score <= floor_eff {
                    continue;
                }
                if heap.len() < k {
                    heap.push(TopKEntry(score, doc_id));
                } else if let Some(TopKEntry(min_score, _)) = heap.peek()
                    && score > *min_score
                {
                    heap.pop();
                    heap.push(TopKEntry(score, doc_id));
                }
            }
        }

        Ok((
            drain_top_k_desc(heap),
            MatchWork {
                postings_bytes: term_bytes.len() as u64,
                // A hint-less slot costs a header probe before the body
                // fetch — two planned ranges instead of one.
                planned_ranges: 1 + u64::from(postings_length.is_none()),
                // The walk's ns travel in the tuple's third element.
                kernel_cpu_ns: 0,
            },
            thread_cpu_delta_ns(kernel_start),
        ))
    }

    /// Build one `TermCursor` per term that resolves in the FST.
    /// Missing terms (FST miss) are silently dropped — fine for OR
    /// semantics where a missing term contributes nothing. Returned
    /// `Vec` may be empty (all terms missed) or shorter than `terms`.
    pub(super) async fn build_term_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
        global_idf: Option<&GlobalTermIdf>,
        count_only: bool,
    ) -> Result<Vec<TermCursor>, FtsError> {
        let fst_bytes = self.dict_bytes_async().await?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let col_meta = &self.columns[column_id as usize];

        // Resolve each present term to either an inline (df=1) value or
        // a PFOR metadata offset, preserving query order. FST misses
        // are dropped (fine for OR; AND callers length-check). Collect
        // the PFOR offsets so all their byte ranges can be fetched in
        // one parallel fan-out below — never the whole postings region.
        // Each resolved entry carries its term's global idf (when in
        // `Bm25Stats::Global`) so the cursor is built with the global
        // value; `None` per term falls back to this superfile's local idf.
        enum Resolved {
            Inline {
                doc_id: u32,
                tf: u32,
                gidf: Option<f32>,
            },
            Pfor {
                gidf: Option<f32>,
                header_probed: bool,
            },
        }
        let mut resolved: Vec<Resolved> = Vec::with_capacity(terms.len());
        let mut pfor_offsets: Vec<(usize, Option<usize>)> = Vec::new();
        for term in terms {
            let key = make_key(&col_meta.name, term);
            let Some(packed) = dict.lookup(&key) else {
                continue;
            };
            let gidf = global_idf.and_then(|m| m.get(*term).copied());
            match FstValue::unpack(packed) {
                FstValue::Inline { doc_id, tf } => {
                    resolved.push(Resolved::Inline { doc_id, tf, gidf });
                }
                FstValue::Pfor {
                    metadata_offset,
                    postings_length_hint,
                } => {
                    pfor_offsets.push((
                        metadata_offset as usize,
                        postings_length_hint.map(|len| len as usize),
                    ));
                    // A hint-less slot (21-bit length overflow) costs a
                    // header probe BEFORE the body fetch — two planned
                    // ranges, recorded on the cursor for the tallies.
                    resolved.push(Resolved::Pfor {
                        gidf,
                        header_probed: postings_length_hint.is_none(),
                    });
                }
            }
        }

        let pfor_bytes = self.fetch_term_postings(&pfor_offsets).await?;
        let mut pfor_iter = pfor_bytes.into_iter();

        let mut cursors: Vec<TermCursor> = Vec::with_capacity(resolved.len());
        for r in resolved {
            match r {
                Resolved::Inline { doc_id, tf, gidf } => {
                    // On a positional column the inline slot carries
                    // the term's single position, tf implied 1 — the
                    // builder only inlines tf == 1 postings there.
                    // Scoring must use the implied tf, never the slot.
                    // (Phrase members recover the position itself with
                    // their own FST lookup — see `build_atom_cursors`.)
                    let tf = match col_meta.positions {
                        true => 1,
                        false => tf,
                    };
                    let dl_norm_k1 = col_meta.dl_norm_k1.get(doc_id);
                    cursors.push(TermCursor::new_inline(
                        doc_id,
                        tf,
                        self.n_docs as u64,
                        dl_norm_k1,
                        gidf,
                    ));
                }
                Resolved::Pfor {
                    gidf,
                    header_probed,
                } => {
                    let term_bytes = pfor_iter.next().expect("one fetched range per PFOR term");
                    cursors.push(TermCursor::new(
                        term_bytes,
                        self.n_docs as u64,
                        col_meta.positions,
                        gidf,
                        header_probed,
                        count_only,
                    )?);
                }
            }
        }
        Ok(cursors)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use bytes::Bytes;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{builder::FtsBuilder, tokenize::AsciiLowerTokenizer};

    #[tokio::test]
    async fn search_returns_exact_doc_ids_for_known_term() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        // "rust" appears in doc 0 and doc 1.
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0), "doc 0 should match");
        assert!(ids.contains(&1), "doc 1 should match");
        assert!(!ids.contains(&2), "doc 2 should not match");
    }

    #[tokio::test]
    async fn search_missing_term_or_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["nonexistent"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_and_short_circuits_on_missing_term() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust", "nonexistent"], 10, BoolMode::And)
            .await
            .expect("search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_and_intersects_term_postings() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        // "rust AND runtime" — both in doc 0 and doc 1.
        let hits = r
            .search("body", &["rust", "runtime"], 10, BoolMode::And)
            .await
            .expect("search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn search_unknown_column_errors() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let err = r
            .search("title", &["rust"], 10, BoolMode::Or)
            .await
            .expect_err("expected error");
        assert!(matches!(err, FtsError::UnknownColumn(_)));
    }

    #[tokio::test]
    async fn search_empty_terms_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &[], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_zero_k_returns_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 0, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_results_sorted_by_score_desc() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        for w in hits.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores should be descending");
        }
    }

    #[tokio::test]
    async fn search_limits_to_k() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 1, BoolMode::Or)
            .await
            .expect("FTS search");
        assert_eq!(hits.len(), 1);
    }

    #[tokio::test]
    async fn df1_single_term_search_returns_one_doc() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["uniqzero"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert_eq!(hits.len(), 1, "df=1 term should return exactly one hit");
        assert_eq!(hits[0].0, 0, "uniqzero lives in doc 0");
        assert!(hits[0].1 > 0.0, "score must be positive");
    }

    #[tokio::test]
    async fn df1_in_or_query_combines_with_df_ge_2() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["uniqtwo", "rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        // uniqtwo → doc 2; rust → docs 0, 1.
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[tokio::test]
    async fn df1_in_and_query_intersects_correctly() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        // uniqzero ∩ rust = {doc 0}.
        let hits = r
            .search("body", &["uniqzero", "rust"], 10, BoolMode::And)
            .await
            .expect("FTS search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids, vec![0]);
        // uniqzero ∩ uniqtwo = ∅ (different docs).
        let hits = r
            .search("body", &["uniqzero", "uniqtwo"], 10, BoolMode::And)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn df1_missing_term_returns_empty() {
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["nonexistentunique"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_excluding_drops_negated_docs() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // "runtime" hits docs 0 and 1; negate "async" (only in doc 0).
        let hits = r
            .search_excluding(
                "body",
                ClauseLists {
                    shoulds: &["runtime"],
                    negatives: &["async"],
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("search excluding");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(ids, vec![1], "doc 0 excluded by negated 'async'");
    }

    #[tokio::test]
    async fn search_excluding_negation_only_errors() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let err = r
            .search_excluding(
                "body",
                ClauseLists {
                    negatives: &["rust"],
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect_err("negation-only");
        assert!(matches!(err, FtsError::NegationOnly));
    }

    #[tokio::test]
    async fn search_excluding_no_terms_at_all_is_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let hits = r
            .search_excluding("body", ClauseLists::default(), 10, f32::NEG_INFINITY)
            .await
            .expect("empty");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn search_with_floor_prunes_below_floor() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // An impossibly high floor prunes every doc.
        let hits = r
            .search_with_floor("body", &["rust"], 10, BoolMode::Or, 1e9)
            .await
            .expect("floored search");
        assert!(hits.is_empty(), "floor above all scores prunes everything");
    }

    #[tokio::test]
    async fn search_multi_weights_and_combines_columns() {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("title".into(), false).expect("register");
        b.register_column("body".into(), false).expect("register");
        // doc 0: title "rust"; doc 1: body "rust"; doc 2: neither.
        b.add_doc(0, 0, "rust").expect("add");
        b.add_doc(1, 0, "systems").expect("add");
        b.add_doc(0, 1, "python").expect("add");
        b.add_doc(1, 1, "rust ml").expect("add");
        b.add_doc(0, 2, "go").expect("add");
        b.add_doc(1, 2, "concurrency").expect("add");
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"title","tokenizer":"ascii_lower"},{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let hits = r
            .search_multi(&[("title", 1.0), ("body", 1.0)], "rust", 10, BoolMode::Or)
            .await
            .expect("multi");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn search_or_range_restricts_to_doc_id_window() {
        // Larger corpus so an OR query spans several doc ids and the
        // ranged path actually clips some out.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..8u32 {
            b.add_doc(0, i, "alpha beta").expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        // Restrict to [2, 5): only docs 2,3,4 are eligible.
        let hits = r
            .search_or_range_pretokenized("body", &["alpha", "beta"], 100, 2, 5)
            .await
            .expect("ranged search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(
            ids,
            [2u32, 3, 4].into_iter().collect(),
            "only docs in [2,5) returned"
        );
    }

    /// Regression: the ranged OR entry must produce the same results as
    /// the un-ranged path for ANY partition of the doc space, on BOTH of
    /// the kernels its dispatch can now pick. Before the fix it hardcoded
    /// MaxScore+BMM, so a query sliced into sub-ranges (the fan-out shape
    /// a compacted table takes) ran a different kernel than the same query
    /// un-ranged — uniform broad ORs degraded 11-24x post-compaction.
    #[tokio::test]
    async fn search_or_range_partitions_agree_with_unranged() {
        /// Docs in the planted corpus — spans several 4096-doc OR windows
        /// and many 128-doc posting blocks.
        const N_DOCS: u32 = 6_000;
        /// Ask for every match so partition union == full result set.
        const K_ALL: usize = N_DOCS as usize;
        /// Top-k size for the truncated comparison.
        const K_TOP: usize = 10;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            // Deterministic mixed-df corpus: four uniform terms with
            // varying tf (windowed-union shape), plus one rare term
            // (dominant-UB / BMM shape when queried with two commons).
            let mut text = String::new();
            for (t, name) in ["alpha", "beta", "gamma", "delta"].iter().enumerate() {
                let h = i.wrapping_mul(31).wrapping_add(t as u32 * 17) % 5;
                for _ in 0..h {
                    text.push_str(name);
                    text.push(' ');
                }
            }
            if i % 2000 == 7 {
                text.push_str("rareterm ");
            }
            if text.is_empty() {
                text.push_str("filler");
            }
            b.add_doc(0, i, &text).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // k-gated routing exercises both ranged kernels: the uniform OR takes
        // the windowed scan at K_ALL (deep k) but MaxScore at the small top-k;
        // the dominant-UB OR stays on MaxScore throughout. Assert it rather
        // than assume it, so a corpus tweak can't silently test one branch twice.
        let shapes: [&[&str]; 2] = [
            &["alpha", "beta", "gamma", "delta"],
            &["rareterm", "alpha", "beta"],
        ];
        let column_id = r.resolve_column_id("body").expect("column");
        let uniform = r
            .build_term_cursors(column_id, shapes[0], None, false)
            .await
            .expect("cursors");
        assert!(
            route_or_to_windowed(&uniform, K_ALL),
            "uniform OR at K_ALL (deep k) must route to the windowed ranged branch"
        );
        assert!(
            !route_or_to_windowed(&uniform, K_TOP),
            "uniform OR at small k must route to the MaxScore ranged branch"
        );
        let dominant = r
            .build_term_cursors(column_id, shapes[1], None, false)
            .await
            .expect("cursors");
        assert!(
            !route_or_to_windowed(&dominant, K_ALL) && !route_or_to_windowed(&dominant, K_TOP),
            "dominant-UB OR must route to MaxScore at every k (not uniform, no ≥100k list)"
        );
        // Uneven partitions, including window-boundary-crossing cuts.
        let partitions: [&[(u32, u32)]; 3] = [
            &[(0, N_DOCS)],
            &[(0, 3_000), (3_000, N_DOCS)],
            &[(0, 100), (100, 4_097), (4_097, 5_000), (5_000, N_DOCS)],
        ];

        for terms in shapes {
            let full = r
                .search("body", terms, K_ALL, BoolMode::Or)
                .await
                .expect("un-ranged search");
            let mut full_sorted: Vec<(u32, u32)> =
                full.iter().map(|&(d, s)| (d, s.to_bits())).collect();
            full_sorted.sort_unstable();

            for cuts in partitions {
                let mut merged: Vec<(u32, f32)> = Vec::new();
                for &(lo, hi) in cuts {
                    merged.extend(
                        r.search_or_range_pretokenized("body", terms, K_ALL, lo, hi)
                            .await
                            .expect("ranged search"),
                    );
                }
                let mut merged_sorted: Vec<(u32, u32)> =
                    merged.iter().map(|&(d, s)| (d, s.to_bits())).collect();
                merged_sorted.sort_unstable();
                assert_eq!(
                    merged_sorted, full_sorted,
                    "partition union must equal the un-ranged result \
                     (terms={terms:?}, cuts={cuts:?})"
                );

                // Top-k contract: resorting the merged pool by
                // (score desc, doc asc) reproduces the un-ranged top-k.
                let mut pool = merged.clone();
                pool.sort_unstable_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .expect("BM25 scores are finite")
                        .then(a.0.cmp(&b.0))
                });
                pool.truncate(K_TOP);
                let top: Vec<(u32, u32)> = pool.iter().map(|&(d, s)| (d, s.to_bits())).collect();
                let full_top: Vec<(u32, u32)> = full
                    .iter()
                    .take(K_TOP)
                    .map(|&(d, s)| (d, s.to_bits()))
                    .collect();
                assert_eq!(
                    top, full_top,
                    "merged top-{K_TOP} must equal un-ranged top-{K_TOP} \
                     (terms={terms:?}, cuts={cuts:?})"
                );
            }
        }
    }

    /// The prebuilt-cursor ranged path must be byte-identical to fresh
    /// per-call builds — it is the same search minus the redundant fetch
    /// and parse, so any divergence is a sharing bug (walk state leaking
    /// between clones, stale first-block decode, ...). One set serves
    /// overlapping windows and a repeated window to force reuse.
    #[tokio::test]
    async fn search_or_range_prebuilt_matches_fresh_calls() {
        /// Docs in the planted corpus (multiple OR windows and blocks).
        const N_DOCS: u32 = 6_000;
        /// Ask for every match so whole result sets are compared.
        const K_ALL: usize = N_DOCS as usize;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            for (t, name) in ["alpha", "beta", "gamma", "delta"].iter().enumerate() {
                let h = i.wrapping_mul(31).wrapping_add(t as u32 * 17) % 5;
                for _ in 0..h {
                    text.push_str(name);
                    text.push(' ');
                }
            }
            if text.is_empty() {
                text.push_str("filler");
            }
            b.add_doc(0, i, &text).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let terms: &[&str] = &["alpha", "beta", "gamma", "delta"];
        let set = r
            .build_or_cursor_set("body", terms, None)
            .await
            .expect("set");
        let windows = [
            (0u32, N_DOCS),
            (0, 3_000),
            (2_000, 4_097),
            (3_000, N_DOCS),
            (0, N_DOCS),
        ];
        for (lo, hi) in windows {
            let fresh = r
                .search_or_range_pretokenized("body", terms, K_ALL, lo, hi)
                .await
                .expect("fresh ranged search");
            let pre = r
                .search_or_range_prebuilt(&set, K_ALL, lo, hi, f32::NEG_INFINITY)
                .expect("prebuilt ranged search");
            let fresh_bits: Vec<(u32, u32)> =
                fresh.iter().map(|&(d, s)| (d, s.to_bits())).collect();
            let pre_bits: Vec<(u32, u32)> = pre.iter().map(|&(d, s)| (d, s.to_bits())).collect();
            assert_eq!(pre_bits, fresh_bits, "window ({lo},{hi})");
        }
    }

    #[tokio::test]
    async fn search_or_range_degenerate_inputs_are_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Empty terms, k == 0, and an inverted range all short-circuit.
        assert!(
            r.search_or_range_pretokenized("body", &[], 10, 0, 3)
                .await
                .expect("empty terms")
                .is_empty()
        );
        assert!(
            r.search_or_range_pretokenized("body", &["rust"], 0, 0, 3)
                .await
                .expect("zero k")
                .is_empty()
        );
        assert!(
            r.search_or_range_pretokenized("body", &["rust"], 10, 3, 3)
                .await
                .expect("empty range")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn search_or_range_with_floor_prunes() {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..8u32 {
            b.add_doc(0, i, "alpha beta").expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let hits = r
            .search_or_range_pretokenized_with_floor(
                "body",
                &["alpha", "beta"],
                100,
                0,
                8,
                1e9,
                None,
            )
            .await
            .expect("floored ranged search");
        assert!(hits.is_empty(), "floor above all scores prunes everything");
    }
}
