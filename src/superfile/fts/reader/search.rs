// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Scored FTS search entry points on [`FtsReader`]: the public
//! `search` / `search_with_floor` / pretokenized and doc-id-ranged
//! variants, the multi-column `search_multi`, the single-term
//! BlockMaxWAND fast path, and the atom (phrase-aware) scored walk.
//! Its own `impl FtsReader` block, split from the reader `core`.

use std::collections::{BinaryHeap, HashMap};

use bytes::Bytes;
use futures::future::try_join_all;
use rustc_hash::FxHashMap;

use super::{
    bounds::BoundDecoder,
    core::*,
    cursor::{CursorUse, TermCursor, TermMeta},
    filter::{AtomExcludeFilter, ExcludeFilter},
    metadata::NormTable,
    options::BoolMode,
    phrase::{AnyCursor, approx_align, batch_phrase_with_terms},
    scorers::{BUFFERED_TOPK_MIN_K, kth_scored, score_postings},
    sink::{Collector, LiveFloor, TopKBuffer, TopKEntry, and_heap_push, drain_top_k_desc},
    work::{
        MatchWork, atom_cursor_bytes, atom_planned_ranges, term_cursor_bytes, term_cursor_ranges,
    },
};
use crate::{
    runtime_metrics::{
        cpu::{thread_cpu_delta_ns, thread_cpu_ns},
        op_stats::metering_active,
    },
    superfile::{
        ReadError,
        bits::{EXPAND_SLACK, expand_bitset, expand_bitset_scalar},
        error::FtsError,
        format,
        fts::{
            bm25,
            posting::{BLOCK_LEN, BlockHeader, decode_block},
            short::{decode_short, short_df},
        },
        id_space::{DocMap, FtsDocId, RowId, RowSet},
        lazy_source::Source,
    },
    utils::terms::{DictEntry, make_key},
};

/// Bytes per doc-id map entry: one little-endian `u32` row.
const DOC_MAP_ENTRY_BYTES: usize = 4;
/// Bytes of the doc-id map a sparse translation reads at a time. Small
/// under test, so a fixture of a thousand documents spans many pages.
const DOC_MAP_PAGE_BYTES: usize = if cfg!(test) { 64 } else { 64 * 1024 };
/// A translation reads map pages instead of the whole map while the pages
/// come to at most one this-many-th of it.
const DOC_MAP_SPARSE_SHARE: usize = 8;

/// An unranked match holding at least one row in this many of its blob's
/// rows is put back in row order through a bitset of the blob's rows
/// instead of a sort. At that density reading the bitset's words back
/// costs no more than touching the rows themselves, and the sort's
/// `n log n` was most of a dense match's time on a reordered blob.
const ROW_ORDER_BITSET_MIN_DENSITY: usize = 64;

/// One scored term's open-wave fetch result: the dictionary resolution
/// plus, for a PFOR term, its full postings range and header df.
/// Cloning is cheap — `Bytes` is a refcounted slice.
#[derive(Clone)]
pub(crate) enum FetchedTermSlot {
    /// df=1 inline dictionary slot (no postings-region bytes exist).
    Inline { doc_id: u32, tf: u32 },
    /// Postings-form term: the fetched range and its df. `short` says
    /// whether the range is a short-form body or the long form's
    /// metadata + skip table + blocks.
    Pfor {
        bytes: Bytes,
        df: u64,
        short: bool,
        /// The term's offset in the postings region: its identity in this
        /// file, which keys the reader's block-table cache.
        metadata_offset: u64,
    },
}

/// One superfile's open-wave fetches for a global-stats query's scored
/// terms, keyed by term. A key that is present with a `None` value is a
/// *resolved miss* (the term is absent from this superfile) — the walk
/// wave must not re-probe the dictionary for it. Built by
/// [`FtsReader::fetch_scored_terms`]; consumed by the cursor builds via
/// the `prefetched` parameter.
pub(crate) struct FetchedTermMemo {
    map: HashMap<Box<str>, Option<FetchedTermSlot>>,
}

impl FetchedTermMemo {
    /// A memo from slots resolved elsewhere — by the open wave, or from a
    /// table-level term index that already knows where each term's
    /// postings sit, so the cursor build needs no dictionary at all.
    pub(crate) fn from_slots(
        slots: impl IntoIterator<Item = (Box<str>, Option<FetchedTermSlot>)>,
    ) -> Self {
        Self {
            map: slots.into_iter().collect(),
        }
    }

    /// `Some(entry)` when the open wave resolved this term (entry `None`
    /// = absent from this superfile); `None` when the term was not part
    /// of the fetch (e.g. a negated term — unscored, never gathered).
    pub(crate) fn lookup(&self, term: &str) -> Option<Option<FetchedTermSlot>> {
        self.map.get(term).cloned()
    }

    fn contains(&self, term: &str) -> bool {
        self.map.contains_key(term)
    }

    /// Whether every one of `terms` was resolved into this memo — the
    /// condition under which a cursor build needs no dictionary at all.
    pub(crate) fn covers(&self, terms: &[&str]) -> bool {
        terms.iter().all(|t| self.contains(t))
    }

    /// Record `term` as resolved absent from this superfile, unless the
    /// memo already holds it: a caller that knows the term has no postings
    /// here (a table-level term index lists none) spares the cursor build
    /// a dictionary read that would only confirm the absence.
    pub(crate) fn note_absent(&mut self, term: &str) {
        self.map.entry(Box::from(term)).or_insert(None);
    }

    /// This superfile's df contribution per term (0 for a miss or an
    /// un-fetched term, 1 for an inline slot).
    pub(crate) fn df(&self, term: &str) -> u64 {
        match self.map.get(term) {
            Some(Some(FetchedTermSlot::Inline { .. })) => 1,
            Some(Some(FetchedTermSlot::Pfor { df, .. })) => *df,
            _ => 0,
        }
    }
}

/// Threshold-first seed tuning for the single-term ranked walk. The walk
/// fills its top-k heap in doc-id order, so its skip threshold rises only
/// as the heap fills and it decodes many blocks before the threshold is
/// high enough to skip them. Decoding the highest-block-max blocks up front
/// (found via the coarse block-max table) raises the k-th threshold first,
/// so the main walk prunes those otherwise-wasted decodes.
pub(super) mod seed {
    /// Don't seed lists shorter than this many blocks — the setup isn't
    /// worth it when there's little waste to prune.
    pub const MIN_BLOCKS: usize = 256;
    /// Minimum coarse spans to gather when selecting seed-block candidates.
    pub const TOP_SPANS: usize = 16;
    /// Floor on seed-block count (enough for a tight threshold at small k).
    pub const MIN_BLOCKS_SEED: usize = 32;
    /// Hard cap so a pathological k can't make the seed scan the whole list.
    pub const MAX_BLOCKS_SEED: usize = 2048;
}

/// Seed-block count for top-k `k`: `(k + k/4) / 2`, a little over half a
/// block per top-k doc, clamped. The best blocks often hold more than one
/// top-k doc each, so about half a block per doc already puts the seed's
/// k-th close to the final one; the blocks past that cost more to decode
/// than the walk saves (measured: a block per doc, or a quarter of one, both
/// decode more in total than half). Saturating so a merge caller's huge `k`
/// clamps to the cap rather than overflowing.
pub(super) fn seed_blocks_for(k: usize) -> usize {
    (k.saturating_add(k / 4) / 2).clamp(seed::MIN_BLOCKS_SEED, seed::MAX_BLOCKS_SEED)
}

/// Returns `rows` in ascending order. Rows a doc-id map already left in
/// order come back as they are; a set dense enough
/// ([`ROW_ORDER_BITSET_MIN_DENSITY`]) is put in order through a bitset of
/// the blob's `n_rows` rows ([`rows_in_order`]); anything else, and a set
/// the bitset refuses, is sorted.
fn into_row_order(mut rows: Vec<RowId>, n_rows: usize) -> Vec<RowId> {
    if rows.is_sorted() {
        return rows;
    }
    if rows.len().saturating_mul(ROW_ORDER_BITSET_MIN_DENSITY) >= n_rows
        && let Some(ordered) = rows_in_order(&rows, n_rows)
    {
        return ordered;
    }
    rows.sort_unstable();
    rows
}

/// Returns `rows` in ascending order, read back from a bitset of the
/// `n_rows` rows in which each is marked: the expansion the posting
/// decoder uses for a presence block.
///
/// Each row must be distinct and below `n_rows`. `None` when one is not,
/// which only a corrupt doc-id map produces (its length is checked at
/// open, its values are not, and a lazy open skips the CRC): the caller
/// then sorts, which returns such a row rather than failing the query.
fn rows_in_order(rows: &[RowId], n_rows: usize) -> Option<Vec<RowId>> {
    let word_bytes = (u64::BITS / u8::BITS) as usize;
    let mut bits = vec![0u8; n_rows.div_ceil(u64::BITS as usize) * word_bytes];
    for row in rows {
        let r = row.get() as usize;
        if r >= n_rows {
            return None;
        }
        bits[r / u8::BITS as usize] |= 1 << (r % u8::BITS as usize);
    }
    let mut out = vec![0u32; rows.len() + EXPAND_SLACK];
    let n = match expand_bitset(&bits, 0, &mut out) {
        Some(n) => n,
        None => expand_bitset_scalar(&bits, 0, &mut out),
    };
    // A row named twice sets one bit: fewer rows come back than went in.
    if n != rows.len() {
        return None;
    }
    out.truncate(n);
    Some(out.into_iter().map(RowId::new).collect())
}

impl FtsReader {
    /// Build a [`FetchedTermMemo`] for `terms` whose dictionary values are
    /// already known — from a table-level term index — fetching only the
    /// postings ranges those values name, never the dictionary.
    pub(crate) async fn memo_from_dict_values(
        &self,
        terms: &[(&str, u64, DictEntry)],
    ) -> Result<FetchedTermMemo, FtsError> {
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut order: Vec<(usize, u64, bool, u64)> = Vec::new();
        let mut slots: Vec<(Box<str>, Option<FetchedTermSlot>)> = Vec::with_capacity(terms.len());
        for (i, (term, df, value)) in terms.iter().enumerate() {
            match value {
                DictEntry::Inline { doc_id, tf } => slots.push((
                    Box::from(*term),
                    Some(FetchedTermSlot::Inline {
                        doc_id: *doc_id,
                        tf: *tf,
                    }),
                )),
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => {
                    ranges.push((*metadata_offset as usize, *postings_length as usize));
                    order.push((i, *df, *short, *metadata_offset));
                }
            }
        }
        let fetched = self.fetch_term_postings(&ranges).await?;
        for ((i, df, short, metadata_offset), bytes) in order.into_iter().zip(fetched) {
            slots.push((
                Box::from(terms[i].0),
                Some(FetchedTermSlot::Pfor {
                    bytes,
                    df,
                    short,
                    metadata_offset,
                }),
            ));
        }
        Ok(FetchedTermMemo::from_slots(slots))
    }

    /// Ranked search over heterogeneous atoms — the walk every
    /// phrase-bearing query takes. With musts, the match set is their
    /// intersection and shoulds are scoring-only (the clause model);
    /// with none, the shoulds' union matches. Docs excluded by
    /// `filter` never reach the heap; docs scoring strictly below
    /// `floor_eff` are dropped at admission.
    ///
    /// Phrase verification (the position decode) dominates this walk,
    /// and its bar-skip is only as good as the bar — so the walk keeps
    /// the bar live: `live_floor`, when present, is re-read every
    /// candidate (other segments of the fan-out raise it as their
    /// heaps fill) and this walk publishes its own kth-best back as it
    /// improves. A one-shot `floor_eff` snapshot instead left each
    /// segment verifying against its slowly-filling local heap alone,
    /// which showed up as phrase shapes degrading to count-like work.
    fn run_atoms_search(
        &self,
        column_id: u32,
        mut musts: Vec<AnyCursor>,
        mut shoulds: Vec<AnyCursor>,
        k: usize,
        mut filter: Option<AtomExcludeFilter>,
        floor_eff: f32,
        live_floor: Option<&LiveFloor>,
        exact_count: bool,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        debug_assert!(
            !exact_count || (floor_eff == f32::NEG_INFINITY && live_floor.is_none()),
            "an exact count takes no floor"
        );
        let dl_norm_k1 = &self.columns[column_id as usize].dl_norm_k1();
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut count = 0u64;
        // The strict-below admission cutoff, refreshed from the live
        // floor: everything published there is a real kth-best, so the
        // same `next_down` the caller applied to its snapshot keeps
        // exact ties admissible.
        let floor_now = || match live_floor {
            Some(live) => live.load().next_down().max(floor_eff),
            None => floor_eff,
        };
        // Publish this walk's kth-best once its heap is full — the
        // fan-out's other segments prune against it immediately.
        let publish = |heap: &BinaryHeap<TopKEntry>| {
            if let Some(live) = live_floor
                && heap.len() >= k
                && let Some(worst) = heap.peek()
            {
                live.raise(worst.0);
            }
        };

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
                let floor_live = floor_now();
                if admitted {
                    count += 1;
                    let norm = dl_norm_k1.get(doc);
                    let score: f32 = shoulds
                        .iter()
                        .filter(|a| !a.is_exhausted() && a.current_doc_id() == doc)
                        .map(|a| a.score_current(norm))
                        .sum();
                    if score > floor_live {
                        and_heap_push(&mut heap, k, None, score, doc);
                        publish(&heap);
                    }
                }
                let Some(next) = doc.checked_add(1) else {
                    break;
                };
                let bar = match !exact_count && heap.len() >= k {
                    true => heap.peek().expect("heap len == k").0.max(floor_live),
                    false => floor_live,
                };
                for (a, &others) in shoulds.iter_mut().zip(&others_ub) {
                    if !a.is_exhausted() && a.current_doc_id() == doc {
                        a.skip_to_pruned(next, bar - others, dl_norm_k1)?;
                    }
                }
            }
            return Ok((drain_top_k_desc(heap), count));
        }

        // Must-driven walk, two-phase per candidate: (1) align every must by its
        // cheap approximation (a phrase advances only to a member co-occurrence,
        // no positions); (2) bar-skip on block-max UBs; (3) verify phrase
        // adjacency and score only on the survivors. The old single-phase walk
        // verified phrases *during* alignment, so a phrase of common words cost
        // the same with or without a selective co-clause; verifying only on the
        // aligned intersection lets that clause prune the position work.
        let should_ub: f32 = shoulds.iter().map(AnyCursor::term_max_bm25).sum();
        let should_others_ub: Vec<f32> = {
            let must_ub_total: f32 = musts.iter().map(AnyCursor::term_max_bm25).sum();
            atom_slack(&shoulds, must_ub_total)
        };
        // One phrase ANDed with terms walks the must intersection itself, a
        // block at a time, every doc it lands on verified; it prunes by its
        // own bound against the bar less everything else could add.
        let batched = batch_phrase_with_terms(&mut musts)?;
        let must_others_ub = atom_slack(&musts, should_ub);
        let mut target = 0u32;
        'docs: loop {
            let floor_live = floor_now();
            let bar = match !exact_count && heap.len() >= k {
                true => heap.peek().expect("heap len == k").0.max(floor_live),
                false => floor_live,
            };
            let aligned = match batched {
                Some(at) => {
                    let a = &mut musts[at];
                    a.skip_to_pruned(target, bar - must_others_ub[at], dl_norm_k1)?;
                    if a.is_exhausted() {
                        break 'docs;
                    }
                    a.current_doc_id()
                }
                // Phase 1 — approximate alignment (phrase = member
                // co-occurrence, no positions).
                None => match approx_align(&mut musts, target) {
                    Some(aligned) => aligned,
                    None => break 'docs,
                },
            };
            // Bar skip on block-max UBs, before any position work: a candidate
            // whose musts + shoulds can't reach the kth-best is dead without a
            // verify. `>=`, not `>`: a doc exactly at the bar can still displace
            // the incumbent kth-best on the ascending-doc-id tie-break.
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
            if scoring_needed {
                // Phase 2 — verify phrase adjacency at the aligned candidate;
                // terms match trivially. Only survivors reach the position
                // decode and scoring. The batched walk verified its doc
                // already, and brings the terms to it only now, to score.
                let mut matched = true;
                for a in musts.iter_mut() {
                    let holds = match batched {
                        Some(_) => {
                            a.skip_to(aligned)?;
                            true
                        }
                        None => a.verify_at(aligned)?,
                    };
                    if !holds {
                        matched = false;
                        break;
                    }
                }
                let admitted = matched
                    && match filter.as_mut() {
                        Some(f) => f.admits(aligned)?,
                        None => true,
                    };
                if admitted {
                    count += 1;
                    let norm = dl_norm_k1.get(aligned);
                    let mut score: f32 = musts.iter().map(|a| a.score_current(norm)).sum();
                    for (sh, &others) in shoulds.iter_mut().zip(&should_others_ub) {
                        sh.skip_to_pruned(aligned, bar - others, dl_norm_k1)?;
                        if !sh.is_exhausted() && sh.current_doc_id() == aligned {
                            score += sh.score_current(norm);
                        }
                    }
                    if score > floor_live {
                        and_heap_push(&mut heap, k, None, score, aligned);
                        publish(&heap);
                    }
                }
            }
            let Some(next) = aligned.checked_add(1) else {
                break;
            };
            target = next;
        }
        Ok((drain_top_k_desc(heap), count))
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
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
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
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
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
                    idf: QueryIdf::ThisFile,
                    ..ClauseLists::default()
                },
                k,
                floor,
            )
            .await?;
        self.hits_to_rows_async(self.run_prepared_in_blob_ids(prep)?)
            .await
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
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        let prep = self.prepare_clauses(column, lists, k, floor).await?;
        self.hits_to_rows_async(self.run_prepared_in_blob_ids(prep)?)
            .await
    }

    /// I/O half of an un-ranged clause search: resolve the column,
    /// classify the query shape, and fetch every cursor
    /// [`Self::run_prepared`] needs to score. The single-atom shape
    /// finishes here since it's cheap; the phrase-atom shape also
    /// finishes here, but only because it isn't wired to the reader
    /// pool yet, not because it's cheap.
    /// Collapse repeated clause terms into distinct terms plus a per-term repeat
    /// count (query-term-frequency). Order-preserving; O(n²) over the handful of
    /// query terms (the same shape the count path already dedups with). Returns
    /// each term with count 1 — a no-op — for a distinct-term query.
    fn dedup_terms_with_qtf<'a>(terms: &[&'a str]) -> (Vec<&'a str>, Vec<u32>) {
        let mut distinct: Vec<&str> = Vec::with_capacity(terms.len());
        let mut qtf: Vec<u32> = Vec::with_capacity(terms.len());
        for &t in terms {
            match distinct.iter().position(|&d| d == t) {
                Some(i) => qtf[i] += 1,
                None => {
                    distinct.push(t);
                    qtf.push(1);
                }
            }
        }
        (distinct, qtf)
    }

    /// The doc-id map for a filter: read now when it checks rows against an
    /// allow set, and the identity otherwise, which such a filter never
    /// consults.
    async fn filter_doc_map(&self, checks_rows: bool) -> Result<DocMap, FtsError> {
        match checks_rows {
            true => Ok(self.ensure_doc_map().await?.clone()),
            false => Ok(DocMap::Identity),
        }
    }

    pub(crate) async fn prepare_clauses(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
        floor: f32,
    ) -> Result<PreparedClauses, FtsError> {
        // No doc-id map is read here: the scoring that follows returns blob
        // ids, and its caller translates the top hits on its own runtime
        // ([`Self::hits_to_rows_async`]), reading only the map pages they
        // fall in. Reading the whole map before scoring cost every
        // superfile a ranked search touched four bytes a document.
        self.build_clauses(column, lists, k, floor).await
    }

    /// [`Self::prepare_clauses`] before the doc-id map: the cursors, or the
    /// final hits in blob ids.
    async fn build_clauses(
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
                count: 0,
                exact_count: lists.exact_count,
                postings_bytes: 0,
                planned_ranges: 0,
                kernel_cpu_ns: 0,
            });
        }
        if lists.no_positive_atoms() {
            if lists.no_negative_atoms() {
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    count: 0,
                    exact_count: lists.exact_count,
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
                .build_atom_cursors(
                    column_id,
                    lists.musts,
                    lists.must_phrases,
                    lists.idf,
                    lists.prefetched,
                    CursorUse::Score,
                )
                .await?;
            if must_atoms.iter().any(Option::is_none) {
                // A must atom can never match in this superfile. The
                // atoms that DID build still cost their bytes.
                let built: Vec<AnyCursor> = must_atoms.into_iter().flatten().collect();
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    count: 0,
                    exact_count: lists.exact_count,
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
                    lists.idf,
                    lists.prefetched,
                    CursorUse::Score,
                )
                .await?;
            let should_atoms: Vec<AnyCursor> = should_built.into_iter().flatten().collect();
            // Negatives are a hard exclusion filter, not scored, so their
            // idf is irrelevant — always build them local.
            let (negative_built, negative_dict) = self
                .build_atom_cursors(
                    column_id,
                    lists.negatives,
                    lists.negative_phrases,
                    QueryIdf::ThisFile,
                    None,
                    CursorUse::Match,
                )
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
            // No positive atom of the query is in this superfile, so nothing
            // here can match: return before the filter, which would read the
            // doc-id map to check rows there are none of.
            if must_atoms.is_empty() && should_atoms.is_empty() {
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    count: 0,
                    exact_count: lists.exact_count,
                    postings_bytes,
                    planned_ranges,
                    kernel_cpu_ns: 0,
                });
            }
            // The gate exists when there is anything to gate on: a negated
            // atom, or a pushed-down row set.
            let filter = match (negative_atoms.is_empty(), lists.allow.clone()) {
                (true, None) => None,
                (_, allow) => {
                    let doc_map = self.filter_doc_map(allow.is_some()).await?;
                    Some(AtomExcludeFilter::with_allow(
                        negative_atoms,
                        allow.map(RowSet::new),
                        doc_map,
                    ))
                }
            };
            // The atom walk is the whole kernel for phrase shapes —
            // `run_prepared` sees only the finished `Done` — so bracket
            // its on-CPU time here (sync section, no awaits inside).
            // Gated: an unmetered process must not pay the procfs reads.
            let kernel_start = metering_active().then(thread_cpu_ns).flatten();
            let (hits, count) = self.run_atoms_search(
                column_id,
                must_atoms,
                should_atoms,
                k,
                filter,
                floor_eff,
                lists.live_floor,
                lists.exact_count,
            )?;
            return Ok(PreparedClauses::Done {
                hits,
                count,
                exact_count: lists.exact_count,
                postings_bytes,
                planned_ranges,
                kernel_cpu_ns: thread_cpu_delta_ns(kernel_start),
            });
        }

        // Negatives are a hard exclusion filter that only seeks doc ids, so
        // they are built to count: no tfs, idf or bounds. They come from the
        // memo like the scored terms where the index located them; a lazy
        // reader otherwise fetches the dictionary whole to find them.
        let neg_cursors = match lists.negatives {
            [] => Vec::new(),
            negatives => {
                self.build_term_cursors(
                    column_id,
                    negatives,
                    QueryIdf::ThisFile,
                    CursorUse::Count,
                    None,
                    lists.prefetched,
                )
                .await?
            }
        };
        // term-dictionary ranges the builds below request — one per
        // `build_term_cursors` call that the memo does not cover (the
        // dictionary fetch is a real byte-source range on every query,
        // warm or cold).
        let negatives_from_memo = lists
            .prefetched
            .is_some_and(|m| lists.negatives.iter().all(|t| m.contains(t)));
        let mut dict_ranges = u64::from(!lists.negatives.is_empty() && !negatives_from_memo);
        // The gate exists when there is anything to gate on: a negated
        // term present in this superfile, or a pushed-down row set.
        let neg_filter = match (neg_cursors.is_empty(), lists.allow.clone()) {
            (true, None) => None,
            (_, allow) => {
                // The positive terms are resolved below, after the filter;
                // with an allow set the map is read here so a filter that
                // checks rows always holds the real one.
                let doc_map = self.filter_doc_map(allow.is_some()).await?;
                Some(ExcludeFilter::with_allow(
                    neg_cursors,
                    allow.map(RowSet::new),
                    doc_map,
                ))
            }
        };

        // Fold repeated MUST/SHOULD terms into one weighted cursor each: the
        // repeat count becomes a query-term-frequency multiplier on the term's
        // idf. BM25 is linear in `idf_weight`, so this scores identically to N
        // duplicate cursors at 1/N the cursor + scan cost — e.g. `+to +be +or
        // +not +to +be` builds 4 cursors, not 6. A no-op for distinct-term
        // queries. The single-term fast path below gates on the pre-dedup count,
        // so a repeated lone term (`+to +to`) routes through the weighted path.
        let (musts, must_qtf) = Self::dedup_terms_with_qtf(lists.musts);
        let (shoulds, should_qtf) = Self::dedup_terms_with_qtf(lists.shoulds);

        // Single-atom fast path: BlockMaxWAND-driven block skipping.
        // One term scores identically whichever clause list it sits
        // in (a lone must and a lone should both rank that term's
        // postings), so both shapes take it. The walk scores with the
        // query's idf and rescales its stored bounds by the exact linear
        // ratio to it, so the skip decisions stay as tight as they would
        // be at the idf the bounds were stored at.
        if !lists.exact_count && lists.musts.len() + lists.shoulds.len() == 1 {
            let term = lists
                .musts
                .iter()
                .chain(lists.shoulds)
                .next()
                .expect("one atom");
            let gidf = lists.idf.term(term);
            let mut filter = neg_filter;
            let filter_postings_bytes = filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
            let filter_ranges = filter.as_ref().map_or(0, ExcludeFilter::planned_ranges);
            let (result, term_work, kernel_cpu_ns) = self
                .search_single_term_bmw(
                    column_id,
                    term,
                    k,
                    filter.as_mut(),
                    floor_eff,
                    gidf,
                    lists.prefetched,
                )
                .await?;
            // +1: the walk's dictionary read — performed here on the
            // single-pass path, by the open wave on the memo path; either
            // way the query read it once and the walk wave reports it.
            dict_ranges += 1;
            return Ok(PreparedClauses::Done {
                hits: result,
                count: 0,
                exact_count: false,
                postings_bytes: term_work.postings_bytes + filter_postings_bytes,
                planned_ranges: term_work.planned_ranges + filter_ranges + dict_ranges,
                kernel_cpu_ns,
            });
        }

        if musts.is_empty() {
            let cursors = self
                .build_term_cursors(
                    column_id,
                    &shoulds,
                    lists.idf,
                    CursorUse::Score,
                    Some(&should_qtf),
                    lists.prefetched,
                )
                .await?;
            dict_ranges += 1;
            if cursors.is_empty() {
                let postings_bytes = neg_filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
                let planned_ranges =
                    neg_filter.as_ref().map_or(0, ExcludeFilter::planned_ranges) + dict_ranges;
                return Ok(PreparedClauses::Done {
                    hits: Vec::new(),
                    count: 0,
                    exact_count: lists.exact_count,
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
                exact_count: lists.exact_count,
            });
        }
        // Build must cursors; if any must is missing, the
        // intersection is empty.
        let must_cursors = self
            .build_term_cursors(
                column_id,
                &musts,
                lists.idf,
                CursorUse::Score,
                Some(&must_qtf),
                lists.prefetched,
            )
            .await?;
        dict_ranges += 1;
        if must_cursors.len() != musts.len() {
            let postings_bytes = term_cursor_bytes(&must_cursors)
                + neg_filter.as_ref().map_or(0, ExcludeFilter::postings_bytes);
            let planned_ranges = term_cursor_ranges(&must_cursors)
                + neg_filter.as_ref().map_or(0, ExcludeFilter::planned_ranges)
                + dict_ranges;
            return Ok(PreparedClauses::Done {
                hits: Vec::new(),
                count: 0,
                exact_count: lists.exact_count,
                postings_bytes,
                planned_ranges,
                kernel_cpu_ns: 0,
            });
        }
        if shoulds.is_empty() {
            return Ok(PreparedClauses::Must {
                column_id,
                must_cursors,
                filter: neg_filter,
                k,
                floor_eff,
                dict_ranges,
                exact_count: lists.exact_count,
            });
        }
        // Shoulds absent from this superfile contribute nothing;
        // when none survive, the walk is a plain must intersection.
        let should_cursors = self
            .build_term_cursors(
                column_id,
                &shoulds,
                lists.idf,
                CursorUse::Score,
                Some(&should_qtf),
                lists.prefetched,
            )
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
                exact_count: lists.exact_count,
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
            exact_count: lists.exact_count,
        })
    }

    /// CPU half paired with [`Self::prepare_clauses`] — scores the
    /// cursors it fetched. No I/O, so it can run on the reader pool.
    /// Translate a ranked kernel's doc ids into Parquet rows.
    ///
    /// The kernels work in the blob's own id space from end to end, so
    /// anything handing hits to a caller passes them through here, once.
    /// A no-op on any blob without a doc-id map, which is every blob
    /// through `V7`.
    ///
    /// Order is left alone: a ranked result is ordered by score, and
    /// translating ids does not disturb that. The unranked siblings
    /// promise ascending ids instead, which translation *does* disturb,
    /// so they use [`Self::ids_to_rows`].
    ///
    /// No hits read no map.
    pub(super) fn hits_to_rows(
        &self,
        hits: Vec<(FtsDocId, f32)>,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let map = self.doc_map()?;
        Ok(hits
            .into_iter()
            .map(|(doc, score)| (map.row_of(doc), score))
            .collect())
    }

    /// Translate an unranked kernel's doc ids into Parquet rows, kept
    /// ascending.
    ///
    /// The unranked walks emit ids in blob order, which callers read as
    /// ascending row order: a row-keyed bitmap is built from them, a
    /// membership filter is probed with them, and the exact-match second
    /// pass decodes them in order. On a blob storing its documents in an
    /// order of its own those are different orders, so this re-orders
    /// ([`into_row_order`]). There is nothing to re-order when there is no
    /// map.
    ///
    /// No ids read no map.
    pub(super) fn ids_to_rows(&self, ids: Vec<FtsDocId>) -> Result<Vec<RowId>, FtsError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let map = self.doc_map()?;
        let rows = ids.iter().map(|&id| map.row_of(id));
        if !self.has_doc_map() {
            return Ok(rows.collect());
        }
        Ok(into_row_order(rows.collect(), self.n_docs() as usize))
    }

    /// [`Self::ids_to_rows`] on the caller's runtime, reading only the map
    /// pages the ids fall in when the map is not in memory yet and they are
    /// few (see [`Self::rows_from_map_pages`]); otherwise the whole map, once,
    /// as before.
    pub(super) async fn ids_to_rows_async(
        &self,
        ids: Vec<FtsDocId>,
    ) -> Result<Vec<RowId>, FtsError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(rows) = self.rows_from_map_pages(&ids).await? {
            return Ok(into_row_order(rows, self.n_docs() as usize));
        }
        self.ensure_doc_map().await?;
        self.ids_to_rows(ids)
    }

    /// The rows `ids` stand for, read from just the doc-id map pages they
    /// fall in, or `None` when the whole map is the better read: it is in
    /// memory already, the blob keeps row order or is in memory, or the
    /// pages would add up to more than [`DOC_MAP_SPARSE_SHARE`] of it.
    ///
    /// A match of a few rows in each of thousands of reordered superfiles
    /// would otherwise read every superfile's whole map, four bytes a
    /// document, to translate a handful of ids: hundreds of megabytes a
    /// superfile, held at once across the fan-out. The pages are not kept
    /// and, unlike the whole-map read, not checked against the map's CRC,
    /// which covers the whole body.
    async fn rows_from_map_pages(&self, ids: &[FtsDocId]) -> Result<Option<Vec<RowId>>, FtsError> {
        if self.doc_map.get().is_some() || !matches!(self.source, Source::Lazy(_)) {
            return Ok(None);
        }
        let Some(region) = self.doc_map_region.clone() else {
            return Ok(None);
        };
        let page_of = |id: FtsDocId| id.get() as usize * DOC_MAP_ENTRY_BYTES / DOC_MAP_PAGE_BYTES;
        let mut pages: Vec<usize> = ids.iter().map(|&id| page_of(id)).collect();
        pages.sort_unstable();
        pages.dedup();
        if pages.len() * DOC_MAP_PAGE_BYTES * DOC_MAP_SPARSE_SHARE > region.len() {
            return Ok(None);
        }
        // One read per run of adjacent pages, `(first, last)` inclusive.
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for page in pages {
            match runs.last_mut() {
                Some((_, last)) if *last + 1 == page => *last = page,
                _ => runs.push((page, page)),
            }
        }
        let reads = runs.iter().map(|&(first, last)| {
            let start = region.start + first * DOC_MAP_PAGE_BYTES;
            let end = (region.start + (last + 1) * DOC_MAP_PAGE_BYTES).min(region.end);
            async move {
                self.source
                    .range_async(start..end)
                    .await
                    .map_err(|e| FtsError::RangeFetch {
                        what: "fts/doc-map page",
                        source: e,
                    })
            }
        });
        let bytes = try_join_all(reads).await?;
        ids.iter()
            .map(|&id| {
                let at = id.get() as usize * DOC_MAP_ENTRY_BYTES;
                let run = runs.partition_point(|&(_, last)| last < at / DOC_MAP_PAGE_BYTES);
                let off = at - runs[run].0 * DOC_MAP_PAGE_BYTES;
                bytes[run]
                    .get(off..off + DOC_MAP_ENTRY_BYTES)
                    .and_then(|b| b.try_into().ok())
                    .map(|b| RowId::new(u32::from_le_bytes(b)))
                    .ok_or_else(|| {
                        FtsError::Read(ReadError::Malformed(format!(
                            "fts doc-map has no entry for blob doc id {}",
                            id.get()
                        )))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }

    #[cfg(test)]
    pub(crate) fn run_prepared(
        &self,
        prep: PreparedClauses,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        self.run_prepared_in_blob_ids(prep)
            .and_then(|h| self.hits_to_rows(h))
    }

    /// [`Self::search_excluding`] scoring every match in one pass: the
    /// top-k plus the exact match count. Nothing is pruned, so this does
    /// more work than the ranked search. The supertable fan-out pairs
    /// `prepare_clauses` with `run_prepared_with_count` itself; this
    /// single-call form is what the kernel tests drive.
    #[cfg(test)]
    pub(crate) async fn search_excluding_with_count(
        &self,
        column: &str,
        lists: ClauseLists<'_>,
        k: usize,
    ) -> Result<(Vec<(RowId, f32)>, u64), FtsError> {
        let lists = ClauseLists {
            exact_count: true,
            live_floor: None,
            ..lists
        };
        let prep = self
            .prepare_clauses(column, lists, k, f32::NEG_INFINITY)
            .await?;
        self.run_prepared_with_count(prep)
    }

    /// [`Self::run_prepared`] for clauses prepared with
    /// `ClauseLists::exact_count`: the top-k with the exact match count.
    /// One pass for every shape: each match is scored, so the count and
    /// the top-k come from the same walk, and nothing is pruned.
    ///
    /// The top-k is the ranked search's up to float summation order: the
    /// exhaustive walks add a doc's term scores in a different order than
    /// the pruned kernels, so a total can differ in its last bits, and two
    /// docs tied at the k-th score within that error may swap.
    #[cfg(test)]
    pub(crate) fn run_prepared_with_count(
        &self,
        prep: PreparedClauses,
    ) -> Result<(Vec<(RowId, f32)>, u64), FtsError> {
        let (hits, count) = self.run_prepared_with_count_blob(prep)?;
        Ok((self.hits_to_rows(hits)?, count))
    }

    fn run_prepared_with_count_blob(
        &self,
        prep: PreparedClauses,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        debug_assert!(
            prep.exact_count(),
            "run_prepared_with_count takes clauses prepared with exact_count"
        );
        let (hits, count) = match prep {
            PreparedClauses::Done { hits, count, .. } => (hits, count),
            // One required term matches exactly its own postings, as a
            // one-term union does: that walk scores a block at a time.
            PreparedClauses::Must {
                column_id,
                must_cursors,
                mut filter,
                k,
                ..
            } if must_cursors.len() == 1 => {
                self.run_windowed_union_counted(column_id, must_cursors, k, filter.as_mut())?
            }
            PreparedClauses::Must {
                column_id,
                must_cursors,
                mut filter,
                k,
                ..
            } => self.run_and_intersect_counted(column_id, must_cursors, k, filter.as_mut())?,
            PreparedClauses::MustShould {
                column_id,
                must_cursors,
                should_cursors,
                mut filter,
                k,
                ..
            } => self.run_must_should_counted(
                column_id,
                must_cursors,
                should_cursors,
                k,
                filter.as_mut(),
            )?,
            PreparedClauses::Or {
                column_id,
                cursors,
                mut filter,
                k,
                ..
            } => self.run_windowed_union_counted(column_id, cursors, k, filter.as_mut())?,
        };
        Ok((hits, count))
    }

    /// [`Self::run_prepared_with_count`]'s hits, in blob ids.
    pub(crate) fn run_prepared_with_count_in_blob_ids(
        &self,
        prep: PreparedClauses,
    ) -> Result<(Vec<(FtsDocId, f32)>, u64), FtsError> {
        self.run_prepared_with_count_blob(prep)
    }

    /// [`Self::hits_to_rows`] on the caller's runtime, reading only the map
    /// pages the hits fall in when the map is not in memory yet (see
    /// [`Self::rows_from_map_pages`]).
    pub(crate) async fn hits_to_rows_async(
        &self,
        hits: Vec<(FtsDocId, f32)>,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        if hits.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<FtsDocId> = hits.iter().map(|&(doc, _)| doc).collect();
        if let Some(rows) = self.rows_from_map_pages(&ids).await? {
            return Ok(rows
                .into_iter()
                .zip(hits)
                .map(|(row, (_, score))| (row, score))
                .collect());
        }
        self.ensure_doc_map().await?;
        self.hits_to_rows(hits)
    }

    /// [`Self::run_prepared`] without the translation.
    pub(crate) fn run_prepared_in_blob_ids(
        &self,
        prep: PreparedClauses,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        match prep {
            PreparedClauses::Done { hits, .. } => Ok(hits),
            PreparedClauses::Must {
                column_id,
                must_cursors,
                mut filter,
                dict_ranges: _,
                exact_count: _,
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
                exact_count: _,
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
                exact_count: _,
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
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        self.search_or_range_pretokenized_with_floor(
            column,
            terms,
            k,
            doc_id_start,
            doc_id_end,
            f32::NEG_INFINITY,
            QueryIdf::ThisFile,
        )
        .await
    }

    /// [`Self::search_or_range_pretokenized`] with a score floor — see
    /// [`Self::search_with_floor`] for the floor contract.
    pub(crate) async fn search_or_range_pretokenized_with_floor(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
        idf: QueryIdf<'_>,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        let set = self.build_or_cursor_set(column, terms, idf, None).await?;
        let hits =
            self.search_or_range_prebuilt_in_blob_ids(&set, k, doc_id_start, doc_id_end, floor)?;
        self.hits_to_rows_async(hits).await
    }

    /// Build the OR cursors for `terms` once — the postings fetch and
    /// skip-table parse — for reuse across doc-id sub-ranges via
    /// [`Self::search_or_range_prebuilt`]. An intra-superfile fan-out
    /// that builds per slice re-fetches every term's full posting bytes
    /// and re-parses its skip table per slice (measured at 1M as 2.5x
    /// cold bytes when slicing widened); clones of these cursors share
    /// `bytes` and the `Arc` skip table instead.
    ///
    /// `idf` is baked into the cursors here (see
    /// [`Self::build_term_cursors`]), so every sub-range sharing a set
    /// must want the same override — it does: one gather per query.
    pub(crate) async fn build_or_cursor_set(
        &self,
        column: &str,
        terms: &[&str],
        idf: QueryIdf<'_>,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<OrCursorSet, FtsError> {
        self.build_cursor_set(column, terms, idf, CursorUse::Score, prefetched)
            .await
    }

    /// [`Self::build_or_cursor_set`] for cursors built for `purpose`: the
    /// ranked fan-out scores its slices, the unranked one only matches
    /// them ([`Self::match_cursor_set`]).
    pub(super) async fn build_cursor_set(
        &self,
        column: &str,
        terms: &[&str],
        idf: QueryIdf<'_>,
        purpose: CursorUse,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<OrCursorSet, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let cursors = if terms.is_empty() {
            Vec::new()
        } else {
            self.build_term_cursors(column_id, terms, idf, purpose, None, prefetched)
                .await?
        };
        // An unranked slice translates its ids to rows off this runtime; a
        // ranked one returns blob ids for its caller to translate.
        if !cursors.is_empty() && purpose != CursorUse::Score {
            self.ensure_doc_map().await?;
        }
        Ok(OrCursorSet { column_id, cursors })
    }

    /// Multi-term OR over `[doc_id_start, doc_id_end)` against prebuilt
    /// cursors — the ranged fan-out's per-slice call;
    /// [`Self::search_or_range_pretokenized_with_floor`] delegates here.
    /// The ranged path carries no negation in v1.
    ///
    /// Runs the same windowed MaxScore union kernel as the single-shot path, so
    /// a query runs the identical kernel whether or not the fan-out sliced it —
    /// hardcoding a single-regime scorer here once caused an 11-24x post-compact
    /// broad-OR regression.
    #[cfg(test)]
    pub(crate) fn search_or_range_prebuilt(
        &self,
        set: &OrCursorSet,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        self.search_or_range_prebuilt_in_blob_ids(set, k, doc_id_start, doc_id_end, floor)
            .and_then(|h| self.hits_to_rows(h))
    }

    /// [`Self::search_or_range_prebuilt`] without the translation: the
    /// slice's top hits in blob ids, for [`Self::hits_to_rows_async`].
    pub(crate) fn search_or_range_prebuilt_in_blob_ids(
        &self,
        set: &OrCursorSet,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        floor: f32,
    ) -> Result<Vec<(FtsDocId, f32)>, FtsError> {
        if set.cursors.is_empty() || k == 0 || doc_id_start >= doc_id_end {
            return Ok(Vec::new());
        }
        let cursors = set.cursors.clone();
        // The range is a slice of the blob's id space, taken to split one
        // superfile across threads; it carries no row meaning, so a
        // reordered blob slices exactly as an unreordered one does.
        self.run_windowed_maxscore(
            set.column_id,
            cursors,
            k,
            None,
            floor.next_down(),
            doc_id_start,
            doc_id_end,
        )
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
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        // Tokenize the query with each column's configured tokenizer so
        // per-column analyzers are honored — a table may index different
        // columns with different analyzers.
        // FxHashMap: the combine does a per-doc insert across columns; the
        // default SipHash is needless work for small integer (doc-id) keys.
        // Each column's hits come back already translated into rows, so
        // the cross-column merge is keyed by rows, not blob ids.
        let mut combined: FxHashMap<RowId, f32> = FxHashMap::default();
        for (col_name, weight) in columns {
            let col_id = self.resolve_column_id(col_name)?;
            let tok = &self.columns[col_id as usize].tokenizer;
            let term_strings: Vec<String> = tok.tokenize(query).collect();
            let term_refs: Vec<&str> = term_strings.iter().map(|s| s.as_str()).collect();
            let per_col = self.search(col_name, &term_refs, usize::MAX, mode).await?;
            for (row, s) in per_col {
                *combined.entry(row).or_insert(0.0) += s * weight;
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
        idf: TermIdf,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<(Vec<(FtsDocId, f32)>, MatchWork, u64), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        // Resolve the term: from the open-wave memo when the df gather
        // prefetched it (global stats — the bytes below are the ones that
        // wave fetched), else via the dictionary.
        enum SingleSource {
            Absent,
            Inline { doc_id: u32, tf: u32 },
            Bytes { bytes: Bytes, short: bool },
        }
        let source = if let Some(entry) = prefetched.and_then(|m| m.lookup(term)) {
            match entry {
                None => SingleSource::Absent,
                Some(FetchedTermSlot::Inline { doc_id, tf }) => SingleSource::Inline { doc_id, tf },
                Some(FetchedTermSlot::Pfor { bytes, short, .. }) => {
                    SingleSource::Bytes { bytes, short }
                }
            }
        } else {
            let dict_bytes = self.dict_bytes_async().await?;
            let dict = Self::open_dict(&dict_bytes)?;
            let key = make_key(&col_meta.name, term);
            match dict.lookup(&key) {
                None => SingleSource::Absent,
                Some(packed) => match packed {
                    DictEntry::Inline { doc_id, tf } => SingleSource::Inline { doc_id, tf },
                    DictEntry::Pfor {
                        metadata_offset,
                        postings_length,
                        short,
                    } => {
                        // Fetch only this term's byte range (metadata header
                        // + skip table + blocks). The returned buffer starts
                        // at the metadata header, so all indexing below is
                        // relative to offset 0.
                        let mut fetched = self
                            .fetch_term_postings(&[(
                                metadata_offset as usize,
                                postings_length as usize,
                            )])
                            .await?;
                        SingleSource::Bytes {
                            bytes: fetched.pop().expect("one fetched range for one PFOR term"),
                            short,
                        }
                    }
                },
            }
        };
        if matches!(source, SingleSource::Absent) {
            // No norms are read for a term this superfile does not hold:
            // the fan-out reaches every superfile, and one without the term
            // costs no length-array bytes.
            return Ok((Vec::new(), MatchWork::default(), 0));
        }
        // Every scoring branch below reads the column's norms. Load them
        // here, on the caller's runtime — the prewarm every other scoring
        // shape makes in its cursor build — so no kernel reaches the
        // synchronous fallback in `ColumnMeta::norms`: on a lazy source that
        // fallback fetches the length array through the sync bridge from
        // whatever thread the kernel is on, and on an `infino-io` worker
        // that parks the runtime's own core on a storage read. This path had
        // no prewarm of its own, and two cold scored queries racing on one
        // column could park every core with nothing left to drive the read
        // that would release them.
        self.ensure_norms(column_id).await?;
        let term_bytes = match source {
            SingleSource::Absent => return Ok((Vec::new(), MatchWork::default(), 0)),
            SingleSource::Inline { doc_id, tf } => {
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
                let idf_t = idf.resolve(col_meta, 1)?;
                let idf_weight = idf_t;
                // Drop the lone match if a negated term excludes it.
                // The inline slot read no postings-region bytes; the
                // work-stats byte count for this path is genuinely zero.
                if let Some(f) = filter.as_deref_mut()
                    && !f.admits(doc_id)
                {
                    return Ok((Vec::new(), MatchWork::default(), 0));
                }
                let dl_norm_k1 = col_meta.dl_norm_k1().get(doc_id);
                let score = bm25::score_with_dl_norm_k1(idf_weight, tf, dl_norm_k1);
                if score <= floor_eff {
                    return Ok((Vec::new(), MatchWork::default(), 0));
                }
                return Ok((
                    vec![(FtsDocId::new(doc_id), score)],
                    MatchWork::default(),
                    0,
                ));
            }
            SingleSource::Bytes { bytes, short: true } => {
                // Short-form term: one block already, so there is nothing
                // to skip — decode the body and score every posting.
                let kernel_start = metering_active().then(thread_cpu_ns).flatten();
                let mut buf_d = [0u32; BLOCK_LEN];
                let mut buf_t = [0u32; BLOCK_LEN];
                let decoded =
                    decode_short(bytes.as_ref(), col_meta.positions, &mut buf_d, &mut buf_t)
                        .ok_or_else(|| {
                            FtsError::Read(ReadError::Malformed(
                                "short-form term body does not decode".into(),
                            ))
                        })?;
                let idf_weight = idf.resolve(col_meta, decoded.n as u64)?;
                let dl_norm_k1 = col_meta.dl_norm_k1();
                let mut heap: BinaryHeap<TopKEntry> =
                    BinaryHeap::with_capacity(k.min(decoded.n).max(1));
                for j in 0..decoded.n {
                    let doc_id = buf_d[j];
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(doc_id)
                    {
                        continue;
                    }
                    let score =
                        bm25::score_with_dl_norm_k1(idf_weight, buf_t[j], dl_norm_k1.get(doc_id));
                    if score <= floor_eff {
                        continue;
                    }
                    if heap.len() < k {
                        heap.push(TopKEntry(score, FtsDocId::new(doc_id)));
                    } else if let Some(mut worst) = heap.peek_mut()
                        && score > worst.0
                    {
                        *worst = TopKEntry(score, FtsDocId::new(doc_id));
                    }
                }
                return Ok((
                    drain_top_k_desc(heap),
                    MatchWork {
                        postings_bytes: bytes.len() as u64,
                        planned_ranges: 1,
                        kernel_cpu_ns: 0,
                    },
                    thread_cpu_delta_ns(kernel_start),
                ));
            }
            SingleSource::Bytes {
                bytes,
                short: false,
            } => bytes,
        };
        let postings = term_bytes.as_ref();

        // Everything below is the synchronous scoring walk (no awaits):
        // bracket it on this thread for the per-query kernel CPU stat.
        // Gated: an unmetered process must not pay the procfs reads on
        // the most common query shape.
        let kernel_start = metering_active().then(thread_cpu_ns).flatten();
        let term_meta = TermMeta::parse(postings, 0, col_meta.positions)?;

        let local_idf = bm25::idf(col_meta.scored_doc_count(), term_meta.df);
        let idf_t = idf.resolve(col_meta, term_meta.df)?;
        let idf_weight = idf_t;
        // Every stored bound — per block and per span — is decoded at the
        // same idf and statistics the scores below use, so the skip tests
        // compare like with like. See `BoundDecoder`.
        let bounds = BoundDecoder::new(col_meta, idf_t, local_idf);
        let dl_norm_k1 = col_meta.dl_norm_k1();

        let coarse_span = format::fts::COARSE_BLOCK_MAX_SPAN;
        // Every long-form term carries a coarse block-max table.
        let coarse_enabled = term_meta.has_coarse;

        // Threshold-first seed. The doc-id-order walk below raises its k-th
        // best slowly, so on a common term it decodes most of its blocks
        // before the bar has risen enough to skip them. To raise the bar
        // first, decode the highest-block-max blocks up front
        // — found cheaply via the coarse level — and take the k-th score of
        // those as a floor for the main walk's skip bar.
        //
        // Correctness: `seed_threshold` is the k-th of a *subset* of docs, so
        // it is a true lower bound on the final k-th; skipping a block whose
        // block-max is *strictly* below it drops only docs that score below a
        // lower bound on the k-th — never a top-k or tied-at-k doc. The seed
        // does not touch `heap`; the main walk re-decodes any seed block it
        // needs and admits in doc-id order, so the top-k (and its tie-break)
        // is bit-identical to the un-seeded walk. Only applied without a
        // negation filter (so the seed's admitted set matches the final one)
        // and on lists long enough to have waste to prune.
        let seed_threshold: f32 =
            if coarse_enabled && filter.is_none() && term_meta.num_blocks >= seed::MIN_BLOCKS {
                let num_coarse = term_meta.num_blocks.div_ceil(coarse_span);
                // Seed depth scales with k: a fixed handful of blocks gives a
                // tight k-th only at small k; at k=100/1000 the top-k spreads
                // across more blocks, so decode more to raise the threshold.
                let m_want = seed_blocks_for(k);
                // Gather enough top spans to contain the top `m_want` blocks
                // (over-cover: worst case each is in its own span).
                let mut spans: Vec<(f32, usize)> = (0..num_coarse)
                    .map(|g| (bounds.bound(term_meta.coarse_slot(postings, g)), g))
                    .collect();
                let s = m_want.max(seed::TOP_SPANS).min(spans.len());
                spans.select_nth_unstable_by(s.saturating_sub(1), |a, b| b.0.total_cmp(&a.0));
                // Candidate blocks from those spans, then the top blocks by
                // block-max across them.
                let mut cand: Vec<(f32, usize)> = Vec::with_capacity(s * coarse_span);
                for &(_, g) in &spans[..s] {
                    let start = g * coarse_span;
                    let end = (start + coarse_span).min(term_meta.num_blocks);
                    for bi in start..end {
                        let (_, raw) = term_meta.skip_entry(postings, bi);
                        cand.push((bounds.bound(raw), bi));
                    }
                }
                let m = m_want.min(cand.len());
                cand.select_nth_unstable_by(m.saturating_sub(1), |a, b| b.0.total_cmp(&a.0));
                // Decode those blocks, score each whole, and take the k-th best of
                // those scores: a heap at a small `k`, a buffer cut by selection
                // at a large one, where a sift per admission would cost more than
                // the walk the seed saves. Clamp the preallocation to the corpus
                // size: `k` is caller-controlled and may be `usize::MAX`.
                let mut blocks = cand[..m].iter();
                let decode = |ids: &mut [u32; BLOCK_LEN], tfs: &mut [u32; BLOCK_LEN]| {
                    let &(_, bi) = blocks.next()?;
                    let range = term_meta.block_range_in_term(postings, bi, None);
                    let bytes = &postings[range.start..range.end];
                    let hdr = BlockHeader::parse(
                        bytes,
                        term_meta.prev_last_doc_id(postings, bi),
                        self.wide_blocks(),
                    );
                    Some(decode_block(bytes, &hdr, ids, tfs))
                };
                let cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
                match k >= BUFFERED_TOPK_MIN_K {
                    true => {
                        let buffer = <TopKBuffer as Collector>::with_capacity(k, cap);
                        kth_scored(buffer, k, floor_eff, idf_weight, dl_norm_k1, decode)
                    }
                    false => {
                        let heap = <BinaryHeap<TopKEntry> as Collector>::with_capacity(k, cap);
                        kth_scored(heap, k, floor_eff, idf_weight, dl_norm_k1, decode)
                    }
                }
            } else {
                f32::NEG_INFINITY
            };

        let capacity = k.min(term_meta.num_blocks * BLOCK_LEN).max(1);
        let walk = SingleTermWalk {
            term_meta: &term_meta,
            postings,
            bounds: &bounds,
            idf_weight,
            dl_norm_k1,
            k,
            floor_eff,
            seed_threshold,
            coarse_enabled,
            wide_blocks: self.wide_blocks(),
        };
        // At a large `k` most docs the walk scores are admitted, and a heap's
        // sift per admission dominates; the buffer pays an append instead.
        let hits = match k >= BUFFERED_TOPK_MIN_K {
            true => walk.run(TopKBuffer::with_capacity(k, capacity), filter),
            false => walk.run(
                <BinaryHeap<TopKEntry> as Collector>::with_capacity(k, capacity),
                filter,
            ),
        };

        Ok((
            hits,
            MatchWork {
                postings_bytes: term_bytes.len() as u64,
                planned_ranges: 1,
                // The walk's ns travel in the tuple's third element.
                kernel_cpu_ns: 0,
            },
            thread_cpu_delta_ns(kernel_start),
        ))
    }

    /// Build one `TermCursor` per term that resolves in the term dictionary.
    /// Missing terms (dictionary miss) are silently dropped — fine for OR
    /// semantics where a missing term contributes nothing. Returned
    /// `Vec` may be empty (all terms missed) or shorter than `terms`.
    ///
    /// A thin wrapper over [`Self::build_term_cursors_opt`] that discards
    /// the per-term slot positions; callers that need to know *which*
    /// term missed (bare-atom builds, which keep a `None` slot for it)
    /// use the `_opt` form directly.
    pub(super) async fn build_term_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
        idf: QueryIdf<'_>,
        purpose: CursorUse,
        qtf: Option<&[u32]>,
        prefetched: Option<&FetchedTermMemo>,
    ) -> Result<Vec<TermCursor>, FtsError> {
        Ok(self
            .build_term_cursors_opt(column_id, terms, idf, purpose, qtf, prefetched, None)
            .await?
            .into_iter()
            .flatten()
            .collect())
    }

    /// Open-wave fetch for table-wide idf: resolve `terms` in this
    /// superfile's dictionary and fetch each PFOR term's full postings
    /// range — exactly the reads the cursor build performs — recording
    /// df per term so the caller can aggregate corpus-wide idf BEFORE
    /// any walk starts. The returned memo travels back into the walk
    /// wave, whose cursor builds then issue no byte-source reads for
    /// these terms (see the `prefetched` parameter on
    /// [`Self::build_term_cursors_opt`]).
    ///
    /// Work accounting stays with the walk wave: the cursors built from
    /// the memo report the same posting bytes and planned ranges they
    /// would have reported fetching for themselves, so per-query work
    /// stats under global stats equal the per-superfile plan — the
    /// whole point of fusing the gather into this wave.
    pub(crate) async fn fetch_scored_terms(
        &self,
        column: &str,
        terms: &[&str],
    ) -> Result<FetchedTermMemo, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let col_meta = &self.columns[column_id as usize];
        let dict_bytes = self.dict_bytes_async().await?;
        let dict = Self::open_dict(&dict_bytes)?;

        // Resolve every term, collecting the PFOR ranges for one
        // coalesced fetch (mirrors `build_term_cursors_opt`).
        enum Pre {
            Inline { doc_id: u32, tf: u32 },
            Pfor { short: bool, metadata_offset: u64 },
        }
        let mut resolved: Vec<(Box<str>, Option<Pre>)> = Vec::with_capacity(terms.len());
        let mut pfor_offsets: Vec<(usize, usize)> = Vec::new();
        for term in terms {
            let key = make_key(&col_meta.name, term);
            let pre = dict.lookup(&key).map(|packed| match packed {
                DictEntry::Inline { doc_id, tf } => Pre::Inline { doc_id, tf },
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => {
                    pfor_offsets.push((metadata_offset as usize, postings_length as usize));
                    Pre::Pfor {
                        short,
                        metadata_offset,
                    }
                }
            });
            resolved.push((Box::from(*term), pre));
        }
        let pfor_bytes = self.fetch_term_postings(&pfor_offsets).await?;
        let mut pfor_iter = pfor_bytes.into_iter();

        let mut map: HashMap<Box<str>, Option<FetchedTermSlot>> =
            HashMap::with_capacity(resolved.len());
        for (term, pre) in resolved {
            let slot = match pre {
                None => None,
                Some(Pre::Inline { doc_id, tf }) => Some(FetchedTermSlot::Inline { doc_id, tf }),
                Some(Pre::Pfor {
                    short,
                    metadata_offset,
                }) => {
                    let bytes = pfor_iter.next().expect("one fetched range per PFOR term");
                    // df heads the fetched range in either form — the long
                    // form's metadata header, the short body's leading
                    // varint (CPU-only, no extra read).
                    let df = match short {
                        true => u64::from(short_df(bytes.as_ref()).ok_or_else(|| {
                            FtsError::Read(ReadError::Malformed(
                                "short-form term body does not decode".into(),
                            ))
                        })?),
                        false => TermMeta::parse(bytes.as_ref(), 0, col_meta.positions)?.df,
                    };
                    Some(FetchedTermSlot::Pfor {
                        bytes,
                        df,
                        short,
                        metadata_offset,
                    })
                }
            };
            map.insert(term, slot);
        }
        Ok(FetchedTermMemo { map })
    }

    /// Build a `TermCursor` for every term, **preserving input order and
    /// arity**: the result has one slot per input term, `None` where the term
    /// is absent from the term dictionary. One dictionary open and one parallel
    /// postings fan-out for the whole batch — so a multi-term build costs a
    /// single dictionary fetch and a single overlapped range wave, not one of
    /// each per term. `qtf`, when present, is a per-term query-term-frequency
    /// weight (parallel to `terms`): each built cursor folds its weight into
    /// the effective idf, which scales both the score and the BlockMaxWAND skip
    /// ceilings, so a deduplicated repeated term ranks identically to the
    /// duplicates it replaced (BM25 is linear in idf). `None` leaves idf
    /// unweighted. `dict_bytes`, when present, is this reader's dictionary
    /// already fetched by the caller, so it is not fetched again.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn build_term_cursors_opt(
        &self,
        column_id: u32,
        terms: &[&str],
        idf: QueryIdf<'_>,
        purpose: CursorUse,
        qtf: Option<&[u32]>,
        prefetched: Option<&FetchedTermMemo>,
        dict_bytes: Option<&[u8]>,
    ) -> Result<Vec<Option<TermCursor>>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        // Scoring needs the column's norms; an unscored build does not,
        // and must not read the length array for them.
        if purpose.scores() {
            self.ensure_norms(column_id).await?;
        }

        // Resolve each term to an inline (df=1) value, a PFOR metadata
        // offset, or a miss — preserving query order and arity (a miss is a
        // `None` slot, so the caller can tell which term was absent). A term
        // the open-wave memo already resolved (global stats) is served from
        // it — its bytes were fetched when its df was gathered — so the
        // dictionary is opened, and the PFOR fan below issued, only for the
        // terms the memo does not cover (all of them on the single-pass
        // path). Each resolved entry carries its term's idf source, which the
        // cursor resolves when it scores.
        enum Resolved {
            Memo {
                slot: FetchedTermSlot,
                gidf: TermIdf,
            },
            Inline {
                doc_id: u32,
                tf: u32,
                gidf: TermIdf,
            },
            Pfor {
                gidf: TermIdf,
                short: bool,
                metadata_offset: u64,
            },
        }
        let all_prefetched = prefetched.is_some_and(|m| terms.iter().all(|t| m.contains(t)));
        let fetched;
        let dict = match (all_prefetched, dict_bytes) {
            (true, _) => None,
            (false, Some(b)) => Some(Self::open_dict(b)?),
            (false, None) => {
                fetched = self.dict_bytes_async().await?;
                Some(Self::open_dict(&fetched)?)
            }
        };
        let mut resolved: Vec<Option<Resolved>> = Vec::with_capacity(terms.len());
        let mut pfor_offsets: Vec<(usize, usize)> = Vec::new();
        for term in terms {
            let gidf = idf.term(term);
            if let Some(entry) = prefetched.and_then(|m| m.lookup(term)) {
                resolved.push(entry.map(|slot| Resolved::Memo { slot, gidf }));
                continue;
            }
            let dict = dict
                .as_ref()
                .expect("dictionary opened for un-memoed terms");
            let key = make_key(&col_meta.name, term);
            let Some(packed) = dict.lookup(&key) else {
                resolved.push(None);
                continue;
            };
            match packed {
                DictEntry::Inline { doc_id, tf } => {
                    resolved.push(Some(Resolved::Inline { doc_id, tf, gidf }));
                }
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => {
                    pfor_offsets.push((metadata_offset as usize, postings_length as usize));
                    resolved.push(Some(Resolved::Pfor {
                        gidf,
                        short,
                        metadata_offset,
                    }));
                }
            }
        }

        let pfor_bytes = self.fetch_term_postings(&pfor_offsets).await?;
        let mut pfor_iter = pfor_bytes.into_iter();

        let mut cursors: Vec<Option<TermCursor>> = Vec::with_capacity(resolved.len());
        for (i, r) in resolved.into_iter().enumerate() {
            // Per-term query-term-frequency weight (1 when unweighted).
            let weight = qtf.map_or(1, |w| w[i]);
            match r {
                None => cursors.push(None),
                // Memo-served slots build from the open wave's bytes and
                // report the same posting bytes / planned ranges they
                // would have reported fetching for themselves, so work
                // stats stay equal to the per-superfile plan (the walk
                // wave is the only flusher).
                Some(Resolved::Memo {
                    slot: FetchedTermSlot::Inline { doc_id, tf },
                    gidf,
                }) => {
                    let tf = col_meta.inline_tf(tf);
                    cursors.push(Some(TermCursor::new_inline(
                        doc_id, tf, col_meta, gidf, weight, purpose,
                    )?));
                }
                Some(Resolved::Memo {
                    slot:
                        FetchedTermSlot::Pfor {
                            bytes,
                            short,
                            metadata_offset,
                            ..
                        },
                    gidf,
                }) => {
                    cursors.push(Some(TermCursor::for_body(
                        bytes,
                        short,
                        col_meta,
                        gidf,
                        weight,
                        purpose,
                        Some((&self.block_tables, metadata_offset)),
                    )?));
                }
                Some(Resolved::Inline { doc_id, tf, gidf }) => {
                    // Scoring must use the implied tf, never the slot. (Phrase
                    // members recover the position itself with their own
                    // dictionary lookup — see `build_atom_cursors`.)
                    let tf = col_meta.inline_tf(tf);
                    let cursor =
                        TermCursor::new_inline(doc_id, tf, col_meta, gidf, weight, purpose)?;
                    cursors.push(Some(cursor));
                }
                Some(Resolved::Pfor {
                    gidf,
                    short,
                    metadata_offset,
                }) => {
                    let term_bytes = pfor_iter.next().expect("one fetched range per PFOR term");
                    cursors.push(Some(TermCursor::for_body(
                        term_bytes,
                        short,
                        col_meta,
                        gidf,
                        weight,
                        purpose,
                        Some((&self.block_tables, metadata_offset)),
                    )?));
                }
            }
        }
        Ok(cursors)
    }
}

/// The single-term walk's inputs once the term is parsed and the seed
/// taken: the coarse and per-block skips and the per-doc admission, generic
/// over the top-k collector.
struct SingleTermWalk<'a> {
    term_meta: &'a TermMeta,
    postings: &'a [u8],
    bounds: &'a BoundDecoder,
    idf_weight: f32,
    dl_norm_k1: &'a NormTable,
    k: usize,
    floor_eff: f32,
    seed_threshold: f32,
    coarse_enabled: bool,
    /// The file's block header reading (see `ColumnMeta::wide_blocks`).
    wide_blocks: bool,
}

impl SingleTermWalk<'_> {
    /// Walk the term's blocks in doc order into `heap` and return its best
    /// `k`. The bar is the collector's: a heap's k-th best, or a buffer's
    /// k-th best at its last cut, never above the true one.
    fn run<C: Collector>(
        &self,
        mut heap: C,
        mut filter: Option<&mut ExcludeFilter>,
    ) -> Vec<(FtsDocId, f32)> {
        let Self {
            term_meta,
            postings,
            bounds,
            idf_weight,
            dl_norm_k1,
            k,
            floor_eff,
            seed_threshold,
            coarse_enabled,
            wide_blocks,
        } = *self;
        let coarse_span = format::fts::COARSE_BLOCK_MAX_SPAN;
        let mut buf_d = [0u32; BLOCK_LEN];
        let mut buf_t = [0u32; BLOCK_LEN];
        let mut scores: Vec<f32> = Vec::with_capacity(BLOCK_LEN);
        let mut threshold = f32::NEG_INFINITY;
        // Two skip levels. The coarse table bounds a whole span of
        // `COARSE_BLOCK_MAX_SPAN` blocks with a single entry, so a span
        // the running k-th-best already dominates is jumped in one
        // comparison instead of one skip-entry read per block. On a long,
        // heavily-skipped list (a common term at small k) that per-block
        // scan is the dominant cost; the coarse level removes ~31/32 of
        // it. Inside a span that might qualify, the walk falls back to
        // the per-block bar — identical decisions, identical top-k.
        let mut i = 0usize;
        // Where the previous block ended, so the length-coded skip table
        // yields this block's range in one read; `None` after a span jump.
        let mut prev_end: Option<usize> = None;
        while i < term_meta.num_blocks {
            if coarse_enabled && i.is_multiple_of(coarse_span) {
                let coarse_max = bounds.bound(term_meta.coarse_slot(postings, i / coarse_span));
                let span_end = (i + coarse_span).min(term_meta.num_blocks);
                // Seed skip, span-wide (strict): no block in the span can
                // reach the seeded lower bound on the k-th, so none can be
                // top-k. Strict `<` keeps a block exactly at the bound in
                // play, preserving the tie-break.
                if coarse_max < seed_threshold {
                    i = span_end;
                    prev_end = None;
                    continue;
                }
                // Floor skip, span-wide: nothing in the span reaches the
                // caller's floor.
                if coarse_max <= floor_eff {
                    i = span_end;
                    prev_end = None;
                    continue;
                }
                // BMW skip, span-wide: heap full AND no block in the span
                // can beat the kth-best.
                if heap.is_full(k) && coarse_max <= threshold {
                    i = span_end;
                    prev_end = None;
                    continue;
                }
            }

            // last_doc_id (first tuple slot) is unused here — it serves
            // AND-merge seeks, which single-term never does.
            let (_, raw) = term_meta.skip_entry(postings, i);
            let block_max_bm25 = bounds.bound(raw);
            let range = term_meta.block_range_in_term(postings, i, prev_end);
            prev_end = Some(range.end);

            // Seed skip (strict): block can't reach the seeded lower bound
            // on the k-th, so it holds no top-k doc.
            if block_max_bm25 < seed_threshold {
                i += 1;
                continue;
            }
            // Floor skip: nothing in this block can reach the caller's
            // floor — dead regardless of local heap state.
            if block_max_bm25 <= floor_eff {
                i += 1;
                continue;
            }
            // BMW skip: heap full AND this block can't beat the kth-best.
            if heap.is_full(k) && block_max_bm25 <= threshold {
                i += 1;
                continue;
            }

            // Locate the block's bytes.
            let block_bytes = &postings[range.start..range.end];
            let hdr = BlockHeader::parse(
                block_bytes,
                term_meta.prev_last_doc_id(postings, i),
                wide_blocks,
            );

            //  Actual number of real docs in that block.
            let n = decode_block(block_bytes, &hdr, &mut buf_d, &mut buf_t);

            // The whole block scored eight postings at a time, the same bits
            // as one at a time; the per-doc tests below then read the scores.
            score_postings(
                idf_weight,
                &buf_d[..n],
                &buf_t[..n],
                dl_norm_k1,
                &mut scores,
            );
            for (&doc_id, &score) in buf_d[..n].iter().zip(&scores) {
                // Floor gate: strictly-below-floor docs are dead to the
                // caller; keeping them out also keeps the heap's min
                // (the BMW skip bar) honest.
                if score <= floor_eff {
                    continue;
                }
                if heap.is_full(k) && score <= threshold {
                    continue;
                }
                // Drop docs excluded by a negated term (None = keep all). Asked
                // only of a doc that would enter the heap: one that would not
                // is dropped either way, and the docs asked still ascend.
                if let Some(f) = filter.as_deref_mut()
                    && !f.admits(doc_id)
                {
                    continue;
                }
                heap.offer(k, &mut threshold, score, doc_id);
            }
            i += 1;
        }
        heap.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Arc,
    };

    use bytes::Bytes;
    use roaring::RoaringBitmap;

    use super::{super::test_util::*, *};
    use crate::superfile::fts::{builder::FtsBuilder, tokenize::Phrase};

    /// A `k` above the fixture's match count.
    const TABLE_IDF_K: usize = 10;
    /// An arbitrary table-wide idf to score with.
    const TABLE_IDF: f32 = 1.5;
    /// Score slack between the ranked and the exact-count walks: they add
    /// a doc's term scores in different orders.
    const SUM_ORDER_EPS: f32 = 1e-4;
    /// Rows in the reordered fixture: a whole number of bitset words, and a
    /// count that leaves the last word part full.
    const ORDER_FIXTURE_ROWS: [u32; 2] = [4096, 4101];
    /// Every this many rows carry `beta`: a match dense enough for the bitset.
    const BETA_EVERY: u32 = 3;
    /// Every this many rows carry `rare`: a match sparse enough to be sorted.
    const RARE_EVERY: u32 = 128;

    /// An unranked match on a reordered blob comes back as its rows,
    /// ascending, whether it is dense enough to be put in order through a
    /// bitset or sparse enough to be sorted, at a row count that fills the
    /// bitset's last word and at one that does not.
    #[tokio::test]
    async fn unranked_rows_come_back_ascending_at_any_density() {
        for n_rows in ORDER_FIXTURE_ROWS {
            let docs: Vec<String> = (0..n_rows)
                .map(|row| {
                    let mut text = String::from("alpha");
                    if row.is_multiple_of(BETA_EVERY) {
                        text.push_str(" beta");
                    }
                    if row.is_multiple_of(RARE_EVERY) {
                        text.push_str(" rare");
                    }
                    text
                })
                .collect();
            let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            // Fisher-Yates, so the blob's order is a genuine permutation.
            let mut next = xorshift(u64::from(n_rows));
            let mut order: Vec<u32> = (0..n_rows).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, (next() % (i as u64 + 1)) as usize);
            }
            let (blob, json) = build_standard_blob_with(&refs, Some(&order));
            let r = FtsReader::open(blob, &json).expect("open");
            assert!(r.has_doc_map(), "the fixture must be reordered");

            // The fixture takes both paths: `rare` is sorted, `beta` and
            // `alpha` go through the bitset.
            let rows = n_rows as usize;
            let rare = n_rows.div_ceil(RARE_EVERY) as usize;
            let beta = n_rows.div_ceil(BETA_EVERY) as usize;
            assert!(rare * ROW_ORDER_BITSET_MIN_DENSITY < rows);
            assert!(beta * ROW_ORDER_BITSET_MIN_DENSITY >= rows);

            for (term, every) in [("alpha", 1), ("beta", BETA_EVERY), ("rare", RARE_EVERY)] {
                let want: Vec<u32> = (0..n_rows)
                    .filter(|row| row.is_multiple_of(every))
                    .collect();
                let got: Vec<u32> = r
                    .token_match("body", &[term], BoolMode::Or)
                    .await
                    .expect("token match")
                    .0
                    .into_iter()
                    .map(RowId::get)
                    .collect();
                assert_eq!(got, want, "{n_rows} rows, `{term}`");
            }
        }
    }

    /// Doc-id cuts of the slice fixture: the first doc, one inside the first
    /// posting block, block-straddling ones, and the end.
    const SLICE_CUTS: [u32; 6] = [1, 100, 1000, 2047, 2048, 4000];

    /// Cut into doc-id slices at any boundary, an unranked `Or` match on a
    /// reordered blob gives back exactly the whole walk's rows: each slice
    /// ascending, together every row once. One term (its blocks copied
    /// whole) and two (the union walk), dense and sparse.
    #[tokio::test]
    async fn sliced_unranked_match_rows_are_the_whole_walk() {
        let n_rows = ORDER_FIXTURE_ROWS[1];
        let docs: Vec<String> = (0..n_rows)
            .map(|row| {
                let mut text = String::from("alpha");
                if row.is_multiple_of(BETA_EVERY) {
                    text.push_str(" beta");
                }
                if row.is_multiple_of(RARE_EVERY) {
                    text.push_str(" rare");
                }
                text
            })
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let mut next = xorshift(u64::from(n_rows));
        let mut order: Vec<u32> = (0..n_rows).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let r = FtsReader::open(blob, &json).expect("open");
        assert!(r.has_doc_map(), "the fixture must be reordered");

        let mut bounds = vec![0];
        bounds.extend(SLICE_CUTS);
        bounds.push(n_rows);
        for terms in [
            &["alpha"][..],
            &["rare"],
            &["beta", "rare"],
            &["rare", "absent"],
        ] {
            let (whole, _) = r
                .token_match("body", terms, BoolMode::Or)
                .await
                .expect("whole walk");
            let (set, _) = r
                .match_cursor_set("body", terms, None)
                .await
                .expect("cursor set");
            let mut sliced: Vec<RowId> = Vec::new();
            for pair in bounds.windows(2) {
                let (rows, _) = r
                    .token_match_range_prebuilt(&set, pair[0], pair[1])
                    .expect("slice");
                assert!(rows.is_sorted(), "{terms:?}: slice {pair:?} ascending");
                sliced.extend(rows);
            }
            sliced.sort_unstable();
            assert_eq!(sliced, whole, "{terms:?}");
        }
    }

    /// Rows of the blob the row-order cases below are put in order over.
    const ORDER_CASE_ROWS: usize = 8;

    fn row_ids(rows: &[u32]) -> Vec<RowId> {
        rows.iter().copied().map(RowId::new).collect()
    }

    /// Rows the map already left ascending come back as they are; out of
    /// order, dense rows go through the bitset and sparse ones are sorted,
    /// to the same answer.
    #[test]
    fn rows_come_back_ascending_down_every_path() {
        let ascending = row_ids(&[1, 3, 4, 6]);
        assert_eq!(
            into_row_order(ascending.clone(), ORDER_CASE_ROWS),
            ascending
        );
        // Dense enough for the bitset at this blob size.
        assert!(ascending.len() * ROW_ORDER_BITSET_MIN_DENSITY >= ORDER_CASE_ROWS);
        assert_eq!(
            into_row_order(row_ids(&[6, 1, 4, 3]), ORDER_CASE_ROWS),
            ascending
        );
        // Too sparse for the bitset over a blob this many times larger.
        let large = ORDER_CASE_ROWS * ROW_ORDER_BITSET_MIN_DENSITY * ROW_ORDER_BITSET_MIN_DENSITY;
        assert!(ascending.len() * ROW_ORDER_BITSET_MIN_DENSITY < large);
        assert_eq!(into_row_order(row_ids(&[6, 1, 4, 3]), large), ascending);
    }

    /// A corrupt doc-id map, naming a row past the blob's end or one row
    /// twice, is refused by the bitset rather than indexed past it or
    /// silently shortened; the caller's sort returns the rows as they are,
    /// as the sort path always did, instead of failing the query.
    #[test]
    fn a_corrupt_map_falls_back_to_the_sort() {
        let past_the_end = row_ids(&[6, 1, ORDER_CASE_ROWS as u32 + 1, 3]);
        assert_eq!(rows_in_order(&past_the_end, ORDER_CASE_ROWS), None);
        assert_eq!(
            into_row_order(past_the_end, ORDER_CASE_ROWS),
            row_ids(&[1, 3, 6, ORDER_CASE_ROWS as u32 + 1])
        );
        let twice = row_ids(&[6, 1, 6, 3]);
        assert_eq!(rows_in_order(&twice, ORDER_CASE_ROWS), None);
        assert_eq!(
            into_row_order(twice, ORDER_CASE_ROWS),
            row_ids(&[1, 3, 6, 6])
        );
    }

    /// The two walks return the same top-k up to float summation order:
    /// every doc's score agrees within `SUM_ORDER_EPS`, and a doc may be
    /// on one side only when it ties the k-th score within that slack.
    fn assert_same_top_k(counted: &[(RowId, f32)], pruned: &[(RowId, f32)], label: &str) {
        assert_eq!(counted.len(), pruned.len(), "len for {label}");
        let kth = pruned.last().map_or(f32::NEG_INFINITY, |h| h.1);
        let scores = |hits: &[(RowId, f32)]| hits.iter().copied().collect::<HashMap<RowId, f32>>();
        let (a, b) = (scores(counted), scores(pruned));
        for (x, y) in [(&a, &b), (&b, &a)] {
            for (doc, s) in x {
                match y.get(doc) {
                    Some(t) => assert!(
                        (s - t).abs() < SUM_ORDER_EPS,
                        "score of {doc:?} for {label}: {s} vs {t}"
                    ),
                    None => assert!(
                        (s - kth).abs() < SUM_ORDER_EPS,
                        "{doc:?} on one side only and not a k-th tie for {label}: {s} vs {kth}"
                    ),
                }
            }
        }
    }

    /// Many-term ORs, where the ranked and exact-count walks add a doc's
    /// term scores in different orders: the top-k must agree up to that
    /// order, and the count must be exact.
    #[tokio::test]
    async fn exact_count_matches_ranked_on_random_many_term_ors() {
        let mut seed = 7u64;
        let mut rng = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        let vocab: Vec<String> = (0..24).map(|i| format!("w{i}")).collect();
        for trial in 0..6u32 {
            let n_docs = 1500 + (rng() % 2000) as u32;
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..n_docs {
                let mut text = String::new();
                // Early terms are common, late ones rare.
                for (t, term) in vocab.iter().enumerate() {
                    if rng().is_multiple_of(3 + t as u64) {
                        for _ in 0..1 + rng() % 3 {
                            text.push_str(term);
                            text.push(' ');
                        }
                    }
                }
                if text.is_empty() {
                    text.push_str(&vocab[0]);
                }
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(blob, json).expect("open");
            for q in 0..25u32 {
                let n_terms = 2 + (rng() % 19) as usize;
                let shoulds: Vec<&str> = (0..n_terms)
                    .map(|_| vocab[(rng() % 24) as usize].as_str())
                    .collect();
                let negatives: Vec<&str> = match rng() % 3 {
                    0 => vec![vocab[(rng() % 24) as usize].as_str()],
                    _ => Vec::new(),
                };
                let lists = || ClauseLists {
                    shoulds: &shoulds,
                    negatives: &negatives,
                    ..ClauseLists::default()
                };
                let label = format!("trial {trial} q {q} {shoulds:?} -{negatives:?}");
                let matches = r
                    .search_excluding("body", lists(), n_docs as usize, f32::NEG_INFINITY)
                    .await
                    .expect("every match")
                    .len() as u64;
                for k in [1usize, 10, 100] {
                    let pruned = r
                        .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                        .await
                        .expect("pruned search");
                    let (hits, count) = r
                        .search_excluding_with_count("body", lists(), k)
                        .await
                        .expect("counted search");
                    assert_same_top_k(&hits, &pruned, &format!("{label} k={k}"));
                    assert_eq!(count, matches, "count for {label} k={k}");
                }
            }
        }
    }

    /// The exact-count walk on a corpus large enough that whole blocks and
    /// windows fall under the k-th best, which it must still score and count:
    /// a single near-every-doc term, and unions of it with mid and rare
    /// terms. The top-k must agree with the pruned search and the count must
    /// be exact. Run over two corpora: one uniform, and one front-loaded,
    /// whose strong documents all sit in its first windows, so the later
    /// windows' block maxima together stay under the k-th best.
    #[tokio::test]
    async fn exact_count_is_exact_where_blocks_and_windows_cannot_beat_the_kth_best() {
        let mut x: u64 = 0xA076_1D64_78BD_642F;
        let mut rng = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let n_docs = 40_000u32;
        for front_loaded in [false, true] {
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..n_docs {
                // Front-loaded: past the first quarter every document is weak,
                // one occurrence of each term in a long body.
                let weak = front_loaded && i >= n_docs / 4;
                let mut text = String::from("doc");
                if rng() % 100 < 90 {
                    let reps = if weak { 1 } else { 1 + rng() % 4 };
                    for _ in 0..reps {
                        text.push_str(" common");
                    }
                }
                if rng() % 100 < 12 {
                    text.push_str(" mid");
                }
                if rng() % 1000 < 7 {
                    text.push_str(if weak { " rare" } else { " rare rare" });
                }
                let filler = if weak { 40 + rng() % 30 } else { rng() % 30 };
                for _ in 0..filler {
                    text.push_str(" filler");
                }
                b.add_doc(0, i, &text).expect("add doc");
            }
            let blob = Bytes::from(b.finish().expect("finish"));
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(blob, json).expect("open");
            let queries: [&[&str]; 5] = [
                &["common"],
                &["mid"],
                &["common", "mid"],
                &["common", "mid", "rare"],
                &["mid", "rare"],
            ];
            for shoulds in queries {
                let lists = || ClauseLists {
                    shoulds,
                    ..ClauseLists::default()
                };
                let matches = r
                    .search_excluding("body", lists(), n_docs as usize, f32::NEG_INFINITY)
                    .await
                    .expect("every match")
                    .len() as u64;
                for k in [1usize, 10, 100] {
                    let pruned = r
                        .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                        .await
                        .expect("pruned search");
                    let (hits, count) = r
                        .search_excluding_with_count("body", lists(), k)
                        .await
                        .expect("counted search");
                    let label = format!("{shoulds:?} k={k} front_loaded={front_loaded}");
                    assert_same_top_k(&hits, &pruned, &label);
                    assert_eq!(count, matches, "count for {label}");
                }
            }
        }
    }

    /// For every term clause shape, the exact-count walk returns the
    /// pruned search's top-k, and a count equal both to the number of
    /// matches and to the unranked count path's answer.
    #[tokio::test]
    async fn exact_count_matches_pruned_hits_and_unranked_count() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        // (musts, shoulds, negatives) over the 3-doc `body` corpus.
        let shapes: &[(&[&str], &[&str], &[&str])] = &[
            (&[], &["rust"], &[]),
            (&["rust"], &[], &[]),
            (&[], &["rust", "java"], &[]),
            (&["rust", "runtime"], &[], &[]),
            (&["rust"], &["tokio"], &[]),
            (&[], &["rust", "java"], &["tokio"]),
            (&["rust"], &[], &["tokio"]),
            (&["rust", "zzz"], &[], &[]),
            (&["rust", "zzz"], &[], &["tokio"]),
            (&[], &["zzz"], &[]),
        ];
        for &(musts, shoulds, negatives) in shapes {
            let lists = || ClauseLists {
                musts,
                shoulds,
                negatives,
                ..ClauseLists::default()
            };
            let label = format!("{musts:?} {shoulds:?} {negatives:?}");
            let matches = r
                .search_excluding("body", lists(), TABLE_IDF_K, f32::NEG_INFINITY)
                .await
                .expect("every match")
                .len() as u64;
            let (terms, mode) = match musts.is_empty() {
                true => (shoulds, BoolMode::Or),
                false => (musts, BoolMode::And),
            };
            let unranked = r
                .atoms_match_count("body", terms, &[], mode, negatives, &[])
                .await
                .expect("unranked count")
                .0;
            for k in [1usize, 2, TABLE_IDF_K] {
                let pruned = r
                    .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                    .await
                    .expect("pruned search");
                let (hits, count) = r
                    .search_excluding_with_count("body", lists(), k)
                    .await
                    .expect("counted search");
                assert_same_top_k(&hits, &pruned, &format!("{label} k={k}"));
                assert_eq!(count, matches, "count for {label} k={k}");
                assert_eq!(count, unranked, "unranked count for {label} k={k}");
            }
        }
    }

    /// Phrase shapes take the atom walk; with pruning off it must still
    /// return the pruned walk's top-k and the match walk's count.
    #[tokio::test]
    async fn exact_count_matches_on_phrase_shapes() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open phrase blob");
        let phrase = |a: &str, b: &str| vec![Phrase::adjacent(vec![a.to_string(), b.to_string()])];
        let new_york = phrase("new", "york");
        let new_haven = phrase("new", "haven");
        // (must terms, must phrases, should phrases, negative phrases)
        let shapes: &[(
            &[&str],
            &[Phrase<String>],
            &[Phrase<String>],
            &[Phrase<String>],
        )] = &[
            (&[], &[], &new_york, &[]),
            (&[], &new_york, &[], &[]),
            (&["city"], &new_york, &[], &[]),
            (&[], &new_york, &new_haven, &[]),
            (&[], &[], &new_york, &new_haven),
            (&["york"], &[], &[], &new_york),
        ];
        for &(musts, must_phrases, should_phrases, negative_phrases) in shapes {
            let lists = || ClauseLists {
                musts,
                must_phrases,
                should_phrases,
                negative_phrases,
                ..ClauseLists::default()
            };
            let label =
                format!("{musts:?} {must_phrases:?} {should_phrases:?} {negative_phrases:?}");
            let matches = r
                .search_excluding("title", lists(), TABLE_IDF_K, f32::NEG_INFINITY)
                .await
                .expect("every match")
                .len() as u64;
            let (phrases, mode) = match musts.is_empty() && must_phrases.is_empty() {
                true => (should_phrases, BoolMode::Or),
                false => (must_phrases, BoolMode::And),
            };
            let unranked = r
                .atoms_match_count("title", musts, phrases, mode, &[], negative_phrases)
                .await
                .expect("unranked count")
                .0;
            for k in [1usize, 2, TABLE_IDF_K] {
                let pruned = r
                    .search_excluding("title", lists(), k, f32::NEG_INFINITY)
                    .await
                    .expect("pruned search");
                let (hits, count) = r
                    .search_excluding_with_count("title", lists(), k)
                    .await
                    .expect("counted search");
                assert_same_top_k(&hits, &pruned, &format!("{label} k={k}"));
                assert_eq!(count, matches, "count for {label} k={k}");
                assert_eq!(count, unranked, "unranked count for {label} k={k}");
            }
        }
    }

    /// A pushed-down row set bounds the count as it bounds the hits.
    #[tokio::test]
    async fn exact_count_honours_the_allow_set() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let allow: Arc<RoaringBitmap> = Arc::new([0u32, 2].into_iter().collect());
        let lists = || ClauseLists {
            shoulds: &["rust", "java"],
            allow: Some(Arc::clone(&allow)),
            ..ClauseLists::default()
        };
        let pruned = r
            .search_excluding("body", lists(), 1, f32::NEG_INFINITY)
            .await
            .expect("pruned search");
        let (hits, count) = r
            .search_excluding_with_count("body", lists(), 1)
            .await
            .expect("counted search");
        assert_same_top_k(&hits, &pruned, "allow-set");
        assert_eq!(count, 2, "docs 0 and 2 are allowed and match");
    }

    /// A counted single-term search with a negation folds the negated list
    /// into a bitset; under a row-set filter it must not, as that bitset
    /// knows nothing of the rows. Every doc holds `w`, so the fold's density
    /// gate passes: only the row set keeps it off. The count and the hits are
    /// those of the allowed, unnegated docs.
    #[tokio::test]
    async fn a_counted_single_term_with_a_negation_honours_the_allow_set() {
        const N_DOCS: u32 = 2_000;
        /// `x`, the negated term, is in one doc in this many.
        const X_STRIDE: u32 = 7;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let text = match i.is_multiple_of(X_STRIDE) {
                true => format!("w x {}", ["pad "].repeat((i % 5) as usize).concat()),
                false => format!("w {}", ["pad "].repeat((i % 5) as usize).concat()),
            };
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let allow: Arc<RoaringBitmap> = Arc::new((0..N_DOCS).step_by(2).collect());
        let lists = || ClauseLists {
            shoulds: &["w"],
            negatives: &["x"],
            allow: Some(Arc::clone(&allow)),
            ..ClauseLists::default()
        };
        let want = (0..N_DOCS)
            .filter(|i| i.is_multiple_of(2) && !i.is_multiple_of(X_STRIDE))
            .count() as u64;
        for k in [1, 10, 100] {
            let (hits, count) = r
                .search_excluding_with_count("body", lists(), k)
                .await
                .expect("counted search");
            let pruned = r
                .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                .await
                .expect("pruned search");
            assert_eq!(count, want, "k {k}");
            assert!(
                hits.iter().all(|(d, _)| d.get() % 2 == 0),
                "k {k}: {hits:?}"
            );
            assert_same_top_k(&hits, &pruned, "allow-set with a negation");
        }
    }

    /// A counted single-term search passes over a block none of whose scores
    /// can enter the top `k`, testing its scores eight at a time and the rest
    /// one by one. The best doc here sits in the term's last block, a few
    /// docs too short for a group of eight: passing over that block would
    /// lose it.
    #[tokio::test]
    async fn a_counted_single_term_finds_the_best_doc_in_a_short_last_block() {
        /// Docs holding `x`: three full blocks and a short fourth.
        const N_DOCS: u32 = 3 * BLOCK_LEN as u32 + 5;
        /// Tokens per doc, so tf alone orders the scores.
        const DOC_LEN: usize = 12;
        const BEST_TF: usize = 10;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let tf = if i == N_DOCS - 1 { BEST_TF } else { 1 };
            let text = ["x "].repeat(tf).concat() + &["pad "].repeat(DOC_LEN - tf).concat();
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let lists = || ClauseLists {
            shoulds: &["x"],
            ..ClauseLists::default()
        };
        for k in [1, 3] {
            let (hits, count) = r
                .search_excluding_with_count("body", lists(), k)
                .await
                .expect("counted search");
            let pruned = r
                .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                .await
                .expect("pruned search");
            assert_eq!(count, u64::from(N_DOCS), "k {k}");
            assert_eq!(hits[0].0.get(), N_DOCS - 1, "k {k}");
            assert_same_top_k(&hits, &pruned, "short last block");
        }
    }

    /// A single-term search long enough to seed its bar does not seed it
    /// under a negation: a seed over docs the negation drops would set the
    /// bar too high. Here the term's best docs all carry the negated term,
    /// and the docs that do answer sit in blocks whose best score is under
    /// theirs, so a bar seeded from them would skip every answer. Every doc
    /// has one length, so tf alone orders the scores. The answer is the term
    /// ranked alone with the negated docs dropped, compared doc for doc and
    /// bit for bit, with the heap collector and the buffer.
    #[tokio::test]
    async fn a_single_term_seed_is_not_taken_over_negated_docs() {
        /// Docs past the single-term walk's seed floor, every one holding
        /// `x`, which a term in every doc takes in wide blocks.
        const N_DOCS: u32 = (seed::MIN_BLOCKS as u32 + 8) * BLOCK_LEN as u32;
        /// Tokens per doc.
        const DOC_LEN: usize = 5;
        for k in [64, BUFFERED_TOPK_MIN_K + 100] {
            // The best `k` (tf 3) lead the list and are all negated; the `k`
            // that answer (tf 2) sit together at the middle.
            let answer = N_DOCS / 2..N_DOCS / 2 + k as u32;
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for i in 0..N_DOCS {
                let (tf, neg) = match i {
                    _ if i < k as u32 => (3, 1),
                    _ if answer.contains(&i) => (2, 0),
                    _ => (1, 0),
                };
                let text = ["x "].repeat(tf).concat()
                    + &["neg "].repeat(neg).concat()
                    + &["pad "].repeat(DOC_LEN - tf - neg).concat();
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
            let layout = r
                .term_layout("body", "x")
                .await
                .expect("layout")
                .expect("x");
            assert!(layout.num_blocks >= seed::MIN_BLOCKS, "premise: {layout:?}");
            let bits = |hits: Vec<(RowId, f32)>| -> Vec<(u32, u32)> {
                hits.into_iter()
                    .map(|(d, s)| (d.get(), s.to_bits()))
                    .collect()
            };
            let lists = ClauseLists {
                shoulds: &["x"],
                negatives: &["neg"],
                ..ClauseLists::default()
            };
            let got = bits(
                r.search_excluding("body", lists, k, f32::NEG_INFINITY)
                    .await
                    .expect("search"),
            );
            let mut want = bits(
                r.search("body", &["x"], N_DOCS as usize, BoolMode::Or)
                    .await
                    .expect("ranked alone"),
            );
            want.retain(|&(d, _)| d >= k as u32);
            want.truncate(k);
            assert!(want.iter().all(|&(d, _)| answer.contains(&d)), "k {k}");
            assert_eq!(got, want, "k {k}");
        }
    }

    /// A table-wide search scores with its map's idf alone: a scored term
    /// the map leaves out is an error, never this superfile's own idf.
    #[tokio::test]
    async fn a_scored_term_missing_from_the_table_idf_is_an_error() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let table = GlobalTermIdf::new();
        let clauses = ClauseLists {
            musts: &["rust"],
            idf: QueryIdf::Table(&table),
            ..ClauseLists::default()
        };
        let err = r
            .search_excluding("body", clauses, TABLE_IDF_K, f32::NEG_INFINITY)
            .await
            .expect_err("no idf for `rust`");
        assert!(matches!(err, FtsError::MissingIdf), "{err:?}");
    }

    /// The table's idf is the one scored with: doubling it doubles every
    /// score, which a fallback to this superfile's own idf would not do.
    #[tokio::test]
    async fn the_table_idf_is_what_scores() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let scores = |idf: f32| {
            let table = GlobalTermIdf::from([("rust".to_owned(), idf)]);
            let r = &r;
            async move {
                let clauses = ClauseLists {
                    musts: &["rust"],
                    idf: QueryIdf::Table(&table),
                    ..ClauseLists::default()
                };
                let mut hits = r
                    .search_excluding("body", clauses, TABLE_IDF_K, f32::NEG_INFINITY)
                    .await
                    .expect("search");
                hits.sort_by_key(|(d, _)| d.get());
                hits
            }
        };
        let once = scores(TABLE_IDF).await;
        let twice = scores(2.0 * TABLE_IDF).await;
        assert!(!once.is_empty());
        for ((d1, s1), (d2, s2)) in once.iter().zip(&twice) {
            assert_eq!(d1, d2);
            assert!(
                (s2 - 2.0 * s1).abs() <= f32::EPSILON * s2.abs(),
                "{s1} {s2}"
            );
        }
    }

    #[tokio::test]
    async fn search_returns_exact_doc_ids_for_known_term() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("FTS search");
        // "rust" appears in doc 0 and doc 1.
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert_eq!(ids, vec![1], "doc 0 excluded by negated 'async'");
    }

    /// A pushed-down allow-set bounds every clause shape the same way: the
    /// single-term BMW fast path, a multi-term union, a must intersection,
    /// and the phrase walk each rank only the allowed docs, and an
    /// allow-set composes with a negated term.
    #[tokio::test]
    async fn allow_set_bounds_every_clause_shape() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let ids = |hits: &[(RowId, f32)]| {
            let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
            ids.sort_unstable();
            ids
        };
        // "rust" is in docs 0 and 1; allow doc 1 only.
        let only_one: Option<Arc<RoaringBitmap>> = Some(Arc::new([1u32].into_iter().collect()));
        let single = r
            .search_excluding(
                "body",
                ClauseLists {
                    shoulds: &["rust"],
                    allow: only_one.clone(),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("single-term search");
        assert_eq!(ids(&single), vec![1], "single-term path honours the set");

        // "rust" OR "spring": docs 0, 1, 2 match; allow docs 0 and 2.
        let zero_two: Option<Arc<RoaringBitmap>> = Some(Arc::new([0u32, 2].into_iter().collect()));
        let union = r
            .search_excluding(
                "body",
                ClauseLists {
                    shoulds: &["rust", "spring"],
                    allow: zero_two.clone(),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("union search");
        assert_eq!(ids(&union), vec![0, 2], "union path honours the set");

        // "rust" AND "runtime": docs 0 and 1 match; allow docs 0 and 2.
        let intersection = r
            .search_excluding(
                "body",
                ClauseLists {
                    musts: &["rust", "runtime"],
                    allow: zero_two.clone(),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("intersection search");
        assert_eq!(ids(&intersection), vec![0], "must path honours the set");

        // Allow-set plus a negated term: "rust" in docs 0, 1; allow 0 and
        // 2; negate "async" (doc 0) — nothing survives both gates.
        let gated = r
            .search_excluding(
                "body",
                ClauseLists {
                    shoulds: &["rust"],
                    negatives: &["async"],
                    allow: zero_two,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("gated search");
        assert!(gated.is_empty(), "allow-set and negation both apply");

        // Phrase walk: "new york" matches docs 0, 2, 4 of the phrase
        // corpus; allow docs 2 and 3.
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open phrase blob");
        let phrases = vec![Phrase::adjacent(vec![
            "new".to_string(),
            "york".to_string(),
        ])];
        let phrase_hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    allow: Some(Arc::new([2u32, 3].into_iter().collect())),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        assert_eq!(ids(&phrase_hits), vec![2], "phrase walk honours the set");

        // An empty allow-set ranks nothing, whatever matches.
        let none = r
            .search_excluding(
                "title",
                ClauseLists {
                    shoulds: &["new"],
                    allow: Some(Arc::new(RoaringBitmap::new())),
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("empty-set search");
        assert!(none.is_empty());
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

    /// A caller's floor prunes what scores below it and nothing else: a
    /// floored search returns the unfloored ranking cut at the floor, ties
    /// with the floor kept. The
    /// corpus puts whole coarse spans of a long term under the floor (long
    /// documents), one block inside such a span over it (short documents),
    /// and the rest in between, so the walk skips by span, by block and by
    /// document. Checked for one term, at a heap's `k` and a buffer's, and
    /// for an AND with a negation at a buffer's `k`.
    #[tokio::test]
    async fn a_caller_floor_cuts_the_unfloored_ranking() {
        const N_DOCS: u32 = 24_000;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let short = i >= N_DOCS / 2 || (5_000..5_128).contains(&i);
            let mut text = String::from("w ");
            if !i.is_multiple_of(3) {
                text.push_str("v ");
            }
            if i.is_multiple_of(7) {
                text.push_str("x ");
            }
            let pad = if short {
                i as usize % 4
            } else {
                40 + i as usize % 40
            };
            text.push_str(&"pad ".repeat(pad));
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let all = N_DOCS as usize;
        let cut = |hits: Vec<(RowId, f32)>, floor: f32, k: usize| -> Vec<(RowId, f32)> {
            hits.into_iter().filter(|h| h.1 >= floor).take(k).collect()
        };
        let unfloored = r
            .search_with_floor("body", &["w"], all, BoolMode::Or, f32::NEG_INFINITY)
            .await
            .expect("unfloored");
        let floors = {
            let mut scores: Vec<f32> = unfloored.iter().map(|h| h.1).collect();
            scores.dedup();
            [
                scores[scores.len() / 4],
                scores[scores.len() / 2],
                scores[3 * scores.len() / 4],
            ]
        };
        for floor in floors {
            for k in [10, BUFFERED_TOPK_MIN_K + 100, all] {
                let got = r
                    .search_with_floor("body", &["w"], k, BoolMode::Or, floor)
                    .await
                    .expect("floored");
                let want = cut(unfloored.clone(), floor, k);
                assert_eq!(got, want, "single term k={k} floor={floor}");
            }
        }
        let and = |k, floor| {
            r.search_excluding(
                "body",
                ClauseLists {
                    musts: &["w", "v"],
                    negatives: &["x"],
                    ..ClauseLists::default()
                },
                k,
                floor,
            )
        };
        let unfloored = and(all, f32::NEG_INFINITY).await.expect("unfloored and");
        // The match set itself, independent of the walk: `v` where `i % 3 != 0`,
        // `x` where `i % 7 == 0`.
        let matches = |i: u32| !i.is_multiple_of(3) && !i.is_multiple_of(7);
        assert_eq!(
            unfloored.len(),
            (0..N_DOCS).filter(|&i| matches(i)).count(),
            "and match count"
        );
        assert!(
            unfloored.iter().all(|h| matches(h.0.get())),
            "and returned an excluded doc"
        );
        for k in [BUFFERED_TOPK_MIN_K + 100, all] {
            for floor in [f32::NEG_INFINITY, unfloored[unfloored.len() / 2].1] {
                let got = and(k, floor).await.expect("floored and");
                let want = cut(unfloored.clone(), floor, k);
                assert_eq!(got, want, "and with a negation, k={k} floor={floor}");
            }
        }
    }

    /// Pruning compares bounds summed in one order with a bar, while the
    /// scores they cover are summed in another, so near-ties a few ulps apart
    /// are where a bound rounds under the score it covers. Corpora dense in
    /// such near-ties: every pruned ranking must equal the unpruned
    /// exact-count walk's top-k, bit for bit for an AND (one walk sums both),
    /// and up to summation order where the union kernels differ.
    #[tokio::test]
    async fn pruned_rankings_agree_with_the_unpruned_walk_at_near_ties() {
        const TERMS: [&str; 5] = ["ta", "tb", "tc", "td", "te"];
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..60 {
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            let n_docs = 200 + (next() % 2_000) as u32;
            for i in 0..n_docs {
                let mut text = String::new();
                for t in TERMS {
                    if next() % 10 < 8 {
                        for _ in 0..1 + next() % 3 {
                            text.push_str(t);
                            text.push(' ');
                        }
                    }
                }
                text.push_str(&"pad ".repeat((next() % 40) as usize));
                b.add_doc(0, i, text.trim()).expect("add doc");
            }
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
            for (musts, shoulds) in [
                (&TERMS[..], &[][..]),
                (&[][..], &TERMS[..]),
                (&TERMS[..3], &TERMS[3..]),
            ] {
                let lists = || ClauseLists {
                    musts,
                    shoulds,
                    ..ClauseLists::default()
                };
                for k in [1usize, 2, 3, 5, 10, 40, 600] {
                    let pruned = r
                        .search_excluding("body", lists(), k, f32::NEG_INFINITY)
                        .await
                        .expect("pruned");
                    let (counted, _) = r
                        .search_excluding_with_count("body", lists(), k)
                        .await
                        .expect("counted");
                    let label = format!("trial {trial} musts {musts:?} shoulds {shoulds:?} k {k}");
                    match shoulds.is_empty() {
                        // One walk sums an AND either way: the same bits.
                        true => {
                            let bits = |h: &[(RowId, f32)]| {
                                h.iter()
                                    .map(|&(d, s)| (d.get(), s.to_bits()))
                                    .collect::<Vec<_>>()
                            };
                            assert_eq!(bits(&pruned), bits(&counted), "{label}");
                        }
                        // The union kernels add terms in different orders.
                        false => assert_same_top_k(&counted, &pruned, &label),
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn search_multi_weights_and_combines_columns() {
        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75},{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let hits = r
            .search_multi(&[("title", 1.0), ("body", 1.0)], "rust", 10, BoolMode::Or)
            .await
            .expect("multi");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn search_or_range_restricts_to_doc_id_window() {
        // Larger corpus so an OR query spans several doc ids and the
        // ranged path actually clips some out.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..8u32 {
            b.add_doc(0, i, "alpha beta").expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        // Restrict to [2, 5): only docs 2,3,4 are eligible.
        let hits = r
            .search_or_range_pretokenized("body", &["alpha", "beta"], 100, 2, 5)
            .await
            .expect("ranged search");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
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
        /// Docs in the planted corpus — spans several OR windows and many
        /// posting blocks.
        const N_DOCS: u32 = 2 * OR_WINDOW + OR_WINDOW / 3;
        /// Ask for every match so partition union == full result set.
        const K_ALL: usize = N_DOCS as usize;
        /// Top-k size for the truncated comparison.
        const K_TOP: usize = 10;

        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Two shapes through the one ranged union kernel: a uniform OR
        // (exercises the windowed accumulate) and a rare+common OR (exercises
        // the dominant-leader path), each sliced identically to the un-ranged
        // search so the partition union must reproduce the whole-superfile
        // result regardless of where the cuts fall.
        let shapes: [&[&str]; 2] = [
            &["alpha", "beta", "gamma", "delta"],
            &["rareterm", "alpha", "beta"],
        ];
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
                full.iter().map(|&(d, s)| (d.get(), s.to_bits())).collect();
            full_sorted.sort_unstable();

            for cuts in partitions {
                let mut merged: Vec<(RowId, f32)> = Vec::new();
                for &(lo, hi) in cuts {
                    merged.extend(
                        r.search_or_range_pretokenized("body", terms, K_ALL, lo, hi)
                            .await
                            .expect("ranged search"),
                    );
                }
                let mut merged_sorted: Vec<(u32, u32)> = merged
                    .iter()
                    .map(|&(d, s)| (d.get(), s.to_bits()))
                    .collect();
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
                let top: Vec<(u32, u32)> =
                    pool.iter().map(|&(d, s)| (d.get(), s.to_bits())).collect();
                let full_top: Vec<(u32, u32)> = full
                    .iter()
                    .take(K_TOP)
                    .map(|&(d, s)| (d.get(), s.to_bits()))
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

        let mut b = FtsBuilder::new();
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
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let terms: &[&str] = &["alpha", "beta", "gamma", "delta"];
        let set = r
            .build_or_cursor_set("body", terms, QueryIdf::ThisFile, None)
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
                fresh.iter().map(|&(d, s)| (d.get(), s.to_bits())).collect();
            let pre_bits: Vec<(u32, u32)> =
                pre.iter().map(|&(d, s)| (d.get(), s.to_bits())).collect();
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
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..8u32 {
            b.add_doc(0, i, "alpha beta").expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let hits = r
            .search_or_range_pretokenized_with_floor(
                "body",
                &["alpha", "beta"],
                100,
                0,
                8,
                1e9,
                QueryIdf::ThisFile,
            )
            .await
            .expect("floored ranged search");
        assert!(hits.is_empty(), "floor above all scores prunes everything");
    }

    /// Ids translated through doc-id map pages on a lazily read, reordered
    /// blob give the rows the whole map gives: ids in one page, in adjacent
    /// pages, in a lone page and in the map's last page, which is shorter
    /// than the rest. Enough pages to pass the sparse share fall back to the
    /// whole map, with the same rows.
    #[tokio::test]
    async fn map_pages_translate_ids_like_the_whole_map() {
        use super::super::metadata::OpenOptions;
        use crate::superfile::lazy_source::LazyByteSource;

        const N_DOCS: u32 = 1000;
        let docs: Vec<String> = (0..N_DOCS).map(|row| format!("alpha r{row}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let mut next = xorshift(u64::from(N_DOCS));
        let mut order: Vec<u32> = (0..N_DOCS).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let open = || async {
            let src: Arc<dyn LazyByteSource> = Arc::new(AsyncOnlySource(blob.clone()));
            FtsReader::open_lazy(src, &json, OpenOptions { verify_crc: false })
                .await
                .expect("open_lazy")
        };
        let per_page = (DOC_MAP_PAGE_BYTES / DOC_MAP_ENTRY_BYTES) as u32;
        let last_page = (N_DOCS - 1) / per_page;
        assert!(
            !N_DOCS.is_multiple_of(per_page),
            "the map's last page must be shorter than the rest"
        );
        let row = |id: u32| RowId::new(order[id as usize]);

        // Pages 0 to 3 in one run, a lone page, and the last page.
        let ids: Vec<u32> = vec![
            last_page * per_page + 3,
            0,
            per_page - 1,
            per_page,
            3 * per_page,
            N_DOCS / 2,
            N_DOCS - 1,
            last_page * per_page,
        ];
        let r = open().await;
        let hits: Vec<(FtsDocId, f32)> = ids
            .iter()
            .map(|&id| (FtsDocId::new(id), id as f32))
            .collect();
        let got = r
            .hits_to_rows_async(hits)
            .await
            .expect("hits_to_rows_async");
        let want: Vec<(RowId, f32)> = ids.iter().map(|&id| (row(id), id as f32)).collect();
        assert_eq!(got, want);
        let id_list: Vec<FtsDocId> = ids.iter().map(|&id| FtsDocId::new(id)).collect();
        let got = r
            .ids_to_rows_async(id_list)
            .await
            .expect("ids_to_rows_async");
        let mut want: Vec<RowId> = ids.iter().map(|&id| row(id)).collect();
        want.sort_unstable();
        assert_eq!(got, want);
        assert!(
            r.doc_map.get().is_none(),
            "the pages served the translation, not the whole map"
        );

        // One id in every page: past the sparse share, so the whole map.
        let r = open().await;
        let every_page: Vec<u32> = (0..=last_page).map(|p| p * per_page).collect();
        let hits: Vec<(FtsDocId, f32)> = every_page
            .iter()
            .map(|&id| (FtsDocId::new(id), 1.0))
            .collect();
        let got = r
            .hits_to_rows_async(hits)
            .await
            .expect("hits_to_rows_async");
        let want: Vec<(RowId, f32)> = every_page.iter().map(|&id| (row(id), 1.0)).collect();
        assert_eq!(got, want);
        assert!(r.doc_map.get().is_some(), "the whole map was read");
    }
}
