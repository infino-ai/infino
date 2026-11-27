// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Scored FTS search entry points on [`FtsReader`]: the public
//! `search` / `search_with_floor` / pretokenized and doc-id-ranged
//! variants, the multi-column `search_multi`, the single-term
//! BlockMaxWAND fast path, and the atom (phrase-aware) scored walk.
//! Its own `impl FtsReader` block, split from the reader `core`.

use std::{
    collections::{BinaryHeap, HashMap, HashSet, hash_map::Entry},
    ops::Range,
};

use bytes::Bytes;
use futures::{StreamExt, future::try_join_all, stream};
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
use tokio::sync::{Semaphore, SemaphorePermit};

use super::{
    bounds::BoundDecoder,
    core::*,
    cursor::{CursorUse, TermCursor, TermMeta},
    filter::{AllowGate, AtomExcludeFilter, ExcludeFilter},
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
        format::{self, checksum::crc32c_append},
        fts::{
            bm25,
            posting::{BLOCK_LEN, BlockHeader, decode_block},
            short::{decode_short, short_df},
        },
        id_space::{Admitted, BlobIdSet, DocMap, FtsDocId, RowId, RowSet},
    },
    utils::terms::{DictEntry, make_key},
};

/// Clause-term lists up to this long are deduplicated by scanning the terms
/// kept so far; longer ones, which only a prefix expansion produces, by a map.
const DEDUP_SCAN_MAX_TERMS: usize = 64;
/// Bytes per doc-id map entry: one little-endian `u32` row.
const DOC_MAP_ENTRY_BYTES: usize = 4;
/// Bytes of the doc-id map a sparse translation reads at a time. Small
/// under test, so a fixture of a thousand documents spans many pages.
const DOC_MAP_PAGE_BYTES: usize = if cfg!(test) { 64 } else { 64 * 1024 };
/// A ranked translation that would read more than 1/8 of a map keeps the whole map instead, if the
/// heap budget has room; a pass that reads that much without keeping reads every page to check the
/// map's CRC.
const DOC_MAP_SPARSE_SHARE: usize = 8;
/// Pages a translation reads through to join two reads into one: one more request costs about as
/// much as this many more pages.
const DOC_MAP_MERGE_GAP_PAGES: usize = 2;
/// Most pages one read of a map covers (4 MiB), and one batch of a pass that reads pages. Small
/// under test, so a fixture spans several reads.
const DOC_MAP_READ_MAX_PAGES: usize = if cfg!(test) { 8 } else { 64 };
/// Bytes of one read of a map: [`DOC_MAP_READ_MAX_PAGES`] pages.
const DOC_MAP_READ_CAP_BYTES: usize = DOC_MAP_READ_MAX_PAGES * DOC_MAP_PAGE_BYTES;
/// Batches a map pass has in flight at once: enough to hide a remote store's latency, while it
/// holds at most this many batches of bytes.
const DOC_MAP_BATCHES_IN_FLIGHT: usize = 4;

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

/// The reads that cover the map entries at `entries` (ascending byte offsets): their pages, runs at
/// most [`DOC_MAP_MERGE_GAP_PAGES`] apart joined, each read at most [`DOC_MAP_READ_MAX_PAGES`].
fn map_reads(entries: impl Iterator<Item = usize>, map_len: usize) -> Vec<Range<usize>> {
    let mut pages: Vec<(usize, usize)> = Vec::new();
    for page in entries.map(|at| at / DOC_MAP_PAGE_BYTES) {
        match pages.last_mut() {
            Some((first, last))
                if page <= *last + DOC_MAP_MERGE_GAP_PAGES + 1
                    && page < *first + DOC_MAP_READ_MAX_PAGES =>
            {
                *last = page
            }
            _ => pages.push((page, page)),
        }
    }
    pages
        .into_iter()
        .map(|(first, last)| {
            first * DOC_MAP_PAGE_BYTES..((last + 1) * DOC_MAP_PAGE_BYTES).min(map_len)
        })
        .collect()
}

/// `reads` in order, in groups of at most `cap` bytes. A group is fetched at once, so `cap` is what
/// a translation holds.
fn in_flight_batches(reads: &[Range<usize>], cap: usize) -> Vec<&[Range<usize>]> {
    let mut batches = Vec::new();
    let (mut first, mut bytes) = (0, 0);
    for (i, read) in reads.iter().enumerate() {
        if i > first && bytes + read.len() > cap {
            batches.push(&reads[first..i]);
            (first, bytes) = (i, 0);
        }
        bytes += read.len();
    }
    if first < reads.len() {
        batches.push(&reads[first..]);
    }
    batches
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

    /// Collapse repeated clause terms into distinct terms plus a per-term repeat
    /// count (query-term-frequency), in first-seen order. Returns each term with
    /// count 1, a no-op, for a distinct-term query.
    ///
    /// A typed query has a handful of terms, where scanning the terms kept so
    /// far beats hashing each one. A prefix query hands over every term the
    /// prefix expands to in this superfile, thousands for a short prefix, and
    /// the scan is quadratic there, so past [`DEDUP_SCAN_MAX_TERMS`] a map
    /// finds each term's slot instead. Both give the same output.
    fn dedup_terms_with_qtf<'a>(terms: &[&'a str]) -> (Vec<&'a str>, Vec<u32>) {
        let mut distinct: Vec<&str> = Vec::with_capacity(terms.len());
        let mut qtf: Vec<u32> = Vec::with_capacity(terms.len());
        if terms.len() <= DEDUP_SCAN_MAX_TERMS {
            for &t in terms {
                match distinct.iter().position(|&d| d == t) {
                    Some(i) => qtf[i] += 1,
                    None => {
                        distinct.push(t);
                        qtf.push(1);
                    }
                }
            }
        } else {
            let mut slot: FxHashMap<&str, usize> =
                FxHashMap::with_capacity_and_hasher(terms.len(), Default::default());
            for &t in terms {
                match slot.entry(t) {
                    Entry::Occupied(e) => qtf[*e.get()] += 1,
                    Entry::Vacant(e) => {
                        e.insert(distinct.len());
                        distinct.push(t);
                        qtf.push(1);
                    }
                }
            }
        }
        (distinct, qtf)
    }

    /// `allow` as a filter's gate. A row set reads the doc-id map here, so a row gate always holds
    /// the real map; a blob-id set needs none.
    async fn allow_gate(&self, allow: Option<Admitted>) -> Result<Option<AllowGate>, FtsError> {
        Ok(match allow {
            None => None,
            Some(Admitted::Rows(rows)) => Some(AllowGate::Rows {
                rows: RowSet::new(rows),
                doc_map: self.ensure_doc_map().await?.clone(),
            }),
            Some(Admitted::BlobIds(ids)) => Some(AllowGate::BlobIds(BlobIdSet::new(ids))),
        })
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
                    None,
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
                    None,
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
                    None,
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
            // atom, or a pushed-down set.
            let filter = match (negative_atoms.is_empty(), lists.allow.clone()) {
                (true, None) => None,
                (_, allow) => Some(AtomExcludeFilter::with_allow(
                    negative_atoms,
                    self.allow_gate(allow).await?,
                )),
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
        // term present in this superfile, or a pushed-down set.
        let neg_filter = match (neg_cursors.is_empty(), lists.allow.clone()) {
            (true, None) => None,
            (_, allow) => Some(ExcludeFilter::with_allow(
                neg_cursors,
                self.allow_gate(allow).await?,
            )),
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

    /// Translate a ranked kernel's doc ids into Parquet rows.
    ///
    /// [`Self::hits_to_rows_async`] without a runtime, through the whole map, for the tests that
    /// drive kernels directly. A no-op on a blob without a doc-id map (every blob through `V7`).
    ///
    /// Order is left alone: a ranked result is ordered by score, and
    /// translating ids does not disturb that. The unranked siblings
    /// promise ascending ids instead, which translation *does* disturb,
    /// so they use [`Self::ids_to_rows_async`].
    ///
    /// No hits read no map.
    #[cfg(test)]
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
    /// map. Translated by [`Self::rows_of_ids`].
    ///
    /// No ids read no map.
    pub(crate) async fn ids_to_rows_async(
        &self,
        ids: Vec<FtsDocId>,
    ) -> Result<Vec<RowId>, FtsError> {
        let rows = self.rows_of_ids(&ids, false).await?;
        Ok(self.in_row_order(rows))
    }

    /// [`Self::ids_to_rows_async`] through the held map, for a slice's pool job; `None` while no
    /// map is held, and the slice reads pages with [`Self::ids_to_rows_in_pages`].
    pub(crate) fn held_ids_to_rows(
        &self,
        ids: &[FtsDocId],
    ) -> Result<Option<Vec<RowId>>, FtsError> {
        let Some(map) = self.checked_held_map(ids.iter().copied())? else {
            return Ok(None);
        };
        let rows = ids.iter().map(|&id| map.row_of(id)).collect();
        Ok(Some(self.in_row_order(rows)))
    }

    /// [`Self::ids_to_rows_async`] through the map's pages only, never keeping it: one slice of a
    /// match whose file could not keep its map ([`Self::kept_doc_map_within_budget`]).
    pub(crate) async fn ids_to_rows_in_pages(
        &self,
        ids: Vec<FtsDocId>,
        set: &OrCursorSet,
    ) -> Result<Vec<RowId>, FtsError> {
        let rows = self.rows_in_pages(&ids, &set.page_reads).await?;
        Ok(self.in_row_order(rows))
    }

    /// `rows` put in row order when this blob numbers its documents in an order of its own.
    fn in_row_order(&self, rows: Vec<RowId>) -> Vec<RowId> {
        match self.has_doc_map() {
            true => into_row_order(rows, self.n_docs() as usize),
            false => rows,
        }
    }

    /// The rows of `ids`, which must be ascending, through the pages they fall in, keeping nothing.
    /// A slice's pages are a small share of the map, so they never check its CRC; the match checks
    /// it once for the file ([`Self::check_map_crc_for`]).
    pub(super) async fn rows_in_pages(
        &self,
        ids: &[FtsDocId],
        permits: &Semaphore,
    ) -> Result<Vec<RowId>, FtsError> {
        self.check_ids_in_map(ids.iter().copied())?;
        let reads = map_reads(
            ids.iter().map(|id| id.get() as usize * DOC_MAP_ENTRY_BYTES),
            self.doc_map_len(),
        );
        let cap = DOC_MAP_READ_CAP_BYTES;
        self.rows_through_pages(ids, None, &reads, cap, Some(permits))
            .await
    }

    /// The rows `ids` stand for, in the order given:
    ///
    /// ```text
    /// map already kept                             -> translate in memory
    /// reads > 1/8 of the map, heap budget has room -> read it whole, keep it, translate in memory
    /// reads <= 1/8, `keep_pages`                   -> read the pages not kept yet, keep them
    /// otherwise                                    -> read only the pages the ids fall in
    /// ```
    ///
    /// Keeping every file's whole map is what runs a query over hundreds of large files out of
    /// memory, so a map is kept only while the budget has room. Only a ranked search's top hits
    /// set `keep_pages`: they touch a few pages, while an unranked lookup can touch 1/8 of the map.
    pub(super) async fn rows_of_ids(
        &self,
        ids: &[FtsDocId],
        keep_pages: bool,
    ) -> Result<Vec<RowId>, FtsError> {
        let through =
            |map: &DocMap| -> Vec<RowId> { ids.iter().map(|&id| map.row_of(id)).collect() };
        if let Some(map) = self.checked_held_map(ids.iter().copied())? {
            return Ok(through(map));
        }
        self.check_ids_in_map(ids.iter().copied())?;
        // Ranked hits come in score order and need sorting by id; unranked ids are already sorted.
        let by_entry: Option<Vec<u32>> = (!ids.is_sorted()).then(|| {
            let mut order: Vec<u32> = (0..ids.len() as u32).collect();
            order.sort_unstable_by_key(|&i| ids[i as usize]);
            order
        });
        let index = |k: usize| by_entry.as_ref().map_or(k, |order| order[k] as usize);
        let entry_at = |id: FtsDocId| id.get() as usize * DOC_MAP_ENTRY_BYTES;
        let reads = map_reads(
            (0..ids.len()).map(|k| entry_at(ids[index(k)])),
            self.doc_map_len(),
        );
        let read_bytes: usize = reads.iter().map(Range::len).sum();
        let dense = self.reads_most_of_map(read_bytes);
        // Sparse reads are few, so they go out at once.
        if !dense {
            return match keep_pages {
                true => self.rows_through_kept_pages(ids, &reads).await,
                false => {
                    self.rows_through_pages(ids, by_entry.as_deref(), &reads, read_bytes, None)
                        .await
                }
            };
        }
        if let Some(map) = self.kept_doc_map_within_budget().await? {
            return Ok(through(map));
        }
        // A dense pass reads one read cap per batch, a few batches in flight. When the reader
        // checks CRCs it reads every page, so the map's CRC is still checked.
        let reads = match self.verify_crc {
            true => self.every_page(),
            false => reads,
        };
        self.rows_through_pages(
            ids,
            by_entry.as_deref(),
            &reads,
            DOC_MAP_READ_CAP_BYTES,
            None,
        )
        .await
    }

    /// Reads every page once to check the map's CRC when `matched` documents may touch most of it:
    /// a sliced match's slices read only their own pages, which can't check it.
    pub(super) async fn check_map_crc_for(&self, matched: u64) -> Result<(), FtsError> {
        let pages = self.doc_map_len().div_ceil(DOC_MAP_PAGE_BYTES) as u64;
        let touched = matched.min(pages) as usize * DOC_MAP_PAGE_BYTES;
        if !self.verify_crc || !self.reads_most_of_map(touched) {
            return Ok(());
        }
        self.map_pass(&self.every_page(), DOC_MAP_READ_CAP_BYTES, None, |_, _| {})
            .await
    }

    /// Whether reading `bytes` of the map reads most of it: more than 1/[`DOC_MAP_SPARSE_SHARE`].
    fn reads_most_of_map(&self, bytes: usize) -> bool {
        bytes * DOC_MAP_SPARSE_SHARE > self.doc_map_len()
    }

    /// The rows of `docs` through the map's pages in one pass, keeping nothing, for a contains that
    /// could not keep the map; like [`Self::rows_of_ids`], it checks the CRC when it reads most.
    pub(super) async fn bitmap_rows_in_pages(
        &self,
        docs: &RoaringBitmap,
    ) -> Result<RoaringBitmap, FtsError> {
        let reads = map_reads(
            docs.iter().map(|doc| doc as usize * DOC_MAP_ENTRY_BYTES),
            self.doc_map_len(),
        );
        let read_bytes: usize = reads.iter().map(Range::len).sum();
        let reads = match self.verify_crc && self.reads_most_of_map(read_bytes) {
            true => self.every_page(),
            false => reads,
        };
        let cap = DOC_MAP_READ_CAP_BYTES;
        let mut rows = RoaringBitmap::new();
        let mut next = docs.iter().peekable();
        self.map_pass(&reads, cap, None, |first, entries| {
            let end = first + (entries.len() / DOC_MAP_ENTRY_BYTES) as u32;
            while let Some(&doc) = next.peek()
                && doc < end
            {
                let at = (doc - first) as usize * DOC_MAP_ENTRY_BYTES;
                rows.insert(read_u32_le(&entries[at..at + DOC_MAP_ENTRY_BYTES]));
                next.next();
            }
        })
        .await?;
        Ok(rows)
    }

    /// Reads of every page of the map, in order, so a pass over them checks its CRC.
    fn every_page(&self) -> Vec<Range<usize>> {
        let every_page = (0..self.doc_map_len()).step_by(DOC_MAP_PAGE_BYTES);
        map_reads(every_page, self.doc_map_len())
    }

    /// The rows `ids` stand for, in the order given, from `reads` of the map (which cover every
    /// id) with at most `in_flight` bytes read at once. `by_entry` lists the ids' positions in
    /// ascending id order when they are not already ascending.
    async fn rows_through_pages(
        &self,
        ids: &[FtsDocId],
        by_entry: Option<&[u32]>,
        reads: &[Range<usize>],
        in_flight: usize,
        permits: Option<&Semaphore>,
    ) -> Result<Vec<RowId>, FtsError> {
        let index = |k: usize| by_entry.map_or(k, |order| order[k] as usize);
        let mut rows = vec![RowId::new(0); ids.len()];
        let mut next = 0;
        self.map_pass(reads, in_flight, permits, |first, entries| {
            let end = first as usize + entries.len() / DOC_MAP_ENTRY_BYTES;
            while next < ids.len() && (ids[index(next)].get() as usize) < end {
                let i = index(next);
                let at = (ids[i].get() - first) as usize * DOC_MAP_ENTRY_BYTES;
                rows[i] = RowId::new(read_u32_le(&entries[at..at + DOC_MAP_ENTRY_BYTES]));
                next += 1;
            }
        })
        .await?;
        Ok(rows)
    }

    /// The held doc-id map, once `ids` are checked against it; `None` while no map is held.
    fn checked_held_map(
        &self,
        ids: impl Iterator<Item = FtsDocId>,
    ) -> Result<Option<&DocMap>, FtsError> {
        let Some(map) = self.doc_map.get() else {
            return Ok(None);
        };
        self.check_ids_in_map(ids)?;
        Ok(Some(map))
    }

    /// An error naming the first of `ids` past this blob's documents, which the map has no entry
    /// for.
    pub(super) fn check_ids_in_map(
        &self,
        mut ids: impl Iterator<Item = FtsDocId>,
    ) -> Result<(), FtsError> {
        match ids.find(|id| id.get() >= self.n_docs()) {
            Some(id) => Err(FtsError::Read(ReadError::Malformed(format!(
                "fts doc-map has no entry for blob doc id {}",
                id.get()
            )))),
            None => Ok(()),
        }
    }

    /// Bytes of this blob's doc-id map body; zero when it keeps row order.
    fn doc_map_len(&self) -> usize {
        self.doc_map_region.as_ref().map_or(0, Range::len)
    }

    /// Fetches `reads` (from [`map_reads`]) in batches of at most `in_flight` bytes, at most
    /// [`DOC_MAP_BATCHES_IN_FLIGHT`] batches at once, and hands each read to `each` in order with
    /// the first document it covers. Checks the map's CRC when the reads cover it. With `permits`,
    /// each batch holds one from its fetch until it is handed on, so passes sharing them keep at
    /// most that many batches between them.
    async fn map_pass(
        &self,
        reads: &[Range<usize>],
        in_flight: usize,
        permits: Option<&Semaphore>,
        mut each: impl FnMut(u32, &[u8]),
    ) -> Result<(), FtsError> {
        let region = self.map_region().clone();
        let check_crc = self.verify_crc
            && reads.first().is_some_and(|r| r.start == 0)
            && reads.last().is_some_and(|r| r.end == region.len())
            && reads.windows(2).all(|w| w[0].end == w[1].start);
        // A few batches are in flight at once and come back in read order, so the CRC is taken
        // over the map in order. The fetches are built before the stream: a closure inside it
        // would put a borrowed-argument closure in this future, which the fan-out can't prove
        // `Send`.
        let fetches: Vec<_> = in_flight_batches(reads, in_flight)
            .into_iter()
            .map(|batch| self.map_batch(batch, &region, check_crc, permits))
            .collect();
        let mut fetched = stream::iter(fetches).buffered(DOC_MAP_BATCHES_IN_FLIGHT);
        let mut crc = 0;
        while let Some(next) = fetched.next().await {
            let (batch, bytes_of, _permit) = next?;
            for (read, bytes) in batch.iter().zip(bytes_of) {
                let body = &bytes[..read.len()];
                if check_crc {
                    crc = crc32c_append(crc, body);
                    if read.end == region.len() && read_u32_le(&bytes[read.len()..]) != crc {
                        return Err(doc_map_crc_mismatch());
                    }
                }
                each((read.start / DOC_MAP_ENTRY_BYTES) as u32, body);
            }
        }
        Ok(())
    }

    /// One batch of [`Self::map_pass`]'s reads with their bytes, in order; the read that ends the
    /// map also takes the CRC stored right after it when `check_crc`.
    async fn map_batch<'a>(
        &self,
        batch: &'a [Range<usize>],
        region: &Range<usize>,
        check_crc: bool,
        permits: Option<&'a Semaphore>,
    ) -> Result<(&'a [Range<usize>], Vec<Bytes>, Option<SemaphorePermit<'a>>), FtsError> {
        let permit = match permits {
            Some(permits) => Some(
                permits
                    .acquire()
                    .await
                    .map_err(|_| FtsError::TaskDropped("doc-map page reads"))?,
            ),
            None => None,
        };
        let bytes = try_join_all(
            batch
                .iter()
                .map(|read| self.map_read(read, check_crc && read.end == region.len())),
        )
        .await?;
        Ok((batch, bytes, permit))
    }

    /// Reads the map's `read` range, plus the CRC stored right after it when `with_crc`.
    async fn map_read(&self, read: &Range<usize>, with_crc: bool) -> Result<Bytes, FtsError> {
        let region = self.map_region();
        let crc_len = if with_crc { format::CRC_BYTES } else { 0 };
        let span = region.start + read.start..region.start + read.end + crc_len;
        #[cfg(test)]
        self.count_doc_map_read();
        let bytes =
            self.source
                .range_async(span.clone())
                .await
                .map_err(|e| FtsError::RangeFetch {
                    what: "fts/doc-map pages",
                    source: e,
                })?;
        match bytes.len() >= span.len() {
            true => Ok(bytes),
            false => Err(doc_map_short_read(
                "fts doc-map pages",
                bytes.len(),
                span.len(),
            )),
        }
    }

    /// The rows of `ids` through the map pages `reads` cover. A read is skipped when the pages its
    /// ids fall in are all kept; otherwise it is made once, as without keeping, and those pages
    /// are then kept while the heap budget has room. The reads cover at most 1/8 of the map, so
    /// they all go out at once.
    async fn rows_through_kept_pages(
        &self,
        ids: &[FtsDocId],
        reads: &[Range<usize>],
    ) -> Result<Vec<RowId>, FtsError> {
        let page_of = |id: &FtsDocId| id.get() as usize * DOC_MAP_ENTRY_BYTES / DOC_MAP_PAGE_BYTES;
        // Sized by the pages the reads cover, not by the ids: many ids share a page, and
        // `collect` would reserve one slot per id.
        let covered = reads
            .iter()
            .map(|read| read.end.div_ceil(DOC_MAP_PAGE_BYTES) - read.start / DOC_MAP_PAGE_BYTES)
            .sum();
        let mut wanted = HashSet::with_capacity(covered);
        for id in ids {
            wanted.insert(page_of(id));
        }
        let pages: HashMap<usize, Bytes> =
            try_join_all(reads.iter().map(|read| self.read_pages(read, &wanted)))
                .await?
                .into_iter()
                .flatten()
                .collect();
        ids.iter()
            .map(|id| {
                let at = id.get() as usize * DOC_MAP_ENTRY_BYTES % DOC_MAP_PAGE_BYTES;
                let page = pages.get(&page_of(id)).ok_or_else(|| {
                    FtsError::Read(ReadError::Malformed(format!(
                        "fts doc-map page {} was not read for blob doc id {}",
                        page_of(id),
                        id.get()
                    )))
                })?;
                Ok(RowId::new(read_u32_le(&page[at..at + DOC_MAP_ENTRY_BYTES])))
            })
            .collect()
    }

    /// The pages of `read` that `wanted` names, by page number: from the kept pages if all are
    /// kept, else from the whole map if it is kept, else from one read of `read`, which keeps them
    /// while the budget has room.
    async fn read_pages(
        &self,
        read: &Range<usize>,
        wanted: &HashSet<usize>,
    ) -> Result<Vec<(usize, Bytes)>, FtsError> {
        let first = read.start / DOC_MAP_PAGE_BYTES;
        let hit: Vec<usize> = (first..read.end.div_ceil(DOC_MAP_PAGE_BYTES))
            .filter(|page| wanted.contains(page))
            .collect();
        if let Some(kept) = self.kept_pages.all(&hit) {
            return Ok(hit.into_iter().zip(kept).collect());
        }
        let (bytes, fresh) = match self.doc_map.get() {
            Some(DocMap::Permuted(map)) => (map.slice(read.clone()), false),
            _ => (self.map_read(read, false).await?, true),
        };
        // `read` starts on a page, so page `first + i` starts at `i` pages into `bytes`.
        let pages: Vec<(usize, Bytes)> = hit
            .into_iter()
            .map(|page| {
                let at = (page - first) * DOC_MAP_PAGE_BYTES;
                (
                    page,
                    bytes.slice(at..(at + DOC_MAP_PAGE_BYTES).min(read.len())),
                )
            })
            .collect();
        if fresh && self.kept_pages.keep_all(&pages, self.heap_budget) {
            self.loads.loaded();
        }
        Ok(pages)
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

    /// [`Self::hits_to_rows`] on the caller's runtime, score order kept, translated by
    /// [`Self::rows_of_ids`]. Through a map already kept, `hits` is translated in place, with no
    /// id or row buffer beside it.
    pub(crate) async fn hits_to_rows_async(
        &self,
        hits: Vec<(FtsDocId, f32)>,
    ) -> Result<Vec<(RowId, f32)>, FtsError> {
        if let Some(map) = self.checked_held_map(hits.iter().map(|&(doc, _)| doc))? {
            return Ok(hits
                .into_iter()
                .map(|(doc, score)| (map.row_of(doc), score))
                .collect());
        }
        let ids: Vec<FtsDocId> = hits.iter().map(|&(doc, _)| doc).collect();
        let rows = self.rows_of_ids(&ids, true).await?;
        Ok(rows
            .into_iter()
            .zip(hits)
            .map(|(row, (_, score))| (row, score))
            .collect())
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
        Ok(OrCursorSet {
            column_id,
            cursors,
            page_reads: Semaphore::new(DOC_MAP_BATCHES_IN_FLIGHT),
        })
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
        // that would release them. A term of one block or less scores from
        // its own documents' buckets when those can be read alone: on a
        // large superfile the whole array is megabytes for a handful of
        // postings. A weight from this file's own counts reads the array
        // anyway.
        let few = match &source {
            _ if !matches!(idf, TermIdf::Table(_)) => None,
            SingleSource::Inline { doc_id, .. } => self.few_norms(column_id, &[*doc_id]).await?,
            SingleSource::Bytes { bytes, short: true } => {
                let mut buf_d = [0u32; BLOCK_LEN];
                let mut buf_t = [0u32; BLOCK_LEN];
                match decode_short(bytes.as_ref(), col_meta.positions, &mut buf_d, &mut buf_t) {
                    Some(decoded) => self.few_norms(column_id, &buf_d[..decoded.n]).await?,
                    None => None,
                }
            }
            _ => None,
        };
        if few.is_none() {
            self.ensure_norms(column_id).await?;
        }
        let dl_norm_of = |doc: u32| match &few {
            Some(few) => few.get(doc),
            None => col_meta.dl_norm_k1().get(doc),
        };
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
                let dl_norm_k1 = dl_norm_of(doc_id);
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
                        .ok_or_else(short_body_malformed)?;
                let idf_weight = idf.resolve(col_meta, decoded.n as u64)?;
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
                        bm25::score_with_dl_norm_k1(idf_weight, buf_t[j], dl_norm_of(doc_id));
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
                        true => {
                            u64::from(short_df(bytes.as_ref()).ok_or_else(short_body_malformed)?)
                        }
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
        sync::{Arc, atomic::Ordering},
    };

    use bytes::Bytes;
    use futures::{future::join_all, poll};
    use roaring::RoaringBitmap;
    use tokio::{join, task::spawn_blocking};

    use super::{
        super::{metadata::OpenOptions, test_util::*},
        *,
    };
    use crate::{
        memory::ConnectionMemoryBudget,
        superfile::{
            fts::{builder::FtsBuilder, tokenize::Phrase},
            lazy_source::{BytesLazyByteSource, LazyByteSource, Source},
        },
        supertable::reader_cache::disk::HeapBudget,
    };

    /// A `k` above the fixture's match count.
    const TABLE_IDF_K: usize = 10;

    /// An arbitrary table-wide idf to score with.
    const TABLE_IDF: f32 = 1.5;
    /// Score slack between the ranked and the exact-count walks: they add
    /// a doc's term scores in different orders.
    const SUM_ORDER_EPS: f32 = 1e-4;
    /// Documents in the tie-at-the-floor fixture: enough for a long-form
    /// posting list of many blocks.
    const TIE_FLOOR_DOCS: u32 = 6_000;
    /// Table-wide idfs tried on each side of the term's own.
    const TIE_FLOOR_IDF_STEPS: i32 = 2_048;
    /// The relative step between them: a few `f32` steps of the idf ratio,
    /// so the rescale of the stored bounds rounds every way across the
    /// sweep.
    const TIE_FLOOR_IDF_STEP: f32 = 1.0 / 65_536.0;
    /// Doc ids in one doc-id map page.
    const IDS_PER_PAGE: u32 = (DOC_MAP_PAGE_BYTES / DOC_MAP_ENTRY_BYTES) as u32;
    /// Pages between hits that no read joins: one past the merge gap.
    const PAGES_APART: u32 = DOC_MAP_MERGE_GAP_PAGES as u32 + 2;
    /// Rows apart of the documents holding [`GAP_TERM`]: [`PAGES_APART`] pages of the test map.
    const GAP_EVERY: u32 = PAGES_APART * IDS_PER_PAGE;
    /// A term on every [`GAP_EVERY`]th row only.
    const GAP_TERM: &str = "gap";
    /// Slices a sliced match cuts one file into in the tests below: more than a file's reads in
    /// flight.
    const MANY_SLICES: u32 = 16;
    /// A term only the last document of [`dense_translation`]'s blob holds.
    const LAST_DOC_TERM: &str = "r999";
    /// Rows in the reordered fixture: a whole number of bitset words, and a
    /// count that leaves the last word part full.
    const ORDER_FIXTURE_ROWS: [u32; 2] = [4096, 4101];
    /// Every this many rows carry `beta`: a match dense enough for the bitset.
    const BETA_EVERY: u32 = 3;
    /// Every this many rows carry `rare`: a match sparse enough to be sorted.
    const RARE_EVERY: u32 = 128;
    /// Seed of the term lists the dedup test draws.
    const DEDUP_SEED: u64 = 11;
    /// A term list well past the scan's cut.
    const DEDUP_LONG_TERMS: usize = 500;
    /// A term list as long as a short prefix's expansion.
    const DEDUP_PREFIX_TERMS: usize = 5_000;

    /// The map path, which a prefix expansion's long term list takes, must
    /// give the scan path's output: distinct terms in first-seen order, each
    /// with its repeat count. Lists either side of the cut, with and without
    /// repeats, are checked against a plain scan.
    #[test]
    fn long_and_short_term_lists_dedup_alike() {
        let scan = |terms: &[&str]| {
            let mut distinct: Vec<String> = Vec::new();
            let mut qtf: Vec<u32> = Vec::new();
            for &t in terms {
                match distinct.iter().position(|d| d == t) {
                    Some(i) => qtf[i] += 1,
                    None => {
                        distinct.push(t.to_owned());
                        qtf.push(1);
                    }
                }
            }
            (distinct, qtf)
        };
        let mut rng = xorshift(DEDUP_SEED);
        for len in [
            0,
            1,
            2,
            DEDUP_SCAN_MAX_TERMS,
            DEDUP_SCAN_MAX_TERMS + 1,
            DEDUP_LONG_TERMS,
            DEDUP_PREFIX_TERMS,
        ] {
            for vocab in [len.max(1), (len / 3).max(1), 2] {
                let owned: Vec<String> = (0..len)
                    .map(|_| format!("t{}", rng() % vocab as u64))
                    .collect();
                let terms: Vec<&str> = owned.iter().map(String::as_str).collect();
                let (distinct, qtf) = FtsReader::dedup_terms_with_qtf(&terms);
                let (want_distinct, want_qtf) = scan(&terms);
                assert_eq!(distinct, want_distinct, "len {len}, vocab {vocab}");
                assert_eq!(qtf, want_qtf, "len {len}, vocab {vocab}");
                assert_eq!(qtf.iter().sum::<u32>() as usize, len);
            }
        }
    }

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
                let (ids, _) = r
                    .token_match_range_in_blob_ids(&set, pair[0], pair[1])
                    .expect("slice");
                let rows = r.ids_to_rows_async(ids).await.expect("rows");
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
            allow: Some(Admitted::Rows(Arc::clone(&allow))),
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
            allow: Some(Admitted::Rows(Arc::clone(&allow))),
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

    /// A floor at exactly the k-th score keeps the documents tied with it,
    /// at whatever table-wide idf the query scores with. The fan-out sets
    /// such a floor whenever the term index proves the k-th score before a
    /// superfile opens; the walk then skips a block only if its bound is
    /// below the floor, so a bound rescaled to the table's idf that rounds
    /// under its block's best score drops a document the top-k needs.
    /// Every top document here ties (same frequency, same length), so each
    /// block's best is the k-th score itself.
    #[tokio::test]
    async fn a_floor_at_the_kth_score_keeps_its_ties_at_any_table_idf() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for i in 0..TIE_FLOOR_DOCS {
            let mut text = match i.is_multiple_of(2) {
                true => "w ".repeat(1 + (i % 3) as usize),
                false => String::new(),
            };
            text.push_str(&"pad ".repeat(1 + (i % 29) as usize));
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let local = bm25::idf(
            u64::from(TIE_FLOOR_DOCS),
            u64::from(TIE_FLOOR_DOCS.div_ceil(2)),
        );
        for step in (-TIE_FLOOR_IDF_STEPS..=TIE_FLOOR_IDF_STEPS).filter(|s| *s != 0) {
            let idf = local * (1.0 + step as f32 * TIE_FLOOR_IDF_STEP);
            let table = GlobalTermIdf::from([("w".to_owned(), idf)]);
            let search = |floor: f32| {
                let table = &table;
                let r = &r;
                async move {
                    let clauses = ClauseLists {
                        shoulds: &["w"],
                        idf: QueryIdf::Table(table),
                        ..ClauseLists::default()
                    };
                    r.search_excluding("body", clauses, TABLE_IDF_K, floor)
                        .await
                        .expect("search")
                }
            };
            let unfloored = search(f32::NEG_INFINITY).await;
            assert_eq!(unfloored.len(), TABLE_IDF_K);
            let kth = unfloored[TABLE_IDF_K - 1].1;
            assert_eq!(search(kth).await, unfloored, "idf {idf} (local {local})");
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
                    allow: only_one.clone().map(Admitted::Rows),
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
                    allow: zero_two.clone().map(Admitted::Rows),
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
                    allow: zero_two.clone().map(Admitted::Rows),
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
                    allow: zero_two.map(Admitted::Rows),
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
                    allow: Some(Admitted::Rows(Arc::new([2u32, 3].into_iter().collect()))),
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
                    allow: Some(Admitted::Rows(Arc::new(RoaringBitmap::new()))),
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

    #[tokio::test]
    async fn map_pages_translate_ids_like_the_whole_map() {
        // Setup: a lazily read blob of 1000 documents in shuffled order.
        //  - Ids in one page, adjacent pages, a lone page and the short last page: read in pages,
        //    nothing kept, same rows as the whole map.
        //  - Ids past the sparse share, with room in the budget: the whole map read once and kept.
        //  - Hits through the kept map: translated in their own buffer; an id past the end refused.
        // This test pins that both ways name the same rows.
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
            FtsReader::open_lazy(src, &json, OpenOptions { verify_crc: true })
                .await
                .expect("open_lazy")
        };
        let last_page = (N_DOCS - 1) / IDS_PER_PAGE;
        assert!(
            !N_DOCS.is_multiple_of(IDS_PER_PAGE),
            "the map's last page must be shorter than the rest"
        );
        let row = |id: u32| RowId::new(order[id as usize]);

        // Pages 0 to 3 in one run, a lone page, and the last page.
        let ids: Vec<u32> = vec![
            last_page * IDS_PER_PAGE + 3,
            0,
            IDS_PER_PAGE - 1,
            IDS_PER_PAGE,
            3 * IDS_PER_PAGE,
            N_DOCS / 2,
            N_DOCS - 1,
            last_page * IDS_PER_PAGE,
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
        assert!(r.doc_map.get().is_none(), "a lazy reader keeps no map");

        // One id in every page, read with CRC checks on: past the sparse share, so the reader keeps
        // the whole map when the heap budget has room.
        let r = open().await.with_heap_budget(roomy(None));
        let every_page: Vec<u32> = (0..=last_page).map(|p| p * IDS_PER_PAGE).collect();
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
        assert_eq!(r.doc_map_reads(), 0, "one whole read, no page reads");
        assert!(r.doc_map.get().is_some(), "the reader kept the map");

        // Through the kept map, the hits are translated in their own buffer, and an id past the
        // blob's documents is still refused.
        let hits: Vec<(FtsDocId, f32)> = ids
            .iter()
            .map(|&id| (FtsDocId::new(id), id as f32))
            .collect();
        let buffer = hits.as_ptr() as usize;
        let got = r
            .hits_to_rows_async(hits)
            .await
            .expect("hits_to_rows_async");
        let want: Vec<(RowId, f32)> = ids.iter().map(|&id| (row(id), id as f32)).collect();
        assert_eq!(got, want);
        // std reuses a vector's buffer when the element types match in size and alignment, but
        // does not promise to: a failure here means check the allocation, not the rows.
        assert_eq!(got.as_ptr() as usize, buffer, "the hits' buffer is reused");
        let past = vec![(FtsDocId::new(0), 1.0), (FtsDocId::new(N_DOCS), 1.0)];
        assert!(r.hits_to_rows_async(past).await.is_err());
    }

    #[test]
    #[expect(
        clippy::single_range_in_vec_init,
        reason = "a read plan of one range is the expected value"
    )]
    fn map_reads_join_close_pages_and_batches_stay_within_one_cap() {
        // Setup: page lists turned into reads, then reads grouped for flight.
        //  - Duplicate and adjacent pages make one run; a gap of the merge gap is read through; a
        //    wider gap splits; a run splits at the read cap; the last read ends with the map.
        //  - Scattered pages stay one read each.
        //  - Groups hold at most one cap of bytes; reads adding up to exactly a cap go together.
        // This test pins the read plan a translation sends to the store.
        let page = DOC_MAP_PAGE_BYTES;
        let cap = DOC_MAP_READ_MAX_PAGES;
        let far = DOC_MAP_MERGE_GAP_PAGES + 2;
        let map_len = 100 * page - 1;
        let pages = |ps: &[usize]| map_reads(ps.iter().map(|p| p * page), map_len);
        assert_eq!(pages(&[0, 0, 1]), [0..2 * page], "one run");
        assert_eq!(
            pages(&[0, 1 + DOC_MAP_MERGE_GAP_PAGES]),
            [0..(2 + DOC_MAP_MERGE_GAP_PAGES) * page],
            "a gap of the merge gap is read through"
        );
        assert_eq!(
            pages(&[0, far]),
            [0..page, far * page..(far + 1) * page],
            "a wider gap splits"
        );
        let every: Vec<usize> = (0..2 * cap).collect();
        assert_eq!(
            pages(&every),
            [0..cap * page, cap * page..2 * cap * page],
            "split at the cap"
        );
        assert_eq!(
            pages(&[99]),
            [99 * page..map_len],
            "the last page ends with the map"
        );

        let spread: Vec<usize> = (0..100).step_by(far).collect();
        assert_eq!(
            pages(&spread).len(),
            spread.len(),
            "scattered pages stay one read each"
        );

        let scattered: Vec<usize> = (0..cap).map(|i| i * far).collect();
        let reads = pages(&scattered);
        assert_eq!(
            in_flight_batches(&reads, cap * page),
            [&reads[..]],
            "pages adding up to a cap go together"
        );
        let reads = pages(&every);
        assert_eq!(
            in_flight_batches(&reads, cap * page),
            [&reads[..1], &reads[1..]],
            "one cap at a time"
        );
    }

    #[tokio::test]
    async fn a_translation_refuses_an_id_past_the_map() {
        // Setup: a reordered blob of 100 documents and an id one past its last, as only a damaged
        // posting could name.
        //  - The translation returns a malformed-read error instead of row 0.
        // This test pins that the translation refuses it as malformed.
        const N_DOCS: u32 = 100;
        let docs: Vec<String> = (0..N_DOCS).map(|row| format!("alpha r{row}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let order: Vec<u32> = (0..N_DOCS).rev().collect();
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let src: Arc<dyn LazyByteSource> = Arc::new(AsyncOnlySource(blob));
        let r = FtsReader::open_lazy(src, &json, OpenOptions { verify_crc: false })
            .await
            .expect("open_lazy");
        assert!(matches!(
            r.ids_to_rows_async(vec![FtsDocId::new(0), FtsDocId::new(N_DOCS)])
                .await,
            Err(FtsError::Read(ReadError::Malformed(_)))
        ));
    }

    #[tokio::test]
    async fn a_translation_covering_the_whole_map_refuses_a_damaged_one() {
        // Setup: a blob whose doc-id map has one flipped byte, and readers that check CRCs.
        //  - Every id, read in pages (no room in the heap budget) or whole to keep the map.
        //  - Ids [`PAGES_APART`] pages apart, dense but with gaps, read in pages.
        // This test pins that each way refuses the damaged map rather than naming wrong rows.
        const N_DOCS: u32 = 1000;
        let (damaged, json) = damaged_dense_blob().await;
        let open = |blob: Bytes| {
            let json = json.clone();
            async move {
                let src: Arc<dyn LazyByteSource> = Arc::new(AsyncOnlySource(blob));
                FtsReader::open_lazy(src, &json, OpenOptions { verify_crc: true })
                    .await
                    .expect("open_lazy")
            }
        };
        let every_id: Vec<FtsDocId> = (0..N_DOCS).map(FtsDocId::new).collect();
        for (budget, in_pages) in [(Some(0), true), (None, false)] {
            let r = open(damaged.clone()).await.with_heap_budget(roomy(budget));
            assert!(
                matches!(
                    r.ids_to_rows_async(every_id.clone()).await,
                    Err(FtsError::Read(ReadError::ChecksumMismatch {
                        section: "fts/doc-map",
                        ..
                    }))
                ),
                "budget {budget:?}"
            );
            assert_eq!(r.doc_map_reads() > 0, in_pages, "budget {budget:?}");
        }
        // Hits [`PAGES_APART`] pages apart are dense but leave gaps; without room for the map the
        // pass still reads every page, so the damage is caught.
        let gapped: Vec<FtsDocId> = (0..N_DOCS)
            .step_by((PAGES_APART * IDS_PER_PAGE) as usize)
            .map(FtsDocId::new)
            .collect();
        let r = open(damaged).await.with_heap_budget(roomy(Some(0)));
        assert!(matches!(
            r.ids_to_rows_async(gapped).await,
            Err(FtsError::Read(ReadError::ChecksumMismatch {
                section: "fts/doc-map",
                ..
            }))
        ));
    }

    /// A heap budget of `budget` bytes (`None`: no limit) in a process with memory to spare, so
    /// only the budget decides what is kept.
    fn roomy(budget: Option<u64>) -> &'static HeapBudget {
        let heap = HeapBudget::with_budget(budget);
        heap.set_short_of_memory(false);
        heap
    }

    /// A reordered blob of 1000 documents, the row each doc id stands for, and one hit in every
    /// page of its doc-id map, in descending id order like ranked hits.
    fn dense_translation() -> (Bytes, String, Vec<u32>, Vec<(FtsDocId, f32)>) {
        const N_DOCS: u32 = 1000;
        let docs: Vec<String> = (0..N_DOCS).map(|row| format!("alpha r{row}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let order: Vec<u32> = (0..N_DOCS).rev().collect();
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let hits = (0..N_DOCS)
            .step_by(IDS_PER_PAGE as usize)
            .rev()
            .map(|id| (FtsDocId::new(id), id as f32))
            .collect();
        (blob, json, order, hits)
    }

    /// `blob` opened lazily, its kept doc-id map charged to `budget`.
    async fn open_charged(blob: Bytes, json: &str, budget: &'static HeapBudget) -> FtsReader {
        let src: Arc<dyn LazyByteSource> = Arc::new(AsyncOnlySource(blob));
        FtsReader::open_lazy(src, json, OpenOptions { verify_crc: true })
            .await
            .expect("open_lazy")
            .with_heap_budget(budget)
    }

    #[tokio::test]
    async fn without_room_a_dense_translation_reads_pages_and_keeps_nothing() {
        // Setup: hits in every page of the map, unsorted, past the sparse share, so the reader asks
        // to keep the whole map; a heap budget with no room.
        //  - The budget says no, so it reads the pages in several reads and keeps nothing.
        // This test pins that a refused keep still answers the same rows and charges nothing.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(Some(0));
        let r = open_charged(blob, &json, budget).await;
        let want: Vec<(RowId, f32)> = hits
            .iter()
            .map(|&(id, score)| (RowId::new(order[id.get() as usize]), score))
            .collect();
        let got = r.hits_to_rows_async(hits).await.expect("translate");
        assert_eq!(got, want);
        assert!(r.doc_map_reads() > 1, "read in pages");
        assert!(!r.doc_map_is_read(), "nothing kept");
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn without_room_a_dense_pass_holds_a_few_read_caps_at_a_time() {
        // Setup: hits in every page of a map more read caps long than a pass has in flight, a
        // budget with no room, the reads held at the source.
        //  - The pass starts that many reads of one read cap each, and no more until one is back.
        // This test pins what one translation holds while it reads a map it can't keep.
        let (blob, json, _, hits) = dense_translation();
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: true },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(roomy(Some(0)));
        let opened = src.started.load(Ordering::Relaxed);
        let held = src.gate.write().await;
        let mut query = Box::pin(r.hits_to_rows_async(hits));
        assert!(poll!(query.as_mut()).is_pending());
        assert_eq!(
            src.started.load(Ordering::Relaxed) - opened,
            DOC_MAP_BATCHES_IN_FLIGHT,
            "a few read caps in flight"
        );
        drop(held);
        query.await.expect("translate");
        assert!(r.doc_map_reads() > DOC_MAP_BATCHES_IN_FLIGHT);
    }

    #[tokio::test]
    async fn a_dense_pass_checks_its_crc_in_map_order_when_reads_come_back_out_of_order() {
        // Setup: hits in every page of the map, a budget with no room, and a source whose later
        // reads come back first.
        //  - The pass still answers every row and its CRC check still passes.
        // This test pins that the reads in flight are taken in map order.
        let (blob, json, order, hits) = dense_translation();
        let src: Arc<dyn LazyByteSource> = Arc::new(LaterFirstSource(blob));
        let r = FtsReader::open_lazy(src, &json, OpenOptions { verify_crc: true })
            .await
            .expect("open_lazy")
            .with_heap_budget(roomy(Some(0)));
        let want: Vec<(RowId, f32)> = hits
            .iter()
            .map(|&(id, score)| (RowId::new(order[id.get() as usize]), score))
            .collect();
        assert_eq!(r.hits_to_rows_async(hits).await.expect("translate"), want);
        assert!(
            r.doc_map_reads() > DOC_MAP_BATCHES_IN_FLIGHT,
            "read in pages"
        );
    }

    #[tokio::test]
    async fn short_of_memory_a_dense_translation_keeps_nothing() {
        // Setup: hits in every page of the map, past the sparse share, so the reader asks to keep
        // the whole map; a budget with room for any map, and a process short of memory.
        //  - The answer is no: the rows are the same, nothing is kept and nothing is charged.
        // This test pins that memory pressure, not only the budget, stops a keep.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(None);
        budget.set_short_of_memory(true);
        let r = open_charged(blob, &json, budget).await;
        let want: Vec<(RowId, f32)> = hits
            .iter()
            .map(|&(id, score)| (RowId::new(order[id.get() as usize]), score))
            .collect();
        assert_eq!(r.hits_to_rows_async(hits).await.expect("translate"), want);
        assert!(!r.doc_map_is_read(), "nothing kept");
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn without_room_a_sliced_match_keeps_no_map() {
        // Setup: a reordered blob where every document matches, cut into slices like a sliced
        // `token_match`; a heap budget with no room.
        //  - Building the slices' cursors keeps no map (it reads every page once, for the CRC).
        //  - Each slice's ids turn into its rows through its own map pages, and nothing is kept.
        // This test pins that an unranked match over many large files can't keep every file's map
        // past the budget.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(Some(0));
        let r = open_charged(blob, &json, budget).await;
        let (set, _) = r
            .match_cursor_set("body", &["alpha"], None)
            .await
            .expect("cursor set");
        assert!(!r.doc_map_is_read(), "the cursors keep no map");
        let checked = r.doc_map_reads();
        let n_docs = order.len() as u32;
        let mut rows = Vec::new();
        for (start, end) in [(0, n_docs / 2), (n_docs / 2, n_docs)] {
            let (ids, _) = r
                .token_match_range_in_blob_ids(&set, start, end)
                .expect("slice");
            assert!(r.held_ids_to_rows(&ids).expect("held").is_none());
            rows.extend(r.ids_to_rows_in_pages(ids, &set).await.expect("rows"));
        }
        rows.sort_unstable();
        let every_row: Vec<RowId> = (0..n_docs).map(RowId::new).collect();
        assert_eq!(rows, every_row);
        assert!(
            r.doc_map_reads() > checked,
            "the slices read their own pages"
        );
        assert!(!r.doc_map_is_read(), "nothing kept");
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn without_room_a_contains_match_keeps_no_map() {
        // Setup: a reordered blob where every document holds `alpha`; a heap budget with no room.
        //  - An `ILIKE '%lph%'` answered from the dictionary turns its documents into rows through
        //    map pages, and nothing is kept.
        // This test pins that a text filter in a SQL `WHERE` can't keep every file's map past the
        // budget.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(Some(0));
        let r = open_charged(blob, &json, budget).await;
        let (found, _) = r
            .contains_rows("body", &["lph"], None, None)
            .await
            .expect("contains");
        let every_row: RoaringBitmap = (0..order.len() as u32).collect();
        assert_eq!(found[0].proven, every_row);
        assert!(r.doc_map_reads() > 1, "read in pages");
        assert!(!r.doc_map_is_read(), "nothing kept");
        assert_eq!(budget.charged(), 0);
    }

    /// The rows a sliced match of `term` over `r` finds, `slices` slices of it, each through the
    /// held map or its pages as the slice's file decided.
    async fn sliced_rows_of(r: &FtsReader, term: &str, slices: u32) -> Vec<RowId> {
        let (set, _) = r
            .match_cursor_set("body", &[term], None)
            .await
            .expect("cursor set");
        let n_docs = r.n_docs();
        let mut rows = Vec::new();
        for slice in 0..slices {
            let (start, end) = (n_docs * slice / slices, n_docs * (slice + 1) / slices);
            let (ids, _) = r
                .token_match_range_in_blob_ids(&set, start, end)
                .expect("slice");
            match r.held_ids_to_rows(&ids).expect("held") {
                Some(held) => rows.extend(held),
                None => rows.extend(r.ids_to_rows_in_pages(ids, &set).await.expect("pages")),
            }
        }
        rows.sort_unstable();
        rows
    }

    #[tokio::test]
    async fn with_room_a_sliced_match_keeps_its_map_once() {
        // Setup: a reordered blob where every document matches, cut into many slices; a heap
        // budget with room.
        //  - The match keeps the whole map once, when it builds the cursors, and every slice
        //    translates through it: no page reads.
        // This test pins that the keep is decided for the file, not judged slice by slice.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let rows = sliced_rows_of(&r, "alpha", MANY_SLICES).await;
        let every_row: Vec<RowId> = (0..order.len() as u32).map(RowId::new).collect();
        assert_eq!(rows, every_row);
        assert!(r.doc_map_is_read(), "the map is kept");
        assert_eq!(r.doc_map_reads(), 0, "one whole read, no page reads");
        assert!(budget.charged() > 0, "and charged");
    }

    #[tokio::test]
    async fn without_room_a_files_slices_share_its_page_reads() {
        // Setup: a reordered blob where every document matches, in two slices that each need more
        // read caps than a file may have in flight; no room in the heap budget; the reads held at
        // the source.
        //  - Both slices start reading at once, and only a file's share of reads starts between
        //    them; the rest wait until those are back.
        // This test pins that a sliced file holds no more map pages in flight than an unsliced one.
        let (blob, json, order, _) = dense_translation();
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: false },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(roomy(Some(0)));
        let (set, _) = r
            .match_cursor_set("body", &["alpha"], None)
            .await
            .expect("cursor set");
        let n_docs = order.len() as u32;
        let halves = [(0, n_docs / 2), (n_docs / 2, n_docs)].map(|(start, end)| {
            r.token_match_range_in_blob_ids(&set, start, end)
                .expect("slice")
                .0
        });
        let opened = src.started.load(Ordering::Relaxed);
        let held = src.gate.write().await;
        let mut both = Box::pin(join_all(
            halves.map(|ids| r.ids_to_rows_in_pages(ids, &set)),
        ));
        assert!(poll!(both.as_mut()).is_pending());
        assert_eq!(
            src.started.load(Ordering::Relaxed) - opened,
            DOC_MAP_BATCHES_IN_FLIGHT,
            "one file's reads in flight across its slices"
        );
        drop(held);
        let rows: usize = both
            .await
            .into_iter()
            .map(|rows| rows.expect("rows").len())
            .sum();
        assert_eq!(rows, order.len());
    }

    #[tokio::test]
    async fn without_room_a_sliced_match_reads_its_pages_and_keeps_nothing() {
        // Setup: the same match in many slices; a heap budget with no room.
        //  - No slice tries to keep the map again: each reads its own pages.
        // This test pins that a refused keep is decided once and the rows still all come back.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(Some(0));
        let r = open_charged(blob, &json, budget).await;
        let rows = sliced_rows_of(&r, "alpha", MANY_SLICES).await;
        let every_row: Vec<RowId> = (0..order.len() as u32).map(RowId::new).collect();
        assert_eq!(rows, every_row);
        assert!(
            r.doc_map_reads() >= MANY_SLICES as usize,
            "every slice read pages"
        );
        assert!(!r.doc_map_is_read(), "nothing kept");
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_broad_contains_holds_its_bitset_not_a_list() {
        // Setup: a reordered blob where every document holds `alpha`; a heap budget with room.
        //  - A contains matching every document holds about what one matching a single document
        //    holds (both walk the whole dictionary): the bitset, never a list of the documents.
        // This test pins that a broad contains holds about a bit a document, not 8 bytes.
        let (blob, json, order, _) = dense_translation();
        let r = open_charged(blob, &json, roomy(None)).await;
        let held = |needle: &'static str| {
            let r = &r;
            async move {
                let connection = ConnectionMemoryBudget::measured();
                let (found, _) = r
                    .contains_rows("body", &[needle], None, Some(&connection))
                    .await
                    .expect("contains");
                (found[0].proven.len(), connection.peak())
            }
        };
        let (one, narrow) = held(LAST_DOC_TERM).await;
        let (every, broad) = held("lph").await;
        let n_docs = order.len();
        assert_eq!((one, every), (1, n_docs as u64));
        assert!(r.doc_map_is_read(), "the map is kept");
        assert!(
            broad - narrow < n_docs * size_of::<u32>(),
            "held {} bytes more for {n_docs} documents",
            broad - narrow
        );
    }

    #[tokio::test]
    async fn with_room_a_one_document_sliced_match_keeps_its_map() {
        // Setup: a reordered blob and a term only one document holds; a heap budget with room.
        //  - The match keeps the whole map.
        // This test pins that keeping is decided by room alone, not by how many documents match.
        let (blob, json, _, _) = dense_translation();
        let r = open_charged(blob, &json, roomy(None)).await;
        assert_eq!(
            sliced_rows_of(&r, LAST_DOC_TERM, MANY_SLICES).await.len(),
            1
        );
        assert!(r.doc_map_is_read(), "the map is kept");
    }

    #[tokio::test]
    async fn without_room_a_rare_contains_reads_only_its_page() {
        // Setup: a reordered blob and a needle only its last document holds; no room in the heap
        // budget.
        //  - The contains reads the one map page that document falls in, keeps nothing, and names
        //    its row.
        // This test pins the contains page pass when it reads few pages, with gaps and no CRC.
        let (blob, json, order, _) = dense_translation();
        let r = open_charged(blob, &json, roomy(Some(0))).await;
        let (found, _) = r
            .contains_rows("body", &[LAST_DOC_TERM], None, None)
            .await
            .expect("contains");
        let last_row = order.len() as u32 - 1;
        assert_eq!(found[0].proven, RoaringBitmap::from_iter([last_row]));
        assert_eq!(r.doc_map_reads(), 1, "one page");
        assert!(!r.doc_map_is_read(), "nothing kept");
    }

    #[tokio::test]
    async fn without_room_a_sliced_match_and_a_contains_still_check_the_map_crc() {
        // Setup: a blob whose doc-id map has one flipped byte, a reader that checks CRCs, and no
        // room in the heap budget, so nothing is kept.
        //  - A sliced match over every document reads every page once and refuses the map.
        //  - So does a contains over every document.
        //  - So does a contains over documents [`PAGES_APART`] pages apart, damaged in a page it
        //    doesn't touch: dense with gaps, it reads every page to check the CRC.
        // This test pins that reading pages instead of keeping the map still checks its CRC.
        let (damaged, json) = damaged_dense_blob().await;
        let refused = |result: Result<(), FtsError>| {
            matches!(
                result,
                Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/doc-map",
                    ..
                }))
            )
        };
        let r = open_charged(damaged.clone(), &json, roomy(Some(0))).await;
        let built = r.match_cursor_set("body", &["alpha"], None).await;
        assert!(refused(built.map(drop)), "the sliced match");
        let r = open_charged(damaged, &json, roomy(Some(0))).await;
        let found = r.contains_rows("body", &["lph"], None, None).await;
        assert!(refused(found.map(drop)), "the contains");
        let (gapped, json) = damaged_gapped_blob().await;
        let r = open_charged(gapped, &json, roomy(Some(0))).await;
        let found = r.contains_rows("body", &[GAP_TERM], None, None).await;
        assert!(refused(found.map(drop)), "the gapped contains");
    }

    /// A reordered blob of 1000 documents where every [`GAP_EVERY`]th row also holds [`GAP_TERM`],
    /// so that term's documents fall [`PAGES_APART`] pages apart in the map, with one byte of a
    /// page they don't touch flipped.
    async fn damaged_gapped_blob() -> (Bytes, String) {
        const N_DOCS: u32 = 1000;
        let docs: Vec<String> = (0..N_DOCS)
            .map(|row| match row % GAP_EVERY {
                0 => format!("alpha r{row} {GAP_TERM}"),
                _ => format!("alpha r{row}"),
            })
            .collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let order: Vec<u32> = (0..N_DOCS).rev().collect();
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let region = open_charged(blob.clone(), &json, roomy(None))
            .await
            .doc_map_region
            .clone()
            .expect("a reordered blob has a map");
        let untouched = region.start + region.len() / 2;
        let mut damaged = blob.to_vec();
        damaged[untouched] ^= u8::MAX;
        (Bytes::from(damaged), json)
    }

    /// Ranked hits in three pages of [`dense_translation`]'s map: the first, a full page near the
    /// end, and the short last one. None is in the middle page [`damaged_dense_blob`] breaks, and
    /// each takes one read ([`SPARSE_READS`]).
    fn sparse_hits() -> Vec<(FtsDocId, f32)> {
        vec![
            (FtsDocId::new(999), 3.0),
            (FtsDocId::new(900), 2.0),
            (FtsDocId::new(0), 1.0),
        ]
    }

    /// Map reads [`sparse_hits`] makes, one per page.
    const SPARSE_READS: usize = 3;

    /// A heap budget with room for [`sparse_hits`]' pages but not for [`dense_translation`]'s whole
    /// map.
    const ROOM_FOR_PAGES_NOT_MAP: u64 = 1000;

    /// A heap budget with room for any two of [`sparse_hits`]' three pages, not all three.
    const ROOM_FOR_TWO_PAGES: u64 = 2 * DOC_MAP_PAGE_BYTES as u64 + 8;

    /// Translations the concurrency test runs at once.
    const CONCURRENT: usize = 8;

    /// Bytes of [`sparse_hits`]' pages on `r`'s map: two full pages and the short last one.
    fn sparse_page_bytes(r: &FtsReader) -> u64 {
        let tail = r.doc_map_len() % DOC_MAP_PAGE_BYTES;
        assert!(tail > 0, "the fixture's last page must be short");
        (2 * DOC_MAP_PAGE_BYTES + tail) as u64
    }

    /// The rows `hits` stand for on a blob stored in `order`.
    fn rows_of(order: &[u32], hits: &[(FtsDocId, f32)]) -> Vec<(RowId, f32)> {
        hits.iter()
            .map(|&(id, score)| (RowId::new(order[id.get() as usize]), score))
            .collect()
    }

    #[tokio::test]
    async fn with_room_a_sparse_translation_keeps_its_pages_and_a_repeat_reads_none() {
        // Setup: hits in three far-apart pages of a reordered blob's map; a budget with room.
        //  - The first translation reads the three pages and keeps them, reserving their bytes.
        //  - The same translation again reads nothing and answers the same rows.
        // This test pins that a warm ranked search reads no map page it read before.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let hits = sparse_hits();
        let want = rows_of(&order, &hits);
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("cold"),
            want
        );
        assert_eq!(r.doc_map_reads(), SPARSE_READS, "one read per page");
        assert_eq!(
            budget.charged(),
            sparse_page_bytes(&r),
            "the pages kept, the short last one at its length"
        );
        assert!(!r.doc_map_is_read(), "the whole map is not kept");
        assert_eq!(r.hits_to_rows_async(hits).await.expect("warm"), want);
        assert_eq!(
            r.doc_map_reads(),
            SPARSE_READS,
            "a warm translation reads nothing"
        );
    }

    #[tokio::test]
    async fn concurrent_sparse_translations_keep_one_copy_of_each_page() {
        // Setup: the same hits translated eight times at once while the source holds every read.
        //  - Each reads at most its own pages, as with nothing kept, and answers the same rows.
        //  - One copy of each page is kept and reserved, so a repeat reads none.
        // This test pins that searches racing on a page read no more than without keeping, and
        // keep it once.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: true },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(budget);
        let hits = sparse_hits();
        let held = src.gate.write().await;
        let mut all = Box::pin(join_all(
            (0..CONCURRENT).map(|_| r.hits_to_rows_async(hits.clone())),
        ));
        assert!(poll!(all.as_mut()).is_pending());
        drop(held);
        for got in all.await {
            assert_eq!(got.expect("translate"), rows_of(&order, &hits));
        }
        let reads = r.doc_map_reads();
        assert!(reads <= CONCURRENT * SPARSE_READS, "{reads} reads");
        assert_eq!(
            budget.charged(),
            sparse_page_bytes(&r),
            "one copy of each page"
        );
        r.hits_to_rows_async(hits).await.expect("repeat");
        assert_eq!(r.doc_map_reads(), reads, "a repeat reads none");
    }

    #[tokio::test]
    async fn without_room_a_sparse_translation_keeps_no_page_and_reads_again() {
        // Setup: hits in three pages; a budget with no room.
        //  - Each translation reads every page and keeps nothing, so a repeat reads them again.
        // This test pins that a full budget keeps no page and reserves nothing.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(Some(0));
        let r = open_charged(blob, &json, budget).await;
        let hits = sparse_hits();
        for _ in 0..2 {
            assert_eq!(
                r.hits_to_rows_async(hits.clone()).await.expect("translate"),
                rows_of(&order, &hits)
            );
        }
        assert_eq!(
            r.doc_map_reads(),
            2 * SPARSE_READS,
            "every page read each time"
        );
        assert_eq!(budget.charged(), 0, "nothing kept");
        assert_eq!(r.kept_pages.page_count(), 0);
    }

    #[tokio::test]
    async fn short_of_memory_a_sparse_translation_keeps_no_page_and_reads_again() {
        // Setup: three hits in three pages, a budget with no limit, a process short of memory.
        //  - Each translation reads its pages and keeps none, so a repeat reads them again.
        // This test pins that memory pressure alone refuses page keeps the budget has room for.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        budget.set_short_of_memory(true);
        let r = open_charged(blob, &json, budget).await;
        let hits = sparse_hits();
        for _ in 0..2 {
            assert_eq!(
                r.hits_to_rows_async(hits.clone()).await.expect("translate"),
                rows_of(&order, &hits)
            );
        }
        assert_eq!(
            r.doc_map_reads(),
            2 * SPARSE_READS,
            "every page read each time"
        );
        assert_eq!(budget.charged(), 0, "nothing kept");
        assert_eq!(r.kept_pages.page_count(), 0);
    }

    #[tokio::test]
    async fn a_page_read_while_the_whole_map_is_kept_is_refused() {
        // Setup: the whole map is kept, then a page is offered, as by a search whose read was in
        // flight when the map was kept.
        //  - The page is refused and its reservation given back; only the map stays reserved.
        // This test pins that a reader never holds pages beside its whole map.
        let (blob, json, _, hits) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        r.hits_to_rows_async(hits).await.expect("dense");
        assert!(r.doc_map_is_read(), "the whole map is kept");
        let charged = budget.charged();
        r.kept_pages
            .keep_all(&[(0, Bytes::from(vec![0; DOC_MAP_PAGE_BYTES]))], budget);
        assert_eq!(r.kept_pages.page_count(), 0, "the page is refused");
        assert_eq!(budget.charged(), charged, "only the map is charged");
    }

    #[tokio::test]
    async fn a_translation_racing_the_whole_map_keep_keeps_no_page() {
        // Setup: a dense translation keeps the whole map, then a sparse one takes the page path,
        // as one that checked for the map just before it was kept does.
        //  - It takes its pages from the kept map, reads nothing and answers the right rows.
        //  - It keeps no page; only the map stays reserved.
        // This test pins that a search racing the whole-map keep reads nothing and keeps no page.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        r.hits_to_rows_async(hits).await.expect("dense");
        assert!(r.doc_map_is_read(), "the whole map is kept");
        let sparse = sparse_hits();
        let ids: Vec<FtsDocId> = sparse.iter().map(|&(id, _)| id).collect();
        let reads = map_reads(
            ids.iter()
                .rev()
                .map(|id| id.get() as usize * DOC_MAP_ENTRY_BYTES),
            r.doc_map_len(),
        );
        let reads_before = r.doc_map_reads();
        let rows = r
            .rows_through_kept_pages(&ids, &reads)
            .await
            .expect("pages");
        assert_eq!(
            r.doc_map_reads(),
            reads_before,
            "answered from the kept map, no read"
        );
        let want: Vec<RowId> = rows_of(&order, &sparse)
            .into_iter()
            .map(|(row, _)| row)
            .collect();
        assert_eq!(rows, want);
        assert_eq!(
            budget.charged(),
            r.doc_map_len() as u64,
            "no page beside the map"
        );
        assert_eq!(r.kept_pages.page_count(), 0);
    }

    #[tokio::test]
    async fn a_map_kept_for_a_row_check_gives_its_pages_back() {
        // Setup: a sparse translation keeps pages, then a filter that checks rows reads the whole
        // map, which keeps it whatever the budget says.
        //  - The pages drop and give back their reservation; only the whole map stays reserved.
        // This test pins that every way of keeping the whole map drops the pages.
        let (blob, json, _, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        r.hits_to_rows_async(sparse_hits()).await.expect("sparse");
        assert_eq!(budget.charged(), sparse_page_bytes(&r));
        r.ensure_doc_map().await.expect("whole map");
        assert_eq!(
            budget.charged(),
            r.doc_map_len() as u64,
            "only the whole map is charged"
        );
        assert_eq!(r.kept_pages.page_count(), 0);
    }

    #[tokio::test]
    async fn a_whole_map_refused_by_the_budget_keeps_the_pages() {
        // Setup: a sparse translation keeps pages, then a dense one whose whole map does not fit
        // the budget's map share even with nothing else kept.
        //  - The budget refuses the map, and the dense one answers by reading pages.
        //  - The sparse pages stay kept, so repeating the sparse translation reads none.
        // This test pins that a refused whole-map keep leaves the kept pages alone.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(Some(ROOM_FOR_PAGES_NOT_MAP));
        let r = open_charged(blob, &json, budget).await;
        r.hits_to_rows_async(sparse_hits()).await.expect("sparse");
        let kept = budget.charged();
        assert_eq!(kept, sparse_page_bytes(&r));
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("dense"),
            rows_of(&order, &hits)
        );
        assert!(!r.doc_map_is_read(), "no room for the whole map");
        assert_eq!(budget.charged(), kept, "the pages are still kept");
        let reads = r.doc_map_reads();
        r.hits_to_rows_async(sparse_hits())
            .await
            .expect("sparse again");
        assert_eq!(r.doc_map_reads(), reads, "a repeat reads none");
    }

    #[tokio::test]
    async fn a_budget_with_room_for_some_pages_reads_each_page_once() {
        // Setup: a budget with room for two of the three pages a sparse translation needs.
        //  - Two pages are kept and the third is only read: one read per page, none read twice.
        //  - A repeat reads only the page not kept, and reserves nothing more.
        // This test pins that a budget refusing a page costs no extra read.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(Some(ROOM_FOR_TWO_PAGES));
        let r = open_charged(blob, &json, budget).await;
        let hits = sparse_hits();
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("translate"),
            rows_of(&order, &hits)
        );
        assert_eq!(r.doc_map_reads(), SPARSE_READS, "one read per page");
        let charged = budget.charged();
        assert!(
            charged > 0 && charged < sparse_page_bytes(&r),
            "some pages kept, not all: {charged} B"
        );
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("repeat"),
            rows_of(&order, &hits)
        );
        assert_eq!(
            r.doc_map_reads(),
            SPARSE_READS + 1,
            "only the page not kept"
        );
        assert_eq!(budget.charged(), charged, "nothing more reserved");
    }

    #[tokio::test]
    async fn an_unranked_lookup_keeps_no_page() {
        // Setup: sorted doc ids in three far-apart pages, looked up unranked; a budget with room.
        //  - The lookup reads the three pages and keeps none of them.
        //  - A repeat reads them again.
        // This test pins that only a ranked search's few top hits keep pages: an unranked lookup
        // can touch 1/8 of the map.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let mut ids: Vec<FtsDocId> = sparse_hits().into_iter().map(|(id, _)| id).collect();
        ids.sort_unstable();
        let mut want: Vec<RowId> = rows_of(&order, &sparse_hits())
            .into_iter()
            .map(|(row, _)| row)
            .collect();
        want.sort_unstable();
        for reads in [SPARSE_READS, 2 * SPARSE_READS] {
            assert_eq!(
                r.ids_to_rows_async(ids.clone()).await.expect("lookup"),
                want
            );
            assert_eq!(r.doc_map_reads(), reads, "the pages are read every time");
        }
        assert_eq!(r.kept_pages.page_count(), 0);
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_reader_keeps_at_most_its_page_cap() {
        // Setup: ranked hits in one more far-apart page than a reader keeps, still a sparse
        // translation; a budget with room.
        //  - Every page is read once, and only the cap's worth is kept and reserved.
        // This test pins that kept pages stay bounded per file.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let hits: Vec<(FtsDocId, f32)> = (0..=DOC_MAP_KEPT_PAGES_MAX as u32)
            .map(|i| (FtsDocId::new(i * PAGES_APART * IDS_PER_PAGE), 1.0))
            .collect();
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("translate"),
            rows_of(&order, &hits)
        );
        assert_eq!(
            r.doc_map_reads(),
            DOC_MAP_KEPT_PAGES_MAX + 1,
            "one read per page"
        );
        assert!(
            !r.doc_map_is_read(),
            "a sparse translation keeps no whole map"
        );
        assert_eq!(r.kept_pages.page_count(), DOC_MAP_KEPT_PAGES_MAX);
        assert_eq!(
            budget.charged(),
            (DOC_MAP_KEPT_PAGES_MAX * DOC_MAP_PAGE_BYTES) as u64
        );
    }

    #[tokio::test]
    async fn a_read_across_a_gap_keeps_only_the_pages_hit() {
        // Setup: hits in pages 0 and 2, which one read covers, page 1 included.
        //  - The read is made once, and only pages 0 and 2 are kept.
        //  - A repeat reads nothing, though page 1 is not kept.
        // This test pins that a merged read keeps only the pages hit, and a repeat still reuses
        // them.
        let (blob, json, order, _) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let hits = vec![
            (FtsDocId::new(2 * IDS_PER_PAGE), 2.0),
            (FtsDocId::new(0), 1.0),
        ];
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("translate"),
            rows_of(&order, &hits)
        );
        assert_eq!(r.doc_map_reads(), 1, "one read across the gap");
        assert_eq!(r.kept_pages.page_count(), 2, "the gap page is not kept");
        assert_eq!(budget.charged(), 2 * DOC_MAP_PAGE_BYTES as u64);
        assert_eq!(
            r.hits_to_rows_async(hits.clone()).await.expect("repeat"),
            rows_of(&order, &hits)
        );
        assert_eq!(r.doc_map_reads(), 1, "a repeat reads nothing");
    }

    #[tokio::test]
    async fn kept_pages_go_once_the_whole_map_is_kept() {
        // Setup: a sparse translation keeps its pages, then a dense one with room for the map.
        //  - The dense one keeps the whole map, and the pages drop and give back their reservation.
        // This test pins that a reader never holds a page and the whole map at once.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        r.hits_to_rows_async(sparse_hits()).await.expect("sparse");
        assert_eq!(budget.charged(), sparse_page_bytes(&r));
        let want = rows_of(&order, &hits);
        assert_eq!(r.hits_to_rows_async(hits).await.expect("dense"), want);
        assert!(r.doc_map_is_read(), "the whole map is kept");
        assert_eq!(
            budget.charged(),
            r.doc_map_len() as u64,
            "only the whole map is charged"
        );
    }

    #[tokio::test]
    async fn kept_pages_leave_a_dense_translation_checking_the_map_crc() {
        // Setup: a damaged map, a reader that checks CRCs, and pages kept by sparse hits clear of
        // the damage.
        //  - With room for the map, a dense translation reads it whole and fails its CRC check.
        //  - With room for a few pages only, the dense pass reads every page and fails the check.
        // This test pins that kept pages never let a dense translation skip the CRC check.
        let (damaged, json) = damaged_dense_blob().await;
        let (_, _, _, dense) = dense_translation();
        for budget in [None, Some(ROOM_FOR_PAGES_NOT_MAP)] {
            let r = open_charged(damaged.clone(), &json, roomy(budget)).await;
            r.hits_to_rows_async(sparse_hits())
                .await
                .expect("sparse hits avoid the damage");
            assert!(
                matches!(
                    r.hits_to_rows_async(dense.clone()).await,
                    Err(FtsError::Read(ReadError::ChecksumMismatch {
                        section: "fts/doc-map",
                        ..
                    }))
                ),
                "budget {budget:?}"
            );
        }
    }

    /// [`dense_translation`]'s blob with one byte of its doc-id map flipped.
    async fn damaged_dense_blob() -> (Bytes, String) {
        let (blob, json, _, _) = dense_translation();
        let region = open_charged(blob.clone(), &json, roomy(None))
            .await
            .doc_map_region
            .clone()
            .expect("a reordered blob has a map");
        let mut damaged = blob.to_vec();
        damaged[region.start + region.len() / 2] ^= u8::MAX;
        (Bytes::from(damaged), json)
    }

    #[tokio::test]
    async fn a_kept_map_is_charged_until_its_last_holder_drops() {
        // Setup: a dense translation on a reader with no budget limit, then a view of it.
        //  - The reader keeps the map in one read and is charged its length.
        //  - Dropping the reader leaves the charge, since the view still holds the map.
        //  - Dropping the view gives it back.
        // This test pins that a kept map's bytes stay charged exactly as long as the map lives.
        let (blob, json, _, hits) = dense_translation();
        let budget = roomy(None);
        let r = open_charged(blob, &json, budget).await;
        let map_len = r.doc_map_len() as u64;
        r.hits_to_rows_async(hits).await.expect("translate");
        assert_eq!(r.doc_map_reads(), 0, "one whole read");
        assert!(r.doc_map_is_read(), "kept");
        assert_eq!(budget.charged(), map_len);
        let view = r.clone();
        drop(r);
        assert_eq!(budget.charged(), map_len, "the view still holds it");
        drop(view);
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_budget_with_room_for_one_map_keeps_one() {
        // Setup: two readers of the same blob, sharing a budget whose maps' share has room for
        // exactly one map.
        //  - The first dense translation keeps its map.
        //  - The second finds no room, reads pages and keeps nothing.
        // This test pins that the budget, not the hit count alone, decides what is kept.
        let (blob, json, order, hits) = dense_translation();
        let map_len = (order.len() * DOC_MAP_ENTRY_BYTES) as u64;
        let budget = roomy(Some(2 * map_len));
        let first = open_charged(blob.clone(), &json, budget).await;
        let second = open_charged(blob, &json, budget).await;
        let a = first.hits_to_rows_async(hits.clone()).await.expect("first");
        let b = second.hits_to_rows_async(hits).await.expect("second");
        assert_eq!(a, b);
        assert!(first.doc_map_is_read() && !second.doc_map_is_read());
        assert!(second.doc_map_reads() > 0, "the second read pages");
        assert_eq!(budget.charged(), map_len);
    }

    #[tokio::test]
    async fn racing_queries_keep_one_map_and_charge_it_once() {
        // Setup: two dense translations on one reader, both held while the map is being read.
        //  - The first reserves the map's bytes and starts the read.
        //  - The second waits for that read and reserves nothing.
        //  - Once the read is let go, both answer from the one kept map.
        // This test pins that queries racing to keep a map charge it once, not once each.
        let (blob, json, _, hits) = dense_translation();
        let budget = roomy(None);
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: true },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(budget);
        let map_len = r.doc_map_len() as u64;
        let held = src.gate.write().await;
        let mut first = Box::pin(r.hits_to_rows_async(hits.clone()));
        let mut second = Box::pin(r.hits_to_rows_async(hits));
        assert!(poll!(first.as_mut()).is_pending());
        assert!(poll!(second.as_mut()).is_pending());
        assert_eq!(budget.charged(), map_len, "one reservation while both wait");
        drop(held);
        let (a, b) = join!(first, second);
        assert_eq!(a.expect("first"), b.expect("second"));
        assert_eq!(r.doc_map_reads(), 0, "one whole read");
        assert_eq!(budget.charged(), map_len);
    }

    #[tokio::test]
    async fn a_map_kept_regardless_of_room_is_still_charged() {
        // Setup: a budget with no room, and the two reads that keep the map whatever it says.
        //  - A filter's read ([`FtsReader::ensure_doc_map`]) keeps the map and is charged for it.
        //  - The blocking read ([`FtsReader::doc_map`]) on a second reader does the same.
        //  - Dropping both readers gives the charge back.
        // This test pins that a map kept regardless still counts in the budget.
        let (blob, json, order, _) = dense_translation();
        let map_len = (order.len() * DOC_MAP_ENTRY_BYTES) as u64;
        let budget = roomy(Some(0));
        let r = open_charged(blob.clone(), &json, budget).await;
        r.ensure_doc_map().await.expect("ensure_doc_map");
        assert_eq!(budget.charged(), map_len);
        let blocking = open_charged(blob, &json, budget).await;
        let blocking = spawn_blocking(move || blocking.doc_map().map(|_| blocking))
            .await
            .expect("join")
            .expect("doc_map");
        assert_eq!(budget.charged(), 2 * map_len);
        drop((r, blocking));
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_map_on_a_file_held_whole_in_memory_is_not_charged() {
        // Setup: a reordered blob opened from memory, and a dense translation that would keep the
        // map.
        //  - The map is a slice of the blob, taken at open, so nothing is charged or kept after the
        //    translation.
        // This test pins that a map that is a slice of an in-memory blob costs no heap budget.
        let (blob, json, order, hits) = dense_translation();
        let budget = roomy(None);
        let r = FtsReader::open_with(blob, &json, OpenOptions { verify_crc: true })
            .expect("open")
            .with_heap_budget(budget);
        assert!(r.doc_map_is_read(), "taken at open");
        let want: Vec<(RowId, f32)> = hits
            .iter()
            .map(|&(id, score)| (RowId::new(order[id.get() as usize]), score))
            .collect();
        assert_eq!(r.hits_to_rows_async(hits).await.expect("translate"), want);
        assert_eq!(budget.charged(), 0);
        assert_eq!(r.kept_pages.page_count(), 0);
    }

    #[tokio::test]
    async fn a_lazy_open_keeps_no_map_even_with_its_bytes_at_hand() {
        // Setup: a lazy source whose bytes are all local, as after a file's text section was
        // fetched in bulk.
        //  - The open keeps no map from them; the first query that needs the map asks the budget.
        // This test pins that reopening many cached files can't keep their maps past the budget.
        let (blob, json, _, hits) = dense_translation();
        let budget = roomy(Some(0));
        let src = Source::Lazy(Arc::new(BytesLazyByteSource::new(blob)));
        let r = FtsReader::open_with_source(src, &json, OpenOptions { verify_crc: true })
            .expect("open")
            .with_heap_budget(budget);
        assert!(!r.doc_map_is_read(), "nothing kept at open");
        r.hits_to_rows_async(hits).await.expect("translate");
        assert!(!r.doc_map_is_read(), "the budget had no room");
        assert_eq!(budget.charged(), 0);
    }

    #[tokio::test]
    async fn a_sparse_translation_sends_its_page_reads_at_once() {
        // Setup: a map of 10,000 documents and one hit every [`PAGES_APART`] pages, too far apart
        // to join: 40 reads, more than one read cap, under the sparse share.
        //  - Held at the source, the translation has started every read before any comes back.
        // This test pins that a sparse translation takes one round trip, not one per read cap.
        const N_DOCS: u32 = 10_000;
        /// Reads the hits fall in, each its own page.
        const READS: u32 = 40;
        let docs: Vec<String> = (0..N_DOCS).map(|row| format!("alpha r{row}")).collect();
        let refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let order: Vec<u32> = (0..N_DOCS).rev().collect();
        let (blob, json) = build_standard_blob_with(&refs, Some(&order));
        let ids: Vec<FtsDocId> = (0..READS)
            .map(|i| FtsDocId::new(i * PAGES_APART * IDS_PER_PAGE))
            .collect();
        let read_bytes = READS as usize * DOC_MAP_PAGE_BYTES;
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: true },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(roomy(None));
        assert!(
            read_bytes > DOC_MAP_READ_MAX_PAGES * DOC_MAP_PAGE_BYTES,
            "more than one cap"
        );
        assert!(
            read_bytes * DOC_MAP_SPARSE_SHARE <= r.doc_map_len(),
            "sparse"
        );
        let opened = src.started.load(Ordering::Relaxed);
        let held = src.gate.write().await;
        let mut query = Box::pin(r.ids_to_rows_async(ids));
        assert!(poll!(query.as_mut()).is_pending());
        assert_eq!(
            src.started.load(Ordering::Relaxed) - opened,
            READS as usize,
            "all reads in flight"
        );
        drop(held);
        let rows = query.await.expect("translate");
        assert_eq!(rows.len(), READS as usize);
    }

    #[tokio::test]
    async fn a_query_cancelled_mid_read_gives_its_reservation_back() {
        // Setup: a dense translation held while it reads the whole map, then dropped.
        //  - While held it has reserved the map's bytes.
        //  - Dropping it gives them back and leaves the map unread for the next query.
        // This test pins that a cancelled query can't leave the budget charged for a map nobody
        // holds.
        let (blob, json, _, hits) = dense_translation();
        let budget = roomy(None);
        let src = Arc::new(GatedSource::new(blob));
        let r = FtsReader::open_lazy(
            Arc::clone(&src) as Arc<dyn LazyByteSource>,
            &json,
            OpenOptions { verify_crc: true },
        )
        .await
        .expect("open_lazy")
        .with_heap_budget(budget);
        let _held = src.gate.write().await;
        let mut query = Box::pin(r.hits_to_rows_async(hits));
        assert!(poll!(query.as_mut()).is_pending());
        assert_eq!(
            budget.charged(),
            r.doc_map_len() as u64,
            "reserved mid-read"
        );
        drop(query);
        assert_eq!(budget.charged(), 0);
        assert!(!r.doc_map_is_read());
    }
}
