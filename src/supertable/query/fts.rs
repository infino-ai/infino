// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 fan-out on [`Supertable`](super::super::Supertable).
//!
//! ## Public API
//!
//! The sync, user-facing entry points live on
//! [`Supertable`](super::super::Supertable):
//!
//! ```ignore
//! // Bare call: `_id` + `score` only — no scalar decode.
//! let ids: Vec<RecordBatch> =
//!     table.bm25_search("title", "rust async", 10, BoolMode::Or, None)?;
//!
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> =
//!     table.bm25_search("title", "rust async", 10, BoolMode::Or, Some(&["_id", "title", "score"]))?;
//!
//! // Unranked candidate sets (Arrow rows, score == 0.0).
//! let any = table.token_match("title", "rust async", BoolMode::Or, None)?;
//! let exact = table.exact_match("title", "rust async", None)?;
//! ```
//!
//! Internally these drive the async kernel on the snapshot-pinned
//! [`SupertableReader`], whose `bm25_search` (rows) / `bm25_hits`
//! ([`SuperfileHit`], superfile-local) / `bm25_search_prefix` methods are
//! the engine-facing surface. Ranked results are sorted by score
//! *descending* — higher BM25 score is more relevant.
//!
//! ## Strategy
//!
//! Internally pins a snapshot reader and drives the async
//! kernel to completion via the sync→async bridge. The reader
//! holds a pinned `Arc<ManifestSnapshot>`; for each visible superfile we:
//!
//!   1. Fetch the superfile's `SuperfileReader` from the store.
//!   2. Delegate to `SuperfileReader::bm25_search` /
//!      `bm25_search_prefix` (already implemented at the superfile
//!      layer; per-superfile top-k with BlockMaxWAND skip).
//!   3. Tag each `(local_doc_id, score)` with the superfile URI.
//!   4. Concatenate across superfiles and global-top-k by score.
//!
//! Rayon fan-out runs on `options.reader_pool`. For an N-superfile
//! supertable we issue N parallel per-superfile searches; the pool
//! caps concurrency at the configured reader thread count.
//!
//! ## Score comparability across superfiles
//!
//! BM25's IDF is computed from per-superfile `n_docs` and `df`,
//! so a rare term in a small superfile can score higher than the
//! same term in a larger superfile. This is the classical sharded-
//! BM25 problem:
//! treating per-superfile scores as comparable is a documented
//! approximation, accepted in v1 because (a) global IDF would
//! require either a manifest-wide df table or a two-pass query
//! (df gather + score), both with non-trivial memory/latency
//! cost; (b) for k ≥ 10 and reasonably balanced superfiles the top-k
//! *set* converges to the global answer even if score *order*
//! within the set wiggles. Oracle tests assert set membership at
//! `k = 10` against a single-superfile ground truth.
//!
//! ManifestSnapshot-level skip pruning is wired in: each call computes a
//! per-superfile keep/prune mask from the FTS bloom (exact-term
//! mode) or the lex term range (prefix mode) before issuing
//! per-superfile work, so pruned superfiles never trigger a
//! `SuperfileReaderCache::reader` call. Vector + SQL skip remain
//! deferred (see those modules' headers).

use std::{
    borrow::Cow,
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
    slice,
    sync::{
        Arc, Mutex,
        atomic::{self, AtomicU32},
    },
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, LargeStringArray};
use roaring::RoaringBitmap;
use tracing::debug;
use uuid::Uuid;

pub use crate::superfile::fts::reader::BoolMode;
use crate::{
    InfinoError,
    superfile::{
        SuperfileReader,
        error::{FtsError, ReadError},
        fts::{
            reader::ClauseLists,
            tokenize::{AsciiLowerTokenizer, Tokenizer},
        },
    },
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::SuperfileEntry,
        query::{
            SuperfileHit, dispatch,
            exec::common::{resolve_hits_named, take_rows_byte_source},
            prune::{PruneLeaf, select_superfiles},
        },
        reader_cache::disk::ForegroundQueryGuard,
        tombstones::SidecarCache,
    },
};

/// An unranked query's match set: the terms and exact phrases every
/// (`And`) or any (`Or`) of which a doc must contain. Produced by
/// `parse_and_prune` from the clause model — the must side when any
/// must exists (shoulds have no scores to raise unranked), the bare
/// side under the default operator otherwise.
struct UnrankedMatchSet {
    terms: Vec<String>,
    phrases: Vec<Vec<String>>,
    mode: BoolMode,
}

impl Default for UnrankedMatchSet {
    fn default() -> Self {
        Self {
            terms: Vec::new(),
            phrases: Vec::new(),
            mode: BoolMode::Or,
        }
    }
}

impl UnrankedMatchSet {
    fn has_phrases(&self) -> bool {
        !self.phrases.is_empty()
    }
}

/// An unranked query's negated atoms (docs containing any are
/// excluded).
#[derive(Default)]
struct UnrankedNegatives {
    terms: Vec<String>,
    phrases: Vec<Vec<String>>,
}

impl UnrankedNegatives {
    fn is_empty(&self) -> bool {
        self.terms.is_empty() && self.phrases.is_empty()
    }
}

/// Rejection message for a query with negated terms but no positive
/// anchor (e.g. `-foo`). Shared by the scored and unranked FTS paths so
/// both reject the case identically.
const NEGATION_ONLY_QUERY_MSG: &str = "only negated terms; at least one positive term is required";

/// Cross-segment top-k score sharing for the BM25 fan-out.
///
/// Every segment kernel runs an independent top-k; without
/// coordination, segment N knows nothing about the k hits segments
/// 1..N-1 already produced, so it scores blocks the global result can
/// never use. This shares the running **global kth-best score** as a
/// floor: each kernel reads it at start and seeds its pruning
/// structures (BMW block skips, the MaxScore essential boundary, AND
/// block-max bars) from it; each finishing kernel merges its surviving
/// scores back, monotonically raising the floor for the segments still
/// running.
///
/// Correctness: the floor only ever prunes docs scoring **strictly
/// below** the published kth-best (kernels apply it via
/// `floor.next_down()` comparisons), and the published floor is always
/// ≤ the final global kth-best, so every doc that could appear in the
/// merged top-k survives in some segment's result — the merged output
/// is identical to an uncoordinated run, including score ties. Only
/// the amount of *skipped work* depends on segment completion order.
struct SharedTopK {
    k: usize,
    /// Min-heap (via `Reverse`) of the best `k` scores seen so far.
    heap: Mutex<BinaryHeap<Reverse<OrdScore>>>,
    /// f32 bits of the current floor; `NEG_INFINITY` until `k` scores
    /// have been seen. Monotonically non-decreasing.
    floor_bits: AtomicU32,
}

/// Total-order f32 wrapper for the [`SharedTopK`] heap (BM25 scores
/// are finite, but `f32` still needs an `Ord` shim).
#[derive(PartialEq)]
struct OrdScore(f32);
impl Eq for OrdScore {}
impl PartialOrd for OrdScore {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdScore {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl SharedTopK {
    fn new(k: usize) -> Arc<Self> {
        Arc::new(Self {
            k,
            heap: Mutex::new(BinaryHeap::new()),
            floor_bits: AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        })
    }

    /// The current global floor — `NEG_INFINITY` until k scores merged.
    fn floor(&self) -> f32 {
        f32::from_bits(self.floor_bits.load(atomic::Ordering::Acquire))
    }

    /// Merge one finished segment's (tombstone-surviving) scores and
    /// publish the new kth-best as the floor once k scores are known.
    fn merge(&self, scores: impl IntoIterator<Item = f32>) {
        let mut heap = self.heap.lock().expect("SharedTopK mutex poisoned");
        for s in scores {
            if heap.len() < self.k {
                heap.push(Reverse(OrdScore(s)));
            } else if let Some(Reverse(OrdScore(min))) = heap.peek()
                && s > *min
            {
                heap.pop();
                heap.push(Reverse(OrdScore(s)));
            }
        }
        if heap.len() == self.k
            && let Some(Reverse(OrdScore(min))) = heap.peek()
        {
            // The heap min only rises, so a plain store stays monotone
            // under the lock.
            self.floor_bits
                .store(min.to_bits(), atomic::Ordering::Release);
        }
    }
}

impl SupertableReader {
    /// Single-column BM25 search across the pinned manifest's
    /// superfiles. Returns up to `k` highest-scoring hits, sorted
    /// descending by score.
    ///
    /// `query` is tokenized by the v1 [`AsciiLowerTokenizer`] —
    /// the same tokenizer used at index time. Returns
    /// [`QueryError::Store`] if any superfile is unreachable, or
    /// [`QueryError::Parquet`] if a superfile's bytes can't be
    /// queried (column missing from the superfile's FTS index, etc.).
    ///
    /// Empty supertable (no superfiles) returns an empty `Vec`
    /// without consulting the store.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`SupertableReader::bm25_search`], which drives this via the
    /// sync→async bridge.
    ///
    /// [`AsciiLowerTokenizer`]: crate::superfile::fts::tokenize::AsciiLowerTokenizer
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, mode = ?mode))
    )]
    pub(crate) async fn bm25_search_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let pool_threads = manifest.options.reader_pool.current_num_threads();
        let column_owned = column.to_owned();

        // Parse the query once here, not per superfile, resolving the
        // bare tokens' polarity from the default operator (`And` ⇒
        // must, `Or` ⇒ should). The fan-out closures below need owned
        // ('static) data for tokio::spawn, so this is the one place
        // the tokens are copied — the prune and every per-superfile
        // search reuse them.
        let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
        let musts: Vec<String> = clauses.musts.into_iter().map(Cow::into_owned).collect();
        let shoulds: Vec<String> = clauses.shoulds.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = clauses.negatives.into_iter().map(Cow::into_owned).collect();
        let own_phrases = |phrases: Vec<Vec<Cow<'_, str>>>| -> Vec<Vec<String>> {
            phrases
                .into_iter()
                .map(|p| p.into_iter().map(Cow::into_owned).collect())
                .collect()
        };
        let must_phrases = own_phrases(clauses.must_phrases);
        let should_phrases = own_phrases(clauses.should_phrases);
        let negative_phrases = own_phrases(clauses.negative_phrases);
        let has_musts = !musts.is_empty() || !must_phrases.is_empty();
        let has_phrases =
            !must_phrases.is_empty() || !should_phrases.is_empty() || !negative_phrases.is_empty();

        if !has_musts && shoulds.is_empty() && should_phrases.is_empty() {
            // No scorable clause at all. Empty / punctuation-only
            // queries match nothing (not an error); negation-only
            // (e.g. `-foo`) has no anchor to rank — reject up front so
            // the per-superfile kernel never has to, and so the
            // unranked count / token_match path surfaces the identical
            // error (see `parse_and_prune`).
            if negatives.is_empty() && negative_phrases.is_empty() {
                return Ok(Vec::new());
            }
            return Err(QueryError::InvalidQuery(NEGATION_ONLY_QUERY_MSG.to_owned()));
        }

        // Pick the superfiles to search, via the shared two-tier bloom
        // prune. Musts prune hardest: every match contains all of
        // them — a phrase's member terms included, since a phrase
        // match requires every member present — so a superfile
        // lacking any is skipped regardless of `mode`. A pure should
        // query prunes as the flat term list did (phrase members join
        // the union: a doc matching the phrase contains each member).
        // Negated atoms never prune, and shoulds never prune once a
        // must exists, since they only affect scores.
        let (mut prune_terms, prune_mode) = if !has_musts {
            (shoulds.clone(), mode)
        } else {
            (musts.clone(), BoolMode::And)
        };
        match has_musts {
            true => {
                for p in &must_phrases {
                    prune_terms.extend(p.iter().cloned());
                }
            }
            false => {
                for p in &should_phrases {
                    prune_terms.extend(p.iter().cloned());
                }
            }
        }
        let prune_leaf = PruneLeaf::TermPresence {
            column: column_owned.clone(),
            terms: prune_terms,
            mode: prune_mode,
        };
        let kept = select_superfiles(manifest.as_ref(), slice::from_ref(&prune_leaf)).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        // Build the work-unit list. When the reader pool has more
        // threads than there are kept superfiles AND we're on the
        // multi-term OR hot path, slice each superfile into doc_id
        // sub-ranges so the fan-out can saturate every pool thread.
        // Single-term OR, AND, and any query with a must or negated
        // clause stay on the un-ranged call.
        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        // Phrase-bearing queries stay per-superfile: the ranged
        // kernel is the pure term-union fast path.
        let fanout = match has_phrases {
            true => FanOut::PerSuperfile,
            false => fanout_for(musts.len(), shoulds.len(), !negatives.is_empty()),
        };
        let work_units = build_work_units(&kept_refs, fanout, pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, (Option<(u32, u32)>, Uuid))> = work_units
            .into_iter()
            .map(|u| {
                let suid = u.entry.superfile_id;
                (u.entry, (u.range, suid))
            })
            .collect();

        let must_arc: Arc<Vec<String>> = Arc::new(musts);
        let should_arc: Arc<Vec<String>> = Arc::new(shoulds);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives);
        let must_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(must_phrases);
        let should_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(should_phrases);
        let neg_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(negative_phrases);
        let column_arc = Arc::new(column_owned);

        // Cross-segment threshold sharing: each unit reads the global
        // kth-best floor before searching and merges its surviving
        // scores back after — late units skip every block that can't
        // beat what earlier units already found. Tombstoned hits are
        // excluded from the merge so deleted rows never raise the bar.
        let shared = SharedTopK::new(k);
        let tombstones = self.tombstone_cache.clone();
        let now = Instant::now();

        // One shared fan-out (`query::dispatch::fanout`) — the same
        // orchestrator the vector path uses. It warms the tombstone
        // sidecars in one batch, opens each superfile reader and runs the
        // kernel under `tokio::spawn` so cold GETs overlap, then tags +
        // tombstone-filters each unit's hits. The per-unit `params` is
        // the optional doc-id sub-range (`None` searches the whole
        // superfile) plus the superfile id for the tombstone-aware merge.
        let kernel = move |r: Arc<SuperfileReader>, (range, suid): (Option<(u32, u32)>, Uuid)| {
            let column_arc = Arc::clone(&column_arc);
            let must_arc = Arc::clone(&must_arc);
            let should_arc = Arc::clone(&should_arc);
            let neg_arc = Arc::clone(&neg_arc);
            let must_ph_arc = Arc::clone(&must_ph_arc);
            let should_ph_arc = Arc::clone(&should_ph_arc);
            let neg_ph_arc = Arc::clone(&neg_ph_arc);
            let shared = Arc::clone(&shared);
            let tombstones = tombstones.clone();
            async move {
                // Share the global kth-best floor with every superfile —
                // single-term queries included — so each prunes its scored
                // scan against the running top-k instead of returning a full
                // local top-k for the merge to re-sort. Without this the
                // fan-out churns ~(superfiles × k) candidates through the
                // merge heap at large k, which dominates high-k latency.
                // Ties stay correct: the floor prunes only scores strictly
                // below the published kth-best (kernels compare via
                // `floor.next_down()`), so the merged top-k — score ties
                // included — matches an uncoordinated run; only the amount
                // of skipped work depends on segment completion order.
                let floor = shared.floor();
                let hits = match range {
                    // Ranged units exist only for pure multi-should
                    // queries (`fanout_for` never slices when a must
                    // or negated clause exists).
                    Some((start, end)) => {
                        let should_refs: Vec<&str> =
                            should_arc.iter().map(|s| s.as_str()).collect();
                        r.bm25_search_or_range_pretokenized_with_floor(
                            &column_arc,
                            &should_refs,
                            k,
                            start,
                            end,
                            floor,
                        )
                        .await
                        .map_err(fts_read_error)?
                    }
                    None => {
                        let must_refs: Vec<&str> = must_arc.iter().map(|s| s.as_str()).collect();
                        let should_refs: Vec<&str> =
                            should_arc.iter().map(|s| s.as_str()).collect();
                        let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                        r.bm25_search_clauses(
                            &column_arc,
                            ClauseLists {
                                musts: &must_refs,
                                shoulds: &should_refs,
                                negatives: &neg_refs,
                                must_phrases: &must_ph_arc,
                                should_phrases: &should_ph_arc,
                                negative_phrases: &neg_ph_arc,
                            },
                            k,
                            floor,
                        )
                        .await
                        .map_err(fts_read_error)?
                    }
                };
                // Raise the global floor with this unit's surviving
                // scores. Sidecars were prefetched by the dispatcher,
                // so the bitmap lookup is an in-memory hit; on a cache
                // miss/error we simply don't merge (a lower floor is
                // always safe).
                match tombstones.as_ref().map(|c| c.bitmap_for(suid, now)) {
                    Some(Ok(bitmap)) if !bitmap.is_empty() => shared.merge(
                        hits.iter()
                            .filter(|(d, _)| !bitmap.contains(*d))
                            .map(|(_, s)| *s),
                    ),
                    Some(Err(_)) => {}
                    _ => shared.merge(hits.iter().map(|(_, s)| *s)),
                }
                Ok(hits)
            }
        };
        let per_unit = dispatch::fanout_local_hits(self, units, kernel).await?;
        let hits = select_top_k_stable(self, per_unit, k).await?;
        Ok(hits)
    }

    /// Prefix-expanded BM25 search across the pinned manifest's
    /// superfiles. The prefix is ASCII-lowercased before expansion
    /// (matching the v1 tokenizer) and expanded per-superfile to the
    /// concrete term list before `BoolMode::Or` BM25 scoring.
    ///
    /// Returns up to `k` highest-scoring hits, sorted descending
    /// by score.
    ///
    /// Empty supertable (no superfiles) and `k == 0` short-circuit
    /// to an empty `Vec`.
    ///
    /// `pub(crate)` async kernel — the public surface is the sync
    /// [`SupertableReader::bm25_search_prefix`].
    pub(crate) async fn bm25_search_prefix_async(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let manifest = self.manifest();
        let pool_threads = manifest.options.reader_pool.current_num_threads();
        let column_owned = column.to_owned();
        let prefix_owned = prefix.to_owned();

        // ManifestSnapshot-level term-range skip uses the same
        // lowercased prefix bytes the v1 tokenizer +
        // FST-expansion path use, so the skip's
        // lex-range overlap test exactly matches the
        // tokenizer's interpretation of the prefix.
        let prefix_lower = prefix_owned.to_ascii_lowercase();

        // Superfile selection via the shared two-tier prune — the
        // single-`Prefix`-leaf case (part-level term-range skip →
        // lazy-load surviving parts → per-superfile term-range skip).
        let kept = select_superfiles(
            manifest.as_ref(),
            &[PruneLeaf::Prefix {
                column: column_owned.clone(),
                prefix: prefix_lower.as_bytes().to_vec(),
            }],
        )
        .await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        // Prefix expansion is always multi-term OR with no negation, so
        // it is directly sub-range eligible.
        let work_units = build_work_units(&kept_refs, FanOut::SubRanges, pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, Option<(u32, u32)>)> =
            work_units.into_iter().map(|u| (u.entry, u.range)).collect();

        let column_arc = Arc::new(column_owned);
        let prefix_arc = Arc::new(prefix_owned);

        // Shared fan-out — see `bm25_search` for the rationale; the
        // kernel differs only in calling the prefix search variants.
        let kernel = move |r: Arc<SuperfileReader>, range: Option<(u32, u32)>| {
            let column_arc = Arc::clone(&column_arc);
            let prefix_arc = Arc::clone(&prefix_arc);
            async move {
                match range {
                    Some((start, end)) => r
                        .bm25_search_prefix_range(&column_arc, &prefix_arc, k, start, end)
                        .await
                        .map_err(fts_read_error),
                    None => r
                        .bm25_search_prefix(&column_arc, &prefix_arc, k)
                        .await
                        .map_err(fts_read_error),
                }
            }
        };
        let per_unit = dispatch::fanout_local_hits(self, units, kernel).await?;
        let hits = select_top_k_stable(self, per_unit, k).await?;
        Ok(hits)
    }

    /// Parse `query` into positive and negated tokens, then select the
    /// superfiles to scan. Pruning keys on the **positives only** — a
    /// negated term must never drop a superfile: a superfile lacking it
    /// excludes nothing, and under `And` keying on it would wrongly prune
    /// every superfile that doesn't carry it. This mirrors the BM25
    /// search path so the unranked `token_match` / `count` surfaces honor
    /// negation the same way scored search does.
    ///
    /// Returns `(positives, negatives, kept)`. A query with no tokens at
    /// all yields an empty `kept`, so the caller returns the empty result
    /// (`[]` / count `0`). A negation-only query (negated terms but no
    /// positive, e.g. `-foo`) is rejected with [`QueryError::InvalidQuery`],
    /// the same as the scored search path — there is no positive anchor to
    /// match against.
    /// Parse `query` into clauses, resolve the unranked **match set**
    /// terms, and bloom-prune the superfile list.
    ///
    /// Unranked matching has no scores for a should clause to raise,
    /// so the match set is the musts' intersection whenever any must
    /// exists (`+a b` matches exactly the docs containing `a`; the
    /// bare `b` is scoring-only and contributes nothing here) —
    /// keeping `token_match` / `count` consistent with which docs the
    /// scored search returns. With no musts, the bare terms match
    /// under `mode` exactly as before.
    ///
    /// Returns `(match_set, negatives, kept)`.
    async fn parse_and_prune(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<
        (
            UnrankedMatchSet,
            UnrankedNegatives,
            Vec<Arc<SuperfileEntry>>,
        ),
        QueryError,
    > {
        let clauses = AsciiLowerTokenizer.parse(query).into_clauses(mode);
        let musts: Vec<String> = clauses.musts.into_iter().map(Cow::into_owned).collect();
        let shoulds: Vec<String> = clauses.shoulds.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = clauses.negatives.into_iter().map(Cow::into_owned).collect();
        let own_phrases = |phrases: Vec<Vec<Cow<'_, str>>>| -> Vec<Vec<String>> {
            phrases
                .into_iter()
                .map(|p| p.into_iter().map(Cow::into_owned).collect())
                .collect()
        };
        let must_phrases = own_phrases(clauses.must_phrases);
        let should_phrases = own_phrases(clauses.should_phrases);
        let negative_phrases = own_phrases(clauses.negative_phrases);
        let negs = UnrankedNegatives {
            terms: negatives,
            phrases: negative_phrases,
        };
        let has_musts = !musts.is_empty() || !must_phrases.is_empty();
        if !has_musts && shoulds.is_empty() && should_phrases.is_empty() {
            if negs.terms.is_empty() && negs.phrases.is_empty() {
                // No tokens at all (empty/whitespace query) — nothing to
                // match, not an error.
                return Ok((UnrankedMatchSet::default(), negs, Vec::new()));
            }
            // Negation-only (e.g. `-foo`): reject, matching the scored
            // search path, which has no positive anchor to rank or match.
            return Err(QueryError::InvalidQuery(NEGATION_ONLY_QUERY_MSG.to_owned()));
        }
        // Unranked matching has no scores for a should to raise, so
        // the match set is the must side whenever any must exists.
        let match_set = match has_musts {
            true => UnrankedMatchSet {
                terms: musts,
                phrases: must_phrases,
                mode: BoolMode::And,
            },
            false => UnrankedMatchSet {
                terms: shoulds,
                phrases: should_phrases,
                mode,
            },
        };
        // Prune on the match set's terms plus its phrases' members —
        // a phrase match requires every member present.
        let mut prune_terms = match_set.terms.clone();
        for p in &match_set.phrases {
            prune_terms.extend(p.iter().cloned());
        }
        let prune_leaf = PruneLeaf::TermPresence {
            column: column.to_owned(),
            terms: prune_terms,
            mode: match_set.mode,
        };
        let kept =
            select_superfiles(self.manifest().as_ref(), slice::from_ref(&prune_leaf)).await?;
        Ok((match_set, negs, kept))
    }

    /// Unranked token match across the pinned snapshot. Returns
    /// every row matching `query`'s tokens under `mode` (`Or` = any
    /// token, `And` = every token) as [`SuperfileHit`]s — **no scoring**
    /// (`score` is left `0.0`; these results are unordered). Superfile
    /// skip uses the same term-bloom prune as BM25.
    ///
    /// With a `+must` clause, the match set is the musts' intersection
    /// and bare (should) tokens are ignored — they only affect scores,
    /// and there are none here (see [`Self::parse_and_prune`]).
    ///
    /// `pub(crate)` async kernel; the public surface is the sync
    /// [`SupertableReader::token_match`].
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, mode = ?mode))
    )]
    pub(crate) async fn token_match_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let (match_set, negatives, kept) = self.parse_and_prune(column, query, mode).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let match_mode = match_set.mode;
        let has_negatives = !negatives.is_empty();
        let phrase_involved = match_set.has_phrases() || !negatives.phrases.is_empty();
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(match_set.terms);
        let phrase_arc: Arc<Vec<Vec<String>>> = Arc::new(match_set.phrases);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives.terms);
        let neg_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(negatives.phrases);
        let kernel = move |r: Arc<SuperfileReader>, _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let term_arc = Arc::clone(&term_arc);
            let phrase_arc = Arc::clone(&phrase_arc);
            let neg_arc = Arc::clone(&neg_arc);
            let neg_ph_arc = Arc::clone(&neg_ph_arc);
            async move {
                let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                // Any phrase atom (match or negated) takes the
                // phrase-aware walk; plain-token queries keep the
                // optimized token_match path unchanged.
                let docs = match phrase_involved {
                    true => r
                        .atoms_match_ids(&column_arc, &refs, &phrase_arc, match_mode)
                        .await
                        .map_err(fts_read_error)?,
                    false => r
                        .token_match(&column_arc, &refs, match_mode)
                        .await
                        .map_err(fts_read_error)?,
                };
                // Drop any positive match that also carries a negated
                // atom (union of the negatives). The df / count fast
                // paths can't express exclusion, so negation forces a
                // materialized walk over both sets.
                let docs = if has_negatives {
                    let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                    let excluded: RoaringBitmap = match neg_ph_arc.is_empty() {
                        true => r
                            .token_match(&column_arc, &neg_refs, BoolMode::Or)
                            .await
                            .map_err(fts_read_error)?,
                        false => r
                            .atoms_match_ids(&column_arc, &neg_refs, &neg_ph_arc, BoolMode::Or)
                            .await
                            .map_err(fts_read_error)?,
                    }
                    .into_iter()
                    .collect();
                    docs.into_iter()
                        .filter(|d| !excluded.contains(*d))
                        .collect::<Vec<_>>()
                } else {
                    docs
                };
                Ok(docs.into_iter().map(|d| (d, 0.0f32)).collect::<Vec<_>>())
            }
        };
        let per_unit = dispatch::fanout_local_hits(self, units, kernel).await?;
        // Exact pre-size: `Flatten`'s size_hint is opaque, and growth
        // reallocations copy the whole hit vec repeatedly at 1M hits.
        let total: usize = per_unit.iter().map(Vec::len).sum();
        let mut hits: Vec<SuperfileHit> = Vec::with_capacity(total);
        for unit in per_unit {
            hits.extend(unit);
        }
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(hits)
    }

    /// Count documents whose `column` matches `query`'s tokens under
    /// `mode` (`Or` = any token, `And` = every token), over this reader's
    /// pinned snapshot — **count only, no scoring and no row
    /// materialization**.
    ///
    /// With a `+must` clause, the count is the musts' intersection
    /// cardinality — bare (should) tokens affect only scores, so they
    /// never change which docs are counted (see
    /// [`Self::parse_and_prune`]). `count("+climate policy")` is the
    /// number of docs containing `climate`.
    ///
    /// Fast path: a single-token query against a superfile with no
    /// tombstones resolves from the term dictionary's stored document
    /// frequency ([`SuperfileReader::term_df`]) — O(1) per superfile, no
    /// posting decode. A multi-token query, or a superfile with deletes,
    /// falls back to materializing the matching local doc ids and
    /// counting those not tombstoned. Tombstoned (deleted) rows are
    /// always excluded so the count matches what a search would return.
    pub(crate) async fn token_match_count_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<u64, QueryError> {
        let (match_set, negatives, kept) = self.parse_and_prune(column, query, mode).await?;
        if kept.is_empty() {
            return Ok(0);
        }

        let match_mode = match_set.mode;
        let single_term = match_set.terms.len() == 1 && !match_set.has_phrases();
        let has_negatives = !negatives.is_empty();
        let phrase_involved = match_set.has_phrases() || !negatives.phrases.is_empty();
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(match_set.terms);
        let phrase_arc: Arc<Vec<Vec<String>>> = Arc::new(match_set.phrases);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives.terms);
        let neg_ph_arc: Arc<Vec<Vec<String>>> = Arc::new(negatives.phrases);
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();

        // Shared fan-out (`dispatch::fanout_with`): warms tombstones,
        // spawns + opens each superfile concurrently, and short-circuits
        // on the first error. The per-superfile body returns this
        // superfile's match count; the totals are summed.
        let per_superfile = dispatch::fanout_with(
            self,
            units,
            true,
            true,
            move |r, entry, tombstone_cache, now, _params: ()| {
                let column_arc = Arc::clone(&column_arc);
                let term_arc = Arc::clone(&term_arc);
                let phrase_arc = Arc::clone(&phrase_arc);
                let neg_arc = Arc::clone(&neg_arc);
                let neg_ph_arc = Arc::clone(&neg_ph_arc);
                async move {
                    // Tombstone bitmap for this superfile (None = no deletes).
                    let tomb = match tombstone_cache.as_ref() {
                        Some(c) => {
                            let b = c
                                .bitmap_for(entry.superfile_id, now)
                                .map_err(|e| QueryError::Store(format!("tombstone cache: {e}")))?;
                            if b.is_empty() { None } else { Some(b) }
                        }
                        None => None,
                    };
                    let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                    // Negated terms or deletes both force materialization:
                    // the df read and the bare match count can't subtract
                    // excluded or tombstoned docs. Materialize the positive
                    // matches, then drop any doc carrying a negated term
                    // (union of the negatives) or a tombstone.
                    if has_negatives || tomb.is_some() {
                        let docs = match phrase_involved {
                            true => r
                                .atoms_match_ids(&column_arc, &refs, &phrase_arc, match_mode)
                                .await
                                .map_err(fts_read_error)?,
                            false => r
                                .token_match(&column_arc, &refs, match_mode)
                                .await
                                .map_err(fts_read_error)?,
                        };
                        let excluded: RoaringBitmap = if has_negatives {
                            let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                            match neg_ph_arc.is_empty() {
                                true => r
                                    .token_match(&column_arc, &neg_refs, BoolMode::Or)
                                    .await
                                    .map_err(fts_read_error)?,
                                false => r
                                    .atoms_match_ids(
                                        &column_arc,
                                        &neg_refs,
                                        &neg_ph_arc,
                                        BoolMode::Or,
                                    )
                                    .await
                                    .map_err(fts_read_error)?,
                            }
                            .into_iter()
                            .collect()
                        } else {
                            RoaringBitmap::new()
                        };
                        let n = docs
                            .iter()
                            .filter(|d| {
                                !excluded.contains(**d)
                                    && tomb.as_ref().is_none_or(|b| !b.contains(**d))
                            })
                            .count() as u64;
                        return Ok::<u64, QueryError>(n);
                    }
                    // No negatives and no deletes (the common case): count
                    // without materializing ids — a single token resolves
                    // O(1) from the stored df, multi-token tallies the
                    // match walk through the counting sink.
                    let n = if single_term {
                        r.term_df(&column_arc, &term_arc[0])
                            .await
                            .map_err(fts_read_error)?
                    } else if phrase_involved {
                        r.atoms_match_count(&column_arc, &refs, &phrase_arc, match_mode)
                            .await
                            .map_err(fts_read_error)?
                    } else {
                        r.token_match_count(&column_arc, &refs, match_mode)
                            .await
                            .map_err(fts_read_error)?
                    };
                    Ok(n)
                }
            },
        )
        .await?;
        Ok(per_superfile.into_iter().sum())
    }

    /// Unranked two-pass exact match of the **raw string** `value`
    /// against `column` across the pinned snapshot. Returns the rows
    /// whose stored value equals `value` exactly as [`SuperfileHit`]s —
    /// **no scoring**. See [`crate::superfile::SuperfileReader::exact_match`]
    /// for the per-superfile two-pass (token-AND prune + raw verify).
    ///
    /// `pub(crate)` async kernel; the public surface is the sync
    /// [`SupertableReader::exact_match`].
    pub(crate) async fn exact_match_async(
        &self,
        column: &str,
        value: &str,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let manifest = self.manifest();
        let term_strings: Vec<String> = AsciiLowerTokenizer.tokenize(value).collect();
        // Tokens prune superfiles via the term bloom (AND); a token-less
        // value (e.g. punctuation only) can't prune, so keep all.
        let leaves = if term_strings.is_empty() {
            Vec::new()
        } else {
            vec![PruneLeaf::TermPresence {
                column: column.to_owned(),
                terms: term_strings.clone(),
                mode: BoolMode::And,
            }]
        };
        let kept = select_superfiles(manifest.as_ref(), &leaves).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_arc = Arc::new(column.to_owned());
        let value_arc = Arc::new(value.to_owned());
        let tokens_arc = Arc::new(term_strings);
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let value_arc = Arc::clone(&value_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            async move {
                let candidates: Vec<u32> = if tokens_arc.is_empty() {
                    (0..r.n_docs() as u32).collect()
                } else {
                    let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                    r.token_match(&column_arc, &refs, BoolMode::And)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string()))?
                };
                if candidates.is_empty() {
                    return Ok(Vec::new());
                }
                let batch = if r.can_take_by_local_doc_ids() {
                    r.take_by_local_doc_ids(&candidates, &[column_arc.as_str()])
                        .map_err(|e| QueryError::Parquet(e.to_string()))?
                } else {
                    take_rows_byte_source(&r, &candidates, &[column_arc.as_str()])
                        .await
                        .map_err(|e| QueryError::Execute(e.to_string()))?
                };
                let values = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| {
                        QueryError::Execute(format!(
                            "exact_match column '{}' is not LargeUtf8",
                            column_arc
                        ))
                    })?;
                let mut hits: Vec<SuperfileHit> = candidates
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| {
                        !values.is_null(*index) && values.value(*index) == value_arc.as_str()
                    })
                    .map(|(_, &local_doc_id)| SuperfileHit {
                        superfile: entry.uri,
                        local_doc_id,
                        score: 0.0,
                        stable_id: None,
                    })
                    .collect();
                dispatch::apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut hits, now)?;
                Ok(hits)
            }
        };
        let per_unit = dispatch::fanout_with(self, units, true, true, body).await?;
        let mut hits: Vec<SuperfileHit> = per_unit.into_iter().flatten().collect();
        dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        Ok(hits)
    }
}

impl SupertableReader {
    /// Single-column BM25 search over this reader's pinned snapshot,
    /// materialized as Arrow rows.
    ///
    /// This is the user-facing row-returning path. It runs the same
    /// BM25 hit kernel the SQL TVF uses, then resolves those top-k hits
    /// through the shared row materializer. Returned batches include
    /// `_id`, every visible scalar column, and a trailing `score` column.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = self.bm25_search_async(column, query, k, mode).await?;
            // `projection` selects columns by name (any of `_id`, the
            // visible scalar columns, or the trailing `score`); `None`
            // returns `_id` + `score` only. The shared resolver decodes
            // only the projected columns.
            let batch = resolve_hits_named(self, &hits, projection, "bm25_search")
                .await
                .map_err(|e| QueryError::Execute(e.to_string()))?;
            Ok(vec![batch])
        })
    }

    /// Low-level BM25 search over this reader's pinned snapshot.
    ///
    /// Drives the internal async kernel to completion via the
    /// sync→async bridge ([`SupertableReader::block_on`]). Returns up
    /// to `k` hits sorted by BM25 score *descending*.
    ///
    /// ## Query clauses (`+term`, `-term`)
    ///
    /// A `+`-prefixed term is a **must**: every hit contains it. A
    /// `-`-prefixed term is a **must-not**: docs containing it are
    /// excluded, regardless of score. Bare terms take their polarity
    /// from `mode`, the default operator — `And` requires them like
    /// musts; `Or` makes them scoring-only **shoulds** when a must
    /// exists (`"+climate policy"` matches the docs containing
    /// `climate`, ranking those that also mention `policy` higher)
    /// and a plain union when none does. A query with only negated
    /// terms is an error.
    pub fn bm25_hits(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_search_async(column, query, k, mode))
    }

    /// Prefix-expanded BM25 search — see [`SupertableReader::bm25_search`]
    /// for the bridge semantics.
    pub fn bm25_search_prefix(
        &self,
        column: &str,
        prefix: &str,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_search_prefix_async(column, prefix, k))
    }

    /// Unranked token match over this reader's pinned snapshot. Returns
    /// every row whose `column` matches `query`'s tokens under `mode`
    /// (`Or` = any token, `And` = every token). With a `+must` clause
    /// the match set is the musts' intersection and bare terms are
    /// ignored — unranked matching has no scores for a should to
    /// raise; `-term` exclusions apply. The returned hits are
    /// **unranked** — `score` is `0.0` and order is unspecified — unlike
    /// the ranked [`SupertableReader::bm25_search`]. Drives the async
    /// kernel via the sync→async bridge ([`SupertableReader::block_on`]).
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.token_match_async(column, query, mode))
    }

    /// Count documents matching `query`'s tokens under `mode` over this
    /// reader's pinned snapshot — count only, no scoring or row
    /// materialization. A single-token query on a delete-free superfile
    /// resolves in O(1) from the stored document frequency. Drives the
    /// async kernel via the sync→async bridge.
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.token_match_count_async(column, query, mode))
    }

    /// Unranked exact match of the raw string `value` against `column`
    /// over this reader's pinned snapshot — the two-pass index-pruned,
    /// text-verified match (see
    /// [`SuperfileReader::exact_match`](crate::superfile::SuperfileReader::exact_match)).
    /// Returns the rows whose stored value equals `value` exactly;
    /// hits are **unranked** (`score` is `0.0`).
    pub fn exact_match(&self, column: &str, value: &str) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.exact_match_async(column, value))
    }
}

/// One unit of per-superfile search work scheduled into the reader
/// pool's `par_iter`. `range == None` means "the whole superfile" and
/// dispatches to the un-ranged BM25 API; `range == Some((start,
/// end))` means "only doc_ids in [start, end)" and dispatches to
/// the range-aware OR path.
struct WorkUnit {
    entry: Arc<SuperfileEntry>,
    range: Option<(u32, u32)>,
}

/// Minimum docs per sub-range. Below this width, splitting adds
/// more pool-scheduling + per-shard top-K-merge overhead than it
/// saves in scoring work. Tuned to be coarse — the heuristic only
/// needs to avoid splitting toy superfiles; production superfiles at
/// the scales we benchmark (1.25M docs/superfile after 10M × cpus/2
/// row-shard) are well above this floor.
const SUBRANGE_MIN_DOCS: u32 = 50_000;

/// Map a per-superfile FTS read error to the query-layer error. A
/// phrase query against a column indexed without positions, or a query
/// with no positive clause to rank, is a malformed *request* — surface
/// it as [`QueryError::InvalidQuery`] so the caller sees a bad-input
/// error, not a storage/scan failure. Everything else is a genuine
/// read error and stays [`QueryError::Parquet`].
fn fts_read_error(e: ReadError) -> QueryError {
    match &e {
        ReadError::Fts(fts)
            if matches!(
                fts.as_ref(),
                FtsError::PositionsUnavailable { .. } | FtsError::NegationOnly
            ) =>
        {
            QueryError::InvalidQuery(e.to_string())
        }
        _ => QueryError::Parquet(e.to_string()),
    }
}

/// Minimum query term count that makes OR sub-range fan-out eligible.
/// The range-aware Block-Max MaxScore path is only wired up for
/// multi-term OR, so single-term queries stay whole-superfile.
const OR_FANOUT_MIN_TERMS: usize = 2;

/// How a query fans out over the kept superfiles.
enum FanOut {
    /// One un-ranged unit per superfile.
    PerSuperfile,
    /// Additionally slice big superfiles into doc-id sub-ranges when the
    /// reader pool has spare threads.
    SubRanges,
}

/// Pick the fan-out for a term query: only the pure multi-should
/// union (a flat multi-term OR — no must and no negated clause) has a
/// range-aware kernel, so everything else stays one un-ranged unit
/// per superfile.
fn fanout_for(n_musts: usize, n_shoulds: usize, has_negatives: bool) -> FanOut {
    if n_musts == 0 && n_shoulds >= OR_FANOUT_MIN_TERMS && !has_negatives {
        FanOut::SubRanges
    } else {
        FanOut::PerSuperfile
    }
}

/// Slice the kept superfiles into parallel work units — one
/// [`WorkUnit`] per (superfile, doc_id sub-range) tuple.
///
/// `FanOut::SubRanges` slices only when:
///   1. The reader pool has more threads than kept superfiles —
///      otherwise every thread is already saturated by one superfile
///      and splitting just adds overhead.
///   2. The candidate sub-range width is at least
///      `SUBRANGE_MIN_DOCS` — below that, BMM bookkeeping +
///      cross-sub-range top-K merge dominate the parallel win.
///
/// Otherwise each kept superfile becomes a single un-ranged work unit
/// — identical to the original `par_iter` over superfiles shape.
fn build_work_units(
    kept: &[&Arc<SuperfileEntry>],
    fanout: FanOut,
    pool_threads: usize,
) -> Vec<WorkUnit> {
    let want_subranges = pool_threads.div_ceil(kept.len().max(1)).max(1);
    if matches!(fanout, FanOut::PerSuperfile) || want_subranges <= 1 {
        return kept
            .iter()
            .map(|e| WorkUnit {
                entry: Arc::clone(e),
                range: None,
            })
            .collect();
    }

    let mut units: Vec<WorkUnit> = Vec::with_capacity(kept.len() * want_subranges);
    for entry in kept {
        let n_docs = entry.n_docs as u32;
        if n_docs == 0 {
            continue;
        }
        // Round the sub-range count down to avoid producing
        // narrower-than-floor slices. With `want_subranges = 2` on
        // a 1.25M-doc superfile, stride = 625K (well above floor) so
        // both sub-ranges fire. With a tiny superfile (e.g., 10K
        // docs, well below `SUBRANGE_MIN_DOCS`), the division
        // collapses to 1 sub-range = full superfile.
        let cap_by_floor = (n_docs / SUBRANGE_MIN_DOCS).max(1) as usize;
        let n_sub = want_subranges.min(cap_by_floor);
        if n_sub <= 1 {
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: None,
            });
            continue;
        }
        let stride = n_docs.div_ceil(n_sub as u32);
        let mut start: u32 = 0;
        while start < n_docs {
            let end = start.saturating_add(stride).min(n_docs);
            units.push(WorkUnit {
                entry: Arc::clone(entry),
                range: Some((start, end)),
            });
            start = end;
        }
    }
    units
}

/// Merge per-superfile hits and return the top-k by *descending*
/// score (highest BM25 = most relevant). Uses a min-heap of size k
/// so we never sort more than k elements.
/// Select the global top-k deterministically and compaction-stably: order
/// by score descending, breaking ties on the stable `_id` (ascending).
///
/// A plain score-only merge (`top_k_descending`) leaves the choice among
/// score-tied hits to segment completion order — the cross-superfile floor
/// changes which ties each segment returns, so the surviving tied docs vary
/// run to run. Physical keys (superfile uuid + local offset) would break the
/// tie but shift on every compaction. The stable `_id` is invariant across
/// compaction, so tie-breaking on it yields the same top-k as a
/// single-segment engine's docid-ordered ties, independent of layout or
/// completion order. `_id`s are resolved up front here — cheap because the
/// shared floor caps the candidate set near k.
async fn select_top_k_stable(
    tr: &SupertableReader,
    per_unit: Vec<Vec<SuperfileHit>>,
    k: usize,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let mut cands: Vec<SuperfileHit> = per_unit.into_iter().flatten().collect();
    // Narrow to the top-k *by score plus its boundary ties* before touching
    // `_id`. `_id` resolution costs a decode per hit, so it must stay
    // top-k-sized (never per-candidate — that's what the fan-out defers).
    // Partition at the k-th best score, then keep everything scoring at or
    // above it: the strictly-better hits are always in, and the ties at the
    // k-th score are the only ones whose inclusion the `_id` order decides.
    if cands.len() > k {
        cands.select_nth_unstable_by(k - 1, |a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal)
        });
        let kth_score = cands[k - 1].score;
        cands.retain(|c| c.score >= kth_score);
    }
    dispatch::attach_stable_ids_to_hits(tr, &mut cands).await?;
    // Total order: score desc, then stable `_id` asc — deterministic and
    // invariant across compaction (unlike physical superfile/offset keys).
    cands.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then(a.stable_id.cmp(&b.stable_id))
    });
    cands.truncate(k);
    Ok(cands)
}

impl Supertable {
    /// Single-column BM25 search over the current snapshot, returning
    /// Arrow rows best-score-first (BM25 relevance, higher is better).
    ///
    /// The query string carries lucene-style clause sigils: `+term`
    /// is a must (every hit contains it), `-term` a must-not (hard
    /// exclusion), and bare terms take their polarity from `mode`,
    /// the default operator (`And` ⇒ must, `Or` ⇒ scoring-only should
    /// once any must exists). `"+climate policy"` under `Or` matches
    /// the docs containing `climate` and ranks those also mentioning
    /// `policy` higher.
    ///
    /// A double-quoted run of words is an **exact phrase** atom: the
    /// words must appear adjacent and in order, verified against
    /// token positions. A phrase takes any clause polarity —
    /// `"new york" hotel`, `+"new york" +hotel`, `-"new york"` — and
    /// scores as one BM25 atom whose `tf` is the number of phrase
    /// occurrences and whose `idf` is the sum of its members'. Phrase
    /// queries require the column to be indexed with token positions
    /// (the `positions` flag on the column's FTS build config, off by
    /// default); against a positionless column they return a typed
    /// error rather than silently degrading to a bag-of-words match.
    /// A single-word phrase (`"york"`) is just that term.
    ///
    /// `score` is a similarity (higher is better) — the opposite
    /// direction from [`Supertable::vector_search`]'s distance. Fuse the
    /// two with [`Supertable::hybrid_search`], not by raw score.
    ///
    /// Pins a fresh reader (applying the read-consistency policy), runs
    /// the BM25 fan-out, and resolves the top-`k` hits to Arrow rows.
    ///
    /// `projection` selects output columns by name (any of `_id`, the
    /// visible scalar columns, or the trailing `score`); `None` returns
    /// the engine-native result — `_id` + `score` only. Only the
    /// projected scalar columns are decoded, so materializing row data
    /// is an explicit opt-in by column name.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, BoolMode, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema, vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox"]))])?)?;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = posts.bm25_search("body", "fox", 10, BoolMode::Or, None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Name columns to materialize row data:
    /// let rows = posts.bm25_search("body", "fox", 10, BoolMode::Or, Some(&["_id", "body", "score"]))?;
    /// assert_eq!(rows[0].num_columns(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, mode = ?mode))
    )]
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, k, mode = ?mode, "bm25_search");
        self.reader()
            .bm25_search(column, query, k, mode, projection)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("bm25_search", None))
    }

    /// Unranked token match over one FTS column: every row whose
    /// `column` matches `query`'s tokens under `mode` (`Or` = any token,
    /// `And` = every token). With a `+must` clause the match set is
    /// the musts' intersection and bare terms are ignored (no scores
    /// for a should to raise); `-term` exclusions apply. Quoted
    /// phrases participate as atoms exactly as in
    /// [`Supertable::bm25_search`]: an exact-adjacency match against
    /// token positions, requiring a positions-indexed column. Returns
    /// Arrow rows like [`Supertable::bm25_search`], but the `score`
    /// column is `0.0` and row order is unspecified — a candidate
    /// set, not a ranking. `projection` follows the same rules as
    /// `bm25_search`.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, mode = ?mode))
    )]
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, mode = ?mode, "token_match");
        let reader = self.reader();
        let hits = reader
            .token_match(column, query, mode)
            .map_err(|e| InfinoError::from(e).with_context("token_match", None))?;
        let batch = self
            .block_on_query(resolve_hits_named(
                &reader,
                &hits,
                projection,
                "token_match",
            ))
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("token_match", None))?;
        Ok(vec![batch])
    }

    /// Unranked exact match: rows whose `column` value equals `value`
    /// exactly (index-pruned, then text-verified). Returns Arrow rows
    /// like [`Supertable::bm25_search`], with `score` fixed at `0.0` and
    /// unspecified row order. `projection` follows the same rules as
    /// `bm25_search`.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column))
    )]
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, "exact_match");
        let reader = self.reader();
        let hits = reader
            .exact_match(column, value)
            .map_err(|e| InfinoError::from(e).with_context("exact_match", None))?;
        let batch = self
            .block_on_query(resolve_hits_named(
                &reader,
                &hits,
                projection,
                "exact_match",
            ))
            .map_err(|e| InfinoError::Query(e.to_string()).with_context("exact_match", None))?;
        Ok(vec![batch])
    }

    /// Count documents whose `column` matches `query`'s tokens under
    /// `mode` (`Or` = any token, `And` = every token) over the current
    /// snapshot — count only, no scoring or row materialization. A
    /// single-token query on a delete-free snapshot resolves in O(1) per
    /// superfile from the term dictionary's document frequency, so
    /// counting a high-frequency term is cheap.
    ///
    /// With a `+must` clause the count is the musts' intersection
    /// cardinality — bare (should) terms affect only scores, never
    /// which docs count, so `count("+climate policy")` is the number
    /// of docs containing `climate`. A lone must keeps the O(1) df
    /// fast path. `-term` exclusions apply as in search. Quoted
    /// phrases count exact-adjacency matches (verified against token
    /// positions, so the column must be positions-indexed) — every
    /// match is verified, giving exact phrase counts.
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use infino::arrow_array::{LargeStringArray, RecordBatch};
    /// # use infino::arrow_schema::{DataType, Field, Schema};
    /// # use infino::{connect, BoolMode, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema,
    /// #     vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox", "a lazy dog"]))],
    /// # )?)?;
    /// let n = posts.count("body", "fox", BoolMode::Or)?;
    /// assert_eq!(n, 1);
    /// // `+must` defines the count; bare terms are scoring-only:
    /// let n = posts.count("body", "+quick lazy", BoolMode::Or)?;
    /// assert_eq!(n, 1); // docs containing `quick`
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        self.reader()
            .count(column, query, mode)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("count", None))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, future::Future, sync::Arc};

    use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;
    use datafusion::prelude::{col, lit};
    use tokio::runtime::Builder;

    use super::{BoolMode, FanOut, build_work_units, fanout_for};
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            vector::layout::VectorLayout,
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        },
        test_helpers::default_tokenizer as tok,
    };

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: Future>(fut: F) -> F::Output {
        Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime")
            .block_on(fut)
    }

    fn schema_id_title() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn options_one_superfile_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    fn build_batch(_start: u64, titles: &[&str]) -> RecordBatch {
        let titles_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)]).expect("batch")
    }

    /// Build a single SuperfileBuilder containing the same docs as
    /// the supertable across all superfiles. Used as the oracle for
    /// per-superfile-vs-global BM25 set-membership tests.
    fn build_oracle_superfile(titles: &[&str]) -> Arc<SuperfileReader> {
        // The oracle path goes directly through SuperfileBuilder
        // (not through Supertable::append's auto-injection), so
        // we build the effective schema by hand: `_id` is
        // `Decimal128(38, 0)`, ids are 0..n.
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
            }],
            vec![],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let n = titles.len();
        let ids = Decimal128Array::from((0..n as i128).collect::<Vec<_>>())
            .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
            .expect("decimal128");
        let titles_arr = LargeStringArray::from(titles.to_vec());
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = Bytes::from(b.finish().expect("finish"));
        Arc::new(SuperfileReader::open(bytes).expect("open"))
    }

    #[test]
    fn negation_excludes_across_superfiles() {
        // 3 commits → 3 superfiles. "alpha -beta" must drop the one doc
        // containing beta and keep the other two alpha docs.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_hits("title", "alpha -beta", 10, BoolMode::Or)
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus beta: {hits:?}");

        // Positive-only stays untouched: all three alpha docs.
        let hits = r
            .bm25_hits("title", "alpha", 10, BoolMode::Or)
            .expect("positive search");
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn negated_term_does_not_prune_superfiles() {
        // "delta" exists only in superfile 2. Under And, if the negated
        // term leaked into the bloom prune, superfiles 1 and 3 (no delta)
        // would be wrongly dropped and the result would be empty; the
        // correct answer is superfile 1's two alpha docs.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha one", "alpha two"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["gamma three"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r
            .bm25_hits("title", "alpha -delta", 10, BoolMode::And)
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus delta: {hits:?}");
    }

    #[test]
    fn negation_only_query_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let res = r.bm25_hits("title", "-alpha", 10, BoolMode::Or);
        assert!(res.is_err(), "negation-only must error; got {res:?}");
    }

    #[test]
    fn count_and_token_match_negation_only_query_errors() {
        // The unranked count / token_match surfaces reject a negation-only
        // query (`-foo`) the same way the scored path does — there is no
        // positive anchor to match against. A token-less query (empty /
        // whitespace) is still 0 / empty, not an error.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader();

        for mode in [BoolMode::Or, BoolMode::And] {
            assert!(
                r.count("title", "-alpha", mode).is_err(),
                "negation-only count must error ({mode:?})"
            );
            assert!(
                r.token_match("title", "-alpha", mode).is_err(),
                "negation-only token_match must error ({mode:?})"
            );
        }
        // No positive anchor across several negated terms either.
        assert!(r.count("title", "-alpha -beta", BoolMode::Or).is_err());
        // Token-less queries stay non-error, 0 / empty.
        assert_eq!(r.count("title", "", BoolMode::Or).expect("empty"), 0);
        assert!(
            r.token_match("title", "   ", BoolMode::Or)
                .expect("blank")
                .is_empty()
        );
    }

    #[test]
    fn bm25_search_empty_supertable_returns_empty_without_store_calls() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 5, BoolMode::Or)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 0, BoolMode::Or)
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_returns_descending_score_order() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &[
                "rust rust rust async",
                "rust async runtime",
                "rust embedded",
                "python data",
            ],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 4, BoolMode::Or)
            .expect("query");
        // Should return 3 hits (the python doc has no `rust`).
        assert_eq!(hits.len(), 3);
        // Strictly descending.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn bm25_search_carries_superfile_uri_for_each_hit() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust rust async"])).expect("a1");
        w.commit().expect("c1");
        w.append(&build_batch(10, &["rust runtime"])).expect("a2");
        w.commit().expect("c2");

        let r = st.reader();
        assert_eq!(r.n_superfiles(), 2);
        let hits = r
            .bm25_hits("title", "rust", 5, BoolMode::Or)
            .expect("query");
        assert_eq!(hits.len(), 2);
        // Both superfile URIs should appear.
        let mut uris: Vec<_> = hits.iter().map(|h| h.superfile).collect();
        uris.sort();
        let expected: Vec<_> = {
            let mut v: Vec<_> = r.manifest().superfiles.iter().map(|e| e.uri).collect();
            v.sort();
            v
        };
        assert_eq!(uris, expected);
    }

    #[test]
    fn bm25_search_oracle_top_k_set_matches_single_superfile() {
        // Plant a corpus where the top-k under BM25 is unambiguous
        // regardless of per-superfile-vs-global IDF variation: 3 docs
        // contain the rare term `nimblefox`, distributed across 3
        // superfiles; the other 9 docs share only generic terms with
        // each other and with the query, so they score zero against
        // `nimblefox`. The set membership check survives even
        // though per-superfile IDF for `nimblefox` differs from
        // global IDF (it's `df=1` in each superfile vs `df=3` global).
        let titles = vec![
            "lookup nimblefox special token",   // 0  — match
            "ordinary common everyday text",    // 1
            "more usual filler corpus copy",    // 2
            "something boring without it",      // 3
            "mid corpus another nimblefox row", // 4  — match
            "generic page that adds nothing",   // 5
            "another stuffer no rare terms",    // 6
            "more padding here for filler",     // 7
            "tail nimblefox final superfile",   // 8  — match
            "another tail row",                 // 9
            "yet another normal title",         // 10
            "wrapping up the corpus today",     // 11
        ];

        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(4) {
            let end = (chunk_start + 4).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(st.reader().n_superfiles(), 3);

        let oracle = build_oracle_superfile(&titles);
        // Single-superfile `SuperfileReader` oracle: async-only search,
        // driven on a throwaway runtime. The supertable reader below
        // uses its sync public API.
        let oracle_hits = block_on(oracle.bm25_hits_async("title", "nimblefox", 5, BoolMode::Or))
            .expect("oracle");
        // Oracle should find exactly 3 docs containing `nimblefox`.
        assert_eq!(oracle_hits.len(), 3);
        let oracle_set: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_set, [0u32, 4, 8].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_hits("title", "nimblefox", 5, BoolMode::Or)
            .expect("supertable query");
        assert_eq!(st_hits.len(), 3);
        // Resolve supertable hits to global doc-ids via superfile
        // ordering (superfiles appear in append order; chunk size = 4).
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 4 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_globals, oracle_set);
    }

    #[test]
    fn bm25_search_prefix_oracle_top_k_set_matches_single_superfile() {
        let titles = vec![
            "rust async runtime",
            "rust embedded systems",
            "ruby gemfile config",
            "rustacean conference",
            "python machine learning",
            "python web framework",
            "rusty pipe rebuild",
            "go concurrency model",
        ];
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for chunk_start in (0..titles.len()).step_by(2) {
            let end = (chunk_start + 2).min(titles.len());
            let chunk = &titles[chunk_start..end];
            w.append(&build_batch(chunk_start as u64, chunk))
                .expect("append");
            w.commit().expect("commit");
        }

        let oracle = build_oracle_superfile(&titles);
        let oracle_hits = block_on(oracle.bm25_search_prefix("title", "rust", 5)).expect("oracle");
        let oracle_globals: HashSet<u32> = oracle_hits.iter().map(|(d, _)| *d).collect();

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_search_prefix("title", "rust", 5)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: HashSet<u32> = st_hits
            .iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.superfile)
                    .expect("superfile in manifest");
                (seg_idx as u32) * 2 + h.local_doc_id
            })
            .collect();
        assert_eq!(st_hits.len(), oracle_hits.len());
        assert_eq!(st_globals, oracle_globals);
        // Prefix-expansion sanity: we should hit "rust*" and
        // "rusty*" / "rustacean*" but not "ruby*".
        assert!(st_hits.len() >= 4);
    }

    #[test]
    fn bm25_search_prefix_unmatched_prefix_returns_empty() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r.bm25_search_prefix("title", "zzzz", 10).expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_prefix_lowercases_input() {
        // Index stores tokenized terms (lowercased); user provides
        // mixed-case prefix; we lowercase before expansion so the
        // FST walk finds the matching subtree.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["Rust async runtime"]))
            .expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let hits = r.bm25_search_prefix("title", "RUST", 5).expect("query");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn bm25_search_unknown_column_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader();
        let err = r
            .bm25_hits("missing_column", "rust", 5, BoolMode::Or)
            .expect_err("expected error");
        assert!(matches!(err, QueryError::Parquet(_)), "got {err:?}");
    }

    #[test]
    fn bm25_search_results_global_top_k_caps_at_k() {
        // 4 superfiles × 1 doc each = 4 hits; ask for k=2; expect 2.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for i in 0..4 {
            w.append(&build_batch(i * 10, &["rust async runtime"]))
                .expect("a");
            w.commit().expect("c");
        }
        let r = st.reader();
        let hits = r
            .bm25_hits("title", "rust", 2, BoolMode::Or)
            .expect("query");
        assert_eq!(hits.len(), 2);
    }

    fn seeded_three_doc_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["the quick brown fox", "a lazy dog", "quick thinking"],
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn supertable_bm25_search_rows_default_and_projected() {
        let st = seeded_three_doc_supertable();

        // Bare call → `_id` + `score` only (no scalar decode).
        let bare = st
            .bm25_search("title", "fox", 10, BoolMode::Or, None)
            .expect("bm25 rows");
        assert_eq!(bare.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(bare[0].num_columns(), 2, "_id + score");

        // Named projection materializes the requested columns.
        let rows = st
            .bm25_search(
                "title",
                "fox",
                10,
                BoolMode::Or,
                Some(&["_id", "title", "score"]),
            )
            .expect("bm25 projected rows");
        assert_eq!(rows[0].num_columns(), 3);
    }

    #[test]
    fn supertable_token_match_and_exact_match_rows() {
        let st = seeded_three_doc_supertable();

        // token_match: any row containing "quick" (Or over one token).
        let tm = st
            .token_match("title", "quick", BoolMode::Or, None)
            .expect("token_match");
        assert_eq!(tm.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

        // exact_match: only the row equal to the raw string.
        let em = st
            .exact_match("title", "a lazy dog", Some(&["_id", "title"]))
            .expect("exact_match");
        assert_eq!(em.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(em[0].num_columns(), 2);
    }

    #[test]
    fn reader_token_match_and_exact_match_hits() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();

        // token_match And requires every token to be present.
        let any = r.token_match("title", "quick", BoolMode::And).expect("tm");
        assert_eq!(any.len(), 2);

        // Token-less value (punctuation only) prunes nothing and matches
        // no stored row exactly.
        let none = r.exact_match("title", "!!!").expect("em punctuation");
        assert!(none.is_empty());

        // Exact verify against a real row.
        let one = r.exact_match("title", "quick thinking").expect("em");
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn token_match_empty_query_short_circuits() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();
        // A query that tokenizes to nothing returns empty without
        // touching the store.
        let hits = r
            .token_match("title", "   ", BoolMode::Or)
            .expect("tm empty");
        assert!(hits.is_empty());
    }

    /// Two-superfile fixture for the clause model: `climate` docs are
    /// split across superfiles, and one superfile has no `climate` at
    /// all (so the must prune drops it).
    fn seeded_clause_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["climate change policy", "climate science report"],
        ))
        .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(
            10,
            &["policy analysis quarterly", "climate policy summit"],
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    /// Positional twin of the options fixture, for phrase queries.
    fn options_positional_one_superfile_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig {
                column: "title".into(),
                positions: true,
            }],
            vec![],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Two superfiles with controlled "new york" adjacency: docs in
    /// the first commit match (0, 1), the second commit has both
    /// words non-adjacent plus one more match.
    fn seeded_phrase_supertable() -> Supertable {
        let st = Supertable::create(options_positional_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["new york city", "the new york times"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(10, &["york loves new haven", "big new york"]))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn phrase_query_end_to_end() {
        let st = seeded_phrase_supertable();
        let r = st.reader();

        // Ranked: exactly the adjacent-in-order docs across both
        // superfiles.
        let hits = r
            .bm25_hits("title", r#""new york""#, 10, BoolMode::Or)
            .expect("phrase hits");
        assert_eq!(hits.len(), 3, "three docs contain the phrase");

        // Count = the phrase match set.
        let n = r
            .count("title", r#""new york""#, BoolMode::Or)
            .expect("phrase count");
        assert_eq!(n, 3);
        // The non-adjacent doc is the difference vs the token AND.
        let and_count = r
            .count("title", "+new +york", BoolMode::Or)
            .expect("token and count");
        assert_eq!(and_count, 4);

        // Phrase composed with clauses: must-phrase + must-term.
        let hits = r
            .bm25_hits("title", r#"+"new york" +the"#, 10, BoolMode::Or)
            .expect("phrase + term");
        assert_eq!(hits.len(), 1);

        // Negated phrase: docs with `york` minus the phrase docs.
        let n = r
            .count("title", r#"york -"new york""#, BoolMode::Or)
            .expect("negated phrase count");
        assert_eq!(n, 1);
    }

    #[test]
    fn phrase_on_positionless_table_errors() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        let err = r
            .bm25_hits("title", r#""climate change""#, 10, BoolMode::Or)
            .expect_err("typed error expected");
        // A phrase on a positionless column is a bad *request*, not a
        // read failure — it surfaces as InvalidQuery, and the message
        // explains the missing positions.
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "phrase on positionless column should be InvalidQuery, got {err:?}"
        );
        assert!(
            err.to_string().contains("positions"),
            "error should say positions are missing: {err}"
        );
        let err = r
            .count("title", r#""climate change""#, BoolMode::Or)
            .expect_err("count errors too");
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "count phrase on positionless column should be InvalidQuery, got {err:?}"
        );
        assert!(err.to_string().contains("positions"));
    }

    #[test]
    fn must_should_match_set_and_count_across_superfiles() {
        let st = seeded_clause_supertable();
        let r = st.reader();

        // 3 docs contain `climate`; `policy` is scoring-only and must
        // not pull in "policy analysis quarterly".
        let hits = r
            .bm25_hits("title", "+climate policy", 10, BoolMode::Or)
            .expect("bm25 +climate policy");
        assert_eq!(hits.len(), 3, "match set is the must set");

        // Count agrees with the scored match set and ignores shoulds.
        let n = r
            .count("title", "+climate policy", BoolMode::Or)
            .expect("count +climate policy");
        assert_eq!(n, 3);
        // Flat OR over the same tokens is the union — strictly bigger.
        let union = r
            .count("title", "climate policy", BoolMode::Or)
            .expect("count union");
        assert_eq!(union, 4);

        // Docs matching must+should outrank must-only docs: both
        // climate∧policy docs come first.
        let top2: Vec<f32> = hits.iter().take(2).map(|h| h.score).collect();
        let third = hits[2].score;
        assert!(
            top2.iter().all(|s| *s > third),
            "climate∧policy docs must outrank climate-only: {hits:?}"
        );
    }

    #[test]
    fn must_should_token_match_matches_musts_only() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // Unranked matching has no scores for the should to raise —
        // the match set is exactly the must set.
        let tm = r
            .token_match("title", "+climate policy", BoolMode::Or)
            .expect("tm +climate policy");
        assert_eq!(tm.len(), 3);
    }

    #[test]
    fn must_should_with_negation_across_superfiles() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // Negation still excludes: drop the summit doc from the
        // climate must set.
        let hits = r
            .bm25_hits("title", "+climate policy -summit", 10, BoolMode::Or)
            .expect("bm25 with negation");
        assert_eq!(hits.len(), 2);
        let n = r
            .count("title", "+climate policy -summit", BoolMode::Or)
            .expect("count with negation");
        assert_eq!(n, 2);
    }

    #[test]
    fn absent_must_prunes_every_superfile() {
        let st = seeded_clause_supertable();
        let r = st.reader();
        // The must term exists nowhere: bloom-prune (or the empty
        // intersection) yields no hits despite the common should.
        let hits = r
            .bm25_hits("title", "+zzzabsent policy", 10, BoolMode::Or)
            .expect("bm25 absent must");
        assert!(hits.is_empty());
        let n = r
            .count("title", "+zzzabsent policy", BoolMode::Or)
            .expect("count absent must");
        assert_eq!(n, 0);
    }

    #[test]
    fn token_match_no_match_returns_empty() {
        let st = seeded_three_doc_supertable();
        let r = st.reader();
        let hits = r
            .token_match("title", "nonexistentterm", BoolMode::Or)
            .expect("tm");
        assert!(hits.is_empty());
    }

    #[test]
    fn fanout_for_only_multi_term_or_without_negation_subranges() {
        // Multi-should union (flat multi-term OR), no negation →
        // sub-range eligible.
        assert!(matches!(fanout_for(0, 2, false), FanOut::SubRanges));
        // Single should stays per-superfile.
        assert!(matches!(fanout_for(0, 1, false), FanOut::PerSuperfile));
        // Negation disables sub-ranges.
        assert!(matches!(fanout_for(0, 2, true), FanOut::PerSuperfile));
        // Any must clause (including flat And queries, whose bare
        // terms all resolve to musts) stays per-superfile.
        assert!(matches!(fanout_for(2, 0, false), FanOut::PerSuperfile));
        assert!(matches!(fanout_for(1, 1, false), FanOut::PerSuperfile));
    }

    #[test]
    fn build_work_units_per_superfile_is_one_unranged_unit_each() {
        use std::collections::HashMap;

        use uuid::Uuid;

        use crate::supertable::manifest::{SuperfileEntry, SuperfileUri};

        fn entry(n_docs: u64) -> Arc<SuperfileEntry> {
            let id = Uuid::new_v4();
            Arc::new(SuperfileEntry {
                birth_version: 0,
                superfile_id: id,
                uri: SuperfileUri(id),
                n_docs,
                id_min: 0,
                id_max: n_docs.saturating_sub(1) as i128,
                scalar_stats: HashMap::new(),
                fts_summary: HashMap::new(),
                vector_summary: HashMap::new(),
                partition_key: Vec::new(),
                partition_hint: None,
                vector_layout: VectorLayout::Ivf,
                subsection_offsets: None,
            })
        }

        let e0 = entry(100);
        let e1 = entry(200);
        let kept = vec![&e0, &e1];

        // PerSuperfile always yields exactly one un-ranged unit per kept
        // superfile regardless of pool width.
        let units = build_work_units(&kept, FanOut::PerSuperfile, 8);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));

        // SubRanges with one pool thread collapses to per-superfile too
        // (no spare threads to slice across).
        let units = build_work_units(&kept, FanOut::SubRanges, 1);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));

        // Tiny superfiles below SUBRANGE_MIN_DOCS never slice even with
        // spare threads.
        let units = build_work_units(&kept, FanOut::SubRanges, 16);
        assert_eq!(units.len(), 2);
        assert!(units.iter().all(|u| u.range.is_none()));
    }

    #[test]
    fn build_work_units_slices_large_superfiles_when_threads_spare() {
        use std::collections::HashMap;

        use uuid::Uuid;

        use crate::supertable::manifest::{SuperfileEntry, SuperfileUri};

        let id = Uuid::new_v4();
        // One large superfile, well above SUBRANGE_MIN_DOCS (50k).
        let big = Arc::new(SuperfileEntry {
            birth_version: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 200_000,
            id_min: 0,
            id_max: 199_999,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        });
        let kept = vec![&big];
        // 4 spare threads, 1 superfile → slice into multiple ranged units
        // that tile [0, n_docs) without gaps.
        let units = build_work_units(&kept, FanOut::SubRanges, 4);
        assert!(units.len() > 1, "large superfile sliced into sub-ranges");
        let mut cursor = 0u32;
        for u in &units {
            let (start, end) = u.range.expect("ranged unit");
            assert_eq!(start, cursor);
            cursor = end;
        }
        assert_eq!(cursor, 200_000, "sub-ranges tile the whole superfile");
    }

    #[test]
    fn count_single_term_sums_df_across_superfiles() {
        // 3 commits → 3 superfiles. Single-term count takes the O(1)
        // term_df fast path (no deletes) and sums across superfiles.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);
        assert_eq!(st.count("title", "beta", BoolMode::Or).expect("count"), 2);
        assert_eq!(st.count("title", "gamma", BoolMode::Or).expect("count"), 2);
        assert_eq!(st.count("title", "absent", BoolMode::Or).expect("count"), 0);
    }

    #[test]
    fn count_multi_term_sums_across_superfiles() {
        // 3 commits → 3 superfiles. Multi-term queries take the general
        // `token_match` branch (not the single-term df fast path), so this
        // exercises summing per-superfile match counts across superfiles
        // for both OR (union spans all three) and AND (intersection lands
        // in one). Doc ids are globally unique across commits.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["beta gamma", "delta"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(4, &["alpha delta", "beta"]))
            .expect("append");
        w.commit().expect("commit");

        // OR "alpha beta": alpha∪beta matches in all three superfiles
        // (2 + 1 + 2) — proves the per-superfile counts are summed.
        assert_eq!(st.count("title", "alpha beta", BoolMode::Or).expect("c"), 5);
        // OR "gamma delta": 1 + 2 + 1 across the three superfiles.
        assert_eq!(
            st.count("title", "gamma delta", BoolMode::Or).expect("c"),
            4
        );
        // AND "alpha beta": both terms only in the first superfile's
        // "alpha beta" doc → 1 (the other superfiles contribute 0).
        assert_eq!(
            st.count("title", "alpha beta", BoolMode::And).expect("c"),
            1
        );
        // AND "alpha delta": both terms only in the third superfile.
        assert_eq!(
            st.count("title", "alpha delta", BoolMode::And).expect("c"),
            1
        );

        // Cross-check every shape against token_match cardinality.
        let r = st.reader();
        for (q, mode) in [
            ("alpha beta", BoolMode::Or),
            ("gamma delta", BoolMode::Or),
            ("alpha beta", BoolMode::And),
            ("alpha delta", BoolMode::And),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_honors_or_and_modes() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta"],
        ))
        .expect("append");
        w.commit().expect("commit");

        // OR: docs containing alpha OR delta → all three.
        assert_eq!(
            st.count("title", "alpha delta", BoolMode::Or).expect("c"),
            3
        );
        // AND: docs containing both alpha AND beta → just "alpha beta".
        assert_eq!(
            st.count("title", "alpha beta", BoolMode::And).expect("c"),
            1
        );
        // AND with no doc holding both → 0.
        assert_eq!(
            st.count("title", "gamma delta", BoolMode::And).expect("c"),
            0
        );
    }

    #[test]
    fn count_agrees_with_token_match_len() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta"],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        for (q, mode) in [
            ("alpha", BoolMode::Or),
            ("alpha delta", BoolMode::Or),
            ("alpha beta", BoolMode::And),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_empty_query_and_empty_supertable_are_zero() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        // Empty supertable: nothing matches.
        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("c"), 0);
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");
        // Token-less queries produce no terms → 0.
        assert_eq!(st.count("title", "", BoolMode::Or).expect("c"), 0);
        assert_eq!(st.count("title", "   ", BoolMode::Or).expect("c"), 0);
    }

    #[test]
    fn count_excludes_tombstoned_docs() {
        // Storage-backed so delete (tombstones) is available. After a
        // delete, the single-term count must drop the term_df fast path
        // and subtract the tombstone — df would over-count.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha one", "alpha two", "alpha three"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer slot so `delete` can acquire it

        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);

        let stats = st
            .delete(col("title").eq(lit("alpha two")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);

        // term_df still says 3; the count must subtract the tombstone → 2.
        assert_eq!(
            st.count("title", "alpha", BoolMode::Or)
                .expect("count after delete"),
            2
        );
    }

    #[test]
    fn count_excludes_negated_terms() {
        // A count query with a negated term must drop the docs matching
        // that term, the same way a scored search does. The earlier count
        // path tokenized "alpha -beta" into ["alpha", "beta"] and counted
        // "beta" as a positive, so it over-counted instead of excluding.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta", "alpha gamma"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(2, &["alpha delta"])).expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(3, &["beta gamma"])).expect("append");
        w.commit().expect("commit");

        // "alpha" matches three docs across the superfiles; "-beta" drops
        // the one that also contains beta → 2. Mirrors the search-side
        // `negation_excludes_across_superfiles`.
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count"),
            2
        );
        // Positive-only count is unchanged: all three alpha docs.
        assert_eq!(st.count("title", "alpha", BoolMode::Or).expect("count"), 3);
        // A negated term absent from the corpus excludes nothing.
        assert_eq!(
            st.count("title", "alpha -absent", BoolMode::Or)
                .expect("count"),
            3
        );
    }

    #[test]
    fn count_with_negation_agrees_with_token_match() {
        // The count↔token_match invariant must hold for negated queries
        // too, across OR / AND and single- vs multi-positive shapes.
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha beta", "alpha gamma", "beta delta", "gamma delta"],
        ))
        .expect("append");
        w.commit().expect("commit");
        let r = st.reader();
        for (q, mode) in [
            ("alpha -beta", BoolMode::Or),
            ("alpha gamma -delta", BoolMode::Or),
            ("alpha -gamma", BoolMode::And),
            ("beta -alpha", BoolMode::Or),
        ] {
            let c = r.count("title", q, mode).expect("count");
            let n = r.token_match("title", q, mode).expect("token_match").len() as u64;
            assert_eq!(c, n, "count vs token_match for {q:?} {mode:?}");
        }
    }

    #[test]
    fn count_excludes_negated_terms_and_tombstones() {
        // Negation and deletes compose: the materialized count drops both
        // negated-term docs and tombstoned docs in one pass.
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(
            0,
            &["alpha one", "alpha two", "alpha beta", "alpha three"],
        ))
        .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer slot so `delete` can acquire it

        // 4 alpha docs minus the one also containing beta → 3.
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count"),
            3
        );

        // Delete one of the surviving alpha docs; the count drops it too.
        let stats = st
            .delete(col("title").eq(lit("alpha two")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        assert_eq!(
            st.count("title", "alpha -beta", BoolMode::Or)
                .expect("count after delete"),
            2
        );
    }
}
