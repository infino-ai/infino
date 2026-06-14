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
//! holds a pinned `Arc<Manifest>`; for each visible superfile we:
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
//! Manifest-level skip pruning is wired in: each call computes a
//! per-superfile keep/prune mask from the FTS bloom (exact-term
//! mode) or the lex term range (prefix mode) before issuing
//! per-superfile work, so pruned superfiles never trigger a
//! `SuperfileReaderCache::reader` call. Vector + SQL skip remain
//! deferred (see those modules' headers).

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::slice;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;

use crate::superfile::SuperfileReader;
pub use crate::superfile::fts::reader::BoolMode;
use crate::superfile::fts::tokenize::{AsciiLowerTokenizer, Tokenizer};
use crate::supertable::error::QueryError;
use crate::supertable::handle::{Supertable, SupertableReader};
use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::query::SuperfileHit;
use crate::supertable::query::exec::common::resolve_hits_named;
use crate::supertable::query::prune::{PruneLeaf, select_superfiles};

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
    heap: std::sync::Mutex<std::collections::BinaryHeap<std::cmp::Reverse<OrdScore>>>,
    /// f32 bits of the current floor; `NEG_INFINITY` until `k` scores
    /// have been seen. Monotonically non-decreasing.
    floor_bits: std::sync::atomic::AtomicU32,
}

/// Total-order f32 wrapper for the [`SharedTopK`] heap (BM25 scores
/// are finite, but `f32` still needs an `Ord` shim).
#[derive(PartialEq)]
struct OrdScore(f32);
impl Eq for OrdScore {}
impl PartialOrd for OrdScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl SharedTopK {
    fn new(k: usize) -> Arc<Self> {
        Arc::new(Self {
            k,
            heap: std::sync::Mutex::new(std::collections::BinaryHeap::new()),
            floor_bits: std::sync::atomic::AtomicU32::new(f32::NEG_INFINITY.to_bits()),
        })
    }

    /// The current global floor — `NEG_INFINITY` until k scores merged.
    fn floor(&self) -> f32 {
        f32::from_bits(self.floor_bits.load(std::sync::atomic::Ordering::Acquire))
    }

    /// Merge one finished segment's (tombstone-surviving) scores and
    /// publish the new kth-best as the floor once k scores are known.
    fn merge(&self, scores: impl IntoIterator<Item = f32>) {
        let mut heap = self.heap.lock().expect("SharedTopK mutex poisoned");
        for s in scores {
            if heap.len() < self.k {
                heap.push(std::cmp::Reverse(OrdScore(s)));
            } else if let Some(std::cmp::Reverse(OrdScore(min))) = heap.peek()
                && s > *min
            {
                heap.pop();
                heap.push(std::cmp::Reverse(OrdScore(s)));
            }
        }
        if heap.len() == self.k
            && let Some(std::cmp::Reverse(OrdScore(min))) = heap.peek()
        {
            // The heap min only rises, so a plain store stays monotone
            // under the lock.
            self.floor_bits
                .store(min.to_bits(), std::sync::atomic::Ordering::Release);
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

        // Parse the query once here, not per superfile. The fan-out
        // closures below need owned ('static) data for tokio::spawn,
        // so this is the one place the tokens are copied — the prune
        // and every per-superfile search reuse them.
        let parsed = AsciiLowerTokenizer.parse(query);
        let positives: Vec<String> = parsed.positives.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = parsed.negatives.into_iter().map(Cow::into_owned).collect();

        // Pick the superfiles to search, via the shared two-tier bloom
        // prune. Only positive terms prune — a negated term must never
        // drop a superfile: a superfile without it is the best case (none
        // of its docs can be excluded), and under `And` it would even
        // prune every superfile that lacks the negated term.
        // The leaf takes `positives` by value to avoid cloning the
        // list; we take it back right after the call.
        let prune_leaf = PruneLeaf::TermPresence {
            column: column_owned.clone(),
            terms: positives,
            mode,
        };
        let kept = select_superfiles(manifest.as_ref(), slice::from_ref(&prune_leaf)).await?;
        let PruneLeaf::TermPresence {
            terms: positives, ..
        } = prune_leaf
        else {
            unreachable!("leaf constructed as TermPresence above")
        };
        if kept.is_empty() {
            return Ok(Vec::new());
        }

        // Build the work-unit list. When the reader pool has more
        // threads than there are kept superfiles AND we're on the
        // multi-term OR hot path, slice each superfile into doc_id
        // sub-ranges so the fan-out can saturate every pool thread.
        // Single-term OR and AND stay on the un-ranged call.
        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        let fanout = fanout_for(mode, positives.len(), !negatives.is_empty());
        let work_units = build_work_units(&kept_refs, fanout, pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, (Option<(u32, u32)>, uuid::Uuid))> = work_units
            .into_iter()
            .map(|u| {
                let suid = u.entry.superfile_id;
                (u.entry, (u.range, suid))
            })
            .collect();

        let term_arc: Arc<Vec<String>> = Arc::new(positives);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives);
        let column_arc = Arc::new(column_owned);

        // Cross-segment threshold sharing: each unit reads the global
        // kth-best floor before searching and merges its surviving
        // scores back after — late units skip every block that can't
        // beat what earlier units already found. Tombstoned hits are
        // excluded from the merge so deleted rows never raise the bar.
        let shared = SharedTopK::new(k);
        let tombstones = self.tombstone_cache.clone();
        let now = std::time::Instant::now();

        // One shared fan-out (`query::dispatch::fanout`) — the same
        // orchestrator the vector path uses. It warms the tombstone
        // sidecars in one batch, opens each superfile reader and runs the
        // kernel under `tokio::spawn` so cold GETs overlap, then tags +
        // tombstone-filters each unit's hits. The per-unit `params` is
        // the optional doc-id sub-range (`None` searches the whole
        // superfile) plus the superfile id for the tombstone-aware merge.
        let kernel = move |r: Arc<SuperfileReader>,
                           (range, suid): (Option<(u32, u32)>, uuid::Uuid)| {
            let column_arc = Arc::clone(&column_arc);
            let term_arc = Arc::clone(&term_arc);
            let neg_arc = Arc::clone(&neg_arc);
            let shared = Arc::clone(&shared);
            let tombstones = tombstones.clone();
            async move {
                let term_refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                let floor = shared.floor();
                let hits = match range {
                    // Ranged units exist only for negation-free queries
                    // (`fanout_for` never slices otherwise).
                    Some((start, end)) => r
                        .bm25_search_or_range_pretokenized_with_floor(
                            &column_arc,
                            &term_refs,
                            k,
                            start,
                            end,
                            floor,
                        )
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string()))?,
                    None if neg_arc.is_empty() => r
                        .bm25_search_pretokenized_with_floor(
                            &column_arc,
                            &term_refs,
                            k,
                            mode,
                            floor,
                        )
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string()))?,
                    None => {
                        // Negated queries run unfloored (the excluding
                        // kernel carries no cross-segment floor today);
                        // their survivors still merge below, which is
                        // harmless bookkeeping.
                        let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                        r.bm25_search_pretokenized_excluding(
                            &column_arc,
                            &term_refs,
                            &neg_refs,
                            k,
                            mode,
                        )
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string()))?
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
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;

        Ok(top_k_descending(per_unit, k))
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

        // Manifest-level term-range skip uses the same
        // lowercased prefix bytes the v1 tokenizer +
        // FST-expansion path use, so the skip's
        // lex-range overlap test exactly matches the
        // tokenizer's interpretation of the prefix.
        let prefix_lower = prefix_owned.to_ascii_lowercase();

        // Superfile selection via the shared two-tier prune — the
        // single-`Prefix`-leaf case (part-level term-range skip →
        // lazy-load surviving parts → per-superfile term-range skip).
        let kept = crate::supertable::query::prune::select_superfiles(
            manifest.as_ref(),
            &[crate::supertable::query::prune::PruneLeaf::Prefix {
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
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                    None => r
                        .bm25_search_prefix(&column_arc, &prefix_arc, k)
                        .await
                        .map_err(|e| QueryError::Parquet(e.to_string())),
                }
            }
        };
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;

        Ok(top_k_descending(per_unit, k))
    }

    /// Unranked token match across the pinned snapshot. Returns
    /// every row matching `query`'s tokens under `mode` (`Or` = any
    /// token, `And` = every token) as [`SuperfileHit`]s — **no scoring**
    /// (`score` is left `0.0`; these results are unordered). Superfile
    /// skip uses the same term-bloom prune as BM25.
    ///
    /// `pub(crate)` async kernel; the public surface is the sync
    /// [`SupertableReader::token_match`].
    pub(crate) async fn token_match_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let manifest = self.manifest();
        let term_strings: Vec<String> = AsciiLowerTokenizer.tokenize(query).collect();
        if term_strings.is_empty() {
            return Ok(Vec::new());
        }
        let kept = crate::supertable::query::prune::select_superfiles(
            manifest.as_ref(),
            &[crate::supertable::query::prune::PruneLeaf::TermPresence {
                column: column.to_owned(),
                terms: term_strings.clone(),
                mode,
            }],
        )
        .await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(term_strings);
        let kernel = move |r: Arc<SuperfileReader>, _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let term_arc = Arc::clone(&term_arc);
            async move {
                let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                let docs = r
                    .token_match(&column_arc, &refs, mode)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?;
                Ok(docs.into_iter().map(|d| (d, 0.0f32)).collect::<Vec<_>>())
            }
        };
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;
        Ok(per_unit.into_iter().flatten().collect())
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
            vec![crate::supertable::query::prune::PruneLeaf::TermPresence {
                column: column.to_owned(),
                terms: term_strings,
                mode: BoolMode::And,
            }]
        };
        let kept =
            crate::supertable::query::prune::select_superfiles(manifest.as_ref(), &leaves).await?;
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_arc = Arc::new(column.to_owned());
        let value_arc = Arc::new(value.to_owned());
        let kernel = move |r: Arc<SuperfileReader>, _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let value_arc = Arc::clone(&value_arc);
            async move {
                let docs = r
                    .exact_match(&column_arc, &value_arc)
                    .await
                    .map_err(|e| QueryError::Parquet(e.to_string()))?;
                Ok(docs.into_iter().map(|d| (d, 0.0f32)).collect::<Vec<_>>())
            }
        };
        let per_unit = crate::supertable::query::dispatch::fanout(self, units, kernel).await?;
        Ok(per_unit.into_iter().flatten().collect())
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
    /// ## Negation (`-term`)
    ///
    /// A `-`-prefixed query term excludes every doc containing it,
    /// regardless of score; `mode` applies to the positive terms only.
    /// `"rust -python"` scores docs with `rust`, dropping any that also
    /// contain `python`. A query with only negated terms is an error.
    pub fn bm25_hits(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
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
        self.block_on(self.bm25_search_prefix_async(column, prefix, k))
    }

    /// Unranked token match over this reader's pinned snapshot. Returns
    /// every row whose `column` matches `query`'s tokens under `mode`
    /// (`Or` = any token, `And` = every token). The returned hits are
    /// **unranked** — `score` is `0.0` and order is unspecified — unlike
    /// the ranked [`SupertableReader::bm25_search`]. Drives the async
    /// kernel via the sync→async bridge ([`SupertableReader::block_on`]).
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.block_on(self.token_match_async(column, query, mode))
    }

    /// Unranked exact match of the raw string `value` against `column`
    /// over this reader's pinned snapshot — the two-pass index-pruned,
    /// text-verified match (see
    /// [`SuperfileReader::exact_match`](crate::superfile::SuperfileReader::exact_match)).
    /// Returns the rows whose stored value equals `value` exactly;
    /// hits are **unranked** (`score` is `0.0`).
    pub fn exact_match(&self, column: &str, value: &str) -> Result<Vec<SuperfileHit>, QueryError> {
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

/// Pick the fan-out for a term query: only multi-term OR has a
/// range-aware kernel, and negation has no ranged kernel (v1), so
/// everything else stays one un-ranged unit per superfile.
fn fanout_for(mode: BoolMode, n_positives: usize, has_negatives: bool) -> FanOut {
    if mode == BoolMode::Or && n_positives >= OR_FANOUT_MIN_TERMS && !has_negatives {
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
fn top_k_descending(per_superfile: Vec<Vec<SuperfileHit>>, k: usize) -> Vec<SuperfileHit> {
    #[derive(PartialEq)]
    struct MinByScore(SuperfileHit);
    impl Eq for MinByScore {}
    impl PartialOrd for MinByScore {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for MinByScore {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .0
                .score
                .partial_cmp(&self.0.score)
                .unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::with_capacity(k + 1);
    for hit in per_superfile.into_iter().flatten() {
        if heap.len() < k {
            heap.push(MinByScore(hit));
        } else if let Some(worst) = heap.peek()
            && hit.score > worst.0.score
        {
            heap.pop();
            heap.push(MinByScore(hit));
        }
    }
    let mut result: Vec<SuperfileHit> = heap.into_iter().map(|m| m.0).collect();
    result.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    result
}

impl Supertable {
    /// Single-column BM25 search over the current snapshot, returning
    /// Arrow rows best-score-first (BM25 relevance, higher is better).
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
    /// # use arrow_array::{LargeStringArray, RecordBatch};
    /// # use arrow_schema::{DataType, Field, Schema};
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
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        self.reader()
            .bm25_search(column, query, k, mode, projection)
            .map_err(crate::InfinoError::from)
    }

    /// Unranked token match over one FTS column: every row whose
    /// `column` matches `query`'s tokens under `mode` (`Or` = any token,
    /// `And` = every token). Returns Arrow rows like
    /// [`Supertable::bm25_search`], but the `score` column is `0.0` and
    /// row order is unspecified — a candidate set, not a ranking.
    /// `projection` follows the same rules as `bm25_search`.
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        let reader = self.reader();
        let hits = reader
            .token_match(column, query, mode)
            .map_err(crate::InfinoError::from)?;
        let batch = self
            .block_on_query(resolve_hits_named(
                &reader,
                &hits,
                projection,
                "token_match",
            ))
            .map_err(|e| crate::InfinoError::Query(e.to_string()))?;
        Ok(vec![batch])
    }

    /// Unranked exact match: rows whose `column` value equals `value`
    /// exactly (index-pruned, then text-verified). Returns Arrow rows
    /// like [`Supertable::bm25_search`], with `score` fixed at `0.0` and
    /// unspecified row order. `projection` follows the same rules as
    /// `bm25_search`.
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, crate::InfinoError> {
        let reader = self.reader();
        let hits = reader
            .exact_match(column, value)
            .map_err(crate::InfinoError::from)?;
        let batch = self
            .block_on_query(resolve_hits_named(
                &reader,
                &hits,
                projection,
                "exact_match",
            ))
            .map_err(|e| crate::InfinoError::Query(e.to_string()))?;
        Ok(vec![batch])
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::{FtsConfig, SuperfileBuilder};

    use crate::supertable::error::QueryError;
    use crate::supertable::{Supertable, SupertableOptions};

    use super::BoolMode;

    use crate::test_helpers::default_tokenizer as tok;

    /// Drive an async future to completion on a throwaway current-thread
    /// runtime. Used only for the single-superfile `SuperfileReader`
    /// oracle, whose search surface is async-only; the supertable
    /// reader's own search methods are sync and need no runtime here.
    fn block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
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
    fn build_oracle_superfile(titles: &[&str]) -> Arc<crate::superfile::SuperfileReader> {
        // The oracle path goes directly through SuperfileBuilder
        // (not through Supertable::append's auto-injection), so
        // we build the effective schema by hand: `_id` is
        // `Decimal128(38, 0)`, ids are 0..n.
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            Field::new(
                "_id",
                DataType::Decimal128(
                    crate::supertable::options::DECIMAL128_PRECISION,
                    crate::supertable::options::DECIMAL128_SCALE,
                ),
                false,
            ),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = crate::superfile::builder::BuilderOptions::new(
            schema.clone(),
            "_id",
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        );
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let n = titles.len();
        let ids = arrow_array::Decimal128Array::from((0..n as i128).collect::<Vec<_>>())
            .with_precision_and_scale(
                crate::supertable::options::DECIMAL128_PRECISION,
                crate::supertable::options::DECIMAL128_SCALE,
            )
            .expect("decimal128");
        let titles_arr = LargeStringArray::from(titles.to_vec());
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles_arr)]).expect("batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        let bytes = bytes::Bytes::from(b.finish().expect("finish"));
        Arc::new(crate::superfile::SuperfileReader::open(bytes).expect("open"))
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
        let oracle_set: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();
        assert_eq!(oracle_set, [0u32, 4, 8].iter().copied().collect());

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_hits("title", "nimblefox", 5, BoolMode::Or)
            .expect("supertable query");
        assert_eq!(st_hits.len(), 3);
        // Resolve supertable hits to global doc-ids via superfile
        // ordering (superfiles appear in append order; chunk size = 4).
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
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
        let oracle_globals: std::collections::HashSet<u32> =
            oracle_hits.iter().map(|(d, _)| *d).collect();

        let st_reader = st.reader();
        let st_hits = st_reader
            .bm25_search_prefix("title", "rust", 5)
            .expect("supertable query");
        let manifest = st_reader.manifest();
        let st_globals: std::collections::HashSet<u32> = st_hits
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
}
