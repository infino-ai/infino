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
//!     table.bm25_search("title", "rust async", 10, Bm25SearchOptions::new(), None)?;
//!
//! // Materialize row data by naming the columns to decode.
//! let rows: Vec<RecordBatch> =
//!     table.bm25_search("title", "rust async", 10, Bm25SearchOptions::new(), Some(&["_id", "title", "score"]))?;
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
//! This is the classical sharded-BM25 problem: IDF computed from each
//! superfile's own `n_docs` and `df` makes a rare term in a small
//! superfile score higher than the same term in a larger one, so ranking
//! would drift as the table fragments. A term search therefore gathers the
//! corpus-wide document count and per-term document-frequencies once (a
//! dictionary-only df pass fused with the query's own reads, cached per
//! manifest generation) and scores every superfile against that single
//! table-wide IDF, so a fragmented table ranks like one unified corpus.
//! The count comes from the superfiles resident in the snapshot: on a
//! lazily loaded manifest with none resident it is zero, and each
//! superfile falls back to its own statistics, as prefix search always
//! does.
//!
//! Oracle tests assert that a fragmented table reproduces the
//! single-superfile ranking.
//!
//! ManifestSnapshot-level skip pruning is wired in: each call computes a
//! per-superfile keep/prune mask from the FTS bloom (exact-term
//! mode) or the lex term range (prefix mode) before issuing
//! per-superfile work, so pruned superfiles never trigger a
//! `SuperfileReaderCache::reader` call. Vector + SQL skip remain
//! deferred (see those modules' headers).

use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeSet, HashMap, HashSet},
    iter::repeat_n,
    slice,
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
    },
    time::Instant,
};

use arrow::record_batch::RecordBatch;
use arrow_array::{Array, LargeStringArray};
use futures::future::join;
use roaring::RoaringBitmap;
use tokio::sync::OnceCell;
use tracing::{Instrument, debug, warn};
use uuid::Uuid;

/// Fewest should-terms for which a ranged kernel is shipped to the
/// reader pool (oneshot bridge) instead of running inline on the tokio
/// worker. Multi-term kernels are multi-millisecond sync blocks — run
/// inline they starve woken tasks (one slice per query measured waiting
/// ~6 ms for a worker at 1M post-compaction). Below this many terms the
/// kernel is sub-millisecond and the bridge round-trip costs more than
/// it saves (`two_term_or` measured +~0.1 ms when bridged). Reuses
/// [`OR_WINDOW_MIN_TERMS`]: the same boundary below which the windowed
/// union kernel isn't worth its bookkeeping, so the two thresholds
/// cannot drift apart.
const RANGED_KERNEL_POOL_MIN_TERMS: usize = OR_WINDOW_MIN_TERMS;

/// Fewest summed term document frequencies for which the un-ranged
/// clause kernel runs on the reader pool instead of inline. Measured
/// masses split bimodal — cheap matches stay under 4,000, real scans
/// start past 60,000 — so this sits in the gap.
const UNRANGED_KERNEL_POOL_MIN_MASS: u64 = 20_000;

pub use crate::superfile::fts::reader::BoolMode;
#[cfg(feature = "detailed-tracing")]
use crate::utils::trace::OpOrigin;
use crate::{
    InfinoError, config,
    runtime_bridge::run_on_pool_unless_blocking,
    runtime_metrics::op_stats,
    superfile::{
        SuperfileReader,
        builder::FtsConfig,
        fts::{
            bm25,
            bm25::Bm25Params,
            reader::{
                Bm25SearchOptions, ClauseLists, ColumnLengthStats, FetchedTermMemo, GlobalTermIdf,
                LiveFloor, MatchWork, OR_WINDOW_MIN_TERMS, OrCursorSet, PreparedClauses, QueryIdf,
                ScoringView,
            },
            tokenize::Phrase,
        },
        id_space::RowId,
    },
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        manifest::{
            ManifestSnapshot, SuperfileEntry, SuperfileUri, encoding::FTS_SUMMARY_TOO_OLD_MSG,
            term_index,
        },
        query::{
            SuperfileHit,
            candidate::{CandidatePlan, CandidateScope, TermMemos},
            dispatch,
            exec::common::{
                hit_output_schema, resolve_hits_named, take_file_columns, take_rows_byte_source,
            },
            prune::{PruneLeaf, select_superfiles},
            tier,
        },
        reader_cache::disk::ForegroundQueryGuard,
        schema::FieldId,
        tombstones::SidecarCache,
    },
    utils::{
        terms::DictEntry,
        trace::{self, detail_span, tiered_span},
    },
};

/// Per-superfile open-wave fetches for one global-stats query, keyed by
/// superfile id — `None` when every scored term came from the idf cache
/// (or the query scores no term), in which case the walk wave fetches
/// for itself.
type PrefetchMemos = Option<Arc<HashMap<Uuid, Arc<FetchedTermMemo>>>>;

/// Cap on cached (column, term) global-idf entries. Past it the map is
/// cleared and repopulates from subsequent queries — an epoch reset
/// instead of LRU bookkeeping, sized far above any realistic distinct
/// scored-term working set.
const GLOBAL_IDF_CACHE_MAX_TERMS: usize = 65_536;

/// Process-lifetime cache of global BM25 idf per `(column, term)`,
/// valid for exactly one manifest generation.
///
/// Global idf is a pure function of the pinned snapshot (corpus-wide
/// `N` plus the term's summed `df`), so without a cache every query
/// re-runs the dictionary gather fan over
/// all unpruned superfiles — measured as a flat ~0.3–1.7 ms added to
/// every warm query at 10M docs / 256 superfiles. Caching per
/// generation makes only the first query for a term pay the fan.
///
/// One generation at a time: a commit publishes a higher
/// `manifest_id`, and the first query against the newer snapshot
/// resets the map. A reader still pinned to an older snapshot bypasses
/// the cache (never repopulates backwards), so mixed-generation
/// readers stay correct at the cost of gathering uncached.
pub(crate) struct GlobalIdfCache {
    state: RwLock<GlobalIdfCacheState>,
}

struct GlobalIdfCacheState {
    manifest_id: u64,
    idf: HashMap<(Box<str>, Box<str>), f32>,
}

impl Default for GlobalIdfCache {
    fn default() -> Self {
        Self {
            state: RwLock::new(GlobalIdfCacheState {
                manifest_id: 0,
                idf: HashMap::new(),
            }),
        }
    }
}

impl GlobalIdfCache {
    /// Cached idf per term under `manifest_id`, `None` per miss. Every
    /// slot is a miss when the cache tracks a different generation.
    fn get(&self, manifest_id: u64, column: &str, terms: &[String]) -> Vec<Option<f32>> {
        let state = self.state.read().expect("global idf cache lock");
        if state.manifest_id != manifest_id {
            return vec![None; terms.len()];
        }
        terms
            .iter()
            .map(|t| {
                state
                    .idf
                    .get(&(Box::from(column), Box::from(t.as_str())))
                    .copied()
            })
            .collect()
    }

    /// Record gathered idfs under `manifest_id`. A newer generation
    /// resets the map to it; an older one is dropped (a pinned
    /// old-snapshot reader must not clobber current-generation
    /// entries).
    fn insert(&self, manifest_id: u64, column: &str, entries: &[(&str, f32)]) {
        let mut state = self.state.write().expect("global idf cache lock");
        match manifest_id.cmp(&state.manifest_id) {
            Ordering::Less => return,
            Ordering::Greater => {
                state.manifest_id = manifest_id;
                state.idf.clear();
            }
            Ordering::Equal => {}
        }
        if state.idf.len() + entries.len() > GLOBAL_IDF_CACHE_MAX_TERMS {
            state.idf.clear();
        }
        for (term, idf) in entries {
            state
                .idf
                .insert((Box::from(column), Box::from(*term)), *idf);
        }
    }
}

/// An unranked query's match set: the terms and exact phrases every
/// (`And`) or any (`Or`) of which a doc must contain. Produced by
/// `parse_and_prune` from the clause model — the must side when any
/// must exists (shoulds have no scores to raise unranked), the bare
/// side under the default operator otherwise.
struct UnrankedMatchSet {
    terms: Vec<String>,
    phrases: Vec<Phrase<String>>,
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

/// One parsed unranked match, ready to run against any of its superfiles:
/// the terms and phrases on each side of the negation, the index
/// locations that let a superfile skip its dictionary, and the statistics
/// sink. Shared by the walks that produce hits and the one that produces
/// columns, so the two cannot drift apart in what they match.
struct MatchKernel {
    /// The matched column's field id, which finds the column under the
    /// name a file written before a rename labelled it with.
    column_field_id: Option<FieldId>,
    column: String,
    mode: BoolMode,
    /// Any phrase atom, matched or negated, takes the phrase-aware walk.
    phrase_involved: bool,
    has_negatives: bool,
    terms: Vec<String>,
    phrases: Vec<Phrase<String>>,
    negatives: Vec<String>,
    negative_phrases: Vec<Phrase<String>>,
    locations: IndexLocations,
    op_stats: Option<Arc<op_stats::OpStatsCollector>>,
}

impl MatchKernel {
    /// What every walk of superfile `suid` starts from: the matched column
    /// under the name `r` knows it by, the index's memo of the terms'
    /// postings there, and the terms.
    async fn file_inputs(
        &self,
        r: &SuperfileReader,
        suid: Uuid,
    ) -> (String, Option<Arc<FetchedTermMemo>>, Vec<&str>) {
        // A file written before a rename labels the column as it was then,
        // and its dictionary is keyed by that label; the id is what finds
        // the column in either file.
        let column = r
            .column_alias(self.column_field_id, &self.column)
            .to_owned();
        let memo = memo_from_locations(r, &self.locations, suid).await;
        let refs: Vec<&str> = self.terms.iter().map(String::as_str).collect();
        (column, memo, refs)
    }

    /// The rows of superfile `suid`, read through `r`, that the match
    /// keeps: the positive walk minus any row a negated atom reaches.
    async fn docs(&self, r: &SuperfileReader, suid: Uuid) -> Result<Vec<RowId>, QueryError> {
        let (column, memo, refs) = self.file_inputs(r, suid).await;
        // Any phrase atom (match or negated) takes the phrase-aware walk;
        // plain-token queries keep the optimized token_match path.
        let (docs, mut work) = match self.phrase_involved {
            true => {
                r.atoms_match_ids(&column, &refs, &self.phrases, self.mode)
                    .await?
            }
            false => {
                r.token_match_prefetched(&column, &refs, self.mode, memo.as_deref())
                    .await?
            }
        };
        // Drop any positive match that also carries a negated atom (union
        // of the negatives). The df / count fast paths can't express
        // exclusion, so negation forces a materialized walk over both sets.
        let docs = if self.has_negatives {
            let neg_refs: Vec<&str> = self.negatives.iter().map(String::as_str).collect();
            let (neg_docs, neg_work) = match self.negative_phrases.is_empty() {
                true => {
                    r.token_match_prefetched(&column, &neg_refs, BoolMode::Or, memo.as_deref())
                        .await?
                }
                false => {
                    r.atoms_match_ids(&column, &neg_refs, &self.negative_phrases, BoolMode::Or)
                        .await?
                }
            };
            work.merge(neg_work);
            let excluded: RoaringBitmap = neg_docs.into_iter().map(RowId::get).collect();
            docs.into_iter()
                .filter(|d| !excluded.contains(d.get()))
                .collect::<Vec<_>>()
        } else {
            docs
        };
        // One flush per superfile: positive + negation walks.
        self.flush(&work);
        Ok(docs)
    }

    /// Whether one superfile's walk may be cut into doc-id slices: a plain
    /// union of terms, the walk [`Self::slice`] runs one slice of. A
    /// phrase, a negation or a conjunction of several terms walks the
    /// whole superfile through [`Self::docs`].
    fn slices(&self) -> bool {
        !self.phrase_involved
            && !self.has_negatives
            && (matches!(self.mode, BoolMode::Or) || self.terms.len() == 1)
    }

    /// The cursors every slice of superfile `suid` walks, built once.
    async fn cursor_set(&self, r: &SuperfileReader, suid: Uuid) -> Result<OrCursorSet, QueryError> {
        let (column, memo, refs) = self.file_inputs(r, suid).await;
        let (set, work) = r.match_cursor_set(&column, &refs, memo.as_deref()).await?;
        self.flush(&work);
        Ok(set)
    }

    /// The rows in doc-id slice `[start, end)` of a superfile that `set`,
    /// its [`Self::cursor_set`], matches. A [`Self::slices`] kernel only.
    fn slice(
        &self,
        r: &SuperfileReader,
        set: &OrCursorSet,
        start: u32,
        end: u32,
    ) -> Result<Vec<RowId>, QueryError> {
        let (rows, work) = r.token_match_range_prebuilt(set, start, end)?;
        self.flush(&work);
        Ok(rows)
    }

    /// Adds a walk's work to the statement's statistics.
    fn flush(&self, work: &MatchWork) {
        if let Some(stats) = &self.op_stats {
            stats.add_fts_postings_bytes(work.postings_bytes);
            stats.add_planned_read_ranges(work.planned_ranges);
            stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
        }
    }
}

/// The presence leaf a match set prunes with: its bare terms plus every
/// phrase's members, under the strongest mode those atoms allow.
///
/// A phrase's members are conjunctive — a match contains all of them, adjacent
/// — but that only constrains the whole query when nothing else could satisfy
/// it. So the mode is `And` when the caller already requires every atom (a
/// must-side prune), and also for the single quoted phrase with no bare terms,
/// where matching the query *is* matching the phrase. A disjunction of a phrase
/// with anything else is `(a AND b) OR c`, which one presence leaf cannot
/// express, so it falls back to the union over every atom — weaker, never
/// wrong.
///
/// Flattening the members into the query's mode unconditionally is what makes a
/// quoted phrase of common words prune nothing: `"wine beer"` asks only whether
/// a superfile holds `wine` or `beer`, which at corpus scale every superfile
/// does.
fn presence_leaf(
    column: &str,
    terms: &[String],
    phrases: &[Phrase<String>],
    mode: BoolMode,
) -> PruneLeaf {
    let mut atoms: Vec<String> = terms.to_vec();
    for p in phrases {
        atoms.extend(p.iter().cloned());
    }
    let conjunctive = matches!(mode, BoolMode::And) || (terms.is_empty() && phrases.len() == 1);
    PruneLeaf::TermPresence {
        column: column.to_owned(),
        terms: atoms,
        mode: match conjunctive {
            true => BoolMode::And,
            false => mode,
        },
    }
}

/// An unranked query's negated atoms (docs containing any are
/// excluded).
#[derive(Default)]
struct UnrankedNegatives {
    terms: Vec<String>,
    phrases: Vec<Phrase<String>>,
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

/// Message for a bm25 / token query naming a column that carries no
/// full-text index. Names the requested column and the searchable set so the
/// caller can correct the request, rather than failing deep in the scan with
/// an opaque "missing full-text section" error once a candidate superfile is
/// opened.
fn no_fts_index_message(column: &str, fts_columns: &[FtsConfig]) -> String {
    if fts_columns.is_empty() {
        return format!(
            "no full-text index for column {column:?}: this table has no \
             full-text-indexed columns"
        );
    }
    let available = fts_columns
        .iter()
        .map(|c| c.column.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!("no full-text index for column {column:?}; full-text-indexed columns: {available}")
}

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
    /// The best `k` scores seen so far, descending, each with the largest
    /// `_id` its superfile holds. A finished unit's scores merge in by one
    /// linear pass, so a unit costs its own length plus `k` whatever the
    /// heap of the moment holds — not a heap operation per score, which
    /// the first wave of units, all finishing against an empty heap under
    /// one lock, paid in full.
    top: Mutex<Vec<FloorEntry>>,
    /// The current floor — `NEG_INFINITY` until `k` scores have been
    /// seen, monotone after. Shared with the atom walks as their live
    /// mid-walk bar (see [`LiveFloor`]), so it rises both from finished
    /// segments' merges and from running walks' local kth-bests.
    floor: LiveFloor,
}

/// One score in the [`SharedTopK`] heap with the largest `_id` of the
/// superfile it came from: what a tie at the floor is compared against.
/// Ordered by score, then by `_id` range *descending*, so of two equal
/// scores the one from the higher range is the lesser and leaves the
/// heap first: the stable top-k keeps the lower `_id` of a tie, and so
/// does the heap — a seed placeholder (`i128::MAX`) in particular gives
/// way to a real score equal to it. BM25 scores are finite, but `f32`
/// still needs an `Ord` shim.
#[derive(PartialEq, Clone, Copy)]
struct FloorEntry {
    score: f32,
    id_max: i128,
}
impl Eq for FloorEntry {}
impl PartialOrd for FloorEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FloorEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.id_max.cmp(&self.id_max))
    }
}

impl SharedTopK {
    fn new(k: usize) -> Arc<Self> {
        Arc::new(Self {
            k,
            top: Mutex::new(Vec::new()),
            floor: LiveFloor::new(),
        })
    }

    /// The current global floor — `NEG_INFINITY` until k scores merged.
    fn floor(&self) -> f32 {
        self.floor.load()
    }

    /// The floor as the live handle the atom walks read and raise
    /// mid-walk.
    fn live_floor(&self) -> &LiveFloor {
        &self.floor
    }

    /// Merge one finished segment's (tombstone-surviving) scores, from a
    /// superfile whose `_id`s reach `id_max`, and publish the new kth-best
    /// as the floor once k scores are known.
    ///
    /// A unit's scores are first held against the live floor — the floor
    /// only rises, so a score below it can never enter — which leaves
    /// nothing for most units once `k` scores are known. What remains
    /// enters the top: a few entries by binary search and insertion, a
    /// first-wave unit's many by one sorted linear merge. A unit's hits
    /// arrive in document order, so nothing here relies on theirs. The
    /// ceiling-ordered fan-out calls this from the one task that drives
    /// it, as each unit finishes; only the unordered fan-out's units call
    /// it themselves, and those all run at once.
    fn merge(&self, id_max: i128, scores: impl IntoIterator<Item = f32>) {
        /// Entries up to which insertion beats a sort and a merge.
        const FEW: usize = 32;
        let floor = self.floor.load();
        let mut entering: Vec<FloorEntry> = scores
            .into_iter()
            .filter(|&score| score >= floor)
            .map(|score| FloorEntry { score, id_max })
            .collect();
        if entering.is_empty() {
            return;
        }
        if entering.len() > FEW {
            entering.sort_unstable_by(|a, b| b.cmp(a));
        }
        let k = self.k;
        let mut top = self.top.lock().expect("SharedTopK mutex poisoned");
        if top.len() >= k {
            let min = top[k - 1];
            entering.retain(|e| *e > min);
            if entering.is_empty() {
                return;
            }
        }
        if entering.len() <= FEW {
            for e in entering {
                let at = top.partition_point(|t| *t >= e);
                if at < k {
                    top.insert(at, e);
                    top.truncate(k);
                }
            }
        } else {
            let mut merged: Vec<FloorEntry> =
                Vec::with_capacity((top.len() + entering.len()).min(k));
            let (mut i, mut j) = (0, 0);
            while merged.len() < k && (i < top.len() || j < entering.len()) {
                let take_top = match (top.get(i), entering.get(j)) {
                    (Some(a), Some(b)) => a >= b,
                    (Some(_), None) => true,
                    _ => false,
                };
                if take_top {
                    merged.push(top[i]);
                    i += 1;
                } else {
                    merged.push(entering[j]);
                    j += 1;
                }
            }
            *top = merged;
        }
        if top.len() == k {
            // `raise` rather than a plain store: running walks may have
            // published a higher local kth than this merge's k-th, and the
            // floor must never move down.
            self.floor.raise(top[k - 1].score);
        }
    }

    /// Whether a superfile whose `_id`s start at `id_min` and whose
    /// ceiling equals `floor` can be skipped: `k` scores are known, and
    /// every one at the floor came from a superfile whose `_id`s all lie
    /// below `id_min`. The final top-k breaks a tie at the k-th score by
    /// ascending `_id`, so a document of this superfile tied at the floor
    /// ranks behind each of those, and one below it is out regardless.
    /// Sound for any `_id` ranges: every `_id` of this superfile is at
    /// least `id_min`, and every tied document's is at most its
    /// superfile's `id_max`. A seed placeholder carries `i128::MAX` and so
    /// never justifies a skip.
    ///
    /// Of the entries at the floor the last is the one from the highest
    /// range (equal scores order by range descending), so one look settles
    /// it. A floor the live walks have raised above the k-th entry has no
    /// entry to compare against here and keeps the superfile open.
    fn tie_skippable(&self, floor: f32, id_min: i128) -> bool {
        let top = self.top.lock().expect("SharedTopK mutex poisoned");
        top.len() >= self.k && {
            let last = top[self.k - 1];
            last.score == floor && last.id_max < id_min
        }
    }
}

/// Where each kept superfile's postings for a query's terms sit, per the
/// term index: superfile → `(term, df, location)` for the terms it holds.
/// `terms` are the ones asked about, so a term missing from an indexed
/// superfile's entry is known absent there. `by_superfile` is empty when
/// the table has no index, or it could not answer, in which case every
/// cursor build reads its superfile's dictionary as before.
pub(crate) struct LocatedTerms {
    terms: Vec<String>,
    by_superfile: HashMap<Uuid, Arc<Vec<(String, u64, term_index::Location)>>>,
}

/// Shared handle to one query's [`LocatedTerms`].
pub(crate) type IndexLocations = Arc<LocatedTerms>;

/// The term index's locations for every exact-match term a candidate plan
/// resolves, per column, over the superfiles `kept`; see
/// [`IndexLocations`]. Empty when the table has no index.
pub(crate) type PlanLocations = HashMap<String, IndexLocations>;

/// The term index's postings locations for `terms` in every superfile of
/// `kept` that `manifest`'s index lists; see [`IndexLocations`]. Empty when
/// the table has no index; an index that cannot be read is an error.
pub(crate) async fn index_locations_for(
    manifest: &ManifestSnapshot,
    column: &str,
    terms: &[&str],
    kept: &[Arc<SuperfileEntry>],
) -> Result<IndexLocations, QueryError> {
    let owned: Vec<String> = terms.iter().map(|t| (*t).to_owned()).collect();
    let by_superfile = match (manifest.field_id(column), manifest.term_index().await?) {
        (Some(column), Some(index)) => index
            .locations(column, terms, kept)
            .await?
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect(),
        _ => HashMap::new(),
    };
    Ok(Arc::new(LocatedTerms {
        terms: owned,
        by_superfile,
    }))
}

/// [`index_locations_for`] for every exact-match term of `plan`, per
/// column; see [`PlanLocations`].
pub(crate) async fn plan_locations_for(
    manifest: &ManifestSnapshot,
    plan: &CandidatePlan,
    kept: &[Arc<SuperfileEntry>],
) -> Result<PlanLocations, QueryError> {
    let mut out = PlanLocations::new();
    for (column, terms) in plan.term_requests() {
        let refs: Vec<&str> = terms.iter().map(String::as_str).collect();
        let located = index_locations_for(manifest, &column, &refs, kept).await?;
        out.insert(column, located);
    }
    Ok(out)
}

/// Build a plan's per-column memos for `superfile` from [`PlanLocations`]:
/// one memo per column the index located terms in.
pub(crate) async fn memos_from_plan_locations(
    r: &SuperfileReader,
    locations: &PlanLocations,
    superfile: Uuid,
) -> TermMemos {
    let mut memos = TermMemos::new();
    for (column, locations) in locations {
        if let Some(memo) = memo_from_locations(r, locations, superfile).await {
            memos.insert(column.clone(), memo);
        }
    }
    memos
}

/// A prefetched-term memo for `superfile` built from the index's
/// locations: the postings ranges are fetched, the dictionary is not, and
/// a term the index lists no posting for in this superfile is recorded as
/// a resolved miss, so its absence costs no dictionary read either. A term
/// whose location the index chose not to carry stays out of the memo and
/// resolves through the dictionary. `None` when the index does not list
/// this superfile or the fetch failed — the cursor build then reads the
/// dictionary; the cost is a read, never the answer.
pub(crate) async fn memo_from_locations(
    r: &SuperfileReader,
    locations: &IndexLocations,
    superfile: Uuid,
) -> Option<Arc<FetchedTermMemo>> {
    let located = locations.by_superfile.get(&superfile)?;
    let pairs: Vec<(&str, u64, DictEntry)> = located
        .iter()
        .filter_map(|(t, df, loc)| loc.to_dict_value().map(|v| (t.as_str(), *df, v)))
        .collect();
    let mut memo = r.term_memo_from_dict_values(&pairs).await.ok()?;
    let present: HashSet<&str> = located.iter().map(|(t, _, _)| t.as_str()).collect();
    for term in &locations.terms {
        if !present.contains(term.as_str()) {
            memo.note_absent(term);
        }
    }
    Some(Arc::new(memo))
}

/// What a ranked fan-out unit carries to its kernel and its skip test.
#[derive(Clone)]
struct UnitParams {
    /// The doc-id sub-range the unit scores, `None` for the whole superfile.
    range: Option<(u32, u32)>,
    superfile_id: Uuid,
    uri: SuperfileUri,
    /// The highest score the unit can reach, from the term index.
    ceiling: f32,
    /// The superfile's `_id` range.
    id_min: i128,
    id_max: i128,
}

/// Whether a superfile whose best possible score is `ceiling` can still
/// place a document once the running k-th score is `floor`. Only a ceiling
/// strictly below the floor cannot: a ceiling equal to it may hold a
/// document that ties the k-th, and the stable `_id` order may admit that
/// tie, so an equal ceiling is opened. This comparison handles an exact tie
/// on its own; a ceiling that rounding put an ulp below a real score is
/// protected by the widening the term index applies to every ceiling
/// (`CEILING_SLACK`), not by this test.
pub(crate) fn ceiling_can_compete(ceiling: f32, floor: f32) -> bool {
    // Written as "not strictly less" rather than `>=` so an incomparable
    // ceiling (a NaN from a degenerate rescale) is opened, never skipped.
    ceiling.partial_cmp(&floor) != Some(std::cmp::Ordering::Less)
}

/// The score a superfile's ceiling is actually tested against: the running
/// k-th score, raised by the caller's tolerance.
///
/// At the default tolerance of zero this returns `floor` bit for bit, because
/// multiplying by `1.0` is exact, so exact scoring keeps precisely the
/// comparison it had before tolerance existed. A negative or non-finite
/// tolerance is treated as zero rather than rejected: the worst such a value
/// could do is lower the bar and open superfiles that need not be opened,
/// which is slow and still correct, but defining it away is cheaper than
/// reasoning about it at every call site.
pub(crate) fn competitive_floor(floor: f32, tolerance: f32) -> f32 {
    match tolerance.is_finite() && tolerance > 0.0 {
        true => floor * (1.0 + tolerance),
        false => floor,
    }
}

/// What a superfile scores a query at: the query's BM25 parameters, and
/// for `column`, as the file labels it, the average length `avgdl` when it
/// is scored at the table's.
fn table_scoring(column: &str, params: Option<Bm25Params>, avgdl: Option<f32>) -> ScoringView<'_> {
    ScoringView {
        params,
        avgdl: avgdl.map(|avgdl| (column, avgdl)),
    }
}

/// The parameters `column` declares, which every superfile scores it at
/// unless a query overrides them.
fn column_params(manifest: &ManifestSnapshot, column: &str) -> Option<Bm25Params> {
    manifest
        .fts_configs()
        .iter()
        .find(|c| c.column == column)
        .map(|c| c.bm25)
}

/// The average length `entry`'s `column` was built at, as its manifest
/// summary records it. A file whose summary records none is scored at
/// that average rather than the table's: its stored bounds and term-index
/// ceilings only hold there, and without the value nothing can rescale
/// them before the file is opened.
pub(super) fn baked_avgdl(entry: &SuperfileEntry, column: FieldId) -> Option<f32> {
    entry
        .fts_summary
        .get(&column)
        .and_then(|summary| summary.baked_avgdl)
}

/// Scales each superfile's score ceiling, stored at the average length it
/// was built at, so it stays an upper bound at the table's average
/// `table_avgdl`. A file built at a smaller average scores at most
/// `table_avgdl / baked` higher there: the length norm
/// `1 - b + b·dl/avgdl` falls by at most that ratio, and `tf / (tf + k1·norm)`
/// rises by no more than the norm falls. A file that records no average is
/// scored at the average it was built at, where its ceiling already holds.
fn ceilings_at_table_avgdl(
    ceilings: &mut HashMap<Uuid, f32>,
    kept: &[Arc<SuperfileEntry>],
    column: FieldId,
    table_avgdl: f32,
) {
    for entry in kept {
        let Some(ceiling) = ceilings.get_mut(&entry.superfile_id) else {
            continue;
        };
        if let Some(baked) = baked_avgdl(entry, column)
            && baked > 0.0
            && baked < table_avgdl
        {
            // Rounded up so the division's rounding never leaves the
            // ceiling below a score it has to cap.
            *ceiling = (*ceiling * (table_avgdl / baked)).next_up();
        }
    }
}

/// A ranked query's clauses, owned and resolved to terms: what the scoring
/// core takes, whether they came from parsing a query or expanding a
/// prefix.
#[derive(Default)]
struct ScoredClauses {
    musts: Vec<String>,
    shoulds: Vec<String>,
    negatives: Vec<String>,
    must_phrases: Vec<Phrase<String>>,
    should_phrases: Vec<Phrase<String>>,
    negative_phrases: Vec<Phrase<String>>,
}

impl SupertableReader {
    /// [`plan_locations_for`] on this reader's manifest.
    pub(crate) async fn plan_locations(
        &self,
        plan: &CandidatePlan,
        kept: &[Arc<SuperfileEntry>],
    ) -> Result<PlanLocations, QueryError> {
        plan_locations_for(self.manifest(), plan, kept).await
    }

    /// [`index_locations_for`] on this reader's manifest.
    pub(crate) async fn index_locations(
        &self,
        column: &str,
        terms: &[&str],
        kept: &[Arc<SuperfileEntry>],
    ) -> Result<IndexLocations, QueryError> {
        index_locations_for(self.manifest(), column, terms, kept).await
    }

    /// Single-column BM25 search across the pinned manifest's
    /// superfiles. Returns up to `k` highest-scoring hits, sorted
    /// descending by score.
    ///
    /// `query` is tokenized by the same tokenizer the column was
    /// indexed with (its per-column analyzer). Returns
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
    /// Reject an out-of-range query-time override before the fan-out
    /// starts. A declared pair is validated at `create_table`; this is
    /// the same check for the per-search form, so a caller sees the
    /// bounds rather than a silently strange ranking.
    fn validate_bm25_params(p: Bm25Params) -> Result<(), QueryError> {
        let (k1, b) = (p.k1, p.b);
        match k1.is_finite() && k1 > 0.0 && b.is_finite() && (0.0..=1.0).contains(&b) {
            true => Ok(()),
            false => Err(QueryError::InvalidQuery(format!(
                "bm25 k1 must be finite and > 0, b must be finite and in [0, 1]; \
                 got k1={k1}, b={b}"
            ))),
        }
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(column = column, k = k, mode = ?opts.mode, role = self.role().as_str(), origin = OpOrigin::Query.as_str()))
    )]
    pub(crate) async fn bm25_search_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.bm25_search_scoped_async(column, query, k, opts, None)
            .await
    }

    /// [`Self::bm25_search_async`] confined to a [`CandidateScope`] — what a
    /// SQL `WHERE` pushed into the `bm25_search` / `hybrid_search` table
    /// functions admits. Only the scope's superfiles are searched, and
    /// under a bounded scope each superfile's kernel admits only its
    /// candidate rows into the top-k heap, so the k hits are the k best
    /// *among rows satisfying the predicate*, not the predicate applied to
    /// a global top-k. `None` is the unscoped search.
    pub(crate) async fn bm25_search_scoped_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        scope: Option<&CandidateScope>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let (hits, _) = self
            .bm25_scored_async(column, query, k, opts, scope, false)
            .await?;
        Ok(hits)
    }

    /// The shared body of the ranked searches. With `exact_count` every
    /// match is scored in one pass and the exact match count comes back
    /// with the hits; otherwise the kernels prune and the count is 0.
    async fn bm25_scored_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        scope: Option<&CandidateScope>,
        exact_count: bool,
    ) -> Result<(Vec<SuperfileHit>, u64), QueryError> {
        if k == 0 {
            return Ok((Vec::new(), 0));
        }
        // Destructured once here: `mode` shapes the clause split, and
        // `bm25` — when set — overrides what each column declared, which
        // every per-superfile reader below has to apply identically or two
        // superfiles would score one query two ways.
        let Bm25SearchOptions {
            mode,
            bm25: bm25_params,
            score_tolerance,
        } = opts;
        if let Some(p) = bm25_params {
            Self::validate_bm25_params(p)?;
        }
        let manifest = self.manifest();
        let column_owned = column.to_owned();

        // Resolve the query tokenizer, which doubles as the column's
        // full-text-index check: a `None` here means `column` carries no
        // full-text index, so every candidate superfile would lack the
        // full-text section this scan reads and the low-level reader would
        // fail deep in the scan with an opaque missing-metadata error. Reject
        // up front instead, naming the column and the searchable set.
        let Some(tokenizer) = manifest.try_fts_tokenizer_for(column) else {
            return Err(QueryError::InvalidQuery(no_fts_index_message(
                column,
                &manifest.fts_configs(),
            )));
        };
        // Parse the query once here, not per superfile, resolving the
        // bare tokens' polarity from the default operator (`And` ⇒
        // must, `Or` ⇒ should). The fan-out closures below need owned
        // ('static) data for tokio::spawn, so this is the one place
        // the tokens are copied — the prune and every per-superfile
        // search reuse them.
        let clauses = tokenizer.parse(query).into_clauses(mode);
        let musts: Vec<String> = clauses.musts.into_iter().map(Cow::into_owned).collect();
        let shoulds: Vec<String> = clauses.shoulds.into_iter().map(Cow::into_owned).collect();
        let negatives: Vec<String> = clauses.negatives.into_iter().map(Cow::into_owned).collect();
        // `Phrase::map` keeps each term's offset, which is what a
        // phrase on a stopworded column needs: its terms were not
        // adjacent in the query and must not be required adjacent here.
        let own_phrases = |phrases: Vec<Phrase<Cow<'_, str>>>| -> Vec<Phrase<String>> {
            phrases.iter().map(|p| p.map(|t| t.to_string())).collect()
        };
        let must_phrases = own_phrases(clauses.must_phrases);
        let should_phrases = own_phrases(clauses.should_phrases);
        let negative_phrases = own_phrases(clauses.negative_phrases);
        let has_musts = !musts.is_empty() || !must_phrases.is_empty();

        if !has_musts && shoulds.is_empty() && should_phrases.is_empty() {
            // No scorable clause at all. Empty / punctuation-only
            // queries match nothing (not an error); negation-only
            // (e.g. `-foo`) has no anchor to rank — reject up front so
            // the per-superfile kernel never has to, and so the
            // unranked count / token_match path surfaces the identical
            // error (see `parse_and_prune`).
            if negatives.is_empty() && negative_phrases.is_empty() {
                return Ok((Vec::new(), 0));
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
        let prune_leaf = match has_musts {
            true => presence_leaf(&column_owned, &musts, &must_phrases, BoolMode::And),
            false => presence_leaf(&column_owned, &shoulds, &should_phrases, mode),
        };
        let kept = self.select_scored(column, &prune_leaf, scope).await?;
        self.bm25_search_clauses(
            column,
            ScoredClauses {
                musts,
                shoulds,
                negatives,
                must_phrases,
                should_phrases,
                negative_phrases,
            },
            kept,
            k,
            bm25_params,
            score_tolerance,
            scope,
            exact_count,
        )
        .await
    }

    /// The superfiles `leaf` keeps for `column`. A pushed-down `WHERE`
    /// narrows them to the superfiles its `scope` admits, the statistics
    /// survivors that still hold a candidate row. The global-idf gather
    /// still probes every superfile the terms may live in, so idf stays
    /// table-wide and a scoped query ranks on the same scale as an
    /// unscoped one.
    async fn select_scored(
        &self,
        column: &str,
        leaf: &PruneLeaf,
        scope: Option<&CandidateScope>,
    ) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
        let manifest = self.manifest();
        let select_span = trace::phase(self.phase_spans(), || {
            detail_span!(
                "fts.select_superfiles",
                manifest_superfiles = manifest.superfiles.len(),
                survivors = tracing::field::Empty,
            )
        });
        let mut kept = select_fts_superfiles(manifest.as_ref(), slice::from_ref(leaf), column)
            .instrument(select_span.clone())
            .await?;
        if let Some(scope) = scope {
            let admitted: HashSet<Uuid> = scope
                .admitted_superfiles()
                .map(|e| e.superfile_id)
                .collect();
            kept.retain(|e| admitted.contains(&e.superfile_id));
        }
        select_span.record("survivors", kept.len());
        trace::end(select_span);
        Ok(kept)
    }

    /// Ranks `clauses` over the superfiles in `kept`, with idf from the
    /// whole table: the shared core of term and prefix search. With
    /// `exact_count` nothing is pruned and no superfile is skipped, and
    /// the exact match count (deleted rows excluded) comes back with the
    /// hits; otherwise the count is 0.
    async fn bm25_search_clauses(
        &self,
        column: &str,
        clauses: ScoredClauses,
        mut kept: Vec<Arc<SuperfileEntry>>,
        k: usize,
        bm25_params: Option<Bm25Params>,
        score_tolerance: f32,
        scope: Option<&CandidateScope>,
        exact_count: bool,
    ) -> Result<(Vec<SuperfileHit>, u64), QueryError> {
        let ScoredClauses {
            musts,
            shoulds,
            negatives,
            must_phrases,
            should_phrases,
            negative_phrases,
        } = clauses;
        let has_phrases =
            !must_phrases.is_empty() || !should_phrases.is_empty() || !negative_phrases.is_empty();
        let manifest = self.manifest();
        let pool_threads = manifest.options.reader_pool.current_num_threads();
        let column_owned = column.to_owned();
        // Idf is computed against the table-wide totals.
        manifest
            .derive_length_totals()
            .await
            .map_err(QueryError::ManifestLoad)?;
        let corpus = manifest
            .fts_length_stats(column)
            .ok_or_else(|| QueryError::Unsupported(FTS_SUMMARY_TOO_OLD_MSG.to_owned()))?;
        // No document in the table has a token in this column, so nothing
        // can match and no term has an idf to score with.
        if corpus.n_scored_docs == 0 {
            return Ok((Vec::new(), 0));
        }
        // Every superfile scores at the table's average length, in the
        // fixed point a file stores it in, so a file built at exactly this
        // average takes the declared path.
        let table_avgdl = bm25::stored_avgdl(corpus.avgdl());

        let phases = self.phase_spans();
        if kept.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Corpus-wide idf per scored term comes from an
        // OPEN WAVE fused with the query's own reads: each kept superfile
        // fetches its scored terms' dictionary slots and postings ranges —
        // exactly the reads its walk needs, handed back to it via a memo —
        // and reports df; superfiles that may contain a scored term but were
        // pruned from scoring (the AND-shape residual) contribute df through
        // a dict-only probe in the same wave. Repeat terms skip the wave
        // entirely via the per-generation idf cache. The scored set is every
        // term that contributes to a score: the bare musts + shoulds, plus
        // each member of a scored (must/should) phrase — a phrase's score
        // is Σ member idf. Negated terms/phrases are pure exclusions, so
        // their idf never matters and they stay out of the wave.
        let scored: Vec<String> = {
            let mut seen: HashSet<&str> = HashSet::new();
            musts
                .iter()
                .chain(shoulds.iter())
                .chain(
                    must_phrases
                        .iter()
                        .chain(should_phrases.iter())
                        .flat_map(|phrase| phrase.iter()),
                )
                .filter(|t| seen.insert(t.as_str()))
                .cloned()
                .collect()
        };
        let (global_idf, prefetch_memos) = self
            .global_idf_open_wave(manifest.as_ref(), column, &scored, &kept, corpus)
            .instrument(trace::phase(phases, || {
                tiered_span!("fts.global_idf", terms = scored.len())
            }))
            .await?;
        let global_idf = Arc::new(global_idf);

        // Build the work-unit list. When the reader pool has more
        // threads than there are kept superfiles AND we're on the
        // multi-term OR hot path, slice each superfile into doc_id
        // sub-ranges so the fan-out can saturate every pool thread.
        // Single-term OR, AND, and any query with a must or negated
        // clause stay on the un-ranged call.
        // Bound-ordered opening. With the term index present and no scoring
        // override, order superfiles by the highest score this query can
        // reach in each, so the first opens raise the shared floor and any
        // later superfile that cannot beat it is never opened. An override
        // changes the parameters the stored ceilings were baked at; until the
        // rescale from the manifest's length stats exists, such a query keeps
        // the unordered path — correct, just unpruned.
        // One span over the three awaits that read the term index (its load,
        // the score ceilings, the postings locations), with the synchronous
        // work between them.
        let term_span = trace::phase(phases, || {
            detail_span!("fts.term_index", terms = tracing::field::Empty)
        });
        let term_index = manifest.term_index().instrument(term_span.clone()).await?;
        // Every scored term, phrase members included, once: what the index
        // is asked for locations.
        let mut all_terms: Vec<&str> = musts
            .iter()
            .chain(shoulds.iter())
            .map(String::as_str)
            .collect();
        for p in must_phrases.iter().chain(should_phrases.iter()) {
            all_terms.extend(p.iter().map(String::as_str));
        }
        all_terms.sort_unstable();
        all_terms.dedup();
        // The index serves a ranked query when every scored term's idf is
        // table-wide and no scoring override is in force: ceilings and
        // tiers hold impacts at the declared parameters.
        // An exact count opens every superfile, so it skips the index read.
        let index_column = match (&term_index, bm25_params, manifest.field_id(column)) {
            (Some(index), None, Some(column_id))
                if !exact_count && all_terms.iter().all(|t| global_idf.contains_key(*t)) =>
            {
                Some((index, column_id))
            }
            _ => None,
        };
        let mut ceilings: Option<HashMap<Uuid, f32>> = None;
        // Tier selection: with every scored term's tier readable, the query
        // chooses its superfiles from the documents the tiers name — per
        // superfile a bound that knows which documents carry several of the
        // terms, and a superfile a required term has shown holds no
        // candidate is not opened. The bounds tighten the ceilings, and the
        // merge's lower bound on the k-th score is seeded into the shared
        // floor. A phrase is bounded by its members and a negation by its
        // positive terms, but neither yields a threshold: such a query is
        // bounded and ordered, and seeds nothing. A query of more terms
        // than the merge tracks keeps the ceilings. Only a complete index
        // can rule a superfile out: one that does not yet list every live
        // superfile knows nothing about the rest.
        // Rows deleted from each kept superfile: the tiers still hold them,
        // so the merge reads past them, and a kernel admits only the live
        // rows, so its k best are live. A sidecar that cannot be read
        // leaves the query to its ceilings and the kernel's own filter.
        // Whether any kept superfile has deleted rows, from the sidecar
        // cache's view of the manifest and without reading a sidecar: a
        // unit whose superfile has them builds its live-row set from the
        // kept set's row counts.
        let tombstoned = self
            .tombstone_cache
            .as_ref()
            .is_some_and(|c| kept.iter().any(|e| c.may_have_tombstones(e.superfile_id)));
        // A scope that admits only some rows of a superfile is one the
        // index cannot see, and the merge's threshold is the k-th score
        // over every row: seeding the floor with it, or ruling a superfile
        // out under it, would prune the admitted rows that score below it.
        // Such a search keeps its ceilings; a scope that only chooses
        // superfiles leaves the merge sound, since it works over the kept
        // set.
        // A table of one superfile has nothing for a merge to rule out or
        // order, and its kernel seeds its own floor from its own blocks; the
        // merge would only cost its reads and rounds on every query, and the
        // gate its samples learning so.
        let tiers_serve = index_column.is_some()
            && kept.len() > 1
            && scope.is_none_or(|s| !s.bounds_rows(&kept))
            && all_terms.len() <= tier::MAX_TIER_TERMS
            && manifest.term_index_complete()
            && !config::global().diagnostics.disable_tier_selection;
        // Whether to merge at all is the table's learned call per class of
        // query; the search is timed from here so the call can be judged.
        let rarest_idf = all_terms
            .iter()
            .filter_map(|t| global_idf.get(*t))
            .fold(0.0f32, |m, &idf| m.max(idf));
        let policy_class = tier::TierPolicy::class(
            shoulds.is_empty() && should_phrases.is_empty(),
            musts.is_empty() && must_phrases.is_empty(),
            k,
            all_terms.len(),
            rarest_idf,
        );
        let ranked_started = Instant::now();
        let learned = manifest.options.tier_merge_gate;
        let merge_now = tiers_serve && (!learned || self.tier_policy().merge(policy_class));
        // The deny sets, resolved only for a merge: it reads past the
        // deleted rows, and the units then admit rows by the same sets, so
        // a delete committed between the two reads cannot leave the
        // threshold counting a row a kernel drops. Without a merge each
        // unit looks its own up, as before. A sidecar that cannot be read
        // leaves the merge unprepared and every unit to its own lookup.
        let mut deny: Vec<Option<Arc<RoaringBitmap>>> = vec![None; kept.len()];
        let mut deny_known = true;
        if merge_now && let Some(cache) = &self.tombstone_cache {
            let now = Instant::now();
            let ids: Vec<Uuid> = kept.iter().map(|e| e.superfile_id).collect();
            cache.prefetch(&ids, now).await;
            for (slot, entry) in deny.iter_mut().zip(kept.iter()) {
                match dispatch::tombstone_deny_set(cache, entry.superfile_id, now) {
                    Ok(set) => *slot = set,
                    Err(_) => {
                        deny_known = false;
                        break;
                    }
                }
            }
        }
        let deny_by_superfile: Arc<HashMap<Uuid, Arc<RoaringBitmap>>> =
            Arc::new(match merge_now && deny_known {
                true => kept
                    .iter()
                    .zip(&deny)
                    .map(|(e, d)| (e.superfile_id, d.clone().unwrap_or_default()))
                    .collect(),
                false => HashMap::new(),
            });
        // The merge, when it runs, runs beside the first opens rather than
        // before them: its tier reads are a round trip of their own on a
        // cold store, and spent ahead of the opens they would cost what
        // they save. The fan-out starts in ceiling order; the units not
        // yet issued when the merge reports are re-planned by what it
        // found — answered from the index, ruled out, or reordered by
        // their tighter bounds — and its threshold is seeded into the
        // floor. What the merge needs is gathered here and it is started
        // at the fan-out; `selected` records that it reported in time.
        let selected = AtomicBool::new(false);
        // Whether the merge, having reported, did anything the opens
        // would not have: ruled a superfile out, answered for one, or
        // seeded a threshold the kernels prune against. What the policy
        // learns from, timing aside.
        let pruned = AtomicBool::new(false);
        // Whether the merge declined before any read — a term with
        // postings in a layer without a tier, a superfile the root does
        // not list: the class learns there is nothing to merge here.
        let declined = AtomicBool::new(false);
        let merge_terms: Vec<String> = all_terms.iter().map(|t| (*t).to_owned()).collect();
        let merge_term_refs: Vec<&str> = merge_terms.iter().map(String::as_str).collect();
        let merge_kept: Vec<Arc<SuperfileEntry>> = match merge_now {
            true => kept.clone(),
            false => Vec::new(),
        };
        let mut tier_query: Option<tier::TierQuery<'_>> = None;
        if let Some((_, column_id)) = index_column
            && merge_now
        {
            // Bits over the distinct terms: the bare terms score on their
            // own; a match must carry the musts and every member of a
            // must-phrase.
            let index_of = |n: &String| merge_term_refs.iter().position(|t| *t == n.as_str());
            let bits = |names: &[String]| -> tier::TermMask {
                names
                    .iter()
                    .filter_map(index_of)
                    .fold(0, |m, i| m | (1 << i))
            };
            let scoring = bits(&musts) | bits(&shoulds);
            // A phrase's members in order, a repeated word repeated: the
            // kernel weights every occurrence.
            let members = |p: &[String]| -> Vec<usize> { p.iter().filter_map(index_of).collect() };
            let tier_phrases: Vec<(Vec<usize>, bool)> = must_phrases
                .iter()
                .map(|p| (members(p), true))
                .chain(should_phrases.iter().map(|p| (members(p), false)))
                .collect();
            let required = bits(&musts)
                | tier_phrases
                    .iter()
                    .filter(|p| p.1)
                    .flat_map(|p| p.0.iter())
                    .fold(0, |m, &i| m | (1 << i));
            // A bare term repeated in the query is weighted by its count in
            // the kernels; the tiers score each term once, so such a query
            // is not served.
            let repeated = musts.len() + shoulds.len() != scoring.count_ones() as usize;
            if let Some(params) = column_params(manifest, column)
                && !repeated
                && deny_known
            {
                tier_query = Some(tier::TierQuery {
                    column: column_id,
                    terms: &merge_term_refs,
                    required,
                    scoring,
                    phrases: tier_phrases,
                    // A negated phrase excludes as a negated term does: the
                    // heads cannot see the adjacency that unmakes a match.
                    negated: !negatives.is_empty() || !negative_phrases.is_empty(),
                    k,
                    deny,
                    table_avgdl,
                    params,
                    open_cost_ns: self.open_cost_ns(),
                    entry_cost_ns: self.tier_entry_cost_ns(),
                    merge_gate: manifest.options.tier_merge_gate,
                });
            }
        }
        if let Some((index, column_id)) = index_column {
            // Ceilings rescale by each term's table idf, so they order and
            // skip only when the map holds every scored term.
            let terms: Vec<&str> = musts
                .iter()
                .chain(shoulds.iter())
                .map(String::as_str)
                .collect();
            let phrases: Vec<Vec<&str>> = must_phrases
                .iter()
                .chain(should_phrases.iter())
                .map(|p| p.iter().map(String::as_str).collect())
                .collect();
            let gidf = global_idf.clone();
            // Every term is in the map (the guard above); the fallback
            // only satisfies the type.
            let idf_used = move |term: &str| gidf.get(term).copied().unwrap_or(f32::INFINITY);
            let mut c = index
                .query_ceilings(column_id, &terms, &phrases, &kept, &idf_used)
                .instrument(term_span.clone())
                .await?;
            ceilings_at_table_avgdl(&mut c, &kept, column_id, table_avgdl);
            let ceiling_of =
                |e: &Arc<SuperfileEntry>| c.get(&e.superfile_id).copied().unwrap_or(f32::INFINITY);
            kept.sort_by(|a, b| {
                ceiling_of(b)
                    .total_cmp(&ceiling_of(a))
                    .then(a.id_min.cmp(&b.id_min))
            });
            ceilings = Some(c);
        }
        // The index also knows where each term's postings sit in every
        // indexed superfile, so a cursor set can be built from those
        // locations and the superfile's dictionary never read. That holds
        // for every term the walk looks up, the negated ones included: they
        // score nothing, but a superfile that had to find one in its
        // dictionary would read the dictionary whole — tens of megabytes on
        // a large superfile, for every superfile the query opens.
        let mut located: Vec<&str> = all_terms.clone();
        located.extend(negatives.iter().map(String::as_str));
        for p in &negative_phrases {
            located.extend(p.iter().map(String::as_str));
        }
        located.sort_unstable();
        located.dedup();
        let index_locations = self
            .index_locations(column, &located, &kept)
            .instrument(term_span.clone())
            .await?;
        term_span.record("terms", located.len());
        trace::end(term_span);
        let kept_refs: Vec<&Arc<SuperfileEntry>> = kept.iter().collect();
        // Phrase-bearing queries stay per-superfile: the ranged
        // kernel is the pure term-union fast path. So does a search
        // with a per-row scope — the ranged union kernel carries no
        // admission gate, only the clause kernels do. A scope whose
        // candidate sets admit every row of every kept superfile gates
        // nothing, so it is dropped here and the fast path stays open.
        let per_row_scope: Option<Arc<HashMap<SuperfileUri, Arc<RoaringBitmap>>>> = scope
            .filter(|s| s.bounds_rows(&kept))
            .and_then(|s| s.allow.clone())
            .map(Arc::new);
        // An exact count walks each superfile whole: the ranged union
        // kernel prunes and does not count. A superfile with deleted rows
        // is walked by the clause kernels, which admit only its live rows;
        // the ranged union kernel has no such gate.
        let fanout = match exact_count || has_phrases || per_row_scope.is_some() || tombstoned {
            true => FanOut::PerSuperfile,
            false => fanout_for(musts.len(), shoulds.len(), !negatives.is_empty()),
        };
        let work_units = build_work_units(&kept_refs, fanout, pool_threads);
        let units: Vec<(Arc<SuperfileEntry>, UnitParams)> = work_units
            .into_iter()
            .map(|u| {
                let suid = u.entry.superfile_id;
                let uri = u.entry.uri;
                let ceiling = ceilings
                    .as_ref()
                    .and_then(|c| c.get(&suid).copied())
                    .unwrap_or(f32::INFINITY);
                let params = UnitParams {
                    range: u.range,
                    superfile_id: suid,
                    uri,
                    ceiling,
                    id_min: u.entry.id_min,
                    id_max: u.entry.id_max,
                };
                (u.entry, params)
            })
            .collect();
        let must_arc: Arc<Vec<String>> = Arc::new(musts);
        let should_arc: Arc<Vec<String>> = Arc::new(shoulds);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives);
        let must_ph_arc: Arc<Vec<Phrase<String>>> = Arc::new(must_phrases);
        let should_ph_arc: Arc<Vec<Phrase<String>>> = Arc::new(should_phrases);
        let neg_ph_arc: Arc<Vec<Phrase<String>>> = Arc::new(negative_phrases);
        let column_field_id = self.manifest().field_id(column);
        let column_arc = Arc::new(column_owned);
        // Exact count only: each superfile's row count, to turn its
        // tombstones into an allow-set, and the running total.
        // What a live-row set is built against: every kept superfile's
        // row count, when an exact count or a deleted row needs one.
        let row_counts: Arc<HashMap<Uuid, u64>> = Arc::new(match exact_count || tombstoned {
            true => kept.iter().map(|e| (e.superfile_id, e.n_docs)).collect(),
            false => HashMap::new(),
        });
        let total = Arc::new(AtomicU64::new(0));
        let count_handle = Arc::clone(&total);

        // Cross-segment threshold sharing: each unit reads the global
        // kth-best floor before searching and merges its surviving
        // scores back after — late units skip every block that can't
        // beat what earlier units already found. Tombstoned hits are
        // excluded from the merge so deleted rows never raise the bar.
        let shared = SharedTopK::new(k);
        let floor_handle = Arc::clone(&shared);
        let merger = Arc::clone(&shared);
        let tombstones = self.tombstone_cache.clone();
        let op_stats = self.op_stats.clone();
        let now = Instant::now();

        // Ranged units are slices of ONE superfile: share its cursor build
        // across them (keyed by superfile id) instead of re-fetching and
        // re-parsing every term's postings per slice — measured at 1M as
        // 2.5x cold bytes when slicing widened. The OnceCell coalesces
        // concurrent slices of a file; un-ranged units never touch this.
        let cursor_sets = SharedCursorSets::default();
        // Ranged kernels are multi-millisecond SYNC scans; run them as a
        // rayon wave on the reader pool, bridged with a oneshot, per the
        // concurrency contract. Inline on tokio workers they block the
        // runtime: with 8 slices the OnceCell wake of one waiter reliably
        // lost the worker race and sat a full kernel duration in a run
        // queue — measured at 1M post-compact as one ~6 ms-starved slice
        // per query gating a 9.6 ms wall over ~6 ms of actual work.
        let reader_pool = Arc::clone(&manifest.options.reader_pool);

        // One shared fan-out (`query::dispatch::fanout`) — the same
        // orchestrator the vector path uses. It warms the tombstone
        // sidecars in one batch, opens each superfile reader and runs the
        // kernel under `tokio::spawn` so cold GETs overlap, then tags +
        // tombstone-filters each unit's hits. The per-unit `params` is
        // the optional doc-id sub-range (`None` searches the whole
        // superfile) plus the superfile id for the tombstone-aware merge.
        // The superfiles scored at the table's average: those that record
        // the average they were built at (see `baked_avgdl`).
        let at_table_avgdl: Arc<HashSet<Uuid>> = Arc::new(match manifest.field_id(column) {
            Some(column_id) => kept
                .iter()
                .filter(|entry| baked_avgdl(entry, column_id).is_some())
                .map(|entry| entry.superfile_id)
                .collect(),
            None => HashSet::new(),
        });
        let merge_in_unit = exact_count || ceilings.is_none();
        // What the merge, started at the fan-out below, scores and seeds.
        let merge_idf = global_idf.clone();
        let merge_floor = Arc::clone(&shared);
        let kernel = move |r: Arc<SuperfileReader>, params: UnitParams| {
            let UnitParams {
                range,
                superfile_id: suid,
                uri,
                id_max,
                ..
            } = params;
            let avgdl = at_table_avgdl.contains(&suid).then_some(table_avgdl);
            let column_arc = Arc::clone(&column_arc);
            let must_arc = Arc::clone(&must_arc);
            let should_arc = Arc::clone(&should_arc);
            let neg_arc = Arc::clone(&neg_arc);
            let must_ph_arc = Arc::clone(&must_ph_arc);
            let should_ph_arc = Arc::clone(&should_ph_arc);
            let neg_ph_arc = Arc::clone(&neg_ph_arc);
            let shared = Arc::clone(&shared);
            let cursor_sets = Arc::clone(&cursor_sets);
            let reader_pool = Arc::clone(&reader_pool);
            let tombstones = tombstones.clone();
            let deny_by_superfile = Arc::clone(&deny_by_superfile);
            let global_idf = global_idf.clone();
            let prefetch_memos = prefetch_memos.clone();
            let index_locations = Arc::clone(&index_locations);
            let op_stats = op_stats.clone();
            let row_counts = Arc::clone(&row_counts);
            let total = Arc::clone(&total);
            // This superfile's admitted rows under a bounded scope. `kept`
            // holds only superfiles the scope admits, so the lookup
            // succeeds; an absent entry would mean no row and is treated
            // as exactly that rather than as "every row".
            let allow: Option<Arc<RoaringBitmap>> = per_row_scope.as_ref().map(|rows| {
                rows.get(&uri)
                    .cloned()
                    .unwrap_or_else(|| Arc::new(RoaringBitmap::new()))
            });
            async move {
                // A file written before a rename labels the column as it was
                // then, and its dictionary is keyed by that label; the id is
                // what finds the column in either file.
                let column_arc = r.column_alias(column_field_id, &column_arc).to_owned();
                let scoring = table_scoring(&column_arc, bm25_params, avgdl);

                // This superfile's open-wave fetches (global stats): the
                // cursor builds below serve the scored terms from the memo
                // instead of re-reading what the df wave already fetched.
                let memo = async {
                    match prefetch_memos.as_ref().and_then(|m| m.get(&suid)).cloned() {
                        Some(memo) => Some(memo),
                        // No open-wave memo: build one from the term index's
                        // locations, fetching postings only. A failure here
                        // costs the dictionary read, never the answer.
                        None => memo_from_locations(&r, &index_locations, suid).await,
                    }
                };
                // What scoring reads besides the postings — the length buckets
                // and the doc-id map — goes in the same wave as the postings,
                // so a cold superfile costs one round trip before its kernel
                // rather than three. A read that fails here is read again by
                // the kernel's own entry, which reports it.
                // A resident superfile has every byte in memory: nothing to
                // prewarm, and on a warm table a future per superfile is
                // measurable against a query of a millisecond.
                let prewarm = async {
                    if r.parquet_bytes().is_some() {
                        return;
                    }
                    let scoring = async {
                        if let Some(fts) = r.fts()
                            && let Err(error) = fts.prewarm_scoring(&column_arc).await
                        {
                            debug!(%error, "prewarm of the scoring reads failed; the kernel reads them");
                        }
                    };
                    // The id sidecar's directory too, so resolving this
                    // superfile's hits costs one round trip for their blocks.
                    join(scoring, r.prewarm_id_directory()).await;
                };
                let (memo, ()): (Option<Arc<FetchedTermMemo>>, ()) = join(memo, prewarm).await;
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
                // An exact count prunes nothing, so it takes no floor.
                let floor = match exact_count {
                    true => f32::NEG_INFINITY,
                    false => shared.floor(),
                };
                let (live_floor, allow) = {
                    // The set the merge read for this superfile, when every
                    // sidecar was readable then; else this unit's own lookup.
                    let tomb = match deny_by_superfile.get(&suid) {
                        Some(set) => Some(Ok(Arc::clone(set))),
                        None => tombstones.as_ref().map(|c| c.bitmap_for(suid, now)),
                    };
                    // The atom walks may also read the floor LIVE mid-walk and
                    // publish their own local kth into it — but a kernel heap
                    // is pre-tombstone-filter, so only a superfile with no
                    // tombstoned rows may participate: its local kth is a
                    // floor the merge (which sees only surviving scores) can
                    // never contradict. The sidecars were warmed by the
                    // dispatcher, so this lookup is an in-memory hit; on a
                    // miss/error the unit just keeps the snapshot floor.
                    let live_floor = if exact_count {
                        None
                    } else {
                        match &tomb {
                            Some(Ok(bitmap)) if !bitmap.is_empty() => None,
                            Some(Err(_)) => None,
                            _ => Some(shared.live_floor()),
                        }
                    };
                    // A superfile with deleted rows admits only its live
                    // rows: a kernel's heap then holds the k best live
                    // documents, where the filter after the walk would have
                    // dropped the deleted among the k best of all (and an
                    // exact count counts the live rows alone, as `count`
                    // does).
                    let allow = match &tomb {
                        Some(Ok(deleted)) if !deleted.is_empty() => {
                            let rows = row_counts.get(&suid).copied().ok_or_else(|| {
                                QueryError::Internal(format!(
                                    "superfile {suid} missing from the kept set"
                                ))
                            })?;
                            let rows = u32::try_from(rows).map_err(|_| {
                                QueryError::Internal(format!(
                                    "superfile {suid} row count exceeds a local id"
                                ))
                            })?;
                            let mut live = match &allow {
                                Some(set) => set.as_ref().clone(),
                                None => {
                                    let mut all = RoaringBitmap::new();
                                    all.insert_range(0..rows);
                                    all
                                }
                            };
                            live -= deleted.as_ref();
                            Some(Arc::new(live))
                        }
                        _ => allow,
                    };
                    (live_floor, allow)
                };
                let hits = match range {
                    // Ranged units exist only for pure multi-should
                    // queries (`fanout_for` never slices when a must
                    // or negated clause exists).
                    Some((start, end)) => {
                        let cell = cursor_cell(&cursor_sets, suid);
                        // The global idf is one map for the whole query, so
                        // every slice of a superfile wants cursors built
                        // with the same override — sharing the cursor set
                        // across slices stays correct under global stats.
                        let set = cell
                            .get_or_try_init(|| async {
                                let should_refs: Vec<&str> =
                                    should_arc.iter().map(|s| s.as_str()).collect();
                                let set = r
                                    .bm25_or_cursor_set(
                                        &column_arc,
                                        &should_refs,
                                        QueryIdf::Table(&global_idf),
                                        memo.as_deref(),
                                        &scoring,
                                    )
                                    .await?;
                                // Flushed inside the OnceCell init so slices
                                // sharing this superfile's cursor set count
                                // its posting bytes exactly once.
                                if let Some(stats) = &op_stats {
                                    stats.add_fts_postings_bytes(set.postings_bytes());
                                    stats.add_planned_read_ranges(set.planned_ranges());
                                }
                                Ok::<_, QueryError>(Arc::new(set))
                            })
                            .await?;
                        // Heavy kernels go to the reader pool; trivial ones
                        // run inline where the oneshot round-trip would cost
                        // more than the scan — see the gate's doc comment. A
                        // caller blocking on the query runs even a heavy one
                        // inline, as no runtime worker waits under it.
                        let hits = if should_arc.len() >= RANGED_KERNEL_POOL_MIN_TERMS {
                            let kernel_reader = Arc::clone(&r);
                            let kernel_set = Arc::clone(set);
                            let kernel_stats = op_stats.clone();
                            let kernel_column = column_arc.clone();
                            run_on_pool_unless_blocking(
                                Some(&reader_pool),
                                "ranged fts kernel: reader pool dropped result",
                                move || {
                                    op_stats::timed_kernel(&kernel_stats, || {
                                        kernel_reader.bm25_search_or_range_prebuilt(
                                            &kernel_set,
                                            k,
                                            start,
                                            end,
                                            floor,
                                            &table_scoring(&kernel_column, bm25_params, avgdl),
                                        )
                                    })
                                },
                            )
                            .await
                            .map_err(|e| QueryError::Internal(e.to_string()))??
                        } else {
                            op_stats::timed_kernel(&op_stats, || {
                                r.bm25_search_or_range_prebuilt(set, k, start, end, floor, &scoring)
                            })?
                        };
                        r.hits_to_rows_async(hits).await?
                    }
                    None => {
                        let must_refs: Vec<&str> = must_arc.iter().map(|s| s.as_str()).collect();
                        let should_refs: Vec<&str> =
                            should_arc.iter().map(|s| s.as_str()).collect();
                        let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                        let prep = r
                            .prepare_clauses(
                                &column_arc,
                                ClauseLists {
                                    musts: &must_refs,
                                    shoulds: &should_refs,
                                    negatives: &neg_refs,
                                    must_phrases: &must_ph_arc,
                                    should_phrases: &should_ph_arc,
                                    negative_phrases: &neg_ph_arc,
                                    idf: QueryIdf::Table(&global_idf),
                                    prefetched: memo.as_deref(),
                                    live_floor,
                                    allow,
                                    exact_count,
                                },
                                k,
                                floor,
                                &scoring,
                            )
                            .instrument(trace::phase(phases, || detail_span!("fts.prepare")))
                            .await?;
                        if let Some(stats) = &op_stats {
                            stats.add_fts_postings_bytes(prep.postings_bytes());
                            stats.add_planned_read_ranges(prep.planned_ranges());
                            // Single-term / phrase shapes finish inside
                            // `prepare_clauses`; their walk's on-CPU time
                            // rides the `Done` (0 for cursor shapes, whose
                            // kernels are bracketed below).
                            stats.add_kernel_cpu_ns(prep.inline_kernel_cpu_ns());
                        }
                        if exact_count {
                            // Every match is scored, so the scan is the heavy
                            // case by definition: a pool job unless the walk
                            // already finished inside `prepare_clauses`.
                            let (hits, count) = match prep {
                                prep @ PreparedClauses::Done { .. } => {
                                    r.run_prepared_with_count_in_blob_ids(prep, &scoring)?
                                }
                                prep => {
                                    let kernel_reader = Arc::clone(&r);
                                    let kernel_stats = op_stats.clone();
                                    let kernel_column = column_arc.clone();
                                    run_on_pool_unless_blocking(
                                        Some(&reader_pool),
                                        "exact-count fts kernel: reader pool dropped result",
                                        move || {
                                            op_stats::timed_kernel(&kernel_stats, || {
                                                kernel_reader.run_prepared_with_count_in_blob_ids(
                                                    prep,
                                                    &table_scoring(
                                                        &kernel_column,
                                                        bm25_params,
                                                        avgdl,
                                                    ),
                                                )
                                            })
                                        },
                                    )
                                    .await
                                    .map_err(|e| QueryError::Internal(e.to_string()))??
                                }
                            };
                            total.fetch_add(count, AtomicOrdering::Relaxed);
                            r.hits_to_rows_async(hits).await?
                        } else {
                            // The kernels score in blob ids; the top hits are
                            // translated to rows here, on this runtime, reading
                            // only the doc-id map pages they fall in.
                            let hits = match prep {
                                // Already-final shapes: the walk (and its
                                // kernel time) happened inside
                                // `prepare_clauses`. Its hits are blob ids
                                // like every other shape's, and are turned
                                // into rows below with the rest.
                                prep @ PreparedClauses::Done { .. } => {
                                    r.run_prepared_in_blob_ids(prep, &scoring)?
                                }
                                // Gate on posting mass, not term count: this
                                // scan isn't sliced, so a rare-term query
                                // with many terms can be cheaper than a
                                // common-term pair.
                                prep if prep.posting_mass() >= UNRANGED_KERNEL_POOL_MIN_MASS => {
                                    let kernel_reader = Arc::clone(&r);
                                    let kernel_stats = op_stats.clone();
                                    let kernel_column = column_arc.clone();
                                    run_on_pool_unless_blocking(
                                        Some(&reader_pool),
                                        "un-ranged fts kernel: reader pool dropped result",
                                        move || {
                                            op_stats::timed_kernel(&kernel_stats, || {
                                                kernel_reader.run_prepared_in_blob_ids(
                                                    prep,
                                                    &table_scoring(
                                                        &kernel_column,
                                                        bm25_params,
                                                        avgdl,
                                                    ),
                                                )
                                            })
                                        },
                                    )
                                    .await
                                    .map_err(|e| QueryError::Internal(e.to_string()))??
                                }
                                prep => op_stats::timed_kernel(&op_stats, || {
                                    r.run_prepared_in_blob_ids(prep, &scoring)
                                })?,
                            };
                            r.hits_to_rows_async(hits).await?
                        }
                    }
                };
                // Raise the global floor with this unit's surviving
                // scores. Sidecars were prefetched by the dispatcher,
                // so the bitmap lookup is an in-memory hit; on a cache
                // miss/error we simply don't merge (a lower floor is
                // always safe).
                // Under the ceiling-ordered fan-out the driver merges a
                // unit's surviving scores as the unit finishes; the
                // unordered fan-out's units, all running at once, merge
                // their own here.
                if merge_in_unit {
                    match tombstones.as_ref().map(|c| c.bitmap_for(suid, now)) {
                        Some(Ok(bitmap)) if !bitmap.is_empty() => shared.merge(
                            id_max,
                            hits.iter()
                                .filter(|(d, _)| !bitmap.contains(d.get()))
                                .map(|(_, s)| *s),
                        ),
                        Some(Err(_)) => {}
                        _ => shared.merge(id_max, hits.iter().map(|(_, s)| *s)),
                    }
                }
                Ok(rows_as_local_ids(hits))
            }
        };
        let fanout_span = trace::phase(phases, || {
            tiered_span!(
                "fts.fanout",
                units = units.len(),
                resolved = tracing::field::Empty,
                ruled_out = tracing::field::Empty,
            )
        });
        let fanout_records = fanout_span.clone();
        // The merge, prepared here. With its tier chunks in memory it runs
        // first: it costs no round trip, and the kernels it spares cost
        // more than the wait. With chunks to read from storage it runs
        // beside the first opens, and re-plans the units not yet issued
        // when it reports. Either way one plan applies what it found.
        let mut units = units;
        // Each superfile's smallest `_id`, for narrowing score ties: taken
        // from the units here rather than looked up per tied superfile.
        let id_mins: HashMap<SuperfileUri, i128> = units
            .iter()
            .map(|(entry, _)| (entry.uri, entry.id_min))
            .collect();
        let mut settled_before: Vec<Vec<SuperfileHit>> = Vec::new();
        let mut replan: Option<dispatch::ReplanFuture<'_, UnitParams, Vec<SuperfileHit>>> = None;
        if let (Some(query), Some((index, _))) = (tier_query, index_column) {
            let global_idf = merge_idf;
            let pool = &manifest.options.reader_pool;
            let span = fanout_span.clone();
            let merger = merge_floor;
            // The kept superfiles in the order the merge bounds them.
            let merged_ids: Vec<Uuid> = merge_kept.iter().map(|e| e.superfile_id).collect();
            // Units per superfile: a superfile the index answers for is
            // answered only when none of its units was issued — an issued
            // unit's kernel answers for its rows, and the rest open as planned.
            let units_of: HashMap<Uuid, usize> =
                units.iter().fold(HashMap::new(), |mut m, (e, _)| {
                    *m.entry(e.superfile_id).or_insert(0) += 1;
                    m
                });
            let reader = self;
            let selected = &selected;
            let pruned = &pruned;
            // What a selection does to the units it is handed: the
            // superfiles the index answers for are settled and their
            // scores folded into the floor, the ruled-out dropped, the rest
            // reordered by their tighter bounds, and the threshold seeded.
            let make_plan = move |selection: tier::TierSelection| -> dispatch::Replan<'_, UnitParams, Vec<SuperfileHit>> {
                if let Some(stats) = &reader.op_stats {
                    stats.add_tier_read(selection.tier_bytes, selection.entries_read);
                }
                if let Some(per_entry) = selection.merge_ns.checked_div(selection.entries_read) {
                    reader.note_tier_entry_cost(per_entry);
                }
                Box::new(move |remaining: dispatch::Units<UnitParams>| {
                    let bounds_of: HashMap<Uuid, tier::SuperfileBounds> =
                        merged_ids.iter().copied().zip(selection.bounds).collect();
                    let mut answers: HashMap<Uuid, Vec<(u32, f32)>> = selection
                        .resolved
                        .into_iter()
                        .map(|(si, hits)| (merged_ids[si], hits))
                        .collect();
                    let mut remaining_of: HashMap<Uuid, usize> = HashMap::new();
                    for (e, _) in &remaining {
                        *remaining_of.entry(e.superfile_id).or_insert(0) += 1;
                    }
                    answers.retain(|id, _| remaining_of.get(id) == units_of.get(id));
                    let answered: HashSet<Uuid> = answers.keys().copied().collect();
                    let mut keep: Vec<(Arc<SuperfileEntry>, UnitParams, f32)> =
                        Vec::with_capacity(remaining.len());
                    let mut settled: Vec<Vec<SuperfileHit>> = Vec::new();
                    let mut ruled_out = 0u64;
                    for (entry, mut params) in remaining {
                        let id = entry.superfile_id;
                        if answered.contains(&id) {
                            // The merge read past every deleted row, so
                            // what it answers with is live; its scores
                            // are real documents at known ids, and raise
                            // the floor and settle ties.
                            if let Some(hits) = answers.remove(&id) {
                                merger.merge(entry.id_max, hits.iter().map(|h| h.1));
                                settled.push(dispatch::tag_hits(&entry, hits));
                            }
                            continue;
                        }
                        let Some(b) = bounds_of.get(&id) else {
                            keep.push((entry, params, f32::NEG_INFINITY));
                            continue;
                        };
                        if b.ub == f32::NEG_INFINITY {
                            ruled_out += 1;
                            continue;
                        }
                        // A merged bound is at most the ceiling; the skip
                        // decision reads it as such.
                        params.ceiling = b.ub.min(params.ceiling);
                        keep.push((entry, params, b.lb));
                    }
                    // The rest open by their upper bound, as the ceiling
                    // path opens by ceiling, then by their best known
                    // candidate; equal bounds open in `_id` order, so a run
                    // of superfiles that can only tie is settled by the
                    // first of them.
                    keep.sort_by(|(a, x, xl), (b, y, yl)| {
                        y.ceiling
                            .total_cmp(&x.ceiling)
                            .then(yl.total_cmp(xl))
                            .then(a.id_min.cmp(&b.id_min))
                    });
                    // The tiers' lower bound on the k-th score is a floor
                    // every kernel may prune against; the real scores that
                    // replace these placeholders only raise it.
                    if selection.theta_lb.is_finite() {
                        merger.merge(i128::MAX, repeat_n(selection.theta_lb, k));
                    }
                    let resolved = answered.len() as u64;
                    if resolved > 0 && let Some(stats) = &reader.op_stats {
                        stats.add_superfiles_resolved(resolved);
                    }
                    span.record("resolved", resolved);
                    span.record("ruled_out", ruled_out);
                    selected.store(true, AtomicOrdering::Relaxed);
                    pruned.store(
                        resolved + ruled_out > 0 || selection.theta_lb.is_finite(),
                        AtomicOrdering::Relaxed,
                    );
                    (keep.into_iter().map(|(e, p, _)| (e, p)).collect(), settled)
                })
            };
            // The tiers only make the query faster, and the ceilings are in
            // hand: a tier that cannot be read or does not verify leaves
            // the query to them.
            let unreadable = |e: term_index::TermIndexError| {
                warn!(error = %e, "term-index tier unreadable; the query keeps its ceilings");
            };
            let prepared = tier::prepare_merge(index, &query, &merge_kept, &global_idf, pool)
                .await
                .unwrap_or_else(|e| {
                    unreadable(e);
                    None
                });
            if prepared.is_none() {
                declined.store(true, AtomicOrdering::Relaxed);
            }
            let window = manifest.options.ordered_open_window();
            match prepared {
                Some(prepared) if prepared.resident => {
                    let selection = prepared.run().await.unwrap_or_else(|e| {
                        unreadable(e);
                        None
                    });
                    if let Some(selection) = selection {
                        let (keep, settled) = make_plan(selection)(units);
                        units = keep;
                        settled_before = settled;
                    }
                }
                // On a table whose kept units all issue at once, a merge that
                // must read its tiers would report after every open is under
                // way and change none; its reads would only compete with the
                // opens, and the gate would spend samples on a plan that
                // cannot act. Its first wave is read without waiting, for the
                // next query of these terms, and nothing is reported.
                Some(prepared) if units.len() <= window => {
                    prepared.prefetch(Arc::clone(index));
                }
                Some(prepared) => {
                    replan = Some(Box::pin(async move {
                        match prepared.run().await {
                            Ok(Some(selection)) => Some(make_plan(selection)),
                            Ok(None) => None,
                            Err(e) => {
                                unreadable(e);
                                None
                            }
                        }
                    }));
                }
                None => {}
            }
        }
        let per_unit = match !exact_count && ceilings.is_some() {
            // Units are in descending ceiling order. A unit whose ceiling is
            // strictly below the running k-th score cannot place a document
            // in the top k — not even a tie the stable `_id` order could
            // admit — so it is never opened; one whose ceiling equals it
            // can only tie, and is not opened either once every document
            // at the floor precedes its own in `_id` order.
            true => {
                let window = manifest.options.ordered_open_window();
                // A superfile whose ceiling only ties the running k-th score
                // is skipped once the tie is settled by `_id` order: every
                // `_id` it holds is past the tied document's. That needs
                // an exact comparison, no tolerance widening the floor.
                let tie_walk = score_tolerance <= 0.0;
                // The driver folds each finished unit's surviving scores
                // (tombstoned ones are already gone) into the floor.
                let id_max_of: HashMap<SuperfileUri, i128> =
                    units.iter().map(|(_, p)| (p.uri, p.id_max)).collect();
                dispatch::fanout_local_hits_ordered(
                    self,
                    units,
                    window,
                    move |params: &UnitParams| {
                        let floor = competitive_floor(floor_handle.floor(), score_tolerance);
                        !ceiling_can_compete(params.ceiling, floor)
                            || (tie_walk
                                && params.ceiling == floor
                                && floor_handle.tie_skippable(floor, params.id_min))
                    },
                    kernel,
                    move |hits: &Vec<SuperfileHit>| {
                        if let Some(first) = hits.first()
                            && let Some(&id_max) = id_max_of.get(&first.superfile)
                        {
                            merger.merge(id_max, hits.iter().map(|h| h.score));
                        }
                    },
                    replan,
                )
                .instrument(fanout_span)
                .await?
            }
            false => {
                dispatch::fanout_local_hits(self, units, kernel)
                    .instrument(fanout_span)
                    .await?
            }
        };
        // The index served but no merge reported in time — it declined,
        // or every unit was issued before it did: it answered for nothing
        // and ruled nothing out, and the span says so.
        if tiers_serve && !selected.load(AtomicOrdering::Relaxed) {
            fanout_records.record("resolved", 0u64);
            fanout_records.record("ruled_out", 0u64);
        }
        let mut per_unit = per_unit;
        per_unit.extend(settled_before);
        let hits = select_top_k_stable(self, per_unit, k, &id_mins).await?;
        // Noted for the way the policy chose, whether or not the merge
        // then ran: a shape it declines every time is thereby measured as
        // the cost of asking, and the class settles on not asking.
        if tiers_serve && learned {
            // A merge that reported says whether it did anything; one that
            // declined did nothing by construction; one that did not get
            // to report says nothing.
            let outcome = if selected.load(AtomicOrdering::Relaxed) {
                Some(pruned.load(AtomicOrdering::Relaxed))
            } else if declined.load(AtomicOrdering::Relaxed) {
                Some(false)
            } else {
                None
            };
            self.tier_policy().note(
                policy_class,
                merge_now,
                outcome,
                ranked_started.elapsed().as_nanos() as u64,
            );
        }
        Ok((hits, count_handle.load(AtomicOrdering::Relaxed)))
    }

    /// Global BM25 idf per scored term, via the
    /// fused open wave: corpus-wide `N` from the manifest, df per term
    /// summed across (a) the scoring-kept superfiles — which fetch their
    /// scored terms' postings ranges here, the very reads their walks
    /// need, returned to them as per-superfile memos — and (b) the
    /// presence residual (superfiles that may contain a scored term but
    /// were pruned from scoring), which contribute df through a
    /// dict-only probe in the same wave. Terms already cached for this
    /// manifest generation skip the wave entirely.
    ///
    /// Work accounting: memo fetches are NOT flushed here — the walk
    /// wave's cursors report them, keeping per-query stats equal to the
    /// per-superfile plan. Residual probes flush here (they have no
    /// walk), bounded by presence − kept superfiles, dict-only.
    async fn global_idf_open_wave(
        &self,
        manifest: &ManifestSnapshot,
        column: &str,
        terms: &[String],
        kept: &[Arc<SuperfileEntry>],
        corpus: ColumnLengthStats,
    ) -> Result<(GlobalTermIdf, PrefetchMemos), QueryError> {
        let mut map = GlobalTermIdf::with_capacity(terms.len());
        // The collection size idf is computed against: documents that
        // carry tokens in this column, summed table-wide. It is the
        // population the per-term document frequencies below are counted
        // over, so the two have to come from the same corpus — a row
        // that is null here can never contribute to a `df`, and counting
        // it in `N` would weight the column's common terms too heavily
        // against its rare ones.
        let global_n = corpus.n_scored_docs;
        if terms.is_empty() || global_n == 0 {
            return Ok((map, None));
        }
        // Idf is a pure function of the snapshot, so serve repeat terms
        // from the per-generation cache and run the wave only for misses.
        let manifest_id = manifest.manifest_id;
        let cache = self.global_idf_cache();
        let cached = cache.get(manifest_id, column, terms);
        for (t, c) in terms.iter().zip(cached.iter()) {
            if let Some(idf) = c {
                map.insert(t.clone(), *idf);
            }
        }
        let misses: Vec<String> = terms
            .iter()
            .zip(cached.iter())
            .filter(|(_, c)| c.is_none())
            .map(|(t, _)| t.clone())
            .collect();
        if misses.is_empty() {
            return Ok((map, None));
        }
        let column_id = manifest
            .field_id(column)
            .ok_or_else(|| QueryError::InvalidQuery(format!("unknown column '{column}'")))?;

        // A complete term index already holds every term's gross df in
        // every live superfile — the same numbers a superfile's dictionary
        // would give — so the corpus-wide df is a sum over its postings and
        // no dictionary is opened. The walk builds its memos from the
        // index's locations. An incomplete index takes the wave below, like
        // a table with no index.
        if manifest.term_index_complete()
            && let Some(index) = manifest.term_index().await?
        {
            let mut fresh: Vec<(&str, f32)> = Vec::with_capacity(misses.len());
            let asked: Vec<&str> = misses.iter().map(String::as_str).collect();
            let runs = index.runs_many(column_id, &asked).await?;
            // The index keeps postings for removed superfiles, so only live
            // ones count; liveness is checked for the terms' holders alone.
            let holders: HashSet<Uuid> = runs
                .iter()
                .flat_map(|r| term_index::postings_of(r))
                .filter_map(|p| index.superfile_id(p.superfile))
                .collect();
            let live = manifest
                .live_among(&index, &holders)
                .await
                .map_err(QueryError::ManifestLoad)?;
            for (t, runs) in misses.iter().zip(runs) {
                let df: u64 = term_index::postings_of(&runs)
                    .filter(|p| {
                        index
                            .superfile_id(p.superfile)
                            .is_some_and(|id| live.contains(&id))
                    })
                    .map(|p| p.df)
                    .sum();
                let idf = bm25::idf(global_n, df.min(global_n));
                map.insert(t.clone(), idf);
                fresh.push((t.as_str(), idf));
            }
            cache.insert(manifest_id, column, &fresh);
            return Ok((map, None));
        }

        // Presence prune over the missing terms: every superfile whose
        // bloom may contain any of them owes a df contribution.
        let prune = PruneLeaf::TermPresence {
            column: column.to_owned(),
            terms: misses.clone(),
            mode: BoolMode::Or,
        };
        let presence: Vec<Arc<SuperfileEntry>> =
            select_fts_superfiles(manifest, slice::from_ref(&prune), column).await?;
        let kept_ids: HashSet<Uuid> = kept.iter().map(|e| e.superfile_id).collect();
        let column_field_id = self.manifest().field_id(column);
        let column_arc = Arc::new(column.to_owned());
        let terms_arc: Arc<Vec<String>> = Arc::new(misses.clone());
        let units: Vec<(Arc<SuperfileEntry>, (Uuid, bool))> = presence
            .into_iter()
            .map(|e| {
                let suid = e.superfile_id;
                let full = kept_ids.contains(&suid);
                (e, (suid, full))
            })
            .collect();
        let op_stats = self.op_stats.clone();
        let per_sf: Vec<(Uuid, Vec<u64>, Option<Arc<FetchedTermMemo>>)> = dispatch::fanout_with(
            self,
            units,
            false,
            self.open_intent(),
            move |r, _entry, _sidecars, _now, (suid, full): (Uuid, bool)| {
                let column_arc = Arc::clone(&column_arc);
                let terms_arc = Arc::clone(&terms_arc);
                let op_stats = op_stats.clone();
                async move {
                    // A file written before a rename labels the column as it was
                    // then, and its dictionary is keyed by that label; the id is
                    // what finds the column in either file.
                    let column_arc = r.column_alias(column_field_id, &column_arc).to_owned();

                    let refs: Vec<&str> = terms_arc.iter().map(String::as_str).collect();
                    if full {
                        // Scoring superfile: fetch the scored terms outright
                        // — dictionary + postings ranges, the walk's own
                        // reads — and hand them back through the memo. The
                        // walk wave flushes this work when it builds the
                        // cursors, so nothing is flushed here.
                        let memo = r.fetch_scored_terms(&column_arc, &refs).await?;
                        let dfs: Vec<u64> = refs.iter().map(|t| memo.df(t)).collect();
                        Ok::<_, QueryError>((suid, dfs, Some(Arc::new(memo))))
                    } else {
                        // Residual superfile (contains a scored term, pruned
                        // from scoring): df only, from the dictionary value
                        // + header hint — no postings body.
                        let (dfs, work) = r.term_dfs(&column_arc, &refs).await?;
                        if let Some(stats) = &op_stats {
                            stats.add_fts_postings_bytes(work.postings_bytes);
                            stats.add_planned_read_ranges(work.planned_ranges);
                            stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                        }
                        Ok((suid, dfs, None))
                    }
                }
            },
        )
        .await?;

        let mut global_df = vec![0u64; misses.len()];
        let mut memos: HashMap<Uuid, Arc<FetchedTermMemo>> = HashMap::new();
        for (suid, dfs, memo) in per_sf {
            for (i, d) in dfs.into_iter().enumerate() {
                global_df[i] += d;
            }
            if let Some(m) = memo {
                memos.insert(suid, m);
            }
        }
        let mut fresh: Vec<(&str, f32)> = Vec::with_capacity(misses.len());
        for (i, t) in misses.iter().enumerate() {
            // Clamp to the collection size: df counts tombstoned docs until
            // compaction, and idf needs df <= n_docs.
            let df = global_df[i].min(global_n);
            let idf = bm25::idf(global_n, df);
            map.insert(t.clone(), idf);
            fresh.push((t.as_str(), idf));
        }
        cache.insert(manifest_id, column, &fresh);
        Ok((map, Some(Arc::new(memos))))
    }

    /// Prefix-expanded BM25 search across the pinned manifest's
    /// superfiles. The prefix is ASCII-lowercased, expanded to every
    /// indexed term in the table that starts with it, and the expansion is
    /// ranked as a `BoolMode::Or` query, so each term carries table-wide
    /// idf exactly as it would if searched on its own.
    ///
    /// Returns up to `k` highest-scoring hits, sorted descending by score.
    /// Empty supertable (no superfiles) and `k == 0` short-circuit to an
    /// empty `Vec`.
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
        // As in `bm25_search_async`: a prefix query over a column with no
        // full-text index would otherwise fail deep in the scan with an opaque
        // missing-metadata error. Reject up front, naming the searchable set.
        if manifest.try_fts_tokenizer_for(column).is_none() {
            return Err(QueryError::InvalidQuery(no_fts_index_message(
                column,
                &manifest.fts_configs(),
            )));
        }
        // Expansion matches the dictionary's lowercased bytes, and the
        // term-range prune compares the same bytes, so both see one prefix.
        let prune_leaf = PruneLeaf::Prefix {
            column: column.to_owned(),
            prefix: prefix.to_ascii_lowercase().into_bytes(),
        };
        let kept = self.select_scored(column, &prune_leaf, None).await?;
        let shoulds = self
            .expand_prefix(column, &prune_leaf, kept.clone())
            .await?;
        if shoulds.is_empty() {
            return Ok(Vec::new());
        }
        let clauses = ScoredClauses {
            shoulds,
            ..ScoredClauses::default()
        };
        // Prefix search takes no search options, so columns score with the
        // parameters they declared.
        let (hits, _) = self
            .bm25_search_clauses(column, clauses, kept, k, None, 0.0, None, false)
            .await?;
        Ok(hits)
    }

    /// Every term in `column` starting with `prefix_leaf`'s prefix, across
    /// the superfiles in `kept`, deduplicated and in byte order.
    async fn expand_prefix(
        &self,
        column: &str,
        prefix_leaf: &PruneLeaf,
        kept: Vec<Arc<SuperfileEntry>>,
    ) -> Result<Vec<String>, QueryError> {
        let PruneLeaf::Prefix { prefix, .. } = prefix_leaf else {
            return Err(QueryError::Internal(
                "prefix expansion without a prefix".into(),
            ));
        };
        let column_field_id = self.manifest().field_id(column);
        let column_arc: Arc<str> = Arc::from(column);
        let prefix_arc: Arc<[u8]> = Arc::from(prefix.as_slice());
        let pool = Arc::clone(&self.manifest().options.reader_pool);
        let units = kept.into_iter().map(|entry| (entry, ())).collect();
        let per_superfile: Vec<Vec<Vec<u8>>> = dispatch::fanout_with(
            self,
            units,
            false,
            self.open_intent(),
            move |r, _entry, _sidecars, _now, ()| {
                let column_arc = Arc::clone(&column_arc);
                let prefix_arc = Arc::clone(&prefix_arc);
                let pool = Arc::clone(&pool);
                async move {
                    let fts = r
                        .fts()
                        .ok_or_else(|| QueryError::Internal("superfile has no FTS index".into()))?;
                    // A file written before a rename labels the column as it
                    // was then; the id finds it in either file.
                    let column = r.column_alias(column_field_id, &column_arc);
                    Ok(fts
                        .terms_with_prefix(column, &prefix_arc, Some(&pool))
                        .await?)
                }
            },
        )
        .await?;
        let terms: BTreeSet<Vec<u8>> = per_superfile.into_iter().flatten().collect();
        // Dictionary keys are valid UTF-8 by construction: a tokenizer emits
        // `&str` terms.
        Ok(terms
            .into_iter()
            .filter_map(|t| String::from_utf8(t).ok())
            .collect())
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
        let manifest = self.manifest();
        // Same up-front check as the scored path: without a full-text index
        // on `column` there is no analyzer to parse the query with, and no
        // postings to match it against.
        let Some(tokenizer) = manifest.try_fts_tokenizer_for(column) else {
            return Err(QueryError::InvalidQuery(no_fts_index_message(
                column,
                &manifest.fts_configs(),
            )));
        };
        let clauses = tokenizer.parse(query).into_clauses(mode);
        // Drop repeated tokens within each clause role. Unranked
        // matching is set-valued — an AND/OR/exclude over a term repeated
        // in the query (e.g. `+to +be +or +not +to +be`) is idempotent —
        // so a duplicate only adds a redundant cursor that intersects (or
        // unions) a list with itself. Order-preserving so the rarest-first
        // cursor ordering downstream is unaffected. Phrase members are
        // *not* deduped: position matters there. (Count path only; the
        // scored path must keep repeats, which can affect BM25.)
        // Linear dedup, not a HashSet: clause token lists are tiny (a
        // handful of terms), so an O(n²) scan over the already-kept
        // tokens is cheaper than allocating a set + hashing, and — unlike
        // the set — it adds no per-query allocation on the overwhelmingly
        // common no-duplicate query. Order-preserving; only the first
        // occurrence's `String` is materialized.
        let dedup = |tokens: Vec<Cow<'_, str>>| -> Vec<String> {
            let mut out: Vec<String> = Vec::with_capacity(tokens.len());
            for t in tokens {
                if !out.iter().any(|k| k.as_str() == t.as_ref()) {
                    out.push(t.into_owned());
                }
            }
            out
        };
        let musts: Vec<String> = dedup(clauses.musts);
        let shoulds: Vec<String> = dedup(clauses.shoulds);
        let negatives: Vec<String> = dedup(clauses.negatives);
        // `Phrase::map` keeps each term's offset, which is what a
        // phrase on a stopworded column needs: its terms were not
        // adjacent in the query and must not be required adjacent here.
        let own_phrases = |phrases: Vec<Phrase<Cow<'_, str>>>| -> Vec<Phrase<String>> {
            phrases.iter().map(|p| p.map(|t| t.to_string())).collect()
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
        let prune_leaf =
            presence_leaf(column, &match_set.terms, &match_set.phrases, match_set.mode);
        let kept = select_fts_superfiles(
            self.manifest().as_ref(),
            slice::from_ref(&prune_leaf),
            column,
        )
        .await?;
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
        tracing::instrument(skip_all, fields(column = column, mode = ?mode, role = self.role().as_str(), origin = OpOrigin::Query.as_str()))
    )]
    pub(crate) async fn token_match_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.token_match_hits_async(column, query, mode, true).await
    }

    /// [`Self::token_match_async`] with a say over whether every hit
    /// leaves carrying its stable `_id`. A caller that never reads the
    /// ids, an aggregate over the match that projects only scalar
    /// columns, passes `false` and skips the stamping. A file whose ids
    /// follow from span arithmetic is stamped as the fan-out tags its
    /// hits, so there the skip saves one pass over the hits; on a
    /// cell-packed or gapped-span file stamping reads the file's `_id`
    /// column, and that read is what the skip saves. Hits then carry a
    /// stable id only where the fan-out tagged one, and must not be asked
    /// for `_id`.
    pub(crate) async fn token_match_hits_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        stamp_stable_ids: bool,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let Some((units, kernel)) = self.match_kernel(column, query, mode).await? else {
            return Ok(Vec::new());
        };
        let phases = self.phase_spans();
        let fanout_span = trace::phase(phases, || tiered_span!("fts.fanout", units = units.len()));
        let per_unit = dispatch::fanout_local_hits(self, units, move |r, suid| {
            let kernel = Arc::clone(&kernel);
            async move {
                Ok(kernel
                    .docs(&r, suid)
                    .await?
                    .into_iter()
                    .map(|d| (d.get(), 0.0f32))
                    .collect::<Vec<_>>())
            }
        })
        .instrument(fanout_span)
        .await?;
        // Exact pre-size: `Flatten`'s size_hint is opaque, and growth
        // reallocations copy the whole hit vec repeatedly at 1M hits.
        let total: usize = per_unit.iter().map(Vec::len).sum();
        let mut hits: Vec<SuperfileHit> = Vec::with_capacity(total);
        for unit in per_unit {
            hits.extend(unit);
        }
        if stamp_stable_ids {
            dispatch::attach_stable_ids_to_hits(self, &mut hits).await?;
        }
        Ok(hits)
    }

    /// The matching rows' `names` columns, one batch per superfile in no
    /// particular order, for a consumer that reads neither `_id` nor the
    /// score: an aggregate over the match. Each superfile's rows go from
    /// the kernel's walk to a column take and out as their own batch, with
    /// no hit per row, no stable-id stamping and no global interleave in
    /// between, which on a match of a few hundred thousand rows is most of
    /// what producing a hit set costs. Tombstoned rows are left out, as
    /// the hit path leaves them out.
    ///
    /// A plain union of terms over a superfile holding a large share of the
    /// table is cut into doc-id slices, as many as its share of the reader
    /// pool ([`build_work_units`]), each walked and taken on its own and out
    /// as its own batch: a table compacted into one file would otherwise run
    /// the whole match on one thread.
    pub(crate) async fn token_match_columns_async(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        names: &[&str],
    ) -> Result<Vec<RecordBatch>, QueryError> {
        let Some((units, kernel)) = self.match_kernel(column, query, mode).await? else {
            return Ok(Vec::new());
        };
        let reader_pool = Arc::clone(&self.manifest().options.reader_pool);
        let fanout = match kernel.slices() {
            true => FanOut::SubRanges,
            false => FanOut::PerSuperfile,
        };
        let kept: Vec<&Arc<SuperfileEntry>> = units.iter().map(|(entry, _)| entry).collect();
        let units: Vec<(Arc<SuperfileEntry>, (Uuid, Option<(u32, u32)>))> =
            build_work_units(&kept, fanout, reader_pool.current_num_threads())
                .into_iter()
                .map(|unit| {
                    let suid = unit.entry.superfile_id;
                    (unit.entry, (suid, unit.range))
                })
                .collect();
        let cursor_sets = SharedCursorSets::default();
        let manifest = Arc::clone(self.manifest());
        let out_schema =
            hit_output_schema(&manifest.stored_schema(), names).map_err(QueryError::DataFusion)?;
        let names: Arc<Vec<String>> = Arc::new(names.iter().map(|s| (*s).to_string()).collect());
        let op_stats = self.op_stats.clone();
        let shared = self.shared_inner();
        let schema_id = manifest.table_schema().schema_id();
        let phases = self.phase_spans();
        let fanout_span = trace::phase(phases, || tiered_span!("fts.fanout", units = units.len()));
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         (suid, range): (Uuid, Option<(u32, u32)>)| {
            let kernel = Arc::clone(&kernel);
            let manifest = Arc::clone(&manifest);
            let out_schema = Arc::clone(&out_schema);
            let names = Arc::clone(&names);
            let op_stats = op_stats.clone();
            let shared = Arc::clone(&shared);
            let cursor_sets = Arc::clone(&cursor_sets);
            let reader_pool = Arc::clone(&reader_pool);
            async move {
                let docs = match range {
                    None => kernel.docs(&r, suid).await?,
                    Some((start, end)) => {
                        let cell = cursor_cell(&cursor_sets, suid);
                        let set = cell
                            .get_or_try_init(|| async {
                                Ok::<_, QueryError>(Arc::new(kernel.cursor_set(&r, suid).await?))
                            })
                            .await?;
                        // A slice's walk is a sync scan of a share of the
                        // file: a rayon job on the reader pool, so the
                        // slices run side by side and no runtime worker
                        // blocks under one.
                        let (set, slice_reader, slice_kernel) =
                            (Arc::clone(set), Arc::clone(&r), Arc::clone(&kernel));
                        run_on_pool_unless_blocking(
                            Some(&reader_pool),
                            "match slice: reader pool dropped result",
                            move || slice_kernel.slice(&slice_reader, &set, start, end),
                        )
                        .await
                        .map_err(|e| QueryError::Internal(e.to_string()))??
                    }
                };
                let mut locals: Vec<u32> = docs.into_iter().map(RowId::get).collect();
                if let Some(cache) = cache.as_ref()
                    && let Some(deny) =
                        dispatch::tombstone_deny_set(cache, entry.superfile_id, now)?
                {
                    locals.retain(|local| !deny.contains(*local));
                }
                let refs: Vec<&str> = names.iter().map(String::as_str).collect();
                // The same rows of the same superfile under the same columns
                // and schema are the same batch for as long as the immutable
                // superfile lives; the hit path memoises them under that
                // key, and so does this.
                let decoded = &shared.decoded_scalar_cache;
                if let Some(batch) = decoded.get(entry.uri, schema_id, &locals, &refs) {
                    return Ok(batch);
                }
                let batch =
                    take_file_columns(&manifest, &r, locals.clone(), &refs, &out_schema, &op_stats)
                        .await
                        .map_err(QueryError::DataFusion)?;
                decoded.insert(entry.uri, schema_id, &locals, &refs, batch.clone());
                Ok(batch)
            }
        };
        dispatch::fanout_with(self, units, true, self.open_intent(), body)
            .instrument(fanout_span)
            .await
    }

    /// `scope` narrowed to the rows `query`'s terms match under `mode`: the
    /// rows a BM25 search of the same query can return, which is all a
    /// `WHERE` on that search ever needs checked. A superfile the term
    /// bloom rules out, or that the scope does not admit, is not opened;
    /// in the rest the match is intersected with the scope's set when it
    /// bounds one (already free of tombstones) and has the superfile's
    /// tombstones taken out when it does not. The result is bounded, its
    /// sets free of tombstones, and holds only superfiles with a row left.
    pub(crate) async fn match_scope(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        scope: &CandidateScope,
    ) -> Result<CandidateScope, QueryError> {
        let Some((units, kernel)) = self.match_kernel(column, query, mode).await? else {
            return Ok(CandidateScope::empty());
        };
        let admitted: HashSet<SuperfileUri> =
            scope.admitted_superfiles().map(|entry| entry.uri).collect();
        let units: Vec<(Arc<SuperfileEntry>, (Uuid, Option<Arc<RoaringBitmap>>))> = units
            .into_iter()
            .filter(|(entry, _)| admitted.contains(&entry.uri))
            .map(|(entry, suid)| {
                let within = scope
                    .allow
                    .as_ref()
                    .and_then(|allow| allow.get(&entry.uri).cloned());
                (entry, (suid, within))
            })
            .collect();
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         (suid, within): (Uuid, Option<Arc<RoaringBitmap>>)| {
            let kernel = Arc::clone(&kernel);
            async move {
                let mut rows: RoaringBitmap = kernel
                    .docs(&r, suid)
                    .await?
                    .into_iter()
                    .map(RowId::get)
                    .collect();
                match within {
                    Some(within) => rows &= within.as_ref(),
                    None => {
                        if let Some(cache) = cache.as_ref()
                            && let Some(deny) =
                                dispatch::tombstone_deny_set(cache, entry.superfile_id, now)?
                        {
                            rows -= deny.as_ref();
                        }
                    }
                }
                Ok((entry, rows))
            }
        };
        let matched = dispatch::fanout_with(self, units, true, self.open_intent(), body).await?;
        let mut superfiles = Vec::with_capacity(matched.len());
        let mut allow = HashMap::with_capacity(matched.len());
        for (entry, rows) in matched {
            if !rows.is_empty() {
                allow.insert(entry.uri, Arc::new(rows));
                superfiles.push(entry);
            }
        }
        Ok(CandidateScope {
            superfiles,
            allow: Some(allow),
        })
    }

    /// What every unranked match walk starts from: the superfiles the
    /// query's terms can match (`None` when there are none) and the kernel
    /// that finds the matching rows in any one of them.
    async fn match_kernel(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
    ) -> Result<Option<(Vec<(Arc<SuperfileEntry>, Uuid)>, Arc<MatchKernel>)>, QueryError> {
        let phases = self.phase_spans();
        let select_span = trace::phase(phases, || {
            detail_span!(
                "fts.select_superfiles",
                manifest_superfiles = self.manifest().superfiles.len(),
                survivors = tracing::field::Empty,
            )
        });
        let (match_set, negatives, kept) = self
            .parse_and_prune(column, query, mode)
            .instrument(select_span.clone())
            .await?;
        select_span.record("survivors", kept.len());
        trace::end(select_span);
        if kept.is_empty() {
            return Ok(None);
        }
        // Every plain term the kernel resolves, positive and negated: the
        // index's locations for them let each superfile skip its dictionary.
        let all_terms: Vec<&str> = match_set
            .terms
            .iter()
            .chain(negatives.terms.iter())
            .map(String::as_str)
            .collect();
        let locations = self
            .index_locations(column, &all_terms, &kept)
            .instrument(trace::phase(phases, || {
                detail_span!("fts.term_index", terms = all_terms.len())
            }))
            .await?;
        let units: Vec<(Arc<SuperfileEntry>, Uuid)> = kept
            .into_iter()
            .map(|e| {
                let id = e.superfile_id;
                (e, id)
            })
            .collect();
        let kernel = MatchKernel {
            column_field_id: self.manifest().field_id(column),
            column: column.to_owned(),
            mode: match_set.mode,
            phrase_involved: match_set.has_phrases() || !negatives.phrases.is_empty(),
            has_negatives: !negatives.is_empty(),
            terms: match_set.terms,
            phrases: match_set.phrases,
            negatives: negatives.terms,
            negative_phrases: negatives.phrases,
            locations,
            op_stats: self.op_stats.clone(),
        };
        Ok(Some((units, Arc::new(kernel))))
    }

    /// A single term's count, straight from the term index, or `None` when
    /// that cannot be exact.
    ///
    /// The index already records a per-superfile document frequency for every
    /// term it routes, so counting one term is summing numbers it holds — no
    /// superfile need be opened at all. Four conditions have to hold for that
    /// sum to be the same answer the fan-out would give:
    ///
    /// - **One bare term.** Phrases need positions, and several terms need the
    ///   union or intersection of their doc sets, neither of which is a sum.
    /// - **No negations.** An excluded term removes docs the df still counts.
    /// - **Every surviving superfile is indexed.** One the index does not list
    ///   contributes a df nobody recorded.
    /// - **None of them has a tombstone sidecar.** The recorded df is GROSS, so
    ///   a deleted doc is still in it.
    ///
    /// Any of those failing falls through to the fan-out, which is always
    /// correct and merely slower. An index that cannot be read is an error.
    async fn count_from_term_index(
        &self,
        column: &str,
        match_set: &UnrankedMatchSet,
        negatives: &UnrankedNegatives,
        kept: &[Arc<SuperfileEntry>],
    ) -> Result<Option<u64>, QueryError> {
        if match_set.has_phrases() || !negatives.is_empty() || match_set.terms.len() != 1 {
            return Ok(None);
        }
        let manifest = self.manifest();
        let Some(index) = manifest.term_index().await? else {
            return Ok(None);
        };
        if !kept.iter().all(|e| index.is_indexed(&e.superfile_id)) {
            return Ok(None);
        }
        // A superfile listed here may hold deleted rows, which the gross df
        // would still count.
        if let Some(seqs) = manifest.get_tombstone_seqs()
            && kept.iter().any(|e| seqs.contains_key(&e.superfile_id))
        {
            return Ok(None);
        }

        let (Some(term), Some(column_id)) = (match_set.terms.first(), manifest.field_id(column))
        else {
            return Ok(None);
        };
        let postings = index.postings(column_id, term).await?;
        let wanted: HashSet<Uuid> = kept.iter().map(|e| e.superfile_id).collect();
        let mut total: u64 = 0;
        for posting in postings.iter() {
            let Some(id) = index.superfile_id(posting.superfile) else {
                return Ok(None);
            };
            if wanted.contains(&id) {
                let Some(sum) = total.checked_add(posting.df) else {
                    return Ok(None);
                };
                total = sum;
            }
        }
        Ok(Some(total))
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
    /// Two fast paths, tried in that order. A single bare term over
    /// delete-free superfiles the term index lists is a sum of the
    /// document frequencies the index already holds, and opens nothing at
    /// all ([`Self::count_from_term_index`]). Failing that, a single-token
    /// query against a superfile with no tombstones resolves from the term
    /// dictionary's stored document frequency
    /// ([`SuperfileReader::term_df`]) — O(1) per superfile, no posting
    /// decode. A multi-token query, or a superfile with deletes, falls back
    /// to materializing the matching local doc ids and counting those not
    /// tombstoned. Tombstoned (deleted) rows are always excluded so the
    /// count matches what a search would return.
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

        if let Some(total) = self
            .count_from_term_index(column, &match_set, &negatives, &kept)
            .await?
        {
            return Ok(total);
        }

        let match_mode = match_set.mode;
        let single_term = match_set.terms.len() == 1 && !match_set.has_phrases();
        let has_negatives = !negatives.is_empty();
        let phrase_involved = match_set.has_phrases() || !negatives.phrases.is_empty();
        let all_terms: Vec<&str> = match_set
            .terms
            .iter()
            .chain(negatives.terms.iter())
            .map(String::as_str)
            .collect();
        let locations = self.index_locations(column, &all_terms, &kept).await?;
        let column_field_id = self.manifest().field_id(column);
        let column_arc = Arc::new(column.to_owned());
        let term_arc: Arc<Vec<String>> = Arc::new(match_set.terms);
        let phrase_arc: Arc<Vec<Phrase<String>>> = Arc::new(match_set.phrases);
        let neg_arc: Arc<Vec<String>> = Arc::new(negatives.terms);
        let neg_ph_arc: Arc<Vec<Phrase<String>>> = Arc::new(negatives.phrases);
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();

        // Shared fan-out (`dispatch::fanout_with`): warms tombstones,
        // spawns + opens each superfile concurrently, and short-circuits
        // on the first error. The per-superfile body returns this
        // superfile's match count; the totals are summed.
        let op_stats = self.op_stats.clone();
        let per_superfile = dispatch::fanout_with(
            self,
            units,
            true,
            self.open_intent(),
            move |r, entry, tombstone_cache, now, _params: ()| {
                let op_stats = op_stats.clone();
                let column_arc = Arc::clone(&column_arc);
                let term_arc = Arc::clone(&term_arc);
                let phrase_arc = Arc::clone(&phrase_arc);
                let neg_arc = Arc::clone(&neg_arc);
                let neg_ph_arc = Arc::clone(&neg_ph_arc);
                let locations = Arc::clone(&locations);
                async move {
                    // A file written before a rename labels the column as it was
                    // then, and its dictionary is keyed by that label; the id is
                    // what finds the column in either file.
                    let column_arc = r.column_alias(column_field_id, &column_arc).to_owned();

                    let memo = memo_from_locations(&r, &locations, entry.superfile_id).await;
                    // Tombstone bitmap for this superfile (None = no deletes).
                    let tomb = match tombstone_cache.as_ref() {
                        Some(c) => {
                            let b = c
                                .bitmap_for(entry.superfile_id, now)
                                .map_err(QueryError::tombstone_cache)?;
                            if b.is_empty() { None } else { Some(b) }
                        }
                        None => None,
                    };
                    let refs: Vec<&str> = term_arc.iter().map(|s| s.as_str()).collect();
                    // Negated terms or deletes both force materialization:
                    // Deletes force materialization: a tombstone bitmap can
                    // only be subtracted from an explicit id set, so when this
                    // superfile has deletes we materialize the positive matches
                    // and drop any doc carrying a negated term (union of the
                    // negatives) or a tombstone. Negation *without* deletes
                    // takes the skip-based counting path below instead.
                    if tomb.is_some() {
                        let (docs, mut work) = match phrase_involved {
                            true => {
                                r.atoms_match_ids(&column_arc, &refs, &phrase_arc, match_mode)
                                    .await?
                            }
                            false => {
                                r.token_match_prefetched(
                                    &column_arc,
                                    &refs,
                                    match_mode,
                                    memo.as_deref(),
                                )
                                .await?
                            }
                        };
                        let excluded: RoaringBitmap = if has_negatives {
                            let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                            let (neg_docs, neg_work) = match neg_ph_arc.is_empty() {
                                true => {
                                    r.token_match_prefetched(
                                        &column_arc,
                                        &neg_refs,
                                        BoolMode::Or,
                                        memo.as_deref(),
                                    )
                                    .await?
                                }
                                false => {
                                    r.atoms_match_ids(
                                        &column_arc,
                                        &neg_refs,
                                        &neg_ph_arc,
                                        BoolMode::Or,
                                    )
                                    .await?
                                }
                            };
                            work.merge(neg_work);
                            neg_docs.into_iter().map(RowId::get).collect()
                        } else {
                            RoaringBitmap::new()
                        };
                        if let Some(stats) = &op_stats {
                            stats.add_fts_postings_bytes(work.postings_bytes);
                            stats.add_planned_read_ranges(work.planned_ranges);
                            stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                        }
                        let n = docs
                            .iter()
                            .filter(|d| {
                                !excluded.contains(d.get())
                                    && tomb.as_ref().is_none_or(|b| !b.contains(d.get()))
                            })
                            .count() as u64;
                        return Ok::<u64, QueryError>(n);
                    }
                    // No deletes (the common case): count without
                    // materializing ids.
                    let (n, work) = if has_negatives {
                        // Negation, delete-free: walk the positive atoms and
                        // skip-exclude the negated ones — the negated union is
                        // never materialized. Covers term-only and phrase
                        // positives alike.
                        let neg_refs: Vec<&str> = neg_arc.iter().map(|s| s.as_str()).collect();
                        r.atoms_match_count(
                            &column_arc,
                            &refs,
                            &phrase_arc,
                            match_mode,
                            &neg_refs,
                            &neg_ph_arc,
                        )
                        .await?
                    } else if single_term {
                        // A single token resolves O(1) from the stored df.
                        r.term_df(&column_arc, &term_arc[0]).await?
                    } else if phrase_involved {
                        r.atoms_match_count(&column_arc, &refs, &phrase_arc, match_mode, &[], &[])
                            .await?
                    } else {
                        // Multi-token AND/OR tallies through the counting sink.
                        r.token_match_count_prefetched(
                            &column_arc,
                            &refs,
                            match_mode,
                            memo.as_deref(),
                        )
                        .await?
                    };
                    if let Some(stats) = &op_stats {
                        stats.add_fts_postings_bytes(work.postings_bytes);
                        stats.add_planned_read_ranges(work.planned_ranges);
                        stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                    }
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
        // `exact_match` prunes through the column's own term dictionary, so
        // a column with no full-text index has nothing to prune with.
        let Some(tokenizer) = manifest.try_fts_tokenizer_for(column) else {
            return Err(QueryError::InvalidQuery(no_fts_index_message(
                column,
                &manifest.fts_configs(),
            )));
        };
        let term_strings: Vec<String> = tokenizer.tokenize(value).collect();
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
        let phases = self.phase_spans();
        let select_span = trace::phase(phases, || {
            detail_span!(
                "fts.select_superfiles",
                manifest_superfiles = manifest.superfiles.len(),
                survivors = tracing::field::Empty,
            )
        });
        let kept = select_fts_superfiles(manifest.as_ref(), &leaves, column)
            .instrument(select_span.clone())
            .await?;
        select_span.record("survivors", kept.len());
        trace::end(select_span);
        if kept.is_empty() {
            return Ok(Vec::new());
        }
        let token_refs: Vec<&str> = term_strings.iter().map(String::as_str).collect();
        let locations = self
            .index_locations(column, &token_refs, &kept)
            .instrument(trace::phase(phases, || {
                detail_span!("fts.term_index", terms = token_refs.len())
            }))
            .await?;
        let units: Vec<(Arc<SuperfileEntry>, ())> = kept.into_iter().map(|e| (e, ())).collect();
        let column_field_id = self.manifest().field_id(column);
        let column_arc = Arc::new(column.to_owned());
        let value_arc = Arc::new(value.to_owned());
        let tokens_arc = Arc::new(term_strings);
        let op_stats = self.op_stats.clone();
        let reader_pool = Arc::clone(&self.manifest().options.reader_pool);
        let body = move |r: Arc<SuperfileReader>,
                         entry: Arc<SuperfileEntry>,
                         tombstone_cache: Option<Arc<SidecarCache>>,
                         now: Instant,
                         _: ()| {
            let column_arc = Arc::clone(&column_arc);
            let value_arc = Arc::clone(&value_arc);
            let tokens_arc = Arc::clone(&tokens_arc);
            let locations = Arc::clone(&locations);
            let op_stats = op_stats.clone();
            let reader_pool = Arc::clone(&reader_pool);
            async move {
                // A file written before a rename labels the column as it was
                // then, and its dictionary is keyed by that label; the id is
                // what finds the column in either file.
                let column_arc = r.column_alias(column_field_id, &column_arc).to_owned();

                let candidates: Vec<u32> = if tokens_arc.is_empty() {
                    (0..r.n_docs() as u32).collect()
                } else {
                    let memo = memo_from_locations(&r, &locations, entry.superfile_id).await;
                    let refs: Vec<&str> = tokens_arc.iter().map(String::as_str).collect();
                    let (docs, work) = r
                        .token_match_prefetched(&column_arc, &refs, BoolMode::And, memo.as_deref())
                        .await?;
                    // The prune pass's posting walk. The verify pass's own
                    // decode is folded into `rows_materialized` below; its
                    // byte and range legs are deliberately unpriced — both
                    // take paths report no planned ranges by design, so the
                    // counter stays identical warm or cold.
                    if let Some(stats) = &op_stats {
                        stats.add_fts_postings_bytes(work.postings_bytes);
                        stats.add_planned_read_ranges(work.planned_ranges);
                        stats.add_kernel_cpu_ns(work.kernel_cpu_ns);
                    }
                    docs.into_iter().map(RowId::get).collect()
                };
                if candidates.is_empty() {
                    return Ok(Vec::new());
                }
                // The verify pass — candidate decode + string compare — is
                // this query's dominant CPU (a token-less value decodes the
                // whole column), so it is bracketed like any other kernel.
                // Only the warm take is inside this bracket; the cold arm's
                // decode is not separable from the fetch it is interleaved
                // with, which the comment on that arm explains.
                let warm_batch = op_stats::timed_kernel(&op_stats, || {
                    if r.can_take_by_local_doc_ids() {
                        r.take_by_local_doc_ids(&candidates, &[column_arc.as_str()])
                            .map(Some)
                            .map_err(QueryError::from)
                    } else {
                        Ok(None)
                    }
                })?;
                let batch = match warm_batch {
                    Some(batch) => batch,
                    // Cold: the fetch and its Parquet decode are interleaved
                    // inside the async reader, so the decode leg is not
                    // separable here and goes uncharged. Bracketing the await
                    // itself would be worse than leaving it at zero — a thread
                    // clock spanning an await bills whatever else the runtime
                    // ran on this thread to this query. The verify comparison
                    // below is charged on both arms.
                    None => take_rows_byte_source(
                        &r,
                        &candidates,
                        &[column_arc.as_str()],
                        Some(&reader_pool),
                    )
                    .await
                    .map_err(QueryError::DataFusion)?,
                };
                // The verify decode materialized one row per candidate,
                // on either arm. Folding it here rather than per-arm keeps
                // the count invariant to which take served the batch.
                if let Some(stats) = &op_stats {
                    stats.add_rows_materialized(candidates.len() as u64);
                }
                let hits = op_stats::timed_kernel(&op_stats, || {
                    let values = batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| {
                            QueryError::Internal(format!(
                                "exact_match column '{}' is not LargeUtf8",
                                column_arc
                            ))
                        })?;
                    Ok::<_, QueryError>(
                        candidates
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| {
                                !values.is_null(*index)
                                    && values.value(*index) == value_arc.as_str()
                            })
                            .map(|(_, &local_doc_id)| SuperfileHit {
                                superfile: entry.uri,
                                local_doc_id,
                                score: 0.0,
                                stable_id: None,
                            })
                            .collect::<Vec<SuperfileHit>>(),
                    )
                });
                let mut hits: Vec<SuperfileHit> = hits?;
                dispatch::apply_tombstone_filter(tombstone_cache.as_ref(), &entry, &mut hits, now)?;
                Ok(hits)
            }
        };
        let fanout_span = trace::phase(phases, || tiered_span!("fts.fanout", units = units.len()));
        let per_unit = dispatch::fanout_with(self, units, true, self.open_intent(), body)
            .instrument(fanout_span)
            .await?;
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
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, QueryError> {
        self.check_projection(projection)?;

        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(async {
            let hits = self.bm25_search_async(column, query, k, opts).await?;
            // `projection` selects columns by name (any of `_id`, the
            // visible scalar columns, or the trailing `score`); `None`
            // returns `_id` + `score` only. The shared resolver decodes
            // only the projected columns.
            let batch = resolve_hits_named(self, &hits, projection)
                .instrument(detail_span!("search.resolve", hits = hits.len()))
                .await?;
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
    ///
    /// Takes the same [`Bm25SearchOptions`] as
    /// [`bm25_search`](Self::bm25_search).
    pub fn bm25_hits(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_search_async(column, query, k, opts))
    }

    /// [`Self::bm25_hits`] scoring every match in one pass: the top-k
    /// hits and the exact match count, deleted rows excluded. Nothing is
    /// pruned, so this costs more than `bm25_hits` plus `count`; it is for
    /// a harness that needs both from one walk, and like the rest of this
    /// reader it is reachable only under `test-helpers`.
    ///
    /// The top-k is `bm25_hits`'s up to float summation order: the two
    /// paths add a doc's term scores in different orders, so a score can
    /// differ in its last bits and two docs tied at the k-th score within
    /// that error may swap. Takes the same [`Bm25SearchOptions`];
    /// `score_tolerance` has no effect here.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn bm25_hits_with_count(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
    ) -> Result<(Vec<SuperfileHit>, u64), QueryError> {
        let _foreground = ForegroundQueryGuard::enter();
        self.block_on(self.bm25_hits_with_count_async(column, query, k, opts))
    }

    /// The async kernel behind [`Self::bm25_hits_with_count`]. With
    /// `k == 0` there is nothing to rank, so the count comes from the
    /// unranked count path.
    #[cfg(any(test, feature = "test-helpers"))]
    pub(crate) async fn bm25_hits_with_count_async(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
    ) -> Result<(Vec<SuperfileHit>, u64), QueryError> {
        if k == 0 {
            let count = self
                .token_match_count_async(column, query, opts.mode)
                .await?;
            return Ok((Vec::new(), count));
        }
        self.bm25_scored_async(column, query, k, opts, None, true)
            .await
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
    /// materialization. A single bare term over delete-free superfiles the
    /// term index lists is answered by summing the document frequencies it
    /// records, without opening any superfile; otherwise a single-token
    /// query on a delete-free superfile resolves in O(1) from the stored
    /// document frequency. Drives the async kernel via the sync→async
    /// bridge.
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

/// The cursors one superfile's doc-id slices share, keyed by superfile id
/// and built by the first slice to need them; the `OnceCell` coalesces
/// concurrent slices of a file.
type SharedCursorSets = Arc<Mutex<HashMap<Uuid, SharedCursorCell>>>;

/// One superfile's slot in [`SharedCursorSets`].
type SharedCursorCell = Arc<OnceCell<Arc<OrCursorSet>>>;

/// Superfile `suid`'s slot in `sets`, made empty by the first slice to
/// ask; the slice then builds or awaits the set through it.
fn cursor_cell(sets: &SharedCursorSets, suid: Uuid) -> SharedCursorCell {
    let mut sets = sets.lock().expect("cursor-set map lock poisoned");
    Arc::clone(sets.entry(suid).or_default())
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

/// Unwrap FTS hits out of row space for the shared per-superfile hit
/// shape.
///
/// A [`SuperfileHit`] carries a bare local id because the vector path
/// fills the same field from its own numbering, which is not the
/// superfile's Parquet rows. FTS hits *are* rows by the time they leave
/// the reader, so the type comes off here, at the one place they enter
/// that shape.
fn rows_as_local_ids(hits: Vec<(RowId, f32)>) -> Vec<(u32, f32)> {
    hits.into_iter().map(|(row, s)| (row.get(), s)).collect()
}

/// The superfiles a full-text query on `column` fans out to: the ones the
/// prune `leaves` keep, minus any whose file does not hold the column —
/// written before it was added — which contribute nothing rather than
/// failing the query. A file written before field ids is taken to hold
/// every column the table had then.
async fn select_fts_superfiles(
    manifest: &ManifestSnapshot,
    leaves: &[PruneLeaf],
    column: &str,
) -> Result<Vec<Arc<SuperfileEntry>>, QueryError> {
    let mut kept = select_superfiles(manifest, leaves).await?;
    if let Some(id) = manifest.field_id(column) {
        kept.retain(|entry| entry.holds_fts_column(id));
    }
    Ok(kept)
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
/// Sub-range count is allocated by **doc mass**: a superfile holding
/// `f` of the surviving docs gets `round(f × pool_threads)` slices.
/// Splitting the pool evenly per *file* instead leaves a compacted
/// table — one large merged superfile plus small remnants — with the
/// same one-or-two units the merged file had when it was dozens of
/// balanced files, so most of the pool idles on remnants while a
/// couple of threads walk nearly the whole corpus. Slices share one
/// cursor build per superfile (see the fan-out's cursor-set cache), so
/// extra units cost decode buffers, not postings fetches.
///
/// `pool_threads` is a target, not a budget: per-file round-half-up
/// plus the ≥ 1-unit clamp can emit up to `kept − 1` units more than
/// there are threads (e.g. three equal files on an 8-thread pool yield
/// 3 × 3 = 9). Excess units queue on the pool — scheduling slop, never
/// extra concurrency.
///
/// Two limits still apply:
///   1. `FanOut::PerSuperfile` (no range-aware kernel for the shape)
///      and a single-threaded pool both collapse to one un-ranged unit
///      per superfile — the original `par_iter` over superfiles shape.
///   2. No slice is narrower than `SUBRANGE_MIN_DOCS`; below that, BMM
///      bookkeeping + the cross-sub-range top-K merge dominate the
///      parallel win.
fn build_work_units(
    kept: &[&Arc<SuperfileEntry>],
    fanout: FanOut,
    pool_threads: usize,
) -> Vec<WorkUnit> {
    let un_ranged = |entry: &Arc<SuperfileEntry>| WorkUnit {
        entry: Arc::clone(entry),
        range: None,
    };
    let total_docs: u64 = kept.iter().map(|e| e.n_docs).sum();
    if matches!(fanout, FanOut::PerSuperfile) || pool_threads <= 1 || total_docs == 0 {
        return kept.iter().map(|e| un_ranged(e)).collect();
    }

    let mut units: Vec<WorkUnit> = Vec::with_capacity(kept.len() + pool_threads);
    for entry in kept {
        let n_docs = entry.n_docs as u32;
        if n_docs == 0 {
            continue;
        }
        // Integer round-half-up of `n_docs / total_docs × pool_threads`.
        // A file holding ~all the docs asks for the whole pool; a
        // remnant holding ~none rounds to 0 and is clamped to one
        // whole-file unit.
        let by_mass = ((entry.n_docs * pool_threads as u64 + total_docs / 2) / total_docs) as usize;
        let cap_by_floor = (n_docs / SUBRANGE_MIN_DOCS).max(1) as usize;
        let n_sub = by_mass.clamp(1, cap_by_floor);
        if n_sub <= 1 {
            units.push(un_ranged(entry));
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

/// Ties at the k-th score more than this many times the places left for
/// them are narrowed by `_id` before every tie's `_id` is resolved.
const TIE_NARROW_MIN_FACTOR: usize = 4;
/// Superfiles of tied candidates whose `_id`s one step resolves together.
pub(crate) const TIE_NARROW_BATCH: usize = 16;

/// `cands` (scoring at least `kth_score`, the k-th best) with the hits tied
/// at `kth_score` cut to the ones that can place: the smallest `_id`s among
/// them, as many as the places the strictly better hits leave. When the
/// ties far outnumber those places — a query whose every match scores the
/// same, over thousands of superfiles — resolving every tie's `_id` cost a
/// sidecar read per candidate. Instead the ties' superfiles are taken in
/// ascending `id_min`, a batch at a time, and taking stops once the next
/// superfile's `id_min` is above the largest `_id` still placing: every
/// `_id` of a superfile is at least its `id_min`, so none of its ties
/// could place. The hits returned are exactly those the full resolution
/// would keep.
async fn narrow_ties_by_id(
    tr: &SupertableReader,
    cands: Vec<SuperfileHit>,
    k: usize,
    kth_score: f32,
    id_mins: &HashMap<SuperfileUri, i128>,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let (mut keep, ties): (Vec<SuperfileHit>, Vec<SuperfileHit>) =
        cands.into_iter().partition(|c| c.score > kth_score);
    let places = k.saturating_sub(keep.len());
    if places == 0 || ties.len() <= places.saturating_mul(TIE_NARROW_MIN_FACTOR) {
        keep.extend(ties);
        return Ok(keep);
    }
    let mut by_superfile: HashMap<SuperfileUri, Vec<SuperfileHit>> = HashMap::new();
    for hit in ties {
        by_superfile.entry(hit.superfile).or_default().push(hit);
    }
    // A superfile `id_mins` does not name sorts first, so it is always
    // resolved.
    let mut groups: Vec<(i128, Vec<SuperfileHit>)> = by_superfile
        .into_iter()
        .map(|(uri, hits)| (id_mins.get(&uri).copied().unwrap_or(i128::MIN), hits))
        .collect();
    groups.sort_unstable_by_key(|(id_min, _)| *id_min);
    let mut placing: Vec<SuperfileHit> = Vec::new();
    let mut groups = groups.into_iter().peekable();
    loop {
        // The largest `_id` still placing, once the places are full.
        let bound = (placing.len() >= places).then(|| {
            placing
                .iter()
                .filter_map(|h| h.stable_id)
                .max()
                .unwrap_or(i128::MAX)
        });
        let mut batch: Vec<SuperfileHit> = Vec::new();
        for _ in 0..TIE_NARROW_BATCH {
            match groups.next_if(|(id_min, _)| bound.is_none_or(|b| *id_min <= b)) {
                Some((_, hits)) => batch.extend(hits),
                None => break,
            }
        }
        if batch.is_empty() {
            break;
        }
        dispatch::attach_stable_ids_to_hits(tr, &mut batch).await?;
        placing.extend(batch);
        if placing.len() > places {
            placing.select_nth_unstable_by_key(places - 1, |h| h.stable_id);
            placing.truncate(places);
        }
    }
    keep.extend(placing);
    Ok(keep)
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
    id_mins: &HashMap<SuperfileUri, i128>,
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
        cands = narrow_ties_by_id(tr, cands, k, kth_score, id_mins).await?;
    }
    dispatch::attach_stable_ids_to_hits(tr, &mut cands).await?;
    // Total order: score desc, then stable `_id` asc — deterministic and
    // invariant across compaction (unlike physical superfile/offset keys).
    // Hits from a single unit already arrive by descending score, so only
    // the runs of tied scores need reordering by `_id`; anything else is
    // sorted whole. `total_cmp` orders a never-NaN score as `partial_cmp`
    // does, with an integer compare.
    let by_score_desc = cands.windows(2).all(|w| w[0].score >= w[1].score);
    match by_score_desc {
        true => {
            let mut i = 0;
            while i < cands.len() {
                let mut j = i + 1;
                while j < cands.len() && cands[j].score == cands[i].score {
                    j += 1;
                }
                if j - i > 1 {
                    cands[i..j].sort_unstable_by_key(|c| c.stable_id);
                }
                i = j;
            }
        }
        false => cands.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(a.stable_id.cmp(&b.stable_id))
        }),
    }
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
    /// # use infino::{connect, Bm25SearchOptions, IndexSpec};
    /// # let db = connect("memory://")?;
    /// # let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::LargeUtf8, false)]));
    /// # let posts = db.create_table("posts", schema.clone(), IndexSpec::new().fts("body"))?;
    /// # posts.append(&RecordBatch::try_new(
    /// #     schema, vec![Arc::new(LargeStringArray::from(vec!["the quick brown fox"]))])?)?;
    /// // Bare call → `_id` + `score`, no scalar decode:
    /// let hits = posts.bm25_search("body", "fox", 10, Bm25SearchOptions::new(), None)?;
    /// assert_eq!(hits[0].num_columns(), 2);
    /// // Name columns to materialize row data:
    /// let rows = posts.bm25_search("body", "fox", 10, Bm25SearchOptions::new(), Some(&["_id", "body", "score"]))?;
    /// assert_eq!(rows[0].num_columns(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[cfg_attr(
        feature = "detailed-tracing",
        // The empty fields are the read's outcome, filled once it has run;
        // see `CloseOut`.
        tracing::instrument(skip_all, fields(
            column = column,
            k = k,
            mode = ?opts.mode,
            role = self.role().as_str(),
            origin = OpOrigin::Query.as_str(),
            rows_out = tracing::field::Empty,
            kernel_cpu_ns = tracing::field::Empty,
            planned_read_ranges = tracing::field::Empty,
            fts_postings_bytes = tracing::field::Empty,
            tier_bytes = tracing::field::Empty,
            tier_entries = tracing::field::Empty,
            superfiles_resolved = tracing::field::Empty,
            rows_materialized = tracing::field::Empty,
            store_heads = tracing::field::Empty,
            store_gets = tracing::field::Empty,
            store_get_bytes = tracing::field::Empty,
            store_bg_gets = tracing::field::Empty,
            store_bg_get_bytes = tracing::field::Empty,
        ))
    )]
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, k, mode = ?opts.mode, "bm25_search");
        let close_out = self.close_out();
        let batches = self
            .reader()?
            .bm25_search(column, query, k, opts, projection)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("bm25_search", None))?;
        close_out.finish_batches(&batches);
        Ok(batches)
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
        // The empty fields are the read's outcome, filled once it has run;
        // see `CloseOut`.
        tracing::instrument(skip_all, fields(
            column = column,
            mode = ?mode,
            role = self.role().as_str(),
            origin = OpOrigin::Query.as_str(),
            rows_out = tracing::field::Empty,
            kernel_cpu_ns = tracing::field::Empty,
            planned_read_ranges = tracing::field::Empty,
            fts_postings_bytes = tracing::field::Empty,
            tier_bytes = tracing::field::Empty,
            tier_entries = tracing::field::Empty,
            superfiles_resolved = tracing::field::Empty,
            rows_materialized = tracing::field::Empty,
            store_heads = tracing::field::Empty,
            store_gets = tracing::field::Empty,
            store_get_bytes = tracing::field::Empty,
            store_bg_gets = tracing::field::Empty,
            store_bg_get_bytes = tracing::field::Empty,
        ))
    )]
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, mode = ?mode, "token_match");
        let close_out = self.close_out();
        let reader = self.reader()?;
        reader
            .check_projection(projection)
            .map_err(|e| InfinoError::from(e).with_context("token_match", None))?;
        let hits = reader
            .token_match(column, query, mode)
            .map_err(|e| InfinoError::from(e).with_context("token_match", None))?;
        let batch = self
            .block_on_query(
                resolve_hits_named(&reader, &hits, projection)
                    .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(|e| InfinoError::from(e).with_context("token_match", None))?;
        close_out.finish(batch.num_rows() as u64);
        Ok(vec![batch])
    }

    /// Unranked exact match: rows whose `column` value equals `value`
    /// exactly (index-pruned, then text-verified). Returns Arrow rows
    /// like [`Supertable::bm25_search`], with `score` fixed at `0.0` and
    /// unspecified row order. `projection` follows the same rules as
    /// `bm25_search`.
    #[cfg_attr(
        feature = "detailed-tracing",
        // The empty fields are the read's outcome, filled once it has run;
        // see `CloseOut`.
        tracing::instrument(skip_all, fields(
            column = column,
            role = self.role().as_str(),
            origin = OpOrigin::Query.as_str(),
            rows_out = tracing::field::Empty,
            kernel_cpu_ns = tracing::field::Empty,
            planned_read_ranges = tracing::field::Empty,
            fts_postings_bytes = tracing::field::Empty,
            tier_bytes = tracing::field::Empty,
            tier_entries = tracing::field::Empty,
            superfiles_resolved = tracing::field::Empty,
            rows_materialized = tracing::field::Empty,
            store_heads = tracing::field::Empty,
            store_gets = tracing::field::Empty,
            store_get_bytes = tracing::field::Empty,
            store_bg_gets = tracing::field::Empty,
            store_bg_get_bytes = tracing::field::Empty,
        ))
    )]
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(column, "exact_match");
        let close_out = self.close_out();
        let reader = self.reader()?;
        reader
            .check_projection(projection)
            .map_err(|e| InfinoError::from(e).with_context("exact_match", None))?;
        let hits = reader
            .exact_match(column, value)
            .map_err(|e| InfinoError::from(e).with_context("exact_match", None))?;
        let batch = self
            .block_on_query(
                resolve_hits_named(&reader, &hits, projection)
                    .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(|e| InfinoError::from(e).with_context("exact_match", None))?;
        close_out.finish(batch.num_rows() as u64);
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
        self.reader()?
            .count(column, query, mode)
            .map_err(InfinoError::from)
            .map_err(|e| e.with_context("count", None))
    }

    /// `text` as the full-text index on `column` tokenizes it: the terms the
    /// column's text was indexed under and a query over it is parsed into —
    /// the column's analyzer with its stopword and stemmer filters — in
    /// order, repeats kept. Two texts that share a token here are texts a
    /// token match on `column` finds together, so a caller judging one text
    /// against another — a question against the rows a query returned —
    /// asks this instead of comparing spellings; its own copy of the rule
    /// drifts from the index the moment a column is declared with another
    /// analyzer. `column` must carry a full-text index: without one there is
    /// no analyzer to tokenize with, and the error names the columns that
    /// have one.
    pub fn tokenize(&self, column: &str, text: &str) -> Result<Vec<String>, InfinoError> {
        let reader = self.reader()?;
        let manifest = reader.manifest();
        let Some(tokenizer) = manifest.try_fts_tokenizer_for(column) else {
            return Err(
                InfinoError::from(QueryError::InvalidQuery(no_fts_index_message(
                    column,
                    &manifest.fts_configs(),
                )))
                .with_context("tokenize", None),
            );
        };
        Ok(tokenizer.tokenize(text).collect())
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        future::Future,
        ops::Range,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Instant,
    };

    use crate::supertable::manifest::ScalarStats;

    /// A phrase's members are conjunctive, so a quoted phrase on its own
    /// prunes on all of them.
    ///
    /// Flattening them into the query's `Or` is what makes a phrase of common
    /// words prune nothing: `"wine beer"` would ask only whether a superfile
    /// holds `wine` or `beer`, which at corpus scale every superfile does, so
    /// the fan-out opens the entire table.
    #[test]
    fn a_lone_phrase_prunes_on_every_member() {
        let phrase = super::Phrase::adjacent(vec!["wine".to_string(), "beer".to_string()]);
        let leaf = super::presence_leaf(
            "text",
            &[],
            std::slice::from_ref(&phrase),
            super::BoolMode::Or,
        );
        let super::PruneLeaf::TermPresence { terms, mode, .. } = leaf else {
            panic!("expected a term-presence leaf");
        };
        assert_eq!(terms, vec!["wine".to_string(), "beer".to_string()]);
        assert!(
            matches!(mode, super::BoolMode::And),
            "a lone phrase requires every member, so the leaf must be And"
        );
    }

    /// The soundness boundary. `wine OR "beer stout"` matches a doc holding
    /// only `wine`, so requiring the phrase's members would prune away
    /// superfiles that genuinely match. One presence leaf cannot express
    /// `(a AND b) OR c`, so the union over every atom is the strongest thing
    /// that stays correct.
    #[test]
    fn a_phrase_beside_a_bare_term_stays_a_union() {
        let phrase = super::Phrase::adjacent(vec!["beer".to_string(), "stout".to_string()]);
        let leaf = super::presence_leaf(
            "text",
            &["wine".to_string()],
            std::slice::from_ref(&phrase),
            super::BoolMode::Or,
        );
        let super::PruneLeaf::TermPresence { mode, terms, .. } = leaf else {
            panic!("expected a term-presence leaf");
        };
        assert!(
            matches!(mode, super::BoolMode::Or),
            "a phrase OR a bare term must not become a conjunction"
        );
        assert_eq!(terms.len(), 3, "every atom still joins the union");
    }

    /// Two phrases under `Or` are `(a AND b) OR (c AND d)`, equally
    /// inexpressible, so they also stay a union.
    #[test]
    fn two_phrases_under_or_stay_a_union() {
        let phrases = vec![
            super::Phrase::adjacent(vec!["wine".to_string(), "beer".to_string()]),
            super::Phrase::adjacent(vec!["gin".to_string(), "tonic".to_string()]),
        ];
        let leaf = super::presence_leaf("text", &[], &phrases, super::BoolMode::Or);
        let super::PruneLeaf::TermPresence { mode, terms, .. } = leaf else {
            panic!("expected a term-presence leaf");
        };
        assert!(matches!(mode, super::BoolMode::Or));
        assert_eq!(terms.len(), 4);
    }

    /// A must-side prune already requires every atom, so a phrase there needs
    /// no special case — and must not lose the conjunction either.
    #[test]
    fn a_must_side_phrase_keeps_its_conjunction() {
        let phrase = super::Phrase::adjacent(vec!["beer".to_string(), "stout".to_string()]);
        let leaf = super::presence_leaf(
            "text",
            &["wine".to_string()],
            std::slice::from_ref(&phrase),
            super::BoolMode::And,
        );
        let super::PruneLeaf::TermPresence { mode, terms, .. } = leaf else {
            panic!("expected a term-presence leaf");
        };
        assert!(matches!(mode, super::BoolMode::And));
        assert_eq!(terms.len(), 3, "musts and phrase members all required");
    }

    /// The skip decision at its boundary: a ceiling equal to the floor is
    /// opened, one an ulp below it is skipped, and nothing is skipped before
    /// a floor exists. Pinned here because the end-to-end tie test cannot
    /// construct an exactly-equal ceiling — the term index widens every
    /// ceiling it hands out.
    #[test]
    fn a_ceiling_equal_to_the_floor_still_competes() {
        let floor = 0.20840901f32;
        assert!(super::ceiling_can_compete(floor, floor));
        assert!(!super::ceiling_can_compete(floor.next_down(), floor));
        assert!(super::ceiling_can_compete(floor.next_up(), floor));
        assert!(super::ceiling_can_compete(0.0, f32::NEG_INFINITY));
        assert!(super::ceiling_can_compete(f32::INFINITY, f32::MAX));
    }

    /// Exact scoring must keep the comparison it had before a tolerance
    /// existed, not something that merely rounds to it. The floor is the
    /// right-hand side of a float comparison, so "close enough" is not a
    /// property: one ulp decides whether a superfile is read.
    #[test]
    fn a_zero_tolerance_leaves_the_floor_bit_for_bit() {
        let floors = [
            0.0f32,
            -0.0,
            f32::MIN_POSITIVE,
            1.0e-30,
            0.20840901,
            1.0,
            12345.678,
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        for floor in floors {
            assert_eq!(
                super::competitive_floor(floor, 0.0).to_bits(),
                floor.to_bits(),
                "a zero tolerance changed the floor {floor}"
            );
        }
    }

    /// A tolerance raises the bar a superfile's ceiling has to clear, which
    /// is the whole mechanism: a ceiling that would have been read because it
    /// ties the k-th score is skipped once the caller accepts a shortfall.
    /// A superfile whose ceiling ties the floor is skippable once `k`
    /// scores are known and every one at the floor came from superfiles
    /// below its `_id` range; a seed placeholder, or a floor score from a
    /// superfile at or above that range, keeps it open.
    /// Merging many scores at once (a sorted merge) keeps the same top-k
    /// as inserting a few at a time, and a floor the walks have raised
    /// drops what it already excludes.
    #[test]
    fn the_shared_floor_merges_many_scores_as_it_inserts_few() {
        let k = 40;
        let mut next = crate::test_helpers::xorshift(7);
        let mut draw =
            |n: usize| -> Vec<f32> { (0..n).map(|_| (next() % 10_000) as f32 / 100.0).collect() };
        let (many, few) = (SharedTopK::new(k), SharedTopK::new(k));
        let mut reference: Vec<f32> = Vec::new();
        for (id_max, batch) in [(100i128, 200usize), (200, 7), (300, 150), (400, 3)] {
            let scores = draw(batch);
            reference.extend(&scores);
            many.merge(id_max, scores.iter().copied());
            for s in &scores {
                few.merge(id_max, [*s]);
            }
            reference.sort_by(|a, b| b.total_cmp(a));
            reference.truncate(k);
            for (label, shared) in [("many", &many), ("few", &few)] {
                let top: Vec<f32> = shared
                    .top
                    .lock()
                    .expect("lock")
                    .iter()
                    .map(|e| e.score)
                    .collect();
                assert_eq!(top, reference, "{label} after id_max {id_max}");
                assert_eq!(shared.floor(), reference[k - 1], "{label} floor");
            }
        }
        // A walk raised the live floor past the k-th: a score under it is
        // dropped before the lock, one equal to it still enters.
        let raised = reference[k / 2];
        let scores_of = |shared: &SharedTopK| -> Vec<f32> {
            shared
                .top
                .lock()
                .expect("lock")
                .iter()
                .map(|e| e.score)
                .collect()
        };
        let at_raised = scores_of(&many).iter().filter(|s| **s == raised).count();
        many.live_floor().raise(raised);
        many.merge(500, [raised - 0.005, raised, raised + 0.005]);
        let top = scores_of(&many);
        assert_eq!(top.len(), k);
        assert!(
            top.contains(&(raised + 0.005)),
            "above the raised floor enters"
        );
        assert_eq!(
            top.iter().filter(|s| **s == raised).count(),
            at_raised + 1,
            "equal to the raised floor enters"
        );
        assert!(!top.contains(&(raised - 0.005)), "under it is dropped");
    }

    #[test]
    fn a_tie_at_the_floor_is_skipped_only_behind_lower_ids() {
        let shared = SharedTopK::new(3);
        assert!(!shared.tie_skippable(1.0, 100), "no scores known yet");
        shared.merge(300, [2.0, 1.0, 1.0]);
        assert_eq!(shared.floor(), 1.0);
        assert!(
            !shared.tie_skippable(1.0, 100),
            "ties from a superfile reaching id 300 are not settled for id 100"
        );
        assert!(
            shared.tie_skippable(1.0, 400),
            "but they are for one past it"
        );
        // Equal scores from a lower range take the tied places.
        shared.merge(50, [1.0, 1.0]);
        assert!(
            shared.tie_skippable(1.0, 100),
            "both ties now sit below id 100"
        );
        assert!(
            !shared.tie_skippable(1.0, 50),
            "a tie at id 50 is not below id 50"
        );
        let seeded = SharedTopK::new(2);
        seeded.merge(i128::MAX, [1.0, 1.0]);
        assert!(
            !seeded.tie_skippable(1.0, 100),
            "a seed is no document and justifies no skip"
        );
        seeded.merge(10, [1.0, 1.0]);
        assert!(
            seeded.tie_skippable(1.0, 100),
            "real scores equal to the seeds replace them"
        );
    }

    #[test]
    fn a_tolerance_raises_the_bar() {
        let floor = 100.0f32;
        assert!(
            super::ceiling_can_compete(floor, floor),
            "exact reads a tie"
        );

        let raised = super::competitive_floor(floor, 0.01);
        assert!(raised > floor, "one percent must raise a positive floor");
        assert!(
            !super::ceiling_can_compete(floor, raised),
            "the tie is now skipped"
        );
        assert!(
            super::ceiling_can_compete(floor * 1.02, raised),
            "a ceiling clearing the tolerance is still read"
        );
    }

    /// A tolerance that means nothing falls back to exact rather than
    /// propagating into the comparison. The failure it prevents is silent:
    /// a NaN threshold makes every comparison incomparable, and the
    /// surrounding predicate opens on incomparable, so the query would stay
    /// correct while reading every superfile for no stated reason.
    #[test]
    fn a_meaningless_tolerance_is_exact() {
        let floor = 7.5f32;
        for tolerance in [-0.0f32, -0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                super::competitive_floor(floor, tolerance).to_bits(),
                floor.to_bits(),
                "tolerance {tolerance} should have been treated as exact"
            );
        }
    }

    /// Below zero the tolerance widens instead of tightening, and that is the
    /// safe direction: a lower bar opens superfiles that need not be opened,
    /// which costs reads and never results. Pinned so the asymmetry is a
    /// recorded decision rather than a surprise, since the running floor
    /// starts at negative infinity and a scorer may yield negative scores.
    #[test]
    fn a_negative_floor_widens_rather_than_tightens() {
        assert_eq!(
            super::competitive_floor(f32::NEG_INFINITY, 0.05),
            f32::NEG_INFINITY,
            "the starting floor admits everything, tolerance or not"
        );
        let floor = -4.0f32;
        let widened = super::competitive_floor(floor, 0.05);
        assert!(widened < floor, "a negative floor moves down, not up");
        assert!(
            super::ceiling_can_compete(floor, widened),
            "so a ceiling at the floor is still read"
        );
    }

    use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use async_trait::async_trait;
    use bytes::Bytes;
    use datafusion::prelude::{col, lit};
    use object_store::MultipartUpload;
    use tempfile::TempDir;
    use tokio::runtime::Builder;
    use uuid::Uuid;

    use super::{
        BoolMode, FanOut, SharedTopK, SuperfileHit, SupertableReader, build_work_units,
        ceilings_at_table_avgdl, fanout_for,
    };
    use crate::{
        config::{CompactionSettings, OptimizeOptions},
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            fts::{
                bm25,
                posting::BLOCK_LEN,
                reader::{Bm25SearchOptions, ColumnLengthStats, top_k_initial_capacity},
            },
            vector::layout::VectorLayout,
        },
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            manifest::{
                FtsSummaryAgg, SuperfileEntry, SuperfileUri,
                commit::{MANIFEST_PARTS_DIR, part_uri, read_pointer},
                list,
                part::{self, ContentHash},
            },
            schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE, FieldId},
        },
        test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
    };

    /// Manifest entry fixture for the work-unit tests. `n_docs` is the
    /// only field the fan-out's slicing reads; everything else is inert.
    fn manifest_entry(n_docs: u64) -> Arc<SuperfileEntry> {
        let id = Uuid::new_v4();
        Arc::new(SuperfileEntry {
            physical_schema: None,
            stem: None,
            birth_version: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs,
            id_min: 0,
            id_max: n_docs.saturating_sub(1) as i128,
            scalar_stats: ScalarStats::default(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

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
        SupertableOptions::new(schema_id_title(), vec![FtsConfig::new("title")], vec![])
            .expect("valid options")
            .with_writer_pool(pool)
    }

    fn build_batch(_start: u64, titles: &[&str]) -> RecordBatch {
        let titles_arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles_arr)]).expect("batch")
    }

    /// All `(title, score)` hits for a bm25_search, in ranked order.
    ///
    /// Projects the `title` column rather than `_id`: the
    /// supertable-injected `_id` embeds superfile/commit identity, so it
    /// is NOT comparable across two independently-built tables. The doc
    /// content is. `k` is set large enough to return every match, so
    /// there is no top-k truncation boundary where score ties could pick
    /// different docs in the two tables.
    fn all_scored(st: &Supertable, query: &str) -> Vec<(String, f32)> {
        // `k` large enough to return every match (no top-k truncation).
        const K_ALL: usize = 1000;
        top_k_scored(st, query, K_ALL)
    }

    /// Ranked top-`k` `(title, score)` for an `Or`-mode bm25_search. A
    /// small `k` (well below the match count) fills the top-k heap and
    /// engages the BMW/MaxScore pruning path; a large `k` returns the
    /// whole match set.
    fn top_k_scored(st: &Supertable, query: &str, k: usize) -> Vec<(String, f32)> {
        use arrow_array::{Float32Array, LargeStringArray};
        let batches = st
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                query,
                k,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["title", "score"]),
            )
            .expect("bm25_search");
        let mut out = Vec::new();
        for b in &batches {
            let titles = b
                .column(0)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("title utf8");
            let scores = b
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("score f32");
            for i in 0..b.num_rows() {
                out.push((titles.value(i).to_string(), scores.value(i)));
            }
        }
        out
    }

    /// The per-generation global-idf cache must refresh when a commit
    /// publishes a new manifest: raising a term's corpus-wide df must
    /// lower its global idf on the SAME handle whose earlier queries
    /// populated the cache. A stale cache would keep the old idf and
    /// this score would not move.
    #[test]
    fn global_idf_refreshes_after_commit_raises_df() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        // Segment 1: "alpha" is rare (1 of 4 uniform-length docs).
        let seg1 = [
            "alpha shared red d00",
            "beta shared red d01",
            "beta shared green d02",
            "beta shared green d03",
        ];
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(0, &seg1)).expect("append seg1");
            w.commit().expect("commit seg1");
        }
        let before = all_scored(&st, "alpha");
        let repeat = all_scored(&st, "alpha");
        assert_eq!(before, repeat, "same snapshot must score identically");
        let before_top = before[0].1;

        // Segment 2: every doc carries "alpha" — global df rises, so
        // global idf (and every alpha score) must drop after the commit.
        let seg2 = [
            "alpha shared red d10",
            "alpha shared red d11",
            "alpha shared green d12",
            "alpha shared green d13",
        ];
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(0, &seg2)).expect("append seg2");
            w.commit().expect("commit seg2");
        }
        let after = all_scored(&st, "alpha");
        let after_top = after.iter().map(|(_, s)| *s).fold(f32::MIN, f32::max);
        assert!(
            after_top < before_top,
            "df went 1/4 -> 5/8: global idf must drop \
             (top before {before_top}, top after {after_top})"
        );
    }

    /// Relative score gap two tables may differ by and still count as
    /// scoring alike: f32 rounding, never a different statistic.
    const SCORE_REL_TOLERANCE: f32 = 1e-5;

    /// Oracle for table-wide statistics: a table split across many
    /// commits, against the same docs in a single superfile (whose
    /// local statistics already are table-wide). Lengths fall with doc id, so each commit's own
    /// average differs from the table's.
    ///
    /// Both statistics are table-wide at query time: idf, and the average
    /// document length every superfile is scored at, whatever average it
    /// was built with. Every commit's documents score as they do in the
    /// single superfile.
    #[test]
    fn global_stats_multi_superfile_converges_to_single_superfile() {
        // 24 docs of *varying* length. The first three tokens carry the
        // query terms (so df/idf drives ranking); the trailing `dNN` is
        // a per-doc unique tag that keeps every title distinct, and a
        // run of filler stretches some documents well past others. It
        // never appears in a query, so it moves no term's weight — it
        // only changes document length, and therefore the average.
        //
        // The lengths vary on purpose: with every document the same size
        // each superfile's average equals the table-wide one no matter
        // how the commits fall, so the test could not tell a table-wide
        // length normalizer from a superfile-local one. Varying them is
        // what makes this an assertion about the normalizer and not only
        // about idf. The filler is front-loaded so the four commits below
        // get visibly different local averages.
        let titles: Vec<String> = (0..24)
            .map(|i| {
                let topic = ["alpha", "beta", "gamma"][i % 3];
                let band = ["red", "green"][(i / 3) % 2];
                let filler = vec!["filler"; 1 + (23 - i) / 3].join(" ");
                format!("{topic} shared {band} d{i:02} {filler}")
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();

        // SINGLE: one commit → one superfile (local stats == global).
        let single = Supertable::create(options_one_superfile_per_commit()).expect("create");
        {
            let mut w = single.writer().expect("writer");
            w.append(&build_batch(0, &refs)).expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(
            single
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len(),
            1,
            "single table must be one superfile"
        );

        // MULTI: four commits of six docs → four superfiles, same docs.
        let multi = Supertable::create(options_one_superfile_per_commit()).expect("create");
        {
            let mut w = multi.writer().expect("writer");
            for chunk in refs.chunks(6) {
                w.append(&build_batch(0, chunk)).expect("append");
                w.commit().expect("commit");
            }
        }
        assert!(
            multi
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len()
                > 1,
            "multi table must be fragmented across superfiles"
        );

        // Titles are unique, so `title -> score` fully identifies a
        // result. Comparing the map (not the ranked list) is robust to
        // tie-break order, which differs between the two tables because
        // it falls back to local doc ids.
        let score_map = |hits: Vec<(String, f32)>| -> std::collections::HashMap<String, f32> {
            hits.into_iter().collect()
        };

        let rel = |a: f32, b: f32| (a - b).abs() / a.abs().max(1.0);

        for q in ["alpha shared", "beta red", "gamma green d05", "shared red"] {
            let single_ref = score_map(all_scored(&single, q));
            let multi_global = score_map(all_scored(&multi, q));

            assert_eq!(
                single_ref.len(),
                multi_global.len(),
                "hit count mismatch for {q:?}"
            );
            for (title, s_score) in &single_ref {
                let g_score = multi_global
                    .get(title)
                    .unwrap_or_else(|| panic!("global result missing {title:?} for {q:?}"));
                assert!(
                    rel(*s_score, *g_score) <= SCORE_REL_TOLERANCE,
                    "global score {g_score} != single score {s_score} for {title:?} / {q:?}"
                );
            }
        }
    }

    /// Rows this column is null for are not documents it has: with them
    /// interleaved, table-wide idf (its collection size) and the average
    /// length both stay what the documents alone give, so every score is
    /// identical to the same documents ingested without the nulls.
    #[test]
    fn null_rows_leave_table_wide_scores_unchanged() {
        let titles: Vec<&str> = vec![
            "alpha shared red d00 filler filler",
            "beta shared green d01 filler",
            "gamma shared red d02",
            "alpha red d03 filler filler filler",
            "beta green d04",
            "gamma shared green d05 filler",
        ];
        let with_nulls: Vec<&str> = titles.iter().flat_map(|t| [*t, ""]).collect();

        let dense = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let sparse = Supertable::create(options_one_superfile_per_commit()).expect("create");
        for (st, rows) in [(&dense, &titles), (&sparse, &with_nulls)] {
            let mut w = st.writer().expect("writer");
            for chunk in rows.chunks(rows.len().div_ceil(2)) {
                w.append(&build_batch(0, chunk)).expect("append");
                w.commit().expect("commit");
            }
        }
        assert_eq!(
            sparse
                .reader()
                .expect("reader")
                .manifest()
                .fts_length_stats("title"),
            dense
                .reader()
                .expect("reader")
                .manifest()
                .fts_length_stats("title"),
            "null rows enter neither total"
        );
        for q in ["alpha shared", "beta red", "gamma green d05", "shared red"] {
            assert_eq!(all_scored(&sparse, q), all_scored(&dense, q), "{q:?}");
        }
    }

    /// The writer hands each new superfile the table's length totals, so
    /// the file declares the running table-wide average as of its commit
    /// — the first commit its own, every later one the average over all
    /// documents committed so far — and the manifest fold agrees.
    #[test]
    fn each_commit_bakes_the_running_table_wide_average() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        // Token totals per commit: 10 over 2 docs, then 4 over 2, then 2
        // over 1 — the empty title is a row this column has no document
        // for and must not enter the denominator.
        let commits: [&[&str]; 3] = [
            &["one two three four five six seven eight", "nine ten"],
            &["a b", "c d"],
            &["x y", ""],
        ];
        for titles in commits {
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(0, titles)).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let mut superfiles = manifest.get_all_superfiles().to_vec();
        assert_eq!(superfiles.len(), 3, "one superfile per commit");
        superfiles.sort_by_key(|sf| sf.id_min);
        let declared: Vec<f32> = superfiles
            .iter()
            .map(|sf| declared_title_avgdl(&reader, sf))
            .collect();
        assert_eq!(declared, vec![5.0, 14.0 / 4.0, 16.0 / 5.0]);
        assert_eq!(
            manifest.fts_length_stats("title"),
            Some(ColumnLengthStats {
                total_tokens: 16,
                n_scored_docs: 5,
            })
        );
        assert_eq!(
            manifest.fts_corpus_stats(&[]).get("title").copied(),
            manifest.fts_length_stats("title")
        );
        // What a compaction replacing the first two commits hands the
        // merged file: the third commit's totals alone.
        assert_eq!(
            manifest
                .fts_corpus_stats(&superfiles[..2])
                .get("title")
                .copied(),
            Some(ColumnLengthStats {
                total_tokens: 2,
                n_scored_docs: 1,
            })
        );
    }

    /// A part size every commit exceeds, so each commit writes its own
    /// manifest part instead of growing the previous one.
    const PART_PER_COMMIT_BYTES: u64 = 1;

    /// Parts loaded when a table opens: none, so every part is fetched on
    /// demand.
    const LAZY_OPEN_PARTS: u32 = 0;

    /// How many times a part-load test opens the table, so a cost paid on
    /// every open shows.
    const REOPENS: usize = 3;

    /// Compaction settings that merge the fixture's small superfiles: a 1 MB
    /// target each falls far below, and a fill floor each clears.
    const COMPACT_TARGET_MB: u64 = 1;
    const COMPACT_MIN_FILL_PERCENT: u8 = 1;

    /// An ordered-open window of one: superfiles open strictly in turn.
    const SEQUENTIAL_OPENS: usize = 1;

    /// Tokens in a long fixture document: enough to pull the table's average
    /// far above a two-token document's.
    const LONG_DOC_TOKENS: usize = 40;

    /// A `k` above every fixture's match count, so no hit is cut off.
    const K_EVERY_MATCH: usize = 100;

    /// Two commits whose term frequencies and lengths differ, so any
    /// statistic taken from one superfile instead of the table moves a
    /// score: `alpha` is in half of the first commit's long documents and
    /// a quarter of the second's short ones, and `zebra` and `zest` share
    /// the prefix `ze` but only `zebra` is in both commits.
    const SKEWED_COMMITS: [&[&str]; 2] = [
        &[
            "alpha zebra one two three four five six",
            "beta one two three four five six seven",
        ],
        &["alpha zebra x", "gamma zest y", "delta z", "epsilon w"],
    ];

    /// Averages for the ceiling-scaling test: a table average, and files
    /// built at half and at twice it.
    const TABLE_AVG: f32 = 4.0;
    const SHORTER_AVG: f32 = 2.0;
    const LONGER_AVG: f32 = 8.0;
    /// A ceiling as the term index stores it, before any scaling.
    const STORED_CEILING: f32 = 1.0;

    /// A ceiling stored at a shorter average than the table's is scaled up
    /// by their ratio. One stored at a longer average still holds, and so
    /// does one from a file that records no average: that file is scored at
    /// the average it was built at.
    #[test]
    fn ceilings_scale_only_for_files_built_at_a_shorter_average() {
        let column = FieldId(0);
        let entry = |baked: Option<f32>| {
            let mut entry = (*manifest_entry(1)).clone();
            entry.fts_summary.insert(
                column,
                FtsSummaryAgg::new_with_params(
                    None,
                    0,
                    (Vec::new(), Vec::new()),
                    ColumnLengthStats::default(),
                    baked,
                ),
            );
            Arc::new(entry)
        };
        let kept = vec![
            entry(Some(SHORTER_AVG)),
            entry(Some(LONGER_AVG)),
            entry(None),
        ];
        let mut ceilings: HashMap<Uuid, f32> = kept
            .iter()
            .map(|e| (e.superfile_id, STORED_CEILING))
            .collect();
        ceilings_at_table_avgdl(&mut ceilings, &kept, column, TABLE_AVG);
        let ceiling = |i: usize| ceilings[&kept[i].superfile_id];
        assert!(ceiling(0) >= STORED_CEILING * TABLE_AVG / SHORTER_AVG);
        assert_eq!(ceiling(1), STORED_CEILING);
        assert_eq!(ceiling(2), STORED_CEILING);
    }

    /// The `title` average length superfile `sf` declares in its footer.
    fn declared_title_avgdl(reader: &SupertableReader, sf: &SuperfileEntry) -> f32 {
        let superfile = reader.open_superfile(sf).expect("open superfile");
        let fts = superfile.fts().expect("fts index");
        fts.fts_columns_config()
            .next()
            .expect("title column")
            .avgdl()
    }

    /// Storage-backed [`options_one_superfile_per_commit`] where every
    /// commit writes its own manifest part, and opening loads no part.
    fn lazy_options(storage: &Arc<dyn StorageProvider>) -> SupertableOptions {
        options_one_superfile_per_commit()
            .with_storage(Arc::clone(storage))
            .with_part_size_threshold_bytes(PART_PER_COMMIT_BYTES)
            .with_eager_load_threshold(LAZY_OPEN_PARTS)
    }

    /// [`lazy_options`] opened with every part loaded.
    fn eager_options(storage: &Arc<dyn StorageProvider>) -> SupertableOptions {
        lazy_options(storage).with_eager_load_threshold(u32::MAX)
    }

    /// Commits `commits` one at a time to the table behind `storage`.
    fn commit_each(storage: &Arc<dyn StorageProvider>, commits: &[&[&str]]) {
        let st = Supertable::create(eager_options(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        for titles in commits {
            w.append(&build_batch(0, titles)).expect("append");
            w.commit().expect("commit");
        }
    }

    /// An in-process table holding [`SKEWED_COMMITS`], one superfile each.
    fn skewed_in_memory_table() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for titles in SKEWED_COMMITS {
            w.append(&build_batch(0, titles)).expect("append");
            w.commit().expect("commit");
        }
        st
    }

    /// Hits keyed by where they live, so two runs over one table compare
    /// without depending on how score ties were ordered.
    fn by_location(mut hits: Vec<SuperfileHit>) -> Vec<(Uuid, u32, f32)> {
        hits.sort_by_key(|h| (h.superfile.0, h.local_doc_id));
        hits.into_iter()
            .map(|h| (h.superfile.0, h.local_doc_id, h.score))
            .collect()
    }

    /// A handle that loads no manifest part at open scores exactly like
    /// one that loads them all.
    #[test]
    fn lazy_handle_scores_like_eager() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        commit_each(&storage, &SKEWED_COMMITS);

        let eager = Supertable::open(eager_options(&storage)).expect("open eager");
        let lazy = Supertable::open(lazy_options(&storage)).expect("open lazy");
        for q in ["alpha", "alpha beta", "zebra gamma", "one x"] {
            assert_eq!(all_scored(&lazy, q), all_scored(&eager, q), "{q:?}");
        }
    }

    /// A commit through a handle that loaded no part still declares the
    /// table-wide average length, not its own.
    #[test]
    fn lazy_writer_bakes_table_avgdl() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        commit_each(&storage, &SKEWED_COMMITS);
        let before: HashSet<Uuid> = Supertable::open(eager_options(&storage))
            .expect("open eager")
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|sf| sf.superfile_id)
            .collect();

        let lazy = Supertable::open(lazy_options(&storage)).expect("open lazy");
        let mut w = lazy.writer().expect("writer");
        w.append(&build_batch(0, &["omega", "omega psi"]))
            .expect("append");
        w.commit().expect("commit");

        let after = Supertable::open(eager_options(&storage)).expect("reopen eager");
        let reader = after.reader().expect("reader");
        let manifest = reader.manifest();
        let added: Vec<&Arc<SuperfileEntry>> = manifest
            .get_all_superfiles()
            .iter()
            .filter(|sf| !before.contains(&sf.superfile_id))
            .collect();
        let [added] = added.as_slice() else {
            panic!("one commit adds one superfile, got {}", added.len());
        };
        let table = manifest.fts_length_stats("title").expect("table totals");
        assert_eq!(
            declared_title_avgdl(&reader, added),
            bm25::stored_avgdl(table.avgdl())
        );
    }

    /// A compaction rewrites the parts it touches, so their totals cover
    /// the merged table and a lazy handle still scores like an eager one.
    #[test]
    fn lazy_handle_after_compaction_scores_like_eager() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        commit_each(&storage, &SKEWED_COMMITS);
        let eager = Supertable::open(eager_options(&storage)).expect("open eager");
        let before = eager
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .len();
        let merge_all = CompactionSettings {
            target_superfile_size_mb: COMPACT_TARGET_MB,
            min_fill_percent: COMPACT_MIN_FILL_PERCENT,
            min_superfiles_for_merge: SKEWED_COMMITS.len() as u64,
            ..CompactionSettings::default()
        };
        eager
            .optimize(&OptimizeOptions::compact(merge_all))
            .expect("optimize");
        let after = Supertable::open(eager_options(&storage)).expect("reopen eager");
        assert!(
            after
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len()
                < before,
            "the fixture compacts"
        );

        let lazy = Supertable::open(lazy_options(&storage)).expect("open lazy");
        for q in ["alpha", "zebra gamma", "one x"] {
            assert_eq!(all_scored(&lazy, q), all_scored(&after, q), "{q:?}");
        }
    }

    /// Liveness for the term index's postings loads only the parts whose id
    /// range covers a candidate, and a superfile the table does not hold is
    /// not live.
    #[test]
    fn live_among_loads_only_the_parts_covering_its_candidates() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        commit_each(&storage, &SKEWED_COMMITS);
        let all = Supertable::open(eager_options(&storage))
            .expect("open eager")
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .to_vec();
        let lazy = Supertable::open(lazy_options(&storage)).expect("open lazy");
        let reader = lazy.reader().expect("reader");
        let manifest = reader.manifest();
        let index = block_on(manifest.term_index())
            .expect("term index loads")
            .expect("term index");

        let one: HashSet<Uuid> = [all[0].superfile_id].into();
        let live = block_on(manifest.live_among(&index, &one)).expect("live");
        assert_eq!(live, one);
        let loaded = manifest
            .get_all_list_entries()
            .iter()
            .filter(|e| manifest.get_cached_part_by_id(&e.part_id).is_some())
            .count();
        assert_eq!(loaded, 1, "only the covering part loads");

        let stranger: HashSet<Uuid> = [Uuid::new_v4()].into();
        let live = block_on(manifest.live_among(&index, &stranger)).expect("live");
        assert!(live.is_empty(), "a superfile the table does not hold");
    }

    /// Term-index ceilings are stored at the average each file was built at.
    /// The first commit here is short documents, so its stored ceiling for
    /// `zeta` is far below what its short match scores once the long second
    /// commit lifts the table's average; the second commit's weaker match
    /// opens first. Unscaled, the first commit's ceiling would fall under
    /// that floor and its better hit would be skipped.
    #[test]
    fn a_file_built_at_a_shorter_average_is_not_skipped() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let filler = |n: usize| vec!["f"; n].join(" ");
        let weaker = format!("zeta zeta {}", filler(LONG_DOC_TOKENS / 2 - 2));
        let long = filler(LONG_DOC_TOKENS);
        commit_each(&storage, &[&["zeta a", "b c"], &[&weaker, &long, &long]]);
        // One open at a time, so the second commit's floor is set before the
        // first commit's ceiling is checked.
        let st = Supertable::open(
            eager_options(&storage).with_bound_ordered_open_window(SEQUENTIAL_OPENS),
        )
        .expect("open");
        let every = all_scored(&st, "zeta");
        assert_eq!(
            every[0].0, "zeta a",
            "the short match ranks first: {every:?}"
        );
        assert_eq!(top_k_scored(&st, "zeta", 1), every[..1]);
    }

    /// Largest allowed gap between an engine score and the brute-force
    /// reference's.
    const MIXED_SCORE_TOLERANCE: f32 = 1e-3;

    /// A table mixing a file without a recorded build-time average and
    /// files with one scores each as its summary says: the first at the
    /// average it was built at, the rest at the table's. The first file's
    /// ceilings stay as stored, which bound that scoring, so opening one
    /// file at a time still finds the best match.
    #[test]
    fn a_table_mixing_recorded_and_missing_averages_scores_each_file_as_recorded() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let filler = |n: usize| vec!["f"; n].join(" ");
        let weaker = format!("zeta zeta {}", filler(LONG_DOC_TOKENS / 2 - 2));
        let long = filler(LONG_DOC_TOKENS);
        let first: [&str; 2] = ["zeta zeta", "b c"];
        let second: [&str; 3] = [&weaker, &long, &long];
        commit_each(&storage, &[&first, &second]);

        // Strip the recorded average from the first commit's file, as a
        // file written before the summary carried one.
        let legacy = lazy_options(&storage).legacy_names();
        let (pointer, _) = block_on(read_pointer(storage.as_ref()))
            .expect("read pointer")
            .expect("pointer");
        let (bytes, _) = block_on(storage.get(&pointer.manifest_uri)).expect("list");
        let mut manifest_list = list::decode(&bytes, &legacy).expect("decode list");
        let entry = &mut manifest_list.parts[0];
        let (bytes, _) = block_on(storage.get(&entry.uri)).expect("part");
        let mut stripped = part::decode(&bytes, &legacy).expect("decode part");
        for superfile in &mut stripped.superfiles {
            let mut summaries = (**superfile).clone();
            for summary in summaries.fts_summary.values_mut() {
                summary.baked_avgdl = None;
            }
            *superfile = Arc::new(summaries);
        }
        let encoded = part::encode(&stripped);
        entry.content_hash = ContentHash::of(&encoded);
        entry.uri = part_uri(&entry.content_hash);
        entry.routing = None;
        block_on(storage.put_overwrite(&entry.uri, Bytes::from(encoded))).expect("write part");
        let encoded = list::encode(&manifest_list).expect("encode list");
        block_on(storage.put_overwrite(&pointer.manifest_uri, Bytes::from(encoded)))
            .expect("rewrite list");

        let tokenizer = default_tokenizer();
        let lengths = |titles: &[&str]| {
            ColumnLengthStats::from_lengths(titles.iter().map(|title| {
                let mut n = 0u32;
                tokenizer.tokenize_each(title, &mut |_| n += 1);
                n
            }))
        };
        let corpus: Vec<(u64, &str)> = first
            .iter()
            .chain(second.iter())
            .enumerate()
            .map(|(i, title)| (i as u64, *title))
            .collect();
        let titles: Vec<&str> = corpus.iter().map(|(_, title)| *title).collect();
        let reference = |titles_scored: &[&str], avgdl: f32| {
            BruteForceBm25::index(&corpus, tokenizer.as_ref())
                .with_avgdl(bm25::stored_avgdl(avgdl))
                .top_k("zeta", corpus.len(), tokenizer.as_ref())
                .into_iter()
                .map(|(doc, score)| (corpus[doc as usize].1, score))
                .filter(|(title, _)| titles_scored.contains(title))
                .collect::<Vec<_>>()
        };
        let mut expected = reference(&first, lengths(&first).avgdl());
        expected.extend(reference(&second, lengths(&titles).avgdl()));
        expected.sort_by(|a, b| b.1.total_cmp(&a.1));

        let st = Supertable::open(
            eager_options(&storage).with_bound_ordered_open_window(SEQUENTIAL_OPENS),
        )
        .expect("open");
        let every = all_scored(&st, "zeta");
        assert_eq!(every.len(), expected.len(), "{every:?} vs {expected:?}");
        for ((title, score), (want_title, want)) in every.iter().zip(&expected) {
            assert_eq!(title, want_title, "{every:?} vs {expected:?}");
            assert!(
                (score - want).abs() <= MIXED_SCORE_TOLERANCE,
                "{title}: engine {score}, reference {want}"
            );
        }
        assert_eq!(top_k_scored(&st, "zeta", 1), every[..1]);
    }

    /// Docs in the large superfile of the ranged-`Or` test: enough that
    /// the fan-out splits it into doc-id sub-ranges.
    const RANGED_FILE_DOCS: usize = 120_000;
    /// Every this-many docs of the large file holds the query term.
    const RANGED_MATCH_EVERY: usize = 10;
    /// Long docs committed beside the large file, so the table's average
    /// length moves well away from the large file's own.
    const RANGED_LONG_DOCS: usize = 1_000;
    /// Tokens in each long doc.
    const RANGED_LONG_DOC_TOKENS: usize = 300;
    /// Reader threads: more than one, so the fan-out uses sub-ranges.
    const RANGED_READER_THREADS: usize = 4;

    /// A superfile split into doc-id sub-ranges builds its shared `Or`
    /// cursors at the table's average length, the same view its ranges
    /// score through. Built at the file's own average instead, the stored
    /// block bounds no longer cap the rescored scores, and once the `x f`
    /// docs set the floor, the pruning skips the block holding the best
    /// doc, so it sits last in the file.
    #[test]
    fn ranged_or_builds_its_cursors_at_the_table_average() {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(RANGED_READER_THREADS)
                .build()
                .expect("pool"),
        );
        let st = Supertable::create(options_one_superfile_per_commit().with_reader_pool(pool))
            .expect("create");
        let short: Vec<&str> = (0..RANGED_FILE_DOCS)
            .map(|i| match i {
                _ if i == RANGED_FILE_DOCS - 1 => "x",
                _ if i % RANGED_MATCH_EVERY == 0 => "x f",
                _ => "f",
            })
            .collect();
        let long_doc = vec!["w"; RANGED_LONG_DOC_TOKENS].join(" ");
        let long = vec![long_doc.as_str(); RANGED_LONG_DOCS];
        let mut w = st.writer().expect("writer");
        for titles in [&short, &long] {
            w.append(&build_batch(0, titles)).expect("append");
            w.commit().expect("commit");
        }
        let top = top_k_scored(&st, "x zzz", 1);
        assert_eq!(
            top.first().map(|(title, _)| title.as_str()),
            Some("x"),
            "{top:?}"
        );
    }

    /// Counts the manifest parts read through it; everything else passes
    /// straight to `inner`.
    #[derive(Debug)]
    struct PartReadCounter {
        inner: Arc<dyn StorageProvider>,
        part_reads: AtomicUsize,
    }

    impl PartReadCounter {
        fn part_reads(&self) -> usize {
            self.part_reads.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl StorageProvider for PartReadCounter {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            self.inner.head(uri).await
        }
        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            if uri.starts_with(MANIFEST_PARTS_DIR) {
                self.part_reads.fetch_add(1, Ordering::SeqCst);
            }
            self.inner.get(uri).await
        }
        async fn get_if_none_match(
            &self,
            uri: &str,
            etag: &str,
        ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
            self.inner.get_if_none_match(uri, etag).await
        }
        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            self.inner.get_range(uri, range).await
        }
        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }
        async fn put_overwrite(&self, uri: &str, bytes: Bytes) -> Result<(), StorageError> {
            self.inner.put_overwrite(uri, bytes).await
        }
        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_if_match(uri, bytes, expected_etag).await
        }
        async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
            self.inner.put_multipart(uri).await
        }
        async fn delete(&self, uri: &str) -> Result<(), StorageError> {
            self.inner.delete(uri).await
        }
        async fn list_with_prefix_metadata(
            &self,
            prefix: &str,
        ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
            self.inner.list_with_prefix_metadata(prefix).await
        }
    }

    /// A list whose part entries carry no length totals, as one written
    /// before they existed, opens without reading a part. The first FTS
    /// query derives the totals, so a lazy handle still scores table-wide,
    /// and the next commit writes them out.
    #[test]
    fn part_totals_missing_from_the_list_are_derived_on_first_query() {
        let dir = TempDir::new().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        commit_each(&local, &SKEWED_COMMITS);
        let eager = Supertable::open(eager_options(&local)).expect("open eager");
        let expected = all_scored(&eager, "alpha");

        let legacy = lazy_options(&local).legacy_names();
        let read_list = || {
            let (pointer, _) = block_on(read_pointer(local.as_ref()))
                .expect("read pointer")
                .expect("pointer");
            let (bytes, _) = block_on(local.get(&pointer.manifest_uri)).expect("list");
            (pointer, list::decode(&bytes, &legacy).expect("decode"))
        };
        let (pointer, mut stripped) = read_list();
        for part in &mut stripped.parts {
            part.fts_length_totals = None;
        }
        let encoded = list::encode(&stripped).expect("encode");
        block_on(local.put_overwrite(&pointer.manifest_uri, Bytes::from(encoded)))
            .expect("rewrite list");

        let counter = Arc::new(PartReadCounter {
            inner: Arc::clone(&local),
            part_reads: AtomicUsize::new(0),
        });
        let storage: Arc<dyn StorageProvider> = Arc::clone(&counter) as _;
        for _ in 0..REOPENS {
            drop(Supertable::open(lazy_options(&storage)).expect("open lazy"));
        }
        assert_eq!(counter.part_reads(), 0, "opening reads no part");

        let lazy = Supertable::open(lazy_options(&storage)).expect("open lazy");
        assert_eq!(all_scored(&lazy, "alpha"), expected);

        let mut w = lazy.writer().expect("writer");
        w.append(&build_batch(0, &["omega"])).expect("append");
        w.commit().expect("commit");
        let (_, persisted) = read_list();
        assert!(
            persisted
                .parts
                .iter()
                .all(|part| part.fts_length_totals.is_some()),
            "the commit persists every part's totals"
        );
    }

    /// A prefix that expands to a single term in every superfile scores
    /// exactly like that term.
    #[test]
    fn prefix_with_one_expansion_scores_like_term() {
        let reader = skewed_in_memory_table().reader().expect("reader");
        let prefix = reader
            .bm25_search_prefix("title", "zeb", K_EVERY_MATCH)
            .expect("prefix search");
        let term = reader
            .bm25_hits("title", "zebra", K_EVERY_MATCH, Bm25SearchOptions::new())
            .expect("term search");
        assert!(!term.is_empty(), "the fixture matches `zebra`");
        assert_eq!(by_location(prefix), by_location(term));
    }

    /// A prefix scores like an `Or` query over every term it expands to
    /// anywhere in the table, including terms only some superfiles hold.
    #[test]
    fn prefix_scores_like_or_over_its_expansion() {
        let reader = skewed_in_memory_table().reader().expect("reader");
        let prefix = reader
            .bm25_search_prefix("title", "ze", K_EVERY_MATCH)
            .expect("prefix search");
        let expansion = reader
            .bm25_hits(
                "title",
                "zebra zest",
                K_EVERY_MATCH,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
            .expect("or search");
        assert_eq!(by_location(prefix), by_location(expansion));
    }

    /// Score slack between the ranked and the exact-count paths: the
    /// kernels add a doc's term scores in different orders.
    const SUM_ORDER_EPS: f32 = 1e-4;

    /// The two paths return the same top-k up to float summation order:
    /// every doc's score agrees within `SUM_ORDER_EPS`, and a doc may be
    /// on one side only when it ties the k-th score within that slack.
    fn assert_same_top_k(counted: &[SuperfileHit], pruned: &[SuperfileHit], label: &str) {
        assert_eq!(counted.len(), pruned.len(), "len for {label}");
        let kth = pruned.last().map_or(f32::NEG_INFINITY, |h| h.score);
        let scores = |hits: &[SuperfileHit]| {
            hits.iter()
                .map(|h| ((h.superfile.0, h.local_doc_id), h.score))
                .collect::<HashMap<_, f32>>()
        };
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

    /// A small deterministic generator for the random-corpus tests.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// Terms `w0..` in the random corpus; early ones are common, late ones rare.
    const RANDOM_VOCAB: u64 = 24;

    fn random_title(rng: &mut Lcg) -> String {
        let mut text = String::new();
        for t in 0..RANDOM_VOCAB {
            if rng.next().is_multiple_of(3 + t) {
                for _ in 0..1 + rng.next() % 3 {
                    text.push_str(&format!("w{t} "));
                }
            }
        }
        if text.is_empty() {
            text.push_str("w0");
        }
        text.trim().to_string()
    }

    /// `commits` superfiles of `docs_each` random titles.
    fn random_table(rng: &mut Lcg, commits: usize, docs_each: usize) -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        for _ in 0..commits {
            let titles: Vec<String> = (0..docs_each).map(|_| random_title(rng)).collect();
            let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            w.append(&build_batch(0, &refs)).expect("append");
            w.commit().expect("commit");
        }
        st
    }

    /// An OR of `n_terms` random terms (repeats allowed), with one negated
    /// term when `negated`.
    fn random_or_query(rng: &mut Lcg, n_terms: usize, negated: bool) -> String {
        let mut q: Vec<String> = (0..n_terms)
            .map(|_| format!("w{}", rng.next() % RANDOM_VOCAB))
            .collect();
        if negated {
            q.push(format!("-w{}", rng.next() % RANDOM_VOCAB));
        }
        q.join(" ")
    }

    /// Many-term ORs over several superfiles: the ranked and exact-count
    /// paths add a doc's term scores in different orders, so the top-k
    /// must agree up to that order, and the count must be exact.
    #[test]
    fn hits_with_count_matches_ranked_on_random_many_term_ors() {
        let mut rng = Lcg(11);
        let reader = random_table(&mut rng, 4, 1500).reader().expect("reader");
        for q in 0..40u32 {
            let n_terms = 2 + (rng.next() % 19) as usize;
            let negated = rng.next().is_multiple_of(3);
            let query = random_or_query(&mut rng, n_terms, negated);
            let expected = reader.count("title", &query, BoolMode::Or).expect("count");
            for k in [1usize, 10, 100] {
                let opts = Bm25SearchOptions::new();
                let pruned = reader
                    .bm25_hits("title", &query, k, opts)
                    .expect("bm25_hits");
                let (hits, count) = reader
                    .bm25_hits_with_count("title", &query, k, opts)
                    .expect("bm25_hits_with_count");
                assert_same_top_k(&hits, &pruned, &format!("q {q} {query:?} k={k}"));
                assert_eq!(count, expected, "q {q} {query:?} k={k}");
            }
        }
    }

    /// Cost of the exact-count mode against ranked plus count at OR widths
    /// 2, 8 and 20 over a 4-superfile table. Not a correctness test; run
    /// it by hand for the numbers:
    /// `cargo test --release --lib --features test-helpers -- --ignored --nocapture exact_count_cost`
    #[test]
    #[ignore]
    fn exact_count_cost_by_or_width() {
        const K: usize = 100;
        const QUERIES: usize = 50;
        let mut rng = Lcg(23);
        let reader = random_table(&mut rng, 4, 10_000).reader().expect("reader");
        for width in [2usize, 8, 20] {
            let queries: Vec<String> = (0..QUERIES)
                .map(|_| random_or_query(&mut rng, width, false))
                .collect();
            for q in &queries {
                reader
                    .bm25_hits("title", q, K, Bm25SearchOptions::new())
                    .expect("warm");
                reader
                    .bm25_hits_with_count("title", q, K, Bm25SearchOptions::new())
                    .expect("warm");
            }
            let ranked = Instant::now();
            for q in &queries {
                reader
                    .bm25_hits("title", q, K, Bm25SearchOptions::new())
                    .expect("ranked");
                reader.count("title", q, BoolMode::Or).expect("count");
            }
            let ranked = ranked.elapsed().as_secs_f64();
            let exact = Instant::now();
            for q in &queries {
                reader
                    .bm25_hits_with_count("title", q, K, Bm25SearchOptions::new())
                    .expect("exact");
            }
            let exact = exact.elapsed().as_secs_f64();
            println!(
                "or width {width:>2}: ranked+count {:>8.1} us/query, exact-count {:>8.1} us/query, {:.2}x",
                ranked * 1e6 / QUERIES as f64,
                exact * 1e6 / QUERIES as f64,
                exact / ranked
            );
        }
    }

    /// Over a table of several superfiles, the exact-count search returns
    /// the pruned search's top-k and the unranked count's total, for every
    /// clause shape, with `k` at zero, below and above the match count.
    #[test]
    fn hits_with_count_matches_bm25_hits_and_count() {
        let reader = skewed_in_memory_table().reader().expect("reader");
        let queries = [
            "alpha",
            "zebra one",
            "+alpha +zebra",
            "+alpha one",
            "alpha -zebra",
            "+zebra -one",
            "nothere",
        ];
        for query in queries {
            let expected = reader.count("title", query, BoolMode::Or).expect("count");
            for k in [0usize, 1, 2, K_EVERY_MATCH] {
                let opts = Bm25SearchOptions::new();
                let pruned = reader
                    .bm25_hits("title", query, k, opts)
                    .expect("bm25_hits");
                let (hits, count) = reader
                    .bm25_hits_with_count("title", query, k, opts)
                    .expect("bm25_hits_with_count");
                assert_same_top_k(&hits, &pruned, &format!("{query:?} k={k}"));
                assert_eq!(count, expected, "{query:?} k={k}");
            }
        }
    }

    /// Phrase queries go through the atom walk; the exact-count search
    /// must agree with the pruned search and the count there too.
    #[test]
    fn hits_with_count_matches_on_phrases() {
        let st = Supertable::create(options_positions_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["new york city", "york new haven"]))
            .expect("append");
        w.commit().expect("commit");
        w.append(&build_batch(
            2,
            &["the new york times", "new haven york", "new york new york"],
        ))
        .expect("append");
        w.commit().expect("commit");
        let reader = st.reader().expect("reader");
        let queries = [
            "\"new york\"",
            "+\"new york\" +city",
            "\"new york\" haven",
            "\"new york\" -\"new haven\"",
            "+york -\"new york\"",
        ];
        for query in queries {
            let expected = reader.count("title", query, BoolMode::Or).expect("count");
            for k in [1usize, 2, K_EVERY_MATCH] {
                let opts = Bm25SearchOptions::new();
                let pruned = reader
                    .bm25_hits("title", query, k, opts)
                    .expect("bm25_hits");
                let (hits, count) = reader
                    .bm25_hits_with_count("title", query, k, opts)
                    .expect("bm25_hits_with_count");
                assert_same_top_k(&hits, &pruned, &format!("{query:?} k={k}"));
                assert_eq!(count, expected, "{query:?} k={k}");
            }
        }
    }

    /// A deleted row is neither a hit nor counted, matching `count`.
    #[test]
    fn hits_with_count_excludes_tombstoned_docs() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha one", "alpha two", "alpha three"]))
            .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.delete(col("title").eq(lit("alpha two")))
            .expect("delete");
        let reader = st.reader().expect("reader");
        let expected = reader.count("title", "alpha", BoolMode::Or).expect("count");
        assert_eq!(expected, 2);
        for k in [1usize, K_EVERY_MATCH] {
            let (hits, count) = reader
                .bm25_hits_with_count("title", "alpha", k, Bm25SearchOptions::new())
                .expect("bm25_hits_with_count");
            assert_eq!(count, expected, "k={k}");
            assert_eq!(hits.len(), k.min(2), "k={k}");
            assert!(
                hits.iter().all(|h| h.local_doc_id != 1),
                "the deleted row is not a hit"
            );
        }
    }

    /// Like [`options_one_superfile_per_commit`] but with the `title`
    /// column positions-indexed, so phrase queries are answerable.
    fn options_positions_one_superfile_per_commit() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig::new("title").positions(true)],
            vec![],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Table-wide statistics must rank phrase-bearing queries
    /// on a fragmented table identically to a single superfile too — a
    /// phrase's score is Σ member idf, so globalizing the members
    /// globalizes the phrase.
    #[test]
    fn global_stats_phrase_query_matches_single_superfile() {
        // 24 uniform-length (4-token) docs: `<topic> quick <w2> dNN`.
        // "quick" is in every doc; "brown" only in the even docs (so
        // "brown" and the phrase "quick brown" have a df that varies by
        // superfile once fragmented). `dNN` keeps titles unique.
        let titles: Vec<String> = (0..24)
            .map(|i| {
                let topic = ["alpha", "beta", "gamma"][i % 3];
                let w2 = if i % 2 == 0 { "brown" } else { "red" };
                format!("{topic} quick {w2} d{i:02}")
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(|s| s.as_str()).collect();

        let single =
            Supertable::create(options_positions_one_superfile_per_commit()).expect("create");
        {
            let mut w = single.writer().expect("writer");
            w.append(&build_batch(0, &refs)).expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(
            single
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len(),
            1,
            "single table must be one superfile"
        );

        let multi =
            Supertable::create(options_positions_one_superfile_per_commit()).expect("create");
        {
            let mut w = multi.writer().expect("writer");
            for chunk in refs.chunks(6) {
                w.append(&build_batch(0, chunk)).expect("append");
                w.commit().expect("commit");
            }
        }
        assert!(
            multi
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len()
                > 1,
            "multi table must be fragmented across superfiles"
        );

        let score_map = |hits: Vec<(String, f32)>| -> std::collections::HashMap<String, f32> {
            hits.into_iter().collect()
        };

        // A bare-should term + a phrase (exercises both gather paths:
        // the bare term and the phrase members), and a pure phrase.
        for q in ["alpha \"quick brown\"", "\"quick brown\""] {
            let single_ref = score_map(all_scored(&single, q));
            let multi_global = score_map(all_scored(&multi, q));

            assert!(!single_ref.is_empty(), "query {q:?} matched nothing");
            assert_eq!(
                single_ref.len(),
                multi_global.len(),
                "hit count mismatch for {q:?}"
            );
            for (title, s_score) in &single_ref {
                let g_score = multi_global
                    .get(title)
                    .unwrap_or_else(|| panic!("global result missing {title:?} for {q:?}"));
                assert!(
                    (s_score - g_score).abs() <= 1e-5 * s_score.abs().max(1.0),
                    "global score {g_score} != single score {s_score} for {title:?} / {q:?}"
                );
            }
        }
    }

    /// Small-`k` oracle for table-wide statistics: with `k` far below the
    /// match count the top-k heap fills, so the BMW/MaxScore pruning
    /// path genuinely runs. The stored per-block skip upper bounds are
    /// rescaled by the global/local idf ratio; if that rescale produced
    /// an invalid (too-low) bound the pruner would wrongly skip a
    /// top-scoring doc and corrupt the result. This asserts the pruned
    /// global top-k still equals the single-superfile top-k.
    #[test]
    fn global_stats_small_k_pruning_matches_single_superfile() {
        // `common` is in every doc, so its postings span more than one
        // block and the pruner has whole blocks it can skip. Three
        // "boost" docs additionally carry a rare, high-idf term at
        // distinct term frequencies, giving them the three
        // strictly-highest, distinct scores — an unambiguous top-3.
        const N: usize = BLOCK_LEN + 32;
        const L: usize = 8; // tokens/doc; uniform so avgdl matches everywhere
        const K: usize = 3;
        // (doc index, boost tf). Distinct tf ⇒ distinct scores; the docs
        // are spread past BLOCK_LEN so a top-k doc sits in a later block
        // the walk must not wrongly prune.
        let boosts = [
            (10usize, 3u32),
            (BLOCK_LEN / 2 + 26, 2),
            (BLOCK_LEN + 22, 1),
        ];
        let titles: Vec<String> = (0..N)
            .map(|i| {
                let bt = boosts
                    .iter()
                    .find(|(idx, _)| *idx == i)
                    .map(|(_, tf)| *tf as usize)
                    .unwrap_or(0);
                let mut toks: Vec<String> = vec!["common".to_string()];
                for _ in 0..bt {
                    toks.push("boost".to_string());
                }
                while toks.len() < L {
                    toks.push("pad".to_string());
                }
                // Unique tag (df=1, never queried, replaces a pad token so
                // length stays L): keeps every title distinct so a top-k
                // doc is identifiable across the two independently-built
                // tables, without affecting any query score.
                toks[L - 1] = format!("d{i:03}");
                toks.join(" ")
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();

        // SINGLE: one commit → one superfile (local stats == global).
        let single = Supertable::create(options_one_superfile_per_commit()).expect("create");
        {
            let mut w = single.writer().expect("writer");
            w.append(&build_batch(0, &refs)).expect("append");
            w.commit().expect("commit");
        }
        assert_eq!(
            single
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len(),
            1
        );

        // MULTI: many small commits → many superfiles, same docs.
        let multi = Supertable::create(options_one_superfile_per_commit()).expect("create");
        {
            let mut w = multi.writer().expect("writer");
            for chunk in refs.chunks(20) {
                w.append(&build_batch(0, chunk)).expect("append");
                w.commit().expect("commit");
            }
        }
        assert!(
            multi
                .reader()
                .expect("reader")
                .manifest()
                .get_all_superfiles()
                .len()
                > 1
        );

        // `+common` is the (huge) match set; the rare `boost` is a
        // scoring-only should whose contribution lifts its docs into the
        // top-k. The must-driven walk prunes candidates using the
        // shoulds' `term_max` upper bound, so a `boost` term_max left
        // un-rescaled (too low) would make the walk over-prune and drop
        // the very docs that belong in the top-k.
        let q = "+common boost";
        let single_ref = top_k_scored(&single, q, K);
        let multi_global = top_k_scored(&multi, q, K);

        // The heap truly filled: `k` results, far below the ~160 matches.
        assert_eq!(
            single_ref.len(),
            K,
            "top-k should be truncated to k (heap full)"
        );
        assert_eq!(multi_global.len(), K, "global top-k should also be k");

        // Same docs, same order, same scores as the single superfile.
        for ((s_title, s_score), (g_title, g_score)) in single_ref.iter().zip(&multi_global) {
            assert_eq!(s_title, g_title, "top-{K} doc/order mismatch under pruning");
            assert!(
                (s_score - g_score).abs() <= 1e-5 * s_score.abs().max(1.0),
                "top-{K} score mismatch: single {s_score} vs global {g_score}"
            );
        }

        // Sanity: the top-k really is the three boost docs (only they
        // carry the rare term), so pruning had to reach them.
        assert!(
            multi_global.iter().all(|(t, _)| t.contains("boost")),
            "top-{K} must be the boost docs, got {multi_global:?}"
        );
    }

    /// Build a single SuperfileBuilder containing the same docs as
    /// the supertable across all superfiles. Used as the oracle for
    /// BM25 set-membership tests.
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
        let opts =
            BuilderOptions::new(schema.clone(), "_id", vec![FtsConfig::new("title")], vec![]);
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

        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "alpha -beta",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus beta: {hits:?}");

        // Positive-only stays untouched: all three alpha docs.
        let hits = r
            .bm25_hits(
                "title",
                "alpha",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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

        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "alpha -delta",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::And),
            )
            .expect("negation search");
        assert_eq!(hits.len(), 2, "alpha minus delta: {hits:?}");
    }

    #[test]
    fn negation_only_query_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["alpha beta"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader().expect("reader");
        let res = r.bm25_hits(
            "title",
            "-alpha",
            10,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
        );
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
        let r = st.reader().expect("reader");

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
        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "rust",
                5,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
            .expect("query");
        assert!(hits.is_empty());
    }

    #[test]
    fn bm25_search_unknown_projection_column_is_a_clean_error() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader().expect("reader");

        let err = r
            .bm25_search(
                "title",
                "rust",
                5,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["title", "does_not_exist"]),
            )
            .expect_err("unknown projection column must error");

        // A bad projection is caller input, not an engine failure: it comes
        // back as InvalidQuery, names the offending column and the valid set,
        // and never leaks the query engine's internals into the message. (The
        // single-table search kernels run without the SQL engine at all, so a
        // "DataFusion"/"Execution error" phrasing would be doubly misleading.)
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "expected InvalidQuery, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("does_not_exist"),
            "names the bad column: {msg}"
        );
        assert!(msg.contains("valid columns"), "lists valid columns: {msg}");
        assert!(
            msg.contains("title") && msg.contains("score"),
            "valid set includes the real columns: {msg}"
        );
        assert!(
            !msg.contains("DataFusion") && !msg.contains("Execution error"),
            "must not leak query-engine internals: {msg}"
        );
    }

    #[test]
    fn bm25_search_k_zero_short_circuits() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&build_batch(0, &["rust async"])).expect("append");
        w.commit().expect("commit");
        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "rust",
                0,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "rust",
                4,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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

        let r = st.reader().expect("reader");
        assert_eq!(r.n_superfiles(), 2);
        let hits = r
            .bm25_hits(
                "title",
                "rust",
                5,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        // Plant a corpus where the top-k under BM25 is unambiguous: 3
        // docs contain the rare term `nimblefox`, one in each of 3
        // superfiles; the other 9 docs never contain it, so they score
        // zero against `nimblefox`.
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
        assert_eq!(st.reader().expect("reader").n_superfiles(), 3);

        let oracle = build_oracle_superfile(&titles);
        // Single-superfile `SuperfileReader` oracle: async-only search,
        // driven on a throwaway runtime. The supertable reader below
        // uses its sync public API.
        let oracle_hits = block_on(oracle.bm25_hits_async("title", "nimblefox", 5, BoolMode::Or))
            .expect("oracle");
        // Oracle should find exactly 3 docs containing `nimblefox`.
        assert_eq!(oracle_hits.len(), 3);
        let oracle_set: HashSet<u32> = oracle_hits.iter().map(|(d, _)| d.get()).collect();
        assert_eq!(oracle_set, [0u32, 4, 8].iter().copied().collect());

        let st_reader = st.reader().expect("reader");
        let st_hits = st_reader
            .bm25_hits(
                "title",
                "nimblefox",
                5,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        // The oracle is one superfile, so its own expansion is the table's.
        let fts = oracle.fts().expect("oracle fts");
        let expansion = block_on(fts.terms_with_prefix("title", b"rust", None)).expect("expand");
        let terms: Vec<&str> = expansion
            .iter()
            .map(|t| std::str::from_utf8(t).expect("utf8 term"))
            .collect();
        let oracle_hits =
            block_on(oracle.bm25_search_pretokenized("title", &terms, 5, BoolMode::Or))
                .expect("oracle");
        let oracle_globals: HashSet<u32> = oracle_hits.iter().map(|(d, _)| d.get()).collect();

        let st_reader = st.reader().expect("reader");
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

        let r = st.reader().expect("reader");
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

        let r = st.reader().expect("reader");
        let hits = r.bm25_search_prefix("title", "RUST", 5).expect("query");
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn bm25_search_unknown_column_errors() {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        // A committed superfile exists, so the query has real data to scan. The
        // queried column carries no full-text index, though: the reject must
        // happen up front, not deep in the scan where the low-level reader
        // would surface an opaque storage-format error.
        w.append(&build_batch(0, &["rust"])).expect("append");
        w.commit().expect("commit");

        let r = st.reader().expect("reader");
        let err = r
            .bm25_hits(
                "missing_column",
                "rust",
                5,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
            .expect_err("expected error");
        assert!(matches!(err, QueryError::InvalidQuery(_)), "got {err:?}");
        let msg = err.to_string();
        assert!(
            msg.contains("no full-text index"),
            "explains the miss: {msg}"
        );
        assert!(msg.contains("missing_column"), "names the column: {msg}");
        assert!(
            !msg.contains("inf.fts.offset") && !msg.contains("parquet"),
            "must not leak the storage-format internals: {msg}"
        );
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
        let r = st.reader().expect("reader");
        let hits = r
            .bm25_hits(
                "title",
                "rust",
                2,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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

    /// The override has to survive the whole path — public options,
    /// `bm25_search_async`, the per-superfile fan-out, both the prepare
    /// and the score halves — and it is only observable through scores,
    /// so this asserts on the numbers rather than on plumbing.
    #[test]
    fn supertable_bm25_search_honors_a_query_time_bm25_params() {
        let st = seeded_three_doc_supertable();
        let scores = |opts: Bm25SearchOptions| -> Vec<(String, f32)> {
            use arrow_array::{Float32Array, LargeStringArray};
            let batches = st
                .reader()
                .expect("reader")
                .bm25_search("title", "quick", 10, opts, Some(&["title", "score"]))
                .expect("bm25_search");
            let mut out = Vec::new();
            for b in &batches {
                let titles = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("title utf8");
                let sc = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("score f32");
                for i in 0..b.num_rows() {
                    out.push((titles.value(i).to_string(), sc.value(i)));
                }
            }
            out
        };

        let base = scores(Bm25SearchOptions::new());
        assert_eq!(base.len(), 2, "two docs contain `quick`");

        // b = 0 disables length normalization entirely, so the two docs'
        // scores must converge: they differ under the default only
        // because one is longer.
        let no_len_norm = scores(Bm25SearchOptions::new().with_bm25(1.2, 0.0));
        assert_eq!(no_len_norm.len(), base.len(), "same match set");
        let spread = |v: &[(String, f32)]| {
            let mut s: Vec<f32> = v.iter().map(|(_, x)| *x).collect();
            s.sort_by(|a, b| b.total_cmp(a));
            s[0] - s[s.len() - 1]
        };
        assert!(
            spread(&no_len_norm) < spread(&base),
            "b=0 must compress the score spread: base {:?} vs override {:?}",
            base,
            no_len_norm
        );
        assert!(
            spread(&no_len_norm) < 1e-6,
            "with b=0 both docs share a length norm, so scores tie: {no_len_norm:?}"
        );

        // An override equal to what the columns declare changes nothing.
        let same = scores(Bm25SearchOptions::new().with_bm25(1.2, 0.75));
        for ((t1, s1), (t2, s2)) in base.iter().zip(same.iter()) {
            assert_eq!(t1, t2);
            assert!((s1 - s2).abs() < 1e-6, "{s1} vs {s2}");
        }
    }

    /// An out-of-range override is rejected before the fan-out, naming
    /// the bounds rather than ranking oddly.
    #[test]
    fn supertable_bm25_search_rejects_an_invalid_override() {
        let st = seeded_three_doc_supertable();
        for (k1, b) in [(0.0_f32, 0.5_f32), (-1.0, 0.5), (1.2, 1.5), (1.2, -0.1)] {
            let err = st
                .reader()
                .expect("reader")
                .bm25_search(
                    "title",
                    "quick",
                    10,
                    Bm25SearchOptions::new().with_bm25(k1, b),
                    None,
                )
                .expect_err("out-of-range override must be refused");
            let msg = err.to_string();
            assert!(
                msg.contains("k1") && msg.contains("b"),
                "the error should name both bounds: {msg}"
            );
        }
    }

    #[test]
    fn supertable_bm25_search_rows_default_and_projected() {
        let st = seeded_three_doc_supertable();

        // Bare call → `_id` + `score` only (no scalar decode).
        let bare = st
            .bm25_search(
                "title",
                "fox",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                None,
            )
            .expect("bm25 rows");
        assert_eq!(bare.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(bare[0].num_columns(), 2, "_id + score");

        // Named projection materializes the requested columns.
        let rows = st
            .bm25_search(
                "title",
                "fox",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
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
        let r = st.reader().expect("reader");

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
        let r = st.reader().expect("reader");
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
            vec![FtsConfig::new("title").positions(true)],
            vec![],
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

    /// Phrase and must-clause queries route through kernels the
    /// single-term and union tests never touch — the phrase cursor
    /// composes its members' idfs and its own term-level bound, and a
    /// `+must` clause runs the ranked-AND membership walk. Both read
    /// stored bounds, so both have to see the correction; the check is
    /// that a declared pair and the same pair reached by override agree
    /// document-for-document and score-for-score.
    #[test]
    fn declared_and_overridden_pairs_agree_on_phrase_and_must_kernels() {
        const K1: f32 = 1.6;
        const B: f32 = 0.4;

        // One table baked at the pair, one baked at the defaults and
        // queried with the pair as an override.
        let baked = {
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("pool"),
            );
            let opts = SupertableOptions::new(
                schema_id_title(),
                vec![FtsConfig::new("title").positions(true).bm25(K1, B)],
                vec![],
            )
            .expect("valid options")
            .with_writer_pool(pool);
            let st = Supertable::create(opts).expect("create");
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(0, &["new york city", "the new york times"]))
                .expect("append");
            w.commit().expect("commit");
            w.append(&build_batch(10, &["york loves new haven", "big new york"]))
                .expect("append");
            w.commit().expect("commit");
            st
        };
        let standard = seeded_phrase_supertable();

        let hits = |st: &Supertable, query: &str, opts: Bm25SearchOptions| {
            st.reader()
                .expect("reader")
                .bm25_hits("title", query, 10, opts)
                .expect("bm25 hits")
        };

        for query in [r#""new york""#, "+new +york", r#""new york" city"#] {
            let from_declared = hits(
                &baked,
                query,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            );
            let from_override = hits(
                &standard,
                query,
                Bm25SearchOptions::new()
                    .with_mode(BoolMode::Or)
                    .with_bm25(K1, B),
            );
            assert!(
                !from_declared.is_empty(),
                "{query} must match something for the comparison to mean anything"
            );
            assert_eq!(
                from_declared.len(),
                from_override.len(),
                "hit count diverged for {query}"
            );
            for (a, b) in from_declared.iter().zip(from_override.iter()) {
                assert_eq!(
                    a.local_doc_id, b.local_doc_id,
                    "doc order diverged for {query}"
                );
                assert!(
                    (a.score - b.score).abs() < 1e-4,
                    "score diverged for {query}: {} vs {}",
                    a.score,
                    b.score
                );
            }
        }
    }

    /// The correction factor composes with the idf rescale — the shipped
    /// factor is `(idf / local_idf) · R`, the path every search takes. An
    /// override must agree with a table baked at that pair.
    #[test]
    fn the_override_composes_with_the_idf_rescale() {
        const K1: f32 = 0.7;
        const B: f32 = 0.9;

        let baked = {
            let pool = Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("pool"),
            );
            let opts = SupertableOptions::new(
                schema_id_title(),
                vec![FtsConfig::new("title").bm25(K1, B)],
                vec![],
            )
            .expect("valid options")
            .with_writer_pool(pool);
            let st = Supertable::create(opts).expect("create");
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(
                0,
                &["the quick brown fox", "a lazy dog", "quick thinking"],
            ))
            .expect("append");
            w.commit().expect("commit");
            st
        };
        let standard = seeded_three_doc_supertable();

        let declared = baked
            .reader()
            .expect("reader")
            .bm25_hits("title", "quick", 10, Bm25SearchOptions::new())
            .expect("declared");
        let overridden = standard
            .reader()
            .expect("reader")
            .bm25_hits(
                "title",
                "quick",
                10,
                Bm25SearchOptions::new().with_bm25(K1, B),
            )
            .expect("overridden");
        assert_eq!(declared.len(), overridden.len(), "hit count");
        for (a, b) in declared.iter().zip(overridden.iter()) {
            assert!(
                (a.score - b.score).abs() < 1e-4,
                "score diverged {} vs {}",
                a.score,
                b.score
            );
        }
    }

    /// Docs in the block-skip fixture: a term carried by every document
    /// spans several whole blocks and the block-max skip has something
    /// to skip *over*. Three documents fit in one partial block, where
    /// the skip path is unreachable.
    const BLOCK_SKIP_DOCS: usize = 5 * BLOCK_LEN;

    /// Longest document, in repetitions of the padding token. Lengths
    /// sweep from 1 to this, so the length norm — the half of the score
    /// `b` controls — varies widely across blocks. A uniform-length
    /// corpus would make the correction factor's length term
    /// degenerate and hide an under-tight bound.
    const BLOCK_SKIP_MAX_PAD: usize = 24;

    /// A corpus large enough that the block-max skip actually runs.
    ///
    /// Every document carries `quick`, so its posting list covers all
    /// `BLOCK_SKIP_DOCS` and is split into whole blocks with a stored
    /// per-block maximum. `brown` is planted on a sparse, irregular
    /// subset, and both the term frequency and the document length
    /// vary per document, so per-block maxima differ and a small `k`
    /// prunes rather than walking everything.
    fn seeded_block_skip_supertable() -> Supertable {
        let st = Supertable::create(options_one_superfile_per_commit()).expect("create");
        let mut w = st.writer().expect("writer");
        let titles: Vec<String> = (0..BLOCK_SKIP_DOCS)
            .map(|i| {
                let mut t = String::from("quick");
                // Term frequency varies: a repeated term saturates
                // differently as k1 moves, so the tf half of the score
                // is exercised too.
                for _ in 0..(i % 3) {
                    t.push_str(" quick");
                }
                if i % 7 == 0 {
                    t.push_str(" brown");
                }
                // Length varies 1..=BLOCK_SKIP_MAX_PAD padding tokens.
                for j in 0..(i % BLOCK_SKIP_MAX_PAD + 1) {
                    t.push_str(&format!(" pad{j}"));
                }
                t
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        w.append(&build_batch(0, &refs)).expect("append");
        w.commit().expect("commit");
        st
    }

    /// A small `k` fills the top-k heap and engages the block-max skips;
    /// a `k` covering the whole match set does not. Under an override
    /// every stored bound is inflated by the correction factor, so a
    /// factor that came out too small would skip a block holding a
    /// qualifying document — visible only as the small-`k` result
    /// disagreeing with the head of the unpruned one.
    ///
    /// The corpus spans several whole blocks on purpose. The skip
    /// compares a *stored per-block maximum* against the current
    /// kth-best score, so a corpus that fits in one partial block never
    /// reaches that comparison and cannot observe the correction at
    /// all — the assertion would hold no matter how wrong the factor
    /// was.
    #[test]
    fn an_override_prunes_without_dropping_hits_across_blocks() {
        let st = seeded_block_skip_supertable();
        let r = st.reader().expect("reader");
        const K_ALL: usize = 4 * BLOCK_SKIP_DOCS;

        for (k1, b) in [(1.6_f32, 0.4_f32), (0.5, 0.9), (1.2, 0.0), (2.0, 1.0)] {
            let opts = || {
                Bm25SearchOptions::new()
                    .with_mode(BoolMode::Or)
                    .with_bm25(k1, b)
            };
            let unpruned = r
                .bm25_hits("title", "quick brown", K_ALL, opts())
                .expect("unpruned");
            assert!(
                unpruned.len() > BLOCK_LEN,
                "fixture must span more than one block; got {}",
                unpruned.len()
            );
            // Small k relative to the match set: the heap fills early
            // and the kth-best score climbs above whole blocks' maxima.
            for k in [1usize, 2, 10, 50] {
                let pruned = r
                    .bm25_hits("title", "quick brown", k, opts())
                    .expect("pruned");
                assert_eq!(
                    pruned.len(),
                    k.min(unpruned.len()),
                    "k={k} at k1={k1} b={b}"
                );
                for (i, hit) in pruned.iter().enumerate() {
                    assert_eq!(
                        hit.local_doc_id, unpruned[i].local_doc_id,
                        "k={k} at k1={k1} b={b}: pruning changed the top-{k} head"
                    );
                    assert!(
                        (hit.score - unpruned[i].score).abs() < 1e-4,
                        "k={k} at k1={k1} b={b}: pruning changed a score"
                    );
                }
            }
        }
    }

    /// The same property on a corpus too small to form a full block.
    /// Kept alongside the multi-block case because it covers the
    /// partial-block tail, which has its own bound handling.
    #[test]
    fn an_override_prunes_without_dropping_hits() {
        let st = seeded_three_doc_supertable();
        let r = st.reader().expect("reader");
        const K_ALL: usize = 1000;

        for (k1, b) in [(1.6_f32, 0.4_f32), (0.5, 0.9), (1.2, 0.0), (2.0, 1.0)] {
            let opts = || {
                Bm25SearchOptions::new()
                    .with_mode(BoolMode::Or)
                    .with_bm25(k1, b)
            };
            let unpruned = r
                .bm25_hits("title", "quick brown", K_ALL, opts())
                .expect("unpruned");
            for k in [1usize, 2] {
                let pruned = r
                    .bm25_hits("title", "quick brown", k, opts())
                    .expect("pruned");
                let want = k.min(unpruned.len());
                assert_eq!(pruned.len(), want, "k={k} at k1={k1} b={b}");
                for (i, hit) in pruned.iter().enumerate() {
                    assert_eq!(
                        hit.local_doc_id, unpruned[i].local_doc_id,
                        "k={k} at k1={k1} b={b}: pruning changed the top-{k} head"
                    );
                }
            }
        }
    }

    #[test]
    fn phrase_query_end_to_end() {
        let st = seeded_phrase_supertable();
        let r = st.reader().expect("reader");

        // Ranked: exactly the adjacent-in-order docs across both
        // superfiles.
        let hits = r
            .bm25_hits(
                "title",
                r#""new york""#,
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
            .bm25_hits(
                "title",
                r#"+"new york" +the"#,
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");
        let err = r
            .bm25_hits(
                "title",
                r#""climate change""#,
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");

        // 3 docs contain `climate`; `policy` is scoring-only and must
        // not pull in "policy analysis quarterly".
        let hits = r
            .bm25_hits(
                "title",
                "+climate policy",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");
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
        let r = st.reader().expect("reader");
        // Negation still excludes: drop the summit doc from the
        // climate must set.
        let hits = r
            .bm25_hits(
                "title",
                "+climate policy -summit",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");
        // The must term exists nowhere: bloom-prune (or the empty
        // intersection) yields no hits despite the common should.
        let hits = r
            .bm25_hits(
                "title",
                "+zzzabsent policy",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
            )
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
        let r = st.reader().expect("reader");
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
        let e0 = manifest_entry(100);
        let e1 = manifest_entry(200);
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

    /// The compacted shape: one merged superfile holding essentially the
    /// whole corpus plus small remnants. Even-per-file allocation gave
    /// the merged file `ceil(threads / n_files)` slices — 2 of 8 on the
    /// 5-superfile table the 1M bench produces post-compaction — while
    /// remnants took units they could not fill. Doc-mass allocation must
    /// hand the merged file the pool and leave remnants whole.
    #[test]
    fn build_work_units_allocates_by_doc_mass_not_file_count() {
        /// Threads in the simulated reader pool.
        const POOL: usize = 8;
        /// Docs in the merged superfile (compaction output).
        const MERGED_DOCS: u64 = 1_000_000;
        /// Docs in each post-merge remnant — below `SUBRANGE_MIN_DOCS`.
        const REMNANT_DOCS: u64 = 10_000;

        let merged = manifest_entry(MERGED_DOCS);
        let remnants = [
            manifest_entry(REMNANT_DOCS),
            manifest_entry(REMNANT_DOCS),
            manifest_entry(REMNANT_DOCS),
            manifest_entry(REMNANT_DOCS),
        ];
        let mut kept = vec![&merged];
        kept.extend(remnants.iter());

        let units = build_work_units(&kept, FanOut::SubRanges, POOL);
        let merged_units: Vec<_> = units
            .iter()
            .filter(|u| u.entry.superfile_id == merged.superfile_id)
            .collect();
        assert_eq!(
            merged_units.len(),
            POOL,
            "merged superfile holds ~all docs so it must take the whole pool"
        );
        for remnant in &remnants {
            let n: Vec<_> = units
                .iter()
                .filter(|u| u.entry.superfile_id == remnant.superfile_id)
                .collect();
            assert_eq!(n.len(), 1, "sub-floor remnant stays one unit");
            assert!(n[0].range.is_none(), "remnant unit must be un-ranged");
        }
        // The merged file's slices tile [0, MERGED_DOCS) without gaps.
        let mut ranges: Vec<(u32, u32)> = merged_units
            .iter()
            .map(|u| u.range.expect("merged units are ranged"))
            .collect();
        ranges.sort_unstable();
        let mut cursor = 0u32;
        for (start, end) in ranges {
            assert_eq!(start, cursor, "sub-ranges tile without gaps");
            assert!(end > start, "sub-range is non-empty");
            cursor = end;
        }
        assert_eq!(cursor, MERGED_DOCS as u32, "sub-ranges cover every doc");
    }

    /// Regression: a sliced fan-out must preallocate top-k heap slots for
    /// the docs it can actually rank, not one whole-superfile heap **per
    /// slice**. Before the fix the slices requested
    /// `slices × min(k, superfile docs)` — 61 MiB against 7.6 MiB
    /// rankable at the 1M × 8-thread shape below, and a pool-sized
    /// multiple (GBs) on a compacted table with a wide reader pool.
    ///
    /// All three trigger conditions are just what a compacted table under
    /// an everyday query looks like:
    ///   1. a bare two-or-more-term OR — `fanout_for(0, >= 2, false)` is
    ///      `SubRanges`, so any two-word query slices;
    ///   2. one merged superfile holding ~all the doc mass, which
    ///      doc-mass allocation hands the entire pool (see
    ///      `build_work_units_allocates_by_doc_mass_not_file_count`) —
    ///      i.e. the shape `optimize` produces;
    ///   3. a large `k`, which the fan-out passes to every slice
    ///      unchanged.
    ///
    /// The slices tile the doc space exactly once, so the slots the query
    /// needs is bounded by `min(k, docs)`. Capacity is computed through
    /// the same `top_k_initial_capacity` both ranged kernels call
    /// (`run_max_score_bmm_range`, `run_windowed_union`), each passing
    /// its own `[start, end)`.
    #[test]
    fn ranged_slice_heaps_are_sized_by_their_own_range() {
        /// Threads in the reader pool = slices the merged file takes.
        const POOL: usize = 8;
        /// Docs in the merged superfile a full optimize produces.
        const MERGED_DOCS: u64 = 1_000_000;
        /// Result size. Large `k` is what makes the over-allocation bite:
        /// at small `k` the cap is `k` and the waste is negligible.
        const K: usize = MERGED_DOCS as usize;
        /// Bytes per heap slot — `TopKEntry` is `(f32, u32)`.
        const SLOT_BYTES: usize = 8;

        let merged = manifest_entry(MERGED_DOCS);
        let kept = vec![&merged];
        let units = build_work_units(&kept, FanOut::SubRanges, POOL);
        assert_eq!(units.len(), POOL, "merged superfile takes the whole pool");

        // What the slices collectively ask for, computed exactly as the
        // ranged kernels do — each scoped to its own sub-range.
        let requested: usize = units
            .iter()
            .map(|u| top_k_initial_capacity(K, u.entry.n_docs, u.range))
            .sum();
        // What the query can possibly need: the slices tile the doc space
        // once, so no more than `min(k, docs)` distinct docs are rankable.
        let needed = top_k_initial_capacity(K, MERGED_DOCS, None);

        let mib = |slots: usize| (slots * SLOT_BYTES) as f64 / (1024.0 * 1024.0);
        assert_eq!(
            requested,
            needed,
            "sliced fan-out requested {requested} top-k slots ({:.1} MiB) for \
             a doc space needing {needed} ({:.1} MiB): a slice must be sized \
             by its own range, not by the whole superfile",
            mib(requested),
            mib(needed),
        );
    }

    /// Control for the regression above: the same corpus and the same `k`
    /// on the un-sliced path allocate exactly once. This pinned the
    /// blow-up to slicing rather than to large `k` on its own, and now
    /// pins the fixed sliced case to the same total.
    #[test]
    fn unsliced_fanout_preallocates_one_top_k_heap_per_superfile() {
        /// Docs in the merged superfile a full optimize produces.
        const MERGED_DOCS: u64 = 1_000_000;
        /// Same large `k` as the sliced repro.
        const K: usize = MERGED_DOCS as usize;
        /// Reader-pool threads; irrelevant under `PerSuperfile`.
        const POOL: usize = 8;

        let merged = manifest_entry(MERGED_DOCS);
        let kept = vec![&merged];
        // A query carrying a must or a negation stays whole-superfile.
        let units = build_work_units(&kept, FanOut::PerSuperfile, POOL);
        assert_eq!(units.len(), 1, "un-ranged fan-out is one unit per file");

        let requested: usize = units
            .iter()
            .map(|u| top_k_initial_capacity(K, u.entry.n_docs, u.range))
            .sum();
        assert_eq!(
            requested,
            top_k_initial_capacity(K, MERGED_DOCS, None),
            "un-sliced scan allocates exactly the docs it can rank"
        );
    }

    /// Pins the documented "target, not a budget" slop: per-file
    /// round-half-up with the ≥ 1 clamp may emit up to `kept − 1` units
    /// beyond the pool. Three equal files on an 8-thread pool round to
    /// 3 slices each; the excess unit queues, it never adds concurrency.
    #[test]
    fn build_work_units_may_oversubscribe_pool_by_kept_minus_one() {
        /// Threads in the simulated reader pool.
        const POOL: usize = 8;
        /// Docs per superfile — equal thirds, each well above the
        /// `SUBRANGE_MIN_DOCS` floor so rounding alone decides.
        const DOCS_EACH: u64 = 400_000;

        let files = [
            manifest_entry(DOCS_EACH),
            manifest_entry(DOCS_EACH),
            manifest_entry(DOCS_EACH),
        ];
        let kept: Vec<_> = files.iter().collect();
        let units = build_work_units(&kept, FanOut::SubRanges, POOL);
        // round(1/3 × 8) = 3 slices per file.
        assert_eq!(units.len(), 9, "3 equal files each round to 3 slices");
        assert!(
            units.len() <= POOL + (kept.len() - 1),
            "oversubscription is bounded by kept − 1"
        );
        // Every file's slices still tile its own doc space exactly.
        for f in &files {
            let mut ranges: Vec<(u32, u32)> = units
                .iter()
                .filter(|u| u.entry.superfile_id == f.superfile_id)
                .map(|u| u.range.expect("equal thirds are sliced"))
                .collect();
            ranges.sort_unstable();
            let mut cursor = 0u32;
            for (start, end) in ranges {
                assert_eq!(start, cursor, "sub-ranges tile without gaps");
                cursor = end;
            }
            assert_eq!(cursor, DOCS_EACH as u32, "sub-ranges cover every doc");
        }
    }

    #[test]
    fn build_work_units_slices_large_superfiles_when_threads_spare() {
        use std::collections::HashMap;

        use uuid::Uuid;

        use crate::supertable::manifest::{SuperfileEntry, SuperfileUri};

        let id = Uuid::new_v4();
        // One large superfile, well above SUBRANGE_MIN_DOCS (50k).
        let big = Arc::new(SuperfileEntry {
            physical_schema: None,
            stem: None,
            birth_version: 0,
            superfile_id: id,
            uri: SuperfileUri(id),
            n_docs: 200_000,
            id_min: 0,
            id_max: 199_999,
            scalar_stats: ScalarStats::default(),
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
        let r = st.reader().expect("reader");
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
        let r = st.reader().expect("reader");
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
        let r = st.reader().expect("reader");
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

    /// Under global statistics on a table whose term index is complete,
    /// each scored term's corpus-wide df is summed from the index and no
    /// superfile is opened for it: the idf is exactly what summing every
    /// superfile's own dictionary gives, at zero opens.
    #[test]
    fn global_idf_comes_from_a_complete_term_index_without_opening_a_superfile() {
        use crate::{
            runtime_metrics::op_stats::with_op_stats,
            superfile::{SuperfileReader, fts::bm25},
        };

        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_one_superfile_per_commit().with_storage(storage))
            .expect("create");
        // Three commits, one superfile each; `alpha` in two, `beta` in one,
        // `shared` in all, `absent` in none.
        for titles in [
            &["alpha shared one", "shared two"][..],
            &["beta shared three", "alpha shared four", "shared five"][..],
            &["shared six"][..],
        ] {
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(0, titles)).expect("append");
            w.commit().expect("commit");
        }
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        assert!(manifest.term_index_complete(), "every commit contributed");
        let entries = manifest.get_all_superfiles().to_vec();
        let terms = ["alpha", "beta", "shared", "absent"];

        // The oracle: every superfile's own dictionary, summed.
        let rt = Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rt");
        let mut expected_df = [0u64; 4];
        for e in &entries {
            let bytes = std::fs::read(dir.path().join(e.uri.storage_path())).expect("bytes");
            let sf = SuperfileReader::open(Bytes::from(bytes)).expect("open");
            let (dfs, _) = rt.block_on(sf.term_dfs("title", &terms)).expect("dfs");
            for (i, d) in dfs.into_iter().enumerate() {
                expected_df[i] += d;
            }
        }
        assert_eq!(expected_df, [2, 1, 6, 0]);
        let corpus = manifest.fts_length_stats("title").expect("table totals");

        let owned: Vec<String> = terms.iter().map(|t| (*t).to_owned()).collect();
        let ((idf, memos), opened) = with_op_stats(|| {
            let (map, memos) = rt
                .block_on(reader.global_idf_open_wave(manifest, "title", &owned, &entries, corpus))
                .expect("wave");
            let opened = crate::runtime_metrics::op_stats::current()
                .expect("metered")
                .score_pruning_survived();
            ((map, memos), opened)
        })
        .0;
        assert_eq!(opened, 0, "the index answers; no superfile is opened");
        assert!(memos.is_none(), "no open wave, so no open-wave memos");
        for (i, t) in terms.iter().enumerate() {
            assert_eq!(
                idf[*t],
                bm25::idf(corpus.n_scored_docs, expected_df[i]),
                "idf of `{t}` from the index equals the dictionaries' sum"
            );
        }
    }
}
