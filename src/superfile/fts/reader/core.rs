// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS blob reader. Multi-column BM25 search.
//!
//! Opens the byte layout produced by [`crate::superfile::fts::builder::FtsBuilder::finish`]
//! and exposes BM25 search per-column or weighted across columns.
//!
//! See `docs/architecture/superfile.md` for the on-disk layout.
//!
//! ## Threading
//!
//! `FtsReader` is `Send + Sync` and immutable after `open()` — concurrent
//! `search` calls share the underlying `Bytes`. The term dictionary view
//! is constructed per call (cheap; opening it reads only its footer, and
//! then it's a borrowed view).

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    mem,
    num::NonZeroUsize,
    ops::Range,
    sync::{Arc, Mutex, OnceLock, PoisonError},
    thread::available_parallelism,
};

use bytes::Bytes;
use futures::future::join;
use roaring::RoaringBitmap;
use rustc_hash::FxHashMap;
use tokio::sync::OnceCell;

use super::{
    cursor::{BlockTableCache, CursorUse, TermCursor, TermMeta},
    filter::ExcludeFilter,
    metadata::{
        ColumnLengthStats, ColumnMeta, FtsColumnConfig, LENGTH_ARRAY, OpenOptions, StoredNorms,
    },
    phrase::{AnyCursor, PhraseCursor, distinct_members},
    scorers::{count_and_planned, cursors_max_doc, df_total},
    search::FetchedTermMemo,
    sink::{LiveFloor, TopKEntry, drain_top_k_desc},
    work::{term_cursor_bytes, term_cursor_ranges},
};
use crate::{
    superfile::{
        ReadError,
        error::FtsError,
        format::{
            self, KEY_SEPARATOR,
            checksum::crc32c,
            fts::{HEADER_SIZE, MAGIC_BYTES, U32_BYTES, U64_BYTES, VERSION_MIN, hdr, term_meta},
        },
        fts::{
            analysis::{chain_tokenizer, check_recorded_tokenizer},
            bm25,
            builder::doc_lengths_entry_size,
            positions::GroupIndex,
            posting::{BLOCK_LEN, ENCODING_BITSET, ENCODING_RUN, decode_block_doc_ids},
            short::{SHORT_MAX_DF, decode_short},
            tokenize::{Phrase, Tokenizer},
        },
        id_space::{DocMap, FtsDocId, RowId},
        lazy_source::{LazyByteSource, PrefetchedSource, RangeCoalescePlan, Source},
    },
    supertable::schema::FieldId,
    utils::terms::{DictEntry, TermBlocks, make_key},
};

/// Largest gap worth overfetching when adjacent term postings share a request.
const TERM_RANGE_COALESCE_MAX_GAP: usize = 64 * 1024;
/// Maximum total gap bytes tolerated in one coalesced postings request.
const TERM_RANGE_COALESCE_MAX_OVERFETCH: usize = 512 * 1024;

/// What a query scores a superfile at in place of what its columns
/// declare. The default changes nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ScoringView<'a> {
    /// The BM25 parameters for every column.
    pub params: Option<bm25::Bm25Params>,
    /// The average document length for the one column named, as this file
    /// labels it.
    pub avgdl: Option<(&'a str, f32)>,
}

/// Per-term table-wide BM25 idf, keyed by term.
pub(crate) type GlobalTermIdf = std::collections::HashMap<String, f32>;

/// The corpus a query's terms are weighted against.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum QueryIdf<'a> {
    /// The whole table's: the map holds every scored term.
    Table(&'a GlobalTermIdf),
    /// This superfile's own statistics, for a superfile searched by itself,
    /// which is then the whole corpus.
    #[default]
    ThisFile,
}

impl QueryIdf<'_> {
    /// The idf source for one `term`.
    pub(crate) fn term(self, term: &str) -> TermIdf {
        match self {
            QueryIdf::Table(map) => map
                .get(term)
                .copied()
                .map_or(TermIdf::Absent, TermIdf::Table),
            QueryIdf::ThisFile => TermIdf::ThisFile,
        }
    }
}

/// The idf one term is weighted with.
#[derive(Clone, Copy, Debug)]
pub(crate) enum TermIdf {
    /// The table-wide value.
    Table(f32),
    /// This superfile's own, from its document frequency here.
    ThisFile,
    /// None was supplied: fine for a cursor that never scores, an error for
    /// one that does.
    Absent,
}

impl TermIdf {
    /// The idf to score with, given the term's document frequency `df` in
    /// this superfile's column `col`.
    pub(super) fn resolve(self, col: &ColumnMeta, df: u64) -> Result<f32, FtsError> {
        match self {
            TermIdf::Table(idf) => Ok(idf),
            TermIdf::ThisFile => Ok(bm25::idf(col.scored_doc_count(), df)),
            TermIdf::Absent => Err(FtsError::MissingIdf),
        }
    }
}

/// A query's parsed clause lists, borrowed for one search call —
/// terms and phrases per polarity, with the default operator already
/// resolved (see `ParsedQuery::into_clauses`). Grouped so the search
/// entry points don't take nine parallel parameters.
#[derive(Default)]
pub(crate) struct ClauseLists<'a> {
    pub musts: &'a [&'a str],
    pub shoulds: &'a [&'a str],
    pub negatives: &'a [&'a str],
    pub must_phrases: &'a [Phrase<String>],
    pub should_phrases: &'a [Phrase<String>],
    pub negative_phrases: &'a [Phrase<String>],
    /// The corpus the scored terms are weighted against.
    pub idf: QueryIdf<'a>,
    /// Open-wave fetches for the scored terms (global stats): the
    /// cursor builds serve these terms from the memo instead of
    /// re-reading the dictionary and postings ranges the df gather
    /// already fetched. `None` on the single-pass path.
    pub prefetched: Option<&'a FetchedTermMemo>,
    /// The query fan-out's live cross-segment floor (see [`LiveFloor`]):
    /// the atom walks re-read it per candidate — pruning phrase verify
    /// work against every segment's published kth-best, not just the
    /// snapshot taken at segment start — and publish their own local
    /// kth into it as their heaps fill. `None` on single-superfile
    /// entry points, where there is no fan-out to share with.
    pub live_floor: Option<&'a LiveFloor>,
    /// The rows this search may rank at all, as this superfile's
    /// `local_doc_id`s — a SQL `WHERE` pushed into the search and
    /// resolved against the index. Every kernel admits a doc into its
    /// heap only if the set holds it, so the top-k fills from allowed
    /// rows instead of being a post-filtered global top-k that
    /// underflows. `None` ranks every matching row. Owned (`Arc`), not
    /// borrowed: the prepared clauses carry it into a `'static` kernel.
    pub allow: Option<Arc<RoaringBitmap>>,
    /// Score every match in one pass and count them, instead of pruning
    /// to the top-k. Set by the exact-count entry points, which pair it
    /// with `run_prepared_with_count`.
    pub exact_count: bool,
}

impl ClauseLists<'_> {
    /// Any phrase atom anywhere routes the query to the atom walks.
    pub(super) fn has_phrases(&self) -> bool {
        !self.must_phrases.is_empty()
            || !self.should_phrases.is_empty()
            || !self.negative_phrases.is_empty()
    }

    /// Nothing to rank or match on the positive side.
    pub(super) fn no_positive_atoms(&self) -> bool {
        self.musts.is_empty()
            && self.shoulds.is_empty()
            && self.must_phrases.is_empty()
            && self.should_phrases.is_empty()
    }

    /// Nothing negated either.
    pub(super) fn no_negative_atoms(&self) -> bool {
        self.negatives.is_empty() && self.negative_phrases.is_empty()
    }
}

/// Output of [`FtsReader::prepare_clauses`], consumed by
/// [`FtsReader::run_prepared`]. Either an already-final result or the
/// cursors for one clause shape still to score. Owns its `ExcludeFilter`
/// rather than borrowing it, so it can move into a `'static` closure.
pub(crate) enum PreparedClauses {
    /// Already final — nothing left for `run_prepared` to do. Carries
    /// the posting (and phrase-position) bytes the inline walk indexed
    /// into, so the fast paths report work like the cursor-carrying
    /// shapes do.
    Done {
        hits: Vec<(FtsDocId, f32)>,
        /// Exact match count when the inline walk ran under
        /// `ClauseLists::exact_count`; 0 otherwise.
        count: u64,
        /// Whether the clauses were prepared for the exact-count walk.
        exact_count: bool,
        postings_bytes: u64,
        /// Byte-source ranges the inline walk requested (0 for the df=1
        /// inline-entry and empty-resolution paths).
        planned_ranges: u64,
        /// On-CPU nanoseconds of the walk that produced `hits` inside
        /// `prepare_clauses` (single-term BMW, atoms search) — the
        /// kernel time `run_prepared` never sees for already-final
        /// shapes. 0 for the trivial early returns.
        kernel_cpu_ns: u64,
    },
    /// AND-only: intersect `must_cursors`.
    Must {
        column_id: u32,
        must_cursors: Vec<TermCursor>,
        filter: Option<ExcludeFilter>,
        k: usize,
        floor_eff: f32,
        /// term-dictionary ranges the builds requested (one per
        /// `build_term_cursors` call — must / should / negation lists).
        dict_ranges: u64,
        /// See [`PreparedClauses::Done::exact_count`].
        exact_count: bool,
    },
    /// AND with should-boosted scoring.
    MustShould {
        column_id: u32,
        must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        filter: Option<ExcludeFilter>,
        k: usize,
        floor_eff: f32,
        /// See [`PreparedClauses::Must::dict_ranges`].
        dict_ranges: u64,
        /// See [`PreparedClauses::Done::exact_count`].
        exact_count: bool,
    },
    /// Plain multi-term OR (no musts) — algorithm choice resolved in
    /// `run_prepared`.
    Or {
        column_id: u32,
        cursors: Vec<TermCursor>,
        filter: Option<ExcludeFilter>,
        k: usize,
        floor_eff: f32,
        /// See [`PreparedClauses::Must::dict_ranges`].
        dict_ranges: u64,
        /// See [`PreparedClauses::Done::exact_count`].
        exact_count: bool,
    },
}

impl PreparedClauses {
    /// Whether scoring these clauses can return a hit, which whoever runs
    /// them translates to a row through the doc-id map.
    pub(crate) fn may_hit(&self) -> bool {
        match self {
            PreparedClauses::Done { hits, .. } => !hits.is_empty(),
            PreparedClauses::Must { must_cursors, .. } => !must_cursors.is_empty(),
            PreparedClauses::MustShould { .. } => true,
            PreparedClauses::Or { cursors, .. } => !cursors.is_empty(),
        }
    }

    /// Whether the clauses were prepared for the exact-count walk.
    pub(crate) fn exact_count(&self) -> bool {
        match self {
            PreparedClauses::Done { exact_count, .. }
            | PreparedClauses::Must { exact_count, .. }
            | PreparedClauses::MustShould { exact_count, .. }
            | PreparedClauses::Or { exact_count, .. } => *exact_count,
        }
    }

    /// Scan-cost proxy callers gate reader-pool dispatch on: the driving
    /// (smallest) posting list for the AND-intersect shapes, the full
    /// union for OR. `Done` has nothing left to scan, so it's zero.
    pub(crate) fn posting_mass(&self) -> u64 {
        match self {
            PreparedClauses::Done { .. } => 0,
            PreparedClauses::Must { must_cursors, .. } => {
                must_cursors.iter().map(|c| c.df).min().unwrap_or(0)
            }
            PreparedClauses::MustShould { must_cursors, .. } => {
                must_cursors.iter().map(|c| c.df).min().unwrap_or(0)
            }
            PreparedClauses::Or { cursors, .. } => df_total(cursors),
        }
    }

    /// Posting-list bytes resident for this prepared query — what the
    /// kernels index into across musts, shoulds, OR terms, and negation
    /// filters (plus phrase position runs on the inline `Done` path).
    /// Deterministic for a given query against a given superfile (cache
    /// temperature never changes it) — the per-query work stats flush
    /// this once per superfile.
    pub(crate) fn postings_bytes(&self) -> u64 {
        let filter_bytes =
            |filter: &Option<ExcludeFilter>| filter.as_ref().map_or(0, |f| f.postings_bytes());
        match self {
            PreparedClauses::Done { postings_bytes, .. } => *postings_bytes,
            PreparedClauses::Must {
                must_cursors,
                filter,
                ..
            } => term_cursor_bytes(must_cursors) + filter_bytes(filter),
            PreparedClauses::MustShould {
                must_cursors,
                should_cursors,
                filter,
                ..
            } => {
                term_cursor_bytes(must_cursors)
                    + term_cursor_bytes(should_cursors)
                    + filter_bytes(filter)
            }
            PreparedClauses::Or {
                cursors, filter, ..
            } => term_cursor_bytes(cursors) + filter_bytes(filter),
        }
    }

    /// On-CPU nanoseconds already spent producing an inline `Done`
    /// result (0 for the cursor-carrying shapes, whose kernels are
    /// bracketed at `run_prepared`).
    pub(crate) fn inline_kernel_cpu_ns(&self) -> u64 {
        match self {
            PreparedClauses::Done { kernel_cpu_ns, .. } => *kernel_cpu_ns,
            _ => 0,
        }
    }

    /// Byte-source ranges this prepared query requested — one per term
    /// posting range across every clause list (see
    /// [`Self::postings_bytes`] for the byte-volume counterpart).
    pub(crate) fn planned_ranges(&self) -> u64 {
        let filter_ranges = |filter: &Option<ExcludeFilter>| {
            filter.as_ref().map_or(0, ExcludeFilter::planned_ranges)
        };
        match self {
            PreparedClauses::Done { planned_ranges, .. } => *planned_ranges,
            PreparedClauses::Must {
                must_cursors,
                filter,
                dict_ranges,
                ..
            } => term_cursor_ranges(must_cursors) + filter_ranges(filter) + dict_ranges,
            PreparedClauses::MustShould {
                must_cursors,
                should_cursors,
                filter,
                dict_ranges,
                ..
            } => {
                term_cursor_ranges(must_cursors)
                    + term_cursor_ranges(should_cursors)
                    + filter_ranges(filter)
                    + dict_ranges
            }
            PreparedClauses::Or {
                cursors,
                filter,
                dict_ranges,
                ..
            } => term_cursor_ranges(cursors) + filter_ranges(filter) + dict_ranges,
        }
    }
}

/// Multi-term OR algorithm selector for the bench harness's
/// `search_with_algo_for_bench` entry point, so head-to-head bench runs can
/// compare the kernels under identical inputs. Production code routes through
/// `FtsReader::dispatch_or_algo`: the rare-led walks for a union with a rare
/// term at small `k`, else `WindowedMaxscore`.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OrAlgo {
    /// Per-candidate Block-Max MaxScore. Superseded on the production path by
    /// `WindowedMaxscore`; retained as a bench baseline and the correctness
    /// oracle the windowed kernels are checked against.
    Bmm,
    /// WAND + Block-Max-WAND. In production, the fallback for a 2-term
    /// rare+common OR the rare-led walk declines (it pivots on the rare term to
    /// skip the common term's long list); a bench baseline otherwise.
    WandBmw,
    /// Exhaustive union walk with SIMD scoring + top-K heap: the unpruned
    /// baseline. Nothing on a query path routes here.
    Exhaustive,
    /// Windowed union: accumulate each term's contribution into a fixed doc-id
    /// window (presence bitset + score array), then drain in doc order into the
    /// top-k heap. Superseded by `WindowedMaxscore` (which adds the
    /// essential/non-essential split); bench-only.
    Windowed,
    /// Windowed MaxScore: **the production union kernel.** The windowed OR-sum
    /// with a per-window essential/non-essential split recomputed from the live
    /// threshold, so one kernel covers both the dense (all-essential OR-sum)
    /// and selective (pruned-leader) regimes without an a-priori route.
    WindowedMaxscore,
}

/// Doc-id window for the windowed union scorer. Power of two so the
/// window base is a cheap mask. At 8192 the per-window state — an
/// `8192 × f32` score accumulator (32 KiB) plus an `8192`-bit presence
/// bitset (1 KiB) — stays L1/L2-resident across the accumulate + drain
/// passes, while a union of common terms pays its per-window costs (the
/// smallest-doc scan, a cursor's step into and out of the window, the
/// drain's pass over the presence words) seldom.
pub(super) const OR_WINDOW: u32 = 8192;
/// Number of 64-bit words in the window presence bitset.
pub(super) const OR_WINDOW_WORDS: usize = (OR_WINDOW as usize).div_ceil(64);

/// Terms from which an unranked union may walk its cursors through a heap
/// keyed by current doc rather than rescanning every cursor per emitted
/// doc. A rescan costs the term count n per doc; the heap costs log n
/// per posting, so a doc matched by m of the terms costs m·log n. The
/// heap therefore loses when the average doc matches more than about
/// n / log n of the terms, which a few overlapping common terms do and a
/// long list of rare ones does not; a union of n sparse terms (an `IN`
/// list of ids, say) costs n² under the rescan and n·log n under the
/// heap. From this many terms on, each walk compares the two by its
/// postings and its rescans ([`or_walk_prefers_heap`],
/// [`or_count_prefers_heap`]); below it the rescan is kept outright.
pub(super) const OR_HEAP_MIN_TERMS: usize = 8;

/// Dominance threshold for the df-anchored union count. When one term's
/// document frequency is at least this multiple of *all the other terms'
/// dfs combined*, a disjunction count is cheaper computed as
/// `df(dominant) + |others \ dominant|` — the dominant term's df is read
/// from its header (no decode) and its skip table is traversed once to
/// membership-test the rarer docs — than by walking every posting of
/// every term.
///
/// The anchored path does one skip-probe per doc in the *others* union,
/// and a skip-probe costs several times what the windowed walk's linear
/// `next()` does. So it only pays when the dominant list — the postings
/// the anchor avoids decoding — is larger than the others' combined df by
/// more than that per-probe penalty. Measurement puts the crossover
/// between ~6× (where the probe cost still loses) and ~130× (a clear
/// win); this threshold sits above the loss region so only a decisively
/// dominant term takes the anchored path. Balanced unions keep the
/// windowed bitset, which is already faster when no term dominates.
pub(super) const OR_COUNT_ANCHOR_DOMINANCE: u64 = 8;

/// Density gate for the full-bitset union count. When the terms' combined
/// postings reach `1/N` of the doc-id space, OR them all into one
/// doc-space bitset and popcount, instead of the per-window walk — a dense
/// union (several common terms, e.g. a 3-4 word title) touches most of the
/// corpus, so the bitset streams the doc ids into memory with no per-doc
/// window bookkeeping. Below the gate the union is sparse and the windowed
/// bitset — whose state stays L1-resident and needs no full-corpus alloc +
/// popcount — is cheaper. Expressed as the divisor `N`: bitset when
/// `total_df >= max_doc / N`.
pub(super) const OR_COUNT_BITSET_DENSITY_DIVISOR: u64 = 16;

/// Multi-term OR dispatch floor. A 2-term OR is already sub-millisecond
/// on MaxScore, so the window's per-window bookkeeping isn't worth it
/// below this many terms. `pub(crate)`: the supertable fan-out reuses
/// this same boundary to decide when a ranged kernel is heavy enough to
/// ship to the reader pool (see `RANGED_KERNEL_POOL_MIN_TERMS`).
pub(crate) const OR_WINDOW_MIN_TERMS: usize = 3;

/// Largest `k` for which a 2-term OR routes to WAND+BMW instead of
/// MaxScore. WAND's pivot pruning needs a high top-k threshold to skip
/// blocks: at small `k` the threshold is high and it clears MaxScore
/// decisively on two comparable terms, but as `k` grows the threshold
/// falls until WAND can no longer prune and its per-iteration cursor
/// re-sort becomes pure overhead — so above this `k` MaxScore wins. The
/// cutoff sits between the common small-`k` page sizes and the rare deep
/// `k`; large-`k` 2-term ORs stay on MaxScore.
pub(super) const WAND_BMW_2TERM_MAX_K: usize = 128;

/// Route a 2-term OR to WAND+BMW only when one term's posting list is at
/// least this many times shorter than the other's (df ratio). That rare
/// "anchor" term is what lets WAND pivot and skip the common term's long
/// list — the source of its win. Two comparable-length lists (e.g. two
/// common words) give WAND nothing to skip, so it loses to MaxScore and
/// stays there. A *score* upper-bound ratio is the wrong test here: a
/// term can dominate the BM25 UB (higher idf) while still being common
/// (long list), which WAND can't skip — only df separates the cases.
pub(super) const WAND_BMW_2TERM_DF_RATIO: u64 = 16;

/// Largest rare-term document frequency the rare-led pair pass ranks in
/// full before falling back. Ranking those documents is the walk the
/// two-term WAND ends in anyway, so the pass costs extra only when it has to
/// fall back; the cap bounds that waste for a "rare" term that is merely
/// less common.
pub(super) const RARE_LED_MAX_DF: u64 = 1 << 17;

/// Postings-equivalent cost of testing one document against one term in a
/// count: a block locate and, on a packed block, a decode. Weighs probing a
/// rare term's documents against folding every term into a bitset.
pub(super) const COUNT_PROBE_POSTINGS: u64 = 40;

/// Initial capacity for a scan's top-k heap, in [`TopKEntry`] slots.
///
/// `docs_in_scope` bounds the distinct doc_ids that can ever enter the
/// heap. It exists because callers may pass `k = usize::MAX`
/// (`search_multi` gathers every match before weighting across
/// columns), and `usize::MAX * size_of::<TopKEntry>()` is not an
/// allocation any machine will serve; the heap still grows on demand.
///
/// `range` is the doc-id window the scan will visit; `None` is a
/// whole-superfile scan, whose scope is `n_docs`. **Every ranged kernel
/// must pass its own `Some((start, end))`** — a slice can only rank the
/// docs inside its window, so sizing it by `n_docs` instead makes a
/// sliced fan-out preallocate `slices × min(k, n_docs)` slots for a doc
/// space its slices collectively walk exactly once. That is a
/// pool-sized multiple on a compacted table, where doc-mass allocation
/// hands one merged superfile the entire reader pool: measured at 1M
/// docs × 8 threads as 61 MiB requested against 7.6 MiB rankable.
/// Guarded by `ranged_slice_heaps_are_sized_by_their_own_range`.
///
/// An un-ranged caller that still has a window handy may pass it — the
/// `min` against `n_docs` makes `Some((0, u32::MAX))` and `None`
/// equivalent.
pub(crate) fn top_k_initial_capacity(k: usize, n_docs: u64, range: Option<(u32, u32)>) -> usize {
    let docs_in_scope = match range {
        Some((start, end)) => (end.saturating_sub(start) as usize).min(n_docs as usize),
        None => n_docs as usize,
    };
    k.min(docs_in_scope).max(1)
}

/// True for a 2-term cursor set where one term's posting list is at least
/// [`WAND_BMW_2TERM_DF_RATIO`]× shorter than the other's — a rare anchor
/// WAND+BMW can pivot on to skip the common term's long list. The whole
/// reason to prefer WAND over MaxScore on a 2-term OR.
pub(super) fn two_term_has_rare_anchor(cursors: &[TermCursor]) -> bool {
    if cursors.len() != 2 {
        return false;
    }
    let lo = cursors[0].df.min(cursors[1].df);
    let hi = cursors[0].df.max(cursors[1].df);
    lo > 0 && hi >= lo.saturating_mul(WAND_BMW_2TERM_DF_RATIO)
}

/// FTS blob reader. Self-contained — owns its `Bytes` (which the storage
/// layer assembled from mmap / range-fetch / full-read).
#[derive(Debug, Clone)]
pub struct FtsReader {
    pub(super) source: Source,
    pub(super) n_docs: u32,
    pub(super) n_terms_total: u32,
    pub(super) dict_range: Range<usize>,
    pub(super) postings_range: Range<usize>,
    /// Byte range of the positions region (CRC stripped). Phrase queries
    /// fetch per-term group ranges out of it via
    /// [`Self::fetch_term_positions`].
    pub(super) positions_range: Range<usize>,
    /// The blob version from the header, kept verbatim: `V7` and `V8`
    /// decode alike, but a migration reports and plans on which one a file
    /// actually is.
    pub(super) version: u32,
    /// Long terms' decoded block tables, shared by every clone of this
    /// reader (see [`BlockTableCache`]).
    pub(super) block_tables: Arc<BlockTableCache>,
    pub(super) columns: Vec<ColumnMeta>,
    pub(super) column_id_by_name: HashMap<String, u32>,
    /// The Parquet row each doc id in this blob stands for. The
    /// identity on a `V7` blob, where a doc id already is a row.
    ///
    /// The kernels never consult it: they walk, score and prune in the
    /// blob's own id space from end to end, and the ids they return are
    /// translated once here. That is what lets the writer group
    /// documents by the terms they share without a Parquet row moving.
    ///
    /// Set at open when the blob is in memory or keeps row order, or when
    /// the bytes the open read already hold it. Otherwise a query reads it
    /// whole before it translates a doc id ([`Self::ensure_doc_map`]): at 4
    /// bytes a document it is hundreds of megabytes on a large file, which an
    /// open, and so a SQL scan that matches no text, should not pay for.
    /// Shared by every view cloned from this reader ([`Self::with_scoring`]),
    /// so one read serves them all.
    doc_map: Arc<OnceCell<DocMap>>,
    /// Where the map's body lies, with its CRC right after it; `None` on a
    /// blob in row order.
    doc_map_region: Option<Range<usize>>,
    /// Whether a read after open checks its CRC, as the open's reads did.
    verify_crc: bool,
}

impl FtsReader {
    /// A view of this reader that scores at `scoring` instead of what its
    /// columns declare, or `None` when that changes no column.
    ///
    /// Two things change per column, and nothing else: the norm decode
    /// table is re-derived at the new parameters and average length (the
    /// per-doc length buckets are shared, not copied), and
    /// [`ColumnMeta::bound_scale`] picks up the factor that keeps the
    /// *stored* bounds, which belong to what the file declares, upper bounds
    /// under the new values. Results stay exact; only pruning power is
    /// traded.
    pub(crate) fn with_scoring(&self, scoring: &ScoringView<'_>) -> Option<Self> {
        let target = |col: &ColumnMeta| {
            let params = scoring.params.unwrap_or(col.params);
            let avgdl = match scoring.avgdl {
                Some((column, avgdl)) if column == col.name => avgdl,
                _ => col.score_avgdl,
            };
            (params, avgdl)
        };
        if self
            .columns
            .iter()
            .all(|col| target(col) == (col.params, col.score_avgdl))
        {
            return None;
        }
        let mut view = self.clone();
        for col in &mut view.columns {
            let (params, avgdl) = target(col);
            if (params, avgdl) == (col.params, col.score_avgdl) {
                continue;
            }
            // The view's norms are derived from the declared ones on first
            // use (`ColumnMeta::norms`), composing the bound factor rather
            // than replacing it.
            col.params = params;
            col.score_avgdl = avgdl;
            col.view_norms = Arc::new(OnceLock::new());
        }
        Some(view)
    }

    /// [`Self::with_scoring`] at `params` alone, unchanged when no column
    /// moves.
    #[cfg(test)]
    pub(crate) fn with_bm25_override(&self, params: bm25::Bm25Params) -> Self {
        let scoring = ScoringView {
            params: Some(params),
            avgdl: None,
        };
        self.with_scoring(&scoring).unwrap_or_else(|| self.clone())
    }

    /// Open with default options (CRC verification on).
    pub fn open(blob: Bytes, columns_json: &str) -> Result<Self, FtsError> {
        Self::open_with(blob, columns_json, OpenOptions::default())
    }

    /// Open with explicit options. Pass
    /// `OpenOptions { verify_crc: false }` to skip the
    /// four per-section CRC scans on trusted-storage cold
    /// opens.
    pub fn open_with(blob: Bytes, columns_json: &str, opts: OpenOptions) -> Result<Self, FtsError> {
        Self::open_with_source(Source::InMemory(blob), columns_json, opts)
    }

    /// Open from a range source without materializing the FTS
    /// subsection. The open reads only the fixed header and the doc-length
    /// directory, which says where each column's length array sits. The
    /// dictionary is fetched on the first lookup that needs it, each
    /// column's length array on its first scored use, and each query term's
    /// postings on demand by [`Self::fetch_term_postings`], mirroring how
    /// the vector reader fetches only probed clusters.
    pub async fn open_lazy(
        source: Arc<dyn LazyByteSource>,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, FtsError> {
        // Length of the FTS subsection itself (≈ `kv::FTS_LENGTH`), not
        // the whole superfile: `source` is the FTS-scoped sub-source.
        let fts_blob_len = source.size() as usize;
        // One GET covers the header, at the width of the widest version
        // rather than of this blob's: the version is inside the bytes being
        // fetched, so sizing the read by it would cost a round trip to learn
        // what the read should have been. Overfetching at most eight bytes on
        // an older blob is the cheaper mistake; fetching twice would break
        // the coalesced group a cold query's reads form, and the same bytes
        // would then arrive as two requests instead of one.
        let header_fetch = HEADER_SIZE.min(fts_blob_len);
        let header = fetch_lazy_range(source.as_ref(), 0..header_fetch, "fts header").await?;
        if header.len() < HEADER_SIZE {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }
        if &header[0..MAGIC_BYTES] != format::fts::MAGIC {
            return Err(FtsError::Read(ReadError::BadMagic {
                section: "fts",
                expected: format::fts::MAGIC,
                actual: header[0..MAGIC_BYTES].to_vec(),
            }));
        }
        let version = read_u32_le(&header[hdr::VERSION_OFF..hdr::VERSION_OFF + U32_BYTES]);
        check_version(version)?;

        let doc_lengths_table_offset =
            read_u64_le(&header[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;
        let n_columns =
            read_u32_le(&header[hdr::N_COLUMNS_OFF..hdr::N_COLUMNS_OFF + U32_BYTES]) as usize;
        // Besides the header, the sync open reads one more thing eagerly:
        // the doc-lengths directory (one small entry per column, plus its
        // CRC), which says where each column's length array sits and the
        // average it was scored at. Everything else stays lazy — the
        // dictionary is fetched on the first lookup that needs it (a
        // query a table-level term index resolves never does), each
        // column's length array on its first scored use, and postings per
        // term — so a cold open costs the header and the directory, both
        // of which a manifest that carries them serves without a read.
        let dir_len = n_columns * doc_lengths_entry_size(version) + 4;
        // Start the span at the doc-id map's length trailer rather than at the
        // directory: it is the eight bytes immediately before, so the open
        // reads it in the same request instead of a second one.
        let dir_start =
            doc_lengths_table_offset.saturating_sub(format::fts::doc_map_len_trailer(version));
        let dir_end = doc_lengths_table_offset
            .saturating_add(dir_len)
            .min(fts_blob_len);
        let directory =
            fetch_lazy_range(source.as_ref(), dir_start..dir_end, "fts/doc_lengths_dir").await?;
        let mut overlay = PrefetchedSource::new(Arc::clone(&source));
        overlay.install(0, header);
        overlay.install(dir_start as u64, directory);
        let reader = Self::open_with_source(Source::Lazy(Arc::new(overlay)), columns_json, opts)?;

        Ok(reader)
    }

    /// Open over an arbitrary byte source. The eager path wraps a
    /// full subsection as [`Source::InMemory`]; lazy callers can pass
    /// a range-backed source without changing the public search API.
    pub(crate) fn open_with_source(
        source: Source,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, FtsError> {
        let source_len = source.len();
        if source_len < HEADER_SIZE {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }
        let header = fetch_source_range(&source, 0..HEADER_SIZE, "fts header")?;

        // Magic check.
        if &header[0..MAGIC_BYTES] != format::fts::MAGIC {
            return Err(FtsError::Read(ReadError::BadMagic {
                section: "fts",
                expected: format::fts::MAGIC,
                actual: header[0..MAGIC_BYTES].to_vec(),
            }));
        }

        let version = read_u32_le(&header[hdr::VERSION_OFF..hdr::VERSION_OFF + U32_BYTES]);
        check_version(version)?;

        let n_columns =
            read_u32_le(&header[hdr::N_COLUMNS_OFF..hdr::N_COLUMNS_OFF + U32_BYTES]) as usize;
        let n_docs = read_u32_le(&header[hdr::N_DOCS_OFF..hdr::N_DOCS_OFF + U32_BYTES]);
        let n_terms_total = read_u32_le(&header[hdr::N_TERMS_OFF..hdr::N_TERMS_OFF + U32_BYTES]);
        let dict_offset =
            read_u64_le(&header[hdr::DICT_OFFSET_OFF..hdr::DICT_OFFSET_OFF + U64_BYTES]) as usize;
        let postings_offset =
            read_u64_le(&header[hdr::POSTINGS_OFFSET_OFF..hdr::POSTINGS_OFFSET_OFF + U64_BYTES])
                as usize;
        let doc_lengths_table_offset =
            read_u64_le(&header[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;
        let positions_offset =
            read_u64_le(&header[hdr::POSITIONS_OFFSET_OFF..hdr::POSITIONS_OFFSET_OFF + U64_BYTES])
                as usize;

        // The doc-id map region, `V8` only: one `u32` per document
        // giving the Parquet row that document's postings belong to, then
        // a CRC. It is the last thing before the doc-lengths directory,
        // and its size follows from the document count, so where it
        // starts is arithmetic rather than a field: the directory offset
        // less the map's own length. That keeps every version's header
        // the same width, which is what lets a cold open read one header
        // span and no more.
        let derived_map_len = (n_docs as usize)
            .saturating_mul(U32_BYTES)
            .saturating_add(format::CRC_BYTES);
        // From `V9` the length is written, as the eight bytes immediately
        // before the directory, so a blob with no map says so rather than
        // being a different version. Reading it costs nothing: the lazy open
        // fetches the directory anyway and starts its span at the trailer.
        let trailer = format::fts::doc_map_len_trailer(version);
        let doc_map_len = match trailer {
            0 => match version == format::fts::VERSION_V8 {
                true => derived_map_len,
                false => 0,
            },
            n => {
                if doc_lengths_table_offset < n || doc_lengths_table_offset > source_len {
                    return Err(FtsError::Read(ReadError::Malformed(format!(
                        "doc-lengths directory at {doc_lengths_table_offset} leaves no room \
                         for the doc-id map length that precedes it"
                    ))));
                }
                let at = doc_lengths_table_offset - n;
                let bytes = fetch_source_range(&source, at..at + n, "fts/doc-map len")?;
                let stored = read_u64_le(&bytes);
                // The trailer sits between the map's CRC and the directory's,
                // covered by neither, so it is held to the document count
                // instead: a map is one id per document plus its CRC, or
                // there is no map. Anything else is a corrupt trailer. Taken
                // at its word, a length past the blob would overflow the
                // arithmetic that places the map, and one a few bytes short
                // would open cleanly without CRC checks and shift every row
                // the map resolves by one.
                if stored != 0 && stored != derived_map_len as u64 {
                    return Err(FtsError::Read(ReadError::Malformed(format!(
                        "fts doc-id map length {stored} is neither 0 nor the \
                         {derived_map_len} bytes a map over {n_docs} docs takes"
                    ))));
                }
                stored as usize
            }
        };
        let doc_map_offset: Option<usize> = match doc_map_len {
            0 => None,
            len => Some(doc_lengths_table_offset.saturating_sub(trailer.saturating_add(len))),
        };

        // Bounds-check every offset against the blob length before
        // any slice indexing. A single byte flip in the header can
        // corrupt these into multi-GB values; without this check
        // they propagate as out-of-range slice indices and panic
        // before the CRC verification can reject the corruption.
        //
        // The `< +4` checks (rather than `<= +4`) admit the legal
        // empty-region case: when every term takes the df=1 inline-entry
        // short-circuit, the postings region body is zero bytes and
        // only the trailing 4-byte CRC32C(empty) sits between
        // `postings_offset` and `doc_lengths_table_offset`.
        // Where the regions before the tail stop: the map's start when there
        // is one, otherwise the length trailer that sits in its place.
        let regions_end =
            doc_map_offset.unwrap_or(doc_lengths_table_offset.saturating_sub(trailer));
        let postings_end = positions_offset;
        if dict_offset < HEADER_SIZE
            || postings_offset < dict_offset + 4
            || postings_end < postings_offset + 4
            || regions_end < postings_end
            || doc_lengths_table_offset < regions_end
            || doc_lengths_table_offset > source_len
            || regions_end < positions_offset + 4
            || doc_map_offset.is_some_and(|mo| doc_lengths_table_offset < mo + format::CRC_BYTES)
        {
            return Err(FtsError::Read(ReadError::Malformed(format!(
                "fts header offsets out of range: dict={dict_offset}, postings={postings_offset}, \
                 positions={positions_offset}, doc_lengths={doc_lengths_table_offset}, \
                 blob_len={}",
                source_len
            ))));
        }

        // Region lengths aren't stored explicitly (each region ends
        // with its CRC32C). Compute from the surrounding offsets —
        // postings end where the positions region begins, positions end
        // where the doc-id map or the doc-lengths directory begins.
        let dict_range = dict_offset..postings_offset.saturating_sub(4); // strip CRC
        let postings_range = postings_offset..postings_end.saturating_sub(4); // strip CRC
        let positions_range = positions_offset..regions_end.saturating_sub(4); // strip CRC

        // Verify dictionary CRC32C (4 bytes after the dictionary body).
        if opts.verify_crc {
            let dict_crc_bytes = fetch_source_range(
                &source,
                postings_offset.saturating_sub(4)..postings_offset,
                "fts/dict crc",
            )?;
            let dict_crc_expected = read_u32_le(&dict_crc_bytes);
            let dict_bytes = fetch_source_range(&source, dict_range.clone(), "fts/dict")?;
            let dict_crc_actual = crc32c(&dict_bytes);
            if dict_crc_expected != dict_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/dict",
                    column: String::new(),
                }));
            }
        }

        // Verify postings region CRC32C.
        if opts.verify_crc {
            let postings_crc_pos = postings_end.saturating_sub(4);
            let postings_crc_bytes =
                fetch_source_range(&source, postings_crc_pos..postings_end, "fts/postings crc")?;
            let postings_crc_expected = read_u32_le(&postings_crc_bytes);
            let postings_bytes =
                fetch_source_range(&source, postings_range.clone(), "fts/postings")?;
            let postings_crc_actual = crc32c(&postings_bytes);
            if postings_crc_expected != postings_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/postings",
                    column: String::new(),
                }));
            }
        }

        // Verify positions region CRC32C.
        if opts.verify_crc {
            // The positions region's CRC trails the region, which is the
            // doc-lengths directory only on a blob with no doc-id map;
            // with one, the map sits between them.
            let crc_pos = regions_end.saturating_sub(4);
            let crc_bytes = fetch_source_range(&source, crc_pos..regions_end, "fts/positions crc")?;
            let crc_expected = read_u32_le(&crc_bytes);
            let pos_bytes = fetch_source_range(&source, positions_range.clone(), "fts/positions")?;
            let crc_actual = crc32c(&pos_bytes);
            if crc_expected != crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/positions",
                    column: String::new(),
                }));
            }
        }

        // Parse columns_json.
        let cols: Vec<FtsColumnConfig> = serde_json::from_str(columns_json).map_err(|e| {
            FtsError::Read(ReadError::Malformed(format!("inf.fts.columns JSON: {e}")))
        })?;
        if cols.len() != n_columns {
            return Err(FtsError::Read(ReadError::Malformed(format!(
                "inf.fts.columns has {} entries, header says {}",
                cols.len(),
                n_columns
            ))));
        }

        // Read doc-lengths directory: n_columns × 16-byte entries + 4-byte CRC.
        //
        // On the lazy open path `open_lazy` already fetched this directory
        // and installed it in the overlay, so reading it here costs no GET.
        // On the eager path the whole subsection is in memory, so every
        // read below is a zero-copy slice.
        let dir_size = n_columns * doc_lengths_entry_size(version);
        let dir_end = doc_lengths_table_offset + dir_size;
        if dir_end + 4 > source_len {
            return Err(FtsError::Read(ReadError::Malformed(
                "doc-lengths directory runs past blob end".into(),
            )));
        }
        let dir_region = fetch_source_range(
            &source,
            doc_lengths_table_offset..dir_end + 4,
            "fts/doc_lengths_dir",
        )?;
        let dir_bytes = &dir_region[..dir_size];
        if opts.verify_crc {
            let dir_crc_expected = read_u32_le(&dir_region[dir_size..dir_size + 4]);
            let dir_crc_actual = crc32c(dir_bytes);
            if dir_crc_expected != dir_crc_actual {
                return Err(FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/doc_lengths_dir",
                    column: String::new(),
                }));
            }
        }

        // Build ColumnMeta vec + column_id_by_name.
        let mut columns = Vec::with_capacity(n_columns);
        let mut column_id_by_name = HashMap::with_capacity(n_columns);
        for (i, col_cfg) in cols.iter().enumerate() {
            let entry_off = i * doc_lengths_entry_size(version);
            let column_id = u32::from_le_bytes([
                dir_bytes[entry_off],
                dir_bytes[entry_off + 1],
                dir_bytes[entry_off + 2],
                dir_bytes[entry_off + 3],
            ]);
            let doc_lengths_offset =
                read_u64_le(&dir_bytes[entry_off + 4..entry_off + 12]) as usize;
            let avgdl_x1000 = read_u32_le(
                &dir_bytes[entry_off + format::fts::DOC_LENGTHS_ENTRY_AVGDL_OFF
                    ..entry_off + format::fts::DOC_LENGTHS_ENTRY_AVGDL_OFF + U32_BYTES],
            ) as u64;

            // Verify column_id matches the JSON's positional column_id.
            if column_id != i as u32 {
                return Err(FtsError::Read(ReadError::Malformed(format!(
                    "doc-lengths directory entry {i} has column_id {column_id}"
                ))));
            }

            // Per-column doc-lengths array: `DOC_LENGTH_BYTES` per doc + CRC.
            // Bounded here; read lazily on the column's first scored use,
            // except that CRC verification reads it now to check it, which
            // on the lazy path is one GET per column at open.
            let array_byte_len = format::fts::DOC_LENGTH_BYTES * n_docs as usize;
            let array_end = doc_lengths_offset + array_byte_len;
            if array_end + 4 > source_len {
                return Err(FtsError::Read(ReadError::Malformed(format!(
                    "doc-lengths array {i} runs past blob end"
                ))));
            }
            // What a blob that stores its buckets recorded for this column:
            // the array beside the lengths, and the totals the writer's own
            // pass produced, so the reader repeats none of it.
            let stored_norms = match format::fts::has_stored_buckets(version) {
                false => None,
                true => {
                    let buckets_start = doc_lengths_offset + array_byte_len;
                    let buckets_end = buckets_start
                        .saturating_add(n_docs as usize * format::fts::DOC_LENGTH_BUCKET_BYTES);
                    if buckets_end + format::CRC_BYTES > source_len {
                        return Err(FtsError::Read(ReadError::Malformed(format!(
                            "doc-lengths bucket array {i} runs past blob end"
                        ))));
                    }
                    let stats = ColumnLengthStats {
                        total_tokens: read_u64_le(&dir_bytes[entry_off + 16..entry_off + 24]),
                        n_scored_docs: read_u64_le(&dir_bytes[entry_off + 24..entry_off + 32]),
                    };
                    Some(Box::new(StoredNorms {
                        buckets: buckets_start..buckets_end,
                        stats,
                        occupied: (dir_bytes[entry_off + 32], dir_bytes[entry_off + 33]),
                    }))
                }
            };
            let baked_avgdl = (avgdl_x1000 as f32) / format::fts::AVGDL_FIXED_POINT_SCALE;
            let params = col_cfg.params();
            check_recorded_tokenizer(&col_cfg.name, col_cfg.tokenizer.as_deref())
                .map_err(FtsError::Read)?;
            let (stopwords, stemmer) = col_cfg.filters().map_err(|(field, value)| {
                FtsError::Read(ReadError::Malformed(format!(
                    "inf.fts.columns: unknown {field} {value:?} for column {:?}",
                    col_cfg.name
                )))
            })?;
            let tokenizer = chain_tokenizer(stopwords, stemmer);
            // The length array is not decoded here: the norms scoring
            // needs from it are built on first scored use
            // (`ColumnMeta::norms`), so a match-only query, or one a
            // table-level term index resolves without this dictionary,
            // opens the column for free. With CRC verification on, the
            // array is still checked at open like every other verified
            // region — an in-memory open slices it for free, and a caller
            // who verifies a lazy source has asked for that read. The
            // object-store options turn verification off, and the check
            // then runs when the array is first read.
            let column = ColumnMeta {
                stored_norms,
                name: col_cfg.name.clone(),
                field_id: col_cfg.field_id.map(FieldId),
                doc_lengths_range: doc_lengths_offset..array_end,
                params,
                positions: col_cfg.positions,
                tokenizer,
                stopwords,
                stemmer,
                stored: col_cfg.stored,
                analysis_revision: col_cfg.analysis_revision,
                source: source.clone(),
                n_docs,
                baked_avgdl,
                score_avgdl: baked_avgdl,
                wide_blocks: format::fts::wide_blocks(version),
                declared_params: params,
                verify_crc: opts.verify_crc,
                base_norms: Arc::new(OnceLock::new()),
                view_norms: Arc::new(OnceLock::new()),
            };
            if opts.verify_crc {
                // The span the CRC covers, which on a blob that stores its
                // buckets is the lengths and the buckets together.
                let array =
                    fetch_source_range(&source, column.array_with_crc_range(), LENGTH_ARRAY)?;
                column.check_array_crc(&array)?;
            }
            columns.push(column);
            column_id_by_name.insert(col_cfg.name.clone(), i as u32);
        }

        // From memory the map is a slice of the blob, taken now; a lazy source
        // that does not hold it yet leaves it to the first query that needs
        // it (see the field).
        let doc_map = Arc::new(OnceCell::new());
        let doc_map_region = match doc_map_offset {
            None => {
                let _ = doc_map.set(DocMap::Identity);
                None
            }
            Some(mo) => {
                let body_end = doc_lengths_table_offset
                    .saturating_sub(format::fts::doc_map_len_trailer(version))
                    .saturating_sub(format::CRC_BYTES);
                // The map's length is the document count's, derived from it
                // on an older blob and checked against it where a blob
                // stores it, so the start cannot disagree with the length;
                // what a corrupt header can still do is put the directory
                // too early for a map of this many documents to fit between
                // it and the regions before it.
                if doc_lengths_table_offset < doc_map_len {
                    return Err(FtsError::Read(ReadError::Malformed(format!(
                        "fts doc-map of {doc_map_len} bytes for {n_docs} docs does not fit \
                         below the doc-lengths directory at {doc_lengths_table_offset}"
                    ))));
                }
                let region = mo..body_end;
                let span = doc_map_span(&region, opts.verify_crc);
                if let Some(bytes) = source.try_get_range_sync(span) {
                    let _ = doc_map.set(read_doc_map(&source, &region, bytes, opts.verify_crc)?);
                }
                Some(region)
            }
        };

        Ok(FtsReader {
            block_tables: Arc::default(),
            source,
            n_docs,
            n_terms_total,
            dict_range,
            postings_range,
            positions_range,
            version,
            columns,
            column_id_by_name,
            doc_map,
            doc_map_region,
            verify_crc: opts.verify_crc,
        })
    }

    /// The doc-id map, read on the caller's runtime if no query has read it
    /// yet. Concurrent first callers share one read, and a failed read is not
    /// kept: the error goes to this query and the next one reads again.
    ///
    /// Every query path awaits this before it translates a doc id, so the
    /// read never goes through [`Self::doc_map`]'s blocking one.
    pub(super) async fn ensure_doc_map(&self) -> Result<&DocMap, FtsError> {
        self.doc_map
            .get_or_try_init(|| async {
                let region = self
                    .doc_map_region
                    .as_ref()
                    .expect("invariant: a blob without a map has it set at open");
                let bytes = self
                    .source
                    .range_async(doc_map_span(region, self.verify_crc))
                    .await
                    .map_err(|e| FtsError::RangeFetch {
                        what: "fts/doc-map",
                        source: e,
                    })?;
                read_doc_map(&self.source, region, bytes, self.verify_crc)
            })
            .await
    }

    /// Everything scoring `column` reads besides the postings — the length
    /// buckets and the doc-id map — fetched together, so a cold superfile's
    /// query issues them in one wave with its postings rather than one after
    /// another. Each is a no-op once read.
    pub(crate) async fn prewarm_scoring(&self, column: &str) -> Result<(), FtsError> {
        let column_id = self.resolve_column_id(column)?;
        let (norms, map) = join(self.ensure_norms(column_id), self.ensure_doc_map()).await;
        norms?;
        map.map(|_| ())
    }

    /// The doc-id map, for a translation that runs off the async path.
    ///
    /// A query has read it by then ([`Self::ensure_doc_map`]), and a
    /// compaction input holds its whole file from open, so this reads only
    /// when neither did, through the source's blocking read. A failed read
    /// is an error: taking blob ids as rows would silently answer with the
    /// wrong rows.
    pub(crate) fn doc_map(&self) -> Result<DocMap, FtsError> {
        if let Some(map) = self.doc_map.get() {
            return Ok(map.clone());
        }
        let region = self
            .doc_map_region
            .as_ref()
            .expect("invariant: a blob without a map has it set at open");
        let span = doc_map_span(region, self.verify_crc);
        let bytes = fetch_source_range(&self.source, span, "fts/doc-map")?;
        let map = read_doc_map(&self.source, region, bytes, self.verify_crc)?;
        // A concurrent first read may have set it meanwhile; both are the
        // same map.
        let _ = self.doc_map.set(map.clone());
        Ok(map)
    }

    /// Whether a query has read the doc-id map yet.
    #[cfg(test)]
    pub(crate) fn doc_map_is_read(&self) -> bool {
        self.doc_map.initialized()
    }

    /// Whether this blob stores its documents under an ordering of its
    /// own rather than in row order.
    pub(crate) fn has_doc_map(&self) -> bool {
        self.doc_map_region.is_some()
    }

    pub fn n_docs(&self) -> u32 {
        self.n_docs
    }

    pub fn n_terms(&self) -> u32 {
        self.n_terms_total
    }

    /// FTS column names in declaration order.
    pub fn fts_columns(&self) -> impl Iterator<Item = &str> {
        self.columns.iter().map(|c| c.name.as_str())
    }

    pub fn fts_columns_config(&self) -> impl Iterator<Item = &ColumnMeta> {
        self.columns.iter()
    }

    /// This superfile's document-length totals for `column`, read from its
    /// length array (now, if no query has). `None` if `column` is not an FTS
    /// column here. The commit path records these on the manifest so
    /// table-wide statistics are a fold over the summaries rather than a
    /// fan-out that reopens every superfile, so an array that cannot be read
    /// is an error, not a column with no tokens: a zero recorded there would
    /// skew every query's statistics for as long as the superfile lives.
    pub fn column_length_stats(&self, column: &str) -> Result<Option<ColumnLengthStats>, FtsError> {
        let Ok(id) = self.resolve_column_id(column) else {
            return Ok(None);
        };
        self.columns[id as usize].try_length_stats().map(Some)
    }

    /// The average document length `column` was built at, which its stored
    /// bounds assume; `None` when the index has no such column.
    pub(crate) fn column_baked_avgdl(&self, column: &str) -> Option<f32> {
        let id = self.resolve_column_id(column).ok()?;
        Some(self.columns[id as usize].baked_avgdl)
    }

    /// The parameters `column` declares: what its bounds, and the heads a
    /// term index takes from it, are scored at.
    pub(crate) fn column_params(&self, column: &str) -> Option<bm25::Bm25Params> {
        let id = self.resolve_column_id(column).ok()?;
        Some(self.columns[id as usize].declared_params)
    }

    test_visible! {
    /// A column's stored per-document lengths, one per local doc id: the
    /// array the norm table is quantized from, as the build wrote it
    /// (saturated at the format's stored maximum). A benchmark oracle
    /// compares it against its own tokenization of the rows it believes a
    /// superfile holds, which pins the row-to-superfile mapping exactly
    /// rather than assuming it from ingest order. Errors if `column` is not
    /// a registered FTS column.
    fn column_doc_lengths(&self, column: &str) -> Result<Vec<u32>, FtsError> {
        let id = self.resolve_column_id(column)?;
        self.read_doc_lengths(id)
    }
    }

    /// Tokenizer configured for `column`, for tokenizing query text so
    /// it matches how the column was indexed. Errors if `column` is not
    /// a registered FTS column.
    pub fn column_tokenizer(&self, column: &str) -> Result<Arc<dyn Tokenizer>, FtsError> {
        let id = self.resolve_column_id(column)?;
        Ok(Arc::clone(&self.columns[id as usize].tokenizer))
    }

    pub(crate) fn dict_bytes(&self) -> Result<Bytes, FtsError> {
        fetch_source_range(&self.source, self.dict_range.clone(), "fts/dict")
    }

    /// Most postings the term at `value` holds: exact for a long term,
    /// read from its header, and the form's limit for a short one.
    pub(crate) fn term_postings_at_most(&self, value: DictEntry) -> Result<u32, FtsError> {
        match value {
            DictEntry::Inline { .. } => Ok(1),
            DictEntry::Pfor { short: true, .. } => Ok(SHORT_MAX_DF as u32),
            DictEntry::Pfor {
                metadata_offset, ..
            } => {
                let start =
                    self.postings_range.start + metadata_offset as usize + term_meta::DF_OFF;
                let df =
                    fetch_source_range(&self.source, start..start + U32_BYTES, "fts/merge df")?;
                Ok(read_u32_le(&df))
            }
        }
    }

    /// Open the term dictionary over fetched bytes, mapping a parse
    /// failure to the reader's malformed-blob error.
    pub(super) fn open_dict(dict_bytes: &[u8]) -> Result<TermBlocks<'_>, FtsError> {
        TermBlocks::open(dict_bytes).map_err(|e| {
            FtsError::Read(ReadError::Malformed(format!(
                "term dictionary parse failed: {e}"
            )))
        })
    }

    /// Build `column_id`'s norms from its length array if a scoring path has
    /// not already, fetching the array on the caller's runtime (no sync
    /// bridge). Every scoring entry point calls this before building
    /// cursors; the synchronous fallback inside [`ColumnMeta::norms`] then
    /// never has to read on a lazy source. A match-only query skips it, and
    /// never touches the array at all.
    pub(super) async fn ensure_norms(&self, column_id: u32) -> Result<(), FtsError> {
        let col = &self.columns[column_id as usize];
        if col.norms_loaded() {
            return Ok(());
        }
        // The array plus its CRC, checked when verification is on. The
        // range is the open path's, bounded against the blob there; a source
        // that answers short errors (`ShortRead`) rather than handing up a
        // truncated buffer, so the slice below is within what came back.
        let array = self
            .source
            .range_async(col.norms_fetch_range())
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: LENGTH_ARRAY,
                source: e,
            })?;
        col.install_fetched_norms(&array)?;
        Ok(())
    }

    /// Async term-dictionary fetch for the query path. Resolves
    /// zero-copy for in-memory / warm sources; for a cold `Lazy`
    /// source it `await`s the object-store range on the caller's
    /// runtime (no sync bridge).
    pub(crate) async fn dict_bytes_async(&self) -> Result<Bytes, FtsError> {
        self.source
            .range_async(self.dict_range.clone())
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: "fts/dict",
                source: e,
            })
    }

    /// Fetch the complete byte range of each requested term — metadata
    /// header + skip table + encoded posting blocks, or a short-form
    /// body — in parallel, coalescing adjacent ranges. `terms` are
    /// `(metadata_offset, postings_length)` pairs from the dictionary
    /// (`DictEntry::Pfor`); the
    /// returned `Bytes` for term `i` starts at that term's metadata
    /// header (offset 0) and runs to the end of its last block, so a
    /// `TermCursor` can index it directly.
    ///
    /// This is the FTS analog of the vector reader's per-probed-cluster
    /// `Source::get_ranges_parallel` fan-out: a query only ever pulls
    /// the bytes of the terms it actually scores, never the whole
    /// postings region. On an in-memory source every range resolves as
    /// a zero-copy slice; on a lazy (object-store) source the cold
    /// ranges are coalesced under one async bridge and returned in
    /// input order.
    ///
    /// The metadata header remains in the returned bytes for validation
    /// and cursor construction.
    pub(super) async fn fetch_term_postings(
        &self,
        terms: &[(usize, usize)],
    ) -> Result<Vec<Bytes>, FtsError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let base = self.postings_range.start;
        let region_len = self.postings_range.len();

        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(terms.len());
        for &(m, postings_length) in terms {
            if postings_length == 0 || m + postings_length > region_len {
                return Err(FtsError::Read(ReadError::Malformed(
                    "term postings range runs past postings region".into(),
                )));
            }
            ranges.push(base + m..base + m + postings_length);
        }
        let plan = RangeCoalescePlan::new(
            &ranges,
            TERM_RANGE_COALESCE_MAX_GAP,
            TERM_RANGE_COALESCE_MAX_OVERFETCH,
        );
        let fetched = self
            .source
            .get_ranges_parallel_async(plan.fetch_ranges())
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: "fts/postings term body",
                source: e,
            })?;
        Ok(plan.restore(&fetched))
    }

    /// Fetch each requested term's position-run bytes from the
    /// positions region — the phrase sibling of
    /// [`fetch_term_postings`](Self::fetch_term_postings): one range
    /// per term, fanned out in parallel, never the whole region.
    /// `terms` pairs are `(positions_offset, positions_length)` from
    /// the terms' metadata; zero-length entries (inline terms) yield
    /// empty buffers without touching the source.
    async fn fetch_term_positions(&self, terms: &[(u64, u32)]) -> Result<Vec<Bytes>, FtsError> {
        if terms.iter().all(|&(_, len)| len == 0) {
            return Ok(vec![Bytes::new(); terms.len()]);
        }
        let region = &self.positions_range;
        let base = region.start;
        let region_len = region.len();
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(terms.len());
        for &(off, len) in terms {
            let off = off as usize;
            let len = len as usize;
            if off + len > region_len {
                return Err(FtsError::Read(ReadError::Malformed(
                    "term positions range runs past positions region".into(),
                )));
            }
            ranges.push(base + off..base + off + len);
        }
        self.source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(|e| FtsError::RangeFetch {
                what: "fts/positions term",
                source: e,
            })
    }

    /// Build one [`AnyCursor`] per requested atom, preserving input
    /// order: first the `terms`, then the `phrases`. An atom whose
    /// term (or any phrase member) is absent from the column yields
    /// `None` — the caller applies polarity semantics (a missing must
    /// empties the result; a missing should or negative is dropped).
    ///
    /// Multi-token phrases require the column to be positional;
    /// otherwise [`FtsError::PositionsUnavailable`].
    /// The second element counts the term-dictionary ranges the builds
    /// requested (one per `build_term_cursors` call plus one per inline
    /// phrase member's position recovery) — real byte-source ranges on
    /// every query, tallied by the caller into the planned count.
    pub(super) async fn build_atom_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
        phrases: &[Phrase<String>],
        idf: QueryIdf<'_>,
        prefetched: Option<&FetchedTermMemo>,
        purpose: CursorUse,
    ) -> Result<(Vec<Option<AnyCursor>>, u64), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        // A scored walk needs the norms before any cursor is built; an
        // unranked match does not, and must not read the length array.
        if purpose.scores() {
            self.ensure_norms(column_id).await?;
        }
        if !phrases.is_empty() && !col_meta.positions {
            return Err(FtsError::PositionsUnavailable {
                column: col_meta.name.clone(),
            });
        }
        let mut dict_ranges = 0u64;
        let mut out: Vec<Option<AnyCursor>> = Vec::with_capacity(terms.len() + phrases.len());
        // All bare terms in one dictionary open + one parallel postings fan-out
        // (arity-preserving: each term maps to its own slot, `None` when
        // absent), rather than a serial per-term build that re-fetched the
        // dictionary and issued a separate range wave for each. One planned
        // dictionary range for the whole batch.
        if !terms.is_empty() {
            let term_cursors = self
                .build_term_cursors_opt(column_id, terms, idf, purpose, None, prefetched, None)
                .await?;
            dict_ranges += 1;
            for cursor in term_cursors {
                out.push(cursor.map(AnyCursor::Term));
            }
        }
        for phrase in phrases {
            let (member_refs, member_at) = distinct_members(phrase);
            // A phrase's score is Σ member idf (see `PhraseCursor::new`), so
            // globalizing the members' idf globalizes the phrase — the
            // per-member rescale ratio cancels out of the phrase's tf/length
            // bound. Build members with the same `idf` as bare terms.
            let cursors = self
                .build_term_cursors(column_id, &member_refs, idf, purpose, None, prefetched)
                .await?;
            dict_ranges += 1;
            if cursors.len() != member_refs.len() {
                // A member is absent — the phrase can never match.
                out.push(None);
                continue;
            }
            // Positional extras per member, kept off the term cursors
            // (whose footprint the term-only kernels depend on): PFOR
            // members re-parse their metadata header from their own
            // bytes; an inline (df=1) member recovers its single
            // position from the dictionary entry the tf-reinterpretation
            // dropped during cursor build.
            let mut positional: Vec<(Option<TermMeta>, Option<u32>)> =
                Vec::with_capacity(cursors.len());
            // Short-form members' inline groups, by member index.
            let mut inline_groups: Vec<(usize, Bytes)> = Vec::new();
            for (cursor, term) in cursors.iter().zip(&member_refs) {
                match (cursor.predecoded, cursor.bytes.is_empty()) {
                    (false, _) => {
                        // This is the phrase member's own term_meta — the
                        // one `decode_current_positions` uses.
                        let term_meta = TermMeta::parse(cursor.bytes.as_ref(), 0, true)?;
                        positional.push((Some(term_meta), None));
                    }
                    (true, false) => {
                        // Short-form member: its position group is inline
                        // in the body, after the postings. Re-decode for
                        // where it starts; the slice becomes the member's
                        // positions bytes.
                        let mut d = [0u32; BLOCK_LEN];
                        let mut t = [0u32; BLOCK_LEN];
                        let decoded = decode_short(cursor.bytes.as_ref(), true, &mut d, &mut t)
                            .ok_or_else(|| {
                                FtsError::Read(ReadError::Malformed(
                                    "short-form term body does not decode".into(),
                                ))
                            })?;
                        inline_groups.push((
                            positional.len(),
                            cursor
                                .bytes
                                .slice(decoded.positions_at.expect("positional short body")..),
                        ));
                        positional.push((Some(TermMeta::for_short(decoded.n as u64)), None));
                    }
                    (true, true) => {
                        dict_ranges += 1;
                        let dict_bytes = self.dict_bytes_async().await?;
                        let dict = Self::open_dict(&dict_bytes)?;
                        let key = make_key(&col_meta.name, term);
                        let packed = dict
                            .lookup(&key)
                            .expect("inline member cursor was built from this dict");
                        let position = match packed {
                            DictEntry::Inline { tf: slot, .. } => slot,
                            DictEntry::Pfor { .. } => {
                                unreachable!("inline cursor from a PFOR dictionary entry")
                            }
                        };
                        positional.push((None, Some(position)));
                    }
                }
            }
            let pos_ranges: Vec<(u64, u32)> = positional
                .iter()
                .map(|(term_meta, _)| {
                    term_meta
                        .map(|tm| (tm.positions_offset, tm.positions_length))
                        .unwrap_or((0, 0))
                })
                .collect();
            let mut positions = self.fetch_term_positions(&pos_ranges).await?;
            for (member, group) in inline_groups {
                positions[member] = group;
            }
            out.push(Some(AnyCursor::Phrase(PhraseCursor::new(
                cursors,
                positions,
                positional,
                member_at,
                phrase.offsets().to_vec(),
            )?)));
        }
        Ok((out, dict_ranges))
    }

    /// Walk the term dictionary and collect every term registered under
    /// `column`, in lex order. Used to populate per-superfile FTS
    /// skip-pruning summaries (term-presence bloom + lex term
    /// range) at commit time.
    ///
    /// Returns an empty `Vec` if `column` is not registered as
    /// an FTS column in this superfile. Cost is O(terms in column)
    /// dictionary decodes; intended to be called once per (superfile,
    /// column) at commit time, not on the query hot path.
    pub fn iter_column_terms(&self, column: &str) -> Result<Vec<Vec<u8>>, FtsError> {
        self.iter_terms_with_prefix(column, b"")
    }

    /// Catches a caller handing in a dictionary of the wrong size, such as
    /// another reader's. A same-sized one is not caught.
    pub(super) fn debug_assert_own_dict(&self, dict_bytes: &[u8]) {
        debug_assert_eq!(
            dict_bytes.len(),
            self.dict_range.len(),
            "dictionary bytes are not the size of this reader's"
        );
    }

    /// Stream a column's postings for the FTS compaction merge: for every term
    /// (lex order) and every doc in its posting list (doc_ids ascending),
    /// invoke `emit(term_bytes, local_doc_id, tf, positions)`. Reuses the
    /// query-path [`TermCursor`] block decode for doc_ids/tfs and the
    /// block's position group for positions, so what is streamed is exactly
    /// what a fresh build produced. `positions` is empty for a non-positional
    /// column; otherwise it holds the `tf` token offsets for this `(term, doc)`
    /// (borrowed from a reused buffer — copy it if you need to retain it past
    /// the call). Tombstone filtering is the caller's job — this streams every
    /// stored posting.
    ///
    /// Synchronous: compaction opens its inputs over resident bytes, so every
    /// range resolves without a runtime. `emit` may return an error to abort.
    pub(crate) fn for_each_term_posting(
        &self,
        column_id: u32,
        emit: impl FnMut(&[u8], u32, u32, &[u32]) -> Result<(), FtsError>,
    ) -> Result<(), FtsError> {
        let column_name = &self.columns[column_id as usize].name;
        let dict_bytes = self.dict_bytes()?;
        let dict = Self::open_dict(&dict_bytes)?;

        // Column-scoped dictionary keys are `column_name <KEY_SEPARATOR> term`;
        // `iter_prefix` yields `(key, packed_value)` in lex term order, so we
        // read the posting metadata straight from the value — no re-lookup.
        let column_prefix = make_key(column_name, "");
        let prefix_len = column_prefix.len();

        let entries = dict.iter_prefix(&column_prefix);
        self.for_each_posting_in(
            column_id,
            entries
                .iter()
                .map(|(key, packed)| (&key[prefix_len..], *packed)),
            &mut Vec::new(),
            emit,
        )
    }

    /// Up to `limit` of a column's terms that are `>= from`, in lex order,
    /// each with its dictionary value. Lets a merge walk many inputs'
    /// vocabularies side by side without loading any of them whole.
    /// `dict_bytes` is this reader's [`Self::dict_bytes`], fetched once by
    /// the caller: on a lazy source each fetch is a full-dictionary read.
    pub(crate) fn column_terms_from(
        &self,
        dict_bytes: &[u8],
        column_id: u32,
        from: &[u8],
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, DictEntry)>, FtsError> {
        let dict = Self::open_dict(dict_bytes)?;
        let prefix = make_key(&self.columns[column_id as usize].name, "");
        let mut start = prefix.clone();
        start.extend_from_slice(from);
        let mut out = Vec::with_capacity(limit);
        dict.for_each_from(&prefix, &start, |key, value| {
            out.push((key[prefix.len()..].to_vec(), value));
            out.len() < limit
        });
        Ok(out)
    }

    /// [`Self::for_each_term_posting`] over the given `(term, dictionary
    /// value)` entries instead of the whole column. `positions_buf` is
    /// scratch for the decoded position runs, owned by the caller so it
    /// can be reused across calls.
    pub(crate) fn for_each_posting_in<'t>(
        &self,
        column_id: u32,
        entries: impl Iterator<Item = (&'t [u8], DictEntry)>,
        positions_buf: &mut Vec<u32>,
        mut emit: impl FnMut(&[u8], u32, u32, &[u32]) -> Result<(), FtsError>,
    ) -> Result<(), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let positional = col_meta.positions;
        let positions_region = &self.positions_range;

        for (term, packed) in entries {
            match packed {
                DictEntry::Inline { doc_id, tf } => {
                    // A positional column only inlines tf == 1 postings; the
                    // slot then carries the term's single position and tf is
                    // implied 1. Non-positional: `tf` is the frequency, no
                    // positions.
                    if positional {
                        emit(term, doc_id, 1, &[tf])?;
                    } else {
                        emit(term, doc_id, tf, &[])?;
                    }
                }
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => {
                    let term_bytes = self.term_bytes_sync(metadata_offset, postings_length)?;

                    if short {
                        // Short-form term: the whole list is the body; its
                        // positions sit in the region at the trailer's range.
                        let mut d = [0u32; BLOCK_LEN];
                        let mut t = [0u32; BLOCK_LEN];
                        let decoded = decode_short(term_bytes.as_ref(), positional, &mut d, &mut t)
                            .ok_or_else(|| {
                                FtsError::Read(ReadError::Malformed(
                                    "short-form term body does not decode".into(),
                                ))
                            })?;
                        // A short body's positions are one group, inline
                        // after its postings.
                        let position_bytes = decoded.positions_at.map(|at| term_bytes.slice(at..));
                        let mut group = GroupIndex::default();
                        if let Some(bytes) = &position_bytes {
                            group
                                .locate(bytes.as_ref(), &mut 0, &t[..decoded.n])
                                .ok_or_else(|| {
                                    FtsError::Read(ReadError::Malformed(
                                        "position group in merge read does not decode".into(),
                                    ))
                                })?;
                        }
                        for i in 0..decoded.n {
                            let positions: &[u32] = match &position_bytes {
                                Some(bytes) => {
                                    positions_buf.clear();
                                    group
                                        .run_positions(bytes.as_ref(), i, t[i], positions_buf)
                                        .ok_or_else(|| {
                                            FtsError::Read(ReadError::Malformed(
                                                "position run overflowing in merge read".into(),
                                            ))
                                        })?;
                                    positions_buf.as_slice()
                                }
                                None => &[],
                            };
                            emit(term, d[i], t[i], positions)?;
                        }
                        continue;
                    }

                    // For a positional column, this term's position groups live
                    // contiguously in the positions region at `positions_offset`,
                    // one group per block in posting order. Read the slice
                    // once and walk it in lockstep with the doc cursor.
                    let position_bytes = if positional {
                        let meta = TermMeta::parse(term_bytes.as_ref(), 0, true)?;
                        let pstart = positions_region.start + meta.positions_offset as usize;
                        let pend = pstart + meta.positions_length as usize;
                        Some(fetch_source_range(
                            &self.source,
                            pstart..pend,
                            "fts/merge positions",
                        )?)
                    } else {
                        None
                    };
                    let mut pos_at = 0usize;

                    // This walk carries postings across into a merge; it
                    // reads doc ids, tfs and positions and never consults
                    // a score bound.
                    let mut cursor = TermCursor::new(
                        term_bytes,
                        col_meta,
                        TermIdf::Absent,
                        1,
                        CursorUse::Match,
                        None,
                    )?;
                    // Each block's runs are one group, decoded whole at the
                    // block's start and sliced per pair.
                    let mut group = GroupIndex::default();
                    while !cursor.is_exhausted() {
                        if let Some(bytes) = &position_bytes {
                            let tfs = &cursor.block_tfs[..cursor.block_n];
                            group
                                .locate(bytes.as_ref(), &mut pos_at, tfs)
                                .ok_or_else(|| {
                                    FtsError::Read(ReadError::Malformed(
                                        "position group in merge read does not decode".into(),
                                    ))
                                })?;
                        }
                        while cursor.pos < cursor.block_n {
                            let doc_id = cursor.block_doc_ids[cursor.pos];
                            let tf = cursor.block_tfs[cursor.pos];
                            let positions: &[u32] = match &position_bytes {
                                Some(bytes) => {
                                    positions_buf.clear();
                                    group
                                        .run_positions(
                                            bytes.as_ref(),
                                            cursor.pos,
                                            tf,
                                            positions_buf,
                                        )
                                        .ok_or_else(|| {
                                            FtsError::Read(ReadError::Malformed(
                                                "position run overflowing in merge read".into(),
                                            ))
                                        })?;
                                    positions_buf.as_slice()
                                }
                                None => &[],
                            };
                            emit(term, doc_id, tf, positions)?;
                            cursor.pos += 1;
                        }
                        cursor.next();
                    }
                }
            }
        }
        Ok(())
    }

    /// Walk a column's postings for the merge's document-order pass, which
    /// needs only which documents carry each term: doc ids are decoded, tfs
    /// and positions are not read. For every term, in lex order,
    /// `on_term(term)` returns what to carry for it, or `None` to skip the
    /// term without reading its postings; `on_doc(&carry, local_doc_id)`
    /// then runs for each of its docs, ascending.
    ///
    /// Synchronous, like [`Self::for_each_term_posting`]: compaction opens
    /// its inputs over resident bytes.
    pub(crate) fn for_each_term_doc<K>(
        &self,
        column_id: u32,
        mut on_term: impl FnMut(&[u8]) -> Option<K>,
        mut on_doc: impl FnMut(&K, u32),
    ) -> Result<(), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dict_bytes = self.dict_bytes()?;
        let dict = Self::open_dict(&dict_bytes)?;
        let prefix = make_key(&col_meta.name, "");
        let mut doc_ids = [0u32; BLOCK_LEN];
        let mut tfs = [0u32; BLOCK_LEN];
        let mut failed = None;
        dict.for_each_prefix(&prefix, |key, packed| {
            let Some(carry) = on_term(&key[prefix.len()..]) else {
                return true;
            };
            let docs = match packed {
                DictEntry::Inline { doc_id, .. } => {
                    on_doc(&carry, doc_id);
                    Ok(())
                }
                DictEntry::Pfor {
                    metadata_offset,
                    postings_length,
                    short,
                } => self
                    .term_bytes_sync(metadata_offset, postings_length)
                    .and_then(|term_bytes| match short {
                        true => {
                            let decoded = decode_short(
                                term_bytes.as_ref(),
                                col_meta.positions,
                                &mut doc_ids,
                                &mut tfs,
                            )
                            .ok_or_else(|| {
                                FtsError::Read(ReadError::Malformed(
                                    "short-form term body does not decode".into(),
                                ))
                            })?;
                            for &doc in &doc_ids[..decoded.n] {
                                on_doc(&carry, doc);
                            }
                            Ok(())
                        }
                        // A `Count` cursor decodes a block's doc ids and
                        // skips its tfs.
                        false => {
                            let mut cursor = TermCursor::new(
                                term_bytes,
                                col_meta,
                                TermIdf::Absent,
                                1,
                                CursorUse::Count,
                                None,
                            )?;
                            while !cursor.is_exhausted() {
                                for &doc in &cursor.block_doc_ids[cursor.pos..cursor.block_n] {
                                    on_doc(&carry, doc);
                                }
                                cursor.pos = cursor.block_n;
                                cursor.advance_block();
                            }
                            Ok(())
                        }
                    }),
            };
            match docs {
                Ok(()) => true,
                Err(e) => {
                    failed = Some(e);
                    false
                }
            }
        });
        failed.map_or(Ok(()), Err)
    }

    /// One PFOR term's bytes, from its metadata header to its end.
    fn term_bytes_sync(
        &self,
        metadata_offset: u64,
        postings_length: u32,
    ) -> Result<Bytes, FtsError> {
        let start = self.postings_range.start + metadata_offset as usize;
        fetch_source_range(
            &self.source,
            start..start + postings_length as usize,
            "fts/merge postings",
        )
    }

    /// Read a column's stored per-doc lengths (token counts), one per
    /// local doc-id in `0..n_docs`. The FTS compaction merge carries these
    /// forward (with the input's doc-id remap) rather than recomputing them
    /// from text. These are the already-clamped values written at build time.
    pub(crate) fn read_doc_lengths(&self, column_id: u32) -> Result<Vec<u32>, FtsError> {
        let n = self.n_docs as usize;
        let range = self.columns[column_id as usize].doc_lengths_range.clone();
        let bytes = fetch_source_range(&self.source, range, "fts/merge doc_lengths")?;
        let region = bytes.as_ref();
        if region.len() < n * format::fts::DOC_LENGTH_BYTES {
            return Err(FtsError::Read(ReadError::Malformed(
                "doc-lengths region shorter than n_docs entries".into(),
            )));
        }
        Ok((0..n).map(|d| read_doc_length(region, d)).collect())
    }

    /// Walk the term dictionary and collect every term registered under
    /// `column` whose bytes begin with `term_prefix`, in lex order.
    ///
    /// Mirrors [`Self::iter_column_terms`] but bounds the walk to a
    /// prefix range instead of the whole column. This is the synchronous,
    /// commit-time form (the writer's term summary reaches it through
    /// `iter_column_terms`); the query path's prefix search expands through
    /// [`Self::terms_with_prefix`], which walks the same dictionary on the
    /// reader pool.
    ///
    /// `term_prefix` is the prefix as it appears in the term dictionary — the
    /// caller is responsible for any tokenizer-level normalization
    /// (e.g. lowercasing). Returns an
    /// empty `Vec` if `column` is not registered or no terms match
    /// the prefix.
    pub fn iter_terms_with_prefix(
        &self,
        column: &str,
        term_prefix: &[u8],
    ) -> Result<Vec<Vec<u8>>, FtsError> {
        if !self.has_column(column) {
            return Ok(Vec::new());
        }
        let dict_bytes = self.dict_bytes()?;
        collect_terms_with_prefix(&dict_bytes, column, term_prefix)
    }

    /// Whether `column` is registered as an FTS column in this superfile.
    pub(super) fn has_column(&self, column: &str) -> bool {
        self.column_id_by_name.contains_key(column)
    }
}

/// Every key of `column`'s dictionary that begins with `term_prefix`, in
/// lex order, with the column key prefix stripped. The CPU half shared by
/// the sync commit-time walk ([`FtsReader::iter_terms_with_prefix`]) and
/// the query path's pooled one (`FtsReader::terms_with_prefix`).
pub(super) fn collect_terms_with_prefix(
    dict_bytes: &[u8],
    column: &str,
    term_prefix: &[u8],
) -> Result<Vec<Vec<u8>>, FtsError> {
    let mut full_prefix = column.as_bytes().to_vec();
    full_prefix.push(KEY_SEPARATOR);
    let column_prefix_len = full_prefix.len();
    full_prefix.extend_from_slice(term_prefix);
    let dict = FtsReader::open_dict(dict_bytes)?;
    let pairs = dict.iter_prefix(&full_prefix);
    Ok(pairs
        .into_iter()
        .map(|(key, _)| key[column_prefix_len..].to_vec())
        .collect())
}

/// One query's built OR cursors for one superfile: the postings fetch
/// and skip-table parse done once, cheaply cloneable per doc-id
/// sub-range. Produced by [`FtsReader::build_or_cursor_set`], consumed
/// by [`FtsReader::search_or_range_prebuilt`].
pub(crate) struct OrCursorSet {
    pub(super) column_id: u32,
    pub(super) cursors: Vec<TermCursor>,
}

impl OrCursorSet {
    /// Posting-list bytes this set's cursors index into — see
    /// [`PreparedClauses::postings_bytes`]. Counted once per superfile
    /// even when ranged slices share the set.
    pub(crate) fn postings_bytes(&self) -> u64 {
        term_cursor_bytes(&self.cursors)
    }

    /// Byte-source ranges the set's build requested (one per PFOR term).
    pub(crate) fn planned_ranges(&self) -> u64 {
        term_cursor_ranges(&self.cursors)
    }
}

/// Merge a `row -> score` map into top-k by descending score, ties
/// broken by ascending row. Used by `search_multi`'s cross-column
/// combiner, where the per-column scores have already been weighted
/// and summed into `scores` — and where each column's hits have
/// already been translated out of the blob's id space, which is why
/// this one heap is keyed by rows rather than blob ids.
pub(super) fn top_k(scores: FxHashMap<RowId, f32>, k: usize) -> Vec<(RowId, f32)> {
    // Iterate in ascending doc_id order so ties resolve deterministically
    // (smaller doc_ids enter the heap first; the strict `score > peek`
    // check below means subsequent equal-score entries don't displace
    // them). Without this, HashMap's hash-order iteration would make the
    // tied result non-deterministic and would disagree with the BMW
    // single-term path (which naturally iterates in doc_id order).
    // pdqsort: doc_ids are unique by construction (HashMap keys).
    let mut sorted: Vec<(RowId, f32)> = scores.into_iter().collect();
    sorted.sort_unstable_by_key(|(d, _)| *d);

    let mut heap: BinaryHeap<TopKEntry<RowId>> =
        BinaryHeap::with_capacity(k.min(sorted.len()).max(1));
    for (row, score) in sorted {
        if heap.len() < k {
            heap.push(TopKEntry(score, row));
        } else if let Some(TopKEntry(top_score, _)) = heap.peek()
            && score > *top_score
        {
            heap.pop();
            heap.push(TopKEntry(score, row));
        }
    }
    drain_top_k_desc(heap)
}

/// The bytes to read for the doc-id map at `region`: its body, and the CRC
/// stored right after it when the reader verifies.
fn doc_map_span(region: &Range<usize>, verify_crc: bool) -> Range<usize> {
    match verify_crc {
        true => region.start..region.end + format::CRC_BYTES,
        false => region.clone(),
    }
}

/// The doc-id map from `bytes`, read at [`doc_map_span`]: checked against its
/// CRC when `verify_crc`, and owned as the reader keeps it.
///
/// From memory the body is a slice of the blob, so the map costs no heap and
/// its pages stay evictable. A lazy source may answer a read this large with
/// a window onto a bigger fetch, and keeping the window would keep that whole
/// fetch alive; the map is the one read the reader holds on to, so over such
/// a source it takes a copy of the body alone.
fn read_doc_map(
    source: &Source,
    region: &Range<usize>,
    bytes: Bytes,
    verify_crc: bool,
) -> Result<DocMap, FtsError> {
    if bytes.len() < region.len() {
        return Err(FtsError::Read(ReadError::Malformed(format!(
            "fts doc-map read {} of its {} bytes",
            bytes.len(),
            region.len()
        ))));
    }
    let body = bytes.slice(..region.len());
    if verify_crc {
        let crc = &bytes[region.len()..];
        if crc.len() < format::CRC_BYTES || read_u32_le(crc) != crc32c(&body) {
            return Err(FtsError::Read(ReadError::ChecksumMismatch {
                section: "fts/doc-map",
                column: String::new(),
            }));
        }
    }
    let body = match source {
        Source::InMemory(_) => body,
        Source::Lazy(_) => Bytes::copy_from_slice(&body),
    };
    Ok(DocMap::permuted(body))
}

pub(super) fn fetch_source_range(
    source: &Source,
    range: Range<usize>,
    what: &'static str,
) -> Result<Bytes, FtsError> {
    source
        .get_range(range)
        .map_err(|e| FtsError::RangeFetch { what, source: e })
}

async fn fetch_lazy_range(
    source: &dyn LazyByteSource,
    range: Range<usize>,
    what: &'static str,
) -> Result<Bytes, FtsError> {
    source
        .range(range.start as u64, range.len() as u64)
        .await
        .map_err(|e| FtsError::RangeFetch { what, source: e })
}

#[inline]
pub(super) fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
pub(super) fn read_u64_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(buf)
}

/// Unranked multi-term OR walk: the union of the cursors' doc ids below
/// `end`, in ascending order. A k-way merge — each step finds the minimum
/// current doc id across the live cursors, hands it to `emit`, and
/// advances every cursor sitting on it (so the next minimum is strictly
/// greater and `emit` is called exactly once per distinct doc). No
/// scoring; the caller wants membership, not rank.
pub(super) fn or_walk_unranked(mut cursors: Vec<TermCursor>, end: u32, mut emit: impl FnMut(u32)) {
    loop {
        let min_doc = cursors
            .iter()
            .filter(|c| !c.is_exhausted())
            .map(TermCursor::current_doc_id)
            .min();
        let Some(min_doc) = min_doc.filter(|&doc| doc < end) else {
            break;
        };
        emit(min_doc);
        for c in cursors.iter_mut() {
            if !c.is_exhausted() && c.current_doc_id() == min_doc {
                c.next();
            }
        }
    }
}

/// [`or_walk_unranked`] through a min-heap of (current doc, cursor): each
/// step pops the smallest current doc, emits it once however many cursors
/// sit on it (equal docs pop consecutively), and advances only the
/// cursor popped. Cost is the union size times the log of the term
/// count, independent of how many terms there are per step.
fn or_walk_heap(mut cursors: Vec<TermCursor>, end: u32, mut emit: impl FnMut(u32)) {
    let mut heap = doc_heap(&cursors, |c| {
        (!c.is_exhausted()).then(|| c.current_doc_id())
    });
    let mut last: Option<u32> = None;
    while let Some(Reverse((doc, i))) = heap.pop() {
        if doc >= end {
            break;
        }
        if last != Some(doc) {
            emit(doc);
            last = Some(doc);
        }
        let c = &mut cursors[i];
        c.next();
        if !c.is_exhausted() {
            heap.push(Reverse((c.current_doc_id(), i)));
        }
    }
}

/// The frontier a heap union walk pops from: a min-heap of `(current doc,
/// index)` over the `walkers` that `current` finds a doc for (the live
/// ones). Shared by the term walk and the phrase-aware atom walk.
pub(super) fn doc_heap<T>(
    walkers: &[T],
    current: impl Fn(&T) -> Option<u32>,
) -> BinaryHeap<Reverse<(u32, usize)>> {
    walkers
        .iter()
        .enumerate()
        .filter_map(|(i, walker)| current(walker).map(|doc| Reverse((doc, i))))
        .collect()
}

/// Whether walking a union of `terms` walkers holding `total_df` postings
/// through a heap takes fewer steps than rescanning every live walker
/// `rescans` times: the heap pays a log-of-term-count step per posting,
/// a rescan the term count. Below [`OR_HEAP_MIN_TERMS`] the rescan is kept
/// outright. Over raw numbers so each walk's boundary is unit-testable
/// without building cursors.
fn heap_beats_rescan(terms: usize, total_df: u64, rescans: u64) -> bool {
    if terms < OR_HEAP_MIN_TERMS {
        return false;
    }
    let terms = terms as u64;
    let heap_steps = total_df.saturating_mul(u64::from(terms.ilog2()) + 1);
    heap_steps < terms.saturating_mul(rescans)
}

/// Whether the per-doc union walks ([`or_merge_unranked`], the atom walk in
/// `Or` mode) should take the heap over `terms` walkers holding `total_df`
/// postings in `doc_space` docs. The rescan runs once per doc of the
/// union, which holds at most every doc and at most every posting. So a
/// sparse union over many terms (an `IN` list of ids) takes the heap, and
/// a dense one, whose average doc holds more than about n / log n of the
/// terms (eight or ten common words in a natural-language query), keeps
/// the rescan, which costs it less than the heap's per-posting steps.
pub(super) fn or_walk_prefers_heap(terms: usize, total_df: u64, doc_space: u64) -> bool {
    heap_beats_rescan(terms, total_df, doc_space.min(total_df))
}

/// The union's doc ids in ascending order, collected into a `Vec`: see
/// [`or_merge_unranked_in`], over the whole doc-id space.
pub(super) fn or_merge_unranked(cursors: Vec<TermCursor>) -> Vec<u32> {
    or_merge_unranked_in(cursors, 0, u32::MAX)
}

/// The union's doc ids in `[start, end)`, ascending: each cursor first
/// skips to `start`, and the walk stops at the first doc at or past
/// `end`. One term's ids are its postings, copied a decoded block at a
/// time ([`single_term_docs`]); several go through the rescan walk
/// ([`or_walk_unranked`]), unless the union is sparse enough over many
/// terms that the heap walk takes fewer steps ([`or_walk_prefers_heap`],
/// [`or_merge_heap`]).
pub(super) fn or_merge_unranked_in(mut cursors: Vec<TermCursor>, start: u32, end: u32) -> Vec<u32> {
    if start > 0 {
        for cursor in &mut cursors {
            cursor.skip_to(start);
        }
    }
    if let [cursor] = cursors.as_mut_slice() {
        return single_term_docs(cursor, start, end);
    }
    let doc_space = u64::from(cursors_max_doc(&cursors)) + 1;
    if or_walk_prefers_heap(cursors.len(), df_total(&cursors), doc_space) {
        return or_merge_heap(cursors, end);
    }
    let mut out = Vec::new();
    or_walk_unranked(cursors, end, |doc| out.push(doc));
    out
}

/// One term's doc ids from where `cursor` stands, at or past `start`, up
/// to `end`: each decoded block's ids are copied whole, where the per-doc
/// walk would compare and step once per posting. The reserve is the
/// term's df or the slice's width, whichever is smaller: the df counts
/// the whole file, and every slice of it would reserve that much.
fn single_term_docs(cursor: &mut TermCursor, start: u32, end: u32) -> Vec<u32> {
    let cap = cursor.df.min(u64::from(end.saturating_sub(start)));
    let mut out = Vec::with_capacity(usize::try_from(cap).unwrap_or(0));
    while !cursor.is_exhausted() {
        let ids = &cursor.block_doc_ids[cursor.pos..cursor.block_n];
        let below = ids.partition_point(|&doc| doc < end);
        out.extend_from_slice(&ids[..below]);
        if below < ids.len() {
            break;
        }
        cursor.advance_by(ids.len());
    }
    out
}

/// The union's doc ids below `end` through [`or_walk_heap`], into a vector
/// of its own and out of line. Collected into one vector with the rescan
/// in one function, the heap path cost the dense unions the rescan keeps 3
/// to 7% in a release build (eight and ten common terms over a million
/// docs); apart, they ran as they do without the heap.
#[inline(never)]
fn or_merge_heap(cursors: Vec<TermCursor>, end: u32) -> Vec<u32> {
    let mut out = Vec::new();
    or_walk_heap(cursors, end, |doc| out.push(doc));
    out
}

/// The union's cardinality via a block-at-a-time disjunction count.
/// Walks the cursors one fixed doc-id window at a time, marks each
/// matching doc in a small presence bitset, and accumulates the
/// per-window popcount. Windows partition the doc-id space disjointly,
/// so a doc matching several terms is counted once and no doc spans two
/// windows — the tally equals the distinct-doc union size.
///
/// This replaces the per-doc k-way merge the count path used to share
/// with [`or_merge_unranked`]: that walk rescanned every cursor for each
/// matched doc (cost ∝ union size × term count), which degraded
/// super-linearly on long common-term unions. The windowed walk advances
/// each cursor once per doc and scans the cursor set only once per
/// window, so its cost scales with the union size, not the product. It
/// mirrors the window machinery of [`FtsReader::run_windowed_union`] but
/// drops scoring and the top-k heap, since a count needs neither order
/// nor scores. No doc-id list is materialized.
pub(super) fn or_count_unranked(mut cursors: Vec<TermCursor>) -> u64 {
    // Two terms: `|a ∪ b| = df(a) + df(b) − |a ∩ b|`, and the intersection
    // runs whichever of probing and folding reads less, so the union never
    // pays for a walk of its own.
    if cursors.len() == 2 {
        // Saturating: a df comes from the term's stored metadata, and a
        // corrupt one must not wrap the count.
        let both = cursors[0].df.saturating_add(cursors[1].df);
        return both.saturating_sub(count_and_planned(cursors));
    }
    // Dense union (no dominant term, but the terms together cover a large
    // fraction of the corpus — e.g. a 3-4 common-word title): OR all doc
    // ids into one doc-space bitset and popcount. Avoids the per-window
    // bookkeeping the walk below pays per doc.
    let total_df = df_total(&cursors);
    let max_doc = cursors
        .iter()
        .filter_map(|c| c.blocks.last())
        .map(|b| b.last_doc_id)
        .max()
        .unwrap_or(0);
    if total_df.saturating_mul(OR_COUNT_BITSET_DENSITY_DIVISOR) >= u64::from(max_doc) {
        return or_count_bitset(cursors, max_doc);
    }
    // Fast path: when one term's list dwarfs the rest, count it as
    // `df(dominant) + |others \ dominant|` instead of walking the
    // dominant term's whole posting list. See `or_count_anchored`.
    if let Some(anchor) = dominant_anchor_index(&cursors) {
        return or_count_anchored(cursors, anchor);
    }
    if or_count_prefers_heap(cursors.len(), total_df, max_doc) {
        let mut n = 0u64;
        or_walk_heap(cursors, u32::MAX, |_| n += 1);
        return n;
    }
    let mut present = [0u64; OR_WINDOW_WORDS];
    let mut n = 0u64;
    loop {
        // Smallest current doc among live cursors, aligned down to a
        // window boundary — O(terms) per window, not per doc.
        let mut min_doc = u32::MAX;
        for c in &cursors {
            if !c.is_exhausted() {
                min_doc = min_doc.min(c.current_doc_id());
            }
        }
        if min_doc == u32::MAX {
            break;
        }
        let base = min_doc & !(OR_WINDOW - 1);
        // Saturate so a doc id within OR_WINDOW of u32::MAX can't overflow
        // `base + OR_WINDOW` (matches run_windowed_union); real doc ids
        // never reach that range, so the window stays full-width.
        let window_end = base.saturating_add(OR_WINDOW);
        // Mark each cursor's docs in [base, window_end). `d - base` is in
        // range because every live cursor sits at >= min_doc >= base.
        for c in &mut cursors {
            mark_window(c, base, window_end, &mut present);
        }
        // Count distinct docs in this window and clear for reuse.
        for word in present.iter_mut() {
            n += word.count_ones() as u64;
            *word = 0;
        }
    }
    n
}

/// Mark `c`'s docs below `window_end` into `present`, a window bitmap whose
/// bit 0 is doc `base`, and leave the cursor on its first doc at or past
/// `window_end`. Reads each decoded block as a slice, so a dense term costs
/// a store per doc and a cursor move per block, not a cursor step per doc.
/// Every doc the cursor sits on must be at or past `base`.
#[inline]
pub(super) fn mark_window(c: &mut TermCursor, base: u32, window_end: u32, present: &mut [u64]) {
    while !c.is_exhausted() {
        let docs = &c.block_doc_ids[c.pos..c.block_n];
        let take = docs.partition_point(|&d| d < window_end);
        for &d in &docs[..take] {
            let local = d - base;
            present[(local >> 6) as usize] |= 1u64 << (local & 63);
        }
        if take == 0 {
            break;
        }
        let rest = docs.len() - take;
        c.advance_by(take);
        if rest > 0 {
            break;
        }
    }
}

/// Whether a union count over `terms` cursors holding `total_df` postings
/// in a doc space up to `max_doc` is cheaper through the heap walk than
/// through the windowed walk. Over raw numbers so the boundary is
/// unit-testable without building cursors.
///
/// The windowed walk rescans the whole cursor set once per touched
/// window and then pays one bit op per posting; the heap walk pays a
/// log-of-term-count step per posting and no rescans. A sparse union
/// over many terms (an `IN` list of ids, say) touches a window per few
/// postings, so the rescans dominate and the heap wins; a dense union
/// over a few dozen common terms packs thousands of postings per window,
/// so the rescans are noise and the windowed walk wins. Touched windows
/// are bounded by both the window count and the posting count. Below
/// [`OR_HEAP_MIN_TERMS`] the windowed walk is kept outright.
fn or_count_prefers_heap(terms: usize, total_df: u64, max_doc: u32) -> bool {
    let n_windows = u64::from(max_doc / OR_WINDOW) + 1;
    heap_beats_rescan(terms, total_df, n_windows.min(total_df))
}

/// Disjunction cardinality via a full doc-space bitset: OR every term's
/// doc ids into one bitset, then popcount. Iterates blocks at the byte
/// level so a **bitset-encoded block** (`ENCODING_BITSET`, dense — a common
/// term) is merged by a word-aligned `union[w] |= block[w]` with **no
/// decode**; a PACKED block is decoded doc-ids-only (no tf) and scattered.
/// Overlap between terms is deduplicated for free by the OR. Called only
/// for dense unions (see the gate in [`or_count_unranked`]); the
/// full-corpus alloc + popcount would dominate on a sparse one.
fn or_count_bitset(cursors: Vec<TermCursor>, max_doc: u32) -> u64 {
    let words = max_doc as usize / 64 + 1;
    with_fold_bitset(words, |union| {
        or_cursors_into_bitset(union, &cursors);
        popcount_and_clear(union)
    })
}

/// The set bits of `words`, which are left all zero.
pub(super) fn popcount_and_clear(words: &mut [u64]) -> u64 {
    words
        .iter_mut()
        .map(|w| u64::from(mem::take(w).count_ones()))
        .sum()
}

/// Doc-space bitsets a counting fold works in, all zero between uses.
#[derive(Default)]
struct FoldBitsets {
    a: Vec<u64>,
    b: Vec<u64>,
}

/// Fold bitsets kept between counts, for whichever thread counts next.
static FOLD_POOL: Mutex<Vec<FoldBitsets>> = Mutex::new(Vec::new());

/// Most fold bitsets [`FOLD_POOL`] keeps: one pair per CPU, as no more counts
/// fold at once to any purpose. However many threads count, the pool holds
/// at most this many pairs.
fn fold_pool_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| available_parallelism().map_or(1, NonZeroUsize::get))
}

/// Most words a kept fold bitset may hold. A kept pair holds only what the
/// largest superfile it counted needed — an eighth of a byte per doc — so
/// this bounds the rare outsized one: 64M docs, 8 MiB a bitset, covers a
/// default-sized superfile of docs down to about 16 bytes each. A superfile
/// past it allocates its bitsets per count instead.
const FOLD_BITSETS_KEEP_WORDS: usize = 1 << 20;

/// Run `f` over one all-zero doc-space bitset of `words` words, which it must
/// leave all zero, as [`with_fold_bitsets`] does for two.
fn with_fold_bitset<R>(words: usize, f: impl FnOnce(&mut [u64]) -> R) -> R {
    with_fold_sets(words, false, |a, _| f(a))
}

/// Run `f` over two all-zero doc-space bitsets of `words` words, which it
/// must leave all zero (a fold reads its result with
/// [`popcount_and_clear`]). They come from [`FOLD_POOL`], so a count neither
/// allocates nor clears a corpus-sized bitset before it starts, and go back
/// only once `f` returns: a count that panicked part-way drops its bitsets
/// rather than hand them on marked.
pub(super) fn with_fold_bitsets<R>(words: usize, f: impl FnOnce(&mut [u64], &mut [u64]) -> R) -> R {
    with_fold_sets(words, true, f)
}

/// The pooled bitsets behind [`with_fold_bitset`] and [`with_fold_bitsets`];
/// the second is empty, and never grown, unless `both`.
fn with_fold_sets<R>(words: usize, both: bool, f: impl FnOnce(&mut [u64], &mut [u64]) -> R) -> R {
    if words > FOLD_BITSETS_KEEP_WORDS {
        let mut b = if both { vec![0u64; words] } else { Vec::new() };
        return f(&mut vec![0u64; words], &mut b);
    }
    let pool = || FOLD_POOL.lock().unwrap_or_else(PoisonError::into_inner);
    let mut fb = pool().pop().unwrap_or_default();
    if fb.a.len() < words {
        fb.a.resize(words, 0);
    }
    if both && fb.b.len() < words {
        fb.b.resize(words, 0);
    }
    let b: &mut [u64] = if both { &mut fb.b[..words] } else { &mut [] };
    let out = f(&mut fb.a[..words], b);
    debug_assert!(
        fb.a.iter().chain(&fb.b).all(|&w| w == 0),
        "a fold must leave its bitsets clear"
    );
    let mut kept = pool();
    if kept.len() < fold_pool_cap() {
        kept.push(fb);
    }
    out
}

/// OR one cursor's doc presence into `dest` (a doc-space bitset spanning at
/// least the cursor's largest doc id), reading blocks at the byte level: a
/// **BITSET** block is word-copied at its aligned base word — no expansion to
/// doc ids; a **PACKED** block is decoded doc-ids-only (no tf) and scattered; an
/// inline (df=1) cursor scatters its single pre-decoded doc. `scratch` is a
/// reused `BLOCK_LEN` decode buffer. Shared by the union count
/// ([`or_count_bitset`]) and the counts with exclusions (`count_excluding`)
/// so a common term's dense blocks are merged at memory bandwidth either way.
pub(super) fn or_cursor_into_bitset(
    dest: &mut [u64],
    c: &TermCursor,
    scratch: &mut [u32; BLOCK_LEN],
) {
    // Inline (df=1) and short-form cursors carry their whole list
    // pre-decoded and have no block bytes to slice.
    if c.predecoded {
        for &d in &c.block_doc_ids[..c.block_n] {
            mark_in(dest, 0, d);
        }
        return;
    }
    for b in 0..c.blocks.len() {
        or_block_into(dest, c, b, 0, scratch);
    }
}

/// [`or_cursor_into_bitset`] for each of `cursors`.
pub(super) fn or_cursors_into_bitset(dest: &mut [u64], cursors: &[TermCursor]) {
    let mut scratch = [0u32; BLOCK_LEN];
    for c in cursors {
        or_cursor_into_bitset(dest, c, &mut scratch);
    }
}

/// OR block `b` of `c` into `dest`, a bitset whose bit 0 is doc `base` (a
/// multiple of 64), dropping docs past its span: a **bitset** block's
/// presence words are copied at their aligned word, a **run** block is set
/// as a range, a **packed** block is decoded doc-ids-only and scattered. The
/// block bytes are borrowed in place rather than `.slice()`d, so the walk
/// over a long list takes no refcount per block.
fn or_block_into(
    dest: &mut [u64],
    c: &TermCursor,
    b: usize,
    base: u32,
    scratch: &mut [u32; BLOCK_LEN],
) {
    debug_assert_eq!(base % 64, 0, "a bitset's origin is word-aligned");
    let block = c.blocks[b];
    let bytes = &c.bytes[block.block_byte_offset..block.block_byte_end];
    let hdr = c.block_header(b);
    if hdr.encoding == ENCODING_BITSET {
        debug_assert_eq!(hdr.base % 64, 0, "a bitset block starts on a word");
        // Tfs trail; the presence words are everything between them.
        let presence = &bytes[hdr.payload()..bytes.len() - hdr.tfs_size()];
        let (block_word, base_word) = ((hdr.base / 64) as usize, (base / 64) as usize);
        for (i, chunk) in presence.chunks_exact(8).enumerate() {
            let Some(wi) = (block_word + i).checked_sub(base_word) else {
                continue;
            };
            if wi >= dest.len() {
                break;
            }
            dest[wi] |= u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        }
    } else if hdr.encoding == ENCODING_RUN {
        let span_end = u64::from(base) + dest.len() as u64 * 64;
        let lo = hdr.base.max(base);
        let hi = (u64::from(hdr.base) + hdr.count() as u64).min(span_end) as u32;
        if lo < hi {
            set_bit_range(dest, lo - base, hi - base);
        }
    } else {
        let n = decode_block_doc_ids(bytes, &hdr, scratch);
        for &d in &scratch[..n] {
            mark_in(dest, base, d);
        }
    }
}

/// Set doc `d`'s bit in `dest`, a bitset whose bit 0 is doc `base`, if `d`
/// falls inside its span.
#[inline(always)]
fn mark_in(dest: &mut [u64], base: u32, d: u32) {
    if let Some(local) = d.checked_sub(base)
        && let Some(word) = dest.get_mut((local >> 6) as usize)
    {
        *word |= 1u64 << (local & 63);
    }
}

/// Set bits `[from, to)` of a doc-space bitset: the edge words masked, the
/// words between filled whole.
pub(super) fn set_bit_range(dest: &mut [u64], from: u32, to: u32) {
    if from >= to {
        return;
    }
    let (first, last) = ((from / 64) as usize, ((to - 1) / 64) as usize);
    let lo = u64::MAX << (from % 64);
    let hi = u64::MAX >> (63 - (to - 1) % 64);
    if first == last {
        dest[first] |= lo & hi;
        return;
    }
    dest[first] |= lo;
    dest[first + 1..last].fill(u64::MAX);
    dest[last] |= hi;
}

/// Pick the cursor whose df dominates the union, or `None` if no term is
/// dominant enough for the anchored count to pay off. Dominant means the
/// largest df is at least [`OR_COUNT_ANCHOR_DOMINANCE`]× the sum of all
/// the other dfs — so the dominant list is longer than everything else
/// combined by a wide margin, exactly when replacing its full walk with a
/// df read + skip-probe wins. Requires ≥ 2 cursors (a single cursor's
/// count is just its df, and the windowed walk handles it trivially).
fn dominant_anchor_index(cursors: &[TermCursor]) -> Option<usize> {
    dominant_anchor_of_dfs(cursors.iter().map(|c| c.df))
}

/// The routing decision behind [`dominant_anchor_index`], over raw dfs so
/// the boundary behaviour is unit-testable without building cursors.
/// Returns the index of the term whose df is at least
/// [`OR_COUNT_ANCHOR_DOMINANCE`]× the sum of all the others' (a `>=`
/// boundary — exactly `N×` still routes to the anchor), or `None` when no
/// term dominates or there are fewer than two terms.
fn dominant_anchor_of_dfs(dfs: impl IntoIterator<Item = u64>) -> Option<usize> {
    let dfs: Vec<u64> = dfs.into_iter().collect();
    if dfs.len() < 2 {
        return None;
    }
    let (max_idx, &max_df) = dfs.iter().enumerate().max_by_key(|&(_, df)| *df)?;
    let others_df: u64 = dfs
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != max_idx)
        .map(|(_, df)| *df)
        .sum();
    // `others_df == 0` (every other term inline/df-1-summing-to-0 is
    // impossible, but guard anyway) trivially dominates.
    match max_df >= others_df.saturating_mul(OR_COUNT_ANCHOR_DOMINANCE) {
        true => Some(max_idx),
        false => None,
    }
}

/// Disjunction cardinality anchored on a dominant term. The union size is
/// `df(anchor) + |(the other cursors' union) \ anchor|`: every doc in the
/// anchor is counted once by its df (no decode), and each doc in the
/// rarer cursors' union is added once iff the anchor does not already
/// contain it. The anchor is advanced monotonically by `skip_to` as the
/// others' merge frontier ascends, so its skip table is traversed once
/// rather than its whole posting list decoded.
fn or_count_anchored(mut cursors: Vec<TermCursor>, anchor_idx: usize) -> u64 {
    let mut anchor = cursors.swap_remove(anchor_idx);
    let mut n = anchor.df;
    // The rarer cursors' docs, one window at a time, deduplicated by the
    // window bitmap. A window holding few of them tests each against the
    // anchor; a window holding many reads the anchor's own window from its
    // blocks and counts the rarer docs it lacks a word at a time.
    let mut present = [0u64; OR_WINDOW_WORDS];
    let mut anchor_win = [0u64; OR_WINDOW_WORDS];
    let mut anchor_block = 0usize;
    let mut scratch = [0u32; BLOCK_LEN];
    loop {
        let mut min_doc = u32::MAX;
        for c in &cursors {
            if !c.is_exhausted() {
                min_doc = min_doc.min(c.current_doc_id());
            }
        }
        if min_doc == u32::MAX {
            break;
        }
        let base = min_doc & !(OR_WINDOW - 1);
        let window_end = base.saturating_add(OR_WINDOW);
        for c in &mut cursors {
            mark_window(c, base, window_end, &mut present);
        }
        let marked: u32 = present.iter().map(|w| w.count_ones()).sum();
        if marked >= ANCHOR_WINDOW_MIN_DOCS {
            anchor_win.fill(0);
            or_blocks_into_window(
                &anchor,
                &mut anchor_block,
                base,
                &mut anchor_win,
                &mut scratch,
            );
            for (w, a) in present.iter_mut().zip(&anchor_win) {
                n += u64::from((mem::take(w) & !a).count_ones());
            }
            continue;
        }
        // Membership by bit-test, not `skip_to`: the anchor is the dominant
        // (common ⇒ dense) term, so its blocks are bitsets. `contains` answers
        // in one word-load; `skip_to` would decode each block — expanding its
        // all its doc ids — as the frontier drags the anchor across its whole list.
        for (word_idx, word) in present.iter_mut().enumerate() {
            let mut bits = mem::take(word);
            while bits != 0 {
                let doc = base + ((word_idx as u32) << 6) + bits.trailing_zeros();
                bits &= bits - 1;
                if !anchor.contains(doc) {
                    n += 1;
                }
            }
        }
    }
    n
}

/// Rarer-term docs in a window at which the anchored count reads the
/// anchor's window from its blocks rather than testing each doc: a dense
/// anchor covers a window in a few dozen blocks, whose presence words cost
/// less than this many membership tests.
const ANCHOR_WINDOW_MIN_DOCS: u32 = 64;

/// OR `c`'s docs in the window `[base, base + OR_WINDOW)` into `dest` (bit 0
/// is doc `base`), from its blocks rather than by walking it: a bitset block's
/// presence words are copied, a run block is set as a range, a packed block is
/// decoded and scattered. `from_block` is the first block that may reach the
/// window; it advances past blocks that end before it, so windows must come
/// in ascending order.
fn or_blocks_into_window(
    c: &TermCursor,
    from_block: &mut usize,
    base: u32,
    dest: &mut [u64; OR_WINDOW_WORDS],
    scratch: &mut [u32; BLOCK_LEN],
) {
    if c.predecoded {
        for &d in &c.block_doc_ids[..c.block_n] {
            mark_in(dest, base, d);
        }
        return;
    }
    while *from_block < c.blocks.len() && c.blocks[*from_block].last_doc_id < base {
        *from_block += 1;
    }
    let end = base.saturating_add(OR_WINDOW);
    for b in *from_block..c.blocks.len() {
        let first = match b {
            0 => 0,
            _ => c.blocks[b - 1].last_doc_id + 1,
        };
        if first >= end {
            break;
        }
        or_block_into(dest, c, b, base, scratch);
    }
}

/// Document `d`'s stored length from a doc-lengths array of `u16`
/// little-endian entries.
#[inline]
pub(super) fn read_doc_length(region: &[u8], d: usize) -> u32 {
    let at = d * format::fts::DOC_LENGTH_BYTES;
    u32::from(u16::from_le_bytes([region[at], region[at + 1]]))
}

impl FtsReader {
    /// Whether this file's posting blocks take the wide header reading
    /// (see [`format::fts::wide_blocks`]).
    #[inline]
    pub(super) fn wide_blocks(&self) -> bool {
        format::fts::wide_blocks(self.version)
    }
}

/// Refuse a blob version this crate does not read: one below
/// [`VERSION_MIN`] with the instruction to reindex it, anything else
/// unknown as unsupported.
fn check_version(version: u32) -> Result<(), FtsError> {
    match version {
        format::fts::VERSION_V7
        | format::fts::VERSION_V8
        | format::fts::VERSION_V9
        | format::fts::VERSION_V10
        | format::fts::VERSION_V11 => Ok(()),
        v if v < VERSION_MIN => Err(FtsError::IndexTooOld { version: v }),
        v => Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
            "fts section version {v}"
        )))),
    }
}

#[cfg(test)]
mod tests {

    /// A fold that panics part-way drops its bitsets rather than return
    /// them marked, so the folds after it start from all zero, for one bitset
    /// or two and whatever size each asks for; a superfile past what the pool
    /// keeps folds over bitsets of its own, which the pool never takes.
    #[test]
    fn a_fold_after_a_panicked_one_starts_from_clear_bitsets() {
        const WORDS: usize = 64;
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            with_fold_bitsets(WORDS, |a, b| {
                a.fill(u64::MAX);
                b.fill(u64::MAX);
                panic!("a fold fails part-way");
            })
        }));
        assert!(panicked.is_err());
        with_fold_bitset(WORDS / 2, |a| {
            assert!(a.iter().all(|&w| w == 0), "the first bitset is cleared");
        });
        with_fold_bitsets(WORDS, |a, b| {
            assert!(
                a.iter().chain(b.iter()).all(|&w| w == 0),
                "both bitsets are cleared at their full size"
            );
        });
        with_fold_bitsets(FOLD_BITSETS_KEEP_WORDS + 1, |a, b| {
            assert_eq!(
                (a.len(), b.len()),
                (FOLD_BITSETS_KEEP_WORDS + 1, FOLD_BITSETS_KEEP_WORDS + 1)
            );
            assert!(a.iter().chain(b.iter()).all(|&w| w == 0));
        });
        let pool = FOLD_POOL.lock().expect("pool");
        assert!(
            pool.iter().all(|fb| fb.a.len() <= FOLD_BITSETS_KEEP_WORDS),
            "the pool kept an outsized bitset"
        );
    }

    /// However many threads fold at once, the pool keeps at most one pair of
    /// bitsets per CPU once they are done.
    #[test]
    fn the_fold_pool_keeps_no_more_bitsets_than_cpus() {
        const WORDS: usize = 64;
        let threads = 4 * fold_pool_cap();
        let all_folding = Barrier::new(threads);
        thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    with_fold_bitsets(WORDS, |_, _| {
                        all_folding.wait();
                    });
                });
            }
        });
        let kept = FOLD_POOL.lock().expect("pool").len();
        assert!(
            kept <= fold_pool_cap(),
            "{kept} pairs kept, cap {}",
            fold_pool_cap()
        );
    }

    /// What the writer stores for a column's norms must be exactly what the
    /// reader used to derive from its lengths. The scorer indexes the buckets
    /// and divides by the totals, so a writer that disagreed with the old
    /// derivation would change scores silently, on files that still open and
    /// still pass their CRCs.
    ///
    /// Documents are built at a spread of lengths so the quantization is
    /// exercised across buckets rather than landing them all in one.
    /// A lazy reader that does not verify what it reads fetches the bucket
    /// array alone to score, a third of the length span, and scores as an
    /// eager reader does; one that verifies reads the span whole, with its
    /// CRC.
    #[tokio::test]
    async fn an_unverified_lazy_open_reads_only_the_buckets_to_score() {
        /// Documents in the fixture.
        const N_DOCS: u32 = 512;

        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for doc in 0..N_DOCS {
            let n_tokens = (doc % 64) + 1;
            let text = (0..n_tokens)
                .map(|t| format!("t{t}"))
                .collect::<Vec<_>>()
                .join(" ");
            b.add_doc(0, doc, &text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let eager = FtsReader::open(blob.clone(), json).expect("open");

        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(blob.clone()));
        let lazy = FtsReader::open_lazy(src, json, OpenOptions { verify_crc: false })
            .await
            .expect("open_lazy");
        let column = &lazy.columns[0];
        let stored = column
            .stored_norms
            .as_ref()
            .expect("the current version stores its norms");
        assert_eq!(
            column.norms_fetch_range(),
            stored.buckets,
            "unverified, the read is the bucket array alone"
        );
        assert_eq!(
            column.norms_fetch_range().len() * 3 + format::CRC_BYTES,
            column.array_with_crc_range().len(),
            "two bytes of length and one of bucket per document, then the CRC"
        );
        assert_eq!(
            column.norms().dl_norm_k1.buckets(),
            eager.columns[0].norms().dl_norm_k1.buckets(),
            "the lazy reader scores by the same buckets"
        );

        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(blob));
        let verifying = FtsReader::open_lazy(src, json, OpenOptions { verify_crc: true })
            .await
            .expect("open_lazy");
        let column = &verifying.columns[0];
        assert_eq!(
            column.norms_fetch_range(),
            column.array_with_crc_range(),
            "a verified read takes the span the CRC covers"
        );
    }

    #[test]
    fn stored_norms_match_what_deriving_them_would_give() {
        /// Documents in the fixture.
        const N_DOCS: u32 = 512;

        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for doc in 0..N_DOCS {
            // Lengths from 1 to 64 tokens, so the buckets spread.
            let n_tokens = (doc % 64) + 1;
            let text = (0..n_tokens)
                .map(|t| format!("t{t}"))
                .collect::<Vec<_>>()
                .join(" ");
            b.add_doc(0, doc, &text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let reader = FtsReader::open(blob, json).expect("open");

        let column = &reader.columns[0];
        let stored = column
            .stored_norms
            .as_ref()
            .expect("the current version stores its norms");

        let lengths = reader.read_doc_lengths(0).expect("doc lengths");
        assert_eq!(lengths.len(), N_DOCS as usize);

        let norms = column.norms();
        let derived: Vec<u8> = lengths.iter().map(|&dl| bm25::quantize_len(dl)).collect();
        assert_eq!(
            norms.dl_norm_k1.buckets(),
            derived.as_slice(),
            "the stored buckets must equal quantizing the stored lengths"
        );

        let expected_stats = ColumnLengthStats::from_lengths(lengths.iter().copied());
        assert_eq!(
            stored.stats, expected_stats,
            "the stored totals must equal summing the stored lengths"
        );
        assert_eq!(
            norms.length_stats, expected_stats,
            "and the totals the scorer divides by must be those"
        );

        let lo = derived.iter().copied().min().expect("a non-empty column");
        let hi = derived.iter().copied().max().expect("a non-empty column");
        assert_eq!(
            stored.occupied,
            (lo, hi),
            "the stored occupied range must be the buckets the documents actually use"
        );
    }

    /// The reader refuses to open a column whose entry names a filter
    /// this engine cannot reproduce, and names the field and the value
    /// so the operator can see which engine version is needed.
    ///
    /// Asserted through `open` rather than on the resolver alone: the
    /// resolver is where the decision is made, but the `map_err` that
    /// turns it into a read error is the part a caller actually sees,
    /// and a `?` dropped there would let the column open unfiltered.
    /// A column entry with no `tokenizer` key, or one naming `standard`,
    /// opens; one naming any other analyzer refuses the file with the
    /// re-create instruction rather than querying it with `standard`.
    #[test]
    fn open_accepts_only_an_absent_or_standard_recorded_tokenizer() {
        let (blob, json) = build_blob();
        let absent = json.replace(r#","tokenizer":"standard""#, "");
        assert_ne!(absent, json, "fixture did not patch");
        FtsReader::open(blob.clone(), &absent).expect("an absent tokenizer opens");
        FtsReader::open(blob.clone(), &json).expect("standard opens");

        let removed = json.replace(r#""tokenizer":"standard""#, r#""tokenizer":"ascii_lower""#);
        let msg = FtsReader::open(blob, &removed)
            .expect_err("a removed analyzer must fail the open")
            .to_string();
        assert!(
            msg.contains("ascii_lower") && msg.contains("re-create"),
            "the error names the analyzer and the fix, got: {msg}"
        );
    }

    #[tokio::test]
    async fn open_refuses_a_column_naming_an_unreproducible_filter() {
        let (blob, json) = build_blob();
        // Sanity: the fixture opens before it is tampered with, so a
        // failure below is the filter and not the fixture.
        FtsReader::open(blob.clone(), &json).expect("untampered fixture opens");
        for (field, value) in [("stopwords", "german"), ("stemmer", "porter")] {
            let patched = json.replace(
                r#""tokenizer":"#,
                &format!(r#""{field}":"{value}","tokenizer":"#),
            );
            assert_ne!(patched, json, "{field}: fixture did not patch");
            let err = FtsReader::open(blob.clone(), &patched)
                .expect_err("an unreproducible filter must fail the open");
            let msg = err.to_string();
            assert!(
                msg.contains(field) && msg.contains(value),
                "{field}: the error must name the field and value, got: {msg}"
            );
        }
    }
    use std::{
        collections::{HashMap, HashSet},
        panic::{AssertUnwindSafe, catch_unwind},
        sync::{
            Barrier, Mutex,
            atomic::{AtomicBool, AtomicU32, Ordering},
        },
        thread::{self, ThreadId},
        time::Duration,
    };

    use async_trait::async_trait;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use tokio::{
        join, spawn,
        sync::Notify,
        task::{spawn_blocking, yield_now},
        time::{sleep, timeout},
    };

    use super::{super::test_util::*, *};
    use crate::superfile::{
        BytesLazyByteSource, LazyByteSourceError,
        fts::{bm25, builder::FtsBuilder, reader::BoolMode, tokenize::StandardTokenizer},
    };

    /// Byte offset of the blob's version field.
    const VERSION_FIELD: std::ops::Range<usize> = 8..12;

    /// A corpus big enough to produce several posting blocks, so the
    /// blob carries a real skip table and coarse table rather than a
    /// single degenerate block.
    fn multi_block_blob() -> (Bytes, &'static str) {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for doc in 0..600u32 {
            let text = match doc % 3 {
                0 => format!("common shared alpha d{doc}"),
                1 => format!("common beta d{doc}"),
                _ => format!("common shared gamma delta d{doc}"),
            };
            b.add_doc(0, doc, &text).expect("add doc");
        }
        (
            Bytes::from(b.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
        )
    }

    /// A blob older than the oldest version this engine reads is refused
    /// on open, eager and lazy alike, with an error that says to reindex;
    /// a version past the newest is unsupported, not too old.
    #[tokio::test]
    async fn a_blob_below_the_minimum_version_is_refused_with_a_reindex_hint() {
        let (blob, json) = multi_block_blob();
        let stamp = |version: u32| {
            let mut bytes = blob.to_vec();
            bytes[VERSION_FIELD].copy_from_slice(&version.to_le_bytes());
            Bytes::from(bytes)
        };
        let old = format::fts::VERSION_MIN - 1;
        let err = FtsReader::open(stamp(old), json).expect_err("a pre-minimum blob");
        assert!(
            matches!(err, FtsError::IndexTooOld { version } if version == old),
            "{err:?}"
        );
        assert!(err.to_string().contains("reindex"), "{err}");

        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(stamp(old)));
        let err = FtsReader::open_lazy(src, json, OpenOptions::default())
            .await
            .expect_err("a pre-minimum blob, lazily");
        assert!(matches!(err, FtsError::IndexTooOld { .. }), "{err:?}");

        let err =
            FtsReader::open(stamp(format::fts::VERSION_V11 + 1), json).expect_err("a future blob");
        assert!(
            matches!(err, FtsError::Read(ReadError::UnsupportedVersion(_))),
            "{err:?}"
        );
    }

    #[test]
    fn current_version_bounds_need_no_scale_correction() {
        let (blob, json) = multi_block_blob();
        assert_eq!(
            read_u32_le(&blob[VERSION_FIELD]),
            format::fts::VERSION_CURRENT,
            "the builder writes the current version"
        );
        let r = FtsReader::open(blob, json).expect("open");
        assert_eq!(
            r.columns[0].bound_scale(),
            1.0,
            "bounds baked at the current scale need no correction"
        );
    }

    #[test]
    fn a_reused_scored_view_matches_a_freshly_derived_one() {
        // The view is memoized on the reader so it is not rebuilt per
        // call. Reuse is only safe because deriving it is a pure
        // function of the override, so a cached view and a fresh one
        // must agree on everything scoring reads — otherwise a query
        // would score differently depending on whether it happened to
        // be the first to ask.
        let (blob, json) = multi_block_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let over = bm25::Bm25Params::new(1.4, 0.6);
        let first = r.with_bm25_override(over);
        let second = r.with_bm25_override(over);
        let a = &first.columns[0];
        let b = &second.columns[0];
        assert_eq!(a.params, b.params);
        assert_eq!(a.avgdl(), b.avgdl());
        assert_eq!(a.bound_scale(), b.bound_scale());
        assert_eq!(a.length_stats(), b.length_stats());
        for doc in 0..r.n_docs() {
            assert_eq!(
                a.dl_norm_k1().get(doc),
                b.dl_norm_k1().get(doc),
                "doc {doc} normalizer differs between two derivations"
            );
        }
    }

    #[test]
    fn a_current_file_is_scored_at_the_average_it_declares() {
        // The writer bakes the table-wide average into the file, so the
        // reader takes it as given: no correction, no inflation.
        let (blob, json) = multi_block_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let col = &r.columns[0];
        assert_eq!(col.bound_scale(), 1.0);
        assert_eq!(
            col.avgdl(),
            col.length_stats().avgdl(),
            "dense: declared == over documents"
        );

        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "alpha beta gamma").expect("doc");
        b.add_doc(0, 1, "").expect("null row");
        let sparse = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(
            sparse.columns[0].avgdl(),
            3.0,
            "the declared average counts the document that carries tokens, not the null row"
        );
        assert_eq!(sparse.columns[0].bound_scale(), 1.0);
    }

    /// A corpus whose common term's blocks tie closely, with every other
    /// row null when `sparse`, so block-max pruning is live at small `k`
    /// and the corrected collection size and average both matter.
    fn tied_corpus(sparse: bool) -> (Bytes, &'static str) {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        let mut next = 0u32;
        for doc in 0..900u32 {
            let text = match doc % 4 {
                0 => format!("common shared d{doc}"),
                1 => format!("common common beta d{doc} pad pad"),
                2 => format!("common gamma d{doc} pad"),
                _ => format!("common shared delta d{doc} pad pad pad pad"),
            };
            b.add_doc(0, next, &text).expect("add doc");
            next += 1;
            if sparse {
                b.add_doc(0, next, "").expect("null row");
                next += 1;
            }
        }
        (
            Bytes::from(b.finish().expect("finish")),
            r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#,
        )
    }

    /// A corpus built twice: once in row order, once under an ordering
    /// of its own with the map that undoes it. Every query must return
    /// the same rows with the same scores.
    ///
    /// This is the whole contract of the doc-id map. The second blob's
    /// postings, skip tables, block maxima and doc lengths are all in a
    /// different order from the first's, and the kernels work in that
    /// order from end to end; only the ids handed back are translated.
    /// If any path returned a blob id where a row was meant, or
    /// translated one twice, the two rankings would disagree.
    #[tokio::test]
    async fn a_reordered_blob_answers_exactly_like_a_row_ordered_one() {
        const N_DOCS: u32 = 900;
        // A deterministic permutation that moves every document and is
        // its own kind of scramble rather than a rotation: coprime
        // stride over the corpus length.
        const STRIDE: u32 = 401;
        fn text_for(row: u32) -> String {
            let mut t = String::new();
            for _ in 0..=(row % 3) {
                t.push_str("alpha ");
            }
            if !row.is_multiple_of(3) {
                t.push_str("beta ");
            }
            if row.is_multiple_of(7) {
                t.push_str("gamma ");
            }
            for f in 0..(row % 5) {
                t.push_str(&format!("pad{f} "));
            }
            t.trim().to_string()
        }

        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;

        let mut plain = FtsBuilder::new();
        plain
            .register_column("body".into(), false)
            .expect("register");
        for row in 0..N_DOCS {
            plain.add_doc(0, row, &text_for(row)).expect("add doc");
        }
        let plain_blob = Bytes::from(plain.finish().expect("finish"));
        let plain_reader = FtsReader::open(plain_blob.clone(), json).expect("open plain");

        // `map[id]` is the row that doc id stands for. Documents are fed
        // in that order, so the blob's own ids run 0..N ascending while
        // the rows they carry do not.
        let map: Vec<u32> = (0..N_DOCS).map(|id| (id * STRIDE) % N_DOCS).collect();
        let mut distinct: Vec<u32> = map.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            N_DOCS as usize,
            "the map must be a permutation"
        );
        assert!(
            map.iter().enumerate().any(|(i, &r)| i as u32 != r),
            "and must move something"
        );

        let mut reordered = FtsBuilder::new();
        reordered
            .register_column("body".into(), false)
            .expect("register");
        for (id, &row) in map.iter().enumerate() {
            reordered
                .add_doc(0, id as u32, &text_for(row))
                .expect("add doc");
        }
        reordered.doc_map = Some(map.clone());
        let reordered_blob = Bytes::from(reordered.finish().expect("finish"));

        // The map no longer moves the version: its length is recorded in the
        // trailer before the doc-lengths directory, so one version covers a
        // blob with a map and one without. What still has to differ is
        // whether the reader finds a map at all.
        assert_eq!(
            read_u32_le(&reordered_blob[VERSION_FIELD]),
            format::fts::VERSION_CURRENT,
            "a blob with a map is written at the current version"
        );
        assert_eq!(
            read_u32_le(&plain_blob[VERSION_FIELD]),
            format::fts::VERSION_CURRENT,
            "and so is one without"
        );
        let reordered_reader = FtsReader::open(reordered_blob, json).expect("open reordered");
        assert!(reordered_reader.has_doc_map());
        assert!(!plain_reader.has_doc_map());

        for terms in [
            &["alpha"][..],
            &["beta"][..],
            &["gamma"][..],
            &["alpha", "beta"][..],
            &["alpha", "beta", "gamma"][..],
        ] {
            // Taken whole, the two must agree on every row and score.
            let mut want_all = top(&plain_reader, terms, N_DOCS as usize + 1).await;
            let mut got_all = top(&reordered_reader, terms, N_DOCS as usize + 1).await;
            assert_eq!(want_all.len(), got_all.len(), "{terms:?} match count");
            let by_row = |v: &mut Vec<(RowId, f32)>| v.sort_by_key(|&(d, _)| d);
            by_row(&mut want_all);
            by_row(&mut got_all);
            for ((dw, sw), (dg, sg)) in want_all.iter().zip(got_all.iter()) {
                assert_eq!(dw, dg, "{terms:?}: row {dw} missing or extra");
                assert!(
                    (sw - sg).abs() < 1e-5,
                    "{terms:?} row {dw}: score {sw} vs {sg}"
                );
            }

            // Truncated, the two agree on the scores at every rank but
            // need not name the same rows, because equal scores are
            // broken by the blob's own doc id and reordering is exactly
            // a change to that. So check the score sequence, and that
            // each row a truncated search returns really does carry the
            // score its rank claims.
            let truth: HashMap<RowId, f32> = want_all.iter().copied().collect();
            for k in [1usize, 5, 50, 400] {
                let want = top(&plain_reader, terms, k).await;
                let got = top(&reordered_reader, terms, k).await;
                assert_eq!(want.len(), got.len(), "{terms:?} k={k} length");
                for (rank, ((_, sw), (dg, sg))) in want.iter().zip(got.iter()).enumerate() {
                    assert!(
                        (sw - sg).abs() < 1e-5,
                        "{terms:?} k={k} rank {rank}: score {sw} vs {sg}"
                    );
                    assert!(
                        truth.get(dg).is_some_and(|t| (t - sg).abs() < 1e-5),
                        "{terms:?} k={k} rank {rank}: row {dg} scored {sg}, not its real score"
                    );
                }
            }
        }
    }

    /// A corrupt doc-id map is refused at open rather than silently
    /// answering with the wrong rows.
    ///
    /// The map is the one region whose damage a query could not
    /// otherwise notice: every id it returns would still be in range and
    /// still carry a real score, just attached to the wrong document. So
    /// both of its guards are checked, the region's own checksum and its
    /// length against the document count.
    #[tokio::test]
    async fn a_damaged_doc_map_is_refused_at_open() {
        const N_DOCS: u32 = 300;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for row in 0..N_DOCS {
            b.add_doc(0, row, &format!("alpha t{}", row % 17))
                .expect("add doc");
        }
        b.doc_map = Some((0..N_DOCS).rev().collect());
        let blob = b.finish().expect("finish");
        assert_eq!(
            read_u32_le(&blob[VERSION_FIELD]),
            format::fts::VERSION_CURRENT,
            "fixture must be a mapped blob"
        );
        FtsReader::open(Bytes::from(blob.clone()), json).expect("the fixture opens clean");

        let dls_off =
            read_u64_le(&blob[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;
        // Where the map starts is derived, not stored: the directory
        // offset less one entry per document and the region's checksum.
        let map_len = N_DOCS as usize * U32_BYTES + format::CRC_BYTES;
        let map_off = dls_off - map_len;

        // A flipped byte inside the region.
        let mut flipped = blob.clone();
        flipped[map_off] ^= 0xff;
        let err = FtsReader::open(Bytes::from(flipped), json).expect_err("corrupt map");
        assert!(
            matches!(
                err,
                FtsError::Read(ReadError::ChecksumMismatch {
                    section: "fts/doc-map",
                    ..
                })
            ),
            "expected a doc-map checksum failure, got {err:?}"
        );

        // A moved directory. Deriving the map's start from the directory
        // means moving the directory moves the map with it, so the bytes
        // read as the map are no longer the bytes written as one and its
        // checksum fails. Nothing is read at the boundary the header
        // claims without being checked first.
        let mut moved_dir = blob.clone();
        moved_dir[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES]
            .copy_from_slice(&((dls_off - U32_BYTES) as u64).to_le_bytes());
        let err = FtsReader::open(Bytes::from(moved_dir), json).expect_err("moved directory");
        assert!(
            matches!(
                err,
                FtsError::Read(ReadError::Malformed(_))
                    | FtsError::Read(ReadError::ChecksumMismatch { .. })
            ),
            "expected the moved boundary to be refused, got {err:?}"
        );

        // A document count the region cannot hold. The map's size comes
        // from it, so inflating it claims a region larger than the blob
        // has room for below the directory.
        let mut huge_docs = blob.clone();
        huge_docs[hdr::N_DOCS_OFF..hdr::N_DOCS_OFF + U32_BYTES]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        let err = FtsReader::open(Bytes::from(huge_docs), json).expect_err("impossible map");
        assert!(
            matches!(err, FtsError::Read(ReadError::Malformed(_))),
            "expected a malformed-header failure, got {err:?}"
        );
    }

    /// The map region round-trips for any permutation, at sizes either
    /// side of the block and partition boundaries.
    ///
    /// The map is written as a flat array and read back as one, so a
    /// mistake here is an off-by-one that survives both checksums: the
    /// ids stay in range and the search still answers, just about
    /// different documents. Checking the identity permutation alongside
    /// the scrambled ones keeps the degenerate case honest too, since it
    /// is the one where a bug is invisible to a parity test.
    #[tokio::test]
    async fn a_doc_map_round_trips_for_any_permutation() {
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        for (n_docs, seed) in [(1u32, 1u64), (2, 2), (7, 3), (128, 4), (129, 5), (513, 6)] {
            let mut rng = StdRng::seed_from_u64(seed);
            // Fisher-Yates, so every size gets a genuine permutation
            // rather than a stride that happens to be well behaved.
            let mut map: Vec<u32> = (0..n_docs).collect();
            for i in (1..map.len()).rev() {
                map.swap(i, rng.random_range(0..=i));
            }
            let text = |row: u32| format!("alpha t{} p{}", row % 11, row % 5);

            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for (id, &row) in map.iter().enumerate() {
                b.add_doc(0, id as u32, &text(row)).expect("add doc");
            }
            b.doc_map = Some(map.clone());
            let blob = Bytes::from(b.finish().expect("finish"));
            let r = FtsReader::open(blob, json).expect("open");

            assert!(r.has_doc_map(), "n_docs {n_docs}: the map must survive");
            let doc_map = r.doc_map().expect("an in-memory map is read at open");
            for (id, &row) in map.iter().enumerate() {
                assert_eq!(
                    doc_map.row_of(FtsDocId::new(id as u32)),
                    RowId::new(row),
                    "n_docs {n_docs}: doc id {id} must name row {row}"
                );
            }
            // And the rows a search names are the rows that carry the
            // term, not the blob positions that happen to hold them.
            let mut got: Vec<u32> = r
                .search("body", &["alpha"], n_docs as usize, BoolMode::Or)
                .await
                .expect("search")
                .into_iter()
                .map(|(row, _)| row.get())
                .collect();
            got.sort_unstable();
            assert_eq!(
                got,
                (0..n_docs).collect::<Vec<_>>(),
                "n_docs {n_docs}: every row carries `alpha`"
            );

            // The unranked path promises ascending rows, which on a
            // reordered blob means it must re-sort after translating:
            // the walk emits blob ids in order, and those are a
            // different order.
            let matched = r
                .token_match("body", &["alpha"], BoolMode::Or)
                .await
                .expect("token match")
                .0;
            assert!(
                matched.windows(2).all(|w| w[0] < w[1]),
                "n_docs {n_docs}: unranked rows must come back ascending, got {matched:?}"
            );
            assert_eq!(
                matched.iter().map(|r| r.get()).collect::<Vec<_>>(),
                (0..n_docs).collect::<Vec<_>>(),
                "n_docs {n_docs}: and name every row exactly once"
            );
        }
    }

    /// Smallest read the window source answers with a window onto its
    /// whole backing fetch rather than a copy, the way the block cache
    /// answers a read of a cache block or more.
    const WINDOW_AT_LEAST: u64 = 1024;

    /// Documents in the reordered blob: enough that its doc-id map is the
    /// one read of a lazy open the window source answers with a window.
    const WINDOWED_MAP_DOCS: u32 = 1024;

    /// The fetch a [`WindowSource`] hands out windows onto. Each window
    /// holds one reference to it, so the count on the inner `Arc` says who
    /// still has one.
    struct Backing(Arc<Vec<u8>>);

    impl AsRef<[u8]> for Backing {
        fn as_ref(&self) -> &[u8] {
            &self.0
        }
    }

    /// A source that answers every read of at least [`WINDOW_AT_LEAST`]
    /// bytes with a window onto one shared fetch, and smaller reads with a
    /// copy, the way the block cache serves a cold read.
    struct WindowSource {
        backing: Arc<Vec<u8>>,
    }

    #[async_trait]
    impl LazyByteSource for WindowSource {
        fn size(&self) -> u64 {
            self.backing.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let range = start as usize..(start + len) as usize;
            Ok(if len < WINDOW_AT_LEAST {
                Bytes::copy_from_slice(&self.backing[range])
            } else {
                Bytes::from_owner(Backing(Arc::clone(&self.backing))).slice(range)
            })
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// A reader opened over a source that answers big reads with windows
    /// onto a larger fetch keeps none of that fetch alive through its
    /// doc-id map. The window the map arrives in can be the tail of a
    /// fetch many times the map's size, and the reader holds the map for
    /// its whole life, so over a lazy source it takes a copy of the map
    /// alone.
    #[tokio::test]
    async fn a_lazy_reader_keeps_no_window_onto_a_fetch_through_its_doc_map() {
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let map: Vec<u32> = (0..WINDOWED_MAP_DOCS).rev().collect();
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for (id, &row) in map.iter().enumerate() {
            b.add_doc(0, id as u32, &format!("alpha t{}", row % 7))
                .expect("add doc");
        }
        b.doc_map = Some(map.clone());
        let backing = Arc::new(b.finish().expect("finish"));
        assert!(
            u64::from(WINDOWED_MAP_DOCS) * (U32_BYTES as u64) >= WINDOW_AT_LEAST,
            "the map must be big enough to arrive as a window"
        );

        let source: Arc<dyn LazyByteSource> = Arc::new(WindowSource {
            backing: Arc::clone(&backing),
        });
        let reader = FtsReader::open_lazy(source, json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");

        assert!(reader.has_doc_map(), "the map must survive a lazy open");
        let doc_map = reader.ensure_doc_map().await.expect("doc-id map");
        for (id, &row) in map.iter().enumerate() {
            assert_eq!(
                doc_map.row_of(FtsDocId::new(id as u32)),
                RowId::new(row),
                "doc id {id} must name row {row}"
            );
        }
        assert_eq!(
            Arc::strong_count(&backing),
            2,
            "only this test and the source hold the fetch; the reader kept a window onto it"
        );
    }

    /// The ranged union kernel windows on the blob's own ids while
    /// handing back rows, so slicing a reordered blob into windows and
    /// merging must equal the unsliced search.
    ///
    /// This is the one kernel whose input is in blob space and whose
    /// output is in row space at the same time. If the window bounds
    /// were translated, or the hits were not, the union of the slices
    /// would drop or double-count documents that a single call finds.
    #[tokio::test]
    async fn a_sliced_range_union_over_a_reordered_blob_equals_the_unsliced_search() {
        const N_DOCS: u32 = 600;
        const STRIDE: u32 = 137;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let text = |row: u32| format!("alpha beta t{} t{}", row % 23, row % 31);

        let map: Vec<u32> = (0..N_DOCS).map(|id| (id * STRIDE) % N_DOCS).collect();
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for (id, &row) in map.iter().enumerate() {
            b.add_doc(0, id as u32, &text(row)).expect("add doc");
        }
        b.doc_map = Some(map);
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");

        let terms = ["alpha", "beta"];
        let whole = r
            .search_or_range_pretokenized("body", &terms, N_DOCS as usize, 0, N_DOCS)
            .await
            .expect("unsliced");

        assert_eq!(
            whole.len(),
            N_DOCS as usize,
            "every row carries both terms, so the unsliced search is not empty"
        );

        let mut sliced: Vec<(RowId, f32)> = Vec::new();
        for &(lo, hi) in &[(0u32, 149u32), (149, 400), (400, N_DOCS)] {
            sliced.extend(
                r.search_or_range_pretokenized("body", &terms, N_DOCS as usize, lo, hi)
                    .await
                    .expect("sliced"),
            );
        }

        let key = |v: &mut Vec<(RowId, f32)>| {
            v.sort_by_key(|&(d, _)| d);
            v.iter()
                .map(|&(d, s)| (d.get(), s.to_bits()))
                .collect::<Vec<_>>()
        };
        let mut whole_sorted = whole;
        assert_eq!(
            key(&mut sliced),
            key(&mut whole_sorted),
            "the windows must partition the blob exactly once"
        );
    }

    /// Ranked results with pruning live (`k` far below the match count).
    async fn top(r: &FtsReader, terms: &[&str], k: usize) -> Vec<(RowId, f32)> {
        r.search("body", terms, k, BoolMode::Or)
            .await
            .expect("search")
    }

    #[tokio::test]
    async fn null_rows_change_neither_idf_nor_the_average() {
        // The same documents with and without null rows interleaved must
        // score identically: the collection size idf uses and the
        // average length both count documents that carry tokens. Doc
        // ids differ (nulls occupy rows), so compare by score sequence.
        let json = tied_corpus(false).1;
        let dense = FtsReader::open(tied_corpus(false).0, json).expect("open");
        let sparse = FtsReader::open(tied_corpus(true).0, json).expect("open");
        assert_eq!(
            sparse.columns[0].scored_doc_count(),
            dense.columns[0].scored_doc_count()
        );
        assert_eq!(sparse.columns[0].avgdl(), dense.columns[0].avgdl());
        for terms in [
            &["common"][..],
            &["common", "shared"][..],
            &["gamma", "beta"][..],
        ] {
            let a: Vec<(RowId, f32)> = top(&dense, terms, 20).await;
            let b: Vec<(RowId, f32)> = top(&sparse, terms, 20).await;
            let a_docs: Vec<u32> = a.iter().map(|(d, _)| d.get()).collect();
            let b_docs: Vec<u32> = b.iter().map(|(d, _)| d.get() / 2).collect();
            assert_eq!(
                a_docs, b_docs,
                "{terms:?}: same documents in the same order"
            );
            let a_scores: Vec<f32> = a.iter().map(|(_, s)| *s).collect();
            let b_scores: Vec<f32> = b.iter().map(|(_, s)| *s).collect();
            assert_eq!(a_scores, b_scores, "{terms:?}: same scores");
        }
    }

    #[test]
    fn a_chopped_run_counts_every_piece_toward_the_document_length() {
        // A 511-character run is three tokens, not one: the length that
        // enters normalization is the truthful one, and the run is no
        // longer the "shortest document in the corpus".
        let run = "a".repeat(511);
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, &run).expect("doc 0");
        b.add_doc(0, 1, &format!("{run} b")).expect("doc 1");
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(r.read_doc_lengths(0).expect("doc lengths"), vec![3, 4]);
    }

    #[test]
    fn open_accepts_valid_blob() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open should succeed");
        assert_eq!(r.n_docs(), 3);
        assert!(r.n_terms() > 0);
        assert_eq!(r.fts_columns().collect::<Vec<_>>(), vec!["body"]);
    }

    #[test]
    fn for_each_term_posting_round_trips_doc_ids_and_tfs() {
        use std::collections::BTreeMap;
        // Docs (from build_blob): 0 "rust async runtime", 1 "tokio is a rust
        // runtime", 2 "java spring boot".
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");

        let mut got: BTreeMap<Vec<u8>, Vec<(u32, u32)>> = BTreeMap::new();
        r.for_each_term_posting(0, |term, doc_id, tf, positions| {
            assert!(
                positions.is_empty(),
                "non-positional column yields no positions"
            );
            got.entry(term.to_vec()).or_default().push((doc_id, tf));
            Ok(())
        })
        .expect("stream postings");

        // doc_ids ascending within each term's list.
        for postings in got.values() {
            assert!(
                postings.windows(2).all(|w| w[0].0 < w[1].0),
                "doc_ids must be ascending"
            );
        }
        let t = |s: &str| s.as_bytes().to_vec();
        assert_eq!(
            got.get(&t("rust")).expect("term streamed").as_slice(),
            &[(0, 1), (1, 1)]
        );
        assert_eq!(
            got.get(&t("runtime")).expect("term streamed").as_slice(),
            &[(0, 1), (1, 1)]
        );
        assert_eq!(
            got.get(&t("async")).expect("term streamed").as_slice(),
            &[(0, 1)]
        );
        assert_eq!(
            got.get(&t("tokio")).expect("term streamed").as_slice(),
            &[(1, 1)]
        );
        assert_eq!(
            got.get(&t("java")).expect("term streamed").as_slice(),
            &[(2, 1)]
        );
        assert_eq!(
            got.get(&t("boot")).expect("term streamed").as_slice(),
            &[(2, 1)]
        );
        // Every stored term was streamed exactly once.
        assert_eq!(got.len() as u32, r.n_terms());
    }

    #[test]
    fn for_each_term_posting_round_trips_positions() {
        use std::collections::BTreeMap;
        // doc 0 "a b a", doc 1 "b a c". "a"/"b" are df=2 (PFOR path); "c" is
        // df=1 (inline path). Positions are token offsets within each doc.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true)
            .expect("register positional column");
        b.add_doc(0, 0, "a b a").expect("add doc 0");
        b.add_doc(0, 1, "b a c").expect("add doc 1");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let r = FtsReader::open(Bytes::from(bytes), json).expect("open");

        let mut got: BTreeMap<Vec<u8>, Vec<(u32, u32, Vec<u32>)>> = BTreeMap::new();
        r.for_each_term_posting(0, |term, doc_id, tf, positions| {
            got.entry(term.to_vec())
                .or_default()
                .push((doc_id, tf, positions.to_vec()));
            Ok(())
        })
        .expect("stream positional postings");

        let t = |s: &str| s.as_bytes().to_vec();
        // PFOR positional: multi-doc terms, tf and positions per doc.
        assert_eq!(
            got.get(&t("a")).expect("term streamed").as_slice(),
            &[(0, 2, vec![0, 2]), (1, 1, vec![1])]
        );
        assert_eq!(
            got.get(&t("b")).expect("term streamed").as_slice(),
            &[(0, 1, vec![1]), (1, 1, vec![0])]
        );
        // Inline positional (df=1): the single position comes from the slot.
        assert_eq!(
            got.get(&t("c")).expect("term streamed").as_slice(),
            &[(1, 1, vec![2])]
        );
    }

    /// The doc-only walk visits exactly the `(term, doc)` pairs the full
    /// walk does, in the same order, over inline, short and long terms
    /// (long ones dense enough for bitset blocks) on a positional and a
    /// plain column; and a term it is told to skip contributes no doc.
    #[test]
    fn for_each_term_doc_matches_the_full_walk_and_skips_terms() {
        const N_DOCS: u32 = 700;
        let mut b = FtsBuilder::new();
        b.register_column("pos".into(), true).expect("register");
        b.register_column("flat".into(), false).expect("register");
        for i in 0..N_DOCS {
            // `every` is long and dense, `some` long and sparse, `few`
            // short, `u{i}` inline.
            let mut text = format!("every every u{i}");
            if i.is_multiple_of(3) {
                text.push_str(" some");
            }
            if i.is_multiple_of(50) {
                text.push_str(" few");
            }
            b.add_doc(0, i, &text).expect("add pos");
            b.add_doc(1, i, &text).expect("add flat");
        }
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"pos","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true},{"name":"flat","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(bytes), json).expect("open");
        for column_id in 0..2 {
            let mut full: Vec<(Vec<u8>, u32)> = Vec::new();
            r.for_each_term_posting(column_id, |term, doc, _, _| {
                full.push((term.to_vec(), doc));
                Ok(())
            })
            .expect("full walk");
            let mut docs_only: Vec<(Vec<u8>, u32)> = Vec::new();
            r.for_each_term_doc(
                column_id,
                |term| Some(term.to_vec()),
                |term, doc| {
                    docs_only.push((term.clone(), doc));
                },
            )
            .expect("doc-only walk");
            assert_eq!(
                docs_only, full,
                "column {column_id}: same pairs, same order"
            );

            let skipped = b"some".as_slice();
            let mut kept: Vec<(Vec<u8>, u32)> = Vec::new();
            r.for_each_term_doc(
                column_id,
                |term| (term != skipped).then(|| term.to_vec()),
                |term, doc| kept.push((term.clone(), doc)),
            )
            .expect("walk with a skip");
            let expected: Vec<(Vec<u8>, u32)> =
                full.into_iter().filter(|(t, _)| t != skipped).collect();
            assert_eq!(
                kept, expected,
                "column {column_id}: a skipped term has no docs"
            );
        }
    }

    #[test]
    fn add_prebuilt_term_posting_round_trips_read_to_write() {
        use std::collections::BTreeMap;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;

        // Build A from text.
        let mut a = FtsBuilder::new();
        a.register_column("body".into(), true).expect("register a");
        a.add_doc(0, 0, "a b a").expect("a doc 0");
        a.add_doc(0, 1, "b a c").expect("a doc 1");
        let ra = FtsReader::open(Bytes::from(a.finish().expect("finish a")), json).expect("open a");

        // Build B by streaming A's postings straight into the prebuilt path —
        // no re-tokenization. Doc lengths carried over ("a b a"/"b a c" = 3/3).
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register b");
        ra.for_each_term_posting(0, |term, doc_id, tf, positions| {
            let term_str = std::str::from_utf8(term).expect("utf8 term");
            b.add_prebuilt_term_posting(0, term_str, FtsDocId::new(doc_id), tf, positions)
                .expect("prebuilt push");
            Ok(())
        })
        .expect("feed prebuilt postings");
        b.append_prebuilt_doc_lengths(0, &ra.read_doc_lengths(0).expect("doc lengths"));
        let rb = FtsReader::open(Bytes::from(b.finish().expect("finish b")), json).expect("open b");

        // The two readers must expose identical postings (doc_ids, tfs,
        // positions) for every term.
        let collect = |r: &FtsReader| {
            let mut m: BTreeMap<Vec<u8>, Vec<(u32, u32, Vec<u32>)>> = BTreeMap::new();
            r.for_each_term_posting(0, |t, d, tf, p| {
                m.entry(t.to_vec()).or_default().push((d, tf, p.to_vec()));
                Ok(())
            })
            .expect("collect");
            m
        };
        assert_eq!(
            collect(&ra),
            collect(&rb),
            "prebuilt-fed postings must match"
        );
        assert_eq!(rb.n_docs(), 2);
        assert_eq!(rb.n_terms(), ra.n_terms());
    }

    #[test]
    fn add_prebuilt_term_posting_spilled_round_trips() {
        use std::collections::BTreeMap;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;

        let mut a = FtsBuilder::new();
        a.register_column("body".into(), true).expect("register a");
        a.add_doc(0, 0, "a b c a").expect("a doc 0");
        a.add_doc(0, 1, "b c d").expect("a doc 1");
        a.add_doc(0, 2, "a d e").expect("a doc 2");
        let ra = FtsReader::open(Bytes::from(a.finish().expect("finish a")), json).expect("open a");

        // Force the spilled accumulator with a 1-byte threshold: the first
        // prebuilt push spills the column, so the rest exercise the transition
        // + push_prebuilt_spilled (partition + position-blob writes).
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register b");
        b.set_spill_threshold_bytes(1);
        ra.for_each_term_posting(0, |term, doc_id, tf, positions| {
            let term_str = std::str::from_utf8(term).expect("utf8 term");
            b.add_prebuilt_term_posting(0, term_str, FtsDocId::new(doc_id), tf, positions)
                .expect("prebuilt push (spilled)");
            Ok(())
        })
        .expect("feed prebuilt postings");
        b.append_prebuilt_doc_lengths(0, &ra.read_doc_lengths(0).expect("doc lengths"));
        let rb = FtsReader::open(Bytes::from(b.finish().expect("finish b")), json).expect("open b");

        let collect = |r: &FtsReader| {
            let mut m: BTreeMap<Vec<u8>, Vec<(u32, u32, Vec<u32>)>> = BTreeMap::new();
            r.for_each_term_posting(0, |t, d, tf, p| {
                m.entry(t.to_vec()).or_default().push((d, tf, p.to_vec()));
                Ok(())
            })
            .expect("collect");
            m
        };
        assert_eq!(
            collect(&ra),
            collect(&rb),
            "spilled prebuilt-fed postings must match a fresh build"
        );
        assert_eq!(rb.n_docs(), 3);
        assert_eq!(rb.n_terms(), ra.n_terms());
    }

    /// A `V7` blob stores each length in two bytes, saturating: a document
    /// past 65,535 tokens reads back at the cap, the exact total still
    /// feeds the average, and the doc is scored and found like any other
    /// (its stored bound and its bucket both come from the saturated
    /// length, so the block skip cannot drop it).
    #[tokio::test]
    async fn doc_lengths_store_in_two_bytes_and_saturate() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        let long = "tok ".repeat(70_000);
        b.add_doc(0, 0, &long).expect("long doc");
        b.add_doc(0, 1, "tok tok tok").expect("short doc");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(
            r.read_doc_lengths(0).expect("doc lengths"),
            vec![format::fts::DOC_LENGTH_STORED_MAX, 3]
        );
        assert_eq!(
            r.columns[0].doc_lengths_range.len(),
            2 * format::fts::DOC_LENGTH_BYTES
        );
        let hits = r
            .search("body", &["tok"], 10, BoolMode::Or)
            .await
            .expect("search");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn read_doc_lengths_returns_token_counts() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b a").expect("doc 0"); // 3 tokens
        b.add_doc(0, 1, "b a c d").expect("doc 1"); // 4 tokens
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(r.read_doc_lengths(0).expect("doc lengths"), vec![3, 4]);
    }

    #[test]
    fn declared_bm25_params_round_trip_through_the_column_entry() {
        // A column whose entry records a non-standard pair: the reader
        // must score with what the file says, never with the crate
        // default, because the stored bounds were baked at that pair.
        let mut b = FtsBuilder::new();
        b.register_column_with_tokenizer(
            "body".into(),
            false,
            Arc::new(StandardTokenizer),
            bm25::Bm25Params::new(1.4, 0.6),
        )
        .expect("register");
        b.add_doc(0, 0, "a b a").expect("doc 0");
        b.add_doc(0, 1, "b a c d").expect("doc 1");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.4,"b":0.6}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(r.columns[0].params, bm25::Bm25Params::new(1.4, 0.6));
        assert_ne!(r.columns[0].params, bm25::Bm25Params::STANDARD);
    }

    #[test]
    fn norm_table_is_built_from_the_declared_pair() {
        // The per-doc buckets are lengths (parameter-free); the 256-entry
        // decode table is where the parameters live. Two readers over the
        // same corpus at different pairs must therefore disagree on the
        // norm and agree on nothing else being different.
        let build = |params: bm25::Bm25Params| {
            let mut b = FtsBuilder::new();
            b.register_column_with_tokenizer(
                "body".into(),
                false,
                Arc::new(StandardTokenizer),
                params,
            )
            .expect("register");
            b.add_doc(0, 0, "a b a").expect("doc 0");
            b.add_doc(0, 1, "b a c d e f g h").expect("doc 1");
            b.finish().expect("finish")
        };

        let std_json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let alt_json = r#"[{"name":"body","tokenizer":"standard","k1":1.4,"b":0.6}]"#;
        let std_reader = FtsReader::open(Bytes::from(build(bm25::Bm25Params::STANDARD)), std_json)
            .expect("open standard");
        let alt_reader = FtsReader::open(
            Bytes::from(build(bm25::Bm25Params::new(1.4, 0.6))),
            alt_json,
        )
        .expect("open alt");

        let std_norm = std_reader.columns[0].dl_norm_k1().get(1);
        let alt_norm = alt_reader.columns[0].dl_norm_k1().get(1);
        assert_ne!(
            std_norm, alt_norm,
            "the decode table must reflect the declared pair"
        );
        // And each matches the closed form for its own pair, against the
        // quantized length the scorer actually sees.
        let dl = bm25::stored_len(8);
        let avgdl = std_reader.columns[0].avgdl();
        assert!((std_norm - bm25::Bm25Params::STANDARD.dl_norm_k1(dl, avgdl)).abs() < 1e-6);
        assert!((alt_norm - bm25::Bm25Params::new(1.4, 0.6).dl_norm_k1(dl, avgdl)).abs() < 1e-6);
    }

    #[test]
    fn bound_scale_is_one_for_the_same_pair_and_above_one_otherwise() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for d in 0..64u32 {
            let words = (d % 30) + 1;
            let text: String = (0..words).map(|w| format!("t{w} ")).collect();
            b.add_doc(0, d, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let col = &r.columns[0];
        let baked = col.params;

        assert_eq!(
            col.dl_norm_k1().bound_scale(col.dl_norm_k1(), baked, baked),
            1.0,
            "no correction when the query scores at the baked pair"
        );

        for (k1, b_param) in [(1.4_f32, 0.75_f32), (0.9, 0.75), (1.2, 0.4), (1.2, 0.0)] {
            let query = bm25::Bm25Params::new(k1, b_param);
            let other = col.dl_norm_k1().rescored(col.avgdl(), query);
            let r_factor = col.dl_norm_k1().bound_scale(&other, baked, query);
            assert!(
                r_factor >= 1.0,
                "the factor must never shrink a bound: {k1}/{b_param} gave {r_factor}"
            );
        }
    }

    #[test]
    fn bound_scale_keeps_the_bound_above_every_rescored_score() {
        // The property the whole correction rests on: for any (tf, dl) in
        // the column, R * score_at(baked) >= score_at(query). Checked
        // against the closed form over the corpus's own length range
        // rather than a hand-picked case.
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        for d in 0..128u32 {
            let words = (d % 50) + 1;
            let text: String = (0..words).map(|w| format!("t{w} ")).collect();
            b.add_doc(0, d, text.trim()).expect("add doc");
        }
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let col = &r.columns[0];
        let baked = col.params;
        let avgdl = col.avgdl();

        for (k1, b_param) in [
            (1.4_f32, 0.75_f32),
            (0.5, 0.75),
            (2.0, 0.75),
            (1.2, 0.0),
            (1.2, 1.0),
            (0.5, 0.2),
            (2.0, 1.0),
        ] {
            let query = bm25::Bm25Params::new(k1, b_param);
            let other = col.dl_norm_k1().rescored(avgdl, query);
            let r_factor = col.dl_norm_k1().bound_scale(&other, baked, query);
            for dl in 1..=50u32 {
                for tf in [1u32, 2, 3, 7, 20, 100] {
                    let idf = 1.0_f32;
                    let at_baked = bm25::score(idf, tf, bm25::stored_len(dl), avgdl, baked);
                    let at_query = bm25::score(idf, tf, bm25::stored_len(dl), avgdl, query);
                    assert!(
                        r_factor * at_baked >= at_query - 1e-6,
                        "R={r_factor} too small for k1={k1} b={b_param} tf={tf} dl={dl}: \
                         {} vs {}",
                        r_factor * at_baked,
                        at_query
                    );
                }
            }
        }
    }

    /// The end-to-end guarantee behind the correction factor: scoring
    /// with an override must produce exactly what a file *built* at
    /// those parameters produces. Scores never depend on which pair the
    /// bounds were baked at — only pruning does — so any divergence
    /// means a bound was left too small and a qualifying document was
    /// skipped.
    #[tokio::test]
    async fn an_override_scores_identically_to_a_file_baked_at_that_pair() {
        let corpus: Vec<String> = (0..400u32)
            .map(|d| {
                let words = (d % 37) + 1;
                let mut text: String = (0..words).map(|w| format!("t{} ", w % 11)).collect();
                if d % 3 == 0 {
                    text.push_str("common ");
                }
                if d % 29 == 0 {
                    text.push_str("rare ");
                }
                text.trim().to_string()
            })
            .collect();

        let build = |params: bm25::Bm25Params| {
            let mut b = FtsBuilder::new();
            b.register_column_with_tokenizer(
                "body".into(),
                false,
                Arc::new(StandardTokenizer),
                params,
            )
            .expect("register");
            for (i, t) in corpus.iter().enumerate() {
                b.add_doc(0, i as u32, t).expect("add doc");
            }
            Bytes::from(b.finish().expect("finish"))
        };

        for (k1, b_param) in [(1.4_f32, 0.6_f32), (0.5, 0.9), (2.0, 0.2), (1.2, 0.0)] {
            let params = bm25::Bm25Params::new(k1, b_param);
            let json_std = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let json_alt = format!(
                r#"[{{"name":"body","tokenizer":"standard","k1":{k1:?},"b":{b_param:?}}}]"#
            );

            let baked_standard =
                FtsReader::open(build(bm25::Bm25Params::STANDARD), json_std).expect("open std");
            let baked_at_params = FtsReader::open(build(params), &json_alt).expect("open alt");
            // Same query parameters, reached two ways: an override on a
            // file baked at the standard pair, and a file baked at the
            // pair itself.
            let overridden = baked_standard.with_bm25_override(params);

            for terms in [
                &["common"][..],
                &["rare"][..],
                &["common", "rare"][..],
                &["t0", "t1", "common"][..],
            ] {
                for k in [1usize, 10, 100] {
                    let a = overridden
                        .search("body", terms, k, BoolMode::Or)
                        .await
                        .expect("override search");
                    let b = baked_at_params
                        .search("body", terms, k, BoolMode::Or)
                        .await
                        .expect("baked search");
                    assert_eq!(
                        a.len(),
                        b.len(),
                        "hit count diverged for {terms:?} k={k} at k1={k1} b={b_param}"
                    );
                    for ((da, sa), (db, sb)) in a.iter().zip(b.iter()) {
                        assert_eq!(
                            da, db,
                            "doc order diverged for {terms:?} k={k} at k1={k1} b={b_param}"
                        );
                        assert!(
                            (sa - sb).abs() < 1e-4,
                            "score diverged for doc {da} on {terms:?} at k1={k1} b={b_param}: \
                             {sa} vs {sb}"
                        );
                    }
                }
            }
        }
    }

    /// An override equal to what the column already declares is a no-op:
    /// no rescaled table, no correction factor.
    #[test]
    fn an_override_matching_the_declared_pair_changes_nothing() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b a").expect("doc 0");
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        let same = r.with_bm25_override(bm25::Bm25Params::STANDARD);
        assert_eq!(same.columns[0].bound_scale(), 1.0);
        assert_eq!(same.columns[0].params, r.columns[0].params);
    }

    #[test]
    fn open_rejects_bad_magic() {
        let (mut blob_vec, json) = build_blob();
        let mut bytes = blob_vec.to_vec();
        bytes[0] = b'X';
        blob_vec = Bytes::from(bytes);
        let err = FtsReader::open(blob_vec, &json).expect_err("expected error");
        assert!(matches!(err, FtsError::Read(ReadError::BadMagic { .. })));
    }

    #[test]
    fn open_rejects_short_blob() {
        let err = FtsReader::open(Bytes::from(vec![0u8; 8]), "[]").expect_err("expected error");
        assert!(matches!(err, FtsError::Read(_)));
    }

    #[test]
    fn open_rejects_columns_json_mismatch() {
        let (blob, _) = build_blob();
        // Header says n_columns=1; pass a 2-column JSON.
        let bad_json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75},{"name":"title","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let err = FtsReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(err, FtsError::Read(ReadError::Malformed(_))));
    }

    #[test]
    fn df1_terms_take_the_inline_form_and_others_do_not() {
        // A df=1 term (tf 1) is an inline dictionary entry; df ≥ 2 terms
        // are postings-form entries (short here — they fit one block).
        let (blob, json) = build_mixed_df_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let dict_bytes = r.dict_bytes().expect("dict");
        let dict = FtsReader::open_dict(&dict_bytes).expect("open dict");

        for term in ["common", "rust"] {
            match dict.lookup(&make_key("body", term)).expect("in dict") {
                DictEntry::Pfor { short, .. } => assert!(short, "{term}: df ≥ 2 in one block"),
                DictEntry::Inline { .. } => panic!("{term}: df ≥ 2 must not inline"),
            }
        }
        for (term, doc_id) in [("uniqzero", 0u32), ("uniqtwo", 2u32)] {
            match dict.lookup(&make_key("body", term)).expect("in dict") {
                DictEntry::Inline { doc_id: d, tf } => {
                    assert_eq!(d, doc_id);
                    assert_eq!(tf, 1);
                }
                DictEntry::Pfor { .. } => panic!("{term}: df=1 must inline"),
            }
        }
    }

    #[test]
    fn df1_inline_path_skips_postings_region_writes() {
        // A blob with only df=1 terms should produce a much smaller
        // postings region than a blob with the same term count but
        // df ≥ 2 — the inline form writes nothing for df=1.

        let mut b_inline = FtsBuilder::new();
        b_inline
            .register_column("body".into(), false)
            .expect("register column");
        for i in 0..20 {
            b_inline
                .add_doc(0, i, &format!("uniq{i:03}"))
                .expect("add doc");
        }
        let blob_inline = b_inline.finish().expect("finish inline");

        let mut b_pfor = FtsBuilder::new();
        b_pfor
            .register_column("body".into(), false)
            .expect("register column");
        // Same 20 terms but all appearing in every doc → df = 20 → PFOR.
        for i in 0..20 {
            let text = (0..20)
                .map(|j| format!("uniq{j:03}"))
                .collect::<Vec<_>>()
                .join(" ");
            b_pfor.add_doc(0, i, &text).expect("add doc");
        }
        let blob_pfor = b_pfor.finish().expect("finish pfor");

        // Extract postings-region sizes from the headers.
        let postings_off_i = u64::from_le_bytes(
            blob_inline[32..40]
                .try_into()
                .expect("postings_off_i slice is 8 bytes"),
        ) as usize;
        // The postings region ends where the positions region begins
        // (header bytes [48..56]).
        let positions_off_i = u64::from_le_bytes(
            blob_inline[48..56]
                .try_into()
                .expect("positions_off_i slice is 8 bytes"),
        ) as usize;
        let postings_size_inline = positions_off_i - postings_off_i;

        let postings_off_p = u64::from_le_bytes(
            blob_pfor[32..40]
                .try_into()
                .expect("postings_off_p slice is 8 bytes"),
        ) as usize;
        let positions_off_p = u64::from_le_bytes(
            blob_pfor[48..56]
                .try_into()
                .expect("positions_off_p slice is 8 bytes"),
        ) as usize;
        let postings_size_pfor = positions_off_p - postings_off_p;

        // Inline-only blob's postings region holds just the trailing
        // CRC32 (4 B). The df=20 blob holds 20 short-form terms (a varint
        // df, a 3-byte tf bitmap, 20 group-varint deltas ≈ 29 B each).
        assert_eq!(
            postings_size_inline, 4,
            "all-df=1 postings region should hold only the trailing CRC32; \
             got {postings_size_inline} bytes"
        );
        assert!(
            postings_size_pfor > 20 * 20,
            "df=20 postings region should be hundreds of bytes; got {postings_size_pfor}"
        );
    }

    #[test]
    fn iter_column_terms_lists_every_term_in_lex_order() {
        // build_blob plants the union of tokens across the 3 docs.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let terms: Vec<String> = r
            .iter_column_terms("body")
            .expect("iter terms")
            .into_iter()
            .map(|b| String::from_utf8(b).expect("utf8"))
            .collect();
        // Dictionary iteration is lex-ordered.
        let mut sorted = terms.clone();
        sorted.sort();
        assert_eq!(terms, sorted, "terms must be in lex order");
        for expected in [
            "rust", "async", "runtime", "tokio", "java", "spring", "boot",
        ] {
            assert!(terms.contains(&expected.to_string()), "missing {expected}");
        }
    }

    #[test]
    fn iter_column_terms_unknown_column_is_empty() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        assert!(r.iter_column_terms("nope").expect("ok").is_empty());
    }

    #[test]
    fn iter_terms_with_prefix_bounds_the_walk() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // "runtime" begins with "run"; nothing else does.
        let terms: Vec<String> = r
            .iter_terms_with_prefix("body", b"run")
            .expect("prefix walk")
            .into_iter()
            .map(|b| String::from_utf8(b).expect("utf8"))
            .collect();
        assert_eq!(terms, vec!["runtime".to_string()]);
        // A prefix that matches nothing returns empty.
        assert!(
            r.iter_terms_with_prefix("body", b"zzz")
                .expect("prefix walk")
                .is_empty()
        );
    }

    /// A scan's top-k heap is preallocated for the docs it can actually
    /// rank: the whole superfile un-ranged, the window's width when
    /// ranged. Sizing a ranged scan by `n_docs` is what made a sliced
    /// fan-out preallocate one whole-superfile heap per slice.
    #[test]
    fn top_k_capacity_is_scoped_to_the_range_the_scan_visits() {
        /// Docs in the notional superfile.
        const N_DOCS: u64 = 1_000_000;
        /// Result size large enough that the scope, not `k`, is the cap.
        const BIG_K: usize = N_DOCS as usize;

        // Un-ranged: scope is the whole superfile.
        assert_eq!(top_k_initial_capacity(BIG_K, N_DOCS, None), N_DOCS as usize);
        // Ranged: scope is the window, not the file — an eighth of the
        // doc space preallocates an eighth of the slots.
        let eighth = (N_DOCS / 8) as u32;
        assert_eq!(
            top_k_initial_capacity(BIG_K, N_DOCS, Some((0, eighth))),
            eighth as usize
        );
        assert_eq!(
            top_k_initial_capacity(BIG_K, N_DOCS, Some((eighth, 2 * eighth))),
            eighth as usize
        );
        // A window wider than the file (un-ranged callers pass
        // `[0, u32::MAX)`) collapses back to the whole-superfile scope.
        assert_eq!(
            top_k_initial_capacity(BIG_K, N_DOCS, Some((0, u32::MAX))),
            N_DOCS as usize
        );
        // Small `k` still wins over the scope, and the floor is 1 slot so
        // a `k = 0` or empty-range caller never asks for a zero-capacity
        // heap.
        assert_eq!(top_k_initial_capacity(10, N_DOCS, Some((0, eighth))), 10);
        assert_eq!(top_k_initial_capacity(0, N_DOCS, None), 1);
        assert_eq!(top_k_initial_capacity(BIG_K, N_DOCS, Some((5, 5))), 1);
        // `k = usize::MAX` (`search_multi`) is capped by the scope, never
        // turned into an unservable allocation.
        assert_eq!(
            top_k_initial_capacity(usize::MAX, N_DOCS, None),
            N_DOCS as usize
        );
    }

    #[test]
    fn read_u32_le_and_u64_le_decode_little_endian() {
        let b32 = [0x78, 0x56, 0x34, 0x12];
        assert_eq!(read_u32_le(&b32), 0x1234_5678);
        let b64 = [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert_eq!(read_u64_le(&b64), 1);
    }

    #[test]
    fn the_count_walk_takes_the_heap_only_when_rescans_outnumber_heap_steps() {
        // 1M docs is 245 windows. 5000 single-doc terms: 5000 rescans of
        // 245 windows against 5000 heap steps of 13 → heap.
        assert!(or_count_prefers_heap(5000, 5000, 1_000_000));
        // Ten terms of 10k postings each: 10 rescans of 245 windows
        // against 100k heap steps of 4 → windowed.
        assert!(!or_count_prefers_heap(10, 100_000, 1_000_000));
        // Below the term threshold the windowed walk is kept however
        // sparse the union.
        assert!(!or_count_prefers_heap(OR_HEAP_MIN_TERMS - 1, 7, 1_000_000));
        // One window: 1000 single-doc terms in 3000 docs are rescanned
        // once, against 1000 heap steps of 10 → windowed.
        assert!(!or_count_prefers_heap(1000, 1000, 3_000));
    }

    #[test]
    fn the_match_walk_takes_the_heap_only_for_a_sparse_union() {
        // The unions measured through `token_match` on 1M docs, each routed
        // to the walk that ran it faster. Eight common terms in ~30% of the
        // docs each: 2.4M postings × 4 heap steps against 8 rescans of the
        // 1M docs → rescan; ten likewise.
        assert!(!or_walk_prefers_heap(8, 2_400_000, 1_000_000));
        assert!(!or_walk_prefers_heap(10, 3_000_000, 1_000_000));
        // Forty mid terms in ~1% each: 400k postings × 6 against 40
        // rescans of the 400k postings → heap; 5000 single-doc ids too.
        assert!(or_walk_prefers_heap(40, 400_000, 1_000_000));
        assert!(or_walk_prefers_heap(5000, 5000, 1_000_000));
        // Below the term threshold the rescan is kept however sparse.
        assert!(!or_walk_prefers_heap(OR_HEAP_MIN_TERMS - 1, 7, 1_000_000));
    }

    #[test]
    fn dominant_anchor_routes_at_the_dominance_boundary() {
        // The union count anchors on a term only when its df is at least
        // OR_COUNT_ANCHOR_DOMINANCE× the others' combined df. Pin the
        // routing at the boundary (values chosen against the 8× default).
        let k = OR_COUNT_ANCHOR_DOMINANCE;
        // Exactly N× the rest → dominates (the boundary is inclusive).
        assert_eq!(dominant_anchor_of_dfs([100 * k, 100]), Some(0));
        // One below the boundary → no anchor, fall back to windowed.
        assert_eq!(dominant_anchor_of_dfs([100 * k - 1, 100]), None);
        // The anchor is the max-df term wherever it sits.
        assert_eq!(dominant_anchor_of_dfs([100, 100 * k]), Some(1));
        // Single term → no anchor (its count is just its df).
        assert_eq!(dominant_anchor_of_dfs([100 * k]), None);
        // Two-way tie → neither dominates.
        assert_eq!(dominant_anchor_of_dfs([500, 500]), None);
        // Dominance is measured against the *sum* of the others, not the
        // largest single other: 8×(100+100)=1600 > 1500, so no anchor.
        assert_eq!(dominant_anchor_of_dfs([1500, 100, 100]), None);
        // …and just clears when the sum is small enough.
        assert_eq!(dominant_anchor_of_dfs([100 * k, 60, 40]), Some(0));
        // Empty → no anchor.
        assert_eq!(dominant_anchor_of_dfs([0u64; 0]), None);
    }

    #[test]
    fn top_k_keeps_highest_scores_with_doc_id_tiebreak() {
        let mut scores: FxHashMap<RowId, f32> = FxHashMap::default();
        scores.insert(RowId::new(0), 1.0);
        scores.insert(RowId::new(1), 3.0);
        scores.insert(RowId::new(2), 2.0);
        scores.insert(RowId::new(3), 3.0); // tie with row 1 on score 3.0
        let out = top_k(scores, 2);
        // Descending score; ties broken by ascending row ⇒ row 1 before 3.
        assert_eq!(out, vec![(RowId::new(1), 3.0), (RowId::new(3), 3.0)]);
    }

    #[test]
    fn top_k_smaller_than_k_returns_all_sorted() {
        let mut scores: FxHashMap<RowId, f32> = FxHashMap::default();
        scores.insert(RowId::new(5), 2.0);
        scores.insert(RowId::new(9), 5.0);
        let out = top_k(scores, 10);
        assert_eq!(out, vec![(RowId::new(9), 5.0), (RowId::new(5), 2.0)]);
    }

    /// A lazy source that records every range it serves.
    #[derive(Debug)]
    struct RecordingSource {
        inner: BytesLazyByteSource,
        ranges: Mutex<Vec<(u64, u64)>>,
    }
    /// The bytes a lazy reader fetches to score `column`: the bucket array a
    /// blob that stores one, else the length array it derives the buckets
    /// from. What a test that watches the norms read has to watch.
    fn norms_region(column: &ColumnMeta) -> Range<usize> {
        match &column.stored_norms {
            Some(stored) => stored.buckets.clone(),
            None => column.doc_lengths_range.clone(),
        }
    }

    impl RecordingSource {
        fn note(&self, start: u64, len: u64) {
            self.ranges.lock().expect("ranges").push((start, len));
        }
        fn touches(&self, since: usize, region: &Range<usize>) -> usize {
            self.ranges.lock().expect("ranges")[since..]
                .iter()
                .filter(|&&(s, l)| (s as usize) < region.end && (s + l) as usize > region.start)
                .count()
        }
        fn len(&self) -> usize {
            self.ranges.lock().expect("ranges").len()
        }
    }
    #[async_trait]
    impl LazyByteSource for RecordingSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }
        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            self.note(start, len);
            self.inner.range(start, len).await
        }
        fn try_get_range_sync(&self, start: u64, len: u64) -> Option<Bytes> {
            self.note(start, len);
            self.inner.try_get_range_sync(start, len)
        }
    }

    /// A lazy open reads the header and the doc-lengths directory and
    /// nothing else. A match-only query served from a memo of already
    /// resolved terms then reads postings alone — neither the dictionary
    /// nor the length array — and the first scored query reads the
    /// dictionary (there is no memo) and the length array exactly once;
    /// a second scored query reads neither again.
    #[tokio::test]
    async fn open_lazy_reads_header_and_directory_only_until_a_query_needs_more() {
        let (blob, json) = build_blob();
        // An eager twin resolves the term's dictionary entry for the memo,
        // the way a table-level term index would hand it to a reader.
        let eager = FtsReader::open(blob.clone(), &json).expect("eager open");
        let facts = eager
            .term_index_facts("body", &["rust"])
            .await
            .expect("facts");
        let fact = facts[0].clone().expect("`rust` is indexed");

        let recording = Arc::new(RecordingSource {
            inner: BytesLazyByteSource::new(blob),
            ranges: Mutex::new(Vec::new()),
        });
        let src: Arc<dyn LazyByteSource> = recording.clone();
        let r = FtsReader::open_lazy(src, &json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        let dictionary = r.dict_range.clone();
        let lengths = norms_region(&r.columns[0]);
        let opened = recording.len();
        assert_eq!(
            recording.touches(0, &dictionary),
            0,
            "open reads no dictionary: {:?}",
            recording.ranges.lock().expect("ranges")
        );
        assert_eq!(
            recording.touches(0, &lengths),
            0,
            "open reads no length array"
        );

        let memo = r
            .memo_from_dict_values(&[("rust", fact.df, fact.entry)])
            .await
            .expect("memo");
        let (docs, _) = r
            .token_match_prefetched("body", &["rust"], BoolMode::Or, Some(&memo))
            .await
            .expect("match from the memo");
        assert!(!docs.is_empty());
        assert_eq!(
            recording.touches(opened, &dictionary),
            0,
            "a memo-served match reads no dictionary"
        );
        assert_eq!(
            recording.touches(opened, &lengths),
            0,
            "a match-only query reads no length array"
        );

        let before_scoring = recording.len();
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("scored search");
        assert!(!hits.is_empty());
        assert_eq!(
            recording.touches(before_scoring, &dictionary),
            1,
            "the dictionary, once"
        );
        assert_eq!(
            recording.touches(before_scoring, &lengths),
            1,
            "the length array, once"
        );

        let before_second = recording.len();
        r.search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("second scored search");
        assert_eq!(
            recording.touches(before_second, &lengths),
            0,
            "the norms are resident"
        );
    }

    /// The term-index pass reads the dictionary once per superfile: the
    /// caller's fetch serves every facts batch, none of which fetch it again.
    #[tokio::test]
    async fn term_index_pass_reads_the_dictionary_once() {
        // One term per batch, so every term is its own batch.
        const BATCH: usize = 1;
        let (blob, json) = build_mixed_df_blob();
        let recording = Arc::new(RecordingSource {
            inner: BytesLazyByteSource::new(blob),
            ranges: Mutex::new(Vec::new()),
        });
        let src: Arc<dyn LazyByteSource> = recording.clone();
        let r = FtsReader::open_lazy(src, &json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        let dictionary = r.dict_range.clone();
        let opened = recording.len();
        let dict_bytes = r.dict_bytes_async().await.expect("dict");

        let mut batches = 0;
        let mut after: Option<Vec<u8>> = None;
        loop {
            batches += 1;
            let chunk = r
                .term_index_facts_after(&dict_bytes, "body", after.as_deref(), BATCH, 0)
                .await
                .expect("facts batch");
            let done = chunk.len() < BATCH;
            after = chunk.into_iter().last().map(|(term, _)| term);
            if done {
                break;
            }
        }
        assert!(batches > 1, "the pass spans several batches");
        assert_eq!(
            recording.touches(opened, &dictionary),
            1,
            "only the caller's fetch reads the dictionary"
        );
    }

    /// A source that only serves bytes asynchronously, the way an
    /// object store or a cold disk cache does.
    ///
    /// `BytesLazyByteSource` answers a synchronous read from memory, so
    /// a region left out of the open-time prefetch is still served and
    /// the omission never shows. This one refuses, which is what turns
    /// "the prefetch is missing a region the open needs" into a failure
    /// rather than a slower path.
    struct AsyncOnlySource(Bytes);

    #[async_trait]
    impl LazyByteSource for AsyncOnlySource {
        fn size(&self) -> u64 {
            self.0.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let s = start as usize;
            let e = s + len as usize;
            match e <= self.0.len() {
                true => Ok(self.0.slice(s..e)),
                false => Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: self.0.len() as u64,
                }),
            }
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// The doc-id map's length trailer is the one field between the map and
    /// the directory that no CRC covers, so it is held to the document
    /// count: a map is one id per document plus its CRC, or there is none.
    /// Both ways a corrupt trailer could lie are refused, eager and lazy,
    /// with CRC checks on and off: a length past the blob, which would
    /// overflow the arithmetic placing the map, and one a few bytes short,
    /// which would open and shift every row the map resolves by one.
    #[tokio::test]
    async fn a_doc_map_length_that_disagrees_with_the_document_count_is_refused() {
        const N_DOCS: u32 = 64;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        let map: Vec<u32> = (0..N_DOCS).rev().collect();
        for (id, &row) in map.iter().enumerate() {
            b.add_doc(0, id as u32, &format!("alpha uq{row}"))
                .expect("add doc");
        }
        b.doc_map = Some(map);
        let blob = Bytes::from(b.finish().expect("finish"));
        assert!(
            format::fts::has_stored_buckets(read_u32_le(&blob[VERSION_FIELD])),
            "fixture must store its map length"
        );

        let dir_off =
            read_u64_le(&blob[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;
        let trailer = dir_off - format::fts::DOC_MAP_LEN_TRAILER_BYTES..dir_off;
        let map_len = N_DOCS as u64 * U32_BYTES as u64 + format::CRC_BYTES as u64;
        assert_eq!(read_u64_le(&blob[trailer.clone()]), map_len);
        let stamp = |len: u64| {
            let mut bytes = blob.to_vec();
            bytes[trailer.clone()].copy_from_slice(&len.to_le_bytes());
            Bytes::from(bytes)
        };

        for (bad, what) in [(u64::MAX, "past the blob"), (map_len - 4, "one id short")] {
            let err = FtsReader::open(stamp(bad), json).expect_err(what);
            assert!(
                matches!(err, FtsError::Read(ReadError::Malformed(_))),
                "{what}, eager: {err:?}"
            );
            for opts in [OpenOptions::default(), OpenOptions::for_object_store()] {
                let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(stamp(bad)));
                let err = FtsReader::open_lazy(src, json, opts).await.expect_err(what);
                assert!(
                    matches!(err, FtsError::Read(ReadError::Malformed(_))),
                    "{what}, lazy: {err:?}"
                );
            }
        }

        // The honest value still opens, and resolves the rows it names.
        let r = FtsReader::open(stamp(map_len), json).expect("the written length");
        assert!(r.has_doc_map());
        let hits = r
            .search("body", &["uq7"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert_eq!(
            hits.iter().map(|(d, _)| d.get()).collect::<Vec<_>>(),
            vec![7]
        );
    }

    /// Opening a blob that stores its documents in an order of its own
    /// must work on a source that cannot be read synchronously.
    ///
    /// The open runs against the prefetch overlay alone, and the map is
    /// not in it: the first query that needs the map reads it, so a search
    /// over an async-only source must still name the right rows.
    #[tokio::test]
    async fn open_lazy_serves_a_reordered_blob_from_an_async_only_source() {
        const N_DOCS: u32 = 400;
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), false).expect("register");
        // Fed in reverse, with the map that puts each document back.
        let map: Vec<u32> = (0..N_DOCS).rev().collect();
        for (id, &row) in map.iter().enumerate() {
            b.add_doc(0, id as u32, &format!("alpha t{} uq{row}", row % 23))
                .expect("add doc");
        }
        b.doc_map = Some(map);
        let blob = Bytes::from(b.finish().expect("finish"));
        assert_eq!(
            read_u32_le(&blob[VERSION_FIELD]),
            format::fts::VERSION_CURRENT,
            "fixture must be a mapped blob"
        );

        let src: Arc<dyn LazyByteSource> = Arc::new(AsyncOnlySource(blob));
        let r = FtsReader::open_lazy(src, json, OpenOptions::for_object_store())
            .await
            .expect("a reordered blob must open from an async-only source");
        assert!(r.has_doc_map());
        assert_eq!(r.n_docs(), N_DOCS);

        // And the rows it names are the rows that hold the term.
        let hits = r
            .search("body", &["uq7"], 10, BoolMode::Or)
            .await
            .expect("search");
        assert_eq!(
            hits.iter().map(|(d, _)| d.get()).collect::<Vec<_>>(),
            vec![7],
            "the map must survive the lazy open"
        );
    }

    /// A source that serves bytes asynchronously only, the way an object
    /// store does, and records which thread each read ran on. Each read
    /// yields once before it returns, so concurrent first readers interleave
    /// as they do over a network. Once a test names the doc-id map's region,
    /// the next `fail_map_reads` reads of it fail.
    struct MapProbeSource {
        blob: Bytes,
        reads: Mutex<Vec<(ThreadId, Range<usize>)>>,
        map: OnceLock<Range<usize>>,
        fail_map_reads: AtomicU32,
    }

    impl MapProbeSource {
        /// The threads of the reads that overlapped the doc-id map.
        fn map_reads(&self) -> Vec<ThreadId> {
            let map = self.map.get().expect("the test names the map's region");
            self.reads
                .lock()
                .expect("reads")
                .iter()
                .filter(|(_, read)| read.start < map.end && read.end > map.start)
                .map(|(thread, _)| *thread)
                .collect()
        }
    }

    #[async_trait]
    impl LazyByteSource for MapProbeSource {
        fn size(&self) -> u64 {
            self.blob.len() as u64
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let read = start as usize..(start + len) as usize;
            yield_now().await;
            self.reads
                .lock()
                .expect("reads")
                .push((thread::current().id(), read.clone()));
            let of_map = self
                .map
                .get()
                .is_some_and(|map| read.start < map.end && read.end > map.start);
            if of_map
                && self
                    .fail_map_reads
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                    .is_ok()
            {
                return Err(LazyByteSourceError::OutOfBounds {
                    start,
                    len,
                    size: 0,
                });
            }
            Ok(self.blob.slice(read))
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// [`probed_reader`]'s documents, row by row.
    const PROBE_DOCS: [&str; 4] = ["alpha one", "beta two", "alpha three", "gamma four"];
    /// Rows of [`PROBE_DOCS`] that hold [`PROBE_TERM`].
    const PROBE_ROWS: [u32; 2] = [0, 2];
    /// The term [`PROBE_DOCS`] hold in [`PROBE_ROWS`].
    const PROBE_TERM: &str = "alpha";
    /// The only row of [`PROBE_DOCS`] holding the phrase "alpha one".
    const PROBE_PHRASE_ROW: u32 = 0;
    /// A term in no document.
    const MISSING_TERM: &str = "zzq";
    /// Documents per term a term-index head keeps in these tests: more than
    /// [`PROBE_TERM`] has, so its head holds all of them.
    const PROBE_HEAD_DEPTH: usize = 8;
    /// Average lengths the scored views in the tests below score at, each
    /// unlike the blob's own.
    const VIEW_AVERAGE_LENGTHS: [f32; 3] = [3.0, 4.0, 5.0];
    /// Flips every bit of the byte a test damages.
    const DAMAGE_MASK: u8 = 0xFF;

    /// A lazy reader over a blob with positions that stores [`PROBE_DOCS`]
    /// in reverse, on a [`MapProbeSource`] that knows where the map lies.
    /// `verify_crc` is the reader's.
    async fn probed_reader(verify_crc: bool) -> (Arc<MapProbeSource>, FtsReader) {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true)
            .expect("register column");
        let order: Vec<u32> = (0..PROBE_DOCS.len() as u32).rev().collect();
        for (position, &row) in order.iter().enumerate() {
            b.add_doc(0, position as u32, PROBE_DOCS[row as usize])
                .expect("add doc");
        }
        b.doc_map = Some(order);
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let source = Arc::new(MapProbeSource {
            blob,
            reads: Mutex::new(Vec::new()),
            map: OnceLock::new(),
            fail_map_reads: AtomicU32::new(0),
        });
        let src: Arc<dyn LazyByteSource> = source.clone();
        let r = FtsReader::open_lazy(src, json, OpenOptions { verify_crc })
            .await
            .expect("open_lazy");
        let map = r
            .doc_map_region
            .clone()
            .expect("a reordered blob has a map");
        source.map.set(map).expect("named once");
        (source, r)
    }

    /// The rows a search for [`PROBE_TERM`] returns, filtered to an allow set
    /// of every row: the filter checks rows, so it needs the map before any
    /// cursor does.
    async fn filtered_search(r: &FtsReader) -> Result<Vec<u32>, FtsError> {
        let allow: Arc<RoaringBitmap> = Arc::new((0..r.n_docs()).collect());
        let hits = r
            .search_excluding(
                "body",
                ClauseLists {
                    shoulds: &[PROBE_TERM],
                    allow: Some(allow),
                    ..Default::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await?;
        let mut rows: Vec<u32> = hits.into_iter().map(|(row, _)| row.get()).collect();
        rows.sort_unstable();
        Ok(rows)
    }

    /// A lazy open reads none of the doc-id map, and a contains that
    /// matches nothing never reads it. One that matches reads it on the
    /// query's own runtime, not through the sync bridge: under a
    /// current-thread runtime the bridge reads on a thread of its own, so
    /// a map read on any other thread than this one went through it.
    #[tokio::test]
    async fn a_lazy_reader_reads_its_doc_map_on_the_query_runtime_and_only_for_a_hit() {
        let (source, r) = probed_reader(false).await;
        assert!(source.map_reads().is_empty(), "the open read the map");

        let (rows, _) = r
            .contains_rows("body", &["zzq"], None, None)
            .await
            .expect("contains");
        assert!(
            rows[0].proven.is_empty() && rows[0].doubtful.is_empty(),
            "no row holds the needle"
        );
        assert!(
            source.map_reads().is_empty(),
            "a contains that matched nothing read the map"
        );

        let (rows, _) = r
            .contains_rows("body", &[PROBE_TERM], None, None)
            .await
            .expect("contains");
        assert_eq!(rows[0].proven, RoaringBitmap::from_iter(PROBE_ROWS));
        let here = thread::current().id();
        assert_eq!(
            source.map_reads(),
            [here],
            "one read of the map, on this runtime and not through the sync bridge"
        );
    }

    /// Every row of `r`'s blob, as an allow set that admits them all.
    fn every_row(r: &FtsReader) -> Arc<RoaringBitmap> {
        Arc::new((0..r.n_docs()).collect())
    }

    /// Rows, ascending.
    fn sorted(rows: impl IntoIterator<Item = u32>) -> Vec<u32> {
        let mut rows: Vec<u32> = rows.into_iter().collect();
        rows.sort_unstable();
        rows
    }

    /// The rows a ranked search for `lists` returns, ascending.
    async fn search_rows(r: &FtsReader, lists: ClauseLists<'_>) -> Vec<u32> {
        let hits = r
            .search_excluding("body", lists, 10, f32::NEG_INFINITY)
            .await
            .expect("search");
        sorted(hits.into_iter().map(|(row, _)| row.get()))
    }

    /// A query with no hit in this file reads none of its map, whatever
    /// path it takes: a missing term, a negated term the file holds, an
    /// allow set with a missing phrase, two terms that never share a
    /// document, an OR cursor set or a term-index head of a missing term.
    #[tokio::test]
    async fn no_query_without_a_hit_reads_the_doc_map() {
        let (source, r) = probed_reader(false).await;
        let allow = every_row(&r);
        let missing_phrase = [Phrase::adjacent(vec![
            MISSING_TERM.to_string(),
            PROBE_TERM.to_string(),
        ])];
        let (rows, _) = r
            .token_match("body", &[MISSING_TERM], BoolMode::Or)
            .await
            .expect("token_match");
        assert!(rows.is_empty(), "token_match");
        let (rows, _) = r
            .atoms_match_ids("body", &[PROBE_TERM, "gamma"], &[], BoolMode::And)
            .await
            .expect("atoms_match_ids");
        assert!(rows.is_empty(), "no row holds both terms");
        let missing = search_rows(
            &r,
            ClauseLists {
                shoulds: &[MISSING_TERM],
                ..Default::default()
            },
        )
        .await;
        assert!(missing.is_empty(), "ranked search");
        let negated = search_rows(
            &r,
            ClauseLists {
                shoulds: &[MISSING_TERM],
                negatives: &[PROBE_TERM],
                ..Default::default()
            },
        )
        .await;
        assert!(negated.is_empty(), "ranked search with a negated term");
        let phrase = search_rows(
            &r,
            ClauseLists {
                should_phrases: &missing_phrase,
                allow: Some(Arc::clone(&allow)),
                ..Default::default()
            },
        )
        .await;
        assert!(phrase.is_empty(), "filtered phrase search");
        let set = r
            .build_or_cursor_set("body", &[MISSING_TERM], QueryIdf::ThisFile, None)
            .await
            .expect("cursor set");
        assert!(set.cursors.is_empty(), "OR cursor set");
        let dict = r.dict_bytes_async().await.expect("dictionary");
        let facts = r
            .term_index_facts_with(&dict, "body", &[MISSING_TERM], PROBE_HEAD_DEPTH)
            .await
            .expect("term facts");
        assert!(facts[0].is_none(), "term facts");
        assert!(
            source.map_reads().is_empty(),
            "a query without a hit read the map"
        );
    }

    /// Each kind of query with a hit reads the map once, on its own runtime
    /// rather than through the sync bridge, and names the right rows.
    #[tokio::test]
    async fn each_query_with_a_hit_reads_the_doc_map_once_on_its_runtime() {
        let here = thread::current().id();
        let hit_phrase = [Phrase::adjacent(vec![
            PROBE_TERM.to_string(),
            "one".to_string(),
        ])];
        for case in [
            "ranked search",
            "filtered phrase search",
            "token_match",
            "atoms_match_ids",
            "OR cursor set",
            "term-index head",
        ] {
            let (source, r) = probed_reader(false).await;
            let rows = match case {
                "ranked search" => {
                    search_rows(
                        &r,
                        ClauseLists {
                            shoulds: &[PROBE_TERM],
                            ..Default::default()
                        },
                    )
                    .await
                }
                "filtered phrase search" => {
                    search_rows(
                        &r,
                        ClauseLists {
                            must_phrases: &hit_phrase,
                            allow: Some(every_row(&r)),
                            ..Default::default()
                        },
                    )
                    .await
                }
                "token_match" => {
                    let (rows, _) = r
                        .token_match("body", &[PROBE_TERM], BoolMode::Or)
                        .await
                        .expect("token_match");
                    sorted(rows.into_iter().map(|row| row.get()))
                }
                "atoms_match_ids" => {
                    let (rows, _) = r
                        .atoms_match_ids("body", &[PROBE_TERM], &[], BoolMode::Or)
                        .await
                        .expect("atoms_match_ids");
                    sorted(rows.into_iter().map(|row| row.get()))
                }
                "OR cursor set" => {
                    let set = r
                        .build_or_cursor_set("body", &[PROBE_TERM], QueryIdf::ThisFile, None)
                        .await
                        .expect("cursor set");
                    let hits = r
                        .search_or_range_prebuilt(&set, 10, 0, r.n_docs(), f32::NEG_INFINITY)
                        .expect("ranged search");
                    sorted(hits.into_iter().map(|(row, _)| row.get()))
                }
                "term-index head" => {
                    let dict = r.dict_bytes_async().await.expect("dictionary");
                    let facts = r
                        .term_index_facts_with(&dict, "body", &[PROBE_TERM], PROBE_HEAD_DEPTH)
                        .await
                        .expect("term facts");
                    let fact = facts[0].as_ref().expect("the term is in the file");
                    sorted(fact.head.docs.iter().map(|&(row, _, _)| row))
                }
                other => unreachable!("no case {other}"),
            };
            let expected = match case {
                "filtered phrase search" => vec![PROBE_PHRASE_ROW],
                _ => PROBE_ROWS.to_vec(),
            };
            assert_eq!(rows, expected, "{case}");
            assert_eq!(source.map_reads(), [here], "{case}");
        }
    }

    /// The blocking read of the map (what a compaction input's remap uses)
    /// returns a failed read as an error and keeps nothing, so the next
    /// read gets the real map, never blob ids taken as rows.
    #[tokio::test]
    async fn a_failed_blocking_doc_map_read_is_an_error_and_not_kept() {
        let (source, r) = probed_reader(true).await;
        source.fail_map_reads.store(1, Ordering::SeqCst);
        assert!(
            matches!(
                r.doc_map(),
                Err(FtsError::RangeFetch {
                    what: "fts/doc-map",
                    ..
                })
            ),
            "a failed map read must be an error"
        );
        assert!(!r.doc_map_is_read(), "a failed read is not kept");
        let map = r.doc_map().expect("the next read");
        let last_row = PROBE_DOCS.len() as u32 - 1;
        assert_eq!(
            map.row_of(FtsDocId::new(0)),
            RowId::new(last_row),
            "blob id 0 holds the last row"
        );
    }

    /// A search filtered to an allow set checks rows before any cursor
    /// reads the map, so the filter reads it first, on the query's runtime.
    #[tokio::test]
    async fn a_filtered_search_reads_the_doc_map_on_the_query_runtime() {
        let (source, r) = probed_reader(true).await;
        assert_eq!(filtered_search(&r).await.expect("search"), PROBE_ROWS);
        assert_eq!(source.map_reads(), [thread::current().id()]);
    }

    /// A failed read of the map fails the query that needed it, and is not
    /// kept: taking blob ids as rows would answer every later query on this
    /// reader with the wrong rows. The next query reads it again.
    #[tokio::test]
    async fn a_failed_doc_map_read_fails_its_query_and_the_next_one_reads_again() {
        let (source, r) = probed_reader(true).await;
        source.fail_map_reads.store(1, Ordering::SeqCst);
        assert!(
            matches!(
                filtered_search(&r).await,
                Err(FtsError::RangeFetch {
                    what: "fts/doc-map",
                    ..
                })
            ),
            "a failed map read must fail the query"
        );
        assert_eq!(filtered_search(&r).await.expect("search"), PROBE_ROWS);
        assert_eq!(source.map_reads().len(), 2, "the second query read again");
    }

    /// A damaged map read lazily is refused, as it is at an in-memory open.
    #[tokio::test]
    async fn a_damaged_doc_map_read_lazily_is_refused() {
        let (source, _) = probed_reader(true).await;
        let map = source.map.get().expect("map").clone();
        let mut damaged = source.blob.to_vec();
        damaged[map.start] ^= DAMAGE_MASK;
        let damaged = Arc::new(MapProbeSource {
            blob: Bytes::from(damaged),
            reads: Mutex::new(Vec::new()),
            map: OnceLock::new(),
            fail_map_reads: AtomicU32::new(0),
        });
        let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75,"positions":true}]"#;
        let src: Arc<dyn LazyByteSource> = damaged;
        let r = FtsReader::open_lazy(src, json, OpenOptions::default())
            .await
            .expect("the open reads none of the map");
        assert!(matches!(
            r.ensure_doc_map().await,
            Err(FtsError::Read(ReadError::ChecksumMismatch {
                section: "fts/doc-map",
                ..
            }))
        ));
        assert!(!r.doc_map_is_read(), "a refused map is not kept");
    }

    /// Scored views of one reader and concurrent first queries share one
    /// read of the map: on a large file it is hundreds of megabytes, and a
    /// table-wide average length moves with every commit, so views come and
    /// go while the reader stays cached.
    #[tokio::test]
    async fn views_and_concurrent_queries_share_one_read_of_the_doc_map() {
        let (source, r) = probed_reader(false).await;
        let views: Vec<FtsReader> = VIEW_AVERAGE_LENGTHS
            .into_iter()
            .map(|avgdl| {
                r.with_scoring(&ScoringView {
                    params: None,
                    avgdl: Some(("body", avgdl)),
                })
                .expect("a view at a new average length")
            })
            .collect();
        let (a, b, c) = join!(
            views[0].ensure_doc_map(),
            views[1].ensure_doc_map(),
            r.ensure_doc_map()
        );
        for map in [a, b, c] {
            map.expect("doc-id map");
        }
        views[2].ensure_doc_map().await.expect("doc-id map");
        assert_eq!(source.map_reads().len(), 1, "one read serves them all");
    }

    #[tokio::test]
    async fn open_lazy_round_trips_a_search() {
        // Wrap the eager blob in a whole-blob lazy source so the lazy
        // open path (header and length directory, everything else on
        // first use) runs and serves a real query.
        let (blob, json) = build_blob();
        let src: Arc<dyn LazyByteSource> = Arc::new(BytesLazyByteSource::new(blob));
        let r = FtsReader::open_lazy(src, &json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        assert_eq!(r.n_docs(), 3);
        let hits = r
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("search over lazy reader");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }

    /// How long the second of two racing searches may take while the first
    /// is held on its read: far above a search over three documents.
    const RACING_SEARCH_DEADLINE: Duration = Duration::from_secs(10);

    /// Poll cadence while waiting for a paused read to be reached.
    const PAUSE_POLL: Duration = Duration::from_millis(5);

    /// A whole-blob source that serves bytes only asynchronously, the way
    /// an object store does, and parks the first read touching `region`
    /// until told to go on.
    struct PausingSource {
        inner: BytesLazyByteSource,
        region: Range<usize>,
        armed: AtomicBool,
        paused: AtomicBool,
        resume: Notify,
    }

    #[async_trait]
    impl LazyByteSource for PausingSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let touches =
                (start as usize) < self.region.end && (start + len) as usize > self.region.start;
            if touches && self.armed.swap(false, Ordering::SeqCst) {
                self.paused.store(true, Ordering::SeqCst);
                self.resume.notified().await;
            }
            self.inner.range(start, len).await
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// A lazy reader over a [`PausingSource`] parked on the column's length
    /// array, armed after the open so only a query's read of the array
    /// pauses; the source is returned so a test can watch and release it.
    async fn reader_pausing_on_the_length_array() -> (Arc<PausingSource>, Arc<FtsReader>) {
        let (blob, json) = build_blob();
        let eager = FtsReader::open(blob.clone(), &json).expect("eager open");
        let lengths = norms_region(&eager.columns[0]);
        let source = Arc::new(PausingSource {
            inner: BytesLazyByteSource::new(blob),
            region: lengths,
            armed: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            resume: Notify::new(),
        });
        let src: Arc<dyn LazyByteSource> = source.clone();
        let reader = Arc::new(
            FtsReader::open_lazy(src, &json, OpenOptions::for_object_store())
                .await
                .expect("open_lazy"),
        );
        source.armed.store(true, Ordering::SeqCst);
        (source, reader)
    }

    /// Two scored single-term searches racing on a cold column: the second
    /// completes while the first is still waiting for the length array.
    ///
    /// The single-term path used to reach the norms through the synchronous
    /// fallback, whose initializer held the norms cell's lock across the
    /// read, so the second search parked on that lock — a thread, and on an
    /// `infino-io` worker the core the read needed — for as long as the
    /// read took. Two workers, so a parked one is visible. This pins the
    /// prewarm on the single-term path; the fallback's own lock behaviour
    /// is pinned below.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cold_scored_search_does_not_hold_a_racing_one_on_the_norms() {
        let (source, reader) = reader_pausing_on_the_length_array().await;

        let first = {
            let reader = Arc::clone(&reader);
            spawn(async move { reader.search("body", &["rust"], 10, BoolMode::Or).await })
        };
        // Bounded: a reader that stopped reading the array on a scored
        // search would otherwise wait here forever instead of failing.
        timeout(RACING_SEARCH_DEADLINE, async {
            while !source.paused.load(Ordering::SeqCst) {
                sleep(PAUSE_POLL).await;
            }
        })
        .await
        .expect("the first scored search must read the length array");
        let second = {
            let reader = Arc::clone(&reader);
            spawn(async move { reader.search("body", &["rust"], 10, BoolMode::Or).await })
        };
        let second_done = timeout(RACING_SEARCH_DEADLINE, second).await;
        // Release the first read before asserting, so a failure leaves no
        // thread parked behind it.
        source.resume.notify_one();
        let first_hits = first
            .await
            .expect("first search task")
            .expect("first search");
        let second_hits = second_done
            .expect("the second search must not wait on the first search's read")
            .expect("second search task")
            .expect("second search");
        let first_ids: HashSet<u32> = first_hits.iter().map(|(d, _)| d.get()).collect();
        let second_ids: HashSet<u32> = second_hits.iter().map(|(d, _)| d.get()).collect();
        assert_eq!(first_ids, second_ids);
        assert!(first_ids.contains(&0) && first_ids.contains(&1));
    }

    /// Two threads reaching a cold column's norms through the synchronous
    /// fallback itself, the first parked on its read: the second returns
    /// while the first is still parked, because the read runs outside the
    /// cell. This is the assertion that fails if the read ever moves back
    /// inside `get_or_init`, where the initializer's lock would hold the
    /// second thread for as long as the first's read takes; the racing
    /// search above no longer reaches the fallback, so it cannot tell.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_parked_fallback_read_does_not_hold_a_racing_one() {
        let (source, reader) = reader_pausing_on_the_length_array().await;
        let fallback = |reader: Arc<FtsReader>| {
            spawn_blocking(move || {
                reader.columns[0].norms();
            })
        };

        let first = fallback(Arc::clone(&reader));
        timeout(RACING_SEARCH_DEADLINE, async {
            while !source.paused.load(Ordering::SeqCst) {
                sleep(PAUSE_POLL).await;
            }
        })
        .await
        .expect("the first fallback must read the length array");
        let second = timeout(RACING_SEARCH_DEADLINE, fallback(Arc::clone(&reader))).await;
        let first_still_parked = !first.is_finished();
        // Release the first read before asserting, so a failure leaves no
        // thread parked behind it.
        source.resume.notify_one();
        first.await.expect("first fallback task");
        second
            .expect("the second fallback must not wait on the first's read")
            .expect("second fallback task");
        assert!(
            first_still_parked,
            "the first read was still parked when the second returned"
        );
        assert!(
            reader.columns[0].norms_loaded(),
            "either read fills the cell; both are the same table"
        );
    }

    /// A whole-blob source whose first read touching `region` fails the
    /// way a truncated object does, and serves every read after it.
    struct FailingOnceSource {
        inner: BytesLazyByteSource,
        region: Range<usize>,
        armed: AtomicBool,
    }

    #[async_trait]
    impl LazyByteSource for FailingOnceSource {
        fn size(&self) -> u64 {
            self.inner.size()
        }

        async fn range(&self, start: u64, len: u64) -> Result<Bytes, LazyByteSourceError> {
            let touches =
                (start as usize) < self.region.end && (start + len) as usize > self.region.start;
            if touches && self.armed.swap(false, Ordering::SeqCst) {
                return Err(LazyByteSourceError::ShortRead {
                    start,
                    requested: len,
                    got: 0,
                });
            }
            self.inner.range(start, len).await
        }

        fn try_get_range_sync(&self, _start: u64, _len: u64) -> Option<Bytes> {
            None
        }
    }

    /// A lazy reader over the test blob whose next read of the length array
    /// fails once, armed after the open so that read is the first after it;
    /// with the eager reader's length totals for `body`, the real ones.
    async fn lazy_reader_failing_once_on_the_length_array() -> (FtsReader, Option<ColumnLengthStats>)
    {
        let (blob, json) = build_blob();
        let eager = FtsReader::open(blob.clone(), &json).expect("eager open");
        let real = eager.column_length_stats("body").expect("eager lengths");
        let source = Arc::new(FailingOnceSource {
            inner: BytesLazyByteSource::new(blob),
            region: norms_region(&eager.columns[0]),
            armed: AtomicBool::new(false),
        });
        let src: Arc<dyn LazyByteSource> = source.clone();
        let reader = FtsReader::open_lazy(src, &json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        source.armed.store(true, Ordering::SeqCst);
        (reader, real)
    }

    /// The commit path records a column's length totals on the manifest, so a
    /// length array that cannot be read is an error there, never a column with
    /// no tokens; and like the scoring fallback, it is not cached.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unreadable_length_array_is_an_error_not_empty_length_stats() {
        let (reader, real) = lazy_reader_failing_once_on_the_length_array().await;
        let err = reader
            .column_length_stats("body")
            .expect_err("an unreadable length array must not count as empty");
        assert!(
            matches!(
                err,
                FtsError::RangeFetch {
                    what: "fts/doc_lengths_array",
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(
            !reader.columns[0].norms_loaded(),
            "the failure is not cached"
        );
        assert_eq!(
            reader.column_length_stats("body").expect("the next read"),
            real,
            "the next read gets the real totals"
        );
    }

    /// An unranked match (the matching rows, or their count, under OR or AND)
    /// never scores, so it must not read the length array.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unranked_match_never_reads_the_length_array() {
        let (blob, json) = multi_block_blob();
        let terms = ["common", "shared"];
        let eager = FtsReader::open(blob.clone(), json).expect("eager open");
        let source = Arc::new(FailingOnceSource {
            inner: BytesLazyByteSource::new(blob),
            region: norms_region(&eager.columns[0]),
            armed: AtomicBool::new(false),
        });
        let src: Arc<dyn LazyByteSource> = source.clone();
        let reader = FtsReader::open_lazy(src, json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        source.armed.store(true, Ordering::SeqCst);

        for mode in [BoolMode::Or, BoolMode::And] {
            let (want, _) = eager
                .token_match("body", &terms, mode)
                .await
                .expect("eager");
            let (got, _) = reader
                .token_match("body", &terms, mode)
                .await
                .expect("match");
            assert_eq!(got, want, "{mode:?} rows");
            let (want, _) = eager
                .token_match_count("body", &terms, mode)
                .await
                .expect("eager count");
            let (got, _) = reader
                .token_match_count("body", &terms, mode)
                .await
                .expect("count");
            assert_eq!(got, want, "{mode:?} count");
        }
        assert!(
            source.armed.load(Ordering::SeqCst),
            "an unranked match must not read the length array"
        );
        assert!(!reader.columns[0].norms_loaded());
    }

    /// An unranked phrase or boolean match (the rows, or their count with a
    /// negation) never scores, so it must not read the length array: on a
    /// cold reader that read is a GET for nothing, and a failure there would
    /// fail a match that needs no norms.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unranked_phrase_or_boolean_match_never_reads_the_length_array() {
        let mut b = FtsBuilder::new();
        b.register_column("body".into(), true).expect("register");
        for doc in 0..64u32 {
            let text = match doc % 4 {
                0 => "the quick brown fox jumps",
                1 => "a quick red fox",
                2 => "brown dogs and a quick fox",
                _ => "slow brown turtle",
            };
            b.add_doc(0, doc, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","positions":true,"k1":1.2,"b":0.75}]"#;
        let eager = FtsReader::open(blob.clone(), json).expect("eager open");
        let source = Arc::new(FailingOnceSource {
            inner: BytesLazyByteSource::new(blob),
            region: norms_region(&eager.columns[0]),
            armed: AtomicBool::new(false),
        });
        let src: Arc<dyn LazyByteSource> = source.clone();
        let reader = FtsReader::open_lazy(src, json, OpenOptions::for_object_store())
            .await
            .expect("open_lazy");
        source.armed.store(true, Ordering::SeqCst);

        let quick_fox = vec![Phrase::adjacent(vec![
            "quick".to_string(),
            "fox".to_string(),
        ])];
        let slow_brown = vec![Phrase::adjacent(vec![
            "slow".to_string(),
            "brown".to_string(),
        ])];
        for mode in [BoolMode::Or, BoolMode::And] {
            let (want, _) = eager
                .atoms_match_ids("body", &["brown"], &quick_fox, mode)
                .await
                .expect("eager ids");
            let (got, _) = reader
                .atoms_match_ids("body", &["brown"], &quick_fox, mode)
                .await
                .expect("ids");
            assert_eq!(got, want, "{mode:?} rows");
            let (want, _) = eager
                .atoms_match_count("body", &["brown"], &quick_fox, mode, &["dogs"], &slow_brown)
                .await
                .expect("eager count");
            let (got, _) = reader
                .atoms_match_count("body", &["brown"], &quick_fox, mode, &["dogs"], &slow_brown)
                .await
                .expect("count");
            assert_eq!(got, want, "{mode:?} count");
        }
        assert!(
            source.armed.load(Ordering::SeqCst),
            "an unranked phrase or boolean match must not read the length array"
        );
        assert!(!reader.columns[0].norms_loaded());
    }

    /// A length-array read that fails in the synchronous fallback scores
    /// that call against the empty table and leaves the column's cell
    /// empty, so the next read fills it: the failure is not what every
    /// later query on the reader scores against.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failed_fallback_read_of_the_norms_is_not_cached() {
        let (reader, _) = lazy_reader_failing_once_on_the_length_array().await;
        let column = &reader.columns[0];

        // The fallback survives the failure without publishing it.
        let _empty = column.norms();
        assert!(
            !column.norms_loaded(),
            "a failed read must not fill the norms cell"
        );

        // The next scoring entry point reads the array again, and scores.
        let hits = reader
            .search("body", &["rust"], 10, BoolMode::Or)
            .await
            .expect("search after the failed read");
        assert!(column.norms_loaded(), "the retry must fill the norms cell");
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| d.get()).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }

    /// The union count's helpers on inputs the corpora above do not reach.
    mod union_count_edges {
        use super::*;

        /// `body` cursors for `terms` over `docs`, for a count.
        async fn count_cursors(docs: &[&str], terms: &[&str]) -> Vec<TermCursor> {
            let mut b = FtsBuilder::new();
            b.register_column("body".into(), false).expect("register");
            for (i, text) in docs.iter().enumerate() {
                b.add_doc(0, i as u32, text).expect("add doc");
            }
            let json = r#"[{"name":"body","tokenizer":"standard","k1":1.2,"b":0.75}]"#;
            let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
            r.build_term_cursors(0, terms, QueryIdf::ThisFile, CursorUse::Count, None, None)
                .await
                .expect("cursors")
        }

        /// A two-term union counts `df(a) + df(b) - |a ∩ b|` from the stored
        /// dfs; a corrupt df must saturate the count, not wrap it.
        #[tokio::test]
        async fn a_two_term_union_count_saturates_on_corrupt_dfs() {
            let docs: Vec<&str> = (0..300)
                .map(|i| if i % 3 == 0 { "a b" } else { "a" })
                .collect();
            let cursors = count_cursors(&docs, &["a", "b"]).await;
            assert_eq!(
                or_count_unranked(cursors.clone()),
                300,
                "premise: sound dfs"
            );
            let mut huge = cursors.clone();
            huge[0].df = u64::MAX;
            assert_eq!(or_count_unranked(huge), u64::MAX - 100);
            let mut zero = cursors;
            zero[0].df = 0;
            zero[1].df = 0;
            assert_eq!(or_count_unranked(zero), 0);
        }

        /// An empty or inverted range sets nothing; a range across words
        /// sets exactly its bits.
        #[test]
        fn set_bit_range_sets_exactly_the_range() {
            let mut dest = [0u64; 4];
            for (from, to) in [(0, 0), (64, 64), (70, 5), (200, 1)] {
                set_bit_range(&mut dest, from, to);
                assert_eq!(dest, [0; 4], "[{from}, {to})");
            }
            set_bit_range(&mut dest, 3, 130);
            assert_eq!(dest, [u64::MAX << 3, u64::MAX, 0b11, 0]);
        }

        /// A short-form cursor has no blocks to read: its decoded docs inside
        /// the window are marked, the rest left out.
        #[tokio::test]
        async fn or_blocks_into_window_marks_a_predecoded_cursor() {
            let planted = [3usize, 70, OR_WINDOW as usize + 4, OR_WINDOW as usize + 904];
            let docs: Vec<&str> = (0..2 * OR_WINDOW as usize)
                .map(|i| if planted.contains(&i) { "rare" } else { "x" })
                .collect();
            let cursors = count_cursors(&docs, &["rare"]).await;
            assert!(cursors[0].predecoded, "premise: a short-form term");
            let mut scratch = [0u32; BLOCK_LEN];
            for (base, want) in [(0u32, [3usize, 70]), (OR_WINDOW, [4, 904])] {
                let mut dest = [0u64; OR_WINDOW_WORDS];
                let mut from_block = 0usize;
                or_blocks_into_window(&cursors[0], &mut from_block, base, &mut dest, &mut scratch);
                let mut expect = [0u64; OR_WINDOW_WORDS];
                for local in want {
                    expect[local / 64] |= 1 << (local % 64);
                }
                assert_eq!(dest, expect, "window at {base}");
            }
        }
    }
}
