// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS blob reader. Multi-column BM25 search.
//!
//! Opens the byte layout produced by [`super::builder::FtsBuilder::finish`]
//! and exposes BM25 search per-column or weighted across columns.
//!
//! See `docs/architecture/superfile.md` for the on-disk layout.
//!
//! ## Threading
//!
//! `FtsReader` is `Send + Sync` and immutable after `open()` — concurrent
//! `search` calls share the underlying `Bytes`. The DictReader is
//! constructed per call (cheap; the FST validates its header in O(1) and
//! then it's a borrowed view).

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
    ops::Range,
    sync::Arc,
};

use bytes::Bytes;
use serde::Deserialize;

use crate::{
    runtime_metrics::{
        cpu::{thread_cpu_delta_ns, thread_cpu_ns},
        op_stats::{metering_active, timed_section},
    },
    superfile::{
        ReadError,
        error::FtsError,
        format::{
            self, FST_SEPARATOR,
            checksum::crc32c,
            fts::{
                HEADER_SIZE_V1_LEGACY as FTS_HEADER_SIZE, MAGIC_BYTES, U32_BYTES, U64_BYTES, hdr,
                skip_entry, term_meta,
            },
        },
        fts::{
            bm25,
            builder::{
                DOC_LENGTHS_ENTRY_SIZE, SKIP_ENTRY_SIZE, TERM_META_POSITIONAL_SIZE, TERM_META_SIZE,
            },
            dict::{DictReader, make_key},
            fst_value::FstValue,
            positions::{decode_run, skip_run},
            posting::{BLOCK_LEN, decode_block},
            tokenize::{Tokenizer, tokenizer_for_name},
        },
        lazy_source::{LazyByteSource, PrefetchedSource, RangeCoalescePlan, Source},
    },
};

/// Largest gap worth overfetching when adjacent term postings share a request.
const TERM_RANGE_COALESCE_MAX_GAP: usize = 64 * 1024;
/// Maximum total gap bytes tolerated in one coalesced postings request.
const TERM_RANGE_COALESCE_MAX_OVERFETCH: usize = 512 * 1024;

/// Boolean-mode for multi-term queries.
/// Default operator for a query's bare (sigil-less) terms. Terms
/// carrying an explicit clause sigil keep their polarity regardless
/// of mode: `+term` is a must (every hit contains it), `-term` a
/// must-not (hard exclusion).
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum BoolMode {
    /// Bare terms are musts: all of them must match the doc.
    And,
    /// Bare terms are shoulds: any of them matching contributes to
    /// the doc's score. When the query also carries `+must` terms,
    /// the musts alone define the match set and bare terms become
    /// scoring-only. The default.
    #[default]
    Or,
}

/// Which BM25 collection statistics to score term rarity (idf) with
/// across the superfiles a query fans out over.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum Bm25Stats {
    /// Score each superfile against its own local document count and
    /// term document-frequencies. Fast (full fan-out, no extra pass),
    /// but a term's idf — and therefore a doc's score — depends on
    /// which superfile it lands in, so scores are only approximately
    /// comparable across superfiles and ranking drifts as the table
    /// fragments. The default.
    #[default]
    PerSuperfile,
    /// Score every superfile against table-wide idf: the document count
    /// and per-term document-frequencies aggregated across all
    /// superfiles in the query's manifest snapshot. A term then has one
    /// idf for the whole table, so a fragmented table ranks like a
    /// single unified corpus, at the cost of a document-frequency
    /// gather before scoring. (Length normalization still uses each
    /// superfile's own average document length.)
    Global,
}

impl From<&str> for Bm25Stats {
    fn from(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "global" => Bm25Stats::Global,
            _ => Bm25Stats::PerSuperfile,
        }
    }
}

/// Options for a BM25 search: the boolean `mode` and the corpus-statistics
/// `stats`. Set fields with the `with_*` builders; [`Default`] is
/// [`BoolMode::Or`] with [`Bm25Stats::PerSuperfile`].
///
/// ```ignore
/// // OR mode, per-superfile stats (the defaults):
/// Bm25SearchOptions::new()
/// // AND mode, global stats:
/// Bm25SearchOptions::new().with_mode(BoolMode::And).with_stats(Bm25Stats::Global)
/// ```
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Bm25SearchOptions {
    /// Boolean mode for the query's bare terms (`Or` = should, `And` = must).
    pub mode: BoolMode,
    /// Which BM25 corpus statistics to score with.
    pub stats: Bm25Stats,
}

impl Bm25SearchOptions {
    /// Default options: `Or` mode, per-superfile statistics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the boolean mode for the query's bare terms.
    pub fn with_mode(mut self, mode: BoolMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set which BM25 corpus statistics to score with.
    pub fn with_stats(mut self, stats: Bm25Stats) -> Self {
        self.stats = stats;
        self
    }
}

/// Per-term global BM25 idf (the raw `idf`, not `idf × (k1+1)`) keyed
/// by term, used by [`Bm25Stats::Global`]. A term absent from the map
/// falls back to that superfile's local idf.
pub(crate) type GlobalTermIdf = std::collections::HashMap<String, f32>;

/// A query's parsed clause lists, borrowed for one search call —
/// terms and phrases per polarity, with the default operator already
/// resolved (see `ParsedQuery::into_clauses`). Grouped so the search
/// entry points don't take nine parallel parameters.
#[derive(Default)]
pub(crate) struct ClauseLists<'a> {
    pub musts: &'a [&'a str],
    pub shoulds: &'a [&'a str],
    pub negatives: &'a [&'a str],
    pub must_phrases: &'a [Vec<String>],
    pub should_phrases: &'a [Vec<String>],
    pub negative_phrases: &'a [Vec<String>],
    /// Per-term global idf for [`Bm25Stats::Global`]; `None` scores
    /// with per-superfile local idf (the default).
    pub global_idf: Option<&'a GlobalTermIdf>,
}

impl ClauseLists<'_> {
    /// Any phrase atom anywhere routes the query to the atom walks.
    fn has_phrases(&self) -> bool {
        !self.must_phrases.is_empty()
            || !self.should_phrases.is_empty()
            || !self.negative_phrases.is_empty()
    }

    /// Nothing to rank or match on the positive side.
    fn no_positive_atoms(&self) -> bool {
        self.musts.is_empty()
            && self.shoulds.is_empty()
            && self.must_phrases.is_empty()
            && self.should_phrases.is_empty()
    }

    /// Nothing negated either.
    fn no_negative_atoms(&self) -> bool {
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
        hits: Vec<(u32, f32)>,
        postings_bytes: u64,
        /// Byte-source ranges the inline walk requested (0 for the df=1
        /// inline-FST and empty-resolution paths).
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
        /// FST-dictionary ranges the builds requested (one per
        /// `build_term_cursors` call — must / should / negation lists).
        dict_ranges: u64,
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
    },
}

/// Sum of the posting-byte ranges a cursor set indexes into (each cursor's
/// term metadata + skip table + posting blocks). Feeds the per-query work
/// stats ([`crate::runtime_metrics::op_stats`]).
fn term_cursor_bytes(cursors: &[TermCursor]) -> u64 {
    cursors.iter().map(|c| c.bytes.len() as u64).sum()
}

/// [`term_cursor_bytes`] for heterogeneous atoms: a phrase member counts
/// its posting bytes **and** its position runs — positional verification
/// is exactly the work that separates phrase cost from term cost.
fn atom_cursor_bytes(atoms: &[AnyCursor]) -> u64 {
    atoms
        .iter()
        .map(|a| match a {
            AnyCursor::Term(c) => c.bytes.len() as u64,
            AnyCursor::Phrase(p) => p
                .members
                .iter()
                .map(|m| m.cursor.bytes.len() as u64 + m.positions.len() as u64)
                .sum(),
        })
        .sum()
}

/// Byte-source ranges a cursor set's build requested — one per PFOR
/// term. Inline (df=1) cursors plan no fetch (their `bytes` is empty),
/// matching the single-term arm's "bytes 0 implies ranges 0".
fn term_cursor_ranges(cursors: &[TermCursor]) -> u64 {
    cursors
        .iter()
        .filter(|c| !c.bytes.is_empty())
        .map(|c| 1 + u64::from(c.header_probed))
        .sum()
}

/// Byte-source ranges the atoms' builds requested: one per PFOR term's
/// posting range; a phrase member adds one for its postings and one for
/// its position runs. Inline legs (empty buffers) plan no fetch.
fn atom_planned_ranges(atoms: &[AnyCursor]) -> u64 {
    let term_ranges = |c: &TermCursor| {
        if c.bytes.is_empty() {
            0
        } else {
            1 + u64::from(c.header_probed)
        }
    };
    atoms
        .iter()
        .map(|a| match a {
            AnyCursor::Term(c) => term_ranges(c),
            AnyCursor::Phrase(p) => p
                .members
                .iter()
                .map(|m| term_ranges(&m.cursor) + u64::from(!m.positions.is_empty()))
                .sum(),
        })
        .sum()
}

/// Work tallies from one unranked match / dictionary walk — the posting
/// bytes indexed and the byte-source ranges the plan requested. Returned
/// alongside match results so the supertable flushes once per superfile;
/// `pub` because the carrying fns are the test-helpers-widened surface
/// (the module gate in `lib.rs` keeps it crate-private in normal builds).
#[derive(Debug, Default, Clone, Copy)]
pub struct MatchWork {
    /// Posting (and phrase-position / header) bytes the walk indexed into.
    pub postings_bytes: u64,
    /// Byte-source ranges the walk's build requested, pre-coalesce.
    pub planned_ranges: u64,
    /// Bracketed on-CPU ns of the walk's synchronous scoring/merge
    /// section (gated on `metering_active`; 0 when unmetered).
    pub kernel_cpu_ns: u64,
}

impl MatchWork {
    /// Tallies for a plain term-cursor set.
    fn for_cursors(cursors: &[TermCursor]) -> Self {
        Self {
            postings_bytes: term_cursor_bytes(cursors),
            planned_ranges: term_cursor_ranges(cursors),
            kernel_cpu_ns: 0,
        }
    }

    /// Tallies for a heterogeneous atom set (terms + phrases).
    fn for_atoms(atoms: &[AnyCursor]) -> Self {
        Self {
            postings_bytes: atom_cursor_bytes(atoms),
            planned_ranges: atom_planned_ranges(atoms),
            kernel_cpu_ns: 0,
        }
    }

    /// Fold another walk's tallies into this one (e.g. a negation set).
    pub fn merge(&mut self, other: MatchWork) {
        self.postings_bytes += other.postings_bytes;
        self.planned_ranges += other.planned_ranges;
        self.kernel_cpu_ns += other.kernel_cpu_ns;
    }
}

impl PreparedClauses {
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
            PreparedClauses::Or { cursors, .. } => cursors.iter().map(|c| c.df).sum(),
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

impl From<&str> for BoolMode {
    fn from(s: &str) -> Self {
        match s {
            "and" => BoolMode::And,
            "or" => BoolMode::Or,
            _ => BoolMode::Or,
        }
    }
}

/// Multi-term OR algorithm selector for the bench harness's
/// `search_with_algo_for_bench` entry point. Production code routes
/// through `FtsReader::dispatch_or_algo`, which picks
/// automatically; this enum exists so head-to-head bench runs can
/// compare all three under identical inputs.
#[doc(hidden)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OrAlgo {
    /// Block-Max MaxScore: production default for dominant-term ORs.
    Bmm,
    /// WAND + Block-Max-WAND: historical baseline; retained for
    /// regression comparisons.
    WandBmw,
    /// Exhaustive union walk with SIMD scoring + top-K heap. Wins
    /// when no term dominates (uniform `term_max_bm25` upper bounds)
    /// so BMM/BMW's skip checks rarely trigger and become pure
    /// overhead.
    Exhaustive,
    /// Windowed union: accumulate each term's contribution into a
    /// fixed doc-id window (presence bitset + score array), then drain
    /// in doc order into the top-k heap. Removes the per-doc f-way
    /// merge; wins when no term dominates and the union is large (the
    /// MaxScore-can't-prune case).
    Windowed,
}

/// Doc-id window for the windowed union scorer. Power of two so the
/// window base is a cheap mask. At 4096 the per-window state — a
/// `4096 × f32` score accumulator (16 KiB) plus a `4096`-bit presence
/// bitset (512 B) — stays L1/L2-resident across the accumulate + drain
/// passes.
const OR_WINDOW: u32 = 4096;
/// Number of 64-bit words in the window presence bitset.
const OR_WINDOW_WORDS: usize = (OR_WINDOW as usize).div_ceil(64);

/// Multi-term OR dispatch floor. A 2-term OR is already sub-millisecond
/// on MaxScore, so the window's per-window bookkeeping isn't worth it
/// below this many terms. `pub(crate)`: the supertable fan-out reuses
/// this same boundary to decide when a ranged kernel is heavy enough to
/// ship to the reader pool (see `RANGED_KERNEL_POOL_MIN_TERMS`).
pub(crate) const OR_WINDOW_MIN_TERMS: usize = 3;
/// Route a multi-term OR to the windowed union scorer only when the top
/// term's score upper bound is at most this multiple of the *average*
/// term upper bound — i.e. no single term dominates. Uniform terms sit at
/// ~1.0× the average (MaxScore can't prune them → windowed wins); a
/// dominant rare term sits well above it (MaxScore prunes hard → it stays
/// on MaxScore). Calibrated on the 1M tier; re-measured on every bench
/// run by the superfile tier's per-algorithm probes
/// (`benches/utils/superfile.rs`), whose shapes sit on both sides of
/// this threshold.
const OR_WINDOW_DOMINANCE_MULT: f32 = 1.5;

/// Largest `k` for which a 2-term OR routes to WAND+BMW instead of
/// MaxScore. WAND's pivot pruning needs a high top-k threshold to skip
/// blocks: at small `k` the threshold is high and it clears MaxScore
/// decisively on two comparable terms, but as `k` grows the threshold
/// falls until WAND can no longer prune and its per-iteration cursor
/// re-sort becomes pure overhead — so above this `k` MaxScore wins. The
/// cutoff sits between the common small-`k` page sizes and the rare deep
/// `k`; large-`k` 2-term ORs stay on MaxScore.
const WAND_BMW_2TERM_MAX_K: usize = 128;

/// Route a 2-term OR to WAND+BMW only when one term's posting list is at
/// least this many times shorter than the other's (df ratio). That rare
/// "anchor" term is what lets WAND pivot and skip the common term's long
/// list — the source of its win. Two comparable-length lists (e.g. two
/// common words) give WAND nothing to skip, so it loses to MaxScore and
/// stays there. A *score* upper-bound ratio is the wrong test here: a
/// term can dominate the BM25 UB (higher idf) while still being common
/// (long list), which WAND can't skip — only df separates the cases.
const WAND_BMW_2TERM_DF_RATIO: u64 = 16;

/// True when **no single term dominates** the score upper bound:
/// `max_ub <= OR_WINDOW_DOMINANCE_MULT * avg_ub`. Uniform terms sit near
/// the average — MaxScore can't prune them (its essential set never
/// shrinks); a dominant (typically rare) term sits well above it, and
/// MaxScore / WAND can skip hard against it. Shared by the windowed-union
/// and 2-term WAND routers, which want opposite sides of this test. Cheap
/// — the per-term upper bounds are already on the cursors.
fn no_dominant_term_ub(cursors: &[TermCursor]) -> bool {
    let total: f32 = cursors.iter().map(|c| c.term_max_bm25).sum();
    if total <= 0.0 {
        return false;
    }
    let max = cursors
        .iter()
        .map(|c| c.term_max_bm25)
        .fold(0.0f32, f32::max);
    let avg = total / cursors.len() as f32;
    max <= OR_WINDOW_DOMINANCE_MULT * avg
}

/// Choose the windowed union scorer over MaxScore+BMM for a multi-term
/// OR: true when there are enough terms to amortize the window and no
/// single term dominates (so MaxScore degrades to scoring the whole
/// union).
fn prefer_windowed_union(cursors: &[TermCursor]) -> bool {
    cursors.len() >= OR_WINDOW_MIN_TERMS && no_dominant_term_ub(cursors)
}

/// Minimum dominant-term df for the deep-`k` reroute to the windowed
/// scorer to pay off. Below this the union is small enough that
/// MaxScore's scalar full scan beats the windowed scorer's fixed
/// per-window setup, so short unions stay on MaxScore. Calibrated on the
/// warm FTS bench: the win concentrates on dominant lists in the millions
/// (~corpus-scale), the mid range is a wash, and lists shorter than this
/// regressed. See [`or_topk_pruning_ineffective`].
const OR_WINDOWED_MIN_DOMINANT_DF: u64 = 100_000;

/// True when block-max pruning (MaxScore / WAND) can no longer skip
/// blocks at this `k`, so both degrade to a full union scan — at which
/// point the SIMD windowed scorer does that same scan far faster than
/// MaxScore's scalar per-doc f-way merge.
///
/// Pruning stays alive only while the top-k threshold sits *above* the
/// common (low-idf, longest-list) term's score upper bound: only then
/// can that term's blocks be skipped. The threshold holds there only as
/// long as the rarer terms alone can fill the heap. Once `k` reaches the
/// combined df of every term *except* the single longest list, the heap
/// must admit docs from that longest list's tail — docs whose only
/// matching term is the common one — the threshold collapses toward its
/// low upper bound, and no block clears the skip test any more.
///
/// `rest_df` (sum of all dfs but the largest) is a conservative bound on
/// how many docs the rarer terms can contribute: it ignores overlap, so
/// the true fillable count is `<= rest_df`. When `k >= rest_df` the heap
/// therefore *cannot* be filled without the common term's tail, and
/// pruning is dead. This is `df`-only — no extra reads — and
/// self-correcting: a "rare" second term that is actually long keeps
/// `rest_df` high, leaves pruning alive, and stays on MaxScore.
///
/// Reroute only when the dominant list is also *long* (`max_df >=`
/// [`OR_WINDOWED_MIN_DOMINANT_DF`]). Once pruning is dead both scorers
/// scan the whole union, but the windowed scorer's per-window bookkeeping
/// (a 4096-wide score accumulator + presence bitset) only pays off when
/// the dominant list is long enough to amortize it; on a short union it
/// is pure overhead and MaxScore's scalar scan is faster. A union with
/// fewer than the floor's matches can't have a longer dominant list than
/// the floor, so this gates out exactly the small-union case that
/// regressed on the bench.
fn or_topk_pruning_ineffective(cursors: &[TermCursor], k: usize) -> bool {
    let max_df = cursors.iter().map(|c| c.df).max().unwrap_or(0);
    let total_df: u64 = cursors.iter().map(|c| c.df).sum();
    or_reroute_by_df(max_df, total_df, cursors.len(), k)
}

/// Pure df-math behind [`or_topk_pruning_ineffective`], split out so the
/// routing decision can be unit-tested at df values (hundreds of
/// thousands) that a fast in-memory corpus can't reach.
fn or_reroute_by_df(max_df: u64, total_df: u64, n_terms: usize, k: usize) -> bool {
    if n_terms < 2 || max_df < OR_WINDOWED_MIN_DOMINANT_DF {
        return false;
    }
    let rest_df = total_df.saturating_sub(max_df);
    k as u64 >= rest_df
}

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
fn two_term_has_rare_anchor(cursors: &[TermCursor]) -> bool {
    if cursors.len() != 2 {
        return false;
    }
    let lo = cursors[0].df.min(cursors[1].df);
    let hi = cursors[0].df.max(cursors[1].df);
    lo > 0 && hi >= lo.saturating_mul(WAND_BMW_2TERM_DF_RATIO)
}

/// Per-doc BM25 length normalizer, quantized to one byte per doc.
///
/// The scorer needs `dl_norm_k1[doc] = K1·(1 - B + B·dl/avgdl)` for
/// every scored doc. Held as an `f32` per doc, that table is 4 bytes ×
/// n_docs — at multi-million-doc scale too large to stay cache-resident,
/// so each scored doc pays a scattered load from a table that overflows
/// cache. Instead the doc length is quantized to one byte
/// ([`bm25::quantize_len`]) and a 256-entry table decodes each bucket to
/// its norm value: the per-doc table is 4× smaller (one byte), and the
/// decode table is 1 KiB (L1-resident). A scored doc reads
/// `lut[bytes[doc]]` — one load from the small per-doc table plus one L1
/// lookup — instead of one load from a 4×-larger table.
#[derive(Debug, Clone)]
pub struct NormTable {
    /// Per-doc quantized length bucket. Empty for a column with no docs.
    bytes: Vec<u8>,
    /// Bucket → `K1·(1 - B + B·dequantize_len(bucket)/avgdl)`. A fixed
    /// 256-entry table, boxed so `ColumnMeta` stays pointer-sized (it is
    /// scanned by non-scoring paths — column lookup, listing) while the
    /// `u8` bucket index into a fixed-length array lets the compiler drop
    /// the bounds check in `get`.
    lut: Box<[f32; 256]>,
}

impl NormTable {
    /// Build from a column's per-doc lengths and average length. An
    /// `avgdl` of `0.0` (empty column) yields an empty table; it is
    /// never indexed because `search` short-circuits on empty columns.
    fn new(doc_lengths: impl Iterator<Item = u32>, n_docs: usize, avgdl: f32) -> Self {
        if avgdl <= 0.0 {
            return Self::empty();
        }
        let inv_avgdl = 1.0_f32 / avgdl;
        // Fill the boxed table in place so the 256 f32s land on the heap
        // directly rather than being built on the stack and moved.
        let mut lut = Box::new([0.0_f32; 256]);
        for (b, slot) in lut.iter_mut().enumerate() {
            let dl = bm25::dequantize_len(b as u8) as f32;
            *slot = bm25::K1 * (1.0 - bm25::B + bm25::B * dl * inv_avgdl);
        }
        let mut bytes = Vec::with_capacity(n_docs);
        for dl in doc_lengths {
            bytes.push(bm25::quantize_len(dl));
        }
        Self { bytes, lut }
    }

    /// `dl_norm_k1` for a doc (length quantized): one per-doc byte load
    /// plus one L1 decode-table lookup. Hot path — keep it inlined.
    #[inline(always)]
    fn get(&self, doc: u32) -> f32 {
        self.lut[self.bytes[doc as usize] as usize]
    }

    /// Number of docs in the table. Test-only: the query path indexes
    /// by doc id and never needs the count.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.bytes.len()
    }

    /// An empty table: `bytes` is empty, so `get` must never be called on
    /// it. For call sites that need a `&NormTable` but provably never index
    /// it — an unranked (`bar == NEG_INFINITY`) phrase seek, which does no
    /// scoring. The `lut` is a zeroed 256-entry table, allocated but never
    /// read.
    fn empty() -> Self {
        Self {
            bytes: Vec::new(),
            lut: Box::new([0.0; 256]),
        }
    }
}

/// Per-column metadata, indexed by column_id (declaration order).
#[derive(Debug, Clone)]
pub struct ColumnMeta {
    pub name: String,
    /// Byte range into [`FtsReader::blob`] holding this column's
    /// `u32` doc-lengths array (4 bytes per doc, length × n_docs).
    pub doc_lengths_range: Range<usize>,
    /// Average doc length across this column. `0.0` if the column has
    /// no docs.
    pub avgdl: f32,
    /// Per-doc BM25 length normalizer, byte-quantized — see
    /// [`NormTable`]. Computed once per reader at `open` time from the
    /// column's on-disk doc-lengths array. The hot scoring loop reads
    /// `dl_norm_k1.get(d)` and multiplies-out to `idf · tf · (K1+1) /
    /// (tf + dl_norm_k1.get(d))`.
    pub dl_norm_k1: NormTable,
    /// Whether this column's index carries token positions (from
    /// `inf.fts.columns`); phrase queries require it.
    pub positions: bool,
    /// Tokenizer for this column, reconstructed at open time from the
    /// `tokenizer` name in `inf.fts.columns`. Query terms for this
    /// column must be tokenized with it to match how the column was
    /// indexed.
    pub tokenizer: Arc<dyn Tokenizer>,
}

/// JSON-deserialized form of one entry in `inf.fts.columns`. The KV
/// value is a JSON array of these, in declaration order.
#[derive(Debug, Clone, Deserialize)]
pub struct FtsColumnConfig {
    pub name: String,
    /// The column's analyzer name: `"ascii_lower"` (the default) or
    /// `"standard"`. A missing field deserializes to `"ascii_lower"`
    /// for backward compatibility with files written before the
    /// analyzer name was recorded.
    #[serde(default = "default_tokenizer")]
    pub tokenizer: String,
    /// Whether this column's index records token positions (phrase
    /// support). Files written before positions existed lack the
    /// field, which can only mean no positions — so a missing field
    /// deserializes to `false`.
    #[serde(default)]
    pub positions: bool,
}

fn default_tokenizer() -> String {
    "ascii_lower".to_string()
}

/// Per-open knobs for [`FtsReader::open_with`]. Mirrors the
/// vector reader's `OpenOptions` so the superfile layer can
/// pass a single `verify_crc` flag through to both
/// sub-readers.
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Verify the four per-section CRC32C checks (FST,
    /// postings region, doc-lengths directory, per-column
    /// doc-lengths arrays). Defaults to `true`; flip to
    /// `false` only when the underlying storage already
    /// validates checksums (content-addressed object
    /// store, ZFS, etc.) to skip the scan on cold open.
    pub verify_crc: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self { verify_crc: true }
    }
}

impl OpenOptions {
    pub fn for_object_store() -> Self {
        Self { verify_crc: false }
    }
}

/// FTS blob reader. Self-contained — owns its `Bytes` (which the storage
/// layer assembled from mmap / range-fetch / full-read).
#[derive(Debug)]
pub struct FtsReader {
    source: Source,
    n_docs: u32,
    n_terms_total: u32,
    fst_range: Range<usize>,
    postings_range: Range<usize>,
    /// Byte range of the positions region (CRC stripped) — `Some`
    /// iff the blob is v2. Phrase queries fetch per-term run ranges
    /// out of it via [`Self::fetch_term_positions`].
    positions_range: Option<Range<usize>>,
    columns: Vec<ColumnMeta>,
    column_id_by_name: HashMap<String, u32>,
}

impl FtsReader {
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
    /// subsection. Three open-time GETs prefetch the only regions a
    /// reader needs before it can serve queries: the fixed header, the
    /// FST term directory (contiguous after the header), and the
    /// doc-length tables (the trailing region, needed to build BM25
    /// normalization). The postings region stays lazy — each query
    /// term's bytes are fetched on demand by [`Self::fetch_term_postings`],
    /// mirroring how the vector reader fetches only probed clusters.
    pub async fn open_lazy(
        source: Arc<dyn LazyByteSource>,
        columns_json: &str,
        opts: OpenOptions,
    ) -> Result<Self, FtsError> {
        // Length of the FTS subsection itself (≈ `kv::FTS_LENGTH`), not
        // the whole superfile: `source` is the FTS-scoped sub-source.
        let fts_blob_len = source.size() as usize;
        // One GET covers either header size: any real FTS blob is
        // larger than the 56-byte v2 header (header + FST CRC +
        // postings CRC + a non-empty doc-lengths directory), so
        // fetching the v2 span up front costs no extra round-trip on
        // v1 blobs and saves one on v2.
        let header_fetch = format::fts::HEADER_SIZE_V2.min(fts_blob_len);
        let header = fetch_lazy_range(source.as_ref(), 0..header_fetch, "fts header").await?;
        if header.len() < FTS_HEADER_SIZE {
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
        if version != format::fts::VERSION_V1_LEGACY && version != format::fts::VERSION_V2 {
            return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                "fts section version {version}"
            ))));
        }
        // The FST directory starts right after whichever header
        // applies; a v2 header's extension bytes are already in the
        // fetched span (and in the overlay below), so
        // `open_with_source` re-reads them without another GET.
        let header_size = match version {
            v if v == format::fts::VERSION_V2 => format::fts::HEADER_SIZE_V2,
            _ => FTS_HEADER_SIZE,
        };
        if header.len() < header_size {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }

        let postings_offset =
            read_u64_le(&header[hdr::POSTINGS_OFFSET_OFF..hdr::POSTINGS_OFFSET_OFF + U64_BYTES])
                as usize;
        let doc_lengths_table_offset =
            read_u64_le(&header[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;

        // Prefetch the FST directory ([48..postings_offset], contiguous
        // after the header) so every later `dict_bytes()` resolves from
        // the overlay instead of a fresh GET per search, and the
        // doc-length tail ([doc_lengths_table_offset..fts_blob_len]) so
        // `open_with_source` builds its BM25 norm tables without
        // touching the source again. The doc-lengths region is the
        // *trailing* region of the FTS blob (it follows the postings),
        // so `..fts_blob_len` is the tail — directory + every per-column
        // doc-length array + their CRCs — fetched in one range GET, not
        // the whole blob (the FST is a separate range above; postings
        // stay lazy).
        //
        // Both ranges are known exactly once the header is parsed and
        // neither depends on the other, so they fire **concurrently**:
        // the FTS open spends 2 serial RTTs (header, then this parallel
        // pair) instead of 3. On a warm/in-memory source both resolve
        // through the sync zero-copy path at no cost. The doc-length
        // tail is fetched whole (one range) rather than dir-then-arrays,
        // keeping the open-time GET count minimal and avoiding
        // per-column range calls during metadata decode.
        let (fst_region, doc_lengths_tail) = futures::try_join!(
            fetch_lazy_range(source.as_ref(), header_size..postings_offset, "fts/dict"),
            fetch_lazy_range(
                source.as_ref(),
                doc_lengths_table_offset..fts_blob_len,
                "fts/doc_lengths_tail",
            ),
        )?;

        let mut overlay = PrefetchedSource::new(source);
        overlay.install(0, header);
        overlay.install(header_size as u64, fst_region);
        overlay.install(doc_lengths_table_offset as u64, doc_lengths_tail);

        Self::open_with_source(Source::Lazy(Arc::new(overlay)), columns_json, opts)
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
        if source_len < FTS_HEADER_SIZE {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }
        let header = fetch_source_range(&source, 0..FTS_HEADER_SIZE, "fts header")?;

        // Magic check.
        if &header[0..MAGIC_BYTES] != format::fts::MAGIC {
            return Err(FtsError::Read(ReadError::BadMagic {
                section: "fts",
                expected: format::fts::MAGIC,
                actual: header[0..MAGIC_BYTES].to_vec(),
            }));
        }

        // Version check. v1 = no positions (48-byte header); v2 adds
        // the positions-region offset at [48..56] and a positions
        // region between the postings and the doc-lengths directory.
        let version = read_u32_le(&header[hdr::VERSION_OFF..hdr::VERSION_OFF + U32_BYTES]);
        let positional_blob = match version {
            v if v == format::fts::VERSION_V1_LEGACY => false,
            v if v == format::fts::VERSION_V2 => true,
            _ => {
                return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                    "fts section version {version}"
                ))));
            }
        };
        let header_size = match positional_blob {
            true => format::fts::HEADER_SIZE_V2,
            false => FTS_HEADER_SIZE,
        };
        if source_len < header_size {
            return Err(FtsError::Read(ReadError::MissingKv("fts header")));
        }

        let n_columns =
            read_u32_le(&header[hdr::N_COLUMNS_OFF..hdr::N_COLUMNS_OFF + U32_BYTES]) as usize;
        let n_docs = read_u32_le(&header[hdr::N_DOCS_OFF..hdr::N_DOCS_OFF + U32_BYTES]);
        let n_terms_total = read_u32_le(&header[hdr::N_TERMS_OFF..hdr::N_TERMS_OFF + U32_BYTES]);
        let fst_offset =
            read_u64_le(&header[hdr::FST_OFFSET_OFF..hdr::FST_OFFSET_OFF + U64_BYTES]) as usize;
        let postings_offset =
            read_u64_le(&header[hdr::POSTINGS_OFFSET_OFF..hdr::POSTINGS_OFFSET_OFF + U64_BYTES])
                as usize;
        let doc_lengths_table_offset =
            read_u64_le(&header[hdr::DOC_LENGTHS_DIR_OFF..hdr::DOC_LENGTHS_DIR_OFF + U64_BYTES])
                as usize;
        // The v2 extension lives past the 48 bytes fetched above; on
        // the lazy path it resolves from the prefetch overlay.
        let positions_offset: Option<usize> = match positional_blob {
            true => {
                let ext = fetch_source_range(
                    &source,
                    FTS_HEADER_SIZE..format::fts::HEADER_SIZE_V2,
                    "fts header ext",
                )?;
                Some(read_u64_le(&ext[0..U64_BYTES]) as usize)
            }
            false => None,
        };

        // Bounds-check every offset against the blob length before
        // any slice indexing. A single byte flip in the header can
        // corrupt these into multi-GB values; without this check
        // they propagate as out-of-range slice indices and panic
        // before the CRC verification can reject the corruption.
        //
        // The `< +4` checks (rather than `<= +4`) admit the legal
        // empty-region case: when every term takes the df=1 inline-FST
        // short-circuit, the postings region body is zero bytes and
        // only the trailing 4-byte CRC32C(empty) sits between
        // `postings_offset` and `doc_lengths_table_offset`.
        let postings_end = positions_offset.unwrap_or(doc_lengths_table_offset);
        if fst_offset < header_size
            || postings_offset < fst_offset + 4
            || postings_end < postings_offset + 4
            || doc_lengths_table_offset < postings_end
            || doc_lengths_table_offset > source_len
            || positions_offset.is_some_and(|po| doc_lengths_table_offset < po + 4)
        {
            return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                "fts header offsets out of range: fst={fst_offset}, postings={postings_offset}, \
                 positions={positions_offset:?}, doc_lengths={doc_lengths_table_offset}, \
                 blob_len={}",
                source_len
            ))));
        }

        // Region lengths aren't stored explicitly (each region ends
        // with its CRC32C). Compute from the surrounding offsets —
        // postings end where the positions region begins (or the
        // doc-lengths directory on a v1 blob), positions end where the
        // directory begins.
        let fst_range = fst_offset..postings_offset.saturating_sub(4); // strip CRC
        let postings_range = postings_offset..postings_end.saturating_sub(4); // strip CRC
        let positions_range: Option<Range<usize>> =
            positions_offset.map(|po| po..doc_lengths_table_offset.saturating_sub(4));

        // Verify FST CRC32C (4 bytes after fst body).
        if opts.verify_crc {
            let fst_crc_bytes = fetch_source_range(
                &source,
                postings_offset.saturating_sub(4)..postings_offset,
                "fts/dict crc",
            )?;
            let fst_crc_expected = read_u32_le(&fst_crc_bytes);
            let fst_bytes = fetch_source_range(&source, fst_range.clone(), "fts/dict")?;
            let fst_crc_actual = crc32c(&fst_bytes);
            if fst_crc_expected != fst_crc_actual {
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

        // Verify positions region CRC32C (v2 blobs only).
        if opts.verify_crc
            && let Some(pos_range) = &positions_range
        {
            let crc_pos = doc_lengths_table_offset.saturating_sub(4);
            let crc_bytes = fetch_source_range(
                &source,
                crc_pos..doc_lengths_table_offset,
                "fts/positions crc",
            )?;
            let crc_expected = read_u32_le(&crc_bytes);
            let pos_bytes = fetch_source_range(&source, pos_range.clone(), "fts/positions")?;
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
            FtsError::Read(ReadError::MalformedVersion(format!(
                "inf.fts.columns JSON: {e}"
            )))
        })?;
        if cols.len() != n_columns {
            return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                "inf.fts.columns has {} entries, header says {}",
                cols.len(),
                n_columns
            ))));
        }

        // Read doc-lengths directory: n_columns × 16-byte entries + 4-byte CRC.
        //
        // On the lazy open path this directory — and every per-column
        // array fetched below — falls inside the
        // `[doc_lengths_table_offset..fts_blob_len]` tail that
        // `open_lazy` already fetched in one GET and installed in the
        // overlay, so these `fetch_source_range` calls resolve from the
        // overlay with **no** per-column GETs. On the eager path the
        // whole subsection is in memory, so they are zero-copy slices.
        let dir_size = n_columns * DOC_LENGTHS_ENTRY_SIZE;
        let dir_end = doc_lengths_table_offset + dir_size;
        if dir_end + 4 > source_len {
            return Err(FtsError::Read(ReadError::MalformedVersion(
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
            let entry_off = i * DOC_LENGTHS_ENTRY_SIZE;
            let column_id = u32::from_le_bytes([
                dir_bytes[entry_off],
                dir_bytes[entry_off + 1],
                dir_bytes[entry_off + 2],
                dir_bytes[entry_off + 3],
            ]);
            let doc_lengths_offset =
                read_u64_le(&dir_bytes[entry_off + 4..entry_off + 12]) as usize;
            let avgdl_x1000 = read_u32_le(&dir_bytes[entry_off + 12..entry_off + 16]) as u64;

            // Verify column_id matches the JSON's positional column_id.
            if column_id != i as u32 {
                return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                    "doc-lengths directory entry {i} has column_id {column_id}"
                ))));
            }

            // Per-column doc-lengths array: 4 * n_docs bytes + 4-byte CRC.
            // `doc_lengths_offset` lies within the prefetched doc-lengths
            // tail, so on the lazy path this resolves from the overlay
            // (see the directory comment above) — no per-column GET.
            let array_byte_len = 4 * n_docs as usize;
            let array_end = doc_lengths_offset + array_byte_len;
            if array_end + 4 > source_len {
                return Err(FtsError::Read(ReadError::MalformedVersion(format!(
                    "doc-lengths array {i} runs past blob end"
                ))));
            }
            let array_region = fetch_source_range(
                &source,
                doc_lengths_offset..array_end + 4,
                "fts/doc_lengths_array",
            )?;
            if opts.verify_crc {
                let array_crc_expected =
                    read_u32_le(&array_region[array_byte_len..array_byte_len + 4]);
                let array_crc_actual = crc32c(&array_region[..array_byte_len]);
                if array_crc_expected != array_crc_actual {
                    return Err(FtsError::Read(ReadError::ChecksumMismatch {
                        section: "fts/doc_lengths_array",
                        column: format!(" (column '{}')", col_cfg.name),
                    }));
                }
            }

            let avgdl = (avgdl_x1000 as f32) / format::fts::AVGDL_FIXED_POINT_SCALE;
            // Per-doc length normalizer, byte-quantized (see `NormTable`).
            // For avgdl == 0 (empty column) this is an empty table; it'll
            // never be indexed since `search` short-circuits.
            let n = n_docs as usize;
            let dl_norm_k1 = NormTable::new(
                (0..n).map(|d| read_u32_le(&array_region[d * 4..d * 4 + 4])),
                n,
                avgdl,
            );
            let tokenizer = tokenizer_for_name(&col_cfg.tokenizer).ok_or_else(|| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "inf.fts.columns: unknown tokenizer {:?} for column {:?}",
                    col_cfg.tokenizer, col_cfg.name
                )))
            })?;
            columns.push(ColumnMeta {
                name: col_cfg.name.clone(),
                doc_lengths_range: doc_lengths_offset..array_end,
                avgdl,
                dl_norm_k1,
                positions: col_cfg.positions,
                tokenizer,
            });
            column_id_by_name.insert(col_cfg.name.clone(), i as u32);
        }

        Ok(FtsReader {
            source,
            n_docs,
            n_terms_total,
            fst_range,
            postings_range,
            positions_range,
            columns,
            column_id_by_name,
        })
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

    /// Tokenizer configured for `column`, for tokenizing query text so
    /// it matches how the column was indexed. Errors if `column` is not
    /// a registered FTS column.
    pub fn column_tokenizer(&self, column: &str) -> Result<Arc<dyn Tokenizer>, FtsError> {
        let id = self.resolve_column_id(column)?;
        Ok(Arc::clone(&self.columns[id as usize].tokenizer))
    }

    fn dict_bytes(&self) -> Result<Bytes, FtsError> {
        fetch_source_range(&self.source, self.fst_range.clone(), "fts/dict")
    }

    /// Async FST-dictionary fetch for the query path. Resolves
    /// zero-copy for in-memory / warm sources; for a cold `Lazy`
    /// source it `await`s the object-store range on the caller's
    /// runtime (no sync bridge).
    async fn dict_bytes_async(&self) -> Result<Bytes, FtsError> {
        self.source
            .range_async(self.fst_range.clone())
            .await
            .map_err(|e| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "fts/dict range fetch failed: {e}"
                )))
            })
    }

    /// Fetch the complete byte range of each requested term — metadata
    /// header (20 bytes) + skip table + encoded posting blocks — in
    /// parallel. `terms` are `(metadata_offset, postings_length)` pairs
    /// stored in the FST (`FstValue::Pfor`); the
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
    /// Whenever the FST value carries the length, this is a single
    /// range batch. The metadata header remains in the returned bytes
    /// for validation and cursor construction.
    ///
    /// A `None` length means the FST value held `PFOR_LENGTH_UNKNOWN`;
    /// its real length is read from the header first.
    async fn fetch_term_postings(
        &self,
        terms: &[(usize, Option<usize>)],
    ) -> Result<Vec<Bytes>, FtsError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        // Recover the lengths the FST could not express. `postings_length`
        // sits at offset 12 in both header strides, so 20 bytes covers it.
        let probe_ranges: Vec<(usize, usize)> = terms
            .iter()
            .filter(|(_, len)| len.is_none())
            .map(|&(metadata_offset, _)| (metadata_offset, TERM_META_SIZE))
            .collect();
        let probed = self.fetch_ranges(&probe_ranges).await?;

        let mut resolved: Vec<(usize, usize)> = Vec::with_capacity(terms.len());
        let mut next_probe = 0usize;
        for &(metadata_offset, slot_length) in terms {
            let postings_length = match slot_length {
                Some(length) => length,
                None => {
                    let header = probed.get(next_probe).ok_or_else(|| {
                        FtsError::Read(ReadError::MalformedVersion(
                            "fetched fewer term metadata headers than probed".into(),
                        ))
                    })?;
                    next_probe += 1;
                    header_postings_length(header.as_ref())?
                }
            };
            resolved.push((metadata_offset, postings_length));
        }

        self.fetch_ranges(&resolved).await
    }

    /// Fetch each `(metadata_offset, length)` range from the postings
    /// region in parallel, coalescing adjacent ranges, and return the
    /// per-request slices in input order. The byte-level half of
    /// [`Self::fetch_term_postings`]; every length here is already
    /// known to be real.
    async fn fetch_ranges(&self, terms: &[(usize, usize)]) -> Result<Vec<Bytes>, FtsError> {
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let base = self.postings_range.start;
        let region_len = self.postings_range.len();

        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(terms.len());
        for &(m, postings_length) in terms {
            if postings_length < TERM_META_SIZE || m + postings_length > region_len {
                return Err(FtsError::Read(ReadError::MalformedVersion(
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
            .map_err(|e| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "fts/postings term body range fetch failed: {e}"
                )))
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
        let region = self.positions_range.as_ref().ok_or_else(|| {
            FtsError::Read(ReadError::MalformedVersion(
                "positional term in a blob with no positions region".into(),
            ))
        })?;
        let base = region.start;
        let region_len = region.len();
        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(terms.len());
        for &(off, len) in terms {
            let off = off as usize;
            let len = len as usize;
            if off + len > region_len {
                return Err(FtsError::Read(ReadError::MalformedVersion(
                    "term positions range runs past positions region".into(),
                )));
            }
            ranges.push(base + off..base + off + len);
        }
        self.source
            .get_ranges_parallel_async(&ranges)
            .await
            .map_err(|e| {
                FtsError::Read(ReadError::MalformedVersion(format!(
                    "fts/positions term range fetch failed: {e}"
                )))
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
    /// The second element counts the FST-dictionary ranges the builds
    /// requested (one per `build_term_cursors` call plus one per inline
    /// phrase member's position recovery) — real byte-source ranges on
    /// every query, tallied by the caller into the planned count.
    async fn build_atom_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
        phrases: &[Vec<String>],
        global_idf: Option<&GlobalTermIdf>,
    ) -> Result<(Vec<Option<AnyCursor>>, u64), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        if !phrases.is_empty() && !col_meta.positions {
            return Err(FtsError::PositionsUnavailable {
                column: col_meta.name.clone(),
            });
        }
        let mut dict_ranges = 0u64;
        let mut out: Vec<Option<AnyCursor>> = Vec::with_capacity(terms.len() + phrases.len());
        for term in terms {
            let mut cursors = self
                .build_term_cursors(column_id, &[term], global_idf)
                .await?;
            dict_ranges += 1;
            out.push(cursors.pop().map(AnyCursor::Term));
        }
        for phrase in phrases {
            let member_refs: Vec<&str> = phrase.iter().map(|t| t.as_str()).collect();
            // A phrase's score is Σ member idf (see `PhraseCursor::new`), so
            // globalizing the members' idf globalizes the phrase — the
            // per-member rescale ratio cancels out of the phrase's tf/length
            // bound. Build members with the same `global_idf` as bare terms.
            let cursors = self
                .build_term_cursors(column_id, &member_refs, global_idf)
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
            // position from the FST slot the tf-reinterpretation
            // dropped during cursor build.
            let mut positional: Vec<(Option<TermMeta>, Option<u32>)> =
                Vec::with_capacity(cursors.len());
            for (cursor, term) in cursors.iter().zip(&member_refs) {
                match cursor.bytes.is_empty() {
                    false => {
                        let term_meta = TermMeta::parse(cursor.bytes.as_ref(), 0, true)?;
                        positional.push((Some(term_meta), None));
                    }
                    true => {
                        dict_ranges += 1;
                        let fst_bytes = self.dict_bytes_async().await?;
                        let dict = DictReader::open(&fst_bytes).map_err(|e| {
                            FtsError::Read(ReadError::MalformedVersion(format!(
                                "FST parse failed: {e}"
                            )))
                        })?;
                        let key = make_key(&col_meta.name, term);
                        let packed = dict
                            .lookup(&key)
                            .expect("inline member cursor was built from this dict");
                        let position = match FstValue::unpack(packed) {
                            FstValue::Inline { tf: slot, .. } => slot,
                            FstValue::Pfor { .. } => {
                                unreachable!("inline cursor from a PFOR FST value")
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
            let positions = self.fetch_term_positions(&pos_ranges).await?;
            out.push(Some(AnyCursor::Phrase(PhraseCursor::new(
                cursors, positions, positional,
            )?)));
        }
        Ok((out, dict_ranges))
    }

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

    /// Unranked doc-at-a-time walk over heterogeneous atoms, calling
    /// `on_doc` for every matching doc in ascending order. `And` walks
    /// the atoms' intersection (a phrase atom's own verification is
    /// part of its cursor); `Or` walks their union. The shared spine
    /// of the phrase-aware `token_match` / `count` entries.
    fn walk_atoms_match(
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
    fn resolve_column_id(&self, column: &str) -> Result<u32, FtsError> {
        self.column_id_by_name
            .get(column)
            .copied()
            .ok_or_else(|| FtsError::UnknownColumn(column.to_string()))
    }

    /// Walk the FST and collect every term registered under
    /// `column`, in lex order. Used to populate per-superfile FTS
    /// skip-pruning summaries (term-presence bloom + lex term
    /// range) at commit time.
    ///
    /// Returns an empty `Vec` if `column` is not registered as
    /// an FTS column in this superfile. Cost is O(terms in column)
    /// FST decodes; intended to be called once per (superfile,
    /// column) at commit time, not on the query hot path.
    pub fn iter_column_terms(&self, column: &str) -> Result<Vec<Vec<u8>>, FtsError> {
        self.iter_terms_with_prefix(column, b"")
    }

    /// Stream a column's postings for the FTS compaction merge: for every term
    /// (lex order) and every doc in its posting list (doc_ids ascending),
    /// invoke `emit(term_bytes, local_doc_id, tf, positions)`. Reuses the
    /// query-path [`TermCursor`] block decode for doc_ids/tfs and the
    /// positional `decode_run` for positions, so what is streamed is exactly
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
        mut emit: impl FnMut(&[u8], u32, u32, &[u32]) -> Result<(), FtsError>,
    ) -> Result<(), FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let positional = col_meta.positions;
        let n_docs = u64::from(self.n_docs);
        let column_name = col_meta.name.clone();
        let region_base = self.postings_range.start;
        let positions_region = self.positions_range.clone();

        let fst_bytes = self.dict_bytes()?;
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;

        // Column-scoped FST keys are `column_name <FST_SEPARATOR> term`;
        // `iter_prefix` yields `(key, packed_value)` in lex term order, so we
        // read the posting metadata straight from the value — no re-lookup.
        let mut column_prefix = column_name.as_bytes().to_vec();
        column_prefix.push(FST_SEPARATOR);
        let prefix_len = column_prefix.len();

        // Reused across (term, doc) to hold the decoded position run.
        let mut positions_buf: Vec<u32> = Vec::new();

        for (key, packed) in dict.iter_prefix(&column_prefix) {
            let term = &key[prefix_len..];
            match FstValue::unpack(packed) {
                FstValue::Inline { doc_id, tf } => {
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
                FstValue::Pfor {
                    metadata_offset,
                    postings_length_hint,
                } => {
                    let start = region_base + metadata_offset as usize;
                    let postings_length = match postings_length_hint {
                        Some(len) => len as usize,
                        None => {
                            let header = fetch_source_range(
                                &self.source,
                                start..start + TERM_META_SIZE,
                                "fts/merge header",
                            )?;
                            header_postings_length(header.as_ref())?
                        }
                    };
                    let term_bytes = fetch_source_range(
                        &self.source,
                        start..start + postings_length,
                        "fts/merge postings",
                    )?;

                    // For a positional column, this term's position runs live
                    // contiguously in the positions region at `positions_offset`,
                    // one `decode_run` per doc in posting order. Read the slice
                    // once and walk it in lockstep with the doc cursor.
                    let position_bytes = if positional {
                        let meta = TermMeta::parse(term_bytes.as_ref(), 0, true)?;
                        let region = positions_region.as_ref().ok_or_else(|| {
                            FtsError::Read(ReadError::MalformedVersion(
                                "positional column missing a positions region".into(),
                            ))
                        })?;
                        let pstart = region.start + meta.positions_offset as usize;
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

                    let mut cursor = TermCursor::new(term_bytes, n_docs, positional, None, false)?;
                    while !cursor.is_exhausted() {
                        while cursor.pos < cursor.block_n {
                            let doc_id = cursor.block_doc_ids[cursor.pos];
                            let tf = cursor.block_tfs[cursor.pos];
                            let positions: &[u32] = match &position_bytes {
                                Some(bytes) => {
                                    positions_buf.clear();
                                    decode_run(bytes.as_ref(), &mut pos_at, tf, &mut positions_buf)
                                        .ok_or_else(|| {
                                            FtsError::Read(ReadError::MalformedVersion(
                                                "truncated position run in merge read".into(),
                                            ))
                                        })?;
                                    &positions_buf
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

    /// Read a column's stored per-doc lengths (token counts), one `u32` per
    /// local doc-id in `0..n_docs`. The FTS compaction merge carries these
    /// forward (with the input's doc-id remap) rather than recomputing them
    /// from text. These are the already-clamped values written at build time.
    pub(crate) fn read_doc_lengths(&self, column_id: u32) -> Result<Vec<u32>, FtsError> {
        let n = self.n_docs as usize;
        let range = self.columns[column_id as usize].doc_lengths_range.clone();
        let bytes = fetch_source_range(&self.source, range, "fts/merge doc_lengths")?;
        let region = bytes.as_ref();
        if region.len() < n * U32_BYTES {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "doc-lengths region shorter than n_docs entries".into(),
            )));
        }
        Ok((0..n)
            .map(|d| read_u32_le(&region[d * U32_BYTES..d * U32_BYTES + U32_BYTES]))
            .collect())
    }

    /// Walk the FST and collect every term registered under
    /// `column` whose bytes begin with `term_prefix`, in lex order.
    ///
    /// Mirrors [`Self::iter_column_terms`] but bounds the walk to a
    /// prefix range instead of the whole column. Used by
    /// [`SuperfileReader::bm25_search_prefix`] to expand a
    /// prefix into the concrete terms list before delegating to
    /// `search` in OR mode.
    ///
    /// `term_prefix` is the prefix as it appears in the FST — the
    /// caller is responsible for any tokenizer-level normalization
    /// (e.g. ASCII-lowercasing for the v1 tokenizer). Returns an
    /// empty `Vec` if `column` is not registered or no terms match
    /// the prefix.
    pub fn iter_terms_with_prefix(
        &self,
        column: &str,
        term_prefix: &[u8],
    ) -> Result<Vec<Vec<u8>>, FtsError> {
        if !self.column_id_by_name.contains_key(column) {
            return Ok(Vec::new());
        }
        let mut full_prefix = column.as_bytes().to_vec();
        full_prefix.push(FST_SEPARATOR);
        let column_prefix_len = full_prefix.len();
        full_prefix.extend_from_slice(term_prefix);
        let fst_bytes = self
            .dict_bytes()
            .expect("FST bytes must be available for term iteration");
        let dict = DictReader::open(&fst_bytes).map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "FST parse failed: {e}"
            )))
        })?;
        let pairs = dict.iter_prefix(&full_prefix);
        Ok(pairs
            .into_iter()
            .map(|(key, _)| key[column_prefix_len..].to_vec())
            .collect())
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
                self.build_term_cursors(column_id, lists.negatives, None)
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
                .build_term_cursors(column_id, lists.shoulds, lists.global_idf)
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
            .build_term_cursors(column_id, lists.musts, lists.global_idf)
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
            .build_term_cursors(column_id, lists.shoulds, lists.global_idf)
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
            self.build_term_cursors(column_id, terms, global_idf)
                .await?
        };
        Ok(OrCursorSet { column_id, cursors })
    }

    /// Multi-term OR over `[doc_id_start, doc_id_end)` against prebuilt
    /// cursors — the ranged fan-out's per-slice call;
    /// [`Self::search_or_range_pretokenized_with_floor`] delegates here.
    /// The ranged path carries no negation in v1.
    ///
    /// Kernel choice mirrors `dispatch_or_algo` instead of
    /// hardcoding MaxScore+BMM: on a broad OR over uniform-upper-bound
    /// terms BMM cannot prune (every block max ties), so it degrades to
    /// per-doc min-scan bookkeeping over ~the whole union — the exact
    /// shape `run_windowed_union` exists for, and it is natively ranged.
    /// Hardcoding BMM here made the SAME query run a different kernel
    /// depending on whether the fan-out sliced (few large superfiles,
    /// i.e. post-compaction) or not (many small ones, pre-compaction) —
    /// measured at 1M as the 11-24x post-compact broad-OR regression.
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
        if prefer_windowed_union(&cursors) {
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
        let term_meta = TermMeta::parse(postings, metadata_offset, col_meta.positions)?;

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
    async fn build_term_cursors(
        &self,
        column_id: u32,
        terms: &[&str],
        global_idf: Option<&GlobalTermIdf>,
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
                    )?);
                }
            }
        }
        Ok(cursors)
    }

    /// Multi-term OR via WAND + BlockMaxWAND.
    ///
    /// Algorithm: maintain a `TermCursor` per query term. Each
    /// iteration sorts cursors by current `doc_id`, computes the
    /// **WAND pivot** (smallest j such that the prefix-sum of
    /// term-level upper bounds exceeds the kth-best score), then
    /// applies the **BMW augmentation** (per-block UBs across the
    /// pivot prefix). If the pivot doc can't beat the threshold even
    /// with full per-block UBs, advance the leftmost cursor past the
    /// smallest block-end among the prefix; otherwise score the doc
    /// and advance.
    ///
    /// Reference: Ding & Suel, "Faster Top-k Document Retrieval Using
    /// Block-Max Indexes", SIGIR 2011.
    ///
    /// Result invariants: top-k by descending BM25 score, ties broken
    /// by ascending doc_id.
    ///
    /// Production path for small-`k`, **floor-free** 2-term ORs (see
    /// `dispatch_or_algo`), and the `search_with_algo_for_bench`
    /// entry point. Cursor construction is shared with the BMM path.
    ///
    /// Carries **no cross-segment floor and no exclude filter** — the
    /// dispatcher only routes here when both are absent (`floor_eff` is
    /// `NEG_INFINITY` and the query has no negation). Seeding WAND's pivot
    /// threshold from a finite floor was tried and reverted: it skipped
    /// blocks that still held qualifying docs at higher floors (caught by
    /// `wand_bmw_2term_no_floor_agrees_with_bmm` vs the floored BMM). When
    /// a floor is live, MaxScore handles it instead.
    fn run_wand_bmw(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // `search_multi` passes `k = usize::MAX` to gather every
        // matching doc before weighting across columns; cap initial
        // capacity at n_docs (the upper bound on distinct doc_ids in
        // the heap) so we don't try to allocate `usize::MAX * size_of::<TopKEntry>()`.
        // The BinaryHeap grows on demand if needed.
        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        // Reused index buffer to avoid per-iteration allocation.
        let mut idx: Vec<usize> = Vec::with_capacity(cursors.len());

        loop {
            // Drop exhausted cursors. Doing this in-place keeps idx
            // valid for the next iteration without re-allocation.
            cursors.retain(|c| !c.is_exhausted());
            if cursors.is_empty() {
                break;
            }

            // Sort cursor indices ascending by current doc_id.
            idx.clear();
            idx.extend(0..cursors.len());
            // Per-iteration WAND cursor reorder; pdqsort because
            // cursors hold distinct current doc_ids in the heap
            // state used by this scan.
            idx.sort_unstable_by_key(|&i| cursors[i].current_doc_id());

            // WAND pivot: smallest j such that the prefix-sum of
            // *term-level* upper bounds exceeds the threshold.
            let mut accum_term_ub: f32 = 0.0;
            let mut pivot_j: Option<usize> = None;
            for (j, &ci) in idx.iter().enumerate() {
                accum_term_ub += cursors[ci].term_max_bm25;
                if accum_term_ub > threshold {
                    pivot_j = Some(j);
                    break;
                }
            }

            let Some(mut pivot_j) = pivot_j else {
                // Sum of all remaining term UBs ≤ threshold: no
                // future doc can beat the heap. Done.
                break;
            };

            let pivot_doc = cursors[idx[pivot_j]].current_doc_id();

            // Extend the pivot prefix to include any cursors past
            // `pivot_j` that are also at `pivot_doc`. They contribute
            // to both the BMW upper-bound sum and the actual score,
            // so missing them under-counts the BMW UB and could
            // trigger an incorrect skip.
            while pivot_j + 1 < idx.len() && cursors[idx[pivot_j + 1]].current_doc_id() == pivot_doc
            {
                pivot_j += 1;
            }

            // BMW augmentation: sum of per-block upper bounds for the
            // block that would contain `pivot_doc` in each prefix
            // cursor. Lagging cursors' current decoded block is for
            // an earlier doc whose UB doesn't bound their
            // contribution at pivot_doc; `shallow_advance_block_to`
            // moves the lightweight inspect-block pointer to the
            // pivot-doc block without decoding, then
            // `inspect_block_max_bm25` reads that block's UB.
            let mut accum_block_ub: f32 = 0.0;
            for &ci in &idx[..=pivot_j] {
                cursors[ci].shallow_advance_block_to(pivot_doc);
                accum_block_ub += cursors[ci].inspect_block_max_bm25();
            }

            if accum_block_ub <= threshold {
                // No doc in [pivot_doc, smallest_pivot_block_end]
                // can beat the kth-best score. Advance the leftmost
                // cursor to the next interesting doc — either one
                // past the smallest pivot-block-end among the prefix,
                // or a suffix cursor's current doc if that's closer.
                // The suffix cap matters for recall: without it,
                // leftmost can leap multiple blocks past pivot_doc
                // and overshoot a doc one of the suffix cursors is
                // sitting at, leaving that doc with too few cursors
                // ever positioned on it to score correctly.
                let mut target = u32::MAX;
                for &ci in &idx[..=pivot_j] {
                    let last = cursors[ci].inspect_block_last_doc_id();
                    if last < target {
                        target = last;
                    }
                }
                let mut effective_target = target.saturating_add(1);
                for &ci in &idx[pivot_j + 1..] {
                    let d = cursors[ci].current_doc_id();
                    if d < effective_target {
                        effective_target = d;
                    }
                }
                cursors[idx[0]].skip_to(effective_target);
                continue;
            }

            // Align every lagging cursor in the pivot prefix to
            // `pivot_doc` so its contribution is included in this
            // doc's score. If any cursor's posting list doesn't
            // contain `pivot_doc` (the seek lands past it), abandon
            // this pivot — re-sort and re-pivot next iteration. This
            // is the WAND alignment step (Ding & Suel §3); without
            // it, lagging cursors that DO have pivot_doc in their
            // posting list get advanced past it on subsequent
            // iterations without ever contributing to its score,
            // producing under-counted scores and missing top-k hits.
            let mut aligned = true;
            for &ci in &idx[..=pivot_j] {
                if cursors[ci].current_doc_id() < pivot_doc {
                    cursors[ci].skip_to(pivot_doc);
                    if cursors[ci].current_doc_id() != pivot_doc {
                        aligned = false;
                        break;
                    }
                }
            }
            if !aligned {
                continue;
            }

            // All prefix cursors are at pivot_doc. Score it by summing
            // contributions from every cursor at pivot_doc (cursors
            // beyond the prefix may also be at pivot_doc — they
            // contribute too). SIMD-pack up to 4 cursors per scoring
            // call.
            let norm = dl_norm_k1.get(pivot_doc);
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in &cursors {
                if cursor.current_doc_id() == pivot_doc {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                }
            }
            if packed > 0 {
                score += bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Update heap.
            if heap.len() < k {
                heap.push(TopKEntry(score, pivot_doc));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if let Some(TopKEntry(min_score, _)) = heap.peek()
                && score > *min_score
            {
                heap.pop();
                heap.push(TopKEntry(score, pivot_doc));
                threshold = heap.peek().expect("non-empty").0;
            }

            // Advance every cursor at pivot_doc (the prefix, plus any
            // cursors past the prefix that happened to be at it).
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == pivot_doc {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR via Block-Max MaxScore (BMM).
    ///
    /// Algorithm sketch (Turtle & Flood 1995, Strohman & Croft 2007;
    /// the "Block-Max" augmentation per Petri & Moffat 2017):
    ///
    ///   1. Sort cursors in *descending* `term_max_bm25`.
    ///   2. Compute suffix sums: `partial_max[i] = sum_{j>=i} cursors[j].term_max_bm25`.
    ///   3. Partition into **essential** prefix `cursors[0..f]` and
    ///      **non-essential** suffix `cursors[f..n]` where
    ///      `f = min{ i : partial_max[i] <= threshold }`. A doc that
    ///      appears only in non-essential cursors has max-possible
    ///      score `partial_max[f] <= threshold` and can't make top-k.
    ///   4. Find next candidate doc as the smallest `current_doc_id`
    ///      among essential cursors. (Non-essential cursors are
    ///      skipped *to* the candidate, not iterated for new candidates.)
    ///   5. Apply BMW-style block-skip on the leftmost essential: if
    ///      `leftmost_block_ub + sum_other_term_ubs <= threshold`,
    ///      no doc in the leftmost's current block can beat top-k —
    ///      jump leftmost past its block.
    ///   6. Score: sum essential contributions, then run the
    ///      non-essential loop with **block-level** early termination
    ///      using `current_block_max_bm25` of the remaining cursors.
    ///   7. Update heap; recompute `f` from the new threshold; repeat.
    ///
    /// **When is BMM better than WAND+BMW?** When query terms have
    /// similar upper bounds (3+ same-rank Zipfian terms is the
    /// canonical case) — WAND's pivot moves around because no single
    /// cursor dominates, while MaxScore stably partitions essential
    /// vs non-essential. WAND wins when one term has much higher UB
    /// (rare + common); the partition collapses to a single
    /// essential cursor anyway and WAND's pivot is tighter.
    ///
    /// The router [`Self::dispatch_or_algo`] picks between
    /// the two using a UB-spread heuristic. Both algorithms share
    /// cursor construction via [`Self::build_term_cursors`] so the
    /// router doesn't pay for cursor work twice.
    fn run_max_score_bmm(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        self.run_max_score_bmm_range(column_id, cursors, k, 0, u32::MAX, filter, floor_eff)
    }

    /// Multi-term AND via leapfrog intersection over the skip table.
    ///
    /// The smallest-df cursor is the leader: every matching doc must
    /// be in its posting list. For each leader doc, every other
    /// cursor runs `skip_to(candidate)` — a skip-table-driven jump
    /// that decodes at most one block per call (and zero if the
    /// target lies in the already-decoded block). If any cursor
    /// lands past the candidate, that doc isn't in the intersection;
    /// the candidate is bumped to the new high-water mark and the
    /// remaining cursors re-skip. When all cursors converge on the
    /// same doc, the BM25 contribution from each is summed.
    ///
    /// Cost is bounded by `min_df` leader steps × `n_terms` skip_to
    /// calls, with each skip_to a constant-or-O(log) skip-table walk.
    /// The old `run_and` did a full PFOR decode of every term's full
    /// posting list (dominated by the largest list, e.g. ~hundreds of
    /// K postings for a common Zipfian term) followed by a HashMap
    /// intersection — orders of magnitude more work than this when
    /// any term is rare.
    fn run_and_intersect(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // Smallest-df cursor at index 0 = leader. The remaining order
        // doesn't matter for correctness but ascending-df reduces the
        // expected number of leapfrog bumps per candidate.
        cursors.sort_by_key(|c| c.block_count());

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut sink = ScoreSink {
            heap: &mut heap,
            k,
            filter,
            floor_eff,
        };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        Ok(drain_top_k_desc(heap))
    }

    /// Ranked must+should walk: the match set is the musts'
    /// intersection (driven by the same flat-merge as
    /// [`run_and_intersect`](Self::run_and_intersect), so the two
    /// always agree on which docs match), and each matching doc's
    /// score additionally collects every should term that lands on it.
    /// Shoulds never affect matching — a doc containing every must and
    /// no should still matches, with its must-only score.
    fn run_must_should(
        &self,
        column_id: u32,
        mut must_cursors: Vec<TermCursor>,
        should_cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        debug_assert!(
            !must_cursors.is_empty() && !should_cursors.is_empty(),
            "dispatch routes empty-side shapes to the AND/OR kernels"
        );
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        must_cursors.sort_by_key(|c| c.block_count());

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let should_ub = should_cursors.iter().map(|c| c.term_max_bm25).sum();
        let mut sink = MustShouldSink {
            heap: &mut heap,
            k,
            filter,
            floor_eff,
            shoulds: should_cursors,
            should_ub,
            dl_norm_k1,
        };
        self.and_flat_merge(&mut must_cursors, dl_norm_k1, &mut sink);
        Ok(drain_top_k_desc(heap))
    }

    /// Unranked multi-term AND: the matching doc ids in ascending order
    /// via the block flat-merge in [`and_flat_merge`](Self::and_flat_merge),
    /// with no BM25 scoring and no top-k heap. Because it shares that
    /// traversal with the ranked [`run_and_intersect`](Self::run_and_intersect),
    /// the two always agree on which docs match, and an unranked count
    /// over high-frequency terms costs the same posting-list work as the
    /// ranked search minus the scoring.
    fn collect_and_intersect(&self, column_id: u32, mut cursors: Vec<TermCursor>) -> Vec<u32> {
        if cursors.is_empty() {
            return Vec::new();
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        cursors.sort_by_key(|c| c.block_count());
        let mut sink = CollectSink { out: Vec::new() };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        sink.out
    }

    /// Unranked multi-term AND **count**: the size of the intersection
    /// via the same flat-merge as [`collect_and_intersect`](Self::collect_and_intersect),
    /// but through a [`CountSink`] that tallies hits instead of
    /// collecting them — no `Vec<u32>` materialized.
    fn count_and_intersect(&self, column_id: u32, mut cursors: Vec<TermCursor>) -> u64 {
        if cursors.is_empty() {
            return 0;
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;
        cursors.sort_by_key(|c| c.block_count());
        let mut sink = CountSink { n: 0 };
        self.and_flat_merge(&mut cursors, dl_norm_k1, &mut sink);
        sink.n
    }

    /// Dispatch to the 2-term specialization or the general `n >= 3`
    /// (and `n == 1`) flat-merge. The 2-term shape walks the two sorted
    /// `block_doc_ids` arrays with two index pointers instead of calling
    /// `skip_to` per leader doc — removing the function-call +
    /// within-block linear-scan overhead on the hottest AND case
    /// (rare ∧ common). The general path keeps the per-doc leapfrog,
    /// which amortizes well with the block-max pruning a scoring sink
    /// drives.
    fn and_flat_merge<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        if cursors.len() == 2 {
            self.and_flat_merge_2term(cursors, dl_norm_k1, sink);
        } else {
            self.and_flat_merge_general(cursors, dl_norm_k1, sink);
        }
    }

    /// General `n >= 3`-term AND path. Same shape as the 2-term path:
    /// block-max pruning at the top, then a flat-merge over the
    /// leader's decoded `block_doc_ids` against each non-leader's
    /// decoded `block_doc_ids`. For each leader doc, every non-leader's
    /// `pos` is advanced with a tight `pos += 1` scan instead of
    /// `skip_to` — no function-call or within-block linear-scan
    /// overhead per leader doc, just integer comparisons over the
    /// already-decoded buffers. When any cursor exhausts its block,
    /// the outer loop crosses blocks via `next()` and re-aligns.
    fn and_flat_merge_general<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        'outer: loop {
            if cursors[0].is_exhausted() {
                break;
            }

            // Block-max-AND pruning (scoring sinks only; the unranked
            // sink's `bar()` is NEG_INFINITY, so this whole block is
            // skipped). The bar is the kth-best once the heap fills, or
            // the caller's seeded floor before that — whichever is
            // higher. If the leader's current block can't possibly
            // produce a bar-beating score, skip the whole block — the
            // safest UB sums leader's block_max with each other cursor's
            // max block_max across all blocks that overlap the leader's
            // block doc-id range.
            let bar = sink.bar();
            if bar > f32::NEG_INFINITY {
                let range_start = cursors[0].current_doc_id();
                let range_end = cursors[0].current_block_last_doc_id();
                let leader_block_max = cursors[0].current_block_max_bm25();
                let mut other_ub = 0.0_f32;
                for c in cursors[1..].iter_mut() {
                    other_ub += c.block_max_in_range(range_start, range_end);
                }
                if leader_block_max + other_ub <= bar {
                    cursors[0].skip_to(range_end.saturating_add(1));
                    continue;
                }
            }

            // Align every non-leader cursor to >= leader's current doc.
            // Largest landing-doc becomes the new alignment target if
            // any cursor jumped past leader. If any cursor crossed
            // leader's current block, restart the outer loop so pruning
            // re-fires on leader's new block; otherwise the flat-merge
            // proceeds in the current decoded blocks.
            let leader_doc = cursors[0].current_doc_id();
            let leader_block_end = cursors[0].current_block_last_doc_id();
            let mut max_other = leader_doc;
            let mut crossed_block = false;
            for c in cursors[1..].iter_mut() {
                c.skip_to(leader_doc);
                if c.is_exhausted() {
                    break 'outer;
                }
                let here = c.current_doc_id();
                if here > leader_block_end {
                    crossed_block = true;
                }
                if here > max_other {
                    max_other = here;
                }
            }
            if max_other > leader_doc {
                cursors[0].skip_to(max_other);
                if cursors[0].is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat-merge across decoded blocks. Split leader off so
            // both leader and others borrow mutably without overlap;
            // the inner loop reads each cursor's `block_doc_ids` and
            // updates its `pos` directly.
            let (leader_slice, others) = cursors.split_at_mut(1);
            let c0 = &mut leader_slice[0];
            let lb_n = c0.block_n;
            let mut i = c0.pos;
            while i < lb_n {
                let a = c0.block_doc_ids[i];

                // For each non-leader, walk its `pos` forward through
                // the decoded block until block_doc_ids[pos] >= a (or
                // the block exhausts). If any block exhausts, break
                // out to the outer loop's block-crossing step. If any
                // cursor lands above `a`, the leader doc isn't in the
                // intersection — advance leader only.
                let mut block_exhausted = false;
                let mut all_match = true;
                for o in others.iter_mut() {
                    while o.pos < o.block_n && o.block_doc_ids[o.pos] < a {
                        o.pos += 1;
                    }
                    if o.pos >= o.block_n {
                        block_exhausted = true;
                        break;
                    }
                    if o.block_doc_ids[o.pos] != a {
                        all_match = false;
                        break;
                    }
                }
                if block_exhausted {
                    break;
                }
                if all_match {
                    let score = if sink.needs_score() {
                        let norm = dl_norm_k1.get(a);
                        let mut score =
                            bm25::score_with_dl_norm_k1(c0.idf_x_k1p1, c0.block_tfs[i], norm);
                        for o in others.iter() {
                            score +=
                                bm25::score_with_dl_norm_k1(o.idf_x_k1p1, o.block_tfs[o.pos], norm);
                        }
                        score
                    } else {
                        0.0
                    };
                    sink.emit(a, score);
                    i += 1;
                    for o in others.iter_mut() {
                        o.pos += 1;
                    }
                } else {
                    i += 1;
                }
            }
            c0.pos = i;

            // Cross blocks for whichever cursors exhausted. The outer
            // loop's alignment step re-pulls everyone to the new leader
            // doc on the next iteration.
            if c0.pos >= c0.block_n {
                c0.next();
            }
            for o in others.iter_mut() {
                if o.pos >= o.block_n {
                    o.next();
                }
            }
        }
    }

    /// 2-term specialization. While both cursors share a doc-id region
    /// covered by their respective decoded blocks, do a flat
    /// sorted-merge over the two `block_doc_ids` arrays: no `skip_to`
    /// function calls per leader doc, no per-doc within-block linear
    /// scan — just two index pointers walking forward. When either
    /// block exhausts, the cursor crosses to its next block (decoding
    /// on demand) and the merge resumes.
    fn and_flat_merge_2term<S: AndSink>(
        &self,
        cursors: &mut [TermCursor],
        dl_norm_k1: &NormTable,
        sink: &mut S,
    ) {
        debug_assert_eq!(cursors.len(), 2);
        // Split into two simultaneous mutable refs so the inner loop
        // can read both cursors' decoded buffers and update both
        // positions without borrow-checker contortions.
        let (left, right) = cursors.split_at_mut(1);
        let c0 = &mut left[0];
        let c1 = &mut right[0];

        'outer: loop {
            if c0.is_exhausted() || c1.is_exhausted() {
                break;
            }

            // Block-max-AND pruning at the leader's current block
            // (scoring sinks only; the unranked sink's `bar()` is
            // NEG_INFINITY, so this is skipped). The bar is the kth-best
            // once the heap fills, or the caller's seeded floor before
            // that — whichever is higher.
            let bar = sink.bar();
            if bar > f32::NEG_INFINITY {
                let range_start = c0.current_doc_id();
                let range_end = c0.current_block_last_doc_id();
                let ub =
                    c0.current_block_max_bm25() + c1.block_max_in_range(range_start, range_end);
                if ub <= bar {
                    c0.skip_to(range_end.saturating_add(1));
                    continue;
                }
            }

            // Align c1 with c0 at the current leader doc. After this
            // call both cursors are positioned on doc_ids >= leader.
            // If c1 jumped past the leader's current block we'll bump
            // the leader via the outer loop's next iteration.
            c1.skip_to(c0.current_doc_id());
            if c1.is_exhausted() {
                break 'outer;
            }
            // If c1 sits above c0's pos, pull c0 forward to align.
            // When that pull crosses c0's current block, restart the
            // outer loop so pruning re-fires on c0's new block;
            // otherwise fall through and let the flat-merge handle
            // the within-block divergence inline.
            if c1.current_doc_id() > c0.current_doc_id() {
                let crossed_block = c1.current_doc_id() > c0.current_block_last_doc_id();
                c0.skip_to(c1.current_doc_id());
                if c0.is_exhausted() {
                    break 'outer;
                }
                if crossed_block {
                    continue;
                }
            }

            // Flat sorted-merge within the overlap of the two decoded
            // blocks. Pre-load all locals; the borrow checker is
            // satisfied because c0/c1 are independently mutable refs.
            let lb_n = c0.block_n;
            let rb_n = c1.block_n;
            let mut i = c0.pos;
            let mut j = c1.pos;
            let c0_idf = c0.idf_x_k1p1;
            let c1_idf = c1.idf_x_k1p1;
            while i < lb_n && j < rb_n {
                let a = c0.block_doc_ids[i];
                let b = c1.block_doc_ids[j];
                if a < b {
                    i += 1;
                } else if a > b {
                    j += 1;
                } else {
                    let score = if sink.needs_score() {
                        let norm = dl_norm_k1.get(a);
                        bm25::score_with_dl_norm_k1(c0_idf, c0.block_tfs[i], norm)
                            + bm25::score_with_dl_norm_k1(c1_idf, c1.block_tfs[j], norm)
                    } else {
                        0.0
                    };
                    sink.emit(a, score);
                    i += 1;
                    j += 1;
                }
            }
            c0.pos = i;
            c1.pos = j;

            // Whichever cursor exhausted its block crosses to its next
            // block; the other holds. The outer loop re-checks
            // is_exhausted and re-aligns on the next iteration.
            if i >= lb_n {
                c0.next();
            }
            if j >= rb_n {
                c1.next();
            }
        }
    }

    /// MaxScore+BMM constrained to the doc_id half-open range
    /// `[doc_id_start, doc_id_end)`. Used by the supertable layer's
    /// intra-superfile parallel fan-out: when the reader pool has more
    /// threads than superfiles, each superfile is split into N sub-ranges
    /// and the per-sub-range searches run in parallel, each producing
    /// its own top-K heap that the caller merges.
    ///
    /// Setting `doc_id_start == 0` and `doc_id_end == u32::MAX`
    /// reproduces the un-ranged BMM walk byte-for-byte (the seek is
    /// a no-op and the upper-bound check trivially never fires).
    ///
    /// **Pruning trade**: each sub-range maintains an independent
    /// top-K heap + BMM threshold. The threshold tightens slower than
    /// in the un-ranged walk because each sub-range sees only `1/N`
    /// of the docs, so the per-sub-range BMW block-skip fires less
    /// aggressively. Net wall-time win comes from spreading the
    /// scoring work across more cores; the per-sub-range work loss
    /// from looser pruning is bounded by the bookkeeping path (and
    /// in practice ~10–20% of single-thread serial), well below the
    /// 2× cores-doubled headroom.
    fn run_max_score_bmm_range(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        doc_id_start: u32,
        doc_id_end: u32,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        // Sub-range seek: jump every cursor past any doc_id below
        // the lower bound. Cursors already past the bound stay where
        // they are; cursors whose entire posting list sits below the
        // bound become exhausted. The skip_to walks the skip-table
        // (cross-block) when needed, so we don't decode blocks we'll
        // never score.
        if doc_id_start > 0 {
            for cursor in &mut cursors {
                cursor.skip_to(doc_id_start);
            }
        }

        // Sort descending by term-max UB. Stability isn't required —
        // ties (equal `term_max_bm25` across terms) are rare and the
        // tie-break is arbitrary as long as the prefix-sum invariant
        // holds.
        cursors.sort_unstable_by(|a, b| {
            b.term_max_bm25
                .partial_cmp(&a.term_max_bm25)
                .unwrap_or(Ordering::Equal)
        });

        // Suffix sums of term_max_bm25. partial_max[0] = total UB,
        // partial_max[n] = 0. Monotonically decreasing.
        let n = cursors.len();
        let mut partial_max = vec![0.0_f32; n + 1];
        for i in (0..n).rev() {
            partial_max[i] = partial_max[i + 1] + cursors[i].term_max_bm25;
        }

        // Sized by this slice's own window: only docs inside it can be
        // ranked here, and a sliced fan-out would otherwise preallocate
        // one whole-superfile heap per slice.
        let initial_cap =
            top_k_initial_capacity(k, u64::from(self.n_docs), Some((doc_id_start, doc_id_end)));
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        // Seed the pruning threshold with the caller's floor: docs
        // strictly below it can never matter, so the MaxScore
        // machinery (essential boundary, block skips, heap admission)
        // starts from the floor instead of from zero. BM25 scores are
        // positive, so an unfloored run keeps the original 0.0 seed.
        let mut threshold: f32 = floor_eff.max(0.0);

        let recompute_f = |partial_max: &[f32], threshold: f32| -> usize {
            // Essential boundary: smallest f such that
            // partial_max[f] ≤ threshold. Linear scan from the front —
            // for typical N ≤ 8 query terms this is cheaper than a
            // binary search's branch-and-bound overhead.
            let mut f = 0;
            while f < partial_max.len() - 1 && partial_max[f] > threshold {
                f += 1;
            }
            f
        };
        // With a zero threshold only partial_max[n]=0 satisfies, so
        // f=n (all terms essential); a seeded floor can already shrink
        // the essential set before the first doc is scored.
        let mut f_essential: usize = recompute_f(&partial_max, threshold);

        // Total term-level UB. Used for the block-skip bound on
        // essential cursors below.
        let total_term_ub = partial_max[0];

        loop {
            // **f=1 block-batch fast path.** Once threshold rises
            // enough that only `cursors[0]` (highest term_max) is
            // essential, the candidate set is *exactly* `cursors[0]`'s
            // posting list. We can decode one of its blocks and
            // process every doc in the block inline — no per-doc
            // pivot search, no per-doc cursor sort. The outer loop's
            // overhead amortizes over ~128 docs per block instead of
            // 1 doc per iteration. This is the steady state for
            // dominator queries (wide-UB) and for similar-UB queries
            // after the heap fills with multi-term hits.
            if f_essential == 1 {
                if cursors[0].is_exhausted() || cursors[0].current_doc_id() >= doc_id_end {
                    break;
                }
                // Block-skip: if `block_max + sum_others_term_max`
                // can't beat threshold, skip the block.
                let block_ub = cursors[0].current_block_max_bm25()
                    + (total_term_ub - cursors[0].term_max_bm25);
                if block_ub <= threshold {
                    let end = cursors[0].current_block_last_doc_id();
                    cursors[0].skip_to(end.saturating_add(1));
                    continue;
                }

                let block_end = cursors[0].current_block_last_doc_id();
                let mut f_changed = false;
                // Per-doc UB tightening: bound this doc's max possible
                // score by `essential_score + sum_others_term_max`.
                // If even this can't beat the heap threshold, skip
                // the non-essential lookups + heap update entirely
                // — those are the dominant per-doc cost. Only docs
                // where the essential alone is "in striking distance"
                // pay the full lookup price.
                let others_term_ub = total_term_ub - cursors[0].term_max_bm25;
                while !cursors[0].is_exhausted()
                    && cursors[0].current_doc_id() <= block_end
                    && cursors[0].current_doc_id() < doc_id_end
                {
                    let candidate = cursors[0].current_doc_id();
                    // Drop docs excluded by a negated term (None = keep
                    // all): skip without scoring.
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(candidate)
                    {
                        cursors[0].next();
                        continue;
                    }
                    let norm = dl_norm_k1.get(candidate);
                    let essential_score = bm25::score_with_dl_norm_k1(
                        cursors[0].idf_x_k1p1,
                        cursors[0].current_tf(),
                        norm,
                    );
                    if essential_score + others_term_ub <= threshold {
                        // No combination of non-essential
                        // contributions at `candidate` can push it
                        // above threshold. Skip lookup + heap.
                        cursors[0].next();
                        continue;
                    }
                    // SIMD-pack non-essentials at `candidate`.
                    let mut idfs = [cursors[0].idf_x_k1p1, 0.0, 0.0, 0.0];
                    let mut tfs = [cursors[0].current_tf() as f32, 0.0, 0.0, 0.0];
                    let mut packed = 1;
                    let mut score: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(1) {
                        cursor.skip_to(candidate);
                        if cursor.current_doc_id() == candidate {
                            idfs[packed] = cursor.idf_x_k1p1;
                            tfs[packed] = cursor.current_tf() as f32;
                            packed += 1;
                            if packed == 4 {
                                score += bm25::score_simd_x4(idfs, tfs, norm);
                                idfs = [0.0; 4];
                                tfs = [0.0; 4];
                                packed = 0;
                            }
                        }
                    }
                    if packed > 0 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                    }

                    if heap.len() < k {
                        heap.push(TopKEntry(score, candidate));
                        if heap.len() == k {
                            // max(): a seeded floor must never be
                            // lowered by a weaker local kth-best.
                            threshold = heap.peek().expect("non-empty").0.max(threshold);
                            let new_f = recompute_f(&partial_max, threshold);
                            if new_f != f_essential {
                                f_essential = new_f;
                                f_changed = true;
                            }
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, candidate));
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                        let new_f = recompute_f(&partial_max, threshold);
                        if new_f != f_essential {
                            f_essential = new_f;
                            f_changed = true;
                        }
                    }

                    cursors[0].next();

                    if f_changed {
                        break;
                    }
                }
                continue;
            }

            // Pick the next candidate doc: smallest current_doc_id
            // among essential cursors. (Non-essential cursors only
            // get probed via skip_to once we have a candidate.)
            // Specialized for f=2 (the most common steady state for
            // similar-UB queries) to avoid the iter loop overhead.
            let (candidate, leftmost_essential) = if f_essential == 2 {
                let d0 = cursors[0].current_doc_id();
                let d1 = cursors[1].current_doc_id();
                if d0 == u32::MAX && d1 == u32::MAX {
                    break;
                }
                if d0 <= d1 { (d0, 0) } else { (d1, 1) }
            } else {
                let mut candidate = u32::MAX;
                let mut leftmost_essential: usize = 0;
                for (i, cursor) in cursors.iter().take(f_essential).enumerate() {
                    let d = cursor.current_doc_id();
                    if d < candidate {
                        candidate = d;
                        leftmost_essential = i;
                    }
                }
                if candidate == u32::MAX {
                    break;
                }
                (candidate, leftmost_essential)
            };
            // Sub-range upper bound: every subsequent candidate is
            // monotonically increasing, so once we cross the bound
            // there's no work left for this sub-range.
            if candidate >= doc_id_end {
                break;
            }

            // **BMW-style block-skip on the leftmost essential.** Bound
            // the score of any doc in `leftmost_essential`'s current
            // block by `current_block_max + sum_of_other_term_UBs`. If
            // that bound can't beat the threshold, no doc in this
            // block can make top-k — skip the cursor past its current
            // block. This is what makes BMM competitive with WAND+BMW
            // on dominant-term queries; without it MaxScore scans
            // every doc in the dominant term's posting list.
            let leftmost_term_ub = cursors[leftmost_essential].term_max_bm25;
            let leftmost_block_ub = cursors[leftmost_essential].current_block_max_bm25();
            // others_ub = sum of OTHER cursors' term UBs (essential + non-essential).
            // We use term-level UBs for the others as a conservative bound; using
            // their per-block UBs would tighten further but require keeping them
            // synced with the candidate, which we only do lazily in the
            // non-essential probe below.
            let others_ub = total_term_ub - leftmost_term_ub;
            if leftmost_block_ub + others_ub <= threshold {
                let last_in_block = cursors[leftmost_essential].current_block_last_doc_id();
                cursors[leftmost_essential].skip_to(last_in_block.saturating_add(1));
                continue;
            }

            // Drop docs excluded by a negated term before scoring —
            // the non-essential probes below are the dominant per-doc
            // cost and an excluded doc can never enter the heap. The
            // essential-cursor advance after this block still runs, so
            // the walk progresses.
            let admitted = match filter.as_deref_mut() {
                Some(f) => f.admits(candidate),
                None => true,
            };
            if admitted {
                // Score essential contributions at the candidate doc.
                // SIMD-pack up to 4 cursors per scoring call. (Essential
                // scoring has no early-bail; non-essential scoring below
                // does, so it stays scalar to keep `score` always
                // up-to-date for the bail check.)
                let norm = dl_norm_k1.get(candidate);
                let mut score: f32 = 0.0;
                let mut idfs = [0.0_f32; 4];
                let mut tfs = [0.0_f32; 4];
                let mut packed = 0;
                for cursor in cursors.iter().take(f_essential) {
                    if cursor.current_doc_id() == candidate {
                        idfs[packed] = cursor.idf_x_k1p1;
                        tfs[packed] = cursor.current_tf() as f32;
                        packed += 1;
                        if packed == 4 {
                            score += bm25::score_simd_x4(idfs, tfs, norm);
                            idfs = [0.0; 4];
                            tfs = [0.0; 4];
                            packed = 0;
                        }
                    }
                }
                if packed > 0 {
                    score += bm25::score_simd_x4(idfs, tfs, norm);
                }

                // Per-doc UB tightening: bound the doc's max possible
                // score by `essential_score + sum_non_essentials_term_max`.
                // If even this can't beat threshold, skip the
                // non-essential probe + heap update entirely. This is
                // looser than the per-non-essential block_ub bound below
                // but spares the `skip_to` cursor advances themselves —
                // those are the dominant per-doc cost.
                let non_essentials_term_ub = partial_max[f_essential];
                if score + non_essentials_term_ub > threshold {
                    // Tighter pre-bail using non-essential block_max
                    // (which is tighter than term_max). Use shallow
                    // advance — moves the lightweight inspect-block
                    // pointer to candidate's block without decoding,
                    // amortized O(1). If even this tighter UB can't beat
                    // threshold, skip the deep skip_to pass entirely.
                    let mut remaining_block_ub: f32 = 0.0;
                    for cursor in cursors.iter_mut().skip(f_essential) {
                        cursor.shallow_advance_block_to(candidate);
                        remaining_block_ub += cursor.inspect_block_max_bm25();
                    }

                    if score + remaining_block_ub > threshold {
                        for cursor in cursors.iter_mut().skip(f_essential) {
                            let block_ub = cursor.inspect_block_max_bm25();
                            if score + remaining_block_ub <= threshold {
                                break;
                            }
                            cursor.skip_to(candidate);
                            if cursor.current_doc_id() == candidate {
                                score += bm25::score_with_dl_norm_k1(
                                    cursor.idf_x_k1p1,
                                    cursor.current_tf(),
                                    norm,
                                );
                            }
                            remaining_block_ub -= block_ub;
                        }
                    }
                }
                // (If essential score + remaining_block_ub already ≤ threshold,
                // we don't bother scoring non-essentials — the doc can't beat
                // the kth-best.)

                // Update heap. `threshold` is kept in sync with
                // heap.peek().0 every time we mutate the heap, so we can
                // gate the replace-or-skip decision against the local
                // f32 instead of paying for a heap.peek() per iter.
                // (max(): a seeded floor must never be lowered by a
                // weaker local kth-best.)
                if heap.len() < k {
                    heap.push(TopKEntry(score, candidate));
                    if heap.len() == k {
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                        f_essential = recompute_f(&partial_max, threshold);
                    }
                } else if score > threshold {
                    heap.pop();
                    heap.push(TopKEntry(score, candidate));
                    threshold = heap.peek().expect("non-empty").0.max(threshold);
                    f_essential = recompute_f(&partial_max, threshold);
                }
            }

            // Advance every essential cursor that was at the candidate
            // doc. (Non-essential cursors stay where skip_to landed
            // them; the next iteration's skip_to will move them as
            // needed for the next candidate.)
            for cursor in cursors.iter_mut().take(f_essential) {
                if cursor.current_doc_id() == candidate {
                    cursor.next();
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Windowed union scorer for multi-term OR — the fast path for
    /// uniform-upper-bound / common-term ORs, where MaxScore can't prune
    /// and degrades to scoring the whole union with per-doc f-way merge
    /// overhead.
    ///
    /// Walks the doc-id space one `OR_WINDOW`-doc window at a time. Within
    /// a window each cursor streams its postings **sequentially**,
    /// accumulating its BM25 contribution into `scores[doc - base]` and
    /// marking a presence bit — no per-doc min-scan across cursors, no
    /// heap touch during accumulation. The window is then drained in
    /// ascending doc order (bit-trick over the presence bitset) and each
    /// distinct matching doc is offered to the top-k heap once. Empty
    /// windows are skipped (the base jumps to the next live doc), so a
    /// sparse union costs only its non-empty windows.
    ///
    /// **Exact top-k:** same result set/order as [`Self::run_max_score_bmm`]
    /// — same heap-admission rule (`score > threshold`, floor-seeded), same
    /// `(score desc, doc asc)` tie-break, docs offered in ascending order.
    /// The one nuance is summation *order*: contributions are summed
    /// term-major here vs. per-doc-major in MaxScore, and f32 add is
    /// non-associative, so a score can differ by ≤1 ULP. Validated against
    /// the brute-force BM25 oracle; if a boundary tie ever flips, the
    /// accumulator would move to f64.
    ///
    /// Negation: the [`ExcludeFilter`] is applied at **drain** (globally
    /// ascending → satisfies its monotonic-feed contract), never during the
    /// term-major accumulation.
    fn run_windowed_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
        mut filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
        doc_id_start: u32,
        doc_id_end: u32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // A top-0 request admits nothing. Guard here too (callers already
        // short-circuit) so the heap-admission `else if` below can never
        // run against an empty heap.
        if k == 0 {
            return Ok(Vec::new());
        }
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        if doc_id_start > 0 {
            for c in &mut cursors {
                c.skip_to(doc_id_start);
            }
        }

        // This scan's own window, as in the MaxScore path. Un-ranged
        // callers pass `[0, u32::MAX)`, which the `n_docs` cap collapses
        // back to a whole-superfile scope.
        let initial_cap =
            top_k_initial_capacity(k, u64::from(self.n_docs), Some((doc_id_start, doc_id_end)));
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        // Floor-seeded threshold, identical to the MaxScore path.
        let mut threshold: f32 = floor_eff.max(0.0);

        // Per-window state, allocated once and reused across windows.
        // Cleared lazily during the drain (only touched slots), so reset
        // cost is proportional to matches, not to OR_WINDOW.
        let mut scores = vec![0.0f32; OR_WINDOW as usize];
        let mut present = [0u64; OR_WINDOW_WORDS];

        loop {
            // Next non-empty window: smallest current doc among live
            // cursors, aligned down to a window boundary. O(f) per window
            // (not per doc) — this replaces MaxScore's per-doc min-scan.
            let mut min_doc = u32::MAX;
            for c in &cursors {
                if !c.is_exhausted() {
                    min_doc = min_doc.min(c.current_doc_id());
                }
            }
            if min_doc == u32::MAX || min_doc >= doc_id_end {
                break;
            }
            let base = min_doc & !(OR_WINDOW - 1);
            // saturating: a doc id within OR_WINDOW of u32::MAX would
            // overflow `base + OR_WINDOW` (panic in debug; wrap in release,
            // which makes window_end < base → the accumulate loop stalls and
            // the outer loop spins). Saturate, then clamp to doc_id_end.
            let window_end = base.saturating_add(OR_WINDOW).min(doc_id_end);

            // Accumulate each cursor's contributions in [base, window_end).
            // Sequential walk per cursor; `d - base` is in range because
            // every live cursor sits at `>= min_doc >= base`.
            for c in &mut cursors {
                while !c.is_exhausted() {
                    let d = c.current_doc_id();
                    if d >= window_end {
                        break;
                    }
                    let pos = c.pos;
                    if pos + bm25::SCORE_SIMD_LANES <= c.block_n {
                        let doc_ids = [
                            c.block_doc_ids[pos],
                            c.block_doc_ids[pos + 1],
                            c.block_doc_ids[pos + 2],
                            c.block_doc_ids[pos + 3],
                        ];
                        if doc_ids[bm25::SCORE_SIMD_LANES - 1] < window_end {
                            let contributions = bm25::score_one_term_x4(
                                c.idf_x_k1p1,
                                [
                                    c.block_tfs[pos],
                                    c.block_tfs[pos + 1],
                                    c.block_tfs[pos + 2],
                                    c.block_tfs[pos + 3],
                                ],
                                [
                                    dl_norm_k1.get(doc_ids[0]),
                                    dl_norm_k1.get(doc_ids[1]),
                                    dl_norm_k1.get(doc_ids[2]),
                                    dl_norm_k1.get(doc_ids[3]),
                                ],
                            );
                            for lane in 0..bm25::SCORE_SIMD_LANES {
                                let local = (doc_ids[lane] - base) as usize;
                                scores[local] += contributions[lane];
                                present[local >> 6] |= 1u64 << (local & 63);
                            }
                            c.advance_by(bm25::SCORE_SIMD_LANES);
                            continue;
                        }
                    }
                    let local = (d - base) as usize;
                    scores[local] += bm25::score_with_dl_norm_k1(
                        c.idf_x_k1p1,
                        c.current_tf(),
                        dl_norm_k1.get(d),
                    );
                    present[local >> 6] |= 1u64 << (local & 63);
                    c.next();
                }
            }

            // Drain ascending; clear touched slots for reuse; apply
            // negation; offer to the heap.
            for (word_idx, word) in present.iter_mut().enumerate() {
                let mut bits = *word;
                *word = 0;
                while bits != 0 {
                    let b = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let local = (word_idx << 6) | b;
                    let score = scores[local];
                    scores[local] = 0.0;
                    let doc = base + local as u32;
                    if let Some(f) = filter.as_deref_mut()
                        && !f.admits(doc)
                    {
                        continue;
                    }
                    if heap.len() < k {
                        heap.push(TopKEntry(score, doc));
                        if heap.len() == k {
                            threshold = heap.peek().expect("non-empty").0.max(threshold);
                        }
                    } else if score > threshold {
                        heap.pop();
                        heap.push(TopKEntry(score, doc));
                        threshold = heap.peek().expect("non-empty").0.max(threshold);
                    }
                }
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Exhaustive union walk for multi-term OR. No threshold-driven
    /// block skipping — every doc in the union of the cursor postings
    /// is scored and offered to the top-K heap.
    ///
    /// **Not on the production path.** `dispatch_or_algo` routes
    /// to MaxScore+BMM or the windowed union; this function is reachable
    /// only via `search_with_algo_for_bench(OrAlgo::Exhaustive)`. It exists
    /// because the supertable bench surfaced one specific shape where
    /// it narrowly wins, and we want the option available for future
    /// re-routing work without re-implementing it.
    ///
    /// **When this can beat BMM (measured at 10M × 8 superfiles)**:
    /// - **Prefix expansions over very-rare terms, in parallel mode.**
    ///   E.g., `term0009*` expanding to 10 terms at Zipfian rank
    ///   90–99 (df ≈ 0.1% each). On the supertable parallel bench,
    ///   exhaustive ran at 40.2 ms vs BMM's 54.0 ms — a 26% win. The
    ///   per-superfile work is tiny (∼12 K matching docs across 10
    ///   short cursors) so BMM's per-block bookkeeping
    ///   (`f_essential` recomputation, `shallow_advance_block_to`,
    ///   `inspect_block_max_bm25`) dominates over actual scoring
    ///   work.
    ///
    /// **When BMM is strictly better — measured regressions if we
    /// route to exhaustive**:
    /// - **Mid-rank uniform-UB queries.** Five terms at rank 50–54
    ///   (df ≈ 0.4% each): exhaustive serial 174 ms vs BMM 99 ms —
    ///   a **76% regression**. Three terms at rank 50–52: exhaustive
    ///   serial 93 ms vs BMM 61 ms — a **52% regression**. Enough
    ///   matching docs exist that BMM's skip-pruning actually fires
    ///   and amortizes its bookkeeping.
    /// - **Any dominant-term query.** BMM's `f_essential == 1` fast
    ///   path collapses to a block-batch loop on the dominant
    ///   cursor's postings — about as tight as exhaustive could be,
    ///   and with skip on top.
    /// - **Single-term queries.** Don't go through OR dispatch
    ///   anyway; `search_single_term_bmw` handles them.
    ///
    /// **Routing heuristic if revisited**: the obvious-looking
    /// `max(term_max_bm25) / sum(term_max_bm25) < 1.5/n_cursors`
    /// (uniform UB) **over-routes** because it admits mid-rank
    /// queries where BMM wins. A better rule would gate on
    /// *absolute* low total df **and** uniform UB — e.g.,
    /// `σdf < n_docs / 100 AND max_ub/sum_ub < 1.5/n_cursors`.
    /// Empirically that admits the prefix-of-rare-terms shape and
    /// excludes the mid-rank multi-term shapes. Not yet wired up:
    /// the single-query parallel win (26% on prefix) hasn't
    /// justified the routing-heuristic maintenance cost yet.
    ///
    /// Algorithm: classic k-way merge over `TermCursor`s. Each
    /// iteration finds the smallest current `doc_id` among live
    /// cursors, sums BM25 contributions from all cursors at that
    /// doc, advances those cursors, pushes into the top-K min-heap.
    ///
    /// Result invariants match [`Self::run_max_score_bmm`]: top-k by
    /// descending BM25 score, ties broken by ascending doc_id.
    fn run_exhaustive_union(
        &self,
        column_id: u32,
        mut cursors: Vec<TermCursor>,
        k: usize,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let col_meta = &self.columns[column_id as usize];
        let dl_norm_k1 = &col_meta.dl_norm_k1;

        let initial_cap = top_k_initial_capacity(k, u64::from(self.n_docs), None);
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(initial_cap);
        let mut threshold: f32 = 0.0;

        loop {
            // Find smallest current doc_id across all live cursors —
            // the next candidate to score. Exhausted cursors report
            // `u32::MAX`, which can't be smaller than any live cursor's
            // doc_id, so this terminates naturally when every cursor
            // has been drained.
            let mut candidate = u32::MAX;
            for cursor in &cursors {
                let d = cursor.current_doc_id();
                if d < candidate {
                    candidate = d;
                }
            }
            if candidate == u32::MAX {
                break;
            }

            // Score: sum BM25 from every cursor positioned at the
            // candidate doc. Pack up to 4 cursors per SIMD scoring
            // call, matching the BMM essential-scoring shape.
            let norm = dl_norm_k1.get(candidate);
            let mut score: f32 = 0.0;
            let mut idfs = [0.0_f32; 4];
            let mut tfs = [0.0_f32; 4];
            let mut packed = 0;
            for cursor in cursors.iter_mut() {
                if cursor.current_doc_id() == candidate {
                    idfs[packed] = cursor.idf_x_k1p1;
                    tfs[packed] = cursor.current_tf() as f32;
                    packed += 1;
                    if packed == 4 {
                        score += bm25::score_simd_x4(idfs, tfs, norm);
                        idfs = [0.0; 4];
                        tfs = [0.0; 4];
                        packed = 0;
                    }
                    cursor.next();
                }
            }
            if packed > 0 {
                score += bm25::score_simd_x4(idfs, tfs, norm);
            }

            // Top-K update. `threshold` mirrors `heap.peek().0` so
            // the replace-or-skip branch doesn't re-peek per iter.
            if heap.len() < k {
                heap.push(TopKEntry(score, candidate));
                if heap.len() == k {
                    threshold = heap.peek().expect("non-empty").0;
                }
            } else if score > threshold {
                heap.pop();
                heap.push(TopKEntry(score, candidate));
                threshold = heap.peek().expect("non-empty").0;
            }
        }

        Ok(drain_top_k_desc(heap))
    }

    /// Multi-term OR dispatch. Routes everything to MaxScore+BMM.
    ///
    /// **Routing decision (1M docs — head-to-head WAND+BMW vs MaxScore+BMM):**
    ///
    /// | Query shape                                 | WAND+BMW | MaxScore+BMM |
    /// |---|---|---|
    /// | two-term wide (rank 1 + 50)                 | 1.25 ms  | **0.28 ms**  |
    /// | three-term wide (rank 1 + 50 + 100)         | 17.2 ms  | 18.3 ms      |
    /// | three-term similar UBs (rank 50/51/52)      | 28.3 ms  | **24.7 ms**  |
    /// | five-term similar UBs (rank 50–54)          | 59.1 ms  | **55.1 ms**  |
    ///
    /// BMM wins on most shapes once we have:
    ///   1. A precomputed per-doc length-norm table (no per-call
    ///      `dl/avgdl` work in scoring).
    ///   2. SIMD x4 scoring of all aligned cursors per doc.
    ///   3. A block-batch fast path when only one cursor is essential
    ///      (`f_essential == 1`) — the steady state for wide-UB and
    ///      heap-warmed similar-UB queries.
    ///
    /// **Exhaustive union walk** ([`Self::run_exhaustive_union`]) is
    /// implemented and reachable via `search_with_algo_for_bench`,
    /// but the dispatcher does NOT route to it. Empirically it
    /// regressed mid-rank uniform-UB shapes by 50–80% — see
    /// `run_exhaustive_union`'s doc comment for the cost model and
    /// the one shape (prefix-of-very-rare-terms in parallel mode)
    /// where it narrowly wins. WAND+BMW remains in the codebase
    /// for the same reason — bench-harness comparison only.
    fn dispatch_or_algo(
        &self,
        column_id: u32,
        cursors: Vec<TermCursor>,
        k: usize,
        filter: Option<&mut ExcludeFilter>,
        floor_eff: f32,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        // Route on upper-bound *spread*, not term count: when no single
        // term dominates, MaxScore's essential set never shrinks and it
        // degrades to scoring the whole union with per-doc f-way merge
        // overhead — the windowed union scorer is dramatically faster
        // there. A dominant-term query stays on MaxScore, which prunes
        // hard (its block-skip / f→1 fast path); windowing would lose by
        // scoring every windowed doc.
        // A 2-term OR of one rare + one common term is a worst case for
        // MaxScore: it scores the common term's long posting list end to
        // end. WAND+BMW pivots on the rare (short) term and skips most of
        // the common term's list — a large win. For two comparable-length
        // common terms there is no short anchor to skip on, so WAND's
        // per-iteration cursor re-sort just adds overhead and MaxScore
        // wins; those fall through. Route 2-term ORs to WAND only when
        // (a) one list is much shorter than the other (df ratio,
        // `two_term_has_rare_anchor`); (b) k is small — at large k the
        // top-k threshold is too low for WAND to prune; (c) no negation —
        // `run_wand_bmw` applies no exclude filter; and (d) no
        // cross-segment floor (`floor_eff` unset) — seeding WAND's
        // threshold from a floor mis-prunes, so a live floor stays on
        // MaxScore.
        // The dominance heuristic (`prefer_windowed_union`) assumes a
        // dominant term means MaxScore prunes hard — true only at small
        // `k`. At large `k` relative to the rarer terms' combined df the
        // top-k threshold collapses to the common term's upper bound,
        // MaxScore can skip nothing, and it degrades to a *scalar* full
        // scan. `or_topk_pruning_ineffective` catches exactly that case
        // (including a 2-term rare+common OR too deep for WAND) and
        // routes it to the SIMD windowed scorer, which does the same
        // full scan without the per-doc f-way merge. Small-`k`
        // dominant-term ORs fail this test and stay on MaxScore, where
        // pruning still wins.
        let no_floor = floor_eff == f32::NEG_INFINITY;
        if cursors.len() == 2
            && k <= WAND_BMW_2TERM_MAX_K
            && filter.is_none()
            && no_floor
            && two_term_has_rare_anchor(&cursors)
        {
            self.run_wand_bmw(column_id, cursors, k)
        } else if prefer_windowed_union(&cursors) || or_topk_pruning_ineffective(&cursors, k) {
            self.run_windowed_union(column_id, cursors, k, filter, floor_eff, 0, u32::MAX)
        } else {
            self.run_max_score_bmm(column_id, cursors, k, filter, floor_eff)
        }
    }

    /// Bench/dev helper: force the multi-term OR path to use a specific
    /// algorithm regardless of the dispatcher's heuristic. Used by the
    /// superfile tier's per-algorithm probes
    /// (`benches/utils/superfile.rs`) to compare WAND+BMW, MaxScore+BMM,
    /// and the windowed union under identical inputs so the dispatch
    /// thresholds are validated against measured numbers every run.
    ///
    /// **Not part of the stable API** — production code should use
    /// `search`, which routes through `dispatch_or_algo`.
    #[doc(hidden)]
    pub async fn search_with_algo_for_bench(
        &self,
        column: &str,
        terms: &[&str],
        k: usize,
        algo: OrAlgo,
    ) -> Result<Vec<(u32, f32)>, FtsError> {
        let column_id = self.resolve_column_id(column)?;
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let cursors = self.build_term_cursors(column_id, terms, None).await?;
        if cursors.is_empty() {
            return Ok(Vec::new());
        }
        // Bench-only selector; never carries negation or a floor.
        match algo {
            OrAlgo::Bmm => self.run_max_score_bmm(column_id, cursors, k, None, f32::NEG_INFINITY),
            OrAlgo::WandBmw => self.run_wand_bmw(column_id, cursors, k),
            OrAlgo::Exhaustive => self.run_exhaustive_union(column_id, cursors, k),
            OrAlgo::Windowed => {
                self.run_windowed_union(column_id, cursors, k, None, f32::NEG_INFINITY, 0, u32::MAX)
            }
        }
    }
}

/// One query's built OR cursors for one superfile: the postings fetch
/// and skip-table parse done once, cheaply cloneable per doc-id
/// sub-range. Produced by [`FtsReader::build_or_cursor_set`], consumed
/// by [`FtsReader::search_or_range_prebuilt`].
pub(crate) struct OrCursorSet {
    column_id: u32,
    cursors: Vec<TermCursor>,
}

impl OrCursorSet {
    /// Number of expanded terms this set was built from — used to gate
    /// ranged-kernel pool dispatch by scan cost, the same signal the
    /// plain multi-should path gates on.
    pub(crate) fn len(&self) -> usize {
        self.cursors.len()
    }

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

/// Top-k min-heap entry `(score, doc_id)`, shared by every search
/// path (single-term BMW, WAND+BMW, MaxScore+BMM, exhaustive union,
/// AND intersection, and the `search_multi` combiner).
///
/// Ordering is **reversed** on purpose: smaller score is "greater",
/// so `BinaryHeap::peek()` returns the smallest-score entry. Once the
/// heap holds k entries, `peek()` is the current kth-best score — the
/// bar a new doc must beat (also the BMW/BMM pruning threshold).
/// Tie-break: larger doc_id is "greater", so on equal scores the
/// smaller doc_id survives in the heap.
#[derive(Debug, Copy, Clone)]
struct TopKEntry(f32, u32);
impl PartialEq for TopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl Eq for TopKEntry {}
impl PartialOrd for TopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Score is inverted (lower score = greater) so the max-heap's
        // peek is the worst kept entry; the doc-id leg must NOT be
        // inverted, so that among score-tied entries the LARGER doc id
        // is greater — i.e. peek — and is the one evicted when a
        // better doc arrives, keeping the smaller id.
        other
            .0
            .partial_cmp(&self.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.1.cmp(&other.1))
    }
}

/// Drain a top-k min-heap into the public result order: descending
/// score, ascending doc_id on ties.
///
/// pdqsort: entries are unique by `(score, doc_id)` — every search
/// path offers each doc_id to its heap at most once — so an unstable
/// sort has no observable reorderings.
fn drain_top_k_desc(heap: BinaryHeap<TopKEntry>) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = heap.into_iter().map(|TopKEntry(s, d)| (d, s)).collect();
    out.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    out
}

/// One member term of a [`PhraseCursor`]: its posting cursor, its
/// fetched position runs, and a lazily-built per-block cache of each
/// pair's run offset.
struct PhraseMember {
    cursor: TermCursor,
    /// The term's complete position runs (empty for an inline df=1
    /// member, whose single position is `inline_position`).
    positions: Bytes,
    /// The term's parsed metadata header, re-parsed from the cursor's
    /// own bytes at member build — the source of the per-block
    /// position-run offsets. `None` for an inline member (no postings
    /// bytes). Kept here, not on [`TermCursor`] or [`BlockMeta`]:
    /// plain term queries never touch positions, and their hot
    /// structures must not grow for the phrase path's benefit.
    term_meta: Option<TermMeta>,
    /// The single position of an inline (df=1, tf=1) member — the
    /// inline FST value's slot carries it instead of a tf. `None` for
    /// PFOR members.
    inline_position: Option<u32>,
    /// The member's bare idf (the cursor stores only `idf × (K1+1)`).
    idf: f32,
    /// Byte offset of each decoded-block pair's run within
    /// `positions`, valid for `run_offsets_block`. Rebuilt on block
    /// crossings by one `skip_run` walk over the block's runs.
    run_offsets: Vec<u32>,
    /// Which block index `run_offsets` covers (`usize::MAX` = none).
    run_offsets_block: usize,
    /// Scratch for the member's decoded positions at the aligned doc.
    pos_scratch: Vec<u32>,
}

/// Sentinel for [`PhraseMember::run_offsets_block`]: no block cached.
const NO_BLOCK_CACHED: usize = usize::MAX;

impl PhraseMember {
    /// The member's positions at its cursor's current doc, decoded
    /// into `pos_scratch`. The cursor must be positioned on a doc
    /// (not exhausted).
    fn decode_current_positions(&mut self) -> Result<(), FtsError> {
        self.pos_scratch.clear();
        if let Some(p) = self.inline_position {
            self.pos_scratch.push(p);
            return Ok(());
        }
        let block = self.cursor.current_block;
        if self.run_offsets_block != block {
            // One forward walk locates every pair's run in this
            // block: start at the block's recorded first-run offset
            // (the skip entry's fourth field), then skip each pair's
            // `tf` varints.
            self.run_offsets.clear();
            let term_meta = self.term_meta.as_ref().expect("PFOR member has term meta");
            let mut at =
                term_meta.positions_block_offset(self.cursor.bytes.as_ref(), block) as usize;
            for i in 0..self.cursor.block_n {
                self.run_offsets.push(at as u32);
                skip_run(&self.positions, &mut at, self.cursor.block_tfs[i]).ok_or_else(|| {
                    FtsError::Read(ReadError::MalformedVersion(
                        "position runs truncated within block".into(),
                    ))
                })?;
            }
            self.run_offsets_block = block;
        }
        let pair = self.cursor.pos;
        let mut at = self.run_offsets[pair] as usize;
        decode_run(
            &self.positions,
            &mut at,
            self.cursor.block_tfs[pair],
            &mut self.pos_scratch,
        )
        .ok_or_else(|| {
            FtsError::Read(ReadError::MalformedVersion(
                "position run truncated or overflowing".into(),
            ))
        })?;
        Ok(())
    }
}

/// Doc-at-a-time cursor over an exact phrase: the members'
/// intersection drives doc alignment, and a doc matches only when the
/// members' positions verify adjacency (member `i` at `p + i` for
/// some anchor `p`). Scores as one BM25 atom with `tf` = the number
/// of verified anchors and `idf` = Σ member idf. Exposes the same
/// notion of term- and block-level upper bounds as [`TermCursor`], so
/// the atom walks can prune with it:
/// `bound = phrase_idf × min_i(member_bound_i / idf_i)` — sound
/// because the phrase tf in any doc is ≤ every member's tf there and
/// the BM25 tf-factor is monotone in tf.
struct PhraseCursor {
    members: Vec<PhraseMember>,
    /// Member indices in ascending posting-list length (rarest first).
    /// The doc-alignment in [`Self::seek_match`] is a set intersection —
    /// order-independent — so it probes members rarest-first: the short
    /// lists drive the candidate doc and the long lists (a common word
    /// like "the") are only skip-confirmed last, once per candidate,
    /// instead of being re-skipped on every advance of a rare member.
    /// Positional verification still runs in query order (`members`
    /// order), which the phrase adjacency check requires.
    align_order: Vec<usize>,
    /// Σ member idf × (K1 + 1) — the phrase's scoring constant.
    idf_x_k1p1: f32,
    /// Phrase-scaled term-level upper bound (see type docs).
    term_max_bm25: f32,
    /// Aligned-and-verified doc, or `u32::MAX` when exhausted.
    current_doc: u32,
    /// Number of verified anchors at `current_doc`.
    current_tf: u32,
    /// Reused across `verify_at_aligned` calls to hold the candidate
    /// phrase-start positions as they are filtered member by member —
    /// avoids a per-doc allocation on the hot verify path.
    verify_scratch: Vec<u32>,
}

impl PhraseCursor {
    /// Build from member cursors (query order), their fetched
    /// position runs, and their positional metadata — `(term_meta,
    /// inline_position)` per member, exactly one of the two present —
    /// then seek to the first matching doc.
    fn new(
        cursors: Vec<TermCursor>,
        positions: Vec<Bytes>,
        positional: Vec<(Option<TermMeta>, Option<u32>)>,
    ) -> Result<Self, FtsError> {
        debug_assert!(cursors.len() >= 2, "single-token phrases degrade to terms");
        debug_assert_eq!(cursors.len(), positions.len());
        debug_assert_eq!(cursors.len(), positional.len());
        let mut idf_sum = 0.0f32;
        let mut min_scaled_bound = f32::INFINITY;
        let members: Vec<PhraseMember> = cursors
            .into_iter()
            .zip(positions)
            .zip(positional)
            .map(|((cursor, positions), (term_meta, inline_position))| {
                let idf = cursor.idf_x_k1p1 / (bm25::K1 + 1.0);
                min_scaled_bound = min_scaled_bound.min(cursor.term_max_bm25 / idf);
                idf_sum += idf;
                PhraseMember {
                    cursor,
                    positions,
                    term_meta,
                    inline_position,
                    idf,
                    run_offsets: Vec::new(),
                    run_offsets_block: NO_BLOCK_CACHED,
                    pos_scratch: Vec::new(),
                }
            })
            .collect();
        // Rarest-first probe order for alignment: fewest posting blocks
        // (shortest list) first. Query order is preserved in `members`.
        let mut align_order: Vec<usize> = (0..members.len()).collect();
        align_order.sort_by_key(|&i| members[i].cursor.block_count());
        let mut cursor = Self {
            idf_x_k1p1: idf_sum * (bm25::K1 + 1.0),
            term_max_bm25: idf_sum * min_scaled_bound,
            members,
            align_order,
            current_doc: 0,
            current_tf: 0,
            verify_scratch: Vec::new(),
        };
        cursor.seek_match(0, f32::NEG_INFINITY, &NormTable::empty())?;
        Ok(cursor)
    }

    #[inline]
    fn is_exhausted(&self) -> bool {
        self.current_doc == u32::MAX
    }

    #[inline]
    fn current_doc_id(&self) -> u32 {
        self.current_doc
    }

    /// Advance to the first verified phrase match at doc ≥ `target`.
    fn skip_to(&mut self, target: u32) -> Result<(), FtsError> {
        if self.is_exhausted() || self.current_doc >= target {
            return Ok(());
        }
        self.seek_match(target, f32::NEG_INFINITY, &NormTable::empty())
    }

    /// [`Self::skip_to`] for ranked walks: additionally skips docs
    /// whose phrase contribution provably can't matter. `bar` is the
    /// most this atom may need to contribute (the walk's pruning bar
    /// minus every other atom's upper bound); a doc whose phrase
    /// score bound falls strictly below it is passed over without any
    /// position work — sound for top-k because the doc's total score
    /// then can't reach the bar, but NOT for match/count walks, which
    /// must keep using [`Self::skip_to`].
    fn skip_to_pruned(
        &mut self,
        target: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        if self.is_exhausted() || self.current_doc >= target {
            return Ok(());
        }
        self.seek_match(target, bar, dl_norm_k1)
    }

    /// Leapfrog the members to their next common doc ≥ `from`, verify
    /// adjacency there, and repeat until a match or exhaustion. When
    /// `bar` is finite, aligned docs are pre-screened without touching
    /// positions: the phrase tf can't exceed any member's tf, so the
    /// BM25 score at the members' minimum tf bounds the phrase's
    /// contribution, and a doc strictly below `bar` is skipped before
    /// the run decode. (`<`, not `<=`: a doc exactly at the bar can
    /// still displace the incumbent kth-best on the ascending-doc-id
    /// tie-break, so it must be verified.)
    fn seek_match(
        &mut self,
        mut from: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        'docs: loop {
            // Align every member to the same doc ≥ `from`, probing
            // rarest-first so a common member is skip-confirmed last
            // rather than re-skipped on every rare-member advance.
            let mut aligned = from;
            let mut oi = 0usize;
            while oi < self.align_order.len() {
                let mi = self.align_order[oi];
                let c = &mut self.members[mi].cursor;
                c.skip_to(aligned);
                if c.is_exhausted() {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return Ok(());
                }
                let here = c.current_doc_id();
                if here > aligned {
                    // Restart alignment at the higher doc.
                    aligned = here;
                    oi = 0;
                    continue;
                }
                oi += 1;
            }

            if bar > f32::NEG_INFINITY {
                let min_tf = self
                    .members
                    .iter()
                    .map(|m| m.cursor.current_tf())
                    .min()
                    .expect("members >= 2");
                let ub =
                    bm25::score_with_dl_norm_k1(self.idf_x_k1p1, min_tf, dl_norm_k1.get(aligned));
                if ub < bar {
                    from = match aligned.checked_add(1) {
                        Some(next) => next,
                        None => {
                            self.current_doc = u32::MAX;
                            self.current_tf = 0;
                            return Ok(());
                        }
                    };
                    continue 'docs;
                }
            }

            // Verify adjacency at the aligned doc.
            let tf = self.verify_at_aligned()?;
            if tf > 0 {
                self.current_doc = aligned;
                self.current_tf = tf;
                return Ok(());
            }
            from = match aligned.checked_add(1) {
                Some(next) => next,
                None => {
                    self.current_doc = u32::MAX;
                    self.current_tf = 0;
                    return Ok(());
                }
            };
            continue 'docs;
        }
    }

    /// Count the phrase's anchors at the members' aligned doc: the
    /// first member's positions `p` where member `i` also has `p + i`
    /// for every `i`. Member position lists are ascending, so each
    /// probe is a binary search over a per-doc-tf-sized slice.
    fn verify_at_aligned(&mut self) -> Result<u32, FtsError> {
        // Staged, rarest-first, lazy-decode verification. A phrase match
        // starting at position `s` has member `j` at `s + j`, so any
        // member can seed the candidate starts: the rarest member (by
        // posting length — `align_order[0]`) seeds them, then each
        // remaining member filters the survivors *in rarest-first order*.
        //
        // Two wins over decode-all-then-probe-from-the-first-member:
        //   * The seed loop is as short as the rarest member's per-doc tf,
        //     not the (often common) query-first member's.
        //   * Decoding is lazy: a common member's positions — whose
        //     per-block run-offset walk is the real cost — are only
        //     decoded once some candidate survives every rarer member.
        //     On the huge co-occurrence sets a phrase with a common word
        //     produces, almost every doc is rejected by a rare member
        //     first, so the common members are never decoded there.
        let anchor = self.align_order[0];
        let anchor_off = anchor as u32;
        self.members[anchor].decode_current_positions()?;
        self.verify_scratch.clear();
        for &pa in &self.members[anchor].pos_scratch {
            if let Some(start) = pa.checked_sub(anchor_off) {
                self.verify_scratch.push(start);
            }
        }
        for oi in 1..self.align_order.len() {
            if self.verify_scratch.is_empty() {
                break;
            }
            let j = self.align_order[oi];
            self.members[j].decode_current_positions()?;
            let plist = &self.members[j].pos_scratch;
            let off = j as u32;
            // Compact the survivors in place: keep a start iff member `j`
            // holds `start + j`.
            let mut w = 0usize;
            for r in 0..self.verify_scratch.len() {
                let start = self.verify_scratch[r];
                let keep = start
                    .checked_add(off)
                    .is_some_and(|want| plist.binary_search(&want).is_ok());
                if keep {
                    self.verify_scratch[w] = start;
                    w += 1;
                }
            }
            self.verify_scratch.truncate(w);
        }
        Ok(self.verify_scratch.len() as u32)
    }

    /// Score the phrase at its current doc with the caller-supplied
    /// per-doc BM25 normalization.
    #[inline]
    fn score_current(&self, dl_norm_k1: f32) -> f32 {
        bm25::score_with_dl_norm_k1(self.idf_x_k1p1, self.current_tf, dl_norm_k1)
    }

    /// Phrase-scaled block-level upper bound over `[range_start,
    /// range_end]` — the block analog of `term_max_bm25`.
    fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        let mut min_scaled = f32::INFINITY;
        for m in self.members.iter_mut() {
            let b = m.cursor.block_max_in_range(range_start, range_end);
            min_scaled = min_scaled.min(b / m.idf);
        }
        let idf_sum = self.idf_x_k1p1 / (bm25::K1 + 1.0);
        idf_sum * min_scaled
    }
}

/// A query atom's cursor: a plain term or an exact phrase. The atom
/// walks below are heterogeneous doc-at-a-time loops over this enum —
/// deliberately separate from the field-level optimized kernels
/// (flat-merge AND, MaxScore/BMM, windowed union), which keep serving
/// term-only queries unchanged. A query containing any phrase routes
/// here: correctness-first walks whose per-doc cost is dominated by
/// the phrase verification itself.
enum AnyCursor {
    Term(TermCursor),
    Phrase(PhraseCursor),
}

impl AnyCursor {
    #[inline]
    fn is_exhausted(&self) -> bool {
        match self {
            AnyCursor::Term(c) => c.is_exhausted(),
            AnyCursor::Phrase(c) => c.is_exhausted(),
        }
    }

    #[inline]
    fn current_doc_id(&self) -> u32 {
        match self {
            AnyCursor::Term(c) => c.current_doc_id(),
            AnyCursor::Phrase(c) => c.current_doc_id(),
        }
    }

    /// Advance to the first (phrase: first *verified*) doc ≥ `target`.
    fn skip_to(&mut self, target: u32) -> Result<(), FtsError> {
        match self {
            AnyCursor::Term(c) => {
                c.skip_to(target);
                Ok(())
            }
            AnyCursor::Phrase(c) => c.skip_to(target),
        }
    }

    /// [`Self::skip_to`] with the ranked walks' pruning bar: a phrase
    /// atom skips docs it provably can't lift over the bar without
    /// doing any position work (see [`PhraseCursor::skip_to_pruned`]).
    /// Term atoms ignore the bar — their per-doc score costs nothing
    /// beyond the postings walk itself.
    fn skip_to_pruned(
        &mut self,
        target: u32,
        bar: f32,
        dl_norm_k1: &NormTable,
    ) -> Result<(), FtsError> {
        match self {
            AnyCursor::Term(c) => {
                c.skip_to(target);
                Ok(())
            }
            AnyCursor::Phrase(c) => c.skip_to_pruned(target, bar, dl_norm_k1),
        }
    }

    /// BM25 contribution at the cursor's current doc.
    #[inline]
    fn score_current(&self, dl_norm_k1: f32) -> f32 {
        match self {
            AnyCursor::Term(c) => {
                bm25::score_with_dl_norm_k1(c.idf_x_k1p1, c.current_tf(), dl_norm_k1)
            }
            AnyCursor::Phrase(c) => c.score_current(dl_norm_k1),
        }
    }

    /// Atom-level score upper bound (any doc).
    #[inline]
    fn term_max_bm25(&self) -> f32 {
        match self {
            AnyCursor::Term(c) => c.term_max_bm25,
            AnyCursor::Phrase(c) => c.term_max_bm25,
        }
    }

    /// Score upper bound over the doc range (see the cursors' docs).
    #[inline]
    fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        match self {
            AnyCursor::Term(c) => c.block_max_in_range(range_start, range_end),
            AnyCursor::Phrase(c) => c.block_max_in_range(range_start, range_end),
        }
    }
}

/// Atom-walk exclusion gate: the heterogeneous sibling of
/// [`ExcludeFilter`], additionally able to exclude docs containing a
/// negated *phrase*. Same monotonic-doc contract.
struct AtomExcludeFilter {
    atoms: Vec<AnyCursor>,
    last_doc: u32,
}

impl AtomExcludeFilter {
    fn new(atoms: Vec<AnyCursor>) -> Self {
        Self { atoms, last_doc: 0 }
    }

    /// `false` iff `doc` matches any negated atom.
    fn admits(&mut self, doc: u32) -> Result<bool, FtsError> {
        debug_assert!(
            doc >= self.last_doc,
            "AtomExcludeFilter fed non-monotonic doc: {doc} < {}",
            self.last_doc
        );
        self.last_doc = doc;
        for a in &mut self.atoms {
            a.skip_to(doc)?;
            if !a.is_exhausted() && a.current_doc_id() == doc {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Exclusion gate for negated (`-term`) clauses: holds one
/// [`TermCursor`] per negated term, streamed with `skip_to` (a common
/// negated list is never fully decoded). A doc is rejected if it appears
/// in any negated term's list.
///
/// Kernels take `Option<&mut ExcludeFilter>` (`None` = no negation)
/// rather than a generic filter parameter: monomorphizing the OR kernel
/// measured 25-30% slower even with a no-op filter, while the `None`
/// branch is constant per query, perfectly predicted, and free.
pub(crate) struct ExcludeFilter {
    cursors: Vec<TermCursor>,
    /// Last doc-id passed to `admits`; guards the monotonic call order.
    last_doc: u32,
}

impl ExcludeFilter {
    fn new(cursors: Vec<TermCursor>) -> Self {
        Self {
            cursors,
            last_doc: 0,
        }
    }

    /// Posting-list bytes the negation cursors index into — see
    /// [`PreparedClauses::postings_bytes`].
    fn postings_bytes(&self) -> u64 {
        term_cursor_bytes(&self.cursors)
    }

    /// Byte-source ranges the negation cursors' builds requested (one per
    /// PFOR term) — see [`PreparedClauses::planned_ranges`].
    fn planned_ranges(&self) -> u64 {
        term_cursor_ranges(&self.cursors)
    }
}

impl ExcludeFilter {
    /// `false` iff `doc` is in any negated list.
    ///
    /// `doc` must be non-decreasing across a search: `skip_to` only
    /// moves forward. Every kernel walks candidates ascending, so this
    /// holds; the debug-assert guards a future caller that breaks it.
    #[inline]
    fn admits(&mut self, doc: u32) -> bool {
        debug_assert!(
            doc >= self.last_doc,
            "ExcludeFilter fed non-monotonic doc: {doc} < {}",
            self.last_doc
        );
        self.last_doc = doc;
        for c in &mut self.cursors {
            c.skip_to(doc);
            if !c.is_exhausted() && c.current_doc_id() == doc {
                return false;
            }
        }
        true
    }
}

/// Per-hit action for the multi-term AND flat-merge intersection.
///
/// The traversal in [`FtsReader::and_flat_merge_general`] /
/// [`FtsReader::and_flat_merge_2term`] — cursor alignment, block
/// crossing, and the in-block pointer walk — runs identically whether
/// the caller wants ranked hits or just the matching doc ids. Only the
/// action at each converged doc differs, so both go through one
/// traversal parameterized by this trait and cannot disagree on which
/// docs match.
///
/// [`ScoreSink`] computes BM25 and feeds a top-k heap (the ranked
/// search path); [`CollectSink`] records the doc id and computes no
/// score (the unranked `token_match` / count path). The traversal is
/// monomorphized per sink, so `needs_score()` folds to a constant: the
/// scorer compiles to a dedicated copy with scoring inlined, and the
/// collector's copy drops the scoring arithmetic as dead code.
trait AndSink {
    /// Block-max pruning bar: docs whose block can't reach this score
    /// are skipped. Returning `NEG_INFINITY` (the default) disables
    /// pruning, which is what an unranked sink wants — it has no score
    /// threshold to prune against.
    fn bar(&self) -> f32 {
        f32::NEG_INFINITY
    }

    /// Whether the traversal should compute a hit's BM25 score. A sink
    /// that returns `false` skips all scoring arithmetic — what makes an
    /// unranked count over a large intersection cheaper than ranking it.
    fn needs_score(&self) -> bool;

    /// Record one doc in the intersection. `score` is meaningful only
    /// when [`needs_score`](AndSink::needs_score) returns `true`;
    /// otherwise it is `0.0` and ignored.
    fn emit(&mut self, doc: u32, score: f32);
}

/// Ranked sink: floor-gates each hit and pushes it into the top-k heap.
struct ScoreSink<'a> {
    heap: &'a mut BinaryHeap<TopKEntry>,
    k: usize,
    filter: Option<&'a mut ExcludeFilter>,
    floor_eff: f32,
}

impl AndSink for ScoreSink<'_> {
    fn bar(&self) -> f32 {
        // kth-best once the heap fills, else the caller's seeded floor —
        // whichever is higher.
        if self.heap.len() >= self.k {
            self.heap
                .peek()
                .expect("heap len == k")
                .0
                .max(self.floor_eff)
        } else {
            self.floor_eff
        }
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, score: f32) {
        // Floor gate: strictly-below-floor docs are dead to the caller.
        if score > self.floor_eff {
            and_heap_push(self.heap, self.k, self.filter.as_deref_mut(), score, doc);
        }
    }
}

/// Ranked must+should sink: scores the must intersection like
/// [`ScoreSink`], then adds each should term's contribution for docs
/// it lands on before heap admission. The should cursors ride inside
/// the sink — the AND walk emits docs in ascending order, so each
/// should list is `skip_to`-streamed forward at most once per query,
/// exactly like [`ExcludeFilter`]'s negated cursors.
struct MustShouldSink<'a> {
    heap: &'a mut BinaryHeap<TopKEntry>,
    k: usize,
    filter: Option<&'a mut ExcludeFilter>,
    floor_eff: f32,
    shoulds: Vec<TermCursor>,
    /// Σ `term_max_bm25` over the should cursors — the most the
    /// shoulds can add to any single doc's score.
    should_ub: f32,
    /// Per-doc BM25 length normalization for the column, for scoring
    /// the should terms at emitted docs.
    dl_norm_k1: &'a NormTable,
}

impl AndSink for MustShouldSink<'_> {
    fn bar(&self) -> f32 {
        // The AND walk's block-max arithmetic bounds the MUST portion
        // of a doc's score only, so the pruning bar is lowered by the
        // most the shoulds could add: a must block that can't reach
        // (kth-best − should_ub) can't produce a top-k doc even with
        // every should matching at its maximum.
        let full_bar = if self.heap.len() >= self.k {
            self.heap
                .peek()
                .expect("heap len == k")
                .0
                .max(self.floor_eff)
        } else {
            self.floor_eff
        };
        full_bar - self.should_ub
    }

    fn needs_score(&self) -> bool {
        true
    }

    fn emit(&mut self, doc: u32, must_score: f32) {
        let norm = self.dl_norm_k1.get(doc);
        let mut score = must_score;
        for c in &mut self.shoulds {
            c.skip_to(doc);
            if !c.is_exhausted() && c.current_doc_id() == doc {
                score += bm25::score_with_dl_norm_k1(c.idf_x_k1p1, c.current_tf(), norm);
            }
        }
        // Floor gate on the FULL score — a must-only score below the
        // floor can still survive once its shoulds are added.
        if score > self.floor_eff {
            and_heap_push(self.heap, self.k, self.filter.as_deref_mut(), score, doc);
        }
    }
}

/// Unranked sink: collect the matching doc ids in ascending order, no
/// scoring, no top-k. Drives the `token_match` AND path through the
/// same optimized flat-merge the scorer uses.
struct CollectSink {
    out: Vec<u32>,
}

impl AndSink for CollectSink {
    fn needs_score(&self) -> bool {
        false
    }

    fn emit(&mut self, doc: u32, _score: f32) {
        self.out.push(doc);
    }
}

/// Unranked counting sink: tally the intersection size without
/// materializing the ids. Drives the count path through the same
/// flat-merge as [`CollectSink`] but skips the `Vec<u32>` — for a
/// high-cardinality count that allocation (4 bytes/doc) is pure waste.
struct CountSink {
    n: u64,
}

impl AndSink for CountSink {
    fn needs_score(&self) -> bool {
        false
    }

    fn emit(&mut self, _doc: u32, _score: f32) {
        self.n += 1;
    }
}

/// Push `(score, doc_id)` into the top-k AND heap with the same
/// tie-break (asc doc_id) the OR paths use, so AND and OR rankings
/// agree on score-tied docs.
///
/// `filter` drops docs excluded by a negated (`-term`) clause before
/// they enter the heap; `None` admits everything.
#[inline]
fn and_heap_push(
    heap: &mut BinaryHeap<TopKEntry>,
    k: usize,
    filter: Option<&mut ExcludeFilter>,
    score: f32,
    doc_id: u32,
) {
    if let Some(f) = filter
        && !f.admits(doc_id)
    {
        return;
    }
    if heap.len() < k {
        heap.push(TopKEntry(score, doc_id));
    } else if let Some(&worst) = heap.peek()
        && (score > worst.0 || (score == worst.0 && doc_id < worst.1))
    {
        heap.pop();
        heap.push(TopKEntry(score, doc_id));
    }
}

/// Merge a `doc_id -> score` map into top-k by descending score, ties
/// broken by ascending doc_id. Used by `search_multi`'s cross-column
/// combiner, where the per-column scores have already been weighted
/// and summed into `scores`.
fn top_k(scores: HashMap<u32, f32>, k: usize) -> Vec<(u32, f32)> {
    // Iterate in ascending doc_id order so ties resolve deterministically
    // (smaller doc_ids enter the heap first; the strict `score > peek`
    // check below means subsequent equal-score entries don't displace
    // them). Without this, HashMap's hash-order iteration would make the
    // tied result non-deterministic and would disagree with the BMW
    // single-term path (which naturally iterates in doc_id order).
    // pdqsort: doc_ids are unique by construction (HashMap keys).
    let mut sorted: Vec<(u32, f32)> = scores.into_iter().collect();
    sorted.sort_unstable_by_key(|(d, _)| *d);

    let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::with_capacity(k.min(sorted.len()).max(1));
    for (doc_id, score) in sorted {
        if heap.len() < k {
            heap.push(TopKEntry(score, doc_id));
        } else if let Some(TopKEntry(top_score, _)) = heap.peek()
            && score > *top_score
        {
            heap.pop();
            heap.push(TopKEntry(score, doc_id));
        }
    }
    drain_top_k_desc(heap)
}

fn fetch_source_range(source: &Source, range: Range<usize>, what: &str) -> Result<Bytes, FtsError> {
    source.get_range(range).map_err(|e| {
        FtsError::Read(ReadError::MalformedVersion(format!(
            "{what} lazy source range fetch failed: {e}"
        )))
    })
}

async fn fetch_lazy_range(
    source: &dyn LazyByteSource,
    range: Range<usize>,
    what: &str,
) -> Result<Bytes, FtsError> {
    source
        .range(range.start as u64, range.len() as u64)
        .await
        .map_err(|e| {
            FtsError::Read(ReadError::MalformedVersion(format!(
                "{what} lazy source range fetch failed: {e}"
            )))
        })
}

#[inline]
fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
fn read_u64_le(b: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(buf)
}

/// Unranked multi-term OR walk: the union of the cursors' doc ids in
/// ascending order. A k-way merge — each step finds the minimum current
/// doc id across the live cursors, hands it to `emit`, and advances
/// every cursor sitting on it (so the next minimum is strictly greater
/// and `emit` is called exactly once per distinct doc). No scoring; the
/// caller wants membership, not rank.
fn or_walk_unranked(mut cursors: Vec<TermCursor>, mut emit: impl FnMut(u32)) {
    loop {
        let min_doc = cursors
            .iter()
            .filter(|c| !c.is_exhausted())
            .map(TermCursor::current_doc_id)
            .min();
        let Some(min_doc) = min_doc else { break };
        emit(min_doc);
        for c in cursors.iter_mut() {
            if !c.is_exhausted() && c.current_doc_id() == min_doc {
                c.next();
            }
        }
    }
}

/// The union's doc ids ([`or_walk_unranked`] collected into a `Vec`).
fn or_merge_unranked(cursors: Vec<TermCursor>) -> Vec<u32> {
    let mut out = Vec::new();
    or_walk_unranked(cursors, |doc| out.push(doc));
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
fn or_count_unranked(mut cursors: Vec<TermCursor>) -> u64 {
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
            while !c.is_exhausted() {
                let d = c.current_doc_id();
                if d >= window_end {
                    break;
                }
                let local = (d - base) as usize;
                present[local >> 6] |= 1u64 << (local & 63);
                c.next();
            }
        }
        // Count distinct docs in this window and clear for reuse.
        for word in present.iter_mut() {
            n += word.count_ones() as u64;
            *word = 0;
        }
    }
    n
}

/// Read `postings_length` out of a term metadata header, given only
/// enough bytes to cover that field.
fn header_postings_length(header: &[u8]) -> Result<usize, FtsError> {
    let field_end = term_meta::POSTINGS_LENGTH_OFF + U32_BYTES;
    if header.len() < field_end {
        return Err(FtsError::Read(ReadError::MalformedVersion(
            "term metadata header shorter than its postings_length field".into(),
        )));
    }
    Ok(read_u32_le(&header[term_meta::POSTINGS_LENGTH_OFF..field_end]) as usize)
}

/// Parsed per-(column, term) metadata header from the postings
/// region. The byte layout is documented once, on the writer side —
/// see [`TERM_META_SIZE`] in `builder.rs` — this struct is its
/// read-side mirror and must stay in sync with that doc.
///
/// [`TermMeta::parse`] is the single place that validates untrusted
/// offsets (the FST value points here) against the postings region:
/// both the fixed 20-byte header and the skip table it declares are
/// bounds-checked before any caller touches a byte. Both the
/// single-term BMW path and [`TermCursor::new`] go through here, so
/// the header layout is interpreted in exactly one spot.
#[derive(Debug, Copy, Clone)]
struct TermMeta {
    /// Document frequency — number of docs containing the term.
    df: u64,
    /// Byte length of the term's whole region (header + skip table +
    /// blocks), relative to the term's `metadata_offset`.
    postings_length: usize,
    /// Number of PFOR blocks (= number of skip-table entries).
    num_blocks: usize,
    /// Absolute offset (within the postings region) of the first
    /// skip-table entry: `metadata_offset + TERM_META_SIZE`.
    skip_start: usize,
    /// This term's byte offset in the positions region (positional
    /// columns; zero otherwise).
    positions_offset: u64,
    /// Byte length of this term's position runs (positional columns;
    /// zero otherwise).
    positions_length: u32,
}

impl TermMeta {
    /// Parse + bounds-validate the header and its skip table.
    /// Returns `Err` (never panics) on a corrupt or malicious
    /// `metadata_offset` — the crate-wide "untrusted input yields
    /// `Err`, not a slice-index panic" rule.
    fn parse(postings: &[u8], metadata_offset: usize, positional: bool) -> Result<Self, FtsError> {
        // Positional columns carry the extended 32-byte header (the
        // term's positions offset + length after `num_blocks`); the
        // skip table starts after whichever stride applies. The
        // positions fields themselves are consumed by the phrase read
        // path, not here.
        let term_meta_size = match positional {
            true => TERM_META_POSITIONAL_SIZE,
            false => TERM_META_SIZE,
        };
        if metadata_offset + term_meta_size > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "term metadata offset out of postings region".into(),
            )));
        }
        let df = read_u32_le(
            &postings[metadata_offset + term_meta::DF_OFF
                ..metadata_offset + term_meta::DF_OFF + U32_BYTES],
        ) as u64;
        // bytes [4..12] = self-offset (redundant; u64); skip
        let postings_length = read_u32_le(
            &postings[metadata_offset + term_meta::POSTINGS_LENGTH_OFF
                ..metadata_offset + term_meta::POSTINGS_LENGTH_OFF + U32_BYTES],
        ) as usize;
        let num_blocks = read_u32_le(
            &postings[metadata_offset + term_meta::NUM_BLOCKS_OFF
                ..metadata_offset + term_meta::NUM_BLOCKS_OFF + U32_BYTES],
        ) as usize;

        let (positions_offset, positions_length) = match positional {
            true => (
                read_u64_le(
                    &postings[metadata_offset + term_meta::POSITIONS_OFFSET_OFF
                        ..metadata_offset + term_meta::POSITIONS_OFFSET_OFF + U64_BYTES],
                ),
                read_u32_le(
                    &postings[metadata_offset + term_meta::POSITIONS_LENGTH_OFF
                        ..metadata_offset + term_meta::POSITIONS_LENGTH_OFF + U32_BYTES],
                ),
            ),
            false => (0, 0),
        };

        // The last block's end offset comes straight from
        // `postings_length`; bound it now instead of slicing OOB later.
        if metadata_offset + postings_length > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "term postings length exceeds the fetched term range".into(),
            )));
        }
        let skip_start = metadata_offset + term_meta_size;
        let skip_end = skip_start + num_blocks * SKIP_ENTRY_SIZE;
        if skip_end > postings.len() {
            return Err(FtsError::Read(ReadError::MalformedVersion(
                "skip table runs past postings region".into(),
            )));
        }
        Ok(Self {
            df,
            postings_length,
            num_blocks,
            skip_start,
            positions_offset,
            positions_length,
        })
    }

    /// Decode skip-table entry `i` into `(last_doc_id,
    /// block_offset_in_term, block_max_bm25)`. `block_offset_in_term`
    /// is relative to the term's `metadata_offset`; `block_max_bm25`
    /// is recovered from the fixed-point `max_bm25_x1000` field. The
    /// reserved field (entry bytes 12..16) is ignored. Per-entry on
    /// purpose — the single-term BMW walk streams entries without
    /// materializing a `Vec`.
    #[inline]
    fn skip_entry(&self, postings: &[u8], i: usize) -> (u32, usize, f32) {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        let entry_off = self.skip_start + i * SKIP_ENTRY_SIZE;
        let last_doc_id = read_u32_le(
            &postings[entry_off + skip_entry::LAST_DOC_ID_OFF
                ..entry_off + skip_entry::LAST_DOC_ID_OFF + U32_BYTES],
        );
        let block_offset = read_u32_le(
            &postings[entry_off + skip_entry::BLOCK_OFFSET_OFF
                ..entry_off + skip_entry::BLOCK_OFFSET_OFF + U32_BYTES],
        ) as usize;
        let max_bm25_x1000 = read_u32_le(
            &postings[entry_off + skip_entry::MAX_BM25_OFF
                ..entry_off + skip_entry::MAX_BM25_OFF + U32_BYTES],
        );
        // Decode to a guaranteed upper bound on the block's BM25. The
        // builder ceil()s on encode, but `x1000 as f32 / SCALE` can still
        // round a hair below the true max (f32 division), and superfiles
        // written before the encode-side ceil truncated outright. Add one
        // fixed-point step before unscaling so the decoded bound is always
        // >= the true block max. This matters for the cross-superfile
        // floor: block-skip compares `block_max <= floor`, and a bound
        // that dips below a score-tied block's true max would let a rising
        // floor skip that block, dropping tied hits by completion order
        // (nondeterministic top-k). The +1 step costs ~1/SCALE of pruning
        // tightness — negligible — and keeps the top-k deterministic.
        (
            last_doc_id,
            block_offset,
            max_bm25_x1000.saturating_add(1) as f32 / format::fts::BLOCK_MAX_BM25_FIXED_POINT_SCALE,
        )
    }

    /// This block's position-run byte offset within the term's
    /// positions bytes — the skip entry's fourth field (zero on
    /// positionless columns, where it is the reserved slot).
    #[inline]
    fn positions_block_offset(&self, postings: &[u8], i: usize) -> u32 {
        debug_assert!(i < self.num_blocks, "skip entry {i} >= {}", self.num_blocks);
        let entry_off = self.skip_start + i * SKIP_ENTRY_SIZE;
        read_u32_le(
            &postings[entry_off + skip_entry::POSITIONS_BLOCK_OFFSET_OFF
                ..entry_off + skip_entry::POSITIONS_BLOCK_OFFSET_OFF + U32_BYTES],
        )
    }

    /// End offset (relative to the term's `metadata_offset`) of block
    /// `i`'s bytes. Blocks are concatenated back-to-back, so each
    /// block ends where the next one's `block_offset` begins; the last
    /// block ends at `postings_length`.
    #[inline]
    fn block_end_in_term(&self, postings: &[u8], i: usize) -> usize {
        if i + 1 < self.num_blocks {
            let next_off = self.skip_start + (i + 1) * SKIP_ENTRY_SIZE;
            read_u32_le(&postings[next_off + 4..next_off + 8]) as usize
        } else {
            self.postings_length
        }
    }
}

/// Per-term per-block metadata, parsed once at `TermCursor` construction.
#[derive(Debug, Clone, Copy)]
struct BlockMeta {
    /// Largest doc_id present in this block.
    last_doc_id: u32,
    /// Absolute byte offset (within the FTS postings region) of this
    /// block's encoded bytes.
    block_byte_offset: usize,
    /// Absolute byte offset of the first byte AFTER this block. For
    /// the last block of a term it's `metadata_offset + postings_length`.
    block_byte_end: usize,
    /// Per-block BM25 upper bound, recovered from the skip table's
    /// fixed-point `max_bm25_x1000` field.
    block_max_bm25: f32,
}

/// Per-query-term cursor used by [`FtsReader::run_max_score_bmm`]
/// (and by [`FtsReader::run_wand_bmw`] in the bench-only path).
///
/// State:
///   - `blocks`: parsed skip table — one entry per block, lets us
///     decide whether to decode a block before paying the cost.
///   - `current_block` + `pos`: where we are in the term's posting
///     list. `pos == block_n` is treated as "advance to next block".
///   - `block_doc_ids` / `block_tfs`: decoded buffers for the current
///     block, reused across blocks.
///
/// `current_doc_id() == u32::MAX` is the "exhausted" sentinel; the
/// WAND loop drops cursors that are exhausted at the top of each
/// iteration.
#[derive(Clone)]
pub(crate) struct TermCursor {
    /// Precomputed `idf * (K1 + 1)` — the score numerator's
    /// per-cursor constant. Computed once at cursor build so the
    /// hot inner loop fits one multiply + add + divide per call.
    /// (The bare `idf` value isn't kept on the cursor — every hot
    /// scoring path uses `score_with_dl_norm_k1` which takes
    /// `idf_x_k1p1` directly.)
    idf_x_k1p1: f32,
    /// Maximum block-max-BM25 across all blocks. Used by the WAND
    /// pivot test (term-level upper bound).
    term_max_bm25: f32,
    /// Document frequency of the term (postings list length). Used by
    /// the 2-term OR router to detect a rare anchor term (short list),
    /// where WAND+BMW can skip the other term's long list.
    df: u64,
    /// Per-block metadata (the parsed skip table). Read-only after
    /// build and `Arc`-shared, so cloning a cursor for another doc-id
    /// sub-range costs the ~1 KiB decode buffers, never a re-parse.
    blocks: Arc<[BlockMeta]>,
    /// Decoded buffers for the current block. Reused across decodes.
    block_doc_ids: Vec<u32>,
    block_tfs: Vec<u32>,
    /// Number of valid entries in the decoded block buffers (the
    /// last block may be partial).
    block_n: usize,
    /// Index into `blocks` of the currently-decoded block. Equal to
    /// `blocks.len()` once exhausted.
    current_block: usize,
    /// Position within the currently-decoded block. Always `<
    /// block_n` while not exhausted.
    pos: usize,
    /// Index into `blocks` of the block being inspected by the BMW
    /// upper-bound check. Standard block-cursor split:
    /// `shallow_advance_block_to(pivot_doc)` updates this without
    /// decoding the block, so subsequent BMW UB lookups for
    /// monotonically-increasing pivot docs are amortized O(1). Always
    /// `>= current_block`; synced up whenever `current_block` is
    /// advanced.
    inspect_block: usize,
    /// This term's own postings bytes — the metadata header (offset
    /// 0), skip table, and encoded blocks, fetched as a single
    /// contiguous range by [`FtsReader::fetch_term_postings`]. All
    /// `BlockMeta` byte offsets are relative to the start of this
    /// buffer. Empty for inline (df=1) cursors, which never decode.
    /// Mirrors the vector reader's per-probed-cluster buffers: the
    /// search hot loops index only the bytes this term touches, never
    /// the whole postings region.
    ///
    /// Deliberately carries NO positional state: term cursors are the
    /// hot per-query unit the multi-cursor kernels iterate over, and
    /// the positional extras matter only to phrase members —
    /// [`PhraseMember`] re-derives them from these bytes instead, so
    /// plain term queries never pay for them in cursor or block-meta
    /// footprint.
    bytes: Bytes,
    /// True when this term's FST slot carried no postings-length hint,
    /// so the build probed the 20-byte header before fetching the body
    /// — two planned byte-source ranges instead of one.
    header_probed: bool,
}

impl TermCursor {
    /// Parse one term's metadata + skip table out of its own postings
    /// byte range and decode its first block. `term_bytes` starts at
    /// the term's 20-byte metadata header (offset 0) and runs to the
    /// end of its last block — the contiguous range
    /// [`FtsReader::fetch_term_postings`] fetched for this term.
    fn new(
        term_bytes: Bytes,
        n_docs: u64,
        positional: bool,
        global_idf: Option<f32>,
        header_probed: bool,
    ) -> Result<Self, FtsError> {
        let postings: &[u8] = term_bytes.as_ref();
        let metadata_offset = 0usize;

        let term_meta = TermMeta::parse(postings, metadata_offset, positional)?;
        let local_idf = bm25::idf(n_docs, term_meta.df);
        let idf = global_idf.unwrap_or(local_idf);
        // Stored per-block BMW upper bounds bake in the LOCAL idf. Only a
        // global-idf override needs to rescale them by global/local:
        // block_max = local_idf_x_k1p1 × (an idf-independent tf-factor),
        // so the linear rescale is exact and keeps the BMW skip UBs
        // consistent with the global-idf scores computed from
        // `idf_x_k1p1` below. `None` (the default per-superfile path, and
        // the case where a gathered global idf happens to equal the
        // local one) leaves the stored value untouched — the block loop
        // does no extra work, matching the per-superfile scorer exactly.
        let idf_rescale = match global_idf {
            Some(_) if local_idf > 0.0 && idf != local_idf => Some(idf / local_idf),
            _ => None,
        };

        // Collect straight into the `Arc` allocation: `0..num_blocks` is
        // an exact-size iterator, so this writes each entry in place —
        // one allocation, no intermediate `Vec` + copy. The skip table
        // is ~a quarter of a long term's cursor-build bytes (one 32-byte
        // entry per 128-doc block), so the doubled write showed up on
        // common-term queries.
        let mut term_max_bm25: f32 = 0.0;
        let blocks: Arc<[BlockMeta]> = (0..term_meta.num_blocks)
            .map(|i| {
                let (last_doc_id, block_offset_in_term, raw_block_max) =
                    term_meta.skip_entry(postings, i);
                let block_max_bm25 = match idf_rescale {
                    Some(ratio) => raw_block_max * ratio,
                    None => raw_block_max,
                };
                term_max_bm25 = term_max_bm25.max(block_max_bm25);

                BlockMeta {
                    last_doc_id,
                    block_byte_offset: metadata_offset + block_offset_in_term,
                    block_byte_end: metadata_offset + term_meta.block_end_in_term(postings, i),
                    block_max_bm25,
                }
            })
            .collect();

        let mut cursor = Self {
            idf_x_k1p1: idf * (bm25::K1 + 1.0),
            term_max_bm25,
            df: term_meta.df,
            blocks,
            block_doc_ids: vec![0u32; BLOCK_LEN],
            block_tfs: vec![0u32; BLOCK_LEN],
            block_n: 0,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: term_bytes,
            header_probed,
        };
        if !cursor.blocks.is_empty() {
            cursor.decode_current_block();
        }
        Ok(cursor)
    }

    /// Synthesize a cursor for a df=1 inline-encoded term. Skips the
    /// postings-region read entirely — the caller already has
    /// (doc_id, tf) from unpacking the FST value, and BMW upper bound
    /// for a 1-doc term equals that doc's actual BM25 score (only one
    /// doc means min_dl = dl and max_tf = tf, so the per-block UB
    /// formula collapses to the score itself). Computed at query time
    /// since there's no skip-table entry stored for inline terms.
    fn new_inline(
        doc_id: u32,
        tf: u32,
        n_docs: u64,
        dl_norm_k1: f32,
        global_idf: Option<f32>,
    ) -> Self {
        let idf = global_idf.unwrap_or_else(|| bm25::idf(n_docs, 1));
        let idf_x_k1p1 = idf * (bm25::K1 + 1.0);
        let block_max_bm25 = bm25::score_with_dl_norm_k1(idf_x_k1p1, tf, dl_norm_k1);

        let blocks: Arc<[BlockMeta]> = Arc::from([BlockMeta {
            last_doc_id: doc_id,
            // No postings-region bytes back this cursor; the decoded
            // buffer is pre-filled below so `decode_current_block` is
            // never called against these offsets.
            block_byte_offset: 0,
            block_byte_end: 0,
            block_max_bm25,
        }]);

        let mut block_doc_ids = vec![0u32; BLOCK_LEN];
        let mut block_tfs = vec![0u32; BLOCK_LEN];
        block_doc_ids[0] = doc_id;
        block_tfs[0] = tf;

        Self {
            idf_x_k1p1,
            term_max_bm25: block_max_bm25,
            df: 1,
            blocks,
            block_doc_ids,
            block_tfs,
            block_n: 1,
            current_block: 0,
            pos: 0,
            inspect_block: 0,
            bytes: Bytes::new(),
            header_probed: false,
        }
    }

    fn decode_current_block(&mut self) {
        let block = self.blocks[self.current_block];
        let bytes = self
            .bytes
            .slice(block.block_byte_offset..block.block_byte_end);
        self.block_n = decode_block(&bytes, &mut self.block_doc_ids, &mut self.block_tfs);
        self.pos = 0;
    }

    fn is_exhausted(&self) -> bool {
        self.current_block >= self.blocks.len()
    }

    /// Block count, used as a cheap proxy for df when AND intersection
    /// picks the rarest cursor as the leader. Block count is an exact
    /// upper bound on df: a term's df is `(blocks - 1) * BLOCK_LEN +
    /// last_block_n`, so cursors compare in the same order by block
    /// count as they do by df. Inline cursors return 1.
    #[inline(always)]
    fn block_count(&self) -> usize {
        self.blocks.len()
    }

    #[inline(always)]
    fn current_doc_id(&self) -> u32 {
        if self.is_exhausted() || self.pos >= self.block_n {
            u32::MAX
        } else {
            self.block_doc_ids[self.pos]
        }
    }

    #[inline(always)]
    fn current_tf(&self) -> u32 {
        debug_assert!(!self.is_exhausted() && self.pos < self.block_n);
        self.block_tfs[self.pos]
    }

    #[inline(always)]
    fn current_block_max_bm25(&self) -> f32 {
        if self.is_exhausted() {
            0.0
        } else {
            self.blocks[self.current_block].block_max_bm25
        }
    }

    /// Largest doc_id in the cursor's current block. Used by the BMW
    /// skip step to compute the smallest "next interesting doc_id"
    /// across the prefix.
    #[inline(always)]
    fn current_block_last_doc_id(&self) -> u32 {
        if self.is_exhausted() {
            u32::MAX
        } else {
            self.blocks[self.current_block].last_doc_id
        }
    }

    /// Shallow-advance the inspect-block pointer to the block that
    /// would contain `target`. Does NOT decode and does NOT touch the
    /// doc cursor (`current_block`, `pos`, decoded buffers stay put);
    /// only the lightweight `inspect_block` index moves. Used by the
    /// BMW UB sum at `pivot_doc` for cursors whose current_doc lags
    /// pivot_doc — their relevant block-max is the block containing
    /// pivot_doc, not their current decoded block.
    ///
    /// Monotonically advances; calling this for monotonically-
    /// increasing `target` across WAND iterations gives amortized
    /// O(1) per call.
    fn shallow_advance_block_to(&mut self, target: u32) {
        // Never let inspect_block fall behind current_block — once
        // the doc cursor has decoded past a block, that block's
        // metadata is no longer relevant.
        if self.inspect_block < self.current_block {
            self.inspect_block = self.current_block;
        }
        while self.inspect_block < self.blocks.len()
            && self.blocks[self.inspect_block].last_doc_id < target
        {
            self.inspect_block += 1;
        }
    }

    /// Maximum `block_max_bm25` across all blocks of this cursor whose
    /// doc-id range overlaps `[range_start, range_end]` (inclusive on
    /// both ends). Used by AND block-max pruning to compute a safe
    /// upper bound on this cursor's contribution across the leader's
    /// current block — a single-block lookup at one boundary
    /// underestimates when the leader's range spans multiple
    /// cursor blocks with varying block_max. Uses `inspect_block` as
    /// a hint pointer so monotonically-advancing leader ranges amortize
    /// to O(1) amortized per call.
    fn block_max_in_range(&mut self, range_start: u32, range_end: u32) -> f32 {
        // Advance inspect_block to the first block whose last_doc_id
        // could intersect the range. shallow_advance_block_to lands on
        // the first block with last_doc_id >= range_start, which is
        // exactly the first block that can overlap the range.
        self.shallow_advance_block_to(range_start);
        let mut max: f32 = 0.0;
        let mut i = self.inspect_block;
        while i < self.blocks.len() {
            // Block i starts at the doc right after the previous block's
            // last_doc_id (or doc 0 if i == 0). Once block_start exceeds
            // range_end the rest of the blocks lie strictly past the
            // range; stop walking.
            let block_start = if i == 0 {
                0u32
            } else {
                self.blocks[i - 1].last_doc_id.saturating_add(1)
            };
            if block_start > range_end {
                break;
            }
            let m = self.blocks[i].block_max_bm25;
            if m > max {
                max = m;
            }
            i += 1;
        }
        max
    }

    /// Block-max-BM25 at the inspect-block pointer. Pair with
    /// `shallow_advance_block_to(pivot_doc)` to bound the cursor's
    /// contribution at pivot_doc.
    fn inspect_block_max_bm25(&self) -> f32 {
        if self.inspect_block >= self.blocks.len() {
            0.0
        } else {
            self.blocks[self.inspect_block].block_max_bm25
        }
    }

    /// Last doc_id in the block at the inspect-block pointer. Used
    /// for the BMW skip target — the smallest "next interesting doc"
    /// across the prefix is one past the smallest such block-end.
    fn inspect_block_last_doc_id(&self) -> u32 {
        if self.inspect_block >= self.blocks.len() {
            u32::MAX
        } else {
            self.blocks[self.inspect_block].last_doc_id
        }
    }

    /// Advance one position. Crosses block boundaries automatically;
    /// decodes the next block on demand.
    #[inline(always)]
    fn next(&mut self) {
        if self.is_exhausted() {
            return;
        }
        self.pos += 1;
        if self.pos >= self.block_n {
            self.advance_block();
        }
    }

    /// Advance a known in-block batch, crossing to the next block when
    /// `count` consumes its remaining postings. Unlike [`Self::next`],
    /// callers must not start at or advance past the decoded block end.
    #[inline(always)]
    fn advance_by(&mut self, count: usize) {
        debug_assert!(!self.is_exhausted());
        debug_assert!(count > 0 && self.pos + count <= self.block_n);
        self.pos += count;
        // The assertion above makes equality equivalent to `>=` here.
        if self.pos == self.block_n {
            self.advance_block();
        }
    }

    /// Move to and decode the next posting block, or mark the cursor
    /// exhausted when the current block is the last one.
    #[inline(always)]
    fn advance_block(&mut self) {
        self.current_block += 1;
        if self.current_block > self.inspect_block {
            self.inspect_block = self.current_block;
        }
        if self.current_block < self.blocks.len() {
            self.decode_current_block();
        }
    }

    /// Skip forward so `current_doc_id() >= target`. Uses the skip
    /// table to skip whole blocks when the entire block precedes
    /// `target`. Common-case fast path (target lies within the
    /// already-decoded current block) is just an inlined `pos++`
    /// scan — no re-decode, no `is_exhausted` rechecks.
    #[inline(always)]
    fn skip_to(&mut self, target: u32) {
        if self.is_exhausted() {
            return;
        }
        let cur_block = self.current_block;
        let cur_block_last = self.blocks[cur_block].last_doc_id;
        if cur_block_last >= target {
            // Fast path: target is in our currently-decoded block.
            // Just scan pos forward. The `current_doc_id() >= target`
            // guard from before is folded into this scan — if pos is
            // already at-or-past, the loop body doesn't execute.
            let n = self.block_n;
            while self.pos < n && self.block_doc_ids[self.pos] < target {
                self.pos += 1;
            }
            if self.pos < n {
                return;
            }
            // Walked off the end of the decoded block (rare under
            // skip-table invariants); fall through to cross-block.
        }
        self.skip_to_cross_block(target);
    }

    /// Cross-block path of `skip_to`: target is past the current
    /// decoded block. Advances `current_block` via the skip table,
    /// decodes the new block (only when crossing), and scans pos.
    /// Pulled out so the within-block fast path stays small enough
    /// to inline at every call site.
    #[cold]
    fn skip_to_cross_block(&mut self, target: u32) {
        while self.current_block < self.blocks.len()
            && self.blocks[self.current_block].last_doc_id < target
        {
            self.current_block += 1;
        }
        if self.current_block > self.inspect_block {
            self.inspect_block = self.current_block;
        }
        if self.is_exhausted() {
            return;
        }
        self.decode_current_block();
        while self.pos < self.block_n && self.block_doc_ids[self.pos] < target {
            self.pos += 1;
        }
        if self.pos >= self.block_n {
            self.current_block += 1;
            if self.current_block > self.inspect_block {
                self.inspect_block = self.current_block;
            }
            if self.current_block < self.blocks.len() {
                self.decode_current_block();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, sync::Arc};

    use super::*;
    use crate::superfile::{
        BytesLazyByteSource,
        fts::{builder::FtsBuilder, tokenize::AsciiLowerTokenizer},
    };

    fn build_blob() -> (Bytes, String) {
        // 3 docs, 1 column.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        b.add_doc(0, 0, "rust async runtime").expect("add doc");
        b.add_doc(0, 1, "tokio is a rust runtime").expect("add doc");
        b.add_doc(0, 2, "java spring boot").expect("add doc");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        (Bytes::from(bytes), json.to_string())
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
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), true)
            .expect("register positional column");
        b.add_doc(0, 0, "a b a").expect("add doc 0");
        b.add_doc(0, 1, "b a c").expect("add doc 1");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#;
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

    #[test]
    fn add_prebuilt_term_posting_round_trips_read_to_write() {
        use std::collections::BTreeMap;
        let json = r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#;

        // Build A from text.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut a = FtsBuilder::new(tok.clone());
        a.register_column("body".into(), true).expect("register a");
        a.add_doc(0, 0, "a b a").expect("a doc 0");
        a.add_doc(0, 1, "b a c").expect("a doc 1");
        let ra = FtsReader::open(Bytes::from(a.finish().expect("finish a")), json).expect("open a");

        // Build B by streaming A's postings straight into the prebuilt path —
        // no re-tokenization. Doc lengths carried over ("a b a"/"b a c" = 3/3).
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), true).expect("register b");
        ra.for_each_term_posting(0, |term, doc_id, tf, positions| {
            let term_str = std::str::from_utf8(term).expect("utf8 term");
            b.add_prebuilt_term_posting(0, term_str, doc_id, tf, positions)
                .expect("prebuilt push");
            Ok(())
        })
        .expect("feed prebuilt postings");
        b.set_prebuilt_doc_lengths(0, ra.read_doc_lengths(0).expect("doc lengths"));
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
        let json = r#"[{"name":"body","tokenizer":"ascii_lower","positions":true}]"#;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut a = FtsBuilder::new(tok.clone());
        a.register_column("body".into(), true).expect("register a");
        a.add_doc(0, 0, "a b c a").expect("a doc 0");
        a.add_doc(0, 1, "b c d").expect("a doc 1");
        a.add_doc(0, 2, "a d e").expect("a doc 2");
        let ra = FtsReader::open(Bytes::from(a.finish().expect("finish a")), json).expect("open a");

        // Force the spilled accumulator with a 1-byte threshold: the first
        // prebuilt push spills the column, so the rest exercise the transition
        // + push_prebuilt_spilled (partition + position-blob writes).
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), true).expect("register b");
        b.set_spill_threshold_bytes(1);
        ra.for_each_term_posting(0, |term, doc_id, tf, positions| {
            let term_str = std::str::from_utf8(term).expect("utf8 term");
            b.add_prebuilt_term_posting(0, term_str, doc_id, tf, positions)
                .expect("prebuilt push (spilled)");
            Ok(())
        })
        .expect("feed prebuilt postings");
        b.set_prebuilt_doc_lengths(0, ra.read_doc_lengths(0).expect("doc lengths"));
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

    #[test]
    fn read_doc_lengths_returns_token_counts() {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        b.add_doc(0, 0, "a b a").expect("doc 0"); // 3 tokens
        b.add_doc(0, 1, "b a c d").expect("doc 1"); // 4 tokens
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(b.finish().expect("finish")), json).expect("open");
        assert_eq!(r.read_doc_lengths(0).expect("doc lengths"), vec![3, 4]);
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
        let bad_json = r#"[{"name":"body","tokenizer":"ascii_lower"},{"name":"title","tokenizer":"ascii_lower"}]"#;
        let err = FtsReader::open(blob, bad_json).expect_err("expected error");
        assert!(matches!(
            err,
            FtsError::Read(ReadError::MalformedVersion(_))
        ));
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
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0), "doc 0 should match");
        assert!(ids.contains(&1), "doc 1 should match");
        assert!(!ids.contains(&2), "doc 2 should not match");
    }

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
            if i % 2 == 0 {
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
    async fn token_match_doc_set_matches_bm25_for_same_terms() {
        // token_match(Or) must return exactly the doc set bm25 ranks.
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open FtsReader");
        let mut bm25: Vec<u32> = r
            .search("body", &["rust", "java"], 10, BoolMode::Or)
            .await
            .expect("search")
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        bm25.sort_unstable();
        let boolean = r
            .token_match("body", &["rust", "java"], BoolMode::Or)
            .await
            .expect("boolean")
            .0;
        assert_eq!(bm25, boolean, "boolean Or doc set == bm25 doc set");
    }

    #[tokio::test]
    async fn exhaustive_and_bmm_agree_on_top_k() {
        // Build a larger blob so multi-term OR queries are
        // interesting (some docs have multiple terms, some have one).
        // Both algorithms must return identical top-K (descending
        // score, ascending doc_id tiebreak).
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        // 20 docs sprinkled with mixed term combinations.
        let docs = [
            "alpha",
            "beta",
            "gamma",
            "alpha beta",
            "alpha gamma",
            "beta gamma",
            "alpha beta gamma",
            "delta",
            "epsilon",
            "alpha delta",
            "beta epsilon",
            "gamma delta",
            "alpha beta delta",
            "alpha epsilon gamma",
            "delta epsilon",
            "alpha alpha alpha",
            "beta beta beta",
            "gamma gamma",
            "alpha beta gamma delta epsilon",
            "epsilon",
        ];
        for (i, text) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, text).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        // Three terms with similar UBs — the heuristic should pick
        // exhaustive for this shape, but we cross-check by calling
        // both paths directly via the bench harness.
        let terms: &[&str] = &["alpha", "beta", "gamma"];
        let bmm = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Bmm)
            .await
            .expect("bmm");
        let exh = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Exhaustive)
            .await
            .expect("exhaustive");
        assert_eq!(bmm.len(), exh.len(), "result length mismatch");
        for ((d_bmm, s_bmm), (d_exh, s_exh)) in bmm.iter().zip(exh.iter()) {
            assert_eq!(d_bmm, d_exh, "doc_id mismatch");
            assert!(
                (s_bmm - s_exh).abs() < 1e-4,
                "score mismatch: bmm={s_bmm} exhaustive={s_exh}"
            );
        }
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

    /// Build a corpus that exercises both the df=1 inline-encoded
    /// path and the df ≥ 2 PFOR path side-by-side.
    fn build_mixed_df_blob() -> (Bytes, String) {
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        // `common`     → df = 3 (PFOR form)
        // `rust`       → df = 2 (PFOR form)
        // `uniqzero`  → df = 1 (inline form)
        // `uniqtwo`  → df = 1 (inline form)
        b.add_doc(0, 0, "common rust uniqzero").expect("add doc");
        b.add_doc(0, 1, "common rust").expect("add doc");
        b.add_doc(0, 2, "common uniqtwo").expect("add doc");
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        (Bytes::from(bytes), json.to_string())
    }

    #[test]
    fn df1_inline_form_flag_set_on_fst_value() {
        // Verify the FST values for df=1 terms have bit 0 set
        // (inline form) and df ≥ 2 terms have bit 0 clear (PFOR).
        let (blob, _json) = build_mixed_df_blob();
        // Re-parse the blob enough to reach the FST bytes.
        let header_size = 48usize;
        let fst_off =
            u64::from_le_bytes(blob[24..32].try_into().expect("fst_off slice is 8 bytes")) as usize;
        let postings_off = u64::from_le_bytes(
            blob[32..40]
                .try_into()
                .expect("postings_off slice is 8 bytes"),
        ) as usize;
        // FST bytes occupy [fst_off, postings_off - 4) (last 4 = FST CRC).
        let fst_bytes = &blob[fst_off..postings_off - 4];
        let dict = DictReader::open(fst_bytes).expect("open dict");
        assert_eq!(header_size, 48);

        let val_common = dict.lookup(b"body\x1Fcommon").expect("common in FST");
        let val_rust = dict.lookup(b"body\x1Frust").expect("rust in FST");
        let val_uniq_d0 = dict.lookup(b"body\x1Funiqzero").expect("uniqzero in FST");
        let val_uniq_d2 = dict.lookup(b"body\x1Funiqtwo").expect("uniqtwo in FST");

        assert_eq!(val_common & 1, 0, "df=3 common term must use PFOR form");
        assert_eq!(val_rust & 1, 0, "df=2 rust term must use PFOR form");
        assert_eq!(val_uniq_d0 & 1, 1, "df=1 uniqzero must use inline form");
        assert_eq!(val_uniq_d2 & 1, 1, "df=1 uniqtwo must use inline form");

        // Decode the inline values and check (doc_id, tf) match.
        match FstValue::unpack(val_uniq_d0) {
            FstValue::Inline { doc_id, tf } => {
                assert_eq!(doc_id, 0);
                assert_eq!(tf, 1);
            }
            FstValue::Pfor { .. } => panic!("expected inline form"),
        }
        match FstValue::unpack(val_uniq_d2) {
            FstValue::Inline { doc_id, tf } => {
                assert_eq!(doc_id, 2);
                assert_eq!(tf, 1);
            }
            FstValue::Pfor { .. } => panic!("expected inline form"),
        }
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

    #[test]
    fn df1_inline_path_skips_postings_region_writes() {
        // A blob with only df=1 terms should produce a much smaller
        // postings region than a blob with the same term count but
        // df ≥ 2 — the inline form writes nothing for df=1.
        let tok = Arc::new(AsciiLowerTokenizer);

        let mut b_inline = FtsBuilder::new(tok.clone());
        b_inline
            .register_column("body".into(), false)
            .expect("register column");
        for i in 0..20 {
            b_inline
                .add_doc(0, i, &format!("uniq{i:03}"))
                .expect("add doc");
        }
        let blob_inline = b_inline.finish().expect("finish inline");

        let mut b_pfor = FtsBuilder::new(tok);
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
        // v2 layout: the postings region ends where the positions
        // region begins (header bytes [48..56]).
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
        // CRC32 (4 B). PFOR blob holds 20 terms × (20 B metadata +
        // 16 B skip table × 1 block + ~tens of bytes per PFOR block).
        assert_eq!(
            postings_size_inline, 4,
            "all-df=1 postings region should hold only the trailing CRC32; \
             got {postings_size_inline} bytes"
        );
        assert!(
            postings_size_pfor > 20 * 36,
            "PFOR postings region should be hundreds of bytes; got {postings_size_pfor}"
        );
    }

    // ── ExcludeFilter (negation gate) ─────────────────────────────────
    // `build_blob` plants: "rust" in docs 0 and 1, "java" in doc 2.

    /// Build an `ExcludeFilter` over `terms` from the planted blob.
    async fn exclude_filter_for(reader: &FtsReader, terms: &[&str]) -> ExcludeFilter {
        let column_id = reader.resolve_column_id("body").expect("column exists");
        let cursors = reader
            .build_term_cursors(column_id, terms, None)
            .await
            .expect("build cursors");
        ExcludeFilter::new(cursors)
    }

    #[tokio::test]
    async fn exclude_filter_rejects_docs_in_negated_list() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let mut f = exclude_filter_for(&r, &["rust"]).await;
        // "rust" is in docs 0 and 1 → excluded; doc 2 survives.
        assert!(!f.admits(0));
        assert!(!f.admits(1));
        assert!(f.admits(2));
    }

    #[tokio::test]
    async fn exclude_filter_missing_term_excludes_nothing() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // A negated term absent from the dictionary yields no cursor, so
        // the filter admits every doc.
        let mut f = exclude_filter_for(&r, &["nonexistent"]).await;
        assert!(f.admits(0));
        assert!(f.admits(1));
        assert!(f.admits(2));
    }

    #[tokio::test]
    async fn exclude_filter_multiple_negated_terms() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        // Negating "rust" (docs 0,1) and "java" (doc 2) excludes all
        // three — a doc is dropped if it matches ANY negated term.
        let mut f = exclude_filter_for(&r, &["rust", "java"]).await;
        assert!(!f.admits(0));
        assert!(!f.admits(1));
        assert!(!f.admits(2));
    }

    #[tokio::test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "non-monotonic")]
    async fn exclude_filter_panics_on_non_monotonic_feed() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let mut f = exclude_filter_for(&r, &["rust"]).await;
        // Feed a descending doc-id: `skip_to` can't seek backwards, so
        // the debug assertion catches the contract violation.
        let _ = f.admits(1);
        let _ = f.admits(0);
    }

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn open_with_verify_crc_off_succeeds() {
        // The trusted-storage fast path skips the four CRC scans but must
        // still produce a fully usable reader.
        let (blob, json) = build_blob();
        let r = FtsReader::open_with(blob, &json, OpenOptions { verify_crc: false })
            .expect("open with crc off");
        assert_eq!(r.n_docs(), 3);
        assert_eq!(r.fts_columns().collect::<Vec<_>>(), vec!["body"]);
    }

    #[test]
    fn open_with_object_store_options_matches_crc_off() {
        // `for_object_store` is the named constructor for the crc-off
        // OpenOptions the lazy/object-store path uses.
        let opts = OpenOptions::for_object_store();
        assert!(!opts.verify_crc);
        let (blob, json) = build_blob();
        FtsReader::open_with(blob, &json, opts).expect("open object-store options");
    }

    #[test]
    fn default_open_options_verifies_crc() {
        assert!(OpenOptions::default().verify_crc);
    }

    #[test]
    fn default_tokenizer_helper_is_ascii_lower() {
        assert_eq!(default_tokenizer(), "ascii_lower");
    }

    #[test]
    fn fts_column_config_missing_tokenizer_defaults() {
        // A column JSON without the optional `tokenizer` field decodes to
        // the ascii_lower default (round-trips an old file written before
        // the field existed).
        let (blob, _) = build_blob();
        let json = r#"[{"name":"body"}]"#;
        let r = FtsReader::open(blob, json).expect("open with terse json");
        let cfg = r.fts_columns_config().next().expect("one column");
        assert_eq!(cfg.name, "body");
    }

    #[test]
    fn fts_columns_config_exposes_per_column_metadata() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        let cols: Vec<&ColumnMeta> = r.fts_columns_config().collect();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].name, "body");
        // Three non-empty docs ⇒ a positive average doc length and a
        // populated per-doc normalization table.
        assert!(cols[0].avgdl > 0.0);
        assert_eq!(cols[0].dl_norm_k1.len(), 3);
    }

    #[test]
    fn norm_table_footprint_is_one_byte_per_doc() {
        // Memory guard: the resident length-norm table must stay at one
        // byte per doc (plus the fixed 256-entry decode LUT), not the
        // 4-byte-per-doc `f32` table it replaced. Build enough
        // varied-length docs that the per-doc term dominates the LUT.
        const N: u32 = 5_000;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false)
            .expect("register column");
        for d in 0..N {
            // Lengths cycle 1..=40 tokens so norms span many buckets and
            // the table isn't a degenerate single value.
            let words = (d % 40) + 1;
            let text: String = (0..words).map(|w| format!("t{}x{w} ", d % 97)).collect();
            b.add_doc(0, d, text.trim()).expect("add doc");
        }
        let bytes = b.finish().expect("finish");
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(Bytes::from(bytes), json).expect("open");
        let nt = &r.columns[0].dl_norm_k1;

        let per_doc = nt.bytes.capacity(); // 1 byte/doc
        let lut = std::mem::size_of_val(&*nt.lut); // 256 * 4 = 1 KiB
        let m2_bytes = per_doc + lut;
        let f32_baseline = N as usize * std::mem::size_of::<f32>();

        assert_eq!(nt.bytes.len(), N as usize, "one bucket byte per doc");
        assert_eq!(nt.lut.len(), 256, "fixed 256-entry decode table");
        // The whole point: strictly smaller than the old f32 table, and
        // asymptotically 4× smaller (per-doc term is 1 byte vs 4).
        assert!(
            m2_bytes < f32_baseline,
            "norm table {m2_bytes} B not smaller than f32 baseline {f32_baseline} B"
        );
        assert_eq!(
            per_doc * 4,
            f32_baseline,
            "per-doc term is exactly 4× smaller"
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
        // FST iteration is lex-ordered.
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

    // ---- phrase atoms ----

    /// Corpus with controlled adjacency for "new york": docs 0, 2
    /// match (doc 4 twice); docs 1, 3 contain both words but never
    /// adjacent in order.
    fn build_phrase_blob() -> (Bytes, &'static str) {
        use crate::superfile::fts::builder::FtsBuilder;
        let mut b = FtsBuilder::new(crate::test_helpers::default_tokenizer());
        b.register_column("title".into(), true).expect("register");
        let docs = [
            "new york city",
            "york new haven",
            "the new york times",
            "new haven york",
            "new york new york",
        ];
        for (i, d) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, d).expect("add doc");
        }
        (
            Bytes::from(b.finish().expect("finish")),
            r#"[{"name":"title","tokenizer":"ascii_lower","positions":true}]"#,
        )
    }

    fn phrase(terms: &[&str]) -> Vec<Vec<String>> {
        vec![terms.iter().map(|t| t.to_string()).collect()]
    }

    #[tokio::test]
    async fn phrase_matches_adjacent_in_order_only() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let phrases = phrase(&["new", "york"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("phrase search");
        let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        let mut sorted = ids.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 2, 4], "adjacency in order only");
        // Doc 4 has the phrase twice — highest tf, and with uniform
        // doc lengths in play its score must strictly exceed doc 0's
        // (same length, tf 1... doc 0 len 3, doc 4 len 4; tf=2 wins).
        assert_eq!(hits[0].0, 4, "double occurrence ranks first");
    }

    #[tokio::test]
    async fn phrase_composes_with_clauses() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let ny = phrase(&["new", "york"]);

        // Must-phrase + must-term: "the" only in doc 2.
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    musts: &["the"],
                    must_phrases: &ny,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("must phrase + term");
        assert_eq!(
            hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
            vec![2],
            "+\"new york\" +the"
        );

        // Negated phrase: haven-docs minus the phrase docs.
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    shoulds: &["haven"],
                    negative_phrases: &ny,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("negated phrase");
        let mut ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![1, 3], "haven docs don't contain the phrase");
    }

    #[tokio::test]
    async fn phrase_with_absent_member_matches_nothing() {
        let (blob, json) = build_phrase_blob();
        let r = FtsReader::open(blob, json).expect("open");
        let ghost = phrase(&["new", "zealand"]);
        let hits = r
            .search_excluding(
                "title",
                ClauseLists {
                    must_phrases: &ghost,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect("ghost phrase");
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn phrase_on_positionless_column_is_typed_error() {
        use crate::superfile::fts::builder::FtsBuilder;
        let mut b = FtsBuilder::new(crate::test_helpers::default_tokenizer());
        b.register_column("title".into(), false).expect("register");
        b.add_doc(0, 0, "new york").expect("add doc");
        let blob = Bytes::from(b.finish().expect("finish"));
        let r =
            FtsReader::open(blob, r#"[{"name":"title","tokenizer":"ascii_lower"}]"#).expect("open");
        let phrases = phrase(&["new", "york"]);
        let err = r
            .search_excluding(
                "title",
                ClauseLists {
                    should_phrases: &phrases,
                    ..ClauseLists::default()
                },
                10,
                f32::NEG_INFINITY,
            )
            .await
            .expect_err("must be a typed error");
        assert!(matches!(err, FtsError::PositionsUnavailable { .. }));
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

        // Uniform 4-term OR routes to the windowed union; the
        // rare+common mix keeps a dominant term UB and stays on BMM.
        // Assert the routing rather than assume it — a corpus tweak that
        // silently stopped exercising one branch would otherwise turn
        // this into a test of the other branch twice.
        let shapes: [&[&str]; 2] = [
            &["alpha", "beta", "gamma", "delta"],
            &["rareterm", "alpha", "beta"],
        ];
        let column_id = r.resolve_column_id("body").expect("column");
        let uniform_cursors = r
            .build_term_cursors(column_id, shapes[0], None)
            .await
            .expect("cursors");
        assert!(
            prefer_windowed_union(&uniform_cursors),
            "uniform shape must route to the windowed ranged branch"
        );
        let dominant_cursors = r
            .build_term_cursors(column_id, shapes[1], None)
            .await
            .expect("cursors");
        assert!(
            !prefer_windowed_union(&dominant_cursors),
            "dominant-UB shape must route to the BMM ranged branch"
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

    #[tokio::test]
    async fn search_with_algo_wand_bmw_agrees_with_bmm() {
        // The historical WAND+BMW baseline must agree with the production
        // BMM path on the planted corpus.
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        let docs = [
            "alpha beta",
            "alpha",
            "beta gamma",
            "alpha beta gamma",
            "gamma",
            "alpha gamma",
            "beta",
            "alpha beta gamma",
        ];
        for (i, t) in docs.iter().enumerate() {
            b.add_doc(0, i as u32, t).expect("add");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let terms: &[&str] = &["alpha", "beta", "gamma"];
        let bmm = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::Bmm)
            .await
            .expect("bmm");
        let wand = r
            .search_with_algo_for_bench("body", terms, 5, OrAlgo::WandBmw)
            .await
            .expect("wand");
        assert_eq!(bmm.len(), wand.len());
        for ((db, sb), (dw, sw)) in bmm.iter().zip(wand.iter()) {
            assert_eq!(db, dw, "doc_id mismatch");
            assert!((sb - sw).abs() < 1e-4, "score mismatch {sb} vs {sw}");
        }
    }

    #[tokio::test]
    async fn wand_bmw_exercises_block_skips_on_multi_block_lists() {
        // A corpus large enough that the common terms span several
        // 128-doc posting blocks, with five query terms of differing
        // document frequency and a handful of docs carrying all five.
        // Running WAND+BMW at a small k forces the pivot to move, the
        // block-upper-bound skip to fire, lagging cursors to re-align,
        // and the 4-wide SIMD scoring pack to be used on the
        // all-terms docs — then cross-checks the result against BMM.

        /// Total planted docs; well over several `BLOCK_LEN` (128) so
        /// the dense-term posting lists occupy multiple blocks.
        const N_DOCS: u32 = 400;
        /// Requested top-K — small, so the heap fills early and the
        /// score threshold starts pruning blocks.
        const K: usize = 5;

        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::new();
            // `alpha` in ~every doc, `beta` in ~half, `gamma` every
            // 5th, `delta` every 13th, `epsilon` every 29th — a
            // descending-df mix that makes the WAND pivot non-trivial.
            text.push_str("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 5 == 0 {
                text.push_str("gamma ");
            }
            if i % 13 == 0 {
                text.push_str("delta ");
            }
            if i % 29 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");

        let terms: &[&str] = &["alpha", "beta", "gamma", "delta", "epsilon"];
        let wand = r
            .search_with_algo_for_bench("body", terms, K, OrAlgo::WandBmw)
            .await
            .expect("wand");
        let bmm = r
            .search_with_algo_for_bench("body", terms, K, OrAlgo::Bmm)
            .await
            .expect("bmm");
        assert_eq!(wand.len(), bmm.len(), "result length mismatch");
        assert_eq!(wand.len(), K, "expected a full top-K");
        for ((dw, sw), (db, sb)) in wand.iter().zip(bmm.iter()) {
            assert_eq!(dw, db, "doc_id mismatch wand={dw} bmm={db}");
            assert!((sw - sb).abs() < 1e-4, "score mismatch {sw} vs {sb}");
        }
    }

    #[tokio::test]
    async fn windowed_union_agrees_with_bmm() {
        // The windowed union scorer must return the identical top-k as
        // the production MaxScore+BMM path — across term counts, k values,
        // and the uniform-UB (common-term) shape it targets. N_DOCS spans
        // multiple windows (and many BLOCK_LEN=128 posting blocks), so the
        // walk exercises the multi-window path: base advancing to the next
        // window, empty-window skipping, and cross-window monotonicity —
        // not just a single window. Tied to OR_WINDOW so it keeps crossing
        // the boundary if the window size changes.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha zeta eta theta "); // ~every doc
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            if i % 7 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");
        let uniform_terms: &[&str] = &["zeta", "eta", "theta"];
        let uniform_cursors = r
            .build_term_cursors(col, uniform_terms, None)
            .await
            .expect("uniform cursors");
        assert!(
            prefer_windowed_union(&uniform_cursors),
            "production router should select windowed union for equal upper bounds"
        );

        let shapes: &[&[&str]] = &[
            &["alpha", "beta"],
            &["alpha", "beta", "gamma"],
            &["beta", "gamma", "delta"], // no single dominator
            &["alpha", "beta", "gamma", "delta", "epsilon"],
            uniform_terms,
        ];
        for terms in shapes {
            for k in [1usize, 5, 50, 1000] {
                let bmm = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Bmm)
                    .await
                    .expect("bmm");
                let win = r
                    .search_with_algo_for_bench("body", terms, k, OrAlgo::Windowed)
                    .await
                    .expect("windowed");
                assert_eq!(bmm.len(), win.len(), "len mismatch {terms:?} k={k}");
                for ((db, sb), (dw, sw)) in bmm.iter().zip(win.iter()) {
                    assert_eq!(db, dw, "doc_id mismatch {terms:?} k={k}: bmm={db} win={dw}");
                    assert!(
                        (sb - sw).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sb} vs {sw}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn wand_bmw_2term_no_floor_agrees_with_bmm() {
        // The small-k 2-term production path (`run_wand_bmw`) must return
        // the identical top-k as MaxScore+BMM on the same inputs, across k.
        // It is only reached floor-free (the dispatcher routes to MaxScore
        // when a cross-segment floor is live), so both sides run unfloored
        // (`NEG_INFINITY`). Multi-window corpus so WAND exercises block
        // skips; `gamma` rarer than `beta` rarer than `alpha`.
        const N_DOCS: u32 = OR_WINDOW * 2 + 500;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        let shapes: &[&[&str]] = &[&["alpha", "beta"], &["beta", "gamma"], &["alpha", "gamma"]];
        for terms in shapes {
            for k in [1usize, 5, 50, 128] {
                let cw = r
                    .build_term_cursors(col, terms, None)
                    .await
                    .expect("cursors");
                let cb = r
                    .build_term_cursors(col, terms, None)
                    .await
                    .expect("cursors");
                let wand = r.run_wand_bmw(col, cw, k).expect("wand");
                let bmm = r
                    .run_max_score_bmm(col, cb, k, None, f32::NEG_INFINITY)
                    .expect("bmm");
                assert_eq!(wand.len(), bmm.len(), "len mismatch {terms:?} k={k}");
                for ((dw, sw), (db, sb)) in wand.iter().zip(bmm.iter()) {
                    assert_eq!(dw, db, "doc mismatch {terms:?} k={k}: {dw} vs {db}");
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score mismatch {terms:?} k={k}: {sw} vs {sb}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn two_term_rare_anchor_gates_on_df_ratio() {
        // `df` is read onto the cursor, and the 2-term WAND router fires
        // only when one posting list is >= WAND_BMW_2TERM_DF_RATIO× shorter
        // than the other (a rare anchor), not when both terms are common.
        const N_DOCS: u32 = 4000;
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("common "); // every doc
            if i % 2 == 0 {
                text.push_str("frequent "); // half the docs
            }
            if i % 200 == 0 {
                text.push_str("rare "); // ~1/200 of docs
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        // common (df≈N) + rare (df≈N/200): ratio 200 ≥ 16 → anchor.
        let anchored = r
            .build_term_cursors(col, &["common", "rare"], None)
            .await
            .expect("cursors");
        assert!(
            two_term_has_rare_anchor(&anchored),
            "rare+common should have a rare anchor"
        );
        // common (df≈N) + frequent (df≈N/2): ratio 2 < 16 → no anchor.
        let uniform = r
            .build_term_cursors(col, &["common", "frequent"], None)
            .await
            .expect("cursors");
        assert!(
            !two_term_has_rare_anchor(&uniform),
            "two common terms should not anchor"
        );
    }

    #[test]
    fn deep_k_dominant_union_reroutes_only_when_list_is_long() {
        // The deep-k reroute to the windowed scorer fires only when
        // pruning is dead (k reaches the rarer terms' combined df) AND the
        // dominant list is long enough to amortize the window setup.
        const LONG: u64 = 3_000_000; // dominant common term
        const RARE: u64 = 500; // rare second term
        let total = LONG + RARE;

        // Deep k (>= rest_df) over a long dominant list: reroute.
        assert!(
            or_reroute_by_df(LONG, total, 2, 1000),
            "deep k over a long dominant list should reroute to windowed"
        );
        // Shallow k (< rest_df): the rare term still fills the heap, pruning
        // is alive, stay on MaxScore.
        assert!(
            !or_reroute_by_df(LONG, total, 2, 100),
            "k below the rare term's df keeps pruning alive → MaxScore"
        );
        // Exact boundary k == rest_df: the heap needs one doc beyond the
        // rare term's list, so pruning is already dead → reroute (the test
        // is `>=`, so the boundary counts).
        assert!(
            or_reroute_by_df(LONG, total, 2, RARE as usize),
            "k exactly at rest_df should reroute"
        );
        // One below the boundary (k == rest_df - 1): rare term still fills
        // the heap, stay on MaxScore.
        assert!(
            !or_reroute_by_df(LONG, total, 2, RARE as usize - 1),
            "k just below rest_df keeps pruning alive → MaxScore"
        );
        // Long list but only one term: not an OR.
        assert!(
            !or_reroute_by_df(LONG, LONG, 1, 1000),
            "single term is not a union"
        );
        // Small union (dominant list below the floor): too little work to
        // amortize the window; stay on MaxScore even at deep k. This is the
        // case that regressed before the floor was added.
        let small = OR_WINDOWED_MIN_DOMINANT_DF - 1;
        assert!(
            !or_reroute_by_df(small, small + RARE, 2, 1_000_000),
            "a union below the dominant-df floor must not reroute"
        );
        // Exactly at the floor with deep k: reroute.
        assert!(
            or_reroute_by_df(
                OR_WINDOWED_MIN_DOMINANT_DF,
                OR_WINDOWED_MIN_DOMINANT_DF + RARE,
                2,
                1000
            ),
            "at the dominant-df floor a deep-k union reroutes"
        );
    }

    #[tokio::test]
    async fn windowed_union_negation_agrees_with_bmm() {
        // The windowed scorer applies the ExcludeFilter (negation) at
        // drain. Drive a negated query straight through run_windowed_union
        // and check it matches MaxScore+BMM with the same exclusion — BMM's
        // negation is the oracle-validated reference, so equality proves
        // the windowed filter arm. (Calls the scorers directly so the
        // windowed arm is exercised regardless of the production dispatch.)
        const N_DOCS: u32 = OR_WINDOW + 1000; // spans more than one window
        let tok = Arc::new(AsciiLowerTokenizer);
        let mut b = FtsBuilder::new(tok);
        b.register_column("body".into(), false).expect("register");
        for i in 0..N_DOCS {
            let mut text = String::from("alpha ");
            if i % 2 == 0 {
                text.push_str("beta ");
            }
            if i % 3 == 0 {
                text.push_str("gamma ");
            }
            if i % 5 == 0 {
                text.push_str("delta ");
            }
            if i % 7 == 0 {
                text.push_str("epsilon ");
            }
            b.add_doc(0, i, text.trim()).expect("add doc");
        }
        let blob = Bytes::from(b.finish().expect("finish"));
        let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
        let r = FtsReader::open(blob, json).expect("open");
        let col = r.resolve_column_id("body").expect("col");

        // (positive terms, negated terms)
        let cases: &[(&[&str], &[&str])] = &[
            (&["alpha", "beta", "gamma"], &["delta"]),
            (&["beta", "gamma", "delta"], &["epsilon"]),
            (&["alpha", "beta", "gamma", "delta"], &["epsilon", "gamma"]),
        ];
        for (pos, neg) in cases {
            for k in [1usize, 5, 50] {
                let mut wf = ExcludeFilter::new(
                    r.build_term_cursors(col, neg, None)
                        .await
                        .expect("neg cursors"),
                );
                let win = r
                    .run_windowed_union(
                        col,
                        r.build_term_cursors(col, pos, None)
                            .await
                            .expect("pos cursors"),
                        k,
                        Some(&mut wf),
                        f32::NEG_INFINITY,
                        0,
                        u32::MAX,
                    )
                    .expect("windowed");
                let mut bf = ExcludeFilter::new(
                    r.build_term_cursors(col, neg, None)
                        .await
                        .expect("neg cursors"),
                );
                let bmm = r
                    .run_max_score_bmm(
                        col,
                        r.build_term_cursors(col, pos, None)
                            .await
                            .expect("pos cursors"),
                        k,
                        Some(&mut bf),
                        f32::NEG_INFINITY,
                    )
                    .expect("bmm");
                assert_eq!(win.len(), bmm.len(), "len {pos:?} -{neg:?} k={k}");
                for ((dw, sw), (db, sb)) in win.iter().zip(bmm.iter()) {
                    assert_eq!(
                        dw, db,
                        "doc mismatch {pos:?} -{neg:?} k={k}: win={dw} bmm={db}"
                    );
                    assert!(
                        (sw - sb).abs() < 1e-4,
                        "score mismatch {pos:?} -{neg:?} k={k}: {sw} vs {sb}"
                    );
                }
            }
        }

        // Sanity: the filter is actually active — at a high k the negated
        // query must return strictly fewer docs than the positive-only one
        // (the negated term excludes a non-empty set).
        let pos: &[&str] = &["alpha", "beta", "gamma"];
        let neg: &[&str] = &["delta"];
        let unfiltered = r
            .run_windowed_union(
                col,
                r.build_term_cursors(col, pos, None).await.expect("pos"),
                N_DOCS as usize,
                None,
                f32::NEG_INFINITY,
                0,
                u32::MAX,
            )
            .expect("unfiltered");
        let mut f = ExcludeFilter::new(r.build_term_cursors(col, neg, None).await.expect("neg"));
        let filtered = r
            .run_windowed_union(
                col,
                r.build_term_cursors(col, pos, None).await.expect("pos"),
                N_DOCS as usize,
                Some(&mut f),
                f32::NEG_INFINITY,
                0,
                u32::MAX,
            )
            .expect("filtered");
        assert!(
            filtered.len() < unfiltered.len(),
            "negation should drop docs: filtered={} unfiltered={}",
            filtered.len(),
            unfiltered.len()
        );
    }

    #[tokio::test]
    async fn search_with_algo_empty_and_zero_k_short_circuit() {
        let (blob, json) = build_blob();
        let r = FtsReader::open(blob, &json).expect("open");
        assert!(
            r.search_with_algo_for_bench("body", &[], 5, OrAlgo::Bmm)
                .await
                .expect("empty")
                .is_empty()
        );
        assert!(
            r.search_with_algo_for_bench("body", &["rust"], 0, OrAlgo::Exhaustive)
                .await
                .expect("zero k")
                .is_empty()
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
    fn top_k_keeps_highest_scores_with_doc_id_tiebreak() {
        let mut scores: HashMap<u32, f32> = HashMap::new();
        scores.insert(0, 1.0);
        scores.insert(1, 3.0);
        scores.insert(2, 2.0);
        scores.insert(3, 3.0); // tie with doc 1 on score 3.0
        let out = top_k(scores, 2);
        // Descending score; ties broken by ascending doc_id ⇒ doc 1 before 3.
        assert_eq!(out, vec![(1, 3.0), (3, 3.0)]);
    }

    #[test]
    fn top_k_smaller_than_k_returns_all_sorted() {
        let mut scores: HashMap<u32, f32> = HashMap::new();
        scores.insert(5, 2.0);
        scores.insert(9, 5.0);
        let out = top_k(scores, 10);
        assert_eq!(out, vec![(9, 5.0), (5, 2.0)]);
    }

    #[test]
    fn drain_top_k_desc_orders_descending_with_tiebreak() {
        let mut heap: BinaryHeap<TopKEntry> = BinaryHeap::new();
        heap.push(TopKEntry(1.0, 4));
        heap.push(TopKEntry(2.0, 1));
        heap.push(TopKEntry(2.0, 0)); // tie with doc 1
        let out = drain_top_k_desc(heap);
        assert_eq!(out, vec![(0, 2.0), (1, 2.0), (4, 1.0)]);
    }

    #[tokio::test]
    async fn open_lazy_round_trips_a_search() {
        // Wrap the eager blob in a whole-blob lazy source so the lazy
        // open path (header + FST + doc-length tail prefetch) runs and
        // serves a real query.
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
        let ids: HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
        assert!(ids.contains(&0) && ids.contains(&1));
    }
}
