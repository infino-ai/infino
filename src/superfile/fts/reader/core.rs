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
//! `search` calls share the underlying `Bytes`. The DictReader is
//! constructed per call (cheap; the FST validates its header in O(1) and
//! then it's a borrowed view).

use std::{
    collections::{BinaryHeap, HashMap},
    ops::Range,
    sync::Arc,
};

use bytes::Bytes;

use super::{
    cursor::{TermCursor, TermMeta},
    filter::ExcludeFilter,
    metadata::{ColumnMeta, FtsColumnConfig, NormTable, OpenOptions},
    phrase::{AnyCursor, PhraseCursor},
    sink::{TopKEntry, drain_top_k_desc},
    work::{term_cursor_bytes, term_cursor_ranges},
};
use crate::superfile::{
    ReadError,
    error::FtsError,
    format::{
        self, FST_SEPARATOR,
        checksum::crc32c,
        fts::{
            HEADER_SIZE_V1_LEGACY as FTS_HEADER_SIZE, MAGIC_BYTES, U32_BYTES, U64_BYTES, hdr,
            term_meta,
        },
    },
    fts::{
        builder::{DOC_LENGTHS_ENTRY_SIZE, TERM_META_SIZE},
        dict::{DictReader, make_key},
        fst_value::FstValue,
        positions::decode_run,
        posting::{self, BLOCK_LEN, ENCODING_BITSET, decode_block_doc_ids},
        tokenize::{Tokenizer, tokenizer_for_name},
    },
    lazy_source::{LazyByteSource, PrefetchedSource, RangeCoalescePlan, Source},
};

/// Largest gap worth overfetching when adjacent term postings share a request.
const TERM_RANGE_COALESCE_MAX_GAP: usize = 64 * 1024;
/// Maximum total gap bytes tolerated in one coalesced postings request.
const TERM_RANGE_COALESCE_MAX_OVERFETCH: usize = 512 * 1024;

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
pub(super) const OR_WINDOW: u32 = 4096;
/// Number of 64-bit words in the window presence bitset.
pub(super) const OR_WINDOW_WORDS: usize = (OR_WINDOW as usize).div_ceil(64);

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

/// Score upper-bound spread below which a multi-term OR counts as
/// "common-heavy" (no single term dominates): `max_ub <=
/// OR_WINDOW_DOMINANCE_MULT * avg_ub`. Uniform terms sit near the average;
/// a dominant (rarer) term sits well above it. See [`no_dominant_term_ub`].
pub(super) const OR_WINDOW_DOMINANCE_MULT: f32 = 1.5;

/// `k` cutoff for a common-heavy OR: at or below it MaxScore prunes (its
/// heap fills and the threshold rises above the common terms' block maxima),
/// so keep it there; above it pruning is dead and the SIMD windowed scan
/// wins. Bench-tuned to the x86 target (the crossover is lower on wide-SIMD
/// x86 than on ARM).
pub(super) const OR_WINDOWED_UNIFORM_MAX_PRUNING_K: usize = 32;

/// True when no single term dominates the score upper bound
/// (`max_ub <= OR_WINDOW_DOMINANCE_MULT * avg_ub`). Such a "common-heavy" OR
/// only prunes on MaxScore at small `k`; a dominant term skips hard at any `k`.
pub(super) fn no_dominant_term_ub(cursors: &[TermCursor]) -> bool {
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

/// Route a multi-term OR to the non-pruning windowed scan instead of MaxScore,
/// true only where MaxScore's pruning is dead at this `k`: a dominant long list
/// too deep to fill the heap without its tail ([`or_topk_pruning_ineffective`]),
/// or a common-heavy shape past [`OR_WINDOWED_UNIFORM_MAX_PRUNING_K`]. Otherwise
/// MaxScore. Shared by the single-shot and ranged OR entries so a query runs the
/// same kernel whether or not the fan-out sliced it.
pub(super) fn route_or_to_windowed(cursors: &[TermCursor], k: usize) -> bool {
    or_topk_pruning_ineffective(cursors, k)
        || (k > OR_WINDOWED_UNIFORM_MAX_PRUNING_K
            && cursors.len() >= OR_WINDOW_MIN_TERMS
            && no_dominant_term_ub(cursors))
}

/// Minimum dominant-term df for the deep-`k` reroute to the windowed
/// scorer to pay off. Below this the union is small enough that
/// MaxScore's scalar full scan beats the windowed scorer's fixed
/// per-window setup, so short unions stay on MaxScore. Calibrated on the
/// warm FTS bench: the win concentrates on dominant lists in the millions
/// (~corpus-scale), the mid range is a wash, and lists shorter than this
/// regressed. See [`or_topk_pruning_ineffective`].
pub(super) const OR_WINDOWED_MIN_DOMINANT_DF: u64 = 100_000;

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
pub(super) fn or_topk_pruning_ineffective(cursors: &[TermCursor], k: usize) -> bool {
    let max_df = cursors.iter().map(|c| c.df).max().unwrap_or(0);
    let total_df: u64 = cursors.iter().map(|c| c.df).sum();
    or_reroute_by_df(max_df, total_df, cursors.len(), k)
}

/// Pure df-math behind [`or_topk_pruning_ineffective`], split out so the
/// routing decision can be unit-tested at df values (hundreds of
/// thousands) that a fast in-memory corpus can't reach.
pub(super) fn or_reroute_by_df(max_df: u64, total_df: u64, n_terms: usize, k: usize) -> bool {
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
#[derive(Debug)]
pub struct FtsReader {
    pub(super) source: Source,
    pub(super) n_docs: u32,
    pub(super) n_terms_total: u32,
    pub(super) fst_range: Range<usize>,
    pub(super) postings_range: Range<usize>,
    /// Byte range of the positions region (CRC stripped) — `Some`
    /// iff the blob is v2. Phrase queries fetch per-term run ranges
    /// out of it via [`Self::fetch_term_positions`].
    pub(super) positions_range: Option<Range<usize>>,
    /// True iff the blob is `VERSION_V3` — its positional terms carry a
    /// position run-offset sub-index between skip table and blocks, which
    /// the phrase decode uses to reach a pair's runs by skipping
    /// `< POSITION_SUBINDEX_STRIDE` runs. `V1`/`V2` blobs lack it and take
    /// the block-start skip-walk fallback.
    pub(super) has_position_subindex: bool,
    /// True iff the blob is `VERSION_V4` — some posting blocks may be
    /// bitset-encoded, so the unranked count kernels prefer membership
    /// bit-tests (no decode) over decoding a common term's blocks.
    pub(super) has_bitset_blocks: bool,
    pub(super) columns: Vec<ColumnMeta>,
    pub(super) column_id_by_name: HashMap<String, u32>,
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
        if version != format::fts::VERSION_V1_LEGACY
            && version != format::fts::VERSION_V2
            && version != format::fts::VERSION_V3
            && version != format::fts::VERSION_V4
        {
            return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                "fts section version {version}"
            ))));
        }
        // The FST directory starts right after whichever header
        // applies; a v2/v3/v4 header's extension bytes are already in the
        // fetched span (and in the overlay below), so
        // `open_with_source` re-reads them without another GET. (v3/v4 share
        // v2's header size.)
        let header_size = match version {
            v if v == format::fts::VERSION_V2
                || v == format::fts::VERSION_V3
                || v == format::fts::VERSION_V4 =>
            {
                format::fts::HEADER_SIZE_V2
            }
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
        // v3/v4 share v2's header and region layout; v3+ additionally
        // carries a per-term position sub-index (handled in the phrase
        // decode), and v4 may store dense blocks in the bitset encoding
        // (self-describing per block, handled in the codec).
        let positional_blob = match version {
            v if v == format::fts::VERSION_V1_LEGACY => false,
            v if v == format::fts::VERSION_V2 => true,
            v if v == format::fts::VERSION_V3 => true,
            v if v == format::fts::VERSION_V4 => true,
            _ => {
                return Err(FtsError::Read(ReadError::UnsupportedVersion(format!(
                    "fts section version {version}"
                ))));
            }
        };
        let has_position_subindex =
            version == format::fts::VERSION_V3 || version == format::fts::VERSION_V4;
        let has_bitset_blocks = version == format::fts::VERSION_V4;
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
            has_position_subindex,
            has_bitset_blocks,
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
    pub(super) async fn dict_bytes_async(&self) -> Result<Bytes, FtsError> {
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
    pub(super) async fn fetch_term_postings(
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
    pub(super) async fn build_atom_cursors(
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
                .build_term_cursors(column_id, &[term], global_idf, false)
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
                .build_term_cursors(column_id, &member_refs, global_idf, false)
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
                        // This is the phrase member's own term_meta —
                        // the one `decode_current_positions` uses — so it
                        // carries the sub-index when the blob has one.
                        let term_meta = TermMeta::parse(
                            cursor.bytes.as_ref(),
                            0,
                            true,
                            self.has_position_subindex,
                        )?;
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
                        let meta = TermMeta::parse(term_bytes.as_ref(), 0, true, false)?;
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

                    let mut cursor =
                        TermCursor::new(term_bytes, n_docs, positional, None, false, false)?;
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

/// Merge a `doc_id -> score` map into top-k by descending score, ties
/// broken by ascending doc_id. Used by `search_multi`'s cross-column
/// combiner, where the per-column scores have already been weighted
/// and summed into `scores`.
pub(super) fn top_k(scores: HashMap<u32, f32>, k: usize) -> Vec<(u32, f32)> {
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
pub(super) fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[inline]
pub(super) fn read_u64_le(b: &[u8]) -> u64 {
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
pub(super) fn or_walk_unranked(mut cursors: Vec<TermCursor>, mut emit: impl FnMut(u32)) {
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
pub(super) fn or_merge_unranked(cursors: Vec<TermCursor>) -> Vec<u32> {
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
pub(super) fn or_count_unranked(mut cursors: Vec<TermCursor>) -> u64 {
    // Fast path: when one term's list dwarfs the rest, count it as
    // `df(dominant) + |others \ dominant|` instead of walking the
    // dominant term's whole posting list. See `or_count_anchored`.
    if let Some(anchor) = dominant_anchor_index(&cursors) {
        return or_count_anchored(cursors, anchor);
    }
    // Dense union (no dominant term, but the terms together cover a large
    // fraction of the corpus — e.g. a 3-4 common-word title): OR all doc
    // ids into one doc-space bitset and popcount. Avoids the per-window
    // bookkeeping the walk below pays per doc.
    let total_df: u64 = cursors.iter().map(|c| c.df).sum();
    let max_doc = cursors
        .iter()
        .filter_map(|c| c.blocks.last())
        .map(|b| b.last_doc_id)
        .max()
        .unwrap_or(0);
    if total_df.saturating_mul(OR_COUNT_BITSET_DENSITY_DIVISOR) >= u64::from(max_doc) {
        return or_count_bitset(cursors, max_doc);
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
    let mut union = vec![0u64; words];
    let mut scratch = [0u32; BLOCK_LEN];
    for c in &cursors {
        or_cursor_into_bitset(&mut union, c, &mut scratch);
    }
    union.iter().map(|w| w.count_ones() as u64).sum()
}

/// OR one cursor's doc presence into `dest` (a doc-space bitset spanning at
/// least the cursor's largest doc id), reading blocks at the byte level: a
/// **BITSET** block is word-copied at its aligned base word — no expansion to
/// doc ids; a **PACKED** block is decoded doc-ids-only (no tf) and scattered; an
/// inline (df=1) cursor scatters its single pre-decoded doc. `scratch` is a
/// reused `BLOCK_LEN` decode buffer. Shared by the union count
/// ([`or_count_bitset`]) and the intersection count (`count_and_intersect_bitset`)
/// so a common term's dense blocks are merged at memory bandwidth either way.
pub(super) fn or_cursor_into_bitset(
    dest: &mut [u64],
    c: &TermCursor,
    scratch: &mut [u32; BLOCK_LEN],
) {
    // Inline (df=1) cursors carry their single doc pre-decoded and have no
    // postings bytes to slice.
    if c.bytes.is_empty() {
        for &d in &c.block_doc_ids[..c.block_n] {
            dest[(d >> 6) as usize] |= 1u64 << (d & 63);
        }
        return;
    }
    for block in c.blocks.iter() {
        let bytes = c.bytes.slice(block.block_byte_offset..block.block_byte_end);
        let bytes = bytes.as_ref();
        if bytes[posting::ENCODING_OFF] == ENCODING_BITSET {
            // Word-OR the presence bitset in at its aligned base word.
            // Tfs trail; the bitset is everything between them.
            let base_word = read_u32_le(&bytes[4..8]) as usize / 64;
            let tf_bits = bytes[2] as usize;
            let tfs_size = BLOCK_LEN * tf_bits / 8;
            let presence = &bytes[posting::HEADER_SIZE..bytes.len() - tfs_size];
            for (i, chunk) in presence.chunks_exact(8).enumerate() {
                dest[base_word + i] |= u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            }
        } else {
            let n = decode_block_doc_ids(bytes, scratch);
            for &d in &scratch[..n] {
                dest[(d >> 6) as usize] |= 1u64 << (d & 63);
            }
        }
    }
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
    // Frontier merge over the remaining (rarer) cursors: smallest current
    // doc, membership-test the anchor, then advance every other cursor
    // sitting on that doc so each distinct doc is visited once.
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
        // Membership by bit-test, not `skip_to`: the anchor is the dominant
        // (common ⇒ dense) term, so its blocks are bitsets. `contains` answers
        // in one word-load; `skip_to` would decode each block — expanding its
        // ~128 doc ids — as the frontier drags the anchor across its whole list.
        if !anchor.contains(min_doc) {
            n += 1;
        }
        for c in cursors.iter_mut() {
            if !c.is_exhausted() && c.current_doc_id() == min_doc {
                c.next();
            }
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{super::test_util::*, *};
    use crate::superfile::{
        BytesLazyByteSource,
        fts::{builder::FtsBuilder, reader::BoolMode, tokenize::AsciiLowerTokenizer},
    };

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
