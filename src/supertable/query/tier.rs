// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Choosing the superfiles a ranked query opens from the term index's
//! tiers, before any superfile is opened.
//!
//! A per-term score ceiling cannot tell the superfiles that hold a
//! multi-term query's top k from the rest: on an arrival-ordered layout
//! every superfile holds a near-best document for every common term, so
//! the sum of per-term maxima is near its ceiling everywhere. What does
//! tell them apart is a *document* that scores on several of the query's
//! terms, and the tier carries exactly that: per superfile, each term's
//! best documents there — its *head* — with the facts their scores are
//! computed from, in blocks, each block followed by a *tail* that bounds
//! every document of the superfile the block leaves unread.
//!
//! The merge proceeds in rounds. The first reads one block of every
//! superfile for every term; each later round reads the next block of
//! only the superfiles that are still in play. A round's reads are issued
//! together and overlap; its decoding, scoring and joining run as one
//! wave on the reader pool. After each round, per superfile, the
//! documents read so far are joined across the terms, and two bounds fall
//! out:
//!
//! - `θ_lb`, a lower bound on the final k-th score: the k-th best sum
//!   among documents carrying every required term, whose scores can only
//!   grow as more of their terms are read.
//! - `ub(s)`, an upper bound on any document of superfile `s`: the best
//!   over its known documents of the known sum plus, for each term not yet
//!   seen for it, that term's tail in `s`; or, for a document in no head
//!   yet, the sum of the tails.
//!
//! A superfile whose `ub(s)` falls below `θ_lb` holds no document that can
//! place in the top k and is never opened. The tails fall quickly with
//! depth on long documents, so most superfiles are ruled out after a block
//! or two, and the survivors open in order of their best known candidate
//! with `θ_lb` seeded as the shared floor. A superfile whose heads for
//! every term were read to the end is answered from the index outright.
//! A round that neither rules a superfile out nor raises the threshold
//! ends the merge: where every superfile holds competitive documents the
//! bounds never fall below the threshold, and each further round would
//! cost what the opens then cost anyway. While the threshold still rises
//! the merge reads on — a conjunction's candidates may sit deeper than
//! the first block, and a required term's head read to its end rules out
//! every superfile where the terms never meet.
//!
//! Scores are the kernel's own arithmetic at the table's statistics (see
//! [`Scoring`]); keys and tails stored at a superfile's own average length
//! are scaled to the table's as the ceilings are.

use std::{
    cmp::{Ordering as CmpOrdering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU32, AtomicU64, Ordering},
    },
    time::Instant,
};

use futures::future::try_join_all;
use rayon::{
    ThreadPool,
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use roaring::RoaringBitmap;
use tokio::task;
use uuid::Uuid;

use super::fts::baked_avgdl;
use crate::{
    runtime_bridge::run_on_pool,
    superfile::fts::{
        bm25::{Bm25Params, score_with_dl_norm_k1},
        reader::{GlobalTermIdf, norm_table},
    },
    supertable::{
        error::QueryError,
        manifest::{
            SuperfileEntry,
            part::ContentHash,
            term_index::{
                BLOCK_ENTRIES, SliceRun, TermIndex, TermIndexError, TierChunkData, TierChunkRef,
                TierDirectory, TierEntry,
            },
        },
        schema::FieldId,
    },
};

/// Terms up to which a candidate's summed contributions equal the kernel's
/// score bit for bit: one term is one product, two commute, three or more
/// depend on the order the kernel adds them in.
const EXACT_SUM_TERMS: usize = 2;
/// Widening applied to a bound taken from a stored key — a tail or a
/// ceiling — which is rounded once more than the kernel rounds a score.
/// Mirrors the term index's ceiling slack.
const BOUND_SLACK: f32 = 1.0 + 8.0 * f32::EPSILON;
/// A term's tier up to this size is read whole in the merge's first wave: it
/// costs what its first block costs, one ranged read, and the rounds after
/// the first then need no storage round trip. A larger tier — a common
/// term's, tens of blocks for every superfile — reads its first block alone,
/// and each later block only in the round that merges it, which at a small
/// k is rarely reached. Measured on a term index over fifty thousand
/// superfiles: a common term's whole tier is 80 to 100 MB, its first blocks
/// 11 to 13 MB.
const TIER_FIRST_WAVE_MAX_BYTES: u64 = 4 << 20;
/// Most distinct terms, phrase members included, the merge serves: a
/// document's terms are a bit mask. Past two bare terms a candidate's sum
/// carries slack (see [`EXACT_SUM_TERMS`]); the bounds hold for any
/// number, and the gate on the merge (see `SupertableOptions::
/// tier_merge_gate`) is what keeps a many-term query from paying for a
/// first round that cannot pay back.
pub(crate) const MAX_TIER_TERMS: usize = 64;

/// A set of query terms, a bit per term in query order.
pub(crate) type TermMask = u64;

/// Whether a ranked query merges its tiers at all, learned per table from
/// what the two ways cost: per class of query — conjunction, disjunction
/// or a mix; a bucket of `k`; a bucket of term count; a bucket of how rare
/// the rarest term is — a running average of the ranked search's wall
/// with the merge and without it. The two
/// ways alternate until each has [`TierPolicy::MIN_SAMPLES`] queries, so
/// both averages are taken over the same mix of queries; then the cheaper
/// way is taken and one query in [`TierPolicy::EXPLORE_EVERY`] takes the
/// other, so both stay current as the table and its cache change. What
/// the merge buys — opens it rules out, the threshold it seeds the kernels
/// with — is thereby measured as what it is worth here, on this table,
/// rather than modelled.
pub(crate) struct TierPolicy {
    /// Per class, the wall of the ranked search without the merge (`[0]`)
    /// and with it (`[1]`), in nanoseconds; zero until measured.
    walls: Vec<[AtomicU64; 2]>,
    /// Per class, how many queries each way has been measured over.
    samples: Vec<[AtomicU32; 2]>,
    trials: Vec<AtomicU32>,
    /// Per class, merges in a row that reported and ruled nothing out,
    /// answered for nothing and seeded no threshold; reset by one that did.
    fruitless: Vec<AtomicU32>,
}

impl TierPolicy {
    const SHAPES: usize = 3;
    const K_BUCKETS: usize = 4;
    const TERM_BUCKETS: usize = 5;
    const RARITY_BUCKETS: usize = 4;
    const CLASSES: usize =
        Self::SHAPES * Self::K_BUCKETS * Self::TERM_BUCKETS * Self::RARITY_BUCKETS;
    /// One query in this many takes the way the averages do not favour,
    /// while the two averages are within [`Self::CLEAR_GAP`] of each other.
    const EXPLORE_EVERY: u32 = 16;
    /// One query in this many looks again once the averages differ by
    /// [`Self::CLEAR_GAP`] or more: a decision that clear is rarely wrong,
    /// and every look costs a query the slower way — at a billion
    /// documents, seconds against a tenth of one.
    const EXPLORE_EVERY_CLEAR: u32 = 64;
    /// The ratio of the two averages past which the decision is clear.
    const CLEAR_GAP: u64 = 4;
    /// Queries each way is measured over, alternating, before the averages
    /// are compared: queries of a class differ in cost several-fold, so an
    /// average over one query does not compare with one over the mix.
    const MIN_SAMPLES: u32 = 8;
    /// The average's reach: each sample moves it a `2^-SMOOTHING` of the way.
    const SMOOTHING: u32 = 3;
    /// Merges in a row that ruled nothing out, answered for nothing and
    /// seeded no threshold, after which the class stops merging until an
    /// explore tick finds otherwise. Such a merge changes nothing about the
    /// opens or the kernels and only costs, and that is known the moment it
    /// reports, with no timing to average; a merge that did any of the three
    /// is left to the timed comparison.
    const FRUITLESS_LIMIT: u32 = 2;

    pub(crate) fn new() -> Self {
        Self {
            walls: (0..Self::CLASSES)
                .map(|_| [AtomicU64::new(0), AtomicU64::new(0)])
                .collect(),
            samples: (0..Self::CLASSES)
                .map(|_| [AtomicU32::new(0), AtomicU32::new(0)])
                .collect(),
            trials: (0..Self::CLASSES).map(|_| AtomicU32::new(0)).collect(),
            fruitless: (0..Self::CLASSES).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    /// The class of a query: whether its terms are all required, none, or
    /// some; its `k`; its distinct term count; how rare its rarest term
    /// is, as that term's table-wide idf. The kernels' work and what the
    /// merge can rule out both follow the rarest term far more than the
    /// term count: a conjunction anchored on a term in one document in
    /// ten thousand is a different query from one over three common
    /// words, and the two ways cost differently on each. Idf is a function
    /// of the term's share of the corpus alone (ln(1 + (N − df + ½)/(df +
    /// ½)), close to ln(N/df)), so the buckets mean the same on any table:
    /// a share of at most one in ten thousand, in a thousand, in a
    /// hundred, or more.
    pub(crate) fn class(
        all_required: bool,
        none_required: bool,
        k: usize,
        terms: usize,
        rarest_idf: f32,
    ) -> usize {
        let shape = match (all_required, none_required) {
            (true, _) => 0,
            (false, true) => 1,
            _ => 2,
        };
        let kb = match k {
            0..=10 => 0,
            11..=100 => 1,
            101..=1000 => 2,
            _ => 3,
        };
        let tb = match terms {
            0..=1 => 0,
            2 => 1,
            3..=4 => 2,
            5..=8 => 3,
            _ => 4,
        };
        let rb = match rarest_idf {
            r if r >= 1e4f32.ln() => 0,
            r if r >= 1e3f32.ln() => 1,
            r if r >= 1e2f32.ln() => 2,
            _ => 3,
        };
        ((shape * Self::K_BUCKETS + kb) * Self::TERM_BUCKETS + tb) * Self::RARITY_BUCKETS + rb
    }

    /// Whether a query of `class` merges its tiers this time.
    pub(crate) fn merge(&self, class: usize) -> bool {
        let trial = self.trials[class].fetch_add(1, Ordering::Relaxed);
        let explore = trial.is_multiple_of(Self::EXPLORE_EVERY);
        // A class whose merges keep finding nothing to prune does not merge,
        // whatever the walls say, except to look again now and then.
        if self.fruitless[class].load(Ordering::Relaxed) >= Self::FRUITLESS_LIMIT {
            return explore;
        }
        let n_without = self.samples[class][0].load(Ordering::Relaxed);
        let n_with = self.samples[class][1].load(Ordering::Relaxed);
        if n_without < Self::MIN_SAMPLES || n_with < Self::MIN_SAMPLES {
            // Alternate, the less-measured way first on a tie.
            return n_with <= n_without;
        }
        let without = self.walls[class][0].load(Ordering::Relaxed);
        let with = self.walls[class][1].load(Ordering::Relaxed);
        let favoured = with <= without;
        let clear = without.max(with) >= Self::CLEAR_GAP.saturating_mul(without.min(with));
        let explore = match clear {
            true => trial.is_multiple_of(Self::EXPLORE_EVERY_CLEAR),
            false => explore,
        };
        if explore { !favoured } else { favoured }
    }

    /// Fold a ranked search's wall into its class's average for the way
    /// it went: an eighth of the way from the old value to the sample.
    /// `pruned` says what a merge that reported did: `Some(true)` when it
    /// ruled a superfile out, answered for one, or seeded a threshold,
    /// `Some(false)` when none of those, `None` when it did not report
    /// (declined, or the opens finished first), which says nothing about
    /// its worth.
    pub(crate) fn note(&self, class: usize, merged: bool, pruned: Option<bool>, wall_ns: u64) {
        let arm = usize::from(merged);
        self.samples[class][arm].fetch_add(1, Ordering::Relaxed);
        fold_mean(&self.walls[class][arm], wall_ns, Self::SMOOTHING);
        match pruned {
            Some(true) => self.fruitless[class].store(0, Ordering::Relaxed),
            Some(false) => {
                self.fruitless[class].fetch_add(1, Ordering::Relaxed);
            }
            None => {}
        }
    }
}
/// Entries the merge's first round may decode: a query whose first round
/// alone would pass it is not merged. Sized for a table of tens of
/// thousands of superfiles: a common term's first round there is one block
/// of up to 64 entries per superfile, about three million at fifty
/// thousand, so two such terms merge and a third declines.
const MAX_FIRST_ROUND_ENTRIES: u64 = 1 << 23;
/// Entries the merge reads per query before it stops regardless of what is
/// still in play. What has been read by then still bounds every superfile
/// soundly; the merge just stops refining. Reads are held only for the
/// superfiles still in play — a superfile ruled out lets its go — so what
/// the merge holds at once is well under this.
const MAX_ENTRIES_PER_QUERY: u64 = 1 << 26;
/// Superfiles one pool job decodes in a round. A superfile's share of a
/// round is several microseconds, so jobs this size still dwarf the
/// pool's scheduling while a round of a hundred superfiles spreads over
/// a few dozen threads.
const SLOTS_PER_JOB: usize = 4;

/// What the merge decided for one kept superfile.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SuperfileBounds {
    /// Lower bound on the best score a document of this superfile reaches
    /// (`NEG_INFINITY` when no candidate was seen). Orders the opens.
    pub(crate) lb: f32,
    /// Upper bound on the score of any document of this superfile;
    /// `NEG_INFINITY` when the merge ruled it out.
    pub(crate) ub: f32,
}

/// The query the merge serves: bare terms, the ones a match must carry,
/// the column's statistics they score at, and how many documents it wants.
pub(crate) struct TierQuery<'a> {
    pub(crate) column: FieldId,
    /// Every distinct term the query scores, phrase members included.
    pub(crate) terms: &'a [&'a str],
    /// A bit per term (in `terms` order) a matching document must carry:
    /// the musts and the members of must-phrases.
    pub(crate) required: TermMask,
    /// A bit per term that scores on its own (a bare term); a term that is
    /// only a phrase member scores through the phrase.
    pub(crate) scoring: TermMask,
    /// Each phrase as its members (indices into `terms`, in phrase order,
    /// a repeated word repeated) and whether a match must carry it. A
    /// phrase scores at most its members' idf sum — every occurrence
    /// counted, as the kernel weights it — times the smallest member key
    /// of the document, the bound the ceiling path uses.
    pub(crate) phrases: Vec<(Vec<usize>, bool)>,
    /// Whether the query negates terms or phrases: a document the heads
    /// show cannot then be taken as a match, so the merge bounds and
    /// orders but seeds no threshold and answers for nothing.
    pub(crate) negated: bool,
    pub(crate) k: usize,
    /// Per kept superfile, the rows deleted since it was written, when
    /// any: a deleted document is in the tiers but not in the table, so
    /// the merge reads past it as a kernel's tombstone filter drops it.
    pub(crate) deny: Vec<Option<Arc<RoaringBitmap>>>,
    /// The average length every kept superfile scores at.
    pub(crate) table_avgdl: f32,
    /// The column's declared parameters, which every kept superfile scores
    /// at.
    pub(crate) params: Bm25Params,
    /// What one superfile open costs on this table, in nanoseconds of
    /// query wall; zero when not yet known. A round of the merge is read
    /// while the opens the last round saved are worth more than the round
    /// took — the trade that differs between a table whose opens are a
    /// few microseconds of warm memory and one whose opens are reads.
    pub(crate) open_cost_ns: u64,
    /// What one tier entry costs a merge on this table, in nanoseconds of
    /// query wall; zero when not yet known.
    pub(crate) entry_cost_ns: u64,
    /// Whether to skip the merge where its first round, at that per-entry
    /// cost, costs more than every open it could save at the open cost.
    pub(crate) merge_gate: bool,
    /// Whether the merge runs whatever the gate would say: a table with
    /// more kept superfiles than one open window, where the opens the
    /// merge can save dwarf a round's cost and only the merge can tell
    /// tied superfiles apart. The measured costs still decide how deep it
    /// reads.
    pub(crate) forced: bool,
}

/// The merge's outcome for one query.
#[derive(Debug)]
pub(crate) struct TierSelection {
    /// Bounds per kept superfile, in the order they were given.
    pub(crate) bounds: Vec<SuperfileBounds>,
    /// Lower bound on the final k-th score; `NEG_INFINITY` when fewer than
    /// `k` documents were seen with every required term.
    pub(crate) theta_lb: f32,
    /// The highest `_id` among the superfiles holding the `k` candidates
    /// that set [`Self::theta_lb`], equal candidates taken from the lowest
    /// `_id` ranges first. The `k` candidates are real documents scoring at
    /// least the threshold, at `_id`s no higher than this, so a document
    /// that only ties the threshold in a superfile whose `_id`s all lie
    /// above it ranks behind all `k`: seeding the floor with it lets the
    /// fan-out skip such a superfile on the tie. `i128::MAX` when no
    /// threshold was found.
    pub(crate) theta_id_max: i128,
    /// Per superfile the index answers for outright — every term's head
    /// there was read to its end, with scores the kernel would compute bit
    /// for bit — its place among the kept and its matching documents as
    /// `(row, score)`. Such a superfile need not be opened; tombstones
    /// still apply. Empty unless the scores are exact and the caller asked.
    pub(crate) resolved: Vec<(usize, Vec<(u32, f32)>)>,
    /// Tier bytes fetched from storage: directories and block windows.
    pub(crate) tier_bytes: u64,
    /// Tier entries decoded.
    pub(crate) entries_read: u64,
    /// Wall the merge's rounds took, decoding included.
    pub(crate) merge_ns: u64,
}

/// One term's chunked tier in one layer: where its regions sit, and
/// the regions once read.
struct Source {
    object: ContentHash,
    chunks: Vec<TierChunkRef>,
    data: Vec<Arc<TierChunkData>>,
    /// What the ordinals in the chunks are relative to: the run's layer
    /// base.
    base: u32,
}

/// What the merge knows of one term in one kept superfile.
#[derive(Clone)]
struct HeadState {
    /// Whether any document of the term was read for the superfile.
    seen: bool,
    /// Documents read so far as `(row, contribution, tf and length)`, in
    /// ascending row (the last packed by [`pack_tf_len`]).
    docs: Vec<Read>,
    /// Documents read since the last refresh, in the order read.
    pending: Vec<Read>,
    /// Whether the index holds the term's every document of the superfile
    /// in the head, so reading it to its end settles which documents
    /// carry the term.
    whole: bool,
    /// Blocks the head spans when whole (its documents over a block's
    /// entries, rounded up): what finishing it costs.
    blocks: u32,
    /// Upper bound, at the table's stats, on the contribution of any
    /// document of the superfile not in `docs`. Zero once every document
    /// is in `docs`.
    tail: f32,
    /// Upper bound, at the table's stats, on the term's contribution to
    /// any document of the superfile: its stored ceiling. Zero when the
    /// superfile holds no document of the term.
    ceiling: f32,
    /// Nothing more of the head can be read: it was read to its end, or
    /// the superfile holds no block beyond.
    done: bool,
    /// Which of the term's sources holds this superfile's head, when
    /// chunked.
    source: Option<usize>,
}

impl HeadState {
    fn absent() -> Self {
        Self {
            seen: false,
            docs: Vec::new(),
            pending: Vec::new(),
            whole: true,
            blocks: 0,
            tail: 0.0,
            ceiling: 0.0,
            done: true,
            source: None,
        }
    }
}

/// One document read under one term: its row, its contribution, and its
/// term frequency and length bucket packed by [`pack_tf_len`].
type Read = (u32, f32, u32);

/// Bits of a packed `(tf, length bucket)` the length bucket takes.
const LEN_BITS: u32 = 8;
/// The packed term frequency that stands for any frequency too large to
/// pack: the document's phrase bound then falls back to its contribution.
const TF_UNKNOWN: u32 = u32::MAX >> LEN_BITS;

/// The largest frequency a phrase is bounded at by the kernel's own
/// arithmetic. The score `idf·tf / (tf + norm)` rises with `tf` by at least
/// `norm / (tf·(tf + 1 + norm))` of itself per step, with a norm no smaller
/// than `k1·(1 − b)` (0.3 at the defaults): up to this frequency the step
/// is more than twice the rounding the arithmetic can add, so a document
/// at a lower frequency never rounds above the bound. Past it the phrase
/// is bounded by its members' keys, with slack.
const PHRASE_TF_EXACT_MAX: u32 = 512;

/// A document's term frequency and one-byte length bucket in one word.
fn pack_tf_len(tf: u32, len: u8) -> u32 {
    (tf.min(TF_UNKNOWN) << LEN_BITS) | u32::from(len)
}

/// The term frequency and length bucket [`pack_tf_len`] packed;
/// [`TF_UNKNOWN`] for a frequency too large to pack.
fn unpack_tf_len(packed: u32) -> (u32, u8) {
    (packed >> LEN_BITS, (packed & 0xff) as u8)
}

/// The shape of the query the merge bounds: which terms a match must
/// carry, which score on their own, the phrases, and whether a document
/// the heads show can be trusted to match (no must-phrase to verify, no
/// negated term to check, a bare term to score on).
#[derive(Clone)]
struct Shape {
    required: TermMask,
    scoring: TermMask,
    phrases: Vec<(Vec<usize>, bool)>,
    trusted: bool,
    /// Whether a document's score is a sum of at most two atoms — bare
    /// terms and phrases, each scored as the kernel scores it — with
    /// nothing negated: such a sum is the kernel's bit for bit, since two
    /// additions commute, and its bound needs no slack.
    exact_sums: bool,
}

/// What the merge knows of one kept superfile, and what it derives from
/// it: each term's documents in row order, joined across the terms in one
/// pass whenever a round read something for the superfile, giving the
/// bounds and the top candidates.
struct Slot {
    heads: Vec<HeadState>,
    /// The best `k` candidate sums (narrowed by the lower slack) of
    /// documents carrying every required term, descending: what the
    /// superfile offers the threshold.
    top: Vec<f32>,
    /// Whether a round read something for the superfile since its bounds
    /// and `top` were last computed.
    touched: bool,
    /// Upper bound on any of the superfile's documents, lower bound on its
    /// best candidate (see [`Slot::refresh`]).
    ub: f32,
    lb: f32,
    /// Whether the merge still reads for the superfile.
    live: bool,
    /// The ordinal the superfile's heads are filed under in the index.
    ordinal: u32,
    /// The ratio the superfile's stored keys and tails are scaled by to
    /// hold at the table's average length.
    scale_up: f32,
    /// Rows deleted from the superfile: read past, never candidates.
    deny: Option<Arc<RoaringBitmap>>,
    /// The highest `_id` the superfile holds; `i128::MAX` until known,
    /// which never lets a threshold it sets justify a skip.
    id_max: i128,
}

impl Slot {
    /// A superfile before any round: every head absent, nothing bounded.
    fn new(terms: usize, scale_up: f32, deny: Option<Arc<RoaringBitmap>>) -> Self {
        Self {
            heads: vec![HeadState::absent(); terms],
            top: Vec::new(),
            touched: false,
            ub: f32::INFINITY,
            lb: f32::NEG_INFINITY,
            live: true,
            ordinal: u32::MAX,
            scale_up,
            deny,
            id_max: i128::MAX,
        }
    }
}

/// Buffers a refresh works in, reused across superfiles.
#[derive(Default)]
struct Scratch {
    /// A contribution of zero per term, for a document read under none.
    zero: Vec<f32>,
    merged: Vec<Read>,
    at: Vec<usize>,
    contrib: Vec<f32>,
    tf_len: Vec<u32>,
    candidates: Vec<f32>,
}

/// One block decoded for one superfile and term in a round.
struct BlockRead<'a> {
    ti: usize,
    dir: &'a TierDirectory,
    block: usize,
    body: &'a [u8],
}

/// The merge's state across rounds, moved to the reader pool for each
/// round's decoding and back.
struct Merge {
    slots: Vec<Slot>,
    scoring: Scoring,
    shape: Shape,
    k: usize,
    lb_slack: f32,
    ub_slack: f32,
    theta_lb: f32,
    /// See [`TierSelection::theta_id_max`].
    theta_id_max: i128,
    entries_read: u64,
}

/// Whether `opens` superfile opens at `open_cost_ns` each are worth
/// `rounds` more rounds at `round_ns` each. With the open cost not yet
/// measured, any open is taken as worth it.
fn pays(opens: u64, rounds: u64, open_cost_ns: u64, round_ns: u64) -> bool {
    if open_cost_ns == 0 {
        return opens > 0;
    }
    u128::from(opens) * u128::from(open_cost_ns) >= u128::from(rounds) * u128::from(round_ns)
}

/// Whether the first wave reads a term's chunked tier whole: when the
/// term's tier across all its sources is small enough that the whole of it
/// costs what its first blocks cost, one ranged read per source. Judged per
/// term, not per source: a common term spread over many layers is small in
/// each and large in all, and read whole it would cost the merge every
/// block of every superfile when a round or two of them is what it reads.
fn read_whole_first(term_sources: &[Source]) -> bool {
    let whole: u64 = term_sources
        .iter()
        .flat_map(|s| s.chunks.iter())
        .map(|c| c.len)
        .sum();
    whole <= TIER_FIRST_WAVE_MAX_BYTES
}

/// Per term and source, whether a superfile still within reach of `bar`
/// reads a head from it: a chunk no such head reads is not fetched.
fn sources_read_on(slots: &[Slot], sources: &[Vec<Source>], bar: f32) -> Vec<Vec<bool>> {
    let mut needed: Vec<Vec<bool>> = sources.iter().map(|s| vec![false; s.len()]).collect();
    for slot in slots.iter().filter(|s| s.reads_on(bar)) {
        for (ti, head) in slot.heads.iter().enumerate() {
            if let (false, Some(si)) = (head.done, head.source) {
                needed[ti][si] = true;
            }
        }
    }
    needed
}

/// Fold `sample` into the running average in `cell`, moving it a
/// `2^-shift` of the way from its old value; the first sample sets it.
pub(crate) fn fold_mean(cell: &AtomicU64, sample: u64, shift: u32) {
    let old = cell.load(Ordering::Relaxed);
    let new = if old == 0 {
        sample
    } else {
        old - (old >> shift) + (sample >> shift)
    };
    cell.store(new, Ordering::Relaxed);
}

/// A merge prepared up to its reads: the kept superfiles' slots bounded
/// by their postings, the inline heads absorbed, the chunked heads
/// located. [`PreparedMerge::run`] reads them and merges.
pub(crate) struct PreparedMerge<'a> {
    index: &'a TermIndex,
    pool: &'a Arc<ThreadPool>,
    merge: Merge,
    sources: Vec<Vec<Source>>,
    required: TermMask,
    merge_gate: bool,
    forced: bool,
    open_cost_ns: u64,
    /// Whether the sums are the kernel's bit for bit, so a superfile whose
    /// heads were read whole can be answered from the index.
    exact_scores: bool,
    /// Whether every chunked head the merge reads is in memory: the merge
    /// then costs no storage round trip, and is worth running before any
    /// open for the kernels it spares; one that must read from storage
    /// runs beside the first opens instead.
    pub(crate) resident: bool,
}

/// Prepare the merge of `query` over the tiers of `index`. `None` when a
/// term has postings in a layer without a tier, a kept superfile is not
/// in the root, or a table-wide idf is missing: the caller then keeps its
/// ceilings.
pub(crate) async fn prepare_merge<'a>(
    index: &'a TermIndex,
    query: &TierQuery<'_>,
    kept: &[Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
    pool: &'a Arc<ThreadPool>,
) -> Result<Option<PreparedMerge<'a>>, TermIndexError> {
    let terms = query.terms;
    let nt = terms.len();
    let k = query.k;
    if nt == 0 || nt > MAX_TIER_TERMS || k == 0 || kept.is_empty() {
        return Ok(None);
    }
    // The gate: the merge's first round decodes a block per term per
    // superfile; where that, at the per-entry cost measured here, costs
    // more than every open it could save at the open cost measured here,
    // it cannot pay back and the ceilings serve instead.
    if query.merge_gate
        && !query.forced
        && query.open_cost_ns > 0
        && query.entry_cost_ns > 0
        && (nt as u64)
            .saturating_mul(BLOCK_ENTRIES as u64)
            .saturating_mul(query.entry_cost_ns)
            > query.open_cost_ns
    {
        return Ok(None);
    }
    // A superfile the root does not list has postings the merge would never
    // see; bounding it at nothing would be wrong, so no selection is made.
    let listed: HashSet<Uuid> = index.root().superfiles.iter().copied().collect();
    if kept.iter().any(|e| !listed.contains(&e.superfile_id)) {
        return Ok(None);
    }
    let mut slot_of: HashMap<Uuid, usize> = HashMap::with_capacity(kept.len());
    for (i, e) in kept.iter().enumerate() {
        slot_of.insert(e.superfile_id, i);
    }
    let mut idf: Vec<f32> = Vec::with_capacity(nt);
    for term in terms {
        let Some(&idf_table) = global_idf.get(*term) else {
            return Ok(None);
        };
        idf.push(idf_table);
    }
    // The terms' runs first, and nothing more if a term has postings in a
    // layer without a tier: no sound bound exists for it, and a table
    // indexed before tiers would otherwise pay a slot per superfile on
    // every query only to be declined.
    let runs_per_term = index.runs_many(query.column, terms).await?;
    if runs_per_term.iter().any(|runs| {
        runs.iter()
            .any(|r| r.run.tier.is_none() && !r.run.postings.is_empty())
    }) {
        return Ok(None);
    }
    // Every kept superfile scores at the table's average length (its norm
    // table is rebuilt there when it records the average it was built at),
    // so a candidate's contribution is computed exactly as its kernel
    // would. A superfile that records no average is scored at an average
    // the index does not know, so no selection is made.
    let mut slots: Vec<Slot> = Vec::with_capacity(kept.len());
    for e in kept {
        let Some(baked) = baked_avgdl(e, query.column).filter(|b| *b > 0.0) else {
            return Ok(None);
        };
        // A superfile built at a smaller average scores at most
        // `table/baked` higher at the table's: the length norm falls by at
        // most that ratio, and `tf / (tf + norm)` rises by no more than the
        // norm falls.
        let scale_up = if baked < query.table_avgdl {
            (query.table_avgdl / baked).next_up()
        } else {
            1.0
        };
        let mut slot = Slot::new(nt, scale_up, query.deny.get(slots.len()).cloned().flatten());
        slot.id_max = e.id_max;
        slots.push(slot);
    }
    // The slot of a superfile the index files under `ordinal`; `None` for
    // one the query does not keep.
    let slot_for = |ordinal: u32| {
        index
            .superfile_id(ordinal)
            .and_then(|id| slot_of.get(&id).copied())
    };
    let mut sources: Vec<Vec<Source>> = (0..nt).map(|_| Vec::new()).collect();
    let mut inline: Vec<(usize, TierEntry)> = Vec::new();
    // What the first round decodes: the first block of every chunked head,
    // which holds the head's documents up to a block's worth.
    let mut first_round = 0u64;
    for (ti, runs) in runs_per_term.iter().enumerate() {
        let idf_table = idf[ti];
        for SliceRun {
            tier_object,
            run,
            base,
        } in runs.iter()
        {
            // Runs without a tier and without postings were passed over
            // above; one with postings declined the merge there.
            let Some(tier) = &run.tier else {
                continue;
            };
            let chunked = !tier.chunks.is_empty();
            let source = if chunked {
                let Some(object) = *tier_object else {
                    return Ok(None);
                };
                sources[ti].push(Source {
                    object,
                    chunks: tier.chunks.clone(),
                    data: Vec::new(),
                    base: *base,
                });
                Some(sources[ti].len() - 1)
            } else {
                None
            };
            for p in &run.postings {
                let Some(si) = slot_for(p.superfile) else {
                    continue;
                };
                let slot = &mut slots[si];
                slot.ordinal = p.superfile;
                // The stored bound and tail are keys at the superfile's own
                // average; the table's idf and the scale to its average make
                // them bounds on the term's contribution, as the ceiling
                // path makes its ceilings. A stored key is rounded once more
                // than the kernel rounds a score, so what it bounds is
                // widened by an ulp, as the ceiling path widens its own.
                let ceiling = p.bound * idf_table * slot.scale_up * BOUND_SLACK;
                // An inline head is in hand whole, so what it leaves out is
                // bounded by its tail; a chunked head bounds nothing below
                // its ceiling until a block of it is read (a build that
                // recorded no tail leaves the ceiling either way).
                let tail = if p.tail.is_finite() && !chunked {
                    (p.tail * idf_table * slot.scale_up * BOUND_SLACK).min(ceiling)
                } else {
                    ceiling
                };
                slot.heads[ti] = HeadState {
                    seen: false,
                    docs: Vec::new(),
                    pending: Vec::new(),
                    whole: p.tail.is_finite() && p.tail <= 0.0,
                    blocks: p.df.div_ceil(BLOCK_ENTRIES as u64) as u32,
                    tail,
                    ceiling,
                    done: !chunked,
                    source,
                };
                if chunked {
                    first_round += p.df.min(BLOCK_ENTRIES as u64);
                }
            }
            if !chunked {
                inline.extend(tier.inline.iter().map(|e| (ti, *e)));
            }
        }
    }

    // A query whose first round alone would pass the entry cap is not merged.
    if first_round > MAX_FIRST_ROUND_ENTRIES {
        return Ok(None);
    }

    // A contribution is the kernel's own arithmetic, so a one- or two-term
    // sum is the score bit for bit (addition commutes); a longer sum can
    // land an ulp from the kernel's, which adds in its own order, so its
    // bounds take the slack a product of roundings needs.
    let exact_scores = nt <= EXACT_SUM_TERMS && query.phrases.is_empty() && !query.negated;
    let atoms = query.scoring.count_ones() as usize + query.phrases.len();
    let shape = Shape {
        exact_sums: atoms <= EXACT_SUM_TERMS && !query.negated,
        required: query.required,
        scoring: query.scoring,
        phrases: query.phrases.clone(),
        // A document the heads show matches only if nothing unseen can
        // unmake it: no must-phrase whose adjacency the heads cannot
        // verify, no negated term the heads do not read — and only
        // through a bare term it scores on, since a phrase alone may not
        // match at all.
        trusted: !query.negated && query.phrases.iter().all(|p| !p.1) && query.scoring != 0,
    };
    // A shape whose candidates cannot be trusted yields no threshold: before
    // any open it rules a superfile out only through a required head the
    // index holds whole. It is merged all the same, for its bounds: a
    // document's bound — exact for a phrase whose members were all read —
    // is what lets the fan-out skip a superfile that can only tie the floor
    // the opens build, which a ceiling, loose by the length rescale, never
    // does.
    // What lies past a block is bounded through its tail, already widened
    // where the tail is set; a read document's sum needs widening only
    // when the kernel may add it in another order, by a rounding per
    // addition — the upper bound per document, as [`Shape::exact_sums`]
    // and the phrases it scores decide.
    let slack = (8 + nt) as f32 * f32::EPSILON;
    let lb_slack = if exact_scores { 1.0 } else { 1.0 - slack };
    let ub_slack = 1.0 + slack;
    let scoring = Scoring::new(idf, query.params, query.table_avgdl);
    // Inline heads are whole: their entries are in hand before any read.
    let mut entries_read = 0u64;
    for (ti, e) in &inline {
        let Some(si) = slot_for(e.superfile) else {
            continue;
        };
        let c = scoring.contribution(*ti, e);
        slots[si].absorb(*ti, e.doc_id, c, pack_tf_len(e.tf, e.len));
        entries_read += 1;
    }
    let mut merge = Merge {
        slots,
        scoring,
        shape,
        k,
        lb_slack,
        ub_slack,
        theta_lb: f32::NEG_INFINITY,
        theta_id_max: i128::MAX,
        entries_read,
    };
    let mut scratch = Scratch::default();
    for slot in merge.slots.iter_mut() {
        if slot.touched {
            slot.refresh(
                &merge.shape,
                &merge.scoring.idf,
                &merge.scoring.norm,
                k,
                lb_slack,
                ub_slack,
                &mut scratch,
            );
        }
    }

    let mut resident = true;
    for source in sources.iter().flatten() {
        // The first round's chunk decides whether the merge can run before any
        // open: a later round's is read only if the merge goes on.
        resident &= index
            .tier_chunks_resident(&source.object, &source.chunks[..source.chunks.len().min(1)])
            .await;
    }
    Ok(Some(PreparedMerge {
        index,
        pool,
        merge,
        sources,
        required: query.required,
        merge_gate: query.merge_gate,
        forced: query.forced,
        open_cost_ns: query.open_cost_ns,
        exact_scores,
        resident,
    }))
}

impl PreparedMerge<'_> {
    /// Start the first wave's reads and return, for a query that cannot use
    /// the plan in time — its opens all issue at once — but whose next query
    /// of these terms can: the reads land in the chunk cache, and that query
    /// finds them resident and merges before it opens.
    pub(crate) fn prefetch(self, index: Arc<TermIndex>) {
        for term_sources in &self.sources {
            let whole = read_whole_first(term_sources);
            for source in term_sources {
                let upto = match whole {
                    true => source.chunks.len(),
                    false => source.chunks.len().min(1),
                };
                let chunks = source.chunks[..upto].to_vec();
                let object = source.object;
                let index = Arc::clone(&index);
                task::spawn(async move {
                    let _ = index.tier_chunks(&object, &chunks).await;
                });
            }
        }
    }

    /// Read the chunked heads and merge, round by round.
    ///
    /// `floor` reads the running k-th score the opens beside the merge have
    /// built (`NEG_INFINITY` until they have k): a superfile whose bound is
    /// at or below it, or below the merge's own threshold, is skipped or
    /// ruled out whatever more of its heads would show, so the merge reads
    /// no further for it, and goes on reading only while a round brings
    /// enough superfiles to that point to pay for itself.
    pub(crate) async fn run(
        self,
        floor: Arc<dyn Fn() -> f32 + Send + Sync>,
    ) -> Result<Option<TierSelection>, TermIndexError> {
        let PreparedMerge {
            index,
            pool,
            merge,
            sources,
            required,
            merge_gate,
            forced,
            open_cost_ns,
            exact_scores,
            ..
        } = self;
        // A round merges the next block of every live head. The blocks of one
        // depth lie together, one chunk per term and layer. The first wave
        // reads a small tier whole and a large tier's first block, every
        // term's at once; each later round reads, of a large tier, only the
        // next block's chunk, and only where a live superfile still reads
        // that term's head. A common term's tier on a table of tens of
        // thousands of superfiles is tens of chunks of megabytes each, and
        // a merge that stops after a round or two needs none of the rest.
        let max_rounds = match crate::config::global().diagnostics.tier_rounds {
            0 => usize::MAX,
            n => n as usize,
        };
        // With the gate off the merge is under test and reads on while a round
        // excludes anything, whatever the opens were measured to cost: the
        // rounds it takes then depend on the table alone. A merge forced by
        // the table's size keeps the measured costs: they decide its depth.
        let open_cost_ns = if merge_gate || forced {
            open_cost_ns
        } else {
            0
        };
        let whole_first: Vec<bool> = sources.iter().map(|s| read_whole_first(s)).collect();
        let mut tier_bytes = 0u64;
        let mut merge_ns = 0u64;
        let mut merge = merge;
        let mut sources = sources;
        for round in 0..max_rounds {
            // The bar a superfile's bound must clear to still matter: the
            // merge's threshold, or the floor the opens have built.
            let bar = merge.theta_lb.max(floor());
            let reachable: Vec<bool> = merge.slots.iter().map(|s| s.reads_on(bar)).collect();
            // What this round still has to read, per source: a small tier
            // whole in the first wave; of a large one, the chunk at this
            // round's depth, where a head still within reach reads it.
            let needed = sources_read_on(&merge.slots, &sources, bar);
            let reads: Vec<(usize, usize, Range<usize>)> = sources
                .iter()
                .enumerate()
                .flat_map(|(ti, term_sources)| {
                    let whole = whole_first[ti];
                    let needed = &needed[ti];
                    term_sources
                        .iter()
                        .enumerate()
                        .filter_map(move |(si, source)| {
                            let have = source.data.len();
                            if have > round || have >= source.chunks.len() || !needed[si] {
                                return None;
                            }
                            let upto = match whole {
                                true => source.chunks.len(),
                                false => round + 1,
                            };
                            Some((ti, si, have..upto))
                        })
                })
                .collect();
            let fetches: Vec<_> = reads
                .iter()
                .map(|(ti, si, span)| {
                    let source = &sources[*ti][*si];
                    index.tier_chunks(&source.object, &source.chunks[span.clone()])
                })
                .collect();
            let fetched: Vec<(Vec<Arc<TierChunkData>>, u64)> = try_join_all(fetches).await?;
            for ((ti, si, _), (data, bytes)) in reads.iter().zip(fetched) {
                sources[*ti][*si].data.extend(data);
                tier_bytes += bytes;
            }
            // Decode, score, join and bound the round as one job on the reader
            // pool: the next block of every live superfile whose head goes on.
            let (merge_back, sources_back, outcome) =
                run_on_pool(Some(pool), "tier merge", move || {
                    let mut merge = merge;
                    let sources = sources;
                    let outcome = {
                        let mut reads: Vec<Vec<BlockRead<'_>>> =
                            (0..merge.slots.len()).map(|_| Vec::new()).collect();
                        let mut read_any = false;
                        for (slot_i, slot) in merge.slots.iter_mut().enumerate() {
                            if !slot.reads_on(bar) {
                                continue;
                            }
                            for (ti, state) in slot.heads.iter_mut().enumerate() {
                                if state.done {
                                    continue;
                                }
                                let Some(si) = state.source else {
                                    continue;
                                };
                                let Some(chunk) = sources[ti][si].data.get(round) else {
                                    state.done = true;
                                    continue;
                                };
                                // The chunk stores ordinals relative to its
                                // layer's base.
                                let Some(stored) = slot.ordinal.checked_sub(sources[ti][si].base)
                                else {
                                    state.done = true;
                                    continue;
                                };
                                let Some(block) = chunk.dir.block_of(stored) else {
                                    state.done = true;
                                    continue;
                                };
                                reads[slot_i].push(BlockRead {
                                    ti,
                                    dir: &chunk.dir,
                                    block,
                                    body: &chunk.bodies[block],
                                });
                                read_any = true;
                            }
                        }
                        if !read_any {
                            None
                        } else {
                            let started = Instant::now();
                            let excluded = merge.round(&reads);
                            let round_ns = started.elapsed().as_nanos() as u64;
                            // A required term whose head the index holds whole
                            // settles, read to its end, which documents carry the
                            // term — ruling out, or answering for, every live
                            // superfile the conjunction has no joint document in.
                            // Finishing it is worth the rounds it still needs while
                            // the live opens at stake cover them at this round's
                            // cost; with the open cost unknown it is finished
                            // regardless.
                            let rounds_to_settle = merge
                                .slots
                                .iter()
                                .filter(|s| s.live)
                                .flat_map(|s| s.heads.iter().enumerate())
                                .filter(|(ti, h)| required & (1 << ti) != 0 && h.whole && !h.done)
                                .map(|(_, h)| h.blocks.saturating_sub(round as u32 + 1).max(1))
                                .max();
                            let settling = rounds_to_settle.is_some_and(|rounds| {
                                let live = merge.slots.iter().filter(|s| s.live).count() as u64;
                                pays(live, u64::from(rounds), open_cost_ns, round_ns)
                            });
                            Some((round_ns, excluded, settling))
                        }
                    };
                    (merge, sources, outcome)
                })
                .await
                .map_err(|e| TermIndexError::Open(QueryError::Internal(e.to_string())))?;
            merge = merge_back;
            sources = sources_back;
            let Some((round_ns, excluded, settling)) = outcome else {
                break;
            };
            merge_ns += round_ns;
            // Worth reading on: the superfiles this round brought out of
            // reach — ruled out, or bounded at or below the bar, so skipped
            // whatever their heads hold — at what an open costs here, cover
            // what a round costs; the next round is taken to settle as many
            // as this one did. With the open cost unknown, any is taken as
            // worth it. A round that settled none is not read past: on a
            // query whose best documents are the sum of different
            // documents' terms, deeper blocks keep every bound above the
            // bar, and reading them buys nothing.
            let bar_after = merge.theta_lb.max(floor());
            let settled = merge
                .slots
                .iter()
                .zip(&reachable)
                .filter(|(s, was)| **was && !s.reads_on(bar_after))
                .count() as u64;
            let worth = pays(settled, 1, open_cost_ns, round_ns);
            let read_on = worth || settling;
            tracing::debug!(
                round,
                excluded,
                settled,
                live = merge.slots.iter().filter(|s| s.live).count(),
                theta = merge.theta_lb,
                bar = bar_after,
                entries = merge.entries_read,
                round_us = round_ns / 1000,
                open_cost_us = open_cost_ns / 1000,
                settling,
                read_on,
                "tier merge round"
            );
            if !read_on || merge.entries_read >= MAX_ENTRIES_PER_QUERY {
                break;
            }
        }

        let bounds: Vec<SuperfileBounds> = merge
            .slots
            .iter()
            .map(|s| SuperfileBounds {
                lb: s.lb * merge.lb_slack,
                ub: if s.live { s.ub } else { f32::NEG_INFINITY },
            })
            .collect();
        let mut resolved = Vec::new();
        if exact_scores {
            for (si, slot) in merge.slots.iter().enumerate() {
                // Every head read to its end, and the sums exact: the superfile's
                // matching documents are its candidates carrying the required
                // terms, scored as the kernel would score them. One the merge
                // ruled out holds none that place.
                let whole = slot.live && slot.heads.iter().all(|h| h.done && h.tail <= 0.0);
                if whole {
                    let mut docs: Vec<(u32, f32)> = Vec::new();
                    let mut scratch = Scratch::default();
                    slot.join(&mut scratch, |row, mask, contrib, _| {
                        if mask & required == required {
                            docs.push((row, contrib.iter().sum()));
                        }
                    });
                    resolved.push((si, docs));
                }
            }
        }
        Ok(Some(TierSelection {
            bounds,
            theta_lb: merge.theta_lb,
            theta_id_max: merge.theta_id_max,
            resolved,
            tier_bytes,
            entries_read: merge.entries_read,
            merge_ns,
        }))
    }
}

impl Merge {
    /// One round's decoding: per superfile that read a block, its join,
    /// best sums and candidates grow entry by entry, and its bounds and
    /// top candidates are recomputed, in jobs of [`SLOTS_PER_JOB`]
    /// superfiles on the pool; then the threshold over every superfile's
    /// candidates, and the exclusion of what cannot reach it. Returns how
    /// many superfiles the round ruled out.
    fn round(&mut self, reads: &[Vec<BlockRead<'_>>]) -> usize {
        let shape = &self.shape;
        let (k, lb_slack, ub_slack) = (self.k, self.lb_slack, self.ub_slack);
        let scoring = &self.scoring;
        let entries: u64 = self
            .slots
            .par_chunks_mut(SLOTS_PER_JOB)
            .zip(reads.par_chunks(SLOTS_PER_JOB))
            .map(|(slots, reads)| {
                let mut decoded: Vec<TierEntry> = Vec::new();
                let mut scratch = Scratch::default();
                let mut n = 0u64;
                for (slot, reads) in slots.iter_mut().zip(reads) {
                    for read in reads {
                        let state = &mut slot.heads[read.ti];
                        if read
                            .dir
                            .decode_block(read.block, read.body, &mut decoded)
                            .is_err()
                        {
                            // A block that does not decode leaves the rest
                            // of the head unread: it bounds nothing below
                            // its ceiling, and the superfile opens as it
                            // would have.
                            state.tail = state.ceiling;
                            state.done = true;
                            continue;
                        }
                        n += decoded.len() as u64;
                        let tail_key = read.dir.block_tail(read.block);
                        state.tail =
                            (tail_key * scoring.idf[read.ti] * slot.scale_up * BOUND_SLACK)
                                .min(state.ceiling);
                        if tail_key <= 0.0 || decoded.len() < BLOCK_ENTRIES {
                            state.done = true;
                        }
                        for e in &decoded {
                            let c = scoring.contribution(read.ti, e);
                            slot.absorb(read.ti, e.doc_id, c, pack_tf_len(e.tf, e.len));
                        }
                    }
                    if slot.touched {
                        slot.refresh(
                            shape,
                            &scoring.idf,
                            &scoring.norm,
                            k,
                            lb_slack,
                            ub_slack,
                            &mut scratch,
                        );
                    }
                }
                n
            })
            .sum();
        self.entries_read += entries;
        // Threshold: the k-th best sum over documents the query returns,
        // across every superfile read, in play or not. Each superfile's
        // top candidates are sorted, so a k-way walk finds it in k steps.
        // Equal candidates come from the lowest `_id` ranges first, so the
        // highest `_id` the walk draws on is as low as the k allow.
        let (kth, id_max) = kth_candidate(&self.slots, k);
        if let Some(kth) = kth {
            if kth > self.theta_lb {
                self.theta_lb = kth;
                self.theta_id_max = id_max;
            } else if kth == self.theta_lb {
                self.theta_id_max = self.theta_id_max.min(id_max);
            }
        }
        // Rule out what cannot compete.
        let mut excluded = 0usize;
        if self.theta_lb.is_finite() {
            let theta = self.theta_lb;
            for slot in self.slots.iter_mut() {
                if slot.live && slot.ub < theta {
                    slot.live = false;
                    excluded += 1;
                    // Out of play for good: its candidates stay for the
                    // threshold, its reads are no longer needed.
                    for h in slot.heads.iter_mut() {
                        h.docs = Vec::new();
                        h.pending = Vec::new();
                    }
                }
            }
        }
        excluded
    }
}

/// The k-th best candidate across `slots`, if there are `k`, and the highest
/// `_id` among the superfiles the walk drew its `k` candidates from: a k-way
/// walk over each superfile's sorted candidates, taking equal candidates
/// from the lowest `_id` ranges first.
fn kth_candidate(slots: &[Slot], k: usize) -> (Option<f32>, i128) {
    let mut heap: BinaryHeap<(OrdF32, Reverse<i128>, usize, usize)> = BinaryHeap::new();
    for (si, slot) in slots.iter().enumerate() {
        if let Some(&best) = slot.top.first() {
            heap.push((OrdF32(best), Reverse(slot.id_max), si, 0));
        }
    }
    let mut kth = None;
    let mut id_max = i128::MIN;
    for _ in 0..k {
        let Some((OrdF32(value), Reverse(slot_id_max), si, at)) = heap.pop() else {
            return (None, i128::MAX);
        };
        kth = Some(value);
        id_max = id_max.max(slot_id_max);
        if let Some(&next) = slots[si].top.get(at + 1) {
            heap.push((OrdF32(next), Reverse(slot_id_max), si, at + 1));
        }
    }
    (kth, id_max)
}

impl Slot {
    /// Whether the merge still reads for the superfile against `bar`, the
    /// threshold or the floor: it is live and its bound clears the bar. A
    /// superfile bounded at or below it is ruled out or skipped whatever
    /// more of its heads would show; the bar only rises, so it stays so.
    fn reads_on(&self, bar: f32) -> bool {
        self.live && (self.ub > bar || self.ub.is_nan())
    }

    /// Take in one entry: document `row` read under term `ti` with
    /// contribution `c` and its frequency and length packed by
    /// [`pack_tf_len`]. It waits in the term's pending list until the next
    /// refresh merges it into the term's documents.
    fn absorb(&mut self, ti: usize, row: u32, c: f32, tf_len: u32) {
        self.heads[ti].seen = true;
        if self.deny.as_ref().is_some_and(|d| d.contains(row)) {
            return;
        }
        self.heads[ti].pending.push((row, c, tf_len));
        self.touched = true;
    }

    /// Visit every document read under any term, in ascending row, with
    /// the terms it was read under, and per term its contribution and its
    /// packed frequency and length (zero where it was not read): a k-way
    /// merge of the terms' documents.
    fn join(&self, scratch: &mut Scratch, mut visit: impl FnMut(u32, TermMask, &[f32], &[u32])) {
        let nt = self.heads.len();
        scratch.at.clear();
        scratch.at.resize(nt, 0);
        scratch.contrib.clear();
        scratch.contrib.resize(nt, 0.0);
        scratch.tf_len.clear();
        scratch.tf_len.resize(nt, 0);
        loop {
            let mut next: Option<u32> = None;
            for (ti, h) in self.heads.iter().enumerate() {
                if let Some(&(row, _, _)) = h.docs.get(scratch.at[ti]) {
                    next = Some(next.map_or(row, |n| n.min(row)));
                }
            }
            let Some(row) = next else { break };
            let mut mask: TermMask = 0;
            for (ti, h) in self.heads.iter().enumerate() {
                scratch.contrib[ti] = 0.0;
                scratch.tf_len[ti] = 0;
                if let Some(&(r, c, tf_len)) = h.docs.get(scratch.at[ti])
                    && r == row
                {
                    mask |= 1 << ti;
                    scratch.contrib[ti] = c;
                    scratch.tf_len[ti] = tf_len;
                    scratch.at[ti] += 1;
                }
            }
            visit(row, mask, &scratch.contrib, &scratch.tf_len);
        }
    }

    /// Merge what was read since the last refresh into each term's
    /// documents, then recompute the bounds and the top candidates in one
    /// join pass.
    ///
    /// A document scores at most its bare terms — each its contribution
    /// if read under it, else the term's tail — plus, per phrase, the
    /// members' idf sum times the smallest member key, a read member's own
    /// or the tail's; a document read under no term takes every tail. A
    /// required term's head read to its end holds every document of the
    /// term there, so a document outside it carries no such term and is no
    /// candidate; a superfile that lacks a required term holds no match at
    /// all. The lower bound is the best bare sum of a document carrying
    /// every required term, where such a document can be trusted to match,
    /// and the candidates are those sums narrowed by the lower slack.
    #[allow(clippy::too_many_arguments)]
    fn refresh(
        &mut self,
        shape: &Shape,
        idf: &[f32],
        norm: &[f32; 256],
        k: usize,
        lb_slack: f32,
        ub_slack: f32,
        scratch: &mut Scratch,
    ) {
        self.touched = false;
        for h in self.heads.iter_mut() {
            if h.pending.is_empty() {
                continue;
            }
            h.pending.sort_unstable_by_key(|d| d.0);
            merge_sorted(&mut h.docs, &h.pending, &mut scratch.merged);
            h.pending.clear();
        }
        let nt = self.heads.len();
        // Required terms whose heads are whole: a candidate must be in them.
        let mut settled: TermMask = 0;
        for (ti, h) in self.heads.iter().enumerate() {
            if shape.required & (1 << ti) != 0 {
                if h.ceiling <= 0.0 && !h.seen {
                    self.top.clear();
                    self.ub = f32::NEG_INFINITY;
                    self.lb = f32::NEG_INFINITY;
                    return;
                }
                if h.tail <= 0.0 {
                    settled |= 1 << ti;
                }
            }
        }
        let heads = &self.heads;
        // The bound for a document read under `mask` with contributions
        // `c` and packed frequencies and lengths `tf_len` (every tail for
        // one read under no term). Widened by the slack only where its sum
        // may land an ulp from the kernel's: past two atoms, or through a
        // phrase bounded by its members' keys rather than scored as the
        // kernel scores it.
        let doc_ub = |mask: TermMask, c: &[f32], tf_len: &[u32]| -> f32 {
            let mut ub = 0.0f32;
            let mut exact = shape.exact_sums;
            for (t, h) in heads.iter().enumerate() {
                if shape.scoring & (1 << t) != 0 {
                    ub += if mask & (1 << t) != 0 { c[t] } else { h.tail };
                }
            }
            for (members, _) in &shape.phrases {
                if let Some(score) = phrase_atom_bound(members, mask, tf_len, idf, norm) {
                    ub += score;
                    continue;
                }
                exact = false;
                let mut idf_sum = 0.0f32;
                let mut min_key = f32::INFINITY;
                for &t in members {
                    idf_sum += idf[t];
                    let key = if mask & (1 << t) != 0 {
                        c[t] / idf[t]
                    } else {
                        heads[t].tail / idf[t]
                    };
                    min_key = min_key.min(key);
                }
                ub += idf_sum * min_key;
            }
            match exact {
                true => ub,
                false => ub * ub_slack,
            }
        };
        scratch.zero.clear();
        scratch.zero.resize(nt, 0.0);
        scratch.tf_len.clear();
        scratch.tf_len.resize(nt, 0);
        let no_tf_len = std::mem::take(&mut scratch.tf_len);
        let mut ub = if settled == 0 {
            doc_ub(0, &scratch.zero, &no_tf_len)
        } else {
            f32::NEG_INFINITY
        };
        let mut lb = f32::NEG_INFINITY;
        scratch.candidates.clear();
        let mut candidates = std::mem::take(&mut scratch.candidates);
        scratch.tf_len = no_tf_len;
        self.join(scratch, |_, mask, contrib, tf_len| {
            if mask & settled != settled {
                return;
            }
            ub = ub.max(doc_ub(mask, contrib, tf_len));
            if shape.trusted && mask & shape.required == shape.required {
                let bare = bare_sum(contrib, shape.scoring);
                lb = lb.max(bare);
                candidates.push(bare * lb_slack);
            }
        });
        self.top.clear();
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k, |a, b| b.total_cmp(a));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| b.total_cmp(a));
        self.top.extend_from_slice(&candidates);
        scratch.candidates = candidates;
        self.ub = ub;
        self.lb = lb;
    }
}

/// Merge `incoming`, in ascending row, into `docs`, also ascending,
/// through `scratch`.
fn merge_sorted(docs: &mut Vec<Read>, incoming: &[Read], scratch: &mut Vec<Read>) {
    scratch.clear();
    let mut at = 0usize;
    for &item in incoming {
        while at < docs.len() && docs[at].0 < item.0 {
            scratch.push(docs[at]);
            at += 1;
        }
        scratch.push(item);
    }
    scratch.extend_from_slice(&docs[at..]);
    std::mem::swap(docs, scratch);
}

/// What phrase `members` (term indices in phrase order, a repeated word
/// repeated) scores at most in a document read under one or more of them,
/// as the kernel scores a phrase: one atom weighted by the members' idf
/// summed in phrase order, at the phrase's frequency. Each occurrence of
/// the phrase puts every member at its own position, one per occurrence
/// start, so the phrase occurs at most as often as any member does; the
/// members read for the document give that frequency and the document's
/// length, and the score rises with frequency, so this is an upper bound —
/// and the kernel's own score, bit for bit, where the phrase occurs that
/// often. A member not read for the document needs no frequency: the
/// phrase cannot occur more often than a read member allows. `None` when
/// no member was read at a frequency up to [`PHRASE_TF_EXACT_MAX`]: the
/// caller then bounds the phrase by its members' keys.
fn phrase_atom_bound(
    members: &[usize],
    mask: TermMask,
    tf_len: &[u32],
    idf: &[f32],
    norm: &[f32; 256],
) -> Option<f32> {
    let mut idf_sum = 0.0f32;
    let mut most: Option<(u32, u8)> = None;
    for &t in members {
        idf_sum += idf[t];
        if mask & (1 << t) == 0 {
            continue;
        }
        let (tf, len) = unpack_tf_len(tf_len[t]);
        if tf <= PHRASE_TF_EXACT_MAX && most.is_none_or(|(m, _)| tf < m) {
            most = Some((tf, len));
        }
    }
    let (tf, len) = most?;
    Some(score_with_dl_norm_k1(idf_sum, tf, norm[usize::from(len)]))
}

/// The sum of a document's bare-term contributions.
fn bare_sum(contrib: &[f32], scoring: TermMask) -> f32 {
    contrib
        .iter()
        .enumerate()
        .filter(|(t, _)| scoring & (1 << t) != 0)
        .map(|(_, c)| c)
        .sum()
}

/// How a tier entry's contribution to a query is computed: the kernel's
/// arithmetic at the table's statistics.
struct Scoring {
    /// Per term, the table-wide idf the kernel weights it with.
    idf: Vec<f32>,
    /// Per length bucket, the length norm the kernel adds to `tf`: the
    /// kernel's own table, at the column's parameters and the average
    /// every kept superfile scores at.
    norm: Arc<[f32; 256]>,
}

impl Scoring {
    fn new(idf: Vec<f32>, params: Bm25Params, avgdl: f32) -> Self {
        Self {
            idf,
            norm: norm_table(avgdl, params),
        }
    }

    /// What term `ti` contributes to the score of a document of `e.tf`
    /// occurrences and length bucket `e.len`: the kernel's own arithmetic.
    fn contribution(&self, ti: usize, e: &TierEntry) -> f32 {
        score_with_dl_norm_k1(self.idf[ti], e.tf, self.norm[usize::from(e.len)])
    }
}

/// Total-order wrapper for the threshold heap.
#[derive(PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.0.total_cmp(&other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frequency and length the bound oracles do not model: a phrase over
    /// such documents is bounded by its members' contributions.
    const UNREAD_TF_LEN: u32 = TF_UNKNOWN << LEN_BITS;
    /// A length norm the same for every bucket.
    const FLAT_NORM: [f32; 256] = [1.0; 256];

    /// Cases the bound oracle below draws.
    const BOUND_CASES: u64 = 4_000;
    /// Documents in the oracle's universe per case.
    const UNIVERSE: u32 = 48;
    /// Most terms the oracle joins: enough for phrases of three over bare
    /// terms, few enough that a universe of 48 documents exercises every
    /// mask.
    const TERMS: usize = 5;

    /// The bounds hold against every corpus the read heads could have come
    /// from: documents outside a term's head carry at most that term's tail
    /// (nothing when the head was read to its end), a phrase scores its
    /// members' idf sum times its smallest member key and is present
    /// wherever its members all are (the worst case for a bound), and the
    /// true best candidate's score sits between the lower and the upper
    /// bound.
    #[test]
    #[allow(clippy::needless_range_loop)]
    fn bounds_hold_against_every_corpus_the_heads_allow() {
        let mut next = crate::test_helpers::xorshift(0x9E37_79B9_7F4A_7C15);
        for case in 0..BOUND_CASES {
            let nt = 1 + (next() % TERMS as u64) as usize;
            let all: TermMask = (1u64 << nt) - 1;
            // Bare terms, and up to two phrases over the rest, each of two
            // to four members, a word possibly repeated; a phrase may be
            // required.
            let mut scoring: TermMask = next() % (1u64 << nt);
            let mut phrases: Vec<(Vec<usize>, bool)> = Vec::new();
            if nt >= 2 {
                for _ in 0..(next() % 3) {
                    let len = 2 + (next() % 3) as usize;
                    let members: Vec<usize> =
                        (0..len).map(|_| (next() % nt as u64) as usize).collect();
                    phrases.push((members, next().is_multiple_of(2)));
                }
            }
            if phrases.is_empty() && scoring == 0 {
                scoring = all;
            }
            let mut required: TermMask = (next() % (1u64 << nt)) & scoring;
            for (members, must) in &phrases {
                if *must {
                    for &t in members {
                        required |= 1 << t;
                    }
                }
            }
            let negated = next().is_multiple_of(4);
            let shape = Shape {
                required,
                scoring,
                phrases: phrases.clone(),
                trusted: !negated && phrases.iter().all(|p| !p.1),
                exact_sums: false,
            };
            let idf: Vec<f32> = (0..nt)
                .map(|_| 0.5 + (next() % 1000) as f32 / 500.0)
                .collect();
            let mut slot = Slot::new(nt, 1.0, None);
            // Per term: a head of documents with contributions, and a tail
            // (zero means the head is whole); every document outside the
            // head takes a hidden contribution within the tail.
            let mut truth: Vec<Vec<f32>> = Vec::with_capacity(nt);
            for ti in 0..nt {
                let whole = next().is_multiple_of(3);
                let in_head = (next() % u64::from(UNIVERSE)) as u32;
                let tail = if whole {
                    0.0
                } else {
                    idf[ti] * (next() % 1000) as f32 / 1000.0
                };
                let mut contribution = vec![0.0f32; UNIVERSE as usize];
                let mut ceiling = tail;
                for doc in 0..UNIVERSE {
                    if next() % u64::from(UNIVERSE) < u64::from(in_head) {
                        // A head entry contributes at least the tail: the
                        // head is the term's best documents.
                        let c = tail + idf[ti] * (next() % 1000) as f32 / 1000.0;
                        slot.absorb(ti, doc, c, UNREAD_TF_LEN);
                        contribution[doc as usize] = c;
                        ceiling = ceiling.max(c);
                    } else if !whole && next().is_multiple_of(2) {
                        contribution[doc as usize] = tail * (next() % 1000) as f32 / 1000.0;
                    }
                }
                slot.heads[ti].tail = tail;
                slot.heads[ti].ceiling = ceiling;
                slot.heads[ti].done = whole;
                truth.push(contribution);
            }
            slot.refresh(
                &shape,
                &idf,
                &FLAT_NORM,
                10,
                1.0,
                1.0,
                &mut Scratch::default(),
            );
            let (ub, lb) = (slot.ub, slot.lb);
            // The truth: the best document carrying every required term,
            // scored as the kernel would at worst for the bound — every
            // phrase present wherever its members all are.
            let mut best = f32::NEG_INFINITY;
            for doc in 0..UNIVERSE as usize {
                let carries = |t: usize| truth[t][doc] > 0.0;
                if (0..nt).any(|t| required & (1 << t) != 0 && !carries(t)) {
                    continue;
                }
                let mut score = 0.0f32;
                for t in 0..nt {
                    if scoring & (1 << t) != 0 {
                        score += truth[t][doc];
                    }
                }
                for (members, _) in &phrases {
                    let mut idf_sum = 0.0f32;
                    let mut min_key = f32::INFINITY;
                    for &t in members {
                        idf_sum += idf[t];
                        min_key = min_key.min(truth[t][doc] / idf[t]);
                    }
                    score += idf_sum * min_key;
                }
                best = best.max(score);
            }
            assert!(
                best <= ub * BOUND_SLACK,
                "case {case}: the best candidate {best} exceeds the upper bound {ub}"
            );
            assert!(
                lb <= best || lb == f32::NEG_INFINITY,
                "case {case}: the lower bound {lb} exceeds the best candidate {best}"
            );
        }
    }

    /// The policy alternates until each way has its samples, then takes
    /// the cheaper way and tries the other one query in sixteen while the
    /// two are within a few times of each other.
    #[test]
    fn the_policy_measures_both_ways_before_it_chooses() {
        let policy = TierPolicy::new();
        let class = TierPolicy::class(true, false, 10, 2, 5.0);
        let mut ways = Vec::new();
        for _ in 0..(2 * TierPolicy::MIN_SAMPLES) {
            let with = policy.merge(class);
            ways.push(with);
            // Merging costs twice what not merging costs, and prunes.
            policy.note(
                class,
                with,
                with.then_some(true),
                if with { 2_000 } else { 1_000 },
            );
        }
        assert_eq!(
            ways,
            (0..2 * TierPolicy::MIN_SAMPLES)
                .map(|i| i % 2 == 0)
                .collect::<Vec<_>>(),
            "the two ways alternate, the unmeasured one first"
        );
        let later: Vec<bool> = (0..2 * TierPolicy::EXPLORE_EVERY)
            .map(|_| policy.merge(class))
            .collect();
        let explored = later.iter().filter(|w| **w).count();
        assert_eq!(explored, 2, "one query in sixteen tries the dearer way");
        assert!(
            later[0] && !later[1] && later[TierPolicy::EXPLORE_EVERY as usize],
            "the trials divisible by sixteen explore: {later:?}"
        );
        let other = TierPolicy::class(false, true, 10, 2, 5.0);
        assert!(
            policy.merge(other),
            "another class starts its own measurement"
        );
    }

    /// Two merges in a row that prune nothing stop a class merging, whatever
    /// the walls say; the explore tick asks again, and a merge that prunes
    /// starts the count over. A merge that did not report changes nothing.
    #[test]
    fn a_class_whose_merges_prune_nothing_stops_merging() {
        let policy = TierPolicy::new();
        let class = TierPolicy::class(true, false, 100, 2, 3.0);
        // Merging is cheap here, so the walls alone would keep it.
        assert!(policy.merge(class));
        policy.note(class, true, Some(false), 1_000);
        assert!(!policy.merge(class));
        policy.note(class, false, None, 10_000);
        assert!(policy.merge(class));
        policy.note(class, true, Some(false), 1_000);
        let next: Vec<bool> = (0..3).map(|_| policy.merge(class)).collect();
        assert_eq!(next, vec![false, false, false], "fruitless twice: off");
        for _ in 0..3 {
            policy.note(class, false, None, 10_000);
        }
        // Trials so far: 6; the sixteenth asks again.
        let until_explore: Vec<bool> = (6..TierPolicy::EXPLORE_EVERY)
            .map(|_| {
                let m = policy.merge(class);
                policy.note(class, m, None, 10_000);
                m
            })
            .collect();
        assert!(until_explore.iter().all(|m| !m));
        assert!(policy.merge(class), "the explore tick merges");
        policy.note(class, true, Some(true), 1_000);
        assert!(
            policy.merge(class),
            "a merge that pruned starts the count over"
        );
        policy.note(class, true, None, 1_000);
        policy.note(class, true, Some(false), 1_000);
        assert!(
            policy.merge(class),
            "an unreported merge is neutral, and one fruitless merge does not stop the class"
        );
    }

    /// Once the two ways' averages differ by [`TierPolicy::CLEAR_GAP`] or
    /// more, the class looks at the slower way one query in
    /// [`TierPolicy::EXPLORE_EVERY_CLEAR`] rather than one in
    /// [`TierPolicy::EXPLORE_EVERY`]; a narrower gap keeps the closer look.
    #[test]
    fn a_clear_decision_is_explored_less_often() {
        /// Trials counted after the samples are in: two clear-explore ticks.
        const TRIALS: u32 = 2 * TierPolicy::EXPLORE_EVERY_CLEAR;
        let settled = |with_ns: u64, without_ns: u64| {
            let policy = TierPolicy::new();
            let class = TierPolicy::class(true, false, 10, 2, 9.0);
            for _ in 0..TierPolicy::MIN_SAMPLES {
                let m = policy.merge(class);
                policy.note(class, m, Some(true), if m { with_ns } else { without_ns });
                let m = policy.merge(class);
                policy.note(class, m, Some(true), if m { with_ns } else { without_ns });
            }
            let trials_so_far = 2 * TierPolicy::MIN_SAMPLES;
            // Count from the next multiple of the wider period, so each
            // arm sees whole periods.
            let skip =
                TierPolicy::EXPLORE_EVERY_CLEAR - trials_so_far % TierPolicy::EXPLORE_EVERY_CLEAR;
            for _ in 0..skip {
                let m = policy.merge(class);
                policy.note(class, m, Some(true), if m { with_ns } else { without_ns });
            }
            (0..TRIALS)
                .filter(|_| {
                    let m = policy.merge(class);
                    policy.note(class, m, Some(true), if m { with_ns } else { without_ns });
                    !m
                })
                .count() as u32
        };
        assert_eq!(
            settled(1_000, 10_000),
            TRIALS / TierPolicy::EXPLORE_EVERY_CLEAR,
            "ten times faster with the merge: the slower way one query in sixty-four"
        );
        assert_eq!(
            settled(1_000, 2_000),
            TRIALS / TierPolicy::EXPLORE_EVERY,
            "twice as fast: still one query in sixteen"
        );
    }

    /// Classes split at the bucket edges of `k`, the term count and the
    /// rarest term's idf, and every class fits the tables.
    #[test]
    fn the_policy_class_splits_at_its_bucket_edges() {
        let class = |k, terms, idf| TierPolicy::class(true, false, k, terms, idf);
        for (a, b) in [(10, 11), (100, 101), (1000, 1001)] {
            assert_ne!(class(a, 2, 5.0), class(b, 2, 5.0), "k {a} and {b}");
            assert_eq!(class(a, 2, 5.0), class(a - 1, 2, 5.0), "k {a} and one less");
        }
        for (a, b) in [(1, 2), (2, 3), (4, 5), (8, 9)] {
            assert_ne!(class(10, a, 5.0), class(10, b, 5.0), "terms {a} and {b}");
        }
        for (a, b) in [(9.3, 9.1), (7.0, 6.8), (4.7, 4.5)] {
            assert_ne!(class(10, 2, a), class(10, 2, b), "idf {a} and {b}");
        }
        assert_eq!(class(10, 2, 1e4f32.ln()), class(10, 2, 12.0));
        assert_ne!(
            TierPolicy::class(true, false, 10, 2, 5.0),
            TierPolicy::class(false, false, 10, 2, 5.0),
            "a conjunction and a mix differ"
        );
        for shape in [(true, false), (false, true), (false, false)] {
            for k in [1, 50, 500, 5000] {
                for terms in [1, 2, 3, 6, 40] {
                    for idf in [0.1, 5.0, 8.0, 12.0] {
                        let c = TierPolicy::class(shape.0, shape.1, k, terms, idf);
                        assert!(
                            c < TierPolicy::CLASSES,
                            "class {c} of {}",
                            TierPolicy::CLASSES
                        );
                    }
                }
            }
        }
    }

    /// The first sample sets a running mean; each later one moves it a
    /// power-of-two fraction of the way.
    #[test]
    fn a_running_mean_is_set_by_its_first_sample_then_moves_a_fraction() {
        let cell = AtomicU64::new(0);
        fold_mean(&cell, 800, 3);
        assert_eq!(cell.load(Ordering::Relaxed), 800);
        fold_mean(&cell, 0, 3);
        assert_eq!(
            cell.load(Ordering::Relaxed),
            700,
            "an eighth of the way to zero"
        );
        fold_mean(&cell, 0, 2);
        assert_eq!(
            cell.load(Ordering::Relaxed),
            525,
            "a quarter of the way to zero"
        );
        fold_mean(&cell, 525, 2);
        assert_eq!(
            cell.load(Ordering::Relaxed),
            525,
            "a sample at the mean leaves it"
        );
    }

    /// Opens pay for rounds at the measured costs; with no open cost
    /// measured, any open does.
    #[test]
    fn opens_pay_for_rounds_at_the_measured_costs() {
        assert!(
            pays(1, 1, 0, 1_000),
            "unmeasured: one open is worth a round"
        );
        assert!(!pays(0, 1, 0, 1_000), "unmeasured: no open is not");
        assert!(
            pays(10, 1, 100, 1_000),
            "ten opens at 100 ns pay a 1 µs round"
        );
        assert!(!pays(9, 1, 100, 1_000));
        assert!(!pays(10, 2, 100, 1_000), "but not two rounds");
        assert!(
            pays(u64::MAX, 1, u64::MAX, u64::MAX),
            "no overflow at the extremes"
        );
    }

    /// A head read a block at a time: each block read lowers the tail,
    /// so the superfile's upper bound never rises, and the top candidates
    /// are the read documents' exact sums.
    #[test]
    fn a_head_read_in_blocks_bounds_what_it_has_not_read() {
        let shape = Shape {
            required: 0,
            scoring: 0b11,
            phrases: Vec::new(),
            trusted: true,
            exact_sums: false,
        };
        let idf = vec![1.0f32, 1.0];
        let mut slot = Slot::new(2, 1.0, None);
        let mut scratch = Scratch::default();
        // Term 1 is read whole at once; term 0 comes in two blocks with
        // tails 0.5 then 0.2.
        for (doc, c) in [(1u32, 0.9f32), (4, 0.7), (9, 0.6)] {
            slot.absorb(1, doc, c, UNREAD_TF_LEN);
        }
        slot.heads[1].tail = 0.0;
        slot.heads[1].ceiling = 0.9;
        slot.heads[1].done = true;
        for (doc, c) in [(4u32, 0.8f32), (2, 0.6), (7, 0.55)] {
            slot.absorb(0, doc, c, UNREAD_TF_LEN);
        }
        slot.heads[0].tail = 0.5;
        slot.heads[0].ceiling = 0.8;
        slot.refresh(&shape, &idf, &FLAT_NORM, 3, 1.0, 1.0, &mut scratch);
        let first = slot.ub;
        // Document 4 carries both terms: 0.8 + 0.7. Document 1 is read under
        // term 1 only, so it may still carry term 0 up to the tail: 0.9 + 0.5.
        let close = |got: &[f32], want: &[f32]| {
            got.len() == want.len() && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-6)
        };
        assert!(close(&slot.top, &[1.5, 0.9, 0.6]), "top {:?}", slot.top);
        assert!(
            (first - 1.5).abs() < 1e-6,
            "ub {first}: 0.8 + 0.7 and 0.9 + 0.5 tie"
        );
        for (doc, c) in [(1u32, 0.45f32), (9, 0.3)] {
            slot.absorb(0, doc, c, UNREAD_TF_LEN);
        }
        slot.heads[0].tail = 0.2;
        slot.refresh(&shape, &idf, &FLAT_NORM, 3, 1.0, 1.0, &mut scratch);
        assert!(slot.ub <= first, "ub rose from {first} to {}", slot.ub);
        assert!(close(&slot.top, &[1.5, 1.35, 0.9]), "top {:?}", slot.top);
        assert!(
            (slot.ub - 1.5).abs() < 1e-6,
            "ub {}: document 4 is the best",
            slot.ub
        );
    }

    /// A deleted row is read past: it is neither a candidate nor a bound.
    #[test]
    fn a_denied_row_is_read_past() {
        let shape = Shape {
            required: 0,
            scoring: 0b1,
            phrases: Vec::new(),
            trusted: true,
            exact_sums: false,
        };
        let deny: RoaringBitmap = [4u32].into_iter().collect();
        let mut slot = Slot::new(1, 1.0, Some(Arc::new(deny)));
        for (doc, c) in [(4u32, 0.8f32), (2, 0.6)] {
            slot.absorb(0, doc, c, UNREAD_TF_LEN);
        }
        slot.heads[0].tail = 0.0;
        slot.heads[0].ceiling = 0.8;
        slot.heads[0].done = true;
        slot.refresh(
            &shape,
            &[1.0],
            &FLAT_NORM,
            2,
            1.0,
            1.0,
            &mut Scratch::default(),
        );
        assert_eq!(slot.top, vec![0.6], "row 4 is deleted");
        assert!((slot.ub - 0.6).abs() < 1e-6);
    }

    /// The threshold's `_id` is the highest among the superfiles its `k`
    /// candidates come from, and equal candidates are drawn from the lowest
    /// `_id` ranges first.
    #[test]
    fn seed_carries_max_id_of_theta_candidates() {
        let slot = |top: &[f32], id_max: i128| {
            let mut s = Slot::new(1, 1.0, None);
            s.top = top.to_vec();
            s.id_max = id_max;
            s
        };
        let slots = [
            slot(&[5.0], 12),
            slot(&[5.0, 4.0], 9),
            slot(&[5.0], 7),
            slot(&[6.0], 20),
        ];
        assert_eq!(kth_candidate(&slots, 1), (Some(6.0), 20));
        assert_eq!(
            kth_candidate(&slots, 3),
            (Some(5.0), 20),
            "6.0 from 20, then the 5.0 ties from 7 and 9"
        );
        assert_eq!(
            kth_candidate(&slots[..3], 2),
            (Some(5.0), 9),
            "of three tied superfiles the two lowest"
        );
        assert_eq!(
            kth_candidate(&slots, 6),
            (None, i128::MAX),
            "five candidates are not six"
        );
    }

    /// A phrase over documents read under one or more members is bounded
    /// as the kernel scores it: never below the score at any frequency the
    /// read members allow, and equal to it, bit for bit, at the most they
    /// allow.
    #[test]
    fn phrase_atom_bound_never_below_the_kernel_and_equal_at_its_most() {
        const CASES: u64 = 20_000;
        const MAX_TF: u64 = 5;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        let mut norm = [0.0f32; 256];
        for (i, n) in norm.iter_mut().enumerate() {
            *n = 0.3 + i as f32 / 97.0;
        }
        for _ in 0..CASES {
            let nt = 1 + (next() % 3) as usize;
            let idf: Vec<f32> = (0..nt)
                .map(|_| 0.1 + (next() % 1000) as f32 / 97.0)
                .collect();
            let members: Vec<usize> = (0..2 + next() % 3)
                .map(|_| (next() % nt as u64) as usize)
                .collect();
            let len = (next() % 256) as u8;
            let tf: Vec<u32> = (0..nt).map(|_| 1 + (next() % MAX_TF) as u32).collect();
            let packed: Vec<u32> = tf.iter().map(|&t| pack_tf_len(t, len)).collect();
            // Any subset of the members read, one at least.
            let read = members[(next() % members.len() as u64) as usize];
            let mask: TermMask = ((1 << nt) - 1) & (next() | (1 << read));
            let most = members
                .iter()
                .filter(|&&t| mask & (1 << t) != 0)
                .map(|&t| tf[t])
                .min()
                .expect("a member read");
            let idf_sum = members.iter().fold(0.0f32, |sum, &t| sum + idf[t]);
            let bound = phrase_atom_bound(&members, mask, &packed, &idf, &norm).expect("one read");
            for phrase_tf in 0..=most {
                let kernel = score_with_dl_norm_k1(idf_sum, phrase_tf, norm[usize::from(len)]);
                assert!(bound >= kernel, "{members:?} tf {tf:?} at {phrase_tf}");
                if phrase_tf == most {
                    assert_eq!(bound.to_bits(), kernel.to_bits(), "{members:?} tf {tf:?}");
                }
            }
        }
    }

    /// Up to the frequency the phrase bound is exact at, the kernel's score
    /// rises strictly with frequency under every idf and every length norm
    /// the defaults allow: a document at a lower frequency never rounds
    /// above the bound.
    #[test]
    fn phrase_score_rises_with_frequency_up_to_its_exact_cap() {
        const MIN_NORM: f32 = 0.3;
        const NORMS: u32 = 400;
        const IDFS: [f32; 5] = [0.01, 0.7, 3.3, 17.0, 45.0];
        for idf in IDFS {
            for n in 0..NORMS {
                let norm = MIN_NORM + n as f32 * 0.137;
                for tf in 0..PHRASE_TF_EXACT_MAX {
                    let lower = score_with_dl_norm_k1(idf, tf, norm);
                    let higher = score_with_dl_norm_k1(idf, tf + 1, norm);
                    assert!(higher > lower, "idf {idf} norm {norm} tf {tf}");
                }
            }
        }
    }

    /// A phrase over a document read under no member with a frequency
    /// small enough to keep is left to the bound by keys; one member is
    /// enough, and a repeated member is not divided by its repeats (an
    /// occurrence of "a a" can start at every "a" but the last).
    #[test]
    fn phrase_atom_bound_needs_one_member_with_its_frequency() {
        let norm = [1.0f32; 256];
        let idf = [1.0, 1.0];
        let packed = [pack_tf_len(3, 3), pack_tf_len(u32::MAX, 3)];
        assert!(phrase_atom_bound(&[0, 1], 0b10, &packed, &idf, &norm).is_none());
        assert!(phrase_atom_bound(&[0, 1], 0b00, &packed, &idf, &norm).is_none());
        assert_eq!(
            phrase_atom_bound(&[0, 1], 0b01, &packed, &idf, &norm),
            Some(score_with_dl_norm_k1(2.0, 3, 1.0))
        );
        assert_eq!(
            phrase_atom_bound(&[0, 0], 0b01, &packed, &idf, &norm),
            Some(score_with_dl_norm_k1(2.0, 3, 1.0)),
            "\"a a\" occurs twice in \"a a a\": no division by repeats"
        );
        assert_eq!(unpack_tf_len(pack_tf_len(7, 200)), (7, 200));
    }

    /// A source of `chunks` chunks of `chunk_bytes` each.
    fn source(chunks: usize, chunk_bytes: u64) -> Source {
        Source {
            object: ContentHash([0; 32]),
            chunks: (0..chunks)
                .map(|i| TierChunkRef {
                    offset: i as u64 * chunk_bytes,
                    len: chunk_bytes,
                    n_blocks: 1,
                })
                .collect(),
            data: Vec::new(),
            base: 0,
        }
    }

    /// The first wave reads a term whole by the term's tier across all its
    /// sources: small in every layer is not small, when the layers are
    /// many.
    #[test]
    fn first_wave_reads_whole_only_when_the_terms_tier_is_small() {
        const MB: u64 = 1 << 20;
        let small = vec![source(3, MB)];
        assert!(read_whole_first(&small), "3 MB in all is read whole");
        let spread: Vec<Source> = (0..14).map(|_| source(4, MB / 2)).collect();
        assert!(
            spread
                .iter()
                .all(|s| s.chunks.iter().map(|c| c.len).sum::<u64>() <= 2 * MB),
            "every source alone is small"
        );
        assert!(
            !read_whole_first(&spread),
            "fourteen sources of 2 MB are 28 MB: first blocks only"
        );
    }

    /// A later round fetches a source's chunk only where a live superfile
    /// still reads that term's head there.
    #[test]
    fn later_rounds_read_only_sources_a_live_head_reads() {
        let sources = vec![vec![source(4, 1), source(4, 1)], vec![source(4, 1)]];
        let mut reading = Slot::new(2, 1.0, None);
        reading.heads[0] = HeadState {
            done: false,
            source: Some(1),
            ..HeadState::absent()
        };
        reading.heads[1] = HeadState {
            done: true,
            source: Some(0),
            ..HeadState::absent()
        };
        let mut ruled_out = Slot::new(2, 1.0, None);
        ruled_out.live = false;
        ruled_out.heads[0] = HeadState {
            done: false,
            source: Some(0),
            ..HeadState::absent()
        };
        let needed = sources_read_on(&[reading, ruled_out], &sources, f32::NEG_INFINITY);
        assert_eq!(
            needed,
            vec![vec![false, true], vec![false]],
            "only the live, unfinished head's source is read on"
        );
    }
}
