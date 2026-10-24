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
    cmp::Ordering as CmpOrdering,
    collections::{BinaryHeap, HashMap, HashSet},
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
    /// Per class, merges in a row that reported and ruled nothing out nor
    /// answered for anything; reset by one that did.
    fruitless: Vec<AtomicU32>,
}

impl TierPolicy {
    const SHAPES: usize = 3;
    const K_BUCKETS: usize = 4;
    const TERM_BUCKETS: usize = 5;
    const RARITY_BUCKETS: usize = 4;
    const CLASSES: usize =
        Self::SHAPES * Self::K_BUCKETS * Self::TERM_BUCKETS * Self::RARITY_BUCKETS;
    /// One query in this many takes the way the averages do not favour.
    const EXPLORE_EVERY: u32 = 16;
    /// Queries each way is measured over, alternating, before the averages
    /// are compared: queries of a class differ in cost several-fold, so an
    /// average over one query does not compare with one over the mix.
    const MIN_SAMPLES: u32 = 8;
    /// The average's reach: each sample moves it a `2^-SMOOTHING` of the way.
    const SMOOTHING: u32 = 3;
    /// Merges in a row that found nothing to rule out or answer for, after
    /// which the class stops merging until an explore tick finds otherwise.
    /// The merge is there to prune; where it prunes nothing it only costs,
    /// and that is known the moment it reports, with no timing to average.
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
        if explore { !favoured } else { favoured }
    }

    /// Fold a ranked search's wall into its class's average for the way
    /// it went: an eighth of the way from the old value to the sample.
    /// `pruned` says what a merge that reported found: `Some(true)` when it
    /// ruled a superfile out or answered for one, `Some(false)` when it
    /// found nothing, `None` when it did not report (declined, or the opens
    /// finished first), which says nothing about its worth.
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
/// Entries the merge reads per query before it stops regardless of what is
/// still in play. What has been read by then still bounds every superfile
/// soundly; the merge just stops refining.
const MAX_ENTRIES_PER_QUERY: u64 = 1 << 20;
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
    /// Whether the query negates terms: a document the heads show cannot
    /// then be taken as a match, so the merge bounds and orders but seeds
    /// no threshold and answers for nothing.
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
}

/// The merge's outcome for one query.
#[derive(Debug)]
pub(crate) struct TierSelection {
    /// Bounds per kept superfile, in the order they were given.
    pub(crate) bounds: Vec<SuperfileBounds>,
    /// Lower bound on the final k-th score; `NEG_INFINITY` when fewer than
    /// `k` documents were seen with every required term.
    pub(crate) theta_lb: f32,
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

/// One term's chunked tier in one segment: where its regions sit, and
/// the regions once read.
struct Source {
    object: ContentHash,
    chunks: Vec<TierChunkRef>,
    data: Vec<Arc<TierChunkData>>,
}

/// What the merge knows of one term in one kept superfile.
#[derive(Clone)]
struct HeadState {
    /// Whether any document of the term was read for the superfile.
    seen: bool,
    /// Documents read so far as `(row, contribution)`, in ascending row.
    docs: Vec<(u32, f32)>,
    /// Documents read since the last refresh, in the order read.
    pending: Vec<(u32, f32)>,
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
        }
    }
}

/// Buffers a refresh works in, reused across superfiles.
#[derive(Default)]
struct Scratch {
    /// A contribution of zero per term, for a document read under none.
    zero: Vec<f32>,
    merged: Vec<(u32, f32)>,
    at: Vec<usize>,
    contrib: Vec<f32>,
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
/// term has postings in a segment without a tier, a kept superfile is not
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
        && query.open_cost_ns > 0
        && query.entry_cost_ns > 0
        && (nt as u64)
            .saturating_mul(BLOCK_ENTRIES as u64)
            .saturating_mul(query.entry_cost_ns)
            > query.open_cost_ns
    {
        return Ok(None);
    }
    // The first round decodes a block per term per kept superfile: a query
    // whose first round alone would pass the entry cap is not merged.
    if (nt as u64)
        .saturating_mul(kept.len() as u64)
        .saturating_mul(BLOCK_ENTRIES as u64)
        > MAX_ENTRIES_PER_QUERY
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
        slots.push(Slot::new(
            nt,
            scale_up,
            query.deny.get(slots.len()).cloned().flatten(),
        ));
    }
    // The slot of a superfile the index files under `ordinal`; `None` for
    // one the query does not keep.
    let slot_for = |ordinal: u32| {
        index
            .superfile_id(ordinal)
            .and_then(|id| slot_of.get(&id).copied())
    };
    let runs_per_term = index.runs_many(query.column, terms).await?;
    let mut sources: Vec<Vec<Source>> = (0..nt).map(|_| Vec::new()).collect();
    let mut inline: Vec<(usize, TierEntry)> = Vec::new();
    for (ti, runs) in runs_per_term.iter().enumerate() {
        let idf_table = idf[ti];
        for SliceRun { tier_object, run } in runs.iter() {
            let tier = match &run.tier {
                Some(t) => t,
                // Postings without a tier: this segment predates tiers or
                // kept none for the term; no sound bound exists for it.
                None if run.postings.is_empty() => continue,
                None => return Ok(None),
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
            }
            if !chunked {
                inline.extend(tier.inline.iter().map(|e| (ti, *e)));
            }
        }
    }

    // A contribution is the kernel's own arithmetic, so a one- or two-term
    // sum is the score bit for bit (addition commutes); a longer sum can
    // land an ulp from the kernel's, which adds in its own order, so its
    // bounds take the slack a product of roundings needs.
    let exact_scores = nt <= EXACT_SUM_TERMS && query.phrases.is_empty() && !query.negated;
    let shape = Shape {
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
    // A shape whose candidates cannot be trusted yields no threshold, so
    // before any open it can rule a superfile out only through a required
    // head the index holds whole — read to its end, that head settles
    // which documents carry the term. With no such head among the kept
    // superfiles the merge could exclude nothing and would only cost a
    // round: the ceilings serve instead.
    if !shape.trusted
        && !slots.iter().any(|s| {
            s.heads
                .iter()
                .enumerate()
                .any(|(ti, h)| shape.required & (1 << ti) != 0 && h.whole && h.ceiling > 0.0)
        })
    {
        return Ok(None);
    }
    // What lies past a block is bounded through its tail, already widened
    // where the tail is set; a read document's sum needs widening only
    // when the kernel may add it in another order, by a rounding per
    // addition.
    let (lb_slack, ub_slack) = if exact_scores {
        (1.0, 1.0)
    } else {
        let slack = (8 + nt) as f32 * f32::EPSILON;
        (1.0 - slack, 1.0 + slack)
    };
    let scoring = Scoring::new(idf, query.params, query.table_avgdl);
    // Inline heads are whole: their entries are in hand before any read.
    let mut entries_read = 0u64;
    for (ti, e) in &inline {
        let Some(si) = slot_for(e.superfile) else {
            continue;
        };
        let c = scoring.contribution(*ti, e);
        slots[si].absorb(*ti, e.doc_id, c);
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
        entries_read,
    };
    let mut scratch = Scratch::default();
    for slot in merge.slots.iter_mut() {
        if slot.touched {
            slot.refresh(
                &merge.shape,
                &merge.scoring.idf,
                k,
                lb_slack,
                ub_slack,
                &mut scratch,
            );
        }
    }

    let mut resident = true;
    for source in sources.iter().flatten() {
        resident &= index
            .tier_chunks_resident(&source.object, &source.chunks)
            .await;
    }
    Ok(Some(PreparedMerge {
        index,
        pool,
        merge,
        sources,
        required: query.required,
        merge_gate: query.merge_gate,
        open_cost_ns: query.open_cost_ns,
        exact_scores,
        resident,
    }))
}

impl PreparedMerge<'_> {
    /// Read the chunked heads and merge, round by round.
    pub(crate) async fn run(self) -> Result<Option<TierSelection>, TermIndexError> {
        let PreparedMerge {
            index,
            pool,
            merge,
            mut sources,
            required,
            merge_gate,
            open_cost_ns,
            exact_scores,
            ..
        } = self;
        // Read: every term's whole tier, one wave of ranged reads, all started
        // at once. A term's chunks lie back to back, so this is a read per
        // term and segment, and the rounds then run without a storage round
        // trip between them. Every read starts before the first completes, so
        // a merge dropped early — its query's opens all finished — leaves each
        // read under way, landing in memory for the next query of the term.
        let mut tier_bytes = 0u64;
        let reads: Vec<(usize, usize)> = sources
            .iter()
            .enumerate()
            .flat_map(|(ti, term_sources)| (0..term_sources.len()).map(move |si| (ti, si)))
            .collect();
        let fetches: Vec<_> = reads
            .iter()
            .map(|&(ti, si)| {
                let source = &sources[ti][si];
                index.tier_chunks(&source.object, &source.chunks)
            })
            .collect();
        let fetched: Vec<(Vec<Arc<TierChunkData>>, u64)> = try_join_all(fetches).await?;
        for (&(ti, si), (data, bytes)) in reads.iter().zip(fetched) {
            sources[ti][si].data = data;
            tier_bytes += bytes;
        }
        // Decode, score, join and bound, round by round, as one job on the
        // reader pool: a round is the next block of every live superfile whose
        // head goes on, and the merge stops when a round moves nothing.
        let max_rounds = match crate::config::global().diagnostics.tier_rounds {
            0 => usize::MAX,
            n => n as usize,
        };
        // With the gate off the merge is under test and reads on while a round
        // excludes anything, whatever the opens were measured to cost: the
        // rounds it takes then depend on the table alone.
        let open_cost_ns = if merge_gate { open_cost_ns } else { 0 };
        let (merge, merge_ns) = run_on_pool(Some(pool), "tier merge", move || {
            let mut merge = merge;
            let mut merge_ns = 0u64;
            for round in 0..max_rounds {
                let mut reads: Vec<Vec<BlockRead<'_>>> =
                    (0..merge.slots.len()).map(|_| Vec::new()).collect();
                let mut read_any = false;
                for (slot_i, slot) in merge.slots.iter_mut().enumerate() {
                    if !slot.live {
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
                        let Some(block) = chunk.dir.block_of(slot.ordinal) else {
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
                    break;
                }
                let started = Instant::now();
                let excluded = merge.round(&reads);
                let round_ns = started.elapsed().as_nanos() as u64;
                merge_ns += round_ns;
                // Worth reading on: the opens this round saved, at what an open
                // costs here, cover what the round cost. With the open cost
                // unknown, any exclusion is taken as worth it. A round that
                // found no threshold saved nothing and is not read past: the
                // joint documents a conjunction needs for one sit deeper than
                // the heads' top blocks, and reading on for them has bought
                // nothing where it was measured.
                let worth = pays(excluded as u64, 1, open_cost_ns, round_ns);
                // A required term whose head the index holds whole settles,
                // read to its end, which documents carry the term — ruling
                // out, or answering for, every live superfile the conjunction
                // has no joint document in. Finishing it is worth the rounds
                // it still needs while the live opens at stake cover them at
                // this round's cost; with the open cost unknown it is finished
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
                let read_on = worth || settling;
                tracing::debug!(
                    round,
                    excluded,
                    live = merge.slots.iter().filter(|s| s.live).count(),
                    theta = merge.theta_lb,
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
            (merge, merge_ns)
        })
        .await
        .map_err(|e| TermIndexError::Open(QueryError::Internal(e.to_string())))?;

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
                    slot.join(&mut scratch, |row, mask, contrib| {
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
                            slot.absorb(read.ti, e.doc_id, c);
                        }
                    }
                    if slot.touched {
                        slot.refresh(shape, &scoring.idf, k, lb_slack, ub_slack, &mut scratch);
                    }
                }
                n
            })
            .sum();
        self.entries_read += entries;
        // Threshold: the k-th best sum over documents the query returns,
        // across every superfile read, in play or not. Each superfile's
        // top candidates are sorted, so a k-way walk finds it in k steps.
        let mut heap: BinaryHeap<(OrdF32, usize, usize)> = BinaryHeap::new();
        for (si, slot) in self.slots.iter().enumerate() {
            if let Some(&best) = slot.top.first() {
                heap.push((OrdF32(best), si, 0));
            }
        }
        let mut kth = None;
        for _ in 0..k {
            let Some((OrdF32(value), si, at)) = heap.pop() else {
                kth = None;
                break;
            };
            kth = Some(value);
            if let Some(&next) = self.slots[si].top.get(at + 1) {
                heap.push((OrdF32(next), si, at + 1));
            }
        }
        if let Some(kth) = kth {
            self.theta_lb = self.theta_lb.max(kth);
        }
        // Rule out what cannot compete.
        let mut excluded = 0usize;
        if self.theta_lb.is_finite() {
            let theta = self.theta_lb;
            for slot in self.slots.iter_mut() {
                if slot.live && slot.ub < theta {
                    slot.live = false;
                    excluded += 1;
                }
            }
        }
        excluded
    }
}

impl Slot {
    /// Take in one entry: document `row` read under term `ti` with
    /// contribution `c`. It waits in the term's pending list until the
    /// next refresh merges it into the term's documents.
    fn absorb(&mut self, ti: usize, row: u32, c: f32) {
        self.heads[ti].seen = true;
        if self.deny.as_ref().is_some_and(|d| d.contains(row)) {
            return;
        }
        self.heads[ti].pending.push((row, c));
        self.touched = true;
    }

    /// Visit every document read under any term, in ascending row, with
    /// the terms it was read under and its contribution per term (zero
    /// where it was not read): a k-way merge of the terms' documents.
    fn join(&self, scratch: &mut Scratch, mut visit: impl FnMut(u32, TermMask, &[f32])) {
        let nt = self.heads.len();
        scratch.at.clear();
        scratch.at.resize(nt, 0);
        scratch.contrib.clear();
        scratch.contrib.resize(nt, 0.0);
        loop {
            let mut next: Option<u32> = None;
            for (ti, h) in self.heads.iter().enumerate() {
                if let Some(&(row, _)) = h.docs.get(scratch.at[ti]) {
                    next = Some(next.map_or(row, |n| n.min(row)));
                }
            }
            let Some(row) = next else { break };
            let mut mask: TermMask = 0;
            for (ti, h) in self.heads.iter().enumerate() {
                scratch.contrib[ti] = 0.0;
                if let Some(&(r, c)) = h.docs.get(scratch.at[ti])
                    && r == row
                {
                    mask |= 1 << ti;
                    scratch.contrib[ti] = c;
                    scratch.at[ti] += 1;
                }
            }
            visit(row, mask, &scratch.contrib);
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
    fn refresh(
        &mut self,
        shape: &Shape,
        idf: &[f32],
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
        // `c` (every tail for one read under no term).
        let doc_ub = |mask: TermMask, c: &[f32]| -> f32 {
            let mut ub = 0.0f32;
            for (t, h) in heads.iter().enumerate() {
                if shape.scoring & (1 << t) != 0 {
                    ub += if mask & (1 << t) != 0 { c[t] } else { h.tail };
                }
            }
            for (members, _) in &shape.phrases {
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
            ub
        };
        scratch.zero.clear();
        scratch.zero.resize(nt, 0.0);
        let mut ub = if settled == 0 {
            doc_ub(0, &scratch.zero)
        } else {
            f32::NEG_INFINITY
        };
        let mut lb = f32::NEG_INFINITY;
        scratch.candidates.clear();
        let mut candidates = std::mem::take(&mut scratch.candidates);
        self.join(scratch, |_, mask, contrib| {
            if mask & settled != settled {
                return;
            }
            ub = ub.max(doc_ub(mask, contrib));
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
        self.ub = ub * ub_slack;
        self.lb = lb;
    }
}

/// Merge `incoming`, in ascending row, into `docs`, also ascending,
/// through `scratch`.
fn merge_sorted(
    docs: &mut Vec<(u32, f32)>,
    incoming: &[(u32, f32)],
    scratch: &mut Vec<(u32, f32)>,
) {
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
                        slot.absorb(ti, doc, c);
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
            slot.refresh(&shape, &idf, 10, 1.0, 1.0, &mut Scratch::default());
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
    /// the cheaper way and tries the other one query in sixteen.
    #[test]
    fn the_policy_measures_both_ways_before_it_chooses() {
        let policy = TierPolicy::new();
        let class = TierPolicy::class(true, false, 10, 2, 5.0);
        let mut ways = Vec::new();
        for _ in 0..(2 * TierPolicy::MIN_SAMPLES) {
            let with = policy.merge(class);
            ways.push(with);
            // Merging costs ten times what not merging costs, and prunes.
            policy.note(
                class,
                with,
                with.then_some(true),
                if with { 10_000 } else { 1_000 },
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
        };
        let idf = vec![1.0f32, 1.0];
        let mut slot = Slot::new(2, 1.0, None);
        let mut scratch = Scratch::default();
        // Term 1 is read whole at once; term 0 comes in two blocks with
        // tails 0.5 then 0.2.
        for (doc, c) in [(1u32, 0.9f32), (4, 0.7), (9, 0.6)] {
            slot.absorb(1, doc, c);
        }
        slot.heads[1].tail = 0.0;
        slot.heads[1].ceiling = 0.9;
        slot.heads[1].done = true;
        for (doc, c) in [(4u32, 0.8f32), (2, 0.6), (7, 0.55)] {
            slot.absorb(0, doc, c);
        }
        slot.heads[0].tail = 0.5;
        slot.heads[0].ceiling = 0.8;
        slot.refresh(&shape, &idf, 3, 1.0, 1.0, &mut scratch);
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
            slot.absorb(0, doc, c);
        }
        slot.heads[0].tail = 0.2;
        slot.refresh(&shape, &idf, 3, 1.0, 1.0, &mut scratch);
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
        };
        let deny: RoaringBitmap = [4u32].into_iter().collect();
        let mut slot = Slot::new(1, 1.0, Some(Arc::new(deny)));
        for (doc, c) in [(4u32, 0.8f32), (2, 0.6)] {
            slot.absorb(0, doc, c);
        }
        slot.heads[0].tail = 0.0;
        slot.heads[0].ceiling = 0.8;
        slot.heads[0].done = true;
        slot.refresh(&shape, &[1.0], 2, 1.0, 1.0, &mut Scratch::default());
        assert_eq!(slot.top, vec![0.6], "row 4 is deleted");
        assert!((slot.ub - 0.6).abs() < 1e-6);
    }
}
