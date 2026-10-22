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
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use futures::future::try_join_all;
use rayon::{
    ThreadPool,
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::{ParallelSlice, ParallelSliceMut},
};
use rustc_hash::FxHashMap;
use uuid::Uuid;

use super::fts::baked_avgdl;
use crate::{
    runtime_bridge::run_on_pool,
    superfile::fts::{
        bm25::{Bm25Params, dequantize_len, score},
        reader::GlobalTermIdf,
    },
    supertable::{
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
/// Widening applied to an upper bound before it is compared against the
/// threshold: the bound is assembled from separately rounded terms the
/// score itself does not go through. Mirrors the term index's ceiling
/// slack.
const BOUND_SLACK: f32 = 1.0 + 8.0 * f32::EPSILON;
/// Narrowing applied to a lower bound as it is assembled: a stored impact
/// times the idf ratio can land an ulp above the score the kernel computes
/// for the same document, and a threshold an ulp above a real k-th score
/// would prune a document tied with it. The mirror of [`BOUND_SLACK`].
const LOWER_SLACK: f32 = 1.0 - 8.0 * f32::EPSILON;
/// Most query terms the merge serves. One term is one product and two
/// commute, so a candidate's sum is the kernel's score bit for bit and the
/// bounds are tight; past two the bounds carry slack and the sum of tails
/// grows with the terms, so the merge rarely rules a superfile out and
/// costs every round for nothing — such queries keep the ceilings.
pub(crate) const MAX_TIER_TERMS: usize = 2;
/// Entries the merge reads per query before it stops regardless of what is
/// still in play. What has been read by then still bounds every superfile
/// soundly; the merge just stops refining.
const MAX_ENTRIES_PER_QUERY: u64 = 1 << 20;
/// Superfiles one pool job decodes in a round. A superfile's share of a
/// round is a few microseconds, so jobs this size keep the pool's
/// scheduling a small fraction of the work.
const SLOTS_PER_JOB: usize = 16;

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
    pub(crate) terms: &'a [&'a str],
    /// A bit per term (in `terms` order) a matching document must carry:
    /// every bit for a conjunction, none for a disjunction, the musts' for
    /// a mix.
    pub(crate) required: u8,
    pub(crate) k: usize,
    /// The average length every kept superfile scores at.
    pub(crate) table_avgdl: f32,
    /// The column's declared parameters, which every kept superfile scores
    /// at.
    pub(crate) params: Bm25Params,
    /// Whether the caller can use documents the index settles outright
    /// (it cannot when a scope admits rows the index does not see).
    pub(crate) want_resolved: bool,
    /// What one superfile open costs on this table, in nanoseconds of
    /// query wall; zero when not yet known. A round of the merge is read
    /// while the opens the last round saved are worth more than the round
    /// took — the trade that differs between a table whose opens are a
    /// few microseconds of warm memory and one whose opens are reads.
    pub(crate) open_cost_ns: u64,
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
}

/// One term's chunked tier in one segment: where its regions sit, and
/// the regions once read.
struct Source {
    object: ContentHash,
    chunks: Vec<TierChunkRef>,
    data: Vec<TierChunkData>,
}

/// What the merge knows of one term in one kept superfile.
#[derive(Clone)]
struct HeadState {
    /// Whether any document of the term was read for the superfile.
    seen: bool,
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
            tail: 0.0,
            ceiling: 0.0,
            done: true,
            source: None,
        }
    }
}

/// One document as the merge has read it: the terms it was read under,
/// its summed contributions, and its place in the superfile's candidates
/// (`NOT_A_CANDIDATE` until it carries every required term).
#[derive(Clone, Copy)]
struct Joined {
    mask: u8,
    sum: f32,
    candidate: u32,
}

const NOT_A_CANDIDATE: u32 = u32::MAX;

/// What the merge knows of one kept superfile, and what it derives from
/// it. Every entry read updates the join, the best sum of its mask and
/// the candidates in place; the bounds and the top candidates are
/// recomputed only in a round that read something for the superfile.
struct Slot {
    heads: Vec<HeadState>,
    /// The heads joined, by row.
    joined: FxHashMap<u32, Joined>,
    /// Per set of terms, the best sum of a document read under exactly
    /// those terms at the time (`NEG_INFINITY` when none): what the upper
    /// bound is assembled from.
    best: [f32; 1 << MAX_TIER_TERMS],
    /// The sums (narrowed by the lower slack) of documents carrying every
    /// required term, in no order, updated in place as a document's sum
    /// grows.
    candidates: Vec<f32>,
    /// The best `k` of `candidates`, descending: what the superfile offers
    /// the threshold.
    top: Vec<f32>,
    /// Whether a round read something for the superfile since its bounds
    /// and `top` were last computed.
    touched: bool,
    /// Upper bound on any of the superfile's documents, lower bound on its
    /// best candidate (see [`bounds_of`]).
    ub: f32,
    lb: f32,
    /// Whether the merge still reads for the superfile.
    live: bool,
    /// The ordinal the superfile's heads are filed under in the index.
    ordinal: u32,
    /// The ratio the superfile's stored keys and tails are scaled by to
    /// hold at the table's average length.
    scale_up: f32,
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
    required: u8,
    k: usize,
    lb_slack: f32,
    ub_slack: f32,
    theta_lb: f32,
    entries_read: u64,
}

/// Select superfiles for `query` with the tiers of `index`, decoding on
/// `pool`. `None` when a term has postings in a segment without a tier, a
/// kept superfile is not in the root, or a table-wide idf is missing: the
/// caller then keeps its ceilings.
pub(crate) async fn select_with_tiers(
    index: &TermIndex,
    query: &TierQuery<'_>,
    kept: &[Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
    pool: &Arc<ThreadPool>,
) -> Result<Option<TierSelection>, TermIndexError> {
    let terms = query.terms;
    let nt = terms.len();
    let k = query.k;
    if nt == 0 || nt > MAX_TIER_TERMS || k == 0 || kept.is_empty() {
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
        slots.push(Slot {
            heads: vec![HeadState::absent(); nt],
            joined: FxHashMap::default(),
            best: [f32::NEG_INFINITY; 1 << MAX_TIER_TERMS],
            candidates: Vec::new(),
            top: Vec::new(),
            touched: false,
            ub: f32::INFINITY,
            lb: f32::NEG_INFINITY,
            live: true,
            ordinal: u32::MAX,
            scale_up,
        });
    }
    let runs_per_term =
        try_join_all(terms.iter().map(|term| index.runs(query.column, term))).await?;
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
                let Some(id) = index.superfile_id(p.superfile) else {
                    continue;
                };
                let Some(&si) = slot_of.get(&id) else {
                    continue;
                };
                let slot = &mut slots[si];
                slot.ordinal = p.superfile;
                // The stored bound and tail are keys at the superfile's own
                // average; the table's idf and the scale to its average make
                // them bounds on the term's contribution, as the ceiling
                // path makes its ceilings.
                let ceiling = p.bound * idf_table * slot.scale_up;
                // A build that recorded no tail leaves the ceiling as the
                // bound.
                let tail = if p.tail.is_finite() {
                    (p.tail * idf_table * slot.scale_up).min(ceiling)
                } else {
                    ceiling
                };
                slot.heads[ti] = HeadState {
                    seen: false,
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
    let exact_scores = nt <= EXACT_SUM_TERMS;
    let (lb_slack, ub_slack) = if exact_scores {
        (1.0, 1.0)
    } else {
        (LOWER_SLACK, BOUND_SLACK)
    };
    let scoring = Scoring {
        idf,
        params: query.params,
        avgdl: query.table_avgdl,
    };
    // Inline heads are whole: their entries are in hand before any read.
    let mut entries_read = 0u64;
    for (ti, e) in &inline {
        let Some(id) = index.superfile_id(e.superfile) else {
            continue;
        };
        let Some(&si) = slot_of.get(&id) else {
            continue;
        };
        let c = scoring.contribution(*ti, e);
        slots[si].absorb(*ti, e.doc_id, c, query.required, lb_slack);
        entries_read += 1;
    }
    let mut merge = Merge {
        slots,
        scoring,
        required: query.required,
        k,
        lb_slack,
        ub_slack,
        theta_lb: f32::NEG_INFINITY,
        entries_read,
    };
    for slot in merge.slots.iter_mut() {
        if slot.touched {
            slot.refresh(query.required, k, ub_slack);
        }
    }

    // Read: every term's whole tier, one wave of ranged reads. A term's
    // chunks lie back to back, so this is a read per term in the common
    // case, and the rounds then run without a storage round trip between
    // them.
    let mut tier_bytes = 0u64;
    let reads: Vec<(usize, usize)> = sources
        .iter()
        .enumerate()
        .flat_map(|(ti, term_sources)| (0..term_sources.len()).map(move |si| (ti, si)))
        .collect();
    let fetched = try_join_all(reads.iter().map(|&(ti, si)| {
        let source = &sources[ti][si];
        index.tier_chunks(&source.object, &source.chunks)
    }))
    .await?;
    for (&(ti, si), (data, bytes)) in reads.iter().zip(fetched) {
        sources[ti][si].data = data;
        tier_bytes += bytes;
    }
    // Decode, score, join and bound, round by round, as one job on the
    // reader pool: a round is the next block of every live superfile whose
    // head goes on, and the merge stops when a round moves nothing.
    let required = query.required;
    let max_rounds = match crate::config::global().diagnostics.tier_rounds {
        0 => usize::MAX,
        n => n as usize,
    };
    let open_cost_ns = query.open_cost_ns;
    let merge = run_on_pool(Some(pool), "tier merge", move || {
        let mut merge = merge;
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
            // Worth reading on: the opens this round saved, at what an open
            // costs here, cover what the round cost — or no threshold exists
            // yet, so nothing could have been saved and the next round may
            // be the one that finds it. With the open cost unknown, any
            // exclusion is taken as worth it.
            let worth = if open_cost_ns == 0 {
                excluded > 0
            } else {
                u128::from(excluded as u64) * u128::from(open_cost_ns) >= u128::from(round_ns)
            };
            let read_on = !merge.theta_lb.is_finite() || worth;
            tracing::debug!(
                round,
                excluded,
                live = merge.slots.iter().filter(|s| s.live).count(),
                theta = merge.theta_lb,
                entries = merge.entries_read,
                round_us = round_ns / 1000,
                open_cost_us = open_cost_ns / 1000,
                read_on,
                "tier merge round"
            );
            if !read_on || merge.entries_read >= MAX_ENTRIES_PER_QUERY {
                break;
            }
        }
        merge
    })
    .await
    .map_err(|e| TermIndexError::Malformed(e.to_string()))?;

    let bounds: Vec<SuperfileBounds> = merge
        .slots
        .iter()
        .map(|s| SuperfileBounds {
            lb: s.lb * lb_slack,
            ub: if s.live { s.ub } else { f32::NEG_INFINITY },
        })
        .collect();
    let mut resolved = Vec::new();
    if query.want_resolved && exact_scores {
        for (si, slot) in merge.slots.iter().enumerate() {
            // Every head read to its end, and the sums exact: the superfile's
            // matching documents are its candidates carrying the required
            // terms, scored as the kernel would score them. One the merge
            // ruled out holds none that place.
            let whole = slot.live && slot.heads.iter().all(|h| h.done && h.tail <= 0.0);
            if whole {
                let mut docs: Vec<(u32, f32)> = slot
                    .joined
                    .iter()
                    .filter(|(_, j)| j.mask & required == required)
                    .map(|(&doc, j)| (doc, j.sum))
                    .collect();
                docs.sort_unstable_by_key(|d| d.0);
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
    }))
}

impl Merge {
    /// One round's decoding: per superfile that read a block, its join,
    /// best sums and candidates grow entry by entry, and its bounds and
    /// top candidates are recomputed, in jobs of [`SLOTS_PER_JOB`]
    /// superfiles on the pool; then the threshold over every superfile's
    /// candidates, and the exclusion of what cannot reach it. Returns how
    /// many superfiles the round ruled out.
    fn round(&mut self, reads: &[Vec<BlockRead<'_>>]) -> usize {
        let required = self.required;
        let (k, lb_slack, ub_slack) = (self.k, self.lb_slack, self.ub_slack);
        let scoring = &self.scoring;
        let entries: u64 = self
            .slots
            .par_chunks_mut(SLOTS_PER_JOB)
            .zip(reads.par_chunks(SLOTS_PER_JOB))
            .map(|(slots, reads)| {
                let mut decoded: Vec<TierEntry> = Vec::new();
                let mut n = 0u64;
                for (slot, reads) in slots.iter_mut().zip(reads) {
                    for read in reads {
                        let state = &mut slot.heads[read.ti];
                        if read
                            .dir
                            .decode_block(read.block, read.body, &mut decoded)
                            .is_err()
                        {
                            // A block that does not decode leaves the head
                            // where it was: the bound stays its tail, which
                            // is sound, and the superfile opens as it would
                            // have.
                            state.done = true;
                            continue;
                        }
                        n += decoded.len() as u64;
                        let tail_key = read.dir.block_tail(read.block);
                        state.tail =
                            (tail_key * scoring.idf[read.ti] * slot.scale_up).min(state.ceiling);
                        if tail_key <= 0.0 || decoded.len() < BLOCK_ENTRIES {
                            state.done = true;
                        }
                        for e in &decoded {
                            let c = scoring.contribution(read.ti, e);
                            slot.absorb(read.ti, e.doc_id, c, required, lb_slack);
                        }
                    }
                    if slot.touched {
                        slot.refresh(required, k, ub_slack);
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
                if slot.live && slot.ub * BOUND_SLACK < theta {
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
    /// contribution `c`. Its join grows by the term, the best sum of its
    /// new mask takes it in, and it enters the candidates — or its
    /// candidate grows — once it carries every required term.
    fn absorb(&mut self, ti: usize, row: u32, c: f32, required: u8, lb_slack: f32) {
        let bit = 1u8 << ti;
        self.heads[ti].seen = true;
        self.touched = true;
        let j = self.joined.entry(row).or_insert(Joined {
            mask: 0,
            sum: 0.0,
            candidate: NOT_A_CANDIDATE,
        });
        if j.mask & bit != 0 {
            return;
        }
        j.mask |= bit;
        j.sum += c;
        let m = j.mask as usize;
        self.best[m] = self.best[m].max(j.sum);
        if j.mask & required == required {
            let v = j.sum * lb_slack;
            if j.candidate == NOT_A_CANDIDATE {
                j.candidate = self.candidates.len() as u32;
                self.candidates.push(v);
            } else {
                self.candidates[j.candidate as usize] = v;
            }
        }
    }

    /// Recompute the bounds and the top candidates from what was absorbed.
    fn refresh(&mut self, required: u8, k: usize, ub_slack: f32) {
        self.touched = false;
        self.top.clear();
        self.top.extend_from_slice(&self.candidates);
        if self.top.len() > k {
            self.top.select_nth_unstable_by(k, |a, b| b.total_cmp(a));
            self.top.truncate(k);
        }
        self.top.sort_unstable_by(|a, b| b.total_cmp(a));
        let (ub, lb) = bounds_of(&self.heads, &self.best, required, ub_slack);
        self.ub = ub;
        self.lb = lb;
    }
}

/// `(upper bound, lower bound)` on the score of a document of one
/// superfile, from what has been read of it: a document read under some
/// terms scores at most its known sum plus the tails of the terms it has
/// not been read under; one read under none at most the sum of the tails.
/// A required term's head read to its end holds every document of the
/// term there, so a document outside it carries no such term and is no
/// candidate; a superfile that lacks a required term holds no match at
/// all. The lower bound is the best known sum of a document carrying every
/// required term.
///
/// `best` holds, per mask, the best sum of a document while it was read
/// under exactly that mask. A document since read under more terms is
/// also in a larger mask, where its bound is tighter; what stays in the
/// smaller mask bounds it from above still, since every contribution it
/// gained is at least the tail of its term.
fn bounds_of(
    heads: &[HeadState],
    best: &[f32; 1 << MAX_TIER_TERMS],
    required: u8,
    ub_slack: f32,
) -> (f32, f32) {
    let nt = heads.len();
    let mut tails = [0.0f32; MAX_TIER_TERMS];
    // Required terms whose heads are whole: a candidate must be in them.
    let mut settled = 0u8;
    for (ti, h) in heads.iter().enumerate() {
        if required & (1 << ti) != 0 {
            if h.ceiling <= 0.0 && !h.seen {
                return (f32::NEG_INFINITY, f32::NEG_INFINITY);
            }
            if h.tail <= 0.0 {
                settled |= 1 << ti;
            }
        }
        tails[ti] = h.tail;
    }
    let mut ub = if settled == 0 {
        tails[..nt].iter().sum()
    } else {
        f32::NEG_INFINITY
    };
    let mut lb = f32::NEG_INFINITY;
    for (m, &sum) in best.iter().enumerate().take(1 << nt) {
        let mask = m as u8;
        if sum == f32::NEG_INFINITY || mask & settled != settled {
            continue;
        }
        let missing: f32 = (0..nt)
            .filter(|t| mask & (1 << t) == 0)
            .map(|t| tails[t])
            .sum();
        ub = ub.max(sum * ub_slack + missing);
        if mask & required == required {
            lb = lb.max(sum);
        }
    }
    (ub, lb)
}

/// How a tier entry's contribution to a query is computed: the kernel's
/// arithmetic at the table's statistics.
struct Scoring {
    /// Per term, the table-wide idf the kernel weights it with.
    idf: Vec<f32>,
    params: Bm25Params,
    /// The average length every kept superfile scores at.
    avgdl: f32,
}

impl Scoring {
    /// What term `ti` contributes to the score of a document of `e.tf`
    /// occurrences and length bucket `e.len`: the kernel's own expression.
    fn contribution(&self, ti: usize, e: &TierEntry) -> f32 {
        score(
            self.idf[ti],
            e.tf,
            dequantize_len(e.len),
            self.avgdl,
            self.params,
        )
    }
}

/// Total-order wrapper for the threshold heap.
#[derive(PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> Ordering {
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
    /// Most terms the oracle joins: what the merge serves.
    const TERMS: usize = MAX_TIER_TERMS;

    /// The bounds hold against every corpus the read heads could have come
    /// from: documents outside a term's head carry at most that term's tail
    /// (nothing when the head was read to its end), and the true best
    /// candidate's score sits between the lower and the upper bound.
    #[test]
    fn bounds_hold_against_every_corpus_the_heads_allow() {
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for case in 0..BOUND_CASES {
            let nt = 1 + (next() % TERMS as u64) as usize;
            let required = (next() % (1u64 << nt)) as u8;
            // Per term: a head of documents with contributions, and a tail
            // (zero means the head is whole); every document outside the
            // head takes a hidden contribution within the tail.
            let mut slot = Slot {
                heads: vec![HeadState::absent(); nt],
                joined: FxHashMap::default(),
                best: [f32::NEG_INFINITY; 1 << MAX_TIER_TERMS],
                candidates: Vec::new(),
                top: Vec::new(),
                touched: false,
                ub: f32::INFINITY,
                lb: f32::NEG_INFINITY,
                live: true,
                ordinal: 0,
                scale_up: 1.0,
            };
            let mut truth: Vec<Vec<f32>> = Vec::with_capacity(nt);
            for ti in 0..nt {
                let whole = next() % 3 == 0;
                let in_head = (next() % u64::from(UNIVERSE)) as u32;
                let tail = if whole {
                    0.0
                } else {
                    (next() % 1000) as f32 / 1000.0
                };
                let mut contribution = vec![0.0f32; UNIVERSE as usize];
                let mut ceiling = tail;
                for doc in 0..UNIVERSE {
                    if next() % u64::from(UNIVERSE) < u64::from(in_head) {
                        // A head entry contributes at least the tail: the
                        // head is the term's best documents.
                        let c = tail + (next() % 1000) as f32 / 1000.0;
                        slot.absorb(ti, doc, c, required, 1.0);
                        contribution[doc as usize] = c;
                        ceiling = ceiling.max(c);
                    } else if !whole && next() % 2 == 0 {
                        contribution[doc as usize] = tail * (next() % 1000) as f32 / 1000.0;
                    }
                }
                slot.heads[ti].tail = tail;
                slot.heads[ti].ceiling = ceiling;
                slot.heads[ti].done = whole;
                truth.push(contribution);
            }
            slot.refresh(required, 10, 1.0);
            let (ub, lb) = (slot.ub, slot.lb);
            // The truth: the best document carrying every required term.
            let mut best = f32::NEG_INFINITY;
            for doc in 0..UNIVERSE as usize {
                let contributions = truth.iter().map(|t| t[doc]);
                let carries_required = contributions
                    .clone()
                    .enumerate()
                    .all(|(t, c)| required & (1 << t) == 0 || c > 0.0);
                if carries_required {
                    best = best.max(contributions.sum());
                }
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
}
