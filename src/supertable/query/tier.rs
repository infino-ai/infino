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
//!
//! Scores are the kernel's own arithmetic at the table's statistics (see
//! [`Scoring`]); keys and tails stored at a superfile's own average length
//! are scaled to the table's as the ceilings are.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    mem,
    sync::Arc,
};

use bytes::Bytes;
use futures::future::try_join_all;
use rayon::{
    ThreadPool,
    iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
};
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
                BLOCK_ENTRIES, SliceRun, TermIndex, TermIndexError, TierChunkRef, TierDirectory,
                TierEntry, idf_ratio, local_idf,
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
/// Most query terms the merge serves: a candidate's terms are a bit set in
/// a byte, and the per-superfile state has one slot per term.
pub(crate) const MAX_TIER_TERMS: usize = 6;
/// Entries the merge reads per query before it stops regardless of what is
/// still in play. What has been read by then still bounds every superfile
/// soundly; the merge just stops refining.
const MAX_ENTRIES_PER_QUERY: u64 = 1 << 20;

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
/// their directories once read.
struct Source {
    object: ContentHash,
    chunks: Vec<TierChunkRef>,
    dirs: Vec<Option<Arc<TierDirectory>>>,
}

/// What the merge knows of one term in one kept superfile.
#[derive(Clone)]
struct HeadState {
    /// Documents read so far as `(row, contribution)`, in ascending row.
    docs: Vec<(u32, f32)>,
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
            docs: Vec::new(),
            tail: 0.0,
            ceiling: 0.0,
            done: true,
            source: None,
        }
    }
}

/// What the merge knows of one kept superfile, and what it derives from
/// it: recomputed only in a round that read something for the superfile.
struct Slot {
    heads: Vec<HeadState>,
    /// The heads joined: per document read under any term, the terms it
    /// was read under and its summed contributions, in ascending row.
    joined: Vec<(u32, u8, f32)>,
    /// The best sums (narrowed by the lower slack) of documents carrying
    /// every required term, descending, at most `k` of them: what the
    /// superfile offers the threshold.
    candidates: Vec<f32>,
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

/// One block read for one superfile and term in a round.
struct BlockRead {
    ti: usize,
    dir: Arc<TierDirectory>,
    block: usize,
    body: Bytes,
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
            joined: Vec::new(),
            candidates: Vec::new(),
            ub: f32::INFINITY,
            lb: f32::NEG_INFINITY,
            live: true,
            ordinal: u32::MAX,
            scale_up,
        });
    }
    let scored_docs: Vec<u64> = kept
        .iter()
        .map(|e| {
            e.fts_summary
                .get(&query.column)
                .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                .unwrap_or(e.n_docs)
        })
        .collect();

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
                    dirs: vec![None; tier.chunks.len()],
                    chunks: tier.chunks.clone(),
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
                // The stored ceiling is an impact at the superfile's own idf;
                // the ratio to the table's rescales it, as the ceiling path
                // does. The tail is a key at the superfile's own average.
                let ratio = idf_ratio(local_idf(scored_docs[si], p.df), idf_table);
                let ceiling = p.bound * ratio * slot.scale_up;
                // A build that recorded no tail leaves the ceiling as the
                // bound.
                let tail = if p.tail.is_finite() {
                    (p.tail * idf_table * slot.scale_up).min(ceiling)
                } else {
                    ceiling
                };
                slot.heads[ti] = HeadState {
                    docs: Vec::new(),
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
    let mut touched: Vec<bool> = vec![false; slots.len()];
    for (ti, e) in &inline {
        let Some(id) = index.superfile_id(e.superfile) else {
            continue;
        };
        let Some(&si) = slot_of.get(&id) else {
            continue;
        };
        slots[si].heads[*ti]
            .docs
            .push((e.doc_id, scoring.contribution(*ti, e)));
        touched[si] = true;
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
    for (si, t) in touched.iter().enumerate() {
        if *t {
            for h in merge.slots[si].heads.iter_mut() {
                h.docs.sort_unstable_by_key(|d| d.0);
            }
            merge.slots[si].refresh(query.required, k, lb_slack, ub_slack);
        }
    }

    let mut tier_bytes = 0u64;
    for round in 0.. {
        // Read: the next block of every live superfile whose head goes on.
        // Directories first, then the blocks, each wave issued together.
        let mut need_dir: Vec<(usize, usize)> = Vec::new();
        for (ti, term_sources) in sources.iter().enumerate() {
            for (si, source) in term_sources.iter().enumerate() {
                if round < source.chunks.len() && source.dirs[round].is_none() {
                    need_dir.push((ti, si));
                }
            }
        }
        let dirs = try_join_all(need_dir.iter().map(|&(ti, si)| {
            let source = &sources[ti][si];
            index.tier_directory(&source.object, &source.chunks[round])
        }))
        .await?;
        for (&(ti, si), (dir, fetched)) in need_dir.iter().zip(dirs) {
            sources[ti][si].dirs[round] = Some(dir);
            tier_bytes += fetched;
        }
        // Per (term, source): the blocks the live superfiles need.
        let mut wanted: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
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
                let Some(dir) = sources[ti][si].dirs.get(round).and_then(Option::as_ref) else {
                    state.done = true;
                    continue;
                };
                let Some(block) = dir.block_of(slot.ordinal) else {
                    state.done = true;
                    continue;
                };
                wanted.entry((ti, si)).or_default().push((slot_i, block));
            }
        }
        let groups: Vec<((usize, usize), Vec<(usize, usize)>)> = wanted.into_iter().collect();
        let bodies = try_join_all(groups.iter().map(|((ti, si), needs)| {
            let source = &sources[*ti][*si];
            let dir = source.dirs[round].as_ref().expect("read above");
            let blocks: Vec<usize> = needs.iter().map(|n| n.1).collect();
            async move {
                index
                    .tier_blocks(&source.object, &source.chunks[round], dir, &blocks)
                    .await
            }
        }))
        .await?;
        let mut reads: Vec<Vec<BlockRead>> = (0..merge.slots.len()).map(|_| Vec::new()).collect();
        let mut read_any = false;
        for (((ti, si), needs), (bodies, fetched)) in groups.iter().zip(bodies) {
            tier_bytes += fetched;
            let dir = sources[*ti][*si].dirs[round].as_ref().expect("read above");
            for (&(slot_i, block), body) in needs.iter().zip(bodies) {
                reads[slot_i].push(BlockRead {
                    ti: *ti,
                    dir: Arc::clone(dir),
                    block,
                    body,
                });
                read_any = true;
            }
        }
        if round > 0 && !read_any {
            break;
        }
        // Decode, score, join and bound, as one wave on the reader pool.
        merge = run_on_pool(Some(pool), "tier merge", move || {
            merge.round(reads);
            merge
        })
        .await
        .map_err(|e| TermIndexError::Malformed(e.to_string()))?;
        if !read_any || merge.entries_read >= MAX_ENTRIES_PER_QUERY {
            break;
        }
    }

    let required = query.required;
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
                resolved.push((
                    si,
                    slot.joined
                        .iter()
                        .filter(|(_, mask, _)| mask & required == required)
                        .map(|&(doc, _, sum)| (doc, sum))
                        .collect(),
                ));
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
    /// One round's decoding: per superfile that read a block, its heads
    /// grow and its join and bounds are recomputed, in parallel; then the
    /// threshold over every superfile's candidates, and the exclusion of
    /// what cannot reach it.
    fn round(&mut self, reads: Vec<Vec<BlockRead>>) {
        let required = self.required;
        let (k, lb_slack, ub_slack) = (self.k, self.lb_slack, self.ub_slack);
        let scoring = &self.scoring;
        let entries: u64 = self
            .slots
            .par_iter_mut()
            .zip(reads.into_par_iter())
            .map_init(Scratch::default, |scratch, (slot, reads)| {
                if reads.is_empty() {
                    return 0;
                }
                let mut n = 0u64;
                for read in reads {
                    let state = &mut slot.heads[read.ti];
                    if read
                        .dir
                        .decode_block(read.block, &read.body, &mut scratch.decoded)
                        .is_err()
                    {
                        // A block that does not decode leaves the head where
                        // it was: the bound stays its tail, which is sound,
                        // and the superfile opens as it would have.
                        state.done = true;
                        continue;
                    }
                    n += scratch.decoded.len() as u64;
                    merge_sorted(
                        &mut state.docs,
                        scratch
                            .decoded
                            .iter()
                            .map(|e| (e.doc_id, scoring.contribution(read.ti, e))),
                        &mut scratch.merged,
                    );
                    let tail_key = read.dir.block_tail(read.block);
                    state.tail =
                        (tail_key * scoring.idf[read.ti] * slot.scale_up).min(state.ceiling);
                    if tail_key <= 0.0 || scratch.decoded.len() < BLOCK_ENTRIES {
                        state.done = true;
                    }
                }
                slot.refresh(required, k, lb_slack, ub_slack);
                n
            })
            .sum();
        self.entries_read += entries;
        // Threshold: the k-th best sum over documents the query returns,
        // across every superfile read, in play or not. Each superfile's
        // candidates are sorted, so a k-way walk finds it in k steps.
        let mut heap: BinaryHeap<(OrdF32, usize, usize)> = BinaryHeap::new();
        for (si, slot) in self.slots.iter().enumerate() {
            if let Some(&best) = slot.candidates.first() {
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
            if let Some(&next) = self.slots[si].candidates.get(at + 1) {
                heap.push((OrdF32(next), si, at + 1));
            }
        }
        if let Some(kth) = kth {
            self.theta_lb = self.theta_lb.max(kth);
        }
        // Rule out what cannot compete.
        if self.theta_lb.is_finite() {
            let theta = self.theta_lb;
            for slot in self.slots.iter_mut() {
                if slot.live && slot.ub * BOUND_SLACK < theta {
                    slot.live = false;
                }
            }
        }
    }
}

impl Slot {
    /// Recompute the join, the candidates and the bounds from the heads.
    fn refresh(&mut self, required: u8, k: usize, lb_slack: f32, ub_slack: f32) {
        join(&self.heads, &mut self.joined);
        self.candidates.clear();
        for &(_, mask, sum) in &self.joined {
            if mask & required == required {
                self.candidates.push(sum * lb_slack);
            }
        }
        self.candidates.sort_unstable_by(|a, b| b.total_cmp(a));
        self.candidates.truncate(k);
        let (ub, lb) = bounds_of(&self.heads, &self.joined, required, ub_slack);
        self.ub = ub;
        self.lb = lb;
    }
}

/// Per-task buffers for a round's decoding.
#[derive(Default)]
struct Scratch {
    decoded: Vec<TierEntry>,
    merged: Vec<(u32, f32)>,
}

/// Merge `incoming`, in ascending row, into `docs`, also ascending, through
/// `scratch`.
fn merge_sorted(
    docs: &mut Vec<(u32, f32)>,
    incoming: impl Iterator<Item = (u32, f32)>,
    scratch: &mut Vec<(u32, f32)>,
) {
    scratch.clear();
    let mut at = 0usize;
    for item in incoming {
        while at < docs.len() && docs[at].0 < item.0 {
            scratch.push(docs[at]);
            at += 1;
        }
        scratch.push(item);
    }
    scratch.extend_from_slice(&docs[at..]);
    mem::swap(docs, scratch);
}

/// Join one superfile's heads across the terms: per document read under
/// any term, the terms it was read under and its summed contributions, in
/// ascending row.
fn join(heads: &[HeadState], out: &mut Vec<(u32, u8, f32)>) {
    out.clear();
    let mut at = [0usize; MAX_TIER_TERMS];
    loop {
        let mut next: Option<u32> = None;
        for (ti, h) in heads.iter().enumerate() {
            if let Some(&(doc, _)) = h.docs.get(at[ti]) {
                next = Some(next.map_or(doc, |n: u32| n.min(doc)));
            }
        }
        let Some(doc) = next else { break };
        let mut mask = 0u8;
        let mut sum = 0.0f32;
        for (ti, h) in heads.iter().enumerate() {
            if let Some(&(d, c)) = h.docs.get(at[ti])
                && d == doc
            {
                mask |= 1 << ti;
                sum += c;
                at[ti] += 1;
            }
        }
        out.push((doc, mask, sum));
    }
}

/// `(upper bound, lower bound)` on the score of a document of one
/// superfile, from what has been read of it: a document read under some
/// terms scores at most its known sum plus the tails of the terms it has
/// not been read under; one read under none at most the sum of the tails.
/// A superfile that lacks a required term holds no match at all. The lower
/// bound is the best known sum of a document carrying every required term.
fn bounds_of(
    heads: &[HeadState],
    joined: &[(u32, u8, f32)],
    required: u8,
    ub_slack: f32,
) -> (f32, f32) {
    let mut tails = [0.0f32; MAX_TIER_TERMS];
    for (ti, h) in heads.iter().enumerate() {
        if required & (1 << ti) != 0 && h.ceiling <= 0.0 && h.docs.is_empty() {
            return (f32::NEG_INFINITY, f32::NEG_INFINITY);
        }
        tails[ti] = h.tail;
    }
    let mut ub: f32 = tails[..heads.len()].iter().sum();
    let mut lb = f32::NEG_INFINITY;
    for &(_, mask, sum) in joined {
        let missing: f32 = (0..heads.len())
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

    #[test]
    fn merge_sorted_keeps_rows_ascending() {
        let mut docs = vec![(1, 0.5), (4, 0.25), (9, 0.125)];
        let mut scratch = Vec::new();
        merge_sorted(
            &mut docs,
            [(0, 1.0), (4, 0.75), (10, 0.5)].into_iter(),
            &mut scratch,
        );
        let rows: Vec<u32> = docs.iter().map(|d| d.0).collect();
        assert_eq!(rows, [0, 1, 4, 4, 9, 10]);
    }
}
