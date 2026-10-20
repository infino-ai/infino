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
//! terms, and the tier carries exactly that: each term's best documents
//! table-wide, by impact, with the superfile and row of each.
//!
//! The merge reads the query terms' tiers in descending impact, a block
//! at a time, and keeps per superfile and per set of terms seen (a
//! *mask*) the best partial score of any document. Two bounds fall out:
//!
//! - `θ_lb`, a lower bound on the final k-th score — the k-th best score
//!   among documents every term has reported, whose scores are therefore
//!   final.
//! - `ub(s)`, an upper bound on any document of superfile `s`: the best
//!   over masks of a known partial score plus, for each term the mask
//!   lacks, the *frontier* — the largest impact that term can still
//!   contribute in `s`, which is the next unread block's maximum while the
//!   tier is being read and the superfile's stored tail once it is not.
//!
//! A superfile whose `ub(s)` falls below `θ_lb` holds no document that can
//! place in the top k and is never opened. The walk stops when reading on
//! could exclude nothing more: every surviving superfile's bound stays at
//! or above `θ_lb` even with its frontiers lowered all the way to its
//! tails. The survivors open in order of their best known candidate, with
//! `θ_lb` seeded as the shared floor, so even the first open prunes from
//! its first block.
//!
//! Scores are rescaled from each superfile's own statistics to the
//! query's, as the ceilings are: by the ratio of the table idf to the
//! superfile's, and by the average-length factor — up for an upper bound,
//! down for a lower one — so a bound stays a bound at the table's average.

use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
};

use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::{
    superfile::fts::{
        bm25::{Bm25Params, dequantize_len, score_with_dl_norm_k1},
        reader::GlobalTermIdf,
    },
    supertable::{
        manifest::{
            SuperfileEntry,
            part::ContentHash,
            term_index::{
                BLOCK_ENTRIES, Posting, SliceRun, TermIndex, TermIndexError, TierChunkRef,
                TierDirectory, TierEntry,
            },
        },
        schema::FieldId,
    },
};

/// Blocks per term a merge reads before it gives up on a threshold: a
/// conjunction whose terms' best documents are not each other's has found
/// fewer than `k` joint candidates by then, and reading on to the cap costs
/// more than the opens it might save. The bounds read so far stay sound
/// and order the opens; nothing is excluded without a threshold.
const THRESHOLD_PROBE_BLOCKS_PER_TERM: usize = 4;
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
/// Most query terms the merge tracks masks for: masks are a bit set, and
/// the per-superfile table has one slot per mask.
const MAX_TIER_TERMS: usize = 6;
/// Entries the merge reads per query before it stops regardless of the
/// fixpoint test. On long text the test converges within a few blocks per
/// term; on short text nearly every candidate ties, every unread block can
/// still hold a tie, and the test never converges — on a titles table it
/// read whole tiers of a hundred thousand entries per term and took a
/// hundred milliseconds. What has been read by then still bounds every
/// superfile soundly; the merge just stops refining.
const MAX_ENTRIES_PER_QUERY: u64 = 1 << 17;
/// Entries the merge reads when the candidates' scores are not the
/// query's (a phrase, a negation): there is no threshold to converge on,
/// so it reads enough to rank the superfiles and exhaust small tiers.
const INEXACT_ENTRIES_PER_QUERY: u64 = 1 << 16;
/// Rounds of the merge — one block each — that may pass without any
/// frontier falling, any candidate appearing or any superfile ruled out
/// before the merge stops: on short text the impacts tie across whole
/// tiers, and once a round changes nothing, reading on can only add ties.
const STALE_ROUNDS_BEFORE_STOP: u32 = 2;

/// What the merge decided for one kept superfile.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SuperfileBounds {
    /// Lower bound on the best score a document of this superfile reaches
    /// (`NEG_INFINITY` when no candidate was seen). Orders the opens.
    pub(crate) lb: f32,
    /// Upper bound on the score of any document of this superfile.
    pub(crate) ub: f32,
}

/// The merge's outcome for one query.
#[derive(Debug)]
pub(crate) struct TierSelection {
    /// Bounds per kept superfile, keyed by superfile id.
    pub(crate) bounds: HashMap<Uuid, SuperfileBounds>,
    /// Lower bound on the final k-th score; `NEG_INFINITY` when fewer than
    /// `k` documents were seen with every term.
    pub(crate) theta_lb: f32,
    /// Per superfile the index answers for outright — every term's tier
    /// holds the term's whole list and the merge read all of it, with
    /// scores the kernel would compute bit for bit — its matching
    /// documents as `(row, score)`. Such a superfile need not be opened;
    /// tombstones still apply. Empty unless the scores are exact.
    pub(crate) resolved: HashMap<Uuid, Vec<(u32, f32)>>,
    /// Tier bytes consumed: directories and block bodies.
    pub(crate) tier_bytes: u64,
    /// Tier entries decoded.
    pub(crate) entries_read: u64,
}

/// One tier being read: its chunks in order and where the read stands.
struct Stream {
    term: usize,
    /// The tier object the chunks' offsets are relative to.
    object: ContentHash,
    chunks: Vec<TierChunkRef>,
    chunk: usize,
    block: usize,
    /// The current chunk's directory, once read.
    dir: Option<Arc<TierDirectory>>,
    /// Largest scale any kept superfile applies to this term's stored
    /// impacts, so a block maximum bounds every superfile's unread entries.
    scale_up: f32,
}

/// Per (term, kept superfile): what the superfile can still contribute
/// beyond the tier, and how a stored key scales into a bound there.
#[derive(Clone, Copy, Default)]
struct TermInFile {
    /// Multiplies a stored key — at the superfile's own average length —
    /// into an upper bound on a score at the table's.
    scale_ub: f32,
    /// Upper bound, at the table's stats, on a document the tier does not
    /// hold.
    tail: f32,
}

/// Select superfiles for a query over `terms` with the tiers of `index`.
/// `required` is a bit per term (in `terms` order) a matching document must
/// carry: every bit for a conjunction, none for a disjunction, the musts'
/// for a mix. `exact` says a candidate's summed contributions are the score
/// the query would give it; when they are not — a phrase scores only where
/// its words are adjacent, a negation removes documents — the merge takes
/// no threshold from them and rules out only what a required term has shown
/// to hold nothing. `params` are the column's declared parameters, which
/// every kept superfile scores at. `None` when a term has postings in a
/// segment without a
/// tier, or there are more terms than the merge tracks: the caller then
/// orders by ceiling as before.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn select_with_tiers(
    index: &TermIndex,
    column: FieldId,
    terms: &[&str],
    required: u8,
    exact: bool,
    kept: &[Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
    table_avgdl: f32,
    params: Bm25Params,
    k: usize,
) -> Result<Option<TierSelection>, TermIndexError> {
    let nt = terms.len();
    if nt == 0 || nt > MAX_TIER_TERMS || k == 0 || kept.is_empty() {
        return Ok(None);
    }
    // A superfile the root does not list has postings the merge would never
    // see; bounding it at nothing would be wrong, so no selection is made.
    let listed: HashSet<Uuid> = index.root().superfiles.iter().copied().collect();
    if kept.iter().any(|e| !listed.contains(&e.superfile_id)) {
        return Ok(None);
    }
    // Kept superfiles by ordinal: what an entry's ordinal resolves to, and
    // the per-superfile statistics the rescale needs.
    let mut slot_of: HashMap<Uuid, usize> = HashMap::with_capacity(kept.len());
    for (i, e) in kept.iter().enumerate() {
        slot_of.insert(e.superfile_id, i);
    }
    // Every kept superfile scores at the table's average length (its norm
    // table is rebuilt there when it records the average it was built at),
    // so a candidate's contribution is computed exactly as its kernel
    // would. A superfile that records no average is scored at an average
    // the index does not know, so no selection is made.
    let mut scale_up: Vec<f32> = Vec::with_capacity(kept.len());
    for e in kept {
        let Some(baked) = baked_avgdl_of(e, column) else {
            return Ok(None);
        };
        // A superfile built at a smaller average scores at most
        // `table/baked` higher at the table's: the length norm falls by at
        // most that ratio, and `tf / (tf + norm)` rises by no more than the
        // norm falls.
        scale_up.push(if baked < table_avgdl {
            (table_avgdl / baked).next_up()
        } else {
            1.0
        });
    }

    let n_sf = kept.len();
    let mut per_term: Vec<Vec<TermInFile>> = vec![vec![TermInFile::default(); n_sf]; nt];
    let mut idf: Vec<f32> = Vec::with_capacity(nt);
    // Per term, whether every segment's tier holds the term's whole list.
    let mut complete: Vec<bool> = vec![true; nt];
    let mut streams: Vec<Stream> = Vec::new();
    // Entries of tiers small enough to ride in their runs, with their term.
    let mut inline: Vec<(usize, TierEntry)> = Vec::new();
    for (ti, term) in terms.iter().enumerate() {
        let Some(&idf_table) = global_idf.get(*term) else {
            return Ok(None);
        };
        idf.push(idf_table);
        let runs = index.runs(column, term).await?;
        let mut scale_up_max = 0.0f32;
        for SliceRun { tier_object, run } in runs.iter() {
            let tier = match &run.tier {
                Some(t) => t,
                // Postings without a tier: this segment predates tiers or
                // kept none for the term; no sound frontier exists for it.
                None if run.postings.is_empty() => continue,
                None => return Ok(None),
            };
            complete[ti] &= tier.complete;
            for p in &run.postings {
                let Some(id) = index.superfile_id(p.superfile) else {
                    continue;
                };
                let Some(&slot) = slot_of.get(&id) else {
                    continue;
                };
                let entry = TermInFile {
                    scale_ub: idf_table * scale_up[slot],
                    tail: tail_of(p) * idf_table * scale_up[slot],
                };
                scale_up_max = scale_up_max.max(entry.scale_ub);
                per_term[ti][slot] = entry;
            }
            if tier.chunks.is_empty() {
                inline.extend(tier.inline.iter().map(|e| (ti, *e)));
                continue;
            }
            let Some(object) = *tier_object else {
                return Ok(None);
            };
            streams.push(Stream {
                term: ti,
                object,
                chunks: tier.chunks.clone(),
                chunk: 0,
                block: 0,
                dir: None,
                scale_up: scale_up_max,
            });
        }
        // Every stream of a term bounds with the term's largest scale.
        for s in streams.iter_mut().filter(|s| s.term == ti) {
            s.scale_up = scale_up_max;
        }
    }

    let n_masks = 1usize << nt;
    // A contribution is the kernel's own arithmetic, so a one- or two-term
    // sum is the score bit for bit (addition commutes); a longer sum can
    // land an ulp from the kernel's, which adds in its own order, so its
    // bounds take the slack a product of roundings needs.
    let exact_scores = exact && nt <= EXACT_SUM_TERMS;
    let (lb_slack, ub_slack) = if exact_scores {
        (1.0, 1.0)
    } else {
        (LOWER_SLACK, BOUND_SLACK)
    };
    let scoring = Scoring {
        idf,
        params,
        avgdl: table_avgdl,
    };
    let mut cands = Candidates {
        acc: FxHashMap::default(),
        best_lb: vec![f32::NEG_INFINITY; n_sf * n_masks],
        best_ub: vec![f32::NEG_INFINITY; n_sf * n_masks],
        topk: BinaryHeap::with_capacity(k + 1),
        n_masks,
        required,
        k,
        exact,
        lb_slack,
        ub_slack,
    };
    let mut theta_lb = f32::NEG_INFINITY;
    let mut tier_bytes = 0u64;
    let mut entries_read = 0u64;
    let mut decoded: Vec<TierEntry> = Vec::new();
    // Frontier per term: the largest rescaled impact any unread block can
    // hold, `None` once every stream of the term is exhausted.
    let mut frontier: Vec<Option<f32>> = vec![Some(f32::INFINITY); nt];
    let mut excluded = vec![false; n_sf];
    // What the last round left behind, to tell a round that moved nothing.
    let mut prev_frontier = frontier.clone();
    let mut prev_candidates = 0usize;
    let mut prev_excluded = 0usize;
    let mut stale_rounds = 0u32;

    // The inline entries are in hand before any chunk is read; they take
    // part in the first round's threshold and exclusion like a read block.
    entries_read += inline.len() as u64;
    for (ti, e) in &inline {
        cands.absorb(*ti, e, index, &slot_of, &scoring);
    }

    // Each round settles what is known — threshold, frontiers, exclusions
    // — then reads the block most likely to move them.
    loop {
        if let Some(min) = cands.threshold() {
            theta_lb = min;
        }
        // Frontiers over the unread blocks.
        for (ti, slot) in frontier.iter_mut().enumerate() {
            let mut f: Option<f32> = None;
            for s in streams.iter_mut().filter(|s| s.term == ti) {
                if let Some(max) = next_block_max(index, s, &mut tier_bytes).await? {
                    let scaled = max * s.scale_up;
                    f = Some(f.map_or(scaled, |cur: f32| cur.max(scaled)));
                }
            }
            *slot = f;
        }
        // Exclude what cannot compete, and test whether reading on could
        // exclude anything more.
        let mut can_still_shrink = false;
        if theta_lb.is_finite() {
            for (slot, out) in excluded.iter_mut().enumerate() {
                if *out {
                    continue;
                }
                let (ub_now, ub_floor) = bounds_of(
                    slot,
                    n_masks,
                    &cands.best_ub,
                    &per_term,
                    &frontier,
                    required,
                );
                if ub_now * BOUND_SLACK < theta_lb {
                    *out = true;
                } else if ub_floor * BOUND_SLACK < theta_lb {
                    can_still_shrink = true;
                }
            }
            if !can_still_shrink {
                break;
            }
        }
        let n_excluded = excluded.iter().filter(|e| **e).count();
        let moved = frontier != prev_frontier
            || cands.acc.len() != prev_candidates
            || n_excluded != prev_excluded;
        prev_frontier.clone_from(&frontier);
        prev_candidates = cands.acc.len();
        prev_excluded = n_excluded;
        stale_rounds = if moved { 0 } else { stale_rounds + 1 };
        if stale_rounds >= STALE_ROUNDS_BEFORE_STOP {
            break;
        }
        let cap = if exact {
            MAX_ENTRIES_PER_QUERY
        } else {
            INEXACT_ENTRIES_PER_QUERY
        };
        if entries_read >= cap {
            break;
        }
        let probe = (THRESHOLD_PROBE_BLOCKS_PER_TERM * BLOCK_ENTRIES * nt) as u64;
        if !theta_lb.is_finite() && entries_read >= probe {
            break;
        }
        // The stream whose next block has the highest (scaled) maximum.
        let mut pick: Option<(usize, f32)> = None;
        for (si, s) in streams.iter_mut().enumerate() {
            let Some(max) = next_block_max(index, s, &mut tier_bytes).await? else {
                continue;
            };
            let scaled = max * s.scale_up;
            if pick.is_none_or(|(_, m)| scaled > m) {
                pick = Some((si, scaled));
            }
        }
        let Some((si, _)) = pick else {
            break;
        };
        let s = &mut streams[si];
        let dir = s
            .dir
            .as_ref()
            .expect("a stream with a next block has its directory");
        let body = index
            .tier_block(&s.object, &s.chunks[s.chunk], dir, s.block)
            .await?;
        tier_bytes += body.len() as u64;
        dir.decode_block(s.block, &body, &mut decoded)?;
        entries_read += decoded.len() as u64;
        for e in &decoded {
            cands.absorb(s.term, e, index, &slot_of, &scoring);
        }
        s.block += 1;
    }

    let mut bounds = HashMap::with_capacity(n_sf);
    for (slot, e) in kept.iter().enumerate() {
        let (ub_now, _) = bounds_of(
            slot,
            n_masks,
            &cands.best_ub,
            &per_term,
            &frontier,
            required,
        );
        let ub = if excluded[slot] {
            f32::NEG_INFINITY
        } else {
            ub_now
        };
        let mut lb = f32::NEG_INFINITY;
        for m in 1..n_masks {
            if m as u8 & required != required {
                continue;
            }
            lb = lb.max(cands.best_lb[slot * n_masks + m]);
        }
        bounds.insert(e.superfile_id, SuperfileBounds { lb, ub });
    }
    // With every term's whole list read and every score exact, the index
    // knows each superfile's matching documents and their scores: the
    // candidates carrying the required terms, as the kernel would find and
    // score them.
    let mut resolved: HashMap<Uuid, Vec<(u32, f32)>> = HashMap::new();
    let all_read = complete.iter().all(|c| *c) && frontier.iter().all(Option::is_none);
    if exact_scores && all_read {
        for (&(ordinal, doc_id), &(mask, sum)) in &cands.acc {
            if mask & required != required {
                continue;
            }
            let Some(id) = index.superfile_id(ordinal) else {
                continue;
            };
            if slot_of.contains_key(&id) {
                resolved.entry(id).or_default().push((doc_id, sum));
            }
        }
        // A kept superfile no candidate names holds no match at all.
        for e in kept {
            resolved.entry(e.superfile_id).or_default();
        }
    }
    Ok(Some(TierSelection {
        bounds,
        theta_lb,
        resolved,
        tier_bytes,
        entries_read,
    }))
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
        let norm = self.params.dl_norm_k1(dequantize_len(e.len), self.avgdl);
        score_with_dl_norm_k1(self.idf[ti], e.tf, norm)
    }
}

/// The average length `entry`'s `column` was built at, as its manifest
/// summary records it.
fn baked_avgdl_of(entry: &SuperfileEntry, column: FieldId) -> Option<f32> {
    entry
        .fts_summary
        .get(&column)
        .and_then(|s| s.baked_avgdl)
        .filter(|b| *b > 0.0)
}

/// What the merge knows of the documents the tiers have shown it.
struct Candidates {
    /// Per (superfile ordinal, row): the terms seen and their summed
    /// contributions.
    acc: FxHashMap<(u32, u32), (u8, f32)>,
    /// Per superfile, per term mask: best known lower-bound sum.
    best_lb: Vec<f32>,
    /// Per superfile, per term mask: best known upper-bound sum.
    best_ub: Vec<f32>,
    /// The `k` best lower bounds among documents carrying every term.
    topk: BinaryHeap<Reverse<OrdF32>>,
    n_masks: usize,
    /// The mask of the terms a matching document must carry.
    required: u8,
    k: usize,
    /// Whether a candidate's sum is a score the query gives.
    exact: bool,
    /// What a sum is narrowed and widened by for its lower and upper bound:
    /// one when the sum is the kernel's bit for bit.
    lb_slack: f32,
    ub_slack: f32,
}

impl Candidates {
    /// Fold one tier entry of term `ti` in.
    fn absorb(
        &mut self,
        ti: usize,
        e: &TierEntry,
        index: &TermIndex,
        slot_of: &HashMap<Uuid, usize>,
        scoring: &Scoring,
    ) {
        let Some(id) = index.superfile_id(e.superfile) else {
            return;
        };
        let Some(&slot) = slot_of.get(&id) else {
            return;
        };
        let cand = self.acc.entry((e.superfile, e.doc_id)).or_insert((0, 0.0));
        cand.0 |= 1 << ti;
        cand.1 += scoring.contribution(ti, e);
        let (lb, ub) = (cand.1 * self.lb_slack, cand.1 * self.ub_slack);
        let at = slot * self.n_masks + cand.0 as usize;
        if lb > self.best_lb[at] {
            self.best_lb[at] = lb;
        }
        if ub > self.best_ub[at] {
            self.best_ub[at] = ub;
        }
        // A candidate carrying every required term is a document the query
        // returns, and its contributions so far are a lower bound on its
        // score: for a disjunction any candidate, for a conjunction one
        // with every term, for a mix one with every must.
        if self.exact && cand.0 & self.required == self.required {
            self.topk.push(Reverse(OrdF32(lb)));
            if self.topk.len() > self.k {
                self.topk.pop();
            }
        }
    }

    /// The k-th best lower bound among candidates carrying the required
    /// terms, once `k` have been seen.
    fn threshold(&self) -> Option<f32> {
        if self.topk.len() < self.k {
            return None;
        }
        self.topk.peek().map(|Reverse(OrdF32(min))| *min)
    }
}

/// The upper bound on a document a posting's tier does not hold: the
/// stored tail, or the bound when the build recorded no tail.
fn tail_of(p: &Posting) -> f32 {
    if p.tail.is_finite() { p.tail } else { p.bound }
}

/// `(bound now, bound with every frontier lowered to the tails)` for one
/// superfile: the best over masks of a known partial plus what the missing
/// terms can still add, and the sum over all terms for a document outside
/// every tier.
/// A superfile's upper bounds: `now`, with every unread block still able
/// to hold a term's best document, and `floor`, as if every stream were
/// exhausted and only the tails remained — what further reading can at
/// best bring the bound down to.
///
/// A candidate missing a required term only exists if that term still has
/// documents in the superfile the merge has not seen: unread blocks, or a
/// tail above zero. A term whose streams are exhausted with a zero tail
/// has shown every document it has there, so a candidate without it is no
/// candidate at all.
fn bounds_of(
    slot: usize,
    n_masks: usize,
    best_ub: &[f32],
    per_term: &[Vec<TermInFile>],
    frontier: &[Option<f32>],
    required: u8,
) -> (f32, f32) {
    let add = |mask: usize, at_tails: bool| -> f32 {
        let mut sum = 0.0f32;
        for (ti, pt) in per_term.iter().enumerate() {
            if mask & (1 << ti) != 0 {
                continue;
            }
            let tail = pt[slot].tail;
            let unread = match (at_tails, frontier[ti]) {
                (false, Some(f)) => f.max(tail),
                _ => tail,
            };
            let needed = required & (1 << ti) != 0;
            if needed && unread <= 0.0 && (at_tails || frontier[ti].is_none()) {
                return f32::NEG_INFINITY;
            }
            sum += unread;
        }
        sum
    };
    let mut now = add(0, false);
    let mut floor = add(0, true);
    for m in 1..n_masks {
        let v = best_ub[slot * n_masks + m];
        if v > f32::NEG_INFINITY {
            now = now.max(v + add(m, false));
            floor = floor.max(v + add(m, true));
        }
    }
    (now, floor)
}

/// The next unread block's maximum impact in `s`, reading the directory
/// of the chunk that holds it; `None` once the stream is exhausted.
async fn next_block_max(
    index: &TermIndex,
    s: &mut Stream,
    tier_bytes: &mut u64,
) -> Result<Option<f32>, TermIndexError> {
    loop {
        let Some(chunk_ref) = s.chunks.get(s.chunk) else {
            return Ok(None);
        };
        if s.dir.is_none() {
            s.dir = Some(index.tier_directory(&s.object, chunk_ref).await?);
            *tier_bytes += TierDirectory::len_for(chunk_ref.n_entries) as u64;
        }
        let dir = s.dir.as_ref().expect("read above");
        if s.block < dir.n_blocks() {
            return Ok(Some(dir.block_max(s.block)));
        }
        s.chunk += 1;
        s.block = 0;
        s.dir = None;
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
