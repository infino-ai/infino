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
//! only the superfiles that are still in play. After each round, per
//! superfile, the documents read so far are joined across the terms, and
//! two bounds fall out:
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
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, HashSet},
    sync::Arc,
};

use uuid::Uuid;

use crate::{
    superfile::fts::{
        bm25::{Bm25Params, dequantize_len, idf as bm25_idf, score_with_dl_norm_k1},
        reader::GlobalTermIdf,
    },
    supertable::{
        manifest::{
            SuperfileEntry,
            part::ContentHash,
            term_index::{
                BLOCK_ENTRIES, SliceRun, TermIndex, TermIndexError, TierChunkRef, TierDirectory,
                TierEntry,
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
/// Most query terms the merge tracks masks for: masks are a bit set, and
/// the per-superfile table has one slot per mask.
const MAX_TIER_TERMS: usize = 6;
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
    /// Upper bound on the score of any document of this superfile.
    pub(crate) ub: f32,
}

/// The merge's outcome for one query.
#[derive(Debug)]
pub(crate) struct TierSelection {
    /// Bounds per kept superfile, keyed by superfile id.
    pub(crate) bounds: HashMap<Uuid, SuperfileBounds>,
    /// Lower bound on the final k-th score; `NEG_INFINITY` when fewer than
    /// `k` documents were seen with every required term.
    pub(crate) theta_lb: f32,
    /// Per superfile the index answers for outright — every term's head
    /// there was read to its end, with scores the kernel would compute bit
    /// for bit — its matching documents as `(row, score)`. Such a superfile
    /// need not be opened; tombstones still apply. Empty unless the scores
    /// are exact.
    pub(crate) resolved: HashMap<Uuid, Vec<(u32, f32)>>,
    /// Tier bytes consumed: directories and block bodies.
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
    /// Blocks read so far.
    blocks: usize,
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
            blocks: 0,
            done: true,
            source: None,
        }
    }
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
/// segment without a tier, a kept superfile is not in the root, or a
/// table-wide idf is missing: the caller then keeps its ceilings.
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
    let scored_docs: Vec<u64> = kept
        .iter()
        .map(|e| {
            e.fts_summary
                .get(&column)
                .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                .unwrap_or(e.n_docs)
        })
        .collect();
    let n_sf = kept.len();

    let mut idf: Vec<f32> = Vec::with_capacity(nt);
    let mut heads: Vec<Vec<HeadState>> = vec![vec![HeadState::absent(); nt]; n_sf];
    let mut sources: Vec<Vec<Source>> = (0..nt).map(|_| Vec::new()).collect();
    let mut ordinal_of: Vec<u32> = vec![u32::MAX; n_sf];
    let mut inline: Vec<(usize, TierEntry)> = Vec::new();
    for (ti, term) in terms.iter().enumerate() {
        let Some(&idf_table) = global_idf.get(*term) else {
            return Ok(None);
        };
        idf.push(idf_table);
        let runs = index.runs(column, term).await?;
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
                let Some(&slot) = slot_of.get(&id) else {
                    continue;
                };
                ordinal_of[slot] = p.superfile;
                // The stored ceiling is an impact at the superfile's own idf;
                // the ratio to the table's rescales it, as the ceiling path
                // does. The tail is a key at the superfile's own average.
                let n = scored_docs[slot];
                let local_idf = bm25_idf(n, p.df.min(n));
                let ratio = if local_idf > 0.0 {
                    idf_table / local_idf
                } else {
                    1.0
                };
                let up = scale_up[slot];
                let ceiling = p.bound * ratio * up;
                // A tail is a key at the superfile's own average; a build
                // that recorded none leaves the ceiling as the bound.
                let tail = if p.tail.is_finite() {
                    (p.tail * idf_table * up).min(ceiling)
                } else {
                    ceiling
                };
                heads[slot][ti] = HeadState {
                    docs: Vec::new(),
                    tail,
                    ceiling,
                    blocks: 0,
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
    let mut entries_read = 0u64;
    let mut tier_bytes = 0u64;
    // Inline heads are whole: their entries are in hand before any read.
    for (ti, e) in &inline {
        let Some(id) = index.superfile_id(e.superfile) else {
            continue;
        };
        let Some(&slot) = slot_of.get(&id) else {
            continue;
        };
        heads[slot][*ti]
            .docs
            .push((e.doc_id, scoring.contribution(*ti, e)));
        entries_read += 1;
    }
    for slot in heads.iter_mut() {
        for h in slot.iter_mut() {
            h.docs.sort_unstable_by_key(|d| d.0);
        }
    }

    let mut live: Vec<bool> = vec![true; n_sf];
    let mut theta_lb = f32::NEG_INFINITY;
    let mut decoded: Vec<TierEntry> = Vec::new();
    let mut joined: Vec<(u32, u8, f32)> = Vec::new();
    for round in 0.. {
        // Read: the next block of every live superfile whose head goes on.
        let mut read_any = false;
        for slot in 0..n_sf {
            if !live[slot] {
                continue;
            }
            for ti in 0..nt {
                let state = &heads[slot][ti];
                if state.done {
                    continue;
                }
                let Some(si) = state.source else {
                    continue;
                };
                let source = &mut sources[ti][si];
                let Some(chunk) = source.chunks.get(round) else {
                    heads[slot][ti].done = true;
                    continue;
                };
                if source.dirs[round].is_none() {
                    let dir = index.tier_directory(&source.object, chunk).await?;
                    tier_bytes += TierDirectory::len_for(chunk.n_blocks) as u64;
                    source.dirs[round] = Some(dir);
                }
                let dir = source.dirs[round].as_ref().expect("read above");
                let Some(block) = dir.block_of(ordinal_of[slot]) else {
                    heads[slot][ti].done = true;
                    continue;
                };
                let body = index.tier_block(&source.object, chunk, dir, block).await?;
                tier_bytes += body.len() as u64;
                dir.decode_block(block, &body, &mut decoded)?;
                entries_read += decoded.len() as u64;
                let state = &mut heads[slot][ti];
                state.docs.extend(
                    decoded
                        .iter()
                        .map(|e| (e.doc_id, scoring.contribution(ti, e))),
                );
                state.docs.sort_unstable_by_key(|d| d.0);
                state.blocks += 1;
                let tail_key = dir.block_tail(block);
                state.tail = (tail_key * scoring.idf[ti] * scale_up[slot]).min(state.ceiling);
                if tail_key <= 0.0 || decoded.len() < BLOCK_ENTRIES {
                    state.done = true;
                }
                read_any = true;
            }
        }
        if round > 0 && !read_any {
            break;
        }
        // Threshold: the k-th best sum over documents the query returns,
        // across every superfile read, in play or not.
        let mut topk: BinaryHeap<Reverse<OrdF32>> = BinaryHeap::with_capacity(k + 1);
        for slot_heads in &heads {
            join(slot_heads, &mut joined);
            for &(_, mask, sum) in &joined {
                if mask & required == required {
                    topk.push(Reverse(OrdF32(sum * lb_slack)));
                    if topk.len() > k {
                        topk.pop();
                    }
                }
            }
        }
        if topk.len() >= k
            && let Some(Reverse(OrdF32(min))) = topk.peek()
        {
            theta_lb = theta_lb.max(*min);
        }
        // Rule out what cannot compete.
        if theta_lb.is_finite() {
            for slot in 0..n_sf {
                if !live[slot] {
                    continue;
                }
                let (ub, _) = bounds_of(&heads[slot], required, ub_slack, &mut joined);
                if ub * BOUND_SLACK < theta_lb {
                    live[slot] = false;
                }
            }
        }
        if !read_any || entries_read >= MAX_ENTRIES_PER_QUERY {
            break;
        }
    }

    let mut bounds = HashMap::with_capacity(n_sf);
    let mut resolved: HashMap<Uuid, Vec<(u32, f32)>> = HashMap::new();
    for (slot, e) in kept.iter().enumerate() {
        let (ub, lb) = bounds_of(&heads[slot], required, ub_slack, &mut joined);
        let ub = if live[slot] { ub } else { f32::NEG_INFINITY };
        bounds.insert(
            e.superfile_id,
            SuperfileBounds {
                lb: lb * lb_slack,
                ub,
            },
        );
        // Every head read to its end, and the sums exact: the superfile's
        // matching documents are its candidates carrying the required
        // terms, scored as the kernel would score them.
        let whole = heads[slot].iter().all(|h| h.done && h.tail <= 0.0);
        if exact_scores && whole {
            join(&heads[slot], &mut joined);
            resolved.insert(
                e.superfile_id,
                joined
                    .iter()
                    .filter(|(_, mask, _)| mask & required == required)
                    .map(|&(doc, _, sum)| (doc, sum))
                    .collect(),
            );
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

/// Join one superfile's heads across the terms: per document read under
/// any term, the terms it was read under and its summed contributions, in
/// ascending row.
fn join(heads: &[HeadState], out: &mut Vec<(u32, u8, f32)>) {
    out.clear();
    let mut at: Vec<usize> = vec![0; heads.len()];
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
    required: u8,
    ub_slack: f32,
    joined: &mut Vec<(u32, u8, f32)>,
) -> (f32, f32) {
    for (ti, h) in heads.iter().enumerate() {
        if required & (1 << ti) != 0 && h.ceiling <= 0.0 && h.docs.is_empty() {
            return (f32::NEG_INFINITY, f32::NEG_INFINITY);
        }
    }
    let tails: Vec<f32> = heads.iter().map(|h| h.tail).collect();
    let mut ub: f32 = tails.iter().sum();
    let mut lb = f32::NEG_INFINITY;
    join(heads, joined);
    for &(_, mask, sum) in joined.iter() {
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
