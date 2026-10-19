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

use std::collections::{HashMap, HashSet};

use bytes::Bytes;
use rustc_hash::FxHashMap;
use uuid::Uuid;

use crate::{
    superfile::fts::{bm25::idf as bm25_idf, reader::GlobalTermIdf},
    supertable::{
        manifest::{
            SuperfileEntry,
            part::ContentHash,
            term_index::{
                Posting, SliceRun, TermIndex, TermIndexError, TierChunk, TierChunkRef, TierEntry,
            },
        },
        schema::FieldId,
    },
};

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
    /// Tier chunk bytes consumed.
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
    bytes: Option<Bytes>,
    /// Largest scale any kept superfile applies to this term's stored
    /// impacts, so a block maximum bounds every superfile's unread entries.
    scale_up: f32,
}

/// Per (term, kept superfile): how a stored impact maps to the query's
/// scoring, and what the superfile can still contribute beyond the tier.
#[derive(Clone, Copy)]
struct TermInFile {
    /// Multiplies a stored impact into a lower bound at the table's stats.
    scale_lb: f32,
    /// Multiplies a stored impact into an upper bound at the table's stats.
    scale_ub: f32,
    /// Upper bound, at the table's stats, on a document the tier does not
    /// hold.
    tail: f32,
}

impl Default for TermInFile {
    fn default() -> Self {
        Self {
            scale_lb: 0.0,
            scale_ub: 0.0,
            tail: 0.0,
        }
    }
}

/// Select superfiles for a query over `terms` with the tiers of `index`.
/// `required` is a bit per term (in `terms` order) a matching document must
/// carry: every bit for a conjunction, none for a disjunction, the musts'
/// for a mix. `exact` says a candidate's summed impacts are the score the
/// query would give it; when they are not — a phrase scores only where its
/// words are adjacent, a negation removes documents — the merge takes no
/// threshold from them and rules out only what a required term has shown
/// to hold nothing. `None` when a term has postings in a segment without a
/// tier, or there are more terms than the merge tracks: the caller then
/// orders by ceiling as before.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn select_with_tiers(
    index: &TermIndex,
    column: FieldId,
    terms: &[&str],
    required: u8,
    exact: bool,
    kept: &[std::sync::Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
    table_avgdl: f32,
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
    let scored_docs: Vec<u64> = kept
        .iter()
        .map(|e| {
            e.fts_summary
                .get(&column)
                .and_then(|s| s.length_stats.as_ref().map(|l| l.n_scored_docs))
                .unwrap_or(e.n_docs)
        })
        .collect();
    // Average-length factors per kept superfile: a file built at a smaller
    // average scores at most `table/baked` higher at the table's, and one
    // built at a larger average at least `table/baked` of its stored score.
    let avgdl_factor: Vec<(f32, f32)> = kept
        .iter()
        .map(|e| {
            match e
                .fts_summary
                .get(&column)
                .and_then(|s| s.baked_avgdl)
                .filter(|b| *b > 0.0)
            {
                Some(baked) if baked < table_avgdl => (1.0, (table_avgdl / baked).next_up()),
                Some(baked) if baked > table_avgdl => ((table_avgdl / baked).next_down(), 1.0),
                _ => (1.0, 1.0),
            }
        })
        .collect();

    let n_sf = kept.len();
    let mut per_term: Vec<Vec<TermInFile>> = vec![vec![TermInFile::default(); n_sf]; nt];
    let mut streams: Vec<Stream> = Vec::new();
    for (ti, term) in terms.iter().enumerate() {
        let Some(&idf_table) = global_idf.get(*term) else {
            return Ok(None);
        };
        let runs = index.runs(column, term).await?;
        let mut scale_up_max = 0.0f32;
        for SliceRun { tier_object, run } in runs {
            let tier = match &run.tier {
                Some(t) => t,
                // Postings without a tier: this segment predates tiers or
                // kept none for the term; no sound frontier exists for it.
                None if run.postings.is_empty() => continue,
                None => return Ok(None),
            };
            let Some(object) = tier_object else {
                return Ok(None);
            };
            for p in &run.postings {
                let Some(id) = index.superfile_id(p.superfile) else {
                    continue;
                };
                let Some(&slot) = slot_of.get(&id) else {
                    continue;
                };
                let n = scored_docs[slot];
                let local_idf = bm25_idf(n, p.df.min(n));
                let ratio = if local_idf > 0.0 {
                    idf_table / local_idf
                } else {
                    1.0
                };
                let (down, up) = avgdl_factor[slot];
                let entry = TermInFile {
                    scale_lb: ratio * down * LOWER_SLACK,
                    scale_ub: ratio * up,
                    tail: tail_of(p) * ratio * up,
                };
                scale_up_max = scale_up_max.max(entry.scale_ub);
                per_term[ti][slot] = entry;
            }
            streams.push(Stream {
                term: ti,
                object,
                chunks: tier.chunks.clone(),
                chunk: 0,
                block: 0,
                bytes: None,
                scale_up: scale_up_max,
            });
        }
        // Every stream of a term bounds with the term's largest scale.
        for s in streams.iter_mut().filter(|s| s.term == ti) {
            s.scale_up = scale_up_max;
        }
    }

    let n_masks = 1usize << nt;
    let full = (n_masks - 1) as u8;
    let mut acc: FxHashMap<(u32, u32), (u8, f32, f32)> = FxHashMap::default();
    // Per superfile, per mask: best known (lower-bound sum, upper-bound sum).
    let mut best_lb = vec![f32::NEG_INFINITY; n_sf * n_masks];
    let mut best_ub = vec![f32::NEG_INFINITY; n_sf * n_masks];
    let mut topk: std::collections::BinaryHeap<std::cmp::Reverse<OrdF32>> =
        std::collections::BinaryHeap::with_capacity(k + 1);
    let mut theta_lb = f32::NEG_INFINITY;
    let mut tier_bytes = 0u64;
    let mut entries_read = 0u64;
    let mut decoded: Vec<TierEntry> = Vec::new();
    // Frontier per term: the largest rescaled impact any unread block can
    // hold, `None` once every stream of the term is exhausted.
    let mut frontier: Vec<Option<f32>> = vec![Some(f32::INFINITY); nt];
    let mut excluded = vec![false; n_sf];

    loop {
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
        // Read it.
        {
            let s = &mut streams[si];
            let bytes = s
                .bytes
                .as_ref()
                .expect("a stream with a next block has its chunk loaded");
            let chunk = TierChunk::open(bytes)?;
            chunk.decode_block(s.block, &mut decoded)?;
            entries_read += decoded.len() as u64;
            let ti = s.term;
            for e in &decoded {
                let Some(id) = index.superfile_id(e.superfile) else {
                    continue;
                };
                let Some(&slot) = slot_of.get(&id) else {
                    continue;
                };
                let tf = per_term[ti][slot];
                let cand = acc.entry((e.superfile, e.doc_id)).or_insert((0, 0.0, 0.0));
                cand.0 |= 1 << ti;
                cand.1 += e.impact * tf.scale_lb;
                cand.2 += e.impact * tf.scale_ub;
                let at = slot * n_masks + cand.0 as usize;
                if cand.1 > best_lb[at] {
                    best_lb[at] = cand.1;
                }
                if cand.2 > best_ub[at] {
                    best_ub[at] = cand.2;
                }
                if exact && cand.0 == full {
                    topk.push(std::cmp::Reverse(OrdF32(cand.1)));
                    if topk.len() > k {
                        topk.pop();
                    }
                }
            }
            s.block += 1;
        }
        if topk.len() >= k
            && let Some(std::cmp::Reverse(OrdF32(min))) = topk.peek()
        {
            theta_lb = *min;
        }
        // Frontiers after this block.
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
        if !theta_lb.is_finite() {
            let cap = if exact {
                MAX_ENTRIES_PER_QUERY
            } else {
                INEXACT_ENTRIES_PER_QUERY
            };
            if entries_read >= cap {
                break;
            }
            continue;
        }
        // Exclude what cannot compete, and test whether reading on could
        // exclude anything more.
        let mut can_still_shrink = false;
        for (slot, out) in excluded.iter_mut().enumerate() {
            if *out {
                continue;
            }
            let (ub_now, ub_floor) =
                bounds_of(slot, n_masks, &best_ub, &per_term, &frontier, required);
            if ub_now * BOUND_SLACK < theta_lb {
                *out = true;
            } else if ub_floor * BOUND_SLACK < theta_lb {
                can_still_shrink = true;
            }
        }
        if !can_still_shrink || entries_read >= MAX_ENTRIES_PER_QUERY {
            break;
        }
    }

    let mut bounds = HashMap::with_capacity(n_sf);
    for (slot, e) in kept.iter().enumerate() {
        let (ub_now, _) = bounds_of(slot, n_masks, &best_ub, &per_term, &frontier, required);
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
            lb = lb.max(best_lb[slot * n_masks + m]);
        }
        bounds.insert(e.superfile_id, SuperfileBounds { lb, ub });
    }
    Ok(Some(TierSelection {
        bounds,
        theta_lb,
        tier_bytes,
        entries_read,
    }))
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

/// The next unread block's maximum impact in `s`, loading the chunk that
/// holds it; `None` once the stream is exhausted.
async fn next_block_max(
    index: &TermIndex,
    s: &mut Stream,
    tier_bytes: &mut u64,
) -> Result<Option<f32>, TermIndexError> {
    loop {
        let Some(chunk_ref) = s.chunks.get(s.chunk) else {
            return Ok(None);
        };
        if s.bytes.is_none() {
            let bytes = index.tier_chunk(&s.object, chunk_ref).await?;
            *tier_bytes += bytes.len() as u64;
            s.bytes = Some(bytes);
        }
        let chunk = TierChunk::open(s.bytes.as_ref().expect("loaded above"))?;
        if s.block < chunk.n_blocks() {
            return Ok(Some(chunk.block_max(s.block)));
        }
        s.chunk += 1;
        s.block = 0;
        s.bytes = None;
    }
}

/// Total-order wrapper for the threshold heap.
#[derive(PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
