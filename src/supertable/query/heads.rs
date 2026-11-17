// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Choosing the superfiles a ranked query opens from the term index's
//! *level heads*: per term and layer, the term's best documents across
//! every superfile of the layer, in descending key, read a level at a time
//! from the top.
//!
//! Each round reads the next level of every head that still matters, adds
//! what it read to the candidates — documents with the contributions of
//! the terms seen for them so far — and recomputes two bounds:
//!
//! - `θ`, a lower bound on the final k-th score: the k-th best sum among
//!   candidates that carry every required term. Contributions only grow as
//!   more terms are seen.
//! - per superfile, an upper bound on any of its documents: for a
//!   candidate, its known sum plus, per term not yet seen for it, what that
//!   term's unread entries can still contribute at the candidate's length;
//!   for a document in no head yet, the *length-coherent* bound below.
//!
//! A superfile whose bound falls below `θ` holds nothing that can place
//! and is never opened; the rest open with `θ` seeded as the floor, in the
//! order of their best candidate. A superfile every term's head lists
//! whole is answered from the index outright. Reading stops when no head
//! can lower a bound that still matters, when the heads are read to their
//! ends, or at the read budget; whatever is still in play then opens, so
//! the answer is exact whatever the heads hold.
//!
//! **The length-coherent bound.** A document has one length for all its
//! terms, so in a length bucket `L` it scores at most
//! `Σ_t idf_t · tf_t / (tf_t + norm_L)` over tfs that keep each term's key
//! under its head's frontier, each tf under the term's largest tf in `L`,
//! and their total under the bucket's longest length. Each term's
//! contribution is concave in its tf, so the continuous optimum is a
//! water-filling — every term takes tokens until its marginal gain falls to
//! a common level — and it bounds the integer optimum from above.
//!
//! Scores are the kernel's own arithmetic at the table's statistics; keys
//! stored at a superfile's own average length are scaled to the table's
//! as the ceilings are.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock},
    time::Instant,
};

use futures::future::try_join_all;
use roaring::RoaringBitmap;

use super::{
    fts::baked_avgdl,
    tier::{
        BOUND_SLACK, EXACT_SUM_TERMS, MAX_TIER_TERMS, SuperfileBounds, TermMask, TierQuery,
        TierSelection,
    },
};
use crate::{
    superfile::fts::{
        bm25::{dequantize_len, quantize_len, score_with_dl_norm_k1},
        reader::{GlobalTermIdf, norm_table},
    },
    supertable::{
        manifest::{
            SuperfileEntry,
            part::ContentHash,
            term_index::{BucketMaxTf, LevelRef, SliceRun, TermIndex, TermIndexError},
        },
        schema::FieldId,
    },
};

/// Head bytes a query reads at most before it stops and opens what is
/// still in play: a few seconds of reads on a cold store.
pub(crate) const HEAD_READ_BUDGET_BYTES: u64 = 256 << 20;
/// Steps of the water-filling's search for its common marginal gain; each
/// halves the interval, so the allocation it bounds with is within a
/// negligible share of the budget it was given.
const WATER_FILL_STEPS: usize = 64;
/// Length buckets: the one-byte codes the length quantizer gives.
const LEN_BUCKETS: usize = 1 << u8::BITS;

/// The longest length each bucket holds: the quantizer keeps a length's
/// top bits, so a bucket spans up to the next code's first length. A code
/// no length quantizes to, and the last one, hold up to the largest.
static BUCKET_MAX_LEN: LazyLock<[u32; LEN_BUCKETS]> = LazyLock::new(|| {
    let reachable = |b: u8| quantize_len(dequantize_len(b)) == b;
    let mut out = [u32::MAX; LEN_BUCKETS];
    for b in 0..=u8::MAX {
        if !reachable(b) {
            continue;
        }
        let next = (b..=u8::MAX).skip(1).find(|&n| reachable(n));
        if let Some(next) = next {
            out[usize::from(b)] = dequantize_len(next) - 1;
        }
    }
    out
});

/// One term's head in one layer, as far as the query has read it.
struct Head {
    term: usize,
    object: Option<ContentHash>,
    /// Added to the ordinals the head's levels store.
    base: u32,
    levels: Vec<LevelRef>,
    /// The next level to read.
    next: usize,
    /// Bound on the stored key of every entry not yet read.
    frontier: f32,
    floor: f32,
    buckets: Option<BucketMaxTf>,
}

impl Head {
    fn done(&self) -> bool {
        self.next >= self.levels.len()
    }

    /// Bound on the stored key of any document of the term in the layer
    /// that the query has not read: past the head's end only the floor.
    fn unread(&self) -> f32 {
        match self.done() {
            true => self.floor,
            false => self.frontier.max(self.floor),
        }
    }

    /// Whether every document of the term in the layer has been read.
    fn whole(&self) -> bool {
        self.done() && self.floor <= 0.0
    }
}

/// A document some head named: its length bucket, the terms seen for it
/// and what each contributed.
struct Candidate {
    len: u8,
    known: TermMask,
    contrib: Vec<f32>,
}

impl Candidate {
    fn sum(&self) -> f32 {
        self.contrib.iter().sum()
    }
}

/// One kept superfile: per term, the head its documents of the term lie
/// in (`None` when it has none), how far its stored keys scale at the
/// table's average, and its deleted rows.
struct Slot {
    heads: Vec<Option<usize>>,
    scale_up: f32,
    deny: Option<Arc<RoaringBitmap>>,
    id_max: i128,
}

/// The query's terms' weights and the kernel's length norms.
struct Weights {
    idf: Vec<f32>,
    norm: Arc<[f32; LEN_BUCKETS]>,
}

impl Weights {
    /// What term `t` contributes to a document of `tf` occurrences in
    /// bucket `len`: the kernel's own arithmetic.
    fn contribution(&self, t: usize, tf: u32, len: u8) -> f32 {
        score_with_dl_norm_k1(self.idf[t], tf, self.norm[usize::from(len)])
    }
}

/// Whether every run of `terms` in `column` that has postings carries a
/// level head: a term index built since level heads, or folded since.
pub(crate) async fn available(
    index: &TermIndex,
    column: FieldId,
    terms: &[&str],
) -> Result<bool, TermIndexError> {
    let runs = index.runs_many(column, terms).await?;
    Ok(runs.iter().all(|runs| {
        runs.iter()
            .all(|r| r.run.postings.is_empty() || r.run.head.is_some())
    }))
}

/// Choose the superfiles `query` opens from the level heads, or `None`
/// when the heads cannot serve it: a phrase or a negation, a term with
/// postings in a layer that has no level head, or a superfile the index
/// does not list or whose average length is unknown.
pub(crate) async fn select(
    index: &TermIndex,
    query: &TierQuery<'_>,
    kept: &[Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
) -> Result<Option<TierSelection>, TermIndexError> {
    let started = Instant::now();
    let terms = query.terms;
    let nt = terms.len();
    if nt == 0
        || nt > MAX_TIER_TERMS
        || query.k == 0
        || kept.is_empty()
        || !query.phrases.is_empty()
        || query.negated
    {
        return Ok(None);
    }
    let mut idf = Vec::with_capacity(nt);
    for term in terms {
        let Some(&w) = global_idf.get(*term) else {
            return Ok(None);
        };
        idf.push(w);
    }
    let weights = Weights {
        idf,
        norm: norm_table(query.table_avgdl, query.params),
    };
    let slot_of: HashMap<_, usize> = kept
        .iter()
        .enumerate()
        .map(|(i, e)| (e.superfile_id, i))
        .collect();
    let mut slots = Vec::with_capacity(kept.len());
    for (i, e) in kept.iter().enumerate() {
        let Some(baked) = baked_avgdl(e, query.column).filter(|b| *b > 0.0) else {
            return Ok(None);
        };
        if !index.is_indexed(&e.superfile_id) {
            return Ok(None);
        }
        // A superfile built at a smaller average scores at most
        // `table/baked` higher at the table's.
        let scale_up = match baked < query.table_avgdl {
            true => (query.table_avgdl / baked).next_up(),
            false => 1.0,
        };
        slots.push(Slot {
            heads: vec![None; nt],
            scale_up,
            deny: query.deny.get(i).cloned().flatten(),
            id_max: e.id_max,
        });
    }
    let slot_for = |ordinal: u32| {
        index
            .superfile_id(ordinal)
            .and_then(|id| slot_of.get(&id).copied())
    };
    let runs_per_term = index.runs_many(query.column, terms).await?;
    let mut heads: Vec<Head> = Vec::new();
    let mut candidates: HashMap<(u32, u32), Candidate> = HashMap::new();
    let mut entries_read = 0u64;
    for (t, runs) in runs_per_term.iter().enumerate() {
        for SliceRun {
            tier_object,
            run,
            base,
        } in runs.iter()
        {
            if run.postings.is_empty() {
                continue;
            }
            let Some(head) = &run.head else {
                return Ok(None);
            };
            let h = heads.len();
            for p in &run.postings {
                if let Some(s) = slot_for(p.superfile) {
                    slots[s].heads[t] = Some(h);
                }
            }
            // Before a level is read, the best key a posting records bounds
            // every entry.
            let frontier = run
                .postings
                .iter()
                .map(|p| p.bound)
                .fold(head.floor, f32::max);
            for e in &head.inline {
                if let Some(s) = slot_for(e.superfile) {
                    absorb(
                        &mut candidates,
                        &slots,
                        &weights,
                        nt,
                        t,
                        s,
                        e.doc_id,
                        e.tf,
                        e.len,
                    );
                    entries_read += 1;
                }
            }
            heads.push(Head {
                term: t,
                object: *tier_object,
                base: *base,
                levels: head.levels.clone(),
                next: 0,
                frontier,
                floor: head.floor,
                buckets: head.buckets.clone(),
            });
        }
    }
    let required = query.required;
    let k = query.k;
    let exact_scores = nt <= EXACT_SUM_TERMS;
    // Past two terms a candidate's sum may round differently from the
    // kernel's, which adds in its own order.
    let lb_slack = match exact_scores {
        true => 1.0,
        false => 1.0 - nt as f32 * 2.0 * f32::EPSILON,
    };
    let mut fetched = 0u64;
    let mut theta;
    let mut bounds;
    // What the last round left: its threshold and the slots it left in play.
    let mut last: Option<(f32, usize)> = None;
    loop {
        theta = kth_lower_bound(&candidates, required, k, lb_slack);
        bounds = slot_bounds(&slots, &heads, &candidates, &weights, required, lb_slack);
        if fetched >= HEAD_READ_BUDGET_BYTES {
            break;
        }
        // A round that neither raised the threshold nor ruled a slot out
        // ends the reading: on documents that tie across the table the
        // bounds sit at the threshold however deep the heads are read, and
        // the opens settle the ties by `_id` instead.
        let in_play = bounds.iter().filter(|(_, ub)| *ub > theta).count();
        if theta.is_finite() && last == Some((theta, in_play)) {
            break;
        }
        last = Some((theta, in_play));
        let wanted = heads_that_matter(&slots, &heads, &candidates, &weights, &bounds, theta);
        if wanted.is_empty() {
            break;
        }
        let reads = wanted.iter().map(|&h| {
            let head = &heads[h];
            let level = head.levels[head.next];
            let object = head.object;
            async move {
                let Some(object) = object else {
                    return Err(TermIndexError::Malformed(
                        "a head with levels and no object".into(),
                    ));
                };
                index.head_level(&object, &level).await
            }
        });
        let levels = try_join_all(reads).await?;
        for (&h, (blocks, bytes)) in wanted.iter().zip(levels) {
            fetched += bytes;
            let (t, base) = (heads[h].term, heads[h].base);
            for block in blocks.iter() {
                for e in &block.entries {
                    entries_read += 1;
                    if let Some(s) = slot_for(base + e.superfile) {
                        absorb(
                            &mut candidates,
                            &slots,
                            &weights,
                            nt,
                            t,
                            s,
                            e.doc_id,
                            e.tf,
                            e.len,
                        );
                    }
                }
            }
            let head = &mut heads[h];
            head.next += 1;
            if let Some(last) = blocks.last() {
                head.frontier = last.last_key;
            }
        }
    }
    let resolved = match exact_scores {
        true => resolve(&slots, &heads, &candidates, required),
        false => Vec::new(),
    };
    let theta_id_max = theta_id_max(&candidates, &slots, required, k, lb_slack, theta);
    Ok(Some(TierSelection {
        bounds: bounds
            .into_iter()
            .map(|(lb, ub)| SuperfileBounds { lb, ub })
            .collect(),
        theta_lb: theta,
        theta_id_max,
        resolved,
        tier_bytes: fetched,
        entries_read,
        merge_ns: started.elapsed().as_nanos() as u64,
    }))
}

/// Add term `t`'s entry for row `row` of slot `s` to the candidates,
/// unless the row is deleted.
#[allow(clippy::too_many_arguments)]
fn absorb(
    candidates: &mut HashMap<(u32, u32), Candidate>,
    slots: &[Slot],
    weights: &Weights,
    nt: usize,
    t: usize,
    s: usize,
    row: u32,
    tf: u32,
    len: u8,
) {
    if slots[s].deny.as_ref().is_some_and(|d| d.contains(row)) {
        return;
    }
    let c = candidates
        .entry((s as u32, row))
        .or_insert_with(|| Candidate {
            len,
            known: 0,
            contrib: vec![0.0; nt],
        });
    if c.known & (1 << t) == 0 {
        c.known |= 1 << t;
        c.contrib[t] = weights.contribution(t, tf, len);
    }
}

/// The k-th best lower bound among candidates carrying every required
/// term; `NEG_INFINITY` when fewer than `k` do.
fn kth_lower_bound(
    candidates: &HashMap<(u32, u32), Candidate>,
    required: TermMask,
    k: usize,
    lb_slack: f32,
) -> f32 {
    let mut lbs: Vec<f32> = candidates
        .values()
        .filter(|c| c.known & required == required)
        .map(|c| c.sum() * lb_slack)
        .collect();
    if lbs.len() < k {
        return f32::NEG_INFINITY;
    }
    let (_, kth, _) = lbs.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    *kth
}

/// Per term of a slot, what an unread entry of the term can still
/// contribute in bucket `len`: its head's unread key at the slot's scale,
/// held under the term's largest tf in the bucket. Zero for a term the slot
/// has no documents of, or none of that length.
fn term_cap(head: &Head, scale_up: f32, weights: &Weights, len: u8) -> f32 {
    let idf = weights.idf[head.term];
    let by_key = idf * head.unread() * scale_up * BOUND_SLACK;
    match &head.buckets {
        None => by_key,
        Some(buckets) => match buckets.get(len) {
            None => 0.0,
            Some(max_tf) => by_key.min(weights.contribution(head.term, max_tf, len) * BOUND_SLACK),
        },
    }
}

/// Per slot, `(lb, ub)`: the best lower bound among its candidates
/// carrying every required term, and a bound on any of its documents;
/// `NEG_INFINITY` for a slot that holds no match.
fn slot_bounds(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    required: TermMask,
    lb_slack: f32,
) -> Vec<(f32, f32)> {
    // Slots of one layer read the same heads at the same scale: their
    // unseen bound is computed once.
    let mut by_shape: HashMap<(&[Option<usize>], u32), f32> = HashMap::new();
    let mut out: Vec<(f32, f32)> = slots
        .iter()
        .map(|slot| {
            let unseen = *by_shape
                .entry((&slot.heads, slot.scale_up.to_bits()))
                .or_insert_with(|| unseen_bound(slot, heads, weights, required));
            (f32::NEG_INFINITY, unseen)
        })
        .collect();
    for (&(s, _), c) in candidates {
        let slot = &slots[s as usize];
        let (lb, ub) = &mut out[s as usize];
        if c.known & required == required {
            *lb = lb.max(c.sum() * lb_slack);
        }
        *ub = ub.max(candidate_bound(slot, heads, weights, required, c));
    }
    out
}

/// A bound on a candidate's score: its known contributions, and per term
/// not yet seen for it what the term's unread entries can still add.
/// `NEG_INFINITY` when a required term can no longer be seen for it.
fn candidate_bound(
    slot: &Slot,
    heads: &[Head],
    weights: &Weights,
    required: TermMask,
    c: &Candidate,
) -> f32 {
    let mut ub = c.sum() * BOUND_SLACK;
    for (t, h) in slot.heads.iter().enumerate() {
        if c.known & (1 << t) != 0 {
            continue;
        }
        let cap = h.map_or(0.0, |h| term_cap(&heads[h], slot.scale_up, weights, c.len));
        if cap <= 0.0 && required & (1 << t) != 0 {
            return f32::NEG_INFINITY;
        }
        ub += cap;
    }
    ub
}

/// A bound on any document of `slot` that no head has named yet;
/// `NEG_INFINITY` when none can match: a required term the slot lacks, or
/// one whose head was read whole.
fn unseen_bound(slot: &Slot, heads: &[Head], weights: &Weights, required: TermMask) -> f32 {
    let present: Vec<usize> = slot.heads.iter().flatten().copied().collect();
    for (t, h) in slot.heads.iter().enumerate() {
        if required & (1 << t) != 0 && h.is_none_or(|h| heads[h].whole()) {
            return f32::NEG_INFINITY;
        }
    }
    if present.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut best = f32::NEG_INFINITY;
    let mut caps = Vec::with_capacity(present.len());
    let mut idfs = Vec::with_capacity(present.len());
    for len in 0..LEN_BUCKETS {
        let len = len as u8;
        caps.clear();
        idfs.clear();
        for &h in &present {
            let cap = term_cap(&heads[h], slot.scale_up, weights, len);
            if cap > 0.0 {
                caps.push(cap);
                idfs.push(weights.idf[heads[h].term]);
            }
        }
        if caps.is_empty() {
            continue;
        }
        let bound = water_fill(
            &idfs,
            &caps,
            weights.norm[usize::from(len)],
            BUCKET_MAX_LEN[usize::from(len)],
        );
        best = best.max(bound * BOUND_SLACK);
    }
    best
}

/// The largest `Σ_t min(idf_t · x_t / (x_t + norm), cap_t)` over real
/// `x_t ≥ 0` with `Σ x_t ≤ tokens`: a bound on the integer optimum, since
/// the integers are among the reals.
fn water_fill(idfs: &[f32], caps: &[f32], norm: f32, tokens: u32) -> f32 {
    let n = f64::from(norm.max(0.0));
    let budget = f64::from(tokens);
    // Tokens each term takes to reach its cap: past them it gains nothing.
    let need: Vec<f64> = idfs
        .iter()
        .zip(caps)
        .map(|(&idf, &cap)| {
            let (idf, cap) = (f64::from(idf), f64::from(cap));
            match cap < idf {
                true => cap * n / (idf - cap),
                false => f64::INFINITY,
            }
        })
        .collect();
    let all_caps: f64 = caps.iter().map(|&c| f64::from(c)).sum();
    if need.iter().sum::<f64>() <= budget {
        return all_caps as f32;
    }
    // At a common marginal gain `λ`, term `t` takes `√(idf·n/λ) − n` tokens,
    // up to its need. The total falls as `λ` rises; find the smallest `λ`
    // whose total still covers the budget, and bound with its allocation,
    // which gains at least what the budget's own does.
    let take = |lambda: f64| -> Vec<f64> {
        idfs.iter()
            .zip(&need)
            .map(|(&idf, &need)| ((f64::from(idf) * n / lambda).sqrt() - n).clamp(0.0, need))
            .collect()
    };
    let (mut lo, mut hi) = (
        0.0f64,
        idfs.iter().map(|&i| f64::from(i)).fold(0.0, f64::max) / n.max(f64::MIN_POSITIVE),
    );
    for _ in 0..WATER_FILL_STEPS {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        match take(mid).iter().sum::<f64>() >= budget {
            true => lo = mid,
            false => hi = mid,
        }
    }
    let alloc = match lo > 0.0 {
        true => take(lo),
        false => need.iter().map(|&x| x.min(budget)).collect(),
    };
    alloc
        .iter()
        .zip(idfs.iter().zip(caps))
        .map(|(&x, (&idf, &cap))| {
            let value = match x.is_finite() {
                true => f64::from(idf) * x / (x + n),
                false => f64::from(idf),
            };
            value.min(f64::from(cap))
        })
        .sum::<f64>() as f32
}

/// The heads whose next level could lower a bound that still competes:
/// one a slot's unseen bound, or a candidate missing the term, rests on
/// while it is above `θ`.
fn heads_that_matter(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    bounds: &[(f32, f32)],
    theta: f32,
) -> Vec<usize> {
    let mut wanted = vec![false; heads.len()];
    for (slot, &(_, ub)) in slots.iter().zip(bounds) {
        if ub <= theta {
            continue;
        }
        for &h in slot.heads.iter().flatten() {
            wanted[h] = true;
        }
    }
    for (&(s, _), c) in candidates {
        let slot = &slots[s as usize];
        if bounds[s as usize].1 <= theta {
            continue;
        }
        for (t, h) in slot.heads.iter().enumerate() {
            if let Some(h) = *h
                && c.known & (1 << t) == 0
                && term_cap(&heads[h], slot.scale_up, weights, c.len) > 0.0
            {
                wanted[h] = true;
            }
        }
    }
    wanted
        .iter()
        .enumerate()
        .filter(|&(h, &w)| w && !heads[h].done())
        .map(|(h, _)| h)
        .collect()
}

/// Per slot every term's head lists whole, its matching documents with
/// their exact scores: the index answers for the slot outright.
fn resolve(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    required: TermMask,
) -> Vec<(usize, Vec<(u32, f32)>)> {
    let whole: Vec<bool> = slots
        .iter()
        .map(|slot| {
            slot.heads.iter().flatten().next().is_some()
                && slot.heads.iter().flatten().all(|&h| heads[h].whole())
        })
        .collect();
    let mut out: HashMap<usize, Vec<(u32, f32)>> = HashMap::new();
    for (s, w) in whole.iter().enumerate() {
        if *w {
            out.insert(s, Vec::new());
        }
    }
    for (&(s, row), c) in candidates {
        if let Some(hits) = out.get_mut(&(s as usize))
            && c.known & required == required
        {
            hits.push((row, c.sum()));
        }
    }
    let mut out: Vec<(usize, Vec<(u32, f32)>)> = out.into_iter().collect();
    out.sort_unstable_by_key(|(s, _)| *s);
    out
}

/// The highest `_id` among the superfiles holding the `k` candidates that
/// set `theta`, equal candidates taken from the lowest `_id` ranges first;
/// `i128::MAX` when no threshold was found.
fn theta_id_max(
    candidates: &HashMap<(u32, u32), Candidate>,
    slots: &[Slot],
    required: TermMask,
    k: usize,
    lb_slack: f32,
    theta: f32,
) -> i128 {
    if !theta.is_finite() {
        return i128::MAX;
    }
    let mut top: Vec<(f32, i128)> = candidates
        .iter()
        .filter(|(_, c)| c.known & required == required)
        .map(|(&(s, _), c)| (c.sum() * lb_slack, slots[s as usize].id_max))
        .filter(|(lb, _)| *lb >= theta)
        .collect();
    top.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    top.iter().take(k).map(|t| t.1).max().unwrap_or(i128::MAX)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    /// Proptest cases per property.
    const CASES: u32 = 256;

    /// The integer optimum by brute force: every split of up to `tokens`
    /// tokens over the terms.
    fn brute_force(idfs: &[f32], caps: &[f32], norm: f32, tokens: u32) -> f32 {
        fn go(i: usize, left: u32, idfs: &[f32], caps: &[f32], norm: f32) -> f32 {
            if i == idfs.len() {
                return 0.0;
            }
            (0..=left)
                .map(|x| {
                    let value = match x {
                        0 => 0.0,
                        x => (idfs[i] * x as f32 / (x as f32 + norm)).min(caps[i]),
                    };
                    value + go(i + 1, left - x, idfs, caps, norm)
                })
                .fold(0.0, f32::max)
        }
        go(0, tokens, idfs, caps, norm)
    }

    #[test]
    fn every_bucket_holds_its_longest_length() {
        for b in (0..=u8::MAX).filter(|&b| quantize_len(dequantize_len(b)) == b) {
            let max = BUCKET_MAX_LEN[usize::from(b)];
            assert_eq!(quantize_len(max), b, "bucket {b}");
            if max < u32::MAX {
                assert_ne!(quantize_len(max + 1), b, "bucket {b} ends at {max}");
            }
        }
    }

    #[test]
    fn a_term_alone_is_bounded_by_its_cap_or_its_tokens() {
        // One token: tf one.
        let one = water_fill(&[2.0], &[10.0], 1.5, 1);
        assert!((one - 2.0 / 2.5).abs() < 1e-5, "{one}");
        // A cap below what the tokens reach.
        assert_eq!(water_fill(&[2.0], &[0.5], 1.5, 100), 0.5);
        // No norm: any token scores the whole idf.
        assert_eq!(water_fill(&[2.0, 3.0], &[5.0, 5.0], 0.0, 1), 5.0);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        /// The water-filling bounds the integer optimum from above, and
        /// never exceeds the sum of the caps.
        #[test]
        fn water_filling_bounds_the_integer_optimum(
            terms in prop::collection::vec((0.1f32..8.0, 0.0f32..10.0), 1..4),
            norm in 0.05f32..20.0,
            tokens in 1u32..12,
        ) {
            let idfs: Vec<f32> = terms.iter().map(|t| t.0).collect();
            let caps: Vec<f32> = terms.iter().map(|t| t.1).collect();
            let bound = water_fill(&idfs, &caps, norm, tokens);
            let exact = brute_force(&idfs, &caps, norm, tokens);
            prop_assert!(bound >= exact * (1.0 - 1e-5), "bound {bound} < optimum {exact}");
            prop_assert!(bound <= caps.iter().sum::<f32>() * (1.0 + 1e-5));
        }
    }
}
