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
    collections::{BTreeMap, HashMap},
    sync::{Arc, LazyLock},
    time::Instant,
};

use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tracing::debug;
use uuid::Uuid;

use super::{
    dispatch,
    fts::{baked_avgdl, competitive_floor},
    tier::{
        BOUND_SLACK, EXACT_SUM_TERMS, MAX_TIER_TERMS, SuperfileBounds, TermMask, TierQuery,
        TierSelection,
    },
};
use crate::{
    config,
    superfile::{
        SuperfileReader,
        fts::{
            bm25::{dequantize_len, quantize_len, score_with_dl_norm_k1, tf_factor},
            reader::{FtsReader, GlobalTermIdf, norm_table},
        },
        id_space::FtsDocId,
    },
    supertable::{
        error::QueryError,
        manifest::{
            ManifestSnapshot, SuperfileEntry,
            part::ContentHash,
            term_index::{BucketMaxTf, LevelRef, SliceRun, TermIndex, TermIndexError},
        },
        reader_cache::disk::ReadIntent,
        schema::FieldId,
    },
    utils::terms::DictEntry,
};

/// Head entries a query reads at most before it stops and opens what is
/// still in play. Every entry read becomes a candidate each later round
/// bounds again, so this caps the selection's work whatever the heads hold:
/// a phrase of common words, whose bound its members' keys hold above the
/// threshold however deep they are read, stops here instead of reading
/// hundreds of millions of entries to save no open.
const HEAD_READ_BUDGET_ENTRIES: u64 = 256 << 10;
/// How far a stored key may sit under one recomputed from its superfile's
/// recorded average: a few roundings.
const KEY_TOLERANCE: f32 = 1.0 + 4.0 * f32::EPSILON;
/// Most candidates a query probes: each costs, per missing term, a share of
/// its superfile's reads of that term — the header and skip table once,
/// then the blocks covering the candidates — so this bounds a query's
/// probe reads near the read budget of a cold store.
const PROBE_MAX_CANDIDATES: usize = 4_096;
/// Candidates the first wave of probes reads per result asked for. The
/// best-bounded candidates are the likeliest to hold the query, so a wave
/// this deep finds the threshold when probes can find it at all; when it
/// finds none, or one that can rule no superfile out, the rest are not
/// probed. Measured on a 10B-row table, phrases of common words found no
/// threshold in 4,096 probed candidates and ruled out nothing.
const PROBE_FIRST_WAVE_PER_RESULT: usize = 16;
/// Steps of the water-filling's search for its common marginal gain; each
/// halves the interval, so the allocation it bounds with is within a
/// negligible share of the budget it was given.
const WATER_FILL_STEPS: usize = 64;
/// Length buckets: the one-byte codes the length quantizer gives.
const LEN_BUCKETS: usize = 1 << u8::BITS;
/// The largest term frequency a bound searches when a term's per-bucket
/// maxima are unknown: past any tf a document of a length bucket holds.
const TF_SEARCH_MAX: u32 = u32::MAX >> 1;

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
    /// Per term, its frequency here once known.
    tf: Vec<u32>,
    /// Per bare term, its contribution once known; zero otherwise.
    contrib: Vec<f32>,
    /// Per phrase, its contribution once a probe has read it.
    phrases: Vec<Option<f32>>,
}

impl Candidate {
    fn sum(&self) -> f32 {
        self.contrib.iter().sum::<f32>() + self.phrases.iter().flatten().sum::<f32>()
    }

    /// The terms the document is known to carry: a probe can learn that a
    /// term is absent, which makes it known but not present.
    fn present(&self) -> TermMask {
        self.tf
            .iter()
            .enumerate()
            .filter(|&(t, &tf)| self.known & (1 << t) != 0 && tf > 0)
            .fold(0, |mask, (t, _)| mask | (1 << t))
    }
}

/// One kept superfile: per term, the head its documents of the term lie
/// in (`None` when it has none), how far its stored keys scale at the
/// table's average, and its deleted rows.
struct Slot {
    heads: Vec<Option<usize>>,
    scale_up: f32,
    deny: Option<Arc<RoaringBitmap>>,
    id_min: i128,
    id_max: i128,
    /// Whether the superfile was built at exactly the table's average, so
    /// its stored keys are the kernel's own `tf / (tf + norm)`.
    same_norms: bool,
    /// The length norms its stored keys were computed at: its own average's.
    key_norm: Arc<[f32; LEN_BUCKETS]>,
}

/// One phrase of the query: its members (term indices in order, a
/// repeated word repeated), its weight — the members' idfs summed, every
/// occurrence counted, as the kernel weights it — and whether a match must
/// carry it.
struct Phrase {
    members: Vec<usize>,
    /// Each member's offset from the first.
    offsets: Vec<u32>,
    idf: f32,
    required: bool,
}

/// The query's terms' weights, its shape and the kernel's length norms.
struct Weights {
    idf: Vec<f32>,
    norm: Arc<[f32; LEN_BUCKETS]>,
    /// A bit per term that scores on its own; a phrase member that is not
    /// also a bare term scores only through its phrase.
    scoring: TermMask,
    /// A bit per term a match must carry: the musts and every member of a
    /// must-phrase.
    required: TermMask,
    phrases: Vec<Phrase>,
}

impl Weights {
    /// What term `t` contributes to a document of `tf` occurrences in
    /// bucket `len`: the kernel's own arithmetic.
    fn contribution(&self, t: usize, tf: u32, len: u8) -> f32 {
        score_with_dl_norm_k1(self.idf[t], tf, self.norm[usize::from(len)])
    }

    /// Whether a candidate is known to match, so its known sum is a lower
    /// bound on a match's score: it carries every required term and a bare
    /// term or a phrase at least, and every required phrase a probe has
    /// found in it.
    fn counts(&self, c: &Candidate) -> bool {
        let found = |v: &Option<f32>| matches!(v, Some(v) if *v > 0.0);
        let present = c.present();
        present & self.required == self.required
            && (present & self.scoring != 0 || c.phrases.iter().any(found))
            && self
                .phrases
                .iter()
                .zip(&c.phrases)
                .all(|(p, v)| !p.required || found(v))
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
/// when the heads cannot serve it: a negation, a term with
/// postings in a layer that has no level head, or a superfile the index
/// does not list or whose average length is unknown.
pub(crate) async fn select<'a>(
    index: &Arc<TermIndex>,
    manifest: &'a ManifestSnapshot,
    column: &'a str,
    query: &TierQuery<'_>,
    kept: &'a [Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
) -> Result<Option<Selected<'a>>, TermIndexError> {
    let started = Instant::now();
    let terms = query.terms;
    let nt = terms.len();
    if nt == 0 || nt > MAX_TIER_TERMS || query.k == 0 || kept.is_empty() || query.negated {
        return Ok(None);
    }
    let mut idf = Vec::with_capacity(nt);
    for term in terms {
        let Some(&w) = global_idf.get(*term) else {
            return Ok(None);
        };
        idf.push(w);
    }
    let phrases = query
        .phrases
        .iter()
        .zip(&query.phrase_offsets)
        .map(|((members, required), offsets)| Phrase {
            idf: members.iter().map(|&m| idf[m]).sum(),
            members: members.clone(),
            offsets: offsets.clone(),
            required: *required,
        })
        .collect();
    let weights = Weights {
        idf,
        norm: norm_table(query.table_avgdl, query.params),
        scoring: query.scoring,
        required: query.required,
        phrases,
    };
    let slot_of: HashMap<_, usize> = kept
        .iter()
        .enumerate()
        .map(|(i, e)| (e.superfile_id, i))
        .collect();
    let mut slots = Vec::with_capacity(kept.len());
    let mut key_norms: HashMap<u32, Arc<[f32; LEN_BUCKETS]>> = HashMap::new();
    key_norms.insert(query.table_avgdl.to_bits(), Arc::clone(&weights.norm));
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
            id_min: e.id_min,
            id_max: e.id_max,
            same_norms: baked == query.table_avgdl,
            key_norm: Arc::clone(
                key_norms
                    .entry(baked.to_bits())
                    .or_insert_with(|| norm_table(baked, query.params)),
            ),
        });
    }
    let slot_for = |ordinal: u32| {
        index
            .superfile_id(ordinal)
            .and_then(|id| slot_of.get(&id).copied())
    };
    let runs_per_term = index.runs_many(query.column, terms).await?;
    let mut heads: Vec<Head> = Vec::new();
    // Where each kept superfile's postings of each term sit, for probes.
    let mut postings: HashMap<(usize, usize), (DictEntry, u64)> = HashMap::new();
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
                    if let Some(entry) = p.location.to_dict_value() {
                        postings.insert((s, t), (entry, p.df));
                    }
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
                    absorb(&mut candidates, &weights, nt, t, s, e.doc_id, e.tf, e.len);
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
    let k = query.k;
    // Scores are the kernel's bit for bit up to two scored parts, bare
    // terms and phrases, a phrase's once a probe has read its positions.
    let exact_scores =
        weights.scoring.count_ones() as usize + weights.phrases.len() <= EXACT_SUM_TERMS;
    // Past two terms a candidate's sum may round differently from the
    // kernel's, which adds in its own order.
    let lb_slack = match exact_scores {
        true => 1.0,
        false => 1.0 - nt as f32 * 2.0 * f32::EPSILON,
    };
    let mut state = Selection {
        manifest,
        kept,
        column,
        slots,
        heads,
        candidates,
        weights,
        postings,
        k,
        lb_slack,
        answers: exact_scores && !query.rows_scoped,
        index: Arc::clone(index),
        slot_of,
        score_tolerance: query.score_tolerance,
        // With the table's gate off, the rounds are not priced: every
        // round the bounds call for is read, up to the budget.
        open_cost_ns: match query.merge_gate {
            true => query.open_cost_ns,
            false => 0,
        },
        entry_cost_ns: query.entry_cost_ns,
        rounds: 0,
        entries_read,
        fetched: 0,
    };
    let (mut theta, bounds) = state.read_rounds().await?;
    // A scope that admits only some rows: the threshold over every row is
    // no floor for the admitted ones, and an answer would hold rows the
    // scope drops. Only the bounds stand.
    if query.rows_scoped {
        theta = f32::NEG_INFINITY;
    }
    let now = state
        .finish(
            theta,
            bounds,
            state.fetched,
            state.entries_read,
            started,
            state.rounds,
        )
        .await?;
    // Probes: candidates whose bounds still reach above the threshold for a
    // term or phrase not yet known for them have it read from the covering
    // posting blocks only; their exact scores raise the threshold, under
    // which the heads bound the unseen documents, and a superfile so
    // settled is answered or ruled out without an open. They run beside
    // the opens, not before them: measured on a 10B-row table, probes of
    // phrases and unions of common words ruled out nothing and, run first,
    // added their reads to every such query.
    let probes = match query.rows_scoped {
        true => None,
        false => {
            let ubs: Vec<f32> = now.bounds.iter().map(|b| b.ub).collect();
            let first = k
                .saturating_mul(PROBE_FIRST_WAVE_PER_RESULT)
                .min(PROBE_MAX_CANDIDATES);
            let plan = match state.probes_can_pay(&ubs, now.theta_lb) {
                true => state.probe_plan(now.theta_lb, first),
                false => BTreeMap::new(),
            };
            (!plan.is_empty()).then_some(ProbeStage {
                state,
                plan,
                left: PROBE_MAX_CANDIDATES - first,
            })
        }
    };
    Ok(Some(Selected { now, probes }))
}

/// What the level heads chose: the selection from their reads, and the
/// probes that may tighten it, which the caller runs beside the opens.
pub(crate) struct Selected<'a> {
    /// The selection from the heads' reads alone.
    pub(crate) now: TierSelection,
    /// The candidates to probe, when any can still change the selection.
    pub(crate) probes: Option<ProbeStage<'a>>,
}

/// A selection's state: what it read and what it was asked.
struct Selection<'a> {
    manifest: &'a ManifestSnapshot,
    kept: &'a [Arc<SuperfileEntry>],
    column: &'a str,
    slots: Vec<Slot>,
    heads: Vec<Head>,
    candidates: HashMap<(u32, u32), Candidate>,
    weights: Weights,
    /// Where each kept superfile's postings of each term sit, for probes.
    postings: HashMap<(usize, usize), (DictEntry, u64)>,
    k: usize,
    lb_slack: f32,
    /// Whether the index may answer for a superfile: scores exact and
    /// every row in scope.
    answers: bool,
    /// The index the heads' levels are read from.
    index: Arc<TermIndex>,
    /// Each kept superfile's slot.
    slot_of: HashMap<Uuid, usize>,
    /// The query's score tolerance: bounds under the threshold widened by
    /// it are skipped by the opens anyway.
    score_tolerance: f32,
    /// The table's measured cost of one open, zero when the rounds are
    /// not priced.
    open_cost_ns: u64,
    /// The table's measured cost of one head entry; zero when not known.
    entry_cost_ns: u64,
    /// Rounds of head levels read.
    rounds: usize,
    /// Head entries read.
    entries_read: u64,
    /// Bytes the head levels' reads fetched.
    fetched: u64,
}

impl Selection<'_> {
    /// Whether probing can still change which superfiles open: with no
    /// threshold yet, only probes find one; with one, only while some
    /// superfile in play (bound `ubs` above `theta`) has unseen documents
    /// bounded at or under what the candidates could raise it to, the
    /// `k`-th best candidate bound. A superfile whose unread documents
    /// bound above that stays in play whatever the probes find.
    fn probes_can_pay(&self, ubs: &[f32], theta: f32) -> bool {
        if !theta.is_finite() {
            return true;
        }
        let (slots, heads, weights) = (&self.slots, &self.heads, &self.weights);
        let mut best: Vec<f32> = self
            .candidates
            .iter()
            .map(|(&(s, _), c)| candidate_bound(&slots[s as usize], heads, weights, c))
            .filter(|b| b.is_finite())
            .collect();
        if best.len() < self.k {
            return true;
        }
        let (_, kth, _) = best.select_nth_unstable_by(self.k - 1, |a, b| b.total_cmp(a));
        let reachable = *kth;
        slots
            .iter()
            .zip(ubs)
            .any(|(slot, &ub)| ub > theta && unseen_bound(slot, heads, weights) <= reachable)
    }

    /// The probes of at most `cap` of the best open candidates.
    fn probe_plan(&self, theta: f32, cap: usize) -> BTreeMap<usize, Vec<Probe>> {
        probe_plan(
            &self.slots,
            &self.heads,
            &self.candidates,
            &self.weights,
            &self.postings,
            theta,
            cap,
        )
    }

    /// The selection at `theta` and `bounds`: the superfiles the index
    /// answers for, their rows read from the map pages they fall in.
    async fn finish(
        &self,
        theta: f32,
        bounds: Vec<(f32, f32)>,
        tier_bytes: u64,
        entries_read: u64,
        started: Instant,
        rounds: usize,
    ) -> Result<TierSelection, TermIndexError> {
        let (slots, heads, candidates, weights) =
            (&self.slots, &self.heads, &self.candidates, &self.weights);
        let theta_id_max = theta_id_max(candidates, slots, weights, self.k, self.lb_slack, theta);
        let mut resolved = match self.answers {
            true => resolve(slots, heads, candidates, weights, theta, theta_id_max),
            false => Vec::new(),
        };
        // The heads name documents by each superfile's own id; its answers
        // are its rows, read from the map pages the few documents fall in.
        // Only slots with hits read rows; most resolved slots have none.
        let with_hits: Vec<usize> = (0..resolved.len())
            .filter(|&i| !resolved[i].1.is_empty())
            .collect();
        let mapped = with_hits.iter().map(|&i| {
            let (s, hits) = &resolved[i];
            let entry = &self.kept[*s];
            async move {
                let reader = open_fts(self.manifest, entry).await?;
                to_rows(fts_of(&reader), hits.iter().copied()).await
            }
        });
        let rows = try_join_all(mapped).await?;
        for (i, rows) in with_hits.into_iter().zip(rows) {
            resolved[i].1 = rows;
        }
        let in_play = bounds.iter().filter(|(_, ub)| *ub > theta).count();
        debug!(
            slots = slots.len(),
            in_play,
            theta,
            resolved = resolved.len(),
            candidates = candidates.len(),
            entries_read,
            head_bytes = tier_bytes,
            rounds,
            total_ms = started.elapsed().as_millis() as u64,
            "level heads selection"
        );
        Ok(TierSelection {
            bounds: bounds
                .into_iter()
                .map(|(lb, ub)| SuperfileBounds { lb, ub })
                .collect(),
            theta_lb: theta,
            theta_id_max,
            resolved,
            tier_bytes,
            entries_read,
            merge_ns: started.elapsed().as_nanos() as u64,
        })
    }
}

/// Probes of a selection's open candidates, run beside the opens: their
/// selection replaces the heads' for the superfiles not yet opened.
pub(crate) struct ProbeStage<'a> {
    state: Selection<'a>,
    /// The first wave's probes.
    plan: BTreeMap<usize, Vec<Probe>>,
    /// Candidates a second wave may still probe.
    left: usize,
}

impl ProbeStage<'_> {
    /// Read the probes, fold what they found into the candidates, and
    /// select again.
    pub(crate) async fn run(self) -> Result<TierSelection, TermIndexError> {
        let started = Instant::now();
        let Self {
            mut state,
            plan,
            left,
        } = self;
        let first_superfiles = plan.len();
        let mut probes = state.probe_wave(plan).await?;
        let (rounds, fetched) = (state.rounds, state.fetched);
        // With the threshold the first wave found, the rounds a query of
        // phrases deferred can now rule superfiles out. Any other
        // query read its rounds before the opens.
        let (mut theta, mut bounds) = match state.waits_for_threshold() {
            true => state.read_rounds().await?,
            false => state.settle(),
        };
        let mut waves = 1;
        let mut second_superfiles = 0;
        // The first wave read the best candidates. Where they held too few
        // of the query's documents for a threshold, or one that rules no
        // superfile out, the rest would cost their reads and change no open.
        let ubs: Vec<f32> = bounds.iter().map(|&(_, ub)| ub).collect();
        if theta.is_finite() && state.probes_can_pay(&ubs, theta) {
            let plan = state.probe_plan(theta, left);
            if !plan.is_empty() {
                second_superfiles = plan.len();
                probes += state.probe_wave(plan).await?;
                (theta, bounds) = state.settle();
                waves += 1;
            }
        }
        let (rounds, fetched) = (state.rounds - rounds, state.fetched - fetched);
        debug!(
            waves,
            first_superfiles, second_superfiles, probes, rounds, "level heads probe waves"
        );
        let mut selection = state
            .finish(theta, bounds, fetched, probes, started, waves)
            .await?;
        // The stage's time is mostly its probes, which are not head
        // entries: its count and time stay out of the table's measured
        // cost of reading one, which prices the rounds.
        selection.entries_read = 0;
        selection.merge_ns = 0;
        Ok(selection)
    }
}

impl Selection<'_> {
    /// Whether the query finds no threshold in the heads, which hold no
    /// positions: no candidate is known to match until a probe finds a
    /// phrase in it, when the query requires a phrase or scores no bare
    /// term (see [`Weights::counts`]).
    fn waits_for_threshold(&self) -> bool {
        let w = &self.weights;
        w.phrases.iter().any(|p| p.required) || (w.scoring == 0 && !w.phrases.is_empty())
    }

    /// The slot of the superfile at `ordinal` in the index, when kept.
    fn slot_for(&self, ordinal: u32) -> Option<usize> {
        self.index
            .superfile_id(ordinal)
            .and_then(|id| self.slot_of.get(&id).copied())
    }

    /// Read rounds of head levels while they can still change which
    /// superfiles open, and settle: the threshold and each slot's bounds.
    ///
    /// A query of phrases alone, or one requiring a phrase, finds no
    /// threshold in the heads, which hold no positions, and without one no
    /// bound rules a superfile out:
    /// deeper levels would only reorder the opens, at the cost of bounding
    /// every candidate they add each round. Past its first round it waits
    /// for a threshold, which the probes beside the opens find, and then
    /// reads on.
    async fn read_rounds(&mut self) -> Result<(f32, Vec<(f32, f32)>), TermIndexError> {
        let no_threshold_here = self.waits_for_threshold();
        // What the last round left: its threshold, the slots it left in
        // play, and the largest of their bounds.
        let mut last: Option<(f32, usize, f32)> = None;
        // The host may cap the rounds, for diagnosis; zero is no cap.
        let max_rounds = match config::global().diagnostics.tier_rounds {
            0 => usize::MAX,
            n => n as usize,
        };
        loop {
            let (theta, bounds) = self.settle();
            if no_threshold_here && !theta.is_finite() && self.rounds > 0 {
                return Ok((theta, bounds));
            }
            // A round that neither raised the threshold, nor ruled a slot
            // out, nor lowered the largest bound still in play ends the
            // reading: on documents that tie across the table the bounds
            // sit at the threshold however deep the heads are read, and the
            // opens settle the ties by `_id` instead. A round that only
            // lowered the bounds is progress: the next may take them under
            // the threshold, and the price below decides whether it is
            // worth reading.
            let in_play = bounds.iter().filter(|(_, ub)| *ub > theta).count();
            let top = bounds
                .iter()
                .map(|&(_, ub)| ub)
                .filter(|ub| *ub > theta)
                .fold(f32::NEG_INFINITY, f32::max);
            if (theta.is_finite() || no_threshold_here) && last == Some((theta, in_play, top)) {
                return Ok((theta, bounds));
            }
            last = Some((theta, in_play, top));
            // Bounds under the threshold widened by the tolerance are
            // skipped by the opens anyway: reading further for them buys
            // nothing.
            let bar = competitive_floor(theta, self.score_tolerance);
            let wanted =
                heads_that_matter(&self.slots, &self.heads, &self.candidates, &bounds, bar);
            if wanted.is_empty() || self.rounds == max_rounds {
                return Ok((theta, bounds));
            }
            // The round's price: its entries, and the candidates every
            // later round bounds again. Past the budget, or where it costs
            // more than every open still in play at this table's measured
            // costs, the opens settle the rest instead.
            let heads = &self.heads;
            let round_entries: u64 = wanted
                .iter()
                .map(|&h| u64::from(heads[h].levels[heads[h].next].n_entries))
                .sum();
            let price = RoundPrice {
                entries_read: self.entries_read,
                round_entries,
                candidates: self.candidates.len() as u64,
                in_play: in_play as u64,
                open_cost_ns: self.open_cost_ns,
                entry_cost_ns: self.entry_cost_ns,
            };
            if !price.worth_reading() {
                return Ok((theta, bounds));
            }
            self.rounds += 1;
            let index = &self.index;
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
            let nt = self.weights.idf.len();
            for (&h, (blocks, bytes)) in wanted.iter().zip(levels) {
                self.fetched += bytes;
                let (t, base) = (self.heads[h].term, self.heads[h].base);
                for block in blocks.iter() {
                    for e in &block.entries {
                        self.entries_read += 1;
                        if let Some(s) = self.slot_for(base + e.superfile) {
                            absorb(
                                &mut self.candidates,
                                &self.weights,
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
                let head = &mut self.heads[h];
                head.next += 1;
                if let Some(last) = blocks.last() {
                    head.frontier = last.last_key;
                }
            }
        }
    }

    /// The threshold and each slot's bounds from the candidates now.
    fn settle(&self) -> (f32, Vec<(f32, f32)>) {
        settle(
            &self.slots,
            &self.heads,
            &self.candidates,
            &self.weights,
            self.k,
            self.lb_slack,
        )
    }

    /// Read one wave of probes, each superfile's beside the others', and
    /// fold what they found into the candidates; the number of probes read.
    async fn probe_wave(
        &mut self,
        plan: BTreeMap<usize, Vec<Probe>>,
    ) -> Result<u64, TermIndexError> {
        let (manifest, kept, column) = (self.manifest, self.kept, self.column);
        let phrases = &self.weights.phrases;
        let reads = plan.into_iter().map(|(s, probes)| {
            let entry = &kept[s];
            async move {
                let reader = open_fts(manifest, entry).await?;
                let fts = fts_of(&reader);
                let reads = probes.into_iter().map(|(kind, members, docs)| async move {
                    let read = match kind {
                        ProbeKind::Term(_) => {
                            let (entry, df) = members[0];
                            fts.probe_tfs(column, entry, df, &docs).await
                        }
                        ProbeKind::Phrase(ph) => {
                            fts.probe_phrase_tfs(column, &members, &phrases[ph].offsets, &docs)
                                .await
                        }
                    };
                    let tfs = read.map_err(|e| TermIndexError::Open(QueryError::from(e)))?;
                    Ok::<_, TermIndexError>((kind, docs, tfs))
                });
                Ok::<_, TermIndexError>((s, try_join_all(reads).await?))
            }
        });
        let mut probes = 0u64;
        for (s, probed) in try_join_all(reads).await? {
            probes += probed.len() as u64;
            for (kind, docs, tfs) in probed {
                for (doc, tf) in docs.into_iter().zip(tfs) {
                    let Some(c) = self.candidates.get_mut(&(s as u32, doc)) else {
                        continue;
                    };
                    let weights = &self.weights;
                    match kind {
                        ProbeKind::Term(t) => {
                            c.known |= 1 << t;
                            c.tf[t] = tf;
                            if tf > 0 && weights.scoring & (1 << t) != 0 {
                                c.contrib[t] = weights.contribution(t, tf, c.len);
                            }
                        }
                        ProbeKind::Phrase(ph) => {
                            let norm = weights.norm[usize::from(c.len)];
                            c.phrases[ph] = Some(match tf {
                                0 => 0.0,
                                tf => score_with_dl_norm_k1(weights.phrases[ph].idf, tf, norm),
                            });
                        }
                    }
                }
            }
        }
        Ok(probes)
    }
}

/// What a round of head reads would cost and could save.
struct RoundPrice {
    /// Entries the query has read so far.
    entries_read: u64,
    /// Entries the round would read.
    round_entries: u64,
    /// Candidates every later round bounds again.
    candidates: u64,
    /// Superfiles still in play: the opens the round could save.
    in_play: u64,
    /// The table's measured cost of one open; zero when not yet known.
    open_cost_ns: u64,
    /// The table's measured cost of one entry; zero when not yet known.
    entry_cost_ns: u64,
}

impl RoundPrice {
    /// Whether the round is worth reading: never past
    /// [`HEAD_READ_BUDGET_ENTRIES`], and only while it costs no more than
    /// every open still in play at this table's measured costs — the first
    /// round, a block per head, included. On a table whose opens are cheap
    /// (warm, few superfiles opened together) a many-term query's first
    /// round costs more than the opens it could save, and the opens then
    /// settle the query alone. Until both costs are measured every round is
    /// read.
    fn worth_reading(&self) -> bool {
        if self.entries_read + self.round_entries > HEAD_READ_BUDGET_ENTRIES {
            return false;
        }
        if self.open_cost_ns == 0 || self.entry_cost_ns == 0 {
            return true;
        }
        (self.round_entries + self.candidates).saturating_mul(self.entry_cost_ns)
            <= self.in_play.saturating_mul(self.open_cost_ns)
    }
}

/// What one probe reads in one superfile.
#[derive(Debug, Clone, Copy)]
enum ProbeKind {
    /// A bare term's frequency.
    Term(usize),
    /// A phrase's occurrences.
    Phrase(usize),
}

/// One superfile's probe: what it reads, the members' dictionary values
/// and document counts there, and the candidates it reads them for.
type Probe = (ProbeKind, Vec<(DictEntry, u64)>, Vec<u32>);

/// Per superfile to probe, its probes: the best open candidates (see
/// [`best_open`]) a bare term or a phrase is still unknown for, each for
/// the unknowns its superfile holds postings of. Their exact scores raise
/// the threshold, under which the heads then bound the unseen documents.
fn probe_plan(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    postings: &HashMap<(usize, usize), (DictEntry, u64)>,
    theta: f32,
    cap: usize,
) -> BTreeMap<usize, Vec<Probe>> {
    let held = |s: usize, t: usize| postings.contains_key(&(s, t));
    // What is unknown for a candidate that its superfile holds postings of.
    let unknowns = |s: usize, c: &Candidate| -> Vec<ProbeKind> {
        let terms = (0..slots[s].heads.len())
            .filter(|&t| c.known & (1 << t) == 0 && weights.scoring & (1 << t) != 0 && held(s, t))
            .map(ProbeKind::Term);
        let phrases = weights
            .phrases
            .iter()
            .enumerate()
            .filter(|(ph, p)| c.phrases[*ph].is_none() && p.members.iter().all(|&m| held(s, m)))
            .map(|(ph, _)| ProbeKind::Phrase(ph));
        terms.chain(phrases).collect()
    };
    let open = best_open(slots, heads, candidates, weights, theta, cap, |s, c| {
        !unknowns(s, c).is_empty()
    });
    let mut docs: BTreeMap<(usize, usize, bool), Vec<u32>> = BTreeMap::new();
    for (_, s, doc) in open {
        for kind in unknowns(s, &candidates[&(s as u32, doc)]) {
            let key = match kind {
                ProbeKind::Term(t) => (s, t, false),
                ProbeKind::Phrase(ph) => (s, ph, true),
            };
            docs.entry(key).or_default().push(doc);
        }
    }
    let mut plan: BTreeMap<usize, Vec<Probe>> = BTreeMap::new();
    for ((s, i, phrase), mut list) in docs {
        list.sort_unstable();
        let (kind, members) = match phrase {
            false => (ProbeKind::Term(i), vec![postings[&(s, i)]]),
            true => (
                ProbeKind::Phrase(i),
                weights.phrases[i]
                    .members
                    .iter()
                    .map(|&m| postings[&(s, m)])
                    .collect(),
            ),
        };
        plan.entry(s).or_default().push((kind, members, list));
    }
    plan
}

/// `entry`'s reader, opened through the reader cache, for its full-text
/// index.
async fn open_fts(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
) -> Result<Arc<SuperfileReader>, TermIndexError> {
    let store = Arc::clone(&manifest.options.store);
    let reader = dispatch::open_reader(
        &store,
        manifest.options.disk_cache.as_ref(),
        manifest.options.storage.as_ref(),
        entry,
        ReadIntent::Stream,
    )
    .await
    .map_err(TermIndexError::Open)?;
    match reader.fts().is_some() {
        true => Ok(reader),
        false => Err(TermIndexError::Open(QueryError::Internal(
            "a superfile the heads name has no full-text index".into(),
        ))),
    }
}

/// The full-text index of a reader [`open_fts`] opened.
fn fts_of(reader: &SuperfileReader) -> &FtsReader {
    reader.fts().expect("open_fts checked the full-text index")
}

/// Each `(document, score)`'s row, in order: read from the map pages the
/// documents fall in.
async fn to_rows(
    fts: &FtsReader,
    hits: impl Iterator<Item = (u32, f32)>,
) -> Result<Vec<(u32, f32)>, TermIndexError> {
    let docs = hits
        .map(|(doc, score)| (FtsDocId::new(doc), score))
        .collect();
    Ok(fts
        .hits_to_rows_async(docs)
        .await
        .map_err(|e| TermIndexError::Open(QueryError::from(e)))?
        .into_iter()
        .map(|(row, score)| (row.get(), score))
        .collect())
}

/// The candidates `keep` accepts whose bounds still reach above `theta`,
/// best bound first, at most `cap` of them, as
/// `(bound, slot, document)`; never one of a slot with deletions.
fn best_open(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    theta: f32,
    cap: usize,
    keep: impl Fn(usize, &Candidate) -> bool,
) -> Vec<(f32, usize, u32)> {
    let mut open: Vec<(f32, usize, u32)> = candidates
        .iter()
        .filter(|((s, _), c)| slots[*s as usize].deny.is_none() && keep(*s as usize, c))
        .map(|(&(s, doc), c)| {
            (
                candidate_bound(&slots[s as usize], heads, weights, c),
                s as usize,
                doc,
            )
        })
        .filter(|(ub, _, _)| *ub > theta)
        .collect();
    open.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    open.truncate(cap);
    open
}

/// Add term `t`'s entry for document `doc` of slot `s` to the
/// candidates. A head names documents by the superfile's own id, which a
/// slot's deleted rows are not keyed by: a slot with deletions is bounded
/// by its candidates but sets no threshold and is answered by its open.
#[allow(clippy::too_many_arguments)]
fn absorb(
    candidates: &mut HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    nt: usize,
    t: usize,
    s: usize,
    doc: u32,
    tf: u32,
    len: u8,
) {
    let c = candidates
        .entry((s as u32, doc))
        .or_insert_with(|| Candidate {
            len,
            known: 0,
            tf: vec![0; nt],
            contrib: vec![0.0; nt],
            phrases: vec![None; weights.phrases.len()],
        });
    if c.known & (1 << t) == 0 {
        c.known |= 1 << t;
        c.tf[t] = tf;
        if weights.scoring & (1 << t) != 0 {
            c.contrib[t] = weights.contribution(t, tf, len);
        }
    }
}

/// The threshold and each slot's bounds from what the candidates hold now.
fn settle(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    k: usize,
    lb_slack: f32,
) -> (f32, Vec<(f32, f32)>) {
    (
        kth_lower_bound(candidates, slots, weights, k, lb_slack),
        slot_bounds(slots, heads, candidates, weights, lb_slack),
    )
}

/// The k-th best lower bound among candidates carrying every required
/// term; `NEG_INFINITY` when fewer than `k` do.
fn kth_lower_bound(
    candidates: &HashMap<(u32, u32), Candidate>,
    slots: &[Slot],
    weights: &Weights,
    k: usize,
    lb_slack: f32,
) -> f32 {
    let mut lbs: Vec<f32> = candidates
        .iter()
        .filter(|((s, _), c)| slots[*s as usize].deny.is_none() && weights.counts(c))
        .map(|(_, c)| c)
        .map(|c| c.sum() * lb_slack)
        .collect();
    if lbs.len() < k {
        return f32::NEG_INFINITY;
    }
    let (_, kth, _) = lbs.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    *kth
}

/// The most occurrences of `head`'s term an unread document of bucket `len`
/// in `slot` can hold: the largest tf whose key, at the norms the superfile
/// was built at, stays under the head's unread bound, and at most `max_tf`.
fn unread_tf(head: &Head, slot: &Slot, len: u8, max_tf: u32) -> u32 {
    // A key recomputed from an average read back from the manifest may
    // differ from the stored one by a rounding.
    let cap = match slot.same_norms {
        true => head.unread(),
        false => head.unread() * KEY_TOLERANCE,
    };
    max_tf_under(cap, slot.key_norm[usize::from(len)], max_tf)
}

/// Per term of a slot, what an unread entry of the term can still
/// contribute in bucket `len`: its head's unread key at the slot's scale,
/// held under the term's largest tf in the bucket. Zero for a term the slot
/// has no documents of, or none of that length.
fn term_cap(head: &Head, slot: &Slot, weights: &Weights, len: u8) -> f32 {
    let idf = weights.idf[head.term];
    let by_key = idf * head.unread() * slot.scale_up * BOUND_SLACK;
    let max_tf = match &head.buckets {
        None => TF_SEARCH_MAX,
        Some(buckets) => match buckets.get(len) {
            None => return 0.0,
            Some(max_tf) => max_tf,
        },
    };
    let tf = unread_tf(head, slot, len, max_tf);
    by_key.min(weights.contribution(head.term, tf, len) * BOUND_SLACK)
}

/// Unseen bounds by slot shape: slots of one layer built at one average read
/// the same heads at the same norms.
type ShapeBounds<'a> = HashMap<(&'a [Option<usize>], *const [f32; LEN_BUCKETS]), f32>;

/// `unseen_bound` for `slot`, computed once per shape.
fn cached_unseen_bound<'a>(
    by_shape: &mut ShapeBounds<'a>,
    slot: &'a Slot,
    heads: &[Head],
    weights: &Weights,
) -> f32 {
    *by_shape
        .entry((&slot.heads, Arc::as_ptr(&slot.key_norm)))
        .or_insert_with(|| unseen_bound(slot, heads, weights))
}

/// Per slot, `(lb, ub)`: the best lower bound among its candidates
/// carrying every required term, and a bound on any of its documents;
/// `NEG_INFINITY` for a slot that holds no match.
fn slot_bounds(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    lb_slack: f32,
) -> Vec<(f32, f32)> {
    let mut by_shape = ShapeBounds::new();
    let mut out: Vec<(f32, f32)> = slots
        .iter()
        .map(|slot| {
            let unseen = cached_unseen_bound(&mut by_shape, slot, heads, weights);
            (f32::NEG_INFINITY, unseen)
        })
        .collect();
    for (&(s, _), c) in candidates {
        let slot = &slots[s as usize];
        let (lb, ub) = &mut out[s as usize];
        if slot.deny.is_none() && weights.counts(c) {
            *lb = lb.max(c.sum() * lb_slack);
        }
        *ub = ub.max(candidate_bound(slot, heads, weights, c));
    }
    out
}

/// What a head's unread entries can be in bucket `len` at a slot's scale:
/// a bound on their key, and the largest tf the term has there when known.
/// `None` when the term has no document of that length.
fn unread_facts(head: &Head, scale_up: f32, len: u8) -> Option<(f32, Option<u32>)> {
    let key = head.unread() * scale_up * BOUND_SLACK;
    match &head.buckets {
        None => Some((key, None)),
        Some(buckets) => buckets.get(len).map(|tf| (key, Some(tf))),
    }
}

/// A bound on what `phrase` adds to a document of bucket `len` in `slot`:
/// its weight times its occurrences' key, which is at most every member's
/// key, with occurrences no more than any member's tf and the length over
/// the member count. A candidate's known members give their tfs; the rest
/// bound by their heads. Zero when the document cannot hold the phrase.
fn phrase_cap(
    phrase: &Phrase,
    slot: &Slot,
    heads: &[Head],
    weights: &Weights,
    len: u8,
    seen: Option<&Candidate>,
) -> f32 {
    let norm = weights.norm[usize::from(len)];
    let mut tf = BUCKET_MAX_LEN[usize::from(len)] / phrase.members.len().max(1) as u32;
    for &m in &phrase.members {
        if let Some(c) = seen
            && c.known & (1 << m) != 0
        {
            tf = tf.min(c.tf[m]);
            continue;
        }
        let Some(h) = slot.heads[m] else {
            return 0.0;
        };
        let head = &heads[h];
        let max_tf = match &head.buckets {
            None => TF_SEARCH_MAX,
            Some(buckets) => match buckets.get(len) {
                None => return 0.0,
                Some(max_tf) => max_tf,
            },
        };
        // The member's unread documents hold at most the largest tf keying
        // under its bound at the norms the superfile was built at.
        tf = tf.min(unread_tf(head, slot, len, max_tf));
    }
    if tf == 0 {
        return 0.0;
    }
    // At most that many occurrences, scored as the kernel scores them, bit
    // for bit: a superfile whose phrase only ties the floor is settled by
    // `_id` order unopened.
    score_with_dl_norm_k1(phrase.idf, tf, norm)
}

/// Whether term `t` can still be found for a document of bucket `len` in
/// `slot` that no read entry of it named.
fn may_hold(slot: &Slot, heads: &[Head], t: usize, len: u8) -> bool {
    slot.heads[t]
        .and_then(|h| unread_facts(&heads[h], slot.scale_up, len))
        .is_some_and(|(key, _)| key > 0.0)
}

/// A bound on a candidate's score: its known contributions, per bare term
/// not yet seen for it what the term's unread entries can still add, and
/// per phrase its cap. `NEG_INFINITY` when the candidate can no longer
/// carry a required term or phrase.
fn candidate_bound(slot: &Slot, heads: &[Head], weights: &Weights, c: &Candidate) -> f32 {
    // A required term a probe found absent: the document cannot match.
    if c.known & weights.required & !c.present() != 0 {
        return f32::NEG_INFINITY;
    }
    // Every bare term seen and every phrase probed: the known sum is the
    // score, the kernel's own bit for bit up to two scored parts.
    if c.known & weights.scoring == weights.scoring
        && c.phrases.iter().all(Option::is_some)
        && weights.scoring.count_ones() as usize + weights.phrases.len() <= EXACT_SUM_TERMS
    {
        return c.sum();
    }
    let mut ub = c.sum() * BOUND_SLACK;
    for (t, h) in slot.heads.iter().enumerate() {
        if c.known & (1 << t) != 0 {
            continue;
        }
        if weights.required & (1 << t) != 0 && !may_hold(slot, heads, t, c.len) {
            return f32::NEG_INFINITY;
        }
        if weights.scoring & (1 << t) != 0 {
            ub += h.map_or(0.0, |h| term_cap(&heads[h], slot, weights, c.len));
        }
    }
    for (phrase, known) in weights.phrases.iter().zip(&c.phrases) {
        // A probed phrase is in the sum already.
        let cap = match known {
            Some(v) => *v,
            None => phrase_cap(phrase, slot, heads, weights, c.len, Some(c)),
        };
        if cap <= 0.0 && phrase.required {
            return f32::NEG_INFINITY;
        }
        if known.is_none() {
            ub += cap;
        }
    }
    ub
}

/// A bound on any document of `slot` that no head has named yet;
/// `NEG_INFINITY` when none can match: a required term the slot lacks, or
/// one whose head was read whole.
fn unseen_bound(slot: &Slot, heads: &[Head], weights: &Weights) -> f32 {
    for (t, h) in slot.heads.iter().enumerate() {
        if weights.required & (1 << t) != 0 && h.is_none_or(|h| heads[h].whole()) {
            return f32::NEG_INFINITY;
        }
    }
    let bare: Vec<usize> = slot
        .heads
        .iter()
        .enumerate()
        .filter(|(t, _)| weights.scoring & (1 << t) != 0)
        .filter_map(|(_, h)| *h)
        .collect();
    if bare.is_empty() && weights.phrases.is_empty() {
        return f32::NEG_INFINITY;
    }
    if let [h] = bare[..]
        && weights.phrases.is_empty()
    {
        return single_term_bound(&heads[h], slot, weights);
    }
    let mut best = f32::NEG_INFINITY;
    let mut caps = Vec::with_capacity(bare.len());
    let mut idfs = Vec::with_capacity(bare.len());
    'bucket: for len in 0..LEN_BUCKETS {
        let len = len as u8;
        let mut phrases = 0.0f32;
        for phrase in &weights.phrases {
            let cap = phrase_cap(phrase, slot, heads, weights, len, None);
            if cap <= 0.0 && phrase.required {
                continue 'bucket;
            }
            phrases += cap;
        }
        caps.clear();
        idfs.clear();
        for &h in &bare {
            let cap = term_cap(&heads[h], slot, weights, len);
            if cap > 0.0 {
                caps.push(cap);
                idfs.push(weights.idf[heads[h].term]);
            }
        }
        if caps.is_empty() && phrases <= 0.0 {
            continue;
        }
        let terms = match caps.is_empty() {
            true => 0.0,
            false => water_fill(
                &idfs,
                &caps,
                weights.norm[usize::from(len)],
                BUCKET_MAX_LEN[usize::from(len)],
            ),
        };
        best = best.max((terms + phrases) * BOUND_SLACK);
    }
    best
}

/// The exact bound on an unread document of one term: per length bucket,
/// the largest tf whose key, at the norms its superfile was built at, stays
/// under the head's unread bound, scored as the kernel scores it at the
/// table's. A document that only ties the threshold then bounds at it bit
/// for bit, and is settled by `_id` order unopened.
fn single_term_bound(head: &Head, slot: &Slot, weights: &Weights) -> f32 {
    let mut best = f32::NEG_INFINITY;
    for len in 0..LEN_BUCKETS {
        let len = len as u8;
        let max_tf = match &head.buckets {
            None => TF_SEARCH_MAX,
            Some(buckets) => match buckets.get(len) {
                None => continue,
                Some(tf) => tf,
            },
        };
        let tf = unread_tf(head, slot, len, max_tf);
        if tf > 0 {
            best = best.max(weights.contribution(head.term, tf, len));
        }
    }
    best
}

/// The largest tf up to `max_tf` whose key `tf / (tf + norm)` stays at or
/// under `cap`; zero when even one occurrence keys above it. The key
/// rises with tf.
fn max_tf_under(cap: f32, norm: f32, max_tf: u32) -> u32 {
    if max_tf == 0 || tf_factor(1, norm) > cap {
        return 0;
    }
    let (mut lo, mut hi) = (1u32, max_tf);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        match tf_factor(mid, norm) <= cap {
            true => lo = mid,
            false => hi = mid - 1,
        }
    }
    lo
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
                && may_hold(slot, heads, t, c.len)
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
    weights: &Weights,
    theta: f32,
    theta_id_max: i128,
) -> Vec<(usize, Vec<(u32, f32)>)> {
    // A slot is answered whole when every head lists it whole, or when its
    // unseen documents bound strictly under the threshold and probes have
    // filled in every candidate's terms.
    let mut by_shape = ShapeBounds::new();
    let mut whole: Vec<bool> = slots
        .iter()
        .map(|slot| {
            slot.deny.is_none()
                && slot.heads.iter().flatten().next().is_some()
                && (slot.heads.iter().flatten().all(|&h| heads[h].whole())
                    || cached_unseen_bound(&mut by_shape, slot, heads, weights) < theta)
        })
        .collect();
    for (&(s, _), c) in candidates {
        let slot = &slots[s as usize];
        // A candidate bounded under the threshold cannot place, known or not.
        if !whole[s as usize] || candidate_bound(slot, heads, weights, c) < theta {
            continue;
        }
        let missing = slot.heads.iter().enumerate().any(|(t, h)| {
            // A term whose head is read whole is absent where unseen.
            h.is_some_and(|h| !heads[h].whole())
                && weights.scoring & (1 << t) != 0
                && c.known & (1 << t) == 0
        });
        // A phrase not probed is unknown where its members may both occur.
        let phrase_open = weights
            .phrases
            .iter()
            .zip(&c.phrases)
            .any(|(p, v)| v.is_none() && p.members.iter().all(|&m| slot.heads[m].is_some()));
        if missing || phrase_open {
            whole[s as usize] = false;
        }
    }
    let mut out: HashMap<usize, Vec<(u32, f32)>> = HashMap::new();
    for (s, w) in whole.iter().enumerate() {
        if *w {
            out.insert(s, Vec::new());
        }
    }
    for (&(s, row), c) in candidates {
        // Only what can still place: a document below the threshold cannot,
        // nor one that only ties it in a superfile whose `_id`s all lie past
        // the threshold's documents', which rank ahead of it.
        let score = c.sum();
        let places = score > theta || (score == theta && slots[s as usize].id_min <= theta_id_max);
        if let Some(hits) = out.get_mut(&(s as usize))
            && weights.counts(c)
            && places
        {
            hits.push((row, score));
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
    weights: &Weights,
    k: usize,
    lb_slack: f32,
    theta: f32,
) -> i128 {
    if !theta.is_finite() {
        return i128::MAX;
    }
    let mut top: Vec<(f32, i128)> = candidates
        .iter()
        .filter(|((s, _), c)| slots[*s as usize].deny.is_none() && weights.counts(c))
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
    use crate::superfile::fts::bm25::Bm25Params;

    /// Proptest cases per property.
    const CASES: u32 = 256;

    /// A round priced at the given work and costs.
    fn round(
        round_entries: u64,
        candidates: u64,
        in_play: u64,
        open: u64,
        entry: u64,
    ) -> RoundPrice {
        RoundPrice {
            entries_read: 0,
            round_entries,
            candidates,
            in_play,
            open_cost_ns: open,
            entry_cost_ns: entry,
        }
    }

    /// A slot of one superfile reading `heads[0]`, built at `baked` under a
    /// table whose average is `table`.
    fn slot(baked: f32, table: f32) -> Slot {
        Slot {
            heads: vec![Some(0)],
            scale_up: (table / baked).max(1.0),
            deny: None,
            id_min: 0,
            id_max: 0,
            same_norms: baked == table,
            key_norm: norm_table(baked, Bm25Params::default()),
        }
    }

    /// Superfiles built at averages above, at and below the table's bound
    /// an unread document of one term at the best score any document can
    /// reach whose key, at its own build norms, stays under the head's
    /// bound: a tie with the threshold bounds at it, and the `_id` order
    /// settles it unopened.
    #[test]
    fn a_superfile_bounds_its_unseen_documents_exactly_at_its_own_average() {
        /// The largest tf of the term in any length, so the brute force
        /// below searches all the bound may use.
        const MAX_TF: u32 = 4096;
        let table = 2.5;
        let mut buckets = BucketMaxTf::default();
        for len in 0..=u8::MAX {
            buckets.observe(len, MAX_TF);
        }
        let weights = Weights {
            idf: vec![3.0],
            norm: norm_table(table, Bm25Params::default()),
            scoring: 1,
            required: 0,
            phrases: Vec::new(),
        };
        for baked in [2.0f32, 2.5, 3.1] {
            let slot = slot(baked, table);
            // The head's next entry: one occurrence in a one-token title.
            let key = tf_factor(1, slot.key_norm[1]);
            let head = Head {
                term: 0,
                object: None,
                base: 0,
                levels: Vec::new(),
                next: 0,
                frontier: key,
                floor: key,
                buckets: Some(buckets.clone()),
            };
            let heads = [head];
            let bounds = slot_bounds(
                std::slice::from_ref(&slot),
                &heads,
                &HashMap::new(),
                &weights,
                1.0,
            );
            let cap = match baked == table {
                true => key,
                false => key * KEY_TOLERANCE,
            };
            let oracle = (0..=u8::MAX)
                .flat_map(|len| (1..=MAX_TF).map(move |tf| (len, tf)))
                .filter(|&(len, tf)| tf_factor(tf, slot.key_norm[usize::from(len)]) <= cap)
                .map(|(len, tf)| weights.contribution(0, tf, len))
                .fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(bounds[0].1, oracle, "built at {baked}");
        }
    }

    #[test]
    fn a_round_is_read_while_it_costs_less_than_the_opens_it_can_save() {
        // 1,000 entries at 100 ns against 10 opens at 1 ms: read.
        assert!(round(1_000, 0, 10, 1_000_000, 100).worth_reading());
        // The candidates already held are bounded again: priced in.
        assert!(!round(1_000, 200_000, 10, 1_000_000, 100).worth_reading());
        // Warm opens of microseconds: a large round is not worth them.
        assert!(!round(100_000, 0, 200, 5_000, 100).worth_reading());
        // Nor is a first round, nothing read and no candidate held: ten
        // terms' blocks over twenty layers at a microsecond an entry,
        // against 48 warm opens of 20 µs that run together.
        assert!(!round(12_800, 0, 48, 20_000, 1_000).worth_reading());
        // Costs not yet measured: read.
        assert!(round(100_000, 0, 1, 0, 100).worth_reading());
        assert!(round(100_000, 0, 1, 1_000, 0).worth_reading());
    }

    #[test]
    fn no_round_is_read_past_the_budget() {
        let mut past = round(2, 0, 1_000, u64::MAX, 1);
        past.entries_read = HEAD_READ_BUDGET_ENTRIES - 1;
        assert!(!past.worth_reading(), "past the budget nothing is read");
        past.round_entries = 1;
        assert!(past.worth_reading(), "up to the budget it is");
    }

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
