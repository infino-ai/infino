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
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::{Arc, LazyLock},
    time::Instant,
};

use futures::future::try_join_all;
use roaring::RoaringBitmap;
use tracing::debug;

use super::{
    dispatch,
    exec::common::RowCheck,
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

/// Head bytes a query reads at most before it stops and opens what is
/// still in play: a few seconds of reads on a cold store.
pub(crate) const HEAD_READ_BUDGET_BYTES: u64 = 256 << 20;
/// Most candidates a query probes: each costs, per missing term, a share of
/// its superfile's reads of that term — the header and skip table once,
/// then the blocks covering the candidates — so this bounds a query's
/// probe reads near the read budget of a cold store.
const PROBE_MAX_CANDIDATES: usize = 4_096;
/// Steps of the water-filling's search for its common marginal gain; each
/// halves the interval, so the allocation it bounds with is within a
/// negligible share of the budget it was given.
const WATER_FILL_STEPS: usize = 64;
/// Length buckets: the one-byte codes the length quantizer gives.
const LEN_BUCKETS: usize = 1 << u8::BITS;
/// The largest term frequency a bound searches when a term's per-bucket
/// maxima are unknown: past any tf a document of a length bucket holds.
const TF_SEARCH_MAX: u32 = u32::MAX >> 1;
/// Nanoseconds in a millisecond, for the selection's diagnostic timings.
const NANOS_PER_MS: u64 = 1_000_000;

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
    /// Whether the query's row predicate was checked on the document's row
    /// and admits it; a document it rejects leaves the candidates.
    admitted: bool,
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
    /// Whether the query has a row predicate: only a candidate it was
    /// checked to admit is known to match.
    filtered: bool,
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
        (!self.filtered || c.admitted)
            && present & self.required == self.required
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
/// when the heads cannot serve it: a phrase or a negation, a term with
/// postings in a layer that has no level head, or a superfile the index
/// does not list or whose average length is unknown.
pub(crate) async fn select(
    index: &TermIndex,
    manifest: &ManifestSnapshot,
    column: &str,
    query: &TierQuery<'_>,
    kept: &[Arc<SuperfileEntry>],
    global_idf: &GlobalTermIdf,
) -> Result<Option<TierSelection>, TermIndexError> {
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
        filtered: query.check.is_some(),
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
            id_min: e.id_min,
            id_max: e.id_max,
            same_norms: baked == query.table_avgdl,
        });
    }
    let slot_for = |ordinal: u32| {
        index
            .superfile_id(ordinal)
            .and_then(|id| slot_of.get(&id).copied())
    };
    let runs_per_term = index.runs_many(query.column, terms).await?;
    let runs_ns = started.elapsed().as_nanos() as u64;
    let mut levels_ns = 0u64;
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
    let mut fetched = 0u64;
    let mut probes = 0u64;
    let mut theta;
    let mut bounds;
    // What the last round left: its threshold and the slots it left in play.
    let mut last: Option<(f32, usize)> = None;
    // The host may cap the rounds, for diagnosis; zero is no cap.
    let max_rounds = match config::global().diagnostics.tier_rounds {
        0 => usize::MAX,
        n => n as usize,
    };
    let mut rounds = 0usize;
    loop {
        (theta, bounds) = settle(&slots, &heads, &candidates, &weights, k, lb_slack);
        if fetched >= HEAD_READ_BUDGET_BYTES {
            break;
        }
        // A round that neither raised the threshold nor ruled a slot out
        // ends the reading: on documents that tie across the table the
        // bounds sit at the threshold however deep the heads are read, and
        // the opens settle the ties by `_id` instead; a query of required
        // phrases finds no threshold in the heads, which hold no
        // positions, and its opens, in bound order, find it.
        let in_play = bounds.iter().filter(|(_, ub)| *ub > theta).count();
        let no_threshold_here = weights.phrases.iter().any(|p| p.required);
        if (theta.is_finite() || no_threshold_here) && last == Some((theta, in_play)) {
            break;
        }
        last = Some((theta, in_play));
        // Bounds under the threshold widened by the tolerance are skipped
        // by the opens anyway: reading further for them buys nothing.
        let bar = competitive_floor(theta, query.score_tolerance);
        let wanted = heads_that_matter(&slots, &heads, &candidates, &bounds, bar);
        if wanted.is_empty() || rounds == max_rounds {
            break;
        }
        rounds += 1;
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
        let levels_started = Instant::now();
        let levels = try_join_all(reads).await?;
        levels_ns += levels_started.elapsed().as_nanos() as u64;
        for (&h, (blocks, bytes)) in wanted.iter().zip(levels) {
            fetched += bytes;
            let (t, base) = (heads[h].term, heads[h].base);
            for block in blocks.iter() {
                for e in &block.entries {
                    entries_read += 1;
                    if let Some(s) = slot_for(base + e.superfile) {
                        absorb(&mut candidates, &weights, nt, t, s, e.doc_id, e.tf, e.len);
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
    // A row predicate is checked on the best candidates first: only rows it
    // admits set the threshold.
    let check_started = Instant::now();
    if let Some(check) = &query.check {
        check_candidates(
            manifest,
            kept,
            &slots,
            &heads,
            &weights,
            &mut candidates,
            check,
            theta,
        )
        .await?;
        (theta, bounds) = settle(&slots, &heads, &candidates, &weights, k, lb_slack);
    }
    let check_ns = check_started.elapsed().as_nanos() as u64;
    let probe_started = Instant::now();
    let mut probed_superfiles = 0usize;
    // Probes: candidates whose bounds still reach above the threshold for a
    // term or phrase not yet known for them have it read from the covering
    // posting blocks only; their exact scores raise the threshold, under
    // which the heads bound the unseen documents, and a superfile so
    // settled is answered or ruled out without an open.
    if !query.rows_scoped || query.check.is_some() {
        let plan = probe_plan(&slots, &heads, &candidates, &weights, &postings, theta);
        probed_superfiles = plan.len();
        if !plan.is_empty() {
            let weights = &weights;
            let reads = plan.into_iter().map(|(s, probes)| {
                let entry = &kept[s];
                async move {
                    let reader = open_fts(manifest, entry).await?;
                    let fts = fts_of(&reader);
                    let mut out = Vec::with_capacity(probes.len());
                    for (kind, members, docs) in probes {
                        let read = match kind {
                            ProbeKind::Term(_) => {
                                let (entry, df) = members[0];
                                fts.probe_tfs(column, entry, df, &docs).await
                            }
                            ProbeKind::Phrase(ph) => {
                                fts.probe_phrase_tfs(
                                    column,
                                    &members,
                                    &weights.phrases[ph].offsets,
                                    &docs,
                                )
                                .await
                            }
                        };
                        let tfs = read.map_err(|e| TermIndexError::Open(QueryError::from(e)))?;
                        out.push((kind, docs, tfs));
                    }
                    Ok::<_, TermIndexError>((s, out))
                }
            });
            for (s, probed) in try_join_all(reads).await? {
                probes += probed.len() as u64;
                for (kind, docs, tfs) in probed {
                    for (doc, tf) in docs.into_iter().zip(tfs) {
                        let Some(c) = candidates.get_mut(&(s as u32, doc)) else {
                            continue;
                        };
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
            (theta, bounds) = settle(&slots, &heads, &candidates, weights, k, lb_slack);
        }
    }
    let probe_ns = probe_started.elapsed().as_nanos() as u64;
    // A scope that admits only some rows: the threshold over every row is
    // no floor for the admitted ones, and an answer would hold rows the
    // scope drops. Only the bounds stand.
    if query.rows_scoped && query.check.is_none() {
        theta = f32::NEG_INFINITY;
    }
    let theta_id_max = theta_id_max(&candidates, &slots, &weights, k, lb_slack, theta);
    let mut resolved = match exact_scores && (!query.rows_scoped || query.check.is_some()) {
        true => resolve(&slots, &heads, &candidates, &weights, theta, theta_id_max),
        false => Vec::new(),
    };
    // The heads name documents by each superfile's own id; its answers are
    // its rows, read from the map pages the few documents fall in.
    let mapped = resolved.iter().map(|(s, hits)| {
        let entry = &kept[*s];
        async move {
            if hits.is_empty() {
                return Ok(Vec::new());
            }
            let reader = open_fts(manifest, entry).await?;
            to_rows(fts_of(&reader), hits.iter().copied()).await
        }
    });
    let rows_started = Instant::now();
    let rows = try_join_all(mapped).await?;
    for ((_, hits), rows) in resolved.iter_mut().zip(rows) {
        *hits = rows;
    }
    let (mut below, mut tied, mut above) = (0usize, 0usize, 0usize);
    for &(_, ub) in &bounds {
        match ub.total_cmp(&theta) {
            Ordering::Less => below += 1,
            Ordering::Equal => tied += 1,
            Ordering::Greater => above += 1,
        }
    }
    debug!(
        slots = slots.len(),
        same_norms = slots.iter().filter(|s| s.same_norms).count(),
        below,
        tied,
        above,
        theta,
        resolved = resolved.len(),
        candidates = candidates.len(),
        entries_read,
        head_bytes = fetched,
        rounds,
        probed_superfiles,
        probes,
        runs_ms = runs_ns / NANOS_PER_MS,
        levels_ms = levels_ns / NANOS_PER_MS,
        check_ms = check_ns / NANOS_PER_MS,
        probe_ms = probe_ns / NANOS_PER_MS,
        rows_ms = rows_started.elapsed().as_millis() as u64,
        total_ms = started.elapsed().as_millis() as u64,
        "level heads selection"
    );
    Ok(Some(TierSelection {
        bounds: bounds
            .into_iter()
            .map(|(lb, ub)| SuperfileBounds { lb, ub })
            .collect(),
        theta_lb: theta,
        theta_id_max,
        resolved,
        tier_bytes: fetched,
        entries_read: entries_read + probes,
        merge_ns: started.elapsed().as_nanos() as u64,
    }))
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
    let open = best_open(slots, heads, candidates, weights, theta, |s, c| {
        (!weights.filtered || c.admitted) && !unknowns(s, c).is_empty()
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

/// Check the query's row predicate on its best open candidates' rows (see
/// [`best_open`]), reading only the pages that hold them: a
/// candidate the predicate admits is marked so, one it rejects is dropped.
#[allow(clippy::too_many_arguments)]
async fn check_candidates(
    manifest: &ManifestSnapshot,
    kept: &[Arc<SuperfileEntry>],
    slots: &[Slot],
    heads: &[Head],
    weights: &Weights,
    candidates: &mut HashMap<(u32, u32), Candidate>,
    check: &RowCheck,
    theta: f32,
) -> Result<(), TermIndexError> {
    let mut by_slot: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    for (_, s, doc) in best_open(slots, heads, candidates, weights, theta, |_, c| !c.admitted) {
        by_slot.entry(s).or_default().push(doc);
    }
    let checks = by_slot.into_iter().map(|(s, mut docs)| {
        docs.sort_unstable();
        let entry = &kept[s];
        async move {
            let reader = open_fts(manifest, entry).await?;
            let rows = to_rows(fts_of(&reader), docs.iter().map(|&d| (d, 0.0))).await?;
            let asked: RoaringBitmap = rows.iter().map(|&(row, _)| row).collect();
            let admitted = check
                .keep(&reader, &asked)
                .await
                .map_err(TermIndexError::Open)?;
            let verdicts: Vec<(u32, bool)> = docs
                .iter()
                .zip(&rows)
                .map(|(&doc, &(row, _))| (doc, admitted.contains(row)))
                .collect();
            Ok::<_, TermIndexError>((s, verdicts))
        }
    });
    for (s, verdicts) in try_join_all(checks).await? {
        for (doc, admits) in verdicts {
            match admits {
                true => {
                    if let Some(c) = candidates.get_mut(&(s as u32, doc)) {
                        c.admitted = true;
                    }
                }
                false => {
                    candidates.remove(&(s as u32, doc));
                }
            }
        }
    }
    Ok(())
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
/// best bound first, at most [`PROBE_MAX_CANDIDATES`], as
/// `(bound, slot, document)`; never one of a slot with deletions.
fn best_open(
    slots: &[Slot],
    heads: &[Head],
    candidates: &HashMap<(u32, u32), Candidate>,
    weights: &Weights,
    theta: f32,
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
    open.truncate(PROBE_MAX_CANDIDATES);
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
            admitted: false,
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
                .or_insert_with(|| unseen_bound(slot, heads, weights));
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
    let mut key = f32::INFINITY;
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
        match slot.same_norms {
            // Stored keys are the kernel's: the member's unread documents
            // hold at most the largest tf keying under its bound.
            true => tf = tf.min(max_tf_under(head.unread(), norm, max_tf)),
            false => {
                key = key.min(head.unread() * slot.scale_up * BOUND_SLACK);
                tf = tf.min(max_tf);
            }
        }
    }
    if tf == 0 || key <= 0.0 {
        return 0.0;
    }
    // At most that many occurrences, scored as the kernel scores them, bit
    // for bit: a superfile whose phrase only ties the floor is settled by
    // `_id` order unopened.
    let by_tf = score_with_dl_norm_k1(phrase.idf, tf, norm);
    match key.is_finite() {
        false => by_tf,
        true => (by_tf * BOUND_SLACK).min(phrase.idf * key),
    }
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
            ub += h.map_or(0.0, |h| term_cap(&heads[h], slot.scale_up, weights, c.len));
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
        && slot.same_norms
    {
        return single_term_bound(&heads[h], weights);
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
            let cap = term_cap(&heads[h], slot.scale_up, weights, len);
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

/// The exact bound on an unread document of one term in a superfile whose
/// stored keys are the kernel's own: per length bucket, the largest tf
/// whose key stays under the head's unread bound, scored as the kernel
/// scores it. A document that only ties the threshold then bounds at it
/// bit for bit, and is settled by `_id` order unopened.
fn single_term_bound(head: &Head, weights: &Weights) -> f32 {
    let cap = head.unread();
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
        let tf = max_tf_under(cap, weights.norm[usize::from(len)], max_tf);
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
    let mut whole: Vec<bool> = slots
        .iter()
        .map(|slot| {
            slot.deny.is_none()
                && slot.heads.iter().flatten().next().is_some()
                && (slot.heads.iter().flatten().all(|&h| heads[h].whole())
                    || unseen_bound(slot, heads, weights) < theta)
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
        // A row the predicate was not checked on may or may not be admitted.
        let unchecked = weights.filtered && !c.admitted;
        if missing || phrase_open || unchecked {
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
