// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! MVCC OPANN maintenance for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable. The hidden index is a
//! derived, cell-ordered acceleration layer maintained with OPANN-style
//! logical updates expressed as append/MVCC physical swaps:
//!
//!   1. Assign incoming vectors to nearest manifest centroids with zero GETs.
//!   2. For each touched cell only: append one delta superfile (no GETs).
//!   3. Compaction merges multiple small IVF superfiles per cell toward one packed
//!      base via the standard `merge_superfiles` path.
//!   4. Locally refresh touched cell centroids and counts.
//!   5. Split overflow cells in place: partition an over-cap cell into K
//!      children (Sq8+ε capacitated k-means; K self-tuned upward from
//!      ⌈rows/cap⌉ until each child's rows route to it at nprobe=1),
//!      append the K children as one packed superfile (child 0 keeps the cell
//!      id, the rest appended), and mark the parent cell superseded in the
//!      manifest — no rewrite, no removal at split time.
//!   6. Readers, per-cell counts, merges, and split selection all skip the
//!      superseded parent; its blocks are logically dead.
//!   7. Compaction later reclaims the superseded parent's dead blocks.
//!
//! Split stays on stored Sq8+ε bytes. Row assignment dequantizes
//! manifest centroids and rows to fp32 before [`distance`]; rows are
//! re-spliced with [`encode_encoded_rows`], never decoded to full fp32 corpora.

use std::{
    cmp::Ordering,
    collections::HashMap,
    sync::{Mutex, PoisonError},
};

use crate::{
    config,
    superfile::vector::{
        cell_posting::{
            EncodedCellRow, MaterializedIvfRow, dequantize_sq8_residual_into,
            manifest_centroid_components_from_row,
        },
        distance::{
            Metric, distance, nearest_k_centroids_transposed, normalize, relative_score_window,
        },
        kmeans::{kmeans, kmeans_pp},
        reservoir::Reservoir,
        spill::SpilledCellRows,
    },
    supertable::{
        error::BuildError,
        manifest::{
            ClusterCentroids, RABITQ_ADMIT_CELL_SHORTLIST_FRACTION,
            RABITQ_ADMIT_CELL_SHORTLIST_MIN, RabitqAdmitContext, list::WIDTH_LAW_KS,
        },
    },
};

/// Overflow threshold for cell split. Sourced from
/// `vector.cell_split_doc_cap`.
pub(crate) fn cell_split_doc_cap() -> u64 {
    config::global().vector.cell_split_doc_cap
}

/// True when a merged cell superfile should be split into two sub-cells.
pub(crate) fn split_overflow_needed(n_docs: u64) -> bool {
    n_docs > cell_split_doc_cap()
}

/// Ashman-D threshold that triggers a modality-driven split, or `0.0` when the
/// plain [`cell_split_doc_cap`] trigger is in force. Sourced from
/// `vector.cell_split_modality_d`.
pub(crate) fn cell_split_modality_d() -> f64 {
    config::global().vector.cell_split_modality_d
}

/// Target number of *whole* modes grouped per cell for the modality trigger. A
/// cell holding `<= R` modes is healthy and left whole; one holding more splits
/// into `ceil(K/R)` children of ~R whole modes each. Grouping (R>1), not
/// one-mode-per-cell, because 1-mode/cell routes *worst* at nprobe=1 (measured
/// 0.967 vs 0.996 at 4/cell) — tight one-mode centroids mis-route boundary
/// queries. The `<= R` stop bounds the grid: children settle at `<= R` modes and
/// aren't re-split, so the cell count converges rather than running away.
const MODALITY_MODES_PER_CELL: usize = 4;

/// Smallest cell (in live rows) the modality recursion will split — a pure
/// sliver-guard tied to the fine-IVF granularity (a child needs ~≥2 fine
/// clusters at `kmeans_pts_per_centroid`≈64 to be viable), NOT a mode-isolation
/// bar. It must sit *below* the natural mode size at every scale, or the
/// recursion can't reach one-mode leaves: e.g. at a 200k drain (mode ≈195 docs)
/// a floor of 512 stops recursion at ~390-doc 2-mode leaves, which then grow
/// past 512 and re-split every batch → runaway over-fragmentation (200k×5 →
/// 1759 cells / 0.758). At 128 the D-stop (unimodal) governs instead, so the
/// recursion isolates one-mode leaves at any scale. Larger scales are
/// unaffected (their modes ≫ any small floor; the D-stop fires first).
pub(crate) const MODALITY_MIN_CELL_DOCS: u64 = 128;

/// True when a cell is a *candidate* for the modality-driven split — either it
/// overflows the hard cap, or the modality trigger is on and the cell is large
/// enough to test. The actual split decision (Ashman D) is made in
/// [`crate::supertable::writer::split_overflow_cell`], where the rows are
/// resident; this only gates which cells that decision runs on.
pub(crate) fn split_candidate(n_docs: u64) -> bool {
    split_overflow_needed(n_docs)
        || (cell_split_modality_d() > 0.0 && n_docs >= MODALITY_MIN_CELL_DOCS)
}

/// Append-only count bookkeeping for touched cells.
pub(crate) fn apply_cell_count_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
) -> ClusterCentroids {
    let mut updated = base.clone();
    for (&cell, &count) in count_updates {
        if let Some(slot) = updated.counts.get_mut(cell as usize) {
            *slot = count;
        }
    }
    updated
}

/// Apply count updates from maintenance (incoming routing / compaction).
pub(crate) fn apply_cell_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
) -> ClusterCentroids {
    apply_cell_count_updates(base, count_updates)
}

/// Replica candidates considered per row beyond its primary cell — the
/// SPANN-style closure depth. Together with the closure distance ratio this
/// bounds the candidate pool; the configured replica budget
/// (`drain_replica_target_factor`) still decides how many candidates are
/// actually materialized, thinnest margins first.
pub(crate) const REPLICA_CLOSURE_MAX_REPLICAS: usize = 3;

/// A cell qualifies as a replica candidate when the row's distance to it is
/// within this multiple of the row's primary-cell distance. Rows deep inside
/// their cell (small primary distance) get a proportionally tight window and
/// therefore no replicas; genuine boundary rows qualify toward every nearby
/// cell, not only the single second-nearest.
pub(crate) const REPLICA_CLOSURE_DISTANCE_RATIO: f32 = 1.2;

/// K-means sample size for the cell-split planner: `clusters × this`, floored
/// at [`SPLIT_KMEANS_SAMPLE_MIN`]. Centroids train on a strided sample of the
/// cell, then the full cell is assigned under `metric`.
const SPLIT_KMEANS_SAMPLE_PER_CLUSTER: usize = 2048;
/// Lower bound on the split planner's k-means training sample.
const SPLIT_KMEANS_SAMPLE_MIN: usize = 4096;
/// Lloyd iterations for the split planner's k-means — short, since it trains on
/// a sample and the per-child pack re-trains fine centroids downstream.
const SPLIT_KMEANS_ITERS: usize = 10;
/// Fixed XOR mixed into the split cell id to seed the split's k-means, keeping
/// its PRNG stream distinct from other per-cell seeds.
const SPLIT_KMEANS_SEED_XOR: u64 = 0x5157_5f4b_4d45_414e;
/// Route-fidelity target for the self-tuning split: the fraction of rows that
/// must land in their NEAREST sub-centroid (== query routing) for a split to be
/// accepted. Below this, a cell can't be cleanly partitioned at the current `k`
/// (its natural groups are bigger than the per-child capacity, forcing docs off
/// their nearest centroid → nprobe=1 misses), so the planner tries a larger `k`.
const SPLIT_ROUTE_FIDELITY_TARGET: f64 = 0.97;
/// Multiplicative step when the self-tuning split raises `k` to chase route
/// fidelity (more, smaller children → each holds fewer whole groups → less
/// capacity-forced displacement).
const SPLIT_SELF_TUNE_K_STEP: f64 = 1.5;
/// Cap on how far the self-tuning split may raise `k` above the cap-minimum
/// `⌈rows/cap⌉`. Bounds the extra children (and the retry cost) when even a fine
/// split can't reach the fidelity target (near-degenerate / one-giant-blob).
const SPLIT_SELF_TUNE_K_MAX_FACTOR: usize = 4;
/// Primary cell assignment plus the row's replica-candidate cells.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundaryAssignment {
    pub primary: u32,
    /// Up to [`REPLICA_CLOSURE_MAX_REPLICAS`] cells within the closure
    /// distance ratio of the primary, each with the row's margin to the
    /// primary/candidate Voronoi boundary. Smaller margin means closer to
    /// the boundary and therefore a better replication candidate. Fixed-size
    /// (`None`-padded) so the per-row hot assign path stays allocation-free.
    pub replicas: [Option<(u32, f32)>; REPLICA_CLOSURE_MAX_REPLICAS],
}

fn boundary_margin(
    clusters: &ClusterCentroids,
    metric: Metric,
    primary: u32,
    neighbor: u32,
    primary_score: f32,
    neighbor_score: f32,
) -> f32 {
    let gap = (neighbor_score - primary_score).max(0.0);
    let c1 = clusters.centroid(primary as usize);
    let c2 = clusters.centroid(neighbor as usize);
    match metric {
        Metric::L2Sq => {
            let separation = distance(metric, c1, c2).sqrt();
            if separation > 0.0 {
                gap / (2.0 * separation)
            } else {
                f32::INFINITY
            }
        }
        Metric::Cosine | Metric::NegDot => {
            let separation = distance(metric, c1, c2).abs();
            if separation > 0.0 {
                gap / separation
            } else {
                f32::INFINITY
            }
        }
    }
}

/// Assignment shortlist width for `n_cells` grid cells: the shared 1-bit
/// admit fraction of the grid with the shared meaningful-window floor,
/// capped at the grid. Below the floor the window covers every cell and
/// [`boundary_assignment_fp32`] takes its exact-scan arm — small grids
/// (and small-dim tests, where a short sign sketch is noise) keep the
/// exact assignment they always had; the prefilter engages only where it
/// pays (measured shapes: 103 of 512, 205 of 1024).
pub(crate) fn assignment_shortlist_window(n_cells: usize) -> usize {
    let scaled = (n_cells as f64 * RABITQ_ADMIT_CELL_SHORTLIST_FRACTION).ceil() as usize;
    scaled
        .max(RABITQ_ADMIT_CELL_SHORTLIST_MIN)
        .min(n_cells.max(1))
}

/// Drain-side boundary assignment: decode the Sq8+ε row once, then assign
/// through the shared 1-bit shortlist + exact rescore. Same assignment
/// semantics as `nearest-two by score then Voronoi margin`.
pub(crate) fn boundary_assignment_encoded(
    clusters: &ClusterCentroids,
    metric: Metric,
    row: &EncodedCellRow,
    admit_ctx: &RabitqAdmitContext,
    window: usize,
) -> BoundaryAssignment {
    let row_fp = dequantize_row(row, clusters.dim as usize);
    boundary_assignment_fp32(clusters, metric, &row_fp, admit_ctx, window)
}

/// Boundary assignment for an fp32 row (commit buffer path and the drain's
/// decoded rows): 1-bit admit shortlist over the grid (XOR+popcount, the
/// same estimator the query-side prefilter uses), exact fp32 rescore of
/// the shortlisted cells only, then the nearest-two + Voronoi-margin
/// closure on the exact scores. Placement is exact within the window;
/// per-row cost scales with `window` (20% of cells) instead of the grid.
pub(crate) fn boundary_assignment_fp32(
    clusters: &ClusterCentroids,
    metric: Metric,
    row_fp: &[f32],
    admit_ctx: &RabitqAdmitContext,
    window: usize,
) -> BoundaryAssignment {
    let n_cent = clusters.n_cent as usize;
    let top_k = REPLICA_CLOSURE_MAX_REPLICAS + 1;
    let ranked: Vec<(u32, f32)> = if window >= n_cent {
        // Window covers the grid: the exact blocked-SIMD scan is cheaper
        // than encode + estimate + rescore.
        nearest_k_centroids_transposed(
            metric,
            row_fp,
            clusters.transposed(),
            n_cent,
            clusters.dim as usize,
            None,
            top_k,
        )
    } else {
        let admit = admit_ctx.encode(row_fp);
        let mut exact: Vec<(u32, f32)> = clusters
            .admit_shortlist(metric, &admit, window)
            .into_iter()
            .map(|(cell, _)| (cell, clusters.score_one(metric, cell as usize, row_fp)))
            .collect();
        exact.sort_unstable_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        exact.truncate(top_k);
        exact
    };
    boundary_from_ranked(clusters, metric, &ranked)
}

/// Shared closure tail: primary = best-ranked cell; replicas = ranked
/// cells within the closure distance ratio, carrying their margin to the
/// shared Voronoi boundary.
fn boundary_from_ranked(
    clusters: &ClusterCentroids,
    metric: Metric,
    ranked: &[(u32, f32)],
) -> BoundaryAssignment {
    let mut replicas = [None; REPLICA_CLOSURE_MAX_REPLICAS];
    let Some(&(primary, primary_score)) = ranked.first() else {
        return BoundaryAssignment {
            primary: 0,
            replicas,
        };
    };
    // Closure pool: every ranked cell whose distance sits within the ratio
    // window of the primary. The margin (distance to the shared Voronoi
    // boundary) orders candidates globally at the budget cut. Same window
    // definition as the routing cutoff (`relative_score_window`), so
    // replication and probing agree on what "near the boundary" means.
    let closure_threshold =
        relative_score_window(primary_score, REPLICA_CLOSURE_DISTANCE_RATIO - 1.0);
    for (slot, &(cell, score)) in ranked.iter().skip(1).enumerate() {
        if score > closure_threshold {
            break;
        }
        replicas[slot] = Some((
            cell,
            boundary_margin(clusters, metric, primary, cell, primary_score, score),
        ));
    }
    BoundaryAssignment { primary, replicas }
}

/// Dequantize one Sq8+ε residual row to fp32.
fn dequantize_row(row: &EncodedCellRow, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    dequantize_row_into(row, &mut out);
    out
}

/// [`dequantize_row`] into a caller-owned scratch (hot loops reuse one
/// allocation). The scratch length is the row's `dim`.
fn dequantize_row_into(row: &EncodedCellRow, out: &mut [f32]) {
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        out,
    );
}

/// Ashman D of a two-means partition, measured on the 1-D projection onto the
/// inter-centroid axis. A k=2 split of a single coherent mode is not free: along
/// the split axis each half is a half-normal, so a unimodal cell sits at a
/// baseline `D ~= 2.6-3.1` (means +/-0.8 sigma over within-std ~0.6 sigma), NOT
/// near zero. A cell spanning two cleanly separated modes scores far higher —
/// the inter-mode gap dwarfs the within-mode spread, so D runs into the tens or
/// hundreds. The operating threshold must therefore sit *above* the unimodal
/// baseline (~4-5 leaves ample margin); a threshold near 2 would split every
/// cell. The projection is what makes this work in high dimension: spread is
/// measured only along the inter-centroid axis, not diluted by the `dim - 1`
/// directions the split does not separate (the failure mode of a raw
/// variance/inertia ratio). `points` is a flat `m * dim` buffer, `cents` is
/// `2 * dim`; the axis length cancels in D, so the raw projection `p · (c1 - c0)`
/// suffices. `0.0` for a degenerate partition (identical centroids or one empty
/// side); `f32::INFINITY` for zero within-side spread (perfectly separated).
fn ashman_d(points: &[f32], dim: usize, cents: &[f32]) -> f64 {
    let m = points.len() / dim;
    if m < 2 || dim == 0 || cents.len() < 2 * dim {
        return 0.0;
    }
    let mut axis = vec![0f32; dim];
    let mut norm2 = 0f64;
    for j in 0..dim {
        let d = cents[dim + j] - cents[j];
        axis[j] = d;
        norm2 += f64::from(d) * f64::from(d);
    }
    if norm2 <= 1e-12 {
        return 0.0;
    }
    let project =
        |v: &[f32]| -> f64 { (0..dim).map(|j| f64::from(v[j]) * f64::from(axis[j])).sum() };
    // Split the projection at the midpoint between the two centroid feet, and
    // accumulate per-side mean/variance of the projection coordinate.
    let mid = 0.5 * (project(&cents[..dim]) + project(&cents[dim..2 * dim]));
    let mut cnt = [0f64; 2];
    let mut sum = [0f64; 2];
    let mut sumsq = [0f64; 2];
    for i in 0..m {
        let p = project(&points[i * dim..(i + 1) * dim]);
        let s = usize::from(p >= mid);
        cnt[s] += 1.0;
        sum[s] += p;
        sumsq[s] += p * p;
    }
    if cnt[0] < 1.0 || cnt[1] < 1.0 {
        return 0.0;
    }
    let mean = [sum[0] / cnt[0], sum[1] / cnt[1]];
    let var = [
        (sumsq[0] / cnt[0] - mean[0] * mean[0]).max(0.0),
        (sumsq[1] / cnt[1] - mean[1] * mean[1]).max(0.0),
    ];
    let denom = (var[0] + var[1]).sqrt();
    if denom <= 0.0 {
        return f64::INFINITY;
    }
    std::f64::consts::SQRT_2 * (mean[1] - mean[0]).abs() / denom
}

/// Max recursion depth of the in-memory k-finder — bounds k to `2^depth`.
const MODALITY_MAX_DEPTH: usize = 6;

/// Per-branch seed perturbations mixed into the child recursion seeds so the
/// left and right sub-groups draw *decorrelated* strided samples (an unperturbed
/// seed would resample the same strides on both sides).
const MODALITY_RECURSE_SEED_LEFT: u64 = 0x1111;
const MODALITY_RECURSE_SEED_RIGHT: u64 = 0x2222;

/// Decode a cell's encoded rows to one flat `n * dim` fp32 buffer, once, so the
/// in-memory recursion re-clusters on fp32 without re-dequantizing per level.
/// The old cross-pass cascade re-materialized and re-decoded every cell at every
/// level; this decodes each cell exactly once.
fn decode_rows(rows: &[&EncodedCellRow], dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows.len() * dim);
    for &row in rows {
        out.extend_from_slice(&dequantize_row(row, dim));
    }
    out
}

/// Reliable mode count by recursive binary bisection of a cell's rows, entirely
/// in memory. At each node: take a fresh strided sample of *this sub-group's*
/// rows (a fresh sample of the real sub-group, NOT a shrinking sub-slice — that
/// noise is what made earlier in-memory counters over-fragment), two-means +
/// [`ashman_d`]; below `threshold` the node is one mode (leaf), else partition
/// the sub-group by nearest centroid and recurse both sides. Leaf count = k.
/// `idx` indexes rows into `decoded`. This is the same validated per-cut test as
/// the cross-pass binary cascade (→ the reliable k), but without the per-level
/// Blob re-reads.
fn recursive_binary_k(
    decoded: &[f32],
    dim: usize,
    idx: &[usize],
    seed: u64,
    threshold: f64,
    depth: usize,
) -> usize {
    let m = idx.len();
    if (m as u64) < MODALITY_MIN_CELL_DOCS || depth == 0 {
        return 1;
    }
    let sample_n = m.min((2 * SPLIT_KMEANS_SAMPLE_PER_CLUSTER).max(SPLIT_KMEANS_SAMPLE_MIN));
    let mut sample = Vec::with_capacity(sample_n * dim);
    for s in 0..sample_n {
        let i = idx[s * m / sample_n];
        sample.extend_from_slice(&decoded[i * dim..(i + 1) * dim]);
    }
    let cents = kmeans(&sample, dim, 2, SPLIT_KMEANS_ITERS, seed);
    if cents.len() < 2 * dim || ashman_d(&sample, dim, &cents) < threshold {
        return 1;
    }
    let (c0, c1) = (&cents[..dim], &cents[dim..2 * dim]);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &i in idx {
        let v = &decoded[i * dim..(i + 1) * dim];
        if distance(Metric::L2Sq, v, c0) <= distance(Metric::L2Sq, v, c1) {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    if left.is_empty() || right.is_empty() {
        return 1;
    }
    let seed_l = seed ^ MODALITY_RECURSE_SEED_LEFT;
    let seed_r = seed ^ MODALITY_RECURSE_SEED_RIGHT;
    recursive_binary_k(decoded, dim, &left, seed_l, threshold, depth - 1)
        + recursive_binary_k(decoded, dim, &right, seed_r, threshold, depth - 1)
}

/// The split plan for a candidate cell given its resident `rows`:
/// `Some((k, self_tune))` — split into `k` children — or `None` to leave it
/// whole. Over the hard `cell_split_doc_cap` a cell splits into the cap-derived
/// `k` with the executor self-tuning `k` up for route fidelity (the backstop
/// path). Otherwise, when the modality trigger is on (`cell_split_modality_d >
/// 0`), an **in-memory recursive binary** finds the reliable mode count `k` (one
/// materialize, no cross-pass cascade — see [`recursive_binary_k`]); a cell with
/// `<= R` modes is left whole, one with more splits into `g = ceil(K/R)` children
/// with the executor self-tuning `k` up from `g` for route fidelity. The count
/// must come from the recursion on rows — every up-front / summary estimate
/// over-counts on real embeddings (recursive-Ashman-on-centroids 6692 / 0.344 vs
/// validated binary-on-rows 1025 / 0.996). With the trigger off (default) this is
/// the plain over-cap check.
pub(crate) fn cell_split_plan(
    rows: &[&EncodedCellRow],
    dim: usize,
    split_cell: u32,
    modality_d: f64,
) -> Option<(usize, bool)> {
    let n_docs = rows.len() as u64;
    let cap = cell_split_doc_cap().max(1) as usize;
    let k_by_cap = rows.len().div_ceil(cap).max(2);
    let threshold = modality_d;
    // Modality trigger off (default): the caller's over-cap gate is the sole
    // split trigger, so a cell that reaches here is a confirmed split — partition
    // into the cap-derived k (the executor self-tunes k upward for route
    // fidelity). A just-over-cap cell splits 2-way; a bulk overflow into more.
    if threshold <= 0.0 {
        return Some((k_by_cap, true));
    }
    // Modality trigger on: a hard-cap overflow still always splits; below the
    // cap, split only a genuinely multimodal cell (Ashman D below), leaving a
    // unimodal cell whole (`None`).
    if split_overflow_needed(n_docs) {
        return Some((k_by_cap, true));
    }
    if n_docs < MODALITY_MIN_CELL_DOCS {
        return None;
    }
    let seed = (split_cell as u64) ^ SPLIT_KMEANS_SEED_XOR;
    let decoded = decode_rows(rows, dim);
    let idx: Vec<usize> = (0..rows.len()).collect();
    // In-memory recursive binary finds the reliable mode count K (materialize once, no
    // cross-pass cascade). The count must come from the ROWS — every summary estimate
    // over-counts on real embeddings, incl. recursing on the fine centroids
    // (6692 cells / 0.344) vs the validated binary-on-rows (1025 / 0.996), because
    // averaged centroids shed within-mode noise and read as extra modes.
    let k = recursive_binary_k(&decoded, dim, &idx, seed, threshold, MODALITY_MAX_DEPTH);
    // Whole-mode grouping: split only when the cell holds MORE than R whole modes, into
    // ceil(K/R) children of ~R modes each (never one-mode-per-cell). A cell already at
    // `<= R` modes is healthy and left whole — the stop that keeps the grid from running
    // away under streaming. `self_tune = true`: the executor raises k from this start
    // toward route fidelity, so a grouped child whose centroid mis-routes its modes gets
    // sub-split until assign == route.
    let r = MODALITY_MODES_PER_CELL;
    if k <= r {
        return None;
    }
    let g = k.div_ceil(r).max(2);
    Some((g, true))
}

/// One capacitated split attempt at a fixed `k`: greedy-k-means++ (random at
/// `k = 2`) centroids on a strided sample, then a capacity-bounded
/// nearest-centroid assignment (`cap_target` rows per child, spilling the
/// overflow to the next-nearest). Returns the `k * dim` centroids, the per-row
/// child assignment, and the **route fidelity** — the fraction of rows placed in
/// their NEAREST child, which predicts nprobe=1 recall (a doc in its nearest
/// cell is found at nprobe=1; a spilled one is not). The self-tuning
/// [`plan_sq8_split_kway`] calls this across a ladder of `k`.
fn capacitated_split_at_k(
    rows: &[&EncodedCellRow],
    split_cell: u32,
    dim: usize,
    metric: Metric,
    k: usize,
    cap_target: usize,
) -> (Vec<f32>, Vec<u32>, f64) {
    let n = rows.len();
    let mut assign = vec![0u32; n];
    let sample_n = n.min((k * SPLIT_KMEANS_SAMPLE_PER_CLUSTER).max(SPLIT_KMEANS_SAMPLE_MIN));
    let mut sample = Vec::with_capacity(sample_n * dim);
    for s in 0..sample_n {
        let idx = s * n / sample_n;
        sample.extend_from_slice(&dequantize_row(rows[idx], dim));
    }
    let seed = (split_cell as u64) ^ SPLIT_KMEANS_SEED_XOR;
    let cents = if k > 2 {
        kmeans_pp(&sample, dim, k, SPLIT_KMEANS_ITERS, seed)
    } else {
        kmeans(&sample, dim, k, SPLIT_KMEANS_ITERS, seed)
    };
    if cents.len() < k * dim {
        return (cents, assign, 0.0);
    }
    // Per-row distances to every centroid; track the uncapped nearest (for route
    // fidelity) and the nearest-vs-next margin (fill order — strongest preference
    // first, so a full child bumps the rows that least mind their next-nearest).
    let mut row_dists = vec![0f32; n * k];
    let mut nearest = vec![0u32; n];
    let mut order: Vec<(usize, f32)> = Vec::with_capacity(n);
    for (i, row) in rows.iter().copied().enumerate() {
        let rv = dequantize_row(row, dim);
        let base = i * k;
        let (mut best, mut second, mut best_c) = (f32::INFINITY, f32::INFINITY, 0usize);
        for c in 0..k {
            let d = distance(metric, &rv, &cents[c * dim..(c + 1) * dim]);
            row_dists[base + c] = d;
            if d < best {
                second = best;
                best = d;
                best_c = c;
            } else if d < second {
                second = d;
            }
        }
        nearest[i] = best_c as u32;
        order.push((i, second - best));
    }
    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let mut counts = vec![0usize; k];
    for (i, _) in order {
        let base = i * k;
        let mut best_c = usize::MAX;
        let mut best_d = f32::INFINITY;
        for c in 0..k {
            if counts[c] < cap_target && row_dists[base + c] < best_d {
                best_d = row_dists[base + c];
                best_c = c;
            }
        }
        if best_c == usize::MAX {
            best_c = (0..k)
                .min_by(|&a, &b| {
                    row_dists[base + a]
                        .partial_cmp(&row_dists[base + b])
                        .unwrap_or(Ordering::Equal)
                })
                .unwrap_or(0);
        }
        assign[i] = best_c as u32;
        counts[best_c] += 1;
    }
    let faithful = (0..n).filter(|&i| assign[i] == nearest[i]).count();
    (cents, assign, faithful as f64 / n as f64)
}

/// Split `split_cell` into sub-cells via capacitated k-means with self-tuned k.
/// Returns `k' * dim` centroid components and a `0..k'` per-row assignment
/// aligned to `rows`. Starts at the passed cap-minimum `k = ⌈rows/cap⌉` and
/// raises k (more, smaller children) until the capacitated assignment reaches
/// the route-fidelity target — so a cell whose natural groups exceed one child's
/// capacity is cut into enough children that each holds ~whole groups
/// (`assign == route`, nprobe=1 recall) instead of scattering the overflow. The
/// child count `k'` may exceed `k`; the caller derives it from the centroid
/// length. Deterministic single path — capacitated always populates ≥2 children
/// for `rows ≥ 2`, so there is no fallback.
pub(crate) fn plan_sq8_split_kway(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
    k: usize,
    self_tune: bool,
) -> (Vec<f32>, Vec<u32>) {
    let dim = clusters.dim as usize;
    let k = k.max(2).min(rows.len().max(2));
    if rows.len() < 2 {
        // Caller guards on MIN_ROWS_TO_SPLIT_CELL; stay defensive against a
        // degenerate one-row input.
        let c = manifest_centroid_components_from_row(rows[0], dim);
        let mut cents = Vec::with_capacity(k * dim);
        for _ in 0..k {
            cents.extend_from_slice(&c);
        }
        return (cents, vec![0u32; rows.len()]);
    }

    // Self-tuning k. `cap_target` is the cap-minimum child size (≈ the doc cap);
    // start at the passed `k = ⌈rows/cap⌉` and, while route fidelity is below
    // target, raise k (more, smaller children — each holds fewer whole groups, so
    // the capacitated assignment spills fewer rows off their nearest centroid).
    // Keep the highest-fidelity attempt. Larger k yields children well under
    // `cap_target` that fit without bumping, so `assign == route` and nprobe=1
    // recall is preserved even when a coarse cell packs many natural groups.
    let n = rows.len();
    let cap_target = n.div_ceil(k).max(1);
    // `self_tune = false` pins the split to exactly `k` (the caller already
    // knows the right child count — e.g. the recursive mode count); `true`
    // raises `k` toward the route-fidelity target for the cap-derived backstop.
    let k_max = if self_tune {
        k.saturating_mul(SPLIT_SELF_TUNE_K_MAX_FACTOR).min(n).max(k)
    } else {
        k
    };
    let mut best: Option<(f64, Vec<f32>, Vec<u32>)> = None;
    let mut k_try = k;
    loop {
        let (cents, cand, rf) =
            capacitated_split_at_k(rows, split_cell, dim, metric, k_try, cap_target);
        if best.as_ref().is_none_or(|b| rf > b.0) {
            best = Some((rf, cents, cand));
        }
        if rf >= SPLIT_ROUTE_FIDELITY_TARGET || k_try >= k_max {
            break;
        }
        k_try = ((k_try as f64 * SPLIT_SELF_TUNE_K_STEP).ceil() as usize)
            .max(k_try + 1)
            .min(k_max);
    }
    let (rf, cents, cand) = best.expect("self-tune loop sets best on the first iteration");
    if tracing::enabled!(tracing::Level::DEBUG) {
        // Per-split trace: the cap-derived starting k, the k the self-tune
        // settled on, the achieved route fidelity, and the child-size spread
        // (min/max) — the levers that explain a split's nprobe=1 recall.
        let k_final = (cents.len() / dim).max(1);
        let mut sizes = vec![0usize; k_final];
        for &c in &cand {
            sizes[c as usize] += 1;
        }
        tracing::debug!(
            cell = split_cell,
            rows = n,
            cap_target,
            k_start = k,
            k_final,
            route_fidelity = rf,
            child_min = sizes.iter().copied().min().unwrap_or(0),
            child_max = sizes.iter().copied().max().unwrap_or(0),
            "cell split planned"
        );
    }
    (cents, cand)
}

// ---------- Drain-time probe-width calibration ----------

/// Rows reservoir-sampled as stand-in queries for probe-width calibration.
/// Corpus rows are the right calibration distribution — on Cohere-1M/768d,
/// stored-row queries and the dataset's held-out test queries measured the
/// same top-k cell spread.
const WIDTH_LAW_QUERY_SAMPLE: usize = 256;
/// Mean top-k coverage a probe width must reach to be recorded in the law —
/// the engine's recall@k acceptance bar.
const WIDTH_LAW_TARGET_COVERAGE: f64 = 0.99;
/// Fixed seed for the calibration reservoir, so a re-drained identical
/// corpus stamps an identical law.
const WIDTH_LAW_SAMPLE_SEED: u64 = 0x51ED_CA1B;
/// Rows decoded per chunk while scoring a spilled cell.
const WIDTH_LAW_SCORE_CHUNK: usize = 1024;

/// Frozen query sample: dequantized fp32 vectors + their stable ids
/// (self-hit exclusion while scoring).
struct WidthLawQueries {
    queries: Vec<f32>,
    ids: Vec<i128>,
}

/// Drain-time probe-width calibration: measures, on this table's own data,
/// how many grid cells (in routing order) cover the exact top-k, and stamps
/// the result into the manifest's [`CellRoutingParams::width_for_k`] law.
///
/// How far the true top-k spreads over cells is a property of the corpus —
/// synthetic clustered data concentrates (1 cell at k = 10), real text
/// embeddings spray (Cohere-1M/768d measured ~30 of 256 cells at k = 100,
/// identical under a converged reference clustering, so it is the data and
/// not grid quality) — which is why the default probe width must be
/// measured per table rather than hardcoded.
///
/// Lifecycle inside one clean (non-resumed) drain:
///   1. [`Self::offer`] on every spilled row — [`Reservoir`]-samples the
///      stand-in queries (sequential; the spill loop holds `&mut`).
///   2. [`Self::freeze`] once all batches have spilled.
///   3. [`Self::score_cell`] per cell during the pack fan-out — re-reads
///      the cell's spill (the pack pass reads it anyway), scores every row
///      with the shared [`distance`] kernel (rows unit-normalized for
///      cosine, matching the rerank kernels' norm division), and merges
///      that cell's per-query top-k under one lock per cell.
///   4. [`Self::finish`] before the drain's manifest stamp — ranks cells
///      with the same [`ClusterCentroids::rank_cells`] routing order
///      queries use and extracts the width law.
///
/// Resumed drains skip calibration entirely (checkpointed batches never
/// re-stream), keeping whatever law the manifest already carries.
pub(crate) struct WidthLawCalibration {
    dim: usize,
    metric: Metric,
    reservoir: Reservoir,
    /// Stable id of each reservoir slot, kept in lockstep through
    /// [`Reservoir::update_traced`].
    slot_ids: Vec<i128>,
    dequant_scratch: Vec<f32>,
    frozen: Option<WidthLawQueries>,
    /// Per-query `(score, cell, stable id)` candidates, truncated to the
    /// largest law `k` as cells merge in. The stable id lets [`Self::finish`]
    /// deduplicate boundary replicas (drain replica factor > 1.0) to their
    /// best-scored copy, so one neighbor can never occupy several top-k
    /// slots and narrow the law. Lock poisoning is recovered, not
    /// propagated: each merge is an atomic append+truncate, so a panicked
    /// pack worker leaves the held data usable.
    tops: Mutex<Vec<Vec<(f32, u32, i128)>>>,
}

impl WidthLawCalibration {
    pub(crate) fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            reservoir: Reservoir::new(WIDTH_LAW_QUERY_SAMPLE, dim, WIDTH_LAW_SAMPLE_SEED),
            slot_ids: Vec::with_capacity(WIDTH_LAW_QUERY_SAMPLE),
            dequant_scratch: vec![0f32; dim],
            frozen: None,
            tops: Mutex::new(Vec::new()),
        }
    }

    /// Offer one spilled row as a calibration-query candidate.
    pub(crate) fn offer(&mut self, row: &MaterializedIvfRow) {
        debug_assert!(self.frozen.is_none(), "offer after freeze");
        dequantize_row_into(&row.encoded, &mut self.dequant_scratch);
        if let Some(slot) = self.reservoir.update_traced(&self.dequant_scratch) {
            if slot == self.slot_ids.len() {
                self.slot_ids.push(row.stable_id);
            } else {
                self.slot_ids[slot] = row.stable_id;
            }
        }
    }

    /// Freeze the sampled queries. Called once, after the last batch
    /// spilled and before cell packing scores.
    pub(crate) fn freeze(&mut self) {
        let queries = self.reservoir.sample().to_vec();
        let ids = self.slot_ids.clone();
        *self.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![Vec::new(); ids.len()];
        self.frozen = Some(WidthLawQueries { queries, ids });
    }

    /// Score every row of one spilled cell against the frozen queries and
    /// merge into the per-query candidate lists. Safe to call from the
    /// pack fan-out workers; the merge takes one lock per cell.
    pub(crate) fn score_cell(&self, cell: u32, spill: &SpilledCellRows) -> Result<(), BuildError> {
        let Some(frozen) = self.frozen.as_ref() else {
            return Err(BuildError::Store(
                "width-law score_cell before freeze".into(),
            ));
        };
        let n_queries = frozen.ids.len();
        if n_queries == 0 {
            return Ok(());
        }
        let k_max = *WIDTH_LAW_KS.last().expect("law has k points");
        let mut partial: Vec<Vec<(f32, u32, i128)>> = vec![Vec::new(); n_queries];
        let mut reader = spill.reader()?;
        let mut remaining = spill.n_rows();
        let mut scratch = vec![0f32; self.dim];
        while remaining > 0 {
            let chunk = reader.next_chunk(WIDTH_LAW_SCORE_CHUNK.min(remaining))?;
            remaining -= chunk.len();
            for row in &chunk {
                dequantize_row_into(&row.encoded, &mut scratch);
                if self.metric == Metric::Cosine {
                    // The rerank kernels divide by the stored row norm;
                    // unit-normalizing the row lets the shared [`distance`]
                    // kernel (which assumes unit inputs for cosine) score
                    // with the same ranking. Query scaling is per-query
                    // monotone and cannot reorder its candidates.
                    normalize(&mut scratch);
                }
                for (qi, q) in frozen.queries.chunks_exact(self.dim).enumerate() {
                    // Self-hit: a sampled query trivially covers itself.
                    if row.stable_id == frozen.ids[qi] {
                        continue;
                    }
                    partial[qi].push((distance(self.metric, q, &scratch), cell, row.stable_id));
                }
            }
            // Bound the per-cell partials the same way the merge does.
            for cand in &mut partial {
                truncate_ascending(cand, k_max);
            }
        }
        let mut tops = self.tops.lock().unwrap_or_else(PoisonError::into_inner);
        for (qi, cand) in partial.into_iter().enumerate() {
            merge_candidates(&mut tops[qi], cand, k_max);
        }
        Ok(())
    }

    /// Extract the width law: cells (in the grid's routing order) needed
    /// for mean [`WIDTH_LAW_TARGET_COVERAGE`] coverage of the exact top-k
    /// at each [`WIDTH_LAW_KS`] point. Each point is measured over the
    /// queries whose candidate count reaches its `k` — one boundary-starved
    /// query excludes itself, not the whole sample. Points NO query can
    /// support stay `0` (uncalibrated). Measured points are floored to be
    /// monotone in `k`. `None` when nothing was sampled.
    pub(crate) fn finish(self, grid: &ClusterCentroids) -> Option<[u32; WIDTH_LAW_KS.len()]> {
        let frozen = self.frozen?;
        let n_queries = frozen.ids.len();
        if n_queries == 0 || grid.n_cent == 0 {
            return None;
        }
        let n_cells = grid.n_cent as usize;
        let tops = self
            .tops
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner);
        // Superseded or empty cells inflate ranks slightly (routing skips
        // them, this count does not) — an over-probe, never an under-probe;
        // fresh drains, the normal calibration moment, have neither.
        let mut law = [0u32; WIDTH_LAW_KS.len()];
        let mut coverage_sums: Vec<Vec<f64>> = vec![vec![0f64; n_cells]; WIDTH_LAW_KS.len()];
        let mut support = [0usize; WIDTH_LAW_KS.len()];
        let mut rank_of_cell = vec![0u32; n_cells];
        for (qi, cand) in tops.iter().enumerate() {
            let q = &frozen.queries[qi * self.dim..(qi + 1) * self.dim];
            let ranked = grid.rank_cells(self.metric, q);
            for (rank, (cell, _)) in ranked.iter().enumerate() {
                if let Some(slot) = rank_of_cell.get_mut(*cell as usize) {
                    *slot = rank as u32;
                }
            }
            // [`merge_candidates`] keeps one entry per stable id, so the
            // accumulator only needs ranking by score for the prefix walk.
            let mut sorted = cand.clone();
            sorted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            for (ki, &k) in WIDTH_LAW_KS.iter().enumerate() {
                if sorted.len() < k {
                    // Below this point's k: the query measures the smaller
                    // points and sits out this one.
                    continue;
                }
                support[ki] += 1;
                // Per-rank counts of this query's top-k, then a prefix walk
                // accumulates the mean coverage curve.
                let mut per_rank = vec![0u32; n_cells];
                for (_, cell, _) in &sorted[..k] {
                    // Candidate cell ids come from the same drain that built
                    // `grid` and split parents keep their slot when
                    // superseded, so an out-of-range id is unreachable today
                    // — but calibration is best-effort diagnostics: an
                    // inconsistent input abandons the law (unstamped ⇒
                    // default routing) rather than panicking the drain.
                    let &rank = rank_of_cell.get(*cell as usize)?;
                    per_rank[rank as usize] += 1;
                }
                let mut covered = 0u32;
                for (rank, count) in per_rank.iter().enumerate() {
                    covered += count;
                    coverage_sums[ki][rank] += f64::from(covered) / k as f64;
                }
            }
        }
        for (ki, sums) in coverage_sums.iter().enumerate() {
            if support[ki] == 0 {
                continue;
            }
            let target = WIDTH_LAW_TARGET_COVERAGE * support[ki] as f64;
            if let Some(rank) = sums.iter().position(|&s| s >= target) {
                law[ki] = (rank + 1) as u32;
            }
        }
        // Coverage need only grows with k, so each measured point is floored
        // by the measured points below it — sampling noise near the target
        // must never let a larger k probe FEWER cells than a smaller one (a
        // recall inversion at query time). Unmeasured points (0) stay 0; the
        // interpolator skips them.
        let mut floor = 0u32;
        for w in law.iter_mut().filter(|w| **w > 0) {
            *w = (*w).max(floor);
            floor = *w;
        }
        (law.iter().any(|&w| w > 0)).then_some(law)
    }
}

/// Merge one cell's candidates into a query's accumulator: collapse to the
/// best-scored copy per stable id FIRST (boundary replicas of one row are
/// one neighbor), then keep the ascending-best `cap`. The dedup must
/// precede the truncate — replicated copies of near rows filling raw slots
/// would evict distinct farther neighbors and stamp a narrower law than a
/// real query experiences.
fn merge_candidates(acc: &mut Vec<(f32, u32, i128)>, mut cand: Vec<(f32, u32, i128)>, cap: usize) {
    acc.append(&mut cand);
    acc.sort_unstable_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.total_cmp(&b.0)));
    acc.dedup_by_key(|c| c.2);
    truncate_ascending(acc, cap);
}

/// Keep the ascending-best `cap` candidates in place.
fn truncate_ascending(cand: &mut Vec<(f32, u32, i128)>, cap: usize) {
    if cand.len() > cap {
        cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        cand.truncate(cap);
    }
}

/// Two-centroid (`k = 2`) test-only wrapper over [`plan_sq8_split_kway`],
/// returning the two-centroid / `u8`-assignment shape the split unit tests
/// were written against. The production split path calls
/// [`plan_sq8_split_kway`] directly.
#[cfg(test)]
pub(crate) fn plan_sq8_split(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let dim = clusters.dim as usize;
    let (cents, assign) = plan_sq8_split_kway(rows, clusters, split_cell, metric, 2, true);
    let c0 = cents[..dim].to_vec();
    let c1 = cents[dim..2 * dim].to_vec();
    (c0, c1, assign.iter().map(|&a| a as u8).collect())
}

/// Replace cell `cell_id`'s centroid with sub-cell 0 and append sub-cells
/// `1..k` as fresh cells at the end of the grid. Returns the grown grid and the
/// `k` sub-cell ids (index 0 == the reused `cell_id`; the rest are the new
/// ids), aligned to `sub_centroids` (`k * dim` fp32).
pub(crate) fn insert_split_centroids(
    base: &ClusterCentroids,
    cell_id: u32,
    sub_centroids: &[f32],
    k: usize,
) -> (ClusterCentroids, Vec<u32>) {
    let dim = base.dim as usize;
    let p = cell_id as usize;
    let old_n = base.n_cent as usize;
    let new_n = old_n + (k - 1);

    let mut fp32 = vec![0f32; new_n * dim];
    for c in 0..old_n {
        fp32[c * dim..(c + 1) * dim].copy_from_slice(base.centroid(c));
    }
    // Sub-cell 0 reuses the split cell's slot; 1..k append.
    fp32[p * dim..(p + 1) * dim].copy_from_slice(&sub_centroids[..dim]);
    let mut ids = vec![cell_id];
    for j in 1..k {
        let new_id = old_n + (j - 1);
        fp32[new_id * dim..(new_id + 1) * dim]
            .copy_from_slice(&sub_centroids[j * dim..(j + 1) * dim]);
        ids.push(new_id as u32);
    }

    // Counts must have one entry per cell: grow to `new_n` so every sub-cell has
    // a slot. Cloning `base.counts` alone leaves it at `old_n`, which silently
    // passes in-memory but truncates the wire encoding (counts and centroids are
    // adjacent) → the grid fails to reopen from storage.
    let mut counts = base.counts.clone();
    counts.resize(new_n, 0);
    let updated = ClusterCentroids::from_fp32(new_n as u32, base.dim, &fp32, counts);
    (updated, ids)
}

/// Binary variant (`k = 2`): replace `cell_id`'s centroid and append one new
/// sub-cell. Test-only wrapper preserving the single-new-id shape the unit
/// tests use; the production split path calls [`insert_split_centroids`].
#[cfg(test)]
pub(crate) fn insert_split_centroid(
    base: &ClusterCentroids,
    cell_id: u32,
    sub_centroids: &[f32],
) -> (ClusterCentroids, u32) {
    let (updated, ids) = insert_split_centroids(base, cell_id, sub_centroids, 2);
    (updated, ids[1])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    /// Boundary replicas must count as ONE neighbor in the width law:
    /// [`WidthLawCalibration::finish`] dedups candidates to the
    /// best-scored copy per stable id before measuring coverage. The
    /// fixture plants one id three times in the two query-nearest cells;
    /// without the dedup those copies pad the top-10 and the law would
    /// stop at width 2, hiding the two real neighbors in the 4th-ranked
    /// cell.
    #[test]
    fn width_law_finish_dedups_replicated_rows() {
        const DIM: usize = 4;
        // Grid ranked against the e0 query: cells 0, 1, 2, 3 in order.
        let grid = ClusterCentroids::from_fp32(
            4,
            DIM as u32,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.9, 0.1, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            vec![1; 4],
        );
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine);
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        // (score, cell, stable id): id 1 replicated across cells 0 and 1;
        // ids 2..=8 fill the near cells; ids 9 and 10 sit in cell 3 and
        // only enter the top-10 once the replicas collapse to one slot.
        let mut cands = vec![(0.01, 0, 1), (0.02, 1, 1), (0.03, 1, 1)];
        cands.extend((2..=8).map(|id| (0.03 + id as f32 * 0.01, (id % 2) as u32, id as i128)));
        cands.push((0.5, 3, 9));
        cands.push((0.6, 3, 10));
        // Plant through the same merge the scorer uses — dedup happens
        // there, BEFORE the truncate, so replicas can never occupy slots.
        let mut acc = Vec::new();
        merge_candidates(&mut acc, cands, *WIDTH_LAW_KS.last().expect("knots"));
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];

        let law = cal.finish(&grid).expect("law from planted candidates");
        // k=1: the best copy of id 1 sits in the top-ranked cell.
        assert_eq!(law[0], 1, "top-1 coverage is the nearest cell");
        // k=10: deduped top-10 = ids 1..=10, whose coverage needs the
        // 4th-ranked cell. Replica padding would have stopped at 2.
        assert_eq!(
            law[1], 4,
            "replicated copies must not pad top-k coverage (got width {})",
            law[1]
        );
        // 10 deduped candidates cannot support the k=100/1000 points.
        assert_eq!(&law[2..], &[0, 0], "unsupported points stay uncalibrated");
    }

    /// Per-query point support and the monotone floor. Query A (10
    /// candidates spread over four cells) cannot support k=100 — it must
    /// sit that point out rather than zero it for the whole sample. Query B
    /// (100 candidates in one cell) then measures k=100 alone at width 1,
    /// NARROWER than the two-query k=10 point (width 4) — the monotone
    /// floor lifts it, because a larger k probing fewer cells would invert
    /// recall at query time.
    #[test]
    fn width_law_supports_points_per_query_and_stays_monotone() {
        const DIM: usize = 4;
        let grid = ClusterCentroids::from_fp32(
            4,
            DIM as u32,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.9, 0.1, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0,
            ],
            vec![1; 4],
        );
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine);
        let mut queries = vec![0.0f32; 2 * DIM];
        queries[0] = 1.0;
        queries[DIM] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries,
            ids: vec![998, 999],
        });
        let k_max = *WIDTH_LAW_KS.last().expect("knots");
        // A: distinct ids 1..=10 spread 2/2/3/3 over cells 0..=3, so its
        // 0.99 top-10 coverage needs the 4th-ranked cell.
        let a: Vec<(f32, u32, i128)> = (1..=10)
            .map(|id| {
                let cell = match id {
                    1 | 2 => 0u32,
                    3 | 4 => 1,
                    5..=7 => 2,
                    _ => 3,
                };
                (id as f32 * 0.01, cell, id as i128)
            })
            .collect();
        // B: distinct ids 100..=199, every one in the top-ranked cell.
        let b: Vec<(f32, u32, i128)> = (100..200)
            .map(|id| (id as f32 * 0.001, 0u32, id as i128))
            .collect();
        let (mut acc_a, mut acc_b) = (Vec::new(), Vec::new());
        merge_candidates(&mut acc_a, a, k_max);
        merge_candidates(&mut acc_b, b, k_max);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc_a, acc_b];

        let law = cal.finish(&grid).expect("law from planted candidates");
        assert_eq!(law[0], 1, "top-1: both queries covered by the nearest cell");
        assert_eq!(
            law[1], 4,
            "k=10 measured over both queries needs A's spread"
        );
        assert_eq!(
            law[2], 4,
            "k=100: B alone measures width 1; the monotone floor lifts it \
             to the k=10 width instead of stamping a recall inversion"
        );
        assert_eq!(law[3], 0, "k=1000 has no supporting query and stays 0");
    }

    /// An out-of-range cell id in the candidates (impossible in the current
    /// drain flow, by construction) abandons the law instead of panicking —
    /// calibration is diagnostics riding a drain and must never abort one.
    #[test]
    fn width_law_bails_on_out_of_range_cell_id() {
        const DIM: usize = 4;
        let grid = ClusterCentroids::from_fp32(
            2,
            DIM as u32,
            &[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ],
            vec![1; 2],
        );
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine);
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        // Cell 7 does not exist in the two-cell grid.
        let mut acc = Vec::new();
        merge_candidates(
            &mut acc,
            vec![(0.01, 7, 1)],
            *WIDTH_LAW_KS.last().expect("knots"),
        );
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];
        assert!(
            cal.finish(&grid).is_none(),
            "inconsistent calibration input must abandon the law, not panic"
        );
    }
    use crate::superfile::vector::{
        cell_posting::{encode_blob, load_encoded_rows_from_blob},
        rerank_codec::{RerankCodec, SQ8_FIXED_OFFSET, SQ8_FIXED_SCALE},
    };

    fn synth_centroids(n_cent: u32, dim: u32) -> ClusterCentroids {
        let nc = n_cent as usize;
        let d = dim as usize;
        let mut fp32 = vec![0f32; nc * d];
        for c in 0..nc {
            for j in 0..d {
                fp32[c * d + j] = c as f32 * 0.5 + j as f32 * 0.01;
            }
        }
        let counts = vec![100; nc];
        ClusterCentroids::from_fp32(n_cent, dim, &fp32, counts)
    }

    fn synth_rows(dim: usize, n: usize, offset: f32) -> Vec<EncodedCellRow> {
        let mut ids = Vec::new();
        let mut vecs = Vec::new();
        for i in 0..n as u32 {
            ids.push(i);
            for d in 0..dim {
                vecs.push(offset + i as f32 * 0.01 + d as f32 * 0.001);
            }
        }
        let blob =
            encode_blob(Metric::L2Sq, dim, &ids, &vecs, RerankCodec::Sq8Residual).expect("encode");
        let stable_ids: Vec<i128> = (0..n).map(|i| i as i128).collect();
        load_encoded_rows_from_blob(&blob, &stable_ids, None).expect("load")
    }

    /// Prod-faithful cell: `n_blobs` gaussian centers in general position
    /// (each `N(0, 1)`), each with `per_blob` normalized points `center + N(0,
    /// sigma)`. Mirrors a 100M coarse cell — ~16 data centers, tight balls,
    /// unit-normalized, full embedding dim — which is where high-dim k-means
    /// fragility (distance concentration + collided seeds) actually shows up,
    /// unlike the tight colinear `synth_rows` blobs.
    fn synth_gaussian_cell(
        dim: usize,
        n_blobs: usize,
        per_blob: usize,
        sigma: f32,
        seed: u64,
    ) -> Vec<EncodedCellRow> {
        use rand::{SeedableRng, rngs::StdRng};
        use rand_distr::{Distribution, Normal};
        let mut rng = StdRng::seed_from_u64(seed);
        let unit = Normal::new(0.0f32, 1.0).expect("unit normal");
        let noise = Normal::new(0.0f32, sigma).expect("noise normal");
        let n = n_blobs * per_blob;
        let mut ids = Vec::with_capacity(n);
        let mut vecs = Vec::with_capacity(n * dim);
        for _ in 0..n_blobs {
            let center: Vec<f32> = (0..dim).map(|_| unit.sample(&mut rng)).collect();
            for _ in 0..per_blob {
                let mut v: Vec<f32> = (0..dim)
                    .map(|d| center[d] + noise.sample(&mut rng))
                    .collect();
                crate::superfile::vector::distance::normalize(&mut v);
                ids.push(ids.len() as u32);
                vecs.extend_from_slice(&v);
            }
        }
        let blob =
            encode_blob(Metric::L2Sq, dim, &ids, &vecs, RerankCodec::Sq8Residual).expect("encode");
        let stable_ids: Vec<i128> = (0..n).map(|i| i as i128).collect();
        load_encoded_rows_from_blob(&blob, &stable_ids, None).expect("load")
    }

    fn synth_fixed_rows(dim: usize, n: usize, code: u8) -> Vec<EncodedCellRow> {
        let scale: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_SCALE; dim]);
        let offset: Arc<[f32]> = Arc::from(vec![SQ8_FIXED_OFFSET; dim]);
        (0..n)
            .map(|id| EncodedCellRow {
                stable_id: id as i128,
                rerank_codec: RerankCodec::Sq8FixedResidual,
                scale: Arc::clone(&scale),
                offset: Arc::clone(&offset),
                codes: vec![code; dim],
                residuals: vec![0; dim],
                norm_sq: None,
            })
            .collect()
    }

    /// Rotation seed for the assignment-test admit contexts.
    const TEST_ROT_SEED: u64 = 7;

    /// Closure replication: a row equidistant-ish to several cells collects a
    /// replica candidate for every cell inside the distance-ratio window
    /// (ordered nearest-first), and a row deep inside its cell collects none.
    /// (4 cells ⇒ the shortlist window covers the grid, so this exercises the
    /// exact-scan arm.)
    #[test]
    fn boundary_assignment_closure_matches_distance_ratio() {
        let dim = 4usize;
        // Four centroids at 0, 1, 2, 30 on every axis.
        let mut fp32 = Vec::new();
        for base in [0.0f32, 1.0, 2.0, 30.0] {
            fp32.extend(std::iter::repeat_n(base, dim));
        }
        let clusters = ClusterCentroids::from_fp32(4, dim as u32, &fp32, vec![1; 4]);
        let ctx = RabitqAdmitContext::new(dim, TEST_ROT_SEED);
        let window = assignment_shortlist_window(4);

        // Row at 0.9: distances (L2Sq per dim) to cells 0/1/2 are 0.81, 0.01,
        // 1.21 (per-dim) — cell 1 is primary; cell 0 and 2 are far outside a
        // 1.2 ratio window of 0.01. No replicas.
        let deep = vec![0.9f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, Metric::L2Sq, &deep, &ctx, window);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas, [None; REPLICA_CLOSURE_MAX_REPLICAS]);

        // Row at 1.01 — just past the exact midpoint region between cells 0.98
        // and 1.02... use 1.5: exactly between cells 1 and 2 (distances equal),
        // both inside each other's ratio window; cell 0 at 1.5 distance 2.25
        // per dim is outside 1.2 × 0.25. Expect primary = 1 (tie broken by
        // lower id) and exactly one replica: cell 2.
        let boundary = vec![1.5f32; dim];
        let assignment = boundary_assignment_fp32(&clusters, Metric::L2Sq, &boundary, &ctx, window);
        assert_eq!(assignment.primary, 1);
        assert_eq!(assignment.replicas[0].map(|(cell, _)| cell), Some(2));
        assert_eq!(assignment.replicas[1], None);
        let margin = assignment.replicas[0].expect("replica").1;
        assert!(
            margin.is_finite() && margin >= 0.0,
            "boundary margin must be a finite non-negative distance, got {margin}"
        );
    }

    /// The shortlist window is the shared 20% fraction with the shared 48
    /// floor, capped at the grid: at or under the floor the window covers
    /// every cell (exact assignment), past it the 20% slice scales.
    #[test]
    fn assignment_shortlist_window_scales_with_grid() {
        // At or under the floor: the whole grid (exact-scan arm).
        assert_eq!(assignment_shortlist_window(1), 1);
        assert_eq!(assignment_shortlist_window(16), 16);
        assert_eq!(assignment_shortlist_window(48), 48);
        // Floor binds until 20% overtakes it at 240 cells.
        assert_eq!(
            assignment_shortlist_window(64),
            RABITQ_ADMIT_CELL_SHORTLIST_MIN
        );
        assert_eq!(
            assignment_shortlist_window(240),
            RABITQ_ADMIT_CELL_SHORTLIST_MIN
        );
        // Plain 20% past the floor.
        assert_eq!(assignment_shortlist_window(256), 52);
        assert_eq!(assignment_shortlist_window(512), 103);
        assert_eq!(assignment_shortlist_window(1024), 205);
    }

    /// The 1-bit shortlisted assignment must agree with the exact scan on
    /// rows that clearly belong to a cell — the regime every committed row
    /// is in. Planted well-separated centroids, rows jittered around them;
    /// primaries must match the exact path cell-for-cell. The grid sits
    /// past the shared floor so the shortlist arm actually engages.
    #[test]
    fn shortlisted_assignment_matches_exact_on_planted_cells() {
        let dim = 64usize;
        let n_cells = 300usize;
        let mut fp32 = vec![0.0f32; n_cells * dim];
        for (c, chunk) in fp32.chunks_mut(dim).enumerate() {
            // Distinct direction per cell: two active axes with distinct
            // magnitudes keep centroids well separated.
            chunk[c % dim] = 4.0 + (c / dim) as f32;
            chunk[(c * 7 + 3) % dim] = 2.0;
        }
        let clusters =
            ClusterCentroids::from_fp32(n_cells as u32, dim as u32, &fp32, vec![1; n_cells]);
        let ctx = RabitqAdmitContext::new(dim, TEST_ROT_SEED);
        let window = assignment_shortlist_window(n_cells);
        assert!(window < n_cells, "test must exercise the shortlist arm");

        let mut state = 0x9e37_79b9_97f4_a7c5u64;
        let mut jitter = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            ((state >> 33) % 1000) as f32 / 1000.0 * 0.2 - 0.1
        };
        for c in 0..n_cells {
            let mut row = fp32[c * dim..(c + 1) * dim].to_vec();
            for v in row.iter_mut() {
                *v += jitter();
            }
            let shortlisted = boundary_assignment_fp32(&clusters, Metric::L2Sq, &row, &ctx, window);
            let exact = boundary_assignment_fp32(&clusters, Metric::L2Sq, &row, &ctx, n_cells);
            assert_eq!(
                shortlisted.primary, exact.primary,
                "cell {c}: shortlisted primary diverged from exact"
            );
            assert_eq!(shortlisted.primary, c as u32, "cell {c}: wrong placement");
        }
    }

    #[test]
    fn insert_split_centroid_extends_n_cent() {
        let base = synth_centroids(4, 8);
        let sub = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        ];
        let (updated, new_id) = insert_split_centroid(&base, 2, &sub);
        assert_eq!(new_id, 4);
        assert_eq!(updated.n_cent, 5);
        // Counts and centroids must both match n_cent, or the wire encoding
        // (counts adjacent to centroids) truncates and the grid fails to reopen.
        assert_eq!(updated.counts.len(), 5);
        assert_eq!(updated.centroids.len(), 5 * base.dim as usize);
        // Round-trips through the manifest wire format cleanly.
        let bytes = crate::supertable::manifest::encoding::encode_cluster_centroids(&updated);
        let decoded = crate::supertable::manifest::encoding::decode_cluster_centroids(&bytes)
            .expect("split grid must reopen from wire bytes");
        assert_eq!(decoded.n_cent, 5);
        assert_eq!(decoded.centroids.len(), 5 * base.dim as usize);
    }

    #[test]
    fn modality_primitives_separate_and_count_k() {
        let dim = 64usize;
        let threshold = 4.0;
        // Ashman D of the cell's strongest two-means seam, on a strided sample.
        let d_sample = |rows: &[EncodedCellRow], seed: u64| -> f64 {
            let refs: Vec<&EncodedCellRow> = rows.iter().collect();
            let decoded = decode_rows(&refs, dim);
            let c = kmeans(&decoded, dim, 2, SPLIT_KMEANS_ITERS, seed);
            ashman_d(&decoded, dim, &c)
        };
        // The in-memory recursive binary mode count.
        let k_of = |rows: &[EncodedCellRow], seed: u64| -> usize {
            let refs: Vec<&EncodedCellRow> = rows.iter().collect();
            let decoded = decode_rows(&refs, dim);
            let idx: Vec<usize> = (0..rows.len()).collect();
            recursive_binary_k(&decoded, dim, &idx, seed, threshold, MODALITY_MAX_DEPTH)
        };
        // A k=2 split of a single isotropic gaussian is not a no-op: along the
        // split axis each half is a half-normal, so Ashman D sits near the
        // unimodal baseline (~2.6-3.1). The recursive counter returns 1.
        for (sigma, seed) in [(0.1f32, 11u64), (0.03, 13)] {
            let uni = synth_gaussian_cell(dim, 1, 1000, sigma, seed);
            let d = d_sample(&uni, 0);
            assert!(
                (2.0..4.0).contains(&d),
                "unimodal (sigma {sigma}) D should sit near the ~3 baseline, got {d}"
            );
            assert_eq!(
                k_of(&uni, 0),
                1,
                "unimodal cell -> k = 1, got {}",
                k_of(&uni, 0)
            );
        }
        // Two well-separated modes score far above the baseline (~3 vs hundreds).
        let bi = synth_gaussian_cell(dim, 2, 700, 0.02, 12);
        assert!(
            d_sample(&bi, 0) > 100.0,
            "separated modes should score far above the baseline, got {}",
            d_sample(&bi, 0)
        );
        // Three well-separated modes: the recursive counter recovers k = 3,
        // stopping each branch at the unimodal D threshold (not over-fragmenting).
        let tri = synth_gaussian_cell(dim, 3, 700, 0.02, 21);
        assert_eq!(
            k_of(&tri, 0),
            3,
            "three separated modes -> k = 3, got {}",
            k_of(&tri, 0)
        );
    }

    #[test]
    fn plan_sq8_split_separates_two_blobs() {
        let dim = 4usize;
        let mut rows = synth_rows(dim, 10, 0.0);
        rows.extend(synth_rows(dim, 10, 10.0));
        let clusters = synth_centroids(4, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (c0, c1, assign) = plan_sq8_split(&refs, &clusters, 1, Metric::L2Sq);
        assert_eq!(c0.len(), dim);
        assert_eq!(c1.len(), dim);
        let dist: f32 = (0..dim).map(|d| (c0[d] - c1[d]).abs()).sum();
        assert!(dist > 1.0, "split centroids should separate, got {dist}");
        // Assignment is aligned to `rows` and routes each row to one sub-cell;
        // the two well-separated blobs land on opposite sides.
        assert_eq!(assign.len(), rows.len());
        assert_ne!(
            assign[0],
            assign[rows.len() - 1],
            "the two separated blobs should split across sub-cells"
        );
    }

    #[test]
    fn plan_fixed_residual_split_preserves_payloads() {
        let dim = 4usize;
        let mut rows = synth_fixed_rows(dim, 10, 64);
        rows.extend(synth_fixed_rows(dim, 10, 192));
        let before: Vec<(Vec<u8>, Vec<u8>)> = rows
            .iter()
            .map(|row| (row.codes.clone(), row.residuals.clone()))
            .collect();
        let clusters = synth_centroids(4, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (left, right, _assign) = plan_sq8_split(&refs, &clusters, 1, Metric::Cosine);
        let separation: f32 = left.iter().zip(&right).map(|(a, b)| (a - b).abs()).sum();
        assert!(separation > 1.0);
        let after: Vec<(Vec<u8>, Vec<u8>)> = rows
            .iter()
            .map(|row| (row.codes.clone(), row.residuals.clone()))
            .collect();
        assert_eq!(after, before);
    }

    /// Split `n_blobs` equal gaussian blobs `k`-ways and assert every child
    /// lands under `2× mean` with no empty sub-cell (the balance invariant the
    /// split loop needs to converge). Prints the distribution so a run shows how
    /// close to the `⌈n_blobs/k⌉`-blob optimum the seeding got.
    fn assert_kway_split_balanced(
        dim: usize,
        n_blobs: usize,
        per_blob: usize,
        k: usize,
        seed: u64,
    ) {
        let rows = synth_gaussian_cell(dim, n_blobs, per_blob, 0.05, seed);
        let n = rows.len();
        let clusters = synth_centroids(1, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (cents, assign) = plan_sq8_split_kway(&refs, &clusters, 0, Metric::L2Sq, k, true);
        // The planner self-tunes k UPWARD for route fidelity, so the child count
        // is the returned centroid count, not the requested `k`.
        let kk = (cents.len() / dim).max(1);
        let mut counts = vec![0usize; kk];
        for &a in &assign {
            counts[a as usize] += 1;
        }
        let mean = n / kk;
        let max_child = *counts.iter().max().expect("kk >= 1");
        let empty = counts.iter().filter(|&&c| c == 0).count();
        // Route-fidelity: fraction of rows in their NEAREST centroid's child —
        // the ms-scale predictor of nprobe=1 recall (a doc at its nearest cell is
        // found at nprobe=1; a spilled one is not). A naive geometric split would
        // score ≈ 1/kk; capacitated + self-tuned k must stay high.
        let route_faithful = assign
            .iter()
            .enumerate()
            .filter(|&(i, &a)| {
                let rv = dequantize_row(refs[i], dim);
                let nearest = (0..kk)
                    .min_by(|&x, &y| {
                        distance(Metric::L2Sq, &rv, &cents[x * dim..(x + 1) * dim])
                            .partial_cmp(&distance(
                                Metric::L2Sq,
                                &rv,
                                &cents[y * dim..(y + 1) * dim],
                            ))
                            .unwrap_or(Ordering::Equal)
                    })
                    .unwrap_or(0);
                a as usize == nearest
            })
            .count();
        let route_frac = route_faithful as f64 / n as f64;
        let mut sorted = counts.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        eprintln!(
            "[split-test] blobs={n_blobs} n={n} k_req={k} k_used={kk} mean={mean} max={max_child} \
             empty={empty} route_fidelity={route_frac:.3} cells(desc)={sorted:?}",
        );
        assert_eq!(
            empty, 0,
            "no empty sub-cells; blobs={n_blobs} kk={kk} got {counts:?}"
        );
        // Every child ≤ the cap_target (`⌈n/k_req⌉`, the cap-minimum size the
        // planner holds fixed while raising k), plus rounding slack.
        assert!(
            max_child <= n.div_ceil(k) + 1,
            "over cap_target (blobs={n_blobs} k_req={k} k_used={kk}): max {max_child} vs {} \
             got {counts:?}",
            n.div_ceil(k),
        );
        // The whole point of self-tuning: raise k until rows sit in their nearest
        // child. Bar set below the 0.97 self-tune target (which achieved runs
        // hover just above) with margin for k-means's ULP-level non-determinism,
        // but well above the ~0.77 fixed-k capacitated / ~1/k naive-geometric
        // regimes it must never regress to.
        assert!(
            route_frac >= 0.95,
            "low route-fidelity {route_frac:.3} (blobs={n_blobs} k_req={k} k_used={kk}) — \
             self-tuning must raise k until most rows are in their nearest child"
        );
    }

    /// K-way k-means split must stay balanced when the cell holds MANY more
    /// equal-mass blobs than `k` — the 100M regime. A coarse cell spans
    /// `4096 data clusters / 256 grid cells ≈ 16` centers and splits
    /// `k = ⌈rows/cap⌉ ≈ 10`; the grid is uneven, so worst-case cells span more
    /// (~2× the mean → ~32 centers, k≈20). Plain random / single-D² seeding
    /// collides in high dim and piles several blobs onto one child, stalling the
    /// split loop (observed: 256→647 cells, 170k median at 100M). Greedy
    /// k-means++ must land every child under `2× mean` with no empty sub-cell,
    /// across the whole blobs:k regime. (The existing median test uses
    /// `use_kmeans = false` and never exercises this path.)
    #[test]
    fn plan_sq8_split_kway_kmeans_balances_many_equal_blobs() {
        let dim = 1024usize; // prod embedding dim (where distance concentration bites)
        // First-split average: 16 centers, k=10.
        assert_kway_split_balanced(dim, 16, 100, 10, 42);
        // Worst-case uneven cell: ~2× the centers, proportionally larger k — the
        // harder seeding regime (more clusters, greedy trials grow only as ln k).
        assert_kway_split_balanced(dim, 32, 100, 20, 7);
        assert_kway_split_balanced(dim, 64, 60, 40, 101);
    }

    /// Self-tuning must RAISE k above the passed cap-minimum when a cell packs
    /// more groups than that k can hold cleanly: passing k=2 on a 16-group cell
    /// should return more than 2 children (each holding ~whole groups) rather
    /// than a lopsided binary cut.
    #[test]
    fn plan_sq8_split_kway_self_tunes_k_upward() {
        let dim = 1024usize;
        let rows = synth_gaussian_cell(dim, 16, 100, 0.05, 42);
        let clusters = synth_centroids(1, dim as u32);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (cents, assign) = plan_sq8_split_kway(&refs, &clusters, 0, Metric::L2Sq, 2, true);
        let kk = cents.len() / dim;
        let populated = {
            let mut seen = vec![false; kk.max(1)];
            for &a in &assign {
                seen[a as usize] = true;
            }
            seen.iter().filter(|&&s| s).count()
        };
        eprintln!("[self-tune] k_req=2 k_used={kk} populated={populated}");
        assert!(
            kk > 2,
            "self-tuning must raise k above 2 on a 16-group cell, got {kk}"
        );
        assert!(
            populated >= 2,
            "at least 2 sub-cells populated, got {populated}"
        );
    }
}
