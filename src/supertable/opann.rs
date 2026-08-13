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
    sync::{
        Mutex, PoisonError,
        atomic::{AtomicU32, Ordering as AtomicOrdering},
    },
};

use crate::{
    config,
    superfile::vector::{
        cell_posting::{EncodedCellRow, MaterializedIvfRow, manifest_centroid_components_from_row},
        distance::{
            Metric, distance, nearest_k_centroids_bytes, nearest_k_centroids_transposed, normalize,
            relative_score_window,
        },
        kmeans::{kmeans, kmeans_pp},
        quant::BitQuantizer,
        reader::CellFineCalibrationView,
        reservoir::Reservoir,
        rotation::RandomRotation,
        spill::SpilledCellRows,
    },
    supertable::{
        error::BuildError,
        manifest::{
            ClusterCentroids, RABITQ_ADMIT_CELL_SHORTLIST_FRACTION,
            RABITQ_ADMIT_CELL_SHORTLIST_MIN, RabitqAdmitContext,
            list::{WIDTH_LAW_KS, WIDTH_LAW_MAX_K},
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
    let dim = out.len();
    row.rerank_codec
        .ops()
        .expect("encoded row uses a quantized-rerank codec")
        .dequantize_row_into(
            &row.codes,
            &row.residuals,
            dim,
            &row.scale,
            &row.offset,
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
pub(crate) const WIDTH_LAW_QUERY_SAMPLE: usize = 256;

/// Terminal fallback for a `target_recall` that fails validation, used only
/// when the configured value is also out of range. The acceptance bar the
/// project gates on, so a table calibrated after a bad knob is still held
/// to the shipped standard rather than to an arbitrary number.
const ACCEPTANCE_BAR_RECALL: f64 = 0.99;

/// Z-multiplier on the width crossing's sampling error: the crossing
/// tests `mean − Z·SE` against the target, so `0` crosses on the
/// measured mean and a positive Z makes the sample PROVE the narrower
/// width before stamping it.
///
/// `0` (crossing on the mean) is what the width walk uses. It was
/// 1.645 — a one-sided 95% bound — because on a 0.99 target the raw
/// mean turned sampling error into a serving lottery (measured at 10M
/// post-compact: width-1 draws served 0.980–0.991, width-2 draws
/// 0.994, identical corpus), and straddling a hard acceptance bar is a
/// violation. That argument is specific to targets AT the bar. With
/// `vector.target_recall` configurable, a table set below the bar has
/// deliberately traded recall for latency and is not protecting
/// anything by over-provisioning: measured at 10M, target 0.90 stamped
/// width [7,7,9,10] under the bound versus [5,5,6,8] on the mean, and
/// served 0.9523 versus 0.9406 — a quarter more cells bought recall
/// nobody asked for. The mean also tracks the configured target more
/// closely, which is the point of exposing it.
///
/// The variance the bound guarded against is NOT re-measured here (one
/// run per setting says nothing about run-to-run spread); if a table
/// at a bar-level target shows stamp instability, this is the knob
/// that addresses it.
const WIDTH_LAW_CONFIDENCE_Z: f64 = 0.0;

/// Rerank-law distractor pool: each calibration query counts 1-bit-estimate
/// distractors only within its `RERANK_LAW_POOL_CELLS` grid-nearest cells —
/// the pool a width-law sweep would actually scan. A `k` point whose
/// measured width exceeds this stays uncalibrated (the pool under-counts
/// its distractors), falling back to the configured `rerank_mult`.
pub(crate) const RERANK_LAW_POOL_CELLS: usize = 64;

/// Headroom multiplier when sizing a calibration's distractor pool from
/// the stamped width law: the pool must cover the width queries will
/// actually sweep, plus room for the width to grow between
/// recalibrations (drains max-merge width upward; the pool is fixed at
/// freeze time).
const RERANK_LAW_POOL_MARGIN: usize = 2;

/// Distractor-pool size for a calibration over a grid whose width law
/// is already stamped: twice the widest stamped point, floored at the
/// legacy [`RERANK_LAW_POOL_CELLS`] and capped at the grid. A fixed
/// 64-cell pool under-covers fine geometries (measured on Cohere-1M:
/// widths 79-104 at k >= 10), clearing exactly the rerank points the
/// law-served default needs most.
pub(crate) fn rerank_pool_hint(width_for_k: &[u32; WIDTH_LAW_KS.len()], n_cent: usize) -> usize {
    let widest = width_for_k.iter().copied().max().unwrap_or(0) as usize;
    (widest * RERANK_LAW_POOL_MARGIN)
        .max(RERANK_LAW_POOL_CELLS)
        .min(n_cent.max(1))
}

/// Rerank-law estimate histogram resolution: per-query counts of pool-row
/// 1-bit estimates, binned linearly over `[-Σ|q_rot|, +Σ|q_rot|]` (the
/// sign-dot estimator's exact range). A candidate's distractor count reads
/// the prefix INCLUDING its own bin — rank error is bounded by one bin's
/// occupancy and always over-counts (a wider law, never a narrower one).
const RERANK_LAW_EST_BINS: usize = 4096;
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
    tops: Mutex<Vec<Vec<(f32, u32, i128, f32)>>>,
    /// `(query index, stable id, cell) -> fine-centroid rank` of that
    /// candidate's fine cluster within THAT cell, recorded by
    /// [`Self::observe_shard_views`] after each shard is packed (fine
    /// clusters exist only post-pack). The cell is part of the key:
    /// boundary replicas of one row live in several cells, and a rank
    /// observed in one cell's fine geometry says nothing about another's —
    /// the law walk looks up the SURVIVING copy's cell. Entries for
    /// candidates later evicted from `tops` are harmless — the walk only
    /// looks up ids that survived.
    fine_ranks: Mutex<HashMap<(u32, i128, u32), u32>>,
    /// Largest fine-cluster count observed across packed cells: the depth
    /// law's search domain.
    max_fine: AtomicU32,
    /// Distractor-pool size (cells) chosen at [`Self::freeze`]; the
    /// legacy floor until then.
    pool_cells: usize,
    /// Recall this drain calibrates its laws to hold — the per-table
    /// `vector.target_recall`. EVERY stage (width, fine depth, rerank
    /// budget) crosses at this value; there is deliberately no padded
    /// per-stage variant. The stages compound, but on the SAME queries
    /// — a query whose neighbors land in the probed cells is also the
    /// query whose neighbors sit shallow in them — so the product runs
    /// far above the independent-stages worst case. Measured on
    /// Cohere-10M with every stage at the target: 0.993 served 0.9959,
    /// 0.93 served 0.9693, 0.90 served 0.9577. Padding would only
    /// widen probes for a loss that does not occur.
    target_recall: f64,
    /// Rerank-law observation state, armed by [`Self::freeze`]; `None`
    /// (e.g. planted test fixtures) measures no rerank law.
    rerank: Option<RerankLawObservation>,
}

/// Streaming state for the rerank law: per query, the 1-bit-encoded query
/// (same rotation + estimator as the scan's shortlist) and a histogram of
/// pool-row estimates, from which each exact-top-k candidate's estimate
/// rank — the survivor budget that keeps it — is read at [`finish`].
///
/// [`finish`]: WidthLawCalibration::finish
struct RerankLawObservation {
    quant: BitQuantizer,
    /// Flat `n_queries x dim` rotated queries.
    q_rot: Vec<f32>,
    /// Per query `Σ q_rot[d]` — the estimator's per-query identity term.
    q_total: Vec<f32>,
    /// Per query `Σ |q_rot[d]|` — the estimate range for binning.
    q_l1: Vec<f32>,
    /// Per query: ascending cell ids of its `RERANK_LAW_POOL_CELLS`
    /// grid-nearest cells.
    pools: Vec<Vec<u32>>,
    /// Per query estimate histogram (`RERANK_LAW_EST_BINS` bins,
    /// bin 0 = best estimate). `u64` + saturating merges: a dense pool
    /// on a large table must never wrap a bin and silently NARROW the
    /// measured survivor budget.
    hist: Mutex<Vec<Vec<u64>>>,
}

impl RerankLawObservation {
    /// Histogram bin for `est` under this query's `[-l1, +l1]` range;
    /// bin 0 holds the best (largest) estimates.
    fn bin(&self, qi: usize, est: f32) -> usize {
        let l1 = self.q_l1[qi];
        if l1 <= 0.0 {
            return RERANK_LAW_EST_BINS - 1;
        }
        let frac = ((l1 - est) / (2.0 * l1)).clamp(0.0, 1.0);
        ((frac * RERANK_LAW_EST_BINS as f32) as usize).min(RERANK_LAW_EST_BINS - 1)
    }
}

/// Both laws the drain calibration measures — cells for the width sweep,
/// fine runs per probed cell for the depth floor. Same knots, same
/// coverage target, same monotone flooring.
pub(crate) struct CalibratedLaws {
    pub(crate) width_for_k: [u32; WIDTH_LAW_KS.len()],
    pub(crate) fine_for_k: [u32; WIDTH_LAW_KS.len()],
    /// Global 1-bit-estimate survivor budget (rows) for 0.99 containment
    /// of the exact top-k — the measured replacement for `k x rerank_mult`.
    pub(crate) rerank_for_k: [u32; WIDTH_LAW_KS.len()],
    /// Distractor-pool size (cells) this calibration measured rerank
    /// against — the stamp records it so a later, wider law knows
    /// whether the budget's evidence still covers it.
    pub(crate) pool_cells: u32,
}

#[cfg(test)]
mod pool_hint_tests {
    use super::*;

    /// The pool covers twice the widest stamped point, never dips below
    /// the legacy floor, and never exceeds the grid.
    #[test]
    fn rerank_pool_hint_scales_with_the_stamped_width() {
        // No stamp yet: legacy floor.
        assert_eq!(rerank_pool_hint(&[0, 0, 0, 0], 256), RERANK_LAW_POOL_CELLS);
        // Narrow law: floor still wins.
        assert_eq!(rerank_pool_hint(&[1, 2, 8, 16], 256), RERANK_LAW_POOL_CELLS);
        // The Cohere-1M shape that motivated this: widths 79-104 need a
        // pool past the floor.
        assert_eq!(rerank_pool_hint(&[33, 79, 97, 104], 256), 208);
        // Grid caps the pool.
        assert_eq!(rerank_pool_hint(&[33, 79, 97, 104], 150), 150);
        // Tiny grids never zero the pool.
        assert_eq!(rerank_pool_hint(&[1, 0, 0, 0], 0), 1);
    }
}

/// Post-stamp guard shared by both stamp sites (drain max-merge and
/// recalibration replace): a rerank point whose STAMPED width exceeds
/// the calibration's distractor pool is cleared. The previous value was
/// certified against a narrower geometry — its distractor counts stop
/// at [`RERANK_LAW_POOL_CELLS`] cells — so carrying it into a wider law
/// silently under-provisions the survivor budget; `0` falls back to the
/// configured `rerank_mult`, which is the safe default for the width
/// the pool never measured.
pub(crate) fn clear_rerank_beyond_pool(
    width_for_k: &[u32; WIDTH_LAW_KS.len()],
    rerank_for_k: &mut [u32; WIDTH_LAW_KS.len()],
    pool_cells: &[u32; WIDTH_LAW_KS.len()],
) {
    for ((w, r), pool) in width_for_k
        .iter()
        .zip(rerank_for_k.iter_mut())
        .zip(pool_cells.iter())
    {
        if *w > *pool {
            *r = 0;
        }
    }
}

/// Per-knot max-merge of a fresh rerank measurement into the live stamp,
/// carrying pool provenance with each kept value: a knot keeps the pool
/// of WHICHEVER calibration's value survives (ties keep the wider pool —
/// the stronger certificate). Shared by both stamp sites so mixed-origin
/// stamps stay per-knot honest — a surviving narrow-pool point must not
/// invalidate fresh wide-pool neighbors, and a fresh pool must not
/// launder an old point past the width its own evidence covered.
pub(crate) fn merge_rerank_with_pools(
    rerank: &mut [u32; WIDTH_LAW_KS.len()],
    pools: &mut [u32; WIDTH_LAW_KS.len()],
    measured: &[u32; WIDTH_LAW_KS.len()],
    measured_pool: u32,
) {
    for ((slot, pool), m) in rerank.iter_mut().zip(pools.iter_mut()).zip(measured.iter()) {
        if *m > *slot {
            *slot = *m;
            *pool = measured_pool;
        } else if *m == *slot && *m > 0 {
            *pool = (*pool).max(measured_pool);
        }
        // m < slot (incl. m == 0): keep the old value and its pool.
    }
}

/// Floor measured law points to be monotone in `k`, skipping unmeasured
/// (`0`) points — shared by the width and fine-depth walks.
fn floor_monotone(law: &mut [u32; WIDTH_LAW_KS.len()]) {
    let mut floor = 0u32;
    for w in law.iter_mut().filter(|w| **w > 0) {
        *w = (*w).max(floor);
        floor = *w;
    }
}

impl WidthLawCalibration {
    pub(crate) fn new(dim: usize, metric: Metric, target_recall: f64) -> Self {
        // The target arrives from user YAML, so validate it here rather than
        // let a bad value reach the walk, where it fails silently: NaN makes
        // every crossing comparison false, so nothing is ever stamped and the
        // rerank cutoff collapses to zero; zero or negative is satisfied by
        // the first candidate, stamping a width of 1 that under-provisions
        // every query; above 1.0 no width can ever satisfy it. The configured
        // value is the fallback but not a trusted one — it comes off the same
        // YAML surface — so it is checked too, terminating on the shipped bar.
        let valid = |v: f64| v.is_finite() && v > 0.0 && v <= 1.0;
        let fallback_target = {
            let configured = config::global().vector.target_recall;
            if valid(configured) {
                configured
            } else {
                ACCEPTANCE_BAR_RECALL
            }
        };
        let target_recall = if valid(target_recall) {
            target_recall
        } else {
            tracing::warn!(
                target_recall,
                fallback = fallback_target,
                "vector.target_recall must be in (0, 1]; falling back to the default"
            );
            fallback_target
        };
        Self {
            dim,
            metric,
            reservoir: Reservoir::new(WIDTH_LAW_QUERY_SAMPLE, dim, WIDTH_LAW_SAMPLE_SEED),
            slot_ids: Vec::with_capacity(WIDTH_LAW_QUERY_SAMPLE),
            dequant_scratch: vec![0f32; dim],
            frozen: None,
            tops: Mutex::new(Vec::new()),
            fine_ranks: Mutex::new(HashMap::new()),
            max_fine: AtomicU32::new(0),
            pool_cells: RERANK_LAW_POOL_CELLS,
            target_recall,
            rerank: None,
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

    /// Freeze the sampled queries and arm the rerank-law observation
    /// (rotated queries, distractor pools from the grid, empty histograms).
    /// Called once, after the last batch spilled and before cell packing
    /// scores. The grid must be final by now — spills are already assigned
    /// to its cells.
    /// `pool_cells` sizes each query's distractor pool (see
    /// [`rerank_pool_hint`]); clamped to `[RERANK_LAW_POOL_CELLS,
    /// n_cent]` so a hint can never under-pool below the legacy floor
    /// or over-pool past the grid.
    pub(crate) fn freeze(&mut self, grid: &ClusterCentroids, rot_seed: u64, pool_cells: usize) {
        self.pool_cells = pool_cells
            .max(RERANK_LAW_POOL_CELLS)
            .min((grid.n_cent as usize).max(1));
        let queries = self.reservoir.sample().to_vec();
        let ids = self.slot_ids.clone();
        let n_queries = ids.len();
        *self.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![Vec::new(); n_queries];
        if n_queries > 0 && grid.n_cent > 0 {
            let rotation = RandomRotation::new(self.dim, rot_seed);
            let mut q_rot = vec![0f32; n_queries * self.dim];
            let mut q_total = Vec::with_capacity(n_queries);
            let mut q_l1 = Vec::with_capacity(n_queries);
            let mut pools = Vec::with_capacity(n_queries);
            for (qi, q) in queries.chunks_exact(self.dim).enumerate() {
                let out = &mut q_rot[qi * self.dim..(qi + 1) * self.dim];
                rotation.apply(q, out);
                q_total.push(out.iter().sum());
                q_l1.push(out.iter().map(|v| v.abs()).sum());
                let mut pool: Vec<u32> = grid
                    .rank_cells(self.metric, q)
                    .into_iter()
                    .take(self.pool_cells)
                    .map(|(cell, _)| cell)
                    .collect();
                pool.sort_unstable();
                pools.push(pool);
            }
            self.rerank = Some(RerankLawObservation {
                quant: BitQuantizer::new(self.dim),
                q_rot,
                q_total,
                q_l1,
                pools,
                hist: Mutex::new(vec![Vec::new(); n_queries]),
            });
        }
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
        let mut partial: Vec<Vec<(f32, u32, i128, f32)>> = vec![Vec::new(); n_queries];
        let members = self.pool_members(cell);
        let mut hist_local: HashMap<usize, Vec<u64>> = HashMap::new();
        let mut reader = spill.reader()?;
        let mut remaining = spill.n_rows();
        let mut scratch = vec![0f32; self.dim];
        while remaining > 0 {
            let chunk = reader.next_chunk(WIDTH_LAW_SCORE_CHUNK.min(remaining))?;
            remaining -= chunk.len();
            self.score_slice(
                frozen,
                cell,
                &chunk,
                &members,
                &mut scratch,
                &mut partial,
                &mut hist_local,
            );
        }
        self.merge_partial(partial, hist_local);
        Ok(())
    }

    /// [`Self::score_cell`] for already-materialized rows: the compaction
    /// recalibration pass reads live rows back from stored superfiles
    /// (no spill exists), then scores them through the same core.
    pub(crate) fn score_rows(
        &self,
        cell: u32,
        rows: &[MaterializedIvfRow],
    ) -> Result<(), BuildError> {
        let Some(frozen) = self.frozen.as_ref() else {
            return Err(BuildError::Store(
                "width-law score_rows before freeze".into(),
            ));
        };
        let n_queries = frozen.ids.len();
        if n_queries == 0 {
            return Ok(());
        }
        let mut partial: Vec<Vec<(f32, u32, i128, f32)>> = vec![Vec::new(); n_queries];
        let members = self.pool_members(cell);
        let mut hist_local: HashMap<usize, Vec<u64>> = HashMap::new();
        let mut scratch = vec![0f32; self.dim];
        for chunk in rows.chunks(WIDTH_LAW_SCORE_CHUNK) {
            self.score_slice(
                frozen,
                cell,
                chunk,
                &members,
                &mut scratch,
                &mut partial,
                &mut hist_local,
            );
        }
        self.merge_partial(partial, hist_local);
        Ok(())
    }

    /// One-lock merge of a cell's scored partials into the per-query
    /// accumulators, plus its estimate-histogram deltas — the
    /// correctness-bearing epilogue shared by [`Self::score_cell`] and
    /// [`Self::score_rows`].
    fn merge_partial(
        &self,
        partial: Vec<Vec<(f32, u32, i128, f32)>>,
        hist_local: HashMap<usize, Vec<u64>>,
    ) {
        let k_max = WIDTH_LAW_MAX_K;
        let mut tops = self.tops.lock().unwrap_or_else(PoisonError::into_inner);
        for (qi, cand) in partial.into_iter().enumerate() {
            merge_candidates(&mut tops[qi], cand, k_max);
        }
        drop(tops);
        if let Some(rl) = self.rerank.as_ref()
            && !hist_local.is_empty()
        {
            let mut hist = rl.hist.lock().unwrap_or_else(PoisonError::into_inner);
            for (qi, delta) in hist_local {
                let slot = &mut hist[qi];
                if slot.is_empty() {
                    *slot = delta;
                } else {
                    for (a, b) in slot.iter_mut().zip(delta) {
                        *a = a.saturating_add(b);
                    }
                }
            }
        }
    }

    /// Queries whose rerank-law distractor pool contains `cell`.
    fn pool_members(&self, cell: u32) -> Vec<usize> {
        let Some(rl) = self.rerank.as_ref() else {
            return Vec::new();
        };
        (0..rl.pools.len())
            .filter(|&qi| rl.pools[qi].binary_search(&cell).is_ok())
            .collect()
    }

    /// Shared scoring core: one chunk of rows against the frozen queries,
    /// appended to `partial` and re-bounded to the merge's `k_max`. For
    /// queries whose distractor pool contains `cell`, every row's 1-bit
    /// estimate — the SAME `estimate_dot_rotated_with_total` the scan's
    /// shortlist ranks by — is histogrammed into `hist_local`, and each
    /// candidate carries its own estimate so [`Self::finish`] can read its
    /// survivor rank; candidates outside the pool carry `NEG_INFINITY`
    /// (unrankable — conservative).
    fn score_slice(
        &self,
        frozen: &WidthLawQueries,
        cell: u32,
        rows: &[MaterializedIvfRow],
        members: &[usize],
        scratch: &mut [f32],
        partial: &mut [Vec<(f32, u32, i128, f32)>],
        hist_local: &mut HashMap<usize, Vec<u64>>,
    ) {
        let k_max = WIDTH_LAW_MAX_K;
        let rl = self.rerank.as_ref();
        let mut est_of = vec![f32::NEG_INFINITY; frozen.ids.len()];
        for row in rows {
            if let Some(rl) = rl
                && row.rabitq_code.len() == rl.quant.code_bytes()
            {
                for &qi in members {
                    // Self-hit: the sampled query's own row is not a
                    // distractor. `finish` already excludes it from the
                    // exact top-k (below), so counting it here would
                    // inflate every measured rerank budget.
                    if row.stable_id == frozen.ids[qi] {
                        continue;
                    }
                    let q_rot = &rl.q_rot[qi * self.dim..(qi + 1) * self.dim];
                    let est = rl.quant.estimate_dot_rotated_with_total(
                        q_rot,
                        &row.rabitq_code,
                        rl.q_total[qi],
                    );
                    // A non-finite estimate (corrupt code, degenerate
                    // quantizer) is not rank evidence: NaN casts to bin 0 —
                    // the BEST bin — and would count as a top distractor in
                    // every pooled query's histogram. Leave the candidate
                    // unrankable (NEG_INFINITY), which the exact walk and
                    // the dedup tie-break already treat conservatively.
                    if est.is_finite() {
                        est_of[qi] = est;
                        let bins = hist_local
                            .entry(qi)
                            .or_insert_with(|| vec![0u64; RERANK_LAW_EST_BINS]);
                        let bin = rl.bin(qi, est);
                        bins[bin] = bins[bin].saturating_add(1);
                    }
                }
            }
            dequantize_row_into(&row.encoded, scratch);
            if self.metric == Metric::Cosine {
                // The rerank kernels divide by the stored row norm;
                // unit-normalizing the row lets the shared [`distance`]
                // kernel (which assumes unit inputs for cosine) score
                // with the same ranking. Query scaling is per-query
                // monotone and cannot reorder its candidates.
                normalize(scratch);
            }
            for (qi, q) in frozen.queries.chunks_exact(self.dim).enumerate() {
                // Self-hit: a sampled query trivially covers itself.
                if row.stable_id == frozen.ids[qi] {
                    continue;
                }
                partial[qi].push((
                    distance(self.metric, q, scratch),
                    cell,
                    row.stable_id,
                    est_of[qi],
                ));
            }
            if rl.is_some() {
                for &qi in members {
                    est_of[qi] = f32::NEG_INFINITY;
                }
            }
        }
        // Bound the per-cell partials the same way the merge does.
        for cand in partial.iter_mut() {
            truncate_ascending(cand, k_max);
        }
    }

    /// Record each surviving candidate's fine-centroid rank within its
    /// cell, from a freshly packed shard. Runs once per shard, after
    /// [`Self::score_cell`] merged that shard's cells: fine clusters only
    /// exist post-pack, so depth is observed here rather than at scoring.
    /// Ranking mirrors query-time fine selection: raw fp32 centroid
    /// distance, ascending, ties by lower cluster index.
    pub(crate) fn observe_shard_views(&self, views: &[CellFineCalibrationView]) {
        let Some(frozen) = self.frozen.as_ref() else {
            return;
        };
        if frozen.ids.is_empty() {
            return;
        }
        // Candidates per cell, gathered once — observation touches only
        // rows that currently matter to some query's top-k.
        let per_cell: HashMap<u32, Vec<(u32, i128)>> = {
            let tops = self.tops.lock().unwrap_or_else(PoisonError::into_inner);
            let mut map: HashMap<u32, Vec<(u32, i128)>> = HashMap::new();
            for (qi, cands) in tops.iter().enumerate() {
                for &(_, cell, id, _) in cands {
                    map.entry(cell).or_default().push((qi as u32, id));
                }
            }
            map
        };
        for view in views {
            let Some(cell_id) = view.cell_id else {
                continue;
            };
            let Some(cands) = per_cell.get(&cell_id) else {
                continue;
            };
            if view.n_fine == 0 || view.dim != self.dim {
                continue;
            }
            self.max_fine
                .fetch_max(view.n_fine as u32, AtomicOrdering::Relaxed);
            // Per-query fine ranking, computed once per query that has
            // candidates in this cell.
            let mut rank_cache: HashMap<u32, Vec<u32>> = HashMap::new();
            let mut ranks = self
                .fine_ranks
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            for &(qi, id) in cands {
                let Some(&cluster) = view.cluster_of_stable.get(&id) else {
                    continue;
                };
                let rank_of = rank_cache.entry(qi).or_insert_with(|| {
                    let q = &frozen.queries[qi as usize * self.dim..(qi as usize + 1) * self.dim];
                    // Full ranking (`k = n_fine`) through the shared
                    // row-major centroid-scan owner — identical distance
                    // and tie-break semantics to query-time fine selection.
                    let ranked = nearest_k_centroids_bytes(
                        self.metric,
                        q,
                        &view.fine_centroids_bytes,
                        view.n_fine,
                        view.dim,
                        view.n_fine,
                    );
                    let mut rank_of = vec![0u32; view.n_fine];
                    for (rank, (c, _)) in ranked.iter().enumerate() {
                        rank_of[*c as usize] = rank as u32;
                    }
                    rank_of
                });
                if let Some(&r) = rank_of.get(cluster as usize) {
                    ranks.insert((qi, id, cell_id), r);
                }
            }
        }
    }

    /// Extract the width law: cells (in the grid's routing order) needed
    /// for mean target-recall coverage of the exact top-k
    /// at each [`WIDTH_LAW_KS`] point. Each point is measured over the
    /// queries whose candidate count reaches its `k` — one boundary-starved
    /// query excludes itself, not the whole sample. Points NO query can
    /// support stay `0` (uncalibrated). Measured points are floored to be
    /// monotone in `k`. `None` when nothing was sampled.
    pub(crate) fn finish(self, grid: &ClusterCentroids) -> Option<CalibratedLaws> {
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
        // Per-query squared coverages alongside the sums: the crossing
        // rule needs the sample's own variance to know when its mean is
        // trustworthy (see the confidence note at the crossing below).
        let mut coverage_sq_sums: Vec<Vec<f64>> = vec![vec![0f64; n_cells]; WIDTH_LAW_KS.len()];
        let mut support = [0usize; WIDTH_LAW_KS.len()];
        // Depth: same prefix-walk over fine-centroid ranks. A candidate
        // whose rank was never observed counts as uncoverable (conservative
        // — deepens the law rather than narrowing it).
        let fine_ranks = self
            .fine_ranks
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let max_fine = self.max_fine.load(AtomicOrdering::Relaxed).max(1) as usize;
        let mut fine_law = [0u32; WIDTH_LAW_KS.len()];
        let mut fine_sums: Vec<Vec<f64>> = vec![vec![0f64; max_fine]; WIDTH_LAW_KS.len()];
        // Rerank: per query, prefix sums of its estimate histogram give
        // each candidate's distractor count (rows with a better-or-equal
        // 1-bit estimate in its pool) — the survivor budget that keeps it.
        let rerank_prefix: Option<Vec<Vec<u64>>> = self.rerank.as_ref().map(|rl| {
            rl.hist
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .iter()
                .map(|h| {
                    let mut run = 0u64;
                    h.iter()
                        .map(|&c| {
                            run = run.saturating_add(c);
                            run
                        })
                        .collect()
                })
                .collect()
        });
        let mut rerank_law = [0u32; WIDTH_LAW_KS.len()];
        // Pooled candidate ranks per k point: mean coverage at budget N is
        // #(query, candidate) pairs with rank <= N over (support x k), so
        // the coverage crossing is EXACTLY the ceil(target x len)-th
        // smallest pooled rank — the same mean-coverage semantic as the
        // width and fine walks (a mean of per-query quantile budgets is
        // NOT: easy queries would subsidize hard ones below target).
        let mut rerank_ranks: [Vec<u64>; WIDTH_LAW_KS.len()] = Default::default();
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
                for (_, cell, _, _) in &sorted[..k] {
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
                    let x = f64::from(covered) / k as f64;
                    coverage_sums[ki][rank] += x;
                    coverage_sq_sums[ki][rank] += x * x;
                }
                let mut per_fine_rank = vec![0u32; max_fine];
                for (_, cell, id, _) in &sorted[..k] {
                    // Look up the rank observed for the SURVIVING copy's
                    // cell — a boundary replica's rank in another cell's
                    // fine geometry would mis-measure the depth this
                    // candidate actually needs at query time.
                    if let Some(&r) = fine_ranks.get(&(qi as u32, *id, *cell)) {
                        per_fine_rank[(r as usize).min(max_fine - 1)] += 1;
                    }
                }
                if let (Some(rl), Some(prefix)) = (self.rerank.as_ref(), rerank_prefix.as_ref()) {
                    // Pool every candidate's distractor count (rows with a
                    // better-or-equal 1-bit estimate in the query's pool).
                    // Unrankable candidates (outside the pool, or an empty
                    // histogram) pool as MAX: they push the coverage
                    // crossing up or leave the point unsupported —
                    // conservative both ways.
                    for (_, _, _, est) in &sorted[..k] {
                        if est.is_finite() && !prefix[qi].is_empty() {
                            rerank_ranks[ki].push(prefix[qi][rl.bin(qi, *est)]);
                        } else {
                            rerank_ranks[ki].push(u64::MAX);
                        }
                    }
                }
                let mut fine_covered = 0u32;
                for (rank, count) in per_fine_rank.iter().enumerate() {
                    fine_covered += count;
                    fine_sums[ki][rank] += f64::from(fine_covered) / k as f64;
                }
            }
        }
        for (ki, sums) in coverage_sums.iter().enumerate() {
            if support[ki] == 0 {
                continue;
            }
            // Width crossing with the sample's own confidence: stamp the
            // smallest width whose coverage LOWER BOUND (mean − z·SE, SE
            // measured from this sample's per-query coverage variance)
            // clears the target — not the raw mean. The raw-mean crossing
            // measured as a run-to-run lottery on marginal geometry: the
            // post-compact 3.5K-cell grid sits with true width-1 coverage
            // near the 0.99 target, so identical corpora stamped width 1
            // (serving 0.980–0.991) or width 2 (serving 0.994) depending
            // on the reservoir draw. Under uncertainty the stamp must
            // round UP — the sample has to PROVE the narrower width. A
            // uniform sample (every query fully covered, variance zero —
            // all synthetic post-drain shapes) has SE = 0 and stamps
            // exactly as before; only genuinely marginal crossings widen.
            let n = support[ki] as f64;
            let target = self.target_recall * n;
            if let Some(rank) = sums.iter().enumerate().position(|(rank, &s)| {
                let mean = s / n;
                let var = (coverage_sq_sums[ki][rank] / n - mean * mean).max(0.0);
                let se = (var / n).sqrt();
                (mean - WIDTH_LAW_CONFIDENCE_Z * se) * n >= target
            }) {
                law[ki] = (rank + 1) as u32;
            }
            let stage_target = self.target_recall * support[ki] as f64;
            if let Some(rank) = fine_sums[ki].iter().position(|&s| s >= stage_target) {
                fine_law[ki] = (rank + 1) as u32;
            }
        }
        for (ki, &w) in law.iter().enumerate() {
            // A k point is rerank-measurable only where the pool covered
            // the measured width — beyond it the distractor counts are
            // partial and the point stays uncalibrated.
            let ranks = &mut rerank_ranks[ki];
            if w == 0 || w as usize > self.pool_cells || ranks.is_empty() {
                continue;
            }
            // Mean-coverage crossing at the stage target: the
            // ceil(target x pooled)-th smallest pooled rank. MAX at the
            // crossing means the evidence cannot certify the target —
            // the point stays uncalibrated (constant fallback).
            ranks.sort_unstable();
            let needed = (self.target_recall * ranks.len() as f64).ceil() as usize;
            if let Some(&crossing) = ranks.get(needed.saturating_sub(1).min(ranks.len() - 1))
                && crossing != u64::MAX
            {
                rerank_law[ki] = crossing.min(u64::from(u32::MAX)) as u32;
            }
        }
        // Coverage need only grows with k, so each measured point is floored
        // by the measured points below it — sampling noise near the target
        // must never let a larger k probe FEWER cells than a smaller one (a
        // recall inversion at query time). Unmeasured points (0) stay 0; the
        // interpolator skips them.
        floor_monotone(&mut law);
        floor_monotone(&mut fine_law);
        floor_monotone(&mut rerank_law);
        (law.iter().any(|&w| w > 0)).then_some(CalibratedLaws {
            width_for_k: law,
            fine_for_k: fine_law,
            rerank_for_k: rerank_law,
            pool_cells: self.pool_cells as u32,
        })
    }
}

/// Merge one cell's candidates into a query's accumulator: collapse to the
/// best-scored copy per stable id FIRST (boundary replicas of one row are
/// one neighbor), then keep the ascending-best `cap`. The dedup must
/// precede the truncate — replicated copies of near rows filling raw slots
/// would evict distinct farther neighbors and stamp a narrower law than a
/// real query experiences. Equal-distance copies (the NORMAL case for a
/// boundary replica — same vector, same exact distance) tie-break toward
/// a rankable candidate: a copy scored outside the rerank pool carries
/// `NEG_INFINITY` est, and keeping that one starves the rerank histogram
/// of the row's rank, under-measuring the budget.
fn merge_candidates(
    acc: &mut Vec<(f32, u32, i128, f32)>,
    mut cand: Vec<(f32, u32, i128, f32)>,
    cap: usize,
) {
    acc.append(&mut cand);
    acc.sort_unstable_by(|a, b| {
        a.2.cmp(&b.2)
            .then_with(|| a.0.total_cmp(&b.0))
            .then_with(|| b.3.is_finite().cmp(&a.3.is_finite()))
            // Total order: two equally-rankable equal-distance replicas
            // tie-break on cell, or the kept copy would depend on merge
            // order — and the fine-rank lookup is keyed by the SURVIVING
            // copy's cell, so an arbitrary keep would make the stamped
            // depth law nondeterministic across runs.
            .then_with(|| a.1.cmp(&b.1))
    });
    acc.dedup_by_key(|c| c.2);
    truncate_ascending(acc, cap);
}

/// Keep the ascending-best `cap` candidates in place.
fn truncate_ascending(cand: &mut Vec<(f32, u32, i128, f32)>, cap: usize) {
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
///
/// Test-only since the batched split landed: production folds every split
/// through [`insert_split_centroids_batch`], and this singleton remains as
/// the reference implementation its equivalence test folds against.
#[cfg(test)]
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

/// Fold every split of `splits` (`(parent_cell, sub_centroids, k)`, parent
/// cells distinct) into ONE grown grid, in the given order. Equivalent to
/// folding the singleton `insert_split_centroids` (now the test-only
/// reference implementation) sequentially over `splits`, but with a single
/// centroid-buffer allocation and one wire-invariant rebuild instead of one
/// per split.
///
/// Child ids are positional ordinals minted off the end of the grid (the id
/// IS the array index IS the reader routing key), so each split's appended
/// ids start at `base.n_cent + Σ (k_j − 1)` over the splits before it —
/// callers must fix the order BEFORE calling (the split pass uses ascending
/// parent id) and must not compute ids per-split off the shared base.
/// Returns the grown grid and, per split, its `k` child ids (index 0 == the
/// reused parent id), aligned to that split's `sub_centroids` (`k * dim`
/// fp32). New children carry count 0; the caller sets real counts on the
/// GROWN grid via [`apply_cell_count_updates`] (out-of-range ids there are
/// silently dropped, so count application must never precede this fold).
pub(crate) fn insert_split_centroids_batch(
    base: &ClusterCentroids,
    splits: &[(u32, &[f32], usize)],
) -> (ClusterCentroids, Vec<Vec<u32>>) {
    debug_assert!(
        {
            let mut parents: Vec<u32> = splits.iter().map(|(p, _, _)| *p).collect();
            parents.sort_unstable();
            parents.windows(2).all(|w| w[0] != w[1])
        },
        "batch splits must target distinct parent cells"
    );
    let dim = base.dim as usize;
    let old_n = base.n_cent as usize;
    let appended: usize = splits.iter().map(|(_, _, k)| k - 1).sum();
    let new_n = old_n + appended;

    let mut fp32 = vec![0f32; new_n * dim];
    for c in 0..old_n {
        fp32[c * dim..(c + 1) * dim].copy_from_slice(base.centroid(c));
    }
    let mut ids_per_split = Vec::with_capacity(splits.len());
    let mut next_id = old_n;
    for &(cell_id, sub_centroids, k) in splits {
        debug_assert_eq!(sub_centroids.len(), k * dim);
        // Sub-cell 0 reuses the split cell's slot; 1..k append.
        let p = cell_id as usize;
        fp32[p * dim..(p + 1) * dim].copy_from_slice(&sub_centroids[..dim]);
        let mut ids = vec![cell_id];
        for j in 1..k {
            fp32[next_id * dim..(next_id + 1) * dim]
                .copy_from_slice(&sub_centroids[j * dim..(j + 1) * dim]);
            ids.push(next_id as u32);
            next_id += 1;
        }
        ids_per_split.push(ids);
    }

    // Counts must have one entry per cell (see the sibling comment in
    // [`insert_split_centroids`]): a short counts vec silently passes
    // in-memory but truncates the wire encoding.
    let mut counts = base.counts.clone();
    counts.resize(new_n, 0);
    let updated = ClusterCentroids::from_fp32(new_n as u32, base.dim, &fp32, counts);
    (updated, ids_per_split)
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

    use bytes::Bytes;

    use super::*;

    /// Every stage crosses at the target itself — no padded per-stage
    /// variant, no derived second constant. Pins that the shipped
    /// default still crosses width at 0.99 (the historical
    /// `WIDTH_LAW_TARGET_COVERAGE`), which is what the acceptance
    /// gates measure against.
    #[test]
    fn every_stage_crosses_at_the_configured_target() {
        /// The historical width crossing this knob's default preserves.
        const SHIPPED_WIDTH: f64 = 0.99;
        assert_eq!(shipped_target_recall(), SHIPPED_WIDTH);
        for target in [0.80, 0.90, 0.95, 0.99] {
            let cal = WidthLawCalibration::new(8, Metric::Cosine, target);
            assert_eq!(
                cal.target_recall, target,
                "the calibration carries the target unmodified"
            );
        }
    }

    /// A target outside `(0, 1]` never reaches the walk. Each of these
    /// fails silently if stored: NaN stamps nothing and zeroes the rerank
    /// cutoff, `<= 0` stamps width 1, `> 1` can never be crossed.
    #[test]
    fn an_out_of_range_target_falls_back_to_the_configured_default() {
        for bad in [f64::NAN, 0.0, -0.5, 1.5, f64::INFINITY, f64::NEG_INFINITY] {
            let cal = WidthLawCalibration::new(8, Metric::Cosine, bad);
            assert_eq!(
                cal.target_recall,
                shipped_target_recall(),
                "target {bad} must fall back, not be stored"
            );
        }
        // The boundary itself is a legal target and must survive untouched.
        let cal = WidthLawCalibration::new(8, Metric::Cosine, 1.0);
        assert_eq!(cal.target_recall, 1.0);
    }

    /// The shipped `vector.target_recall`, read from the live config so
    /// the pin below tracks the YAML default instead of a copy of it.
    fn shipped_target_recall() -> f64 {
        crate::config::global().vector.target_recall
    }

    /// Raw fp32-le centroid-region bytes for planted calibration views.
    fn fp32_le_bytes(vals: &[f32]) -> Bytes {
        Bytes::from(
            vals.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    }

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
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        // (score, cell, stable id): id 1 replicated across cells 0 and 1;
        // ids 2..=8 fill the near cells; ids 9 and 10 sit in cell 3 and
        // only enter the top-10 once the replicas collapse to one slot.
        let ninf = f32::NEG_INFINITY;
        let mut cands = vec![(0.01, 0, 1, ninf), (0.02, 1, 1, ninf), (0.03, 1, 1, ninf)];
        cands
            .extend((2..=8).map(|id| (0.03 + id as f32 * 0.01, (id % 2) as u32, id as i128, ninf)));
        cands.push((0.5, 3, 9, ninf));
        cands.push((0.6, 3, 10, ninf));
        // Plant through the same merge the scorer uses — dedup happens
        // there, BEFORE the truncate, so replicas can never occupy slots.
        let mut acc = Vec::new();
        merge_candidates(&mut acc, cands, WIDTH_LAW_MAX_K);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];

        let law = cal
            .finish(&grid)
            .expect("law from planted candidates")
            .width_for_k;
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

    /// The fine-depth law mirrors the width walk over fine-centroid
    /// ranks: candidates observed via
    /// [`WidthLawCalibration::observe_shard_views`] count at their fine
    /// cluster's per-query rank and the coverage prefix walks to the same
    /// 0.99 target. The fixture puts the top-1 in the rank-1 cluster and
    /// the top-10 tail in the rank-2 cluster, so the two supported points
    /// measure different depths.
    #[test]
    fn depth_law_walks_fine_ranks() {
        const DIM: usize = 4;
        let grid = ClusterCentroids::from_fp32(1, DIM as u32, &[1.0, 0.0, 0.0, 0.0], vec![1; 1]);
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        // Top-10 = ids 1..=10, all in cell 0, scores ascending by id.
        let cands: Vec<(f32, u32, i128, f32)> = (1..=10)
            .map(|id| (id as f32 * 0.01, 0u32, id as i128, f32::NEG_INFINITY))
            .collect();
        let mut acc = Vec::new();
        merge_candidates(&mut acc, cands, WIDTH_LAW_MAX_K);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];

        // Unit fine centroids ranked 0, 1, 2 against the e0 query.
        let mut cluster_of_stable = HashMap::new();
        cluster_of_stable.insert(1i128, 1u32);
        for id in 2..=9i128 {
            cluster_of_stable.insert(id, 0u32);
        }
        cluster_of_stable.insert(10i128, 2u32);
        let view = CellFineCalibrationView {
            cell_id: Some(0),
            dim: DIM,
            n_fine: 3,
            fine_centroids_bytes: fp32_le_bytes(&[
                1.0, 0.0, 0.0, 0.0, //
                0.6, 0.8, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ]),
            cluster_of_stable,
        };
        cal.observe_shard_views(&[view]);

        let laws = cal.finish(&grid).expect("laws from planted candidates");
        assert_eq!(laws.width_for_k[..2], [1, 1], "one cell holds everything");
        assert_eq!(
            laws.fine_for_k[0], 2,
            "top-1 sits in the rank-1 fine cluster"
        );
        assert_eq!(
            laws.fine_for_k[1], 3,
            "top-10 coverage needs the rank-2 cluster"
        );
        assert_eq!(&laws.fine_for_k[2..], &[0, 0], "unsupported points stay 0");
    }

    /// A stamped width beyond the calibration pool clears the rerank
    /// point (previous values were certified for narrower geometry and
    /// under-provision the wider one); widths within the pool keep it.
    #[test]
    fn rerank_points_clear_when_stamped_width_outgrows_pool() {
        let width = [
            1,
            RERANK_LAW_POOL_CELLS as u32,
            (RERANK_LAW_POOL_CELLS + 1) as u32,
            500,
        ];
        let mut rerank = [10, 20, 30, 40];
        clear_rerank_beyond_pool(
            &width,
            &mut rerank,
            &[RERANK_LAW_POOL_CELLS as u32; WIDTH_LAW_KS.len()],
        );
        assert_eq!(
            rerank,
            [10, 20, 0, 0],
            "points at widths beyond the pool must fall back to the constant"
        );
    }

    /// The rerank law reads each exact-top-k candidate's survivor budget
    /// (distractor count) from the planted estimate histogram: the k=1
    /// point takes the best candidate's rank, k=10 the worst's, and points
    /// no query can support stay 0.
    #[test]
    fn rerank_law_reads_survivor_budget_from_histograms() {
        const DIM: usize = 4;
        let grid = ClusterCentroids::from_fp32(1, DIM as u32, &[1.0, 0.0, 0.0, 0.0], vec![1; 1]);
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        // Estimates: id 1 at 0.9 (3 pool rows at-or-better), ids 2..=10 at
        // 0.5 (53 rows at-or-better). q_l1 = 1.0 makes bin() exact.
        let rl = RerankLawObservation {
            quant: BitQuantizer::new(DIM),
            q_rot: vec![0.0; DIM],
            q_total: vec![0.0],
            q_l1: vec![1.0],
            pools: vec![vec![0]],
            hist: Mutex::new(vec![Vec::new()]),
        };
        let mut hist = vec![0u64; RERANK_LAW_EST_BINS];
        hist[rl.bin(0, 0.9)] = 3;
        hist[rl.bin(0, 0.5)] = 50;
        *rl.hist.lock().unwrap_or_else(PoisonError::into_inner) = vec![hist];
        cal.rerank = Some(rl);
        let cands: Vec<(f32, u32, i128, f32)> = (1..=10)
            .map(|id| {
                let est = if id == 1 { 0.9 } else { 0.5 };
                (id as f32 * 0.01, 0u32, id as i128, est)
            })
            .collect();
        let mut acc = Vec::new();
        merge_candidates(&mut acc, cands, WIDTH_LAW_MAX_K);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];

        let laws = cal.finish(&grid).expect("laws from planted candidates");
        assert_eq!(
            laws.rerank_for_k[0], 3,
            "k=1 budget = the best candidate's distractor count"
        );
        assert_eq!(
            laws.rerank_for_k[1], 53,
            "k=10 budget = the worst top-10 candidate's distractor count"
        );
        assert_eq!(
            &laws.rerank_for_k[2..],
            &[0, 0],
            "unsupported points stay uncalibrated"
        );
    }

    /// A candidate whose fine rank was never observed stalls the coverage
    /// walk below target: the point stays 0 (uncalibrated) instead of
    /// stamping a floor shallower than the evidence supports. The stamp
    /// sites treat a 0 point as "keep the previous value".
    #[test]
    fn depth_law_missing_rank_is_conservative() {
        const DIM: usize = 4;
        let grid = ClusterCentroids::from_fp32(1, DIM as u32, &[1.0, 0.0, 0.0, 0.0], vec![1; 1]);
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries: query,
            ids: vec![999],
        });
        let cands: Vec<(f32, u32, i128, f32)> = (1..=10)
            .map(|id| (id as f32 * 0.01, 0u32, id as i128, f32::NEG_INFINITY))
            .collect();
        let mut acc = Vec::new();
        merge_candidates(&mut acc, cands, WIDTH_LAW_MAX_K);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc];

        // id 10 is missing from the view: its rank is unobservable.
        let mut cluster_of_stable = HashMap::new();
        for id in 1..=9i128 {
            cluster_of_stable.insert(id, 0u32);
        }
        let view = CellFineCalibrationView {
            cell_id: Some(0),
            dim: DIM,
            n_fine: 2,
            fine_centroids_bytes: fp32_le_bytes(&[
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0,
            ]),
            cluster_of_stable,
        };
        cal.observe_shard_views(&[view]);

        let laws = cal.finish(&grid).expect("laws from planted candidates");
        assert_eq!(laws.fine_for_k[0], 1, "top-1 was observed at rank 0");
        assert_eq!(
            laws.fine_for_k[1], 0,
            "k=10 misses id 10's rank: 9/10 < 0.99 coverage, point stays 0"
        );
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
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
        let mut queries = vec![0.0f32; 2 * DIM];
        queries[0] = 1.0;
        queries[DIM] = 1.0;
        cal.frozen = Some(WidthLawQueries {
            queries,
            ids: vec![998, 999],
        });
        let k_max = WIDTH_LAW_MAX_K;
        // A: distinct ids 1..=10 spread 2/2/3/3 over cells 0..=3, so its
        // 0.99 top-10 coverage needs the 4th-ranked cell.
        let a: Vec<(f32, u32, i128, f32)> = (1..=10)
            .map(|id| {
                let cell = match id {
                    1 | 2 => 0u32,
                    3 | 4 => 1,
                    5..=7 => 2,
                    _ => 3,
                };
                (id as f32 * 0.01, cell, id as i128, f32::NEG_INFINITY)
            })
            .collect();
        // B: distinct ids 100..=199, every one in the top-ranked cell.
        let b: Vec<(f32, u32, i128, f32)> = (100..200)
            .map(|id| (id as f32 * 0.001, 0u32, id as i128, f32::NEG_INFINITY))
            .collect();
        let (mut acc_a, mut acc_b) = (Vec::new(), Vec::new());
        merge_candidates(&mut acc_a, a, k_max);
        merge_candidates(&mut acc_b, b, k_max);
        *cal.tops.lock().unwrap_or_else(PoisonError::into_inner) = vec![acc_a, acc_b];

        let law = cal
            .finish(&grid)
            .expect("law from planted candidates")
            .width_for_k;
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
        let mut cal = WidthLawCalibration::new(DIM, Metric::Cosine, shipped_target_recall());
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
            vec![(0.01, 7, 1, f32::NEG_INFINITY)],
            WIDTH_LAW_MAX_K,
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

    /// The batch fold must be indistinguishable from folding
    /// [`insert_split_centroids`] sequentially in the same order — same
    /// grown grid, same minted child ids. The batched split pass relies on
    /// this equivalence: ids are positional ordinals, so any drift here
    /// silently re-routes readers.
    #[test]
    fn insert_split_centroids_batch_matches_sequential_fold() {
        let dim = 8usize;
        let base = synth_centroids(6, dim as u32);
        // Three splits with distinct k's; parents in ascending order (the
        // executor's fixed fold order). Distinct value patterns per split so
        // a misplaced copy shows up as a centroid mismatch, not a no-op.
        let sub = |tag: f32, k: usize| -> Vec<f32> {
            (0..k * dim).map(|i| tag + i as f32 * 0.01).collect()
        };
        let (s0, s2, s5) = (sub(1.0, 2), sub(2.0, 4), sub(3.0, 3));
        let splits: Vec<(u32, &[f32], usize)> = vec![(0, &s0, 2), (2, &s2, 4), (5, &s5, 3)];

        let (batched, batched_ids) = insert_split_centroids_batch(&base, &splits);

        let mut folded = base.clone();
        let mut folded_ids = Vec::new();
        for &(cell, sub_centroids, k) in &splits {
            let (next, ids) = insert_split_centroids(&folded, cell, sub_centroids, k);
            folded = next;
            folded_ids.push(ids);
        }

        assert_eq!(batched.n_cent, folded.n_cent);
        assert_eq!(batched.dim, folded.dim);
        assert_eq!(batched.centroids, folded.centroids);
        assert_eq!(batched.counts, folded.counts);
        assert_eq!(batched_ids, folded_ids);
        // Prefix-sum id minting: appended ids are contiguous off the base
        // grid's end, in fold order.
        assert_eq!(batched_ids[0], vec![0, 6]);
        assert_eq!(batched_ids[1], vec![2, 7, 8, 9]);
        assert_eq!(batched_ids[2], vec![5, 10, 11]);
        // Counts cover every cell (wire-encoding invariant) with new
        // children zeroed until the caller applies real counts.
        assert_eq!(batched.counts.len(), 12);
        assert!(batched.counts[6..].iter().all(|&c| c == 0));

        // Round-trips through the manifest wire format cleanly.
        let bytes = crate::supertable::manifest::encoding::encode_cluster_centroids(&batched);
        let decoded = crate::supertable::manifest::encoding::decode_cluster_centroids(&bytes)
            .expect("batch-split grid must reopen from wire bytes");
        assert_eq!(decoded.n_cent, 12);
        assert_eq!(decoded.centroids.len(), 12 * dim);
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
