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
//!   5. Split overflow cells in place: partition an over-cap cell K-way
//!      (Sq8+ε k-means, K = ⌈rows/cap⌉ so each child lands under the cap),
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

use std::{cmp::Ordering, collections::HashMap};

use crate::{
    config,
    superfile::vector::{
        cell_posting::{
            EncodedCellRow, dequantize_sq8_residual_into, manifest_centroid_components_from_row,
            medoid_index_by,
        },
        distance::{Metric, distance, nearest_k_centroids_transposed, relative_score_window},
        kmeans::kmeans,
    },
    supertable::manifest::{
        ClusterCentroids, RABITQ_ADMIT_CELL_SHORTLIST_FRACTION, RABITQ_ADMIT_CELL_SHORTLIST_MIN,
        RabitqAdmitContext,
    },
};

/// Overflow threshold for cell split (OPANN step 7). Sourced from
/// `vector.cell_split_doc_cap`.
pub(crate) fn cell_split_doc_cap() -> u64 {
    config::global().vector.cell_split_doc_cap
}

/// True when a merged cell superfile should be split into two sub-cells.
pub(crate) fn split_overflow_needed(n_docs: u64) -> bool {
    n_docs > cell_split_doc_cap()
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

fn score_row_against_cell(
    clusters: &ClusterCentroids,
    metric: Metric,
    cell: usize,
    row: &EncodedCellRow,
) -> f32 {
    let dim = clusters.dim as usize;
    let mut row_fp = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut row_fp,
    );
    distance(metric, &row_fp, clusters.centroid(cell))
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
    let dim = clusters.dim as usize;
    let mut row_fp = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut row_fp,
    );
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

/// One-cluster [`ClusterCentroids`] prototype from a Sq8+ε row (split k-means seeds).
fn centroid_prototype_from_row(
    template: &ClusterCentroids,
    row: &EncodedCellRow,
) -> ClusterCentroids {
    let dim = template.dim as usize;
    let fp32 = manifest_centroid_components_from_row(row, dim);
    ClusterCentroids::from_fp32(1, template.dim, &fp32, vec![1])
}

fn fp32_distance_between_rows(metric: Metric, a: &EncodedCellRow, b: &EncodedCellRow) -> f32 {
    debug_assert_eq!(a.rerank_codec, b.rerank_codec);
    let dim = a.scale.len();
    let mut af = vec![0f32; dim];
    let mut bf = vec![0f32; dim];
    let divisor = a
        .rerank_codec
        .residual_divisor()
        .expect("encoded row uses residual-family codec");
    dequantize_sq8_residual_into(
        &a.scale,
        &a.offset,
        &a.codes,
        &a.residuals,
        divisor,
        &mut af,
    );
    dequantize_sq8_residual_into(
        &b.scale,
        &b.offset,
        &b.codes,
        &b.residuals,
        divisor,
        &mut bf,
    );
    distance(metric, &af, &bf)
}

/// Medoid index under fp32 dequant + [`distance`] row↔row (discrete k-means
/// centroid update).
fn medoid_index(metric: Metric, shard: &[&EncodedCellRow]) -> usize {
    medoid_index_by(shard, |a, b| fp32_distance_between_rows(metric, a, b))
}

/// 2-way Lloyd k-means on Sq8+ε overflow rows. Returns manifest centroid
/// components (dim each) for the two sub-cells.
/// Plan a binary split of `split_cell`: returns the two sub-cell centroids and,
/// aligned to `rows`, a `0/1` assignment of each row to sub-cell 0 / sub-cell 1
/// (reconciled with the empty-shard fixups, so it exactly matches the two
/// shards the centroids were derived from). The caller routes the cell's
/// materialized rows into the two sub-cells by this assignment.
/// The two diameter endpoints of `split_cell`'s rows: `seed1` is the row
/// farthest from the cell's existing centroid (an extreme edge point); `seed0`
/// is the row farthest from `seed1` (the opposite edge). The `seed0 → seed1`
/// line is the axis the split bisects along. Picking closest-to-centroid vs
/// farthest instead would seed a radius (center + edge), peeling a thin shell
/// off the dense core.
fn pick_split_seeds(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: usize,
    metric: Metric,
) -> (usize, usize) {
    let seed1 = rows
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, split_cell, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, split_cell, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let cent1 = centroid_prototype_from_row(clusters, rows[seed1]);
    let seed0 = rows
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(&cent1, metric, 0, a)
                .partial_cmp(&score_row_against_cell(&cent1, metric, 0, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    (seed0, seed1)
}

/// Dequantize one Sq8+ε residual row to fp32.
fn dequantize_row(row: &EncodedCellRow, dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; dim];
    dequantize_sq8_residual_into(
        &row.scale,
        &row.offset,
        &row.codes,
        &row.residuals,
        row.rerank_codec
            .residual_divisor()
            .expect("encoded row uses residual-family codec"),
        &mut out,
    );
    out
}

/// K-way split of `split_cell` into `k` sub-cells. Returns `k * dim` centroid
/// components and a `0..k` per-row assignment aligned to `rows`. `use_kmeans`
/// assigns each row to its NEAREST sub-centroid (assignment == query routing,
/// so a split doc lands in the cell its query probes); otherwise a diameter
/// median cut into `k` equal-population bins (balanced but assign != route).
/// `k = 2` reproduces the binary split.
pub(crate) fn plan_sq8_split_kway(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
    k: usize,
    use_kmeans: bool,
) -> (Vec<f32>, Vec<u32>) {
    let dim = clusters.dim as usize;
    let p = split_cell as usize;
    let k = k.max(2).min(rows.len().max(2));
    let mut assign = vec![0u32; rows.len()];
    if rows.len() < 2 {
        // Caller guards on MIN_ROWS_TO_SPLIT_CELL; stay defensive so a degenerate
        // input can't panic in medoid_index on an empty shard.
        let c = manifest_centroid_components_from_row(rows[0], dim);
        let mut cents = Vec::with_capacity(k * dim);
        for _ in 0..k {
            cents.extend_from_slice(&c);
        }
        return (cents, assign);
    }

    // k-means: assign each row to its NEAREST sub-centroid — identical to
    // query-time routing, so a split doc lands in the very cell its query
    // probes. Trains on a sample; assigns the full cell under `metric`.
    if use_kmeans {
        let sample_n = rows
            .len()
            .min((k * SPLIT_KMEANS_SAMPLE_PER_CLUSTER).max(SPLIT_KMEANS_SAMPLE_MIN));
        let mut sample = Vec::with_capacity(sample_n * dim);
        for s in 0..sample_n {
            let idx = s * rows.len() / sample_n;
            sample.extend_from_slice(&dequantize_row(rows[idx], dim));
        }
        let cents = kmeans(
            &sample,
            dim,
            k,
            SPLIT_KMEANS_ITERS,
            (split_cell as u64) ^ SPLIT_KMEANS_SEED_XOR,
        );
        if cents.len() >= k * dim {
            let mut populated = vec![false; k];
            for (i, row) in rows.iter().copied().enumerate() {
                let rv = dequantize_row(row, dim);
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = distance(metric, &rv, &cents[c * dim..(c + 1) * dim]);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assign[i] = best as u32;
                populated[best] = true;
            }
            // Accept any non-degenerate result (≥ 2 populated sub-cells). We do
            // NOT gate on balance or cap: assignment is pure NEAREST-centroid,
            // so `assign == route` and nprobe=1 recall holds. A child left over
            // cap is fine — the split loop re-selects it and re-splits, and on
            // real data that converges to ≤cap without a size threshold (the
            // loop, not a gate, bounds cell size; validated at 10M). Only a
            // genuine collapse (<2 populated: near-identical vectors that can't
            // be clustered, and route together anyway) falls through to the
            // axis cut below, purely to guarantee forward progress.
            if populated.iter().filter(|&&u| u).count() >= 2 {
                return (cents, assign);
            }
            assign.iter_mut().for_each(|a| *a = 0);
        }
    }

    // Diameter median cut, generalized to `k` equal-population bins along the
    // `seed0 → seed1` axis. Balanced (±1) regardless of density, so children
    // converge in one pass. Assignment is by axis position, not nearest
    // centroid, so it does NOT match query routing (the recall cost k-means
    // avoids). `seed1` is the row farthest from the cell centroid; `seed0` the
    // row farthest from `seed1` — the two diameter endpoints.
    let (seed0, seed1) = pick_split_seeds(rows, clusters, p, metric);
    let v0 = dequantize_row(rows[seed0], dim);
    let v1 = dequantize_row(rows[seed1], dim);
    let axis: Vec<f32> = (0..dim).map(|d| v1[d] - v0[d]).collect();

    let mut proj: Vec<(usize, f32)> = rows
        .iter()
        .copied()
        .enumerate()
        .map(|(i, row)| {
            let rv = dequantize_row(row, dim);
            let s: f32 = (0..dim).map(|d| rv[d] * axis[d]).sum();
            (i, s)
        })
        .collect();
    proj.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    let n = rows.len();
    for (rank, (i, _)) in proj.iter().enumerate() {
        assign[*i] = ((rank * k) / n).min(k - 1) as u32;
    }

    // Each bin's centroid is its medoid. Shards borrow the input rows (no
    // payload clone): the split extracts up to a full over-cap cell, so cloning
    // every row's Sq8+ε bytes would double the biggest cell's resident bytes.
    let mut cents = vec![0f32; k * dim];
    for c in 0..k {
        let shard: Vec<&EncodedCellRow> = rows
            .iter()
            .copied()
            .enumerate()
            .filter(|(i, _)| assign[*i] == c as u32)
            .map(|(_, r)| r)
            .collect();
        if shard.is_empty() {
            continue;
        }
        let m = medoid_index(metric, &shard);
        cents[c * dim..(c + 1) * dim]
            .copy_from_slice(&manifest_centroid_components_from_row(shard[m], dim));
    }
    (cents, assign)
}

/// Binary median-cut split (`k = 2`, `use_kmeans = false`). Test-only wrapper
/// over [`plan_sq8_split_kway`] preserving the two-centroid / `u8`-assignment
/// shape the median-cut unit tests were written against. The production split
/// path calls [`plan_sq8_split_kway`] directly with k-means assignment.
#[cfg(test)]
pub(crate) fn plan_sq8_split(
    rows: &[&EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let dim = clusters.dim as usize;
    let (cents, assign) = plan_sq8_split_kway(rows, clusters, split_cell, metric, 2, false);
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

    #[test]
    fn pick_split_seeds_returns_diameter_endpoints() {
        let dim = 4usize;
        // 20 rows evenly spaced along a line (row i ≈ 0.01·i per dim), centroid
        // pinned at the line's middle. The two farthest-apart rows are the
        // endpoints (0 and 19) — NOT the middle row a closest-to-centroid seed
        // would pick — so the returned pair must be {0, 19}.
        let rows = synth_rows(dim, 20, 0.0);
        let mid = vec![0.095f32, 0.096, 0.097, 0.098];
        let clusters = ClusterCentroids::from_fp32(1, dim as u32, &mid, vec![rows.len() as u32]);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        let (seed0, seed1) = pick_split_seeds(&refs, &clusters, 0, Metric::L2Sq);
        let mut ends = [seed0, seed1];
        ends.sort_unstable();
        assert_eq!(
            ends,
            [0, rows.len() - 1],
            "seeds must be the diameter endpoints, got {ends:?}"
        );
    }

    /// Termination guarantee behind the split loop, on a density-skewed cell.
    /// The diameter/median cut (the fallback when k-means collapses to <2
    /// populated) slices into `K = ⌈rows/cap⌉` equal-population bins by axis
    /// rank, so every child lands at ≈rows/K — under the cap in ONE pass —
    /// regardless of density (a nearest-seed cut would peel the far tail into
    /// one lopsided child instead). A balanced K-way cut needs no re-split, so
    /// the split loop cannot thrash on pathological input; `MAX_SPLITS_PER_OPTIMIZE`
    /// is only a safety backstop, never the actual bound.
    #[test]
    fn plan_sq8_split_kway_median_balances_skewed_cell_under_cap_in_one_pass() {
        let dim = 4usize;
        // Dense core (16 rows near origin) + far sparse tail (4 rows): a
        // nearest-seed split would peel the 4 tail rows off; the median cut
        // splits by rank regardless of density.
        let mut rows = synth_rows(dim, 16, 0.0);
        rows.extend(synth_rows(dim, 4, 50.0));
        let n = rows.len(); // 20
        let clusters =
            ClusterCentroids::from_fp32(1, dim as u32, &vec![0.0f32; dim], vec![n as u32]);
        let refs: Vec<&EncodedCellRow> = rows.iter().collect();
        // K sized as ⌈rows/cap⌉ for a notional cap of 5.
        let k = 4usize;
        let (_cents, assign) = plan_sq8_split_kway(&refs, &clusters, 0, Metric::L2Sq, k, false);
        let mut counts = vec![0usize; k];
        for &a in &assign {
            counts[a as usize] += 1;
        }
        let target = n.div_ceil(k); // 5
        assert!(
            counts.iter().all(|&c| c > 0),
            "every child populated — no collapse, got {counts:?}"
        );
        let max_bin = *counts.iter().max().expect("k >= 1, so counts is non-empty");
        assert!(
            max_bin <= target,
            "median K-way puts every child at <= ceil(rows/K)={target} in one pass \
             (balanced despite density skew, no re-split, no thrash), got {counts:?}"
        );
    }
}
