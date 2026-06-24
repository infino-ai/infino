// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! MVCC SPFresh maintenance for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable. The hidden index is a
//! derived, cell-ordered acceleration layer maintained with SPFresh/LIRE-style
//! logical updates expressed as append/MVCC physical swaps:
//!
//!   1. Assign incoming vectors to nearest manifest centroids with zero GETs.
//!   2. For each touched cell only: append one delta superfile (no GETs).
//!   3. Compaction merges multiple small IVF superfiles per cell toward one packed
//!      base via the standard `merge_superfiles` path.
//!   4. Locally refresh touched cell centroids and member radii.
//!   5. Split overflow cells (Sq8+ε k-means, N→N+1 centroids).
//!   6. Reassign vectors in the split neighborhood (P−1, P, P₂, P+1).
//!   7. Redrive reassigned rows through the incoming staging region; route
//!      them into per-cell IVF superfiles (same path as commit ingest).
//!
//! Split/reassign stays on stored Sq8+ε bytes. Row assignment and k-means both
//! score via [`distance_encoded_to_centroid`]; rows are re-spliced with
//! [`encode_encoded_rows`], never decoded to fp32 corpora.

use std::{cmp::Ordering, collections::HashMap, env, sync::OnceLock};

use crate::{
    superfile::vector::{
        cell_posting::{EncodedCellRow, manifest_centroid_components_from_row, medoid_index_by},
        distance::Metric,
    },
    supertable::manifest::ClusterCentroids,
};

/// Doc count above which a merged cell superfile is split (SPFresh step 7).
const CELL_SPLIT_DOC_CAP_DEFAULT: u64 = 50_000;

/// Lloyd iterations for 2-way Sq8+ε k-means at split time.
const CELL_SPLIT_KMEANS_ITERS: usize = 5;

/// Overflow threshold for cell split. Override with `INFINO_CELL_SPLIT_DOC_CAP` in tests.
pub(crate) fn cell_split_doc_cap() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        env::var("INFINO_CELL_SPLIT_DOC_CAP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(CELL_SPLIT_DOC_CAP_DEFAULT)
    })
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

/// Apply count and radius updates from maintenance (incoming routing / compaction).
pub(crate) fn apply_cell_updates(
    base: &ClusterCentroids,
    count_updates: &HashMap<u32, u32>,
    radii_updates: &HashMap<u32, f32>,
) -> ClusterCentroids {
    let mut updated = apply_cell_count_updates(base, count_updates);
    if radii_updates.is_empty() {
        return updated;
    }
    if updated.radii.len() != updated.n_cent as usize {
        updated.radii = vec![0.0; updated.n_cent as usize];
    }
    for (&cell, &radius) in radii_updates {
        if let Some(slot) = updated.radii.get_mut(cell as usize)
            && radius > *slot
        {
            *slot = radius;
        }
    }
    updated
}

fn score_row_against_cell(
    clusters: &ClusterCentroids,
    metric: Metric,
    cell: usize,
    row: &EncodedCellRow,
) -> f32 {
    let dim = clusters.dim as usize;
    let fp32 = manifest_centroid_components_from_row(row, dim);
    clusters.score_one(metric, cell, &fp32)
}

/// Build a one-cluster [`ClusterCentroids`] prototype from a stored Sq8+ε row so
/// row↔row distances reuse the same asymmetric kernel as query scoring.
fn centroid_prototype_from_row(
    template: &ClusterCentroids,
    metric: Metric,
    row: &EncodedCellRow,
) -> ClusterCentroids {
    let dim = template.dim as usize;
    let fp32 = manifest_centroid_components_from_row(row, dim);
    ClusterCentroids::from_fp32(metric, 1, template.dim, &fp32, vec![1])
}

fn distance_encoded_rows(
    metric: Metric,
    template: &ClusterCentroids,
    a: &EncodedCellRow,
    b: &EncodedCellRow,
) -> f32 {
    let proto = centroid_prototype_from_row(template, metric, b);
    score_row_against_cell(&proto, metric, 0, a)
}

/// Medoid index under the asymmetric Sq8+ε row↔row distance (discrete k-means
/// centroid update). Shares the all-pairs min-sum loop with the symmetric
/// variant via [`medoid_index_by`]; only the distance kernel differs.
fn medoid_index(template: &ClusterCentroids, metric: Metric, shard: &[EncodedCellRow]) -> usize {
    medoid_index_by(shard, |a, b| distance_encoded_rows(metric, template, a, b))
}

/// 2-way Lloyd k-means on Sq8+ε overflow rows. Returns manifest centroid
/// components (dim each) for the two sub-cells.
pub(crate) fn plan_sq8_split(
    rows: &[EncodedCellRow],
    clusters: &ClusterCentroids,
    split_cell: u32,
    metric: Metric,
) -> (Vec<f32>, Vec<f32>) {
    let dim = clusters.dim as usize;
    let p = split_cell as usize;

    let seed0 = rows
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, p, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, p, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let seed1 = rows
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            score_row_against_cell(clusters, metric, p, a)
                .partial_cmp(&score_row_against_cell(clusters, metric, p, b))
                .unwrap_or(Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut cent0 = centroid_prototype_from_row(clusters, metric, &rows[seed0]);
    let mut cent1 = centroid_prototype_from_row(clusters, metric, &rows[seed1]);

    let mut assign = vec![0u8; rows.len()];
    for _ in 0..CELL_SPLIT_KMEANS_ITERS {
        for (i, row) in rows.iter().enumerate() {
            let d0 = score_row_against_cell(&cent0, metric, 0, row);
            let d1 = score_row_against_cell(&cent1, metric, 0, row);
            assign[i] = u8::from(d1 < d0);
        }
        let mut shard0 = Vec::new();
        let mut shard1 = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if assign[i] == 0 {
                shard0.push(row.clone());
            } else {
                shard1.push(row.clone());
            }
        }
        if shard0.is_empty() || shard1.is_empty() {
            break;
        }
        let m0 = medoid_index(clusters, metric, &shard0);
        let m1 = medoid_index(clusters, metric, &shard1);
        cent0 = centroid_prototype_from_row(clusters, metric, &shard0[m0]);
        cent1 = centroid_prototype_from_row(clusters, metric, &shard1[m1]);
    }

    // Re-assign against the converged centroids: the loop's last `assign` pass
    // ran against the *previous* iteration's centroids (cent0/cent1 are updated
    // after it), so the final shards must reflect one more assignment pass.
    for (i, row) in rows.iter().enumerate() {
        let d0 = score_row_against_cell(&cent0, metric, 0, row);
        let d1 = score_row_against_cell(&cent1, metric, 0, row);
        assign[i] = u8::from(d1 < d0);
    }
    let mut shard0 = Vec::new();
    let mut shard1 = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        if assign[i] == 0 {
            shard0.push(row.clone());
        } else {
            shard1.push(row.clone());
        }
    }
    if shard1.is_empty() {
        shard1.push(rows[seed1].clone());
        shard0.retain(|r| r.stable_id != rows[seed1].stable_id);
    }
    if shard0.is_empty() {
        shard0.push(rows[seed0].clone());
        shard1.retain(|r| r.stable_id != rows[seed0].stable_id);
    }

    let m0 = medoid_index(clusters, metric, &shard0);
    let m1 = medoid_index(clusters, metric, &shard1);
    (
        manifest_centroid_components_from_row(&shard0[m0], dim),
        manifest_centroid_components_from_row(&shard1[m1], dim),
    )
}

/// Assign an encoded row to its nearest manifest cell.
pub(crate) fn nearest_cell_encoded(
    clusters: &ClusterCentroids,
    metric: Metric,
    row: &EncodedCellRow,
) -> u32 {
    let mut best = 0u32;
    let mut best_score = f32::INFINITY;
    for c in 0..clusters.n_cent as usize {
        let score = score_row_against_cell(clusters, metric, c, row);
        if score < best_score {
            best_score = score;
            best = c as u32;
        }
    }
    best
}

/// Max member distance from `cell_id`'s centroid over encoded rows.
pub(crate) fn encoded_shard_radius(
    clusters: &ClusterCentroids,
    metric: Metric,
    cell_id: u32,
    rows: &[EncodedCellRow],
) -> f32 {
    let mut max_r = 0.0f32;
    for row in rows {
        let dist = score_row_against_cell(clusters, metric, cell_id as usize, row);
        if dist > max_r {
            max_r = dist;
        }
    }
    max_r
}

/// Assign an encoded row to the nearest cell among `candidates`.
pub(crate) fn nearest_among_cells_encoded(
    clusters: &ClusterCentroids,
    metric: Metric,
    candidates: &[u32],
    row: &EncodedCellRow,
) -> u32 {
    let mut best = candidates[0];
    let mut best_score = f32::INFINITY;
    for &c in candidates {
        let score = score_row_against_cell(clusters, metric, c as usize, row);
        if score < best_score {
            best_score = score;
            best = c;
        }
    }
    best
}

/// Replace cell `cell_id`'s centroid and append a second sub-cell at `n_cent`.
pub(crate) fn insert_split_centroid(
    base: &ClusterCentroids,
    metric: Metric,
    cell_id: u32,
    sub_centroids: &[f32],
) -> (ClusterCentroids, u32) {
    let dim = base.dim as usize;
    let p = cell_id as usize;
    let old_n = base.n_cent as usize;
    let new_cell_id = base.n_cent;
    let new_n = old_n + 1;

    let mut fp32 = vec![0f32; new_n * dim];
    for c in 0..old_n {
        base.dequantize_into(c, &mut fp32[c * dim..(c + 1) * dim]);
    }
    fp32[p * dim..(p + 1) * dim].copy_from_slice(&sub_centroids[..dim]);
    fp32[old_n * dim..new_n * dim].copy_from_slice(&sub_centroids[dim..2 * dim]);

    let counts = base.counts.clone();
    let mut radii = if base.radii.len() == old_n {
        base.radii.clone()
    } else {
        vec![0.0; old_n]
    };
    radii.resize(new_n, 0.0);

    let updated = ClusterCentroids::from_fp32(metric, new_n as u32, base.dim, &fp32, counts)
        .with_radii(radii);
    (updated, new_cell_id)
}

/// Neighbor cells touched by a split of `split_cell`: P−1, P, the new sub-cell, P+1.
pub(crate) fn reassign_neighborhood(
    split_cell: u32,
    old_n_cent: u32,
    new_cell_id: u32,
) -> Vec<u32> {
    let mut ids = Vec::new();
    if split_cell > 0 {
        ids.push(split_cell - 1);
    }
    ids.push(split_cell);
    ids.push(new_cell_id);
    if split_cell + 1 < old_n_cent {
        ids.push(split_cell + 1);
    }
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Clear per-cell counts/radii when superfiles for those cells are removed and
/// rows are redriven through the incoming staging region.
pub(crate) fn zero_cell_counts(clusters: &mut ClusterCentroids, cells: &[u32]) {
    for &cell in cells {
        let c = cell as usize;
        if c < clusters.counts.len() {
            clusters.counts[c] = 0;
        }
        if c < clusters.radii.len() {
            clusters.radii[c] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::cell_posting::{encode_blob, load_encoded_rows_from_blob};

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
        ClusterCentroids::from_fp32(Metric::L2Sq, n_cent, dim, &fp32, counts)
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
        let blob = encode_blob(Metric::L2Sq, dim, &ids, &vecs).expect("encode");
        let stable_ids: Vec<i128> = (0..n).map(|i| i as i128).collect();
        load_encoded_rows_from_blob(&blob, &stable_ids, None).expect("load")
    }

    #[test]
    fn insert_split_centroid_extends_n_cent() {
        let base = synth_centroids(4, 8);
        let sub = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
        ];
        let (updated, new_id) = insert_split_centroid(&base, Metric::L2Sq, 2, &sub);
        assert_eq!(new_id, 4);
        assert_eq!(updated.n_cent, 5);
    }

    #[test]
    fn reassign_neighborhood_includes_neighbors_and_new_cell() {
        let ids = reassign_neighborhood(3, 8, 8);
        assert_eq!(ids, vec![2, 3, 4, 8]);
    }

    #[test]
    fn plan_sq8_split_separates_two_blobs() {
        let dim = 4usize;
        let mut rows = synth_rows(dim, 10, 0.0);
        rows.extend(synth_rows(dim, 10, 10.0));
        let clusters = synth_centroids(4, dim as u32);
        let (c0, c1) = plan_sq8_split(&rows, &clusters, 1, Metric::L2Sq);
        assert_eq!(c0.len(), dim);
        assert_eq!(c1.len(), dim);
        let dist: f32 = (0..dim).map(|d| (c0[d] - c1[d]).abs()).sum();
        assert!(dist > 1.0, "split centroids should separate, got {dist}");
    }
}
