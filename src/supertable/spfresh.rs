// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! MVCC SPFresh maintenance for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable. The hidden index is a
//! derived, cell-ordered acceleration layer maintained with SPFresh/LIRE-style
//! *logical* updates expressed as append/MVCC physical swaps:
//!
//!   1. Assign incoming vectors to nearest manifest centroids (zero GETs).
//!   2. For each touched cell only: read prior generation → merge batch →
//!      write replacement cell superfile (reuse URI or new URI on split).
//!   3. Mark superseded cell entries in `entries_to_remove`; publish new
//!      manifest snapshot via `persist_commit` (atomic generation swap).
//!   4. Locally refresh touched cell centroids (+ optional k-means(2) split).
//!
//! This is **not** in-place mutation of user Parquet or literal SPFresh disk
//! rebalance. It is generation-based cell superfile replacement.

use std::collections::HashMap;

use crate::superfile::vector::kmeans::kmeans_with_assignments;
use crate::supertable::manifest::ClusterCentroids;

pub(crate) use crate::supertable::handle::{
    GLOBAL_VECTOR_KMEANS_ITERS, GLOBAL_VECTOR_KMEANS_SEED,
};

/// Split when a touched cell exceeds this many rows after merge.
pub(crate) const CELL_SPLIT_THRESHOLD: usize = 4096;

/// Mean fp32 centroid for `(stable_id, vector)` rows in one cell.
pub fn centroid_mean_from_rows(rows: &[(i128, Vec<f32>)], dim: usize) -> Vec<f32> {
    let mut mean = vec![0f32; dim];
    if rows.is_empty() {
        return mean;
    }
    for (_, v) in rows {
        for d in 0..dim {
            mean[d] += v[d];
        }
    }
    let n = rows.len() as f32;
    for x in &mut mean {
        *x /= n;
    }
    mean
}

/// SPFresh local centroid refresh: update touched cells; append split cells.
pub fn apply_cell_centroid_updates(
    base: &ClusterCentroids,
    cell_updates: &HashMap<u32, (Vec<f32>, u32)>,
    new_cells: &[(Vec<f32>, u32)],
) -> ClusterCentroids {
    let dim = base.dim as usize;
    let old_n = base.n_cent as usize;
    let new_n = old_n + new_cells.len();
    let mut fp32 = vec![0f32; new_n * dim];
    let mut counts = vec![0u32; new_n];
    for c in 0..old_n {
        if let Some((mean, count)) = cell_updates.get(&(c as u32)) {
            fp32[c * dim..(c + 1) * dim].copy_from_slice(mean);
            counts[c] = *count;
        } else {
            base.dequantize_into(c, &mut fp32[c * dim..(c + 1) * dim]);
            counts[c] = base.counts[c];
        }
    }
    for (i, (mean, count)) in new_cells.iter().enumerate() {
        let c = old_n + i;
        fp32[c * dim..(c + 1) * dim].copy_from_slice(mean);
        counts[c] = *count;
    }
    ClusterCentroids::from_fp32(new_n as u32, base.dim, &fp32, counts)
}

/// SPFresh/LIRE overflow split: local k-means(2) on one cell's rows.
pub fn split_cell_rows(
    rows: &[(i128, Vec<f32>)],
    dim: usize,
    seed: u64,
) -> Option<(Vec<(i128, Vec<f32>)>, Vec<(i128, Vec<f32>)>)> {
    if rows.len() < 2 {
        return None;
    }
    let mut vectors = Vec::with_capacity(rows.len() * dim);
    for (_, v) in rows {
        vectors.extend_from_slice(v);
    }
    let (_, assignments) =
        kmeans_with_assignments(&vectors, dim, 2, GLOBAL_VECTOR_KMEANS_ITERS, seed);
    let mut left = Vec::new();
    let mut right = Vec::new();
    for (row, &a) in rows.iter().zip(assignments.iter()) {
        if a == 0 {
            left.push(row.clone());
        } else {
            right.push(row.clone());
        }
    }
    if left.is_empty() || right.is_empty() {
        return None;
    }
    Some((left, right))
}
