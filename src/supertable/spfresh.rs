// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Maintenance bookkeeping for the hidden global vector cell index.
//!
//! The user table stays time-ordered and immutable; the hidden index is a
//! derived, cell-ordered acceleration layer. Each commit appends immutable
//! ~8 MB IVF cells (no eager drain). When a region's cells start to overlap,
//! overlap-triggered consolidation re-clusters that region into tight, disjoint
//! cells — that work lives in [`crate::supertable::compaction`]
//! (`overlap_consolidation_jobs`) + [`crate::supertable::writer`]
//! (`recluster_cells`). This module owns the overlap detection
//! ([`hot_overlap_groups`]) and the manifest cell-summary bookkeeping
//! ([`apply_cell_updates`]).

use std::collections::HashMap;

use crate::{
    superfile::vector::distance::{Metric, metric_distance_by},
    supertable::manifest::ClusterCentroids,
};

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

/// Default mean-overlap-degree at or above which a region is consolidated. A
/// query landing in such a region must, on average, scan this many overlapping
/// cells, so re-clustering them into tight, non-overlapping cells pays off.
pub(crate) const CELL_OVERLAP_TAU_DEFAULT: f32 = 3.0;

/// Distance between two cell centroids, in the same metric units the stored
/// cell radii were measured in — so it is directly comparable to `r_i + r_j`,
/// exactly as [`ClusterCentroids::select_cells_adaptive`] compares a centroid
/// score `d` to a radius `r`. (These are fp32 centroids; the stored radii were
/// measured in the Sq8 domain, but the two differ only by quantization noise,
/// which is immaterial for a consolidation *trigger*.)
fn centroid_pair_distance(
    centroids: &[f32],
    dim: usize,
    i: usize,
    j: usize,
    metric: Metric,
) -> f32 {
    metric_distance_by(
        metric,
        dim,
        |d| centroids[i * dim + d],
        |d| centroids[j * dim + d],
    )
}

/// Find the **hot regions** to consolidate among hidden cells, by bounding-sphere
/// overlap. Two cells overlap when the distance between their centroids is below
/// the sum of their radii — their member spheres intersect, so a query in that
/// volume is forced to scan both. Each returned group is a connected component of
/// the overlap graph whose **mean overlap degree ≥ `tau`** (a typical cell in the
/// group overlaps `tau`+ others): the cells consolidation should re-cluster into
/// tight, non-overlapping cells. Pure over centroid + radius metadata, so it
/// costs no object-store reads.
///
/// `centroids` is `n × dim` row-major fp32; `radii` is length `n`. Groups are
/// returned with their member indices sorted ascending.
pub(crate) fn hot_overlap_groups(
    centroids: &[f32],
    radii: &[f32],
    dim: usize,
    metric: Metric,
    tau: f32,
) -> Vec<Vec<usize>> {
    let n = radii.len();
    if n < 2 || dim == 0 || centroids.len() != n * dim {
        return Vec::new();
    }
    // Overlap adjacency: i ~ j iff their bounding spheres intersect.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if centroid_pair_distance(centroids, dim, i, j, metric) < radii[i] + radii[j] {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    // Connected components (iterative DFS); keep those dense enough to be hot.
    let mut seen = vec![false; n];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if seen[start] || adj[start].is_empty() {
            continue;
        }
        let mut stack = vec![start];
        seen[start] = true;
        let mut comp: Vec<usize> = Vec::new();
        while let Some(v) = stack.pop() {
            comp.push(v);
            for &w in &adj[v] {
                if !seen[w] {
                    seen[w] = true;
                    stack.push(w);
                }
            }
        }
        let degree_sum: usize = comp.iter().map(|&v| adj[v].len()).sum();
        let mean_degree = degree_sum as f32 / comp.len() as f32;
        if mean_degree >= tau {
            comp.sort_unstable();
            groups.push(comp);
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_overlap_groups_finds_dense_overlapping_regions() {
        // dim=1, so L2Sq distance is the squared gap and radii are in the same
        // (squared) units — overlap iff gap² < r_i + r_j.

        // Far apart with tiny radii: 10² = 100 ≥ 1 + 1, no overlap.
        let centroids = vec![0.0f32, 10.0];
        let radii = vec![1.0f32, 1.0];
        assert!(hot_overlap_groups(&centroids, &radii, 1, Metric::L2Sq, 1.0).is_empty());

        // One dense cluster: gaps² are 1, 1, 4, all < 3 + 3, so 0~1~2 all overlap.
        let centroids = vec![0.0f32, 1.0, 2.0];
        let radii = vec![3.0f32, 3.0, 3.0];
        assert_eq!(
            hot_overlap_groups(&centroids, &radii, 1, Metric::L2Sq, 2.0),
            vec![vec![0, 1, 2]]
        );
        // Mean overlap degree in that group is 2, so a stricter tau drops it.
        assert!(hot_overlap_groups(&centroids, &radii, 1, Metric::L2Sq, 3.0).is_empty());

        // Two separated overlapping clusters → two groups.
        let centroids = vec![0.0f32, 1.0, 100.0, 101.0];
        let radii = vec![2.0f32, 2.0, 2.0, 2.0];
        assert_eq!(
            hot_overlap_groups(&centroids, &radii, 1, Metric::L2Sq, 1.0),
            vec![vec![0, 1], vec![2, 3]]
        );
    }
}
