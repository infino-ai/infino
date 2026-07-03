// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN — Object-Partitioned Approximate Nearest Neighbor.
//!
//! The routing layer for the hidden vector index: a hierarchical centroid tree
//! ([`tree::CentroidTree`]) over the fine-centroid set, searched on compute with
//! zero object GETs to select the nearest leaves. A query descends the tree in
//! memory, then the caller fetches the selected leaves' run fragments in one
//! coalesced wave.
//!
//! Ported from the `perf/hybrid-spfresh-sq8` branch. The tree structure, the
//! shared best-first [`descent`], and the split/rebalance build are kept; the
//! branch's paged, content-addressed copy-on-write on-disk form is
//! intentionally omitted here (the whole tree is built in memory from the
//! manifest leaves and held resident on table open), and node centroids are
//! stored 32-bit fp32 instead of the branch's Sq8+residual.
#![allow(dead_code)]

pub(crate) mod descent;
pub(crate) mod insert;
pub(crate) mod resident;
pub(crate) mod tree;

#[cfg(test)]
pub(crate) mod test_util {
    //! Shared unit-test fixtures for the OPANN modules.

    use crate::{
        superfile::vector::distance::Metric,
        supertable::{
            manifest::ClusterCentroids,
            opann::tree::{CentroidTree, LeafRef},
        },
    };

    /// Deterministic synthetic cells: `n` centroids in `dim` dims with distinct
    /// directions, each tagged a unique cell id (`i*7 + 1`).
    pub(crate) fn synth_cells(n: usize, dim: usize) -> Vec<(Vec<f32>, f32, u128)> {
        (0..n)
            .map(|i| {
                let c: Vec<f32> = (0..dim)
                    .map(|d| (((i * 31 + d * 7 + 3) % 101) as f32) / 50.0 - 1.0)
                    .collect();
                (c, 0.05, (i as u128) * 7 + 1)
            })
            .collect()
    }

    /// Encode `cells` into a manifest-style fp32 [`ClusterCentroids`] (the form
    /// `CentroidTree::build` consumes) plus the parallel cell-id list.
    pub(crate) fn clusters_from_cells(
        dim: usize,
        cells: &[(Vec<f32>, f32, u128)],
    ) -> (ClusterCentroids, Vec<u128>) {
        let n = cells.len() as u32;
        let flat: Vec<f32> = cells
            .iter()
            .flat_map(|(c, _, _)| c.iter().copied())
            .collect();
        let radii: Vec<f32> = cells.iter().map(|(_, r, _)| *r).collect();
        let ids: Vec<u128> = cells.iter().map(|(_, _, id)| *id).collect();
        let clusters = ClusterCentroids::from_fp32(n, dim as u32, &flat, vec![1u32; n as usize])
            .with_radii(radii);
        (clusters, ids)
    }

    /// Build a routing tree from synthetic `cells` (encodes them first via
    /// [`clusters_from_cells`]). Keeps the test call sites terse.
    pub(crate) fn build_tree(
        metric: Metric,
        dim: usize,
        cells: &[(Vec<f32>, f32, u128)],
    ) -> Option<CentroidTree> {
        let (clusters, ids) = clusters_from_cells(dim, cells);
        // Synthetic cells are whole-superfile leaves: `doc_off = 0`, `count = 0`,
        // and the cell id packed into `superfile_id`.
        let leaf_refs: Vec<LeafRef> = ids
            .iter()
            .map(|&id| LeafRef {
                superfile_id: id,
                doc_off: 0,
                count: 0,
                cluster_id: 0,
            })
            .collect();
        CentroidTree::build(metric, &clusters, &leaf_refs)
    }
}
