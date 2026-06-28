// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN — Object-Partitioned Approximate Nearest Neighbor.
//!
//! The routing layer for the hidden vector index: a hierarchical centroid tree
//! ([`tree::CentroidTree`]) over the cluster centroids, searched on compute with
//! zero object GETs to admit — by covering radius, no fixed probe budget — the
//! IVF clusters that could hold a top-k vector. The cluster payload lives in
//! immutable, object-resident ≤8 MB IVF superfiles; a query descends the tree
//! (cached pages, no GETs), then fetches the admitted clusters' byte ranges in
//! one coalesced wave.
//!
//! Every node centroid is Sq8+residual (the one internal codec), scored through
//! `Sq8ResidualKernel`. SPANN is disk-partitioned ANN; OPANN is
//! object-partitioned ANN.
//!
//! Scope of this module: the in-memory routing tree ([`tree`]), the shared
//! best-first [`descent`], the paged content-addressed on-disk form ([`page`] +
//! [`paged`]), the object-store page store ([`store`]), and the copy-on-write
//! commit insert ([`insert`]).
#![allow(dead_code)]

mod descent;
pub(crate) mod insert;
pub(crate) mod page;
pub(crate) mod paged;
pub(crate) mod store;
pub(crate) mod tree;

#[cfg(test)]
pub(crate) mod test_util {
    //! Shared unit-test fixtures for the OPANN modules (kept in one place so
    //! the tree and page/paged tests don't each carry their own copy).

    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::ClusterCentroids;
    use crate::supertable::opann::page::LeafRef;
    use crate::supertable::opann::tree::CentroidTree;

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

    /// Encode `cells` into a manifest-style Sq8+residual [`ClusterCentroids`]
    /// (the form `CentroidTree::build` consumes) plus the parallel cell-id list.
    /// The fp32 here is the ingestion-surface stand-in for test setup; the tree
    /// build itself only ever sees the encoded centroids.
    pub(crate) fn clusters_from_cells(
        metric: Metric,
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
        let clusters =
            ClusterCentroids::from_fp32(metric, n, dim as u32, &flat, vec![1u32; n as usize])
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
        let (clusters, ids) = clusters_from_cells(metric, dim, cells);
        // Synthetic cells are whole-superfile leaves (the hidden cell shape):
        // `doc_off = 0`, `count = 0`.
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
