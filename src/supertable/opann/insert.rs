// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Batch routing-tree update: splice a commit's new partition leaves into the
//! routing tree (and drop merged-away cells).
//!
//! Ported from the `perf/hybrid-spfresh-sq8` branch's `opann::insert`. That
//! branch expressed this as a copy-on-write page splice (rewriting only the
//! pages on the paths from the touched leaves to the root); the paged,
//! content-addressed on-disk form is intentionally omitted here — the whole
//! tree is resident in memory — so every update is the branch's *batch* path
//! ([`rebuild_tree_batch`]): collect the surviving leaves from the prior tree,
//! drop `removed`, append `added`, then one SIMD-balanced
//! [`CentroidTree::build`]. Node centroids are 32-bit fp32 (the one internal
//! storage), captured at the ingestion surface — never a decode of stored
//! rerank bytes.

use std::collections::HashSet;

use super::tree::{CentroidTree, LeafRef, NodeKind};
use crate::{superfile::vector::distance::Metric, supertable::manifest::ClusterCentroids};

/// One new cluster leaf to splice into the routing tree: the cluster's owning
/// superfile id, its `(doc_off, count)` range within that superfile's IVF (so a
/// probe range-GETs exactly the cluster), its fp32 centroid (the k-means center
/// captured at the ingestion surface — never a decode of a stored centroid),
/// and its covering radius. Every routing leaf is one internal IVF cluster:
/// registration, drain, and compaction all emit per-cluster leaves, so the tree
/// routes straight to clusters with no whole-cell leaf.
#[derive(Clone)]
pub(crate) struct LeafInsert {
    pub(crate) superfile_id: u128,
    pub(crate) doc_off: u32,
    pub(crate) count: u32,
    /// Internal IVF cluster ordinal within `superfile_id` — selects the
    /// cluster's Sq8 scale/offset for the offset probe's rerank decode. 0 for
    /// the whole-cell `(0,0)` legacy leaf (unused there).
    pub(crate) cluster_id: u32,
    pub(crate) centroid_fp32: Vec<f32>,
    pub(crate) radius: f32,
}

/// Update the routing tree: drop every leaf whose cell id is in `removed`,
/// then splice in `added`.
///
/// - `prior == None` (no tree yet): builds a genesis tree from `added` (with
///   nothing to remove).
/// - `prior == Some(tree)`: batch rebuild over survivors + `added`.
///
/// Returns `None` when the tree should not exist (no prior tree and nothing
/// added, or every leaf removed and nothing added).
pub(crate) fn update_tree(
    prior: Option<&CentroidTree>,
    metric: Metric,
    dim: usize,
    removed: &[u128],
    added: &[LeafInsert],
) -> Option<CentroidTree> {
    rebuild_tree_batch(prior, metric, dim, removed, added)
}

/// One batch genesis build from manifest-resident leaves (pre-drain query path
/// and drain/compact batch rebuild). No object I/O.
pub(crate) fn build_genesis_from_leaves(
    metric: Metric,
    dim: usize,
    leaves: &[LeafInsert],
) -> Option<CentroidTree> {
    if leaves.is_empty() {
        return None;
    }
    Some(build_genesis(metric, dim, leaves))
}

fn build_genesis(metric: Metric, dim: usize, leaves: &[LeafInsert]) -> CentroidTree {
    let n = leaves.len() as u32;
    let flat: Vec<f32> = leaves
        .iter()
        .flat_map(|l| l.centroid_fp32.iter().copied())
        .collect();
    let radii: Vec<f32> = leaves.iter().map(|l| l.radius).collect();
    let leaf_refs: Vec<LeafRef> = leaves
        .iter()
        .map(|l| LeafRef {
            superfile_id: l.superfile_id,
            doc_off: l.doc_off,
            count: l.count,
            cluster_id: l.cluster_id,
        })
        .collect();
    let clusters =
        ClusterCentroids::from_fp32(n, dim as u32, &flat, vec![1u32; n as usize]).with_radii(radii);
    CentroidTree::build(metric, &clusters, &leaf_refs)
        .expect("genesis tree from non-empty equal-dim leaves")
}

/// Walk every leaf in the resident tree and materialize [`LeafInsert`] rows for
/// batch rebuild (drain / compact only). fp32 centroids are sliced straight
/// out of the tree's node-centroid block — the 32-bit storage.
pub(crate) fn collect_leaf_inserts_from_tree(tree: &CentroidTree, dim: usize) -> Vec<LeafInsert> {
    let mut out = Vec::new();
    for node in 0..tree.n_nodes() as u32 {
        if let NodeKind::Leaf(leaf) = tree.topo_at(node) {
            let mut centroid_fp32 = vec![0f32; dim];
            centroid_fp32.copy_from_slice(tree.centroid_local(node));
            out.push(LeafInsert {
                superfile_id: leaf.superfile_id,
                doc_off: leaf.doc_off,
                count: leaf.count,
                cluster_id: leaf.cluster_id,
                centroid_fp32,
                radius: tree.radius_local(node),
            });
        }
    }
    out
}

/// Batch-(re)build the routing tree for drain / compact: collect surviving
/// leaves from the prior tree, drop `removed` superfile ids, append `added`,
/// then one SIMD [`CentroidTree::build`].
pub(crate) fn rebuild_tree_batch(
    prior: Option<&CentroidTree>,
    metric: Metric,
    dim: usize,
    removed: &[u128],
    added: &[LeafInsert],
) -> Option<CentroidTree> {
    let removed_set: HashSet<u128> = removed.iter().copied().collect();
    let mut all_leaves: Vec<LeafInsert> = match prior {
        Some(tree) => collect_leaf_inserts_from_tree(tree, dim)
            .into_iter()
            .filter(|l| !removed_set.contains(&l.superfile_id))
            .collect(),
        None => Vec::new(),
    };
    all_leaves.extend(added.iter().cloned());
    if all_leaves.is_empty() {
        return None;
    }
    Some(build_genesis(metric, dim, &all_leaves))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        time::{Duration, Instant},
    };

    use super::*;
    use crate::supertable::opann::test_util::synth_cells;

    fn leaves_from(cells: &[(Vec<f32>, f32, u128)]) -> Vec<LeafInsert> {
        cells
            .iter()
            .map(|(c, r, id)| LeafInsert {
                superfile_id: *id,
                doc_off: 0,
                count: 0,
                cluster_id: 0,
                centroid_fp32: c.clone(),
                radius: *r,
            })
            .collect()
    }

    /// Build a genesis tree, splice in a brand-new leaf, and confirm a descent
    /// over the resulting tree routes a query at the new leaf's centroid to
    /// that new leaf — i.e. the splice is reachable and scored.
    #[test]
    fn insert_routes_to_new_leaf() {
        let (dim, n) = (16usize, 64usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        {
            let metric = Metric::L2Sq;
            let genesis = build_genesis_from_leaves(metric, dim, &leaves).expect("genesis some");

            let new_centroid: Vec<f32> = (0..dim).map(|d| 5.0 + d as f32 * 0.1).collect();
            const NEW_ID: u128 = 999_999;
            let updated = update_tree(
                Some(&genesis),
                metric,
                dim,
                &[],
                &[LeafInsert {
                    superfile_id: NEW_ID,
                    doc_off: 0,
                    count: 0,
                    cluster_id: 0,
                    centroid_fp32: new_centroid.clone(),
                    radius: 0.05,
                }],
            )
            .expect("insert some");

            let probes = updated.select_leaves(&new_centroid, n + 1);
            assert!(
                probes.iter().any(|(leaf, _)| leaf.superfile_id == NEW_ID),
                "{metric:?}: inserted leaf {NEW_ID} not reachable by descent"
            );
        }
    }

    /// Build a genesis tree, delete one cell, and confirm descent no longer
    /// returns it while every other cell is still reachable.
    #[test]
    fn delete_removes_only_the_target() {
        let (dim, n) = (16usize, 64usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        {
            let metric = Metric::L2Sq;
            let genesis = build_genesis_from_leaves(metric, dim, &leaves).expect("genesis some");

            let victim = cells[17].2;
            let updated =
                update_tree(Some(&genesis), metric, dim, &[victim], &[]).expect("delete some");
            // Ask for every cell; the victim must be gone, the rest present.
            let probes = updated.select_leaves(&cells[17].0, n);
            let got: HashSet<u128> = probes.iter().map(|(leaf, _)| leaf.superfile_id).collect();
            assert!(
                !got.contains(&victim),
                "{metric:?}: deleted cell still reachable"
            );
            assert!(
                cells
                    .iter()
                    .filter(|(_, _, id)| *id != victim)
                    .all(|(_, _, id)| got.contains(id)),
                "{metric:?}: a surviving cell went missing after delete"
            );
        }
    }

    /// Genesis over a fine-grained leaf set (the drain shape: thousands of
    /// per-cluster leaves) must complete in batch SIMD-balanced build time,
    /// not the old all-pairs medoid path.
    #[test]
    fn genesis_build_batch_at_scale() {
        let (dim, n) = (1024usize, 8000usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        let time_limit = if cfg!(debug_assertions) {
            Duration::from_secs(120)
        } else {
            Duration::from_secs(10)
        };
        let t = Instant::now();
        let g = build_genesis_from_leaves(Metric::L2Sq, dim, &leaves).expect("genesis some");
        let elapsed = t.elapsed();
        eprintln!(
            "[opann] genesis {n} leaves dim={dim}: {elapsed:?} ({} nodes)",
            g.n_nodes()
        );
        assert!(
            elapsed < time_limit,
            "genesis build of {n} leaves took {elapsed:?}"
        );
        let found: HashSet<u128> = g
            .select_leaves(&cells[0].0, n)
            .into_iter()
            .map(|(l, _)| l.superfile_id)
            .collect();
        assert_eq!(found.len(), n, "every genesis leaf must stay reachable");
    }
}
