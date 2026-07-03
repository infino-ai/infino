// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident per-cell OPANN routing trees.
//!
//! The branch this module is ported from persists its routing tree as
//! content-addressed pages and loads them through a page store; that paging
//! layer is intentionally omitted here. Instead the whole routing forest — one
//! [`CentroidTree`] per coarse cell, over that cell's fine-centroid leaves —
//! is built in memory from the manifest's [`SpfreshRoutingIndex`] (plus the
//! resident fine-centroid blob for hidden cells) and held on the table handle:
//! built on table open, rebuilt when the manifest snapshot changes, swapped
//! through an `ArcSwap`. Descent is pure compute with zero object GETs.

use super::tree::{CentroidTree, LeafRef};
use crate::{
    superfile::vector::distance::{Metric, decode_f32_le_vec},
    supertable::{
        hidden_centroids::ResidentCentroids,
        manifest::{
            ClusterCentroids,
            list::{ClusterRef, SpfreshRoutingIndex},
        },
    },
};

/// Size of one little-endian fp32 component in manifest-encoded centroids.
const F32_BYTES: usize = 4;

/// Bits to shift a cell's forest position into the packed leaf handle; the low
/// half holds the leaf's position within that cell.
const LEAF_PACK_SHIFT: u32 = 32;

/// One coarse cell's resident routing tree plus the leaf list it routes to.
/// `tree` is `None` only when the cell has no resolvable leaves (nothing to
/// probe). `leaves[i]` is the [`ClusterRef`] the tree leaf built at position
/// `i` targets.
pub(crate) struct ResidentCellTree {
    pub(crate) cell_id: u32,
    pub(crate) tree: Option<CentroidTree>,
    pub(crate) leaves: Vec<ClusterRef>,
}

/// The whole table's resident routing forest, tagged with the manifest
/// snapshot (and the resident centroid blob instance) it was built from so
/// readers rebuild exactly when either swaps.
#[derive(Default)]
pub(crate) struct SpfreshResidentTrees {
    pub(crate) manifest_id: u64,
    /// Address of the [`ResidentCentroids`] blob this forest resolved hidden
    /// leaves against (`Arc::as_ptr` as usize; 0 = never built). Hidden
    /// maintenance swaps the blob right after the manifest, so a forest built
    /// inside that window is rebuilt as soon as the new blob lands.
    pub(crate) resident_ptr: usize,
    pub(crate) column: String,
    pub(crate) cells: Vec<ResidentCellTree>,
}

impl SpfreshResidentTrees {
    /// Unpack a routed leaf handle back to its `(cell, leaf)` — the
    /// [`ClusterRef`] the descent selected.
    pub(crate) fn cluster_ref(&self, leaf: &LeafRef) -> Option<(&ResidentCellTree, &ClusterRef)> {
        let cell_pos = (leaf.superfile_id >> LEAF_PACK_SHIFT) as usize;
        let leaf_pos = (leaf.superfile_id & ((1u128 << LEAF_PACK_SHIFT) - 1)) as usize;
        let cell = self.cells.get(cell_pos)?;
        Some((cell, cell.leaves.get(leaf_pos)?))
    }
}

/// Decode a manifest-inline centroid (fp32 little-endian bytes) of length
/// `dim`, or `None` on a length mismatch.
pub(crate) fn decode_manifest_centroid(bytes: &[u8], dim: usize) -> Option<Vec<f32>> {
    (bytes.len() == dim.checked_mul(F32_BYTES)?).then(|| decode_f32_le_vec(bytes))
}

/// Build the resident routing forest for `routing`: one [`CentroidTree`] per
/// cell over that cell's resolvable fine-centroid leaves. A leaf's routing
/// centroid comes inline from the cell's `CellTreeNode` (user-table routing)
/// or from the resident fine-centroid blob indexed by
/// `ClusterRef.cluster_id` (hidden routing); leaves whose centroid can't be
/// resolved are dropped. Each tree leaf's handle packs `(cell position, leaf
/// position)` so an admitted leaf maps straight back to its [`ClusterRef`]
/// via [`SpfreshResidentTrees::cluster_ref`].
pub(crate) fn build_resident_trees(
    manifest_id: u64,
    resident_ptr: usize,
    routing: &SpfreshRoutingIndex,
    resident: &ResidentCentroids,
    metric: Metric,
    dim: usize,
) -> SpfreshResidentTrees {
    let mut cells: Vec<ResidentCellTree> = Vec::with_capacity(routing.cells.len());
    for (cell_pos, cell) in routing.cells.iter().enumerate() {
        let mut flat: Vec<f32> = Vec::new();
        let mut counts: Vec<u32> = Vec::new();
        let mut leaf_refs: Vec<LeafRef> = Vec::new();
        let mut leaves: Vec<ClusterRef> = Vec::new();
        for (leaf_idx, leaf) in cell.leaves.iter().enumerate() {
            let centroid = match cell.nodes.get(leaf_idx) {
                Some(node) => decode_manifest_centroid(&node.centroid, dim),
                None => resident.centroid(leaf.cluster_id).map(<[f32]>::to_vec),
            };
            // A resolvable centroid must be exactly `dim` long — a blob whose
            // dim disagrees with the routing (e.g. an empty default blob)
            // yields short slices that would corrupt the flat build buffer.
            let Some(centroid) = centroid.filter(|c| c.len() == dim) else {
                continue;
            };
            let row_count: u32 = leaf
                .fragments
                .iter()
                .map(|f| f.row_count)
                .fold(0u32, u32::saturating_add);
            let leaf_pos = leaves.len();
            flat.extend_from_slice(&centroid);
            counts.push(row_count.max(1));
            leaf_refs.push(LeafRef {
                superfile_id: ((cell_pos as u128) << LEAF_PACK_SHIFT) | leaf_pos as u128,
                doc_off: 0,
                count: row_count,
                cluster_id: leaf.cluster_id,
            });
            leaves.push(leaf.clone());
        }
        let tree = if leaf_refs.is_empty() {
            None
        } else {
            let clusters =
                ClusterCentroids::from_fp32(leaf_refs.len() as u32, dim as u32, &flat, counts);
            CentroidTree::build(metric, &clusters, &leaf_refs)
        };
        cells.push(ResidentCellTree {
            cell_id: cell.cell_id,
            tree,
            leaves,
        });
    }
    SpfreshResidentTrees {
        manifest_id,
        resident_ptr,
        column: routing.column.clone(),
        cells,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        superfile::vector::distance::encode_f32_le_vec,
        supertable::manifest::list::{CellTree, CellTreeNode, RunFragment, RunFragmentKind},
    };

    fn fragment(uri: &str, rows: u32) -> RunFragment {
        RunFragment {
            superfile_uri: uri.to_string(),
            run_id: 0,
            byte_range: (0, 64),
            row_count: rows,
            kind: RunFragmentKind::Base,
        }
    }

    #[test]
    fn builds_trees_from_inline_and_resident_centroids() {
        const DIM: usize = 4;
        // Cell 0: inline centroids (user-table shape). Cell 1: resident blob
        // centroids (hidden shape).
        let inline_cell = CellTree {
            cell_id: 0,
            nodes: vec![
                CellTreeNode {
                    centroid: encode_f32_le_vec(&[1.0, 0.0, 0.0, 0.0]),
                    left: 0,
                    right: 0,
                },
                CellTreeNode {
                    centroid: encode_f32_le_vec(&[0.0, 1.0, 0.0, 0.0]),
                    left: 0,
                    right: 0,
                },
            ],
            leaves: vec![
                ClusterRef {
                    cell_id: 0,
                    cluster_id: 0,
                    fragments: vec![fragment("a", 10)],
                },
                ClusterRef {
                    cell_id: 0,
                    cluster_id: 1,
                    fragments: vec![fragment("b", 10)],
                },
            ],
        };
        let hidden_cell = CellTree {
            cell_id: 1,
            nodes: Vec::new(),
            leaves: vec![ClusterRef {
                cell_id: 1,
                cluster_id: 0,
                fragments: vec![fragment("c", 5)],
            }],
        };
        let routing = SpfreshRoutingIndex {
            column: "emb".into(),
            centroid_blob_uri: None,
            cells: vec![inline_cell, hidden_cell],
        };
        let resident = ResidentCentroids {
            dim: DIM,
            centroids: Arc::from(vec![0.0f32, 0.0, 1.0, 0.0]),
        };
        let forest = build_resident_trees(7, 1, &routing, &resident, Metric::L2Sq, DIM);
        assert_eq!(forest.manifest_id, 7);
        assert_eq!(forest.cells.len(), 2);
        let t0 = forest.cells[0].tree.as_ref().expect("cell 0 tree");
        let t1 = forest.cells[1].tree.as_ref().expect("cell 1 tree");

        // Cell 0's tree routes a query at the first inline centroid to leaf 0.
        let hits = t0.select_leaves(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        let (cell, cref) = forest.cluster_ref(&hits[0].0).expect("mapped");
        assert_eq!(cell.cell_id, 0);
        assert_eq!(cref.cluster_id, 0);
        assert_eq!(cref.fragments[0].superfile_uri, "a");

        // Cell 1's tree resolves through the resident blob.
        let hits = t1.select_leaves(&[0.0, 0.0, 1.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        let (cell, cref) = forest.cluster_ref(&hits[0].0).expect("mapped");
        assert_eq!(cell.cell_id, 1);
        assert_eq!(cref.fragments[0].superfile_uri, "c");
    }

    #[test]
    fn unresolvable_leaves_are_dropped_not_fatal() {
        const DIM: usize = 4;
        let routing = SpfreshRoutingIndex {
            column: "emb".into(),
            centroid_blob_uri: None,
            cells: vec![CellTree {
                cell_id: 0,
                nodes: Vec::new(),
                // cluster_id 9 is out of range for an empty resident blob.
                leaves: vec![ClusterRef {
                    cell_id: 0,
                    cluster_id: 9,
                    fragments: vec![fragment("a", 10)],
                }],
            }],
        };
        let forest = build_resident_trees(
            1,
            1,
            &routing,
            &ResidentCentroids::default(),
            Metric::L2Sq,
            DIM,
        );
        assert_eq!(forest.cells.len(), 1);
        assert!(forest.cells[0].tree.is_none());
        assert!(forest.cells[0].leaves.is_empty());
    }
}
