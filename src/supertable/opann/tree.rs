// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN routing tree — the hierarchical centroid tree over cell centroids,
//! searched on compute (zero object GETs) to select the `n_probe` nearest
//! cells. Leaves are cells (one per ≤8 MB IVF superfile); internal nodes are
//! coarse routing points (the mean of the cells beneath them, with a covering
//! radius). Every node centroid is stored Sq8+residual — held in one
//! [`ClusterCentroids`] under a single shared quantizer, scored through
//! `Sq8ResidualKernel` ([`ClusterCentroids::score_one`]). No fp32 lives in the
//! tree.
//!
//! The fp32 inputs to [`CentroidTree::build`] are the **ingestion-surface**
//! form only — raw vectors / the k-means output that already produces the cell
//! centroids — quantized once into Sq8+residual before storage. They must
//! **never** come from decoding stored Sq8 centroids back to fp32: cell
//! centroids are read from the (mmap'd) manifest as Sq8+residual and stay that
//! way, scored through the kernel against the fp32 query — exactly like
//! `ClusterCentroids::select_cells_adaptive`. Reconstructing fp32 from stored
//! bytes would bypass the mmap'd manifest and is not allowed.
//!
//! This is the in-memory structure + descent (Phase 1). The paged,
//! copy-on-write on-disk layout — content-addressed pages, `(page_hash, offset)`
//! child links, a manifest-resident root hash, so a commit rewrites only the
//! root→leaf path — layers on top in a later phase.

use std::collections::HashMap;

use crate::superfile::vector::cell_posting::{
    EncodedCellRow, distance_encoded_rows_symmetric, encoded_ivf_kmeans, medoid_index_by,
};
use crate::superfile::vector::distance::Metric;
use crate::supertable::manifest::ClusterCentroids;
use crate::supertable::manifest::part::ContentHash;

use super::descent::best_first;
use super::paged::SplitPages;
use super::page::{ChildLink, NodeTopo, encode_page};

/// Tree fanout: a node has up to this many children. Descent cost is
/// ~`fanout · depth`; depth is `log_fanout(n_cells)`.
const DEFAULT_FANOUT: usize = 16;

/// Encoded k-medoids iterations when splitting a node's cells into child groups.
const TREE_KMEANS_ITERS: usize = 8;

/// One node of the routing tree. The node's centroid lives in
/// [`CentroidTree::centroids`] at the matching index (`node id == centroid id`).
enum NodeKind {
    /// Internal routing node: ids of its child nodes.
    Internal(Vec<u32>),
    /// Leaf: the cell (superfile) id this node routes to.
    Leaf(u128),
}

struct NodeMeta {
    /// Covering radius: the max over cells beneath this node of
    /// `dist(node_centroid, cell_centroid) + cell_radius`. A best-first pruning
    /// hint (not a correctness gate); recall is the empirical bar. Carried into
    /// the page's centroid block by [`CentroidTree::to_page_bytes`].
    radius: f32,
    kind: NodeKind,
}

/// In-memory OPANN routing tree.
pub(crate) struct CentroidTree {
    /// One Sq8+residual centroid per node (index == node id), one shared
    /// quantizer across the whole tree.
    centroids: ClusterCentroids,
    nodes: Vec<NodeMeta>,
    root: u32,
    metric: Metric,
}

impl CentroidTree {
    /// Build a routing tree over the cell centroids `clusters` (Sq8+residual, as
    /// stored in the manifest); leaf `i` routes to `cell_ids[i]`. Everything
    /// stays in the **encoded domain** — grouping is `encoded_ivf_kmeans` over
    /// the stored bytes, internal-node centroids are **medoids** (existing cell
    /// centroids, picked by `medoid_index_by`), and the tree's centroid block is
    /// *sliced* from `clusters` under the same shared quantizer (`select_rows`).
    /// No fp32 vector is ever reconstructed and no centroid is re-quantized.
    /// Returns `None` for empty input, `dim == 0`, or a `cell_ids` length
    /// mismatch.
    pub(crate) fn build(
        metric: Metric,
        clusters: &ClusterCentroids,
        cell_ids: &[u128],
    ) -> Option<Self> {
        let n = clusters.n_cent as usize;
        let dim = clusters.dim as usize;
        if n == 0 || dim == 0 || cell_ids.len() != n {
            return None;
        }
        let rows = clusters.to_encoded_rows();
        let cell_radii: Vec<f32> = if clusters.radii.len() == n {
            clusters.radii.clone()
        } else {
            vec![0.0; n]
        };
        let mut nodes: Vec<NodeMeta> = Vec::new();
        // Per node, the source cluster index whose stored centroid the node
        // reuses (a leaf's own cell; an internal node's group medoid).
        let mut sources: Vec<u32> = Vec::new();
        let indices: Vec<usize> = (0..n).collect();
        let root = build_subtree(
            metric, dim, &rows, &cell_radii, cell_ids, &indices, &mut nodes, &mut sources,
        );
        // Every node centroid IS an existing cell centroid (its source index),
        // sliced from `clusters` under the shared quantizer — no re-quantization.
        // Per-node covering radii override the sliced cell radii.
        let centroids = clusters
            .select_rows(&sources)
            .with_radii(nodes.iter().map(|n| n.radius).collect());
        Some(Self {
            centroids,
            nodes,
            root,
            metric,
        })
    }

    /// The `n_probe` nearest cells to `query`, as `(cell_id, distance)` in the
    /// order the descent reached them. Pure compute — zero object GETs.
    /// Best-first descent over the Sq8+residual node centroids: pop the closest
    /// node; a leaf is a probe, an internal node pushes its children. The first
    /// `n_probe` leaves reached are the routed cells (their ancestors are the
    /// nearest routing points). Approximate by design — `n_probe` is the recall
    /// knob; the caller GETs one object per returned cell.
    pub(crate) fn select_probes(&self, query: &[f32], n_probe: usize) -> Vec<(u128, f32)> {
        if n_probe == 0 || self.nodes.is_empty() || query.len() != self.centroids.dim as usize {
            return Vec::new();
        }
        best_first(
            self.root,
            self.score(self.root, query),
            n_probe,
            |node, kids| match &self.nodes[node as usize].kind {
                NodeKind::Leaf(cell) => Some(*cell),
                NodeKind::Internal(children) => {
                    for &ch in children {
                        kids.push((ch, self.score(ch, query)));
                    }
                    None
                }
            },
        )
    }

    /// Serialize the whole tree as a single self-contained OPANN page — the
    /// degenerate one-page case of the §9 paged layout. Every child link is
    /// local; there are no cross-page links yet (the multi-page splitter and
    /// copy-on-write commit build on this). The node centroids are written as
    /// one [`ClusterCentroids`] block (shared quantizer) via
    /// `encode_cluster_centroids`; the topology leg adds only each node's kind
    /// and child ids. Round-trips through [`super::page::Page`] to an identical
    /// descent.
    pub(crate) fn to_page_bytes(&self) -> Vec<u8> {
        let topo: Vec<NodeTopo> = self
            .nodes
            .iter()
            .map(|n| match &n.kind {
                NodeKind::Leaf(cell) => NodeTopo::Leaf(*cell),
                NodeKind::Internal(children) => {
                    NodeTopo::Internal(children.iter().map(|&c| ChildLink::Local(c)).collect())
                }
            })
            .collect();
        encode_page(self.metric, &self.centroids, &topo, self.root)
    }

    /// Split the tree into bounded, content-addressed pages — the §9 paged
    /// layout. Each page is a connected subtree of at most `max_nodes_per_page`
    /// nodes; an edge that would overflow a page is cut, and the child becomes
    /// the root of a new page reached by a cross-page link. Pages are built
    /// leaf-ward first so each child page's content hash is known before the
    /// parent page that links to it is encoded. Returns the distinct pages
    /// (deduped by hash) and the root page's hash. A budget at least the tree's
    /// node count collapses to one page (the [`Self::to_page_bytes`] case).
    pub(crate) fn to_pages(&self, max_nodes_per_page: usize) -> SplitPages {
        let budget = max_nodes_per_page.max(1);
        let n = self.nodes.len();
        // Subtree sizes. Children always have smaller ids than their parent
        // (children are pushed before parents during build), so an ascending
        // pass is post-order: each child's size is final before its parent.
        let mut size = vec![1usize; n];
        for id in 0..n {
            if let NodeKind::Internal(children) = &self.nodes[id].kind {
                size[id] = 1 + children.iter().map(|&c| size[c as usize]).sum::<usize>();
            }
        }
        // Phase A: assign every node to a page (page 0 roots at the tree root).
        let mut page_of = vec![u32::MAX; n];
        let mut pages_nodes: Vec<Vec<u32>> = vec![Vec::new()];
        let mut page_root: Vec<u32> = vec![self.root];
        assign_page(
            &self.nodes,
            &size,
            budget,
            self.root,
            0,
            &mut page_of,
            &mut pages_nodes,
            &mut page_root,
        );
        // Phase B: encode pages bottom-up; the root page yields the tree's hash.
        let mut pages: HashMap<ContentHash, Vec<u8>> = HashMap::new();
        let root_page = page_of[self.root as usize] as usize;
        let root = self.build_page(root_page, &page_of, &pages_nodes, &page_root, &mut pages);
        SplitPages { pages, root }
    }

    /// Encode page `page` (and, recursively, the child pages it links to)
    /// into `out`, returning its content hash. Local node order is the page's
    /// global node ids sorted ascending, so local indices are deterministic and
    /// the resulting bytes (and hash) are stable. A child in another page is
    /// always that page's root, so a cross-page link is just the child page's
    /// hash. The page's node centroids are sliced out of the whole-tree block
    /// under the same shared quantizer — no re-quantization.
    fn build_page(
        &self,
        page: usize,
        page_of: &[u32],
        pages_nodes: &[Vec<u32>],
        page_root: &[u32],
        out: &mut HashMap<ContentHash, Vec<u8>>,
    ) -> ContentHash {
        let mut ids = pages_nodes[page].clone();
        ids.sort_unstable();
        let local_of: HashMap<u32, u32> = ids
            .iter()
            .enumerate()
            .map(|(li, &g)| (g, li as u32))
            .collect();
        let mut topo: Vec<NodeTopo> = Vec::with_capacity(ids.len());
        for &g in &ids {
            match &self.nodes[g as usize].kind {
                NodeKind::Leaf(cell) => topo.push(NodeTopo::Leaf(*cell)),
                NodeKind::Internal(children) => {
                    // Preserve the node's original child order, tagging each as
                    // an in-page or cross-page link, so every descent path
                    // pushes this node's children in the same order.
                    let mut links = Vec::with_capacity(children.len());
                    for &c in children {
                        let cp = page_of[c as usize] as usize;
                        if cp == page {
                            links.push(ChildLink::Local(local_of[&c]));
                        } else {
                            let hash =
                                self.build_page(cp, page_of, pages_nodes, page_root, out);
                            links.push(ChildLink::Page(hash));
                        }
                    }
                    topo.push(NodeTopo::Internal(links));
                }
            }
        }
        let centroids = self.centroids.select_rows(&ids);
        let root_local = local_of[&page_root[page]];
        let bytes = encode_page(self.metric, &centroids, &topo, root_local);
        let hash = ContentHash::of(&bytes);
        out.entry(hash).or_insert(bytes);
        hash
    }

    /// Distance from `query` to node `node`'s centroid via the single
    /// Sq8+residual scorer.
    #[inline]
    fn score(&self, node: u32, query: &[f32]) -> f32 {
        self.centroids.score_one(self.metric, node as usize, query)
    }

    /// Total node count (leaves + internal). Test/observability only.
    #[cfg(test)]
    pub(crate) fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Recursively build a subtree over `indices` (into the encoded cell `rows`),
/// appending nodes to `nodes` and, per node, the source cell index whose stored
/// centroid the node reuses to `sources` (kept index-aligned). Returns the
/// subtree's root node id. Grouping is encoded-domain k-medoids; internal nodes
/// reuse an existing cell centroid (the group medoid), so no centroid is
/// computed and nothing is decoded to fp32.
#[allow(clippy::too_many_arguments)]
fn build_subtree(
    metric: Metric,
    dim: usize,
    rows: &[EncodedCellRow],
    cell_radii: &[f32],
    cell_ids: &[u128],
    indices: &[usize],
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    // Single cell → leaf.
    if indices.len() == 1 {
        return push_leaf(indices[0], cell_radii, cell_ids, nodes, sources);
    }
    // Small group → one internal node directly over leaf children.
    if indices.len() <= DEFAULT_FANOUT {
        let children: Vec<u32> = indices
            .iter()
            .map(|&i| push_leaf(i, cell_radii, cell_ids, nodes, sources))
            .collect();
        return push_internal(metric, dim, rows, indices, children, nodes, sources);
    }
    // Large group → encoded k-medoids into up to DEFAULT_FANOUT child groups
    // (reusing `encoded_ivf_kmeans` over the stored bytes), recurse.
    let subset: Vec<EncodedCellRow> = indices.iter().map(|&i| rows[i].clone()).collect();
    let (_centroids, assign) = encoded_ivf_kmeans(&subset, metric, DEFAULT_FANOUT, TREE_KMEANS_ITERS);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); DEFAULT_FANOUT];
    for (local, &i) in indices.iter().enumerate() {
        groups[assign[local]].push(i);
    }
    let non_empty: Vec<Vec<usize>> = groups.into_iter().filter(|g| !g.is_empty()).collect();
    // No-progress guard: if clustering failed to split (one group holds
    // everything, or only one non-empty group), fall back to a flat internal
    // node over leaf children rather than recursing forever.
    if non_empty.len() <= 1 || non_empty.iter().any(|g| g.len() == indices.len()) {
        let children: Vec<u32> = indices
            .iter()
            .map(|&i| push_leaf(i, cell_radii, cell_ids, nodes, sources))
            .collect();
        return push_internal(metric, dim, rows, indices, children, nodes, sources);
    }
    let children: Vec<u32> = non_empty
        .into_iter()
        .map(|g| build_subtree(metric, dim, rows, cell_radii, cell_ids, &g, nodes, sources))
        .collect();
    push_internal(metric, dim, rows, indices, children, nodes, sources)
}

/// Append a leaf node for cell `i`; its centroid is cell `i`'s own (source index
/// `i`). Returns its node id.
fn push_leaf(
    i: usize,
    cell_radii: &[f32],
    cell_ids: &[u128],
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    let id = nodes.len() as u32;
    sources.push(i as u32);
    nodes.push(NodeMeta {
        radius: cell_radii[i],
        kind: NodeKind::Leaf(cell_ids[i]),
    });
    id
}

/// Append an internal node whose centroid is the **medoid** of the cells under
/// `indices` — an existing cell centroid (its source index), chosen in the
/// encoded domain via [`medoid_index_by`] — and whose covering radius bounds
/// every child's ball (encoded-domain distances). Children are pushed before
/// the parent, so the parent's id is the largest in its subtree (the overall
/// root is the last node).
fn push_internal(
    metric: Metric,
    dim: usize,
    rows: &[EncodedCellRow],
    indices: &[usize],
    children: Vec<u32>,
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    // Medoid of this subtree's cells — an existing encoded centroid.
    let subset: Vec<EncodedCellRow> = indices.iter().map(|&i| rows[i].clone()).collect();
    let medoid_local =
        medoid_index_by(&subset, |a, b| distance_encoded_rows_symmetric(metric, dim, a, b));
    let medoid = indices[medoid_local];
    // Covering radius from the medoid: max over children of
    // dist(medoid, child_centroid) + child_radius.
    let mut radius = 0.0f32;
    for &ch in &children {
        let child_src = sources[ch as usize] as usize;
        let d = distance_encoded_rows_symmetric(metric, dim, &rows[medoid], &rows[child_src]);
        radius = radius.max(d + nodes[ch as usize].radius);
    }
    let id = nodes.len() as u32;
    sources.push(medoid as u32);
    nodes.push(NodeMeta {
        radius,
        kind: NodeKind::Internal(children),
    });
    id
}

/// Assign the subtree rooted at `node` to pages, starting in page `page`
/// (whose root `node` is). Each child subtree that still fits within `budget`
/// is absorbed whole into the current page; the first that doesn't starts a new
/// page rooted at that child, recursively. Every page is therefore a connected
/// subtree of at most `budget` nodes with a single entry root.
#[allow(clippy::too_many_arguments)]
fn assign_page(
    nodes: &[NodeMeta],
    size: &[usize],
    budget: usize,
    node: u32,
    page: usize,
    page_of: &mut [u32],
    pages_nodes: &mut Vec<Vec<u32>>,
    page_root: &mut Vec<u32>,
) {
    page_of[node as usize] = page as u32;
    pages_nodes[page].push(node);
    if let NodeKind::Internal(children) = &nodes[node as usize].kind {
        for &c in children {
            if pages_nodes[page].len() + size[c as usize] <= budget {
                absorb(nodes, c, page, page_of, pages_nodes);
            } else {
                let np = pages_nodes.len();
                pages_nodes.push(Vec::new());
                page_root.push(c);
                assign_page(nodes, size, budget, c, np, page_of, pages_nodes, page_root);
            }
        }
    }
}

/// Mark the whole subtree rooted at `node` as belonging to `page`. Used when a
/// child subtree fits within the current page's remaining budget.
fn absorb(
    nodes: &[NodeMeta],
    node: u32,
    page: usize,
    page_of: &mut [u32],
    pages_nodes: &mut Vec<Vec<u32>>,
) {
    page_of[node as usize] = page as u32;
    pages_nodes[page].push(node);
    if let NodeKind::Internal(children) = &nodes[node as usize].kind {
        for &c in children {
            absorb(nodes, c, page, page_of, pages_nodes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::superfile::vector::distance::distance;
    use crate::supertable::opann::page::Page;
    use crate::supertable::opann::test_util::{build_tree, synth_cells};

    #[test]
    fn build_has_one_leaf_per_cell_and_descends_all() {
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            // More nodes than cells (internal nodes added), but never fewer.
            assert!(tree.n_nodes() >= n, "{metric:?}: nodes {} < cells {n}", tree.n_nodes());
            // Probing for "everything" returns exactly the cell-id set.
            let q = cells[0].0.clone();
            let all: HashSet<u128> = tree
                .select_probes(&q, n)
                .into_iter()
                .map(|(c, _)| c)
                .collect();
            let want: HashSet<u128> = cells.iter().map(|(_, _, id)| *id).collect();
            assert_eq!(all, want, "{metric:?}: descent must reach every cell");
        }
    }

    #[test]
    fn select_probes_bounded_and_finds_query_cell() {
        let (dim, n, n_probe) = (32usize, 300usize, 12usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            // A query placed exactly at a cell's centroid must route to that
            // cell within a modest probe budget (the tree groups by proximity).
            let mut hits = 0usize;
            let probes_per = [3usize, 17, 123, 250];
            for &target in &probes_per {
                let q = cells[target].0.clone();
                let probes = tree.select_probes(&q, n_probe);
                assert!(probes.len() <= n_probe, "{metric:?}: over budget");
                assert!(!probes.is_empty(), "{metric:?}: empty probe set");
                if probes.iter().any(|(c, _)| *c == cells[target].2) {
                    hits += 1;
                }
            }
            assert_eq!(
                hits,
                probes_per.len(),
                "{metric:?}: query-at-centroid must land its own cell in top-{n_probe}"
            );
        }
    }

    #[test]
    fn matches_flat_nearest_on_a_clustered_layout() {
        // Well-separated clusters: the tree's top-n_probe should overlap the
        // flat brute-force top-n_probe strongly (recall sanity, not exactness).
        let dim = 16usize;
        let mut cells: Vec<(Vec<f32>, f32, u128)> = Vec::new();
        let mut id = 1u128;
        for cluster in 0..8usize {
            for k in 0..16usize {
                let mut c = vec![0.0f32; dim];
                c[cluster % dim] = 5.0 + k as f32 * 0.01;
                c[(cluster + 1) % dim] = k as f32 * 0.02;
                cells.push((c, 0.05, id));
                id += 1;
            }
        }
        let metric = Metric::L2Sq;
        let tree = build_tree(metric, dim, &cells).expect("tree");
        let n_probe = 16usize;
        let mut total_recall = 0.0f64;
        let n_queries = 8usize;
        for cluster in 0..n_queries {
            let mut q = vec![0.0f32; dim];
            q[cluster % dim] = 5.05;
            let got: HashSet<u128> = tree
                .select_probes(&q, n_probe)
                .into_iter()
                .map(|(c, _)| c)
                .collect();
            let mut flat: Vec<(u128, f32)> = cells
                .iter()
                .map(|(c, _, cid)| (*cid, distance(metric, &q, c)))
                .collect();
            flat.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: HashSet<u128> = flat[..n_probe].iter().map(|(cid, _)| *cid).collect();
            let overlap = got.intersection(&want).count();
            total_recall += overlap as f64 / n_probe as f64;
        }
        let recall = total_recall / n_queries as f64;
        assert!(
            recall >= 0.8,
            "tree routing recall@{n_probe} = {recall:.3}, expected >= 0.8 on a clustered layout"
        );
    }

    #[test]
    fn page_round_trip_matches_in_memory_descent() {
        // Serializing to a single page and descending off the bytes must
        // reproduce the in-memory descent *exactly* — same cells, same order,
        // same (bit-identical) distances. Both paths score the same
        // Sq8+residual block through `score_one`; `encode_cluster_centroids`
        // round-trips losslessly, so equality is the right bar, not recall.
        let (dim, n) = (32usize, 250usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            let page = Page::parse(&tree.to_page_bytes()).expect("parse page");
            assert_eq!(page.n_nodes(), tree.n_nodes(), "{metric:?}: node count");
            for &target in &[0usize, 1, 57, 199, 249] {
                let q = &cells[target].0;
                for &k in &[1usize, 8, 32, n] {
                    assert_eq!(
                        tree.select_probes(q, k),
                        page.select_probes(q, k),
                        "{metric:?}: page descent must match in-memory (target {target}, k {k})"
                    );
                }
            }
        }
    }

    #[test]
    fn to_pages_respects_budget_and_preserves_every_cell() {
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        let want_cells: HashSet<u128> = cells.iter().map(|(_, _, id)| *id).collect();
        let tree = build_tree(Metric::L2Sq, dim, &cells).expect("tree");
        for &budget in &[1usize, 4, 16, 64] {
            let split = tree.to_pages(budget);
            assert!(
                split.pages.contains_key(&split.root),
                "budget {budget}: root page must be present"
            );
            let mut leaves: HashSet<u128> = HashSet::new();
            for bytes in split.pages.values() {
                let page = Page::parse(bytes).expect("parse page");
                assert!(
                    page.n_nodes() <= budget,
                    "budget {budget}: page has {} nodes",
                    page.n_nodes()
                );
                // Every multi-page page must carry one covering radius per node
                // (the single-page path always did; the split path must too).
                assert_eq!(
                    page.radii().len(),
                    page.n_nodes(),
                    "budget {budget}: page must carry a radius per node"
                );
                for local in 0..page.n_nodes() as u32 {
                    if let NodeTopo::Leaf(cell) = page.topo_at(local) {
                        assert!(
                            leaves.insert(*cell),
                            "budget {budget}: cell {cell} appears in two pages"
                        );
                    }
                }
            }
            assert_eq!(
                leaves, want_cells,
                "budget {budget}: every cell must appear exactly once across pages"
            );
        }
        // A budget at least the total node count (leaves + internal) collapses
        // to a single page.
        assert_eq!(
            tree.to_pages(tree.n_nodes()).pages.len(),
            1,
            "whole tree fits one page"
        );
    }
}
