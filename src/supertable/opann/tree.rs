// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN routing tree — the hierarchical centroid tree over cell centroids,
//! searched on compute (zero object GETs) to select the `limit` nearest
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
//! `ClusterCentroids::score_clusters_into`. Reconstructing fp32 from stored
//! bytes would bypass the mmap'd manifest and is not allowed.
//!
//! This is the in-memory structure + descent (Phase 1). The paged,
//! copy-on-write on-disk layout — content-addressed pages, `(page_hash, offset)`
//! child links, a manifest-resident root hash, so a commit rewrites only the
//! root→leaf path — layers on top in a later phase.

use std::collections::HashMap;

use crate::superfile::vector::cell_posting::{
    EncodedCellRow, manifest_centroid_components_from_row,
};
use crate::superfile::vector::distance::{Metric, distance};
use crate::supertable::manifest::ClusterCentroids;
use crate::supertable::manifest::part::ContentHash;

#[cfg(test)]
use super::descent::best_first;
use super::page::{ChildLink, LeafRef, NodeTopo, encode_page};
use super::paged::SplitPages;

/// Tree fanout: a node has up to this many children. Descent cost is
/// ~`fanout · depth`; depth is `log_fanout(n_cells)`.
const DEFAULT_FANOUT: usize = 16;

/// One Lloyd reassignment pass after farthest-point seeding.
const PARTITION_LLOYD_ITERS: usize = 1;

/// One node of the routing tree. The node's centroid lives in
/// [`CentroidTree::centroids`] at the matching index (`node id == centroid id`).
enum NodeKind {
    /// Internal routing node: ids of its child nodes.
    Internal(Vec<u32>),
    /// Leaf: the cluster (within an object-resident superfile) this routes to.
    Leaf(LeafRef),
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
    /// stored in the manifest); leaf `i` routes to `leaf_refs[i]`. Splits use
    /// fp32 farthest-point seeds + Sq8 SIMD assignment ([`partition_indices_simd`]);
    /// internal nodes reuse the leaf nearest the group mean. Returns `None` for
    /// empty input, `dim == 0`, or a `leaf_refs` length mismatch.
    pub(crate) fn build(
        metric: Metric,
        clusters: &ClusterCentroids,
        leaf_refs: &[LeafRef],
    ) -> Option<Self> {
        let n = clusters.n_cent as usize;
        let dim = clusters.dim as usize;
        if n == 0 || dim == 0 || leaf_refs.len() != n {
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
            metric,
            clusters,
            dim,
            &rows,
            &cell_radii,
            leaf_refs,
            &indices,
            &mut nodes,
            &mut sources,
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

    /// The `limit` nearest cells to `query`, as `(cell_id, distance)` in the
    /// order the descent reached them. Pure compute — zero object GETs.
    /// Best-first descent over the Sq8+residual node centroids: pop the closest
    /// node; a leaf is a probe, an internal node pushes its children. The first
    /// `limit` leaves reached are the routed cells (their ancestors are the
    /// nearest routing points). Approximate by design — `limit` is the recall
    /// knob; the caller GETs one object per returned cell.
    ///
    /// Test-only: production descent runs over the paged on-disk form
    /// ([`super::paged::PagedTree::select_leaves`]); this in-memory descent is
    /// the oracle the page round-trip tests compare against.
    #[cfg(test)]
    pub(crate) fn select_leaves(&self, query: &[f32], limit: usize) -> Vec<(LeafRef, f32)> {
        if limit == 0 || self.nodes.is_empty() || query.len() != self.centroids.dim as usize {
            return Vec::new();
        }
        best_first(
            self.root,
            self.score(self.root, query),
            limit,
            |node, kids| match &self.nodes[node as usize].kind {
                NodeKind::Leaf(leaf) => Some(*leaf),
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
    ///
    /// Test-only: production serializes through the multi-page splitter
    /// ([`Self::to_pages`]); this single-page form is the round-trip oracle.
    #[cfg(test)]
    pub(crate) fn to_page_bytes(&self) -> Vec<u8> {
        let topo: Vec<NodeTopo> = self
            .nodes
            .iter()
            .map(|n| match &n.kind {
                NodeKind::Leaf(leaf) => NodeTopo::Leaf(*leaf),
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
                NodeKind::Leaf(leaf) => topo.push(NodeTopo::Leaf(*leaf)),
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
                            let hash = self.build_page(cp, page_of, pages_nodes, page_root, out);
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
    /// Sq8+residual scorer. Test-only — used by the in-memory [`Self::select_leaves`]
    /// oracle; production descent scores off the paged form.
    #[cfg(test)]
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
/// subtree's root node id. Large groups are split with [`partition_indices_simd`]
/// (fp32 farthest-point seeds, Sq8+residual assignment via
/// [`ClusterCentroids::score_clusters_into`]); internal nodes reuse an existing
/// leaf centroid nearest the group mean — O(n) per level, not all-pairs medoid.
#[allow(clippy::too_many_arguments)]
fn build_subtree(
    metric: Metric,
    clusters: &ClusterCentroids,
    dim: usize,
    rows: &[EncodedCellRow],
    cell_radii: &[f32],
    leaf_refs: &[LeafRef],
    indices: &[usize],
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    // Single cell → leaf.
    if indices.len() == 1 {
        return push_leaf(indices[0], cell_radii, leaf_refs, nodes, sources);
    }
    // Small group → one internal node directly over leaf children.
    if indices.len() <= DEFAULT_FANOUT {
        let children: Vec<u32> = indices
            .iter()
            .map(|&i| push_leaf(i, cell_radii, leaf_refs, nodes, sources))
            .collect();
        return push_internal(metric, clusters, dim, rows, indices, children, nodes, sources);
    }
    let mut groups =
        partition_indices_simd(metric, clusters, dim, rows, indices, DEFAULT_FANOUT);
    if groups.len() <= 1 {
        groups = partition_indices_chunk(indices, DEFAULT_FANOUT);
    }
    let children: Vec<u32> = groups
        .into_iter()
        .map(|g| {
            build_subtree(
                metric,
                clusters,
                dim,
                rows,
                cell_radii,
                leaf_refs,
                &g,
                nodes,
                sources,
            )
        })
        .collect();
    push_internal(metric, clusters, dim, rows, indices, children, nodes, sources)
}

/// Split `indices` into up to `k` non-empty groups: farthest-point fp32 seeds,
/// then assign each row to its nearest seed via [`ClusterCentroids::score_one`]
/// (SIMD kernel — no re-quantize per split).
fn partition_indices_simd(
    metric: Metric,
    clusters: &ClusterCentroids,
    dim: usize,
    rows: &[EncodedCellRow],
    indices: &[usize],
    k: usize,
) -> Vec<Vec<usize>> {
    let n = indices.len();
    if n == 0 {
        return Vec::new();
    }
    let k = k.min(n).max(1);
    if k == 1 {
        return vec![indices.to_vec()];
    }
    let components: Vec<Vec<f32>> = indices
        .iter()
        .map(|&i| manifest_centroid_components_from_row(&rows[i], dim))
        .collect();
    let seed_locals = farthest_point_locals(&components, k, metric);
    let mut seed_globals: Vec<usize> = seed_locals.iter().map(|&l| indices[l]).collect();
    let mut groups = assign_groups_by_seeds(metric, clusters, indices, &components, &seed_globals);
    for _ in 0..PARTITION_LLOYD_ITERS {
        let mut new_seed_globals = Vec::with_capacity(k);
        for g in &groups {
            if g.is_empty() {
                new_seed_globals.push(*seed_globals.first().unwrap_or(&indices[0]));
            } else {
                new_seed_globals.push(medoid_nearest_to_mean(metric, clusters, dim, rows, g));
            }
        }
        seed_globals = new_seed_globals;
        groups = assign_groups_by_seeds(metric, clusters, indices, &components, &seed_globals);
    }
    groups.into_iter().filter(|g| !g.is_empty()).collect()
}

/// Assign each row to the nearest seed cluster in `clusters` (SIMD `score_one`).
fn assign_groups_by_seeds(
    metric: Metric,
    clusters: &ClusterCentroids,
    indices: &[usize],
    components: &[Vec<f32>],
    seed_globals: &[usize],
) -> Vec<Vec<usize>> {
    let k = seed_globals.len();
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (local, comp) in components.iter().enumerate() {
        let mut best_c = 0usize;
        let mut best = f32::INFINITY;
        for (c, &seed_g) in seed_globals.iter().enumerate() {
            let score = clusters.score_one(metric, seed_g, comp);
            if score < best {
                best = score;
                best_c = c;
            }
        }
        groups[best_c].push(indices[local]);
    }
    groups
}

/// Deterministic equal-size chunk split when SIMD assignment fails to divide.
fn partition_indices_chunk(indices: &[usize], k: usize) -> Vec<Vec<usize>> {
    let n = indices.len();
    let k = k.min(n).max(1);
    let chunk = n.div_ceil(k);
    indices
        .chunks(chunk)
        .map(|c| c.to_vec())
        .filter(|g| !g.is_empty())
        .collect()
}

/// Farthest-point seeding over fp32 component vectors (k-means++ style).
fn farthest_point_locals(components: &[Vec<f32>], k: usize, metric: Metric) -> Vec<usize> {
    let n = components.len();
    let k = k.min(n);
    let mut seeds = vec![0usize];
    while seeds.len() < k {
        let mut best_idx = 0usize;
        let mut best_min = f32::NEG_INFINITY;
        for (i, c) in components.iter().enumerate() {
            if seeds.contains(&i) {
                continue;
            }
            let min_d = seeds
                .iter()
                .map(|&s| distance(metric, c, &components[s]))
                .fold(f32::INFINITY, f32::min);
            if min_d > best_min {
                best_min = min_d;
                best_idx = i;
            }
        }
        seeds.push(best_idx);
    }
    seeds
}

/// Pick the leaf whose stored centroid is nearest the group's fp32 mean — one
/// [`ClusterCentroids::score_clusters_into`] pass (SIMD), O(n).
fn medoid_nearest_to_mean(
    metric: Metric,
    clusters: &ClusterCentroids,
    dim: usize,
    rows: &[EncodedCellRow],
    indices: &[usize],
) -> usize {
    let selected: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let sub = clusters.select_rows(&selected);
    let mut mean = vec![0f64; dim];
    for &global_i in indices {
        let comp = manifest_centroid_components_from_row(&rows[global_i], dim);
        for (acc, &x) in mean.iter_mut().zip(&comp) {
            *acc += x as f64;
        }
    }
    let inv = 1.0 / (indices.len() as f64);
    let mean_f32: Vec<f32> = mean.iter().map(|a| (*a * inv) as f32).collect();
    let mut best_local = 0usize;
    let mut best = f32::INFINITY;
    sub.score_clusters_into(metric, &mean_f32, |c, score| {
        if score < best {
            best = score;
            best_local = c as usize;
        }
    });
    indices[best_local]
}

/// Append a leaf node for cell `i`; its centroid is cell `i`'s own (source index
/// `i`) and its target is `leaf_refs[i]` (the cluster within a superfile, or a
/// whole-superfile leaf for the hidden cell shape). Returns its node id.
fn push_leaf(
    i: usize,
    cell_radii: &[f32],
    leaf_refs: &[LeafRef],
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    let id = nodes.len() as u32;
    sources.push(i as u32);
    nodes.push(NodeMeta {
        radius: cell_radii[i],
        kind: NodeKind::Leaf(leaf_refs[i]),
    });
    id
}

/// Append an internal node whose centroid is the leaf nearest the group's fp32
/// mean (SIMD [`ClusterCentroids::score_clusters_into`]), with covering radius
/// over its children.
fn push_internal(
    metric: Metric,
    clusters: &ClusterCentroids,
    dim: usize,
    rows: &[EncodedCellRow],
    indices: &[usize],
    children: Vec<u32>,
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    let medoid = medoid_nearest_to_mean(metric, clusters, dim, rows, indices);
    let medoid_query = manifest_centroid_components_from_row(&rows[medoid], dim);
    let mut radius = 0.0f32;
    for &ch in &children {
        let child_src = sources[ch as usize] as usize;
        let d = clusters.score_one(metric, child_src, &medoid_query);
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
        for metric in [Metric::L2Sq] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            // More nodes than cells (internal nodes added), but never fewer.
            assert!(
                tree.n_nodes() >= n,
                "{metric:?}: nodes {} < cells {n}",
                tree.n_nodes()
            );
            // Probing for "everything" returns exactly the cell-id set.
            let q = cells[0].0.clone();
            let all: HashSet<u128> = tree
                .select_leaves(&q, n)
                .into_iter()
                .map(|(leaf, _)| leaf.superfile_id)
                .collect();
            let want: HashSet<u128> = cells.iter().map(|(_, _, id)| *id).collect();
            assert_eq!(all, want, "{metric:?}: descent must reach every cell");
        }
    }

    #[test]
    fn select_leaves_bounded_and_finds_query_cell() {
        let (dim, n, limit) = (32usize, 300usize, 12usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            // A query placed exactly at a cell's centroid must route to that
            // cell within a modest probe budget (the tree groups by proximity).
            let mut hits = 0usize;
            let probes_per = [3usize, 17, 123, 250];
            for &target in &probes_per {
                let q = cells[target].0.clone();
                let probes = tree.select_leaves(&q, limit);
                assert!(probes.len() <= limit, "{metric:?}: over budget");
                assert!(!probes.is_empty(), "{metric:?}: empty probe set");
                if probes
                    .iter()
                    .any(|(leaf, _)| leaf.superfile_id == cells[target].2)
                {
                    hits += 1;
                }
            }
            assert_eq!(
                hits,
                probes_per.len(),
                "{metric:?}: query-at-centroid must land its own cell in top-{limit}"
            );
        }
    }

    #[test]
    fn matches_flat_nearest_on_a_clustered_layout() {
        // Well-separated clusters: the tree's top-limit should overlap the
        // flat brute-force top-limit strongly (recall sanity, not exactness).
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
        let limit = 16usize;
        let mut total_recall = 0.0f64;
        let n_queries = 8usize;
        for cluster in 0..n_queries {
            let mut q = vec![0.0f32; dim];
            q[cluster % dim] = 5.05;
            let got: HashSet<u128> = tree
                .select_leaves(&q, limit)
                .into_iter()
                .map(|(leaf, _)| leaf.superfile_id)
                .collect();
            let mut flat: Vec<(u128, f32)> = cells
                .iter()
                .map(|(c, _, cid)| (*cid, distance(metric, &q, c)))
                .collect();
            flat.sort_by(|a, b| a.1.total_cmp(&b.1));
            let want: HashSet<u128> = flat[..limit].iter().map(|(cid, _)| *cid).collect();
            let overlap = got.intersection(&want).count();
            total_recall += overlap as f64 / limit as f64;
        }
        let recall = total_recall / n_queries as f64;
        assert!(
            recall >= 0.8,
            "tree routing recall@{limit} = {recall:.3}, expected >= 0.8 on a clustered layout"
        );
    }

    #[test]
    fn descent_selects_all_replicated_cells_for_one_hot_query() {
        // Reproduces the hidden-index cell geometry of the multi-shard
        // time-mirror (16 commits × 16 writer shards ⇒ 256 whole-cell leaves)
        // *purely in memory* — zero storage, zero bench. Each commit fans its
        // 64 one-hot docs across 16 shards; shard `s` owns the same 4 directions
        // every commit, so its cell centroid is `0.25` on slots `{4s..4s+3}`.
        // That yields 16 distinct centroids, each replicated once per commit ⇒
        // 16 copies, 256 cells total.
        //
        // For a one-hot query `e_j` the 16 copies of group `s* = j/4` sit at
        // centroid distance 0.75; every other cell is at 1.25 — a clean margin.
        // A correct centroid descent must therefore return ALL 16 relevant
        // copies well inside an nprobe=64 budget. This isolates the failing
        // end-to-end recall: if this FAILS the bug is in the tree/descent (cell
        // selection); if it PASSES the bug is downstream of descent (leaf probe,
        // hidden→user remap, or the dual-write mirror), not the router.
        const GROUPS: usize = 16;
        const COPIES: usize = 16;
        const SLOTS_PER_GROUP: usize = 4;
        const DIM: usize = GROUPS * SLOTS_PER_GROUP;
        const N_PROBE: usize = 64;
        const CELL_VALUE: f32 = 1.0 / SLOTS_PER_GROUP as f32;

        let radius = ((1.0 - CELL_VALUE).powi(2)
            + (SLOTS_PER_GROUP as f32 - 1.0) * CELL_VALUE.powi(2))
        .sqrt();
        let metric = Metric::L2Sq;
        let mut cells: Vec<(Vec<f32>, f32, u128)> = Vec::new();
        let mut id = 1u128;
        for s in 0..GROUPS {
            let mut centroid = vec![0.0f32; DIM];
            for slot in 0..SLOTS_PER_GROUP {
                centroid[s * SLOTS_PER_GROUP + slot] = CELL_VALUE;
            }
            for _copy in 0..COPIES {
                cells.push((centroid.clone(), radius, id));
                id += 1;
            }
        }
        assert_eq!(cells.len(), GROUPS * COPIES);
        let tree = build_tree(metric, DIM, &cells).expect("tree");

        let mut total_recall = 0.0f64;
        let mut n_queries = 0usize;
        for s_star in 0..GROUPS {
            let mut q = vec![0.0f32; DIM];
            q[s_star * SLOTS_PER_GROUP] = 1.0;
            let truth: HashSet<u128> = (0..COPIES)
                .map(|c| (s_star * COPIES + c) as u128 + 1)
                .collect();
            let got: HashSet<u128> = tree
                .select_leaves(&q, N_PROBE)
                .into_iter()
                .map(|(leaf, _)| leaf.superfile_id)
                .collect();
            total_recall += got.intersection(&truth).count() as f64 / COPIES as f64;
            n_queries += 1;
        }
        let recall = total_recall / n_queries as f64;
        assert!(
            recall >= 0.99,
            "in-memory OPANN descent returned only {recall:.3} of the replicated \
             relevant cells at nprobe={N_PROBE} (margin 0.75 vs 1.25; a correct \
             centroid descent must return all {COPIES} copies). If this fails the \
             recall miss is in the router; if it passes it is downstream of descent."
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
        for metric in [Metric::L2Sq] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            let page = Page::parse(&tree.to_page_bytes()).expect("parse page");
            assert_eq!(page.n_nodes(), tree.n_nodes(), "{metric:?}: node count");
            for &target in &[0usize, 1, 57, 199, 249] {
                let q = &cells[target].0;
                for &k in &[1usize, 8, 32, n] {
                    assert_eq!(
                        tree.select_leaves(q, k),
                        page.select_leaves(q, k),
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
                    if let NodeTopo::Leaf(leaf) = page.topo_at(local) {
                        assert!(
                            leaves.insert(leaf.superfile_id),
                            "budget {budget}: cell {} appears in two pages",
                            leaf.superfile_id
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
