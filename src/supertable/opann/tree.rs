// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN routing tree — the hierarchical centroid tree over cell centroids,
//! searched on compute (zero object GETs) to select the `limit` nearest
//! cells. Leaves are fine centroids (each pointing at a run fragment set);
//! internal nodes are coarse routing points (a member centroid nearest the
//! group mean, with a covering radius). Every node centroid is held in one
//! [`ClusterCentroids`] block (index == node id), scored through
//! [`ClusterCentroids::score_one`].
//!
//! Ported from the `perf/hybrid-spfresh-sq8` branch's `opann::tree`. That
//! branch stores node centroids as Sq8+residual under one shared quantizer;
//! here the internal storage is 32-bit fp32 (the manifest substrate's
//! `ClusterCentroids`), so scoring is a direct fp32 [`distance`] kernel call.
//! The build/split/descent structure is otherwise identical. The branch's
//! paged, content-addressed copy-on-write on-disk layout is intentionally
//! omitted: the whole tree is resident in memory, built from manifest state
//! at table open / manifest swap.

use super::descent::best_first;
use crate::{
    superfile::vector::distance::{Metric, distance},
    supertable::manifest::ClusterCentroids,
};

/// Tree fanout: a node has up to this many children. Descent cost is
/// ~`fanout · depth`; depth is `log_fanout(n_cells)`.
const DEFAULT_FANOUT: usize = 16;

/// One Lloyd reassignment pass after farthest-point seeding.
const PARTITION_LLOYD_ITERS: usize = 1;

/// A routing-tree leaf's target. `superfile_id` names the owning superfile;
/// `doc_off`/`count` are the run's row range within it; `cluster_id` is the
/// fine centroid ordinal. For the SPFresh manifest integration the caller
/// packs the cell-local leaf ordinal into `superfile_id` so it can map a
/// routed leaf back to its [`crate::supertable::manifest::list::ClusterRef`].
///
/// Lives here rather than in a page module: the branch defines it in
/// `opann::page`, which is not ported (no paged form).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeafRef {
    pub(crate) superfile_id: u128,
    pub(crate) doc_off: u32,
    pub(crate) count: u32,
    pub(crate) cluster_id: u32,
}

/// One node of the routing tree. The node's centroid lives in
/// [`CentroidTree::centroids`] at the matching index (`node id == centroid id`).
pub(super) enum NodeKind {
    /// Internal routing node: ids of its child nodes.
    Internal(Vec<u32>),
    /// Leaf: the fine centroid / run this routes to.
    Leaf(LeafRef),
}

pub(super) struct NodeMeta {
    /// Covering radius: the max over cells beneath this node of
    /// `dist(node_centroid, cell_centroid) + cell_radius`. A best-first pruning
    /// hint (not a correctness gate); recall is the empirical bar.
    pub(super) radius: f32,
    pub(super) kind: NodeKind,
}

/// In-memory OPANN routing tree.
pub(crate) struct CentroidTree {
    /// One fp32 centroid per node (index == node id), held in a
    /// [`ClusterCentroids`] block like the branch's Sq8 form.
    centroids: ClusterCentroids,
    nodes: Vec<NodeMeta>,
    root: u32,
    metric: Metric,
}

impl CentroidTree {
    /// Build a routing tree over the cell centroids `clusters` (fp32, as stored
    /// in the manifest); leaf `i` routes to `leaf_refs[i]`. Splits use
    /// farthest-point seeds + nearest-seed assignment
    /// ([`partition_indices`]); internal nodes reuse the leaf nearest the group
    /// mean. Returns `None` for empty input, `dim == 0`, or a `leaf_refs`
    /// length mismatch.
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
            &cell_radii,
            leaf_refs,
            &indices,
            &mut nodes,
            &mut sources,
        );
        // Every node centroid IS an existing cell centroid (its source index),
        // sliced from `clusters` — no re-encode. Per-node covering radii
        // override the sliced cell radii.
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

    /// The `limit` nearest leaves to `query`, as `(LeafRef, distance)` in the
    /// order the descent reached them. Pure compute — zero object GETs.
    /// Best-first descent over the node centroids: pop the closest node; a
    /// leaf is a probe, an internal node pushes its children. The first
    /// `limit` leaves reached are the routed leaves. Approximate by design —
    /// `limit` is the recall knob.
    pub(crate) fn select_leaves(&self, query: &[f32], limit: usize) -> Vec<(LeafRef, f32)> {
        self.select_leaves_where(query, limit, |_| true)
    }

    /// As [`Self::select_leaves`], but a leaf counts toward `limit` only when
    /// `survives(leaf.superfile_id)` — the survival-aware admission for
    /// filtered search. A leaf whose superfile failed the predicate is
    /// **skipped without consuming budget**, and descent keeps going (adaptive
    /// expansion), so the `limit` returned cells are the vector-nearest
    /// *among the predicate-surviving* superfiles. Routing nodes are never
    /// gated — only leaves — so a survivor reachable through a node that mixes
    /// survivors and non-survivors is still found. With an always-true
    /// predicate this is exactly the unfiltered descent.
    pub(crate) fn select_leaves_where(
        &self,
        query: &[f32],
        limit: usize,
        survives: impl Fn(u128) -> bool,
    ) -> Vec<(LeafRef, f32)> {
        if limit == 0 || self.nodes.is_empty() || query.len() != self.centroids.dim as usize {
            return Vec::new();
        }
        best_first(
            self.root,
            self.score_local(self.root, query),
            limit,
            |node, kids| match &self.nodes[node as usize].kind {
                NodeKind::Leaf(leaf) if survives(leaf.superfile_id) => Some(*leaf),
                NodeKind::Leaf(_) => None,
                NodeKind::Internal(children) => {
                    for &ch in children {
                        kids.push((ch, self.score_local(ch, query)));
                    }
                    None
                }
            },
        )
    }

    /// Distance from `query` to node `node`'s centroid via the shared fp32
    /// scorer.
    #[inline]
    pub(super) fn score_local(&self, node: u32, query: &[f32]) -> f32 {
        self.centroids.score_one(self.metric, node as usize, query)
    }

    /// Node `node`'s covering radius.
    #[inline]
    pub(super) fn radius_local(&self, node: u32) -> f32 {
        self.nodes[node as usize].radius
    }

    /// Node `node`'s fp32 centroid slice (length `dim`) — the 32-bit internal
    /// storage, read directly.
    #[inline]
    pub(super) fn centroid_local(&self, node: u32) -> &[f32] {
        self.centroids.centroid(node as usize)
    }

    /// Node `node`'s topology record (leaf target or child ids).
    #[inline]
    pub(super) fn topo_at(&self, node: u32) -> &NodeKind {
        &self.nodes[node as usize].kind
    }

    /// The root node id.
    #[inline]
    pub(super) fn root_local(&self) -> u32 {
        self.root
    }

    /// The tree's centroid dimensionality.
    #[inline]
    pub(super) fn dim(&self) -> usize {
        self.centroids.dim as usize
    }

    /// Total node count (leaves + internal).
    pub(crate) fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
}

/// Recursively build a subtree over `indices` (into `clusters`' rows),
/// appending nodes to `nodes` and, per node, the source cell index whose stored
/// centroid the node reuses to `sources` (kept index-aligned). Returns the
/// subtree's root node id. Large groups are split with [`partition_indices`]
/// (farthest-point seeds, nearest-seed assignment via
/// [`ClusterCentroids::score_one`]); internal nodes reuse an existing leaf
/// centroid nearest the group mean — O(n) per level, not all-pairs medoid.
#[allow(clippy::too_many_arguments)]
fn build_subtree(
    metric: Metric,
    clusters: &ClusterCentroids,
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
        return push_internal(metric, clusters, indices, children, nodes, sources);
    }
    let mut groups = partition_indices(metric, clusters, indices, DEFAULT_FANOUT);
    if groups.len() <= 1 {
        groups = partition_indices_chunk(indices, DEFAULT_FANOUT);
    }
    let children: Vec<u32> = groups
        .into_iter()
        .map(|g| build_subtree(metric, clusters, cell_radii, leaf_refs, &g, nodes, sources))
        .collect();
    push_internal(metric, clusters, indices, children, nodes, sources)
}

/// Split `indices` into up to `k` non-empty groups: farthest-point seeds, then
/// assign each member to its nearest seed via [`ClusterCentroids::score_one`],
/// one Lloyd pass.
fn partition_indices(
    metric: Metric,
    clusters: &ClusterCentroids,
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
        .map(|&i| clusters.centroid(i).to_vec())
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
                new_seed_globals.push(medoid_nearest_to_mean(metric, clusters, g));
            }
        }
        seed_globals = new_seed_globals;
        groups = assign_groups_by_seeds(metric, clusters, indices, &components, &seed_globals);
    }
    groups.into_iter().filter(|g| !g.is_empty()).collect()
}

/// Assign each member to the nearest seed cluster in `clusters` (`score_one`).
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

/// Deterministic equal-size chunk split when seed assignment fails to divide.
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
/// [`ClusterCentroids::score_clusters_into`] pass, O(n).
fn medoid_nearest_to_mean(metric: Metric, clusters: &ClusterCentroids, indices: &[usize]) -> usize {
    let dim = clusters.dim as usize;
    let selected: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
    let sub = clusters.select_rows(&selected);
    let mut mean = vec![0f64; dim];
    for &global_i in indices {
        for (acc, &x) in mean.iter_mut().zip(clusters.centroid(global_i)) {
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
/// `i`) and its target is `leaf_refs[i]`. Returns its node id.
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
/// mean, with covering radius over its children.
fn push_internal(
    metric: Metric,
    clusters: &ClusterCentroids,
    indices: &[usize],
    children: Vec<u32>,
    nodes: &mut Vec<NodeMeta>,
    sources: &mut Vec<u32>,
) -> u32 {
    let medoid = medoid_nearest_to_mean(metric, clusters, indices);
    let medoid_query = clusters.centroid(medoid);
    let mut radius = 0.0f32;
    for &ch in &children {
        let child_src = sources[ch as usize] as usize;
        let d = clusters.score_one(metric, child_src, medoid_query);
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::supertable::opann::test_util::{build_tree, synth_cells};

    #[test]
    fn build_has_one_leaf_per_cell_and_descends_all() {
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        {
            let metric = Metric::L2Sq;
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
    fn survival_aware_descent_admits_only_surviving_superfiles() {
        // A survival-aware descent must yield exactly the unfiltered descent's
        // leaves, filtered to surviving superfiles, first `k` — i.e. the k
        // vector-nearest *among survivors*, in the same best-first order.
        // Skipping a non-surviving leaf must not perturb the relative order of
        // the survivors (it only frees budget for the next survivor).
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        // Survivors: every third cell's superfile id.
        let surviving: HashSet<u128> = cells
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, c)| c.2)
            .collect();
        let survives = |sid: u128| surviving.contains(&sid);
        for metric in [Metric::L2Sq] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            for &target in &[0usize, 1, 57, 150, 199] {
                let q = &cells[target].0;
                let full = tree.select_leaves(q, n);
                for &k in &[1usize, 8, 32, n] {
                    let expected: Vec<(LeafRef, f32)> = full
                        .iter()
                        .copied()
                        .filter(|(leaf, _)| survives(leaf.superfile_id))
                        .take(k)
                        .collect();
                    let got = tree.select_leaves_where(q, k, survives);
                    assert_eq!(got, expected, "{metric:?} target {target} k {k}");
                    assert!(
                        got.iter().all(|(leaf, _)| survives(leaf.superfile_id)),
                        "every admitted leaf survives"
                    );
                }
            }
        }
    }
}
