// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! §8.5 copy-on-write tree update: splice a commit's new partition leaves into
//! the routing tree (and drop merged-away cells), rewriting only the affected
//! pages.
//!
//! This is the write-path counterpart to the rebuild that used to run every
//! commit. Rebuilding the whole tree needed *every* prior partition centroid in
//! fp32 (the only reason a fp32 routing copy was ever persisted) and re-pushed
//! (nearly) every page. An update needs only this commit's new centroids —
//! captured inline at the ingestion surface — and the ids of any cells a
//! compaction merged away; it rewrites only the pages on the paths from those
//! leaves to the root. Everything else is reused by content hash, so the
//! per-commit PUT count is `O(path)`, not `O(partitions)` (the §8.5 / §12
//! amplification win).
//!
//! Decode-free and per-page (§8.3): a new leaf's fp32 centroid is quantized
//! **under the quantizer of the page it joins** (the page's existing rows are
//! copied byte-for-byte; only the appended row is encoded); a deleted leaf is
//! dropped by slicing the page's surviving rows via `select_rows` (a byte
//! copy); and every ancestor page is re-emitted with its centroid block
//! **verbatim**, only its child-page link redirected. No stored Sq8 centroid is
//! ever reconstructed to fp32, and nothing fp32 is persisted.
//!
//! Internal-node centroids are not recomputed on insert (a new leaf is attached
//! under its page's entry node and covering radii are extended to cover it).

use std::collections::{HashMap, HashSet};

use bytes::Bytes;

use crate::superfile::vector::cell_posting::materialize_sq8_residual_row_into_cluster_quant;
use crate::superfile::vector::centroid_block::metric_stores_norm;
use crate::superfile::vector::distance::Metric;
use crate::supertable::manifest::ClusterCentroids;
use crate::supertable::manifest::part::ContentHash;

use super::page::{ChildLink, LeafRef, NodeTopo, Page, PageError, encode_page};
use super::paged::{PageSource, SplitPages};
use super::tree::CentroidTree;

/// Bytes per stored dim in a centroid row: one Sq8 code + one i8 residual —
/// the same `[codes(dim) ‖ residuals(dim)]` row shape `ClusterCentroids.rows`
/// uses, so an appended row is `dim * ROW_BYTES_PER_DIM` long.
const ROW_BYTES_PER_DIM: usize = 2;

/// One new cluster leaf to splice into the routing tree: the cluster's owning
/// superfile id, its `(doc_off, count)` range within that superfile's IVF (so a
/// probe range-GETs exactly the cluster), its fp32 centroid (the k-means center
/// captured at the ingestion surface — never a decode of a stored centroid),
/// and its covering radius. Every routing leaf is one internal IVF cluster:
/// registration, drain, and compaction all emit per-cluster leaves, so the tree
/// routes straight to clusters with no whole-cell leaf.
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

/// Update the routing tree read through `source`: drop every leaf whose cell id
/// is in `removed`, then splice in `added`.
///
/// - `root == None` (no tree yet): builds a genesis tree from `added` (with
///   nothing to remove) via the existing encoded-domain builder.
/// - `root == Some(h)`: copy-on-write delete + insert, returning only the pages
///   that changed (reachable from the new root).
///
/// Returns `Some(split)` with the changed pages + new root, or `None` when the
/// tree should not exist (no root and nothing added, or every leaf removed and
/// nothing added). With `root == Some`, `removed` empty, and `added` empty,
/// returns `Some` with no pages and the unchanged root (caller writes nothing).
pub(crate) fn update_tree(
    source: &dyn PageSource,
    root: Option<ContentHash>,
    metric: Metric,
    dim: usize,
    removed: &[u128],
    added: &[LeafInsert],
    page_budget: usize,
) -> Result<Option<SplitPages>, PageError> {
    let root = match root {
        None if added.is_empty() => return Ok(None),
        None => return Ok(Some(build_genesis(metric, dim, added, page_budget))),
        Some(root) => root,
    };
    let removed_set: HashSet<u128> = removed.iter().copied().collect();
    let mut overlay: HashMap<ContentHash, Vec<u8>> = HashMap::new();
    // Delete first (a merged-away cell's leaf must be gone before new leaves are
    // attached), then insert this commit's new leaves.
    let after_delete = if removed_set.is_empty() {
        Some(root)
    } else {
        delete_subtree(&mut overlay, source, root, metric, &removed_set)?
    };
    let mut cur = match after_delete {
        None if added.is_empty() => return Ok(None),
        None => return Ok(Some(build_genesis(metric, dim, added, page_budget))),
        Some(cur) => cur,
    };
    for leaf in added {
        cur = insert_one_leaf(&mut overlay, source, cur, metric, dim, leaf, page_budget)?;
    }
    // Keep only what's reachable from the final root — intermediate rewrites
    // superseded within this batch are dropped, so the commit PUTs exactly the
    // new tree's pages.
    retain_reachable(&mut overlay, cur)?;
    Ok(Some(SplitPages {
        pages: overlay,
        root: cur,
    }))
}

/// Insert-only convenience wrapper (no removals). Test-only — production goes
/// through [`update_tree`], which also handles compaction's removals.
#[cfg(test)]
pub(crate) fn insert_leaves(
    source: &dyn PageSource,
    root: Option<ContentHash>,
    metric: Metric,
    dim: usize,
    added: &[LeafInsert],
    page_budget: usize,
) -> Result<Option<SplitPages>, PageError> {
    update_tree(source, root, metric, dim, &[], added, page_budget)
}

/// Build the first tree from a commit's partitions (the genesis case). The
/// fp32 centroids are the ingestion surface, quantized once into one
/// [`ClusterCentroids`] for this commit's tree, then split into pages.
fn build_genesis(
    metric: Metric,
    dim: usize,
    leaves: &[LeafInsert],
    page_budget: usize,
) -> SplitPages {
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
        ClusterCentroids::from_fp32(metric, n, dim as u32, &flat, vec![1u32; n as usize])
            .with_radii(radii);
    // `build` only returns `None` for empty/zero-dim/mismatched input, none of
    // which can happen here (non-empty leaves, fixed dim, aligned leaf refs).
    let tree = CentroidTree::build(metric, &clusters, &leaf_refs)
        .expect("genesis tree from non-empty equal-dim leaves");
    tree.to_pages(page_budget)
}

/// Splice one new leaf into the tree rooted at `root`, returning the new root.
/// Reads through `overlay`-then-`source`; writes the rewritten path pages into
/// `overlay`.
fn insert_one_leaf(
    overlay: &mut HashMap<ContentHash, Vec<u8>>,
    source: &dyn PageSource,
    root: ContentHash,
    metric: Metric,
    dim: usize,
    leaf: &LeafInsert,
    page_budget: usize,
) -> Result<ContentHash, PageError> {
    let path = greedy_leaf_page_path(overlay, source, root, &leaf.centroid_fp32)?;
    let leaf_page_hash = *path.last().expect("path always contains at least the root");
    let leaf_page = Page::parse(&fetch_bytes(overlay, source, &leaf_page_hash)?)?;
    // Rebuild the leaf page balanced (fresh medoids, tight radii, a real
    // hierarchy) so the radius-bounded descent can prune it. For a page with
    // cross-page children — whose child centroids live in other pages, out of
    // this rebuild's reach — fall back to the flat-append + structural resplit.
    // Either way the returned subtree replaces the leaf page under its parent via
    // a single re-pointed cross-page link, so the parent's node count is
    // unchanged and the split never propagates as a fan-out increase.
    let mut new_child = if let Some(split) = rebuild_leaf_page(&leaf_page, metric, dim, leaf, page_budget) {
        for (h, b) in split.pages {
            overlay.insert(h, b);
        }
        split.root
    } else {
        let new_page_bytes = append_leaf_into_page(&leaf_page, metric, dim, leaf);
        let appended = Page::parse(&new_page_bytes)?;
        if appended.topo().len() > page_budget {
            let split = appended.resplit(page_budget);
            let root = split.root;
            for (h, b) in split.pages {
                overlay.insert(h, b);
            }
            root
        } else {
            let h = ContentHash::of(&new_page_bytes);
            overlay.insert(h, new_page_bytes);
            h
        }
    };
    let mut old_child = leaf_page_hash;
    // Walk the page path back up to the root, redirecting each ancestor's link
    // to the rewritten child and re-emitting it (centroid block verbatim — only
    // the child-page link and covering radius change).
    for &anc_hash in path[..path.len() - 1].iter().rev() {
        let anc = Page::parse(&fetch_bytes(overlay, source, &anc_hash)?)?;
        let new_anc = redirect_child_page(
            &anc,
            metric,
            old_child,
            new_child,
            &leaf.centroid_fp32,
            leaf.radius,
        );
        let new_anc_hash = ContentHash::of(&new_anc);
        overlay.insert(new_anc_hash, new_anc);
        old_child = anc_hash;
        new_child = new_anc_hash;
    }
    Ok(new_child)
}

/// Recursively rewrite the subtree rooted at `page_hash`, dropping every leaf
/// whose cell id is in `removed` and pruning internal nodes / child pages left
/// childless. Returns the new page hash, `Some(page_hash)` unchanged when this
/// subtree holds none of the removed cells, or `None` if the whole subtree is
/// emptied (the caller drops the link to it). Decode-free: surviving centroids
/// are sliced byte-for-byte via `select_rows`.
fn delete_subtree(
    overlay: &mut HashMap<ContentHash, Vec<u8>>,
    source: &dyn PageSource,
    page_hash: ContentHash,
    metric: Metric,
    removed: &HashSet<u128>,
) -> Result<Option<ContentHash>, PageError> {
    let page = Page::parse(&fetch_bytes(overlay, source, &page_hash)?)?;
    let n = page.centroids().n_cent as usize;

    // 1. Rewrite child pages first; map each distinct child-page hash to its
    //    rewrite (Some(new) — possibly unchanged — or None if emptied).
    let mut child_map: HashMap<ContentHash, Option<ContentHash>> = HashMap::new();
    for node in page.topo() {
        if let NodeTopo::Internal(children) = node {
            for child in children {
                if let ChildLink::Page(h) = child
                    && !child_map.contains_key(h)
                {
                    let res = delete_subtree(overlay, source, *h, metric, removed)?;
                    child_map.insert(*h, res);
                }
            }
        }
    }

    // 2. Fixpoint over local nodes: a leaf dies if removed; an internal dies if
    //    it has no surviving child (in-page alive child, or non-emptied child
    //    page).
    let mut alive = vec![true; n];
    for (i, node) in page.topo().iter().enumerate() {
        if let NodeTopo::Leaf(leaf) = node
            && removed.contains(&leaf.superfile_id)
        {
            alive[i] = false;
        }
    }
    loop {
        let mut changed = false;
        for (i, node) in page.topo().iter().enumerate() {
            if !alive[i] {
                continue;
            }
            if let NodeTopo::Internal(children) = node {
                let has_alive_child = children.iter().any(|c| match c {
                    ChildLink::Local(cl) => alive[*cl as usize],
                    ChildLink::Page(h) => matches!(child_map.get(h), Some(Some(_))),
                });
                if !has_alive_child {
                    alive[i] = false;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    let root_local = page.root_local() as usize;
    if !alive[root_local] {
        return Ok(None);
    }
    let any_node_removed = alive.iter().any(|a| !*a);
    let any_child_changed = child_map
        .iter()
        .any(|(h, r)| !matches!(r, Some(nh) if nh == h));
    if !any_node_removed && !any_child_changed {
        return Ok(Some(page_hash)); // subtree untouched — reuse the existing page
    }

    // 3. Renumber survivors and rebuild (centroid rows sliced verbatim).
    let survivors: Vec<u32> = (0..n as u32).filter(|i| alive[*i as usize]).collect();
    let new_index: HashMap<u32, u32> = survivors
        .iter()
        .enumerate()
        .map(|(ni, &oi)| (oi, ni as u32))
        .collect();
    let new_cc = page.centroids().select_rows(&survivors);
    let mut new_topo: Vec<NodeTopo> = Vec::with_capacity(survivors.len());
    for &oi in &survivors {
        match &page.topo()[oi as usize] {
            NodeTopo::Leaf(cell) => new_topo.push(NodeTopo::Leaf(*cell)),
            NodeTopo::Internal(children) => {
                let mut kids = Vec::new();
                for child in children {
                    match child {
                        ChildLink::Local(cl) if alive[*cl as usize] => {
                            kids.push(ChildLink::Local(new_index[cl]));
                        }
                        ChildLink::Local(_) => {}
                        ChildLink::Page(h) => {
                            if let Some(Some(nh)) = child_map.get(h) {
                                kids.push(ChildLink::Page(*nh));
                            }
                        }
                    }
                }
                new_topo.push(NodeTopo::Internal(kids));
            }
        }
    }
    let new_root_local = new_index[&(root_local as u32)];
    let bytes = encode_page(metric, &new_cc, &new_topo, new_root_local);
    let hash = ContentHash::of(&bytes);
    overlay.insert(hash, bytes);
    Ok(Some(hash))
}

/// Page bytes for `hash`, preferring the in-batch overlay over the base source.
fn fetch_bytes(
    overlay: &HashMap<ContentHash, Vec<u8>>,
    base: &dyn PageSource,
    hash: &ContentHash,
) -> Result<Bytes, PageError> {
    match overlay.get(hash) {
        // Overlay pages are freshly `encode_page`'d Vec<u8>s; copy once into
        // `Bytes` to match the base source's (mmap-backed) `Bytes`.
        Some(b) => Ok(Bytes::from(b.clone())),
        None => base.fetch(hash),
    }
}

/// The page-hash path `[root, …, P]` from the tree root down to the page `P`
/// that holds the leaf nearest `query`, by greedy nearest-child descent (one
/// step per level). Crosses page boundaries on `Page` links.
fn greedy_leaf_page_path(
    overlay: &HashMap<ContentHash, Vec<u8>>,
    base: &dyn PageSource,
    root: ContentHash,
    query: &[f32],
) -> Result<Vec<ContentHash>, PageError> {
    let mut path = vec![root];
    let mut page_hash = root;
    loop {
        let page = Page::parse(&fetch_bytes(overlay, base, &page_hash)?)?;
        let mut local = page.root_local();
        let next_page = loop {
            match page.topo_at(local) {
                NodeTopo::Leaf(_) => return Ok(path),
                NodeTopo::Internal(children) => {
                    if children.is_empty() {
                        return Ok(path);
                    }
                    let mut best: Option<(f32, ChildLink)> = None;
                    for child in children {
                        let d = match child {
                            ChildLink::Local(cl) => page.score_local(*cl, query),
                            ChildLink::Page(h) => {
                                let cp = Page::parse(&fetch_bytes(overlay, base, h)?)?;
                                cp.score_local(cp.root_local(), query)
                            }
                        };
                        let better = match &best {
                            None => true,
                            Some((bd, _)) => d < *bd,
                        };
                        if better {
                            best = Some((d, child.clone()));
                        }
                    }
                    match best.expect("non-empty children has a best").1 {
                        ChildLink::Local(cl) => local = cl,
                        ChildLink::Page(h) => break h,
                    }
                }
            }
        };
        path.push(next_page);
        page_hash = next_page;
    }
}

/// Rebuild a **pure-cluster** leaf page balanced, splicing in the new leaf —
/// fresh medoid centroids, tight per-node covering radii, and a real hierarchy
/// instead of a flat fan under the page root, which is what lets the
/// radius-bounded descent prune the page. Reuses the genesis builder
/// ([`CentroidTree::build`]) over the page's stored Sq8 leaf rows, with the new
/// leaf encoded into the same per-page quantizer — entirely in the encoded
/// domain, no fp32 centroid reconstructed. Re-pages the rebuilt subtree to
/// `page_budget` via [`CentroidTree::to_pages`].
///
/// Returns `None` for a page that carries cross-page child links (a higher-level
/// page): those children's centroids live in other pages, out of this rebuild's
/// reach, so the caller keeps the flat-append + structural-resplit path for them.
/// Almost all inserts land in pure-cluster leaf pages, so this is the common path.
fn rebuild_leaf_page(
    page: &Page,
    metric: Metric,
    dim: usize,
    leaf: &LeafInsert,
    page_budget: usize,
) -> Option<SplitPages> {
    let pure = page.topo().iter().all(|node| match node {
        NodeTopo::Leaf(_) => true,
        NodeTopo::Internal(children) => {
            children.iter().all(|c| matches!(c, ChildLink::Local(_)))
        }
    });
    if !pure {
        return None;
    }

    // Slice the leaf clusters' rows (with their radii/counts) out of the page
    // block under its shared quantizer; the internal medoid rows are dropped —
    // `build` recomputes them.
    let leaf_locals: Vec<u32> = page
        .topo()
        .iter()
        .enumerate()
        .filter_map(|(i, node)| matches!(node, NodeTopo::Leaf(_)).then_some(i as u32))
        .collect();
    let mut leaf_refs: Vec<LeafRef> = leaf_locals
        .iter()
        .map(|&i| match &page.topo()[i as usize] {
            NodeTopo::Leaf(l) => *l,
            NodeTopo::Internal(_) => unreachable!("filtered to leaves"),
        })
        .collect();
    let mut cc = page.centroids().select_rows(&leaf_locals);
    let store_norm = metric_stores_norm(metric);
    let had_radii = cc.radii.len() == cc.n_cent as usize;

    // Encode the new leaf's fp32 centroid into the page's quantizer and append it
    // — the same single-row re-quantization `append_leaf_into_page` uses.
    let single = ClusterCentroids::single(metric, &leaf.centroid_fp32);
    let src_row = &single.to_encoded_rows()[0];
    let mut row_bytes = vec![0u8; dim * ROW_BYTES_PER_DIM];
    let norm = materialize_sq8_residual_row_into_cluster_quant(
        src_row,
        &cc.scale,
        &cc.offset,
        dim,
        &mut row_bytes,
        store_norm,
    );
    push_row(&mut cc, &row_bytes, leaf.radius, had_radii, store_norm, norm);
    leaf_refs.push(LeafRef {
        superfile_id: leaf.superfile_id,
        doc_off: leaf.doc_off,
        count: leaf.count,
        cluster_id: leaf.cluster_id,
    });

    // `build` only returns `None` for empty / zero-dim / mismatched input; the
    // leaf set is non-empty (we just pushed one) with a fixed dim and aligned
    // refs, so this is always `Some` in practice.
    let tree = CentroidTree::build(metric, &cc, &leaf_refs)?;
    Some(tree.to_pages(page_budget))
}

/// Re-emit `page` with the new leaf appended: its fp32 centroid encoded under
/// this page's own quantizer (existing rows copied verbatim), attached as a
/// child of the page's entry node, covering radius extended. Decode-free.
fn append_leaf_into_page(page: &Page, metric: Metric, dim: usize, leaf: &LeafInsert) -> Vec<u8> {
    let store_norm = metric_stores_norm(metric);
    let mut cc = page.centroids().clone();
    let old_n = cc.n_cent as usize;
    let had_radii = cc.radii.len() == old_n;

    // Encode the new fp32 centroid under THIS page's (scale, offset): quantize
    // the single-centroid form down into the page's quantizer. The page's
    // existing rows are untouched; only the appended row is encoded — no stored
    // centroid is decoded.
    let single = ClusterCentroids::single(metric, &leaf.centroid_fp32);
    let src_row = &single.to_encoded_rows()[0];
    let mut row_bytes = vec![0u8; dim * ROW_BYTES_PER_DIM];
    let norm = materialize_sq8_residual_row_into_cluster_quant(
        src_row,
        &cc.scale,
        &cc.offset,
        dim,
        &mut row_bytes,
        store_norm,
    );
    let new_leaf_local = cc.n_cent;
    push_row(
        &mut cc,
        &row_bytes,
        leaf.radius,
        had_radii,
        store_norm,
        norm,
    );

    let mut topo = page.topo().to_vec();
    topo.push(NodeTopo::Leaf(LeafRef {
        superfile_id: leaf.superfile_id,
        doc_off: leaf.doc_off,
        count: leaf.count,
        cluster_id: leaf.cluster_id,
    }));
    let root_local = page.root_local();
    let new_root = match topo[root_local as usize].clone() {
        NodeTopo::Internal(mut children) => {
            children.push(ChildLink::Local(new_leaf_local));
            topo[root_local as usize] = NodeTopo::Internal(children);
            if had_radii {
                let d = page.score_local(root_local, &leaf.centroid_fp32);
                let ri = root_local as usize;
                cc.radii[ri] = cc.radii[ri].max(d + leaf.radius);
            }
            root_local
        }
        NodeTopo::Leaf(_) => {
            // Single-leaf (leaf-rooted) page: introduce a new internal root over
            // the old leaf and the new one. The internal node reuses the new
            // leaf's centroid bytes as its medoid (no new encode).
            let new_internal = cc.n_cent;
            push_row(
                &mut cc,
                &row_bytes,
                leaf.radius,
                had_radii,
                store_norm,
                norm,
            );
            topo.push(NodeTopo::Internal(vec![
                ChildLink::Local(root_local),
                ChildLink::Local(new_leaf_local),
            ]));
            new_internal
        }
    };
    encode_page(metric, &cc, &topo, new_root)
}

/// Append one already-encoded row (and its bookkeeping) to `cc`, keeping the
/// counts / radii / norms vectors length-aligned with `n_cent`.
fn push_row(
    cc: &mut ClusterCentroids,
    row_bytes: &[u8],
    radius: f32,
    had_radii: bool,
    store_norm: bool,
    norm: Option<f32>,
) {
    cc.rows.extend_from_slice(row_bytes);
    cc.n_cent += 1;
    cc.counts.push(1);
    if had_radii {
        cc.radii.push(radius);
    }
    if store_norm {
        cc.norms
            .get_or_insert_with(Vec::new)
            .push(norm.unwrap_or(0.0));
    }
}

/// Re-emit `page` with the child-page link `old → new` redirected and the
/// covering radius of the node that holds it extended to cover the inserted
/// leaf. The centroid block is copied verbatim — only topology and one radius
/// change, so nothing is decoded or re-quantized.
fn redirect_child_page(
    page: &Page,
    metric: Metric,
    old: ContentHash,
    new: ContentHash,
    leaf_centroid: &[f32],
    leaf_radius: f32,
) -> Vec<u8> {
    let mut cc = page.centroids().clone();
    let had_radii = cc.radii.len() == cc.n_cent as usize;
    let mut topo = page.topo().to_vec();
    for (i, node) in topo.iter_mut().enumerate() {
        if let NodeTopo::Internal(children) = node {
            let mut holds_link = false;
            for child in children.iter_mut() {
                if let ChildLink::Page(h) = child
                    && *h == old
                {
                    *child = ChildLink::Page(new);
                    holds_link = true;
                }
            }
            if holds_link && had_radii {
                let d = page.score_local(i as u32, leaf_centroid);
                cc.radii[i] = cc.radii[i].max(d + leaf_radius);
            }
        }
    }
    encode_page(metric, &cc, &topo, page.root_local())
}

/// Drop any page in `pages` not reachable from `root` (intermediate rewrites
/// superseded within the same batch), so the caller persists exactly the new
/// tree's pages. Pages not present in `pages` (the unchanged, already-stored
/// ones) are simply not walked into.
fn retain_reachable(
    pages: &mut HashMap<ContentHash, Vec<u8>>,
    root: ContentHash,
) -> Result<(), PageError> {
    let mut keep: HashMap<ContentHash, Vec<u8>> = HashMap::new();
    let mut stack = vec![root];
    while let Some(h) = stack.pop() {
        if keep.contains_key(&h) {
            continue;
        }
        if let Some(bytes) = pages.remove(&h) {
            let page = Page::parse(&bytes)?;
            for child in page.referenced_pages() {
                stack.push(child);
            }
            keep.insert(h, bytes);
        }
    }
    *pages = keep;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supertable::opann::paged::{PagedTree, ResidentPageSource};
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

    /// Merge the changed pages over a base page set (overlay wins by hash) — the
    /// resident tree after a commit.
    fn merge(
        base: &HashMap<ContentHash, Vec<u8>>,
        changed: &HashMap<ContentHash, Vec<u8>>,
    ) -> HashMap<ContentHash, Vec<u8>> {
        let mut m = base.clone();
        for (h, b) in changed {
            m.insert(*h, b.clone());
        }
        m
    }

    /// Inserting far more leaves than fit in one page, in batches through the
    /// COW path, must keep **every** reachable page within the node budget (the
    /// insert splits an overflowing page locally via `resplit`) — and every
    /// inserted leaf must stay reachable by descent. Without the split, one page
    /// would grow without bound across batches (the ingest blowup).
    #[test]
    fn cow_insert_splits_overflowing_pages_and_stays_bounded() {
        let (dim, n, budget) = (16usize, 150usize, 8usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let genesis = insert_leaves(
                &ResidentPageSource::from_pages(HashMap::new()),
                None,
                metric,
                dim,
                &leaves[..10],
                budget,
            )
            .expect("genesis ok")
            .expect("genesis some");
            let mut pages = genesis.pages;
            let mut root = genesis.root;
            for batch in leaves[10..].chunks(7) {
                let src = ResidentPageSource::from_pages(pages.clone());
                let split = insert_leaves(&src, Some(root), metric, dim, batch, budget)
                    .expect("insert ok")
                    .expect("insert some");
                for (h, b) in split.pages {
                    pages.insert(h, b);
                }
                root = split.root;
            }
            // Every page reachable from the root respects the node budget.
            let mut seen: HashSet<ContentHash> = HashSet::new();
            let mut stack = vec![root];
            let mut reachable = 0usize;
            while let Some(h) = stack.pop() {
                if !seen.insert(h) {
                    continue;
                }
                reachable += 1;
                let page = Page::parse(&pages[&h]).expect("parse");
                assert!(
                    page.topo().len() <= budget,
                    "{metric:?}: page has {} nodes > budget {budget}",
                    page.topo().len()
                );
                for node in page.topo() {
                    if let NodeTopo::Internal(children) = node {
                        for c in children {
                            if let ChildLink::Page(ph) = c {
                                stack.push(*ph);
                            }
                        }
                    }
                }
            }
            assert!(
                reachable > 1,
                "{metric:?}: expected a multi-page tree after splits, got {reachable}"
            );
            // Every inserted leaf is still reachable by descent.
            let paged = PagedTree::new(ResidentPageSource::from_pages(pages.clone()), root);
            let found: HashSet<u128> = paged
                .select_leaves(&cells[0].0, n)
                .expect("descend")
                .into_iter()
                .map(|(leaf, _)| leaf.superfile_id)
                .collect();
            for c in &cells {
                assert!(
                    found.contains(&c.2),
                    "{metric:?}: leaf {} missing after splits",
                    c.2
                );
            }
        }
    }

    /// A commit's carried-forward resident tree — built in memory from the prior
    /// resident pages overlaid with this commit's changed pages, pruned to the
    /// new root, with zero object I/O ([`super::super::store::build_resident_after_commit`])
    /// — must descend **identically** to a tree loaded fresh from the full merged
    /// page set. i.e. carrying forward == reloading. And it must keep only the
    /// pages reachable from the new root (dropping the superseded old path).
    #[test]
    fn carry_forward_resident_descends_like_fresh_load() {
        use crate::supertable::opann::store::build_resident_after_commit;
        let (dim, n, budget) = (16usize, 150usize, 8usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let genesis = insert_leaves(
                &ResidentPageSource::from_pages(HashMap::new()),
                None,
                metric,
                dim,
                &leaves[..40],
                budget,
            )
            .expect("genesis ok")
            .expect("genesis some");
            let prior = ResidentPageSource::from_pages(genesis.pages.clone());
            let split = insert_leaves(&prior, Some(genesis.root), metric, dim, &leaves[40..90], budget)
                .expect("insert ok")
                .expect("insert some");

            // In-memory carry-forward (prior ∪ changed, pruned to new root).
            let carried = build_resident_after_commit(&prior, &split.pages, split.root)
                .expect("carry forward");
            let n_carried = carried.len();
            // Ground truth: the full merged page set, loaded fresh.
            let merged = merge(&genesis.pages, &split.pages);
            let n_merged = merged.len();
            let fresh = ResidentPageSource::from_pages(merged);

            let carried_tree = PagedTree::new(carried, split.root);
            let fresh_tree = PagedTree::new(fresh, split.root);
            for &t in &[0usize, 1, 45, 89] {
                for &k in &[1usize, 8, n] {
                    assert_eq!(
                        carried_tree.select_leaves(&cells[t].0, k).expect("carried"),
                        fresh_tree.select_leaves(&cells[t].0, k).expect("fresh"),
                        "{metric:?}: carry-forward descent must equal fresh load (t={t}, k={k})"
                    );
                }
            }
            assert!(
                n_carried <= n_merged,
                "{metric:?}: carry-forward keeps only reachable pages ({n_carried} <= {n_merged})"
            );
        }
    }

    /// Build a genesis tree, COW-insert a brand-new leaf, and confirm a paged
    /// descent over the resulting pages routes a query at the new leaf's
    /// centroid to that new leaf — i.e. the splice is reachable and scored.
    #[test]
    fn cow_insert_routes_to_new_leaf() {
        let (dim, n) = (16usize, 64usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let genesis = insert_leaves(
                &ResidentPageSource::from_pages(HashMap::new()),
                None,
                metric,
                dim,
                &leaves,
                8,
            )
            .expect("genesis ok")
            .expect("genesis some");
            let src = ResidentPageSource::from_pages(genesis.pages.clone());

            let new_centroid: Vec<f32> = (0..dim).map(|d| 5.0 + d as f32 * 0.1).collect();
            const NEW_ID: u128 = 999_999;
            let inserted = insert_leaves(
                &src,
                Some(genesis.root),
                metric,
                dim,
                &[LeafInsert {
                    superfile_id: NEW_ID,
                    doc_off: 0,
                    count: 0,
                    cluster_id: 0,
                    centroid_fp32: new_centroid.clone(),
                    radius: 0.05,
                }],
                8,
            )
            .expect("insert ok")
            .expect("insert some");

            let merged = merge(&genesis.pages, &inserted.pages);
            let paged = PagedTree::new(ResidentPageSource::from_pages(merged), inserted.root);
            let probes = paged.select_leaves(&new_centroid, n + 1).expect("descend");
            assert!(
                probes.iter().any(|(leaf, _)| leaf.superfile_id == NEW_ID),
                "{metric:?}: inserted leaf {NEW_ID} not reachable by descent"
            );
        }
    }

    /// Build a genesis tree, COW-delete one cell, and confirm descent no longer
    /// returns it while every other cell is still reachable.
    #[test]
    fn cow_delete_removes_only_the_target() {
        let (dim, n) = (16usize, 64usize);
        let cells = synth_cells(n, dim);
        let leaves = leaves_from(&cells);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let genesis = insert_leaves(
                &ResidentPageSource::from_pages(HashMap::new()),
                None,
                metric,
                dim,
                &leaves,
                8,
            )
            .expect("genesis ok")
            .expect("genesis some");
            let src = ResidentPageSource::from_pages(genesis.pages.clone());

            let victim = cells[17].2;
            let updated = update_tree(&src, Some(genesis.root), metric, dim, &[victim], &[], 8)
                .expect("delete ok")
                .expect("delete some");
            let merged = merge(&genesis.pages, &updated.pages);
            let paged = PagedTree::new(ResidentPageSource::from_pages(merged), updated.root);
            // Ask for every cell; the victim must be gone, the rest present.
            let probes = paged.select_leaves(&cells[17].0, n).expect("descend");
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
}
