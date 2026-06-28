// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The multi-page OPANN routing tree: a [`PagedTree`] descends a tree spread
//! across many content-addressed [`Page`]s, fetching each page from a
//! [`PageSource`] only when descent crosses into it. Warm descent over a
//! resident source does zero object I/O; a page is fetched, hash-verified, and
//! parsed at most once per query and cached for the rest of that descent.
//!
//! This is the read side. [`SplitPages`] is the write side's output
//! ([`super::tree::CentroidTree::to_pages`]) — the set of distinct pages plus
//! the root page's hash. Persisting those pages to object storage and stamping
//! the root hash into the manifest is a later increment; here the source is an
//! abstract byte fetcher (in tests, an in-memory map).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use bytes::Bytes;

use crate::supertable::manifest::part::ContentHash;

use super::descent::best_first;
use super::page::{ChildLink, LeafRef, NodeTopo, Page, PageError};

/// Where a [`PagedTree`] fetches page bytes by content hash. A warm
/// implementation answers from a disk-cache mmap (no object I/O); a cold
/// one issues one object GET. The returned bytes are hash-verified by the
/// caller before use.
pub(crate) trait PageSource {
    fn fetch(&self, hash: &ContentHash) -> Result<Bytes, PageError>;
}

/// A routing tree serialized as a set of distinct content-addressed pages plus
/// the root page's hash — the output of [`super::tree::CentroidTree::to_pages`].
/// Pages dedupe by hash, so two byte-identical pages are stored once.
pub(crate) struct SplitPages {
    pub(crate) pages: HashMap<ContentHash, Vec<u8>>,
    pub(crate) root: ContentHash,
}

/// A [`PageSource`] that overlays a write-side page map on a resident base.
/// Copy-on-write commits fetch unchanged pages from the prior tree and only
/// replace hashes present in `overlay`.
pub(crate) struct OverlayPageSource<'a, B: PageSource + ?Sized> {
    base: &'a B,
    overlay: &'a HashMap<ContentHash, Vec<u8>>,
}

impl<'a, B: PageSource + ?Sized> OverlayPageSource<'a, B> {
    pub(crate) fn new(base: &'a B, overlay: &'a HashMap<ContentHash, Vec<u8>>) -> Self {
        Self { base, overlay }
    }
}

impl<B: PageSource + ?Sized> PageSource for OverlayPageSource<'_, B> {
    fn fetch(&self, hash: &ContentHash) -> Result<Bytes, PageError> {
        if let Some(bytes) = self.overlay.get(hash) {
            return Ok(Bytes::copy_from_slice(bytes));
        }
        self.base.fetch(hash)
    }
}

/// A [`PageSource`] backed by a resident page map — the warm routing layer.
/// Pages are mmap-backed slices (from the disk cache or a packed bundle); descent
/// runs against this map with zero object I/O per query.
pub(crate) struct ResidentPageSource {
    pages: HashMap<ContentHash, Bytes>,
}

impl ResidentPageSource {
    /// Build from owned page byte vectors — the in-memory write side and
    /// tests. Each page is copied once into a `Bytes`.
    pub(crate) fn from_pages(pages: HashMap<ContentHash, Vec<u8>>) -> Self {
        Self::from_byte_pages(
            pages
                .into_iter()
                .map(|(h, v)| (h, Bytes::from(v)))
                .collect(),
        )
    }

    /// Build from already-`Bytes` pages — the disk-cache warm-load path, where
    /// each page is mmap-backed, so the source holds the mapping with no copy.
    pub(crate) fn from_byte_pages(pages: HashMap<ContentHash, Bytes>) -> Self {
        Self { pages }
    }

    /// Number of resident pages. Test/observability only.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.pages.len()
    }

    /// Content hashes of every resident page — i.e. every page reachable from
    /// the root this source was loaded for. GC uses this to mark the live
    /// routing-tree pages so it can sweep the orphaned ones.
    pub(crate) fn page_hashes(&self) -> impl Iterator<Item = ContentHash> + '_ {
        self.pages.keys().copied()
    }

    /// The bytes of resident page `hash`, if present. Used to carry unchanged
    /// pages forward into the next manifest version's resident tree without
    /// re-fetching them (content-addressed: same hash ⇒ same bytes).
    pub(crate) fn page(&self, hash: &ContentHash) -> Option<Bytes> {
        self.pages.get(hash).cloned()
    }
}

impl PageSource for ResidentPageSource {
    fn fetch(&self, hash: &ContentHash) -> Result<Bytes, PageError> {
        self.pages
            .get(hash)
            .cloned()
            .ok_or_else(|| PageError::MissingPage(hash.to_hex()))
    }
}

/// A shared, reference-counted [`PageSource`] is itself a [`PageSource`] — it
/// just forwards to the pointee. Lets a cached `Arc<ResidentPageSource>` be
/// handed to [`PagedTree::new`] (which takes its source by value) without
/// cloning the whole resident page map per query.
impl<T: PageSource + ?Sized> PageSource for Arc<T> {
    fn fetch(&self, hash: &ContentHash) -> Result<Bytes, PageError> {
        (**self).fetch(hash)
    }
}

/// A node handle for cross-page descent: which page it lives in (by content
/// hash) and its local index within that page. `Copy` so it rides the descent
/// heap cheaply.
#[derive(Clone, Copy)]
struct PageNode {
    page: ContentHash,
    local: u32,
}

/// A frontier entry for the radius-bounded descent: a node and the near edge of
/// its covering ball (`max(0, d − R)`, the nearest possible distance to any
/// vector beneath it). Ordered so the heap pops the **smallest** near edge first
/// — the most promising, least-prunable node — via a reversed compare, exactly
/// like the best-first [`Cand`](super::descent).
struct Frontier {
    near: f32,
    node: PageNode,
}

impl PartialEq for Frontier {
    fn eq(&self, other: &Self) -> bool {
        self.near == other.near
    }
}
impl Eq for Frontier {}
impl Ord for Frontier {
    fn cmp(&self, other: &Self) -> Ordering {
        other.near.total_cmp(&self.near)
    }
}
impl PartialOrd for Frontier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reader over a multi-page routing tree. Holds the page source and the root
/// page hash; descent is stateless across calls (each `select_leaves` builds
/// its own per-query page cache).
pub(crate) struct PagedTree<S: PageSource> {
    source: S,
    root: ContentHash,
}

impl<S: PageSource> PagedTree<S> {
    pub(crate) fn new(source: S, root: ContentHash) -> Self {
        Self { source, root }
    }

    /// Up to `limit` `(cell_id, distance)` pairs, by best-first descent that
    /// crosses page boundaries on demand. A page is fetched + verified + parsed
    /// at most once and cached for this descent; scoring runs off each page's
    /// Sq8+residual bytes (no fp32 reconstruction). Errors if a page on the
    /// descent path is missing or fails its content-hash check.
    ///
    /// Equivalent to [`Self::select_leaves_where`] with an always-true survival
    /// predicate (every leaf admitted) — the unfiltered descent. Production
    /// always goes through `select_leaves_where` (even the unfiltered path,
    /// with an always-true predicate); this wrapper is the descent oracle the
    /// round-trip + survival tests compare against.
    #[cfg(test)]
    pub(crate) fn select_leaves(
        &self,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(LeafRef, f32)>, PageError> {
        self.select_leaves_where(query, limit, |_| true)
    }

    /// As [`Self::select_leaves`], but a leaf counts toward `limit` only when
    /// `survives(leaf.superfile_id)` — the §5a survival-aware admission for
    /// filtered search. A leaf whose superfile failed the predicate is **skipped
    /// without consuming budget**, and descent keeps going (adaptive expansion),
    /// so the `limit` returned cells are the vector-nearest *among the
    /// predicate-surviving* superfiles. Routing nodes are never gated — only
    /// leaves — so a survivor reachable through a node that mixes survivors and
    /// non-survivors is still found. With an always-true predicate this is
    /// exactly the unfiltered descent.
    pub(crate) fn select_leaves_where(
        &self,
        query: &[f32],
        limit: usize,
        survives: impl Fn(u128) -> bool,
    ) -> Result<Vec<(LeafRef, f32)>, PageError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut cache: HashMap<ContentHash, Page> = HashMap::new();
        ensure(&mut cache, &self.source, &self.root)?;
        let root_page = &cache[&self.root];
        if query.len() != root_page.dim() {
            return Ok(Vec::new());
        }
        let root_local = root_page.root_local();
        let seed = PageNode {
            page: self.root,
            local: root_local,
        };
        let root_dist = root_page.score_local(root_local, query);

        // A fetch failure mid-descent is recorded here and surfaced after the
        // walk (the descent closure must return `Option`, not `Result`); the
        // offending branch is simply not expanded.
        let mut fetch_err: Option<PageError> = None;
        let probes = best_first(seed, root_dist, limit, |h, kids| {
            // Clone the small topology record out so the page borrow is dropped
            // before we mutate the cache to resolve child pages.
            let topo = cache.get(&h.page)?.topo_at(h.local).clone();
            match topo {
                // A leaf is a probe only if its superfile survived the
                // predicate; otherwise skip it (no children pushed) so descent
                // continues without spending budget on it.
                NodeTopo::Leaf(cell) if survives(cell.superfile_id) => Some(cell),
                NodeTopo::Leaf(_) => None,
                // Walk children in their original order — identical to the
                // in-memory and single-page paths — so the heap pops the same
                // sequence even when distances tie.
                NodeTopo::Internal(children) => {
                    for child in children {
                        match child {
                            ChildLink::Local(cl) => {
                                let d = cache[&h.page].score_local(cl, query);
                                kids.push((
                                    PageNode {
                                        page: h.page,
                                        local: cl,
                                    },
                                    d,
                                ));
                            }
                            ChildLink::Page(ph) => {
                                if let Err(e) = ensure(&mut cache, &self.source, &ph) {
                                    fetch_err.get_or_insert(e);
                                    continue;
                                }
                                let child_page = &cache[&ph];
                                // Every page in a tree shares one dim; a child
                                // page that disagrees (corrupt / crafted bytes)
                                // would index past this page's centroid rows in
                                // the kernel. Reject it instead of panicking.
                                if child_page.dim() != query.len() {
                                    fetch_err.get_or_insert(PageError::DimMismatch {
                                        expected: query.len(),
                                        actual: child_page.dim(),
                                    });
                                    continue;
                                }
                                let croot = child_page.root_local();
                                let d = child_page.score_local(croot, query);
                                kids.push((
                                    PageNode {
                                        page: ph,
                                        local: croot,
                                    },
                                    d,
                                ));
                            }
                        }
                    }
                    None
                }
            }
        });
        match fetch_err {
            Some(e) => Err(e),
            None => Ok(probes),
        }
    }

    /// Radius-bounded branch-and-bound descent — the OPANN admission, **no fixed
    /// leaf budget**. Expands nodes by the near edge of their covering ball
    /// (`max(0, d − R)`, the nearest possible distance to any vector beneath the
    /// node) and prunes a subtree the moment that near edge can no longer beat
    /// the running bound.
    ///
    /// The bound starts unbounded and is frozen to `max(d + r)` over the admitted
    /// cluster leaves once they cumulatively cover `floor` vectors — an upper
    /// bound on the k-th nearest vector's distance, derived entirely from
    /// resident metadata (centroids, radii, counts) with **no payload fetched**.
    /// Returns the cluster leaves whose ball reaches that bound, each with its
    /// centroid distance, for the caller's admission + confirmation to fetch and
    /// rerank. `survives` gates leaves (filtered search) without consuming the
    /// bound, like [`Self::select_leaves_where`].
    ///
    /// Leaves with an unrecorded count (0 — the legacy whole-cell leaf) never
    /// advance `covered`, so the bound never freezes and every reachable leaf is
    /// returned: a correct, unpruned fallback. Pruning is only as tight as the
    /// tree's covering radii — a balanced insert keeps them tight; a flat,
    /// stale-medoid tree yields a loose root radius and little pruning (but never
    /// wrong results).
    pub(crate) fn radius_bounded_descent(
        &self,
        query: &[f32],
        floor: usize,
        survives: impl Fn(u128) -> bool,
    ) -> Result<Vec<(LeafRef, f32)>, PageError> {
        let mut cache: HashMap<ContentHash, Page> = HashMap::new();
        ensure(&mut cache, &self.source, &self.root)?;
        let root_page = &cache[&self.root];
        if query.len() != root_page.dim() {
            return Ok(Vec::new());
        }
        let root_local = root_page.root_local();
        let root_near = (root_page.score_local(root_local, query)
            - root_page.radius_local(root_local))
        .max(0.0);

        let mut heap: BinaryHeap<Frontier> = BinaryHeap::new();
        heap.push(Frontier {
            near: root_near,
            node: PageNode {
                page: self.root,
                local: root_local,
            },
        });

        let mut admitted: Vec<(LeafRef, f32)> = Vec::new();
        let mut covered: u64 = 0;
        let mut admitted_far = 0.0f32;
        let mut bound = f32::INFINITY;
        let mut fetch_err: Option<PageError> = None;

        while let Some(Frontier { near, node }) = heap.pop() {
            // The frontier is ordered by near edge ascending, so once the nearest
            // remaining node's ball can't beat the bound, none can.
            if near > bound {
                break;
            }
            // Clone the small topology record so the page borrow is dropped before
            // resolving child pages (which mutates the cache).
            let topo = match cache.get(&node.page) {
                Some(p) => p.topo_at(node.local).clone(),
                None => continue,
            };
            match topo {
                NodeTopo::Leaf(cell) => {
                    if !survives(cell.superfile_id) {
                        continue;
                    }
                    let page = &cache[&node.page];
                    let d = page.score_local(node.local, query);
                    let r = page.radius_local(node.local);
                    admitted.push((cell, d));
                    covered += cell.count as u64;
                    admitted_far = admitted_far.max(d + r);
                    // Freeze the bound once the nearest admitted clusters cover the
                    // floor: a cluster whose near edge is beyond `admitted_far`
                    // cannot hold a vector nearer than the k-th already covered.
                    if covered >= floor as u64 {
                        bound = bound.min(admitted_far);
                    }
                }
                NodeTopo::Internal(children) => {
                    for child in &children {
                        match child {
                            ChildLink::Local(cl) => {
                                let page = &cache[&node.page];
                                let near_c = (page.score_local(*cl, query)
                                    - page.radius_local(*cl))
                                .max(0.0);
                                if near_c <= bound {
                                    heap.push(Frontier {
                                        near: near_c,
                                        node: PageNode {
                                            page: node.page,
                                            local: *cl,
                                        },
                                    });
                                }
                            }
                            ChildLink::Page(ph) => {
                                if let Err(e) = ensure(&mut cache, &self.source, ph) {
                                    fetch_err.get_or_insert(e);
                                    continue;
                                }
                                let cp = &cache[ph];
                                if cp.dim() != query.len() {
                                    fetch_err.get_or_insert(PageError::DimMismatch {
                                        expected: query.len(),
                                        actual: cp.dim(),
                                    });
                                    continue;
                                }
                                let croot = cp.root_local();
                                let near_c = (cp.score_local(croot, query)
                                    - cp.radius_local(croot))
                                .max(0.0);
                                if near_c <= bound {
                                    heap.push(Frontier {
                                        near: near_c,
                                        node: PageNode {
                                            page: *ph,
                                            local: croot,
                                        },
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        match fetch_err {
            Some(e) => Err(e),
            None => Ok(admitted),
        }
    }
}

/// Ensure the page named by `hash` is parsed into `cache`: fetch its bytes,
/// verify they hash to `hash`, parse, and insert. A no-op if already cached.
fn ensure<S: PageSource>(
    cache: &mut HashMap<ContentHash, Page>,
    source: &S,
    hash: &ContentHash,
) -> Result<(), PageError> {
    if cache.contains_key(hash) {
        return Ok(());
    }
    let bytes = source.fetch(hash)?;
    let actual = ContentHash::of(&bytes);
    if actual != *hash {
        return Err(PageError::ContentHashMismatch {
            expected: hash.to_hex(),
            actual: actual.to_hex(),
        });
    }
    cache.insert(*hash, Page::parse(&bytes)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::ClusterCentroids;
    use crate::supertable::opann::page::encode_page;
    use crate::supertable::opann::test_util::{build_tree, synth_cells};
    use crate::supertable::opann::tree::CentroidTree;

    #[test]
    fn radius_bounded_descent_prunes_far_clusters() {
        // Four well-separated one-hot clusters, each a counted leaf. A query at
        // cluster 0 must admit cluster 0 and prune the rest: the bound freezes
        // once the first (counted) leaf covers the floor, and the orthogonal
        // clusters' near edges fall outside it. The full descent, by contrast,
        // returns all four — so the radius-bounded descent is doing real pruning,
        // not just returning everything and getting recall for free.
        let dim = 4usize;
        let metric = Metric::L2Sq;
        let mut flat = Vec::new();
        for c in 0..4usize {
            let mut v = vec![0.0f32; dim];
            v[c] = 1.0;
            flat.extend_from_slice(&v);
        }
        let clusters = ClusterCentroids::from_fp32(metric, 4, dim as u32, &flat, vec![1; 4])
            .with_radii(vec![0.05; 4]);
        let leaf_refs: Vec<LeafRef> = (0..4)
            .map(|c| LeafRef {
                superfile_id: (c as u128) + 1,
                doc_off: 0,
                count: 100,
                cluster_id: 0,
            })
            .collect();
        let tree = CentroidTree::build(metric, &clusters, &leaf_refs).expect("tree");
        let split = tree.to_pages(16);
        let paged = PagedTree::new(
            ResidentPageSource::from_pages(split.pages.clone()),
            split.root,
        );

        let mut q = vec![0.0f32; dim];
        q[0] = 1.0;
        let full = paged.select_leaves(&q, 100).expect("full descent");
        let bounded = paged
            .radius_bounded_descent(&q, 10, |_| true)
            .expect("bounded descent");

        // Correctness: every admitted leaf is one the full descent also reaches.
        let full_ids: HashSet<u128> = full.iter().map(|(l, _)| l.superfile_id).collect();
        assert!(
            bounded.iter().all(|(l, _)| full_ids.contains(&l.superfile_id)),
            "bounded must be a subset of the full descent"
        );
        // Recall: the query's own cluster is admitted.
        assert!(
            bounded.iter().any(|(l, _)| l.superfile_id == 1),
            "cluster 0 (the query's cluster) must be admitted"
        );
        // Pruning engaged: the far orthogonal clusters are not returned.
        assert!(
            bounded.len() < full.len(),
            "pruning must drop far clusters: bounded={} full={}",
            bounded.len(),
            full.len()
        );
    }

    #[test]
    fn resplit_preserves_descent() {
        // `Page::resplit` must re-page a page into a bounded subtree that
        // descends *identically* to the original — same cells, same order, same
        // distances. (The bounded-split test only checked reachability; this is
        // the routing-correctness gate.) Build the whole tree into one page,
        // then resplit at increasingly tight budgets and compare every descent
        // against the in-memory oracle.
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            // One page holding the entire tree (budget >> node count).
            let whole = tree.to_pages(10 * n);
            let page = Page::parse(&whole.pages[&whole.root]).expect("parse whole page");
            for &budget in &[1usize, 4, 16, 64] {
                let split = page.resplit(budget);
                // Every produced page is within budget.
                for bytes in split.pages.values() {
                    let p = Page::parse(bytes).expect("parse subpage");
                    assert!(p.topo().len() <= budget, "{metric:?} budget {budget}");
                }
                let paged = PagedTree::new(
                    ResidentPageSource::from_pages(split.pages.clone()),
                    split.root,
                );
                for &target in &[0usize, 1, 57, 199] {
                    let q = &cells[target].0;
                    for &k in &[1usize, 8, 32, n] {
                        assert_eq!(
                            tree.select_leaves(q, k),
                            paged.select_leaves(q, k).expect("descend resplit"),
                            "{metric:?} budget {budget} target {target} k {k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn paged_descent_matches_in_memory_across_budgets() {
        // Splitting into pages and descending across them must reproduce the
        // in-memory descent *exactly* — same cells, same order, same distances —
        // for every page budget from "one node per page" (every edge crosses a
        // boundary) up to "whole tree in one page". Ordered child links make all
        // paths push identically, so even the heavily-tied synth fixture (many
        // duplicate centroids → tied distances) must agree bit-for-bit; raw
        // equality is the bar, no canonicalization.
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            for &budget in &[1usize, 4, 16, 64, n + 10] {
                let split = tree.to_pages(budget);
                let paged = PagedTree::new(
                    ResidentPageSource::from_pages(split.pages.clone()),
                    split.root,
                );
                for &target in &[0usize, 1, 57, 150, 199] {
                    let q = &cells[target].0;
                    for &k in &[1usize, 8, 32, n] {
                        assert_eq!(
                            tree.select_leaves(q, k),
                            paged.select_leaves(q, k).expect("descend"),
                            "{metric:?} budget {budget} target {target} k {k}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn survival_aware_descent_admits_only_surviving_superfiles() {
        // §5a: a survival-aware descent must yield exactly the unfiltered
        // descent's leaves, filtered to surviving superfiles, first `k` — i.e.
        // the k vector-nearest *among survivors*, in the same best-first order.
        // Skipping a non-surviving leaf must not perturb the relative order of
        // the survivors (it only frees budget for the next survivor).
        let (dim, n) = (24usize, 200usize);
        let cells = synth_cells(n, dim);
        // Survivors: every third cell's superfile id.
        let surviving: std::collections::HashSet<u128> = cells
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, c)| c.2)
            .collect();
        let survives = |sid: u128| surviving.contains(&sid);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let tree = build_tree(metric, dim, &cells).expect("tree");
            let split = tree.to_pages(16);
            let paged = PagedTree::new(
                ResidentPageSource::from_pages(split.pages.clone()),
                split.root,
            );
            for &target in &[0usize, 1, 57, 150, 199] {
                let q = &cells[target].0;
                let full = paged.select_leaves(q, n).expect("full descent");
                for &k in &[1usize, 8, 32, n] {
                    let expected: Vec<(LeafRef, f32)> = full
                        .iter()
                        .copied()
                        .filter(|(leaf, _)| survives(leaf.superfile_id))
                        .take(k)
                        .collect();
                    let got = paged
                        .select_leaves_where(q, k, survives)
                        .expect("survival descent");
                    assert_eq!(got, expected, "{metric:?} target {target} k {k}");
                    assert!(
                        got.iter().all(|(leaf, _)| survives(leaf.superfile_id)),
                        "every admitted leaf survives"
                    );
                }
            }
        }
    }

    #[test]
    fn missing_page_surfaces_error() {
        let (dim, n) = (16usize, 80usize);
        let cells = synth_cells(n, dim);
        let tree = build_tree(Metric::L2Sq, dim, &cells).expect("tree");
        let split = tree.to_pages(4); // many pages, so a non-root one exists
        let mut pages = split.pages.clone();
        let victim = *pages
            .keys()
            .find(|h| **h != split.root)
            .expect("a non-root page");
        pages.remove(&victim);
        let paged = PagedTree::new(ResidentPageSource::from_pages(pages), split.root);
        // Asking for every cell forces descent to cross into the missing page.
        let res = paged.select_leaves(&cells[40].0, n);
        assert!(matches!(res, Err(PageError::MissingPage(_))), "got {res:?}");
    }

    #[test]
    fn child_page_dim_mismatch_surfaces_error() {
        // Hand-assemble a tree whose root page (dim 4) links to a child page of
        // a different dim (2). Honest splits never produce this, but crafted /
        // corrupt-yet-hash-consistent bytes could; cross-page descent must
        // reject it rather than index past the child's centroid rows. NegDot so
        // no norms are required.
        let child_cc = ClusterCentroids::from_fp32(Metric::NegDot, 1, 2, &[1.0, 0.0], vec![1]);
        let child_bytes = encode_page(
            Metric::NegDot,
            &child_cc,
            &[NodeTopo::Leaf(LeafRef {
                superfile_id: 7,
                doc_off: 0,
                count: 0,
                cluster_id: 0,
            })],
            0,
        );
        let child_hash = ContentHash::of(&child_bytes);

        let root_cc =
            ClusterCentroids::from_fp32(Metric::NegDot, 1, 4, &[1.0, 0.0, 0.0, 0.0], vec![1]);
        let root_topo = vec![NodeTopo::Internal(vec![ChildLink::Page(child_hash)])];
        let root_bytes = encode_page(Metric::NegDot, &root_cc, &root_topo, 0);
        let root_hash = ContentHash::of(&root_bytes);

        let mut pages = HashMap::new();
        pages.insert(root_hash, root_bytes);
        pages.insert(child_hash, child_bytes);
        let paged = PagedTree::new(ResidentPageSource::from_pages(pages), root_hash);

        let res = paged.select_leaves(&[1.0, 0.0, 0.0, 0.0], 4);
        assert!(
            matches!(res, Err(PageError::DimMismatch { .. })),
            "got {res:?}"
        );
    }
}
