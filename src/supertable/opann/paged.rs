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

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;

use crate::supertable::manifest::part::ContentHash;

use super::descent::best_first;
use super::page::{ChildLink, LeafRef, NodeTopo, Page, PageError};

/// Where a [`PagedTree`] fetches page bytes by content hash. A warm
/// implementation answers from a resident cache / mmap (no object I/O); a cold
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

/// A [`PageSource`] backed by a resident in-memory page map — the warm routing
/// layer. The whole tree is loaded once (e.g. by `store::load_resident`) and
/// descent then runs entirely against this map with zero object I/O, which is
/// the OPANN routing model: routing lives on compute, warm descent does no GETs.
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

/// Reader over a multi-page routing tree. Holds the page source and the root
/// page hash; descent is stateless across calls (each `select_probes` builds
/// its own per-query page cache).
pub(crate) struct PagedTree<S: PageSource> {
    source: S,
    root: ContentHash,
}

impl<S: PageSource> PagedTree<S> {
    pub(crate) fn new(source: S, root: ContentHash) -> Self {
        Self { source, root }
    }

    /// Up to `n_probe` `(cell_id, distance)` pairs, by best-first descent that
    /// crosses page boundaries on demand. A page is fetched + verified + parsed
    /// at most once and cached for this descent; scoring runs off each page's
    /// Sq8+residual bytes (no fp32 reconstruction). Errors if a page on the
    /// descent path is missing or fails its content-hash check.
    ///
    /// Equivalent to [`Self::select_probes_where`] with an always-true survival
    /// predicate (every leaf admitted) — the unfiltered descent. Production
    /// always goes through `select_probes_where` (even the unfiltered path,
    /// with an always-true predicate); this wrapper is the descent oracle the
    /// round-trip + survival tests compare against.
    #[cfg(test)]
    pub(crate) fn select_probes(
        &self,
        query: &[f32],
        n_probe: usize,
    ) -> Result<Vec<(LeafRef, f32)>, PageError> {
        self.select_probes_where(query, n_probe, |_| true)
    }

    /// As [`Self::select_probes`], but a leaf counts toward `n_probe` only when
    /// `survives(leaf.superfile_id)` — the §5a survival-aware admission for
    /// filtered search. A leaf whose superfile failed the predicate is **skipped
    /// without consuming budget**, and descent keeps going (adaptive expansion),
    /// so the `n_probe` returned cells are the vector-nearest *among the
    /// predicate-surviving* superfiles. Routing nodes are never gated — only
    /// leaves — so a survivor reachable through a node that mixes survivors and
    /// non-survivors is still found. With an always-true predicate this is
    /// exactly the unfiltered descent.
    pub(crate) fn select_probes_where(
        &self,
        query: &[f32],
        n_probe: usize,
        survives: impl Fn(u128) -> bool,
    ) -> Result<Vec<(LeafRef, f32)>, PageError> {
        if n_probe == 0 {
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
        let probes = best_first(seed, root_dist, n_probe, |h, kids| {
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
    use super::*;
    use crate::superfile::vector::distance::Metric;
    use crate::supertable::manifest::ClusterCentroids;
    use crate::supertable::opann::page::encode_page;
    use crate::supertable::opann::test_util::{build_tree, synth_cells};

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
                            tree.select_probes(q, k),
                            paged.select_probes(q, k).expect("descend"),
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
                let full = paged.select_probes(q, n).expect("full descent");
                for &k in &[1usize, 8, 32, n] {
                    let expected: Vec<(LeafRef, f32)> = full
                        .iter()
                        .copied()
                        .filter(|(leaf, _)| survives(leaf.superfile_id))
                        .take(k)
                        .collect();
                    let got = paged
                        .select_probes_where(q, k, survives)
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
        let res = paged.select_probes(&cells[40].0, n);
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

        let res = paged.select_probes(&[1.0, 0.0, 0.0, 0.0], 4);
        assert!(
            matches!(res, Err(PageError::DimMismatch { .. })),
            "got {res:?}"
        );
    }
}
