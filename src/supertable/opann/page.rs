// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! OPANN routing-tree page — the on-disk unit of the §9 paged, content-
//! addressed, copy-on-write tree. A page holds a contiguous group of routing
//! nodes; it is an immutable object named by the blake3 hash of its bytes
//! (same content-addressing as a manifest part), and a node→child link that
//! crosses a page boundary is the child page's hash (entered at that page's
//! root). The manifest holds only the root page's hash, so a commit rewrites
//! just the leaf→root path.
//!
//! A page reuses the existing centroid codec verbatim: every node centroid is
//! Sq8+residual under one shared per-page quantizer, written with
//! [`encode_cluster_centroids`] and scored straight off the bytes through
//! [`ClusterCentroids::score_one`] — no fp32 centroid is ever reconstructed.
//! The only bytes this module adds on top of that block are the topology leg:
//! each node's kind and its child links.
//!
//! Byte layout (all multi-byte integers little-endian):
//!
//! ```text
//!   magic         "OPNP"            (4)
//!   version       u8                (PAGE_FORMAT_VERSION)
//!   metric        u8                (metric_to_id)
//!   reserved      [0; 2]            (future flags; must be zero)
//!   n_nodes       u32
//!   root_local    u32               (node index descent enters this page at)
//!   centroid_len  u32
//!   centroid      [centroid_len]    (encode_cluster_centroids; row i = node i)
//!   topology      n_nodes records, node-major:
//!     kind        u8                (0 = internal, 1 = leaf)
//!     leaf:   cell_id  u128         (object-resident cell/superfile id)
//!     internal: n_local u16, local[n_local] u32   (child node indices, this page)
//!               n_pages u16, pages[n_pages] [32]  (child page content hashes)
//! ```

use crate::superfile::vector::centroid_block::metric_stores_norm;
use crate::superfile::vector::distance::{Metric, metric_from_id, metric_to_id};
use crate::supertable::manifest::ClusterCentroids;
use crate::supertable::manifest::encoding::{
    DecodeError, decode_cluster_centroids, encode_cluster_centroids,
};
use crate::supertable::manifest::part::{BLAKE3_DIGEST_BYTES, ContentHash};

#[cfg(test)]
use super::descent::best_first;

/// Magic at the start of every OPANN routing-tree page.
const PAGE_MAGIC: [u8; 4] = *b"OPNP";
/// Page wire-format version; bumped on any incompatible layout change.
/// v2: leaf nodes carry `(superfile_id u128, doc_off u32, count u32)` —
/// a cluster within a superfile — instead of a bare `cell_id u128`.
const PAGE_FORMAT_VERSION: u8 = 2;
/// Header bytes reserved after the metric tag (must be zero) — room for
/// future per-page flags without a version bump.
const PAGE_HEADER_RESERVED: usize = 2;

/// Node-kind tag: an internal routing node (child links follow).
const NODE_KIND_INTERNAL: u8 = 0;
/// Node-kind tag: a leaf (a 16-byte cell id follows).
const NODE_KIND_LEAF: u8 = 1;

/// Child-link tag: an in-page child (a u32 local node index follows).
const CHILD_LINK_LOCAL: u8 = 0;
/// Child-link tag: a cross-page child (a 32-byte child-page content hash
/// follows; descent enters that page at its own root).
const CHILD_LINK_PAGE: u8 = 1;

/// Little-endian field widths used by the layout above.
const U16_BYTES: usize = 2;
const U32_BYTES: usize = 4;
const U128_BYTES: usize = 16;

/// Fixed page-header width: magic ‖ version ‖ metric ‖ reserved ‖
/// n_nodes(u32) ‖ root_local(u32) ‖ centroid_len(u32).
const PAGE_HEADER_BYTES: usize = PAGE_MAGIC.len() + 1 + 1 + PAGE_HEADER_RESERVED + U32_BYTES * 3;

/// One node's topology — the leg the centroid block can't carry. The node's
/// centroid, radius, norm and count live in the page's [`ClusterCentroids`]
/// block at this node's index; this adds only the tree structure.
/// One child edge of an internal node, kept in the node's original child
/// order. A `Local` child lives in this same page (descent stays in-page); a
/// `Page` child is another page entered at its own root — a page-boundary
/// crossing. Keeping children in one ordered list (rather than splitting locals
/// from page links) means every descent path — in-memory, single-page, and
/// cross-page — pushes a node's children in the *identical* order, so the
/// best-first heap behaves identically and the three paths agree bit-for-bit
/// even when centroid distances tie.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ChildLink {
    /// Child node within this same page, by local index.
    Local(u32),
    /// Child *page*, by content hash; descent enters it at its own root.
    Page(ContentHash),
}

/// A routing-tree leaf's target: a specific cluster inside an object-resident
/// superfile. `superfile_id` names the superfile; `doc_off`/`count` are that
/// cluster's row range within the superfile's IVF, so a probe is one range-GET
/// of the cluster's bytes. A hidden cell is the degenerate case — one cluster
/// spanning the whole (small) superfile: `doc_off == 0`, `count == n_docs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LeafRef {
    pub(crate) superfile_id: u128,
    pub(crate) doc_off: u32,
    pub(crate) count: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum NodeTopo {
    /// Internal routing node: its children in original order (a mix of in-page
    /// and cross-page links). A single-page serialization has only `Local`.
    Internal(Vec<ChildLink>),
    /// Leaf: the cluster (within an object-resident superfile) this routes to.
    Leaf(LeafRef),
}

/// Failures decoding page bytes.
#[derive(Debug, thiserror::Error)]
pub(crate) enum PageError {
    #[error("bad page magic")]
    BadMagic,
    #[error("unsupported page version: {0}")]
    UnsupportedVersion(u8),
    #[error("unknown metric id: {0}")]
    UnknownMetric(u8),
    #[error("truncated page bytes")]
    Truncated,
    #[error("centroid block decode failed: {0}")]
    Centroid(#[from] DecodeError),
    #[error("node count mismatch: header says {header}, centroid block has {block}")]
    NodeCountMismatch { header: u32, block: u32 },
    #[error("unknown node-kind tag: {0}")]
    BadNodeKind(u8),
    #[error("unknown child-link tag: {0}")]
    BadChildLink(u8),
    #[error("node index {index} out of range for {n_nodes} nodes")]
    NodeIndexOutOfRange { index: u32, n_nodes: u32 },
    #[error("metric requires per-centroid norms but the centroid block has none")]
    MissingNorms,
    #[error("page dim {actual} does not match the tree's dim {expected}")]
    DimMismatch { expected: usize, actual: usize },
    #[error("page not found in source: {0}")]
    MissingPage(String),
    #[error("page content-hash mismatch: expected {expected}, got {actual}")]
    ContentHashMismatch { expected: String, actual: String },
}

/// Serialize a contiguous group of routing-tree nodes into one immutable,
/// content-addressable page. `centroids` holds the nodes' Sq8+residual
/// centroids (row `i` == node `i`) under one shared quantizer; `topo` is the
/// per-node tree structure, index-aligned with the centroid rows;
/// `root_local` is the node index at which descent enters this page. The
/// centroid leg is [`encode_cluster_centroids`] verbatim — no new codec.
pub(crate) fn encode_page(
    metric: Metric,
    centroids: &ClusterCentroids,
    topo: &[NodeTopo],
    root_local: u32,
) -> Vec<u8> {
    debug_assert_eq!(centroids.n_cent as usize, topo.len());
    let centroid_bytes = encode_cluster_centroids(centroids);
    let mut out = Vec::with_capacity(PAGE_HEADER_BYTES + centroid_bytes.len());
    out.extend_from_slice(&PAGE_MAGIC);
    out.push(PAGE_FORMAT_VERSION);
    out.push(metric_to_id(metric) as u8);
    out.extend_from_slice(&[0u8; PAGE_HEADER_RESERVED]);
    out.extend_from_slice(&(topo.len() as u32).to_le_bytes());
    out.extend_from_slice(&root_local.to_le_bytes());
    out.extend_from_slice(&(centroid_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&centroid_bytes);
    for node in topo {
        match node {
            NodeTopo::Leaf(leaf) => {
                out.push(NODE_KIND_LEAF);
                out.extend_from_slice(&leaf.superfile_id.to_le_bytes());
                out.extend_from_slice(&leaf.doc_off.to_le_bytes());
                out.extend_from_slice(&leaf.count.to_le_bytes());
            }
            NodeTopo::Internal(children) => {
                debug_assert!(
                    children.len() <= u16::MAX as usize,
                    "child fanout exceeds u16"
                );
                out.push(NODE_KIND_INTERNAL);
                out.extend_from_slice(&(children.len() as u16).to_le_bytes());
                for child in children {
                    match child {
                        ChildLink::Local(idx) => {
                            out.push(CHILD_LINK_LOCAL);
                            out.extend_from_slice(&idx.to_le_bytes());
                        }
                        ChildLink::Page(hash) => {
                            out.push(CHILD_LINK_PAGE);
                            out.extend_from_slice(&hash.0);
                        }
                    }
                }
            }
        }
    }
    out
}

/// A parsed OPANN routing-tree page: the node centroids (Sq8+residual, scored
/// straight off the bytes via the shared kernel — never decoded to fp32) plus
/// each node's topology. Built by [`Page::parse`] from the content-addressed
/// page bytes.
pub(crate) struct Page {
    metric: Metric,
    centroids: ClusterCentroids,
    topo: Vec<NodeTopo>,
    root_local: u32,
}

impl Page {
    /// Parse page bytes produced by [`encode_page`]. Validates the header,
    /// that the centroid block's node count matches the header, and that every
    /// node/child/root index is in range — so descent can index `topo`
    /// without bounds checks.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, PageError> {
        let mut r = Reader::new(bytes);
        if r.take(PAGE_MAGIC.len())? != PAGE_MAGIC {
            return Err(PageError::BadMagic);
        }
        let version = r.u8()?;
        if version != PAGE_FORMAT_VERSION {
            return Err(PageError::UnsupportedVersion(version));
        }
        let metric_id = r.u8()?;
        let metric = metric_from_id(metric_id as u32).ok_or(PageError::UnknownMetric(metric_id))?;
        r.take(PAGE_HEADER_RESERVED)?;
        let n_nodes = r.u32()?;
        let root_local = r.u32()?;
        let centroid_len = r.u32()? as usize;
        let centroids = decode_cluster_centroids(r.take(centroid_len)?)?;
        if centroids.n_cent != n_nodes {
            return Err(PageError::NodeCountMismatch {
                header: n_nodes,
                block: centroids.n_cent,
            });
        }
        // L2Sq / Cosine fold the per-centroid norm into the distance; a block
        // that declares such a metric but carries no norms would panic in the
        // kernel (`norm.expect`). Reject it here rather than at score time.
        if metric_stores_norm(metric) && centroids.norms.is_none() {
            return Err(PageError::MissingNorms);
        }
        if n_nodes != 0 && root_local >= n_nodes {
            return Err(PageError::NodeIndexOutOfRange {
                index: root_local,
                n_nodes,
            });
        }
        let mut topo = Vec::with_capacity(n_nodes as usize);
        for _ in 0..n_nodes {
            match r.u8()? {
                NODE_KIND_LEAF => {
                    let superfile_id = r.u128()?;
                    let doc_off = r.u32()?;
                    let count = r.u32()?;
                    topo.push(NodeTopo::Leaf(LeafRef {
                        superfile_id,
                        doc_off,
                        count,
                    }));
                }
                NODE_KIND_INTERNAL => {
                    let n_children = r.u16()? as usize;
                    let mut children = Vec::with_capacity(n_children);
                    for _ in 0..n_children {
                        match r.u8()? {
                            CHILD_LINK_LOCAL => {
                                let idx = r.u32()?;
                                if idx >= n_nodes {
                                    return Err(PageError::NodeIndexOutOfRange {
                                        index: idx,
                                        n_nodes,
                                    });
                                }
                                children.push(ChildLink::Local(idx));
                            }
                            CHILD_LINK_PAGE => {
                                let mut h = [0u8; BLAKE3_DIGEST_BYTES];
                                h.copy_from_slice(r.take(BLAKE3_DIGEST_BYTES)?);
                                children.push(ChildLink::Page(ContentHash(h)));
                            }
                            other => return Err(PageError::BadChildLink(other)),
                        }
                    }
                    topo.push(NodeTopo::Internal(children));
                }
                other => return Err(PageError::BadNodeKind(other)),
            }
        }
        Ok(Self {
            metric,
            centroids,
            topo,
            root_local,
        })
    }

    /// Up to `n_probe` `(cell_id, distance)` pairs by best-first descent within
    /// this page, scoring node centroids off the Sq8+residual bytes. Single-
    /// page: child *page* links (`NodeTopo::Internal.pages`) are not followed
    /// here — crossing a page boundary needs a page resolver, wired with the
    /// multi-page tree. A single self-contained page has none.
    ///
    /// Test-only: production descent crosses pages via
    /// [`super::paged::PagedTree::select_probes`]; this single-page descent is a
    /// round-trip oracle.
    #[cfg(test)]
    pub(crate) fn select_probes(&self, query: &[f32], n_probe: usize) -> Vec<(LeafRef, f32)> {
        if n_probe == 0 || self.topo.is_empty() || query.len() != self.centroids.dim as usize {
            return Vec::new();
        }
        best_first(
            self.root_local,
            self.score_local(self.root_local, query),
            n_probe,
            |node, kids| match &self.topo[node as usize] {
                NodeTopo::Leaf(cell) => Some(*cell),
                NodeTopo::Internal(children) => {
                    for child in children {
                        match child {
                            ChildLink::Local(cl) => {
                                kids.push((*cl, self.score_local(*cl, query)));
                            }
                            ChildLink::Page(_) => debug_assert!(
                                false,
                                "single-page select_probes reached a cross-page link; \
                                 use PagedTree for a multi-page tree"
                            ),
                        }
                    }
                    None
                }
            },
        )
    }

    /// Centroid dimension.
    pub(crate) fn dim(&self) -> usize {
        self.centroids.dim as usize
    }

    /// The page's node centroids (Sq8+residual, one shared per-page quantizer).
    /// Exposed so a copy-on-write insert can re-emit the page with the existing
    /// centroid bytes verbatim and only the topology changed — no decode.
    pub(crate) fn centroids(&self) -> &ClusterCentroids {
        &self.centroids
    }

    /// The page's per-node topology, index-aligned with [`Self::centroids`].
    pub(crate) fn topo(&self) -> &[NodeTopo] {
        &self.topo
    }

    /// Node index at which descent enters this page.
    pub(crate) fn root_local(&self) -> u32 {
        self.root_local
    }

    /// The topology record for local node `local`.
    pub(crate) fn topo_at(&self, local: u32) -> &NodeTopo {
        &self.topo[local as usize]
    }

    /// Distance from `query` to local node `local`'s centroid, scored straight
    /// off this page's Sq8+residual bytes (no fp32 reconstruction).
    pub(crate) fn score_local(&self, local: u32, query: &[f32]) -> f32 {
        self.centroids.score_one(self.metric, local as usize, query)
    }

    /// Total node count parsed from the page. Test/observability only.
    #[cfg(test)]
    pub(crate) fn n_nodes(&self) -> usize {
        self.topo.len()
    }

    /// Content hashes of every child *page* this page links to (the
    /// page-boundary crossings, across all internal nodes). Walks the page
    /// graph for resident loading and GC reachability.
    pub(crate) fn referenced_pages(&self) -> Vec<ContentHash> {
        let mut out = Vec::new();
        for node in &self.topo {
            if let NodeTopo::Internal(children) = node {
                for child in children {
                    if let ChildLink::Page(hash) = child {
                        out.push(*hash);
                    }
                }
            }
        }
        out
    }

    /// Per-node covering radii carried in the centroid block (empty if the
    /// block stored none).
    #[cfg(test)]
    pub(crate) fn radii(&self) -> &[f32] {
        &self.centroids.radii
    }
}

/// Bounds-checked sequential little-endian reader over page bytes. Every read
/// advances the cursor and fails with [`PageError::Truncated`] past the end,
/// so the parser never panics on a short or corrupt buffer.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], PageError> {
        let end = self.pos.checked_add(n).ok_or(PageError::Truncated)?;
        let slice = self.buf.get(self.pos..end).ok_or(PageError::Truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, PageError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, PageError> {
        let b = self.take(U16_BYTES)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, PageError> {
        let b = self.take(U32_BYTES)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u128(&mut self) -> Result<u128, PageError> {
        let b = self.take(U128_BYTES)?;
        Ok(u128::from_le_bytes(
            b.try_into().expect("take(16) yields 16 bytes"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-node page: leaves 0 (cell 100) and 1 (cell 200) under internal root
    /// 2. Centroids are axis-aligned so a query near a leaf routes to it first.
    fn three_node_page() -> Vec<u8> {
        let dim = 4u32;
        // Row-major fp32 centroids: node0, node1, then the root mean.
        let centroids_fp32 = [
            1.0, 0.0, 0.0, 0.0, // node 0
            0.0, 1.0, 0.0, 0.0, // node 1
            0.5, 0.5, 0.0, 0.0, // node 2 (mean)
        ];
        let mut cc =
            ClusterCentroids::from_fp32(Metric::L2Sq, 3, dim, &centroids_fp32, vec![1, 1, 1]);
        cc.radii = vec![0.1, 0.2, 0.3];
        let topo = vec![
            NodeTopo::Leaf(LeafRef {
                superfile_id: 100,
                doc_off: 0,
                count: 0,
            }),
            NodeTopo::Leaf(LeafRef {
                superfile_id: 200,
                doc_off: 0,
                count: 0,
            }),
            NodeTopo::Internal(vec![ChildLink::Local(0), ChildLink::Local(1)]),
        ];
        encode_page(Metric::L2Sq, &cc, &topo, 2)
    }

    #[test]
    fn encode_parse_round_trip() {
        let bytes = three_node_page();
        let page = Page::parse(&bytes).expect("parse");
        assert_eq!(page.metric, Metric::L2Sq);
        assert_eq!(page.n_nodes(), 3);
        assert_eq!(page.root_local, 2);
        assert_eq!(page.centroids.n_cent, 3);
        assert_eq!(page.centroids.dim, 4);
        // Radii survive the centroid-block round trip.
        assert_eq!(page.centroids.radii, vec![0.1, 0.2, 0.3]);
        assert_eq!(
            page.topo[0],
            NodeTopo::Leaf(LeafRef {
                superfile_id: 100,
                doc_off: 0,
                count: 0
            })
        );
        assert_eq!(
            page.topo[1],
            NodeTopo::Leaf(LeafRef {
                superfile_id: 200,
                doc_off: 0,
                count: 0
            })
        );
        assert_eq!(
            page.topo[2],
            NodeTopo::Internal(vec![ChildLink::Local(0), ChildLink::Local(1)])
        );
    }

    #[test]
    fn descends_to_nearest_leaf_first() {
        let page = Page::parse(&three_node_page()).expect("parse");
        // Query at node 0's centroid → cell 100 leads, then 200.
        let probes: Vec<u128> = page
            .select_probes(&[1.0, 0.0, 0.0, 0.0], 2)
            .into_iter()
            .map(|(leaf, _)| leaf.superfile_id)
            .collect();
        assert_eq!(probes, vec![100, 200]);
        // Query at node 1's centroid → cell 200 leads.
        let probes: Vec<u128> = page
            .select_probes(&[0.0, 1.0, 0.0, 0.0], 2)
            .into_iter()
            .map(|(leaf, _)| leaf.superfile_id)
            .collect();
        assert_eq!(probes, vec![200, 100]);
        // Probe budget is respected.
        assert_eq!(page.select_probes(&[1.0, 0.0, 0.0, 0.0], 1).len(), 1);
    }

    #[test]
    fn rejects_corrupt_pages() {
        let good = three_node_page();

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xff;
        assert!(matches!(Page::parse(&bad_magic), Err(PageError::BadMagic)));

        let mut bad_version = good.clone();
        bad_version[PAGE_MAGIC.len()] = 0x7f;
        assert!(matches!(
            Page::parse(&bad_version),
            Err(PageError::UnsupportedVersion(0x7f))
        ));

        assert!(matches!(
            Page::parse(&good[..good.len() - 3]),
            Err(PageError::Truncated)
        ));

        // Tamper the header node count so it disagrees with the block.
        let mut bad_count = good.clone();
        let n_nodes_off = PAGE_MAGIC.len() + 1 + 1 + PAGE_HEADER_RESERVED;
        bad_count[n_nodes_off] = 9;
        assert!(matches!(
            Page::parse(&bad_count),
            Err(PageError::NodeCountMismatch { .. })
        ));

        // Corrupt the first topology kind byte (just past the centroid block).
        let mut bad_kind = good.clone();
        let centroid_len_off = PAGE_MAGIC.len() + 1 + 1 + PAGE_HEADER_RESERVED + U32_BYTES * 2;
        let centroid_len = u32::from_le_bytes(
            bad_kind[centroid_len_off..centroid_len_off + U32_BYTES]
                .try_into()
                .expect("4-byte centroid_len field"),
        ) as usize;
        let topo_start = PAGE_HEADER_BYTES + centroid_len;
        bad_kind[topo_start] = 7;
        assert!(matches!(
            Page::parse(&bad_kind),
            Err(PageError::BadNodeKind(7))
        ));
    }

    #[test]
    fn rejects_norm_requiring_metric_without_norms() {
        // A page declaring L2Sq (which folds the per-centroid norm into the
        // distance) but carrying a norms-absent block would panic in the kernel
        // at score time. Parse must reject it instead. (NegDot needs no norms,
        // so a norms-absent NegDot page is accepted.)
        let dim = 2u32;
        let norms_absent = ClusterCentroids {
            n_cent: 1,
            dim,
            scale: vec![1.0, 1.0],
            offset: vec![0.0, 0.0],
            rows: vec![0u8; (dim as usize) * 2],
            norms: None,
            counts: vec![1],
            radii: Vec::new(),
        };
        let topo = vec![NodeTopo::Leaf(LeafRef {
            superfile_id: 1,
            doc_off: 0,
            count: 0,
        })];
        let l2 = encode_page(Metric::L2Sq, &norms_absent, &topo, 0);
        assert!(matches!(Page::parse(&l2), Err(PageError::MissingNorms)));
        // Same bytes under NegDot parse fine — that metric never reads a norm.
        let negdot = encode_page(Metric::NegDot, &norms_absent, &topo, 0);
        assert!(Page::parse(&negdot).is_ok());
    }
}
