// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A knowledge graph's adjacency as a resident index: the [`Hnsw`] graph
//! structure used single-level over an edge table's rows, in the HNSW
//! bundle's lifecycle — built by `optimize()`, published content-addressed,
//! memory-mapped when local, held in the table's resident slot, kept by GC
//! while the manifest references it.
//!
//! An edge table's rows are directed edges: an `Int64` source node, an
//! `Int64` destination node, and the source node's readable key. The
//! platform writes every edge both ways, so every node is a source and has a
//! key. Nodes are numbered by the rank of their id in ascending order, so a
//! seed id is found by binary search over the id section; each node's
//! base-layer neighbours are its edges' destinations, in row order.
//!
//! Payload layout, little-endian: [`ADJACENCY_MAGIC`]; node count `n` (u64);
//! the node ids (`n × i64`, ascending); key offsets (`(n + 1) × u64`) and
//! the keys' UTF-8 bytes; the graph section's length (u64) and the graph
//! ([`Hnsw::to_bytes`]). The id and key sections are served as zero-copy
//! slices of the mapped bundle; the graph decodes as every HNSW graph does.

use bytes::Bytes;

use crate::superfile::vector::hnsw::{Cursor, Hnsw};

/// On-disk magic for an adjacency payload.
const ADJACENCY_MAGIC: &[u8; 8] = b"INFKGA01";
/// Bytes of one node id.
const ID_BYTES: usize = 8;
/// Bytes of one key offset.
const OFFSET_BYTES: usize = 8;

/// The edges of a table gathered into per-node adjacency lists, ready to
/// encode.
#[derive(Debug, Default)]
pub(crate) struct AdjacencyBuild {
    /// Node ids, ascending; node `i` is `ids[i]`.
    ids: Vec<i64>,
    /// Node `i`'s key: the source key of the first row where it is the
    /// source, empty for a node that is only ever a destination.
    keys: Vec<String>,
    /// Node `i`'s destinations, in row order.
    lists: Vec<Vec<u32>>,
}

impl AdjacencyBuild {
    /// Gather `(source, destination, source key)` edges, in the order they
    /// are given.
    pub(crate) fn from_edges<'a>(edges: impl Iterator<Item = (i64, i64, Option<&'a str>)>) -> Self {
        let edges: Vec<(i64, i64, Option<&str>)> = edges.collect();
        let mut ids: Vec<i64> = edges.iter().flat_map(|&(s, d, _)| [s, d]).collect();
        ids.sort_unstable();
        ids.dedup();
        let node = |id: i64| ids.binary_search(&id).map(|i| i as u32);
        let mut keys = vec![String::new(); ids.len()];
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
        for (s, d, key) in edges {
            let (Ok(s), Ok(d)) = (node(s), node(d)) else {
                continue;
            };
            lists[s as usize].push(d);
            if let Some(key) = key
                && keys[s as usize].is_empty()
            {
                keys[s as usize] = key.to_string();
            }
        }
        Self { ids, keys, lists }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub(crate) fn node_count(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }

    /// The payload bytes: ids, keys and the single-level graph.
    pub(crate) fn encode(self) -> Vec<u8> {
        let n = self.ids.len();
        let graph = Hnsw::from_adjacency(self.lists).to_bytes();
        let key_bytes: usize = self.keys.iter().map(String::len).sum();
        let mut out = Vec::with_capacity(
            ADJACENCY_MAGIC.len()
                + 8
                + n * ID_BYTES
                + (n + 1) * OFFSET_BYTES
                + key_bytes
                + 8
                + graph.len(),
        );
        out.extend_from_slice(ADJACENCY_MAGIC);
        out.extend_from_slice(&(n as u64).to_le_bytes());
        for id in &self.ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        let mut offset = 0u64;
        out.extend_from_slice(&offset.to_le_bytes());
        for key in &self.keys {
            offset += key.len() as u64;
            out.extend_from_slice(&offset.to_le_bytes());
        }
        for key in &self.keys {
            out.extend_from_slice(key.as_bytes());
        }
        out.extend_from_slice(&(graph.len() as u64).to_le_bytes());
        out.extend_from_slice(&graph);
        out
    }
}

/// A decoded adjacency payload: the graph, and the id and key sections read
/// in place.
pub(crate) struct AdjacencyIndex {
    graph: Hnsw,
    /// `n × i64`, ascending.
    ids: Bytes,
    /// `(n + 1) × u64`.
    key_offsets: Bytes,
    key_bytes: Bytes,
    n: usize,
}

impl AdjacencyIndex {
    /// Read an [`AdjacencyBuild::encode`] payload. `None` on a bad magic,
    /// truncation, offsets out of order or a key that is not UTF-8, so a
    /// corrupt payload degrades to "no index" rather than a panic. `bundle`
    /// owns the bytes; the id and key sections are `slice_ref` views of it,
    /// so a mapped bundle is never copied.
    pub(crate) fn decode(bundle: &Bytes) -> Option<Self> {
        let bytes: &[u8] = bundle.as_ref();
        let mut c = Cursor::new(bytes);
        if c.take(ADJACENCY_MAGIC.len())? != ADJACENCY_MAGIC {
            return None;
        }
        let n = usize::try_from(c.u64()?).ok()?;
        // Bound every section against the bytes present before taking it, so
        // a corrupt count cannot drive a huge read.
        let ids = bundle.slice_ref(c.take(n.checked_mul(ID_BYTES)?)?);
        let key_offsets = bundle.slice_ref(c.take(n.checked_add(1)?.checked_mul(OFFSET_BYTES)?)?);
        let mut previous = 0u64;
        for i in 0..=n {
            let offset = offset_at(&key_offsets, i);
            if offset < previous {
                return None;
            }
            previous = offset;
        }
        let key_bytes = bundle.slice_ref(c.take(usize::try_from(previous).ok()?)?);
        let graph_len = usize::try_from(c.u64()?).ok()?;
        let graph = Hnsw::from_bytes(c.take(graph_len)?)?;
        if graph.len() != n {
            return None;
        }
        let index = Self {
            graph,
            ids,
            key_offsets,
            key_bytes,
            n,
        };
        for node in 0..n {
            std::str::from_utf8(index.raw_key(node)).ok()?;
        }
        let mut last: Option<i64> = None;
        for node in 0..n {
            let id = index.id(node as u32);
            if last.is_some_and(|l| l >= id) {
                return None;
            }
            last = Some(id);
        }
        Some(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    pub(crate) fn graph(&self) -> &Hnsw {
        &self.graph
    }

    /// Node `node`'s id (the edge table's `src` / `dst` value).
    pub(crate) fn id(&self, node: u32) -> i64 {
        let at = node as usize * ID_BYTES;
        let mut word = [0u8; ID_BYTES];
        word.copy_from_slice(&self.ids[at..at + ID_BYTES]);
        i64::from_le_bytes(word)
    }

    /// The node with id `id`, by binary search over the ascending id section.
    pub(crate) fn node_of(&self, id: i64) -> Option<u32> {
        let (mut lo, mut hi) = (0usize, self.n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.id(mid as u32).cmp(&id) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u32),
            }
        }
        None
    }

    fn raw_key(&self, node: usize) -> &[u8] {
        let start = offset_at(&self.key_offsets, node) as usize;
        let end = offset_at(&self.key_offsets, node + 1) as usize;
        &self.key_bytes[start..end]
    }

    /// Node `node`'s key; empty for a node no row names as a source.
    pub(crate) fn key(&self, node: u32) -> &str {
        // Checked once in `decode`.
        std::str::from_utf8(self.raw_key(node as usize)).unwrap_or_default()
    }

    /// Resident bytes beyond the mapping: the decoded graph's heap.
    pub(crate) fn resident_graph_bytes(&self) -> usize {
        self.graph.heap_bytes()
    }
}

/// Offset `i` of a key-offset section; the caller has checked the range.
fn offset_at(offsets: &Bytes, i: usize) -> u64 {
    let at = i * OFFSET_BYTES;
    let mut word = [0u8; OFFSET_BYTES];
    word.copy_from_slice(&offsets[at..at + OFFSET_BYTES]);
    u64::from_le_bytes(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(edges: &[(i64, i64)]) -> AdjacencyBuild {
        let keyed: Vec<(i64, i64, String)> = edges
            .iter()
            .map(|&(s, d)| (s, d, format!("n{s}")))
            .collect();
        AdjacencyBuild::from_edges(keyed.iter().map(|(s, d, k)| (*s, *d, Some(k.as_str()))))
    }

    /// Ids, keys and every node's edges in row order survive the round trip;
    /// a node is found by its id, and a destination-only node has no key.
    #[test]
    fn a_payload_reads_back_what_it_was_built_from() {
        let edges = [(7, 3), (7, 11), (3, 7), (11, 3), (-5, 7)];
        let index = AdjacencyIndex::decode(&Bytes::from(build(&edges).encode())).expect("decode");
        assert_eq!(index.len(), 4);
        let ids: Vec<i64> = (0..4).map(|n| index.id(n)).collect();
        assert_eq!(ids, vec![-5, 3, 7, 11], "ascending");
        let seven = index.node_of(7).expect("7");
        assert_eq!(index.key(seven), "n7");
        let targets: Vec<i64> = index
            .graph()
            .base_neighbors(seven)
            .iter()
            .map(|&n| index.id(n))
            .collect();
        assert_eq!(targets, vec![3, 11], "row order");
        assert_eq!(index.node_of(4), None);
        assert_eq!(index.key(index.node_of(-5).expect("-5")), "n-5");
    }

    /// A node that is only a destination is still a node, with no key and no
    /// edges.
    #[test]
    fn a_destination_only_node_has_no_key() {
        let index =
            AdjacencyIndex::decode(&Bytes::from(build(&[(1, 2)]).encode())).expect("decode");
        let two = index.node_of(2).expect("2");
        assert_eq!(index.key(two), "");
        assert!(index.graph().base_neighbors(two).is_empty());
    }

    /// A truncated, re-tagged or inconsistent payload is refused.
    #[test]
    fn a_damaged_payload_is_refused() {
        let blob = build(&[(1, 2), (2, 1)]).encode();
        assert!(AdjacencyIndex::decode(&Bytes::from(blob[..blob.len() - 3].to_vec())).is_none());
        let mut other = blob.clone();
        other[0] = b'X';
        assert!(AdjacencyIndex::decode(&Bytes::from(other)).is_none());
        let mut swapped = blob.clone();
        // Swap the two ids so the section is no longer ascending.
        let at = ADJACENCY_MAGIC.len() + 8;
        swapped[at..at + 8].copy_from_slice(&2i64.to_le_bytes());
        swapped[at + 8..at + 16].copy_from_slice(&1i64.to_le_bytes());
        assert!(AdjacencyIndex::decode(&Bytes::from(swapped)).is_none());
        assert!(AdjacencyIndex::decode(&Bytes::from(blob)).is_some());
    }
}
