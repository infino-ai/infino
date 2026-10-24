// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A knowledge graph's adjacency as a resident index: the [`Hnsw`] graph
//! structure used single-level over an edge table's rows, in the HNSW
//! bundle's lifecycle — built by `optimize()`, published content-addressed,
//! fetched once per generation into the table's resident slot, kept by GC
//! while the manifest references it. The node sections are read in place
//! from the fetched bytes; the graph itself is decoded onto the heap, one
//! neighbour list per node, so a resident adjacency takes more memory than
//! its payload.
//!
//! A node is a row of some table: the pair of the table's name and the
//! row's stable `_id`, the same identity every search hit carries
//! (`SuperfileHit::stable_id`), so a node a walk reaches resolves to its row
//! through the engine's own placement lookup, with nothing of its own to
//! parse or hash. An edge table's rows are directed edges: the source row's
//! table and `_id`, the destination row's table and `_id`. Nodes are
//! numbered by the rank of `(table, _id)` in ascending order, so a seed is
//! found by binary search; each node's base-layer neighbours are its edges'
//! destinations, in row order.
//!
//! Payload layout, little-endian: [`ADJACENCY_MAGIC`]; the table count
//! (u64), the table-name offsets (`(t + 1) × u64`) and the names' UTF-8
//! bytes; the node count `n` (u64); each node's table index (`n × u32`) and
//! its `_id` (`n × i128`), ascending by (table index, id); the graph
//! section's length (u64) and the graph ([`Hnsw::to_bytes`]). The node
//! sections are served as zero-copy slices of the mapped bundle; the graph
//! decodes as every HNSW graph does.

use bytes::Bytes;

use crate::superfile::vector::hnsw::{Cursor, Hnsw};

/// On-disk magic for an adjacency payload.
const ADJACENCY_MAGIC: &[u8; 8] = b"INFKGA02";
/// Bytes of one node's table index.
const TABLE_INDEX_BYTES: usize = 4;
/// Bytes of one node id.
const ID_BYTES: usize = 16;
/// Bytes of one table-name offset.
const OFFSET_BYTES: usize = 8;

/// A node's identity: the row `id` of the table numbered `table` in the
/// payload's table list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct NodeRow {
    pub(crate) table: u32,
    pub(crate) id: i128,
}

/// The edges of a table gathered into per-node adjacency lists, ready to
/// encode.
#[derive(Debug, Default)]
pub(crate) struct AdjacencyBuild {
    /// The table names, in first-seen order; a node's `table` indexes here.
    tables: Vec<String>,
    /// The nodes, ascending; node `i` is `nodes[i]`.
    nodes: Vec<NodeRow>,
    /// Node `i`'s destinations, in row order.
    lists: Vec<Vec<u32>>,
}

impl AdjacencyBuild {
    /// Gather `((source table, source id), (destination table, destination
    /// id))` edges, in the order they are given.
    pub(crate) fn from_edges<'a>(
        edges: impl Iterator<Item = ((&'a str, i128), (&'a str, i128))>,
    ) -> Self {
        let mut tables: Vec<String> = Vec::new();
        let mut table_index = |name: &str| -> u32 {
            match tables.iter().position(|t| t == name) {
                Some(i) => i as u32,
                None => {
                    tables.push(name.to_string());
                    (tables.len() - 1) as u32
                }
            }
        };
        let edges: Vec<(NodeRow, NodeRow)> = edges
            .map(|((st, si), (dt, di))| {
                (
                    NodeRow {
                        table: table_index(st),
                        id: si,
                    },
                    NodeRow {
                        table: table_index(dt),
                        id: di,
                    },
                )
            })
            .collect();
        let mut nodes: Vec<NodeRow> = edges.iter().flat_map(|&(s, d)| [s, d]).collect();
        nodes.sort_unstable();
        nodes.dedup();
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nodes.len()];
        for (s, d) in edges {
            let (Ok(s), Ok(d)) = (nodes.binary_search(&s), nodes.binary_search(&d)) else {
                continue;
            };
            lists[s].push(d as u32);
        }
        Self {
            tables,
            nodes,
            lists,
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }

    /// The payload bytes: tables, nodes and the single-level graph.
    pub(crate) fn encode(self) -> Vec<u8> {
        let n = self.nodes.len();
        let graph = Hnsw::from_adjacency(self.lists).to_bytes();
        let name_bytes: usize = self.tables.iter().map(String::len).sum();
        let mut out = Vec::with_capacity(
            ADJACENCY_MAGIC.len()
                + 8
                + (self.tables.len() + 1) * OFFSET_BYTES
                + name_bytes
                + 8
                + n * (TABLE_INDEX_BYTES + ID_BYTES)
                + 8
                + graph.len(),
        );
        out.extend_from_slice(ADJACENCY_MAGIC);
        out.extend_from_slice(&(self.tables.len() as u64).to_le_bytes());
        let mut offset = 0u64;
        out.extend_from_slice(&offset.to_le_bytes());
        for name in &self.tables {
            offset += name.len() as u64;
            out.extend_from_slice(&offset.to_le_bytes());
        }
        for name in &self.tables {
            out.extend_from_slice(name.as_bytes());
        }
        out.extend_from_slice(&(n as u64).to_le_bytes());
        for node in &self.nodes {
            out.extend_from_slice(&node.table.to_le_bytes());
        }
        for node in &self.nodes {
            out.extend_from_slice(&node.id.to_le_bytes());
        }
        out.extend_from_slice(&(graph.len() as u64).to_le_bytes());
        out.extend_from_slice(&graph);
        out
    }
}

/// A decoded adjacency payload: the graph, the table names, and the node
/// sections read in place.
pub(crate) struct AdjacencyIndex {
    graph: Hnsw,
    tables: Vec<String>,
    /// `n × u32`.
    node_tables: Bytes,
    /// `n × i128`.
    node_ids: Bytes,
    n: usize,
}

impl AdjacencyIndex {
    /// Read an [`AdjacencyBuild::encode`] payload. `None` on a bad magic,
    /// truncation, offsets out of order, a name that is not UTF-8, a node
    /// naming no table, or nodes out of order, so a corrupt payload degrades
    /// to "no index" rather than a panic. `bundle` owns the bytes; the node
    /// sections are `slice_ref` views of it, so a mapped bundle is never
    /// copied.
    pub(crate) fn decode(bundle: &Bytes) -> Option<Self> {
        let bytes: &[u8] = bundle.as_ref();
        let mut c = Cursor::new(bytes);
        if c.take(ADJACENCY_MAGIC.len())? != ADJACENCY_MAGIC {
            return None;
        }
        // Bound every section against the bytes present before taking it, so
        // a corrupt count cannot drive a huge read.
        let t = usize::try_from(c.u64()?).ok()?;
        let offsets = c.take(t.checked_add(1)?.checked_mul(OFFSET_BYTES)?)?;
        let mut bounds = Vec::with_capacity(t + 1);
        let mut previous = 0u64;
        for i in 0..=t {
            let offset = u64_at(offsets, i * OFFSET_BYTES);
            if offset < previous {
                return None;
            }
            bounds.push(usize::try_from(offset).ok()?);
            previous = offset;
        }
        let names = c.take(bounds[t])?;
        let mut tables = Vec::with_capacity(t);
        for i in 0..t {
            tables.push(
                std::str::from_utf8(&names[bounds[i]..bounds[i + 1]])
                    .ok()?
                    .to_string(),
            );
        }
        let n = usize::try_from(c.u64()?).ok()?;
        let node_tables = bundle.slice_ref(c.take(n.checked_mul(TABLE_INDEX_BYTES)?)?);
        let node_ids = bundle.slice_ref(c.take(n.checked_mul(ID_BYTES)?)?);
        let graph_len = usize::try_from(c.u64()?).ok()?;
        let graph = Hnsw::from_bytes(c.take(graph_len)?)?;
        if graph.len() != n {
            return None;
        }
        let index = Self {
            graph,
            tables,
            node_tables,
            node_ids,
            n,
        };
        let mut last: Option<NodeRow> = None;
        for node in 0..n {
            let row = index.node(node as u32);
            if row.table as usize >= t || last.is_some_and(|l| l >= row) {
                return None;
            }
            last = Some(row);
        }
        Some(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    pub(crate) fn graph(&self) -> &Hnsw {
        &self.graph
    }

    /// Node `node`'s identity.
    pub(crate) fn node(&self, node: u32) -> NodeRow {
        let at = node as usize;
        let mut table = [0u8; TABLE_INDEX_BYTES];
        table.copy_from_slice(
            &self.node_tables[at * TABLE_INDEX_BYTES..(at + 1) * TABLE_INDEX_BYTES],
        );
        let mut id = [0u8; ID_BYTES];
        id.copy_from_slice(&self.node_ids[at * ID_BYTES..(at + 1) * ID_BYTES]);
        NodeRow {
            table: u32::from_le_bytes(table),
            id: i128::from_le_bytes(id),
        }
    }

    /// The name of the table numbered `table`.
    pub(crate) fn table_name(&self, table: u32) -> &str {
        &self.tables[table as usize]
    }

    /// The index of the table named `name`, when any node is its row.
    pub(crate) fn table_index(&self, name: &str) -> Option<u32> {
        self.tables.iter().position(|t| t == name).map(|i| i as u32)
    }

    /// The node that is row `id` of table `table`, by binary search over the
    /// ascending node section.
    pub(crate) fn node_of(&self, table: u32, id: i128) -> Option<u32> {
        let wanted = NodeRow { table, id };
        let (mut lo, mut hi) = (0usize, self.n);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.node(mid as u32).cmp(&wanted) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u32),
            }
        }
        None
    }

    /// Resident bytes beyond the mapping: the decoded graph's heap and the
    /// table names.
    pub(crate) fn resident_graph_bytes(&self) -> usize {
        self.graph.heap_bytes() + self.tables.iter().map(String::len).sum::<usize>()
    }
}

/// The little-endian u64 at byte `at` of `bytes`; the caller has checked
/// the range.
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let mut word = [0u8; OFFSET_BYTES];
    word.copy_from_slice(&bytes[at..at + OFFSET_BYTES]);
    u64::from_le_bytes(word)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(edges: &[((&str, i128), (&str, i128))]) -> AdjacencyBuild {
        AdjacencyBuild::from_edges(edges.iter().copied())
    }

    /// Tables, node identities and every node's edges in row order survive
    /// the round trip; a node is found by its table and id.
    #[test]
    fn a_payload_reads_back_what_it_was_built_from() {
        let edges = [
            (("issues", 7), ("logs", 3)),
            (("issues", 7), ("values", 11)),
            (("logs", 3), ("issues", 7)),
            (("values", 11), ("logs", 3)),
            (("issues", -5), ("issues", 7)),
        ];
        let index = AdjacencyIndex::decode(&Bytes::from(build(&edges).encode())).expect("decode");
        assert_eq!(index.len(), 4);
        let issues = index.table_index("issues").expect("issues");
        let seven = index.node_of(issues, 7).expect("issues/7");
        let targets: Vec<(&str, i128)> = index
            .graph()
            .base_neighbors(seven)
            .iter()
            .map(|&n| {
                let row = index.node(n);
                (index.table_name(row.table), row.id)
            })
            .collect();
        assert_eq!(targets, vec![("logs", 3), ("values", 11)], "row order");
        assert_eq!(index.node_of(issues, 4), None);
        assert_eq!(index.table_index("code"), None);
        let minus = index.node_of(issues, -5).expect("issues/-5");
        assert_eq!(
            index.node(minus),
            NodeRow {
                table: issues,
                id: -5
            }
        );
        assert!(minus < seven, "ascending by (table, id)");
    }

    /// A node that is only a destination is still a node, with no edges.
    #[test]
    fn a_destination_only_node_has_no_edges() {
        let index = AdjacencyIndex::decode(&Bytes::from(build(&[(("a", 1), ("b", 2))]).encode()))
            .expect("decode");
        let b = index.table_index("b").expect("b");
        let two = index.node_of(b, 2).expect("b/2");
        assert!(index.graph().base_neighbors(two).is_empty());
    }

    /// A truncated, re-tagged or inconsistent payload is refused.
    #[test]
    fn a_damaged_payload_is_refused() {
        let blob = build(&[(("a", 1), ("a", 2)), (("a", 2), ("a", 1))]).encode();
        assert!(AdjacencyIndex::decode(&Bytes::from(blob[..blob.len() - 3].to_vec())).is_none());
        let mut other = blob.clone();
        other[0] = b'X';
        assert!(AdjacencyIndex::decode(&Bytes::from(other)).is_none());
        // Swap the two node ids so the node section is no longer ascending:
        // the ids follow the magic, the table count, two offsets, the one
        // name, the node count and two table indices.
        let at = ADJACENCY_MAGIC.len() + 8 + 2 * OFFSET_BYTES + 1 + 8 + 2 * TABLE_INDEX_BYTES;
        let mut swapped = blob.clone();
        swapped[at..at + ID_BYTES].copy_from_slice(&2i128.to_le_bytes());
        swapped[at + ID_BYTES..at + 2 * ID_BYTES].copy_from_slice(&1i128.to_le_bytes());
        assert!(AdjacencyIndex::decode(&Bytes::from(swapped)).is_none());
        assert!(AdjacencyIndex::decode(&Bytes::from(blob)).is_some());
    }
}
