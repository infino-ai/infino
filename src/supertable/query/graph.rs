// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A table of edges as an adjacency list in memory, for bounded walks.
//!
//! An edge table is an ordinary table whose rows are directed edges: an
//! integer source node, an integer destination node, and the source node's
//! readable key. A walk over it as a recursive CTE scans the whole table once
//! per hop; over this structure each hop touches only the neighbours it
//! expands. [`EdgeGraph`] is built from the table's rows once per manifest
//! snapshot and cached on the table handle (see
//! [`SupertableReader::graph_cache`](crate::supertable::handle::SupertableReader::graph_cache)),
//! so every walk on one snapshot reuses it and a commit invalidates it.
//!
//! Nodes are numbered densely in first-seen order; edges are kept in CSR
//! form (`offsets[i]..offsets[i + 1]` indexes node `i`'s targets). A node's
//! key is taken from the rows where it is the source, so a node that is only
//! ever a destination has no key: tables that record every edge both ways
//! (as the platform's key graph does) name every node.

use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray, StringViewArray};

use crate::supertable::manifest::ManifestSnapshot;

/// One table's walk index, and the snapshot and columns it was built from.
pub(crate) struct CachedEdgeGraph {
    pub(crate) manifest: Arc<ManifestSnapshot>,
    /// `(source, destination, key)` column names.
    pub(crate) columns: [String; 3],
    pub(crate) graph: Arc<EdgeGraph>,
}

/// Errors building an [`EdgeGraph`] from an edge table's rows.
#[derive(Debug, thiserror::Error)]
pub(crate) enum EdgeGraphError {
    /// A node column is not `Int64`, or a key column is not a string.
    #[error("edge column {column} must be {expected}")]
    ColumnType {
        column: &'static str,
        expected: &'static str,
    },
    /// The graph has more nodes or edges than a `u32` index can address.
    #[error("edge table too large for one walk index: {0}")]
    TooLarge(&'static str),
}

/// The edge table's rows as an adjacency list.
#[derive(Debug, Default)]
pub(crate) struct EdgeGraph {
    /// Node id → dense index.
    index: HashMap<i64, u32>,
    /// Key → dense index, for seeding a walk by key.
    by_key: HashMap<String, u32>,
    /// Dense index → node id.
    ids: Vec<i64>,
    /// Dense index → key; empty for a node that is never a source.
    keys: Vec<String>,
    /// CSR offsets, one past the node count.
    offsets: Vec<u32>,
    /// CSR targets, dense indices.
    targets: Vec<u32>,
}

/// One node a walk reached, and the fewest hops it took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reached {
    pub(crate) id: i64,
    pub(crate) key: String,
    pub(crate) hop: u32,
}

/// The strings of a key column, whichever string layout it has.
fn string_values(col: &dyn Array) -> Option<Vec<Option<&str>>> {
    let any = col.as_any();
    if let Some(a) = any.downcast_ref::<StringArray>() {
        Some(a.iter().collect())
    } else if let Some(a) = any.downcast_ref::<LargeStringArray>() {
        Some(a.iter().collect())
    } else {
        any.downcast_ref::<StringViewArray>()
            .map(|a| a.iter().collect())
    }
}

impl EdgeGraph {
    /// Build from batches of `(src, dst, key)` columns, in that order: the
    /// projection the walk's scan selects.
    pub(crate) fn from_batches(batches: &[RecordBatch]) -> Result<Self, EdgeGraphError> {
        let mut graph = Self::default();
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for batch in batches {
            let src = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or(EdgeGraphError::ColumnType {
                    column: "source",
                    expected: "Int64",
                })?;
            let dst = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or(EdgeGraphError::ColumnType {
                    column: "destination",
                    expected: "Int64",
                })?;
            let keys =
                string_values(batch.column(2).as_ref()).ok_or(EdgeGraphError::ColumnType {
                    column: "key",
                    expected: "a string",
                })?;
            for (row, key) in keys.into_iter().enumerate() {
                if src.is_null(row) || dst.is_null(row) {
                    continue;
                }
                let s = graph.node(src.value(row))?;
                let d = graph.node(dst.value(row))?;
                if let Some(key) = key
                    && graph.keys[s as usize].is_empty()
                {
                    graph.keys[s as usize] = key.to_string();
                    graph.by_key.insert(key.to_string(), s);
                }
                edges.push((s, d));
            }
        }
        let n = graph.ids.len();
        if u32::try_from(edges.len()).is_err() {
            return Err(EdgeGraphError::TooLarge("edges"));
        }
        let mut offsets = vec![0u32; n + 1];
        for &(s, _) in &edges {
            offsets[s as usize + 1] += 1;
        }
        for i in 0..n {
            offsets[i + 1] += offsets[i];
        }
        let mut fill = offsets.clone();
        let mut targets = vec![0u32; edges.len()];
        for &(s, d) in &edges {
            targets[fill[s as usize] as usize] = d;
            fill[s as usize] += 1;
        }
        graph.offsets = offsets;
        graph.targets = targets;
        Ok(graph)
    }

    /// The dense index of node `id`, assigning the next one on first sight.
    fn node(&mut self, id: i64) -> Result<u32, EdgeGraphError> {
        if let Some(&i) = self.index.get(&id) {
            return Ok(i);
        }
        let i = u32::try_from(self.ids.len()).map_err(|_| EdgeGraphError::TooLarge("nodes"))?;
        self.index.insert(id, i);
        self.ids.push(id);
        self.keys.push(String::new());
        Ok(i)
    }

    /// The dense index of the node keyed `key`.
    pub(crate) fn find_key(&self, key: &str) -> Option<u32> {
        self.by_key.get(key).copied()
    }

    /// Every node within `hops` of `seeds`, nearest first, at most `limit`:
    /// a breadth-first walk, so each node is reported once with the fewest
    /// hops it takes. The seeds themselves are hop 0. Ties within a hop keep
    /// the order the walk reached them, which follows the edge table's row
    /// order and is stable for one snapshot.
    pub(crate) fn walk(&self, seeds: &[u32], hops: u32, limit: usize) -> Vec<Reached> {
        let mut hop_of: Vec<u32> = vec![u32::MAX; self.ids.len()];
        let mut queue: VecDeque<u32> = VecDeque::new();
        let mut out: Vec<Reached> = Vec::new();
        for &seed in seeds {
            if (seed as usize) < hop_of.len() && hop_of[seed as usize] == u32::MAX {
                hop_of[seed as usize] = 0;
                queue.push_back(seed);
            }
        }
        while let Some(u) = queue.pop_front() {
            if out.len() >= limit {
                break;
            }
            let hop = hop_of[u as usize];
            out.push(Reached {
                id: self.ids[u as usize],
                key: self.keys[u as usize].clone(),
                hop,
            });
            if hop >= hops {
                continue;
            }
            let (a, b) = (
                self.offsets[u as usize] as usize,
                self.offsets[u as usize + 1] as usize,
            );
            for &v in &self.targets[a..b] {
                if hop_of[v as usize] == u32::MAX {
                    hop_of[v as usize] = hop + 1;
                    queue.push_back(v);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use arrow_schema::{DataType, Field, Schema};
    use proptest::prelude::*;

    use super::*;

    /// Nodes in the generated graphs; small, so random edges make cycles.
    const PROP_NODES: i64 = 40;
    /// Edges per generated graph, at most.
    const PROP_MAX_EDGES: usize = 120;
    /// Hops a generated walk may take, at most.
    const PROP_MAX_HOPS: u32 = 5;

    fn batch(edges: &[(i64, i64)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("key", DataType::Utf8, false),
        ]));
        let (src, dst): (Vec<i64>, Vec<i64>) = edges.iter().copied().unzip();
        let keys: Vec<String> = src.iter().map(|s| format!("n{s}")).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(src)),
                Arc::new(Int64Array::from(dst)),
                Arc::new(StringArray::from(keys)),
            ],
        )
        .expect("batch")
    }

    /// The textbook answer: repeated relaxation over the edge list until
    /// nothing moves, each node's fewest hops from any seed.
    fn oracle(edges: &[(i64, i64)], seeds: &[i64], hops: u32) -> BTreeMap<i64, u32> {
        let mut best: BTreeMap<i64, u32> = seeds.iter().map(|&s| (s, 0)).collect();
        loop {
            let mut moved = false;
            for &(s, d) in edges {
                if let Some(&h) = best.get(&s)
                    && h < hops
                    && best.get(&d).is_none_or(|&cur| h + 1 < cur)
                {
                    best.insert(d, h + 1);
                    moved = true;
                }
            }
            if !moved {
                return best;
            }
        }
    }

    #[test]
    fn a_walk_reports_each_node_once_at_its_fewest_hops() {
        // 1→2→3→4, 2→5, 4→1: the cycle back to the seed does not re-report it.
        let edges = [(1, 2), (2, 3), (3, 4), (2, 5), (4, 1)];
        let graph = EdgeGraph::from_batches(&[batch(&edges)]).expect("graph");
        let seed = graph.find_key("n1").expect("n1");
        let reached: Vec<(i64, u32)> = graph
            .walk(&[seed], 3, usize::MAX)
            .into_iter()
            .map(|r| (r.id, r.hop))
            .collect();
        assert_eq!(reached, vec![(1, 0), (2, 1), (3, 2), (5, 2), (4, 3)]);
        assert_eq!(
            graph.walk(&[seed], 3, 2).len(),
            2,
            "the limit cuts the nearest first"
        );
    }

    proptest! {
        /// Against the relaxation oracle on random graphs with cycles,
        /// self-loops and repeated edges: the same nodes, the same hops.
        #[test]
        fn a_walk_matches_the_oracle(
            edges in prop::collection::vec((0..PROP_NODES, 0..PROP_NODES), 1..PROP_MAX_EDGES),
            seed_picks in prop::collection::vec(any::<prop::sample::Index>(), 1..4),
            hops in 0..=PROP_MAX_HOPS,
        ) {
            let graph = EdgeGraph::from_batches(&[batch(&edges)]).expect("graph");
            let seeds: Vec<i64> = seed_picks.iter().map(|p| edges[p.index(edges.len())].0).collect();
            let dense: Vec<u32> = seeds
                .iter()
                .map(|s| graph.find_key(&format!("n{s}")).expect("a source has a key"))
                .collect();
            let got: BTreeMap<i64, u32> = graph
                .walk(&dense, hops, usize::MAX)
                .into_iter()
                .map(|r| (r.id, r.hop))
                .collect();
            prop_assert_eq!(got, oracle(&edges, &seeds, hops));
        }
    }
}
