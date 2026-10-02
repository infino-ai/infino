// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Walks over a table's resident adjacency index — the knowledge graph the
//! platform builds in an edge table and `optimize()` indexes like the HNSW
//! graph (`superfile::vector::adjacency`).
//!
//! A walk reads the index the same way a vector search reads its graph:
//! through the resident slot, which hydrates the published bundle once per
//! generation (memory-mapped when the store is local, single-flight across
//! concurrent first touches) and releases it when the manifest stops
//! referencing it. Seeds are node ids — the edge table's `src` / `dst`
//! values — and a hit carries the node's id and key, so the caller joins
//! back to its rows by either.

use crate::supertable::{SupertableReader, error::QueryError};

/// Probability a PageRank surfer jumps back to a seed at each step: the
/// standard 0.15, under which a node's score is dominated by paths of a few
/// hops — the neighbourhood a GraphRAG answer is drawn from.
pub const PAGERANK_RESTART: f64 = 0.15;

/// Power steps a ranking runs: the residual after them is
/// `(1 - PAGERANK_RESTART)^50`, about 3e-4 of the total score, well under
/// the gap that separates a node from its neighbours in a ranking.
pub const PAGERANK_ITERATIONS: usize = 50;

/// One node a walk reached.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHit {
    /// The node's id: the edge table's `src` / `dst` value.
    pub id: i64,
    /// The node's readable key, as the edge table's key column spells it.
    pub key: String,
    /// Fewest hops from a seed; the seeds are hop 0.
    pub hop: u32,
    /// The node's personalized-PageRank score from [`SupertableReader::graph_rank`];
    /// 0 from a plain walk.
    pub score: f64,
}

/// What a traversal starts from: the nodes with these ids, or these keys.
enum Seeds<'a> {
    Ids(&'a [i64]),
    Keys(&'a [&'a str]),
}

impl SupertableReader {
    test_visible! {
        /// Every node within `hops` edges of the nodes with ids `seeds`,
        /// nearest first, at most `limit`. Breadth-first, so each node is
        /// reported once with the fewest hops it takes; a seed the index does
        /// not hold is skipped. An error when the table has no published
        /// adjacency index: `optimize()` with an adjacency spec builds one.
        fn graph_walk(&self, seeds: &[i64], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            self.traverse(Seeds::Ids(seeds), hops, limit, false)
        }
    }

    test_visible! {
        /// The nodes within `hops` of `seeds`, ranked by personalized PageRank
        /// from the seeds over the subgraph those nodes span, at most `limit`,
        /// highest score first (ties by fewer hops, then id). A node many short
        /// paths from the seeds pass through scores high, so a hub's thousand
        /// neighbours do not outrank the few nodes the seeds share.
        fn graph_rank(&self, seeds: &[i64], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            self.traverse(Seeds::Ids(seeds), hops, limit, true)
        }
    }

    test_visible! {
        /// [`Self::graph_walk`] from the nodes keyed `seeds` — what a SQL
        /// statement seeds with, since it holds keys, not ids.
        fn graph_walk_keys(&self, seeds: &[&str], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            self.traverse(Seeds::Keys(seeds), hops, limit, false)
        }
    }

    test_visible! {
        /// [`Self::graph_rank`] from the nodes keyed `seeds`.
        fn graph_rank_keys(&self, seeds: &[&str], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            self.traverse(Seeds::Keys(seeds), hops, limit, true)
        }
    }

    /// Walk (or, with `rank`, rank) from `seeds` over the resident adjacency
    /// index, hydrating it through the resident slot first.
    fn traverse(
        &self,
        seeds: Seeds<'_>,
        hops: u32,
        limit: usize,
        rank: bool,
    ) -> Result<Vec<GraphHit>, QueryError> {
        let resident = self.block_on(self.resident_vector_index()).ok_or_else(|| {
            QueryError::Execute("no adjacency index is published for this table".into())
        })?;
        let Some(index) = resident.data.as_ref().and_then(|kind| kind.adjacency()) else {
            return Err(QueryError::Execute(
                "the table's resident index is not an adjacency index".into(),
            ));
        };
        let nodes: Vec<u32> = match seeds {
            Seeds::Ids(ids) => ids.iter().filter_map(|&id| index.node_of(id)).collect(),
            Seeds::Keys(keys) => index.nodes_of_keys(keys),
        };
        let hit = |node: u32, hop: u32, score: f64| GraphHit {
            id: index.id(node),
            key: index.key(node).to_string(),
            hop,
            score,
        };
        Ok(if rank {
            index
                .graph()
                .rank_base(&nodes, hops, limit, PAGERANK_RESTART, PAGERANK_ITERATIONS)
                .into_iter()
                .map(|(node, hop, score)| hit(node, hop, score))
                .collect()
        } else {
            index
                .graph()
                .walk_base(&nodes, hops, limit)
                .into_iter()
                .map(|(node, hop)| hit(node, hop, 0.0))
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use tempfile::TempDir;

    use crate::{
        config::{CompactionSettings, OptimizeOptions},
        runtime_bridge::bridge_sync_to_async,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{Supertable, options::SupertableOptions},
    };

    /// Leaves of the hub node in the ranking fixture.
    const LEAVES: i64 = 20;
    /// First leaf's id.
    const FIRST_LEAF: i64 = 100;
    /// How far the ranking's scores may sum away from one.
    const SCORE_TOLERANCE: f64 = 1e-6;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src", DataType::Int64, false),
            Field::new("dst", DataType::Int64, false),
            Field::new("key", DataType::LargeUtf8, false),
        ]))
    }

    /// Every edge of `edges` and its reverse, keyed `n<id>`.
    fn batch(edges: &[(i64, i64)]) -> RecordBatch {
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for &(s, d) in edges {
            src.extend([s, d]);
            dst.extend([d, s]);
        }
        let keys: Vec<String> = src.iter().map(|s| format!("n{s}")).collect();
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from(src)),
                Arc::new(Int64Array::from(dst)),
                Arc::new(LargeStringArray::from(keys)),
            ],
        )
        .expect("batch")
    }

    fn edge_table(dir: &TempDir) -> (Supertable, Arc<dyn StorageProvider>) {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(Arc::clone(&storage));
        (Supertable::create(options).expect("create"), storage)
    }

    fn append(table: &Supertable, edges: &[(i64, i64)]) {
        let mut writer = table.writer().expect("writer");
        writer.append(&batch(edges)).expect("append");
        writer.commit().expect("commit");
    }

    fn indexed() -> OptimizeOptions {
        OptimizeOptions::compact(CompactionSettings::default()).with_adjacency("src", "dst", "key")
    }

    fn walked(table: &Supertable, seed: i64, hops: u32) -> Vec<(i64, u32)> {
        table
            .reader()
            .expect("reader")
            .graph_walk(&[seed], hops, usize::MAX)
            .expect("walk")
            .into_iter()
            .map(|hit| (hit.id, hit.hop))
            .collect()
    }

    /// `optimize()` with the edge columns builds the adjacency, stamps it on
    /// the manifest and the walk reads it: each node once at its fewest
    /// hops, keys included. Optimizing again over the same rows publishes
    /// nothing; an append republishes a new generation that holds the new
    /// edge, and gc keeps the generation the manifest references.
    #[test]
    fn optimize_builds_the_adjacency_a_walk_reads_and_republishes_on_append() {
        let dir = TempDir::new().expect("tempdir");
        let (table, storage) = edge_table(&dir);
        append(&table, &[(1, 2), (2, 3), (3, 4), (2, 5)]);
        assert!(
            table
                .reader()
                .expect("reader")
                .graph_walk(&[1], 2, 10)
                .is_err(),
            "no index before optimize"
        );

        table.optimize(&indexed()).expect("optimize");
        let reader = table.reader().expect("reader");
        let first = reader
            .manifest()
            .resident_vector_index_blob()
            .cloned()
            .expect("stamped");
        assert_eq!(
            walked(&table, 1, 2),
            vec![(1, 0), (2, 1), (3, 2), (5, 2)],
            "nearest first, each node once"
        );
        let hits = reader.graph_walk(&[1], 1, usize::MAX).expect("walk");
        assert_eq!(hits[1].key, "n2");
        let by_key: Vec<(i64, u32)> = reader
            .graph_walk_keys(&["n1", "missing"], 1, usize::MAX)
            .expect("walk by key")
            .into_iter()
            .map(|hit| (hit.id, hit.hop))
            .collect();
        assert_eq!(
            by_key,
            vec![(1, 0), (2, 1)],
            "seeded by key, an unknown key skipped"
        );
        assert_eq!(
            walked(&table, 1, 3).len(),
            5,
            "the whole component in three hops"
        );
        assert_eq!(
            walked(&table, 1, 3).len(),
            5,
            "a second walk reuses the resident index"
        );

        let generation = table.manifest_id();
        table.optimize(&indexed()).expect("optimize again");
        assert_eq!(
            table.manifest_id(),
            generation,
            "same rows, nothing republished"
        );

        append(&table, &[(5, 6)]);
        table.optimize(&indexed()).expect("optimize after append");
        let second = table
            .reader()
            .expect("reader")
            .manifest()
            .resident_vector_index_blob()
            .cloned()
            .expect("restamped");
        assert_ne!(second.uri, first.uri, "a new generation");
        assert!(
            walked(&table, 1, 3).contains(&(6, 3)),
            "the appended edge is in the new generation"
        );

        table.gc(Duration::ZERO).expect("gc");
        bridge_sync_to_async(storage.head(&second.uri))
            .expect("the referenced generation survives gc");
        assert!(
            bridge_sync_to_async(storage.head(&first.uri)).is_err(),
            "the superseded generation is reclaimed"
        );
    }

    /// Seeds 1 and 2 both point at 3; seed 1 also points at hub 4, which
    /// points at twenty leaves. The ranking puts 3 above every leaf, and the
    /// scores sum to one.
    #[test]
    fn a_neighbour_the_seeds_share_outranks_a_hubs_leaves() {
        let dir = TempDir::new().expect("tempdir");
        let (table, _storage) = edge_table(&dir);
        let mut edges = vec![(1, 3), (2, 3), (1, 4)];
        edges.extend((FIRST_LEAF..FIRST_LEAF + LEAVES).map(|leaf| (4, leaf)));
        append(&table, &edges);
        table.optimize(&indexed()).expect("optimize");
        let reader = table.reader().expect("reader");
        let ranked = reader.graph_rank(&[1, 2], 2, usize::MAX).expect("rank");
        let position = |id: i64| ranked.iter().position(|hit| hit.id == id).expect("ranked");
        let best_leaf = (FIRST_LEAF..FIRST_LEAF + LEAVES)
            .map(position)
            .min()
            .expect("leaves");
        assert!(position(3) < best_leaf, "{ranked:?}");
        let total: f64 = ranked.iter().map(|hit| hit.score).sum();
        assert!(
            (total - 1.0).abs() < SCORE_TOLERANCE,
            "scores sum to one: {total}"
        );
        assert_eq!(reader.graph_rank(&[1, 2], 2, 3).expect("rank").len(), 3);
    }
}
