// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Walks over a table's resident adjacency index — the knowledge graph a
//! caller writes into an edge table and `optimize()` indexes like the HNSW
//! graph (`superfile::vector::adjacency`).
//!
//! A walk reads the index the same way a vector search reads its graph:
//! through the resident slot, which hydrates the published bundle once per
//! generation (fetched and decoded once, single-flight across concurrent
//! first touches) and releases it when the manifest stops referencing it.
//! A node is a row — a table's name and the row's stable `_id` — so seeds
//! are rows and what a walk reaches are rows, which the SQL functions
//! (`exec::graph_exec`) resolve to their columns as a search resolves its
//! hits. Edges are directed: a walk follows an edge from its source row to
//! its destination row only.
//!
//! The walk runs on the blocking pool, off the query runtime's workers,
//! and what it reaches is charged to the connection's memory budget for as
//! long as the rows are held; a statement over a neighbourhood the budget
//! cannot hold is refused as over budget, like a search that cannot hold
//! its hits.

use std::{mem::size_of, sync::Arc};

use tokio::task::spawn_blocking;

use crate::{
    memory::Reservation,
    supertable::{SupertableReader, error::QueryError},
};

/// Probability a PageRank surfer jumps back to a seed at each step: the
/// standard 0.15, under which a node's score is dominated by paths of a few
/// hops — the neighbourhood a GraphRAG answer is drawn from.
pub const PAGERANK_RESTART: f64 = 0.15;

/// Power steps a ranking runs: the residual after them is
/// `(1 - PAGERANK_RESTART)^50`, about 3e-4 of the total score, well under
/// the gap that separates a node from its neighbours in a ranking.
pub const PAGERANK_ITERATIONS: usize = 50;

/// One row a walk reached.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHit {
    /// The table the row belongs to.
    pub table: String,
    /// The row's stable `_id`.
    pub id: i128,
    /// Fewest hops from a seed; the seeds are hop 0.
    pub hop: u32,
    /// The row's personalized-PageRank score from [`SupertableReader::graph_rank`];
    /// 0 from a plain walk. The scores sum to one over every row the walk
    /// reached, `limit` or no `limit`.
    pub score: f64,
}

/// What a walk reached, held with the budget reservation that covers it:
/// the rows free their bytes when the last of them is dropped.
pub(crate) struct Reached {
    pub(crate) hits: Vec<GraphHit>,
    _reservation: Reservation,
}

impl SupertableReader {
    test_visible! {
        /// Every row within `hops` edges of the rows `seeds` (each a table
        /// name and an `_id`), nearest first, at most `limit`. Breadth-first,
        /// so each row is reported once with the fewest hops it takes; a seed
        /// the index does not hold is skipped. An error when the table has no
        /// published adjacency index: `optimize()` with an adjacency spec
        /// builds one.
        fn graph_walk(&self, seeds: &[(&str, i128)], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            Ok(self.block_on(self.traverse(seeds, hops, limit, false))?.hits)
        }
    }

    test_visible! {
        /// The rows within `hops` of `seeds`, ranked by personalized PageRank
        /// from the seeds over the subgraph those rows span, at most `limit`,
        /// highest score first (ties by fewer hops, then node). A row many
        /// short paths from the seeds pass through scores high, so a hub's
        /// thousand neighbours do not outrank the few rows the seeds share.
        /// The scores sum to one over the whole subgraph, so with a `limit`
        /// the rows returned sum to less.
        fn graph_rank(&self, seeds: &[(&str, i128)], hops: u32, limit: usize) -> Result<Vec<GraphHit>, QueryError> {
            Ok(self.block_on(self.traverse(seeds, hops, limit, true))?.hits)
        }
    }

    /// Walk (or, with `rank`, rank) from `seeds` over the resident adjacency
    /// index, hydrating it through the resident slot first. The async form
    /// the SQL functions run inside a plan's `execute`: the walk runs on the
    /// blocking pool, and the rows it reaches are reserved against the
    /// connection's memory budget for as long as the returned [`Reached`]
    /// lives.
    pub(crate) async fn traverse(
        &self,
        seeds: &[(&str, i128)],
        hops: u32,
        limit: usize,
        rank: bool,
    ) -> Result<Reached, QueryError> {
        let resident = self.resident_vector_index().await.ok_or_else(|| {
            QueryError::Internal("no adjacency index is published for this table".into())
        })?;
        let Some(index) = resident.data.as_ref().and_then(|kind| kind.adjacency()) else {
            return Err(QueryError::Internal(
                "the table's resident index is not an adjacency index".into(),
            ));
        };
        let nodes: Vec<u32> = seeds
            .iter()
            .filter_map(|&(table, id)| index.node_of(index.table_index(table)?, id))
            .collect();
        let budget = Arc::clone(&self.manifest().options.connection_memory_budget);
        let walked = Arc::clone(&resident);
        let hits = spawn_blocking(move || {
            // The adjacency was there a moment ago on the same `Arc`; the
            // match is for the compiler, not a path a walk takes.
            let Some(index) = walked.data.as_ref().and_then(|kind| kind.adjacency()) else {
                return Vec::new();
            };
            let hit = |node: u32, hop: u32, score: f64| {
                let row = index.node(node);
                GraphHit {
                    table: index.table_name(row.table).to_string(),
                    id: row.id,
                    hop,
                    score,
                }
            };
            if rank {
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
            }
        })
        .await
        .map_err(|e| QueryError::Internal(format!("graph walk task failed: {e}")))?;
        let bytes = hits.len() * size_of::<GraphHit>()
            + hits.iter().map(|hit| hit.table.len()).sum::<usize>();
        let reservation = budget
            .try_reserve(bytes)
            .map_err(|over| QueryError::OverBudget(format!("during graph walk, {over}")))?;
        Ok(Reached {
            hits,
            _reservation: reservation,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use arrow_array::{Decimal128Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        logical_expr::Expr,
        prelude::{col, lit},
        scalar::ScalarValue,
    };
    use tempfile::TempDir;

    use crate::{
        config::{CompactionSettings, OptimizeOptions},
        runtime_bridge::bridge_sync_to_async,
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{
            Supertable,
            options::SupertableOptions,
            schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        },
    };

    /// The one table every node of these fixtures is a row of.
    const ROWS: &str = "rows";
    /// Leaves of the hub node in the ranking fixture.
    const LEAVES: i128 = 20;
    /// First leaf's id.
    const FIRST_LEAF: i128 = 100;
    /// How far the ranking's scores may sum away from one.
    const SCORE_TOLERANCE: f64 = 1e-6;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("src_table", DataType::Utf8, false),
            Field::new(
                "src_id",
                DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
                false,
            ),
            Field::new("dst_table", DataType::Utf8, false),
            Field::new(
                "dst_id",
                DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
                false,
            ),
        ]))
    }

    fn ids(values: Vec<i128>) -> Arc<Decimal128Array> {
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
                .expect("ids"),
        )
    }

    /// Every edge of `edges` and its reverse, every node a row of [`ROWS`].
    fn batch(edges: &[(i128, i128)]) -> RecordBatch {
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for &(s, d) in edges {
            src.extend([s, d]);
            dst.extend([d, s]);
        }
        let tables = vec![ROWS; src.len()];
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(StringArray::from(tables.clone())),
                ids(src),
                Arc::new(StringArray::from(tables)),
                ids(dst),
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

    fn append(table: &Supertable, edges: &[(i128, i128)]) {
        let mut writer = table.writer().expect("writer");
        writer.append(&batch(edges)).expect("append");
        writer.commit().expect("commit");
    }

    fn indexed() -> OptimizeOptions {
        OptimizeOptions::compact(CompactionSettings::default()).with_adjacency(
            "src_table",
            "src_id",
            "dst_table",
            "dst_id",
        )
    }

    fn walked(table: &Supertable, seed: i128, hops: u32) -> Vec<(i128, u32)> {
        table
            .reader()
            .expect("reader")
            .graph_walk(&[(ROWS, seed)], hops, usize::MAX)
            .expect("walk")
            .into_iter()
            .map(|hit| (hit.id, hit.hop))
            .collect()
    }

    /// `optimize()` with the edge columns builds the adjacency, stamps it on
    /// the manifest and the walk reads it: each row once at its fewest hops,
    /// its table named. Optimizing again over the same rows publishes
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
                .graph_walk(&[(ROWS, 1)], 2, 10)
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
            "nearest first, each row once"
        );
        let hits = reader
            .graph_walk(&[(ROWS, 1), ("elsewhere", 1)], 1, usize::MAX)
            .expect("walk");
        assert_eq!(
            hits.len(),
            2,
            "a seed of a table the graph has no row of is skipped"
        );
        assert_eq!((hits[1].table.as_str(), hits[1].id), (ROWS, 2));
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

    /// An id as a SQL literal of the id column's type.
    fn id(value: i128) -> Expr {
        lit(ScalarValue::Decimal128(
            Some(value),
            DECIMAL128_PRECISION,
            DECIMAL128_SCALE,
        ))
    }

    /// The stamped resident-index blob's uri, when one is stamped.
    fn stamped(table: &Supertable) -> Option<String> {
        table
            .reader()
            .expect("reader")
            .manifest()
            .resident_vector_index_blob()
            .map(|reference| reference.uri.clone())
    }

    /// A delete writes a tombstone, which moves neither the row count nor
    /// the id range, so the index must key on the tombstone state too: the
    /// edges through 2 are deleted, the next optimize republishes, and the
    /// walk no longer crosses them. Another spec over the same rows is
    /// another graph and republishes as well. Deleting every edge leaves no
    /// index stamped, so a walk reports none rather than the old edges.
    #[test]
    fn deletes_and_another_spec_rebuild_and_no_edges_clears() {
        let dir = TempDir::new().expect("tempdir");
        let (table, _storage) = edge_table(&dir);
        append(&table, &[(1, 2), (2, 3), (3, 4)]);
        table.optimize(&indexed()).expect("optimize");
        let first = stamped(&table).expect("stamped");
        assert_eq!(walked(&table, 1, 3), vec![(1, 0), (2, 1), (3, 2), (4, 3)]);

        table
            .delete(col("src_id").eq(id(2)).or(col("dst_id").eq(id(2))))
            .expect("delete the edges through 2");
        table.optimize(&indexed()).expect("optimize after delete");
        let second = stamped(&table).expect("restamped");
        assert_ne!(second, first, "a delete republishes");
        assert_eq!(
            walked(&table, 3, 3),
            vec![(3, 0), (4, 1)],
            "the deleted edges are gone from the walk"
        );
        assert!(
            walked(&table, 1, 3).is_empty(),
            "a row no edge names is no node"
        );

        let reversed = OptimizeOptions::compact(CompactionSettings::default()).with_adjacency(
            "dst_table",
            "dst_id",
            "src_table",
            "src_id",
        );
        table
            .optimize(&reversed)
            .expect("optimize with the columns swapped");
        let third = stamped(&table).expect("restamped");
        assert_ne!(third, second, "another spec over the same rows republishes");
        assert_eq!(
            walked(&table, 3, 3),
            vec![(3, 0), (4, 1)],
            "both directions are written, so the reversed graph reads the same"
        );

        table
            .delete(col("src_id").gt(id(0)))
            .expect("delete every edge");
        table.optimize(&indexed()).expect("optimize with no edges");
        assert_eq!(stamped(&table), None, "no edges, no index");
        assert!(
            table
                .reader()
                .expect("reader")
                .graph_walk(&[(ROWS, 3)], 1, 10)
                .is_err(),
            "nothing to walk"
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
        let seeds = [(ROWS, 1), (ROWS, 2)];
        let ranked = reader.graph_rank(&seeds, 2, usize::MAX).expect("rank");
        let position = |id: i128| ranked.iter().position(|hit| hit.id == id).expect("ranked");
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
        assert_eq!(reader.graph_rank(&seeds, 2, 3).expect("rank").len(), 3);
    }
}
