// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Graph walks as DataFusion table-valued functions.
//!
//! `graph_walk(table, seed_table, seed_ids, hops, k)` and
//! `graph_rank(table, seed_table, seed_ids, hops, k)` register via
//! `register_udtf` over an edge table's resident adjacency index (see
//! `supertable::query::graph`) and return rows of `table`: the ones within
//! `hops` edges of the rows `seed_ids` of `seed_table`, at most `k`. The
//! output is a search function's output — `table`'s `_id` and scalar
//! columns plus `score` — because a node is a row and a row a walk reaches
//! is resolved the way a search hit is: by its stable `_id`, through the
//! engine's placement lookup and [`resolve_hits`]. Nothing is parsed or
//! joined by hand:
//!
//! ```sql
//! SELECT title, score FROM graph_walk('issues', 'logs', [42, 43], 2, 50)
//! ```
//!
//! `graph_walk` lists the rows nearest first and its `score` is the hop
//! count (smaller is nearer, as a vector distance is); `graph_rank` orders
//! them by personalized PageRank from the seeds, its `score` that rank
//! (larger is better), so a hub's many leaves do not outrank what the seeds
//! share. The walk itself runs inside the plan's `execute`, like the search
//! kernels, so the index hydrates and the rows resolve on the query runtime.

use std::{fmt, sync::Arc};

use arrow::compute::cast;
use arrow_array::{Array, Decimal128Array, ListArray};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    logical_expr::{Expr, TableType},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
    scalar::ScalarValue,
};
use futures::stream;

use crate::{
    supertable::{
        handle::{SupertableReader, WeakReader},
        options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
        query::{
            SuperfileHit,
            exec::common::{
                arg_to_string, arg_to_usize, output_schema_with_score, resolve_hits,
                search_query_df_error, traced_tvf,
            },
            vector::lookup_user_placements_by_id_opt,
        },
    },
    utils::trace::detail_span,
};

/// SQL name of the breadth-first walk.
pub(crate) const GRAPH_WALK_UDTF: &str = "graph_walk";
/// SQL name of the ranked walk.
pub(crate) const GRAPH_RANK_UDTF: &str = "graph_rank";
/// Argument count after the catalog adapter has taken the table name:
/// `(seed_table, seed_ids, hops, k)`.
const GRAPH_ARG_COUNT: usize = 4;

/// Which traversal a function runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Traversal {
    /// Nearest first; `score` is the hop count.
    Walk,
    /// Highest personalized-PageRank score first.
    Rank,
}

impl Traversal {
    /// The SQL name the traversal is registered under.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Traversal::Walk => GRAPH_WALK_UDTF,
            Traversal::Rank => GRAPH_RANK_UDTF,
        }
    }
}

/// `TableFunctionImpl` for `graph_walk` / `graph_rank`, bound to the edge
/// table's reader and the target table's pinned reader and scalar schema.
/// `call_with_args` parses `(seed_table, seed_ids, hops, k)` and hands back
/// a per-invocation [`GraphTable`].
#[derive(Debug)]
pub(crate) struct GraphFunc {
    graph: WeakReader,
    target: WeakReader,
    target_name: String,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    traversal: Traversal,
}

impl GraphFunc {
    pub(crate) fn new(
        graph: Arc<SupertableReader>,
        target: Arc<SupertableReader>,
        target_name: String,
        scalar_schema: SchemaRef,
        traversal: Traversal,
    ) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            graph: WeakReader::from_reader(&graph),
            target: WeakReader::from_reader(&target),
            target_name,
            scalar_schema,
            output_schema,
            traversal,
        }
    }
}

impl TableFunctionImpl for GraphFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let name = self.traversal.name();
        let args = args.exprs();
        if args.len() != GRAPH_ARG_COUNT {
            return Err(DataFusionError::Plan(format!(
                "{name} expects {} arguments (table, seed_table, seed_ids, hops, k), got {}",
                GRAPH_ARG_COUNT + 1,
                args.len() + 1
            )));
        }
        let seed_table = arg_to_string(&args[0], &format!("{name} seed_table"))?;
        let seed_ids = arg_to_ids(&args[1], &format!("{name} seed_ids"))?;
        let hops = arg_to_usize(&args[2], &format!("{name} hops"))?;
        let hops = u32::try_from(hops)
            .map_err(|_| DataFusionError::Plan(format!("{name} hops must fit in 32 bits")))?;
        let k = arg_to_usize(&args[3], &format!("{name} k"))?;
        let dropped = || {
            DataFusionError::Execution(format!(
                "{name}: supertable consumer dropped before execution"
            ))
        };
        let graph = self.graph.upgrade().ok_or_else(dropped)?;
        let target = self.target.upgrade().ok_or_else(dropped)?;
        Ok(Arc::new(GraphTable {
            graph,
            target,
            target_name: self.target_name.clone(),
            seed_table,
            seed_ids,
            hops,
            k,
            traversal: self.traversal,
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
    }
}

/// One parsed invocation as a `TableProvider`. `scan` lowers to
/// [`GraphExec`].
struct GraphTable {
    graph: Arc<SupertableReader>,
    target: Arc<SupertableReader>,
    target_name: String,
    seed_table: String,
    seed_ids: Vec<i128>,
    hops: u32,
    k: usize,
    traversal: Traversal,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for GraphTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphTable")
            .field("table", &self.target_name)
            .field("seed_table", &self.seed_table)
            .field("seeds", &self.seed_ids.len())
            .field("hops", &self.hops)
            .field("k", &self.k)
            .field("traversal", &self.traversal)
            .finish()
    }
}

#[async_trait]
impl TableProvider for GraphTable {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let projected_schema = match projection {
            Some(indices) => Arc::new(
                self.output_schema
                    .project(indices)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            None => Arc::clone(&self.output_schema),
        };
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Arc::new(GraphExec {
            graph: Arc::clone(&self.graph),
            target: Arc::clone(&self.target),
            target_name: self.target_name.clone(),
            seed_table: self.seed_table.clone(),
            seed_ids: self.seed_ids.clone(),
            hops: self.hops,
            k: self.k,
            traversal: self.traversal,
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
            projection: projection.cloned(),
            projected_schema,
            cache,
        }))
    }
}

/// Custom `ExecutionPlan` that walks the graph inside `execute()`, keeps
/// the rows of the target table, and emits them resolved like search hits:
/// `_id` + projected scalar columns + `score`.
struct GraphExec {
    graph: Arc<SupertableReader>,
    target: Arc<SupertableReader>,
    target_name: String,
    seed_table: String,
    seed_ids: Vec<i128>,
    hops: u32,
    k: usize,
    traversal: Traversal,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl GraphExec {
    fn describe(&self) -> String {
        format!(
            "GraphExec: {}, table={}, seed_table={}, seeds={}, hops={}, k={}",
            self.traversal.name(),
            self.target_name,
            self.seed_table,
            self.seed_ids.len(),
            self.hops,
            self.k
        )
    }
}

impl fmt::Debug for GraphExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for GraphExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for GraphExec {
    fn name(&self) -> &'static str {
        "GraphExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "GraphExec has a single partition; asked for {partition}"
            )));
        }
        let graph = Arc::clone(&self.graph);
        let target = Arc::clone(&self.target);
        let target_name = self.target_name.clone();
        let seed_table = self.seed_table.clone();
        let seed_ids = self.seed_ids.clone();
        let (hops, k, traversal) = (self.hops, self.k, self.traversal);
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            let seeds: Vec<(&str, i128)> = seed_ids
                .iter()
                .map(|&id| (seed_table.as_str(), id))
                .collect();
            // The whole neighbourhood, then the target table's rows of it: a
            // walk bounded at `k` before filtering would stop short of them.
            let reached = graph
                .traverse(&seeds, hops, usize::MAX, traversal == Traversal::Rank)
                .await
                .map_err(search_query_df_error)?;
            let (ids, scores): (Vec<i128>, Vec<f32>) = reached
                .into_iter()
                .filter(|hit| hit.table == target_name)
                .take(k)
                .map(|hit| {
                    let score = match traversal {
                        Traversal::Walk => hit.hop as f32,
                        Traversal::Rank => hit.score as f32,
                    };
                    (hit.id, score)
                })
                .unzip();
            // Rows the edge table names that are gone since are skipped, as
            // a search skips a deleted row.
            let placements =
                lookup_user_placements_by_id_opt(target.manifest(), &ids, &target.op_stats)
                    .await
                    .map_err(search_query_df_error)?;
            let hits: Vec<SuperfileHit> = placements
                .into_iter()
                .zip(ids.iter().zip(scores))
                .filter_map(|(placement, (&id, score))| {
                    let (entry, local_doc_id) = placement?;
                    Some(SuperfileHit {
                        superfile: entry.uri,
                        local_doc_id,
                        score,
                        stable_id: Some(id),
                    })
                })
                .collect();
            resolve_hits(
                &target,
                &hits,
                &scalar_schema,
                &output_schema,
                projection.as_deref(),
            )
            .await
        };

        let span = match self.traversal {
            Traversal::Walk => detail_span!("tvf.graph_walk", rows_out = tracing::field::Empty),
            Traversal::Rank => detail_span!("tvf.graph_rank", rows_out = tracing::field::Empty),
        };
        let fut = traced_tvf(span, fut);
        let stream = stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// The engine's id type, which every `_id` column has.
fn id_type() -> DataType {
    DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE)
}

/// Extract one `_id` or an array literal of them (`[42, 43]`, which the
/// planner const-folds to a `List` scalar, or leaves as `make_array(...)`
/// when an element is not a plain literal). Integer literals and decimal
/// literals (an `_id` past `i64` parses as one) are both ids.
pub(crate) fn arg_to_ids(expr: &Expr, what: &str) -> DfResult<Vec<i128>> {
    match expr {
        Expr::Literal(ScalarValue::List(list), _) => list_literal_to_ids(list, what),
        Expr::ScalarFunction(sf) if sf.func.name() == "make_array" => sf
            .args
            .iter()
            .map(|arg| scalar_expr_to_id(arg, what))
            .collect(),
        other => Ok(vec![scalar_expr_to_id(other, what)?]),
    }
}

/// One id literal.
fn scalar_expr_to_id(expr: &Expr, what: &str) -> DfResult<i128> {
    let Expr::Literal(value, _) = expr else {
        return Err(DataFusionError::Plan(format!(
            "{what} must be an id or an array literal of ids, got {expr:?}"
        )));
    };
    let array = value
        .to_array()
        .and_then(|array| cast(&array, &id_type()).map_err(DataFusionError::from))
        .map_err(|e| DataFusionError::Plan(format!("{what}: not an id: {e}")))?;
    let ids = array
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| DataFusionError::Plan(format!("{what}: cast did not yield the id type")))?;
    if ids.null_count() > 0 {
        return Err(DataFusionError::Plan(format!("{what} is null")));
    }
    Ok(ids.value(0))
}

/// The ids of a single-row `List` scalar (`[42, 43]`).
fn list_literal_to_ids(list: &ListArray, what: &str) -> DfResult<Vec<i128>> {
    if list.len() != 1 {
        return Err(DataFusionError::Plan(format!(
            "{what} list literal must have exactly one row, got {}",
            list.len()
        )));
    }
    let values = cast(&list.value(0), &id_type())
        .map_err(|e| DataFusionError::Plan(format!("{what}: not ids: {e}")))?;
    let ids = values
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| DataFusionError::Plan(format!("{what}: cast did not yield the id type")))?;
    if ids.null_count() > 0 {
        return Err(DataFusionError::Plan(format!(
            "{what} contains null elements"
        )));
    }
    Ok(ids.values().to_vec())
}

#[cfg(test)]
mod tests {
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use datafusion::{
        functions_nested::make_array::MakeArray, logical_expr::ScalarUDF, prelude::lit,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::{
        catalog::{Connection, IndexSpec, connect},
        config::{CompactionSettings, OptimizeOptions},
    };

    /// The table whose rows are the nodes.
    const THINGS: &str = "things";
    /// The edge table.
    const EDGES: &str = "edges";

    /// `things` in `db`, four rows named one to four; returns their ids, in
    /// that order.
    fn things(db: &Connection) -> Vec<i128> {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![
                "one", "two", "three", "four",
            ]))],
        )
        .expect("things");
        db.create_table(THINGS, schema, IndexSpec::new())
            .expect("create things")
            .append(&batch)
            .expect("append things");
        let rows = db
            .query_sql("SELECT _id FROM things ORDER BY _id")
            .expect("ids");
        rows.iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("_id")
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// The chain `one - two - three - four` of `things` rows `ids`, every
    /// edge both ways, as the `edges` table of `graph`, indexed.
    fn edges(graph: &Connection, ids: &[i128]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("src_table", DataType::Utf8, false),
            Field::new("src_id", id_type(), false),
            Field::new("dst_table", DataType::Utf8, false),
            Field::new("dst_id", id_type(), false),
        ]));
        let (mut src, mut dst) = (Vec::new(), Vec::new());
        for pair in ids.windows(2) {
            src.extend([pair[0], pair[1]]);
            dst.extend([pair[1], pair[0]]);
        }
        let tables = vec![THINGS; src.len()];
        let column = |values: Vec<i128>| {
            Arc::new(
                Decimal128Array::from(values)
                    .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
                    .expect("ids"),
            )
        };
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(tables.clone())),
                column(src),
                Arc::new(StringArray::from(tables)),
                column(dst),
            ],
        )
        .expect("edges");
        let table = graph
            .create_table(EDGES, schema, IndexSpec::new())
            .expect("create edges");
        table.append(&batch).expect("append edges");
        table
            .optimize(
                &OptimizeOptions::compact(CompactionSettings::default()).with_adjacency(
                    "src_table",
                    "src_id",
                    "dst_table",
                    "dst_id",
                ),
            )
            .expect("optimize");
    }

    fn column_strings(batches: &[RecordBatch], column: usize) -> Vec<String> {
        batches
            .iter()
            .flat_map(|batch| {
                let values = cast(batch.column(column), &DataType::Utf8).expect("cast");
                values
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("utf8")
                    .iter()
                    .flatten()
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The functions return rows of the named table, resolved like search
    /// hits: a walk's rows nearest first with the hop as the score, a
    /// ranking's with the row both seeds reach ahead of the one only a seed
    /// reaches; `k` bounds the rows; a seed id that is no row is skipped.
    #[test]
    fn a_statement_walks_the_graph_and_gets_the_rows() {
        let dir = TempDir::new().expect("tempdir");
        let db = connect(format!("file://{}", dir.path().display())).expect("connect");
        let ids = things(&db);
        edges(&db, &ids);

        let rows = db
            .query_sql(&format!(
                "SELECT name, score FROM graph_walk('edges', 'things', 'things', [{}], 2, 10) \
                 ORDER BY score",
                ids[0]
            ))
            .expect("walk");
        assert_eq!(column_strings(&rows, 0), vec!["one", "two", "three"]);
        assert_eq!(column_strings(&rows, 1), vec!["0.0", "1.0", "2.0"]);

        let rows = db
            .query_sql(&format!(
                "SELECT name FROM graph_rank('edges', 'things', 'things', [{}, {}], 1, 10) \
                 ORDER BY score DESC",
                ids[0], ids[2]
            ))
            .expect("rank");
        let ranked = column_strings(&rows, 0);
        let position = |name: &str| ranked.iter().position(|n| n == name).expect("ranked");
        assert!(
            position("two") < position("four"),
            "the row both seeds reach outranks the one only `three` reaches: {ranked:?}"
        );

        let rows = db
            .query_sql(&format!(
                "SELECT name FROM graph_walk('edges', 'things', 'things', {}, 3, 2)",
                ids[0]
            ))
            .expect("bounded");
        assert_eq!(
            column_strings(&rows, 0),
            vec!["one", "two"],
            "k bounds the rows; one seed needs no array"
        );

        let rows = db
            .query_sql("SELECT name FROM graph_walk('edges', 'things', 'things', [999999], 3, 10)")
            .expect("unknown seed");
        assert!(column_strings(&rows, 0).is_empty(), "a seed that is no row");

        assert!(
            db.query_sql("SELECT name FROM graph_walk('edges', 'things', 'things', [1], 2)")
                .is_err(),
            "k is required"
        );
        assert!(
            db.query_sql("SELECT name FROM graph_walk('things', 'things', 'things', [1], 2, 10)")
                .is_err(),
            "a table without an adjacency index is refused"
        );
    }

    /// A graph attached from another catalog serves the functions without
    /// naming the edge table, over the attached-to catalog's rows; its edge
    /// table is no table of that catalog.
    #[test]
    fn an_attached_graph_is_walked_without_naming_its_table() {
        let dir = TempDir::new().expect("tempdir");
        let customer = connect("memory://").expect("connect");
        let ids = things(&customer);
        let graph = connect(format!("file://{}", dir.path().display())).expect("connect");
        edges(&graph, &ids);
        customer.attach_graph(graph, EDGES);

        let rows = customer
            .query_sql(&format!(
                "SELECT name FROM graph_walk('things', 'things', [{}], 1, 10) ORDER BY score",
                ids[0]
            ))
            .expect("walk through the attached graph");
        assert_eq!(column_strings(&rows, 0), vec!["one", "two"]);
        assert_eq!(
            customer.list_tables().expect("tables"),
            vec![THINGS.to_string()],
            "the edge table is the graph catalog's alone"
        );
        assert!(
            customer.query_sql("SELECT count(*) FROM edges").is_err(),
            "not reachable by name"
        );
    }

    /// One id, an array literal and an unfolded `make_array` all read as
    /// seeds; a string is refused with the argument named.
    #[test]
    fn seeds_read_from_an_id_or_an_array() {
        assert_eq!(arg_to_ids(&lit(7i64), "seeds").expect("one"), vec![7]);
        let list = ScalarValue::List(ScalarValue::new_list_nullable(
            &[ScalarValue::from(1i64), ScalarValue::from(2i64)],
            &DataType::Int64,
        ));
        assert_eq!(
            arg_to_ids(&Expr::Literal(list, None), "seeds").expect("list"),
            vec![1, 2]
        );
        let make_array = ScalarUDF::from(MakeArray::new());
        let unfolded = make_array.call(vec![lit(3i64), lit(4i64)]);
        assert_eq!(
            arg_to_ids(&unfolded, "seeds").expect("make_array"),
            vec![3, 4]
        );
        let refused = arg_to_ids(&lit("n1"), "seeds").expect_err("not an id");
        assert!(refused.to_string().contains("seeds"), "{refused}");
    }
}
