// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `graph_walk` — a bounded walk over an edge table, as a table function.
//!
//! An edge table is an ordinary table of directed edges: an `Int64` source
//! node, an `Int64` destination node, and the source node's readable key.
//! The same walk is expressible as a recursive CTE, which scans the whole
//! table once per hop; `graph_walk` builds the table's
//! [`EdgeGraph`] once per snapshot and touches only the neighbours each hop
//! expands, so a walk costs what it reaches rather than what the table holds.
//!
//! ## Query shape
//!
//! ```sql
//! -- every node within two hops of either seed, nearest first, at most 100:
//! SELECT key, hop FROM graph_walk('src', 'dst', 'src_key', 2, 100,
//!                                 'logs/12345', 'logs/67890');
//! ```
//!
//! Arguments: the source, destination and key column names, the most hops,
//! the most nodes returned, then one or more seed keys (each its own
//! argument, so a key may hold any character). Output: `key`, `node`, `hop`,
//! one row per node reached, seeds at hop 0, ordered by hop. A seed key the
//! table does not hold reaches nothing.

use std::{fmt, sync::Arc};

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::{TaskContext, context::SessionContext},
    logical_expr::{Expr, TableType},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::stream;

use crate::{
    runtime_bridge::run_on_pool,
    supertable::{
        handle::{SupertableReader, WeakReader},
        query::{
            exec::common::{arg_to_string, arg_to_usize, scope_to_call, traced_tvf},
            graph::{CachedEdgeGraph, EdgeGraph, Reached},
            provider::TABLE_NAME,
        },
    },
    utils::trace::detail_span,
};

/// SQL name of the walk TVF.
pub(crate) const GRAPH_WALK_UDTF: &str = "graph_walk";
/// The most hops one walk may take. Past a handful a walk over a connected
/// graph reaches most of it, and the answer stops being about the seeds.
const MAX_GRAPH_HOPS: usize = 8;
/// The fixed arguments before the seed keys: three columns, hops, limit.
const FIXED_ARGS: usize = 5;
/// Output column names.
const KEY_COLUMN: &str = "key";
const NODE_COLUMN: &str = "node";
const HOP_COLUMN: &str = "hop";

/// The walk's output schema: `key`, `node`, `hop`.
fn walk_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(KEY_COLUMN, DataType::Utf8, false),
        Field::new(NODE_COLUMN, DataType::Int64, false),
        Field::new(HOP_COLUMN, DataType::Int64, false),
    ]))
}

/// Register `graph_walk` on `ctx`, bound to the query's pinned `reader`.
pub(crate) fn register_graph_walk(ctx: &SessionContext, reader: Arc<SupertableReader>) {
    ctx.register_udtf(GRAPH_WALK_UDTF, Arc::new(GraphWalkFunc::new(reader)));
}

/// One parsed `graph_walk` call.
#[derive(Debug, Clone)]
struct WalkCall {
    /// `(source, destination, key)` column names.
    columns: [String; 3],
    hops: u32,
    limit: usize,
    seeds: Vec<String>,
}

impl WalkCall {
    fn parse(args: &[Expr]) -> DfResult<Self> {
        if args.len() <= FIXED_ARGS {
            return Err(DataFusionError::Plan(format!(
                "graph_walk expects (source column, destination column, key column, hops, \
                 limit, seed key, ...), got {} argument(s)",
                args.len()
            )));
        }
        let columns = [
            arg_to_string(&args[0], "graph_walk source column")?,
            arg_to_string(&args[1], "graph_walk destination column")?,
            arg_to_string(&args[2], "graph_walk key column")?,
        ];
        let hops = arg_to_usize(&args[3], "graph_walk hops")?;
        if hops > MAX_GRAPH_HOPS {
            return Err(DataFusionError::Plan(format!(
                "graph_walk hops must be at most {MAX_GRAPH_HOPS}, got {hops}"
            )));
        }
        let limit = arg_to_usize(&args[4], "graph_walk limit")?;
        let seeds = args[FIXED_ARGS..]
            .iter()
            .map(|a| arg_to_string(a, "graph_walk seed key"))
            .collect::<DfResult<Vec<_>>>()?;
        Ok(Self {
            columns,
            // Bounded by MAX_GRAPH_HOPS just above.
            hops: hops as u32,
            limit,
            seeds,
        })
    }
}

/// `TableFunctionImpl` for `graph_walk`.
#[derive(Debug)]
pub(crate) struct GraphWalkFunc {
    reader: WeakReader,
}

impl GraphWalkFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>) -> Self {
        Self {
            reader: WeakReader::from_reader(&reader),
        }
    }
}

impl TableFunctionImpl for GraphWalkFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let call = WalkCall::parse(args.exprs())?;
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "graph_walk: supertable consumer dropped before execution".into(),
            )
        })?;
        scope_to_call(GRAPH_WALK_UDTF, Arc::new(GraphWalkTable { reader, call }))
    }
}

/// One `graph_walk` call as a `TableProvider`; `scan` lowers to [`GraphWalkExec`].
struct GraphWalkTable {
    reader: Arc<SupertableReader>,
    call: WalkCall,
}

impl fmt::Debug for GraphWalkTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphWalkTable")
            .field("call", &self.call)
            .finish()
    }
}

#[async_trait]
impl TableProvider for GraphWalkTable {
    fn schema(&self) -> SchemaRef {
        walk_schema()
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
        Ok(Arc::new(GraphWalkExec::try_new(
            Arc::clone(&self.reader),
            self.call.clone(),
            projection.cloned(),
        )?))
    }
}

/// Custom `ExecutionPlan`: builds (or reuses) the table's [`EdgeGraph`] and
/// walks it inside `execute()`.
struct GraphWalkExec {
    reader: Arc<SupertableReader>,
    call: WalkCall,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl GraphWalkExec {
    fn try_new(
        reader: Arc<SupertableReader>,
        call: WalkCall,
        projection: Option<Vec<usize>>,
    ) -> DfResult<Self> {
        let schema = walk_schema();
        let projected_schema = match &projection {
            Some(indices) => Arc::new(
                schema
                    .project(indices)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            None => schema,
        };
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            reader,
            call,
            projection,
            projected_schema,
            cache,
        })
    }

    fn describe(&self) -> String {
        format!(
            "GraphWalkExec: columns={:?}, hops={}, limit={}, seeds={}",
            self.call.columns,
            self.call.hops,
            self.call.limit,
            self.call.seeds.len()
        )
    }
}

impl fmt::Debug for GraphWalkExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for GraphWalkExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

/// The table's walk index for `columns` on the reader's snapshot: the cached
/// one when it was built from this snapshot and these columns, otherwise a
/// scan of the three columns turned into an [`EdgeGraph`] on the reader pool
/// and cached. Two walks missing at once both build; the second's index
/// replaces the first's, and both are correct for the snapshot.
async fn edge_graph(
    reader: &Arc<SupertableReader>,
    columns: &[String; 3],
) -> DfResult<Arc<EdgeGraph>> {
    let manifest = Arc::clone(reader.manifest());
    if let Some(cached) = &*reader
        .graph_cache()
        .lock()
        .expect("graph_cache mutex poisoned")
        && Arc::ptr_eq(&cached.manifest, &manifest)
        && cached.columns == *columns
    {
        return Ok(Arc::clone(&cached.graph));
    }
    let ctx = reader
        .sql_session_context()
        .map_err(|e| DataFusionError::Plan(e.to_string()))?;
    let names: Vec<&str> = columns.iter().map(String::as_str).collect();
    let batches = ctx
        .table(TABLE_NAME)
        .await?
        .select_columns(&names)?
        .collect()
        .await?;
    let pool = Arc::clone(&reader.options().reader_pool);
    let graph = run_on_pool(
        Some(&pool),
        "graph_walk build: reader pool dropped result",
        move || EdgeGraph::from_batches(&batches),
    )
    .await
    .map_err(|e| DataFusionError::Execution(e.to_string()))?
    .map_err(|e| DataFusionError::Plan(format!("graph_walk: {e}")))?;
    let graph = Arc::new(graph);
    *reader
        .graph_cache()
        .lock()
        .expect("graph_cache mutex poisoned") = Some(CachedEdgeGraph {
        manifest,
        columns: columns.clone(),
        graph: Arc::clone(&graph),
    });
    Ok(graph)
}

/// The nodes reached as the walk's output batch.
fn reached_batch(reached: &[Reached]) -> DfResult<RecordBatch> {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            reached.iter().map(|r| r.key.as_str()),
        )),
        Arc::new(Int64Array::from_iter_values(reached.iter().map(|r| r.id))),
        Arc::new(Int64Array::from_iter_values(
            reached.iter().map(|r| i64::from(r.hop)),
        )),
    ];
    RecordBatch::try_new(walk_schema(), columns)
        .map_err(|e| DataFusionError::Execution(e.to_string()))
}

impl ExecutionPlan for GraphWalkExec {
    fn name(&self) -> &'static str {
        "GraphWalkExec"
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
                "GraphWalkExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let call = self.call.clone();
        let projection = self.projection.clone();
        let fut = async move {
            let graph = edge_graph(&reader, &call.columns).await?;
            let pool = Arc::clone(&reader.options().reader_pool);
            let reached = run_on_pool(
                Some(&pool),
                "graph_walk walk: reader pool dropped result",
                move || {
                    let seeds: Vec<u32> = call
                        .seeds
                        .iter()
                        .filter_map(|key| graph.find_key(key))
                        .collect();
                    graph.walk(&seeds, call.hops, call.limit)
                },
            )
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            let batch = reached_batch(&reached)?;
            match &projection {
                Some(indices) => batch
                    .project(indices)
                    .map_err(|e| DataFusionError::Execution(e.to_string())),
                None => Ok(batch),
            }
        };
        let span = detail_span!("tvf.graph_walk", rows_out = tracing::field::Empty);
        let stream = stream::once(traced_tvf(span, fut));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.projected_schema),
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use datafusion::scalar::ScalarValue;

    use super::*;

    fn lit(s: &str) -> Expr {
        Expr::Literal(ScalarValue::Utf8(Some(s.into())), None)
    }

    fn int(n: i64) -> Expr {
        Expr::Literal(ScalarValue::Int64(Some(n)), None)
    }

    #[test]
    fn a_call_names_three_columns_two_bounds_and_its_seeds() {
        let call = WalkCall::parse(&[
            lit("src"),
            lit("dst"),
            lit("src_key"),
            int(2),
            int(10),
            lit("logs/1"),
            lit("a,b=c"),
        ])
        .expect("parse");
        assert_eq!(call.columns, ["src", "dst", "src_key"].map(String::from));
        assert_eq!((call.hops, call.limit), (2, 10));
        assert_eq!(
            call.seeds,
            vec!["logs/1", "a,b=c"],
            "a seed may hold any character"
        );

        let no_seed = WalkCall::parse(&[lit("src"), lit("dst"), lit("k"), int(2), int(10)]);
        assert!(no_seed.is_err(), "at least one seed");
        let too_far = WalkCall::parse(&[
            lit("src"),
            lit("dst"),
            lit("k"),
            int(MAX_GRAPH_HOPS as i64 + 1),
            int(10),
            lit("s"),
        ]);
        assert!(too_far.is_err(), "hops are bounded");
    }
}
