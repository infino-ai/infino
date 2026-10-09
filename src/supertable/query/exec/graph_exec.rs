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
//! share. Edges are directed: a walk follows an edge from its source row to
//! its destination row only. A ranking's scores sum to one over every row
//! the walk reached — rows of other tables and rows past `k` included — not
//! over the rows returned, so the returned scores are a share of the whole
//! neighbourhood. The walk itself runs inside the plan's `execute`, like
//! the search kernels, so the index hydrates and the rows resolve on the
//! query runtime.
//!
//! Seeds are whole-number literals: an integer, a `DECIMAL` of scale zero
//! (what a statement's wide integers are rewritten to: `CAST('<digits>' AS
//! DECIMAL(38, 0))`), or a list of either. A float literal or a decimal
//! with a fraction is refused rather than rounded to a neighbouring row.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
    time::Instant,
};

use arrow_array::{Array, ListArray};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    logical_expr::{Cast, Expr, TableType, TryCast},
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
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        manifest::SuperfileUri,
        query::{
            SuperfileHit,
            dispatch::apply_tombstone_filter,
            exec::common::{
                arg_to_string, arg_to_usize, output_schema_with_score, resolve_hits, traced_tvf,
            },
            vector::{MissingRow, place_for_scalar_resolve},
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
/// Most hops a walk may take from its seeds. The walk reaches every node
/// within the hops before its rows are filtered to the target table, so the
/// hops bound how much of a connected graph one statement touches; past
/// this many, what a seed reaches it has reached already.
pub(crate) const MAX_GRAPH_HOPS: u32 = 16;
/// Fewest reached rows placed and checked against the tombstones in one
/// round. `k` counts the rows that survive that check, so the reached rows
/// are taken in rounds of at least this many until `k` do, rather than
/// placing every row of a large neighbourhood up front.
const PLACEMENT_ROUND_MIN_ROWS: usize = 256;

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
            .ok()
            .filter(|&hops| hops <= MAX_GRAPH_HOPS)
            .ok_or_else(|| {
                DataFusionError::Plan(format!("{name} hops must be at most {MAX_GRAPH_HOPS}"))
            })?;
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
                .await?;
            let by_id: Vec<SuperfileHit> = reached
                .hits
                .iter()
                .filter(|hit| hit.table == target_name)
                .map(|hit| {
                    let score = match traversal {
                        Traversal::Walk => hit.hop as f32,
                        Traversal::Rank => hit.score as f32,
                    };
                    SuperfileHit::by_id(hit.id, score)
                })
                .collect();
            // Placed by id like a hidden-index hit, then checked against the
            // table's tombstones, as a search's hits are: a row the edge
            // table names that is gone since — rewritten under a new id, or
            // deleted — is skipped. `k` counts the rows that survive, so the
            // reached rows are taken in rounds until `k` do: a deleted row
            // among the first `k` is backfilled by the next, not missing
            // from the result.
            let round = k.max(PLACEMENT_ROUND_MIN_ROWS);
            let mut hits: Vec<SuperfileHit> = Vec::new();
            for chunk in by_id.chunks(round) {
                let placed = place_for_scalar_resolve(&target, chunk, MissingRow::Skip).await?;
                hits.extend(drop_tombstoned(&target, placed).await?);
                if hits.len() >= k {
                    break;
                }
            }
            hits.truncate(k);
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

/// `hits` without the tombstoned ones: each superfile's deny bitmap, from
/// the table's tombstone cache after one batched prefetch, applied through
/// the same post-rank filter the search fan-outs use. Order is kept. The
/// hits are grouped by superfile once and the survivors kept in a set, so
/// the cost is linear in the hits whatever `k` a statement asked for.
async fn drop_tombstoned(
    reader: &SupertableReader,
    hits: Vec<SuperfileHit>,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let Some(cache) = reader.tombstone_cache.as_ref() else {
        return Ok(hits);
    };
    let manifest = reader.manifest();
    // One group per superfile, in first-seen order.
    let mut groups: Vec<(SuperfileUri, Vec<SuperfileHit>)> = Vec::new();
    let mut group_of: HashMap<SuperfileUri, usize> = HashMap::new();
    for hit in &hits {
        let at = *group_of.entry(hit.superfile).or_insert_with(|| {
            groups.push((hit.superfile, Vec::new()));
            groups.len() - 1
        });
        groups[at].1.push(*hit);
    }
    let mut entries = Vec::with_capacity(groups.len());
    for (uri, _) in &groups {
        let entry = manifest
            .lookup_superfile_entry(*uri)
            .await
            .map_err(QueryError::ManifestLoad)?
            .ok_or_else(|| {
                QueryError::Internal(format!(
                    "placed hit names superfile {uri:?} missing from the manifest"
                ))
            })?;
        entries.push(entry);
    }
    let now = Instant::now();
    let ids: Vec<_> = entries.iter().map(|e| e.superfile_id).collect();
    cache.prefetch(&ids, now).await;
    let mut survivors: HashSet<(SuperfileUri, u32)> = HashSet::with_capacity(hits.len());
    for ((uri, mut of_entry), entry) in groups.into_iter().zip(&entries) {
        apply_tombstone_filter(Some(cache), entry, &mut of_entry, now)?;
        survivors.extend(of_entry.iter().map(|h| (uri, h.local_doc_id)));
    }
    let mut kept = hits;
    kept.retain(|h| survivors.contains(&(h.superfile, h.local_doc_id)));
    Ok(kept)
}

/// Extract one `_id` or an array literal of them (`[42, 43]`, which the
/// planner const-folds to a `List` scalar, or leaves as `make_array(...)`
/// when an element is not a plain literal). An id is a whole-number
/// literal: an integer, or a `DECIMAL` of scale zero — the form a
/// statement's wide integers reach the plan in, `CAST('<digits>' AS
/// DECIMAL(38, 0))`, since `query_sql` rewrites every whole-number literal
/// past 64 bits that way to keep it exact. A float literal, or a decimal
/// with a fraction, is refused: rounded, it would walk from a neighbouring
/// row.
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

/// One id literal, as [`arg_to_ids`] reads it: a literal, a cast of a
/// digit string to a whole decimal, or the negation of either.
fn scalar_expr_to_id(expr: &Expr, what: &str) -> DfResult<i128> {
    match expr {
        Expr::Literal(value, _) => scalar_value_to_id(value, what),
        Expr::Cast(Cast { expr: inner, field }) | Expr::TryCast(TryCast { expr: inner, field })
            if matches!(field.data_type(), DataType::Decimal128(_, 0)) =>
        {
            match inner.as_ref() {
                Expr::Literal(
                    ScalarValue::Utf8(Some(text))
                    | ScalarValue::LargeUtf8(Some(text))
                    | ScalarValue::Utf8View(Some(text)),
                    _,
                ) => text.trim().parse::<i128>().map_err(|_| {
                    DataFusionError::Plan(format!("{what}: {text:?} is not a whole-number id"))
                }),
                inner => scalar_expr_to_id(inner, what),
            }
        }
        Expr::Negative(inner) => scalar_expr_to_id(inner, what)?
            .checked_neg()
            .ok_or_else(|| DataFusionError::Plan(format!("{what}: id out of range"))),
        other => Err(DataFusionError::Plan(format!(
            "{what} must be an id or an array literal of ids, got {other:?}"
        ))),
    }
}

/// One id as a scalar: any integer, or a decimal that is a whole number.
fn scalar_value_to_id(value: &ScalarValue, what: &str) -> DfResult<i128> {
    match value {
        ScalarValue::Int8(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::Int16(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::Int32(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::Int64(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::UInt8(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::UInt16(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::UInt32(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::UInt64(Some(v)) => Ok(i128::from(*v)),
        ScalarValue::Decimal128(Some(v), _, scale) => whole_decimal(*v, *scale, what),
        ScalarValue::Float32(Some(_)) | ScalarValue::Float64(Some(_)) => {
            Err(DataFusionError::Plan(format!(
                "{what}: {value} is not a whole-number id (an id has up to 38 digits and no \
                 fraction)"
            )))
        }
        other if other.is_null() => Err(DataFusionError::Plan(format!("{what}: an id is null"))),
        other => Err(DataFusionError::Plan(format!(
            "{what} must be an id or an array literal of ids, got {other:?}"
        ))),
    }
}

/// The whole number a `Decimal128` value of `scale` is, or a refusal when
/// it carries a fraction.
fn whole_decimal(value: i128, scale: i8, what: &str) -> DfResult<i128> {
    if scale == 0 {
        return Ok(value);
    }
    let out_of_range = || DataFusionError::Plan(format!("{what}: id out of range"));
    if scale < 0 {
        let unit = 10i128
            .checked_pow(u32::from(scale.unsigned_abs()))
            .ok_or_else(out_of_range)?;
        return value.checked_mul(unit).ok_or_else(out_of_range);
    }
    let unit = 10i128
        .checked_pow(u32::from(scale.unsigned_abs()))
        .ok_or_else(out_of_range)?;
    if value % unit != 0 {
        return Err(DataFusionError::Plan(format!(
            "{what}: {value}e-{scale} is not a whole-number id"
        )));
    }
    Ok(value / unit)
}

/// The ids of a single-row `List` scalar (`[42, 43]`), each element read
/// as [`scalar_value_to_id`] reads a literal.
fn list_literal_to_ids(list: &ListArray, what: &str) -> DfResult<Vec<i128>> {
    if list.len() != 1 {
        return Err(DataFusionError::Plan(format!(
            "{what} list literal must have exactly one row, got {}",
            list.len()
        )));
    }
    let values = list.value(0);
    (0..values.len())
        .map(|at| {
            let value = ScalarValue::try_from_array(&values, at)
                .map_err(|e| DataFusionError::Plan(format!("{what}: not ids: {e}")))?;
            scalar_value_to_id(&value, what)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use arrow::compute::cast;
    use arrow_array::{Decimal128Array, RecordBatch, StringArray};
    use arrow_schema::{Field, Schema};
    use datafusion::{
        functions_nested::make_array::MakeArray,
        logical_expr::ScalarUDF,
        prelude::{col, lit},
    };
    use tempfile::TempDir;

    use super::*;
    use crate::{
        catalog::{Connection, IndexSpec, connect},
        config::{CompactionSettings, OptimizeOptions},
        supertable::schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
    };

    /// The table whose rows are the nodes.
    const THINGS: &str = "things";
    /// The edge table.
    const EDGES: &str = "edges";
    /// A 32-digit id, as every `_id` of a real table is: past `u64`, so a
    /// statement's rewrite turns its literal into a `CAST` to a decimal.
    const WIDE_ID: i128 = 33_045_841_832_672_128_640_984_815_776_677;

    /// The engine's id type, which every `_id` column has.
    fn id_type() -> DataType {
        DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE)
    }

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

        // A row deleted after the edges were written is still in the graph,
        // and is skipped: the final list is checked against the table's
        // tombstones.
        db.open_table(THINGS)
            .expect("things")
            .delete(col("name").eq(lit("two")))
            .expect("delete two");
        let rows = db
            .query_sql(&format!(
                "SELECT name FROM graph_walk('edges', 'things', 'things', [{}], 2, 10) \
                 ORDER BY score",
                ids[0]
            ))
            .expect("walk past a deleted row");
        assert_eq!(
            column_strings(&rows, 0),
            vec!["one", "three"],
            "the deleted row is skipped; the walk still passes through it"
        );
        let rows = db
            .query_sql(&format!(
                "SELECT name FROM graph_walk('edges', 'things', 'things', [{}], 2, 2) \
                 ORDER BY score",
                ids[0]
            ))
            .expect("walk past a deleted row with k");
        assert_eq!(
            column_strings(&rows, 0),
            vec!["one", "three"],
            "k counts the rows that survive the tombstones, so the deleted row is backfilled"
        );

        assert!(
            db.query_sql("SELECT name FROM graph_walk('edges', 'things', 'things', [1], 2)")
                .is_err(),
            "k is required"
        );
        let refused = db
            .query_sql(&format!(
                "SELECT name FROM graph_walk('edges', 'things', 'things', [{}.5], 2, 10)",
                ids[0]
            ))
            .expect_err("a seed with a fraction");
        assert!(
            refused.to_string().contains("whole-number"),
            "refused, not rounded to a neighbouring row: {refused}"
        );
        assert!(
            db.query_sql(&format!(
                "SELECT name FROM graph_walk('edges', 'things', 'things', [{}], {}, 10)",
                ids[0],
                MAX_GRAPH_HOPS + 1
            ))
            .is_err(),
            "hops past the cap are refused"
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
        let rows_db = connect("memory://").expect("connect");
        let ids = things(&rows_db);
        let graph = connect(format!("file://{}", dir.path().display())).expect("connect");
        edges(&graph, &ids);
        rows_db.attach_graph(&graph, EDGES);

        let walk = format!(
            "SELECT name FROM graph_walk('things', 'things', [{}], 1, 10) ORDER BY score",
            ids[0]
        );
        let rows = rows_db
            .query_sql(&walk)
            .expect("walk through the attached graph");
        assert_eq!(column_strings(&rows, 0), vec!["one", "two"]);
        assert_eq!(
            rows_db.list_tables().expect("tables"),
            vec![THINGS.to_string()],
            "the edge table is the graph catalog's alone"
        );
        assert!(
            rows_db.query_sql("SELECT count(*) FROM edges").is_err(),
            "not reachable by name"
        );

        // The graph is held weakly: once its connection is gone, a walk says
        // so rather than keeping it alive.
        drop(graph);
        let gone = rows_db.query_sql(&walk).expect_err("the graph was dropped");
        assert!(gone.to_string().contains("dropped"), "{gone}");
    }

    /// One id, an array literal and an unfolded `make_array` all read as
    /// seeds; a wide id arrives as the `CAST` of its digits and reads
    /// exactly; a float, a decimal with a fraction and a string are refused
    /// with the argument named.
    #[test]
    fn seeds_read_from_an_id_or_an_array() {
        assert_eq!(arg_to_ids(&lit(7i64), "seeds").expect("one"), vec![7]);
        let cast_of_digits = Expr::Cast(Cast::new(Box::new(lit(WIDE_ID.to_string())), id_type()));
        assert_eq!(
            arg_to_ids(&cast_of_digits, "seeds").expect("cast of digits"),
            vec![WIDE_ID],
            "the digits read exactly, not through a float"
        );
        assert_eq!(
            arg_to_ids(&Expr::Negative(Box::new(lit(7i64))), "seeds").expect("negative"),
            vec![-7]
        );
        let whole = ScalarValue::Decimal128(Some(70), DECIMAL128_PRECISION, 1);
        assert_eq!(
            arg_to_ids(&Expr::Literal(whole, None), "seeds").expect("7.0"),
            vec![7]
        );
        let fraction = ScalarValue::Decimal128(Some(75), DECIMAL128_PRECISION, 1);
        let refused = arg_to_ids(&Expr::Literal(fraction, None), "seeds").expect_err("7.5");
        assert!(refused.to_string().contains("seeds"), "{refused}");
        let refused = arg_to_ids(&lit(7.5f64), "seeds").expect_err("a float");
        assert!(refused.to_string().contains("whole-number"), "{refused}");
        let refused = arg_to_ids(&lit(7.0f64), "seeds").expect_err("a float, even a whole one");
        assert!(refused.to_string().contains("seeds"), "{refused}");
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
