//! Vector kNN as a DataFusion table-valued function.
//!
//! `vector_search(column, query, k)` registers via `register_udtf`
//! and lowers to [`VectorSearchExec`], a custom `ExecutionPlan` that
//! calls the existing async
//! [`SupertableReader::vector_search`](crate::supertable::handle::SupertableReader::vector_search)
//! kernel inside `execute()` and resolves each
//! [`SuperfileHit`] to the supertable's `_id` + projected scalar
//! columns through
//! [`SuperfileReader::take_by_local_doc_ids`].
//!
//! ## Query shape
//!
//! ```sql
//! SELECT _id, score
//! FROM vector_search('embedding', '0.1,0.2, ... ,0.9', 10)
//! ORDER BY score
//! ```
//!
//! The query vector is a *function argument* — the vector column is
//! stripped from the SQL schema at commit and lives in the embedded
//! blob, so it can never be a scanned column. It is passed
//! either as a comma-separated string literal (robust; what the
//! benchmark harness emits) or a SQL array literal `[...]`
//! (`make_array`).
//!
//! Output schema = the supertable scalar schema (`_id` + scalar +
//! FTS columns) plus a `score: Float32` column. `score` is the
//! vector distance under the column's metric (cosine: `1 - dot`,
//! L2-sq: squared distance); **smaller is nearer**, so `ORDER BY
//! score` ascending lists nearest neighbours first. See
//! [`SuperfileHit::score`].

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow::compute::cast;
use arrow_array::{Array, ArrayRef, Float32Array, ListArray};
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::execution::TaskContext;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::{Expr, TableType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use datafusion::scalar::ScalarValue;

use super::common::{arg_to_string, arg_to_usize, output_schema_with_score, resolve_hits};
use crate::superfile::reader::VectorSearchOptions;
use crate::supertable::handle::SupertableReader;

/// SQL name the TVF is registered under.
pub(crate) const VECTOR_SEARCH_UDTF: &str = "vector_search";

/// Register `vector_search(column, query, k)` on `ctx`, bound to the
/// query's pinned `reader` + scalar `schema`. Called from
/// [`Supertable::query_sql`](crate::supertable::handle::Supertable::query_sql).
pub(crate) fn register_vector_search(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        VECTOR_SEARCH_UDTF,
        Arc::new(VectorSearchFunc::new(reader, scalar_schema)),
    );
}

/// `TableFunctionImpl` for `vector_search`. Holds the query's pinned
/// snapshot; `call` parses the SQL arguments and hands back a
/// per-invocation [`VectorSearchTable`].
#[derive(Debug)]
pub(crate) struct VectorSearchFunc {
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl VectorSearchFunc {
    fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader,
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for VectorSearchFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        if args.len() != 3 {
            return Err(DataFusionError::Plan(format!(
                "vector_search expects 3 arguments (column, query_vector, k), got {}",
                args.len()
            )));
        }
        let column = arg_to_string(&args[0], "column")?;
        let query = arg_to_query_vector(&args[1])?;
        let k = arg_to_usize(&args[2], "k")?;
        Ok(Arc::new(VectorSearchTable {
            reader: Arc::clone(&self.reader),
            column,
            query,
            k,
            options: VectorSearchOptions::new(),
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
    }
}

/// One parsed `vector_search(...)` invocation as a `TableProvider`.
/// `scan` lowers to [`VectorSearchExec`]; no scalar `WHERE` filters
/// or `LIMIT` are pushed in (the TVF's `k` is the top-k bound).
struct VectorSearchTable {
    reader: Arc<SupertableReader>,
    column: String,
    query: Vec<f32>,
    k: usize,
    options: VectorSearchOptions,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for VectorSearchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorSearchTable")
            .field("column", &self.column)
            .field("k", &self.k)
            .field("dim", &self.query.len())
            .finish()
    }
}

#[async_trait]
impl TableProvider for VectorSearchTable {
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        let exec = VectorSearchExec::try_new(
            Arc::clone(&self.reader),
            self.column.clone(),
            self.query.clone(),
            self.k,
            self.options,
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
        )?;
        Ok(Arc::new(exec))
    }
}

/// Custom `ExecutionPlan` that runs the vector kNN kernel on the
/// query runtime inside `execute()` and emits the resolved
/// `_id` + scalar columns + `score`.
struct VectorSearchExec {
    reader: Arc<SupertableReader>,
    column: String,
    query: Vec<f32>,
    k: usize,
    options: VectorSearchOptions,
    /// Scalar schema, used as the resolve projection.
    scalar_schema: SchemaRef,
    /// Full (pre-projection) output schema: scalar columns + score.
    output_schema: SchemaRef,
    /// Optional projection into `output_schema`.
    projection: Option<Vec<usize>>,
    /// Output schema after `projection`.
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl VectorSearchExec {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        reader: Arc<SupertableReader>,
        column: String,
        query: Vec<f32>,
        k: usize,
        options: VectorSearchOptions,
        scalar_schema: SchemaRef,
        output_schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> DfResult<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(
                output_schema
                    .project(indices)
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?,
            ),
            None => Arc::clone(&output_schema),
        };
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&projected_schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            reader,
            column,
            query,
            k,
            options,
            scalar_schema,
            output_schema,
            projection,
            projected_schema,
            cache,
        })
    }
}

impl fmt::Debug for VectorSearchExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VectorSearchExec: column={}, k={}, dim={}",
            self.column,
            self.k,
            self.query.len()
        )
    }
}

impl DisplayAs for VectorSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "VectorSearchExec: column={}, k={}, dim={}",
            self.column,
            self.k,
            self.query.len()
        )
    }
}

impl ExecutionPlan for VectorSearchExec {
    fn name(&self) -> &'static str {
        "VectorSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
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
                "VectorSearchExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let column = self.column.clone();
        let query = self.query.clone();
        let k = self.k;
        let options = self.options;
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            let hits = reader
                .vector_search(&column, &query, k, options)
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                projection.as_deref(),
            )
            .await
        };

        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// Extract the query vector from a comma-separated string literal or
/// a SQL array literal (`make_array(...)`).
///
/// `pub(crate)` so `hybrid_search` parses its `q_vec` argument
/// through the exact same path as `vector_search`.
pub(crate) fn arg_to_query_vector(expr: &Expr) -> DfResult<Vec<f32>> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => parse_csv_floats(s),
        // SQL array literal `[...]`: the planner const-folds an
        // all-literal `make_array(...)` into a single-row `List`
        // scalar before the TVF is called.
        Expr::Literal(ScalarValue::List(list), _) => list_literal_to_f32(list),
        // Unfolded `make_array(...)` (e.g. with a non-literal arg).
        Expr::ScalarFunction(sf) if sf.func.name() == "make_array" => {
            sf.args.iter().map(scalar_expr_to_f32).collect()
        }
        other => Err(DataFusionError::Plan(format!(
            "vector_search query vector must be a comma-separated string or array literal, got {other:?}"
        ))),
    }
}

/// Convert a single-row `List` scalar (`[a, b, c]`) to `Vec<f32>`.
fn list_literal_to_f32(list: &ListArray) -> DfResult<Vec<f32>> {
    if list.len() != 1 {
        return Err(DataFusionError::Plan(format!(
            "vector_search query vector list literal must have exactly one row, got {}",
            list.len()
        )));
    }
    array_to_f32(&list.value(0))
}

/// Cast an arbitrary numeric array to `f32` and collect, rejecting
/// nulls (a query vector must be fully specified).
fn array_to_f32(values: &ArrayRef) -> DfResult<Vec<f32>> {
    let casted = cast(values, &DataType::Float32).map_err(|e| {
        DataFusionError::Plan(format!(
            "vector_search query vector: cannot cast elements to f32: {e}"
        ))
    })?;
    let arr = casted
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| {
            DataFusionError::Plan("vector_search query vector: cast did not yield Float32".into())
        })?;
    if arr.null_count() > 0 {
        return Err(DataFusionError::Plan(
            "vector_search query vector contains null elements".into(),
        ));
    }
    Ok(arr.values().iter().copied().collect())
}

fn parse_csv_floats(s: &str) -> DfResult<Vec<f32>> {
    let out: Vec<f32> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<f32>().map_err(|e| {
                DataFusionError::Plan(format!(
                    "vector_search query vector: cannot parse '{p}' as f32: {e}"
                ))
            })
        })
        .collect::<DfResult<_>>()?;
    if out.is_empty() {
        return Err(DataFusionError::Plan(
            "vector_search query vector is empty".to_string(),
        ));
    }
    Ok(out)
}

fn scalar_expr_to_f32(expr: &Expr) -> DfResult<f32> {
    match expr {
        Expr::Literal(sv, _) => scalar_to_f32(sv),
        other => Err(DataFusionError::Plan(format!(
            "vector_search array element must be a numeric literal, got {other:?}"
        ))),
    }
}

fn scalar_to_f32(sv: &ScalarValue) -> DfResult<f32> {
    match sv {
        ScalarValue::Float32(Some(v)) => Ok(*v),
        ScalarValue::Float64(Some(v)) => Ok(*v as f32),
        ScalarValue::Int64(Some(v)) => Ok(*v as f32),
        ScalarValue::Int32(Some(v)) => Ok(*v as f32),
        ScalarValue::UInt64(Some(v)) => Ok(*v as f32),
        ScalarValue::UInt32(Some(v)) => Ok(*v as f32),
        other => Err(DataFusionError::Plan(format!(
            "vector_search numeric literal expected, got {other:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Array, Decimal128Array, FixedSizeListArray, LargeStringArray, RecordBatch};
    use arrow_schema::{Field, Schema};
    use datafusion::prelude::lit;

    use crate::superfile::builder::{FtsConfig, VectorConfig};
    use crate::superfile::vector::distance::Metric;
    use crate::superfile::vector::rerank_codec::RerankCodec;
    use crate::supertable::{Supertable, SupertableOptions};
    use crate::test_helpers::default_tokenizer as tok;

    // ---- vector-column test harness (mirrors query::vector tests) ----

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn options_one_segment_per_commit(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                n_cent: 4,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
            }],
            Some(tok()),
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Doc `i` gets a one-hot vector at dim `(start + i) % dim`.
    fn build_vector_batch(start: u64, n: usize, dim: usize, schema: Arc<Schema>) -> RecordBatch {
        let titles = LargeStringArray::from((0..n).map(|i| format!("doc {i}")).collect::<Vec<_>>());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            let global = (start as usize) + i;
            for d in 0..dim {
                flat.push(if d == global % dim { 1.0 } else { 0.0 });
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(titles), Arc::new(fsl)]).expect("batch")
    }

    /// Single-segment supertable with `n` one-hot docs.
    fn supertable_one_segment(dim: usize, n: usize) -> Supertable {
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_vector_batch(0, n, dim, schema))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    /// `"1,0,0,..."` one-hot query targeting `active`.
    fn csv_one_hot(dim: usize, active: usize) -> String {
        (0..dim)
            .map(|d| if d == active { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float32Array {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32 column")
    }

    fn col_id<'a>(batch: &'a RecordBatch, name: &str) -> &'a Decimal128Array {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal128 _id column")
    }

    fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> &'a LargeStringArray {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large utf8 column")
    }

    // ---- arg parsing (unit) ----

    #[test]
    fn arg_to_query_vector_parses_csv_string() {
        let v = arg_to_query_vector(&lit("0.5, 1, -2.25")).expect("csv vector");
        assert_eq!(v, vec![0.5, 1.0, -2.25]);
    }

    #[test]
    fn arg_to_query_vector_rejects_empty_and_garbage() {
        assert!(arg_to_query_vector(&lit("")).is_err());
        assert!(arg_to_query_vector(&lit("1,foo,3")).is_err());
    }

    // ---- end-to-end through query_sql ----

    #[test]
    fn vector_search_tvf_emits_id_and_score_in_distance_order() {
        let dim = 16;
        let st = supertable_one_segment(dim, 8);
        let sql = format!(
            "SELECT _id, title, score FROM vector_search('emb', '{}', 8)",
            csv_one_hot(dim, 0)
        );
        let batches = st.query_sql(&sql).expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 8, "single segment, k=8 → all 8 docs resolved");

        let b = &batches[0];
        assert_eq!(b.num_columns(), 3);
        // Doc 0 is the exact one-hot match at dim 0 → nearest. `title`
        // is the deterministic anchor (`_id` is generator-assigned).
        assert_eq!(col_str(b, "title").value(0), "doc 0");
        // `_id` resolved for every row: 8 distinct, non-null keys.
        let ids = col_id(b, "_id");
        assert_eq!(ids.null_count(), 0);
        let unique: std::collections::HashSet<i128> =
            (0..ids.len()).map(|i| ids.value(i)).collect();
        assert_eq!(unique.len(), 8, "each hit resolves to a distinct _id");
        // Native emission order (no ORDER BY) is ascending distance.
        let score = col_f32(b, "score");
        for i in 1..score.len() {
            assert!(
                score.value(i - 1) <= score.value(i),
                "scores must be ascending: {} then {}",
                score.value(i - 1),
                score.value(i)
            );
        }
    }

    #[test]
    fn vector_search_tvf_star_projection_appends_score_column() {
        let dim = 16;
        let st = supertable_one_segment(dim, 8);
        let sql = format!(
            "SELECT * FROM vector_search('emb', '{}', 3)",
            csv_one_hot(dim, 0)
        );
        let batches = st.query_sql(&sql).expect("query_sql");
        let b = &batches[0];
        // Scalar schema (_id, title) + score.
        assert_eq!(b.num_columns(), 3);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "title");
        assert_eq!(b.schema().field(2).name(), "score");
        assert_eq!(b.num_rows(), 3);
    }

    #[test]
    fn vector_search_tvf_score_only_projection() {
        let dim = 16;
        let st = supertable_one_segment(dim, 8);
        let sql = format!(
            "SELECT score FROM vector_search('emb', '{}', 2)",
            csv_one_hot(dim, 0)
        );
        let batches = st.query_sql(&sql).expect("query_sql");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 1);
        assert_eq!(b.schema().field(0).name(), "score");
        assert_eq!(b.num_rows(), 2);
    }

    #[test]
    fn vector_search_tvf_score_only_matches_full_projection_scores() {
        // The `score`-only projection decodes no scalar columns (opens
        // no segment readers); it must still produce the exact scores
        // and row count of the fully-resolved projection.
        let dim = 16;
        let st = supertable_one_segment(dim, 8);
        let q = csv_one_hot(dim, 0);
        let full = st
            .query_sql(&format!(
                "SELECT _id, title, score FROM vector_search('emb', '{q}', 5)"
            ))
            .expect("query_sql");
        let only = st
            .query_sql(&format!("SELECT score FROM vector_search('emb', '{q}', 5)"))
            .expect("query_sql");

        let collect_scores = |batches: &[RecordBatch]| -> Vec<f32> {
            let mut out = Vec::new();
            for b in batches {
                let c = col_f32(b, "score");
                out.extend((0..c.len()).map(|i| c.value(i)));
            }
            out
        };
        let full_scores = collect_scores(&full);
        let only_scores = collect_scores(&only);
        assert_eq!(only_scores.len(), 5);
        assert_eq!(
            full_scores, only_scores,
            "score-only projection must not change scores or order"
        );
    }

    #[test]
    fn vector_search_tvf_accepts_sql_array_literal() {
        let dim = 16;
        let st = supertable_one_segment(dim, 8);
        let arr = (0..dim)
            .map(|d| if d == 0 { "1.0" } else { "0.0" })
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!("SELECT title FROM vector_search('emb', [{arr}], 1)");
        let batches = st.query_sql(&sql).expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
        assert_eq!(col_str(&batches[0], "title").value(0), "doc 0");
    }

    #[test]
    fn vector_search_tvf_empty_supertable_returns_no_rows() {
        let dim = 16;
        let st = Supertable::create(options_one_segment_per_commit(dim)).expect("create");
        let sql = format!(
            "SELECT _id, score FROM vector_search('emb', '{}', 5)",
            csv_one_hot(dim, 0)
        );
        let batches = st.query_sql(&sql).expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }
}
