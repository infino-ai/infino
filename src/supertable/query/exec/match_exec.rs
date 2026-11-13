// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Unranked token / exact match as DataFusion table-valued functions.
//!
//! `token_match(column, query [, mode])` and `exact_match(column,
//! value)` are the **unranked** siblings of `bm25_search`. They register
//! via `register_udtf` and lower to [`MatchExec`], a custom
//! `ExecutionPlan` that calls the unranked kernels
//! ([`SupertableReader::token_match`](crate::supertable::handle::SupertableReader::token_match)
//! / `exact_match`) inside `execute()` and resolves each
//! [`SuperfileHit`](crate::supertable::query::SuperfileHit) to the
//! supertable's `_id` + projected scalar columns + `score` via the
//! shared [`resolve_hits`](super::common::resolve_hits).
//!
//! ## Query shape
//!
//! ```sql
//! -- rows whose `title` contains every token (AND) / any token (OR):
//! SELECT _id FROM token_match('title', 'rust async', 'and');
//! SELECT _id FROM token_match('title', 'rust async');          -- OR (default)
//! -- rows whose `title` equals the raw string exactly:
//! SELECT _id FROM exact_match('title', 'Rust Compiler');
//! ```
//!
//! These are unranked: the `score` column is present (for schema
//! uniformity with the other search TVFs) but constant `0.0`. Order is
//! unspecified — add a SQL `ORDER BY` / `LIMIT` for control.

use std::{fmt, sync::Arc};

use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider},
    common::tree_node::Transformed,
    datasource::{DefaultTableSource, provider_as_source},
    error::{DataFusionError, Result as DfResult},
    execution::{TaskContext, context::SessionContext},
    logical_expr::{Expr, LogicalPlan, Sort, TableProviderFilterPushDown, TableType},
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{TryStreamExt, stream};

use crate::{
    superfile::fts::reader::BoolMode,
    supertable::{
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        query::exec::{
            common::{
                Arity, PushedPredicate, arg_to_string, live_reader, output_schema_with_score,
                resolve_hits, scope_to_call, traced_tvf, traced_tvf_rows,
            },
            fts_exec::arg_to_bool_mode,
        },
        query::{match_order::MatchOrder, range_top_k::with_scan_source},
    },
    utils::trace::detail_span,
};

/// SQL name for the unranked token-match TVF.
pub(crate) const TOKEN_MATCH_UDTF: &str = "token_match";
/// SQL name for the unranked exact-match TVF.
pub(crate) const EXACT_MATCH_UDTF: &str = "exact_match";
/// Arguments of `token_match(column, query)`; the optional `mode` makes it
/// `token_match(column, query, mode)`.
pub(crate) const TOKEN_MATCH_ARITY: Arity = Arity {
    min: 2,
    max: 3,
    arguments: "column, query[, mode]",
};
/// Arguments of `exact_match(column, value)`.
pub(crate) const EXACT_MATCH_ARITY: Arity = Arity {
    min: 2,
    max: 2,
    arguments: "column, value",
};

/// Register `token_match` + `exact_match` on `ctx`, bound to the
/// query's pinned `reader` + scalar `schema`. Called from
/// [`Supertable::query_sql`](crate::supertable::handle::Supertable::query_sql).
pub(crate) fn register_match(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        TOKEN_MATCH_UDTF,
        Arc::new(TokenMatchFunc::new(
            Arc::clone(&reader),
            Arc::clone(&scalar_schema),
        )),
    );
    ctx.register_udtf(
        EXACT_MATCH_UDTF,
        Arc::new(ExactMatchFunc::new(reader, scalar_schema)),
    );
}

/// Which unranked kernel a `MatchExec` invocation runs.
#[derive(Debug, Clone)]
enum MatchQuery {
    /// `token_match(col, query, mode)` — token AND/OR retrieval.
    Token { query: String, mode: BoolMode },
    /// `exact_match(col, value)` — two-pass raw-string match.
    Exact { value: String },
}

/// `TableFunctionImpl` for `token_match`.
#[derive(Debug)]
pub(crate) struct TokenMatchFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl TokenMatchFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for TokenMatchFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();
        TOKEN_MATCH_ARITY.check(TOKEN_MATCH_UDTF, args.len())?;
        let column = arg_to_string(&args[0], "token_match column")?;
        let query = arg_to_string(&args[1], "token_match query")?;
        let mode = match args.get(2) {
            Some(expr) => arg_to_bool_mode(expr)?,
            None => BoolMode::Or,
        };
        let reader = live_reader(&self.reader, "token_match")?;
        scope_to_call(
            TOKEN_MATCH_UDTF,
            Arc::new(MatchTable {
                reader,
                column,
                query: MatchQuery::Token { query, mode },
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
                order: None,
            }),
        )
    }
}

/// `TableFunctionImpl` for `exact_match`.
#[derive(Debug)]
pub(crate) struct ExactMatchFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl ExactMatchFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for ExactMatchFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();
        EXACT_MATCH_ARITY.check(EXACT_MATCH_UDTF, args.len())?;
        let column = arg_to_string(&args[0], "exact_match column")?;
        let value = arg_to_string(&args[1], "exact_match value")?;
        let reader = live_reader(&self.reader, "exact_match")?;
        scope_to_call(
            EXACT_MATCH_UDTF,
            Arc::new(MatchTable {
                reader,
                column,
                query: MatchQuery::Exact { value },
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
                order: None,
            }),
        )
    }
}

/// One parsed match invocation as a `TableProvider`. `scan` lowers to
/// [`MatchExec`].
pub(crate) struct MatchTable {
    reader: Arc<SupertableReader>,
    column: String,
    query: MatchQuery,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    /// The `ORDER BY column LIMIT n` above the match, when the plan is one
    /// the walk can honour (see [`MatchTopK`]).
    order: Option<OrderHint>,
}

/// An `ORDER BY column [DESC] LIMIT fetch` over a match.
#[derive(Debug, Clone)]
pub(crate) struct OrderHint {
    column: String,
    descending: bool,
    fetch: usize,
}

impl MatchTable {
    /// This match with `hint` for its walk.
    fn with_order(&self, hint: OrderHint) -> Self {
        Self {
            reader: Arc::clone(&self.reader),
            column: self.column.clone(),
            query: self.query.clone(),
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
            order: Some(hint),
        }
    }
}

impl fmt::Debug for MatchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchTable")
            .field("column", &self.column)
            .field("query", &self.query)
            .field("order", &self.order)
            .finish()
    }
}

#[async_trait]
impl TableProvider for MatchTable {
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
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let exec = MatchExec::try_new(
            Arc::clone(&self.reader),
            self.column.clone(),
            self.query.clone(),
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
        )?
        .with_filters(filters.to_vec())
        .with_order(self.order.clone());
        Ok(Arc::new(exec))
    }

    /// Every `WHERE` filter is `Inexact`, as on `bm25_search`: the scan
    /// uses it to skip superfiles that cannot pass, and
    /// DataFusion keeps a `FilterExec` above that applies it exactly.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/// [`MatchTopK`]: `ORDER BY column LIMIT n` over `token_match` /
/// `exact_match` hands the match the order and the limit, so it opens its
/// superfiles in the column's order and stops once none left can place a
/// row (see [`MatchOrder`]). The rule only marks the match's provider; the
/// sort and the filters stay above it, and the answer is the one they give
/// over the rows the match returns. It fires for a sort on a single plain
/// column, through filters, plain column projections and aliases; whether
/// the walk can honour it is the match's call, made when it runs.
#[derive(Debug, Default)]
pub(crate) struct MatchTopK;

impl OptimizerRule for MatchTopK {
    fn name(&self) -> &str {
        "match_top_k"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        match order_match(&plan)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// The sort with its match's provider swapped for one carrying the order,
/// or `None` when the plan is not a sort over a match.
fn order_match(plan: &LogicalPlan) -> DfResult<Option<LogicalPlan>> {
    let LogicalPlan::Sort(Sort {
        expr,
        input,
        fetch: Some(fetch),
    }) = plan
    else {
        return Ok(None);
    };
    let [sort_expr] = expr.as_slice() else {
        return Ok(None);
    };
    let Expr::Column(column) = &sort_expr.expr else {
        return Ok(None);
    };
    let mut node = input.as_ref();
    let scan = loop {
        match node {
            LogicalPlan::Projection(projection)
                if projection.expr.iter().all(|e| matches!(e, Expr::Column(_))) =>
            {
                node = projection.input.as_ref();
            }
            LogicalPlan::Filter(filter) => node = filter.input.as_ref(),
            LogicalPlan::SubqueryAlias(alias) => node = alias.input.as_ref(),
            LogicalPlan::TableScan(scan) => break scan,
            _ => return Ok(None),
        }
    };
    let Some(table) = scan
        .source
        .downcast_ref::<DefaultTableSource>()
        .and_then(|source| source.table_provider.downcast_ref::<MatchTable>())
    else {
        return Ok(None);
    };
    if table.order.is_some() {
        return Ok(None);
    }
    let hinted = Arc::new(table.with_order(OrderHint {
        column: column.name.clone(),
        descending: !sort_expr.asc,
        fetch: *fetch,
    }));
    let source = provider_as_source(hinted as Arc<dyn TableProvider>);
    Ok(Some(with_scan_source(plan, &source)?))
}

/// Custom `ExecutionPlan` that runs an unranked match kernel on the
/// query runtime inside `execute()` and emits the resolved `_id` +
/// scalar columns + (constant) `score`.
struct MatchExec {
    reader: Arc<SupertableReader>,
    column: String,
    query: MatchQuery,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    /// The pushed-down `WHERE`, used only to read less.
    filters: Vec<Expr>,
    /// The `ORDER BY column LIMIT n` above, when the walk can honour it.
    order: Option<OrderHint>,
    cache: Arc<PlanProperties>,
}

impl MatchExec {
    fn try_new(
        reader: Arc<SupertableReader>,
        column: String,
        query: MatchQuery,
        scalar_schema: SchemaRef,
        output_schema: SchemaRef,
        projection: Option<Vec<usize>>,
    ) -> DfResult<Self> {
        let projected_schema = match &projection {
            Some(indices) => Arc::new(
                output_schema
                    .project(indices)
                    .map_err(QueryError::internal)?,
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
            scalar_schema,
            output_schema,
            projection,
            projected_schema,
            filters: Vec::new(),
            order: None,
            cache,
        })
    }

    fn with_filters(mut self, filters: Vec<Expr>) -> Self {
        self.filters = filters;
        self
    }

    fn with_order(mut self, order: Option<OrderHint>) -> Self {
        self.order = order;
        self
    }

    fn describe(&self) -> String {
        let text = match &self.query {
            MatchQuery::Token { mode, .. } => {
                format!(
                    "MatchExec: kind=token, column={}, mode={:?}",
                    self.column, mode
                )
            }
            MatchQuery::Exact { .. } => {
                format!("MatchExec: kind=exact, column={}", self.column)
            }
        };
        let text = match &self.order {
            Some(hint) => format!(
                "{text}, order={} {} fetch={}",
                hint.column,
                if hint.descending { "desc" } else { "asc" },
                hint.fetch
            ),
            None => text,
        };
        if self.filters.is_empty() {
            text
        } else {
            format!("{text}, filters={}", self.filters.len())
        }
    }
}

impl fmt::Debug for MatchExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for MatchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for MatchExec {
    fn name(&self) -> &'static str {
        "MatchExec"
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
                "MatchExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let column = self.column.clone();
        let query = self.query.clone();
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);
        let filters = self.filters.clone();
        let order_hint = self.order.clone();

        // What the consumer reads decides how much of a hit to build. A
        // projection of no columns at all is `COUNT(*)`: the count
        // primitive answers it without a hit per row. One that leaves out
        // `_id` (an aggregate over scalar columns) skips stamping stable
        // ids, which reads an `_id` column only on a cell-packed or
        // gapped-span file; elsewhere the hits arrive already stamped.
        let id_column = reader.options().id_column.clone();
        let id_idx = scalar_schema.index_of(&id_column).ok();
        let needs_ids = match (&projection, id_idx) {
            (None, _) => true,
            (Some(indices), Some(id_idx)) => indices.contains(&id_idx),
            (Some(_), None) => false,
        };
        // A count with a filter takes the hit walk below, which prunes by it;
        // so does a match under an `ORDER BY ... LIMIT`, which walks in order.
        let counts_only = filters.is_empty()
            && order_hint.is_none()
            && matches!(&projection, Some(indices) if indices.is_empty());
        let score_idx = scalar_schema.fields().len();
        let needs_score = match &projection {
            None => true,
            Some(indices) => indices.contains(&score_idx),
        };
        let span = match &self.query {
            MatchQuery::Token { .. } => {
                detail_span!("tvf.token_match", rows_out = tracing::field::Empty)
            }
            MatchQuery::Exact { .. } => {
                detail_span!("tvf.exact_match", rows_out = tracing::field::Empty)
            }
        };

        // Scalar columns only, neither `_id` nor the score: each
        // superfile's matching rows come out as their own batch, straight
        // from the walk to a column take, and the consumer (an aggregate,
        // a filter) takes them in any order. The projection may name a
        // column twice; the batches carry each distinct column once and
        // are laid out to the projection here.
        if !counts_only
            && !needs_ids
            && !needs_score
            && order_hint.is_none()
            && let (Some(indices), MatchQuery::Token { query, mode }) =
                (projection.clone(), query.clone())
        {
            let columns_schema = Arc::clone(&projected_schema);
            let fut = async move {
                let mut names: Vec<&str> = Vec::new();
                for &p in &indices {
                    let name = scalar_schema.field(p).name().as_str();
                    if !names.contains(&name) {
                        names.push(name);
                    }
                }
                let per_file = reader
                    .token_match_columns_async(&column, &query, mode, &names, &filters)
                    .await?;
                let mut out = Vec::with_capacity(per_file.len());
                for batch in per_file {
                    let columns: Vec<_> = indices
                        .iter()
                        .map(|&p| {
                            let name = scalar_schema.field(p).name();
                            let idx = batch
                                .schema()
                                .index_of(name)
                                .map_err(QueryError::internal)?;
                            Ok::<_, QueryError>(Arc::clone(batch.column(idx)))
                        })
                        .collect::<Result<_, _>>()?;
                    out.push(
                        RecordBatch::try_new_with_options(
                            Arc::clone(&columns_schema),
                            columns,
                            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
                        )
                        .map_err(QueryError::internal)?,
                    );
                }
                Ok::<Vec<RecordBatch>, DataFusionError>(out)
            };
            let fut = traced_tvf_rows(span, fut, |batches: &Vec<RecordBatch>| {
                batches.iter().map(RecordBatch::num_rows).sum()
            });
            let stream = stream::once(fut)
                .map_ok(|batches| stream::iter(batches.into_iter().map(Ok)))
                .try_flatten();
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                projected_schema,
                stream,
            )));
        }
        let count_schema = Arc::clone(&projected_schema);

        let fut = async move {
            if counts_only && let MatchQuery::Token { query, mode } = &query {
                let n = reader
                    .token_match_count_async(&column, query, *mode)
                    .await?;
                let n = usize::try_from(n).map_err(QueryError::internal)?;
                return RecordBatch::try_new_with_options(
                    count_schema,
                    Vec::new(),
                    &RecordBatchOptions::new().with_row_count(Some(n)),
                )
                .map_err(|e| QueryError::internal(e).into());
            }
            // Under an `ORDER BY column LIMIT n` the walk can honour, the
            // match opens its superfiles in the column's order and stops
            // once none left can place a row.
            let order = match &order_hint {
                Some(hint) => {
                    let predicate = PushedPredicate::compile(&filters, &output_schema);
                    MatchOrder::new(
                        &reader,
                        &hint.column,
                        hint.descending,
                        hint.fetch,
                        predicate.as_ref(),
                        &output_schema,
                    )?
                    .map(Arc::new)
                }
                None => None,
            };
            let hits = match &query {
                MatchQuery::Token { query, mode } => {
                    reader
                        .token_match_hits_async(
                            &column,
                            query,
                            *mode,
                            needs_ids,
                            &filters,
                            order.as_ref(),
                        )
                        .await
                }
                MatchQuery::Exact { value } => {
                    reader
                        .exact_match_async(&column, value, &filters, order.as_ref())
                        .await
                }
            }?;
            resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                projection.as_deref(),
            )
            .await
        };
        let fut = traced_tvf(span, fut);
        let stream = stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::{fmt, sync::Arc};

    use arrow::util::display::array_value_to_string;
    use arrow_array::{Array, Date32Array, Int64Array, LargeStringArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema, TimeUnit};
    use datafusion::{
        error::DataFusionError,
        execution::TaskContext,
        physical_plan::{DisplayFormatType, ExecutionPlan},
        prelude::{col, lit},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;
    use tokio::runtime::Runtime;

    use super::{EXACT_MATCH_UDTF, ExactMatchFunc, MatchExec, MatchQuery, TokenMatchFunc};
    use crate::supertable::query::match_order::ORDERED_WINDOW;
    use crate::{
        Connection, FieldPatch, IndexSpec, SchemaPatch, connect,
        runtime_metrics::op_stats::with_op_stats,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{builder::FtsConfig, fts::reader::BoolMode},
        supertable::{
            Supertable, SupertableOptions,
            error::QueryError,
            handle::SupertableReader,
            query::exec::common::{output_schema_with_score, test_support::sql_row_count},
        },
    };

    fn title_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]))
    }

    fn options_title_fts() -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], vec![])
            .expect("valid options")
            .with_writer_pool(pool)
    }

    fn titles_batch(titles: &[&str]) -> RecordBatch {
        let arr = LargeStringArray::from(titles.to_vec());
        RecordBatch::try_new(title_schema(), vec![Arc::new(arr)]).expect("batch")
    }

    fn supertable_with_titles(titles: &[&str]) -> Supertable {
        let st = Supertable::create(options_title_fts()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&titles_batch(titles)).expect("append");
        w.commit().expect("commit");
        st
    }

    fn demo() -> Supertable {
        supertable_with_titles(&[
            "rust async runtime",       // 0
            "python data science",      // 1
            "rust systems programming", // 2
            "go routines",              // 3
        ])
    }

    /// Queries whose aggregates are held to the materialized rows: a bare
    /// term (the one the term index can count alone), a union, a `+must`
    /// clause, and a term no row holds.
    const AGGREGATED_QUERIES: [&str; 4] = ["rust", "rust python", "+rust systems", "nothing_here"];
    /// A table with storage written over several commits, and the row its
    /// delete removes: one the bare term matches, so a count that kept it
    /// would disagree with the rows.
    const STORED_COMMITS: [&[&str]; 3] = [
        &["rust async runtime", "python data science"],
        &["rust systems programming", "go routines"],
        &["rust rust rust", "python rust bindings"],
    ];
    const DELETED_TITLE: &str = "rust systems programming";

    /// `COUNT(*)` over the match function projects no column and is
    /// answered by the count primitive; it must equal the row count the
    /// materialized match gives, and an aggregate over a scalar column
    /// (which projects no `_id`), a grouped aggregate and a projection
    /// naming a column twice must equal the same over the materialized
    /// rows.
    fn assert_aggregates_equal_the_rows(st: &Supertable) {
        for query in AGGREGATED_QUERIES {
            let tvf = format!("token_match('title', '{query}')");
            let counted = sql_row_count(st, &format!("SELECT _id FROM {tvf}"));
            let batches = st
                .reader()
                .expect("reader")
                .query_sql(&format!("SELECT count(*) AS n FROM {tvf}"))
                .expect("count(*)");
            let n = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count is Int64")
                .value(0);
            assert_eq!(n as usize, counted, "count(*) over {tvf}");
            // A scalar aggregate projects `title` and no `_id`: the hits
            // leave unstamped and the rows must still be the matched ones.
            let longest = st
                .reader()
                .expect("reader")
                .query_sql(&format!("SELECT max(length(title)) AS m FROM {tvf}"))
                .expect("max over the match");
            let via_rows = st
                .reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT max(length(title)) AS m FROM (SELECT title FROM {tvf} ORDER BY _id)"
                ))
                .expect("max over the rows");
            assert_eq!(
                format!("{:?}", longest[0].column(0)),
                format!("{:?}", via_rows[0].column(0)),
                "max over {tvf}"
            );
            // A grouped aggregate and a projection naming a column twice go
            // the same per-superfile way; the rows must still be the matched
            // ones under every grouping.
            let grouped = st
                .reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT length(title) AS l, count(*) AS n, min(title) AS t FROM {tvf} \
                     GROUP BY length(title) ORDER BY l"
                ))
                .expect("grouped over the match");
            let grouped_via_rows = st
                .reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT length(title) AS l, count(*) AS n, min(title) AS t \
                     FROM (SELECT title FROM {tvf} ORDER BY _id) GROUP BY length(title) ORDER BY l"
                ))
                .expect("grouped over the rows");
            assert_eq!(
                format!("{grouped:?}"),
                format!("{grouped_via_rows:?}"),
                "grouped aggregate over {tvf}"
            );
            let twice = sql_row_count(st, &format!("SELECT title AS a, title AS b FROM {tvf}"));
            assert_eq!(twice, counted, "a column named twice over {tvf}");
        }
    }

    #[test]
    fn aggregates_over_the_match_function_equal_the_materialized_rows() {
        let st = supertable_with_titles(&[
            "rust async runtime",
            "python data science",
            "rust systems programming",
            "go routines",
            "rust rust rust",
        ]);
        assert_aggregates_equal_the_rows(&st);
    }

    /// The same agreement on a table with storage, which keeps a term
    /// index: over several commits a bare term is counted from the
    /// document frequencies the index holds, and once a delete lands the
    /// count falls back to the fan-out, which leaves the deleted row out.
    #[test]
    fn aggregates_over_the_match_function_on_stored_superfiles_equal_the_rows() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_title_fts().with_storage(storage)).expect("create");
        for titles in STORED_COMMITS {
            let mut w = st.writer().expect("writer");
            w.append(&titles_batch(titles)).expect("append");
            w.commit().expect("commit");
        }
        let index = Runtime::new()
            .expect("runtime")
            .block_on(st.reader().expect("reader").manifest().term_index())
            .expect("term index readable");
        assert!(index.is_some(), "a table with storage keeps a term index");
        assert_aggregates_equal_the_rows(&st);

        let stats = st
            .delete(col("title").eq(lit(DELETED_TITLE)))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        assert_aggregates_equal_the_rows(&st);
    }

    /// Threads of the reader pool the sliced-match fixture reads with: more
    /// than one, so a large superfile's match is cut into doc-id slices.
    const SLICING_READER_THREADS: usize = 4;
    /// Rows of the sliced-match fixture's one superfile: enough for three
    /// slices at the fewest docs a slice is cut at.
    const SLICED_ROWS: usize = 150_000;
    /// Plain unions of terms, the shape whose match is sliced.
    const SLICED_QUERIES: [&str; 2] = ["rust", "rust python"];
    /// A conjunction of two terms, which walks its superfile whole.
    const WHOLE_QUERY: &str = "rust systems";

    /// The rows of each batch the column path gives for `query`: one batch
    /// per superfile, or per slice of one.
    fn column_batch_rows(st: &Supertable, query: &str, mode: BoolMode) -> Vec<usize> {
        let reader = st.reader().expect("reader");
        reader
            .block_on(reader.token_match_columns_async("title", query, mode, &["title"], &[]))
            .expect("columns over the match")
            .iter()
            .map(RecordBatch::num_rows)
            .collect()
    }

    /// The fixture's one superfile is really cut: a plain union comes back
    /// as several slices, together every matched row once, and a
    /// conjunction as one batch.
    fn assert_the_match_is_sliced(st: &Supertable) {
        assert_eq!(st.reader().expect("reader").n_superfiles(), 1);
        for query in SLICED_QUERIES {
            let slices = column_batch_rows(st, query, BoolMode::Or);
            assert!(slices.len() > 1, "{query}: {slices:?} is one batch");
            let matched = sql_row_count(
                st,
                &format!("SELECT _id FROM token_match('title', '{query}')"),
            );
            assert_eq!(slices.iter().sum::<usize>(), matched, "{query}");
        }
        assert_eq!(column_batch_rows(st, WHOLE_QUERY, BoolMode::And).len(), 1);
    }

    /// The same agreement when one superfile is large enough, and the
    /// reader pool wide enough, for an aggregate's match to be cut into
    /// doc-id slices walked side by side: before and after a delete, whose
    /// row each slice must leave out on its own.
    #[test]
    fn aggregates_over_a_sliced_match_equal_the_rows() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let reader_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(SLICING_READER_THREADS)
                .build()
                .expect("reader pool"),
        );
        let st = Supertable::create(
            options_title_fts()
                .with_storage(storage)
                .with_reader_pool(reader_pool),
        )
        .expect("create");
        let titles: Vec<&str> = STORED_COMMITS
            .iter()
            .flat_map(|commit| commit.iter().copied())
            .cycle()
            .take(SLICED_ROWS)
            .collect();
        {
            let mut w = st.writer().expect("writer");
            w.append(&titles_batch(&titles)).expect("append");
            w.commit().expect("commit");
        }
        assert_the_match_is_sliced(&st);
        assert_aggregates_equal_the_rows(&st);

        st.delete(col("title").eq(lit(DELETED_TITLE)))
            .expect("delete");
        assert_the_match_is_sliced(&st);
        assert_aggregates_equal_the_rows(&st);
    }

    /// Two superfiles, one per `bucket`: bucket 0 then bucket 1. Every
    /// title holds `rust`, and each bucket has one title that is exactly
    /// `rust`.
    fn two_bucket_superfiles() -> Supertable {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]));
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let options =
            SupertableOptions::new(Arc::clone(&schema), vec![FtsConfig::new("title")], vec![])
                .expect("valid options")
                .with_writer_pool(pool);
        let st = Supertable::create(options).expect("create");
        for (bucket, titles) in [(0_i64, ["rust", "rust zero"]), (1, ["rust", "rust one"])] {
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(titles.to_vec())),
                    Arc::new(Int64Array::from(vec![bucket; titles.len()])),
                ],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        st
    }

    /// A `WHERE` on the match functions reaches the scan, and the rows
    /// stay what the filter keeps.
    #[test]
    fn where_on_match_functions_is_pushed_to_the_scan() {
        let st = two_bucket_superfiles();
        let cases = [
            ("SELECT title, bucket FROM token_match('title', 'rust')", 2),
            (
                "SELECT _id, title, bucket FROM token_match('title', 'rust')",
                2,
            ),
            ("SELECT title, bucket FROM exact_match('title', 'rust')", 1),
        ];
        for (sql, rows_in_bucket) in cases {
            assert_eq!(
                sql_row_count(&st, sql),
                2 * rows_in_bucket,
                "premise: {sql}"
            );
            let filtered = format!("{sql} WHERE bucket = 1");
            assert_eq!(sql_row_count(&st, &filtered), rows_in_bucket, "{filtered}");
            let plan = explain(&st, &filtered);
            assert!(plan.contains("filters=1"), "filter not pushed: {plan}");
            assert_eq!(sql_row_count(&st, &format!("{sql} WHERE bucket = 7")), 0);
        }
    }

    /// Given the `WHERE`, every match kernel opens only the superfiles whose
    /// statistics can pass it: the hit walk, the column take and the exact
    /// match.
    #[test]
    fn match_kernels_open_only_the_superfiles_the_where_admits() {
        let st = two_bucket_superfiles();
        let rt = Runtime::new().expect("runtime");
        // Rows out and superfiles opened by each kernel, under `filter`.
        let run = |filter: Option<i64>| {
            let (rows, stats) = with_op_stats(|| {
                let reader = st.reader().expect("reader");
                rt.block_on(async {
                    let filters: Vec<_> = filter
                        .map(|bucket| col("bucket").eq(lit(bucket)))
                        .into_iter()
                        .collect();
                    let hits = reader
                        .token_match_hits_async("title", "rust", BoolMode::Or, true, &filters, None)
                        .await
                        .expect("hits");
                    let batches = reader
                        .token_match_columns_async(
                            "title",
                            "rust",
                            BoolMode::Or,
                            &["title"],
                            &filters,
                        )
                        .await
                        .expect("columns");
                    let exact = reader
                        .exact_match_async("title", "rust", &filters, None)
                        .await
                        .expect("exact");
                    let taken: usize = batches.iter().map(RecordBatch::num_rows).sum();
                    (hits.len(), taken, exact.len())
                })
            });
            (rows, stats.score_pruning_survived)
        };
        assert_eq!(run(None), ((4, 4, 2), 6), "premise: each kernel opens both");
        assert_eq!(run(Some(1)), ((2, 2, 1), 3), "each kernel opens one");
        assert_eq!(run(Some(7)), ((0, 0, 0), 0), "no superfile holds bucket 7");
    }

    /// Name of the table the metered SQL fixtures register.
    const METERED_TABLE: &str = "docs";

    /// A `memory://` connection holding [`METERED_TABLE`] with `spec`, and
    /// one superfile per batch of `commits`. Its `query_sql` meters the
    /// work a match function does; `Supertable::query_sql` does not.
    fn metered_table(schema: &Arc<Schema>, spec: IndexSpec, commits: &[RecordBatch]) -> Connection {
        let conn = connect("memory://").expect("connect");
        let table = conn
            .create_table(METERED_TABLE, Arc::clone(schema), spec)
            .expect("create");
        for batch in commits {
            table.append(batch).expect("append");
        }
        conn
    }

    /// The SQL of each match path a `WHERE` reaches: the column take, the
    /// hit walk (`_id`) and the exact match, each over `term` on `title`.
    fn match_path_sqls(term: &str) -> [String; 3] {
        [
            format!("SELECT title FROM token_match('{METERED_TABLE}', 'title', '{term}')"),
            format!("SELECT _id FROM token_match('{METERED_TABLE}', 'title', '{term}')"),
            format!("SELECT _id FROM exact_match('{METERED_TABLE}', 'title', '{term}')"),
        ]
    }

    /// Rows `sql` returns and the superfiles it opened.
    fn metered_rows(conn: &Connection, sql: &str) -> (usize, u64) {
        let (rows, stats) = with_op_stats(|| {
            conn.query_sql(sql)
                .expect("query_sql")
                .iter()
                .map(RecordBatch::num_rows)
                .sum()
        });
        (rows, stats.score_pruning_survived)
    }

    /// One batch of `title` and a second column `other`.
    fn titled_batch(schema: &Arc<Schema>, titles: &[&str], other: Arc<dyn Array>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![Arc::new(LargeStringArray::from(titles.to_vec())), other],
        )
        .expect("batch")
    }

    /// A `WHERE` on a match function in SQL reaches the scan: every match
    /// path opens only the superfile whose statistics can pass it, and
    /// none when no superfile can.
    #[test]
    fn sql_where_on_match_functions_opens_only_the_superfiles_it_admits() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]));
        let commits: Vec<_> = [(0_i64, ["rust", "rust zero"]), (1, ["rust", "rust one"])]
            .into_iter()
            .map(|(bucket, titles)| {
                titled_batch(
                    &schema,
                    &titles,
                    Arc::new(Int64Array::from(vec![bucket; 2])),
                )
            })
            .collect();
        let conn = metered_table(&schema, IndexSpec::new().fts("title"), &commits);
        for sql in match_path_sqls("rust") {
            let per_bucket = if sql.contains(EXACT_MATCH_UDTF) { 1 } else { 2 };
            assert_eq!(
                metered_rows(&conn, &sql),
                (2 * per_bucket, 2),
                "premise: {sql}"
            );
            let one = format!("{sql} WHERE bucket = 1");
            assert_eq!(metered_rows(&conn, &one), (per_bucket, 1), "{one}");
            let none = format!("{sql} WHERE bucket = 7");
            assert_eq!(metered_rows(&conn, &none), (0, 0), "{none}");
        }
    }

    /// A `WHERE` on an indexed column opens no superfile beyond what the
    /// match's own terms keep: no index is read to build the scope.
    #[test]
    fn sql_where_on_an_indexed_column_opens_no_extra_superfile() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("tag", DataType::LargeUtf8, false),
        ]));
        // Both superfiles are tagged `common`; only the first holds `rare`.
        let commits: Vec<_> = [["rare", "plain"], ["plain", "other"]]
            .iter()
            .map(|titles| {
                titled_batch(
                    &schema,
                    titles,
                    Arc::new(LargeStringArray::from(vec!["common"; 2])),
                )
            })
            .collect();
        let conn = metered_table(&schema, IndexSpec::new().fts("title").fts("tag"), &commits);
        for sql in match_path_sqls("rare") {
            let filtered = format!("{sql} WHERE tag = 'common'");
            assert_eq!(metered_rows(&conn, &filtered), (1, 1), "{filtered}");
        }
        for sql in match_path_sqls("zyzzyva") {
            let filtered = format!("{sql} WHERE tag = 'common'");
            assert_eq!(metered_rows(&conn, &filtered), (0, 0), "{filtered}");
        }
    }

    /// `WHERE` shapes that reach statistics pruning on the match paths keep
    /// exactly the rows they should: the synthetic `score`, `_id`, `OR`
    /// branches, and a column added after the superfiles were written.
    #[test]
    fn sql_where_shapes_keep_their_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]));
        let commits: Vec<_> = [(0_i64, ["rust", "rust zero"]), (1, ["rust", "rust one"])]
            .into_iter()
            .map(|(bucket, titles)| {
                titled_batch(
                    &schema,
                    &titles,
                    Arc::new(Int64Array::from(vec![bucket; 2])),
                )
            })
            .collect();
        let conn = metered_table(&schema, IndexSpec::new().fts("title"), &commits);
        conn.apply_schema(
            METERED_TABLE,
            &SchemaPatch::new(vec![FieldPatch::named("added").with_type(DataType::Int64)]),
            None,
        )
        .expect("add a column");
        let id_batches = conn
            .query_sql(&format!(
                "SELECT _id FROM {METERED_TABLE} WHERE bucket = 1 AND title = 'rust'"
            ))
            .expect("id");
        let id = array_value_to_string(id_batches[0].column(0), 0).expect("id text");
        // `rust` is in all four titles, and is exactly one title per bucket.
        // Each case: projection, `WHERE`, rows for token_match, exact_match.
        let cases = [
            ("_id", "score >= 0".to_owned(), 4, 2),
            ("_id", "score < 0".to_owned(), 0, 0),
            ("_id", format!("_id = {id}"), 1, 1),
            ("title", format!("_id = {id}"), 1, 1),
            ("title", "bucket = 0 OR bucket = 1".to_owned(), 4, 2),
            ("title", "bucket = 7 OR title = 'rust one'".to_owned(), 1, 0),
            ("title", "added IS NULL".to_owned(), 4, 2),
            ("_id", "added IS NULL AND bucket = 1".to_owned(), 2, 1),
            ("title", "added = 3".to_owned(), 0, 0),
        ];
        for (projection, filter, token_rows, exact_rows) in cases {
            for (func, rows) in [("token_match", token_rows), ("exact_match", exact_rows)] {
                let sql = format!(
                    "SELECT {projection} FROM {func}('{METERED_TABLE}', 'title', 'rust') \
                     WHERE {filter}"
                );
                assert_eq!(metered_rows(&conn, &sql).0, rows, "{sql}");
            }
        }
    }

    /// A `WHERE` on a range-indexed column retyped after its superfile was
    /// written still finds the row it keeps, on every match path.
    #[test]
    fn sql_where_on_a_retyped_range_column_keeps_its_rows() {
        let cases: [(Arc<dyn Array>, DataType, &str, &str); 2] = [
            (
                Arc::new(Int64Array::from(vec![1_i64, 2])),
                DataType::Float64,
                "1.0",
                "7.0",
            ),
            (
                Arc::new(Date32Array::from(vec![1, 2])),
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                "TIMESTAMP '1970-01-02 00:00:00'",
                "TIMESTAMP '1970-01-08 00:00:00'",
            ),
        ];
        for (values, new_type, kept, absent) in cases {
            let schema = Arc::new(Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, false),
                Field::new("bucket", values.data_type().clone(), false),
            ]));
            let batch = titled_batch(&schema, &["rust", "rust one"], values);
            let conn = metered_table(
                &schema,
                IndexSpec::new().fts("title").range("bucket"),
                &[batch],
            );
            conn.apply_schema(
                METERED_TABLE,
                &SchemaPatch::new(vec![
                    FieldPatch::named("bucket").with_type(new_type.clone()),
                ]),
                None,
            )
            .expect("retype");
            for sql in match_path_sqls("rust") {
                let one = format!("{sql} WHERE bucket = {kept}");
                assert_eq!(metered_rows(&conn, &one).0, 1, "{new_type}: {one}");
                let none = format!("{sql} WHERE bucket = {absent}");
                assert_eq!(metered_rows(&conn, &none).0, 0, "{new_type}: {none}");
            }
        }
    }

    #[test]
    fn token_match_tvf_or_unions_and_intersects() {
        let st = demo();
        // OR (default): rows containing `rust` OR `python` → 0, 1, 2.
        assert_eq!(
            sql_row_count(&st, "SELECT _id FROM token_match('title', 'rust python')"),
            3
        );
        // AND: rows containing both `rust` and `systems` → only doc 2.
        assert_eq!(
            sql_row_count(
                &st,
                "SELECT _id FROM token_match('title', 'rust systems', 'and')"
            ),
            1
        );
    }

    #[test]
    fn exact_match_tvf_matches_only_exact_value() {
        let st = supertable_with_titles(&["rust async", "rust async runtime"]);
        // Both rows contain {rust, async}, but only one equals the string.
        assert_eq!(
            sql_row_count(&st, "SELECT _id FROM exact_match('title', 'rust async')"),
            1
        );
        assert_eq!(
            sql_row_count(
                &st,
                "SELECT _id FROM exact_match('title', 'rust async runtime')"
            ),
            1
        );
        assert_eq!(
            sql_row_count(&st, "SELECT _id FROM exact_match('title', 'rust')"),
            0
        );
    }

    #[test]
    fn token_match_tvf_star_projection_appends_score() {
        let st = demo();
        let batches = st
            .reader()
            .expect("reader")
            .query_sql("SELECT * FROM token_match('title', 'rust')")
            .expect("query_sql");
        let b = &batches[0];
        // scalar schema (_id, title) + score.
        assert_eq!(b.num_columns(), 3);
        assert_eq!(b.schema().field(2).name(), "score");
    }

    #[test]
    fn match_tvf_arity_errors() {
        let st = demo();
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM token_match('title')")
                .is_err(),
            "token_match needs >= 2 args"
        );
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM exact_match('title')")
                .is_err(),
            "exact_match needs 2 args"
        );
    }

    #[test]
    fn public_methods_agree_with_tvfs() {
        let st = demo();
        let reader = st.reader().expect("reader");
        let method = reader
            .token_match("title", "rust systems", BoolMode::And)
            .expect("token_match");
        assert_eq!(method.len(), 1);
        let exact = reader
            .exact_match("title", "go routines")
            .expect("exact_match");
        assert_eq!(exact.len(), 1);
    }

    /// Flatten an `EXPLAIN` result into one searchable string — exercises
    /// `MatchExec`'s `DisplayAs`/`describe`.
    fn explain(st: &Supertable, sql: &str) -> String {
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(&format!("EXPLAIN {sql}"))
            .expect("explain");
        let mut out = String::new();
        for batch in &batches {
            for column in batch.columns() {
                if let Some(strings) = column.as_any().downcast_ref::<StringArray>() {
                    for i in 0..strings.len() {
                        if !strings.is_null(i) {
                            out.push_str(strings.value(i));
                            out.push('\n');
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn match_exec_display_describes_token_and_exact_branches() {
        let st = demo();
        let token = explain(&st, "SELECT _id FROM token_match('title', 'rust', 'and')");
        assert!(
            token.contains("MatchExec") && token.contains("kind=token") && token.contains("And"),
            "token describe missing: {token}"
        );
        let exact = explain(
            &st,
            "SELECT _id FROM exact_match('title', 'rust async runtime')",
        );
        assert!(
            exact.contains("MatchExec") && exact.contains("kind=exact"),
            "exact describe missing: {exact}"
        );
    }

    /// Build an `Arc<SupertableReader>` plus the scalar + output
    /// schemas a `MatchExec` needs, from a committed demo table.
    fn reader_and_schemas() -> (Arc<SupertableReader>, Arc<Schema>, Arc<Schema>) {
        let st = demo();
        let reader = Arc::new(st.reader().expect("reader"));
        let scalar_schema = reader.manifest().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);
        (reader, scalar_schema, output_schema)
    }

    #[test]
    fn match_exec_try_new_rejects_out_of_range_projection() {
        // An index past the end of the output schema fails the
        // `output_schema.project(indices)` call inside `try_new`.
        let (reader, scalar_schema, output_schema) = reader_and_schemas();
        let n_cols = output_schema.fields().len();
        let err = MatchExec::try_new(
            reader,
            "title".into(),
            MatchQuery::Exact {
                value: "rust".into(),
            },
            scalar_schema,
            output_schema,
            Some(vec![n_cols + 5]),
        )
        .expect_err("out-of-range projection must fail");
        // DataFusion only hands in valid indices, so this is our invariant,
        // carried through the plan typed.
        assert!(
            matches!(
                &err,
                DataFusionError::External(e)
                    if matches!(e.downcast_ref::<QueryError>(), Some(QueryError::Internal(_)))
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn match_exec_plan_metadata_and_children() {
        let (reader, scalar_schema, output_schema) = reader_and_schemas();
        let exec = MatchExec::try_new(
            reader,
            "title".into(),
            MatchQuery::Token {
                query: "rust".into(),
                mode: BoolMode::Or,
            },
            scalar_schema,
            output_schema,
            None,
        )
        .expect("try_new");

        assert_eq!(exec.name(), "MatchExec");
        // Single leaf node — no children.
        assert!(exec.children().is_empty());
        // PlanProperties is exposed; single partition.
        let _ = exec.properties();
        // `as_any` downcasts back to the concrete type.
        let arc: Arc<dyn ExecutionPlan> = Arc::new(exec);
        assert!(arc.downcast_ref::<MatchExec>().is_some());

        // `with_new_children` returns the same node (it has none).
        let same = Arc::clone(&arc)
            .with_new_children(vec![])
            .expect("with_new_children");
        assert_eq!(same.name(), "MatchExec");
    }

    #[test]
    fn match_exec_execute_rejects_nonzero_partition() {
        let (reader, scalar_schema, output_schema) = reader_and_schemas();
        let exec = MatchExec::try_new(
            reader,
            "title".into(),
            MatchQuery::Exact {
                value: "rust".into(),
            },
            scalar_schema,
            output_schema,
            None,
        )
        .expect("try_new");
        let ctx = Arc::new(TaskContext::default());
        match exec.execute(1, ctx) {
            Err(DataFusionError::Internal(_)) => {}
            Err(other) => panic!("expected Internal error, got {other:?}"),
            Ok(_) => panic!("nonzero partition must error"),
        }
    }

    #[test]
    fn match_exec_debug_and_display_render_describe() {
        let (reader, scalar_schema, output_schema) = reader_and_schemas();
        let exec = MatchExec::try_new(
            reader,
            "title".into(),
            MatchQuery::Token {
                query: "rust".into(),
                mode: BoolMode::And,
            },
            scalar_schema,
            output_schema,
            None,
        )
        .expect("try_new");
        // `Debug` routes through `describe`.
        let dbg = format!("{exec:?}");
        assert!(
            dbg.contains("MatchExec") && dbg.contains("kind=token") && dbg.contains("And"),
            "got {dbg}"
        );
        // `DisplayAs::fmt_as` also routes through `describe`. Drive it
        // directly via a tiny `Display` adapter.
        struct D<'a>(&'a MatchExec);
        impl fmt::Display for D<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                use datafusion::physical_plan::DisplayAs;
                self.0.fmt_as(DisplayFormatType::Default, f)
            }
        }
        let shown = format!("{}", D(&exec));
        assert!(
            shown.contains("MatchExec") && shown.contains("kind=token"),
            "got {shown}"
        );
    }

    #[test]
    fn token_and_exact_func_call_reject_bad_arity() {
        // Direct `TableFunctionImpl::call` arity checks, without the
        // SQL layer. A live reader keeps the WeakReader upgrade-able.
        use crate::supertable::query::exec::common::test_support::call_tvf;
        let st = demo();
        let reader = Arc::new(st.reader().expect("reader"));
        let scalar_schema = reader.manifest().scalar_schema();
        let tf = TokenMatchFunc::new(Arc::clone(&reader), Arc::clone(&scalar_schema));
        // 1 arg → error; 2 args → ok.
        assert!(call_tvf(&tf, &[]).is_err(), "0 args must fail");
        let ef = ExactMatchFunc::new(reader, scalar_schema);
        assert!(call_tvf(&ef, &[]).is_err(), "0 args must fail");
    }

    /// Construct `MatchTable` directly through the `token_match` TVF
    /// `call` path and exercise its `TableProvider` metadata methods
    /// (`Debug`, `as_any`, `table_type`) — none of which normal query
    /// execution touches.
    #[test]
    fn match_table_trait_methods() {
        use datafusion::{logical_expr::TableType, prelude::lit};

        use super::MatchTable;
        use crate::supertable::query::exec::common::test_support::{call_tvf, scoped_inner};

        let st = demo();
        let reader = Arc::new(st.reader().expect("reader"));
        let scalar_schema = reader.manifest().scalar_schema();
        let func = TokenMatchFunc::new(reader, scalar_schema);
        let table = call_tvf(&func, &[lit("title"), lit("rust")]).expect("match table");

        let dbg = format!("{table:?}");
        assert!(dbg.contains("MatchTable"), "Debug missing: {dbg}");
        assert!(
            scoped_inner(&table).downcast_ref::<MatchTable>().is_some(),
            "as_any downcasts to MatchTable"
        );
        assert_eq!(table.table_type(), TableType::Base);
    }

    #[test]
    fn match_tvf_bad_arg_types_error() {
        let st = demo();
        // Non-string column literal.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM token_match(5, 'rust')")
                .is_err(),
            "non-string column must error"
        );
        // Bad mode value (third arg).
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM token_match('title', 'rust', 'xor')")
                .is_err(),
            "invalid bool mode must error"
        );
    }

    /// Superfiles of the hourly match table, one commit each, hours ascending.
    const HOURLY_FILES: i64 = 24;
    /// Leading hours the rare monument is viewed in.
    const MONUMENT_HOURS: i64 = 5;
    /// Filler pages per hour.
    const HOURLY_FILLER: usize = 20;

    /// A page-view log: per hour, "Zyzzyva" in two languages, the monument
    /// in its first hours, and filler — one superfile per hour, so `hour`
    /// orders the superfiles as time orders a log's.
    fn hourly_conn() -> Connection {
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("lang", DataType::Utf8, false),
            Field::new("hour", DataType::Int64, false),
        ]));
        let commits: Vec<RecordBatch> = (0..HOURLY_FILES)
            .map(|hour| {
                let mut titles: Vec<String> = vec!["Zyzzyva".into(), "Zyzzyva".into()];
                let mut langs: Vec<&str> = vec!["en", "de"];
                if hour < MONUMENT_HOURS {
                    titles.push("Hunnestad Monument".into());
                    langs.push("sv");
                }
                for i in 0..HOURLY_FILLER {
                    titles.push(format!("page {hour} {i}"));
                    langs.push("en");
                }
                let n = titles.len();
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(LargeStringArray::from(
                            titles.iter().map(String::as_str).collect::<Vec<_>>(),
                        )),
                        Arc::new(StringArray::from(langs)),
                        Arc::new(Int64Array::from(vec![hour; n])),
                    ],
                )
                .expect("batch")
            })
            .collect();
        metered_table(&schema, IndexSpec::new().fts("title"), &commits)
    }

    /// The `hour`s `sql` returns, in its order.
    fn hours(conn: &Connection, sql: &str) -> Vec<i64> {
        conn.query_sql(sql)
            .expect("query_sql")
            .iter()
            .flat_map(|b| {
                let hours = b
                    .column_by_name("hour")
                    .expect("hour")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64")
                    .clone();
                hours.values().to_vec()
            })
            .collect()
    }

    /// `ORDER BY hour DESC LIMIT n` over a match opens the newest superfiles
    /// holding a match the `WHERE` keeps, and no more than a window past
    /// them; a second sort key, which the walk does not honour, opens every
    /// one and returns the same hours.
    #[test]
    fn token_match_order_by_time_desc_limit_opens_newest_only() {
        const N: usize = 3;
        let conn = hourly_conn();
        let ordered = format!(
            "SELECT hour, lang FROM token_match('{METERED_TABLE}', 'title', 'zyzzyva') \
             WHERE lang = 'en' ORDER BY hour DESC LIMIT {N}"
        );
        let unordered = format!(
            "SELECT hour, lang FROM token_match('{METERED_TABLE}', 'title', 'zyzzyva') \
             WHERE lang = 'en' ORDER BY hour DESC, lang LIMIT {N}"
        );
        assert_eq!(hours(&conn, &ordered), hours(&conn, &unordered));
        assert_eq!(hours(&conn, &ordered), vec![23, 22, 21]);
        let (rows, opened) = metered_rows(&conn, &ordered);
        assert_eq!(rows, N);
        assert!(
            opened <= (N + ORDERED_WINDOW) as u64,
            "opened {opened} superfiles for the newest {N}"
        );
        let (_, all) = metered_rows(&conn, &unordered);
        assert_eq!(all, HOURLY_FILES as u64, "a second sort key walks them all");
    }

    /// Ascending, through `exact_match`: the oldest superfiles first.
    #[test]
    fn exact_match_order_by_time_asc_limit_opens_oldest_only() {
        const N: usize = 2;
        let conn = hourly_conn();
        let sql = format!(
            "SELECT hour FROM exact_match('{METERED_TABLE}', 'title', 'Hunnestad Monument') \
             ORDER BY hour LIMIT {N}"
        );
        assert_eq!(hours(&conn, &sql), vec![0, 1]);
        let (_, opened) = metered_rows(&conn, &sql);
        assert!(
            opened <= (N + ORDERED_WINDOW) as u64,
            "opened {opened} superfiles for the oldest {N}"
        );
    }

    /// A `WHERE` the walk cannot check before ranking leaves the match to
    /// every superfile, and the answer as the sort gives it.
    #[test]
    fn order_by_time_under_an_uncheckable_where_walks_every_superfile() {
        let conn = hourly_conn();
        let sql = format!(
            "SELECT hour FROM token_match('{METERED_TABLE}', 'title', 'zyzzyva') \
             WHERE score >= 0 ORDER BY hour DESC LIMIT 2"
        );
        assert_eq!(hours(&conn, &sql), vec![23, 23]);
        let (_, opened) = metered_rows(&conn, &sql);
        assert_eq!(opened, HOURLY_FILES as u64);
    }

    /// The plan names the order the match walks in.
    #[test]
    fn explain_match_exec_carries_the_order() {
        let conn = hourly_conn();
        let plan = conn
            .query_sql(&format!(
                "EXPLAIN SELECT hour FROM token_match('{METERED_TABLE}', 'title', 'zyzzyva') \
                 ORDER BY hour DESC LIMIT 5"
            ))
            .expect("explain");
        let text: String = plan
            .iter()
            .flat_map(|b| {
                (0..b.num_rows())
                    .map(|i| array_value_to_string(b.column(1), i).expect("plan text"))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(text.contains("order=hour desc fetch=5"), "{text}");
    }
}
