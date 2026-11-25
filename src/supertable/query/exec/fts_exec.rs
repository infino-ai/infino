// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 full-text search as DataFusion table-valued functions.
//!
//! `bm25_search(column, query, k [, mode])` and
//! `bm25_search_prefix(column, prefix, k)` register via `register_udtf`
//! and lower to [`Bm25Exec`], a custom `ExecutionPlan` that calls the
//! existing async BM25 kernels
//! ([`SupertableReader::bm25_search`](crate::supertable::handle::SupertableReader::bm25_search)
//! / `bm25_search_prefix`) inside `execute()` and resolves
//! each [`SuperfileHit`](crate::supertable::query::SuperfileHit) to the
//! supertable's `_id` + projected scalar columns + `score` via
//! the shared [`resolve_hits`](super::common::resolve_hits).
//!
//! ## Query shape
//!
//! ```sql
//! SELECT _id, score FROM bm25_search('body', 'error timeout', 10) ORDER BY score DESC;
//! SELECT _id, score FROM bm25_search('body', 'error rust', 10, 'and') ORDER BY score DESC;
//! SELECT _id, score FROM bm25_search_prefix('body', 'err', 10) ORDER BY score DESC;
//! ```
//!
//! `score` is BM25 relevance — **higher is better**, so `ORDER BY score
//! DESC` lists the best matches first (the kernels already emit
//! descending). The optional `mode` is `'or'` (default) or `'and'`;
//! prefix search always runs OR over the expanded term set.
//!
//! ## `WHERE` on the function
//!
//! A `WHERE` over the function's output means *the top k among rows
//! satisfying the predicate*, never the predicate applied to a global
//! top-k. Filters are reported `Inexact`, so DataFusion hands them to
//! `scan` and keeps a `FilterExec` above. `bm25_search` lowers them to a
//! [`CandidateScope`](crate::supertable::query::candidate::CandidateScope):
//! the superfiles the manifest statistics and term blooms keep, and —
//! when an FTS-indexed column bounds the rows — the candidate rows the
//! kernel admits into its heap. The function then ranks within the scope,
//! applies the exact predicate to the rows it gets, and widens the search
//! until `k` survive or the fill's ceiling is reached
//! ([`fill_ceiling`]). Stopped short at a ceiling below the table's
//! size, the rows the predicate keeps may rank below everything it looked
//! at: the function then ranks once more within the rows the query's
//! terms match ([`SupertableReader::match_scope`]), checking the predicate
//! on each superfile's best-scoring rows as the search reaches them
//! ([`SupertableReader::bm25_search_checked_async`]), so the answer is the
//! top `k` among matching rows however deep they rank, and the predicate
//! is decoded only for rows that could place. A predicate that
//! cannot be evaluated before ranking (one on `score`) keeps what the
//! widening found. `bm25_search_prefix` takes the widening alone.

use std::{fmt, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableFunctionArgs, TableFunctionImpl, TableProvider},
    common::Statistics,
    error::{DataFusionError, Result as DfResult},
    execution::{TaskContext, context::SessionContext},
    logical_expr::{Expr, TableProviderFilterPushDown, TableType},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};

use crate::{
    superfile::fts::reader::{Bm25SearchOptions, BoolMode},
    supertable::{
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        query::{
            SuperfileHit,
            candidate::RankScope,
            exec::common::{
                Arity, PushedPredicate, RowCheck, arg_to_string, arg_to_usize,
                candidate_plan_for_filters, fill_ceiling, fill_top_k_within, live_reader,
                output_schema_with_score, scope_to_call, top_k_statistics, traced_tvf,
                widening_ended_short,
            },
        },
    },
    utils::trace::detail_span,
};

/// SQL name for the term-based BM25 TVF.
pub(crate) const BM25_SEARCH_UDTF: &str = "bm25_search";
/// SQL name for the prefix BM25 TVF.
pub(crate) const BM25_PREFIX_UDTF: &str = "bm25_search_prefix";

/// Arguments of `bm25_search(column, query, k)`; the optional `mode` makes
/// it `bm25_search(column, query, k, mode)`.
pub(crate) const BM25_SEARCH_ARITY: Arity = Arity {
    min: 3,
    max: 4,
    arguments: "column, query, k[, mode]",
};
/// Arguments of `bm25_search_prefix(column, prefix, k)`.
pub(crate) const BM25_PREFIX_SEARCH_ARITY: Arity = Arity {
    min: 3,
    max: 3,
    arguments: "column, prefix, k",
};

/// Register `bm25_search` + `bm25_search_prefix` on `ctx`, bound to
/// the query's pinned `reader` + scalar `schema`. Called from
/// [`Supertable::query_sql`](crate::supertable::handle::Supertable::query_sql).
pub(crate) fn register_bm25(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        BM25_SEARCH_UDTF,
        Arc::new(Bm25SearchFunc::new(
            Arc::clone(&reader),
            Arc::clone(&scalar_schema),
        )),
    );
    ctx.register_udtf(
        BM25_PREFIX_UDTF,
        Arc::new(Bm25PrefixFunc::new(reader, scalar_schema)),
    );
}

/// Which BM25 kernel a `Bm25Exec` invocation runs.
#[derive(Debug, Clone)]
enum Bm25Query {
    /// `bm25_search(col, query, k, mode)` — tokenized term query.
    Terms { query: String, mode: BoolMode },
    /// `bm25_search_prefix(col, prefix, k)` — last token expanded to
    /// its lex range, OR-scored.
    Prefix { prefix: String },
}

/// `TableFunctionImpl` for `bm25_search`.
#[derive(Debug)]
pub(crate) struct Bm25SearchFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl Bm25SearchFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for Bm25SearchFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();
        BM25_SEARCH_ARITY.check(BM25_SEARCH_UDTF, args.len())?;
        let column = arg_to_string(&args[0], "bm25_search column")?;
        let query = arg_to_string(&args[1], "bm25_search query")?;
        let k = arg_to_usize(&args[2], "bm25_search k")?;
        let mode = match args.get(3) {
            Some(expr) => arg_to_bool_mode(expr)?,
            None => BoolMode::Or,
        };
        let reader = live_reader(&self.reader, "bm25_search")?;
        scope_to_call(
            BM25_SEARCH_UDTF,
            Arc::new(Bm25Table {
                reader,
                column,
                query: Bm25Query::Terms { query, mode },
                k,
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
            }),
        )
    }
}

/// `TableFunctionImpl` for `bm25_search_prefix`.
#[derive(Debug)]
pub(crate) struct Bm25PrefixFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl Bm25PrefixFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for Bm25PrefixFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();

        BM25_PREFIX_SEARCH_ARITY.check(BM25_PREFIX_UDTF, args.len())?;

        let column = arg_to_string(&args[0], "bm25_search_prefix column")?;
        let prefix = arg_to_string(&args[1], "bm25_search_prefix prefix")?;
        let k = arg_to_usize(&args[2], "bm25_search_prefix k")?;
        let reader = live_reader(&self.reader, "bm25_search_prefix")?;

        scope_to_call(
            BM25_PREFIX_UDTF,
            Arc::new(Bm25Table {
                reader,
                column,
                query: Bm25Query::Prefix { prefix },
                k,
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
            }),
        )
    }
}

/// One parsed BM25 invocation as a `TableProvider`. `scan` lowers to
/// [`Bm25Exec`]; the TVF's `k` is the top-k bound.
struct Bm25Table {
    reader: Arc<SupertableReader>,
    column: String,
    query: Bm25Query,
    k: usize,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for Bm25Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bm25Table")
            .field("column", &self.column)
            .field("query", &self.query)
            .field("k", &self.k)
            .finish()
    }
}

#[async_trait]
impl TableProvider for Bm25Table {
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
        let exec = Bm25Exec::try_new(
            Arc::clone(&self.reader),
            self.column.clone(),
            self.query.clone(),
            self.k,
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
            filters.to_vec(),
        )?;
        Ok(Arc::new(exec))
    }

    /// Report every `WHERE` filter as `Inexact`: DataFusion hands the
    /// predicates to [`scan`](Self::scan), where they scope the search to
    /// the rows that can satisfy them, **and** keeps a `FilterExec` above
    /// the scan that re-applies the exact predicate. Correctness never
    /// depends on the pushdown.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/// Custom `ExecutionPlan` that runs a BM25 kernel on the query
/// runtime inside `execute()` and emits the resolved `_id` +
/// scalar columns + `score`.
struct Bm25Exec {
    reader: Arc<SupertableReader>,
    column: String,
    query: Bm25Query,
    k: usize,
    /// `WHERE` predicates DataFusion pushed into this scan (reported
    /// `Inexact`); scope the search and fill its `k` in `execute`.
    filters: Vec<Expr>,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
    /// What the planner is told about this search's size, built once:
    /// DataFusion asks on every optimizer pass. See [`top_k_statistics`].
    statistics: Arc<Statistics>,
}

impl Bm25Exec {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        reader: Arc<SupertableReader>,
        column: String,
        query: Bm25Query,
        k: usize,
        scalar_schema: SchemaRef,
        output_schema: SchemaRef,
        projection: Option<Vec<usize>>,
        filters: Vec<Expr>,
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
        let statistics = top_k_statistics(&projected_schema, k, reader.manifest().n_docs_total());
        Ok(Self {
            reader,
            column,
            query,
            k,
            filters,
            scalar_schema,
            output_schema,
            projection,
            projected_schema,
            cache,
            statistics,
        })
    }

    /// Concise one-line description shared by `Debug` + `DisplayAs`.
    fn describe(&self) -> String {
        match &self.query {
            Bm25Query::Terms { mode, .. } => format!(
                "Bm25Exec: kind=search, column={}, k={}, mode={:?}",
                self.column, self.k, mode
            ),
            Bm25Query::Prefix { .. } => {
                format!(
                    "Bm25Exec: kind=prefix, column={}, k={}",
                    self.column, self.k
                )
            }
        }
    }
}

impl fmt::Debug for Bm25Exec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for Bm25Exec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for Bm25Exec {
    fn name(&self) -> &'static str {
        "Bm25Exec"
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

    fn partition_statistics(&self, _partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        Ok(Arc::clone(&self.statistics))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "Bm25Exec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let column = self.column.clone();
        let query = self.query.clone();
        let k = self.k;
        let filters = self.filters.clone();
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            // What the pushed-down `WHERE` admits: the superfiles the
            // manifest keeps and, under an index-bounded predicate, the
            // documents in each (in blob ids where the file's own text index
            // bounds it alone, so the whole doc-id map is not read). Prefix search has no
            // scoped kernel (its term set is only known per superfile), so it
            // fills its `k` under the exact predicate alone.
            let scope = match &query {
                Bm25Query::Terms { .. } if !filters.is_empty() => {
                    let plan = candidate_plan_for_filters(reader.manifest(), &filters);
                    Some(reader.rank_scope(&filters, &plan).await?)
                }
                Bm25Query::Terms { .. } | Bm25Query::Prefix { .. } => None,
            };
            let predicate = PushedPredicate::compile(&filters, &output_schema);
            // A term search under a `WHERE` that bounds superfiles and can be
            // checked from stored columns ranks within the checked rows once
            // widening comes back short.
            let check = match (&scope, &predicate, &query) {
                (Some(_), Some(predicate), Bm25Query::Terms { .. }) => {
                    RowCheck::new(&reader, predicate, &output_schema)?.map(Arc::new)
                }
                _ => None,
            };
            let checked_when_short = check.is_some() && widening_ended_short(&reader, k, 0);
            let call = RankedCall {
                reader: &reader,
                column: &column,
                query: &query,
                k,
                predicate: predicate.as_ref(),
                scalar_schema: &scalar_schema,
                output_schema: &output_schema,
                projection: projection.as_deref(),
            };
            // Widen within the scope first, up to the fill's ceiling: the
            // rounds stop as soon as `k` of the kernel's best pass the
            // predicate, which for a predicate most top-ranked rows satisfy
            // is a round or two of the search alone.
            let widened = call.rows(scope.as_ref(), checked_when_short).await?;
            let (Some(scope), Some(check), Bm25Query::Terms { query: text, mode }) =
                (scope.as_ref(), check, &query)
            else {
                return Ok(widened);
            };
            if !widening_ended_short(&reader, k, widened.num_rows()) {
                return Ok(widened);
            }
            // Short at the ceiling: rows the predicate keeps may rank below
            // every hit the widening looked at. Rank once within the rows the
            // query's terms match, checking the predicate on each superfile's
            // best rows as the search reaches them.
            let matched = RankScope::from(reader.match_scope(&column, text, *mode, scope).await?);
            call.checked_rows(&matched, check, text, *mode).await
        };

        let span = match &self.query {
            Bm25Query::Terms { .. } => {
                detail_span!(
                    "tvf.bm25_search",
                    k = self.k,
                    rows_out = tracing::field::Empty
                )
            }
            Bm25Query::Prefix { .. } => {
                detail_span!(
                    "tvf.bm25_search_prefix",
                    k = self.k,
                    rows_out = tracing::field::Empty
                )
            }
        };
        let fut = traced_tvf(span, fut);
        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// One `bm25_search` call's fixed inputs, borrowed by each ranking round
/// [`Bm25Exec::execute`] runs.
struct RankedCall<'a> {
    reader: &'a SupertableReader,
    column: &'a str,
    query: &'a Bm25Query,
    k: usize,
    predicate: Option<&'a PushedPredicate>,
    scalar_schema: &'a SchemaRef,
    output_schema: &'a SchemaRef,
    projection: Option<&'a [usize]>,
}

impl RankedCall<'_> {
    /// The kernel's `want` best hits within `scope` (`None` searches the
    /// whole table; the prefix kernel has no scoped form and ignores it).
    async fn hits(
        &self,
        want: usize,
        scope: Option<&RankScope>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        match self.query {
            Bm25Query::Terms { query, mode } => {
                self.reader
                    .bm25_search_scoped_async(
                        self.column,
                        query,
                        want,
                        Bm25SearchOptions::new().with_mode(*mode),
                        scope,
                    )
                    .await
            }
            Bm25Query::Prefix { prefix } => {
                self.reader
                    .bm25_search_prefix_async(self.column, prefix, want)
                    .await
            }
        }
    }

    /// The call's `k` rows within `scope` whose hits pass `check`: one round,
    /// since every hit the search returns has passed it.
    async fn checked_rows(
        &self,
        scope: &RankScope,
        check: Arc<RowCheck>,
        text: &str,
        mode: BoolMode,
    ) -> DfResult<RecordBatch> {
        fill_top_k_within(
            self.reader,
            self.k,
            self.predicate,
            self.scalar_schema,
            self.output_schema,
            self.projection,
            self.k,
            false,
            |want: usize| {
                self.reader.bm25_search_checked_async(
                    self.column,
                    text,
                    want,
                    Bm25SearchOptions::new().with_mode(mode),
                    scope,
                    Arc::clone(&check),
                )
            },
        )
        .await
    }

    /// The call's `k` rows within `scope` under the predicate, widening up to
    /// the fill's ceiling ([`fill_ceiling`]). With `checked_when_short` the
    /// caller checks the predicate exactly if the rows come back short.
    async fn rows(
        &self,
        scope: Option<&RankScope>,
        checked_when_short: bool,
    ) -> DfResult<RecordBatch> {
        let ceiling = fill_ceiling(self.reader, self.k);
        fill_top_k_within(
            self.reader,
            self.k,
            self.predicate,
            self.scalar_schema,
            self.output_schema,
            self.projection,
            ceiling,
            checked_when_short,
            |want: usize| self.hits(want, scope),
        )
        .await
    }
}

/// Parse the optional `mode` argument: `'or'` (default) or `'and'`.
pub(crate) fn arg_to_bool_mode(expr: &Expr) -> DfResult<BoolMode> {
    let s = arg_to_string(expr, "bm25_search mode")?;
    match s.to_ascii_lowercase().as_str() {
        "or" => Ok(BoolMode::Or),
        "and" => Ok(BoolMode::And),
        other => Err(DataFusionError::Plan(format!(
            "bm25_search mode must be 'or' or 'and', got '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use arrow_array::{Array, Float32Array, Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit, lower};
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;
    use tokio::task::spawn_blocking;

    use super::*;
    use crate::{
        FieldPatch, SchemaPatch,
        config::{CompactionSettings, OptimizeOptions},
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{SuperfileReader, builder::FtsConfig, id_space::Admitted},
        supertable::{
            Supertable, SupertableOptions,
            query::{
                candidate::CandidatePlan, dispatch::open_reader,
                exec::common::OVER_FETCH_MIN_CEILING,
            },
            reader_cache::ReadIntent,
        },
        test_helpers::lazy_foreground_disk_cache,
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
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        SupertableOptions::new(title_schema(), vec![FtsConfig::new("title")], vec![])
            .expect("valid options")
            .with_writer_pool(pool)
    }

    fn supertable_with_titles(titles: &[&str]) -> Supertable {
        let st = Supertable::create(options_title_fts()).expect("create");
        let mut w = st.writer().expect("writer");
        let arr = LargeStringArray::from(titles.to_vec());
        let batch = RecordBatch::try_new(title_schema(), vec![Arc::new(arr)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st
    }

    /// Demo corpus: `rust` in docs 0 + 4, `systems` only in doc 4.
    fn demo_corpus() -> Supertable {
        supertable_with_titles(&[
            "rust async runtime",       // 0
            "python data science",      // 1
            "java spring boot",         // 2
            "go routines channels",     // 3
            "rust systems programming", // 4
            "ruby on rails",            // 5
        ])
    }

    fn titles_of(batches: &[RecordBatch]) -> Vec<String> {
        strings_of(batches, "title")
    }

    /// The values of the `LargeUtf8` column `column` in `batches`, in order.
    fn strings_of(batches: &[RecordBatch], column: &str) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            let col = b
                .column_by_name(column)
                .expect("column projected")
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("utf8");
            out.extend((0..col.len()).map(|i| col.value(i).to_string()));
        }
        out
    }

    fn scores_of(batches: &[RecordBatch]) -> Vec<f32> {
        let mut out = Vec::new();
        for b in batches {
            let idx = b.schema().index_of("score").expect("score col");
            let col = b
                .column(idx)
                .as_any()
                .downcast_ref::<Float32Array>()
                .expect("f32");
            for i in 0..col.len() {
                out.push(col.value(i));
            }
        }
        out
    }

    // ---- WHERE on the function: the top-k among rows satisfying it ----

    /// Titles for the pushdown corpora: every row holds `rust`, but the
    /// first `n_strong` rows hold it twice in a two-word title while the
    /// rest hold it once in a five-word title, so BM25 ranks every strong
    /// row above every weak one (tf 2 in a short doc beats tf 1 in a long
    /// doc). A `WHERE` selecting the weak rows is then exactly the case a
    /// post-filtered global top-k gets wrong.
    fn strong_then_weak_titles(n_strong: usize, n_weak: usize) -> Vec<&'static str> {
        let mut titles = vec!["rust rust"; n_strong];
        titles.extend(vec!["rust with four more words"; n_weak]);
        titles
    }

    /// Row count across `batches`.
    fn rows_of(batches: &[RecordBatch]) -> usize {
        batches.iter().map(|b| b.num_rows()).sum()
    }

    /// `title` (FTS) + `category` (FTS, one keyword per row): the predicate
    /// column the index can bound.
    fn options_title_category_fts() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("category", DataType::LargeUtf8, false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig::new("title"), FtsConfig::new("category")],
            vec![],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Five strong `a` rows above three weak `b` rows.
    fn supertable_title_category() -> Supertable {
        let st = Supertable::create(options_title_category_fts()).expect("create");
        let mut w = st.writer().expect("writer");
        let titles = LargeStringArray::from(strong_then_weak_titles(5, 3));
        let categories = LargeStringArray::from(vec!["a", "a", "a", "a", "a", "b", "b", "b"]);
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![Arc::new(titles), Arc::new(categories)],
        )
        .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st
    }

    /// `title` (FTS) + `bucket` (Int64, no index): the predicate column
    /// the index cannot bound at all.
    fn options_title_bucket() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]));
        SupertableOptions::new(schema, vec![FtsConfig::new("title")], vec![])
            .expect("valid options")
            .with_writer_pool(pool)
    }

    /// One batch of strong rows in `bucket` 0 then weak rows in `bucket` 1.
    fn title_bucket_batch(schema: Arc<Schema>, n_strong: usize, n_weak: usize) -> RecordBatch {
        let titles = LargeStringArray::from(strong_then_weak_titles(n_strong, n_weak));
        let buckets: Vec<i64> = (0..n_strong + n_weak)
            .map(|i| i64::from(i >= n_strong))
            .collect();
        RecordBatch::try_new(
            schema,
            vec![Arc::new(titles), Arc::new(Int64Array::from(buckets))],
        )
        .expect("batch")
    }

    /// Five strong rows in `bucket` 0 above three weak rows in `bucket` 1,
    /// one superfile.
    fn supertable_title_bucket() -> Supertable {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&title_bucket_batch(st.options().schema.clone(), 5, 3))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    /// The condition the pushdown exists to fix, asserted before every
    /// filtered query below: the unfiltered top-`k` holds none of the rows
    /// the predicate selects, so filtering it afterwards would return
    /// nothing.
    fn assert_unfiltered_top_k_misses(st: &Supertable, k: usize, column: &str, want: &str) {
        let unfiltered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT {column} FROM bm25_search('title', 'rust', {k})"
            ))
            .expect("unfiltered query_sql");
        let matching = unfiltered
            .iter()
            .map(|b| {
                let idx = b.schema().index_of(column).expect("column");
                (0..b.num_rows())
                    .filter(|&i| {
                        let value = b.column(idx);
                        match value.as_any().downcast_ref::<LargeStringArray>() {
                            Some(s) => s.value(i) == want,
                            None => {
                                let n = value
                                    .as_any()
                                    .downcast_ref::<Int64Array>()
                                    .expect("int64 or large utf8");
                                n.value(i).to_string() == want
                            }
                        }
                    })
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            matching, 0,
            "guard: the unfiltered top-{k} must hold no {column} = {want} row"
        );
    }

    /// A `WHERE` on an FTS-indexed column: the index bounds the rows, the
    /// kernel ranks only within them, and the function returns exactly the
    /// `k` best matching rows — where a post-filtered global top-k had none.
    #[test]
    fn bm25_search_tvf_where_on_indexed_column_ranks_among_matching() {
        let st = supertable_title_category();
        let k = 3;
        assert_unfiltered_top_k_misses(&st, k, "category", "b");
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, category, score FROM bm25_search('title', 'rust', {k}) \
                 WHERE category = 'b'"
            ))
            .expect("filtered query_sql");
        assert_eq!(
            rows_of(&rows),
            k,
            "exactly k rows, all satisfying the predicate"
        );
        for b in &rows {
            let idx = b.schema().index_of("category").expect("category");
            let c = b
                .column(idx)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("utf8");
            for i in 0..b.num_rows() {
                assert_eq!(c.value(i), "b");
            }
        }
        let scores = scores_of(&rows);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "still ranked: {scores:?}");
        }
        // An aggregate over the filtered relation counts the matching rows,
        // not the survivors of a global top-k.
        let count = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT COUNT(*) AS n FROM bm25_search('title', 'rust', {k}) WHERE category = 'b'"
            ))
            .expect("count query_sql");
        let n = count[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(n, k as i64);
    }

    /// Rows holding `rust` twice, which BM25 ranks above every weak row.
    /// All of them are deleted in the test below. Deep enough to empty the
    /// first TWO rounds at the `k` used there (10, then 20) — one empty
    /// round catches a fill that reads a short round as exhaustion, two
    /// catch one that reads a flat hit count as exhaustion, and neither
    /// kernel has run out while the weak rows below are still live.
    const N_TOMBSTONED_STRONG: usize = 20;

    /// Live rows the same query matches, more than `k` of them, every one
    /// ranked below the tombstoned rows.
    const N_LIVE_WEAK: usize = 12;

    /// Neither a short round nor a flat hit count means the kernel is out
    /// of candidates: tombstones are subtracted *after* each unit's
    /// `k`-sized heap (`dispatch::fanout_local_hits`), so deleting the
    /// top-scoring hits empties a round on its own, and deleting enough of
    /// them empties several in a row. Either reading ends the fill early
    /// and returns nothing — the same underflow the pushdown exists to
    /// prevent, reached through the delete path.
    #[test]
    fn bm25_search_tvf_where_fills_past_rounds_emptied_by_tombstones() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options_title_bucket().with_storage(storage)).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&title_bucket_batch(
            st.options().schema.clone(),
            N_TOMBSTONED_STRONG,
            N_LIVE_WEAK,
        ))
        .expect("append");
        w.commit().expect("commit");
        drop(w); // release the writer slot so `delete` can acquire it

        let stats = st
            .delete(col("title").eq(lit("rust rust")))
            .expect("delete");
        assert_eq!(stats.matched() as usize, N_TOMBSTONED_STRONG);

        // At this `k` the fill asks for 10 then 20, and every hit in both
        // rounds is a deleted row, so both come back empty before the live
        // rows below are ever reached.
        let k = 10;
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, bucket, score FROM bm25_search('title', 'rust', {k}) \
                 WHERE bucket = 1"
            ))
            .expect("filtered query_sql");
        assert_eq!(
            rows_of(&rows),
            k,
            "the fill must widen past a round the tombstones emptied"
        );
    }

    /// A `WHERE` on a column with no index: nothing bounds the rows, so
    /// the function applies the exact predicate itself and widens the
    /// search until `k` rows survive.
    #[test]
    fn bm25_search_tvf_where_on_unindexed_column_fills_k() {
        let st = supertable_title_bucket();
        let k = 3;
        assert_unfiltered_top_k_misses(&st, k, "bucket", "1");
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, bucket FROM bm25_search('title', 'rust', {k}) WHERE bucket = 1"
            ))
            .expect("filtered query_sql");
        assert_eq!(rows_of(&rows), k, "k rows survive after over-fetching");
        for b in &rows {
            let idx = b.schema().index_of("bucket").expect("bucket");
            let v = b
                .column(idx)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64");
            for i in 0..b.num_rows() {
                assert_eq!(v.value(i), 1);
            }
        }
        // Fewer matching rows than `k` exist: every one of them, no more.
        let all = st
            .reader()
            .expect("reader")
            .query_sql("SELECT bucket FROM bm25_search('title', 'rust', 10) WHERE bucket = 1")
            .expect("filtered query_sql");
        assert_eq!(rows_of(&all), 3, "the table holds three bucket-1 rows");
        // The predicate's column need not be projected for the fill to
        // work — it is decoded for the check and dropped again.
        let titles_only = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title FROM bm25_search('title', 'rust', {k}) WHERE bucket = 1"
            ))
            .expect("filtered query_sql");
        assert_eq!(rows_of(&titles_only), k);
        assert_eq!(titles_only[0].num_columns(), 1);
    }

    /// Strong rows above the weak ones in the test below: more than the
    /// widening's ceiling lets a small `k` look at, so a fill that only
    /// widens stops with every weak row still below it.
    const N_STRONG_PAST_THE_CEILING: usize = OVER_FETCH_MIN_CEILING + 100;

    /// Weak rows, the only ones the predicate keeps, all ranked below
    /// [`N_STRONG_PAST_THE_CEILING`] strong ones.
    const N_WEAK_PAST_THE_CEILING: usize = 5;

    /// A `WHERE` on a column with no index whose rows rank below the
    /// widening's ceiling: the function finds them anyway, because widening
    /// that stops short at the ceiling makes it check the predicate on the
    /// query's matches and rank within exactly the ones it keeps. Returns
    /// `k` when `k` match, every one when fewer do, and an aggregate counts
    /// all of them.
    #[test]
    fn bm25_search_tvf_where_on_unindexed_column_reaches_rows_below_the_fill_ceiling() {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&title_bucket_batch(
            st.options().schema.clone(),
            N_STRONG_PAST_THE_CEILING,
            N_WEAK_PAST_THE_CEILING,
        ))
        .expect("append");
        w.commit().expect("commit");
        drop(w);
        assert_unfiltered_top_k_misses(&st, OVER_FETCH_MIN_CEILING, "bucket", "1");

        let k = 3;
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, bucket, score FROM bm25_search('title', 'rust', {k}) \
                 WHERE bucket = 1"
            ))
            .expect("filtered query_sql");
        assert_eq!(
            rows_of(&rows),
            k,
            "the k best of the rows the predicate keeps"
        );
        assert!(
            titles_of(&rows)
                .iter()
                .all(|t| t == "rust with four more words"),
            "every row is a weak bucket-1 row"
        );
        let scores = scores_of(&rows);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "still ranked: {scores:?}");
        }

        let all = st
            .reader()
            .expect("reader")
            .query_sql("SELECT bucket FROM bm25_search('title', 'rust', 10) WHERE bucket = 1")
            .expect("filtered query_sql");
        assert_eq!(
            rows_of(&all),
            N_WEAK_PAST_THE_CEILING,
            "fewer than k match: every one of them, no more"
        );

        let count = st
            .reader()
            .expect("reader")
            .query_sql(
                "SELECT COUNT(*) AS n FROM bm25_search('title', 'rust', 10) WHERE bucket = 1",
            )
            .expect("count query_sql");
        let n = count[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(n, N_WEAK_PAST_THE_CEILING as i64);
    }

    /// Weak rows in the first superfile of the spread case below.
    const N_WEAK_IN_FIRST: usize = 2;

    /// Strong rows only, in the spread case's middle superfile: it holds no
    /// row the predicate keeps.
    const N_STRONG_ONLY: usize = 50;

    /// Weak rows in the spread case's last superfile.
    const N_WEAK_IN_LAST: usize = 3;

    /// `bm25_search('title', 'rust', k) WHERE {predicate}`'s rows, `title`
    /// and `score` projected.
    fn filtered(st: &Supertable, k: usize, predicate: &str) -> Vec<RecordBatch> {
        st.reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, score FROM bm25_search('title', 'rust', {k}) WHERE {predicate}"
            ))
            .expect("filtered query_sql")
    }

    /// The rows past the ceiling spread over several superfiles, one of
    /// which holds none of them: the exact pass checks each superfile's
    /// matches and ranks across all of them.
    #[test]
    fn bm25_search_tvf_where_reaches_rows_past_the_ceiling_across_superfiles() {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let schema = st.options().schema.clone();
        for (strong, weak) in [
            (N_STRONG_PAST_THE_CEILING, N_WEAK_IN_FIRST),
            (N_STRONG_ONLY, 0),
            (0, N_WEAK_IN_LAST),
        ] {
            let mut w = st.writer().expect("writer");
            w.append(&title_bucket_batch(Arc::clone(&schema), strong, weak))
                .expect("append");
            w.commit().expect("commit");
        }
        let k = 3;
        assert_eq!(
            titles_of(&filtered(&st, k, "bucket = 1")),
            vec!["rust with four more words"; k],
            "the k best weak rows"
        );
        let past_every_match = N_WEAK_IN_FIRST + N_WEAK_IN_LAST + 1;
        assert_eq!(
            rows_of(&filtered(&st, past_every_match, "bucket = 1")),
            N_WEAK_IN_FIRST + N_WEAK_IN_LAST,
            "every weak row, from both superfiles that hold one"
        );
    }

    /// `title` (FTS) + `category` (FTS) + `bucket` (Int64, no index): a
    /// predicate on both bounds the scope by `category` and leaves `bucket`
    /// to the exact pass.
    fn options_title_category_bucket() -> SupertableOptions {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig::new("title"), FtsConfig::new("category")],
            vec![],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Weak `bucket` 1 rows outside the category the bounded case selects:
    /// the scope leaves them out, so they must never come back.
    const N_WEAK_OTHER_CATEGORY: usize = 4;

    /// A predicate on an indexed column and a column with no index: the
    /// indexed one bounds the scope, widening within it stops short at the
    /// ceiling, and the exact pass checks the other column on the query's
    /// matches inside that scope only.
    #[test]
    fn bm25_search_tvf_where_past_the_ceiling_within_a_bounded_scope() {
        let st = Supertable::create(options_title_category_bucket()).expect("create");
        let n_in = N_STRONG_PAST_THE_CEILING + N_WEAK_PAST_THE_CEILING;
        let n = n_in + N_WEAK_OTHER_CATEGORY;
        let mut titles =
            strong_then_weak_titles(N_STRONG_PAST_THE_CEILING, N_WEAK_PAST_THE_CEILING);
        titles.extend(vec!["rust with four more words"; N_WEAK_OTHER_CATEGORY]);
        let categories: Vec<&str> = (0..n).map(|i| if i < n_in { "b" } else { "a" }).collect();
        let buckets: Vec<i64> = (0..n)
            .map(|i| i64::from(i >= N_STRONG_PAST_THE_CEILING))
            .collect();
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(titles)),
                Arc::new(LargeStringArray::from(categories)),
                Arc::new(Int64Array::from(buckets)),
            ],
        )
        .expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        drop(w);

        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT category, bucket FROM bm25_search('title', 'rust', {}) \
                 WHERE category = 'b' AND bucket = 1",
                N_WEAK_PAST_THE_CEILING + 1
            ))
            .expect("filtered query_sql");
        assert_eq!(
            rows_of(&rows),
            N_WEAK_PAST_THE_CEILING,
            "every weak row of the category, none of the other"
        );
        for b in &rows {
            let idx = b.schema().index_of("category").expect("category");
            let c = b
                .column(idx)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("utf8");
            for i in 0..b.num_rows() {
                assert_eq!(c.value(i), "b");
            }
        }
    }

    /// An `IS NULL` on a column the table gained after its first superfile
    /// was written: that superfile has no such column, its rows read as
    /// null, and the exact pass must read them so rather than as missing.
    #[test]
    fn bm25_search_tvf_where_is_null_reaches_a_superfile_older_than_its_column() {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let mut w = st.writer().expect("writer");
        w.append(&title_bucket_batch(
            st.options().schema.clone(),
            0,
            N_WEAK_PAST_THE_CEILING,
        ))
        .expect("append");
        w.commit().expect("commit");
        drop(w);
        st.apply_schema(
            &SchemaPatch::new(vec![
                FieldPatch::named("note")
                    .with_type(DataType::Int64)
                    .with_nullable(true),
            ]),
            None,
        )
        .expect("add the column");

        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
            Field::new("note", DataType::Int64, true),
        ]));
        let strong = title_bucket_batch(
            Arc::new(Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, false),
                Field::new("bucket", DataType::Int64, false),
            ])),
            N_STRONG_PAST_THE_CEILING,
            0,
        );
        let notes = Int64Array::from(vec![0_i64; N_STRONG_PAST_THE_CEILING]);
        let strong = RecordBatch::try_new(
            schema,
            vec![
                Arc::clone(strong.column(0)),
                Arc::clone(strong.column(1)),
                Arc::new(notes),
            ],
        )
        .expect("batch with the new column");
        let mut w = st.writer().expect("writer");
        w.append(&strong).expect("append");
        w.commit().expect("commit");
        drop(w);

        let k = 3;
        assert_eq!(
            titles_of(&filtered(&st, k, "note IS NULL")),
            vec!["rust with four more words"; k],
            "the k best rows of the superfile older than the column"
        );
    }

    /// Rows holding the rare term the cost case below searches for.
    const N_RARE: usize = 3;

    /// Of the [`N_RARE`] rows, how many the predicate keeps.
    const N_RARE_KEPT: usize = 2;

    /// A rare term under a predicate no index bounds, in a table larger than
    /// the ceiling: widening stops short, and the exact pass checks the
    /// predicate on the term's few matches, never on the rest of the table,
    /// which it would have to decode a column of to check.
    #[test]
    fn bm25_search_tvf_where_on_a_rare_term_checks_only_its_matches() {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let n = N_STRONG_PAST_THE_CEILING + N_RARE;
        let titles: Vec<&str> = (0..n)
            .map(|i| {
                if i < N_STRONG_PAST_THE_CEILING {
                    "rust rust"
                } else {
                    "zyzzyva"
                }
            })
            .collect();
        let buckets: Vec<i64> = (0..n).map(|i| i64::from(i >= n - N_RARE_KEPT)).collect();
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![
                Arc::new(LargeStringArray::from(titles)),
                Arc::new(Int64Array::from(buckets)),
            ],
        )
        .expect("batch");
        let mut w = st.writer().expect("writer");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        drop(w);

        let k = 10;
        let reader = st.reader().expect("reader");
        let rows = reader
            .query_sql(&format!(
                "SELECT title FROM bm25_search('title', 'zyzzyva', {k}) WHERE bucket = 1"
            ))
            .expect("filtered query_sql");
        assert_eq!(rows_of(&rows), N_RARE_KEPT);

        // What the exact pass is handed to check: the same scope the
        // function builds for this `WHERE`, narrowed to the term's matches.
        let filters = vec![col("bucket").eq(lit(1_i64))];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        let matched = reader.block_on(async {
            let scope = reader
                .candidate_scope(&filters, &plan)
                .await
                .expect("scope");
            reader
                .match_scope("title", "zyzzyva", BoolMode::Or, &RankScope::from(scope))
                .await
                .expect("match scope")
        });
        let checked: u64 = matched
            .allow
            .as_ref()
            .expect("a bounded scope")
            .values()
            .map(|rows| rows.len())
            .sum();
        assert_eq!(
            checked, N_RARE as u64,
            "the exact pass checks the term's matches, not the table's rows"
        );
    }

    /// A predicate that reads `score` cannot be evaluated before ranking,
    /// so the function keeps widening instead, and still fills `k`.
    #[test]
    fn bm25_search_tvf_where_on_score_still_fills_k() {
        let st = supertable_title_bucket();
        let k = 3;
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT bucket FROM bm25_search('title', 'rust', {k}) \
                 WHERE bucket = 1 AND score > 0"
            ))
            .expect("filtered query_sql");
        assert_eq!(rows_of(&rows), k);
    }

    /// The prefix function has no scoped kernel and takes the fill path
    /// alone; it fills `k` all the same.
    #[test]
    fn bm25_search_prefix_tvf_where_fills_k() {
        let st = supertable_title_bucket();
        let k = 3;
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, bucket FROM bm25_search_prefix('title', 'rus', {k}) WHERE bucket = 1"
            ))
            .expect("filtered prefix query_sql");
        assert_eq!(rows_of(&rows), k);
    }

    /// A qualified column reference — the shape a `FROM ... AS t` alias
    /// produces — resolves against the function's own output.
    #[test]
    fn bm25_search_tvf_where_with_table_alias_fills_k() {
        let st = supertable_title_bucket();
        let k = 3;
        let rows = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT t.title FROM bm25_search('title', 'rust', {k}) AS t WHERE t.bucket = 1"
            ))
            .expect("aliased query_sql");
        assert_eq!(rows_of(&rows), k);
    }

    /// Manifest statistics scope the search before any superfile opens: a
    /// predicate on an unindexed column whose per-superfile min/max are
    /// disjoint keeps only the superfile that can hold a match, and a
    /// predicate no superfile can satisfy keeps none.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn candidate_scope_prunes_superfiles_by_statistics() {
        let st = Supertable::create(options_title_bucket()).expect("create");
        let schema = st.options().schema.clone();
        // Two commits, two superfiles: bucket 0 only, then bucket 1 only.
        {
            let mut w = st.writer().expect("writer");
            w.append(&title_bucket_batch(Arc::clone(&schema), 4, 0))
                .expect("append");
            w.commit().expect("commit");
        }
        {
            let mut w = st.writer().expect("writer");
            let weak_only = title_bucket_batch(schema, 0, 4);
            w.append(&weak_only).expect("append");
            w.commit().expect("commit");
        }
        let reader = st.reader().expect("reader");
        assert_eq!(
            reader.manifest().superfiles.len(),
            2,
            "premise: two superfiles"
        );

        let filters = [col("bucket").eq(lit(1_i64))];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        assert_eq!(plan, CandidatePlan::Unbounded, "no index bounds `bucket`");
        let scope = reader
            .candidate_scope(&filters, &plan)
            .await
            .expect("scope");
        assert_eq!(
            scope.superfiles.len(),
            1,
            "only the bucket-1 superfile survives"
        );
        assert_eq!(scope.superfiles[0].n_docs, 4);
        assert!(scope.allow.is_none(), "an unbounded plan bounds no row");

        let none = [col("bucket").eq(lit(7_i64))];
        let scope = reader
            .candidate_scope(&none, &candidate_plan_for_filters(reader.manifest(), &none))
            .await
            .expect("scope");
        assert!(
            scope.superfiles.is_empty(),
            "no superfile can hold bucket 7"
        );

        // An indexed predicate bounds rows as well, and the statistics
        // still prune first: `title = 'rust rust'` tokenizes to a term-AND
        // every row of both superfiles holds, but the weak superfile's
        // `title` min/max (one distinct value, `rust with four more words`)
        // cannot contain the literal, so only the strong superfile survives
        // — and within it every row is a candidate.
        let indexed = [col("title").eq(lit("rust rust"))];
        let plan = candidate_plan_for_filters(reader.manifest(), &indexed);
        assert!(
            matches!(plan, CandidatePlan::TermsAll { .. }),
            "the index bounds `title`"
        );
        let scope = reader
            .candidate_scope(&indexed, &plan)
            .await
            .expect("scope");
        assert_eq!(
            scope.superfiles.len(),
            1,
            "the title range of the weak superfile excludes the literal"
        );
        assert_eq!(scope.superfiles[0].n_docs, 4);
        let allow = scope.allow.expect("a bounded plan resolves rows");
        let rows: u64 = allow.values().map(|bm| bm.len()).sum();
        assert_eq!(rows, 4, "every row of the survivor holds the term-AND");
    }

    /// Distinct tags in the last superfile of [`tag_u_supertable`]: past what the manifest's exact
    /// value table for a superfile keeps (128 KiB of values).
    const MANY_DISTINCT_TAGS: usize = 20_000;

    /// Five superfiles with text-indexed `tag` and `u` and range-indexed `n`, in commit order:
    ///
    /// | superfile | tag                              | u     | n      |
    /// |-----------|----------------------------------|-------|--------|
    /// | 0         | a a a a                          | y ... | 0-3    |
    /// | 1         | b rare b rare                    | x x y y | 10-13 |
    /// | 2         | c c c c                          | x ... | 20-23  |
    /// | 3         | b c z z, then `w0` to `w19999`   | y ... | from 30 |
    /// | 4         | !! !! d d                        | v ... | from 20,034 |
    ///
    /// Only superfile 1 holds `rare`, and one row of it holds `rare` beside `x`. Superfile 3 holds
    /// words of several values without holding any one value's words together, and so many
    /// distinct values that the manifest keeps no exact table of them: a value set checks it by
    /// its min/max range alone. Superfile 4 holds `!!`, a value with no words.
    /// On local storage, so commits build the table's term index and a word routes to exactly the
    /// superfiles holding it; the directory lives as long as the table is used.
    fn tag_u_supertable() -> (Supertable, tempfile::TempDir) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("tag", DataType::LargeUtf8, false),
            Field::new("u", DataType::LargeUtf8, false),
            Field::new("n", DataType::Int64, false),
        ]));
        let options = SupertableOptions::new(
            Arc::clone(&schema),
            vec![FtsConfig::new("tag"), FtsConfig::new("u")],
            vec![],
        )
        .expect("valid options")
        .with_range_indexes(&["n".to_owned()]);
        let dir = tempfile::TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(options.with_storage(storage)).expect("create");
        let words = |w: &[&str]| w.iter().map(|w| (*w).to_owned()).collect::<Vec<_>>();
        // Too many distinct values for the manifest to keep the superfile's exact table of them.
        let many: Vec<String> = ["b", "c", "z", "z"]
            .into_iter()
            .map(str::to_owned)
            .chain((0..MANY_DISTINCT_TAGS).map(|i| format!("w{i}")))
            .collect();
        for (base, tags, u) in [
            (
                0,
                words(&["a", "a", "a", "a"]),
                words(&["y", "y", "y", "y"]),
            ),
            (
                10,
                words(&["b", "rare", "b", "rare"]),
                words(&["x", "x", "y", "y"]),
            ),
            (
                20,
                words(&["c", "c", "c", "c"]),
                words(&["x", "x", "x", "x"]),
            ),
            (30, many.clone(), vec!["y".to_owned(); many.len()]),
            (
                30 + many.len() as i64,
                words(&["!!", "!!", "d", "d"]),
                words(&["v", "v", "v", "v"]),
            ),
        ] {
            let rows = tags.len() as i64;
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(tags)),
                    Arc::new(LargeStringArray::from(u)),
                    Arc::new(Int64Array::from_iter_values(base..base + rows)),
                ],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        (st, dir)
    }

    /// The superfiles a `WHERE` on text-indexed columns may match in, as a pushed-down search's
    /// scope and a filtered vector search both choose them, for each shape the `WHERE` can take.
    /// An `OR` under an `AND` prunes, a value of several words prunes a superfile holding its words
    /// but not the value, and an `OR` with a side that bounds nothing prunes by the rest of the
    /// conjunction alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_where_keeps_exactly_the_superfiles_its_words_allow() {
        let (st, _dir) = tag_u_supertable();
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        // The manifest lists superfiles in commit order, the fixture's numbering.
        let in_commit_order = &manifest.superfiles;
        assert_eq!(in_commit_order.len(), 5, "premise: five superfiles");
        let ids = |kept: &[usize]| -> HashSet<u128> {
            kept.iter()
                .map(|&i| in_commit_order[i].superfile_id.as_u128())
                .collect()
        };
        let in_list =
            |values: &[&str]| col("tag").in_list(values.iter().map(|v| lit(*v)).collect(), false);
        let shapes: [(&str, Expr, &[usize]); 7] = [
            ("=", col("tag").eq(lit("rare")), &[1]),
            ("IN", in_list(&["rare", "c"]), &[1, 2, 3]),
            (
                "IN AND word",
                in_list(&["rare", "absent"]).and(col("u").eq(lit("x"))),
                &[1],
            ),
            // Superfile 3's value range spans both values and it holds `b` and `c`, so only the
            // plan's check that a value's words are together rules it out. Superfile 1's exact
            // table of values rules it out.
            ("IN of two-word values", in_list(&["b x", "c y"]), &[]),
            (
                "OR of ANDs under AND",
                col("tag")
                    .eq(lit("rare"))
                    .and(col("u").eq(lit("y")))
                    .or(col("tag").eq(lit("c")).and(col("u").eq(lit("x"))))
                    .and(col("n").gt_eq(lit(0_i64))),
                &[1, 2],
            ),
            (
                "OR with a side that bounds nothing, AND word",
                col("tag")
                    .eq(lit("rare"))
                    .or(lower(col("tag")).eq(lit("y")))
                    .and(col("u").eq(lit("x"))),
                &[1, 2],
            ),
            // `!!` has no words, so no word check can stand for it: superfile 4 stays for its
            // exact table of values, which holds `!!`.
            (
                "IN with a value of no words, AND word",
                in_list(&["!!", "rare"]).and(col("u").eq(lit("v"))),
                &[4],
            ),
        ];
        for (what, filter, want) in shapes {
            let filters = [filter];
            let plan = candidate_plan_for_filters(manifest, &filters);
            let kept: HashSet<u128> = reader
                .superfiles_a_where_may_match(&filters, &plan)
                .await
                .expect("prune")
                .iter()
                .map(|e| e.superfile_id.as_u128())
                .collect();
            assert_eq!(kept, ids(want), "{what}");
        }
        // The scope for `IN AND word` admits the one row holding `rare` beside `x`.
        let filters = [in_list(&["rare", "absent"]).and(col("u").eq(lit("x")))];
        let plan = candidate_plan_for_filters(manifest, &filters);
        let scope = reader
            .candidate_scope(&filters, &plan)
            .await
            .expect("scope");
        let allow = scope.allow.expect("a bounded plan resolves rows");
        let rows: Vec<Vec<u32>> = allow.values().map(|bm| bm.iter().collect()).collect();
        assert_eq!(rows, vec![vec![1]]);
    }

    // ---- unit ----

    #[test]
    fn arg_to_bool_mode_accepts_or_and_case_insensitive_rejects_junk() {
        assert_eq!(arg_to_bool_mode(&lit("or")).expect("or"), BoolMode::Or);
        assert_eq!(arg_to_bool_mode(&lit("OR")).expect("OR"), BoolMode::Or);
        assert_eq!(arg_to_bool_mode(&lit("and")).expect("and"), BoolMode::And);
        assert_eq!(arg_to_bool_mode(&lit("AND")).expect("AND"), BoolMode::And);
        assert!(arg_to_bool_mode(&lit("xor")).is_err());
        assert!(arg_to_bool_mode(&lit(5_i64)).is_err());
    }

    // ---- end-to-end through query_sql ----

    #[test]
    fn bm25_search_tvf_returns_matches_in_descending_score() {
        let st = demo_corpus();
        let batches = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title, score FROM bm25_search('title', 'rust', 10)")
            .expect("query_sql");
        let titles = titles_of(&batches);
        assert_eq!(titles.len(), 2, "only docs 0 + 4 contain 'rust'");
        assert!(titles.iter().all(|t| t.contains("rust")));
        let scores = scores_of(&batches);
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "BM25 scores must be descending: {w:?}");
        }
    }

    #[test]
    fn bm25_search_tvf_and_mode_narrows_to_docs_with_all_terms() {
        let st = demo_corpus();
        // AND: only doc 4 has both `rust` and `systems`.
        let and_rows = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title FROM bm25_search('title', 'rust systems', 10, 'and')")
            .expect("query_sql");
        let and_titles = titles_of(&and_rows);
        assert_eq!(and_titles, vec!["rust systems programming".to_string()]);

        // OR (default): docs 0 + 4 (union of `rust` and `systems`).
        let or_rows = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title FROM bm25_search('title', 'rust systems', 10)")
            .expect("query_sql");
        assert_eq!(titles_of(&or_rows).len(), 2);
    }

    #[test]
    fn bm25_search_tvf_negation_excludes_term() {
        let st = demo_corpus();
        // `-systems` drops doc 4; only doc 0 matches `rust` without it.
        let rows = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title FROM bm25_search('title', 'rust -systems', 10)")
            .expect("query_sql");
        assert_eq!(titles_of(&rows), vec!["rust async runtime".to_string()]);

        // A negation-only query has nothing to rank → error.
        let res = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title FROM bm25_search('title', '-rust', 10)");
        assert!(res.is_err(), "negation-only must error; got {res:?}");
    }

    #[test]
    fn bm25_search_prefix_tvf_expands_prefix() {
        let st = demo_corpus();
        // `rus` expands to `rust` → docs 0 + 4.
        let batches = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title FROM bm25_search_prefix('title', 'rus', 10)")
            .expect("query_sql");
        let titles = titles_of(&batches);
        assert_eq!(titles.len(), 2);
        assert!(titles.iter().all(|t| t.contains("rust")));
    }

    #[test]
    fn bm25_search_tvf_star_projection_appends_score_column() {
        let st = demo_corpus();
        let batches = st
            .reader()
            .expect("reader")
            .query_sql("SELECT * FROM bm25_search('title', 'rust', 10)")
            .expect("query_sql");
        let b = &batches[0];
        // Scalar schema (_id, title) + score.
        assert_eq!(b.num_columns(), 3);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "title");
        assert_eq!(b.schema().field(2).name(), "score");
    }

    #[test]
    fn bm25_search_tvf_empty_supertable_returns_no_rows() {
        let st = Supertable::create(options_title_fts()).expect("create");
        let batches = st
            .reader()
            .expect("reader")
            .query_sql("SELECT title, score FROM bm25_search('title', 'rust', 5)")
            .expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn bm25_search_tvf_arity_error() {
        let st = demo_corpus();
        // 2 args (missing k) → a planning error, the caller's mistake.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT title FROM bm25_search('title', 'rust')")
                .is_err()
        );
    }

    #[test]
    fn bm25_search_prefix_tvf_arity_error() {
        let st = demo_corpus();
        // prefix wants exactly 3 args; 2 → planning error.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT title FROM bm25_search_prefix('title', 'rus')")
                .is_err()
        );
    }

    #[test]
    fn bm25_search_tvf_bad_arg_types_error() {
        let st = demo_corpus();
        // Non-integer k.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT title FROM bm25_search('title', 'rust', 'ten')")
                .is_err(),
            "non-integer k must error"
        );
        // Invalid mode literal.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT title FROM bm25_search('title', 'rust', 10, 'nand')")
                .is_err(),
            "invalid mode must error"
        );
    }

    /// Flatten an `EXPLAIN` result into one searchable string — exercises
    /// `Bm25Exec`'s `DisplayAs`/`describe`.
    fn explain(st: &Supertable, sql: &str) -> String {
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(&format!("EXPLAIN {sql}"))
            .expect("explain");
        let mut out = String::new();
        for batch in &batches {
            for column in batch.columns() {
                if let Some(strings) = column.as_any().downcast_ref::<arrow_array::StringArray>() {
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

    /// Construct `Bm25Table` directly through the TVF `call` path and
    /// exercise its `TableProvider` metadata methods (`Debug`,
    /// `as_any`, `table_type`) plus the lowered `Bm25Exec`'s `name` /
    /// `Debug` — none of which normal query execution touches.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bm25_table_and_exec_trait_methods() {
        use crate::supertable::query::exec::common::test_support::{call_tvf, scoped_inner};
        let st = demo_corpus();
        let reader = Arc::new(st.reader().expect("reader"));
        let scalar_schema = reader.manifest().scalar_schema();
        let func = Bm25SearchFunc::new(reader, scalar_schema);
        let table = call_tvf(&func, &[lit("title"), lit("rust"), lit(10_i64)]).expect("bm25 table");

        // TableProvider metadata.
        let dbg = format!("{table:?}");
        assert!(dbg.contains("Bm25Table"), "Debug missing: {dbg}");
        assert!(
            scoped_inner(&table).downcast_ref::<Bm25Table>().is_some(),
            "as_any downcasts to Bm25Table"
        );
        assert_eq!(table.table_type(), TableType::Base);

        // Lower to the ExecutionPlan and hit its name / Debug.
        let ctx = SessionContext::new();
        let plan = table
            .scan(&ctx.state(), None, &[], None)
            .await
            .expect("scan");
        assert_eq!(plan.name(), "Bm25Exec");
        assert!(
            format!("{plan:?}").contains("Bm25Exec"),
            "Exec Debug missing"
        );
    }

    #[test]
    fn bm25_exec_display_describes_search_and_prefix_branches() {
        let st = demo_corpus();
        let terms = explain(
            &st,
            "SELECT _id FROM bm25_search('title', 'rust', 10, 'and')",
        );
        assert!(
            terms.contains("Bm25Exec") && terms.contains("kind=search") && terms.contains("And"),
            "search describe missing: {terms}"
        );
        let prefix = explain(
            &st,
            "SELECT _id FROM bm25_search_prefix('title', 'rus', 10)",
        );
        assert!(
            prefix.contains("Bm25Exec") && prefix.contains("kind=prefix"),
            "prefix describe missing: {prefix}"
        );
    }

    // ---- A WHERE on text columns over a reordered superfile ------------

    /// Commits the mapped fixture writes: enough that their merge reorders the
    /// documents (a smaller merge keeps arrival order) and so has a doc-id map.
    const MAPPED_COMMITS: usize = 60;
    /// Rows in each of [`MAPPED_COMMITS`].
    const MAPPED_PER_COMMIT: usize = 80;
    /// Rows in the mapped fixture.
    const MAPPED_ROWS: usize = MAPPED_COMMITS * MAPPED_PER_COMMIT;
    /// Values the `tag` column cycles through, row by row.
    const MAPPED_TAGS: [&str; 3] = ["red", "blue", "green"];
    /// Compaction target for the mapped fixture, in MiB: larger than all the
    /// commits together, so they merge into one file.
    const MAPPED_TARGET_MB: u64 = 64;
    /// A `k` above the row count, so a search returns every match.
    const MAPPED_EVERY_HIT: usize = 2 * MAPPED_ROWS;
    /// A `k` well below the match count, so the kernel's heap prunes.
    const MAPPED_FEW_HITS: usize = 10;
    /// Queries the concurrency test runs at once on one reader.
    const MAPPED_CONCURRENT: usize = 8;
    /// Rows the deletes test removes: two that `tag = 'red'` keeps, one it does not.
    const MAPPED_DELETED: [usize; 3] = [3, 2223, 1000];
    /// Commits of the checked fixture: past the fill's smallest ceiling, so a
    /// widening that keeps fewer than `k` rows stops short.
    const CHECKED_COMMITS: usize = OVER_FETCH_MIN_CEILING / MAPPED_PER_COMMIT + 1;
    /// The one row [`checked_where`] keeps.
    const CHECKED_ROW: usize = 6;
    /// Moduli that give each body two words shared with many other rows, so
    /// most terms match many documents.
    const MAPPED_SHARED_WORDS: [usize; 2] = [37, 53];
    /// The `n` range the range test keeps, inclusive.
    const MAPPED_RANGE: [i64; 2] = [1000, 1999];

    /// A `WHERE` only row [`CHECKED_ROW`] passes: it is `red` and the only body
    /// holding its `uq` word.
    fn checked_where() -> String {
        format!("tag = 'red' AND body LIKE '%uq{CHECKED_ROW} %'")
    }

    /// Row `n`'s `body`: every row holds `shared`, the searched word.
    fn mapped_body(n: usize) -> String {
        let [a, b] = MAPPED_SHARED_WORDS;
        format!("uq{n} shared t{} t{}", n % a, n % b)
    }

    /// Whether row `n`'s `body` holds the word `t{w}`.
    fn mapped_holds(n: usize, w: usize) -> bool {
        MAPPED_SHARED_WORDS.iter().any(|m| n % m == w)
    }

    /// A query for each way the search walks, with the rows it matches: a phrase and a negated
    /// phrase (the phrase walk), a negated term, every term required, and any term.
    fn mapped_queries() -> Vec<(&'static str, fn(usize) -> bool)> {
        vec![
            ("\"shared t1\"", |n| n % MAPPED_SHARED_WORDS[0] == 1),
            ("shared -\"shared t1\"", |n| n % MAPPED_SHARED_WORDS[0] != 1),
            ("shared -t1", |n| !mapped_holds(n, 1)),
            ("+shared +t1", |n| mapped_holds(n, 1)),
            ("t1 t2", |n| mapped_holds(n, 1) || mapped_holds(n, 2)),
        ]
    }

    /// Row `n`'s `tag`.
    fn mapped_tag(n: usize) -> &'static str {
        MAPPED_TAGS[n % MAPPED_TAGS.len()]
    }

    fn body_tag_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("body", DataType::LargeUtf8, false),
            Field::new("tag", DataType::LargeUtf8, false),
            Field::new("n", DataType::Int64, false),
        ]))
    }

    /// `commits` commits with `body` and `tag` text-indexed and `n`, the row number,
    /// range-indexed, merged into one
    /// superfile that reorders its documents. Reopened cold behind a lazily
    /// filled disk cache, so a query reads the doc-id map only if it needs it.
    fn mapped_body_tag_table(commits: usize) -> (Supertable, TempDir, TempDir) {
        let storage_dir = TempDir::new().expect("storage tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        let options = || {
            let pool = Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .expect("pool"),
            );
            SupertableOptions::new(
                body_tag_schema(),
                vec![FtsConfig::new("body"), FtsConfig::new("tag")],
                vec![],
            )
            .expect("valid options")
            .with_range_indexes(&["n".to_owned()])
            .with_writer_pool(pool)
            .with_storage(Arc::clone(&storage))
        };
        {
            let table = Supertable::create(options()).expect("create");
            for commit in 0..commits {
                let rows = commit * MAPPED_PER_COMMIT..(commit + 1) * MAPPED_PER_COMMIT;
                let bodies: Vec<String> = rows.clone().map(mapped_body).collect();
                let tags: Vec<&str> = rows.clone().map(mapped_tag).collect();
                let ns: Vec<i64> = rows.map(|n| n as i64).collect();
                let batch = RecordBatch::try_new(
                    body_tag_schema(),
                    vec![
                        Arc::new(LargeStringArray::from(bodies)),
                        Arc::new(LargeStringArray::from(tags)),
                        Arc::new(Int64Array::from(ns)),
                    ],
                )
                .expect("batch");
                let mut w = table.writer().expect("writer");
                w.append(&batch).expect("append");
                w.commit().expect("commit");
            }
            table
                .optimize(&OptimizeOptions::compact(CompactionSettings {
                    target_superfile_size_mb: MAPPED_TARGET_MB,
                    min_fill_percent: 1,
                    ..CompactionSettings::default()
                }))
                .expect("optimize");
        }
        let cache_dir = TempDir::new().expect("cache tempdir");
        let cache = lazy_foreground_disk_cache(Arc::clone(&storage), cache_dir.path());
        let table = Supertable::open(options().with_disk_cache(cache)).expect("open");
        assert_eq!(
            table.reader().expect("reader").n_superfiles(),
            1,
            "one merged file"
        );
        (table, storage_dir, cache_dir)
    }

    /// Deletes the rows [`MAPPED_DELETED`] names.
    fn delete_mapped(st: &Supertable) {
        for n in MAPPED_DELETED {
            st.delete(col("body").eq(lit(mapped_body(n))))
                .expect("delete");
        }
    }

    /// Asserts every superfile's set in `scope` is in blob ids.
    fn assert_in_blob_ids(scope: &RankScope) {
        let sets = scope.allow.as_ref().expect("a bounded scope");
        assert!(sets.values().all(|set| matches!(set, Admitted::BlobIds(_))));
    }

    /// The reader of the table's one superfile, the same one its queries use.
    async fn the_superfile_reader(st: &Supertable) -> Arc<SuperfileReader> {
        let reader = st.reader().expect("reader");
        let manifest = reader.manifest();
        let entry = manifest.get_all_superfiles()[0].clone();
        open_reader(
            &manifest.options.store,
            manifest.options.disk_cache.as_ref(),
            manifest.options.storage.as_ref(),
            &entry,
            ReadIntent::Warm,
        )
        .await
        .expect("open superfile")
    }

    /// The `body` of each row `sql` returns, sorted.
    fn bodies_of(reader: &SupertableReader, sql: &str) -> Vec<String> {
        let mut out = strings_of(&reader.query_sql(sql).expect("sql"), "body");
        out.sort();
        out
    }

    /// The bodies of the rows `keep` admits, sorted.
    fn bodies_kept(keep: impl Fn(usize) -> bool) -> Vec<String> {
        let mut out: Vec<String> = (0..MAPPED_ROWS)
            .filter(|&n| keep(n))
            .map(mapped_body)
            .collect();
        out.sort();
        out
    }

    /// SQL for a top-`k` `bm25_search` of `shared` (in every row) under `filter`.
    fn shared_search(k: usize, filter: &str) -> String {
        format!("SELECT body FROM bm25_search('body', 'shared', {k}) WHERE {filter}")
    }

    /// Each `WHERE` the search runs under, with the rows it keeps. The text indexes bound all but
    /// the `NOT`, which the exact predicate fills.
    fn mapped_wheres() -> Vec<(&'static str, Box<dyn Fn(usize) -> bool>)> {
        vec![
            ("tag = 'red'", Box::new(|n| mapped_tag(n) == "red")),
            (
                "tag = 'red' OR tag = 'blue'",
                Box::new(|n| mapped_tag(n) != "green"),
            ),
            ("NOT (tag = 'red')", Box::new(|n| mapped_tag(n) != "red")),
            (
                "tag = 'red' AND body LIKE '%t5 %'",
                Box::new(|n| mapped_tag(n) == "red" && mapped_body(n).contains("t5 ")),
            ),
            ("tag = 'purple'", Box::new(|_| false)),
            ("body LIKE '%shared%'", Box::new(|_| true)),
        ]
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_where_on_a_reordered_file_keeps_no_doc_id_map() {
        // A reordered superfile, top hits for `shared` under `tag = 'red'`.
        // - The text index bounds the `WHERE`, so the set is in blob ids.
        // - The kernel checks blob ids directly, with no doc-id map.
        // - Only the few top hits become rows, reading just their map pages.
        // This test pins that the whole doc-id map (hundreds of MB on a large
        // file) is never loaded.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let file = the_superfile_reader(&st).await;
        let fts = file.fts().expect("fts");
        assert!(
            fts.has_doc_map(),
            "the merge must reorder for this test to mean anything"
        );
        let reader = st.reader().expect("reader");
        let want = bodies_kept(|n| mapped_tag(n) == "red");
        let few = bodies_of(&reader, &shared_search(MAPPED_FEW_HITS, "tag = 'red'"));
        assert_eq!(few.len(), MAPPED_FEW_HITS);
        assert!(few.iter().all(|b| want.binary_search(b).is_ok()), "{few:?}");
        assert!(
            !fts.doc_map_is_read(),
            "the WHERE kept the whole doc-id map"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_where_on_a_reordered_file_ranks_exactly_its_rows() {
        // A reordered superfile, searched under each `WHERE` of `mapped_wheres`:
        // `OR`, `NOT`, an `AND` across two columns, none kept and all kept.
        // - With `k` above the row count, the hits are exactly the kept rows.
        // - With a small `k`, every hit is a kept row.
        // This test pins that a bounded `WHERE` checked in blob ids keeps the same rows as rows.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let reader = st.reader().expect("reader");
        for (filter, keep) in mapped_wheres() {
            let want = bodies_kept(keep.as_ref());
            assert_eq!(
                bodies_of(&reader, &shared_search(MAPPED_EVERY_HIT, filter)),
                want,
                "every match of {filter}"
            );
            let few = bodies_of(&reader, &shared_search(MAPPED_FEW_HITS, filter));
            assert_eq!(few.len(), want.len().min(MAPPED_FEW_HITS), "{filter}");
            assert!(
                few.iter().all(|b| want.binary_search(b).is_ok()),
                "the top {MAPPED_FEW_HITS} of {filter} are rows it keeps: {few:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_checked_search_on_a_reordered_file_returns_exactly_its_rows() {
        // A reordered superfile past the fill's smallest ceiling, under a
        // `WHERE` that keeps one row, fewer than `k`.
        // - The widening stops short at the ceiling.
        // - The search ranks again within the rows its terms match, checking
        //   the predicate on them in rows, through the doc-id map.
        // This test pins that the answer is exactly the one row the `WHERE` keeps.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(CHECKED_COMMITS);
        let file = the_superfile_reader(&st).await;
        assert!(
            file.fts().expect("fts").has_doc_map(),
            "the merge must reorder for this test to mean anything"
        );
        let reader = st.reader().expect("reader");
        assert_eq!(
            bodies_of(&reader, &shared_search(MAPPED_FEW_HITS, &checked_where())),
            vec![mapped_body(CHECKED_ROW)]
        );
        // The rows the check works on are the same from a blob-id scope as from a row scope.
        let filters = vec![
            col("tag")
                .eq(lit("red"))
                .and(col("body").like(lit(format!("%uq{CHECKED_ROW} %")))),
        ];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        let in_blob_ids = reader
            .block_on(reader.rank_scope(&filters, &plan))
            .expect("scope");
        assert_in_blob_ids(&in_blob_ids);
        let in_rows = RankScope::from(
            reader
                .block_on(reader.candidate_scope(&filters, &plan))
                .expect("scope"),
        );
        let mut matched = Vec::new();
        for scope in [&in_blob_ids, &in_rows] {
            let scope = reader.match_scope("body", "shared", BoolMode::Or, scope);
            matched.push(reader.block_on(scope).expect("match scope").allow);
        }
        assert_eq!(matched[0], matched[1]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_searches_with_and_without_a_where_each_rank_their_rows() {
        // Queries at once on one reader, half under `tag = 'red'` and half not.
        // - All of them translate their hits through the one reader's doc-id map.
        // This test pins that each query still ranks exactly its own rows.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let reader = Arc::new(st.reader().expect("reader"));
        let red = bodies_kept(|n| mapped_tag(n) == "red");
        let every = bodies_kept(|_| true);
        let runs: Vec<_> = (0..MAPPED_CONCURRENT)
            .map(|i| {
                let reader = Arc::clone(&reader);
                let sql = match i % 2 {
                    0 => shared_search(MAPPED_EVERY_HIT, "tag = 'red'"),
                    _ => format!(
                        "SELECT body FROM bm25_search('body', 'shared', {MAPPED_EVERY_HIT})"
                    ),
                };
                spawn_blocking(move || (i, bodies_of(&reader, &sql)))
            })
            .collect();
        for run in runs {
            let (i, got) = run.await.expect("query task");
            let want = match i % 2 {
                0 => &red,
                _ => &every,
            };
            assert_eq!(&got, want, "query {i}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_where_on_a_reordered_file_with_deletes_ranks_its_live_rows() {
        // A reordered superfile with three deleted rows, under `tag = 'red'`.
        // - Deletes are rows, so this superfile's set is in rows, not blob ids.
        // This test pins that the search ranks exactly the live rows the
        // `WHERE` keeps.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        delete_mapped(&st);
        let reader = st.reader().expect("reader");
        assert_eq!(
            bodies_of(&reader, &shared_search(MAPPED_EVERY_HIT, "tag = 'red'")),
            bodies_kept(|n| mapped_tag(n) == "red" && !MAPPED_DELETED.contains(&n)),
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_where_in_blob_ids_ranks_the_same_hits_as_in_rows() {
        // A reordered superfile under `tag = 'red'`, ranked with the scope in blob ids and with
        // the same scope in rows, for a query whose hits score one, two or three terms.
        // - For each `k`, both give the same hits, scores and order.
        // This test pins that checking in blob ids changes no hit, tie or cut at `k`.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let reader = st.reader().expect("reader");
        let filters = vec![col("tag").eq(lit("red"))];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        let red = (0..MAPPED_ROWS).filter(|&n| mapped_tag(n) == "red").count();
        reader.block_on(async {
            let in_blob_ids = reader.rank_scope(&filters, &plan).await.expect("scope");
            let in_rows = RankScope::from(
                reader
                    .candidate_scope(&filters, &plan)
                    .await
                    .expect("scope"),
            );
            assert_in_blob_ids(&in_blob_ids);
            for k in [1, MAPPED_FEW_HITS, red - 1, red, red + 1] {
                let mut hits = Vec::new();
                for scope in [&in_blob_ids, &in_rows] {
                    let opts = Bm25SearchOptions::new();
                    let search = reader.bm25_search_scoped_async(
                        "body",
                        "shared t1 t2",
                        k,
                        opts,
                        Some(scope),
                    );
                    hits.push(search.await.expect("search"));
                }
                assert_eq!(hits[0], hits[1], "k = {k}");
            }
        });
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_blob_id_scope_meeting_new_deletes_ranks_only_live_rows() {
        // A reordered superfile's scope under `tag = 'red'` built in blob ids, then three rows
        // deleted, then searches on the new snapshot under that scope, with `k` small, at the
        // live red count and past it.
        // - The search turns the blob ids into rows and takes the deleted rows out before it
        //   ranks, so it still fills `k`.
        // - It ranks the same hits as a scope built after the deletes, which is in rows.
        // This test pins that a delete seen after the scope was built is never ranked.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let before = st.reader().expect("reader");
        let filters = vec![col("tag").eq(lit("red"))];
        let plan = candidate_plan_for_filters(before.manifest(), &filters);
        let scope = before
            .block_on(before.rank_scope(&filters, &plan))
            .expect("scope");
        assert_in_blob_ids(&scope);
        delete_mapped(&st);
        let after = st.reader().expect("reader");
        let fresh = after
            .block_on(after.rank_scope(&filters, &plan))
            .expect("scope");
        let live = bodies_kept(|n| mapped_tag(n) == "red" && !MAPPED_DELETED.contains(&n)).len();
        for k in [MAPPED_FEW_HITS, live, MAPPED_EVERY_HIT] {
            let mut hits = Vec::new();
            for scope in [&scope, &fresh] {
                let search = after.bm25_search_scoped_async(
                    "body",
                    "shared",
                    k,
                    Bm25SearchOptions::new(),
                    Some(scope),
                );
                hits.push(after.block_on(search).expect("search"));
            }
            assert_eq!(hits[0].len(), k.min(live), "k = {k}: live red rows only");
            assert_eq!(hits[0], hits[1], "k = {k}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn each_query_shape_under_a_blob_id_scope_ranks_the_same_hits_as_in_rows() {
        // A reordered superfile under `tag = 'red'`, searched with each query of
        // `mapped_queries`, with the scope in blob ids and with the same scope in rows.
        // - Each gives exactly the red rows the query matches, the same hits both ways.
        // This test pins that every walk checks a blob-id set as it checks rows.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let reader = st.reader().expect("reader");
        let filters = vec![col("tag").eq(lit("red"))];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        let in_blob_ids = reader
            .block_on(reader.rank_scope(&filters, &plan))
            .expect("scope");
        assert_in_blob_ids(&in_blob_ids);
        let in_rows = RankScope::from(
            reader
                .block_on(reader.candidate_scope(&filters, &plan))
                .expect("scope"),
        );
        for (query, matches) in mapped_queries() {
            let want = (0..MAPPED_ROWS)
                .filter(|&n| mapped_tag(n) == "red" && matches(n))
                .count();
            assert!(want > 0, "{query} matches no red row");
            let mut hits = Vec::new();
            for scope in [&in_blob_ids, &in_rows] {
                let search = reader.bm25_search_scoped_async(
                    "body",
                    query,
                    MAPPED_EVERY_HIT,
                    Bm25SearchOptions::new(),
                    Some(scope),
                );
                hits.push(reader.block_on(search).expect("search"));
            }
            assert_eq!(hits[0].len(), want, "{query}: the red rows it matches");
            assert_eq!(hits[0], hits[1], "{query}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_text_and_range_where_on_a_reordered_file_ranks_exactly_its_rows() {
        // A reordered superfile under `tag = 'red'` and a range on `n`.
        // - The scope stays in rows: a range leaf is checked in rows only.
        // - With `k` above the row count, the hits are exactly the rows both keep.
        // This test pins that a range leaf never goes through blob ids.
        let (st, _storage_dir, _cache_dir) = mapped_body_tag_table(MAPPED_COMMITS);
        let reader = st.reader().expect("reader");
        let [lo, hi] = MAPPED_RANGE;
        let filters = vec![
            col("tag")
                .eq(lit("red"))
                .and(col("n").gt_eq(lit(lo)))
                .and(col("n").lt_eq(lit(hi))),
        ];
        let plan = candidate_plan_for_filters(reader.manifest(), &filters);
        assert!(plan.has_range(), "the plan holds a range leaf");
        let scope = reader
            .block_on(reader.rank_scope(&filters, &plan))
            .expect("scope");
        let sets = scope.allow.as_ref().expect("a bounded scope");
        assert!(sets.values().all(|set| matches!(set, Admitted::Rows(_))));
        let filter = format!("tag = 'red' AND n >= {lo} AND n <= {hi}");
        assert_eq!(
            bodies_of(&reader, &shared_search(MAPPED_EVERY_HIT, &filter)),
            bodies_kept(|n| mapped_tag(n) == "red" && (lo..=hi).contains(&(n as i64))),
        );
    }
}
