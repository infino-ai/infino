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
use futures::{TryStreamExt, stream};

use crate::{
    superfile::fts::reader::BoolMode,
    supertable::{
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        query::exec::{
            common::{
                arg_to_string, live_reader, output_schema_with_score, resolve_hits, scope_to_call,
                traced_tvf, traced_tvf_rows,
            },
            fts_exec::arg_to_bool_mode,
        },
    },
    utils::trace::detail_span,
};

/// SQL name for the unranked token-match TVF.
pub(crate) const TOKEN_MATCH_UDTF: &str = "token_match";
/// SQL name for the unranked exact-match TVF.
pub(crate) const EXACT_MATCH_UDTF: &str = "exact_match";

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
        if args.len() != 2 && args.len() != 3 {
            return Err(DataFusionError::Plan(format!(
                "token_match expects 2 or 3 arguments (column, query[, mode]), got {}",
                args.len()
            )));
        }
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
        if args.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "exact_match expects 2 arguments (column, value), got {}",
                args.len()
            )));
        }
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
            }),
        )
    }
}

/// One parsed match invocation as a `TableProvider`. `scan` lowers to
/// [`MatchExec`].
struct MatchTable {
    reader: Arc<SupertableReader>,
    column: String,
    query: MatchQuery,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for MatchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchTable")
            .field("column", &self.column)
            .field("query", &self.query)
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
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let exec = MatchExec::try_new(
            Arc::clone(&self.reader),
            self.column.clone(),
            self.query.clone(),
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
        )?;
        Ok(Arc::new(exec))
    }
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
            cache,
        })
    }

    fn describe(&self) -> String {
        match &self.query {
            MatchQuery::Token { mode, .. } => {
                format!(
                    "MatchExec: kind=token, column={}, mode={:?}",
                    self.column, mode
                )
            }
            MatchQuery::Exact { .. } => {
                format!("MatchExec: kind=exact, column={}", self.column)
            }
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
        let counts_only = matches!(&projection, Some(indices) if indices.is_empty());
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
                    .token_match_columns_async(&column, &query, mode, &names)
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
            let hits = match &query {
                MatchQuery::Token { query, mode } => {
                    reader
                        .token_match_hits_async(&column, query, *mode, needs_ids)
                        .await
                }
                MatchQuery::Exact { value } => reader.exact_match_async(&column, value).await,
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

    use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        error::DataFusionError,
        execution::TaskContext,
        physical_plan::{DisplayFormatType, ExecutionPlan},
        prelude::{col, lit},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;
    use tokio::runtime::Runtime;

    use super::{ExactMatchFunc, MatchExec, MatchQuery, TokenMatchFunc};
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{builder::FtsConfig, fts::reader::BoolMode},
        supertable::{
            Supertable, SupertableOptions, error::QueryError, handle::SupertableReader,
            query::exec::common::output_schema_with_score,
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

    fn rows(st: &Supertable, sql: &str) -> usize {
        st.reader()
            .expect("reader")
            .query_sql(sql)
            .expect("query_sql")
            .iter()
            .map(RecordBatch::num_rows)
            .sum()
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
            let counted = rows(st, &format!("SELECT _id FROM {tvf}"));
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
            let twice = rows(st, &format!("SELECT title AS a, title AS b FROM {tvf}"));
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
            .block_on(reader.token_match_columns_async("title", query, mode, &["title"]))
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
            let matched = rows(
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

    #[test]
    fn token_match_tvf_or_unions_and_intersects() {
        let st = demo();
        // OR (default): rows containing `rust` OR `python` → 0, 1, 2.
        assert_eq!(
            rows(&st, "SELECT _id FROM token_match('title', 'rust python')"),
            3
        );
        // AND: rows containing both `rust` and `systems` → only doc 2.
        assert_eq!(
            rows(
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
            rows(&st, "SELECT _id FROM exact_match('title', 'rust async')"),
            1
        );
        assert_eq!(
            rows(
                &st,
                "SELECT _id FROM exact_match('title', 'rust async runtime')"
            ),
            1
        );
        assert_eq!(rows(&st, "SELECT _id FROM exact_match('title', 'rust')"), 0);
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
}
