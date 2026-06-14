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

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_schema::SchemaRef;
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

use super::common::{arg_to_string, output_schema_with_score, resolve_hits};
use super::fts_exec::arg_to_bool_mode;
use crate::superfile::fts::reader::BoolMode;
use crate::supertable::handle::{SupertableReader, WeakReader};

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
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
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
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "token_match: supertable consumer dropped before execution".into(),
            )
        })?;
        Ok(Arc::new(MatchTable {
            reader,
            column,
            query: MatchQuery::Token { query, mode },
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
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
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        if args.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "exact_match expects 2 arguments (column, value), got {}",
                args.len()
            )));
        }
        let column = arg_to_string(&args[0], "exact_match column")?;
        let value = arg_to_string(&args[1], "exact_match value")?;
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "exact_match: supertable consumer dropped before execution".into(),
            )
        })?;
        Ok(Arc::new(MatchTable {
            reader,
            column,
            query: MatchQuery::Exact { value },
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
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

        let fut = async move {
            let hits = match &query {
                MatchQuery::Token { query, mode } => {
                    reader.token_match_async(&column, query, *mode).await
                }
                MatchQuery::Exact { value } => reader.exact_match_async(&column, value).await,
            }
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};

    use crate::superfile::builder::FtsConfig;
    use crate::supertable::{Supertable, SupertableOptions};
    use crate::test_helpers::default_tokenizer as tok;

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
        SupertableOptions::new(
            title_schema(),
            vec![FtsConfig {
                column: "title".into(),
            }],
            vec![],
            Some(tok()),
        )
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

    fn rows(st: &Supertable, sql: &str) -> usize {
        st.reader()
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
                .query_sql("SELECT _id FROM token_match('title')")
                .is_err(),
            "token_match needs >= 2 args"
        );
        assert!(
            st.reader()
                .query_sql("SELECT _id FROM exact_match('title')")
                .is_err(),
            "exact_match needs 2 args"
        );
    }

    #[test]
    fn public_methods_agree_with_tvfs() {
        let st = demo();
        let reader = st.reader();
        let method = reader
            .token_match(
                "title",
                "rust systems",
                crate::superfile::fts::reader::BoolMode::And,
            )
            .expect("token_match");
        assert_eq!(method.len(), 1);
        let exact = reader
            .exact_match("title", "go routines")
            .expect("exact_match");
        assert_eq!(exact.len(), 1);
    }
}
