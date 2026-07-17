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

use std::{fmt, sync::Arc};

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

use crate::{
    superfile::fts::reader::BoolMode,
    supertable::{
        handle::{SupertableReader, WeakReader},
        query::exec::common::{
            arg_to_string, arg_to_usize, output_schema_with_score, resolve_hits,
            search_query_df_error,
        },
    },
};

/// SQL name for the term-based BM25 TVF.
pub(crate) const BM25_SEARCH_UDTF: &str = "bm25_search";
/// SQL name for the prefix BM25 TVF.
pub(crate) const BM25_PREFIX_UDTF: &str = "bm25_search_prefix";

/// Minimum argument count for `bm25_search(column, query, k)`.
const BM25_SEARCH_ARG_COUNT_MIN: usize = 3;
/// Maximum argument count: the optional `mode` makes it
/// `bm25_search(column, query, k, mode)`.
const BM25_SEARCH_ARG_COUNT_MAX: usize = 4;
/// Argument count for `bm25_search_prefix(column, prefix, k)`.
const BM25_PREFIX_SEARCH_ARG_COUNT: usize = 3;

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
        if args.len() != BM25_SEARCH_ARG_COUNT_MIN && args.len() != BM25_SEARCH_ARG_COUNT_MAX {
            return Err(DataFusionError::Plan(format!(
                "bm25_search expects {BM25_SEARCH_ARG_COUNT_MIN} or {BM25_SEARCH_ARG_COUNT_MAX} \
                 arguments (column, query, k[, mode]), got {}",
                args.len()
            )));
        }
        let column = arg_to_string(&args[0], "bm25_search column")?;
        let query = arg_to_string(&args[1], "bm25_search query")?;
        let k = arg_to_usize(&args[2], "bm25_search k")?;
        let mode = match args.get(3) {
            Some(expr) => arg_to_bool_mode(expr)?,
            None => BoolMode::Or,
        };
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "bm25_search: supertable consumer dropped before execution".into(),
            )
        })?;
        Ok(Arc::new(Bm25Table {
            reader,
            column,
            query: Bm25Query::Terms { query, mode },
            k,
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
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

        if args.len() != BM25_PREFIX_SEARCH_ARG_COUNT {
            return Err(DataFusionError::Plan(format!(
                "bm25_search_prefix expects {BM25_PREFIX_SEARCH_ARG_COUNT} arguments \
                 (column, prefix, k), got {}",
                args.len()
            )));
        }

        let column = arg_to_string(&args[0], "bm25_search_prefix column")?;
        let prefix = arg_to_string(&args[1], "bm25_search_prefix prefix")?;
        let k = arg_to_usize(&args[2], "bm25_search_prefix k")?;
        let reader = self.reader.upgrade().ok_or_else(|| {
            DataFusionError::Execution(
                "bm25_search_prefix: supertable consumer dropped before execution".into(),
            )
        })?;

        Ok(Arc::new(Bm25Table {
            reader,
            column,
            query: Bm25Query::Prefix { prefix },
            k,
            scalar_schema: Arc::clone(&self.scalar_schema),
            output_schema: Arc::clone(&self.output_schema),
        }))
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
        _filters: &[Expr],
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
        )?;
        Ok(Arc::new(exec))
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
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl Bm25Exec {
    fn try_new(
        reader: Arc<SupertableReader>,
        column: String,
        query: Bm25Query,
        k: usize,
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
            scalar_schema,
            output_schema,
            projection,
            projected_schema,
            cache,
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
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            let hits = match &query {
                Bm25Query::Terms { query, mode } => {
                    reader.bm25_search_async(&column, query, k, *mode).await
                }
                Bm25Query::Prefix { prefix } => {
                    reader.bm25_search_prefix_async(&column, prefix, k).await
                }
            }
            .map_err(search_query_df_error)?;
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
    use arrow_array::{Array, Float32Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::lit;

    use super::*;
    use crate::{
        superfile::builder::FtsConfig,
        supertable::{Supertable, SupertableOptions},
        test_helpers::default_tokenizer as tok,
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
        SupertableOptions::new(
            title_schema(),
            vec![FtsConfig {
                column: "title".into(),
                positions: false,
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
        let mut out = Vec::new();
        for b in batches {
            let idx = b.schema().index_of("title").expect("title col");
            let col = b
                .column(idx)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("utf8");
            for i in 0..col.len() {
                out.push(col.value(i).to_string());
            }
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
            .query_sql("SELECT title FROM bm25_search('title', 'rust systems', 10, 'and')")
            .expect("query_sql");
        let and_titles = titles_of(&and_rows);
        assert_eq!(and_titles, vec!["rust systems programming".to_string()]);

        // OR (default): docs 0 + 4 (union of `rust` and `systems`).
        let or_rows = st
            .reader()
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
            .query_sql("SELECT title FROM bm25_search('title', 'rust -systems', 10)")
            .expect("query_sql");
        assert_eq!(titles_of(&rows), vec!["rust async runtime".to_string()]);

        // A negation-only query has nothing to rank → error.
        let res = st
            .reader()
            .query_sql("SELECT title FROM bm25_search('title', '-rust', 10)");
        assert!(res.is_err(), "negation-only must error; got {res:?}");
    }

    #[test]
    fn bm25_search_prefix_tvf_expands_prefix() {
        let st = demo_corpus();
        // `rus` expands to `rust` → docs 0 + 4.
        let batches = st
            .reader()
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
            .query_sql("SELECT title, score FROM bm25_search('title', 'rust', 5)")
            .expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn bm25_search_tvf_arity_error() {
        let st = demo_corpus();
        // 2 args (missing k) → planning error, surfaced as QueryError::Plan.
        assert!(
            st.reader()
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
                .query_sql("SELECT title FROM bm25_search('title', 'rust', 'ten')")
                .is_err(),
            "non-integer k must error"
        );
        // Invalid mode literal.
        assert!(
            st.reader()
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
    #[tokio::test]
    async fn bm25_table_and_exec_trait_methods() {
        use crate::supertable::query::exec::common::test_support::call_tvf;
        let st = demo_corpus();
        let reader = Arc::new(st.reader());
        let scalar_schema = reader.options().scalar_schema();
        let func = Bm25SearchFunc::new(reader, scalar_schema);
        let table = call_tvf(&func, &[lit("title"), lit("rust"), lit(10_i64)]).expect("bm25 table");

        // TableProvider metadata.
        let dbg = format!("{table:?}");
        assert!(dbg.contains("Bm25Table"), "Debug missing: {dbg}");
        assert!(
            table.downcast_ref::<Bm25Table>().is_some(),
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
}
