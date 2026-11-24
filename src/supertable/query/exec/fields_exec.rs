// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Multi-field BM25 as a DataFusion table-valued function.
//!
//! `bm25_search_fields(clauses, k [, mode [, weights]])` ranks one query
//! over several full-text columns, scored *best fields*: each clause scores
//! a row by its best column, and a row's score is the sum of the clauses it
//! matches (see [`multi_field`](crate::supertable::query::multi_field)). It
//! returns the same columns as `bm25_search`: `_id`, the scalar columns, and
//! `score`, best first.
//!
//! ## Query shape
//!
//! ```sql
//! SELECT _id, score FROM bm25_search_fields(
//!     '[{"query": "error", "fields": ["title", "body"]},
//!       {"query": "timeout", "fields": ["body"]}]',
//!     10) ORDER BY score DESC;
//! SELECT _id, score FROM bm25_search_fields(
//!     '[{"query": "error timeout", "fields": ["title", "body"]}]',
//!     10, 'and', '{"title": 2.0}') ORDER BY score DESC;
//! ```
//!
//! `clauses` is a JSON array of `{"query": string, "fields": [string]}`;
//! `mode` is `'or'` (the default: a row matching any clause is a hit) or
//! `'and'` (it must match every clause), and also the mode of each clause's
//! bare terms; `weights` is a JSON object multiplying each named column's
//! scores (1 for a column it does not name).
//!
//! ## `WHERE` on the function
//!
//! As for `bm25_search`: the answer is the top `k` among rows satisfying the
//! predicate. The predicate scopes every per-column search, the exact
//! predicate is applied to the rows the ranking returns, and the search
//! widens until `k` survive or the fill's ceiling is reached. Stopped short
//! below the table's size, the function checks the predicate on every row
//! some clause matches on some column and ranks once more within exactly
//! the ones it keeps.

use std::{collections::HashMap, fmt, sync::Arc};

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
use futures::future::try_join_all;
use roaring::RoaringBitmap;

use crate::{
    superfile::fts::reader::BoolMode,
    supertable::{
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        manifest::SuperfileUri,
        query::{
            SuperfileHit,
            candidate::CandidateScope,
            exec::{
                common::{
                    Arity, PushedPredicate, arg_to_string, arg_to_usize,
                    candidate_plan_for_filters, exact_scope, fill_ceiling, fill_top_k_within,
                    live_reader, output_schema_with_score, scope_to_call, top_k_statistics,
                    traced_tvf, widening_ended_short,
                },
                fts_exec::arg_to_bool_mode,
            },
            multi_field::{Bm25FieldsOptions, FieldsQuery, clauses_from_json, weights_from_json},
        },
    },
    utils::trace::detail_span,
};

/// SQL name for the multi-field BM25 TVF.
pub(crate) const BM25_FIELDS_UDTF: &str = "bm25_search_fields";

/// Arguments of `bm25_search_fields(clauses, k)`; `mode` and `weights` are
/// optional, in that order.
pub(crate) const BM25_FIELDS_ARITY: Arity = Arity {
    min: 2,
    max: 4,
    arguments: "clauses, k[, mode[, weights]]",
};

/// Register `bm25_search_fields` on `ctx`, bound to the query's pinned
/// `reader` + scalar `schema`.
pub(crate) fn register_bm25_fields(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        BM25_FIELDS_UDTF,
        Arc::new(Bm25FieldsFunc::new(reader, scalar_schema)),
    );
}

/// `TableFunctionImpl` for `bm25_search_fields`.
#[derive(Debug)]
pub(crate) struct Bm25FieldsFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl Bm25FieldsFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for Bm25FieldsFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();
        BM25_FIELDS_ARITY.check(BM25_FIELDS_UDTF, args.len())?;
        let clauses = arg_to_string(&args[0], "bm25_search_fields clauses")?;
        let k = arg_to_usize(&args[1], "bm25_search_fields k")?;
        let mode = match args.get(2) {
            Some(expr) => arg_to_bool_mode(expr)?,
            None => BoolMode::Or,
        };
        let mut opts = Bm25FieldsOptions::new().with_mode(mode);
        if let Some(expr) = args.get(3) {
            let weights = arg_to_string(expr, "bm25_search_fields weights")?;
            opts.weights = weights_from_json(&weights)?;
        }
        let query = FieldsQuery::new(&clauses_from_json(&clauses)?, &opts)?;
        let reader = live_reader(&self.reader, BM25_FIELDS_UDTF)?;
        scope_to_call(
            BM25_FIELDS_UDTF,
            Arc::new(Bm25FieldsTable {
                reader,
                query,
                k,
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
            }),
        )
    }
}

/// One parsed `bm25_search_fields` call as a `TableProvider`; `scan` lowers
/// to [`Bm25FieldsExec`].
struct Bm25FieldsTable {
    reader: Arc<SupertableReader>,
    query: FieldsQuery,
    k: usize,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for Bm25FieldsTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bm25FieldsTable")
            .field("query", &self.query)
            .field("k", &self.k)
            .finish()
    }
}

#[async_trait]
impl TableProvider for Bm25FieldsTable {
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
        let exec = Bm25FieldsExec::try_new(
            Arc::clone(&self.reader),
            self.query.clone(),
            self.k,
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
            filters.to_vec(),
        )?;
        Ok(Arc::new(exec))
    }

    /// Every `WHERE` filter is `Inexact`, as for `bm25_search`: pushed into
    /// [`scan`](Self::scan) to scope the search, and re-applied above it.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/// Runs the multi-field kernel inside `execute()` and emits the resolved
/// `_id` + scalar columns + `score`.
struct Bm25FieldsExec {
    reader: Arc<SupertableReader>,
    query: FieldsQuery,
    k: usize,
    filters: Vec<Expr>,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
    projection: Option<Vec<usize>>,
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
    statistics: Arc<Statistics>,
}

impl Bm25FieldsExec {
    fn try_new(
        reader: Arc<SupertableReader>,
        query: FieldsQuery,
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

    fn describe(&self) -> String {
        let columns: Vec<&str> = self.query.searches().map(|(column, _)| column).collect();
        format!(
            "Bm25FieldsExec: columns=[{}], k={}, mode={:?}",
            columns.join(", "),
            self.k,
            self.query.mode()
        )
    }
}

impl fmt::Debug for Bm25FieldsExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for Bm25FieldsExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for Bm25FieldsExec {
    fn name(&self) -> &'static str {
        "Bm25FieldsExec"
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
                "Bm25FieldsExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let query = self.query.clone();
        let k = self.k;
        let filters = self.filters.clone();
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);

        let fut = async move {
            let scope = match filters.is_empty() {
                true => None,
                false => {
                    let plan = candidate_plan_for_filters(reader.manifest(), &filters);
                    Some(reader.candidate_scope(&filters, &plan).await?)
                }
            };
            let predicate = PushedPredicate::compile(&filters, &output_schema);
            let call = FieldsCall {
                reader: &reader,
                query: &query,
                k,
                predicate: predicate.as_ref(),
                scalar_schema: &scalar_schema,
                output_schema: &output_schema,
                projection: projection.as_deref(),
            };
            let widened = call.rows(scope.as_ref(), None).await?;
            let (Some(scope), Some(predicate)) = (scope.as_ref(), predicate.as_ref()) else {
                return Ok(widened);
            };
            if !widening_ended_short(&reader, k, widened.num_rows()) {
                return Ok(widened);
            }
            // Short at the ceiling: check the predicate on every row some
            // clause matches on some column, and rank once within exactly
            // the ones it keeps.
            let matched = try_join_all(
                query
                    .searches()
                    .map(|(column, text)| reader.match_scope(column, text, query.mode(), scope)),
            )
            .await?;
            let matched = union_scopes(matched);
            match exact_scope(&reader, &matched, predicate, &output_schema).await? {
                Some(exact) => call.rows(Some(&exact), Some(k)).await,
                None => Ok(widened),
            }
        };

        let span = detail_span!(
            "tvf.bm25_search_fields",
            k = self.k,
            rows_out = tracing::field::Empty
        );
        let fut = traced_tvf(span, fut);
        let stream = futures::stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// One `bm25_search_fields` call's fixed inputs, borrowed by each ranking
/// round.
struct FieldsCall<'a> {
    reader: &'a SupertableReader,
    query: &'a FieldsQuery,
    k: usize,
    predicate: Option<&'a PushedPredicate>,
    scalar_schema: &'a SchemaRef,
    output_schema: &'a SchemaRef,
    projection: Option<&'a [usize]>,
}

impl FieldsCall<'_> {
    async fn hits(
        &self,
        want: usize,
        scope: Option<&CandidateScope>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.reader
            .bm25_search_fields_scoped_async(self.query, want, scope)
            .await
    }

    /// The call's `k` rows within `scope` under the predicate, widening to
    /// at most `ceiling` hits a round (`None` is the fill's own ceiling).
    async fn rows(
        &self,
        scope: Option<&CandidateScope>,
        ceiling: Option<usize>,
    ) -> DfResult<RecordBatch> {
        let ceiling = ceiling.unwrap_or_else(|| fill_ceiling(self.reader, self.k));
        fill_top_k_within(
            self.reader,
            self.k,
            self.predicate,
            self.scalar_schema,
            self.output_schema,
            self.projection,
            ceiling,
            false,
            |want: usize| self.hits(want, scope),
        )
        .await
    }
}

/// The rows any of `scopes` admits. Each is bounded (what
/// [`SupertableReader::match_scope`] returns), so the union is too: a
/// superfile's set is the union of its sets.
fn union_scopes(scopes: Vec<CandidateScope>) -> CandidateScope {
    let mut superfiles = Vec::new();
    let mut allow: HashMap<SuperfileUri, RoaringBitmap> = HashMap::new();
    for scope in scopes {
        let sets = scope.allow.unwrap_or_default();
        for entry in scope.superfiles {
            let Some(rows) = sets.get(&entry.uri) else {
                continue;
            };
            match allow.get_mut(&entry.uri) {
                Some(kept) => *kept |= rows.as_ref(),
                None => {
                    allow.insert(entry.uri, rows.as_ref().clone());
                    superfiles.push(entry);
                }
            }
        }
    }
    CandidateScope {
        superfiles,
        allow: Some(
            allow
                .into_iter()
                .map(|(uri, rows)| (uri, Arc::new(rows)))
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::compute::cast;
    use arrow_array::{Array, Int64Array};
    use arrow_schema::DataType;

    use super::*;
    use crate::{
        Bm25SearchOptions,
        supertable::{
            Supertable,
            query::{
                exec::common::OVER_FETCH_MIN_CEILING,
                multi_field::{
                    FieldClause,
                    tests::{
                        Id, Row, corpus_rows, corpus_table, ids_scores, multi_file_corpus, oracle,
                        query_cases, row_ids,
                    },
                },
            },
        },
    };

    /// `clauses` as the TVF's JSON argument, quotes doubled for SQL.
    fn clauses_sql(clauses: &[FieldClause]) -> String {
        let json = serde_json::Value::Array(
            clauses
                .iter()
                .map(|c| serde_json::json!({ "query": c.query(), "fields": c.fields() }))
                .collect(),
        )
        .to_string();
        format!("'{}'", json.replace('\'', "''"))
    }

    fn sql(st: &Supertable, statement: &str) -> Vec<RecordBatch> {
        st.reader()
            .expect("reader")
            .query_sql(statement)
            .unwrap_or_else(|e| panic!("{statement}: {e}"))
    }

    /// `column` of every row `clauses` matches, by `_id`, read through the
    /// single-column searches.
    fn bucket_by_id(st: &Supertable, clauses: &[FieldClause]) -> HashMap<Id, i64> {
        let mut out = HashMap::new();
        for clause in clauses {
            for column in clause.fields() {
                let rows = st
                    .bm25_search(
                        column,
                        clause.query(),
                        100_000,
                        Bm25SearchOptions::new(),
                        Some(&["_id", "bucket"]),
                    )
                    .expect("search");
                for b in &rows {
                    let ids = row_ids(b.column(0));
                    let buckets = b
                        .column(1)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("i64");
                    for (i, id) in ids.iter().enumerate() {
                        out.insert(*id, buckets.value(i));
                    }
                }
            }
        }
        out
    }

    /// The SQL form returns what the Rust API returns, under each mode and
    /// with weights.
    #[test]
    fn sql_round_trip_matches_the_rust_api() {
        let st = multi_file_corpus();
        for clauses in query_cases() {
            for (mode, mode_sql) in [(BoolMode::Or, "or"), (BoolMode::And, "and")] {
                let opts = Bm25FieldsOptions::new()
                    .with_mode(mode)
                    .with_weight("title", 2.0);
                let want = oracle(&st, &clauses, 10, &opts, None);
                let got = ids_scores(&sql(
                    &st,
                    &format!(
                        "SELECT _id, score FROM bm25_search_fields({}, 10, '{mode_sql}', \
                         '{{\"title\": 2.0}}') ORDER BY score DESC, _id ASC",
                        clauses_sql(&clauses)
                    ),
                ));
                assert_eq!(got, want, "{clauses:?} {mode_sql}");
            }
            // The two-argument form: `or`, unweighted.
            let want = oracle(&st, &clauses, 5, &Bm25FieldsOptions::new(), None);
            let got = ids_scores(&sql(
                &st,
                &format!(
                    "SELECT _id, score FROM bm25_search_fields({}, 5) \
                     ORDER BY score DESC, _id ASC",
                    clauses_sql(&clauses)
                ),
            ));
            assert_eq!(got, want, "{clauses:?}");
        }
    }

    /// `SELECT *` returns `_id`, the scalar columns, and `score`.
    #[test]
    fn star_projection_has_scalar_columns_and_score() {
        let st = multi_file_corpus();
        let rows = sql(
            &st,
            r#"SELECT * FROM bm25_search_fields('[{"query": "rust", "fields": ["title", "body"]}]', 4)"#,
        );
        let schema = rows[0].schema();
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(names, ["_id", "title", "body", "tags", "bucket", "score"]);
        assert_eq!(rows.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
    }

    /// A `WHERE` returns the top `k` among the rows it keeps: the
    /// brute-force ranking restricted to them, on an unindexed column and
    /// on a full-text one.
    #[test]
    fn where_ranks_among_the_rows_it_keeps() {
        let st = multi_file_corpus();
        for clauses in query_cases() {
            let bucket = bucket_by_id(&st, &clauses);
            for (mode, mode_sql) in [(BoolMode::Or, "or"), (BoolMode::And, "and")] {
                let opts = Bm25FieldsOptions::new().with_mode(mode);
                for k in [1, 5, 50] {
                    let keep = |id: Id| bucket.get(&id) == Some(&2);
                    let want = oracle(&st, &clauses, k, &opts, Some(&keep));
                    let got = ids_scores(&sql(
                        &st,
                        &format!(
                            "SELECT _id, score FROM bm25_search_fields({}, {k}, '{mode_sql}') \
                             WHERE bucket = 2 ORDER BY score DESC, _id ASC",
                            clauses_sql(&clauses)
                        ),
                    ));
                    assert_eq!(got, want, "{clauses:?} {mode_sql} k={k}");
                }
            }
        }
        // A full-text column the index bounds: every row returned keeps
        // the predicate, still ranked.
        let rows = sql(
            &st,
            r#"SELECT tags, score FROM bm25_search_fields('[{"query": "rust cloud", "fields": ["title", "body"]}]', 8) WHERE tags = 'web'"#,
        );
        assert!(rows.iter().map(RecordBatch::num_rows).sum::<usize>() > 0);
        let all = sql(
            &st,
            r#"SELECT COUNT(*) AS n FROM bm25_search_fields('[{"query": "rust cloud", "fields": ["title", "body"]}]', 100000) WHERE tags = 'web'"#,
        );
        let n = all[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count")
            .value(0);
        let got: usize = rows.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(got, usize::try_from(n).expect("count").min(8));
        for b in &rows {
            let tags = cast(b.column(0), &DataType::Utf8).expect("utf8");
            let tags = tags
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .expect("utf8");
            for i in 0..b.num_rows() {
                assert_eq!(tags.value(i), "web");
            }
        }
    }

    /// Rows the predicate keeps that rank below the widening's ceiling are
    /// still found: the function falls back to checking the predicate on
    /// every row some clause matches, and ranks within the ones it keeps.
    #[test]
    fn where_reaches_rows_below_the_fill_ceiling() {
        let n_strong = OVER_FETCH_MIN_CEILING + 100;
        let n_weak = 5;
        let mut rows: Vec<Row> = (0..n_strong)
            .map(|_| ("rust rust".into(), "rust".into(), "db".into(), 0))
            .collect();
        rows.extend((0..n_weak).map(|_| {
            (
                "rust with four more words".into(),
                "x".into(),
                "web".into(),
                1,
            )
        }));
        let st = corpus_table(&rows, rows.len() / 3);
        let clauses = [FieldClause::new("rust", ["title", "body"])];
        let top = ids_scores(
            &st.bm25_search_fields(
                &clauses,
                OVER_FETCH_MIN_CEILING,
                &Bm25FieldsOptions::new(),
                Some(&["_id", "bucket", "score"]),
            )
            .expect("unfiltered"),
        );
        assert_eq!(top.len(), OVER_FETCH_MIN_CEILING);
        for k in [3, 10] {
            let got = sql(
                &st,
                &format!(
                    "SELECT title, bucket, score FROM bm25_search_fields({}, {k}) \
                     WHERE bucket = 1",
                    clauses_sql(&clauses)
                ),
            );
            let n: usize = got.iter().map(RecordBatch::num_rows).sum();
            assert_eq!(n, k.min(n_weak), "k={k}");
            for b in &got {
                let buckets = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("i64");
                assert!((0..b.num_rows()).all(|i| buckets.value(i) == 1));
            }
        }
    }

    #[test]
    fn bad_arguments_are_planning_errors() {
        let st = multi_file_corpus();
        let reader = st.reader().expect("reader");
        let clauses = r#"'[{"query": "rust", "fields": ["title"]}]'"#;
        for (statement, want) in [
            (
                format!("SELECT * FROM bm25_search_fields({clauses})"),
                "bm25_search_fields expects",
            ),
            (
                format!("SELECT * FROM bm25_search_fields({clauses}, 3, 'or', '{{}}', 9)"),
                "bm25_search_fields expects",
            ),
            (
                "SELECT * FROM bm25_search_fields('[{\"query\": \"rust\"}]', 3)".to_string(),
                "clauses must be a JSON array",
            ),
            (
                format!("SELECT * FROM bm25_search_fields({clauses}, 3, 'xor')"),
                "mode must be 'or' or 'and'",
            ),
            (
                format!("SELECT * FROM bm25_search_fields({clauses}, 3, 'or', '[2]')"),
                "weights must be a JSON object",
            ),
            (
                format!("SELECT * FROM bm25_search_fields({clauses}, 3, 'or', '{{\"title\": 0}}')"),
                "finite and greater than 0",
            ),
            (
                "SELECT * FROM bm25_search_fields('[]', 3)".to_string(),
                "at least one clause",
            ),
        ] {
            let err = reader
                .query_sql(&statement)
                .expect_err(&statement)
                .to_string();
            assert!(err.contains(want), "{statement}: {err}");
        }
    }

    /// Union of match scopes keeps each superfile once with the union of
    /// its rows.
    #[test]
    fn union_scopes_unions_rows_per_superfile() {
        let st = corpus_table(&corpus_rows(80), 40);
        let reader = st.reader().expect("reader");
        let scope = reader
            .block_on(reader.candidate_scope(
                &[],
                &crate::supertable::query::candidate::CandidatePlan::Unbounded,
            ))
            .expect("scope");
        let a = reader
            .block_on(reader.match_scope("title", "rust", BoolMode::Or, &scope))
            .expect("a");
        let b = reader
            .block_on(reader.match_scope("body", "cloud", BoolMode::Or, &scope))
            .expect("b");
        let rows = |s: &CandidateScope| -> HashMap<SuperfileUri, RoaringBitmap> {
            s.allow
                .as_ref()
                .expect("bounded")
                .iter()
                .map(|(uri, rows)| (*uri, rows.as_ref().clone()))
                .collect()
        };
        let (ra, rb) = (rows(&a), rows(&b));
        let u = union_scopes(vec![a, b]);
        let ru = rows(&u);
        let mut uris: Vec<_> = u.superfiles.iter().map(|e| e.uri).collect();
        let n = uris.len();
        uris.sort();
        uris.dedup();
        assert_eq!(uris.len(), n, "each superfile once");
        for (uri, set) in &ru {
            let mut want = ra.get(uri).cloned().unwrap_or_default();
            want |= rb.get(uri).cloned().unwrap_or_default();
            assert_eq!(set, &want);
        }
        assert!(ra.keys().chain(rb.keys()).all(|uri| ru.contains_key(uri)));
    }
}
