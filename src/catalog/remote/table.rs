// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RemoteTable`] — a table handle backed by the hosted service.
//!
//! Implements the internal [`Table`] trait by forwarding each operation over
//! the wire through its [`RemoteCatalog`]. Every query, mutation and schema
//! change is supported; `optimize` and `gc` are deliberately not — on a hosted
//! table those are the platform's (server-side) concern, so they report that
//! rather than forwarding.

#[cfg(any(test, feature = "test-helpers"))]
use std::any::Any;
use std::{sync::Arc, time::Duration};

use std::{mem, ops::ControlFlow};

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use datafusion::{
    common::TableReference,
    datasource::{empty::EmptyTable, provider_as_source},
    error::DataFusionError,
    logical_expr::LogicalPlanBuilder,
    prelude::{Expr, lit},
    sql::{
        sqlparser::ast::{self, BinaryOperator, VisitMut, visit_expressions_mut},
        unparser::Unparser,
    },
};
use serde_json::{Value, json};

use super::{RemoteCatalog, read_arrow, read_json, wire};
use crate::{
    Bm25SearchOptions, BoolMode, GcError, GcReport, InfinoError, MutationStats, OptimizeError,
    OptimizeOptions, ReindexError, ReindexOptions, Supertable, VectorFilter,
    catalog::table::Table,
    dynamic::rows_to_batch,
    superfile::VectorSearchOptions,
    supertable::{
        options::default_id_column,
        query::{
            sorted::sorted_statement,
            sql::{
                conform_value_counts, count_where_exprs, count_where_row, holds_subquery,
                value_count_exprs, value_count_schema,
            },
        },
        reindex::{PlannedRepair, ReindexReport, StalenessReport},
        schema::{TableSchema, change::SchemaPatch},
    },
};

/// A hosted table handle. Holds its `RemoteCatalog`, the table name, and the
/// schema fetched at create/open (so `schema()` is infallible, matching the
/// local handle).
pub(crate) struct RemoteTable {
    catalog: Arc<RemoteCatalog>,
    table_name: String,
    schema: SchemaRef,
}

impl RemoteTable {
    pub(crate) fn new(catalog: Arc<RemoteCatalog>, table_name: String, schema: SchemaRef) -> Self {
        Self {
            catalog,
            table_name,
            schema,
        }
    }
}

/// Wire spelling for a boolean search mode (`"and"` / `"or"`).
fn mode_str(mode: BoolMode) -> &'static str {
    match mode {
        BoolMode::And => "and",
        BoolMode::Or => "or",
    }
}

/// The unparser every statement for the wire is printed with: pretty, so an
/// expression keeps only the parentheses its operators' precedence needs.
fn unparser() -> Unparser<'static> {
    Unparser::default().with_pretty(true)
}

/// `node` with every chain of ANDs or of ORs printed flat. The unparser wraps
/// each binary expression in parentheses, and its pretty pass does not reach
/// an aggregate's `FILTER (WHERE …)`, so a flat chain of N ORs (a predicate
/// over many columns, say) is printed nested N levels deep, and a server whose
/// parser keeps DataFusion's default recursion limit refuses it. AND and OR are
/// associative in SQL's three-valued logic too, so the parentheses between two
/// of the same operator change nothing; any other keeps its own.
fn flat_chains<V: VisitMut>(mut node: V) -> V {
    let _ = visit_expressions_mut(&mut node, |expr| {
        if let ast::Expr::BinaryOp { left, op, right } = expr
            && matches!(op, BinaryOperator::And | BinaryOperator::Or)
        {
            for side in [left, right] {
                if let ast::Expr::Nested(inner) = side.as_mut()
                    && matches!(inner.as_ref(), ast::Expr::BinaryOp { op: inner_op, .. } if inner_op == op)
                {
                    let unwrapped =
                        mem::replace(inner.as_mut(), ast::Expr::Value(ast::Value::Null.into()));
                    **side = unwrapped;
                }
            }
        }
        ControlFlow::<()>::Continue(())
    });
    node
}

/// Render a mutation predicate to SQL for the wire. The public API takes a
/// DataFusion `Expr`, but the endpoint takes a SQL string (the server parses it
/// back against the table's schema), so unparse it here.
fn predicate_to_sql(predicate: &Expr) -> Result<String, InfinoError> {
    unparser()
        .expr_to_sql(predicate)
        .map(|sql| flat_chains(sql).to_string())
        .map_err(|e| {
            InfinoError::Query(format!(
                "cannot express this predicate over the remote transport: {e}"
            ))
        })
}

/// `SELECT column, COUNT(*) FROM table GROUP BY column` over a hosted table,
/// built as a plan over a table of `schema` and unparsed, so the table's and
/// the column's names are quoted wherever SQL needs them.
fn value_count_sql(table: &str, schema: &SchemaRef, column: &str) -> Result<String, InfinoError> {
    let (group, aggregate) = value_count_exprs(column);
    let source = provider_as_source(Arc::new(EmptyTable::new(Arc::clone(schema))));
    LogicalPlanBuilder::scan(TableReference::bare(table), source, None)
        .and_then(|scan| scan.aggregate(group, aggregate))
        .and_then(|grouped| grouped.build())
        .and_then(|plan| unparser().plan_to_sql(&plan))
        .map(|statement| flat_chains(statement).to_string())
        .map_err(|e| {
            InfinoError::Query(format!(
                "cannot express value_counts over the remote transport: {e}"
            ))
        })
}

/// `SELECT COUNT(*) FILTER (WHERE p) AS count_0, ... FROM table [WHERE filter]`
/// over a hosted table, built as a plan over a table of `schema` and unparsed
/// as [`value_count_sql`] is.
fn count_where_sql(
    table: &str,
    schema: &SchemaRef,
    filter: Option<&Expr>,
    predicates: &[Expr],
) -> Result<String, InfinoError> {
    let fail = |e: DataFusionError| {
        InfinoError::Query(format!(
            "cannot express count_where over the remote transport: {e}"
        ))
    };
    let aggregates = count_where_exprs(predicates.iter().cloned()).map_err(fail)?;
    let source = provider_as_source(Arc::new(EmptyTable::new(Arc::clone(schema))));
    LogicalPlanBuilder::scan(TableReference::bare(table), source, None)
        .and_then(|scan| match filter {
            Some(filter) => scan.filter(filter.clone()),
            None => Ok(scan),
        })
        .and_then(|scan| scan.aggregate(Vec::<Expr>::new(), aggregates))
        .and_then(|counted| counted.build())
        .and_then(|plan| unparser().plan_to_sql(&plan))
        .map(|statement| flat_chains(statement).to_string())
        .map_err(fail)
}

/// The statements a hosted `count_where` sends, each with the number of
/// counts it answers, in the predicates' order: one aggregate of every
/// predicate's `COUNT(*) FILTER`, or, when a predicate holds a subquery,
/// which a FILTER clause cannot, one count per predicate under its own
/// `WHERE`, as a local table counts them.
fn count_where_statements(
    table: &str,
    schema: &SchemaRef,
    predicates: &[Expr],
) -> Result<Vec<(String, usize)>, InfinoError> {
    if predicates.is_empty() {
        return Ok(Vec::new());
    }
    if !predicates.iter().any(holds_subquery) {
        let sql = count_where_sql(table, schema, None, predicates)?;
        return Ok(vec![(sql, predicates.len())]);
    }
    predicates
        .iter()
        .map(|predicate| {
            let sql = count_where_sql(table, schema, Some(predicate), &[lit(true)])?;
            Ok((sql, 1))
        })
        .collect()
}

/// Refuse test-only vector tuning overrides on the remote transport. The
/// public search methods always pass the default options, so this can only
/// fire in a `test-helpers` build using the `_with_options` variants.
fn reject_remote_overrides(opts: &VectorSearchOptions) -> Result<(), InfinoError> {
    if opts.nprobe.is_some() || opts.rerank_mult().is_some() {
        return Err(InfinoError::Query(
            "vector tuning overrides are not supported over the remote transport; \
             hosted tables serve engine-decided settings"
                .to_string(),
        ));
    }
    Ok(())
}

/// Parse an update/delete response (`{matched, n_tombstoned, n_not_found}`)
/// into [`MutationStats`].
fn parse_mutation_stats(op: &str, response: ureq::Response) -> Result<MutationStats, InfinoError> {
    let value = read_json(op, response)?;
    let field = |name: &str| -> Result<usize, InfinoError> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| InfinoError::Backend(format!("{op} response missing `{name}`")))
    };
    Ok(MutationStats::from_remote(
        field("matched")?,
        field("n_tombstoned")?,
        field("n_not_found")?,
    ))
}

impl Table for RemoteTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        let body = wire::batches_to_ipc(std::slice::from_ref(batch))?;
        self.catalog
            .post_arrow("append", &[("table", self.table_name.as_str())], body)?;
        Ok(())
    }

    /// The wire carries `?table=` and nothing else per append, so a source
    /// name has nowhere to travel. Refused rather than silently dropped: a
    /// caller asking for named superfiles and getting unnamed ones would have
    /// no way to tell.
    fn append_named(&self, _batch: &RecordBatch, source_name: &str) -> Result<(), InfinoError> {
        Err(InfinoError::Backend(format!(
            "append_named({source_name:?}) is not available on a hosted table: the append wire \
             carries no source name; use append"
        )))
    }

    /// Rows are mapped here against the Arrow schema the table was opened
    /// with, and travel as a batch; the hosted side applies its own rules
    /// to the batch.
    fn append_rows(&self, rows: &[Value]) -> Result<(), InfinoError> {
        let batch = rows_to_batch(rows, &TableSchema::from_user_schema(&self.schema))
            .map_err(|e| InfinoError::Schema(e).with_context("append_rows", None))?;
        self.append(&batch)
    }

    fn append_rows_named(&self, rows: &[Value], source_name: &str) -> Result<(), InfinoError> {
        let batch = rows_to_batch(rows, &TableSchema::from_user_schema(&self.schema))
            .map_err(|e| InfinoError::Schema(e).with_context("append_rows_named", None))?;
        self.append_named(&batch, source_name)
    }

    fn update_rows(&self, predicate: Expr, rows: &[Value]) -> Result<MutationStats, InfinoError> {
        let batch = rows_to_batch(rows, &TableSchema::from_user_schema(&self.schema))
            .map_err(|e| InfinoError::Schema(e).with_context("update_rows", None))?;
        self.update(predicate, &batch)
    }

    fn update(&self, predicate: Expr, batch: &RecordBatch) -> Result<MutationStats, InfinoError> {
        let predicate = predicate_to_sql(&predicate)?;
        let body = wire::batches_to_ipc(std::slice::from_ref(batch))?;
        let response = self.catalog.post_arrow(
            "update",
            &[
                ("table", self.table_name.as_str()),
                ("predicate", &predicate),
            ],
            body,
        )?;
        parse_mutation_stats("update", response)
    }

    fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        let predicate = predicate_to_sql(&predicate)?;
        let response = self.catalog.post_arrow(
            "delete",
            &[
                ("table", self.table_name.as_str()),
                ("predicate", &predicate),
            ],
            Vec::new(),
        )?;
        parse_mutation_stats("delete", response)
    }

    fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "k": k,
            "mode": mode_str(opts.mode),
            "projection": projection,
        });
        let response = self.catalog.post_json("bm25_search", body)?;
        read_arrow("bm25_search", response)
    }

    fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "mode": mode_str(mode),
            "projection": projection,
        });
        let response = self.catalog.post_json("token_match", body)?;
        read_arrow("token_match", response)
    }

    fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "value": value,
            "projection": projection,
        });
        let response = self.catalog.post_json("exact_match", body)?;
        read_arrow("exact_match", response)
    }

    /// The wire has no sorted call, so the sort it stands for travels as SQL
    /// ([`sorted_statement`]) and the hosted side answers it. A hosted handle
    /// holds only the table's Arrow schema, not which columns carry a range
    /// index, so unlike a local table it does not refuse a column without
    /// one: the statement sorts any column, and returns the same rows a local
    /// table returns for a column it does index.
    fn sorted_search(
        &self,
        column: &str,
        descending: bool,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.catalog.query_sql(&sorted_statement(
            &self.table_name,
            column,
            descending,
            k,
            projection,
            &default_id_column(),
        ))
    }

    /// The wire has no value-count call, so the grouped count it stands for
    /// travels as SQL ([`value_count_sql`]) and the hosted side answers it.
    fn value_counts(&self, column: &str) -> Result<RecordBatch, InfinoError> {
        let field = self
            .schema
            .field_with_name(column)
            .map_err(|_| InfinoError::Query(format!("no column `{column}` to count")))?;
        let schema = value_count_schema(field)?;
        let sql = value_count_sql(&self.table_name, &self.schema, column)?;
        let batches = self.catalog.query_sql(&sql)?;
        Ok(conform_value_counts(&batches, &schema)?)
    }

    /// The wire has no predicate-count call, so the counts travel as SQL
    /// aggregates ([`count_where_statements`]) and the hosted side answers
    /// them.
    fn count_where(&self, predicates: &[Expr]) -> Result<Vec<u64>, InfinoError> {
        let mut counts = Vec::with_capacity(predicates.len());
        for (sql, n) in count_where_statements(&self.table_name, &self.schema, predicates)? {
            let batches = self.catalog.query_sql(&sql)?;
            counts.extend(count_where_row(&batches, n)?);
        }
        Ok(counts)
    }

    fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "mode": mode_str(mode),
        });
        let response = self.catalog.post_json("count", body)?;
        read_json("count", response)?
            .get("count")
            .and_then(Value::as_u64)
            .ok_or_else(|| InfinoError::Backend("count response missing `count`".to_string()))
    }

    /// The hosted API has no tokenize operation yet. Refused rather than
    /// approximated: a caller asking how the index reads a text and getting
    /// some other rule's answer would have no way to tell.
    fn tokenize(&self, column: &str, _text: &str) -> Result<Vec<String>, InfinoError> {
        Err(InfinoError::Backend(format!(
            "tokenize({column:?}) is not available on a hosted table: the wire carries no \
             tokenize operation"
        )))
    }

    fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        opts: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        // Options never cross the wire: the public path always passes the
        // default (serving is engine-decided), and the hosted API carries
        // no tuning fields. A diagnostics-build caller reaching for the
        // test-only overrides against a hosted table hears "no" loudly —
        // silently serving defaults would corrupt whatever sweep or oracle
        // measurement asked for the override.
        reject_remote_overrides(&opts)?;
        let mut body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "k": k,
            "projection": projection,
        });
        if let Some(filter) = filter {
            body["filter"] = json!({
                "field_name": filter.column,
                "query": filter.query,
                "mode": mode_str(filter.mode),
            });
        }
        let response = self.catalog.post_json("vector_search", body)?;
        read_arrow("vector_search", response)
    }

    #[allow(clippy::too_many_arguments)]
    fn hybrid_search(
        &self,
        text_column: &str,
        text_query: &str,
        mode: BoolMode,
        vector_column: &str,
        vector_query: &[f32],
        opts: VectorSearchOptions,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        // See `vector_search`: overrides never cross the wire; loud, not silent.
        reject_remote_overrides(&opts)?;
        let body = json!({
            "table_name": self.table_name,
            "text_field": text_column,
            "text_query": text_query,
            "mode": mode_str(mode),
            "vector_field": vector_column,
            "vector_query": vector_query,
            "k": k,
            "projection": projection,
        });
        let response = self.catalog.post_json("hybrid_search", body)?;
        read_arrow("hybrid_search", response)
    }

    fn optimize(&self, _opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        // Compaction on a hosted table is the platform optimizer's job, not a
        // client's; it is deliberately not exposed over the remote transport.
        Err(OptimizeError::NoStorage)
    }

    fn hydrate(
        &self,
        _batches: &mut dyn RecordBatchReader,
        _target_rows: usize,
    ) -> Result<usize, InfinoError> {
        // Hydrate writes superfiles and commits the manifest directly, which
        // needs the storage backend and writer slot the hosted side owns.
        Err(InfinoError::Unsupported(
            "hydrate is not supported over the remote transport".to_string(),
        ))
    }

    fn reindex(&self, _opts: &ReindexOptions) -> Result<ReindexReport, ReindexError> {
        // A reindex rewrites committed superfiles in place, which is the
        // hosted side's job for the same reason compaction is: it needs the
        // storage backend and the writer slot, neither of which a client
        // holds. Deliberately not exposed over the remote transport.
        Err(ReindexError::NoStorage)
    }

    fn reindex_plan(&self, _opts: &ReindexOptions) -> Result<Vec<PlannedRepair>, ReindexError> {
        // Planning reads every superfile's index metadata, same as the
        // staleness report below and for the same reason.
        Err(ReindexError::NoStorage)
    }

    fn index_staleness(&self, _opts: &ReindexOptions) -> Result<StalenessReport, ReindexError> {
        // Reads every superfile's index metadata off the storage backend a
        // client does not hold. Server-side for the same reason the
        // reindex it describes is.
        Err(ReindexError::NoStorage)
    }

    fn gc(&self, _safety_gap: Duration) -> Result<GcReport, GcError> {
        // Retention/GC on a hosted table is server-side; not a client operation.
        Err(GcError::NoStorage)
    }

    fn companion(&self, _purpose: &str) -> Result<Option<Supertable>, InfinoError> {
        // A hosted table is never created with one: `create_table` refuses
        // companions on a hosted connection.
        Ok(None)
    }

    fn apply_schema(
        &self,
        patch: &SchemaPatch,
        expected: Option<u32>,
    ) -> Result<TableSchema, InfinoError> {
        self.catalog.apply_schema(&self.table_name, patch, expected)
    }

    #[cfg(any(test, feature = "test-helpers"))]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, thread};

    use arrow_array::{RecordBatch, RecordBatchIterator};
    use arrow_schema::{ArrowError, DataType, Field, Schema};
    use datafusion::{
        common::TableReference,
        datasource::{empty::EmptyTable, provider_as_source},
        logical_expr::{LogicalPlanBuilder, in_subquery},
        prelude::{col, lit},
        sql::parser::DFParserBuilder,
    };

    use super::{count_where_statements, value_count_sql};
    use crate::{
        InfinoError,
        catalog::{
            remote::{RemoteCatalog, table::RemoteTable},
            table::Table,
        },
    };

    /// The grouped count a hosted table's `value_counts` sends quotes a
    /// column whose name SQL would otherwise misread, and names the count as
    /// the local call does.
    #[test]
    fn value_counts_over_remote_quote_the_column() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "My Tag",
            DataType::Utf8,
            true,
        )]));
        let sql = value_count_sql("docs", &schema, "My Tag").expect("sql");
        assert!(sql.contains("\"My Tag\""), "{sql}");
        assert!(sql.contains("GROUP BY"), "{sql}");
        assert!(sql.contains("\"count(*)\""), "{sql}");
    }

    /// Columns the wide OR chain covers: more than the server's parser nests.
    const WIDE_OR_COLUMNS: usize = 64;
    /// The parser's recursion limit on the server, DataFusion's default
    /// (`sql_parser.recursion_limit`).
    const SERVER_RECURSION_LIMIT: usize = 50;
    /// The stack the wide-expression test runs on: building and printing an
    /// expression this deep recurses once per level, which an unoptimized
    /// build's frames overflow on a test thread's default stack.
    const WIDE_EXPRESSION_STACK_BYTES: usize = 16 * 1024 * 1024;
    /// The deepest the statement's parentheses may nest: the FILTER's own,
    /// the unparser's around the predicate, and the AND group inside it.
    const MAX_PARENTHESIS_DEPTH: i32 = 3;
    /// A table whose name needs quoting.
    const WIDE_TABLE: &str = "wide-docs";

    /// A chain of ORs over many columns reaches the server flat, so its parser
    /// takes it at its default recursion limit; printed with a parenthesis per
    /// OR it was refused (`RecursionLimitExceeded`). Precedence still holds
    /// where it must: an AND under an OR keeps its parentheses.
    #[test]
    fn a_wide_or_reaches_the_server_flat_enough_to_parse() {
        thread::Builder::new()
            .stack_size(WIDE_EXPRESSION_STACK_BYTES)
            .spawn(wide_or_reaches_the_server_flat)
            .expect("a test thread")
            .join()
            .expect("the test thread");
    }

    fn wide_or_reaches_the_server_flat() {
        let names: Vec<String> = (0..WIDE_OR_COLUMNS)
            .map(|i| format!("obj.field_{i}"))
            .collect();
        let schema = Arc::new(Schema::new(
            names
                .iter()
                .map(|n| Field::new(n, DataType::Utf8, true))
                .collect::<Vec<_>>(),
        ));
        let any = names
            .iter()
            .map(|n| col(format!("\"{n}\"")).is_not_null())
            .reduce(|a, b| a.or(b))
            .expect("columns");
        let grouped = col("\"obj.field_0\"")
            .is_not_null()
            .and(col("\"obj.field_1\"").is_null())
            .or(col("\"obj.field_2\"").is_null());
        let statements =
            count_where_statements(WIDE_TABLE, &schema, &[any, grouped]).expect("statements");
        let sql = &statements[0].0;
        let deepest = sql
            .chars()
            .scan(0i32, |depth, c| {
                *depth += match c {
                    '(' => 1,
                    ')' => -1,
                    _ => 0,
                };
                Some(*depth)
            })
            .max()
            .unwrap_or(0);
        assert!(
            deepest <= MAX_PARENTHESIS_DEPTH,
            "nested {deepest} deep: {sql}"
        );
        let column = |i: usize| format!(r#""{WIDE_TABLE}"."obj.field_{i}""#);
        let and_kept = format!("({} IS NOT NULL AND {} IS NULL) OR", column(0), column(1));
        assert!(sql.contains(&and_kept), "{sql}");
        DFParserBuilder::new(sql.as_str())
            .with_recursion_limit(SERVER_RECURSION_LIMIT)
            .build()
            .and_then(|mut parser| parser.parse_statements())
            .unwrap_or_else(|e| panic!("the server's parser refuses it: {e}\n{sql}"));
    }

    /// A hosted `count_where` sends every predicate in one aggregate of
    /// `COUNT(*) FILTER`s, unless one holds a subquery, which a FILTER clause
    /// cannot: then each predicate is counted under its own `WHERE`, as a
    /// local table counts it.
    #[test]
    fn count_where_over_remote_counts_a_subquery_under_its_own_where() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let tags = LogicalPlanBuilder::scan(
            TableReference::bare("tags"),
            provider_as_source(Arc::new(EmptyTable::new(Arc::clone(&schema)))),
            None,
        )
        .and_then(|scan| scan.project(vec![col("n")]))
        .and_then(|projected| projected.build())
        .expect("subquery plan");

        let plain = [col("n").gt(lit(1i64)), col("n").lt(lit(9i64))];
        let statements = count_where_statements("docs", &schema, &plain).expect("statements");
        assert_eq!(statements.len(), 1, "{statements:?}");
        assert_eq!(statements[0].1, plain.len());
        assert!(statements[0].0.contains("FILTER"), "{statements:?}");

        let with_subquery = [
            col("n").gt(lit(1i64)),
            in_subquery(col("n"), Arc::new(tags)),
        ];
        let statements =
            count_where_statements("docs", &schema, &with_subquery).expect("statements");
        let counts: Vec<usize> = statements.iter().map(|(_, n)| *n).collect();
        assert_eq!(counts, [1, 1], "{statements:?}");
        let sql = &statements[1].0;
        let from = sql.find(" FROM ").expect("a FROM clause");
        assert!(
            !sql[..from].contains("IN (SELECT"),
            "no subquery in the counts: {sql}"
        );
        assert!(
            sql[from..].contains("IN (SELECT"),
            "the subquery is the WHERE: {sql}"
        );
    }

    /// Hydrate over the remote transport is refused as unsupported, before any
    /// request is sent.
    #[test]
    fn hydrate_is_unsupported_over_remote() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let catalog =
            RemoteCatalog::new("http://127.0.0.1:1".into(), "db".into(), Some("key".into()))
                .expect("catalog");
        let table = RemoteTable::new(Arc::new(catalog), "t".into(), Arc::clone(&schema));
        let mut empty =
            RecordBatchIterator::new(Vec::<Result<RecordBatch, ArrowError>>::new(), schema);
        let err = table
            .hydrate(&mut empty, 1_000)
            .expect_err("remote hydrate must be refused");
        assert!(matches!(err, InfinoError::Unsupported(_)), "got {err:?}");
    }
}
