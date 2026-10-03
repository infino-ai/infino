// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A caller's order on a ranked search table function's rows.
//!
//! `bm25_search` and `hybrid_search` hand a statement their top `k` in the
//! engine's order. A host may know a better order for the rows than the
//! engine's score — the platform's grader scores each row against the
//! query text with a model, which ranks the rows an answer went on to use
//! above the ones it did not — and it wants that order inside a statement
//! as well as on its own search routes, with the table the statement sees
//! keeping its shape: `k` rows, the same columns, so a `JOIN` or `GROUP BY`
//! over the function reads as it did and only which `k` rows changed.
//!
//! A [`SearchReranker`] set on a connection ([`Connection::set_search_reranker`])
//! does that. When one is set, the two functions ask the engine for the
//! reranker's pool rather than `k` ([`SearchReranker::pool`]), hand the pool
//! to the reranker with the table's text columns among the rows whatever the
//! statement projected (the reranker reads text; the projection is applied
//! afterwards), and emit the `k` rows it hands back. The engine runs no
//! model and knows nothing of what the reranker does; a connection without
//! one serves the functions as before. `vector_search` by a vector,
//! `bm25_search_prefix` and the unranked matches carry no query text a
//! reranker could read against, and are never reranked.
//!
//! [`Connection::set_search_reranker`]: super::Connection::set_search_reranker

use std::{fmt, sync::Arc};

use arrow::compute::concat_batches;
use arrow_array::RecordBatch;
use arrow_schema::{DataType, SchemaRef};
use async_trait::async_trait;
use datafusion::{
    catalog::{Session, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    logical_expr::{Expr, TableProviderFilterPushDown, TableType},
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream, collect,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{future::BoxFuture, stream};

/// An order a host puts on a ranked search's rows before a statement sees
/// them. Set on a connection with `Connection::set_search_reranker`.
pub trait SearchReranker: Send + Sync + fmt::Debug {
    /// Rows the search is asked for when its rows are going to be
    /// reordered, for a statement asking for `k`: the pool the order is
    /// chosen from. Never fewer than `k` are fetched whatever this says.
    fn pool(&self, k: usize) -> usize;

    /// `rows` — the pool, in the engine's order, carrying every text column
    /// of `table` whatever the statement projected — in the order the
    /// statement's `k` are taken from, cut to `k`. The schema comes back as
    /// it went in. `Err` is the failure in a sentence; the statement fails
    /// with it.
    fn rerank<'a>(
        &'a self,
        table: &'a str,
        query_text: &'a str,
        rows: RecordBatch,
        k: usize,
    ) -> BoxFuture<'a, Result<RecordBatch, String>>;
}

/// A ranked search function's provider with the reranker's order on its
/// rows: the inner function was called for the reranker's pool, and this
/// hands a statement the `k` rows the reranker chose from it.
pub(crate) struct RerankedSearch {
    pub(crate) inner: Arc<dyn TableProvider>,
    pub(crate) reranker: Arc<dyn SearchReranker>,
    /// The table the function searched, for the reranker.
    pub(crate) table: String,
    /// The text the function ranked by, for the reranker to read the rows
    /// against.
    pub(crate) query_text: String,
    /// The statement's `k`.
    pub(crate) k: usize,
}

impl fmt::Debug for RerankedSearch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RerankedSearch")
            .field("table", &self.table)
            .field("k", &self.k)
            .finish_non_exhaustive()
    }
}

/// Whether a column is text the reranker reads.
fn is_text(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

#[async_trait]
impl TableProvider for RerankedSearch {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    /// The inner function's filters stand: it reports them as it would
    /// have, and DataFusion keeps its `FilterExec` above this provider's
    /// plan.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }

    /// The inner function is scanned for the statement's columns plus every
    /// text column of the table, so the reranker has text to read whatever
    /// the statement projected; the reranked rows are cut back to the
    /// statement's columns on the way out.
    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let schema = self.inner.schema();
        let requested: Vec<usize> = match projection {
            Some(indices) => indices.clone(),
            None => (0..schema.fields().len()).collect(),
        };
        let mut read = requested.clone();
        for (index, field) in schema.fields().iter().enumerate() {
            if is_text(field.data_type()) && !read.contains(&index) {
                read.push(index);
            }
        }
        let inner = self.inner.scan(state, Some(&read), filters, limit).await?;
        // The requested columns are the first of the read ones, in order.
        let keep: Vec<usize> = (0..requested.len()).collect();
        let exec = RerankExec::try_new(
            inner,
            Arc::clone(&self.reranker),
            self.table.clone(),
            self.query_text.clone(),
            self.k,
            keep,
        )?;
        Ok(Arc::new(exec))
    }
}

/// The plan over a reranked search: runs the inner function's plan to the
/// end, hands its rows to the reranker, and emits the `k` it chose with the
/// statement's columns.
struct RerankExec {
    inner: Arc<dyn ExecutionPlan>,
    reranker: Arc<dyn SearchReranker>,
    table: String,
    query_text: String,
    k: usize,
    /// The statement's columns, as positions in the inner plan's schema.
    keep: Vec<usize>,
    schema: SchemaRef,
    cache: Arc<PlanProperties>,
}

impl RerankExec {
    fn try_new(
        inner: Arc<dyn ExecutionPlan>,
        reranker: Arc<dyn SearchReranker>,
        table: String,
        query_text: String,
        k: usize,
        keep: Vec<usize>,
    ) -> DfResult<Self> {
        let schema = Arc::new(
            inner
                .schema()
                .project(&keep)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?,
        );
        let cache = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            inner,
            reranker,
            table,
            query_text,
            k,
            keep,
            schema,
            cache,
        })
    }

    fn describe(&self) -> String {
        format!("RerankExec: table={}, k={}", self.table, self.k)
    }
}

impl fmt::Debug for RerankExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for RerankExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for RerankExec {
    fn name(&self) -> &'static str {
        "RerankExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.inner]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let inner = children.into_iter().next().ok_or_else(|| {
            DataFusionError::Internal("RerankExec needs its one child".to_string())
        })?;
        Ok(Arc::new(Self::try_new(
            inner,
            Arc::clone(&self.reranker),
            self.table.clone(),
            self.query_text.clone(),
            self.k,
            self.keep.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "RerankExec has a single partition; asked for {partition}"
            )));
        }
        let inner = Arc::clone(&self.inner);
        let reranker = Arc::clone(&self.reranker);
        let table = self.table.clone();
        let query_text = self.query_text.clone();
        let k = self.k;
        let keep = self.keep.clone();
        let out_schema = Arc::clone(&self.schema);
        let fut = async move {
            let read_schema = inner.schema();
            let batches = collect(inner, context).await?;
            let whole = concat_batches(&read_schema, &batches)
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            // One row or none has no order to choose; the reranker is not
            // asked.
            let ordered = if whole.num_rows() <= 1 {
                whole.slice(0, whole.num_rows().min(k))
            } else {
                let ordered = reranker
                    .rerank(&table, &query_text, whole, k)
                    .await
                    .map_err(DataFusionError::Execution)?;
                if ordered.schema() != read_schema {
                    return Err(DataFusionError::Execution(
                        "the search reranker changed the rows' columns".to_string(),
                    ));
                }
                ordered.slice(0, ordered.num_rows().min(k))
            };
            ordered
                .project(&keep)
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            out_schema,
            stream::once(fut),
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use arrow_array::{Int64Array, LargeStringArray, StringArray};
    use arrow_schema::{Field, Schema};
    use futures::FutureExt;

    use super::*;
    use crate::{IndexSpec, connect};

    /// The pool the test reranker names, and the `k` the statements ask for.
    const TEST_POOL: usize = 1_000;
    const TEST_K: usize = 2;

    /// What one call of the test reranker saw.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Seen {
        table: String,
        query_text: String,
        columns: Vec<String>,
        rows: usize,
        k: usize,
    }

    /// A reranker that puts the engine's rows in reverse and remembers what
    /// it was handed.
    #[derive(Debug, Default)]
    struct Reversing {
        seen: Mutex<Vec<Seen>>,
    }

    impl SearchReranker for Reversing {
        fn pool(&self, k: usize) -> usize {
            k.max(TEST_POOL)
        }

        fn rerank<'a>(
            &'a self,
            table: &'a str,
            query_text: &'a str,
            rows: RecordBatch,
            k: usize,
        ) -> BoxFuture<'a, Result<RecordBatch, String>> {
            async move {
                self.seen.lock().expect("seen").push(Seen {
                    table: table.to_string(),
                    query_text: query_text.to_string(),
                    columns: rows
                        .schema()
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect(),
                    rows: rows.num_rows(),
                    k,
                });
                let n = rows.num_rows();
                let indices: Vec<u32> = (0..n as u32).rev().collect();
                arrow::compute::take_record_batch(&rows, &arrow_array::UInt32Array::from(indices))
                    .map_err(|e| e.to_string())
            }
            .boxed()
        }
    }

    /// Three titles, every one holding `rust`, ids 1..=3.
    fn connection_with_docs() -> crate::Connection {
        let db = connect("memory://").expect("connect");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let table = db
            .create_table("docs", Arc::clone(&schema), IndexSpec::new().fts("title"))
            .expect("create");
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(LargeStringArray::from(vec![
                    "rust async runtime",
                    "rust systems programming",
                    "rust on the web",
                ])),
            ],
        )
        .expect("batch");
        table.append(&batch).expect("append");
        db
    }

    fn ids(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                b.column_by_name("id")
                    .expect("id")
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("ids")
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// With a reranker set the statement gets the reranker's `k`, chosen
    /// from the pool the engine was asked for, with the table's text among
    /// what the reranker read even when the statement projected it away;
    /// the statement's columns alone come out. Without one, or when a
    /// function carries no query text, the engine's order stands.
    #[test]
    fn a_statement_takes_the_rerankers_k_from_the_pool() {
        let db = connection_with_docs();
        let engine_order = ids(&db
            .query_sql(&format!(
                "SELECT id FROM bm25_search('docs', 'title', 'rust', {TEST_K})"
            ))
            .expect("engine order"));
        assert_eq!(engine_order.len(), TEST_K);
        // Every match in the engine's order, read before the reranker is
        // set: what the reversal is measured against.
        let all_engine = ids(&db
            .query_sql("SELECT id FROM bm25_search('docs', 'title', 'rust', 10)")
            .expect("all"));
        assert_eq!(all_engine.len(), 3, "three matches");

        let reranker = Arc::new(Reversing::default());
        db.set_search_reranker(Some(reranker.clone()));
        let batches = db
            .query_sql(&format!(
                "SELECT id FROM bm25_search('docs', 'title', 'rust', {TEST_K})"
            ))
            .expect("reranked");
        let reranked = ids(&batches);
        assert_eq!(reranked.len(), TEST_K, "the statement's k, not the pool");
        assert_eq!(
            batches[0].schema().fields().len(),
            1,
            "the statement's columns alone come out"
        );
        let seen = reranker.seen.lock().expect("seen").clone();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].table, "docs");
        assert_eq!(seen[0].query_text, "rust");
        assert_eq!(seen[0].k, TEST_K);
        assert_eq!(seen[0].rows, 3, "every match: the pool, not k");
        assert!(
            seen[0].columns.iter().any(|c| c == "title"),
            "the text column is read for the reranker: {:?}",
            seen[0].columns
        );
        // Reversed: the engine's last match comes first, then its second.
        assert_eq!(reranked, vec![all_engine[2], all_engine[1]]);

        // The unranked and text-less functions are left alone.
        let before = reranker.seen.lock().expect("seen").len();
        db.query_sql("SELECT id FROM token_match('docs', 'title', 'rust', 'or')")
            .expect("token_match runs");
        assert_eq!(reranker.seen.lock().expect("seen").len(), before);

        db.set_search_reranker(None);
        let back = ids(&db
            .query_sql(&format!(
                "SELECT id FROM bm25_search('docs', 'title', 'rust', {TEST_K})"
            ))
            .expect("engine order again"));
        assert_eq!(back, engine_order);
    }

    /// A reranker that hands back other columns fails the statement rather
    /// than feeding a plan rows of another shape.
    #[test]
    fn a_reranker_that_changes_the_columns_fails_the_statement() {
        #[derive(Debug)]
        struct Reshaping;
        impl SearchReranker for Reshaping {
            fn pool(&self, k: usize) -> usize {
                k
            }
            fn rerank<'a>(
                &'a self,
                _table: &'a str,
                _query_text: &'a str,
                rows: RecordBatch,
                _k: usize,
            ) -> BoxFuture<'a, Result<RecordBatch, String>> {
                async move {
                    let schema = Arc::new(Schema::new(vec![Field::new(
                        "other",
                        DataType::Utf8,
                        false,
                    )]));
                    RecordBatch::try_new(
                        schema,
                        vec![Arc::new(StringArray::from(vec!["x"; rows.num_rows()]))],
                    )
                    .map_err(|e| e.to_string())
                }
                .boxed()
            }
        }
        let db = connection_with_docs();
        db.set_search_reranker(Some(Arc::new(Reshaping)));
        let err = db
            .query_sql("SELECT id FROM bm25_search('docs', 'title', 'rust', 2)")
            .expect_err("the shape changed");
        assert!(err.to_string().contains("changed the rows' columns"), "{err}");
    }
}
