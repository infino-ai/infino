// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hybrid retrieval (BM25 × vector) as a DataFusion table-valued
//! function.
//!
//! `hybrid_search(text_col, q_text, vec_col, q_vec, k)` registers via
//! `register_udtf` and lowers to [`HybridSearchExec`], a custom
//! `ExecutionPlan` that runs the existing BM25 and vector kernels
//! ([`SupertableReader::bm25_search`](crate::supertable::handle::SupertableReader::bm25_search)
//! / [`vector_search`](crate::supertable::handle::SupertableReader::vector_search))
//! *concurrently* inside `execute()`, fuses the two rankings with
//! **reciprocal-rank fusion** (RRF), and resolves the fused hits once
//! to the supertable's `_id` + projected scalar columns + `score`
//! through the shared [`resolve_hits`](super::common::resolve_hits).
//!
//! No new kernel — pure composition of the two existing Exec nodes'
//! kernels feeding a fusion operator.
//!
//! ## Query shape
//!
//! ```sql
//! SELECT _id, score
//! FROM hybrid_search('body', 'error timeout', 'embedding', '0.1,0.2, ... ,0.9', 10)
//! ORDER BY score DESC
//! ```
//!
//! `text_col` / `q_text` feed BM25 (OR mode); `vec_col` / `q_vec` feed
//! vector kNN. `q_vec` is parsed exactly like `vector_search` — a
//! comma-separated string or a SQL array literal — since the vector
//! column is stripped from the SQL schema at commit and lives in the
//! embedded blob, so it can never be a scanned column.
//!
//! ## Fusion + score direction
//!
//! Each retriever returns its hits best-first (BM25 descending
//! relevance; vector ascending distance). RRF ignores the raw scores
//! and fuses on **rank only**: a hit at 0-based position `r` in a list
//! contributes `1 / (RRF_K + r + 1)`, and the per-identity
//! contributions from the two lists are summed. The emitted `score`
//! column is the fused RRF score — **higher is better**, so `ORDER BY
//! score DESC` lists the best blended matches first. Identity is the
//! stable `_id`, so a document surfaced by *both* retrievers is boosted
//! above one surfaced by a single retriever.
//!
//! ## `WHERE` on the function
//!
//! A `WHERE` over the function's output means *the top k among rows
//! satisfying the predicate*. Filters are reported `Inexact`, lowered
//! once to a [`CandidateScope`] and handed to **both** retrievers, so the
//! BM25 leg and the vector leg rank within the same superfiles and rows
//! before fusion. What the index could not bound is made up by widening:
//! [`fill_top_k_within`] applies the exact predicate to the fused rows and
//! asks both legs for more until `k` pass or a ceiling stops it. When the
//! widening ends short at a ceiling below the table's size, the function
//! checks the predicate over the scope's rows itself ([`exact_scope`]) and
//! runs both legs once more within exactly the rows it keeps, as
//! `vector_search` and `bm25_search` do. Without a `WHERE`, the vector leg
//! is `vector_search`'s own search, the hidden vector index.

use std::{cmp::Ordering, collections::HashMap, fmt, sync::Arc};

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
use futures::{
    future::{self, BoxFuture},
    stream,
};
use tracing::{Instrument, debug};

#[cfg(feature = "detailed-tracing")]
use crate::utils::trace::OpOrigin;
use crate::{
    InfinoError,
    runtime_metrics::op_stats,
    superfile::{
        fts::reader::{Bm25SearchOptions, BoolMode},
        reader::VectorSearchOptions,
    },
    supertable::{
        QueryError,
        handle::{Supertable, SupertableReader, WeakReader},
        query::{
            SuperfileHit,
            candidate::CandidateScope,
            dispatch::attach_stable_ids_to_hits,
            exec::{
                common::{
                    Arity, PushedPredicate, arg_to_string, arg_to_usize,
                    candidate_plan_for_filters, exact_scope, fill_ceiling, fill_top_k_within,
                    live_reader, output_schema_with_score, resolve_hits_named, scope_to_call,
                    top_k_statistics, traced_tvf, widening_ended_short,
                },
                vector_exec::arg_to_query_vector,
            },
            vector::{
                calibrated_query, free_columns_unambiguous, hits_id_score_batch,
                id_score_projection_indices, sql_id_score_projection,
                user_placement_for_scalar_resolve,
            },
        },
    },
    utils::trace::{self, detail_span},
};

/// SQL name the TVF is registered under.
pub(crate) const HYBRID_SEARCH_UDTF: &str = "hybrid_search";

/// Arguments of `hybrid_search(text_col, q_text, vec_col, q_vec, k)`.
pub(crate) const HYBRID_SEARCH_ARITY: Arity = Arity {
    min: 5,
    max: 5,
    arguments: "text_col, q_text, vec_col, q_vec, k",
};

/// RRF rank-bias constant. `60` is the value from the original
/// Cormack et al. reciprocal-rank-fusion paper and the de-facto
/// default across engines (Elasticsearch, Weaviate, …). Larger values
/// flatten the rank weighting; it never depends on the raw scores.
const RRF_K: f32 = 60.0;

/// The rows a hybrid search ranks within. The BM25 leg ranks within these
/// rows whatever the variant; the variant picks which search the vector leg
/// runs, because each fits a different shape of scope.
#[derive(Clone, Copy)]
enum HybridScope<'a> {
    /// Every row: no `WHERE`. The vector leg is `vector_search`'s own
    /// search, the hidden vector index plus the rows not drained into it
    /// yet, so its cost follows `k`, not the table's file count.
    Table,
    /// What a `WHERE` admits: the files the statistics keep and, when an
    /// index bounds the rows, those rows. The vector leg searches those
    /// user files directly, limited to the rows when they are bounded; it
    /// costs a probe per file, which is why it is only the widening's scope.
    Admitted(&'a CandidateScope),
    /// Exactly the rows a `WHERE` keeps ([`exact_scope`]). The vector leg is
    /// `vector_search`'s filtered search over them: drained rows through the
    /// hidden index, the rest through their user files, each probing a
    /// bounded number of clusters.
    Exact(&'a CandidateScope),
}

impl<'a> HybridScope<'a> {
    /// The scope the BM25 leg ranks within; `None` is the whole table.
    fn candidates(self) -> Option<&'a CandidateScope> {
        match self {
            HybridScope::Table => None,
            HybridScope::Admitted(scope) | HybridScope::Exact(scope) => Some(scope),
        }
    }
}

/// Register `hybrid_search(text_col, q_text, vec_col, q_vec, k)` on
/// `ctx`, bound to the query's pinned `reader` + scalar `schema`.
/// Called from
/// [`Supertable::query_sql`](crate::supertable::handle::Supertable::query_sql).
pub(crate) fn register_hybrid_search(
    ctx: &SessionContext,
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
) {
    ctx.register_udtf(
        HYBRID_SEARCH_UDTF,
        Arc::new(HybridSearchFunc::new(reader, scalar_schema)),
    );
}

impl SupertableReader {
    /// Async kernel: run BM25 (`mode`) and vector kNN concurrently over
    /// this reader's pinned snapshot and fuse the two rankings with
    /// reciprocal-rank fusion, returning the top-`k` fused hits (each
    /// carrying its RRF score; higher is better). The hits are
    /// unresolved — the caller resolves `_id` / scalar columns.
    /// `pub(crate)` kernel; the public surface is the sync
    /// [`SupertableReader::hybrid_search`].
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(skip_all, fields(text_col = text_col, vec_col = vec_col, k = k, mode = ?mode, role = self.role().as_str(), origin = OpOrigin::Query.as_str()))
    )]
    pub(crate) async fn hybrid_search_async(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.hybrid_search_scoped_async(
            text_col,
            q_text,
            mode,
            vec_col,
            q_vec,
            options,
            k,
            HybridScope::Table,
        )
        .await
    }

    /// [`Self::hybrid_search_async`] confined to the rows `scope` names:
    /// what a SQL `WHERE` pushed into the `hybrid_search` table function
    /// admits, or exactly the rows it keeps. Both legs rank within the same
    /// rows, so the fused top-k is the best `k` among them. Fusion joins the
    /// legs on the stable `_id`, which every hit of both legs carries, so
    /// where a hit was found does not change its identity. See
    /// [`HybridScope`] for which search the vector leg runs.
    #[allow(clippy::too_many_arguments)]
    async fn hybrid_search_scoped_async(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
        scope: HybridScope<'_>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        // The vector searches below take the query calibrated for the
        // column's metric (normalized for cosine), as the public vector path
        // does. `vector_hits_within_scope` calibrates for itself; the other
        // two are calibrated here.
        let vector_leg = async {
            let mut hits = match scope {
                HybridScope::Table => {
                    let q_vec = calibrated_query(self, vec_col, q_vec);
                    self.vector_search_global_index_async(vec_col, &q_vec, k, options)
                        .await?
                }
                HybridScope::Admitted(scope) => {
                    // Only superfiles that still hold a candidate row; the
                    // kernel treats a superfile absent from the allow map as
                    // having no admitted row, so none is handed to it.
                    let superfiles: Vec<_> = scope.admitted_superfiles().cloned().collect();
                    if superfiles.is_empty() {
                        return Ok(Vec::new());
                    }
                    let q_vec = calibrated_query(self, vec_col, q_vec);
                    self.vector_fanout_over_superfiles(
                        superfiles,
                        vec_col,
                        &q_vec,
                        k,
                        options,
                        scope.allow.clone(),
                        None,
                    )
                    .await?
                }
                HybridScope::Exact(scope) => {
                    self.vector_hits_within_scope(vec_col, q_vec, k, options, scope)
                        .await?
                }
            };
            // Fusion joins the legs on `_id`, so every vector hit needs one.
            // Hidden-index hits, and hits in cell-packed user files (from
            // their inline id region), carry it from the search. A hit in an
            // older plain-IVF user file whose ids are not one contiguous run
            // can arrive without one (a search over user files, or a table's
            // search before its first drain); its local is its Parquet row,
            // so the `_id` read there is its own. Only those hits are read.
            attach_stable_ids_to_hits(self, &mut hits).await?;
            Ok::<_, QueryError>(hits)
        };
        // Both retrievers run concurrently on the query runtime; each
        // inherits its own manifest skip and returns hits best-first. Each
        // leg has its span, so a trace shows which one the query waited on.
        let phases = self.phase_spans();
        let bm25_leg = self
            .bm25_search_scoped_async(
                text_col,
                q_text,
                k,
                Bm25SearchOptions::new().with_mode(mode),
                scope.candidates(),
            )
            .instrument(trace::phase(phases, || detail_span!("hybrid.bm25")));
        let vector_leg =
            vector_leg.instrument(trace::phase(phases, || detail_span!("hybrid.vector")));
        let (bm25_res, vector_res) = future::join(bm25_leg, vector_leg).await;
        let (bm25_hits, vector_hits) = (bm25_res?, vector_res?);
        let fuse_span = trace::phase(phases, || {
            detail_span!(
                "hybrid.fuse",
                bm25_hits = bm25_hits.len(),
                vector_hits = vector_hits.len(),
                rows_out = tracing::field::Empty,
            )
        });
        // The fusion math is a real (small) kernel section; bracket it so
        // hybrid's kernel CPU covers all three of its compute legs.
        let fused = fuse_span.in_scope(|| {
            op_stats::timed_kernel(&self.op_stats, || rrf_fuse(&bm25_hits, &vector_hits, k))
        })?;
        fuse_span.record("rows_out", fused.len());
        Ok(fused)
    }

    /// Hybrid BM25 + vector search fused with reciprocal-rank fusion
    /// over this reader's pinned snapshot. Returns up to `k` fused hits
    /// carrying the RRF score (higher is better). Sync wrapper over
    /// [`SupertableReader::hybrid_search_async`] — the same fusion the
    /// `hybrid_search` SQL table function runs.
    #[allow(clippy::too_many_arguments)]
    pub fn hybrid_search(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.block_on(self.hybrid_search_async(text_col, q_text, mode, vec_col, q_vec, options, k))
    }
}

impl Supertable {
    /// Hybrid BM25 + vector search over the current snapshot, fusing the
    /// two rankings with reciprocal-rank fusion and returning Arrow rows
    /// best-score-first (RRF score, higher is better).
    ///
    /// `text_col` / `q_text` (under `mode`) drive BM25; `vec_col` /
    /// `q_vec` (with `options`, e.g. `nprobe`) drive vector kNN. `k`
    /// bounds each retriever and the fused result. `projection` follows
    /// the same rules as [`Supertable::bm25_search`].
    #[allow(clippy::too_many_arguments)]
    #[cfg_attr(
        feature = "detailed-tracing",
        // The empty fields are the read's outcome, filled once it has run;
        // see `CloseOut`.
        tracing::instrument(skip_all, fields(
            text_col = text_col,
            vec_col = vec_col,
            k = k,
            mode = ?mode,
            role = self.role().as_str(),
            origin = OpOrigin::Query.as_str(),
            rows_out = tracing::field::Empty,
            kernel_cpu_ns = tracing::field::Empty,
            planned_read_ranges = tracing::field::Empty,
            fts_postings_bytes = tracing::field::Empty,
            vector_cells_scanned = tracing::field::Empty,
            vector_candidates_scanned = tracing::field::Empty,
            vector_rows_reranked = tracing::field::Empty,
            rows_materialized = tracing::field::Empty,
            store_heads = tracing::field::Empty,
            store_gets = tracing::field::Empty,
            store_get_bytes = tracing::field::Empty,
            store_bg_gets = tracing::field::Empty,
            store_bg_get_bytes = tracing::field::Empty,
        ))
    )]
    pub fn hybrid_search(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        debug!(text_col, vec_col, k, "hybrid_search");
        let close_out = self.close_out();
        let reader = self.reader()?;
        reader
            .check_projection(projection)
            .map_err(|e| InfinoError::from(e).with_context("hybrid_search", None))?;
        let hits = reader
            .hybrid_search(text_col, q_text, mode, vec_col, q_vec, options, k)
            .map_err(|e| InfinoError::from(e).with_context("hybrid_search", None))?;
        let batch = self
            .block_on_query(
                async {
                    // `_id` + `score` come free with the search wave, so a
                    // projection of just those needs no placement and no
                    // Parquet decode: the same fast path `vector_search`
                    // takes. Fusion refuses an unstamped hit, so every
                    // fused hit carries its `_id`.
                    let id_column = reader.options().id_column.as_str();
                    if free_columns_unambiguous(&reader.options().schema, id_column)
                        && let Some(indices) = id_score_projection_indices(projection, id_column)
                    {
                        return hits_id_score_batch(&reader, &hits)?
                            .project(&indices)
                            .map_err(|e| QueryError::Internal(e.to_string()));
                    }
                    // Boundary-replica stubs carry an IVF local that does not
                    // address a Parquet row; remap to the owning placement by
                    // stable id before the scalar decode, exactly as the
                    // hybrid TVF and `vector_search` paths do.
                    let hits = user_placement_for_scalar_resolve(&reader, &hits).await?;
                    resolve_hits_named(&reader, &hits, projection).await
                }
                .instrument(detail_span!("search.resolve", hits = hits.len())),
            )
            .map_err(|e| InfinoError::from(e).with_context("hybrid_search", None))?;
        close_out.finish(batch.num_rows() as u64);
        Ok(vec![batch])
    }
}

/// `TableFunctionImpl` for `hybrid_search`. Holds the query's pinned
/// snapshot; `call` parses the SQL arguments and hands back a
/// per-invocation [`HybridSearchTable`].
#[derive(Debug)]
pub(crate) struct HybridSearchFunc {
    reader: WeakReader,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl HybridSearchFunc {
    pub(crate) fn new(reader: Arc<SupertableReader>, scalar_schema: SchemaRef) -> Self {
        let output_schema = output_schema_with_score(&scalar_schema);
        Self {
            reader: WeakReader::from_reader(&reader),
            scalar_schema,
            output_schema,
        }
    }
}

impl TableFunctionImpl for HybridSearchFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let args = args.exprs();
        HYBRID_SEARCH_ARITY.check(HYBRID_SEARCH_UDTF, args.len())?;
        let text_col = arg_to_string(&args[0], "hybrid_search text_col")?;
        let q_text = arg_to_string(&args[1], "hybrid_search q_text")?;
        let vec_col = arg_to_string(&args[2], "hybrid_search vec_col")?;
        let q_vec = arg_to_query_vector(&args[3])?;
        let k = arg_to_usize(&args[4], "hybrid_search k")?;
        let reader = live_reader(&self.reader, "hybrid_search")?;
        scope_to_call(
            HYBRID_SEARCH_UDTF,
            Arc::new(HybridSearchTable {
                reader,
                text_col,
                q_text,
                mode: BoolMode::Or,
                vec_col,
                q_vec,
                options: VectorSearchOptions::new(),
                k,
                scalar_schema: Arc::clone(&self.scalar_schema),
                output_schema: Arc::clone(&self.output_schema),
            }),
        )
    }
}

/// One parsed `hybrid_search(...)` invocation as a `TableProvider`.
/// `scan` lowers to [`HybridSearchExec`]; the TVF's `k` is the top-k
/// bound for *each* retriever and the final fused result.
struct HybridSearchTable {
    reader: Arc<SupertableReader>,
    text_col: String,
    q_text: String,
    mode: BoolMode,
    vec_col: String,
    q_vec: Vec<f32>,
    options: VectorSearchOptions,
    k: usize,
    scalar_schema: SchemaRef,
    output_schema: SchemaRef,
}

impl fmt::Debug for HybridSearchTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HybridSearchTable")
            .field("text_col", &self.text_col)
            .field("vec_col", &self.vec_col)
            .field("k", &self.k)
            .field("dim", &self.q_vec.len())
            .finish()
    }
}

#[async_trait]
impl TableProvider for HybridSearchTable {
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
        let exec = HybridSearchExec::try_new(
            Arc::clone(&self.reader),
            self.text_col.clone(),
            self.q_text.clone(),
            self.mode,
            self.vec_col.clone(),
            self.q_vec.clone(),
            self.options,
            self.k,
            Arc::clone(&self.scalar_schema),
            Arc::clone(&self.output_schema),
            projection.cloned(),
            filters.to_vec(),
        )?;
        Ok(Arc::new(exec))
    }

    /// Report every `WHERE` filter as `Inexact`: DataFusion hands the
    /// predicates to [`scan`](Self::scan), where they scope both
    /// retrievers to the rows that can satisfy them, **and** keeps a
    /// `FilterExec` above the scan that re-applies the exact predicate.
    /// Correctness never depends on the pushdown. What the scope could not
    /// bound, the exec widens for and, when the widening ends short at its
    /// ceiling, checks on the scope's rows itself (`exact_scope`) before
    /// running both legs within exactly the rows the predicate keeps. A
    /// predicate it cannot evaluate before ranking (one on `score`) keeps
    /// what the widening found.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        Ok(vec![TableProviderFilterPushDown::Inexact; filters.len()])
    }
}

/// Custom `ExecutionPlan` that runs the BM25 + vector kernels
/// concurrently on the query runtime inside `execute()`, fuses the
/// two rankings with RRF, and emits the resolved `_id` + scalar
/// columns + fused `score`.
struct HybridSearchExec {
    reader: Arc<SupertableReader>,
    text_col: String,
    q_text: String,
    mode: BoolMode,
    vec_col: String,
    q_vec: Vec<f32>,
    options: VectorSearchOptions,
    k: usize,
    /// `WHERE` predicates DataFusion pushed into this scan (reported
    /// `Inexact`); scope both retrievers and fill the fused `k` in
    /// `execute`.
    filters: Vec<Expr>,
    /// Scalar schema, used as the resolve projection.
    scalar_schema: SchemaRef,
    /// Full (pre-projection) output schema: scalar columns + score.
    output_schema: SchemaRef,
    /// Optional projection into `output_schema`.
    projection: Option<Vec<usize>>,
    /// Output schema after `projection`.
    projected_schema: SchemaRef,
    cache: Arc<PlanProperties>,
    /// What the planner is told about this search's size, built once:
    /// DataFusion asks on every optimizer pass. See [`top_k_statistics`].
    statistics: Arc<Statistics>,
}

impl HybridSearchExec {
    #[allow(clippy::too_many_arguments)]
    fn try_new(
        reader: Arc<SupertableReader>,
        text_col: String,
        q_text: String,
        mode: BoolMode,
        vec_col: String,
        q_vec: Vec<f32>,
        options: VectorSearchOptions,
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
            text_col,
            q_text,
            mode,
            vec_col,
            q_vec,
            options,
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
        format!(
            "HybridSearchExec: text_col={}, vec_col={}, k={}, dim={}",
            self.text_col,
            self.vec_col,
            self.k,
            self.q_vec.len()
        )
    }
}

impl fmt::Debug for HybridSearchExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl DisplayAs for HybridSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}

impl ExecutionPlan for HybridSearchExec {
    fn name(&self) -> &'static str {
        "HybridSearchExec"
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
                "HybridSearchExec has a single partition; asked for {partition}"
            )));
        }
        let reader = Arc::clone(&self.reader);
        let text_col = self.text_col.clone();
        let q_text = self.q_text.clone();
        let mode = self.mode;
        let vec_col = self.vec_col.clone();
        let q_vec = self.q_vec.clone();
        let options = self.options;
        let k = self.k;
        let filters = self.filters.clone();
        let scalar_schema = Arc::clone(&self.scalar_schema);
        let output_schema = Arc::clone(&self.output_schema);
        let projection = self.projection.clone();
        let projected_schema = Arc::clone(&self.projected_schema);
        let id_score_projection = sql_id_score_projection(
            &output_schema,
            projection.as_deref(),
            reader.options().id_column.as_str(),
        );

        let fut = async move {
            let predicate = PushedPredicate::compile(&filters, &output_schema);
            let call = FusedCall {
                reader: &reader,
                text_col: &text_col,
                q_text: &q_text,
                mode,
                vec_col: &vec_col,
                q_vec: &q_vec,
                options,
                k,
                predicate: predicate.as_ref(),
                scalar_schema: &scalar_schema,
                output_schema: &output_schema,
                projection: projection.as_deref(),
            };
            // `_id` and `score` come free with the fused hits (fusion
            // refuses a hit without its `_id`), so a projection of just
            // those needs no placement and no decode, as in
            // `vector_search`. Only without a `WHERE`: a predicate needs
            // the rows to check.
            if filters.is_empty()
                && let Some(indices) = id_score_projection
            {
                let fused = call.fused(k, HybridScope::Table).await?;
                return hits_id_score_batch(&reader, &fused)?
                    .project(&indices)
                    .map_err(|e| QueryError::internal(e).into());
            }
            // What the pushed-down `WHERE` admits, resolved once and handed
            // to both retrievers.
            let scope = match filters.is_empty() {
                true => None,
                false => {
                    let plan = candidate_plan_for_filters(reader.manifest(), &filters);
                    Some(reader.candidate_scope(&filters, &plan).await?)
                }
            };
            // Widen first, up to the fill's ceiling: the rounds stop as soon
            // as `k` fused rows pass the predicate, which for a predicate
            // most top-ranked rows satisfy is a round or two.
            let admitted = scope
                .as_ref()
                .map_or(HybridScope::Table, HybridScope::Admitted);
            let widened = call.rows(admitted, None).await?;
            let (Some(scope), Some(predicate)) = (scope.as_ref(), predicate.as_ref()) else {
                return Ok(widened);
            };
            if !widening_ended_short(&reader, k, widened.num_rows()) {
                return Ok(widened);
            }
            // Short at the ceiling: rows the predicate keeps may rank below
            // every hit either leg looked at. Check the predicate on the
            // scope's rows, and run both legs once within exactly the ones
            // it keeps. The vector leg needs every such row, not only the
            // ones the query's terms match, so the check covers the whole
            // scope. A predicate it cannot check before ranking (one on
            // `score`) keeps what the widening found.
            let Some(exact) = exact_scope(&reader, scope, predicate, &output_schema).await? else {
                return Ok(widened);
            };
            let ranked = call.rows(HybridScope::Exact(&exact), Some(k)).await?;
            // The vector leg within the kept rows probes a bounded number of
            // clusters, so it can miss a kept row the widening found by
            // asking for far more hits. Never return fewer rows than the
            // widening already had.
            Ok(match ranked.num_rows() >= widened.num_rows() {
                true => ranked,
                false => widened,
            })
        };

        let span = detail_span!(
            "tvf.hybrid_search",
            k = self.k,
            rows_out = tracing::field::Empty
        );
        let fut = traced_tvf(span, fut);
        let stream = stream::once(fut);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }
}

/// One `hybrid_search` call's fixed inputs, borrowed by each fused round
/// [`HybridSearchExec::execute`] runs.
struct FusedCall<'a> {
    reader: &'a SupertableReader,
    text_col: &'a str,
    q_text: &'a str,
    mode: BoolMode,
    vec_col: &'a str,
    q_vec: &'a [f32],
    options: VectorSearchOptions,
    k: usize,
    predicate: Option<&'a PushedPredicate>,
    scalar_schema: &'a SchemaRef,
    output_schema: &'a SchemaRef,
    projection: Option<&'a [usize]>,
}

impl FusedCall<'_> {
    /// Both retrievers' `want` best hits within `scope`, run concurrently
    /// and fused with RRF.
    async fn fused(
        &self,
        want: usize,
        scope: HybridScope<'_>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        self.reader
            .hybrid_search_scoped_async(
                self.text_col,
                self.q_text,
                self.mode,
                self.vec_col,
                self.q_vec,
                self.options,
                want,
                scope,
            )
            .await
    }

    /// The call's `k` rows within `scope` under the predicate, widening to
    /// at most `ceiling` fused hits a round: `None` is the fill's own
    /// ceiling ([`fill_ceiling`]), `Some(k)` exactly one round.
    ///
    /// Boxed, and built in a plain function rather than an `async` one: the
    /// fill nests both retrievers' futures, and building it here keeps that
    /// state off the stack frame of every caller that polls it.
    fn rows<'b>(
        &'b self,
        scope: HybridScope<'b>,
        ceiling: Option<usize>,
    ) -> BoxFuture<'b, DfResult<RecordBatch>> {
        let ceiling = ceiling.unwrap_or_else(|| fill_ceiling(self.reader, self.k));
        Box::pin(fill_top_k_within(
            self.reader,
            self.k,
            self.predicate,
            self.scalar_schema,
            self.output_schema,
            self.projection,
            ceiling,
            move |want| async move {
                let fused = self.fused(want, scope).await?;
                // Cell-packed user hits can be boundary stubs whose IVF
                // local does not address a Parquet row; remap those to
                // their owning placement by stable id before the scalar
                // decode, the same step the vector TVF takes.
                user_placement_for_scalar_resolve(self.reader, &fused).await
            },
        ))
    }
}

/// Fuse two rank-ordered hit lists into the top-`k` by reciprocal-rank
/// fusion.
///
/// Each list is assumed best-first. A hit at 0-based position `r`
/// contributes `1 / (RRF_K + r + 1)` to its stable `_id`; contributions
/// from the two lists are summed. The stable id is the only identity the
/// two legs share: a BM25 local follows Parquet order, a vector local
/// follows packed-cell order, and a vector hit from the hidden index names
/// a hidden superfile, so an unstamped hit is refused rather than fused as
/// a second document. A document both legs found keeps the BM25 hit, which
/// already addresses its user row. The result is sorted by fused score
/// descending, `_id` as a total tie-break, so the order and which tied rows
/// survive the cut to `k` do not depend on the `HashMap`'s iteration order
/// or on which superfile a leg found a row in. The returned hits carry the
/// fused RRF score (higher is better).
fn rrf_fuse(
    bm25: &[SuperfileHit],
    vector: &[SuperfileHit],
    k: usize,
) -> Result<Vec<SuperfileHit>, QueryError> {
    let mut acc: HashMap<i128, (SuperfileHit, f32)> =
        HashMap::with_capacity(bm25.len() + vector.len());
    for list in [bm25, vector] {
        for (rank, hit) in list.iter().enumerate() {
            let contribution = 1.0 / (RRF_K + rank as f32 + 1.0);
            let identity = hit.stable_id.ok_or_else(|| {
                QueryError::Internal(format!(
                    "hybrid fusion: hit {:?}/{} has no stable _id",
                    hit.superfile, hit.local_doc_id
                ))
            })?;
            // BM25 goes first: `or_insert` keeps the first hit, so a row
            // both legs found keeps the BM25 hit and its user placement.
            acc.entry(identity).or_insert((*hit, 0.0)).1 += contribution;
        }
    }

    let mut fused: Vec<SuperfileHit> = acc
        .into_iter()
        .map(|(_, (mut hit, score))| {
            hit.score = score;
            hit
        })
        .collect();
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.stable_id.cmp(&b.stable_id))
    });
    fused.truncate(k);
    Ok(fused)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, iter::repeat_n};

    use arrow_array::{
        Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array,
        LargeStringArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        physical_plan::{displayable, joins::HashJoinExec},
        prelude::{col, lit},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        config::OptimizeOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            builder::{FtsConfig, VectorConfig},
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            manifest::SuperfileUri,
            query::exec::common::{OVER_FETCH_MIN_CEILING, test_support::sql_row_count},
        },
    };

    // ---- supertable harness: title (FTS) + emb (vector) ----

    fn fixed_list_f32(dim: usize) -> DataType {
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        )
    }

    fn options_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
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
            vec![FtsConfig::new("title")],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Doc `i` gets `titles[i]` and a one-hot vector at dim `i % dim`.
    fn build_batch(titles: &[&str], dim: usize, schema: Arc<Schema>) -> RecordBatch {
        let n = titles.len();
        let title_arr = LargeStringArray::from(titles.to_vec());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)]).expect("batch")
    }

    /// Demo corpus (single superfile). `async` is unique to doc 0;
    /// `rust` is in docs 0 + 4. Doc `i`'s vector is one-hot at dim `i`.
    fn demo(dim: usize) -> Supertable {
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        let titles = [
            "rust async",   // 0
            "python data",  // 1
            "java spring",  // 2
            "go routines",  // 3
            "rust systems", // 4
            "ruby rails",   // 5
            "scala akka",   // 6
            "kotlin flow",  // 7
        ];
        w.append(&build_batch(&titles, dim, schema))
            .expect("append");
        w.commit().expect("commit");
        st
    }

    fn csv_one_hot(dim: usize, active: usize) -> String {
        (0..dim)
            .map(|d| if d == active { "1" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    }

    fn col_str<'a>(batch: &'a RecordBatch, name: &str) -> &'a LargeStringArray {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large utf8 column")
    }

    fn col_f32<'a>(batch: &'a RecordBatch, name: &str) -> &'a Float32Array {
        let idx = batch.schema().index_of(name).expect("column present");
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("f32 column")
    }

    fn id_set(batches: &[RecordBatch]) -> HashSet<i128> {
        let mut out = HashSet::new();
        for b in batches {
            let idx = b.schema().index_of("_id").expect("_id column");
            let a = b
                .column(idx)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal128 _id");
            for i in 0..a.len() {
                out.insert(a.value(i));
            }
        }
        out
    }

    fn first_title(batches: &[RecordBatch]) -> String {
        let b = &batches[0];
        col_str(b, "title").value(0).to_string()
    }

    fn scores(batches: &[RecordBatch]) -> Vec<f32> {
        let mut out = Vec::new();
        for b in batches {
            let c = col_f32(b, "score");
            out.extend((0..c.len()).map(|i| c.value(i)));
        }
        out
    }

    /// Every value of the string column `name` across `batches`, in order.
    fn strings_of(batches: &[RecordBatch], name: &str) -> Vec<String> {
        batches
            .iter()
            .flat_map(|b| {
                let c = col_str(b, name);
                (0..c.len())
                    .map(|i| c.value(i).to_owned())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// The one value a `SELECT COUNT(*)` returns.
    fn count_of(batches: &[RecordBatch]) -> i64 {
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is int64")
            .value(0)
    }

    /// An `emb` column of `dim`-wide vectors, row `i` set to `points[i]` in
    /// its first two components and zero elsewhere. Requires `dim >= 2`.
    fn graded_emb(points: &[[f32; 2]], dim: usize) -> ArrayRef {
        let flat: Vec<f32> = points
            .iter()
            .flat_map(|&[x, y]| [x, y].into_iter().chain(repeat_n(0.0, dim - 2)))
            .collect();
        Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
                Arc::new(Float32Array::from(flat)) as ArrayRef,
                None,
            )
            .expect("FSL"),
        )
    }

    // ---- rrf_fuse unit (deterministic fusion math) ----

    /// A hit in `seg` whose stable `_id` is its local doc id.
    fn stamped(seg: SuperfileUri, doc: u32, score: f32) -> SuperfileHit {
        SuperfileHit {
            superfile: seg,
            local_doc_id: doc,
            score,
            stable_id: Some(i128::from(doc)),
        }
    }

    #[test]
    fn rrf_fuse_boosts_shared_hits_and_orders_by_fused_score() {
        let seg = SuperfileUri::new_v4();
        let h = |doc: u32, score: f32| stamped(seg, doc, score);
        // BM25 best-first: doc1, doc2, doc3. Vector best-first: doc2, doc4.
        let bm25 = vec![h(1, 9.0), h(2, 8.0), h(3, 7.0)];
        let vector = vec![h(2, 0.1), h(4, 0.2)];
        let fused = rrf_fuse(&bm25, &vector, 10).expect("fuse");

        // doc2 is in both lists → highest fused score.
        let ids: Vec<u32> = fused.iter().map(|x| x.local_doc_id).collect();
        assert_eq!(ids, vec![2, 1, 4, 3], "RRF order: shared hit first");

        let s2 = 1.0 / (RRF_K + 2.0) + 1.0 / (RRF_K + 1.0); // bm25 rank2 + vec rank1
        assert!(
            (fused[0].score - s2).abs() < 1e-6,
            "doc2 fused score must sum both contributions"
        );
    }

    #[test]
    fn rrf_fuse_truncates_to_k() {
        let seg = SuperfileUri::new_v4();
        let h = |doc: u32| stamped(seg, doc, 0.0);
        let bm25 = vec![h(1), h(2), h(3)];
        let vector = vec![h(4), h(5)];
        let fused = rrf_fuse(&bm25, &vector, 2).expect("fuse");
        assert_eq!(fused.len(), 2, "fused list truncates to k");
    }

    #[test]
    fn rrf_fuse_keys_on_stable_id_not_local_doc_id() {
        // The same superfile and local holding two different stable ids are
        // two rows (a vector local and a BM25 local name different rows in a
        // cell-packed file), so they never merge.
        let seg = SuperfileUri::new_v4();
        let bm25 = vec![SuperfileHit {
            stable_id: Some(7),
            ..stamped(seg, 0, 1.0)
        }];
        let vector = vec![SuperfileHit {
            stable_id: Some(8),
            ..stamped(seg, 0, 0.1)
        }];
        let fused = rrf_fuse(&bm25, &vector, 10).expect("fuse");
        assert_eq!(fused.len(), 2, "distinct stable ids → distinct hits");
    }

    #[test]
    fn rrf_fuse_breaks_ties_by_stable_id() {
        // Rank 0 in either list scores the same, so the two rows tie; the
        // smaller `_id` wins the one slot, whatever superfile each names.
        let bm25 = vec![SuperfileHit {
            stable_id: Some(9),
            ..stamped(SuperfileUri::new_v4(), 0, 1.0)
        }];
        let vector = vec![SuperfileHit {
            stable_id: Some(3),
            ..stamped(SuperfileUri::new_v4(), 5, 0.1)
        }];
        let fused = rrf_fuse(&bm25, &vector, 1).expect("fuse");
        assert_eq!(fused[0].stable_id, Some(3));
    }

    #[test]
    fn rrf_fuse_joins_different_layout_locals_by_stable_id() {
        let stable_id = 42i128;
        let bm25_seg = SuperfileUri::new_v4();
        let bm25 = vec![SuperfileHit {
            superfile: bm25_seg,
            local_doc_id: 3,
            score: 1.0,
            stable_id: Some(stable_id),
        }];
        // The vector leg found the same row through another superfile
        // (a hidden-index cell file) at another local.
        let vector = vec![SuperfileHit {
            superfile: SuperfileUri::new_v4(),
            local_doc_id: 17,
            score: 0.1,
            stable_id: Some(stable_id),
        }];

        let fused = rrf_fuse(&bm25, &vector, 10).expect("fuse");
        assert_eq!(fused.len(), 1, "one logical row must fuse once");
        assert_eq!(fused[0].stable_id, Some(stable_id));
        assert_eq!(
            (fused[0].superfile, fused[0].local_doc_id),
            (bm25_seg, 3),
            "a row both legs found keeps the BM25 hit, which addresses its user row"
        );
        let expected = 2.0 / (RRF_K + 1.0);
        assert!((fused[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn rrf_fuse_refuses_an_unstamped_hit() {
        let seg = SuperfileUri::new_v4();
        let bm25 = vec![stamped(seg, 1, 1.0)];
        let vector = vec![SuperfileHit {
            stable_id: None,
            ..stamped(seg, 1, 0.1)
        }];
        let err = rrf_fuse(&bm25, &vector, 10).expect_err("unstamped hit must not fuse");
        assert!(matches!(err, QueryError::Internal(_)), "{err:?}");
    }

    // ---- `_id` + `score` projection on a drained table ----

    /// SQL `hybrid_search` serves a projection of only `_id` and `score` from
    /// the fused hits, as `vector_search` does, instead of first placing each
    /// hit on its user row. On a drained, compacted table every vector hit
    /// names a vector-index file, and placing one means looking its `_id` up
    /// in the user table, which builds an `_id` -> row map per merged file.
    /// The `_id` + `score` query builds none and returns the public search's
    /// rows; adding `title`, which does need the rows, builds one, so the
    /// fixture does exercise placement.
    #[test]
    fn sql_id_score_projection_skips_placement() {
        let dim = 16;
        let k = 4;
        let dir = TempDir::new().expect("tempdir");
        let open = || {
            let storage: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
            options_title_emb(dim).with_storage(storage)
        };
        let st = Supertable::create(open()).expect("create");
        let schema = st.options().schema.clone();
        let titles = ["rust async", "python data", "rust systems", "go routines"];
        for _ in 0..3 {
            let mut w = st.writer().expect("writer");
            w.append(&build_batch(&titles, dim, Arc::clone(&schema)))
                .expect("append");
            w.commit().expect("commit");
        }
        st.drain_vectors_to_cells_sync().expect("drain");
        st.optimize(&OptimizeOptions::default()).expect("optimize");
        drop(st);

        // The vector query favours "python data", which the text query
        // never matches, so the fused top-k holds vector-only hits.
        let sql = |columns: &str| {
            format!(
                "SELECT {columns} FROM hybrid_search('title', 'rust', 'emb', '{}', {k})",
                csv_one_hot(dim, 1)
            )
        };
        // Rows of `query`'s first two columns, and how many `_id` -> row
        // maps it built, on a freshly opened table.
        let run = |query: &str| -> (Vec<(i128, f32)>, usize) {
            let st = Supertable::open(open()).expect("open");
            let reader = st.reader().expect("reader");
            let batches = reader.query_sql(query).expect("sql hybrid_search");
            let maps = reader.block_on(async {
                reader
                    .manifest()
                    .options
                    .gapped_id_placement_cache
                    .lock()
                    .await
                    .entries
                    .len()
            });
            let rows = batches
                .iter()
                .flat_map(|b| {
                    let ids = b
                        .column(0)
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .expect("decimal _id");
                    let scores = b
                        .column(1)
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("f32 score");
                    (0..b.num_rows())
                        .map(|i| (ids.value(i), scores.value(i)))
                        .collect::<Vec<_>>()
                })
                .collect();
            (rows, maps)
        };

        let (with_title, maps) = run(&sql("_id, score, title"));
        assert!(
            maps > 0,
            "placing a vector-only hit builds an `_id` -> row map"
        );
        let (id_score, maps) = run(&sql("_id, score"));
        assert_eq!(maps, 0, "an `_id` + `score` query places nothing");
        assert_eq!(id_score, with_title, "same rows either way");

        let st = Supertable::open(open()).expect("open");
        let q: Vec<f32> = (0..dim).map(|d| if d == 1 { 1.0 } else { 0.0 }).collect();
        let public = st
            .hybrid_search(
                "title",
                "rust",
                BoolMode::Or,
                "emb",
                &q,
                VectorSearchOptions::new(),
                k,
                Some(&["_id", "score"]),
            )
            .expect("public hybrid_search");
        let public: Vec<(i128, f32)> = public
            .iter()
            .flat_map(|b| {
                let ids = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("decimal _id");
                let scores = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .expect("f32 score");
                (0..b.num_rows())
                    .map(|i| (ids.value(i), scores.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(id_score, public, "the public search's rows");
    }

    // ---- end-to-end through query_sql ----

    #[test]
    fn hybrid_search_identity_set_is_union_of_subsearches() {
        // With k = n, the vector retriever returns every doc, so the
        // fused identity set must equal bm25 ∪ vector exactly.
        let dim = 16;
        let st = demo(dim);
        let qv = csv_one_hot(dim, 4);
        let k = 8;

        let hybrid = id_set(
            &st.reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
                ))
                .expect("hybrid query_sql"),
        );
        let bm25 = id_set(
            &st.reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT _id FROM bm25_search('title', 'rust', {k})"
                ))
                .expect("bm25 query_sql"),
        );
        let vector = id_set(
            &st.reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT _id FROM vector_search('emb', '{qv}', {k})"
                ))
                .expect("vector query_sql"),
        );

        let expected: HashSet<i128> = bm25.union(&vector).copied().collect();
        assert_eq!(hybrid, expected, "hybrid identity set = bm25 ∪ vector");
    }

    #[test]
    fn hybrid_search_ranks_doc_top_in_both_retrievers_first() {
        // `async` is unique to doc 0 (BM25 rank 1) and the one-hot
        // query at dim 0 makes doc 0 the exact vector match (rank 1).
        // Top in both → highest RRF → emitted first.
        let dim = 16;
        let st = demo(dim);
        let res = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title, score FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        assert_eq!(first_title(&res), "rust async", "doc top in both ranks #1");

        // RRF score is descending (higher is better).
        let s = scores(&res);
        for w in s.windows(2) {
            assert!(w[0] >= w[1], "fused scores must be descending: {s:?}");
        }
    }

    #[test]
    fn hybrid_search_text_only_match_survives_fusion() {
        // The vector query targets dim 7 (doc 7), but `async` is only
        // in doc 0. Doc 0 must still appear — fusion unions the two
        // candidate sets, it does not intersect them.
        let dim = 16;
        let st = demo(dim);
        let res = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT title FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
                csv_one_hot(dim, 7)
            ))
            .expect("query_sql");
        let titles: HashSet<String> = res
            .iter()
            .flat_map(|b| {
                let c = col_str(b, "title");
                (0..c.len())
                    .map(|i| c.value(i).to_string())
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            titles.contains("rust async"),
            "text-only match must survive fusion; got {titles:?}"
        );
    }

    #[test]
    fn hybrid_search_star_projection_appends_score_column() {
        let dim = 16;
        let st = demo(dim);
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT * FROM hybrid_search('title', 'rust', 'emb', '{}', 3)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        let b = &batches[0];
        // Scalar schema (_id, title) + score.
        assert_eq!(b.num_columns(), 3);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "title");
        assert_eq!(b.schema().field(2).name(), "score");
    }

    #[test]
    fn hybrid_search_empty_supertable_returns_no_rows() {
        let dim = 16;
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT _id, score FROM hybrid_search('title', 'rust', 'emb', '{}', 5)",
                csv_one_hot(dim, 0)
            ))
            .expect("query_sql");
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 0);
    }

    #[test]
    fn hybrid_search_arity_error() {
        let dim = 16;
        let st = demo(dim);
        // 4 args (missing k) → planning error.
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '1,0')")
                .is_err()
        );
    }

    #[test]
    fn hybrid_search_bad_arg_types_error() {
        let dim = 16;
        let st = demo(dim);
        // Non-integer k (5th arg).
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '1,0', 'five')")
                .is_err(),
            "non-integer k must error"
        );
        // Unparseable query vector (4th arg).
        assert!(
            st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM hybrid_search('title', 'rust', 'emb', 'a,b', 3)")
                .is_err(),
            "bad query vector must error"
        );
    }

    #[test]
    fn hybrid_search_sync_method_fuses_both_retrievers() {
        // Exercises the public sync `hybrid_search` wrapper (and through
        // it `hybrid_search_async` + `rrf_fuse`).
        let dim = 16;
        let st = demo(dim);
        let mut qv = vec![0.0_f32; dim];
        qv[0] = 1.0; // one-hot at dim 0 → doc 0 exact vector match.
        let hits = st
            .reader()
            .expect("reader")
            .hybrid_search(
                "title",
                "async",
                BoolMode::Or,
                "emb",
                &qv,
                VectorSearchOptions::new(),
                8,
            )
            .expect("hybrid_search");
        assert!(!hits.is_empty(), "fused hits non-empty");
        // RRF score is higher-is-better → emitted descending.
        for w in hits.windows(2) {
            assert!(w[0].score >= w[1].score, "fused scores descending");
        }
    }

    #[test]
    fn hybrid_search_public_rows_match_sql_tvf() {
        // The public row-returning `Supertable::hybrid_search` must yield
        // the same `_id` set the `hybrid_search` SQL table function does
        // (the SQL TVF fixes mode = Or and default options, so match it).
        let dim = 16;
        let st = demo(dim);
        let mut qv = vec![0.0_f32; dim];
        qv[0] = 1.0;
        let rows = st
            .hybrid_search(
                "title",
                "async",
                BoolMode::Or,
                "emb",
                &qv,
                VectorSearchOptions::new(),
                8,
                Some(&["_id", "score"]),
            )
            .expect("hybrid_search rows");
        let direct = id_set(&rows);
        assert!(!direct.is_empty(), "direct call returns rows");
        assert_eq!(rows[0].num_columns(), 2, "_id + score, no scalar decode");

        // Named projection materializes the requested scalar column.
        let projected = st
            .hybrid_search(
                "title",
                "async",
                BoolMode::Or,
                "emb",
                &qv,
                VectorSearchOptions::new(),
                8,
                Some(&["_id", "title", "score"]),
            )
            .expect("hybrid_search projected");
        assert_eq!(projected[0].num_columns(), 3);

        let via_sql = id_set(
            &st.reader()
                .expect("reader")
                .query_sql(&format!(
                    "SELECT _id FROM hybrid_search('title', 'async', 'emb', '{}', 8)",
                    csv_one_hot(dim, 0)
                ))
                .expect("sql tvf"),
        );
        assert_eq!(direct, via_sql, "direct call and SQL TVF agree on _id set");
    }

    /// Flatten an `EXPLAIN` result into one searchable string — exercises
    /// `HybridSearchExec`'s `DisplayAs`/`describe`.
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

    /// Rows [`demo`] writes.
    const DEMO_ROWS: usize = 8;

    /// Rows of [`demo`] with "rust" in the title; each search below asks for
    /// exactly these.
    const RUST_ROWS: usize = 2;

    /// A `k` past [`DEMO_ROWS`]: a search asking for more rows than its
    /// table holds.
    const K_PAST_THE_TABLE: usize = 100;

    /// The plan DataFusion runs for `sql`, from the reader's own session,
    /// collecting a join's build side into one hash table only under
    /// `collect_rows` rows (DataFusion's default when `None`).
    fn physical_plan(
        st: &Supertable,
        sql: &str,
        collect_rows: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let reader = st.reader().expect("reader");
        let mut state = reader.sql_session_context().expect("session").state();
        if let Some(rows) = collect_rows {
            state
                .config_mut()
                .options_mut()
                .optimizer
                .hash_join_single_partition_threshold_rows = rows;
        }
        let ctx = SessionContext::new_with_state(state);
        reader
            .block_on(async { ctx.sql(sql).await?.create_physical_plan().await })
            .expect("physical plan")
    }

    /// The first hash join in `plan`, top down.
    fn hash_join(plan: &Arc<dyn ExecutionPlan>) -> Option<&HashJoinExec> {
        plan.downcast_ref::<HashJoinExec>()
            .or_else(|| plan.children().into_iter().find_map(hash_join))
    }

    /// Whether a node named `name` runs anywhere in `plan`.
    fn runs(plan: &Arc<dyn ExecutionPlan>, name: &str) -> bool {
        plan.name() == name || plan.children().into_iter().any(|child| runs(child, name))
    }

    /// A ranked search joined back to its table on `_id` builds the join's
    /// hash table on its hits, not on the table: in every shape a query can
    /// spell the join-back in, whether DataFusion collects the build side
    /// into one hash table or partitions both sides, and when `k` asks for
    /// more rows than the table holds. Where the join drops unmatched table
    /// rows, the hits' `_id`s reach the table scan as a filter.
    #[test]
    fn a_search_joined_back_on_id_builds_on_its_hits() {
        let dim = 16;
        let st = demo(dim);
        let qv = csv_one_hot(dim, 0);
        let bm25 = format!("bm25_search('title', 'rust', {RUST_ROWS})");
        let shapes = [
            (
                format!("{bm25} s JOIN supertable t ON t._id = s._id"),
                "Bm25Exec",
                true,
                RUST_ROWS,
            ),
            (
                format!(
                    "vector_search('emb', '{qv}', {RUST_ROWS}) s \
                     JOIN supertable t ON t._id = s._id"
                ),
                "VectorSearchExec",
                true,
                RUST_ROWS,
            ),
            (
                format!(
                    "hybrid_search('title', 'rust', 'emb', '{qv}', {RUST_ROWS}) s \
                     JOIN supertable t ON t._id = s._id"
                ),
                "HybridSearchExec",
                true,
                RUST_ROWS,
            ),
            (
                format!("supertable t JOIN {bm25} s ON t._id = s._id"),
                "Bm25Exec",
                true,
                RUST_ROWS,
            ),
            (
                format!("supertable t WHERE t._id IN (SELECT _id FROM {bm25})"),
                "Bm25Exec",
                true,
                RUST_ROWS,
            ),
            (
                format!("{bm25} s LEFT JOIN supertable t ON t._id = s._id"),
                "Bm25Exec",
                true,
                RUST_ROWS,
            ),
            // Every table row survives a left join from the table, so the
            // hits can filter nothing out of its scan.
            (
                format!("supertable t LEFT JOIN {bm25} s ON t._id = s._id"),
                "Bm25Exec",
                false,
                DEMO_ROWS,
            ),
            (
                format!(
                    "bm25_search('title', 'rust', {K_PAST_THE_TABLE}) s \
                     JOIN supertable t ON t._id = s._id"
                ),
                "Bm25Exec",
                true,
                RUST_ROWS,
            ),
        ];
        // `Some(0)`: nothing is small enough to collect, so every join
        // partitions and the build side comes from comparing row counts alone.
        for collect_rows in [None, Some(0)] {
            for (from, search, filters_scan, rows) in &shapes {
                let sql = format!("SELECT t.title FROM {from}");
                let plan = physical_plan(&st, &sql, collect_rows);
                let join = hash_join(&plan).expect("the join-back plans as a hash join");
                let shown = displayable(plan.as_ref()).indent(true).to_string();
                assert!(
                    runs(join.left(), search),
                    "the join must build on the {search} hits ({collect_rows:?}): {sql}\n{shown}"
                );
                let probe = displayable(join.right().as_ref()).indent(true).to_string();
                assert_eq!(
                    probe.contains("DynamicFilter"),
                    *filters_scan,
                    "the hits' `_id`s reach the table scan iff the join drops unmatched \
                     table rows ({collect_rows:?}): {sql}\n{shown}"
                );
                assert_eq!(sql_row_count(&st, &sql), *rows, "{sql}");
            }
        }
    }

    /// A search's row count is an estimate the planner may act on, never a
    /// fact it may answer from: `COUNT(*)` counts the rows the search finds,
    /// not its `k`, and a `k` near `i64::MAX` plans a join-back without
    /// overflowing DataFusion's join-size arithmetic, because the count it
    /// reports is bounded by the table's rows.
    #[test]
    fn a_search_reports_its_k_as_an_estimate_only() {
        let st = demo(16);
        let count = |sql: &str| -> i64 {
            let batches = st.reader().expect("reader").query_sql(sql).expect("count");
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("count is Int64")
                .value(0)
        };
        assert_eq!(
            count(&format!(
                "SELECT COUNT(*) FROM bm25_search('title', 'rust', {K_PAST_THE_TABLE})"
            )),
            RUST_ROWS as i64,
            "the rows the search finds, not the k it allows"
        );
        assert_eq!(
            count(&format!(
                "SELECT COUNT(*) FROM bm25_search('title', 'rust', {}) s \
                 JOIN supertable t ON t._id = s._id",
                i64::MAX
            )),
            RUST_ROWS as i64
        );
    }

    /// Construct `HybridSearchTable` directly through the TVF `call`
    /// path and exercise its `TableProvider` metadata methods (`Debug`,
    /// `as_any`, `table_type`) plus the lowered `HybridSearchExec`'s
    /// `name` / `Debug` — none of which normal query execution touches.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn hybrid_table_and_exec_trait_methods() {
        use datafusion::{execution::context::SessionContext, prelude::lit};

        let dim = 16;
        let st = demo(dim);
        let reader = Arc::new(st.reader().expect("reader"));
        let scalar_schema = reader.manifest().scalar_schema();
        use crate::supertable::query::exec::common::test_support::{call_tvf, scoped_inner};
        let func = HybridSearchFunc::new(reader, scalar_schema);
        let table = call_tvf(
            &func,
            &[
                lit("title"),
                lit("rust"),
                lit("emb"),
                lit(csv_one_hot(dim, 0)),
                lit(5_i64),
            ],
        )
        .expect("hybrid table");

        let dbg = format!("{table:?}");
        assert!(dbg.contains("HybridSearchTable"), "Debug missing: {dbg}");
        assert!(
            scoped_inner(&table)
                .downcast_ref::<HybridSearchTable>()
                .is_some(),
            "as_any downcasts to HybridSearchTable"
        );
        assert_eq!(table.table_type(), TableType::Base);

        let ctx = SessionContext::new();
        let plan = table
            .scan(&ctx.state(), None, &[], None)
            .await
            .expect("scan");
        assert_eq!(plan.name(), "HybridSearchExec");
        assert!(
            format!("{plan:?}").contains("HybridSearchExec"),
            "Exec Debug missing"
        );
    }

    #[test]
    fn hybrid_exec_display_describes_invocation() {
        let dim = 16;
        let st = demo(dim);
        let text = explain(
            &st,
            &format!(
                "SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{}', 5)",
                csv_one_hot(dim, 0)
            ),
        );
        assert!(
            text.contains("HybridSearchExec")
                && text.contains("text_col=title")
                && text.contains("vec_col=emb")
                && text.contains("k=5"),
            "hybrid describe missing: {text}"
        );
    }

    // ---- sql × vector × fts composed in ONE query ----
    //
    // The TVFs above each test one retriever in isolation. This block
    // exercises the composition the *pushdown contract* promises
    // but nothing else covered: a single SQL statement that JOINs an
    // FTS retriever (`bm25_search`) and a vector retriever
    // (`vector_search`) on the durable `_id` and post-filters with a
    // scalar `WHERE` — across two superfiles.

    /// Schema `[category (scalar), title (FTS), emb (vector)]`. Unlike
    /// `options_title_emb`, `category` is a plain scalar column (not in
    /// any FtsConfig/VectorConfig), so it is filterable from SQL.
    fn options_cat_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("category", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig::new("title")],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Shared base component of every dim in [`build_batch_cat`] embeddings.
    /// Kept well inside the fixed Sq8 grid's `[-1, 1]` span so nothing
    /// clamps during encode.
    const CAT_EMB_BASE: f32 = 0.2;
    /// Per-doc perturbation on the doc's active dim. Large enough that
    /// adjacent docs' cosine scores separate well above Sq8 rerank noise,
    /// small enough that every doc stays in one grid cell.
    const CAT_EMB_EPSILON: f32 = 0.05;

    /// Doc `i` (0-based within the batch) gets `cats[i]`, `titles[i]`, and an
    /// embedding `base·1 + ε·e_{base_dim+i}`. Every doc shares the same
    /// dominant direction, so the global cell grid assigns the whole corpus
    /// to one cell — the routed default (one grid-nearest cell) then covers
    /// every doc and `vector_search` stays exact for the oracle below — while
    /// the ε one-hot component keeps the graded-query ranking strict in `i`.
    fn build_batch_cat(
        cats: &[&str],
        titles: &[&str],
        base_dim: usize,
        dim: usize,
        schema: Arc<Schema>,
    ) -> RecordBatch {
        let n = titles.len();
        let cat_arr = LargeStringArray::from(cats.to_vec());
        let title_arr = LargeStringArray::from(titles.to_vec());
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            let active = base_dim + i;
            for d in 0..dim {
                flat.push(if d == active {
                    CAT_EMB_BASE + CAT_EMB_EPSILON
                } else {
                    CAT_EMB_BASE
                });
            }
        }
        let fsl = FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(flat)) as ArrayRef,
            None,
        )
        .expect("FSL");
        RecordBatch::try_new(
            schema,
            vec![Arc::new(cat_arr), Arc::new(title_arr), Arc::new(fsl)],
        )
        .expect("batch")
    }

    /// Two-superfile corpus (docs 0–3, then 4–7) engineered so the three
    /// retrievers each drop a *different* doc. With dim = 8 and a graded
    /// query vector `[8,7,…,1]`, the cosine distance to one-hot doc `i`
    /// is strictly increasing in `i`, so `vector_search(k=5)` is exactly
    /// {0,1,2,3,4} (deterministic, no ties). Memberships:
    ///   - `rust` (FTS)              = {0,2,3,5,6}
    ///   - vector top-5              = {0,1,2,3,4}
    ///   - `category='systems'`      = {0,1,3,5,7}
    ///   - three-way intersection    = {0,3}
    /// Sole-reason witnesses: doc1 dropped only by FTS, doc2 only by the
    /// scalar filter, doc5 only by the vector cutoff.
    fn demo_cat_two_superfiles(dim: usize) -> Supertable {
        let st = Supertable::create(options_cat_title_emb(dim)).expect("create");
        let schema = st.options().schema.clone();
        // Each writer holds the single-writer lock until dropped, so
        // scope them: seg1's writer must drop before seg2's opens.
        {
            let mut w = st.writer().expect("writer seg1");
            w.append(&build_batch_cat(
                &["systems", "systems", "cooking", "systems"],
                &["rust alpha", "python beta", "rust gamma", "rust delta"],
                0,
                dim,
                schema.clone(),
            ))
            .expect("append seg1");
            w.commit().expect("commit seg1");
        }
        {
            let mut w = st.writer().expect("writer seg2");
            w.append(&build_batch_cat(
                &["cooking", "systems", "cooking", "systems"],
                &["python epsilon", "rust zeta", "rust eta", "python theta"],
                4,
                dim,
                schema,
            ))
            .expect("append seg2");
            w.commit().expect("commit seg2");
        }
        st
    }

    /// Rows across `batches` whose `category` equals `want`.
    fn count_category(batches: &[RecordBatch], want: &str) -> usize {
        batches
            .iter()
            .map(|b| {
                let c = col_str(b, "category");
                (0..c.len()).filter(|&i| c.value(i) == want).count()
            })
            .sum()
    }

    /// A `WHERE` on a column no index bounds: the function applies the
    /// exact predicate to its fused rows and widens both retrievers until
    /// `k` survive — where a post-filtered global top-k came up short.
    #[test]
    fn hybrid_search_where_on_unindexed_column_fills_k() {
        let dim = 16;
        let st = demo_cat_two_superfiles(dim);
        let qv: String = (0..dim)
            .map(|d| (dim - d).to_string())
            .collect::<Vec<_>>()
            .join(",");
        let k = 3;
        // Guard: the unfiltered fused top-3 holds fewer than 3 `cooking`
        // rows (vector rank is doc order, `rust` matches docs 0,2,3,5,6;
        // only doc 2 is both cooking and near the top of either list).
        let unfiltered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT category FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
            ))
            .expect("unfiltered query_sql");
        assert!(
            count_category(&unfiltered, "cooking") < k,
            "guard: a post-filtered global top-{k} would underflow"
        );
        let filtered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT category, score FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k}) \
                 WHERE category = 'cooking'"
            ))
            .expect("filtered query_sql");
        let total: usize = filtered.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, k, "k rows, every one cooking");
        assert_eq!(count_category(&filtered, "cooking"), k);
        let s = scores(&filtered);
        for w in s.windows(2) {
            assert!(w[0] >= w[1], "fused scores stay descending: {s:?}");
        }
    }

    /// Schema `[tag (FTS), title (FTS), emb]`: `tag` is the one-keyword
    /// column the index bounds a `WHERE` on.
    fn options_tag_title_emb(dim: usize) -> SupertableOptions {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("tag", DataType::LargeUtf8, false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("emb", fixed_list_f32(dim), false),
        ]));
        SupertableOptions::new(
            schema,
            vec![FtsConfig::new("tag"), FtsConfig::new("title")],
            vec![VectorConfig {
                column: "emb".into(),
                dim,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Fp32,
                provided_centroids: None,
            }],
        )
        .expect("valid options")
        .with_writer_pool(pool)
    }

    /// Eight rows all titled `rust` (one BM25 score, ties broken by doc
    /// id) with vector `[1, i, 0, …]` (distance to `[1, 0, …]` rises with
    /// `i`), so both retrievers rank doc order; the first five are tagged
    /// `a`, the last three `b`.
    fn supertable_tag_ranked(dim: usize) -> Supertable {
        let st = Supertable::create(options_tag_title_emb(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let n = 8;
        let tags = LargeStringArray::from(
            (0..n)
                .map(|i| if i < 5 { "a" } else { "b" })
                .collect::<Vec<_>>(),
        );
        let titles = LargeStringArray::from(vec!["rust"; n]);
        let points: Vec<[f32; 2]> = (0..n).map(|i| [1.0, i as f32]).collect();
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![Arc::new(tags), Arc::new(titles), graded_emb(&points, dim)],
        )
        .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st
    }

    /// A `WHERE` on an FTS-indexed column: one scope bounds the rows for
    /// both retrievers, so the fused top-k is exactly the `k` best rows
    /// satisfying it — the three `b` rows, in fused order.
    #[test]
    fn hybrid_search_where_on_indexed_column_ranks_among_matching() {
        let dim = 16;
        let st = supertable_tag_ranked(dim);
        let k = 3;
        let qv = csv_one_hot(dim, 0);
        let unfiltered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT tag FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
            ))
            .expect("unfiltered query_sql");
        assert_eq!(
            strings_of(&unfiltered, "tag"),
            vec!["a", "a", "a"],
            "guard: the unfiltered top-{k} is all `a`"
        );
        let filtered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT tag FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k}) WHERE tag = 'b'"
            ))
            .expect("filtered query_sql");
        assert_eq!(strings_of(&filtered, "tag"), vec!["b", "b", "b"]);
        // An aggregate over the filtered relation counts the matching rows,
        // not the survivors of a global fused top-k — the same acceptance
        // case `bm25_search` holds, on the fused function.
        let count = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT COUNT(*) AS n FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k}) \
                 WHERE tag = 'b'"
            ))
            .expect("count query_sql");
        assert_eq!(count_of(&count), k as i64);
    }

    // ---- a WHERE whose rows sit past the widening's ceiling ----

    /// Rows titled `common`, nearer the query than every row the tests'
    /// `WHERE` keeps, and ranked above them by BM25 too: more than the
    /// widening's ceiling lets a small `k` look at, so a fill that only
    /// widens stops with every kept row still beyond it in both legs.
    const N_NEAR_PAST_THE_CEILING: usize = OVER_FETCH_MIN_CEILING + 100;

    /// Rows `WHERE title <> 'common'` keeps, all past the
    /// [`N_NEAR_PAST_THE_CEILING`] rejected ones in both legs.
    const N_FAR_PAST_THE_CEILING: usize = 5;

    /// Rows the drained-plus-tail test appends after the drain, also kept
    /// by the `WHERE`.
    const N_TAIL_PAST_THE_CEILING: usize = 5;

    /// Spacing of the tail rows' first component: tail row `t` is
    /// `[-(t + 1) · step, 1]`, nearer the query than every rare row (whose
    /// first component is at most -1) and farther than every near one.
    const TAIL_STEP: f32 = 0.1;

    /// A `k` above every row the `WHERE` keeps in any of these tests, so a
    /// query at it returns all of them.
    const K_PAST_EVERY_MATCH: usize = N_FAR_PAST_THE_CEILING + N_TAIL_PAST_THE_CEILING + 1;

    /// Vector width of the past-the-ceiling tests: the engine's smallest.
    const PAST_THE_CEILING_DIM: usize = 16;

    /// Rare row `j`'s title: `common` plus padding, so BM25 matches it but
    /// ranks it below every one-word `common` row, and shorter titles above
    /// longer ones: rare row `N_FAR_PAST_THE_CEILING - 1` first.
    fn rare_title(j: usize) -> String {
        format!(
            "common rare{j}{}",
            " pad".repeat(N_FAR_PAST_THE_CEILING - 1 - j)
        )
    }

    /// Row `i` of the near rows: `[1, i]`, distance to `[1, 0]` rising with
    /// `i`, every one nearer than any row with a negative first component.
    fn near_point(i: usize) -> [f32; 2] {
        [1.0, i as f32]
    }

    /// Rare row `j`: `[-(N - j), 1]`, farther than every near row and
    /// nearest for the largest `j`, the reverse of the order rows are
    /// written in. The cosines run from -0.71 to -0.98, well apart.
    fn rare_point(j: usize) -> [f32; 2] {
        [-((N_FAR_PAST_THE_CEILING - j) as f32), 1.0]
    }

    /// One superfile created with `options`, row `i` titled `rows[i].0` with
    /// the vector [`graded_emb`] makes of `rows[i].1`.
    fn supertable_of(options: SupertableOptions, rows: &[(String, [f32; 2])]) -> Supertable {
        let st = Supertable::create(options).expect("create");
        append_rows(&st, rows);
        st
    }

    /// Append and commit `rows` as [`supertable_of`] lays them out.
    fn append_rows(st: &Supertable, rows: &[(String, [f32; 2])]) {
        let mut w = st.writer().expect("writer");
        let titles =
            LargeStringArray::from(rows.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>());
        let points: Vec<[f32; 2]> = rows.iter().map(|(_, p)| *p).collect();
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![Arc::new(titles), graded_emb(&points, PAST_THE_CEILING_DIM)],
        )
        .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }

    /// [`N_NEAR_PAST_THE_CEILING`] near rows titled `common`, then the
    /// [`N_FAR_PAST_THE_CEILING`] rare ones. Searching `common` near
    /// `[1, 0, …]`, both legs rank every near row above every rare one, and
    /// rank the rare rows the same way: `rare4` first, `rare0` last.
    fn common_then_rare() -> Vec<(String, [f32; 2])> {
        (0..N_NEAR_PAST_THE_CEILING)
            .map(|i| ("common".to_owned(), near_point(i)))
            .chain((0..N_FAR_PAST_THE_CEILING).map(|j| (rare_title(j), rare_point(j))))
            .collect()
    }

    /// LocalFs-backed options for the past-the-ceiling tables: the drain and
    /// deletes need storage.
    fn options_on_disk(dir: &TempDir) -> SupertableOptions {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        options_title_emb(PAST_THE_CEILING_DIM).with_storage(storage)
    }

    /// [`supertable_of`] [`common_then_rare`] on disk, every row drained
    /// into the hidden vector index.
    fn drained_common_then_rare(dir: &TempDir) -> Supertable {
        let rows = common_then_rare();
        let st = supertable_of(options_on_disk(dir), &rows);
        st.drain_vectors_to_cells_sync().expect("drain");
        let hidden = st.vector_index_table().expect("hidden index");
        assert_eq!(
            hidden.reader().expect("hidden reader").n_docs_total(),
            rows.len() as u64,
            "every row is in the vector index"
        );
        st
    }

    /// `SELECT {select}` from the `common` search near `[1, 0, …]` at `k`,
    /// `WHERE title <> 'common'`: only the rare rows, and any tail, pass.
    fn where_not_common(st: &Supertable, k: usize, select: &str) -> Vec<RecordBatch> {
        let q = csv_one_hot(PAST_THE_CEILING_DIM, 0);
        st.reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT {select} FROM hybrid_search('title', 'common', 'emb', '{q}', {k}) \
                 WHERE title <> 'common'"
            ))
            .expect("filtered query_sql")
    }

    /// The `k` best rare rows of a [`common_then_rare`] table, best first,
    /// each scored as both legs ranking it at its place among the rare rows:
    /// the exact pass ranks each leg within exactly the rows the `WHERE`
    /// keeps, then fuses. Every rare row comes back at a `k` past them all,
    /// and an aggregate counts each one.
    fn assert_hybrid_ranks_the_rare_rows(st: &Supertable) {
        let k = 3;
        let rows = where_not_common(st, k, "title, score");
        let best: Vec<String> = (0..k)
            .map(|r| rare_title(N_FAR_PAST_THE_CEILING - 1 - r))
            .collect();
        assert_eq!(strings_of(&rows, "title"), best, "the k best, best first");
        for (rank, score) in scores(&rows).into_iter().enumerate() {
            let both_legs = 2.0 / (RRF_K + 1.0 + rank as f32);
            assert!(
                (score - both_legs).abs() < f32::EPSILON,
                "rank {rank}: both legs rank it {rank}th among the kept rows; got {score}"
            );
        }

        let all = where_not_common(st, K_PAST_EVERY_MATCH, "title");
        let every: Vec<String> = (0..N_FAR_PAST_THE_CEILING).rev().map(rare_title).collect();
        assert_eq!(
            strings_of(&all, "title"),
            every,
            "fewer than k match: every one of them, no more"
        );
        assert_eq!(
            count_of(&where_not_common(st, K_PAST_EVERY_MATCH, "COUNT(*) AS n")),
            N_FAR_PAST_THE_CEILING as i64
        );
    }

    /// A predicate no index bounds, whose rows all rank past the widening's
    /// ceiling in both legs: a short widening makes the function check the
    /// predicate over the scope's rows and run both legs within exactly
    /// those, so it still finds them, ranked as each leg ranks them there.
    #[test]
    fn hybrid_search_where_reaches_rows_past_the_fill_ceiling() {
        let st = supertable_of(options_title_emb(PAST_THE_CEILING_DIM), &common_then_rare());
        assert_hybrid_ranks_the_rare_rows(&st);
    }

    /// The same search on a table whose rows have been moved into the hidden
    /// vector index: the vector leg's search over the exact pass's rows goes
    /// through the index's cells, and ranks the rare rows the same way.
    #[test]
    fn hybrid_search_where_reaches_rows_past_the_fill_ceiling_in_the_vector_index() {
        let dir = TempDir::new().expect("tempdir");
        let st = drained_common_then_rare(&dir);
        assert_hybrid_ranks_the_rare_rows(&st);
    }

    /// A deleted row the `WHERE` would keep never comes back, and never
    /// takes a place among the `k`: the exact pass leaves deleted rows out
    /// of the rows both legs rank within.
    #[test]
    fn hybrid_search_where_past_the_fill_ceiling_skips_a_deleted_match() {
        let dir = TempDir::new().expect("tempdir");
        let st = drained_common_then_rare(&dir);
        let best = rare_title(N_FAR_PAST_THE_CEILING - 1);
        st.delete(col("title").eq(lit(best.as_str())))
            .expect("delete the best rare row");
        let k = 3;
        let next: Vec<String> = (1..=k)
            .map(|r| rare_title(N_FAR_PAST_THE_CEILING - 1 - r))
            .collect();
        assert_eq!(
            strings_of(&where_not_common(&st, k, "title"), "title"),
            next,
            "the k best of the rows still there"
        );
        assert_eq!(
            count_of(&where_not_common(&st, K_PAST_EVERY_MATCH, "COUNT(*) AS n")),
            N_FAR_PAST_THE_CEILING as i64 - 1
        );
    }

    /// Rows the `WHERE` keeps on both sides of the drain: the rare rows in
    /// the hidden vector index, and rows appended after it still only in
    /// their user file. The exact pass reaches both.
    #[test]
    fn hybrid_search_where_past_the_fill_ceiling_reaches_drained_and_undrained_rows() {
        let dir = TempDir::new().expect("tempdir");
        let st = drained_common_then_rare(&dir);
        let tail: Vec<(String, [f32; 2])> = (0..N_TAIL_PAST_THE_CEILING)
            .map(|t| {
                (
                    format!("common tail{t}"),
                    [-TAIL_STEP * (t + 1) as f32, 1.0],
                )
            })
            .collect();
        append_rows(&st, &tail);
        let mut found = strings_of(&where_not_common(&st, K_PAST_EVERY_MATCH, "title"), "title");
        found.sort();
        let mut expected: Vec<String> = (0..N_FAR_PAST_THE_CEILING)
            .map(rare_title)
            .chain(tail.into_iter().map(|(title, _)| title))
            .collect();
        expected.sort();
        assert_eq!(found, expected, "every kept row, drained or not");
    }

    /// Rows in the within-the-ceiling test: few enough that a small `k`'s
    /// ceiling covers the whole table.
    const N_WITHIN_THE_CEILING: usize = OVER_FETCH_MIN_CEILING / 2;

    /// On a table no larger than the widening's ceiling the exact pass never
    /// runs: the last round asks both legs for every row, and that has to
    /// reach rows the text doesn't match and the vector ranks last. Here the
    /// kept rows are exactly those, so the widening alone must find them all.
    #[test]
    fn hybrid_search_where_within_the_fill_ceiling_reaches_the_farthest_rows() {
        let rows: Vec<(String, [f32; 2])> = (0..N_WITHIN_THE_CEILING)
            .map(|i| ("common".to_owned(), near_point(i)))
            .chain((0..N_FAR_PAST_THE_CEILING).map(|j| (format!("rare{j}"), rare_point(j))))
            .collect();
        let st = supertable_of(options_title_emb(PAST_THE_CEILING_DIM), &rows);
        let nearest_first: Vec<String> = (0..N_FAR_PAST_THE_CEILING)
            .rev()
            .map(|j| format!("rare{j}"))
            .collect();
        assert_eq!(
            strings_of(&where_not_common(&st, K_PAST_EVERY_MATCH, "title"), "title"),
            nearest_first,
            "every kept row, nearest first"
        );
    }

    /// A `WHERE` the widening fills `k` for keeps the widening's answer on a
    /// table past the ceiling: the exact pass never runs. The predicate is
    /// one no index bounds, and the text matches nothing, so each score is
    /// the vector leg's rank alone, over the whole table: the rows ranked
    /// 2nd to 4th behind the one row the predicate rejects. Ranked within
    /// only the kept rows, as the exact pass would, they would score as 1st
    /// to 3rd.
    #[test]
    fn hybrid_search_where_that_fills_k_keeps_the_widened_ranks() {
        let rows: Vec<(String, [f32; 2])> = (0..N_NEAR_PAST_THE_CEILING)
            .map(|i| {
                let title = if i == 0 { "nearest" } else { "common" };
                (title.to_owned(), near_point(i))
            })
            .collect();
        let st = supertable_of(options_title_emb(PAST_THE_CEILING_DIM), &rows);
        let q = csv_one_hot(PAST_THE_CEILING_DIM, 0);
        let k = 3;
        let got = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT score FROM hybrid_search('title', 'absent', 'emb', '{q}', {k}) \
                 WHERE title <> 'nearest'"
            ))
            .expect("filtered query_sql");
        let widened: Vec<f32> = (1..=k).map(|r| 1.0 / (RRF_K + 1.0 + r as f32)).collect();
        assert_eq!(scores(&got), widened, "ranked over the whole table");
    }

    #[test]
    fn sql_join_of_bm25_and_vector_with_scalar_filter_matches_three_way_intersection() {
        let dim = 16;
        let st = demo_cat_two_superfiles(dim);
        // Graded query vector [dim, dim-1, …, 1] → cosine distance to
        // one-hot doc `i` is strictly increasing in `i`, so the vector
        // distance rank equals the doc id (no ties among the 8 docs).
        let qv: String = (0..dim)
            .map(|d| (dim - d).to_string())
            .collect::<Vec<_>>()
            .join(",");

        // The three single-retriever result sets — the oracle inputs.
        let fts = id_set(
            &st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM bm25_search('title', 'rust', 8)")
                .expect("bm25 query_sql"),
        );
        let vector = id_set(
            &st.reader()
                .expect("reader")
                .query_sql(&format!("SELECT _id FROM vector_search('emb', '{qv}', 5)"))
                .expect("vector query_sql"),
        );
        let scalar = id_set(
            &st.reader()
                .expect("reader")
                .query_sql("SELECT _id FROM supertable WHERE category = 'systems'")
                .expect("scalar query_sql"),
        );
        // Guard against corpus drift: the witnesses below only hold for
        // these exact membership sizes.
        assert_eq!(fts.len(), 5, "'rust' should match 5 titles");
        assert_eq!(vector.len(), 5, "vector top-5");
        assert_eq!(scalar.len(), 5, "5 'systems' docs");

        // THE query under test: FTS ⋈ vector on the durable _id, then a
        // scalar SQL predicate — sql + vector + fts in one statement,
        // spanning two superfiles.
        let combined_batches = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT b._id, b.title AS title, b.category AS category, b.score AS score \
                 FROM bm25_search('title', 'rust', 8) AS b \
                 JOIN vector_search('emb', '{qv}', 5) AS v ON b._id = v._id \
                 WHERE b.category = 'systems' \
                 ORDER BY b.score DESC"
            ))
            .expect("combined sql+vector+fts query");
        let combined = id_set(&combined_batches);

        // 1. Exact correctness against the three-way intersection oracle.
        let fts_vec: HashSet<i128> = fts.intersection(&vector).copied().collect();
        let oracle: HashSet<i128> = fts_vec.intersection(&scalar).copied().collect();
        assert_eq!(
            combined, oracle,
            "combined query must equal fts ∩ vector ∩ scalar"
        );
        assert_eq!(combined.len(), 2, "intersection is exactly two docs");

        // 2. Each retriever is individually load-bearing: there is a doc
        //    dropped *only* because of it (kept by the other two). If any
        //    operator were silently a no-op, one of these would fail.
        let inter = |x: &HashSet<i128>, y: &HashSet<i128>| -> HashSet<i128> {
            x.intersection(y).copied().collect()
        };
        assert!(
            !inter(&vector, &scalar).is_subset(&fts),
            "FTS must be the sole reason ≥1 doc (kept by vector∧scalar) is dropped"
        );
        assert!(
            !inter(&fts, &scalar).is_subset(&vector),
            "vector cutoff must be the sole reason ≥1 doc (kept by fts∧scalar) is dropped"
        );
        assert!(
            !inter(&fts, &vector).is_subset(&scalar),
            "scalar WHERE must be the sole reason ≥1 doc (kept by fts∧vector) is dropped"
        );

        // 3. Every surviving row actually satisfies all three constraints.
        assert!(
            combined.is_subset(&vector),
            "every combined hit is within the vector top-k"
        );
        for b in &combined_batches {
            let cats = col_str(b, "category");
            let titles = col_str(b, "title");
            for i in 0..b.num_rows() {
                assert_eq!(cats.value(i), "systems", "scalar predicate holds on output");
                assert!(
                    titles.value(i).contains("rust"),
                    "FTS predicate holds on output row: {}",
                    titles.value(i)
                );
            }
        }

        // 4. ORDER BY b.score DESC honored (BM25 score: higher = better).
        let mut scores = Vec::new();
        for b in &combined_batches {
            let s = col_f32(b, "score");
            scores.extend((0..s.len()).map(|i| s.value(i)));
        }
        for w in scores.windows(2) {
            assert!(
                w[0] >= w[1],
                "combined scores must be descending: {scores:?}"
            );
        }
    }
}
