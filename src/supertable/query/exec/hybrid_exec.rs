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
//! before fusion; what the index could not bound is made up by
//! [`fill_top_k`] applying the exact predicate and fetching more. Without
//! a `WHERE`, the vector leg is `vector_search`'s own search, the hidden
//! vector index.

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
use futures::{future, stream};
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
                    PushedPredicate, arg_to_string, arg_to_usize, candidate_plan_for_filters,
                    fill_top_k, live_reader, output_schema_with_score, resolve_hits_named,
                    scope_to_call, top_k_statistics, traced_tvf,
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

/// Argument count for
/// `hybrid_search(text_col, q_text, vec_col, q_vec, k)`.
const HYBRID_SEARCH_ARG_COUNT: usize = 5;

/// RRF rank-bias constant. `60` is the value from the original
/// Cormack et al. reciprocal-rank-fusion paper and the de-facto
/// default across engines (Elasticsearch, Weaviate, …). Larger values
/// flatten the rank weighting; it never depends on the raw scores.
const RRF_K: f32 = 60.0;

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
        self.hybrid_search_scoped_async(text_col, q_text, mode, vec_col, q_vec, options, k, None)
            .await
    }

    /// [`Self::hybrid_search_async`] confined to a [`CandidateScope`] —
    /// what a SQL `WHERE` pushed into the `hybrid_search` table function
    /// admits. `None` is the unscoped search.
    ///
    /// Unscoped, the vector leg is the search `vector_search` runs: the
    /// hidden vector index plus the undrained user delta, with the index's
    /// calibrated width and rerank budget, so its cost follows k rather than
    /// the number of superfiles. A scope keeps the leg on the user table,
    /// fanned out over the scope's superfiles with its rows as the
    /// allow-set, so the fused top-k is the best `k` among rows that can
    /// satisfy the predicate; the BM25 leg reads the same scope. Fusion
    /// joins the legs on the stable `_id`, which every hit of both legs
    /// carries, so where a hit was found does not change its identity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn hybrid_search_scoped_async(
        &self,
        text_col: &str,
        q_text: &str,
        mode: BoolMode,
        vec_col: &str,
        q_vec: &[f32],
        options: VectorSearchOptions,
        k: usize,
        scope: Option<&CandidateScope>,
    ) -> Result<Vec<SuperfileHit>, QueryError> {
        // Hybrid's vector arm enters without the sync wrappers — apply the
        // cosine query calibration (#512) at this seam so hybrid scores
        // match the public vector path.
        let q_vec = calibrated_query(self, vec_col, q_vec);
        let vector_leg = async {
            let mut hits = match scope {
                None => {
                    self.vector_search_global_index_async(vec_col, &q_vec, k, options)
                        .await?
                }
                Some(scope) => {
                    // Only superfiles that still hold a candidate row; the
                    // kernel treats a superfile absent from the allow map as
                    // having no admitted row, so none is handed to it.
                    let superfiles: Vec<_> = scope.admitted_superfiles().cloned().collect();
                    if superfiles.is_empty() {
                        return Ok(Vec::new());
                    }
                    self.vector_fanout_over_superfiles(
                        superfiles,
                        vec_col,
                        &q_vec,
                        k,
                        options,
                        scope.allow.clone(),
                    )
                    .await?
                }
            };
            // Fusion joins the legs on `_id`, so every vector hit needs one.
            // Hidden-index hits, and hits in cell-packed user files (from
            // their inline id region), carry it from the search. A hit in an
            // older plain-IVF user file whose ids are not one contiguous run
            // can arrive without one (the scoped fan-out, or a table's search
            // before its first drain); its local is its Parquet row, so the
            // `_id` read there is its own. Only those hits are read.
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
                scope,
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
        if args.len() != HYBRID_SEARCH_ARG_COUNT {
            return Err(DataFusionError::Plan(format!(
                "hybrid_search expects {HYBRID_SEARCH_ARG_COUNT} arguments \
                 (text_col, q_text, vec_col, q_vec, k), got {}",
                args.len()
            )));
        }
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
    /// Correctness never depends on the pushdown.
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
            // `_id` and `score` come free with the fused hits (fusion
            // refuses a hit without its `_id`), so a projection of just
            // those needs no placement and no decode, as in
            // `vector_search`. Only without a `WHERE`: a predicate needs
            // the rows to check.
            if filters.is_empty()
                && let Some(indices) = id_score_projection
            {
                let fused = reader
                    .hybrid_search_scoped_async(
                        &text_col, &q_text, mode, &vec_col, &q_vec, options, k, None,
                    )
                    .await?;
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
            let predicate = PushedPredicate::compile(&filters, &output_schema);
            // Run both retrievers concurrently and fuse with RRF via the
            // shared kernel; the fused set is resolved once per round.
            let search = |want: usize| {
                let reader = &reader;
                let text_col = &text_col;
                let q_text = &q_text;
                let vec_col = &vec_col;
                let q_vec = &q_vec;
                let scope = scope.as_ref();
                async move {
                    let fused = reader
                        .hybrid_search_scoped_async(
                            text_col, q_text, mode, vec_col, q_vec, options, want, scope,
                        )
                        .await?;
                    // Cell-packed user hits can be boundary stubs whose IVF
                    // local does not address a Parquet row; remap those to
                    // their owning placement by stable id before the scalar
                    // decode — the same step the vector TVF takes.
                    user_placement_for_scalar_resolve(reader, &fused).await
                }
            };
            fill_top_k(
                &reader,
                k,
                predicate.as_ref(),
                &scalar_schema,
                &output_schema,
                projection.as_deref(),
                search,
            )
            .await
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
    use std::collections::HashSet;

    use arrow_array::{
        Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array,
        LargeStringArray, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::physical_plan::{displayable, joins::HashJoinExec};
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
            Supertable, SupertableOptions, manifest::SuperfileUri,
            query::exec::common::test_support::sql_row_count,
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
    /// overflowing DataFusion's join-size arithmetic.
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
        let mut flat = Vec::<f32>::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                flat.push(match d {
                    0 => 1.0,
                    1 => i as f32,
                    _ => 0.0,
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
        let batch = RecordBatch::try_new(
            st.options().schema.clone(),
            vec![Arc::new(tags), Arc::new(titles), Arc::new(fsl)],
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
        let tags_of = |batches: &[RecordBatch]| -> Vec<String> {
            batches
                .iter()
                .flat_map(|b| {
                    let t = col_str(b, "tag");
                    (0..t.len())
                        .map(|i| t.value(i).to_owned())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let unfiltered = st
            .reader()
            .expect("reader")
            .query_sql(&format!(
                "SELECT tag FROM hybrid_search('title', 'rust', 'emb', '{qv}', {k})"
            ))
            .expect("unfiltered query_sql");
        assert_eq!(
            tags_of(&unfiltered),
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
        assert_eq!(tags_of(&filtered), vec!["b", "b", "b"]);
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
        let n = count[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is int64")
            .value(0);
        assert_eq!(n, k as i64);
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
