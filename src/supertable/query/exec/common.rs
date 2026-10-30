// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared machinery for the search TVFs' custom `ExecutionPlan`s.
//!
//! All search TVFs (`vector_search`, `bm25_search`,
//! `bm25_search_prefix`, ...) produce a `Vec<SuperfileHit>` from a
//! kernel and then face the same two jobs:
//!
//!   1. **Resolve** each `(superfile, local_doc_id)` hit to the
//!      supertable's `_id` + projected scalar columns via
//!      [`SuperfileReader::take_by_local_doc_ids`], preserving the
//!      kernel's rank order, and append a `score` column.
//!   2. **Parse** the literal SQL arguments (`column`, `k`, ...).
//!
//! [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids

use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt,
    future::Future,
    ops::Range,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use arrow::compute::{concat_batches, filter_record_batch, interleave_record_batch, take};
use arrow_array::{
    ArrayRef, BooleanArray, Decimal128Array, Float32Array, RecordBatch, RecordBatchOptions,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use bytes::{Buf, Bytes, buf::Reader};
use datafusion::{
    catalog::{Session, TableProvider},
    common::{
        Column, DFSchema, TableReference,
        runtime::SpawnedTask,
        tree_node::{Transformed, TreeNode},
    },
    datasource::provider_as_source,
    error::{DataFusionError, Result as DfResult},
    execution::{TaskContext, context::ExecutionProps},
    logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableProviderFilterPushDown, TableType},
    physical_expr::{PhysicalExpr, create_physical_expr},
    physical_plan::{ExecutionPlan, collect},
    scalar::ScalarValue,
};
use futures::{FutureExt, StreamExt, TryStreamExt, future::BoxFuture, stream};
use object_store::{ObjectStore, path::Path as ObjPath};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection,
        },
        async_reader::{AsyncFileReader, ParquetObjectReader, ParquetRecordBatchStreamBuilder},
    },
    errors::{ParquetError, Result as ParquetResult},
    file::{
        metadata::ParquetMetaData,
        reader::{ChunkReader, Length},
    },
};
use rayon::{ThreadPool, prelude::*};
use roaring::RoaringBitmap;
use tokio::sync::Semaphore;
use tracing::{Instrument, Span, debug, warn};

use crate::{
    memory::{over_process_limit, process_limit, process_limit_exceeded, process_over_limit},
    runtime_bridge::{run_on_pool, run_on_pool_unless_blocking},
    runtime_metrics::op_stats::{OpStatsCollector, timed_section},
    runtime_metrics::rss::total_memory_bytes,
    superfile::{
        ReadError, SuperfileReader,
        lazy_source::Source,
        reader::{
            WHOLE_COLUMN_TAKE_DIVISOR, in_projection_order, rank_back_indices,
            row_selection_for_ids, view_read_options, whole_row_group,
        },
    },
    supertable::{
        error::QueryError,
        handle::{SupertableReader, WeakReader},
        manifest::{ManifestSnapshot, SuperfileEntry, SuperfileUri},
        query::{
            SuperfileHit,
            candidate::{CandidatePlan, CandidateScope},
            dispatch::{ReplanFuture, fanout_with_ordered, tombstone_deny_set},
            exec::metered_exec::MeteredExec,
            superfile_reader::superfile_reader,
            vector::row_id_from_manifest_entry,
        },
        schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE, PhysicalSchema, map::FileSchemaMap},
        tombstones::SidecarCache,
    },
    utils::{join::try_join_bounded, trace::record},
};

/// A table function's search, run under `span` with the batch's row count
/// recorded as its `rows_out`. `execute` makes the span under `sql.execute`,
/// so the search kernel's phase spans nest beneath it.
pub(crate) fn traced_tvf<F>(span: Span, search: F) -> impl Future<Output = F::Output>
where
    F: Future<Output = DfResult<RecordBatch>>,
{
    traced_tvf_rows(span, search, RecordBatch::num_rows)
}

/// [`traced_tvf`] for a search whose output counts its rows through `rows`:
/// one that comes out as several batches, one per superfile, records their
/// sum.
pub(crate) fn traced_tvf_rows<T, F>(
    span: Span,
    search: F,
    rows: fn(&T) -> usize,
) -> impl Future<Output = F::Output>
where
    F: Future<Output = DfResult<T>>,
{
    async move {
        let out = search.await;
        if let Ok(output) = &out {
            record("rows_out", rows(output));
        }
        out
    }
    .instrument(span)
}

/// Catalog the per-call scans of [`scope_to_call`] are named under. No
/// user table lives in it, so a call's scan can never share a name — and
/// with it plan equality — with a scan of a real table.
const CALL_SCOPE_CATALOG: &str = "__infino_internal";

/// Schema, inside [`CALL_SCOPE_CATALOG`], of the per-call scans.
const CALL_SCOPE_SCHEMA: &str = "search_tvf_call";

/// Numbers the search table-function calls, so each call's scan gets a
/// name no other call shares (see [`scope_to_call`]).
static SEARCH_CALL_SEQ: AtomicU64 = AtomicU64::new(0);

/// Multiply a search's `k` by this each time the exact predicate leaves
/// fewer than `k` rows standing (see [`fill_top_k`]). Doubling bounds the
/// total work at about twice the final round's, whatever the predicate's
/// selectivity turns out to be.
const OVER_FETCH_GROWTH: usize = 2;

/// How many times `k` a fill may grow to before it gives up and returns
/// what it has (see [`over_fetch_ceiling`]). Without a ceiling the growth
/// runs to the table's row count, so a predicate the index cannot bound
/// and that almost nothing satisfies ends by asking the kernel for a heap
/// the size of the table and resolving every match into one batch — an
/// out-of-memory risk on a query that is cheap to state.
const OVER_FETCH_MAX_K_MULTIPLE: usize = 64;

/// The smallest ceiling a fill will use, whatever `k` is. A multiple
/// alone would give a small `k` a tiny budget — 64 hits at `k = 1` — and
/// a selective predicate needs to look past that before concluding its
/// rows are not there.
pub(crate) const OVER_FETCH_MIN_CEILING: usize = 10_000;

/// The absolute ceiling on one fill's hit count, bounding the rows any
/// single round resolves into memory. A caller asking for more than this
/// still gets its `k`: [`over_fetch_ceiling`] never returns less.
const OVER_FETCH_MAX_HITS: usize = 100_000;

/// Rows of one superfile [`exact_scope`] decodes and checks at a time, when
/// the superfile is too large to take its rows in one read: past
/// `EXACT_SCOPE_TAKE_ROWS` × [`WHOLE_COLUMN_TAKE_DIVISOR`] rows a chunk this
/// size is a sparse take, which decodes only the pages its rows land on, so
/// the predicate's columns over a large superfile never sit in memory whole.
const EXACT_SCOPE_TAKE_ROWS: usize = 1 << 20;

/// Rows of the first chunk [`exact_scope`] reads of a variable-width column,
/// before it knows how many bytes a row decodes to.
const EXACT_SCOPE_FIRST_TAKE_ROWS: usize = 1 << 16;

/// Decoded bytes a chunk of a variable-width column is sized to, from the
/// bytes per row of the chunk before: a string column's chunk holds about
/// this much, however long its values run, where a fixed count of rows of
/// long text could hold gigabytes.
const EXACT_SCOPE_TAKE_BYTES: usize = 32 << 20;

/// The most hits one fill will ask a kernel for: [`OVER_FETCH_MAX_K_MULTIPLE`]
/// times `k`, floored at [`OVER_FETCH_MIN_CEILING`], capped at
/// [`OVER_FETCH_MAX_HITS`], never below the `k` actually asked for, and
/// never above the table's row count.
pub(crate) fn over_fetch_ceiling(k: usize, total: usize) -> usize {
    k.saturating_mul(OVER_FETCH_MAX_K_MULTIPLE)
        .clamp(OVER_FETCH_MIN_CEILING, OVER_FETCH_MAX_HITS)
        .max(k)
        .min(total)
}

/// The reader a search table function captured, still alive when the query
/// runs. The function holds it weakly; finding it gone means the table handle
/// was dropped under a running plan, which is ours, not the query's.
pub(crate) fn live_reader(reader: &WeakReader, function: &str) -> DfResult<Arc<SupertableReader>> {
    reader.upgrade().ok_or_else(|| {
        QueryError::Internal(format!(
            "{function}: supertable consumer dropped before execution"
        ))
        .into()
    })
}

/// Give one search table-function call a scan name of its own, so
/// DataFusion never merges two calls.
///
/// - DataFusion names every table-function scan after the function alone
///   (`token_match()`) and compares scans without their arguments, so two
///   calls with different arguments looked equal and both returned the
///   first call's rows.
/// - The returned provider swaps in a scan of `provider` named
///   `__infino_internal.search_tvf_call.token_match#7`, unique per call,
///   under the function's own alias, so column names don't change.
/// - Identical calls are not merged either; they now run twice.
pub(crate) fn scope_to_call(
    function: &str,
    provider: Arc<dyn TableProvider>,
) -> DfResult<Arc<dyn TableProvider>> {
    let seq = SEARCH_CALL_SEQ.fetch_add(1, Ordering::Relaxed);
    let plan = LogicalPlanBuilder::scan(
        TableReference::full(
            CALL_SCOPE_CATALOG,
            CALL_SCOPE_SCHEMA,
            format!("{function}#{seq}"),
        ),
        provider_as_source(Arc::clone(&provider)),
        None,
    )?
    .build()?;
    Ok(Arc::new(CallScopedTable {
        inner: provider,
        plan,
    }))
}

/// One search table-function call with a scan of its own; built by
/// [`scope_to_call`]. Everything but the logical plan is the wrapped
/// call's, so a planner that scans this provider directly instead of
/// inlining its plan still runs the same search.
struct CallScopedTable {
    inner: Arc<dyn TableProvider>,
    plan: LogicalPlan,
}

impl fmt::Debug for CallScopedTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.inner, f)
    }
}

#[async_trait]
impl TableProvider for CallScopedTable {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn get_logical_plan(&self) -> Option<Cow<'_, LogicalPlan>> {
        Some(Cow::Borrowed(&self.plan))
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        self.inner.scan(state, projection, filters, limit).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DfResult<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
}

/// Meter, execute and account for one physical plan: wrap the root in
/// [`MeteredExec`], collect, then harvest the executed plan's own
/// metrics. The three steps belong together — a site that collects
/// without harvesting reports CPU, page bytes and ranges with no row
/// count beside them, which is how the mutation predicate resolve once
/// reported zero decoded rows for a scan it genuinely paid for. Every
/// SQL execution site (catalog, reader-level, predicate resolve) goes
/// through here; error mapping stays with the caller, whose surface it
/// belongs to.
///
/// When the process has a memory limit (see `memory::resident`), the collect
/// races it, and a process past the limit ends the statement with a
/// `ResourcesExhausted` refusal.
pub(crate) async fn collect_plan_metered(
    plan: &Arc<dyn ExecutionPlan>,
    task_ctx: Arc<TaskContext>,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> DfResult<Vec<RecordBatch>> {
    let metered: Arc<dyn ExecutionPlan> =
        Arc::new(MeteredExec::new(Arc::clone(plan), op_stats.clone()));
    let collected = collect(metered, task_ctx);
    let batches = match process_limit() {
        None => collected.await?,
        // With a process limit, run the collect as its own task and race it
        // against the sampler. The streamed batches are never reserved, so
        // this is the only bound on them.
        //  - over the limit before we start: refuse without spawning anything.
        //  - own task, because a plan over cached data can finish all its work
        //    in one poll, so a check in the same task would only run once the
        //    result is ready.
        //  - limit wins: dropping the task stops DataFusion's partition tasks
        //    and the search functions' fan-out. A batch already being
        //    processed still finishes first.
        //  - `biased` checks the limit first, so a reading over it wins even if
        //    the result lands in the same wakeup.
        Some(limit) => {
            if let Some(anon) = process_over_limit(limit) {
                return Err(over_process_limit(anon, limit));
            }
            let collecting = SpawnedTask::spawn(collected);
            tokio::select! {
                biased;
                anon = process_limit_exceeded(limit) => {
                    return Err(over_process_limit(anon, limit));
                }
                joined = collecting => {
                    joined.map_err(|join| DataFusionError::ExecutionJoin(Box::new(join)))??
                }
            }
        }
    };
    // Harvesting the unwrapped plan and the wrapped one reach the same
    // leaves — the walk dedupes by node identity and sums childless nodes
    // only, and the meter delegates `execute` to this same tree.
    harvest_datafusion_metrics(plan, op_stats);
    Ok(batches)
}

/// Fold DataFusion's row instrumentation into the per-query stats after a
/// SQL plan has executed: leaf (source) operators' `output_rows` sum into
/// `rows_materialized` — the rows the scans decoded from storage.
///
/// Deliberately NOT `elapsed_compute`. That is an `Instant` timer around
/// synchronous poll sections — wall time, so on a busy host it counts
/// whatever the thread was descheduled for, the exact contention bleed
/// per-op metering exists to remove. It also omits Parquet decode
/// entirely. CPU now comes from
/// [`MeteredExec`](crate::supertable::query::exec::metered_exec::MeteredExec),
/// which brackets each scan poll with the same thread-CPU clock the
/// search kernels use, so SQL and search are priced on one clock.
pub(crate) fn harvest_datafusion_metrics(
    plan: &Arc<dyn ExecutionPlan>,
    op_stats: &Option<Arc<OpStatsCollector>>,
) {
    // Deduped by node identity: plans are trees in practice, but nothing
    // forbids an operator `Arc` appearing under two parents, and a shared
    // node's metrics must not be added once per path.
    fn walk(node: &Arc<dyn ExecutionPlan>, seen: &mut HashSet<usize>, leaf_rows: &mut u64) {
        if !seen.insert(Arc::as_ptr(node) as *const () as usize) {
            return;
        }
        if let Some(metrics) = node.metrics()
            && node.children().is_empty()
            && let Some(rows) = metrics.output_rows()
        {
            *leaf_rows += rows as u64;
        }
        for child in node.children() {
            walk(child, seen, leaf_rows);
        }
    }
    let Some(stats) = op_stats else {
        return;
    };
    let mut seen = HashSet::new();
    let mut leaf_rows = 0u64;
    walk(plan, &mut seen, &mut leaf_rows);
    stats.add_rows_materialized(leaf_rows);
}

/// Resolve `hits` to one `RecordBatch`, with `projection` naming the
/// output columns (any of `_id`, the visible scalar columns, or the
/// trailing `score`); `None` returns the engine-native `_id` + `score`
/// pair. Names are resolved to output-schema indices and forwarded to
/// [`resolve_hits`], which decodes only the projected columns. Shared
/// by every public row-returning search method (`bm25_search`,
/// `vector_search`, `token_match`, `exact_match`); `what` labels error
/// messages with the calling method.
pub(crate) async fn resolve_hits_named(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    projection: Option<&[&str]>,
) -> Result<RecordBatch, QueryError> {
    // Stored shape: an index-only FTS column has no Parquet data to
    // project, so its name is rejected up front like any unknown column.
    let scalar_schema = reader.manifest().stored_schema();
    let output_schema = output_schema_with_score(&scalar_schema);
    // `None` is the engine-native result: `_id` + `score` only.
    // `_id` decodes from its own dedicated id pages (cheap by
    // design) and `score` is synthesized from the hits, so the
    // bare call never touches user-column data pages — projecting
    // those is an explicit opt-in by name.
    let id_column = reader.options().id_column.clone();
    // Resolving a projected name to a column is caller-input validation, not
    // query execution: the single-table search methods run their own kernels
    // (no SQL engine involved), so an unknown column must read as a bad request
    // that names the column and the valid set — never as an internal execution
    // failure. The same check runs early (before any search I/O) at the search
    // entry points via [`validate_projection`]; this call is the one source of
    // truth so the two never drift.
    let indices = validate_projection(projection, id_column.as_str(), &output_schema)?;
    // Past name resolution, a failure is ours reading the projected columns;
    // keep it typed so a storage fault or budget refusal keeps its class.
    resolve_hits(reader, hits, &scalar_schema, &output_schema, Some(&indices))
        .await
        .map_err(QueryError::DataFusion)
}

/// Resolve each projected column name to its index in the search output
/// schema, erroring with [`unknown_column_message`] on the first name the
/// output does not carry. `projection` of `None` is the engine-native
/// `_id` + `score` result, whose names are always resolvable.
///
/// This is the single source of truth for projection-name validation. It
/// runs at output-materialization time inside [`resolve_hits_named`], and —
/// for the row-returning search entry points — early, before any search or
/// placement I/O, so a projection naming a nonexistent column fails fast
/// instead of after a full (at billion scale, multi-gigabyte) search whose
/// result is then thrown away. `output_schema` is resident
/// ([`output_schema_with_score`] over the stored scalar schema), so the
/// early check touches no object storage.
pub(crate) fn validate_projection(
    projection: Option<&[&str]>,
    id_column: &str,
    output_schema: &SchemaRef,
) -> Result<Vec<usize>, QueryError> {
    // `None` decodes `_id` + `score` only; both are always resolvable.
    let bare: [&str; 2] = [id_column, SCORE_COLUMN];
    let names: &[&str] = match projection {
        Some(names) => names,
        None => &bare,
    };
    names
        .iter()
        .map(|name| {
            output_schema
                .index_of(name)
                .map_err(|_| QueryError::InvalidQuery(unknown_column_message(name, output_schema)))
        })
        .collect()
}

/// Message for a projection naming a column the search output does not
/// carry. Lists the selectable columns (`_id`, the visible scalar columns,
/// and the trailing `score`) so the caller can correct the request.
fn unknown_column_message(name: &str, output_schema: &SchemaRef) -> String {
    let available = output_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!("unknown column {name:?} in projection; valid columns: {available}")
}

impl SupertableReader {
    /// Reject a projection naming a column the search output does not carry,
    /// before any search or placement I/O. Resident-only: the stored scalar
    /// schema plus the synthesized `score`, no object storage touched. Every
    /// row-returning search entry point calls this at its top so a doomed
    /// projection fails fast instead of after a full (at billion scale,
    /// multi-gigabyte) search whose result is then thrown away. The valid set
    /// is the same one [`resolve_hits_named`] checks at output materialization,
    /// so the two share one source of truth and the error message is identical.
    pub(crate) fn check_projection(&self, projection: Option<&[&str]>) -> Result<(), QueryError> {
        let output_schema = output_schema_with_score(&self.manifest().stored_schema());
        validate_projection(
            projection,
            self.options().id_column.as_str(),
            &output_schema,
        )?;
        Ok(())
    }
}

/// Lower a search table function's pushed-down `WHERE` filters to the
/// candidate plan its kernel ranks within: a token-match superset over
/// the FTS-indexed columns, [`Unbounded`](CandidatePlan::Unbounded) when
/// no filter is index-resolvable. One lowering for all three search
/// functions, so a predicate bounds `vector_search`, `bm25_search` and
/// `hybrid_search` identically.
pub(crate) fn candidate_plan_for_filters(
    manifest: &ManifestSnapshot,
    filters: &[Expr],
) -> CandidatePlan {
    let fts_cols: HashSet<&str> = manifest
        .options
        .fts_columns
        .iter()
        .map(|c| c.column.as_str())
        .collect();
    CandidatePlan::from_filters(filters, &fts_cols, &|col| {
        manifest.try_fts_tokenizer_for(col)
    })
}

/// The exact `WHERE` a search table function was handed, ready to run
/// over the function's own resolved rows.
///
/// What a kernel ranks within is a superset of the predicate: a token
/// match stands in for a string equality, and a column with no full-text
/// index bounds nothing at all. DataFusion's `FilterExec` above the
/// function re-applies the exact predicate, so a function that returned
/// exactly `k` rows can come out under `k` — silently, since the function
/// never sees what the filter dropped. This is the same predicate compiled
/// for the function to apply itself, so it can count the survivors and
/// fetch more until `k` stand (see [`fill_top_k`]).
pub(crate) struct PushedPredicate {
    /// Indices into the function's output schema of the columns the
    /// predicate reads.
    columns: Vec<usize>,
    /// The conjunction of the pushed filters, column references stripped
    /// of their table qualifier so they resolve against the function's
    /// own (unqualified) output schema.
    conjunction: Expr,
}

impl PushedPredicate {
    /// `None` when nothing was pushed, or when the predicate reads a
    /// column the function's output does not carry — nothing here could
    /// evaluate it, and the `FilterExec` above still applies it, exactly
    /// as before the pushdown existed.
    pub(crate) fn compile(filters: &[Expr], output_schema: &SchemaRef) -> Option<Self> {
        let conjunction = unqualified(filters.iter().cloned().reduce(Expr::and)?).ok()?;
        let mut columns = Vec::new();
        for column in conjunction.column_refs() {
            let index = output_schema.index_of(&column.name).ok()?;
            if !columns.contains(&index) {
                columns.push(index);
            }
        }
        Some(Self {
            columns,
            conjunction,
        })
    }

    /// Indices into the output schema of the columns the predicate reads,
    /// in first-seen order.
    pub(crate) fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Bind the predicate to `schema`, a projection of the output schema
    /// that carries every column in `self.columns`.
    pub(crate) fn bind(&self, schema: &SchemaRef) -> DfResult<BoundPredicate> {
        let df_schema = DFSchema::try_from(Arc::clone(schema))?;
        create_physical_expr(&self.conjunction, &df_schema, &ExecutionProps::new())
            .map(BoundPredicate)
    }
}

/// A [`PushedPredicate`] bound to the schema of the batches it checks.
pub(crate) struct BoundPredicate(Arc<dyn PhysicalExpr>);

impl BoundPredicate {
    /// One boolean per row of `batch`: whether the predicate holds there.
    /// A null result is kept as null; like SQL's `WHERE`, every caller
    /// keeps only the rows that are `true`.
    pub(crate) fn mask(&self, batch: &RecordBatch) -> DfResult<BooleanArray> {
        let mask = self.0.evaluate(batch)?.into_array(batch.num_rows())?;
        mask.as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "a pushed-down predicate did not evaluate to a boolean".into(),
                )
            })
    }
}

/// `expr` with every column reference stripped of its table qualifier.
/// DataFusion hands a table function its filters as the planner wrote
/// them — `t.category = 'x'` under `FROM bm25_search(...) AS t` — while
/// the function's output schema names bare columns.
fn unqualified(expr: Expr) -> DfResult<Expr> {
    expr.transform(|e| {
        Ok(match e {
            Expr::Column(c) if c.relation.is_some() => {
                Transformed::yes(Expr::Column(Column::new_unqualified(c.name)))
            }
            other => Transformed::no(other),
        })
    })
    .map(|t| t.data)
}

/// Resolve a search table function's top-`k` under its pushed-down
/// `WHERE`, so the rows it emits are the `k` best *among rows satisfying
/// the predicate*.
///
/// `search(want)` runs the kernel for `want` hits, already scoped to
/// whatever the index could bound. Without a predicate that is the whole
/// job: search `k`, resolve, done. With one, the resolved rows are
/// checked against the exact predicate; when fewer than `k` survive and
/// the kernel had more to give, the search is rerun for
/// [`OVER_FETCH_GROWTH`] times as many hits, until `k` survive, the
/// kernel runs dry, or the fill hits its ceiling. The survivors, in
/// kernel rank order and cut to `k`, are the result — the `FilterExec`
/// above re-checks them and drops nothing. Cost scales with the
/// predicate's selectivity: an exact pushdown pays one round, a bare
/// scalar predicate on a row in a million pays the log₂ of that.
///
/// **The fill is bounded, so it may return fewer than `k` rows even
/// though more satisfy the predicate.** Growth stops at
/// [`over_fetch_ceiling`]; past that the survivors so
/// far are the answer. The alternative is unbounded: a predicate no index
/// can bound and almost no row satisfies would otherwise grow `want` to
/// the table's row count and resolve every match into a single batch.
/// Returning short is the deliberate trade, and it is still strictly more
/// than the caller used to get — before the pushdown, `k` was applied
/// table-wide and the `WHERE` ran afterwards, so the same query returned
/// only whatever survived from one unfiltered top-`k`.
///
/// **The ceiling is the only stopping condition besides finding `k`, and
/// that is deliberate: the hit count carries no exhaustion signal.** A
/// round can come back short, or no longer than the round before it,
/// without the kernel having run out — tombstones are subtracted after
/// each unit's `k`-sized heap (`dispatch::fanout_local_hits`), and fusion
/// and stable-id de-duplication shrink a round too. Reading either as
/// exhaustion returns early while matching rows remain, which is the
/// underflow this function exists to prevent, reached through the delete
/// path instead of the ranking one. So the fill keeps widening until `k`
/// survive or the ceiling stops it, and pays extra rounds on a query that
/// was going to come up short anyway. The signal that would end those
/// rounds honestly is the kernel reporting its own saturation, before
/// tombstones and fusion touch the hits; that is not plumbed today.
///
/// The predicate's columns are decoded alongside the requested ones for
/// the check and projected away before the batch is returned.
pub(crate) async fn fill_top_k<F, Fut>(
    reader: &SupertableReader,
    k: usize,
    predicate: Option<&PushedPredicate>,
    scalar_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection: Option<&[usize]>,
    search: F,
) -> DfResult<RecordBatch>
where
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<Vec<SuperfileHit>, QueryError>>,
{
    let total = usize::try_from(reader.manifest().n_docs_total()).unwrap_or(usize::MAX);
    fill_top_k_within(
        reader,
        k,
        predicate,
        scalar_schema,
        output_schema,
        projection,
        over_fetch_ceiling(k, total),
        search,
    )
    .await
}

/// [`fill_top_k`] with the caller's own `ceiling` on the hits a round may
/// ask for. A ceiling of `k` runs exactly one round: the kernel's `k` best
/// within its scope, checked against the predicate and returned however
/// many survive — what a caller wants when it has a better move than
/// widening, or when its scope already admits only rows the predicate keeps
/// (see [`exact_scope`]).
#[allow(clippy::too_many_arguments)]
pub(crate) async fn fill_top_k_within<F, Fut>(
    reader: &SupertableReader,
    k: usize,
    predicate: Option<&PushedPredicate>,
    scalar_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection: Option<&[usize]>,
    ceiling: usize,
    search: F,
) -> DfResult<RecordBatch>
where
    F: Fn(usize) -> Fut,
    Fut: Future<Output = Result<Vec<SuperfileHit>, QueryError>>,
{
    let Some(predicate) = predicate else {
        let hits = search(k).await?;
        return resolve_hits(reader, &hits, scalar_schema, output_schema, projection).await;
    };
    let requested: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..output_schema.fields().len()).collect(),
    };
    // Decode the requested columns first, in order, then whatever else the
    // predicate reads — so the requested projection is a prefix of the
    // decoded batch and comes back out as its first columns.
    let mut decoded = requested.clone();
    for &column in &predicate.columns {
        if !decoded.contains(&column) {
            decoded.push(column);
        }
    }
    let decoded_schema = Arc::new(
        output_schema
            .project(&decoded)
            .map_err(QueryError::internal)?,
    );
    let bound = predicate.bind(&decoded_schema)?;
    let requested_positions: Vec<usize> = (0..requested.len()).collect();
    let mut want = k.min(ceiling);
    loop {
        let hits = search(want).await?;
        let batch =
            resolve_hits(reader, &hits, scalar_schema, output_schema, Some(&decoded)).await?;
        let mask = bound.mask(&batch)?;
        let kept = filter_record_batch(&batch, &mask).map_err(QueryError::internal)?;
        // The ceiling is the only stopping condition other than finding `k`.
        // Nothing observable here distinguishes a kernel that is out of
        // candidates from one whose hits were thinned after it ran: tombstones
        // are subtracted after each unit's heap, and fusion and stable-id
        // de-duplication also shrink a round. So neither a short round nor a
        // round that returned no more than the last one is evidence, and
        // treating either as exhaustion under-returns.
        let exhausted = want >= ceiling;
        if kept.num_rows() >= k || exhausted {
            let rows = kept.num_rows().min(k);
            return kept
                .slice(0, rows)
                .project(&requested_positions)
                .map_err(|e| QueryError::internal(e).into());
        }
        want = want.saturating_mul(OVER_FETCH_GROWTH).min(ceiling);
    }
}

/// `scope` narrowed to exactly the rows its pushed-down `predicate` keeps:
/// each admitted superfile's candidate rows, decoded on the predicate's
/// columns and kept where it holds. A bounded scope's sets are read as they
/// are, already free of tombstones; an unbounded one stands for every row
/// of each superfile, less its tombstones. A kernel gated on the result
/// admits only rows the `WHERE` keeps, so one round at `k` returns the `k`
/// best of them, or every one when fewer exist: no widening, and no
/// ceiling to come up short at.
///
/// This is how a predicate the index cannot bound — a scalar equality, a
/// range — still means the top `k` among the rows satisfying it. The cost
/// is a decode of the predicate's columns over the scope's rows, so a
/// caller narrows the scope to the rows its search can return first (a
/// text search, to the rows its terms match) and asks for this only once
/// widening has stopped short at its ceiling. With no index to narrow it,
/// the scope is every row the statistics keep, often the whole table.
///
/// What it holds at once is bounded, whatever the table's size:
/// - fixed-width columns over a superfile small enough that its rows are a
///   dense take ([`WHOLE_COLUMN_TAKE_DIVISOR`]) are read in one take, which
///   reads each column once; a larger one in sparse chunks of
///   [`EXACT_SCOPE_TAKE_ROWS`];
/// - variable-width columns in chunks of about [`EXACT_SCOPE_TAKE_BYTES`]
///   decoded;
/// - each chunk is reserved against the connection's memory budget while it
///   is checked, and a budget it doesn't fit refuses the query;
/// - superfiles are checked as many at a time as the reader pool has
///   threads, each holding one chunk.
///
/// `None` when the predicate reads no column or one no superfile stores —
/// the score, the id column, an index-only text column — so it cannot be
/// evaluated before ranking; the caller then keeps what widening found.
pub(crate) async fn exact_scope(
    reader: &SupertableReader,
    scope: &CandidateScope,
    predicate: &PushedPredicate,
    output_schema: &SchemaRef,
) -> DfResult<Option<CandidateScope>> {
    let manifest = Arc::clone(reader.manifest());
    let stored = manifest.stored_schema();
    let id_column = reader.options().id_column.as_str();
    let mut names = Vec::with_capacity(predicate.columns().len());
    for &index in predicate.columns() {
        let name = output_schema.field(index).name();
        if name == SCORE_COLUMN || name == id_column || stored.field_with_name(name).is_err() {
            return Ok(None);
        }
        names.push(name.clone());
    }
    if names.is_empty() {
        return Ok(None);
    }
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let out_schema = hit_output_schema(&stored, &refs)?;
    let fixed_width = out_schema
        .fields()
        .iter()
        .all(|field| field.data_type().primitive_width().is_some());
    let bound = Arc::new(predicate.bind(&out_schema)?);
    let names = Arc::new(names);
    let op_stats = reader.op_stats.clone();
    let budget = Arc::clone(&manifest.options.connection_memory_budget);
    // Superfiles are checked this many at a time, each holding one chunk:
    // the most the pass holds at once, whatever the table's size.
    let window = manifest.options.reader_pool.current_num_threads().max(1);
    // A bounded scope hands each superfile its set, tombstones already
    // subtracted; an unbounded one hands `None`, read as all of its rows.
    let units: Vec<(Arc<SuperfileEntry>, Option<Arc<RoaringBitmap>>)> = scope
        .admitted_superfiles()
        .map(|entry| {
            let rows = scope
                .allow
                .as_ref()
                .and_then(|allow| allow.get(&entry.uri).cloned());
            (Arc::clone(entry), rows)
        })
        .collect();
    let body = move |r: Arc<SuperfileReader>,
                     entry: Arc<SuperfileEntry>,
                     cache: Option<Arc<SidecarCache>>,
                     now: Instant,
                     rows: Option<Arc<RoaringBitmap>>| {
        let manifest = Arc::clone(&manifest);
        let names = Arc::clone(&names);
        let out_schema = Arc::clone(&out_schema);
        let bound = Arc::clone(&bound);
        let op_stats = op_stats.clone();
        let budget = Arc::clone(&budget);
        async move {
            let candidates = match rows {
                Some(rows) => rows,
                None => {
                    let n_docs = u32::try_from(r.n_docs()).map_err(QueryError::internal)?;
                    let mut all = RoaringBitmap::new();
                    all.insert_range(0..n_docs);
                    if let Some(cache) = cache.as_ref()
                        && let Some(deny) = tombstone_deny_set(cache, entry.superfile_id, now)?
                    {
                        all -= deny.as_ref();
                    }
                    Arc::new(all)
                }
            };
            // Fixed-width columns over a superfile a chunk could cover are
            // read in one take, which reads each column once; past that, in
            // chunks small enough to stay sparse takes. Variable-width
            // columns are read in chunks sized by their decoded bytes.
            let n_docs = usize::try_from(r.n_docs()).map_err(QueryError::internal)?;
            let whole_file_rows =
                EXACT_SCOPE_TAKE_ROWS.saturating_mul(WHOLE_COLUMN_TAKE_DIVISOR as usize);
            let mut take_rows = match (fixed_width, n_docs <= whole_file_rows) {
                (true, true) => n_docs.max(1),
                (true, false) => EXACT_SCOPE_TAKE_ROWS,
                (false, _) => EXACT_SCOPE_FIRST_TAKE_ROWS,
            };
            let refs: Vec<&str> = names.iter().map(String::as_str).collect();
            let mut kept = RoaringBitmap::new();
            let mut remaining = candidates.iter();
            loop {
                let chunk: Vec<u32> = remaining.by_ref().take(take_rows).collect();
                if chunk.is_empty() {
                    break;
                }
                if let Some(stats) = &op_stats {
                    stats.add_rows_materialized(chunk.len() as u64);
                }
                let batch =
                    take_file_columns(&manifest, &r, chunk.clone(), &refs, &out_schema, &op_stats)
                        .await
                        .map_err(QueryError::DataFusion)?;
                // The chunk's columns count against the connection's budget
                // while they are checked; a budget they don't fit refuses
                // the query, as a SQL operator that can't spill is refused.
                let bytes = batch.get_array_memory_size();
                let _held = budget.try_reserve(bytes).map_err(|over| {
                    QueryError::OverBudget(format!("during a WHERE's exact pass, {over}"))
                })?;
                if !fixed_width {
                    let per_row = (bytes / chunk.len()).max(1);
                    take_rows = (EXACT_SCOPE_TAKE_BYTES / per_row)
                        .clamp(EXACT_SCOPE_FIRST_TAKE_ROWS, EXACT_SCOPE_TAKE_ROWS);
                }
                let bound = Arc::clone(&bound);
                // The predicate runs over up to a chunk of rows: compute, so
                // the reader pool takes it rather than the tokio worker. It
                // hands back the positions it holds at; the chunk maps them
                // to rows here.
                let (held, ns) = run_on_pool(
                    Some(&manifest.options.reader_pool),
                    "exact scope check: reader pool dropped result",
                    move || {
                        timed_section(|| {
                            bound.mask(&batch).map(|mask| {
                                mask.iter()
                                    .enumerate()
                                    .filter_map(|(at, holds)| (holds == Some(true)).then_some(at))
                                    .collect::<Vec<usize>>()
                            })
                        })
                    },
                )
                .await
                .map_err(QueryError::internal)?;
                if let Some(stats) = &op_stats {
                    stats.add_kernel_cpu_ns(ns);
                }
                kept.extend(
                    held.map_err(QueryError::DataFusion)?
                        .into_iter()
                        .map(|at| chunk[at]),
                );
            }
            Ok((entry.uri, kept))
        }
    };
    let kept = fanout_with_ordered(
        reader,
        units,
        window,
        |_| false,
        body,
        |_| {},
        None::<ReplanFuture<'_, _, _>>,
    )
    .await?;
    let allow = kept
        .into_iter()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(uri, rows)| (uri, Arc::new(rows)))
        .collect();
    Ok(Some(CandidateScope {
        superfiles: scope.superfiles.clone(),
        allow: Some(allow),
    }))
}

/// Output column carrying the per-hit score (vector distance or BM25
/// relevance — direction is the originating TVF's contract).
pub(crate) const SCORE_COLUMN: &str = "score";

/// Search-TVF output schema: the scalar schema with a trailing
/// non-null `score: Float32` appended.
pub(crate) fn output_schema_with_score(scalar_schema: &SchemaRef) -> SchemaRef {
    let mut fields: Vec<Field> = scalar_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(SCORE_COLUMN, DataType::Float32, false));
    Arc::new(Schema::new(fields))
}

/// Resolve `hits` (in kernel rank order) to a `RecordBatch` matching
/// `output_schema` projected by `projection`, preserving rank order.
///
/// `output_schema` is the scalar schema with a trailing `score`
/// column ([`output_schema_with_score`]); `projection` indexes into
/// it, exactly as DataFusion hands to `scan`. **Only the scalar
/// columns the projection actually selects are decoded** — a query
/// that selects just `score` opens no superfile readers and touches no
/// scalar bytes (cost-first: never decode a column the query did not
/// select). The `score` column is synthesized from the hits.
///
/// Selected scalar columns are read per superfile (each
/// `take_by_local_doc_ids` is a column-projected read), concatenated,
/// then a single `take` reorders rows back into the global rank order
/// so row `i` is the `i`-th hit.
pub(crate) async fn resolve_hits(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    scalar_schema: &SchemaRef,
    output_schema: &SchemaRef,
    projection: Option<&[usize]>,
) -> DfResult<RecordBatch> {
    let projected_schema = match projection {
        Some(indices) => Arc::new(
            output_schema
                .project(indices)
                .map_err(QueryError::internal)?,
        ),
        None => Arc::clone(output_schema),
    };
    if hits.is_empty() {
        return Ok(RecordBatch::new_empty(projected_schema));
    }

    // `score` is the trailing column of `output_schema`; every
    // smaller index is a scalar column.
    let score_idx = scalar_schema.fields().len();
    let requested: Vec<usize> = match projection {
        Some(indices) => indices.to_vec(),
        None => (0..output_schema.fields().len()).collect(),
    };

    // Distinct scalar columns the projection selects, in first-seen
    // order — the only columns we decode.
    let mut needed: Vec<&str> = Vec::new();
    for &p in &requested {
        if p != score_idx {
            let name = scalar_schema.field(p).name().as_str();
            if !needed.contains(&name) {
                needed.push(name);
            }
        }
    }

    let id_column = reader.options().id_column.as_str();
    let resolved = if needed.is_empty() {
        None
    } else if needed == [id_column] {
        // Hit → `_id` translation without touching the file: prefer
        // `hit.stable_id` (inline on MultiCell / hidden IVF), else
        // contiguous-span arithmetic. Falls back to the id-page read
        // only when a hit has neither.
        match resolve_ids_arithmetic(reader, hits) {
            Some(batch) => Some(batch?),
            None => Some(resolve_columns(reader, hits, &needed).await?),
        }
    } else if needed.contains(&id_column) {
        // `_id` + scalars: never Parquet-decode `_id` for identity.
        // MultiCell / hidden hits already carry stable `_id` on the hit;
        // id-ordered files use span arithmetic. Only the non-id columns
        // come from `take_by_local_doc_ids`.
        let other: Vec<&str> = needed
            .iter()
            .copied()
            .filter(|name| *name != id_column)
            .collect();
        let other_batch = resolve_columns(reader, hits, &other).await?;
        let id_batch = resolve_ids_arithmetic(reader, hits).ok_or_else(|| {
            QueryError::Internal(
                "resolve_hits: hit set missing stable _id and span arithmetic \
                 (cell-packed hits must carry stable_id from the search wave)"
                    .into(),
            )
        })??;
        let mut fields = vec![id_batch.schema().field(0).as_ref().clone()];
        fields.extend(
            other_batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone()),
        );
        let mut columns = vec![Arc::clone(id_batch.column(0))];
        columns.extend(other_batch.columns().iter().map(Arc::clone));
        Some(
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
                .map_err(QueryError::internal)?,
        )
    } else {
        Some(resolve_columns(reader, hits, &needed).await?)
    };

    // Assemble output columns in the projection's emit order, each
    // drawn from the decoded scalar batch or the synthesized score. The
    // score array is one float per hit, built the first time a
    // projection asks for it.
    let mut score: Option<ArrayRef> = None;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(requested.len());
    for &p in &requested {
        if p == score_idx {
            let built = score.get_or_insert_with(|| {
                Arc::new(Float32Array::from_iter_values(hits.iter().map(|h| h.score))) as ArrayRef
            });
            columns.push(Arc::clone(built));
        } else {
            let name = scalar_schema.field(p).name();
            let rb = resolved
                .as_ref()
                .expect("a scalar column is projected => columns resolved");
            let idx = rb.schema().index_of(name).map_err(QueryError::internal)?;
            columns.push(Arc::clone(rb.column(idx)));
        }
    }

    // `try_new_with_options` carries the row count so a projection
    // that selects no columns (e.g. `COUNT(*)`) still reports
    // `hits.len()` rows.
    RecordBatch::try_new_with_options(
        projected_schema,
        columns,
        &RecordBatchOptions::new().with_row_count(Some(hits.len())),
    )
    .map_err(|e| QueryError::internal(e).into())
}

/// Hit → stable-`_id` translation by manifest arithmetic — the
/// no-I/O fast path for the bare (`None`) projection.
///
/// Ids are minted in contiguous spans and the superfile body stores
/// rows in id order, so when a superfile's manifest stats satisfy
/// `id_max - id_min + 1 == n_docs` the stable id of row `local` is
/// exactly `id_min + local`. Returns the single-`_id`-column batch in
/// hit (rank) order, or `None` when any hit's superfile fails the span
/// check (e.g. a multi-span commit gapped the range) — the caller
/// then falls back to the id-page read.
fn resolve_ids_arithmetic(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> Option<DfResult<RecordBatch>> {
    let manifest = reader.manifest();
    // Hit sets are top-k sized, so per-superfile memoization via a
    // linear scan is cheaper than building a map.
    let mut memo: Vec<(SuperfileUri, i128)> = Vec::new();
    let mut ids: Vec<i128> = Vec::with_capacity(hits.len());
    for hit in hits {
        if let Some(id) = hit.stable_id {
            ids.push(id);
            continue;
        }
        let base = match memo.iter().find(|(uri, _)| *uri == hit.superfile) {
            Some((_, base)) => *base,
            None => {
                let entry = manifest
                    .superfiles
                    .iter()
                    .find(|e| e.uri == hit.superfile)?;
                // `None` when the span is gapped (not contiguous single-append),
                // in which case arithmetic id resolution doesn't apply.
                let base = row_id_from_manifest_entry(entry, 0)?;
                memo.push((hit.superfile, base));
                base
            }
        };
        ids.push(base + i128::from(hit.local_doc_id));
    }

    let array = match Decimal128Array::from_iter_values(ids)
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
    {
        Ok(a) => a,
        Err(e) => {
            return Some(Err(DataFusionError::from(QueryError::Internal(
                e.to_string(),
            ))));
        }
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        reader.options().id_column.clone(),
        DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
        false,
    )]));
    Some(
        RecordBatch::try_new(schema, vec![Arc::new(array) as ArrayRef])
            .map_err(|e| QueryError::internal(e).into()),
    )
}

/// Fill the `stable_id` of each hit that lacks one, in place, using the
/// cheap, cache-backed resolution path: contiguous-span arithmetic where it
/// applies, else a `_id`-column read routed through `resolve_columns` (and
/// thus `decoded_scalar_cache`), so a warm reader decodes each superfile's
/// `_id` column at most once and later queries hit the cache.
///
/// A hit that already has an id is never read again. Its local may not
/// address a Parquet row: a vector hit in a cell-packed file carries the
/// position in its cell's postings (boundary stubs included), stamped from
/// the inline id region at search time, so reading `_id` at that local would
/// overwrite a right id with another row's.
///
/// This is the resolution `resolve_hits` already uses for an id-only
/// projection. The scored fan-out uses it to stamp its final top-k instead
/// of a per-superfile `take_by_local_doc_ids`, which re-runs a scattered
/// Parquet page read (~one page + decompress per hit) on *every* query and
/// dominates large-k scored latency on real corpora (it is ~free only on a
/// tiny table with few pages, which is why synthetic benches never saw it).
pub(crate) async fn stamp_stable_ids(
    reader: &SupertableReader,
    hits: &mut [SuperfileHit],
) -> DfResult<()> {
    let unstamped = hits.iter().filter(|h| h.stable_id.is_none()).count();
    if unstamped == 0 {
        return Ok(());
    }
    // No hit has an id: resolve the slice as it is. An FTS hit set over
    // cell-packed files arrives this way and can hold every row of the
    // table, so copying it out first would cost a pass over all of them.
    if unstamped == hits.len() {
        let ids = resolve_hit_ids(reader, hits).await?;
        for (hit, id) in hits.iter_mut().zip(ids.values()) {
            hit.stable_id = Some(*id);
        }
        return Ok(());
    }
    // Some hits have one (hybrid fuses stamped vector hits with unstamped
    // FTS hits): resolve only the rest, a set no larger than the fused k.
    let missing: Vec<usize> = (0..hits.len())
        .filter(|&i| hits[i].stable_id.is_none())
        .collect();
    let subset: Vec<SuperfileHit> = missing.iter().map(|&i| hits[i]).collect();
    let ids = resolve_hit_ids(reader, &subset).await?;
    for (&i, id) in missing.iter().zip(ids.values()) {
        hits[i].stable_id = Some(*id);
    }
    Ok(())
}

/// The `_id` of each hit in `hits`, in order: span arithmetic where it
/// applies, else a cache-backed `_id`-column read.
async fn resolve_hit_ids(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
) -> DfResult<Decimal128Array> {
    let id_batch = match resolve_ids_arithmetic(reader, hits) {
        Some(batch) => batch?,
        None => {
            let id_column = reader.options().id_column.clone();
            resolve_columns(reader, hits, &[id_column.as_str()]).await?
        }
    };
    let ids = id_batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| QueryError::Internal("stamp_stable_ids: _id column not Decimal128".into()))?
        .clone();
    Ok(ids)
}

/// Build the engine-native `_id` + `score` batch directly from already-resolved
/// stable ids and per-hit scores — the same two-column shape `resolve_hits_named`
/// returns for a `None` projection, but synthesized without opening any
/// superfile. Used by the hidden-index id-only fast path, which holds the
/// stable `_id` after the remap's id-resolution step and so needs neither the
/// user-superfile lookup nor a data-page read. `ids` and `scores` are parallel
/// and already in global rank order.
pub(crate) fn id_score_batch(
    reader: &SupertableReader,
    ids: &[i128],
    scores: &[f32],
) -> DfResult<RecordBatch> {
    let id_array = Decimal128Array::from_iter_values(ids.iter().copied())
        .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
        .map_err(QueryError::internal)?;
    let score_array = Float32Array::from_iter_values(scores.iter().copied());
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            reader.options().id_column.clone(),
            DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
            false,
        ),
        Field::new(SCORE_COLUMN, DataType::Float32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(score_array) as ArrayRef,
        ],
    )
    .map_err(|e| QueryError::internal(e).into())
}

/// One superfile of [`resolve_columns`] once its reader is open: its rows
/// taken already (a lazy reader, dropped since), or a resident reader whose
/// rows are decoded with the others in one pool job.
enum Resolved {
    Taken(usize, RecordBatch),
    Resident(usize, Arc<SuperfileReader>, Option<FileSchemaMap>),
}

/// Read `names` (scalar columns) at the `hits`' `(superfile,
/// local_doc_id)` rows and return them in global rank order.
///
/// Hits are grouped by superfile for one column-projected
/// [`take_by_local_doc_ids`] per superfile; the per-superfile batches are
/// concatenated and a single `take` restores rank order. Caller
/// guarantees `hits` and `names` are both non-empty.
///
/// [`take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
async fn resolve_columns(
    reader: &SupertableReader,
    hits: &[SuperfileHit],
    names: &[&str],
) -> DfResult<RecordBatch> {
    // Group local_doc_ids by superfile, preserving first-seen superfile
    // order and recording where each global hit lands.
    let mut seg_order: Vec<SuperfileUri> = Vec::new();
    let mut seg_locals: Vec<Vec<u32>> = Vec::new();
    let mut placement: Vec<(usize, usize)> = Vec::with_capacity(hits.len());
    // An unranked match can hold millions of hits across thousands of
    // superfiles, so a superfile's slot is looked up, not searched for.
    let mut seg_index: HashMap<SuperfileUri, usize> = HashMap::new();
    for hit in hits {
        let seg_idx = *seg_index.entry(hit.superfile).or_insert_with(|| {
            seg_order.push(hit.superfile);
            seg_locals.push(Vec::new());
            seg_order.len() - 1
        });
        let row = seg_locals[seg_idx].len();
        seg_locals[seg_idx].push(hit.local_doc_id);
        placement.push((seg_idx, row));
    }

    // Open every distinct superfile reader concurrently on the tokio
    // runtime — these are async I/O (in-memory cache lookups /
    // disk-cache cold fetches), so overlapping them is the right
    // model and they cost ~microseconds when warm.
    let manifest = reader.manifest();
    let out_schema = hit_output_schema(&manifest.stored_schema(), names)?;
    let decoded_cache = reader.decoded_scalar_cache();
    let schema_id = manifest.table_schema().schema_id();
    let mut slots: Vec<Option<RecordBatch>> = vec![None; seg_order.len()];
    let mut misses = Vec::new();
    for (index, (&uri, locals)) in seg_order.iter().zip(&seg_locals).enumerate() {
        if let Some(batch) = decoded_cache.get(uri, schema_id, locals, names) {
            slots[index] = Some(batch);
        } else {
            misses.push((index, uri));
        }
    }

    // Opened as the search that found the hits opened: by ranges alone on a
    // table the cache cannot hold, so resolving a hit's columns never starts
    // a whole-file download the search itself declined.
    let intent = reader.open_intent();
    // Each superfile opens, takes its hit rows and drops its reader in one
    // task, at most `superfiles_in_flight` at once: a ranked result touches
    // a few superfiles, but an unranked match can touch every superfile of
    // the table, and a lazy reader holds its parsed footer and page index,
    // several MB of heap, for as long as it is held.
    //
    //   - **Resident readers** (in-memory tier / freshly written):
    //     `take_by_local_doc_ids` is a CPU-bound Parquet page decode over
    //     already-resident bytes, so it runs on `options.reader_pool`
    //     (rayon), bridged back so no tokio worker blocks under it.
    //   - **Lazy readers** stream only projected hit rows through their
    //     existing `LazyByteSource`. That preserves the disk cache's block
    //     layer: cold misses range-fetch, repeated warm reads hit local
    //     blocks, and the file is never materialized.
    //
    // A file whose columns match the table's reads by name; any other reads
    // through its map: the table's columns are projected under the file's
    // names and reshaped to the table's.
    let in_flight = manifest.options.superfiles_in_flight.max(1);
    let manifest_arc = Arc::clone(reader.manifest());
    let names_owned: Arc<Vec<String>> = Arc::new(names.iter().map(|s| (*s).to_string()).collect());
    // Collected before the join: the futures do no work until polled, and a
    // lazy iterator held across the join's awaits trips rustc's "`FnOnce` is
    // not general enough" in the callers' streams. Each unit is spawned when
    // the join first polls it, so at most `in_flight` exist at once, and on a
    // task of its own: polled inside the join they would all decode on one
    // thread.
    let units: Vec<_> = misses
        .into_iter()
        .map(|(index, uri)| {
            let manifest = Arc::clone(&manifest_arc);
            let names = Arc::clone(&names_owned);
            let locals = seg_locals[index].clone();
            let out_schema = Arc::clone(&out_schema);
            async move {
                let entry = manifest
                    .lookup_superfile_entry(uri)
                    .await
                    .map_err(QueryError::ManifestLoad)?
                    .ok_or_else(|| {
                        QueryError::Internal(format!(
                            "resolve_hits: superfile {uri:?} missing from manifest"
                        ))
                    })?;
                let options = &manifest.options;
                let rd = superfile_reader(
                    &options.store,
                    options.disk_cache.as_ref(),
                    options.storage.as_ref(),
                    &entry.uri,
                    &entry.storage_path(),
                    entry.subsection_offsets.as_ref(),
                    intent,
                )
                .await
                .map_err(QueryError::store)?;
                let adapter = file_adapter(&manifest, &rd);
                if rd.can_take_by_local_doc_ids() {
                    // Resident: decoded below with the other resident
                    // files, in one pool job.
                    return Ok::<_, DataFusionError>(Resolved::Resident(index, rd, adapter));
                }
                // A lazy take fetches here and decodes on the reader pool.
                let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
                let batch = take_lazy_file(
                    &rd,
                    &locals,
                    &name_refs,
                    &adapter,
                    &out_schema,
                    &manifest.options.reader_pool,
                )
                .await?;
                Ok(Resolved::Taken(index, batch))
            }
        })
        .collect();
    let mut resident: Vec<(usize, Arc<SuperfileReader>, Vec<u32>, Option<FileSchemaMap>)> =
        Vec::new();
    let mut done: Vec<(usize, RecordBatch)> = Vec::new();
    for unit in try_join_bounded(units, in_flight).await? {
        match unit {
            Resolved::Taken(i, batch) => done.push((i, batch)),
            Resolved::Resident(i, rd, adapter) => {
                resident.push((i, rd, seg_locals[i].clone(), adapter))
            }
        }
    }
    if !resident.is_empty() {
        // Owned inputs so the rayon closure is `'static`.
        let owned_names: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
        let out_schema = Arc::clone(&out_schema);
        let decoded: Vec<((usize, RecordBatch), u64)> = run_on_pool(
            Some(&manifest.options.reader_pool),
            "resolve decode: reader pool dropped result",
            move || {
                let name_refs: Vec<&str> = owned_names.iter().map(String::as_str).collect();
                resident
                    .into_par_iter()
                    .map(|(i, sf, locals, adapter)| {
                        let (batch, ns) = timed_section(|| {
                            take_resident_file(&sf, &locals, &name_refs, &adapter, &out_schema)
                        });
                        batch.map(|batch| ((i, batch), ns))
                    })
                    .collect::<Result<Vec<_>, _>>()
            },
        )
        .await
        .map_err(QueryError::internal)?
        .map_err(|e| DataFusionError::from(QueryError::from(e)))?;
        if let Some(stats) = &reader.op_stats {
            stats.add_kernel_cpu_ns(decoded.iter().map(|(_, ns)| *ns).sum());
        }
        done.extend(decoded.into_iter().map(|(item, _)| item));
    }
    for (i, batch) in done {
        decoded_cache.insert(
            seg_order[i],
            schema_id,
            &seg_locals[i],
            names,
            batch.clone(),
        );
        slots[i] = Some(batch);
    }
    let per_superfile: Vec<RecordBatch> = slots
        .into_iter()
        .map(|s| s.expect("invariant: every superfile resolved by exactly one wave"))
        .collect();
    // Assemble directly into global rank order. `interleave_record_batch`
    // gathers from the per-superfile arrays in one pass; the old
    // concatenate-then-take path allocated and copied every projected column
    // twice before producing the same top-k-sized output.
    // Every hit row was decoded from stored columns (cache hits included:
    // the decoded-scalar cache is per reader generation, so a served batch
    // is still this query's planned materialization).
    if let Some(stats) = &reader.op_stats {
        stats.add_rows_materialized(hits.len() as u64);
    }
    let batches: Vec<&RecordBatch> = per_superfile.iter().collect();
    interleave_record_batch(&batches, &placement)
        .map_err(|error| QueryError::internal(error).into())
}

/// The map from a file's own column names to the table's, when the file
/// was written under different ones (before a rename); `None` when the
/// file's columns are the table's and reads need no translation.
fn file_adapter(manifest: &ManifestSnapshot, rd: &SuperfileReader) -> Option<FileSchemaMap> {
    let map = FileSchemaMap::new(
        &manifest.table_schema(),
        &manifest.options.id_column,
        &PhysicalSchema::of_reader(rd, &manifest.options.legacy_names()),
    );
    (!map.is_identity()).then_some(map)
}

/// `names` at `locals` of one resident superfile, under the table's
/// column names: a CPU-bound Parquet page decode over resident bytes, so
/// callers run it on the reader pool.
fn take_resident_file(
    rd: &SuperfileReader,
    locals: &[u32],
    names: &[&str],
    adapter: &Option<FileSchemaMap>,
    out_schema: &SchemaRef,
) -> Result<RecordBatch, ReadError> {
    match adapter {
        None => rd.take_by_local_doc_ids(locals, names),
        Some(map) => {
            let physical = map.stored_names(names);
            let read = rd.take_by_local_doc_ids(locals, &physical)?;
            map.adapt(&read, out_schema)
                .map_err(|e| ReadError::Columnar(e.to_string()))
        }
    }
}

/// `names` at `locals` of one lazily opened superfile, streamed through
/// its byte source so the disk cache's block layer serves repeated reads.
async fn take_lazy_file(
    rd: &SuperfileReader,
    locals: &[u32],
    names: &[&str],
    adapter: &Option<FileSchemaMap>,
    out_schema: &SchemaRef,
    pool: &ThreadPool,
) -> DfResult<RecordBatch> {
    match adapter {
        None => take_rows_byte_source(rd, locals, names, Some(pool)).await,
        Some(map) => {
            let physical = map.stored_names(names);
            let read = take_rows_byte_source(rd, locals, &physical, Some(pool)).await?;
            map.adapt(&read, out_schema)
                .map_err(|e| DataFusionError::Execution(e.to_string()))
        }
    }
}

/// `names` at `locals` of one superfile, for a caller that wants the rows
/// of each superfile as its own batch rather than a global hit set: the
/// resident decode goes to the reader pool, a lazy reader streams, and the
/// decode is counted as kernel CPU like [`resolve_hits`]'s. The batch
/// carries `out_schema`, [`hit_output_schema`] for `names`.
pub(crate) async fn take_file_columns(
    manifest: &ManifestSnapshot,
    rd: &Arc<SuperfileReader>,
    locals: Vec<u32>,
    names: &[&str],
    out_schema: &SchemaRef,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> DfResult<RecordBatch> {
    let adapter = file_adapter(manifest, rd);
    if !rd.can_take_by_local_doc_ids() {
        return take_lazy_file(
            rd,
            &locals,
            names,
            &adapter,
            out_schema,
            &manifest.options.reader_pool,
        )
        .await;
    }
    let owned_names: Vec<String> = names.iter().map(|s| (*s).to_string()).collect();
    let out_schema = Arc::clone(out_schema);
    let rd = Arc::clone(rd);
    let (batch, ns) = run_on_pool(
        Some(&manifest.options.reader_pool),
        "file columns decode: reader pool dropped result",
        move || {
            let name_refs: Vec<&str> = owned_names.iter().map(String::as_str).collect();
            timed_section(|| take_resident_file(&rd, &locals, &name_refs, &adapter, &out_schema))
        },
    )
    .await
    .map_err(QueryError::internal)?;
    if let Some(stats) = op_stats {
        stats.add_kernel_cpu_ns(ns);
    }
    batch.map_err(|e| QueryError::from(e).into())
}

/// `names` of every row of row group `row_group` of one superfile, in file
/// order, under the table's column names: [`take_file_columns`] for a scan
/// reading whole row groups. Each column is decoded once, with no list of
/// the rows to sort and no copy to put them in its order, which a take of
/// every row would pay. The resident decode goes to the reader pool and is
/// counted as kernel CPU; a lazy reader streams.
///
/// `out_schema` types the batch, field for field with `names`, where the
/// file's columns need adapting. A string or binary column it types as a
/// view decodes straight into views, so a scan that works on views copies
/// no value to make them; the file's own types serve every other column.
pub(crate) async fn read_file_row_group(
    manifest: &ManifestSnapshot,
    rd: &Arc<SuperfileReader>,
    row_group: usize,
    names: &[&str],
    out_schema: &SchemaRef,
    op_stats: &Option<Arc<OpStatsCollector>>,
) -> DfResult<RecordBatch> {
    let adapter = file_adapter(manifest, rd);
    let stored = |names: &[&str]| -> Vec<String> {
        match &adapter {
            Some(map) => map
                .stored_names(names)
                .into_iter()
                .map(str::to_owned)
                .collect(),
            None => names.iter().map(|name| (*name).to_owned()).collect(),
        }
    };
    let physical = stored(names);
    let viewed: Vec<&str> = names
        .iter()
        .zip(out_schema.fields())
        .filter(|(_, field)| matches!(field.data_type(), DataType::Utf8View | DataType::BinaryView))
        .map(|(name, _)| *name)
        .collect();
    let views = stored(&viewed);
    let read = if rd.can_take_by_local_doc_ids() {
        let rd = Arc::clone(rd);
        let (read, ns) = run_on_pool(
            Some(&manifest.options.reader_pool),
            "row group decode: reader pool dropped result",
            move || {
                let refs: Vec<&str> = physical.iter().map(String::as_str).collect();
                let views: Vec<&str> = views.iter().map(String::as_str).collect();
                timed_section(|| rd.read_row_group(row_group, &refs, &views))
            },
        )
        .await
        .map_err(QueryError::internal)?;
        if let Some(stats) = op_stats {
            stats.add_kernel_cpu_ns(ns);
        }
        read.map_err(QueryError::from)?
    } else {
        let refs: Vec<&str> = physical.iter().map(String::as_str).collect();
        let views: Vec<&str> = views.iter().map(String::as_str).collect();
        read_row_group_byte_source(rd, row_group, &refs, &views).await?
    };
    match adapter {
        None => Ok(read),
        Some(map) => map
            .adapt(&read, out_schema)
            .map_err(|e| DataFusionError::Execution(e.to_string())),
    }
}

/// [`SuperfileReader::read_row_group`] for a lazily opened superfile,
/// streamed through its byte source so the disk cache's block layer serves
/// repeated reads. Reading whole row groups needs no page index.
async fn read_row_group_byte_source(
    reader: &SuperfileReader,
    row_group: usize,
    names: &[&str],
    views: &[&str],
) -> DfResult<RecordBatch> {
    let input = ByteSourceAsyncReader {
        source: Source::Lazy(reader.byte_source()),
        metadata: Arc::clone(reader.parquet_metadata()),
        prefetched: Vec::new(),
    };
    let options = if views.is_empty() {
        ArrowReaderOptions::new()
    } else {
        view_read_options(reader.schema(), views)
    };
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(input, options)
        .await
        .map_err(QueryError::store)?;
    let (builder, rows) = whole_row_group(builder, row_group, names).map_err(QueryError::from)?;
    let stream = builder.build().map_err(QueryError::store)?;
    let read_schema = stream.schema().clone();
    let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(QueryError::store)?;
    in_projection_order(&read_schema, batches, names, rows).map_err(|e| QueryError::from(e).into())
}

/// Parquet async reader backed by the `SuperfileReader`'s existing byte source.
/// For disk-cache readers this preserves the block-cache layer instead of
/// bypassing it with a new object-store handle on every scalar projection.
/// The fields a hit batch carries for `names`, in that order: the table's
/// stored columns, which is the shape every adapted file reads into.
pub(crate) fn hit_output_schema(stored: &Schema, names: &[&str]) -> DfResult<SchemaRef> {
    let fields = names
        .iter()
        .map(|name| {
            stored
                .field_with_name(name)
                .map(|f| Arc::new(f.clone()))
                .map_err(|_| DataFusionError::Execution(format!("unknown column {name}")))
        })
        .collect::<DfResult<Vec<_>>>()?;
    Ok(Arc::new(Schema::new(fields)))
}

struct ByteSourceAsyncReader {
    source: Source,
    metadata: Arc<ParquetMetaData>,
    /// Spans fetched ahead of the decode, ascending and disjoint: a read
    /// inside one is served from it (see [`take_prefetch_spans`]).
    prefetched: Vec<(Range<u64>, Bytes)>,
}

impl ByteSourceAsyncReader {
    /// `range` from a prefetched span that holds it whole.
    fn prefetched(&self, range: &Range<u64>) -> Option<Bytes> {
        prefetched_slice(&self.prefetched, range)
    }
}

/// `range` from the one of `spans` (ascending, disjoint) that holds it whole.
fn prefetched_slice(spans: &[(Range<u64>, Bytes)], range: &Range<u64>) -> Option<Bytes> {
    let at = spans.partition_point(|(span, _)| span.end <= range.start);
    let (span, bytes) = spans.get(at)?;
    (span.start <= range.start && range.end <= span.end).then(|| {
        bytes.slice((range.start - span.start) as usize..(range.end - span.start) as usize)
    })
}

/// A take's prefetched spans as a file the synchronous Parquet reader can
/// decode from, off the async runtime. A read outside them is an error, and
/// the take then decodes through its byte source instead.
struct PrefetchedChunks {
    len: u64,
    spans: Vec<(Range<u64>, Bytes)>,
}

impl Length for PrefetchedChunks {
    fn len(&self) -> u64 {
        self.len
    }
}

impl ChunkReader for PrefetchedChunks {
    type T = Reader<Bytes>;

    fn get_read(&self, start: u64) -> ParquetResult<Self::T> {
        let at = self.spans.partition_point(|(span, _)| span.end <= start);
        match self.spans.get(at) {
            Some((span, bytes)) if span.start <= start => {
                Ok(bytes.slice((start - span.start) as usize..).reader())
            }
            _ => Err(ParquetError::General(format!(
                "byte {start} is not prefetched"
            ))),
        }
    }

    fn get_bytes(&self, start: u64, length: usize) -> ParquetResult<Bytes> {
        prefetched_slice(&self.spans, &(start..start + length as u64)).ok_or_else(|| {
            ParquetError::General(format!("bytes {start}+{length} are not prefetched"))
        })
    }
}

/// Most bytes a take fetches ahead for one superfile; past this it streams
/// row group by row group as before.
const TAKE_PREFETCH_MAX_BYTES: u64 = 64 * 1024 * 1024;
/// Spans of one superfile's take prefetch in flight at once.
const TAKE_PREFETCH_CONCURRENCY: usize = 16;
/// The share of the host's memory every take's prefetch together may hold.
const TAKE_PREFETCH_MEMORY_SHARE: u64 = 16;

/// Prefetched bytes every take in the process holds at once, in KiB: a share
/// of the host's memory, never less than one take's
/// [`TAKE_PREFETCH_MAX_BYTES`]. A take waits for room, and holds it until
/// its decode is done.
fn take_prefetch_budget() -> &'static Semaphore {
    static BUDGET: OnceLock<Semaphore> = OnceLock::new();
    BUDGET.get_or_init(|| {
        let bytes = total_memory_bytes()
            .map_or(TAKE_PREFETCH_MAX_BYTES, |total| {
                total / TAKE_PREFETCH_MEMORY_SHARE
            })
            .max(TAKE_PREFETCH_MAX_BYTES);
        Semaphore::new((bytes / 1024) as usize)
    })
}

/// The column chunks of the `roots` columns (top-level, as the Arrow
/// schema numbers them) in every row group `sorted` (ascending, distinct
/// rows) falls in, merged where they touch: the bytes a take's decode reads,
/// so they can be fetched at once instead of one row group after another. A
/// nested root contributes every leaf under it. `None` when they come to
/// more than [`TAKE_PREFETCH_MAX_BYTES`] or a root is not in the file.
fn take_prefetch_spans(
    metadata: &ParquetMetaData,
    sorted: &[u32],
    roots: &[usize],
) -> Option<Vec<Range<u64>>> {
    let descr = metadata.file_metadata().schema_descr();
    if roots
        .iter()
        .any(|&r| r >= descr.root_schema().get_fields().len())
    {
        return None;
    }
    // Chunks are per Parquet leaf, so a struct before a projected column
    // shifts its chunk past its Arrow index.
    let columns: Vec<usize> = (0..descr.num_columns())
        .filter(|&leaf| roots.contains(&descr.get_column_root_idx(leaf)))
        .collect();
    let mut spans: Vec<Range<u64>> = Vec::new();
    let (mut first_row, mut next) = (0u64, 0usize);
    for (index, rg) in metadata.row_groups().iter().enumerate() {
        let end_row = first_row + rg.num_rows() as u64;
        let before = next;
        while next < sorted.len() && u64::from(sorted[next]) < end_row {
            next += 1;
        }
        if next > before {
            for &c in &columns {
                hit_pages(
                    metadata,
                    index,
                    c,
                    first_row,
                    end_row,
                    &sorted[before..next],
                    &mut spans,
                );
            }
        }
        first_row = end_row;
    }
    spans.sort_unstable_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(spans.len());
    for span in spans {
        match merged.last_mut() {
            Some(last) if span.start <= last.end => last.end = last.end.max(span.end),
            _ => merged.push(span),
        }
    }
    let bytes: u64 = merged.iter().map(|r| r.end - r.start).sum();
    (bytes <= TAKE_PREFETCH_MAX_BYTES).then_some(merged)
}

/// The bytes of leaf column `c` in row group `index` (rows `first_row` to
/// `end_row`) that a decode of rows `hits` reads, pushed onto `spans`. With
/// the page index loaded the decode skips pages without a hit, so that is
/// each page holding one, plus the dictionary page; without it, the chunk.
fn hit_pages(
    metadata: &ParquetMetaData,
    index: usize,
    c: usize,
    first_row: u64,
    end_row: u64,
    hits: &[u32],
    spans: &mut Vec<Range<u64>>,
) {
    let (offset, len) = metadata.row_group(index).column(c).byte_range();
    let pages = metadata
        .offset_index()
        .and_then(|oi| oi.get(index))
        .and_then(|cols| cols.get(c))
        .map(|oi| oi.page_locations())
        .filter(|pages| !pages.is_empty());
    let Some(pages) = pages else {
        spans.push(offset..offset + len);
        return;
    };
    let first_page = pages[0].offset as u64;
    if first_page > offset {
        spans.push(offset..first_page);
    }
    let mut hit = 0;
    for (i, page) in pages.iter().enumerate() {
        let start_row = first_row + page.first_row_index as u64;
        let page_end_row = pages
            .get(i + 1)
            .map_or(end_row, |next| first_row + next.first_row_index as u64);
        while hit < hits.len() && u64::from(hits[hit]) < start_row {
            hit += 1;
        }
        if hit < hits.len() && u64::from(hits[hit]) < page_end_row {
            let start = page.offset as u64;
            spans.push(start..start + page.compressed_page_size as u64);
        }
    }
}

impl AsyncFileReader for ByteSourceAsyncReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, ParquetResult<Bytes>> {
        if let Some(bytes) = self.prefetched(&range) {
            return async move { Ok(bytes) }.boxed();
        }
        // `range_async` resolves mmap/block-resident ranges synchronously
        // (zero-copy, no I/O) via `try_get_range_sync`, and only `await`s a
        // real fetch on a cold miss — the same fast path the FTS reader's
        // posting fetches ride.
        let source = self.source.clone();
        let range = range.start as usize..range.end as usize;
        async move {
            source
                .range_async(range)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))
        }
        .boxed()
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, ParquetResult<Vec<Bytes>>> {
        // Each range the prefetch holds is served from it; only the rest go
        // to the source. `get_ranges_parallel_async` serves each resident one
        // synchronously and batches the cold misses into one `try_join_all`,
        // returning bytes in input order.
        let mut out: Vec<Option<Bytes>> = ranges.iter().map(|r| self.prefetched(r)).collect();
        let missing: Vec<usize> = (0..out.len()).filter(|&i| out[i].is_none()).collect();
        if missing.is_empty() {
            return async move { Ok(out.into_iter().flatten().collect()) }.boxed();
        }
        let source = self.source.clone();
        let wanted: Vec<Range<usize>> = missing
            .iter()
            .map(|&i| ranges[i].start as usize..ranges[i].end as usize)
            .collect();
        async move {
            let fetched = source
                .get_ranges_parallel_async(&wanted)
                .await
                .map_err(|error| ParquetError::External(Box::new(error)))?;
            for (i, bytes) in missing.into_iter().zip(fetched) {
                out[i] = Some(bytes);
            }
            Ok(out.into_iter().flatten().collect())
        }
        .boxed()
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, ParquetResult<Arc<ParquetMetaData>>> {
        let metadata = Arc::clone(&self.metadata);
        async move { Ok(metadata) }.boxed()
    }
}

/// The `names` columns at `local_doc_ids` of one superfile, in that order:
/// decoded on `pool` when its Parquet bytes are resident, streamed through
/// its byte source otherwise. The split [`resolve_hits`] makes across many
/// superfiles at once, for a caller holding one.
pub(crate) async fn take_rows(
    reader: &Arc<SuperfileReader>,
    local_doc_ids: &[u32],
    names: &[&str],
    pool: &ThreadPool,
) -> DfResult<RecordBatch> {
    if !reader.can_take_by_local_doc_ids() {
        return take_rows_byte_source(reader, local_doc_ids, names, Some(pool)).await;
    }
    let owned_reader = Arc::clone(reader);
    let ids = local_doc_ids.to_vec();
    let owned_names: Vec<String> = names.iter().map(|name| (*name).to_owned()).collect();
    run_on_pool(
        Some(pool),
        "take rows: reader pool dropped result",
        move || {
            let name_refs: Vec<&str> = owned_names.iter().map(String::as_str).collect();
            owned_reader.take_by_local_doc_ids(&ids, &name_refs)
        },
    )
    .await
    .map_err(QueryError::internal)?
    .map_err(|e| DataFusionError::from(QueryError::from(e)))
}

/// Stream projected rows through a reader's cache-aware byte source.
///
/// Deliberately reports NO planned ranges: whether a take streams page
/// ranges (lazy reader) or decodes resident bytes in place (promoted
/// reader) is reader-cache state, so counting the streamed ranges would
/// make the priced counter cache-dependent. The invariant materialization
/// signals are `rows_materialized` and the decode CPU brackets.
pub(crate) async fn take_rows_byte_source(
    reader: &SuperfileReader,
    local_doc_ids: &[u32],
    names: &[&str],
    pool: Option<&ThreadPool>,
) -> DfResult<RecordBatch> {
    // The bare `_id` projection of a few rows comes from the packed sidecar's
    // blocks when the superfile carries one: a couple of kilobytes a block,
    // no page index. A read of enough rows takes the sidecar whole instead,
    // through the path below. A sidecar that cannot be read leaves the rows
    // to the Parquet column, which holds the same ids.
    if let [name] = names
        && *name == reader.id_column()
        && !reader.id_sidecar_serves(local_doc_ids, name)
    {
        match reader.ids_from_sidecar_blocks(local_doc_ids).await {
            Ok(Some(ids)) => {
                let field = reader
                    .schema()
                    .field_with_name(name)
                    .map_err(QueryError::internal)?
                    .clone();
                return RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![ids])
                    .map_err(|e| QueryError::internal(e).into());
            }
            Ok(None) => {}
            Err(error) => {
                warn!(%error, "reading _id from Parquet, the id sidecar blocks did not serve");
            }
        }
    }
    let metadata = reader
        .parquet_metadata_with_page_index()
        .await
        .map_err(QueryError::from)?;
    let source = Source::Lazy(reader.byte_source());
    // Fetch the take's column chunks at once rather than letting the decode
    // fetch them a row group at a time: a hit set spread over hundreds of
    // row groups otherwise costs that many round trips one after another.
    let mut sorted = local_doc_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let roots: Vec<usize> = names
        .iter()
        .filter_map(|name| reader.schema().index_of(name).ok())
        .collect();
    let mut prefetched = Vec::new();
    // Held until the decode is done: the prefetched bytes live that long.
    let mut _room = None;
    if let Some(spans) = take_prefetch_spans(&metadata, &sorted, &roots).filter(|s| !s.is_empty()) {
        let bytes: u64 = spans.iter().map(|r| r.end - r.start).sum();
        _room = Some(
            take_prefetch_budget()
                .acquire_many(bytes.div_ceil(1024) as u32)
                .await
                .map_err(QueryError::internal)?,
        );
        let fetched: Vec<Bytes> = stream::iter(spans.iter().cloned().map(|span| {
            let source = source.clone();
            async move {
                source
                    .range_exact_async(span.start as usize..span.end as usize)
                    .await
                    .map_err(|e| QueryError::Internal(format!("take prefetch: {e}")))
            }
        }))
        .buffered(TAKE_PREFETCH_CONCURRENCY)
        .try_collect()
        .await?;
        prefetched = spans.into_iter().zip(fetched).collect();
        // The fetch was the I/O; the decode is CPU, and goes to the reader
        // pool, reading only the prefetched bytes.
        let chunks = PrefetchedChunks {
            len: reader.byte_source().size(),
            spans: prefetched.clone(),
        };
        let (metadata, schema, n_docs) = (
            Arc::clone(&metadata),
            Arc::clone(reader.schema()),
            reader.n_docs(),
        );
        let ids = local_doc_ids.to_vec();
        let owned: Vec<String> = names.iter().map(|n| (*n).to_string()).collect();
        let decoded = run_on_pool_unless_blocking(
            pool,
            "take decode: reader pool dropped result",
            move || {
                let names: Vec<&str> = owned.iter().map(String::as_str).collect();
                take_rows_prefetched(chunks, metadata, &schema, n_docs, &ids, &names)
            },
        )
        .await
        .map_err(QueryError::internal)?;
        match decoded {
            Ok(batch) => return Ok(batch),
            // A read outside the prefetch: decode through the source, which
            // still serves the prefetched spans from memory.
            Err(error) => debug!(%error, "take decoding through its byte source"),
        }
    }
    let input = ByteSourceAsyncReader {
        source,
        metadata,
        prefetched,
    };
    take_rows_async(
        input,
        reader.schema(),
        reader.n_docs(),
        local_doc_ids,
        names,
    )
    .await
}

/// Stream the projected `names` columns at `local_doc_ids` from an object-store
/// superfile. Used when a MultiCell tombstone resolve must read `_id` pages
/// without resident parquet bytes; also driven directly by parity tests.
///
/// [`SuperfileReader::take_by_local_doc_ids`]: crate::superfile::SuperfileReader::take_by_local_doc_ids
pub(crate) async fn take_rows_object_store(
    store: Arc<dyn ObjectStore>,
    path: ObjPath,
    file_size: Option<u64>,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    let mut object_reader = ParquetObjectReader::new(store, path).with_preload_offset_index(true);
    if let Some(size) = file_size.filter(|&s| s > 0) {
        // Skip the size-discovery HEAD when the manifest already knows it.
        object_reader = object_reader.with_file_size(size);
    }
    take_rows_async(object_reader, file_schema, n_docs, local_doc_ids, names).await
}

async fn take_rows_async<R>(
    input: R,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch>
where
    R: AsyncFileReader + Unpin + Send + 'static,
{
    let plan = match take_plan(file_schema, n_docs, local_doc_ids, names)? {
        Ok(plan) => plan,
        Err(done) => return Ok(done),
    };
    let builder = ParquetRecordBatchStreamBuilder::new(input)
        .await
        .map_err(QueryError::store)?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), plan.col_indices.iter().copied());
    let stream = builder
        .with_projection(mask)
        .with_row_selection(plan.selection.clone())
        .build()
        .map_err(QueryError::store)?;
    let batches: Vec<RecordBatch> = stream.try_collect().await.map_err(QueryError::store)?;
    finish_take(&plan, batches, local_doc_ids, names)
}

/// [`take_rows_async`] decoding from `chunks`, synchronously: the reader-pool
/// half of a prefetched take.
fn take_rows_prefetched(
    chunks: PrefetchedChunks,
    metadata: Arc<ParquetMetaData>,
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    let plan = match take_plan(file_schema, n_docs, local_doc_ids, names)? {
        Ok(plan) => plan,
        Err(done) => return Ok(done),
    };
    let metadata = ArrowReaderMetadata::try_new(metadata, ArrowReaderOptions::new())
        .map_err(QueryError::store)?;
    let builder = ParquetRecordBatchReaderBuilder::new_with_metadata(chunks, metadata);
    let mask = ProjectionMask::roots(builder.parquet_schema(), plan.col_indices.iter().copied());
    let batches: Vec<RecordBatch> = builder
        .with_projection(mask)
        .with_row_selection(plan.selection.clone())
        .build()
        .map_err(QueryError::store)?
        .collect::<Result<_, _>>()
        .map_err(QueryError::store)?;
    finish_take(&plan, batches, local_doc_ids, names)
}

/// What a take of `local_doc_ids` decodes: the projected columns, the output
/// schema, and the rows as a sorted selection.
struct TakePlan {
    col_indices: Vec<usize>,
    out_schema: SchemaRef,
    sorted: Vec<u32>,
    selection: RowSelection,
}

/// The [`TakePlan`] of a take, or the batch itself when there is nothing to
/// decode: no rows, or no columns this file has.
fn take_plan(
    file_schema: &SchemaRef,
    n_docs: u64,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<Result<TakePlan, RecordBatch>> {
    // Projected column indices (file order) + output fields (caller order).
    let mut col_indices = Vec::with_capacity(names.len());
    let mut out_fields: Vec<Field> = Vec::with_capacity(names.len());
    for &name in names {
        let idx = file_schema
            .index_of(name)
            .map_err(|_| QueryError::Internal(format!("unknown column {name}")))?;
        col_indices.push(idx);
        out_fields.push(file_schema.field(idx).clone());
    }
    let out_schema = Arc::new(Schema::new(out_fields));

    if local_doc_ids.is_empty() {
        return Ok(Err(RecordBatch::new_empty(out_schema)));
    }
    for &d in local_doc_ids {
        if u64::from(d) >= n_docs {
            return Err(DataFusionError::from(QueryError::Internal(format!(
                "doc id {d} out of range (n_docs={n_docs})"
            ))));
        }
    }

    // A projection resolves to no columns at all when every name it asks
    // for belongs to a column this file predates. There is nothing to read,
    // but the row count still has to be right: the caller null-fills to it,
    // and a batch reporting no rows would drop the hits instead.
    if names.is_empty() {
        return RecordBatch::try_new_with_options(
            out_schema,
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(local_doc_ids.len())),
        )
        .map(Err)
        .map_err(|e| QueryError::internal(e).into());
    }

    // Distinct, sorted ids → monotonic skip/select runs (decode only the
    // rows the hits land on, not the whole column). Same selection
    // contract as `take_by_local_doc_ids` — shared helpers, different
    // I/O model (async range GETs here vs resident-bytes decode there).
    let (sorted, selection) = row_selection_for_ids(local_doc_ids);
    Ok(Ok(TakePlan {
        col_indices,
        out_schema,
        sorted,
        selection,
    }))
}

/// A take's decoded `batches` (file column order, sorted rows) as the
/// caller asked: `names` in its order, rows in `local_doc_ids` order.
fn finish_take(
    plan: &TakePlan,
    batches: Vec<RecordBatch>,
    local_doc_ids: &[u32],
    names: &[&str],
) -> DfResult<RecordBatch> {
    let out_schema = Arc::clone(&plan.out_schema);
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(out_schema));
    }
    let read_schema = batches[0].schema();
    let selected = concat_batches(&read_schema, &batches).map_err(QueryError::internal)?;

    // Rank back into the caller's (possibly duplicated) order.
    let indices = rank_back_indices(local_doc_ids, &plan.sorted);

    // Gather columns in caller projection order (parquet returns file order).
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(names.len());
    for &name in names {
        let idx = selected
            .schema()
            .index_of(name)
            .map_err(|_| QueryError::Internal(format!("unknown column {name}")))?;
        columns.push(take(selected.column(idx), &indices, None).map_err(QueryError::internal)?);
    }
    RecordBatch::try_new(out_schema, columns).map_err(|e| QueryError::internal(e).into())
}

/// Extract a string literal argument (a column name, query text, ...).
pub(crate) fn arg_to_string(expr: &Expr, what: &str) -> DfResult<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Ok(s.clone()),
        other => Err(DataFusionError::Plan(format!(
            "{what} must be a string literal, got {other:?}"
        ))),
    }
}

/// Extract a non-negative integer literal argument (`k`).
pub(crate) fn arg_to_usize(expr: &Expr, what: &str) -> DfResult<usize> {
    let n: i64 = match expr {
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => *n,
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => i64::from(*n),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => *n as i64,
        Expr::Literal(ScalarValue::UInt32(Some(n)), _) => i64::from(*n),
        other => {
            return Err(DataFusionError::Plan(format!(
                "{what} must be an integer literal, got {other:?}"
            )));
        }
    };
    usize::try_from(n).map_err(|_| DataFusionError::Plan(format!("{what} must be >= 0, got {n}")))
}

/// Shared test support for the exec-module tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use datafusion::{
        catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider},
        error::Result as DfResult,
        logical_expr::Expr,
        prelude::SessionContext,
    };

    use super::CallScopedTable;

    /// Invoke a table function's `call_with_args` with a throwaway session.
    /// The search TVFs read only the argument exprs, not the session, so a
    /// fresh empty context is enough to satisfy the DataFusion 54 signature.
    pub(crate) fn call_tvf<F: TableFunctionImpl>(
        func: &F,
        exprs: &[Expr],
    ) -> DfResult<Arc<dyn TableProvider>> {
        let ctx = SessionContext::new();
        let state = ctx.state();
        func.call_with_args(TableFunctionArgs::new(exprs, &state))
    }

    /// The call's own provider beneath the per-call scope every search
    /// table function returns (see [`super::scope_to_call`]).
    pub(crate) fn scoped_inner(table: &Arc<dyn TableProvider>) -> &dyn TableProvider {
        table
            .downcast_ref::<CallScopedTable>()
            .expect("search table functions return a call-scoped provider")
            .inner
            .as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::{thread::sleep, time::Duration};

    use arrow_array::{Array, FixedSizeListArray, Int64Array, LargeStringArray, StructArray};
    use arrow_schema::Field;
    use bytes::Bytes;
    use datafusion::prelude::{col, lit};
    use object_store::{ObjectStore, ObjectStoreExt, PutPayload, memory, path::Path as ObjPath};
    use parquet::{
        arrow::ArrowWriter,
        file::{
            metadata::{PageIndexPolicy, ParquetMetaDataReader},
            properties::WriterProperties,
        },
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig},
            fts::reader::{Bm25SearchOptions, BoolMode},
            vector::{distance::Metric, layout::VectorLayout, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            query::fts::TIE_NARROW_BATCH,
            reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore},
        },
        test_helpers::{
            build_title_batch, decimal128_id_field, decimal128_ids, default_supertable_options,
        },
    };

    /// When far more hits tie at the k-th score than there are places, the
    /// top-k is still exactly the smallest `_id`s among the ties: narrowing
    /// the ties by superfile `id_min` before resolving their ids changes
    /// which ids are read, never which hits are returned.
    #[test]
    fn a_tie_far_past_k_keeps_the_smallest_ids() {
        // Within one batch of superfiles, and across several: one row a
        // superfile puts the smallest 20 ids in 20 superfiles, past the first
        // batch, so a walk that stops too early returns larger ids.
        smallest_ids_survive_a_tie(12, 20, 5);
        const _: () = assert!(20 > TIE_NARROW_BATCH);
        smallest_ids_survive_a_tie(100, 1, 20);
    }

    /// `superfiles` commits of `rows` equal titles each: BM25's top `k` are
    /// the `k` smallest `_id`s among them.
    fn smallest_ids_survive_a_tie(superfiles: usize, rows: usize, k: usize) {
        let st = Supertable::create(default_supertable_options()).expect("create");
        for _ in 0..superfiles {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&vec!["tie"; rows]))
                .expect("append");
            w.commit().expect("commit");
        }
        let ids = |batches: Vec<RecordBatch>| -> Vec<i128> {
            batches
                .iter()
                .flat_map(|b| {
                    let col = b
                        .column_by_name("_id")
                        .expect("_id")
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .expect("decimal ids")
                        .clone();
                    (0..col.len()).map(move |i| col.value(i))
                })
                .collect()
        };
        let mut all = ids(st
            .token_match("title", "tie", BoolMode::Or, Some(&["_id"]))
            .expect("token_match"));
        assert_eq!(all.len(), superfiles * rows);
        all.sort_unstable();
        let mut top = ids(st
            .bm25_search("title", "tie", k, Bm25SearchOptions::new(), Some(&["_id"]))
            .expect("bm25_search"));
        top.sort_unstable();
        assert_eq!(
            top,
            all[..k],
            "the {k} smallest ids among {} ties",
            all.len()
        );
    }

    /// A take's prefetch covers the projected column chunks of exactly the
    /// row groups its rows fall in, and declines a column the file lacks.
    #[test]
    fn take_prefetch_spans_cover_the_hit_row_groups_projected_chunks() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
        ]));
        let col =
            |m: i64| Arc::new(Int64Array::from_iter_values((0..1000).map(|i| i * m))) as ArrayRef;
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![col(1), col(2), col(3)]).expect("batch");
        let mut bytes = Vec::new();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .build();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&Bytes::from(bytes))
            .expect("metadata");
        assert_eq!(metadata.num_row_groups(), 10);

        let rows = [5u32, 150, 151, 950];
        let spans = take_prefetch_spans(&metadata, &rows, &[0, 2]).expect("spans");
        let covered = |(offset, len): (u64, u64)| {
            spans
                .iter()
                .any(|s| s.start <= offset && offset + len <= s.end)
        };
        for rg in [0, 1, 9] {
            for c in [0, 2] {
                assert!(
                    covered(metadata.row_group(rg).column(c).byte_range()),
                    "rg {rg} col {c}"
                );
            }
        }
        for rg in [2, 5, 8] {
            assert!(
                !covered(metadata.row_group(rg).column(0).byte_range()),
                "rg {rg} not hit"
            );
        }
        assert!(
            !covered(metadata.row_group(0).column(1).byte_range()),
            "an unprojected column is not fetched"
        );
        assert!(take_prefetch_spans(&metadata, &rows, &[3]).is_none());
    }

    /// With the page index loaded, a take's prefetch covers only the pages
    /// that hold its rows, plus each chunk's dictionary page.
    #[test]
    fn take_prefetch_spans_cover_only_the_hit_pages_with_a_page_index() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let values = Arc::new(Int64Array::from_iter_values((0..1000).map(|i| i % 7))) as ArrayRef;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).expect("batch");
        let mut bytes = Vec::new();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .set_data_page_row_count_limit(10)
            .set_write_batch_size(10)
            .build();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let metadata = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&Bytes::from(bytes))
            .expect("metadata");
        let pages =
            |rg: usize| metadata.offset_index().expect("page index")[rg][0].page_locations();
        assert_eq!(pages(0).len(), 10, "ten pages a row group");

        let rows = [5u32, 37, 150];
        let spans = take_prefetch_spans(&metadata, &rows, &[0]).expect("spans");
        let covered = |offset: u64, len: u64| {
            spans
                .iter()
                .any(|s| s.start <= offset && offset + len <= s.end)
        };
        let page = |rg: usize, i: usize| {
            let p = &pages(rg)[i];
            (p.offset as u64, p.compressed_page_size as u64)
        };
        for (rg, i) in [(0, 0), (0, 3), (1, 5)] {
            let (offset, len) = page(rg, i);
            assert!(covered(offset, len), "rg {rg} page {i} holds a hit");
        }
        for (rg, i) in [(0, 1), (0, 2), (1, 4), (2, 0)] {
            let (offset, len) = page(rg, i);
            assert!(!covered(offset, len), "rg {rg} page {i} holds none");
        }
        let (chunk_start, _) = metadata.row_group(0).column(0).byte_range();
        let first_page = pages(0)[0].offset as u64;
        assert!(chunk_start < first_page, "the chunk has a dictionary page");
        assert!(
            covered(chunk_start, first_page - chunk_start),
            "the dictionary page"
        );
    }

    /// A struct before a projected column fills two leaf chunks, so the
    /// projected column's chunk is found by its leaf, not its Arrow index;
    /// a projected struct brings every leaf under it.
    #[test]
    fn take_prefetch_spans_find_a_column_after_a_struct_by_its_leaves() {
        let leaf =
            |m: i64| Arc::new(Int64Array::from_iter_values((0..100).map(|i| i * m))) as ArrayRef;
        let s = StructArray::from(vec![
            (Arc::new(Field::new("x", DataType::Int64, false)), leaf(1)),
            (Arc::new(Field::new("y", DataType::Int64, false)), leaf(2)),
        ]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", s.data_type().clone(), false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(s), leaf(3)]).expect("batch");
        let mut bytes = Vec::new();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(50))
            .build();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(props)).expect("writer");
        writer.write(&batch).expect("write");
        writer.close().expect("close");
        let metadata = ParquetMetaDataReader::new()
            .parse_and_finish(&Bytes::from(bytes))
            .expect("metadata");
        let rg = metadata.row_group(0);
        assert_eq!(rg.num_columns(), 3, "s.x, s.y, b");
        let covered = |spans: &[Range<u64>], leaf: usize| {
            let (offset, len) = rg.column(leaf).byte_range();
            spans
                .iter()
                .any(|s| s.start <= offset && offset + len <= s.end)
        };

        let b = take_prefetch_spans(&metadata, &[3], &[1]).expect("spans");
        assert!(covered(&b, 2), "b is leaf 2");
        assert!(!covered(&b, 1), "s.y is not projected");
        let s = take_prefetch_spans(&metadata, &[3], &[0]).expect("spans");
        assert!(covered(&s, 0) && covered(&s, 1), "both leaves of s");
        assert!(!covered(&s, 2), "b is not projected");
        assert!(take_prefetch_spans(&metadata, &[3], &[2]).is_none());
    }

    /// Force Snowflake ids in one committed superfile across an ms boundary.
    const ID_GAP_WAIT: Duration = Duration::from_millis(20);

    /// A table far larger than any ceiling, so `total` is never the binding
    /// limit in the ceiling tests below.
    const HUGE_TABLE: usize = 100_000_000;

    /// The over-fetch ceiling is bounded from above in every direction a
    /// caller can push it, and never below the `k` that was asked for.
    /// Before it existed the fill grew to the table's row count, so an
    /// unbounded predicate that almost nothing satisfies ended by asking
    /// the kernel for a table-sized heap.
    #[test]
    fn over_fetch_ceiling_is_bounded_and_never_below_k() {
        // A small k takes the floor, not the (tiny) multiple.
        assert_eq!(over_fetch_ceiling(1, HUGE_TABLE), OVER_FETCH_MIN_CEILING);
        assert_eq!(over_fetch_ceiling(10, HUGE_TABLE), OVER_FETCH_MIN_CEILING);
        // Once the multiple clears the floor it governs.
        assert_eq!(over_fetch_ceiling(1_000, HUGE_TABLE), 64_000);
        // ... until the absolute cap does.
        assert_eq!(over_fetch_ceiling(5_000, HUGE_TABLE), OVER_FETCH_MAX_HITS);
        // A caller asking for more than the cap still gets its own k: the
        // ceiling bounds the over-fetch, never the request.
        let big = OVER_FETCH_MAX_HITS * 2;
        assert_eq!(over_fetch_ceiling(big, HUGE_TABLE), big);
        // The table is always the last word.
        assert_eq!(over_fetch_ceiling(10, 7), 7);
        assert_eq!(over_fetch_ceiling(0, 0), 0);
    }

    #[test]
    fn arg_to_string_accepts_utf8_literal_rejects_int() {
        assert_eq!(
            arg_to_string(&lit("emb"), "column").expect("utf8 literal"),
            "emb"
        );
        assert!(arg_to_string(&lit(3_i64), "column").is_err());
    }

    #[test]
    fn arg_to_string_accepts_large_utf8_and_utf8_view() {
        let large = Expr::Literal(ScalarValue::LargeUtf8(Some("body".into())), None);
        assert_eq!(arg_to_string(&large, "column").expect("large utf8"), "body");
        let view = Expr::Literal(ScalarValue::Utf8View(Some("title".into())), None);
        assert_eq!(arg_to_string(&view, "column").expect("utf8 view"), "title");
    }

    #[test]
    fn arg_to_usize_accepts_int_rejects_negative_and_nonint() {
        assert_eq!(arg_to_usize(&lit(10_i64), "k").expect("int literal"), 10);
        assert!(arg_to_usize(&lit(-1_i64), "k").is_err());
        assert!(arg_to_usize(&lit("nope"), "k").is_err());
    }

    #[test]
    fn arg_to_usize_accepts_all_integer_widths() {
        let i32e = Expr::Literal(ScalarValue::Int32(Some(7)), None);
        let u64e = Expr::Literal(ScalarValue::UInt64(Some(8)), None);
        let u32e = Expr::Literal(ScalarValue::UInt32(Some(9)), None);
        assert_eq!(arg_to_usize(&i32e, "k").expect("i32"), 7);
        assert_eq!(arg_to_usize(&u64e, "k").expect("u64"), 8);
        assert_eq!(arg_to_usize(&u32e, "k").expect("u32"), 9);
    }

    #[test]
    fn output_schema_appends_score() {
        let s = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, true)]));
        let out = output_schema_with_score(&s);
        assert_eq!(out.fields().len(), 2);
        assert_eq!(out.field(1).name(), "score");
        assert_eq!(out.field(1).data_type(), &DataType::Float32);
    }

    /// [`validate_projection`] is the shared, resident-only (no I/O)
    /// projection-name check the search entry points run *before* any search,
    /// so a doomed projection fails fast. It resolves known names to output
    /// indices, defaults `None` to `_id` + `score`, and rejects an unknown
    /// name with the exact `unknown column … in projection` message
    /// `resolve_hits_named` produces at materialization.
    #[test]
    fn validate_projection_rejects_unknown_and_resolves_known() {
        let output_schema = output_schema_with_score(&Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Int64, false),
            Field::new("title", DataType::LargeUtf8, false),
        ])));
        // score is appended after _id (0) and title (1).
        assert_eq!(
            validate_projection(None, "_id", &output_schema).expect("default is valid"),
            vec![0, 2],
            "None resolves to the engine-native _id + score pair"
        );
        assert_eq!(
            validate_projection(Some(&["title", "_id", "score"]), "_id", &output_schema)
                .expect("known columns resolve"),
            vec![1, 0, 2],
            "known names resolve to their output-schema indices in order"
        );
        let err = validate_projection(Some(&["title", "nope"]), "_id", &output_schema)
            .expect_err("an unknown projected column must error");
        assert!(
            matches!(err, QueryError::InvalidQuery(_)),
            "expected InvalidQuery, got {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("unknown column") && msg.contains("in projection"),
            "message must name the unknown column and the projection, got {msg:?}"
        );
    }

    /// The pushed predicate binds to the function's output by bare column
    /// name — a table alias on the reference is dropped — and reports the
    /// columns it reads; a reference to a column the output lacks, or no
    /// filter at all, compiles to nothing.
    #[test]
    fn pushed_predicate_compiles_against_the_output_schema() {
        let out = output_schema_with_score(&Arc::new(Schema::new(vec![
            Field::new("_id", DataType::Int64, false),
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("bucket", DataType::Int64, false),
        ])));
        // Literals arrive type-coerced from the planner (a `LargeUtf8`
        // column compares against a `LargeUtf8` literal), so the test hands
        // over the same shape.
        let qualified = [
            col("t.bucket").eq(lit(1_i64)),
            col("t.title").eq(lit(ScalarValue::LargeUtf8(Some("x".into())))),
        ];
        let predicate = PushedPredicate::compile(&qualified, &out).expect("compiles");
        let mut columns = predicate.columns.clone();
        columns.sort_unstable();
        assert_eq!(columns, vec![1, 2], "title and bucket, by output index");
        assert!(
            predicate
                .conjunction
                .column_refs()
                .iter()
                .all(|c| c.relation.is_none()),
            "qualifiers are stripped: {:?}",
            predicate.conjunction
        );
        // Bound to a projection carrying its columns, it evaluates.
        let projected = Arc::new(out.project(&[2, 1]).expect("project"));
        let bound = predicate.bind(&projected).expect("binds");
        let batch = RecordBatch::try_new(
            Arc::clone(&projected),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 1])) as ArrayRef,
                Arc::new(LargeStringArray::from(vec!["x", "x", "y"])) as ArrayRef,
            ],
        )
        .expect("batch");
        let mask = bound.mask(&batch).expect("mask");
        assert_eq!(
            (0..3).map(|i| mask.value(i)).collect::<Vec<_>>(),
            vec![true, false, false]
        );

        assert!(
            PushedPredicate::compile(&[], &out).is_none(),
            "nothing pushed"
        );
        assert!(
            PushedPredicate::compile(&[col("elsewhere").eq(lit(1_i64))], &out).is_none(),
            "a column the output lacks cannot be checked here"
        );
    }

    // ---- harness exercising resolve_hits_named / resolve_ids_arithmetic /
    //      resolve_columns through the public search methods ----

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

    fn demo(dim: usize) -> Supertable {
        let st = Supertable::create(options_title_emb(dim)).expect("create");
        let mut w = st.writer().expect("writer");
        let schema = st.options().schema.clone();
        w.append(&build_batch(
            &["rust async", "python data", "rust systems", "go routines"],
            dim,
            schema,
        ))
        .expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn resolve_hits_named_id_only_takes_arithmetic_fast_path() {
        // `_id`-only projection drives resolve_hits_named → the
        // resolve_ids_arithmetic no-I/O fast path (single contiguous
        // span: id_max - id_min + 1 == n_docs).
        let st = demo(16);
        let batches = st
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["_id"]),
            )
            .expect("bm25_search _id");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 1);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.num_rows(), 2, "two docs contain 'rust'");
    }

    #[test]
    fn resolve_hits_named_default_is_id_and_score() {
        // `None` projection is the engine-native `_id` + `score` pair.
        let st = demo(16);
        let batches = st
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                None,
            )
            .expect("bm25_search default");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 2);
        assert_eq!(b.schema().field(0).name(), "_id");
        assert_eq!(b.schema().field(1).name(), "score");
    }

    #[test]
    fn resolve_hits_named_scalar_column_decodes_via_resolve_columns() {
        // Naming a scalar column (`title`) forces resolve_columns to
        // decode the column bytes; `score` synthesized alongside.
        let st = demo(16);
        let batches = st
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["_id", "title", "score"]),
            )
            .expect("bm25_search title");
        let b = &batches[0];
        assert_eq!(b.num_columns(), 3);
        let titles = b
            .column(1)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        for i in 0..titles.len() {
            assert!(titles.value(i).contains("rust"));
        }
    }

    #[test]
    fn resolve_hits_named_unknown_column_errors() {
        let st = demo(16);
        let res = st.reader().expect("reader").bm25_search(
            "title",
            "rust",
            10,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
            Some(&["nope"]),
        );
        assert!(res.is_err(), "unknown projected column must error");
    }

    #[test]
    fn resolve_hits_named_empty_hits_returns_empty_batch() {
        let st = demo(16);
        let batches = st
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "nonexistentterm",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["_id"]),
            )
            .expect("bm25_search empty");
        let total: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total, 0);
    }

    // ---- resolve_hits direct: projection branches that the named
    //      wrapper never reaches (score-only, empty projection) ----

    /// Two hits against the demo superfile, with deliberately distinct
    /// scores so the synthesized `score` column can be checked.
    fn two_hits(reader: &SupertableReader) -> Vec<SuperfileHit> {
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        vec![
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: 0,
                score: 1.5,
                stable_id: None,
            },
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: (entry.n_docs - 1) as u32,
                score: 0.5,
                stable_id: None,
            },
        ]
    }

    #[test]
    fn resolve_hits_score_only_synthesizes_score_without_decoding_scalars() {
        // Projecting just the trailing `score` index decodes no scalar
        // columns (the cost-first "open no readers" branch): `needed`
        // is empty, `score` is synthesized straight from the hits.
        let st = demo(16);
        let reader = st.reader().expect("reader");
        let hits = two_hits(&reader);
        let scalar_schema = reader.manifest().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);
        let score_idx = scalar_schema.fields().len();

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                Some(&[score_idx]),
            ))
            .expect("score-only resolve");

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), SCORE_COLUMN);
        assert_eq!(batch.num_rows(), hits.len());
        let scores = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score col");
        assert_eq!(scores.value(0), 1.5);
        assert_eq!(scores.value(1), 0.5);
    }

    #[test]
    fn resolve_hits_none_projection_returns_all_scalar_columns_and_score() {
        // `projection: None` (distinct from `resolve_hits_named`'s
        // `_id`+`score` default) materializes every scalar column plus
        // the trailing synthesized `score`, in schema order.
        let st = demo(16);
        let reader = st.reader().expect("reader");
        let mut hits = two_hits(&reader);
        reader
            .block_on(
                crate::supertable::query::dispatch::attach_stable_ids_to_hits(&reader, &mut hits),
            )
            .expect("stamp stable ids before scalar resolution");
        let scalar_schema = reader.manifest().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                None,
            ))
            .expect("none-projection resolve");

        assert_eq!(batch.num_columns(), output_schema.fields().len());
        assert_eq!(batch.num_rows(), hits.len());
        let last = batch.num_columns() - 1;
        assert_eq!(batch.schema().field(last).name(), SCORE_COLUMN);
        assert_eq!(batch.schema().field(0).name(), "_id");
    }

    #[test]
    fn resolve_hits_empty_projection_preserves_hit_row_count() {
        // A zero-column projection (the `COUNT(*)` shape) emits no
        // columns but must still report `hits.len()` rows — the
        // `with_row_count` path.
        let st = demo(16);
        let reader = st.reader().expect("reader");
        let hits = two_hits(&reader);
        let scalar_schema = reader.manifest().scalar_schema();
        let output_schema = output_schema_with_score(&scalar_schema);

        let batch = reader
            .block_on(resolve_hits(
                &reader,
                &hits,
                &scalar_schema,
                &output_schema,
                Some(&[]),
            ))
            .expect("empty-projection resolve");

        assert_eq!(batch.num_columns(), 0);
        assert_eq!(batch.num_rows(), hits.len());
    }

    // ---- resolve_ids_arithmetic direct: the no-I/O span arithmetic
    //      and its fallback when the span check can't apply ----

    /// FTS-only table (no vector column): commits stay id-ordered, so the
    /// span-arithmetic fast path applies. Vector commits are cell-packed
    /// (`MultiCellIvf`, Parquet reordered by cell) and are covered by
    /// [`resolve_ids_arithmetic_declines_cell_packed_vector_superfiles`].
    fn demo_fts_only() -> Supertable {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("pool"),
        );
        let schema = Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )]));
        let opts = SupertableOptions::new(schema.clone(), vec![FtsConfig::new("title")], vec![])
            .expect("valid options")
            .with_writer_pool(pool);
        let st = Supertable::create(opts).expect("create");
        let mut w = st.writer().expect("writer");
        let titles = LargeStringArray::from(vec![
            "rust async",
            "python data",
            "rust systems",
            "go routines",
        ]);
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(titles) as ArrayRef]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
        st
    }

    #[test]
    fn resolve_ids_arithmetic_maps_local_ids_via_manifest_span() {
        // Single contiguous id-ordered commit => `id_max - id_min + 1 ==
        // n_docs`, so row `local` resolves to `id_min + local` with no file
        // read.
        let st = demo_fts_only();
        let reader = st.reader().expect("reader");
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        let last = (entry.n_docs - 1) as u32;
        let hits = vec![
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: 0,
                score: 0.0,
                stable_id: None,
            },
            SuperfileHit {
                superfile: entry.uri,
                local_doc_id: last,
                score: 0.0,
                stable_id: None,
            },
        ];

        let batch = resolve_ids_arithmetic(&reader, &hits)
            .expect("contiguous span => Some")
            .expect("ok batch");
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "_id");
        let ids = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal id col");
        assert_eq!(ids.value(0), entry.id_min);
        assert_eq!(ids.value(1), entry.id_min + i128::from(last));
    }

    #[test]
    fn resolve_ids_arithmetic_declines_cell_packed_vector_superfiles() {
        // Vector commits write cell-packed MultiCell superfiles whose Parquet
        // rows are cell-ordered, not id-ordered — span arithmetic must bail
        // to the id-page read even though the id span looks contiguous.
        let st = demo(16);
        let reader = st.reader().expect("reader");
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        assert_eq!(entry.vector_layout, VectorLayout::MultiCellIvf);
        let hits = vec![SuperfileHit {
            superfile: entry.uri,
            local_doc_id: 0,
            score: 0.0,
            stable_id: None,
        }];
        assert!(
            resolve_ids_arithmetic(&reader, &hits).is_none(),
            "cell-packed superfiles must fall back to the id-page read",
        );
    }

    #[test]
    fn resolve_ids_arithmetic_returns_none_when_superfile_absent() {
        // A hit naming a superfile not in the manifest fails the
        // lookup, so arithmetic bails to `None` and the caller falls
        // back to the id-page read.
        let st = demo(16);
        let reader = st.reader().expect("reader");
        let hits = vec![SuperfileHit {
            superfile: SuperfileUri::new_v4(),
            local_doc_id: 0,
            score: 0.0,
            stable_id: None,
        }];
        assert!(
            resolve_ids_arithmetic(&reader, &hits).is_none(),
            "unknown superfile must abandon the arithmetic fast path",
        );
    }

    // ---- take_rows_object_store: the cold/object-store row-resolution
    //      path, exercised directly against an in-memory object store ----

    /// Titles for the standalone superfile the cold-path tests stream
    /// rows out of. Five rows so a sub-selection genuinely skips some.
    const TITLES: [&str; 5] = ["alpha", "bravo", "charlie", "delta", "echo"];

    /// Build a standalone superfile (id + `title` scalar columns, no
    /// indexes) whose `title` rows are [`TITLES`], in row order.
    fn titled_superfile_bytes() -> Bytes {
        let schema: Arc<Schema> = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![]);
        let mut b = SuperfileBuilder::new(opts).expect("builder");
        let ids = decimal128_ids(0..TITLES.len() as u64);
        let title = LargeStringArray::from(TITLES.to_vec());
        let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(title)])
            .expect("build batch");
        b.add_batch(&batch, &[]).expect("add_batch");
        Bytes::from(b.finish().expect("finish builder"))
    }

    /// Open `bytes` eagerly just to read the parquet schema + row count
    /// that the cold path passes to [`take_rows_object_store`].
    fn schema_and_n_docs(bytes: &Bytes) -> (SchemaRef, u64) {
        let reader = SuperfileReader::open(bytes.clone()).expect("open");
        (Arc::clone(reader.schema()), reader.n_docs())
    }

    /// Put `bytes` into a fresh in-memory object store and return the
    /// handle + path the cold reader will range-GET against.
    async fn object_store_with(bytes: &Bytes) -> (Arc<dyn ObjectStore>, ObjPath) {
        let store: Arc<dyn ObjectStore> = Arc::new(memory::InMemory::new());
        let path = ObjPath::from("data/seg.sf.parquet");
        store
            .put(&path, PutPayload::from_bytes(bytes.clone()))
            .await
            .expect("put superfile into object store");
        (store, path)
    }

    /// Downcast the single returned `title` column to a string array.
    fn titles_of(batch: &RecordBatch) -> Vec<String> {
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    #[tokio::test]
    async fn take_rows_object_store_streams_rows_in_caller_rank_order() {
        // Out-of-order ids: rows are decoded sorted but ranked back into
        // the caller's order, so output row i is the i-th requested id.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[2, 0, 3],
            &["title"],
        )
        .await
        .expect("take rows");

        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "title");
        assert_eq!(titles_of(&batch), vec!["charlie", "alpha", "delta"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_ranks_back_duplicate_ids() {
        // The same id requested twice must appear twice in the output —
        // rows are decoded once (distinct) and gathered back per request.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[1, 1, 0],
            &["title"],
        )
        .await
        .expect("take rows");

        assert_eq!(titles_of(&batch), vec!["bravo", "bravo", "alpha"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_discovers_size_when_file_size_none() {
        // `None` file_size omits the `with_file_size` shortcut, so the
        // parquet reader discovers the size itself — same rows out.
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(store, path, None, &schema, n_docs, &[4], &["title"])
            .await
            .expect("take rows");

        assert_eq!(titles_of(&batch), vec!["echo"]);
    }

    #[tokio::test]
    async fn take_rows_object_store_empty_ids_returns_empty_batch() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[],
            &["title"],
        )
        .await
        .expect("empty ids");

        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.schema().field(0).name(), "title");
    }

    /// A projection naming only columns this file predates resolves to no
    /// columns at all. There is nothing to read, but the batch still has to
    /// report the rows the caller asked for: it null-fills to that count, and
    /// a batch saying it has none drops the hits instead.
    ///
    /// On the cold path specifically. The resident reader has its own arm for
    /// this, and a `memory://` table only ever takes that one, so a search
    /// through the public API cannot reach here — this is the path an
    /// object-store or cache-cold read uses, which is the one production
    /// takes.
    #[tokio::test]
    async fn take_rows_object_store_empty_projection_keeps_the_row_count() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;
        let ids: Vec<u32> = (0..n_docs as u32).collect();

        let batch = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &ids,
            &[],
        )
        .await
        .expect("a projection of no columns is a batch of no columns");

        assert_eq!(batch.num_columns(), 0, "nothing was asked for");
        assert_eq!(
            batch.num_rows(),
            ids.len(),
            "the rows asked for are still counted, so the caller null-fills them"
        );
    }

    #[tokio::test]
    async fn take_rows_object_store_out_of_range_id_errors() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let err = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[n_docs as u32],
            &["title"],
        )
        .await
        .expect_err("doc id past n_docs must error");
        assert!(
            err.to_string().contains("out of range"),
            "expected an out-of-range error, got {err}",
        );
    }

    #[tokio::test]
    async fn take_rows_object_store_unknown_column_errors() {
        let bytes = titled_superfile_bytes();
        let (schema, n_docs) = schema_and_n_docs(&bytes);
        let (store, path) = object_store_with(&bytes).await;

        let err = take_rows_object_store(
            store,
            path,
            Some(bytes.len() as u64),
            &schema,
            n_docs,
            &[0],
            &["nope"],
        )
        .await
        .expect_err("unknown column must error");
        assert!(
            err.to_string().contains("unknown column"),
            "expected an unknown-column error, got {err}",
        );
    }

    // ---- resolve_columns cold path: lazy (object-store) readers ----
    //
    // `demo()` keeps superfile bytes resident, so its readers are warm
    // and `resolve_columns` only ever takes the rayon decode branch. To
    // drive the cold branch (lazy readers → `take_rows_object_store`),
    // commit to durable storage, then *reopen* with a lazy disk cache:
    // the reopened handle's in-memory tier is empty, so every read cold-
    // fetches a lazy reader from object storage.

    /// Commit four titled docs to `storage` via a throwaway producer.
    fn commit_titles_to(storage: &Arc<dyn StorageProvider>) {
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("single writer pool"),
        );
        let producer = Supertable::create(
            default_supertable_options()
                .with_storage(Arc::clone(storage))
                .with_writer_pool(pool),
        )
        .expect("create producer");
        let mut w = producer.writer().expect("writer");
        w.append(&build_title_batch(&["rust async", "python data"]))
            .expect("first append");
        sleep(ID_GAP_WAIT);
        w.append(&build_title_batch(&["rust systems", "go routines"]))
            .expect("second append");
        w.commit().expect("commit");
        let reader = producer.reader().expect("reader");
        let entry = reader
            .manifest()
            .superfiles
            .first()
            .expect("one committed superfile");
        assert!(
            row_id_from_manifest_entry(entry, 0).is_none(),
            "fixture must force a gapped id span"
        );
    }

    /// A hit set spanning a gapped file and a contiguous one arrives part
    /// stamped (the contiguous file's hits get their id by arithmetic when
    /// tagged), so stamping reads `_id` for the gapped file's hits alone.
    /// Every hit must still end up with its own row's id: the term's ids
    /// equal a scan's.
    #[test]
    fn stamp_stable_ids_over_gapped_and_contiguous_files_matches_a_scan() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        commit_titles_to(&storage);
        let st = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("open");
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&["rust tooling", "java beans"]))
            .expect("append");
        w.commit().expect("commit");
        let reader = st.reader().expect("reader");
        let spans: Vec<bool> = reader
            .manifest()
            .superfiles
            .iter()
            .map(|e| row_id_from_manifest_entry(e, 0).is_some())
            .collect();
        assert_eq!(
            spans.len(),
            2,
            "one gapped file and one contiguous file, got {spans:?}"
        );
        assert!(
            spans.contains(&true) && spans.contains(&false),
            "one gapped file and one contiguous file, got {spans:?}"
        );

        // Each hit's title is read at its own (superfile, local), which the
        // stamp does not touch, so a pair holds only if the id is that row's.
        let hits = reader
            .token_match("title", "rust", BoolMode::And)
            .expect("token_match");
        let titles = reader
            .block_on(resolve_columns(&reader, &hits, &["title"]))
            .expect("read titles");
        let titles = titles
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title");
        let mut matched: Vec<(String, i128)> = hits
            .iter()
            .enumerate()
            .map(|(i, h)| (titles.value(i).to_string(), h.stable_id.expect("stamped")))
            .collect();
        matched.sort_unstable();
        let mut scanned: Vec<(String, i128)> = reader
            .query_sql("SELECT title, _id FROM supertable WHERE title LIKE '%rust%'")
            .expect("scan")
            .iter()
            .flat_map(|b| {
                let titles = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .expect("title");
                let ids = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .expect("decimal `_id`");
                (0..b.num_rows())
                    .map(|i| (titles.value(i).to_string(), ids.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect();
        scanned.sort_unstable();
        assert_eq!(scanned.len(), 3, "rust is in three rows");
        assert_eq!(matched, scanned, "each hit carries its own row's `_id`");
    }

    /// Stamping reads `_id` only for the hits that lack one. A hit that has
    /// one keeps it even when the call must read the `_id` column for another
    /// hit: its local can be a cell-postings position rather than a Parquet
    /// row, so reading `_id` there would replace a right id with a wrong one.
    #[test]
    fn stamp_stable_ids_reads_only_the_hits_without_an_id() {
        /// An id no row holds, so a hit still carrying it was left alone.
        const KEPT_ID: i128 = -7;
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
        commit_titles_to(&storage);
        let st =
            Supertable::open(default_supertable_options().with_storage(storage)).expect("open");
        let reader = st.reader().expect("reader");
        let entry = Arc::clone(&reader.manifest().superfiles[0]);
        let hit = |local_doc_id, stable_id| SuperfileHit {
            superfile: entry.uri,
            local_doc_id,
            score: 0.0,
            stable_id,
        };
        let mut hits = [hit(0, None), hit(1, Some(KEPT_ID))];
        reader
            .block_on(stamp_stable_ids(&reader, &mut hits))
            .expect("stamp");
        assert_eq!(
            hits[1].stable_id,
            Some(KEPT_ID),
            "a stamped hit is not read again"
        );

        let row_0 = reader
            .block_on(resolve_columns(&reader, &[hit(0, None)], &["_id"]))
            .expect("read row 0's `_id`");
        let row_0 = row_0
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal `_id`")
            .value(0);
        assert_eq!(
            hits[0].stable_id,
            Some(row_0),
            "an unstamped hit gets its own row's id"
        );
    }

    /// Reopen the supertable at `consumer_storage` with a lazy
    /// (`LazyForegroundWithBackgroundFill`) disk cache so reads resolve to
    /// lazy range-GET readers. The reopened handle's in-memory tier is
    /// empty, so every read cold-fetches.
    fn open_cold(
        consumer_storage: Arc<dyn StorageProvider>,
        cache_dir: &TempDir,
    ) -> (Arc<DiskCacheStore>, Supertable) {
        let cfg = DiskCacheConfig {
            cache_root: cache_dir.path().to_path_buf(),
            cold_fetch_mode: ColdFetchMode::LazyForegroundWithBackgroundFill,
            mmap_cold_threshold_secs: 0,
            mmap_sweep_interval_secs: 0,
            ..Default::default()
        };
        let cache =
            DiskCacheStore::new_unpinned(Arc::clone(&consumer_storage), cfg).expect("cache");
        let consumer = Supertable::open(
            default_supertable_options()
                .with_storage(consumer_storage)
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("open consumer");
        (cache, consumer)
    }

    /// Producer commits to local-FS storage; consumer reopens cold over
    /// the same storage. Returns the temp dirs (kept alive as RAII
    /// guards), the cache (for stats), and the cold consumer handle.
    fn cold_consumer() -> (TempDir, TempDir, Arc<DiskCacheStore>, Supertable) {
        let storage_dir = TempDir::new().expect("storage tempdir");
        let cache_dir = TempDir::new().expect("cache tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        commit_titles_to(&storage);
        let (cache, consumer) = open_cold(Arc::clone(&storage), &cache_dir);
        (storage_dir, cache_dir, cache, consumer)
    }

    #[test]
    fn resolve_columns_cold_path_streams_scalar_via_object_store() {
        // Naming a non-id scalar column forces resolve_columns; the lazy
        // readers route it through take_rows_object_store (range GETs),
        // and `score` is synthesized alongside in rank order.
        let (_sd, _cd, cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["title", "score"]),
            )
            .expect("cold bm25 with scalar projection");

        let b = &batches[0];
        assert_eq!(b.num_columns(), 2);
        assert_eq!(b.schema().field(0).name(), "title");
        assert_eq!(b.schema().field(1).name(), "score");
        let titles = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title col");
        assert_eq!(titles.len(), 2, "two docs contain 'rust'");
        for i in 0..titles.len() {
            assert!(titles.value(i).contains("rust"));
        }
        // The reopened consumer's in-memory tier is empty, so resolving
        // the rows genuinely cold-fetched through the disk cache — this
        // is what put us on the cold branch.
        assert!(
            cache.stats().n_cold_fetches >= 1,
            "scalar resolution must cold-fetch lazy readers; got {}",
            cache.stats().n_cold_fetches,
        );
    }

    /// An aggregate over the match function that projects scalar columns
    /// only reads each superfile's rows through the lazy take on a cold
    /// reopen, and leaves a deleted row out as the hit path does: it equals
    /// the same aggregate over the materialized rows.
    #[test]
    fn match_columns_cold_path_leaves_deleted_rows_out() {
        let storage_dir = TempDir::new().expect("storage tempdir");
        let cache_dir = TempDir::new().expect("cache tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(storage_dir.path()).expect("provider"));
        commit_titles_to(&storage);
        let producer =
            Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("open producer");
        let stats = producer
            .delete(col("title").eq(lit("rust systems")))
            .expect("delete");
        assert_eq!(stats.matched(), 1);
        drop(producer);

        let (cache, consumer) = open_cold(Arc::clone(&storage), &cache_dir);
        let reader = consumer.reader().expect("reader");
        let scalar_only = reader
            .query_sql(
                "SELECT count(title) AS n, max(length(title)) AS m \
                 FROM token_match('title', 'rust')",
            )
            .expect("aggregate over the match");
        assert!(
            cache.stats().n_cold_fetches >= 1,
            "the scalar-only take must cold-fetch lazy readers; got {}",
            cache.stats().n_cold_fetches,
        );
        let via_rows = reader
            .query_sql(
                "SELECT count(title) AS n, max(length(title)) AS m \
                 FROM (SELECT title FROM token_match('title', 'rust') ORDER BY _id)",
            )
            .expect("aggregate over the rows");
        assert_eq!(format!("{scalar_only:?}"), format!("{via_rows:?}"));
        let n = scalar_only[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("count is Int64")
            .value(0);
        assert_eq!(n, 1, "only the live 'rust' row is counted");
    }

    #[test]
    fn resolve_id_and_scalar_cold_path_uses_final_hit_stamps() {
        let (_sd, _cd, _cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["_id", "title", "score"]),
            )
            .expect("cold bm25 with id and scalar projection");

        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.schema().field(0).name(), "_id");
        assert_eq!(batch.schema().field(1).name(), "title");
        assert_eq!(batch.schema().field(2).name(), "score");
    }

    #[test]
    fn resolve_columns_cold_path_empty_hits_opens_no_readers() {
        // A query that matches nothing produces no hits, so resolve_hits
        // short-circuits before resolve_columns ever opens a reader — no
        // cold-fetch is issued for the (absent) scalar resolution.
        let (_sd, _cd, _cache, consumer) = cold_consumer();

        let batches = consumer
            .reader()
            .expect("reader")
            .bm25_search(
                "title",
                "nonexistentterm",
                10,
                Bm25SearchOptions::new().with_mode(BoolMode::Or),
                Some(&["title", "score"]),
            )
            .expect("cold bm25 with no matches");

        let rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, 0);
    }
}
