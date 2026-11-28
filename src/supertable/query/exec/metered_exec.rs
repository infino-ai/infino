// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`MeteredExec`] — a pass-through `ExecutionPlan` that charges its
//! child's on-CPU time to the op that ran the query.
//!
//! DataFusion reports per-operator time as `elapsed_compute`, an `Instant`
//! timer around synchronous poll sections. That is *wall* time: on a busy
//! host it includes whatever the thread was descheduled for, which is the
//! same contention bleed that per-op metering exists to remove — and the
//! Parquet decode inside a scan node is not in it at all. Search paths, by
//! contrast, fold `thread_cpu_ns` (schedstat). Pricing both through one
//! meter while they read different clocks makes a SQL query and a vector
//! query incomparable.
//!
//! This node closes that: it wraps a child plan and brackets **each poll**
//! of each output partition with the thread-CPU clock, folding the delta
//! into the query's own collector. DataFusion is pull-based, so a poll
//! synchronously drives that partition's operator subtree — decode
//! included — on the polling thread. An `await` on I/O yields out of the
//! poll, so waiting is excluded and only real on-CPU time is counted,
//! which is exactly the contract the search kernels already meet.
//!
//! Placed at BOTH the plan root and around each table scan, because
//! neither alone is enough: DataFusion spawns a task per partition, so a
//! root-only bracket misses work under an internal spawn boundary, while a
//! scan-only bracket misses the aggregation and sort work above it. The
//! shared bracket depth in `op_stats` keeps the overlap from counting
//! twice — it gates every CPU fold, including the search kernels' own
//! brackets, which a poll of this node drives inline. The
//! collector's counters are atomics, so concurrent partitions fold safely.
//!
//! Everything the planner asks of this node is delegated to the child. A
//! wrapper that answers those questions for itself is not transparent: the
//! `ExecutionPlan` defaults report unknown statistics and refuse filter
//! pushdown, and sitting between an aggregate and the scan that way stops
//! `COUNT`/`MIN`/`MAX` folding from manifest statistics and turns an O(1)
//! manifest read into a full columnar scan. Measuring a query must not
//! change the query.

use std::{
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::{
    common::{Statistics, config::ConfigOptions},
    datasource::{physical_plan::FileScanConfig, source::DataSourceExec},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{PhysicalExpr, PhysicalSortExpr},
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
        SendableRecordBatchStream, SortOrderPushdownResult,
        execution_plan::CardinalityEffect,
        filter_pushdown::{
            ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
        },
        projection::ProjectionExec,
    },
};
use futures::Stream;

use crate::{
    runtime_metrics::op_stats::{OpStatsCollector, timed_poll},
    supertable::query::{
        provider::ScanFooters, read_ahead::ScanReadAhead, scan_order::can_read_in_file_order,
    },
};

/// What a meter does with a `LIMIT` fetch from above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Limits {
    /// Passes it on to the scan.
    Pass,
    /// Keeps it above; see [`MeteredExec::without_limit_pushdown`].
    Refuse,
    /// Passes it on to a scan that runs its whole `WHERE` as a row filter;
    /// see [`MeteredExec::passing_limits`].
    PassToRowFilter,
}

/// Wraps `input`, metering every partition's poll time into `op_stats`.
#[derive(Debug)]
pub(crate) struct MeteredExec {
    input: Arc<dyn ExecutionPlan>,
    op_stats: Option<Arc<OpStatsCollector>>,
    /// Footers of the scan below, when it is a table scan. `RowFilterUnderTopK`
    /// reads column sizes from them.
    footers: Option<Arc<ScanFooters>>,
    /// Read-ahead for the table scan below, if it is one. The meter hands it
    /// the scan's filter when the scan runs, and `ReadAheadUnderLimit` holds
    /// it to what the scan has read under a `LIMIT` or top-k.
    read_ahead: Option<Arc<ScanReadAhead>>,
    /// Whether the planner may carry a `LIMIT` fetch past this node into
    /// the child. The provider's meter around a scan that carries filters
    /// refuses it — see [`Self::without_limit_pushdown`].
    limits: Limits,
    /// What the wrapped scan stands for, shown after the node's name in
    /// `EXPLAIN`.
    note: Option<Arc<str>>,
}

impl MeteredExec {
    /// Wrap `input`. With no collector this is still a pass-through node —
    /// the per-poll bracket short-circuits on the same `metering_active`
    /// gate the kernels use, so an unmetered query pays one relaxed load
    /// per poll and no procfs reads.
    pub(crate) fn new(
        input: Arc<dyn ExecutionPlan>,
        op_stats: Option<Arc<OpStatsCollector>>,
    ) -> Self {
        Self {
            input,
            op_stats,
            footers: None,
            read_ahead: None,
            limits: Limits::Pass,
            note: None,
        }
    }

    /// Wrap `input` and stop `LimitPushdown` from embedding a fetch below
    /// this node. The table provider uses this around a filtered scan:
    /// its access plans carry tombstone (and index-candidate) row
    /// selections, and a scan-level limit lets the Parquet opener's limit
    /// pruning replace those selections with whole row groups that the
    /// predicate's statistics prove "fully matching" — returning deleted
    /// rows. With a `FilterExec` above the scan the fetch lives there. An
    /// exact filter leaves no `FilterExec`, so this node refuses the fetch
    /// and the limit node above keeps it. The provider pushes its own
    /// scan-level limit only for filter-less scans, where no predicate can
    /// mark a row group fully matching. `RowFilterUnderTopK` lifts the
    /// refusal for a scan with no selections ([`Self::passing_limits`]).
    pub(crate) fn without_limit_pushdown(
        input: Arc<dyn ExecutionPlan>,
        op_stats: Option<Arc<OpStatsCollector>>,
    ) -> Self {
        Self {
            input,
            op_stats,
            footers: None,
            read_ahead: None,
            limits: Limits::Refuse,
            note: None,
        }
    }

    /// This meter with the footers of the table scan it wraps.
    pub(crate) fn with_footers(mut self, footers: Arc<ScanFooters>) -> Self {
        self.footers = Some(footers);
        self
    }

    /// This meter, with read-ahead for the table scan it wraps.
    pub(crate) fn with_read_ahead(mut self, read_ahead: Arc<ScanReadAhead>) -> Self {
        self.read_ahead = Some(read_ahead);
        self
    }

    /// This meter, showing `note` after its name in `EXPLAIN`.
    pub(crate) fn with_note(mut self, note: Option<Arc<str>>) -> Self {
        self.note = note;
        self
    }

    /// Read-ahead for the table scan this meter wraps, if any.
    pub(crate) fn read_ahead(&self) -> Option<&Arc<ScanReadAhead>> {
        self.read_ahead.as_ref()
    }

    /// The node this meter wraps.
    pub(crate) fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Footers of the table scan this meter wraps, if any.
    pub(crate) fn footers(&self) -> Option<&ScanFooters> {
        self.footers.as_deref()
    }

    /// A meter over `input` with this node's collector, which lets a `LIMIT`
    /// fetch through to it. For a filtered scan that runs its whole `WHERE`
    /// as a row filter and reads every row of the row groups it reads, so the
    /// fetch counts rows that pass and no selection can be lost. Each
    /// partition of such a scan stops after the fetch, so it gains nothing
    /// from more partitions, and an exchange below this node would keep the
    /// fetch from reaching it.
    ///
    /// Given the limit, the Parquet reader reads only the row groups its
    /// statistics prove match, skipping earlier ones, and which it skips
    /// depends on the limit, so pages of one statement would skip or repeat
    /// rows. `ScanOrderUnderLimit` reads such a scan in file order once the
    /// limits are placed; a scan that can't read in file order is refused the
    /// limit, which stays above this meter.
    pub(crate) fn passing_limits(&self, input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(MeteredExec {
            limits: Limits::PassToRowFilter,
            ..self.rewrapped(input)
        })
    }

    /// Whether this meter passes a `LIMIT` to a scan that runs its whole
    /// `WHERE` as a row filter ([`Self::passing_limits`]).
    pub(crate) fn passes_limits_to_row_filter(&self) -> bool {
        self.limits == Limits::PassToRowFilter
    }

    /// Whether a `LIMIT` fetch may pass this meter into its scan: not for a
    /// scan with filters ([`Self::without_limit_pushdown`]), and for one that
    /// runs its `WHERE` as a row filter only when it can read in file order
    /// ([`Self::passing_limits`]).
    fn passes_limits(&self) -> bool {
        match self.limits {
            Limits::Pass => true,
            Limits::Refuse => false,
            Limits::PassToRowFilter => can_read_in_file_order(&self.input),
        }
    }

    /// A meter over `input` with this node's collector and limit policy.
    fn rewrap(&self, input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(self.rewrapped(input))
    }

    /// [`Self::rewrap`], unboxed.
    fn rewrapped(&self, input: Arc<dyn ExecutionPlan>) -> MeteredExec {
        MeteredExec {
            input,
            op_stats: self.op_stats.clone(),
            footers: self.footers.clone(),
            read_ahead: self.read_ahead.clone(),
            limits: self.limits,
            note: self.note.clone(),
        }
    }
}

impl DisplayAs for MeteredExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.note {
            Some(note) => write!(f, "MeteredExec: {note}"),
            None => write!(f, "MeteredExec"),
        }
    }
}

impl ExecutionPlan for MeteredExec {
    fn name(&self) -> &'static str {
        "MeteredExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Optimizer rewrites keep the meter attached to whatever the child
        // became; dropping it here would silently unmeter the scan. A unary
        // node handed anything but one child is a broken rewrite — say so,
        // rather than papering over it by keeping the stale subtree.
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "MeteredExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(self.rewrap(children.swap_remove(0)))
    }

    // ---- Everything below is delegation. See the module header: a meter
    // that answers the planner for itself changes the plan it measures. ----

    fn maintains_input_order(&self) -> Vec<bool> {
        // Rows leave in the order the child produced them.
        vec![true; self.children().len()]
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        // The default is `Statistics::new_unknown`, which would hide the
        // manifest statistics the provider attaches and stop the aggregate
        // rule folding COUNT/MIN/MAX into a constant.
        self.input.partition_statistics(partition)
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        config: &ConfigOptions,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        // A scan that can split itself across partitions should still do so
        // with the meter on. Refusing here doesn't prevent parallelism — it
        // makes the planner insert a `RepartitionExec` *below* this node
        // instead, which both adds an exchange and moves the Parquet decode
        // into a spawned task where this node's thread clock cannot see it.
        Ok(self
            .input
            .repartitioned(target_partitions, config)?
            .map(|input| self.rewrap(input)))
    }

    fn supports_limit_pushdown(&self) -> bool {
        // `false` makes `LimitPushdown` keep its limit node above this
        // meter instead of walking into the scan; see
        // [`Self::passes_limits`].
        self.passes_limits()
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        // See [`Self::passing_limits`].
        vec![self.limits != Limits::PassToRowFilter]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(self
            .input
            .try_swapping_with_projection(projection)?
            .map(|input| self.rewrap(input)))
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DfResult<FilterDescription> {
        // The default bars every parent filter, which leaves a redundant
        // `FilterExec` above a scan that could have pushed the predicate
        // down into the Parquet reader.
        FilterDescription::from_children(parent_filters, &self.children())
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DfResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        Ok(FilterPushdownPropagation::if_all(child_pushdown_result))
    }

    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DfResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        // On today's plans the sort sinks BELOW this node —
        // `maintains_input_order` above is what lets EnforceSorting do
        // that — so the reorder and reverse-scan wins come from that hook
        // and this one is rarely consulted. It still must delegate: the
        // default answer is `Unsupported`, which stops the pushdown walk
        // dead for any shape where the sort cannot sink, and a blocked
        // walk costs that shape its row-group reorder and reverse scan.
        Ok(match self.input.try_pushdown_sort(order)? {
            SortOrderPushdownResult::Exact { inner } => SortOrderPushdownResult::Exact {
                inner: self.rewrap(inner),
            },
            SortOrderPushdownResult::Inexact { inner } => SortOrderPushdownResult::Inexact {
                inner: self.rewrap(inner),
            },
            SortOrderPushdownResult::Unsupported => SortOrderPushdownResult::Unsupported,
        })
    }

    fn fetch(&self) -> Option<usize> {
        self.input.fetch()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        // When the walk may pass, `supports_limit_pushdown` above routes
        // `LimitPushdown` into the child directly and this is rarely
        // consulted; it still delegates so a shape or rule that does ask
        // gets the source's answer, not a meter defaulting to `None`. When
        // the walk may not pass, this is exactly the question that must
        // be refused: with no fetch-capable node here, `LimitPushdown`
        // keeps a limit node above the meter (see
        // [`Self::passes_limits`]).
        if !self.passes_limits() {
            return None;
        }
        self.input.with_fetch(limit).map(|input| self.rewrap(input))
    }

    fn with_preserve_order(&self, preserve_order: bool) -> Option<Arc<dyn ExecutionPlan>> {
        // Reachable, not theoretical: LimitPushdown embeds a fetch into
        // whatever fetch-capable non-pushdown node carries it — a residual
        // `FilterExec` directly above a metered scan is an ordinary shape
        // on this engine — and then calls `with_preserve_order` on that
        // node, which delegates into its input, i.e. into this meter. The
        // answer must come from the source that decides whether it may
        // skip row groups, not from a meter defaulting to `None`.
        self.input
            .with_preserve_order(preserve_order)
            .map(|input| self.rewrap(input))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        // The plan is final now, so hand read-ahead the filter the scan prunes
        // row groups with (join filters included).
        if let Some(read_ahead) = &self.read_ahead
            && let Some((predicate, adapter)) = scan_predicate(&self.input)
        {
            read_ahead.set_predicate(predicate, adapter);
        }
        let inner = self.input.execute(partition, context)?;
        Ok(Box::pin(MeteredStream {
            schema: inner.schema(),
            inner,
            op_stats: self.op_stats.clone(),
        }))
    }
}

/// The filter of the file scan `plan`, if it is one and has a filter, with the
/// scan's adapter from the table's schema to a file's.
fn scan_predicate(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(
    Arc<dyn PhysicalExpr>,
    Option<Arc<dyn PhysicalExprAdapterFactory>>,
)> {
    let scan = plan.downcast_ref::<DataSourceExec>()?;
    let config = scan.data_source().downcast_ref::<FileScanConfig>()?;
    let predicate = config.file_source().filter()?;
    Some((predicate, config.expr_adapter_factory.clone()))
}

/// Per-poll CPU bracket around one partition's stream.
struct MeteredStream {
    inner: SendableRecordBatchStream,
    schema: SchemaRef,
    op_stats: Option<Arc<OpStatsCollector>>,
}

impl Stream for MeteredStream {
    type Item = DfResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        timed_poll(this.op_stats.as_deref(), || {
            Pin::new(&mut this.inner).poll_next(cx)
        })
    }
}

impl RecordBatchStream for MeteredStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
