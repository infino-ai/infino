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
//! `POLL_DEPTH` guard below keeps the overlap from counting twice. The
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
    cell::Cell,
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::{
    common::{Statistics, config::ConfigOptions},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{PhysicalExpr, PhysicalSortExpr},
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

use crate::runtime_metrics::{
    cpu,
    op_stats::{OpStatsCollector, metering_active},
};

thread_local! {
    /// Depth of live [`MeteredStream`] polls on this thread.
    ///
    /// The node is placed both at the plan root and around each table
    /// scan, because neither alone is enough: DataFusion spawns a task per
    /// partition, so a root-only bracket misses every spawned partition,
    /// while a scan-only bracket misses the aggregation and sort work
    /// above it. With both, a single-partition plan would poll the scan
    /// *inside* the root's poll on the same thread and count that time
    /// twice.
    ///
    /// So only the outermost bracket on a given thread measures. On a
    /// spawned partition the scan is outermost on its own thread and
    /// counts there; on the coalescing thread the root counts, and the
    /// nested scan defers to it. Each thread's CPU is attributed exactly
    /// once.
    static POLL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Wraps `input`, metering every partition's poll time into `op_stats`.
#[derive(Debug)]
pub(crate) struct MeteredExec {
    input: Arc<dyn ExecutionPlan>,
    op_stats: Option<Arc<OpStatsCollector>>,
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
        Self { input, op_stats }
    }
}

impl DisplayAs for MeteredExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MeteredExec")
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
        Ok(Arc::new(MeteredExec::new(
            children.swap_remove(0),
            self.op_stats.clone(),
        )))
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
        Ok(self.input.repartitioned(target_partitions, config)?.map(
            |input| -> Arc<dyn ExecutionPlan> {
                Arc::new(MeteredExec::new(input, self.op_stats.clone()))
            },
        ))
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
        Ok(self.input.try_swapping_with_projection(projection)?.map(
            |input| -> Arc<dyn ExecutionPlan> {
                Arc::new(MeteredExec::new(input, self.op_stats.clone()))
            },
        ))
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
        // The default answer is `Unsupported`, which stops the pushdown
        // walk dead. A meter sitting between a sort and the scan would
        // then cost the query its row-group reorder, its reverse scan and
        // any sort elimination — turning an early-terminating TopK into a
        // full scan plus a full sort.
        let rewrap = |input| -> Arc<dyn ExecutionPlan> {
            Arc::new(MeteredExec::new(input, self.op_stats.clone()))
        };
        Ok(match self.input.try_pushdown_sort(order)? {
            SortOrderPushdownResult::Exact { inner } => SortOrderPushdownResult::Exact {
                inner: rewrap(inner),
            },
            SortOrderPushdownResult::Inexact { inner } => SortOrderPushdownResult::Inexact {
                inner: rewrap(inner),
            },
            SortOrderPushdownResult::Unsupported => SortOrderPushdownResult::Unsupported,
        })
    }

    fn with_preserve_order(&self, preserve_order: bool) -> Option<Arc<dyn ExecutionPlan>> {
        // Order-sensitivity has to reach the data source, which decides
        // whether it may skip row groups. Defaulting to `None` silently
        // drops that signal at the meter.
        self.input
            .with_preserve_order(preserve_order)
            .map(|input| -> Arc<dyn ExecutionPlan> {
                Arc::new(MeteredExec::new(input, self.op_stats.clone()))
            })
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let inner = self.input.execute(partition, context)?;
        Ok(Box::pin(MeteredStream {
            schema: inner.schema(),
            inner,
            op_stats: self.op_stats.clone(),
        }))
    }
}

/// Restores `POLL_DEPTH` on unwind as well as on the normal path. A poll
/// that panicked out of a raised depth would leave the thread-local set
/// forever, and every later query polled on that worker would treat its
/// outermost bracket as nested and bill zero CPU.
struct PollDepthGuard(u32);

impl Drop for PollDepthGuard {
    fn drop(&mut self) {
        POLL_DEPTH.with(|d| d.set(self.0));
    }
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
        let Some(stats) = self.op_stats.clone() else {
            return Pin::new(&mut self.inner).poll_next(cx);
        };
        if !metering_active() {
            return Pin::new(&mut self.inner).poll_next(cx);
        }
        // Nested inside another bracket on this thread: that one is already
        // measuring this poll, so counting here would double-charge it.
        if POLL_DEPTH.with(|d| d.get()) > 0 {
            return Pin::new(&mut self.inner).poll_next(cx);
        }
        let restore = PollDepthGuard(POLL_DEPTH.with(|d| d.replace(1)));
        let start = cpu::thread_cpu_ns();
        let out = Pin::new(&mut self.inner).poll_next(cx);
        let delta = cpu::thread_cpu_delta_ns(start);
        drop(restore);
        stats.add_kernel_cpu_ns(delta);
        out
    }
}

impl RecordBatchStream for MeteredStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
