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
//! Placed around the table scan rather than the plan root on purpose:
//! DataFusion spawns a task per partition, so a root-level bracket would
//! see only the coalescing thread. Wrapping the scan means each partition
//! is measured on whichever worker actually polls it; the collector's
//! counters are atomics, so concurrent partitions fold safely.

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
    error::Result as DfResult,
    execution::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
        SendableRecordBatchStream,
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
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        // Optimizer rewrites keep the meter attached to whatever the child
        // became; dropping it here would silently unmeter the scan.
        Ok(Arc::new(MeteredExec::new(
            children
                .into_iter()
                .next()
                .unwrap_or(Arc::clone(&self.input)),
            self.op_stats.clone(),
        )))
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
        POLL_DEPTH.with(|d| d.set(1));
        let start = cpu::thread_cpu_ns();
        let out = Pin::new(&mut self.inner).poll_next(cx);
        let delta = cpu::thread_cpu_delta_ns(start);
        POLL_DEPTH.with(|d| d.set(0));
        stats.add_kernel_cpu_ns(delta);
        out
    }
}

impl RecordBatchStream for MeteredStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}
