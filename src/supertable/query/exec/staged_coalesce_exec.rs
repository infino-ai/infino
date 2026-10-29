// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`StagedCoalesceExec`] and [`ScanGateExec`]: gather a scan's partitions
//! under a small `LIMIT` so that partition 0's first batch can end the query
//! before the other partitions decode anything.
//!
//! `CoalescePartitionsExec` starts every input partition at once. Under
//! `WHERE ... LIMIT n` over a scan DataFusion split into byte ranges, each
//! range then decodes a batch of every projected column, and the first `n`
//! rows to pass, often all in the first range's first batch, end the query:
//! the other ranges' batches are thrown away.
//!
//! The two nodes hold partitions 1.. back until partition 0's scan has handed
//! over its first batch and been asked for another. The gate sits on the scan,
//! below the filter, because a `FilterExec` emits nothing for a batch that
//! keeps no rows: gating on the rows that reach the gather would hold the
//! other partitions back until partition 0 had read everything.
//!
//! ```text
//!   StagedCoalesceExec fetch=n      a fresh gate per query, in the task context
//!     FilterExec fetch=n            stops pulling once it has n rows
//!       ScanGateExec                partition 0 runs; 1.. wait on the gate
//!         MeteredExec
//!           DataSourceExec
//! ```
//!
//! - **Partition 0's first batch is enough:** the filter, or the limit, above
//!   the gate has its `n` rows, so it drops partition 0's scan without asking
//!   for more. The gate stays shut, the gather has its rows, and the other
//!   partitions are dropped without ever being started.
//! - **It falls short:** the node above asks partition 0's scan for another
//!   batch, the gate opens, and every other partition starts at once beside
//!   partition 0, as under a plain gather. A filter that keeps almost nothing
//!   waits for one batch, then reads in parallel as before.
//!
//! The shut gate is only correct when whatever drops partition 0's scan early
//! does so because it already has the gather's `n` rows; `StageLimitGathers`
//! only places the gate under nodes that keep a limit of at least `n`.

use std::{
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::{
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, EmptyRecordBatchStream, ExecutionPlan, PlanProperties,
        RecordBatchStream, SendableRecordBatchStream, coalesce_partitions::CoalescePartitionsExec,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{Stream, StreamExt, TryStreamExt, stream};
use tokio::sync::watch;

/// Gathers its input's partitions under a `LIMIT`, as `CoalescePartitionsExec`
/// does, with a fresh [`ScanGate`] for the [`ScanGateExec`] below; see the
/// module docs.
#[derive(Debug)]
pub(crate) struct StagedCoalesceExec {
    gather: CoalescePartitionsExec,
}

impl StagedCoalesceExec {
    /// Gathers `input`'s partitions, returning at most `fetch` rows.
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, fetch: usize) -> Self {
        Self {
            gather: CoalescePartitionsExec::new(input).with_fetch(Some(fetch)),
        }
    }

    /// The most rows to return: the `LIMIT` (with any `OFFSET`) above.
    fn limit(&self) -> usize {
        self.gather.fetch().unwrap_or(usize::MAX)
    }
}

impl DisplayAs for StagedCoalesceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StagedCoalesceExec: fetch={}", self.limit())
    }
}

impl ExecutionPlan for StagedCoalesceExec {
    fn name(&self) -> &'static str {
        "StagedCoalesceExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.gather.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![self.gather.input()]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "StagedCoalesceExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.swap_remove(0), self.limit())))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.gather.maintains_input_order()
    }

    fn supports_limit_pushdown(&self) -> bool {
        // Rows pass straight through, as through the gather it wraps.
        true
    }

    fn fetch(&self) -> Option<usize> {
        self.gather.fetch()
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        limit.map(|fetch| Arc::new(Self::new(Arc::clone(self.gather.input()), fetch)) as _)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        // Each query gets its own gate, so a plan run twice never sees the
        // first run's.
        let config = context
            .session_config()
            .clone()
            .with_extension(Arc::new(ScanGate::new()));
        let gated = TaskContext::new(
            context.task_id(),
            context.session_id(),
            config,
            context.scalar_functions().clone(),
            context.higher_order_functions().clone(),
            context.aggregate_functions().clone(),
            context.window_functions().clone(),
            context.runtime_env(),
        );
        self.gather.execute(partition, Arc::new(gated))
    }
}

/// Holds a scan's partitions 1.. back until partition 0 has handed over its
/// first batch and been asked for another, when a [`StagedCoalesceExec`] above
/// put a [`ScanGate`] in the task context. Without one, it passes every
/// partition straight through.
#[derive(Debug)]
pub(crate) struct ScanGateExec {
    input: Arc<dyn ExecutionPlan>,
}

impl ScanGateExec {
    /// Gates `input`, a table scan.
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>) -> Self {
        Self { input }
    }
}

impl DisplayAs for ScanGateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScanGateExec")
    }
}

impl ExecutionPlan for ScanGateExec {
    fn name(&self) -> &'static str {
        "ScanGateExec"
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
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "ScanGateExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.swap_remove(0))))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn supports_limit_pushdown(&self) -> bool {
        // Rows pass straight through.
        true
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let Some(gate) = context.session_config().get_extension::<ScanGate>() else {
            return self.input.execute(partition, context);
        };
        let schema = self.schema();
        if partition == 0 {
            return Ok(Box::pin(FirstPartition {
                schema,
                batches: self.input.execute(partition, context)?,
                gate,
                delivered: false,
            }));
        }
        // Not started at all until the gate opens: no file is opened and no
        // byte fetched for a partition the query never needs.
        let input = Arc::clone(&self.input);
        let empty = Arc::clone(&schema);
        let deferred = stream::once(async move {
            if gate.opened().await {
                input.execute(partition, context)
            } else {
                Ok(Box::pin(EmptyRecordBatchStream::new(empty)) as SendableRecordBatchStream)
            }
        })
        .try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, deferred)))
    }
}

/// Where a query's [`ScanGate`] stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateState {
    /// Partition 0 has not yet been asked for a second batch.
    Closed,
    /// Partition 0 fell short: start every other partition.
    Open,
    /// Partition 0's scan was dropped while the gate was shut, by a node that
    /// had its rows: the other partitions are not needed.
    Abandoned,
}

/// One query's gate between a scan's partition 0 and the rest.
#[derive(Debug)]
struct ScanGate {
    state: watch::Sender<GateState>,
}

impl ScanGate {
    fn new() -> Self {
        Self {
            state: watch::Sender::new(GateState::Closed),
        }
    }

    /// Moves a shut gate to `to`; an open or abandoned gate stays as it is.
    fn leave_closed(&self, to: GateState) {
        self.state.send_if_modified(|state| {
            let closed = *state == GateState::Closed;
            if closed {
                *state = to;
            }
            closed
        });
    }

    /// Waits until the gate opens or is abandoned; `true` if it opened.
    async fn opened(&self) -> bool {
        let mut state = self.state.subscribe();
        // The sender lives as long as this gate, so the wait can't fail; if
        // it somehow did, starting the partition is the safe answer.
        state
            .wait_for(|state| *state != GateState::Closed)
            .await
            .map_or(true, |state| *state == GateState::Open)
    }
}

/// Partition 0 of a gated scan: opens the gate when asked for a batch after
/// its first, or when it ends or fails, and abandons it if dropped before.
struct FirstPartition {
    schema: SchemaRef,
    batches: SendableRecordBatchStream,
    gate: Arc<ScanGate>,
    /// Whether a batch has been handed over yet.
    delivered: bool,
}

impl Stream for FirstPartition {
    type Item = DfResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.delivered {
            self.gate.leave_closed(GateState::Open);
        }
        let poll = self.batches.poll_next_unpin(cx);
        match &poll {
            Poll::Ready(Some(Ok(_))) => self.delivered = true,
            Poll::Ready(Some(Err(_)) | None) => self.gate.leave_closed(GateState::Open),
            Poll::Pending => {}
        }
        poll
    }
}

impl RecordBatchStream for FirstPartition {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Drop for FirstPartition {
    fn drop(&mut self) {
        self.gate.leave_closed(GateState::Abandoned);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt,
        sync::{Arc, Mutex},
        time::Duration,
    };

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::{
        datasource::memory::MemorySourceConfig,
        error::{DataFusionError, Result as DfResult},
        execution::TaskContext,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
            collect, limit::LocalLimitExec, stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{StreamExt, stream};
    use tokio::time::timeout;

    use super::{ScanGateExec, StagedCoalesceExec};

    /// Rows in each full test batch.
    const BATCH_ROWS: usize = 8;
    /// Partitions in each test input.
    const PARTITIONS: usize = 4;
    /// A `LIMIT` one full batch meets.
    const FETCH: usize = 5;
    /// How long a test waits before calling a gather stuck.
    const STUCK: Duration = Duration::from_secs(10);

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    fn batch(rows: usize) -> RecordBatch {
        let values = Int64Array::from_iter_values((0..rows).map(|row| row as i64));
        RecordBatch::try_new(schema(), vec![Arc::new(values)]).expect("batch")
    }

    /// A scan that records which of its partitions were started. With
    /// `stall`, partition 0 never ends after its batches: only a gate that
    /// opens on the second ask lets the query finish.
    #[derive(Debug)]
    struct Tracked {
        input: Arc<dyn ExecutionPlan>,
        started: Arc<Mutex<Vec<usize>>>,
        stall: bool,
    }

    impl DisplayAs for Tracked {
        fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Tracked")
        }
    }

    impl ExecutionPlan for Tracked {
        fn name(&self) -> &'static str {
            "Tracked"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> DfResult<Arc<dyn ExecutionPlan>> {
            Err(DataFusionError::Internal("Tracked is a leaf".into()))
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> DfResult<SendableRecordBatchStream> {
            self.started.lock().expect("started").push(partition);
            let batches = self.input.execute(partition, context)?;
            if !(self.stall && partition == 0) {
                return Ok(batches);
            }
            let stalled = batches.chain(stream::pending());
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema(), stalled)))
        }
    }

    /// Runs `LocalLimitExec(fetch)` over a gated scan of `partitions` (each a
    /// list of batch sizes) under a staged gather of `fetch`, returning the
    /// rows gathered and the partitions started, sorted.
    async fn gather(partitions: &[&[usize]], fetch: usize, stall: bool) -> (usize, Vec<usize>) {
        let batches: Vec<Vec<RecordBatch>> = partitions
            .iter()
            .map(|sizes| sizes.iter().map(|&rows| batch(rows)).collect())
            .collect();
        let input = MemorySourceConfig::try_new_exec(&batches, schema(), None).expect("input");
        let started = Arc::new(Mutex::new(Vec::new()));
        let scan = Arc::new(Tracked {
            input,
            started: Arc::clone(&started),
            stall,
        });
        let limited = Arc::new(LocalLimitExec::new(
            Arc::new(ScanGateExec::new(scan)),
            fetch,
        ));
        let staged = Arc::new(StagedCoalesceExec::new(limited, fetch));
        let out = timeout(STUCK, collect(staged, Arc::new(TaskContext::default())))
            .await
            .expect("the gather got stuck")
            .expect("gather");
        let rows = out.iter().map(RecordBatch::num_rows).sum();
        let mut started = started.lock().expect("started").clone();
        started.sort_unstable();
        (rows, started)
    }

    #[tokio::test]
    async fn a_first_batch_that_meets_the_limit_starts_no_other_partition() {
        let full: &[usize] = &[BATCH_ROWS, BATCH_ROWS];
        let (rows, started) = gather(&[full; PARTITIONS], FETCH, false).await;
        assert_eq!(rows, FETCH);
        assert_eq!(started, vec![0]);
    }

    #[tokio::test]
    async fn a_short_first_batch_opens_the_gate_when_another_is_asked_for() {
        // Partition 0 never ends, so only the second ask can open the gate.
        let short: &[usize] = &[FETCH - 1];
        let full: &[usize] = &[BATCH_ROWS];
        let (rows, started) = gather(&[short, full, full, full], FETCH, true).await;
        assert_eq!(rows, FETCH);
        assert!(started.len() > 1, "started {started:?}");
    }

    #[tokio::test]
    async fn an_empty_first_partition_opens_the_gate() {
        let full: &[usize] = &[BATCH_ROWS];
        let (rows, started) = gather(&[&[], full, full, full], FETCH, false).await;
        assert_eq!(rows, FETCH);
        assert!(started.len() > 1, "started {started:?}");
    }

    #[tokio::test]
    async fn a_limit_past_every_row_returns_them_all() {
        let full: &[usize] = &[BATCH_ROWS, BATCH_ROWS];
        let (rows, started) = gather(&[full; PARTITIONS], usize::MAX, false).await;
        assert_eq!(rows, PARTITIONS * 2 * BATCH_ROWS);
        assert_eq!(started, (0..PARTITIONS).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_scan_gate_without_a_staged_gather_passes_every_partition() {
        let full: &[usize] = &[BATCH_ROWS];
        let batches: Vec<Vec<RecordBatch>> = [full; PARTITIONS]
            .iter()
            .map(|sizes| sizes.iter().map(|&rows| batch(rows)).collect())
            .collect();
        let input = MemorySourceConfig::try_new_exec(&batches, schema(), None).expect("input");
        let out = collect(
            Arc::new(ScanGateExec::new(input)),
            Arc::new(TaskContext::default()),
        )
        .await
        .expect("collect");
        let rows: usize = out.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, PARTITIONS * BATCH_ROWS);
    }
}
