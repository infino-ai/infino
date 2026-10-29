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
//! over its first batch and been asked for another, or has ended. The gate
//! sits on the scan, below the filter, because a `FilterExec` emits nothing
//! for a batch that keeps no rows: gating on the rows that reach the gather
//! would hold the other partitions back until partition 0 had read
//! everything.
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
//! - **It gives no batch:** a file whose rows the index or the statistics
//!   rule out gives none. DataFusion shares one queue of files between a
//!   scan's partitions, so partition 0, alone while the gate is shut, would
//!   go on to take every such file, one after another. So `StageLimitGathers`
//!   takes the scan's first file out of it, and partition 0 reads that file
//!   alone, in a scan of its own; the gate opens when it ends, and partition
//!   0 goes on with the shared queue beside the others. At most one file is
//!   read before the others start.
//! - **It fails:** the gate is abandoned, so no other partition starts while
//!   the query returns the error.
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
    datasource::{
        physical_plan::{FileGroup, FileScanConfig},
        source::DataSourceExec,
    },
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
    /// The most rows to return: the `LIMIT` (with any `OFFSET`) above.
    fetch: usize,
}

impl StagedCoalesceExec {
    /// Gathers `input`'s partitions, returning at most `fetch` rows.
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, fetch: usize) -> Self {
        Self {
            gather: CoalescePartitionsExec::new(input).with_fetch(Some(fetch)),
            fetch,
        }
    }
}

impl DisplayAs for StagedCoalesceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StagedCoalesceExec: fetch={}", self.fetch)
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
        Ok(Arc::new(Self::new(children.swap_remove(0), self.fetch)))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.gather.maintains_input_order()
    }

    fn supports_limit_pushdown(&self) -> bool {
        // Rows pass straight through, as through the gather it wraps.
        true
    }

    fn fetch(&self) -> Option<usize> {
        Some(self.fetch)
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
/// first batch and been asked for another, or has ended, when a
/// [`StagedCoalesceExec`] above put a [`ScanGate`] in the task context.
/// Without one, it passes every partition straight through.
#[derive(Debug)]
pub(crate) struct ScanGateExec {
    input: Arc<dyn ExecutionPlan>,
    /// The scan of the one file partition 0 reads before the rest of its
    /// partition, when `input` is a file scan that left it out; read through
    /// `input`'s meter. See the module docs.
    first: Option<FileScanConfig>,
}

impl ScanGateExec {
    /// Gates `input`, a table scan; partition 0 reads `first`, the scan of
    /// a file `input` leaves out, before its own partition of `input`.
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, first: Option<FileScanConfig>) -> Self {
        Self { input, first }
    }

    /// The scan of the file partition 0 reads first, if any.
    #[cfg(test)]
    pub(crate) fn first(&self) -> Option<&FileScanConfig> {
        self.first.as_ref()
    }

    /// Partition 0: the first file, if any, gated by `gate`, then partition 0
    /// of `input` once that file ends.
    fn first_partition(
        &self,
        gate: Option<Arc<ScanGate>>,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let schema = self.schema();
        let Some(first) = &self.first else {
            let batches = self.input.execute(0, context)?;
            return Ok(match gate {
                Some(gate) => Box::pin(FirstPartition::new(schema, batches, gate)),
                None => batches,
            });
        };
        // A fresh scan each run: DataFusion keeps a scan's queue of files on
        // the node, and a second run must read the file again.
        let scan = DataSourceExec::from_data_source(first.clone());
        let batches = Arc::clone(&self.input)
            .with_new_children(vec![scan])?
            .execute(0, Arc::clone(&context))?;
        let lead = match &gate {
            Some(gate) => {
                FirstPartition::new(Arc::clone(&schema), batches, Arc::clone(gate)).boxed()
            }
            None => batches.boxed(),
        };
        let rest = self.deferred(0, gate, context);
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            lead.chain(rest),
        )))
    }

    /// Partition `partition` of `input`, not started until `gate`, if any,
    /// opens: no file is opened and no byte fetched for a partition the
    /// query never needs.
    fn deferred(
        &self,
        partition: usize,
        gate: Option<Arc<ScanGate>>,
        context: Arc<TaskContext>,
    ) -> SendableRecordBatchStream {
        let schema = self.schema();
        let input = Arc::clone(&self.input);
        let empty = Arc::clone(&schema);
        let deferred = stream::once(async move {
            let opened = match gate {
                Some(gate) => gate.opened().await,
                None => true,
            };
            if opened {
                input.execute(partition, context)
            } else {
                Ok(Box::pin(EmptyRecordBatchStream::new(empty)) as SendableRecordBatchStream)
            }
        })
        .try_flatten();
        Box::pin(RecordBatchStreamAdapter::new(schema, deferred))
    }
}

impl DisplayAs for ScanGateExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ScanGateExec")?;
        let Some(first) = &self.first else {
            return Ok(());
        };
        for file in first.file_groups.iter().flat_map(FileGroup::iter) {
            write!(f, ": first={}", file.object_meta.location)?;
        }
        Ok(())
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
        Ok(Arc::new(Self::new(
            children.swap_remove(0),
            self.first.clone(),
        )))
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
        let gate = context.session_config().get_extension::<ScanGate>();
        if partition == 0 {
            return self.first_partition(gate, context);
        }
        match gate {
            Some(gate) => Ok(self.deferred(partition, Some(gate), context)),
            None => self.input.execute(partition, context),
        }
    }
}

/// Where a query's [`ScanGate`] stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateState {
    /// Partition 0 has not yet been asked for a second batch, nor ended.
    Closed,
    /// Partition 0 fell short: start every other partition.
    Open,
    /// Partition 0's scan failed, or was dropped while the gate was shut by
    /// a node that had its rows: the other partitions are not needed.
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
/// its first, or when it ends, and abandons it if it fails or is dropped
/// before.
struct FirstPartition {
    schema: SchemaRef,
    batches: SendableRecordBatchStream,
    gate: Arc<ScanGate>,
    /// Whether a batch has been handed over yet.
    delivered: bool,
}

impl FirstPartition {
    fn new(schema: SchemaRef, batches: SendableRecordBatchStream, gate: Arc<ScanGate>) -> Self {
        Self {
            schema,
            batches,
            gate,
            delivered: false,
        }
    }
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
            Poll::Ready(None) => self.gate.leave_closed(GateState::Open),
            Poll::Ready(Some(Err(_))) => self.gate.leave_closed(GateState::Abandoned),
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
        common::ScalarValue,
        datasource::memory::MemorySourceConfig,
        error::{DataFusionError, Result as DfResult},
        execution::TaskContext,
        logical_expr::Operator,
        physical_plan::{
            DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
            collect,
            expressions::{BinaryExpr, col, lit},
            filter::FilterExec,
            limit::LocalLimitExec,
            stream::RecordBatchStreamAdapter,
        },
    };
    use futures::{StreamExt, stream};
    use tokio::{
        sync::Notify,
        task,
        time::{sleep, timeout},
    };

    use super::{ScanGateExec, StagedCoalesceExec};

    /// Rows in each full test batch.
    const BATCH_ROWS: usize = 8;
    /// Partitions in each test input.
    const PARTITIONS: usize = 4;
    /// A `LIMIT` one full batch meets.
    const FETCH: usize = 5;
    /// How long a test waits before calling a gather stuck.
    const STUCK: Duration = Duration::from_secs(10);
    /// How long partition 0 gives nothing before a test checks that no other
    /// partition started meanwhile.
    const IDLE: Duration = Duration::from_millis(50);

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    fn batch(rows: usize) -> RecordBatch {
        let values = Int64Array::from_iter_values((0..rows).map(|row| row as i64));
        RecordBatch::try_new(schema(), vec![Arc::new(values)]).expect("batch")
    }

    /// An in-memory scan of `partitions`, each a list of batch sizes.
    fn memory_scan(partitions: &[&[usize]]) -> Arc<dyn ExecutionPlan> {
        let batches: Vec<Vec<RecordBatch>> = partitions
            .iter()
            .map(|sizes| sizes.iter().map(|&rows| batch(rows)).collect())
            .collect();
        MemorySourceConfig::try_new_exec(&batches, schema(), None).expect("input")
    }

    /// How partition 0 of a [`Tracked`] scan behaves past its batches.
    #[derive(Debug, Clone)]
    enum First {
        /// Ends after them.
        Ends,
        /// Never ends after them: only a gate that opens on the second ask
        /// lets the query finish.
        NeverEnds,
        /// Gives nothing until notified, then its batches.
        WaitsFor(Arc<Notify>),
        /// Fails before them.
        Fails,
    }

    /// A scan that records which of its partitions were started. Each run of
    /// the plan reads the next of `runs`, counted by partition 0's starts.
    #[derive(Debug)]
    struct Tracked {
        runs: Vec<Arc<dyn ExecutionPlan>>,
        started: Arc<Mutex<Vec<usize>>>,
        first: First,
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
            self.runs[0].properties()
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
            let run = {
                let mut started = self.started.lock().expect("started");
                started.push(partition);
                started.iter().filter(|&&p| p == 0).count() - 1
            };
            let batches = self.runs[run].execute(partition, context)?;
            if partition != 0 {
                return Ok(batches);
            }
            let first = match self.first.clone() {
                First::Ends => return Ok(batches),
                First::NeverEnds => batches.chain(stream::pending()).boxed(),
                First::WaitsFor(notify) => stream::once(async move {
                    notify.notified().await;
                    batches
                })
                .flatten()
                .boxed(),
                First::Fails => stream::once(async {
                    Err(DataFusionError::Execution("partition 0 failed".into()))
                })
                .chain(batches)
                .boxed(),
            };
            Ok(Box::pin(RecordBatchStreamAdapter::new(schema(), first)))
        }
    }

    /// The node over the gate that stops pulling once it has its rows.
    #[derive(Debug, Clone, Copy)]
    enum Above {
        /// A `FilterExec` with the gather's fetch, as over a `WHERE`.
        Filter,
        /// A `LocalLimitExec`, as over a `WHERE` an index answers.
        Limit,
    }

    /// A staged gather of `fetch` over `above` over a gated [`Tracked`] scan
    /// of `runs`, each a list of partitions; and the partitions it started.
    fn staged_plan(
        runs: &[&[&[usize]]],
        fetch: usize,
        above: Above,
        first: First,
    ) -> (Arc<dyn ExecutionPlan>, Arc<Mutex<Vec<usize>>>) {
        let started = Arc::new(Mutex::new(Vec::new()));
        let scan = Arc::new(Tracked {
            runs: runs.iter().map(|run| memory_scan(run)).collect(),
            started: Arc::clone(&started),
            first,
        });
        let gate = Arc::new(ScanGateExec::new(scan, None));
        let above: Arc<dyn ExecutionPlan> = match above {
            Above::Filter => {
                let every_row = Arc::new(BinaryExpr::new(
                    col("a", &schema()).expect("a"),
                    Operator::GtEq,
                    lit(ScalarValue::Int64(Some(0))),
                ));
                FilterExec::try_new(every_row, gate)
                    .expect("filter")
                    .with_fetch(Some(fetch))
                    .expect("a filter takes a fetch")
            }
            Above::Limit => Arc::new(LocalLimitExec::new(gate, fetch)),
        };
        (Arc::new(StagedCoalesceExec::new(above, fetch)), started)
    }

    /// The rows `plan` returns.
    async fn rows(plan: Arc<dyn ExecutionPlan>) -> DfResult<usize> {
        let out = timeout(STUCK, collect(plan, Arc::new(TaskContext::default())))
            .await
            .expect("the gather got stuck")?;
        Ok(out.iter().map(RecordBatch::num_rows).sum())
    }

    /// The partitions started so far, sorted.
    fn sorted(started: &Mutex<Vec<usize>>) -> Vec<usize> {
        let mut started = started.lock().expect("started").clone();
        started.sort_unstable();
        started
    }

    /// Runs a staged gather of `fetch` over `partitions` under `above`,
    /// returning the rows gathered and the partitions started, sorted.
    async fn gather(
        partitions: &[&[usize]],
        fetch: usize,
        above: Above,
        first: First,
    ) -> (usize, Vec<usize>) {
        let (plan, started) = staged_plan(&[partitions], fetch, above, first);
        let rows = rows(plan).await.expect("gather");
        (rows, sorted(&started))
    }

    #[tokio::test]
    async fn a_first_batch_that_meets_the_limit_starts_no_other_partition() {
        let full: &[usize] = &[BATCH_ROWS, BATCH_ROWS];
        for above in [Above::Filter, Above::Limit] {
            let (rows, started) = gather(&[full; PARTITIONS], FETCH, above, First::Ends).await;
            assert_eq!(rows, FETCH, "{above:?}");
            assert_eq!(started, vec![0], "{above:?}");
        }
    }

    #[tokio::test]
    async fn a_short_first_batch_opens_the_gate_when_another_is_asked_for() {
        // Partition 0 never ends, so only the second ask can open the gate.
        let short: &[usize] = &[FETCH - 1];
        let full: &[usize] = &[BATCH_ROWS];
        for above in [Above::Filter, Above::Limit] {
            let partitions = [short, full, full, full];
            let (rows, started) = gather(&partitions, FETCH, above, First::NeverEnds).await;
            assert_eq!(rows, FETCH, "{above:?}");
            assert!(started.len() > 1, "{above:?}: started {started:?}");
        }
    }

    #[tokio::test]
    async fn an_empty_first_partition_opens_the_gate() {
        let full: &[usize] = &[BATCH_ROWS];
        let (rows, started) =
            gather(&[&[], full, full, full], FETCH, Above::Filter, First::Ends).await;
        assert_eq!(rows, FETCH);
        assert!(started.len() > 1, "started {started:?}");
    }

    #[tokio::test]
    async fn the_others_wait_while_the_first_partition_gives_nothing() {
        // Partition 0 gives nothing for a while, then ends empty, like a
        // file the index ruled out: the others start only when it ends.
        let full: &[usize] = &[BATCH_ROWS];
        let notify = Arc::new(Notify::new());
        let (plan, started) = staged_plan(
            &[&[&[], full, full, full]],
            FETCH,
            Above::Filter,
            First::WaitsFor(Arc::clone(&notify)),
        );
        let query = task::spawn(rows(plan));
        sleep(IDLE).await;
        assert_eq!(sorted(&started), vec![0]);
        notify.notify_one();
        let rows = query.await.expect("join").expect("gather");
        assert_eq!(rows, FETCH);
        assert_eq!(sorted(&started), (0..PARTITIONS).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_failing_first_partition_starts_no_other() {
        let full: &[usize] = &[BATCH_ROWS];
        let (plan, started) =
            staged_plan(&[&[full; PARTITIONS]], FETCH, Above::Filter, First::Fails);
        assert!(rows(plan).await.is_err());
        assert_eq!(sorted(&started), vec![0]);
    }

    #[tokio::test]
    async fn a_plan_run_twice_gets_a_fresh_gate_each_time() {
        // The first run's partition 0 meets the limit, so its gate ends up
        // abandoned. The second run's is empty and needs the others: with the
        // first run's gate it would return nothing.
        let full: &[usize] = &[BATCH_ROWS];
        let (plan, started) = staged_plan(
            &[&[full; PARTITIONS], &[&[], full, full, full]],
            FETCH,
            Above::Filter,
            First::Ends,
        );
        assert_eq!(rows(Arc::clone(&plan)).await.expect("first run"), FETCH);
        assert_eq!(sorted(&started), vec![0]);
        assert_eq!(rows(plan).await.expect("second run"), FETCH);
        assert!(sorted(&started).len() > 2, "started {started:?}");
    }

    #[tokio::test]
    async fn a_limit_past_every_row_returns_them_all() {
        let full: &[usize] = &[BATCH_ROWS, BATCH_ROWS];
        let (rows, started) =
            gather(&[full; PARTITIONS], usize::MAX, Above::Limit, First::Ends).await;
        assert_eq!(rows, PARTITIONS * 2 * BATCH_ROWS);
        assert_eq!(started, (0..PARTITIONS).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn a_scan_gate_without_a_staged_gather_passes_every_partition() {
        let full: &[usize] = &[BATCH_ROWS];
        let out = collect(
            Arc::new(ScanGateExec::new(memory_scan(&[full; PARTITIONS]), None)),
            Arc::new(TaskContext::default()),
        )
        .await
        .expect("collect");
        let rows: usize = out.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(rows, PARTITIONS * BATCH_ROWS);
    }
}
