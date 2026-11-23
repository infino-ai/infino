// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`OrderedCoalesceExec`]: gather a scan's partitions under a `LIMIT` with
//! no `ORDER BY` one after another, so the rows come out in the scan's file
//! order and the same rows answer every run.
//!
//! `CoalescePartitionsExec` hands on each partition's batches as they arrive,
//! so which rows fill a `LIMIT n OFFSET m` depends on which partition reads
//! fastest: page after page of an unordered scan can skip rows or show them
//! twice. This gather hands on all of partition 0, then all of partition 1,
//! and so on. Every partition still starts at once, each on its own task, and
//! reads ahead up to [`READ_AHEAD_BATCHES`] batches while the ones before it
//! are handed on, stopping once it has `fetch` rows, which is all it could be
//! asked for. Only the order the partitions are handed on in is fixed; what
//! one partition returns, one task reads.
//!
//! The order is the scan's only when each partition reads a fixed run of the
//! files, in the order they were listed, which `ScanOrderUnderLimit` sets up
//! below this gather.

use std::{fmt, sync::Arc};

use arrow_array::RecordBatch;
use datafusion::{
    common::{Statistics, runtime::SpawnedTask},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
        SendableRecordBatchStream,
        coalesce_partitions::CoalescePartitionsExec,
        limit::LimitStream,
        metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, stream, stream::BoxStream};
use tokio::sync::mpsc;

/// Batches each partition reads ahead of the one being handed on.
const READ_AHEAD_BATCHES: usize = 2;

/// Gathers its input's partitions in partition order, returning at most
/// `fetch` rows; see the module docs.
#[derive(Debug)]
pub(crate) struct OrderedCoalesceExec {
    /// The plain gather this one replaces: its output properties (one
    /// partition, the input's schema) are this one's.
    gather: CoalescePartitionsExec,
    fetch: Option<usize>,
    metrics: ExecutionPlanMetricsSet,
}

impl OrderedCoalesceExec {
    /// Gathers `input`'s partitions in order, returning at most `fetch` rows.
    pub(crate) fn new(input: Arc<dyn ExecutionPlan>, fetch: Option<usize>) -> Self {
        Self {
            gather: CoalescePartitionsExec::new(input).with_fetch(fetch),
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for OrderedCoalesceExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderedCoalesceExec")?;
        if let Some(fetch) = self.fetch {
            write!(f, ": fetch={fetch}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for OrderedCoalesceExec {
    fn name(&self) -> &'static str {
        "OrderedCoalesceExec"
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
                "OrderedCoalesceExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self::new(children.swap_remove(0), self.fetch)))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // Each partition's rows go out in their order, but partitions sorted
        // alike are not sorted once one follows another.
        vec![false]
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        self.gather.partition_statistics(partition)
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self::new(Arc::clone(self.gather.input()), limit)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "OrderedCoalesceExec has one partition; asked for {partition}"
            )));
        }
        let input = self.gather.input();
        let partitions = input.output_partitioning().partition_count();
        let gathered: SendableRecordBatchStream = if partitions <= 1 {
            // One partition is already in order.
            input.execute(0, context)?
        } else {
            let read_ahead = (0..partitions)
                .map(|p| read_ahead(Arc::clone(input), p, self.fetch, Arc::clone(&context)))
                .collect::<Vec<_>>();
            let in_order = stream::iter(read_ahead).flatten();
            Box::pin(RecordBatchStreamAdapter::new(self.schema(), in_order))
        };
        Ok(Box::pin(LimitStream::new(
            gathered,
            0,
            self.fetch,
            BaselineMetrics::new(&self.metrics, partition),
        )))
    }
}

/// Partition `partition` of `input`'s rows, read on a task of its own from
/// now on into a channel of [`READ_AHEAD_BATCHES`] batches; empty batches are
/// left out. The task stops when its stream ends or fails, once it has sent
/// `fetch` rows, or when the stream returned here is dropped; a task that
/// panics ends the stream with an error.
fn read_ahead(
    input: Arc<dyn ExecutionPlan>,
    partition: usize,
    fetch: Option<usize>,
    context: Arc<TaskContext>,
) -> BoxStream<'static, DfResult<RecordBatch>> {
    let (sender, receiver) = mpsc::channel(READ_AHEAD_BATCHES);
    let task = SpawnedTask::spawn(async move {
        let mut batches = match input.execute(partition, context) {
            Ok(batches) => batches,
            Err(error) => {
                // The receiver may be gone already; nothing is waiting then.
                let _ = sender.send(Err(error)).await;
                return;
            }
        };
        let mut wanted = fetch.unwrap_or(usize::MAX);
        while let Some(batch) = batches.next().await {
            let rows = batch.as_ref().map_or(0, RecordBatch::num_rows);
            if batch.is_ok() && rows == 0 {
                continue;
            }
            let failed = batch.is_err();
            if sender.send(batch).await.is_err() || failed {
                return;
            }
            wanted = wanted.saturating_sub(rows);
            if wanted == 0 {
                return;
            }
        }
    });
    let received = stream::unfold(receiver, |mut receiver| async move {
        receiver.recv().await.map(|batch| (batch, receiver))
    });
    // After the last batch, the task's end: a panic becomes the stream's
    // error rather than a partition silently cut short.
    let ended = stream::once(async move {
        task.join()
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))
    })
    .filter_map(|joined| async move { joined.err().map(Err) });
    received.chain(ended).boxed()
}
