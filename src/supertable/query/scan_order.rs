// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ScanOrderUnderLimit`]: under a `LIMIT` with no `ORDER BY`, a table scan
//! returns its rows in file order, the same on every run.
//!
//! With no `ORDER BY`, any rows answer a `LIMIT n OFFSET m`, but paging
//! through a table with growing offsets needs each page to take the rows the
//! one before left off at. A scan in several partitions doesn't keep one
//! order from run to run, for three reasons:
//!
//! - DataFusion shares one queue of files between a scan's partitions:
//!   whichever partition is free takes the next file.
//! - A round-robin exchange deals batches to partitions as they come.
//! - `CoalescePartitionsExec` hands on batches from whichever partition has
//!   one ready.
//!
//! So under a gather that keeps a fetch of at most [`ORDERED_FETCH_BATCHES`]
//! batches, over filters, projections and per-partition limits on a table
//! scan, the rule:
//!
//! - turns the shared queue off, so each partition reads its own run of the
//!   files: DataFusion splits a scan into runs of consecutive files and byte
//!   ranges, partition 0 the first run. The decoded-column scan reads runs of
//!   consecutive row groups ([`CachedScanExec::in_file_order`]).
//! - drops the round-robin exchanges.
//! - swaps the gather for an [`OrderedCoalesceExec`], which hands on all of
//!   partition 0, then all of partition 1, and so on, every partition reading
//!   ahead meanwhile.
//! - drops the round-robin exchanges above that gather. DataFusion puts one
//!   over a gather of more than a batch to spread a computed column, such as
//!   a `CAST`, over partitions, whose batches then come out in any order. On
//!   one partition the computation costs a page of rows.
//!
//! ```text
//!   GlobalLimitExec skip=m fetch=n            GlobalLimitExec skip=m fetch=n
//!     CoalescePartitionsExec fetch=m+n          OrderedCoalesceExec fetch=m+n
//!       MeteredExec                               MeteredExec
//!         DataSourceExec 4 groups                   DataSourceExec 4 groups, preserve_order
//! ```
//!
//! In file order the first rows all come from partition 0, read by one task,
//! where the plain gather takes them from whichever partitions read fastest.
//! Unfiltered, a page of a couple of batches reads as fast either way. Under a
//! filter whose matches cluster away from the first files, partition 0 reads
//! further to find its rows than the plain gather's partitions do between
//! them. A larger fetch would be read mostly one partition at a time, so the
//! rule leaves a fetch of more than [`ORDERED_FETCH_BATCHES`] batches to the
//! plain gather, in no fixed order: pages that end past it can skip or repeat
//! rows.
//!
//! The rows then come in the order the scan lists its files, and within a
//! file in row order, for as long as the table's files stay the same: a
//! commit or a merge that changes them can change the order, as a
//! `LIMIT` with no `ORDER BY` promises nothing across them.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{physical_plan::FileScanConfig, source::DataSourceExec},
    error::Result as DfResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, coalesce_partitions::CoalescePartitionsExec, filter::FilterExec,
        limit::LocalLimitExec, projection::ProjectionExec,
    },
};

use crate::supertable::query::{
    exec::{
        cached_scan_exec::CachedScanExec, metered_exec::MeteredExec,
        ordered_coalesce_exec::OrderedCoalesceExec,
    },
    limit_stage::round_robin_input,
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "ScanOrderUnderLimit";

/// The largest fetch read in file order, in batches of the session's
/// `batch_size`: at the default batch size, 16,384 rows, a page and its
/// offset of up to ten thousand rows among them.
const ORDERED_FETCH_BATCHES: usize = 2;

/// Reads a table scan in file order under a `LIMIT` with no `ORDER BY`; see
/// the module docs.
#[derive(Debug, Default)]
pub(crate) struct ScanOrderUnderLimit;

impl PhysicalOptimizerRule for ScanOrderUnderLimit {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let max_fetch = ORDERED_FETCH_BATCHES.saturating_mul(config.execution.batch_size);
        let plan = plan.transform_down(|node| {
            let Some(fetch) = node
                .downcast_ref::<CoalescePartitionsExec>()
                .and_then(CoalescePartitionsExec::fetch)
                .filter(|&fetch| fetch <= max_fetch)
            else {
                return Ok(Transformed::no(node));
            };
            let [input] = node.children()[..] else {
                return Ok(Transformed::no(node));
            };
            let Some(input) = in_file_order(input)? else {
                return Ok(Transformed::no(node));
            };
            let ordered: Arc<dyn ExecutionPlan> =
                Arc::new(OrderedCoalesceExec::new(input, Some(fetch)));
            Ok(Transformed::yes(ordered))
        })?;
        plan.data
            .transform_up(|node| {
                Ok(match round_robin_input(&node) {
                    Some(input) if over_ordered_gather(input) => {
                        Transformed::yes(Arc::clone(input))
                    }
                    _ => Transformed::no(node),
                })
            })
            .map(|t| t.data)
    }

    fn name(&self) -> &str {
        RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// `plan` reading in file order, with its round-robin exchanges dropped,
/// when it is a table scan below its meter, filters, projections and
/// per-partition limits, in any order; else `None`.
fn in_file_order(plan: &Arc<dyn ExecutionPlan>) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(input) = round_robin_input(plan) {
        return in_file_order(input);
    }
    if let Some(scan) = scan_in_file_order(plan) {
        return Ok(Some(scan));
    }
    let passes_rows_on = plan.downcast_ref::<MeteredExec>().is_some()
        || plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some();
    if !passes_rows_on {
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    in_file_order(child)?
        .map(|child| Arc::clone(plan).with_new_children(vec![child]))
        .transpose()
}

/// `scan`, a Parquet file scan or a decoded-column scan, with each partition
/// reading its own run of the files in order; else `None`.
fn scan_in_file_order(scan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if let Some(cached) = scan.downcast_ref::<CachedScanExec>() {
        return cached.in_file_order();
    }
    scan.downcast_ref::<DataSourceExec>()
        .filter(|source| {
            source
                .data_source()
                .downcast_ref::<FileScanConfig>()
                .is_some()
        })
        .and_then(|source| source.with_preserve_order(true))
}

/// `plan` is an [`OrderedCoalesceExec`], or a node that keeps its one input's
/// order over one.
fn over_ordered_gather(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<OrderedCoalesceExec>().is_some() {
        return true;
    }
    match plan.children()[..] {
        [child] if plan.maintains_input_order() == [true] => over_ordered_gather(child),
        _ => false,
    }
}
