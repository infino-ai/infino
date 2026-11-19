// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`IdOrderUnderTopK`]: an `ORDER BY _id ... LIMIT` over a scan of kept
//! decoded columns reads the scan's rows by id block (see
//! [`CachedScanExec`]'s module docs).
//!
//! DataFusion hands a sort down to the scan below it
//! (`ExecutionPlan::try_pushdown_sort`) only through nodes that pass it on,
//! and on DataFusion 54 a `FilterExec` doesn't. A `WHERE` the index doesn't
//! answer leaves one between the top-k and the scan, so the scan never
//! learns of the sort. This rule tells it: from a top-k whose first key is a
//! column, it walks down through filters, projections and our scan meter to
//! the scan, following the column, and when the column is `_id` the scan
//! reads in id order. The order is only a reading order; the sort above
//! stays, and the answer is the same.
//!
//! A filter on the way gathers the rows that pass into batches of the
//! session's batch size before handing them up, so under a filter keeping a
//! seventh of the rows each partition would read tens of thousands of rows
//! before the top-k saw any, and the bound would come too late to skip much. The
//! rule sets those filters' batches to the top-k's `k` rows: the top-k has
//! its bound once it has seen `k` rows that pass, and the scan stops soon
//! after.
//!
//! It runs before DataFusion's own rules, so the filter pushdown that hands
//! the top-k's bound down to the scan finds the scan already set to read in
//! id order.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DfResult,
    physical_expr::expressions::Column,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, filter::FilterExec, projection::ProjectionExec, sorts::sort::SortExec,
    },
};

use crate::supertable::query::exec::{cached_scan_exec::CachedScanExec, metered_exec::MeteredExec};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "IdOrderUnderTopK";

/// Sets a scan of kept decoded columns under an `ORDER BY _id ... LIMIT` to
/// read in id order; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct IdOrderUnderTopK;

impl PhysicalOptimizerRule for IdOrderUnderTopK {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let Some(sort) = node.downcast_ref::<SortExec>() else {
                return Ok(Transformed::no(node));
            };
            let first = sort.expr().first();
            let Some(column) = first.expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(node));
            };
            let descending = first.options.descending;
            let Some(fetch) = sort.fetch() else {
                return Ok(Transformed::no(node));
            };
            let Some(input) = in_id_order(sort.input(), column.index(), descending, fetch)? else {
                return Ok(Transformed::no(node));
            };
            Ok(Transformed::yes(
                Arc::clone(&node).with_new_children(vec![input])?,
            ))
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

/// `plan` with the scan at its bottom reading in id order, and each filter
/// on the way handing up batches of at most `fetch` rows, or `None` if the
/// walk stops first or the sort's column isn't the scan's `_id`. `column` is
/// the sort column's index in `plan`'s output.
fn in_id_order(
    plan: &Arc<dyn ExecutionPlan>,
    column: usize,
    descending: bool,
    fetch: usize,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(scan) = plan.downcast_ref::<CachedScanExec>() {
        return Ok(scan.in_id_order(column, descending));
    }
    let below = if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        // A filter's own projection picks its output from its input's columns.
        match filter.projection() {
            Some(projection) => projection.get(column).copied(),
            None => Some(column),
        }
    } else if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        // Only a column passed through as it is keeps its order.
        projection
            .expr()
            .get(column)
            .and_then(|e| e.expr.downcast_ref::<Column>())
            .map(Column::index)
    } else if plan.downcast_ref::<MeteredExec>().is_some() {
        Some(column)
    } else {
        // A join, an aggregate, another sort ends the walk.
        None
    };
    let Some(below) = below else {
        return Ok(None);
    };
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    let Some(child) = in_id_order(child, below, descending, fetch)? else {
        return Ok(None);
    };
    let rebuilt = Arc::clone(plan).with_new_children(vec![child])?;
    Ok(Some(match rebuilt.downcast_ref::<FilterExec>() {
        Some(filter) => {
            let rows = fetch.clamp(1, filter.batch_size());
            Arc::new(filter.with_batch_size(rows)?) as Arc<dyn ExecutionPlan>
        }
        None => rebuilt,
    }))
}
