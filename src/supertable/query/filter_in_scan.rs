// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`FilterIntoCachedScan`]: a `WHERE` over a scan of kept decoded columns is
//! checked in the scan too, on the columns as the cache keeps them.
//!
//! The cache keeps a string column of few values as a dictionary, and the
//! scan returns strings: handing every row up to the filter casts every
//! row's key to its string first, though the filter may keep a few rows in a
//! hundred. Checked in the scan, a condition on such a column is checked
//! once per value of the dictionary, and only the rows that pass are cast:
//!
//! ```text
//!   FilterExec  name = 'kernel' AND ts >= t       FilterExec  (same)
//!     CachedScanExec                                CachedScanExec predicate=(same)
//!       every row: keys cast to strings               keys checked by value, then
//!                                                     the passing rows cast
//! ```
//!
//! The filter stays above and checks every row it gets, so the answer is the
//! one it gives; the scan only hands it fewer. The rule fires on a filter
//! whose input is the scan, through the scan's meter, and leaves alone a scan
//! that already checks a `WHERE` or stops after a number of rows.
//!
//! It also leaves alone a scan spread by a round-robin repartition, as a scan
//! of fewer row groups than the statement's partitions is (a small table, or
//! one whose `WHERE` prunes most of them): there the check costs more in the
//! scan's few partitions than in the filter's many.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DfResult,
    physical_expr_common::physical_expr::is_volatile,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{ExecutionPlan, filter::FilterExec},
};

use crate::supertable::query::exec::{cached_scan_exec::CachedScanExec, metered_exec::MeteredExec};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "FilterIntoCachedScan";

/// Hands a filter's `WHERE` to the scan of kept decoded columns below it; see
/// the module docs.
#[derive(Debug, Default)]
pub(crate) struct FilterIntoCachedScan;

impl PhysicalOptimizerRule for FilterIntoCachedScan {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(filter) = node.downcast_ref::<FilterExec>() else {
                return Ok(Transformed::no(node));
            };
            if is_volatile(filter.predicate()) {
                return Ok(Transformed::no(node));
            }
            let input = filter.input();
            let (meter, scan) = match input.downcast_ref::<MeteredExec>() {
                Some(meter) => (Some(input), meter.input()),
                None => (None, input),
            };
            let Some(cached) = scan.downcast_ref::<CachedScanExec>() else {
                return Ok(Transformed::no(node));
            };
            // A scan that stops at `fetch` rows hands on the matches among
            // its first rows; checking the `WHERE` first would hand on the
            // first matches instead.
            if cached.has_predicate() || cached.fetch().is_some() {
                return Ok(Transformed::no(node));
            }
            let checked: Arc<dyn ExecutionPlan> =
                Arc::new(cached.with_predicate(Arc::clone(filter.predicate())));
            let input = match meter {
                Some(meter) => Arc::clone(meter).with_new_children(vec![checked])?,
                None => checked,
            };
            Ok(Transformed::yes(node.with_new_children(vec![input])?))
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
