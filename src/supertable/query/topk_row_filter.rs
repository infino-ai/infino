// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Run a scan's `WHERE` as a Parquet row filter when the scan feeds an
//! `ORDER BY ... LIMIT`. Every other scan keeps it in a `FilterExec` above.
//!
//! `ORDER BY t LIMIT 10` keeps a cutoff, the 10th best `t` seen so far, and
//! DataFusion hands it down into the scan. As a row filter, the cutoff drops a
//! row before its other columns are decoded, and skips whole row groups once
//! it is tight:
//!
//! ```text
//!   SortExec TopK(fetch=10)       cutoff: t < 1372720120
//!          │                              │ handed down
//!   DataSourceExec   predicate = url LIKE '%x%' AND t < 1372720120
//!                    decode `url` and `t` first, the rest only for rows that pass
//! ```
//!
//! Without a `LIMIT` there is no cutoff, and a row filter pays only when the
//! `WHERE` keeps almost no rows, which we can't know before reading them.
//!
//! The rule:
//!   - runs before DataFusion's own rules and only marks the scan. DataFusion's
//!     filter pushdown then moves the `WHERE` into it and drops the
//!     `FilterExec`, which would hold rows back from the sort until it has a
//!     full batch, so the cutoff would never tighten.
//!   - walks down only through filters, projections and our scan meter. A join,
//!     an aggregate or another sort stops it.
//!   - only fires for a `LIMIT` up to [`MAX_FETCH`]. A bigger limit keeps a
//!     loose cutoff that drops few rows, and the row filter's second read pass
//!     then costs more than it saves on object storage.
//!   - only fires when the columns it can skip hold at least [`MIN_SKIP_RATIO`]
//!     times the bytes of the sort's columns. The cutoff reads those anyway, so
//!     a scan of small extra columns gains less than the second pass costs.

use std::{collections::HashSet, sync::Arc};

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{
        physical_plan::{FileScanConfig, FileScanConfigBuilder, ParquetSource},
        source::DataSourceExec,
    },
    error::Result as DfResult,
    object_store::path::Path as ObjPath,
    physical_expr::utils::collect_columns,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, filter::FilterExec, projection::ProjectionExec, sorts::sort::SortExec,
    },
};

use crate::supertable::query::exec::metered_exec::{MeteredExec, ScanFooters};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "RowFilterUnderTopK";

/// Largest `LIMIT` the rule turns the row filter on for.
const MAX_FETCH: usize = 10_000;

/// Least ratio of the bytes the row filter can skip to the bytes of the sort's
/// columns for the rule to turn it on.
const MIN_SKIP_RATIO: u64 = 4;

/// Turns on the row filter for a scan under `ORDER BY ... LIMIT`; see the
/// module docs.
#[derive(Debug, Default)]
pub(crate) struct RowFilterUnderTopK;

impl PhysicalOptimizerRule for RowFilterUnderTopK {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let Some(sort) = node
                .downcast_ref::<SortExec>()
                .filter(|sort| sort.fetch().is_some_and(|fetch| fetch <= MAX_FETCH))
            else {
                return Ok(Transformed::no(node));
            };
            // Columns the cutoff reads.
            let sort_columns: HashSet<String> = sort
                .expr()
                .iter()
                .flat_map(|e| collect_columns(&e.expr))
                .map(|c| c.name().to_owned())
                .collect();
            Ok(match row_filtered(sort.input(), &sort_columns)? {
                Some(input) => Transformed::yes(Arc::clone(&node).with_new_children(vec![input])?),
                None => Transformed::no(node),
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

/// `plan` with the row filter turned on in the scan at the bottom, or `None`
/// when the walk stops before reaching one.
fn row_filtered(
    plan: &Arc<dyn ExecutionPlan>,
    sort_columns: &HashSet<String>,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    // Reached our table scan: the meter and the Parquet scan it wraps.
    if let Some(meter) = plan.downcast_ref::<MeteredExec>() {
        let (Some(scan), Some(footers)) = (
            meter.input().downcast_ref::<DataSourceExec>(),
            meter.footers(),
        ) else {
            return Ok(None);
        };
        return with_row_filter(scan, footers, sort_columns)
            .map(|scan| Arc::clone(plan).with_new_children(vec![scan]))
            .transpose();
    }
    // A node that groups, joins or reorders rows ends the walk.
    if !passes_rows_through(plan) {
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    // Rebuild this node over the marked scan below it.
    row_filtered(child, sort_columns)?
        .map(|child| Arc::clone(plan).with_new_children(vec![child]))
        .transpose()
}

/// Nodes that pass rows on one at a time, so the cutoff means the same thing at
/// the scan. Repartitions and coalesces are added later, by DataFusion's rules.
fn passes_rows_through(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<FilterExec>().is_some() || plan.downcast_ref::<ProjectionExec>().is_some()
}

/// `scan` with its row filter turned on, or `None` to leave it as it is.
fn with_row_filter(
    scan: &DataSourceExec,
    footers: &ScanFooters,
    sort_columns: &HashSet<String>,
) -> Option<Arc<dyn ExecutionPlan>> {
    // Only a Parquet scan has a row filter.
    let config = scan.data_source().downcast_ref::<FileScanConfig>()?;
    let parquet = config.file_source().downcast_ref::<ParquetSource>()?;
    // Worth it only when the columns the cutoff doesn't read are big enough.
    let (sort_bytes, skippable_bytes) = column_bytes(config, footers, sort_columns)?;
    if skippable_bytes < MIN_SKIP_RATIO * sort_bytes.max(1) {
        return None;
    }
    // The `WHERE` itself arrives later, from DataFusion's filter pushdown.
    let source = parquet
        .clone()
        .with_pushdown_filters(true)
        .with_reorder_filters(true);
    let config = FileScanConfigBuilder::from(config.clone())
        .with_source(Arc::new(source))
        .build();
    Some(DataSourceExec::from_data_source(config))
}

/// Stored bytes of the columns `config` reads, as `(sort's columns, the rest)`,
/// summed over the scan's files. One pass per footer, so a wide table with
/// many files stays cheap to plan.
///
/// Columns match by top-level name, so `ORDER BY` an alias counts the aliased
/// column as skippable. That can only turn on a row filter that saves little;
/// the rows are the same either way.
fn column_bytes(
    config: &FileScanConfig,
    footers: &ScanFooters,
    sort_columns: &HashSet<String>,
) -> Option<(u64, u64)> {
    let schema = config.projected_schema().ok()?;
    let read: HashSet<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    // A file split into byte ranges sits in several groups; count it once.
    let paths: HashSet<&ObjPath> = config
        .file_groups
        .iter()
        .flat_map(|group| group.iter())
        .map(|file| &file.object_meta.location)
        .collect();
    let (mut sort_bytes, mut skippable_bytes) = (0, 0);
    // Per leaf column: `Some(true)` the sort reads it, `Some(false)` it can be
    // skipped, `None` the scan doesn't read it.
    let mut is_sort: Vec<Option<bool>> = Vec::new();
    for path in paths {
        // Take the footer out, so the cache's lock isn't held while summing.
        let Some(footer) = footers.get(path).map(|f| Arc::clone(f.value())) else {
            continue;
        };
        // A nested column's leaves are named `item` and the like; its first
        // path part is the column.
        is_sort.clear();
        is_sort.extend(
            footer
                .file_metadata()
                .schema_descr()
                .columns()
                .iter()
                .map(|leaf| {
                    let column = leaf.path().parts().first()?.as_str();
                    read.contains(column).then(|| sort_columns.contains(column))
                }),
        );
        for row_group in footer.row_groups() {
            for (chunk, is_sort) in row_group.columns().iter().zip(&is_sort) {
                let bytes = chunk.uncompressed_size().max(0) as u64;
                match is_sort {
                    Some(true) => sort_bytes += bytes,
                    Some(false) => skippable_bytes += bytes,
                    None => {}
                }
            }
        }
    }
    Some((sort_bytes, skippable_bytes))
}
