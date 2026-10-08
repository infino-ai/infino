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
//!     an aggregate or another sort stops it. Through a projection, a sort
//!     column becomes the scan columns it's computed from (`length(s)` → `s`).
//!   - skips the scan if part of the `WHERE` can't move into it, like
//!     `random()` or a list column. That part stays in a `FilterExec`, which
//!     holds rows back from the sort, so the cutoff never tightens.
//!   - only fires for a `LIMIT` up to [`MAX_FETCH`]. A bigger limit keeps a
//!     loose cutoff that drops few rows, and the row filter's second read pass
//!     then costs more than it saves on object storage.
//!   - only fires when the columns it can skip hold at least [`MIN_SKIP_RATIO`]
//!     times the bytes of the sort's columns. The cutoff reads those anyway, so
//!     a scan of small extra columns gains less than the second pass costs.
//!     Bytes are uncompressed, as that's what decoding costs. A cold read pays
//!     compressed bytes, so if the skipped columns compress well, the ratio
//!     overstates what we save.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{
        physical_plan::{
            FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
            parquet::can_expr_be_pushed_down_with_schemas,
        },
        source::DataSourceExec,
    },
    error::Result as DfResult,
    object_store::path::Path as ObjPath,
    physical_expr::{
        PhysicalExpr,
        utils::{collect_columns, split_conjunction},
    },
    physical_expr_common::physical_expr::is_volatile,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, filter::FilterExec, projection::ProjectionExec, sorts::sort::SortExec,
    },
};
use rustc_hash::FxHashSet;

use crate::supertable::query::{exec::metered_exec::MeteredExec, provider::ScanFooters};

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
            let sort_columns = column_names(sort.expr().iter().map(|e| &e.expr));
            let Some(input) = row_filtered(sort.input(), sort_columns, Vec::new())? else {
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

/// `plan` with the row filter turned on in the scan at the bottom, or `None`
/// if the walk stops first. `sort_columns` are the sort's columns as `plan`
/// names them; `filters` are the `WHERE` parts seen on the way down.
fn row_filtered(
    plan: &Arc<dyn ExecutionPlan>,
    mut sort_columns: FxHashSet<String>,
    mut filters: Vec<Arc<dyn PhysicalExpr>>,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    // Reached our table scan: the meter and the Parquet scan it wraps.
    if let Some(meter) = plan.downcast_ref::<MeteredExec>() {
        let (Some(scan), Some(footers)) = (
            meter.input().downcast_ref::<DataSourceExec>(),
            meter.footers(),
        ) else {
            return Ok(None);
        };
        return with_row_filter(scan, footers, &sort_columns, &filters)
            .map(|scan| Arc::clone(plan).with_new_children(vec![scan]))
            .transpose();
    }
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        // Collect the `WHERE` parts; the scan checks they can all move into it.
        filters.extend(split_conjunction(filter.predicate()).into_iter().cloned());
    } else if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        // A filter above uses this projection's names, not the scan's, so we
        // can't check it. Rare; stop.
        if !filters.is_empty() {
            return Ok(None);
        }
        // Map each sort column to the scan columns it's computed from:
        // `l = length(s)` becomes `s`.
        sort_columns = column_names(
            projection
                .expr()
                .iter()
                .filter(|e| sort_columns.contains(&e.alias))
                .map(|e| &e.expr),
        );
    } else {
        // A join, an aggregate or another sort ends the walk. Repartitions and
        // coalesces come later, from DataFusion's rules.
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    // Rebuild this node over the marked scan below it.
    row_filtered(child, sort_columns, filters)?
        .map(|child| Arc::clone(plan).with_new_children(vec![child]))
        .transpose()
}

/// Columns the `exprs` read, by name.
fn column_names<'a>(exprs: impl Iterator<Item = &'a Arc<dyn PhysicalExpr>>) -> FxHashSet<String> {
    exprs
        .flat_map(collect_columns)
        .map(|c| c.name().to_owned())
        .collect()
}

/// `scan` with its row filter turned on, or `None` to leave it as it is.
fn with_row_filter(
    scan: &DataSourceExec,
    footers: &ScanFooters,
    sort_columns: &FxHashSet<String>,
    filters: &[Arc<dyn PhysicalExpr>],
) -> Option<Arc<dyn ExecutionPlan>> {
    // Only a Parquet scan has a row filter.
    let config = scan.data_source().downcast_ref::<FileScanConfig>()?;
    let parquet = config.file_source().downcast_ref::<ParquetSource>()?;
    // Every part of the `WHERE` must move into the scan, so no `FilterExec`
    // is left.
    let schema = parquet.table_schema().table_schema();
    if !filters
        .iter()
        .all(|f| !is_volatile(f) && can_expr_be_pushed_down_with_schemas(f, schema))
    {
        return None;
    }
    // Worth it only when the columns the cutoff doesn't read are big enough.
    let (sort_bytes, skippable_bytes) = column_bytes(config, footers, sort_columns)?;
    // Zero sort bytes means the sort reads no column (`ORDER BY random()`).
    // No cutoff reaches the scan, so the row filter would only run the `WHERE`.
    if sort_bytes == 0 || skippable_bytes < MIN_SKIP_RATIO * sort_bytes {
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
/// over the scan's files. Each footer has its column totals, so this costs
/// files x columns.
fn column_bytes(
    config: &FileScanConfig,
    footers: &ScanFooters,
    sort_columns: &FxHashSet<String>,
) -> Option<(u64, u64)> {
    let schema = config.projected_schema().ok()?;
    let read: FxHashSet<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    // A file split into byte ranges sits in several groups; count it once.
    let paths: FxHashSet<&ObjPath> = config
        .file_groups
        .iter()
        .flat_map(|group| group.iter())
        .map(|file| &file.object_meta.location)
        .collect();
    let (mut sort_bytes, mut skippable_bytes) = (0, 0);
    for path in paths {
        // Take the footer out, so the cache's lock isn't held while summing.
        let Some(footer) = footers.get(path).map(|f| Arc::clone(f.value())) else {
            continue;
        };
        for (column, bytes) in footer.columns().filter(|(c, _)| read.contains(c)) {
            if sort_columns.contains(column) {
                sort_bytes += bytes;
            } else {
                skippable_bytes += bytes;
            }
        }
    }
    Some((sort_bytes, skippable_bytes))
}
