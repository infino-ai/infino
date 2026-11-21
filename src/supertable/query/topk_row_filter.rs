// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Run a scan's `WHERE` as a Parquet row filter when the scan feeds an
//! `ORDER BY ... LIMIT` or a plain `LIMIT`. Every other scan keeps it in a
//! `FilterExec` above.
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
//! Under a plain `LIMIT` the scan also takes the limit, which the Parquet
//! reader counts after the row filter. It decodes the `WHERE`'s columns, then
//! the other columns only for the first rows that pass, not a whole batch:
//!
//! ```text
//!   GlobalLimitExec fetch=10
//!          │
//!   DataSourceExec   predicate = t >= 1672531200000, limit=10
//!                    decode `t` first, the rest only for 10 rows that pass
//! ```
//!
//! Without a `LIMIT` there is no cutoff or limit, and a row filter pays only
//! when the `WHERE` keeps almost no rows, which we can't know before reading
//! them.
//!
//! The rule:
//!   - runs before DataFusion's own rules and only marks the scan. DataFusion's
//!     filter pushdown then moves the `WHERE` into it and drops the
//!     `FilterExec`, which would hold rows back from the sort until it has a
//!     full batch, so the cutoff would never tighten.
//!   - walks down only through filters, projections and our scan meter (and,
//!     under a plain `LIMIT`, its per-partition limit). A join, an aggregate
//!     or another sort stops it. Through a projection, a sort column becomes
//!     the scan columns it's computed from (`length(s)` → `s`).
//!   - skips the scan if part of the `WHERE` can't move into it, like
//!     `random()` or a list column. That part stays in a `FilterExec`, which
//!     holds rows back from the sort, so the cutoff never tightens.
//!   - only fires for a `LIMIT` up to [`MAX_FETCH`]. Under `ORDER BY`, a bigger
//!     limit keeps a loose cutoff that drops few rows, and the row filter's
//!     second read pass then costs more than it saves on object storage. A
//!     plain `LIMIT` takes the same bound.
//!   - only fires when the columns it can skip hold at least [`MIN_SKIP_RATIO`]
//!     times the bytes of the columns it reads first: the sort's under
//!     `ORDER BY`, the `WHERE`'s under a plain `LIMIT`. Those are read anyway,
//!     so a scan of small extra columns gains less than the second pass costs.
//!     Bytes are uncompressed, as that's what decoding costs. A cold read pays
//!     compressed bytes, so if the skipped columns compress well, the ratio
//!     overstates what we save.
//!   - under a plain `LIMIT`, only fires on a scan with a `WHERE` (one without
//!     takes the limit already) whose files read every row of the row groups
//!     they read. The Parquet reader's limit pruning swaps a file's row
//!     selections for whole row groups the statistics prove match, which would
//!     return the rows a tombstone deletes or an index ruled out.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{
        physical_plan::{
            FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
            parquet::{ParquetAccessPlan, RowGroupAccess, can_expr_be_pushed_down_with_schemas},
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
        ExecutionPlan,
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        sorts::sort::SortExec,
    },
};
use rustc_hash::FxHashSet;

use crate::supertable::query::{
    exec::{cached_scan_exec::CachedScanExec, metered_exec::MeteredExec},
    provider::ScanFooters,
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "RowFilterUnderTopK";

/// Largest `LIMIT` (with its `OFFSET`) the rule turns the row filter on for.
const MAX_FETCH: usize = 10_000;

/// Least ratio of the bytes the row filter can skip to the bytes of the
/// columns it reads first for the rule to turn it on.
const MIN_SKIP_RATIO: u64 = 4;

/// Turns on the row filter for a scan under `ORDER BY ... LIMIT` or a plain
/// `LIMIT`; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct RowFilterUnderTopK;

impl PhysicalOptimizerRule for RowFilterUnderTopK {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let (bound, fetch, input) = if let Some(sort) = node.downcast_ref::<SortExec>() {
                let sort_columns = column_names(sort.expr().iter().map(|e| &e.expr));
                (Bound::TopK(sort_columns), sort.fetch(), sort.input())
            } else if let Some(limit) = node.downcast_ref::<GlobalLimitExec>() {
                let fetch = limit.fetch().map(|fetch| limit.skip() + fetch);
                (Bound::Limit, fetch, limit.input())
            } else {
                return Ok(Transformed::no(node));
            };
            if fetch.is_none_or(|fetch| fetch > MAX_FETCH) {
                return Ok(Transformed::no(node));
            }
            let Some(input) = row_filtered(input, bound, Vec::new())? else {
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

/// What bounds the rows a scan hands up.
enum Bound {
    /// `ORDER BY ... LIMIT`, with the sort's columns as the plan below names
    /// them.
    TopK(FxHashSet<String>),
    /// A plain `LIMIT`.
    Limit,
}

/// `plan` with the row filter turned on in the scan at the bottom, or `None`
/// if the walk stops first. `filters` are the `WHERE` parts seen on the way
/// down.
fn row_filtered(
    plan: &Arc<dyn ExecutionPlan>,
    mut bound: Bound,
    mut filters: Vec<Arc<dyn PhysicalExpr>>,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    // Reached our table scan: the meter and the Parquet scan it wraps.
    if let Some(meter) = plan.downcast_ref::<MeteredExec>() {
        // A scan of kept decoded columns decodes every row of the columns it
        // keeps; the row filter decodes the columns the cutoff doesn't read
        // only for the rows that pass. Where the row filter applies, the
        // Parquet scan the kept columns stand in for serves the statement.
        if let Some(cached) = meter.input().downcast_ref::<CachedScanExec>() {
            return row_filtered(cached.parquet(), bound, filters);
        }
        let (Some(scan), Some(footers)) = (
            meter.input().downcast_ref::<DataSourceExec>(),
            meter.footers(),
        ) else {
            return Ok(None);
        };
        let Some(scan) = with_row_filter(scan, footers, &bound, &filters) else {
            return Ok(None);
        };
        return Ok(Some(match bound {
            Bound::TopK(_) => Arc::clone(plan).with_new_children(vec![scan])?,
            // The scan now runs the whole `WHERE` before it counts rows, so
            // the `LIMIT` may reach it.
            Bound::Limit => meter.passing_limits(scan),
        }));
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
        if let Bound::TopK(sort_columns) = &mut bound {
            *sort_columns = column_names(
                projection
                    .expr()
                    .iter()
                    .filter(|e| sort_columns.contains(&e.alias))
                    .map(|e| &e.expr),
            );
        }
    } else if matches!(bound, Bound::Limit) && plan.downcast_ref::<LocalLimitExec>().is_some() {
        // A partition's share of the same `LIMIT`.
    } else {
        // A join, an aggregate or another sort ends the walk. Repartitions and
        // coalesces come later, from DataFusion's rules.
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    // Rebuild this node over the marked scan below it.
    row_filtered(child, bound, filters)?
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
    bound: &Bound,
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
    // The columns read for every row: the sort's, whose cutoff checks each
    // row, or under a plain `LIMIT` the `WHERE`'s.
    let where_columns;
    let first_columns = match bound {
        Bound::TopK(sort_columns) => sort_columns,
        Bound::Limit => {
            if filters.is_empty() || selects_rows(config) {
                return None;
            }
            where_columns = column_names(filters.iter());
            &where_columns
        }
    };
    // Worth it only when the columns not read first are big enough.
    let (first_bytes, skippable_bytes) = column_bytes(config, footers, first_columns)?;
    // Zero bytes read first means the sort or `WHERE` reads no column
    // (`ORDER BY random()`). Nothing narrows the rows before the rest are
    // decoded, so the row filter would only run the `WHERE`.
    if first_bytes == 0 || skippable_bytes < MIN_SKIP_RATIO * first_bytes {
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

/// Whether a file of `config` reads only some rows of a row group it reads:
/// the rows a tombstone deletes or an index rules out are skipped that way.
fn selects_rows(config: &FileScanConfig) -> bool {
    config
        .file_groups
        .iter()
        .flat_map(|group| group.iter())
        .filter_map(|file| file.extension::<ParquetAccessPlan>())
        .any(|plan| {
            plan.inner()
                .iter()
                .any(|access| matches!(access, RowGroupAccess::Selection(_)))
        })
}

/// Stored bytes of the columns `config` reads, as `(first_columns, the rest)`,
/// over the scan's files. Each footer has its column totals, so this costs
/// files x columns.
fn column_bytes(
    config: &FileScanConfig,
    footers: &ScanFooters,
    first_columns: &FxHashSet<String>,
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
    let (mut first_bytes, mut skippable_bytes) = (0, 0);
    for path in paths {
        // Take the footer out, so the cache's lock isn't held while summing.
        let Some(footer) = footers.get(path).map(|f| Arc::clone(f.value())) else {
            continue;
        };
        for (column, bytes) in footer.columns().filter(|(c, _)| read.contains(c)) {
            if first_columns.contains(column) {
                first_bytes += bytes;
            } else {
                skippable_bytes += bytes;
            }
        }
    }
    Some((first_bytes, skippable_bytes))
}
