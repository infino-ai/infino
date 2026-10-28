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
//!
//! A filtered scan under a plain `LIMIT`, with no `ORDER BY`, reads each of
//! its files whole instead: one partition reads a file's byte ranges front to
//! back, and separate files are still read by separate partitions. Its fetch
//! can't reach the scan (see `MeteredExec::without_limit_pushdown`), so a file
//! split into byte ranges decodes a batch of every column in every range at
//! once, and the first rows to pass end the query:
//!
//! ```text
//!   one compacted superfile
//!   CoalescePartitionsExec(fetch=10)          CoalescePartitionsExec(fetch=10)
//!     FilterExec  price > 500                   FilterExec  price > 500
//!       DataSourceExec  16 byte ranges            DataSourceExec  1 partition
//!       16 batches decoded, 10 rows kept          1 batch decoded, 10 rows kept
//! ```
//!
//! The `WHERE` stays in the `FilterExec`. A row filter would decode only the
//! `WHERE`'s columns for rows that fail, but a filter keeping rows at random
//! leaves a selection of thousands of fragments per row group, which measured
//! slower than decoding the batch whole.
//!
//! The trade: a filter that keeps almost nothing decodes every row either way,
//! so it uses the same CPU, but a file's ranges are now read one after another
//! instead of side by side, so on a table of a few large files it ends later,
//! by up to the number of ranges a file was split into. Separate files keep
//! their own partitions, so that cost is bounded by one file's ranges and
//! never grows with the table. `UnsplitLimitScans` does the joining, after
//! DataFusion's `EnforceDistribution` has split the files.

use std::{mem::take, sync::Arc};

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{
        listing::PartitionedFile,
        physical_plan::{
            FileGroup, FileScanConfig, FileScanConfigBuilder, FileSource, ParquetSource,
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
        ExecutionPlan, Partitioning,
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
        repartition::RepartitionExec,
        sorts::sort::SortExec,
    },
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::supertable::query::{exec::metered_exec::MeteredExec, provider::ScanFooters};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "RowFilterUnderTopK";

/// [`UnsplitLimitScans`]'s name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const UNSPLIT_RULE_NAME: &str = "UnsplitLimitScans";

/// Largest `LIMIT` the rule turns the row filter on for.
const MAX_FETCH: usize = 10_000;

/// Largest plain `LIMIT` a filtered scan reads its files whole for. A limit
/// this small is met by the first batch or so of rows that pass, so the
/// batches a file's other byte ranges decode at the same time are mostly thrown
/// away; a larger limit needs rows from more of the file, where reading its
/// ranges side by side pays. Set a little over one decode batch (DataFusion's
/// 8,192 rows) at the row filter's limit, not measured on its own.
const MAX_WHOLE_FILE_FETCH: usize = 10_000;

/// Least ratio of the bytes the row filter can skip to the bytes of the sort's
/// columns for the rule to turn it on.
const MIN_SKIP_RATIO: u64 = 4;

/// Turns on the row filter for a scan under `ORDER BY ... LIMIT`, and keeps a
/// filtered scan under a plain `LIMIT` whole; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct RowFilterUnderTopK;

impl PhysicalOptimizerRule for RowFilterUnderTopK {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let marked = if let Some(sort) = node
                .downcast_ref::<SortExec>()
                .filter(|sort| sort.fetch().is_some_and(|fetch| fetch <= MAX_FETCH))
            {
                let sort_columns = column_names(sort.expr().iter().map(|e| &e.expr));
                row_filtered(sort.input(), sort_columns, Vec::new())?
            } else if let Some(input) = small_limit_input(&node) {
                files_whole_under_limit(input)?
            } else {
                None
            };
            match marked {
                Some(input) => Ok(Transformed::yes(
                    Arc::clone(&node).with_new_children(vec![input])?,
                )),
                None => Ok(Transformed::no(node)),
            }
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

/// Joins each file's byte ranges back into one partition for a scan
/// [`RowFilterUnderTopK`] marked under a plain `LIMIT`, once DataFusion's
/// `EnforceDistribution` has split its files into ranges: that rule splits a
/// scan without asking the node above it. A file's ranges go to the group its
/// first range is in, in byte order with their access plans, so one partition
/// reads the file front to back and stops when the limit is met; separate
/// files stay in the groups they were in and are still read in parallel.
///
/// A round-robin exchange `EnforceDistribution` put next to the scan goes
/// too, above the meter or below it (below when the scan's files are too
/// small to split, so the meter declined): it would keep pulling batches out
/// of the scan ahead of the limit.
#[derive(Debug, Default)]
pub(crate) struct UnsplitLimitScans;

impl PhysicalOptimizerRule for UnsplitLimitScans {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            if let Some(input) = round_robin_input(&node)
                && input
                    .downcast_ref::<MeteredExec>()
                    .is_some_and(MeteredExec::reads_files_whole)
            {
                return Ok(Transformed::yes(Arc::clone(input)));
            }
            let Some(meter) = node
                .downcast_ref::<MeteredExec>()
                .filter(|meter| meter.reads_files_whole())
            else {
                return Ok(Transformed::no(node));
            };
            let input = round_robin_input(meter.input()).unwrap_or(meter.input());
            let Some(config) = input
                .downcast_ref::<DataSourceExec>()
                .and_then(|scan| scan.data_source().downcast_ref::<FileScanConfig>())
            else {
                return Ok(Transformed::no(node));
            };
            // No file split across groups and no exchange to drop: nothing to join.
            if !split_across_groups(&config.file_groups) && Arc::ptr_eq(input, meter.input()) {
                return Ok(Transformed::no(node));
            }
            // One clone of the config; its groups are moved out and rejoined.
            let mut config = config.clone();
            let groups = files_whole(take(&mut config.file_groups));
            let config = FileScanConfigBuilder::from(config)
                .with_file_groups(groups)
                .build();
            let scan = DataSourceExec::from_data_source(config);
            Ok(Transformed::yes(
                Arc::clone(&node).with_new_children(vec![scan])?,
            ))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        UNSPLIT_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// The input of `node` when it is a round-robin exchange, else `None`.
fn round_robin_input(node: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    node.downcast_ref::<RepartitionExec>()
        .filter(|exchange| matches!(exchange.partitioning(), Partitioning::RoundRobinBatch(_)))
        .map(RepartitionExec::input)
}

/// Whether any file of `groups` is split into byte ranges in more than one
/// group.
fn split_across_groups(groups: &[FileGroup]) -> bool {
    let mut home: FxHashMap<&ObjPath, usize> = FxHashMap::default();
    for (at, group) in groups.iter().enumerate() {
        for file in group.iter() {
            if *home.entry(&file.object_meta.location).or_insert(at) != at {
                return true;
            }
        }
    }
    false
}

/// `groups` with each file's byte ranges joined into the group its first
/// range is in, in byte order; the groups left empty are dropped. A file
/// whole in one group stays where it is, so files keep their own partitions
/// and only a file split into ranges loses the parallelism within it.
fn files_whole(groups: Vec<FileGroup>) -> Vec<FileGroup> {
    let n_groups = groups.len();
    // Each file's ranges with the group it goes to, in the order files first
    // appear.
    let mut files: Vec<(usize, Vec<PartitionedFile>)> = Vec::new();
    let mut index: FxHashMap<ObjPath, usize> = FxHashMap::default();
    for (at, group) in groups.into_iter().enumerate() {
        for file in group.into_inner() {
            let slot = *index
                .entry(file.object_meta.location.clone())
                .or_insert_with(|| {
                    files.push((at, Vec::new()));
                    files.len() - 1
                });
            files[slot].1.push(file);
        }
    }
    let mut out: Vec<Vec<PartitionedFile>> = (0..n_groups).map(|_| Vec::new()).collect();
    for (at, mut ranges) in files {
        ranges.sort_by_key(|file| file.range.as_ref().map_or(0, |range| range.start));
        out[at].extend(ranges);
    }
    out.into_iter()
        .filter(|group| !group.is_empty())
        .map(FileGroup::new)
        .collect()
}

/// The input of a `LIMIT` of at most [`MAX_WHOLE_FILE_FETCH`] rows (its skip
/// included), or `None` for any other node.
fn small_limit_input(node: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    let (input, rows) = if let Some(limit) = node.downcast_ref::<GlobalLimitExec>() {
        (
            limit.input(),
            limit.fetch().map(|f| f.saturating_add(limit.skip())),
        )
    } else if let Some(limit) = node.downcast_ref::<LocalLimitExec>() {
        (limit.input(), Some(limit.fetch()))
    } else {
        return None;
    };
    rows.filter(|&rows| rows <= MAX_WHOLE_FILE_FETCH)
        .map(|_| input)
}

/// `plan` with the meter at the bottom marked to read each file whole, under a
/// plain `LIMIT`: the walk goes down through filters and projections, and
/// `None` when it meets anything else, or a meter that takes the fetch itself
/// (a scan with no filter, which the fetch already stops).
fn files_whole_under_limit(
    plan: &Arc<dyn ExecutionPlan>,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(meter) = plan.downcast_ref::<MeteredExec>() {
        return Ok((!meter.supports_limit_pushdown()).then(|| meter.with_files_whole()));
    }
    if plan.downcast_ref::<FilterExec>().is_none()
        && plan.downcast_ref::<ProjectionExec>().is_none()
    {
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    files_whole_under_limit(child)?
        .map(|child| Arc::clone(plan).with_new_children(vec![child]))
        .transpose()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A file's size in the grouping cases below.
    const FILE_BYTES: u64 = 30;
    /// Bytes in each range a split file is cut into: three to a file.
    const RANGE_BYTES: i64 = 10;
    /// How [`layout`] shows a file that sits whole in its group.
    const WHOLE: i64 = -1;

    /// The `at`-th range of file `path`.
    fn range(path: &str, at: i64) -> PartitionedFile {
        PartitionedFile::new_with_range(
            path.to_string(),
            FILE_BYTES,
            at * RANGE_BYTES,
            (at + 1) * RANGE_BYTES,
        )
    }

    /// File `path`, whole.
    fn whole(path: &str) -> PartitionedFile {
        PartitionedFile::new(path.to_string(), FILE_BYTES)
    }

    /// Each group's files as (path, range start), [`WHOLE`] for a whole file.
    fn layout(groups: &[FileGroup]) -> Vec<Vec<(String, i64)>> {
        groups
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|file| {
                        let start = file.range.as_ref().map_or(WHOLE, |range| range.start);
                        (file.object_meta.location.to_string(), start)
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_split_files_ranges_join_and_other_files_keep_their_groups() {
        // `a` split in three over every group, out of byte order; `b` and `c`
        // whole in groups 1 and 2, as `EnforceDistribution` leaves a scan of
        // one large file and two small ones.
        //  - `a`'s ranges join the group its first one was in, in byte order.
        //  - `b` and `c` keep their own groups, so they still read in parallel.
        let groups = vec![
            FileGroup::new(vec![range("a", 2)]),
            FileGroup::new(vec![range("a", 0), whole("b")]),
            FileGroup::new(vec![range("a", 1), whole("c")]),
        ];
        assert!(split_across_groups(&groups));
        let joined = files_whole(groups);
        assert!(!split_across_groups(&joined));
        assert_eq!(
            layout(&joined),
            vec![
                vec![
                    ("a".to_string(), 0),
                    ("a".to_string(), RANGE_BYTES),
                    ("a".to_string(), 2 * RANGE_BYTES),
                ],
                vec![("b".to_string(), WHOLE)],
                vec![("c".to_string(), WHOLE)],
            ]
        );
    }

    #[test]
    fn whole_files_in_groups_of_their_own_stay_as_they_are() {
        // Files too small to split, one to a group: nothing to join, and each
        // file keeps its partition.
        let groups = vec![
            FileGroup::new(vec![whole("a")]),
            FileGroup::new(vec![whole("b")]),
        ];
        assert!(!split_across_groups(&groups));
        assert_eq!(
            layout(&files_whole(groups)),
            vec![
                vec![("a".to_string(), WHOLE)],
                vec![("b".to_string(), WHOLE)],
            ]
        );
    }
}
