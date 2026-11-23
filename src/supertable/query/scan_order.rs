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
//!
//! A scan that runs its `WHERE` as a Parquet row filter under a plain `LIMIT`
//! (`RowFilterUnderTopK`) takes the limit itself. Given the limit, the Parquet
//! reader reads only the row groups its statistics prove match, skipping
//! earlier ones, and which it skips depends on the limit. So the rule reads
//! every such scan in file order, gathered or not, through the limit
//! `LimitPushdown` leaves between it and its meter under an `OFFSET`, and
//! drops a round-robin exchange above one of a single partition. Its meter
//! hands the scan the limit only when the scan can read in file order
//! ([`can_read_in_file_order`]); the limit stays above it otherwise.
//!
//! ```text
//!   ProjectionExec                            ProjectionExec
//!     RepartitionExec RoundRobinBatch           MeteredExec
//!       MeteredExec                               GlobalLimitExec skip=m fetch=n
//!         GlobalLimitExec skip=m fetch=n            DataSourceExec limit=m+n, preserve_order
//!           DataSourceExec limit=m+n
//! ```

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    datasource::{physical_plan::FileScanConfig, source::DataSourceExec},
    error::Result as DfResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties,
        coalesce_partitions::CoalescePartitionsExec,
        filter::FilterExec,
        limit::{GlobalLimitExec, LocalLimitExec},
        projection::ProjectionExec,
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
                if let Some(meter) = node.downcast_ref::<MeteredExec>()
                    && meter.passes_limits_to_row_filter()
                {
                    // A scan that runs its `WHERE` as a row filter and took a
                    // plain `LIMIT`, gathered or not.
                    return Ok(match in_file_order(&node)? {
                        Some(ordered) => Transformed::yes(ordered),
                        None => Transformed::no(node),
                    });
                }
                Ok(match round_robin_input(&node) {
                    Some(input) if over_file_order(input) => Transformed::yes(Arc::clone(input)),
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
/// when it is a table scan below its meter, filters, projections and limits,
/// in any order; else `None`. Under an `OFFSET`, `LimitPushdown` leaves a
/// `GlobalLimitExec` with the skip between a row-filtered scan and its meter.
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
        || plan.downcast_ref::<LocalLimitExec>().is_some()
        || plan.downcast_ref::<GlobalLimitExec>().is_some();
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

/// Whether `plan`, a table scan below its meter, filters, projections and
/// limits, can read in file order. A meter hands a plain `LIMIT` to a scan
/// that runs its `WHERE` as a row filter only then
/// ([`MeteredExec::passing_limits`]).
pub(crate) fn can_read_in_file_order(plan: &Arc<dyn ExecutionPlan>) -> bool {
    in_file_order(plan).is_ok_and(|ordered| ordered.is_some())
}

/// `plan` gives its rows in file order on one partition: it is an
/// [`OrderedCoalesceExec`], or a scan that runs its `WHERE` as a row filter
/// under a plain `LIMIT` in one partition, which this rule reads in file
/// order, or a node that keeps its one input's order over either.
fn over_file_order(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<OrderedCoalesceExec>().is_some() {
        return true;
    }
    if let Some(meter) = plan.downcast_ref::<MeteredExec>()
        && meter.passes_limits_to_row_filter()
    {
        return meter.supports_limit_pushdown()
            && plan.output_partitioning().partition_count() == 1;
    }
    match plan.children()[..] {
        [child] if plan.maintains_input_order() == [true] => over_file_order(child),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    //! Files of several row groups, which a Parquet scan splits into byte
    //! ranges across partitions and a scan of kept columns reads as runs of
    //! row groups, with and without deleted rows. Unit tests, as setting a
    //! table's row group size is internal.

    use std::{ops::Range, sync::Arc};

    use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::{
        logical_expr::Expr,
        physical_plan::{collect, displayable},
        prelude::{SessionContext, col, lit},
    };
    use tempfile::TempDir;
    use tokio::runtime::{Builder, Runtime};

    use super::ORDERED_FETCH_BATCHES;
    use crate::{
        memory::{ConnectionMemoryBudget, budgeted_session_context},
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{Supertable, SupertableOptions, query::column_cache::DecodedColumnCache},
    };

    /// Superfiles of `t`, fewer than the scan's partitions, so the scan
    /// splits them.
    const FILES: i64 = 3;
    /// Rows in each superfile: file `f` holds `seq` `f * ROWS .. (f + 1) * ROWS`.
    const ROWS: i64 = 4_000;
    /// Rows per row group: eight row groups to a superfile.
    const ROW_GROUP_ROWS: usize = 500;
    /// Partitions of the scan.
    const PARTITIONS: usize = 4;
    /// Rows per page.
    const PAGE: usize = 700;
    /// Times each statement runs, to catch a run that reads in another order.
    const RUNS: usize = 10;
    /// Bytes of decoded columns a table keeping them holds: all of `t`'s.
    const KEPT_BYTES: usize = 64 * 1024 * 1024;
    /// Width of each row's `pad`, which spreads a superfile over enough bytes
    /// to split.
    const PAD_WIDTH: usize = 64;
    /// The filtered pages keep the rows whose `seq` isn't a multiple of this.
    const FILTER_MODULUS: i64 = 3;
    /// [`delete_some`] deletes every row whose `seq` is a multiple of this.
    const DELETE_MODULUS: i64 = 7;
    /// [`delete_some`] deletes this whole row group, the second of the second
    /// superfile.
    const DELETED_ROW_GROUP: Range<i64> = 4_500..5_000;
    /// [`delete_some`] deletes this run across the second and third superfiles.
    const DELETED_ACROSS_FILES: Range<i64> = 7_900..8_100;

    /// How `t` is read.
    #[derive(Clone, Copy, Debug)]
    enum Scan {
        /// A Parquet scan of the superfiles, split into byte ranges.
        Parquet,
        /// A scan of kept decoded columns, in runs of row groups.
        Kept,
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("pad", DataType::LargeUtf8, false),
        ]))
    }

    /// `t`, [`FILES`] superfiles of [`ROWS`] rows in row groups of
    /// [`ROW_GROUP_ROWS`], read by `scan`.
    fn table(dir: &TempDir, scan: Scan) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        options.row_group_size = Some(ROW_GROUP_ROWS);
        // Set either way, so a host's process-wide cache doesn't decide the scan.
        options.decoded_columns = match scan {
            Scan::Parquet => None,
            Scan::Kept => Some(Arc::new(DecodedColumnCache::new(KEPT_BYTES))),
        };
        let st = Supertable::create(options).expect("create");
        for f in 0..FILES {
            let seqs: Vec<i64> = (f * ROWS..(f + 1) * ROWS).collect();
            let pads: Vec<String> = seqs.iter().map(|s| format!("{s:0>PAD_WIDTH$}")).collect();
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(LargeStringArray::from(pads)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st
    }

    /// A session over `st` as `t` in [`PARTITIONS`] partitions, splitting a
    /// Parquet scan of any size into byte ranges.
    fn session(runtime: &Runtime, st: &Supertable) -> SessionContext {
        let ctx =
            budgeted_session_context(&ConnectionMemoryBudget::measured(), None).expect("session");
        runtime.block_on(async {
            for set in [
                format!("SET datafusion.execution.target_partitions = {PARTITIONS}"),
                "SET datafusion.optimizer.repartition_file_min_size = 0".to_owned(),
            ] {
                ctx.sql(&set).await.expect("set");
            }
        });
        st.register_into(&ctx, "t").expect("register");
        ctx
    }

    /// The plan of `sql` and the `seq`s it returns, in order.
    fn run(runtime: &Runtime, ctx: &SessionContext, sql: &str) -> (String, Vec<i64>) {
        runtime.block_on(async {
            let plan = ctx
                .sql(sql)
                .await
                .expect("plan")
                .create_physical_plan()
                .await
                .expect("physical plan");
            let shape = displayable(plan.as_ref()).indent(true).to_string();
            let batches = collect(plan, ctx.task_ctx()).await.expect("collect");
            let seqs = batches
                .iter()
                .flat_map(|b| {
                    let seq = b
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .expect("Int64")
                        .clone();
                    (0..seq.len()).map(move |i| seq.value(i))
                })
                .collect();
            (shape, seqs)
        })
    }

    /// `shape` gathers its partitions one after another: through an
    /// `OrderedCoalesceExec`, alone or under the staged gather.
    fn gathers_in_order(shape: &str) -> bool {
        shape.contains("OrderedCoalesceExec")
            || shape
                .lines()
                .any(|line| line.contains("StagedCoalesceExec:") && line.ends_with(", ordered"))
    }

    /// `shape` reads `t` by `scan` in file order across partitions: some
    /// superfile split into byte ranges, listed as `path:start..end` with a
    /// `start` past zero, or kept columns read in file order by
    /// [`PARTITIONS`] partitions sharing more row groups than that.
    fn reads_in_file_order(shape: &str, scan: Scan) -> bool {
        gathers_in_order(shape)
            && match scan {
                Scan::Parquet => shape
                    .split(".parquet:")
                    .skip(1)
                    .any(|range| !range.starts_with("0..")),
                Scan::Kept => shape.lines().any(|line| {
                    let row_groups = line
                        .split("CachedScanExec: row_groups=")
                        .nth(1)
                        .and_then(|rest| rest.split(',').next())
                        .and_then(|n| n.parse::<usize>().ok());
                    row_groups.is_some_and(|n| n > PARTITIONS)
                        && line.contains(&format!(", partitions={PARTITIONS},"))
                        && line.contains(", file_order")
                }),
            }
    }

    /// The whole of `select` is `want`, read by `scan` in file order, and
    /// every page of it, [`PAGE`] rows at growing offsets, is the slice of
    /// `want` it starts at, on every run.
    fn pages_cover_each_row_once(
        runtime: &Runtime,
        ctx: &SessionContext,
        scan: Scan,
        select: &str,
        want: &[i64],
    ) {
        // The whole read takes the file-order path only within the fetch the
        // rule orders.
        let batch_size = ctx.state().config().batch_size();
        let whole = usize::try_from(FILES * ROWS).expect("rows");
        assert!(
            whole <= ORDERED_FETCH_BATCHES * batch_size,
            "{whole} rows is past the {ORDERED_FETCH_BATCHES} batches of {batch_size} the rule orders"
        );
        let whole_sql = format!("{select} LIMIT {whole}");
        // The first read fills the kept columns a scan of them reads from.
        run(runtime, ctx, &whole_sql);
        let (shape, whole) = run(runtime, ctx, &whole_sql);
        assert!(
            reads_in_file_order(&shape, scan),
            "{scan:?}: {whole_sql} not read in file order:\n{shape}"
        );
        assert_eq!(
            whole, want,
            "{scan:?}: {whole_sql}: each row once, in file order"
        );
        for (p, page) in want.chunks(PAGE).enumerate() {
            let sql = format!("{select} LIMIT {PAGE} OFFSET {}", p * PAGE);
            for r in 0..RUNS {
                let (shape, seqs) = run(runtime, ctx, &sql);
                // A page's fetch is small enough that DataFusion may read the
                // superfiles whole, in groups of files, or in one group, which
                // is in order as is.
                let in_order = match scan {
                    Scan::Parquet => {
                        gathers_in_order(&shape) || shape.contains("file_groups={1 group:")
                    }
                    Scan::Kept => reads_in_file_order(&shape, scan),
                };
                assert!(in_order, "{scan:?}: {sql} not read in file order:\n{shape}");
                assert_eq!(seqs, page, "{scan:?} run {r}: {sql}");
            }
        }
    }

    /// Pages of `t` read by `scan` after deleting the rows `deleted` names
    /// (`predicate`), unfiltered and filtered, return each live row once.
    fn pages_return_each_live_row_once(
        scan: Scan,
        predicate: Option<Expr>,
        deleted: fn(i64) -> bool,
    ) {
        let runtime = Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime");
        let dir = TempDir::new().expect("tempdir");
        let st = table(&dir, scan);
        if let Some(predicate) = predicate {
            st.delete(predicate).expect("delete");
        }
        let ctx = session(&runtime, &st);
        let live: Vec<i64> = (0..FILES * ROWS).filter(|&s| !deleted(s)).collect();
        pages_cover_each_row_once(&runtime, &ctx, scan, "SELECT seq FROM t", &live);
        let kept: Vec<i64> = live
            .iter()
            .copied()
            .filter(|s| s % FILTER_MODULUS != 0)
            .collect();
        let filtered = format!("SELECT seq FROM t WHERE seq % {FILTER_MODULUS} <> 0");
        pages_cover_each_row_once(&runtime, &ctx, scan, &filtered, &kept);
    }

    /// `seq` in `range`.
    fn seq_in(range: Range<i64>) -> Expr {
        col("seq")
            .gt_eq(lit(range.start))
            .and(col("seq").lt(lit(range.end)))
    }

    /// Rows deleted by [`delete_some`]: every [`DELETE_MODULUS`]th,
    /// [`DELETED_ROW_GROUP`], and [`DELETED_ACROSS_FILES`].
    fn deleted_some(seq: i64) -> bool {
        seq % DELETE_MODULUS == 0
            || DELETED_ROW_GROUP.contains(&seq)
            || DELETED_ACROSS_FILES.contains(&seq)
    }

    /// The predicate deleting [`deleted_some`]'s rows.
    fn delete_some() -> Expr {
        (col("seq") % lit(DELETE_MODULUS))
            .eq(lit(0i64))
            .or(seq_in(DELETED_ROW_GROUP))
            .or(seq_in(DELETED_ACROSS_FILES))
    }

    #[test]
    fn pages_over_files_split_into_byte_ranges_return_each_row_once() {
        pages_return_each_live_row_once(Scan::Parquet, None, |_| false);
    }

    #[test]
    fn pages_over_runs_of_kept_row_groups_return_each_row_once() {
        pages_return_each_live_row_once(Scan::Kept, None, |_| false);
    }

    #[test]
    fn pages_over_byte_ranges_after_a_delete_return_each_live_row_once() {
        pages_return_each_live_row_once(Scan::Parquet, Some(delete_some()), deleted_some);
    }

    #[test]
    fn pages_over_kept_row_groups_after_a_delete_return_each_live_row_once() {
        pages_return_each_live_row_once(Scan::Kept, Some(delete_some()), deleted_some);
    }

    #[test]
    fn a_scan_that_cannot_read_in_file_order_is_not_handed_a_plain_limit() {
        // `LIMIT 10 OFFSET 5` over the meter a row-filtered scan gets, but
        // over a scan that can't read in file order (an in-memory one).
        //  - given the limit, such a scan could skip rows by the limit, and
        //    no rule could put them back in order, so pages could skip or
        //    repeat rows. The meter refuses the limit; it stays above.
        // This test pins where the limit lands and the rows of the page.
        use datafusion::{
            datasource::memory::MemorySourceConfig,
            physical_optimizer::{PhysicalOptimizerRule, limit_pushdown::LimitPushdown},
            physical_plan::{ExecutionPlan, limit::GlobalLimitExec},
        };

        use super::{ScanOrderUnderLimit, can_read_in_file_order};
        use crate::supertable::query::exec::metered_exec::MeteredExec;

        let schema = Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..100))],
        )
        .expect("batch");
        let scan: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).expect("scan");
        assert!(!can_read_in_file_order(&scan));
        let meter =
            MeteredExec::without_limit_pushdown(Arc::clone(&scan), None).passing_limits(scan);
        let limit: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(meter, 5, Some(10)));
        let config = datafusion::config::ConfigOptions::default();
        let plan = LimitPushdown::new()
            .optimize(limit, &config)
            .and_then(|plan| ScanOrderUnderLimit.optimize(plan, &config))
            .expect("optimize");
        let shape = displayable(plan.as_ref()).indent(true).to_string();
        let [meter] = plan.children()[..] else {
            panic!("{shape}");
        };
        assert!(plan.downcast_ref::<GlobalLimitExec>().is_some(), "{shape}");
        assert!(meter.downcast_ref::<MeteredExec>().is_some(), "{shape}");
        let [scan] = meter.children()[..] else {
            panic!("{shape}");
        };
        assert_eq!(scan.fetch(), None, "{shape}");

        let runtime = Builder::new_current_thread().build().expect("runtime");
        let batches = runtime
            .block_on(collect(plan, SessionContext::new().task_ctx()))
            .expect("collect");
        let seqs: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                let seq = b
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("Int64")
                    .clone();
                (0..seq.len()).map(move |i| seq.value(i))
            })
            .collect();
        assert_eq!(seqs, (5..15).collect::<Vec<i64>>());
    }
}
