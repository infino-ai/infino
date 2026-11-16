// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! DataFusion [`MemoryPool`] backed by a connection's [`ConnectionMemoryBudget`].
//!
//! SQL is the one path where DataFusion, not infino, allocates the working set
//! (sort / aggregate / join buffers). This pool charges those against the
//! connection budget and lets DataFusion spill instead of us reserving by hand.
//!
//! One counter per connection: a fresh pool per `SessionContext` still shares it.
//!
//! # Spill vs OverBudget
//!
//! For SQL query paths, when the gate refuses (`try_grow` returns `ResourcesExhausted`), what happens
//! next is up to the operator that asked for memory:
//!
//! - Spillable operator like sort, grouped aggregate, sort-merge join: frees
//!   memory by writing its buffered run to disk, then continues. The query still
//!   succeeds, just slower.
//! - Otherwise it surfaces as [`InfinoError::OverBudget`], when:
//!     - the operator can't spill at all (hash-join build side, nested-loop join), or
//!     - it is spillable but can't reserve even the minimum it needs to run the
//!       spill / merge (e.g. the sort's merge reservation).
//!
//! Some operators never ask this pool: the streaming ones — scan, filter,
//! projection, `unnest` — and window functions, which buffer without
//! reserving. Nothing they allocate is gated, however large; the process
//! memory limit (see `resident`) is what bounds them.
//!
//! Spilling needs a disk manager. A connection with a disk cache folder spills
//! into that folder, within a share of the disk ([`SqlSpill`]); any other uses
//! DataFusion's default (the OS temp dir).
//!

use std::{
    fmt,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

use datafusion::{
    error::{DataFusionError, Result as DfResult},
    execution::{
        disk_manager::{DEFAULT_MAX_TEMP_DIRECTORY_SIZE, DiskManager, DiskManagerMode},
        memory_pool::{MemoryLimit, MemoryPool, MemoryReservation},
        runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
        session_state::SessionStateBuilder,
    },
    physical_optimizer::optimizer::PhysicalOptimizer,
    prelude::{SessionConfig, SessionContext},
};
use fs4::total_space;
use tracing::warn;

use crate::{
    memory::ConnectionMemoryBudget,
    supertable::{
        query::{
            join_build_size::RightSizeJoinBuild,
            limit_stage::StageLimitGathers,
            read_ahead_rule::ReadAheadOff,
            sorted_root::{KeepSortedRoot, OrderUnderScalarSubquery},
            topk_row_filter::RowFilterUnderTopK,
            values_subquery::ValuesSubqueryRewrite,
        },
        reader_cache::RunScratch,
    },
};

/// A DataFusion memory pool over a [`ConnectionMemoryBudget`]: measured never
/// refuses, bounded refuses at the 90% gate (DataFusion then spills, or errors
/// if it can't).
#[derive(Debug)]
struct ConnectionBudgetPool {
    budget: Arc<ConnectionMemoryBudget>,
}

/// Pool name for DataFusion 54's `MemoryPool::name` + `Display` (both required;
/// used only in its diagnostics). The budget has no extra state worth printing.
const POOL_NAME: &str = "ConnectionBudgetPool";

/// Unique-key fraction above which DataFusion abandons the partial
/// (per-partition) pre-aggregation phase of a grouped aggregate. DataFusion's
/// default is 0.8; 0.5 abandons it sooner. On a high-cardinality `GROUP BY` the
/// partial phase barely reduces the row count yet re-hashes every key, so
/// skipping it and letting the final aggregate hash once is faster. Low-
/// cardinality groupings, which still dedup below this fraction, keep it.
const PARTIAL_AGG_SKIP_PROBE_RATIO: f64 = 0.5;

/// Folder, under a connection's disk cache folder, that its SQL spills into.
/// A table name cannot start with `_`, so it never names a table's cache.
const SQL_SPILL_SUBDIR: &str = "_sql-spill";

/// Share of the disk holding the cache folder that a connection's SQL may
/// spill to, in divisor form (`total / this`). The spill is never allowed
/// more than DataFusion's own default.
const SQL_SPILL_DISK_SHARE_DIVISOR: u64 = 10;

/// Where a connection's SQL spills when it has a disk cache folder: a folder
/// of this process's own under it, removed when the connection closes and,
/// after a crash, by the next connection to open the cache folder. One disk
/// manager serves every query of the connection, so together they spill at
/// most [`SQL_SPILL_DISK_SHARE_DIVISOR`]'s share of the disk; a query that
/// would pass it fails with DataFusion's resources error.
///
/// Opened on the first query: a connection that runs no SQL makes no folder.
pub(crate) struct SqlSpill {
    root: PathBuf,
    /// The most bytes the connection's queries spill together; `None` takes
    /// the disk's share.
    limit: Option<u64>,
    opened: OnceLock<Option<OpenSqlSpill>>,
}

struct OpenSqlSpill {
    /// Declared first, so its files go before the folder they are in.
    disk_manager: Arc<DiskManager>,
    _folder: RunScratch,
}

impl SqlSpill {
    /// The SQL spill of a connection whose disk cache folder is `cache_dir`.
    pub(crate) fn under(cache_dir: &Path) -> Arc<Self> {
        Arc::new(Self {
            root: cache_dir.join(SQL_SPILL_SUBDIR),
            limit: None,
            opened: OnceLock::new(),
        })
    }

    /// [`Self::under`], spilling at most `limit` bytes.
    #[cfg(test)]
    fn under_with_limit(cache_dir: &Path, limit: u64) -> Arc<Self> {
        Arc::new(Self {
            root: cache_dir.join(SQL_SPILL_SUBDIR),
            limit: Some(limit),
            opened: OnceLock::new(),
        })
    }

    /// The disk manager queries spill through. `None` when the folder could
    /// not be made, which leaves DataFusion's default.
    fn disk_manager(&self) -> Option<Arc<DiskManager>> {
        self.opened
            .get_or_init(|| match self.open() {
                Ok(opened) => Some(opened),
                Err(error) => {
                    warn!(
                        path = %self.root.display(),
                        %error,
                        "SQL spill folder not opened; queries spill to the OS temp directory"
                    );
                    None
                }
            })
            .as_ref()
            .map(|opened| Arc::clone(&opened.disk_manager))
    }

    fn open(&self) -> DfResult<OpenSqlSpill> {
        let folder = RunScratch::open(&self.root)?;
        let limit = self.limit.unwrap_or_else(|| {
            total_space(&self.root)
                .map(|total| total / SQL_SPILL_DISK_SHARE_DIVISOR)
                .unwrap_or(DEFAULT_MAX_TEMP_DIRECTORY_SIZE)
                .min(DEFAULT_MAX_TEMP_DIRECTORY_SIZE)
        });
        let disk_manager = DiskManager::builder()
            .with_mode(DiskManagerMode::Directories(vec![folder.dir()]))
            .with_max_temp_directory_size(limit)
            .build()?;
        Ok(OpenSqlSpill {
            disk_manager: Arc::new(disk_manager),
            _folder: folder,
        })
    }
}

impl fmt::Display for ConnectionBudgetPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(POOL_NAME)
    }
}

impl MemoryPool for ConnectionBudgetPool {
    fn name(&self) -> &str {
        POOL_NAME
    }

    fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
        self.budget.grow_unchecked(additional);
    }

    fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
        self.budget.release(shrink);
    }

    fn try_grow(&self, _reservation: &MemoryReservation, additional: usize) -> DfResult<()> {
        // DataFusion's spilling operators (sort, aggregate) call try_grow and, on `Err`, spill to disk
        // and retry rather than fail. They branch on the error variant (`ResourcesExhausted`), never on
        // its message, so the text here only surfaces if the query still can't fit after spilling.
        //
        // We label it "during SQL query" so that final error matches the ingest and vector over-budget
        // messages once it reaches InfinoError::OverBudget.
        self.budget.try_grow(additional).map_err(|over| {
            DataFusionError::ResourcesExhausted(format!("during SQL query, {over}"))
        })
    }

    fn reserved(&self) -> usize {
        self.budget.used()
    }

    fn memory_limit(&self) -> MemoryLimit {
        match self.budget.limit() {
            Some(limit) => MemoryLimit::Finite(limit),
            None => MemoryLimit::Infinite,
        }
    }
}

/// A `RuntimeEnv` whose memory pool is `budget`. Spillable operators spill
/// through `spill` when the connection has one, else through the default
/// disk manager (OS temp).
fn budgeted_runtime(
    budget: &Arc<ConnectionMemoryBudget>,
    spill: Option<&SqlSpill>,
) -> DfResult<Arc<RuntimeEnv>> {
    let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
        budget: Arc::clone(budget),
    });

    let mut runtime = RuntimeEnvBuilder::new().with_memory_pool(pool).build()?;
    if let Some(disk_manager) = spill.and_then(SqlSpill::disk_manager) {
        runtime.disk_manager = disk_manager;
    }
    Ok(Arc::new(runtime))
}

/// A `SessionContext` whose SQL allocations are gated by `budget` and whose
/// operators spill through `spill` when the connection has one.
pub(crate) fn budgeted_session_context(
    budget: &Arc<ConnectionMemoryBudget>,
    spill: Option<&SqlSpill>,
) -> DfResult<SessionContext> {
    let mut config = SessionConfig::new();
    // We scan string columns as `Utf8View` for fast comparisons, but a SQL
    // result should be `LargeUtf8`. This flag makes DataFusion convert every
    // `Utf8View` back to `LargeUtf8` at the query output, so the view stays
    // internal to the scan and never reaches the caller.
    //
    // It also dodges a DataFusion 54 bug: an ungrouped MIN/MAX over a `Utf8View`
    // column crashes the planner (`ProjectionPushdown`, `Utf8View`-vs-`LargeUtf8`
    // mismatch). Converting the output to `LargeUtf8` keeps the types consistent.
    //
    // One consequence: a column the user themselves declared `Utf8View` is also
    // converted, so SQL string results are always `LargeUtf8`, never a view.
    config.options_mut().optimizer.expand_views_at_output = true;

    // Skip DataFusion's partial (pre-)aggregation sooner on high-cardinality
    // GROUP BY (see PARTIAL_AGG_SKIP_PROBE_RATIO). Execution strategy only;
    // results are identical.
    config
        .options_mut()
        .execution
        .skip_partial_aggregation_probe_ratio_threshold = PARTIAL_AGG_SKIP_PROBE_RATIO;

    // Predicates run in a `FilterExec` above the scan, not as Parquet row
    // filters, except under `ORDER BY ... LIMIT` (see `RowFilterUnderTopK`).
    // Pinned off here because the scan's own flag is ORed with this option.
    config.options_mut().execution.parquet.pushdown_filters = false;

    // Expose the `information_schema` virtual tables (`tables`, `columns`, …)
    // so schema-introspection queries plan and run: `SHOW TABLES`,
    // `SHOW COLUMNS`, and `SELECT ... FROM information_schema.columns`. The
    // provider is synthesized lazily from the registered catalog, so it
    // reflects exactly the tables registered into the session.
    config.options_mut().catalog.information_schema = true;

    // DataFusion's rules, with ours at both ends:
    //  - `RowFilterUnderTopK` first, so DataFusion's filter pushdown sees the
    //    marked scans and drops their `FilterExec`.
    //  - `OrderUnderScalarSubquery` before DataFusion's `OutputRequirements`,
    //    which on DataFusion 54 loses the `ORDER BY` of a query under a
    //    `ScalarSubqueryExec` root.
    //  - `RightSizeJoinBuild` after DataFusion's, for two reasons: its node
    //    passes no filter or projection down, so it must not be in the plan
    //    while DataFusion pushes those into a join's build side, and
    //    `JoinSelection` swaps join sides, which would carry a node placed
    //    earlier onto the probe side. It copies build columns whose buffers
    //    are much larger than their rows, so the join is charged for the rows
    //    it keeps.
    //  - `KeepSortedRoot` after DataFusion's rules, to remove the round-robin
    //    repartition that `EnforceDistribution` adds above a sorted result,
    //    which would return an `ORDER BY` out of order.
    //  - `StageLimitGathers` after DataFusion's rules, once `EnforceDistribution`
    //    has split a filtered scan and `LimitPushdown` has set the gather's
    //    fetch, to start its partitions in two stages under a small `LIMIT`.
    //  - `ReadAheadOff` last, once DataFusion has placed every limit, to turn
    //    read-ahead off for scans a `LIMIT` can stop early.
    let mut rules = PhysicalOptimizer::new().rules;
    rules.insert(0, Arc::new(RowFilterUnderTopK));
    rules.insert(1, Arc::new(OrderUnderScalarSubquery));
    rules.push(Arc::new(RightSizeJoinBuild));
    rules.push(Arc::new(KeepSortedRoot));
    rules.push(Arc::new(StageLimitGathers));
    rules.push(Arc::new(ReadAheadOff));
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(budgeted_runtime(budget, spill)?)
        .with_default_features()
        // A scalar subquery in a `VALUES` cell would be evaluated while the
        // list is planned, before the subquery has run; this plans such a
        // list as one-row projections instead (see the rule's module).
        .with_optimizer_rule(Arc::new(ValuesSubqueryRewrite));
    *builder.physical_optimizers() = Some(PhysicalOptimizer::with_rules(rules));
    Ok(SessionContext::new_with_state(builder.build()))
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::{
            array::{AsArray, Int64Array, StringArray, StringViewArray},
            datatypes::{DataType, Field, Int64Type, Schema},
            record_batch::RecordBatch,
            util::display::array_value_to_string,
        },
        datasource::{MemTable, TableProvider},
        execution::memory_pool::MemoryConsumer,
        physical_plan::{ExecutionPlan, collect, displayable, metrics::MetricsSet},
    };
    use tokio::runtime::Runtime;

    use super::*;
    use crate::{memory::tests::Holder, supertable::query::join_build_size::RIGHT_SIZED_BATCHES};

    /// Rows per input batch in the tests below: DataFusion's default batch
    /// size.
    const BATCH_ROWS: usize = 8 * 1024;
    /// Partitions the partitioned hash join test runs on.
    const JOIN_PARTITIONS: usize = 4;
    /// DataFusion's hash join metric for the bytes its build side charged to
    /// the memory pool, summed over its partitions.
    const BUILD_MEM_USED: &str = "build_mem_used";

    #[test]
    fn measured_pool_never_refuses() {
        let budget = ConnectionMemoryBudget::measured();
        let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
            budget: Arc::clone(&budget),
        });
        let res = MemoryConsumer::new("t").register(&pool);
        res.try_grow(1 << 30).expect("measured never refuses");
        assert_eq!(pool.reserved(), 1 << 30);
        assert!(matches!(pool.memory_limit(), MemoryLimit::Infinite));
    }

    #[test]
    fn bounded_pool_refuses_past_the_gate() {
        // 1000 configured -> gate at 900.
        let budget = ConnectionMemoryBudget::with_limit(1000);
        let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
            budget: Arc::clone(&budget),
        });
        assert!(matches!(pool.memory_limit(), MemoryLimit::Finite(900)));

        let res = MemoryConsumer::new("t").register(&pool);
        res.try_grow(900).expect("exactly at the gate fits");
        res.try_grow(1)
            .expect_err("one byte over the gate is refused");

        // A refusal leaves the counter untouched, and shrink frees it.
        assert_eq!(pool.reserved(), 900);
        res.shrink(900);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn a_refused_operator_takes_back_what_read_ahead_holds() {
        // A hash join's build side can't spill, so it needs read-ahead's
        // memory back. 1000 configured -> gate 900, 600 held.
        let budget = ConnectionMemoryBudget::with_limit(1000);
        let held = Holder::holding(&budget, 600);
        let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
            budget: Arc::clone(&budget),
        });
        let build = MemoryConsumer::new("build").register(&pool);
        build
            .try_grow(800)
            .expect("fits once read-ahead gives back");
        assert_eq!(held.asked(), 1);
        assert_eq!(pool.reserved(), 800);
    }

    #[test]
    fn pools_over_one_budget_share_the_counter() {
        // Two pools, one budget: the ceiling binds the connection, not one ctx.
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate 900
        let pool_a: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
            budget: Arc::clone(&budget),
        });
        let pool_b: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
            budget: Arc::clone(&budget),
        });
        let a = MemoryConsumer::new("a").register(&pool_a);
        let b = MemoryConsumer::new("b").register(&pool_b);

        a.try_grow(600).expect("fits");
        b.try_grow(300).expect("600 + 300 = 900 fits");
        b.try_grow(1).expect_err("the two pools share one ceiling");
    }

    #[test]
    fn grow_charges_past_the_limit_for_unspillable_reservations() {
        // `grow` is infallible (DataFusion's must-succeed reservations): it
        // charges via grow_unchecked, so usage can pass the gate.
        let budget = ConnectionMemoryBudget::with_limit(1000); // gate 900
        let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool { budget });
        let res = MemoryConsumer::new("must-succeed").register(&pool);
        res.grow(5000); // far past the gate, infallible
        assert_eq!(pool.reserved(), 5000);
    }

    // Sum, over the whole plan tree, of the metric `read` picks from each node.
    fn total_metric(
        plan: &dyn ExecutionPlan,
        read: &dyn Fn(&MetricsSet) -> Option<usize>,
    ) -> usize {
        let here = plan.metrics().and_then(|m| read(&m)).unwrap_or(0);
        here + plan
            .children()
            .iter()
            .map(|c| total_metric(c.as_ref(), read))
            .sum::<usize>()
    }

    /// Whether the first column of `batches` is in ascending order, comparing
    /// each value's display string (so keys must sort as strings).
    fn is_sorted(batches: &[RecordBatch]) -> bool {
        let values: Vec<String> = batches
            .iter()
            .flat_map(|b| {
                (0..b.num_rows())
                    .map(move |row| array_value_to_string(b.column(0), row).expect("value"))
            })
            .collect();
        values.windows(2).all(|w| w[0] <= w[1])
    }

    /// Sort `batches` by their string column under a bounded budget; returns
    /// `(rows_out, in_order, spill_count)`. Reservation + single partition are
    /// pinned so the sizing is deterministic (see the caller).
    async fn run_sorted_query(
        schema: &Arc<Schema>,
        batches: &[RecordBatch],
        configured_bytes: u64,
        reservation_bytes: usize,
    ) -> (usize, bool, usize) {
        sort_through(schema, batches, configured_bytes, reservation_bytes, None)
            .await
            .expect("collect")
    }

    /// [`run_sorted_query`], spilling through `spill`, with the query's
    /// error handed back.
    async fn sort_through(
        schema: &Arc<Schema>,
        batches: &[RecordBatch],
        configured_bytes: u64,
        reservation_bytes: usize,
        spill: Option<&SqlSpill>,
    ) -> DfResult<(usize, bool, usize)> {
        let budget = ConnectionMemoryBudget::with_limit(configured_bytes);

        // Build through the production runtime + pool wiring (`budgeted_runtime`);
        // only the SessionConfig is tuned here (that's DataFusion's, not our
        // wiring) so the sort spills deterministically.
        let runtime = budgeted_runtime(&budget, spill).expect("runtime env");
        let mut cfg = SessionConfig::new();
        cfg.options_mut().execution.sort_spill_reservation_bytes = reservation_bytes;
        // One partition = one sorter. The default (num_cpus) gives N sorters,
        // each reserving its own merge budget, which crosses the gate for
        // unrelated reasons.
        cfg.options_mut().execution.target_partitions = 1;
        let ctx = SessionContext::new_with_config_rt(cfg, runtime);
        let table =
            MemTable::try_new(Arc::clone(schema), vec![batches.to_vec()]).expect("memtable");
        ctx.register_table("t", Arc::new(table)).expect("register");

        let df = ctx.sql("SELECT s FROM t ORDER BY s").await.expect("plan");
        let plan = df.create_physical_plan().await.expect("physical plan");
        let out = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        Ok((
            rows,
            is_sorted(&out),
            total_metric(plan.as_ref(), &MetricsSet::spill_count),
        ))
    }

    /// `rows` rows of one 64-byte string column `s`: an 8-digit key counting
    /// down, so `ORDER BY` reverses them all, and filler. In batches of
    /// [`BATCH_ROWS`], so a sort can spill between them.
    fn descending_key_batches(rows: usize) -> (Arc<Schema>, Vec<RecordBatch>) {
        let filler = "x".repeat(56);
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let batches = (0..rows)
            .step_by(BATCH_ROWS)
            .map(|start| {
                let vals: Vec<String> = (start..(start + BATCH_ROWS).min(rows))
                    .map(|i| format!("{:08}{filler}", rows - 1 - i))
                    .collect();
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vals))])
                    .expect("batch")
            })
            .collect();
        (schema, batches)
    }

    /// A connection with a disk cache folder spills its SQL into a folder
    /// under it, within a share of the disk, through one disk manager. A
    /// query that would spill past the limit fails alone, and the folder
    /// goes with the connection.
    #[test]
    fn sql_spills_under_the_cache_folder_within_a_limit() {
        const ROWS: usize = 512 * 1024;
        const RESERVATION: usize = 20 * 1024 * 1024;
        const TIGHT_BUDGET_BYTES: u64 = 32 * 1024 * 1024;
        const ROOMY_BUDGET_BYTES: u64 = 256 * 1024 * 1024;
        /// Less than the sort spills.
        const TINY_SPILL_LIMIT_BYTES: u64 = 1024 * 1024;
        let cache = tempfile::tempdir().expect("cache folder");
        let folder = cache.path().join(SQL_SPILL_SUBDIR);
        let (schema, batches) = descending_key_batches(ROWS);
        let runtime = Runtime::new().expect("tokio runtime");

        let spill = SqlSpill::under(cache.path());
        let disk_manager = spill.disk_manager().expect("spill folder");
        assert!(
            disk_manager
                .temp_dir_paths()
                .iter()
                .all(|path| path.starts_with(&folder))
        );
        let disk_share =
            total_space(cache.path()).expect("disk size") / SQL_SPILL_DISK_SHARE_DIVISOR;
        assert_eq!(
            disk_manager.max_temp_directory_size(),
            disk_share.min(DEFAULT_MAX_TEMP_DIRECTORY_SIZE)
        );
        assert!(Arc::ptr_eq(
            &disk_manager,
            &spill.disk_manager().expect("spill folder")
        ));
        let (rows, sorted, spills) = runtime
            .block_on(sort_through(
                &schema,
                &batches,
                TIGHT_BUDGET_BYTES,
                RESERVATION,
                Some(&spill),
            ))
            .expect("a sort that spills within the limit");
        assert_eq!((rows, sorted), (ROWS, true));
        assert!(spills > 0);
        drop(disk_manager);
        drop(spill);
        assert_eq!(
            std::fs::read_dir(&folder).expect("spill folder").count(),
            0,
            "the connection's folder goes with it"
        );

        let spill = SqlSpill::under_with_limit(cache.path(), TINY_SPILL_LIMIT_BYTES);
        let error = runtime
            .block_on(sort_through(
                &schema,
                &batches,
                TIGHT_BUDGET_BYTES,
                RESERVATION,
                Some(&spill),
            ))
            .expect_err("a sort that spills past the limit");
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        let (rows, sorted, spills) = runtime
            .block_on(sort_through(
                &schema,
                &batches,
                ROOMY_BUDGET_BYTES,
                RESERVATION,
                Some(&spill),
            ))
            .expect("the next query runs");
        assert_eq!((rows, sorted, spills), (ROWS, true, 0));
    }

    #[test]
    fn bounded_pool_spills_a_large_sort_and_returns_correct_results() {
        // Spill, don't refuse. 32 MiB of data through an 8.8 MiB buffer can only
        // finish by spilling, so correct+complete output proves it spilled, and
        // spill_count > 0 confirms it. The generous run below (no spill) shows
        // the spill was the budget's doing. Sizing:
        //
        //   gate               28.8 MiB   (90% of 32 MiB configured)
        //   merge reservation  20 MiB     (held back for the merge; RESERVATION)
        //   sort buffer         8.8 MiB   (gate - reservation)
        //   data               32 MiB     (524,288 rows x 64 B)
        const ROWS: usize = 512 * 1024; // 524,288 rows
        const RESERVATION: usize = 20 * 1024 * 1024; // big, to shrink the buffer

        // Column `s`: 8-digit key + 56 B filler = 64 B/row, keys counting DOWN so
        // ORDER BY must reverse all of them:
        //   "00524287xxx..." ... "00000000xxx..."  ->  "00000000..." ... "00524287..."
        // Filler just pads to 64 B (so ~512k rows = ~32 MiB). Many small batches
        // so the sort can spill between them, not one un-splittable input.
        let (schema, batches) = descending_key_batches(ROWS);

        let runtime = Runtime::new().expect("tokio runtime");
        runtime.block_on(async {
            // Tight: 32 MiB budget, data > buffer -> spills.
            let (rows, sorted, spills) =
                run_sorted_query(&schema, &batches, 32 * 1024 * 1024, RESERVATION).await;
            assert_eq!(rows, ROWS, "every row comes back despite spilling");
            assert!(sorted, "spilled sort still returns rows in order");
            assert!(spills > 0, "the sort must spill under the tight budget");

            // Generous: 256 MiB budget, gate >> data -> no spill.
            let (rows, sorted, spills) =
                run_sorted_query(&schema, &batches, 256 * 1024 * 1024, RESERVATION).await;
            assert_eq!(rows, ROWS);
            assert!(sorted);
            assert_eq!(spills, 0, "a generous budget sorts in memory, no spill");
        });
    }

    #[test]
    fn budget_counter_climbs_per_query_and_returns_to_baseline() {
        // A measured budget tracks usage without ever refusing. Run the same
        // GROUP BY three times on one shared budget:
        //  - each query reserves (peak climbs above 0), then frees it all (used -> 0);
        //  - the peak stays flat across runs, so nothing leaks between queries.
        // Single partition keeps the reservations reproducible run-to-run.
        const GROUPS: usize = 50 * 1024;

        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let vals: Vec<String> = (0..GROUPS).map(|i| format!("key-{i}")).collect();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vals))])
                .expect("batch");

        // One shared budget across all the queries below (as on a connection).
        let budget = ConnectionMemoryBudget::measured();

        let runtime_env = Runtime::new().expect("tokio runtime");
        runtime_env.block_on(async {
            let mut prev_peak = 0;
            for i in 0..3 {
                // Fresh ctx per query, all sharing the one budget.
                let rt = budgeted_runtime(&budget, None).expect("runtime env");

                let mut cfg = SessionConfig::new();
                cfg.options_mut().execution.target_partitions = 1;

                let ctx = SessionContext::new_with_config_rt(cfg, rt);
                let table = MemTable::try_new(Arc::clone(&schema), vec![vec![batch.clone()]])
                    .expect("memtable");

                ctx.register_table("t", Arc::new(table)).expect("register");

                let rows: usize = ctx
                    .sql("SELECT s, COUNT(*) FROM t GROUP BY s")
                    .await
                    .expect("plan")
                    .collect()
                    .await
                    .expect("collect")
                    .iter()
                    .map(|b| b.num_rows())
                    .sum();

                assert_eq!(rows, GROUPS);

                let peak = budget.peak();

                assert!(peak > 0, "query {i} reserved against the budget");
                assert_eq!(budget.used(), 0, "query {i} freed all its reservations");

                if i > 0 {
                    assert_eq!(
                        peak, prev_peak,
                        "identical queries hold a flat peak (no leak)"
                    );
                }

                prev_peak = peak;
            }
        });
    }

    /// `t`: a `Utf8View` group key `g` (zero-padded, so it sorts as a string)
    /// and an `Int64` value `v`, spread over `PARTITIONS` partitions.
    const PARTITIONS: usize = 8;
    const ROWS_PER_PARTITION: usize = 8192;
    /// Enough groups that the result spans several batches, so a split of the
    /// sorted result into partitions shows. With one batch, a split leaves it
    /// all in one partition and the result looks sorted by luck.
    const GROUPS: usize = 20_000;
    /// Digits of the zero-padded group key.
    const KEY_DIGITS: usize = 5;
    /// Rows the top-k shape keeps.
    const TOP_K: usize = 100;

    /// A session over `t`, pinned to `PARTITIONS` target partitions so the
    /// plan is parallel on any host.
    async fn scalar_subquery_session() -> SessionContext {
        let schema = Arc::new(Schema::new(vec![
            Field::new("g", DataType::Utf8View, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let partitions: Vec<Vec<RecordBatch>> = (0..PARTITIONS)
            .map(|p| {
                let groups = StringViewArray::from_iter_values(
                    (0..ROWS_PER_PARTITION)
                        .map(|i| format!("{:0KEY_DIGITS$}", (i * PARTITIONS + p) % GROUPS)),
                );
                let vals = Int64Array::from_iter_values(0..ROWS_PER_PARTITION as i64);
                vec![
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(groups), Arc::new(vals)],
                    )
                    .expect("batch"),
                ]
            })
            .collect();
        let ctx =
            budgeted_session_context(&ConnectionMemoryBudget::measured(), None).expect("session");
        ctx.sql(&format!(
            "SET datafusion.execution.target_partitions = {PARTITIONS}"
        ))
        .await
        .expect("set partitions");
        let table = MemTable::try_new(schema, partitions).expect("memtable");
        ctx.register_table("t", Arc::new(table)).expect("register");
        ctx
    }

    /// Grouped queries over a string key with an uncorrelated scalar subquery,
    /// in `WHERE`, in the `SELECT` list, in `ORDER BY` and under a top-k,
    /// return their groups in `ORDER BY` order from one partition. Both rules
    /// that walk through the `ScalarSubqueryExec` root are needed: without
    /// `OrderUnderScalarSubquery` DataFusion 54 plans the sort away, and
    /// without `KeepSortedRoot`'s step into the root's first child the
    /// round-robin repartition above the output cast splits the sorted result.
    /// A top-k's result fits one batch and comes back sorted even when split,
    /// so the partition count is what catches it there. This is the check to
    /// run before deleting `OrderUnderScalarSubquery` on the move to
    /// DataFusion 55.
    #[test]
    fn order_by_survives_an_uncorrelated_scalar_subquery() {
        let queries = [
            (
                "SELECT g, count(*) AS n FROM t \
                 WHERE v > (SELECT avg(v) FROM t) GROUP BY g ORDER BY g"
                    .to_string(),
                GROUPS,
            ),
            (
                "SELECT g, (SELECT max(v) FROM t) AS m FROM t GROUP BY g ORDER BY g".to_string(),
                GROUPS,
            ),
            (
                "SELECT g FROM t GROUP BY g \
                 ORDER BY concat(g, (SELECT CAST(max(v) AS VARCHAR) FROM t))"
                    .to_string(),
                GROUPS,
            ),
            (
                format!(
                    "SELECT g, count(*) AS n FROM t \
                     WHERE v > (SELECT avg(v) FROM t) GROUP BY g ORDER BY g LIMIT {TOP_K}"
                ),
                TOP_K,
            ),
        ];
        let runtime = Runtime::new().expect("tokio runtime");
        runtime.block_on(async {
            let ctx = scalar_subquery_session().await;
            for (sql, rows) in queries {
                let plan = ctx
                    .sql(&sql)
                    .await
                    .expect("plan")
                    .create_physical_plan()
                    .await
                    .expect("physical plan");
                assert_eq!(
                    plan.properties().output_partitioning().partition_count(),
                    1,
                    "the sorted result is split into partitions: {sql}"
                );
                let out = collect(plan, ctx.task_ctx()).await.expect("collect");
                assert_eq!(
                    out.iter().map(|b| b.num_rows()).sum::<usize>(),
                    rows,
                    "{sql}"
                );
                assert!(is_sorted(&out), "groups came back unsorted: {sql}");
            }
        });
    }

    /// A scalar subquery that returns more than one row is an error, not a
    /// join that repeats the outer rows: DataFusion's `ScalarSubqueryExec`
    /// stays on, and its older plan, a left join, would give a wrong count.
    #[test]
    fn a_scalar_subquery_returning_many_rows_is_an_error() {
        let runtime = Runtime::new().expect("tokio runtime");
        runtime.block_on(async {
            let ctx = scalar_subquery_session().await;
            let result = ctx
                .sql("SELECT count(*) FROM t WHERE v > (SELECT v FROM t)")
                .await
                .expect("plan")
                .collect()
                .await;
            let err = result.expect_err("a many-row scalar subquery fails");
            assert!(err.to_string().contains("more than one row"), "{err}");
        });
    }

    /// Runs `sql` over tables `a` (`small`) and `b` (`large`) with its hash
    /// join in `mode`: one partition for `CollectLeft`, or `JOIN_PARTITIONS`
    /// partitions with every build too big to collect for `Partitioned`.
    /// Returns the result, the plan's total `RIGHT_SIZED_BATCHES`, and the
    /// bytes its join builds charged.
    async fn run_join(
        ctx: SessionContext,
        tables: &[(&str, Arc<MemTable>)],
        mode: &str,
        sql: &str,
    ) -> DfResult<(Vec<RecordBatch>, usize, usize)> {
        let partitions = if mode == "Partitioned" {
            JOIN_PARTITIONS
        } else {
            1
        };
        ctx.sql(&format!(
            "SET datafusion.execution.target_partitions = {partitions}"
        ))
        .await?;
        if mode == "Partitioned" {
            for threshold in [
                "hash_join_single_partition_threshold",
                "hash_join_single_partition_threshold_rows",
            ] {
                ctx.sql(&format!("SET datafusion.optimizer.{threshold} = 0"))
                    .await?;
            }
        }
        for (name, table) in tables {
            ctx.register_table(*name, Arc::clone(table) as Arc<dyn TableProvider>)?;
        }
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let shape = displayable(plan.as_ref()).indent(true).to_string();
        assert!(shape.contains(&format!("mode={mode}")), "{shape}");
        let out = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        let sum = |name: &str| {
            total_metric(plan.as_ref(), &|m| {
                m.sum_by_name(name).map(|v| v.as_usize())
            })
        };
        Ok((out, sum(RIGHT_SIZED_BATCHES), sum(BUILD_MEM_USED)))
    }

    /// An aggregate's result feeding a hash join's build side fits a 64 MiB
    /// budget only with `RightSizeJoinBuild`, with the join collecting one
    /// build and with it partitioned: without the rule every 8,192-row slice
    /// of the aggregate's result is charged for the whole buffers behind it.
    /// The outer query reads both `g.k` and `g.s`, so the build holds both
    /// columns:
    ///
    ///   aggregate result  300,000 groups, key + sum buffers of ~4 MiB capacity each
    ///   build, charged    37 slices x ~8 MiB ≈ 300 MiB without the rule
    ///   build, real       ~5 MB
    ///   gate              57.6 MiB (90% of 64 MiB configured)
    ///
    /// The second session is plain DataFusion, with none of our rules or
    /// settings, on a budget that only measures: its build charges more than
    /// the gate, so the budget is too small for the uncopied build. It measures
    /// rather than refuses because a refusal makes the aggregates give memory
    /// back, and then whether the partitioned build crosses the gate, by about
    /// a megabyte, turns on what else is held at that instant. On DataFusion
    /// 55, which counts a buffer shared by several batches once, the charge
    /// falls under the gate and this half fails; revisit the rule then (see
    /// its module docs).
    #[test]
    fn an_aggregate_feeding_a_join_build_fits_once_its_slices_are_copied() {
        const GROUPS: usize = 300_000;
        const PROBE_REPEATS: usize = 4;
        const CONFIGURED: u64 = 64 * 1024 * 1024;
        const SQL: &str = "SELECT COUNT(*), SUM(g.s) \
                           FROM (SELECT k, SUM(v) AS s FROM a GROUP BY k) g \
                           JOIN b ON g.k = b.k";

        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        // Row `i` has key `i % GROUPS` and value `i`.
        let table_of = |rows: usize| -> Arc<MemTable> {
            let batches = (0..rows)
                .step_by(BATCH_ROWS)
                .map(|start| {
                    let rows = start..(start + BATCH_ROWS).min(rows);
                    let keys =
                        Int64Array::from_iter_values(rows.clone().map(|i| (i % GROUPS) as i64));
                    let vals = Int64Array::from_iter_values(rows.map(|i| i as i64));
                    RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(keys), Arc::new(vals)])
                        .expect("batch")
                })
                .collect();
            Arc::new(MemTable::try_new(Arc::clone(&schema), vec![batches]).expect("memtable"))
        };
        let tables = [
            ("a", table_of(GROUPS)),
            ("b", table_of(GROUPS * PROBE_REPEATS)),
        ];
        // In `a` each key is one row whose value is the key, so each group's
        // sum is its key, and every group meets `PROBE_REPEATS` rows of `b`.
        let expected_count = (GROUPS * PROBE_REPEATS) as i64;
        let expected_sum = (PROBE_REPEATS * GROUPS * (GROUPS - 1) / 2) as i64;

        let runtime = Runtime::new().expect("tokio runtime");
        runtime.block_on(async {
            for mode in ["CollectLeft", "Partitioned"] {
                let budget = ConnectionMemoryBudget::with_limit(CONFIGURED);
                let with_rule = budgeted_session_context(&budget, None).expect("session");
                let (out, right_sized, _) = run_join(with_rule, &tables, mode, SQL)
                    .await
                    .unwrap_or_else(|e| panic!("{mode}: fits with the rule: {e}"));
                let column = |i: usize| out[0].column(i).as_primitive::<Int64Type>().value(0);
                assert_eq!(column(0), expected_count, "{mode}");
                assert_eq!(column(1), expected_sum, "{mode}");
                assert!(right_sized > 0, "{mode}: the build's slices were copied");

                let gate = budget.limit().expect("a bounded budget");
                let without_rule = SessionContext::new_with_config_rt(
                    SessionConfig::new(),
                    budgeted_runtime(&ConnectionMemoryBudget::measured(), None).expect("runtime"),
                );
                let (_, _, charged) = run_join(without_rule, &tables, mode, SQL)
                    .await
                    .unwrap_or_else(|e| panic!("{mode}: a measured budget refuses nothing: {e}"));
                assert!(
                    charged > gate,
                    "{mode}: without the rule the build charges {charged} B, over the {gate} B gate"
                );
            }
        });
    }
}
