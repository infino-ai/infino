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
//! Spilling needs a disk manager; we use DataFusion's default (OS temp dir).
//!

use std::{fmt, sync::Arc};

use datafusion::{
    error::{DataFusionError, Result as DfResult},
    execution::{
        memory_pool::{MemoryLimit, MemoryPool, MemoryReservation},
        runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
        session_state::SessionStateBuilder,
    },
    physical_optimizer::optimizer::PhysicalOptimizer,
    prelude::{SessionConfig, SessionContext},
};

use crate::{
    memory::ConnectionMemoryBudget,
    supertable::query::{
        limit_stage::StageLimitGathers,
        read_ahead_rule::ReadAheadOff,
        sorted_root::{KeepSortedRoot, OrderUnderScalarSubquery},
        topk_row_filter::RowFilterUnderTopK,
        values_subquery::ValuesSubqueryRewrite,
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

/// A `RuntimeEnv` whose memory pool is `budget`. The default disk manager (OS
/// temp) gives spillable operators somewhere to spill.
fn budgeted_runtime(budget: &Arc<ConnectionMemoryBudget>) -> DfResult<Arc<RuntimeEnv>> {
    let pool: Arc<dyn MemoryPool> = Arc::new(ConnectionBudgetPool {
        budget: Arc::clone(budget),
    });

    RuntimeEnvBuilder::new().with_memory_pool(pool).build_arc()
}

/// A `SessionContext` whose SQL allocations are gated by `budget`.
pub(crate) fn budgeted_session_context(
    budget: &Arc<ConnectionMemoryBudget>,
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

    // DataFusion's rules, with ours at both ends:
    //  - `RowFilterUnderTopK` first, so DataFusion's filter pushdown sees the
    //    marked scans and drops their `FilterExec`.
    //  - `OrderUnderScalarSubquery` before DataFusion's `OutputRequirements`,
    //    which on DataFusion 54 loses the `ORDER BY` of a query under a
    //    `ScalarSubqueryExec` root.
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
    rules.push(Arc::new(KeepSortedRoot));
    rules.push(Arc::new(StageLimitGathers));
    rules.push(Arc::new(ReadAheadOff));
    let mut builder = SessionStateBuilder::new()
        .with_config(config)
        .with_runtime_env(budgeted_runtime(budget)?)
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
            array::{Int64Array, StringArray, StringViewArray},
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
            util::display::array_value_to_string,
        },
        datasource::MemTable,
        execution::memory_pool::MemoryConsumer,
        physical_plan::{ExecutionPlan, collect},
    };
    use tokio::runtime::Runtime;

    use super::*;
    use crate::memory::tests::Holder;

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

    // Total `spill_count` across the plan tree; the sort reports it on its node.
    fn total_spill_count(plan: &dyn ExecutionPlan) -> usize {
        let here = plan.metrics().and_then(|m| m.spill_count()).unwrap_or(0);
        here + plan
            .children()
            .iter()
            .map(|c| total_spill_count(c.as_ref()))
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
        let budget = ConnectionMemoryBudget::with_limit(configured_bytes);

        // Build through the production runtime + pool wiring (`budgeted_runtime`);
        // only the SessionConfig is tuned here (that's DataFusion's, not our
        // wiring) so the sort spills deterministically.
        let runtime = budgeted_runtime(&budget).expect("runtime env");
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
        let out = collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .expect("collect");
        let rows: usize = out.iter().map(|b| b.num_rows()).sum();
        (rows, is_sorted(&out), total_spill_count(plan.as_ref()))
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
        const CHUNK: usize = 8 * 1024; // 8,192/batch -> 512 KiB, 64 batches
        const RESERVATION: usize = 20 * 1024 * 1024; // big, to shrink the buffer

        // Column `s`: 8-digit key + 56 B filler = 64 B/row, keys counting DOWN so
        // ORDER BY must reverse all of them:
        //   "00524287xxx..." ... "00000000xxx..."  ->  "00000000..." ... "00524287..."
        // Filler just pads to 64 B (so ~512k rows = ~32 MiB). Many small batches
        // so the sort can spill between them, not one un-splittable input.
        let filler = "x".repeat(56);
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let mut batches = Vec::new();
        for start in (0..ROWS).step_by(CHUNK) {
            let vals: Vec<String> = (start..(start + CHUNK).min(ROWS))
                .map(|i| format!("{:08}{filler}", ROWS - 1 - i))
                .collect();
            batches.push(
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(StringArray::from(vals))])
                    .expect("batch"),
            );
        }

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
                let rt = budgeted_runtime(&budget).expect("runtime env");

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
        let ctx = budgeted_session_context(&ConnectionMemoryBudget::measured()).expect("session");
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
}
