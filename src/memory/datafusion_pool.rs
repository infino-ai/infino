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
//!     - the operator can't spill at all:
//!        - non-spillable (hash-join build side, nested-loop join, window aggregate), or
//!        - a streaming operator (scan / filter / projection) that buffers nothing, so a single
//!          allocation already exceeds the budget and there is nothing to write out; or
//!     - it is spillable but can't reserve even the minimum it needs to run the
//!       spill / merge (e.g. the sort's merge reservation).
//!
//! Spilling needs a disk manager; we use DataFusion's default (OS temp dir).
//!

use std::sync::Arc;

use datafusion::{
    error::{DataFusionError, Result as DfResult},
    execution::{
        memory_pool::{MemoryLimit, MemoryPool, MemoryReservation},
        runtime_env::{RuntimeEnv, RuntimeEnvBuilder},
    },
    prelude::{SessionConfig, SessionContext},
};

use crate::memory::ConnectionMemoryBudget;

/// A DataFusion memory pool over a [`ConnectionMemoryBudget`]: measured never
/// refuses, bounded refuses at the 90% gate (DataFusion then spills, or errors
/// if it can't).
#[derive(Debug)]
struct ConnectionBudgetPool {
    budget: Arc<ConnectionMemoryBudget>,
}

impl MemoryPool for ConnectionBudgetPool {
    fn grow(&self, _reservation: &MemoryReservation, additional: usize) {
        self.budget.grow_unchecked(additional);
    }

    fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
        self.budget.release(shrink);
    }

    fn try_grow(&self, _reservation: &MemoryReservation, additional: usize) -> DfResult<()> {
        self.budget.try_grow(additional).map_err(|_| {
            DataFusionError::ResourcesExhausted(format!(
                "connection memory budget exhausted: {additional} more bytes would cross the limit"
            ))
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
    Ok(SessionContext::new_with_config_rt(
        SessionConfig::new(),
        budgeted_runtime(budget)?,
    ))
}

#[cfg(test)]
mod tests {
    use datafusion::{
        arrow::{
            array::{Array, StringArray},
            datatypes::{DataType, Field, Schema},
            record_batch::RecordBatch,
        },
        datasource::MemTable,
        execution::memory_pool::MemoryConsumer,
        physical_plan::{ExecutionPlan, collect},
    };
    use tokio::runtime::Runtime;

    use super::*;

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

    fn is_sorted(batches: &[RecordBatch]) -> bool {
        let mut prev: Option<String> = None;
        for b in batches {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string column");
            for i in 0..col.len() {
                let v = col.value(i).to_string();
                if prev.as_ref().is_some_and(|p| &v < p) {
                    return false;
                }
                prev = Some(v);
            }
        }
        true
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
}
