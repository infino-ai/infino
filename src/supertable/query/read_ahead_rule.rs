// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ReadAheadOff`]: a plan rule that turns read-ahead off for scans that
//! may stop early, since what read-ahead downloaded past the stop is wasted.
//!
//! It turns read-ahead off under:
//! - a `LIMIT` that can stop the scan in any row group. With a filter, the
//!   `LIMIT` never reaches the scan; it stays in a node above, so this rule
//!   looks for it there:
//!
//!   ```text
//!     GlobalLimitExec fetch=10        ◀── the limit
//!       FilterExec price > 5          ◀── rows pass straight through
//!         MeteredExec
//!           DataSourceExec            ◀── never told; this rule turns read-ahead off
//!   ```
//!
//! - a `DISTINCT ... LIMIT`, which stops once it has enough groups;
//! - a top-k (`ORDER BY ... LIMIT`), whose filter tightens while it runs and
//!   makes the scan skip row groups.
//!
//! A full sort (`ORDER BY` with no `LIMIT`) keeps read-ahead, since it reads
//! every row group. If DataFusion reorders the scan's row groups for it, the
//! reader sees the reads go out of order and turns itself off, wasting at
//! most one batch.
//!
//! The rule looks down from a limit only through nodes that pass rows on as
//! they come (filters, repartitions, our meter, and the nodes DataFusion
//! pushes a limit through). A join's probe side or a streaming window can
//! still stop early; those keep read-ahead and may waste up to two batches.
//! The rule runs after DataFusion places its limits and sorts.

use std::sync::Arc;

use datafusion::{
    config::ConfigOptions,
    error::Result as DfResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, repartition::RepartitionExec,
        sorts::sort::SortExec,
    },
};

use crate::supertable::query::exec::metered_exec::MeteredExec;

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "ReadAheadOff";

/// Turns read-ahead off for scans under a `LIMIT` or top-k that can stop
/// them early; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct ReadAheadOff;

impl PhysicalOptimizerRule for ReadAheadOff {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        turn_off_under_limit(&plan, false);
        Ok(plan)
    }

    fn name(&self) -> &str {
        RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Turn off read-ahead for the scans under `plan` that may stop early.
/// `limited` says a limit or top-k above `plan` reaches them.
fn turn_off_under_limit(plan: &Arc<dyn ExecutionPlan>, limited: bool) {
    if let Some(meter) = plan.downcast_ref::<MeteredExec>()
        && let Some(read_ahead) = meter.read_ahead()
        && (limited || meter.input().fetch().is_some() || is_top_k(meter.input()))
    {
        read_ahead.disable();
    }
    let passes_rows_on = plan.supports_limit_pushdown()
        || plan.is::<FilterExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<MeteredExec>();
    // A limit node's fetch is its `LIMIT`; an `OFFSET` alone has none.
    let below = stops_early(plan) || (passes_rows_on && (limited || plan.fetch().is_some()));
    for child in plan.children() {
        turn_off_under_limit(child, below);
    }
}

/// Whether `plan` can stop the scans below it early: a `DISTINCT ... LIMIT`
/// or a top-k. An ordered top-k aggregate reads all its input, so it can't.
fn stops_early(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let distinct_limit = plan
        .downcast_ref::<AggregateExec>()
        .and_then(AggregateExec::limit_options)
        .is_some_and(|limit| limit.descending().is_none());
    distinct_limit || is_top_k(plan)
}

/// A sort that keeps only its first rows (`ORDER BY ... LIMIT`).
fn is_top_k(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<SortExec>()
        .is_some_and(|sort| sort.fetch().is_some())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::{
        config::ConfigOptions,
        logical_expr::Operator,
        physical_expr::{LexOrdering, PhysicalSortExpr},
        physical_optimizer::PhysicalOptimizerRule,
        physical_plan::{
            ExecutionPlan, Partitioning,
            aggregates::{AggregateExec, AggregateMode, LimitOptions, PhysicalGroupBy},
            empty::EmptyExec,
            expressions::{BinaryExpr, col, lit},
            filter::FilterExec,
            limit::GlobalLimitExec,
            repartition::RepartitionExec,
            sorts::sort::SortExec,
        },
    };

    use crate::{
        memory::{ConnectionMemoryBudget, budgeted_session_context},
        supertable::query::{
            df_object_store::SuperfileObjectStore,
            exec::metered_exec::MeteredExec,
            read_ahead::ScanReadAhead,
            read_ahead_rule::{RULE_NAME, ReadAheadOff},
        },
    };

    /// The `LIMIT` and `OFFSET` the tests' plans use.
    const LIMIT: usize = 10;
    /// Partitions a test's repartition spreads rows over.
    const PARTITIONS: usize = 4;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    /// Whether read-ahead is still on for the scan under the plan `wrap`
    /// builds, after the rule runs.
    fn on_after_rule(wrap: impl FnOnce(Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan>) -> bool {
        on_after_rule_over(Arc::new(EmptyExec::new(schema())), wrap)
    }

    /// [`on_after_rule`] with `input` as the node the scan's meter wraps.
    fn on_after_rule_over(
        input: Arc<dyn ExecutionPlan>,
        wrap: impl FnOnce(Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan>,
    ) -> bool {
        let scan = Arc::new(ScanReadAhead::new(
            Arc::new(SuperfileObjectStore::new()),
            ConnectionMemoryBudget::measured(),
            schema(),
        ));
        let meter: Arc<dyn ExecutionPlan> =
            Arc::new(MeteredExec::new(input, None).with_read_ahead(Arc::clone(&scan)));
        ReadAheadOff
            .optimize(wrap(meter), &ConfigOptions::default())
            .expect("rule");
        scan.is_enabled()
    }

    /// `ORDER BY a`.
    fn a_order() -> LexOrdering {
        let a = col("a", &schema()).expect("a");
        LexOrdering::new(vec![PhysicalSortExpr::new_default(a)]).expect("order")
    }

    /// `SELECT a ... GROUP BY a`, with `limit` as a `DISTINCT ... LIMIT` has it.
    fn group_by_a(input: Arc<dyn ExecutionPlan>, limit: Option<LimitOptions>) -> AggregateExec {
        let a = col("a", &schema()).expect("a");
        AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(a, "a".to_string())]),
            vec![],
            vec![],
            input,
            schema(),
        )
        .expect("aggregate")
        .with_limit_options(limit)
    }

    fn a_gt_5(input: Arc<dyn ExecutionPlan>) -> FilterExec {
        let a = col("a", &schema()).expect("a");
        FilterExec::try_new(Arc::new(BinaryExpr::new(a, Operator::Gt, lit(5i64))), input)
            .expect("filter")
    }

    #[test]
    fn a_limit_above_a_filter_turns_read_ahead_off() {
        assert!(on_after_rule(|scan| Arc::new(a_gt_5(scan))));
        assert!(!on_after_rule(|scan| Arc::new(GlobalLimitExec::new(
            Arc::new(a_gt_5(scan)),
            0,
            Some(LIMIT)
        ))));
        assert!(!on_after_rule(|scan| a_gt_5(scan)
            .with_fetch(Some(LIMIT))
            .expect("fetch")));
    }

    #[test]
    fn a_limit_pushed_into_the_scan_turns_read_ahead_off() {
        let scan_with_fetch = a_gt_5(Arc::new(EmptyExec::new(schema())))
            .with_fetch(Some(LIMIT))
            .expect("fetch");
        assert!(!on_after_rule_over(scan_with_fetch, |meter| meter));
    }

    #[test]
    fn an_offset_without_a_limit_keeps_read_ahead_on() {
        assert!(on_after_rule(|scan| Arc::new(GlobalLimitExec::new(
            Arc::new(a_gt_5(scan)),
            LIMIT,
            None
        ))));
    }

    #[test]
    fn an_order_by_limit_turns_read_ahead_off() {
        assert!(!on_after_rule(|scan| Arc::new(
            SortExec::new(a_order(), Arc::new(a_gt_5(scan))).with_fetch(Some(LIMIT))
        )));
    }

    #[test]
    fn a_top_k_on_the_scan_turns_read_ahead_off() {
        // Above it, through rows passed on as they come.
        assert!(!on_after_rule(|scan| Arc::new(GlobalLimitExec::new(
            Arc::new(SortExec::new(a_order(), scan).with_fetch(Some(LIMIT))),
            0,
            Some(LIMIT),
        ))));
        // Or right under the meter.
        let top_k = Arc::new(
            SortExec::new(a_order(), Arc::new(EmptyExec::new(schema()))).with_fetch(Some(LIMIT)),
        );
        assert!(!on_after_rule_over(top_k, |meter| meter));
    }

    #[test]
    fn a_full_sort_keeps_read_ahead_on() {
        // The sort reads every row group. If the scan's order changes, the
        // reader turns read-ahead off itself.
        assert!(on_after_rule(|scan| Arc::new(SortExec::new(
            a_order(),
            Arc::new(a_gt_5(scan))
        ))));
        let sorted = Arc::new(SortExec::new(a_order(), Arc::new(EmptyExec::new(schema()))));
        assert!(on_after_rule_over(sorted, |meter| meter));
    }

    #[test]
    fn a_sort_above_an_aggregate_keeps_read_ahead_on() {
        // The aggregate reads the whole scan before the sort sees a row.
        assert!(on_after_rule(|scan| Arc::new(SortExec::new(
            a_order(),
            Arc::new(group_by_a(scan, None))
        ))));
    }

    #[test]
    fn a_distinct_limit_turns_read_ahead_off() {
        assert!(!on_after_rule(|scan| Arc::new(group_by_a(
            scan,
            Some(LimitOptions::new(LIMIT))
        ))));
    }

    #[test]
    fn a_limit_reaches_the_scan_through_a_repartition() {
        assert!(!on_after_rule(|scan| {
            let spread = RepartitionExec::try_new(scan, Partitioning::RoundRobinBatch(PARTITIONS))
                .expect("repartition");
            Arc::new(GlobalLimitExec::new(Arc::new(spread), 0, Some(LIMIT)))
        }));
    }

    #[test]
    fn sessions_run_the_rule() {
        let ctx = budgeted_session_context(&ConnectionMemoryBudget::measured()).expect("session");
        assert!(
            ctx.state()
                .physical_optimizers()
                .iter()
                .any(|rule| rule.name() == RULE_NAME)
        );
    }
}
