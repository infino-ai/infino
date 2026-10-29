// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`StageLimitGathers`]: under a plain `LIMIT`, start a filtered scan's
//! partition 0 alone, and the rest only once its first batch falls short.
//!
//! A filtered scan takes no `LIMIT` fetch: its access plans carry tombstone
//! selections, and the Parquet opener's limit pruning could replace them (see
//! `MeteredExec::without_limit_pushdown`). So under `WHERE ... LIMIT n` with
//! no `ORDER BY`, the scan runs as `EnforceDistribution` split it, and the
//! `CoalescePartitionsExec` above starts every partition at once. A large
//! superfile split into 16 byte ranges then decodes a batch of every projected
//! column in each of them, although the first rows to pass end the query:
//!
//! ```text
//!   CoalescePartitionsExec fetch=10          StagedCoalesceExec fetch=10
//!     FilterExec price > 500 fetch=10          FilterExec price > 500 fetch=10
//!       MeteredExec                              ScanGateExec
//!         DataSourceExec 16 byte ranges            MeteredExec
//!                                                    DataSourceExec 16 byte ranges
//!   16 batches decoded, 10 rows kept         range 0 first: 1 batch, 10 rows
//! ```
//!
//! The rule swaps that gather for a [`StagedCoalesceExec`] and puts a
//! [`ScanGateExec`] on the scan, which hold partitions 1.. back until
//! partition 0's scan has been asked for a second batch. A filter most rows
//! pass is answered by partition 0's first batch; a filter that keeps almost
//! nothing waits for that one batch, then reads every partition side by side
//! as before, so nothing is ever read in series past one batch. A round-robin
//! exchange between the gather and the scan goes too: it would start every
//! partition at once regardless.
//!
//! The gate keeps the other partitions shut when partition 0's scan is
//! dropped before that second ask, which is right only if what dropped it
//! already has the gather's rows. So every node between them that keeps a
//! limit of its own must keep at least the gather's; a plan with a smaller
//! one is left as it is.
//!
//! The `WHERE` stays in its `FilterExec`. A Parquet row filter would decode
//! only the `WHERE`'s columns for rows that fail, but a filter keeping rows at
//! random leaves a selection of thousands of fragments per row group, which
//! measured slower than decoding the batch whole.

use std::sync::Arc;

use datafusion::{
    common::tree_node::{Transformed, TreeNode},
    config::ConfigOptions,
    error::Result as DfResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, Partitioning,
        coalesce_partitions::CoalescePartitionsExec, filter::FilterExec, limit::LocalLimitExec,
        projection::ProjectionExec, repartition::RepartitionExec,
    },
};

use crate::supertable::query::exec::{
    metered_exec::MeteredExec,
    staged_coalesce_exec::{ScanGateExec, StagedCoalesceExec},
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "StageLimitGathers";

/// Largest `LIMIT` (with its `OFFSET`) whose gather runs in stages. A limit
/// this small is met by the first batch or so of rows that pass, so the
/// batches every other partition decodes at the same time are mostly thrown
/// away; a larger one needs rows from many partitions, where starting them
/// together pays. Set a little over one decode batch (DataFusion's 8,192
/// rows), at the row filter's limit, not measured on its own.
const MAX_STAGED_FETCH: usize = 10_000;

/// Gathers a filtered scan under a small plain `LIMIT` in stages; see the
/// module docs.
#[derive(Debug, Default)]
pub(crate) struct StageLimitGathers;

impl PhysicalOptimizerRule for StageLimitGathers {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|node| {
            let Some(fetch) = node
                .downcast_ref::<CoalescePartitionsExec>()
                .and_then(|coalesce| coalesce.fetch())
                .filter(|&fetch| fetch <= MAX_STAGED_FETCH)
            else {
                return Ok(Transformed::no(node));
            };
            let [input] = node.children()[..] else {
                return Ok(Transformed::no(node));
            };
            let Some(input) = gated_scan(input, fetch)? else {
                return Ok(Transformed::no(node));
            };
            if input.output_partitioning().partition_count() <= 1 {
                return Ok(Transformed::no(node));
            }
            let staged: Arc<dyn ExecutionPlan> = Arc::new(StagedCoalesceExec::new(input, fetch));
            Ok(Transformed::yes(staged))
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

/// `plan` with its round-robin exchanges dropped and a [`ScanGateExec`] on
/// its scan, when it is a filtered table scan under filters, projections and
/// per-partition limits that keep at least `fetch` rows, or `None`. The
/// scan's meter taking no fetch is what marks it filtered: a scan with no
/// filter takes the `LIMIT` itself and stops on its own. A `WHERE` an index
/// answers leaves no `FilterExec`, so the meter's refused fetch lands in a
/// `LocalLimitExec` right above it.
fn gated_scan(
    plan: &Arc<dyn ExecutionPlan>,
    fetch: usize,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(input) = round_robin_input(plan) {
        return gated_scan(input, fetch);
    }
    if let Some(meter) = plan.downcast_ref::<MeteredExec>() {
        if meter.supports_limit_pushdown() {
            return Ok(None);
        }
        // `EnforceDistribution` puts the exchange below the meter when the
        // scan's files are too small to split.
        let meter = match round_robin_input(meter.input()) {
            Some(scan) => Arc::clone(plan).with_new_children(vec![Arc::clone(scan)])?,
            None => Arc::clone(plan),
        };
        return Ok(Some(Arc::new(ScanGateExec::new(meter))));
    }
    let passes_rows_on = plan.downcast_ref::<FilterExec>().is_some()
        || plan.downcast_ref::<ProjectionExec>().is_some()
        || plan.downcast_ref::<LocalLimitExec>().is_some();
    if !passes_rows_on || plan.fetch().is_some_and(|kept| kept < fetch) {
        return Ok(None);
    }
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    gated_scan(child, fetch)?
        .map(|child| Arc::clone(plan).with_new_children(vec![child]))
        .transpose()
}

/// The input of `node` when it is a round-robin exchange, else `None`.
fn round_robin_input(node: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    node.downcast_ref::<RepartitionExec>()
        .filter(|exchange| matches!(exchange.partitioning(), Partitioning::RoundRobinBatch(_)))
        .map(RepartitionExec::input)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use datafusion::{
        common::ScalarValue,
        config::ConfigOptions,
        logical_expr::Operator,
        physical_optimizer::PhysicalOptimizerRule,
        physical_plan::{
            ExecutionPlan, Partitioning,
            coalesce_partitions::CoalescePartitionsExec,
            empty::EmptyExec,
            expressions::{BinaryExpr, col, lit},
            filter::FilterExec,
            limit::LocalLimitExec,
            repartition::RepartitionExec,
        },
    };

    use super::{MAX_STAGED_FETCH, StageLimitGathers};
    use crate::supertable::query::exec::metered_exec::MeteredExec;

    /// The `LIMIT` the tests' plans use.
    const LIMIT: usize = 10;
    /// Partitions a test's scan reads in.
    const PARTITIONS: usize = 4;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]))
    }

    /// A scan of `partitions` partitions under its meter; `filtered` makes it
    /// a scan that takes no `LIMIT` fetch.
    fn scan(partitions: usize, filtered: bool) -> Arc<dyn ExecutionPlan> {
        let input = Arc::new(EmptyExec::new(schema()).with_partitions(partitions));
        if filtered {
            Arc::new(MeteredExec::without_limit_pushdown(input, None))
        } else {
            Arc::new(MeteredExec::new(input, None))
        }
    }

    /// `WHERE a > 0` over `input`.
    fn filter(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let a = col("a", &schema()).expect("a");
        let predicate = Arc::new(BinaryExpr::new(
            a,
            Operator::Gt,
            lit(ScalarValue::Int64(Some(0))),
        ));
        Arc::new(FilterExec::try_new(predicate, input).expect("filter"))
    }

    /// `input` spread round-robin over `PARTITIONS` partitions.
    fn round_robin(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            RepartitionExec::try_new(input, Partitioning::RoundRobinBatch(PARTITIONS))
                .expect("repartition"),
        )
    }

    /// `input` gathered under `fetch`, after the rule runs.
    fn staged(input: Arc<dyn ExecutionPlan>, fetch: Option<usize>) -> Arc<dyn ExecutionPlan> {
        let gather = Arc::new(CoalescePartitionsExec::new(input).with_fetch(fetch));
        StageLimitGathers
            .optimize(gather, &ConfigOptions::default())
            .expect("rule")
    }

    /// Every node's name in `plan`, top down.
    fn names(plan: &Arc<dyn ExecutionPlan>) -> Vec<String> {
        let mut out = vec![plan.name().to_owned()];
        for child in plan.children() {
            out.extend(names(child));
        }
        out
    }

    #[test]
    fn stages_a_filtered_scan_under_a_small_limit() {
        let plan = staged(filter(scan(PARTITIONS, true)), Some(LIMIT));
        assert_eq!(
            names(&plan),
            [
                "StagedCoalesceExec",
                "FilterExec",
                "ScanGateExec",
                "MeteredExec",
                "EmptyExec"
            ]
        );
        assert_eq!(plan.fetch(), Some(LIMIT));
    }

    #[test]
    fn drops_the_round_robin_exchange_above_or_below_the_meter() {
        let above = staged(round_robin(filter(scan(PARTITIONS, true))), Some(LIMIT));
        let below = staged(
            filter(Arc::new(MeteredExec::without_limit_pushdown(
                round_robin(Arc::new(
                    EmptyExec::new(schema()).with_partitions(PARTITIONS),
                )),
                None,
            ))),
            Some(LIMIT),
        );
        for plan in [above, below] {
            assert_eq!(
                names(&plan),
                [
                    "StagedCoalesceExec",
                    "FilterExec",
                    "ScanGateExec",
                    "MeteredExec",
                    "EmptyExec"
                ]
            );
        }
    }

    #[test]
    fn stages_an_index_answered_scan_under_its_per_partition_limit() {
        let plan = staged(
            Arc::new(LocalLimitExec::new(scan(PARTITIONS, true), LIMIT)),
            Some(LIMIT),
        );
        assert_eq!(
            names(&plan),
            [
                "StagedCoalesceExec",
                "LocalLimitExec",
                "ScanGateExec",
                "MeteredExec",
                "EmptyExec"
            ]
        );
    }

    #[test]
    fn leaves_a_scan_under_a_smaller_limit_of_its_own() {
        // The gate would stay shut once this limit had its rows, short of
        // the gather's.
        let plan = staged(
            Arc::new(LocalLimitExec::new(scan(PARTITIONS, true), LIMIT - 1)),
            Some(LIMIT),
        );
        assert_eq!(plan.name(), "CoalescePartitionsExec");
    }

    #[test]
    fn leaves_a_scan_that_takes_the_limit_itself() {
        let plan = staged(scan(PARTITIONS, false), Some(LIMIT));
        assert_eq!(plan.name(), "CoalescePartitionsExec");
    }

    #[test]
    fn leaves_a_large_limit_and_no_limit() {
        for fetch in [Some(MAX_STAGED_FETCH + 1), None] {
            let plan = staged(filter(scan(PARTITIONS, true)), fetch);
            assert_eq!(plan.name(), "CoalescePartitionsExec");
        }
    }

    #[test]
    fn leaves_a_single_partition() {
        let plan = staged(round_robin(filter(scan(1, true))), Some(LIMIT));
        assert_eq!(
            names(&plan),
            [
                "CoalescePartitionsExec",
                "RepartitionExec",
                "FilterExec",
                "MeteredExec",
                "EmptyExec"
            ]
        );
    }
}
