// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`KeepSortedRoot`] — a physical optimizer rule that stops DataFusion
//! splitting a query's final sorted stream back into partitions.
//!
//! DataFusion's `OutputRequirements` rule records the global `ORDER BY` at
//! the sort itself, not at the plan root. An order-preserving operator above
//! the sort is then free to ask `EnforceDistribution` for parallelism, and a
//! projection that computes anything does: a round-robin repartition lands
//! between the merged, sorted stream and that projection. Each partition it
//! makes is still sorted, but the root now has many partitions and nothing
//! merges them — collecting concatenates them in completion order, so the
//! rows of an `ORDER BY` come back out of order.
//!
//! The projection that triggers it on every string-returning query is the
//! final cast to `LargeUtf8` that `expand_views_at_output` adds (see
//! `budgeted_session_context`): `SELECT CAST(_id AS VARCHAR) ... ORDER BY _id`
//! over two superfiles returned each file's rows ascending but concatenated
//! larger-ids-first.
//!
//! The rule walks down from the root through single-child operators that keep
//! their input's row order, and removes the round-robin repartition it meets
//! whose input is one ordered partition. The operators above then run on that
//! one stream, and the root stays a single, sorted partition. A merge on top
//! would also restore the order, but only by paying for an exchange and a
//! merge to parallelize what is typically a cast.

use std::sync::Arc;

use datafusion::{
    config::ConfigOptions,
    error::Result as DfResult,
    physical_optimizer::PhysicalOptimizerRule,
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, Partitioning, repartition::RepartitionExec,
    },
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "KeepSortedRoot";

/// Removes the round-robin repartition that splits a plan's sorted result
/// above its sort; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct KeepSortedRoot;

impl PhysicalOptimizerRule for KeepSortedRoot {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(unsplit_sorted_stream(&plan)?.unwrap_or(plan))
    }

    fn name(&self) -> &str {
        RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// `plan` without the round-robin repartition over one ordered partition that
/// sits on its order-keeping chain from the root, or `None` when the chain has
/// none. Removing a round-robin repartition never changes which rows a plan
/// returns — only how many partitions carry them — so the rewrite is safe for
/// any operator above it.
fn unsplit_sorted_stream(
    plan: &Arc<dyn ExecutionPlan>,
) -> DfResult<Option<Arc<dyn ExecutionPlan>>> {
    if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
        && matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
        && repartition.input().output_partitioning().partition_count() == 1
        && repartition.input().output_ordering().is_some()
    {
        return Ok(Some(Arc::clone(repartition.input())));
    }
    // Past an operator that reorders rows, or one with several inputs, the
    // root's order no longer comes from this stream.
    let [child] = plan.children()[..] else {
        return Ok(None);
    };
    if plan.maintains_input_order() != [true] {
        return Ok(None);
    }
    match unsplit_sorted_stream(child)? {
        Some(child) => Ok(Some(Arc::clone(plan).with_new_children(vec![child])?)),
        None => Ok(None),
    }
}
