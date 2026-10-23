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
//!
//! A query with an uncorrelated scalar subquery, such as
//! `WHERE v > (SELECT avg(v) FROM t) ORDER BY g`, is planned under a
//! `ScalarSubqueryExec` root whose first child is the query and whose other
//! children are the subqueries it runs first. Both walks down from the root
//! must pass through that first child:
//!
//! ```text
//!   ScalarSubqueryExec
//!   ├── ProjectionExec: CAST(g AS LargeUtf8)     <- the query: rows come from here
//!   │   └── RepartitionExec: RoundRobinBatch(8)  <- KeepSortedRoot removes this
//!   │       └── SortPreservingMergeExec: g ASC   <- OrderUnderScalarSubquery keeps this
//!   │           └── ...
//!   └── AggregateExec: avg(v)                    <- the subquery: one value
//! ```
//!
//! - DataFusion's `OutputRequirements` rule looks for the `ORDER BY` by walking
//!   single-child operators down from the root, so on DataFusion 54 it stops
//!   at this root and records none, and the sort is planned away.
//!   [`OrderUnderScalarSubquery`] runs that rule on the first child instead.
//! - With the sort kept, `EnforceDistribution` still puts the round-robin
//!   repartition between it and the cast, as it does without a subquery, so
//!   [`KeepSortedRoot`] steps into the first child to remove it.

use std::sync::Arc;

use datafusion::{
    config::ConfigOptions,
    error::Result as DfResult,
    physical_optimizer::{PhysicalOptimizerRule, output_requirements::OutputRequirements},
    physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, Partitioning, repartition::RepartitionExec,
        scalar_subquery::ScalarSubqueryExec,
    },
};

/// Rule name, as DataFusion lists it in `EXPLAIN VERBOSE`.
const RULE_NAME: &str = "KeepSortedRoot";

/// [`OrderUnderScalarSubquery`]'s name, as DataFusion lists it in
/// `EXPLAIN VERBOSE`.
const SUBQUERY_RULE_NAME: &str = "OrderUnderScalarSubquery";

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
    // A scalar subquery's root: the rows come from its first child.
    if let Some(subquery) = plan.downcast_ref::<ScalarSubqueryExec>() {
        return match unsplit_sorted_stream(subquery.input())? {
            Some(input) => Ok(Some(with_query_input(plan, input)?)),
            None => Ok(None),
        };
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

/// Records the `ORDER BY` of a query planned under a `ScalarSubqueryExec`
/// root, so DataFusion keeps its sort; see the module docs. Runs before
/// DataFusion's `OutputRequirements` and applies that same rule to the
/// root's first child, the query, where the sort is.
///
/// TODO: delete on the move to DataFusion 55, whose `OutputRequirements`
/// walks through `ScalarSubqueryExec` itself:
/// <https://github.com/apache/datafusion/pull/23677>.
#[derive(Debug, Default)]
pub(crate) struct OrderUnderScalarSubquery;

impl PhysicalOptimizerRule for OrderUnderScalarSubquery {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        let Some(subquery) = plan.downcast_ref::<ScalarSubqueryExec>() else {
            return Ok(plan);
        };
        let input =
            OutputRequirements::new_add_mode().optimize(Arc::clone(subquery.input()), config)?;
        with_query_input(&plan, input)
    }

    fn name(&self) -> &str {
        SUBQUERY_RULE_NAME
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// `plan`, a `ScalarSubqueryExec`, with its first child (the query) replaced
/// by `input` and its subqueries unchanged.
fn with_query_input(
    plan: &Arc<dyn ExecutionPlan>,
    input: Arc<dyn ExecutionPlan>,
) -> DfResult<Arc<dyn ExecutionPlan>> {
    let mut children: Vec<_> = plan.children().into_iter().cloned().collect();
    children[0] = input;
    Arc::clone(plan).with_new_children(children)
}
