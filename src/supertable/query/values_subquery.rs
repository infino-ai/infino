// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`ValuesSubqueryRewrite`] — plan a `VALUES` list that holds a scalar
//! subquery as one-row projections, so the subquery runs before its value
//! is read.
//!
//! DataFusion 54 runs uncorrelated scalar subqueries in a
//! `ScalarSubqueryExec` that wraps the plan and fills each subquery's value
//! in when the plan *executes*. A `VALUES` list is different: the physical
//! planner turns it into an in-memory source by evaluating every cell while
//! it *plans*, before any `ScalarSubqueryExec` has run, so a cell such as
//! `(SELECT COUNT(*) FROM t)` fails planning with the internal error
//! "ScalarSubqueryExpr evaluated before the subquery was executed".
//!
//! A one-row projection over a single empty row computes the same row, and
//! a projection's scalar subqueries go through the normal execute-first
//! path. So a `VALUES` list with a scalar subquery in any cell becomes the
//! `UNION ALL` of one such projection per row, each cell cast to the list's
//! column type and named after its column so the schema is unchanged. A
//! `VALUES` list of plain constants is left alone.
//!
//! Like any `UNION ALL`, the rewritten rows are not ordered among
//! themselves; a query that needs the list's order says `ORDER BY`, which
//! SQL requires of every query anyway.

use datafusion::{
    common::tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    error::Result as DfResult,
    logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder, Values},
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
    prelude::cast,
};

/// The `VALUES`-with-scalar-subquery rewrite. Registered on every SQL
/// session after DataFusion's built-in rules.
#[derive(Debug, Default)]
pub(crate) struct ValuesSubqueryRewrite;

impl OptimizerRule for ValuesSubqueryRewrite {
    fn name(&self) -> &str {
        "values_subquery_rewrite"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        let LogicalPlan::Values(values) = &plan else {
            return Ok(Transformed::no(plan));
        };
        if !values.values.iter().flatten().any(holds_scalar_subquery) {
            return Ok(Transformed::no(plan));
        }
        Ok(Transformed::yes(values_as_projections(values)?))
    }
}

/// Whether `expr` holds a scalar subquery anywhere in its tree.
fn holds_scalar_subquery(expr: &Expr) -> bool {
    let mut found = false;
    expr.apply(|e| {
        if matches!(e, Expr::ScalarSubquery(_)) {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .expect("infallible visit");
    found
}

/// `UNION ALL` of one single-row projection per `VALUES` row, each cell cast
/// to its column's type and aliased to its column's name.
fn values_as_projections(values: &Values) -> DfResult<LogicalPlan> {
    let fields = values.schema.fields();
    let mut union: Option<LogicalPlanBuilder> = None;
    for row in &values.values {
        let cells = row
            .iter()
            .zip(fields.iter())
            .map(|(cell, field)| cast(cell.clone(), field.data_type().clone()).alias(field.name()));
        let projection = LogicalPlanBuilder::empty(true).project(cells)?.build()?;
        union = Some(match union {
            None => LogicalPlanBuilder::from(projection),
            Some(builder) => builder.union(projection)?,
        });
    }
    // `VALUES` never plans with zero rows; were one to, it stays as it was.
    match union {
        Some(builder) => builder.build(),
        None => Ok(LogicalPlan::Values(values.clone())),
    }
}
