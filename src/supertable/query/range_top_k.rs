// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RangeTopK`]: `ORDER BY column LIMIT k` on a range-indexed column reads
//! the `k` rows the range index puts first, instead of every row the range
//! holds.
//!
//! Without it the scan hands every row of the `WHERE`'s range to the sort,
//! which keeps the best `k`: a range holding half the table decodes half the
//! table to return ten rows. The range index already holds the column in
//! order, so the best `k` of each superfile are its first `k` keys inside
//! the range, a block or two away. The `k`-th best key across all of them is
//! the cutoff no returned row can pass, and the scan then decodes only the
//! rows collected below it:
//!
//! ```text
//!   Sort fetch=10: price ASC            Sort fetch=10: price ASC
//!     Filter price > 990                  Filter price > 990
//!       TableScan                           TableScan, top-k hint (price, asc, 10)
//!         every row with price > 990          per superfile: its first 10 keys
//!                                             above 990, cut at the 10th overall
//! ```
//!
//! The rule only marks the scan, with a provider carrying the hint; the
//! provider does the reading (`SupertableProvider::top_k_rows`), since that
//! needs the superfiles. It fires only when every filter under the sort is
//! a range on the sort column: a filter on another column could reject the
//! rows the index puts first, and the cutoff would then drop rows that
//! belong in the answer. A sort with nulls first and no filter is left
//! alone too, as nulls are in no range index and would come first. The
//! sort and the filters stay where they are, so the answer is the one they
//! give over the rows the scan returns.

use std::sync::Arc;

use datafusion::{
    catalog::TableProvider,
    common::tree_node::Transformed,
    datasource::{DefaultTableSource, provider_as_source},
    error::Result as DfResult,
    logical_expr::{
        Expr, Filter, LogicalPlan, Sort, TableScan, TableSource, utils::split_conjunction,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
};

use crate::supertable::query::{
    candidate::single_range,
    provider::{SupertableProvider, TopKHint},
};

/// The rule. Registered on the `query_sql` session after DataFusion's
/// built-in rules, which push the `LIMIT` into the sort as its fetch.
#[derive(Debug, Default)]
pub(crate) struct RangeTopK;

impl OptimizerRule for RangeTopK {
    fn name(&self) -> &str {
        "range_top_k"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DfResult<Transformed<LogicalPlan>> {
        match try_rewrite(&plan)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// The sort with its scan's provider swapped for one carrying the hint, or
/// `None` when the plan is not the shape the hint is sound for.
fn try_rewrite(plan: &LogicalPlan) -> DfResult<Option<LogicalPlan>> {
    let LogicalPlan::Sort(Sort {
        expr,
        input,
        fetch: Some(k),
    }) = plan
    else {
        return Ok(None);
    };
    let [sort_expr] = expr.as_slice() else {
        return Ok(None);
    };
    let Expr::Column(column) = &sort_expr.expr else {
        return Ok(None);
    };
    // Walk to the scan through filters and plain column projections that
    // keep the sort column, collecting the filters' predicates.
    let mut predicates: Vec<Expr> = Vec::new();
    let mut node = input.as_ref();
    let scan = loop {
        match node {
            LogicalPlan::Projection(projection)
                if projection.expr.iter().all(|e| matches!(e, Expr::Column(_))) =>
            {
                node = projection.input.as_ref();
            }
            LogicalPlan::Filter(Filter {
                predicate, input, ..
            }) => {
                predicates.extend(split_conjunction(predicate).into_iter().cloned());
                node = input.as_ref();
            }
            LogicalPlan::TableScan(scan) => break scan,
            _ => return Ok(None),
        }
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.has_top_k() {
        return Ok(None);
    }
    predicates.extend(scan.filters.iter().flat_map(split_conjunction).cloned());
    let range_cols = provider.manifest().table_schema().range_column_types();
    if single_range(&predicates, &column.name, &range_cols).is_none() {
        return Ok(None);
    }
    if predicates.is_empty() && sort_expr.nulls_first {
        return Ok(None);
    }
    let hinted = Arc::new(provider.with_top_k(TopKHint {
        column: column.name.clone(),
        descending: !sort_expr.asc,
        k: *k,
    }));
    let source = provider_as_source(hinted as Arc<dyn TableProvider>);
    Ok(Some(with_scan_source(plan, &source)?))
}

/// Our provider behind `scan`, when it is ours.
fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    let source = scan.source.downcast_ref::<DefaultTableSource>()?;
    source.table_provider.downcast_ref::<SupertableProvider>()
}

/// `plan` with the scan at the bottom of its single-input chain reading
/// from `source`.
fn with_scan_source(plan: &LogicalPlan, source: &Arc<dyn TableSource>) -> DfResult<LogicalPlan> {
    if let LogicalPlan::TableScan(scan) = plan {
        return Ok(LogicalPlan::TableScan(TableScan {
            source: Arc::clone(source),
            ..scan.clone()
        }));
    }
    let inputs = plan
        .inputs()
        .into_iter()
        .map(|input| with_scan_source(input, source))
        .collect::<DfResult<Vec<_>>>()?;
    plan.with_new_exprs(plan.expressions(), inputs)
}
