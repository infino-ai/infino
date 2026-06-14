// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Covered/residual evaluation for filter-aligned aggregates over the
//! manifest-statistics "aggregation tree".
//!
//! For an ungrouped aggregate whose `WHERE` clause is a single-column
//! range, the manifest already knows each segment's bounds for that
//! column. Segments fall into three classes:
//!
//!   * **disjoint** — bounds outside the range: contribute nothing;
//!   * **covered** — bounds fully inside the range AND tombstone-free
//!     AND carrying the stats the aggregate needs: their contribution
//!     is computed from manifest statistics, no scan;
//!   * **boundary** — everything else: scanned for real (the
//!     *residual*), with the original predicate re-applied.
//!
//! The rewrite turns `Aggregate(Filter(TableScan))` into
//!
//! ```text
//! Projection(combine(literal partials, residual partials))
//!   └─ Aggregate(partial fns)
//!        └─ Filter(original predicate)
//!             └─ TableScan(provider restricted to boundary segments)
//! ```
//!
//! so cost scales with the *boundary* segments instead of the rows in
//! range. Time-range partitioned tables align range filters with
//! segment bounds by construction, collapsing wide range aggregates
//! to a couple of boundary scans.
//!
//! Soundness rules (each bail leaves the plan untouched — the normal
//! scan path is always correct):
//!
//!   * the filter must be EXACTLY a conjunction of range comparisons
//!     over one column (an unrecognized conjunct disables the rewrite
//!     — partial predicate coverage would answer the wrong query);
//!   * a segment with any (or unknown) tombstones demotes to
//!     boundary;
//!   * a covered segment missing a required stat demotes to boundary;
//!   * only `COUNT(*)`, `SUM`, `MIN`, `MAX`, `AVG` over bare columns,
//!     no DISTINCT / FILTER / ORDER BY, no GROUP BY;
//!   * a provider already restricted to a segment subset is the
//!     rewrite's own residual — never rewritten again (idempotency).

use std::collections::HashSet;
use std::sync::Arc;

use arrow_schema::DataType;
use datafusion::common::ScalarValue;
use datafusion::common::tree_node::Transformed;
use datafusion::datasource::DefaultTableSource;
use datafusion::datasource::provider_as_source;
use datafusion::error::{DataFusionError, Result as DfResult};
use datafusion::functions::core::expr_fn::{coalesce, greatest, least};
use datafusion::functions_aggregate::expr_fn::{count, max, min, sum};
use datafusion::logical_expr::{
    Expr, Filter, LogicalPlan, LogicalPlanBuilder, Operator, TableScan, lit,
};
use datafusion::optimizer::optimizer::ApplyOrder;
use datafusion::optimizer::{OptimizerConfig, OptimizerRule};
use datafusion::prelude::cast;

use crate::supertable::manifest::{SuperfileEntry, add_sum_arrays};
use crate::supertable::options::{DECIMAL128_PRECISION, DECIMAL128_SCALE};
use crate::supertable::query::provider::SupertableProvider;

/// The covered/residual aggregate rewrite. Registered on the
/// `query_sql` session after DataFusion's built-in rules.
#[derive(Debug, Default)]
pub(crate) struct CoveredAggregateRewrite;

impl OptimizerRule for CoveredAggregateRewrite {
    fn name(&self) -> &str {
        "covered_aggregate_rewrite"
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

/// One supported aggregate output.
enum AggKind {
    CountStar,
    Sum(String),
    Min(String),
    Max(String),
    Avg(String),
}

/// One side of the range, with inclusivity.
#[derive(Clone)]
struct Bound {
    value: ScalarValue,
    inclusive: bool,
}

/// The extracted single-column range filter.
struct RangeFilter {
    column: String,
    lo: Option<Bound>,
    hi: Option<Bound>,
}

/// Segment classification against the range.
enum Class {
    Disjoint,
    Covered,
    Boundary,
}

fn try_rewrite(plan: &LogicalPlan) -> DfResult<Option<LogicalPlan>> {
    let LogicalPlan::Aggregate(agg) = plan else {
        return Ok(None);
    };
    if !agg.group_expr.is_empty() {
        return Ok(None);
    }
    let Some(kinds) = parse_aggregates(&agg.aggr_expr) else {
        return Ok(None);
    };

    // Input shape: Filter over TableScan (a pure-column Projection in
    // between is looked through).
    let (predicate, scan) = match peel_input(agg.input.as_ref()) {
        Some(found) => found,
        None => return Ok(None),
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.is_segment_restricted() {
        // Our own residual scan — never rewrite again.
        return Ok(None);
    }
    let Some(range) = extract_range(&predicate) else {
        return Ok(None);
    };

    // Hierarchical manifests keep segments in lazily-loaded parts; the
    // flat view may be partial, so classification would be unsound.
    let manifest = provider.manifest();
    if manifest.list.is_some() {
        return Ok(None);
    }
    let id_column = manifest.options.id_column.as_str();

    // Classify every segment.
    let mut covered: Vec<&Arc<SuperfileEntry>> = Vec::new();
    let mut boundary: HashSet<uuid::Uuid> = HashSet::new();
    for entry in &manifest.superfiles {
        let class = classify(entry, id_column, &range);
        match class {
            Class::Disjoint => {}
            Class::Covered => {
                if provider.entry_is_clean(entry) && has_required_stats(entry, &kinds) {
                    covered.push(entry);
                } else {
                    boundary.insert(entry.superfile_id);
                }
            }
            Class::Boundary => {
                boundary.insert(entry.superfile_id);
            }
        }
    }
    if covered.is_empty() {
        // Nothing answerable from statistics — the rewrite would only
        // add plan noise over the normal scan.
        return Ok(None);
    }

    // Accumulate the covered partials per aggregate output.
    let mut partials: Vec<Partial> = Vec::with_capacity(kinds.len());
    for kind in &kinds {
        match accumulate_partial(kind, &covered) {
            Some(partial) => partials.push(partial),
            // A fold failure (overflow, type surprise) is always safe
            // to decline.
            None => return Ok(None),
        }
    }

    // Residual: the original predicate over the boundary-restricted
    // provider. An empty boundary set still scans (an empty relation),
    // and the global Aggregate below then yields its empty-input row —
    // exactly what the combiners expect.
    let restricted = Arc::new(provider.restricted_to(boundary));
    let mut builder = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        provider_as_source(restricted as Arc<dyn datafusion::catalog::TableProvider>),
        None,
    )?
    .filter(predicate.clone())?;

    // Partial aggregates over the residual, one batch per output (AVG
    // needs two: sum + non-null count).
    let mut partial_exprs: Vec<Expr> = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        match kind {
            AggKind::CountStar => {
                partial_exprs.push(count(lit(1i64)).alias(format!("__resid_{i}_cnt")));
            }
            AggKind::Sum(col) => {
                partial_exprs
                    .push(sum(datafusion::prelude::col(col)).alias(format!("__resid_{i}_sum")));
            }
            AggKind::Min(col) => {
                partial_exprs
                    .push(min(datafusion::prelude::col(col)).alias(format!("__resid_{i}_min")));
            }
            AggKind::Max(col) => {
                partial_exprs
                    .push(max(datafusion::prelude::col(col)).alias(format!("__resid_{i}_max")));
            }
            AggKind::Avg(col) => {
                partial_exprs
                    .push(sum(datafusion::prelude::col(col)).alias(format!("__resid_{i}_sum")));
                partial_exprs
                    .push(count(datafusion::prelude::col(col)).alias(format!("__resid_{i}_cnt")));
            }
        }
    }
    builder = builder.aggregate(Vec::<Expr>::new(), partial_exprs)?;

    // Final projection: combine each residual partial with its covered
    // literal, aliased to the original aggregate's output name so the
    // rewritten plan's schema matches the original node's exactly.
    let out_names: Vec<String> = agg
        .schema
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    let mut final_exprs: Vec<Expr> = Vec::with_capacity(kinds.len());
    for (i, (kind, partial)) in kinds.iter().zip(&partials).enumerate() {
        let name = &out_names[i];
        let expr = match (kind, partial) {
            (AggKind::CountStar, Partial::Count(n)) => {
                // COUNT over an empty residual is 0, never NULL.
                (datafusion::prelude::col(format!("__resid_{i}_cnt")) + lit(*n)).alias(name)
            }
            (AggKind::Sum(_), Partial::Sum(value)) => {
                let zero = typed_zero(value)?;
                (coalesce(vec![
                    datafusion::prelude::col(format!("__resid_{i}_sum")),
                    lit(zero),
                ]) + lit(value.clone()))
                .alias(name)
            }
            (AggKind::Min(_), Partial::Bound(value)) => least(vec![
                datafusion::prelude::col(format!("__resid_{i}_min")),
                lit(value.clone()),
            ])
            .alias(name),
            (AggKind::Max(_), Partial::Bound(value)) => greatest(vec![
                datafusion::prelude::col(format!("__resid_{i}_max")),
                lit(value.clone()),
            ])
            .alias(name),
            (AggKind::Avg(_), Partial::Avg { sum, count }) => {
                let zero = typed_zero(sum)?;
                let total_sum = coalesce(vec![
                    datafusion::prelude::col(format!("__resid_{i}_sum")),
                    lit(zero),
                ]) + lit(sum.clone());
                let total_cnt = datafusion::prelude::col(format!("__resid_{i}_cnt")) + lit(*count);
                (cast(total_sum, DataType::Float64) / cast(total_cnt, DataType::Float64))
                    .alias(name)
            }
            _ => return Ok(None),
        };
        final_exprs.push(expr);
    }

    let rewritten = builder.project(final_exprs)?.build()?;
    Ok(Some(rewritten))
}

/// Covered-side partial state per aggregate output.
enum Partial {
    Count(i64),
    Sum(ScalarValue),
    Bound(ScalarValue),
    Avg { sum: ScalarValue, count: i64 },
}

/// Fold one aggregate's covered contribution from manifest stats.
/// `None` = decline the rewrite (overflow / unexpected shape).
fn accumulate_partial(kind: &AggKind, covered: &[&Arc<SuperfileEntry>]) -> Option<Partial> {
    match kind {
        AggKind::CountStar => {
            let mut total: i64 = 0;
            for entry in covered {
                total = total.checked_add(i64::try_from(entry.n_docs).ok()?)?;
            }
            Some(Partial::Count(total))
        }
        AggKind::Sum(col) => Some(Partial::Sum(fold_sums(covered, col)?)),
        AggKind::Min(col) => {
            let (min, _) = fold_bounds(covered, col)?;
            Some(Partial::Bound(min))
        }
        AggKind::Max(col) => {
            let (_, max) = fold_bounds(covered, col)?;
            Some(Partial::Bound(max))
        }
        AggKind::Avg(col) => {
            let sum = fold_sums(covered, col)?;
            let mut count: i64 = 0;
            for entry in covered {
                let nulls = *entry.scalar_stats.null_counts.get(col)?;
                let non_null = entry.n_docs.checked_sub(nulls)?;
                count = count.checked_add(i64::try_from(non_null).ok()?)?;
            }
            if count == 0 {
                // All covered values NULL: AVG semantics get fiddly
                // (0/0); decline and let the scan answer.
                return None;
            }
            Some(Partial::Avg { sum, count })
        }
    }
}

fn fold_sums(covered: &[&Arc<SuperfileEntry>], col: &str) -> Option<ScalarValue> {
    let mut acc: Option<arrow_array::ArrayRef> = None;
    for entry in covered {
        let part = entry.scalar_stats.sums.get(col)?;
        acc = Some(match acc {
            None => Arc::clone(part),
            Some(total) => add_sum_arrays(&total, part)?,
        });
    }
    ScalarValue::try_from_array(&acc?, 0).ok()
}

fn fold_bounds(covered: &[&Arc<SuperfileEntry>], col: &str) -> Option<(ScalarValue, ScalarValue)> {
    let mut acc: Option<(ScalarValue, ScalarValue)> = None;
    for entry in covered {
        let (min_arr, max_arr) = entry.scalar_stats.cols.get(col)?;
        let min = ScalarValue::try_from_array(min_arr, 0).ok()?;
        let max = ScalarValue::try_from_array(max_arr, 0).ok()?;
        if min.is_null() || max.is_null() {
            return None;
        }
        acc = match acc {
            None => Some((min, max)),
            Some((cur_min, cur_max)) => {
                let new_min = if min.partial_cmp(&cur_min)? == std::cmp::Ordering::Less {
                    min
                } else {
                    cur_min
                };
                let new_max = if max.partial_cmp(&cur_max)? == std::cmp::Ordering::Greater {
                    max
                } else {
                    cur_max
                };
                Some((new_min, new_max))
            }
        };
    }
    acc
}

/// A zero literal of the same type as `value`, for `coalesce` over an
/// empty residual SUM. `Err` only for types `column_sum` never emits.
fn typed_zero(value: &ScalarValue) -> DfResult<ScalarValue> {
    match value {
        ScalarValue::Int64(_) => Ok(ScalarValue::Int64(Some(0))),
        ScalarValue::UInt64(_) => Ok(ScalarValue::UInt64(Some(0))),
        ScalarValue::Float64(_) => Ok(ScalarValue::Float64(Some(0.0))),
        other => Err(DataFusionError::Internal(format!(
            "covered_agg: unexpected sum type {other:?}"
        ))),
    }
}

/// Parse the aggregate outputs; `None` if any is unsupported.
fn parse_aggregates(exprs: &[Expr]) -> Option<Vec<AggKind>> {
    let mut kinds = Vec::with_capacity(exprs.len());
    for expr in exprs {
        // Outputs may carry an alias wrapper.
        let inner = match expr {
            Expr::Alias(alias) => alias.expr.as_ref(),
            other => other,
        };
        let Expr::AggregateFunction(agg) = inner else {
            return None;
        };
        let params = &agg.params;
        if params.distinct || params.filter.is_some() || !params.order_by.is_empty() {
            return None;
        }
        let kind = match (agg.func.name(), params.args.as_slice()) {
            ("count", []) => AggKind::CountStar,
            ("count", [Expr::Literal(value, _)]) if !value.is_null() => AggKind::CountStar,
            ("sum", [Expr::Column(c)]) => AggKind::Sum(c.name.clone()),
            ("min", [Expr::Column(c)]) => AggKind::Min(c.name.clone()),
            ("max", [Expr::Column(c)]) => AggKind::Max(c.name.clone()),
            // AVG's int arguments arrive wrapped in a coercion cast to
            // Float64; peeling it is safe here (and only here) because
            // the combiner re-casts both totals to Float64 itself.
            ("avg", [arg]) => AggKind::Avg(avg_column(arg)?),
            _ => return None,
        };
        kinds.push(kind);
    }
    Some(kinds)
}

/// AVG's argument: a bare column, or a column under the planner's
/// numeric-coercion cast (`AVG(int_col)` plans as
/// `avg(CAST(int_col AS Float64))`).
fn avg_column(arg: &Expr) -> Option<String> {
    match arg {
        Expr::Column(c) => Some(c.name.clone()),
        Expr::Cast(cast) => match cast.expr.as_ref() {
            Expr::Column(c) => Some(c.name.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Peel `Aggregate`'s input down to `(predicate, scan)`: a `Filter`
/// directly over a `TableScan`, optionally with a pure-column
/// `Projection` above the filter. `None` for any other shape.
fn peel_input(input: &LogicalPlan) -> Option<(Expr, &TableScan)> {
    let mut node = input;
    if let LogicalPlan::Projection(projection) = node {
        if !projection.expr.iter().all(|e| matches!(e, Expr::Column(_))) {
            return None;
        }
        node = projection.input.as_ref();
    }
    let LogicalPlan::Filter(Filter {
        predicate, input, ..
    }) = node
    else {
        return None;
    };
    let LogicalPlan::TableScan(scan) = input.as_ref() else {
        return None;
    };
    Some((predicate.clone(), scan))
}

/// The provider behind a scan, when it is ours.
fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    scan.source
        .as_any()
        .downcast_ref::<DefaultTableSource>()?
        .table_provider
        .as_any()
        .downcast_ref::<SupertableProvider>()
}

/// Strictly extract a single-column range from the predicate: a
/// conjunction whose every leaf is `col <op> literal` (or `literal
/// <op> col`) over ONE shared column, ops in `> >= < <= =` /
/// `BETWEEN`. Any other leaf → `None` (the rewrite must see the whole
/// predicate or nothing).
fn extract_range(predicate: &Expr) -> Option<RangeFilter> {
    let mut leaves: Vec<(String, Operator, ScalarValue)> = Vec::new();
    collect_range_leaves(predicate, &mut leaves)?;
    let mut range: Option<RangeFilter> = None;
    for (column, op, value) in leaves {
        let entry = range.get_or_insert_with(|| RangeFilter {
            column: column.clone(),
            lo: None,
            hi: None,
        });
        if entry.column != column {
            return None;
        }
        let (slot, inclusive) = match op {
            Operator::Gt => (&mut entry.lo, false),
            Operator::GtEq => (&mut entry.lo, true),
            Operator::Lt => (&mut entry.hi, false),
            Operator::LtEq => (&mut entry.hi, true),
            Operator::Eq => {
                // Equality = both bounds inclusive. Two different Eq
                // literals would be an always-false predicate — decline
                // rather than reason about it.
                if entry.lo.is_some() || entry.hi.is_some() {
                    return None;
                }
                entry.lo = Some(Bound {
                    value: value.clone(),
                    inclusive: true,
                });
                entry.hi = Some(Bound {
                    value,
                    inclusive: true,
                });
                continue;
            }
            _ => return None,
        };
        if slot.is_some() {
            // Duplicate bound on one side: tightest-wins logic isn't
            // worth the edge cases; decline.
            return None;
        }
        *slot = Some(Bound { value, inclusive });
    }
    let range = range?;
    (range.lo.is_some() || range.hi.is_some()).then_some(range)
}

fn collect_range_leaves(expr: &Expr, out: &mut Vec<(String, Operator, ScalarValue)>) -> Option<()> {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::And => {
            collect_range_leaves(&binary.left, out)?;
            collect_range_leaves(&binary.right, out)
        }
        Expr::BinaryExpr(binary) => {
            let (column, op, value) = match (binary.left.as_ref(), binary.right.as_ref()) {
                (Expr::Column(c), Expr::Literal(v, _)) => (c.name.clone(), binary.op, v.clone()),
                (Expr::Literal(v, _), Expr::Column(c)) => {
                    (c.name.clone(), binary.op.swap()?, v.clone())
                }
                _ => return None,
            };
            if value.is_null() {
                return None;
            }
            out.push((column, op, value));
            Some(())
        }
        Expr::Between(between) if !between.negated => {
            let Expr::Column(c) = between.expr.as_ref() else {
                return None;
            };
            let (Expr::Literal(lo, _), Expr::Literal(hi, _)) =
                (between.low.as_ref(), between.high.as_ref())
            else {
                return None;
            };
            if lo.is_null() || hi.is_null() {
                return None;
            }
            out.push((c.name.clone(), Operator::GtEq, lo.clone()));
            out.push((c.name.clone(), Operator::LtEq, hi.clone()));
            Some(())
        }
        _ => None,
    }
}

/// Classify one segment's `[seg_min, seg_max]` for `range.column`
/// against the range. Missing bounds → `Boundary` (conservative).
fn classify(entry: &SuperfileEntry, id_column: &str, range: &RangeFilter) -> Class {
    let bounds = if range.column == id_column {
        Some((
            ScalarValue::Decimal128(Some(entry.id_min), DECIMAL128_PRECISION, DECIMAL128_SCALE),
            ScalarValue::Decimal128(Some(entry.id_max), DECIMAL128_PRECISION, DECIMAL128_SCALE),
        ))
    } else {
        entry
            .scalar_stats
            .cols
            .get(&range.column)
            .and_then(|(mn, mx)| {
                let mn = ScalarValue::try_from_array(mn, 0).ok()?;
                let mx = ScalarValue::try_from_array(mx, 0).ok()?;
                (!mn.is_null() && !mx.is_null()).then_some((mn, mx))
            })
    };
    let Some((seg_min, seg_max)) = bounds else {
        return Class::Boundary;
    };

    // Disjoint: the whole segment sits outside the range.
    if let Some(lo) = &range.lo {
        let cmp = seg_max.partial_cmp(&lo.value);
        match (cmp, lo.inclusive) {
            (Some(std::cmp::Ordering::Less), _) => return Class::Disjoint,
            (Some(std::cmp::Ordering::Equal), false) => return Class::Disjoint,
            (None, _) => return Class::Boundary,
            _ => {}
        }
    }
    if let Some(hi) = &range.hi {
        let cmp = seg_min.partial_cmp(&hi.value);
        match (cmp, hi.inclusive) {
            (Some(std::cmp::Ordering::Greater), _) => return Class::Disjoint,
            (Some(std::cmp::Ordering::Equal), false) => return Class::Disjoint,
            (None, _) => return Class::Boundary,
            _ => {}
        }
    }

    // Covered: the whole segment sits inside the range.
    let lo_ok = match &range.lo {
        None => true,
        Some(lo) => match (seg_min.partial_cmp(&lo.value), lo.inclusive) {
            (Some(std::cmp::Ordering::Greater), _) => true,
            (Some(std::cmp::Ordering::Equal), true) => true,
            (None, _) => return Class::Boundary,
            _ => false,
        },
    };
    let hi_ok = match &range.hi {
        None => true,
        Some(hi) => match (seg_max.partial_cmp(&hi.value), hi.inclusive) {
            (Some(std::cmp::Ordering::Less), _) => true,
            (Some(std::cmp::Ordering::Equal), true) => true,
            (None, _) => return Class::Boundary,
            _ => false,
        },
    };
    if lo_ok && hi_ok {
        Class::Covered
    } else {
        Class::Boundary
    }
}

/// Do the manifest stats cover everything `kinds` needs from a
/// covered segment?
fn has_required_stats(entry: &SuperfileEntry, kinds: &[AggKind]) -> bool {
    kinds.iter().all(|kind| match kind {
        AggKind::CountStar => true,
        AggKind::Sum(col) => entry.scalar_stats.sums.contains_key(col),
        AggKind::Min(col) | AggKind::Max(col) => entry.scalar_stats.cols.contains_key(col),
        AggKind::Avg(col) => {
            entry.scalar_stats.sums.contains_key(col)
                && entry.scalar_stats.null_counts.contains_key(col)
        }
    })
}
