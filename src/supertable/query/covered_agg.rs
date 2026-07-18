// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Covered/residual evaluation for filter-aligned aggregates over the
//! manifest-statistics "aggregation tree".
//!
//! Exact low-cardinality frequencies answer `COUNT(*)` over a
//! single-column equality/range and `GROUP BY column, COUNT(*)` directly.
//! Other ungrouped aggregates whose `WHERE` clause is a single-column range
//! use the manifest's segment bounds. Segments fall into three classes:
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
//!     no DISTINCT / FILTER / ORDER BY; the sole grouped shape is one
//!     low-cardinality column plus `COUNT(*)`;
//!   * a provider already restricted to a segment subset is the
//!     rewrite's own residual — never rewritten again (idempotency).

use std::{cmp::Ordering, collections::HashSet, sync::Arc};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion::{
    catalog::TableProvider,
    common::{ScalarValue, tree_node::Transformed},
    datasource::{DefaultTableSource, provider_as_source},
    error::{DataFusionError, Result as DfResult},
    functions::core::expr_fn::{coalesce, greatest, least},
    functions_aggregate::expr_fn::{count, max, min, sum},
    logical_expr::{
        Aggregate, Expr, Filter, LogicalPlan, LogicalPlanBuilder, Operator, TableScan, lit,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
    prelude::{cast, col},
};
use uuid::Uuid;

use crate::supertable::{
    manifest::{SuperfileEntry, add_sum_arrays, list::ScalarValueCounts},
    options::{DECIMAL128_PRECISION, DECIMAL128_SCALE},
    query::provider::SupertableProvider,
};

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
    if let Some(rewritten) = rewrite_grouped_count_from_value_counts(agg)? {
        return Ok(Some(rewritten));
    }
    if !agg.group_expr.is_empty() {
        return Ok(None);
    }
    let Some(kinds) = parse_aggregates(&agg.aggr_expr) else {
        return Ok(None);
    };

    // Unfiltered aggregate over a complete, clean snapshot: every segment is
    // covered, so emit one literal row directly. This handles SUM/AVG as well
    // as COUNT/MIN/MAX and avoids invoking the provider's Parquet scan at all.
    if let Some(scan) = peel_unfiltered_scan(agg.input.as_ref()) {
        let Some(provider) = provider_of(scan) else {
            return Ok(None);
        };
        if provider.is_segment_restricted() {
            return Ok(None);
        }
        let Some(superfiles) = provider.manifest().complete_flat_superfiles() else {
            return Ok(None);
        };
        if !superfiles
            .iter()
            .all(|entry| provider.entry_is_clean(entry) && has_required_stats(entry, &kinds))
        {
            return Ok(None);
        }
        let covered: Vec<&Arc<SuperfileEntry>> = superfiles.iter().collect();
        let mut partials = Vec::with_capacity(kinds.len());
        for kind in &kinds {
            let Some(partial) = accumulate_partial(kind, &covered) else {
                return Ok(None);
            };
            partials.push(partial);
        }
        let out_names: Vec<String> = agg
            .schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let mut expressions = Vec::with_capacity(kinds.len());
        for ((kind, partial), name) in kinds.iter().zip(&partials).zip(out_names) {
            let expression = match (kind, partial) {
                (AggKind::CountStar, Partial::Count(count)) => lit(*count),
                (AggKind::Sum(_), Partial::Sum(value))
                | (AggKind::Min(_), Partial::Bound(value))
                | (AggKind::Max(_), Partial::Bound(value)) => lit(value.clone()),
                (AggKind::Avg(_), Partial::Avg { sum, count }) => {
                    cast(lit(sum.clone()), DataType::Float64) / cast(lit(*count), DataType::Float64)
                }
                _ => return Ok(None),
            };
            expressions.push(expression.alias(name));
        }
        let rewritten = LogicalPlanBuilder::empty(true)
            .project(expressions)?
            .build()?;
        return Ok(Some(rewritten));
    }

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

    let manifest = provider.manifest();
    // Hierarchical manifests may expose a partial flat view. Rewrite only
    // when the resident entries are provably complete; eagerly hydrated
    // persisted tables then receive the same no-scan aggregate path as
    // in-process tables, while genuinely lazy views decline safely.
    let Some(superfiles) = manifest.complete_flat_superfiles() else {
        return Ok(None);
    };
    let id_column = manifest.options.id_column.as_str();

    if matches!(kinds.as_slice(), [AggKind::CountStar])
        && superfiles
            .iter()
            .all(|entry| provider.entry_is_clean(entry))
        && let Some(value_counts) = provider.exact_value_counts(&range.column)
        && let Some(count) = count_range_from_value_counts(&value_counts, &range)
    {
        let name = agg.schema.field(0).name().clone();
        return Ok(Some(
            LogicalPlanBuilder::empty(true)
                .project(vec![lit(count).alias(name)])?
                .build()?,
        ));
    }

    // Classify every segment.
    let mut covered: Vec<&Arc<SuperfileEntry>> = Vec::new();
    let mut boundary: HashSet<Uuid> = HashSet::new();
    for entry in superfiles {
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
        provider_as_source(restricted as Arc<dyn TableProvider>),
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
            AggKind::Sum(column) => {
                partial_exprs.push(sum(col(column)).alias(format!("__resid_{i}_sum")));
            }
            AggKind::Min(column) => {
                partial_exprs.push(min(col(column)).alias(format!("__resid_{i}_min")));
            }
            AggKind::Max(column) => {
                partial_exprs.push(max(col(column)).alias(format!("__resid_{i}_max")));
            }
            AggKind::Avg(column) => {
                partial_exprs.push(sum(col(column)).alias(format!("__resid_{i}_sum")));
                partial_exprs.push(count(col(column)).alias(format!("__resid_{i}_cnt")));
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
                (col(format!("__resid_{i}_cnt")) + lit(*n)).alias(name)
            }
            (AggKind::Sum(_), Partial::Sum(value)) => {
                let zero = typed_zero(value)?;
                (coalesce(vec![col(format!("__resid_{i}_sum")), lit(zero)]) + lit(value.clone()))
                    .alias(name)
            }
            (AggKind::Min(_), Partial::Bound(value)) => {
                least(vec![col(format!("__resid_{i}_min")), lit(value.clone())]).alias(name)
            }
            (AggKind::Max(_), Partial::Bound(value)) => {
                greatest(vec![col(format!("__resid_{i}_max")), lit(value.clone())]).alias(name)
            }
            (AggKind::Avg(_), Partial::Avg { sum, count }) => {
                let zero = typed_zero(sum)?;
                let total_sum =
                    coalesce(vec![col(format!("__resid_{i}_sum")), lit(zero)]) + lit(sum.clone());
                let total_cnt = col(format!("__resid_{i}_cnt")) + lit(*count);
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

fn rewrite_grouped_count_from_value_counts(aggregate: &Aggregate) -> DfResult<Option<LogicalPlan>> {
    let [Expr::Column(group_column)] = aggregate.group_expr.as_slice() else {
        return Ok(None);
    };
    let Some(kinds) = parse_aggregates(&aggregate.aggr_expr) else {
        return Ok(None);
    };
    if !matches!(kinds.as_slice(), [AggKind::CountStar]) {
        return Ok(None);
    }
    let Some(scan) = peel_unfiltered_scan(aggregate.input.as_ref()) else {
        return Ok(None);
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.is_segment_restricted() {
        return Ok(None);
    }
    let Some(superfiles) = provider.manifest().complete_flat_superfiles() else {
        return Ok(None);
    };
    if !superfiles.iter().all(|entry| {
        provider.entry_is_clean(entry)
            && entry
                .scalar_stats
                .get(&group_column.name)
                .is_some_and(|stats| stats.null_count == Some(0))
    }) {
        return Ok(None);
    }

    let Some(value_counts) = provider.exact_value_counts(&group_column.name) else {
        return Ok(None);
    };
    let mut rows = Vec::with_capacity(value_counts.entries().len());
    for (value, count) in value_counts.entries() {
        let Ok(count) = i64::try_from(*count) else {
            return Ok(None);
        };
        rows.push(vec![lit(value.clone()), lit(count)]);
    }
    let aliases: Vec<Expr> = aggregate
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| col(format!("column{}", index + 1)).alias(field.name()))
        .collect();
    Ok(Some(
        LogicalPlanBuilder::values_with_schema(rows, &aggregate.schema)?
            .project(aliases)?
            .alias(scan.table_name.clone())?
            .build()?,
    ))
}

fn count_range_from_value_counts(
    value_counts: &ScalarValueCounts,
    range: &RangeFilter,
) -> Option<i64> {
    let mut total = 0u64;
    for (value, count) in value_counts.entries() {
        if value_matches_range(value, range)? {
            total = total.checked_add(*count)?;
        }
    }
    i64::try_from(total).ok()
}

fn value_matches_range(value: &ScalarValue, range: &RangeFilter) -> Option<bool> {
    if let Some(lo) = &range.lo {
        let bound = lo.value.cast_to(&value.data_type()).ok()?;
        let ordering = value.partial_cmp(&bound)?;
        if ordering == Ordering::Less || (ordering == Ordering::Equal && !lo.inclusive) {
            return Some(false);
        }
    }
    if let Some(hi) = &range.hi {
        let bound = hi.value.cast_to(&value.data_type()).ok()?;
        let ordering = value.partial_cmp(&bound)?;
        if ordering == Ordering::Greater || (ordering == Ordering::Equal && !hi.inclusive) {
            return Some(false);
        }
    }
    Some(true)
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
                let nulls = entry.scalar_stats.get(col)?.null_count?;
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
    let mut acc: Option<ArrayRef> = None;
    for entry in covered {
        let part = entry.scalar_stats.get(col)?.sum.as_ref()?;
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
        let agg = entry.scalar_stats.get(col)?;
        let min = ScalarValue::try_from_array(&agg.min, 0).ok()?;
        let max = ScalarValue::try_from_array(&agg.max, 0).ok()?;
        if min.is_null() || max.is_null() {
            return None;
        }
        acc = match acc {
            None => Some((min, max)),
            Some((cur_min, cur_max)) => {
                let new_min = if min.partial_cmp(&cur_min)? == Ordering::Less {
                    min
                } else {
                    cur_min
                };
                let new_max = if max.partial_cmp(&cur_max)? == Ordering::Greater {
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

/// Peel an unfiltered aggregate input down to its table scan, looking through
/// the same pure-column projection admitted by [`peel_input`].
fn peel_unfiltered_scan(input: &LogicalPlan) -> Option<&TableScan> {
    let mut node = input;
    if let LogicalPlan::Projection(projection) = node {
        if !projection
            .expr
            .iter()
            .all(|expr| matches!(expr, Expr::Column(_)))
        {
            return None;
        }
        node = projection.input.as_ref();
    }
    match node {
        LogicalPlan::TableScan(scan) => Some(scan),
        _ => None,
    }
}

/// The provider behind a scan, when it is ours.
fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    // DataFusion 54 dropped `as_any` for an `Any` supertrait; downcast through
    // its provided `downcast_ref` (auto-derefs the `Arc`).
    let source = scan.source.downcast_ref::<DefaultTableSource>()?;
    source.table_provider.downcast_ref::<SupertableProvider>()
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
        entry.scalar_stats.get(&range.column).and_then(|agg| {
            let mn = ScalarValue::try_from_array(&agg.min, 0).ok()?;
            let mx = ScalarValue::try_from_array(&agg.max, 0).ok()?;
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
            (Some(Ordering::Less), _) => return Class::Disjoint,
            (Some(Ordering::Equal), false) => return Class::Disjoint,
            (None, _) => return Class::Boundary,
            _ => {}
        }
    }
    if let Some(hi) = &range.hi {
        let cmp = seg_min.partial_cmp(&hi.value);
        match (cmp, hi.inclusive) {
            (Some(Ordering::Greater), _) => return Class::Disjoint,
            (Some(Ordering::Equal), false) => return Class::Disjoint,
            (None, _) => return Class::Boundary,
            _ => {}
        }
    }

    // Covered: the whole segment sits inside the range.
    let lo_ok = match &range.lo {
        None => true,
        Some(lo) => match (seg_min.partial_cmp(&lo.value), lo.inclusive) {
            (Some(Ordering::Greater), _) => true,
            (Some(Ordering::Equal), true) => true,
            (None, _) => return Class::Boundary,
            _ => false,
        },
    };
    let hi_ok = match &range.hi {
        None => true,
        Some(hi) => match (seg_max.partial_cmp(&hi.value), hi.inclusive) {
            (Some(Ordering::Less), _) => true,
            (Some(Ordering::Equal), true) => true,
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
        AggKind::Sum(col) => entry.scalar_stats.get(col).is_some_and(|a| a.sum.is_some()),
        AggKind::Min(col) | AggKind::Max(col) => entry.scalar_stats.contains_key(col),
        AggKind::Avg(col) => entry
            .scalar_stats
            .get(col)
            .is_some_and(|a| a.sum.is_some() && a.null_count.is_some()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule opts into the `rewrite` entry point (rather than the
    /// legacy `try_optimize`), and carries its registered name. Neither
    /// the `supports_rewrite` flag nor `name` is observed during a plain
    /// query, so assert them directly.
    /// `classify` buckets a segment against an id-column range: disjoint when
    /// the segment sits wholly outside, covered when wholly inside, boundary
    /// when it straddles an edge or the column has no usable stats.
    #[test]
    fn classify_buckets_segment_against_id_range() {
        use std::collections::HashMap;

        use crate::{
            superfile::vector::layout::VectorLayout, supertable::manifest::SuperfileEntry,
        };
        let entry = SuperfileEntry {
            birth_version: 0,
            superfile_id: uuid::Uuid::new_v4(),
            uri: crate::supertable::manifest::SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 10,
            id_max: 20,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            vector_layout: VectorLayout::Ivf,
            subsection_offsets: None,
        };
        let dec =
            |v: i128| ScalarValue::Decimal128(Some(v), DECIMAL128_PRECISION, DECIMAL128_SCALE);
        let bound = |v: i128, inclusive: bool| Bound {
            value: dec(v),
            inclusive,
        };
        let rf = |lo: Option<Bound>, hi: Option<Bound>| RangeFilter {
            column: "_id".to_string(),
            lo,
            hi,
        };

        assert!(matches!(
            classify(&entry, "_id", &rf(Some(bound(25, true)), None)),
            Class::Disjoint
        ));
        assert!(matches!(
            classify(&entry, "_id", &rf(None, Some(bound(5, true)))),
            Class::Disjoint
        ));
        assert!(matches!(
            classify(
                &entry,
                "_id",
                &rf(Some(bound(0, true)), Some(bound(30, true)))
            ),
            Class::Covered
        ));
        assert!(matches!(
            classify(&entry, "_id", &rf(Some(bound(15, true)), None)),
            Class::Boundary
        ));
        // Unknown column with no scalar stats → boundary (can't prune).
        assert!(matches!(
            classify(
                &entry,
                "_id",
                &RangeFilter {
                    column: "other".to_string(),
                    lo: None,
                    hi: None
                }
            ),
            Class::Boundary
        ));
    }

    /// `typed_zero` maps each supported sum type to its zero and errors on
    /// an unsupported type.
    #[test]
    fn typed_zero_maps_numeric_types_and_rejects_others() {
        assert!(matches!(
            typed_zero(&ScalarValue::Int64(Some(5))),
            Ok(ScalarValue::Int64(Some(0)))
        ));
        assert!(matches!(
            typed_zero(&ScalarValue::UInt64(Some(5))),
            Ok(ScalarValue::UInt64(Some(0)))
        ));
        assert!(matches!(
            typed_zero(&ScalarValue::Float64(Some(5.0))),
            Ok(ScalarValue::Float64(Some(v))) if v == 0.0
        ));
        assert!(typed_zero(&ScalarValue::Utf8(Some("x".into()))).is_err());
    }

    /// `avg_column` reads the column name directly and through a wrapping
    /// cast; a non-column argument yields `None`.
    #[test]
    fn avg_column_reads_through_cast_and_rejects_non_column() {
        use arrow_schema::DataType;
        use datafusion::{
            logical_expr::{Cast, Expr},
            prelude::{col, lit},
        };

        assert_eq!(avg_column(&col("price")), Some("price".to_string()));
        let casted = Expr::Cast(Cast::new(Box::new(col("price")), DataType::Float64));
        assert_eq!(avg_column(&casted), Some("price".to_string()));
        assert_eq!(avg_column(&lit(1.0)), None);
    }

    #[test]
    #[allow(deprecated)] // `supports_rewrite` is the targeted method.
    fn rule_opts_into_rewrite_and_reports_name() {
        let rule = CoveredAggregateRewrite;
        assert!(rule.supports_rewrite(), "rule must use the rewrite path");
        assert_eq!(rule.name(), "covered_aggregate_rewrite");
    }
}
