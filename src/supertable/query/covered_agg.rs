// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Covered/residual evaluation for filter-aligned aggregates over the
//! manifest-statistics "aggregation tree".
//!
//! Exact low-cardinality frequencies answer `COUNT(*)` over a
//! single-column equality/range and `GROUP BY column, COUNT(*)` directly,
//! unfiltered or filtered to the column's non-null values.
//! `COUNT(*)` over a range on a range-indexed column is counted from each
//! superfile's range index, by a scan that emits the count as rows with no
//! columns (`SupertableProvider::counting`); so is each bucket of a
//! bucketed range aggregation written as `COUNT(*) FILTER (WHERE range)`
//! outputs, every bucket's count in one row of one scan (`RangeCounts`).
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
//!     boundary — one superfile's deletes send that superfile to the scan,
//!     not the whole table;
//!   * a segment whose range column holds any (or an unknown number of)
//!     nulls is boundary: a NULL never matches the range, but the
//!     segment's statistics count its row;
//!   * a covered segment missing a required stat demotes to boundary;
//!   * only `COUNT(*)`, `COUNT`, `SUM`, `MIN`, `MAX`, `AVG` over bare
//!     columns, no DISTINCT / FILTER / ORDER BY; the sole grouped shape is
//!     one low-cardinality column plus `COUNT(*)`, over the whole table or
//!     filtered by `column IS NOT NULL` alone;
//!   * a provider already restricted to a segment subset is the
//!     rewrite's own residual — never rewritten again (idempotency);
//!   * a scan carrying a filter the provider answers exactly (which
//!     leaves no `Filter` node to read) is never rewritten.
//!
//! The grouped count answers its covered segments from their value counts
//! and groups the boundary ones by a scan, adding the two sets of groups up
//! by value. A residual scan says in `EXPLAIN` how many segments statistics
//! answered and why it reads the others; a `debug` event says the same, and
//! why an aggregate is left to the scan entirely.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use arrow_array::ArrayRef;
use arrow_schema::DataType;
use datafusion::{
    catalog::TableProvider,
    common::{
        DFSchema, ScalarValue,
        tree_node::{Transformed, TreeNodeRecursion},
    },
    datasource::{DefaultTableSource, provider_as_source},
    error::{DataFusionError, Result as DfResult},
    functions::core::expr_fn::{coalesce, greatest, least},
    functions_aggregate::expr_fn::{count, max, min, sum},
    logical_expr::{
        Aggregate, Expr, Filter, LogicalPlan, LogicalPlanBuilder, Operator, TableScan, TableSource,
        lit, utils::split_conjunction,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
    prelude::{cast, col, ident},
};
use futures::future::try_join_all;
use tracing::debug;

use crate::supertable::{
    error::QueryError,
    manifest::{
        ManifestSnapshot, SuperfileEntry, add_sum_arrays, list::ScalarValueCounts,
        load_section_groups_blocking,
    },
    query::{
        provider::{CountedRange, SupertableProvider, TombstoneState},
        range_counts::RangeBucket,
        skip::{ColumnTypeGuard, file_lacks_column},
        value_count_carry::{counts_of, counts_without},
    },
    schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE, FieldId},
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
        // `try_rewrite` declines with `None`. An error is a failure building
        // our replacement from the manifest's own statistics: ours, never the
        // query's, so it crosses DataFusion as an engine fault.
        match try_rewrite(&plan).map_err(QueryError::internal)? {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// One supported aggregate output.
#[derive(Debug, PartialEq)]
enum AggKind {
    CountStar,
    /// `COUNT(column)`: the column's non-null rows.
    Count(String),
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

/// Load every manifest part of each table an unfiltered, ungrouped aggregate
/// in `plan` reads, subqueries included, so the rewrite below finds the
/// whole table in memory and answers from statistics. That aggregate's scan
/// would load every part anyway, so loading them before optimization adds no
/// reads.
pub(crate) async fn load_unfiltered_aggregate_inputs(plan: &LogicalPlan) -> Result<(), QueryError> {
    let mut sources: Vec<Arc<dyn TableSource>> = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::Aggregate(agg) = node
            && agg.group_expr.is_empty()
            && let Some(scan) = peel_unfiltered_scan(agg.input.as_ref())
            && provider_of(scan).is_some_and(|provider| !provider.is_segment_restricted())
        {
            sources.push(Arc::clone(&scan.source));
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(QueryError::DataFusion)?;
    try_join_all(sources.iter().filter_map(|source| {
        let source = source.downcast_ref::<DefaultTableSource>()?;
        let provider = source.table_provider.downcast_ref::<SupertableProvider>()?;
        Some(provider.load_every_part())
    }))
    .await?;
    Ok(())
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
    if let Some(rewritten) = rewrite_bucketed_range_counts(agg)? {
        return Ok(Some(rewritten));
    }
    let Some(kinds) = parse_aggregates(&agg.aggr_expr) else {
        return Ok(None);
    };

    // Unfiltered aggregate over a complete snapshot: a clean segment is
    // covered by its statistics. When every segment is clean, emit one
    // literal row directly. This handles SUM/AVG as well as COUNT/MIN/MAX
    // and avoids invoking the provider's Parquet scan at all. Otherwise only
    // the segments with deletes (or an unknown tombstone state) or a
    // missing stat are scanned.
    if let Some(scan) = peel_unfiltered_scan(agg.input.as_ref()) {
        let Some(provider) = provider_of(scan) else {
            return Ok(None);
        };
        if provider.is_segment_restricted() {
            return Ok(None);
        }
        let manifest = provider.manifest();
        let Some(superfiles) = manifest.resident_superfiles() else {
            return Ok(None);
        };
        let mut split = Split::new("unfiltered aggregate");
        for entry in superfiles.iter() {
            match scan_reason(provider, entry, || {
                has_required_stats(manifest, entry, &kinds)
            }) {
                None => split.answer(entry),
                Some(reason) => split.scan(reason, entry),
            }
        }
        if split.scans_any() && split.answered.is_empty() {
            split.decline("no superfile is answerable from statistics");
            return Ok(None);
        }
        let Some(partials) = accumulate_partials(manifest, &kinds, &split.answered) else {
            split.decline("a statistic does not fold");
            return Ok(None);
        };
        if split.scans_any() {
            return rewrite_with_residual(agg, scan, provider, &kinds, partials, split, None);
        }
        split.log();
        let mut expressions = Vec::with_capacity(kinds.len());
        for (i, (kind, partial)) in kinds.iter().zip(&partials).enumerate() {
            let field = agg.schema.field(i);
            let expression = match (kind, partial) {
                (AggKind::CountStar | AggKind::Count(_), Partial::Count(count)) => lit(*count),
                (AggKind::Sum(_), Partial::Sum(value))
                | (AggKind::Min(_), Partial::Bound(value))
                | (AggKind::Max(_), Partial::Bound(value)) => {
                    covered_literal(value, field.data_type())
                }
                (AggKind::Avg(_), Partial::Avg { sum, count }) => {
                    cast(lit(sum.clone()), DataType::Float64) / cast(lit(*count), DataType::Float64)
                }
                _ => return Ok(None),
            };
            expressions.push(expression.alias(field.name().clone()));
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
    // Hierarchical manifests may expose a partial flat view. Answer from
    // statistics only when the resident entries are provably complete;
    // eagerly hydrated persisted tables then receive the same no-scan
    // aggregate path as in-process tables, while genuinely lazy views
    // decline safely. The range count below scans, so it needs no complete
    // view.
    let superfiles = manifest.complete_flat_superfiles();
    let id_column = manifest.options.id_column.as_str();

    if matches!(kinds.as_slice(), [AggKind::CountStar])
        && let Some(counts_split) =
            split_value_counts(provider, &range.column, false, "value count")
    {
        // A range index counts every superfile, deleted rows left out, from
        // a block or two each: cheaper than scanning the ones with deletes,
        // so with any to scan it answers below.
        let scans = counts_split.split.scans_any();
        if (!scans || provider.counted_range(&predicate).is_none())
            && let Some(counts) = answered_counts(provider, &range.column, &counts_split)
            && let Some(count) = count_range_from_value_counts(counts.iter(), &range)
        {
            let split = counts_split.split;
            if !scans {
                split.log();
                let name = agg.schema.field(0).name().clone();
                return Ok(Some(
                    LogicalPlanBuilder::empty(true)
                        .project(vec![lit(count).alias(name)])?
                        .build()?,
                ));
            }
            return rewrite_with_residual(
                agg,
                scan,
                provider,
                &kinds,
                vec![Partial::Count(count)],
                split,
                Some(&predicate),
            );
        }
    }

    // A range-indexed column counts the range from each superfile's index:
    // the count reads a block or two per superfile, not the rows.
    if matches!(kinds.as_slice(), [AggKind::CountStar])
        && let Some(counted) = provider.counted_range(&predicate)
    {
        let input = counting_scan(provider, scan, counted, &predicate)?;
        return Ok(Some(LogicalPlan::Aggregate(
            Aggregate::try_new_with_schema(
                Arc::new(input),
                Vec::new(),
                agg.aggr_expr.clone(),
                Arc::clone(&agg.schema),
            )?,
        )));
    }

    let Some(superfiles) = superfiles else {
        return Ok(None);
    };
    // Classify every segment.
    let mut split = Split::new("range aggregate");
    for entry in superfiles {
        match classify(manifest, entry, id_column, &range) {
            Class::Disjoint => split.skip(),
            Class::Covered => {
                match scan_reason(provider, entry, || {
                    has_required_stats(manifest, entry, &kinds)
                }) {
                    None => split.answer(entry),
                    Some(reason) => split.scan(reason, entry),
                }
            }
            Class::Boundary => split.scan(ScanReason::Boundary, entry),
        }
    }
    if split.answered.is_empty() {
        // Nothing answerable from statistics — the rewrite would only
        // add plan noise over the normal scan.
        split.decline("no superfile is answerable from statistics");
        return Ok(None);
    }
    let Some(partials) = accumulate_partials(manifest, &kinds, &split.answered) else {
        // A fold failure (overflow, type surprise) is always safe to
        // decline.
        split.decline("a statistic does not fold");
        return Ok(None);
    };
    rewrite_with_residual(
        agg,
        scan,
        provider,
        &kinds,
        partials,
        split,
        Some(&predicate),
    )
}

/// Why `entry` must be scanned rather than answered from its statistics,
/// whose other requirements `has_stats` checks; `None` when its statistics
/// answer for it.
fn scan_reason(
    provider: &SupertableProvider,
    entry: &SuperfileEntry,
    has_stats: impl FnOnce() -> bool,
) -> Option<ScanReason> {
    match provider.tombstone_state(entry) {
        TombstoneState::Deleted => Some(ScanReason::DeletedRows),
        TombstoneState::Unknown => Some(ScanReason::DeletesUnknown),
        TombstoneState::Clean => (!has_stats()).then_some(ScanReason::MissingStatistics),
    }
}

/// Each aggregate's contribution from the `covered` superfiles' statistics;
/// `None` when one does not fold (overflow, a type surprise).
fn accumulate_partials(
    manifest: &ManifestSnapshot,
    kinds: &[AggKind],
    covered: &[Arc<SuperfileEntry>],
) -> Option<Vec<Partial>> {
    kinds
        .iter()
        .map(|kind| accumulate_partial(manifest, kind, covered))
        .collect()
}

/// Why one superfile is scanned rather than answered from its statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ScanReason {
    /// The range cuts across its bounds, or its range column holds nulls.
    Boundary,
    /// It has deleted rows, which its statistics still count.
    DeletedRows,
    /// Its deleted rows could not be read.
    DeletesUnknown,
    /// It lacks a statistic the aggregate needs, or recorded it in a type
    /// the column no longer has.
    MissingStatistics,
}

impl ScanReason {
    fn describe(self) -> &'static str {
        match self {
            Self::Boundary => "cut by the range",
            Self::DeletedRows => "with deleted rows",
            Self::DeletesUnknown => "with unreadable deletes",
            Self::MissingStatistics => "missing statistics",
        }
    }
}

/// Which superfiles one aggregate answers from statistics and which it
/// scans, and why: the residual scan's note in `EXPLAIN`, and a `debug`
/// event, so a slow aggregate says what sent it to the data.
struct Split {
    /// The aggregate's shape, for the event.
    shape: &'static str,
    /// Superfiles answered from statistics.
    answered: Vec<Arc<SuperfileEntry>>,
    /// Superfiles that hold no row the aggregate reads.
    skipped: usize,
    /// Superfiles scanned.
    scanned: Vec<Arc<SuperfileEntry>>,
    /// How many are scanned for each reason.
    reasons: BTreeMap<ScanReason, usize>,
}

impl Split {
    fn new(shape: &'static str) -> Self {
        Self {
            shape,
            answered: Vec::new(),
            skipped: 0,
            scanned: Vec::new(),
            reasons: BTreeMap::new(),
        }
    }

    fn answer(&mut self, entry: &Arc<SuperfileEntry>) {
        self.answered.push(Arc::clone(entry));
    }

    fn skip(&mut self) {
        self.skipped += 1;
    }

    fn scan(&mut self, reason: ScanReason, entry: &Arc<SuperfileEntry>) {
        *self.reasons.entry(reason).or_default() += 1;
        self.scanned.push(Arc::clone(entry));
    }

    fn scans_any(&self) -> bool {
        !self.scanned.is_empty()
    }

    /// `provider` restricted to the superfiles scanned, its scan noted with
    /// what it stands for.
    fn restrict(&self, provider: &SupertableProvider) -> SupertableProvider {
        let scanned = self
            .scanned
            .iter()
            .map(|entry| entry.superfile_id)
            .collect();
        provider.restricted_to(scanned).with_scan_note(self.note())
    }

    /// What the residual scan stands for, as `EXPLAIN` shows it: e.g.
    /// "scans 2 superfiles (2 with deleted rows); 14 answered from
    /// statistics".
    fn note(&self) -> String {
        let reasons: Vec<String> = self
            .reasons
            .iter()
            .map(|(reason, n)| format!("{n} {}", reason.describe()))
            .collect();
        let mut note = format!(
            "scans {} superfiles ({}); {} answered from statistics",
            self.scanned.len(),
            reasons.join(", "),
            self.answered.len()
        );
        if self.skipped > 0 {
            note.push_str(&format!(", {} outside the range", self.skipped));
        }
        note
    }

    /// Record the split taken.
    fn log(&self) {
        debug!(
            shape = self.shape,
            answered = self.answered.len(),
            skipped = self.skipped,
            scanned = ?self.reasons,
            "aggregate answered from statistics"
        );
    }

    /// Record why the aggregate is left to the scan.
    fn decline(&self, why: &str) {
        debug!(
            shape = self.shape,
            answered = self.answered.len(),
            skipped = self.skipped,
            scanned = ?self.reasons,
            why,
            "aggregate not answered from statistics"
        );
    }
}

/// Answer `agg` from `partials`, each aggregate's contribution from
/// statistics, plus a scan of the superfiles `split` scans, filtered by
/// `predicate` when the query has one.
fn rewrite_with_residual(
    agg: &Aggregate,
    scan: &TableScan,
    provider: &SupertableProvider,
    kinds: &[AggKind],
    partials: Vec<Partial>,
    split: Split,
    predicate: Option<&Expr>,
) -> DfResult<Option<LogicalPlan>> {
    split.log();
    // Residual: the original predicate over the boundary-restricted
    // provider. An empty boundary set still scans (an empty relation),
    // and the global Aggregate below then yields its empty-input row —
    // exactly what the combiners expect.
    let restricted = Arc::new(split.restrict(provider));
    let mut builder = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        provider_as_source(restricted as Arc<dyn TableProvider>),
        None,
    )?;
    if let Some(predicate) = predicate {
        builder = builder.filter(predicate.clone())?;
    }

    // Partial aggregates over the residual, one batch per output (AVG
    // needs two: sum + non-null count). Each column is named exactly, as
    // the query named it: `col` would parse the name as SQL, folding
    // `AvgTicketPrice` to `avgticketprice` and reading `geo.bytes` as
    // column `bytes` of a table `geo`.
    let mut partial_exprs: Vec<Expr> = Vec::new();
    for (i, kind) in kinds.iter().enumerate() {
        match kind {
            AggKind::CountStar => {
                partial_exprs.push(count(lit(1i64)).alias(format!("__resid_{i}_cnt")));
            }
            AggKind::Count(column) => {
                partial_exprs.push(count(ident(column)).alias(format!("__resid_{i}_cnt")));
            }
            AggKind::Sum(column) => {
                partial_exprs.push(sum(ident(column)).alias(format!("__resid_{i}_sum")));
            }
            AggKind::Min(column) => {
                partial_exprs.push(min(ident(column)).alias(format!("__resid_{i}_min")));
            }
            AggKind::Max(column) => {
                partial_exprs.push(max(ident(column)).alias(format!("__resid_{i}_max")));
            }
            AggKind::Avg(column) => {
                // Explicitly cast: the analyzer's coercion pass has already
                // run, so a bare `sum(col)` over a numeric narrower than
                // the sum accumulator (Int32, Float32, ...) is untypeable
                // here. Float64 matches the cast the planner had put on
                // the original AVG argument, which parse_aggregates peeled.
                partial_exprs.push(
                    sum(cast(ident(column), DataType::Float64)).alias(format!("__resid_{i}_sum")),
                );
                partial_exprs.push(count(ident(column)).alias(format!("__resid_{i}_cnt")));
            }
        }
    }
    builder = builder.aggregate(Vec::<Expr>::new(), partial_exprs)?;

    // Final projection: combine each residual partial with its covered
    // literal, aliased to the original aggregate's output name so the
    // rewritten plan's schema matches the original node's exactly.
    let mut final_exprs: Vec<Expr> = Vec::with_capacity(kinds.len());
    for (i, (kind, partial)) in kinds.iter().zip(&partials).enumerate() {
        let field = agg.schema.field(i);
        let name = field.name().clone();
        let expr = match (kind, partial) {
            (AggKind::CountStar | AggKind::Count(_), Partial::Count(n)) => {
                // COUNT over an empty residual is 0, never NULL.
                (col(format!("__resid_{i}_cnt")) + lit(*n)).alias(name)
            }
            (AggKind::Sum(_), Partial::Sum(value)) => {
                let zero = typed_zero(value)?;
                (coalesce(vec![col(format!("__resid_{i}_sum")), lit(zero)]) + lit(value.clone()))
                    .alias(name)
            }
            (AggKind::Min(_), Partial::Bound(value)) => least(vec![
                col(format!("__resid_{i}_min")),
                covered_literal(value, field.data_type()),
            ])
            .alias(name),
            (AggKind::Max(_), Partial::Bound(value)) => greatest(vec![
                col(format!("__resid_{i}_max")),
                covered_literal(value, field.data_type()),
            ])
            .alias(name),
            (AggKind::Avg(_), Partial::Avg { sum, count }) => {
                // The residual partial is Float64 by construction (its sum
                // is over an explicit cast), so the whole combiner stays in
                // Float64 — the coalesce zero and the covered sum must
                // match it, since nothing coerces after this rule.
                let total_sum = coalesce(vec![col(format!("__resid_{i}_sum")), lit(0.0_f64)])
                    + cast(lit(sum.clone()), DataType::Float64);
                let total_cnt = col(format!("__resid_{i}_cnt")) + lit(*count);
                (total_sum / cast(total_cnt, DataType::Float64)).alias(name)
            }
            _ => return Ok(None),
        };
        final_exprs.push(expr);
    }

    let rewritten = builder.project(final_exprs)?.build()?;
    Ok(Some(rewritten))
}

/// The covered statistic as a literal, cast to the aggregate's own output
/// type when the stored value's type differs. Optimizer rules run after
/// the analyzer's type-coercion pass, so nothing downstream repairs a
/// mismatch — and one exists for every non-FTS string column: the scan
/// serves it as `Utf8View` while the manifest stores its bounds as
/// `LargeUtf8`. Uncast, the bound fails arrow's comparison kernel at
/// execution (filtered arm) or breaks the rewritten plan's schema
/// (unfiltered arm).
fn covered_literal(value: &ScalarValue, output: &DataType) -> Expr {
    if &value.data_type() == output {
        lit(value.clone())
    } else {
        cast(lit(value.clone()), output.clone())
    }
}

/// `GROUP BY column, COUNT(*)`, over the whole table or filtered to the
/// column's non-null values, from the superfiles' exact value counts. A
/// superfile whose statistics do not describe its live rows (it has deleted
/// rows, or does not record its nulls where the null group is counted) is
/// scanned and grouped on its own; the two sets of groups are added up by
/// value:
///
/// ```text
/// Projection(the aggregate's names)
///   └─ Aggregate(group by value, SUM(count))
///        └─ Union
///             ├─ Aggregate(group by column, COUNT(*))
///             │    └─ [Filter(column IS NOT NULL)]
///             │         └─ TableScan(the scanned superfiles)
///             └─ Values(value, count)               -- from statistics
/// ```
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
    // The value counts are exact counts of the column's non-null values.
    // Over the whole table the null group is counted from the null counts
    // as well; filtered to the non-null values, there is none.
    let (scan, nulls_filtered) = match peel_unfiltered_scan(aggregate.input.as_ref()) {
        Some(scan) => (scan, false),
        None => match peel_not_null_filter(aggregate.input.as_ref(), &group_column.name) {
            Some(scan) => (scan, true),
            None => return Ok(None),
        },
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.is_segment_restricted() {
        return Ok(None);
    }
    let column = group_column.name.as_str();
    let Some(counts_split) = split_value_counts(provider, column, !nulls_filtered, "grouped count")
    else {
        return Ok(None);
    };
    let Some(counts) = answered_counts(provider, column, &counts_split) else {
        counts_split
            .split
            .decline("the column has no exact value counts");
        return Ok(None);
    };
    let CountsSplit { split, nulls, .. } = counts_split;

    let group_type = aggregate.schema.field(0).data_type().clone();
    let mut rows = Vec::with_capacity(counts.len() + 1);
    for (value, count) in counts.iter() {
        let Ok(count) = i64::try_from(count) else {
            split.decline("a count overflows");
            return Ok(None);
        };
        let Ok(value) = value.cast_to(&group_type) else {
            split.decline("a value does not cast to the column's type");
            return Ok(None);
        };
        rows.push(vec![lit(value), lit(count)]);
    }
    if !nulls_filtered && nulls > 0 {
        let (Ok(nulls), Ok(null)) = (i64::try_from(nulls), ScalarValue::try_from(&group_type))
        else {
            split.decline("the null group does not fit");
            return Ok(None);
        };
        rows.push(vec![lit(null), lit(nulls)]);
    }

    if !split.scans_any() {
        // A column with no value at all has no group to list, and a values
        // list takes at least one row; the scan answers the empty grouping.
        if rows.is_empty() {
            split.decline("the column holds no value");
            return Ok(None);
        }
        split.log();
        let aliases: Vec<Expr> = aggregate
            .schema
            .fields()
            .iter()
            .enumerate()
            .map(|(index, field)| col(format!("column{}", index + 1)).alias(field.name()))
            .collect();
        return Ok(Some(
            LogicalPlanBuilder::values_with_schema(rows, &aggregate.schema)?
                .project(aliases)?
                .alias(scan.table_name.clone())?
                .build()?,
        ));
    }

    split.log();
    let restricted = Arc::new(split.restrict(provider));
    let mut scanned = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        provider_as_source(restricted as Arc<dyn TableProvider>),
        None,
    )?;
    if nulls_filtered {
        scanned = scanned.filter(Expr::Column(group_column.clone()).is_not_null())?;
    }
    let scanned = scanned
        .aggregate(
            vec![Expr::Column(group_column.clone()).alias(GROUP_VALUE)],
            vec![count(lit(1i64)).alias(GROUP_COUNT)],
        )?
        .build()?;
    // The groups from statistics take the scanned groups' schema, so the
    // union, and the sum over it, keep the grouping column's nullability.
    let groups = if rows.is_empty() {
        LogicalPlanBuilder::from(scanned)
    } else {
        let schema = Arc::new(DFSchema::from_unqualified_fields(
            scanned
                .schema()
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect::<Vec<_>>()
                .into(),
            HashMap::new(),
        )?);
        // A union is named by its first input.
        let answered = LogicalPlanBuilder::values_with_schema(rows, &schema)?.build()?;
        LogicalPlanBuilder::from(scanned).union(answered)?
    };
    // A group's sum is never null: a group has at least one row. The
    // coalesce says so, keeping the count as non-null as `COUNT(*)`.
    let summed = groups.aggregate(
        vec![col(GROUP_VALUE)],
        vec![sum(col(GROUP_COUNT)).alias(GROUP_COUNT)],
    )?;
    Ok(Some(
        summed
            .project(as_output(
                &aggregate.schema,
                vec![
                    col(GROUP_VALUE),
                    coalesce(vec![col(GROUP_COUNT), lit(0i64)]),
                ],
            ))?
            .build()?,
    ))
}

/// The grouped count's value and count, between its own nodes.
const GROUP_VALUE: &str = "__group_value";
const GROUP_COUNT: &str = "__group_count";

/// `exprs`, one per field of `schema`, each named as that field and under its
/// qualifier, so the plan they top has exactly `schema`.
fn as_output(schema: &DFSchema, exprs: Vec<Expr>) -> Vec<Expr> {
    exprs
        .into_iter()
        .zip(schema.iter())
        .map(|(expr, (qualifier, field))| expr.alias_qualified(qualifier.cloned(), field.name()))
        .collect()
}

/// The rows of column `id` in `entry` that hold no value: all of them when
/// the superfile does not hold the column, else its recorded null count.
/// `None` when it records none.
fn file_null_count(entry: &SuperfileEntry, id: FieldId) -> Option<u64> {
    match file_lacks_column(entry, id) {
        true => Some(entry.n_docs),
        false => entry
            .scalar_stats
            .column(id)
            .and_then(|stats| stats.null_count()),
    }
}

/// How a column's value counts answer one statement: the superfiles whose
/// statistics describe their live rows, and the rows among them holding no
/// value; the rest are scanned.
struct CountsSplit {
    id: FieldId,
    split: Split,
    /// The answered superfiles' null rows; 0 unless they were counted.
    nulls: u64,
}

/// Split the superfiles for counting `column` by value from statistics: one
/// with deleted rows, or (with `count_nulls`) without a recorded null count,
/// is scanned. `None`, logged, when the counts can't answer at all: a
/// segment-restricted provider (the rewrite's own residual), a manifest not
/// fully loaded, a column mid-retype, or no superfile answered.
fn split_value_counts(
    provider: &SupertableProvider,
    column: &str,
    count_nulls: bool,
    shape: &'static str,
) -> Option<CountsSplit> {
    let mut split = Split::new(shape);
    if provider.is_segment_restricted() {
        return None;
    }
    let manifest = provider.manifest();
    let Some(superfiles) = manifest.complete_flat_superfiles() else {
        split.decline("the manifest is not fully loaded");
        return None;
    };
    if !counts_describe_column(manifest, column, superfiles) {
        split.decline("the column's statistics are of another type");
        return None;
    }
    let id = manifest.field_id(column)?;
    // The null counts and value counts are read in place; their groups are
    // read and decoded off the runtime's queue first, as a scan's are.
    load_section_groups_blocking(superfiles, &[id]);
    let mut nulls = 0u64;
    for entry in superfiles {
        let file_nulls = file_null_count(entry, id);
        match scan_reason(provider, entry, || !count_nulls || file_nulls.is_some()) {
            None => {
                split.answer(entry);
                if count_nulls {
                    let Some(total) = nulls.checked_add(file_nulls.unwrap_or(0)) else {
                        split.decline("the null count overflows");
                        return None;
                    };
                    nulls = total;
                }
            }
            Some(reason) => split.scan(reason, entry),
        }
    }
    if split.answered.is_empty() && split.scans_any() {
        split.decline("no superfile is answerable from statistics");
        return None;
    }
    Some(CountsSplit { id, split, nulls })
}

/// A column's value counts over the superfiles a [`CountsSplit`] answers.
enum Counts {
    /// Every superfile's: the table's own counts.
    Table(Arc<ScalarValueCounts>),
    /// Some of them.
    Answered(HashMap<ScalarValue, u64>),
}

impl Counts {
    fn len(&self) -> usize {
        match self {
            Self::Table(counts) => counts.entries().len(),
            Self::Answered(counts) => counts.len(),
        }
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (&ScalarValue, u64)> + '_> {
        match self {
            Self::Table(counts) => Box::new(counts.entries().iter().map(|(v, n)| (v, *n))),
            Self::Answered(counts) => Box::new(counts.iter().map(|(v, n)| (v, *n))),
        }
    }
}

/// `column`'s value counts over the superfiles `counts_split` answers: the
/// table's carried counts less the scanned superfiles' own, or, where those
/// can't be taken off (a scanned superfile holding more distinct values
/// than it counts), the answered superfiles' counts added up. `None` when
/// the answered superfiles' values are not all counted.
fn answered_counts(
    provider: &SupertableProvider,
    column: &str,
    counts_split: &CountsSplit,
) -> Option<Counts> {
    let total = provider.exact_value_counts(column);
    let scanned = &counts_split.split.scanned;
    if scanned.is_empty() {
        return total.map(Counts::Table);
    }
    total
        .and_then(|total| counts_without(&total, scanned, counts_split.id))
        .or_else(|| counts_of(&counts_split.split.answered, counts_split.id))
        .map(Counts::Answered)
}

/// `column`'s live rows by value, and the rows holding no value, when the
/// manifest's statistics describe every superfile's live rows (see
/// [`split_value_counts`]); `None` otherwise.
pub(crate) fn value_counts_from_statistics(
    provider: &SupertableProvider,
    column: &str,
) -> Option<(Vec<(ScalarValue, u64)>, u64)> {
    let counts_split = split_value_counts(provider, column, true, "value_counts")?;
    if counts_split.split.scans_any() {
        return None;
    }
    let counts = answered_counts(provider, column, &counts_split)?;
    let values = counts.iter().map(|(v, n)| (v.clone(), n)).collect();
    Some((values, counts_split.nulls))
}

fn count_range_from_value_counts<'a>(
    value_counts: impl Iterator<Item = (&'a ScalarValue, u64)>,
    range: &RangeFilter,
) -> Option<i64> {
    let mut total = 0u64;
    for (value, count) in value_counts {
        if value_matches_range(value, range)? {
            total = total.checked_add(count)?;
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
fn accumulate_partial(
    manifest: &ManifestSnapshot,
    kind: &AggKind,
    covered: &[Arc<SuperfileEntry>],
) -> Option<Partial> {
    match kind {
        AggKind::CountStar => {
            let mut total: i64 = 0;
            for entry in covered {
                total = total.checked_add(i64::try_from(entry.n_docs).ok()?)?;
            }
            Some(Partial::Count(total))
        }
        AggKind::Count(col) => Some(Partial::Count(non_null_count(
            covered,
            manifest.field_id(col)?,
        )?)),
        AggKind::Sum(col) => Some(Partial::Sum(fold_sums(covered, manifest.field_id(col)?)?)),
        AggKind::Min(col) => {
            let (min, _) = fold_bounds(covered, manifest.field_id(col)?)?;
            Some(Partial::Bound(min))
        }
        AggKind::Max(col) => {
            let (_, max) = fold_bounds(covered, manifest.field_id(col)?)?;
            Some(Partial::Bound(max))
        }
        AggKind::Avg(col) => {
            let col = manifest.field_id(col)?;
            let sum = fold_sums(covered, col)?;
            let count = non_null_count(covered, col)?;
            if count == 0 {
                // All covered values NULL: AVG semantics get fiddly
                // (0/0); decline and let the scan answer.
                return None;
            }
            Some(Partial::Avg { sum, count })
        }
    }
}

/// The non-null rows of `col` across `covered`: each superfile's row count
/// less its recorded null count, and none from a superfile that does not
/// hold the column. `None` when a superfile holding it lacks the null count.
fn non_null_count(covered: &[Arc<SuperfileEntry>], col: FieldId) -> Option<i64> {
    let mut count: i64 = 0;
    for entry in covered {
        if file_lacks_column(entry, col) {
            continue;
        }
        let nulls = entry.scalar_stats.column(col)?.null_count()?;
        let non_null = entry.n_docs.checked_sub(nulls)?;
        count = count.checked_add(i64::try_from(non_null).ok()?)?;
    }
    Some(count)
}

fn fold_sums(covered: &[Arc<SuperfileEntry>], col: FieldId) -> Option<ScalarValue> {
    let mut acc: Option<ArrayRef> = None;
    for entry in covered {
        let part = entry.scalar_stats.column(col)?.sum()?;
        acc = Some(match acc {
            None => part,
            Some(total) => add_sum_arrays(&total, &part)?,
        });
    }
    ScalarValue::try_from_array(&acc?, 0).ok()
}

fn fold_bounds(
    covered: &[Arc<SuperfileEntry>],
    col: FieldId,
) -> Option<(ScalarValue, ScalarValue)> {
    let mut acc: Option<(ScalarValue, ScalarValue)> = None;
    for entry in covered {
        let (min, max) = entry.scalar_stats.column(col)?.min_max_values()?;
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

/// The scan under a range count: `predicate`'s rows, counted from each
/// superfile's range index and emitted as rows with no columns.
fn counting_scan(
    provider: &SupertableProvider,
    scan: &TableScan,
    counted: CountedRange,
    predicate: &Expr,
) -> DfResult<LogicalPlan> {
    let counting = Arc::new(provider.counting(counted));
    let filters = split_conjunction(predicate).into_iter().cloned().collect();
    LogicalPlanBuilder::scan_with_filters(
        scan.table_name.clone(),
        provider_as_source(counting as Arc<dyn TableProvider>),
        Some(Vec::new()),
        filters,
    )?
    .build()
}

/// A bucketed range aggregation in one statement — `COUNT(*) FILTER
/// (WHERE r)` for several ranges `r`, ungrouped — counted from the range
/// index. A bucket's rows are those of the `WHERE` and its own filter
/// together, so each is one range count over that conjunction; one
/// [`RangeCounts`](super::range_counts::RangeCounts) scan counts them all,
/// as one row, under the aggregate's own names. Buckets may overlap, as a
/// range aggregation's may: each is counted on its own. An unfiltered
/// `COUNT(*)` beside them with no `WHERE` (the request's total) is the
/// table's live rows, counted in the same scan.
///
/// Declines (the scan answers) unless every output is a `COUNT(*)` and at
/// least one carries a filter, and every bucket's rows are one interval on
/// one range-indexed column.
fn rewrite_bucketed_range_counts(agg: &Aggregate) -> DfResult<Option<LogicalPlan>> {
    let Some(buckets) = bucket_counts(&agg.aggr_expr) else {
        return Ok(None);
    };
    if buckets.iter().all(|(_, filter)| filter.is_none()) {
        return Ok(None);
    }
    let (outer, scan) = match peel_input(agg.input.as_ref()) {
        Some((predicate, scan)) => (Some(predicate), scan),
        None => match peel_unfiltered_scan(agg.input.as_ref()) {
            Some(scan) => (None, scan),
            None => return Ok(None),
        },
    };
    let Some(provider) = provider_of(scan) else {
        return Ok(None);
    };
    if provider.is_segment_restricted() {
        return Ok(None);
    }

    let mut ranges = Vec::with_capacity(buckets.len());
    for (i, (_, filter)) in buckets.into_iter().enumerate() {
        let predicate = match (outer.clone(), filter) {
            (Some(outer), Some(filter)) => outer.and(filter),
            (Some(predicate), None) | (None, Some(predicate)) => predicate,
            (None, None) => {
                ranges.push(RangeBucket {
                    name: bucket_name(i),
                    range: None,
                    filters: Vec::new(),
                });
                continue;
            }
        };
        let Some(range) = provider.counted_range(&predicate) else {
            return Ok(None);
        };
        ranges.push(RangeBucket {
            name: bucket_name(i),
            range: Some(range),
            filters: split_conjunction(&predicate).into_iter().cloned().collect(),
        });
    }

    let counts = Arc::new(provider.range_counts(ranges));
    let joined = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        provider_as_source(counts as Arc<dyn TableProvider>),
        None,
    )?;
    let names = agg
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, field)| ident(bucket_name(i)).alias(field.name().clone()));
    Ok(Some(joined.project(names)?.build()?))
}

/// The name a bucket's count carries until the bucketed rewrite's final
/// projection gives it the aggregate's own.
fn bucket_name(bucket: usize) -> String {
    format!("__range_bucket_{bucket}")
}

/// Each output as a `COUNT(*)` without its filter, beside the filter;
/// `None` if any output is not a `COUNT(*)` or is DISTINCT or ordered.
fn bucket_counts(exprs: &[Expr]) -> Option<Vec<(Expr, Option<Expr>)>> {
    exprs
        .iter()
        .map(|expr| {
            let inner = match expr {
                Expr::Alias(alias) => alias.expr.as_ref(),
                other => other,
            };
            let Expr::AggregateFunction(agg) = inner else {
                return None;
            };
            let params = &agg.params;
            if params.distinct
                || !params.order_by.is_empty()
                || !counts_rows(agg.func.name(), &params.args)
            {
                return None;
            }
            let mut unfiltered = agg.clone();
            let filter = unfiltered.params.filter.take().map(|filter| *filter);
            Some((Expr::AggregateFunction(unfiltered), filter))
        })
        .collect()
}

/// Whether an aggregate named `name` over `args` counts rows: `COUNT(*)`,
/// or `COUNT` of a non-null literal.
fn counts_rows(name: &str, args: &[Expr]) -> bool {
    match (name, args) {
        ("count", []) => true,
        ("count", [Expr::Literal(value, _)]) => !value.is_null(),
        _ => false,
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
            (name, args) if counts_rows(name, args) => AggKind::CountStar,
            ("count", [Expr::Column(c)]) => AggKind::Count(c.name.clone()),
            ("sum", [Expr::Column(c)]) => AggKind::Sum(c.name.clone()),
            ("min", [Expr::Column(c)]) => AggKind::Min(c.name.clone()),
            ("max", [Expr::Column(c)]) => AggKind::Max(c.name.clone()),
            // AVG's int arguments arrive wrapped in a coercion cast to
            // Float64; peeling it is safe here (and only here) because
            // the residual partial re-applies the cast explicitly and the
            // combiner works in Float64 throughout — no coercion pass runs
            // after an optimizer rule, so every expression this rewrite
            // emits must arrive fully typed.
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

/// Peel a grouped count's input down to its table scan when its one filter is
/// `column IS NOT NULL` on the grouping column, looking through the same
/// pure-column projection [`peel_input`] admits. `None` for any other filter.
fn peel_not_null_filter<'a>(input: &'a LogicalPlan, column: &str) -> Option<&'a TableScan> {
    let (predicate, scan) = peel_input(input)?;
    match predicate {
        Expr::IsNotNull(inner) => match inner.as_ref() {
            Expr::Column(c) if c.name == column => Some(scan),
            _ => None,
        },
        _ => None,
    }
}

/// The provider behind a scan, when it is ours and the scan carries no
/// filter the provider answers exactly.
///
/// An exact filter leaves no `Filter` node above the scan — DataFusion
/// hands it to the scan alone — so the scan looks unfiltered while its rows
/// are only the filter's. Every rewrite here answers from statistics that
/// describe whole segments, so each would count rows the filter excludes;
/// refusing the provider leaves the plan to the scan, which is exact. A
/// counting provider (this rule's own range count) is refused for the same
/// reason.
fn provider_of(scan: &TableScan) -> Option<&SupertableProvider> {
    // DataFusion 54 dropped `as_any` for an `Any` supertrait; downcast through
    // its provided `downcast_ref` (auto-derefs the `Arc`).
    let source = scan.source.downcast_ref::<DefaultTableSource>()?;
    let provider = source.table_provider.downcast_ref::<SupertableProvider>()?;
    (!provider.has_exact_filter(&scan.filters) && !provider.is_counting()).then_some(provider)
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
///
/// A NULL never matches a range, so a segment whose range column holds
/// nulls (or an unknown number of them) is never covered: its row count,
/// and every other column's statistics, include rows the filter drops.
/// `_id` is never null.
fn classify(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
    id_column: &str,
    range: &RangeFilter,
) -> Class {
    let (bounds, null_free) = if range.column == id_column {
        (
            Some((
                ScalarValue::Decimal128(Some(entry.id_min), DECIMAL128_PRECISION, DECIMAL128_SCALE),
                ScalarValue::Decimal128(Some(entry.id_max), DECIMAL128_PRECISION, DECIMAL128_SCALE),
            )),
            true,
        )
    } else {
        let stats = manifest
            .field_id(&range.column)
            .and_then(|id| entry.scalar_stats.column(id));
        (
            stats.and_then(|stats| {
                let (mn, mx) = stats.min_max_values()?;
                (!mn.is_null() && !mx.is_null()).then_some((mn, mx))
            }),
            stats.is_some_and(|stats| stats.null_count() == Some(0)),
        )
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
    if lo_ok && hi_ok && null_free {
        Class::Covered
    } else {
        Class::Boundary
    }
}

/// Whether `column`'s recorded counts describe what its rows read as, across
/// every one of `superfiles`.
///
/// A count shortcut answers from per-file `value_counts`, recorded in whatever
/// type each file was written in. While a column is converting, its files hold
/// two types, so counts taken from them key values that are now one value
/// apart and order them as the old type did: a `Utf8` `"7"` and a retyped `7`
/// are separate keys, and `"7" >= "10"` holds as text where `7 >= 10` does
/// not. The scan sees the column as the table now holds it, so a shortcut has
/// to decline exactly where [`has_required_stats`] does.
fn counts_describe_column(
    manifest: &ManifestSnapshot,
    column: &str,
    superfiles: &[Arc<SuperfileEntry>],
) -> bool {
    let Some(id) = manifest.field_id(column) else {
        return false;
    };
    let guard = ColumnTypeGuard::new(manifest, id);
    !guard.aggregates_mix_types() && !superfiles.iter().any(|e| guard.stats_are_stale(e))
}

/// Do the manifest stats cover everything `kinds` needs from a
/// covered segment?
///
/// A statistic is only usable when it was recorded in the type the table
/// reads the column as today. After a retype the recorded bounds describe
/// the old type, and they do not survive the cast: a `LargeUtf8` column
/// holding "7" and "10" records min "10" and max "7" lexicographically, so
/// answering `MAX` from them after a change to `Int64` returns 7 and `MIN`
/// returns 10 — wrong, and wrong in opposite directions. A sum or a null
/// count recorded in the old type is no better. So a file whose statistics
/// have gone stale, or a column mid-conversion whose files disagree, falls
/// back to the scan, which reads the data through the cast and derives the
/// answer afresh. Compaction re-derives the statistics in the new type when
/// it rewrites the file, and the covered path picks up again.
fn has_required_stats(
    manifest: &ManifestSnapshot,
    entry: &SuperfileEntry,
    kinds: &[AggKind],
) -> bool {
    let usable = |col: &str| {
        let id = manifest.field_id(col)?;
        let guard = ColumnTypeGuard::new(manifest, id);
        if guard.aggregates_mix_types() || guard.stats_are_stale(entry) {
            return None;
        }
        entry.scalar_stats.column(id)
    };
    // A file without the column holds it null in every row: its count is
    // known without statistics.
    let absent = |col: &str| {
        manifest
            .field_id(col)
            .is_some_and(|id| file_lacks_column(entry, id))
    };
    kinds.iter().all(|kind| match kind {
        AggKind::CountStar => true,
        AggKind::Count(col) => absent(col) || usable(col).is_some_and(|a| a.null_count().is_some()),
        AggKind::Sum(col) => usable(col).is_some_and(|a| a.sum().is_some()),
        AggKind::Min(col) | AggKind::Max(col) => usable(col).is_some(),
        AggKind::Avg(col) => {
            usable(col).is_some_and(|a| a.sum().is_some() && a.null_count().is_some())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supertable::manifest::ScalarStats;

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
            physical_schema: None,
            stem: None,
            birth_version: 0,
            superfile_id: uuid::Uuid::new_v4(),
            uri: crate::supertable::manifest::SuperfileUri::new_v4(),
            n_docs: 1,
            id_min: 10,
            id_max: 20,
            scalar_stats: ScalarStats::default(),
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
        let manifest = ManifestSnapshot::empty(Arc::new(
            crate::supertable::options::SupertableOptions::new(
                Arc::new(arrow_schema::Schema::empty()),
                vec![],
                vec![],
            )
            .expect("options"),
        ));

        assert!(matches!(
            classify(&manifest, &entry, "_id", &rf(Some(bound(25, true)), None)),
            Class::Disjoint
        ));
        assert!(matches!(
            classify(&manifest, &entry, "_id", &rf(None, Some(bound(5, true)))),
            Class::Disjoint
        ));
        assert!(matches!(
            classify(
                &manifest,
                &entry,
                "_id",
                &rf(Some(bound(0, true)), Some(bound(30, true)))
            ),
            Class::Covered
        ));
        assert!(matches!(
            classify(&manifest, &entry, "_id", &rf(Some(bound(15, true)), None)),
            Class::Boundary
        ));
        // Unknown column with no scalar stats → boundary (can't prune).
        assert!(matches!(
            classify(
                &manifest,
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

    /// `parse_aggregates` reads `COUNT(column)` as that column's non-null
    /// count, and still reads `COUNT(1)` as a row count.
    #[test]
    fn parse_aggregates_reads_count_of_a_column() {
        let kinds = parse_aggregates(&[count(col("fare")), count(lit(1i64)), sum(col("fare"))])
            .expect("every aggregate is supported");
        assert_eq!(
            kinds,
            vec![
                AggKind::Count("fare".to_string()),
                AggKind::CountStar,
                AggKind::Sum("fare".to_string()),
            ]
        );
    }

    #[test]
    #[allow(deprecated)] // `supports_rewrite` is the targeted method.
    fn rule_opts_into_rewrite_and_reports_name() {
        let rule = CoveredAggregateRewrite;
        assert!(rule.supports_rewrite(), "rule must use the rewrite path");
        assert_eq!(rule.name(), "covered_aggregate_rewrite");
    }
}
