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
//!   * **covered** — bounds fully inside the range AND carrying the
//!     stats the aggregate needs, as its live rows have them: their
//!     contribution is computed from manifest statistics, no scan. A
//!     segment with deletes has its statistics corrected for its deleted
//!     rows (see [`super::deleted_stats`]);
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
//!   * a segment with deletes whose statistics are not corrected for
//!     them exactly, or whose deletes are unknown, demotes to boundary —
//!     one superfile's deletes send that superfile to the scan, not the
//!     whole table;
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
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt, mem, ptr,
    sync::{Arc, Mutex},
    time::Instant,
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
    execution::{SessionState, SessionStateBuilder},
    functions::core::expr_fn::{coalesce, greatest, least},
    functions_aggregate::expr_fn::{count, max, min, sum},
    logical_expr::{
        Aggregate, EmptyRelation, Expr, Filter, LogicalPlan, LogicalPlanBuilder, Operator,
        TableScan, TableSource, lit, utils::split_conjunction,
    },
    optimizer::{OptimizerConfig, OptimizerRule, optimizer::ApplyOrder},
    prelude::{cast, col, ident},
};
use futures::future::{join_all, try_join_all};
use roaring::RoaringBitmap;
use tracing::debug;

use crate::supertable::{
    error::QueryError,
    manifest::{
        ManifestSnapshot, SuperfileEntry, add_sum_arrays, list::ScalarValueCounts,
        load_section_groups_blocking,
    },
    query::{
        deleted_stats::{
            CountsChange, DeletedRowStatsRequest, LiveBound, LiveColumn, RequestedColumn, Unknown,
            better_bound, read_requested, strictly_better,
        },
        provider::{CountedRange, SupertableProvider},
        range_counts::RangeBucket,
        skip::{ColumnTypeGuard, file_lacks_column},
        value_count_carry::{counts_of, counts_without, take_off},
    },
    schema::{DECIMAL128_PRECISION, DECIMAL128_SCALE, FieldId},
};

/// The covered/residual aggregate rewrite. Registered on the
/// `query_sql` session after DataFusion's built-in rules.
#[derive(Default)]
pub(crate) struct CoveredAggregateRewrite {
    /// Set on the rule of a dry run (see [`prepare_aggregate_inputs`]): the
    /// deleted rows each rewritten aggregate wants, by the provider to read
    /// them through. A collecting rule logs nothing; the statement's own
    /// optimization reports what it does.
    collector: Option<Arc<Mutex<Vec<(SupertableProvider, Vec<DeletedRowStatsRequest>)>>>>,
}

impl fmt::Debug for CoveredAggregateRewrite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CoveredAggregateRewrite")
            .field("collecting", &self.collector.is_some())
            .finish()
    }
}

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
        match try_rewrite(&plan, self.collector.as_deref()).map_err(QueryError::internal)? {
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

impl AggKind {
    /// The column the aggregate reads; `None` for `COUNT(*)`.
    fn column(&self) -> Option<&str> {
        match self {
            Self::CountStar => None,
            Self::Count(column)
            | Self::Sum(column)
            | Self::Min(column)
            | Self::Max(column)
            | Self::Avg(column) => Some(column),
        }
    }
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

/// Prepare `plan`'s aggregates to be answered from statistics, before it is
/// optimized:
///
///   * load every manifest part of each table an unfiltered, ungrouped
///     aggregate reads, subqueries included, so the rewrite finds the whole
///     table in memory (that aggregate's scan would load every part anyway);
///   * for a table with deletes under an aggregate of a shape the rewrite
///     answers, read its superfiles' deleted rows, so the rewrite's lookups
///     of them are cache reads;
///   * optimize the statement as it will be planned, and read the statistics
///     of the deleted rows the rewrite would use to answer superfiles with
///     deletes from their statistics, where reading those rows costs fewer
///     pages than scanning the live ones would (see [`read_requested`]). They
///     are kept for the table, so a later statement reads only rows deleted
///     since.
///
/// The rewrite stays exact without any of this: what is not read here, it
/// scans. `EXPLAIN` runs this too, as it plans the statement.
pub(crate) async fn prepare_aggregate_inputs(
    plan: &LogicalPlan,
    state: &SessionState,
) -> Result<(), QueryError> {
    let mut sources: Vec<Arc<dyn TableSource>> = Vec::new();
    let mut aggregated: Vec<Arc<dyn TableSource>> = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::Aggregate(agg) = node
            && let Some(scan) = aggregated_scan(agg)
        {
            if agg.group_expr.is_empty() && peel_unfiltered_scan(agg.input.as_ref()).is_some() {
                sources.push(Arc::clone(&scan.source));
            }
            if answerable_shape(agg) && reads_deleted_rows(agg) {
                aggregated.push(Arc::clone(&scan.source));
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(QueryError::DataFusion)?;
    try_join_all(
        sources
            .iter()
            .filter_map(|source| Some(supertable_provider(source)?.load_every_part())),
    )
    .await?;

    // A table without deletes has nothing to read: its statistics answer as
    // they are. Each table's deleted rows are read once, before the rewrite
    // looks them up.
    let mut with_deletes: Vec<&SupertableProvider> = Vec::new();
    for source in &aggregated {
        if let Some(provider) = supertable_provider(source)
            && provider.may_have_deletes()
            && !with_deletes
                .iter()
                .any(|seen| ptr::eq(seen.deleted_stats(), provider.deleted_stats()))
        {
            with_deletes.push(provider);
        }
    }
    if with_deletes.is_empty() {
        return Ok(());
    }
    join_all(
        with_deletes
            .iter()
            .map(|provider| provider.prefetch_listed_tombstones()),
    )
    .await;

    // The rewrite runs on the plan the optimizer's other rules leave it:
    // literals coerced, ranges simplified, aliases flattened. Optimizing the
    // statement here, as it will be planned, has the rewrite name the
    // deleted rows it would read in exactly that shape. A plan it fails to
    // optimize is the planner's to report when it runs.
    let collector = Arc::default();
    if let Err(error) = dry_run(state, &collector).optimize(plan) {
        debug!(%error, "aggregate inputs left unread: the statement does not optimize");
        return Ok(());
    }
    let requests = mem::take(&mut *collector.lock().expect("requests rows poisoned"));
    for (provider, requests) in requests {
        let cache = provider.deleted_stats();
        cache.prune_unlisted(|| {
            provider
                .manifest()
                .resident_superfiles()
                .map(|listed| listed.iter().map(|entry| entry.superfile_id).collect())
                .unwrap_or_default()
        });
        read_requested(&provider, state, cache, requests).await;
    }
    Ok(())
}

/// Our provider behind `source`, when it is one.
fn supertable_provider(source: &Arc<dyn TableSource>) -> Option<&SupertableProvider> {
    source
        .downcast_ref::<DefaultTableSource>()?
        .table_provider
        .downcast_ref::<SupertableProvider>()
}

/// Whether `agg` has a shape the rewrite answers: at most one grouping
/// column, and only `COUNT`, `SUM`, `MIN`, `MAX` and `AVG`, none of them
/// distinct or ordered. Any other aggregate never reads deleted rows' or
/// tombstones' state.
fn answerable_shape(agg: &Aggregate) -> bool {
    agg.group_expr.len() <= 1
        && agg.aggr_expr.iter().all(|expr| match unaliased(expr) {
            Expr::AggregateFunction(func) => {
                matches!(func.func.name(), "count" | "sum" | "min" | "max" | "avg")
                    && !func.params.distinct
                    && func.params.order_by.is_empty()
            }
            _ => false,
        })
}

/// `state` with the rewrite collecting into `collector` what it wants.
fn dry_run(
    state: &SessionState,
    collector: &Arc<Mutex<Vec<(SupertableProvider, Vec<DeletedRowStatsRequest>)>>>,
) -> SessionState {
    let collecting: Arc<dyn OptimizerRule + Send + Sync> = Arc::new(CoveredAggregateRewrite {
        collector: Some(Arc::clone(collector)),
    });
    let rules = state
        .optimizers()
        .iter()
        .map(|rule| match rule.name() == collecting.name() {
            true => Arc::clone(&collecting),
            false => Arc::clone(rule),
        })
        .collect();
    SessionStateBuilder::new_from_existing(state.clone())
        .with_optimizer_rules(rules)
        .build()
}

/// Whether the rewrite may read deleted rows' statistics for `agg`: over
/// the whole table, a range on one column, or a grouping column's non-null
/// values. Under any other filter it declines, so the pre-pass reads
/// nothing for it.
fn reads_deleted_rows(agg: &Aggregate) -> bool {
    if peel_unfiltered_scan(agg.input.as_ref()).is_some() {
        return true;
    }
    match agg.group_expr.as_slice() {
        [] => peel_input(agg.input.as_ref())
            .is_some_and(|(predicate, _)| extract_range(&predicate).is_some()),
        [Expr::Column(column)] => peel_not_null_filter(agg.input.as_ref(), &column.name).is_some(),
        _ => false,
    }
}

/// The deleted rows whose statistics a statement wants, and whether the
/// rewrite looking for them only looks (see
/// [`CoveredAggregateRewrite::collector`]).
struct StatsRequests {
    list: Vec<DeletedRowStatsRequest>,
    quiet: bool,
}

impl StatsRequests {
    fn logged() -> Self {
        Self {
            list: Vec::new(),
            quiet: false,
        }
    }
}

/// The scan an aggregate the rewrite may answer reads: under no filter, or
/// under one filter, behind at most a projection of columns.
fn aggregated_scan(agg: &Aggregate) -> Option<&TableScan> {
    let scan = peel_unfiltered_scan(agg.input.as_ref())
        .or_else(|| peel_input(agg.input.as_ref()).map(|(_, scan)| scan))?;
    provider_of(scan).is_some().then_some(scan)
}

fn try_rewrite(
    plan: &LogicalPlan,
    collector: Option<&Mutex<Vec<(SupertableProvider, Vec<DeletedRowStatsRequest>)>>>,
) -> DfResult<Option<LogicalPlan>> {
    let mut wants = StatsRequests {
        list: Vec::new(),
        quiet: collector.is_some(),
    };
    let rewritten = try_rewrite_wanting(plan, &mut wants)?;
    if let Some(collector) = collector
        && !wants.list.is_empty()
        && let LogicalPlan::Aggregate(agg) = plan
        && let Some(provider) = aggregated_scan(agg).and_then(provider_of)
    {
        collector
            .lock()
            .expect("requests rows poisoned")
            .push((provider.share(), wants.list));
    }
    Ok(rewritten)
}

/// [`try_rewrite`], adding to `requests` the deleted rows whose statistics
/// would answer superfiles it leaves to the scan.
fn try_rewrite_wanting(
    plan: &LogicalPlan,
    requests: &mut StatsRequests,
) -> DfResult<Option<LogicalPlan>> {
    let LogicalPlan::Aggregate(agg) = plan else {
        return Ok(None);
    };
    if let Some(rewritten) = rewrite_grouped_count_from_value_counts(agg, requests)? {
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

    // Unfiltered aggregate over a complete snapshot: every superfile is
    // answered from its statistics, corrected for its deleted rows, where
    // those answer for its live rows. When every superfile is, emit one
    // literal row directly. This handles SUM/AVG as well as COUNT/MIN/MAX
    // and avoids invoking the provider's Parquet scan at all. Otherwise only
    // the superfiles they don't answer for are scanned.
    if let Some(scan) = peel_unfiltered_scan(agg.input.as_ref()) {
        let Some(provider) = provider_of(scan) else {
            return Ok(None);
        };
        // One read of each superfile's deleted rows for the statement: the
        // statistics and the residual scan see the same ones.
        let provider = &provider.reading_tombstones_once();
        let manifest = provider.manifest();
        let Some(superfiles) = manifest.resident_superfiles() else {
            return Ok(None);
        };
        let Some((partials, split)) = answer_from_statistics(
            provider,
            &kinds,
            &superfiles,
            None,
            "unfiltered aggregate",
            requests,
        ) else {
            return Ok(None);
        };
        if split.scans_any() {
            return rewrite_with_residual(agg, scan, provider, &kinds, partials, split, None);
        }
        split.log();
        return literal_answer(agg, &partials).map(Some);
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
    let Some(range) = extract_range(&predicate) else {
        return Ok(None);
    };
    let provider = &provider.reading_tombstones_once();

    let manifest = provider.manifest();
    // Hierarchical manifests may expose a partial flat view. Answer from
    // statistics only when the resident entries are provably complete;
    // eagerly hydrated persisted tables then receive the same no-scan
    // aggregate path as in-process tables, while genuinely lazy views
    // decline safely. The range count below scans, so it needs no complete
    // view.
    let superfiles = manifest.complete_flat_superfiles();
    let id_column = manifest.options.id_column.as_str();

    // A range index counts every superfile, deleted rows left out, from a
    // block or two each: cheaper than scanning the ones with deletes, so
    // with any to scan it answers below, and no deleted row is read for
    // the value counts.
    let range_indexed = provider.counted_range(&predicate).is_some();
    let mut unread = StatsRequests {
        list: Vec::new(),
        quiet: requests.quiet,
    };
    let counts_wanted = match range_indexed {
        true => &mut unread,
        false => &mut *requests,
    };
    if matches!(kinds.as_slice(), [AggKind::CountStar])
        && let Some(counts_split) =
            split_value_counts(provider, &range.column, false, "value count", counts_wanted)
    {
        let scans = counts_split.split.scans_any();
        if (!scans || !range_indexed)
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
    let Some((partials, split)) = answer_from_statistics(
        provider,
        &kinds,
        superfiles,
        Some((&range, id_column)),
        "range aggregate",
        requests,
    ) else {
        return Ok(None);
    };
    if !split.scans_any() {
        split.log();
        return literal_answer(agg, &partials).map(Some);
    }
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

/// `agg` answered by `partials` alone, one literal row: every superfile it
/// reads was answered from statistics, and none is scanned.
fn literal_answer(agg: &Aggregate, partials: &[Partial]) -> DfResult<LogicalPlan> {
    let mut expressions = Vec::with_capacity(partials.len());
    for (i, partial) in partials.iter().enumerate() {
        let field = agg.schema.field(i);
        let output = field.data_type();
        let expression = match partial {
            Partial::Count(count) => lit(*count),
            Partial::Sum(Some(value)) | Partial::Bound(Some(value)) => {
                covered_literal(value, output)
            }
            // No live value: SUM, MIN and MAX of nothing are NULL.
            Partial::Sum(None) | Partial::Bound(None) | Partial::Avg(None) => {
                lit(ScalarValue::try_from(output)?)
            }
            Partial::Avg(Some((sum, count))) => {
                cast(lit(sum.clone()), DataType::Float64) / cast(lit(*count), DataType::Float64)
            }
        };
        expressions.push(expression.alias(field.name().clone()));
    }
    LogicalPlanBuilder::empty(true)
        .project(expressions)?
        .build()
}

/// A column an aggregate reads, as the manifest's statistics know it.
struct StatsColumn {
    name: String,
    id: FieldId,
    /// The type the table stores the column as, which its deleted rows are
    /// read as.
    data_type: Option<DataType>,
    guard: ColumnTypeGuard,
}

impl StatsColumn {
    fn new(manifest: &ManifestSnapshot, name: &str) -> Option<Self> {
        let id = manifest.field_id(name)?;
        let data_type = manifest
            .stored_schema()
            .field_with_name(name)
            .ok()
            .map(|field| field.data_type().clone());
        Some(Self {
            name: name.to_string(),
            id,
            data_type,
            guard: ColumnTypeGuard::new(manifest, id),
        })
    }

    /// Whether `entry`'s statistics for the column describe its values as
    /// the table reads them today.
    ///
    /// A statistic is only usable when it was recorded in the type the
    /// table reads the column as today. After a retype the recorded bounds
    /// describe the old type, and they do not survive the cast: a
    /// `LargeUtf8` column holding "7" and "10" records min "10" and max "7"
    /// lexicographically, so answering `MAX` from them after a change to
    /// `Int64` returns 7 and `MIN` returns 10 — wrong, and wrong in opposite
    /// directions. A sum or a null count recorded in the old type is no
    /// better. So a file whose statistics have gone stale, or a column
    /// mid-conversion whose files disagree, falls back to the scan, which
    /// reads the data through the cast and derives the answer afresh.
    /// Compaction re-derives the statistics in the new type when it
    /// rewrites the file, and the statistics answer again.
    fn current_in(&self, entry: &SuperfileEntry) -> bool {
        !self.guard.aggregates_mix_types() && !self.guard.stats_are_stale(entry)
    }

    /// The column as a read of deleted rows names it.
    fn requests(&self) -> RequestedColumn {
        RequestedColumn {
            field: self.id,
            name: self.name.clone(),
        }
    }
}

/// What one superfile's live rows give one aggregate.
enum Term {
    Count(i64),
    /// The sum; `None` when no live row holds a value.
    Sum(Option<ScalarValue>),
    /// The sum and the count of the values; `None` when there are none.
    Avg(Option<(ScalarValue, i64)>),
    Bound(LiveBound),
}

/// The terms of one superfile answered from statistics, one per aggregate.
struct Answered {
    entry: Arc<SuperfileEntry>,
    deleted: Arc<RoaringBitmap>,
    terms: Vec<Term>,
}

/// Answer `kinds` from the statistics of each of `superfiles` that holds a
/// row the `range` (with the table's id column) keeps, corrected for its
/// deleted rows, and split off the ones they don't answer for to be
/// scanned. Deleted rows whose statistics would answer a scanned superfile
/// are added to `requests`. `None`, logged, when nothing is answered or the
/// answers do not fold.
fn answer_from_statistics(
    provider: &SupertableProvider,
    kinds: &[AggKind],
    superfiles: &[Arc<SuperfileEntry>],
    range: Option<(&RangeFilter, &str)>,
    shape: &'static str,
    requests: &mut StatsRequests,
) -> Option<(Vec<Partial>, Split)> {
    let manifest = provider.manifest();
    let mut split = Split::new(shape, requests.quiet);
    let columns: Vec<Option<StatsColumn>> = kinds
        .iter()
        .map(|kind| match kind.column() {
            None => Some(None),
            Some(name) => StatsColumn::new(manifest, name).map(Some),
        })
        .collect::<Option<_>>()?;
    let mut candidates: Vec<Answered> = Vec::new();
    for entry in superfiles {
        if let Some((range, id_column)) = range {
            match classify(manifest, entry, id_column, range) {
                Class::Disjoint => {
                    split.skip();
                    continue;
                }
                Class::Boundary => {
                    split.scan(ScanReason::Boundary, entry);
                    continue;
                }
                Class::Covered => {}
            }
        }
        let Ok(deleted) = provider.deleted_rows(entry.superfile_id, Instant::now()) else {
            split.scan(ScanReason::DeletesUnknown, entry);
            continue;
        };
        let mut terms = Vec::with_capacity(kinds.len());
        let mut want: BTreeSet<RequestedColumn> = BTreeSet::new();
        let mut unknown = None;
        // One view of each column per superfile, whatever number of
        // aggregates read it.
        let mut lives: HashMap<FieldId, Result<LiveColumn, Unknown>> = HashMap::new();
        for (kind, column) in kinds.iter().zip(&columns) {
            let live = column.as_ref().map(|column| {
                lives
                    .entry(column.id)
                    .or_insert_with(|| live_of(provider, column, entry, &deleted))
                    .as_ref()
                    .map_err(|why| *why)
            });
            match term(kind, live, entry, &deleted) {
                Ok(term) => terms.push(term),
                Err(why) => {
                    if let (Unknown::Unread, Some(column)) = (why, column) {
                        want.insert(column.requests());
                    }
                    unknown = Some(unknown.map_or(why, |seen: Unknown| seen.max(why)));
                }
            }
        }
        match unknown {
            None => candidates.push(Answered {
                entry: Arc::clone(entry),
                deleted,
                terms,
            }),
            Some(why) => {
                split.scan(scan_reason(why, &deleted), entry);
                if why == Unknown::Unread {
                    requests.list.push(DeletedRowStatsRequest {
                        entry: Arc::clone(entry),
                        deleted,
                        columns: want,
                    });
                }
            }
        }
    }

    let candidates = settle_one_sided_bounds(kinds, &columns, candidates, &mut split, requests);

    if candidates.is_empty() && split.scans_any() {
        // Nothing answerable from statistics — the rewrite would only add
        // plan noise over the normal scan.
        split.decline("no superfile is answerable from statistics");
        return None;
    }
    let Some(partials) = fold_terms(kinds, &candidates) else {
        // A fold failure (overflow, type surprise) is always safe to
        // decline.
        split.decline("a statistic does not fold");
        return None;
    };
    for answered in candidates {
        split.answer_past(&answered.entry, !answered.deleted.is_empty());
    }
    Some((partials, split))
}

/// The `candidates` whose `MIN` or `MAX` bounds a column only from one side
/// (see [`LiveBound::OneSided`]) settled: such a bound answers when another
/// superfile's exact bound is no worse, since the answer lies on that bound
/// or beyond it, whatever the first one's live rows hold. A superfile it
/// does not answer for is split off to be scanned, its deleted rows added
/// to `requests` when their statistics would answer it.
fn settle_one_sided_bounds(
    kinds: &[AggKind],
    columns: &[Option<StatsColumn>],
    mut candidates: Vec<Answered>,
    split: &mut Split,
    requests: &mut StatsRequests,
) -> Vec<Answered> {
    for (at, kind) in kinds.iter().enumerate() {
        let max = match kind {
            AggKind::Min(_) => false,
            AggKind::Max(_) => true,
            _ => continue,
        };
        let best = candidates
            .iter()
            .filter_map(|answered| match &answered.terms[at] {
                Term::Bound(LiveBound::Exact(Some(value))) => Some(value),
                _ => None,
            })
            .try_fold(None, |best: Option<&ScalarValue>, value| {
                Some(Some(match best {
                    None => value,
                    Some(best) => better_bound(best, value, max)?,
                }))
            })
            .flatten()
            .cloned();
        let mut kept = Vec::with_capacity(candidates.len());
        for mut answered in candidates {
            if let Term::Bound(LiveBound::OneSided {
                bound,
                exact_once_read,
            }) = &answered.terms[at]
            {
                let beaten = best
                    .as_ref()
                    .and_then(|best| strictly_better(bound, best, max))
                    .is_some_and(|better| !better);
                if !beaten {
                    if *exact_once_read && let Some(column) = &columns[at] {
                        requests.list.push(DeletedRowStatsRequest {
                            entry: Arc::clone(&answered.entry),
                            deleted: Arc::clone(&answered.deleted),
                            columns: [column.requests()].into_iter().collect(),
                        });
                    }
                    split.scan(ScanReason::DeletedRows, &answered.entry);
                    continue;
                }
                answered.terms[at] = Term::Bound(LiveBound::Exact(None));
            }
            kept.push(answered);
        }
        candidates = kept;
    }
    candidates
}

/// `column` of `entry` as its live rows have it, `deleted` of its rows
/// deleted; unknown when its statistics are of a type the column no longer
/// has.
fn live_of<'a>(
    provider: &SupertableProvider,
    column: &StatsColumn,
    entry: &'a SuperfileEntry,
    deleted: &Arc<RoaringBitmap>,
) -> Result<LiveColumn<'a>, Unknown> {
    if !file_lacks_column(entry, column.id) && !column.current_in(entry) {
        return Err(Unknown::Scan);
    }
    let data_type = column.data_type.as_ref().ok_or(Unknown::Scan)?;
    Ok(live_column(provider, entry, column.id, data_type, deleted))
}

/// What `entry`'s live rows give `kind`, `deleted` of its rows deleted,
/// from `live`, the view of the column it reads (`None` for `COUNT(*)`).
fn term(
    kind: &AggKind,
    live: Option<Result<&LiveColumn, Unknown>>,
    entry: &SuperfileEntry,
    deleted: &RoaringBitmap,
) -> Result<Term, Unknown> {
    let live_rows = || {
        entry
            .n_docs
            .checked_sub(deleted.len())
            .and_then(|rows| i64::try_from(rows).ok())
            .ok_or(Unknown::Scan)
    };
    let Some(live) = live else {
        // `COUNT(*)`: the rows less the deleted ones.
        return live_rows().map(Term::Count);
    };
    let live = live?;
    let count = |live: &LiveColumn| {
        live.non_null()
            .and_then(|n| i64::try_from(n).map_err(|_| Unknown::Scan))
    };
    Ok(match kind {
        AggKind::CountStar => Term::Count(live_rows()?),
        AggKind::Count(_) => Term::Count(count(live)?),
        AggKind::Sum(_) => Term::Sum(live.sum()?),
        AggKind::Avg(_) => Term::Avg(match live.sum()? {
            None => None,
            Some(sum) => Some((sum, count(live)?)),
        }),
        AggKind::Min(_) => Term::Bound(live.bound(false)?),
        AggKind::Max(_) => Term::Bound(live.bound(true)?),
    })
}

/// `entry`'s column `id`, read as `data_type`, as its live rows have it,
/// `deleted` of its rows deleted: corrected by the statistics the table
/// keeps of exactly those rows, when it has them.
fn live_column<'a>(
    provider: &SupertableProvider,
    entry: &'a SuperfileEntry,
    id: FieldId,
    data_type: &DataType,
    deleted: &Arc<RoaringBitmap>,
) -> LiveColumn<'a> {
    let lacks = file_lacks_column(entry, id);
    let delta = match deleted.is_empty() || lacks {
        true => None,
        false => provider
            .deleted_stats()
            .get(entry.superfile_id, id, data_type, deleted),
    };
    LiveColumn::new(entry, id, lacks, deleted.len(), delta)
}

/// Each aggregate's contribution from `answered`; `None` when one does not
/// fold (overflow, a type surprise).
fn fold_terms(kinds: &[AggKind], answered: &[Answered]) -> Option<Vec<Partial>> {
    let mut partials = Vec::with_capacity(kinds.len());
    for (at, kind) in kinds.iter().enumerate() {
        let terms = answered.iter().map(|answered| &answered.terms[at]);
        let partial = match kind {
            AggKind::CountStar | AggKind::Count(_) => {
                let mut total = 0i64;
                for term in terms {
                    let Term::Count(count) = term else {
                        return None;
                    };
                    total = total.checked_add(*count)?;
                }
                Partial::Count(total)
            }
            AggKind::Sum(_) => {
                let mut total: Option<ArrayRef> = None;
                for term in terms {
                    let Term::Sum(sum) = term else {
                        return None;
                    };
                    total = add_sum(total, sum.as_ref())?;
                }
                Partial::Sum(match total {
                    None => None,
                    Some(total) => Some(ScalarValue::try_from_array(&total, 0).ok()?),
                })
            }
            AggKind::Avg(_) => {
                let (mut total, mut values): (Option<ArrayRef>, i64) = (None, 0);
                for term in terms {
                    let Term::Avg(avg) = term else {
                        return None;
                    };
                    if let Some((sum, count)) = avg {
                        total = add_sum(total, Some(sum))?;
                        values = values.checked_add(*count)?;
                    }
                }
                let sum = total.and_then(|total| ScalarValue::try_from_array(&total, 0).ok());
                Partial::Avg(match (sum, values) {
                    (Some(sum), values) if values > 0 => Some((sum, values)),
                    (None, 0) => None,
                    _ => return None,
                })
            }
            AggKind::Min(_) | AggKind::Max(_) => {
                let max = matches!(kind, AggKind::Max(_));
                let mut best: Option<ScalarValue> = None;
                for term in terms {
                    match term {
                        Term::Bound(LiveBound::Exact(None)) => {}
                        Term::Bound(LiveBound::Exact(Some(value))) => {
                            best = Some(match best {
                                None => value.clone(),
                                Some(best) => better_bound(&best, value, max)?.clone(),
                            });
                        }
                        _ => return None,
                    }
                }
                Partial::Bound(best)
            }
        };
        partials.push(partial);
    }
    Some(partials)
}

/// `total` plus `sum`, as one-row sum arrays; `total` unchanged for no sum.
/// `None` when they don't add (overflow, a type mismatch).
fn add_sum(total: Option<ArrayRef>, sum: Option<&ScalarValue>) -> Option<Option<ArrayRef>> {
    let Some(sum) = sum else {
        return Some(total);
    };
    let sum = sum.to_array().ok()?;
    Some(Some(match total {
        None => sum,
        Some(total) => add_sum_arrays(&total, &sum)?,
    }))
}

/// Why a superfile whose live rows its statistics don't answer for is
/// scanned, by why they don't and whether it has deleted rows.
fn scan_reason(why: Unknown, deleted: &RoaringBitmap) -> ScanReason {
    match (why, deleted.is_empty()) {
        (Unknown::Unread, _) | (Unknown::Scan, false) => ScanReason::DeletedRows,
        (Unknown::Scan, true) => ScanReason::MissingStatistics,
    }
}

/// Why one superfile is scanned rather than answered from its statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ScanReason {
    /// The range cuts across its bounds, or its range column holds nulls.
    Boundary,
    /// It has deleted rows its statistics can't be corrected for.
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
    /// Of those, the ones with deleted rows, corrected for them.
    corrected: usize,
    /// Superfiles that hold no row the aggregate reads.
    skipped: usize,
    /// Superfiles scanned.
    scanned: Vec<Arc<SuperfileEntry>>,
    /// How many are scanned for each reason.
    reasons: BTreeMap<ScanReason, usize>,
    /// Log nothing: the split is only looked at (see [`StatsRequests`]).
    quiet: bool,
}

impl Split {
    fn new(shape: &'static str, quiet: bool) -> Self {
        Self {
            quiet,
            shape,
            answered: Vec::new(),
            corrected: 0,
            skipped: 0,
            scanned: Vec::new(),
            reasons: BTreeMap::new(),
        }
    }

    /// `entry` answered from its statistics, `corrected` for its deleted
    /// rows when it has some.
    fn answer_past(&mut self, entry: &Arc<SuperfileEntry>, corrected: bool) {
        self.answered.push(Arc::clone(entry));
        self.corrected += usize::from(corrected);
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
    /// statistics, 9 of them past their deleted rows".
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
        if self.corrected > 0 {
            note.push_str(&format!(
                ", {} of them past their deleted rows",
                self.corrected
            ));
        }
        if self.skipped > 0 {
            note.push_str(&format!(", {} outside the range", self.skipped));
        }
        note
    }

    /// Record the split taken.
    fn log(&self) {
        if self.quiet {
            return;
        }
        debug!(
            shape = self.shape,
            answered = self.answered.len(),
            corrected = self.corrected,
            skipped = self.skipped,
            scanned = ?self.reasons,
            "aggregate answered from statistics"
        );
    }

    /// Record why the aggregate is left to the scan.
    fn decline(&self, why: &str) {
        if self.quiet {
            return;
        }
        debug!(
            shape = self.shape,
            answered = self.answered.len(),
            corrected = self.corrected,
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
    // rewritten plan's schema matches the original node's exactly. A
    // covered side with no live value leaves the residual's own answer,
    // NULL when the residual has none either.
    let mut final_exprs: Vec<Expr> = Vec::with_capacity(kinds.len());
    for (i, (kind, partial)) in kinds.iter().zip(&partials).enumerate() {
        let field = agg.schema.field(i);
        let name = field.name().clone();
        let resid = |suffix: &str| col(format!("__resid_{i}_{suffix}"));
        let expr = match (kind, partial) {
            (AggKind::CountStar | AggKind::Count(_), Partial::Count(n)) => {
                // COUNT over an empty residual is 0, never NULL.
                resid("cnt") + lit(*n)
            }
            (AggKind::Sum(_), Partial::Sum(None)) => resid("sum"),
            (AggKind::Sum(_), Partial::Sum(Some(value))) => {
                let zero = typed_zero(value)?;
                coalesce(vec![resid("sum"), lit(zero)]) + lit(value.clone())
            }
            (AggKind::Min(_), Partial::Bound(None)) => resid("min"),
            (AggKind::Min(_), Partial::Bound(Some(value))) => least(vec![
                resid("min"),
                covered_literal(value, field.data_type()),
            ]),
            (AggKind::Max(_), Partial::Bound(None)) => resid("max"),
            (AggKind::Max(_), Partial::Bound(Some(value))) => greatest(vec![
                resid("max"),
                covered_literal(value, field.data_type()),
            ]),
            (AggKind::Avg(_), Partial::Avg(None)) => {
                resid("sum") / cast(resid("cnt"), DataType::Float64)
            }
            (AggKind::Avg(_), Partial::Avg(Some((sum, count)))) => {
                // The residual partial is Float64 by construction (its sum
                // is over an explicit cast), so the whole combiner stays in
                // Float64 — the coalesce zero and the covered sum must
                // match it, since nothing coerces after this rule.
                let total_sum = coalesce(vec![resid("sum"), lit(0.0_f64)])
                    + cast(lit(sum.clone()), DataType::Float64);
                let total_cnt = resid("cnt") + lit(*count);
                total_sum / cast(total_cnt, DataType::Float64)
            }
            _ => return Ok(None),
        };
        final_exprs.push(expr.alias(name));
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
fn rewrite_grouped_count_from_value_counts(
    aggregate: &Aggregate,
    requests: &mut StatsRequests,
) -> DfResult<Option<LogicalPlan>> {
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
    // One read of each superfile's deleted rows for the statement: the
    // counts and the residual scan see the same ones.
    let provider = &provider.reading_tombstones_once();
    let column = group_column.name.as_str();
    let Some(counts_split) =
        split_value_counts(provider, column, !nulls_filtered, "grouped count", requests)
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

    let group_type = aggregate.schema.field(0).data_type();
    let nulls = if nulls_filtered { 0 } else { nulls };
    let rows = match group_rows(&counts, nulls, group_type) {
        Ok(rows) => rows,
        Err(why) => {
            split.decline(why);
            return Ok(None);
        }
    };

    if !split.scans_any() {
        split.log();
        // No live row holds a group: the grouping is empty.
        if rows.is_empty() {
            return Ok(Some(LogicalPlan::EmptyRelation(EmptyRelation {
                produce_one_row: false,
                schema: Arc::clone(&aggregate.schema),
            })));
        }
        // Named and qualified exactly as the aggregate's output, which may be
        // the plan's root: the schema a rule leaves must be the one it found.
        let columns = (1..=aggregate.schema.fields().len())
            .map(|index| col(format!("column{index}")))
            .collect();
        return Ok(Some(
            LogicalPlanBuilder::values_with_schema(rows, &aggregate.schema)?
                .project(as_output(&aggregate.schema, columns))?
                .build()?,
        ));
    }

    split.log();
    let restricted = split.restrict(provider);
    union_with_scanned_groups(
        aggregate,
        scan,
        restricted,
        group_column,
        nulls_filtered,
        rows,
    )
    .map(Some)
}

/// One row of `(value, count)` per value of `counts`, cast to the grouping
/// column's `group_type`, and one for the null group when `nulls` rows are
/// null; why not, when one does not fit.
fn group_rows(
    counts: &Counts,
    nulls: u64,
    group_type: &DataType,
) -> Result<Vec<Vec<Expr>>, &'static str> {
    let mut rows = Vec::with_capacity(counts.len() + 1);
    for (value, count) in counts.iter() {
        let count = i64::try_from(count).map_err(|_| "a count overflows")?;
        let value = value
            .cast_to(group_type)
            .map_err(|_| "a value does not cast to the column's type")?;
        rows.push(vec![lit(value), lit(count)]);
    }
    if nulls > 0 {
        let (Ok(nulls), Ok(null)) = (i64::try_from(nulls), ScalarValue::try_from(group_type))
        else {
            return Err("the null group does not fit");
        };
        rows.push(vec![lit(null), lit(nulls)]);
    }
    Ok(rows)
}

/// The grouped count over the `restricted` scan of `scan`'s superfiles,
/// added up by value with the `rows` answered from statistics, and named
/// as `aggregate`'s output.
fn union_with_scanned_groups(
    aggregate: &Aggregate,
    scan: &TableScan,
    restricted: SupertableProvider,
    group_column: &datafusion::common::Column,
    nulls_filtered: bool,
    rows: Vec<Vec<Expr>>,
) -> DfResult<LogicalPlan> {
    let mut scanned = LogicalPlanBuilder::scan(
        scan.table_name.clone(),
        provider_as_source(Arc::new(restricted) as Arc<dyn TableProvider>),
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
    summed
        .project(as_output(
            &aggregate.schema,
            vec![
                col(GROUP_VALUE),
                coalesce(vec![col(GROUP_COUNT), lit(0i64)]),
            ],
        ))?
        .build()
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

/// How a column's value counts answer one statement: the superfiles whose
/// statistics, corrected for their deleted rows, describe their live rows,
/// and the rows among them holding no value; the rest are scanned.
struct CountsSplit {
    id: FieldId,
    split: Split,
    /// What each answered superfile's deleted rows take off its stored
    /// counts, in the order of `split.answered`.
    changes: Vec<CountsChange>,
    /// The answered superfiles' live null rows; 0 unless they were counted.
    nulls: u64,
}

/// Split the superfiles for counting `column` by value from statistics: one
/// whose deleted rows' values are not known, or (with `count_nulls`) whose
/// live null rows are not, is scanned, and the deleted rows whose
/// statistics would tell are added to `requests`. `None`, logged, when the
/// counts can't answer at all: a segment-restricted provider (the rewrite's
/// own residual), a manifest not fully loaded, a column mid-retype, or no
/// superfile answered.
fn split_value_counts(
    provider: &SupertableProvider,
    column: &str,
    count_nulls: bool,
    shape: &'static str,
    requests: &mut StatsRequests,
) -> Option<CountsSplit> {
    let split = Split::new(shape, requests.quiet);
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
    let stats_column = StatsColumn::new(manifest, column)?;
    let data_type = stats_column.data_type.clone()?;
    let id = stats_column.id;
    // The null counts and value counts are read in place; their groups are
    // read and decoded off the runtime's queue first, as a scan's are.
    load_section_groups_blocking(superfiles, &[id]);
    let mut counts_split = CountsSplit {
        id,
        split,
        changes: Vec::new(),
        nulls: 0,
    };
    for entry in superfiles {
        let Ok(deleted) = provider.deleted_rows(entry.superfile_id, Instant::now()) else {
            counts_split.split.scan(ScanReason::DeletesUnknown, entry);
            continue;
        };
        let live = live_column(provider, entry, id, &data_type, &deleted);
        let answer = live.counts_change().and_then(|change| match count_nulls {
            true => live.live_nulls().map(|nulls| (change, nulls)),
            false => Ok((change, 0)),
        });
        match answer {
            Ok((change, nulls)) => {
                let Some(total) = counts_split.nulls.checked_add(nulls) else {
                    counts_split.split.decline("the null count overflows");
                    return None;
                };
                counts_split.nulls = total;
                counts_split.changes.push(change);
                counts_split.split.answer_past(entry, !deleted.is_empty());
            }
            Err(why) => {
                counts_split.split.scan(scan_reason(why, &deleted), entry);
                if why == Unknown::Unread {
                    requests.list.push(DeletedRowStatsRequest {
                        entry: Arc::clone(entry),
                        deleted,
                        columns: [stats_column.requests()].into_iter().collect(),
                    });
                }
            }
        }
    }
    if counts_split.split.answered.is_empty() && counts_split.split.scans_any() {
        counts_split
            .split
            .decline("no superfile is answerable from statistics");
        return None;
    }
    Some(counts_split)
}

/// A column's value counts over the superfiles a [`CountsSplit`] answers.
enum Counts {
    /// Every superfile's, none with deletes: the table's own counts.
    Table(Arc<ScalarValueCounts>),
    /// Some of them, or some with deletes.
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

/// `column`'s live value counts over the superfiles `counts_split`
/// answers: the table's carried counts less the scanned superfiles' own,
/// or, where those can't be taken off (a scanned superfile holding more
/// distinct values than it counts), the answered superfiles' counts added
/// up; then less what each answered superfile's deleted rows held. `None`
/// when the answered superfiles' values are not all counted.
fn answered_counts(
    provider: &SupertableProvider,
    column: &str,
    counts_split: &CountsSplit,
) -> Option<Counts> {
    let total = provider.exact_value_counts(column);
    let CountsSplit {
        id, split, changes, ..
    } = counts_split;
    let unchanged = changes.iter().all(|c| *c == CountsChange::Unchanged);
    if !split.scans_any() && unchanged {
        return total.map(Counts::Table);
    }
    let mut counts = total
        .and_then(|total| counts_without(&total, &split.scanned, *id))
        .or_else(|| counts_of(&split.answered, *id))?;
    for (entry, change) in split.answered.iter().zip(changes) {
        match change {
            CountsChange::Unchanged => {}
            CountsChange::All => {
                let stored = entry.scalar_stats.column(*id)?.value_counts()?;
                counts = take_off(counts, stored.entries().iter().map(|(v, n)| (v, *n)))?;
            }
            CountsChange::Less(less) => {
                counts = take_off(counts, less.iter().map(|(v, n)| (v, *n)))?;
            }
        }
    }
    Some(Counts::Answered(counts))
}

/// `column`'s live rows by value, and the rows holding no value, when the
/// manifest's statistics, corrected for each superfile's deleted rows from
/// what the table has read of them, describe every superfile's live rows
/// (see [`split_value_counts`]); `None` otherwise.
pub(crate) fn value_counts_from_statistics(
    provider: &SupertableProvider,
    column: &str,
) -> Option<(Vec<(ScalarValue, u64)>, u64)> {
    let provider = &provider.reading_tombstones_once();
    let counts_split = split_value_counts(
        provider,
        column,
        true,
        "value_counts",
        &mut StatsRequests::logged(),
    )?;
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

/// Each aggregate's contribution from the superfiles answered from
/// statistics.
enum Partial {
    Count(i64),
    /// The sum; `None` when no answered row holds a value.
    Sum(Option<ScalarValue>),
    /// The bound; `None` when no answered row holds a value.
    Bound(Option<ScalarValue>),
    /// The sum and count of the values; `None` when there are none.
    Avg(Option<(ScalarValue, i64)>),
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
            let inner = unaliased(expr);
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

/// `expr` under every alias it carries. A plan built through the DataFrame
/// API names an aggregate as it is made (`count_all()` is `COUNT(*)` named
/// so) and again as it is selected, and is only flattened once analyzed.
fn unaliased(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = alias.expr.as_ref();
    }
    expr
}

/// Parse the aggregate outputs; `None` if any is unsupported.
fn parse_aggregates(exprs: &[Expr]) -> Option<Vec<AggKind>> {
    let mut kinds = Vec::with_capacity(exprs.len());
    for expr in exprs {
        // Outputs may carry an alias wrapper, or several.
        let inner = unaliased(expr);
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

/// The provider behind a scan, when it is ours, not this rule's own
/// residual scan (which is never rewritten again), and the scan carries no
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
    (!provider.is_segment_restricted()
        && !provider.has_exact_filter(&scan.filters)
        && !provider.is_counting())
    .then_some(provider)
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
/// to decline exactly where [`StatsColumn::current_in`] does.
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
        let rule = CoveredAggregateRewrite::default();
        assert!(rule.supports_rewrite(), "rule must use the rewrite path");
        assert_eq!(rule.name(), "covered_aggregate_rewrite");
    }
}
