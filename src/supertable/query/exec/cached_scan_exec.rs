// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`CachedScanExec`]: a table scan served from decoded columns kept across
//! statements ([`column_cache`](crate::supertable::query::column_cache)).
//!
//! The provider's scan otherwise hands its files to DataFusion's Parquet
//! reader, which decompresses and decodes every row group's column chunks on
//! every statement. With a decoded-column cache configured, the provider builds
//! this node instead whenever it can. Each partition walks its share of the
//! files' row groups; for each it takes every projected column from the cache,
//! decodes the ones the cache lacks from the row group whole
//! ([`read_file_row_group`], with the schema adaptation the search paths
//! use), keeps them, and drops the row group's deleted rows. When an index
//! chose a file's rows (an `ORDER BY ... LIMIT` over a range, or a scan the
//! full-text index narrowed in some superfiles only), only those rows come
//! out, and a row group holding none of them is not read.
//!
//! A row group whose chosen rows are fewer than half its live rows, and whose
//! columns the cache lacks, has just those rows read from its pages, as the
//! Parquet scan reads them, and nothing kept: one selective statement does not
//! decode a row group whole or push out the columns other statements read
//! whole. The cache remembers the row group, and the next scan that chooses
//! rows in it decodes and keeps its columns
//! ([`DecodedColumnCache::read_in_part_before`]). A row group whose columns
//! are kept hands over its chosen rows from them.
//!
//! A parent hands down filters that tighten as the statement runs
//! (`DynamicFilterPhysicalExpr`): a top-k its running bound (the k-th best
//! sort value so far), a hash join a filter on its probe side built from the
//! rows of its build side, a `MIN` or `MAX` aggregate the best value so far.
//! Each rules out only rows that can't change its parent's answer, and the
//! parent still applies its own check, so the scan keeps every such filter.
//! Every filter skips a row group its statistics rule out, just before the
//! row group is read. When the scan reads best first for a sort above it —
//! what a top-k asks for — its bound also drops the rows it rules out from
//! each batch the scan hands over; a hash join's or aggregate's filter, which
//! asks for no order, does not, matching the Parquet scan, which filters rows
//! only under `ORDER BY ... LIMIT`. Asked for a sort on one of its columns,
//! the scan reads the row groups whose statistics start best first, so a
//! top-k settles early and the rest go unread.
//!
//! How much that saves depends on how the sort column's values cluster by
//! row group. On a column whose values follow the order rows are written in
//! (as `_id`'s do), few row groups reach a settled bound and the rest are
//! skipped; on one whose values spread across every row group, the bound
//! skips none, and each row group and batch is checked for nothing.
//!
//! It answers what the Parquet scan would: the same columns, typed as the scan
//! schema types them, each file's rows in file order, without the rows a
//! tombstone deletes. A `WHERE` stays above it in a `FilterExec`, as above the
//! Parquet scan, and like the Parquet scan it skips a row group whose
//! statistics show the `WHERE` can't match ([`RowGroupFilter`]): such a row
//! group is neither decoded nor kept, so a first run decodes what the Parquet
//! scan would, and a table larger than the cache doesn't decode the rest of
//! it on every run.
//!
//! The Parquet scan serves the statement instead when it reads less:
//! - the columns this scan would decode don't fit the cache. Kept, they
//!   would push each other out, and every run would decode them all again;
//! - a filtered `ORDER BY ... LIMIT` whose other columns are wide. The top-k
//!   rule (`RowFilterUnderTopK`) takes back the Parquet scan this one stands
//!   in for, whose row filter decodes those columns only for the rows that
//!   pass.

use std::{
    collections::{HashMap, HashSet},
    fmt,
    ops::Range,
    option::IntoIter,
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array, UInt64Array,
    builder::BooleanBufferBuilder, cast::AsArray, new_null_array, types::Int16Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef, SortOptions};
use datafusion::{
    arrow::compute::{
        can_cast_types, cast_with_options, concat, filter, filter_record_batch, sort_to_indices,
        take,
    },
    common::{
        Column as StatsColumn, ScalarValue, Statistics, config::ConfigOptions,
        pruning::PruningStatistics,
    },
    datasource::physical_plan::parquet::ParquetAccessPlan,
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{
        EquivalenceProperties, PhysicalExpr, PhysicalSortExpr,
        expressions::{Column, DynamicFilterPhysicalExpr, Literal},
        utils::{collect_columns, reassign_expr_columns, split_conjunction},
    },
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_expr_common::physical_expr::{snapshot_generation, snapshot_physical_expr},
    physical_optimizer::pruning::PruningPredicate,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
        },
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        sort_pushdown::SortOrderPushdownResult,
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt, future, stream};
use parquet::{
    arrow::{arrow_reader::statistics::StatisticsConverter, parquet_to_arrow_schema},
    file::metadata::{ParquetMetaData, RowGroupMetaData},
};
use roaring::RoaringBitmap;
use tokio::spawn;
use tracing::Instrument;

use crate::{
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        manifest::{ManifestSnapshot, SuperfileUri, hll::HllSketch},
        query::{
            column_cache::{ColumnKey, DecodedColumnCache, RowGroupKey},
            exec::common::{hit_output_schema, read_file_row_group, take_file_columns},
            read_ahead::{file_pruning, file_schema, pruning_for, statistics_pruned},
            schema_adapter::TableExprAdapterFactory,
        },
        schema::{FieldId, cast::CAST_OPTIONS},
    },
};

/// Bytes a decoded string or binary value takes per row beside its data: a
/// view, as the scan reads non-indexed strings, and the most any of Arrow's
/// variable-width layouts takes.
const VIEW_BYTES_PER_ROW: usize = 16;

/// Bytes per row a string column kept as a dictionary takes: its `Int16` key.
const DICTIONARY_KEY_BYTES_PER_ROW: usize = 2;

/// A superfile's string column is kept as a dictionary when it holds at most
/// one value for this many rows ([`dictionary_columns`]): its keys then take
/// 2 bytes a row where views take 16, and its values a small share of the
/// rows' bytes.
const DICTIONARY_ROWS_PER_VALUE: usize = 16;

/// How far under the most values `Int16` keys reach a distinct-sketch
/// estimate must sit for its column to be kept as a dictionary: the sketch
/// counts within about 3.25% (one standard error), so half the keys' reach
/// leaves a row group whose values outrun its keys no real chance. An exact
/// value table needs no margin.
const SKETCH_KEY_MARGIN: usize = 2;

/// Metric counting the row groups a scan decoded because the cache lacked
/// one of their projected columns, as `EXPLAIN ANALYZE` lists it. A run over
/// columns the cache keeps decodes none.
const ROW_GROUPS_DECODED: &str = "row_groups_decoded";

/// Metric counting the row groups a scan read only the chosen rows of, from
/// their pages, keeping nothing, as `EXPLAIN ANALYZE` lists it.
const ROW_GROUPS_READ_IN_PART: &str = "row_groups_read_in_part";

/// How `EXPLAIN ANALYZE` names the rows of the units a scan read.
const ROWS_READ: &str = "rows_read";

/// How `EXPLAIN ANALYZE` names the row groups the filters that tighten as
/// the statement runs ruled out before they were read.
const ROW_GROUPS_SKIPPED: &str = "row_groups_skipped";

/// How `EXPLAIN ANALYZE` names the rows those filters dropped from the
/// batches the scan read.
const ROWS_DROPPED: &str = "rows_dropped";

/// One file of the scan: what its row groups are read from.
#[derive(Debug)]
pub(crate) struct CachedFile {
    pub(crate) uri: SuperfileUri,
    pub(crate) reader: Arc<SuperfileReader>,
    /// The footer [`RowGroupFilter`] prunes by: the file's own, without the
    /// bounds that leave out a float column's NaNs, as the Parquet scan
    /// prunes by
    /// ([`without_nan_blind_bounds`](crate::supertable::query::nan_bounds::without_nan_blind_bounds)).
    pub(crate) footer: Arc<ParquetMetaData>,
    /// Rows in each row group, in file order.
    pub(crate) row_counts: Arc<[u32]>,
    /// The file's deleted rows, by position in the file.
    pub(crate) tombstones: Arc<RoaringBitmap>,
    /// The only rows to read, by position in the file, when an index chose
    /// them (an `ORDER BY ... LIMIT` over a range reads the rows up to its
    /// cutoff); `None` reads every live row. Live rows only: none of them is
    /// in [`Self::tombstones`].
    pub(crate) selected: Option<RoaringBitmap>,
}

/// The scan's `WHERE`, for skipping the row groups whose statistics show it
/// can't match, as the Parquet scan skips them.
#[derive(Debug)]
pub(crate) struct RowGroupFilter {
    /// The `WHERE`, written against `table`.
    pub(crate) predicate: Arc<dyn PhysicalExpr>,
    pub(crate) table: SchemaRef,
    /// Maps the `WHERE` onto each file's own schema, as the Parquet scan's
    /// adapter does.
    pub(crate) adapter: Arc<dyn PhysicalExprAdapterFactory>,
}

impl RowGroupFilter {
    /// Which of `file`'s row groups the statistics leave to read; `None`
    /// when the `WHERE` doesn't map onto the file, and every row group is
    /// read.
    fn plan(&self, file: &CachedFile) -> Option<ParquetAccessPlan> {
        let pruning = file_pruning(&self.predicate, &self.adapter, &self.table, &file.footer)?;
        let all = ParquetAccessPlan::new_all(file.row_counts.len());
        Some(statistics_pruned(
            all,
            &file.footer,
            &pruning,
            &file.uri.0.to_string(),
        ))
    }
}

/// One row group of one file: the unit a partition reads.
#[derive(Debug, Clone)]
struct RowGroupUnit {
    file: Arc<CachedFile>,
    row_group: usize,
    /// The row group's rows, by position in the file.
    rows: Range<u32>,
    /// Whether a read that lacks a column decodes the row group whole and
    /// keeps its columns: it is read whole, the scan chose at least half its
    /// live rows, or a scan read some of its rows before. Otherwise such a
    /// read takes the chosen rows from the pages and keeps nothing.
    keep: bool,
}

/// What every partition reads with.
#[derive(Debug)]
struct ScanContext {
    /// The scan's output: the projected columns, typed as the scan schema
    /// types them.
    schema: SchemaRef,
    manifest: Arc<ManifestSnapshot>,
    op_stats: Option<Arc<OpStatsCollector>>,
    cache: Arc<DecodedColumnCache>,
    schema_id: u32,
    /// For each file, which of `schema`'s columns are kept as dictionaries
    /// ([`dictionary_columns`]).
    dictionaries: HashMap<SuperfileUri, Arc<[bool]>>,
    /// The `WHERE` the scan checks before it hands rows on
    /// ([`CachedScanExec::with_predicate`]): over the kept dictionaries
    /// themselves, so only the rows that pass are made strings.
    predicate: Option<Arc<dyn PhysicalExpr>>,
}

/// A table scan served from the decoded-column cache; see the module docs.
#[derive(Debug, Clone)]
pub(crate) struct CachedScanExec {
    context: Arc<ScanContext>,
    units: Arc<[RowGroupUnit]>,
    /// Row groups the `WHERE` ruled out by their statistics.
    pruned: usize,
    partitions: usize,
    fetch: Option<usize>,
    statistics: Arc<Statistics>,
    properties: Arc<PlanProperties>,
    /// The Parquet scan this one stands in for, metered as the provider
    /// builds it. A plan that reads better through the Parquet scan, an
    /// `ORDER BY ... LIMIT` whose row filter decodes the wide columns only
    /// for the rows that pass, takes it back ([`Self::parquet`]).
    parquet: Arc<dyn ExecutionPlan>,
    /// Filters a parent pushed down that tighten while the statement runs
    /// ([`LiveFilters`]); `None` without any.
    live: Option<Arc<LiveFilters>>,
    /// Set when the scan took a sort-order pushdown, so it reads its row
    /// groups best first for a sort above it — what a top-k asks for. Only
    /// then are rows dropped per batch by the tightening bound; a hash join
    /// or aggregate pushes a filter but asks for no order, so its filter
    /// skips whole row groups ([`LiveFilters`]) but drops no rows.
    reads_best_first: bool,
    /// The `EXPLAIN ANALYZE` counters: the rows each partition hands on, the
    /// row groups decoded ([`ROW_GROUPS_DECODED`]) and read in part
    /// ([`ROW_GROUPS_READ_IN_PART`]), the rows of the units read
    /// ([`ROWS_READ`]), the row groups skipped ([`ROW_GROUPS_SKIPPED`]) and
    /// the rows dropped ([`ROWS_DROPPED`]).
    metrics: ExecutionPlanMetricsSet,
}

/// Filters pushed down from above that tighten as the statement runs: a
/// top-k's bound on its sort key (the k-th best value so far), a hash join's
/// filter on its probe side, a `MIN` or `MAX` aggregate's best value so far.
/// Each is checked against a row group's statistics just before the row
/// group is read ([`RowGroupGate`]), and a row group that can no longer
/// reach the result is skipped. When the scan reads best first for a sort
/// above it — what a top-k asks for — the tightening bound is also checked
/// against each batch the scan hands over, and the rows it rules out are
/// dropped before the node above sees them; a join's or aggregate's filter,
/// which asks for no order, only skips row groups. A row such a filter rules
/// out can't change its parent's answer, so dropping it changes no answer;
/// the rows that pass are still checked above.
#[derive(Debug)]
struct LiveFilters {
    filters: Vec<Arc<dyn PhysicalExpr>>,
    /// Maps a filter onto each file's own schema, as the scan's `WHERE` is.
    adapter: Arc<dyn PhysicalExprAdapterFactory>,
}

impl LiveFilters {
    /// `batch` without the rows the filters as they stand rule out. A filter
    /// that rules nothing out yet (a top-k's before its first `k` rows) is
    /// not evaluated.
    fn keep(&self, mut batch: RecordBatch) -> DfResult<RecordBatch> {
        for filter in &self.filters {
            let now = snapshot_physical_expr(Arc::clone(filter))?;
            if now
                .downcast_ref::<Literal>()
                .is_some_and(|literal| literal.value() == &ScalarValue::Boolean(Some(true)))
            {
                continue;
            }
            let mask = now.evaluate(&batch)?.into_array(batch.num_rows())?;
            let mask = mask
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Internal("a live filter is not boolean".to_owned())
                })?;
            batch = filter_record_batch(&batch, mask)?;
        }
        Ok(batch)
    }
}

/// What [`RowGroupGate`] holds for one file: its schema, read once, and for
/// each filter the generation its pruning was built at and the pruning.
struct FileGate {
    /// The file's schema its statistics are read by ([`file_schema`]); `None`
    /// when it doesn't parse, and nothing on the file is pruned.
    schema: Option<SchemaRef>,
    /// Each filter's generation when its pruning was built, and the pruning:
    /// `None` for a filter that rules nothing out yet (a top-k's before its
    /// first `k` rows) or doesn't map onto the file.
    filters: Vec<(Option<u64>, Option<PruningPredicate>)>,
}

/// [`LiveFilters`] checked against the statistics of one row group at a
/// time, for one partition of the scan. A file's schema is parsed once; each
/// filter's pruning is built for a file once, and again only when the filter
/// has tightened since. A check reads the statistics of the one row group it
/// decides.
struct RowGroupGate {
    live: Arc<LiveFilters>,
    /// The scan's output schema, the filters are written against.
    table: SchemaRef,
    /// Each file's schema and per-filter pruning ([`FileGate`]), filled in as
    /// the file's row groups first come up and the filters tighten.
    built: HashMap<SuperfileUri, FileGate>,
}

impl RowGroupGate {
    fn new(live: Arc<LiveFilters>, table: SchemaRef) -> Self {
        Self {
            live,
            table,
            built: HashMap::new(),
        }
    }

    /// Whether the filters as they stand rule `unit`'s row group out by its
    /// statistics.
    fn skip(&mut self, unit: &RowGroupUnit) -> bool {
        let Self { live, table, built } = self;
        let live: &LiveFilters = live;
        let table: &SchemaRef = table;
        let file = &unit.file;
        let gate = built.entry(file.uri).or_insert_with(|| FileGate {
            schema: file_schema(table, &file.footer),
            filters: live.filters.iter().map(|_| (None, None)).collect(),
        });
        let Some(schema) = &gate.schema else {
            // The file's schema didn't parse; nothing on it is pruned.
            return false;
        };
        live.filters
            .iter()
            .zip(gate.filters.iter_mut())
            .any(|(filter, (generation, pruning))| {
                let now = snapshot_generation(filter);
                if *generation != Some(now) {
                    *generation = Some(now);
                    *pruning = snapshot_physical_expr(Arc::clone(filter))
                        .ok()
                        .and_then(|expr| pruning_for(&expr, &live.adapter, table, schema))
                        .filter(|pruning| !pruning.always_true());
                }
                pruning.as_ref().is_some_and(|pruning| {
                    let statistics = OneRowGroup {
                        footer: &file.footer,
                        row_group: unit.row_group,
                        schema,
                    };
                    pruning
                        .prune(&statistics)
                        .is_ok_and(|keep| keep.first() == Some(&false))
                })
            })
    }
}

/// The statistics of one row group of a file, for pruning, read as
/// DataFusion's Parquet scan reads a file's row groups.
struct OneRowGroup<'a> {
    footer: &'a ParquetMetaData,
    row_group: usize,
    /// The file's schema, as its pruning was built against.
    schema: &'a Schema,
}

impl<'a> OneRowGroup<'a> {
    fn converter(&self, column: &StatsColumn) -> Option<StatisticsConverter<'a>> {
        StatisticsConverter::try_new(
            &column.name,
            self.schema,
            self.footer.file_metadata().schema_descr(),
        )
        .ok()
        .map(|converter| converter.with_missing_null_counts_as_zero(true))
    }

    fn group(&self) -> IntoIter<&'a RowGroupMetaData> {
        self.footer.row_groups().get(self.row_group).into_iter()
    }
}

impl PruningStatistics for OneRowGroup<'_> {
    fn min_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?.row_group_mins(self.group()).ok()
    }

    fn max_values(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?.row_group_maxes(self.group()).ok()
    }

    fn num_containers(&self) -> usize {
        1
    }

    fn null_counts(&self, column: &StatsColumn) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_null_counts(self.group())
            .ok()
            .map(|counts| Arc::new(counts) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        let rows = self.footer.row_groups().get(self.row_group)?.num_rows();
        let rows = u64::try_from(rows).ok()?;
        Some(Arc::new(UInt64Array::from(vec![rows])) as ArrayRef)
    }

    fn contained(
        &self,
        _column: &StatsColumn,
        _values: &HashSet<ScalarValue>,
    ) -> Option<BooleanArray> {
        None
    }
}

impl CachedScanExec {
    /// A scan of `schema`'s columns over `files`, read as up to
    /// `target_partitions` partitions, skipping the row groups `filter` rules
    /// out, standing in for `parquet`. `None`, and the Parquet scan serves the
    /// statement, when:
    /// - a column is not one the files store in a type that converts to
    ///   `schema`'s (the Parquet scan's schema adapter serves those);
    /// - the columns it would decode don't fit the cache ([`decoded_bytes`]):
    ///   kept, they would push each other out, and every run would decode
    ///   them all again where the Parquet scan decodes only what it reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        schema: SchemaRef,
        files: Vec<CachedFile>,
        filter: Option<RowGroupFilter>,
        manifest: Arc<ManifestSnapshot>,
        op_stats: Option<Arc<OpStatsCollector>>,
        cache: Arc<DecodedColumnCache>,
        target_partitions: usize,
        statistics: Statistics,
        parquet: Arc<dyn ExecutionPlan>,
    ) -> Option<Self> {
        let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        let stored = hit_output_schema(&manifest.stored_schema(), &names).ok()?;
        let convertible = stored
            .fields()
            .iter()
            .zip(schema.fields())
            .all(|(from, to)| can_cast_types(from.data_type(), to.data_type()));
        if !convertible {
            return None;
        }
        let schema_id = manifest.table_schema().schema_id();
        let mut units = Vec::new();
        let mut pruned = 0;
        for file in files.into_iter().map(Arc::new) {
            let plan = filter.as_ref().and_then(|filter| filter.plan(&file));
            let mut first = 0u32;
            for (row_group, &n) in file.row_counts.iter().enumerate() {
                let rows = first..first + n;
                first += n;
                if plan
                    .as_ref()
                    .is_some_and(|plan| !plan.should_scan(row_group))
                {
                    pruned += 1;
                    continue;
                }
                // A row group every row of which is deleted reads nothing, nor
                // does one holding none of the rows the index chose.
                let live = u64::from(n) - file.tombstones.range_cardinality(rows.clone());
                let chosen = file
                    .selected
                    .as_ref()
                    .map_or(live, |selected| selected.range_cardinality(rows.clone()));
                if chosen == 0 {
                    continue;
                }
                let keep = file.selected.is_none()
                    || chosen.saturating_mul(2) >= live
                    || cache.read_in_part_before(&RowGroupKey {
                        uri: file.uri,
                        schema_id,
                        row_group,
                    });
                units.push(RowGroupUnit {
                    file: Arc::clone(&file),
                    row_group,
                    rows,
                    keep,
                });
            }
        }
        let dictionaries = dictionary_columns(&manifest, &schema, &units);
        if decoded_bytes(
            units.iter().filter(|unit| unit.keep),
            &schema,
            &dictionaries,
        ) > cache.max_bytes()
        {
            return None;
        }
        let partitions = target_partitions.clamp(1, units.len().max(1));
        let properties = plan_properties(&schema, partitions);
        Some(Self {
            context: Arc::new(ScanContext {
                schema,
                manifest,
                op_stats,
                cache,
                schema_id,
                dictionaries,
                predicate: None,
            }),
            units: units.into(),
            pruned,
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
            parquet,
            live: None,
            reads_best_first: false,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    /// The Parquet scan this one stands in for, metered as the provider
    /// builds it.
    pub(crate) fn parquet(&self) -> &Arc<dyn ExecutionPlan> {
        &self.parquet
    }
}

/// About how many bytes the columns of `schema` take decoded over `units`'
/// row groups (those a scan would keep), from the footers:
/// - a fixed-width column, its width per row;
/// - any other, [`VIEW_BYTES_PER_ROW`] per row plus its values' bytes, which
///   the footer records unencoded (a dictionary-encoded column stores far
///   fewer bytes than it decodes to), or failing that its chunk's
///   uncompressed size.
///
/// A column a file holds under another name, after a rename, isn't found by
/// this name and counts as nothing; the estimate decides only which scan
/// serves the statement, never what it returns.
fn decoded_bytes<'a>(
    units: impl Iterator<Item = &'a RowGroupUnit>,
    schema: &Schema,
    dictionaries: &HashMap<SuperfileUri, Arc<[bool]>>,
) -> usize {
    units
        .map(|unit| {
            let dictionary = dictionaries.get(&unit.file.uri);
            let meta = unit.file.reader.parquet_metadata();
            let Some(row_group) = meta.row_groups().get(unit.row_group) else {
                return 0;
            };
            let rows = usize::try_from(row_group.num_rows()).unwrap_or(0);
            schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, field)| {
                    let Some(chunk) = row_group.columns().iter().find(|chunk| {
                        chunk
                            .column_descr()
                            .path()
                            .parts()
                            .first()
                            .is_some_and(|root| root == field.name())
                    }) else {
                        return 0;
                    };
                    if dictionary.is_some_and(|kept| kept[i]) {
                        // The keys, and at most one value in
                        // `DICTIONARY_ROWS_PER_VALUE` rows.
                        let values = chunk
                            .unencoded_byte_array_data_bytes()
                            .unwrap_or_else(|| chunk.uncompressed_size());
                        return rows
                            .saturating_mul(DICTIONARY_KEY_BYTES_PER_ROW)
                            .saturating_add(
                                usize::try_from(values).unwrap_or(0) / DICTIONARY_ROWS_PER_VALUE,
                            );
                    }
                    match field.data_type().primitive_width() {
                        Some(width) => rows.saturating_mul(width),
                        None => {
                            let values = chunk
                                .unencoded_byte_array_data_bytes()
                                .unwrap_or_else(|| chunk.uncompressed_size());
                            rows.saturating_mul(VIEW_BYTES_PER_ROW)
                                .saturating_add(usize::try_from(values).unwrap_or(0))
                        }
                    }
                })
                .sum::<usize>()
        })
        .sum()
}

/// For each file `units` read, which of `schema`'s columns the cache keeps as
/// dictionaries: the string columns whose superfile statistics count at most
/// one value per [`DICTIONARY_ROWS_PER_VALUE`] rows, and no more values than
/// `Int16` keys reach. The superfile's exact value table counts them where
/// it keeps one, and its distinct sketch estimates them otherwise; a column
/// with neither is kept as the scan reads it. The superfiles are written
/// without Parquet dictionaries, so the dictionary is built once, when the
/// row group is first decoded, and every later statement reads it kept.
fn dictionary_columns(
    manifest: &ManifestSnapshot,
    schema: &Schema,
    units: &[RowGroupUnit],
) -> HashMap<SuperfileUri, Arc<[bool]>> {
    let table = manifest.table_schema();
    let fields: Vec<Option<FieldId>> = schema
        .fields()
        .iter()
        .map(|field| {
            matches!(
                field.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
            )
            .then(|| table.id_of(field.name()))
            .flatten()
        })
        .collect();
    if fields.iter().all(Option::is_none) {
        return HashMap::new();
    }
    let read: HashSet<SuperfileUri> = units.iter().map(|unit| unit.file.uri).collect();
    manifest
        .superfiles
        .iter()
        .filter(|entry| read.contains(&entry.uri))
        .map(|entry| {
            let rows = usize::try_from(entry.n_docs).unwrap_or(usize::MAX);
            let kept: Arc<[bool]> = fields
                .iter()
                .map(|field| {
                    let Some(stats) = field.and_then(|id| entry.scalar_stats.column(id)) else {
                        return false;
                    };
                    // The values, and what they must leave under the keys'
                    // reach: an estimate keeps a margin, an exact count none.
                    let distinct = match stats.value_table_len() {
                        Some(exact) => Some((exact, exact)),
                        None => stats
                            .sketch()
                            .and_then(HllSketch::from_bytes)
                            .map(|sketch| sketch.estimate().ceil() as usize)
                            .map(|estimate| (estimate, estimate.saturating_mul(SKETCH_KEY_MARGIN))),
                    };
                    distinct.is_some_and(|(distinct, reach)| {
                        reach <= i16::MAX as usize
                            && distinct.saturating_mul(DICTIONARY_ROWS_PER_VALUE) <= rows
                    })
                })
                .collect();
            (entry.uri, kept)
        })
        .collect()
}

/// `array` as the cache keeps a column of few values: a dictionary of
/// `Int16` keys over its strings. `None` when it does not take that type.
fn as_dictionary(array: &ArrayRef) -> Option<ArrayRef> {
    let keep = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));
    cast_with_options(array, &keep, &CAST_OPTIONS).ok()
}

/// A kept column as the scan types it: a kept dictionary as the scan's
/// strings, every other column as it is.
fn as_scanned(kept: ArrayRef, scanned: &DataType) -> DfResult<ArrayRef> {
    match kept.data_type() {
        DataType::Dictionary(..) if kept.data_type() != scanned => {
            Ok(cast_with_options(&kept, scanned, &CAST_OPTIONS)?)
        }
        _ => Ok(kept),
    }
}

/// `units` ordered by the row-group statistics of `column`, smallest minimum
/// first (largest maximum first when `descending`); a row group without a
/// statistic for the column, and one whose bounds don't cast, ordered where
/// `nulls_first` puts them, so a `NULLS FIRST` sort reads the row groups that
/// may hold its leading nulls first. `None` when no file has statistics for
/// it. Each file's bounds are cast to `data_type`, the type the scan reads
/// the column as, so files written before the column was widened order with
/// the rest.
fn best_first(
    units: &[RowGroupUnit],
    column: &str,
    data_type: &DataType,
    descending: bool,
    nulls_first: bool,
) -> Option<Vec<RowGroupUnit>> {
    // Each file's bounds, read and cast once, laid back to back in `all`,
    // then one trailing null row the units with no usable bound point at: a
    // file whose bounds don't cast, and a row group past the ones its file
    // has statistics for. The key column is one `take` from `all`, so each
    // file allocates once however many row groups it holds.
    let mut offsets: HashMap<SuperfileUri, Option<(usize, usize)>> = HashMap::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();
    let mut len = 0usize;
    for unit in units {
        offsets.entry(unit.file.uri).or_insert_with(|| {
            let array = row_group_bounds(&unit.file.footer, column, descending)
                .and_then(|array| cast_with_options(&array, data_type, &CAST_OPTIONS).ok())?;
            let at = len;
            let rows = array.len();
            len += rows;
            arrays.push(array);
            Some((at, rows))
        });
    }
    let null_at = len;
    arrays.push(new_null_array(data_type, 1));
    let arrays: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
    let all = concat(&arrays).ok()?;
    let indices = UInt64Array::from_iter_values(units.iter().map(|unit| {
        let at = offsets
            .get(&unit.file.uri)
            .copied()
            .flatten()
            .filter(|&(_, rows)| unit.row_group < rows)
            .map_or(null_at, |(at, _)| at + unit.row_group);
        at as u64
    }));
    let keys = take(all.as_ref(), &indices, None).ok()?;
    if keys.null_count() == keys.len() {
        return None;
    }
    // Arrow's sort orders every value of one type, NaNs and all.
    let options = SortOptions {
        descending,
        nulls_first,
    };
    let order = sort_to_indices(&keys, Some(options), None).ok()?;
    Some(
        order
            .values()
            .iter()
            .map(|&at| units[at as usize].clone())
            .collect(),
    )
}

/// Each row group's minimum (maximum when `descending`) of `column` in a
/// file with `footer`; `None` when the file does not hold it or records no
/// statistics a converter reads.
fn row_group_bounds(footer: &ParquetMetaData, column: &str, descending: bool) -> Option<ArrayRef> {
    let file = footer.file_metadata();
    let schema = parquet_to_arrow_schema(file.schema_descr(), file.key_value_metadata()).ok()?;
    let converter = StatisticsConverter::try_new(column, &schema, file.schema_descr()).ok()?;
    match descending {
        false => converter.row_group_mins(footer.row_groups().iter()).ok(),
        true => converter.row_group_maxes(footer.row_groups().iter()).ok(),
    }
}

/// The plan properties of a scan of `schema` read as `partitions` partitions.
fn plan_properties(schema: &SchemaRef, partitions: usize) -> Arc<PlanProperties> {
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new(Arc::clone(schema)),
        Partitioning::UnknownPartitioning(partitions),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ))
}

/// Which of `rows` (positions in the file) are in `selected`, one entry per
/// row of the range.
fn selected_mask(selected: &RoaringBitmap, rows: &Range<u32>) -> BooleanArray {
    let n = rows.len();
    let mut mask = BooleanBufferBuilder::new(n);
    mask.append_n(n, false);
    for row in selected.range(rows.clone()) {
        mask.set_bit((row - rows.start) as usize, true);
    }
    BooleanArray::new(mask.finish(), None)
}

/// `array` typed as `target`, the scan schema's type for its column.
fn as_scan_type(array: &ArrayRef, target: &DataType) -> DfResult<ArrayRef> {
    Ok(if array.data_type() == target {
        Arc::clone(array)
    } else {
        cast_with_options(array, target, &CAST_OPTIONS)?
    })
}

/// `batch` as slices of at most `rows` rows each, sharing its buffers.
fn sliced(batch: RecordBatch, rows: usize) -> Vec<RecordBatch> {
    let rows = rows.max(1);
    let n = batch.num_rows();
    (0..n)
        .step_by(rows)
        .map(|at| batch.slice(at, rows.min(n - at)))
        .collect()
}

impl ScanContext {
    /// The cache's key for `unit`'s row group.
    fn row_group_key(&self, unit: &RowGroupUnit) -> RowGroupKey {
        RowGroupKey {
            uri: unit.file.uri,
            schema_id: self.schema_id,
            row_group: unit.row_group,
        }
    }

    /// The cache's key for `column` of `unit`'s row group.
    fn key(&self, unit: &RowGroupUnit, column: &str) -> ColumnKey {
        ColumnKey {
            uri: unit.file.uri,
            schema_id: self.schema_id,
            column: column.to_owned(),
            row_group: unit.row_group,
        }
    }

    /// The `missing` projected columns of `unit`'s row group, decoded, typed
    /// as the scan schema types them, and kept in the cache.
    async fn decode_and_keep(
        self: Arc<Self>,
        unit: RowGroupUnit,
        missing: Vec<usize>,
    ) -> DfResult<Vec<ArrayRef>> {
        let fields = self.schema.fields();
        let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
        // Read as the scan types them, so a string the scan works on as a
        // view decodes into one directly.
        let typed = Arc::new(
            self.schema
                .project(&missing)
                .map_err(QueryError::internal)?,
        );
        let read = read_file_row_group(
            &self.manifest,
            &unit.file.reader,
            unit.row_group,
            &names,
            &typed,
            &self.op_stats,
        )
        .await?;
        let mut decoded = Vec::with_capacity(missing.len());
        let dictionary = self.dictionaries.get(&unit.file.uri);
        for (&i, array) in missing.iter().zip(read.columns()) {
            let array = as_scan_type(array, fields[i].data_type())?;
            // A column of few values is kept as a dictionary; this statement
            // reads the strings it decoded.
            let kept = match dictionary.is_some_and(|kept| kept[i]) {
                true => as_dictionary(&array).unwrap_or_else(|| Arc::clone(&array)),
                false => Arc::clone(&array),
            };
            self.cache.insert(
                self.key(&unit, fields[i].name()),
                Arc::clone(&kept),
                kept.get_array_memory_size(),
            );
            decoded.push(array);
        }
        Ok(decoded)
    }

    /// `unit`'s row group, every projected column, without its deleted rows
    /// and, when the index chose the file's rows, without the rows it didn't.
    /// A decode of the columns the cache lacks counts in `decoded`; a read of
    /// just the chosen rows from the pages, keeping nothing, in `read_in_part`.
    async fn read(
        self: &Arc<Self>,
        unit: &RowGroupUnit,
        decoded: &Count,
        read_in_part: &Count,
    ) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
        // As kept: a column kept as a dictionary stays one until the rows
        // the predicate keeps are known.
        let mut columns: Vec<Option<ArrayRef>> = fields
            .iter()
            .map(|f| self.cache.get(&self.key(unit, f.name())))
            .collect();
        let missing: Vec<usize> = (0..columns.len())
            .filter(|&i| columns[i].is_none())
            .collect();
        if !missing.is_empty()
            && !unit.keep
            && let Some(selected) = &unit.file.selected
        {
            read_in_part.add(1);
            return self
                .read_chosen_rows(unit, selected, columns, missing)
                .await;
        }
        if !missing.is_empty() {
            decoded.add(1);
            // The decode runs as its own task: a `LIMIT` above drops this
            // partition's stream once it has its rows, and a decode dropped
            // halfway would have spent its CPU without keeping the columns.
            // Run to the end, it keeps them for the next statement.
            let decoded = spawn(
                Arc::clone(self)
                    .decode_and_keep(unit.clone(), missing.clone())
                    .in_current_span(),
            )
            .await
            .map_err(QueryError::internal)??;
            for (i, array) in missing.into_iter().zip(decoded) {
                columns[i] = Some(array);
            }
        }
        let columns = columns
            .into_iter()
            .map(|c| c.ok_or_else(|| DataFusionError::Internal("a column was not read".into())))
            .collect::<DfResult<Vec<_>>>()?;
        let n = unit.rows.len();
        let start = unit.rows.start;
        // The chosen rows are live, so they need no check against the
        // tombstones.
        let keep = match &unit.file.selected {
            Some(selected) => Some(selected_mask(selected, &unit.rows)),
            None => {
                let mut keep: Option<BooleanBufferBuilder> = None;
                for row in unit.file.tombstones.range(unit.rows.clone()) {
                    let mask = keep.get_or_insert_with(|| {
                        let mut mask = BooleanBufferBuilder::new(n);
                        mask.append_n(n, true);
                        mask
                    });
                    mask.set_bit((row - start) as usize, false);
                }
                keep.map(|mut mask| BooleanArray::new(mask.finish(), None))
            }
        };
        let live = keep;
        let keep = match &self.predicate {
            Some(predicate) => passing_rows(predicate, &self.schema, &columns, live, n),
            None => live,
        };
        let rows = keep.as_ref().map_or(n, BooleanArray::true_count);
        let columns = match &keep {
            Some(mask) => columns
                .iter()
                .map(|column| filter(column.as_ref(), mask))
                .collect::<Result<Vec<_>, _>>()?,
            None => columns,
        };
        let columns = columns
            .into_iter()
            .zip(fields)
            .map(|(column, field)| as_scanned(column, field.data_type()))
            .collect::<DfResult<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?)
    }

    /// The `selected` rows of `unit`'s row group: the `missing` columns read
    /// from the pages for just those rows and not kept, the others taken from
    /// the kept `columns`. The cache remembers the row group, so the next scan
    /// that chooses rows in it decodes and keeps its columns.
    async fn read_chosen_rows(
        &self,
        unit: &RowGroupUnit,
        selected: &RoaringBitmap,
        columns: Vec<Option<ArrayRef>>,
        missing: Vec<usize>,
    ) -> DfResult<RecordBatch> {
        self.cache.note_read_in_part(self.row_group_key(unit));
        let fields = self.schema.fields();
        let rows: Vec<u32> = selected.range(unit.rows.clone()).collect();
        let chosen = rows.len();
        let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
        let stored = hit_output_schema(&self.manifest.stored_schema(), &names)?;
        let taken = take_file_columns(
            &self.manifest,
            &unit.file.reader,
            rows,
            &names,
            &stored,
            &self.op_stats,
        )
        .await?;
        if taken.num_rows() != chosen {
            return Err(DataFusionError::Internal(format!(
                "read {} rows of a row group for {chosen} chosen ones",
                taken.num_rows()
            )));
        }
        let mask = selected_mask(selected, &unit.rows);
        // `missing` lists the absent columns in schema order, as the take
        // returned them.
        let mut taken = taken.columns().iter();
        let columns = columns
            .into_iter()
            .zip(fields)
            .map(|(column, field)| match column {
                // A column kept as a dictionary comes out as the scan's
                // strings.
                Some(kept) => as_scanned(filter(kept.as_ref(), &mask)?, field.data_type()),
                None => {
                    let array = taken
                        .next()
                        .ok_or_else(|| DataFusionError::Internal("a column was not read".into()))?;
                    as_scan_type(array, field.data_type())
                }
            })
            .collect::<DfResult<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(chosen)),
        )
        .map_err(DataFusionError::from)
    }
}

/// Which of a row group's `rows` rows the scan hands on: those `live` keeps
/// (every row without it) that pass the conjuncts of `predicate` the scan
/// checks, over `columns` as kept. Never null.
///
/// The filter above checks every row it gets, and DataFusion's `AND` checks
/// its right side on every row of the batch, not only on the rows its left
/// side keeps, once the left keeps more than a fifth of them. Handed fewer
/// rows than the files hold, the filter can so check a conjunct on a row an
/// earlier conjunct drops, where without a cache it wouldn't: a guard like
/// `x <> 0 AND 10 / x > 1` would fail. So when a conjunct reads a column kept
/// as a dictionary, the scan
/// - checks every conjunct, in order, each on the rows that passed the ones
///   before it, as `AND` would, and never on a deleted or unselected row. The
///   rows it hands on pass all of them, so the filter checks each conjunct
///   only on rows it has already been checked on;
/// - leaves the last conjunct to the filter when it reads no kept
///   dictionary: the filter checks it once, on rows that pass every conjunct
///   before it;
/// - checks a conjunct on one kept dictionary once per value the rows use,
///   and any other on the rows themselves;
/// - stops at a conjunct that fails to evaluate on the rows, and hands on the
///   rows it has: the filter then checks that conjunct on the same rows, and
///   fails as it does with no scan below it.
///
/// A row group whose dictionaries no conjunct reads is handed on as it is.
fn passing_rows(
    predicate: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    columns: &[ArrayRef],
    live: Option<BooleanArray>,
    rows: usize,
) -> Option<BooleanArray> {
    let conjuncts: Vec<(&Arc<dyn PhysicalExpr>, Vec<usize>, bool)> = split_conjunction(predicate)
        .into_iter()
        .map(|conjunct| {
            let mut used: Vec<usize> = collect_columns(conjunct)
                .iter()
                .map(|column| column.index())
                .collect();
            used.sort_unstable();
            used.dedup();
            let on_dictionary = used
                .iter()
                .any(|&i| kept_as_dictionary(&columns[i], schema.field(i)));
            (conjunct, used, on_dictionary)
        })
        .collect();
    if !conjuncts.iter().any(|(_, _, on_dictionary)| *on_dictionary) {
        return live;
    }
    let last = conjuncts.len() - 1;
    let mut passing = live.unwrap_or_else(|| BooleanArray::from(vec![true; rows]));
    for (at, (conjunct, used, on_dictionary)) in conjuncts.iter().enumerate() {
        if at == last && !on_dictionary {
            break;
        }
        if passing.true_count() == 0 {
            break;
        }
        let on_rows = || conjunct_on_rows(conjunct, schema, columns, used, &passing);
        let checked = match used.as_slice() {
            [only] if *on_dictionary => {
                dictionary_conjunct(conjunct, schema.field(*only), &columns[*only], &passing)
                    .or_else(|_| on_rows())
            }
            _ => on_rows(),
        };
        match checked {
            Ok(next) => passing = next,
            Err(_) => break,
        }
    }
    Some(passing)
}

/// Whether `column`, as kept, is a dictionary of a column the scan returns
/// as something else.
fn kept_as_dictionary(column: &ArrayRef, field: &Field) -> bool {
    matches!(column.data_type(), DataType::Dictionary(..))
        && !matches!(field.data_type(), DataType::Dictionary(..))
}

/// The rows of `passing` that pass `conjunct`, which reads only the column
/// `field`, kept as the dictionary `kept`: the conjunct is checked once on
/// each value the passing rows use (cast as the scan types the column), and
/// once on a null when a passing row has none, then taken for each row by its
/// key. Values only other rows use, deleted ones among them, are not checked.
fn dictionary_conjunct(
    conjunct: &Arc<dyn PhysicalExpr>,
    field: &Field,
    kept: &ArrayRef,
    passing: &BooleanArray,
) -> DfResult<BooleanArray> {
    let Some(dictionary) = kept.as_dictionary_opt::<Int16Type>() else {
        return Err(DataFusionError::Internal(format!(
            "column {} is kept as {}, not a dictionary of Int16 keys",
            field.name(),
            kept.data_type()
        )));
    };
    let keys = dictionary.keys();
    // Each value a passing row uses, once, and where it sits among them.
    let mut slot: Vec<Option<usize>> = vec![None; dictionary.values().len()];
    let mut used: Vec<u32> = Vec::new();
    let mut null_row = false;
    for row in passing.values().set_indices() {
        if keys.is_null(row) {
            null_row = true;
            continue;
        }
        let value = keys.value(row) as usize;
        if slot[value].is_none() {
            slot[value] = Some(used.len());
            used.push(value as u32);
        }
    }
    let one = Arc::new(Schema::new(vec![field.clone().with_nullable(true)]));
    let expr = reassign_expr_columns(Arc::clone(conjunct), &one)?;
    let check = |values: ArrayRef| -> DfResult<BooleanArray> {
        let n = values.len();
        let batch = RecordBatch::try_new(Arc::clone(&one), vec![values])?;
        boolean(expr.evaluate(&batch)?.into_array(n)?)
    };
    let values = take(dictionary.values().as_ref(), &UInt32Array::from(used), None)?;
    let on_values = check(cast_with_options(
        &values,
        field.data_type(),
        &CAST_OPTIONS,
    )?)?;
    let on_null = match null_row {
        true => {
            let on_null = check(new_null_array(field.data_type(), 1))?;
            on_null.is_valid(0) && on_null.value(0)
        }
        false => false,
    };
    let mut next = BooleanBufferBuilder::new(passing.len());
    next.append_n(passing.len(), false);
    for row in passing.values().set_indices() {
        let passes = match keys.is_null(row) {
            true => on_null,
            false => {
                let at = slot[keys.value(row) as usize].expect("a passing row's value is checked");
                on_values.is_valid(at) && on_values.value(at)
            }
        };
        if passes {
            next.set_bit(row, true);
        }
    }
    Ok(BooleanArray::new(next.finish(), None))
}

/// The rows of `passing` that pass `conjunct`, which reads the columns
/// `used`, checked on those rows alone, a column kept as a dictionary among
/// them cast to the scan's type first.
fn conjunct_on_rows(
    conjunct: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
    columns: &[ArrayRef],
    used: &[usize],
    passing: &BooleanArray,
) -> DfResult<BooleanArray> {
    let projected = Arc::new(Schema::new(
        used.iter()
            .map(|&i| schema.field(i).clone())
            .collect::<Vec<_>>(),
    ));
    let arrays = used
        .iter()
        .map(|&i| {
            let rows = filter(columns[i].as_ref(), passing)?;
            as_scanned(rows, schema.field(i).data_type())
        })
        .collect::<DfResult<Vec<_>>>()?;
    let n = passing.true_count();
    let batch = RecordBatch::try_new_with_options(
        Arc::clone(&projected),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(n)),
    )?;
    let expr = reassign_expr_columns(Arc::clone(conjunct), &projected)?;
    let checked = boolean(expr.evaluate(&batch)?.into_array(n)?)?;
    let mut next = BooleanBufferBuilder::new(passing.len());
    next.append_n(passing.len(), false);
    for (at, row) in passing.values().set_indices().enumerate() {
        if checked.is_valid(at) && checked.value(at) {
            next.set_bit(row, true);
        }
    }
    Ok(BooleanArray::new(next.finish(), None))
}

/// `array` as the booleans a predicate gives.
fn boolean(array: ArrayRef) -> DfResult<BooleanArray> {
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| DataFusionError::Internal(format!("a predicate gave {}", array.data_type())))
}

impl CachedScanExec {
    /// This scan, checking `predicate` (a `WHERE` over its output, as the
    /// filter above it checks it) on each row group that keeps a column it
    /// reads as a dictionary, before handing the row group's rows on: its
    /// conjuncts in order, on the live rows, each on the rows the ones before
    /// it kept ([`passing_rows`]). A conjunct on one kept dictionary is
    /// checked once per value the rows use rather than once per row, and only
    /// the rows that pass are cast to the strings the scan returns. The
    /// filter above still checks every row it gets, so the answer is the one
    /// it gives.
    pub(crate) fn with_predicate(&self, predicate: Arc<dyn PhysicalExpr>) -> Self {
        let context = &self.context;
        Self {
            context: Arc::new(ScanContext {
                schema: Arc::clone(&context.schema),
                manifest: Arc::clone(&context.manifest),
                op_stats: context.op_stats.clone(),
                cache: Arc::clone(&context.cache),
                schema_id: context.schema_id,
                dictionaries: context.dictionaries.clone(),
                predicate: Some(predicate),
            }),
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.clone()
        }
    }

    /// Whether the scan checks a `WHERE` itself ([`Self::with_predicate`]).
    pub(crate) fn has_predicate(&self) -> bool {
        self.context.predicate.is_some()
    }
}

impl DisplayAs for CachedScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let columns: Vec<&str> = self
            .context
            .schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        write!(
            f,
            "CachedScanExec: row_groups={}, pruned={}, partitions={}, projection=[{}]",
            self.units.len(),
            self.pruned,
            self.partitions,
            columns.join(", ")
        )?;
        if let Some(predicate) = &self.context.predicate {
            write!(f, ", predicate={predicate}")?;
        }
        if let Some(fetch) = self.fetch {
            write!(f, ", fetch={fetch}")?;
        }
        Ok(())
    }
}

impl ExecutionPlan for CachedScanExec {
    fn name(&self) -> &'static str {
        "CachedScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        Ok(match partition {
            None => Arc::clone(&self.statistics),
            Some(_) => Arc::new(Statistics::new_unknown(&self.context.schema)),
        })
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            fetch: limit,
            metrics: ExecutionPlanMetricsSet::new(),
            ..self.clone()
        }))
    }

    /// Keeps the filters a parent pushed down that tighten as the statement
    /// runs ([`LiveFilters`]: a top-k's, a hash join's, a `MIN` or `MAX`
    /// aggregate's), to skip row groups and drop rows by; every filter stays
    /// checked above.
    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DfResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let filters: Vec<Arc<dyn PhysicalExpr>> = child_pushdown_result
            .parent_filters
            .iter()
            .map(|result| Arc::clone(&result.filter))
            .collect();
        let none = vec![PushedDown::No; filters.len()];
        let live: Vec<Arc<dyn PhysicalExpr>> = filters
            .into_iter()
            .filter(|filter| filter.downcast_ref::<DynamicFilterPhysicalExpr>().is_some())
            .collect();
        let propagation = FilterPushdownPropagation::with_parent_pushdown_result(none);
        if phase != FilterPushdownPhase::Post || live.is_empty() {
            return Ok(propagation);
        }
        Ok(propagation.with_updated_node(Arc::new(Self {
            live: Some(Arc::new(LiveFilters {
                filters: live,
                adapter: Arc::new(TableExprAdapterFactory::new(&self.context.manifest)),
            })),
            ..self.clone()
        })))
    }

    /// For a sort on one of the scan's columns, reads the row groups that
    /// statistics say start best first, so a top-k above settles its bound
    /// early and [`LiveFilters`] skip the rest. The order is only a reading
    /// order: the sort stays.
    fn try_pushdown_sort(
        &self,
        order: &[PhysicalSortExpr],
    ) -> DfResult<SortOrderPushdownResult<Arc<dyn ExecutionPlan>>> {
        let Some(first) = order.first() else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        let Some(column) = first.expr.downcast_ref::<Column>() else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        let Ok(field) = self.context.schema.field_with_name(column.name()) else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        let Some(units) = best_first(
            &self.units,
            column.name(),
            field.data_type(),
            first.options.descending,
            first.options.nulls_first,
        ) else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        Ok(SortOrderPushdownResult::Inexact {
            inner: Arc::new(Self {
                units: units.into(),
                reads_best_first: true,
                ..self.clone()
            }),
        })
    }

    fn execute(
        &self,
        partition: usize,
        task: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        if partition >= self.partitions {
            return Err(DataFusionError::Internal(format!(
                "CachedScanExec has {} partitions; asked for {partition}",
                self.partitions
            )));
        }
        let units: Vec<RowGroupUnit> = self
            .units
            .iter()
            .skip(partition)
            .step_by(self.partitions)
            .cloned()
            .collect();
        let context = Arc::clone(&self.context);
        let mut gate = self
            .live
            .clone()
            .map(|live| RowGroupGate::new(live, Arc::clone(&self.context.schema)));
        // Rows are dropped per batch only when the scan reads best first for
        // a sort above it, which a top-k asks for and a hash join or an
        // aggregate does not; it matches the Parquet scan, which filters rows
        // only under `ORDER BY ... LIMIT`. Their filters still skip whole row
        // groups above, but dropping their rows here, where most pass, is
        // work for nothing.
        let kept = self.reads_best_first.then(|| self.live.clone()).flatten();
        let output_rows = MetricBuilder::new(&self.metrics).output_rows(partition);
        let decoded = MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_DECODED, partition);
        let read_in_part =
            MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_READ_IN_PART, partition);
        let rows_read = MetricBuilder::new(&self.metrics).counter(ROWS_READ, partition);
        let skipped = MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_SKIPPED, partition);
        let dropped = MetricBuilder::new(&self.metrics).counter(ROWS_DROPPED, partition);
        // A row group goes out in batch-size slices, as the Parquet scan's
        // rows do: the filter above then works a batch at a time, and a
        // `LIMIT` met in the first slice leaves the rest of the row group.
        let batch_size = task.session_config().batch_size();
        let batches = stream::iter(units)
            .filter(move |unit| {
                // Checked as each row group comes up, so a bound tightened by
                // the rows read so far skips it.
                let skip = gate.as_mut().is_some_and(|gate| gate.skip(unit));
                if skip {
                    skipped.add(1);
                }
                future::ready(!skip)
            })
            .then(move |unit| {
                let context = Arc::clone(&context);
                let decoded = decoded.clone();
                let read_in_part = read_in_part.clone();
                rows_read.add(unit.rows.len());
                async move { context.read(&unit, &decoded, &read_in_part).await }
            })
            .map_ok(move |batch| stream::iter(sliced(batch, batch_size).into_iter().map(Ok)))
            .try_flatten()
            // Each slice as it goes out, so a bound the rows before it
            // tightened drops what can no longer reach the result.
            .and_then(move |batch| {
                future::ready(match &kept {
                    Some(live) => {
                        let rows = batch.num_rows();
                        live.keep(batch)
                            .inspect(|left| dropped.add(rows - left.num_rows()))
                    }
                    None => Ok(batch),
                })
            })
            .try_filter(|batch| future::ready(batch.num_rows() > 0))
            // A partition stops once it has the scan's fetch: the limit above
            // takes at most that many rows from any one of them.
            .scan(self.fetch.unwrap_or(usize::MAX), |remaining, batch| {
                future::ready(match batch {
                    Err(error) => Some(Err(error)),
                    Ok(_) if *remaining == 0 => None,
                    Ok(batch) => {
                        let rows = batch.num_rows().min(*remaining);
                        *remaining -= rows;
                        Some(Ok(batch.slice(0, rows)))
                    }
                })
            })
            .inspect_ok(move |batch| output_rows.add(batch.num_rows()));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.context.schema),
            batches,
        )))
    }
}

/// The scan schema's projection by `indices`, every column when `None`.
pub(crate) fn projected_schema(
    schema: &SchemaRef,
    indices: Option<&Vec<usize>>,
) -> DfResult<SchemaRef> {
    Ok(match indices {
        Some(indices) => Arc::new(schema.project(indices).map_err(QueryError::internal)?),
        None => Arc::clone(schema),
    })
}

/// Whether `schema` has a column for the cache to keep. A scan of no column
/// (`COUNT(*)`) has nothing to decode, so the Parquet scan serves it.
pub(crate) fn has_columns(schema: &Schema) -> bool {
    !schema.fields().is_empty()
}

#[cfg(test)]
mod tests {
    use arrow::util::pretty::pretty_format_batches;
    use arrow_array::{Array, Int64Array, LargeStringArray, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::builder::FtsConfig,
        supertable::{Supertable, SupertableOptions, SupertableReader},
    };

    /// Superfiles the dictionary test's table is written as.
    const FILES: i64 = 3;
    /// Rows in each.
    const ROWS_PER_FILE: i64 = 2000;
    /// Rows per row group, so each file has several.
    const ROW_GROUP_ROWS: usize = 500;
    /// The few values `region` takes.
    const REGIONS: [&str; 5] = ["af", "ap", "eu", "sa", "us"];
    /// Rows the delete removes: those with `seq` below this.
    const DELETED_BELOW: i64 = 300;
    /// Kept bytes: room for every column of every row group.
    const KEPT_BYTES: usize = 64 << 20;
    /// What `code` holds on a deleted row: no number, so a cast of it fails.
    const NOT_A_NUMBER: &str = "x";
    /// Every this many rows `code` is null.
    const NULL_EVERY: i64 = 7;
    /// The numbers `code` and `tag` take otherwise, as strings: few values.
    const CODES: i64 = 5;
    /// The region whose rows hold [`NOT_A_NUMBER`] in `tag`, live rows among
    /// them, so a cast of `tag` on them fails.
    const GUARDED_REGION: &str = "af";
    /// Statements over `region` (few values), `msg` (one per row) and `seq`.
    const STATEMENTS: &[&str] = &[
        "SELECT region, count(*) AS n FROM supertable WHERE seq < 4500 GROUP BY region ORDER BY region",
        "SELECT region, msg FROM supertable WHERE region = 'eu' ORDER BY seq LIMIT 5",
        "SELECT count(*) AS n FROM supertable WHERE region IN ('af', 'us') AND seq > 1000",
        "SELECT region, min(seq) AS lo, max(seq) AS hi FROM supertable GROUP BY region ORDER BY region DESC",
    ];

    /// A condition that fails on a deleted row's `code`, which holds no
    /// number: a scan that checked it on deleted rows would fail.
    const FAILS_ON_DELETED_ROWS: &str = "CAST(code AS INT) > 2";
    /// A guard ahead of a division by zero, after a condition on a kept
    /// dictionary. Without a cache the division sees only the rows both
    /// conditions before it keep; a scan that dropped the other regions'
    /// rows and left the guard to the filter would hand it a batch where the
    /// guard keeps more than a fifth of the rows, and the filter's `AND`
    /// would then divide every row by its `seq % 4`, zeros included.
    const GUARDS_A_DIVISION: &str = "region = 'eu' AND seq % 4 <> 0 AND 100 / (seq % 4) > 1";
    /// A condition that fails on live rows, a cast of the guarded region's
    /// `tag`, with or without a cache.
    const FAILS_ON_LIVE_ROWS: &str = "region = 'af' AND CAST(tag AS INT) > 2";
    /// Conditions over `code` and `tag`, each run grouped by `region` so the
    /// statement reads the scan rather than the files' statistics.
    const CONDITIONS: &[&str] = &[
        "code != '2'",
        "code NOT IN ('1', '3')",
        "code IS NULL",
        "code IS NOT NULL AND region LIKE 'a%'",
        "region LIKE '%u%' AND seq > 2500",
        "region NOT LIKE 'e%' AND code = '4'",
        "code IN ('1', '5') OR code IS NULL",
        FAILS_ON_DELETED_ROWS,
        "100 / seq >= 0 AND region = 'eu'",
        "region <> 'af' AND msg LIKE '%7'",
        GUARDS_A_DIVISION,
        FAILS_ON_LIVE_ROWS,
    ];

    fn region_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("msg", DataType::Utf8, false),
            Field::new("code", DataType::Utf8, true),
            Field::new("tag", DataType::Utf8, false),
        ]))
    }

    /// The table, keeping decoded columns in `kept` (none with `None`), its
    /// rows below [`DELETED_BELOW`] deleted.
    fn region_table(dir: &TempDir, kept: Option<Arc<DecodedColumnCache>>) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options = SupertableOptions::new(region_schema(), vec![], vec![])
            .expect("options")
            .with_storage(storage);
        options.decoded_columns = kept;
        options.row_group_size = Some(ROW_GROUP_ROWS);
        let st = Supertable::create(options).expect("create");
        for file in 0..FILES {
            let seqs: Vec<i64> = (file * ROWS_PER_FILE..(file + 1) * ROWS_PER_FILE).collect();
            let regions: Vec<&str> = seqs
                .iter()
                .map(|s| REGIONS[(*s as usize * 7) % REGIONS.len()])
                .collect();
            let msgs: Vec<String> = seqs.iter().map(|s| format!("message {s}")).collect();
            let codes: Vec<Option<String>> = seqs
                .iter()
                .map(|&s| match s {
                    s if s % NULL_EVERY == 0 => None,
                    s if s < DELETED_BELOW => Some(NOT_A_NUMBER.to_string()),
                    s => Some(((s * 3) % CODES + 1).to_string()),
                })
                .collect();
            let tags: Vec<String> = seqs
                .iter()
                .zip(&regions)
                .map(|(&s, &region)| match region == GUARDED_REGION {
                    true => NOT_A_NUMBER.to_string(),
                    false => (s % CODES + 1).to_string(),
                })
                .collect();
            let batch = RecordBatch::try_new(
                region_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(regions)),
                    Arc::new(StringArray::from(msgs)),
                    Arc::new(StringArray::from(codes)),
                    Arc::new(StringArray::from(tags)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
        st
    }

    fn answer(st: &Supertable, sql: &str) -> String {
        outcome(st, sql).expect(sql)
    }

    /// `sql`'s answer on `st`, or the error it fails with.
    fn outcome(st: &Supertable, sql: &str) -> Result<String, String> {
        let batches = st
            .reader()
            .expect("reader")
            .query_sql(sql)
            .map_err(|error| error.to_string())?;
        Ok(pretty_format_batches(&batches).expect("format").to_string())
    }

    /// Whether the table `answer` prints has a row of exactly `cells`.
    fn has_row(answer: &str, cells: &[&str]) -> bool {
        answer.lines().any(|line| {
            let row: Vec<&str> = line
                .split('|')
                .map(str::trim)
                .filter(|cell| !cell.is_empty())
                .collect();
            row == cells
        })
    }

    /// The scan checks a `WHERE` on a column it keeps as a dictionary, and a
    /// statement over such columns answers as without a cache: with nulls,
    /// `!=`, `NOT IN`, `IS NULL` and `LIKE`; with a condition that fails on a
    /// deleted row, which the scan never checks; and with a guard ahead of a
    /// condition that fails on the rows the guard drops. A statement that
    /// fails without a cache fails with one.
    #[test]
    fn a_where_checked_on_kept_dictionaries_answers_as_without_a_cache() {
        let plain_dir = TempDir::new().expect("tempdir");
        let plain = region_table(&plain_dir, None);
        let kept_dir = TempDir::new().expect("tempdir");
        let kept = region_table(
            &kept_dir,
            Some(Arc::new(DecodedColumnCache::new(KEPT_BYTES))),
        );
        let grouped = |condition: &str| {
            format!(
                "SELECT region, count(*) AS n FROM supertable WHERE {condition} \
                 GROUP BY region ORDER BY region"
            )
        };
        for pass in ["decoded", "kept", "kept again"] {
            for condition in CONDITIONS {
                let sql = grouped(condition);
                match outcome(&plain, &sql) {
                    Ok(expected) => {
                        assert_eq!(outcome(&kept, &sql), Ok(expected), "{pass}: {sql}");
                    }
                    Err(error) => {
                        assert_eq!(*condition, FAILS_ON_LIVE_ROWS, "{sql}: {error}");
                        assert!(outcome(&kept, &sql).is_err(), "{pass}: {sql}");
                    }
                }
            }
            // The guarded division keeps the live eu rows whose `seq % 4` is
            // not 0 (100 divided by 1, 2 or 3 is over 1): three in four of
            // the 1,140 live eu rows.
            let guarded = answer(&kept, &grouped(GUARDS_A_DIVISION));
            assert!(has_row(&guarded, &["eu", "855"]), "{pass}: {guarded}");
            // `100 / seq` fails only on `seq = 0`, which is deleted: every
            // live row passes, with or without a cache.
            let all_live = "SELECT count(*) AS n FROM supertable WHERE 100 / seq >= 0";
            let live = (FILES * ROWS_PER_FILE - DELETED_BELOW).to_string();
            for st in [&plain, &kept] {
                let counted = answer(st, all_live);
                assert!(has_row(&counted, &[live.as_str()]), "{pass}: {counted}");
            }
        }
        let fails = grouped(FAILS_ON_LIVE_ROWS);
        assert!(outcome(&plain, &fails).is_err(), "{fails}");
        // The filter above the scan hands its `WHERE` to the scan.
        let plan = explained(
            &kept.reader().expect("reader"),
            "SELECT region, max(msg) AS m FROM supertable WHERE region = 'eu' GROUP BY region",
        );
        assert!(
            plan.lines()
                .any(|line| line.contains("CachedScanExec") && line.contains("predicate=")),
            "the scan checks the WHERE:\n{plan}"
        );
    }

    /// A string column of few values is kept as a dictionary of `Int16`
    /// keys, and statements over it answer as without a cache: on the run
    /// that decodes and keeps it, and on the ones that read it kept.
    #[test]
    fn a_column_of_few_strings_is_kept_as_a_dictionary_and_answers_the_same() {
        let plain_dir = TempDir::new().expect("tempdir");
        let plain = region_table(&plain_dir, None);
        let kept_dir = TempDir::new().expect("tempdir");
        let cache = Arc::new(DecodedColumnCache::new(KEPT_BYTES));
        let kept = region_table(&kept_dir, Some(Arc::clone(&cache)));
        for pass in ["decoded", "kept", "kept again"] {
            for sql in STATEMENTS {
                assert_eq!(answer(&kept, sql), answer(&plain, sql), "{pass}: {sql}");
            }
        }
        let reader = kept.reader().expect("reader");
        let manifest = reader.manifest();
        let schema_id = manifest.table_schema().schema_id();
        let dictionary = DataType::Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8));
        for entry in manifest.superfiles.iter() {
            let region = cache
                .get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: "region".to_string(),
                    row_group: 0,
                })
                .expect("region kept");
            assert_eq!(region.data_type(), &dictionary, "{:?}", entry.uri);
            // A column of one value per row is kept as the scan reads it.
            let msg = cache
                .get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: "msg".to_string(),
                    row_group: 0,
                })
                .expect("msg kept");
            assert_eq!(msg.data_type(), &DataType::Utf8View, "{:?}", entry.uri);
        }
    }

    /// Rows in the batch the slicing test cuts.
    const ROWS: i64 = 10;
    /// Rows per slice in the slicing test: not a divisor of [`ROWS`].
    const SLICE: usize = 4;

    #[test]
    fn a_batch_is_cut_into_slices_of_the_batch_size_in_order() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let values: ArrayRef = Arc::new(Int64Array::from((0..ROWS).collect::<Vec<_>>()));
        let batch = RecordBatch::try_new(schema, vec![values]).expect("batch");
        let slices = sliced(batch, SLICE);
        let sizes: Vec<usize> = slices.iter().map(RecordBatch::num_rows).collect();
        assert_eq!(sizes, vec![4, 4, 2]);
        let back: Vec<i64> = slices
            .iter()
            .flat_map(|s| {
                let n = s
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("ints")
                    .clone();
                (0..n.len()).map(move |i| n.value(i))
            })
            .collect();
        assert_eq!(back, (0..ROWS).collect::<Vec<_>>());
        let empty = RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
            "n",
            DataType::Int64,
            false,
        )])));
        assert!(sliced(empty, SLICE).is_empty());
    }

    /// Rows in each superfile of the narrowed table, one row group each.
    const NARROWED_ROWS: i64 = 5_000;
    /// Rows of the first superfile holding `x`: few enough for the index to
    /// select them, and far fewer than half the row group.
    const SPARSE: i64 = 100;
    /// Rows of the second superfile holding `x`: more than half of it, so
    /// the index does not select them and the superfile is read whole.
    const DENSE: i64 = 3_000;
    /// Rows deleted: those whose `seq` leaves [`DELETED_REMAINDER`] divided
    /// by this, some of each superfile's `x` rows among them.
    const DELETED_EVERY: i64 = 10;
    /// See [`DELETED_EVERY`].
    const DELETED_REMAINDER: i64 = 3;
    /// Kept bytes: room for every column of the narrowed table.
    const NARROWED_KEPT_BYTES: usize = 64 << 20;
    /// Rows the `ORDER BY ... LIMIT` statements take.
    const TOP_ROWS: usize = 10;

    /// A scan the index narrows in the first superfile only.
    const PARTLY_NARROWED: &str = "SELECT seq FROM supertable WHERE k = 'x' ORDER BY seq";
    /// A scan the index narrows in both superfiles, to one row each.
    const FULLY_NARROWED: &str =
        "SELECT seq FROM supertable WHERE k IN ('v200', 'v8200') ORDER BY seq";

    fn narrowed_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("k", DataType::LargeUtf8, false),
            Field::new("pad", DataType::LargeUtf8, false),
        ]))
    }

    /// The `seq` of row `i` of superfile `file`, and its `k`.
    fn narrowed_row(file: i64, i: i64) -> (i64, String) {
        let seq = file * NARROWED_ROWS + i;
        let x = match file {
            0 => i < SPARSE,
            _ => i < DENSE,
        };
        (seq, if x { "x".to_owned() } else { format!("v{seq}") })
    }

    /// Every live `x` row's `seq`, ascending.
    fn live_x() -> Vec<i64> {
        (0..2)
            .flat_map(|file| (0..NARROWED_ROWS).map(move |i| narrowed_row(file, i)))
            .filter(|(seq, k)| k == "x" && seq % DELETED_EVERY != DELETED_REMAINDER)
            .map(|(seq, _)| seq)
            .collect()
    }

    /// Two superfiles of [`NARROWED_ROWS`] rows, `k` indexed, its rows
    /// deleted as [`DELETED_EVERY`] says, keeping decoded columns in `kept`
    /// when given.
    fn narrowed_table(dir: &TempDir, kept: Option<Arc<DecodedColumnCache>>) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let mut options =
            SupertableOptions::new(narrowed_schema(), vec![FtsConfig::new("k")], vec![])
                .expect("options")
                .with_storage(storage);
        options.decoded_columns = kept;
        options.row_group_size = Some(NARROWED_ROWS as usize);
        let st = Supertable::create(options).expect("create");
        for file in 0..2 {
            let (seqs, ks): (Vec<i64>, Vec<String>) =
                (0..NARROWED_ROWS).map(|i| narrowed_row(file, i)).unzip();
            let pads: Vec<String> = seqs.iter().map(|s| format!("{s:0>48}")).collect();
            let batch = RecordBatch::try_new(
                narrowed_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(LargeStringArray::from(ks)),
                    Arc::new(LargeStringArray::from(pads)),
                ],
            )
            .expect("batch");
            let mut writer = st.writer().expect("writer");
            writer.append(&batch).expect("append");
            writer.commit().expect("commit");
        }
        st.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(DELETED_REMAINDER)))
            .expect("delete");
        st
    }

    /// The `seq` column of `sql`'s rows, in order.
    fn seqs_of(reader: &SupertableReader, sql: &str) -> Vec<i64> {
        reader
            .query_sql(sql)
            .unwrap_or_else(|error| panic!("{sql}: {error}"))
            .iter()
            .flat_map(|b| {
                let seqs = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..seqs.len()).map(move |i| seqs.value(i))
            })
            .collect()
    }

    /// The `EXPLAIN` of `sql` on `reader`, every plan line joined.
    fn explained(reader: &SupertableReader, sql: &str) -> String {
        reader
            .query_sql(&format!("EXPLAIN {sql}"))
            .expect("explain")
            .iter()
            .flat_map(|b| {
                let plans = b
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("plan column")
                    .clone();
                (0..plans.len()).map(move |i| plans.value(i).to_owned())
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// How many of `reader`'s superfiles have `column` of their first row
    /// group, the only one each holds, kept in `kept`.
    fn superfiles_keeping(
        reader: &SupertableReader,
        kept: &DecodedColumnCache,
        column: &str,
    ) -> usize {
        let manifest = reader.manifest();
        let schema_id = manifest.table_schema().schema_id();
        manifest
            .superfiles
            .iter()
            .filter(|entry| {
                kept.get(&ColumnKey {
                    uri: entry.uri,
                    schema_id,
                    column: column.to_owned(),
                    row_group: 0,
                })
                .is_some()
            })
            .count()
    }

    /// A scan the index narrows in some superfiles only reads them through
    /// kept columns, and answers as the same table keeping no columns does:
    /// the plain scan and `ORDER BY ... LIMIT` both ways (the path that can
    /// hand back the Parquet scan), on the first pass and the kept ones. The
    /// narrowed row group's few rows are read from its pages and kept only
    /// on its second read, so the first statement keeps the columns of the
    /// superfile read whole alone.
    #[test]
    fn a_partly_narrowed_scan_answers_as_the_table_without_kept_columns() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        let plan = explained(&with, PARTLY_NARROWED);
        assert!(plan.contains("CachedScanExec"), "{plan}");
        let plan = explained(&without, PARTLY_NARROWED);
        assert!(!plan.contains("CachedScanExec"), "{plan}");

        // First read: `k` is kept for the superfile read whole only. (The
        // delete that built the table may have kept `seq` already.)
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0);
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            1,
            "the narrowed row group's first read keeps nothing"
        );
        // Second read of the narrowed row group: its columns are kept too.
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            2,
            "the second read keeps the narrowed row group"
        );

        let statements = [
            PARTLY_NARROWED.to_owned(),
            format!(
                "SELECT seq, pad FROM supertable WHERE k = 'x' ORDER BY seq DESC LIMIT {TOP_ROWS}"
            ),
            format!(
                "SELECT seq, k, pad FROM supertable WHERE k = 'x' ORDER BY seq LIMIT {TOP_ROWS}"
            ),
        ];
        let mut descending = live_x();
        descending.reverse();
        let wanted = [
            live_x(),
            descending.into_iter().take(TOP_ROWS).collect::<Vec<_>>(),
            live_x().into_iter().take(TOP_ROWS).collect::<Vec<_>>(),
        ];
        for pass in ["first", "kept", "kept again"] {
            for (sql, want) in statements.iter().zip(&wanted) {
                let got = seqs_of(&with, sql);
                assert_eq!(got, seqs_of(&without, sql), "{pass}: {sql}");
                assert_eq!(&got, want, "{pass}: {sql}");
            }
        }
    }

    /// A narrowed row group none of whose rows were read in part before
    /// hands over the columns kept for it and reads the others for its
    /// selected rows from its pages, keeping them only for the row group read
    /// whole; and answers as the table keeping no columns does.
    #[test]
    fn a_narrowed_row_group_takes_kept_columns_and_reads_the_rest_from_its_pages() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        // A scan narrowed nowhere keeps `seq` of both row groups.
        let every = "SELECT seq FROM supertable ORDER BY seq";
        assert_eq!(seqs_of(&with, every), seqs_of(&without, every));
        assert_eq!(superfiles_keeping(&with, &kept, "seq"), 2);
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0);
        // `k` is kept for the row group read whole only; the narrowed one's
        // `seq` comes from the kept column, its `k` from its pages.
        let got = seqs_of(&with, PARTLY_NARROWED);
        assert_eq!(got, seqs_of(&without, PARTLY_NARROWED));
        assert_eq!(got, live_x());
        assert_eq!(
            superfiles_keeping(&with, &kept, "k"),
            1,
            "k of the row group read whole only"
        );
    }

    /// A scan the index narrows in every superfile reads their selected
    /// pages through the Parquet scan, keeping none of its columns, and answers as
    /// the table keeping no columns does.
    #[test]
    fn a_scan_narrowed_in_every_superfile_reads_the_files() {
        let kept = Arc::new(DecodedColumnCache::new(NARROWED_KEPT_BYTES));
        let (with_dir, without_dir) = (
            TempDir::new().expect("tempdir"),
            TempDir::new().expect("tempdir"),
        );
        let with = narrowed_table(&with_dir, Some(Arc::clone(&kept)));
        let without = narrowed_table(&without_dir, None);
        let (with, without) = (
            with.reader().expect("reader"),
            without.reader().expect("reader"),
        );
        let plan = explained(&with, FULLY_NARROWED);
        assert!(!plan.contains("CachedScanExec"), "{plan}");
        assert!(plan.contains("DataSourceExec"), "{plan}");
        for pass in ["first", "again"] {
            let got = seqs_of(&with, FULLY_NARROWED);
            assert_eq!(got, seqs_of(&without, FULLY_NARROWED), "{pass}");
            assert_eq!(got, vec![200, 8200], "{pass}");
        }
        assert_eq!(superfiles_keeping(&with, &kept, "k"), 0, "k is not kept");
    }
}
