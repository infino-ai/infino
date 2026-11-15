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
//! use), keeps them, and drops the row group's deleted rows. When the range
//! index chose a file's rows (an `ORDER BY ... LIMIT` over a range), only
//! those rows come out, and a row group holding none of them is not read.
//!
//! A top-k above it hands down its running bound (the k-th best sort value
//! so far) as a filter that tightens as rows arrive. Asked for a sort on one
//! of its columns, the scan reads the row groups whose statistics start best
//! first, and skips each row group the bound rules out by its statistics just
//! before reading it, so the top-k settles early and the rest go unread.
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

use std::{fmt, ops::Range, sync::Arc};

use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, builder::BooleanBufferBuilder,
};
use arrow_schema::{Schema, SchemaRef};
use datafusion::{
    arrow::compute::{can_cast_types, cast_with_options, filter_record_batch},
    common::{ScalarValue, Statistics, config::ConfigOptions},
    datasource::physical_plan::parquet::ParquetAccessPlan,
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{
        EquivalenceProperties, PhysicalExpr, PhysicalSortExpr,
        expressions::{Column, DynamicFilterPhysicalExpr, Literal},
    },
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_expr_common::physical_expr::snapshot_physical_expr,
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
    file::metadata::ParquetMetaData,
};
use roaring::RoaringBitmap;
use tokio::spawn;
use tracing::Instrument;

use crate::{
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        manifest::{ManifestSnapshot, SuperfileUri},
        query::{
            column_cache::{ColumnKey, DecodedColumnCache},
            exec::common::{hit_output_schema, read_file_row_group},
            read_ahead::{file_pruning, statistics_pruned},
            schema_adapter::TableExprAdapterFactory,
        },
        schema::cast::CAST_OPTIONS,
    },
};

/// Bytes a decoded string or binary value takes per row beside its data: a
/// view, as the scan reads non-indexed strings, and the most any of Arrow's
/// variable-width layouts takes.
const VIEW_BYTES_PER_ROW: usize = 16;

/// Metric counting the row groups a scan decoded because the cache lacked
/// one of their projected columns, as `EXPLAIN ANALYZE` lists it. A run over
/// columns the cache keeps decodes none.
const ROW_GROUPS_DECODED: &str = "row_groups_decoded";

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
    /// The only rows to read, by position in the file, when the range
    /// index chose them (an `ORDER BY ... LIMIT` over a range reads the rows
    /// up to its cutoff); `None` reads every row.
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
    /// The rows each partition hands on, which the per-query counters take
    /// as the rows the scan materialized, as they take a Parquet scan's; and
    /// the row groups it decoded ([`ROW_GROUPS_DECODED`]).
    metrics: ExecutionPlanMetricsSet,
    /// Filters a parent pushed down that tighten while the statement runs
    /// ([`LiveFilters`]); `None` without any.
    live: Option<Arc<LiveFilters>>,
}

/// Filters pushed down from above that tighten as the statement runs, such
/// as a top-k's bound on its sort key (the k-th best value so far). Each is
/// checked against a row group's statistics just before the row group is
/// read, and a row group that can no longer reach the result is skipped;
/// then against each batch the scan hands over, and the rows it rules out
/// are dropped before the filter above checks them. A row a top-k's bound
/// rules out can't reach its result, so dropping it changes no answer; the
/// rows that pass are still checked above.
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

    /// Whether the filters as they stand rule `unit` out by its statistics.
    fn skip(&self, unit: &RowGroupUnit, table: &SchemaRef) -> bool {
        self.filters.iter().any(|filter| {
            let Ok(now) = snapshot_physical_expr(Arc::clone(filter)) else {
                return false;
            };
            RowGroupFilter {
                predicate: now,
                table: Arc::clone(table),
                adapter: Arc::clone(&self.adapter),
            }
            .plan(&unit.file)
            .is_some_and(|plan| !plan.should_scan(unit.row_group))
        })
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
                let chosen = file
                    .selected
                    .as_ref()
                    .is_none_or(|selected| selected.range_cardinality(rows.clone()) > 0);
                if chosen && n > 0 && file.tombstones.range_cardinality(rows.clone()) < u64::from(n)
                {
                    units.push(RowGroupUnit {
                        file: Arc::clone(&file),
                        row_group,
                        rows,
                    });
                }
            }
        }
        if decoded_bytes(&units, &schema) > cache.max_bytes() {
            return None;
        }
        let partitions = target_partitions.clamp(1, units.len().max(1));
        let schema_id = manifest.table_schema().schema_id();
        let properties = plan_properties(&schema, partitions);
        Some(Self {
            context: Arc::new(ScanContext {
                schema,
                manifest,
                op_stats,
                cache,
                schema_id,
            }),
            units: units.into(),
            pruned,
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
            parquet,
            metrics: ExecutionPlanMetricsSet::new(),
            live: None,
        })
    }

    /// The Parquet scan this one stands in for, metered as the provider
    /// builds it.
    pub(crate) fn parquet(&self) -> &Arc<dyn ExecutionPlan> {
        &self.parquet
    }
}

/// About how many bytes the columns of `schema` take decoded over `units`'
/// row groups, from the footers:
/// - a fixed-width column, its width per row;
/// - any other, [`VIEW_BYTES_PER_ROW`] per row plus its values' bytes, which
///   the footer records unencoded (a dictionary-encoded column stores far
///   fewer bytes than it decodes to), or failing that its chunk's
///   uncompressed size.
///
/// A column a file holds under another name, after a rename, isn't found by
/// this name and counts as nothing; the estimate decides only which scan
/// serves the statement, never what it returns.
fn decoded_bytes(units: &[RowGroupUnit], schema: &Schema) -> usize {
    units
        .iter()
        .map(|unit| {
            let meta = unit.file.reader.parquet_metadata();
            let Some(row_group) = meta.row_groups().get(unit.row_group) else {
                return 0;
            };
            let rows = usize::try_from(row_group.num_rows()).unwrap_or(0);
            schema
                .fields()
                .iter()
                .map(|field| {
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

/// `units` ordered by the row-group statistics of `column`, smallest minimum
/// first (largest maximum first when `descending`), row groups without
/// statistics last; `None` when no file has statistics for it.
fn best_first(units: &[RowGroupUnit], column: &str, descending: bool) -> Option<Vec<RowGroupUnit>> {
    let mut keyed: Vec<(Option<ScalarValue>, RowGroupUnit)> = Vec::with_capacity(units.len());
    let mut any = false;
    // Each file's bounds, read from its footer once.
    let mut bounds: Vec<(Arc<CachedFile>, Option<ArrayRef>)> = Vec::new();
    for unit in units {
        let array = match bounds
            .iter()
            .find(|(file, _)| Arc::ptr_eq(file, &unit.file))
        {
            Some((_, array)) => array.clone(),
            None => {
                let array = row_group_bounds(&unit.file.footer, column, descending);
                bounds.push((Arc::clone(&unit.file), array.clone()));
                array
            }
        };
        let key = array
            .as_ref()
            .and_then(|array| ScalarValue::try_from_array(array, unit.row_group).ok())
            .filter(|value| !value.is_null());
        any |= key.is_some();
        keyed.push((key, unit.clone()));
    }
    if !any {
        return None;
    }
    keyed.sort_by(|(a, _), (b, _)| match (a, b) {
        (Some(a), Some(b)) => {
            let order = a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
            if descending { order.reverse() } else { order }
        }
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });
    Some(keyed.into_iter().map(|(_, unit)| unit).collect())
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
        for (&i, array) in missing.iter().zip(read.columns()) {
            let target = fields[i].data_type();
            let array = if array.data_type() == target {
                Arc::clone(array)
            } else {
                cast_with_options(array, target, &CAST_OPTIONS)?
            };
            self.cache.insert(
                self.key(&unit, fields[i].name()),
                Arc::clone(&array),
                array.get_array_memory_size(),
            );
            decoded.push(array);
        }
        Ok(decoded)
    }

    /// `unit`'s row group, every projected column, without its deleted rows
    /// and, when the index chose the file's rows, without the rows it didn't.
    /// A decode of the columns the cache lacks counts in `decoded`.
    async fn read(self: &Arc<Self>, unit: &RowGroupUnit, decoded: &Count) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
        let mut columns: Vec<Option<ArrayRef>> = fields
            .iter()
            .map(|f| self.cache.get(&self.key(unit, f.name())))
            .collect();
        let missing: Vec<usize> = (0..columns.len())
            .filter(|&i| columns[i].is_none())
            .collect();
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
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(n)),
        )?;
        let start = unit.rows.start;
        let keep = match &unit.file.selected {
            Some(selected) => {
                let mut mask = BooleanBufferBuilder::new(n);
                mask.append_n(n, false);
                for row in selected.range(unit.rows.clone()) {
                    if !unit.file.tombstones.contains(row) {
                        mask.set_bit((row - start) as usize, true);
                    }
                }
                Some(mask)
            }
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
                keep
            }
        };
        Ok(match keep {
            Some(mut mask) => filter_record_batch(&batch, &BooleanArray::new(mask.finish(), None))?,
            None => batch,
        })
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

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    /// Keeps the filters a parent pushed down that tighten as the statement
    /// runs ([`LiveFilters`]), to skip row groups by; every filter stays
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
        let Some(units) = best_first(&self.units, column.name(), first.options.descending) else {
            return Ok(SortOrderPushdownResult::Unsupported);
        };
        Ok(SortOrderPushdownResult::Inexact {
            inner: Arc::new(Self {
                units: units.into(),
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
        let output_rows = MetricBuilder::new(&self.metrics).output_rows(partition);
        let decoded = MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_DECODED, partition);
        let live = self.live.clone();
        let kept = self.live.clone();
        let context_schema = Arc::clone(&self.context.schema);
        // A row group goes out in batch-size slices, as the Parquet scan's
        // rows do: the filter above then works a batch at a time, and a
        // `LIMIT` met in the first slice leaves the rest of the row group.
        let batch_size = task.session_config().batch_size();
        let batches = stream::iter(units)
            .filter(move |unit| {
                // Checked as each row group comes up, so a bound tightened by
                // the rows read so far skips it.
                let skip = live
                    .as_ref()
                    .is_some_and(|live| live.skip(unit, &context_schema));
                future::ready(!skip)
            })
            .then(move |unit| {
                let context = Arc::clone(&context);
                let decoded = decoded.clone();
                async move { context.read(&unit, &decoded).await }
            })
            .map_ok(move |batch| stream::iter(sliced(batch, batch_size).into_iter().map(Ok)))
            .try_flatten()
            // Each slice as it goes out, so a bound the rows before it
            // tightened drops what can no longer reach the result.
            .and_then(move |batch| {
                future::ready(match &kept {
                    Some(live) => live.keep(batch),
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
    use arrow_array::{Array, Int64Array};
    use arrow_schema::{DataType, Field};

    use super::*;

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
}
