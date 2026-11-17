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
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::{
    arrow::compute::{can_cast_types, cast_with_options, filter, filter_record_batch},
    common::Statistics,
    datasource::physical_plan::parquet::ParquetAccessPlan,
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{EquivalenceProperties, PhysicalExpr},
    physical_expr_adapter::PhysicalExprAdapterFactory,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        metrics::{Count, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt, future, stream};
use parquet::file::metadata::ParquetMetaData;
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
            column_cache::{ColumnKey, DecodedColumnCache, RowGroupKey},
            exec::common::{hit_output_schema, read_file_row_group, take_file_columns},
            read_ahead::{file_pruning, statistics_pruned},
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

/// Metric counting the row groups a scan read only the chosen rows of, from
/// their pages, keeping nothing, as `EXPLAIN ANALYZE` lists it.
const ROW_GROUPS_READ_IN_PART: &str = "row_groups_read_in_part";

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
}

/// A table scan served from the decoded-column cache; see the module docs.
#[derive(Debug)]
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
        if decoded_bytes(units.iter().filter(|unit| unit.keep), &schema) > cache.max_bytes() {
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
            }),
            units: units.into(),
            pruned,
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
            parquet,
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
fn decoded_bytes<'a>(units: impl Iterator<Item = &'a RowGroupUnit>, schema: &Schema) -> usize {
    units
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
fn typed(array: &ArrayRef, target: &DataType) -> DfResult<ArrayRef> {
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
        for (&i, array) in missing.iter().zip(read.columns()) {
            let array = typed(array, fields[i].data_type())?;
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
    /// A decode of the columns the cache lacks counts in `decoded`; a read of
    /// just the chosen rows from the pages, keeping nothing, in `read_in_part`.
    async fn read(
        self: &Arc<Self>,
        unit: &RowGroupUnit,
        decoded: &Count,
        read_in_part: &Count,
    ) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
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
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(n)),
        )?;
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
        Ok(match keep {
            Some(mask) => filter_record_batch(&batch, &mask)?,
            None => batch,
        })
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
                Some(kept) => filter(kept.as_ref(), &mask).map_err(DataFusionError::from),
                None => {
                    let array = taken
                        .next()
                        .ok_or_else(|| DataFusionError::Internal("a column was not read".into()))?;
                    typed(array, field.data_type())
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
            context: Arc::clone(&self.context),
            units: Arc::clone(&self.units),
            pruned: self.pruned,
            parquet: Arc::clone(&self.parquet),
            partitions: self.partitions,
            fetch: limit,
            statistics: Arc::clone(&self.statistics),
            properties: Arc::clone(&self.properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
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
        let read_in_part =
            MetricBuilder::new(&self.metrics).counter(ROW_GROUPS_READ_IN_PART, partition);
        // A row group goes out in batch-size slices, as the Parquet scan's
        // rows do: the filter above then works a batch at a time, and a
        // `LIMIT` met in the first slice leaves the rest of the row group.
        let batch_size = task.session_config().batch_size();
        let batches = stream::iter(units)
            .then(move |unit| {
                let context = Arc::clone(&context);
                let decoded = decoded.clone();
                let read_in_part = read_in_part.clone();
                async move { context.read(&unit, &decoded, &read_in_part).await }
            })
            .map_ok(move |batch| stream::iter(sliced(batch, batch_size).into_iter().map(Ok)))
            .try_flatten()
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
            .collect()
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

        // First read: only the whole superfile's columns are kept.
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        let whole_only = kept.len();
        assert!(whole_only > 0, "the superfile read whole keeps its columns");
        // Second read of the narrowed row group: its columns are kept too.
        assert_eq!(seqs_of(&with, PARTLY_NARROWED), live_x());
        assert_eq!(
            kept.len(),
            2 * whole_only,
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
        assert_eq!(kept.len(), 2, "seq of both row groups");
        // `k` is kept for the row group read whole only; the narrowed one's
        // `seq` comes from the kept column, its `k` from its pages.
        let got = seqs_of(&with, PARTLY_NARROWED);
        assert_eq!(got, seqs_of(&without, PARTLY_NARROWED));
        assert_eq!(got, live_x());
        assert_eq!(kept.len(), 3, "k of the row group read whole");
    }

    /// A scan the index narrows in every superfile reads their selected
    /// pages through the Parquet scan, never kept columns, and answers as
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
        assert_eq!(kept.len(), 0, "no column is kept");
    }
}
