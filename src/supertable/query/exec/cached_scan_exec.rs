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
    sync::Arc,
};

use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, builder::BooleanBufferBuilder,
};
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::{
    arrow::compute::{can_cast_types, cast_with_options, filter_record_batch},
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
        manifest::{ManifestSnapshot, SuperfileUri, hll::HllSketch},
        query::{
            column_cache::{ColumnKey, DecodedColumnCache},
            exec::common::{hit_output_schema, read_file_row_group},
            read_ahead::{file_pruning, statistics_pruned},
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
    /// For each file, which of `schema`'s columns are kept as dictionaries
    /// ([`dictionary_columns`]).
    dictionaries: HashMap<SuperfileUri, Arc<[bool]>>,
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
        let dictionaries = dictionary_columns(&manifest, &schema, &units);
        if decoded_bytes(&units, &schema, &dictionaries) > cache.max_bytes() {
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
                dictionaries,
            }),
            units: units.into(),
            pruned,
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
            parquet,
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
fn decoded_bytes(
    units: &[RowGroupUnit],
    schema: &Schema,
    dictionaries: &HashMap<SuperfileUri, Arc<[bool]>>,
) -> usize {
    units
        .iter()
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
                    let distinct = stats.value_table_len().or_else(|| {
                        stats
                            .sketch()
                            .and_then(HllSketch::from_bytes)
                            .map(|sketch| sketch.estimate().ceil() as usize)
                    });
                    distinct.is_some_and(|distinct| {
                        distinct <= i16::MAX as usize
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
        let dictionary = self.dictionaries.get(&unit.file.uri);
        for (&i, array) in missing.iter().zip(read.columns()) {
            let target = fields[i].data_type();
            let array = if array.data_type() == target {
                Arc::clone(array)
            } else {
                cast_with_options(array, target, &CAST_OPTIONS)?
            };
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
    async fn read(self: &Arc<Self>, unit: &RowGroupUnit) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
        let mut columns: Vec<Option<ArrayRef>> = fields
            .iter()
            .map(|f| {
                self.cache
                    .get(&self.key(unit, f.name()))
                    .map(|kept| as_scanned(kept, f.data_type()))
                    .transpose()
            })
            .collect::<DfResult<_>>()?;
        let missing: Vec<usize> = (0..columns.len())
            .filter(|&i| columns[i].is_none())
            .collect();
        if !missing.is_empty() {
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
            context: Arc::clone(&self.context),
            units: Arc::clone(&self.units),
            pruned: self.pruned,
            parquet: Arc::clone(&self.parquet),
            partitions: self.partitions,
            fetch: limit,
            statistics: Arc::clone(&self.statistics),
            properties: Arc::clone(&self.properties),
        }))
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
        // A row group goes out in batch-size slices, as the Parquet scan's
        // rows do: the filter above then works a batch at a time, and a
        // `LIMIT` met in the first slice leaves the rest of the row group.
        let batch_size = task.session_config().batch_size();
        let batches = stream::iter(units)
            .then(move |unit| {
                let context = Arc::clone(&context);
                async move { context.read(&unit).await }
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
            });
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
    use arrow_array::{Array, Int64Array, StringArray};
    use arrow_schema::{DataType, Field};
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::{Supertable, SupertableOptions, query::scalar_cache::ByteLru},
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
    /// Statements over `region` (few values), `msg` (one per row) and `seq`.
    const STATEMENTS: &[&str] = &[
        "SELECT region, count(*) AS n FROM supertable WHERE seq < 4500 GROUP BY region ORDER BY region",
        "SELECT region, msg FROM supertable WHERE region = 'eu' ORDER BY seq LIMIT 5",
        "SELECT count(*) AS n FROM supertable WHERE region IN ('af', 'us') AND seq > 1000",
        "SELECT region, min(seq) AS lo, max(seq) AS hi FROM supertable GROUP BY region ORDER BY region DESC",
    ];

    fn region_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("seq", DataType::Int64, false),
            Field::new("region", DataType::Utf8, false),
            Field::new("msg", DataType::Utf8, false),
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
            let batch = RecordBatch::try_new(
                region_schema(),
                vec![
                    Arc::new(Int64Array::from(seqs)),
                    Arc::new(StringArray::from(regions)),
                    Arc::new(StringArray::from(msgs)),
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
        let batches = st.reader().expect("reader").query_sql(sql).expect(sql);
        pretty_format_batches(&batches).expect("format").to_string()
    }

    /// A string column of few values is kept as a dictionary of `Int16`
    /// keys, and statements over it answer as without a cache: on the run
    /// that decodes and keeps it, and on the ones that read it kept.
    #[test]
    fn a_column_of_few_strings_is_kept_as_a_dictionary_and_answers_the_same() {
        let plain_dir = TempDir::new().expect("tempdir");
        let plain = region_table(&plain_dir, None);
        let kept_dir = TempDir::new().expect("tempdir");
        let cache = Arc::new(ByteLru::new(KEPT_BYTES));
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
}
