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
//! decodes the ones the cache lacks through the superfile reader
//! ([`take_file_columns`], the read and schema adaptation the search paths
//! use), keeps them, and drops the row group's deleted rows.
//!
//! It answers what the Parquet scan would: the same columns, typed as the scan
//! schema types them, each file's rows in file order, without the rows a
//! tombstone deletes. A `WHERE` stays above it in a `FilterExec`, as above the
//! Parquet scan. It does not skip a row group by its statistics: the filter
//! above drops the rows instead, which over decoded arrays costs less than the
//! decode it saves.

use std::{fmt, ops::Range, sync::Arc};

use arrow_array::{
    ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, builder::BooleanBufferBuilder,
};
use arrow_schema::{Schema, SchemaRef};
use datafusion::{
    arrow::compute::{can_cast_types, cast_with_options, filter_record_batch},
    common::Statistics,
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
        SendableRecordBatchStream,
        execution_plan::{Boundedness, EmissionType},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt, future, stream};
use roaring::RoaringBitmap;

use crate::{
    runtime_metrics::op_stats::OpStatsCollector,
    superfile::SuperfileReader,
    supertable::{
        error::QueryError,
        manifest::{ManifestSnapshot, SuperfileUri},
        query::{
            column_cache::{ColumnKey, DecodedColumnCache},
            exec::common::{hit_output_schema, take_file_columns},
        },
        schema::cast::CAST_OPTIONS,
    },
};

/// One file of the scan: what its row groups are read from.
#[derive(Debug)]
pub(crate) struct CachedFile {
    pub(crate) uri: SuperfileUri,
    pub(crate) reader: Arc<SuperfileReader>,
    /// Rows in each row group, in file order.
    pub(crate) row_counts: Arc<[u32]>,
    /// The file's deleted rows, by position in the file.
    pub(crate) tombstones: Arc<RoaringBitmap>,
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
    /// The same columns typed as the files store them, which the reader
    /// decodes into.
    stored: SchemaRef,
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
    partitions: usize,
    fetch: Option<usize>,
    statistics: Arc<Statistics>,
    properties: Arc<PlanProperties>,
}

impl CachedScanExec {
    /// A scan of `schema`'s columns over `files`, read as up to
    /// `target_partitions` partitions, or `None` when a column is not one the
    /// files store in a type that converts to `schema`'s: the Parquet scan
    /// then serves it, with its own schema adapter.
    pub(crate) fn try_new(
        schema: SchemaRef,
        files: Vec<CachedFile>,
        manifest: Arc<ManifestSnapshot>,
        op_stats: Option<Arc<OpStatsCollector>>,
        cache: Arc<DecodedColumnCache>,
        target_partitions: usize,
        statistics: Statistics,
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
        for file in files.into_iter().map(Arc::new) {
            let mut first = 0u32;
            for (row_group, &n) in file.row_counts.iter().enumerate() {
                let rows = first..first + n;
                first += n;
                // A row group every row of which is deleted reads nothing.
                if n > 0 && file.tombstones.range_cardinality(rows.clone()) < u64::from(n) {
                    units.push(RowGroupUnit {
                        file: Arc::clone(&file),
                        row_group,
                        rows,
                    });
                }
            }
        }
        let partitions = target_partitions.clamp(1, units.len().max(1));
        let schema_id = manifest.table_schema().schema_id();
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            Partitioning::UnknownPartitioning(partitions),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Some(Self {
            context: Arc::new(ScanContext {
                schema,
                stored,
                manifest,
                op_stats,
                cache,
                schema_id,
            }),
            units: units.into(),
            partitions,
            fetch: None,
            statistics: Arc::new(statistics),
            properties,
        })
    }
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

    /// `unit`'s row group, every projected column, without its deleted rows.
    async fn read(&self, unit: &RowGroupUnit) -> DfResult<RecordBatch> {
        let fields = self.schema.fields();
        let mut columns: Vec<Option<ArrayRef>> = fields
            .iter()
            .map(|f| self.cache.get(&self.key(unit, f.name())))
            .collect();
        let missing: Vec<usize> = (0..columns.len())
            .filter(|&i| columns[i].is_none())
            .collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|&i| fields[i].name().as_str()).collect();
            let stored = Arc::new(
                self.stored
                    .project(&missing)
                    .map_err(QueryError::internal)?,
            );
            let read = take_file_columns(
                &self.manifest,
                &unit.file.reader,
                unit.rows.clone().collect(),
                &names,
                &stored,
                &self.op_stats,
            )
            .await?;
            for (&i, array) in missing.iter().zip(read.columns()) {
                let target = fields[i].data_type();
                let array = if array.data_type() == target {
                    Arc::clone(array)
                } else {
                    cast_with_options(array, target, &CAST_OPTIONS)?
                };
                self.cache.insert(
                    self.key(unit, fields[i].name()),
                    Arc::clone(&array),
                    array.get_array_memory_size(),
                );
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
        let deleted = unit.file.tombstones.range(unit.rows.clone());
        let mut keep: Option<BooleanBufferBuilder> = None;
        for row in deleted {
            let mask = keep.get_or_insert_with(|| {
                let mut mask = BooleanBufferBuilder::new(n);
                mask.append_n(n, true);
                mask
            });
            mask.set_bit((row - unit.rows.start) as usize, false);
        }
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
            "CachedScanExec: row_groups={}, partitions={}, projection=[{}]",
            self.units.len(),
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
            partitions: self.partitions,
            fetch: limit,
            statistics: Arc::clone(&self.statistics),
            properties: Arc::clone(&self.properties),
        }))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
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
        let batches = stream::iter(units)
            .then(move |unit| {
                let context = Arc::clone(&context);
                async move { context.read(&unit).await }
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
