// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Document columns on the read path: [`DocumentBinding`] rebuilds a
//! document column's values from a batch of the stored columns it draws on,
//! and [`DocumentExec`] does that over a table scan, so a statement reads a
//! document column like any other.
//!
//! ```text
//!   SELECT source, ts FROM t WHERE ...
//!
//!   FilterExec / SortExec ...           (a predicate on `source` runs here,
//!          │                             over the rebuilt text)
//!   DocumentExec   source ← rebuild(source#skeleton, source#residual, leaves)
//!          │       ts     ← ts
//!   table scan     ts, source#skeleton, source#residual, every stored leaf
//! ```
//!
//! The scan cannot know which columns a row's skeleton names before it has
//! read the skeleton, so it reads every column a document can draw on
//! ([`TableSchema::document_leaf_columns`]). The search paths resolve a
//! handful of rows and read only the columns their skeletons name, and so
//! does [`LateDocumentExec`], which reads a document column after a plan's
//! limit for the rows it keeps (`LateDocuments` moves the rebuild there).

use std::{collections::HashMap, fmt, sync::Arc};

use arrow::compute::{cast, concat_batches};
use arrow_array::{
    Array, ArrayRef, Decimal128Array, LargeStringArray, RecordBatch, RecordBatchOptions,
    builder::LargeStringBuilder, cast::AsArray,
};
use arrow_schema::{DataType, Schema, SchemaRef};
use datafusion::{
    common::{ColumnStatistics, Statistics, config::ConfigOptions},
    error::{DataFusionError, Result as DfResult},
    execution::TaskContext,
    physical_expr::{EquivalenceProperties, PhysicalExpr, PhysicalSortExpr, expressions::Column},
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
        PlanProperties, SendableRecordBatchStream,
        execution_plan::{CardinalityEffect, EmissionType},
        filter_pushdown::{
            ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
        },
        stream::RecordBatchStreamAdapter,
    },
};
use futures::{StreamExt, TryStreamExt, future, stream};
use roaring::RoaringBitmap;

use crate::{
    document::{rebuild, render_value, slots},
    supertable::{
        handle::WeakReader,
        manifest::SuperfileUri,
        query::{
            SuperfileHit,
            exec::common::{live_reader, resolve_documents},
            vector::lookup_scanned_placements,
        },
        schema::{DocumentColumn, FieldId, TableSchema},
    },
};

/// Where the columns a document column draws on sit in a batch.
#[derive(Debug, Clone)]
pub(crate) struct DocumentBinding {
    /// The document column.
    document: DocumentColumn,
    /// The skeleton column's position.
    skeleton: usize,
    /// The residual column's position.
    residual: usize,
    /// Each other column of the batch that is a column of the table, by id.
    leaves: HashMap<FieldId, usize>,
}

impl DocumentBinding {
    /// Bind `document` to the columns of `schema`, matched by name against
    /// `table`. `None` when `schema` lacks the skeleton or residual column.
    pub(crate) fn bind(
        table: &TableSchema,
        document: &DocumentColumn,
        schema: &Schema,
    ) -> Option<Self> {
        let leaves = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(at, field)| Some((table.id_of(field.name())?, at)))
            .filter(|(id, _)| *id != document.skeleton_id && *id != document.residual_id)
            .collect();
        Some(Self {
            document: document.clone(),
            skeleton: schema.index_of(&document.skeleton).ok()?,
            residual: schema.index_of(&document.residual).ok()?,
            leaves,
        })
    }

    /// The document column over `batch`'s rows: each row's document rebuilt
    /// from its skeleton, residual and leaves, null where the skeleton is.
    pub(crate) fn rebuild(&self, batch: &RecordBatch) -> DfResult<ArrayRef> {
        let skeletons = as_large_strings(batch.column(self.skeleton))?;
        let residuals = as_large_strings(batch.column(self.residual))?;
        let mut out = LargeStringBuilder::with_capacity(batch.num_rows(), 0);
        for row in 0..batch.num_rows() {
            if skeletons.is_null(row) {
                out.append_null();
                continue;
            }
            let residual = (!residuals.is_null(row)).then(|| residuals.value(row));
            let value_of = |field: FieldId, element: Option<usize>| {
                let &at = self.leaves.get(&field)?;
                render_value(batch.column(at).as_ref(), row, element)
            };
            let document = rebuild(skeletons.value(row), residual, value_of).map_err(|e| {
                DataFusionError::Execution(format!("document column `{}`: {e}", self.document.name))
            })?;
            out.append_value(document);
        }
        Ok(Arc::new(out.finish()))
    }

    /// The document column this binding rebuilds.
    pub(crate) fn document(&self) -> &DocumentColumn {
        &self.document
    }
}

/// `array`, a string column of any encoding, as `LargeUtf8`.
fn as_large_strings(array: &ArrayRef) -> DfResult<LargeStringArray> {
    Ok(cast(array, &DataType::LargeUtf8)?
        .as_string::<i64>()
        .clone())
}

/// The ids of the columns `skeletons` name, a string column of any
/// encoding, each once. A skeleton that does not parse names nothing here;
/// its rebuild reports it.
pub(crate) fn referenced_fields(skeletons: &ArrayRef) -> DfResult<Vec<FieldId>> {
    let skeletons = as_large_strings(skeletons)?;
    let mut fields = Vec::new();
    for skeleton in skeletons.iter().flatten() {
        for slot in slots(skeleton).unwrap_or_default() {
            if let Some(field) = slot.field()
                && !fields.contains(&field)
            {
                fields.push(field);
            }
        }
    }
    Ok(fields)
}

/// The columns a projection of `wanted` (names, in order) has to read so
/// that every document column among them can be rebuilt: the names that
/// are not document columns, then each projected document's skeleton and
/// residual, then `leaves`, each once. `None` when no document column is
/// projected.
pub(crate) fn stored_inputs<'a>(
    wanted: &[&'a str],
    documents: &'a [DocumentColumn],
    leaves: &'a [String],
) -> Option<Vec<&'a str>> {
    let projected: Vec<&DocumentColumn> = documents
        .iter()
        .filter(|d| wanted.contains(&d.name.as_str()))
        .collect();
    if projected.is_empty() {
        return None;
    }
    let mut inputs: Vec<&str> = Vec::new();
    let mut push = |name: &'a str| {
        if !inputs.contains(&name) {
            inputs.push(name);
        }
    };
    for name in wanted {
        if !documents.iter().any(|d| d.name == *name) {
            push(name);
        }
    }
    for document in &projected {
        push(document.skeleton.as_str());
        push(document.residual.as_str());
    }
    for leaf in leaves {
        push(leaf.as_str());
    }
    Some(inputs)
}

/// What one output column of a [`DocumentExec`] is.
#[derive(Debug, Clone)]
pub(crate) enum Output {
    /// The input column at this position, passed through.
    Input(usize),
    /// A document column rebuilt from the input.
    Document(DocumentBinding),
}

/// Rebuilds the document columns of its output from the stored columns its
/// input reads, passing every other column through. One row out per row in,
/// in order.
pub(crate) struct DocumentExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    outputs: Arc<Vec<Output>>,
    properties: Arc<PlanProperties>,
    /// How the documents can be read after the rest of the plan instead
    /// ([`LateDocumentExec`]); `None` when this scan's rows cannot be found
    /// again by their `_id`.
    late: Option<Arc<LateRows>>,
}

impl fmt::Debug for DocumentExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DocumentExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl DocumentExec {
    /// `schema`'s columns over `input`: each a document column of `table`
    /// rebuilt from the input, or the input's column of the same name.
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
        table: &TableSchema,
    ) -> DfResult<Self> {
        let input_schema = input.schema();
        let outputs = schema
            .fields()
            .iter()
            .map(|field| match table.document_named(field.name()) {
                Some(document) => DocumentBinding::bind(table, &document, &input_schema)
                    .map(Output::Document)
                    .ok_or_else(|| {
                        DataFusionError::Internal(format!(
                            "document column `{}`: its stored columns are not in the scan",
                            document.name
                        ))
                    }),
                None => input_schema
                    .index_of(field.name())
                    .map(Output::Input)
                    .map_err(DataFusionError::from),
            })
            .collect::<DfResult<Vec<_>>>()?;
        let properties = output_properties(&input, &schema, &outputs);
        Ok(Self {
            input,
            schema,
            outputs: Arc::new(outputs),
            properties,
            late: None,
        })
    }

    /// This rebuild, which the plan may move after its limit: `late` finds
    /// the input's rows again by its `_id` column.
    pub(crate) fn with_late(mut self, late: Arc<LateRows>) -> Self {
        self.late = Some(late);
        self
    }

    /// The scan the documents are rebuilt over.
    pub(crate) fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// What each output column is.
    pub(crate) fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// How the documents can be read late, if they can.
    pub(crate) fn late(&self) -> Option<&Arc<LateRows>> {
        self.late.as_ref()
    }
}

/// The properties of a [`DocumentExec`] with `schema`'s `outputs` over
/// `input`: rows leave in the order they arrive, so each of the input's
/// orderings holds over the passed-through columns, up to its first key on
/// a column the output does not pass through. A planner that sorted below
/// this node relies on that order above it. The partitions are the
/// input's, their keys not carried.
fn output_properties(
    input: &Arc<dyn ExecutionPlan>,
    schema: &SchemaRef,
    outputs: &[Output],
) -> Arc<PlanProperties> {
    let mut to_output: HashMap<usize, usize> = HashMap::new();
    for (at, output) in outputs.iter().enumerate() {
        if let Output::Input(from) = output {
            to_output.entry(*from).or_insert(at);
        }
    }
    let orderings: Vec<Vec<PhysicalSortExpr>> = input
        .equivalence_properties()
        .oeq_class()
        .iter()
        .map(|ordering| {
            ordering
                .iter()
                .map_while(|sort| {
                    let column = sort.expr.downcast_ref::<Column>()?;
                    let at = *to_output.get(&column.index())?;
                    Some(PhysicalSortExpr {
                        expr: Arc::new(Column::new(column.name(), at)),
                        options: sort.options,
                    })
                })
                .collect::<Vec<_>>()
        })
        .filter(|ordering| !ordering.is_empty())
        .collect();
    let input_properties = input.properties();
    Arc::new(PlanProperties::new(
        EquivalenceProperties::new_with_orderings(Arc::clone(schema), orderings),
        Partitioning::UnknownPartitioning(input_properties.output_partitioning().partition_count()),
        input_properties.emission_type,
        input_properties.boundedness,
    ))
}

/// `batch`'s columns as `outputs` say, under `schema`.
fn project(batch: &RecordBatch, outputs: &[Output], schema: &SchemaRef) -> DfResult<RecordBatch> {
    let columns = outputs
        .iter()
        .map(|output| match output {
            Output::Input(at) => Ok(Arc::clone(batch.column(*at))),
            Output::Document(binding) => binding.rebuild(batch),
        })
        .collect::<DfResult<Vec<_>>>()?;
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
    )?)
}

impl DisplayAs for DocumentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let documents: Vec<&str> = self
            .outputs
            .iter()
            .filter_map(|output| match output {
                Output::Document(binding) => Some(binding.document.name.as_str()),
                Output::Input(_) => None,
            })
            .collect();
        write!(f, "DocumentExec: rebuild=[{}]", documents.join(", "))
    }
}

impl ExecutionPlan for DocumentExec {
    fn name(&self) -> &'static str {
        "DocumentExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "DocumentExec requires exactly one child; got {}",
                children.len()
            )));
        }
        let input = children.swap_remove(0);
        let properties = output_properties(&input, &self.schema, &self.outputs);
        Ok(Arc::new(Self {
            input,
            schema: Arc::clone(&self.schema),
            outputs: Arc::clone(&self.outputs),
            properties,
            late: self.late.clone(),
        }))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    /// One row out per row in, in order, so a `LIMIT` above may move below:
    /// then only the rows it keeps are rebuilt.
    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    /// The input's row count, the passed-through columns' statistics, and
    /// nothing for a rebuilt document column.
    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        let input = self.input.partition_statistics(partition)?;
        let column_statistics = self
            .outputs
            .iter()
            .map(|output| match output {
                Output::Input(at) => input
                    .column_statistics
                    .get(*at)
                    .cloned()
                    .unwrap_or_else(ColumnStatistics::new_unknown),
                Output::Document(_) => ColumnStatistics::new_unknown(),
            })
            .collect();
        Ok(Arc::new(Statistics {
            num_rows: input.num_rows,
            total_byte_size: input.total_byte_size,
            column_statistics,
        }))
    }

    /// A parent filter on passed-through columns moves into the input (the
    /// columns keep their names there); one on a document column cannot, as
    /// the input does not have it, and stays above.
    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DfResult<FilterDescription> {
        FilterDescription::from_children(parent_filters, &self.children())
    }

    fn handle_child_pushdown_result(
        &self,
        _phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &ConfigOptions,
    ) -> DfResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        Ok(FilterPushdownPropagation::if_all(child_pushdown_result))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let outputs = Arc::clone(&self.outputs);
        let schema = Arc::clone(&self.schema);
        let stream_schema = Arc::clone(&self.schema);
        let rebuilt = input.map(move |batch| project(&batch?, &outputs, &schema));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            rebuilt,
        )))
    }
}

/// What reading a scan's documents after the rest of the plan needs: the
/// table to read them from, and each row's place, found again from its
/// `_id` among the superfiles the scan read, at a row the scan did not skip
/// as deleted.
pub(crate) struct LateRows {
    /// The table, at the snapshot the scan read.
    reader: WeakReader,
    /// Each superfile the scan read, with the rows it skipped as deleted.
    scanned: HashMap<SuperfileUri, Arc<RoaringBitmap>>,
    /// The table's `_id` column, which the scan reads beside the rest.
    row_id: String,
}

impl fmt::Debug for LateRows {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateRows")
            .field("superfiles", &self.scanned.len())
            .field("row_id", &self.row_id)
            .finish()
    }
}

impl LateRows {
    pub(crate) fn new(
        reader: WeakReader,
        scanned: HashMap<SuperfileUri, Arc<RoaringBitmap>>,
        row_id: String,
    ) -> Self {
        Self {
            reader,
            scanned,
            row_id,
        }
    }

    /// The `_id` column the scan reads, by which each row is found again.
    pub(crate) fn row_id(&self) -> &str {
        &self.row_id
    }

    /// `batch` (`schema`'s columns, a placeholder at each of `documents`,
    /// then each row's `_id`) under `schema`, each of `documents` rebuilt at
    /// its row and the `_id` dropped.
    async fn rebuild(
        &self,
        batch: &RecordBatch,
        documents: &[(usize, DocumentColumn)],
        schema: &SchemaRef,
    ) -> DfResult<RecordBatch> {
        let rows = batch.num_rows();
        let ids = batch
            .column(batch.num_columns() - 1)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .filter(|ids| ids.null_count() == 0)
            .ok_or_else(|| {
                DataFusionError::Internal(format!(
                    "late documents: `{}` is not a non-null decimal column",
                    self.row_id
                ))
            })?
            .values()
            .to_vec();
        let reader = live_reader(&self.reader, "document column")?;
        let placements =
            lookup_scanned_placements(reader.manifest(), &ids, &reader.op_stats, &self.scanned)
                .await?;
        let hits = placements
            .into_iter()
            .zip(&ids)
            .map(|(placement, &id)| {
                let (entry, local) = placement.ok_or_else(|| {
                    DataFusionError::Internal(format!(
                        "late documents: no row the scan read has `{}` {id}",
                        self.row_id
                    ))
                })?;
                Ok(SuperfileHit {
                    superfile: entry.uri,
                    local_doc_id: local,
                    score: 0.0,
                    stable_id: Some(id),
                })
            })
            .collect::<DfResult<Vec<_>>>()?;
        // A document projected twice is read once.
        let mut unique: Vec<DocumentColumn> = Vec::new();
        for (_, document) in documents {
            if !unique.iter().any(|d| d.name == document.name) {
                unique.push(document.clone());
            }
        }
        let rebuilt = resolve_documents(&reader, &hits, &unique).await?;
        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .map(
                |(at, field)| match documents.iter().find(|(i, _)| *i == at) {
                    Some((_, document)) => {
                        let k = unique
                            .iter()
                            .position(|d| d.name == document.name)
                            .expect("every document is in `unique`");
                        Ok(cast(&rebuilt[k], field.data_type())?)
                    }
                    None => Ok(Arc::clone(batch.column(at))),
                },
            )
            .collect::<DfResult<Vec<_>>>()?;
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )?)
    }
}

/// The second read of a document column: above the plan's last limit or
/// top-k, its rows' documents rebuilt from just those rows.
///
/// ```text
///   LateDocumentExec   source ← rebuilt at each row's `_id`
///          │
///   GlobalLimitExec / SortExec TopK ...   (the rows that survive)
///          │
///   FilterExec ...                        (never reads `source`)
///          │
///   ProjectionExec     source ← NULL, _id#row ← _id
///          │
///   table scan         the other columns and `_id` only
/// ```
///
/// Its input carries the output's columns, a placeholder at each document,
/// then each row's `_id`. It gathers the input's rows (at most the limit),
/// places each at the superfile and row the scan read it from
/// ([`LateRows`]), and reads the documents at those rows alone
/// ([`resolve_documents`]). Rows leave in the order they arrived; every
/// column but the documents passes through.
pub(crate) struct LateDocumentExec {
    input: Arc<dyn ExecutionPlan>,
    schema: SchemaRef,
    /// Each rebuilt output column's position and document.
    documents: Arc<Vec<(usize, DocumentColumn)>>,
    late: Arc<LateRows>,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for LateDocumentExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateDocumentExec")
            .field("schema", &self.schema)
            .finish()
    }
}

impl LateDocumentExec {
    /// `replaced`'s output over `input`, which carries `replaced`'s columns,
    /// a placeholder at each of `documents`, and then each row's `_id`.
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        replaced: &Arc<dyn ExecutionPlan>,
        documents: Vec<(usize, DocumentColumn)>,
        late: Arc<LateRows>,
    ) -> DfResult<Self> {
        let schema = replaced.schema();
        let input_schema = input.schema();
        if input_schema.fields().len() != schema.fields().len() + 1 {
            return Err(DataFusionError::Internal(format!(
                "LateDocumentExec: input has {} columns for {} out",
                input_schema.fields().len(),
                schema.fields().len()
            )));
        }
        // Gathers its input before it emits.
        let properties = Arc::new(PlanProperties::new(
            replaced.equivalence_properties().clone(),
            input.output_partitioning().clone(),
            EmissionType::Final,
            replaced.boundedness(),
        ));
        Ok(Self {
            input,
            schema,
            documents: Arc::new(documents),
            late,
            properties,
        })
    }
}

impl DisplayAs for LateDocumentExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let documents: Vec<&str> = self
            .documents
            .iter()
            .map(|(_, document)| document.name.as_str())
            .collect();
        write!(f, "LateDocumentExec: rebuild=[{}]", documents.join(", "))
    }
}

impl ExecutionPlan for LateDocumentExec {
    fn name(&self) -> &'static str {
        "LateDocumentExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DfResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "LateDocumentExec requires exactly one child; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(Self {
            input: children.swap_remove(0),
            schema: Arc::clone(&self.schema),
            documents: Arc::clone(&self.documents),
            late: Arc::clone(&self.late),
            properties: Arc::clone(&self.properties),
        }))
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }

    /// The input's row count and the passed-through columns' statistics;
    /// nothing for a rebuilt document column.
    fn partition_statistics(&self, partition: Option<usize>) -> DfResult<Arc<Statistics>> {
        let input = self.input.partition_statistics(partition)?;
        let column_statistics = (0..self.schema.fields().len())
            .map(|at| match self.documents.iter().any(|(i, _)| *i == at) {
                true => ColumnStatistics::new_unknown(),
                false => input
                    .column_statistics
                    .get(at)
                    .cloned()
                    .unwrap_or_else(ColumnStatistics::new_unknown),
            })
            .collect();
        Ok(Arc::new(Statistics {
            num_rows: input.num_rows,
            total_byte_size: input.total_byte_size,
            column_statistics,
        }))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DfResult<SendableRecordBatchStream> {
        let input = self.input.execute(partition, context)?;
        let input_schema = self.input.schema();
        let schema = Arc::clone(&self.schema);
        let documents = Arc::clone(&self.documents);
        let late = Arc::clone(&self.late);
        let rebuilt = async move {
            let batches: Vec<RecordBatch> = input.try_collect().await?;
            let batch = concat_batches(&input_schema, &batches)?;
            if batch.num_rows() == 0 {
                return Ok(None);
            }
            late.rebuild(&batch, &documents, &schema).await.map(Some)
        };
        let batches = stream::once(rebuilt).filter_map(|batch| future::ready(batch.transpose()));
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            batches,
        )))
    }
}
