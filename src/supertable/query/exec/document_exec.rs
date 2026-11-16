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
//! handful of rows and read only the columns their skeletons name.

use std::{collections::HashMap, fmt, sync::Arc};

use arrow::compute::cast;
use arrow_array::{
    Array, ArrayRef, LargeStringArray, RecordBatch, RecordBatchOptions,
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
        execution_plan::CardinalityEffect,
        filter_pushdown::{
            ChildPushdownResult, FilterDescription, FilterPushdownPhase, FilterPushdownPropagation,
        },
        stream::RecordBatchStreamAdapter,
    },
};
use futures::StreamExt;

use crate::{
    document::{rebuild, render_value, slots},
    supertable::schema::{DocumentColumn, FieldId, TableSchema},
};

/// Where the columns a document column draws on sit in a batch.
#[derive(Debug, Clone)]
pub(crate) struct DocumentBinding {
    /// The document column's name, for errors.
    name: String,
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
            name: document.name.clone(),
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
                DataFusionError::Execution(format!("document column `{}`: {e}", self.name))
            })?;
            out.append_value(document);
        }
        Ok(Arc::new(out.finish()))
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
enum Output {
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
        })
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
                Output::Document(binding) => Some(binding.name.as_str()),
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
