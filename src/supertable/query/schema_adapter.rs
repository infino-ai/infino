// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The SQL scan's view of one file.
//!
//! DataFusion plans a scan against the table's schema and, per file, asks
//! an adapter to rewrite each predicate and projection expression against
//! the file's own schema. The default adapter resolves a column by name,
//! which is wrong once a column has been renamed and casts with options the
//! engine does not control. This one resolves every column through the
//! file's [`FileSchemaMap`]: by field id, by creation name for a file
//! written before ids, to a typed null when the file predates the column,
//! and through the engine's one cast kernel when the file holds the column
//! in another type. A file matching the table rewrites nothing.

use std::{
    fmt,
    hash::{Hash, Hasher},
    sync::Arc,
};

use arrow_array::RecordBatch;
use arrow_schema::{DataType, FieldRef, Schema, SchemaRef};
use datafusion::{
    common::{
        Result, ScalarValue,
        metadata::FieldMetadata,
        tree_node::{Transformed, TransformedResult, TreeNode},
    },
    error::DataFusionError,
    logical_expr::ColumnarValue,
    physical_expr::{
        PhysicalExpr,
        expressions::{CastExpr, Column, Literal},
    },
    physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory},
};

use crate::supertable::{
    manifest::ManifestSnapshot,
    schema::{
        LegacyNames, PhysicalKind, PhysicalSchema, TableSchema,
        cast::{CAST_OPTIONS, null_padded_numbers, parses_text_as_number},
        map::FileSchemaMap,
    },
};

/// Builds one [`TableExprAdapter`] per scanned file, against the table
/// schema of the snapshot the scan was planned on.
#[derive(Debug)]
pub(crate) struct TableExprAdapterFactory {
    table: Arc<TableSchema>,
    id_column: String,
    legacy: LegacyNames,
}

impl TableExprAdapterFactory {
    pub(crate) fn new(manifest: &ManifestSnapshot) -> Self {
        Self {
            table: manifest.table_schema(),
            id_column: manifest.options.id_column.clone(),
            legacy: manifest.options.legacy_names(),
        }
    }
}

impl PhysicalExprAdapterFactory for TableExprAdapterFactory {
    fn create(
        &self,
        logical_file_schema: SchemaRef,
        physical_file_schema: SchemaRef,
    ) -> Result<Arc<dyn PhysicalExprAdapter>> {
        let map = FileSchemaMap::new(
            &self.table,
            &self.id_column,
            &PhysicalSchema::of_stored_schema(&physical_file_schema).resolved(&self.legacy),
        );
        Ok(Arc::new(TableExprAdapter {
            logical: logical_file_schema,
            physical: physical_file_schema,
            map,
        }))
    }
}

/// Rewrites expressions planned on the table schema (`logical`) to run on
/// one file's schema (`physical`).
#[derive(Debug)]
struct TableExprAdapter {
    logical: SchemaRef,
    physical: SchemaRef,
    map: FileSchemaMap,
}

impl PhysicalExprAdapter for TableExprAdapter {
    fn rewrite(&self, expr: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
        expr.transform(|expr| {
            let Some(column) = expr.downcast_ref::<Column>().cloned() else {
                return Ok(Transformed::no(expr));
            };
            self.rewrite_column(expr, &column)
        })
        .data()
    }
}

impl TableExprAdapter {
    fn rewrite_column(
        &self,
        expr: Arc<dyn PhysicalExpr>,
        column: &Column,
    ) -> Result<Transformed<Arc<dyn PhysicalExpr>>> {
        // A column the plan does not know is not the table's to resolve.
        let Ok(logical_field) = self.logical.field_with_name(column.name()) else {
            return Ok(Transformed::no(expr));
        };
        let stored = self
            .map
            .resolve(column.name())
            .and_then(|r| r.physical.as_ref())
            .filter(|p| p.kind == PhysicalKind::Stored);
        let Some(stored) = stored else {
            if !logical_field.is_nullable() {
                return Err(DataFusionError::Execution(format!(
                    "column `{}` is not nullable but the file does not hold it",
                    column.name()
                )));
            }
            let null = ScalarValue::Null.cast_to(logical_field.data_type())?;
            return Ok(Transformed::yes(Arc::new(Literal::new_with_metadata(
                null,
                Some(FieldMetadata::from(logical_field)),
            ))));
        };
        let resolved = Column::new_with_schema(&stored.name, self.physical.as_ref())?;
        if stored.data_type == *logical_field.data_type() {
            if resolved.index() == column.index() && resolved.name() == column.name() {
                return Ok(Transformed::no(expr));
            }
            return Ok(Transformed::yes(Arc::new(resolved)));
        }
        // DataFusion's cast parses text with Arrow's rules; the engine's
        // whitespace rule (see `schema::cast`) runs first, so a scan reads
        // what the cast kernel and compaction read.
        let input: Arc<dyn PhysicalExpr> =
            if parses_text_as_number(&stored.data_type, logical_field.data_type()) {
                Arc::new(NullPaddedNumbers {
                    input: Arc::new(resolved),
                    to: logical_field.data_type().clone(),
                })
            } else {
                Arc::new(resolved)
            };
        Ok(Transformed::yes(Arc::new(CastExpr::new_with_target_field(
            input,
            Arc::new(logical_field.clone()),
            Some(CAST_OPTIONS),
        ))))
    }
}

/// Text about to be cast to the number type `to`, with each value that has
/// whitespace around its digits set to null first: the cast kernel's rule,
/// applied inside a scan.
#[derive(Debug, Eq)]
struct NullPaddedNumbers {
    input: Arc<dyn PhysicalExpr>,
    to: DataType,
}

// `Arc<dyn PhysicalExpr>` compares and hashes through the trait object, so
// these are written out rather than derived.
impl PartialEq for NullPaddedNumbers {
    fn eq(&self, other: &Self) -> bool {
        self.input.eq(&other.input) && self.to == other.to
    }
}

impl Hash for NullPaddedNumbers {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.input.hash(state);
        self.to.hash(state);
    }
}

impl fmt::Display for NullPaddedNumbers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "null_padded_numbers({})", self.input)
    }
}

impl PhysicalExpr for NullPaddedNumbers {
    fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> {
        let field = self.input.return_field(input_schema)?;
        Ok(Arc::new(field.as_ref().clone().with_nullable(true)))
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        match self.input.evaluate(batch)? {
            ColumnarValue::Array(array) => {
                Ok(ColumnarValue::Array(null_padded_numbers(&array, &self.to)?))
            }
            ColumnarValue::Scalar(scalar) => {
                let array = null_padded_numbers(&scalar.to_array()?, &self.to)?;
                Ok(ColumnarValue::Scalar(ScalarValue::try_from_array(
                    &array, 0,
                )?))
            }
        }
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(format!(
                "NullPaddedNumbers takes exactly one input; got {}",
                children.len()
            )));
        }
        Ok(Arc::new(NullPaddedNumbers {
            input: children.swap_remove(0),
            to: self.to.clone(),
        }))
    }

    fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // In SQL terms it is the column itself; the cast above it is what
        // shows.
        self.input.fmt_sql(f)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Decimal128Array, LargeStringArray};
    use arrow_schema::Field;

    use super::*;
    use crate::supertable::schema::{FieldId, with_field_id};

    /// Precision of the `_id` column, as the fixtures store it.
    const ID_PRECISION: u8 = 38;

    fn table() -> Arc<TableSchema> {
        Arc::new(TableSchema::from_user_schema(&Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, true),
            Field::new("score", DataType::Int64, true),
        ])))
    }

    fn factory() -> TableExprAdapterFactory {
        let table = table();
        TableExprAdapterFactory {
            legacy: LegacyNames::new(Arc::clone(&table), "_id"),
            table,
            id_column: "_id".into(),
        }
    }

    fn logical() -> SchemaRef {
        table().stored_schema("_id")
    }

    fn stamped(name: &str, id: u32, data_type: DataType) -> Field {
        with_field_id(&Field::new(name, data_type, true), FieldId(id))
    }

    #[test]
    fn a_matching_file_leaves_columns_alone() {
        let logical = logical();
        let adapter = factory()
            .create(Arc::clone(&logical), Arc::clone(&logical))
            .expect("adapter");
        let expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("score", 2));
        let out = adapter.rewrite(Arc::clone(&expr)).expect("rewrite");
        assert!(out.downcast_ref::<Column>().is_some_and(|c| c.index() == 2));
    }

    #[test]
    fn a_renamed_column_resolves_by_id_and_a_missing_one_becomes_a_null_literal() {
        let physical: SchemaRef = Arc::new(Schema::new(vec![
            stamped("_id", 0, DataType::Decimal128(38, 0)),
            stamped("points", 2, DataType::Int64),
        ]));
        let adapter = factory().create(logical(), physical).expect("adapter");
        let score = adapter
            .rewrite(Arc::new(Column::new("score", 2)))
            .expect("rewrite");
        let score = score.downcast_ref::<Column>().expect("column");
        assert_eq!((score.name(), score.index()), ("points", 1));

        let title = adapter
            .rewrite(Arc::new(Column::new("title", 1)))
            .expect("rewrite");
        let title = title.downcast_ref::<Literal>().expect("null literal");
        assert_eq!(*title.value(), ScalarValue::LargeUtf8(None));

        // The id column is never nullable and every file holds it; a file
        // without it is refused rather than read as null.
        let no_id: SchemaRef = Arc::new(Schema::new(vec![stamped("points", 2, DataType::Int64)]));
        let adapter = factory().create(logical(), no_id).expect("adapter");
        assert!(adapter.rewrite(Arc::new(Column::new("_id", 0))).is_err());
    }

    #[test]
    fn a_column_in_another_type_is_cast_with_the_engine_kernel() {
        let physical: SchemaRef = Arc::new(Schema::new(vec![
            stamped("_id", 0, DataType::Decimal128(38, 0)),
            stamped("title", 1, DataType::LargeUtf8),
            stamped("score", 2, DataType::Int32),
        ]));
        let adapter = factory().create(logical(), physical).expect("adapter");
        let score = adapter
            .rewrite(Arc::new(Column::new("score", 2)))
            .expect("rewrite");
        let cast = score.downcast_ref::<CastExpr>().expect("cast");
        assert_eq!(cast.cast_type(), &DataType::Int64);
        assert!(
            cast.cast_options().safe,
            "a value that does not fit is null, not an error"
        );
        assert!(
            cast.expr().downcast_ref::<Column>().is_some(),
            "a number read as another number needs no text rule"
        );
    }

    #[test]
    fn text_read_as_a_number_keeps_the_whitespace_rule() {
        let physical: SchemaRef = Arc::new(Schema::new(vec![
            stamped("_id", 0, DataType::Decimal128(ID_PRECISION, 0)),
            stamped("title", 1, DataType::LargeUtf8),
            stamped("score", 2, DataType::LargeUtf8),
        ]));
        let adapter = factory()
            .create(logical(), Arc::clone(&physical))
            .expect("adapter");
        let score = adapter
            .rewrite(Arc::new(Column::new("score", 2)))
            .expect("rewrite");
        let cast = score.downcast_ref::<CastExpr>().expect("cast");
        assert!(
            cast.expr().downcast_ref::<NullPaddedNumbers>().is_some(),
            "the text is guarded before DataFusion parses it"
        );

        let ids = Decimal128Array::from(vec![1, 2, 3])
            .with_precision_and_scale(ID_PRECISION, 0)
            .expect("decimal ids");
        let batch = RecordBatch::try_new(
            physical,
            vec![
                Arc::new(ids),
                Arc::new(LargeStringArray::from(vec!["a", "b", "c"])),
                Arc::new(LargeStringArray::from(vec!["5", " 5", "5 "])),
            ],
        )
        .expect("batch");
        let ColumnarValue::Array(scores) = score.evaluate(&batch).expect("evaluate") else {
            panic!("a column evaluates to an array");
        };
        assert!(scores.is_valid(0));
        assert!(scores.is_null(1));
        assert!(scores.is_null(2));
    }
}
