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

use std::sync::Arc;

use arrow_schema::SchemaRef;
use datafusion::{
    common::{
        Result, ScalarValue,
        metadata::FieldMetadata,
        tree_node::{Transformed, TransformedResult, TreeNode},
    },
    error::DataFusionError,
    physical_expr::{
        PhysicalExpr,
        expressions::{CastExpr, Column, Literal},
    },
    physical_expr_adapter::{PhysicalExprAdapter, PhysicalExprAdapterFactory},
};

use crate::supertable::{
    manifest::ManifestSnapshot,
    schema::{
        LegacyNames, PhysicalKind, PhysicalSchema, TableSchema, cast::CAST_OPTIONS,
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
        Ok(Transformed::yes(Arc::new(CastExpr::new_with_target_field(
            Arc::new(resolved),
            Arc::new(logical_field.clone()),
            Some(CAST_OPTIONS),
        ))))
    }
}

#[cfg(test)]
mod tests {
    use arrow_schema::{DataType, Field, Schema};

    use super::*;
    use crate::supertable::schema::{FieldId, with_field_id};

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
    }
}
