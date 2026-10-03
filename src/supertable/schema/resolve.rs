// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How a batch meets the table's schema.
//!
//! A batch that matches the table — the same column names and types, in
//! any order — passes through with its arrays untouched. Otherwise the
//! schema grows from the data: a column the table does not have is added
//! with the batch's type, a nullable column the batch does not carry is
//! filled with nulls, and anything else is refused as a whole. A type
//! never changes from a batch; that takes a schema write.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{DataType, Field, Schema};

use super::{TableSchema, change::SchemaChange, error::SchemaError};

/// A column a batch adds to the table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddColumn {
    pub name: String,
    pub data_type: DataType,
}

/// A batch brought to the table's shape.
#[derive(Debug)]
pub struct ResolvedBatch {
    /// The table's columns in declared order, then the added columns in
    /// the order the batch carried them.
    pub batch: RecordBatch,
    /// The columns the batch adds, nullable, typed from the batch.
    pub added: Vec<AddColumn>,
}

/// Bring `batch` to `schema`'s shape, or refuse it whole.
pub fn resolve_batch(
    batch: &RecordBatch,
    schema: &TableSchema,
    id_column: &str,
) -> Result<ResolvedBatch, SchemaError> {
    let input = batch.schema();
    if input.fields().iter().any(|f| f.name() == id_column) {
        return Err(SchemaError::NameTaken {
            name: id_column.to_owned(),
        });
    }
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    for field in schema.fields() {
        match input.index_of(&field.name) {
            Ok(i) => {
                let array = batch.column(i);
                if array.data_type() != &field.data_type {
                    return Err(SchemaError::TypeMismatch {
                        column: field.name.clone(),
                        frozen: field.data_type.clone(),
                        offered: array.data_type().clone(),
                    });
                }
                if !field.nullable && array.null_count() > 0 {
                    return Err(SchemaError::NullInNonNullable {
                        column: field.name.clone(),
                    });
                }
                columns.push(Arc::clone(array));
            }
            Err(_) if field.nullable => {
                columns.push(new_null_array(&field.data_type, batch.num_rows()));
            }
            Err(_) => {
                return Err(SchemaError::MissingColumn {
                    column: field.name.clone(),
                });
            }
        }
        fields.push(Field::new(
            &field.name,
            field.data_type.clone(),
            field.nullable,
        ));
    }
    let mut added = Vec::new();
    for (i, field) in input.fields().iter().enumerate() {
        if schema.id_of(field.name()).is_some() {
            continue;
        }
        added.push(AddColumn {
            name: field.name().clone(),
            data_type: field.data_type().clone(),
        });
        columns.push(Arc::clone(batch.column(i)));
        fields.push(Field::new(field.name(), field.data_type().clone(), true));
    }
    let current = schema.fields().len() as u32;
    if current + added.len() as u32 > schema.max_fields() {
        return Err(SchemaError::FieldCapExceeded {
            cap: schema.max_fields(),
            current,
            fields: added.into_iter().map(|a| a.name).collect(),
        });
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .expect("columns were taken from one batch and share its row count");
    Ok(ResolvedBatch { batch, added })
}

/// The schema a commit of `batches` publishes over `current`: `current`
/// plus a nullable column for every column a batch carries that the table
/// does not have, typed from the batch. `None` when nothing is added.
///
/// Each batch was resolved against the schema of its own append, so by the
/// time they commit together the table may have moved: a column one batch
/// adds may now exist (same type: it is stored; another type: refused),
/// and two batches may add the same column (same type, or refused). The
/// id column is the batch's first column and is not a table field.
pub fn union_schema<'a>(
    current: &TableSchema,
    batches: impl IntoIterator<Item = &'a RecordBatch>,
    id_column: &str,
) -> Result<Option<TableSchema>, SchemaError> {
    let mut added: Vec<AddColumn> = Vec::new();
    for batch in batches {
        for field in batch.schema().fields() {
            if field.name() == id_column {
                continue;
            }
            let frozen = match current.id_of(field.name()) {
                Some(id) => current
                    .fields()
                    .iter()
                    .find(|f| f.id == id)
                    .map(|f| &f.data_type),
                None => added
                    .iter()
                    .find(|a| a.name == *field.name())
                    .map(|a| &a.data_type),
            };
            match frozen {
                Some(frozen) if frozen != field.data_type() => {
                    return Err(SchemaError::TypeMismatch {
                        column: field.name().clone(),
                        frozen: frozen.clone(),
                        offered: field.data_type().clone(),
                    });
                }
                Some(_) => {}
                None => added.push(AddColumn {
                    name: field.name().clone(),
                    data_type: field.data_type().clone(),
                }),
            }
        }
    }
    if added.is_empty() {
        return Ok(None);
    }
    let changes: Vec<SchemaChange> = added
        .into_iter()
        .map(|a| SchemaChange::AddColumn {
            name: a.name,
            data_type: a.data_type,
            nullable: true,
            index: None,
        })
        .collect();
    current.apply(&changes).map(Some)
}

/// `batch` in `target`'s shape: `target`'s columns in its order, taken from
/// `batch` by name, null-filled where `batch` lacks them. Every column
/// `batch` has must be in `target` with its type; [`union_schema`] over
/// the batch guarantees that.
pub fn conform(batch: &RecordBatch, target: &Arc<Schema>) -> RecordBatch {
    if batch.schema().fields() == target.fields() {
        return batch.clone();
    }
    let columns: Vec<ArrayRef> = target
        .fields()
        .iter()
        .map(|field| match batch.schema().index_of(field.name()) {
            Ok(i) => Arc::clone(batch.column(i)),
            Err(_) => new_null_array(field.data_type(), batch.num_rows()),
        })
        .collect();
    RecordBatch::try_new(Arc::clone(target), columns)
        .expect("every column is the batch's or a null array of the batch's row count")
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Decimal128Array, Int64Array, LargeStringArray};

    use super::{
        super::{DECIMAL128_PRECISION, DECIMAL128_SCALE, FieldId},
        *,
    };

    fn table() -> TableSchema {
        TableSchema::from_user_schema(&Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("score", DataType::Int64, true),
        ]))
    }

    fn batch(fields: Vec<(&str, ArrayRef)>) -> RecordBatch {
        let schema = Schema::new(
            fields
                .iter()
                .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
                .collect::<Vec<_>>(),
        );
        RecordBatch::try_new(
            Arc::new(schema),
            fields.into_iter().map(|(_, a)| a).collect(),
        )
        .expect("batch")
    }

    fn titles() -> ArrayRef {
        Arc::new(LargeStringArray::from(vec!["a", "b"]))
    }

    fn ints(v: Vec<Option<i64>>) -> ArrayRef {
        Arc::new(Int64Array::from(v))
    }

    #[test]
    fn a_matching_batch_passes_through_with_its_arrays_in_table_order() {
        let t = table();
        let scores = ints(vec![Some(1), Some(2)]);
        let b = batch(vec![("score", Arc::clone(&scores)), ("title", titles())]);
        let resolved = resolve_batch(&b, &t, "_id").expect("resolve");
        assert!(resolved.added.is_empty());
        let names: Vec<String> = resolved
            .batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["title", "score"]);
        assert!(Arc::ptr_eq(resolved.batch.column(1), &scores), "no copy");
    }

    #[test]
    fn an_unknown_column_is_added_and_an_absent_nullable_one_is_null_filled() {
        let t = table();
        let b = batch(vec![("title", titles()), ("tag", titles())]);
        let resolved = resolve_batch(&b, &t, "_id").expect("resolve");
        assert_eq!(
            resolved.added,
            vec![AddColumn {
                name: "tag".into(),
                data_type: DataType::LargeUtf8
            }]
        );
        let names: Vec<String> = resolved
            .batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        assert_eq!(names, ["title", "score", "tag"]);
        assert_eq!(
            resolved.batch.column(1).null_count(),
            2,
            "score is null-filled"
        );
    }

    #[test]
    fn a_different_type_a_missing_required_column_and_a_null_in_it_are_refused() {
        let t = table();
        let wrong = batch(vec![("title", titles()), ("score", titles())]);
        assert!(matches!(
            resolve_batch(&wrong, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, frozen: DataType::Int64, offered: DataType::LargeUtf8 }) if column == "score"
        ));
        let missing = batch(vec![("score", ints(vec![Some(1), Some(2)]))]);
        assert!(matches!(
            resolve_batch(&missing, &t, "_id"),
            Err(SchemaError::MissingColumn { column }) if column == "title"
        ));
        let nulls: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None]));
        let with_null = batch(vec![("title", nulls)]);
        assert!(matches!(
            resolve_batch(&with_null, &t, "_id"),
            Err(SchemaError::NullInNonNullable { column }) if column == "title"
        ));
        // A mismatch alongside an unknown column refuses the whole batch.
        let both = batch(vec![
            ("title", titles()),
            ("score", titles()),
            ("tag", titles()),
        ]);
        assert!(matches!(
            resolve_batch(&both, &t, "_id"),
            Err(SchemaError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn the_id_column_and_the_field_cap_are_enforced() {
        let t = table();
        let with_id = batch(vec![
            ("title", titles()),
            ("_id", ints(vec![Some(1), Some(2)])),
        ]);
        assert!(matches!(
            resolve_batch(&with_id, &t, "_id"),
            Err(SchemaError::NameTaken { name }) if name == "_id"
        ));
        let mut capped = table();
        capped.max_fields = 2;
        let over = batch(vec![("title", titles()), ("tag", titles())]);
        assert!(matches!(
            resolve_batch(&over, &capped, "_id"),
            Err(SchemaError::FieldCapExceeded { cap: 2, current: 2, fields }) if fields == vec!["tag".to_string()]
        ));
    }

    #[test]
    fn the_union_adds_once_and_refuses_a_type_that_disagrees() {
        let t = table();
        let a = batch(vec![
            ("title", titles()),
            ("tag", ints(vec![Some(1), None])),
        ]);
        let b = batch(vec![
            ("title", titles()),
            ("tag", ints(vec![None, Some(2)])),
        ]);
        let union = union_schema(&t, [&a, &b], "_id")
            .expect("union")
            .expect("tag is new");
        assert_eq!(union.id_of("tag"), Some(FieldId(3)));
        assert_eq!(union.schema_id(), t.schema_id() + 1);
        assert!(
            union_schema(&t, [&batch(vec![("title", titles())])], "_id")
                .expect("nothing new")
                .is_none()
        );
        let other = batch(vec![(
            "tag",
            Arc::new(LargeStringArray::from(vec!["x", "y"])) as ArrayRef,
        )]);
        assert!(matches!(
            union_schema(&t, [&a, &other], "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "tag"
        ));
        assert!(matches!(
            union_schema(&t, [&batch(vec![("score", Arc::new(LargeStringArray::from(vec!["x", "y"])) as ArrayRef)])], "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "score"
        ));

        let target = union.scalar_schema("_id");
        let ids: ArrayRef = Arc::new(
            Decimal128Array::from(vec![7i128, 8])
                .with_precision_and_scale(DECIMAL128_PRECISION, DECIMAL128_SCALE)
                .expect("id type"),
        );
        let shaped = conform(&batch(vec![("_id", ids), ("title", titles())]), &target);
        assert_eq!(shaped.schema().fields(), target.fields());
        assert_eq!(shaped.column(3).null_count(), 2, "tag is null-filled");
        assert_eq!(shaped.column(2).null_count(), 2, "score is null-filled");
        assert_eq!(shaped.column(1).null_count(), 0, "title is the batch's");
    }
}
