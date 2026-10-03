// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How one superfile's columns line up with the table's.
//!
//! A table's schema moves while its files stay as they were written: a
//! column may have been added since a file was written, renamed, or
//! retyped, and a file written before ids existed names its columns only
//! by name. [`FileSchemaMap`] resolves the table's columns against one
//! file's [`PhysicalSchema`] once, by field id (or by creation name for a
//! file without ids), and every path that reads the file — hit
//! materialisation, the SQL scan, compaction — asks it where a column is
//! and how its values reach the table's type. A file whose columns match
//! the table's by name and type takes the identity path, which is how
//! every file reads today.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, new_null_array};
use arrow_schema::{ArrowError, DataType, SchemaRef};

use super::{
    FieldId, PhysicalColumn, PhysicalKind, PhysicalSchema, TableSchema, cast::cast_column,
};

/// Where a table column is in one file.
#[derive(Debug, Clone, PartialEq)]
pub struct Resolution {
    /// The table column's id.
    pub id: FieldId,
    /// The table column's current name.
    pub name: String,
    /// The table column's current type.
    pub data_type: DataType,
    /// The file's column that holds it, or `None` when the file was
    /// written before the column existed (it reads as null).
    pub physical: Option<PhysicalColumn>,
}

impl Resolution {
    /// Whether reading this column from the file needs a cast.
    pub fn needs_cast(&self) -> bool {
        self.physical
            .as_ref()
            .is_some_and(|p| !types_agree(p, &self.data_type))
    }
}

/// Whether the file holds the column in the table's type. A vector column
/// lives in the vector blob as `dim` floats; its declared list type's item
/// nullability is not something the blob records, so only the dimension
/// is compared for it.
fn types_agree(physical: &PhysicalColumn, table_type: &DataType) -> bool {
    match physical.kind {
        PhysicalKind::Vector => matches!(
            (&physical.data_type, table_type),
            (DataType::FixedSizeList(_, a), DataType::FixedSizeList(_, b)) if a == b
        ),
        PhysicalKind::Stored | PhysicalKind::FtsIndexOnly => physical.data_type == *table_type,
    }
}

/// The table's columns resolved against one file.
#[derive(Debug, Clone)]
pub struct FileSchemaMap {
    columns: Vec<Resolution>,
    identity: bool,
}

impl FileSchemaMap {
    /// Resolve `table`'s columns (the id column `id_column` first) against
    /// `physical`, by field id. A file written before ids has its columns
    /// resolved to ids when its physical schema is derived
    /// ([`PhysicalSchema::resolved`]), so by the time a file is read here
    /// every column it holds is named by id.
    pub fn new(table: &TableSchema, id_column: &str, physical: &PhysicalSchema) -> Self {
        let find = |id: FieldId| -> Option<PhysicalColumn> { physical.column_by_id(id).cloned() };
        let id_field = TableSchema::id_field(id_column);
        let mut columns = vec![Resolution {
            id: FieldId::ID_COLUMN,
            name: id_column.to_owned(),
            data_type: id_field.data_type().clone(),
            physical: find(FieldId::ID_COLUMN),
        }];
        columns.extend(table.fields().iter().map(|f| Resolution {
            id: f.id,
            name: f.name.clone(),
            data_type: f.data_type.clone(),
            physical: find(f.id),
        }));
        let identity = columns.iter().all(|r| {
            r.physical
                .as_ref()
                .is_some_and(|p| p.name == r.name && types_agree(p, &r.data_type))
        });
        Self { columns, identity }
    }

    /// Whether every table column is in the file under its current name
    /// and type, so the file reads by name with no adaptation.
    pub fn is_identity(&self) -> bool {
        self.identity
    }

    /// Whether the file holds some column in a type other than the
    /// table's: what makes it a compaction input regardless of size.
    pub fn has_stale_type(&self) -> bool {
        self.columns.iter().any(Resolution::needs_cast)
    }

    /// The columns the file holds in a type other than the table's.
    pub fn stale_columns(&self) -> impl Iterator<Item = FieldId> + '_ {
        self.columns.iter().filter(|r| r.needs_cast()).map(|r| r.id)
    }

    /// Where the table column named `name` is in the file; `None` for a
    /// name the table does not have.
    pub fn resolve(&self, name: &str) -> Option<&Resolution> {
        self.columns.iter().find(|r| r.name == name)
    }

    /// Every table column's resolution, the id column first.
    pub fn columns(&self) -> &[Resolution] {
        &self.columns
    }

    /// Whether the file holds the column with id `id` in a form the index
    /// path can use: `kind` says which form.
    pub fn holds(&self, id: FieldId, kind: PhysicalKind) -> bool {
        self.columns.iter().any(|r| {
            r.id == id
                && r.physical.as_ref().is_some_and(|p| match kind {
                    // A stored text column feeds the FTS blob as well.
                    PhysicalKind::FtsIndexOnly => {
                        p.kind == PhysicalKind::FtsIndexOnly || p.kind == PhysicalKind::Stored
                    }
                    other => p.kind == other,
                })
        })
    }

    /// The file's names of the Parquet columns that hold the table columns
    /// in `names`: what to project out of the file. A column the file does
    /// not store contributes no name.
    pub fn stored_names<'a>(&'a self, names: &[&str]) -> Vec<&'a str> {
        names
            .iter()
            .filter_map(|name| self.resolve(name))
            .filter_map(|r| r.physical.as_ref())
            .filter(|p| p.kind == PhysicalKind::Stored)
            .map(|p| p.name.as_str())
            .collect()
    }

    /// `batch`, read from the file with columns under their file names,
    /// reshaped to `out`: every field of `out` is a table column, taken
    /// from its file column (cast when the types differ) or null-filled
    /// when the file has none.
    pub fn adapt(&self, batch: &RecordBatch, out: &SchemaRef) -> Result<RecordBatch, ArrowError> {
        let n = batch.num_rows();
        let columns = out
            .fields()
            .iter()
            .map(|field| -> Result<ArrayRef, ArrowError> {
                let resolution = self.resolve(field.name()).ok_or_else(|| {
                    ArrowError::SchemaError(format!(
                        "column `{}` is not a table column",
                        field.name()
                    ))
                })?;
                let Some(physical) = resolution
                    .physical
                    .as_ref()
                    .filter(|p| p.kind == PhysicalKind::Stored)
                else {
                    return Ok(new_null_array(field.data_type(), n));
                };
                let array = batch.column_by_name(&physical.name).ok_or_else(|| {
                    ArrowError::SchemaError(format!(
                        "file column `{}` was not read for table column `{}`",
                        physical.name,
                        field.name()
                    ))
                })?;
                if array.data_type() == field.data_type() {
                    Ok(Arc::clone(array))
                } else {
                    cast_column(array, field.data_type())
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        RecordBatch::try_new(Arc::clone(out), columns)
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int32Array, Int64Array, LargeStringArray};
    use arrow_schema::{Field, Schema};

    use super::{super::LegacyNames, *};

    fn table() -> TableSchema {
        TableSchema::from_user_schema(&Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("score", DataType::Int64, true),
        ]))
    }

    fn legacy(table: &TableSchema) -> LegacyNames {
        LegacyNames::new(Arc::new(table.clone()), "_id")
    }

    fn column(name: &str, id: Option<FieldId>, data_type: DataType) -> PhysicalColumn {
        PhysicalColumn {
            name: name.into(),
            id,
            data_type,
            kind: PhysicalKind::Stored,
        }
    }

    fn physical(columns: Vec<PhysicalColumn>) -> PhysicalSchema {
        let fields: Vec<Field> = columns
            .iter()
            .map(|c| {
                let f = Field::new(&c.name, c.data_type.clone(), true);
                match c.id {
                    Some(id) => super::super::with_field_id(&f, id),
                    None => f,
                }
            })
            .collect();
        PhysicalSchema::of_stored_schema(&Schema::new(fields))
    }

    #[test]
    fn a_file_matching_the_table_takes_the_identity_path() {
        let t = table();
        let current = physical(vec![
            column(
                "_id",
                Some(FieldId::ID_COLUMN),
                TableSchema::id_field("_id").data_type().clone(),
            ),
            column("title", Some(FieldId(1)), DataType::LargeUtf8),
            column("score", Some(FieldId(2)), DataType::Int64),
        ]);
        let map = FileSchemaMap::new(&t, "_id", &current);
        assert!(map.is_identity());
        assert!(!map.has_stale_type());
        assert_eq!(
            map.stored_names(&["score", "title"]),
            vec!["score", "title"]
        );
    }

    #[test]
    fn a_file_without_ids_resolves_by_its_creation_names() {
        let t = table();
        let old = physical(vec![
            column(
                "_id",
                None,
                TableSchema::id_field("_id").data_type().clone(),
            ),
            column("title", None, DataType::LargeUtf8),
            column("score", None, DataType::Int64),
        ])
        .resolved(&legacy(&t));
        let map = FileSchemaMap::new(&t, "_id", &old);
        assert!(
            map.is_identity(),
            "a pre-id file with the same names and types is current"
        );
        assert_eq!(
            map.resolve("score")
                .and_then(|r| r.physical.as_ref())
                .map(|p| p.name.as_str()),
            Some("score")
        );
    }

    #[test]
    fn a_renamed_column_resolves_by_id_and_a_missing_one_reads_as_null() {
        let t = table();
        // The file was written when `score` was called `points`, before `title` existed.
        let old = physical(vec![
            column(
                "_id",
                Some(FieldId::ID_COLUMN),
                TableSchema::id_field("_id").data_type().clone(),
            ),
            column("points", Some(FieldId(2)), DataType::Int64),
        ]);
        let map = FileSchemaMap::new(&t, "_id", &old);
        assert!(!map.is_identity());
        assert!(!map.has_stale_type());
        let score = map.resolve("score").expect("table column");
        assert_eq!(
            score.physical.as_ref().map(|p| p.name.as_str()),
            Some("points")
        );
        assert!(
            map.resolve("title")
                .expect("table column")
                .physical
                .is_none()
        );
        assert_eq!(map.stored_names(&["title", "score"]), vec!["points"]);

        let read = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "points",
                DataType::Int64,
                true,
            )])),
            vec![Arc::new(Int64Array::from(vec![7, 9]))],
        )
        .expect("batch");
        let out: SchemaRef = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, true),
            Field::new("score", DataType::Int64, true),
        ]));
        let adapted = map.adapt(&read, &out).expect("adapt");
        assert_eq!(
            adapted.column(0).null_count(),
            2,
            "an absent column is null"
        );
        let score = adapted
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64");
        assert_eq!(score.values(), &[7, 9]);
    }

    #[test]
    fn a_column_in_another_type_is_stale_and_cast_on_read() {
        let t = table();
        let narrow = physical(vec![
            column(
                "_id",
                Some(FieldId::ID_COLUMN),
                TableSchema::id_field("_id").data_type().clone(),
            ),
            column("title", Some(FieldId(1)), DataType::LargeUtf8),
            column("score", Some(FieldId(2)), DataType::Int32),
        ]);
        let map = FileSchemaMap::new(&t, "_id", &narrow);
        assert!(!map.is_identity());
        assert!(map.has_stale_type());
        let read = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("score", DataType::Int32, true),
                Field::new("title", DataType::LargeUtf8, true),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![3])),
                Arc::new(LargeStringArray::from(vec!["a"])),
            ],
        )
        .expect("batch");
        let out: SchemaRef = Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Int64,
            true,
        )]));
        let adapted = map.adapt(&read, &out).expect("adapt");
        let score = adapted
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64");
        assert_eq!(score.values(), &[3]);
    }

    #[test]
    fn index_presence_follows_the_physical_kind_and_survives_ipc() {
        let t = TableSchema::from_user_schema(&Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4),
                true,
            ),
        ]));
        let p = PhysicalSchema::new(vec![
            column(
                "_id",
                Some(FieldId::ID_COLUMN),
                TableSchema::id_field("_id").data_type().clone(),
            ),
            PhysicalColumn {
                name: "title".into(),
                id: Some(FieldId(1)),
                data_type: DataType::LargeUtf8,
                kind: PhysicalKind::FtsIndexOnly,
            },
            PhysicalColumn {
                name: "emb".into(),
                id: Some(FieldId(2)),
                data_type: DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, false)),
                    4,
                ),
                kind: PhysicalKind::Vector,
            },
        ]);
        let p = PhysicalSchema::from_ipc(&p.to_ipc()).expect("decode");
        assert_eq!(p.columns()[1].kind, PhysicalKind::FtsIndexOnly);
        assert_eq!(p.columns()[2].kind, PhysicalKind::Vector);

        let map = FileSchemaMap::new(&t, "_id", &p);
        assert!(
            map.is_identity(),
            "name and type match; where a column lives is not identity"
        );
        assert!(
            !map.has_stale_type(),
            "a vector column is compared by dimension, not by its list type\'s item nullability"
        );
        assert!(map.holds(FieldId(1), PhysicalKind::FtsIndexOnly));
        assert!(!map.holds(FieldId(1), PhysicalKind::Stored));
        assert!(map.holds(FieldId(2), PhysicalKind::Vector));
        assert!(map.holds(FieldId::ID_COLUMN, PhysicalKind::Stored));
        assert_eq!(
            map.stored_names(&["title", "emb", "_id"]),
            vec!["_id"],
            "only the id column is in the Parquet body"
        );
    }
}
