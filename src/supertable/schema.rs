// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Stable column identity for a supertable.
//!
//! Every user column carries a [`FieldId`] minted once from a monotonic
//! counter and never reused. Names are labels: a column keeps its id across
//! a rename, and a dropped column's id is tombstoned rather than recycled,
//! so an id in any persisted artifact always names exactly one column. The
//! id rides as Arrow field metadata under [`FIELD_ID_META_KEY`], which
//! parquet-rs writes as the Parquet `field_id` and reads back, so every
//! stored column is self-describing.
//!
//! The supertable-injected id column is the row identity, not a user
//! column, and carries no field id.

use std::{collections::HashMap, fmt, sync::Arc};

use arrow_schema::{DataType, Field, Schema};

use crate::utils::terms::make_key;

/// Arrow field-metadata key parquet-rs maps to the Parquet `field_id`.
/// Equal to `parquet::arrow::PARQUET_FIELD_ID_META_KEY`; spelled out so
/// the schema module does not depend on the parquet crate.
pub const FIELD_ID_META_KEY: &str = "PARQUET:field_id";

/// Stable identity of one user column. Minted at `1` for a table's first
/// column and never reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub u32);

impl FieldId {
    /// The supertable-injected id column. Not a user column and never
    /// minted, so it is the one id outside `1..=last_field_id`. It is
    /// stamped on the id column's field so every stored column carries an
    /// id and the summary codecs need no name at all.
    pub const ID_COLUMN: FieldId = FieldId(0);
    /// Reserved key under which a manifest part's aggregate carries the
    /// birth-version range of its superfiles. Not a column.
    pub const BIRTH_VERSION: FieldId = FieldId(u32::MAX);

    /// The key a table-level term artifact (the term index, the term-stats
    /// sidecar) files `term` of this column under: the id in decimal, then
    /// the same separator and term bytes as a superfile's own dictionary
    /// key, so one encoding serves both tiers and a rename changes nothing
    /// at the table level.
    pub fn term_key(self, term: &str) -> Vec<u8> {
        make_key(&self.to_string(), term)
    }
}

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// One live user column: its identity, label and physical type.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDef {
    /// Stable identity; never changes once minted.
    pub id: FieldId,
    /// Current label; may change through a rename.
    pub name: String,
    /// Physical type of the column's values.
    pub data_type: DataType,
    /// Whether the column admits nulls.
    pub nullable: bool,
}

/// The table's current user columns, in declared order, with the counter
/// the next column is minted from.
#[derive(Debug, Clone, PartialEq)]
pub struct TableSchema {
    fields: Vec<FieldDef>,
    /// Ids retired by a drop. Never reused.
    tombstoned: Vec<FieldId>,
    /// The highest id ever minted, live or tombstoned.
    last_field_id: u32,
    /// Increments on every change to this document.
    schema_id: u32,
}

impl TableSchema {
    /// The schema of a table created from `user`: ids `1..=n` in declared
    /// order, `schema_id` 1.
    pub fn from_user_schema(user: &Schema) -> Self {
        let fields: Vec<FieldDef> = user
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| FieldDef {
                id: FieldId(i as u32 + 1),
                name: f.name().clone(),
                data_type: f.data_type().clone(),
                nullable: f.is_nullable(),
            })
            .collect();
        let last_field_id = fields.len() as u32;
        Self {
            fields,
            tombstoned: Vec::new(),
            last_field_id,
            schema_id: 1,
        }
    }

    /// Live columns in declared order.
    pub fn fields(&self) -> &[FieldDef] {
        &self.fields
    }

    /// Ids retired by a drop.
    pub fn tombstoned(&self) -> &[FieldId] {
        &self.tombstoned
    }

    /// The highest id ever minted, live or tombstoned.
    pub fn last_field_id(&self) -> u32 {
        self.last_field_id
    }

    /// The version of this document; increments on every change.
    pub fn schema_id(&self) -> u32 {
        self.schema_id
    }

    /// The id of the live column named `name`.
    pub fn id_of(&self, name: &str) -> Option<FieldId> {
        self.fields.iter().find(|f| f.name == name).map(|f| f.id)
    }

    /// The current name of the live column with id `id`.
    pub fn name_of(&self, id: FieldId) -> Option<&str> {
        self.fields
            .iter()
            .find(|f| f.id == id)
            .map(|f| f.name.as_str())
    }

    /// `stored` with every field stamped with its id under
    /// [`FIELD_ID_META_KEY`]: a live column gets its id, the field named
    /// `id_column` gets [`FieldId::ID_COLUMN`], and a field that is neither
    /// is left as it is. Existing metadata on a field is kept.
    pub fn stamp_field_ids(&self, stored: &Schema, id_column: &str) -> Arc<Schema> {
        let fields: Vec<Arc<Field>> = stored
            .fields()
            .iter()
            .map(|f| {
                let id = if f.name() == id_column {
                    Some(FieldId::ID_COLUMN)
                } else {
                    self.id_of(f.name())
                };
                match id {
                    Some(id) => Arc::new(with_field_id(f, id)),
                    None => Arc::clone(f),
                }
            })
            .collect();
        Arc::new(Schema::new_with_metadata(fields, stored.metadata().clone()))
    }
}

/// `batch` with [`FIELD_ID_META_KEY`] removed from every field: ids are an
/// internal identity, and a query result carries the user's columns, not
/// the engine's bookkeeping. Other field metadata is kept.
pub fn strip_field_ids(batch: arrow_array::RecordBatch) -> arrow_array::RecordBatch {
    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| f.metadata().contains_key(FIELD_ID_META_KEY))
    {
        return batch;
    }
    let fields: Vec<Arc<Field>> = schema
        .fields()
        .iter()
        .map(|f| {
            let mut m = f.metadata().clone();
            m.remove(FIELD_ID_META_KEY);
            Arc::new(f.as_ref().clone().with_metadata(m))
        })
        .collect();
    let stripped = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    arrow_array::RecordBatch::try_new(stripped, batch.columns().to_vec())
        .expect("same arrays under a schema that differs only in field metadata")
}

/// `field` with `id` stamped under [`FIELD_ID_META_KEY`], other metadata
/// kept.
pub fn with_field_id(field: &Field, id: FieldId) -> Field {
    let mut metadata: HashMap<String, String> = field.metadata().clone();
    metadata.insert(FIELD_ID_META_KEY.to_string(), id.to_string());
    field.clone().with_metadata(metadata)
}

/// The id stamped on `field`, if any. A field written without an id (a
/// file from before ids existed, or the injected id column) has none.
pub fn field_id_of(field: &Field) -> Option<FieldId> {
    field
        .metadata()
        .get(FIELD_ID_META_KEY)
        .and_then(|v| v.parse::<u32>().ok())
        .map(FieldId)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn user() -> Schema {
        Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("score", DataType::Int64, true),
        ])
    }

    #[test]
    fn a_new_table_mints_ids_in_declared_order_from_one() {
        let ts = TableSchema::from_user_schema(&user());
        assert_eq!(ts.id_of("title"), Some(FieldId(1)));
        assert_eq!(ts.id_of("score"), Some(FieldId(2)));
        assert_eq!(ts.id_of("absent"), None);
        assert_eq!(ts.name_of(FieldId(2)), Some("score"));
        assert_eq!(ts.last_field_id(), 2);
        assert_eq!(ts.schema_id(), 1);
        assert!(ts.tombstoned().is_empty());
    }

    #[test]
    fn stamping_puts_the_id_in_field_metadata_and_reads_it_back() {
        let ts = TableSchema::from_user_schema(&user());
        let stamped = ts.stamp_field_ids(&user(), "_id");
        assert_eq!(field_id_of(stamped.field(0)), Some(FieldId(1)));
        assert_eq!(field_id_of(stamped.field(1)), Some(FieldId(2)));
        // Name, type and nullability are untouched.
        assert_eq!(stamped.field(1).name(), "score");
        assert_eq!(stamped.field(1).data_type(), &DataType::Int64);
        assert!(stamped.field(1).is_nullable());
    }

    #[test]
    fn stamping_keeps_existing_metadata_and_skips_unknown_fields() {
        let ts = TableSchema::from_user_schema(&user());
        let mut keep = HashMap::new();
        keep.insert("note".to_string(), "x".to_string());
        let with_meta = Schema::new(vec![
            Field::new("_id", DataType::UInt64, false),
            Field::new("title", DataType::LargeUtf8, false).with_metadata(keep),
        ]);
        let stamped = ts.stamp_field_ids(&with_meta, "_id");
        assert_eq!(field_id_of(stamped.field(0)), Some(FieldId::ID_COLUMN));
        assert_eq!(field_id_of(stamped.field(1)), Some(FieldId(1)));
        assert_eq!(
            stamped.field(1).metadata().get("note").map(String::as_str),
            Some("x")
        );
    }

    #[test]
    fn the_metadata_key_is_the_one_parquet_rs_reads() {
        assert_eq!(FIELD_ID_META_KEY, parquet::arrow::PARQUET_FIELD_ID_META_KEY);
    }

    #[test]
    fn a_malformed_id_reads_as_none() {
        let mut m = HashMap::new();
        m.insert(FIELD_ID_META_KEY.to_string(), "seven".to_string());
        let f = Field::new("x", DataType::Int64, true).with_metadata(m);
        assert_eq!(field_id_of(&f), None);
    }
}

/// What one superfile physically holds: its stored columns with their
/// names, ids and types, exactly as its Parquet schema declares them.
/// Derived from the file once, when it is built, and carried on its
/// manifest entry so that deciding whether the file matches the table
/// never needs the file's footer.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalSchema {
    columns: Vec<PhysicalColumn>,
}

/// One stored column of a superfile.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicalColumn {
    /// The column's name as the file declares it.
    pub name: String,
    /// `None` on a column written before ids existed.
    pub id: Option<FieldId>,
    /// The column's type as the file declares it.
    pub data_type: DataType,
}

impl PhysicalSchema {
    /// The physical schema a stored Arrow schema (a superfile's
    /// `ARROW:schema`, or the builder's schema that wrote it) describes.
    pub fn of_stored_schema(schema: &Schema) -> Self {
        Self {
            columns: schema
                .fields()
                .iter()
                .map(|f| PhysicalColumn {
                    name: f.name().clone(),
                    id: field_id_of(f),
                    data_type: f.data_type().clone(),
                })
                .collect(),
        }
    }

    /// The stored columns in file order.
    pub fn columns(&self) -> &[PhysicalColumn] {
        &self.columns
    }

    /// Everything a superfile physically holds: its stored Parquet columns,
    /// then the index-only FTS columns and the vector columns, which live
    /// in the blobs rather than the Parquet body. The one place a file's
    /// physical schema is derived; the builder calls it on the file it just
    /// wrote and stamps the result on the manifest entry.
    pub fn of_reader(reader: &crate::superfile::SuperfileReader) -> Self {
        let mut columns = Self::of_stored_schema(reader.schema()).columns;
        if let Some(fts) = reader.fts() {
            for meta in fts.fts_columns_config() {
                if !meta.stored {
                    columns.push(PhysicalColumn {
                        name: meta.name.clone(),
                        id: meta.field_id,
                        data_type: DataType::LargeUtf8,
                    });
                }
            }
        }
        if let Some(vec) = reader.vec() {
            for col in vec.vector_columns_config() {
                columns.push(PhysicalColumn {
                    name: col.name.clone(),
                    id: col.field_id,
                    data_type: DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, false)),
                        col.dim as i32,
                    ),
                });
            }
        }
        Self { columns }
    }

    /// Arrow IPC bytes of the stored schema this describes, the form the
    /// manifest part carries.
    pub fn to_ipc(&self) -> Vec<u8> {
        let fields: Vec<Arc<Field>> = self
            .columns
            .iter()
            .map(|c| {
                let f = Field::new(&c.name, c.data_type.clone(), true);
                Arc::new(match c.id {
                    Some(id) => with_field_id(&f, id),
                    None => f,
                })
            })
            .collect();
        let schema = Schema::new(fields);
        let mut buf = Vec::new();
        {
            let mut w = arrow::ipc::writer::StreamWriter::try_new(&mut buf, &schema)
                .expect("IPC stream writer over an in-memory buffer");
            w.finish().expect("finish IPC stream");
        }
        buf
    }

    /// Inverse of [`Self::to_ipc`].
    pub fn from_ipc(bytes: &[u8]) -> Result<Self, String> {
        let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
            .map_err(|e| format!("physical schema IPC: {e}"))?;
        Ok(Self::of_stored_schema(&reader.schema()))
    }
}

/// Resolves the column *names* that parts and lists written before field
/// ids existed use as summary keys. Such a table could never have renamed
/// a column, so a stored name is the creation name, which is the current
/// name; the id column and the birth-version aggregate key are the two
/// names that are not user columns.
#[derive(Debug, Clone)]
pub struct LegacyNames {
    schema: Arc<TableSchema>,
    id_column: String,
}

/// The reserved aggregate key older lists carry the birth-version range
/// under.
pub const LEGACY_BIRTH_VERSION_KEY: &str = "__infino_birth_version";

impl LegacyNames {
    /// A resolver over `schema`, whose id column is named `id_column`.
    pub fn new(schema: Arc<TableSchema>, id_column: impl Into<String>) -> Self {
        Self {
            schema,
            id_column: id_column.into(),
        }
    }

    /// A resolver for an artifact that has no user columns (synthetic
    /// parts, the hidden vector-index sibling's blobs).
    pub fn none() -> Self {
        Self::new(
            Arc::new(TableSchema::from_user_schema(&Schema::empty())),
            "_id",
        )
    }

    /// The id of a column a superfile stores: the id its writer stamped when
    /// there is one, else the name resolved as a column written before ids
    /// existed.
    pub fn resolve_stored(&self, stamped: Option<FieldId>, name: &str) -> Option<FieldId> {
        stamped.or_else(|| self.resolve(name))
    }

    /// The id a stored column name resolves to, or `None` for a name the
    /// table does not have.
    pub fn resolve(&self, name: &str) -> Option<FieldId> {
        if name == self.id_column {
            Some(FieldId::ID_COLUMN)
        } else if name == LEGACY_BIRTH_VERSION_KEY {
            Some(FieldId::BIRTH_VERSION)
        } else {
            self.schema.id_of(name)
        }
    }
}

/// The on-disk spelling of a summary key: the id in decimal.
pub fn wire_key(id: FieldId) -> String {
    id.0.to_string()
}

/// Reads a summary key back. A key written by this engine is a decimal id;
/// a key from an older artifact is a column name, resolved through
/// `legacy` when one is given. `None` means the key names nothing the
/// table knows and the entry is dropped (it was data about a column that no
/// longer exists in a form the table can use).
pub fn parse_wire_key(key: &str, legacy: Option<&LegacyNames>) -> Option<FieldId> {
    match legacy {
        None => key.parse::<u32>().ok().map(FieldId),
        Some(names) => names.resolve(key),
    }
}

#[cfg(test)]
mod key_tests {
    use super::*;

    #[test]
    fn stamping_marks_the_id_column_with_the_reserved_id() {
        let ts = TableSchema::from_user_schema(&Schema::new(vec![Field::new(
            "score",
            DataType::Int64,
            true,
        )]));
        let stored = ts.stamp_field_ids(
            &Schema::new(vec![
                Field::new("_id", DataType::Decimal128(38, 0), false),
                Field::new("score", DataType::Int64, true),
            ]),
            "_id",
        );
        assert_eq!(field_id_of(stored.field(0)), Some(FieldId::ID_COLUMN));
        assert_eq!(field_id_of(stored.field(1)), Some(FieldId(1)));
    }

    #[test]
    fn wire_keys_are_decimal_ids_and_legacy_names_resolve() {
        let ts = Arc::new(TableSchema::from_user_schema(&Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("score", DataType::Int64, true),
        ])));
        assert_eq!(wire_key(FieldId(2)), "2");
        assert_eq!(parse_wire_key("2", None), Some(FieldId(2)));
        assert_eq!(parse_wire_key("score", None), None);
        let legacy = LegacyNames::new(Arc::clone(&ts), "doc_id");
        assert_eq!(parse_wire_key("score", Some(&legacy)), Some(FieldId(2)));
        assert_eq!(
            parse_wire_key("doc_id", Some(&legacy)),
            Some(FieldId::ID_COLUMN)
        );
        assert_eq!(
            parse_wire_key(LEGACY_BIRTH_VERSION_KEY, Some(&legacy)),
            Some(FieldId::BIRTH_VERSION)
        );
        assert_eq!(parse_wire_key("gone", Some(&legacy)), None);
    }

    #[test]
    fn a_physical_schema_round_trips_through_ipc_with_ids() {
        let ts = TableSchema::from_user_schema(&Schema::new(vec![Field::new(
            "score",
            DataType::Int64,
            true,
        )]));
        let stored = ts.stamp_field_ids(
            &Schema::new(vec![
                Field::new("_id", DataType::Decimal128(38, 0), false),
                Field::new("score", DataType::Int64, true),
            ]),
            "_id",
        );
        let ps = PhysicalSchema::of_stored_schema(&stored);
        assert_eq!(ps.columns()[0].id, Some(FieldId::ID_COLUMN));
        assert_eq!(ps.columns()[1].id, Some(FieldId(1)));
        let back = PhysicalSchema::from_ipc(&ps.to_ipc()).expect("decode");
        assert_eq!(back, ps);
        assert!(PhysicalSchema::from_ipc(b"garbage").is_err());
    }
}
