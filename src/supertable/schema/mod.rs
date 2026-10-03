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
//!
//! [`TableSchema`] is also the schema *document*: the index a column
//! carries, the type it is converting from, the caps, and the counters
//! travel with the fields, and [`TableSchema::to_json`] /
//! [`TableSchema::from_json`] are its one spelling, written into the
//! manifest list and returned to a caller that asks for the schema.

pub mod cast;
pub mod change;
pub mod error;
#[cfg(test)]
mod generations;
pub mod map;
pub mod resolve;
pub mod types;

use std::{collections::HashMap, fmt, sync::Arc};

use arrow_schema::{DataType, Field, Schema};
use serde_json::{Map, Value};
use types::{data_type_from_keys, type_keys};

use crate::{
    superfile::{
        builder::FtsConfig,
        fts::{
            analysis::{Base, Stemmer, Stopwords, chain_tokenizer},
            bm25::Bm25Params,
            tokenize::Tokenizer,
        },
        vector::{builder::VectorConfig, distance::Metric, rerank_codec::RerankCodec},
    },
    utils::terms::make_key,
};

/// How many fields a table may hold before an append or schema write
/// that would add one is refused.
pub const DEFAULT_MAX_FIELDS: u32 = 10_000;

/// Precision of the id column's `Decimal128`: the most a `Decimal128` can
/// carry, so every 128-bit id fits without truncation, Parquet annotates
/// the column as `DECIMAL(38, 0)`, and sort order matches the `i128`.
pub(crate) const DECIMAL128_PRECISION: u8 = 38;

/// Scale of the id column's `Decimal128`: ids are integers, so with
/// precision 38 and scale 0 the column behaves as a signed 128-bit
/// integer for Arrow's comparison kernels and Parquet's statistics alike.
pub(crate) const DECIMAL128_SCALE: i8 = 0;

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

/// The index a column carries. Part of the column's definition: the
/// analyzer chain and the vector metric are fixed when the index is
/// created, so they live with the field rather than in a parallel list.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnIndex {
    /// A full-text index over a string column.
    Fts {
        /// The base tokenizer's name.
        analyzer: String,
        /// The stopword filter applied after tokenizing.
        stopwords: Stopwords,
        /// The stemmer applied after tokenizing.
        stemmer: Stemmer,
        /// Whether token positions are recorded (phrase queries).
        positions: bool,
        /// Whether the text is also kept in the Parquet body.
        stored: bool,
        /// The BM25 parameters the index is scored with.
        bm25: Bm25Params,
    },
    /// A vector index over a `vector` (fixed-size `f32` list) column; the
    /// dimension is the column type's.
    Vector {
        /// The distance the index is built and searched with.
        metric: Metric,
        /// Seed of the random rotation applied before quantization.
        rot_seed: u64,
        /// How candidates are re-scored after the coarse search.
        rerank_codec: RerankCodec,
    },
}

/// One live user column: its identity, label, physical type and index.
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
    /// The index the column carries, if any.
    pub index: Option<ColumnIndex>,
    /// The type the column is being converted from: set while files
    /// written in that type remain, cleared when the last is rewritten.
    pub converting_from: Option<DataType>,
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
    /// The most fields the table may hold.
    max_fields: u32,
}

impl TableSchema {
    /// The schema of a table created from `user`: ids `1..=n` in declared
    /// order, `schema_id` 1, no indexes.
    pub fn from_user_schema(user: &Schema) -> Self {
        Self::from_options(user, &[], &[])
    }

    /// The schema of a table created from `user` with the indexes `fts` and
    /// `vectors` configure: ids `1..=n` in declared order, `schema_id` 1.
    /// Every config names a field of `user`; the options constructor checks
    /// that before this is reached.
    pub fn from_options(user: &Schema, fts: &[FtsConfig], vectors: &[VectorConfig]) -> Self {
        let fields: Vec<FieldDef> = user
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| FieldDef {
                id: FieldId(i as u32 + 1),
                name: f.name().clone(),
                data_type: f.data_type().clone(),
                nullable: f.is_nullable(),
                index: column_index(f.name(), fts, vectors),
                converting_from: None,
            })
            .collect();
        let last_field_id = fields.len() as u32;
        Self {
            fields,
            tombstoned: Vec::new(),
            last_field_id,
            schema_id: 1,
            max_fields: DEFAULT_MAX_FIELDS,
        }
    }

    /// The user schema: every live column in declared order, unstamped.
    pub fn user_schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(
            self.fields
                .iter()
                .map(|f| Arc::new(Field::new(&f.name, f.data_type.clone(), f.nullable)))
                .collect::<Vec<_>>(),
        ))
    }

    /// The id column's field, which every stored schema starts with.
    pub fn id_field(id_column: &str) -> Arc<Field> {
        Arc::new(Field::new(
            id_column,
            DataType::Decimal128(DECIMAL128_PRECISION, DECIMAL128_SCALE),
            false,
        ))
    }

    /// The id column followed by every live column, stamped with ids: the
    /// shape of a batch after the id column is attached.
    pub fn effective_schema(&self, id_column: &str) -> Arc<Schema> {
        self.stored_fields(id_column, |_| true)
    }

    /// The id column followed by every live column the Parquet body may
    /// hold, stamped with ids. A vector-indexed column lives in the vector
    /// blob, never in Parquet.
    pub fn scalar_schema(&self, id_column: &str) -> Arc<Schema> {
        self.stored_fields(id_column, |f| {
            !matches!(f.index, Some(ColumnIndex::Vector { .. }))
        })
    }

    /// [`Self::scalar_schema`] without the index-only full-text columns:
    /// exactly the columns a superfile's Parquet body holds.
    pub fn stored_schema(&self, id_column: &str) -> Arc<Schema> {
        self.stored_fields(id_column, |f| match &f.index {
            Some(ColumnIndex::Vector { .. }) => false,
            Some(ColumnIndex::Fts { stored, .. }) => *stored,
            None => true,
        })
    }

    fn stored_fields(&self, id_column: &str, keep: impl Fn(&FieldDef) -> bool) -> Arc<Schema> {
        let mut fields = vec![Self::id_field(id_column)];
        fields.extend(
            self.fields
                .iter()
                .filter(|f| keep(f))
                .map(|f| Arc::new(Field::new(&f.name, f.data_type.clone(), f.nullable))),
        );
        self.stamp_field_ids(&Schema::new(fields), id_column)
    }

    /// The full-text index configs the schema's fields carry, in declared
    /// order.
    pub fn fts_configs(&self) -> Vec<FtsConfig> {
        self.fields
            .iter()
            .filter_map(|f| match &f.index {
                Some(ColumnIndex::Fts {
                    analyzer,
                    stopwords,
                    stemmer,
                    positions,
                    stored,
                    bm25,
                }) => Some(FtsConfig {
                    column: f.name.clone(),
                    analyzer: analyzer.clone(),
                    stopwords: *stopwords,
                    stemmer: *stemmer,
                    positions: *positions,
                    stored: *stored,
                    bm25: *bm25,
                    carried_analysis_revision: None,
                }),
                _ => None,
            })
            .collect()
    }

    /// The vector index configs the schema's fields carry, in declared
    /// order. A config built here carries no provided centroids: those are
    /// a creation-time seed, not part of the schema.
    pub fn vector_configs(&self) -> Vec<VectorConfig> {
        self.fields
            .iter()
            .filter_map(|f| match (&f.index, &f.data_type) {
                (
                    Some(ColumnIndex::Vector {
                        metric,
                        rot_seed,
                        rerank_codec,
                    }),
                    DataType::FixedSizeList(_, dim),
                ) => Some(
                    VectorConfig::new(f.name.clone(), *dim as usize, *rot_seed, *metric)
                        .with_rerank_codec(*rerank_codec),
                ),
                _ => None,
            })
            .collect()
    }

    /// The analyzer chain of the full-text column named `column`, or `None`
    /// when the column carries no full-text index — the signal a search
    /// path uses to reject up front instead of failing deep in the scan.
    /// The whole chain, not the base: every caller tokenizes query-side
    /// text and must produce the forms the column was indexed under.
    pub fn fts_tokenizer_for(&self, column: &str) -> Option<Arc<dyn Tokenizer>> {
        let field = self.fields.iter().find(|f| f.name == column)?;
        match &field.index {
            Some(ColumnIndex::Fts {
                analyzer,
                stopwords,
                stemmer,
                ..
            }) => Some(chain_tokenizer(
                Base::from_name(analyzer)?,
                *stopwords,
                *stemmer,
            )),
            _ => None,
        }
    }

    /// The most fields the table may hold.
    pub fn max_fields(&self) -> u32 {
        self.max_fields
    }

    /// The document as JSON: the fields with their ids, types and indexes,
    /// the tombstoned ids, the caps and the counters.
    pub fn to_json(&self) -> Value {
        let fields: Vec<Value> = self
            .fields
            .iter()
            .map(|f| {
                let mut keys = type_keys(&f.data_type);
                keys.insert("id".into(), Value::from(f.id.0));
                keys.insert("name".into(), Value::from(f.name.as_str()));
                keys.insert("nullable".into(), Value::from(f.nullable));
                if let Some(index) = &f.index {
                    keys.insert("index".into(), index_to_json(index));
                }
                if let Some(from) = &f.converting_from {
                    keys.insert("converting_from".into(), Value::Object(type_keys(from)));
                }
                Value::Object(keys)
            })
            .collect();
        let mut doc = Map::new();
        doc.insert("schema_id".into(), Value::from(self.schema_id));
        doc.insert("last_field_id".into(), Value::from(self.last_field_id));
        doc.insert("max_fields".into(), Value::from(self.max_fields));
        doc.insert("fields".into(), Value::Array(fields));
        doc.insert(
            "tombstoned".into(),
            Value::Array(self.tombstoned.iter().map(|t| Value::from(t.0)).collect()),
        );
        Value::Object(doc)
    }

    /// The document `json` spells. Structural checks only: the writer was
    /// this engine, so values are trusted as they were validated when
    /// written.
    pub fn from_json(json: &Value) -> Result<Self, String> {
        let doc = json
            .as_object()
            .ok_or_else(|| "schema document is not an object".to_string())?;
        let u32_of = |key: &str| -> Result<u32, String> {
            doc.get(key)
                .and_then(Value::as_u64)
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| format!("schema document has no {key}"))
        };
        let schema_id = u32_of("schema_id")?;
        let last_field_id = u32_of("last_field_id")?;
        let max_fields = match doc.get("max_fields") {
            None => DEFAULT_MAX_FIELDS,
            Some(_) => u32_of("max_fields")?,
        };
        let fields = doc
            .get("fields")
            .and_then(Value::as_array)
            .ok_or_else(|| "schema document has no fields".to_string())?
            .iter()
            .map(field_from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let tombstoned = doc
            .get("tombstoned")
            .and_then(Value::as_array)
            .map(|ids| {
                ids.iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|id| u32::try_from(id).ok())
                            .map(FieldId)
                            .ok_or_else(|| "tombstoned id is not a u32".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            fields,
            tombstoned,
            last_field_id,
            schema_id,
            max_fields,
        })
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
    /// `FIELD_ID_META_KEY`: a live column gets its id, the field named
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

/// The index `fts` or `vectors` configure for the column named `name`.
fn column_index(name: &str, fts: &[FtsConfig], vectors: &[VectorConfig]) -> Option<ColumnIndex> {
    if let Some(fc) = fts.iter().find(|fc| fc.column == name) {
        return Some(ColumnIndex::Fts {
            analyzer: fc.analyzer.clone(),
            stopwords: fc.stopwords,
            stemmer: fc.stemmer,
            positions: fc.positions,
            stored: fc.stored,
            bm25: fc.bm25,
        });
    }
    vectors
        .iter()
        .find(|vc| vc.column == name)
        .map(|vc| ColumnIndex::Vector {
            metric: vc.metric,
            rot_seed: vc.rot_seed,
            rerank_codec: vc.rerank_codec,
        })
}

pub(crate) fn index_to_json(index: &ColumnIndex) -> Value {
    let mut out = Map::new();
    match index {
        ColumnIndex::Fts {
            analyzer,
            stopwords,
            stemmer,
            positions,
            stored,
            bm25,
        } => {
            out.insert("kind".into(), Value::from("fts"));
            out.insert("analyzer".into(), Value::from(analyzer.as_str()));
            if let Some(name) = stopwords.as_str() {
                out.insert("stopwords".into(), Value::from(name));
            }
            if let Some(name) = stemmer.as_str() {
                out.insert("stemmer".into(), Value::from(name));
            }
            out.insert("positions".into(), Value::from(*positions));
            out.insert("stored".into(), Value::from(*stored));
            out.insert("k1".into(), Value::from(bm25.k1));
            out.insert("b".into(), Value::from(bm25.b));
        }
        ColumnIndex::Vector {
            metric,
            rot_seed,
            rerank_codec,
        } => {
            out.insert("kind".into(), Value::from("vector"));
            out.insert("metric".into(), Value::from(metric.name()));
            out.insert("rot_seed".into(), Value::from(*rot_seed));
            out.insert(
                "rerank_codec".into(),
                serde_json::to_value(rerank_codec).expect("codec name"),
            );
        }
    }
    Value::Object(out)
}

pub(crate) fn index_from_json(json: &Value) -> Result<ColumnIndex, String> {
    let obj = json
        .as_object()
        .ok_or_else(|| "index is not an object".to_string())?;
    let str_of = |key: &str| -> Result<&str, String> {
        obj.get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| format!("index has no {key}"))
    };
    let bool_of = |key: &str| -> Result<bool, String> {
        obj.get(key)
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("index has no {key}"))
    };
    let f32_of = |key: &str| -> Result<f32, String> {
        obj.get(key)
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .ok_or_else(|| format!("index has no {key}"))
    };
    match str_of("kind")? {
        "fts" => Ok(ColumnIndex::Fts {
            analyzer: str_of("analyzer")?.to_owned(),
            stopwords: named(obj, "stopwords", Stopwords::from_name)?.unwrap_or(Stopwords::None),
            stemmer: named(obj, "stemmer", Stemmer::from_name)?.unwrap_or(Stemmer::None),
            positions: bool_of("positions")?,
            stored: bool_of("stored")?,
            bm25: Bm25Params::new(f32_of("k1")?, f32_of("b")?),
        }),
        "vector" => Ok(ColumnIndex::Vector {
            metric: {
                let name = str_of("metric")?;
                Metric::from_name(name)
                    .ok_or_else(|| format!("index has an unknown metric '{name}'"))?
            },
            rot_seed: obj
                .get("rot_seed")
                .and_then(Value::as_u64)
                .ok_or_else(|| "index has no rot_seed".to_string())?,
            rerank_codec: serde_json::from_value(
                obj.get("rerank_codec")
                    .cloned()
                    .ok_or_else(|| "index has no rerank_codec".to_string())?,
            )
            .map_err(|e| format!("index rerank_codec: {e}"))?,
        }),
        other => Err(format!("unknown index kind '{other}'")),
    }
}

/// An optional named part of an index (`stopwords`, `stemmer`): absent is
/// `None`, present must parse.
fn named<T>(
    obj: &Map<String, Value>,
    key: &str,
    parse: fn(&str) -> Option<T>,
) -> Result<Option<T>, String> {
    match obj.get(key).and_then(Value::as_str) {
        None => Ok(None),
        Some(name) => parse(name)
            .map(Some)
            .ok_or_else(|| format!("index has an unknown {key} '{name}'")),
    }
}

fn field_from_json(json: &Value) -> Result<FieldDef, String> {
    let keys = json
        .as_object()
        .ok_or_else(|| "field is not an object".to_string())?;
    let id = keys
        .get("id")
        .and_then(Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
        .map(FieldId)
        .ok_or_else(|| "field has no id".to_string())?;
    let name = keys
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "field has no name".to_string())?
        .to_owned();
    let nullable = keys
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let index = keys.get("index").map(index_from_json).transpose()?;
    let converting_from = keys
        .get("converting_from")
        .map(|from| {
            from.as_object()
                .ok_or_else(|| "converting_from is not an object".to_string())
                .and_then(data_type_from_keys)
        })
        .transpose()?;
    Ok(FieldDef {
        id,
        name,
        data_type: data_type_from_keys(keys)?,
        nullable,
        index,
        converting_from,
    })
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

    fn indexed() -> (Schema, Vec<FtsConfig>, Vec<VectorConfig>) {
        let user = Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 4),
                true,
            ),
            Field::new("score", DataType::Int64, true),
        ]);
        let mut fts = FtsConfig::new("title")
            .analyzer("ascii_lower")
            .stopwords(Stopwords::English)
            .stemmer(Stemmer::English);
        fts.positions = true;
        fts.stored = false;
        fts.bm25 = Bm25Params::new(0.9, 0.4);
        let vector = VectorConfig::new("emb".to_string(), 4, 77, Metric::L2Sq)
            .with_rerank_codec(RerankCodec::Fp32);
        (user, vec![fts], vec![vector])
    }

    #[test]
    fn indexes_ride_on_their_fields_and_derive_the_configs_back() {
        let (user, fts, vectors) = indexed();
        let ts = TableSchema::from_options(&user, &fts, &vectors);
        assert_eq!(
            ts.fields()[0]
                .index
                .as_ref()
                .map(|i| matches!(i, ColumnIndex::Fts { .. })),
            Some(true)
        );
        assert!(ts.fields()[2].index.is_none());
        assert_eq!(*ts.user_schema(), user);

        let derived_fts = ts.fts_configs();
        assert_eq!(derived_fts.len(), 1);
        let d = &derived_fts[0];
        assert_eq!(
            (
                d.column.as_str(),
                d.analyzer.as_str(),
                d.stopwords,
                d.stemmer,
                d.positions,
                d.stored,
                d.bm25
            ),
            (
                "title",
                "ascii_lower",
                Stopwords::English,
                Stemmer::English,
                true,
                false,
                Bm25Params::new(0.9, 0.4)
            )
        );
        let derived_vec = ts.vector_configs();
        assert_eq!(derived_vec.len(), 1);
        let v = &derived_vec[0];
        assert_eq!(
            (
                v.column.as_str(),
                v.dim,
                v.rot_seed,
                v.metric,
                v.rerank_codec
            ),
            ("emb", 4, 77, Metric::L2Sq, RerankCodec::Fp32)
        );
        assert!(v.provided_centroids.is_none());
    }

    #[test]
    fn the_document_round_trips_through_json() {
        let (user, fts, vectors) = indexed();
        let mut ts = TableSchema::from_options(&user, &fts, &vectors);
        ts.fields[2].converting_from = Some(DataType::Int32);
        ts.tombstoned.push(FieldId(9));
        ts.last_field_id = 9;
        ts.schema_id = 4;
        ts.max_fields = 50;
        let json = ts.to_json();
        assert_eq!(json["schema_id"], 4);
        assert_eq!(json["fields"][0]["type"], "large_utf8");
        assert_eq!(json["fields"][0]["index"]["kind"], "fts");
        assert_eq!(json["fields"][0]["index"]["stopwords"], "english");
        assert_eq!(json["fields"][1]["type"], "vector");
        assert_eq!(json["fields"][1]["dim"], 4);
        assert_eq!(json["fields"][1]["index"]["metric"], "l2sq");
        assert_eq!(json["fields"][1]["index"]["rerank_codec"], "fp32");
        assert_eq!(json["fields"][2]["converting_from"]["type"], "i32");
        assert_eq!(TableSchema::from_json(&json).expect("decode"), ts);

        // A document without indexes spells no stopwords or stemmer and
        // reads back with the defaults.
        let plain = TableSchema::from_user_schema(&user);
        let json = plain.to_json();
        assert!(json["fields"][0].get("index").is_none());
        assert_eq!(TableSchema::from_json(&json).expect("decode"), plain);
    }

    #[test]
    fn a_document_missing_its_counters_or_with_an_unknown_index_is_refused() {
        let (user, fts, vectors) = indexed();
        let ts = TableSchema::from_options(&user, &fts, &vectors);
        let mut json = ts.to_json();
        json["fields"][0]["index"]["kind"] = Value::from("bloom");
        assert!(TableSchema::from_json(&json).is_err());
        let mut json = ts.to_json();
        json["fields"][0]["index"]["stemmer"] = Value::from("klingon");
        assert!(TableSchema::from_json(&json).is_err());
        let mut json = ts.to_json();
        json.as_object_mut()
            .expect("object")
            .remove("last_field_id");
        assert!(TableSchema::from_json(&json).is_err());
        let mut json = ts.to_json();
        json.as_object_mut().expect("object").remove("max_fields");
        assert_eq!(
            TableSchema::from_json(&json).expect("decode").max_fields(),
            DEFAULT_MAX_FIELDS,
            "the cap defaults when a document predates it"
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
    /// Where the file holds the column.
    pub kind: PhysicalKind,
}

/// Where a superfile physically holds a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalKind {
    /// In the Parquet body (and, for an indexed text column, in the FTS
    /// blob as well).
    Stored,
    /// Only in the FTS blob: an index-only text column.
    FtsIndexOnly,
    /// Only in the vector blob.
    Vector,
}

/// Field-metadata key the physical schema's IPC form carries a non-stored
/// column's kind under; a stored column carries none.
const PHYSICAL_KIND_META_KEY: &str = "infino:physical";

impl PhysicalSchema {
    /// A physical schema of exactly `columns`, in file order.
    pub fn new(columns: Vec<PhysicalColumn>) -> Self {
        Self { columns }
    }

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
                    kind: PhysicalKind::Stored,
                })
                .collect(),
        }
    }

    /// The column with field id `id`, if the file holds one.
    pub fn column_by_id(&self, id: FieldId) -> Option<&PhysicalColumn> {
        self.columns.iter().find(|c| c.id == Some(id))
    }

    /// The column named `name`, if the file holds one.
    pub fn column_by_name(&self, name: &str) -> Option<&PhysicalColumn> {
        self.columns.iter().find(|c| c.name == name)
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
    pub fn of_reader(reader: &crate::superfile::SuperfileReader, legacy: &LegacyNames) -> Self {
        let mut columns = Self::of_stored_schema(reader.schema()).columns;
        if let Some(fts) = reader.fts() {
            for meta in fts.fts_columns_config() {
                if !meta.stored {
                    columns.push(PhysicalColumn {
                        name: meta.name.clone(),
                        id: meta.field_id,
                        data_type: DataType::LargeUtf8,
                        kind: PhysicalKind::FtsIndexOnly,
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
                    kind: PhysicalKind::Vector,
                });
            }
        }
        Self { columns }.resolved(legacy)
    }

    /// This schema with every column that carries no id given the id its
    /// name resolves to through `legacy`: a file written before ids names
    /// its columns by their creation names, and once resolved here the
    /// manifest entry carries ids for it like any other file's.
    pub fn resolved(mut self, legacy: &LegacyNames) -> Self {
        for column in &mut self.columns {
            if column.id.is_none() {
                column.id = legacy.resolve(&column.name);
            }
        }
        self
    }

    /// Arrow IPC bytes of the stored schema this describes, the form the
    /// manifest part carries.
    pub fn to_ipc(&self) -> Vec<u8> {
        let fields: Vec<Arc<Field>> = self
            .columns
            .iter()
            .map(|c| {
                let mut f = Field::new(&c.name, c.data_type.clone(), true);
                if let Some(id) = c.id {
                    f = with_field_id(&f, id);
                }
                let kind = match c.kind {
                    PhysicalKind::Stored => None,
                    PhysicalKind::FtsIndexOnly => Some("fts"),
                    PhysicalKind::Vector => Some("vector"),
                };
                if let Some(kind) = kind {
                    let mut metadata = f.metadata().clone();
                    metadata.insert(PHYSICAL_KIND_META_KEY.to_string(), kind.to_string());
                    f = f.with_metadata(metadata);
                }
                Arc::new(f)
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
        let mut physical = Self::of_stored_schema(&reader.schema());
        for (column, field) in physical.columns.iter_mut().zip(reader.schema().fields()) {
            column.kind = match field
                .metadata()
                .get(PHYSICAL_KIND_META_KEY)
                .map(String::as_str)
            {
                None => PhysicalKind::Stored,
                Some("fts") => PhysicalKind::FtsIndexOnly,
                Some("vector") => PhysicalKind::Vector,
                Some(other) => {
                    return Err(format!("physical schema: unknown column kind '{other}'"));
                }
            };
        }
        Ok(physical)
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
