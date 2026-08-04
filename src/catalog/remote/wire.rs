// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Wire encoding for the remote transport.
//!
//! Requests are JSON; read responses are Arrow IPC streams that decode straight
//! into the engine's native `Vec<RecordBatch>`. This module owns the
//! translations: an Arrow schema and an [`IndexSpec`] to their JSON request
//! shapes, `RecordBatch`es to/from the Arrow IPC stream, and an HTTP status to
//! an [`InfinoError`]. The string spellings mirror what the hosted service
//! accepts.

use std::{io::Cursor, sync::Arc};

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use serde_json::{Value, json};

use crate::{IndexSpec, InfinoError, Metric};

/// Content type for an Arrow IPC streaming body — the encoding for `append`
/// bodies and read responses.
pub(crate) const ARROW_STREAM_CONTENT_TYPE: &str = "application/vnd.apache.arrow.stream";

/// Canonical wire spelling for a scalar Arrow type, matching what the hosted
/// service accepts. Non-scalar types are rejected so a schema that can't cross
/// the wire fails loudly at the client instead of diverging from local.
fn scalar_type_str(data_type: &DataType) -> Result<&'static str, InfinoError> {
    Ok(match data_type {
        DataType::Utf8 => "utf8",
        DataType::LargeUtf8 => "large_utf8",
        DataType::Boolean => "bool",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        other => {
            return Err(InfinoError::Schema(format!(
                "column type not supported over the remote transport: {other:?}"
            )));
        }
    })
}

/// One Arrow field as its `{name, type, nullable}` JSON descriptor. `nullable`
/// is always written explicitly (the server defaults an omitted value to
/// `true`, which would silently flip a non-nullable column). A
/// `FixedSizeList<Float32, dim>` maps to `{type: "vector", dim}`; a `List<T>`
/// to `{type: "list", item}`.
fn field_to_json(field: &Field) -> Result<Value, InfinoError> {
    let name = field.name();
    let nullable = field.is_nullable();
    match field.data_type() {
        DataType::FixedSizeList(item, dim) if *item.data_type() == DataType::Float32 => {
            Ok(json!({ "name": name, "type": "vector", "dim": dim, "nullable": nullable }))
        }
        DataType::List(item) => Ok(json!({
            "name": name,
            "type": "list",
            "item": scalar_type_str(item.data_type())?,
            "nullable": nullable,
        })),
        other => Ok(json!({ "name": name, "type": scalar_type_str(other)?, "nullable": nullable })),
    }
}

/// A schema as the JSON array of field descriptors the create-table request
/// carries.
pub(crate) fn schema_to_json(schema: &Schema) -> Result<Vec<Value>, InfinoError> {
    schema.fields().iter().map(|f| field_to_json(f)).collect()
}

/// The wire spelling for a vector distance metric.
pub(crate) fn metric_str(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::L2Sq => "l2sq",
        Metric::NegDot => "negdot",
    }
}

/// An [`IndexSpec`] as the `indexes` object of a create-table request:
/// `{fts: [col, …], vector: [{column, dim, metric}, …]}`. Absent index kinds
/// are omitted (the server treats a missing key as "none").
pub(crate) fn index_spec_to_json(spec: &IndexSpec) -> Value {
    let mut indexes = serde_json::Map::new();
    let fts = spec.fts_columns();
    if !fts.is_empty() {
        indexes.insert("fts".to_string(), json!(fts));
    }
    let vectors: Vec<Value> = spec
        .vector_indexes()
        .map(|(column, dim, metric)| {
            json!({
                "column": column,
                "dim": dim,
                "metric": metric_str(metric),
            })
        })
        .collect();
    if !vectors.is_empty() {
        indexes.insert("vector".to_string(), Value::Array(vectors));
    }
    Value::Object(indexes)
}

/// Encode record batches as one Arrow IPC stream. An empty slice yields an
/// empty body (mirrors the server's empty-result encoding).
pub(crate) fn batches_to_ipc(batches: &[RecordBatch]) -> Result<Vec<u8>, InfinoError> {
    let Some(first) = batches.first() else {
        return Ok(Vec::new());
    };
    let schema = first.schema();
    let mut out = Vec::new();
    let mut writer = StreamWriter::try_new(&mut out, &schema)
        .map_err(|e| InfinoError::Backend(format!("arrow ipc writer: {e}")))?;
    for batch in batches {
        writer
            .write(batch)
            .map_err(|e| InfinoError::Backend(format!("arrow ipc write: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| InfinoError::Backend(format!("arrow ipc finish: {e}")))?;
    Ok(out)
}

/// Decode an Arrow IPC stream into record batches. An empty body is an empty
/// result (mirrors the server's empty-result encoding).
pub(crate) fn ipc_to_batches(bytes: &[u8]) -> Result<Vec<RecordBatch>, InfinoError> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| InfinoError::Backend(format!("arrow ipc reader: {e}")))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| InfinoError::Backend(format!("arrow ipc read: {e}")))
}

/// The Arrow scalar type for a wire type spelling — the inverse of
/// [`scalar_type_str`], accepting the same aliases the hosted service does.
fn scalar_type_from_str(name: &str) -> Result<DataType, InfinoError> {
    Ok(match name {
        "utf8" | "string" => DataType::Utf8,
        "large_utf8" | "large_string" => DataType::LargeUtf8,
        "bool" | "boolean" => DataType::Boolean,
        "i32" | "int32" => DataType::Int32,
        "i64" | "int64" => DataType::Int64,
        "u32" | "uint32" => DataType::UInt32,
        "u64" | "uint64" => DataType::UInt64,
        "f32" | "float32" => DataType::Float32,
        "f64" | "float64" | "double" => DataType::Float64,
        other => {
            return Err(InfinoError::Schema(format!(
                "unknown column type in schema response: {other}"
            )));
        }
    })
}

fn field_from_json(value: &Value) -> Result<Field, InfinoError> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| InfinoError::Schema("schema field missing `name`".to_string()))?;
    let type_name = value
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| InfinoError::Schema("schema field missing `type`".to_string()))?;
    let nullable = value
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let data_type = match type_name {
        "vector" => {
            let dim = value
                .get("dim")
                .and_then(Value::as_u64)
                .ok_or_else(|| InfinoError::Schema("vector column missing `dim`".to_string()))?;
            let item = Arc::new(Field::new("item", DataType::Float32, true));
            DataType::FixedSizeList(item, dim as i32)
        }
        "list" => {
            let item_type = value
                .get("item")
                .and_then(Value::as_str)
                .ok_or_else(|| InfinoError::Schema("list column missing `item`".to_string()))?;
            let item = Arc::new(Field::new("item", scalar_type_from_str(item_type)?, true));
            DataType::List(item)
        }
        other => scalar_type_from_str(other)?,
    };
    Ok(Field::new(name, data_type, nullable))
}

/// Decode a schema response (`[{name, type, nullable}, …]`) into an Arrow
/// schema — the inverse of [`schema_to_json`].
pub(crate) fn json_to_schema(fields: &[Value]) -> Result<Schema, InfinoError> {
    let fields = fields
        .iter()
        .map(field_from_json)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Schema::new(fields))
}

/// Map an HTTP error status to an [`InfinoError`]. Best-effort: only a few
/// statuses map to a typed variant; the rest become `Backend`. `op` labels the
/// operation for context.
pub(crate) fn status_to_error(op: &str, code: u16, body: &str) -> InfinoError {
    match code {
        404 => InfinoError::NotFound(format!("{op}: {body}")),
        409 => InfinoError::AlreadyExists(format!("{op}: {body}")),
        412 => InfinoError::Conflict(format!("{op}: {body}")),
        401 | 403 => {
            InfinoError::Backend(format!("{op}: unauthorized (check the API key): {body}"))
        }
        _ => InfinoError::Backend(format!("{op}: server returned {code}: {body}")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, FieldRef, Fields, Schema};

    use super::*;

    #[test]
    fn schema_json_preserves_nullable_and_types() {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("body", DataType::LargeUtf8, true),
        ]);
        let json = schema_to_json(&schema).expect("schema to json");
        assert_eq!(
            json[0],
            json!({"name": "id", "type": "i32", "nullable": false})
        );
        assert_eq!(
            json[1],
            json!({"name": "body", "type": "large_utf8", "nullable": true})
        );
    }

    #[test]
    fn schema_json_maps_vector_column() {
        let item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Schema::new(vec![Field::new(
            "embedding",
            DataType::FixedSizeList(item, 384),
            false,
        )]);
        let json = schema_to_json(&schema).expect("schema to json");
        assert_eq!(
            json[0],
            json!({"name": "embedding", "type": "vector", "dim": 384, "nullable": false})
        );
    }

    #[test]
    fn schema_json_rejects_unsupported_type() {
        let schema = Schema::new(vec![Field::new("ts", DataType::Date32, true)]);
        assert!(matches!(
            schema_to_json(&schema),
            Err(InfinoError::Schema(_))
        ));
    }

    #[test]
    fn index_spec_json_shape() {
        let spec = IndexSpec::new()
            .fts("body")
            .vector("embedding", 384, Metric::Cosine);
        let json = index_spec_to_json(&spec);
        assert_eq!(json["fts"], json!(["body"]));
        assert_eq!(
            json["vector"][0],
            json!({"column": "embedding", "dim": 384, "metric": "cosine"})
        );
    }

    #[test]
    fn empty_index_spec_is_empty_object() {
        assert_eq!(index_spec_to_json(&IndexSpec::new()), json!({}));
    }

    #[test]
    fn metric_spellings() {
        assert_eq!(metric_str(Metric::Cosine), "cosine");
        assert_eq!(metric_str(Metric::L2Sq), "l2sq");
        assert_eq!(metric_str(Metric::NegDot), "negdot");
    }

    fn sample_batch() -> RecordBatch {
        let fields: Fields = vec![FieldRef::from(Field::new("id", DataType::Int32, false))].into();
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(vec![1, 2, 3]))])
            .expect("build batch")
    }

    #[test]
    fn ipc_round_trips() {
        let batch = sample_batch();
        let bytes = batches_to_ipc(std::slice::from_ref(&batch)).expect("encode");
        let back = ipc_to_batches(&bytes).expect("decode");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0], batch);
    }

    #[test]
    fn empty_batches_round_trip_to_empty() {
        assert!(batches_to_ipc(&[]).expect("encode empty").is_empty());
        assert!(ipc_to_batches(&[]).expect("decode empty").is_empty());
    }

    #[test]
    fn status_maps_to_typed_errors() {
        assert!(matches!(
            status_to_error("open_table", 404, "no such table"),
            InfinoError::NotFound(_)
        ));
        assert!(matches!(
            status_to_error("create_table", 409, "exists"),
            InfinoError::AlreadyExists(_)
        ));
        assert!(matches!(
            status_to_error("delete", 412, "lost the CAS"),
            InfinoError::Conflict(_)
        ));
        assert!(matches!(
            status_to_error("append", 401, "bad key"),
            InfinoError::Backend(_)
        ));
        assert!(matches!(
            status_to_error("query_sql", 500, "boom"),
            InfinoError::Backend(_)
        ));
    }
}
