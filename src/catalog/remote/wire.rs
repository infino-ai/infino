// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Wire encoding for the remote transport.
//!
//! Requests are JSON; read responses are Arrow IPC streams that decode straight
//! into the engine's native `Vec<RecordBatch>`. This module owns the
//! translations: a schema to and from Arrow's own IPC encoding, an
//! [`IndexSpec`] to its JSON request shape, `RecordBatch`es to/from the Arrow
//! IPC stream, and an HTTP status to an [`InfinoError`].
//!
//! The schema crosses as an Arrow IPC schema message rather than a hand-rolled
//! JSON descriptor. That is Arrow's canonical encoding, so every type the
//! engine can hold survives the trip, nested structs and lists included, and
//! there is no per-type spelling table to keep in step with the service.

use std::io::Cursor;

use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Value, json};

use crate::{IndexSpec, InfinoError, Metric, superfile::fts::tokenize::ASCII_LOWER_TOKENIZER};

/// Content type for an Arrow IPC streaming body — the encoding for `append`
/// bodies and read responses.
pub(crate) const ARROW_STREAM_CONTENT_TYPE: &str = "application/vnd.apache.arrow.stream";

/// The `encoding` value that asks a schema response for Arrow IPC rather than
/// the JSON column descriptors. Only IPC can carry a nested column type, so the
/// client always asks for it.
pub(crate) const IPC_ENCODING: &str = "ipc";

/// The wire spelling for a vector distance metric.
pub(crate) fn metric_str(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::L2Sq => "l2sq",
        Metric::NegDot => "negdot",
    }
}

/// An [`IndexSpec`] as the `indexes` object of a create-table request:
/// `{fts: [entry, …], vector: [{column, dim, metric}, …]}`. Absent index kinds
/// are omitted (the server treats a missing key as "none").
///
/// Each FTS entry is a bare column name when it uses the default `ascii_lower`
/// analyzer, or a `{column, analyzer}` object when it names a different one, so
/// the chosen analyzer reaches the server rather than being dropped. The bare
/// form for the default keeps the request identical to what older servers
/// expect.
pub(crate) fn index_spec_to_json(spec: &IndexSpec) -> Value {
    let mut indexes = serde_json::Map::new();
    let columns = spec.fts_columns();
    if !columns.is_empty() {
        let fts: Vec<Value> = columns
            .iter()
            .zip(spec.fts_analyzers())
            .map(|(column, analyzer)| {
                if analyzer == ASCII_LOWER_TOKENIZER {
                    json!(column)
                } else {
                    json!({ "column": column, "analyzer": analyzer })
                }
            })
            .collect();
        indexes.insert("fts".to_string(), Value::Array(fts));
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

/// Encode a schema as the `schema_ipc` field of a create-table request: an
/// Arrow IPC stream carrying only the schema message, base64 for the JSON body.
///
/// Base64 rather than a raw body because the request also carries `table_name`
/// and the index spec, which belong in JSON; the overhead is irrelevant on a
/// schema of a few kilobytes.
pub(crate) fn schema_to_ipc_base64(schema: &Schema) -> Result<String, InfinoError> {
    let mut out = Vec::new();
    let mut writer = StreamWriter::try_new(&mut out, schema)
        .map_err(|e| InfinoError::Backend(format!("arrow ipc schema writer: {e}")))?;
    writer
        .finish()
        .map_err(|e| InfinoError::Backend(format!("arrow ipc schema finish: {e}")))?;
    Ok(BASE64.encode(&out))
}

/// Decode a schema response body — an Arrow IPC stream whose schema message is
/// the whole payload — into an Arrow schema. Raw bytes, not base64: the
/// response body is the schema, so it needs no JSON envelope.
pub(crate) fn ipc_to_schema(bytes: &[u8]) -> Result<Schema, InfinoError> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|e| InfinoError::Backend(format!("arrow ipc schema reader: {e}")))?;
    Ok(reader.schema().as_ref().clone())
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
    fn index_spec_json_carries_a_non_default_analyzer() {
        // A named analyzer must cross the wire as a {column, analyzer} object,
        // so the server builds the index with it rather than the default. A
        // default-analyzer column alongside it stays a bare string, so only the
        // columns that need the object form pay for it.
        let spec = IndexSpec::new()
            .fts("title")
            .fts_with_analyzer("body", "standard");
        let json = index_spec_to_json(&spec);
        assert_eq!(
            json["fts"],
            json!(["title", {"column": "body", "analyzer": "standard"}])
        );
    }

    #[test]
    fn index_spec_json_bare_form_for_the_default_analyzer() {
        // An explicit ascii_lower is the default, so it serializes identically to
        // a bare column name — no needless object form, and byte-identical to
        // what an older server expects.
        let spec = IndexSpec::new().fts_with_analyzer("body", "ascii_lower");
        assert_eq!(index_spec_to_json(&spec)["fts"], json!(["body"]));
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
    fn schema_ipc_round_trips_a_nested_column() {
        // A struct column is the case the previous JSON descriptors could not
        // express at all, so it is the one worth pinning.
        let inner: Fields = vec![
            FieldRef::from(Field::new("url", DataType::Utf8, true)),
            FieldRef::from(Field::new("height", DataType::Int64, true)),
        ]
        .into();
        let schema = Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("image", DataType::Struct(inner), true),
        ]);

        let encoded = schema_to_ipc_base64(&schema).expect("encode schema");
        let bytes = BASE64.decode(&encoded).expect("valid base64");
        let back = ipc_to_schema(&bytes).expect("decode schema");

        assert_eq!(back, schema, "the schema survives the round trip exactly");
    }

    #[test]
    fn schema_ipc_rejects_a_body_that_is_not_an_ipc_stream() {
        assert!(matches!(
            ipc_to_schema(b"not arrow ipc"),
            Err(InfinoError::Backend(_))
        ));
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
