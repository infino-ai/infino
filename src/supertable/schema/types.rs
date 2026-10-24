// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How a column's type is spelled in the schema document.
//!
//! A type is a set of keys spliced into the field object that carries it:
//! `"type"` names the type, and a type with parameters adds them next to
//! it (`{"type": "vector", "dim": 384}`, `{"type": "decimal", "precision":
//! 10, "scale": 2}`). Scalars are spelled by short names (`i64`,
//! `large_utf8`); a nested type carries its parts as field objects so the
//! spelling round-trips exactly, which the list depends on: the schema it
//! persists is the one every file is checked against. A type with no
//! spelling of its own travels as the Arrow IPC bytes of a one-field
//! schema under `"type": "arrow"`.

use std::sync::Arc;

use arrow_schema::{DataType, Field, Fields, IntervalUnit, Schema, TimeUnit};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::{Map, Value};

/// The keys that spell a type, in a field object.
pub type TypeKeys = Map<String, Value>;

/// The name of a type's spelling when it needs no spelling of its own.
const ARROW_FALLBACK: &str = "arrow";

/// The inner field name Arrow gives a list's element by convention, and the
/// one the compact `vector` spelling rebuilds. A fixed-size list of `f32`
/// whose element is named anything else is spelled `fixed_size_list`, which
/// carries the element's own field and so survives the round trip.
const LIST_ITEM: &str = "item";

/// `data_type` as the keys a field object carries for it.
pub fn type_keys(data_type: &DataType) -> TypeKeys {
    let mut out = TypeKeys::new();
    let tag: &str = match data_type {
        DataType::FixedSizeList(item, dim)
            if item.data_type() == &DataType::Float32 && item.name() == LIST_ITEM =>
        {
            out.insert("dim".into(), Value::from(*dim));
            if !item.is_nullable() {
                out.insert("item_nullable".into(), Value::from(false));
            }
            "vector"
        }
        DataType::FixedSizeList(item, dim) => {
            out.insert("item".into(), Value::Object(field_keys(item)));
            out.insert("dim".into(), Value::from(*dim));
            "fixed_size_list"
        }
        DataType::List(item) => {
            out.insert("item".into(), Value::Object(field_keys(item)));
            "list"
        }
        DataType::LargeList(item) => {
            out.insert("item".into(), Value::Object(field_keys(item)));
            "large_list"
        }
        DataType::Struct(fields) => {
            out.insert("fields".into(), fields_value(fields));
            "struct"
        }
        DataType::Map(entries, keys_sorted) => {
            out.insert("entries".into(), Value::Object(field_keys(entries)));
            out.insert("keys_sorted".into(), Value::from(*keys_sorted));
            "map"
        }
        DataType::Timestamp(unit, tz) => {
            if let Some(tz) = tz {
                out.insert("tz".into(), Value::from(tz.as_ref()));
            }
            match unit {
                TimeUnit::Second => "timestamp_s",
                TimeUnit::Millisecond => "timestamp_ms",
                TimeUnit::Microsecond => "timestamp_us",
                TimeUnit::Nanosecond => "timestamp_ns",
            }
        }
        DataType::Decimal128(precision, scale) => {
            decimal_keys(&mut out, *precision, *scale);
            "decimal"
        }
        DataType::Decimal256(precision, scale) => {
            decimal_keys(&mut out, *precision, *scale);
            "decimal256"
        }
        DataType::FixedSizeBinary(width) => {
            out.insert("width".into(), Value::from(*width));
            "fixed_size_binary"
        }
        other => match scalar_spelling(other) {
            Some(spelling) => spelling,
            None => {
                out.insert(
                    "ipc".into(),
                    Value::from(BASE64.encode(one_field_ipc(other))),
                );
                ARROW_FALLBACK
            }
        },
    };
    out.insert("type".into(), Value::from(tag));
    out
}

/// The type a field object's keys spell.
pub fn data_type_from_keys(keys: &Map<String, Value>) -> Result<DataType, String> {
    let tag = keys
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "field has no type".to_string())?;
    let item = |keys: &Map<String, Value>| -> Result<Arc<Field>, String> {
        let item = keys
            .get("item")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{tag} type has no item"))?;
        field_from_keys(item).map(Arc::new)
    };
    let dim = |keys: &Map<String, Value>| -> Result<i32, String> {
        let dim = keys
            .get("dim")
            .and_then(Value::as_i64)
            .and_then(|d| i32::try_from(d).ok())
            .ok_or_else(|| format!("{tag} type has no dim"))?;
        // Arrow sizes a fixed-size list's child as `dim * rows`, so a
        // negative dim overflows that multiplication the first time a row
        // is null-filled. Refuse it where the number is still the user's.
        if dim <= 0 {
            return Err(format!("{tag} type needs a positive dim, not {dim}"));
        }
        Ok(dim)
    };
    Ok(match tag {
        "vector" => {
            let nullable = keys
                .get("item_nullable")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            DataType::FixedSizeList(
                Arc::new(Field::new(LIST_ITEM, DataType::Float32, nullable)),
                dim(keys)?,
            )
        }
        "fixed_size_list" => DataType::FixedSizeList(item(keys)?, dim(keys)?),
        "list" => DataType::List(item(keys)?),
        "large_list" => DataType::LargeList(item(keys)?),
        "struct" => {
            let fields = keys
                .get("fields")
                .and_then(Value::as_array)
                .ok_or_else(|| "struct type has no fields".to_string())?;
            DataType::Struct(fields_from_value(fields)?)
        }
        "map" => {
            let entries = keys
                .get("entries")
                .and_then(Value::as_object)
                .ok_or_else(|| "map type has no entries".to_string())?;
            let keys_sorted = keys
                .get("keys_sorted")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            DataType::Map(Arc::new(field_from_keys(entries)?), keys_sorted)
        }
        "timestamp_s" | "timestamp_ms" | "timestamp_us" | "timestamp_ns" => {
            let unit = match tag {
                "timestamp_s" => TimeUnit::Second,
                "timestamp_ms" => TimeUnit::Millisecond,
                "timestamp_us" => TimeUnit::Microsecond,
                _ => TimeUnit::Nanosecond,
            };
            let tz = keys
                .get("tz")
                .and_then(Value::as_str)
                .map(|tz| Arc::from(tz.to_owned()));
            DataType::Timestamp(unit, tz)
        }
        "decimal" => {
            let (precision, scale) = decimal_from_keys(keys)?;
            DataType::Decimal128(precision, scale)
        }
        "decimal256" => {
            let (precision, scale) = decimal_from_keys(keys)?;
            DataType::Decimal256(precision, scale)
        }
        "fixed_size_binary" => {
            let width = keys
                .get("width")
                .and_then(Value::as_i64)
                .and_then(|w| i32::try_from(w).ok())
                .ok_or_else(|| "fixed_size_binary type has no width".to_string())?;
            if width <= 0 {
                return Err(format!(
                    "fixed_size_binary type needs a positive width, not {width}"
                ));
            }
            DataType::FixedSizeBinary(width)
        }
        ARROW_FALLBACK => {
            let ipc = keys
                .get("ipc")
                .and_then(Value::as_str)
                .ok_or_else(|| "arrow type has no ipc bytes".to_string())?;
            let bytes = BASE64
                .decode(ipc)
                .map_err(|e| format!("arrow type ipc bytes: {e}"))?;
            one_field_from_ipc(&bytes)?
        }
        other => {
            scalar_from_spelling(other).ok_or_else(|| format!("unknown field type '{other}'"))?
        }
    })
}

/// A field object: its type's keys plus `name` and `nullable`.
pub fn field_keys(field: &Field) -> Map<String, Value> {
    let mut out = type_keys(field.data_type());
    out.insert("name".into(), Value::from(field.name().as_str()));
    out.insert("nullable".into(), Value::from(field.is_nullable()));
    out
}

/// The field a field object describes. `nullable` defaults to `true`.
pub fn field_from_keys(keys: &Map<String, Value>) -> Result<Field, String> {
    let name = keys
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "field has no name".to_string())?;
    let nullable = keys
        .get("nullable")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    Ok(Field::new(name, data_type_from_keys(keys)?, nullable))
}

fn fields_value(fields: &Fields) -> Value {
    Value::Array(
        fields
            .iter()
            .map(|f| Value::Object(field_keys(f)))
            .collect(),
    )
}

fn fields_from_value(fields: &[Value]) -> Result<Fields, String> {
    fields
        .iter()
        .map(|f| {
            f.as_object()
                .ok_or_else(|| "a nested field is not an object".to_string())
                .and_then(field_from_keys)
                .map(Arc::new)
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Fields::from)
}

fn decimal_keys(out: &mut TypeKeys, precision: u8, scale: i8) {
    out.insert("precision".into(), Value::from(precision));
    out.insert("scale".into(), Value::from(scale));
}

fn decimal_from_keys(keys: &Map<String, Value>) -> Result<(u8, i8), String> {
    let precision = keys
        .get("precision")
        .and_then(Value::as_u64)
        .and_then(|p| u8::try_from(p).ok())
        .ok_or_else(|| "decimal type has no precision".to_string())?;
    let scale = keys
        .get("scale")
        .and_then(Value::as_i64)
        .and_then(|s| i8::try_from(s).ok())
        .ok_or_else(|| "decimal type has no scale".to_string())?;
    Ok((precision, scale))
}

/// The short name of a type with no parameters.
fn scalar_spelling(data_type: &DataType) -> Option<&'static str> {
    Some(match data_type {
        DataType::Null => "null",
        DataType::Boolean => "bool",
        DataType::Int8 => "i8",
        DataType::Int16 => "i16",
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        DataType::UInt8 => "u8",
        DataType::UInt16 => "u16",
        DataType::UInt32 => "u32",
        DataType::UInt64 => "u64",
        DataType::Float16 => "f16",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => "utf8",
        DataType::LargeUtf8 => "large_utf8",
        DataType::Utf8View => "utf8_view",
        DataType::Binary => "binary",
        DataType::LargeBinary => "large_binary",
        DataType::BinaryView => "binary_view",
        DataType::Date32 => "date32",
        DataType::Date64 => "date64",
        DataType::Time32(TimeUnit::Second) => "time32_s",
        DataType::Time32(TimeUnit::Millisecond) => "time32_ms",
        DataType::Time64(TimeUnit::Microsecond) => "time64_us",
        DataType::Time64(TimeUnit::Nanosecond) => "time64_ns",
        DataType::Duration(TimeUnit::Second) => "duration_s",
        DataType::Duration(TimeUnit::Millisecond) => "duration_ms",
        DataType::Duration(TimeUnit::Microsecond) => "duration_us",
        DataType::Duration(TimeUnit::Nanosecond) => "duration_ns",
        DataType::Interval(IntervalUnit::YearMonth) => "interval_year_month",
        DataType::Interval(IntervalUnit::DayTime) => "interval_day_time",
        DataType::Interval(IntervalUnit::MonthDayNano) => "interval_month_day_nano",
        _ => return None,
    })
}

/// The type a short name spells; the long Arrow names are accepted too.
fn scalar_from_spelling(spelling: &str) -> Option<DataType> {
    Some(match spelling {
        "null" => DataType::Null,
        "bool" | "boolean" => DataType::Boolean,
        "i8" | "int8" => DataType::Int8,
        "i16" | "int16" => DataType::Int16,
        "i32" | "int32" => DataType::Int32,
        "i64" | "int64" => DataType::Int64,
        "u8" | "uint8" => DataType::UInt8,
        "u16" | "uint16" => DataType::UInt16,
        "u32" | "uint32" => DataType::UInt32,
        "u64" | "uint64" => DataType::UInt64,
        "f16" | "float16" => DataType::Float16,
        "f32" | "float32" => DataType::Float32,
        "f64" | "float64" | "double" => DataType::Float64,
        "utf8" | "string" => DataType::Utf8,
        "large_utf8" | "large_string" => DataType::LargeUtf8,
        "utf8_view" => DataType::Utf8View,
        "binary" => DataType::Binary,
        "large_binary" => DataType::LargeBinary,
        "binary_view" => DataType::BinaryView,
        "date32" | "date" => DataType::Date32,
        "date64" => DataType::Date64,
        "time32_s" => DataType::Time32(TimeUnit::Second),
        "time32_ms" => DataType::Time32(TimeUnit::Millisecond),
        "time64_us" => DataType::Time64(TimeUnit::Microsecond),
        "time64_ns" => DataType::Time64(TimeUnit::Nanosecond),
        "duration_s" => DataType::Duration(TimeUnit::Second),
        "duration_ms" => DataType::Duration(TimeUnit::Millisecond),
        "duration_us" => DataType::Duration(TimeUnit::Microsecond),
        "duration_ns" => DataType::Duration(TimeUnit::Nanosecond),
        "interval_year_month" => DataType::Interval(IntervalUnit::YearMonth),
        "interval_day_time" => DataType::Interval(IntervalUnit::DayTime),
        "interval_month_day_nano" => DataType::Interval(IntervalUnit::MonthDayNano),
        "timestamp" => DataType::Timestamp(TimeUnit::Millisecond, None),
        _ => return None,
    })
}

/// Arrow IPC bytes of a schema with one field of `data_type`.
fn one_field_ipc(data_type: &DataType) -> Vec<u8> {
    let schema = Schema::new(vec![Field::new("item", data_type.clone(), true)]);
    let mut buf = Vec::new();
    {
        let mut w = arrow::ipc::writer::StreamWriter::try_new(&mut buf, &schema)
            .expect("IPC stream writer over an in-memory buffer");
        w.finish().expect("finish IPC stream");
    }
    buf
}

fn one_field_from_ipc(bytes: &[u8]) -> Result<DataType, String> {
    let reader = arrow::ipc::reader::StreamReader::try_new(std::io::Cursor::new(bytes), None)
        .map_err(|e| format!("arrow type ipc bytes: {e}"))?;
    let schema = reader.schema();
    match schema.fields().first() {
        Some(field) if schema.fields().len() == 1 => Ok(field.data_type().clone()),
        _ => Err("arrow type ipc bytes hold no single field".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data_type: DataType) {
        let keys = type_keys(&data_type);
        let back = data_type_from_keys(&keys).expect("decode");
        assert_eq!(back, data_type, "{keys:?}");
    }

    #[test]
    fn every_scalar_spelling_round_trips() {
        for dt in [
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::BinaryView,
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time32(TimeUnit::Millisecond),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Duration(TimeUnit::Second),
            DataType::Duration(TimeUnit::Nanosecond),
            DataType::Interval(IntervalUnit::YearMonth),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            DataType::Timestamp(TimeUnit::Nanosecond, Some(Arc::from("+05:30"))),
            DataType::Decimal128(38, 0),
            DataType::Decimal128(10, -2),
            DataType::Decimal256(76, 10),
            DataType::FixedSizeBinary(16),
        ] {
            round_trip(dt);
        }
    }

    #[test]
    fn vectors_spell_as_vector_with_dim_and_keep_item_nullability() {
        let nullable_item =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 384);
        let keys = type_keys(&nullable_item);
        assert_eq!(keys["type"], "vector");
        assert_eq!(keys["dim"], 384);
        assert!(!keys.contains_key("item_nullable"));
        round_trip(nullable_item);

        let non_null_item =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, false)), 4);
        assert_eq!(type_keys(&non_null_item)["item_nullable"], false);
        round_trip(non_null_item);
    }

    #[test]
    fn nested_types_round_trip_with_their_parts() {
        round_trip(DataType::FixedSizeList(
            Arc::new(Field::new("x", DataType::Int16, false)),
            3,
        ));
        round_trip(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Utf8,
            true,
        ))));
        round_trip(DataType::LargeList(Arc::new(Field::new(
            "elem",
            DataType::Decimal128(5, 2),
            false,
        ))));
        let inner = DataType::Struct(Fields::from(vec![
            Field::new("a", DataType::Int64, false),
            Field::new(
                "b",
                DataType::List(Arc::new(Field::new("item", DataType::Boolean, true))),
                true,
            ),
        ]));
        round_trip(inner.clone());
        let entries = Field::new(
            "entries",
            DataType::Struct(Fields::from(vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", inner, true),
            ])),
            false,
        );
        round_trip(DataType::Map(Arc::new(entries), true));
    }

    #[test]
    fn a_type_without_a_spelling_travels_as_arrow_ipc() {
        let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let keys = type_keys(&dictionary);
        assert_eq!(keys["type"], "arrow");
        assert!(keys["ipc"].is_string());
        round_trip(dictionary);
    }

    #[test]
    fn long_arrow_names_are_accepted_on_read() {
        for (spelling, expected) in [
            ("int64", DataType::Int64),
            ("string", DataType::Utf8),
            ("large_string", DataType::LargeUtf8),
            ("boolean", DataType::Boolean),
            ("double", DataType::Float64),
            ("date", DataType::Date32),
            (
                "timestamp",
                DataType::Timestamp(TimeUnit::Millisecond, None),
            ),
        ] {
            let mut keys = Map::new();
            keys.insert("type".into(), Value::from(spelling));
            assert_eq!(data_type_from_keys(&keys).expect("decode"), expected);
        }
    }

    #[test]
    fn malformed_type_keys_are_errors_not_guesses() {
        for json in [
            r#"{}"#,
            r#"{"type": "nope"}"#,
            r#"{"type": "vector"}"#,
            r#"{"type": "decimal", "precision": 5}"#,
            r#"{"type": "list"}"#,
            r#"{"type": "struct"}"#,
            r#"{"type": "arrow", "ipc": "!!"}"#,
        ] {
            let keys: Map<String, Value> = serde_json::from_str(json).expect("json");
            assert!(data_type_from_keys(&keys).is_err(), "{json}");
        }
    }

    #[test]
    fn a_field_object_carries_name_and_nullability() {
        let field = Field::new("score", DataType::Float64, false);
        let keys = field_keys(&field);
        assert_eq!(keys["name"], "score");
        assert_eq!(keys["nullable"], false);
        assert_eq!(field_from_keys(&keys).expect("decode"), field);
        let mut without_nullable = keys.clone();
        without_nullable.remove("nullable");
        assert!(
            field_from_keys(&without_nullable)
                .expect("decode")
                .is_nullable(),
            "nullable defaults to true"
        );
    }
}

#[cfg(test)]
mod vector_spelling_tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field};

    use super::*;

    /// Dimension of the vectors here; any fixed width does.
    const DIM: i32 = 384;

    /// The compact `vector` spelling rebuilds the element field, so it is
    /// only usable for the element name it rebuilds. A column whose element
    /// carries another name, which is what a Parquet source commonly hands
    /// over, takes the general spelling and keeps it.
    #[test]
    fn a_vector_whose_element_is_named_otherwise_keeps_its_name() {
        let conventional = DataType::FixedSizeList(
            Arc::new(Field::new(LIST_ITEM, DataType::Float32, true)),
            DIM,
        );
        let keys = type_keys(&conventional);
        assert_eq!(keys.get("type").and_then(Value::as_str), Some("vector"));
        assert_eq!(
            data_type_from_keys(&keys).expect("decode"),
            conventional,
            "the conventional element still takes the compact spelling"
        );

        let named = DataType::FixedSizeList(
            Arc::new(Field::new("element", DataType::Float32, true)),
            DIM,
        );
        let keys = type_keys(&named);
        assert_eq!(
            keys.get("type").and_then(Value::as_str),
            Some("fixed_size_list"),
            "an element named otherwise cannot use a spelling that renames it"
        );
        assert_eq!(
            data_type_from_keys(&keys).expect("decode"),
            named,
            "and the name survives, so a reopened table takes the same batches"
        );

        // The element's nullability survives either way.
        let not_null = DataType::FixedSizeList(
            Arc::new(Field::new(LIST_ITEM, DataType::Float32, false)),
            DIM,
        );
        assert_eq!(
            data_type_from_keys(&type_keys(&not_null)).expect("decode"),
            not_null
        );
    }
}
