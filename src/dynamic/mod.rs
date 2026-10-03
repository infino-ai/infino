// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Rows as JSON documents become one Arrow batch.
//!
//! Every document is flattened to dot paths and every path is given one
//! Arrow type from its values: `Boolean`, `Int64`, `Float64`, `LargeUtf8`,
//! or a `List` of those for arrays. The mapper infers and never checks:
//! the table's schema is consulted only to resolve an ambiguous literal
//! (an integral number on a `Float64` path is a float, a string on a
//! `Utf8` path is `Utf8`), so a batch that disagrees with the schema is
//! refused by the same resolver an Arrow producer meets, with the same
//! error. A template decides the type and index of a path the table does
//! not have yet. JSON `null` and `[]` create nothing. Columns come out in
//! key order, which is sorted, so a batch's shape does not depend on the
//! order a producer wrote its keys in.

use std::{collections::HashMap, sync::Arc};

use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, FixedSizeListArray, Float32Array,
    Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, LargeListArray, LargeStringArray,
    ListArray, RecordBatch, RecordBatchOptions, StringArray, StringViewArray,
    TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde_json::{Map, Value};

use crate::supertable::schema::{
    Detected, INDEX_META_KEY, TableSchema, error::SchemaError, index_to_json,
};

/// Integers above this magnitude are not exactly representable as `f64`.
const F64_EXACT_INT_BOUND: i128 = 1 << 53;
/// Milliseconds in a day, for `Date64`.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// A value a document carries on one path, with arrays kept whole.
#[derive(Debug, Clone)]
enum Leaf<'a> {
    Scalar(&'a Value),
    List(Vec<&'a Value>),
}

/// The kind of value a path carries, before a type is chosen for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Int,
    Float,
    Str,
}

impl Kind {
    fn of(value: &Value) -> Option<Kind> {
        match value {
            Value::Bool(_) => Some(Kind::Bool),
            Value::Number(n) if n.is_f64() => Some(Kind::Float),
            Value::Number(_) => Some(Kind::Int),
            Value::String(_) => Some(Kind::Str),
            _ => None,
        }
    }

    fn detected(self) -> Detected {
        match self {
            Kind::Bool => Detected::Boolean,
            Kind::Int => Detected::Integer,
            Kind::Float => Detected::Float,
            Kind::Str => Detected::String,
        }
    }

    /// The type this kind infers on a path the table does not have.
    fn inferred(self) -> DataType {
        match self {
            Kind::Bool => DataType::Boolean,
            Kind::Int => DataType::Int64,
            Kind::Float => DataType::Float64,
            Kind::Str => DataType::LargeUtf8,
        }
    }

    /// The kind two values share: integers and floats are numbers, and
    /// anything else must agree.
    fn join(self, other: Kind) -> Option<Kind> {
        match (self, other) {
            (a, b) if a == b => Some(a),
            (Kind::Int, Kind::Float) | (Kind::Float, Kind::Int) => Some(Kind::Float),
            _ => None,
        }
    }
}

/// What the mapper knows about one path across the batch.
struct Column<'a> {
    path: String,
    /// One entry per row; `None` where the row lacks the path.
    cells: Vec<Option<Leaf<'a>>>,
    /// The distinct kinds of the scalars seen (or of the list elements
    /// seen), in order of appearance.
    kinds: Vec<Kind>,
    /// Whether any value was an array.
    list: bool,
}

impl Column<'_> {
    /// Every scalar value the column carries, lists flattened.
    fn values(&self) -> impl Iterator<Item = &Value> + '_ {
        self.cells.iter().flatten().flat_map(|leaf| match leaf {
            Leaf::Scalar(v) => vec![*v],
            Leaf::List(vs) => vs.clone(),
        })
    }

    /// The one kind the column's values share. Integers and floats are
    /// numbers; strings and integers both name a point in time when the
    /// column is a timestamp; anything else that mixes is refused — as a
    /// mixed array on a list path, as a type disagreement otherwise.
    fn settle_kind(&self, target: Option<&DataType>) -> Result<Kind, SchemaError> {
        match self.kinds.as_slice() {
            [] => Err(SchemaError::InvalidRow {
                row: 0,
                reason: format!("path `{}` has no values", self.path),
            }),
            [kind] => Ok(*kind),
            [a, b] if a.join(*b).is_some() => Ok(a.join(*b).expect("joinable")),
            kinds
                if matches!(target, Some(DataType::Timestamp(_, _)))
                    && kinds.iter().all(|k| matches!(k, Kind::Str | Kind::Int))
                    && self.values().all(parses_as_time) =>
            {
                Ok(Kind::Str)
            }
            kinds if self.list => Err(SchemaError::MixedArray {
                column: self.path.clone(),
                types: kinds.iter().map(|k| k.inferred().to_string()).collect(),
            }),
            kinds => Err(SchemaError::TypeMismatch {
                column: self.path.clone(),
                frozen: kinds[0].inferred(),
                offered: kinds[1].inferred(),
            }),
        }
    }
}

/// Map `rows` to one batch under `schema`'s rules. Every row is a JSON
/// object; see the module docs for how paths are typed.
pub fn rows_to_batch(rows: &[Value], schema: &TableSchema) -> Result<RecordBatch, SchemaError> {
    let mut columns: Vec<Column<'_>> = Vec::new();
    let mut by_path: HashMap<String, usize> = HashMap::new();
    for (row, value) in rows.iter().enumerate() {
        let object = value.as_object().ok_or_else(|| SchemaError::InvalidRow {
            row,
            reason: "a row is a JSON object".to_owned(),
        })?;
        let mut leaves = Vec::new();
        flatten(object, "", 1, schema.max_depth(), &mut leaves)?;
        for (path, leaf) in leaves {
            let index = match by_path.get(&path) {
                Some(&i) => i,
                None => {
                    columns.push(Column {
                        path: path.clone(),
                        cells: vec![None; rows.len()],
                        kinds: Vec::new(),
                        list: false,
                    });
                    by_path.insert(path, columns.len() - 1);
                    columns.len() - 1
                }
            };
            let column = &mut columns[index];
            observe(column, &leaf)?;
            column.cells[row] = Some(leaf);
        }
    }

    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for column in &columns {
        // A path that only ever carried empty arrays creates nothing.
        let Some(first_kind) = column.kinds.first().copied() else {
            continue;
        };
        let detected = if column.list {
            Detected::List
        } else {
            first_kind.detected()
        };
        let mut metadata = HashMap::new();
        let target = match schema.id_of(&column.path) {
            Some(id) => schema
                .fields()
                .iter()
                .find(|f| f.id == id)
                .map(|f| f.data_type.clone()),
            None => match schema.template_for(&column.path, detected) {
                Some(template) => {
                    if let Some(index) = &template.index {
                        metadata
                            .insert(INDEX_META_KEY.to_owned(), index_to_json(index).to_string());
                    }
                    template.data_type.clone()
                }
                None => None,
            },
        };
        let kind = column.settle_kind(target.as_ref())?;
        let data_type = resolve_type(column, kind, target.as_ref());
        let array = build_array(column, &data_type)?;
        fields.push(Field::new(&column.path, data_type, true).with_metadata(metadata));
        arrays.push(array);
    }
    RecordBatch::try_new_with_options(
        Arc::new(Schema::new(fields)),
        arrays,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .map_err(|e| SchemaError::InvalidRow {
        row: 0,
        reason: e.to_string(),
    })
}

/// Record one more value of a column: scalars and arrays never mix on one
/// path, every value is a scalar JSON kind, and the kinds seen are kept for
/// [`Column::settle_kind`] once the column's target type is known.
fn observe<'a>(column: &mut Column<'a>, leaf: &Leaf<'a>) -> Result<(), SchemaError> {
    let (values, list): (Vec<&Value>, bool) = match leaf {
        Leaf::Scalar(v) => (vec![*v], false),
        Leaf::List(vs) => (vs.clone(), true),
    };
    let seen_scalars = !column.kinds.is_empty() && !column.list;
    if (column.list && !list) || (seen_scalars && list) {
        return Err(SchemaError::MixedArray {
            column: column.path.clone(),
            types: vec!["array".to_owned(), "scalar".to_owned()],
        });
    }
    if list {
        column.list = true;
    }
    for value in values {
        let kind = Kind::of(value).ok_or_else(|| SchemaError::MixedArray {
            column: column.path.clone(),
            types: vec![json_kind_name(value).to_owned()],
        })?;
        if !column.kinds.contains(&kind) {
            column.kinds.push(kind);
        }
    }
    Ok(())
}

fn json_kind_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Flatten `object` to dot paths under `prefix`. `depth` is the object's
/// nesting level, `1` for a document itself.
fn flatten<'a>(
    object: &'a Map<String, Value>,
    prefix: &str,
    depth: u32,
    max_depth: u32,
    out: &mut Vec<(String, Leaf<'a>)>,
) -> Result<(), SchemaError> {
    for (key, value) in object {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Null => {}
            Value::Object(inner) => {
                if depth + 1 > max_depth {
                    return Err(SchemaError::DepthExceeded {
                        cap: max_depth,
                        path,
                    });
                }
                flatten(inner, &path, depth + 1, max_depth, out)?;
            }
            Value::Array(items) => {
                if items.is_empty() {
                    continue;
                }
                if items.iter().all(Value::is_object) {
                    // An array of objects: one list per leaf path, with the
                    // elements' values in order.
                    let mut per_path: Vec<(String, Vec<&'a Value>)> = Vec::new();
                    for item in items {
                        let mut leaves = Vec::new();
                        let inner = item.as_object().expect("every item is an object");
                        if depth + 1 > max_depth {
                            return Err(SchemaError::DepthExceeded {
                                cap: max_depth,
                                path,
                            });
                        }
                        flatten(inner, &path, depth + 1, max_depth, &mut leaves)?;
                        for (leaf_path, leaf) in leaves {
                            let values = match leaf {
                                Leaf::Scalar(v) => vec![v],
                                Leaf::List(vs) => vs,
                            };
                            match per_path.iter_mut().find(|(p, _)| *p == leaf_path) {
                                Some((_, all)) => all.extend(values),
                                None => per_path.push((leaf_path, values)),
                            }
                        }
                    }
                    out.extend(
                        per_path
                            .into_iter()
                            .map(|(p, values)| (p, Leaf::List(values))),
                    );
                } else {
                    let mut values = Vec::with_capacity(items.len());
                    for item in items {
                        match item {
                            Value::Null => {}
                            Value::Array(_) | Value::Object(_) => {
                                return Err(SchemaError::MixedArray {
                                    column: path,
                                    types: vec![json_kind_name(item).to_owned()],
                                });
                            }
                            scalar => values.push(scalar),
                        }
                    }
                    if !values.is_empty() {
                        out.push((path, Leaf::List(values)));
                    }
                }
            }
            scalar => out.push((path, Leaf::Scalar(scalar))),
        }
    }
    Ok(())
}

/// The Arrow type for `column`: `target` (the table's or a template's
/// type) when the values are exactly representable in it, else the type
/// `kind` infers. Only the ambiguity a JSON literal leaves is resolved
/// here; a disagreement is left for the resolver to refuse.
fn resolve_type(column: &Column<'_>, kind: Kind, target: Option<&DataType>) -> DataType {
    let scalar_target = match target {
        Some(DataType::List(item)) | Some(DataType::LargeList(item)) if column.list => {
            Some(item.data_type())
        }
        Some(DataType::FixedSizeList(item, dim)) if column.list => {
            let fits = matches!(kind, Kind::Int | Kind::Float)
                && item.data_type() == &DataType::Float32
                && column.cells.iter().flatten().all(|leaf| match leaf {
                    Leaf::List(values) => values.len() == *dim as usize,
                    Leaf::Scalar(_) => false,
                });
            if fits {
                return DataType::FixedSizeList(Arc::clone(item), *dim);
            }
            None
        }
        Some(other) if !column.list => Some(other),
        _ => None,
    };
    let values = || column.values();
    let element = match (kind, scalar_target) {
        (Kind::Int, Some(t)) if is_integer_type(t) && values().all(|v| int_fits(v, t)) => t.clone(),
        (Kind::Int, Some(DataType::Float64)) if values().all(int_exact_in_f64) => DataType::Float64,
        (Kind::Int | Kind::Float, Some(DataType::Float32)) if values().all(float_exact_in_f32) => {
            DataType::Float32
        }
        (Kind::Str, Some(t @ (DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View))) => {
            t.clone()
        }
        (Kind::Str, Some(t @ DataType::Timestamp(_, _))) if values().all(parses_as_time) => {
            t.clone()
        }
        (Kind::Int, Some(t @ DataType::Timestamp(_, _))) => t.clone(),
        (Kind::Str, Some(t @ (DataType::Date32 | DataType::Date64)))
            if values().all(parses_as_date) =>
        {
            t.clone()
        }
        (kind, _) => kind.inferred(),
    };
    if column.list {
        match target {
            Some(DataType::LargeList(_)) => {
                DataType::LargeList(Arc::new(Field::new("item", element, true)))
            }
            _ => DataType::List(Arc::new(Field::new("item", element, true))),
        }
    } else {
        element
    }
}

fn is_integer_type(t: &DataType) -> bool {
    matches!(
        t,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn as_i128(value: &Value) -> Option<i128> {
    let n = value.as_number()?;
    n.as_i64()
        .map(i128::from)
        .or_else(|| n.as_u64().map(i128::from))
}

fn int_fits(value: &Value, t: &DataType) -> bool {
    let Some(v) = as_i128(value) else {
        return false;
    };
    match t {
        DataType::Int8 => i8::try_from(v).is_ok(),
        DataType::Int16 => i16::try_from(v).is_ok(),
        DataType::Int32 => i32::try_from(v).is_ok(),
        DataType::Int64 => i64::try_from(v).is_ok(),
        DataType::UInt8 => u8::try_from(v).is_ok(),
        DataType::UInt16 => u16::try_from(v).is_ok(),
        DataType::UInt32 => u32::try_from(v).is_ok(),
        DataType::UInt64 => u64::try_from(v).is_ok(),
        _ => false,
    }
}

fn int_exact_in_f64(value: &Value) -> bool {
    as_i128(value).is_some_and(|v| v.abs() <= F64_EXACT_INT_BOUND)
}

fn float_exact_in_f32(value: &Value) -> bool {
    value.as_f64().is_some_and(|v| (v as f32) as f64 == v)
}

/// A string as a point in time: RFC 3339 with an offset, or a naive
/// `YYYY-MM-DDTHH:MM:SS[.fff]` read as UTC. `None` for anything else.
fn parse_time(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|t| t.with_timezone(&Utc))
        .ok()
        .or_else(|| {
            NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .ok()
                .map(|t| t.and_utc())
        })
}

/// Whether `value` names a point in time: a time string, or an integral
/// epoch count.
fn parses_as_time(value: &Value) -> bool {
    as_i128(value).is_some() || value.as_str().is_some_and(|s| parse_time(s).is_some())
}

/// A string as a calendar day, `YYYY-MM-DD`.
fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()
}

fn parses_as_date(value: &Value) -> bool {
    value.as_str().is_some_and(|s| parse_date(s).is_some())
}

/// Days since the Unix epoch for `date`.
fn epoch_days(date: NaiveDate) -> i32 {
    (date - NaiveDate::from_ymd_opt(1970, 1, 1).expect("the epoch is a date")).num_days() as i32
}

/// `column`'s values as an array of `data_type`, null where a row lacks
/// the path. The type was chosen from these values, so every one of them
/// fits.
fn build_array(column: &Column<'_>, data_type: &DataType) -> Result<ArrayRef, SchemaError> {
    match data_type {
        DataType::List(item) | DataType::LargeList(item) => {
            let mut offsets: Vec<usize> = Vec::with_capacity(column.cells.len() + 1);
            let mut nulls = Vec::with_capacity(column.cells.len());
            let mut flat: Vec<Option<&Value>> = Vec::new();
            offsets.push(0);
            for cell in &column.cells {
                match cell {
                    Some(Leaf::List(values)) => {
                        flat.extend(values.iter().copied().map(Some));
                        nulls.push(true);
                    }
                    _ => nulls.push(false),
                }
                offsets.push(flat.len());
            }
            let child = build_scalar_array(&flat, item.data_type())?;
            let validity = NullBuffer::from(nulls);
            let array: ArrayRef = if matches!(data_type, DataType::List(_)) {
                let offsets = OffsetBuffer::<i32>::new(
                    offsets.iter().map(|&o| o as i32).collect::<Vec<_>>().into(),
                );
                Arc::new(ListArray::new(
                    Arc::clone(item),
                    offsets,
                    child,
                    Some(validity),
                ))
            } else {
                let offsets = OffsetBuffer::<i64>::new(
                    offsets.iter().map(|&o| o as i64).collect::<Vec<_>>().into(),
                );
                Arc::new(LargeListArray::new(
                    Arc::clone(item),
                    offsets,
                    child,
                    Some(validity),
                ))
            };
            Ok(array)
        }
        DataType::FixedSizeList(item, dim) => {
            let mut flat: Vec<Option<&Value>> = Vec::new();
            let mut nulls = Vec::with_capacity(column.cells.len());
            for cell in &column.cells {
                match cell {
                    Some(Leaf::List(values)) => {
                        flat.extend(values.iter().copied().map(Some));
                        nulls.push(true);
                    }
                    _ => {
                        flat.extend(std::iter::repeat_n(None, *dim as usize));
                        nulls.push(false);
                    }
                }
            }
            let child = build_scalar_array(&flat, item.data_type())?;
            Ok(Arc::new(FixedSizeListArray::new(
                Arc::clone(item),
                *dim,
                child,
                Some(NullBuffer::from(nulls)),
            )))
        }
        scalar => {
            let values: Vec<Option<&Value>> = column
                .cells
                .iter()
                .map(|cell| match cell {
                    Some(Leaf::Scalar(v)) => Some(*v),
                    _ => None,
                })
                .collect();
            build_scalar_array(&values, scalar)
        }
    }
}

fn build_scalar_array(
    values: &[Option<&Value>],
    data_type: &DataType,
) -> Result<ArrayRef, SchemaError> {
    macro_rules! ints {
        ($array:ty, $native:ty) => {
            Arc::new(<$array>::from(
                values
                    .iter()
                    .map(|v| {
                        v.and_then(as_i128)
                            .and_then(|n| <$native>::try_from(n).ok())
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef
        };
    }
    let array: ArrayRef = match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_bool))
                .collect::<Vec<_>>(),
        )),
        DataType::Int8 => ints!(Int8Array, i8),
        DataType::Int16 => ints!(Int16Array, i16),
        DataType::Int32 => ints!(Int32Array, i32),
        DataType::Int64 => ints!(Int64Array, i64),
        DataType::UInt8 => ints!(UInt8Array, u8),
        DataType::UInt16 => ints!(UInt16Array, u16),
        DataType::UInt32 => ints!(UInt32Array, u32),
        DataType::UInt64 => ints!(UInt64Array, u64),
        DataType::Float32 => Arc::new(Float32Array::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_f64).map(|f| f as f32))
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_f64))
                .collect::<Vec<_>>(),
        )),
        DataType::Utf8 => Arc::new(StringArray::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_str))
                .collect::<Vec<_>>(),
        )),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_str))
                .collect::<Vec<_>>(),
        )),
        DataType::Utf8View => Arc::new(StringViewArray::from(
            values
                .iter()
                .map(|v| v.and_then(Value::as_str))
                .collect::<Vec<_>>(),
        )),
        DataType::Date32 => Arc::new(Date32Array::from(
            values
                .iter()
                .map(|v| (*v)?.as_str().and_then(parse_date).map(epoch_days))
                .collect::<Vec<_>>(),
        )),
        DataType::Date64 => Arc::new(Date64Array::from(
            values
                .iter()
                .map(|v| {
                    (*v)?
                        .as_str()
                        .and_then(parse_date)
                        .map(|d| i64::from(epoch_days(d)) * MILLIS_PER_DAY)
                })
                .collect::<Vec<_>>(),
        )),
        DataType::Timestamp(unit, tz) => {
            // A string is a point in time; an integral literal is already an
            // epoch count in the column's unit.
            let stamps: Vec<Option<i64>> = values
                .iter()
                .map(|v| {
                    let v = (*v)?;
                    if let Some(n) = as_i128(v) {
                        return i64::try_from(n).ok();
                    }
                    let t = parse_time(v.as_str()?)?;
                    match unit {
                        TimeUnit::Second => Some(t.timestamp()),
                        TimeUnit::Millisecond => Some(t.timestamp_millis()),
                        TimeUnit::Microsecond => Some(t.timestamp_micros()),
                        TimeUnit::Nanosecond => t.timestamp_nanos_opt(),
                    }
                })
                .collect();
            match unit {
                TimeUnit::Second => {
                    Arc::new(TimestampSecondArray::from(stamps).with_timezone_opt(tz.clone()))
                }
                TimeUnit::Millisecond => {
                    Arc::new(TimestampMillisecondArray::from(stamps).with_timezone_opt(tz.clone()))
                }
                TimeUnit::Microsecond => {
                    Arc::new(TimestampMicrosecondArray::from(stamps).with_timezone_opt(tz.clone()))
                }
                TimeUnit::Nanosecond => {
                    Arc::new(TimestampNanosecondArray::from(stamps).with_timezone_opt(tz.clone()))
                }
            }
        }
        other => {
            return Err(SchemaError::InvalidRow {
                row: 0,
                reason: format!("values cannot be mapped to `{other}`"),
            });
        }
    };
    Ok(array)
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        cast::AsArray,
        types::{Float64Type, Int32Type, Int64Type},
    };
    use serde_json::json;

    use super::*;
    use crate::supertable::schema::{ColumnIndex, Template, resolve::resolve_batch};

    fn table(fields: Vec<(&str, DataType)>) -> TableSchema {
        TableSchema::from_user_schema(&Schema::new(
            fields
                .into_iter()
                .map(|(n, t)| Field::new(n, t, true))
                .collect::<Vec<_>>(),
        ))
    }

    /// Column types by name. Columns come out in key order, which
    /// serde_json keeps sorted, so a batch's shape never depends on the
    /// order a producer wrote its keys in.
    fn types(batch: &RecordBatch) -> HashMap<String, DataType> {
        batch
            .schema()
            .fields()
            .iter()
            .map(|f| (f.name().clone(), f.data_type().clone()))
            .collect()
    }

    fn col<'a>(batch: &'a RecordBatch, name: &str) -> &'a ArrayRef {
        batch.column_by_name(name).expect("column")
    }

    #[test]
    fn the_number_rule_types_each_path_from_its_literals() {
        let t = table(vec![("n", DataType::Int64), ("f", DataType::Float64)]);
        // Integral literals on a Float64 path are floats; on an unknown
        // path, Int64; a float literal is Float64 everywhere.
        let batch = rows_to_batch(
            &[
                json!({"n": 1, "f": 5, "g": 7, "h": 2.5, "s": "x", "b": true}),
                json!({"n": 2, "f": 6.5, "g": 8, "h": 3.0, "s": "y", "b": false}),
            ],
            &t,
        )
        .expect("map");
        assert_eq!(
            types(&batch),
            HashMap::from([
                ("n".to_string(), DataType::Int64),
                ("f".to_string(), DataType::Float64),
                ("g".to_string(), DataType::Int64),
                ("h".to_string(), DataType::Float64),
                ("s".to_string(), DataType::LargeUtf8),
                ("b".to_string(), DataType::Boolean),
            ])
        );
        assert_eq!(
            col(&batch, "f").as_primitive::<Float64Type>().values(),
            &[5.0, 6.5]
        );
        resolve_batch(&batch, &t, "_id").expect("stored");

        // A float literal on an Int64 path is typed Float64, and the
        // resolver refuses it; so is `5.0`, and a string on a numeric path.
        for row in [json!({"n": 1.5}), json!({"n": 5.0}), json!({"n": "42"})] {
            let batch = rows_to_batch(&[row], &t).expect("map");
            assert!(matches!(
                resolve_batch(&batch, &t, "_id"),
                Err(SchemaError::TypeMismatch { column, .. }) if column == "n"
            ));
        }
        // 2^53 + 1 is not exact in Float64, so it stays an integer and is
        // refused on a Float64 path; 2^53 is stored.
        let big = rows_to_batch(&[json!({"f": 9007199254740993i64})], &t).expect("map");
        assert_eq!(types(&big)["f"], DataType::Int64);
        assert!(resolve_batch(&big, &t, "_id").is_err());
        let exact = rows_to_batch(&[json!({"f": 9007199254740992i64})], &t).expect("map");
        assert_eq!(types(&exact)["f"], DataType::Float64);
    }

    #[test]
    fn a_narrower_integer_column_takes_literals_that_fit() {
        let t = table(vec![("small", DataType::Int32), ("name", DataType::Utf8)]);
        let batch = rows_to_batch(&[json!({"small": 7, "name": "a"})], &t).expect("map");
        assert_eq!(types(&batch)["small"], DataType::Int32);
        assert_eq!(types(&batch)["name"], DataType::Utf8);
        assert_eq!(
            col(&batch, "small").as_primitive::<Int32Type>().values(),
            &[7]
        );
        let wide = rows_to_batch(&[json!({"small": 5_000_000_000i64})], &t).expect("map");
        assert_eq!(types(&wide)["small"], DataType::Int64);
        assert!(resolve_batch(&wide, &t, "_id").is_err());
    }

    #[test]
    fn documents_flatten_to_dot_paths_and_arrays_become_lists() {
        let t = table(vec![]);
        let batch = rows_to_batch(
            &[
                json!({"user": {"name": "ann", "tags": ["a", "b"]}, "n": [1, 2], "m": [1, 2.5], "e": [], "z": null}),
                json!({"user": {"name": "bob"}, "n": [3], "items": [{"k": 1}, {"k": 2}]}),
            ],
            &t,
        )
        .expect("map");
        let list_of = |t: DataType| DataType::List(Arc::new(Field::new("item", t, true)));
        assert_eq!(
            types(&batch),
            HashMap::from([
                ("user.name".to_string(), DataType::LargeUtf8),
                ("user.tags".to_string(), list_of(DataType::LargeUtf8)),
                ("n".to_string(), list_of(DataType::Int64)),
                ("m".to_string(), list_of(DataType::Float64)),
                ("items.k".to_string(), list_of(DataType::Int64)),
            ])
        );
        assert_eq!(batch.num_rows(), 2);
        let n = col(&batch, "n").as_list::<i32>();
        assert_eq!(n.value(0).as_primitive::<Int64Type>().values(), &[1, 2]);
        assert_eq!(n.value(1).as_primitive::<Int64Type>().values(), &[3]);
        assert!(
            col(&batch, "user.tags").is_null(1),
            "the row without tags is null"
        );
        assert!(col(&batch, "items.k").is_null(0));

        assert!(matches!(
            rows_to_batch(&[json!({"x": [1, "a"]})], &t),
            Err(SchemaError::MixedArray { column, .. }) if column == "x"
        ));
        assert!(matches!(
            rows_to_batch(&[json!({"x": [[1], [2]]})], &t),
            Err(SchemaError::MixedArray { .. })
        ));
        assert!(matches!(
            rows_to_batch(&[json!({"x": 1}), json!({"x": [1]})], &t),
            Err(SchemaError::MixedArray { .. })
        ));
        assert!(matches!(
            rows_to_batch(&[json!({"x": 1}), json!({"x": "a"})], &t),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "x"
        ));
        assert!(matches!(
            rows_to_batch(&[json!([1, 2])], &t),
            Err(SchemaError::InvalidRow { row: 0, .. })
        ));
    }

    #[test]
    fn temporal_columns_take_the_literals_a_document_can_carry() {
        let t = table(vec![
            ("at", DataType::Timestamp(TimeUnit::Millisecond, None)),
            ("day", DataType::Date32),
        ]);
        let batch = rows_to_batch(
            &[
                json!({"at": "2026-09-22T10:00:00", "day": "2026-09-22"}),
                json!({"at": "2026-09-22T10:00:00Z", "day": "2026-09-23"}),
                json!({"at": 1_790_000_000_000_i64}),
            ],
            &t,
        )
        .expect("map");
        assert_eq!(
            types(&batch)["at"],
            DataType::Timestamp(TimeUnit::Millisecond, None)
        );
        assert_eq!(types(&batch)["day"], DataType::Date32);
        let at = col(&batch, "at").as_primitive::<arrow_array::types::TimestampMillisecondType>();
        assert_eq!(at.value(0), at.value(1), "a naive time reads as UTC");
        assert_eq!(at.value(2), 1_790_000_000_000);
        let day = col(&batch, "day").as_primitive::<arrow_array::types::Date32Type>();
        assert_eq!(day.value(1) - day.value(0), 1);
        assert!(col(&batch, "day").is_null(2));
        resolve_batch(&batch, &t, "_id").expect("stored");
        // A string that is not a time stays a string, so the resolver refuses it.
        let bad = rows_to_batch(&[json!({"at": "yesterday"})], &t).expect("map");
        assert_eq!(types(&bad)["at"], DataType::LargeUtf8);
        assert!(resolve_batch(&bad, &t, "_id").is_err());
    }

    #[test]
    fn depth_is_capped_and_an_empty_document_adds_nothing() {
        let t = table(vec![]);
        let deep = json!({"a": {"b": {"c": {"d": 1}}}});
        // Depth counts objects: the document is 1, `a` 2, `b` 3, `c` 4.
        let mut doc = t.to_json();
        doc["max_depth"] = json!(4);
        let four = TableSchema::from_json(&doc).expect("schema");
        assert!(rows_to_batch(std::slice::from_ref(&deep), &four).is_ok());
        doc["max_depth"] = json!(3);
        let three = TableSchema::from_json(&doc).expect("schema");
        assert!(matches!(
            rows_to_batch(&[deep], &three),
            Err(SchemaError::DepthExceeded { cap: 3, path }) if path == "a.b.c"
        ));
        let empty = rows_to_batch(&[json!({}), json!({"z": null, "e": []})], &t).expect("map");
        assert_eq!(empty.num_columns(), 0);
        assert_eq!(empty.num_rows(), 2);
    }

    #[test]
    fn templates_pin_types_and_attach_indexes_to_new_paths_only() {
        let mut doc = table(vec![("price", DataType::Int64)]).to_json();
        doc["templates"] = json!([
            {"name": "prices", "match": "integer", "path": "*_price", "type": "f64"},
            {"name": "text", "match": "string", "path": "body*", "index": {"kind": "fts"}},
            {"name": "when", "path": "*_at", "type": "timestamp_us", "tz": "UTC"},
        ]);
        let t = TableSchema::from_json(&doc).expect("schema");
        assert_eq!(t.templates().len(), 3);
        assert_eq!(t.templates()[0].data_type, Some(DataType::Float64));
        assert_eq!(t.templates()[0].matches, Some(Detected::Integer));
        assert_eq!(
            t.template_for("list_price", Detected::Integer)
                .map(|t| t.name.as_str()),
            Some("prices")
        );
        assert_eq!(
            t.templates()[1].index,
            Some(ColumnIndex::Fts {
                analyzer: "standard".into(),
                stopwords: Default::default(),
                stemmer: Default::default(),
                positions: false,
                stored: true,
                bm25: crate::superfile::fts::bm25::Bm25Params::STANDARD,
            })
        );
        let batch = rows_to_batch(
            &[json!({
                "list_price": 5,
                "price": 7,
                "body": "hello",
                "created_at": "2026-10-03T12:00:00Z",
                "updated_at": "not a time"
            })],
            &t,
        )
        .expect("map");
        let f = batch.schema();
        let field = |name: &str| f.field_with_name(name).expect("field");
        assert_eq!(
            field("list_price").data_type(),
            &DataType::Float64,
            "pinned by template"
        );
        assert_eq!(
            field("price").data_type(),
            &DataType::Int64,
            "the live column's type wins"
        );
        assert!(field("body").metadata().contains_key(INDEX_META_KEY));
        assert_eq!(
            field("created_at").data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))
        );
        assert_eq!(
            field("updated_at").data_type(),
            &DataType::LargeUtf8,
            "a string that is not a time stays a string, and the resolver decides"
        );
        let resolved = resolve_batch(&batch, &t, "_id").expect("resolve");
        let body = resolved
            .added
            .iter()
            .find(|a| a.name == "body")
            .expect("body");
        assert!(matches!(body.index, Some(ColumnIndex::Fts { .. })));
        // The first matching template wins, and a template only applies to
        // the kind it names.
        assert_eq!(
            t.template_for("body_text", Detected::String)
                .map(|t| t.name.as_str()),
            Some("text")
        );
        assert!(t.template_for("body_text", Detected::Integer).is_none());
        assert_eq!(
            t.template_for("x_at", Detected::Integer)
                .map(|t| t.name.as_str()),
            Some("when")
        );

        // A vector template: a numeric list of the declared length maps to
        // the vector type; another length does not.
        let mut doc = table(vec![]).to_json();
        doc["templates"] =
            json!([{"name": "vec", "match": "list", "path": "emb", "type": "vector", "dim": 2}]);
        let t = TableSchema::from_json(&doc).expect("schema");
        let batch = rows_to_batch(&[json!({"emb": [0.5, 1.5]})], &t).expect("map");
        assert!(matches!(
            types(&batch)["emb"],
            DataType::FixedSizeList(_, 2)
        ));
        let batch = rows_to_batch(&[json!({"emb": [0.5, 1.5, 2.5]})], &t).expect("map");
        assert!(matches!(types(&batch)["emb"], DataType::List(_)));
    }

    #[test]
    fn a_template_whose_index_does_not_fit_is_refused_when_the_column_joins() {
        let mut doc = table(vec![]).to_json();
        doc["templates"] = json!([{"name": "bad", "path": "n", "index": {"kind": "fts"}}]);
        let t = TableSchema::from_json(&doc).expect("schema");
        let batch = rows_to_batch(&[json!({"n": 1})], &t).expect("map");
        let resolved = resolve_batch(&batch, &t, "_id").expect("resolve");
        let changes: Vec<_> = resolved
            .added
            .into_iter()
            .map(
                |a| crate::supertable::schema::change::SchemaChange::AddColumn {
                    name: a.name,
                    data_type: a.data_type,
                    nullable: true,
                    index: a.index,
                },
            )
            .collect();
        assert!(matches!(
            t.apply(&changes),
            Err(SchemaError::InvalidIndex { column, .. }) if column == "n"
        ));
        let _ = Template {
            name: String::new(),
            matches: None,
            path: String::new(),
            data_type: None,
            index: None,
        };
    }
}
