// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Rows as JSON documents become one Arrow batch.
//!
//! Every document is flattened to dot paths and every path is given one
//! Arrow type from its values: `Boolean`, `Int64`, `Float64`, `LargeUtf8`,
//! or a `List` of those for arrays. The one exception is a path the table
//! already stores as a struct: flattening stops there and the object fills
//! the struct's declared fields, since `a.b` would otherwise become a
//! second column beside the `a` the caller meant to fill.
//!
//! The mapper infers and never checks:
//! the table's schema is consulted only to resolve an ambiguous literal
//! (an integral number on a `Float64` path is a float, a string on a
//! `Utf8` path is `Utf8`), so a batch that disagrees with the schema is
//! refused by the same resolver an Arrow producer meets, with the same

use std::{collections::HashMap, sync::Arc};

use arrow::{
    buffer::{NullBuffer, OffsetBuffer},
    datatypes::i256,
};
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Date64Array, Decimal128Array, Decimal256Array,
    FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeListArray, LargeStringArray, ListArray, RecordBatch, RecordBatchOptions, StringArray,
    StringViewArray, StructArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array, timezone::Tz,
};
use arrow_schema::{DataType, Field, Fields, Schema, TimeUnit};
use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};
use serde_json::{Map, Value};

use crate::supertable::schema::{TableSchema, error::SchemaError};

/// Integers above this magnitude are not exactly representable as `f64`.
const F64_EXACT_INT_BOUND: i128 = 1 << 53;
/// Integers above this magnitude are not exactly representable as `f32`.
const F32_EXACT_INT_BOUND: i128 = 1 << 24;
/// The most significant digits a `Decimal128` holds.
const DECIMAL128_DIGITS: u32 = 38;
/// The most significant digits a `Decimal256` holds.
const DECIMAL256_DIGITS: u32 = 76;
/// Milliseconds in a day, for `Date64`.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// A value a document carries on one path, with arrays kept whole. A list
/// position no value reached is `None`, which is how the leaves of an
/// array of objects stay the same length as the array.
#[derive(Debug, Clone)]
enum Leaf<'a> {
    Scalar(&'a Value),
    List(Vec<Option<&'a Value>>),
    /// A whole JSON object, kept unflattened because the path names a live
    /// column whose type is a struct: its keys are that struct's fields, not
    /// columns of their own.
    Object(&'a Value),
}

/// The kind of value a path carries, before a type is chosen for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Bool,
    Int,
    Float,
    Str,
    /// A JSON object filling a struct column. It never mixes with another
    /// kind and never infers a type of its own — only a column the table
    /// already declares as a struct takes one.
    Object,
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

    /// The type this kind infers on a path the table does not have.
    fn inferred(self) -> DataType {
        match self {
            Kind::Bool => DataType::Boolean,
            Kind::Int => DataType::Int64,
            Kind::Float => DataType::Float64,
            Kind::Str => DataType::LargeUtf8,
            // Unreachable in practice: a path only takes this kind when the
            // table already names it a struct, and then the column's own type
            // is used. Inferring a struct's shape from a document would make
            // the fields of one row the schema for the rest.
            Kind::Object => DataType::Struct(Fields::empty()),
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
    /// `(row, leaf)` for the rows that carry this path, in ascending row
    /// order. Sparse on purpose: a dense cell per row per column makes the
    /// memory the product of the two, so a body of many rows each carrying
    /// a different key costs rows x columns cells to hold a handful of
    /// values. The field cap bounds the columns, not that product.
    cells: Vec<(usize, Leaf<'a>)>,
    /// The distinct kinds of the scalars seen (or of the list elements
    /// seen), in order of appearance.
    kinds: Vec<Kind>,
    /// Whether any value was an array.
    list: bool,
}

impl<'a> Column<'a> {
    /// The column over `rows` rows: the leaf each row carries, `None` where
    /// it carries none. One pass over the sparse cells, no materialised
    /// dense vector.
    fn by_row(&self, rows: usize) -> impl Iterator<Item = Option<&Leaf<'a>>> + '_ {
        let mut next = 0usize;
        (0..rows).map(move |row| match self.cells.get(next) {
            Some((at, leaf)) if *at == row => {
                next += 1;
                Some(leaf)
            }
            _ => None,
        })
    }

    /// Every scalar value the column carries, lists flattened.
    fn values(&self) -> impl Iterator<Item = &Value> + '_ {
        self.cells
            .iter()
            .map(|(_, leaf)| leaf)
            .flat_map(|leaf| match leaf {
                Leaf::Scalar(v) => vec![*v],
                Leaf::List(vs) => vs.iter().flatten().copied().collect(),
                // A struct's own values are read per child field when the
                // array is built, not through the column's scalars.
                Leaf::Object(_) => Vec::new(),
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
                    && self
                        .values()
                        .all(|v| parses_as_time_in(v, timestamp_zone(target))) =>
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

/// The live column a flattened `path` nests under, if any: `a.b.c` checks
/// `a` and `a.b`. Documents flatten, so such a path cannot fill the column
/// it nests under.
fn live_prefix_of(path: &str, schema: &TableSchema) -> Option<String> {
    let mut at = 0;
    while let Some(dot) = path[at..].find('.') {
        let end = at + dot;
        let prefix = &path[..end];
        if schema.id_of(prefix).is_some() {
            return Some(prefix.to_owned());
        }
        at = end + 1;
    }
    None
}

/// Map `rows` to one batch under `schema`'s rules. Every row is a JSON
/// object; see the module docs for how paths are typed.
pub fn rows_to_batch(rows: &[Value], schema: &TableSchema) -> Result<RecordBatch, SchemaError> {
    let mut columns: Vec<Column<'_>> = Vec::new();
    let mut by_path: HashMap<String, usize> = HashMap::new();
    // Each path the table does not have is a field the resolver would add,
    // and each costs one cell per row right here. The cap that bounds those
    // fields is therefore checked as the paths are discovered: a body with
    // more distinct keys than the table admits is refused before its
    // columns exist, not after they have been allocated.
    let live_fields = schema.fields().len() as u32;
    // Whether `path` names a live column stored as a struct, which is what
    // stops `flatten` descending into the object that fills it.
    let is_struct_column = |path: &str| {
        schema
            .id_of(path)
            .and_then(|id| schema.fields().iter().find(|f| f.id == id))
            .is_some_and(|f| matches!(f.data_type, DataType::Struct(_)))
    };
    let mut new_paths: Vec<String> = Vec::new();
    for (row, value) in rows.iter().enumerate() {
        let object = value.as_object().ok_or_else(|| SchemaError::InvalidRow {
            row,
            reason: "a row is a JSON object".to_owned(),
        })?;
        let mut leaves = Vec::new();
        flatten(
            object,
            "",
            1,
            schema.max_depth(),
            &mut leaves,
            &is_struct_column,
        )?;
        for (path, leaf) in leaves {
            let index = match by_path.get(&path) {
                Some(&i) => i,
                None => {
                    if schema.id_of(&path).is_none() {
                        // A path nesting under a column the table already
                        // has would become a second column beside it rather
                        // than filling it: the declared column would stay
                        // null while the value sat in one whose name has to
                        // be quoted to be read. Refuse instead of writing
                        // the value where the caller is not looking.
                        if let Some(shadowed) = live_prefix_of(&path, schema) {
                            return Err(SchemaError::PathShadowsColumn {
                                path,
                                column: shadowed,
                            });
                        }
                        new_paths.push(path.clone());
                        if live_fields + new_paths.len() as u32 > schema.max_fields() {
                            return Err(SchemaError::FieldCapExceeded {
                                cap: schema.max_fields(),
                                current: live_fields,
                                fields: new_paths,
                            });
                        }
                    }
                    columns.push(Column {
                        path: path.clone(),
                        cells: Vec::new(),
                        kinds: Vec::new(),
                        list: false,
                    });
                    by_path.insert(path, columns.len() - 1);
                    columns.len() - 1
                }
            };
            let column = &mut columns[index];
            // Cells land in row order, so a repeat of this row is the last
            // one pushed.
            if column.cells.last().is_some_and(|(at, _)| *at == row) {
                // Two keys of one document flattened to one path — a
                // literal `a.b` beside a nested `a: {b: …}`. Keeping
                // either would drop the other silently.
                return Err(SchemaError::InvalidRow {
                    row,
                    reason: format!("two keys flatten to the same path `{}`", column.path),
                });
            }
            observe(column, &leaf)?;
            column.cells.push((row, leaf));
        }
    }

    let mut fields = Vec::with_capacity(columns.len());
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(columns.len());
    for column in &columns {
        let metadata = HashMap::new();
        // A path the table already has is written in that column's type; a
        // path it does not is inferred from the values.
        let target = schema.id_of(&column.path).and_then(|id| {
            schema
                .fields()
                .iter()
                .find(|f| f.id == id)
                .map(|f| f.data_type.clone())
        });
        if column.kinds.is_empty() {
            // Only empty arrays. On a list column the table already has,
            // that is a row of empty lists and the column's own type says
            // what they are lists of. On a path the table does not have,
            // there is no element type to infer and nothing is created.
            match &target {
                Some(list @ (DataType::List(_) | DataType::LargeList(_))) if column.list => {
                    arrays.push(build_array(column, rows.len(), list)?);
                    fields
                        .push(Field::new(&column.path, list.clone(), true).with_metadata(metadata));
                }
                _ => {}
            }
            continue;
        }
        let kind = column.settle_kind(target.as_ref())?;
        let data_type = resolve_type(column, kind, target.as_ref());
        // A path carrying both integers and floats is a float column, and
        // an integer past f64's exact range would be rounded on the way in.
        // The declared-target arms of `resolve_type` already refuse that by
        // falling through to the kind's own type; a column the documents
        // are creating has no target to fall back to, so refuse here rather
        // than store a number the caller never sent.
        if matches!(element_of(&data_type), DataType::Float64)
            && let Some(value) = column.values().find_map(inexact_in_f64)
        {
            return Err(SchemaError::IntegerNotExactInFloat {
                column: column.path.clone(),
                value,
                stored: value as f64,
            });
        }
        let array = build_array(column, rows.len(), &data_type)?;
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
    if let Leaf::Object(_) = leaf {
        if column.list || column.kinds.iter().any(|k| *k != Kind::Object) {
            return Err(SchemaError::MixedArray {
                column: column.path.clone(),
                types: vec!["object".to_owned(), "scalar".to_owned()],
            });
        }
        if !column.kinds.contains(&Kind::Object) {
            column.kinds.push(Kind::Object);
        }
        return Ok(());
    }
    if column.kinds.contains(&Kind::Object) {
        return Err(SchemaError::MixedArray {
            column: column.path.clone(),
            types: vec!["object".to_owned(), "scalar".to_owned()],
        });
    }
    let (values, list): (Vec<&Value>, bool) = match leaf {
        Leaf::Scalar(v) => (vec![*v], false),
        Leaf::List(vs) => (vs.iter().flatten().copied().collect(), true),
        Leaf::Object(_) => unreachable!("handled above"),
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
    structs: &dyn Fn(&str) -> bool,
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
                // A column the table declares a struct takes the object
                // whole. Flattening past it would make `image.url` a second
                // column beside `image` and leave the declared one null,
                // which is what `live_prefix_of` refuses.
                if structs(&path) {
                    out.push((path, Leaf::Object(value)));
                    continue;
                }
                if depth + 1 > max_depth {
                    return Err(SchemaError::DepthExceeded {
                        cap: max_depth,
                        path,
                    });
                }
                flatten(inner, &path, depth + 1, max_depth, out, structs)?;
            }
            Value::Array(items) => {
                if items.is_empty() {
                    // An empty list, which is not the same as no list: a
                    // column whose only values were `[]` carries no kind to
                    // type it from and is dropped in `rows_to_batch`, but a
                    // column the table already has stores the empty list the
                    // caller sent rather than a null.
                    out.push((path, Leaf::List(Vec::new())));
                    continue;
                }
                if items.iter().all(Value::is_object) {
                    // An array of objects: one list per leaf path, with the
                    // elements' values in order. A leaf an element does not
                    // carry takes a null in that element's place, so reading
                    // one position across the leaves reads one element of
                    // the array; packing the values instead would make the
                    // lists disagree about what a position means as soon as
                    // two elements carried different keys.
                    let mut per_path: Vec<(String, Vec<Option<&'a Value>>)> = Vec::new();
                    for (element, item) in items.iter().enumerate() {
                        let mut leaves = Vec::new();
                        let inner = item.as_object().expect("every item is an object");
                        if depth + 1 > max_depth {
                            return Err(SchemaError::DepthExceeded {
                                cap: max_depth,
                                path,
                            });
                        }
                        flatten(inner, &path, depth + 1, max_depth, &mut leaves, structs)?;
                        let before: Vec<usize> = per_path.iter().map(|(_, v)| v.len()).collect();
                        for (leaf_path, leaf) in leaves {
                            let values = match leaf {
                                Leaf::Scalar(v) => vec![Some(v)],
                                // One position per element is the invariant
                                // that lets a position name the same element
                                // in every leaf. A nested array wants several
                                // positions for one element, which would make
                                // the leaves disagree about what a position
                                // means, so it is refused rather than stored
                                // misaligned.
                                // A struct inside an array of objects is a
                                // position problem of the same kind.
                                Leaf::List(_) | Leaf::Object(_) => {
                                    return Err(SchemaError::NestedArray { path: leaf_path });
                                }
                            };
                            match per_path.iter_mut().find(|(p, _)| *p == leaf_path) {
                                Some((_, all)) => all.extend(values),
                                None => {
                                    // A path first seen on a later element
                                    // is null in the elements before it.
                                    let mut all = vec![None; element];
                                    all.extend(values);
                                    per_path.push((leaf_path, all));
                                }
                            }
                        }
                        for ((_, all), filled) in per_path.iter_mut().zip(before) {
                            if all.len() == filled {
                                all.push(None);
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
                            // A position, not an absence: dropping it would
                            // shift every element after it one place left,
                            // and the caller would read back a shorter list
                            // than they sent.
                            Value::Null => values.push(None),
                            Value::Array(_) | Value::Object(_) => {
                                return Err(SchemaError::MixedArray {
                                    column: path,
                                    types: vec![json_kind_name(item).to_owned()],
                                });
                            }
                            scalar => values.push(Some(scalar)),
                        }
                    }
                    out.push((path, Leaf::List(values)));
                }
            }
            scalar => out.push((path, Leaf::Scalar(scalar))),
        }
    }
    Ok(())
}

/// The Arrow type for `column`: `target` (the table's
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
                && column
                    .cells
                    .iter()
                    .map(|(_, leaf)| leaf)
                    .all(|leaf| match leaf {
                        Leaf::List(values) => values.len() == *dim as usize,
                        Leaf::Scalar(_) | Leaf::Object(_) => false,
                    })
                && column.values().all(float_in_f32_range);
            if fits {
                return DataType::FixedSizeList(Arc::clone(item), *dim);
            }
            None
        }
        Some(other) if !column.list => Some(other),
        _ => None,
    };
    // A struct column's type is the table's: its fields are declared, never
    // inferred from a document.
    if kind == Kind::Object
        && let Some(stored @ DataType::Struct(_)) = target
    {
        return stored.clone();
    }
    let values = || column.values();
    let element = match (kind, scalar_target) {
        // `Kind::Float` and `Kind::Str` reach an integer column too: `1.0` and
        // `"42"` name integers, and `int_fits` reads all three spellings.
        (Kind::Int | Kind::Float | Kind::Str, Some(t))
            if is_integer_type(t) && values().all(|v| int_fits(v, t)) =>
        {
            t.clone()
        }
        (Kind::Int, Some(DataType::Float64)) if values().all(int_exact_in_f64) => DataType::Float64,
        // A float literal into an `f32` column is a request to round: that is
        // what the narrower column is for, and refusing `0.1` because it is
        // not a dyadic rational would leave the column unusable from JSON.
        // Only a value narrowing destroys — one that saturates to an infinity
        // or collapses to zero — is refused, by `float_in_f32_range`.
        (Kind::Float, Some(DataType::Float32)) if values().all(float_in_f32_range) => {
            DataType::Float32
        }
        // An integer is held to the stricter bar, as it is on `Float64`: a
        // literal past the mantissa is a different integer once stored, not a
        // coarser one, and the caller asked for the integer they wrote.
        (Kind::Int, Some(DataType::Float32)) if values().all(int_exact_in_f32) => DataType::Float32,
        (Kind::Str, Some(t @ (DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View))) => {
            t.clone()
        }
        (Kind::Int | Kind::Float | Kind::Str, Some(t @ DataType::Decimal128(precision, scale)))
            if values().all(|v| decimal_fits(v, *precision, *scale, DECIMAL128_DIGITS)) =>
        {
            t.clone()
        }
        (Kind::Int | Kind::Float | Kind::Str, Some(t @ DataType::Decimal256(precision, scale)))
            if values().all(|v| decimal_fits(v, *precision, *scale, DECIMAL256_DIGITS)) =>
        {
            t.clone()
        }
        // After the text arm above, so a number in a text column stays text.
        (Kind::Str, Some(DataType::Float64)) if values().all(|v| as_number(v).is_some()) => {
            DataType::Float64
        }
        (Kind::Str, Some(DataType::Float32)) if values().all(float_in_f32_range) => {
            DataType::Float32
        }
        (Kind::Str, Some(t @ DataType::Timestamp(_, _)))
            if values().all(|v| parses_as_time_in(v, timestamp_zone(Some(t)))) =>
        {
            t.clone()
        }
        (Kind::Int, Some(t @ DataType::Timestamp(_, _)))
            if values().all(|v| parses_as_time_in(v, timestamp_zone(Some(t)))) =>
        {
            t.clone()
        }
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

/// The integer `value` names: an integer literal, a float literal with no
/// fractional part, or a string spelling one.
///
/// JSON has one number type, so a producer that round-trips through a double
/// writes `1.0` where it means `1`, and one avoiding a double's precision
/// loss writes `"9007199254740993"`. Both named the integer they wrote. A
/// float past the mantissa is not an exact integer at all and is left to the
/// float path.
fn as_integer(value: &Value) -> Option<i128> {
    if let Some(n) = as_i128(value) {
        return Some(n);
    }
    if let Some(text) = value.as_str() {
        return text.trim().parse::<i128>().ok();
    }
    let float = value.as_f64()?;
    (float.fract() == 0.0 && float.abs() <= F64_EXACT_INT_BOUND as f64).then_some(float as i128)
}

/// The number `value` names: a numeric literal, or a string spelling one.
fn as_number(value: &Value) -> Option<f64> {
    match value.as_f64() {
        Some(float) => Some(float),
        None => value.as_str()?.trim().parse::<f64>().ok(),
    }
}

/// The unscaled integer `value` names in a decimal column of `scale`, as a
/// decimal string: `12.34` at scale 2 is `1234`. `None` when the value is not
/// a number, needs more fractional digits than the column keeps, or does not
/// fit `precision`.
///
/// Read from the decimal *text*, never through an `f64`: `12.34` has no exact
/// binary form, so scaling it as a float yields `1233.9999…` and truncates to
/// `1233`. `serde_json` preserves the literal the caller wrote, which is the
/// only place the exact digits still exist.
fn decimal_unscaled(value: &Value, precision: u8, scale: i8, max_digits: u32) -> Option<i256> {
    let text = match value {
        Value::String(text) => text.trim().to_owned(),
        Value::Number(number) => number.to_string(),
        _ => return None,
    };
    let (sign, digits) = match text.strip_prefix('-') {
        Some(rest) => (-1i32, rest),
        None => (1i32, text.strip_prefix('+').unwrap_or(&text)),
    };
    // An exponent would have to be folded into the scale before the digits
    // line up; a caller that wants one can send the expanded form.
    if digits.is_empty() || digits.contains(['e', 'E']) {
        return None;
    }
    let (whole, fraction) = match digits.split_once('.') {
        Some((whole, fraction)) => (whole, fraction),
        None => (digits, ""),
    };
    if !whole
        .bytes()
        .chain(fraction.bytes())
        .all(|b| b.is_ascii_digit())
        || whole.is_empty() && fraction.is_empty()
    {
        return None;
    }
    // Rounding is the caller's to do: a value with more fractional digits than
    // the column keeps would be stored as a different number, silently.
    let scale_digits = usize::try_from(scale).ok()?;
    if fraction.len() > scale_digits {
        return None;
    }
    let padded = format!(
        "{whole}{fraction}{}",
        "0".repeat(scale_digits - fraction.len())
    );
    if padded.len() as u32 > max_digits {
        return None;
    }
    let unscaled = i256::from_string(&padded)?;
    let unscaled = if sign < 0 {
        unscaled.checked_neg()?
    } else {
        unscaled
    };
    // `precision` is the digit count the column declares; a value needing more
    // is out of range for it.
    let significant = padded.trim_start_matches('0').len();
    if significant > usize::from(precision) {
        return None;
    }
    Some(unscaled)
}

fn decimal_fits(value: &Value, precision: u8, scale: i8, max_digits: u32) -> bool {
    decimal_unscaled(value, precision, scale, max_digits).is_some()
}

fn int_fits(value: &Value, t: &DataType) -> bool {
    let Some(v) = as_integer(value) else {
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
    as_integer(value).is_some_and(|v| v.abs() <= F64_EXACT_INT_BOUND)
}

/// The integer `value` holds when it is one `f64` cannot carry exactly.
/// A float literal is not an integer and is left alone: it arrived as an
/// `f64` and is stored as the one it arrived as.
fn inexact_in_f64(value: &Value) -> Option<i128> {
    as_integer(value).filter(|v| v.abs() > F64_EXACT_INT_BOUND)
}

/// The element type of a list type, or the type itself when it is not a
/// list: the type one of the column's values is actually stored as.
fn element_of(data_type: &DataType) -> &DataType {
    match data_type {
        DataType::List(item) | DataType::LargeList(item) | DataType::FixedSizeList(item, _) => {
            item.data_type()
        }
        other => other,
    }
}

/// Whether the integer `value` holds is one `f32` carries exactly. The
/// mantissa is 24 bits, so this is the `f32` twin of [`int_exact_in_f64`].
fn int_exact_in_f32(value: &Value) -> bool {
    as_integer(value).is_some_and(|v| v.abs() <= F32_EXACT_INT_BOUND)
}

/// Whether `value` survives narrowing to `f32`. Rounding a coordinate to
/// `f32` precision is what a `Float32` column is for, but a magnitude
/// `f32` cannot reach is not rounded: it saturates to an infinity, or a
/// nonzero value collapses to zero. Either one is a different number
/// rather than a coarser one, so the column does not take it.
fn float_in_f32_range(value: &Value) -> bool {
    as_number(value).is_some_and(|v| {
        let narrowed = v as f32;
        narrowed.is_finite() && (narrowed != 0.0 || v == 0.0)
    })
}

/// The naive forms a timestamp string may take, in the order they are
/// tried. A naive string names a wall clock with no zone of its own, so the
/// column's zone decides the instant it marks (see [`parse_time_in`]).
///
/// The space-separated form is what Python's `datetime.isoformat(sep=" ")`,
/// pandas and most SQL shells print, and the date-only form is a day at
/// midnight.
const NAIVE_TIME_FORMATS: [&str; 2] = ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"];

/// A string as a point in time for a column in `tz`.
///
/// A string carrying its own offset (RFC 3339) names an instant outright and
/// the column's zone does not enter into it. A naive one names a wall clock,
/// and the instant it marks depends on where that clock is: `10:30` on a
/// `America/New_York` column is 10:30 in New York, not 10:30 UTC. Reading it
/// as UTC stores an instant hours from the one the caller wrote, and says
/// nothing about it — the write succeeds.
///
/// `None` for a zone the build cannot resolve, rather than a silent fallback
/// to UTC, which is the error this exists to prevent.
fn parse_time_in(s: &str, tz: Option<&str>) -> Option<DateTime<Utc>> {
    if let Ok(fixed) = DateTime::parse_from_rfc3339(s) {
        return Some(fixed.with_timezone(&Utc));
    }
    let naive = NAIVE_TIME_FORMATS
        .iter()
        .find_map(|format| NaiveDateTime::parse_from_str(s, format).ok())
        .or_else(|| parse_date(s).and_then(|d| d.and_hms_opt(0, 0, 0)))?;
    match tz {
        None => Some(naive.and_utc()),
        Some(name) => {
            let zone: Tz = name.parse().ok()?;
            // A wall clock a zone skips (the spring-forward gap) or repeats
            // (the autumn fall-back) names no single instant. Take the
            // earliest the zone offers rather than refuse the row: the
            // alternative is a write that fails twice a year on data that
            // round-tripped the rest of it.
            zone.from_local_datetime(&naive)
                .earliest()
                .map(|t| t.with_timezone(&Utc))
        }
    }
}

/// The zone a timestamp type carries, if it is a timestamp and is zoned.
fn timestamp_zone(data_type: Option<&DataType>) -> Option<&str> {
    match data_type {
        Some(DataType::Timestamp(_, tz)) => tz.as_deref(),
        _ => None,
    }
}

/// Whether `value` names a point in time a column in `tz` can hold: a time
/// string, or an epoch count. An integer past `i64` names no instant, and a
/// column that accepted it would hold a null where the literal was.
fn parses_as_time_in(value: &Value, tz: Option<&str>) -> bool {
    match as_i128(value) {
        Some(n) => i64::try_from(n).is_ok(),
        None => value
            .as_str()
            .is_some_and(|s| parse_time_in(s, tz).is_some()),
    }
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
/// the path. The type is chosen from these values, so they fit it; a value
/// that does not is refused rather than stored as a null, since a null is
/// what the caller reads back for a key it never sent.
fn build_array(
    column: &Column<'_>,
    rows: usize,
    data_type: &DataType,
) -> Result<ArrayRef, SchemaError> {
    match data_type {
        DataType::List(item) | DataType::LargeList(item) => {
            let mut offsets: Vec<usize> = Vec::with_capacity(rows + 1);
            let mut nulls = Vec::with_capacity(rows);
            let mut flat: Vec<Option<&Value>> = Vec::new();
            offsets.push(0);
            for cell in column.by_row(rows) {
                match cell {
                    Some(Leaf::List(values)) => {
                        flat.extend(values.iter().copied());
                        nulls.push(true);
                    }
                    _ => nulls.push(false),
                }
                offsets.push(flat.len());
            }
            let child = build_scalar_array(&column.path, &flat, item.data_type())?;
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
            let mut nulls = Vec::with_capacity(rows);
            for cell in column.by_row(rows) {
                match cell {
                    Some(Leaf::List(values)) => {
                        flat.extend(values.iter().copied());
                        nulls.push(true);
                    }
                    _ => {
                        flat.extend(std::iter::repeat_n(None, *dim as usize));
                        nulls.push(false);
                    }
                }
            }
            let child = build_scalar_array(&column.path, &flat, item.data_type())?;
            Ok(Arc::new(FixedSizeListArray::new(
                Arc::clone(item),
                *dim,
                child,
                Some(NullBuffer::from(nulls)),
            )))
        }
        DataType::Struct(fields) => {
            let values: Vec<Option<&Value>> = column
                .by_row(rows)
                .map(|cell| match cell {
                    Some(Leaf::Object(v)) => Some(*v),
                    _ => None,
                })
                .collect();
            build_struct_array(&column.path, &values, fields)
        }
        scalar => {
            let values: Vec<Option<&Value>> = column
                .by_row(rows)
                .map(|cell| match cell {
                    Some(Leaf::Scalar(v)) => Some(*v),
                    _ => None,
                })
                .collect();
            build_scalar_array(&column.path, &values, scalar)
        }
    }
}

/// The struct column `fields` declares, read from one JSON object per row.
///
/// A field the object does not carry is null, and a key the struct does not
/// declare is refused: a document cannot add a field to a declared struct,
/// the way a top-level key adds a column, because the struct's shape is the
/// table's and a silent drop would lose the value.
fn build_struct_array(
    path: &str,
    values: &[Option<&Value>],
    fields: &Fields,
) -> Result<ArrayRef, SchemaError> {
    let mut nulls = Vec::with_capacity(values.len());
    let mut objects: Vec<Option<&Map<String, Value>>> = Vec::with_capacity(values.len());
    for value in values {
        match value.and_then(|v| v.as_object()) {
            Some(object) => {
                for key in object.keys() {
                    if !fields.iter().any(|f| f.name() == key) {
                        return Err(SchemaError::UnknownStructField {
                            column: path.to_owned(),
                            field: key.clone(),
                        });
                    }
                }
                nulls.push(true);
                objects.push(Some(object));
            }
            None => {
                nulls.push(false);
                objects.push(None);
            }
        }
    }
    let mut children: Vec<ArrayRef> = Vec::with_capacity(fields.len());
    for field in fields {
        let child_path = format!("{path}.{}", field.name());
        let child: Vec<Option<&Value>> = objects
            .iter()
            .map(|object| match object.and_then(|o| o.get(field.name())) {
                // A key carrying `null` and a key absent are the same thing:
                // the field has no value in this row.
                Some(Value::Null) | None => None,
                Some(value) => Some(value),
            })
            .collect();
        children.push(match field.data_type() {
            DataType::Struct(inner) => build_struct_array(&child_path, &child, inner)?,
            other => build_scalar_array(&child_path, &child, other)?,
        });
    }
    Ok(Arc::new(StructArray::new(
        fields.clone(),
        children,
        Some(NullBuffer::from(nulls)),
    )))
}

fn build_scalar_array(
    column: &str,
    values: &[Option<&Value>],
    data_type: &DataType,
) -> Result<ArrayRef, SchemaError> {
    // A value outside the column's range is refused, not narrowed to a null:
    // the caller sent a number, and reading back a null would say they sent
    // nothing. `None` stays `None`, which is the key a row did not carry.
    macro_rules! ints {
        ($array:ty, $native:ty) => {{
            let mut out: Vec<Option<$native>> = Vec::with_capacity(values.len());
            for value in values {
                match value.and_then(as_integer) {
                    None => out.push(None),
                    Some(n) => match <$native>::try_from(n) {
                        Ok(n) => out.push(Some(n)),
                        Err(_) => {
                            return Err(SchemaError::ValueOutOfRange {
                                column: column.to_owned(),
                                value: n.to_string(),
                                data_type: data_type.to_string(),
                            });
                        }
                    },
                }
            }
            Arc::new(<$array>::from(out)) as ArrayRef
        }};
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
                .map(|v| (*v).and_then(as_number).map(|f| f as f32))
                .collect::<Vec<_>>(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            values
                .iter()
                .map(|v| (*v).and_then(as_number))
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
        DataType::Decimal128(precision, scale) => {
            let mut out: Vec<Option<i128>> = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    None => out.push(None),
                    Some(value) => {
                        let unscaled =
                            decimal_unscaled(value, *precision, *scale, DECIMAL128_DIGITS)
                                .and_then(i256::to_i128)
                                .ok_or_else(|| SchemaError::ValueOutOfRange {
                                    column: column.to_owned(),
                                    value: value.to_string(),
                                    data_type: data_type.to_string(),
                                })?;
                        out.push(Some(unscaled));
                    }
                }
            }
            Arc::new(
                Decimal128Array::from(out)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| SchemaError::InvalidRow {
                        row: 0,
                        reason: e.to_string(),
                    })?,
            )
        }
        DataType::Decimal256(precision, scale) => {
            let mut out: Vec<Option<i256>> = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    None => out.push(None),
                    Some(value) => {
                        let unscaled =
                            decimal_unscaled(value, *precision, *scale, DECIMAL256_DIGITS)
                                .ok_or_else(|| SchemaError::ValueOutOfRange {
                                    column: column.to_owned(),
                                    value: value.to_string(),
                                    data_type: data_type.to_string(),
                                })?;
                        out.push(Some(unscaled));
                    }
                }
            }
            Arc::new(
                Decimal256Array::from(out)
                    .with_precision_and_scale(*precision, *scale)
                    .map_err(|e| SchemaError::InvalidRow {
                        row: 0,
                        reason: e.to_string(),
                    })?,
            )
        }
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
                    let t = parse_time_in(v.as_str()?, tz.as_deref())?;
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
        Array,
        cast::AsArray,
        types::{
            Decimal128Type, Float32Type, Float64Type, Int32Type, Int64Type,
            TimestampMillisecondType,
        },
    };
    use serde_json::json;

    use super::*;
    use crate::supertable::schema::resolve::resolve_batch;

    fn table(fields: Vec<(&str, DataType)>) -> TableSchema {
        TableSchema::from_user_schema(&Schema::new(
            fields
                .into_iter()
                .map(|(n, t)| Field::new(n, t, true))
                .collect::<Vec<_>>(),
        ))
    }

    /// A path carrying an integer past f64's exact range and a float would
    /// store the integer rounded. The declared-target arms already refuse
    /// this; a column the documents create has no target, so it is refused
    /// here. Without the guard the batch holds 9007199254740992.
    #[test]
    fn an_integer_a_float_column_cannot_hold_is_refused() {
        let big = (1i64 << 53) + 1;
        let err = rows_to_batch(&[json!({"n": big}), json!({"n": 1.5})], &table(Vec::new()))
            .expect_err("the integer would be rounded");
        assert!(
            matches!(
                &err,
                SchemaError::IntegerNotExactInFloat { column, value, .. }
                    if column == "n" && *value == big as i128
            ),
            "{err:?}"
        );
    }

    /// The guard is about the integer, not about floats: a column of plain
    /// integers keeps Int64 and its full range, and a float column of
    /// floats is untouched.
    #[test]
    fn integers_alone_keep_their_range_and_floats_alone_are_fine() {
        let big = i64::MAX;
        let ints = rows_to_batch(&[json!({"n": big})], &table(Vec::new())).expect("ints map");
        assert_eq!(types(&ints)["n"], DataType::Int64);
        assert_eq!(col(&ints, "n").as_primitive::<Int64Type>().value(0), big);

        let floats = rows_to_batch(&[json!({"f": 1.5}), json!({"f": 2.5})], &table(Vec::new()))
            .expect("floats map");
        assert_eq!(types(&floats)["f"], DataType::Float64);
    }

    /// An integer inside f64's exact range still joins a float column.
    #[test]
    fn an_exact_integer_still_joins_a_float_column() {
        let batch = rows_to_batch(&[json!({"n": 3}), json!({"n": 1.5})], &table(Vec::new()))
            .expect("an exact integer is fine beside a float");
        assert_eq!(types(&batch)["n"], DataType::Float64);
    }

    /// A number past the column's range is refused. It used to be narrowed
    /// to a null, so `append_rows([{"n": u64::MAX}])` returned `Ok` and `n`
    /// read back as though the row had never carried it.
    #[test]
    fn a_number_the_column_cannot_hold_is_refused_not_nulled() {
        let past_i64 = json!(u64::MAX);
        let err = rows_to_batch(&[json!({"n": past_i64})], &table(Vec::new()))
            .expect_err("u64::MAX does not fit the Int64 the values infer");
        assert!(
            matches!(
                &err,
                SchemaError::ValueOutOfRange { column, value, .. }
                    if column == "n" && value == &u64::MAX.to_string()
            ),
            "{err:?}"
        );
    }

    /// A key a row simply does not carry is still a null, which is the one
    /// thing the refusal above must not swallow.
    #[test]
    fn a_missing_key_is_still_null() {
        let batch = rows_to_batch(&[json!({"n": 1}), json!({"other": 2})], &table(Vec::new()))
            .expect("a row that omits a key maps");
        let n = col(&batch, "n").as_primitive::<Int64Type>();
        assert_eq!(n.value(0), 1);
        assert!(n.is_null(1), "the row that omitted `n` reads null");
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

        // A fractional literal on an Int64 path is typed Float64, and the
        // resolver refuses it.
        let batch = rows_to_batch(&[json!({"n": 1.5})], &t).expect("map");
        assert!(matches!(
            resolve_batch(&batch, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "n"
        ));
        // `5.0` and `"42"` name the integer 5 and the integer 42: JSON has
        // one number type, so a producer that round-trips through a double
        // writes the first and one avoiding a double's precision loss writes
        // the second. Both are stored as the integer they name.
        for row in [json!({"n": 5.0}), json!({"n": "42"})] {
            let batch = rows_to_batch(&[row], &t).expect("map");
            assert_eq!(types(&batch)["n"], DataType::Int64);
            resolve_batch(&batch, &t, "_id").expect("stored");
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
    fn a_literal_a_narrow_column_cannot_hold_keeps_its_own_type() {
        // A narrow type is chosen only when every value fits it, so a
        // literal that does not fit leaves the path at the type it infers
        // and the resolver refuses the batch. It is never quietly nulled,
        // saturated or rounded into the column.
        let t = table(vec![("tiny", DataType::Int8), ("f", DataType::Float32)]);
        let batch = rows_to_batch(&[json!({"tiny": 999})], &t).expect("map");
        assert_eq!(types(&batch)["tiny"], DataType::Int64);
        assert!(matches!(
            resolve_batch(&batch, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "tiny"
        ));
        let batch = rows_to_batch(&[json!({"f": 1e300})], &t).expect("map");
        assert_eq!(types(&batch)["f"], DataType::Float64);
        assert!(matches!(
            resolve_batch(&batch, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "f"
        ));
    }

    #[test]
    fn a_magnitude_a_vector_or_a_timestamp_cannot_hold_is_refused() {
        let vector =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 2);
        let t = table(vec![("emb", vector.clone())]);
        // A coordinate `f32` cannot reach would be stored as an infinity,
        // or flushed to zero, so the vector type does not take the list.
        for row in [json!({"emb": [1e300, 1.0]}), json!({"emb": [1e-300, 1.0]})] {
            let batch = rows_to_batch(&[row], &t).expect("map");
            assert!(matches!(types(&batch)["emb"], DataType::List(_)));
            assert!(matches!(
                resolve_batch(&batch, &t, "_id"),
                Err(SchemaError::TypeMismatch { column, .. }) if column == "emb"
            ));
        }
        // A coordinate `f32` holds less precisely is still that coordinate.
        let batch = rows_to_batch(&[json!({"emb": [0.1, 0.2]})], &t).expect("map");
        assert_eq!(types(&batch)["emb"], vector);
        resolve_batch(&batch, &t, "_id").expect("stored");

        // An integral literal on a timestamp is an epoch count in the
        // column's unit; one past `i64` names no instant and must not land
        // as a null.
        let t = table(vec![(
            "at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
        )]);
        assert!(matches!(
            rows_to_batch(&[json!({"at": u64::MAX})], &t),
            Err(SchemaError::ValueOutOfRange { column, .. }) if column == "at"
        ));
        // Beside a time string it is refused outright, rather than joining
        // the string's column and reading back as a null.
        assert!(matches!(
            rows_to_batch(
                &[json!({"at": "2026-09-22T10:00:00Z"}), json!({"at": u64::MAX})],
                &t
            ),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "at"
        ));
        // An epoch count `i64` holds is still taken.
        let batch = rows_to_batch(&[json!({"at": 1_790_000_000_000_i64})], &t).expect("map");
        assert_eq!(
            types(&batch)["at"],
            DataType::Timestamp(TimeUnit::Millisecond, None)
        );
    }

    #[test]
    fn the_field_cap_bounds_the_columns_a_body_can_allocate() {
        // The cap is the resolver's, enforced as the paths are discovered:
        // a body cannot build a column per distinct key and only then meet
        // the cap that exists to bound them.
        let mut doc = table(vec![("title", DataType::LargeUtf8)]).to_json();
        doc["max_fields"] = json!(3);
        let t = TableSchema::from_json(&doc).expect("schema");
        let at_cap =
            rows_to_batch(&[json!({"title": "t", "a": 1, "b": 2})], &t).expect("at the cap");
        assert_eq!(at_cap.num_columns(), 3);
        resolve_batch(&at_cap, &t, "_id").expect("the resolver agrees");
        assert!(matches!(
            rows_to_batch(&[json!({"a": 1, "b": 2, "c": 3})], &t),
            Err(SchemaError::FieldCapExceeded { cap: 3, current: 1, fields })
                if fields == vec!["a".to_string(), "b".to_string(), "c".to_string()]
        ));
        // The cap counts fields, not cells: one path over many rows is one
        // field.
        let many: Vec<Value> = (0..4).map(|i| json!({"a": i})).collect();
        rows_to_batch(&many, &t).expect("one path");
    }

    #[test]
    fn an_array_of_objects_gives_every_leaf_one_position_per_element() {
        let t = table(vec![]);
        let batch = rows_to_batch(&[json!({"xs": [{"a": 1, "b": 2}, {"a": 3}]})], &t).expect("map");
        let a = col(&batch, "xs.a").as_list::<i32>().value(0);
        assert_eq!(a.as_primitive::<Int64Type>().values(), &[1, 3]);
        let b = col(&batch, "xs.b").as_list::<i32>().value(0);
        let b = b.as_primitive::<Int64Type>();
        assert_eq!(b.len(), 2, "as long as the array");
        assert_eq!(b.value(0), 2);
        assert!(b.is_null(1), "the element without `b` is null in its place");
        resolve_batch(&batch, &t, "_id").expect("stored");

        // A path the earlier elements do not carry is null in their places.
        let batch = rows_to_batch(&[json!({"xs": [{"a": 1}, {"a": 2, "b": 9}]})], &t).expect("map");
        let b = col(&batch, "xs.b").as_list::<i32>().value(0);
        let b = b.as_primitive::<Int64Type>();
        assert_eq!(b.len(), 2);
        assert!(b.is_null(0));
        assert_eq!(b.value(1), 9);
    }

    /// An `f32` column takes a decimal literal. Rounding to `f32` precision
    /// is what the narrower column is for; the guard used to demand the
    /// value survive the round trip bit-exactly, which only dyadic
    /// rationals do, so `0.1` was refused and the column was unusable from
    /// JSON. The vector path already used the range guard.
    #[test]
    fn an_f32_column_takes_a_decimal_literal() {
        let t = table(vec![("score", DataType::Float32)]);
        let batch = rows_to_batch(&[json!({"score": 0.1}), json!({"score": 2.72})], &t)
            .expect("0.1 rounds into an f32 column");
        assert_eq!(types(&batch)["score"], DataType::Float32);
        let scores = col(&batch, "score").as_primitive::<Float32Type>();
        assert_eq!(scores.value(0), 0.1f32);
        assert_eq!(scores.value(1), 2.72f32);
    }

    /// Rounding is admitted; losing the number is not. A magnitude `f32`
    /// cannot reach saturates to an infinity, and a nonzero value below its
    /// smallest subnormal collapses to zero — each a different number
    /// rather than a coarser one, so the column does not take it and the
    /// batch carries the `Float64` the literal infers.
    #[test]
    fn an_f32_column_refuses_a_value_narrowing_would_destroy() {
        let t = table(vec![("score", DataType::Float32)]);
        for value in [json!(1e50), json!(1e-50)] {
            // The mapper infers and never checks: the column's type is not
            // taken, so the literal's own `Float64` is, and the resolver
            // refuses the disagreement as it would an Arrow producer's.
            let batch = rows_to_batch(&[json!({"score": value})], &t).expect("map");
            assert_eq!(types(&batch)["score"], DataType::Float64);
            assert!(
                matches!(
                    resolve_batch(&batch, &t, "_id"),
                    Err(SchemaError::TypeMismatch { column, .. }) if column == "score"
                ),
                "{value} should not reach an f32 column"
            );
        }
    }

    /// An integer is held to the exact bar on `f32` as on `f64`: past the
    /// mantissa it would read back as a different integer.
    #[test]
    fn an_integer_past_the_f32_mantissa_does_not_join_an_f32_column() {
        let t = table(vec![("n", DataType::Float32)]);
        let fits = rows_to_batch(&[json!({"n": 1 << 24})], &t).expect("2^24 is exact in f32");
        assert_eq!(types(&fits)["n"], DataType::Float32);

        let past = rows_to_batch(&[json!({"n": (1i64 << 24) + 1})], &t).expect("map");
        assert_eq!(
            types(&past)["n"],
            DataType::Int64,
            "the column's type is not taken"
        );
        assert!(matches!(
            resolve_batch(&past, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "n"
        ));
    }

    /// A null inside an array is a position, not an absence. Dropping it
    /// shifted every element after it one place left, so a caller read back
    /// a shorter list than they sent, with a 200.
    #[test]
    fn a_null_inside_an_array_keeps_its_position() {
        let t = table(vec![(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        )]);
        let batch = rows_to_batch(&[json!({"tags": [1, null, 3]})], &t).expect("the list maps");
        let lists = col(&batch, "tags").as_list::<i32>();
        let values = lists.value(0);
        let values = values.as_primitive::<Int64Type>();
        assert_eq!(values.len(), 3, "the null holds its place");
        assert_eq!(values.value(0), 1);
        assert!(values.is_null(1));
        assert_eq!(values.value(2), 3);
    }

    /// An empty array is an empty list, not a missing value. On a column the
    /// table already has, the element type comes from the column, so there
    /// is nothing to infer and the row stores the empty list it was sent.
    #[test]
    fn an_empty_array_is_an_empty_list_on_a_known_column() {
        let t = table(vec![(
            "tags",
            DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
        )]);
        let batch = rows_to_batch(&[json!({"tags": []})], &t).expect("the empty list maps");
        let lists = col(&batch, "tags").as_list::<i32>();
        assert!(!lists.is_null(0), "an empty list is not a null");
        assert_eq!(lists.value(0).len(), 0);
    }

    /// A path the table does not have, carrying only empty arrays, still
    /// creates nothing: there is no element type to infer from no elements.
    #[test]
    fn an_empty_array_on_an_unknown_path_creates_no_column() {
        let batch = rows_to_batch(&[json!({"tags": []})], &table(Vec::new())).expect("maps");
        assert!(batch.column_by_name("tags").is_none());
    }

    /// The timestamp spellings a producer actually writes: the
    /// space-separated form Python and most SQL shells print, and a bare
    /// date, which is that day at midnight.
    #[test]
    fn a_timestamp_column_takes_the_spellings_producers_write() {
        let t = table(vec![(
            "at",
            DataType::Timestamp(TimeUnit::Millisecond, None),
        )]);
        for (sent, expected) in [
            ("2024-01-15T10:30:00", 1_705_314_600_000i64),
            ("2024-01-15 10:30:00", 1_705_314_600_000),
            ("2024-01-15", 1_705_276_800_000),
            ("2024-01-15T10:30:00Z", 1_705_314_600_000),
        ] {
            let batch = rows_to_batch(&[json!({"at": sent})], &t)
                .unwrap_or_else(|e| panic!("{sent} should parse: {e}"));
            assert_eq!(
                types(&batch)["at"],
                DataType::Timestamp(TimeUnit::Millisecond, None),
                "{sent}"
            );
            assert_eq!(
                col(&batch, "at")
                    .as_primitive::<TimestampMillisecondType>()
                    .value(0),
                expected,
                "{sent}"
            );
        }
    }

    /// A naive string names a wall clock, and the column's zone says where
    /// that clock is. Read as UTC it marks an instant hours away, and the
    /// write succeeds, so the error is silent: `10:30` on a New York column
    /// used to be stored as `05:30` New York.
    #[test]
    fn a_naive_string_is_the_wall_clock_of_the_columns_zone() {
        let t = table(vec![(
            "at",
            DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".into())),
        )]);
        let batch = rows_to_batch(&[json!({"at": "2024-01-15T10:30:00"})], &t).expect("maps");
        // 10:30 in New York in January is 15:30 UTC.
        assert_eq!(
            col(&batch, "at")
                .as_primitive::<TimestampMillisecondType>()
                .value(0),
            1_705_332_600_000,
        );
    }

    /// A string carrying its own offset names an instant outright, so the
    /// column's zone does not move it.
    #[test]
    fn an_offset_in_the_string_wins_over_the_columns_zone() {
        let t = table(vec![(
            "at",
            DataType::Timestamp(TimeUnit::Millisecond, Some("America/New_York".into())),
        )]);
        let batch = rows_to_batch(&[json!({"at": "2024-01-15T10:30:00Z"})], &t).expect("maps");
        assert_eq!(
            col(&batch, "at")
                .as_primitive::<TimestampMillisecondType>()
                .value(0),
            1_705_314_600_000,
        );
    }

    /// A decimal column takes a number or the string spelling of one, and
    /// the digits are read from the literal rather than through an `f64`:
    /// `12.34` has no exact binary form, and scaling it as a float gives
    /// 1233.9999… , which truncates to the wrong unscaled integer.
    #[test]
    fn a_decimal_column_takes_numbers_and_their_string_spellings() {
        let t = table(vec![("amount", DataType::Decimal128(12, 2))]);
        for sent in [json!(12.34), json!("12.34"), json!("-12.34"), json!(12)] {
            let batch = rows_to_batch(&[json!({"amount": sent})], &t)
                .unwrap_or_else(|e| panic!("{sent} should map: {e}"));
            assert_eq!(types(&batch)["amount"], DataType::Decimal128(12, 2));
        }
        // Values chosen because scaling them through an `f64` gets them
        // wrong: 1.15 * 100 is 114.999…, 8.70 * 100 is 869.999…, and
        // 1234567.89 * 100 is 123456788.999…, each truncating one low.
        // Reading the digits is what makes them exact.
        for (sent, unscaled) in [
            (json!(1.15), 115i128),
            (json!("8.70"), 870),
            (json!(1234567.89), 123_456_789),
            (json!(12.34), 1234),
            (json!(-12.34), -1234),
        ] {
            let batch = rows_to_batch(&[json!({"amount": sent})], &t)
                .unwrap_or_else(|e| panic!("{sent} should map: {e}"));
            assert_eq!(
                col(&batch, "amount")
                    .as_primitive::<Decimal128Type>()
                    .value(0),
                unscaled,
                "{sent}: the unscaled integer comes from the digits, not a float"
            );
        }
    }

    /// More fractional digits than the column keeps would be stored as a
    /// different number. Rounding is the caller's to do.
    #[test]
    fn a_decimal_column_refuses_digits_it_would_have_to_drop() {
        let t = table(vec![("amount", DataType::Decimal128(10, 2))]);
        let batch = rows_to_batch(&[json!({"amount": "12.345"})], &t).expect("map");
        assert_eq!(types(&batch)["amount"], DataType::LargeUtf8);
        assert!(matches!(
            resolve_batch(&batch, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "amount"
        ));
    }

    /// JSON has one number type, so a producer that round-trips through a
    /// double writes `1.0` where it means `1`, and one avoiding a double's
    /// precision loss quotes the integer. Both name the integer they wrote.
    #[test]
    fn an_integer_column_takes_the_float_and_string_spellings() {
        let t = table(vec![("n", DataType::Int64)]);
        for sent in [json!(1.0), json!("42"), json!(7)] {
            let batch = rows_to_batch(&[json!({"n": sent})], &t)
                .unwrap_or_else(|e| panic!("{sent} should map: {e}"));
            assert_eq!(types(&batch)["n"], DataType::Int64, "{sent}");
        }
        let batch = rows_to_batch(&[json!({"n": "42"})], &t).expect("maps");
        assert_eq!(col(&batch, "n").as_primitive::<Int64Type>().value(0), 42);
    }

    /// Quoting an integer is how a producer avoids a double's precision
    /// loss, so a quoted one past the mantissa is held to the same bar an
    /// unquoted one is: storing it in a float column would round it, which
    /// is the loss the quoting was avoiding.
    #[test]
    fn a_quoted_integer_past_the_mantissa_is_refused_by_a_float_column() {
        let t = table(vec![("f", DataType::Float64)]);
        let err = rows_to_batch(&[json!({"f": "9007199254740993"})], &t)
            .expect_err("the integer would be rounded");
        assert!(
            matches!(
                &err,
                SchemaError::IntegerNotExactInFloat { column, value, .. }
                    if column == "f" && *value == 9_007_199_254_740_993
            ),
            "{err:?}"
        );
    }

    /// A fractional literal is not an integer, and is left to be refused as
    /// the type disagreement it is rather than truncated.
    #[test]
    fn a_fractional_literal_is_not_an_integer() {
        let t = table(vec![("n", DataType::Int64)]);
        let batch = rows_to_batch(&[json!({"n": 1.5})], &t).expect("map");
        assert_eq!(types(&batch)["n"], DataType::Float64);
        assert!(matches!(
            resolve_batch(&batch, &t, "_id"),
            Err(SchemaError::TypeMismatch { column, .. }) if column == "n"
        ));
    }

    /// A number in a text column stays text: the text arm is tried before
    /// the numeric ones, so `"42"` on a `LargeUtf8` column is the string.
    #[test]
    fn a_numeric_string_in_a_text_column_stays_text() {
        let t = table(vec![("s", DataType::LargeUtf8)]);
        let batch = rows_to_batch(&[json!({"s": "42"})], &t).expect("maps");
        assert_eq!(types(&batch)["s"], DataType::LargeUtf8);
    }

    /// A string on a path the table does not have is still a string: the
    /// numeric arms need a column to read the type from.
    #[test]
    fn a_numeric_string_on_an_unknown_path_is_text() {
        let batch = rows_to_batch(&[json!({"s": "42"})], &table(Vec::new())).expect("maps");
        assert_eq!(types(&batch)["s"], DataType::LargeUtf8);
    }

    /// A document fills a struct column rather than flattening past it.
    /// `{"image": {"url": …}}` used to become the path `image.url`, which
    /// nests under the live `image` column and was refused outright.
    #[test]
    fn a_document_fills_a_struct_column() {
        let t = table(vec![(
            "image",
            DataType::Struct(Fields::from(vec![
                Field::new("url", DataType::Utf8, true),
                Field::new("width", DataType::Int32, true),
            ])),
        )]);
        let batch = rows_to_batch(
            &[
                json!({"image": {"url": "a.png", "width": 10}}),
                // A field the object leaves out is null, as a column a row
                // leaves out is.
                json!({"image": {"url": "b.png"}}),
                json!({"other": 1}),
            ],
            &t,
        )
        .expect("the documents fill the struct");
        assert_eq!(
            types(&batch)["image"],
            DataType::Struct(Fields::from(vec![
                Field::new("url", DataType::Utf8, true),
                Field::new("width", DataType::Int32, true),
            ]))
        );
        let images = col(&batch, "image").as_struct();
        let urls = images
            .column_by_name("url")
            .expect("url")
            .as_string::<i32>();
        let widths = images
            .column_by_name("width")
            .expect("width")
            .as_primitive::<Int32Type>();
        assert_eq!(urls.value(0), "a.png");
        assert_eq!(widths.value(0), 10);
        assert_eq!(urls.value(1), "b.png");
        assert!(widths.is_null(1), "a field the object omits is null");
        assert!(images.is_null(2), "a row that omits the column is null");
    }

    /// A struct nested in a struct is filled the same way.
    #[test]
    fn a_nested_struct_column_is_filled_too() {
        let inner = Fields::from(vec![Field::new("city", DataType::Utf8, true)]);
        let t = table(vec![(
            "owner",
            DataType::Struct(Fields::from(vec![
                Field::new("name", DataType::Utf8, true),
                Field::new("address", DataType::Struct(inner), true),
            ])),
        )]);
        let batch = rows_to_batch(
            &[json!({"owner": {"name": "ada", "address": {"city": "london"}}})],
            &t,
        )
        .expect("the nested struct fills");
        let owner = col(&batch, "owner").as_struct();
        let address = owner
            .column_by_name("address")
            .expect("address")
            .as_struct();
        assert_eq!(
            address
                .column_by_name("city")
                .expect("city")
                .as_string::<i32>()
                .value(0),
            "london"
        );
    }

    /// A struct's fields are the table's. A top-level key the table has not
    /// seen adds a column; a key a declared struct does not have is a
    /// mistake, and storing the row without it would lose the value.
    #[test]
    fn a_key_a_struct_column_does_not_declare_is_refused() {
        let t = table(vec![(
            "image",
            DataType::Struct(Fields::from(vec![Field::new("url", DataType::Utf8, true)])),
        )]);
        let err = rows_to_batch(&[json!({"image": {"url": "a", "height": 1}})], &t)
            .expect_err("`height` is not a field of the struct");
        assert!(
            matches!(
                &err,
                SchemaError::UnknownStructField { column, field }
                    if column == "image" && field == "height"
            ),
            "{err:?}"
        );
    }

    /// A column the table does not declare a struct still flattens, so a
    /// document cannot grow a column beneath a live scalar one.
    #[test]
    fn a_document_still_cannot_nest_under_a_scalar_column() {
        let t = table(vec![("image", DataType::Utf8)]);
        let err = rows_to_batch(&[json!({"image": {"url": "a"}})], &t)
            .expect_err("a scalar column is not filled by an object");
        assert!(
            matches!(&err, SchemaError::PathShadowsColumn { column, .. } if column == "image"),
            "{err:?}"
        );
    }

    /// A path the table does not have still flattens to dot paths, which is
    /// how a document grows new columns.
    #[test]
    fn an_unknown_path_still_flattens_to_dot_paths() {
        let batch = rows_to_batch(&[json!({"a": {"b": 1}})], &table(Vec::new())).expect("maps");
        assert_eq!(types(&batch)["a.b"], DataType::Int64);
    }
}
