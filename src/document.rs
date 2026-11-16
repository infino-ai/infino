// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Document columns: a whole JSON document rebuilt from a table's columns
//! instead of stored beside them.
//!
//! A table that keeps each row's original JSON next to the typed columns
//! parsed out of it stores every value twice. A document column
//! ([`IndexSpec::document`](crate::IndexSpec::document)) stores the shape of
//! the document once per row and takes the values from the columns:
//!
//! ```text
//!   original    {"host":{"name":"web-1"},"bytes":512,"at":"01/02/2023"}
//!   skeleton    {"host":{"name":#3},"bytes":#4,"at":#r0}
//!   residual    ["01/02/2023"]
//!                         │
//!                         ▼  rebuild: #3 ← column 3, #4 ← column 4, #r0 ← residual[0]
//!   document    {"host":{"name":"web-1"},"bytes":512,"at":"01/02/2023"}
//! ```
//!
//! ## The skeleton
//!
//! The document's compact JSON text (key order kept, no whitespace) with
//! every scalar value replaced by a [`Slot`]: `#` and a token, written where
//! a value goes and never inside a string, so it cannot be confused with
//! text.
//!
//! | slot           | value                                                    |
//! |----------------|----------------------------------------------------------|
//! | `#<id>`        | the row's value in the column with field id `id`         |
//! | `#<id>.<i>`    | element `i` of the row's value in that (list) column     |
//! | `#r<k>`        | entry `k` of the row's residual                          |
//!
//! Anything else stays as written: `null`, `{}`, `[]`, object keys, and
//! any value the writer chose to keep literal. Keys are never rewritten, so
//! `{"log.file.path":#7}` and `{"log":{"file":{"path":#7}}}` rebuild as
//! the two different documents they were sent as.
//!
//! ## The residual
//!
//! A JSON array whose entries are the raw JSON texts of the values no column
//! renders exactly as written: a date in another format, a number written
//! with a trailing `.0`. Null when the document needed none.
//!
//! ## Writing one
//!
//! The engine does not build skeletons; the writer does, and appends the
//! skeleton and residual as two ordinary columns of the batch. For each
//! scalar value of the document it compares [`render_value`] of the value
//! it stored with the token it parsed: equal byte for byte, the value is a
//! slot; anything else goes to the residual. That rule is what makes the
//! rebuilt document equal the original exactly.

use std::{fmt, iter::repeat_n, str::FromStr};

use arrow::datatypes::ArrowNativeType;
use arrow_array::{
    Array, ArrowPrimitiveType, BooleanArray, DictionaryArray, FixedSizeListArray, GenericListArray,
    OffsetSizeTrait, PrimitiveArray,
    cast::AsArray,
    types::{
        ArrowDictionaryKeyType, Date32Type, Date64Type, Decimal128Type, Decimal256Type,
        Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};
use arrow_schema::{DataType, TimeUnit};
use chrono::{DateTime, Datelike, NaiveDate, TimeDelta, Timelike};
use thiserror::Error;

use crate::supertable::schema::FieldId;

/// Suffix of the stored column holding a document column's skeletons:
/// `source` is rebuilt from `source#skeleton`.
pub const SKELETON_SUFFIX: &str = "#skeleton";

/// Suffix of the stored column holding a document column's residuals:
/// `source` is rebuilt with `source#residual`.
pub const RESIDUAL_SUFFIX: &str = "#residual";

/// The character that opens a slot.
const SLOT_MARK: char = '#';

/// The character that marks a residual slot after [`SLOT_MARK`].
const RESIDUAL_MARK: char = 'r';

/// The character between a slot's field id and its element index.
const ELEMENT_MARK: char = '.';

/// Milliseconds in a day: a `Date64` holding a whole number of them is a
/// date, any other a moment.
const MILLIS_PER_DAY: i64 = 86_400_000;

/// Nanoseconds in a millisecond and in a microsecond: a timestamp renders
/// with three fractional digits, or six or nine when it carries more.
const NANOS_PER_MILLI: u32 = 1_000_000;
const NANOS_PER_MICRO: u32 = 1_000;

/// The last year RFC 3339 can write; the first is year 0.
const MAX_YEAR: i32 = 9999;

/// The largest decimal exponent at which a number still renders without an
/// exponent (`1e21` is the first that does not), and the smallest: the
/// thresholds ECMAScript's `Number.prototype.toString` uses, which is what
/// `JSON.stringify` writes.
const PLAIN_EXPONENT_MAX: i32 = 21;
const PLAIN_EXPONENT_MIN: i32 = -6;

/// The name of the stored column holding the skeletons of the document
/// column `document`.
pub fn skeleton_column(document: &str) -> String {
    format!("{document}{SKELETON_SUFFIX}")
}

/// The name of the stored column holding the residuals of the document
/// column `document`.
pub fn residual_column(document: &str) -> String {
    format!("{document}{RESIDUAL_SUFFIX}")
}

/// One value position of a skeleton. `Display` writes it as the skeleton
/// spells it (`#7`, `#7.2`, `#r0`); `FromStr` reads that spelling back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    /// The row's value in the column with this field id.
    Value(FieldId),
    /// Element `index` of the row's value in this (list) column.
    Element(FieldId, usize),
    /// Entry `index` of the row's residual.
    Residual(usize),
}

impl Slot {
    /// The column the slot reads, or `None` for a residual slot.
    pub fn field(&self) -> Option<FieldId> {
        match self {
            Slot::Value(field) | Slot::Element(field, _) => Some(*field),
            Slot::Residual(_) => None,
        }
    }
}

impl fmt::Display for Slot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Slot::Value(field) => write!(f, "{SLOT_MARK}{}", field.0),
            Slot::Element(field, index) => write!(f, "{SLOT_MARK}{}{ELEMENT_MARK}{index}", field.0),
            Slot::Residual(index) => write!(f, "{SLOT_MARK}{RESIDUAL_MARK}{index}"),
        }
    }
}

impl FromStr for Slot {
    type Err = DocumentError;

    /// Read a slot written as [`Slot`]'s `Display` writes it, `#` included.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let bad = || DocumentError::BadSlot {
            slot: text.to_owned(),
        };
        let token = text.strip_prefix(SLOT_MARK).ok_or_else(bad)?;
        parse_token(token).ok_or_else(bad)
    }
}

/// Why a skeleton could not be rebuilt.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum DocumentError {
    /// A `#` outside a string is not followed by a slot the grammar names.
    #[error("skeleton holds `{slot}`, which is not a slot")]
    BadSlot {
        /// The text from the `#` to the end of the token.
        slot: String,
    },
    /// The skeleton ends inside a string.
    #[error("skeleton ends inside a string")]
    UnterminatedString,
    /// The residual is not a JSON array of values.
    #[error("residual is not a JSON array of values: {reason}")]
    BadResidual {
        /// What is wrong with it.
        reason: String,
    },
    /// A residual slot names an entry the residual does not have.
    #[error("skeleton names residual entry {index}, and the residual has {len}")]
    MissingResidual {
        /// The entry named.
        index: usize,
        /// How many entries the residual has (zero when it is null).
        len: usize,
    },
    /// A column slot names a value the row does not hold: the column is
    /// null there, is not in the table, or cannot be rendered as JSON.
    #[error("skeleton names {slot}, and the row holds no value there")]
    MissingValue {
        /// The slot.
        slot: Slot,
    },
}

/// A slot's token, the text after the `#`: `7`, `7.2` or `r0`.
fn parse_token(token: &str) -> Option<Slot> {
    let digits = |text: &str| -> Option<usize> {
        (!text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()))
            .then(|| text.parse().ok())
            .flatten()
    };
    if let Some(index) = token.strip_prefix(RESIDUAL_MARK) {
        return digits(index).map(Slot::Residual);
    }
    let field = |text: &str| {
        digits(text)
            .and_then(|id| u32::try_from(id).ok())
            .map(FieldId)
    };
    match token.split_once(ELEMENT_MARK) {
        Some((id, index)) => Some(Slot::Element(field(id)?, digits(index)?)),
        None => field(token).map(Slot::Value),
    }
}

/// Whether `b` can be part of a slot's token.
fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == ELEMENT_MARK as u8
}

/// A piece of a skeleton: literal text to copy, or a slot to fill.
enum Piece<'a> {
    Text(&'a str),
    Slot(Slot),
}

/// Walk `skeleton`, handing each literal run and each slot to `piece` in
/// order. Strings are copied whole, so a `#` inside one is text.
fn walk<'a>(
    skeleton: &'a str,
    mut piece: impl FnMut(Piece<'a>) -> Result<(), DocumentError>,
) -> Result<(), DocumentError> {
    let bytes = skeleton.as_bytes();
    let mut text_from = 0;
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'"' => at = string_end(bytes, at).ok_or(DocumentError::UnterminatedString)?,
            b'#' => {
                piece(Piece::Text(&skeleton[text_from..at]))?;
                let end = at
                    + 1
                    + bytes[at + 1..]
                        .iter()
                        .take_while(|b| is_token_byte(**b))
                        .count();
                let slot = &skeleton[at..end];
                piece(Piece::Slot(slot.parse()?))?;
                at = end;
                text_from = end;
            }
            _ => at += 1,
        }
    }
    piece(Piece::Text(&skeleton[text_from..]))
}

/// The position just past the string opening at `start`, or `None` when
/// the text ends inside it.
fn string_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut at = start + 1;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at += 2,
            b'"' => return Some(at + 1),
            _ => at += 1,
        }
    }
    None
}

/// The slots of `skeleton`, in the order they appear: what a reader needs
/// to know which columns a document draws on.
pub fn slots(skeleton: &str) -> Result<Vec<Slot>, DocumentError> {
    let mut out = Vec::new();
    walk(skeleton, |piece| {
        if let Piece::Slot(slot) = piece {
            out.push(slot);
        }
        Ok(())
    })?;
    Ok(out)
}

/// The residual text holding `values`, each a raw JSON value text, in
/// order: what a writer stores beside a skeleton whose residual slots are
/// `#r0`, `#r1`, ….
pub fn residual<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut out = String::from("[");
    for (i, value) in values.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(value);
    }
    out.push(']');
    out
}

/// The raw text of each entry of the residual array `text`, whitespace
/// around entries trimmed and everything inside them kept as written.
pub fn residual_values(text: &str) -> Result<Vec<&str>, DocumentError> {
    let bytes = text.as_bytes();
    let bad = |reason: &str| DocumentError::BadResidual {
        reason: reason.to_owned(),
    };
    let skip_ws = |mut at: usize| {
        while at < bytes.len() && bytes[at].is_ascii_whitespace() {
            at += 1;
        }
        at
    };
    let mut at = skip_ws(0);
    if bytes.get(at) != Some(&b'[') {
        return Err(bad("it does not open with `[`"));
    }
    at = skip_ws(at + 1);
    let mut values = Vec::new();
    if bytes.get(at) == Some(&b']') {
        at += 1;
    } else {
        loop {
            let end = value_end(bytes, at).ok_or_else(|| bad("an entry is not a JSON value"))?;
            values.push(&text[at..end]);
            at = skip_ws(end);
            match bytes.get(at) {
                Some(b',') => at = skip_ws(at + 1),
                Some(b']') => {
                    at += 1;
                    break;
                }
                _ => return Err(bad("entries are not separated by `,` and closed by `]`")),
            }
        }
    }
    if skip_ws(at) != bytes.len() {
        return Err(bad("text follows the closing `]`"));
    }
    Ok(values)
}

/// The position just past the JSON value starting at `start`: a string, an
/// object or array (matched to its closing bracket, strings skipped), or a
/// scalar running to the next separator. `None` when there is no value
/// there or it does not close.
fn value_end(bytes: &[u8], start: usize) -> Option<usize> {
    match bytes.get(start)? {
        b'"' => string_end(bytes, start),
        b'[' | b'{' => {
            let mut depth = 0usize;
            let mut at = start;
            while at < bytes.len() {
                match bytes[at] {
                    b'"' => {
                        at = string_end(bytes, at)?;
                        continue;
                    }
                    b'[' | b'{' => depth += 1,
                    b']' | b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            return Some(at + 1);
                        }
                    }
                    _ => {}
                }
                at += 1;
            }
            None
        }
        _ => {
            let len = bytes[start..]
                .iter()
                .take_while(|b| !matches!(b, b',' | b']' | b'}') && !b.is_ascii_whitespace())
                .count();
            (len > 0).then_some(start + len)
        }
    }
}

/// Rebuild the document `skeleton` describes: its literal text as written,
/// each column slot filled with `value_of(field, element)` (the canonical
/// JSON text of the row's value there, as [`render_value`] gives it), and
/// each residual slot with that entry of `residual`.
///
/// A slot with nothing to fill it is an error, never a silent `null`: a
/// writer only writes a slot for a value the column holds.
pub fn rebuild(
    skeleton: &str,
    residual: Option<&str>,
    value_of: impl Fn(FieldId, Option<usize>) -> Option<String>,
) -> Result<String, DocumentError> {
    let mut out = String::with_capacity(skeleton.len() * 2);
    let mut entries: Option<Vec<&str>> = None;
    walk(skeleton, |piece| {
        match piece {
            Piece::Text(text) => out.push_str(text),
            Piece::Slot(slot) => match slot {
                Slot::Value(field) | Slot::Element(field, _) => {
                    let element = match slot {
                        Slot::Element(_, index) => Some(index),
                        _ => None,
                    };
                    let value =
                        value_of(field, element).ok_or(DocumentError::MissingValue { slot })?;
                    out.push_str(&value);
                }
                Slot::Residual(index) => {
                    if entries.is_none() {
                        entries = Some(match residual {
                            Some(text) => residual_values(text)?,
                            None => Vec::new(),
                        });
                    }
                    let entries = entries.as_ref().expect("filled above");
                    let entry = entries.get(index).ok_or(DocumentError::MissingResidual {
                        index,
                        len: entries.len(),
                    })?;
                    out.push_str(entry);
                }
            },
        }
        Ok(())
    })?;
    Ok(out)
}

/// The canonical JSON text of the value at `row` of `array`, or of element
/// `element` of the list there; `None` when the value (or the element) is
/// null, absent, or of a type with no JSON rendering.
///
/// | type                         | text                                              |
/// |------------------------------|---------------------------------------------------|
/// | strings                      | a JSON string, escaped as `serde_json` escapes it  |
/// | integers                     | base 10                                           |
/// | floats                       | the shortest digits that round-trip, laid out as `JSON.stringify` lays them out (`1.5`, `100`, `1e+21`, `1e-7`); `None` for NaN and infinities |
/// | decimals                     | the decimal at its scale (`12.50`)                 |
/// | booleans                     | `true` / `false`                                  |
/// | timestamps                   | RFC 3339 in UTC with milliseconds (`"2023-01-01T09:37:20.000Z"`), six or nine fractional digits when the value carries more |
/// | `Date32`, whole-day `Date64` | `"2023-01-01"`; a `Date64` with a time of day as a timestamp |
/// | lists (no `element`)         | `[` the elements, nulls as `null`, `]`            |
/// | dictionaries                 | the value the key points at                       |
///
/// The rendering is the one contract between a writer and this reader: a
/// value is slotted only when this text equals the token it was parsed from.
pub fn render_value(array: &dyn Array, row: usize, element: Option<usize>) -> Option<String> {
    if row >= array.len() || array.is_null(row) {
        return None;
    }
    match element {
        None => {
            let mut out = String::new();
            write_value(array, row, &mut out)?;
            Some(out)
        }
        Some(index) => {
            let (values, start, len) = list_slice(array, row)?;
            (index < len)
                .then(|| render_value(values, start + index, None))
                .flatten()
        }
    }
}

/// The child values of the list at `row` of `array`, with where the row's
/// elements start in them and how many there are. `None` when `array` is
/// not a list.
fn list_slice(array: &dyn Array, row: usize) -> Option<(&dyn Array, usize, usize)> {
    fn offsets<O: OffsetSizeTrait>(
        list: &GenericListArray<O>,
        row: usize,
    ) -> (&dyn Array, usize, usize) {
        let offsets = list.value_offsets();
        let start = offsets[row].as_usize();
        let end = offsets[row + 1].as_usize();
        (list.values().as_ref(), start, end - start)
    }
    match array.data_type() {
        DataType::List(_) => Some(offsets(array.as_list::<i32>(), row)),
        DataType::LargeList(_) => Some(offsets(array.as_list::<i64>(), row)),
        DataType::FixedSizeList(_, size) => {
            let list: &FixedSizeListArray = array.as_fixed_size_list();
            let size = usize::try_from(*size).ok()?;
            Some((list.values().as_ref(), row * size, size))
        }
        _ => None,
    }
}

/// Append the JSON text of the non-null value at `row` of `array` to `out`;
/// `None` (with `out` partly written) when it has no rendering.
fn write_value(array: &dyn Array, row: usize, out: &mut String) -> Option<()> {
    match array.data_type() {
        DataType::Utf8 => write_string(array.as_string::<i32>().value(row), out),
        DataType::LargeUtf8 => write_string(array.as_string::<i64>().value(row), out),
        DataType::Utf8View => write_string(array.as_string_view().value(row), out),
        DataType::Boolean => {
            let bools: &BooleanArray = array.as_boolean();
            out.push_str(if bools.value(row) { "true" } else { "false" });
            Some(())
        }
        DataType::Int8 => write_display(array.as_primitive::<Int8Type>(), row, out),
        DataType::Int16 => write_display(array.as_primitive::<Int16Type>(), row, out),
        DataType::Int32 => write_display(array.as_primitive::<Int32Type>(), row, out),
        DataType::Int64 => write_display(array.as_primitive::<Int64Type>(), row, out),
        DataType::UInt8 => write_display(array.as_primitive::<UInt8Type>(), row, out),
        DataType::UInt16 => write_display(array.as_primitive::<UInt16Type>(), row, out),
        DataType::UInt32 => write_display(array.as_primitive::<UInt32Type>(), row, out),
        DataType::UInt64 => write_display(array.as_primitive::<UInt64Type>(), row, out),
        DataType::Float32 => {
            let value = array.as_primitive::<Float32Type>().value(row);
            write_float(value.is_finite(), &format!("{value:e}"), out)
        }
        DataType::Float64 => {
            let value = array.as_primitive::<Float64Type>().value(row);
            write_float(value.is_finite(), &format!("{value:e}"), out)
        }
        DataType::Decimal128(..) => {
            out.push_str(&array.as_primitive::<Decimal128Type>().value_as_string(row));
            Some(())
        }
        DataType::Decimal256(..) => {
            out.push_str(&array.as_primitive::<Decimal256Type>().value_as_string(row));
            Some(())
        }
        DataType::Timestamp(unit, _) => {
            let value = match unit {
                TimeUnit::Second => array.as_primitive::<TimestampSecondType>().value(row),
                TimeUnit::Millisecond => {
                    array.as_primitive::<TimestampMillisecondType>().value(row)
                }
                TimeUnit::Microsecond => {
                    array.as_primitive::<TimestampMicrosecondType>().value(row)
                }
                TimeUnit::Nanosecond => array.as_primitive::<TimestampNanosecondType>().value(row),
            };
            write_timestamp(*unit, value, out)
        }
        DataType::Date32 => {
            let days = array.as_primitive::<Date32Type>().value(row);
            write_date(i64::from(days), out)
        }
        DataType::Date64 => {
            let millis = array.as_primitive::<Date64Type>().value(row);
            match millis % MILLIS_PER_DAY {
                0 => write_date(millis / MILLIS_PER_DAY, out),
                _ => write_timestamp(TimeUnit::Millisecond, millis, out),
            }
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(..) => {
            let (values, start, len) = list_slice(array, row)?;
            out.push('[');
            for i in 0..len {
                if i > 0 {
                    out.push(',');
                }
                if values.is_null(start + i) {
                    out.push_str("null");
                } else {
                    write_value(values, start + i, out)?;
                }
            }
            out.push(']');
            Some(())
        }
        DataType::Dictionary(key, _) => match key.as_ref() {
            DataType::Int8 => write_keyed(array.as_dictionary::<Int8Type>(), row, out),
            DataType::Int16 => write_keyed(array.as_dictionary::<Int16Type>(), row, out),
            DataType::Int32 => write_keyed(array.as_dictionary::<Int32Type>(), row, out),
            DataType::Int64 => write_keyed(array.as_dictionary::<Int64Type>(), row, out),
            DataType::UInt8 => write_keyed(array.as_dictionary::<UInt8Type>(), row, out),
            DataType::UInt16 => write_keyed(array.as_dictionary::<UInt16Type>(), row, out),
            DataType::UInt32 => write_keyed(array.as_dictionary::<UInt32Type>(), row, out),
            DataType::UInt64 => write_keyed(array.as_dictionary::<UInt64Type>(), row, out),
            _ => None,
        },
        _ => None,
    }
}

/// `text` as a JSON string.
fn write_string(text: &str, out: &mut String) -> Option<()> {
    out.push_str(&serde_json::to_string(text).ok()?);
    Some(())
}

/// The integer at `row`, in base 10.
fn write_display<T: ArrowPrimitiveType>(
    array: &PrimitiveArray<T>,
    row: usize,
    out: &mut String,
) -> Option<()>
where
    T::Native: fmt::Display,
{
    out.push_str(&array.value(row).to_string());
    Some(())
}

/// The value the key at `row` of `dictionary` points at.
fn write_keyed<K: ArrowDictionaryKeyType>(
    dictionary: &DictionaryArray<K>,
    row: usize,
    out: &mut String,
) -> Option<()> {
    let key = dictionary.keys().value(row).as_usize();
    let values = dictionary.values();
    if values.is_null(key) {
        return None;
    }
    write_value(values.as_ref(), key, out)
}

/// A float given as Rust's shortest round-trip scientific text (`{:e}`:
/// `1.5e0`, `-2.5e-7`, `1e21`), laid out as ECMAScript's
/// `Number.prototype.toString` lays out those digits: plain decimal for a
/// decimal exponent from -6 to 20, scientific with an explicit sign
/// otherwise. Not finite: no rendering.
fn write_float(finite: bool, scientific: &str, out: &mut String) -> Option<()> {
    if !finite {
        return None;
    }
    let (mantissa, exponent) = scientific.split_once('e')?;
    let exponent: i32 = exponent.parse().ok()?;
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if digits.bytes().all(|b| b == b'0') {
        out.push('0');
        return Some(());
    }
    if negative {
        out.push('-');
    }
    // ECMAScript's `n`: the value is 0.<digits> × 10^n.
    let n = exponent + 1;
    let k = digits.len() as i32;
    if k <= n && n <= PLAIN_EXPONENT_MAX {
        out.push_str(&digits);
        out.extend(repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= PLAIN_EXPONENT_MAX {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if PLAIN_EXPONENT_MIN < n && n <= 0 {
        out.push_str("0.");
        out.extend(repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if n - 1 >= 0 { '+' } else { '-' });
        out.push_str(&(n - 1).abs().to_string());
    }
    Some(())
}

/// A timestamp of `unit` as an RFC 3339 string in UTC: milliseconds always,
/// microseconds or nanoseconds when the value carries them. A year RFC 3339
/// cannot write (before 0 or after 9999) has no rendering.
fn write_timestamp(unit: TimeUnit, value: i64, out: &mut String) -> Option<()> {
    let at = match unit {
        TimeUnit::Second => DateTime::from_timestamp(value, 0)?,
        TimeUnit::Millisecond => DateTime::from_timestamp_millis(value)?,
        TimeUnit::Microsecond => DateTime::from_timestamp_micros(value)?,
        TimeUnit::Nanosecond => DateTime::from_timestamp_nanos(value),
    };
    let naive = at.naive_utc();
    if !(0..=MAX_YEAR).contains(&naive.year()) {
        return None;
    }
    let nanos = naive.nanosecond();
    let fraction = if nanos % NANOS_PER_MILLI == 0 {
        format!("{:03}", nanos / NANOS_PER_MILLI)
    } else if nanos % NANOS_PER_MICRO == 0 {
        format!("{:06}", nanos / NANOS_PER_MICRO)
    } else {
        format!("{nanos:09}")
    };
    out.push('"');
    out.push_str(&naive.format("%Y-%m-%dT%H:%M:%S").to_string());
    out.push('.');
    out.push_str(&fraction);
    out.push_str("Z\"");
    Some(())
}

/// A date `days` after 1970-01-01 as an RFC 3339 full date.
fn write_date(days: i64, out: &mut String) -> Option<()> {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
    let date = epoch.checked_add_signed(TimeDelta::try_days(days)?)?;
    if !(0..=MAX_YEAR).contains(&date.year()) {
        return None;
    }
    out.push('"');
    out.push_str(&date.format("%Y-%m-%d").to_string());
    out.push('"');
    Some(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{
        ArrayRef, BooleanArray, Date32Array, Date64Array, Float32Array, Float64Array, Int64Array,
        LargeStringArray, ListArray, StringArray, StringViewArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
        types::Int32Type,
    };

    use super::*;

    /// `2023-01-01T09:37:20Z` in seconds since the epoch.
    const SECONDS: i64 = 1_672_565_840;
    /// 2023-01-01 in days since the epoch.
    const DAYS: i32 = 19_358;

    fn one(array: ArrayRef) -> Option<String> {
        render_value(array.as_ref(), 0, None)
    }

    #[test]
    fn a_slot_writes_and_reads_back_its_own_spelling() {
        for (slot, text) in [
            (Slot::Value(FieldId(7)), "#7"),
            (Slot::Element(FieldId(7), 2), "#7.2"),
            (Slot::Residual(0), "#r0"),
            (Slot::Residual(12), "#r12"),
        ] {
            assert_eq!(slot.to_string(), text);
            assert_eq!(text.parse::<Slot>(), Ok(slot));
        }
        for bad in [
            "7",
            "#",
            "#r",
            "#x1",
            "#7.",
            "#.2",
            "#7.2.3",
            "#r1.2",
            "#99999999999",
        ] {
            assert!(bad.parse::<Slot>().is_err(), "{bad}");
        }
    }

    #[test]
    fn slots_are_found_outside_strings_only() {
        let skeleton = r##"{"a#1":#1,"b":[#2.0,"#3",{"c":#r0}],"d\"#4":null}"##;
        assert_eq!(
            slots(skeleton),
            Ok(vec![
                Slot::Value(FieldId(1)),
                Slot::Element(FieldId(2), 0),
                Slot::Residual(0),
            ])
        );
        assert_eq!(
            slots(r#"{"a":"open"#),
            Err(DocumentError::UnterminatedString)
        );
        assert!(matches!(
            slots("{\"a\":#q}"),
            Err(DocumentError::BadSlot { .. })
        ));
    }

    #[test]
    fn rebuild_fills_slots_and_keeps_everything_else_as_written() {
        let skeleton = r##"{"log.file.path":#7,"log":{"file":{"path":#7}},"t":[#8.1,#r1],"n":null,"o":{},"a":[],"x":#r0}"##;
        let residual = r#"[ "01/02/2023" , {"k": [1, "]"]} ]"#;
        let value_of = |field: FieldId, element: Option<usize>| match (field.0, element) {
            (7, None) => Some(r#""/a""#.to_owned()),
            (8, Some(1)) => Some("2".to_owned()),
            _ => None,
        };
        assert_eq!(
            rebuild(skeleton, Some(residual), value_of),
            Ok(r#"{"log.file.path":"/a","log":{"file":{"path":"/a"}},"t":[2,{"k": [1, "]"]}],"n":null,"o":{},"a":[],"x":"01/02/2023"}"#.to_owned())
        );
        assert_eq!(
            rebuild("#9", None, value_of),
            Err(DocumentError::MissingValue {
                slot: Slot::Value(FieldId(9))
            })
        );
        assert_eq!(
            rebuild("[#r2]", Some("[1]"), value_of),
            Err(DocumentError::MissingResidual { index: 2, len: 1 })
        );
        assert_eq!(
            rebuild("[#r0]", None, value_of),
            Err(DocumentError::MissingResidual { index: 0, len: 0 })
        );
        assert!(matches!(
            rebuild("[#r0]", Some("[1,]"), value_of),
            Err(DocumentError::BadResidual { .. })
        ));
    }

    #[test]
    fn residual_builds_what_residual_values_reads() {
        let entries = [r#""a,b""#, "1.0", r#"{"x":[1,{"y":"}"}]}"#, "null"];
        let text = residual(entries);
        assert_eq!(residual_values(&text), Ok(entries.to_vec()));
        assert_eq!(residual_values(" [ ] "), Ok(vec![]));
    }

    #[test]
    fn strings_render_as_json_strings() {
        let text = "a \"q\" \\ \n\t\u{1} é";
        let want = serde_json::to_string(text).expect("json");
        assert_eq!(
            one(Arc::new(StringArray::from(vec![text]))),
            Some(want.clone())
        );
        assert_eq!(
            one(Arc::new(LargeStringArray::from(vec![text]))),
            Some(want.clone())
        );
        assert_eq!(
            one(Arc::new(StringViewArray::from(vec![text]))),
            Some(want.clone())
        );
        let dictionary: DictionaryArray<Int32Type> = vec![text].into_iter().collect();
        assert_eq!(one(Arc::new(dictionary)), Some(want));
        assert_eq!(one(Arc::new(StringArray::from(vec![None::<&str>]))), None);
    }

    #[test]
    fn numbers_and_booleans_render_as_json_writes_them() {
        assert_eq!(
            one(Arc::new(Int64Array::from(vec![-42]))),
            Some("-42".into())
        );
        assert_eq!(
            one(Arc::new(UInt64Array::from(vec![u64::MAX]))),
            Some(u64::MAX.to_string())
        );
        assert_eq!(
            one(Arc::new(BooleanArray::from(vec![true]))),
            Some("true".into())
        );
        for (value, text) in [
            (0.25, "0.25"),
            (1.0, "1"),
            (100.0, "100"),
            (-0.0, "0"),
            (123456.789, "123456.789"),
            (1e21, "1e+21"),
            (1.5e300, "1.5e+300"),
            (1e20, "100000000000000000000"),
            (1e-7, "1e-7"),
            (0.000001, "0.000001"),
            (2.5e-10, "2.5e-10"),
            (0.1 + 0.2, "0.30000000000000004"),
        ] {
            assert_eq!(
                one(Arc::new(Float64Array::from(vec![value]))),
                Some(text.into())
            );
        }
        assert_eq!(
            one(Arc::new(Float32Array::from(vec![0.1f32]))),
            Some("0.1".into())
        );
        assert_eq!(one(Arc::new(Float64Array::from(vec![f64::NAN]))), None);
        assert_eq!(one(Arc::new(Float64Array::from(vec![f64::INFINITY]))), None);
    }

    #[test]
    fn times_render_as_rfc_3339_in_utc_with_milliseconds() {
        let want = r#""2023-01-01T09:37:20.000Z""#;
        assert_eq!(
            one(Arc::new(TimestampSecondArray::from(vec![SECONDS]))),
            Some(want.into())
        );
        assert_eq!(
            one(Arc::new(TimestampMillisecondArray::from(vec![
                SECONDS * 1_000
            ]))),
            Some(want.into())
        );
        assert_eq!(
            one(Arc::new(
                TimestampMillisecondArray::from(vec![SECONDS * 1_000 + 5]).with_timezone("+02:00")
            )),
            Some(r#""2023-01-01T09:37:20.005Z""#.into())
        );
        assert_eq!(
            one(Arc::new(TimestampMicrosecondArray::from(vec![
                SECONDS * 1_000_000 + 1
            ]))),
            Some(r#""2023-01-01T09:37:20.000001Z""#.into())
        );
        assert_eq!(
            one(Arc::new(TimestampNanosecondArray::from(vec![
                SECONDS * 1_000_000_000 + 7
            ]))),
            Some(r#""2023-01-01T09:37:20.000000007Z""#.into())
        );
        assert_eq!(
            one(Arc::new(Date32Array::from(vec![DAYS]))),
            Some(r#""2023-01-01""#.into())
        );
        let day_ms = i64::from(DAYS) * MILLIS_PER_DAY;
        assert_eq!(
            one(Arc::new(Date64Array::from(vec![day_ms]))),
            Some(r#""2023-01-01""#.into())
        );
        assert_eq!(
            one(Arc::new(Date64Array::from(vec![day_ms + 1]))),
            Some(r#""2023-01-01T00:00:00.001Z""#.into())
        );
    }

    #[test]
    fn a_list_renders_whole_or_by_element() {
        let list: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), None, Some(3)]),
            None,
            Some(vec![]),
        ]));
        assert_eq!(
            render_value(list.as_ref(), 0, None),
            Some("[1,null,3]".into())
        );
        assert_eq!(render_value(list.as_ref(), 0, Some(2)), Some("3".into()));
        assert_eq!(render_value(list.as_ref(), 0, Some(1)), None);
        assert_eq!(render_value(list.as_ref(), 0, Some(3)), None);
        assert_eq!(render_value(list.as_ref(), 1, None), None);
        assert_eq!(render_value(list.as_ref(), 2, None), Some("[]".into()));
        // An element of a column that is not a list has no rendering.
        let ints: ArrayRef = Arc::new(Int64Array::from(vec![1]));
        assert_eq!(render_value(ints.as_ref(), 0, Some(0)), None);
    }
}
