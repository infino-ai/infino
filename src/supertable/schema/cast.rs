// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The one cast a column goes through when a file holds it in a type other
//! than the table's.
//!
//! A column's type can change after files were written in the old one.
//! Until compaction rewrites such a file, every read of it casts the
//! column on the way out, and compaction casts it once more as it rewrites
//! the file. Both go through this kernel with the same options, so a row
//! reads the same value before and after its file is rewritten: a value
//! that does not cast is null on both sides, never an error on one.
//!
//! Text read as a number keeps the rule tables were written under: a value
//! with whitespace before or after its digits does not parse, so `" 5"` is
//! null, not 5. Arrow's own text-to-number cast trims that whitespace from
//! version 59 on; without the rule here, a value a table returned as null
//! would come back as a number after an upgrade, beside a compacted file
//! that already stores the null.

use std::sync::Arc;

use arrow::{
    compute::{CastOptions, cast_with_options, nullif},
    util::display::FormatOptions,
};
use arrow_array::{Array, ArrayRef, BooleanArray, cast::AsArray};
use arrow_schema::{ArrowError, DataType};

/// The options every type conversion in the engine uses: a value that
/// cannot be represented in the target type becomes null.
pub const CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: true,
    format_options: FormatOptions::new(),
};

/// `array` in type `to`, under [`CAST_OPTIONS`] and the whitespace rule in
/// the module docs.
pub fn cast_column(array: &ArrayRef, to: &DataType) -> Result<ArrayRef, ArrowError> {
    cast_with_options(&null_padded_numbers(array, to)?, to, &CAST_OPTIONS)
}

/// Whether casting `from` to `to` parses text as a number: text to an
/// integer, a float or a duration, the casts whose parsing trims
/// whitespace from Arrow 59 on.
pub(crate) fn parses_text_as_number(from: &DataType, to: &DataType) -> bool {
    matches!(
        from,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    ) && (to.is_integer() || to.is_floating() || matches!(to, DataType::Duration(_)))
}

/// `array` with each value that starts or ends with ASCII whitespace set
/// to null, when casting it to `to` parses text as a number; `array`
/// unchanged otherwise. Applied before the cast, it keeps the rule in the
/// module docs.
pub(crate) fn null_padded_numbers(array: &ArrayRef, to: &DataType) -> Result<ArrayRef, ArrowError> {
    if !parses_text_as_number(array.data_type(), to) {
        return Ok(Arc::clone(array));
    }
    let padded: BooleanArray = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().iter().map(padded).collect(),
        DataType::LargeUtf8 => array.as_string::<i64>().iter().map(padded).collect(),
        DataType::Utf8View => array.as_string_view().iter().map(padded).collect(),
        _ => return Ok(Arc::clone(array)),
    };
    if padded.true_count() == 0 {
        return Ok(Arc::clone(array));
    }
    nullif(array, &padded)
}

/// Whether `value` starts or ends with ASCII whitespace; `None` stays null.
fn padded(value: Option<&str>) -> Option<bool> {
    let bytes = value?.as_bytes();
    Some(
        bytes.first().is_some_and(u8::is_ascii_whitespace)
            || bytes.last().is_some_and(u8::is_ascii_whitespace),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray};

    use super::*;

    #[test]
    fn a_value_that_does_not_cast_becomes_null_not_an_error() {
        let strings: ArrayRef = Arc::new(LargeStringArray::from(vec![
            Some("5"),
            Some("5.0"),
            Some(" 5"),
            Some(""),
            Some("99999999999999999999"),
            None,
        ]));
        let ints = cast_column(&strings, &DataType::Int64).expect("cast");
        let ints = ints.as_any().downcast_ref::<Int64Array>().expect("i64");
        assert_eq!(ints.value(0), 5);
        assert!(ints.is_null(1), "a float literal is not an integer");
        assert!(ints.is_null(2), "no trimming");
        assert!(ints.is_null(3));
        assert!(ints.is_null(4), "overflow is null");
        assert!(ints.is_null(5));
    }

    #[test]
    fn text_with_whitespace_around_its_digits_is_not_a_number() {
        let text: ArrayRef = Arc::new(StringArray::from(vec![
            Some("7"),
            Some("7 "),
            Some("\t7"),
            Some("7\n"),
            Some(" 7.5"),
        ]));
        for to in [DataType::Int64, DataType::UInt32, DataType::Float64] {
            let numbers = cast_column(&text, &to).expect("cast");
            assert!(numbers.is_valid(0), "{to}");
            for row in 1..numbers.len() {
                assert!(numbers.is_null(row), "{to}: row {row}");
            }
        }

        let view: ArrayRef = Arc::new(StringViewArray::from(vec![Some("7"), Some(" 7")]));
        let numbers = cast_column(&view, &DataType::Int64).expect("cast");
        assert!(numbers.is_valid(0));
        assert!(numbers.is_null(1));

        // Text read as text keeps its whitespace.
        let large = cast_column(&text, &DataType::LargeUtf8).expect("cast");
        let large = large
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("large utf8");
        assert_eq!(large.value(1), "7 ");
    }

    #[test]
    fn a_narrowing_cast_nulls_what_does_not_fit() {
        let wide: ArrayRef = Arc::new(Int64Array::from(vec![1, i64::from(i32::MAX) + 1]));
        let narrow = cast_column(&wide, &DataType::Int32).expect("cast");
        let narrow = narrow.as_any().downcast_ref::<Int32Array>().expect("i32");
        assert_eq!(narrow.value(0), 1);
        assert!(narrow.is_null(1));
    }
}
