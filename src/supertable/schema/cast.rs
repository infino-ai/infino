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

use arrow::{
    compute::{CastOptions, cast_with_options},
    util::display::FormatOptions,
};
use arrow_array::ArrayRef;
use arrow_schema::{ArrowError, DataType};

/// The options every type conversion in the engine uses: a value that
/// cannot be represented in the target type becomes null.
pub const CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: true,
    format_options: FormatOptions::new(),
};

/// `array` in type `to`, under [`CAST_OPTIONS`].
pub fn cast_column(array: &ArrayRef, to: &DataType) -> Result<ArrayRef, ArrowError> {
    cast_with_options(array, to, &CAST_OPTIONS)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, Int32Array, Int64Array, LargeStringArray};

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
    fn a_narrowing_cast_nulls_what_does_not_fit() {
        let wide: ArrayRef = Arc::new(Int64Array::from(vec![1, i64::from(i32::MAX) + 1]));
        let narrow = cast_column(&wide, &DataType::Int32).expect("cast");
        let narrow = narrow.as_any().downcast_ref::<Int32Array>().expect("i32");
        assert_eq!(narrow.value(0), 1);
        assert!(narrow.is_null(1));
    }
}
