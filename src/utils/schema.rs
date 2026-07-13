// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::collections::HashMap;

use arrow_schema::{Field, SchemaRef};

/// Compare two schemas by field *set*, ignoring column order.
///
/// Two schemas are equal when they declare the same set of fields —
/// matched by name, with identical data type and nullability —
/// regardless of the order those fields appear in. Field-level and
/// schema-level metadata are not considered.
///
/// Order-insensitivity is deliberate: the append path
/// (`split_vectors`) extracts every column by name and re-projects
/// scalars into the supertable's declared order, so a batch whose
/// columns are merely permuted is still a valid append. Rejecting it
/// on order alone would be a false mismatch. Type or nullability
/// drift, a missing column, or an extra column are all still
/// mismatches.
///
/// Cost is O(number of columns), run once per append batch — trivial
/// next to the per-row work an append performs.
pub(crate) fn compare_schema(a: &SchemaRef, b: &SchemaRef) -> bool {
    if a.fields().len() != b.fields().len() {
        return false;
    }

    // Index `b`'s fields by name. Equal field counts plus a hit for
    // every one of `a`'s names means the name sets match exactly
    // (Arrow schema construction forbids duplicate field names).
    let b_by_name: HashMap<&str, &Field> = b
        .fields()
        .iter()
        .map(|f| (f.name().as_str(), f.as_ref()))
        .collect();

    a.fields().iter().all(|fa| {
        b_by_name.get(fa.name().as_str()).is_some_and(|fb| {
            fa.data_type() == fb.data_type() && fa.is_nullable() == fb.is_nullable()
        })
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_schema::{DataType, Field, Schema, SchemaRef};

    use super::compare_schema;

    fn schema(fields: Vec<Field>) -> SchemaRef {
        Arc::new(Schema::new(fields))
    }

    #[test]
    fn identical_schemas_match() {
        let a = schema(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]);
        let b = a.clone();
        assert!(compare_schema(&a, &b));
    }

    #[test]
    fn reordered_columns_match() {
        let a = schema(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]);
        let b = schema(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new("id", DataType::UInt64, false),
        ]);
        assert!(compare_schema(&a, &b));
    }

    #[test]
    fn differing_field_count_mismatches() {
        let a = schema(vec![Field::new("id", DataType::UInt64, false)]);
        let b = schema(vec![
            Field::new("id", DataType::UInt64, false),
            Field::new("title", DataType::LargeUtf8, false),
        ]);
        assert!(!compare_schema(&a, &b));
    }

    #[test]
    fn renamed_column_mismatches() {
        let a = schema(vec![Field::new("id", DataType::UInt64, false)]);
        let b = schema(vec![Field::new("ident", DataType::UInt64, false)]);
        assert!(!compare_schema(&a, &b));
    }

    #[test]
    fn differing_type_mismatches() {
        let a = schema(vec![Field::new("id", DataType::UInt64, false)]);
        let b = schema(vec![Field::new("id", DataType::Int64, false)]);
        assert!(!compare_schema(&a, &b));
    }

    #[test]
    fn differing_nullability_mismatches() {
        let a = schema(vec![Field::new("id", DataType::UInt64, false)]);
        let b = schema(vec![Field::new("id", DataType::UInt64, true)]);
        assert!(!compare_schema(&a, &b));
    }
}
