// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The one error every path that touches the schema returns: an append
//! whose batch disagrees with the table, the row mapper, and the schema
//! write. Each variant's message names the remedy.

use arrow_schema::DataType;
use thiserror::Error;

use super::FieldId;

/// Why a batch, a row document or a schema write was refused.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum SchemaError {
    /// A batch offers a column in a type other than the one the table
    /// froze for it.
    #[error(
        "column `{column}` is `{frozen}` but the batch carries `{offered}`; \
         apply the schema with the column's type changed to store it"
    )]
    TypeMismatch {
        /// The column's name.
        column: String,
        /// The type the table froze for it.
        frozen: DataType,
        /// The type the batch carries.
        offered: DataType,
    },
    /// A batch lacks a column the table requires a value for.
    #[error("column `{column}` is not nullable and the batch does not carry it")]
    MissingColumn {
        /// The column's name.
        column: String,
    },
    /// A batch carries a null in a column the table requires a value for.
    #[error("column `{column}` is not nullable and the batch carries a null in it")]
    NullInNonNullable {
        /// The column's name.
        column: String,
    },
    /// A value does not fit the column's type. The mapper picks a type the
    /// values fit, so this is a value the type was not picked from: one the
    /// table already froze, or one a wider value shares a column with.
    #[error("field `{column}` cannot hold {value} as `{data_type}`")]
    ValueOutOfRange {
        /// The field's path.
        column: String,
        /// The value, as it was written.
        value: String,
        /// The type the column holds.
        data_type: String,
    },

    /// A column holds an integer too large to be exact in `f64` beside a
    /// value that makes the column a float.
    #[error(
        "field `{column}` holds the integer {value}, which a float column \
         cannot hold exactly: it would read back as {stored}"
    )]
    IntegerNotExactInFloat {
        /// The field's path.
        column: String,
        /// The integer that would change.
        value: i128,
        /// What it would read back as.
        stored: f64,
    },

    /// A document's array mixes element types.
    #[error("field `{column}` is an array of more than one type ({types:?})")]
    MixedArray {
        /// The field's path.
        column: String,
        /// The element types seen.
        types: Vec<String>,
    },
    /// Adding these fields would exceed the table's field cap.
    #[error(
        "adding {fields:?} would take the table over its cap of {cap} fields \
         (it has {current}); raise `max_fields` in the schema or fix the producer"
    )]
    FieldCapExceeded {
        /// The table's `max_fields`.
        cap: u32,
        /// Live fields before the addition.
        current: u32,
        /// The fields that would have been added.
        fields: Vec<String>,
    },
    /// A document nests deeper than the table allows.
    #[error("field `{path}` nests deeper than the table's cap of {cap} levels")]
    DepthExceeded {
        /// The table's depth cap.
        cap: u32,
        /// The field that nests past it.
        path: String,
    },
    /// A setting was given a value beyond what the engine admits.
    #[error("`{setting}` cannot be {value}; the most this engine admits is {cap}")]
    CapExceeded {
        /// What was being set.
        setting: String,
        /// The value submitted.
        value: u32,
        /// The largest value admitted.
        cap: u32,
    },
    /// A submitted type carries a parameter the engine cannot build an
    /// array from, such as a negative vector dimension.
    #[error("column `{column}` has an unusable type: {reason}")]
    InvalidType {
        /// The column's name.
        column: String,
        /// What is wrong with the type.
        reason: String,
    },
    /// A new column was submitted without a type.
    #[error("column `{column}` is new and needs a type")]
    TypeRequired {
        /// The column's name.
        column: String,
    },
    /// A document nests under a path the table already has as a column.
    /// Documents flatten to dot paths, so the nested value would become a
    /// second column beside the one that is there rather than filling it.
    #[error(
        "field `{path}` nests under live column `{column}`; a document cannot \
         add a column beneath one the table already has"
    )]
    PathShadowsColumn {
        /// The flattened path the document produced.
        path: String,
        /// The live column it would shadow.
        column: String,
    },

    /// A document carries a key a declared struct column does not have. A
    /// top-level key the table has not seen adds a column; a struct's fields
    /// are the table's, so an undeclared one is a mistake rather than a
    /// field to add, and dropping it would lose the value silently.
    #[error(
        "column `{column}` has no field `{field}`; a struct column's fields are \
         the ones it was declared with, so add it with a schema change first"
    )]
    UnknownStructField {
        /// The struct column.
        column: String,
        /// The key the document carried that it does not declare.
        field: String,
    },

    /// An element of an array of objects carries an array of its own. The
    /// leaves of an array of objects line up one position per element, so
    /// that reading one position across them reads one element; a nested
    /// array needs more than one position and would slide the leaves out of
    /// step with each other.
    #[error(
        "field `{path}` is an array inside an array of objects, which has no \
         single position per element to hold it"
    )]
    NestedArray {
        /// The flattened path of the offending leaf.
        path: String,
    },

    /// Two columns of a submitted Arrow schema carry the same name.
    #[error("duplicate column name: {name}")]
    DuplicateColumn {
        /// The repeated name.
        name: String,
    },

    /// A schema problem the engine does not classify further. The reason is
    /// the originating error's own message. New causes earn their own
    /// variant over time; matching this one means "schema problem, details
    /// only in the text".
    #[error("{reason}")]
    Invalid {
        /// What went wrong.
        reason: String,
    },

    /// A submitted name collides with a live column.
    #[error("the name `{name}` is taken by a live column")]
    NameTaken {
        /// The name in question.
        name: String,
    },
    /// A submitted id matches no live column.
    #[error("no live column has id {id}")]
    UnknownFieldId {
        /// The id in question.
        id: FieldId,
    },
    /// A change that needs an empty table on a table with rows.
    #[error("column `{column}` cannot become non-nullable while the table holds rows")]
    NotEmpty {
        /// The column's name.
        column: String,
    },
    /// A submitted identity attribute differs from the table's.
    #[error("`{attribute}` is part of the table's identity and cannot change; create a new table")]
    IdentityChange {
        /// What was asked to change.
        attribute: String,
    },
    /// Dropping or retyping the column the global vector index is built on.
    #[error("column `{column}` backs the table's vector index and cannot be dropped or retyped")]
    BacksGlobalVectorIndex {
        /// The column's name.
        column: String,
    },
    /// A second type change on a column whose conversion is outstanding.
    #[error("column `{column}` is still converting; run optimize before changing its type again")]
    ConversionInProgress {
        /// The column's name.
        column: String,
    },
    /// `expected_schema_id` did not match the current document.
    #[error("the schema is at version {current}, not {expected}; re-read it and resubmit")]
    SchemaConflict {
        /// The `schema_id` the caller expected.
        expected: u32,
        /// The `schema_id` the table is at.
        current: u32,
    },
    /// An index that cannot be built on the column it was given.
    #[error("column `{column}` cannot carry that index: {reason}")]
    InvalidIndex {
        /// The column's name.
        column: String,
        /// Why the index does not fit.
        reason: String,
    },
    /// A document that is not an object, or cannot be mapped to columns.
    #[error("row {row} cannot be mapped: {reason}")]
    InvalidRow {
        /// The row's position in the request.
        row: usize,
        /// What was wrong with it.
        reason: String,
    },
    /// `create_table` on a name that exists.
    #[error("table `{name}` already exists")]
    TableExists {
        /// The table's name.
        name: String,
    },
}

impl SchemaError {
    /// A stable name for the cause, for a caller that has to tell one refusal
    /// from another.
    ///
    /// The message says what went wrong and names the column, cap or version
    /// at fault; it is written for a person and may be reworded. This is the
    /// part a program matches on, so it is the variant's own name and does
    /// not change once shipped. The language bindings carry it across, since
    /// a caller there sees one exception class for every schema refusal and
    /// would otherwise have to match on prose.
    ///
    /// Exhaustive on purpose, with no fallback arm: the enum is
    /// `#[non_exhaustive]` to callers outside the crate, but inside it a new
    /// variant must name itself here rather than inherit someone else's name
    /// or a catch-all. A caller matching on these names should still keep a
    /// default branch, since a newer engine can send one this build has
    /// never seen.
    pub fn kind(&self) -> &'static str {
        match self {
            SchemaError::TypeMismatch { .. } => "TypeMismatch",
            SchemaError::MissingColumn { .. } => "MissingColumn",
            SchemaError::NullInNonNullable { .. } => "NullInNonNullable",
            SchemaError::ValueOutOfRange { .. } => "ValueOutOfRange",
            SchemaError::IntegerNotExactInFloat { .. } => "IntegerNotExactInFloat",
            SchemaError::MixedArray { .. } => "MixedArray",
            SchemaError::FieldCapExceeded { .. } => "FieldCapExceeded",
            SchemaError::DepthExceeded { .. } => "DepthExceeded",
            SchemaError::CapExceeded { .. } => "CapExceeded",
            SchemaError::InvalidType { .. } => "InvalidType",
            SchemaError::TypeRequired { .. } => "TypeRequired",
            SchemaError::PathShadowsColumn { .. } => "PathShadowsColumn",
            SchemaError::UnknownStructField { .. } => "UnknownStructField",
            SchemaError::NestedArray { .. } => "NestedArray",
            SchemaError::DuplicateColumn { .. } => "DuplicateColumn",
            SchemaError::Invalid { .. } => "Invalid",
            SchemaError::NameTaken { .. } => "NameTaken",
            SchemaError::UnknownFieldId { .. } => "UnknownFieldId",
            SchemaError::NotEmpty { .. } => "NotEmpty",
            SchemaError::IdentityChange { .. } => "IdentityChange",
            SchemaError::BacksGlobalVectorIndex { .. } => "BacksGlobalVectorIndex",
            SchemaError::ConversionInProgress { .. } => "ConversionInProgress",
            SchemaError::SchemaConflict { .. } => "SchemaConflict",
            SchemaError::InvalidIndex { .. } => "InvalidIndex",
            SchemaError::InvalidRow { .. } => "InvalidRow",
            SchemaError::TableExists { .. } => "TableExists",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One of every variant. Kept beside [`every_kind_is_the_variants_own_name`],
    /// which is what gives it a point: a sample that is wrong in any way the
    /// name does not depend on costs nothing.
    fn one_of_every_variant() -> Vec<SchemaError> {
        let column = || "c".to_string();
        vec![
            SchemaError::TypeMismatch {
                column: column(),
                frozen: DataType::Int64,
                offered: DataType::Utf8,
            },
            SchemaError::MissingColumn { column: column() },
            SchemaError::NullInNonNullable { column: column() },
            SchemaError::ValueOutOfRange {
                column: column(),
                value: "1".into(),
                data_type: "Int8".into(),
            },
            SchemaError::IntegerNotExactInFloat {
                column: column(),
                value: 1,
                stored: 1.0,
            },
            SchemaError::MixedArray {
                column: column(),
                types: vec!["number".into()],
            },
            SchemaError::FieldCapExceeded {
                cap: 1,
                current: 1,
                fields: vec![column()],
            },
            SchemaError::DepthExceeded {
                cap: 1,
                path: "a.b".into(),
            },
            SchemaError::CapExceeded {
                setting: "max_fields".into(),
                value: 2,
                cap: 1,
            },
            SchemaError::InvalidType {
                column: column(),
                reason: "r".into(),
            },
            SchemaError::TypeRequired { column: column() },
            SchemaError::PathShadowsColumn {
                path: "a.b".into(),
                column: column(),
            },
            SchemaError::UnknownStructField {
                column: column(),
                field: "f".into(),
            },
            SchemaError::NestedArray { path: "a.b".into() },
            SchemaError::DuplicateColumn { name: column() },
            SchemaError::Invalid { reason: "r".into() },
            SchemaError::NameTaken { name: column() },
            SchemaError::UnknownFieldId { id: FieldId(1) },
            SchemaError::NotEmpty { column: column() },
            SchemaError::IdentityChange {
                attribute: "id_column".into(),
            },
            SchemaError::BacksGlobalVectorIndex { column: column() },
            SchemaError::ConversionInProgress { column: column() },
            SchemaError::SchemaConflict {
                expected: 1,
                current: 2,
            },
            SchemaError::InvalidIndex {
                column: column(),
                reason: "r".into(),
            },
            SchemaError::InvalidRow {
                row: 0,
                reason: "r".into(),
            },
            SchemaError::TableExists { name: column() },
        ]
    }

    /// Exhaustive on purpose, and the reason [`one_of_every_variant`] can be
    /// trusted to be every variant: a variant added to the enum does not
    /// compile until it is listed here, and a reader who comes here to add an
    /// arm is standing next to the samples.
    fn _every_variant_has_a_sample(error: &SchemaError) {
        match error {
            SchemaError::TypeMismatch { .. }
            | SchemaError::MissingColumn { .. }
            | SchemaError::NullInNonNullable { .. }
            | SchemaError::ValueOutOfRange { .. }
            | SchemaError::IntegerNotExactInFloat { .. }
            | SchemaError::MixedArray { .. }
            | SchemaError::FieldCapExceeded { .. }
            | SchemaError::DepthExceeded { .. }
            | SchemaError::CapExceeded { .. }
            | SchemaError::InvalidType { .. }
            | SchemaError::TypeRequired { .. }
            | SchemaError::PathShadowsColumn { .. }
            | SchemaError::UnknownStructField { .. }
            | SchemaError::NestedArray { .. }
            | SchemaError::DuplicateColumn { .. }
            | SchemaError::Invalid { .. }
            | SchemaError::NameTaken { .. }
            | SchemaError::UnknownFieldId { .. }
            | SchemaError::NotEmpty { .. }
            | SchemaError::IdentityChange { .. }
            | SchemaError::BacksGlobalVectorIndex { .. }
            | SchemaError::ConversionInProgress { .. }
            | SchemaError::SchemaConflict { .. }
            | SchemaError::InvalidIndex { .. }
            | SchemaError::InvalidRow { .. }
            | SchemaError::TableExists { .. } => {}
        }
    }

    /// The variant's own name, taken from the derived `Debug`, which prints
    /// it ahead of the fields. Read rather than written down, so the check
    /// below compares `kind()` against the enum itself and not against a
    /// second list that could carry the same typo.
    fn variant_name(error: &SchemaError) -> String {
        let debug = format!("{error:?}");
        debug
            .split([' ', '{', '('])
            .next()
            .unwrap_or_default()
            .to_owned()
    }

    /// Every variant's kind is its own name. `kind()` is 26 hand-written
    /// arms and the contract is that the names never change once shipped, so
    /// a typo in any one of them would otherwise ship: the arms a reader
    /// happens to spot-check are the only ones a list of expected strings
    /// covers.
    #[test]
    fn every_kind_is_the_variants_own_name() {
        let samples = one_of_every_variant();
        for error in &samples {
            assert_eq!(
                error.kind(),
                variant_name(error),
                "kind() disagrees with the variant's name"
            );
        }
        // Distinct, or two variants a caller is meant to tell apart answer
        // the same name.
        let mut kinds: Vec<&str> = samples.iter().map(SchemaError::kind).collect();
        kinds.sort_unstable();
        let total = kinds.len();
        kinds.dedup();
        assert_eq!(kinds.len(), total, "two variants share a kind");
    }

    /// The kind is a name, not the message: it carries no column, cap or
    /// value, so it stays stable while the prose moves.
    #[test]
    fn the_kind_carries_none_of_the_message() {
        let error = SchemaError::NameTaken {
            name: "title".into(),
        };
        assert_eq!(error.kind(), "NameTaken");
        assert!(
            !error.kind().contains("title"),
            "the kind names the cause, not the instance"
        );
        assert!(error.to_string().contains("title"), "the message does");
    }
}
