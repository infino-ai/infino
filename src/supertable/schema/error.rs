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
