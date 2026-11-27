// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SQL's view of a table's columns reports the types the table declares.
//!
//! The scan reads string columns without a full-text index as string views,
//! an internal choice for faster comparisons. `DESCRIBE` and
//! `information_schema.columns` must not show that choice: they report each
//! column as the table declares it, the same as the table's own schema.

use std::sync::Arc;

use arrow::util::display::{ArrayFormatter, FormatOptions};
use infino::{
    Connection, IndexSpec,
    arrow_array::{ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// The table every test here creates.
const TABLE: &str = "logs";

/// The engine's row id column, which SQL shows ahead of the declared columns.
const ID_COLUMN: &str = "_id";

/// A table with a full-text indexed string, an unindexed large string, an
/// unindexed string and an integer, holding one row.
fn seeded_table(dir: &TempDir) -> (Connection, Arc<Schema>) {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("resource", DataType::LargeUtf8, true),
        Field::new("service", DataType::Utf8, true),
        Field::new("count", DataType::Int64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec!["a line"])),
        Arc::new(LargeStringArray::from(vec![Some(r#"{"app":"beeper"}"#)])),
        Arc::new(StringArray::from(vec![Some("bridge")])),
        Arc::new(Int64Array::from(vec![1])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("batch");
    let logs = db
        .create_table(TABLE, schema.clone(), IndexSpec::new().fts("body"))
        .expect("create_table");
    logs.append(&batch, None).expect("append");
    (db, schema)
}

/// `(column, type)` pairs from the first two text columns of `sql`'s rows,
/// skipping the row id.
fn column_types(db: &Connection, sql: &str) -> Vec<(String, String)> {
    let batches = db.query_sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let options = FormatOptions::default();
    let mut out = Vec::new();
    for batch in &batches {
        let names = ArrayFormatter::try_new(batch.column(0), &options).expect("names");
        let types = ArrayFormatter::try_new(batch.column(1), &options).expect("types");
        for row in 0..batch.num_rows() {
            let name = names.value(row).to_string();
            if name != ID_COLUMN {
                out.push((name, types.value(row).to_string()));
            }
        }
    }
    out
}

/// `(column, type)` pairs as the table's schema declares them.
fn declared(schema: &Schema) -> Vec<(String, String)> {
    schema
        .fields()
        .iter()
        .map(|f| (f.name().clone(), f.data_type().to_string()))
        .collect()
}

#[test]
fn describe_reports_each_column_as_declared() {
    let dir = TempDir::new().expect("tempdir");
    let (db, schema) = seeded_table(&dir);
    assert_eq!(
        column_types(&db, &format!("DESCRIBE {TABLE}")),
        declared(&schema)
    );
}

#[test]
fn show_columns_reports_each_column_as_declared() {
    let dir = TempDir::new().expect("tempdir");
    let (db, schema) = seeded_table(&dir);
    // `SHOW COLUMNS` leads with the catalog, schema and table names; the
    // column name and its type follow.
    let batches = db
        .query_sql(&format!("SHOW COLUMNS FROM {TABLE}"))
        .expect("show columns");
    let options = FormatOptions::default();
    let mut shown = Vec::new();
    for batch in &batches {
        let names = ArrayFormatter::try_new(
            batch.column_by_name("column_name").expect("column_name"),
            &options,
        )
        .expect("names");
        let types = ArrayFormatter::try_new(
            batch.column_by_name("data_type").expect("data_type"),
            &options,
        )
        .expect("types");
        for row in 0..batch.num_rows() {
            let name = names.value(row).to_string();
            if name != ID_COLUMN {
                shown.push((name, types.value(row).to_string()));
            }
        }
    }
    assert_eq!(shown, declared(&schema));
}

#[test]
fn information_schema_reports_each_column_as_declared() {
    let dir = TempDir::new().expect("tempdir");
    let (db, schema) = seeded_table(&dir);
    assert_eq!(
        column_types(
            &db,
            &format!(
                "SELECT column_name, data_type FROM information_schema.columns \
                 WHERE table_name = '{TABLE}' ORDER BY ordinal_position"
            ),
        ),
        declared(&schema)
    );
}
