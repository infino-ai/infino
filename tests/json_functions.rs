// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! JSON accessors over text columns that hold a JSON document.
//!
//! The fixture is shaped like a log table: a body, and a `resource` column
//! whose JSON keys vary by row and themselves contain dots. The same document
//! is stored under each string type a table can declare, so every accessor is
//! checked against all of them.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow::util::display::{ArrayFormatter, FormatOptions};
use infino::{
    Connection, IndexSpec,
    arrow_array::{
        Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch, StringArray, StringViewArray,
    },
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// The table name every test in this file creates.
const TABLE: &str = "logs";

/// The JSON documents, one per row. The last row is not JSON at all, and the
/// third lacks the image key the others carry.
const RESOURCES: [Option<&str>; 5] = [
    Some(
        r#"{"app":"open-webui","container.image.name":"ghcr.io/open-webui/open-webui","k8s.container.restart_count":"0","limits":{"cpu":2,"burst":true}}"#,
    ),
    Some(
        r#"{"app":"beeper","container.image.name":"ghcr.io/beeper/bridge-manager","k8s.container.restart_count":"3","limits":{"cpu":4,"burst":false},"network":"[{\"name\":\"cbr0\",\"ips\":[\"10.244.0.177\"]}]"}"#,
    ),
    Some(r#"{"app":"beeper","limits":{"cpu":1}}"#),
    None,
    Some("not json"),
];

/// Rows whose image name ends in `open-webui`.
const OPEN_WEBUI_ROWS: i64 = 1;

/// Rows that carry an image name at all.
const ROWS_WITH_IMAGE: i64 = 2;

/// Rows whose `app` is `beeper`.
const BEEPER_ROWS: i64 = 2;

/// Sum of `limits.cpu` across the rows that carry it.
const TOTAL_CPU: i64 = 7;

/// The string columns holding the same documents, one per string type.
const JSON_COLUMNS: [&str; 3] = ["resource_large", "resource_utf8", "resource_view"];

/// Create the table and append the fixture rows.
fn seeded_table(dir: &TempDir) -> Connection {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("resource_large", DataType::LargeUtf8, true),
        Field::new("resource_utf8", DataType::Utf8, true),
        Field::new("resource_view", DataType::Utf8View, true),
    ]));
    let bodies: Vec<String> = (0..RESOURCES.len()).map(|i| format!("line {i}")).collect();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(bodies)),
        Arc::new(LargeStringArray::from_iter(RESOURCES)),
        Arc::new(StringArray::from_iter(RESOURCES)),
        Arc::new(StringViewArray::from_iter(RESOURCES)),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("batch");
    let logs = db
        .create_table(TABLE, schema, IndexSpec::new().fts("body"))
        .expect("create_table");
    logs.append(&batch).expect("append");
    db
}

/// The single integer in a one-row, one-column result.
fn only_i64(db: &Connection, sql: &str) -> i64 {
    let batches = db.query_sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    batches
        .iter()
        .filter(|b| b.num_rows() > 0)
        .map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap_or_else(|| panic!("{sql}: column is {}", b.column(0).data_type()))
                .value(0)
        })
        .next()
        .unwrap_or_else(|| panic!("{sql}: no rows"))
}

/// Every value of the first column, rendered as text, in row order.
fn first_column(db: &Connection, sql: &str) -> Vec<String> {
    let batches = db.query_sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let options = FormatOptions::default().with_null("NULL");
    batches
        .iter()
        .flat_map(|b| {
            let column = b.column(0);
            let formatter = ArrayFormatter::try_new(column, &options).expect("formatter");
            (0..column.len())
                .map(|i| formatter.value(i).to_string())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_key_filters_rows_on_every_string_type() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let rows = only_i64(
            &db,
            &format!(
                "SELECT count(*) FROM {TABLE} \
                 WHERE json_get_str({column}, 'container.image.name') LIKE '%open-webui'"
            ),
        );
        assert_eq!(rows, OPEN_WEBUI_ROWS, "{column}");
    }
}

#[test]
fn a_key_groups_rows_on_every_string_type() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let groups = first_column(
            &db,
            &format!(
                "SELECT concat(app, '=', n) FROM (\
                   SELECT json_get_str({column}, 'app') AS app, count(*) AS n FROM {TABLE} \
                   WHERE json_get_str({column}, 'app') IS NOT NULL \
                   GROUP BY json_get_str({column}, 'app')\
                 ) ORDER BY app"
            ),
        );
        assert_eq!(
            groups,
            vec![format!("beeper={BEEPER_ROWS}"), "open-webui=1".to_string()],
            "{column}"
        );
    }
}

#[test]
fn a_missing_key_a_null_and_a_non_json_value_read_as_null() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let rows = only_i64(
            &db,
            &format!(
                "SELECT count(*) FROM {TABLE} \
                 WHERE json_get_str({column}, 'container.image.name') IS NOT NULL"
            ),
        );
        assert_eq!(rows, ROWS_WITH_IMAGE, "{column}");
    }
}

#[test]
fn a_nested_path_reads_a_typed_value() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let total = only_i64(
            &db,
            &format!("SELECT sum(json_get_int({column}, 'limits', 'cpu')) FROM {TABLE}"),
        );
        assert_eq!(total, TOTAL_CPU, "{column}");
    }
}

#[test]
fn the_arrow_operators_read_a_key() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let rows = only_i64(
            &db,
            &format!("SELECT count(*) FROM {TABLE} WHERE ({column}->>'app') = 'beeper'"),
        );
        assert_eq!(rows, BEEPER_ROWS, "{column} ->>");
        let total = only_i64(
            &db,
            &format!("SELECT sum(json_get_int({column}->'limits', 'cpu')) FROM {TABLE}"),
        );
        assert_eq!(total, TOTAL_CPU, "{column} ->");
    }
}

#[test]
fn json_held_as_a_string_inside_json_reads_through_a_second_call() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);
    for column in JSON_COLUMNS {
        let names = first_column(
            &db,
            &format!(
                "SELECT json_get_str(json_get_str({column}, 'network'), 0, 'name') FROM {TABLE} \
                 WHERE json_contains({column}, 'network')"
            ),
        );
        assert_eq!(names, vec!["cbr0".to_string()], "{column}");
    }
}
