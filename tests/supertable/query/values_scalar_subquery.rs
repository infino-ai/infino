// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A scalar subquery inside a `VALUES` list, through the public
//! `Connection::query_sql`.
//!
//! DataFusion 54 evaluates a `VALUES` list's cells while it plans the
//! query, but runs uncorrelated scalar subqueries only once the plan
//! executes; a subquery in a `VALUES` cell therefore failed every such
//! statement with the internal error "ScalarSubqueryExpr evaluated before
//! the subquery was executed". These pin that the list now plans as
//! one-row projections and returns the subquery's value, in the shapes an
//! agent writes: a bare list, a labelled multi-row list mixing constants and
//! subqueries, a `WITH` list, and a list used as a scalar subquery itself.

use std::sync::Arc;

use infino::{
    Connection, IndexSpec,
    arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use tempfile::TempDir;

/// Rows committed per append; two appends make two superfiles.
const FIRST: &[(&str, i64)] = &[
    ("the quick brown fox", 1),
    ("a lazy dog", 2),
    ("red fox", 3),
];
/// Second commit.
const SECOND: &[(&str, i64)] = &[("grey wolf", 4), ("fox and wolf", 5), ("nothing", 6)];
/// Rows across both commits.
const TOTAL_ROWS: i64 = 6;
/// Largest `n` in the corpus.
const MAX_N: i64 = 6;
/// Rows whose title holds `fox`.
const FOX_ROWS: i64 = 3;
/// The constant row in the mixed list.
const CONST_VALUE: i64 = 7;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("n", DataType::Int64, false),
    ]))
}

fn batch(rows: &[(&str, i64)]) -> RecordBatch {
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(LargeStringArray::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("batch")
}

fn corpus(dir: &TempDir) -> Connection {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let docs = db
        .create_table("docs", schema(), IndexSpec::new().fts("title"))
        .expect("create_table");
    docs.append(&batch(FIRST)).expect("append 1");
    docs.append(&batch(SECOND)).expect("append 2");
    db
}

/// Every `(label, value)` row of a two-column result, in result order.
fn label_values(batches: &[RecordBatch]) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    for b in batches {
        let labels = b.column(0);
        let values = b
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("value column is Int64");
        for i in 0..b.num_rows() {
            let label = match labels.as_any().downcast_ref::<LargeStringArray>() {
                Some(a) => a.value(i).to_string(),
                None => labels
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("label column is a string")
                    .value(i)
                    .to_string(),
            };
            out.push((label, values.value(i)));
        }
    }
    out
}

/// The single Int64 value of a one-row, one-column result.
fn single_i64(batches: &[RecordBatch]) -> i64 {
    let rows: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let col = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64 column");
            (0..col.len()).map(|i| col.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(rows.len(), 1, "one row");
    rows[0]
}

#[test]
fn values_cell_scalar_subquery_returns_its_value() {
    let dir = TempDir::new().expect("tempdir");
    let db = corpus(&dir);

    let bare = db
        .query_sql("SELECT * FROM (VALUES ((SELECT COUNT(*) FROM docs))) AS v(c)")
        .expect("bare VALUES with a scalar subquery");
    assert_eq!(single_i64(&bare), TOTAL_ROWS);

    let nested = db
        .query_sql(
            "SELECT n FROM docs WHERE n = (SELECT x FROM (VALUES ((SELECT MAX(n) FROM docs))) AS v(x))",
        )
        .expect("VALUES inside a scalar subquery");
    assert_eq!(single_i64(&nested), MAX_N);
}

#[test]
fn values_rows_mixing_constants_and_subqueries() {
    let dir = TempDir::new().expect("tempdir");
    let db = corpus(&dir);

    let mixed = db
        .query_sql(
            "SELECT * FROM (VALUES ('total', (SELECT COUNT(*) FROM docs)), \
             ('max', (SELECT MAX(n) FROM docs)), ('const', 7)) AS v(k, x) ORDER BY k",
        )
        .expect("multi-row VALUES with scalar subqueries");
    assert_eq!(
        label_values(&mixed),
        vec![
            ("const".to_string(), CONST_VALUE),
            ("max".to_string(), MAX_N),
            ("total".to_string(), TOTAL_ROWS),
        ]
    );

    let cte = db
        .query_sql(
            "WITH terms(t, c) AS (VALUES \
             ('fox', (SELECT COUNT(*) FROM docs WHERE title LIKE '%fox%')), \
             ('all', (SELECT COUNT(*) FROM docs))) \
             SELECT t, c FROM terms ORDER BY t",
        )
        .expect("WITH ... VALUES with scalar subqueries");
    assert_eq!(
        label_values(&cte),
        vec![
            ("all".to_string(), TOTAL_ROWS),
            ("fox".to_string(), FOX_ROWS)
        ]
    );
}

#[test]
fn constant_values_list_is_unchanged() {
    let dir = TempDir::new().expect("tempdir");
    let db = corpus(&dir);

    // No subquery: the list stays a `VALUES` source, rows in list order.
    let constants = db
        .query_sql("SELECT * FROM (VALUES ('b', 2), ('a', 1)) AS v(k, x)")
        .expect("constant VALUES");
    assert_eq!(
        label_values(&constants),
        vec![("b".to_string(), 2), ("a".to_string(), 1)]
    );
}
