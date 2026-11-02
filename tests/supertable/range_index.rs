// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Range index end to end, through the public API: a table that
//! range-indexes its numeric columns answers every range shape exactly as
//! the same rows in an unindexed table do — across several superfiles,
//! after a delete, after compaction, and after the connection is reopened.
//!
//! The unindexed table is the oracle: it answers by scanning. Every shape
//! the range index serves is here — a count (answered from the index alone),
//! a bucketed count (`COUNT(*) FILTER (WHERE range)` per bucket, likewise),
//! a range's rows, `BETWEEN`, two conjuncts folded into one interval, an
//! `ORDER BY ... LIMIT` the index serves in order, integer, float and date
//! columns — beside shapes it must leave to the scan: two columns, an `OR`.
//! A range-indexed column is found by its id, so a rename keeps its index
//! and a name dropped and reused for a new column does not inherit it.

use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use arrow_array::{Array, Date32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    ColumnIndex, Connection, FieldPatch, IndexSpec, OptimizeOptions, SchemaPatch, connect,
};
use tempfile::TempDir;

/// Rows in each committed superfile.
const ROWS_PER_FILE: i64 = 3000;
/// Superfiles each table is written as, one commit each.
const FILES: i64 = 4;
/// Superfiles of one row each in the one-row fixture.
const ONE_ROW_FILES: usize = 3;
/// Every this-many rows `qty` is null.
const QTY_NULL_EVERY: i64 = 17;
/// The first `day`: 2022-01-08 as days since the epoch.
const FIRST_DAY: i32 = 19_000;
/// Distinct days the rows spread over.
const DAYS: i64 = 365;
/// Rows the delete removes: those with `seq` below this.
const DELETED_BELOW: i64 = 500;
/// The physical operator a scan's rows pass through when a filter is
/// checked on them; a count the range index answers has none.
const FILTER_OPERATOR: &str = "FilterExec";

/// The shapes both tables answer, `{t}` standing for the table. Each is
/// ordered so its rows compare as text, and a top-k selects only the sort
/// column, as rows tied on it may come back in either order.
const QUERIES: &[&str] = &[
    "SELECT count(*) AS n FROM {t} WHERE qty > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty >= -10 AND qty < 10",
    "SELECT count(*) AS n FROM {t} WHERE qty BETWEEN -5 AND 5",
    "SELECT count(*) AS n FROM {t} WHERE qty = 7",
    "SELECT count(*) AS n FROM {t} WHERE qty < -1000",
    "SELECT count(*) AS n FROM {t} WHERE 100 < qty",
    "SELECT count(*) AS n FROM {t} WHERE seq >= 1000 AND seq < 1100",
    "SELECT count(*) AS n FROM {t} WHERE price > 990.5",
    "SELECT seq FROM {t} WHERE price > 990.5 ORDER BY seq",
    "SELECT seq FROM {t} WHERE price BETWEEN 100 AND 101 ORDER BY seq",
    "SELECT seq, qty FROM {t} WHERE qty = 0 ORDER BY seq",
    "SELECT qty FROM {t} WHERE qty > 200 ORDER BY qty LIMIT 10",
    "SELECT qty FROM {t} WHERE qty < -200 ORDER BY qty DESC LIMIT 10",
    "SELECT price FROM {t} WHERE price < 50 ORDER BY price DESC LIMIT 5",
    "SELECT qty FROM {t} ORDER BY qty LIMIT 3",
    "SELECT seq FROM {t} WHERE day >= DATE '2022-06-01' AND day < DATE '2022-06-03' ORDER BY seq",
    "SELECT sum(seq) AS s FROM {t} WHERE qty > 0 AND price > 500",
    "SELECT count(*) AS n FROM {t} WHERE qty > 240 OR price < 1",
    "SELECT seq FROM {t} WHERE qty > 240 AND seq < 3000 ORDER BY seq",
    BUCKETED_QTY,
    "SELECT count(*) FILTER (WHERE qty < 0) AS low, count(*) FILTER (WHERE qty >= 0 AND qty < 200) AS high FROM {t} WHERE qty > -200",
    "SELECT count(*) AS n, count(*) FILTER (WHERE qty > 100) AS big FROM {t} WHERE qty > 0",
    BUCKETED_DAY,
    "SELECT count(*) FILTER (WHERE price < 250) AS a, count(*) FILTER (WHERE price >= 250 AND price < 750.5) AS b, count(*) FILTER (WHERE price >= 750.5) AS c FROM {t}",
    "SELECT count(*) FILTER (WHERE qty > 0) AS a, count(*) FILTER (WHERE price > 500) AS b FROM {t} WHERE seq < 6000",
    "SELECT count(*) FILTER (WHERE qty > 0) AS a, count(*) FILTER (WHERE qty > 100 OR qty < -100) AS b FROM {t}",
];

/// A bucketed range aggregation in one statement, its buckets overlapping
/// as a range aggregation's may.
const BUCKETED_QTY: &str = "SELECT count(*) FILTER (WHERE qty < -100) AS a, count(*) FILTER (WHERE qty >= -100 AND qty < 0) AS b, count(*) FILTER (WHERE qty >= 0 AND qty < 100) AS c, count(*) FILTER (WHERE qty >= 50) AS d FROM {t}";
/// Date buckets, quarter by quarter.
const BUCKETED_DAY: &str = "SELECT count(*) FILTER (WHERE day < DATE '2022-04-01') AS q1, count(*) FILTER (WHERE day >= DATE '2022-04-01' AND day < DATE '2022-07-01') AS q2, count(*) FILTER (WHERE day >= DATE '2022-07-01') AS rest FROM {t}";
/// How a plan shows an aggregate that checks its own filter on each row; a
/// bucketed count the range index answers has none.
const BUCKET_FILTER: &str = "FILTER (WHERE";

/// The shapes both tables answer once `qty` is renamed `amount` and a new
/// `qty` is added: the range index follows `amount`, and the new `qty`
/// holds only the rows written after it.
const RENAMED_QUERIES: &[&str] = &[
    "SELECT count(*) AS n FROM {t} WHERE amount > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty BETWEEN -5 AND 5",
    "SELECT seq FROM {t} WHERE qty BETWEEN -5 AND 5 ORDER BY seq",
    "SELECT amount FROM {t} WHERE amount > 200 ORDER BY amount LIMIT 10",
    "SELECT qty FROM {t} WHERE qty < -150 ORDER BY qty LIMIT 10",
];

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("qty", DataType::Int32, true),
        Field::new("price", DataType::Float64, false),
        Field::new("day", DataType::Date32, false),
    ]))
}

/// Rows `first..first + ROWS_PER_FILE`: `qty` spread over `-250..250` with
/// nulls, `price` over `0..1000`, `day` over a year — none of them in the
/// order the rows arrive.
fn rows(first: i64) -> RecordBatch {
    let seq: Vec<i64> = (first..first + ROWS_PER_FILE).collect();
    let qty: Vec<Option<i32>> = seq
        .iter()
        .map(|i| (i % QTY_NULL_EVERY != 0).then(|| ((i * 37) % 500 - 250) as i32))
        .collect();
    let price: Vec<f64> = seq
        .iter()
        .map(|i| ((i * 7919) % 10_000) as f64 / 10.0)
        .collect();
    let day: Vec<i32> = seq
        .iter()
        .map(|i| FIRST_DAY + ((i * 13) % DAYS) as i32)
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(seq)),
            Arc::new(Int32Array::from(qty)),
            Arc::new(Float64Array::from(price)),
            Arc::new(Date32Array::from(day)),
        ],
    )
    .expect("batch")
}

/// The two tables, each written as [`FILES`] superfiles of the same rows.
fn create(db: &Connection) {
    let plain = db
        .create_table("plain", schema(), IndexSpec::new())
        .expect("create plain");
    let ranged = db
        .create_table(
            "ranged",
            schema(),
            IndexSpec::new()
                .range("seq")
                .range("qty")
                .range("price")
                .range("day"),
        )
        .expect("create ranged");
    for file in 0..FILES {
        let batch = rows(file * ROWS_PER_FILE);
        plain.append(&batch).expect("append plain");
        ranged.append(&batch).expect("append ranged");
    }
}

/// `query` on `table`, as text.
fn answer(db: &Connection, query: &str, table: &str) -> String {
    let batches = db
        .query_sql(&query.replace("{t}", table))
        .unwrap_or_else(|e| panic!("{query} on {table}: {e}"));
    pretty_format_batches(&batches).expect("format").to_string()
}

/// Every one of `queries`' answers from both tables, the same.
fn assert_same_answers(db: &Connection, queries: &[&str], when: &str) {
    for query in queries {
        assert_eq!(
            answer(db, query, "ranged"),
            answer(db, query, "plain"),
            "{when}: {query}"
        );
    }
}

#[test]
fn a_range_indexed_table_answers_as_a_scan_does_through_deletes_compaction_and_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf-8 path");
    let db = connect(path).expect("connect");
    create(&db);
    assert_same_answers(&db, QUERIES, "after load");

    // A count over a range-indexed column reads no row to check the range,
    // on an integer and on a float column alike.
    for query in [
        "EXPLAIN SELECT count(*) FROM {t} WHERE qty >= -10 AND qty < 10",
        "EXPLAIN SELECT count(*) FROM {t} WHERE price > 990.5",
    ] {
        assert!(
            answer(&db, query, "plain").contains(FILTER_OPERATOR),
            "{query}"
        );
        assert!(
            !answer(&db, query, "ranged").contains(FILTER_OPERATOR),
            "{query}"
        );
    }
    // Nor does a bucketed one check any bucket's range on a row.
    for bucketed in [BUCKETED_QTY, BUCKETED_DAY] {
        let query = format!("EXPLAIN {bucketed}");
        assert!(
            answer(&db, &query, "plain").contains(BUCKET_FILTER),
            "{query}"
        );
        let ranged = answer(&db, &query, "ranged");
        assert!(
            !ranged.contains(BUCKET_FILTER) && !ranged.contains(FILTER_OPERATOR),
            "{query}: {ranged}"
        );
    }

    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
    }
    assert_same_answers(&db, QUERIES, "after a delete");

    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .optimize(&OptimizeOptions::default())
            .expect("optimize");
    }
    assert_same_answers(&db, QUERIES, "after compaction");

    drop(db);
    let reopened = connect(path).expect("reconnect");
    assert_same_answers(&reopened, QUERIES, "after a reopen");
}

#[test]
fn a_range_index_follows_its_column_through_a_rename_and_not_into_a_reused_name() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    create(&db);
    for table in ["plain", "ranged"] {
        let before = db
            .apply_schema(table, &SchemaPatch::new(vec![]), None)
            .expect("read the schema");
        let qty = before.id_of("qty").expect("qty");
        db.apply_schema(
            table,
            &SchemaPatch::new(vec![FieldPatch::named("amount").with_id(qty)]),
            None,
        )
        .expect("rename qty");
        let mut new_qty = FieldPatch::named("qty").with_type(DataType::Int32);
        if table == "ranged" {
            new_qty = new_qty.with_index(ColumnIndex::Range);
        }
        let after = db
            .apply_schema(table, &SchemaPatch::new(vec![new_qty]), None)
            .expect("add a new qty");
        assert_ne!(after.id_of("qty"), Some(qty), "a new column takes a new id");
    }
    // One more superfile, with values in both columns.
    let first = FILES * ROWS_PER_FILE;
    let old = rows(first);
    let new_qty: Int32Array = (first..first + ROWS_PER_FILE)
        .map(|i| Some(((i * 11) % 400 - 200) as i32))
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("amount", DataType::Int32, true),
        Field::new("price", DataType::Float64, false),
        Field::new("day", DataType::Date32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let mut columns = old.columns().to_vec();
    columns.push(Arc::new(new_qty) as Arc<dyn Array>);
    let batch = RecordBatch::try_new(schema, columns).expect("batch");
    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .append(&batch)
            .expect("append");
    }
    assert_same_answers(&db, RENAMED_QUERIES, "after a rename and a reused name");
}

/// Superfiles of a single row each, as a writer committing one document at
/// a time leaves them, answer as the scan does.
#[test]
fn one_row_superfiles_answer_as_a_scan_does() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let plain = db
        .create_table("plain", schema(), IndexSpec::new())
        .expect("create plain");
    let ranged = db
        .create_table(
            "ranged",
            schema(),
            IndexSpec::new()
                .range("seq")
                .range("qty")
                .range("price")
                .range("day"),
        )
        .expect("create ranged");
    let batch = rows(0);
    for row in 0..ONE_ROW_FILES {
        let one = batch.slice(row, 1);
        plain.append(&one).expect("append plain");
        ranged.append(&one).expect("append ranged");
    }
    assert_same_answers(&db, QUERIES, "over one-row superfiles");
}

#[test]
fn a_range_index_on_a_column_it_cannot_key_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::LargeUtf8, false),
        Field::new("n", DataType::Int64, false),
    ]));
    for spec in [
        IndexSpec::new().range("name"),
        IndexSpec::new().range("missing"),
        IndexSpec::new().range("n").range("n"),
        IndexSpec::new().fts("name").range("name"),
    ] {
        assert!(
            db.create_table("t", Arc::clone(&schema), spec).is_err(),
            "the table must not be created"
        );
    }
    db.create_table("t", schema, IndexSpec::new().range("n"))
        .expect("an integer column takes a range index");
}
