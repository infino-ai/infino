// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Where a `WHERE` runs: as a Parquet row filter in a scan under
//! `ORDER BY ... LIMIT`, in a `FilterExec` above the scan everywhere else.
//!
//! `EXPLAIN ANALYZE` tells which. A row filter counts the rows it checks in
//! `pushdown_rows_matched` and `pushdown_rows_pruned`; without one both are 0.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{
    Array, Int64Array, LargeStringArray, ListArray, RecordBatch, StringArray, types::Int64Type,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::prelude::{col, lit};
use infino::{ConnectOptions, Connection, Consistency, IndexSpec, Supertable, connect_with};
use tempfile::TempDir;

/// Rows per append. Two appends make two superfiles.
const ROWS_PER_APPEND: i64 = 20_000;
/// Rows the queries ask for.
const FETCH: usize = 5;
/// The digit the `LIKE` filter looks for in `s`.
const DIGIT: char = '7';
/// Spreads `n` over the rows so the sort column is not already in order.
const SHUFFLE: i64 = 7_919;
/// Pads `s` so it is a wide column, like real text, many times the size of `n`.
/// Has no digits, so it never matches the `LIKE`.
const PAD: &str =
    "--------------------------------------------------------------------------------";
/// The metrics `EXPLAIN ANALYZE` prints for rows a row filter kept and dropped.
const ROW_FILTER_METRICS: [&str; 2] = ["pushdown_rows_matched=", "pushdown_rows_pruned="];

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("m", DataType::Int64, false),
        Field::new("s", DataType::LargeUtf8, false),
        Field::new(
            "l",
            DataType::List(Arc::new(Field::new_list_field(DataType::Int64, true))),
            false,
        ),
    ]))
}

/// Rows `n` in `[lo, lo + ROWS_PER_APPEND)` in shuffled order, `m = n % 10`,
/// `s = "r<n>" + PAD`, `l = [n]`. `l` is a list, which a row filter can't read.
fn batch(lo: i64) -> RecordBatch {
    let n: Vec<i64> = (0..ROWS_PER_APPEND)
        .map(|i| lo + (i * SHUFFLE) % ROWS_PER_APPEND)
        .collect();
    let m: Vec<i64> = n.iter().map(|v| v % 10).collect();
    let s: Vec<String> = n.iter().map(|v| format!("r{v}{PAD}")).collect();
    let l = ListArray::from_iter_primitive::<Int64Type, _, _>(n.iter().map(|v| Some([Some(*v)])));
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(n)),
            Arc::new(Int64Array::from(m)),
            Arc::new(LargeStringArray::from(s)),
            Arc::new(l),
        ],
    )
    .expect("valid batch")
}

/// A no-index table `t` of two superfiles, `n` from 0 to 2 * ROWS_PER_APPEND.
fn fixture() -> (TempDir, Connection, Supertable) {
    fixture_with(IndexSpec::new())
}

/// The same table with `index` on it.
fn fixture_with(index: IndexSpec) -> (TempDir, Connection, Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let db = connect_with(
        dir.path().join("db").to_str().expect("utf-8 path"),
        ConnectOptions::new().with_read_consistency(Consistency::Strong),
    )
    .expect("connect");
    let table = db.create_table("t", schema(), index).expect("create_table");
    table.append(&batch(0)).expect("append");
    table.append(&batch(ROWS_PER_APPEND)).expect("append");
    (dir, db, table)
}

/// Column `n` of every result row, in order.
fn n_values(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let n = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("n column")
                .clone();
            (0..n.len()).map(move |i| n.value(i))
        })
        .collect()
}

/// The `n` an `ORDER BY n LIMIT FETCH` over rows whose `s` has `DIGIT` returns,
/// skipping `deleted`.
fn expected(descending: bool, deleted: &[i64]) -> Vec<i64> {
    let mut n: Vec<i64> = (0..2 * ROWS_PER_APPEND)
        .filter(|v| format!("r{v}").contains(DIGIT) && !deleted.contains(v))
        .collect();
    if descending {
        n.reverse();
    }
    n.truncate(FETCH);
    n
}

/// The plan text `EXPLAIN <options> sql` prints.
fn explain(db: &Connection, options: &str, sql: &str) -> String {
    let batches = db
        .query_sql(&format!("EXPLAIN {options} {sql}"))
        .expect("explain");
    batches
        .iter()
        .flat_map(|b| {
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len()).map(move |i| plans.value(i).to_owned())
        })
        .collect()
}

/// Whether a scan in `sql`'s plan checked rows with a row filter. It need not
/// drop any: here a row group can end before the sort has a cutoff.
fn row_filter_ran(db: &Connection, sql: &str) -> bool {
    let plan = explain(db, "ANALYZE", sql);
    ROW_FILTER_METRICS.iter().any(|metric| {
        plan.split(metric)
            .skip(1)
            .any(|rest| rest.split([',', ']']).next() != Some("0"))
    })
}

#[test]
fn order_by_limit_runs_its_filter_inside_the_scan() {
    // `WHERE s LIKE '%7%' ORDER BY n LIMIT 5`, ascending and descending.
    //  - the rule marks the scan, and DataFusion moves the `WHERE` into it.
    //  - the `FilterExec` is gone: it would hold rows back from the sort.
    // This test pins the plan shape and the rows.
    let (_dir, db, _table) = fixture();
    for (order, descending) in [("ASC", false), ("DESC", true)] {
        let sql =
            format!("SELECT n FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY n {order} LIMIT {FETCH}");
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got, expected(descending, &[]), "{sql}");
        assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
        let plan = explain(&db, "", &sql);
        assert!(!plan.contains("FilterExec"), "{plan}");
    }
}

#[test]
fn a_filter_without_limit_stays_above_the_scan() {
    // `COUNT(*) WHERE s LIKE '%7%'`, no `ORDER BY ... LIMIT`.
    //  - no cutoff, so the `WHERE` stays in a `FilterExec` above the scan.
    // This test pins the count and that no row filter runs.
    let (_dir, db, _table) = fixture();
    let sql = format!("SELECT COUNT(*) FROM t WHERE s LIKE '%{DIGIT}%'");
    let batches = db.query_sql(&sql).expect("sql");
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count")
        .value(0);
    let want = (0..2 * ROWS_PER_APPEND)
        .filter(|v| format!("r{v}").contains(DIGIT))
        .count();
    assert_eq!(count as usize, want);
    assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
    let plan = explain(&db, "", &sql);
    assert!(plan.contains("FilterExec"), "{plan}");
}

#[test]
fn a_join_under_order_by_limit_gets_no_row_filter() {
    // A self-join under `ORDER BY ... LIMIT`.
    //  - the walk stops at the join; a join's own filter is slower as a row filter.
    // This test pins the rows and that neither scan gets a row filter.
    let (_dir, db, _table) = fixture();
    let sql = format!(
        "SELECT a.n FROM t a JOIN t b ON a.n = b.n WHERE a.s LIKE '%{DIGIT}%' ORDER BY a.n LIMIT {FETCH}"
    );
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    assert_eq!(got, expected(false, &[]));
    assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
}

#[test]
fn deleted_rows_stay_out_under_the_row_filter() {
    // Delete the two smallest matching rows, then `ORDER BY n LIMIT 5`.
    //  - the scan skips deleted rows first; the row filter only drops more.
    // This test pins that the deleted rows never come back.
    let (_dir, db, table) = fixture();
    let deleted: Vec<i64> = expected(false, &[]).into_iter().take(2).collect();
    for n in &deleted {
        table.delete(col("n").eq(lit(*n))).expect("delete");
    }
    let sql = format!("SELECT n FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY n LIMIT {FETCH}");
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    assert_eq!(got, expected(false, &deleted));
    assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
}

#[test]
fn order_by_limit_without_where_runs_the_cutoff_inside_the_scan() {
    // `SELECT n, s ORDER BY n LIMIT 5`, no `WHERE`.
    //  - the row filter is the sort's cutoff alone.
    // This test pins the rows and that the row filter runs.
    let (_dir, db, _table) = fixture();
    let sql = format!("SELECT n, s FROM t ORDER BY n LIMIT {FETCH}");
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    assert_eq!(got, (0..FETCH as i64).collect::<Vec<_>>());
    assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
}

#[test]
fn a_computed_column_between_sort_and_scan_keeps_the_row_filter() {
    // `SELECT n, s || 'x'` puts a projection between the sort and the scan.
    //  - a projection passes rows on one at a time, so the walk goes through it.
    // This test pins the rows and that the row filter runs.
    let (_dir, db, _table) = fixture();
    let sql = format!(
        "SELECT n, s || 'x' AS sx FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY n LIMIT {FETCH}"
    );
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    assert_eq!(got, expected(false, &[]));
    assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
}

#[test]
fn a_computed_sort_key_is_costed_on_the_columns_it_reads() {
    // `WHERE s LIKE '%7%' ORDER BY <computed> LIMIT 5`.
    //  - `length(s), n` reads `s` and `n`: nothing left to skip, no row filter.
    //  - `n + 0 DESC` reads `n`: `s` is left to skip, row filter on.
    //  - `random()` reads no column: no cutoff reaches the scan, no row filter.
    // This test pins the rows and that the cost check counts scan columns.
    let (_dir, db, _table) = fixture();
    for (sql, want, ran) in [
        (
            format!(
                "SELECT n, length(s) AS l FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY l, n LIMIT {FETCH}"
            ),
            vec![7, 17, 27, 37, 47],
            false,
        ),
        (
            format!(
                "SELECT n + 0 AS k, s FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY k DESC LIMIT {FETCH}"
            ),
            expected(true, &[]),
            true,
        ),
    ] {
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got, want, "{sql}");
        assert_eq!(row_filter_ran(&db, &sql), ran, "{sql}");
    }
    let sql =
        format!("SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY random() LIMIT {FETCH}");
    assert_eq!(n_values(&db.query_sql(&sql).expect("sql")).len(), FETCH);
    assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
}

#[test]
fn a_conjunct_the_scan_cant_take_gets_no_row_filter() {
    // `WHERE s LIKE '%7%' AND <x> ORDER BY n LIMIT 5`, `<x>` can't move into
    // the scan.
    //  - `random() >= 0` is volatile; `array_length(l) = 1` reads a list.
    //  - `<x>` stays in a `FilterExec`, which holds rows back from the sort.
    //  - so the rule leaves the scan alone.
    // This test pins the rows and that no row filter runs.
    let (_dir, db, _table) = fixture();
    for conjunct in ["random() >= 0", "array_length(l) = 1"] {
        let sql = format!(
            "SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' AND {conjunct} ORDER BY n LIMIT {FETCH}"
        );
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got, expected(false, &[]), "{sql}");
        assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
    }
}

#[test]
fn an_aggregate_under_order_by_limit_gets_no_row_filter() {
    // `GROUP BY s ORDER BY c DESC LIMIT 5`.
    //  - the limit sorts groups, not rows, so its cutoff never reaches the scan.
    // This test pins that the walk stops at the aggregate.
    let (_dir, db, _table) = fixture();
    let sql = format!(
        "SELECT s, COUNT(*) AS c FROM t WHERE s LIKE '%{DIGIT}%' GROUP BY s ORDER BY c DESC, s LIMIT {FETCH}"
    );
    let rows: usize = db
        .query_sql(&sql)
        .expect("sql")
        .iter()
        .map(RecordBatch::num_rows)
        .sum();
    assert_eq!(rows, FETCH);
    assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
}

#[test]
fn an_index_answered_where_under_order_by_limit_returns_the_right_rows() {
    // `s ILIKE '%7%'` on a full-text column, under `ORDER BY n LIMIT 5`.
    //  - the index picks the matching rows; there is no `FilterExec`.
    //  - the row filter adds only the cutoff on top.
    // This test pins the rows.
    let (_dir, db, _table) = fixture_with(IndexSpec::new().fts("s"));
    let sql = format!("SELECT n FROM t WHERE s ILIKE '%{DIGIT}%' ORDER BY n LIMIT {FETCH}");
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    assert_eq!(got, expected(false, &[]));
}

#[test]
fn an_offset_under_the_row_filter_returns_the_right_rows() {
    // `ORDER BY n LIMIT 5 OFFSET 3`: the sort keeps 8 rows and drops the first 3.
    //  - the cutoff is the 8th best `n`, so the row filter must not drop rows 4-8.
    // This test pins the rows and that the row filter runs.
    let (_dir, db, _table) = fixture();
    let sql =
        format!("SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' ORDER BY n LIMIT {FETCH} OFFSET 3");
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    let all: Vec<i64> = (0..2 * ROWS_PER_APPEND)
        .filter(|v| format!("r{v}").contains(DIGIT))
        .collect();
    assert_eq!(got, all[3..3 + FETCH]);
    assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
}

#[test]
fn a_scan_with_little_to_skip_gets_no_row_filter() {
    // `ORDER BY n LIMIT 5` over scans whose other columns are small next to `n`.
    //  - `SELECT n WHERE n % 7 = 0` reads only `n`, which the cutoff reads anyway.
    //  - `SELECT n, m` adds `m`, no bigger than `n`.
    // This test pins the rows and that the rule leaves both scans alone.
    let (_dir, db, _table) = fixture();
    for (sql, want) in [
        (
            format!("SELECT n FROM t WHERE n % 7 = 0 ORDER BY n LIMIT {FETCH}"),
            (0..).step_by(7).take(FETCH).collect::<Vec<i64>>(),
        ),
        (
            format!("SELECT n, m FROM t ORDER BY n LIMIT {FETCH}"),
            (0..FETCH as i64).collect(),
        ),
    ] {
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got, want, "{sql}");
        assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
    }
}

#[test]
fn a_large_limit_gets_no_row_filter() {
    // `ORDER BY n LIMIT 10001`, one past the largest limit the rule handles.
    //  - a big limit keeps a loose cutoff, so the row filter drops few rows.
    // This test pins the boundary: 10000 gets the row filter, 10001 does not.
    let (_dir, db, _table) = fixture();
    for (limit, want) in [(10_000, true), (10_001, false)] {
        let sql = format!("SELECT n, s FROM t ORDER BY n LIMIT {limit}");
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got, (0..limit).collect::<Vec<i64>>(), "{sql}");
        assert_eq!(row_filter_ran(&db, &sql), want, "{sql}");
    }
}
