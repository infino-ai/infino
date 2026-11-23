// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Where a `WHERE` runs: as a Parquet row filter in a scan under
//! `ORDER BY ... LIMIT`, or under a plain `LIMIT` when the scan reads wide
//! columns the `WHERE` doesn't, in a `FilterExec` above the scan everywhere
//! else. Any other filtered scan under a plain `LIMIT` is gathered in stages:
//! its first partition alone, then the rest side by side if that falls short.
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
/// How `EXPLAIN` shows a plain `LIMIT`'s gather run in stages.
const STAGED: &str = "StagedCoalesceExec: fetch=";
/// How `EXPLAIN` shows the gate that holds a staged scan's partitions 1.. back.
const GATE: &str = "ScanGateExec";
/// The largest `LIMIT` whose gather runs in stages: DataFusion's default
/// batch size, the most rows partition 0's first batch can hold.
const MAX_FETCH: usize = 8_192;
/// How `EXPLAIN ANALYZE` shows how many files a scan opened.
const FILES_OPENED: &str = "files_opened=";
/// The value of `m` the plain-`LIMIT` filter keeps: a tenth of the rows.
const M: i64 = 3;
/// How `EXPLAIN` shows a scan that stops after the `LIMIT`'s rows.
const SCAN_LIMIT: &str = "], limit=";
/// Rows of the one-superfile table, appended at once in order of `n`: a
/// superfile holds 65,536 rows per row group, so this makes two.
const ONE_FILE_ROWS: i64 = 100_000;
/// The first `n` of that superfile's second row group.
const SECOND_GROUP: i64 = 65_536;

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
    batch_of(
        (0..ROWS_PER_APPEND)
            .map(|i| lo + (i * SHUFFLE) % ROWS_PER_APPEND)
            .collect(),
    )
}

/// Rows with these `n`, in this order, and `m`, `s`, `l` as in [`batch`].
fn batch_of(n: Vec<i64>) -> RecordBatch {
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
    let mut n = matching(deleted);
    if descending {
        n.reverse();
    }
    n.truncate(FETCH);
    n
}

/// Every `n` whose `s` has `DIGIT`, ascending, skipping `deleted`.
fn matching(deleted: &[i64]) -> Vec<i64> {
    (0..2 * ROWS_PER_APPEND)
        .filter(|v| format!("r{v}").contains(DIGIT) && !deleted.contains(v))
        .collect()
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

/// Asserts `sql` returns `FETCH` distinct rows, each in `all`.
fn assert_fetch_of(db: &Connection, sql: &str, all: &[i64]) {
    let mut got = n_values(&db.query_sql(sql).expect("sql"));
    assert_eq!(got.len(), FETCH, "{sql}");
    got.sort_unstable();
    got.dedup();
    assert_eq!(got.len(), FETCH, "{sql}: distinct rows");
    assert!(got.iter().all(|n| all.contains(n)), "{sql}: {got:?}");
}

#[test]
fn a_plain_limit_gathers_its_filtered_scan_in_stages() {
    // `WHERE s LIKE '%7%' LIMIT 5`, no `ORDER BY`, and the same with an offset.
    //  - the gather runs the scan's first partition alone, then the rest side
    //    by side only if that falls short; no exchange sits next to the scan.
    //  - the `WHERE` stays in a `FilterExec`; no row filter.
    // This test pins the plan shape and that every row returned matches.
    let (_dir, db, _table) = fixture();
    let all = matching(&[]);
    for (offset, fetch) in [("", FETCH), (" OFFSET 3", FETCH + 3)] {
        let sql = format!("SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' LIMIT {FETCH}{offset}");
        assert_fetch_of(&db, &sql, &all);
        assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
        let plan = explain(&db, "", &sql);
        assert!(plan.contains(&format!("{STAGED}{fetch}")), "{plan}");
        assert!(plan.contains("FilterExec"), "{plan}");
        let below = plan.split_once(STAGED).expect("a staged gather").1;
        assert!(below.contains(GATE), "{plan}");
        assert!(!below.contains("RepartitionExec"), "{plan}");
    }
}

/// How many files the scans in `sql`'s plan opened.
fn files_opened(db: &Connection, sql: &str) -> usize {
    let plan = explain(db, "ANALYZE", sql);
    plan.split(FILES_OPENED)
        .skip(1)
        .map(|rest| {
            rest.split([',', ']'])
                .next()
                .and_then(|count| count.trim().parse::<usize>().ok())
                .expect("a file count")
        })
        .sum()
}

#[test]
fn a_first_batch_that_meets_a_plain_limit_opens_no_other_file() {
    // `WHERE s LIKE '%-%' LIMIT 5`: every row matches.
    //  - partition 0's first batch has the rows, so the gate stays shut and
    //    the other superfile is never opened.
    // This test pins the rows and the files opened.
    let (_dir, db, _table) = fixture();
    let sql = format!("SELECT n, s FROM t WHERE s LIKE '%-%' LIMIT {FETCH}");
    assert_eq!(n_values(&db.query_sql(&sql).expect("sql")).len(), FETCH);
    assert!(explain(&db, "", &sql).contains(STAGED), "{sql}");
    assert_eq!(files_opened(&db, &sql), 1, "{sql}");
}

#[test]
fn a_first_file_that_gives_no_row_opens_the_gate_when_it_ends() {
    // `s ILIKE '%r39999%' LIMIT 5` on a full-text column: the one match is in
    // the second superfile, so the index rules out every row of the first.
    //  - partition 0 gives no batch and ends; the gate opens and partition 1
    //    finds the row.
    // This test pins the row and that both files were opened.
    let (_dir, db, _table) = fixture_with(IndexSpec::new().fts("s"));
    let last = 2 * ROWS_PER_APPEND - 1;
    let sql = format!("SELECT n, s FROM t WHERE s ILIKE '%r{last}%' LIMIT {FETCH}");
    assert_eq!(n_values(&db.query_sql(&sql).expect("sql")), vec![last]);
    assert!(explain(&db, "", &sql).contains(STAGED), "{sql}");
    assert_eq!(files_opened(&db, &sql), 2, "{sql}");
}

#[test]
fn an_index_answered_where_under_a_plain_limit_is_gathered_in_stages() {
    // `s ILIKE '%7%' LIMIT 5` on a full-text column.
    //  - the index picks the rows, so there is no `FilterExec`, and the scan
    //    still takes no fetch: its gather runs in stages all the same.
    // This test pins the rows and the plan shape.
    let (_dir, db, _table) = fixture_with(IndexSpec::new().fts("s"));
    let sql = format!("SELECT n, s FROM t WHERE s ILIKE '%{DIGIT}%' LIMIT {FETCH}");
    assert_fetch_of(&db, &sql, &matching(&[]));
    let plan = explain(&db, "", &sql);
    assert!(plan.contains(STAGED) && plan.contains(GATE), "{plan}");
}

#[test]
fn deleted_rows_stay_out_under_a_plain_limit() {
    // Delete every odd `n`, then `WHERE s LIKE '%7%' LIMIT <every match>`.
    //  - the first partition falls short, so every partition is read; the
    //    scan skips deleted rows in each.
    // This test pins that every live match comes back and no deleted one does.
    let (_dir, db, table) = fixture();
    table
        .delete((col("n") % lit(2_i64)).eq(lit(1_i64)))
        .expect("delete");
    let want: Vec<i64> = matching(&[]).into_iter().filter(|n| n % 2 == 0).collect();
    assert!(
        want.len() <= MAX_FETCH,
        "the limit must stay under the staged gather's"
    );
    let sql = format!(
        "SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' LIMIT {}",
        want.len()
    );
    let mut got = n_values(&db.query_sql(&sql).expect("sql"));
    got.sort_unstable();
    assert_eq!(got, want);
    assert!(explain(&db, "", &sql).contains(STAGED), "{sql}");
}

#[test]
fn a_plain_limit_is_gathered_at_once_without_a_where_or_past_the_largest_limit() {
    // `SELECT n, s LIMIT 5` has no `WHERE`: its fetch already reaches the scan.
    // `WHERE ... LIMIT 8193` is one past the largest staged limit.
    // This test pins the row counts and that neither gather runs in stages.
    let (_dir, db, _table) = fixture();
    for (sql, rows) in [
        (format!("SELECT n, s FROM t LIMIT {FETCH}"), FETCH),
        (
            format!(
                "SELECT n, s FROM t WHERE s LIKE '%{DIGIT}%' LIMIT {}",
                MAX_FETCH + 1
            ),
            matching(&[]).len().min(MAX_FETCH + 1),
        ),
    ] {
        assert_eq!(
            n_values(&db.query_sql(&sql).expect("sql")).len(),
            rows,
            "{sql}"
        );
        assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
        assert!(!explain(&db, "", &sql).contains(STAGED), "{sql}");
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

/// Every `n` whose `m` is `M`, skipping `deleted`.
fn matching_m(deleted: &[i64]) -> Vec<i64> {
    (0..2 * ROWS_PER_APPEND)
        .filter(|v| v % 10 == M && !deleted.contains(v))
        .collect()
}

#[test]
fn a_plain_limit_runs_its_filter_inside_the_scan_when_it_reads_narrow_columns() {
    // `SELECT n, s WHERE m = 3 LIMIT 5`, and the same with an offset.
    //  - the `WHERE` reads `m` alone; `s` is many times wider.
    //  - the rule marks the scan, DataFusion moves the `WHERE` into it, and
    //    the scan takes the limit, so it decodes `s` only for rows that pass.
    //  - the gather still runs partition 0 alone first; its first file has
    //    the rows, so the other superfile is never opened.
    // This test pins the plan shape, the files opened and that every row
    // returned matches.
    let (_dir, db, _table) = fixture();
    let all = matching_m(&[]);
    for (offset, fetch) in [("", FETCH), (" OFFSET 3", FETCH + 3)] {
        let sql = format!("SELECT n, s FROM t WHERE m = {M} LIMIT {FETCH}{offset}");
        assert_fetch_of(&db, &sql, &all);
        assert!(row_filter_ran(&db, &sql), "no row filter for: {sql}");
        let plan = explain(&db, "", &sql);
        assert!(!plan.contains("FilterExec"), "{plan}");
        assert!(plan.contains(&format!("{SCAN_LIMIT}{fetch}")), "{plan}");
        assert!(plan.contains(&format!("{STAGED}{fetch}")), "{plan}");
        assert!(plan.contains(GATE), "{plan}");
        assert_eq!(files_opened(&db, &sql), 1, "{sql}");
    }
}

#[test]
fn deleted_rows_stay_out_of_a_plain_limit_whose_where_every_row_meets() {
    // Delete every odd `n`, then `WHERE n >= 0 LIMIT <every live row>`.
    //  - the statistics prove every row group matches `n >= 0`. Given the
    //    limit, the Parquet reader would read such row groups whole, past the
    //    deletes, so the rule leaves a scan with deleted rows alone.
    // This test pins that no deleted row comes back.
    let (_dir, db, table) = fixture();
    table
        .delete((col("n") % lit(2_i64)).eq(lit(1_i64)))
        .expect("delete");
    let want: Vec<i64> = (0..2 * ROWS_PER_APPEND).filter(|n| n % 2 == 0).collect();
    for limit in [FETCH, want.len()] {
        let sql = format!("SELECT n, s FROM t WHERE n >= 0 LIMIT {limit}");
        let got = n_values(&db.query_sql(&sql).expect("sql"));
        assert_eq!(got.len(), limit, "{sql}");
        assert!(got.iter().all(|n| n % 2 == 0), "{sql}: {got:?}");
        assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
    }
}

#[test]
fn a_plain_limit_whose_where_reads_the_wide_column_keeps_its_filter_above() {
    // `SELECT n, m WHERE s LIKE '%7%' LIMIT 5`: the `WHERE` reads the wide
    // column, and nothing wide is left to skip.
    // This test pins the rows and that the `WHERE` stays in a `FilterExec`.
    let (_dir, db, _table) = fixture();
    let sql = format!("SELECT n, m FROM t WHERE s LIKE '%{DIGIT}%' LIMIT {FETCH}");
    assert_fetch_of(&db, &sql, &matching(&[]));
    assert!(!row_filter_ran(&db, &sql), "row filter ran for: {sql}");
    assert!(explain(&db, "", &sql).contains("FilterExec"), "{sql}");
}

/// A no-index table `t` of one superfile, `n` from 0 to [`ONE_FILE_ROWS`] in
/// file order, in two row groups split at [`SECOND_GROUP`].
fn one_file_fixture() -> (TempDir, Connection) {
    let dir = TempDir::new().expect("tempdir");
    let db = connect_with(
        dir.path().join("db").to_str().expect("utf-8 path"),
        ConnectOptions::new().with_read_consistency(Consistency::Strong),
    )
    .expect("connect");
    let table = db
        .create_table("t", schema(), IndexSpec::new())
        .expect("create_table");
    table
        .append(&batch_of((0..ONE_FILE_ROWS).collect()))
        .expect("append");
    (dir, db)
}

/// Asserts `SELECT n, s WHERE <filter> LIMIT limit OFFSET offset` returns
/// the slice of `matches` from `offset` on, in that order, and that its scan
/// ran the `WHERE` as a row filter and took the limit. Returns the page.
fn assert_page_of(
    db: &Connection,
    filter: &str,
    matches: &[i64],
    limit: usize,
    offset: usize,
) -> Vec<i64> {
    let sql = format!("SELECT n, s FROM t WHERE {filter} LIMIT {limit} OFFSET {offset}");
    let got = n_values(&db.query_sql(&sql).expect("sql"));
    let end = matches.len().min(offset + limit);
    assert_eq!(got, matches[offset.min(end)..end], "{sql}");
    assert!(row_filter_ran(db, &sql), "no row filter for: {sql}");
    let plan = explain(db, "", &sql);
    assert!(
        plan.contains(&format!("{SCAN_LIMIT}{}", limit + offset)),
        "{plan}"
    );
    got
}

/// [`assert_page_of`] for `WHERE n >= from` on the one-superfile table.
fn assert_page_in_file_order(db: &Connection, from: i64, limit: i64, offset: i64) -> Vec<i64> {
    let matches: Vec<i64> = (from..ONE_FILE_ROWS).collect();
    let size = |v: i64| usize::try_from(v).expect("non-negative");
    assert_page_of(
        db,
        &format!("n >= {from}"),
        &matches,
        size(limit),
        size(offset),
    )
}

#[test]
fn pages_of_a_plain_limit_read_the_row_groups_in_file_order() {
    // One superfile of two row groups; `WHERE n >= 60000` keeps the last
    // 5,536 rows of the first and every row of the second.
    //  - given the limit, the Parquet reader would read only row groups the
    //    statistics prove match (the second), skipping the first, and how
    //    many depends on the limit: `LIMIT 10` would start at 65536 and a
    //    deeper page would repeat rows. The scan keeps file order instead.
    // This test pins that each page, one that crosses the row groups among
    // them, holds the rows that follow the page before it, and that paging
    // walks the matches once each.
    let (_dir, db) = one_file_fixture();
    let from = 60_000;
    let into_second = SECOND_GROUP - from;
    for offset in [0, into_second - 6, into_second, into_second + 4] {
        assert_page_in_file_order(&db, from, 10, offset);
    }
    let page = 1_000;
    let walked: Vec<i64> = (0..8)
        .flat_map(|p| assert_page_in_file_order(&db, from, page, p * page))
        .collect();
    assert_eq!(walked, (from..from + 8 * page).collect::<Vec<i64>>());
}

#[test]
fn a_plain_limit_page_that_spans_two_batches_keeps_file_order() {
    // `WHERE n >= 64000` keeps the last 1,536 rows of the first row group and
    // every row of the second, which the reader hands on in batches of 8,192:
    // the first ends at 73727. A page from 73723 takes rows from two batches
    // of the second row group, past the largest staged gather. With an
    // `OFFSET`, a limit node sits between the scan and its meter.
    // This test pins that page and the one before it.
    let (_dir, db) = one_file_fixture();
    let from = 64_000;
    let second_batch = SECOND_GROUP + MAX_FETCH as i64;
    for offset in [second_batch - from - 15, second_batch - from - 5] {
        assert_page_in_file_order(&db, from, 10, offset);
    }
}

#[test]
fn rows_inside_a_plain_limit_page_keep_file_order() {
    // `WHERE n >= 60000 LIMIT 10 OFFSET 5530` on one superfile, a single
    // partition: the scan takes the limit, and under the `OFFSET` a limit
    // node sits between the scan and its meter.
    //  - DataFusion puts a round-robin exchange over a single partition to
    //    spread a projection, which hands the page's batches on in any order.
    //    The exchange is dropped once the scan reads in file order.
    // This test pins the page's rows in order, on every run, and that no
    // exchange is left.
    let (_dir, db) = one_file_fixture();
    let sql = "SELECT n, s FROM t WHERE n >= 60000 LIMIT 10 OFFSET 5530";
    for _ in 0..10 {
        assert_page_in_file_order(&db, 60_000, 10, 5_530);
    }
    let plan = explain(&db, "", sql);
    assert!(!plan.contains("RoundRobinBatch"), "{plan}");
}

/// Superfiles of the several-file table.
const FILES: i64 = 3;
/// Rows of each of those superfiles, one row group each.
const FILE_ROWS: i64 = 4_000;

/// Whether a `WHERE` keeps the row with this `n`.
type Keeps = fn(i64) -> bool;

/// A no-index table `t` of [`FILES`] superfiles, appended one after another,
/// `n` from 0 to `FILES * FILE_ROWS` in file order.
fn several_files_fixture() -> (TempDir, Connection) {
    let dir = TempDir::new().expect("tempdir");
    let db = connect_with(
        dir.path().join("db").to_str().expect("utf-8 path"),
        ConnectOptions::new().with_read_consistency(Consistency::Strong),
    )
    .expect("connect");
    let table = db
        .create_table("t", schema(), IndexSpec::new())
        .expect("create_table");
    for f in 0..FILES {
        table
            .append(&batch_of((f * FILE_ROWS..(f + 1) * FILE_ROWS).collect()))
            .expect("append");
    }
    (dir, db)
}

#[test]
fn pages_of_a_plain_limit_over_several_files_return_each_row_once_in_order() {
    // Three superfiles of 4,000 rows, read in three partitions, paged 700
    // rows at a time with the row filter on.
    //  - `WHERE n >= 3000` keeps the last 1,000 rows of the first file and
    //    every row of the others. Given the limit, the Parquet reader would
    //    read first the files the statistics prove match.
    //  - `WHERE m < 5` keeps half the rows of every file, spread out.
    //  - the partitions are gathered one after another, each reading its own
    //    file in order.
    // This test pins that every page is the slice of the matches it starts
    // at, in order, and that paging walks the matches once each.
    const PAGE: usize = 700;
    let (_dir, db) = several_files_fixture();
    let filters: [(&str, Keeps); 2] = [("n >= 3000", |n| n >= 3_000), ("m < 5", |n| n % 10 < 5)];
    for (filter, keeps) in filters {
        let matches: Vec<i64> = (0..FILES * FILE_ROWS).filter(|&n| keeps(n)).collect();
        let walked: Vec<i64> = (0..matches.len().div_ceil(PAGE))
            .flat_map(|p| assert_page_of(&db, filter, &matches, PAGE, p * PAGE))
            .collect();
        assert_eq!(walked, matches, "{filter}");
    }
}
