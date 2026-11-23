// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A `LIMIT ... OFFSET ...` with no `ORDER BY` returns the same rows on every
//! run, so paging through a table with growing offsets returns each row
//! once: page `p` is rows `p * k ..` of the table read with no `OFFSET`.
//! With and without a `WHERE`, and on pages small enough for the staged
//! gather and too large for it. The scan reads its files in order, one
//! partition after another, so the first rows are the oldest commit's. A
//! fetch of more than two batches is gathered as fast as the partitions
//! read, in no fixed order.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use infino::{Connection, IndexSpec, connect};
use tempfile::TempDir;

/// Commits of `t`, each a superfile, enough for several partitions.
const COMMITS: i64 = 12;
/// Rows in each commit: commit `c` holds `seq` `c * ROWS .. (c + 1) * ROWS`.
const ROWS: i64 = 1_000;
/// Rows per page.
const PAGE: usize = 700;
/// Times each statement runs, to catch a run that reads in another order.
const RUNS: usize = 10;
/// The default `batch_size`: the file order holds for a fetch of up to two.
const BATCH: usize = 8192;

/// `t`, [`COMMITS`] commits of [`ROWS`] rows of `seq`, under `root`.
fn table(root: &TempDir) -> Connection {
    table_of(root, COMMITS)
}

/// `t`, `commits` commits of [`ROWS`] rows of `seq`, under `root`.
fn table_of(root: &TempDir, commits: i64) -> Connection {
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new())
        .expect("create t");
    for c in 0..commits {
        let seq = Int64Array::from((c * ROWS..(c + 1) * ROWS).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(seq)]).expect("batch");
        t.append(&batch, None).expect("append");
    }
    conn
}

/// The `seq` values `sql` returns, in order.
fn seqs(conn: &Connection, sql: &str) -> Vec<i64> {
    conn.query_sql(sql)
        .expect("query")
        .iter()
        .flat_map(|b| {
            let seq = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("Int64")
                .clone();
            (0..seq.len())
                .map(move |i| seq.value(i))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `plan` gathers the partitions in order: an ordered gather, or a staged
/// one that keeps the order.
fn gathers_in_order(plan: &str) -> bool {
    plan.lines().any(|line| {
        line.contains("OrderedCoalesceExec")
            || (line.contains("StagedCoalesceExec") && line.contains(", ordered"))
    })
}

/// The lines of the plan `EXPLAIN` gives for `sql`.
fn plan(conn: &Connection, sql: &str) -> String {
    let mut lines = String::new();
    for b in conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain") {
        let text = b
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("plan text");
        for i in (0..text.len()).filter(|&i| text.is_valid(i)) {
            lines.push_str(text.value(i));
            lines.push('\n');
        }
    }
    lines
}

/// `sql` returns `want` on every one of [`RUNS`] runs.
fn same_every_run(conn: &Connection, sql: &str, want: &[i64]) {
    for run in 0..RUNS {
        assert_eq!(seqs(conn, sql), want, "run {run}: {sql}");
    }
}

/// Every page of `select` ([`PAGE`] rows at growing offsets) is the slice of
/// the whole answer under a `LIMIT` it starts at, the same on every run; and
/// the whole answer is `kept`'s rows, each once, in file order.
fn pages_cover_each_row_once(conn: &Connection, select: &str, kept: impl Fn(i64) -> bool) {
    let total = usize::try_from(COMMITS * ROWS).expect("rows fit");
    let whole_sql = format!("{select} LIMIT {total}");
    assert!(
        gathers_in_order(&plan(conn, &whole_sql)),
        "no ordered gather in the plan of {whole_sql}:\n{}",
        plan(conn, &whole_sql)
    );
    let whole = seqs(conn, &whole_sql);
    let want: Vec<i64> = (0..COMMITS * ROWS).filter(|&seq| kept(seq)).collect();
    assert_eq!(whole, want, "{whole_sql}: each row once, in file order");
    same_every_run(conn, &whole_sql, &whole);

    for (p, page) in whole.chunks(PAGE).enumerate() {
        let sql = format!("{select} LIMIT {PAGE} OFFSET {}", p * PAGE);
        same_every_run(conn, &sql, page);
    }
}

#[test]
fn unfiltered_pages_return_each_row_once() {
    let root = TempDir::new().expect("tempdir");
    let conn = table(&root);
    pages_cover_each_row_once(&conn, "SELECT seq FROM t", |_| true);
}

#[test]
fn filtered_pages_return_each_row_once() {
    let root = TempDir::new().expect("tempdir");
    let conn = table(&root);
    pages_cover_each_row_once(&conn, "SELECT seq FROM t WHERE seq % 3 <> 0", |seq| {
        seq % 3 != 0
    });
    // A small page deep in the table: the staged gather, still in order.
    let sql = "SELECT seq FROM t WHERE seq % 3 <> 0 LIMIT 20 OFFSET 2990";
    assert!(
        plan(&conn, sql).contains("StagedCoalesceExec: fetch=3010, ordered"),
        "{sql} not staged in order:\n{}",
        plan(&conn, sql)
    );
    let want: Vec<i64> = (0..COMMITS * ROWS)
        .filter(|seq| seq % 3 != 0)
        .skip(2990)
        .take(20)
        .collect();
    same_every_run(&conn, sql, &want);
}

#[test]
fn pages_of_a_computed_column_return_each_row_once() {
    // A computed column over a gather of more than a batch: DataFusion
    // spreads the computation over partitions with a round-robin exchange
    // above the gather, and the partitions' batches come out in any order.
    let root = TempDir::new().expect("tempdir");
    let conn = table(&root);
    let select = "SELECT abs(seq) AS seq FROM t";
    pages_cover_each_row_once(&conn, select, |_| true);
    let whole_sql = format!("{select} LIMIT {}", COMMITS * ROWS);
    assert!(
        !plan(&conn, &whole_sql).contains("RoundRobinBatch"),
        "{whole_sql} deals the gathered rows out again:\n{}",
        plan(&conn, &whole_sql)
    );
}

#[test]
fn a_fetch_of_more_than_two_batches_takes_the_plain_gather() {
    // In file order the first rows come from one partition, read by one
    // task: past a couple of batches, the plain gather reads them faster.
    let commits = 20;
    let total = usize::try_from(commits * ROWS).expect("rows fit");
    assert!(total > 2 * BATCH, "the table outgrows the ordered fetch");
    let root = TempDir::new().expect("tempdir");
    let conn = table_of(&root, commits);
    let at_most = format!("SELECT abs(seq) AS seq FROM t LIMIT {}", 2 * BATCH);
    assert!(
        gathers_in_order(&plan(&conn, &at_most)),
        "{at_most} not gathered in order:\n{}",
        plan(&conn, &at_most)
    );
    let past = format!("SELECT seq FROM t LIMIT {}", 2 * BATCH + 1);
    let past_plan = plan(&conn, &past);
    assert!(
        !gathers_in_order(&past_plan) && past_plan.contains("CoalescePartitionsExec"),
        "{past} gathered in order:\n{past_plan}"
    );
    let rows = seqs(&conn, &past);
    assert_eq!(
        rows.len(),
        2 * BATCH + 1,
        "{past}: the fetch, of a larger table"
    );
    let distinct: HashSet<i64> = rows.iter().copied().collect();
    assert_eq!(distinct.len(), rows.len(), "{past}: each row once");
    assert!(
        rows.iter().all(|seq| (0..commits * ROWS).contains(seq)),
        "{past}: rows of the table"
    );
}
