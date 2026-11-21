// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A top-k the range index answers reads only the rows it chose. With a
//! column far wider than the sort's selected, the scan of kept decoded
//! columns still serves it, taking that column for just those rows, on the
//! first run and on every run after, rather than the Parquet scan decoding
//! whole pages of it. Its answer is the full sort's, deleted rows left out.
//! Its own binary, because the host configuration (and with it the process's
//! decoded-column cache) is read once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use datafusion::prelude::{col, lit};
use infino::{
    IndexSpec,
    arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: less than `body` takes over
/// the row groups a top-k touches, more than `ts` does.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// Commits of `t`, each a superfile of one row group.
const COMMITS: i64 = 4;

/// Rows in each commit.
const ROWS: i64 = 20_000;

/// A prime no row count is a multiple of, so `ts` is not in file order.
const SPREAD: i64 = 7_919;

/// Bytes of padding in each `body`: wide, like a stored document.
const BODY_BYTES: usize = 2_000;

/// The `LIMIT` of the top-k.
const K: usize = 5;

/// Runs of each statement: the first reads the chosen rows; later ones find
/// the row groups read in part before.
const RUNS: usize = 3;

/// Rows deleted: those with `ts` below this.
const DELETED_BELOW: i64 = 2;

/// Points the process's configuration at a temp dir that keeps [`KEPT_BYTES`]
/// of decoded columns, before anything reads it.
fn kept_columns_config() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // Leaked so the config file outlives the read below and every test.
        let config = Box::leak(Box::new(TempDir::new().expect("tempdir")));
        std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
        std::fs::write(
            config.path().join("infino/config.yaml"),
            format!("memory:\n  decoded_column_cache_bytes: {KEPT_BYTES}\n"),
        )
        .expect("config file");
        // SAFETY: inside the `Once`, and the test calls this before it
        // touches the environment or the configuration, so no other thread
        // reads either while it is set.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
        assert_eq!(
            infino::config::global().memory.decoded_column_cache_bytes,
            KEPT_BYTES
        );
    });
}

/// `(ts, body)` of every row `sql` returns, in order.
fn rows(conn: &infino::Connection, sql: &str) -> Vec<(i64, String)> {
    conn.query_sql(sql)
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .iter()
        .flat_map(|b| {
            let ts = b
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("ts")
                .clone();
            let body = b
                .column(1)
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("body")
                .clone();
            (0..ts.len()).map(move |i| (ts.value(i), body.value(i).to_owned()))
        })
        .collect()
}

/// The plan `EXPLAIN ANALYZE` gives for `sql`.
fn analyzed(conn: &infino::Connection, sql: &str) -> String {
    conn.query_sql(&format!("EXPLAIN ANALYZE {sql}"))
        .unwrap_or_else(|e| panic!("{sql}: {e}"))
        .iter()
        .flat_map(|b| {
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len())
                .filter(|&i| plans.is_valid(i))
                .map(|i| plans.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `body` of the row holding `ts`.
fn body(ts: i64) -> String {
    format!("{ts}{}", "x".repeat(BODY_BYTES))
}

#[test]
fn a_top_k_the_index_answers_reads_a_wide_column_for_its_rows_alone() {
    // `SELECT ts, body ORDER BY ts LIMIT 5`, ascending and descending (nulls
    // last, the order the range index answers), with
    // `ts` range-indexed and every `ts` below 2 deleted.
    //  - the index chooses the rows; the scan of kept columns reads `body`
    //    for those rows from the pages, and decodes no row group whole.
    //  - on later runs too, though a scan read those row groups in part
    //    before: keeping `body` for them whole would not fit the cache.
    // This test pins the rows and that the Parquet scan never serves it.
    kept_columns_config();
    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new().range("ts"))
        .expect("create t");
    for commit in 0..COMMITS {
        let ts: Vec<i64> = (0..ROWS)
            .map(|i| (i * SPREAD % ROWS) * COMMITS + commit)
            .collect();
        let bodies: Vec<String> = ts.iter().map(|&v| body(v)).collect();
        t.append(
            &RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(ts)),
                    Arc::new(LargeStringArray::from(bodies)),
                ],
            )
            .expect("batch"),
        )
        .expect("append");
    }
    t.delete(col("ts").lt(lit(DELETED_BELOW))).expect("delete");
    let k = i64::try_from(K).expect("k");
    let last = COMMITS * ROWS - 1;
    for (order, want) in [
        (
            "ASC",
            (DELETED_BELOW..DELETED_BELOW + k).collect::<Vec<_>>(),
        ),
        ("DESC", (last - k + 1..=last).rev().collect()),
    ] {
        let sql = format!("SELECT ts, body FROM t ORDER BY ts {order} NULLS LAST LIMIT {K}");
        let want: Vec<(i64, String)> = want.into_iter().map(|v| (v, body(v))).collect();
        for run in 0..RUNS {
            assert_eq!(rows(&conn, &sql), want, "run {run}: {sql}");
            let plan = analyzed(&conn, &sql);
            assert!(plan.contains("CachedScanExec"), "run {run}: {plan}");
            assert!(!plan.contains("DataSourceExec"), "run {run}: {plan}");
            assert!(plan.contains("row_groups_decoded=0"), "run {run}: {plan}");
        }
    }
}
