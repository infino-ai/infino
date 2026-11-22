// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A `LIMIT ... OFFSET ...` with no `ORDER BY` over a scan of kept decoded
//! columns returns the same rows on every run, in file order, so paging with
//! growing offsets returns each row once. Its own binary, because the host
//! configuration (and with it the process's decoded-column cache) is read
//! once per process.

#![deny(clippy::unwrap_used)]

use std::{
    env, fs,
    sync::{Arc, Once},
};

use infino::{
    Connection, IndexSpec,
    arrow_array::{Array, Int64Array, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    config::global,
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: far more than the table
/// holds, so every column the first statement decodes stays.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;
/// Commits of `t`, each a superfile, enough for several partitions.
const COMMITS: i64 = 12;
/// Rows in each commit: commit `c` holds `seq` `c * ROWS .. (c + 1) * ROWS`.
const ROWS: i64 = 1_000;
/// Rows per page.
const PAGE: usize = 700;
/// Times each statement runs, to catch a run that reads in another order.
const RUNS: usize = 10;
/// How `EXPLAIN` shows a scan of kept columns read in file order.
const CACHED_IN_FILE_ORDER: &str = ", file_order";

/// Points the process's configuration at a temp dir that keeps [`KEPT_BYTES`]
/// of decoded columns, before any test reads it.
fn kept_columns_config() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // Leaked so the config file outlives the read below and every test.
        let config = Box::leak(Box::new(TempDir::new().expect("tempdir")));
        fs::create_dir_all(config.path().join("infino")).expect("config dir");
        fs::write(
            config.path().join("infino/config.yaml"),
            format!("memory:\n  decoded_column_cache_bytes: {KEPT_BYTES}\n"),
        )
        .expect("config file");
        // SAFETY: inside the `Once`, and every test calls this before it
        // touches the environment or the configuration, so no other thread
        // reads either while it is set.
        unsafe { env::set_var("XDG_CONFIG_HOME", config.path()) };
        assert_eq!(global().memory.decoded_column_cache_bytes, KEPT_BYTES);
    });
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

#[test]
fn pages_over_kept_columns_return_each_row_once() {
    kept_columns_config();

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new())
        .expect("create t");
    for c in 0..COMMITS {
        let seq = Int64Array::from((c * ROWS..(c + 1) * ROWS).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(seq)]).expect("batch");
        t.append(&batch).expect("append");
    }

    for (select, kept) in [
        ("SELECT seq FROM t", (|_| true) as fn(i64) -> bool),
        ("SELECT seq FROM t WHERE seq % 3 <> 0", |seq| seq % 3 != 0),
        // A computed column, which a round-robin exchange above the gather
        // would spread over partitions.
        ("SELECT abs(seq) AS seq FROM t", |_| true),
    ] {
        let total = usize::try_from(COMMITS * ROWS).expect("rows fit");
        let whole_sql = format!("{select} LIMIT {total}");
        let want: Vec<i64> = (0..COMMITS * ROWS).filter(|&seq| kept(seq)).collect();
        // The first run decodes the columns and keeps them; the later runs
        // scan the kept columns.
        assert_eq!(seqs(&conn, &whole_sql), want, "decoding: {whole_sql}");
        assert!(
            plan(&conn, &whole_sql).contains(CACHED_IN_FILE_ORDER),
            "kept columns not read in file order for {whole_sql}:\n{}",
            plan(&conn, &whole_sql)
        );
        for run in 0..RUNS {
            assert_eq!(seqs(&conn, &whole_sql), want, "run {run}: {whole_sql}");
        }
        for (p, page) in want.chunks(PAGE).enumerate() {
            let sql = format!("{select} LIMIT {PAGE} OFFSET {}", p * PAGE);
            for run in 0..RUNS {
                assert_eq!(seqs(&conn, &sql), page, "run {run}: {sql}");
            }
        }
    }
}
