// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A top-k over a scan of kept decoded columns reads row groups best first
//! and skips the ones its tightening bound rules out by their statistics,
//! and answers as a full sort would: on the id, with a key on a column no
//! row holds a value in before it and without, and on another column,
//! ascending and descending, under a filter no index answers, deleted rows
//! left out. Its own binary, because the host configuration (and with it the
//! process's decoded-column cache) is read once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use datafusion::prelude::{col, lit};
use infino::{
    IndexSpec,
    arrow_array::{Array, Decimal128Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: far more than the table
/// holds, so every column the first statement decodes stays.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// Rows in each of the table's commits.
const ROWS: i64 = 3_000;

/// Commits, each a superfile.
const COMMITS: i64 = 3;

/// The `_id` column across `batches`, in order.
fn ids(batches: &[RecordBatch]) -> Vec<i128> {
    batches
        .iter()
        .flat_map(|b| {
            let ids = b
                .column_by_name("_id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                .expect("_id")
                .clone();
            (0..ids.len()).map(move |i| ids.value(i))
        })
        .collect()
}

/// The physical plan text `EXPLAIN` gives for `sql`.
fn physical_plan(conn: &infino::Connection, sql: &str) -> String {
    conn.query_sql(&format!("EXPLAIN {sql}"))
        .expect("explain")
        .iter()
        .flat_map(|b| {
            let kinds = b
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan kind")
                .clone();
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len())
                .filter(|i| plans.is_valid(*i) && kinds.value(*i) == "physical_plan")
                .map(|i| plans.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_top_k_over_kept_columns_answers_as_a_full_sort() {
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        format!("memory:\n  decoded_column_cache_bytes: {KEPT_BYTES}\n"),
    )
    .expect("config file");
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
    assert_eq!(
        infino::config::global().memory.decoded_column_cache_bytes,
        KEPT_BYTES
    );

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("note", DataType::LargeUtf8, true),
    ]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new())
        .expect("create t");
    for commit in 0..COMMITS {
        let seq: Vec<i64> = (commit * ROWS..(commit + 1) * ROWS).collect();
        let note: Vec<Option<&str>> = seq.iter().map(|_| None).collect();
        t.append(
            &RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seq)),
                    Arc::new(LargeStringArray::from(note)),
                ],
            )
            .expect("batch"),
        )
        .expect("append");
    }
    t.delete(col("seq").lt(lit(30_i64))).expect("delete");

    let every = {
        let mut all = ids(&conn
            .query_sql("SELECT _id FROM t WHERE seq % 7 = 0")
            .expect("sql"));
        all.sort_unstable();
        all
    };
    for k in [1_usize, 10, 500, 5_000] {
        let mut want = every.clone();
        want.truncate(k);
        for keys in ["_id ASC", "note ASC NULLS LAST, _id ASC"] {
            let sql = format!("SELECT seq, \"_id\" AS \"_id\" FROM t WHERE seq % 7 = 0 ORDER BY {keys} LIMIT {k}");
            for run in ["decoding", "from kept columns"] {
                let got = ids(&conn.query_sql(&sql).unwrap_or_else(|e| panic!("{run}: {e}")));
                assert_eq!(got, want, "{run}: {sql}");
            }
            let plan = physical_plan(&conn, &sql);
            assert!(plan.contains("CachedScanExec"), "{plan}");
        }
    }

    // A top-k on another column reads row groups best first and skips the
    // ones its bound rules out as it tightens: the same rows as a full sort.
    let seqs = |batches: &[RecordBatch]| -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect()
    };
    let mut kept: Vec<i64> = (30..COMMITS * ROWS).filter(|s| s % 7 == 0).collect();
    for (order, descending) in [("ASC", false), ("DESC", true)] {
        if descending {
            kept.reverse();
        }
        for k in [1_usize, 10, 500, 5_000] {
            let sql = format!("SELECT seq FROM t WHERE seq % 7 = 0 ORDER BY seq {order} LIMIT {k}");
            for run in ["first", "again"] {
                let got = seqs(&conn.query_sql(&sql).unwrap_or_else(|e| panic!("{run}: {e}")));
                assert_eq!(got, kept[..k.min(kept.len())], "{run}: {sql}");
            }
        }
    }
}
