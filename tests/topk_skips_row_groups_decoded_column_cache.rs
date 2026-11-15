// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A top-k over a scan of kept decoded columns reads row groups best first
//! (on the id, id blocks in order of their bounds) and skips the ones its
//! tightening bound rules out by their statistics, and answers as a full
//! sort would: on the id, with a key on a column no row holds a value in
//! before it and without, and on another column,
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

/// Fewest commits, each a superfile of one row group.
const MIN_COMMITS: i64 = 3;

/// The `LIMIT`s small enough that a top-k on the id settles within the
/// first id block each partition reads.
const SMALL_K: [usize; 2] = [1, 10];

/// How `EXPLAIN ANALYZE` names the rows of the units the scan read.
const ROWS_READ: &str = "rows_read";

/// How `EXPLAIN ANALYZE` names the row groups the scan decoded.
const ROW_GROUPS_DECODED: &str = "row_groups_decoded";

/// The scale of each suffix `EXPLAIN ANALYZE` prints a count with.
const COUNT_SUFFIXES: [(&str, f64); 3] = [(" K", 1e3), (" M", 1e6), (" B", 1e9)];

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

/// The scan's metric `name` as `EXPLAIN ANALYZE` prints it running `sql`,
/// read back from its rounded form (`3.07 K`).
fn scan_metric(conn: &infino::Connection, sql: &str, name: &str) -> f64 {
    let lines: Vec<String> = conn
        .query_sql(&format!("EXPLAIN ANALYZE {sql}"))
        .expect("explain analyze")
        .iter()
        .flat_map(|b| {
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len())
                .filter(|i| plans.is_valid(*i))
                .flat_map(|i| {
                    plans
                        .value(i)
                        .lines()
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let line = lines
        .iter()
        .find(|line| line.contains("CachedScanExec"))
        .unwrap_or_else(|| panic!("no CachedScanExec in {lines:#?}"));
    let value = line
        .split(&format!("{name}="))
        .nth(1)
        .and_then(|rest| rest.split([',', ']']).next())
        .unwrap_or_else(|| panic!("no {name} in {line}"));
    let (number, scale) = COUNT_SUFFIXES
        .iter()
        .find_map(|&(suffix, scale)| value.strip_suffix(suffix).map(|n| (n, scale)))
        .unwrap_or((value, 1.0));
    number.parse::<f64>().expect("a count") * scale
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
    // A row group per partition at least, as a table of any size has: with
    // fewer, a repartition between the scan and the sort keeps the sort from
    // reaching the scan.
    let commits = std::thread::available_parallelism()
        .map_or(MIN_COMMITS, |n| {
            i64::try_from(n.get()).unwrap_or(MIN_COMMITS)
        })
        .max(MIN_COMMITS);
    for commit in 0..commits {
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

    // Read cold past every row the filter keeps, the scan takes every id
    // block, a row group's blocks going to whichever partitions are free:
    // the blocks of a row group read side by side decode it once.
    let every_block = "SELECT seq, \"_id\" AS \"_id\" FROM t WHERE seq % 7 = 0 \
                       ORDER BY _id ASC LIMIT 1000000";
    assert_eq!(
        scan_metric(&conn, every_block, ROW_GROUPS_DECODED),
        commits as f64,
        "{every_block}"
    );
    t.delete(col("seq").lt(lit(30_i64))).expect("delete");

    let every = {
        let mut all = ids(&conn
            .query_sql("SELECT _id FROM t WHERE seq % 7 = 0")
            .expect("sql"));
        all.sort_unstable();
        all
    };
    // On the id, the scan reads by id block (ids in blocks of 1024 rows,
    // three a commit) under the filter no index answers: `k` within one
    // block, across several, and past every row the filter keeps.
    for (order, descending) in [("ASC", false), ("DESC", true)] {
        for k in [1_usize, 10, 500, 1_500, 1_000_000] {
            let mut want = every.clone();
            if descending {
                want.reverse();
            }
            want.truncate(k);
            for keys in [
                format!("_id {order}"),
                format!("note ASC NULLS LAST, _id {order}"),
            ] {
                let sql = format!(
                    "SELECT seq, \"_id\" AS \"_id\" FROM t WHERE seq % 7 = 0 ORDER BY {keys} LIMIT {k}"
                );
                for run in ["decoding", "from kept columns"] {
                    let got = ids(&conn
                        .query_sql(&sql)
                        .unwrap_or_else(|e| panic!("{run}: {e}")));
                    assert_eq!(got, want, "{run}: {sql}");
                }
                let plan = physical_plan(&conn, &sql);
                assert!(plan.contains("CachedScanExec"), "{plan}");
                // A sort whose first key is the id reads by id block.
                if keys.starts_with("_id") {
                    let direction = if descending { "desc" } else { "asc" };
                    assert!(plan.contains(&format!("id_blocks={direction}")), "{plan}");
                    // A small top-k stops after the first block or so each
                    // partition reads. Read by row group, every partition
                    // reads a whole one of 3 blocks before the bound settles,
                    // and the scan reads at least two thirds of the rows.
                    if SMALL_K.contains(&k) {
                        let read = scan_metric(&conn, &sql, ROWS_READ);
                        let rows = (commits * ROWS) as f64;
                        assert!(read * 2.0 < rows, "read {read} of {rows} rows: {sql}");
                    }
                }
            }
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
    let mut kept: Vec<i64> = (30..commits * ROWS).filter(|s| s % 7 == 0).collect();
    for (order, descending) in [("ASC", false), ("DESC", true)] {
        if descending {
            kept.reverse();
        }
        for k in [1_usize, 10, 500, 5_000] {
            let sql = format!("SELECT seq FROM t WHERE seq % 7 = 0 ORDER BY seq {order} LIMIT {k}");
            for run in ["first", "again"] {
                let got = seqs(
                    &conn
                        .query_sql(&sql)
                        .unwrap_or_else(|e| panic!("{run}: {e}")),
                );
                assert_eq!(got, kept[..k.min(kept.len())], "{run}: {sql}");
            }
        }
    }
}
