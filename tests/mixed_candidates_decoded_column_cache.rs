// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A scan whose index narrows some superfiles and reads the others whole
//! takes their rows from kept decoded columns, as a scan the index narrows
//! nowhere does, and answers the same: on the run that decodes the columns
//! and on the runs that read them kept, deleted rows left out. Its own
//! binary, because the host configuration (and with it the process's
//! decoded-column cache) is read once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use datafusion::prelude::{col, lit};
use infino::{
    Connection, FtsField, IndexSpec,
    arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: far more than the table
/// holds, so every column the first statement decodes stays.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// Rows in each superfile.
const ROWS: i64 = 5_000;

/// Rows of the first superfile holding `x`: few enough for the index to
/// select them.
const SPARSE: i64 = 100;

/// Rows of the second superfile holding `x`: more than half of it, so the
/// index does not select them and the superfile is read whole.
const DENSE: i64 = 3_000;

/// Rows deleted: those whose `seq` leaves [`DELETED_REMAINDER`] divided by
/// this, some of each superfile's `x` rows among them.
const DELETED_EVERY: i64 = 10;

/// See [`DELETED_EVERY`].
const DELETED_REMAINDER: i64 = 3;

/// The statement: every live `x` row, in `seq` order.
const SELECT_X: &str = "SELECT seq FROM t WHERE k = 'x' ORDER BY seq";

/// Where an analyzed plan names the filter, and how it counts the rows it
/// kept out of those it was handed: `selectivity=60.78% (2.79 K/4.59 K)`.
const FILTER_OPERATOR: &str = "FilterExec";
const SELECTIVITY: &str = "selectivity=";

/// How a plan shows the row groups the scan of kept decoded columns reads:
/// each superfile is one row group, and both hold live `x` rows.
const READS_BOTH_ROW_GROUPS: &str = "CachedScanExec: row_groups=2,";

/// The same once every `x` row of the first superfile is deleted: its row
/// group holds no live row the index selected, so it is not read.
const READS_ONE_ROW_GROUP: &str = "CachedScanExec: row_groups=1,";

/// Rows a count DataFusion prints with a `K` suffix may be off by: it keeps
/// two decimals of thousands.
const PRINTED_THOUSANDS_ROUNDING: f64 = 10.0;

/// The `seq` of row `i` of superfile `file`, and its `k`.
fn row(file: i64, i: i64) -> (i64, String) {
    let seq = file * ROWS + i;
    let x = match file {
        0 => i < SPARSE,
        _ => i < DENSE,
    };
    (seq, if x { "x".to_owned() } else { format!("v{seq}") })
}

/// Every `seq` the statement should return.
fn expected() -> Vec<i64> {
    (0..2)
        .flat_map(|file| (0..ROWS).map(move |i| row(file, i)))
        .filter(|(seq, k)| k == "x" && seq % DELETED_EVERY != DELETED_REMAINDER)
        .map(|(seq, _)| seq)
        .collect()
}

/// Every live row the scan should hand the filter: the first superfile's
/// `x` rows, which the index selects, and every row of the second, which
/// it reads whole.
fn expected_scanned() -> usize {
    (0..2)
        .flat_map(|file| (0..ROWS).map(move |i| (file, row(file, i))))
        .filter(|(file, (seq, k))| {
            (*file == 1 || k == "x") && seq % DELETED_EVERY != DELETED_REMAINDER
        })
        .count()
}

/// The rows the filter of an analyzed plan was handed (the denominator of
/// its selectivity, which DataFusion prints as `24` or `4.59 K`).
fn filter_input_rows(plan: &str) -> f64 {
    let filter = &plan[plan.find(FILTER_OPERATOR).expect("the plan has a filter")..];
    let selectivity = &filter[filter
        .find(SELECTIVITY)
        .expect("the filter counts its rows")
        + SELECTIVITY.len()..];
    let input = &selectivity[selectivity.find('/').expect("kept / handed") + 1..];
    let input = &input[..input.find(')').expect("the count ends")];
    let (number, scale) = match input.split_once(' ') {
        Some((number, "K")) => (number, 1e3),
        Some((number, "M")) => (number, 1e6),
        _ => (input, 1.0),
    };
    number.parse::<f64>().expect("a row count") * scale
}

/// The `seq` column across `batches`, in order.
fn seqs(batches: &[RecordBatch]) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let seqs = b
                .column_by_name("seq")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                .expect("seq")
                .clone();
            (0..seqs.len()).map(move |i| seqs.value(i))
        })
        .collect()
}

/// The plan `explain` (`EXPLAIN` or `EXPLAIN ANALYZE`) prints for
/// [`SELECT_X`], as text.
fn explain(conn: &Connection, explain: &str) -> String {
    conn.query_sql(&format!("{explain} {SELECT_X}"))
        .unwrap_or_else(|error| panic!("{explain}: {error}"))
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
                .map(|i| plans.value(i).to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn a_scan_the_index_narrows_in_some_superfiles_reads_kept_decoded_columns() {
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
        Field::new("k", DataType::LargeUtf8, false),
    ]));
    // `k` indexed as a keyword is: its whole values, no positions.
    let t = conn
        .create_table(
            "t",
            Arc::clone(&schema),
            IndexSpec::new().fts(FtsField::new("k").positions(false)),
        )
        .expect("create t");
    for file in 0..2 {
        let (seq, k): (Vec<i64>, Vec<String>) = (0..ROWS).map(|i| row(file, i)).unzip();
        t.append(
            &RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seq)),
                    Arc::new(LargeStringArray::from(k)),
                ],
            )
            .expect("batch"),
        )
        .expect("append");
    }
    t.delete((col("seq") % lit(DELETED_EVERY)).eq(lit(DELETED_REMAINDER)))
        .expect("delete");

    let plan = explain(&conn, "EXPLAIN");
    assert!(plan.contains(READS_BOTH_ROW_GROUPS), "{plan}");
    for run in ["decoding", "from kept columns", "from kept columns again"] {
        let batches = conn
            .query_sql(SELECT_X)
            .unwrap_or_else(|error| panic!("{run}: {error}"));
        assert_eq!(seqs(&batches), expected(), "{run}");
    }
    // Only the selected rows of the narrowed superfile reach the filter;
    // reading it whole too would hand over every live row of both.
    let analyzed = explain(&conn, "EXPLAIN ANALYZE");
    let handed = filter_input_rows(&analyzed);
    assert!(
        (handed - expected_scanned() as f64).abs() <= PRINTED_THOUSANDS_ROUNDING,
        "the filter was handed {handed} rows, not {}: {analyzed}",
        expected_scanned()
    );

    // Delete the rest of the first superfile's `x` rows: the index still
    // selects them, but none is live, so its row group is not read.
    t.delete(col("k").eq(lit("x")).and(col("seq").lt(lit(ROWS))))
        .expect("delete the sparse rows");
    let plan = explain(&conn, "EXPLAIN");
    assert!(plan.contains(READS_ONE_ROW_GROUP), "{plan}");
    let batches = conn.query_sql(SELECT_X).expect("after the second delete");
    let rest: Vec<i64> = expected().into_iter().filter(|seq| *seq >= ROWS).collect();
    assert_eq!(seqs(&batches), rest);
}
