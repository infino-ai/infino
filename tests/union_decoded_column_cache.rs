// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A `UNION ALL` of two tables whose same-named columns carry different
//! field ids, feeding `count(DISTINCT)`, plans and answers the same when the
//! SQL scan keeps decoded columns across statements: the second run reads
//! the columns the first one decoded. Its own binary, because the host
//! configuration (and with it the process's decoded-column cache) is read
//! once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::{
    IndexSpec,
    arrow_array::{Int64Array, LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: far more than the test
/// tables hold, so every column the first statement decodes stays.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// The union's statement: every row of both tables, and their distinct
/// values of the column the two tables number differently.
const UNION_COUNT_DISTINCT: &str = "WITH u AS (SELECT ts, v AS w FROM a UNION ALL \
     SELECT ts, v FROM b) SELECT count(*) AS n, count(DISTINCT w) AS d FROM u";

/// `(count(*), count(DISTINCT w))` of the one row `batches` hold.
fn counts(batches: &[RecordBatch]) -> (i64, i64) {
    let batch = batches
        .iter()
        .find(|batch| batch.num_rows() > 0)
        .expect("a row");
    let column = |name: &str| {
        batch
            .column_by_name(name)
            .and_then(|column| column.as_any().downcast_ref::<Int64Array>())
            .expect("a count")
            .value(0)
    };
    (column("n"), column("d"))
}

#[test]
fn a_union_feeding_count_distinct_answers_the_same_from_kept_decoded_columns() {
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        format!("memory:\n  decoded_column_cache_bytes: {KEPT_BYTES}\n"),
    )
    .expect("config file");
    // The host may already point `XDG_CONFIG_HOME` somewhere (CI runners
    // do); the configuration is read once per process, below, so pointing
    // it at the test's own directory first is all the ownership needed.
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
    assert_eq!(
        infino::config::global().memory.decoded_column_cache_bytes,
        KEPT_BYTES
    );

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    // `a` declares `ts, v`, `b` declares `v, extra, ts`: the same names
    // under different field ids.
    let a_schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Int64, false),
        Field::new("v", DataType::LargeUtf8, true),
    ]));
    let b_schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::LargeUtf8, true),
        Field::new("extra", DataType::Int64, true),
        Field::new("ts", DataType::Int64, false),
    ]));
    let a = conn
        .create_table("a", Arc::clone(&a_schema), IndexSpec::new())
        .expect("create a");
    let b = conn
        .create_table("b", Arc::clone(&b_schema), IndexSpec::new())
        .expect("create b");
    a.append(
        &RecordBatch::try_new(
            a_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(LargeStringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .expect("a batch"),
    )
    .expect("append a");
    b.append(
        &RecordBatch::try_new(
            b_schema,
            vec![
                Arc::new(LargeStringArray::from(vec!["y", "w"])),
                Arc::new(Int64Array::from(vec![Some(0), None])),
                Arc::new(Int64Array::from(vec![4, 5])),
            ],
        )
        .expect("b batch"),
    )
    .expect("append b");

    // The first run decodes the columns and keeps them; the second plans
    // over the kept ones.
    for run in ["decoding", "from kept columns"] {
        let batches = conn
            .query_sql(UNION_COUNT_DISTINCT)
            .unwrap_or_else(|error| panic!("union feeding count(DISTINCT), {run}: {error}"));
        assert_eq!(counts(&batches), (5, 4), "{run}");
    }
}
