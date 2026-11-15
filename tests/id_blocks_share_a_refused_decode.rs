// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! A read by id block decodes a row group once even when the decoded-column
//! cache refuses its column: the cache's budget admits the scan by its
//! estimate, but the decoded `_id` array charges a few bytes more than the
//! whole cache, and every later block of the row group takes the decoded
//! column the first left instead of decoding the row group again. Its own
//! binary, because the host configuration (and with it the process's
//! decoded-column cache) is read once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use infino::{
    IndexSpec,
    arrow_array::{Array, Int64Array, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Rows in the table's one commit, a superfile of one row group of many id
/// blocks.
const ROWS: i64 = 60_000;

/// Bytes a decoded `_id` takes per row: a 128-bit decimal.
const ID_BYTES: u64 = 16;

/// How `EXPLAIN ANALYZE` names the row groups the scan decoded.
const ROW_GROUPS_DECODED: &str = "row_groups_decoded";

/// The scan's metric `name` as `EXPLAIN ANALYZE` prints it running `sql`.
fn scan_metric(conn: &infino::Connection, sql: &str, name: &str) -> String {
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
    assert!(line.contains("id_blocks=asc"), "{line}");
    line.split(&format!("{name}="))
        .nth(1)
        .and_then(|rest| rest.split([',', ']']).next())
        .unwrap_or_else(|| panic!("no {name} in {line}"))
        .to_owned()
}

#[test]
fn the_blocks_of_a_row_group_the_cache_refuses_decode_it_once() {
    let budget = ROWS as u64 * ID_BYTES;
    let config = TempDir::new().expect("tempdir");
    std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
    std::fs::write(
        config.path().join("infino/config.yaml"),
        format!("memory:\n  decoded_column_cache_bytes: {budget}\n"),
    )
    .expect("config file");
    // SAFETY: this is the only test in the binary and nothing else has
    // started yet, so no other thread reads the environment.
    unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
    assert_eq!(
        infino::config::global().memory.decoded_column_cache_bytes,
        budget
    );

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![Field::new("seq", DataType::Int64, false)]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new())
        .expect("create t");
    t.append(
        &RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from((0..ROWS).collect::<Vec<_>>()))],
        )
        .expect("batch"),
    )
    .expect("append");

    // Past every row the filter keeps, the scan reads every id block.
    let sql = "SELECT _id FROM t WHERE _id % 7 = 0 ORDER BY _id ASC LIMIT 1000000";
    for run in ["first", "again"] {
        assert_eq!(
            scan_metric(&conn, sql, ROW_GROUPS_DECODED),
            "1",
            "{run}: {sql}"
        );
    }
}
