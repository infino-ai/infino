// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A process memory limit ends a SQL statement that grows past it, where the
//! connection memory budget does not.
//!
//! The statement repeats each row's full-text column once per element of an
//! `unnest`ed array. Those repeated rows stream between operators, which
//! reserve nothing against the connection budget, and the sort above holds
//! them all. A limit a little above the process's resting memory must refuse
//! the statement as `OverBudget` part-way through; the process keeps serving.
//!
//! Its own test binary because the limit reads the whole process: in a binary
//! running other tests in parallel, their memory would move the reading.

#![cfg(target_os = "linux")]
#![deny(clippy::unwrap_used)]

use std::{fs, sync::Arc};

use infino::{
    ConnectOptions, IndexSpec, InfinoError,
    arrow_array::{LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect, connect_with,
};
use tempfile::TempDir;

/// Rows in the table.
const ROWS: usize = 4096;
/// Bytes of text per row: long enough that repeating it is real memory.
const BODY_BYTES: usize = 2048;
/// Elements in the `unnest`ed array: each row's text is repeated this often.
const FAN_OUT: usize = 33;
/// How far above the process's resting anonymous memory the limit sits. The
/// statement holds about `ROWS * BODY_BYTES * FAN_OUT` (≈ 270 MiB) of repeated
/// text, far past it; opening the connection and planning stay well under.
const HEADROOM_BYTES: u64 = 64 * 1024 * 1024;
/// `/proc/self/status` reports `RssAnon` in kB.
const KIB: u64 = 1024;

/// The process's anonymous resident bytes, as the limit reads them.
fn rss_anon_bytes() -> u64 {
    let status = fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let kb: u64 = status
        .lines()
        .find_map(|line| line.strip_prefix("RssAnon:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|kb| kb.parse().ok())
        .expect("RssAnon in /proc/self/status");
    kb * KIB
}

/// A table `docs(name, body)` with `body` full-text indexed, in `dir`.
fn load_docs(dir: &TempDir) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]));
    let names: Vec<String> = (0..ROWS).map(|i| format!("doc {i}")).collect();
    let bodies: Vec<String> = (0..ROWS)
        .map(|i| {
            let mut body = String::with_capacity(BODY_BYTES + 16);
            let mut word = i;
            while body.len() < BODY_BYTES {
                body.push_str(&format!("w{} ", word % 5000));
                word += 7;
            }
            body
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(LargeStringArray::from(names)),
            Arc::new(LargeStringArray::from(bodies)),
        ],
    )
    .expect("docs batch");
    let db = connect(dir.path().to_str().expect("utf8 tempdir path")).expect("writer connect");
    db.create_table("docs", schema, IndexSpec::new().fts("body"))
        .expect("create docs")
        .append(&batch)
        .expect("append docs");
}

#[test]
fn a_statement_that_grows_past_the_process_memory_limit_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    load_docs(&dir);
    let uri = dir.path().to_str().expect("utf8 tempdir path").to_string();

    let limit = rss_anon_bytes() + HEADROOM_BYTES;
    let conn = connect_with(
        &uri,
        ConnectOptions::new().with_process_memory_limit_bytes(limit),
    )
    .expect("connect with a process memory limit");
    let orgs = (0..FAN_OUT)
        .map(|i| format!("'org{i}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT org, body FROM (SELECT unnest(ARRAY[{orgs}]) AS org, body FROM docs) sub \
         ORDER BY org"
    );

    let err = conn
        .query_sql(&sql)
        .expect_err("the repeated text passes the process memory limit");
    assert!(
        matches!(&err, InfinoError::OverBudget(msg) if msg.contains("process memory limit")),
        "expected the process-limit OverBudget, got {err:?}"
    );

    // The refusal ended one statement, not the process: a connection without
    // the limit still answers.
    let rows: usize = connect(&uri)
        .expect("connect without a limit")
        .query_sql("SELECT name FROM docs WHERE name = 'doc 7'")
        .expect("the engine still serves")
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert_eq!(rows, 1);
}
