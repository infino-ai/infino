// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The process memory limit ends a SQL statement that grows past it, where the
//! connection memory budget does not.
//!
//! The heavy statement repeats each row's full-text column once per element of
//! an `unnest`ed array. Those repeated rows stream between operators, which
//! reserve nothing against the connection budget, and the sort above holds
//! them all. A limit a little above the process's resting memory must refuse
//! the statement part-way through, and the same connection must answer again
//! once the memory has gone back.
//!
//! Its own test binary, and one test, because the limit belongs to the whole
//! process: tests running beside it would move the reading, and each phase
//! sets the limit for everything in the process.

#![cfg(target_os = "linux")]
#![deny(clippy::unwrap_used)]

use std::{
    fs,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use infino::{
    Connection, IndexSpec, InfinoError,
    arrow_array::{LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
    memory::set_process_memory_limit,
};
use tempfile::TempDir;

/// Rows in the table.
const ROWS: usize = 4096;
/// Bytes of text per row: long enough that repeating it is real memory.
const BODY_BYTES: usize = 2048;
/// Distinct filler words the text is drawn from.
const VOCAB: usize = 5000;
/// Step between one filler word and the next, so rows differ.
const WORD_STRIDE: usize = 7;
/// Elements in the `unnest`ed array: each row's text is repeated this often.
const FAN_OUT: usize = 33;
/// How far above the process's resting anonymous memory the limit sits. The
/// statement holds about `ROWS * BODY_BYTES * FAN_OUT` (≈ 270 MiB) of repeated
/// text, far past it; opening the table and planning stay well under.
const HEADROOM_BYTES: u64 = 64 * 1024 * 1024;
/// How long the memory a refused statement held may take to go back.
const DRAIN_DEADLINE: Duration = Duration::from_secs(30);
/// How often the drain is checked.
const DRAIN_POLL: Duration = Duration::from_millis(100);
/// `/proc/self/status` reports `RssAnon` in kB.
const KIB: u64 = 1024;
/// A statement that reads almost nothing.
const SMALL_QUERY: &str = "SELECT name FROM docs WHERE name = 'doc 7'";

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
fn load_docs(dir: &TempDir) -> Connection {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]));
    let names: Vec<String> = (0..ROWS).map(|i| format!("doc {i}")).collect();
    let bodies: Vec<String> = (0..ROWS)
        .map(|i| {
            let mut body = String::with_capacity(BODY_BYTES);
            let mut word = i;
            while body.len() < BODY_BYTES {
                body.push_str(&format!("w{} ", word % VOCAB));
                word += WORD_STRIDE;
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
    let db = connect(dir.path().to_str().expect("utf8 tempdir path")).expect("connect");
    db.create_table("docs", schema, IndexSpec::new().fts("body"))
        .expect("create docs")
        .append(&batch)
        .expect("append docs");
    db
}

/// Rows `sql` returns on `conn`.
fn rows(conn: &Connection, sql: &str) -> Result<usize, InfinoError> {
    Ok(conn.query_sql(sql)?.iter().map(|b| b.num_rows()).sum())
}

fn is_process_limit_refusal(err: &InfinoError) -> bool {
    matches!(err, InfinoError::OverBudget(msg) if msg.contains("process memory limit"))
}

#[test]
fn a_statement_that_grows_past_the_process_memory_limit_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let conn = load_docs(&dir);

    // A limit above any process lets a statement through.
    set_process_memory_limit(Some(u64::MAX));
    assert_eq!(rows(&conn, SMALL_QUERY).expect("under the limit"), 1);

    // A process already over the limit is refused before the plan starts:
    // any live process holds more than one byte of anonymous memory.
    set_process_memory_limit(Some(1));
    let err = rows(&conn, SMALL_QUERY).expect_err("the process is over a 1-byte limit");
    assert!(is_process_limit_refusal(&err), "got {err:?}");

    // A statement that grows past the limit is refused part-way through.
    let limit = rss_anon_bytes() + HEADROOM_BYTES;
    set_process_memory_limit(Some(limit));
    let orgs = (0..FAN_OUT)
        .map(|i| format!("'org{i}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let heavy = format!(
        "SELECT org, body FROM (SELECT unnest(ARRAY[{orgs}]) AS org, body FROM docs) sub \
         ORDER BY org"
    );
    let err = rows(&conn, &heavy).expect_err("the repeated text passes the limit");
    assert!(is_process_limit_refusal(&err), "got {err:?}");

    // The refusal ended one statement, not the connection: once the memory the
    // statement held has gone back under the limit, the same connection
    // answers.
    let started = Instant::now();
    while rss_anon_bytes() > limit {
        assert!(
            started.elapsed() < DRAIN_DEADLINE,
            "the refused statement's memory did not go back under the limit within {DRAIN_DEADLINE:?}"
        );
        thread::sleep(DRAIN_POLL);
    }
    assert_eq!(
        rows(&conn, SMALL_QUERY).expect("the connection answers after a refusal"),
        1
    );
}
