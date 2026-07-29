// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! One connection, one budget, many handles at once.
//!
//! The per-connection memory budget is a single counter shared (cloned `Arc`)
//! by every table opened on a connection. These tests drive several tables at
//! once from separate threads, ingesting and querying, and check that the
//! shared budget holds: usage never crosses the gate, refusals are graceful
//! (`InfinoError::OverBudget`, never a crash or a mislabelled error), and the
//! committed data stays consistent with the appends that were admitted.
//!
//! Ingest is the gating pressure here (each `append` builds a superfile and
//! reserves its build scratch); the concurrent `bm25_search` calls exercise
//! reader + writer handles coexisting on one budget. Reads over local storage
//! are resident, so they don't reserve anything themselves; they just run
//! alongside the ingest.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, thread};

use infino::{
    Bm25SearchOptions, ConnectOptions, Connection, IndexSpec, InfinoError, Supertable,
    arrow_array::RecordBatch,
    connect_with,
    test_helpers::{build_title_batch, schema_id_title},
};
use tempfile::TempDir;

const N_TABLES: usize = 4;
const APPENDS_PER_TABLE: usize = 6;
/// Rows per appended batch. Large enough that one build's scratch is a real
/// allocation against the budget, not rounding noise.
const BATCH_ROWS: usize = 4000;
const TITLE: &str = "connection memory budget concurrent ingest and query row";
const TOP_K: usize = 10;
/// Configured budget for the bounded run (the enforced gate is 90% of this).
/// Sized to fit a couple of concurrent builds comfortably, so an uncontended
/// append always commits, while four threads building at once can contend.
const BOUNDED_BUDGET_BYTES: u64 = 3_700_000;

/// A `BATCH_ROWS`-row title batch (every row the same text: enough bytes to
/// make the build reserve, and all rows match the `budget` token for BM25).
fn big_title_batch() -> RecordBatch {
    let titles = vec![TITLE; BATCH_ROWS];
    build_title_batch(&titles)
}

/// A connection over a fresh local-fs tempdir with `budget_bytes` (0 = measured).
fn local_conn(budget_bytes: u64) -> (TempDir, Connection) {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf8 tempdir path").to_string();
    let conn = connect_with(
        &uri,
        ConnectOptions::new().with_connection_memory_budget_bytes(budget_bytes),
    )
    .expect("connect_with local fs + budget");
    (dir, conn)
}

/// `n` FTS tables `t0..tn`, each on the same connection (so all share one budget).
fn create_tables(conn: &Connection, n: usize) -> Vec<Supertable> {
    (0..n)
        .map(|i| {
            conn.create_table(
                &format!("t{i}"),
                schema_id_title(),
                IndexSpec::new().fts("title"),
            )
            .expect("create_table")
        })
        .collect()
}

/// Rows currently committed to table `ti`, read back through catalog SQL.
fn committed_rows(conn: &Connection, i: usize) -> usize {
    conn.query_sql(&format!("SELECT _id FROM t{i}"))
        .expect("read committed rows")
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

#[test]
fn concurrent_ingest_and_query_stay_within_one_connection_budget() {
    let (_dir, conn) = local_conn(BOUNDED_BUDGET_BYTES);
    let tables = create_tables(&conn, N_TABLES);
    // One batch, shared read-only across every thread (RecordBatch is Sync).
    let batch = big_title_batch();

    // Each thread owns one table (distinct handles, so the per-handle
    // single-writer slot never collides) and interleaves gated appends with
    // reads while its siblings do the same against the one shared budget. A
    // refused op returns OverBudget and the thread carries on; any other error
    // is a real failure.
    let ok_appends: Vec<usize> = thread::scope(|s| {
        let joins: Vec<_> = tables
            .iter()
            .map(|t| {
                let batch = &batch;
                s.spawn(move || {
                    let mut committed = 0usize;
                    for _ in 0..APPENDS_PER_TABLE {
                        match t.append(batch) {
                            Ok(()) => committed += 1,
                            Err(InfinoError::OverBudget(_)) => {}
                            Err(other) => panic!("append: non-budget error {other:?}"),
                        }
                        match t.bm25_search(
                            "title",
                            "budget",
                            TOP_K,
                            Bm25SearchOptions::new(),
                            None,
                        ) {
                            Ok(_) | Err(InfinoError::OverBudget(_)) => {}
                            Err(other) => panic!("query: non-budget error {other:?}"),
                        }
                    }
                    committed
                })
            })
            .collect();
        joins
            .into_iter()
            .map(|j| j.join().expect("worker thread panicked"))
            .collect()
    });

    let budget = tables[0].local_handle().options().connection_budget();
    let limit = budget.limit().expect("bounded budget has a gate");
    eprintln!(
        "[budget-concurrent] peak={} denials={} gate={limit}",
        budget.peak(),
        budget.denials()
    );

    // Never crossed the ceiling: ingest reserves fallibly, so `used` can never
    // pass the gate, and peak (the high-water mark) stays within it.
    assert!(
        budget.peak() <= limit,
        "peak {} crossed the gate {limit}",
        budget.peak()
    );
    // The budget was actually charged (an uncontended build fit), so the
    // assertion above isn't passing vacuously on an untouched counter.
    assert!(
        budget.peak() > 0,
        "budget never charged; fixture didn't exercise it"
    );

    // Each table holds exactly the rows whose append was admitted: concurrent
    // refusals left no partial state, and the shared counter didn't corrupt a
    // commit on another handle.
    for (i, admitted) in ok_appends.iter().enumerate() {
        assert_eq!(
            committed_rows(&conn, i),
            admitted * BATCH_ROWS,
            "table t{i}: committed rows must match admitted appends",
        );
    }
}

#[test]
fn measured_budget_admits_the_same_concurrent_load() {
    // The same adversarial load under a measured (unbounded) budget: it must
    // never deny, so the refusals in the bounded run are the gate's doing, not
    // a broken fixture or a concurrency bug that trips the counter spuriously.
    let (_dir, conn) = local_conn(0);
    let tables = create_tables(&conn, N_TABLES);
    let batch = big_title_batch();

    thread::scope(|s| {
        for t in &tables {
            let batch = &batch;
            s.spawn(move || {
                for _ in 0..APPENDS_PER_TABLE {
                    t.append(batch)
                        .expect("measured budget never refuses an append");
                    t.bm25_search("title", "budget", TOP_K, Bm25SearchOptions::new(), None)
                        .expect("measured budget never refuses a query");
                }
            });
        }
    });

    let budget = tables[0].local_handle().options().connection_budget();
    assert!(
        budget.limit().is_none(),
        "budget_bytes=0 is measured, not bounded"
    );
    assert_eq!(budget.denials(), 0, "measured budget must never deny");
    assert!(budget.peak() > 0, "measured budget still tracks usage");

    for i in 0..N_TABLES {
        assert_eq!(
            committed_rows(&conn, i),
            APPENDS_PER_TABLE * BATCH_ROWS,
            "table t{i}: every append must commit under a measured budget",
        );
    }
}

#[test]
fn budget_gate_is_shared_across_tables_on_one_connection() {
    // A 1-byte budget floors the 90% gate to 0, so any build is refused. Two
    // distinct tables on one connection: both appends refuse, and the single
    // connection counter records both, proving the budget binds the connection,
    // not one table. Deterministic (no timing), unlike the concurrent runs.
    let (_dir, conn) = local_conn(1);
    let tables = create_tables(&conn, 2);
    let (a, b) = (&tables[0], &tables[1]);
    let batch = big_title_batch();

    assert!(
        matches!(a.append(&batch), Err(InfinoError::OverBudget(_))),
        "first table's append must be refused over budget",
    );
    assert!(
        matches!(b.append(&batch), Err(InfinoError::OverBudget(_))),
        "second table's append must be refused over budget",
    );

    let budget = a.local_handle().options().connection_budget();
    assert!(
        Arc::ptr_eq(budget, b.local_handle().options().connection_budget()),
        "both tables must share one connection budget",
    );
    assert!(
        budget.denials() >= 2,
        "both refusals must land on the one connection counter; got {}",
        budget.denials(),
    );
    assert_eq!(
        budget.peak(),
        0,
        "a refused build reserves nothing, so peak stays 0"
    );
}
