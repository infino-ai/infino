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
//! The limit also covers planning. A scan opens the files it keeps while it is planned, and on a
//! table of tens of thousands of files their metadata alone can pass the limit before execution
//! starts. With planning held on its first read (the table's term index, which the scan's prune
//! loads before it opens a file), a statement whose process passes the limit must be refused while
//! it is still planning, through a connection and through a table's own SQL, which plan on separate
//! paths. Once the memory is back under the same limit, the same statement answers. A delete's
//! predicate scan, refused the same way, must stop rather than read on.
//!
//! Its own test binary, and one test, because the limit belongs to the whole
//! process: tests running beside it would move the reading, and each phase
//! sets the limit for everything in the process.

#![cfg(target_os = "linux")]
#![deny(clippy::unwrap_used)]

use std::{
    fs,
    hint::black_box,
    sync::{Arc, Mutex, PoisonError, mpsc},
    thread,
    time::{Duration, Instant},
};

use datafusion::prelude::{col, lit};
use infino::{
    ConnectOptions, Connection, Consistency, IndexSpec, InfinoError,
    arrow_array::{LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect, connect_with,
    memory::set_process_memory_limit,
    runtime_metrics::op_stats::with_op_stats,
    supertable::{
        Supertable, SupertableOptions, SupertableReader,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options, slow_storage::SlowStorage},
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
/// A statement answered from the manifest: it opens the table and reads no superfile.
const OPEN_QUERY: &str = "SELECT count(*) FROM docs";
/// A statement whose scan reads storage while it plans: its prune loads the table's term index
/// before it opens a file. `name` has no term index of its own, so the prune keeps every file; the
/// read is what holds planning.
const SCAN_QUERY: &str = "SELECT count(*) FROM docs WHERE name LIKE 'doc 1%'";
/// [`OPEN_QUERY`] through a table's own SQL.
const TABLE_OPEN_QUERY: &str = "SELECT count(*) FROM supertable";
/// [`SCAN_QUERY`] through a table's own SQL.
const TABLE_SCAN_QUERY: &str = "SELECT count(*) FROM supertable WHERE title LIKE 'doc 1%'";
/// The titles [`TABLE_SCAN_QUERY`] matches, as a delete's predicate.
const DELETED_TITLES: &str = "doc 1%";
/// A fragment of the path of every term index object: the delete's scan reads one while it plans.
const TERM_INDEX_OBJECTS: &str = "term-index/";
/// Reads are held, not slowed: the test acts while planning waits on one.
const NO_READ_DELAY: Duration = Duration::ZERO;
/// Bytes the test itself takes to push the process past the limit: well past [`HEADROOM_BYTES`], so
/// the sampler sees it however the allocator rounds.
const PUSH_BYTES: usize = 4 * HEADROOM_BYTES as usize;
/// A byte written to every page the push takes, so the pages are resident.
const PUSH_FILL: u8 = 0x5a;
/// How long a refusal may take once the process is over the limit: the sampler reads every 100 ms,
/// so this is generous.
const REFUSAL_BOUND: Duration = Duration::from_secs(10);
/// How long the scan may take to reach its first read.
const PLANNING_READ_DEADLINE: Duration = Duration::from_secs(30);

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

/// A connection to `dir` whose storage reads go through [`SlowStorage`], and the stores it built.
/// Its tables keep the snapshot they opened with, so a pointer reread can't get held before the
/// statement starts its race.
fn holding_conn(dir: &TempDir) -> (Connection, Arc<Mutex<Vec<Arc<SlowStorage>>>>) {
    let stores = Arc::new(Mutex::new(Vec::<Arc<SlowStorage>>::new()));
    let built = Arc::clone(&stores);
    let options = ConnectOptions::new()
        .with_read_consistency(Consistency::Snapshot)
        .with_storage_wrapper(move |inner| {
            let slow = SlowStorage::wrap(inner, NO_READ_DELAY);
            built
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(Arc::clone(&slow));
            slow
        });
    let conn = connect_with(dir.path().to_str().expect("utf8 tempdir path"), options)
        .expect("connect with held reads");
    (conn, stores)
}

/// Apply `act` to every store built so far.
fn each_store(stores: &Mutex<Vec<Arc<SlowStorage>>>, act: impl Fn(&SlowStorage)) {
    stores
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .for_each(|store| act(store));
}

/// Reads started on every store built so far.
fn reads(stores: &Mutex<Vec<Arc<SlowStorage>>>) -> usize {
    stores
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .map(|store| store.reads())
        .sum()
}

/// Options for a table of full-text indexed titles on `storage`. The index gives the table a term
/// index, which planning reads to prune.
fn titles_options(storage: Arc<dyn StorageProvider>) -> SupertableOptions {
    default_supertable_options().with_storage(storage)
}

/// A table of [`ROWS`] titles `doc 0`, `doc 1`, ... in one file in `dir`, and the storage it is on.
fn titles_table(dir: &TempDir) -> Arc<dyn StorageProvider> {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("titles storage"));
    let titles: Vec<String> = (0..ROWS).map(|i| format!("doc {i}")).collect();
    let titles: Vec<&str> = titles.iter().map(String::as_str).collect();
    let table = Supertable::create(titles_options(Arc::clone(&storage))).expect("create titles");
    let mut writer = table.writer().expect("titles writer");
    writer
        .append(&build_title_batch(&titles))
        .expect("append titles");
    writer.commit().expect("commit titles");
    storage
}

/// Rows `sql` returns through `reader`'s own SQL.
fn table_rows(reader: &SupertableReader, sql: &str) -> Result<usize, InfinoError> {
    let batches = reader.query_sql(sql).map_err(InfinoError::from)?;
    Ok(batches.iter().map(|b| b.num_rows()).sum())
}

/// Wait until the process's anonymous memory is back under `limit`.
fn wait_under(limit: u64) {
    let started = Instant::now();
    while rss_anon_bytes() > limit {
        assert!(
            started.elapsed() < DRAIN_DEADLINE,
            "the refused statement's memory did not go back under the limit within {DRAIN_DEADLINE:?}"
        );
        thread::sleep(DRAIN_POLL);
    }
}

/// Set the limit [`HEADROOM_BYTES`] above the process's memory now, run `query` on a thread of its
/// own, wait until it starts a read (`reads` counts them, the caller holds them), then push the
/// process past the limit. Returns the limit and how `query` ended.
///
/// A held read never finishes, so only a refusal can end `query`: a statement not raced against the
/// limit while it plans waits out [`REFUSAL_BOUND`] and fails the test.
fn refused_while_planning(
    reads: impl Fn() -> usize,
    query: impl FnOnce() -> Result<usize, InfinoError> + Send + 'static,
) -> (u64, Result<usize, InfinoError>) {
    let reads_before = reads();
    let limit = rss_anon_bytes() + HEADROOM_BYTES;
    set_process_memory_limit(Some(limit));
    let (sent, received) = mpsc::channel();
    thread::spawn(move || {
        let _ = sent.send(query());
    });
    let started = Instant::now();
    while reads() == reads_before {
        assert!(
            started.elapsed() < PLANNING_READ_DEADLINE,
            "the scan made no read within {PLANNING_READ_DEADLINE:?}"
        );
        thread::sleep(DRAIN_POLL);
    }
    let push = black_box(vec![PUSH_FILL; PUSH_BYTES]);
    let ended = received
        .recv_timeout(REFUSAL_BOUND)
        .expect("a statement still planning is refused once the process is over the limit");
    drop(push);
    (limit, ended)
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
    wait_under(limit);
    assert_eq!(
        rows(&conn, SMALL_QUERY).expect("the connection answers after a refusal"),
        1
    );

    // A statement still planning is refused once the process passes the limit. A fresh connection
    // opens the table from the manifest, then holds every read, so the scan's first read waits in
    // planning.
    let (planning_conn, stores) = holding_conn(&dir);
    set_process_memory_limit(Some(u64::MAX));
    rows(&planning_conn, OPEN_QUERY).expect("opens docs");
    each_store(&stores, SlowStorage::hold_reads);
    let querying = planning_conn.clone();
    let (limit, ended) =
        refused_while_planning(|| reads(&stores), move || rows(&querying, SCAN_QUERY));
    let err = ended.expect_err("the process is over the limit");
    assert!(is_process_limit_refusal(&err), "got {err:?}");
    wait_under(limit);
    each_store(&stores, SlowStorage::release_reads);
    assert_eq!(
        rows(&planning_conn, SCAN_QUERY).expect("the connection answers under the same limit"),
        1
    );

    // The same through a table's own SQL, which plans on its own path.
    let table_dir = TempDir::new().expect("titles tempdir");
    let held = SlowStorage::wrap(titles_table(&table_dir), NO_READ_DELAY);
    set_process_memory_limit(Some(u64::MAX));
    let reader = Supertable::open(titles_options(held.clone()))
        .expect("open titles")
        .reader()
        .expect("titles reader");
    table_rows(&reader, TABLE_OPEN_QUERY).expect("opens titles");
    held.hold_reads();
    let querying = reader.clone();
    let (limit, ended) = refused_while_planning(
        || held.reads(),
        move || table_rows(&querying, TABLE_SCAN_QUERY),
    );
    let err = ended.expect_err("the process is over the limit");
    assert!(is_process_limit_refusal(&err), "got {err:?}");
    wait_under(limit);
    held.release_reads();
    // Under a limit the statement runs on a task of its own, and still reports the one superfile
    // its plan weighed.
    let (answered, stats) = with_op_stats(|| table_rows(&reader, TABLE_SCAN_QUERY));
    assert_eq!(answered.expect("the table answers under the same limit"), 1);
    assert_eq!(stats.superfiles_considered, 1);

    // A delete resolves its predicate with a scan of its own, which must stop with the refusal, or
    // it reads on past the limit with nothing watching it.
    //  - the delete rereads the table's pointer first, so only term index reads are held.
    //  - a freshly opened table has none of them cached, so the scan must read one.
    // This test pins that the held read is dropped with the refused scan.
    held.hold_reads_of(TERM_INDEX_OBJECTS);
    let mut writer = Supertable::open(titles_options(held.clone()))
        .expect("reopen titles")
        .writer()
        .expect("titles writer");
    let (limit, ended) = refused_while_planning(
        || held.reads_of(TERM_INDEX_OBJECTS),
        move || {
            writer
                .delete(col("title").like(lit(DELETED_TITLES)))
                .map(|pending| pending.matched)
                .map_err(InfinoError::from)
        },
    );
    let err = ended.expect_err("the process is over the limit");
    assert!(is_process_limit_refusal(&err), "got {err:?}");
    let started = Instant::now();
    while held.in_flight() > 0 {
        assert!(
            started.elapsed() < REFUSAL_BOUND,
            "the refused delete's scan still has a read in flight after {REFUSAL_BOUND:?}"
        );
        thread::sleep(DRAIN_POLL);
    }
    wait_under(limit);
    held.release_reads();
}
