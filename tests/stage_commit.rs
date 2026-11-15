// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Integration coverage for [`Supertable::stage`] and [`Supertable::commit_staged`]:
//! staged rows stay out of the table until committed, land together in one
//! commit, search like appended rows, and survive writers on other
//! connections committing to the same table at the same time; a staged batch
//! lands once, and one that cannot land is refused with nothing published.

#![deny(clippy::unwrap_used)]

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Barrier},
    thread,
};

use infino::{
    Bm25SearchOptions, Connection, IndexSpec, InfinoError, Metric, StagedAppend, Supertable,
    arrow_array::{
        Array, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
        StringArray,
    },
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use tempfile::TempDir;

const TABLE: &str = "docs";
/// Rows in each staged batch.
const BATCH_ROWS: i64 = 40;
/// Batches each test stages.
const BATCHES: i64 = 3;
/// Writers staging into the same table at once in the concurrency test.
const WRITERS: i64 = 4;
/// Hits to ask a search for: more than any test's matching rows.
const TOP_K: usize = 1_000;
/// Threads committing the same staged batch at once.
const RACERS: usize = 8;
/// Dimension of the vector column in the vector-table test.
const DIM: i32 = 16;

fn user_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]))
}

/// Rows `lo..lo + BATCH_ROWS`, each body naming its batch (`batch<b>`) so a
/// search can tell which batches are in the table.
fn batch(b: i64) -> RecordBatch {
    let lo = b * BATCH_ROWS;
    let n = Int64Array::from((lo..lo + BATCH_ROWS).collect::<Vec<_>>());
    let body = LargeStringArray::from(
        (lo..lo + BATCH_ROWS)
            .map(|i| format!("row{i} batch{b} common"))
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(user_schema(), vec![Arc::new(n), Arc::new(body)]).expect("valid batch")
}

fn open_db(dir: &TempDir) -> Connection {
    connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect")
}

fn create(db: &Connection) -> Supertable {
    db.create_table(TABLE, user_schema(), IndexSpec::new().fts("body"))
        .expect("create_table")
}

fn count(db: &Connection) -> i64 {
    let batches = db
        .query_sql(&format!("SELECT COUNT(*) FROM {TABLE}"))
        .expect("count");
    batches
        .iter()
        .find(|b| b.num_rows() > 0)
        .and_then(|b| b.column(0).as_any().downcast_ref::<Int64Array>())
        .map(|a| a.value(0))
        .expect("one count")
}

/// Every file under `root`.
fn files_under(root: &Path) -> BTreeSet<PathBuf> {
    let mut found = BTreeSet::new();
    let mut dirs = vec![root.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                found.insert(path);
            }
        }
    }
    found
}

/// One row with a column the table's schema lacks.
fn wider_batch() -> RecordBatch {
    let wider = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("extra", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        wider,
        vec![
            Arc::new(Int64Array::from(vec![-1])),
            Arc::new(LargeStringArray::from(vec!["wider"])),
            Arc::new(StringArray::from(vec![Some("x")])),
        ],
    )
    .expect("valid batch")
}

fn assert_unsupported(result: Result<(), InfinoError>, why: &str) {
    match result {
        Err(InfinoError::Unsupported(_)) => {}
        other => panic!("{why}: expected Unsupported, got {other:?}"),
    }
}

fn hits(table: &Supertable, term: &str) -> usize {
    table
        .bm25_search("body", term, TOP_K, Bm25SearchOptions::new(), None)
        .expect("bm25_search")
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

#[test]
fn staged_rows_join_the_table_only_when_committed() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let staged: Vec<StagedAppend> = (0..BATCHES)
        .map(|b| table.stage(&batch(b)).expect("stage"))
        .collect();
    assert!(staged.iter().all(|s| s.rows() == BATCH_ROWS as u64));
    assert_eq!(count(&db), 0, "staged rows are not in the table");
    assert_eq!(hits(&table, "common"), 0);

    table.commit_staged(&staged).expect("commit_staged");
    assert_eq!(count(&db), BATCHES * BATCH_ROWS);
    for b in 0..BATCHES {
        assert_eq!(
            hits(&table, &format!("batch{b}")),
            BATCH_ROWS as usize,
            "batch {b} searches like appended rows"
        );
    }
    assert_eq!(hits(&table, "common"), (BATCHES * BATCH_ROWS) as usize);

    // Reopened, the table lists the committed superfiles.
    let reopened = open_db(&dir);
    assert_eq!(count(&reopened), BATCHES * BATCH_ROWS);
}

#[test]
fn staged_and_appended_rows_mix_in_one_table() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let staged = table.stage(&batch(0)).expect("stage");
    table.append(&batch(1)).expect("append");
    assert_eq!(
        count(&db),
        BATCH_ROWS,
        "the append landed, the staged batch not yet"
    );
    table.commit_staged(&[staged]).expect("commit_staged");
    assert_eq!(count(&db), 2 * BATCH_ROWS);
}

/// A staged batch lands once: committed again, alone or beside a fresh
/// batch, it is passed over.
#[test]
fn a_staged_batch_committed_again_is_passed_over() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let landed = table.stage(&batch(0)).expect("stage");
    let once = std::slice::from_ref(&landed);
    table.commit_staged(once).expect("first commit_staged");
    table.commit_staged(once).expect("committed again");
    assert_eq!(count(&db), BATCH_ROWS);

    let fresh = table.stage(&batch(1)).expect("stage");
    table
        .commit_staged(&[fresh, landed])
        .expect("the fresh batch lands beside the committed one");
    assert_eq!(count(&db), 2 * BATCH_ROWS);

    assert_eq!(hits(&table, "batch0"), BATCH_ROWS as usize);
}

/// A commit refused for one staged batch publishes none of them and leaves
/// the others free to commit.
#[test]
fn a_refused_commit_releases_the_batches_it_held() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);
    let other = db
        .create_table("other", user_schema(), IndexSpec::new().fts("body"))
        .expect("create_table");

    let fresh = table.stage(&batch(0)).expect("stage");
    let foreign = other.stage(&batch(1)).expect("stage on the other table");
    let both = [fresh, foreign];
    assert_unsupported(
        table.commit_staged(&both),
        "one of the batches was staged on another table",
    );
    assert_eq!(count(&db), 0, "nothing of the refused commit landed");
    table
        .commit_staged(&both[..1])
        .expect("the fresh batch commits on its own");
    assert_eq!(count(&db), BATCH_ROWS);
}

/// Threads committing the same staged batch at once land it once; a thread
/// that finds another committing it is told to retry, and its retry finds
/// the batch landed.
#[test]
fn one_staged_batch_committed_from_several_threads_lands_once() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);
    let staged = [table.stage(&batch(0)).expect("stage")];

    let barrier = Barrier::new(RACERS);
    thread::scope(|s| {
        let racers: Vec<_> = (0..RACERS)
            .map(|_| {
                s.spawn(|| {
                    barrier.wait();
                    loop {
                        match table.commit_staged(&staged) {
                            Err(InfinoError::Conflict(_)) => thread::yield_now(),
                            done => break done,
                        }
                    }
                })
            })
            .collect();
        for racer in racers {
            racer.join().expect("racer").expect("commit_staged");
        }
    });
    assert_eq!(count(&db), BATCH_ROWS);
    assert_eq!(hits(&table, "batch0"), BATCH_ROWS as usize);
}

/// A batch staged on one table is refused by another, which stays empty.
#[test]
fn a_batch_staged_on_another_table_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);
    let other = db
        .create_table("other", user_schema(), IndexSpec::new().fts("body"))
        .expect("create_table");

    let staged = [table.stage(&batch(0)).expect("stage")];
    assert_unsupported(
        other.commit_staged(&staged),
        "its superfiles are in the first table's storage",
    );
    assert_eq!(hits(&other, "common"), 0);
    table
        .commit_staged(&staged)
        .expect("its own table still commits it");
    assert_eq!(count(&db), BATCH_ROWS);
}

/// A staged batch whose superfiles were deleted meanwhile - as a
/// garbage-collection sweep deletes objects no manifest lists - is refused
/// rather than published as superfiles no read can open.
#[test]
fn a_batch_whose_superfiles_were_reclaimed_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let before = files_under(dir.path());
    let staged = [table.stage(&batch(0)).expect("stage")];
    let written: Vec<PathBuf> = files_under(dir.path())
        .difference(&before)
        .cloned()
        .collect();
    assert!(!written.is_empty(), "staging writes the superfiles");
    for path in &written {
        fs::remove_file(path).expect("remove a staged superfile");
    }

    assert_unsupported(
        table.commit_staged(&staged),
        "the staged superfiles are gone",
    );
    assert_eq!(count(&db), 0);
    let again = [table.stage(&batch(0)).expect("stage the rows again")];
    table.commit_staged(&again).expect("commit_staged");
    assert_eq!(count(&db), BATCH_ROWS);
}

/// A batch staged before the table's schema moved is refused; staged again
/// under the new schema, its rows land.
#[test]
fn a_batch_staged_before_the_schema_moved_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let staged = [table.stage(&batch(0)).expect("stage")];
    table.append(&wider_batch()).expect("append adds a column");
    assert_unsupported(
        table.commit_staged(&staged),
        "the table's schema moved since staging",
    );
    assert_eq!(count(&db), 1);
    let again = [table.stage(&batch(0)).expect("stage the rows again")];
    table.commit_staged(&again).expect("commit_staged");
    assert_eq!(count(&db), 1 + BATCH_ROWS);
}

/// Committing nothing, or only empty staged batches, writes no commit.
#[test]
fn committing_nothing_is_a_no_op() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);
    table.commit_staged(&[]).expect("an empty commit_staged");
    assert_eq!(count(&db), 0);
}

#[test]
fn a_table_with_vector_columns_refuses_staging() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(Arc::clone(&item), DIM),
            false,
        ),
    ]));
    let table = db
        .create_table(
            TABLE,
            Arc::clone(&schema),
            IndexSpec::new().vector("emb", DIM as usize, Metric::Cosine),
        )
        .expect("create_table");
    let values = Float32Array::from(vec![0.5_f32; DIM as usize]);
    let emb = FixedSizeListArray::try_new(item, DIM, Arc::new(values), None).expect("one vector");
    let rows = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![1])), Arc::new(emb)],
    )
    .expect("valid batch");
    match table.stage(&rows) {
        Err(InfinoError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(count(&db), 0);
}

#[test]
fn a_batch_that_adds_a_column_is_refused_and_appends() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let rows = wider_batch();
    match table.stage(&rows) {
        Err(InfinoError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    table.append(&rows).expect("append adds the column");
    assert_eq!(count(&db), 1);
}

#[test]
fn a_table_with_no_storage_refuses_staging() {
    let db = connect("memory://").expect("connect");
    let table = create(&db);
    match table.stage(&batch(0)) {
        Err(InfinoError::Unsupported(_)) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(count(&db), 0);
}

#[test]
fn writers_on_several_connections_commit_to_one_table() {
    let dir = TempDir::new().expect("tempdir");
    create(&open_db(&dir));

    thread::scope(|s| {
        for w in 0..WRITERS {
            let dir = &dir;
            s.spawn(move || {
                let db = open_db(dir);
                let table = db.open_table(TABLE).expect("open_table");
                let staged: Vec<StagedAppend> = (0..BATCHES)
                    .map(|b| table.stage(&batch(w * BATCHES + b)).expect("stage"))
                    .collect();
                table.commit_staged(&staged).expect("commit_staged");
            });
        }
    });

    let db = open_db(&dir);
    assert_eq!(count(&db), WRITERS * BATCHES * BATCH_ROWS);
    let table = db.open_table(TABLE).expect("open_table");
    assert_eq!(
        hits(&table, "common"),
        (WRITERS * BATCHES * BATCH_ROWS) as usize
    );
}
