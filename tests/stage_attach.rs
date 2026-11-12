// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Integration coverage for [`Supertable::stage`] and [`Supertable::attach`]:
//! staged rows stay out of the table until attached, land together in one
//! commit, search like appended rows, and survive writers on other
//! connections attaching to the same table at the same time.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, thread};

use infino::{
    Bm25SearchOptions, Connection, IndexSpec, StagedAppend, Supertable,
    arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
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

fn hits(table: &Supertable, term: &str) -> usize {
    table
        .bm25_search("body", term, TOP_K, Bm25SearchOptions::new(), None)
        .expect("bm25_search")
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

#[test]
fn staged_rows_join_the_table_only_when_attached() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let staged: Vec<StagedAppend> = (0..BATCHES)
        .map(|b| table.stage(&batch(b)).expect("stage"))
        .collect();
    assert!(staged.iter().all(|s| s.rows() == BATCH_ROWS as u64));
    assert_eq!(count(&db), 0, "staged rows are not in the table");
    assert_eq!(hits(&table, "common"), 0);

    table.attach(&staged).expect("attach");
    assert_eq!(count(&db), BATCHES * BATCH_ROWS);
    for b in 0..BATCHES {
        assert_eq!(
            hits(&table, &format!("batch{b}")),
            BATCH_ROWS as usize,
            "batch {b} searches like appended rows"
        );
    }
    assert_eq!(hits(&table, "common"), (BATCHES * BATCH_ROWS) as usize);

    // Reopened, the table lists the attached superfiles.
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
    table.attach(&[staged]).expect("attach");
    assert_eq!(count(&db), 2 * BATCH_ROWS);
}

#[test]
fn attaching_the_same_staged_batch_twice_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let staged = [table.stage(&batch(0)).expect("stage")];
    table.attach(&staged).expect("first attach");
    assert!(
        table.attach(&staged).is_err(),
        "its superfiles are listed already"
    );
    assert_eq!(count(&db), BATCH_ROWS);
}

#[test]
fn a_batch_that_adds_a_column_is_refused_and_appends() {
    let dir = TempDir::new().expect("tempdir");
    let db = open_db(&dir);
    let table = create(&db);

    let wider = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("extra", DataType::Utf8, true),
    ]));
    let rows = RecordBatch::try_new(
        wider,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(LargeStringArray::from(vec!["one"])),
            Arc::new(StringArray::from(vec![Some("x")])),
        ],
    )
    .expect("valid batch");
    assert!(table.stage(&rows).is_err());
    table.append(&rows).expect("append adds the column");
    assert_eq!(count(&db), 1);
}

#[test]
fn a_table_with_no_storage_refuses_staging() {
    let db = connect("memory://").expect("connect");
    let table = create(&db);
    assert!(table.stage(&batch(0)).is_err());
    assert_eq!(count(&db), 0);
}

#[test]
fn writers_on_several_connections_attach_to_one_table() {
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
                table.attach(&staged).expect("attach");
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
