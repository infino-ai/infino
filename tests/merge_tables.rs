// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! `merge_tables`: tables loaded on their own, merged into one.

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::{BoolMode, IndexSpec, connect};

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

fn batch(word: &str, rows: usize) -> RecordBatch {
    let titles: Vec<String> = (0..rows).map(|i| format!("{word} row {i}")).collect();
    let column: ArrayRef = Arc::new(LargeStringArray::from(titles));
    RecordBatch::try_new(schema(), vec![column]).expect("batch")
}

#[test]
fn tables_loaded_apart_merge_into_one_and_outlive_their_sources() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8")).expect("connect");
    let spec = || IndexSpec::new().fts("title");
    let target = db.create_table("t", schema(), spec()).expect("target");
    target
        .append(&batch("before", 5))
        .expect("rows already there");
    let sources: Vec<_> = ["a", "b", "c"]
        .iter()
        .map(|name| {
            let source = db.create_table(name, schema(), spec()).expect("source");
            for _ in 0..3 {
                source.append(&batch(name, 10)).expect("append");
            }
            source
        })
        .collect();
    let refs: Vec<_> = sources.iter().collect();

    let report = target.merge_tables(&refs).expect("merge");
    assert_eq!(report.rows, 90);
    assert_eq!(report.merged, 9);
    assert_eq!(
        report.spliced, 3,
        "every source's index layers were stacked"
    );
    let again = target.merge_tables(&refs).expect("merge again");
    assert_eq!((again.merged, again.copied), (0, 0), "a rerun adds nothing");

    drop(sources);
    for name in ["a", "b", "c"] {
        db.drop_table(name, true).expect("drop source");
    }
    let target = db.open_table("t").expect("reopen");
    let count = |word: &str| target.count("title", word, BoolMode::Or).expect("count");
    assert_eq!(count("row"), 95);
    assert_eq!(
        count("b"),
        30,
        "a source's rows are searchable in the target"
    );
    assert_eq!(count("before"), 5);
}

/// Two processes merging the same sources into one table, as when the job moves to another
/// worker mid-merge, add every row once.
#[test]
fn two_merges_into_one_table_at_once_add_each_row_once() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_str().expect("utf-8").to_owned();
    let db = connect(&root).expect("connect");
    let spec = || IndexSpec::new().fts("title");
    db.create_table("t", schema(), spec()).expect("target");
    for name in ["a", "b", "c", "d"] {
        let s = db.create_table(name, schema(), spec()).expect("source");
        for _ in 0..4 {
            s.append(&batch(name, 25)).expect("append");
        }
    }
    let run = || {
        let root = root.clone();
        std::thread::spawn(move || {
            let db = connect(&root).expect("connect");
            let t = db.open_table("t").expect("t");
            let sources: Vec<_> = ["a", "b", "c", "d"]
                .iter()
                .map(|n| db.open_table(n).expect("source"))
                .collect();
            let refs: Vec<_> = sources.iter().collect();
            t.merge_tables(&refs).map(|r| r.merged)
        })
    };
    let (x, y) = (run(), run());
    let (x, y) = (x.join().expect("join"), y.join().expect("join"));
    x.expect("first merge");
    y.expect("second merge");
    let t = connect(&root).expect("connect").open_table("t").expect("t");
    let n = t.count("title", "row", BoolMode::Or).expect("count");
    assert_eq!(n, 400);
}

/// A merge freezes its sources: they take no more rows and no compaction, so a rerun never
/// meets merged rows under new superfile ids and copies them again.
#[test]
fn a_merged_source_is_frozen_so_a_rerun_adds_nothing() {
    use infino::{CompactionSettings, InfinoError, OptimizeOptions};
    let dir = tempfile::tempdir().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8")).expect("connect");
    let spec = || IndexSpec::new().fts("title");
    let t = db.create_table("t", schema(), spec()).expect("target");
    let a = db.create_table("a", schema(), spec()).expect("source");
    for _ in 0..4 {
        a.append(&batch("a", 10)).expect("append");
    }
    t.merge_tables(&[&a]).expect("first merge");
    assert!(matches!(
        a.append(&batch("a", 10)),
        Err(InfinoError::Unsupported(_))
    ));
    let settings = CompactionSettings {
        min_superfiles_for_merge: 2,
        ..CompactionSettings::default()
    };
    a.optimize(&OptimizeOptions::compact(settings))
        .expect("an optimize of a frozen table does nothing");
    let again = t.merge_tables(&[&a]).expect("rerun");
    assert_eq!((again.merged, again.copied), (0, 0));
    let count =
        |table: &infino::Supertable| table.count("title", "row", BoolMode::Or).expect("count");
    assert_eq!((count(&t), count(&a)), (40, 40));
}
