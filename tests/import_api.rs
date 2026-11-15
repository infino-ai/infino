// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! An import driven through the public API alone.

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use infino::{BoolMode, ImportState, IndexSpec, Writers, connect};

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
fn two_nodes_import_through_the_public_api() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8")).expect("connect");
    let t = db
        .create_table("t", schema(), IndexSpec::new().fts("title"))
        .expect("table");
    assert_eq!(
        t.begin_import("nightly", Some(Writers::Auto)).expect("begin"),
        ImportState::Open
    );
    let (a, b) = (
        t.open_import_replica("nightly", "n1").expect("a"),
        t.open_import_replica("nightly", "n2").expect("b"),
    );
    assert!(a.append_batch(&batch("alpha", 10), "1").expect("append"));
    assert!(b.append_batch(&batch("beta", 10), "2").expect("append"));
    assert!(
        b.append_batch(&batch("alpha", 10), "1").expect("a resend"),
        "the replica takes it; the graft keeps one copy"
    );

    let coordinator = t.import_coordinator().expect("coordinator");
    coordinator.graft("nightly").expect("graft");
    assert_eq!(
        t.count("title", "row", BoolMode::Or).expect("count"),
        20,
        "queryable during the import, each batch once"
    );
    assert_eq!(coordinator.open_imports().expect("open"), vec!["nightly"]);
    let status = t.import_status("nightly").expect("status");
    assert_eq!(
        (status.state, status.writers),
        (ImportState::Open, Writers::Auto)
    );
    assert!(status.last_activity.is_some());

    a.seal().expect("seal");
    b.seal().expect("seal");
    let outcome = t.finish_import("nightly", Some(20)).expect("finish");
    assert_eq!((outcome.rows, outcome.duplicate_rows), (20, 10));
    assert!(coordinator.claim_report("nightly").expect("first claim"));
    assert!(!coordinator.claim_report("nightly").expect("second claim"));
    assert!(coordinator.open_imports().expect("open").is_empty());
}
