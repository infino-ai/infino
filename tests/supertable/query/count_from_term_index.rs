// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A single-term count is answerable from the term index alone — it already
//! records a per-superfile document frequency — so the fan-out need not open
//! anything.
//!
//! These tests pin the conditions under which that shortcut is exact. The
//! shortcut is invisible from outside (the answer is the same either way), so
//! every test asserts the ANSWER against the fan-out's, which is what actually
//! matters: a faster count that disagrees is a bug, not an optimisation.

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use datafusion::prelude::{col, lit};
use infino::{
    BoolMode,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
};
use tempfile::TempDir;

/// Titles for the first superfile. `rust` appears in three, `go` in one.
const SEG1: &[&str] = &[
    "rust systems programming",
    "rust memory safety",
    "rust and go compared",
    "python data science",
];
/// Titles for the second superfile, so the count must sum across superfiles.
const SEG2: &[&str] = &[
    "rust web services",
    "go concurrency patterns",
    "java enterprise tooling",
];

fn schema() -> Arc<arrow_schema::Schema> {
    Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        "title",
        arrow_schema::DataType::LargeUtf8,
        false,
    )]))
}

fn batch(titles: &[&str]) -> RecordBatch {
    let col: ArrayRef = Arc::new(LargeStringArray::from(titles.to_vec()));
    RecordBatch::try_new(schema(), vec![col]).expect("batch")
}

/// A storage-backed table of two superfiles. Storage is what gives the table a
/// term index at all; a storage-less table has none and always fans out.
fn two_superfiles_on_storage(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = SupertableOptions::new(schema(), vec![FtsConfig::new("title")], vec![])
        .expect("options")
        .with_storage(storage);
    let st = Supertable::create(options).expect("create");
    let mut w = st.writer().expect("writer");
    w.append(&batch(SEG1)).expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&batch(SEG2)).expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

/// The count a full fan-out would produce, for the shortcut to be checked
/// against: the cardinality of the match set.
fn fanout_count(st: &Supertable, term: &str) -> u64 {
    st.reader()
        .expect("reader")
        .token_match("title", term, BoolMode::Or)
        .expect("token_match")
        .len() as u64
}

/// The shortcut must agree with the fan-out, summed across superfiles.
#[test]
fn a_single_term_count_matches_the_fan_out() {
    let dir = TempDir::new().expect("tempdir");
    let st = two_superfiles_on_storage(&dir);
    let reader = st.reader().expect("reader");
    for term in ["rust", "go", "python", "java"] {
        let counted = reader.count("title", term, BoolMode::Or).expect("count");
        assert_eq!(
            counted,
            fanout_count(&st, term),
            "count of {term:?} must equal the match set's cardinality"
        );
    }
    // Sanity: the fixture really does span both superfiles, so a sum is
    // being exercised rather than a single superfile's df.
    assert_eq!(
        reader.count("title", "rust", BoolMode::Or).expect("count"),
        4,
        "rust appears in three seg1 titles and one seg2 title"
    );
}

/// A term in no superfile counts zero rather than falling into an empty sum
/// that happens to look right.
#[test]
fn a_missing_term_counts_zero() {
    let dir = TempDir::new().expect("tempdir");
    let st = two_superfiles_on_storage(&dir);
    let counted = st
        .reader()
        .expect("reader")
        .count("title", "cobol", BoolMode::Or)
        .expect("count");
    assert_eq!(counted, 0);
}

/// **The condition that matters most.** The index's document frequency is
/// GROSS: it still counts a row that has since been deleted. So once a
/// superfile carries a tombstone sidecar the shortcut must step aside and let
/// the fan-out answer, or a delete would be invisible to `count`.
#[test]
fn a_delete_is_not_counted() {
    let dir = TempDir::new().expect("tempdir");
    let st = two_superfiles_on_storage(&dir);
    let before = st
        .reader()
        .expect("reader")
        .count("title", "rust", BoolMode::Or)
        .expect("count");
    assert_eq!(before, 4);

    st.delete(col("title").eq(lit("rust memory safety")))
        .expect("delete");

    let after = st
        .reader()
        .expect("reader")
        .count("title", "rust", BoolMode::Or)
        .expect("count");
    assert_eq!(
        after, 3,
        "a deleted row must leave the count, not linger in the index's gross df"
    );
    assert_eq!(
        after,
        fanout_count(&st, "rust"),
        "and still match the fan-out"
    );
}

/// Multi-term, phrase and negated queries are not sums of document
/// frequencies, so they must keep fanning out — and keep agreeing with the
/// match set they describe.
#[test]
fn shapes_that_are_not_a_sum_still_agree() {
    let dir = TempDir::new().expect("tempdir");
    let st = two_superfiles_on_storage(&dir);
    let reader = st.reader().expect("reader");

    // OR over two terms is a union, not a sum: `rust and go compared` holds
    // both, so summing their dfs would double-count it.
    let union = reader
        .count("title", "rust go", BoolMode::Or)
        .expect("count");
    assert_eq!(union, fanout_count(&st, "rust go"));
    assert!(
        union < 4 + 2,
        "a union must not be the sum of the two dfs, got {union}"
    );

    // AND and negation narrow the set; both must match their fan-out.
    for query in ["+rust +go", "rust -go"] {
        let counted = reader.count("title", query, BoolMode::Or).expect("count");
        let expected = reader
            .token_match("title", query, BoolMode::Or)
            .expect("token_match")
            .len() as u64;
        assert_eq!(counted, expected, "{query:?} must agree with its match set");
    }
}
