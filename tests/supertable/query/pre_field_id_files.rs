// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Files written before field ids existed carry no id on any column: their
//! blobs label columns with the table's names at the time. Two questions
//! about such a file have to be answered from the creation schema rather
//! than from whatever the table has become since — what it calls a column,
//! and whether it can hold one at all.

use arrow_schema::DataType;
use infino::{
    Bm25Params, BoolMode, ColumnIndex, FieldId, FieldPatch, SchemaPatch, Stemmer, Stopwords,
    test_helpers::{copy_dir_recursive, old_format_fts_fixture, open_old_format_fts_fixture},
};
use tempfile::TempDir;

/// A term every row of the fixture carries.
const SHARED_TERM: &str = "shared";
/// The fixture's one full-text column, and its id once the table is opened.
const TITLE_ID: FieldId = FieldId(1);

/// Renaming the column leaves every existing file labelling it the old way.
/// Resolving the file's label from the table's *current* name finds nothing
/// in a file written before ids, and the error fails the whole query rather
/// than that file contributing nothing.
#[test]
fn a_renamed_column_still_searches_files_written_before_field_ids() {
    let dir = TempDir::new().expect("tempdir");
    copy_dir_recursive(&old_format_fts_fixture(), dir.path());
    let (_storage, st) = open_old_format_fts_fixture(dir.path(), |o| o);

    let before = st
        .reader()
        .expect("reader")
        .token_match("title", SHARED_TERM, BoolMode::Or)
        .expect("the fixture answers under its own name")
        .len();
    assert!(before > 0, "the fixture carries the term");

    st.apply_schema(
        &SchemaPatch::new(vec![FieldPatch::named("headline").with_id(TITLE_ID)]),
        None,
    )
    .expect("rename the column");

    let after = st
        .reader()
        .expect("reader")
        .token_match("headline", SHARED_TERM, BoolMode::Or)
        .expect("the old files answer under the new name")
        .len();
    assert_eq!(after, before, "the same rows, found by id");
}

/// A column added after the upgrade cannot be in a file written before it.
/// Taking such a file to hold every column asks it for one it has never had,
/// and the lookup fails the whole query instead of skipping the file.
#[test]
fn a_column_added_after_the_upgrade_skips_files_written_before_field_ids() {
    let dir = TempDir::new().expect("tempdir");
    copy_dir_recursive(&old_format_fts_fixture(), dir.path());
    let (_storage, st) = open_old_format_fts_fixture(dir.path(), |o| o);

    st.apply_schema(
        &SchemaPatch::new(vec![
            FieldPatch::named("body")
                .with_type(DataType::LargeUtf8)
                .with_index(ColumnIndex::Fts {
                    analyzer: "standard".into(),
                    stopwords: Stopwords::None,
                    stemmer: Stemmer::None,
                    positions: false,
                    stored: true,
                    bm25: Bm25Params::default(),
                }),
        ]),
        None,
    )
    .expect("add a column the written files cannot hold");

    let hits = st
        .reader()
        .expect("reader")
        .token_match("body", SHARED_TERM, BoolMode::Or)
        .expect("files that predate the column contribute nothing, rather than failing");
    assert!(hits.is_empty(), "no file holds the new column yet");
}
