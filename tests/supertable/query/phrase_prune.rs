// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A lone quoted phrase prunes on all of its members, which is a *more*
//! aggressive prune than the union it replaced.
//!
//! That is the direction that loses results when it is wrong, so these tests
//! assert the hits a phrase query returns rather than the shape of the leaf it
//! builds: a superfile holding the phrase must survive the prune, and the
//! answer must be the same one a table with no pruning opportunity gives.

use std::{collections::HashSet, sync::Arc};

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use infino::{
    BoolMode,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
};
use tempfile::TempDir;

/// Holds the phrase itself, so this superfile must survive any prune.
const HAS_PHRASE: &[&str] = &["wine beer tasting", "unrelated filler text"];
/// Holds both members, but never adjacent. Surviving the prune is correct —
/// presence cannot tell adjacency — and it must contribute no hits.
const HAS_BOTH_APART: &[&str] = &["wine tasting and craft beer", "more filler"];
/// Holds one member only. A prune that requires every member may drop it, and
/// must: no document here can contain the phrase.
const HAS_ONE: &[&str] = &["wine cellars", "beer gardens are elsewhere"];

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

/// One superfile per slice, on storage so the table has a term index at all.
fn table_of(dir: &TempDir, segments: &[&[&str]]) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = SupertableOptions::new(
        schema(),
        vec![FtsConfig::new("title").positions(true)],
        vec![],
    )
    .expect("options")
    .with_storage(storage);
    let st = Supertable::create(options).expect("create");
    let mut w = st.writer().expect("writer");
    for seg in segments {
        w.append(&batch(seg)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

fn phrase_hits(st: &Supertable, query: &str) -> usize {
    st.reader()
        .expect("reader")
        .token_match("title", query, BoolMode::Or)
        .expect("token_match")
        .len()
}

/// The result a phrase query must give, whatever the prune does: only the
/// document where the members are adjacent.
#[test]
fn a_phrase_finds_its_match_across_superfiles() {
    let dir = TempDir::new().expect("tempdir");
    let st = table_of(&dir, &[HAS_PHRASE, HAS_BOTH_APART, HAS_ONE]);
    assert_eq!(
        phrase_hits(&st, "\"wine beer\""),
        1,
        "the phrase occurs once, in the first superfile"
    );
}

/// Superfile order must not matter: the same corpus with the phrase-bearing
/// superfile last gives the same answer. A prune that is order-sensitive would
/// show up here and nowhere else.
#[test]
fn a_phrase_is_found_wherever_its_superfile_sits() {
    let dir = TempDir::new().expect("tempdir");
    let st = table_of(&dir, &[HAS_ONE, HAS_BOTH_APART, HAS_PHRASE]);
    assert_eq!(phrase_hits(&st, "\"wine beer\""), 1);
}

/// Members split across superfiles, with no superfile holding both: no
/// document can contain the phrase, so the answer is zero. Requiring every
/// member may prune both superfiles away, which is sound precisely because the
/// answer is empty either way.
#[test]
fn a_phrase_whose_members_never_share_a_superfile_finds_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let st = table_of(&dir, &[&["wine cellars only"], &["beer gardens only"]]);
    assert_eq!(phrase_hits(&st, "\"wine beer\""), 0);
}

/// The soundness boundary, end to end. `wine OR "beer stout"` matches every
/// document holding `wine`, including those in a superfile with no `stout` at
/// all. Requiring the phrase's members here would drop them.
#[test]
fn a_phrase_beside_a_bare_term_keeps_the_bare_term_s_matches() {
    let dir = TempDir::new().expect("tempdir");
    // Second superfile has `wine` and no `stout` anywhere.
    let st = table_of(
        &dir,
        &[&["beer stout tasting", "wine list"], &["wine merchants"]],
    );
    let hits = st
        .reader()
        .expect("reader")
        .token_match("title", "wine \"beer stout\"", BoolMode::Or)
        .expect("token_match");
    let ids: HashSet<i128> = hits
        .iter()
        .map(|h| h.stable_id.expect("hits carry stable _id"))
        .collect();
    assert_eq!(
        ids.len(),
        3,
        "both `wine` rows and the phrase row must match, got {}",
        ids.len()
    );
}

/// A must-side phrase prunes hardest, and must still find its match.
#[test]
fn a_must_phrase_finds_its_match() {
    let dir = TempDir::new().expect("tempdir");
    let st = table_of(&dir, &[HAS_PHRASE, HAS_BOTH_APART, HAS_ONE]);
    assert_eq!(phrase_hits(&st, "+\"wine beer\""), 1);
}
