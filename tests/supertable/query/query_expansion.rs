// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-time term expansion (stop terms + term groups) through the
//! table layer's unranked surface, registered per column on the handle.
//!
//! The central claim is the **stemmed-index equivalence**: corpus A keeps
//! its surface forms and carries the expansion; corpus B has every
//! planted family member rewritten to its head and carries none. For
//! every query over heads, `token_match` on A must return the same
//! documents as `token_match` on B, in both boolean modes, with and
//! without clause sigils — and `count` must agree with `token_match`.
//! Docs are identified by a per-row `tag` column (the two tables are
//! built independently, so `_id`s differ; the tag is content).
//!
//! Alongside: the all-stop fallback, sigils surviving stop removal,
//! phrases left untouched, and bloom-prune soundness for a rare member
//! that lives in one superfile only.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    InfinoError,
    superfile::{
        builder::FtsConfig,
        fts::reader::{BoolMode, QueryExpansion},
    },
    supertable::{Supertable, SupertableOptions},
};
use rayon::ThreadPoolBuilder;

/// Single-thread writer pool for deterministic builds.
const RAYON_POOL_THREADS: usize = 1;

/// The planted families: every member rewrites to its head in corpus B.
const FAMILIES: &[(&str, &[&str])] = &[
    ("run", &["runs", "running", "ran"]),
    ("fail", &["fails", "failing", "failed"]),
];

/// Stop terms shared by both vocabularies.
const STOP: &[&str] = &["the", "and", "of"];

/// Corpus A: `(title with surface forms, tag)`. Tags are unique so hits
/// compare by content across independently built tables.
const SURFACE: &[(&str, &str)] = &[
    ("running fails on the login page", "t0"),
    ("the job ran and failed", "t1"),
    ("runs of the suite pass", "t2"),
    ("run failing tests first", "t3"),
    ("login page redesign", "t4"),
    ("nothing to see here", "t5"),
    ("runner runs runs", "t6"),
    ("fail fast fail often", "t7"),
    ("the who", "t8"),
];

/// `title` (FTS, positions on so phrases are answerable) + `tag`.
fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("tag", DataType::LargeUtf8, false),
    ]))
}

fn options() -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(
        schema(),
        vec![FtsConfig::new("title").positions(true)],
        vec![],
    )
    .expect("valid options")
    .with_writer_pool(pool)
}

/// One committed superfile per `commits` entry.
fn table(commits: &[Vec<(String, String)>]) -> Supertable {
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    for docs in commits {
        let titles =
            LargeStringArray::from(docs.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>());
        let tags = LargeStringArray::from(docs.iter().map(|(_, g)| g.as_str()).collect::<Vec<_>>());
        let batch =
            RecordBatch::try_new(schema(), vec![Arc::new(titles), Arc::new(tags)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    st
}

fn owned(docs: &[(&str, &str)]) -> Vec<(String, String)> {
    docs.iter()
        .map(|(t, g)| (t.to_string(), g.to_string()))
        .collect()
}

/// Corpus B: every family member replaced by its head — the text a
/// stemmer would have indexed.
fn stemmed(docs: &[(&str, &str)]) -> Vec<(String, String)> {
    docs.iter()
        .map(|(title, tag)| {
            let rewritten: Vec<&str> = title
                .split_whitespace()
                .map(|w| {
                    FAMILIES
                        .iter()
                        .find(|(_, members)| members.contains(&w))
                        .map_or(w, |(head, _)| head)
                })
                .collect();
            (rewritten.join(" "), tag.to_string())
        })
        .collect()
}

fn vocabulary() -> Arc<QueryExpansion> {
    let mut exp = QueryExpansion::new().stop(STOP.iter().copied());
    for (head, members) in FAMILIES {
        exp = exp.group(*head, members.iter().copied());
    }
    Arc::new(exp)
}

/// The stop-only half of the vocabulary, for corpus B: a stemmed index
/// that also drops stop words.
fn stop_only() -> Arc<QueryExpansion> {
    Arc::new(QueryExpansion::new().stop(STOP.iter().copied()))
}

/// `token_match` hits as the set of their `tag`s.
fn tags(st: &Supertable, query: &str, mode: BoolMode) -> HashSet<String> {
    let batches = st
        .token_match("title", query, mode, Some(&["tag"]))
        .expect("token_match");
    let mut out = HashSet::new();
    for b in &batches {
        let col = b
            .column(0)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("tag is LargeUtf8");
        for i in 0..col.len() {
            out.insert(col.value(i).to_string());
        }
    }
    out
}

fn count(st: &Supertable, query: &str, mode: BoolMode) -> u64 {
    st.count("title", query, mode).expect("count")
}

/// Queries over heads that carry no stop word.
const HEAD_QUERIES: &[&str] = &[
    "run",
    "fail",
    "run fail",
    "+run fail",
    "run -fail",
    "+run -fail",
    "login run",
    "+login -run",
    "runner run",
    "\"login page\" run",
    "+\"login page\" -run",
    "run pass",
];

/// Queries that also carry stop words.
const STOP_QUERIES: &[&str] = &[
    "the run",
    "run and fail",
    "+the run",
    "the run -fail",
    "\"the who\" run",
    "of run of fail",
];

#[test]
fn token_match_and_count_match_a_stemmed_index_over_heads() {
    // A: surface forms + full vocabulary, split over three superfiles so
    // the fan-out and the bloom prune are both in play.
    let surface = owned(SURFACE);
    let a = table(&[
        surface[..3].to_vec(),
        surface[3..6].to_vec(),
        surface[6..].to_vec(),
    ]);
    a.set_query_expansion("title", Some(vocabulary()))
        .expect("register on A");
    // B: heads only, no expansion — the stemmed index.
    let b = table(&[stemmed(SURFACE)]);

    for query in HEAD_QUERIES {
        for mode in [BoolMode::Or, BoolMode::And] {
            let got = tags(&a, query, mode);
            let want = tags(&b, query, mode);
            assert_eq!(got, want, "token_match for {query:?} in {mode:?}");
            assert_eq!(
                count(&a, query, mode),
                want.len() as u64,
                "count for {query:?} in {mode:?}"
            );
        }
    }
    // The families genuinely matter: without the expansion `run` alone
    // misses the inflected docs.
    a.set_query_expansion("title", None).expect("clear");
    let literal = tags(&a, "run", BoolMode::Or);
    assert_eq!(
        literal,
        HashSet::from(["t3".to_string()]),
        "the literal token `run` is in one doc"
    );
}

#[test]
fn stop_terms_match_a_stop_filtered_stemmed_index() {
    let a = table(&[owned(SURFACE)]);
    a.set_query_expansion("title", Some(vocabulary()))
        .expect("register on A");
    // B carries only the stop terms: a stemmed index whose analyzer also
    // dropped stop words.
    let b = table(&[stemmed(SURFACE)]);
    b.set_query_expansion("title", Some(stop_only()))
        .expect("register on B");

    for query in STOP_QUERIES {
        for mode in [BoolMode::Or, BoolMode::And] {
            let got = tags(&a, query, mode);
            let want = tags(&b, query, mode);
            assert_eq!(got, want, "token_match for {query:?} in {mode:?}");
            assert_eq!(
                count(&a, query, mode),
                want.len() as u64,
                "count for {query:?}"
            );
        }
    }
}

#[test]
fn stop_removal_spares_sigils_and_phrases() {
    let a = table(&[owned(SURFACE)]);
    a.set_query_expansion("title", Some(vocabulary()))
        .expect("register");

    // A bare stop term is dropped: `the run` matches exactly what `run`
    // matches (every family member).
    assert_eq!(
        tags(&a, "the run", BoolMode::And),
        tags(&a, "run", BoolMode::And),
        "a bare stop term contributes nothing"
    );
    // `+the` keeps its sigil and is a real must: only docs holding `the`.
    let must_the = tags(&a, "+the run", BoolMode::Or);
    assert_eq!(
        must_the,
        HashSet::from(["t0".to_string(), "t1".into(), "t2".into(), "t8".into()]),
        "an explicit +stop term is a must"
    );
    // `-the` keeps its sigil and excludes.
    assert_eq!(
        tags(&a, "run -the", BoolMode::Or),
        HashSet::from(["t3".to_string(), "t6".into()]),
        "an explicit -stop term excludes"
    );
    // Phrase words are never removed: `"the who"` still needs both words
    // adjacent, and finds the band.
    assert_eq!(
        tags(&a, "\"the who\"", BoolMode::Or),
        HashSet::from(["t8".to_string()])
    );
}

#[test]
fn an_all_stop_query_keeps_its_bare_tokens() {
    let a = table(&[owned(SURFACE)]);
    let before = tags(&a, "the who", BoolMode::And);
    assert_eq!(before, HashSet::from(["t8".to_string()]));
    a.set_query_expansion(
        "title",
        Some(Arc::new(QueryExpansion::new().stop(["the", "who"]))),
    )
    .expect("register");
    // Every bare token is a stop term; dropping them all would make the
    // query empty, so the tokens are kept and the band is still found.
    assert_eq!(tags(&a, "the who", BoolMode::And), before);
    assert_eq!(count(&a, "the who", BoolMode::And), 1);
    // With a must present the stop terms do go: `+login the who` is just
    // `+login`.
    assert_eq!(
        tags(&a, "+login the who", BoolMode::Or),
        tags(&a, "+login", BoolMode::Or)
    );
}

#[test]
fn a_rare_member_held_by_one_superfile_is_never_pruned() {
    // `ran` (a member, never a head) lives only in the third superfile,
    // whose other docs hold no `run` form at all. A must group pruned on
    // its head — or on all members conjunctively — would drop that
    // superfile and lose the doc; the leaf must keep a superfile holding
    // any member.
    let commits = vec![
        owned(&[("run tests first", "s0"), ("login page", "s1")]),
        owned(&[("runs of tests", "s2"), ("nothing here", "s3")]),
        owned(&[("the job ran", "s4"), ("unrelated text", "s5")]),
    ];
    let st = table(&commits);
    st.set_query_expansion("title", Some(vocabulary()))
        .expect("register");
    for mode in [BoolMode::And, BoolMode::Or] {
        assert_eq!(
            tags(&st, "+run", mode),
            HashSet::from(["s0".to_string(), "s2".into(), "s4".into()]),
            "+run must reach the superfile holding only `ran` ({mode:?})"
        );
        assert_eq!(count(&st, "+run", mode), 3);
    }
    // Conjunction across groups and terms still prunes correctly: `+run
    // +tests` reaches s0 and s2 (both superfiles hold a member and `tests`).
    assert_eq!(
        tags(&st, "+run +tests", BoolMode::Or),
        HashSet::from(["s0".to_string(), "s2".into()])
    );
    // And a group nobody holds matches nothing, prune or not.
    st.set_query_expansion(
        "title",
        Some(Arc::new(QueryExpansion::new().group("zzz", ["yyy", "xxx"]))),
    )
    .expect("register");
    assert!(tags(&st, "zzz", BoolMode::Or).is_empty());
    assert_eq!(count(&st, "+zzz", BoolMode::And), 0);
}

#[test]
fn registration_is_validated_and_clearing_restores_literal_matching() {
    let st = table(&[owned(SURFACE)]);
    let err = st
        .set_query_expansion("tag", Some(vocabulary()))
        .expect_err("`tag` has no full-text index");
    assert!(matches!(err, InfinoError::Config(_)), "got {err:?}");
    let err = st
        .set_query_expansion(
            "title",
            Some(Arc::new(QueryExpansion::new().stop(["login page"]))),
        )
        .expect_err("two-word entry");
    assert!(matches!(err, InfinoError::Config(_)), "got {err:?}");

    st.set_query_expansion("title", Some(vocabulary()))
        .expect("register");
    assert_eq!(tags(&st, "run", BoolMode::Or).len(), 5);
    st.set_query_expansion("title", None).expect("clear");
    assert_eq!(
        tags(&st, "run", BoolMode::Or),
        HashSet::from(["t3".to_string()]),
        "cleared: the literal token only"
    );
}
