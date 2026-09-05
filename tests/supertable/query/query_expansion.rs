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
//! phrases left untouched, bloom-prune soundness for a rare member that
//! lives in one superfile only, and the ranked path — `bm25_search`
//! picks the registration up, a per-call expansion overrides it, and a
//! search with no expansion is bit-identical to one with an empty one.
//! `hybrid_search` runs that same expanded text arm: with a group
//! registered every surface form is fused in as a text hit, and clearing
//! it restores the literal token.

#![deny(clippy::unwrap_used)]

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    InfinoError,
    superfile::{
        builder::{FtsConfig, VectorConfig},
        fts::reader::{Bm25SearchOptions, BoolMode, QueryExpansion},
    },
    supertable::{Supertable, SupertableOptions, query::vector::VectorSearchOptions},
    test_helpers::{
        brute_force_bm25::{BruteForceBm25, OracleQuery},
        default_tokenizer, default_vector_config,
    },
};
use rayon::ThreadPoolBuilder;

/// A `k` covering every match on the planted corpora.
const K_ALL: usize = 64;
/// Score tolerance against the brute-force oracle: same formula, same
/// inputs (every planted doc is short enough for an exact length norm),
/// so only f64-vs-f32 idf rounding and sum order remain.
const ORACLE_SCORE_TOLERANCE: f32 = 1e-3;

/// Single-thread writer pool for deterministic builds.
const RAYON_POOL_THREADS: usize = 1;

/// Dimension of the hybrid fixture's vector column: the engine's
/// minimum, and what `default_vector_config` builds.
const HYBRID_DIM: usize = 16;
/// Random-rotation seed for the hybrid fixture's vector index.
const HYBRID_ROT_SEED: u64 = 7;
/// Rows for the hybrid test: three surface forms of `run` and three
/// unrelated titles. Row `i` carries a one-hot embedding at dim `i`.
const HYBRID_DOCS: &[(&str, &str)] = &[
    ("run tests first", "h0"),
    ("the suite runs nightly", "h1"),
    ("running late again", "h2"),
    ("login page redesign", "h3"),
    ("nothing to see here", "h4"),
    ("pass the suite", "h5"),
];
/// Tags of the [`HYBRID_DOCS`] rows holding any surface form of `run`.
const HYBRID_RUN_FORMS: &[&str] = &["h0", "h1", "h2"];
/// Tags of the [`HYBRID_DOCS`] rows holding the literal token `run`.
const HYBRID_LITERAL_RUN: &[&str] = &["h0"];
/// The unrelated row the hybrid query vector points at, so the vector
/// arm's best hit is one the text arm never surfaces.
const HYBRID_QUERY_ROW: usize = 3;

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

/// `title` (full-text indexed) + `tag`.
fn scalar_fields() -> Vec<Field> {
    vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("tag", DataType::LargeUtf8, false),
    ]
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(scalar_fields()))
}

/// The element field shared by the `emb` column type and its arrays.
fn embedding_item_field() -> Field {
    Field::new("item", DataType::Float32, true)
}

/// The scalar fields plus `emb`, a fixed-size `f32` vector column.
fn hybrid_schema() -> Arc<Schema> {
    let mut fields = scalar_fields();
    fields.push(Field::new(
        "emb",
        DataType::FixedSizeList(Arc::new(embedding_item_field()), HYBRID_DIM as i32),
        false,
    ));
    Arc::new(Schema::new(fields))
}

/// Options over `schema`: `title` full-text indexed with positions on
/// (so phrases are answerable), `vectors` indexed, and a single-thread
/// writer pool for deterministic builds.
fn options_for(schema: Arc<Schema>, vectors: Vec<VectorConfig>) -> SupertableOptions {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(
        schema,
        vec![FtsConfig::new("title").positions(true)],
        vectors,
    )
    .expect("valid options")
    .with_writer_pool(pool)
}

fn options() -> SupertableOptions {
    options_for(schema(), vec![])
}

/// The `title` and `tag` arrays for one batch of docs.
fn text_columns(docs: &[(String, String)]) -> (LargeStringArray, LargeStringArray) {
    let titles = LargeStringArray::from(docs.iter().map(|(t, _)| t.as_str()).collect::<Vec<_>>());
    let tags = LargeStringArray::from(docs.iter().map(|(_, g)| g.as_str()).collect::<Vec<_>>());
    (titles, tags)
}

/// One committed superfile per `commits` entry.
fn table(commits: &[Vec<(String, String)>]) -> Supertable {
    let st = Supertable::create(options()).expect("create");
    let mut w = st.writer().expect("writer");
    for docs in commits {
        let (titles, tags) = text_columns(docs);
        let batch =
            RecordBatch::try_new(schema(), vec![Arc::new(titles), Arc::new(tags)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    st
}

/// A unit vector along dim `active`.
fn one_hot(active: usize) -> Vec<f32> {
    (0..HYBRID_DIM)
        .map(|d| if d == active { 1.0 } else { 0.0 })
        .collect()
}

/// `n` rows of embeddings, row `i` one-hot at dim `i`.
fn one_hot_embeddings(n: usize) -> FixedSizeListArray {
    assert!(n <= HYBRID_DIM, "one dim per row");
    let flat: Vec<f32> = (0..n).flat_map(one_hot).collect();
    FixedSizeListArray::try_new(
        Arc::new(embedding_item_field()),
        HYBRID_DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("embeddings")
}

/// One superfile holding [`HYBRID_DOCS`] with its one-hot embeddings.
fn hybrid_table() -> Supertable {
    let st = Supertable::create(options_for(
        hybrid_schema(),
        vec![default_vector_config("emb", HYBRID_ROT_SEED)],
    ))
    .expect("create");
    let mut w = st.writer().expect("writer");
    let (titles, tags) = text_columns(&owned(HYBRID_DOCS));
    let batch = RecordBatch::try_new(
        hybrid_schema(),
        vec![
            Arc::new(titles),
            Arc::new(tags),
            Arc::new(one_hot_embeddings(HYBRID_DOCS.len())),
        ],
    )
    .expect("batch");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
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

/// Column 0 of a batch projected with `tag` first.
fn tag_column(b: &RecordBatch) -> &LargeStringArray {
    b.column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("tag is LargeUtf8")
}

/// The `tag`s of batches projected as `["tag"]`, as a set.
fn tag_set(batches: &[RecordBatch]) -> HashSet<String> {
    let mut out = HashSet::new();
    for b in batches {
        let col = tag_column(b);
        for i in 0..col.len() {
            out.insert(col.value(i).to_string());
        }
    }
    out
}

/// `token_match` hits as the set of their `tag`s.
fn tags(st: &Supertable, query: &str, mode: BoolMode) -> HashSet<String> {
    tag_set(
        &st.token_match("title", query, mode, Some(&["tag"]))
            .expect("token_match"),
    )
}

fn count(st: &Supertable, query: &str, mode: BoolMode) -> u64 {
    st.count("title", query, mode).expect("count")
}

/// `tag → score` from batches projected as `["tag", "score"]`.
fn tag_scores(batches: &[RecordBatch]) -> HashMap<String, f32> {
    let mut out = HashMap::new();
    for b in batches {
        let tag = tag_column(b);
        let score = b
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score is f32");
        for i in 0..b.num_rows() {
            out.insert(tag.value(i).to_string(), score.value(i));
        }
    }
    out
}

/// Ranked hits as `tag → score`, through the options-taking entry.
fn scored(st: &Supertable, query: &str, opts: &Bm25SearchOptions) -> HashMap<String, f32> {
    tag_scores(
        &st.bm25_search_with_options("title", query, K_ALL, opts, Some(&["tag", "score"]))
            .expect("bm25_search"),
    )
}

/// Assert the fused head is exactly `want`: the `want.len()` best fused
/// scores belong to `want`, and the last of them sits strictly above the
/// best remaining row. Under reciprocal-rank fusion a row both arms
/// surfaced scores the sum of two reciprocal ranks, so on a table this
/// small the head is precisely the rows the text arm surfaced.
fn assert_fused_head(scores: &HashMap<String, f32>, want: &[&str], what: &str) {
    let mut ranked: Vec<(&str, f32)> = scores.iter().map(|(t, s)| (t.as_str(), *s)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    assert!(
        ranked.len() > want.len(),
        "{what}: expected vector-only rows below the head: {ranked:?}"
    );
    let (head, tail) = ranked.split_at(want.len());
    assert_eq!(
        head.iter().map(|(t, _)| *t).collect::<HashSet<_>>(),
        want.iter().copied().collect::<HashSet<_>>(),
        "{what}: fused head"
    );
    let head_floor = head.last().map_or(f32::INFINITY, |(_, s)| *s);
    let tail_ceiling = tail.first().map_or(f32::NEG_INFINITY, |(_, s)| *s);
    assert!(
        head_floor > tail_ceiling,
        "{what}: the head must sit strictly above the vector-only rows: {ranked:?}"
    );
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

/// The ranked path over heads: on corpus A with the vocabulary,
/// `bm25_search` matches exactly the documents the stemmed corpus B
/// matches without it, and its scores are the group oracle's — Σ member
/// tf under the commonest member's idf. (Scores are *not* those of a
/// stemmed index: a stem's df there is the union of its forms' docs,
/// while a group deliberately takes the commonest form's df, so the
/// score claim is against the oracle, the match-set claim against B.)
#[test]
fn bm25_over_heads_matches_the_stemmed_index_and_scores_like_the_group_oracle() {
    let a = table(&[owned(SURFACE)]);
    a.set_query_expansion("title", Some(vocabulary()))
        .expect("register on A");
    let b = table(&[stemmed(SURFACE)]);
    for query in HEAD_QUERIES {
        for mode in [BoolMode::Or, BoolMode::And] {
            let opts = Bm25SearchOptions::new().with_mode(mode);
            let got: HashSet<String> = scored(&a, query, &opts).into_keys().collect();
            let want: HashSet<String> = scored(&b, query, &opts).into_keys().collect();
            assert_eq!(got, want, "match set for {query:?} in {mode:?}");
        }
    }

    // Scores against the brute-force oracle with group semantics. Row
    // `i` of the corpus is tag `t{i}`, so oracle doc ids map to tags.
    let corp: Vec<(u64, &str)> = SURFACE
        .iter()
        .enumerate()
        .map(|(i, (title, _))| (i as u64, *title))
        .collect();
    let oracle = BruteForceBm25::index(&corp, default_tokenizer().as_ref());
    let run = || {
        vec![
            "run".to_string(),
            "runs".into(),
            "running".into(),
            "ran".into(),
        ]
    };
    let fail = || {
        vec![
            "fail".to_string(),
            "fails".into(),
            "failing".into(),
            "failed".into(),
        ]
    };
    let cases: Vec<(&str, BoolMode, OracleQuery)> = vec![
        (
            "run",
            BoolMode::Or,
            OracleQuery {
                should_groups: vec![run()],
                ..OracleQuery::default()
            },
        ),
        (
            "run fail",
            BoolMode::Or,
            OracleQuery {
                should_groups: vec![run(), fail()],
                ..OracleQuery::default()
            },
        ),
        (
            "run fail",
            BoolMode::And,
            OracleQuery {
                must_groups: vec![run(), fail()],
                ..OracleQuery::default()
            },
        ),
        (
            "+run fail",
            BoolMode::Or,
            OracleQuery {
                must_groups: vec![run()],
                should_groups: vec![fail()],
                ..OracleQuery::default()
            },
        ),
        (
            "run -fail",
            BoolMode::Or,
            OracleQuery {
                should_groups: vec![run()],
                negative_groups: vec![fail()],
                ..OracleQuery::default()
            },
        ),
        (
            "login run",
            BoolMode::And,
            OracleQuery {
                musts: vec!["login".to_string()],
                must_groups: vec![run()],
                ..OracleQuery::default()
            },
        ),
    ];
    for (query, mode, oq) in cases {
        let got = scored(&a, query, &Bm25SearchOptions::new().with_mode(mode));
        let want: HashMap<String, f32> = oracle
            .top_k_expanded(&oq, K_ALL)
            .into_iter()
            .map(|(id, s)| (format!("t{id}"), s))
            .collect();
        assert_eq!(
            got.keys().collect::<HashSet<_>>(),
            want.keys().collect::<HashSet<_>>(),
            "oracle match set for {query:?} in {mode:?}"
        );
        for (tag, w) in &want {
            let g = got[tag];
            assert!(
                (g - w).abs() < ORACLE_SCORE_TOLERANCE,
                "{query:?} in {mode:?}: doc {tag} scored {g}, the oracle {w}"
            );
        }
    }
}

#[test]
fn bm25_search_applies_the_registration_and_a_per_call_expansion_overrides_it() {
    let st = table(&[owned(SURFACE)]);
    let default_opts = Bm25SearchOptions::new();
    // Nothing registered: the literal token only.
    assert_eq!(
        scored(&st, "run", &default_opts)
            .keys()
            .collect::<HashSet<_>>(),
        HashSet::from([&"t3".to_string()])
    );
    // Registered: every surface form.
    st.set_query_expansion("title", Some(vocabulary()))
        .expect("register");
    assert_eq!(scored(&st, "run", &default_opts).len(), 5);
    // A per-call vocabulary replaces the registration for that call: a
    // narrower family finds fewer docs, ...
    let narrow = Arc::new(QueryExpansion::new().group("run", ["runs"]));
    let narrowed = scored(
        &st,
        "run",
        &Bm25SearchOptions::new().with_expansion(Some(narrow)),
    );
    assert_eq!(
        narrowed.keys().cloned().collect::<HashSet<_>>(),
        HashSet::from(["t2".to_string(), "t3".into(), "t6".into()])
    );
    // ... and an empty one runs the literal search.
    let literal = scored(
        &st,
        "run",
        &Bm25SearchOptions::new().with_expansion(Some(Arc::new(QueryExpansion::new()))),
    );
    assert_eq!(
        literal.keys().cloned().collect::<HashSet<_>>(),
        HashSet::from(["t3".to_string()])
    );
    // The registration is untouched by per-call overrides.
    assert_eq!(scored(&st, "run", &default_opts).len(), 5);
    // A per-call entry the analyzer refuses is a query error, not a
    // silent literal search.
    let bad = Arc::new(QueryExpansion::new().stop(["login page"]));
    let err = st
        .bm25_search_with_options(
            "title",
            "run",
            K_ALL,
            &Bm25SearchOptions::new().with_expansion(Some(bad)),
            None,
        )
        .expect_err("two-word entry");
    assert!(matches!(err, InfinoError::Query(_)), "got {err:?}");
}

#[test]
fn no_expansion_and_an_empty_expansion_return_identical_batches() {
    let st = table(&[owned(SURFACE)]);
    for (query, mode) in [
        ("run login", BoolMode::Or),
        ("+run -login the", BoolMode::And),
        ("\"login page\" fails", BoolMode::Or),
    ] {
        let none = st
            .bm25_search_with_options(
                "title",
                query,
                K_ALL,
                &Bm25SearchOptions::new().with_mode(mode),
                Some(&["tag", "score"]),
            )
            .expect("bm25 without expansion");
        let empty = st
            .bm25_search_with_options(
                "title",
                query,
                K_ALL,
                &Bm25SearchOptions::new()
                    .with_mode(mode)
                    .with_expansion(Some(Arc::new(QueryExpansion::new()))),
                Some(&["tag", "score"]),
            )
            .expect("bm25 with an empty expansion");
        assert_eq!(none, empty, "{query:?} in {mode:?}");
    }
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

/// `hybrid_search` runs the same expanded text arm as `bm25_search`:
/// with the group registered every surface form of `run` is a text hit,
/// and after clearing only the literal token is. The vector arm is held
/// fixed — `k` is the row count, so it surfaces every row and fusion is
/// a union that can only reorder — and the query vector points at a row
/// the text arm never matches, so the head of the fused ranking is
/// exactly the rows the text arm surfaced.
#[test]
fn hybrid_search_applies_the_registration_and_clearing_restores_literal_matching() {
    let st = hybrid_table();
    let k = HYBRID_DOCS.len();
    let query_vector = one_hot(HYBRID_QUERY_ROW);
    let every_tag: HashSet<String> = HYBRID_DOCS.iter().map(|(_, g)| g.to_string()).collect();
    // The fixture's premise: at `k` = the row count the vector arm alone
    // reaches every row.
    let reached = tag_set(
        &st.vector_search(
            "emb",
            &query_vector,
            k,
            VectorSearchOptions::new(),
            None,
            Some(&["tag"]),
        )
        .expect("vector_search"),
    );
    assert_eq!(reached, every_tag, "the vector arm must surface every row");

    let fused = |st: &Supertable| {
        tag_scores(
            &st.hybrid_search(
                "title",
                "run",
                BoolMode::Or,
                "emb",
                &query_vector,
                VectorSearchOptions::new(),
                k,
                Some(&["tag", "score"]),
            )
            .expect("hybrid_search"),
        )
    };

    st.set_query_expansion("title", Some(vocabulary()))
        .expect("register");
    let with_group = fused(&st);
    assert_eq!(
        with_group.keys().cloned().collect::<HashSet<_>>(),
        every_tag,
        "registered: fusion is a union, so every row is returned"
    );
    assert_fused_head(&with_group, HYBRID_RUN_FORMS, "registered");

    st.set_query_expansion("title", None).expect("clear");
    let literal = fused(&st);
    assert_eq!(
        literal.keys().cloned().collect::<HashSet<_>>(),
        every_tag,
        "cleared: fusion is a union, so every row is returned"
    );
    assert_fused_head(&literal, HYBRID_LITERAL_RUN, "cleared");
}
