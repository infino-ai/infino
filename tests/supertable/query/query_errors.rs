// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Error- and edge-path coverage for the SQL search TVFs exposed by
//! [`Supertable::query_sql`](infino::supertable::Supertable).
//!
//! The behavioural `query/*` files drive each TVF with *valid* input
//! and assert ranking / fusion contracts. This file deliberately walks
//! the *rejection* surface instead: each `TableFunctionImpl::call`
//! validates argument count and argument type before building its
//! `TableProvider`, and a wrong-arity / wrong-type / unknown-column
//! invocation must surface as a `query_sql` error rather than a panic.
//! A few valid-but-unusual queries (`k = 0`, an empty-result token,
//! `SELECT _id` vs `SELECT *` vs `SELECT <scalar>, score`) round out the
//! alternate execute / projection branches, and a couple of malformed
//! base-table statements pin the plain SQL-error path.
//!
//! Everything here asserts behaviour through the published
//! `query_sql` surface — no internal types are touched.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config},
};

/// `default_vector_config` is dim=16, cosine, n_cent=4.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 13;
/// Docs per commit; two commits keep the corpus multi-superfile.
const DOCS_PER_COMMIT: usize = 8;
/// Top-k used by the valid-projection walks. Larger than the corpus so
/// no query truncates a node's work.
const SURFACE_TOP_K: usize = 32;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Schema `[title (FTS), rating (scalar), emb (vector)]`. The vector
/// column is stripped from the SQL scalar schema at commit, so SQL only
/// ever sees `_id`, `title`, `rating`.
fn options_title_rating_emb() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

/// Doc `i` (within the batch) gets `titles[i]`, rating `base + i`, and a
/// one-hot embedding at global dim `(base + i) % DIM`.
fn build_batch(titles: &[&str], base: usize, schema: Arc<Schema>) -> RecordBatch {
    let n = titles.len();
    let title_arr = LargeStringArray::from(titles.to_vec());
    let ratings: Vec<i64> = (0..n).map(|i| (base + i) as i64).collect();
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        let active = (base + i) % DIM;
        for d in 0..DIM {
            flat.push(if d == active { 1.0 } else { 0.0 });
        }
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(title_arr),
            Arc::new(Int64Array::from(ratings)),
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Two-superfile corpus (docs 0-7, then 8-15). `rust` recurs across both
/// superfiles; doc 0's title is unique on `async`.
fn demo_table() -> Supertable {
    let st = Supertable::create(options_title_rating_emb()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(
        &[
            "rust async",
            "python data",
            "java spring",
            "go rust",
            "ruby rails",
            "scala akka",
            "kotlin flow",
            "rust systems",
        ],
        0,
        schema.clone(),
    ))
    .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&build_batch(
        &[
            "swift ui",
            "rust web",
            "elixir otp",
            "haskell lazy",
            "rust embedded",
            "perl regex",
            "lua script",
            "rust async runtime",
        ],
        DOCS_PER_COMMIT,
        schema,
    ))
    .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

/// One-hot query vector at `active`, as the CSV literal the vector TVFs
/// parse.
fn csv_one_hot(active: usize) -> String {
    (0..DIM)
        .map(|d| if d == active { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Assert every query in `queries` is rejected by `query_sql` (returns
/// `Err`, never panics). `label` names the validator family for the
/// failure message.
fn assert_all_error(st: &Supertable, label: &str, queries: &[String]) {
    for q in queries {
        let result = st.reader().query_sql(q);
        assert!(
            result.is_err(),
            "[{label}] expected an error, query unexpectedly succeeded: {q}"
        );
    }
}

// ---------------------------------------------------------------------
// bm25_search — arity + type validators in fts_exec.rs::Bm25SearchFunc.
// ---------------------------------------------------------------------

/// Wrong argument count for `bm25_search` (it accepts 3 or 4 args).
#[test]
fn bm25_search_wrong_arity_errors() {
    let st = demo_table();
    assert_all_error(
        &st,
        "bm25_search arity",
        &[
            // Too few.
            "SELECT _id FROM bm25_search('title')".to_string(),
            "SELECT _id FROM bm25_search('title', 'rust')".to_string(),
            // Too many (one past the 4-arg `mode` form).
            "SELECT _id FROM bm25_search('title', 'rust', 8, 'and', 'extra')".to_string(),
        ],
    );
}

/// Wrong argument *types* for `bm25_search`: a bare number where a
/// column-name string is expected, a string where `k` is expected, and
/// an unknown boolean mode.
#[test]
fn bm25_search_wrong_arg_types_error() {
    let st = demo_table();
    assert_all_error(
        &st,
        "bm25_search types",
        &[
            // Column position must be a string literal, not an int.
            "SELECT _id FROM bm25_search(42, 'rust', 8)".to_string(),
            // `k` must be an integer literal, not a string.
            "SELECT _id FROM bm25_search('title', 'rust', 'eight')".to_string(),
            // mode must be 'or' / 'and'.
            "SELECT _id FROM bm25_search('title', 'rust', 8, 'maybe')".to_string(),
        ],
    );
}

/// Querying a column that does not exist must error rather than match.
#[test]
fn bm25_search_unknown_column_errors() {
    let st = demo_table();
    assert_all_error(
        &st,
        "bm25_search unknown column",
        &["SELECT _id FROM bm25_search('nonexistent', 'rust', 8)".to_string()],
    );
}

// ---------------------------------------------------------------------
// bm25_search_prefix — arity + type validators in Bm25PrefixFunc.
// ---------------------------------------------------------------------

/// Wrong argument count for `bm25_search_prefix` (exactly 3 args).
#[test]
fn bm25_search_prefix_wrong_arity_errors() {
    let st = demo_table();
    assert_all_error(
        &st,
        "bm25_search_prefix arity",
        &[
            "SELECT _id FROM bm25_search_prefix('title', 'rus')".to_string(),
            "SELECT _id FROM bm25_search_prefix('title', 'rus', 8, 'and')".to_string(),
        ],
    );
}

/// Wrong argument types for `bm25_search_prefix`.
#[test]
fn bm25_search_prefix_wrong_arg_types_error() {
    let st = demo_table();
    assert_all_error(
        &st,
        "bm25_search_prefix types",
        &[
            // Prefix must be a string literal, not an int.
            "SELECT _id FROM bm25_search_prefix('title', 7, 8)".to_string(),
            // `k` must be an integer literal, not a string.
            "SELECT _id FROM bm25_search_prefix('title', 'rus', 'big')".to_string(),
        ],
    );
}

// ---------------------------------------------------------------------
// vector_search — arity + type / vector-parse validators in vector_exec.rs.
// ---------------------------------------------------------------------

/// Wrong argument count for `vector_search` (exactly 3 args).
#[test]
fn vector_search_wrong_arity_errors() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    assert_all_error(
        &st,
        "vector_search arity",
        &[
            format!("SELECT _id FROM vector_search('emb', '{qv}')"),
            format!("SELECT _id FROM vector_search('emb', '{qv}', 8, 'extra')"),
        ],
    );
}

/// Wrong argument types / unparseable vector literals for
/// `vector_search`: a bare int column, a non-numeric CSV element, an
/// empty vector string, and a non-integer `k`.
#[test]
fn vector_search_wrong_arg_types_error() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    assert_all_error(
        &st,
        "vector_search types",
        &[
            // Column must be a string literal.
            format!("SELECT _id FROM vector_search(1, '{qv}', 8)"),
            // CSV vector with a non-numeric element.
            "SELECT _id FROM vector_search('emb', '1,two,3', 8)".to_string(),
            // Empty CSV vector.
            "SELECT _id FROM vector_search('emb', '', 8)".to_string(),
            // `k` must be an integer literal, not a string.
            format!("SELECT _id FROM vector_search('emb', '{qv}', 'k')"),
        ],
    );
}

/// Querying a non-existent vector column must error.
#[test]
fn vector_search_unknown_column_errors() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    assert_all_error(
        &st,
        "vector_search unknown column",
        &[format!(
            "SELECT _id FROM vector_search('missing', '{qv}', 8)"
        )],
    );
}

// ---------------------------------------------------------------------
// token_match / exact_match — validators in match_exec.rs.
// ---------------------------------------------------------------------

/// Wrong argument count for `token_match` (2 or 3 args) and
/// `exact_match` (exactly 2 args).
#[test]
fn match_tvfs_wrong_arity_error() {
    let st = demo_table();
    assert_all_error(
        &st,
        "match arity",
        &[
            // token_match: too few / too many.
            "SELECT _id FROM token_match('title')".to_string(),
            "SELECT _id FROM token_match('title', 'rust', 'and', 'extra')".to_string(),
            // exact_match: too few / too many.
            "SELECT _id FROM exact_match('title')".to_string(),
            "SELECT _id FROM exact_match('title', 'rust async', 'extra')".to_string(),
        ],
    );
}

/// Wrong argument types for the match TVFs: an int column / query, and
/// an unknown boolean mode for `token_match`.
#[test]
fn match_tvfs_wrong_arg_types_error() {
    let st = demo_table();
    assert_all_error(
        &st,
        "match types",
        &[
            // token_match column must be a string literal.
            "SELECT _id FROM token_match(5, 'rust')".to_string(),
            // token_match mode must be 'or' / 'and'.
            "SELECT _id FROM token_match('title', 'rust', 'nope')".to_string(),
            // exact_match value must be a string literal.
            "SELECT _id FROM exact_match('title', 99)".to_string(),
        ],
    );
}

// ---------------------------------------------------------------------
// hybrid_search — arity + type validators in hybrid_exec.rs.
// ---------------------------------------------------------------------

/// Wrong argument count for `hybrid_search` (exactly 5 args).
#[test]
fn hybrid_search_wrong_arity_errors() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    assert_all_error(
        &st,
        "hybrid_search arity",
        &[
            format!("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}')"),
            format!("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', 8, 'extra')"),
        ],
    );
}

/// Wrong argument types for `hybrid_search`: int text column, bad vector
/// literal, and a non-integer `k`.
#[test]
fn hybrid_search_wrong_arg_types_error() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    assert_all_error(
        &st,
        "hybrid_search types",
        &[
            // text_col must be a string literal.
            format!("SELECT _id FROM hybrid_search(0, 'rust', 'emb', '{qv}', 8)"),
            // vec_col must be a string literal.
            format!("SELECT _id FROM hybrid_search('title', 'rust', 1, '{qv}', 8)"),
            // Vector literal must parse.
            "SELECT _id FROM hybrid_search('title', 'rust', 'emb', 'x,y', 8)".to_string(),
            // `k` must be an integer literal.
            format!("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', 'k')"),
        ],
    );
}

// ---------------------------------------------------------------------
// Valid-but-unusual queries — alternate execute / projection branches.
// ---------------------------------------------------------------------

/// `SELECT _id` (id-only fast path), `SELECT *` (full scalar
/// materialization), and `SELECT <scalar>, score` (mixed projection)
/// over the ranked TVFs all succeed and project the expected columns.
#[test]
fn varied_projections_succeed_for_ranked_tvfs() {
    let st = demo_table();
    let qv = csv_one_hot(0);

    // _id-only projection (arithmetic id fast path, no scalar decode).
    for q in [
        format!("SELECT _id FROM bm25_search('title', 'rust', {SURFACE_TOP_K})"),
        format!("SELECT _id FROM vector_search('emb', '{qv}', {SURFACE_TOP_K})"),
    ] {
        let b = st.reader().query_sql(&q).expect("id-only query");
        assert_eq!(b[0].schema().field(0).name(), "_id", "{q}");
        assert_eq!(
            b[0].num_columns(),
            1,
            "id-only must project one column: {q}"
        );
    }

    // Mixed scalar + score projection drives the scalar-decode path.
    let mixed = st
        .reader()
        .query_sql(&format!(
            "SELECT rating, score FROM bm25_search('title', 'rust', {SURFACE_TOP_K})"
        ))
        .expect("mixed projection");
    assert!(mixed[0].schema().index_of("rating").is_ok());
    assert!(mixed[0].schema().index_of("score").is_ok());

    // `SELECT *` materializes every scalar column plus score.
    let star = st
        .reader()
        .query_sql(&format!(
            "SELECT * FROM vector_search('emb', '{qv}', {SURFACE_TOP_K})"
        ))
        .expect("star projection");
    assert!(star[0].schema().index_of("title").is_ok());
    assert!(star[0].schema().index_of("rating").is_ok());
    assert!(star[0].schema().index_of("score").is_ok());
    assert!(
        star[0].schema().index_of("emb").is_err(),
        "vector column must never reach SQL"
    );
}

/// `k = 0` is valid and must yield an empty result (the empty-hits
/// branch in `resolve_hits`), not an error.
#[test]
fn zero_k_yields_empty_result() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    for q in [
        "SELECT _id, score FROM bm25_search('title', 'rust', 0)".to_string(),
        format!("SELECT _id, score FROM vector_search('emb', '{qv}', 0)"),
        format!("SELECT * FROM hybrid_search('title', 'rust', 'emb', '{qv}', 0)"),
    ] {
        let b = st.reader().query_sql(&q).expect("k=0 query");
        assert_eq!(row_count(&b), 0, "k=0 must produce no rows: {q}");
    }
}

/// A query term that matches nothing exercises the empty-result path of
/// the ranked + unranked TVFs without erroring.
#[test]
fn empty_result_queries_return_no_rows() {
    let st = demo_table();
    for q in [
        // No document contains this token.
        format!("SELECT _id FROM bm25_search('title', 'zzzznomatch', {SURFACE_TOP_K})"),
        "SELECT _id FROM token_match('title', 'zzzznomatch')".to_string(),
        // No title equals this exact value.
        "SELECT _id FROM exact_match('title', 'no such exact title')".to_string(),
    ] {
        let b = st.reader().query_sql(&q).expect("empty-result query");
        assert_eq!(row_count(&b), 0, "expected no matches: {q}");
    }
}

// ---------------------------------------------------------------------
// Base-table malformed SQL — the plain query-error path.
// ---------------------------------------------------------------------

/// Malformed or semantically invalid base-table statements must surface
/// as `query_sql` errors, not panics.
#[test]
fn malformed_base_table_sql_errors() {
    let st = demo_table();
    assert_all_error(
        &st,
        "base-table malformed",
        &[
            // Syntactically invalid SQL.
            "SELECT * FROM supertable WHERE".to_string(),
            "SELCT * FROM supertable".to_string(),
            // References a column that does not exist.
            "SELECT no_such_column FROM supertable".to_string(),
            // References a table that does not exist.
            "SELECT * FROM no_such_table".to_string(),
        ],
    );
}
