// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Direct integration coverage for the sync reader search methods that
//! mirror `bm25_search` / `vector_search`: `token_match`, `exact_match`,
//! and `hybrid_search`.
//!
//! The in-crate unit tests (`match_exec.rs`, `hybrid_exec.rs`) cover the
//! SQL table-valued functions and the RRF math on a single superfile.
//! This file exercises the published
//! `SupertableReader::{token_match, exact_match, hybrid_search}` surface
//! the way a Rust consumer calls it — `st.reader().token_match(..)` —
//! over a committed **multi-superfile** corpus so the per-superfile work
//! fans out and merges across superfiles.
//!
//! Hits are identified by their `(superfile, local_doc_id)` pair (the same
//! identity RRF fuses on); expectations are pinned by comparing against
//! the `bm25_search` / `vector_search` sub-searches rather than by
//! hand-mapping doc positions, so the assertions hold regardless of how
//! a commit shards rows into superfiles.

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::supertable::query::SuperfileHit;
use infino::supertable::query::vector::VectorSearchOptions;
use infino::supertable::{SuperfileUri, Supertable, SupertableOptions};
use infino::test_helpers::{default_tokenizer, default_vector_config};

/// `default_vector_config` is dim=16, cosine, n_cent=4.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 7;
/// Rayon pool size for deterministic builds.
const RAYON_POOL_THREADS: usize = 2;
/// One-hot query dimension; doc 0 ("rust async") is one-hot at dim 0, so
/// this query's exact nearest neighbour is doc 0.
const QUERY_DIM: usize = 0;
/// Top-k ≥ the corpus size, so a retriever returns its whole candidate
/// set and RRF never truncates the fused union.
const TOP_K: usize = 32;
/// Smaller top-k for the ranking-order assertions.
const RANK_TOP_K: usize = 8;
/// `rust` is sprinkled across both superfiles; this guards against corpus
/// drift silently making the cross-superfile assertions trivial.
const MIN_RUST_HITS: usize = 4;

/// First superfile's titles; doc `i`'s embedding is one-hot at dim `i`.
const SEG1_TITLES: &[&str] = &[
    "rust async",   // 0  — `async` is unique to this doc
    "python data",  // 1
    "java spring",  // 2
    "go rust",      // 3
    "ruby rails",   // 4
    "scala akka",   // 5
    "kotlin flow",  // 6
    "rust systems", // 7  — only doc with both `rust` and `systems`
];
/// Second superfile's titles; doc `i` (global) is one-hot at dim `i`.
const SEG2_TITLES: &[&str] = &[
    "swift ui",      // 8
    "rust web",      // 9
    "haskell pure",  // 10
    "elixir beam",   // 11
    "rust embedded", // 12
    "clojure lisp",  // 13
    "erlang otp",    // 14
    "rust macro",    // 15
];

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Schema `[title (FTS), emb (vector)]`. The vector column never
/// surfaces as a scalar SQL column; here it backs `vector_search` /
/// `hybrid_search` only.
fn options_title_emb() -> SupertableOptions {
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
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
    .with_writer_pool(writer_pool)
}

/// Doc `i` (within the batch) gets `titles[i]` and a one-hot embedding
/// at global dim `base_dim + i`.
fn build_batch(titles: &[&str], base_dim: usize, schema: Arc<Schema>) -> RecordBatch {
    let title_arr = LargeStringArray::from(titles.to_vec());
    let mut flat = Vec::<f32>::with_capacity(titles.len() * DIM);
    for i in 0..titles.len() {
        let active = base_dim + i;
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
    RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(fsl)]).expect("batch")
}

/// Two committed superfiles built from [`SEG1_TITLES`] then [`SEG2_TITLES`].
fn demo_two_superfiles() -> Supertable {
    let st = Supertable::create(options_title_emb()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(SEG1_TITLES, 0, schema.clone()))
        .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&build_batch(SEG2_TITLES, SEG1_TITLES.len(), schema))
        .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

fn one_hot(active: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| if d == active { 1.0 } else { 0.0 })
        .collect()
}

/// Stable per-row identity: the `(superfile, local_doc_id)` pair RRF and
/// the resolve path both key on.
fn hit_ids(hits: &[SuperfileHit]) -> HashSet<(SuperfileUri, u32)> {
    hits.iter().map(|h| (h.superfile, h.local_doc_id)).collect()
}

#[test]
fn token_match_or_is_the_unranked_bm25_candidate_set() {
    let st = demo_two_superfiles();
    let reader = st.reader();
    let token = reader
        .token_match("title", "rust", BoolMode::Or)
        .expect("token_match OR");
    let bm25 = reader
        .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
        .expect("bm25_search OR");

    assert!(
        bm25.len() >= MIN_RUST_HITS,
        "'rust' should match across both superfiles"
    );
    assert_eq!(
        hit_ids(&token),
        hit_ids(&bm25),
        "token_match OR must return exactly the BM25 OR candidate set, unranked"
    );
    assert!(
        token.iter().all(|h| h.score == 0.0),
        "token_match is unranked; score must be 0.0"
    );
}

#[test]
fn token_match_and_intersects_tokens() {
    let st = demo_two_superfiles();
    let reader = st.reader();
    // Only "rust systems" carries both tokens.
    let token = reader
        .token_match("title", "rust systems", BoolMode::And)
        .expect("token_match AND");
    let bm25 = reader
        .bm25_hits("title", "rust systems", TOP_K, BoolMode::And)
        .expect("bm25_search AND");

    assert!(!token.is_empty(), "AND of present tokens must match a doc");
    assert_eq!(
        hit_ids(&token),
        hit_ids(&bm25),
        "token_match AND must equal the BM25 AND candidate set"
    );
}

#[test]
fn exact_match_is_raw_value_equality_not_token() {
    let st = demo_two_superfiles();
    let reader = st.reader();
    // The raw title "rust async" equals exactly the docs that the
    // token-AND prune leaves (only doc 0 has both `rust` and `async`).
    let exact = reader
        .exact_match("title", "rust async")
        .expect("exact_match hit");
    let token_and = reader
        .token_match("title", "rust async", BoolMode::And)
        .expect("token_match AND");

    assert!(!exact.is_empty(), "raw value present must match");
    assert_eq!(
        hit_ids(&exact),
        hit_ids(&token_and),
        "exact 'rust async' must equal the docs whose whole title is that string"
    );
    assert!(
        exact.iter().all(|h| h.score == 0.0),
        "exact_match is unranked; score must be 0.0"
    );

    // A bare token no title *equals* returns nothing, even though many
    // titles *contain* it — exact_match compares the whole stored value.
    let none = reader
        .exact_match("title", "rust")
        .expect("exact_match miss");
    assert!(
        none.is_empty(),
        "exact_match compares the whole stored value, not tokens"
    );
}

#[test]
fn hybrid_search_unions_bm25_and_vector_and_orders_by_score() {
    let st = demo_two_superfiles();
    let reader = st.reader();
    let q = one_hot(QUERY_DIM);

    let hybrid = reader
        .hybrid_search(
            "title",
            "rust",
            BoolMode::Or,
            "emb",
            &q,
            VectorSearchOptions::new(),
            TOP_K,
        )
        .expect("hybrid_search");
    let bm25 = reader
        .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
        .expect("bm25_search");
    let vector = reader
        .vector_hits("emb", &q, TOP_K, VectorSearchOptions::new(), None)
        .expect("vector_search");

    assert!(
        bm25.len() >= MIN_RUST_HITS,
        "'rust' should match across both superfiles"
    );
    assert!(!vector.is_empty(), "vector retriever returned nothing");

    let expected: HashSet<(SuperfileUri, u32)> =
        hit_ids(&bm25).union(&hit_ids(&vector)).copied().collect();
    assert_eq!(
        hit_ids(&hybrid),
        expected,
        "hybrid identity set must equal bm25 ∪ vector at the same k"
    );

    for w in hybrid.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "fused RRF scores must be descending"
        );
    }
}

#[test]
fn hybrid_search_doc_top_in_both_retrievers_ranks_first() {
    let st = demo_two_superfiles();
    let reader = st.reader();
    let q = one_hot(QUERY_DIM);

    // `async` is unique to doc 0 (BM25 #1); the one-hot query at
    // `QUERY_DIM` makes doc 0 the exact vector match (#1). Top in both
    // retrievers ⇒ highest RRF ⇒ emitted first.
    let hybrid = reader
        .hybrid_search(
            "title",
            "async",
            BoolMode::Or,
            "emb",
            &q,
            VectorSearchOptions::new(),
            RANK_TOP_K,
        )
        .expect("hybrid_search");
    let bm25 = reader
        .bm25_hits("title", "async", RANK_TOP_K, BoolMode::Or)
        .expect("bm25_search");
    let vector = reader
        .vector_hits("emb", &q, RANK_TOP_K, VectorSearchOptions::new(), None)
        .expect("vector_search");

    assert!(
        !hybrid.is_empty() && !bm25.is_empty() && !vector.is_empty(),
        "all retrievers must return hits"
    );
    let top = (hybrid[0].superfile, hybrid[0].local_doc_id);
    assert_eq!(
        top,
        (bm25[0].superfile, bm25[0].local_doc_id),
        "fused #1 must be the BM25 #1"
    );
    assert_eq!(
        top,
        (vector[0].superfile, vector[0].local_doc_id),
        "fused #1 must also be the vector #1 (top in both ⇒ first)"
    );
}
