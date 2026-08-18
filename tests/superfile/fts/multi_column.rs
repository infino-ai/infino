// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Oracle tests for multi-column weighted BM25 (`bm25_search_multi`,
//! "most fields" semantics: each column is scored independently and the
//! per-column scores are summed by weight).
//!
//! The pipeline test only checks *membership* of the multi-column
//! result. This module pins the *scores*: a doc's combined score must
//! equal `Σ_col weight_col · bm25_col(doc, query)`, where each column's
//! BM25 uses that column's own df / avgdl / doc-length — the surface no
//! single-column oracle can reach. The reference is two independent
//! [`BruteForceBm25`] indices (one per column) combined by the same
//! weighted sum, cross-checked against the reader across several weight
//! configurations.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    test_helpers::{brute_force_bm25::BruteForceBm25, decimal128_ids, default_tokenizer},
};

/// Score-equality tolerance between the two BM25 scorers.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

/// Build a two-FTS-column superfile (`title`, `body`) from a corpus of
/// `(doc_id, title_text, body_text)`. `doc_id` == row index, so the
/// reader's `local_doc_id` is the user id.
fn build_two_column(corpus: &[(u64, &str, &str)]) -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![
            FtsConfig {
                column: "title".into(),
                positions: false,
            },
            FtsConfig {
                column: "body".into(),
                positions: false,
            },
        ],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(corpus.iter().map(|(i, _, _)| *i));
    let titles = LargeStringArray::from(corpus.iter().map(|(_, t, _)| *t).collect::<Vec<_>>());
    let bodies = LargeStringArray::from(corpus.iter().map(|(_, _, b)| *b).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(titles), Arc::new(bodies)],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    SuperfileReader::open(bytes).expect("open superfile")
}

/// The "most fields" oracle: per-doc `Σ_col weight · bm25_col(doc)`,
/// where each column's per-doc score comes from an independent
/// brute-force index over that column's text. A doc is in the result
/// iff it matches the query in at least one column. Returns descending
/// by score, doc-id tie-break — matching the reader's contract.
fn multi_column_oracle(
    per_column: &[(f32, &BruteForceBm25)],
    query: &str,
    k: usize,
) -> Vec<(u64, f32)> {
    let tok = default_tokenizer();
    let mut summed: HashMap<u64, f32> = HashMap::new();
    for (weight, oracle) in per_column {
        // k = n_docs ⇒ every matching doc in this column, fully scored.
        for (doc, score) in oracle.top_k(query, oracle.n_docs() as usize, tok.as_ref()) {
            *summed.entry(doc).or_insert(0.0) += weight * score;
        }
    }
    let mut scored: Vec<(u64, f32)> = summed.into_iter().collect();
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.truncate(k);
    scored
}

/// Planted two-column corpus. `rust` appears in various title/body
/// combinations so the weighting is observable: some docs match only in
/// title, some only in body, some in both, with different tfs and doc
/// lengths per column.
fn corpus() -> Vec<(u64, &'static str, &'static str)> {
    vec![
        (0, "rust async runtime", "tokio reactor loop"), // title-only match for "rust"
        (1, "python data tools", "rust ffi bindings safe"), // body-only match
        (2, "rust web framework", "rust actix axum tower"), // both columns
        (3, "go concurrency model", "channels select goroutine"), // no match
        (4, "rust rust systems", "low level rust rust rust"), // high tf both columns
        (5, "java spring boot", "enterprise beans only"), // no match
        (6, "rust", "a b c d e f g h rust"),             // short title vs long body (dl-norm)
        (7, "embedded rust firmware", "rust"),           // long title vs short body
    ]
}

/// Cross-check the reader's `bm25_search_multi` against the weighted
/// two-column oracle for a query at several weight configs.
async fn assert_multi_matches_oracle(
    reader: &SuperfileReader,
    title_oracle: &BruteForceBm25,
    body_oracle: &BruteForceBm25,
    query: &str,
    title_w: f32,
    body_w: f32,
    k: usize,
) {
    let got = reader
        .bm25_search_multi(
            &[("title", title_w), ("body", body_w)],
            query,
            k,
            BoolMode::Or,
        )
        .await
        .expect("bm25_search_multi");
    let want = multi_column_oracle(&[(title_w, title_oracle), (body_w, body_oracle)], query, k);

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d as u64).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    // Full k (k >= match count): the sets must be identical.
    if k >= want.len() {
        assert_eq!(
            got_ids, want_ids,
            "multi({title_w},{body_w}) {query:?}: match sets disagree"
        );
    } else {
        assert!(
            got_ids.is_subset(&want_ids),
            "multi({title_w},{body_w}) {query:?}: returned a non-match"
        );
    }

    // Per-doc weighted score must match the oracle within tolerance.
    let want_scores: HashMap<u64, f32> = want.iter().map(|(d, s)| (*d, *s)).collect();
    for (d, s) in &got {
        let w = want_scores[&(*d as u64)];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "multi({title_w},{body_w}) {query:?} doc {d}: reader={s} oracle={w}"
        );
    }
}

#[tokio::test]
async fn multi_column_weighted_scores_match_oracle() {
    let corp = corpus();
    let reader = build_two_column(&corp);
    let tok = default_tokenizer();
    let title_corpus: Vec<(u64, &str)> = corp.iter().map(|(i, t, _)| (*i, *t)).collect();
    let body_corpus: Vec<(u64, &str)> = corp.iter().map(|(i, _, b)| (*i, *b)).collect();
    let title_oracle = BruteForceBm25::index(&title_corpus, tok.as_ref());
    let body_oracle = BruteForceBm25::index(&body_corpus, tok.as_ref());

    // Several weightings, including the asymmetric and degenerate (0)
    // cases, so the weight actually drives the combined score and rank.
    for (tw, bw) in [(1.0, 1.0), (2.0, 1.0), (1.0, 3.0), (5.0, 0.0), (0.0, 1.0)] {
        for k in [8usize, 3] {
            assert_multi_matches_oracle(&reader, &title_oracle, &body_oracle, "rust", tw, bw, k)
                .await;
        }
    }
}

#[tokio::test]
async fn multi_column_multi_term_query_scores_match_oracle() {
    // A two-term query so each column contributes an OR union that the
    // weighted sum must combine correctly across columns.
    let corp = corpus();
    let reader = build_two_column(&corp);
    let tok = default_tokenizer();
    let title_corpus: Vec<(u64, &str)> = corp.iter().map(|(i, t, _)| (*i, *t)).collect();
    let body_corpus: Vec<(u64, &str)> = corp.iter().map(|(i, _, b)| (*i, *b)).collect();
    let title_oracle = BruteForceBm25::index(&title_corpus, tok.as_ref());
    let body_oracle = BruteForceBm25::index(&body_corpus, tok.as_ref());

    for (tw, bw) in [(1.0, 1.0), (3.0, 1.0), (1.0, 2.0)] {
        assert_multi_matches_oracle(
            &reader,
            &title_oracle,
            &body_oracle,
            "rust async",
            tw,
            bw,
            8,
        )
        .await;
    }
}

#[tokio::test]
async fn multi_column_weight_scales_single_column_match_linearly() {
    // Doc 0 matches "rust" only in its title ("rust async runtime"); its
    // body ("tokio reactor loop") has no "rust", so the body column
    // contributes exactly 0 to doc 0's combined score. The combined
    // score therefore equals `title_weight · title_score(doc 0)`, and
    // must scale *linearly* with the title weight — a direct proof the
    // weight is multiplied in, not ignored, that needs no oracle.
    let corp = corpus();
    let reader = build_two_column(&corp);

    let score_of = |hits: Vec<(u32, f32)>, doc: u32| -> f32 {
        hits.iter()
            .find(|(d, _)| *d == doc)
            .map(|(_, s)| *s)
            .expect("doc present in results")
    };

    let s1 = score_of(
        reader
            .bm25_search_multi(&[("title", 1.0), ("body", 1.0)], "rust", 8, BoolMode::Or)
            .await
            .expect("w=1"),
        0,
    );
    let s2 = score_of(
        reader
            .bm25_search_multi(&[("title", 2.0), ("body", 1.0)], "rust", 8, BoolMode::Or)
            .await
            .expect("w=2"),
        0,
    );
    // Body contributes 0 to doc 0, so doubling the title weight doubles
    // the whole combined score.
    assert!(
        (s2 - 2.0 * s1).abs() <= SCORE_ABS_TOLERANCE,
        "title-only doc 0: score must scale with title weight — s(w=1)={s1}, s(w=2)={s2}"
    );
    assert!(s1 > 0.0, "doc 0 must score for a title match");
}
