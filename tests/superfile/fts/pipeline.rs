// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end FTS pipeline integration test.
//!
//! Builds a small multi-column FTS index, opens it through `FtsReader`,
//! exercises the search API. Mirrors the planted-ground-truth
//! correctness pattern used across the integration tests.

use bytes::Bytes;
use infino::superfile::fts::builder::FtsBuilder;
use infino::superfile::fts::reader::{BoolMode, FtsReader};
use infino::test_helpers::default_tokenizer;

/// Document count for the two-column FTS pipeline fixture.
const FTS_PIPELINE_N_DOCS: u32 = 4;
/// BM25 top-k used across the pipeline searches.
const FTS_PIPELINE_SEARCH_K: usize = 10;
/// Top-k for the explicit top-k-limit test (expects a single hit).
const FTS_PIPELINE_TOPK_LIMIT_K: usize = 1;
/// Multi-column BM25 weights (title weighted above body).
const FTS_TITLE_WEIGHT: f32 = 2.0;
const FTS_BODY_WEIGHT: f32 = 1.0;
/// Corpus size for the Block-Max-WAND multi-block fixture.
const BMW_CORPUS_N_DOCS: u32 = 500;
/// Term-frequency cycle modulus (`i % BMW_TF_MODULUS + 1` → 1..=8).
const BMW_TF_MODULUS: u32 = 8;
/// Filler-token-count modulus, varying doc lengths.
const BMW_FILLER_MODULUS: u32 = 20;
/// Score-equality tolerance comparing the BMW vs multi-term paths.
const BMW_SCORE_TOLERANCE: f32 = 1e-5;

fn build_with_two_columns() -> (Bytes, String) {
    let tok = default_tokenizer();
    let mut b = FtsBuilder::new(tok);
    let _ = b.register_column("title".into()).expect("register column");
    let _ = b.register_column("body".into()).expect("register column");

    // Doc 0: title="rust runtime", body="tokio fast async"
    b.add_doc(0, 0, "rust runtime").expect("add doc");
    b.add_doc(1, 0, "tokio fast async").expect("add doc");
    // Doc 1: title="async io", body="rust ecosystem mature"
    b.add_doc(0, 1, "async io").expect("add doc");
    b.add_doc(1, 1, "rust ecosystem mature").expect("add doc");
    // Doc 2: title="java spring boot", body="enterprise grade only"
    b.add_doc(0, 2, "java spring boot").expect("add doc");
    b.add_doc(1, 2, "enterprise grade only").expect("add doc");
    // Doc 3: title="empty filler", body="filler only here"
    b.add_doc(0, 3, "empty filler").expect("add doc");
    b.add_doc(1, 3, "filler only here").expect("add doc");

    let bytes = b.finish().expect("finish fts");
    let json =
        r#"[{"name":"title","tokenizer":"ascii_lower"},{"name":"body","tokenizer":"ascii_lower"}]"#;
    (Bytes::from(bytes), json.to_string())
}

#[tokio::test]
async fn end_to_end_routing_per_column() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    assert_eq!(r.n_docs(), FTS_PIPELINE_N_DOCS);

    // "rust" appears in title doc 0; in body doc 1. Routing must NOT
    // confuse these (column isolation).
    let title_hits = r
        .search("title", &["rust"], FTS_PIPELINE_SEARCH_K, BoolMode::Or)
        .await
        .expect("FTS search");
    let title_ids: Vec<u32> = title_hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(title_ids, vec![0]);

    let body_hits = r
        .search("body", &["rust"], FTS_PIPELINE_SEARCH_K, BoolMode::Or)
        .await
        .expect("FTS search");
    let body_ids: Vec<u32> = body_hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(body_ids, vec![1]);
}

#[tokio::test]
async fn end_to_end_and_intersection() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    // body: "tokio AND async" → only doc 0.
    let hits = r
        .search(
            "body",
            &["tokio", "async"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::And,
        )
        .await
        .expect("search");
    let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(ids, vec![0]);
}

#[tokio::test]
async fn end_to_end_or_union() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    // body: "tokio OR mature" → docs 0 and 1.
    let hits = r
        .search(
            "body",
            &["tokio", "mature"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::Or,
        )
        .await
        .expect("search");
    let mut ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
    ids.sort();
    assert_eq!(ids, vec![0, 1]);
}

#[tokio::test]
async fn end_to_end_missing_term_or_drops() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    // OR with a missing term still returns hits for the present term.
    let hits = r
        .search(
            "body",
            &["nonexistent", "tokio"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::Or,
        )
        .await
        .expect("search");
    let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(ids, vec![0]);
}

#[tokio::test]
async fn end_to_end_missing_term_and_short_circuits() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    let hits = r
        .search(
            "body",
            &["nonexistent", "tokio"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::And,
        )
        .await
        .expect("search");
    assert!(hits.is_empty());
}

#[tokio::test]
async fn end_to_end_unknown_column_errors() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    let err = r
        .search(
            "not_a_column",
            &["rust"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::Or,
        )
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn end_to_end_top_k_limits_results() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    let hits = r
        .search("body", &["filler"], FTS_PIPELINE_TOPK_LIMIT_K, BoolMode::Or)
        .await
        .expect("FTS search");
    // Even though "filler" might match multiple docs, only 1 returned.
    assert!(hits.len() <= 1);
}

#[tokio::test]
async fn end_to_end_score_is_positive() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    let hits = r
        .search("body", &["tokio"], FTS_PIPELINE_SEARCH_K, BoolMode::Or)
        .await
        .expect("FTS search");
    for (_, s) in &hits {
        assert!(*s > 0.0, "BM25 score should be positive");
        assert!(s.is_finite());
    }
}

#[tokio::test]
async fn end_to_end_search_multi_weighted_combine() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    // search "rust" across both columns, title weighted 2x.
    let hits = r
        .search_multi(
            &[("title", FTS_TITLE_WEIGHT), ("body", FTS_BODY_WEIGHT)],
            "rust",
            FTS_PIPELINE_SEARCH_K,
            BoolMode::Or,
        )
        .await
        .expect("FTS multi-column search");
    let ids: Vec<u32> = hits.iter().map(|(d, _)| *d).collect();
    // doc 0 (title: "rust runtime") and doc 1 (body: "rust ecosystem mature") should both rank.
    assert!(ids.contains(&0));
    assert!(ids.contains(&1));
}

#[test]
fn end_to_end_n_terms_and_columns_reported_correctly() {
    let (blob, json) = build_with_two_columns();
    let r = FtsReader::open(blob, &json).expect("open FtsReader");
    let cols: Vec<&str> = r.fts_columns().collect();
    assert_eq!(cols, vec!["title", "body"]);
    assert!(r.n_terms() > 0);
}

/// BMW correctness: build a posting list spanning many blocks, run a
/// single-term query (which goes through the BMW path) and confirm the
/// top-k matches a hand-rolled brute-force scan (which goes through the
/// existing multi-term path with one term).
#[tokio::test]
async fn bmw_single_term_matches_brute_force() {
    use infino::superfile::fts::reader::BoolMode;

    let mut b = FtsBuilder::new(default_tokenizer());
    b.register_column("body".into()).expect("register column");

    // Build ~500 docs with the term "foo" appearing in every other doc
    // with varying tfs (1..=8) — produces multiple posting blocks, plenty
    // of skip-table headroom for BMW to chew on.
    for i in 0..BMW_CORPUS_N_DOCS {
        let n_foo = (i % BMW_TF_MODULUS) as usize + 1;
        let foos = std::iter::repeat_n("foo", n_foo)
            .collect::<Vec<_>>()
            .join(" ");
        let mut text = foos.clone();
        // Add some filler tokens so doc lengths vary.
        for j in 0..(i % BMW_FILLER_MODULUS) {
            text.push_str(&format!(" filler{j}"));
        }
        b.add_doc(0, i, &text).expect("add doc");
    }
    let bytes = b.finish().expect("finish fts");
    let json = r#"[{"name":"body","tokenizer":"ascii_lower"}]"#;
    let r = FtsReader::open(Bytes::from(bytes), json).expect("open FtsReader");

    // Single term → BMW path.
    let bmw_hits = r
        .search("body", &["foo"], FTS_PIPELINE_SEARCH_K, BoolMode::Or)
        .await
        .expect("FTS search");
    assert!(!bmw_hits.is_empty());
    assert_eq!(bmw_hits.len(), FTS_PIPELINE_SEARCH_K);

    // Two terms (one bogus, OR mode drops it) → multi-term full-decode path.
    // Should produce the same results as the BMW path.
    let mt_hits = r
        .search(
            "body",
            &["foo", "nonexistent_term_xyz"],
            FTS_PIPELINE_SEARCH_K,
            BoolMode::Or,
        )
        .await
        .expect("search");

    let bmw_ids: Vec<u32> = bmw_hits.iter().map(|(d, _)| *d).collect();
    let mt_ids: Vec<u32> = mt_hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(bmw_ids, mt_ids, "BMW and multi-term paths must agree");

    // Scores should match within FP noise.
    for ((d_bmw, s_bmw), (d_mt, s_mt)) in bmw_hits.iter().zip(mt_hits.iter()) {
        assert_eq!(d_bmw, d_mt);
        assert!(
            (s_bmw - s_mt).abs() < BMW_SCORE_TOLERANCE,
            "scores differ: BMW={s_bmw} multi-term={s_mt}"
        );
    }
}
