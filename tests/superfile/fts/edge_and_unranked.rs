// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Edge-case matrix and the unranked match spine.
//!
//! * **Edge cases** the fuzz oracle doesn't reach by construction: `k =
//!   0`, a single-doc corpus, a degenerate all-equal-length corpus
//!   (constant dl-norm), and duplicate query terms (the fuzz generator
//!   dedups atoms, so this pins whatever the parser + scorer actually do
//!   with a repeated term, against the same parsed clauses).
//! * **The unranked spine** (`token_match` / `token_match_count`): a
//!   separate path from ranked search that must never prune. Pinned
//!   against corpus truth for both modes, with the ordering and
//!   `count == ids.len()` invariants, plus the cross-spine invariant
//!   that a ranked search returning every match (`k ≥ n`) has exactly
//!   `token_match_count` hits.

use std::collections::HashSet;

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use crate::fts::brute_force_oracle::{build_infino_superfile, corpus};

const SCORE_ABS_TOLERANCE: f32 = 1e-3;
const K_ALL: usize = 64;

/// Doc-ids whose text contains `term` as a whitespace token.
fn docs_with(corp: &[(u64, &str)], term: &str) -> HashSet<u64> {
    corp.iter()
        .filter(|(_, t)| t.split_whitespace().any(|w| w == term))
        .map(|(i, _)| *i)
        .collect()
}

async fn ranked_ids(reader: &SuperfileReader, query: &str, k: usize, mode: BoolMode) -> Vec<u64> {
    reader
        .bm25_hits_async("title", query, k, mode)
        .await
        .expect("bm25 search")
        .into_iter()
        .map(|(d, _)| d as u64)
        .collect()
}

// ── edge cases ────────────────────────────────────────────────────────

#[tokio::test]
async fn k_zero_returns_empty() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    // A query that otherwise matches many docs; k=0 must short-circuit to
    // empty across OR and AND.
    assert!(ranked_ids(&r, "rust", 0, BoolMode::Or).await.is_empty());
    assert!(
        ranked_ids(&r, "rust web", 0, BoolMode::And)
            .await
            .is_empty()
    );
}

#[tokio::test]
async fn k_one_returns_single_best() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let hits = ranked_ids(&r, "rare-token-zzz", 1, BoolMode::Or).await;
    assert_eq!(hits, vec![17], "k=1 returns the single unique match");
    // A common term: k=1 must return a top-scoring doc that the oracle
    // also ranks first (score-wise; tie-break may pick either doc).
    let got = ranked_ids(&r, "rust", 1, BoolMode::Or).await;
    assert_eq!(got.len(), 1);
    let top_score = oracle
        .top_k("rust", 1, tok.as_ref())
        .first()
        .expect("oracle top-1")
        .1;
    // The returned doc's score must equal the oracle's best score.
    let got_score = reader_score(&r, "rust", got[0]).await;
    assert!(
        (got_score - top_score).abs() <= SCORE_ABS_TOLERANCE,
        "k=1 score {got_score} != oracle best {top_score}"
    );
}

/// Score of a specific doc for a single-term query (via a k=all search).
async fn reader_score(reader: &SuperfileReader, query: &str, doc: u64) -> f32 {
    reader
        .bm25_hits_async("title", query, K_ALL, BoolMode::Or)
        .await
        .expect("search")
        .into_iter()
        .find(|(d, _)| *d as u64 == doc)
        .map(|(_, s)| s)
        .expect("doc present")
}

#[tokio::test]
async fn k_far_exceeds_matches_returns_all() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    // "rust" matches a known number of docs; k a thousand times larger
    // must return exactly that set, not panic or pad.
    let want = docs_with(&corp, "rust");
    let got: HashSet<u64> = ranked_ids(&r, "rust", 100_000, BoolMode::Or)
        .await
        .into_iter()
        .collect();
    assert_eq!(got, want, "k >> n must return exactly the match set");
}

#[tokio::test]
async fn single_doc_corpus() {
    // Degenerate corpus of one doc: avgdl == dl, so the length norm is
    // exactly 1 - b + b = 1. A matching query returns [0]; a non-matching
    // one returns empty.
    let corp = vec![(0u64, "rust async runtime tokio")];
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());

    let hits = ranked_hits(&r, "rust", K_ALL).await;
    assert_eq!(hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(), vec![0]);
    let want = oracle.top_k("rust", K_ALL, tok.as_ref());
    assert_eq!(want.len(), 1);
    assert!((hits[0].1 - want[0].1).abs() <= SCORE_ABS_TOLERANCE);

    assert!(
        ranked_ids(&r, "zzz-absent", K_ALL, BoolMode::Or)
            .await
            .is_empty(),
        "non-matching query on a single-doc corpus is empty"
    );
}

#[tokio::test]
async fn all_equal_length_docs_constant_norm() {
    // Every doc is exactly 3 tokens, so dl == avgdl and the length norm
    // is constant across docs; ranking then depends only on tf and idf.
    // Cross-check the full ranking against the oracle so the constant-norm
    // regime is pinned (a dl-norm bug that cancels on varied lengths could
    // hide here otherwise).
    let corp = vec![
        (0u64, "rust rust rust"), // tf=3
        (1, "rust rust alpha"),   // tf=2
        (2, "rust alpha beta"),   // tf=1
        (3, "alpha beta gamma"),  // tf=0
        (4, "rust rust rust"),    // tf=3 (ties doc 0)
    ];
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let got = ranked_hits(&r, "rust", K_ALL).await;
    let want = oracle.top_k("rust", K_ALL, tok.as_ref());

    // Same match set.
    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "constant-norm match set");
    // Per-doc scores agree.
    let want_scores: std::collections::HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        assert!((s - want_scores[d]).abs() <= SCORE_ABS_TOLERANCE);
    }
}

#[tokio::test]
async fn duplicate_query_term_matches_parsed_clauses() {
    // The fuzz generator dedups atoms, so a repeated term is never
    // generated. Pin whatever the parser + scorer actually do with
    // "rust rust" by scoring the *same parsed clauses* through the oracle
    // — the reader must agree with its own clause interpretation, so a
    // duplicate that double-counts in one path but not the other is
    // caught.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());

    let query = "rust rust";
    let clauses = tok.parse(query).into_clauses(BoolMode::Or);
    let shoulds: Vec<String> = clauses.shoulds.iter().map(|t| t.to_string()).collect();
    let want = oracle.top_k_clauses(&[], &shoulds, &[], K_ALL);

    let got = ranked_hits(&r, query, K_ALL).await;
    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "duplicate-term match set");
    let want_scores: std::collections::HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        assert!(
            (s - want_scores[d]).abs() <= SCORE_ABS_TOLERANCE,
            "duplicate-term score on doc {d}: reader={s} oracle={}",
            want_scores[d]
        );
    }
}

/// Ranked search returning `(doc, score)` pairs.
async fn ranked_hits(reader: &SuperfileReader, query: &str, k: usize) -> Vec<(u64, f32)> {
    reader
        .bm25_hits_async("title", query, k, BoolMode::Or)
        .await
        .expect("search")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect()
}

// ── unranked spine ────────────────────────────────────────────────────

#[tokio::test]
async fn token_match_matches_corpus_truth() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);

    // OR: union of the two terms' docs.
    let or_ids = r
        .token_match("title", &["rust", "python"], BoolMode::Or)
        .await
        .expect("token_match")
        .0;
    let or_want: HashSet<u64> = docs_with(&corp, "rust")
        .union(&docs_with(&corp, "python"))
        .copied()
        .collect();
    assert_eq!(
        or_ids.iter().map(|d| *d as u64).collect::<HashSet<_>>(),
        or_want,
        "token_match OR"
    );
    // Ascending order is part of the contract.
    let mut sorted = or_ids.clone();
    sorted.sort_unstable();
    assert_eq!(or_ids, sorted, "token_match must return ascending ids");

    // AND: intersection.
    let and_ids = r
        .token_match("title", &["rust", "async"], BoolMode::And)
        .await
        .expect("token_match")
        .0;
    let and_want: HashSet<u64> = docs_with(&corp, "rust")
        .intersection(&docs_with(&corp, "async"))
        .copied()
        .collect();
    assert_eq!(
        and_ids.iter().map(|d| *d as u64).collect::<HashSet<_>>(),
        and_want,
        "token_match AND"
    );
}

#[tokio::test]
async fn token_match_count_equals_ids_len() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    for (tokens, mode) in [
        (vec!["rust"], BoolMode::Or),
        (vec!["rust", "python"], BoolMode::Or),
        (vec!["rust", "async"], BoolMode::And),
        (vec!["rust", "web", "framework"], BoolMode::And),
        (vec!["definitely-absent"], BoolMode::Or),
    ] {
        let ids = r
            .token_match("title", &tokens, mode)
            .await
            .expect("token_match")
            .0;
        let count = r
            .token_match_count("title", &tokens, mode)
            .await
            .expect("token_match_count")
            .0;
        assert_eq!(
            count as usize,
            ids.len(),
            "count must equal ids len for {tokens:?} {mode:?}"
        );
    }
}

#[tokio::test]
async fn ranked_hits_count_equals_unranked_count_when_k_covers_all() {
    // Cross-spine invariant: a ranked search with k >= n returns every
    // match, so its hit count must equal the unranked token_match_count
    // for the same tokens and mode.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    for (query, tokens, mode) in [
        ("rust", vec!["rust"], BoolMode::Or),
        ("rust python", vec!["rust", "python"], BoolMode::Or),
        ("rust async", vec!["rust", "async"], BoolMode::And),
    ] {
        let ranked = ranked_ids(&r, query, K_ALL, mode).await;
        let count = r
            .token_match_count("title", &tokens, mode)
            .await
            .expect("token_match_count")
            .0;
        assert_eq!(
            ranked.len(),
            count as usize,
            "ranked hits ({}) != unranked count ({count}) for {query:?} {mode:?}",
            ranked.len()
        );
    }
}
