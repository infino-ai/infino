// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end tests for BM25 negation (`-term`) via
//! `SuperfileReader::bm25_search`. Expected sets are computed directly
//! from the planted corpus text, so each test pins the semantics:
//!
//! * positives are scored under `mode` (And/Or);
//! * a `-term` is a hard filter — docs containing it are removed,
//!   regardless of score;
//! * a query with only negated terms is an error (nothing to rank).

use std::collections::HashSet;

use infino::superfile::{SuperfileReader, fts::reader::BoolMode};

use crate::fts::brute_force_oracle::{
    build_infino_superfile, build_infino_superfile_positional, build_multi_block_corpus,
    build_multi_block_reader, corpus,
};

// ── corpus-truth helpers ──────────────────────────────────────────────
//
// The planted superfile has user `doc_id` == row index, so the reader's
// `local_doc_id` is the user id. "Term in doc" = whitespace-token match,
// which equals the tokenizer's view for this all-lowercase corpus.

/// Doc-ids whose text contains `term` as a whitespace token.
fn docs_with(corp: &[(u64, &str)], term: &str) -> HashSet<u64> {
    corp.iter()
        .filter(|(_, t)| t.split_whitespace().any(|w| w == term))
        .map(|(i, _)| *i)
        .collect()
}

/// Docs matching ANY of `terms` (the OR positive set).
fn or_match(corp: &[(u64, &str)], terms: &[&str]) -> HashSet<u64> {
    terms.iter().flat_map(|t| docs_with(corp, t)).collect()
}

/// Docs matching ALL of `terms` (the AND positive set).
fn and_match(corp: &[(u64, &str)], terms: &[&str]) -> HashSet<u64> {
    let mut sets = terms.iter().map(|t| docs_with(corp, t));
    let Some(first) = sets.next() else {
        return HashSet::new();
    };
    sets.fold(first, |acc, s| acc.intersection(&s).copied().collect())
}

/// Remove every doc containing any of `negatives` from `base`.
fn exclude(base: HashSet<u64>, corp: &[(u64, &str)], negatives: &[&str]) -> HashSet<u64> {
    let drop: HashSet<u64> = or_match(corp, negatives);
    base.difference(&drop).copied().collect()
}

/// Doc-ids whose text contains `phrase` as a contiguous, in-order token run.
fn docs_with_phrase(corp: &[(u64, &str)], phrase: &[&str]) -> HashSet<u64> {
    corp.iter()
        .filter(|(_, t)| {
            let toks: Vec<&str> = t.split_whitespace().collect();
            toks.windows(phrase.len()).any(|w| w == phrase)
        })
        .map(|(i, _)| *i)
        .collect()
}

/// A phrase as the `&[Vec<String>]` the count/search API expects.
fn phrase(tokens: &[&str]) -> Vec<Vec<String>> {
    vec![tokens.iter().map(|t| t.to_string()).collect()]
}

/// Run a query and collect the result doc-ids as a set.
async fn search_set(
    reader: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
) -> HashSet<u64> {
    reader
        .bm25_hits_async("title", query, k, mode)
        .await
        .expect("bm25_search")
        .into_iter()
        .map(|(d, _)| d as u64)
        .collect()
}

/// k large enough to capture every match on the 60-doc corpus, so
/// top-k truncation never hides a set-membership disagreement.
const K_ALL: usize = 64;

// ── OR + negation ─────────────────────────────────────────────────────

#[tokio::test]
async fn or_single_positive_minus_negative() {
    // "rust -async": docs with rust, minus docs with async.
    // Exercises the single-positive-term path (single-term BMW) + gate.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust -async", K_ALL, BoolMode::Or).await;
    let want = exclude(or_match(&corp, &["rust"]), &corp, &["async"]);
    assert_eq!(got, want, "rust -async (OR)");
}

/// Count with negation goes through the skip-based exclusion path
/// (`atoms_match_count` with negated atoms) rather than materializing the
/// negated union. Pin it against corpus truth for OR and AND positives
/// with single and multiple negatives.
#[tokio::test]
async fn count_with_negation_matches_corpus_truth() {
    let corp = corpus();
    // Positional index so the phrase cases below can run; term-only counts
    // are identical on a positional index.
    let r = build_infino_superfile_positional(&corp);
    async fn count(r: &SuperfileReader, pos: &[&str], mode: BoolMode, neg: &[&str]) -> u64 {
        r.atoms_match_count("title", pos, &[], mode, neg, &[])
            .await
            .expect("atoms_match_count")
            .0
    }
    // OR positive, one negative.
    assert_eq!(
        count(&r, &["rust"], BoolMode::Or, &["async"]).await,
        exclude(or_match(&corp, &["rust"]), &corp, &["async"]).len() as u64,
        "rust -async (OR count)"
    );
    // OR positive, multiple negatives.
    assert_eq!(
        count(&r, &["rust", "python"], BoolMode::Or, &["async", "web"]).await,
        exclude(
            or_match(&corp, &["rust", "python"]),
            &corp,
            &["async", "web"]
        )
        .len() as u64,
        "rust python -async -web (OR count)"
    );
    // AND positive, one negative.
    assert_eq!(
        count(&r, &["rust", "web"], BoolMode::And, &["async"]).await,
        exclude(and_match(&corp, &["rust", "web"]), &corp, &["async"]).len() as u64,
        "+rust +web -async (AND count)"
    );
    // No negatives ⇒ identical to the plain match count (the `None`-filter walk).
    assert_eq!(
        count(&r, &["rust"], BoolMode::Or, &[]).await,
        or_match(&corp, &["rust"]).len() as u64,
        "rust (OR count, no negation)"
    );

    // Positive phrase + negated term: +"web framework" -go. Docs with the
    // adjacent phrase, minus docs containing "go" (drops doc 7, keeps 8).
    let got = r
        .atoms_match_count(
            "title",
            &[],
            &phrase(&["web", "framework"]),
            BoolMode::And,
            &["go"],
            &[],
        )
        .await
        .expect("atoms_match_count")
        .0;
    let want = exclude(
        docs_with_phrase(&corp, &["web", "framework"]),
        &corp,
        &["go"],
    );
    assert_eq!(
        got,
        want.len() as u64,
        "+\"web framework\" -go (phrase + neg term)"
    );

    // Negated phrase: web -"web framework". Docs with "web", minus docs where
    // it is the adjacent phrase — keeps doc 4 ("web" but not "web framework").
    let got = r
        .atoms_match_count(
            "title",
            &["web"],
            &[],
            BoolMode::Or,
            &[],
            &phrase(&["web", "framework"]),
        )
        .await
        .expect("atoms_match_count")
        .0;
    let base = docs_with(&corp, "web");
    let drop = docs_with_phrase(&corp, &["web", "framework"]);
    let want = base.difference(&drop).count() as u64;
    assert_eq!(got, want, "web -\"web framework\" (term + neg phrase)");
}

#[tokio::test]
async fn or_multi_positive_minus_negative() {
    // "rust python -web": (rust ∪ python) minus web. Multi-term OR
    // path (MaxScore+BMM) + gate.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust python -web", K_ALL, BoolMode::Or).await;
    let want = exclude(or_match(&corp, &["rust", "python"]), &corp, &["web"]);
    assert_eq!(got, want, "rust python -web (OR)");
}

#[tokio::test]
async fn or_multiple_negatives() {
    // "rust -async -web": rust minus (async ∪ web).
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust -async -web", K_ALL, BoolMode::Or).await;
    let want = exclude(or_match(&corp, &["rust"]), &corp, &["async", "web"]);
    assert_eq!(got, want, "rust -async -web (OR)");
}

#[tokio::test]
async fn or_four_term_minus_negative() {
    // Four positive OR terms exercise the MaxScore+BMM SIMD-x4 scoring
    // pack (4 aligned cursors per doc) alongside the gate.
    // (rust ∪ python ∪ javascript ∪ go) minus framework.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(
        &r,
        "rust python javascript go -framework",
        K_ALL,
        BoolMode::Or,
    )
    .await;
    let want = exclude(
        or_match(&corp, &["rust", "python", "javascript", "go"]),
        &corp,
        &["framework"],
    );
    assert_eq!(got, want, "rust python javascript go -framework (OR)");
}

#[tokio::test]
async fn or_multi_positive_multi_negative() {
    // Multiple positives AND multiple negatives together.
    // (rust ∪ python) minus (web ∪ go).
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust python -web -go", K_ALL, BoolMode::Or).await;
    let want = exclude(or_match(&corp, &["rust", "python"]), &corp, &["web", "go"]);
    assert_eq!(got, want, "rust python -web -go (OR)");
}

// ── AND + negation ────────────────────────────────────────────────────

#[tokio::test]
async fn and_two_term_minus_negative() {
    // "rust async -tokio" AND (2-term path, run_and_intersect_2term):
    // docs with both rust and async ({0,20,22}), minus tokio (doc 0)
    // → {20, 22}.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust async -tokio", K_ALL, BoolMode::And).await;
    let want = exclude(and_match(&corp, &["rust", "async"]), &corp, &["tokio"]);
    assert_eq!(got, want, "rust async -tokio (AND)");
    assert_eq!(want, HashSet::from([20u64, 22]), "sanity: planted truth");
}

#[tokio::test]
async fn and_two_term_multiple_negatives() {
    // 2-term AND with two negatives. AND(rust, async) = {0,20,22};
    // tokio ∈ {0}, await ∈ {20} → {22}.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust async -tokio -await", K_ALL, BoolMode::And).await;
    let want = exclude(
        and_match(&corp, &["rust", "async"]),
        &corp,
        &["tokio", "await"],
    );
    assert_eq!(got, want, "rust async -tokio -await (AND)");
    assert_eq!(want, HashSet::from([22u64]), "sanity: planted truth");
}

#[tokio::test]
async fn and_three_term_minus_negative_keeps_match() {
    // 3+ positive terms route through run_and_intersect_general (the
    // separate ≥3 kernel, not the 2-term specialization). AND(rust,
    // web, framework) = {8}; negating a term absent from doc 8 keeps it.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust web framework -python", K_ALL, BoolMode::And).await;
    let want = exclude(
        and_match(&corp, &["rust", "web", "framework"]),
        &corp,
        &["python"],
    );
    assert_eq!(got, want, "rust web framework -python (AND)");
    assert_eq!(want, HashSet::from([8u64]), "sanity: planted truth");
}

#[tokio::test]
async fn and_three_term_negative_empties() {
    // Same general (≥3) AND kernel, but the negated term IS in the only
    // match (doc 8 has "axum") → exclusion empties the result.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust web framework -axum", K_ALL, BoolMode::And).await;
    let want = exclude(
        and_match(&corp, &["rust", "web", "framework"]),
        &corp,
        &["axum"],
    );
    assert_eq!(got, want, "rust web framework -axum (AND)");
    assert!(
        got.is_empty(),
        "negating axum must empty the result; got {got:?}"
    );
}

// ── negation overrides score ──────────────────────────────────────────

#[tokio::test]
async fn negation_drops_doc_regardless_of_score() {
    // doc 0 = "rust async runtime tokio" — contains "async" (would
    // score) but also "tokio". "async -tokio" must drop doc 0 even
    // though it matches the positive term. Negation is a hard filter,
    // not a score penalty.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "async -tokio", K_ALL, BoolMode::Or).await;
    assert!(
        !got.contains(&0),
        "doc 0 has tokio and must be excluded; got {got:?}"
    );
    let want = exclude(or_match(&corp, &["async"]), &corp, &["tokio"]);
    assert_eq!(got, want, "async -tokio (OR)");
}

// ── edge cases ────────────────────────────────────────────────────────

#[tokio::test]
async fn negated_term_absent_is_noop() {
    // "rust -xyzzy": xyzzy is in no doc, so it excludes nothing →
    // identical to plain "rust".
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let with_neg = search_set(&r, "rust -xyzzy", K_ALL, BoolMode::Or).await;
    let plain = search_set(&r, "rust", K_ALL, BoolMode::Or).await;
    assert_eq!(with_neg, plain, "absent negated term must be a no-op");
}

#[tokio::test]
async fn negated_equals_positive_is_empty() {
    // "rust -rust": every positive match is also excluded → empty.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "rust -rust", K_ALL, BoolMode::Or).await;
    assert!(got.is_empty(), "rust -rust must be empty; got {got:?}");
}

#[tokio::test]
async fn uppercase_negation_is_normalized() {
    // "-ASYNC" must exclude the same docs as "-async" (the negated
    // side is tokenized/lowercased like the index).
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let upper = search_set(&r, "rust -ASYNC", K_ALL, BoolMode::Or).await;
    let lower = search_set(&r, "rust -async", K_ALL, BoolMode::Or).await;
    assert_eq!(upper, lower, "negated term must be case-normalized");
}

#[tokio::test]
async fn negation_only_query_is_error() {
    // "-rust": no positive clause to rank → error (not an empty result,
    // not a silent OR of "rust"). M1 introduces FtsError::NegationOnly.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let res = r
        .bm25_hits_async("title", "-rust", K_ALL, BoolMode::Or)
        .await;
    assert!(res.is_err(), "negation-only query must error; got {res:?}");
}

// ── multi-block negated list ──────────────────────────────────────────

#[tokio::test]
async fn negation_with_multi_block_negated_list() {
    // In the 1000-doc planted corpus, `beta` (every 4th doc, ~250
    // postings) spans two PFOR blocks, so the exclude cursor must cross
    // a block boundary mid-walk. Truth: alpha (d % 3 == 0) minus beta
    // (d % 4 == 0).
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let got: HashSet<u64> = r
        .bm25_hits_async("title", "alpha -beta", 400, BoolMode::Or)
        .await
        .expect("multi-block negation")
        .into_iter()
        .map(|(d, _)| d as u64)
        .collect();
    let want: HashSet<u64> = (0..corp.len() as u64)
        .filter(|d| d % 3 == 0 && d % 4 != 0)
        .collect();
    assert_eq!(got, want, "alpha minus beta over multi-block postings");
}
