// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end tests for the flat clause model in `bm25_search`
//! queries: `+term` (must), bare term (should under `Or`, must under
//! `And`), `-term` (must-not). Expected sets are computed directly
//! from the planted corpus text, and scored results are compared
//! against the brute-force clause oracle, so each test pins the
//! semantics:
//!
//! * musts define the match set — their intersection, regardless of
//!   how many shoulds match;
//! * a should is scoring-only — it can raise a matching doc's rank
//!   but never adds or removes a match;
//! * a `-term` is a hard filter, unchanged from the negation work;
//! * under `And`, bare terms fold into the musts (default operator),
//!   so the sigil is only observable under `Or`.

use std::collections::HashSet;

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use crate::fts::brute_force_oracle::{
    build_infino_superfile, build_multi_block_corpus, build_multi_block_reader, corpus,
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

/// Docs matching ALL of `terms` (the must intersection).
fn and_match(corp: &[(u64, &str)], terms: &[&str]) -> HashSet<u64> {
    let mut sets = terms.iter().map(|t| docs_with(corp, t));
    let Some(first) = sets.next() else {
        return HashSet::new();
    };
    sets.fold(first, |acc, s| acc.intersection(&s).copied().collect())
}

/// Remove every doc containing any of `negatives` from `base`.
fn exclude(base: HashSet<u64>, corp: &[(u64, &str)], negatives: &[&str]) -> HashSet<u64> {
    let drop: HashSet<u64> = negatives.iter().flat_map(|t| docs_with(corp, t)).collect();
    base.difference(&drop).copied().collect()
}

/// Run a query and return the ranked hits.
async fn search_hits(
    reader: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
) -> Vec<(u64, f32)> {
    reader
        .bm25_hits_async("title", query, k, mode)
        .await
        .expect("bm25 query")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect()
}

/// The hit set alone (order-insensitive).
async fn search_set(
    reader: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
) -> HashSet<u64> {
    search_hits(reader, query, k, mode)
        .await
        .into_iter()
        .map(|(d, _)| d)
        .collect()
}

/// k large enough to capture every match on the 60-doc corpus, so
/// top-k truncation never hides a set-membership disagreement.
const K_ALL: usize = 64;

/// k covering every match in the 1000-doc multi-block corpus.
const K_ALL_MULTI_BLOCK: usize = 1024;

/// Score-equality tolerance comparing the two BM25 scorers (they
/// associate the identical formula's f32 operations differently).
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

// ── match-set semantics ───────────────────────────────────────────────

#[tokio::test]
async fn must_defines_match_set_should_never_extends_it() {
    // "+rust web": match set is exactly the rust docs. Docs with only
    // "web" (e.g. javascript/go web docs) must not appear even though
    // the should term hits them.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "+rust web", K_ALL, BoolMode::Or).await;
    assert_eq!(got, docs_with(&corp, "rust"), "+rust web (Or)");
}

#[tokio::test]
async fn matching_should_outranks_must_only_docs() {
    // "+rust async": every rust∧async doc must rank strictly above
    // every rust-only doc — the async idf dwarfs the dl-norm spread
    // on this corpus.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let hits = search_hits(&r, "+rust async", K_ALL, BoolMode::Or).await;
    let both = and_match(&corp, &["rust", "async"]);
    assert!(!both.is_empty(), "corpus sanity: rust∧async docs exist");
    let min_both = hits
        .iter()
        .filter(|(d, _)| both.contains(d))
        .map(|(_, s)| *s)
        .fold(f32::INFINITY, f32::min);
    let max_must_only = hits
        .iter()
        .filter(|(d, _)| !both.contains(d))
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        min_both > max_must_only,
        "rust∧async docs must outrank rust-only docs: {min_both} vs {max_must_only}"
    );
}

#[tokio::test]
async fn all_must_sigils_equal_and_mode() {
    // "+rust +web" under Or ≡ "rust web" under And — identical hits
    // AND identical scores (both run the same AND kernel).
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let sigils = search_hits(&r, "+rust +web", K_ALL, BoolMode::Or).await;
    let and_mode = search_hits(&r, "rust web", K_ALL, BoolMode::And).await;
    assert_eq!(sigils, and_mode, "+rust +web (Or) vs rust web (And)");
}

#[tokio::test]
async fn and_mode_folds_bare_terms_into_musts() {
    // Under And the bare term is a must too, so the sigil changes
    // nothing: "+rust web" (And) ≡ "rust web" (And).
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let mixed = search_hits(&r, "+rust web", K_ALL, BoolMode::And).await;
    let flat = search_hits(&r, "rust web", K_ALL, BoolMode::And).await;
    assert_eq!(mixed, flat, "+rust web (And) vs rust web (And)");
}

#[tokio::test]
async fn must_should_with_negation() {
    // "+rust web -async": rust docs minus async docs; web only scores.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "+rust web -async", K_ALL, BoolMode::Or).await;
    let want = exclude(docs_with(&corp, "rust"), &corp, &["async"]);
    assert_eq!(got, want, "+rust web -async (Or)");
}

#[tokio::test]
async fn absent_should_is_ignored() {
    // A should term absent from the index contributes nothing — the
    // hits (ids and scores) equal the bare must query's.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let with_ghost = search_hits(&r, "+rust zzzabsent", K_ALL, BoolMode::Or).await;
    let without = search_hits(&r, "+rust", K_ALL, BoolMode::Or).await;
    assert_eq!(with_ghost, without, "+rust zzzabsent vs +rust");
}

#[tokio::test]
async fn absent_must_empties_the_result() {
    // A must term absent from the index empties the intersection no
    // matter how common the should is.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let got = search_set(&r, "+zzzabsent rust", K_ALL, BoolMode::Or).await;
    assert!(got.is_empty(), "+zzzabsent rust (Or) must match nothing");
}

#[tokio::test]
async fn bare_plus_token_contributes_nothing() {
    // A lone "+" (no token after the sigil) is dropped by the parser;
    // the query behaves as if it weren't there.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let with_bare = search_hits(&r, "rust + web", K_ALL, BoolMode::Or).await;
    let without = search_hits(&r, "rust web", K_ALL, BoolMode::Or).await;
    assert_eq!(with_bare, without, "bare + must be a no-op");
}

#[tokio::test]
async fn single_must_equals_single_term_query() {
    // One atom scores identically whichever clause list it sits in.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let must = search_hits(&r, "+rust", K_ALL, BoolMode::Or).await;
    let bare = search_hits(&r, "rust", K_ALL, BoolMode::Or).await;
    assert_eq!(must, bare, "+rust vs rust (single-atom fast path)");
}

// ── oracle comparison (scores, small corpus) ──────────────────────────

/// Assert reader hits equal the clause oracle's: same doc set, and
/// each doc's score within tolerance. Rank order can differ on f32
/// ties, so compare per-doc scores rather than sequence order.
async fn assert_matches_clause_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    mode: BoolMode,
    k: usize,
) {
    let tok = default_tokenizer();
    let clauses = tok.parse(query).into_clauses(mode);
    let musts: Vec<String> = clauses.musts.iter().map(|t| t.to_string()).collect();
    let shoulds: Vec<String> = clauses.shoulds.iter().map(|t| t.to_string()).collect();
    let negatives: Vec<String> = clauses.negatives.iter().map(|t| t.to_string()).collect();

    let got = search_hits(reader, query, k, mode).await;
    let want = oracle.top_k_clauses(&musts, &shoulds, &negatives, k);

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "query {query:?}: match sets disagree");

    let want_scores: std::collections::HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        let w = want_scores[d];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "query {query:?} doc {d}: reader score {s} vs oracle {w}"
        );
    }
}

#[tokio::test]
async fn oracle_must_should_small_corpus() {
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_matches_clause_oracle(&r, &oracle, "+rust async", BoolMode::Or, K_ALL).await;
    assert_matches_clause_oracle(&r, &oracle, "+rust web async", BoolMode::Or, K_ALL).await;
    assert_matches_clause_oracle(&r, &oracle, "+web +framework rust", BoolMode::Or, K_ALL).await;
    assert_matches_clause_oracle(&r, &oracle, "+rust web -async", BoolMode::Or, K_ALL).await;
}

// ── oracle comparison (multi-block corpus) ────────────────────────────

#[tokio::test]
async fn oracle_must_should_multi_block() {
    // The 1000-doc planted corpus gives every clause term a
    // multi-block posting list, so the must walk crosses block
    // boundaries and the should cursors stream through skip_to across
    // blocks — the shapes the block-max bar arithmetic must survive.
    let owned = build_multi_block_corpus();
    let r = build_multi_block_reader(&owned);
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());

    // must=alpha (~334 postings, 3 blocks), should=beta (~250, 2 blocks)
    assert_matches_clause_oracle(&r, &oracle, "+alpha beta", BoolMode::Or, K_ALL_MULTI_BLOCK).await;
    // two musts, two shoulds
    assert_matches_clause_oracle(
        &r,
        &oracle,
        "+alpha +beta gamma delta",
        BoolMode::Or,
        K_ALL_MULTI_BLOCK,
    )
    .await;
    // rare must, common should, negation
    assert_matches_clause_oracle(
        &r,
        &oracle,
        "+epsilon alpha -delta",
        BoolMode::Or,
        K_ALL_MULTI_BLOCK,
    )
    .await;
    // truncated top-k (k ≪ matches) exercises the lowered pruning bar
    assert_matches_clause_oracle(&r, &oracle, "+alpha beta gamma", BoolMode::Or, 10).await;
}

// ── error semantics ───────────────────────────────────────────────────

#[tokio::test]
async fn negation_only_still_errors() {
    // The clause model doesn't change the negation-only rule: no must
    // and no should ⇒ nothing to rank.
    let corp = corpus();
    let r = build_infino_superfile(&corp);
    let err = r
        .bm25_hits_async("title", "-rust", K_ALL, BoolMode::Or)
        .await
        .expect_err("negation-only must error");
    let msg = err.to_string();
    assert!(
        msg.contains("negat") || msg.contains("Negation"),
        "unexpected error: {msg}"
    );
}
