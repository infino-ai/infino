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

use super::corpus_truth::docs_with;
use crate::fts::brute_force_oracle::{
    build_infino_superfile, build_multi_block_corpus, build_multi_block_reader, corpus,
};

// ── corpus-truth helpers ──────────────────────────────────────────────
//
// The planted superfile has user `doc_id` == row index, so the reader's
// `local_doc_id` is the user id. "Term in doc" = whitespace-token match,
// which equals the tokenizer's view for this all-lowercase corpus.

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
        .map(|(d, s)| (u64::from(d.get()), s))
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

/// One required term with optional terms, pruned, against the same query
/// unpruned: a `k` larger than the match count never fills the heap, so it
/// prunes nothing, and its top `k` must be the pruned answer exactly (same
/// docs, same scores, same order). Corpora span many posting blocks, mix a
/// near-every-doc term with mid and rare ones, and vary document length so
/// block bounds differ from block to block.
#[tokio::test]
async fn one_must_with_shoulds_pruned_matches_unpruned() {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let vocab = ["t0", "t1", "t2", "t3", "t4", "t5"];
    for trial in 0..10u32 {
        let n_docs = 2_000 + rng() % 10_000;
        // Inclusion odds out of 1000: one stopword-like, the rest varied.
        let odds: Vec<u64> = (0..vocab.len())
            .map(|t| match t {
                0 => 900,
                _ => 5 + rng() % 400,
            })
            .collect();
        let mut owned: Vec<(u64, String)> = Vec::with_capacity(n_docs as usize);
        for d in 0..n_docs {
            let mut text = String::new();
            for (t, term) in vocab.iter().enumerate() {
                if rng() % 1000 < odds[t] {
                    for _ in 0..1 + rng() % 4 {
                        text.push_str(term);
                        text.push(' ');
                    }
                }
            }
            for _ in 0..rng() % 30 {
                text.push_str("filler ");
            }
            text.push_str("doc");
            owned.push((d, text));
        }
        let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
        let reader = build_infino_superfile(&refs);
        for _ in 0..8 {
            let must = vocab[(rng() % vocab.len() as u64) as usize];
            let mut shoulds: Vec<&str> = Vec::new();
            for _ in 0..1 + rng() % 3 {
                let s = vocab[(rng() % vocab.len() as u64) as usize];
                if s != must && !shoulds.contains(&s) {
                    shoulds.push(s);
                }
            }
            if shoulds.is_empty() {
                continue;
            }
            let mut query = format!("+{must}");
            for s in &shoulds {
                query.push(' ');
                query.push_str(s);
            }
            if rng() % 4 == 0 {
                query.push_str(" -filler");
            }
            let full = search_hits(&reader, &query, n_docs as usize + 1, BoolMode::Or).await;
            // Small `k` probes the optional terms strongest first; a `k` past
            // the top-k buffer's threshold probes them in query order.
            for k in [1usize, 5, 10, 100, 1_000] {
                let pruned = search_hits(&reader, &query, k, BoolMode::Or).await;
                let want = &full[..k.min(full.len())];
                assert_eq!(
                    pruned, want,
                    "trial {trial} query {query:?} k={k}: pruned top-k differs from unpruned"
                );
            }
        }
    }
}

/// A two-term OR of a rare term and a near-every-doc term at the small `k`
/// that routes it to the rare-term-first pass, against the same query at a
/// `k` past the match count, which prunes nothing. Rare terms range from a
/// handful of documents (fewer than `k`, so the pass must fall back) to a
/// few percent of the corpus.
#[tokio::test]
async fn rare_and_common_pair_matches_unpruned() {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for trial in 0..10u32 {
        let n_docs = 3_000 + rng() % 12_000;
        // Odd trials: the common term covers a fifth of the corpus, so its idf
        // and its best score are real, and reaches that score in short
        // documents that repeat it, while rare-term documents are long. The
        // rare-term pass must then often fall back rather than finish.
        let contested = trial % 2 == 1;
        let common_odds: u64 = if contested { 200 } else { 920 };
        let rare_odds: Vec<u64> = (0..4)
            .map(|_| {
                if contested {
                    1 + rng() % 12
                } else {
                    1 + rng() % 40
                }
            })
            .collect();
        let mut owned: Vec<(u64, String)> = Vec::with_capacity(n_docs as usize);
        for d in 0..n_docs {
            let mut text = String::new();
            let short = contested && rng() % 50 == 0;
            if rng() % 1000 < common_odds {
                let reps = if short { 4 + rng() % 8 } else { 1 + rng() % 6 };
                for _ in 0..reps {
                    text.push_str("the ");
                }
            }
            let mut has_rare = false;
            for (r, odds) in rare_odds.iter().enumerate() {
                if !short && rng() % 1000 < *odds {
                    has_rare = true;
                    for _ in 0..1 + rng() % 3 {
                        text.push_str(&format!("rare{r} "));
                    }
                }
            }
            let filler = match (contested, short, has_rare) {
                (_, true, _) => 0,
                (true, _, true) => 60 + rng() % 60,
                _ => rng() % 40,
            };
            for _ in 0..filler {
                text.push_str("filler ");
            }
            text.push_str("doc");
            owned.push((d, text));
        }
        let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
        let reader = build_infino_superfile(&refs);
        for r in 0..4 {
            for query in [format!("the rare{r}"), format!("rare{r} the")] {
                let full = search_hits(&reader, &query, n_docs as usize + 1, BoolMode::Or).await;
                for k in [1usize, 10, 100, 128] {
                    let got = search_hits(&reader, &query, k, BoolMode::Or).await;
                    assert_eq!(
                        got,
                        &full[..k.min(full.len())],
                        "trial {trial} query {query:?} k={k}"
                    );
                }
            }
        }
    }
}

/// Intersections of a rare term (under a sixty-fourth of the corpus, so the
/// rarest-driven membership walk takes them) with mid and common terms,
/// pruned, against the same query at a `k` past the match count, which
/// prunes nothing: the pruned top `k` must be the unpruned one exactly.
#[tokio::test]
async fn intersection_pruned_matches_unpruned() {
    let mut state: u64 = 0xD1B5_4A32_D192_ED03;
    let mut rng = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for trial in 0..8u32 {
        let n_docs = 20_000 + rng() % 20_000;
        let rare_odds: Vec<u64> = (0..3).map(|_| 3 + rng() % 12).collect();
        let mut owned: Vec<(u64, String)> = Vec::with_capacity(n_docs as usize);
        for d in 0..n_docs {
            let mut text = String::new();
            for (term, odds) in [
                ("common", 600u64),
                ("mid", 150),
                ("mida", 300),
                ("midb", 400),
            ] {
                if rng() % 1000 < odds {
                    for _ in 0..1 + rng() % 4 {
                        text.push_str(term);
                        text.push(' ');
                    }
                }
            }
            for (r, odds) in rare_odds.iter().enumerate() {
                if rng() % 1000 < *odds {
                    for _ in 0..1 + rng() % 3 {
                        text.push_str(&format!("rare{r} "));
                    }
                }
            }
            for _ in 0..rng() % 40 {
                text.push_str("filler ");
            }
            text.push_str("doc");
            owned.push((d, text));
        }
        let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
        let reader = build_infino_superfile(&refs);
        let mut queries: Vec<String> = Vec::new();
        for r in 0..3 {
            queries.push(format!("+rare{r} +common"));
            queries.push(format!("+rare{r} +mid"));
            queries.push(format!("+rare{r} +mid +common"));
        }
        queries.push("+mid +common".into());
        // Two mid-density terms, neither rare nor dense enough for presence
        // bits: the ranked walk that replaced the flat merge.
        queries.push("+mida +midb".into());
        queries.push("+mid +mida".into());
        queries.push("+mid +mida +midb".into());
        for query in &queries {
            let full = search_hits(&reader, query, n_docs as usize + 1, BoolMode::Or).await;
            for k in [1usize, 10, 100] {
                let got = search_hits(&reader, query, k, BoolMode::Or).await;
                assert_eq!(
                    got,
                    &full[..k.min(full.len())],
                    "trial {trial} query {query:?} k={k}"
                );
            }
        }
    }
}
