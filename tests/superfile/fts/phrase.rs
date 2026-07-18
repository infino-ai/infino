// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end tests for exact phrase queries (`"a b"`), pinning the
//! semantics against corpus truth and the brute-force phrase oracle:
//!
//! * a phrase matches only contiguous, in-order token sequences;
//! * its per-doc tf is the number of occurrence starts (overlaps
//!   like "a a a" for "a a" count each start);
//! * it composes with the clause model in every polarity;
//! * scores agree with the oracle's sum-idf phrase atom.

use std::collections::{HashMap, HashSet};

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use crate::fts::brute_force_oracle::{
    build_infino_superfile_positional, build_multi_block_corpus, corpus,
};

/// k large enough to capture every match on the 60-doc corpus.
const K_ALL: usize = 64;
/// k covering every match in the 1000-doc multi-block corpus.
const K_ALL_MULTI_BLOCK: usize = 1024;
/// Score-equality tolerance between the two BM25 implementations.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

async fn search_hits(
    reader: &SuperfileReader,
    query: &str,
    k: usize,
    mode: BoolMode,
) -> Vec<(u64, f32)> {
    reader
        .bm25_hits_async("title", query, k, mode)
        .await
        .expect("phrase query")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect()
}

/// Compare reader results against the phrase oracle for `query`:
/// identical match sets, per-doc scores within tolerance.
async fn assert_matches_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    mode: BoolMode,
    k: usize,
) {
    let tok = default_tokenizer();
    let clauses = tok.parse(query).into_clauses(mode);
    let own = |v: Vec<std::borrow::Cow<'_, str>>| -> Vec<String> {
        v.into_iter().map(|t| t.into_owned()).collect()
    };
    let own_ph = |v: Vec<Vec<std::borrow::Cow<'_, str>>>| -> Vec<Vec<String>> {
        v.into_iter()
            .map(|p| p.into_iter().map(|t| t.into_owned()).collect())
            .collect()
    };
    let want = oracle.top_k_atoms(
        &own(clauses.musts),
        &own_ph(clauses.must_phrases),
        &own(clauses.shoulds),
        &own_ph(clauses.should_phrases),
        &own(clauses.negatives),
        &own_ph(clauses.negative_phrases),
        k,
    );
    let got = search_hits(reader, query, k, mode).await;

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(got_ids, want_ids, "query {query:?}: match sets disagree");

    let want_scores: HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        let w = want_scores[d];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "query {query:?} doc {d}: reader score {s} vs oracle {w}"
        );
    }
}

// ── planted semantics ─────────────────────────────────────────────────

/// Small corpus with deliberate phrase shapes: boundaries, repeats,
/// overlapping self-phrases, and near-misses.
fn phrase_corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "new york city"),               // match at doc start
        (1, "i love new york"),             // match at doc end
        (2, "york new haven"),              // both words, wrong order
        (3, "new deal in york county"),     // both words, not adjacent
        (4, "new york new york"),           // phrase twice
        (5, "buzz buzz buzz"),              // overlapping "buzz buzz"
        (6, "the new york times daily"),    // interior match
        (7, "brand new yorkshire pudding"), // prefix-ish near miss
    ]
}

#[tokio::test]
async fn phrase_matches_only_contiguous_in_order() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let hits = search_hits(&r, r#""new york""#, K_ALL, BoolMode::Or).await;
    let mut ids: Vec<u64> = hits.iter().map(|(d, _)| *d).collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![0, 1, 4, 6], "boundaries + interior + repeats");
}

#[tokio::test]
async fn overlapping_occurrences_each_count() {
    // "buzz buzz" in "buzz buzz buzz": starts at 0 and 1 → tf 2,
    // which must outscore a doc where... there is only one such doc;
    // pin via the oracle instead.
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_matches_oracle(&r, &oracle, r#""buzz buzz""#, BoolMode::Or, K_ALL).await;
}

#[tokio::test]
async fn phrase_oracle_agreement_small_corpus() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    for query in [
        r#""new york""#,
        r#""new york" city"#,
        r#"+"new york" +times"#,
        r#"haven -"new york""#,
        r#""new york city""#,
        r#""york new""#,
        r#"+"new york" -times"#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL).await;
        assert_matches_oracle(&r, &oracle, query, BoolMode::And, K_ALL).await;
    }
}

#[tokio::test]
async fn three_token_phrase_and_absent_member() {
    let corp = phrase_corpus();
    let r = build_infino_superfile_positional(&corp);
    let hits = search_hits(&r, r#""new york city""#, K_ALL, BoolMode::Or).await;
    assert_eq!(
        hits.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
        vec![0],
        "three-token phrase"
    );
    let hits = search_hits(&r, r#""new zealand""#, K_ALL, BoolMode::Or).await;
    assert!(hits.is_empty(), "absent member matches nothing");
}

#[tokio::test]
async fn single_and_repeated_words_on_sixty_doc_corpus() {
    // The negation-suite corpus, positional: phrase behavior must
    // agree with the oracle on organic text too.
    let corp = corpus();
    let r = build_infino_superfile_positional(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    for query in [
        r#""rust async""#,
        r#""web framework""#,
        r#""search engine" rust"#,
        r#"+"inverted index""#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL).await;
    }
}

// ── multi-block corpus (positions cross skip boundaries) ─────────────

#[tokio::test]
async fn phrase_oracle_agreement_multi_block() {
    // "alpha beta" is adjacent in every doc divisible by 12 (~83
    // docs), with alpha spanning 3 PFOR blocks — the per-block run
    // offsets and the block-crossing position decode are all
    // exercised, as is a phrase should over a large union.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());
    for query in [
        r#""alpha beta""#,
        r#""alpha beta gamma""#,
        r#"+"alpha beta" epsilon"#,
        r#"delta -"alpha beta""#,
        r#""beta gamma" "gamma delta""#,
    ] {
        assert_matches_oracle(&r, &oracle, query, BoolMode::Or, K_ALL_MULTI_BLOCK).await;
    }
    // Truncated top-k exercises the atom walk's pruning bar.
    assert_matches_oracle(&r, &oracle, r#""alpha beta" gamma"#, BoolMode::Or, 10).await;
}
