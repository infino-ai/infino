// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Oracle tests for two reader surfaces the single-column term/phrase
//! oracles don't exercise:
//!
//! * **Prefix search** (`bm25_search_prefix`): the FST expands the
//!   prefix to every indexed term that starts with it and runs an OR
//!   over the expansion. Cross-checked against the brute-force OR of the
//!   same expansion computed from corpus truth.
//! * **Score-floor pruning** (`bm25_search_pretokenized_with_floor`):
//!   the floor is a work-pruning hint the supertable fan-out uses to
//!   share the global kth-best across segments. Verified by its
//!   load-bearing contract — a floor set at the unfloored kth-best score
//!   must preserve the top-k exactly (same docs, same scores) — rather
//!   than as an exact output filter, which it is not: under a large k a
//!   kernel may still emit some below-floor docs (they lose in the
//!   global merge).

use std::collections::HashSet;

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};

use crate::fts::brute_force_oracle::{build_infino_superfile, build_multi_block_corpus};

const SCORE_ABS_TOLERANCE: f32 = 1e-3;
const K_ALL: usize = 64;

// ── prefix search ─────────────────────────────────────────────────────

/// Corpus with deliberate shared prefixes so an expansion is non-trivial:
/// `run`, `runtime`, `running`, `rust`, `rustacean` all share `ru`, and
/// `rust*` is its own sub-group. `python`/`pandas` share `p` but never
/// `ru`/`rust`.
fn prefix_corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "run the loop"),
        (1, "runtime scheduler tokio"),
        (2, "running fast async"),
        (3, "rust systems language"),
        (4, "rustacean community crate"),
        (5, "rust runtime bindings"), // two ru*-terms in one doc → OR sums both
        (6, "python data pipeline"),
        (7, "pandas dataframe numpy"),
        (8, "go concurrency channels"),
    ]
}

/// Distinct indexed terms (whitespace tokens, lowercased) beginning with
/// `prefix` — the set the FST expansion must reproduce.
fn expansion(corp: &[(u64, &str)], prefix: &str) -> Vec<String> {
    let mut terms: HashSet<String> = HashSet::new();
    for (_, text) in corp {
        for w in text.split_whitespace() {
            if w.starts_with(prefix) {
                terms.insert(w.to_string());
            }
        }
    }
    terms.into_iter().collect()
}

async fn assert_prefix_matches_oracle(
    reader: &SuperfileReader,
    oracle: &BruteForceBm25,
    corp: &[(u64, &str)],
    prefix: &str,
    k: usize,
) {
    let got = reader
        .bm25_search_prefix("title", prefix, k)
        .await
        .expect("prefix search")
        .0;
    let terms = expansion(corp, prefix);
    let want = oracle.top_k_terms(&terms, k);

    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d as u64).collect();
    let want_ids: HashSet<u64> = want.iter().map(|(d, _)| *d).collect();
    if k >= want.len() {
        assert_eq!(got_ids, want_ids, "prefix {prefix:?}: match sets disagree");
    } else {
        assert!(
            got_ids.is_subset(&want_ids),
            "prefix {prefix:?}: returned a non-match"
        );
    }
    let want_scores: std::collections::HashMap<u64, f32> = want.into_iter().collect();
    for (d, s) in &got {
        let w = want_scores[&(*d as u64)];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "prefix {prefix:?} doc {d}: reader={s} oracle={w}"
        );
    }
}

#[tokio::test]
async fn prefix_expansion_matches_brute_force_or() {
    let corp = prefix_corpus();
    let reader = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());

    // Broad ("ru*" = run/runtime/running/rust/rustacean), narrower
    // ("rust*" = rust/rustacean), and single-term ("runt*" = runtime).
    for prefix in ["ru", "run", "rust", "runt", "p"] {
        assert_prefix_matches_oracle(&reader, &oracle, &corp, prefix, K_ALL).await;
    }
    // Truncated top-k over the broad expansion keeps the pruning bar live.
    assert_prefix_matches_oracle(&reader, &oracle, &corp, "ru", 2).await;
}

#[tokio::test]
async fn prefix_uppercase_is_normalized() {
    // The prefix is lowercased before FST lookup, so "RU" and "ru" expand
    // to the same terms.
    let corp = prefix_corpus();
    let reader = build_infino_superfile(&corp);
    let upper = reader
        .bm25_search_prefix("title", "RU", K_ALL)
        .await
        .expect("upper")
        .0;
    let lower = reader
        .bm25_search_prefix("title", "ru", K_ALL)
        .await
        .expect("lower")
        .0;
    assert_eq!(upper, lower, "prefix must be case-normalized");
    assert!(!lower.is_empty(), "ru* must expand to something");
}

#[tokio::test]
async fn prefix_no_match_returns_empty() {
    let corp = prefix_corpus();
    let reader = build_infino_superfile(&corp);
    let hits = reader
        .bm25_search_prefix("title", "zzz", K_ALL)
        .await
        .expect("prefix search")
        .0;
    assert!(hits.is_empty(), "no term starts with zzz");
}

// ── score-floor pruning ───────────────────────────────────────────────

/// Verify the score floor's load-bearing contract: **top-k preservation
/// under a floor at the kth-best score**. The supertable fan-out seeds
/// each segment's floor with the global kth-best so segments skip work
/// on docs that can't make the global top-k; correctness requires that a
/// floor at or below the true kth-best never removes a doc that belongs
/// in the top-k. (The floor is a work-pruning hint, not an output
/// filter — under a large k a kernel like MaxScore may still emit some
/// below-floor docs — so "the output shrinks" is not a valid universal
/// assertion; top-k preservation is.)
///
/// Concretely: take the unfloored top-k, set the floor to its kth score,
/// re-run floored, and require an identical top-k (same docs, same
/// scores). A floor that drops a top-k doc, or that treats "equal to the
/// floor" as pruned, backfills with a lower-scoring doc and diverges.
/// `terms` is pre-tokenized; `mode` selects the kernel family.
async fn assert_floor_preserves_topk(reader: &SuperfileReader, terms: &[&str], mode: BoolMode) {
    const K: usize = 16;
    let full = reader
        .bm25_search_pretokenized_with_floor("title", terms, K, mode, f32::NEG_INFINITY)
        .await
        .expect("unfloored top-k");
    assert_eq!(
        full.len(),
        K,
        "corpus must supply more than K matches for {terms:?} {mode:?}"
    );

    // Floor at the kth-best score — an active, load-bearing value: the
    // kernels prune every block whose max can't reach it, yet the full
    // top-k must still come back because all K docs are at or above it.
    let kth = full.last().expect("k>0").1;
    let floored = reader
        .bm25_search_pretokenized_with_floor("title", terms, K, mode, kth)
        .await
        .expect("floored top-k");

    let full_ids: HashSet<u64> = full.iter().map(|(d, _)| *d as u64).collect();
    let got_ids: HashSet<u64> = floored.iter().map(|(d, _)| *d as u64).collect();
    assert_eq!(
        got_ids, full_ids,
        "floor={kth} on {terms:?} {mode:?}: floored top-k dropped/added a doc vs unfloored"
    );

    // Scores identical (compare per doc; the floor must not perturb them).
    let full_scores: std::collections::HashMap<u64, f32> =
        full.iter().map(|(d, s)| (*d as u64, *s)).collect();
    for (d, s) in &floored {
        let w = full_scores[&(*d as u64)];
        assert!(
            (s - w).abs() <= SCORE_ABS_TOLERANCE,
            "floor perturbed a top-k score on doc {d}: floored={s} unfloored={w}"
        );
    }
}

#[tokio::test]
async fn floor_single_term_bmw_preserves_topk() {
    // Single term ⇒ the BMW kernel; the floor seeds its threshold.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&refs);
    assert_floor_preserves_topk(&reader, &["alpha"], BoolMode::Or).await;
}

#[tokio::test]
async fn floor_multi_term_maxscore_preserves_topk() {
    // Multi-term OR ⇒ MaxScore+BMM; the floor lifts the initial bar.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&refs);
    assert_floor_preserves_topk(&reader, &["alpha", "beta", "gamma"], BoolMode::Or).await;
}

#[tokio::test]
async fn floor_and_intersection_preserves_topk() {
    // AND intersection ⇒ block-max-AND skips seeded from the floor.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&refs);
    assert_floor_preserves_topk(&reader, &["alpha", "beta"], BoolMode::And).await;
}
