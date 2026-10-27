// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Oracle tests for score-floor pruning
//! (`bm25_search_pretokenized_with_floor`), a reader surface the
//! single-column term/phrase oracles don't exercise. The floor is a
//! work-pruning hint the supertable fan-out uses to share the global
//! kth-best across segments. Verified by its load-bearing contract — a
//! floor set at the unfloored kth-best score must preserve the top-k
//! exactly (same docs, same scores) — rather than as an exact output
//! filter, which it is not: under a large k a kernel may still emit some
//! below-floor docs (they lose in the global merge).

use std::collections::HashSet;

use infino::superfile::{SuperfileReader, fts::reader::BoolMode};

use crate::fts::brute_force_oracle::{build_infino_superfile, build_multi_block_corpus};

const SCORE_ABS_TOLERANCE: f32 = 1e-3;

// ── score-floor pruning ───────────────────────────────────────────────

/// The `k` the floor tests rank to.
const FLOOR_K: usize = 16;

/// A `k` at which the union's top k collect into a buffer, under the
/// corpus's 600-odd `alpha | beta | gamma` matches.
const FLOOR_LARGE_K: usize = 512;

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
async fn assert_floor_preserves_topk(
    reader: &SuperfileReader,
    terms: &[&str],
    mode: BoolMode,
    k: usize,
) {
    let full = reader
        .bm25_search_pretokenized_with_floor("title", terms, k, mode, f32::NEG_INFINITY)
        .await
        .expect("unfloored top-k");
    assert_eq!(
        full.len(),
        k,
        "corpus must supply more than k matches for {terms:?} {mode:?}"
    );

    // Floor at the kth-best score — an active, load-bearing value: the
    // kernels prune every block whose max can't reach it, yet the full
    // top-k must still come back because all k docs are at or above it.
    let kth = full.last().expect("k>0").1;
    let floored = reader
        .bm25_search_pretokenized_with_floor("title", terms, k, mode, kth)
        .await
        .expect("floored top-k");

    let full_ids: HashSet<u64> = full.iter().map(|(d, _)| u64::from(d.get())).collect();
    let got_ids: HashSet<u64> = floored.iter().map(|(d, _)| u64::from(d.get())).collect();
    assert_eq!(
        got_ids, full_ids,
        "floor={kth} on {terms:?} {mode:?}: floored top-k dropped/added a doc vs unfloored"
    );

    // Scores identical (compare per doc; the floor must not perturb them).
    let full_scores: std::collections::HashMap<u64, f32> =
        full.iter().map(|(d, s)| (u64::from(d.get()), *s)).collect();
    for (d, s) in &floored {
        let w = full_scores[&(u64::from(d.get()))];
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
    assert_floor_preserves_topk(&reader, &["alpha"], BoolMode::Or, FLOOR_K).await;
}

#[tokio::test]
async fn floor_multi_term_maxscore_preserves_topk() {
    // Multi-term OR ⇒ the windowed kernel; the floor lifts the initial bar
    // and prunes from the first window. At a large `k` the top k collect
    // into a buffer rather than a heap.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&refs);
    for k in [FLOOR_K, FLOOR_LARGE_K] {
        assert_floor_preserves_topk(&reader, &["alpha", "beta", "gamma"], BoolMode::Or, k).await;
    }
}

#[tokio::test]
async fn floor_and_intersection_preserves_topk() {
    // AND intersection ⇒ block-max-AND skips seeded from the floor.
    let owned = build_multi_block_corpus();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&refs);
    assert_floor_preserves_topk(&reader, &["alpha", "beta"], BoolMode::And, FLOOR_K).await;
}
