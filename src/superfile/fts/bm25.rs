// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 scoring math.
//!
//! Pure functions — no allocation, no I/O. Standard BM25 defaults:
//! `k1 = 1.2`, `b = 0.75`. The formula is the canonical BM25-with-IDF:
//!
//! ```text
//!   idf(N, df)            = ln( 1 + (N - df + 0.5) / (df + 0.5) )
//!
//!   norm(dl, avgdl)       = 1 - b + b * dl / avgdl
//!
//!   tf_factor(tf, dl, avgdl)
//!                         = tf * (k1 + 1) / ( tf + k1 * norm(dl, avgdl) )
//!
//!   score(idf, tf, dl, avgdl)
//!                         = idf * tf_factor(tf, dl, avgdl)
//! ```
//!
//! `idf(N, df)` is monotonic in `df` (smaller `df` → larger `idf`); always
//! non-negative because we use the +0.5 / +0.5 form ("BM25+1") which keeps
//! the log argument ≥ 1.
//!
//! `block_upper_bound` is the per-block maximum BM25 contribution used by
//! BlockMaxWAND for early termination — given a block's `max_tf` and
//! `min_dl`, compute the largest possible score any single doc in that
//! block can contribute. If this upper bound can't beat the current top-k
//! worst score, the whole block is skipped.

use wide::f32x4;

/// Standard BM25 default `k1` — term-frequency saturation parameter.
pub const K1: f32 = 1.2;

/// Standard BM25 default `b` — length-normalization parameter.
pub const B: f32 = 0.75;

/// Plus-half IDF smoothing term. Added to both the numerator and
/// denominator of the IDF log argument so it stays ≥ 1 (hence
/// `idf >= 0`) for every valid `(N, df)` — the "BM25+1" form.
const IDF_SMOOTHING: f64 = 0.5;

/// SIMD lane count for the four-wide BM25 scorer ([`f32x4`]). The
/// multi-term path scores this many cursors at one doc per call.
const SCORE_SIMD_LANES: usize = 4;

/// BM25 inverse-document-frequency. Plus-half smoothing keeps the log
/// argument ≥ 1, so `idf(N, df) >= 0` for all valid `(N, df)`.
///
/// Panics in debug builds if `df > n_docs` (caller bug).
#[inline]
pub fn idf(n_docs: u64, df: u64) -> f32 {
    debug_assert!(df <= n_docs, "df ({df}) > n_docs ({n_docs})");
    let n = n_docs as f64;
    let df = df as f64;
    let arg = 1.0 + (n - df + IDF_SMOOTHING) / (df + IDF_SMOOTHING);
    arg.ln() as f32
}

/// Per-doc BM25 contribution for a single (column, term, doc).
///
/// `tf`    — term frequency in this document, this column.
/// `dl`    — this document's length in this column (in tokens).
/// `avgdl` — average document length across the superfile, this column.
#[inline(always)]
pub fn score(idf_t: f32, tf: u32, dl: u32, avgdl: f32) -> f32 {
    let tf = tf as f32;
    // avgdl is precomputed at build time and stored in the doc-lengths
    // directory; if a superfile has zero docs we wouldn't be calling this
    // function, but guard anyway against a divide-by-zero on degenerate
    // input.
    let norm = if avgdl > 0.0 {
        1.0 - B + B * (dl as f32) / avgdl
    } else {
        1.0
    };
    let denom = tf + K1 * norm;
    if denom == 0.0 {
        // tf=0 should never reach this function (callers gate on
        // posting list membership), but stay defensive.
        return 0.0;
    }
    idf_t * tf * (K1 + 1.0) / denom
}

/// BM25 score using a precomputed `dl_norm_k1 = K1 * (1 - B + B * dl/avgdl)`
/// and `idf_x_k1p1 = idf * (K1 + 1)`.
///
/// Both `dl_norm_k1` (per doc) and `idf_x_k1p1` (per cursor) are
/// computed once at reader open / cursor build. The hot inner loop
/// drops to a single multiply + add + divide per call.
///
/// Caller invariant: `tf > 0` (callers gate on posting list membership)
/// and `dl_norm_k1 > 0` (precomputed positive at reader open, since
/// `K1 > 0` and `1 - B + B * dl/avgdl > 0` for any non-negative dl).
/// So the denominator is always positive.
#[inline(always)]
pub fn score_with_dl_norm_k1(idf_x_k1p1: f32, tf: u32, dl_norm_k1: f32) -> f32 {
    let tf = tf as f32;
    idf_x_k1p1 * tf / (tf + dl_norm_k1)
}

/// Score four cursors at the same doc in one SIMD operation. Pad
/// unused lanes with `idf_x_k1p1 = 0` and `tf = 0` (yielding 0
/// contribution; division by `dl_norm_k1` is finite). Returns the
/// horizontal sum of the four lanes — the doc's combined score.
///
/// `idfs_x_k1p1[i] = cursors[i].idf * (K1 + 1)` is precomputed at
/// cursor build, so this fits one multiply + add + divide per lane.
///
/// Used by the multi-term scoring path when 3-4 cursors are at the
/// same doc; saves the function-call overhead and lets the CPU
/// pipeline four divisions in parallel (the dominant cost in the
/// scalar `score`).
#[inline(always)]
pub fn score_simd_x4(
    idfs_x_k1p1: [f32; SCORE_SIMD_LANES],
    tfs: [f32; SCORE_SIMD_LANES],
    dl_norm_k1: f32,
) -> f32 {
    let idf_v = f32x4::from(idfs_x_k1p1);
    let tf_v = f32x4::from(tfs);
    let denom = tf_v + f32x4::splat(dl_norm_k1);
    let num = idf_v * tf_v;
    let scores = num / denom;
    scores.reduce_add()
}

/// Per-block upper bound on BM25 contribution. Used by BlockMaxWAND:
/// if this upper bound for the block can't beat the current top-k
/// threshold, the entire block of postings can be skipped.
///
/// The bound is achieved at `tf = max_tf` (highest possible numerator)
/// and `dl = min_dl` (smallest length-norm denominator).
#[inline]
pub fn block_upper_bound(idf_t: f32, max_tf: u32, min_dl: u32, avgdl: f32) -> f32 {
    score(idf_t, max_tf, min_dl, avgdl)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `f32` near-equality with a small absolute tolerance.
    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // --- idf -------------------------------------------------------------

    #[test]
    fn idf_is_non_negative_for_all_valid_inputs() {
        // Sweep representative (N, df) pairs.
        let cases = [
            (1u64, 0u64),
            (1, 1),
            (10, 0),
            (10, 1),
            (10, 5),
            (10, 10),
            (1_000_000, 0),
            (1_000_000, 1),
            (1_000_000, 500_000),
            (1_000_000, 1_000_000),
        ];
        for (n, df) in cases {
            let i = idf(n, df);
            assert!(i >= 0.0, "idf({n},{df}) = {i} should be >= 0");
            assert!(i.is_finite(), "idf({n},{df}) = {i} should be finite");
        }
    }

    #[test]
    fn idf_is_monotonic_in_df() {
        // For fixed N, smaller df → larger idf. This is the rare-terms-
        // matter property — the whole point of IDF.
        let n = 1_000_000u64;
        let dfs = [1u64, 10, 100, 1_000, 10_000, 100_000, 500_000];
        let mut prev = f32::INFINITY;
        for df in dfs {
            let cur = idf(n, df);
            assert!(
                cur < prev,
                "idf at df={df} ({cur}) should be < idf at smaller df ({prev})"
            );
            prev = cur;
        }
    }

    #[test]
    fn idf_reaches_zero_at_full_corpus() {
        // df == N → log(1 + 0.5 / (N + 0.5)) → small positive but
        // strictly above 0; check it's small.
        let i = idf(1_000_000, 1_000_000);
        assert!(i > 0.0 && i < 1e-5, "idf at df=N ({i}) should be ≈ 0+");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "df")]
    fn idf_debug_panics_on_df_greater_than_n_docs() {
        let _ = idf(10, 11);
    }

    // --- score ----------------------------------------------------------

    #[test]
    fn score_is_non_negative() {
        // Sweep across realistic (idf, tf, dl, avgdl) inputs.
        let i = idf(1_000_000, 1_000);
        for tf in [1, 2, 5, 10, 100] {
            for dl in [1, 10, 100, 1_000, 10_000] {
                for avgdl in [10.0, 100.0, 1_000.0] {
                    let s = score(i, tf, dl, avgdl);
                    assert!(
                        s >= 0.0,
                        "score(i={i}, tf={tf}, dl={dl}, avgdl={avgdl}) = {s}"
                    );
                    assert!(s.is_finite());
                }
            }
        }
    }

    #[test]
    fn score_grows_with_tf() {
        // Holding everything else fixed, more occurrences of the query
        // term in this doc should increase the score.
        let i = idf(1_000_000, 100);
        let s1 = score(i, 1, 200, 200.0);
        let s2 = score(i, 5, 200, 200.0);
        let s3 = score(i, 100, 200, 200.0);
        assert!(s1 < s2 && s2 < s3);
    }

    #[test]
    fn score_saturates_with_tf() {
        // BM25's whole point: tf saturation. score(tf=1000) is not
        // ~1000× score(tf=1); the gap shrinks as tf grows.
        let i = idf(1_000_000, 100);
        let s_low = score(i, 1, 200, 200.0);
        let s_mid = score(i, 10, 200, 200.0);
        let s_high = score(i, 1_000, 200, 200.0);

        // Linear scaling would predict s_high ≈ 100 × s_mid.
        // Saturating scaling predicts s_high < 2 × s_mid (rough bound).
        assert!(s_high > s_mid && s_mid > s_low);
        assert!(
            s_high < 2.0 * s_mid,
            "tf should saturate, not scale linearly"
        );
    }

    #[test]
    fn score_decreases_with_doc_length() {
        // Longer docs should score lower for the same (term, tf).
        let i = idf(1_000_000, 100);
        let s_short = score(i, 3, 50, 200.0);
        let s_long = score(i, 3, 800, 200.0);
        assert!(s_short > s_long);
    }

    #[test]
    fn score_at_avgdl_uses_unit_norm() {
        // When dl == avgdl, the length-norm factor is exactly 1.
        // Then score reduces to: idf * tf * (k1+1) / (tf + k1).
        let i = 2.0_f32;
        let tf = 5;
        let avgdl = 200.0;
        let dl = 200;
        let expected = i * (tf as f32) * (K1 + 1.0) / ((tf as f32) + K1);
        let actual = score(i, tf, dl, avgdl);
        assert!(
            approx(actual, expected, 1e-5),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn score_handles_degenerate_avgdl_zero() {
        // Defensive: avgdl=0 must not panic or NaN.
        let s = score(1.0, 1, 100, 0.0);
        assert!(s.is_finite());
        assert!(s >= 0.0);
    }

    #[test]
    fn score_at_b_zero_drops_length_norm() {
        // Reference test using a manual computation with B=0:
        //   norm = 1; score = idf * tf * (k1+1) / (tf + k1)
        // We can't easily plug in B=0 without changing the constant,
        // but we can verify the formula at dl == avgdl directly (which
        // gives norm=1 regardless of B). Done in score_at_avgdl_uses_unit_norm.
        // This test instead verifies that a small dl drives norm < 1
        // and therefore score *up* relative to dl=avgdl.
        let i = 2.0_f32;
        let s_at_avgdl = score(i, 5, 200, 200.0);
        let s_short = score(i, 5, 1, 200.0);
        assert!(s_short > s_at_avgdl);
    }

    #[test]
    fn score_at_b_one_extreme() {
        // At dl == 0 (extreme short doc), norm = 1 - b + 0 = 0.25
        // (with default b=0.75). Score should be max for the (idf, tf)
        // shape — strictly larger than any positive-length variant.
        let i = 2.0_f32;
        let s_zero_dl = score(i, 5, 0, 200.0);
        let s_one_dl = score(i, 5, 1, 200.0);
        assert!(s_zero_dl > s_one_dl);
    }

    // --- block_upper_bound ----------------------------------------------

    #[test]
    fn upper_bound_at_extreme_inputs_matches_score() {
        // block_upper_bound(idf, max_tf, min_dl, avgdl) is just
        // score(idf, max_tf, min_dl, avgdl). Verify identity.
        let i = idf(1_000_000, 1_000);
        for max_tf in [1, 5, 50] {
            for min_dl in [1, 100, 1_000] {
                for avgdl in [50.0, 500.0] {
                    let ub = block_upper_bound(i, max_tf, min_dl, avgdl);
                    let s = score(i, max_tf, min_dl, avgdl);
                    assert!(approx(ub, s, 1e-6));
                }
            }
        }
    }

    #[test]
    fn upper_bound_is_real_upper_bound() {
        // For any (tf, dl) in the block (tf ≤ max_tf, dl ≥ min_dl),
        // the upper bound must dominate the actual score.
        let i = idf(1_000_000, 1_000);
        let max_tf = 10;
        let min_dl = 50;
        let avgdl = 200.0;
        let ub = block_upper_bound(i, max_tf, min_dl, avgdl);

        for tf in 1..=max_tf {
            for dl in min_dl..=(min_dl * 5) {
                let s = score(i, tf, dl, avgdl);
                assert!(
                    s <= ub + 1e-6,
                    "score(tf={tf}, dl={dl}) = {s} should be ≤ ub {ub}"
                );
            }
        }
    }

    // --- constant sanity ------------------------------------------------

    #[test]
    fn lucene_defaults_match() {
        // Belt and braces — if anyone changes K1 or B, BM25 results
        // shift across the entire codebase. Lock the values at test
        // time.
        assert!(approx(K1, 1.2, 1e-6));
        assert!(approx(B, 0.75, 1e-6));
    }

    // --- SIMD parity ----------------------------------------------------

    #[test]
    fn simd_x4_equals_scalar_sum() {
        // Summing four scalar `score()` calls must agree with the
        // four-lane `score_simd_x4` to within rounding error.
        let dl = 200u32;
        let avgdl = 200.0;
        let k1_norm = K1 * (1.0 - B + B * dl as f32 / avgdl);
        let triples: [(f32, u32); 4] = [(1.5, 1), (1.7, 2), (2.0, 1), (1.2, 3)];
        let scalar: f32 = triples
            .iter()
            .map(|(idf, tf)| score(*idf, *tf, dl, avgdl))
            .sum();
        let idfs_x_k1p1 = [
            triples[0].0 * (K1 + 1.0),
            triples[1].0 * (K1 + 1.0),
            triples[2].0 * (K1 + 1.0),
            triples[3].0 * (K1 + 1.0),
        ];
        let tfs = [
            triples[0].1 as f32,
            triples[1].1 as f32,
            triples[2].1 as f32,
            triples[3].1 as f32,
        ];
        let simd = score_simd_x4(idfs_x_k1p1, tfs, k1_norm);
        assert!((scalar - simd).abs() < 1e-4, "simd={simd} scalar={scalar}");
    }
}
