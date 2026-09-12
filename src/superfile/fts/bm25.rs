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
//!                         = tf / ( tf + k1 * norm(dl, avgdl) )
//!
//!   score(idf, tf, dl, avgdl)
//!                         = idf * tf_factor(tf, dl, avgdl)
//! ```
//!
//! The textbook statement of BM25 carries a `(k1 + 1)` in the numerator
//! of `tf_factor`. It is a constant multiplier on every score a query
//! produces, so it cannot change a ranking, and it is conventionally
//! omitted — keeping it would make every score a fixed 2.2× (at the
//! default `k1`) of what the same parameters produce elsewhere, which
//! is invisible to anyone reading the order and wrong for anyone
//! reading the number: a score threshold, a weighted fusion against
//! vector distances, a table set beside another engine's.
//!
//! `idf(N, df)` is monotonic in `df` (smaller `df` → larger `idf`); always
//! non-negative because we use the +0.5 / +0.5 form ("BM25+1") which keeps
//! the log argument ≥ 1.

use wide::f32x4;

use crate::superfile::format;

/// Standard BM25 default `k1` — term-frequency saturation parameter.
pub const K1: f32 = 1.2;

/// Standard BM25 default `b` — length-normalization parameter.
pub const B: f32 = 0.75;

/// A column's BM25 similarity parameters.
///
/// [`Default`] is the standard pair (`k1 = 1.2`, `b = 0.75`) — the values every
/// superfile written before the parameters were recordable was built
/// with, and the values a column that declares nothing still uses. That
/// meaning is frozen: a file whose column entry carries no parameters
/// can only have been built with this pair, so the default here must
/// never track a change to what the *API* recommends.
///
/// `#[non_exhaustive]`: build with [`Bm25Params::new`] so a further
/// similarity parameter can be added without breaking callers.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Bm25Params {
    /// Term-frequency saturation. Must be `> 0` and finite.
    pub k1: f32,
    /// Length normalization, in `[0, 1]`. `0` disables it entirely.
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: K1, b: B }
    }
}

impl Bm25Params {
    /// The standard pair — `k1 = 1.2`, `b = 0.75`.
    pub const STANDARD: Self = Self { k1: K1, b: B };

    /// Unvalidated constructor. Callers that accept these from a user
    /// validate first (`SupertableOptions::new` for a declared column,
    /// the search options for an override) so the error names the
    /// column or argument at fault.
    pub const fn new(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }

    /// `k1 · (1 − b + b·dl/avgdl)` — the per-doc length normalizer the
    /// scorer divides by, precomputed per length bucket in the reader's
    /// norm table. `avgdl <= 0` (an empty column) yields `k1`, the
    /// unit-norm value; such a column is never scored.
    #[inline]
    pub(crate) fn dl_norm_k1(&self, dl: u32, avgdl: f32) -> f32 {
        let norm = if avgdl > 0.0 {
            1.0 - self.b + self.b * (dl as f32) / avgdl
        } else {
            1.0
        };
        self.k1 * norm
    }
}

/// Plus-half IDF smoothing term. Added to both the numerator and
/// denominator of the IDF log argument so it stays ≥ 1 (hence
/// `idf >= 0`) for every valid `(N, df)` — the "BM25+1" form.
const IDF_SMOOTHING: f64 = 0.5;

/// SIMD lane count for the four-wide BM25 scorers ([`f32x4`]). The
/// multi-term path scores this many cursors at one doc per call; the
/// windowed-union path scores this many docs from one cursor.
pub(super) const SCORE_SIMD_LANES: usize = 4;

/// Document lengths below this are quantized to one byte with no loss;
/// at or above it, an 8-bit floating representation kicks in. Equal to
/// `2^(LEN_QUANT_MANTISSA_BITS + 1)` so the exact region and the
/// float region join without a monotonicity gap.
const LEN_QUANT_EXACT_MAX: u32 = 1 << (LEN_QUANT_MANTISSA_BITS + 1);

/// Mantissa bits kept when a length is quantized into the float region.
/// Three mantissa bits (plus the implicit leading 1) cap the decoded
/// length's relative error at `2^-MANTISSA_BITS` = 12.5% (the codec
/// truncates rather than rounds, which is what makes it idempotent) —
/// and BM25 only feels a fraction of that, since the length enters the
/// norm as `1 - b + b·dl/avgdl` (the `1 - b` term carries no error). In
/// exchange the resident length-norm table shrinks 4× (one byte per doc
/// instead of an `f32`), which is what keeps it cache-resident at scale.
const LEN_QUANT_MANTISSA_BITS: u32 = 3;

/// Quantize a document length into one byte via an 8-bit float:
/// lengths `< LEN_QUANT_EXACT_MAX` are stored exactly; larger lengths
/// keep the top `LEN_QUANT_MANTISSA_BITS` bits below the leading 1 plus
/// an exponent. Monotonic non-decreasing in `len`, so it never reorders
/// two docs whose true lengths differ by more than one bucket. Inverse
/// of [`dequantize_len`].
#[inline]
pub(super) fn quantize_len(len: u32) -> u8 {
    if len < LEN_QUANT_EXACT_MAX {
        len as u8
    } else {
        // floor(log2 len); len >= LEN_QUANT_EXACT_MAX >= 8 ⇒ bits >= 3.
        let bits = u32::BITS - 1 - len.leading_zeros();
        let mantissa = (len >> (bits - LEN_QUANT_MANTISSA_BITS)) & 0x07;
        // Exponent field is offset by 1 so the smallest float-region
        // code sits just above the exact region (no gap, stays monotone).
        let exponent = bits - LEN_QUANT_MANTISSA_BITS + 1;
        ((exponent << LEN_QUANT_MANTISSA_BITS) | mantissa) as u8
    }
}

/// Decode a byte produced by [`quantize_len`] back to a representative
/// document length. The result is the exact length for small docs and a
/// bucket representative (within ~6.25%) for larger ones.
#[inline]
pub(super) fn dequantize_len(b: u8) -> u32 {
    let i = b as u32;
    if i < LEN_QUANT_EXACT_MAX {
        i
    } else {
        let mantissa = i & 0x07;
        let exponent = (i >> LEN_QUANT_MANTISSA_BITS) - 1;
        (mantissa | 0x08) << exponent
    }
}

/// An average document length as the doc-lengths directory stores it:
/// rounded to [`format::fts::AVGDL_FIXED_POINT_SCALE`]. The single source
/// for that fixed point — the builder writes this value and bakes bounds
/// at [`stored_avgdl`], the reader recovers the same [`stored_avgdl`], so
/// the two agree to the bit.
#[inline]
pub fn avgdl_x1000(avgdl: f32) -> u32 {
    (avgdl * format::fts::AVGDL_FIXED_POINT_SCALE)
        .round()
        .clamp(0.0, u32::MAX as f32) as u32
}

/// The average document length scoring actually uses for a column whose
/// exact average is `avgdl`: what [`avgdl_x1000`] round-trips to.
#[inline]
pub fn stored_avgdl(avgdl: f32) -> f32 {
    avgdl_x1000(avgdl) as f32 / format::fts::AVGDL_FIXED_POINT_SCALE
}

/// The document length the scorer actually sees for a doc of `len`
/// tokens: the one-byte stored representative ([`quantize_len`] then
/// [`dequantize_len`]). Exact below 16 tokens, truncated downward by up to
/// one bucket (≤ 12.5%) above. A reference scorer that wants to reproduce
/// the engine's scores — rather than textbook BM25 on exact lengths — feeds
/// this length into its normalization.
#[inline]
pub fn stored_len(len: u32) -> u32 {
    dequantize_len(quantize_len(len))
}

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
/// `tf`     — term frequency in this document, this column.
/// `dl`     — this document's length in this column (in tokens).
/// `avgdl`  — average document length across the superfile, this column.
/// `params` — the column's similarity parameters.
#[inline(always)]
pub fn score(idf_t: f32, tf: u32, dl: u32, avgdl: f32, params: Bm25Params) -> f32 {
    let tf = tf as f32;
    // avgdl is precomputed at build time and stored in the doc-lengths
    // directory; if a superfile has zero docs we wouldn't be calling this
    // function, but guard anyway against a divide-by-zero on degenerate
    // input.
    let denom = tf + params.dl_norm_k1(dl, avgdl);
    if denom == 0.0 {
        // tf=0 should never reach this function (callers gate on
        // posting list membership), but stay defensive.
        return 0.0;
    }
    idf_t * tf / denom
}

/// BM25 score using a precomputed `dl_norm_k1 = K1 * (1 - B + B * dl/avgdl)`
/// and `idf_weight = idf` (folding in any query-term-frequency weight).
///
/// Both `dl_norm_k1` (per doc) and `idf_weight` (per cursor) are
/// computed once at reader open / cursor build. The hot inner loop
/// drops to a single multiply + add + divide per call.
///
/// Caller invariant: `tf > 0` (callers gate on posting list membership)
/// and `dl_norm_k1 > 0` (precomputed positive at reader open, since
/// `K1 > 0` and `1 - B + B * dl/avgdl > 0` for any non-negative dl).
/// So the denominator is always positive.
#[inline(always)]
pub fn score_with_dl_norm_k1(idf_weight: f32, tf: u32, dl_norm_k1: f32) -> f32 {
    let tf = tf as f32;
    idf_weight * tf / (tf + dl_norm_k1)
}

/// Score four cursors at the same doc in one SIMD operation. Pad
/// unused lanes with `idf_weight = 0` and `tf = 0` (yielding 0
/// contribution; division by `dl_norm_k1` is finite). Returns the
/// horizontal sum of the four lanes — the doc's combined score.
///
/// `idfs_weight[i] = cursors[i].idf` is precomputed at
/// cursor build, so this fits one multiply + add + divide per lane.
///
/// Used by the multi-term scoring path when 3-4 cursors are at the
/// same doc; saves the function-call overhead and lets the CPU
/// pipeline four divisions in parallel (the dominant cost in the
/// scalar `score`).
#[inline(always)]
pub fn score_simd_x4(
    idfs_weight: [f32; SCORE_SIMD_LANES],
    tfs: [f32; SCORE_SIMD_LANES],
    dl_norm_k1: f32,
) -> f32 {
    let idf_v = f32x4::from(idfs_weight);
    let tf_v = f32x4::from(tfs);
    let denom = tf_v + f32x4::splat(dl_norm_k1);
    let num = idf_v * tf_v;
    let scores = num / denom;
    scores.reduce_add()
}

/// Score one cursor at four documents in one SIMD operation. Each
/// document has its own term frequency and length normalization; the
/// cursor's precomputed `idf` is shared across all lanes.
/// Returns the four independent contributions without reducing them.
/// Callers pass posting-list term frequencies (`tf > 0`); unlike
/// [`score_simd_x4`], this path does not use zero-padded lanes.
#[inline(always)]
pub(super) fn score_one_term_x4(
    idf_weight: f32,
    tfs: [u32; SCORE_SIMD_LANES],
    dl_norm_k1: [f32; SCORE_SIMD_LANES],
) -> [f32; SCORE_SIMD_LANES] {
    let tf_v = f32x4::from([tfs[0] as f32, tfs[1] as f32, tfs[2] as f32, tfs[3] as f32]);
    let scores = f32x4::splat(idf_weight) * tf_v / (tf_v + f32x4::from(dl_norm_k1));
    scores.to_array()
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
                    let s = score(i, tf, dl, avgdl, Bm25Params::STANDARD);
                    assert!(
                        s >= 0.0,
                        "score(i={i}, tf={tf}, dl={dl}, avgdl={avgdl}, Bm25Params::STANDARD) = {s}"
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
        let s1 = score(i, 1, 200, 200.0, Bm25Params::STANDARD);
        let s2 = score(i, 5, 200, 200.0, Bm25Params::STANDARD);
        let s3 = score(i, 100, 200, 200.0, Bm25Params::STANDARD);
        assert!(s1 < s2 && s2 < s3);
    }

    #[test]
    fn score_saturates_with_tf() {
        // BM25's whole point: tf saturation. score(tf=1000) is not
        // ~1000× score(tf=1); the gap shrinks as tf grows.
        let i = idf(1_000_000, 100);
        let s_low = score(i, 1, 200, 200.0, Bm25Params::STANDARD);
        let s_mid = score(i, 10, 200, 200.0, Bm25Params::STANDARD);
        let s_high = score(i, 1_000, 200, 200.0, Bm25Params::STANDARD);

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
        let s_short = score(i, 3, 50, 200.0, Bm25Params::STANDARD);
        let s_long = score(i, 3, 800, 200.0, Bm25Params::STANDARD);
        assert!(s_short > s_long);
    }

    #[test]
    fn score_at_avgdl_uses_unit_norm() {
        // When dl == avgdl, the length-norm factor is exactly 1.
        // Then score reduces to: idf * tf / (tf + k1).
        let i = 2.0_f32;
        let tf = 5;
        let avgdl = 200.0;
        let dl = 200;
        let expected = i * (tf as f32) / ((tf as f32) + K1);
        let actual = score(i, tf, dl, avgdl, Bm25Params::STANDARD);
        assert!(
            approx(actual, expected, 1e-5),
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn score_handles_degenerate_avgdl_zero() {
        // Defensive: avgdl=0 must not panic or NaN.
        let s = score(1.0, 1, 100, 0.0, Bm25Params::STANDARD);
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
        let s_at_avgdl = score(i, 5, 200, 200.0, Bm25Params::STANDARD);
        let s_short = score(i, 5, 1, 200.0, Bm25Params::STANDARD);
        assert!(s_short > s_at_avgdl);
    }

    #[test]
    fn score_at_b_one_extreme() {
        // At dl == 0 (extreme short doc), norm = 1 - b + 0 = 0.25
        // (with default b=0.75). Score should be max for the (idf, tf)
        // shape — strictly larger than any positive-length variant.
        let i = 2.0_f32;
        let s_zero_dl = score(i, 5, 0, 200.0, Bm25Params::STANDARD);
        let s_one_dl = score(i, 5, 1, 200.0, Bm25Params::STANDARD);
        assert!(s_zero_dl > s_one_dl);
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
            .map(|(idf, tf)| score(*idf, *tf, dl, avgdl, Bm25Params::STANDARD))
            .sum();
        let idfs_weight = [triples[0].0, triples[1].0, triples[2].0, triples[3].0];
        let tfs = [
            triples[0].1 as f32,
            triples[1].1 as f32,
            triples[2].1 as f32,
            triples[3].1 as f32,
        ];
        let simd = score_simd_x4(idfs_weight, tfs, k1_norm);
        assert!((scalar - simd).abs() < 1e-4, "simd={simd} scalar={scalar}");
    }

    #[test]
    fn one_term_simd_x4_equals_scalar_lanes() {
        let idf_weight = idf(1_000_000, 10_000);
        let tfs = [1, 2, 5, 9];
        let dl_norm_k1 = [0.4, 0.9, 1.2, 3.5];
        let simd = score_one_term_x4(idf_weight, tfs, dl_norm_k1);

        for lane in 0..SCORE_SIMD_LANES {
            let scalar = score_with_dl_norm_k1(idf_weight, tfs[lane], dl_norm_k1[lane]);
            assert!(
                approx(simd[lane], scalar, 1e-6),
                "lane {lane}: simd={} scalar={scalar}",
                simd[lane]
            );
        }
    }

    // --- stored average -------------------------------------------------

    #[test]
    fn stored_average_rounds_to_the_directory_fixed_point_and_round_trips() {
        assert_eq!(avgdl_x1000(1.002), 1002);
        assert_eq!(avgdl_x1000(1.0024), 1002);
        assert_eq!(avgdl_x1000(1.0026), 1003);
        assert_eq!(avgdl_x1000(0.0), 0);
        assert_eq!(avgdl_x1000(-1.0), 0, "a negative average clamps to zero");
        for avgdl in [1.002f32, 3.2, 89.0, 292.4, 4_000_000.0] {
            let stored = stored_avgdl(avgdl);
            assert_eq!(
                avgdl_x1000(stored),
                avgdl_x1000(avgdl),
                "{avgdl}: stored value must re-encode to the same fixed point"
            );
            assert_eq!(stored_avgdl(stored), stored, "{avgdl}: idempotent");
            assert!(
                (stored - avgdl).abs()
                    <= 0.5 / format::fts::AVGDL_FIXED_POINT_SCALE + avgdl * f32::EPSILON
            );
        }
    }

    // --- length quantization --------------------------------------------

    #[test]
    fn len_quant_is_exact_below_threshold() {
        // Small documents (the common case) round-trip with zero error.
        for len in 0..LEN_QUANT_EXACT_MAX {
            let b = quantize_len(len);
            assert_eq!(dequantize_len(b), len, "len {len} not exact");
        }
    }

    #[test]
    fn len_quant_is_monotonic_non_decreasing() {
        // A larger true length never encodes to a smaller byte, so
        // quantization can't invert the length ordering of two docs.
        let mut prev = 0u8;
        for len in 0..100_000u32 {
            let b = quantize_len(len);
            assert!(b >= prev, "len {len}: byte {b} < previous {prev}");
            prev = b;
        }
    }

    #[test]
    fn len_quant_round_trip_relative_error_is_bounded() {
        // Truncating codec: decoded length is within 2^-MANTISSA_BITS
        // (12.5% at 3 mantissa bits) of the input, always rounding down.
        let max_rel = 2f64.powi(-(LEN_QUANT_MANTISSA_BITS as i32));
        for len in LEN_QUANT_EXACT_MAX..1_000_000u32 {
            let decoded = dequantize_len(quantize_len(len));
            let rel = (len as f64 - decoded as f64).abs() / len as f64;
            assert!(
                rel <= max_rel,
                "len {len}: decoded {decoded}, rel err {rel} > {max_rel}"
            );
        }
    }

    #[test]
    fn len_quant_is_idempotent_over_realistic_lengths() {
        // Decoding a quantized length and re-quantizing it yields the
        // same byte — the byte↔length map is stable, so a doc's bucket
        // never drifts across rebuilds. (Checked over lengths a real
        // corpus can produce; bytes no length maps to are irrelevant.)
        for len in 0..2_000_000u32 {
            let b = quantize_len(len);
            assert_eq!(
                quantize_len(dequantize_len(b)),
                b,
                "len {len} → byte {b} not idempotent"
            );
        }
    }
}
