// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! 1-bit RaBitQ-style sign quantizer with SIMD estimator.
//!
//! Each rotated f32 vector becomes one bit per dimension: 1 if positive,
//! 0 if non-positive. The estimator dot-products the rotated query
//! against the codebook of `±1` signs implied by the bits — yielding
//! an unbiased estimate of `<R·query, R·doc>` (which equals
//! `<query, doc>` because `R` is orthogonal).
//!
//! The `sign_table` is a precomputed lookup of all 256 byte values to
//! their 8-lane `±1.0` expansions. SIMD-friendly: each input byte
//! becomes one `f32x8` register load; multiplication against the
//! query lanes is one fused-multiply-add.
//!
//! ## AVX-512 fast path
//!
//! On hosts with AVX-512F, [`BitQuantizer::estimate_dot_rotated_with_total`]
//! takes a precomputed `q_total = Σ_d q_rot[d]` and computes the
//! estimate as `2·pos_sum − q_total`, where `pos_sum =
//! Σ_{bit_d = 1} q_rot[d]`. The masked sum is implemented with
//! `_mm512_mask_add_ps` keyed by the doc's bit pattern: 16 query
//! lanes per iteration, one instruction per masked add. This
//! eliminates the 8 KB sign-table look-up from the inner loop
//! (one 4 KB LLC saving per 16 lanes scanned) and reduces the
//! per-iteration work to `loadu_ps + mask_add` (two µops on
//! Sapphire Rapids).
//!
//! The default [`BitQuantizer::estimate_dot_rotated`] entry point
//! is unchanged: it computes `q_total` inline (one extra dim-pass
//! per call) and dispatches. The hot per-candidate IVF-scan loop
//! in `superfile::vector::reader::score_cluster_codes` calls the
//! `_with_total` variant directly with a per-query precomputed
//! `q_total` so the per-candidate cost stays on the fast path.
//!
//! See `docs/architecture/superfile.md` (Vector index algorithm
//! subsection) for the full RaBitQ rationale and recall trade-offs.

use wide::f32x8;

#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::avx512_enabled;

/// Number of sign bits packed into one code byte (1-bit RaBitQ packs
/// one sign per dimension, eight per byte). One `f32x8` SIMD block
/// also covers exactly this many dimensions, so the same constant
/// drives both the bit-packing and the per-block SIMD stride.
const BITS_PER_CODE_BYTE: usize = 8;

/// Number of distinct byte values a code byte can take (`2^8`). The
/// sign table holds one [`BITS_PER_CODE_BYTE`]-wide `±1` expansion
/// per pattern, so it is `SIGN_TABLE_BYTE_PATTERNS * BITS_PER_CODE_BYTE`
/// floats.
const SIGN_TABLE_BYTE_PATTERNS: usize = 256;

/// RaBitQ codebook sign for a set bit.
const RABITQ_POSITIVE_SIGN: f32 = 1.0;
/// RaBitQ codebook sign for a clear bit.
const RABITQ_NEGATIVE_SIGN: f32 = -1.0;

/// Coefficient on `pos_sum` in the RaBitQ dot identity
/// `dot = RABITQ_DOT_POS_COEFF·pos_sum − q_total`, where
/// `pos_sum = Σ_{bit_d = 1} q_rot[d]`.
// Referenced only by the x86-gated AVX-512 estimator; dead on other targets.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const RABITQ_DOT_POS_COEFF: f32 = 2.0;

/// Lane count of an AVX-512 f32 register (512-bit / 32-bit), the
/// dims-per-iteration of the masked-add estimator.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX512_F32_LANES: usize = 16;

/// 1-bit quantizer + estimator for vectors of fixed dimension `dim`.
/// Construct once per column at index-build time; reuse for both
/// encoding (build-side) and dot-estimation (query-side).
#[derive(Debug, Clone)]
pub struct BitQuantizer {
    pub dim: usize,
    sign_table: Box<[f32; SIGN_TABLE_BYTE_PATTERNS * BITS_PER_CODE_BYTE]>,
}

impl BitQuantizer {
    /// Build the sign lookup table for vectors of dimension `dim`.
    /// Cost: `256 * 8 * 4 = 8 KB` heap, computed once.
    pub fn new(dim: usize) -> Self {
        let mut table = Box::new([0.0f32; SIGN_TABLE_BYTE_PATTERNS * BITS_PER_CODE_BYTE]);
        for b in 0..SIGN_TABLE_BYTE_PATTERNS {
            for bit in 0..BITS_PER_CODE_BYTE {
                let set = (b >> bit) & 1;
                table[b * BITS_PER_CODE_BYTE + bit] = if set == 1 {
                    RABITQ_POSITIVE_SIGN
                } else {
                    RABITQ_NEGATIVE_SIGN
                };
            }
        }
        Self {
            dim,
            sign_table: table,
        }
    }

    /// Number of bytes required to hold one encoded vector.
    /// `ceil(dim / 8)`.
    #[inline]
    pub fn code_bytes(&self) -> usize {
        self.dim.div_ceil(BITS_PER_CODE_BYTE)
    }

    /// Encode one already-rotated f32 vector into bits. `out` must be
    /// exactly `code_bytes()` long.
    ///
    /// Hot dense path at build time: every input vector is bit-packed
    /// here exactly once. The 8-lane SIMD loop processes one output
    /// byte per iteration via `f32x8::simd_gt(ZERO).to_bitmask()` —
    /// lowers to one `_mm256_cmp_ps` + one `_mm256_movemask_ps` on
    /// AVX2 hosts and falls back to two `_mm_cmpgt_ps` + two
    /// `_mm_movemask_ps` (combined) on SSE2 hosts via `wide`'s
    /// `pick!` dispatch. Tail dimensions (`dim % 8 != 0`) go through
    /// a scalar bit-set loop into the partial last byte.
    #[inline]
    pub fn encode_rotated_into(&self, rotated: &[f32], out: &mut [u8]) {
        debug_assert_eq!(rotated.len(), self.dim);
        debug_assert_eq!(out.len(), self.code_bytes());
        let zero = f32x8::ZERO;
        let full_bytes = self.dim / BITS_PER_CODE_BYTE;
        for byte_idx in 0..full_bytes {
            let lane: [f32; BITS_PER_CODE_BYTE] = rotated
                [byte_idx * BITS_PER_CODE_BYTE..byte_idx * BITS_PER_CODE_BYTE + BITS_PER_CODE_BYTE]
                .try_into()
                .expect("slice [byte_idx*8..byte_idx*8+8] has length 8");
            let v = f32x8::from(lane);
            // `to_bitmask` returns one u32 whose low 8 bits are the
            // sign/comparison bits for each lane, in lane-order — bit
            // 0 = lane 0 > 0.0, bit 7 = lane 7 > 0.0. Exactly the
            // bit-order the scalar reference loop produces.
            out[byte_idx] = v.simd_gt(zero).to_bitmask() as u8;
        }
        let tail_start = full_bytes * BITS_PER_CODE_BYTE;
        if tail_start < self.dim {
            let mut byte: u8 = 0;
            for i in 0..(self.dim - tail_start) {
                if rotated[tail_start + i] > 0.0 {
                    byte |= 1u8 << i;
                }
            }
            out[full_bytes] = byte;
        }
    }

    /// Estimate `<q_rot, doc_rot>` from the bit-encoded `code` of
    /// `doc_rot`. The result is an unbiased estimator of the rotated
    /// dot product (which equals the un-rotated dot product because
    /// `R` is orthogonal). Variance bounds depend on `dim` — see the
    /// RaBitQ paper for the details.
    ///
    /// Computes `q_total = Σ_d q_rot[d]` inline before dispatching;
    /// hot loops scoring many docs against the same query should
    /// instead call [`estimate_dot_rotated_with_total`] with a
    /// per-query precomputed `q_total` to amortize the dim-pass.
    ///
    /// [`estimate_dot_rotated_with_total`]: BitQuantizer::estimate_dot_rotated_with_total
    #[inline]
    pub fn estimate_dot_rotated(&self, q_rot: &[f32], code: &[u8]) -> f32 {
        let q_total: f32 = q_rot.iter().sum();
        self.estimate_dot_rotated_with_total(q_rot, code, q_total)
    }

    /// Like [`estimate_dot_rotated`] but takes a precomputed
    /// `q_total = Σ_d q_rot[d]`. Use this in per-candidate hot loops
    /// where the same query is scored against many docs — the AVX-512
    /// path uses the algebraic identity
    /// `dot = Σ_d q_rot[d] · (2·bit_d − 1) = 2·pos_sum − q_total`
    /// (where `pos_sum = Σ_{bit_d = 1} q_rot[d]`), and the masked
    /// `pos_sum` computation is the cheap part — the `q_total`
    /// term is purely per-query and shouldn't be recomputed per
    /// candidate.
    ///
    /// On non-AVX-512 hosts this falls through to the existing
    /// 256-bit `wide::f32x8` kernel via the sign-table lookup —
    /// `q_total` is ignored in that path. So the result is exactly
    /// the same numeric value regardless of which path runs (modulo
    /// f32 add-order divergence well below the recall test
    /// tolerances).
    #[inline]
    pub fn estimate_dot_rotated_with_total(&self, q_rot: &[f32], code: &[u8], q_total: f32) -> f32 {
        debug_assert_eq!(q_rot.len(), self.dim);
        debug_assert_eq!(code.len(), self.code_bytes());

        #[cfg(target_arch = "x86_64")]
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` which requires
            // `avx512f`; `_mm512_mask_add_ps` is part of AVX-512F.
            return unsafe { estimate_dot_rotated_avx512(q_rot, code, q_total, self.dim) };
        }
        let _ = q_total; // ignored on the wide fallback
        estimate_dot_rotated_wide(&self.sign_table, q_rot, code, self.dim)
    }
}

/// Portable `wide::f32x8` (256-bit) RaBitQ estimator via the 8KB
/// sign-table lookup. The kernel that has shipped since the
/// quantizer existed; remains the universal fallback on every
/// non-AVX-512 host.
#[inline]
fn estimate_dot_rotated_wide(
    sign_table: &[f32; SIGN_TABLE_BYTE_PATTERNS * BITS_PER_CODE_BYTE],
    q_rot: &[f32],
    code: &[u8],
    dim: usize,
) -> f32 {
    let full_bytes = dim / BITS_PER_CODE_BYTE;
    let mut acc = f32x8::ZERO;
    for byte_idx in 0..full_bytes {
        let b = code[byte_idx] as usize;
        let signs_slice: &[f32; BITS_PER_CODE_BYTE] = (&sign_table
            [b * BITS_PER_CODE_BYTE..b * BITS_PER_CODE_BYTE + BITS_PER_CODE_BYTE])
            .try_into()
            .expect("slice [b*8..b*8+8] has length 8");
        let q_slice: &[f32; BITS_PER_CODE_BYTE] = (&q_rot
            [byte_idx * BITS_PER_CODE_BYTE..byte_idx * BITS_PER_CODE_BYTE + BITS_PER_CODE_BYTE])
            .try_into()
            .expect("slice [byte_idx*8..byte_idx*8+8] has length 8");
        let signs = f32x8::from(*signs_slice);
        let q_block = f32x8::from(*q_slice);
        acc += q_block * signs;
    }
    let mut sum: f32 = acc.reduce_add();

    // Tail: dims [full_bytes*8 .. dim] handled scalar.
    let tail_start = full_bytes * BITS_PER_CODE_BYTE;
    if tail_start < dim {
        let byte = code[full_bytes] as usize;
        for i in 0..(dim - tail_start) {
            let bit = (byte >> i) & 1;
            let s = if bit == 1 {
                RABITQ_POSITIVE_SIGN
            } else {
                RABITQ_NEGATIVE_SIGN
            };
            sum += q_rot[tail_start + i] * s;
        }
    }
    sum
}

/// AVX-512 RaBitQ estimator. Mathematical identity:
///
/// ```text
/// dot = Σ_d q_rot[d] * (2·bit_d − 1)
///     = 2 * Σ_{bit_d = 1} q_rot[d]  −  Σ_d q_rot[d]
///     = 2·pos_sum − q_total
/// ```
///
/// `pos_sum` is computed with `_mm512_mask_add_ps` keyed by the
/// 16-bit doc mask formed from two consecutive code bytes: one
/// instruction adds 16 query lanes (or skips them) into the
/// accumulator, doing in 16 lanes what the wide kernel does in 8.
///
/// Eliminates the 8 KB sign-table lookup that dominated LLC
/// pressure on the IVF scan; no per-iteration table load means
/// the kernel is throughput-bound on `vmovups + vmaskz_addps`.
///
/// # Safety
///
/// Callers must ensure the target supports `avx512f`. The
/// `avx512_enabled()` gate guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn estimate_dot_rotated_avx512(q_rot: &[f32], code: &[u8], q_total: f32, dim: usize) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(q_rot.len(), dim);
    debug_assert_eq!(code.len(), dim.div_ceil(BITS_PER_CODE_BYTE));

    // SAFETY: each iteration reads 16 fp32s from `q_rot` (guarded
    // by `i + 16 <= dim`) and 2 bytes from `code` (guarded by
    // `i / 8 + 2 <= dim.div_ceil(8)` which `i + 16 <= dim` implies
    // when `dim` is a multiple of 8 and `i` is a multiple of 16).
    // `_mm512_loadu_ps` is unaligned.
    unsafe {
        let mut pos_sum = _mm512_setzero_ps();
        let mut i: usize = 0;
        // Process 16 dims per iteration. Each iteration consumes
        // exactly 2 code bytes (16 bits = 16 lanes).
        while i + AVX512_F32_LANES <= dim {
            let bits = u16::from_le_bytes([
                code[i / BITS_PER_CODE_BYTE],
                code[i / BITS_PER_CODE_BYTE + 1],
            ]);
            let q = _mm512_loadu_ps(q_rot.as_ptr().add(i));
            pos_sum = _mm512_mask_add_ps(pos_sum, bits, pos_sum, q);
            i += AVX512_F32_LANES;
        }
        let mut pos: f32 = _mm512_reduce_add_ps(pos_sum);

        // Tail of 8 lanes if `dim % 16 >= 8` — same shape as one
        // SIMD iteration but with 8 lanes via the 256-bit half-
        // register `__m256` and a `__mmask8` keyed by one code
        // byte. Lets us still avoid the scalar loop for the
        // common case of `dim % 8 == 0` and `dim % 16 == 8`
        // (e.g. dim = 24, 40, 56, ... — rare but cheap to be
        // correct about).
        if i + BITS_PER_CODE_BYTE <= dim {
            let bits = code[i / BITS_PER_CODE_BYTE];
            let q8 = _mm256_loadu_ps(q_rot.as_ptr().add(i));
            let masked = _mm256_maskz_mov_ps(bits, q8);
            // Horizontal sum of 8 fp32. AVX-512 lacks a 256-bit
            // reduce_add intrinsic on stable; fold via the
            // standard zero-extend-into-zmm trick: cast to zmm,
            // mask off the high lanes, reduce.
            let zext = _mm512_zextps256_ps512(masked);
            pos += _mm512_reduce_add_ps(zext);
            i += BITS_PER_CODE_BYTE;
        }
        // Scalar tail for `dim % 8 != 0`.
        while i < dim {
            let bit = ((code[i / BITS_PER_CODE_BYTE] >> (i % BITS_PER_CODE_BYTE)) & 1) != 0;
            if bit {
                pos += q_rot[i];
            }
            i += 1;
        }
        RABITQ_DOT_POS_COEFF * pos - q_total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // --- code_bytes ----------------------------------------------------

    #[test]
    fn code_bytes_for_byte_aligned_dims() {
        for &dim in &[8, 16, 32, 64, 128, 256, 384, 768, 1024] {
            assert_eq!(BitQuantizer::new(dim).code_bytes(), dim / 8);
        }
    }

    #[test]
    fn code_bytes_for_non_aligned_dims_rounds_up() {
        assert_eq!(BitQuantizer::new(1).code_bytes(), 1);
        assert_eq!(BitQuantizer::new(7).code_bytes(), 1);
        assert_eq!(BitQuantizer::new(9).code_bytes(), 2);
        assert_eq!(BitQuantizer::new(15).code_bytes(), 2);
        assert_eq!(BitQuantizer::new(17).code_bytes(), 3);
    }

    // --- encode --------------------------------------------------------

    #[test]
    fn encode_all_positive_sets_every_bit() {
        let q = BitQuantizer::new(8);
        let v = vec![1.0; 8];
        let mut out = vec![0u8; 1];
        q.encode_rotated_into(&v, &mut out);
        assert_eq!(out, vec![0xFF]);
    }

    #[test]
    fn encode_all_negative_clears_every_bit() {
        let q = BitQuantizer::new(8);
        let v = vec![-1.0; 8];
        let mut out = vec![0u8; 1];
        q.encode_rotated_into(&v, &mut out);
        assert_eq!(out, vec![0x00]);
    }

    #[test]
    fn encode_zero_is_negative() {
        // The contract: `> 0.0` sets the bit. Exactly zero stays cleared.
        let q = BitQuantizer::new(8);
        let v = vec![0.0; 8];
        let mut out = vec![0u8; 1];
        q.encode_rotated_into(&v, &mut out);
        assert_eq!(out, vec![0x00]);
    }

    #[test]
    fn encode_single_positive_dim_sets_one_bit() {
        let q = BitQuantizer::new(8);
        for i in 0..8 {
            let mut v = vec![-1.0; 8];
            v[i] = 1.0;
            let mut out = vec![0u8; 1];
            q.encode_rotated_into(&v, &mut out);
            assert_eq!(out, vec![1u8 << i], "dim {i}");
        }
    }

    #[test]
    fn encode_non_aligned_dim_uses_partial_byte() {
        // dim=12 → ceil(12/8) = 2 bytes; bits 0..12 used.
        let q = BitQuantizer::new(12);
        let mut v = vec![-1.0; 12];
        v[0] = 1.0;
        v[11] = 1.0;
        let mut out = vec![0u8; 2];
        q.encode_rotated_into(&v, &mut out);
        assert_eq!(out, vec![0x01, 0x08]); // bit 0 of byte 0 + bit 3 of byte 1
    }

    // --- estimate ------------------------------------------------------

    #[test]
    fn estimate_query_against_self_returns_l1_sum_of_query() {
        // If the doc encodes as the sign of the query (perfect
        // alignment) then estimate = Σ |q[i]|.
        let q = BitQuantizer::new(8);
        let q_rot = vec![3.0, -1.0, 2.0, -4.0, 5.0, -6.0, 7.0, -2.0];
        let mut code = vec![0u8; 1];
        q.encode_rotated_into(&q_rot, &mut code);
        let est = q.estimate_dot_rotated(&q_rot, &code);
        let expected: f32 = q_rot.iter().map(|x| x.abs()).sum();
        assert!(approx(est, expected, 1e-5));
    }

    #[test]
    fn estimate_query_against_opposite_returns_negative_sum() {
        // If the code encodes the OPPOSITE signs of the query, the
        // estimator sums all `-|q[i]|`.
        let q = BitQuantizer::new(8);
        let q_rot = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let neg = q_rot.iter().map(|&x| -x).collect::<Vec<_>>();
        let mut code = vec![0u8; 1];
        q.encode_rotated_into(&neg, &mut code);
        let est = q.estimate_dot_rotated(&q_rot, &code);
        let expected: f32 = -q_rot.iter().map(|x| x.abs()).sum::<f32>();
        assert!(approx(est, expected, 1e-5));
    }

    #[test]
    fn estimate_handles_tail_dim() {
        // dim = 12: 1 full byte + 4 tail bits.
        let q = BitQuantizer::new(12);
        let q_rot: Vec<f32> = (1..=12).map(|i| i as f32).collect();
        let mut code = vec![0u8; 2];
        q.encode_rotated_into(&q_rot, &mut code);
        let est = q.estimate_dot_rotated(&q_rot, &code);
        let expected: f32 = q_rot.iter().sum(); // all positive, all signs match
        assert!(approx(est, expected, 1e-5));
    }

    #[test]
    fn estimate_zero_query_yields_zero() {
        let q = BitQuantizer::new(16);
        let q_rot = vec![0.0; 16];
        let any_code = vec![0xAAu8; 2];
        assert_eq!(q.estimate_dot_rotated(&q_rot, &any_code), 0.0);
    }

    #[test]
    fn estimate_is_unbiased_indicator_of_alignment() {
        // Stronger query alignment with the encoded sign pattern
        // produces a larger estimate.
        let q = BitQuantizer::new(8);
        let q_rot = vec![1.0; 8];

        // Code with all bits set (= all docs positive) → estimate = +8.
        let code_all = vec![0xFFu8];
        // Code with all bits cleared → estimate = -8.
        let code_none = vec![0x00u8];
        // Code with half the bits set → estimate = 0.
        let code_half = vec![0x0Fu8]; // 4 bits → 4 positive, 4 negative

        assert!(approx(q.estimate_dot_rotated(&q_rot, &code_all), 8.0, 1e-5));
        assert!(approx(
            q.estimate_dot_rotated(&q_rot, &code_none),
            -8.0,
            1e-5
        ));
        assert!(approx(
            q.estimate_dot_rotated(&q_rot, &code_half),
            0.0,
            1e-5
        ));
    }

    // --- sanity --------------------------------------------------------

    #[test]
    fn sign_table_has_correct_size() {
        let q = BitQuantizer::new(128);
        assert_eq!(q.sign_table.len(), 256 * 8);
    }

    #[test]
    fn quantizer_is_clone() {
        let q = BitQuantizer::new(64);
        let _q2 = q.clone();
    }

    // --- AVX-512 parity ------------------------------------------------

    /// Deterministic pseudo-random `f32` vector for parity tests.
    fn fake_vec(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as i32;
                (x as f32) * 1e-6
            })
            .collect()
    }

    /// Build an arbitrary code from the quantizer's encode of a
    /// pseudo-random doc vector. Avoids degenerate all-1 / all-0
    /// codes that the existing tests probe.
    fn fake_code(quant: &BitQuantizer, seed: u32) -> Vec<u8> {
        let d_vec = fake_vec(quant.dim, seed);
        let mut code = vec![0u8; quant.code_bytes()];
        quant.encode_rotated_into(&d_vec, &mut code);
        code
    }

    /// AVX-512 RaBitQ estimator vs the wide sign-table kernel
    /// across a length sweep. Targets dims that exercise:
    /// - the 16-lane unroll boundary (16, 32, 48, 64),
    /// - the 8-lane half-tail (24, 40, 56),
    /// - the scalar tail (7, 15, 17, 23, 25, ...).
    ///
    /// Tolerance: `1e-4 * max(1, |result|)`. Both kernels do the
    /// same arithmetic identity (Σ q · (2b−1)) but in different
    /// reduction orders; tolerance must cover one ULP per
    /// accumulator slot times √(dim/16), which works out to ≪ 1e-4
    /// at our scales.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn estimate_avx512_matches_wide_across_lengths() {
        if !avx512_enabled() {
            eprintln!("estimate_avx512_matches_wide_across_lengths: skipped, no AVX-512");
            return;
        }
        for dim in [
            1usize, 7, 8, 15, 16, 17, 23, 24, 31, 32, 40, 48, 64, 96, 128, 384, 768,
        ] {
            let q = BitQuantizer::new(dim);
            let q_rot = fake_vec(dim, 0xC0DE);
            let code = fake_code(&q, 0xD0DE);
            let q_total: f32 = q_rot.iter().sum();
            let want = estimate_dot_rotated_wide(&q.sign_table, &q_rot, &code, dim);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { estimate_dot_rotated_avx512(&q_rot, &code, q_total, dim) };
            let tol = 1e-4 * want.abs().max(1.0) + 1e-5 * (dim as f32).sqrt();
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: avx512 {got} vs wide {want} (tol {tol})"
            );
        }
    }

    /// Public `estimate_dot_rotated` and the explicit
    /// `estimate_dot_rotated_with_total` must return the same value
    /// — the former just computes `q_total` inline before delegating.
    /// Pins the per-query precompute → per-candidate kernel split
    /// against a future regression that uses different math in the
    /// two paths.
    #[test]
    fn estimate_inline_and_precomputed_total_agree() {
        for &dim in &[16usize, 32, 33, 64, 384] {
            let q = BitQuantizer::new(dim);
            let q_rot = fake_vec(dim, 0xFEED);
            let code = fake_code(&q, 0xBABE);
            let inline = q.estimate_dot_rotated(&q_rot, &code);
            let q_total: f32 = q_rot.iter().sum();
            let precomp = q.estimate_dot_rotated_with_total(&q_rot, &code, q_total);
            assert_eq!(
                inline, precomp,
                "dim {dim}: inline {inline} vs precomp {precomp}"
            );
        }
    }

    // --- AVX-512 microbench (run by hand) ------------------------------
    //
    // Direct head-to-head per-kernel timings. Run with:
    //
    // ```text
    // cargo test --release --lib superfile::vector::quant::tests::\
    //   avx512_microbench -- --ignored --nocapture
    // ```

    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn avx512_microbench_estimate_dot_rotated() {
        if !avx512_enabled() {
            eprintln!("avx512_microbench: skipped, no AVX-512 on this host");
            return;
        }
        use std::{hint::black_box, time::Instant};

        eprintln!();
        eprintln!("### RaBitQ estimator — AVX-512 mask-add vs wide sign-table (ns per call)\n");
        eprintln!("| kernel | dim | wide ns | avx512 ns | speedup |");
        eprintln!("|--------|----:|--------:|----------:|--------:|");

        for &dim in &[128usize, 384, 768, 1024, 1536] {
            let q = BitQuantizer::new(dim);
            let q_rot = fake_vec(dim, 0xC0DE);
            let code = fake_code(&q, 0xD0DE);
            let q_total: f32 = q_rot.iter().sum();
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            // Warmup — black_box inputs to prevent the compiler hoisting
            // the call out of the loop on loop-invariant slice refs.
            for _ in 0..(iters / 10).max(64) {
                black_box(estimate_dot_rotated_wide(
                    black_box(&q.sign_table),
                    black_box(&q_rot),
                    black_box(&code),
                    black_box(dim),
                ));
            }
            let t = Instant::now();
            for _ in 0..iters {
                black_box(estimate_dot_rotated_wide(
                    black_box(&q.sign_table),
                    black_box(&q_rot),
                    black_box(&code),
                    black_box(dim),
                ));
            }
            let wide_ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

            // SAFETY: gated on avx512_enabled() above.
            for _ in 0..(iters / 10).max(64) {
                black_box(unsafe {
                    estimate_dot_rotated_avx512(
                        black_box(&q_rot),
                        black_box(&code),
                        black_box(q_total),
                        black_box(dim),
                    )
                });
            }
            let t = Instant::now();
            for _ in 0..iters {
                black_box(unsafe {
                    estimate_dot_rotated_avx512(
                        black_box(&q_rot),
                        black_box(&code),
                        black_box(q_total),
                        black_box(dim),
                    )
                });
            }
            let avx_ns = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

            eprintln!(
                "| `quant::estimate_dot_rotated` | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );
        }
    }
}
