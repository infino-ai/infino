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
//! See `docs/architecture/superfile.md` (Vector index algorithm
//! subsection) for the full RaBitQ rationale and recall trade-offs.

use wide::{CmpGt, f32x8};

/// 1-bit quantizer + estimator for vectors of fixed dimension `dim`.
/// Construct once per column at index-build time; reuse for both
/// encoding (build-side) and dot-estimation (query-side).
#[derive(Debug, Clone)]
pub struct BitQuantizer {
    pub dim: usize,
    sign_table: Box<[f32; 256 * 8]>,
}

impl BitQuantizer {
    /// Build the sign lookup table for vectors of dimension `dim`.
    /// Cost: `256 * 8 * 4 = 8 KB` heap, computed once.
    pub fn new(dim: usize) -> Self {
        let mut table = Box::new([0.0f32; 256 * 8]);
        for b in 0..256usize {
            for bit in 0..8 {
                let set = (b >> bit) & 1;
                table[b * 8 + bit] = if set == 1 { 1.0 } else { -1.0 };
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
        self.dim.div_ceil(8)
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
        let full_bytes = self.dim / 8;
        for byte_idx in 0..full_bytes {
            let lane: [f32; 8] = rotated[byte_idx * 8..byte_idx * 8 + 8]
                .try_into()
                .expect("slice [byte_idx*8..byte_idx*8+8] has length 8");
            let v = f32x8::from(lane);
            // `to_bitmask` returns one u32 whose low 8 bits are the
            // sign/comparison bits for each lane, in lane-order — bit
            // 0 = lane 0 > 0.0, bit 7 = lane 7 > 0.0. Exactly the
            // bit-order the scalar reference loop produces.
            out[byte_idx] = v.simd_gt(zero).to_bitmask() as u8;
        }
        let tail_start = full_bytes * 8;
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
    #[inline]
    pub fn estimate_dot_rotated(&self, q_rot: &[f32], code: &[u8]) -> f32 {
        debug_assert_eq!(q_rot.len(), self.dim);
        debug_assert_eq!(code.len(), self.code_bytes());

        let full_bytes = self.dim / 8;
        let mut acc = f32x8::ZERO;
        for byte_idx in 0..full_bytes {
            let b = code[byte_idx] as usize;
            let signs_slice: &[f32; 8] = (&self.sign_table[b * 8..b * 8 + 8])
                .try_into()
                .expect("slice [b*8..b*8+8] has length 8");
            let q_slice: &[f32; 8] = (&q_rot[byte_idx * 8..byte_idx * 8 + 8])
                .try_into()
                .expect("slice [byte_idx*8..byte_idx*8+8] has length 8");
            let signs = f32x8::from(*signs_slice);
            let q_block = f32x8::from(*q_slice);
            acc += q_block * signs;
        }
        let mut sum: f32 = acc.reduce_add();

        // Tail: dims [full_bytes*8 .. dim] handled scalar.
        let tail_start = full_bytes * 8;
        if tail_start < self.dim {
            let byte = code[full_bytes] as usize;
            for i in 0..(self.dim - tail_start) {
                let bit = (byte >> i) & 1;
                let s = if bit == 1 { 1.0 } else { -1.0 };
                sum += q_rot[tail_start + i] * s;
            }
        }
        sum
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
}
