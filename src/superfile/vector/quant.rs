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

use crate::superfile::vector::distance::sum_f32;
#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::{avx2_enabled, avx512_enabled, has_vbmi};

/// Number of sign bits packed into one code byte (1-bit RaBitQ packs
/// one sign per dimension, eight per byte). One `f32x8` SIMD block
/// also covers exactly this many dimensions, so the same constant
/// drives both the bit-packing and the per-block SIMD stride.
const BITS_PER_CODE_BYTE: usize = 8;

/// Rows per packed FastScan block: one AVX-512 byte register's worth of
/// lanes. The packed layout groups the same code-byte position of
/// [`LUT_BLOCK_ROWS`] consecutive rows so one register load feeds one
/// table lookup for all of them.
pub(crate) const LUT_BLOCK_ROWS: usize = 64;

/// Dimensions folded into one nibble group: each 4 query dims collapse
/// into a 16-entry signed-i8 lookup table indexed by 4 code bits.
const LUT_GROUP_DIMS: usize = 4;

/// Entries per nibble table (2^[`LUT_GROUP_DIMS`]).
const LUT_ENTRIES_PER_GROUP: usize = 16;

/// The i8 quantization ceiling for LUT entries.
const LUT_I8_MAX: f32 = 127.0;

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
        let q_total: f32 = sum_f32(q_rot);
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

// ---------------- FastScan LUT transposed code scan ----------------
//
// The warm 1-bit scan's fast path: the codes equivalent of the routing
// layer's transposed centroid cache (`build_transposed_centroid_cache`
// + `for_each_centroid_block_scores` in `distance`). The query folds
// once into per-group nibble tables (4 dims -> 16 signed-i8 entries);
// cluster codes — transposed position-major in [`LUT_BLOCK_ROWS`]-row
// blocks — are then scored 64 rows per table permute. Estimates come
// back i8-quantized (bounded by [`LutQuery::quantization_bound`]); the
// exact rerank downstream consumes them exactly as it consumes the
// full-precision estimator's output.

/// A query folded into FastScan nibble tables. Build once per query
/// per column (pure function of the rotated query); share across every
/// cluster scanned with it.
pub(crate) struct LutQuery {
    /// `groups * `[`LUT_ENTRIES_PER_GROUP`] signed entries; group `g`
    /// covers query dims `[g*4, g*4+4)`.
    luts: Vec<i8>,
    /// Undoes the i8 quantization: multiply summed table entries by
    /// this to get back to estimate scale.
    inv_scale: f32,
    pub(crate) groups: usize,
    /// Exact worst-case |accumulator| for THIS query: the sum over
    /// groups of each group's max |entry|. The i16 kernels are safe
    /// iff this fits i16 — see [`LutQuery::fits_i16`].
    worst_abs: u32,
}

impl LutQuery {
    pub(crate) fn new(q_rot: &[f32]) -> LutQuery {
        let dim = q_rot.len();
        let groups = dim.div_ceil(LUT_GROUP_DIMS);
        // Scale so the largest-magnitude group sum maps to i8 range.
        let mut max_abs = 1e-12f32;
        for g in 0..groups {
            let d0 = g * LUT_GROUP_DIMS;
            let s: f32 = q_rot[d0..(d0 + LUT_GROUP_DIMS).min(dim)]
                .iter()
                .map(|v| v.abs())
                .sum();
            max_abs = max_abs.max(s);
        }
        let scale = LUT_I8_MAX / max_abs;
        let mut luts = vec![0i8; groups * LUT_ENTRIES_PER_GROUP];
        let mut worst_abs = 0u32;
        for g in 0..groups {
            let d0 = g * LUT_GROUP_DIMS;
            let mut group_max = 0u32;
            for nib in 0..LUT_ENTRIES_PER_GROUP as u32 {
                let mut s = 0f32;
                for (bit, d) in (d0..(d0 + LUT_GROUP_DIMS).min(dim)).enumerate() {
                    let sign = if (nib >> bit) & 1 == 1 { 1.0 } else { -1.0 };
                    s += sign * q_rot[d];
                }
                let entry = (s * scale).round().clamp(-LUT_I8_MAX, LUT_I8_MAX) as i8;
                luts[g * LUT_ENTRIES_PER_GROUP + nib as usize] = entry;
                group_max = group_max.max(entry.unsigned_abs() as u32);
            }
            worst_abs += group_max;
        }
        LutQuery {
            luts,
            inv_scale: 1.0 / scale,
            groups,
            worst_abs,
        }
    }

    /// Whether the i16 block accumulators can represent every possible
    /// row sum for this query — the exact per-query bound, not a dim
    /// heuristic (768d always fits; 1536d fits for real queries; only
    /// pathological shapes fall back). When false the caller keeps the
    /// exact row-major estimator, which is the MORE precise path — the
    /// fallback costs speed, never correctness.
    #[inline]
    pub(crate) fn fits_i16(&self) -> bool {
        let fits = self.worst_abs <= i16::MAX as u32;
        if !fits {
            lut_overflow_fallback_warn_once(self.groups);
        }
        fits
    }

    /// Worst-case absolute error of the LUT estimate vs the exact
    /// estimator: one rounding half-step per group.
    #[cfg(test)]
    pub(crate) fn quantization_bound(&self) -> f32 {
        self.groups as f32 * 0.5 * self.inv_scale
    }
}

/// One-time visibility for the (rare) i16-bound fallback, so a corpus
/// running on the exact estimator is diagnosable rather than silent.
fn lut_overflow_fallback_warn_once(groups: usize) {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            groups,
            "vector scan: FastScan LUT path disabled for this query shape \
             (i16 accumulator bound); using the exact row-major estimator"
        );
    });
}

/// Whether the transposed LUT scan path exists on this host. ISA gate
/// only — the per-query accumulator bound is [`LutQuery::fits_i16`].
#[inline]
pub(crate) fn lut_scan_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        has_vbmi() || avx2_enabled()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Build the block-transposed code cache for one cluster: `cnt`
/// row-major rows of `cb` code bytes become position-major
/// [`LUT_BLOCK_ROWS`]-row blocks —
/// `out[block*cb*64 + pos*64 + lane] = codes[(block*64+lane)*cb + pos]`.
/// The tail block zero-pads missing rows (a zero code scores the
/// constant `sum(-|q|)` per group; callers slice results to `cnt`).
///
/// Same role as [`distance`]'s `build_transposed_centroid_cache`, but
/// byte-granular and hot at first touch (once per cluster per cache
/// lifetime), so the interior is a word-parallel 8x8 byte-tile
/// transpose (three shift/mask exchange rounds per tile, Hacker's
/// Delight 7-3) with scalar edges; safe Rust throughout.
pub(crate) fn build_transposed_code_cache(codes: &[u8], cnt: usize, cb: usize) -> Vec<u8> {
    /// One 8x8 byte tile held in 8 little-endian u64 rows.
    #[inline]
    fn transpose_8x8(rows: &mut [u64; 8]) {
        /// Byte lanes exchanged in round 1 (adjacent bytes).
        const M1: u64 = 0x00FF_00FF_00FF_00FF;
        /// Byte lanes exchanged in round 2 (byte pairs).
        const M2: u64 = 0x0000_FFFF_0000_FFFF;
        /// Byte lanes exchanged in round 3 (byte quads).
        const M4: u64 = 0x0000_0000_FFFF_FFFF;
        for i in [0usize, 2, 4, 6] {
            let t = ((rows[i] >> 8) ^ rows[i + 1]) & M1;
            rows[i + 1] ^= t;
            rows[i] ^= t << 8;
        }
        for i in [0usize, 1, 4, 5] {
            let t = ((rows[i] >> 16) ^ rows[i + 2]) & M2;
            rows[i + 2] ^= t;
            rows[i] ^= t << 16;
        }
        for i in [0usize, 1, 2, 3] {
            let t = ((rows[i] >> 32) ^ rows[i + 4]) & M4;
            rows[i + 4] ^= t;
            rows[i] ^= t << 32;
        }
    }
    /// Square byte-tile edge for the word-parallel transpose.
    const TILE: usize = 8;
    let blocks = cnt.div_ceil(LUT_BLOCK_ROWS);
    let mut out = vec![0u8; blocks * cb * LUT_BLOCK_ROWS];
    let full_row_tiles = cnt / TILE;
    let full_col_tiles = cb / TILE;
    for rt in 0..full_row_tiles {
        let r0 = rt * TILE;
        let block = r0 / LUT_BLOCK_ROWS;
        let lane0 = r0 % LUT_BLOCK_ROWS;
        let base = block * cb * LUT_BLOCK_ROWS;
        for ct in 0..full_col_tiles {
            let p0 = ct * TILE;
            let mut tile = [0u64; TILE];
            for (i, row) in tile.iter_mut().enumerate() {
                let src = (r0 + i) * cb + p0;
                *row = u64::from_le_bytes(
                    codes[src..src + TILE].try_into().expect("8-byte row slice"),
                );
            }
            transpose_8x8(&mut tile);
            for (j, row) in tile.iter().enumerate() {
                let dst = base + (p0 + j) * LUT_BLOCK_ROWS + lane0;
                out[dst..dst + TILE].copy_from_slice(&row.to_le_bytes());
            }
        }
        // Column tail (cb not a multiple of 8).
        for p in full_col_tiles * TILE..cb {
            for i in 0..TILE {
                out[base + p * LUT_BLOCK_ROWS + lane0 + i] = codes[(r0 + i) * cb + p];
            }
        }
    }
    // Row tail (cnt not a multiple of 8).
    for r in full_row_tiles * TILE..cnt {
        let block = r / LUT_BLOCK_ROWS;
        let lane = r % LUT_BLOCK_ROWS;
        let base = block * cb * LUT_BLOCK_ROWS;
        for p in 0..cb {
            out[base + p * LUT_BLOCK_ROWS + lane] = codes[r * cb + p];
        }
    }
    out
}

/// Drive the block scorers over every [`LUT_BLOCK_ROWS`]-row block of
/// a transposed code cache, handing each block's 64 estimates to
/// `reduce(base_row, scores)`. Tier dispatch is hoisted outside the
/// block loop (same shape as [`distance`]'s
/// `for_each_centroid_block_scores`); all tiers produce bit-identical
/// scores (integer accumulation in one order, one f32 scale at the
/// end). Callers gate the *path* on [`lut_scan_supported`] +
/// [`LutQuery::fits_i16`]; the scalar tier closes the dispatch.
pub(crate) fn for_each_code_block_scores(
    cache: &[u8],
    cb: usize,
    lut: &LutQuery,
    mut reduce: impl FnMut(usize, &[f32; LUT_BLOCK_ROWS]),
) {
    debug_assert_eq!(cache.len() % (cb * LUT_BLOCK_ROWS), 0);
    debug_assert!(lut.worst_abs <= i16::MAX as u32);
    let n_blocks = cache.len() / (cb * LUT_BLOCK_ROWS);
    let mut scores = [0f32; LUT_BLOCK_ROWS];
    #[cfg(target_arch = "x86_64")]
    {
        if has_vbmi() {
            for block in 0..n_blocks {
                let span = &cache[block * cb * LUT_BLOCK_ROWS..(block + 1) * cb * LUT_BLOCK_ROWS];
                // SAFETY: gated on `has_vbmi()` which implies `avx512f` +
                // `avx512bw` and checks `avx512vbmi`.
                unsafe { score_code_block64_transposed_avx512(span, cb, lut, &mut scores) };
                reduce(block * LUT_BLOCK_ROWS, &scores);
            }
            return;
        }
        if avx2_enabled() {
            for block in 0..n_blocks {
                let span = &cache[block * cb * LUT_BLOCK_ROWS..(block + 1) * cb * LUT_BLOCK_ROWS];
                // SAFETY: gated on `avx2_enabled()` which requires `avx2`.
                unsafe { score_code_block64_transposed_avx2(span, cb, lut, &mut scores) };
                reduce(block * LUT_BLOCK_ROWS, &scores);
            }
            return;
        }
    }
    for block in 0..n_blocks {
        let span = &cache[block * cb * LUT_BLOCK_ROWS..(block + 1) * cb * LUT_BLOCK_ROWS];
        score_code_block64_transposed_scalar(span, cb, lut, &mut scores);
        reduce(block * LUT_BLOCK_ROWS, &scores);
    }
}

/// Portable reference tier: same i16 accumulation as the SIMD kernels,
/// lane by lane. `wide` offers no byte-table permute, so the portable
/// form stays scalar — reachable only when both x86 tiers are disabled
/// (or off-x86, where [`lut_scan_supported`] never selects this path).
fn score_code_block64_transposed_scalar(
    block: &[u8],
    cb: usize,
    lut: &LutQuery,
    est_out: &mut [f32; LUT_BLOCK_ROWS],
) {
    let mut acc = [0i16; LUT_BLOCK_ROWS];
    for p in 0..cb {
        let g_lo = 2 * p;
        let g_hi = 2 * p + 1;
        let bytes = &block[p * LUT_BLOCK_ROWS..(p + 1) * LUT_BLOCK_ROWS];
        for (lane, &b) in bytes.iter().enumerate() {
            let lo = (b & 0x0F) as usize;
            acc[lane] += i16::from(lut.luts[g_lo * LUT_ENTRIES_PER_GROUP + lo]);
            if g_hi < lut.groups {
                let hi = (b >> 4) as usize;
                acc[lane] += i16::from(lut.luts[g_hi * LUT_ENTRIES_PER_GROUP + hi]);
            }
        }
    }
    for (lane, &a) in acc.iter().enumerate() {
        est_out[lane] = f32::from(a) * lut.inv_scale;
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx512f`, `avx512bw`, and
/// `avx512vbmi`. [`has_vbmi`] guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn score_code_block64_transposed_avx512(
    block: &[u8],
    cb: usize,
    lut: &LutQuery,
    est_out: &mut [f32; LUT_BLOCK_ROWS],
) {
    use std::arch::x86_64::*;
    // SAFETY: `block.len() == cb * 64` (the driver slices exactly that
    // span), so every 64-byte load at `p * 64` for `p < cb` is in
    // bounds. Each 16-byte LUT load at `g * 16` is in bounds because
    // `g_lo, g_hi < lut.groups` are checked and
    // `luts.len() == groups * 16`. `_mm512_permutexvar_epi8` requires
    // `avx512vbmi`, guaranteed by the caller per the `# Safety`
    // contract. i16 accumulators cannot overflow: the driver
    // debug-asserts `lut.worst_abs <= i16::MAX` (exact per-query bound).
    unsafe {
        let mut acc_lo = _mm512_setzero_si512();
        let mut acc_hi = _mm512_setzero_si512();
        let nib_mask = _mm512_set1_epi8(0x0F);
        for p in 0..cb {
            let bytes = _mm512_loadu_si512(block.as_ptr().add(p * LUT_BLOCK_ROWS) as *const _);
            let g_lo = 2 * p;
            let g_hi = 2 * p + 1;
            let lut_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(
                lut.luts.as_ptr().add(g_lo * LUT_ENTRIES_PER_GROUP) as *const _,
            ));
            let idx_lo = _mm512_and_si512(bytes, nib_mask);
            let val_lo = _mm512_permutexvar_epi8(idx_lo, lut_lo);
            acc_lo = _mm512_add_epi16(acc_lo, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(val_lo)));
            acc_hi = _mm512_add_epi16(
                acc_hi,
                _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(val_lo, 1)),
            );
            if g_hi < lut.groups {
                let lut_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(
                    lut.luts.as_ptr().add(g_hi * LUT_ENTRIES_PER_GROUP) as *const _,
                ));
                let idx_hi = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), nib_mask);
                let val_hi = _mm512_permutexvar_epi8(idx_hi, lut_hi);
                acc_lo =
                    _mm512_add_epi16(acc_lo, _mm512_cvtepi8_epi16(_mm512_castsi512_si256(val_hi)));
                acc_hi = _mm512_add_epi16(
                    acc_hi,
                    _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(val_hi, 1)),
                );
            }
        }
        let mut tmp = [0i16; LUT_BLOCK_ROWS];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut _, acc_lo);
        _mm512_storeu_si512(tmp.as_mut_ptr().add(32) as *mut _, acc_hi);
        for r in 0..LUT_BLOCK_ROWS {
            est_out[r] = f32::from(tmp[r]) * lut.inv_scale;
        }
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`. [`avx2_enabled`]
/// guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn score_code_block64_transposed_avx2(
    block: &[u8],
    cb: usize,
    lut: &LutQuery,
    est_out: &mut [f32; LUT_BLOCK_ROWS],
) {
    use std::arch::x86_64::*;
    // SAFETY: `block.len() == cb * 64` (the driver slices exactly that
    // span), so the two 32-byte loads at `p * 64` and `p * 64 + 32` are
    // in bounds for `p < cb`. Each 16-byte LUT load at `g * 16` is in
    // bounds because `g_lo, g_hi < lut.groups` and
    // `luts.len() == groups * 16`. `_mm256_shuffle_epi8` looks up
    // within 128-bit halves; the table is broadcast to both halves and
    // indices are masked to 0..15 with the high bit clear, so both
    // halves index the same 16-entry table. i16 accumulators cannot
    // overflow: the driver debug-asserts `lut.worst_abs <= i16::MAX`.
    unsafe {
        // 64 i16 lanes = four 256-bit accumulators: [0..16), [16..32)
        // for the low 32 rows, [32..48), [48..64) for the high 32.
        let mut acc = [_mm256_setzero_si256(); 4];
        let nib_mask = _mm256_set1_epi8(0x0F);
        for p in 0..cb {
            let lo32 = _mm256_loadu_si256(block.as_ptr().add(p * LUT_BLOCK_ROWS) as *const _);
            let hi32 = _mm256_loadu_si256(block.as_ptr().add(p * LUT_BLOCK_ROWS + 32) as *const _);
            let g_lo = 2 * p;
            let g_hi = 2 * p + 1;
            let lut_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                lut.luts.as_ptr().add(g_lo * LUT_ENTRIES_PER_GROUP) as *const _,
            ));
            let val_a = _mm256_shuffle_epi8(lut_lo, _mm256_and_si256(lo32, nib_mask));
            acc[0] = _mm256_add_epi16(acc[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_a)));
            acc[1] = _mm256_add_epi16(
                acc[1],
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_a, 1)),
            );
            let val_b = _mm256_shuffle_epi8(lut_lo, _mm256_and_si256(hi32, nib_mask));
            acc[2] = _mm256_add_epi16(acc[2], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_b)));
            acc[3] = _mm256_add_epi16(
                acc[3],
                _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_b, 1)),
            );
            if g_hi < lut.groups {
                let lut_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                    lut.luts.as_ptr().add(g_hi * LUT_ENTRIES_PER_GROUP) as *const _,
                ));
                let idx_a = _mm256_and_si256(_mm256_srli_epi16(lo32, 4), nib_mask);
                let val_a = _mm256_shuffle_epi8(lut_hi, idx_a);
                acc[0] =
                    _mm256_add_epi16(acc[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_a)));
                acc[1] = _mm256_add_epi16(
                    acc[1],
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_a, 1)),
                );
                let idx_b = _mm256_and_si256(_mm256_srli_epi16(hi32, 4), nib_mask);
                let val_b = _mm256_shuffle_epi8(lut_hi, idx_b);
                acc[2] =
                    _mm256_add_epi16(acc[2], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(val_b)));
                acc[3] = _mm256_add_epi16(
                    acc[3],
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(val_b, 1)),
                );
            }
        }
        let mut tmp = [0i16; LUT_BLOCK_ROWS];
        for (i, a) in acc.iter().enumerate() {
            _mm256_storeu_si256(tmp.as_mut_ptr().add(i * 16) as *mut _, *a);
        }
        for r in 0..LUT_BLOCK_ROWS {
            est_out[r] = f32::from(tmp[r]) * lut.inv_scale;
        }
    }
}
// -------------- END FastScan LUT transposed code scan --------------

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

    /// The word-parallel transposed code cache builder vs a plain
    /// nested-loop reference, across row counts that cross the 64-row
    /// block boundary and code widths that exercise the 8-byte tile
    /// edges (odd `cb` = column tail; odd `cnt` = row tail).
    #[test]
    fn build_transposed_code_cache_matches_reference() {
        for &(cnt, cb) in &[
            (1usize, 12usize),
            (7, 12),
            (8, 12),
            (63, 96),
            (64, 96),
            (65, 96),
            (100, 13),
            (129, 96),
            (200, 8),
        ] {
            let codes: Vec<u8> = (0..cnt * cb)
                .map(|i| ((i as u32).wrapping_mul(2654435761) >> 24) as u8)
                .collect();
            let blocks = cnt.div_ceil(LUT_BLOCK_ROWS);
            let mut want = vec![0u8; blocks * cb * LUT_BLOCK_ROWS];
            for r in 0..cnt {
                let block = r / LUT_BLOCK_ROWS;
                let lane = r % LUT_BLOCK_ROWS;
                for p in 0..cb {
                    want[block * cb * LUT_BLOCK_ROWS + p * LUT_BLOCK_ROWS + lane] =
                        codes[r * cb + p];
                }
            }
            let got = build_transposed_code_cache(&codes, cnt, cb);
            assert_eq!(got, want, "cnt {cnt} cb {cb}");
        }
    }

    /// Every LUT scan tier must produce bit-identical scores: the
    /// accumulation is integer (order-free) and the only float op is
    /// one final scale. Mirrors `sq8_simd`'s exact-equality tier
    /// tests, including the raw feature probes that bypass the config
    /// gates so a diagnostics toggle cannot skip a tier here.
    #[test]
    fn code_block_scores_transposed_match_scalar_reference() {
        for &dim in &[12usize, 60, 764, 768, 1024] {
            let quant = BitQuantizer::new(dim);
            let cb = quant.code_bytes();
            let cnt = 130;
            let mut codes = Vec::with_capacity(cnt * cb);
            for row in 0..cnt {
                codes.extend_from_slice(&fake_code(&quant, 0x1234 + row as u32));
            }
            let cache = build_transposed_code_cache(&codes, cnt, cb);
            let lut = LutQuery::new(&fake_vec(dim, 0xBEEF));
            assert!(lut.fits_i16(), "dim {dim} must fit i16");
            let n_blocks = cache.len() / (cb * LUT_BLOCK_ROWS);
            let mut want = vec![[0f32; LUT_BLOCK_ROWS]; n_blocks];
            for (block, out) in want.iter_mut().enumerate() {
                score_code_block64_transposed_scalar(
                    &cache[block * cb * LUT_BLOCK_ROWS..(block + 1) * cb * LUT_BLOCK_ROWS],
                    cb,
                    &lut,
                    out,
                );
            }
            #[cfg(target_arch = "x86_64")]
            for tier in ["avx2", "avx512"] {
                let mut got = [0f32; LUT_BLOCK_ROWS];
                for (block, want_block) in want.iter().enumerate() {
                    let span =
                        &cache[block * cb * LUT_BLOCK_ROWS..(block + 1) * cb * LUT_BLOCK_ROWS];
                    match tier {
                        "avx2" if std::arch::is_x86_feature_detected!("avx2") => {
                            // SAFETY: gated on the raw avx2 probe above.
                            unsafe { score_code_block64_transposed_avx2(span, cb, &lut, &mut got) }
                        }
                        "avx512"
                            if std::arch::is_x86_feature_detected!("avx512f")
                                && std::arch::is_x86_feature_detected!("avx512bw")
                                && std::arch::is_x86_feature_detected!("avx512vbmi") =>
                        {
                            // SAFETY: gated on the raw avx512f/bw/vbmi probes above.
                            unsafe {
                                score_code_block64_transposed_avx512(span, cb, &lut, &mut got)
                            }
                        }
                        _ => continue,
                    }
                    assert_eq!(
                        &got[..],
                        &want_block[..],
                        "tier {tier} dim {dim} block {block}"
                    );
                }
            }
        }
    }

    /// The i8-quantized LUT estimate must sit within its analytic
    /// rounding bound of the exact estimator on every row: half a
    /// quantization step per nibble group.
    #[test]
    fn lut_estimate_within_quantization_bound() {
        for &dim in &[60usize, 768, 1024] {
            let quant = BitQuantizer::new(dim);
            let cb = quant.code_bytes();
            let q_rot = fake_vec(dim, 0xC0FFEE);
            let lut = LutQuery::new(&q_rot);
            let bound = lut.quantization_bound() + 1e-3;
            let cnt = 96;
            let mut codes = Vec::with_capacity(cnt * cb);
            for row in 0..cnt {
                codes.extend_from_slice(&fake_code(&quant, 0x77 + row as u32));
            }
            let cache = build_transposed_code_cache(&codes, cnt, cb);
            let mut checked = 0usize;
            for_each_code_block_scores(&cache, cb, &lut, |base_r, scores| {
                for (lane, &est) in scores.iter().enumerate() {
                    let r = base_r + lane;
                    if r >= cnt {
                        continue;
                    }
                    let exact = quant.estimate_dot_rotated(&q_rot, &codes[r * cb..(r + 1) * cb]);
                    assert!(
                        (est - exact).abs() <= bound,
                        "dim {dim} row {r}: lut {est} vs exact {exact} (bound {bound})"
                    );
                    checked += 1;
                }
            });
            assert_eq!(checked, cnt);
        }
    }

    /// `fits_i16` is the exact per-query accumulator bound: an
    /// adversarial equal-magnitude query saturates every group to the
    /// i8 ceiling, so past 258 groups (1,032 dims) it must decline —
    /// and a realistic 768-dim query must pass.
    #[test]
    fn lut_query_i16_bound_is_exact_per_query() {
        assert!(LutQuery::new(&fake_vec(768, 0xAB)).fits_i16());
        // Equal |q| per dim -> every group's max entry is 127 ->
        // worst_abs = groups * 127 = 512 * 127 > i16::MAX.
        let adversarial = vec![1.0f32; 2048];
        assert!(!LutQuery::new(&adversarial).fits_i16());
        // The same 2048 dims with mass concentrated in one group keeps
        // the other groups' entries tiny: the exact bound admits it
        // where a dim heuristic would have declined.
        let mut concentrated = vec![1e-4f32; 2048];
        concentrated[0] = 1.0;
        assert!(LutQuery::new(&concentrated).fits_i16());
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
