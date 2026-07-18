// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! SIMD primitives for the Sq8 build pipeline.
//!
//! Owns the per-tier (scalar/`wide`/AVX2/AVX-512) implementations of
//! the two SIMD-accelerated steps of the Sq8 column build:
//!
//! 1. **Per-dim min/max accumulation** ([`update_min_max`]) — folded
//!    into pass 2 so the per-cluster `(scale, offset)` quantizer
//!    falls out of the same row scan that writes the spill bucket.
//! 2. **f32 → u8 encode** ([`sq8_encode_row`]) — the hot loop in
//!    pass 3 that materializes the on-disk Sq8 codes from the
//!    cluster's fp32 rows and the precomputed
//!    [`Sq8EncodeConsts`] (`inv_scale`, `c2`) FMA constants.
//!
//! Both entry points are runtime-dispatched through
//! [`crate::superfile::vector::simd_dispatch`]: AVX-512F →
//! AVX2 + FMA → portable `wide::f32x8`. The tiers are kept here
//! (rather than in `simd_dispatch.rs`) because they're all
//! Sq8-encode-specific — the dispatch gates are shared, the
//! kernels are not.
//!
//! All three tiers of [`sq8_encode_row`] produce **byte-identical**
//! output for finite inputs (the parity tests at the bottom of this
//! module assert that across a dim sweep that hits every tail
//! shape). The encode is `q = FMA(x, inv_scale, c2)` followed by
//! `clamp(0, 255)` and a truncating cast — `c2` carries the
//! `+ 0.5` bias that turns the truncating cast into round-half-up,
//! which equals round-half-away on the non-negative `[0, 255]`
//! domain.

#[cfg(target_arch = "x86_64")]
use std::ptr::write_unaligned;

use crate::superfile::vector::distance::dot;
#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::{avx2_enabled, avx512_enabled};

/// Maximum Sq8 code value. Sq8 quantizes each component to a single
/// unsigned byte, so the encoder clamps the FMA result to
/// `[0, SQ8_CODE_MAX]` before the truncating cast to `u8`.
const SQ8_CODE_MAX: f32 = 255.0;

/// Round-half-up bias folded into the per-cluster `c2` encode
/// constant. Adding `0.5` before a truncating cast turns truncation
/// into round-half-up, which equals round-half-away on the
/// non-negative `[0, SQ8_CODE_MAX]` domain the encoder operates in.
const SQ8_ROUND_HALF_BIAS: f32 = 0.5;

/// Lane count of the portable `wide::f32x8` register and the AVX2
/// 256-bit f32 register. The 8-lane kernels process this many f32s
/// per iteration.
const F32X8_LANES: usize = 8;

/// Lane count of an AVX-512 f32 register (512-bit / 32-bit). The
/// AVX-512 kernels process this many f32s per iteration, with an
/// 8-lane half-tail for `dim % AVX512_F32_LANES == F32X8_LANES`.
// Referenced only by the x86-gated AVX-512 kernels; dead on other targets.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX512_F32_LANES: usize = 16;

/// Per-cluster Sq8 encode constants. Each cluster contributes one
/// `(inv_scale[dim], c2[dim])` pair where
///     `c2[d] = (-offset[d]) * inv_scale[d] + 0.5`
/// so the per-row encode collapses to a single FMA:
///     `q = x * inv_scale + c2 = (x - offset) * inv_scale + 0.5`
/// followed by `clamp(0, 255)` and a truncating cast to `u8`. The
/// `+ 0.5` is what turns the truncating cast into round-half-up,
/// which equals round-half-away on the non-negative domain that
/// the Sq8 encoder operates in (the float result is in `[0, 255]`
/// for in-cluster values; clamp catches the fp-noise overshoot at
/// both ends).
///
/// Precomputing per cluster (rather than per row) hoists `1/scale`
/// and the constant fold out of the dim-axis inner loop, so the
/// SIMD encoder loops over a single FMA + cast + narrow per lane.
pub(super) struct Sq8EncodeConsts {
    pub(super) inv_scale: Vec<f32>,
    pub(super) c2: Vec<f32>,
}

impl Sq8EncodeConsts {
    /// Derive `(inv_scale, c2)` from the cluster's `(scale, offset)`.
    /// Builds both vectors at the final length via iterator chains
    /// — no fill-then-overwrite pass.
    pub(super) fn from_scale_offset(scale: &[f32], offset: &[f32]) -> Self {
        debug_assert_eq!(scale.len(), offset.len());
        let inv_scale: Vec<f32> = scale.iter().map(|s| 1.0 / s).collect();
        let c2: Vec<f32> = offset
            .iter()
            .zip(inv_scale.iter())
            .map(|(o, inv)| (-*o).mul_add(*inv, SQ8_ROUND_HALF_BIAS))
            .collect();
        Self { inv_scale, c2 }
    }
}

/// Per-row, per-cluster `(min, max)` update for the Sq8 quantizer.
/// Called from pass 2 for every fp32 row routed to its bucket: takes
/// the row and the destination cluster's `min[dim]` / `max[dim]`
/// slices and updates them in-place with the per-dim element-wise
/// `min`/`max`.
///
/// Folds what used to be pass 3's per-cluster min/max scan into the
/// pass that already touches every row anyway, eliminating one
/// full re-read of the cluster's fp32 bytes after pass 2 spills them.
/// Three-tier dispatch: AVX-512 (16-lane `vminps` + `vmaxps`) →
/// AVX2 (8-lane `vminps` + `vmaxps`) → portable `wide::f32x8`.
#[inline]
pub(super) fn update_min_max(row: &[f32], min_slice: &mut [f32], max_slice: &mut [f32]) {
    debug_assert_eq!(row.len(), min_slice.len());
    debug_assert_eq!(row.len(), max_slice.len());

    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on avx512_enabled() which requires `avx512f`.
            unsafe { update_min_max_avx512(row, min_slice, max_slice) };
            return;
        }
        if avx2_enabled() {
            // SAFETY: gated on avx2_enabled() which requires `avx2`.
            unsafe { update_min_max_avx2(row, min_slice, max_slice) };
            return;
        }
    }
    update_min_max_wide(row, min_slice, max_slice);
}

/// Portable `wide::f32x8` (256-bit) per-dim min/max update. Universal
/// fallback for aarch64 / SSE-only x86_64 / `INFINO_DISABLE_AVX2=1`.
#[inline]
fn update_min_max_wide(row: &[f32], min_slice: &mut [f32], max_slice: &mut [f32]) {
    use wide::f32x8;
    let dim = row.len();
    let full = dim - dim % F32X8_LANES;
    let mut i = 0;
    while i < full {
        let r: [f32; F32X8_LANES] = row[i..i + F32X8_LANES].try_into().expect("len 8");
        let mn: [f32; F32X8_LANES] = min_slice[i..i + F32X8_LANES].try_into().expect("len 8");
        let mx: [f32; F32X8_LANES] = max_slice[i..i + F32X8_LANES].try_into().expect("len 8");
        let r_v = f32x8::from(r);
        let new_min = r_v.fast_min(f32x8::from(mn)).to_array();
        let new_max = r_v.fast_max(f32x8::from(mx)).to_array();
        min_slice[i..i + F32X8_LANES].copy_from_slice(&new_min);
        max_slice[i..i + F32X8_LANES].copy_from_slice(&new_max);
        i += F32X8_LANES;
    }
    while i < dim {
        let x = row[i];
        if x < min_slice[i] {
            min_slice[i] = x;
        }
        if x > max_slice[i] {
            max_slice[i] = x;
        }
        i += 1;
    }
}

/// AVX2 per-dim min/max update. 8 lanes per iteration with
/// `_mm256_min_ps` + `_mm256_max_ps` (single-instruction parallel
/// reduce). Lifts every AVX2 host that lacks AVX-512.
///
/// # Safety
///
/// Callers must ensure the target supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn update_min_max_avx2(row: &[f32], min_slice: &mut [f32], max_slice: &mut [f32]) {
    use std::arch::x86_64::*;
    let dim = row.len();
    let full = dim - dim % F32X8_LANES;
    let mut i = 0;
    // SAFETY: each iteration reads 24 bytes (one f32 lane of row,
    // min, max each at offset `i`) and writes 16 bytes (min, max
    // at the same offset). `i + 8 <= dim` keeps every load/store
    // in bounds; loads/stores are all unaligned.
    unsafe {
        while i < full {
            let r = _mm256_loadu_ps(row.as_ptr().add(i));
            let mn = _mm256_loadu_ps(min_slice.as_ptr().add(i));
            let mx = _mm256_loadu_ps(max_slice.as_ptr().add(i));
            let new_mn = _mm256_min_ps(r, mn);
            let new_mx = _mm256_max_ps(r, mx);
            _mm256_storeu_ps(min_slice.as_mut_ptr().add(i), new_mn);
            _mm256_storeu_ps(max_slice.as_mut_ptr().add(i), new_mx);
            i += F32X8_LANES;
        }
    }
    while i < dim {
        let x = row[i];
        if x < min_slice[i] {
            min_slice[i] = x;
        }
        if x > max_slice[i] {
            max_slice[i] = x;
        }
        i += 1;
    }
}

/// AVX-512 per-dim min/max update. 16 lanes per iteration with
/// `_mm512_min_ps` + `_mm512_max_ps`. Strictly faster than the AVX2
/// path on Sapphire Rapids / Granite Rapids / Zen 4+ because every
/// reduction step processes twice the lanes per cycle.
///
/// # Safety
///
/// Callers must ensure the target supports `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn update_min_max_avx512(row: &[f32], min_slice: &mut [f32], max_slice: &mut [f32]) {
    use std::arch::x86_64::*;
    let dim = row.len();
    let full16 = dim - dim % AVX512_F32_LANES;
    let mut i = 0;
    // SAFETY: per-iteration windows of 16 f32s in each of row /
    // min / max; bounded by `i + 16 <= dim`. Unaligned loads /
    // stores.
    unsafe {
        while i < full16 {
            let r = _mm512_loadu_ps(row.as_ptr().add(i));
            let mn = _mm512_loadu_ps(min_slice.as_ptr().add(i));
            let mx = _mm512_loadu_ps(max_slice.as_ptr().add(i));
            let new_mn = _mm512_min_ps(r, mn);
            let new_mx = _mm512_max_ps(r, mx);
            _mm512_storeu_ps(min_slice.as_mut_ptr().add(i), new_mn);
            _mm512_storeu_ps(max_slice.as_mut_ptr().add(i), new_mx);
            i += AVX512_F32_LANES;
        }
        // 8-lane half-tail for `dim % 16 == 8` (rare but
        // possible — keeps the kernel correct without falling to
        // scalar on the same iteration count).
        if i + F32X8_LANES <= dim {
            let r = _mm256_loadu_ps(row.as_ptr().add(i));
            let mn = _mm256_loadu_ps(min_slice.as_ptr().add(i));
            let mx = _mm256_loadu_ps(max_slice.as_ptr().add(i));
            let new_mn = _mm256_min_ps(r, mn);
            let new_mx = _mm256_max_ps(r, mx);
            _mm256_storeu_ps(min_slice.as_mut_ptr().add(i), new_mn);
            _mm256_storeu_ps(max_slice.as_mut_ptr().add(i), new_mx);
            i += F32X8_LANES;
        }
    }
    while i < dim {
        let x = row[i];
        if x < min_slice[i] {
            min_slice[i] = x;
        }
        if x > max_slice[i] {
            max_slice[i] = x;
        }
        i += 1;
    }
}

/// SIMD f32 → u8 Sq8 encode. Writes one row's `dim` codes to
/// `dst[0..dim]` from `row[0..dim]` and the cluster's precomputed
/// `inv_scale` / `c2` arrays.
///
/// All three tiers use the same FMA + clamp + truncating-int-cast
/// sequence so the byte output is bit-identical across paths:
///
///   q = FMA(x, inv_scale, c2)
///   q_clamped = max(0, min(255, q))
///   dst[d] = (q_clamped) as u8     // truncating cast
///
/// The scalar fallback in the tail loop uses `f32::mul_add` to keep
/// FMA semantics consistent with the SIMD lanes — `(x * inv) + c2`
/// without FMA would double-round and produce a ≤1 ulp drift from
/// the SIMD result, which is enough to flip a quantization boundary
/// and change the on-disk byte for that lane.
#[inline]
pub(super) fn sq8_encode_row(row: &[f32], inv_scale: &[f32], c2: &[f32], dst: &mut [u8]) {
    debug_assert_eq!(row.len(), inv_scale.len());
    debug_assert_eq!(row.len(), c2.len());
    debug_assert_eq!(row.len(), dst.len());

    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512f` feature detection.
            unsafe { sq8_encode_row_avx512(row, inv_scale, c2, dst) };
            return;
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2` feature detection.
            unsafe { sq8_encode_row_avx2(row, inv_scale, c2, dst) };
            return;
        }
    }
    sq8_encode_row_wide(row, inv_scale, c2, dst);
}

/// Symmetric clamp bound for the Sq8+ε i8 residual leg (matches IVF builder).
const SQ8_RESIDUAL_I8_CLAMP: f32 = 127.0;

/// Encode one fp32 row into Sq8+ε `(codes, residuals)` at the cluster's
/// `(scale, offset)`, using the precomputed code-leg `consts` (build once
/// per cluster, not per row). The Sq8 code leg uses the SIMD
/// [`sq8_encode_row`]; the residual leg captures quantization error at
/// `scale[d] / residual_divisor` steps and reconstructs the corrected
/// component into `recon` in the same pass.
///
/// When `store_norm`, returns the residual-corrected `‖x‖²` via the SIMD
/// [`dot`] helper over `recon` — the corrected vector is materialized by the
/// residual pass itself, so there is no second decode of the just-written
/// bytes (contrast a follow-up `sq8_residual_norm_sq`, which re-widens and
/// re-reconstructs every dim).
pub(super) fn encode_sq8_residual_row(
    row: &[f32],
    consts: &Sq8EncodeConsts,
    scale: &[f32],
    offset: &[f32],
    codes: &mut [u8],
    residuals: &mut [u8],
    recon: &mut [f32],
    store_norm: bool,
    residual_divisor: f32,
) -> Option<f32> {
    debug_assert_eq!(row.len(), scale.len());
    debug_assert_eq!(row.len(), offset.len());
    debug_assert_eq!(row.len(), codes.len());
    debug_assert_eq!(row.len(), residuals.len());
    debug_assert_eq!(row.len(), recon.len());
    sq8_encode_row(row, &consts.inv_scale, &consts.c2, codes);
    debug_assert!(residual_divisor > 0.0);
    let inv_div = 1.0 / residual_divisor;
    for d in 0..row.len() {
        let base = (codes[d] as f32).mul_add(scale[d], offset[d]);
        let step = scale[d] * inv_div;
        let rq = if step > 0.0 {
            ((row[d] - base) / step)
                .round()
                .clamp(-SQ8_RESIDUAL_I8_CLAMP, SQ8_RESIDUAL_I8_CLAMP) as i8
        } else {
            0
        };
        residuals[d] = rq.to_le_bytes()[0];
        recon[d] = base + (rq as f32) * step;
    }
    store_norm.then(|| dot(recon, recon))
}

/// Portable Sq8 encode. `wide::f32x8` handles the FMA + clamp in
/// SIMD; the narrow to u8 falls back to a per-lane scalar cast
/// because `wide` doesn't expose a saturating u8 narrow. Still
/// vectorizes the dominant fp work.
//
// `q.max(0).min(255)` instead of `q.clamp(0, 255)`: matches the
// SIMD `_mm{256,512}_max_ps(_mm{256,512}_min_ps(q, 255), 0)` NaN
// handling exactly so the wide and AVX{2,512} tiers produce
// bit-identical bytes. clamp() returns NaN for NaN input where
// the SIMD intrinsics return one of the limits; FMA on finite
// inputs is always finite so the difference is unreachable in
// the encode pipeline, but the parity tests assert byte-equality
// across all tiers and the two-step formulation keeps that
// pre-condition independent of the input domain.
#[allow(clippy::manual_clamp)]
fn sq8_encode_row_wide(row: &[f32], inv_scale: &[f32], c2: &[f32], dst: &mut [u8]) {
    use wide::f32x8;
    let dim = row.len();
    let zero = f32x8::splat(0.0);
    let max255 = f32x8::splat(SQ8_CODE_MAX);
    let mut i = 0;
    while i + F32X8_LANES <= dim {
        let r: [f32; F32X8_LANES] = row[i..i + F32X8_LANES].try_into().expect("len 8");
        let inv: [f32; F32X8_LANES] = inv_scale[i..i + F32X8_LANES].try_into().expect("len 8");
        let c: [f32; F32X8_LANES] = c2[i..i + F32X8_LANES].try_into().expect("len 8");
        let q = f32x8::from(r).mul_add(f32x8::from(inv), f32x8::from(c));
        let q_clamped = q.fast_max(zero).fast_min(max255).to_array();
        for k in 0..F32X8_LANES {
            dst[i + k] = q_clamped[k] as u8;
        }
        i += F32X8_LANES;
    }
    while i < dim {
        let q = row[i].mul_add(inv_scale[i], c2[i]);
        let q_clamped = q.max(0.0).min(SQ8_CODE_MAX);
        dst[i] = q_clamped as u8;
        i += 1;
    }
}

/// AVX2 Sq8 encode. 8 lanes per iteration: `VFMADD213PS` →
/// `VMINPS`/`VMAXPS` clamp → `VCVTTPS2DQ` truncate-to-i32 →
/// `VPACKUSDW` (i32 → u16 saturating) → `VPACKUSWB`
/// (u16 → u8 saturating) → 8-byte store. The saturating narrow
/// is a no-op because we already clamped to `[0, 255]` in float
/// space; it's the cheapest path that produces packed u8.
///
/// On the question of "why pack at all when the wide path just
/// does `q_clamped[k] as u8` per lane": the wide path is doing
/// the same thing — it just spills the f32x8 to a stack array
/// and runs eight per-lane scalar narrows + eight scalar stores.
/// `VPACKUSDW` + `VPACKUSWB` is the SIMD-native way to do that
/// narrow, and the final `_mm_cvtsi128_si64` + `write_unaligned`
/// is one 8-byte store instead of eight 1-byte stores. Same
/// arithmetic, vectorized.
///
/// # Safety
///
/// Callers must ensure the target supports `avx2` and `fma`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::manual_clamp)] // see sq8_encode_row_wide note
unsafe fn sq8_encode_row_avx2(row: &[f32], inv_scale: &[f32], c2: &[f32], dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let dim = row.len();
    let zero = _mm256_setzero_ps();
    let max255 = _mm256_set1_ps(SQ8_CODE_MAX);
    let mut i = 0;
    // SAFETY: each iteration reads 8 f32 lanes from `row`,
    // `inv_scale`, `c2` at offset `i` and writes 8 bytes to
    // `dst[i..i+8]`. `i + 8 <= dim` bounds every load and store.
    unsafe {
        while i + F32X8_LANES <= dim {
            let r = _mm256_loadu_ps(row.as_ptr().add(i));
            let inv = _mm256_loadu_ps(inv_scale.as_ptr().add(i));
            let c = _mm256_loadu_ps(c2.as_ptr().add(i));
            // q = x * inv + c  (single rounding)
            let q = _mm256_fmadd_ps(r, inv, c);
            // clamp to [0, 255]
            let q_clamped = _mm256_max_ps(_mm256_min_ps(q, max255), zero);
            // truncate-to-i32; lanes are in [0, 255] so the i32
            // values are non-negative and PACKUSDW interprets them
            // as unsigned correctly.
            let q_i32 = _mm256_cvttps_epi32(q_clamped);
            // Narrow 8 × i32 → 8 × u8 via two saturating packs.
            // `_mm256_extracti128_si256::<1>(q_i32)` is the Rust
            // stable intrinsic — the imm8 lane index is a const
            // generic argument (turbofish), not a runtime arg, so
            // the call site reads as one runtime parameter.
            let lo = _mm256_castsi256_si128(q_i32);
            let hi = _mm256_extracti128_si256::<1>(q_i32);
            let packed_u16 = _mm_packus_epi32(lo, hi); // 8 × u16
            let packed_u8 = _mm_packus_epi16(packed_u16, packed_u16); // low 8 bytes valid
            // Store the low 64 bits (8 bytes). Unaligned.
            let dst_ptr = dst.as_mut_ptr().add(i) as *mut i64;
            write_unaligned(dst_ptr, _mm_cvtsi128_si64(packed_u8));
            i += F32X8_LANES;
        }
    }
    while i < dim {
        let q = row[i].mul_add(inv_scale[i], c2[i]);
        let q_clamped = q.max(0.0).min(SQ8_CODE_MAX);
        dst[i] = q_clamped as u8;
        i += 1;
    }
}

/// AVX-512 Sq8 encode. 16 lanes per iteration: `VFMADD213PS` →
/// `VMINPS`/`VMAXPS` clamp → `VCVTTPS2DQ` truncate-to-i32 →
/// `VPMOVUSDB` (single-instruction unsigned-saturating narrow
/// i32 → u8) → 16-byte store. The narrow is one instruction
/// instead of the two pack steps the AVX2 path needs, which is
/// where the AVX-512 path picks up its extra factor over AVX2
/// on top of the 2× lane count.
///
/// # Safety
///
/// Callers must ensure the target supports `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::manual_clamp)] // see sq8_encode_row_wide note
unsafe fn sq8_encode_row_avx512(row: &[f32], inv_scale: &[f32], c2: &[f32], dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let dim = row.len();
    let zero = _mm512_setzero_ps();
    let max255 = _mm512_set1_ps(SQ8_CODE_MAX);
    let mut i = 0;
    // SAFETY: each iteration reads 16 f32 lanes from `row`,
    // `inv_scale`, `c2` at offset `i` and writes 16 bytes to
    // `dst[i..i+16]`. `i + 16 <= dim` bounds every load and store.
    unsafe {
        while i + AVX512_F32_LANES <= dim {
            let r = _mm512_loadu_ps(row.as_ptr().add(i));
            let inv = _mm512_loadu_ps(inv_scale.as_ptr().add(i));
            let c = _mm512_loadu_ps(c2.as_ptr().add(i));
            let q = _mm512_fmadd_ps(r, inv, c);
            let q_clamped = _mm512_max_ps(_mm512_min_ps(q, max255), zero);
            let q_i32 = _mm512_cvttps_epi32(q_clamped);
            // Single-instruction unsigned-saturating narrow i32 → u8.
            // Result lanes are guaranteed in [0, 255] so the
            // saturation is a no-op; we use the unsigned variant
            // because it interprets the source as u32 (the i32
            // values are non-negative so the bit pattern matches).
            let packed_u8 = _mm512_cvtusepi32_epi8(q_i32); // __m128i (16 bytes)
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, packed_u8);
            i += AVX512_F32_LANES;
        }
    }
    // 8-lane half-tail for `dim % 16 == 8`.
    #[cfg(target_arch = "x86_64")]
    if i + F32X8_LANES <= dim {
        // SAFETY: target_feature `avx512f` implies `avx2` + `fma`
        // are available; we just call the AVX2 path on the
        // remaining 8 lanes (no need to re-emit the same code).
        unsafe { sq8_encode_row_avx2_unsafe_tail8(row, inv_scale, c2, dst, &mut i) };
    }
    while i < dim {
        let q = row[i].mul_add(inv_scale[i], c2[i]);
        let q_clamped = q.max(0.0).min(SQ8_CODE_MAX);
        dst[i] = q_clamped as u8;
        i += 1;
    }
}

/// 8-lane Sq8 encode tail used by the AVX-512 kernel when
/// `dim % 16 == 8`. Same FMA + clamp + pack sequence as
/// `sq8_encode_row_avx2` but inlined so the AVX-512 entry point
/// keeps target_feature("avx512f") and we don't pay a tail call.
///
/// # Safety
///
/// Callers must ensure `avx2` + `fma` are available, and that
/// `*i + 8 <= row.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq8_encode_row_avx2_unsafe_tail8(
    row: &[f32],
    inv_scale: &[f32],
    c2: &[f32],
    dst: &mut [u8],
    i: &mut usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_ps();
    let max255 = _mm256_set1_ps(SQ8_CODE_MAX);
    // SAFETY: caller-guaranteed `*i + 8 <= row.len()`.
    unsafe {
        let r = _mm256_loadu_ps(row.as_ptr().add(*i));
        let inv = _mm256_loadu_ps(inv_scale.as_ptr().add(*i));
        let c = _mm256_loadu_ps(c2.as_ptr().add(*i));
        let q = _mm256_fmadd_ps(r, inv, c);
        let q_clamped = _mm256_max_ps(_mm256_min_ps(q, max255), zero);
        let q_i32 = _mm256_cvttps_epi32(q_clamped);
        let lo = _mm256_castsi256_si128(q_i32);
        let hi = _mm256_extracti128_si256::<1>(q_i32);
        let packed_u16 = _mm_packus_epi32(lo, hi);
        let packed_u8 = _mm_packus_epi16(packed_u16, packed_u16);
        let dst_ptr = dst.as_mut_ptr().add(*i) as *mut i64;
        write_unaligned(dst_ptr, _mm_cvtsi128_si64(packed_u8));
        *i += F32X8_LANES;
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use super::*;
    use crate::superfile::vector::rerank_codec::{
        SQ8_FIXED_OFFSET, SQ8_FIXED_RESIDUAL_DIVISOR, SQ8_FIXED_SCALE,
    };

    /// Deterministic synthetic Sq8 encode inputs spanning a
    /// realistic dynamic range: per-dim mins / maxes drawn from
    /// distinct centroid offsets so `(x - offset) / scale` exercises
    /// the full `[0, 255]` quantization range and the boundary
    /// rounding behaviour where SIMD and scalar paths could
    /// disagree by 1 ulp.
    fn synth_sq8_inputs(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 33) as u32) as f32 / (u32::MAX as f32)
        };
        let mut scale = vec![0.0f32; dim];
        let mut offset = vec![0.0f32; dim];
        let mut row = vec![0.0f32; dim];
        for d in 0..dim {
            offset[d] = -2.0 + 4.0 * next();
            let span = 0.1 + 4.0 * next();
            scale[d] = span / 255.0;
            // Mix of in-range, low-tail, and high-tail samples to
            // exercise clamp + rounding boundaries.
            let pick = next();
            row[d] = if pick < 0.05 {
                offset[d] - 0.5
            } else if pick > 0.95 {
                offset[d] + span + 0.5
            } else {
                offset[d] + span * next()
            };
        }
        (row, scale, offset)
    }

    /// Scalar reference Sq8 encode using the exact same FMA + clamp
    /// + truncating-cast sequence that `sq8_encode_row_*` use. This
    /// is the byte-pattern oracle the SIMD parity tests assert
    /// against — the on-disk Sq8 format is what this function
    /// produces, regardless of which tier the runtime picks.
    #[allow(clippy::manual_clamp)] // bit-match SIMD NaN handling
    fn sq8_encode_row_reference(row: &[f32], inv_scale: &[f32], c2: &[f32], dst: &mut [u8]) {
        debug_assert_eq!(row.len(), dst.len());
        for d in 0..row.len() {
            let q = row[d].mul_add(inv_scale[d], c2[d]);
            let q_clamped = q.max(0.0).min(255.0);
            dst[d] = q_clamped as u8;
        }
    }

    /// `Sq8EncodeConsts::from_scale_offset` produces the same
    /// `(inv_scale, c2)` arrays as the algebraic identity it claims
    /// to encode: `inv_scale[d] = 1/scale[d]` and
    /// `c2[d] = (-offset[d]) * inv_scale[d] + 0.5`.
    #[test]
    fn sq8_encode_consts_match_algebraic_identity() {
        let dim = 384;
        let (_row, scale, offset) = synth_sq8_inputs(dim, 0xC0FFEE);
        let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);
        assert_eq!(consts.inv_scale.len(), dim);
        assert_eq!(consts.c2.len(), dim);
        for d in 0..dim {
            let want_inv = 1.0 / scale[d];
            let want_c2 = (-offset[d]).mul_add(want_inv, 0.5);
            assert_eq!(consts.inv_scale[d], want_inv);
            assert_eq!(consts.c2[d], want_c2);
        }
    }

    /// Update-min/max parity: every tier converges on the same
    /// per-dim (min, max) across a sweep of dims that hits the
    /// 16-, 8-, and scalar-tail paths.
    #[test]
    fn update_min_max_simd_paths_match_scalar() {
        let dims = [
            1, 7, 8, 15, 16, 17, 24, 31, 32, 33, 47, 48, 96, 384, 512, 1023,
        ];
        for &dim in &dims {
            let (row, _scale, _offset) = synth_sq8_inputs(dim, dim as u64 * 13);
            let mut mn_ref = vec![f32::INFINITY; dim];
            let mut mx_ref = vec![f32::NEG_INFINITY; dim];
            for d in 0..dim {
                if row[d] < mn_ref[d] {
                    mn_ref[d] = row[d];
                }
                if row[d] > mx_ref[d] {
                    mx_ref[d] = row[d];
                }
            }

            for tier in ["wide", "avx2", "avx512"] {
                let mut mn = vec![f32::INFINITY; dim];
                let mut mx = vec![f32::NEG_INFINITY; dim];
                match tier {
                    "wide" => update_min_max_wide(&row, &mut mn, &mut mx),
                    #[cfg(target_arch = "x86_64")]
                    "avx2" if std::is_x86_feature_detected!("avx2") => {
                        unsafe { update_min_max_avx2(&row, &mut mn, &mut mx) };
                    }
                    #[cfg(target_arch = "x86_64")]
                    "avx512" if std::is_x86_feature_detected!("avx512f") => {
                        unsafe { update_min_max_avx512(&row, &mut mn, &mut mx) };
                    }
                    _ => continue,
                };
                assert_eq!(mn, mn_ref, "tier {} min mismatch at dim {}", tier, dim);
                assert_eq!(mx, mx_ref, "tier {} max mismatch at dim {}", tier, dim);
            }
        }
    }

    /// Sq8 SIMD encode parity: every tier produces byte-identical
    /// codes to the FMA-based scalar reference across a sweep of
    /// dims that hits all per-tier tail paths.
    #[test]
    fn sq8_encode_row_simd_paths_match_scalar() {
        let dims = [
            1, 7, 8, 15, 16, 17, 24, 31, 32, 33, 47, 48, 96, 384, 512, 1023,
        ];
        for &dim in &dims {
            let (row, scale, offset) = synth_sq8_inputs(dim, dim as u64 * 17 + 1);
            let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);
            let mut dst_ref = vec![0u8; dim];
            sq8_encode_row_reference(&row, &consts.inv_scale, &consts.c2, &mut dst_ref);

            // Wide tier — universally available.
            let mut dst_wide = vec![0u8; dim];
            sq8_encode_row_wide(&row, &consts.inv_scale, &consts.c2, &mut dst_wide);
            assert_eq!(dst_wide, dst_ref, "wide path mismatch at dim {}", dim);

            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                let mut dst_avx2 = vec![0u8; dim];
                unsafe {
                    sq8_encode_row_avx2(&row, &consts.inv_scale, &consts.c2, &mut dst_avx2);
                }
                assert_eq!(dst_avx2, dst_ref, "avx2 path mismatch at dim {}", dim);
            }
            #[cfg(target_arch = "x86_64")]
            if std::is_x86_feature_detected!("avx512f") {
                let mut dst_avx512 = vec![0u8; dim];
                unsafe {
                    sq8_encode_row_avx512(&row, &consts.inv_scale, &consts.c2, &mut dst_avx512);
                }
                assert_eq!(dst_avx512, dst_ref, "avx512 path mismatch at dim {}", dim);
            }
        }
    }

    #[test]
    fn fixed_residual_grid_round_trips_with_one_step_error() {
        let row = [-1.0, 1.0, -0.5, 0.0, 0.5, 0.123_456, -0.987_654];
        let dim = row.len();
        let scale = vec![SQ8_FIXED_SCALE; dim];
        let offset = vec![SQ8_FIXED_OFFSET; dim];
        let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);
        let mut codes = vec![0; dim];
        let mut residuals = vec![0; dim];
        let mut recon = vec![0.0; dim];
        let norm = encode_sq8_residual_row(
            &row,
            &consts,
            &scale,
            &offset,
            &mut codes,
            &mut residuals,
            &mut recon,
            true,
            SQ8_FIXED_RESIDUAL_DIVISOR,
        )
        .expect("norm");
        assert_eq!((codes[0], residuals[0]), (0, 0));
        assert_eq!((codes[1], residuals[1]), (u8::MAX, 0));
        let step = SQ8_FIXED_SCALE / SQ8_FIXED_RESIDUAL_DIVISOR;
        for (actual, decoded) in row.iter().zip(&recon) {
            assert!((actual - decoded).abs() <= step * 1.01);
        }
        let expected_norm: f32 = recon.iter().map(|value| value * value).sum();
        assert!((norm - expected_norm).abs() <= f32::EPSILON * expected_norm.max(1.0));
        assert!(
            residuals
                .iter()
                .all(|byte| *byte != i8::MIN.to_le_bytes()[0])
        );
    }

    /// Microbench: per-tier Sq8 encode throughput across the same
    /// dim grid the AVX-512 / AVX2 distance microbenches use. Gated
    /// `#[ignore]` so `cargo test --release` skips it; run via
    /// `cargo test --release --lib sq8_encode_microbench --
    /// --ignored --nocapture` to print a markdown table.
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    fn sq8_encode_microbench() {
        use std::time::Instant;
        let dims: &[usize] = &[128, 384, 768, 1024, 1536];
        let iters: usize = 200_000;

        println!("\n### Sq8 f32 → u8 encode per-tier ns / row (dim sweep)\n");
        println!("| dim | scalar ns | wide ns | avx2 ns | avx512 ns |");
        println!("|----:|----------:|--------:|--------:|----------:|");

        for &dim in dims {
            let (row, scale, offset) = synth_sq8_inputs(dim, dim as u64 * 23 + 5);
            let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);
            let mut dst = vec![0u8; dim];

            // Scalar reference — what the on-disk encode would
            // be without any of this commit's SIMD work.
            let t0 = Instant::now();
            for _ in 0..iters {
                sq8_encode_row_reference(&row, &consts.inv_scale, &consts.c2, &mut dst);
                black_box(&dst);
            }
            let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

            let t0 = Instant::now();
            for _ in 0..iters {
                sq8_encode_row_wide(&row, &consts.inv_scale, &consts.c2, &mut dst);
                black_box(&dst);
            }
            let wide_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

            #[cfg(target_arch = "x86_64")]
            let avx2_ns =
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    let t0 = Instant::now();
                    for _ in 0..iters {
                        unsafe {
                            sq8_encode_row_avx2(&row, &consts.inv_scale, &consts.c2, &mut dst);
                        }
                        black_box(&dst);
                    }
                    Some(t0.elapsed().as_nanos() as f64 / iters as f64)
                } else {
                    None
                };
            #[cfg(not(target_arch = "x86_64"))]
            let avx2_ns: Option<f64> = None;

            #[cfg(target_arch = "x86_64")]
            let avx512_ns = if std::is_x86_feature_detected!("avx512f") {
                let t0 = Instant::now();
                for _ in 0..iters {
                    unsafe {
                        sq8_encode_row_avx512(&row, &consts.inv_scale, &consts.c2, &mut dst);
                    }
                    black_box(&dst);
                }
                Some(t0.elapsed().as_nanos() as f64 / iters as f64)
            } else {
                None
            };
            #[cfg(not(target_arch = "x86_64"))]
            let avx512_ns: Option<f64> = None;

            let fmt = |x: Option<f64>| match x {
                Some(v) => format!("{:>7.1}", v),
                None => "    n/a".to_string(),
            };
            println!(
                "| {:>3} | {:>9.1} | {:>7.1} | {} | {} |",
                dim,
                scalar_ns,
                wide_ns,
                fmt(avx2_ns),
                fmt(avx512_ns),
            );
        }
    }
}
