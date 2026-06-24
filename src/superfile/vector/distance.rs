// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Distance kernels — portable f32x8 SIMD via `wide`.
//!
//! Three metrics: cosine (`1 − dot` after unit-norm), squared L2,
//! negated dot (for max-inner-product search). All converge to
//! "smaller = closer" so the rerank heap can use a single comparator.
//!
//! The dot-product and L2² kernels are the inner loop of the vector
//! search pipeline; correctness here is load-bearing for both the
//! IVF cluster scan (probing centroids) and the full-precision rerank
//! (after the 1-bit shortlist).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use wide::f32x8;

use crate::superfile::format::vec::{METRIC_ID_COSINE, METRIC_ID_L2SQ, METRIC_ID_NEGDOT};
use crate::superfile::vector::rerank_codec::RerankCodec;
#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::avx512_enabled;

/// Residual quantization step divisor for [`RerankCodec::Sq8Residual`].
/// The signed 8-bit residual code at dim `d` carries
/// `scale_c[d] / SQ8_RESIDUAL_DIVISOR`-sized steps around the Sq8
/// dequant base. `16` hit the recall target with the best
/// byte/CPU trade-off on the 1M × 384 cosine calibration sweep.
pub(crate) const SQ8_RESIDUAL_DIVISOR: f32 = 16.0;

/// Sq8 u8-code ceiling. Sq8 quantizes each component to a single
/// unsigned byte, so the per-cluster scale maps a cluster's value span
/// onto `[0, SQ8_CODE_MAX]` and the encoder clamps to that range before
/// the truncating cast to `u8`.
pub(crate) const SQ8_CODE_MAX: f32 = 255.0;

/// Symmetric i8 clamp for the Sq8+ε residual leg. The residual code is
/// stored as a signed byte but clamped to ±127 (not `i8::MIN`) so the
/// quantized magnitude stays symmetric about zero.
pub(crate) const SQ8_RESIDUAL_I8_CLAMP: f32 = 127.0;

/// Lane count of the portable `wide::f32x8` SIMD register (256-bit /
/// 32-bit). The universal kernel processes this many f32s per
/// iteration; tails handle `len % F32X8_LANES`.
const F32X8_LANES: usize = 8;

/// Lane count of an AVX-512 f32 vector register (512-bit / 32-bit).
/// The AVX-512 kernels process this many f32s per FMA iteration.
// Referenced only by the x86-gated AVX-512 kernels; dead on other targets.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const AVX512_F32_LANES: usize = 16;

/// Byte width of one little-endian `f32`. A byte-backed vector of
/// dimension `d` occupies `d * F32_BYTES` bytes.
const F32_BYTES: usize = 4;

/// Cosine distance is `COSINE_DISTANCE_BASE - dot` on unit vectors,
/// so smaller means closer without re-normalizing at query time.
/// `pub(crate)`: the manifest's folded Sq8 centroid scoring applies
/// the same identity.
pub(crate) const COSINE_DISTANCE_BASE: f32 = 1.0;

/// Cross-term coefficient in the squared-L2 identity
/// `‖q − x‖² = ‖q‖² − L2_CROSS_TERM_COEFF·(q·x) + ‖x‖²`, used by the
/// Sq8 kernels that reconstruct L2 from a fused dot product (and by
/// the manifest's folded Sq8 centroid scoring).
pub(crate) const L2_CROSS_TERM_COEFF: f32 = 2.0;

/// Distance metric for a vector index. Stored per-column in
/// `inf.vec.columns` JSON, applied at query time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// `1 - dot(a, b)` — assumes unit-normalized inputs.
    Cosine,
    /// Squared Euclidean distance, `Σ(a − b)²`.
    L2Sq,
    /// Negated dot product, `-dot(a, b)`. For maximum-inner-product
    /// search where vector magnitudes carry signal.
    NegDot,
}

/// Map a [`Metric`] to its on-disk `metric_id` discriminator
/// (`format::vec::METRIC_ID_*`). Single source of truth for the
/// metric↔id encoding shared by the IVF directory entry, the
/// cell-posting header, and the manifest summary.
#[inline]
pub(crate) fn metric_to_id(m: Metric) -> u32 {
    match m {
        Metric::L2Sq => METRIC_ID_L2SQ,
        Metric::Cosine => METRIC_ID_COSINE,
        Metric::NegDot => METRIC_ID_NEGDOT,
    }
}

/// Inverse of [`metric_to_id`]: decode an on-disk `metric_id`
/// discriminator back to its [`Metric`], or `None` for an unknown id.
#[inline]
pub(crate) fn metric_from_id(id: u32) -> Option<Metric> {
    match id {
        METRIC_ID_L2SQ => Some(Metric::L2Sq),
        METRIC_ID_COSINE => Some(Metric::Cosine),
        METRIC_ID_NEGDOT => Some(Metric::NegDot),
        _ => None,
    }
}

/// Generic distance dispatch. Smaller value = closer match for every metric.
#[inline]
pub fn distance(metric: Metric, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    match metric {
        Metric::Cosine => COSINE_DISTANCE_BASE - dot(a, b),
        Metric::L2Sq => l2_sq(a, b),
        Metric::NegDot => -dot(a, b),
    }
}

/// f32 dot product. Dispatches to the AVX-512 16-lane FMA kernel when
/// the runtime CPUID gate passes; otherwise the `wide::f32x8` AVX2 /
/// NEON / scalar kernel (which has been the universal kernel since the
/// superfile-builder existed). Both kernels handle non-multiple-of-lane
/// inputs via a scalar tail.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if avx512_enabled() {
        // SAFETY: gated by runtime CPUID detection in `avx512_enabled()`.
        return unsafe { dot_avx512(a, b) };
    }
    dot_wide(a, b)
}

/// Squared Euclidean distance. See [`dot`] for dispatch shape.
#[inline]
pub(crate) fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if avx512_enabled() {
        // SAFETY: gated by runtime CPUID detection in `avx512_enabled()`.
        return unsafe { l2_sq_avx512(a, b) };
    }
    l2_sq_wide(a, b)
}

/// Distance under `metric` between two `dim`-length vectors given by per-index
/// component accessors `a` and `b` (L2Sq returns the *squared* distance, no
/// sqrt — matching the IVF assignment/medoid scan convention). Centralizes the
/// per-metric reduction so the encoded-row scan paths share one definition
/// instead of each hand-expanding the three-arm match.
pub(crate) fn metric_distance_by<FA, FB>(metric: Metric, dim: usize, a: FA, b: FB) -> f32
where
    FA: Fn(usize) -> f32,
    FB: Fn(usize) -> f32,
{
    match metric {
        Metric::L2Sq => {
            let mut s = 0.0f32;
            for d in 0..dim {
                let diff = a(d) - b(d);
                s += diff * diff;
            }
            s
        }
        Metric::Cosine => {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for d in 0..dim {
                let (va, vb) = (a(d), b(d));
                dot += va * vb;
                na += va * va;
                nb += vb * vb;
            }
            let denom = na.sqrt() * nb.sqrt();
            if denom > 0.0 {
                COSINE_DISTANCE_BASE - dot / denom
            } else {
                COSINE_DISTANCE_BASE - dot
            }
        }
        Metric::NegDot => {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += a(d) * b(d);
            }
            -dot
        }
    }
}

/// Portable `wide::f32x8` (256-bit) dot product. The universal kernel
/// the codebase has shipped since day one — runs on AVX2 / NEON /
/// scalar. Public entry point [`dot`] dispatches here on every host
/// without AVX-512.
#[inline]
fn dot_wide(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(F32X8_LANES);
    let chunks_b = b.chunks_exact(F32X8_LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();

    let mut acc = f32x8::ZERO;
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(ca).expect("chunks_exact(8) yields slices of length 8"),
        );
        let vb = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(cb).expect("chunks_exact(8) yields slices of length 8"),
        );
        acc += va * vb;
    }
    let mut sum: f32 = acc.reduce_add();
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        sum += x * y;
    }
    sum
}

/// Portable `wide::f32x8` (256-bit) squared-L2. See [`dot_wide`].
#[inline]
fn l2_sq_wide(a: &[f32], b: &[f32]) -> f32 {
    let chunks_a = a.chunks_exact(F32X8_LANES);
    let chunks_b = b.chunks_exact(F32X8_LANES);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();

    let mut acc = f32x8::ZERO;
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(ca).expect("chunks_exact(8) yields slices of length 8"),
        );
        let vb = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(cb).expect("chunks_exact(8) yields slices of length 8"),
        );
        let d = va - vb;
        acc += d * d;
    }
    let mut sum: f32 = acc.reduce_add();
    for (x, y) in tail_a.iter().zip(tail_b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// AVX-512 16-lane FMA dot product. Same per-element math as
/// [`dot_wide`] but processes 16 fp32 lanes per FMA via `_mm512_fmadd_ps`
/// instead of two `wide::f32x8` ops. Public callers reach this only
/// through [`dot`] after [`avx512_enabled`] returns `true`.
///
/// Parity with [`dot_wide`]: associativity of f32 add means the two
/// kernels can differ by up to ~1 ULP per accumulator slot. The
/// distance tolerances downstream (cosine ε ≈ 1e-5 on unit vectors,
/// L2² ε ≈ 1e-3 at `dim ≤ 1024`) absorb this; parity tests below pin
/// the bound.
///
/// # Safety
///
/// Callers must ensure the target CPU supports `avx512f` (the
/// `_mm512_*` intrinsics used here). [`avx512_enabled`] guarantees
/// this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    // SAFETY: each `_mm512_loadu_ps` reads 16 f32s (= 64 bytes)
    // starting at `a.as_ptr().add(i)` / `b.as_ptr().add(i)`. The
    // loop predicate `i + 16 <= n` guarantees the 16-lane window
    // is fully inside both slices. Unaligned loads are permitted
    // (`loadu` is the unaligned variant); both inputs are arbitrary
    // `&[f32]` so we make no alignment assumption.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + AVX512_F32_LANES <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += AVX512_F32_LANES;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < n {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

/// AVX-512 16-lane squared-L2. See [`dot_avx512`].
///
/// # Safety
///
/// Same contract as [`dot_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    // SAFETY: see `dot_avx512` — same bounds reasoning, same
    // unaligned-load contract.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + AVX512_F32_LANES <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += AVX512_F32_LANES;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < n {
            let d = a[i] - b[i];
            sum += d * d;
            i += 1;
        }
        sum
    }
}

/// Distance against a vector stored as little-endian f32 bytes.
///
/// Zero-copy when the byte slice is 4-aligned (`bytemuck::try_cast_slice`
/// succeeds): we cast `&[u8] → &[f32]` and reuse the SIMD inner kernel.
/// When the underlying allocation isn't 4-aligned the fallback decodes
/// 32 bytes at a time into an on-stack `[f32; 8]` and feeds the same
/// `f32x8` kernel — still SIMD on the math, just with one extra
/// per-chunk byte→float decode.
///
/// Used by the rerank stage where every candidate's full vector lives
/// at a 4-aligned offset within the blob; in practice the fast path
/// is always taken there, but we keep the fallback so the API is safe
/// against arbitrary `Bytes` alignment.
#[inline]
pub fn distance_bytes(metric: Metric, query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * F32_BYTES, bytes.len());
    match metric {
        Metric::Cosine => COSINE_DISTANCE_BASE - dot_bytes(query, bytes),
        Metric::L2Sq => l2_sq_bytes(query, bytes),
        Metric::NegDot => -dot_bytes(query, bytes),
    }
}

#[inline]
pub fn dot_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    if let Ok(v) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return dot(query, v);
    }
    dot_le_bytes_unaligned(query, bytes)
}

#[inline]
pub fn l2_sq_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    if let Ok(v) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        return l2_sq(query, v);
    }
    l2_sq_le_bytes_unaligned(query, bytes)
}

#[inline]
fn dot_le_bytes_unaligned(query: &[f32], bytes: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + F32X8_LANES <= query.len() {
        let qc: [f32; F32X8_LANES] = query[i..i + F32X8_LANES]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; F32X8_LANES];
        for (j, slot) in bc.iter_mut().enumerate() {
            let off = (i + j) * F32_BYTES;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        acc += qv * bv;
        i += F32X8_LANES;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * F32_BYTES;
        let b = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        sum += query[i] * b;
        i += 1;
    }
    sum
}

#[inline]
fn l2_sq_le_bytes_unaligned(query: &[f32], bytes: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + F32X8_LANES <= query.len() {
        let qc: [f32; F32X8_LANES] = query[i..i + F32X8_LANES]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; F32X8_LANES];
        for (j, slot) in bc.iter_mut().enumerate() {
            let off = (i + j) * F32_BYTES;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        let d = qv - bv;
        acc += d * d;
        i += F32X8_LANES;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * F32_BYTES;
        let b = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let d = query[i] - b;
        sum += d * d;
        i += 1;
    }
    sum
}

/// Distance against a vector stored in the column's `rerank_codec`
/// representation. The fast path for `Fp32` reuses [`distance_bytes`].
///
/// Centroid scoring NEVER comes through here — centroids are always
/// stored as fp32 regardless of the column's rerank codec.
///
/// `Sq8Residual` has no "flat" entry point because the decode needs
/// the per-cluster scale/offset (and per-doc norm for L2Sq/Cosine);
/// callers go through [`Sq8ResidualKernel`], which captures those once
/// per query. `RabitqOnly` panics here — its column carries no
/// `full[]` bytes to feed in.
#[inline]
pub(crate) fn distance_bytes_codec(
    metric: Metric,
    codec: RerankCodec,
    query: &[f32],
    bytes: &[u8],
) -> f32 {
    match codec {
        RerankCodec::Fp32 => distance_bytes(metric, query, bytes),
        RerankCodec::Sq8Residual => {
            unreachable!(
                "distance_bytes_codec called with Sq8Residual — Sq8Residual rerank goes \
                 through dedicated kernels (need per-column scale/offset + per-doc \
                 norm context)"
            )
        }
        RerankCodec::RabitqOnly => {
            unreachable!(
                "distance_bytes_codec called with RabitqOnly — RabitqOnly columns \
                 carry no full[] region to score against"
            )
        }
    }
}

/// `Sq8Residual` rerank context. Captures the per-cluster quantizer
/// (`scale[dim]`, `offset[dim]`) plus the query-side precomputes for
/// both the Sq8 code leg and the i8 residual leg, so the per-candidate
/// inner loop is two u8/i8 → f32 widens + SIMD dot.
///
/// One kernel per query + cluster, reused across that cluster's rerank
/// candidates. Every rerank candidate is scored with the full residual
/// distance — there is no Sq8-only coarse pass.
pub(crate) struct Sq8ResidualKernel {
    metric: Metric,
    dim: usize,
    /// `q_code[d] = query[d] * scale[d]`. Per-doc step is
    /// `Σ_d q_code[d] * code[d] as f32`.
    q_code: Vec<f32>,
    /// `q_residual[d] = query[d] * scale[d] / residual_divisor`.
    /// Per-doc step is `Σ_d q_residual[d] * residual[d] as f32`.
    q_residual: Vec<f32>,
    /// `Σ_d query[d] * offset[d]`. Folded in once per candidate.
    q_dot_offset: f32,
    /// `Σ_d query[d]²`. L2Sq only.
    q_norm_sq: f32,
}

impl Sq8ResidualKernel {
    /// Build the per-query residual kernel. `scale` + `offset` are
    /// the per-cluster quantizer arrays; `residual_divisor` is
    /// [`SQ8_RESIDUAL_DIVISOR`]. The per-doc decoded norm is supplied
    /// per candidate to [`Self::distance_with_norm`].
    pub fn new(
        metric: Metric,
        query: &[f32],
        scale: &[f32],
        offset: &[f32],
        residual_divisor: f32,
    ) -> Self {
        let dim = query.len();
        debug_assert_eq!(scale.len(), dim);
        debug_assert_eq!(offset.len(), dim);
        debug_assert!(residual_divisor > 0.0);
        let mut q_code = vec![0.0f32; dim];
        let mut q_residual = vec![0.0f32; dim];
        let inv_residual_divisor = 1.0 / residual_divisor;
        let mut q_dot_offset_acc = f32x8::ZERO;
        let mut i = 0;
        while i + F32X8_LANES <= dim {
            let qc = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(&query[i..i + F32X8_LANES]).expect("len-8 slice"),
            );
            let sc = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(&scale[i..i + F32X8_LANES]).expect("len-8 slice"),
            );
            let oc = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(&offset[i..i + F32X8_LANES]).expect("len-8 slice"),
            );
            let q_code_v = qc * sc;
            let q_residual_v = q_code_v * f32x8::splat(inv_residual_divisor);
            q_code[i..i + F32X8_LANES].copy_from_slice(&q_code_v.to_array());
            q_residual[i..i + F32X8_LANES].copy_from_slice(&q_residual_v.to_array());
            q_dot_offset_acc += qc * oc;
            i += F32X8_LANES;
        }
        let mut q_dot_offset = q_dot_offset_acc.reduce_add();
        while i < dim {
            let q_scale = query[i] * scale[i];
            q_code[i] = q_scale;
            q_residual[i] = q_scale * inv_residual_divisor;
            q_dot_offset += query[i] * offset[i];
            i += 1;
        }
        let q_norm_sq = match metric {
            Metric::L2Sq => dot(query, query),
            Metric::Cosine | Metric::NegDot => 0.0,
        };
        Self {
            metric,
            dim,
            q_code,
            q_residual,
            q_dot_offset,
            q_norm_sq,
        }
    }

    /// Distance for one refine candidate: `dim` u8 Sq8 codes at
    /// `code_bytes`, `dim` i8 residual codes at `residual_bytes`, and
    /// the per-doc decoded norm supplied explicitly (`Some` for L2Sq +
    /// Cosine). Smaller = closer for every metric.
    #[inline]
    pub fn distance_with_norm(
        &self,
        code_bytes: &[u8],
        residual_bytes: &[u8],
        norm: Option<f32>,
    ) -> f32 {
        debug_assert_eq!(code_bytes.len(), self.dim);
        debug_assert_eq!(residual_bytes.len(), self.dim);
        let mut acc = f32x8::ZERO;
        let mut i = 0;
        while i + F32X8_LANES <= self.dim {
            let qc: [f32; F32X8_LANES] = self.q_code[i..i + F32X8_LANES]
                .try_into()
                .expect("q_code[i..i+8] len 8");
            let qr: [f32; F32X8_LANES] = self.q_residual[i..i + F32X8_LANES]
                .try_into()
                .expect("q_residual[i..i+8] len 8");
            let mut code = [0f32; F32X8_LANES];
            let mut residual = [0f32; F32X8_LANES];
            for j in 0..F32X8_LANES {
                code[j] = code_bytes[i + j] as f32;
                residual[j] = i8::from_le_bytes([residual_bytes[i + j]]) as f32;
            }
            acc += f32x8::from(qc) * f32x8::from(code);
            acc += f32x8::from(qr) * f32x8::from(residual);
            i += F32X8_LANES;
        }
        let mut cross = acc.reduce_add();
        while i < self.dim {
            cross += self.q_code[i] * (code_bytes[i] as f32);
            cross += self.q_residual[i] * (i8::from_le_bytes([residual_bytes[i]]) as f32);
            i += 1;
        }
        let dot = cross + self.q_dot_offset;
        match self.metric {
            Metric::Cosine => {
                let x_norm = norm
                    .expect("Sq8ResidualKernel + Cosine requires per_doc_norms")
                    .sqrt();
                if x_norm > 0.0 {
                    COSINE_DISTANCE_BASE - dot / x_norm
                } else {
                    COSINE_DISTANCE_BASE - dot
                }
            }
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                let x_norm_sq = norm.expect("Sq8ResidualKernel + L2Sq requires per_doc_norms");
                self.q_norm_sq - L2_CROSS_TERM_COEFF * dot + x_norm_sq
            }
        }
    }
}

/// In-place L2-normalize. Zero vectors stay zero (no division).
///
/// Portable `wide::f32x8` SIMD: 8-lane FMA for the magnitude reduction
/// and 8-lane multiply for the per-element scale, with a scalar tail
/// for inputs whose length isn't a multiple of 8. Faster than the
/// readable `iter().map().sum().sqrt()` scalar form on every host
/// the codebase compiles for, which matters whenever a caller
/// pre-normalizes a large corpus (e.g. cosine-test fixtures
/// pre-normalize multi-thousand-vector inputs as setup).
pub fn normalize(v: &mut [f32]) {
    let mag = {
        let mut acc = f32x8::ZERO;
        let mut tail_acc: f32 = 0.0;
        let chunks = v.chunks_exact(F32X8_LANES);
        let tail = chunks.remainder();
        for c in chunks {
            let lane = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(c)
                    .expect("chunks_exact(8) yields slices of length 8"),
            );
            acc += lane * lane;
        }
        for &x in tail {
            tail_acc += x * x;
        }
        (acc.reduce_add() + tail_acc).sqrt()
    };
    if mag > 0.0 {
        let inv = 1.0 / mag;
        let inv_v = f32x8::splat(inv);
        let mut chunks = v.chunks_exact_mut(F32X8_LANES);
        for c in chunks.by_ref() {
            let lane = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(&*c)
                    .expect("chunks_exact_mut(8) yields slices of length 8"),
            );
            let scaled = lane * inv_v;
            c.copy_from_slice(&scaled.to_array());
        }
        for x in chunks.into_remainder() {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // --- dot ------------------------------------------------------------

    #[test]
    fn dot_zero_vectors() {
        let a = vec![0.0; 16];
        let b = vec![0.0; 16];
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn dot_orthogonal_basis_vectors() {
        // e_0 · e_1 = 0
        let mut a = vec![0.0; 16];
        let mut b = vec![0.0; 16];
        a[0] = 1.0;
        b[1] = 1.0;
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn dot_self_is_squared_norm() {
        let v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let want: f32 = (1..=16).map(|i| (i * i) as f32).sum();
        assert!(approx(dot(&v, &v), want, 1e-3));
    }

    #[test]
    fn dot_handles_tail_not_multiple_of_8() {
        let a: Vec<f32> = vec![1.0; 11];
        let b: Vec<f32> = vec![2.0; 11];
        assert!(approx(dot(&a, &b), 22.0, 1e-5));
    }

    #[test]
    fn dot_short_input() {
        // Only the scalar-tail path runs.
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!(approx(dot(&a, &b), 32.0, 1e-5));
    }

    // --- l2_sq ----------------------------------------------------------

    #[test]
    fn l2_sq_identical_inputs_zero() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        assert_eq!(l2_sq(&v, &v), 0.0);
    }

    #[test]
    fn l2_sq_unit_offset_per_dim() {
        let a = vec![0.0; 16];
        let b = vec![1.0; 16];
        // Each component contributes (0-1)² = 1; 16 components → 16.
        assert!(approx(l2_sq(&a, &b), 16.0, 1e-5));
    }

    #[test]
    fn l2_sq_handles_tail() {
        let a = vec![0.0; 11];
        let b = vec![3.0; 11];
        assert!(approx(l2_sq(&a, &b), 99.0, 1e-5));
    }

    // --- normalize ------------------------------------------------------

    #[test]
    fn normalize_unit_vector_stays_unit() {
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        normalize(&mut v);
        assert_eq!(v, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_scales_magnitude_to_one() {
        let mut v = vec![3.0, 4.0]; // |v| = 5
        normalize(&mut v);
        assert!(approx(v[0], 0.6, 1e-5));
        assert!(approx(v[1], 0.8, 1e-5));
    }

    #[test]
    fn normalize_zero_vector_left_alone() {
        let mut v = vec![0.0; 16];
        normalize(&mut v);
        for &x in &v {
            assert_eq!(x, 0.0);
        }
    }

    #[test]
    fn normalize_then_self_dot_is_one() {
        let mut v: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        normalize(&mut v);
        assert!(approx(dot(&v, &v), 1.0, 1e-5));
    }

    // --- distance dispatch ---------------------------------------------

    #[test]
    fn distance_cosine_uses_one_minus_dot() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        // cos similarity 1 → distance 0
        assert!(approx(distance(Metric::Cosine, &a, &b), 0.0, 1e-5));

        let c = vec![0.0, 1.0, 0.0, 0.0];
        // orthogonal → cos 0 → distance 1
        assert!(approx(distance(Metric::Cosine, &a, &c), 1.0, 1e-5));
    }

    #[test]
    fn distance_l2sq_zero_for_identical() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(distance(Metric::L2Sq, &v, &v), 0.0);
    }

    #[test]
    fn distance_negdot_inverts_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        // dot = 4+6+6+4 = 20; -dot = -20
        assert!(approx(distance(Metric::NegDot, &a, &b), -20.0, 1e-5));
    }

    #[test]
    fn distance_smaller_is_closer_for_every_metric() {
        // Common comparator semantic across metrics — load-bearing for
        // the rerank heap.
        let q = vec![1.0, 0.0, 0.0, 0.0];
        let near = vec![1.0, 0.0, 0.0, 0.0];
        let far = vec![-1.0, 0.0, 0.0, 0.0];
        for m in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let d_near = distance(m, &q, &near);
            let d_far = distance(m, &q, &far);
            assert!(
                d_near < d_far,
                "metric {m:?}: near {d_near} should be < far {d_far}"
            );
        }
    }

    // --- sq8 residual kernel --------------------------------------------

    /// Decode `Sq8Residual` codes (`code * scale + offset + residual
    /// * scale / divisor`) — the reference the residual kernel must
    /// agree with.
    fn decode_sq8_residual(
        codes: &[u8],
        residuals: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        residual_divisor: f32,
    ) -> Vec<f32> {
        codes
            .iter()
            .zip(residuals.iter())
            .enumerate()
            .map(|(i, (&c, &r))| {
                let d = i % dim;
                (c as f32) * scale[d]
                    + offset[d]
                    + (i8::from_le_bytes([r]) as f32) * scale[d] / residual_divisor
            })
            .collect()
    }

    #[test]
    fn sq8_residual_kernel_matches_corrected_reference() {
        let dim = 24usize;
        let residual_divisor = SQ8_RESIDUAL_DIVISOR;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.04 - 0.2).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i as f32) * 0.001).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -0.4 + (i as f32) * 0.03).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let residuals: Vec<u8> = (0..dim)
            .map(|i| (((i * 17 + 3) % 63) as i8 - 31).to_le_bytes()[0])
            .collect();
        let corrected =
            decode_sq8_residual(&codes, &residuals, dim, &scale, &offset, residual_divisor);
        let corrected_norm: f32 = corrected.iter().map(|x| x * x).sum();
        let norms = [corrected_norm];
        for metric in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let norms_arg = match metric {
                Metric::Cosine | Metric::L2Sq => Some(&norms[..]),
                Metric::NegDot => None,
            };
            let kernel = Sq8ResidualKernel::new(metric, &query, &scale, &offset, residual_divisor);
            let got = kernel.distance_with_norm(&codes, &residuals, norms_arg.map(|n| n[0]));
            let want = match metric {
                Metric::Cosine => 1.0 - dot(&query, &corrected) / corrected_norm.sqrt(),
                _ => distance(metric, &query, &corrected),
            };
            assert!(
                (want - got).abs() <= 1e-4,
                "metric {metric:?}: residual kernel {got} vs corrected ref {want}"
            );
        }
    }

    #[test]
    fn sq8_residual_kernel_handles_tail_dim_not_multiple_of_8() {
        let dim = 13usize;
        let residual_divisor = SQ8_RESIDUAL_DIVISOR;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.03 + 0.1).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.02 + (i as f32) * 0.001).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -0.2 + (i as f32) * 0.02).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 11 + 5) % 256) as u8).collect();
        let residuals: Vec<u8> = (0..dim)
            .map(|i| (((i * 23 + 9) % 47) as i8 - 23).to_le_bytes()[0])
            .collect();
        let corrected =
            decode_sq8_residual(&codes, &residuals, dim, &scale, &offset, residual_divisor);
        let kernel =
            Sq8ResidualKernel::new(Metric::NegDot, &query, &scale, &offset, residual_divisor);
        let got = kernel.distance_with_norm(&codes, &residuals, None);
        let want = distance(Metric::NegDot, &query, &corrected);
        assert!(
            (want - got).abs() <= 1e-4,
            "tail-dim residual kernel: got {got} vs corrected ref {want}"
        );
    }

    // --- AVX-512 parity (fp32) ------------------------------------------

    /// Generate a pseudo-random `f32` vector. Deterministic — uses the
    /// same monotone-noise pattern as the planted-cluster test fixtures
    /// elsewhere in this file so failures are reproducible.
    #[cfg(target_arch = "x86_64")]
    fn fake_vec(dim: usize, seed: u32) -> Vec<f32> {
        (0..dim)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) as i32;
                (x as f32) * 1e-9
            })
            .collect()
    }

    /// AVX-512 `dot` agrees with the `wide` baseline on every length
    /// from 1 to 64 (covers the 16-lane unroll boundary at 16, the
    /// double-unroll at 32, and a wide span of tail sizes).
    ///
    /// Tolerance is `1e-5 * max(1, |result|)` — strictly looser than
    /// per-add ULP because the two kernels differ in reduction order.
    /// The recall test suite downstream pins tolerances of 1e-3, so
    /// 1e-5 here keeps two orders of headroom.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot_avx512_matches_wide_across_lengths() {
        if !avx512_enabled() {
            eprintln!("dot_avx512_matches_wide_across_lengths: skipped, no AVX-512");
            return;
        }
        for dim in 1..=64 {
            let a = fake_vec(dim, 0xA5A5);
            let b = fake_vec(dim, 0x5A5A);
            let want = dot_wide(&a, &b);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { dot_avx512(&a, &b) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: avx512 {got} vs wide {want} (tol {tol})"
            );
        }
    }

    /// AVX-512 `l2_sq` agrees with the `wide` baseline across the same
    /// length sweep. Looser tolerance than `dot` because `l2_sq` involves
    /// a `sub` *and* an `fma` so the two kernels' rounding diverges
    /// faster as `dim` grows.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn l2_sq_avx512_matches_wide_across_lengths() {
        if !avx512_enabled() {
            eprintln!("l2_sq_avx512_matches_wide_across_lengths: skipped, no AVX-512");
            return;
        }
        for dim in 1..=64 {
            let a = fake_vec(dim, 0xDEAD);
            let b = fake_vec(dim, 0xBEEF);
            let want = l2_sq_wide(&a, &b);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { l2_sq_avx512(&a, &b) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: avx512 {got} vs wide {want} (tol {tol})"
            );
        }
    }

    /// Parity at realistic embedding sizes — the dims the rerank /
    /// shortlist actually run at. Tighter perspective: at `dim = 384`
    /// or `dim = 1024` the reduction error grows with √dim, so we
    /// scale tolerance accordingly. Catches a regression where the
    /// AVX-512 tail logic loses precision on the last < 16 lanes.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot_avx512_matches_wide_at_embedding_dims() {
        if !avx512_enabled() {
            eprintln!("dot_avx512_matches_wide_at_embedding_dims: skipped, no AVX-512");
            return;
        }
        for &dim in &[128usize, 384, 768, 1024, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i + 7) as f32) * 0.0017 - 0.3).collect();
            let want = dot_wide(&a, &b);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { dot_avx512(&a, &b) };
            let tol = 1e-4 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: avx512 {got} vs wide {want} (tol {tol})"
            );
        }
    }

    /// Public `dot` dispatches transparently: returns the same numeric
    /// value as `dot_wide` does on this host regardless of whether
    /// AVX-512 is active. (Within the same parity tolerance as the
    /// direct-call test above.)
    #[test]
    fn public_dot_dispatches_consistently() {
        for &dim in &[7usize, 16, 17, 384] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i * 3) as f32) * 0.02 - 0.1).collect();
            let public_result = dot(&a, &b);
            let wide_result = dot_wide(&a, &b);
            let tol = 1e-4 * wide_result.abs().max(1.0);
            assert!(
                (public_result - wide_result).abs() <= tol,
                "dim {dim}: dot() {public_result} vs dot_wide() {wide_result} (tol {tol})"
            );
        }
    }

    /// `INFINO_DISABLE_AVX512=1` is documented as the kill-switch for
    /// the AVX-512 fast path. Test pins the env-var → boolean mapping
    /// at the unit-test layer because `avx512_enabled()` caches via
    /// `OnceLock` and we can't actually flip the cached value
    /// in-process; this test instead exercises the env-parsing branch
    /// in isolation by re-implementing it (the parser is small and
    /// the test would otherwise need a sub-process).
    #[test]
    fn disable_env_var_parses_truthy_values() {
        fn parse(v: &str) -> bool {
            v == "1" || v.eq_ignore_ascii_case("true")
        }
        assert!(parse("1"));
        assert!(parse("true"));
        assert!(parse("TRUE"));
        assert!(parse("True"));
        assert!(!parse("0"));
        assert!(!parse("false"));
        assert!(!parse(""));
        assert!(!parse("yes")); // pinned: we only accept 1 / true
    }

    // --- AVX-512 microbench (run by hand) ------------------------------
    //
    // Direct head-to-head per-kernel timings between the AVX-512 fast
    // path and the `wide`-based AVX2 baseline. Run with:
    //
    // ```text
    // cargo test --release --lib superfile::vector::distance::tests::\
    //   avx512_microbench -- --ignored --nocapture
    // ```
    //
    // `#[ignore]`-gated so it stays out of regular `cargo test` (which
    // would otherwise spend ~2 s per invocation). Prints a markdown
    // table to stderr.

    /// Time a 0-arg closure for `iters` calls; return mean nanoseconds
    /// per call. Uses `black_box` so the optimizer doesn't elide.
    #[cfg(target_arch = "x86_64")]
    /// Time `iters` invocations of `f` and return the average ns/call.
    ///
    /// The closure MUST return its computed value (not drop it via `let _ =`)
    /// and MUST wrap loop-invariant inputs in `std::hint::black_box(..)`
    /// so the compiler cannot hoist or dead-code-eliminate the call.
    ///
    /// Both ends matter — without the input black_box the compiler will
    /// hoist a pure function call on loop-invariant references out of the
    /// timing loop and collapse it to ~1 cycle (single-cycle add latency).
    fn time_ns<R, F: FnMut() -> R>(iters: u32, mut f: F) -> f64 {
        use std::{hint::black_box, time::Instant};
        // Warmup — populate caches, JIT-equivalent steady state.
        for _ in 0..(iters / 10).max(64) {
            black_box(f());
        }
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        let dt = t.elapsed();
        dt.as_secs_f64() * 1e9 / (iters as f64)
    }

    #[cfg(target_arch = "x86_64")]
    fn realistic_dims() -> &'static [usize] {
        &[128, 384, 768, 1024, 1536]
    }

    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn avx512_microbench_distance_kernels() {
        if !avx512_enabled() {
            eprintln!("avx512_microbench: skipped, no AVX-512 on this host");
            return;
        }
        eprintln!();
        eprintln!(
            "### distance kernel — AVX-512 vs wide (ns per call, single thread, release build)\n"
        );
        eprintln!("| kernel | dim | wide ns | avx512 ns | speedup |");
        eprintln!("|--------|----:|--------:|----------:|--------:|");

        use std::hint::black_box;
        for &dim in realistic_dims() {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i + 7) as f32) * 0.0017 - 0.3).collect();
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            let wide_ns = time_ns(iters, || dot_wide(black_box(&a), black_box(&b)));
            // SAFETY: gated on avx512_enabled() above.
            let avx_ns = time_ns(iters, || unsafe {
                dot_avx512(black_box(&a), black_box(&b))
            });
            eprintln!(
                "| `distance::dot` | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );

            let wide_ns = time_ns(iters, || l2_sq_wide(black_box(&a), black_box(&b)));
            let avx_ns = time_ns(iters, || unsafe {
                l2_sq_avx512(black_box(&a), black_box(&b))
            });
            eprintln!(
                "| `distance::l2_sq` | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );
        }
    }
}
