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

use std::sync::Arc;

use wide::f32x8;

use crate::superfile::vector::rerank_codec::RerankCodec;
#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::{avx2_enabled, avx512_enabled};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Residual quantization step divisor for [`RerankCodec::Sq8ResidualEpsilon`].
/// The signed 8-bit residual code at dim `d` carries
/// `scale_c[d] / SQ8_RESIDUAL_DIVISOR`-sized steps around the Sq8
/// dequant base. `16` hit the recall target with the best
/// byte/CPU trade-off on the 1M × 384 cosine calibration sweep.
pub(crate) const SQ8_RESIDUAL_DIVISOR: f32 = 16.0;

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

/// Distance metric for a vector column. Stored per-column in
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
/// `Sq8` doesn't have a "flat" entry point because the decode needs
/// the per-column scale/offset (and per-doc norm for L2Sq). Sq8
/// callers go through [`Sq8Kernel`] which captures those once per
/// query. `None` panics here — its column carries no `full[]` bytes
/// to feed in.
#[inline]
pub(crate) fn distance_bytes_codec(
    metric: Metric,
    codec: RerankCodec,
    query: &[f32],
    bytes: &[u8],
) -> f32 {
    match codec {
        RerankCodec::Fp32 => distance_bytes(metric, query, bytes),
        RerankCodec::Sq8ResidualEpsilon => {
            unreachable!(
                "distance_bytes_codec called with Sq8ResidualEpsilon — Sq8ResidualEpsilon rerank goes \
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

/// Sq8 rerank context. Captures the per-column quantizer
/// (`scale[dim]` + `offset[dim]`), optional per-doc cached
/// decoded-norms (`Σ_d x_decoded²`, only populated for L2Sq),
/// and the per-query precomputes that fold scale/offset into
/// the query side so the per-doc inner loop is a plain u8→f32
/// widen + SIMD dot.
///
/// One kernel per query, reused across every rerank candidate.
/// The per-query precompute is two dim-passes (`q · scale`,
/// `q · offset`, plus `q · q` for L2Sq), amortized over
/// `k × rerank_mult` candidates so it costs ≪ 1 % of search time
/// at typical `rerank_mult = 256`.
pub(crate) struct Sq8Kernel {
    metric: Metric,
    dim: usize,
    /// `q_prime[d] = query[d] * scale[d]`. The per-doc inner
    /// step is `Σ_d q_prime[d] * code[d] as f32`.
    q_prime: Vec<f32>,
    /// `Σ_d query[d] * offset[d]`. Per-query constant — added
    /// once per candidate at the end of the inner reduction to
    /// recover `dot(query, x_decoded)`.
    q_dot_offset: f32,
    /// `Σ_d query[d]²`. L2Sq only — used in
    /// `dist = q_norm_sq − 2·dot + x_norm_sq[pos]`.
    q_norm_sq: f32,
    /// Optional per-doc `Σ_d x_decoded²` table, indexed by the
    /// rerank shortlist's `pos` field. `Some` for L2Sq columns,
    /// `None` for NegDot. `Some` for L2Sq (stores `‖x‖²`) and
    /// Cosine (stores `‖x‖²`; rerank divides by `√norm`). Shared by
    /// refcount (`Arc`) so the kernel is `'static` and can run on a
    /// rayon worker — no per-query copy.
    per_doc_norms: Option<Arc<[f32]>>,
}

impl Sq8Kernel {
    /// Build the per-query kernel. `scale` + `offset` are the
    /// per-dim quantizer arrays from the column's `codec_meta`.
    /// `per_doc_norms` is `Some` for L2Sq and Cosine columns.
    pub fn new(
        metric: Metric,
        query: &[f32],
        scale: &[f32],
        offset: &[f32],
        per_doc_norms: Option<Arc<[f32]>>,
    ) -> Self {
        let dim = query.len();
        debug_assert_eq!(scale.len(), dim);
        debug_assert_eq!(offset.len(), dim);
        // Build q_prime + q_dot_offset in one SIMD pass per
        // dim — both fold over the same query.
        let mut q_prime = vec![0.0f32; dim];
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
            let qp = qc * sc;
            // Write q_prime out as 8 f32s. `wide::f32x8::to_array`
            // is the safe accessor; the per-lane copy compiles to
            // a single 32-byte mov on AVX2.
            q_prime[i..i + F32X8_LANES].copy_from_slice(&qp.to_array());
            q_dot_offset_acc += qc * oc;
            i += F32X8_LANES;
        }
        let mut q_dot_offset: f32 = q_dot_offset_acc.reduce_add();
        while i < dim {
            q_prime[i] = query[i] * scale[i];
            q_dot_offset += query[i] * offset[i];
            i += 1;
        }
        // q_norm_sq is only needed for L2Sq, but it's cheap to
        // always compute — one extra `dim/8` SIMD reduce.
        let q_norm_sq = match metric {
            Metric::L2Sq => dot(query, query),
            Metric::Cosine | Metric::NegDot => 0.0,
        };
        Self {
            metric,
            dim,
            q_prime,
            q_dot_offset,
            q_norm_sq,
            per_doc_norms,
        }
    }

    /// Distance for one rerank candidate at position `pos`, with
    /// `dim` u8 codes at `code_bytes`. Smaller = closer for every
    /// metric (matches the [`distance`] dispatch convention).
    #[inline]
    pub fn distance_at(&self, pos: u32, code_bytes: &[u8]) -> f32 {
        let norm = self.per_doc_norms.as_ref().map(|norms| norms[pos as usize]);
        self.distance_with_norm(code_bytes, norm)
    }

    #[inline]
    pub fn distance_with_norm(&self, code_bytes: &[u8], norm: Option<f32>) -> f32 {
        debug_assert_eq!(code_bytes.len(), self.dim);
        // Per-doc inner reduction: Σ_d q_prime[d] * code[d] as f32.
        // Dispatches to AVX-512 (16-lane FMA with VPMOVZXBD widen)
        // when the runtime gate passes; otherwise the f32x8 widen-
        // and-FMA scalar-tier kernel.
        let qp_code_dot = sq8_dot(&self.q_prime, code_bytes, self.dim);
        // `dot(query, x_decoded) = qp_code_dot + q_dot_offset` because
        // x_decoded[d] = code[d] * scale[d] + offset[d], so
        // Σ_d q[d] * x_decoded[d] = Σ_d q_prime[d] * code[d]
        //                         + Σ_d q[d] * offset[d].
        let dot = qp_code_dot + self.q_dot_offset;
        match self.metric {
            Metric::Cosine => {
                let x_norm = norm
                    .expect("Sq8Kernel + Cosine requires per_doc_norms")
                    .sqrt();
                if x_norm > 0.0 {
                    COSINE_DISTANCE_BASE - dot / x_norm
                } else {
                    COSINE_DISTANCE_BASE - dot
                }
            }
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                let x_norm_sq = norm.expect("Sq8Kernel + L2Sq requires per_doc_norms");
                self.q_norm_sq - L2_CROSS_TERM_COEFF * dot + x_norm_sq
            }
        }
    }
}

/// `Sq8ResidualEpsilon` rerank context — the residual-corrected sibling of
/// [`Sq8Kernel`]. Captures the per-cluster quantizer (`scale[dim]`,
/// `offset[dim]`) plus the query-side precomputes for both the Sq8
/// code leg and the i8 residual leg, so the per-candidate inner loop
/// is two u8/i8 → f32 widens + SIMD dot.
///
/// Applied only to the small final-refine set the Sq8 score selects,
/// so it never runs over the full shortlist. One kernel per query +
/// cluster, reused across that cluster's refine candidates.
pub(crate) struct Sq8ResidualEpsilonKernel<'a> {
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
    /// Per-doc `Σ_d x_corrected²` table (residual-corrected norms),
    /// indexed by the shortlist's `pos`. `Some` for L2Sq + Cosine.
    per_doc_norms: Option<&'a [f32]>,
}

impl<'a> Sq8ResidualEpsilonKernel<'a> {
    /// Build the per-query residual kernel. `scale` + `offset` are
    /// the per-cluster quantizer arrays; `residual_divisor` is
    /// [`SQ8_RESIDUAL_DIVISOR`]. `per_doc_norms` is `Some` for L2Sq
    /// and Cosine columns.
    pub fn new(
        metric: Metric,
        query: &[f32],
        scale: &[f32],
        offset: &[f32],
        residual_divisor: f32,
        per_doc_norms: Option<&'a [f32]>,
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
            per_doc_norms,
        }
    }

    /// Distance for one refine candidate at position `pos`, with
    /// `dim` u8 Sq8 codes at `code_bytes` and `dim` i8 residual
    /// codes at `residual_bytes`. Smaller = closer for every metric.
    #[inline]
    pub fn distance_at(&self, pos: u32, code_bytes: &[u8], residual_bytes: &[u8]) -> f32 {
        let norm = self.per_doc_norms.map(|norms| norms[pos as usize]);
        self.distance_with_norm(code_bytes, residual_bytes, norm)
    }

    /// Like [`Self::distance_at`] but takes the per-doc decoded-norm
    /// explicitly — for lazy object-store paths that fetch norms into
    /// a sparse `pos → norm` map rather than a contiguous slice.
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
                    .expect("Sq8ResidualEpsilonKernel + Cosine requires per_doc_norms")
                    .sqrt();
                if x_norm > 0.0 {
                    COSINE_DISTANCE_BASE - dot / x_norm
                } else {
                    COSINE_DISTANCE_BASE - dot
                }
            }
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                let x_norm_sq =
                    norm.expect("Sq8ResidualEpsilonKernel + L2Sq requires per_doc_norms");
                self.q_norm_sq - L2_CROSS_TERM_COEFF * dot + x_norm_sq
            }
        }
    }
}

/// Dot-product reduction for `Sq8Kernel::distance_at`:
/// `Σ_d q_prime[d] * (code_bytes[d] as f32)` over the first `dim`
/// dimensions. This is the `q_prime · code` half of the Sq8 distance
/// expansion — the `Σ q[d] * offset[d]` half is folded into
/// `Sq8Kernel::q_dot_offset` once at query-prep time.
///
/// Three-tier dispatch:
///
/// 1. AVX-512 (16-lane FMA + `vpmovzxbd` u8 → i32 widen)
/// 2. AVX2 (8-lane FMA + `vpmovzxbd` u8 → i32 widen — same widen
///    instruction in a half-width register, no scalar per-lane
///    casts in the hot loop)
/// 3. Portable `wide::f32x8` with per-lane scalar `as f32` widen
///    (aarch64 / SSE-only / `INFINO_DISABLE_AVX2=1`)
///
/// All three paths compute exactly the same reduction in
/// `bit-identical` lane order up to f32 add-tree associativity (the
/// reduce tree's shape differs between 8-lane and 16-lane
/// accumulators, but the per-pair multiplies are identical and the
/// resulting sum differs only by an FMA-vs-multiply rounding ε per
/// reduction step — well below Sq8's per-lane quantization error).
///
/// Inputs are pre-validated by `Sq8Kernel::distance_at`'s
/// `debug_assert_eq!(code_bytes.len(), self.dim)`. `q_prime.len()`
/// is guaranteed `== dim` by `Sq8Kernel::new`.
#[inline]
pub(crate) fn sq8_dot(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` which requires `avx512f`.
            return unsafe { sq8_dot_avx512(q_prime, code_bytes, dim) };
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2_enabled()` which requires `avx2`.
            return unsafe { sq8_dot_avx2(q_prime, code_bytes, dim) };
        }
    }
    sq8_dot_wide(q_prime, code_bytes, dim)
}

/// Upper chunk length for [`u8_sum_sumsq`]'s u32 accumulators:
/// `U8_SUMSQ_CHUNK · 255² < u32::MAX`, so a per-chunk Σcode² cannot
/// overflow before it spills into the u64 total.
const U8_SUMSQ_CHUNK: usize = 16_384;

/// Σcode and Σcode² over one Sq8 code row, both as f32.
///
/// Exact integer accumulation: u32 lane math inside bounded chunks
/// (u8-widening adds + `pmaddwd`-shaped squares that LLVM
/// auto-vectorizes at the `x86-64-v3` baseline — 64-bit accumulators
/// would defeat vectorization), spilled into u64 totals between
/// chunks so no input length can overflow. Used by the manifest's
/// folded Sq8 centroid scoring to reconstruct `‖centroid‖²` without
/// dequantizing.
pub(crate) fn u8_sum_sumsq(codes: &[u8]) -> (f32, f32) {
    let mut sum: u64 = 0;
    let mut sumsq: u64 = 0;
    for chunk in codes.chunks(U8_SUMSQ_CHUNK) {
        let mut s: u32 = 0;
        let mut sq: u32 = 0;
        for &b in chunk {
            let v = b as u32;
            s += v;
            sq += v * v;
        }
        sum += u64::from(s);
        sumsq += u64::from(sq);
    }
    (sum as f32, sumsq as f32)
}

/// Portable `wide::f32x8` (256-bit) Sq8 dot product. Same per-
/// element math as the AVX-512 path, processed 8 lanes at a time
/// with a per-lane scalar `u8 as f32` widen. Universal fallback
/// for aarch64, SSE-only x86_64 hosts, and
/// `INFINO_DISABLE_AVX2=1` / `INFINO_DISABLE_AVX512=1` A/B runs.
#[inline]
fn sq8_dot_wide(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + F32X8_LANES <= dim {
        let qc: [f32; F32X8_LANES] = q_prime[i..i + F32X8_LANES]
            .try_into()
            .expect("q_prime[i..i+8] len 8");
        let mut bc = [0f32; F32X8_LANES];
        for (j, slot) in bc.iter_mut().enumerate() {
            *slot = code_bytes[i + j] as f32;
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        acc += qv * bv;
        i += F32X8_LANES;
    }
    let mut dot = acc.reduce_add();
    while i < dim {
        dot += q_prime[i] * (code_bytes[i] as f32);
        i += 1;
    }
    dot
}

/// AVX2 Sq8 dot product. Same shape as the AVX-512 path but
/// 8 lanes per iteration via 256-bit registers. The win vs the
/// portable wide kernel is the u8 → f32 widen: a single
/// `vpmovzxbd` (zero-extend 8 u8 to 8 i32) + `vcvtdq2ps` (convert
/// 8 i32 to 8 f32) pair, instead of 8 scalar `as f32` casts the
/// compiler can't always hoist out of the SIMD loop. Lifts every
/// AVX2 host (g5, Graviton-on-x86, Skylake, Zen 2 / 3, ...) that
/// lacks AVX-512.
///
/// # Safety
///
/// Callers must ensure the target supports `avx2`. `avx2_enabled()`
/// guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq8_dot_avx2(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    debug_assert_eq!(q_prime.len(), dim);
    debug_assert_eq!(code_bytes.len(), dim);

    // SAFETY: each iteration reads 8 f32s from `q_prime` and 8
    // bytes from `code_bytes`. The `i + 8 <= dim` predicate
    // guarantees both windows are in bounds. `_mm_loadl_epi64`
    // reads exactly 64 bits = 8 bytes; `_mm256_loadu_ps` reads
    // 32 bytes (8 f32s). Both are unaligned loads.
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + F32X8_LANES <= dim {
            // Load 8 u8 doc codes into the low 64 bits of an xmm
            // register; the high 64 bits are zero.
            let codes_u8 = _mm_loadl_epi64(code_bytes.as_ptr().add(i) as *const __m128i);
            // `_mm256_cvtepu8_epi32` (VPMOVZXBD): zero-extend the
            // low 8 bytes to 8 × i32 in a 256-bit register.
            let codes_i32 = _mm256_cvtepu8_epi32(codes_u8);
            // `_mm256_cvtepi32_ps` (VCVTDQ2PS): 8 i32 → 8 f32.
            let codes_f32 = _mm256_cvtepi32_ps(codes_i32);
            let q = _mm256_loadu_ps(q_prime.as_ptr().add(i));
            acc = _mm256_fmadd_ps(q, codes_f32, acc);
            i += F32X8_LANES;
        }
        // Horizontal add 8 fp32 lanes. Standard hadd-tree.
        let lo = _mm256_castps256_ps128(acc);
        let hi = _mm256_extractf128_ps(acc, 1);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        let mut dot = _mm_cvtss_f32(sums2);
        while i < dim {
            dot += q_prime[i] * (code_bytes[i] as f32);
            i += 1;
        }
        dot
    }
}

/// AVX-512 Sq8 dot product. The win vs the `wide` kernel is two
/// stacked sources of speedup: the f32 FMA is 16-wide instead of
/// 8, **and** the u8 → f32 widen is a single `vpmovzxbd` +
/// `vcvtdq2ps` pair instead of 8 scalar `as f32` casts.
///
/// # Safety
///
/// Callers must ensure the target supports `avx512f`. `avx512_enabled()`
/// guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq8_dot_avx512(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    debug_assert_eq!(q_prime.len(), dim);
    debug_assert_eq!(code_bytes.len(), dim);

    // SAFETY: each iteration reads 16 f32s from `q_prime` and 16
    // bytes from `code_bytes`. The `i + 16 <= dim` predicate
    // guarantees both windows are in bounds. `_mm_loadu_si128`
    // and `_mm512_loadu_ps` are unaligned loads so no alignment
    // assumption is needed.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + AVX512_F32_LANES <= dim {
            // Load 16 u8 doc codes (one 128-bit lane) and widen
            // to 16 × i32 then convert to 16 × f32.
            let codes = _mm_loadu_si128(code_bytes.as_ptr().add(i) as *const __m128i);
            let codes_i32 = _mm512_cvtepu8_epi32(codes);
            let codes_f32 = _mm512_cvtepi32_ps(codes_i32);
            let q = _mm512_loadu_ps(q_prime.as_ptr().add(i));
            acc = _mm512_fmadd_ps(q, codes_f32, acc);
            i += AVX512_F32_LANES;
        }
        let mut dot = _mm512_reduce_add_ps(acc);
        while i < dim {
            dot += q_prime[i] * (code_bytes[i] as f32);
            i += 1;
        }
        dot
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

    // --- sq8 kernel -----------------------------------------------------

    /// Encode `values` to u8 codes using the same per-dim
    /// `scale`/`offset` the kernel will decode under.
    fn encode_sq8(values: &[f32], dim: usize, scale: &[f32], offset: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len());
        for row in values.chunks_exact(dim) {
            for d in 0..dim {
                let q = ((row[d] - offset[d]) / scale[d]).round().clamp(0.0, 255.0) as u8;
                out.push(q);
            }
        }
        out
    }

    /// Decode the same u8 codes back to fp32 — the reference the
    /// kernel must agree with.
    fn decode_sq8(codes: &[u8], dim: usize, scale: &[f32], offset: &[f32]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c as f32) * scale[i % dim] + offset[i % dim])
            .collect()
    }

    /// Decode `Sq8ResidualEpsilon` codes (`code * scale + offset + residual
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
            let kernel = Sq8ResidualEpsilonKernel::new(
                metric,
                &query,
                &scale,
                &offset,
                residual_divisor,
                norms_arg,
            );
            let got = kernel.distance_at(0, &codes, &residuals);
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
        let kernel = Sq8ResidualEpsilonKernel::new(
            Metric::NegDot,
            &query,
            &scale,
            &offset,
            residual_divisor,
            None,
        );
        let got = kernel.distance_at(0, &codes, &residuals);
        let want = distance(Metric::NegDot, &query, &corrected);
        assert!(
            (want - got).abs() <= 1e-4,
            "tail-dim residual kernel: got {got} vs corrected ref {want}"
        );
    }

    #[test]
    fn sq8_kernel_dot_matches_decoded_reference() {
        let dim = 16usize;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i as f32) * 0.002).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -1.0 + (i as f32) * 0.1).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
        let decoded = decode_sq8(&codes, dim, &scale, &offset);

        for m in [Metric::Cosine, Metric::NegDot] {
            let norms = if m == Metric::Cosine {
                Some(vec![decoded.iter().map(|x| x * x).sum::<f32>()])
            } else {
                None
            };
            let want = match m {
                Metric::Cosine => {
                    let x_norm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if x_norm > 0.0 {
                        1.0 - dot(&query, &decoded) / x_norm
                    } else {
                        1.0 - dot(&query, &decoded)
                    }
                }
                Metric::NegDot => distance(m, &query, &decoded),
                Metric::L2Sq => unreachable!(),
            };
            let kernel = Sq8Kernel::new(m, &query, &scale, &offset, norms.clone().map(Arc::from));
            let got = kernel.distance_at(0, &codes);
            let err = (want - got).abs();
            assert!(
                err <= 1e-4,
                "metric {m:?}: kernel {got} vs decoded ref {want} (err {err})"
            );
        }
    }

    #[test]
    fn sq8_kernel_l2sq_matches_decoded_reference() {
        let dim = 24usize;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.07 - 0.1).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.02 + (i as f32) * 0.003).collect();
        let offset: Vec<f32> = (0..dim).map(|i| 0.5 - (i as f32) * 0.05).collect();
        // Two docs with very different codes — exercise both
        // pos=0 and pos=1 into the norms table.
        let codes_doc0: Vec<u8> = (0..dim).map(|i| ((i * 7) % 256) as u8).collect();
        let codes_doc1: Vec<u8> = (0..dim).map(|i| ((i * 31 + 12) % 256) as u8).collect();
        let decoded0 = decode_sq8(&codes_doc0, dim, &scale, &offset);
        let decoded1 = decode_sq8(&codes_doc1, dim, &scale, &offset);
        let norm0: f32 = decoded0.iter().map(|x| x * x).sum();
        let norm1: f32 = decoded1.iter().map(|x| x * x).sum();
        let per_doc_norms = vec![norm0, norm1];

        let kernel = Sq8Kernel::new(
            Metric::L2Sq,
            &query,
            &scale,
            &offset,
            Some(Arc::from(per_doc_norms.clone())),
        );

        let got0 = kernel.distance_at(0, &codes_doc0);
        let want0 = distance(Metric::L2Sq, &query, &decoded0);
        assert!(
            (want0 - got0).abs() <= 1e-3,
            "doc0: kernel {got0} vs decoded ref {want0}"
        );

        let got1 = kernel.distance_at(1, &codes_doc1);
        let want1 = distance(Metric::L2Sq, &query, &decoded1);
        assert!(
            (want1 - got1).abs() <= 1e-3,
            "doc1: kernel {got1} vs decoded ref {want1}"
        );
    }

    #[test]
    fn sq8_kernel_handles_tail_dim_not_multiple_of_8() {
        // Dim 13: one SIMD chunk + 5-lane tail. The kernel's
        // per-query loop must merge the tail into q_prime /
        // q_dot_offset; the per-doc loop must merge the tail
        // into `cross`.
        let dim = 13usize;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.03 + 0.1).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i as f32) * 0.001).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -0.1 + (i as f32) * 0.02).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 11 + 5) % 256) as u8).collect();
        let decoded = decode_sq8(&codes, dim, &scale, &offset);

        let kernel = Sq8Kernel::new(Metric::NegDot, &query, &scale, &offset, None);
        let got = kernel.distance_at(0, &codes);
        let want = distance(Metric::NegDot, &query, &decoded);
        assert!(
            (want - got).abs() <= 1e-4,
            "tail-dim Sq8 kernel: got {got} vs decoded ref {want}"
        );
    }

    #[test]
    fn sq8_full_round_trip_within_recall_tolerance_of_fp32() {
        // Multi-doc corpus so per-dim min < max (a single-doc
        // corpus collapses to scale=1.0/offset=x per dim — the
        // degenerate-dim guard, not the real quantizer).
        //
        // Worst-case per-dim quantization error is `scale/2 ≈
        // (max-min)/510`. For this corpus, per-dim span ≈ 32 →
        // error ≈ 0.063 per dim. |q-x|² over 16 dims is bounded
        // by ≈ Σ_d (2·|q_d-x_d|·0.063 + 0.063²) ≈ a few units.
        // The test pins generous tolerances per metric to stay
        // robust against rounding on different platforms.
        let dim = 16usize;
        let n_docs = 32usize;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.5).collect();
        let corpus: Vec<f32> = (0..n_docs)
            .flat_map(|i| (0..dim).map(move |j| ((i * 7 + j * 3) as f32 % 32.0) - 8.0))
            .collect();

        let mut min_v = vec![f32::INFINITY; dim];
        let mut max_v = vec![f32::NEG_INFINITY; dim];
        for row in corpus.chunks_exact(dim) {
            for (d, &x) in row.iter().enumerate() {
                min_v[d] = min_v[d].min(x);
                max_v[d] = max_v[d].max(x);
            }
        }
        // Sanity check: per-dim span is non-zero, so we're
        // exercising real quantization rather than the
        // degenerate-dim guard. Catches a future test edit that
        // accidentally re-shrinks the corpus.
        for d in 0..dim {
            assert!(
                max_v[d] - min_v[d] > 0.0,
                "test corpus must span each dim: dim {d} has min == max"
            );
        }

        let mut scale = vec![0.0f32; dim];
        let mut offset = vec![0.0f32; dim];
        for d in 0..dim {
            offset[d] = min_v[d];
            scale[d] = (max_v[d] - min_v[d]) / 255.0;
        }
        let codes_all = encode_sq8(&corpus, dim, &scale, &offset);
        let decoded_all = decode_sq8(&codes_all, dim, &scale, &offset);

        // Per-doc norms for the L2Sq branch — indexed by pos
        // matching the builder's contract.
        let per_doc_norms: Vec<f32> = decoded_all
            .chunks_exact(dim)
            .map(|row| row.iter().map(|x| x * x).sum::<f32>())
            .collect();

        for m in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let norms_arg: Option<Arc<[f32]>> = match m {
                Metric::L2Sq | Metric::Cosine => Some(Arc::from(per_doc_norms.clone())),
                Metric::NegDot => None,
            };
            let kernel = Sq8Kernel::new(m, &query, &scale, &offset, norms_arg);
            // Probe a handful of doc positions — exercises both
            // norms-table indexing and the per-doc inner loop on
            // independent codes.
            for pos in [0u32, 1, 5, 17, 31] {
                let codes_doc = &codes_all[(pos as usize) * dim..(pos as usize + 1) * dim];
                let decoded_doc = &decoded_all[(pos as usize) * dim..(pos as usize + 1) * dim];
                let got = kernel.distance_at(pos, codes_doc);
                let want_fp32 = distance(
                    m,
                    &query,
                    &corpus[(pos as usize) * dim..(pos as usize + 1) * dim],
                );
                let want_decoded = match m {
                    Metric::Cosine => {
                        let x_norm = per_doc_norms[pos as usize].sqrt();
                        if x_norm > 0.0 {
                            1.0 - dot(&query, decoded_doc) / x_norm
                        } else {
                            1.0 - dot(&query, decoded_doc)
                        }
                    }
                    _ => distance(m, &query, decoded_doc),
                };
                // Kernel must match the decoded reference very
                // tightly — it's doing the same math, just fused
                // through the per-query precompute. Difference
                // from fp32 is the quantization error itself.
                assert!(
                    (got - want_decoded).abs() <= 1e-3,
                    "metric {m:?} pos {pos}: kernel {got} vs decoded ref {want_decoded}"
                );
                // Cosine Sq8 normalizes the decoded vector at rerank;
                // [`distance`] assumes unit-norm fp32 inputs, so the
                // fp32 reference is only meaningful for L2Sq / NegDot.
                if m != Metric::Cosine {
                    let rel = (got - want_fp32).abs() / want_fp32.abs().max(1e-2);
                    assert!(
                        rel <= 0.1 || (got - want_fp32).abs() <= 1.0,
                        "metric {m:?} pos {pos}: Sq8 {got} vs fp32 {want_fp32} (rel {rel})"
                    );
                }
            }
        }
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

    // --- AVX-512 parity -------------------------------------------------

    /// AVX-512 `sq8_dot` agrees with the `wide` baseline
    /// across a length sweep. The dot product is `Σ q_prime[d] *
    /// (code[d] as f32)` so values are integer-magnitude on the
    /// doc side — exact widen, reduction-order is the only divergence.
    /// Tolerance is correspondingly tight.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sq8_dot_avx512_matches_wide_across_lengths() {
        if !avx512_enabled() {
            eprintln!("sq8_dot_avx512_matches_wide_across_lengths: skipped, no AVX-512");
            return;
        }
        for dim in [1usize, 7, 15, 16, 17, 31, 32, 33, 64, 96, 128, 384, 768] {
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let want = sq8_dot_wide(&q_prime, &codes, dim);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { sq8_dot_avx512(&q_prime, &codes, dim) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: sq8 avx512 {got} vs sq8 wide {want} (tol {tol})"
            );
        }
    }

    // --- AVX2 parity ----------------------------------------------------

    /// AVX2 `sq8_dot_avx2` agrees with the portable wide
    /// kernel across a length sweep. Inner math is identical (FMA
    /// of q_prime against the u8-widened doc codes); the only
    /// difference is how the widen happens. Tolerance is one
    /// add-tree ULP per accumulator slot times √(dim/8); the
    /// constant `1e-5 * |result|` more than covers that.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sq8_dot_avx2_matches_wide_across_lengths() {
        if !avx2_enabled() {
            eprintln!("sq8_dot_avx2_matches_wide_across_lengths: skipped, no AVX2");
            return;
        }
        for dim in [
            1usize, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 96, 128, 384, 768,
        ] {
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let want = sq8_dot_wide(&q_prime, &codes, dim);
            // SAFETY: gated on avx2_enabled() above.
            let got = unsafe { sq8_dot_avx2(&q_prime, &codes, dim) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: sq8 avx2 {got} vs wide {want} (tol {tol})"
            );
        }
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
        use std::hint::black_box;
        use std::time::Instant;
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

    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn avx512_microbench_sq8_kernel() {
        if !avx512_enabled() {
            eprintln!("avx512_microbench: skipped, no AVX-512 on this host");
            return;
        }
        eprintln!();
        eprintln!(
            "### Sq8 cross-product kernel — AVX-512 (vpmovzxbd widen) vs wide (ns per call)\n"
        );
        eprintln!("| kernel | dim | wide ns | avx512 ns | speedup |");
        eprintln!("|--------|----:|--------:|----------:|--------:|");

        use std::hint::black_box;
        for &dim in realistic_dims() {
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            let wide_ns = time_ns(iters, || {
                sq8_dot_wide(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            // SAFETY: gated on avx512_enabled() above.
            let avx_ns = time_ns(iters, || unsafe {
                sq8_dot_avx512(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            eprintln!(
                "| `Sq8Kernel::distance_at` (dot) | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );
        }
    }

    // --- AVX2 microbench (run by hand) ---------------------------------
    //
    // Measures the AVX2 widen-FMA paths against the portable
    // scalar-widen kernels they replace on AVX2 hosts. Run with:
    //
    // ```text
    // cargo test --release --lib superfile::vector::distance::tests::\
    //   avx2_microbench -- --ignored --nocapture
    // ```
    //
    // On hosts with AVX-512, the AVX2 widen path is not the runtime
    // default (the dispatch chain picks AVX-512 first), but the
    // parity tests + this microbench still exercise it via direct
    // call to keep the AVX2 baseline a first-class measurable tier.

    // --- Unified 4-tier per-kernel microbench --------------------------
    //
    // One run, every kernel × every SIMD tier × every realistic dim,
    // emitted as a single markdown table. Replaces ad-hoc per-tier
    // microbenches that only ever showed two columns side-by-side
    // (wide vs avx512, or wide vs avx2). Run with:
    //
    // ```text
    // cargo test --release --lib simd_microbench_all_tiers \
    //   -- --ignored --nocapture
    // ```
    //
    // Columns mean exactly what they say: ns/call for that kernel
    // routed through that specific implementation, irrespective of
    // what the runtime dispatch chain would have picked. Columns
    // without a dedicated path (e.g. `dot` fp32 has no separate
    // AVX2 kernel — the wide path *is* the AVX2 path via `wide`)
    // are printed as `—` so the table doesn't lie about coverage.

    /// Scalar fp32 dot. No SIMD types — the absolute baseline.
    /// Compiler will autovectorize this on most x86_64 targets but
    /// the scalar source is what we measure, so the result is
    /// representative of "what you get with no hand-tuned SIMD".
    #[cfg(target_arch = "x86_64")]
    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            s += a[i] * b[i];
        }
        s
    }

    /// Scalar fp32 L2².
    #[cfg(target_arch = "x86_64")]
    fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
        let mut s = 0.0f32;
        for i in 0..a.len() {
            let d = a[i] - b[i];
            s += d * d;
        }
        s
    }

    /// Scalar Sq8 dot-product kernel core: `Σ q'[d] * code[d]`
    /// after per-lane u8→f32 widening. Used inside `Sq8Kernel::
    /// distance_at`; this is the part the SIMD paths accelerate.
    #[cfg(target_arch = "x86_64")]
    fn sq8_dot_scalar(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
        let mut s = 0.0f32;
        for d in 0..dim {
            s += q_prime[d] * (code_bytes[d] as f32);
        }
        s
    }

    /// Single-pane microbench: every kernel × scalar/wide/AVX2/AVX-512
    /// at every realistic dim, one markdown table.
    ///
    /// When a kernel has no dedicated AVX2 implementation (e.g. fp32
    /// `dot`/`l2_sq` — the `wide::f32x8` path already lowers to
    /// `__m256` + `vfmadd*ps` under the `x86-64-v3` target this crate
    /// pins via `.cargo/config.toml`, so a hand-written AVX2 kernel
    /// would emit the same instructions), the AVX2 column shows
    /// `wide(=AVX2)` followed by the wide ns to make it clear that
    /// the dispatch chain on an AVX2-only host actually runs at the
    /// wide column's number. Kernels that *do* have a separate AVX2
    /// path (the Sq8 widen kernel — wide had per-lane scalar widen,
    /// AVX2 has VPMOVZXBD + shift) shows the dedicated AVX2 timing.
    #[test]
    #[ignore = "perf microbench, not a correctness gate"]
    #[cfg(target_arch = "x86_64")]
    fn simd_microbench_all_tiers() {
        use std::hint::black_box;
        let avx2 = avx2_enabled();
        let avx512 = avx512_enabled();
        eprintln!();
        eprintln!(
            "### vector distance kernels — per-tier ns / call on this host (single thread, release)\n"
        );
        eprintln!("host caps: avx2={avx2}, avx512f={avx512}");
        eprintln!(
            "build:     `target-cpu=x86-64-v3` (Haswell+AVX2+FMA baseline) from .cargo/config.toml\n"
        );
        eprintln!("| kernel | dim | scalar ns | wide ns | avx2 ns | avx512 ns |");
        eprintln!("|--------|----:|----------:|--------:|--------:|----------:|");

        /// Format an AVX2 cell: `Some(ns)` for a dedicated AVX2
        /// kernel, `None` for a kernel whose AVX2 dispatch falls
        /// through to wide (the wide ns is passed so the cell
        /// shows the actual runtime cost on an AVX2-only host).
        fn avx2_cell(v: Option<f64>, wide_ns: f64) -> String {
            match v {
                Some(x) => format!("{:>7.1}", x),
                None => format!("wide(={:>5.1})", wide_ns),
            }
        }

        /// Format an AVX-512 cell: `Some(ns)` for a dedicated kernel,
        /// `None` when AVX-512 isn't enabled on this host.
        fn avx512_cell(v: Option<f64>) -> String {
            match v {
                Some(x) => format!("{:>7.1}", x),
                None => "      —".to_string(),
            }
        }

        for &dim in realistic_dims() {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
            let b: Vec<f32> = (0..dim).map(|i| ((i + 7) as f32) * 0.0017 - 0.3).collect();
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            // --- distance::dot (fp32) ---
            let s = time_ns(iters, || dot_scalar(black_box(&a), black_box(&b)));
            let w = time_ns(iters, || dot_wide(black_box(&a), black_box(&b)));
            // No dedicated AVX2 path — `wide::f32x8` on x86-64-v3
            // lowers straight to `__m256` + `vfmadd*ps`, so the wide
            // path *is* the AVX2 path for this kernel. AVX2 column
            // prints `wide(=<wide ns>)` to make that explicit.
            let a2 = None::<f64>;
            let a5 = if avx512 {
                Some(time_ns(iters, || unsafe {
                    dot_avx512(black_box(&a), black_box(&b))
                }))
            } else {
                None
            };
            eprintln!(
                "| `distance::dot` (fp32) | {dim} | {:>9.1} | {:>7.1} | {} | {} |",
                s,
                w,
                avx2_cell(a2, w),
                avx512_cell(a5),
            );

            // --- distance::l2_sq (fp32) ---
            let s = time_ns(iters, || l2_sq_scalar(black_box(&a), black_box(&b)));
            let w = time_ns(iters, || l2_sq_wide(black_box(&a), black_box(&b)));
            let a2 = None::<f64>;
            let a5 = if avx512 {
                Some(time_ns(iters, || unsafe {
                    l2_sq_avx512(black_box(&a), black_box(&b))
                }))
            } else {
                None
            };
            eprintln!(
                "| `distance::l2_sq` (fp32) | {dim} | {:>9.1} | {:>7.1} | {} | {} |",
                s,
                w,
                avx2_cell(a2, w),
                avx512_cell(a5),
            );

            // --- sq8_dot (the Sq8Kernel hot loop core) ---
            let s = time_ns(iters, || {
                sq8_dot_scalar(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            let w = time_ns(iters, || {
                sq8_dot_wide(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            let a2 = if avx2 {
                Some(time_ns(iters, || unsafe {
                    sq8_dot_avx2(black_box(&q_prime), black_box(&codes), black_box(dim))
                }))
            } else {
                None
            };
            let a5 = if avx512 {
                Some(time_ns(iters, || unsafe {
                    sq8_dot_avx512(black_box(&q_prime), black_box(&codes), black_box(dim))
                }))
            } else {
                None
            };
            eprintln!(
                "| `Sq8Kernel::distance_at` (dot) | {dim} | {:>9.1} | {:>7.1} | {} | {} |",
                s,
                w,
                avx2_cell(a2, w),
                avx512_cell(a5),
            );
        }

        eprintln!();
        eprintln!(
            "Notes: `wide(=N.N)` in the AVX2 column means there is no \
             dedicated AVX2 kernel — the dispatch on an AVX2-only host \
             actually runs the wide kernel at that timing. This applies to \
             the fp32 `dot` / `l2_sq` kernels because `wide::f32x8` on \
             `target-cpu=x86-64-v3` lowers to `__m256` + `vfmadd*ps`, \
             which is what a hand-written AVX2 kernel would emit. The \
             Sq8 widen kernel has a dedicated AVX2 path (visible \
             above) because the wide path previously did per-lane scalar \
             widening; the dedicated AVX2 path replaces that with \
             VPMOVZXBD / VPMOVZXWD + shift."
        );
    }

    /// AVX2 fp32-equivalent Sq8 widen path vs the portable
    /// scalar-widen `_wide` kernel. Captures the "lift the AVX2
    /// fallback path" win; the complementary Sq8Kernel rerank cache
    /// is a data-structure change exercised by the IVF rerank benches
    /// end-to-end.
    #[test]
    #[ignore]
    #[cfg(target_arch = "x86_64")]
    fn avx2_microbench_widen_kernels() {
        if !avx2_enabled() {
            eprintln!("avx2_microbench: skipped, no AVX2 on this host");
            return;
        }
        eprintln!();
        eprintln!("### AVX2 widen + FMA vs portable scalar-widen wide path (ns per call)\n");
        eprintln!("| kernel | dim | wide ns | avx2 ns | speedup |");
        eprintln!("|--------|----:|--------:|--------:|--------:|");

        use std::hint::black_box;
        for &dim in realistic_dims() {
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            let wide_sq8_ns = time_ns(iters, || {
                sq8_dot_wide(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            // SAFETY: gated on avx2_enabled() above.
            let avx2_sq8_ns = time_ns(iters, || unsafe {
                sq8_dot_avx2(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            eprintln!(
                "| `Sq8Kernel::distance_at` (dot) | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_sq8_ns,
                avx2_sq8_ns,
                wide_sq8_ns / avx2_sq8_ns,
            );
        }
    }
}
