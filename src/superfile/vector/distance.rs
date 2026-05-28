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

use wide::f32x8;

use crate::superfile::vector::rerank_codec::RerankCodec;
use crate::superfile::vector::simd_dispatch::{avx512_enabled, has_bf16_dot};

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
        Metric::Cosine => 1.0 - dot(a, b),
        Metric::L2Sq => l2_sq(a, b),
        Metric::NegDot => -dot(a, b),
    }
}

/// f32 dot product. Dispatches to the AVX-512 16-lane FMA kernel when
/// the runtime CPUID gate passes; otherwise the `wide::f32x8` AVX2 /
/// NEON / scalar kernel (which has been the universal kernel since the
/// segment-builder existed). Both kernels handle non-multiple-of-lane
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
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
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
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();

    let mut acc = f32x8::ZERO;
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = f32x8::from(
            <[f32; 8]>::try_from(ca).expect("chunks_exact(8) yields slices of length 8"),
        );
        let vb = f32x8::from(
            <[f32; 8]>::try_from(cb).expect("chunks_exact(8) yields slices of length 8"),
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
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let tail_a = chunks_a.remainder();
    let tail_b = chunks_b.remainder();

    let mut acc = f32x8::ZERO;
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let va = f32x8::from(
            <[f32; 8]>::try_from(ca).expect("chunks_exact(8) yields slices of length 8"),
        );
        let vb = f32x8::from(
            <[f32; 8]>::try_from(cb).expect("chunks_exact(8) yields slices of length 8"),
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
    use std::arch::x86_64::*;
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
        while i + 16 <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            acc = _mm512_fmadd_ps(va, vb, acc);
            i += 16;
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
    use std::arch::x86_64::*;
    let n = a.len();
    // SAFETY: see `dot_avx512` — same bounds reasoning, same
    // unaligned-load contract.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 16 <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i));
            let d = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(d, d, acc);
            i += 16;
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
    debug_assert_eq!(query.len() * 4, bytes.len());
    match metric {
        Metric::Cosine => 1.0 - dot_bytes(query, bytes),
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
    while i + 8 <= query.len() {
        let qc: [f32; 8] = query[i..i + 8]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; 8];
        for (j, slot) in bc.iter_mut().enumerate() {
            let off = (i + j) * 4;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        acc += qv * bv;
        i += 8;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * 4;
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
    while i + 8 <= query.len() {
        let qc: [f32; 8] = query[i..i + 8]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; 8];
        for (j, slot) in bc.iter_mut().enumerate() {
            let off = (i + j) * 4;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        let d = qv - bv;
        acc += d * d;
        i += 8;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * 4;
        let b = f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let d = query[i] - b;
        sum += d * d;
        i += 1;
    }
    sum
}

/// Distance against a vector stored in the column's `rerank_codec`
/// representation. The fast path for `Fp32` reuses [`distance_bytes`];
/// `Bf16` widens 8 × bf16 → 8 × f32 per inner step and reuses the same
/// `f32x8` math. The bf16 widening is exact (a left-shift by 16 of the
/// u16 bit pattern reinterpreted as fp32), so the only error vs. an
/// fp32 store is the round-to-nearest-even at encode time — bounded by
/// `relative_eps ≈ 2⁻⁸ ≈ 4 × 10⁻³` per lane.
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
        RerankCodec::Bf16 => distance_bytes_bf16(metric, query, bytes),
        RerankCodec::Sq8 => {
            unreachable!(
                "distance_bytes_codec called with Sq8 — Sq8 rerank goes through \
                 Sq8Kernel (needs per-column scale/offset + per-doc norm context)"
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
pub(crate) struct Sq8Kernel<'a> {
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
    /// `None` for Cosine / NegDot (the `x²` term cancels out).
    per_doc_norms: Option<&'a [f32]>,
}

impl<'a> Sq8Kernel<'a> {
    /// Build the per-query kernel. `scale` + `offset` are the
    /// per-dim quantizer arrays from the column's `codec_meta`.
    /// `per_doc_norms` is `Some` iff the column metric is L2Sq.
    pub fn new(
        metric: Metric,
        query: &[f32],
        scale: &[f32],
        offset: &[f32],
        per_doc_norms: Option<&'a [f32]>,
    ) -> Self {
        let dim = query.len();
        debug_assert_eq!(scale.len(), dim);
        debug_assert_eq!(offset.len(), dim);
        // Build q_prime + q_dot_offset in one SIMD pass per
        // dim — both fold over the same query.
        let mut q_prime = vec![0.0f32; dim];
        let mut q_dot_offset_acc = f32x8::ZERO;
        let mut i = 0;
        while i + 8 <= dim {
            let qc = f32x8::from(<[f32; 8]>::try_from(&query[i..i + 8]).expect("len-8 slice"));
            let sc = f32x8::from(<[f32; 8]>::try_from(&scale[i..i + 8]).expect("len-8 slice"));
            let oc = f32x8::from(<[f32; 8]>::try_from(&offset[i..i + 8]).expect("len-8 slice"));
            let qp = qc * sc;
            // Write q_prime out as 8 f32s. `wide::f32x8::to_array`
            // is the safe accessor; the per-lane copy compiles to
            // a single 32-byte mov on AVX2.
            q_prime[i..i + 8].copy_from_slice(&qp.to_array());
            q_dot_offset_acc += qc * oc;
            i += 8;
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
        debug_assert_eq!(code_bytes.len(), self.dim);
        // Per-doc inner reduction: Σ_d q_prime[d] * code[d] as f32.
        // Dispatches to AVX-512 (16-lane FMA with VPMOVZXBD widen)
        // when the runtime gate passes; otherwise the f32x8 widen-
        // and-FMA kernel that has shipped since 012.
        let cross = sq8_cross_product(&self.q_prime, code_bytes, self.dim);
        // `dot(query, x_decoded) = cross + q_dot_offset` because
        // x_decoded[d] = code[d] * scale[d] + offset[d], so
        // Σ_d q[d] * x_decoded[d] = Σ_d q_prime[d] * code[d]
        //                         + Σ_d q[d] * offset[d].
        let dot = cross + self.q_dot_offset;
        match self.metric {
            Metric::Cosine => 1.0 - dot,
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                // |q - x|² = |q|² − 2·q·x + |x|². The |x|²
                // term is precomputed per-doc at encode time
                // (using the *decoded* values, so this matches
                // exactly what the kernel reconstructs).
                let norms = self
                    .per_doc_norms
                    .expect("Sq8Kernel + L2Sq requires per_doc_norms");
                let x_norm_sq = norms[pos as usize];
                self.q_norm_sq - 2.0 * dot + x_norm_sq
            }
        }
    }
}

/// Cross-product reduction for `Sq8Kernel::distance_at`:
/// `Σ_d q_prime[d] * (code_bytes[d] as f32)` over the first `dim`
/// dimensions. Dispatches to the AVX-512 kernel (16-lane FMA with
/// `vpmovzxbd` u8 → i32 widen + `vcvtdq2ps`) when the runtime gate
/// passes; otherwise the `wide::f32x8` path that has shipped since
/// 012.
///
/// Inputs are pre-validated by `Sq8Kernel::distance_at`'s
/// `debug_assert_eq!(code_bytes.len(), self.dim)`. `q_prime.len()`
/// is guaranteed `== dim` by `Sq8Kernel::new`.
#[inline]
fn sq8_cross_product(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx512_enabled() {
        // SAFETY: gated on `avx512_enabled()` which requires `avx512f`.
        return unsafe { sq8_cross_product_avx512(q_prime, code_bytes, dim) };
    }
    sq8_cross_product_wide(q_prime, code_bytes, dim)
}

/// Portable `wide::f32x8` (256-bit) Sq8 cross product. Same per-
/// element math as the AVX-512 path, processed 8 lanes at a time
/// with a per-lane scalar `u8 as f32` widen.
#[inline]
fn sq8_cross_product_wide(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= dim {
        let qc: [f32; 8] = q_prime[i..i + 8].try_into().expect("q_prime[i..i+8] len 8");
        let mut bc = [0f32; 8];
        for (j, slot) in bc.iter_mut().enumerate() {
            *slot = code_bytes[i + j] as f32;
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        acc += qv * bv;
        i += 8;
    }
    let mut cross = acc.reduce_add();
    while i < dim {
        cross += q_prime[i] * (code_bytes[i] as f32);
        i += 1;
    }
    cross
}

/// AVX-512 Sq8 cross product. The win vs the `wide` kernel is two
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
unsafe fn sq8_cross_product_avx512(q_prime: &[f32], code_bytes: &[u8], dim: usize) -> f32 {
    use std::arch::x86_64::*;
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
        while i + 16 <= dim {
            // Load 16 u8 doc codes (one 128-bit lane) and widen
            // to 16 × i32 then convert to 16 × f32.
            let codes = _mm_loadu_si128(code_bytes.as_ptr().add(i) as *const __m128i);
            let codes_i32 = _mm512_cvtepu8_epi32(codes);
            let codes_f32 = _mm512_cvtepi32_ps(codes_i32);
            let q = _mm512_loadu_ps(q_prime.as_ptr().add(i));
            acc = _mm512_fmadd_ps(q, codes_f32, acc);
            i += 16;
        }
        let mut cross = _mm512_reduce_add_ps(acc);
        while i < dim {
            cross += q_prime[i] * (code_bytes[i] as f32);
            i += 1;
        }
        cross
    }
}

/// Distance against a vector stored as little-endian bf16 bytes
/// (2 bytes per dim). See [`distance_bytes_codec`] for context.
#[inline]
pub(crate) fn distance_bytes_bf16(metric: Metric, query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
    match metric {
        Metric::Cosine => 1.0 - dot_bf16_bytes(query, bytes),
        Metric::L2Sq => l2_sq_bf16_bytes(query, bytes),
        Metric::NegDot => -dot_bf16_bytes(query, bytes),
    }
}

/// Encode an fp32 value as bf16 (the upper 16 bits of the fp32
/// representation, with round-to-nearest-even on the low 16). NaN
/// inputs return a canonical bf16 NaN (sign-preserving). The
/// widening `bf16 → f32` is exact: read the u16 as the high 16
/// bits of an fp32, low 16 bits zero. So a value that round-trips
/// `fp32 → bf16 → fp32` differs from the input by at most one ULP
/// of bf16, i.e. relative error ≤ 2⁻⁸.
#[inline]
pub(crate) fn fp32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        // NaN — keep the sign + exponent and ensure the bf16 mantissa
        // is non-zero so it stays a NaN.
        ((bits >> 16) | 0x0040) as u16
    } else {
        // Round-to-nearest-even: add 0x7FFF, plus the LSB of the
        // truncated high half so exact midpoints round to even.
        let lsb = (bits >> 16) & 1;
        let bias = 0x7FFF_u32 + lsb;
        (bits.wrapping_add(bias) >> 16) as u16
    }
}

/// Widening `bf16 → f32`. Exact: reads the bf16 bit pattern as the
/// top 16 bits of an fp32, low 16 bits zero. NaN/Inf/Subnormals
/// pass through cleanly because the fp32 exponent field maps 1:1.
#[inline]
pub(crate) fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

/// bf16 dot product. Dispatches to the AVX-512 BF16 16-lane
/// `vdpbf16ps` kernel when the runtime gate passes (Sapphire Rapids+,
/// Zen 4+); otherwise the `wide::f32x8` widen-and-FMA kernel that
/// has shipped since 012.
#[inline]
fn dot_bf16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
    #[cfg(target_arch = "x86_64")]
    if has_bf16_dot() {
        // SAFETY: `has_bf16_dot()` returns true only on hosts with
        // both `avx512f` (foundation) and `avx512bf16` (VDPBF16PS).
        return unsafe { dot_bf16_bytes_avx512(query, bytes) };
    }
    dot_bf16_bytes_wide(query, bytes)
}

/// Portable `wide::f32x8` (256-bit) bf16-bytes dot product. Same
/// per-element math as `dot_bf16_bytes_avx512`, processed 8 lanes
/// at a time with a manual bf16 → f32 widen per lane. Universal
/// fallback for non-AVX-512BF16 hosts.
#[inline]
fn dot_bf16_bytes_wide(query: &[f32], bytes: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= query.len() {
        let qc: [f32; 8] = query[i..i + 8]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; 8];
        let off = i * 2;
        for (j, slot) in bc.iter_mut().enumerate() {
            let bf = u16::from_le_bytes([bytes[off + j * 2], bytes[off + j * 2 + 1]]);
            *slot = bf16_to_f32(bf);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        acc += qv * bv;
        i += 8;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * 2;
        let bf = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        sum += query[i] * bf16_to_f32(bf);
        i += 1;
    }
    sum
}

/// AVX-512 BF16 bf16-bytes dot product via `_mm512_dpbf16_ps`
/// (VDPBF16PS). One instruction performs 32 bf16-pair multiply-adds
/// into 16 f32 accumulator lanes — both the doubled vector width
/// **and** the native bf16 dot product motivate moving here.
///
/// The fp32 query side has to be packed down to bf16 inside the
/// loop because the natively-stored representation is bf16 on the
/// document side, and `vdpbf16ps` expects `__m512bh` for both
/// operands. We pack via `_mm512_cvtne2ps_pbh` (round-to-nearest-
/// even — exactly the same rounding the `fp32_to_bf16` encoder
/// uses, so the parity-with-fp32 tolerance is unchanged from the
/// `wide` kernel).
///
/// Reference implementations: oneDNN's `bf16_dot_kernel`, faer's
/// `pulp::Pulp::vsr_dpbf16_ps` (which we re-derive directly
/// because `pulp` doesn't expose VDPBF16PS without a feature
/// gate).
///
/// # Safety
///
/// Callers must ensure the target supports `avx512f` + `avx512bf16`
/// (the `_mm512_dpbf16_ps` and `_mm512_cvtne2ps_pbh` intrinsics).
/// `has_bf16_dot()` guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn dot_bf16_bytes_avx512(query: &[f32], bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n = query.len();
    debug_assert_eq!(bytes.len(), n * 2);

    // SAFETY: each iteration reads 32 fp32s from `query` (= 16 from
    // each `q_lo`/`q_hi` load) and 32 bf16s = 64 bytes from `bytes`
    // (one `_mm512_loadu_si512`). Both loads are gated by
    // `i + 32 <= n`, which keeps both windows inside their slices.
    // Loads are unaligned (`loadu` / `loadu_si512`) so we make no
    // alignment assumption on the caller's buffers. The bytes load
    // is treated as 32 × i16 (bf16 bit patterns), which is valid
    // because every bit pattern is a defined bf16 value (no
    // illegal encodings).
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            // 32 fp32 query lanes → two 16-lane vectors.
            let q_lo = _mm512_loadu_ps(query.as_ptr().add(i));
            let q_hi = _mm512_loadu_ps(query.as_ptr().add(i + 16));
            // Pack the two fp32 vectors into one bf16 register.
            // `_mm512_cvtne2ps_pbh(hi, lo)` rounds 32 fp32s to bf16
            // (round-to-nearest-even) and concatenates them lo-then-
            // hi. This matches `fp32_to_bf16`'s rounding mode so
            // the AVX-512 path has the same per-lane bf16-encoding
            // error as the wide / scalar reference.
            let q_bf16 = _mm512_cvtne2ps_pbh(q_hi, q_lo);
            // 64 bytes of doc-side bf16 = 32 bf16 lanes. The on-disk
            // layout is little-endian u16 per lane (encoded by
            // `fp32_to_bf16`), which is exactly the in-register
            // layout `__m512bh` expects on little-endian x86_64.
            let d_bits = _mm512_loadu_si512(bytes.as_ptr().add(i * 2) as *const __m512i);
            let d_bf16 = std::mem::transmute::<__m512i, __m512bh>(d_bits);
            acc = _mm512_dpbf16_ps(acc, q_bf16, d_bf16);
            i += 32;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        // Tail < 32: fall back to scalar bf16 widen. The longest
        // possible tail is 31 lanes; this is fast in practice
        // because callers' dims (384, 768, 1024, 1536) are all
        // multiples of 32.
        while i < n {
            let off = i * 2;
            let bf = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            sum += query[i] * bf16_to_f32(bf);
            i += 1;
        }
        sum
    }
}

/// bf16 squared-L2. Dispatches to the AVX-512 BF16 kernel when
/// available; otherwise the `wide` widen-and-FMA kernel.
///
/// Unlike `dot_bf16_bytes`, this kernel does not use VDPBF16PS
/// directly — `vdpbf16ps` is a fused multiply-add of bf16 pairs,
/// which doesn't fit the `(q − d)²` shape. The AVX-512 path
/// widens the doc bf16 to two fp32 vectors (cheap: one
/// `_mm512_cvtpbh_ps` per half-batch), subtracts the query, and
/// uses two `vfmadd231ps` to square-accumulate. Still ~1.7× faster
/// than the `wide::f32x8` baseline because the loop body
/// processes 16 lanes per FMA instead of 8 and the bf16 widen is
/// a single vector instruction instead of a per-lane scalar
/// `u16::from_le_bytes` + `bf16_to_f32`.
#[inline]
fn l2_sq_bf16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
    #[cfg(target_arch = "x86_64")]
    if has_bf16_dot() {
        // SAFETY: `has_bf16_dot()` guarantees `avx512f` + `avx512bf16`.
        return unsafe { l2_sq_bf16_bytes_avx512(query, bytes) };
    }
    l2_sq_bf16_bytes_wide(query, bytes)
}

/// Portable `wide::f32x8` (256-bit) bf16-bytes squared-L2. See
/// [`dot_bf16_bytes_wide`] for kernel shape; same widen, different
/// inner math.
#[inline]
fn l2_sq_bf16_bytes_wide(query: &[f32], bytes: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + 8 <= query.len() {
        let qc: [f32; 8] = query[i..i + 8]
            .try_into()
            .expect("slice [i..i+8] has length 8");
        let mut bc = [0f32; 8];
        let off = i * 2;
        for (j, slot) in bc.iter_mut().enumerate() {
            let bf = u16::from_le_bytes([bytes[off + j * 2], bytes[off + j * 2 + 1]]);
            *slot = bf16_to_f32(bf);
        }
        let qv = f32x8::from(qc);
        let bv = f32x8::from(bc);
        let d = qv - bv;
        acc += d * d;
        i += 8;
    }
    let mut sum = acc.reduce_add();
    while i < query.len() {
        let off = i * 2;
        let bf = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let d = query[i] - bf16_to_f32(bf);
        sum += d * d;
        i += 1;
    }
    sum
}

/// AVX-512 BF16 bf16-bytes squared-L2. Widens the doc bf16 to two
/// fp32 `__m512` halves via `_mm512_cvtpbh_ps` (exact widen — bf16
/// is the upper 16 bits of fp32), subtracts the matching query
/// halves, and FMAs the squared difference into the accumulator.
///
/// # Safety
///
/// Same contract as [`dot_bf16_bytes_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn l2_sq_bf16_bytes_avx512(query: &[f32], bytes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    let n = query.len();
    debug_assert_eq!(bytes.len(), n * 2);

    // SAFETY: bounds reasoning matches `dot_bf16_bytes_avx512`.
    // The bf16-as-i16 transmute is sound because every bit pattern
    // is a valid bf16 (no illegal encodings); the widen is just a
    // bit reinterpretation (low 16 bits of fp32 zeroed).
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + 32 <= n {
            // 32 doc bf16 lanes split into two fp32 halves (16 each).
            let d_bits = _mm512_loadu_si512(bytes.as_ptr().add(i * 2) as *const __m512i);
            // Bottom 256 bits → bf16 lanes 0..16 → fp32 vector (lo)
            let d_lo_bh = std::mem::transmute::<__m256i, __m256bh>(_mm512_castsi512_si256(d_bits));
            // Top 256 bits → bf16 lanes 16..32 → fp32 vector (hi)
            let d_hi_bh =
                std::mem::transmute::<__m256i, __m256bh>(_mm512_extracti64x4_epi64(d_bits, 1));
            let d_lo = _mm512_cvtpbh_ps(d_lo_bh);
            let d_hi = _mm512_cvtpbh_ps(d_hi_bh);

            let q_lo = _mm512_loadu_ps(query.as_ptr().add(i));
            let q_hi = _mm512_loadu_ps(query.as_ptr().add(i + 16));

            let diff_lo = _mm512_sub_ps(q_lo, d_lo);
            let diff_hi = _mm512_sub_ps(q_hi, d_hi);
            acc = _mm512_fmadd_ps(diff_lo, diff_lo, acc);
            acc = _mm512_fmadd_ps(diff_hi, diff_hi, acc);

            i += 32;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < n {
            let off = i * 2;
            let bf = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
            let d = query[i] - bf16_to_f32(bf);
            sum += d * d;
            i += 1;
        }
        sum
    }
}

/// In-place L2-normalize. Zero vectors stay zero (no division).
pub fn normalize(v: &mut [f32]) {
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 0.0 {
        let inv = 1.0 / mag;
        for x in v.iter_mut() {
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

    // --- bf16 round-trip + distance ------------------------------------

    fn encode_bf16(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for &x in values {
            out.extend_from_slice(&fp32_to_bf16(x).to_le_bytes());
        }
        out
    }

    #[test]
    fn bf16_round_trip_exact_for_representable_values() {
        // Values whose low 16 mantissa bits are already zero round-trip
        // exactly through bf16: 0, 1, integers with bf16 mantissa
        // precision, powers of two.
        for &x in &[0.0f32, 1.0, -1.0, 2.0, 4.0, 0.5, -0.5] {
            let bf = fp32_to_bf16(x);
            assert_eq!(bf16_to_f32(bf), x, "value {x} did not round-trip");
        }
    }

    #[test]
    fn bf16_round_trip_within_relative_tolerance() {
        // Arbitrary fp32s round-trip within ~2⁻⁸ relative error.
        for &x in &[1.234e-3f32, 0.123_456_7, 1.5e3, -7.7e-2, 42.0] {
            let bf = fp32_to_bf16(x);
            let r = bf16_to_f32(bf);
            let err = ((r - x) / x).abs();
            assert!(err <= 1.0 / 128.0, "value {x}: round-trip err {err}");
        }
    }

    #[test]
    fn bf16_ties_round_to_even() {
        // Midpoint between two bf16 values: exact-half mantissa.
        // 1.0 has bits 0x3F80_0000. A midpoint is 0x3F80_8000 (the
        // mantissa LSB-of-bf16 is 0, low 16 = 0x8000). Tie should
        // round DOWN to 1.0 (even mantissa), not up.
        let mid_down = f32::from_bits(0x3F80_8000);
        assert_eq!(bf16_to_f32(fp32_to_bf16(mid_down)), 1.0);
        // Next bf16 is 0x3F81 → 1.0078125; midpoint at 0x3F81_8000
        // rounds UP to 1.015625 because 0x3F81's mantissa LSB is 1
        // (so down would be odd, up is even).
        let mid_up = f32::from_bits(0x3F81_8000);
        assert_eq!(
            bf16_to_f32(fp32_to_bf16(mid_up)),
            f32::from_bits(0x3F82_0000)
        );
    }

    #[test]
    fn distance_bytes_bf16_matches_fp32_within_tolerance() {
        let q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.3).collect();
        let bytes = encode_bf16(&v);
        for m in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let d_ref = distance(m, &q, &v);
            let d_bf16 = distance_bytes_bf16(m, &q, &bytes);
            let abs_err = (d_ref - d_bf16).abs();
            let rel_err = abs_err / d_ref.abs().max(1e-6);
            // bf16 has 8 bits of mantissa → per-lane relative error
            // ~2⁻⁸ ≈ 4e-3. Sum-of-products amplifies by √dim ≈ 4.
            assert!(
                rel_err <= 2e-2 || abs_err <= 1e-3,
                "metric {m:?}: bf16 {d_bf16} vs fp32 {d_ref} (rel {rel_err})"
            );
        }
    }

    #[test]
    fn distance_bytes_codec_dispatches_correctly() {
        let q: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let v: Vec<f32> = (0..8).map(|i| (i as f32) * 0.2 - 0.5).collect();
        let bytes_fp32: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes().into_iter()).collect();
        let bytes_bf16 = encode_bf16(&v);

        let d_fp32 = distance_bytes_codec(Metric::L2Sq, RerankCodec::Fp32, &q, &bytes_fp32);
        let d_bf16 = distance_bytes_codec(Metric::L2Sq, RerankCodec::Bf16, &q, &bytes_bf16);

        // fp32 path must equal the plain f32 reference exactly.
        assert_eq!(d_fp32, distance(Metric::L2Sq, &q, &v));
        // bf16 path must be within tolerance of the reference.
        let err = (d_bf16 - d_fp32).abs();
        assert!(err <= 5e-3, "bf16 dispatch err {err}");
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

    #[test]
    fn sq8_kernel_dot_matches_decoded_reference() {
        let dim = 16usize;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.05 - 0.3).collect();
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 + (i as f32) * 0.002).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -1.0 + (i as f32) * 0.1).collect();
        let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
        let decoded = decode_sq8(&codes, dim, &scale, &offset);

        for m in [Metric::Cosine, Metric::NegDot] {
            let want = distance(m, &query, &decoded);
            let kernel = Sq8Kernel::new(m, &query, &scale, &offset, None);
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

        let kernel = Sq8Kernel::new(Metric::L2Sq, &query, &scale, &offset, Some(&per_doc_norms));

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
            let norms_arg: Option<&[f32]> = match m {
                Metric::L2Sq => Some(&per_doc_norms),
                _ => None,
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
                let want_decoded = distance(m, &query, decoded_doc);
                // Kernel must match the decoded reference very
                // tightly — it's doing the same math, just fused
                // through the per-query precompute. Difference
                // from fp32 is the quantization error itself.
                assert!(
                    (got - want_decoded).abs() <= 1e-3,
                    "metric {m:?} pos {pos}: kernel {got} vs decoded ref {want_decoded}"
                );
                let rel = (got - want_fp32).abs() / want_fp32.abs().max(1e-2);
                assert!(
                    rel <= 0.1 || (got - want_fp32).abs() <= 1.0,
                    "metric {m:?} pos {pos}: Sq8 {got} vs fp32 {want_fp32} (rel {rel})"
                );
            }
        }
    }

    #[test]
    fn distance_bytes_bf16_handles_tail_dim_not_multiple_of_8() {
        // Dim 11: 1 SIMD chunk of 8 + scalar tail of 3. Both branches
        // must round-trip values consistently; the test catches a tail
        // path that skipped bf16 widening (would surface as an
        // order-of-magnitude error, not the ~0.3 % bf16 round-trip
        // error we tolerate here).
        let q: Vec<f32> = (0..11).map(|i| (i as f32) * 0.1).collect();
        let v: Vec<f32> = (0..11).map(|i| (i as f32) * 0.2 + 0.1).collect();
        let bytes = encode_bf16(&v);
        let d_ref = distance(Metric::L2Sq, &q, &v);
        let d_bf16 = distance_bytes_bf16(Metric::L2Sq, &q, &bytes);
        let rel = (d_ref - d_bf16).abs() / d_ref.abs().max(1e-6);
        assert!(
            rel <= 1e-2,
            "tail-dim bf16 {d_bf16} vs fp32 {d_ref} (rel {rel})"
        );
    }

    // --- AVX-512 parity (plan 014 Phase 1, fp32) ------------------------

    /// Generate a pseudo-random `f32` vector. Deterministic — uses the
    /// same monotone-noise pattern as the planted-cluster test fixtures
    /// elsewhere in this file so failures are reproducible.
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

    // --- AVX-512 parity (plan 014 Phase 1, bf16) ------------------------

    /// AVX-512 BF16 `dot_bf16_bytes` agrees with the `wide` baseline
    /// across length sweep covering the 32-lane unroll boundary
    /// and a representative span of tail sizes (the AVX-512 kernel's
    /// tail is anything `< 32`, fairly large compared to the 16-lane
    /// fp32 kernel).
    ///
    /// Tolerance: looser than the fp32 parity bound because the
    /// AVX-512 kernel and the wide kernel both round the query side
    /// to bf16 differently — the wide kernel doesn't (it does the
    /// FMA in fp32 against widened bf16 doc lanes), the AVX-512
    /// kernel does (VDPBF16PS expects both operands as bf16, so the
    /// fp32 query is packed via `vcvtne2ps2bf16` round-to-nearest-
    /// even on every iteration). Net per-lane error: bounded by
    /// one bf16 ULP (~2⁻⁸ relative) on top of the doc-side
    /// quantization that already exists in both paths.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn dot_bf16_bytes_avx512_matches_wide_across_lengths() {
        if !has_bf16_dot() {
            eprintln!(
                "dot_bf16_bytes_avx512_matches_wide_across_lengths: skipped, no AVX-512 BF16"
            );
            return;
        }
        for dim in [1usize, 7, 16, 31, 32, 33, 64, 95, 96, 97, 128, 384] {
            let query = fake_vec(dim, 0xB16F);
            let doc_f32 = fake_vec(dim, 0xD0C0);
            let bytes = encode_bf16(&doc_f32);
            let want = dot_bf16_bytes_wide(&query, &bytes);
            // SAFETY: gated on has_bf16_dot() above.
            let got = unsafe { dot_bf16_bytes_avx512(&query, &bytes) };
            // Tolerance: 1 bf16 ULP per lane (≈ 2⁻⁸ relative) on
            // the query-side rounding, accumulated √dim ways.
            let tol = 5e-3 * want.abs().max(1.0) + 1e-5 * (dim as f32).sqrt();
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: bf16 avx512 {got} vs bf16 wide {want} (tol {tol})"
            );
        }
    }

    /// AVX-512 BF16 `l2_sq_bf16_bytes` agrees with the `wide`
    /// baseline. The AVX-512 path widens bf16 doc lanes to fp32
    /// before subtracting (exact widen — bf16 IS the upper 16 bits
    /// of fp32, low 16 bits zero), so the only difference vs the
    /// wide kernel is reduction order. Tolerance accordingly is
    /// the same √dim-scaled ULP bound as the fp32 `l2_sq_avx512`
    /// parity test.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn l2_sq_bf16_bytes_avx512_matches_wide_across_lengths() {
        if !has_bf16_dot() {
            eprintln!(
                "l2_sq_bf16_bytes_avx512_matches_wide_across_lengths: skipped, no AVX-512 BF16"
            );
            return;
        }
        for dim in [1usize, 7, 16, 31, 32, 33, 64, 95, 96, 97, 128, 384] {
            let query = fake_vec(dim, 0xE2C5);
            let doc_f32 = fake_vec(dim, 0x9501);
            let bytes = encode_bf16(&doc_f32);
            let want = l2_sq_bf16_bytes_wide(&query, &bytes);
            // SAFETY: gated on has_bf16_dot() above.
            let got = unsafe { l2_sq_bf16_bytes_avx512(&query, &bytes) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: bf16 l2_sq avx512 {got} vs bf16 l2_sq wide {want} (tol {tol})"
            );
        }
    }

    /// AVX-512 `sq8_cross_product` agrees with the `wide` baseline
    /// across a length sweep. The cross product is `Σ q_prime[d] *
    /// (code[d] as f32)` so values are integer-magnitude on the
    /// doc side — exact widen, reduction-order is the only divergence.
    /// Tolerance is correspondingly tight.
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn sq8_cross_product_avx512_matches_wide_across_lengths() {
        if !avx512_enabled() {
            eprintln!("sq8_cross_product_avx512_matches_wide_across_lengths: skipped, no AVX-512");
            return;
        }
        for dim in [1usize, 7, 15, 16, 17, 31, 32, 33, 64, 96, 128, 384, 768] {
            let q_prime: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.013 - 0.4).collect();
            let codes: Vec<u8> = (0..dim).map(|i| ((i * 17 + 3) % 256) as u8).collect();
            let want = sq8_cross_product_wide(&q_prime, &codes, dim);
            // SAFETY: gated on avx512_enabled() above.
            let got = unsafe { sq8_cross_product_avx512(&q_prime, &codes, dim) };
            let tol = 1e-5 * want.abs().max(1.0);
            assert!(
                (want - got).abs() <= tol,
                "dim {dim}: sq8 avx512 {got} vs sq8 wide {want} (tol {tol})"
            );
        }
    }

    /// Public `dot_bf16_bytes` dispatches transparently — returns
    /// the same numeric value on this host regardless of whether
    /// the AVX-512 path is taken. Within the same parity tolerance
    /// as the direct-call test above.
    #[test]
    fn public_dot_bf16_bytes_dispatches_consistently() {
        for &dim in &[7usize, 32, 33, 384] {
            let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let doc_f32: Vec<f32> = (0..dim).map(|i| ((i * 3) as f32) * 0.02 - 0.1).collect();
            let bytes = encode_bf16(&doc_f32);
            let public_result = dot_bf16_bytes(&query, &bytes);
            let wide_result = dot_bf16_bytes_wide(&query, &bytes);
            let tol = 5e-3 * wide_result.abs().max(1.0) + 1e-5 * (dim as f32).sqrt();
            assert!(
                (public_result - wide_result).abs() <= tol,
                "dim {dim}: dot_bf16_bytes() {public_result} vs dot_bf16_bytes_wide() {wide_result} (tol {tol})"
            );
        }
    }

    // --- AVX-512 microbench (plan 014 — run by hand) -------------------
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
    fn avx512_microbench_bf16_kernels() {
        if !has_bf16_dot() {
            eprintln!("avx512_microbench: skipped, no AVX-512 BF16 on this host");
            return;
        }
        eprintln!();
        eprintln!("### bf16 distance kernel — AVX-512 BF16 (VDPBF16PS) vs wide (ns per call)\n");
        eprintln!("| kernel | dim | wide ns | avx512_bf16 ns | speedup |");
        eprintln!("|--------|----:|--------:|---------------:|--------:|");

        use std::hint::black_box;
        for &dim in realistic_dims() {
            let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001 - 0.5).collect();
            let doc: Vec<f32> = (0..dim).map(|i| ((i + 7) as f32) * 0.0017 - 0.3).collect();
            let bytes = encode_bf16(&doc);
            let iters: u32 = (10_000_000u64 / (dim as u64).max(1)).max(50_000) as u32;

            let wide_ns = time_ns(iters, || {
                dot_bf16_bytes_wide(black_box(&query), black_box(&bytes))
            });
            // SAFETY: gated on has_bf16_dot() above.
            let avx_ns = time_ns(iters, || unsafe {
                dot_bf16_bytes_avx512(black_box(&query), black_box(&bytes))
            });
            eprintln!(
                "| `distance::dot_bf16_bytes` | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );

            let wide_ns = time_ns(iters, || {
                l2_sq_bf16_bytes_wide(black_box(&query), black_box(&bytes))
            });
            let avx_ns = time_ns(iters, || unsafe {
                l2_sq_bf16_bytes_avx512(black_box(&query), black_box(&bytes))
            });
            eprintln!(
                "| `distance::l2_sq_bf16_bytes` | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
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
                sq8_cross_product_wide(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            // SAFETY: gated on avx512_enabled() above.
            let avx_ns = time_ns(iters, || unsafe {
                sq8_cross_product_avx512(black_box(&q_prime), black_box(&codes), black_box(dim))
            });
            eprintln!(
                "| `Sq8Kernel::distance_at` (cross-product) | {dim} | {:>7.1} | {:>7.1} | {:>5.2}× |",
                wide_ns,
                avx_ns,
                wide_ns / avx_ns,
            );
        }
    }
}
