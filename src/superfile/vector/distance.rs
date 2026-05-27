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

/// f32 dot product. Vectorised in 8-lane chunks; a scalar tail handles
/// inputs whose length isn't a multiple of 8.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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

/// Squared Euclidean distance. Vectorised; scalar tail.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
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
/// representation. The `Fp32` fast path reuses [`distance_bytes`];
/// `Bf16` widens 8 bf16 lanes to f32 per inner step and reuses the
/// same `f32x8` math.
///
/// `Sq8` doesn't have a flat entry point because the decode needs the
/// candidate cluster's scale/offset (and per-doc norm for L2Sq). Sq8
/// callers go through [`Sq8Kernel`] which captures those once per query
/// and cluster. `None` panics here because its column carries no
/// `full[]` bytes to score.
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
            unreachable!("distance_bytes_codec called with Sq8; Sq8 rerank goes through Sq8Kernel")
        }
        RerankCodec::RabitqOnly => unreachable!(
            "distance_bytes_codec called with RabitqOnly; \
             RabitqOnly columns have no full[] region"
        ),
    }
}

/// Sq8 rerank context. Captures the per-column quantizer and the
/// per-query precomputes that fold scale/offset into the query side so
/// the per-doc inner loop is a plain u8-to-f32 widen + SIMD dot.
pub(crate) struct Sq8Kernel<'a> {
    metric: Metric,
    dim: usize,
    q_prime: Vec<f32>,
    q_dot_offset: f32,
    q_norm_sq: f32,
    per_doc_norms: Option<&'a [f32]>,
}

impl<'a> Sq8Kernel<'a> {
    /// Build the per-query kernel. `scale` + `offset` are the per-dim
    /// quantizer arrays from the column's `codec_meta`. `per_doc_norms`
    /// is `Some` iff the column metric is L2Sq or Cosine.
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
        let mut q_prime = vec![0.0f32; dim];
        let mut q_dot_offset_acc = f32x8::ZERO;
        let mut i = 0;
        while i + 8 <= dim {
            let qc = f32x8::from(<[f32; 8]>::try_from(&query[i..i + 8]).expect("len-8 slice"));
            let sc = f32x8::from(<[f32; 8]>::try_from(&scale[i..i + 8]).expect("len-8 slice"));
            let oc = f32x8::from(<[f32; 8]>::try_from(&offset[i..i + 8]).expect("len-8 slice"));
            let qp = qc * sc;
            q_prime[i..i + 8].copy_from_slice(&qp.to_array());
            q_dot_offset_acc += qc * oc;
            i += 8;
        }
        let mut q_dot_offset = q_dot_offset_acc.reduce_add();
        while i < dim {
            q_prime[i] = query[i] * scale[i];
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
            q_prime,
            q_dot_offset,
            q_norm_sq,
            per_doc_norms,
        }
    }

    /// Distance for one rerank candidate at position `pos`, with
    /// `dim` u8 codes at `code_bytes`. Smaller = closer for every
    /// metric.
    #[inline]
    pub fn distance_at(&self, pos: u32, code_bytes: &[u8]) -> f32 {
        debug_assert_eq!(code_bytes.len(), self.dim);
        let mut acc = f32x8::ZERO;
        let mut i = 0;
        while i + 8 <= self.dim {
            let qc: [f32; 8] = self.q_prime[i..i + 8]
                .try_into()
                .expect("q_prime[i..i+8] len 8");
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
        while i < self.dim {
            cross += self.q_prime[i] * (code_bytes[i] as f32);
            i += 1;
        }
        let dot = cross + self.q_dot_offset;
        match self.metric {
            Metric::Cosine => {
                let norms = self
                    .per_doc_norms
                    .expect("Sq8Kernel + Cosine requires per_doc_norms");
                let x_norm = norms[pos as usize].sqrt();
                if x_norm > 0.0 {
                    1.0 - dot / x_norm
                } else {
                    1.0 - dot
                }
            }
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                let norms = self
                    .per_doc_norms
                    .expect("Sq8Kernel + L2Sq requires per_doc_norms");
                let x_norm_sq = norms[pos as usize];
                self.q_norm_sq - 2.0 * dot + x_norm_sq
            }
        }
    }
}

/// Distance against a vector stored as little-endian bf16 bytes
/// (2 bytes per dim). See [`distance_bytes_codec`] for context.
#[inline]
pub(crate) fn distance_bytes_bf16(metric: Metric, query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
    match metric {
        Metric::Cosine => {
            let (dot, norm_sq) = dot_and_norm_sq_bf16_bytes(query, bytes);
            let norm = norm_sq.sqrt();
            if norm > 0.0 {
                1.0 - dot / norm
            } else {
                1.0 - dot
            }
        }
        Metric::L2Sq => l2_sq_bf16_bytes(query, bytes),
        Metric::NegDot => -dot_bf16_bytes(query, bytes),
    }
}

/// Encode an fp32 value as bf16 with round-to-nearest-even on the
/// truncated low 16 bits. NaN inputs return a sign-preserving bf16
/// NaN. Widening back to f32 is exact: the bf16 bits become the top
/// 16 bits of the fp32 representation and the low 16 bits are zero.
#[inline]
pub(crate) fn fp32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if (bits & 0x7FFF_FFFF) > 0x7F80_0000 {
        ((bits >> 16) | 0x0040) as u16
    } else {
        let lsb = (bits >> 16) & 1;
        let bias = 0x7FFF_u32 + lsb;
        (bits.wrapping_add(bias) >> 16) as u16
    }
}

/// Widen bf16 to f32 exactly.
#[inline]
pub(crate) fn bf16_to_f32(bf: u16) -> f32 {
    f32::from_bits((bf as u32) << 16)
}

#[inline]
fn dot_bf16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
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

#[inline]
fn dot_and_norm_sq_bf16_bytes(query: &[f32], bytes: &[u8]) -> (f32, f32) {
    debug_assert_eq!(query.len() * 2, bytes.len());
    let mut dot_acc = f32x8::ZERO;
    let mut norm_acc = f32x8::ZERO;
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
        dot_acc += qv * bv;
        norm_acc += bv * bv;
        i += 8;
    }
    let mut dot_sum = dot_acc.reduce_add();
    let mut norm_sum = norm_acc.reduce_add();
    while i < query.len() {
        let off = i * 2;
        let bf = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let x = bf16_to_f32(bf);
        dot_sum += query[i] * x;
        norm_sum += x * x;
        i += 1;
    }
    (dot_sum, norm_sum)
}

#[inline]
fn l2_sq_bf16_bytes(query: &[f32], bytes: &[u8]) -> f32 {
    debug_assert_eq!(query.len() * 2, bytes.len());
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

/// In-place L2-normalize. Zero vectors stay zero (no division).
///
/// Both passes (sum-of-squares and per-lane scaling) are vectorised
/// in 8-lane `f32x8` chunks; a scalar tail handles inputs whose
/// length isn't a multiple of 8. Same SIMD-then-scalar shape as
/// [`dot`] and [`l2_sq`] above; load-bearing for the build-time
/// "normalize then rotate then bit-quantize" pipeline where this
/// runs once per input vector.
pub fn normalize(v: &mut [f32]) {
    let mag = {
        let mut acc = f32x8::ZERO;
        let mut tail_acc: f32 = 0.0;
        let chunks = v.chunks_exact(8);
        let tail = chunks.remainder();
        for c in chunks {
            let lane = f32x8::from(
                <[f32; 8]>::try_from(c).expect("chunks_exact(8) yields slices of length 8"),
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
        let mut chunks = v.chunks_exact_mut(8);
        for c in chunks.by_ref() {
            let lane = f32x8::from(
                <[f32; 8]>::try_from(&*c).expect("chunks_exact_mut(8) yields slices of length 8"),
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
            let d_ref = match m {
                Metric::Cosine => {
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    1.0 - dot(&q, &v) / norm
                }
                _ => distance(m, &q, &v),
            };
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

        let decoded_norm: f32 = decoded.iter().map(|x| x * x).sum();
        for m in [Metric::Cosine, Metric::NegDot] {
            let norms = [decoded_norm];
            let want = match m {
                Metric::Cosine => 1.0 - dot(&query, &decoded) / decoded_norm.sqrt(),
                _ => distance(m, &query, &decoded),
            };
            let norms_arg = if matches!(m, Metric::Cosine) {
                Some(&norms[..])
            } else {
                None
            };
            let kernel = Sq8Kernel::new(m, &query, &scale, &offset, norms_arg);
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

        // Per-doc norms for the L2Sq/Cosine branches — indexed by pos
        // matching the builder's contract.
        let per_doc_norms: Vec<f32> = decoded_all
            .chunks_exact(dim)
            .map(|row| row.iter().map(|x| x * x).sum::<f32>())
            .collect();

        for m in [Metric::Cosine, Metric::L2Sq, Metric::NegDot] {
            let norms_arg: Option<&[f32]> = match m {
                Metric::L2Sq | Metric::Cosine => Some(&per_doc_norms),
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
                let fp32_doc = &corpus[(pos as usize) * dim..(pos as usize + 1) * dim];
                let want_fp32 = match m {
                    Metric::Cosine => {
                        let norm = fp32_doc.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot(&query, fp32_doc) / norm
                    }
                    _ => distance(m, &query, fp32_doc),
                };
                let want_decoded = match m {
                    Metric::Cosine => {
                        let norm = per_doc_norms[pos as usize].sqrt();
                        1.0 - dot(&query, decoded_doc) / norm
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
}
