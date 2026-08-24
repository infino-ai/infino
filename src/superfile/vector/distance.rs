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
#[cfg(test)]
use std::sync::Arc;

use wide::f32x8;

use crate::superfile::vector::rerank_codec::{
    RerankCodec, SQ16_CODE_MAX, SQ16_FIXED_OFFSET, SQ16_FIXED_SCALE,
};
#[cfg(target_arch = "x86_64")]
use crate::superfile::vector::simd_dispatch::{avx2_enabled, avx512_enabled};

/// Residual quantization step divisor for [`RerankCodec::Sq8Residual`].
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

/// Centroid batch width for the transposed routing cache. Chosen to match the
/// AVX-512 f32 lane count; the portable fallback consumes each 16-lane block as
/// two 8-lane halves.
pub(crate) const CENTROID_BATCH_LANES: usize = AVX512_F32_LANES;

/// Byte width of one little-endian `f32`. A byte-backed vector of
/// dimension `d` occupies `d * F32_BYTES` bytes.
const F32_BYTES: usize = 4;

/// Cosine distance is `COSINE_DISTANCE_BASE - dot` on unit vectors,
/// so smaller means closer without re-normalizing at query time.
pub(crate) const COSINE_DISTANCE_BASE: f32 = 1.0;

/// Cross-term coefficient in the squared-L2 identity
/// `‖q − x‖² = ‖q‖² − L2_CROSS_TERM_COEFF·(q·x) + ‖x‖²`, used by the
/// Sq8 rerank kernels that reconstruct L2 from a fused dot product.
pub(crate) const L2_CROSS_TERM_COEFF: f32 = 2.0;

/// Half-code slack before an out-of-grid `Sq16Adaptive` component counts as
/// clamped, mirroring the residual family's tripwire tolerance.
const SQ16_CLAMP_DETECT_SLACK_CODES: f32 = 0.5;

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

/// Build the block-transposed centroid cache used by
/// [`nearest_two_centroids_transposed`].
///
/// Canonical centroid storage is cluster-major (`centroid -> dim`). This derived
/// cache is block-transposed in 16-centroid groups:
/// `block -> dim -> centroid_lane`. That gives the AVX-512 hot loop contiguous
/// loads for 16 centroid lanes at one query dimension.
pub(crate) fn transpose_centroids_cluster_major(
    centroids: &[f32],
    n_cent: usize,
    dim: usize,
) -> Vec<f32> {
    debug_assert_eq!(centroids.len(), n_cent * dim);
    let n_blocks = n_cent.div_ceil(CENTROID_BATCH_LANES);
    let mut transposed = vec![0f32; n_blocks * dim * CENTROID_BATCH_LANES];
    for block in 0..n_blocks {
        let centroid_base = block * CENTROID_BATCH_LANES;
        let block_base = block * dim * CENTROID_BATCH_LANES;
        for d in 0..dim {
            let dst = block_base + d * CENTROID_BATCH_LANES;
            for lane in 0..CENTROID_BATCH_LANES {
                let centroid = centroid_base + lane;
                if centroid < n_cent {
                    transposed[dst + lane] = centroids[centroid * dim + d];
                }
            }
        }
    }
    transposed
}

/// Widen a score by a relative slack: the shared "score window" used by the
/// hidden/user cell-routing cutoff (probe cells while their score stays
/// within `slack` of the nearest) and by replica closure (a cell is a
/// replica candidate while the row's distance stays within the closure
/// ratio of its primary). One definition keeps routing and replication
/// interpreting boundary geometry identically.
#[inline]
pub(crate) fn relative_score_window(base: f32, slack: f32) -> f32 {
    base + base.abs().max(f32::EPSILON) * slack.max(0.0)
}

/// Insert `(centroid, score)` into an ascending top-`k` vec, keeping the
/// lowest-index winner on equal scores (matching the naive scalar scan).
/// The one ranked-insertion shared by the blocked top-k reducers.
#[inline]
pub(crate) fn insert_ranked(top: &mut Vec<(u32, f32)>, k: usize, centroid: u32, score: f32) {
    if top.len() == k && score >= top[k - 1].1 {
        return;
    }
    let pos = top
        .iter()
        .position(|&(_, s)| score < s)
        .unwrap_or(top.len());
    top.insert(pos, (centroid, score));
    top.truncate(k);
}

/// Drive the blocked centroid scorers over every block of a transposed
/// centroid cache, feeding each block's lane scores to `reduce(base, scores)`.
/// The single owner of the AVX-512 / `wide` dispatch skeleton — every
/// nearest-centroid shape (argmin, top-k) is a reducer over this driver.
#[inline]
fn for_each_centroid_block_scores(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    n_cent: usize,
    dim: usize,
    mut reduce: impl FnMut(usize, &[f32]),
) {
    debug_assert_eq!(query.len(), dim);
    debug_assert_eq!(
        transposed.len(),
        n_cent.div_ceil(CENTROID_BATCH_LANES) * dim * CENTROID_BATCH_LANES
    );
    let n_blocks = n_cent.div_ceil(CENTROID_BATCH_LANES);

    #[cfg(target_arch = "x86_64")]
    if avx512_enabled() {
        for block in 0..n_blocks {
            // SAFETY: gated by runtime CPUID detection in `avx512_enabled()`.
            let scores = unsafe {
                score_centroid_block16_transposed_avx512(metric, query, transposed, dim, block)
            };
            reduce(block * CENTROID_BATCH_LANES, &scores);
        }
        return;
    }

    for block in 0..n_blocks {
        let base_centroid = block * CENTROID_BATCH_LANES;
        for half in 0..CENTROID_BATCH_LANES / F32X8_LANES {
            let lane_offset = half * F32X8_LANES;
            let scores = score_centroid_block8_transposed_wide(
                metric,
                query,
                transposed,
                dim,
                block,
                lane_offset,
            );
            reduce(base_centroid + lane_offset, &scores);
        }
    }
}

/// Return the single closest centroid in a block-transposed fp32 centroid
/// cache: the k-means assign step's hot call. Same blocked scoring kernels
/// as [`nearest_k_centroids_transposed`], reduced with a scalar best tracker
/// (no per-call allocation) and the same tie-breaking as the naive scalar
/// loop — lowest centroid index wins on equal scores.
pub(crate) fn nearest_centroid_transposed(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    n_cent: usize,
    dim: usize,
) -> (u32, f32) {
    debug_assert!(n_cent > 0);
    let mut best = (0u32, f32::INFINITY);
    for_each_centroid_block_scores(metric, query, transposed, n_cent, dim, |base, scores| {
        for (lane, &score) in scores.iter().enumerate() {
            let centroid = base + lane;
            if centroid < n_cent && score < best.1 {
                best = (centroid as u32, score);
            }
        }
    });
    best
}

/// Return the closest two centroids in a block-transposed fp32 centroid cache.
/// Thin wrapper over [`nearest_k_centroids_transposed`] with `k = 2` — kept
/// for the scalar-reference equivalence tests that pin the top-k reduction.
#[cfg(test)]
pub(crate) fn nearest_two_centroids_transposed(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    n_cent: usize,
    dim: usize,
    counts: Option<&[u32]>,
) -> Option<((u32, f32), Option<(u32, f32)>)> {
    let top = nearest_k_centroids_transposed(metric, query, transposed, n_cent, dim, counts, 2);
    let mut it = top.into_iter();
    it.next().map(|best| (best, it.next()))
}

/// Return the closest `k` centroids (ascending by score) in a block-transposed
/// fp32 centroid cache. Full blocks score multiple centroids per SIMD
/// register: AVX-512 scores 16 centroids from contiguous loads, and the
/// portable fallback scores each block as two contiguous `wide::f32x8`
/// halves. `counts = Some(..)` skips zero-count centroids; `None` keeps every
/// centroid eligible. `k` is expected to be small (replica closure / boundary
/// assignment); the reduction is an insertion top-k.
pub(crate) fn nearest_k_centroids_transposed(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    n_cent: usize,
    dim: usize,
    counts: Option<&[u32]>,
    k: usize,
) -> Vec<(u32, f32)> {
    debug_assert!(counts.is_none_or(|counts| counts.len() >= n_cent));
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k.saturating_add(1));
    if k == 0 {
        return top;
    }
    for_each_centroid_block_scores(metric, query, transposed, n_cent, dim, |base, scores| {
        for (lane, &score) in scores.iter().enumerate() {
            let centroid = base + lane;
            if centroid < n_cent && centroid_included(counts, centroid) {
                insert_ranked(&mut top, k, centroid as u32, score);
            }
        }
    });
    top
}

#[inline]
fn centroid_included(counts: Option<&[u32]>, centroid: usize) -> bool {
    counts.is_none_or(|counts| counts[centroid] != 0)
}

/// Score `query` against every centroid in a block-transposed fp32 centroid
/// cache and return the dense per-centroid score vector (`scores[c]` is
/// centroid `c`'s distance). The full-scan reducer over
/// [`for_each_centroid_block_scores`] — callers that need every score in
/// centroid order (cell ranking, per-cluster candidate emission) use this
/// instead of hand-rolling a `(0..n_cent).map(distance)` loop.
pub(crate) fn all_centroid_scores_transposed(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    n_cent: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; n_cent];
    for_each_centroid_block_scores(metric, query, transposed, n_cent, dim, |base, scores| {
        for (lane, &score) in scores.iter().enumerate() {
            let centroid = base + lane;
            if centroid < n_cent {
                out[centroid] = score;
            }
        }
    });
    out
}

/// Top-`k` nearest centroids over the *row-major fp32-bytes* layout — the
/// on-disk shape of a superfile subsection's centroid region, scored in
/// place with no decode or transpose copy. The row-major sibling of
/// [`nearest_k_centroids_transposed`]: same ascending order, same
/// deterministic lowest-index tie-break via [`insert_ranked`]. These are the
/// only two centroid-scan owners; every caller routes through one of them
/// according to its memory layout.
pub(crate) fn nearest_k_centroids_bytes(
    metric: Metric,
    query: &[f32],
    centroids_bytes: &[u8],
    n_cent: usize,
    dim: usize,
    k: usize,
) -> Vec<(u32, f32)> {
    // Centroids are stored fp32 regardless of the column's rerank codec —
    // only the per-doc `full[]` region compresses. `distance_bytes` assumes
    // fp32, which is correct here.
    let stride = dim * 4;
    debug_assert!(centroids_bytes.len() >= n_cent * stride);
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k.saturating_add(1));
    if k == 0 {
        return top;
    }
    for c in 0..n_cent {
        let bytes = &centroids_bytes[c * stride..(c + 1) * stride];
        insert_ranked(&mut top, k, c as u32, distance_bytes(metric, query, bytes));
    }
    top
}

#[inline]
fn score_centroid_block8_transposed_wide(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    dim: usize,
    block: usize,
    lane_offset: usize,
) -> [f32; F32X8_LANES] {
    let mut acc = f32x8::ZERO;
    let block_base = block * dim * CENTROID_BATCH_LANES;
    for (d, &q_scalar) in query[..dim].iter().enumerate() {
        let q = f32x8::splat(q_scalar);
        let row = block_base + d * CENTROID_BATCH_LANES + lane_offset;
        let c = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(&transposed[row..row + F32X8_LANES])
                .expect("transposed centroid row has 8-lane half"),
        );
        match metric {
            Metric::L2Sq => {
                let diff = q - c;
                acc += diff * diff;
            }
            Metric::Cosine | Metric::NegDot => {
                acc += q * c;
            }
        }
    }
    let mut scores = acc.to_array();
    match metric {
        Metric::Cosine => {
            for score in &mut scores {
                *score = COSINE_DISTANCE_BASE - *score;
            }
        }
        Metric::NegDot => {
            for score in &mut scores {
                *score = -*score;
            }
        }
        Metric::L2Sq => {}
    }
    scores
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn score_centroid_block16_transposed_avx512(
    metric: Metric,
    query: &[f32],
    transposed: &[f32],
    dim: usize,
    block: usize,
) -> [f32; AVX512_F32_LANES] {
    // SAFETY: called only after `avx512_enabled()`. Each load reads one full
    // 16-f32 transposed row. The transposed cache is allocated as
    // `n_blocks * dim * 16`, and callers pass `block < n_blocks`.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let block_base = block * dim * CENTROID_BATCH_LANES;
        for (d, &q_scalar) in query[..dim].iter().enumerate() {
            let q = _mm512_set1_ps(q_scalar);
            let row = block_base + d * CENTROID_BATCH_LANES;
            let c = _mm512_loadu_ps(transposed.as_ptr().add(row));
            match metric {
                Metric::L2Sq => {
                    let diff = _mm512_sub_ps(q, c);
                    acc = _mm512_fmadd_ps(diff, diff, acc);
                }
                Metric::Cosine | Metric::NegDot => {
                    acc = _mm512_fmadd_ps(q, c, acc);
                }
            }
        }
        let mut scores = [0f32; AVX512_F32_LANES];
        _mm512_storeu_ps(scores.as_mut_ptr(), acc);
        match metric {
            Metric::Cosine => {
                for score in &mut scores {
                    *score = COSINE_DISTANCE_BASE - *score;
                }
            }
            Metric::NegDot => {
                for score in &mut scores {
                    *score = -*score;
                }
            }
            Metric::L2Sq => {}
        }
        scores
    }
}

/// Horizontal sum `Σ a[d]`. Dispatches AVX-512 → `wide::f32x8` like
/// [`dot`]. Precompute once per query for RaBitQ's `q_total` term.
#[inline]
pub(crate) fn sum_f32(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx512_enabled() {
        // SAFETY: gated by runtime CPUID detection in `avx512_enabled()`.
        return unsafe { sum_f32_avx512(a) };
    }
    sum_f32_wide(a)
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

/// Portable `wide::f32x8` horizontal sum. See [`dot_wide`].
#[inline]
fn sum_f32_wide(a: &[f32]) -> f32 {
    let chunks = a.chunks_exact(F32X8_LANES);
    let tail = chunks.remainder();
    let mut acc = f32x8::ZERO;
    for c in chunks {
        let va = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(c).expect("chunks_exact(8) yields slices of length 8"),
        );
        acc += va;
    }
    let mut sum: f32 = acc.reduce_add();
    for &x in tail {
        sum += x;
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

/// AVX-512 16-lane horizontal sum. See [`dot_avx512`].
///
/// # Safety
///
/// Same contract as [`dot_avx512`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sum_f32_avx512(a: &[f32]) -> f32 {
    let n = a.len();
    // SAFETY: see `dot_avx512` — same bounds reasoning.
    unsafe {
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + AVX512_F32_LANES <= n {
            let va = _mm512_loadu_ps(a.as_ptr().add(i));
            acc = _mm512_add_ps(va, acc);
            i += AVX512_F32_LANES;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < n {
            sum += a[i];
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
/// `Sq8Residual` doesn't have a "flat" entry point because decoding needs
/// scale/offset and, for L2Sq/Cosine, a per-doc norm. Its callers use
/// [`Sq8ResidualKernel`]. `RabitqOnly` carries no `full[]` bytes.
#[inline]
pub(crate) fn distance_bytes_codec(
    metric: Metric,
    codec: RerankCodec,
    query: &[f32],
    bytes: &[u8],
) -> f32 {
    match codec {
        RerankCodec::Fp32 => distance_bytes(metric, query, bytes),
        RerankCodec::Sq8Residual | RerankCodec::Sq8FixedResidual => {
            unreachable!(
                "distance_bytes_codec called with residual-family codec — rerank goes \
                 through dedicated kernels (need per-column scale/offset + per-doc \
                 norm context)"
            )
        }
        RerankCodec::Sq16 | RerankCodec::Sq16Adaptive => {
            unreachable!(
                "distance_bytes_codec called with a single-u16-plane codec — Sq16 / \
                 Sq16Adaptive rerank goes through Sq16Kernel (u16 → f32 dequant front \
                 on the fp32 distance path; adaptive folds the per-cluster ruler)"
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
#[cfg(test)]
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

#[cfg(test)]
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

/// `Sq8Residual` rerank context. Captures the per-cluster quantizer
/// (`scale[dim]`, `offset[dim]`) plus query-side precomputes for both stored
/// bytes, so the per-candidate inner loop is two u8/i8 → f32 widens + SIMD dot.
///
/// One kernel is built per query + cluster and reused across every RaBitQ
/// shortlist survivor assigned to that cluster.
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
    /// Build the per-query residual kernel. `scale` + `offset` are the
    /// per-cluster quantizer arrays; `residual_divisor` is
    /// [`SQ8_RESIDUAL_DIVISOR`].
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

    /// Score one candidate with both stored bytes and its decoded norm.
    /// `norm` is absent only for NegDot, where the norm term cancels.
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

/// `Sq16` rerank context — the flat single-plane analogue of
/// [`Sq8ResidualKernel`]. One `u16` code per dimension on the fixed
/// cosine grid ([`SQ16_FIXED_OFFSET`] / [`SQ16_FIXED_SCALE`]), scored
/// in a single pass. This is the [`distance`] fp32 path with a
/// `u16 → f32` dequant folded into the query-side precompute.
///
/// Reconstruction is `x[d] = code[d] * scale + offset`, so folding the
/// grid into the query once gives
/// `dot(query, x) = Σ_d q_prime[d] * code[d] + q_dot_offset`, where
/// `q_prime[d] = query[d] * SQ16_FIXED_SCALE` and
/// `q_dot_offset = SQ16_FIXED_OFFSET * Σ_d query[d]`.
///
/// Because the grid is global constants (matching the codec's empty
/// `codec_meta`), the kernel holds no per-cluster or per-doc state —
/// one kernel per query, reused across every candidate.
pub(crate) struct Sq16Kernel {
    metric: Metric,
    dim: usize,
    /// `q_prime[d] = query[d] * SQ16_FIXED_SCALE`.
    q_prime: Vec<f32>,
    /// `SQ16_FIXED_OFFSET * Σ_d query[d]`. Folded in once per candidate.
    q_dot_offset: f32,
    /// `Σ_d query[d]²`. L2Sq only (Sq16 is cosine-only in practice).
    q_norm_sq: f32,
}

impl Sq16Kernel {
    /// Build the per-query kernel. No quantizer arrays or per-doc norms
    /// are needed — the fixed grid is baked into the constants.
    pub fn new(metric: Metric, query: &[f32]) -> Self {
        let dim = query.len();
        let mut q_prime = vec![0.0f32; dim];
        let scale_v = f32x8::splat(SQ16_FIXED_SCALE);
        let mut q_sum_acc = f32x8::ZERO;
        let mut i = 0;
        while i + F32X8_LANES <= dim {
            let qc = f32x8::from(
                <[f32; F32X8_LANES]>::try_from(&query[i..i + F32X8_LANES]).expect("len-8 slice"),
            );
            q_prime[i..i + F32X8_LANES].copy_from_slice(&(qc * scale_v).to_array());
            q_sum_acc += qc;
            i += F32X8_LANES;
        }
        let mut q_sum = q_sum_acc.reduce_add();
        while i < dim {
            q_prime[i] = query[i] * SQ16_FIXED_SCALE;
            q_sum += query[i];
            i += 1;
        }
        let q_dot_offset = SQ16_FIXED_OFFSET * q_sum;
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
        }
    }

    /// Adaptive-ruler counterpart of [`Self::new`]: the single-`u16`-plane
    /// kernel over a per-cluster fitted grid (`Sq16Adaptive`) instead of the
    /// fixed `[-1, 1]` constants. The scoring body ([`Self::distance_with_norm`])
    /// is identical — only the query fold differs, so this is the sole ruler
    /// override. `q_prime[d] = query[d]·scale[d]` and `q_dot_offset =
    /// Σ_d offset[d]·query[d]` (vs the fixed grid's `OFFSET·Σq`). `scale`/`offset`
    /// are the probed cluster's stored quantizer (`scale.len() == query.len()`).
    pub fn new_adaptive(metric: Metric, query: &[f32], scale: &[f32], offset: &[f32]) -> Self {
        let dim = query.len();
        debug_assert_eq!(scale.len(), dim);
        debug_assert_eq!(offset.len(), dim);
        let mut q_prime = vec![0.0f32; dim];
        for d in 0..dim {
            q_prime[d] = query[d] * scale[d];
        }
        let q_dot_offset = dot(query, offset);
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
        }
    }

    /// Distance for one candidate whose `full[]` row is `dim`
    /// little-endian `u16` codes (`code_bytes.len() == dim * 2`), with
    /// its stored per-doc dequantized norm `‖d̂‖²` in `norm` (absent
    /// only for NegDot, where the norm term cancels). Mirrors
    /// [`Sq8ResidualKernel::distance_with_norm`] so the cosine ranking
    /// is the norm-corrected `base − dot/‖d̂‖`, apples-to-apples with the
    /// Sq8 family. Smaller = closer.
    #[inline]
    pub fn distance_with_norm(&self, code_bytes: &[u8], norm: Option<f32>) -> f32 {
        debug_assert_eq!(code_bytes.len(), self.dim * 2);
        // dot(query, x_decoded) = Σ q_prime[d]·code[d] + q_dot_offset.
        let dot = sq16_cross(&self.q_prime, code_bytes) + self.q_dot_offset;
        match self.metric {
            Metric::Cosine => {
                let x_norm = norm
                    .expect("Sq16Kernel + Cosine requires per_doc_norms")
                    .sqrt();
                if x_norm > 0.0 {
                    COSINE_DISTANCE_BASE - dot / x_norm
                } else {
                    COSINE_DISTANCE_BASE - dot
                }
            }
            Metric::NegDot => -dot,
            Metric::L2Sq => {
                let x_norm_sq = norm.expect("Sq16Kernel + L2Sq requires per_doc_norms");
                self.q_norm_sq - L2_CROSS_TERM_COEFF * dot + x_norm_sq
            }
        }
    }
}

/// `Σ_d q_prime[d] · code_u16[d]` over `dim` little-endian `u16` codes —
/// the [`Sq16Kernel`] cross term and the whole of its per-candidate
/// arithmetic. Tier-dispatched: AVX-512F (16 codes/iteration), AVX2+FMA
/// (8), and a safe `wide` fallback. The rerank phase at 1M measured this
/// conversion as 92% of warm query wall when it ran through per-byte
/// slice indexing; the intrinsic tiers convert with `cvtepu16` directly
/// from the unaligned code bytes. Tiers may differ in the final f32 by
/// add-order/FMA rounding only (same contract as
/// [`BitQuantizer::estimate_dot_rotated_with_total`]).
#[inline]
fn sq16_cross(q_prime: &[f32], code_bytes: &[u8]) -> f32 {
    // Release-enforced: the intrinsic tiers below read `2 * dim` code
    // bytes through unaligned pointer loads, so a short slice would be
    // an out-of-bounds read, not a panic. One predictable branch per
    // candidate is free next to the ~dim FMAs that follow.
    assert_eq!(code_bytes.len(), q_prime.len() * 2);
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` (avx512f+bw+dq+vl);
            // the kernel uses AVX-512F conversions/FMA plus AVX2 loads,
            // all implied by that gate. Bounds: the loop reads
            // `16 * 2` code bytes and 16 `q_prime` floats per iteration
            // strictly below `dim - dim % 16`, and the scalar tail
            // covers the remainder — no out-of-bounds access.
            return unsafe { sq16_cross_avx512(q_prime, code_bytes) };
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2_enabled()`, which detects both
            // `avx2` and `fma` — the two features the function enables.
            // Bounds as above with an 8-code stride.
            return unsafe { sq16_cross_avx2(q_prime, code_bytes) };
        }
    }
    sq16_cross_wide(q_prime, code_bytes)
}

/// SQ8 walk dot: `Σ_d code_u8[d] · q_i8[d]` (unsigned code byte × signed int8
/// query), int32. A *ranking proxy* for the graph walk only — the per-query
/// offset baseline (`Σ 128·q_i8`) is a constant that doesn't change intra-query
/// order, so it's dropped; exact scores come from the Sq16 refine afterward.
/// `code_u8` is the contiguous high byte of each Sq16 code. int8 products over
/// ≤768 dims stay far inside int32. Runtime-dispatched to AVX-512-VNNI
/// `vpdpbusd`, else a scalar fallback. `code_u8.len() == q_i8.len() == dim`.
pub(crate) fn sq8_walk_dot(code_u8: &[u8], q_i8: &[i8]) -> i32 {
    debug_assert_eq!(code_u8.len(), q_i8.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
        {
            // SAFETY: gated on avx512f+bw+vnni (the features the kernel
            // enables). Each iteration reads 64 `code_u8` and 64 `q_i8`
            // strictly below `dim - dim % 64`; the scalar tail covers the
            // remainder — no out-of-bounds access.
            return unsafe { sq8_dot_vnni(code_u8, q_i8) };
        }
    }
    sq8_walk_dot_scalar(code_u8, q_i8)
}

fn sq8_walk_dot_scalar(code_u8: &[u8], q_i8: &[i8]) -> i32 {
    code_u8
        .iter()
        .zip(q_i8)
        .map(|(&c, &q)| c as i32 * q as i32)
        .sum()
}

/// AVX-512-VNNI `vpdpbusd` tier of [`sq8_walk_dot`]: 64 unsigned code bytes ×
/// 64 signed int8 query values per instruction, int32 accumulate.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn sq8_dot_vnni(code_u8: &[u8], q_i8: &[i8]) -> i32 {
    use core::arch::x86_64::{
        _mm512_dpbusd_epi32, _mm512_loadu_epi8, _mm512_reduce_add_epi32, _mm512_setzero_si512,
    };
    // SAFETY: called only after avx512f+bw+vnni detection. Each iteration loads
    // 64 bytes from `code_u8` and `q_i8` strictly below `dim - dim % 64`; the
    // scalar tail covers the remainder — no read past either slice.
    unsafe {
        let dim = q_i8.len();
        let mut acc = _mm512_setzero_si512();
        let mut d = 0usize;
        while d + 64 <= dim {
            let a = _mm512_loadu_epi8(code_u8.as_ptr().add(d) as *const i8);
            let b = _mm512_loadu_epi8(q_i8.as_ptr().add(d));
            acc = _mm512_dpbusd_epi32(acc, a, b);
            d += 64;
        }
        let mut s = _mm512_reduce_add_epi32(acc);
        while d < dim {
            s += code_u8[d] as i32 * q_i8[d] as i32;
            d += 1;
        }
        s
    }
}

/// Quantize a query to signed int8 for [`sq8_walk_dot`]: scale by the query's
/// own max magnitude so the largest component maps to ±127. The scale is a
/// per-query constant, so it doesn't affect the walk's intra-query ranking.
pub(crate) fn quantize_query_i8(query: &[f32]) -> Vec<i8> {
    let max = query.iter().fold(0.0f32, |m, &q| m.max(q.abs())).max(1e-12);
    let qs = 127.0 / max;
    query
        .iter()
        .map(|&q| (q * qs).round().clamp(-127.0, 127.0) as i8)
        .collect()
}

/// Safe `wide` tier of [`sq16_cross`]: fixed-size 16-byte chunks so the
/// u16 conversions compile without per-byte bounds checks.
fn sq16_cross_wide(q_prime: &[f32], code_bytes: &[u8]) -> f32 {
    let mut acc = f32x8::ZERO;
    let mut q_chunks = q_prime.chunks_exact(F32X8_LANES);
    let mut c_chunks = code_bytes.chunks_exact(F32X8_LANES * 2);
    for (qp, cb) in (&mut q_chunks).zip(&mut c_chunks) {
        let qp: [f32; F32X8_LANES] = qp.try_into().expect("len-8 q_prime chunk");
        let cb: [u8; F32X8_LANES * 2] = cb.try_into().expect("len-16 code chunk");
        let mut lanes = [0f32; F32X8_LANES];
        for (j, lane) in lanes.iter_mut().enumerate() {
            *lane = f32::from(u16::from_le_bytes([cb[2 * j], cb[2 * j + 1]]));
        }
        acc = f32x8::from(qp).mul_add(f32x8::from(lanes), acc);
    }
    let mut cross = acc.reduce_add();
    for (qp, cb) in q_chunks
        .remainder()
        .iter()
        .zip(c_chunks.remainder().chunks_exact(2))
    {
        cross += qp * f32::from(u16::from_le_bytes([cb[0], cb[1]]));
    }
    cross
}

/// AVX2+FMA tier of [`sq16_cross`]: 8 u16 codes load as one 128-bit
/// vector, widen (`cvtepu16_epi32`), convert (`cvtepi32_ps`), and fold
/// into a single FMA accumulator; scalar tail for `dim % 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sq16_cross_avx2(q_prime: &[f32], code_bytes: &[u8]) -> f32 {
    let dim = q_prime.len();
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY (callee): unaligned loads only, within `i + 8 <= dim`
    // (16 code bytes, 8 floats per iteration).
    unsafe {
        while i + 8 <= dim {
            let raw = _mm_loadu_si128(code_bytes.as_ptr().add(2 * i) as *const __m128i);
            let vals = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(raw));
            let q = _mm256_loadu_ps(q_prime.as_ptr().add(i));
            acc = _mm256_fmadd_ps(q, vals, acc);
            i += 8;
        }
        let hi = _mm256_extractf128_ps(acc, 1);
        let lo = _mm256_castps256_ps128(acc);
        let sum4 = _mm_add_ps(lo, hi);
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        let mut cross = _mm_cvtss_f32(sum1);
        while i < dim {
            let b = 2 * i;
            cross += q_prime[i] * f32::from(u16::from_le_bytes([code_bytes[b], code_bytes[b + 1]]));
            i += 1;
        }
        cross
    }
}

/// AVX-512F tier of [`sq16_cross`]: 16 u16 codes per iteration
/// (256-bit load → 512-bit widen/convert → one FMA accumulator).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq16_cross_avx512(q_prime: &[f32], code_bytes: &[u8]) -> f32 {
    let dim = q_prime.len();
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    // SAFETY (callee): unaligned loads only, within `i + 16 <= dim`
    // (32 code bytes, 16 floats per iteration).
    unsafe {
        while i + 16 <= dim {
            let raw = _mm256_loadu_si256(code_bytes.as_ptr().add(2 * i) as *const __m256i);
            let vals = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(raw));
            let q = _mm512_loadu_ps(q_prime.as_ptr().add(i));
            acc = _mm512_fmadd_ps(q, vals, acc);
            i += 16;
        }
        let mut cross = _mm512_reduce_add_ps(acc);
        while i < dim {
            let b = 2 * i;
            cross += q_prime[i] * f32::from(u16::from_le_bytes([code_bytes[b], code_bytes[b + 1]]));
            i += 1;
        }
        cross
    }
}

/// Encode one fp32 vector into `dim` little-endian `u16` Sq16 codes —
/// the inverse of [`Sq16Kernel`]'s dequant. Per dimension:
/// `code = round((v - SQ16_FIXED_OFFSET) / SQ16_FIXED_SCALE)` clamped
/// to `0..=65535`. `out.len()` must be `src.len() * 2`.
#[inline]
pub(crate) fn encode_sq16_row(src: &[f32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), src.len() * 2);
    let inv_scale = 1.0 / SQ16_FIXED_SCALE;
    for (d, &v) in src.iter().enumerate() {
        let code = (((v - SQ16_FIXED_OFFSET) * inv_scale).round()).clamp(0.0, SQ16_CODE_MAX) as u16;
        let b = d * 2;
        out[b..b + 2].copy_from_slice(&code.to_le_bytes());
    }
}

/// Dequantize one row of `dim` little-endian `u16` Sq16 codes to fp32 —
/// the inverse of [`encode_sq16_row`]. Per dimension:
/// `x = code · SQ16_FIXED_SCALE + SQ16_FIXED_OFFSET`. `code.len()` must be
/// `out.len() * 2`.
#[inline]
pub(crate) fn dequantize_sq16_into(code: &[u8], out: &mut [f32]) {
    let dim = out.len();
    debug_assert_eq!(code.len(), dim * 2);
    for (d, slot) in out.iter_mut().enumerate() {
        let b = d * 2;
        let c = u16::from_le_bytes([code[b], code[b + 1]]) as f32;
        *slot = c * SQ16_FIXED_SCALE + SQ16_FIXED_OFFSET;
    }
}

/// Dequantized-vector squared norm `Σ_d (code[d]·scale + offset)²` for a
/// row of `dim` little-endian `u16` Sq16 codes. This is the per-doc
/// `‖d̂‖²` the encoder stores in `codec_meta` and the cosine
/// [`Sq16Kernel::distance_with_norm`] divides by (after `sqrt`), so it
/// must decode with the exact same grid the kernel uses.
#[inline]
pub(crate) fn sq16_decoded_norm_sq(code_bytes: &[u8], dim: usize) -> f32 {
    debug_assert_eq!(code_bytes.len(), dim * 2);
    let mut acc = f32x8::ZERO;
    let off_v = f32x8::splat(SQ16_FIXED_OFFSET);
    let scale_v = f32x8::splat(SQ16_FIXED_SCALE);
    let mut i = 0;
    while i + F32X8_LANES <= dim {
        let mut code = [0f32; F32X8_LANES];
        for (j, lane) in code.iter_mut().enumerate() {
            let b = 2 * (i + j);
            *lane = u16::from_le_bytes([code_bytes[b], code_bytes[b + 1]]) as f32;
        }
        let x = f32x8::from(code) * scale_v + off_v;
        acc += x * x;
        i += F32X8_LANES;
    }
    let mut s = acc.reduce_add();
    while i < dim {
        let b = 2 * i;
        let code = u16::from_le_bytes([code_bytes[b], code_bytes[b + 1]]) as f32;
        let x = code * SQ16_FIXED_SCALE + SQ16_FIXED_OFFSET;
        s += x * x;
        i += 1;
    }
    s
}

/// Adaptive-ruler counterparts of [`encode_sq16_row`] /
/// [`dequantize_sq16_into`] / [`sq16_decoded_norm_sq`]: identical single-`u16`
/// -plane layout, but the grid is the cluster's fitted `scale[d]`/`offset[d]`
/// (`x = code·scale[d] + offset[d]`) instead of the fixed `[-1, 1]` constants.
/// Used by `Sq16Adaptive`; `scale.len() == offset.len() == dim`.
///
/// On the build path the ruler is fit to the cluster's own `[min,max]`, so
/// every component lands in `0..=65535` and the clamp never trips. On merge the
/// destination reuses the first input's ruler, so a component of a later input
/// that falls outside that range is clamped to the grid edge here.
///
/// Returns the number of components that landed beyond the grid (past a
/// half-code slack) and were clamped. Build callers ignore it; the merge
/// transcode feeds it to the maintenance clamp tripwire so a destination ruler
/// that fails to cover its inputs shouts instead of silently losing recall.
#[inline]
pub(crate) fn encode_sq16_adaptive_row(
    src: &[f32],
    scale: &[f32],
    offset: &[f32],
    out: &mut [u8],
) -> u64 {
    debug_assert_eq!(out.len(), src.len() * 2);
    debug_assert_eq!(scale.len(), src.len());
    debug_assert_eq!(offset.len(), src.len());
    let mut clamped = 0u64;
    for (d, &v) in src.iter().enumerate() {
        // Guard a zero-span dimension: when the ruler's scale is 0, map every
        // value to code 0 so decode returns the constant offset exactly. The
        // build fit assigns scale = 1.0 (not 0) for a constant dim, so on the
        // real build/merge path this is a defensive floor rather than the
        // expected shape.
        let code = if scale[d] > 0.0 {
            let q = (v - offset[d]) / scale[d];
            if !(-SQ16_CLAMP_DETECT_SLACK_CODES..=SQ16_CODE_MAX + SQ16_CLAMP_DETECT_SLACK_CODES)
                .contains(&q)
            {
                clamped += 1;
            }
            q.round().clamp(0.0, SQ16_CODE_MAX) as u16
        } else {
            0
        };
        let b = d * 2;
        out[b..b + 2].copy_from_slice(&code.to_le_bytes());
    }
    clamped
}

/// Inverse of [`encode_sq16_adaptive_row`]. `code.len() == out.len() * 2`.
#[inline]
pub(crate) fn dequantize_sq16_adaptive_into(
    code: &[u8],
    scale: &[f32],
    offset: &[f32],
    out: &mut [f32],
) {
    let dim = out.len();
    debug_assert_eq!(code.len(), dim * 2);
    debug_assert_eq!(scale.len(), dim);
    debug_assert_eq!(offset.len(), dim);
    for (d, slot) in out.iter_mut().enumerate() {
        let b = d * 2;
        let c = u16::from_le_bytes([code[b], code[b + 1]]) as f32;
        *slot = c * scale[d] + offset[d];
    }
}

/// Adaptive-ruler [`sq16_decoded_norm_sq`]: `Σ_d (code[d]·scale[d] + offset[d])²`.
#[inline]
pub(crate) fn sq16_adaptive_norm_sq(
    code_bytes: &[u8],
    dim: usize,
    scale: &[f32],
    offset: &[f32],
) -> f32 {
    debug_assert_eq!(code_bytes.len(), dim * 2);
    debug_assert_eq!(scale.len(), dim);
    debug_assert_eq!(offset.len(), dim);
    let mut s = 0.0f32;
    for d in 0..dim {
        let b = 2 * d;
        let c = u16::from_le_bytes([code_bytes[b], code_bytes[b + 1]]) as f32;
        let x = c * scale[d] + offset[d];
        s += x * x;
    }
    s
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
#[cfg(test)]
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

/// Portable `wide::f32x8` (256-bit) Sq8 dot product. Same per-
/// element math as the AVX-512 path, processed 8 lanes at a time
/// with a per-lane scalar `u8 as f32` widen. Universal fallback
/// for aarch64, SSE-only x86_64 hosts, and
/// `INFINO_DISABLE_AVX2=1` / `INFINO_DISABLE_AVX512=1` A/B runs.
#[cfg(test)]
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
#[cfg(all(test, target_arch = "x86_64"))]
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
#[cfg(all(test, target_arch = "x86_64"))]
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

/// Dequantize one Sq8+ε vector: `out[d] = offset[d] + scale[d]·(code[d] +
/// residual[d]/residual_divisor)`. Dispatches AVX-512 → AVX2 →
/// `wide::f32x8` like [`dot`].
#[inline]
pub(crate) fn dequantize_sq8_residual_into(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    out: &mut [f32],
) {
    let dim = out.len();
    debug_assert_eq!(scale.len(), dim);
    debug_assert_eq!(offset.len(), dim);
    debug_assert_eq!(codes.len(), dim);
    debug_assert_eq!(residuals.len(), dim);
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` which requires `avx512f`.
            unsafe {
                dequantize_sq8_residual_avx512(
                    scale,
                    offset,
                    codes,
                    residuals,
                    residual_divisor,
                    out,
                    dim,
                );
            }
            return;
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2_enabled()` which requires `avx2`.
            unsafe {
                dequantize_sq8_residual_avx2(
                    scale,
                    offset,
                    codes,
                    residuals,
                    residual_divisor,
                    out,
                    dim,
                );
            }
            return;
        }
    }
    dequantize_sq8_residual_wide(scale, offset, codes, residuals, residual_divisor, out, dim);
}

#[inline]
fn sq8_residual_component_scalar(
    scale: f32,
    offset: f32,
    code: u8,
    residual_byte: u8,
    residual_divisor: f32,
) -> f32 {
    let inv_div = 1.0 / residual_divisor;
    offset + scale * (code as f32 + (i8::from_le_bytes([residual_byte]) as f32) * inv_div)
}

#[inline]
fn dequantize_sq8_residual_wide(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    out: &mut [f32],
    dim: usize,
) {
    let inv_div = 1.0 / residual_divisor;
    let inv_v = f32x8::splat(inv_div);
    let mut i = 0;
    while i + F32X8_LANES <= dim {
        let off = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(&offset[i..i + F32X8_LANES])
                .expect("offset[i..i+8] len 8"),
        );
        let sc = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(&scale[i..i + F32X8_LANES])
                .expect("scale[i..i+8] len 8"),
        );
        let mut code_bc = [0f32; F32X8_LANES];
        let mut res_bc = [0f32; F32X8_LANES];
        for j in 0..F32X8_LANES {
            code_bc[j] = codes[i + j] as f32;
            res_bc[j] = i8::from_le_bytes([residuals[i + j]]) as f32;
        }
        let codes_v = f32x8::from(code_bc);
        let res_v = f32x8::from(res_bc);
        let term = codes_v + res_v * inv_v;
        let decoded = off + sc * term;
        out[i..i + F32X8_LANES].copy_from_slice(&decoded.to_array());
        i += F32X8_LANES;
    }
    while i < dim {
        out[i] = sq8_residual_component_scalar(
            scale[i],
            offset[i],
            codes[i],
            residuals[i],
            residual_divisor,
        );
        i += 1;
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`. [`avx2_enabled`] guarantees
/// this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_sq8_residual_avx2(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    out: &mut [f32],
    dim: usize,
) {
    // SAFETY: each iteration reads/writes 8-element windows; `i + 8 <= dim`
    // keeps all pointers in bounds. Unaligned loads/stores throughout.
    unsafe {
        let inv_div = _mm256_set1_ps(1.0 / residual_divisor);
        let mut i = 0;
        while i + F32X8_LANES <= dim {
            let codes_u8 = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
            let codes_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes_u8));
            let res_u8 = _mm_loadl_epi64(residuals.as_ptr().add(i) as *const __m128i);
            let res_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(res_u8));
            let term = _mm256_fmadd_ps(res_f32, inv_div, codes_f32);
            let off = _mm256_loadu_ps(offset.as_ptr().add(i));
            let sc = _mm256_loadu_ps(scale.as_ptr().add(i));
            let decoded = _mm256_fmadd_ps(sc, term, off);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), decoded);
            i += F32X8_LANES;
        }
        while i < dim {
            out[i] = sq8_residual_component_scalar(
                scale[i],
                offset[i],
                codes[i],
                residuals[i],
                residual_divisor,
            );
            i += 1;
        }
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx512f`. [`avx512_enabled`]
/// guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dequantize_sq8_residual_avx512(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    out: &mut [f32],
    dim: usize,
) {
    // SAFETY: each iteration reads/writes 16-element windows; `i + 16 <= dim`
    // keeps all pointers in bounds. Unaligned loads/stores throughout.
    unsafe {
        let inv_div = _mm512_set1_ps(1.0 / residual_divisor);
        let mut i = 0;
        while i + AVX512_F32_LANES <= dim {
            let codes_u8 = _mm_loadu_si128(codes.as_ptr().add(i) as *const __m128i);
            let codes_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes_u8));
            let res_u8 = _mm_loadu_si128(residuals.as_ptr().add(i) as *const __m128i);
            let res_f32 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(res_u8));
            let term = _mm512_fmadd_ps(res_f32, inv_div, codes_f32);
            let off = _mm512_loadu_ps(offset.as_ptr().add(i));
            let sc = _mm512_loadu_ps(scale.as_ptr().add(i));
            let decoded = _mm512_fmadd_ps(sc, term, off);
            _mm512_storeu_ps(out.as_mut_ptr().add(i), decoded);
            i += AVX512_F32_LANES;
        }
        while i < dim {
            out[i] = sq8_residual_component_scalar(
                scale[i],
                offset[i],
                codes[i],
                residuals[i],
                residual_divisor,
            );
            i += 1;
        }
    }
}

/// `||x||²` for one Sq8+ε vector without materializing fp32 storage.
/// Dispatches AVX-512 → AVX2 → `wide::f32x8` like
/// [`dequantize_sq8_residual_into`].
#[inline]
pub(crate) fn sq8_residual_norm_sq(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
) -> f32 {
    let dim = scale.len();
    debug_assert_eq!(offset.len(), dim);
    debug_assert_eq!(codes.len(), dim);
    debug_assert_eq!(residuals.len(), dim);
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` which requires `avx512f`.
            return unsafe {
                sq8_residual_norm_sq_avx512(scale, offset, codes, residuals, residual_divisor, dim)
            };
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2_enabled()` which requires `avx2`.
            return unsafe {
                sq8_residual_norm_sq_avx2(scale, offset, codes, residuals, residual_divisor, dim)
            };
        }
    }
    sq8_residual_norm_sq_wide(scale, offset, codes, residuals, residual_divisor, dim)
}

#[inline]
fn sq8_residual_norm_sq_wide(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    dim: usize,
) -> f32 {
    let inv_div = 1.0 / residual_divisor;
    let inv_v = f32x8::splat(inv_div);
    let mut acc = f32x8::ZERO;
    let mut i = 0;
    while i + F32X8_LANES <= dim {
        let off = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(&offset[i..i + F32X8_LANES])
                .expect("offset[i..i+8] len 8"),
        );
        let sc = f32x8::from(
            <[f32; F32X8_LANES]>::try_from(&scale[i..i + F32X8_LANES])
                .expect("scale[i..i+8] len 8"),
        );
        let mut code_bc = [0f32; F32X8_LANES];
        let mut res_bc = [0f32; F32X8_LANES];
        for j in 0..F32X8_LANES {
            code_bc[j] = codes[i + j] as f32;
            res_bc[j] = i8::from_le_bytes([residuals[i + j]]) as f32;
        }
        let codes_v = f32x8::from(code_bc);
        let res_v = f32x8::from(res_bc);
        let term = codes_v + res_v * inv_v;
        let decoded = off + sc * term;
        acc += decoded * decoded;
        i += F32X8_LANES;
    }
    let mut sum = acc.reduce_add();
    while i < dim {
        let v = sq8_residual_component_scalar(
            scale[i],
            offset[i],
            codes[i],
            residuals[i],
            residual_divisor,
        );
        sum += v * v;
        i += 1;
    }
    sum
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`. [`avx2_enabled`] guarantees
/// this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sq8_residual_norm_sq_avx2(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    dim: usize,
) -> f32 {
    // SAFETY: each iteration reads 8-element windows; `i + 8 <= dim` keeps
    // all pointers in bounds.
    unsafe {
        let inv_div = _mm256_set1_ps(1.0 / residual_divisor);
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + F32X8_LANES <= dim {
            let codes_u8 = _mm_loadl_epi64(codes.as_ptr().add(i) as *const __m128i);
            let codes_f32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes_u8));
            let res_u8 = _mm_loadl_epi64(residuals.as_ptr().add(i) as *const __m128i);
            let res_f32 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(res_u8));
            let term = _mm256_fmadd_ps(res_f32, inv_div, codes_f32);
            let off = _mm256_loadu_ps(offset.as_ptr().add(i));
            let sc = _mm256_loadu_ps(scale.as_ptr().add(i));
            let decoded = _mm256_fmadd_ps(sc, term, off);
            acc = _mm256_fmadd_ps(decoded, decoded, acc);
            i += F32X8_LANES;
        }
        let mut sum = horizontal_sum_avx256(acc);
        while i < dim {
            let v = sq8_residual_component_scalar(
                scale[i],
                offset[i],
                codes[i],
                residuals[i],
                residual_divisor,
            );
            sum += v * v;
            i += 1;
        }
        sum
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx512f`. [`avx512_enabled`]
/// guarantees this at the dispatch site.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sq8_residual_norm_sq_avx512(
    scale: &[f32],
    offset: &[f32],
    codes: &[u8],
    residuals: &[u8],
    residual_divisor: f32,
    dim: usize,
) -> f32 {
    // SAFETY: each iteration reads 16-element windows; `i + 16 <= dim` keeps
    // all pointers in bounds.
    unsafe {
        let inv_div = _mm512_set1_ps(1.0 / residual_divisor);
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        while i + AVX512_F32_LANES <= dim {
            let codes_u8 = _mm_loadu_si128(codes.as_ptr().add(i) as *const __m128i);
            let codes_f32 = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes_u8));
            let res_u8 = _mm_loadu_si128(residuals.as_ptr().add(i) as *const __m128i);
            let res_f32 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(res_u8));
            let term = _mm512_fmadd_ps(res_f32, inv_div, codes_f32);
            let off = _mm512_loadu_ps(offset.as_ptr().add(i));
            let sc = _mm512_loadu_ps(scale.as_ptr().add(i));
            let decoded = _mm512_fmadd_ps(sc, term, off);
            acc = _mm512_fmadd_ps(decoded, decoded, acc);
            i += AVX512_F32_LANES;
        }
        let mut sum = _mm512_reduce_add_ps(acc);
        while i < dim {
            let v = sq8_residual_component_scalar(
                scale[i],
                offset[i],
                codes[i],
                residuals[i],
                residual_divisor,
            );
            sum += v * v;
            i += 1;
        }
        sum
    }
}

/// Decode little-endian f32 bytes into `out`. On little-endian hosts the
/// layout matches native `f32`, so the hot path is unaligned SIMD load/store.
#[inline]
pub(crate) fn decode_f32_le_into(bytes: &[u8], out: &mut [f32]) {
    debug_assert_eq!(bytes.len(), out.len() * F32_BYTES);
    if let Ok(decoded) = bytemuck::try_cast_slice::<u8, f32>(bytes) {
        out.copy_from_slice(decoded);
        return;
    }
    let n = out.len();
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            // SAFETY: gated on `avx512_enabled()` which requires `avx512f`.
            unsafe {
                decode_f32_le_avx512(bytes, out, n);
            }
            return;
        }
        if avx2_enabled() {
            // SAFETY: gated on `avx2_enabled()` which requires `avx2`.
            unsafe {
                decode_f32_le_avx2(bytes, out, n);
            }
            return;
        }
    }
    decode_f32_le_wide(bytes, out, n);
}

#[inline]
fn decode_f32_le_wide(bytes: &[u8], out: &mut [f32], n: usize) {
    let mut i = 0;
    while i + F32X8_LANES <= n {
        let mut lane = [0f32; F32X8_LANES];
        for (j, slot) in lane.iter_mut().enumerate() {
            let b = (i + j) * F32_BYTES;
            *slot = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
        }
        out[i..i + F32X8_LANES].copy_from_slice(&lane);
        i += F32X8_LANES;
    }
    while i < n {
        let b = i * F32_BYTES;
        out[i] = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
        i += 1;
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_f32_le_avx2(bytes: &[u8], out: &mut [f32], n: usize) {
    // SAFETY: `i + 8 <= n` and each f32 is 4 bytes ⇒ byte offset `i*4+32`
    // stays inside `bytes`.
    unsafe {
        let src = bytes.as_ptr();
        let dst = out.as_mut_ptr();
        let mut i = 0;
        while i + F32X8_LANES <= n {
            let v = _mm256_loadu_ps(src.add(i * F32_BYTES) as *const f32);
            _mm256_storeu_ps(dst.add(i), v);
            i += F32X8_LANES;
        }
        while i < n {
            let b = i * F32_BYTES;
            *dst.add(i) = f32::from_le_bytes([
                *src.add(b),
                *src.add(b + 1),
                *src.add(b + 2),
                *src.add(b + 3),
            ]);
            i += 1;
        }
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx512f`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn decode_f32_le_avx512(bytes: &[u8], out: &mut [f32], n: usize) {
    // SAFETY: `i + 16 <= n` keeps the 64-byte load inside `bytes`.
    unsafe {
        let src = bytes.as_ptr();
        let dst = out.as_mut_ptr();
        let mut i = 0;
        while i + AVX512_F32_LANES <= n {
            let v = _mm512_loadu_ps(src.add(i * F32_BYTES) as *const f32);
            _mm512_storeu_ps(dst.add(i), v);
            i += AVX512_F32_LANES;
        }
        while i < n {
            let b = i * F32_BYTES;
            *dst.add(i) = f32::from_le_bytes([
                *src.add(b),
                *src.add(b + 1),
                *src.add(b + 2),
                *src.add(b + 3),
            ]);
            i += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn horizontal_sum_avx256(v: __m256) -> f32 {
    // SAFETY: `_mm256_*` shuffle/add intrinsics only touch `v`.
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }
}

/// Decode little-endian f32 bytes into a new vector.
#[inline]
pub(crate) fn decode_f32_le_vec(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % F32_BYTES, 0);
    let mut out = vec![0f32; bytes.len() / F32_BYTES];
    decode_f32_le_into(bytes, &mut out);
    out
}

/// Add `row` into `acc` element-wise (`acc[d] += row[d] as f64`). Keeps
/// f64 precision for k-means / centroid-merge accumulators while using
/// AVX2/AVX-512 for the fp32 load + widen.
#[inline]
pub(crate) fn add_f32_to_f64_acc(acc: &mut [f64], row: &[f32]) {
    debug_assert_eq!(acc.len(), row.len());
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        // SAFETY: gated on `avx2_enabled()`.
        unsafe {
            add_f32_to_f64_acc_avx2(acc, row);
        }
        return;
    }
    add_f32_to_f64_acc_scalar(acc, row);
}

/// Like [`add_f32_to_f64_acc`] but scales each lane by `weight` first.
#[inline]
pub(crate) fn add_weighted_f32_to_f64_acc(acc: &mut [f64], row: &[f32], weight: f64) {
    debug_assert_eq!(acc.len(), row.len());
    #[cfg(target_arch = "x86_64")]
    if avx2_enabled() {
        // SAFETY: gated on `avx2_enabled()`.
        unsafe {
            add_weighted_f32_to_f64_acc_avx2(acc, row, weight);
        }
        return;
    }
    for (a, &x) in acc.iter_mut().zip(row.iter()) {
        *a += x as f64 * weight;
    }
}

/// Write `out[d] = (acc[d] * inv) as f32` after a f64 reduction pass.
#[inline]
pub(crate) fn f64_acc_mean_into_f32(acc: &[f64], inv: f64, out: &mut [f32]) {
    debug_assert_eq!(acc.len(), out.len());
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = (a * inv) as f32;
    }
}

/// Mean of `n` cluster-major fp32 vectors (`vectors.len() == n * dim`).
#[inline]
pub(crate) fn mean_f32_cluster_major(vectors: &[f32], dim: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(vectors.len(), n * dim);
    let mut acc = vec![0f64; dim];
    for i in 0..n {
        add_f32_to_f64_acc(&mut acc, &vectors[i * dim..(i + 1) * dim]);
    }
    let mut out = vec![0f32; dim];
    if n > 0 {
        f64_acc_mean_into_f32(&acc, 1.0 / n as f64, &mut out);
    }
    out
}

#[inline]
fn add_f32_to_f64_acc_scalar(acc: &mut [f64], row: &[f32]) {
    for (a, &x) in acc.iter_mut().zip(row.iter()) {
        *a += x as f64;
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_f32_to_f64_acc_avx2(acc: &mut [f64], row: &[f32]) {
    // SAFETY: each iteration touches 8 f32 / 4 f64 lanes; `i + 8 <= len`
    // keeps all pointers in bounds.
    unsafe {
        let mut i = 0;
        let n = acc.len();
        while i + F32X8_LANES <= n {
            let vf = _mm256_loadu_ps(row.as_ptr().add(i));
            let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vf));
            let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vf, 1));
            let alo = _mm256_loadu_pd(acc.as_mut_ptr().add(i));
            let ahi = _mm256_loadu_pd(acc.as_mut_ptr().add(i + 4));
            _mm256_storeu_pd(acc.as_mut_ptr().add(i), _mm256_add_pd(alo, lo));
            _mm256_storeu_pd(acc.as_mut_ptr().add(i + 4), _mm256_add_pd(ahi, hi));
            i += F32X8_LANES;
        }
        while i < n {
            acc[i] += row[i] as f64;
            i += 1;
        }
    }
}

/// # Safety
///
/// Callers must ensure the target supports `avx2`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_weighted_f32_to_f64_acc_avx2(acc: &mut [f64], row: &[f32], weight: f64) {
    // SAFETY: same bounds contract as [`add_f32_to_f64_acc_avx2`].
    unsafe {
        let w = _mm256_set1_pd(weight);
        let mut i = 0;
        let n = acc.len();
        while i + F32X8_LANES <= n {
            let vf = _mm256_loadu_ps(row.as_ptr().add(i));
            let lo = _mm256_cvtps_pd(_mm256_castps256_ps128(vf));
            let hi = _mm256_cvtps_pd(_mm256_extractf128_ps(vf, 1));
            let wlo = _mm256_mul_pd(lo, w);
            let whi = _mm256_mul_pd(hi, w);
            let alo = _mm256_loadu_pd(acc.as_mut_ptr().add(i));
            let ahi = _mm256_loadu_pd(acc.as_mut_ptr().add(i + 4));
            _mm256_storeu_pd(acc.as_mut_ptr().add(i), _mm256_add_pd(alo, wlo));
            _mm256_storeu_pd(acc.as_mut_ptr().add(i + 4), _mm256_add_pd(ahi, whi));
            i += F32X8_LANES;
        }
        while i < n {
            acc[i] += row[i] as f64 * weight;
            i += 1;
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
    // Normalize only when the magnitude is a normal float. Zero-norm
    // vectors keep the existing leave-alone policy, and a subnormal
    // magnitude joins it: `1.0 / mag` there rounds toward infinity and
    // would poison every component with inf/NaN instead of normalizing.
    if mag.is_normal() {
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

    /// The SQ8 walk kernel's SIMD tier must match its scalar reference exactly
    /// (integer dot — bit-exact, no float slack). Covers a 64-aligned width and
    /// widths with a scalar tail (`dim % 64 != 0`).
    #[test]
    fn sq8_walk_dot_simd_matches_scalar() {
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for dim in [64usize, 100, 768] {
            let code: Vec<u8> = (0..dim).map(|_| (next() & 0xff) as u8).collect();
            let q: Vec<i8> = (0..dim)
                .map(|_| ((next() % 255) as i32 - 127) as i8)
                .collect();
            let scalar = sq8_walk_dot_scalar(&code, &q);
            assert_eq!(
                sq8_walk_dot(&code, &q),
                scalar,
                "runtime-dispatched sq8_walk_dot != scalar at dim={dim}"
            );
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
            {
                // SAFETY: gated on the exact features `sq8_dot_vnni` enables.
                let vnni = unsafe { sq8_dot_vnni(&code, &q) };
                assert_eq!(
                    vnni, scalar,
                    "avx512-vnni sq8 kernel != scalar at dim={dim}"
                );
            }
        }
    }

    fn scalar_nearest_two_centroids(
        metric: Metric,
        query: &[f32],
        centroids: &[f32],
        n_cent: usize,
        dim: usize,
        counts: Option<&[u32]>,
    ) -> Option<((u32, f32), Option<(u32, f32)>)> {
        let mut best: Option<(u32, f32)> = None;
        let mut second: Option<(u32, f32)> = None;
        for c in 0..n_cent {
            if counts.is_some_and(|counts| counts[c] == 0) {
                continue;
            }
            let score = distance(metric, query, &centroids[c * dim..(c + 1) * dim]);
            match best {
                None => best = Some((c as u32, score)),
                Some((_, best_score)) if score < best_score => {
                    second = best;
                    best = Some((c as u32, score));
                }
                _ => {
                    if second.is_none_or(|(_, second_score)| score < second_score) {
                        second = Some((c as u32, score));
                    }
                }
            }
        }
        best.map(|best| (best, second))
    }

    fn assert_nearest_two_matches(
        got: Option<((u32, f32), Option<(u32, f32)>)>,
        expected: Option<((u32, f32), Option<(u32, f32)>)>,
    ) {
        let (got_best, got_second) = got.expect("nearest result");
        let (expected_best, expected_second) = expected.expect("scalar nearest result");
        assert_eq!(got_best.0, expected_best.0);
        assert!(
            approx(got_best.1, expected_best.1, 1e-3),
            "best score got {} expected {}",
            got_best.1,
            expected_best.1
        );
        let got_second = got_second.expect("second result");
        let expected_second = expected_second.expect("scalar second result");
        assert_eq!(got_second.0, expected_second.0);
        assert!(
            approx(got_second.1, expected_second.1, 1e-3),
            "second score got {} expected {}",
            got_second.1,
            expected_second.1
        );
    }

    /// The k-means assign kernel must agree with the naive per-centroid
    /// scan: same argmin on random data, and lowest-index winner on exact
    /// ties (duplicated centroids). Covers lane-tail shapes (`n_cent` not a
    /// multiple of the SIMD block width).
    #[test]
    fn nearest_centroid_transposed_matches_naive_scan() {
        for n_cent in [128usize, 130, 144, 160] {
            let dim = 33;
            let mut centroids = Vec::with_capacity(n_cent * dim);
            for c in 0..n_cent {
                for d in 0..dim {
                    centroids.push(((c * 31 + d * 17) % 29) as f32 * 0.04 - 0.5);
                }
            }
            // Exact tie: centroid n-1 duplicates centroid 3; the naive
            // scan keeps the lower index and the blocked kernel must too.
            let dup = centroids[3 * dim..4 * dim].to_vec();
            let last = (n_cent - 1) * dim;
            centroids[last..last + dim].copy_from_slice(&dup);
            let transposed = transpose_centroids_cluster_major(&centroids, n_cent, dim);
            for probe in 0..64 {
                let query: Vec<f32> = (0..dim)
                    .map(|d| ((probe * 13 + d * 7) % 23) as f32 * 0.05 - 0.4)
                    .collect();
                let mut naive = (0u32, f32::INFINITY);
                for c in 0..n_cent {
                    let dist = l2_sq(&query, &centroids[c * dim..(c + 1) * dim]);
                    if dist < naive.1 {
                        naive = (c as u32, dist);
                    }
                }
                let blocked =
                    nearest_centroid_transposed(Metric::L2Sq, &query, &transposed, n_cent, dim);
                assert_eq!(
                    blocked.0, naive.0,
                    "n_cent {n_cent} probe {probe}: blocked argmin diverged from naive"
                );
            }
            // Tie probe: query exactly at the duplicated centroid.
            let tie_query = dup.clone();
            let blocked =
                nearest_centroid_transposed(Metric::L2Sq, &tie_query, &transposed, n_cent, dim);
            assert_eq!(blocked.0, 3, "tie must resolve to the lowest index");
        }
    }

    #[test]
    fn nearest_two_centroids_transposed_matches_scalar_reference() {
        let dim = 17;
        let n_cent = 19;
        let query: Vec<f32> = (0..dim)
            .map(|d| ((d * 37 % 23) as f32 - 11.0) * 0.031)
            .collect();
        let mut centroids = Vec::with_capacity(n_cent * dim);
        for c in 0..n_cent {
            for d in 0..dim {
                centroids.push(((c * 13 + d * 7) % 31) as f32 * 0.02 - 0.3 + c as f32 * 0.001);
            }
        }
        let transposed = transpose_centroids_cluster_major(&centroids, n_cent, dim);
        for metric in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
            let got =
                nearest_two_centroids_transposed(metric, &query, &transposed, n_cent, dim, None);
            let expected =
                scalar_nearest_two_centroids(metric, &query, &centroids, n_cent, dim, None);
            assert_nearest_two_matches(got, expected);
        }
    }

    #[test]
    fn nearest_two_centroids_transposed_honors_zero_counts() {
        let dim = 17;
        let n_cent = 19;
        let query: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01).collect();
        let mut centroids = Vec::with_capacity(n_cent * dim);
        for c in 0..n_cent {
            for d in 0..dim {
                centroids.push(c as f32 + d as f32 * 0.001);
            }
        }
        let mut counts = vec![1u32; n_cent];
        counts[0] = 0;
        let transposed = transpose_centroids_cluster_major(&centroids, n_cent, dim);
        let got = nearest_two_centroids_transposed(
            Metric::L2Sq,
            &query,
            &transposed,
            n_cent,
            dim,
            Some(&counts),
        );
        let expected = scalar_nearest_two_centroids(
            Metric::L2Sq,
            &query,
            &centroids,
            n_cent,
            dim,
            Some(&counts),
        );
        assert_nearest_two_matches(got, expected);
        assert_ne!(got.expect("nearest result").0.0, 0);
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
    fn sum_f32_matches_scalar_reference() {
        for len in [1, 7, 8, 15, 16, 17, 384] {
            let v: Vec<f32> = (0..len).map(|i| (i as f32) * 0.25 - 1.0).collect();
            let expected: f32 = v.iter().sum();
            assert!(
                approx(sum_f32(&v), expected, 1e-4),
                "len={len}: got {} expected {expected}",
                sum_f32(&v)
            );
        }
    }

    #[test]
    fn sq8_residual_norm_sq_matches_dequant_dot_self() {
        let dim = 17;
        let scale: Vec<f32> = (0..dim).map(|i| 0.01 * (i as f32 + 1.0)).collect();
        let offset: Vec<f32> = (0..dim).map(|i| -0.5 + 0.03 * i as f32).collect();
        let codes: Vec<u8> = (0..dim).map(|i| (i * 17 % 256) as u8).collect();
        let residuals: Vec<u8> = (0..dim)
            .map(|i| ((i as i8).wrapping_mul(3)).to_le_bytes()[0])
            .collect();
        let mut decoded = vec![0f32; dim];
        dequantize_sq8_residual_into(
            &scale,
            &offset,
            &codes,
            &residuals,
            SQ8_RESIDUAL_DIVISOR,
            &mut decoded,
        );
        let expected = dot(&decoded, &decoded);
        let got = sq8_residual_norm_sq(&scale, &offset, &codes, &residuals, SQ8_RESIDUAL_DIVISOR);
        // Relative tolerance: the norm's magnitude here is ~1e4, where an
        // absolute 1e-4 sits below f32 rounding noise — the SIMD kernel and
        // the scalar dequant+dot reference sum in different orders (and CI's
        // coverage build changes codegen), so they legitimately diverge past
        // it. 1e-5 relative matches the cross-arch bound the cluster-scorer
        // self-check uses.
        assert!(
            (got - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
            "norm {got} vs dequant-dot-self {expected}"
        );
    }

    #[test]
    fn decode_f32_le_into_round_trip() {
        let values: Vec<f32> = (0..19).map(|i| i as f32 * 0.125 - 2.0).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut out = vec![0f32; values.len()];
        decode_f32_le_into(&bytes, &mut out);
        assert_eq!(out, values);
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

    #[test]
    fn normalize_degenerate_magnitude_never_produces_inf() {
        // All-subnormal components: their squares flush to zero in f32,
        // so the magnitude is zero or subnormal — the vector must be
        // left alone rather than scaled by an infinite 1/mag.
        let mut v = vec![f32::from_bits(1); 16]; // smallest positive subnormal
        let before = v.clone();
        normalize(&mut v);
        assert_eq!(v, before);
        for &x in &v {
            assert!(x.is_finite());
        }
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
            let got =
                kernel.distance_with_norm(&codes, &residuals, norms_arg.map(|norms| norms[0]));
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

    /// Round-trip the flat Sq16 codec: encode a unit-normalized fp32
    /// vector onto the fixed cosine grid, score it against a query with
    /// [`Sq16Kernel`], and confirm the cosine distance tracks the exact
    /// fp32 reference to ~16-bit precision. The fp32 reference is the
    /// same raw-dot cosine (`1 − q·x`) the [`distance`] dispatch uses,
    /// so this is an apples-to-apples quantization-error bound. Sweeps a
    /// SIMD-aligned dim, a tail dim, and a production-shaped dim.
    /// Every [`sq16_cross`] tier this host supports agrees with the safe
    /// `wide` tier and with an f64 scalar reference, to f32
    /// add-order/FMA tolerance — the cross-tier contract the 1-bit
    /// estimator documents. Sweeps SIMD-aligned dims, tail dims, and the
    /// production 768.
    #[test]
    fn sq16_cross_tiers_agree() {
        for &dim in &[8usize, 13, 100, 384, 768, 771] {
            let q_prime: Vec<f32> = (0..dim)
                .map(|i| ((i as f32) * 0.013 - 0.2).sin() * 0.001)
                .collect();
            let codes: Vec<u8> = (0..dim)
                .flat_map(|i| (((i * 2_654_435_761) % 65_536) as u16).to_le_bytes())
                .collect();
            let exact: f64 = (0..dim)
                .map(|i| {
                    f64::from(q_prime[i])
                        * f64::from(u16::from_le_bytes([codes[2 * i], codes[2 * i + 1]]))
                })
                .sum();
            let abs_sum: f64 = (0..dim)
                .map(|i| {
                    (f64::from(q_prime[i])
                        * f64::from(u16::from_le_bytes([codes[2 * i], codes[2 * i + 1]])))
                    .abs()
                })
                .sum();
            let tol = (abs_sum * 1e-5 + 1e-3) as f32;
            let reference = sq16_cross_wide(&q_prime, &codes);
            assert!(
                ((f64::from(reference) - exact).abs() as f32) <= tol,
                "dim {dim}: wide {reference} vs exact {exact}"
            );
            #[cfg(target_arch = "x86_64")]
            {
                if avx2_enabled() {
                    // SAFETY: gated on runtime AVX2 detection.
                    let got = unsafe { sq16_cross_avx2(&q_prime, &codes) };
                    assert!(
                        (got - reference).abs() <= tol,
                        "dim {dim}: avx2 {got} vs wide {reference}"
                    );
                }
                if avx512_enabled() {
                    // SAFETY: gated on runtime AVX-512 detection.
                    let got = unsafe { sq16_cross_avx512(&q_prime, &codes) };
                    assert!(
                        (got - reference).abs() <= tol,
                        "dim {dim}: avx512 {got} vs wide {reference}"
                    );
                }
            }
        }
    }

    #[test]
    fn sq16_round_trip_within_16bit_tolerance_of_fp32() {
        fn normalize(v: &mut [f32]) {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in v.iter_mut() {
                    *x /= n;
                }
            }
        }

        for &dim in &[8usize, 13, 100, 384] {
            // Deterministic pseudo-vectors in roughly [-1, 1], then
            // unit-normalized so cosine semantics (raw dot) hold and
            // every component lands inside the grid without clamping.
            let mut query: Vec<f32> = (0..dim).map(|i| ((i as f32) * 0.017 - 0.4).sin()).collect();
            let mut vec: Vec<f32> = (0..dim)
                .map(|i| ((i as f32) * 0.023 + 0.11).cos())
                .collect();
            normalize(&mut query);
            normalize(&mut vec);

            // Encode onto the Sq16 grid exactly as the builder write
            // path does (codes + per-doc dequantized norm), then score
            // through the reader-side kernel with that stored norm.
            let mut bytes = vec![0u8; dim * 2];
            encode_sq16_row(&vec, &mut bytes);
            let norm_sq = sq16_decoded_norm_sq(&bytes, dim);

            let kernel = Sq16Kernel::new(Metric::Cosine, &query);
            let got = kernel.distance_with_norm(&bytes, Some(norm_sq));
            // fp32 reference: query and vec are both unit vectors, so the
            // exact cosine distance is `1 - dot(query, vec)`.
            let want = distance(Metric::Cosine, &query, &vec);

            // Norm-corrected Sq16 tracks the fp32 cosine tighter than the
            // uncorrected `1 - dot` did: the per-doc `‖d̂‖` division
            // removes the dequantized-scale drift, leaving only direction
            // quantization.
            let rel = (got - want).abs() / want.abs();
            assert!(
                rel < 1e-4,
                "dim {dim}: Sq16 cosine {got} vs fp32 ref {want} (rel err {rel:e})"
            );

            // Encode/decode/kernel self-consistency: decoding the stored
            // codes and scoring with the SAME norm correction
            // (`1 - dot/‖d̂‖`) must reproduce the kernel's value.
            let decoded: Vec<f32> = (0..dim)
                .map(|d| {
                    let b = d * 2;
                    let code = u16::from_le_bytes([bytes[b], bytes[b + 1]]);
                    code as f32 * SQ16_FIXED_SCALE + SQ16_FIXED_OFFSET
                })
                .collect();
            let decoded_norm = decoded.iter().map(|x| x * x).sum::<f32>().sqrt();
            let decoded_ref = COSINE_DISTANCE_BASE - dot(&query, &decoded) / decoded_norm;
            assert!(
                (got - decoded_ref).abs() <= 1e-4,
                "dim {dim}: kernel {got} disagrees with decoded ref {decoded_ref}"
            );
        }
    }

    /// `Sq16Adaptive`: a per-cluster fitted ruler (NOT the fixed `[-1, 1]`
    /// grid) round-trips within its per-dim step, and the adaptive kernel's L2
    /// distance matches the exact fp32 L2 — i.e. overriding only the ruler is
    /// arithmetically sound. Uses arbitrary (non-normalized, out-of-`[-1,1]`)
    /// values the fixed grid could not represent, sizing the ruler from the
    /// corpus's per-dim `[min,max]` exactly as the build/merge path does.
    #[test]
    fn sq16_adaptive_round_trip_and_kernel_match_fp32_l2() {
        for &dim in &[8usize, 13, 100, 384] {
            let corpus: Vec<Vec<f32>> = (0..16)
                .map(|c| {
                    (0..dim)
                        .map(|i| ((c * 7 + i) as f32 * 0.031).sin() * 5.0 - 1.0)
                        .collect()
                })
                .collect();
            // Per-dim ruler = union of the corpus range (min-of-mins / max-of-maxes),
            // then scale = span / 65535, offset = min — the u16 analogue of the
            // build path's `derive_sq8_quantizer_from_min_max`.
            let mut min = vec![f32::INFINITY; dim];
            let mut max = vec![f32::NEG_INFINITY; dim];
            for v in &corpus {
                for d in 0..dim {
                    min[d] = min[d].min(v[d]);
                    max[d] = max[d].max(v[d]);
                }
            }
            let scale: Vec<f32> = (0..dim)
                .map(|d| {
                    let s = max[d] - min[d];
                    if s > 0.0 { s / 65535.0 } else { 0.0 }
                })
                .collect();
            let offset = min.clone();
            let query = &corpus[3];
            let kernel = Sq16Kernel::new_adaptive(Metric::L2Sq, query, &scale, &offset);
            for v in &corpus {
                let mut bytes = vec![0u8; dim * 2];
                encode_sq16_adaptive_row(v, &scale, &offset, &mut bytes);

                // Round-trip: every component within one quant step (no clamp,
                // because the ruler covers the corpus by construction).
                let mut decoded = vec![0.0f32; dim];
                dequantize_sq16_adaptive_into(&bytes, &scale, &offset, &mut decoded);
                for d in 0..dim {
                    assert!(
                        (decoded[d] - v[d]).abs() <= scale[d] + 1e-4,
                        "dim {dim} d{d}: decoded {} vs {} (step {})",
                        decoded[d],
                        v[d],
                        scale[d]
                    );
                }

                // Adaptive kernel L2 vs exact fp32 L2.
                let norm = sq16_adaptive_norm_sq(&bytes, dim, &scale, &offset);
                let got = kernel.distance_with_norm(&bytes, Some(norm));
                let want = distance(Metric::L2Sq, query, v);
                assert!(
                    (got - want).abs() <= 1e-2 + 1e-3 * want.abs(),
                    "dim {dim}: adaptive L2 {got} vs fp32 {want}"
                );
            }
        }
    }

    /// Recall/ordering sanity: the per-doc-norm correction must never
    /// make Sq16's top-1 ranking worse than uncorrected `1 - dot`. This
    /// plants a case where it is strictly better: the true nearest
    /// neighbor is a unit vector aligned with the query, while a
    /// distractor has a HIGHER raw dot (larger norm) but LOWER cosine.
    /// Norm-corrected cosine ranks the true NN first; raw `1 - dot`
    /// (no correction) wrongly ranks the distractor first.
    #[test]
    fn sq16_norm_correction_ranks_planted_case_at_least_as_well() {
        let dim = 8usize;
        let inv = 1.0 / (dim as f32).sqrt();
        // Unit query with equal components.
        let query = vec![inv; dim];
        // True NN = query direction (cosine 1.0, unit norm).
        let true_nn = query.clone();
        // Unit vector `e ⊥ q` (even/odd sign split → dot(q,e)=0 for even
        // dim, ‖e‖=1), used to build a cosine-0.9 direction.
        let mut e = vec![0.0f32; dim];
        for (i, ei) in e.iter_mut().enumerate() {
            *ei = if i % 2 == 0 { inv } else { -inv };
        }
        let c = 0.9f32;
        let s = (1.0 - c * c).sqrt();
        let d_unit: Vec<f32> = (0..dim).map(|i| c * query[i] + s * e[i]).collect();
        // Scale the distractor so raw dot (1.08) beats the true NN's
        // (1.0) while its cosine (0.9) is lower. Components stay in the
        // [-1, 1] Sq16 grid (asserted), so no clamping distorts the case.
        let distractor: Vec<f32> = d_unit.iter().map(|v| v * 1.2).collect();
        for &v in true_nn.iter().chain(distractor.iter()) {
            assert!(v.abs() <= 1.0, "component {v} outside Sq16 grid");
        }

        let encode = |v: &[f32]| {
            let mut b = vec![0u8; dim * 2];
            encode_sq16_row(v, &mut b);
            b
        };
        let nn_b = encode(&true_nn);
        let dis_b = encode(&distractor);

        let kernel = Sq16Kernel::new(Metric::Cosine, &query);
        let nn_corr = kernel.distance_with_norm(&nn_b, Some(sq16_decoded_norm_sq(&nn_b, dim)));
        let dis_corr = kernel.distance_with_norm(&dis_b, Some(sq16_decoded_norm_sq(&dis_b, dim)));
        assert!(
            nn_corr < dis_corr,
            "norm-corrected: true NN {nn_corr} must rank before distractor {dis_corr}"
        );

        // Uncorrected `1 - dot` (the pre-correction behavior) ranks the
        // distractor first — the exact failure the norm division fixes,
        // so with-norm is here strictly better and never worse.
        let decode = |b: &[u8]| -> Vec<f32> {
            (0..dim)
                .map(|d| {
                    let o = d * 2;
                    u16::from_le_bytes([b[o], b[o + 1]]) as f32 * SQ16_FIXED_SCALE
                        + SQ16_FIXED_OFFSET
                })
                .collect()
        };
        let nn_raw = COSINE_DISTANCE_BASE - dot(&query, &decode(&nn_b));
        let dis_raw = COSINE_DISTANCE_BASE - dot(&query, &decode(&dis_b));
        assert!(
            dis_raw < nn_raw,
            "sanity: uncorrected 1-dot should (wrongly) rank the distractor first \
             (nn_raw {nn_raw}, dis_raw {dis_raw})"
        );
    }

    /// Bisection: does Sq16 match/beat Sq8FixedResidual at the KERNEL
    /// level, on identical data, isolated from routing/shortlist/
    /// pos-mapping? Sq16's grid (2/65535) is finer than Sq8+8's effective
    /// step (2/65280), so its recall@10 vs fp32 truth MUST be >= C1's.
    /// If it holds here, any end-to-end Sq16<C1 gap is a PIPELINE bug, not
    /// the codec; if it fails here, the codec/kernel is the culprit and we
    /// have it reproduced in isolation. Run:
    ///   cargo test -- --nocapture sq16_vs_sq8residual_kernel_recall
    #[test]
    fn sq16_vs_sq8residual_kernel_recall() {
        use std::collections::HashSet;

        use crate::superfile::vector::{
            rerank_codec::{SQ8_FIXED_OFFSET, SQ8_FIXED_RESIDUAL_DIVISOR, SQ8_FIXED_SCALE},
            sq8_simd::{Sq8EncodeConsts, encode_sq8_residual_row},
        };

        fn next_u64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f32(state: &mut u64) -> f32 {
            let u = (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32;
            u * 2.0 - 1.0
        }
        fn fill_unit(state: &mut u64, out: &mut [f32]) {
            for x in out.iter_mut() {
                *x = next_f32(state);
            }
            let n = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in out.iter_mut() {
                    *x /= n;
                }
            }
        }
        // Indices of the k smallest distances (ascending).
        fn topk_asc(dists: &[f32], k: usize) -> Vec<usize> {
            let mut idx: Vec<usize> = (0..dists.len()).collect();
            idx.sort_by(|&a, &b| dists[a].total_cmp(&dists[b]));
            idx.truncate(k);
            idx
        }

        const K: usize = 10;
        let dim = 768usize;
        let mut rng = 0xDEAD_BEEF_1234_5678u64;

        let scale = vec![SQ8_FIXED_SCALE; dim];
        let offset = vec![SQ8_FIXED_OFFSET; dim];
        let divisor = SQ8_FIXED_RESIDUAL_DIVISOR;
        let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);

        // Synthetic unit-norm corpus and queries. Both codecs quantize the
        // same vectors and are scored against the same fp32 truth, so this
        // isolates the codec's quantization error from routing.
        // Kept small so this runs in the default (debug) test suite as a CI
        // guard on the recall bound; the ≥ property is a codec invariant, not
        // sample-size-dependent, and the rng is seeded so recall is
        // deterministic. Larger sweeps run via the bench harness.
        #[allow(non_snake_case)]
        let N = 800usize;
        #[allow(non_snake_case)]
        let Q = 80usize;

        // Corpus: encoded into both codecs with their per-doc norms.
        let mut corpus: Vec<Vec<f32>> = Vec::with_capacity(N);
        let mut sq16_buf = vec![0u8; N * dim * 2];
        let mut sq8_code = vec![0u8; N * dim];
        let mut sq8_res = vec![0u8; N * dim];
        let mut sq16_norms = vec![0.0f32; N];
        let mut sq8_norms = vec![0.0f32; N];
        let mut recon = vec![0.0f32; dim];
        for i in 0..N {
            let mut c = vec![0.0f32; dim];
            fill_unit(&mut rng, &mut c);
            let sc = &mut sq16_buf[i * dim * 2..(i + 1) * dim * 2];
            encode_sq16_row(&c, sc);
            sq16_norms[i] = sq16_decoded_norm_sq(sc, dim);
            let n = encode_sq8_residual_row(
                &c,
                &consts,
                &scale,
                &offset,
                &mut sq8_code[i * dim..(i + 1) * dim],
                &mut sq8_res[i * dim..(i + 1) * dim],
                &mut recon,
                true,
                divisor,
            )
            .expect("store_norm=true yields a per-doc norm");
            sq8_norms[i] = n;
            corpus.push(c);
        }

        let (mut r_sq16, mut r_sq16_nn, mut r_sq8, mut total) = (0usize, 0usize, 0usize, 0usize);
        let mut rng_q = 0x0BAD_F00D_CAFE_BABEu64;
        for _ in 0..Q {
            let mut q = vec![0.0f32; dim];
            fill_unit(&mut rng_q, &mut q);
            // fp32 truth: both unit ⇒ cosine == dot; larger dot = closer.
            let truth_scores: Vec<f32> = corpus.iter().map(|c| dot(&q, c)).collect();
            let mut ti: Vec<usize> = (0..N).collect();
            ti.sort_by(|&a, &b| truth_scores[b].total_cmp(&truth_scores[a]));
            let truth: HashSet<usize> = ti.into_iter().take(K).collect();

            let sq16_kernel = Sq16Kernel::new(Metric::Cosine, &q);
            let sq8_kernel = Sq8ResidualKernel::new(Metric::Cosine, &q, &scale, &offset, divisor);

            let sq16_d: Vec<f32> = (0..N)
                .map(|i| {
                    sq16_kernel.distance_with_norm(
                        &sq16_buf[i * dim * 2..(i + 1) * dim * 2],
                        Some(sq16_norms[i]),
                    )
                })
                .collect();
            let sq8_d: Vec<f32> = (0..N)
                .map(|i| {
                    sq8_kernel.distance_with_norm(
                        &sq8_code[i * dim..(i + 1) * dim],
                        &sq8_res[i * dim..(i + 1) * dim],
                        Some(sq8_norms[i]),
                    )
                })
                .collect();
            // Sq16 without the norm division (raw 1 - dot on decoded).
            let sq16_nn_d: Vec<f32> = (0..N)
                .map(|i| {
                    let base = i * dim * 2;
                    let mut d = 0.0f32;
                    for (j, &qj) in q.iter().enumerate().take(dim) {
                        let b = base + j * 2;
                        let code = u16::from_le_bytes([sq16_buf[b], sq16_buf[b + 1]]) as f32;
                        d += qj * (code * SQ16_FIXED_SCALE + SQ16_FIXED_OFFSET);
                    }
                    COSINE_DISTANCE_BASE - d
                })
                .collect();

            for &i in topk_asc(&sq16_d, K).iter() {
                if truth.contains(&i) {
                    r_sq16 += 1;
                }
            }
            for &i in topk_asc(&sq16_nn_d, K).iter() {
                if truth.contains(&i) {
                    r_sq16_nn += 1;
                }
            }
            for &i in topk_asc(&sq8_d, K).iter() {
                if truth.contains(&i) {
                    r_sq8 += 1;
                }
            }
            total += K;
        }

        let rec = |x: usize| x as f64 / total as f64;
        eprintln!(
            "\n### Kernel-level recall@{K} vs fp32 truth (synthetic, N={N}, Q={Q}, dim={dim})"
        );
        eprintln!("Sq16 (norm)      : {:.4}", rec(r_sq16));
        eprintln!("Sq16 (no-norm)   : {:.4}", rec(r_sq16_nn));
        eprintln!("Sq8FixedResidual : {:.4}", rec(r_sq8));
        eprintln!("Sq16norm - C1    : {:+.4}", rec(r_sq16) - rec(r_sq8));
        eprintln!("Sq16norm - nonorm: {:+.4}", rec(r_sq16) - rec(r_sq16_nn));

        // Theorem: the finer grid means Sq16 must not trail C1 at the
        // kernel level. A loose margin absorbs tie-break noise; the
        // printed numbers are the real diagnostic.
        assert!(
            rec(r_sq16) >= rec(r_sq8) - 0.002,
            "Sq16 kernel recall {:.4} trails C1 {:.4} by >0.002 — codec-level bug reproduced",
            rec(r_sq16),
            rec(r_sq8)
        );
    }

    /// Microbench: intrinsic per-candidate rerank scoring cost of the
    /// two-leg `Sq8ResidualKernel` (u8 coarse + i8 residual) vs the
    /// single-leg `Sq16Kernel` (one u16 plane), isolated from the
    /// shortlist / IVF / IO. Both on the fixed cosine grid. This is the
    /// "one leg vs two" cost the end-to-end query only shows diluted
    /// (rerank is a small slice of total query time).
    ///
    /// Ignored by default (it allocates hundreds of MB and only means
    /// anything in release). Run with:
    ///   cargo test --release -- --ignored --nocapture rerank_kernel_leg_cost_microbench
    #[test]
    #[ignore = "microbench: run explicitly in release with --nocapture"]
    fn rerank_kernel_leg_cost_microbench() {
        use std::{hint::black_box, time::Instant};

        use crate::superfile::vector::{
            rerank_codec::{SQ8_FIXED_OFFSET, SQ8_FIXED_RESIDUAL_DIVISOR, SQ8_FIXED_SCALE},
            sq8_simd::{Sq8EncodeConsts, encode_sq8_residual_row},
        };

        // Deterministic splitmix64 → f32 in [-1, 1]; no external RNG so
        // the numbers reproduce bit-for-bit across runs.
        fn next_u64(state: &mut u64) -> u64 {
            *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn next_f32(state: &mut u64) -> f32 {
            let u = (next_u64(state) >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
            u * 2.0 - 1.0
        }
        fn fill_unit(state: &mut u64, out: &mut [f32]) {
            for x in out.iter_mut() {
                *x = next_f32(state);
            }
            let n = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            if n > 0.0 {
                for x in out.iter_mut() {
                    *x /= n;
                }
            }
        }

        const N: usize = 100_000;
        const PASSES: usize = 7;
        let mut rng = 0x1234_5678_9ABC_DEF0u64;

        eprintln!("\n### Rerank kernel per-candidate cost — Sq8Residual (2 legs) vs Sq16 (1 leg)");
        eprintln!("N = {N} candidates, cosine/fixed-grid, median of {PASSES} timed passes\n");
        eprintln!(
            "{:>6}  {:>16}  {:>16}  {:>10}",
            "dim", "sq8_residual ns", "sq16 ns", "sq16/sq8"
        );

        for &dim in &[768usize, 1536] {
            let scale = vec![SQ8_FIXED_SCALE; dim];
            let offset = vec![SQ8_FIXED_OFFSET; dim];
            let divisor = SQ8_FIXED_RESIDUAL_DIVISOR;
            let consts = Sq8EncodeConsts::from_scale_offset(&scale, &offset);

            // Random unit query, one kernel per codec (built once, as the
            // reader does).
            let mut query = vec![0.0f32; dim];
            fill_unit(&mut rng, &mut query);
            let sq16_kernel = Sq16Kernel::new(Metric::Cosine, &query);
            let sq8_kernel =
                Sq8ResidualKernel::new(Metric::Cosine, &query, &scale, &offset, divisor);

            // Encode all N candidates once into each codec's on-disk
            // byte layout. The fp32 source is dropped after encoding so
            // only the packed codec buffers stay resident.
            let mut sq16_buf = vec![0u8; N * dim * 2];
            let mut sq8_code = vec![0u8; N * dim];
            let mut sq8_res = vec![0u8; N * dim];
            // Both codecs now carry per-doc dequantized norms for the
            // norm-corrected cosine kernel; store both.
            let mut sq8_norms = vec![0.0f32; N];
            let mut sq16_norms = vec![0.0f32; N];
            let mut cand = vec![0.0f32; dim];
            let mut recon = vec![0.0f32; dim];
            for i in 0..N {
                fill_unit(&mut rng, &mut cand);
                let sq16_code = &mut sq16_buf[i * dim * 2..(i + 1) * dim * 2];
                encode_sq16_row(&cand, sq16_code);
                sq16_norms[i] = sq16_decoded_norm_sq(sq16_code, dim);
                let norm = encode_sq8_residual_row(
                    &cand,
                    &consts,
                    &scale,
                    &offset,
                    &mut sq8_code[i * dim..(i + 1) * dim],
                    &mut sq8_res[i * dim..(i + 1) * dim],
                    &mut recon,
                    true,
                    divisor,
                )
                .expect("store_norm=true yields a per-doc norm");
                sq8_norms[i] = norm;
            }

            // Score every candidate through the codec's per-candidate
            // distance call — the exact call the reader makes. black_box
            // on inputs + the accumulated sink so nothing folds away.
            let time_sq16 = || {
                let mut sink = 0.0f32;
                for i in 0..N {
                    let code = black_box(&sq16_buf[i * dim * 2..(i + 1) * dim * 2]);
                    sink += sq16_kernel.distance_with_norm(code, Some(black_box(sq16_norms[i])));
                }
                black_box(sink);
            };
            let time_sq8 = || {
                let mut sink = 0.0f32;
                for i in 0..N {
                    let code = black_box(&sq8_code[i * dim..(i + 1) * dim]);
                    let res = black_box(&sq8_res[i * dim..(i + 1) * dim]);
                    sink += sq8_kernel.distance_with_norm(code, res, Some(black_box(sq8_norms[i])));
                }
                black_box(sink);
            };

            let median = |f: &mut dyn FnMut()| -> f64 {
                f(); // warm-up pass, not timed
                let mut samples: Vec<f64> = (0..PASSES)
                    .map(|_| {
                        let t = Instant::now();
                        f();
                        t.elapsed().as_secs_f64()
                    })
                    .collect();
                samples.sort_by(|a, b| a.total_cmp(b));
                samples[PASSES / 2]
            };

            let mut sq16_fn = time_sq16;
            let mut sq8_fn = time_sq8;
            let sq16_secs = median(&mut sq16_fn);
            let sq8_secs = median(&mut sq8_fn);
            let sq16_ns = sq16_secs / N as f64 * 1e9;
            let sq8_ns = sq8_secs / N as f64 * 1e9;
            eprintln!(
                "{dim:>6}  {sq8_ns:>16.2}  {sq16_ns:>16.2}  {:>10.3}",
                sq16_ns / sq8_ns
            );
        }
        eprintln!();
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
