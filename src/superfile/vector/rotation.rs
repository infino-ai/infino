// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Random orthogonal rotation built from a deterministic seed.
//!
//! Sign-quantization (1-bit RaBitQ) is useless without rotation: every
//! component of the input would map to the same handful of possibilities
//! and the bit-pattern would carry almost no information. A random
//! orthogonal `R` turns each bit into an LSH-style hyperplane, spreading
//! the data's variance so sign-encoding becomes informative.
//!
//! Construction — **fast structured rotation**, not a dense matrix. A
//! dense `dim × dim` matrix-vector product is `O(dim²)` per vector,
//! which dominates build *and* search as the embedding dimension grows
//! (e.g. 384 → 1024 is a 7.1× blow-up in this step alone). Instead we
//! use the standard fast-Johnson–Lindenstrauss / "structured spinner"
//! construction RaBitQ and FAISS use:
//!
//! ```text
//!   R · x  =  (HD)_3 (HD)_2 (HD)_1 · x
//! ```
//!
//! where each stage is a random sign-flip diagonal `D_s` (±1, seeded)
//! followed by a normalized Walsh–Hadamard transform `H`. The WHT is
//! `O(dim·log dim)` via the butterfly, and both `D` and the normalized
//! `H` are orthonormal, so the composition is an orthonormal rotation —
//! it preserves L2 norm and inner products exactly when `dim` is a power
//! of two. Three stages give the variance-mixing the 1-bit codes need.
//!
//! Non-power-of-two `dim` is zero-padded up to the next power of two for
//! the transform, then the leading `dim` components are returned; that
//! is a fast random projection (approximately norm-preserving), which is
//! all the LSH sign-encoding requires. Power-of-two dims (the common
//! case — 1024, 512, 256) take the exact-isometry path with no padding.
//!
//! Determinism: `RandomRotation::new(dim, seed)` derives the same sign
//! diagonals for the same `(dim, seed)` pair on any platform. The reader
//! reconstructs the exact rotation the builder used from the stored
//! `(dim, rot_seed)` in the per-column subsection header, so nothing
//! about the rotation is persisted — swapping the construction is not an
//! on-disk format change.

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};
use wide::f32x8;

/// Number of (sign-flip, Walsh–Hadamard) stages composed into the
/// rotation. Three is the standard "structured spinner" depth: it mixes
/// variance across coordinates well enough for the 1-bit RaBitQ codes
/// while staying `O(dim·log dim)`.
const ROTATION_STAGES: usize = 3;

/// `wide::f32x8` lane width (butterfly + sign/scale loops).
///
/// The butterfly stays on `wide` (256-bit / AVX2) deliberately: a
/// 16-lane AVX-512 variant was measured **~8% slower** across the dim
/// sweep (memory-bound load×2/store×2 with near-zero compute, so the
/// wider lane buys nothing and the AVX-512 downclock costs).
const F32X8_LANES: usize = 8;

/// Fast structured random rotation (sign-flip + Walsh–Hadamard,
/// `O(dim·log dim)` per `apply`). See the module docs.
#[derive(Debug)]
pub struct RandomRotation {
    pub dim: usize,
    /// Transform working size: `dim` rounded up to a power of two.
    padded_dim: usize,
    /// One `±1` sign-flip diagonal per stage, each length `padded_dim`.
    signs: Vec<Vec<f32>>,
}

impl RandomRotation {
    /// Build the rotation. Only the seeded `±1` sign diagonals are
    /// materialized (`O(stages · dim)` memory); the Walsh–Hadamard part
    /// is implicit in [`apply`](Self::apply).
    pub fn new(dim: usize, seed: u64) -> Self {
        let padded_dim = dim.max(1).next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);
        // Rademacher `±1` draws via the sign of a standard normal
        // (`P(<0) = P(≥0) = 0.5`), reusing the same seeded RNG +
        // distribution the dense rotation used so determinism is
        // identical in shape.
        let normal = Normal::new(0.0f32, 1.0).expect("valid stddev");
        let signs = (0..ROTATION_STAGES)
            .map(|_| {
                (0..padded_dim)
                    .map(|_| {
                        if normal.sample(&mut rng) >= 0.0 {
                            1.0f32
                        } else {
                            -1.0f32
                        }
                    })
                    .collect()
            })
            .collect();
        RandomRotation {
            dim,
            padded_dim,
            signs,
        }
    }

    /// Compute `out = R · x`. Both slices must have length `dim`.
    ///
    /// Runs the staged sign-flip + Walsh–Hadamard transform in a
    /// thread-local scratch buffer (one allocation per thread, reused),
    /// then copies the leading `dim` components into `out`.
    #[inline]
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        let m = self.padded_dim;
        // Normalized WHT scales each transform by 1/sqrt(m); applied once
        // per stage keeps the whole composition orthonormal.
        let scale = 1.0 / (m as f32).sqrt();
        SCRATCH.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.resize(m, 0.0);
            buf[..self.dim].copy_from_slice(x);
            for stage_signs in &self.signs {
                apply_signs(&mut buf, stage_signs);
                walsh_hadamard(&mut buf);
                scale_in_place(&mut buf, scale);
            }
            out.copy_from_slice(&buf[..self.dim]);
        });
    }
}

thread_local! {
    /// Per-thread reusable transform buffer (length = `padded_dim`).
    /// Avoids an allocation on every `apply` (the per-vector hot path
    /// at build and per-query at search).
    static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `buf[i] *= signs[i]` (`±1` flip), 8 lanes at a time.
#[inline]
fn apply_signs(buf: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(buf.len(), signs.len());
    let n = buf.len();
    let mut i = 0;
    while i + F32X8_LANES <= n {
        let b =
            f32x8::from(<[f32; F32X8_LANES]>::try_from(&buf[i..i + F32X8_LANES]).expect("len-8"));
        let s =
            f32x8::from(<[f32; F32X8_LANES]>::try_from(&signs[i..i + F32X8_LANES]).expect("len-8"));
        buf[i..i + F32X8_LANES].copy_from_slice(&(b * s).to_array());
        i += F32X8_LANES;
    }
    while i < n {
        buf[i] *= signs[i];
        i += 1;
    }
}

/// `buf[i] *= scale`, 8 lanes at a time.
#[inline]
fn scale_in_place(buf: &mut [f32], scale: f32) {
    let v = f32x8::splat(scale);
    let n = buf.len();
    let mut i = 0;
    while i + F32X8_LANES <= n {
        let b =
            f32x8::from(<[f32; F32X8_LANES]>::try_from(&buf[i..i + F32X8_LANES]).expect("len-8"));
        buf[i..i + F32X8_LANES].copy_from_slice(&(b * v).to_array());
        i += F32X8_LANES;
    }
    while i < n {
        buf[i] *= scale;
        i += 1;
    }
}

/// In-place (unnormalized) Walsh–Hadamard transform. `a.len()` must be a
/// power of two. Strides `h ≥ 8` run 8 lanes at a time on `wide::f32x8`
/// (256-bit / AVX2); `h ∈ {1,2,4}` fall back to scalar. A 16-lane
/// AVX-512 variant was tried and measured slower (memory-bound; see the
/// note on [`F32X8_LANES`]), so this is the only butterfly.
#[inline]
fn walsh_hadamard(a: &mut [f32]) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            if h >= F32X8_LANES {
                let mut j = i;
                while j < i + h {
                    let x = f32x8::from(
                        <[f32; F32X8_LANES]>::try_from(&a[j..j + F32X8_LANES]).expect("len-8"),
                    );
                    let y = f32x8::from(
                        <[f32; F32X8_LANES]>::try_from(&a[j + h..j + h + F32X8_LANES])
                            .expect("len-8"),
                    );
                    a[j..j + F32X8_LANES].copy_from_slice(&(x + y).to_array());
                    a[j + h..j + h + F32X8_LANES].copy_from_slice(&(x - y).to_array());
                    j += F32X8_LANES;
                }
            } else {
                for j in i..i + h {
                    let x = a[j];
                    let y = a[j + h];
                    a[j] = x + y;
                    a[j + h] = x - y;
                }
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::vector::distance::dot;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Column `i` of `R`: the image of the `i`-th standard basis
    /// vector, `R · e_i`. For an orthonormal `R` the columns are unit
    /// vectors, pairwise orthogonal.
    fn column(rot: &RandomRotation, i: usize) -> Vec<f32> {
        let mut e = vec![0.0f32; rot.dim];
        e[i] = 1.0;
        let mut out = vec![0.0f32; rot.dim];
        rot.apply(&e, &mut out);
        out
    }

    // --- structural ----------------------------------------------------

    #[test]
    fn new_with_dim_8_succeeds() {
        let r = RandomRotation::new(8, 42);
        assert_eq!(r.dim, 8);
        assert_eq!(r.padded_dim, 8);
        assert_eq!(r.signs.len(), ROTATION_STAGES);
    }

    #[test]
    fn new_with_realistic_dim_succeeds() {
        for &dim in &[16, 64, 128, 384, 768, 1024] {
            let r = RandomRotation::new(dim, 7);
            assert_eq!(r.dim, dim);
            assert!(r.padded_dim >= dim && r.padded_dim.is_power_of_two());
        }
    }

    // --- orthonormality (power-of-two dims are exact isometries) -------

    #[test]
    fn columns_are_unit_vectors() {
        let r = RandomRotation::new(64, 7);
        for i in 0..r.dim {
            let c = column(&r, i);
            let mag_sq = dot(&c, &c);
            assert!(approx(mag_sq, 1.0, 1e-4), "column {i} mag² = {mag_sq}");
        }
    }

    #[test]
    fn columns_are_pairwise_orthogonal() {
        let r = RandomRotation::new(32, 11);
        for i in 0..r.dim {
            let ci = column(&r, i);
            for j in (i + 1)..r.dim {
                let cj = column(&r, j);
                let p = dot(&ci, &cj);
                assert!(approx(p, 0.0, 1e-4), "columns {i}, {j} dot = {p}");
            }
        }
    }

    // --- determinism ---------------------------------------------------

    #[test]
    fn same_seed_yields_same_rotation() {
        let r1 = RandomRotation::new(64, 12345);
        let r2 = RandomRotation::new(64, 12345);
        assert_eq!(r1.signs, r2.signs);
        // And the same output on the same input.
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let (mut a, mut b) = (vec![0.0; 64], vec![0.0; 64]);
        r1.apply(&x, &mut a);
        r2.apply(&x, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_yields_different_rotation() {
        let r1 = RandomRotation::new(64, 1);
        let r2 = RandomRotation::new(64, 2);
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 + 1.0).collect();
        let (mut a, mut b) = (vec![0.0; 64], vec![0.0; 64]);
        r1.apply(&x, &mut a);
        r2.apply(&x, &mut b);
        assert_ne!(a, b);
    }

    // --- apply ---------------------------------------------------------

    #[test]
    fn apply_preserves_l2_norm() {
        // Orthogonal `R` is an isometry: |R·x| = |x|.
        let r = RandomRotation::new(64, 42);
        let mut x = vec![0.0f32; 64];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32) * 0.1 - 1.5;
        }
        let mag_in = dot(&x, &x).sqrt();
        let mut y = vec![0.0; 64];
        r.apply(&x, &mut y);
        let mag_out = dot(&y, &y).sqrt();
        assert!(
            approx(mag_in, mag_out, 1e-3),
            "input |x| = {mag_in}, output |R·x| = {mag_out}"
        );
    }

    #[test]
    fn apply_zero_vector_yields_zero() {
        let r = RandomRotation::new(32, 0xCAFE_F00D);
        let x = vec![0.0; 32];
        let mut y = vec![1.0; 32];
        r.apply(&x, &mut y);
        for &v in &y {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn apply_preserves_inner_products() {
        // Orthogonal `R` preserves dot products: <R·x, R·y> = <x, y>.
        let r = RandomRotation::new(32, 7);
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.3 - 4.0).collect();
        let y: Vec<f32> = (0..32).map(|i| (i as f32) * -0.2 + 1.7).collect();
        let mut rx = vec![0.0; 32];
        let mut ry = vec![0.0; 32];
        r.apply(&x, &mut rx);
        r.apply(&y, &mut ry);
        let inner_in = dot(&x, &y);
        let inner_out = dot(&rx, &ry);
        assert!(
            approx(inner_in, inner_out, 1e-3),
            "<x,y> = {inner_in}, <Rx,Ry> = {inner_out}"
        );
    }

    #[test]
    fn apply_is_linear() {
        // R(x + αy) == R(x) + αR(y).
        let r = RandomRotation::new(16, 99);
        let x: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5).collect();
        let alpha = 2.5;

        let mut rx = vec![0.0; 16];
        let mut ry = vec![0.0; 16];
        r.apply(&x, &mut rx);
        r.apply(&y, &mut ry);

        let combined: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + alpha * b).collect();
        let mut r_combined = vec![0.0; 16];
        r.apply(&combined, &mut r_combined);

        for i in 0..16 {
            let expected = rx[i] + alpha * ry[i];
            assert!(
                approx(r_combined[i], expected, 1e-3),
                "linearity broken at i={i}: got {} expected {expected}",
                r_combined[i]
            );
        }
    }
}
