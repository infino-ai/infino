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
//! Construction: sample a `dim × dim` Gaussian matrix from a seeded RNG,
//! Gram-Schmidt-orthonormalize the rows, store row-major. `apply(x)` is
//! the matrix-vector product `R · x`.
//!
//! Determinism: `RandomRotation::new(dim, seed)` returns the same matrix
//! for the same `(dim, seed)` pair on any platform. The reader needs to
//! reconstruct the exact rotation the builder used (we store
//! `(dim, rot_seed)` in the per-column subsection header), so this
//! guarantee is load-bearing.

use crate::superfile::vector::distance::dot;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// Minimum row magnitude before Gram-Schmidt normalization divides by
/// it. A row whose magnitude falls at or below this is treated as
/// numerically zero (a near-degenerate Gaussian draw) and left
/// unnormalized rather than blown up by division.
const GRAM_SCHMIDT_MIN_ROW_MAGNITUDE: f32 = 1e-12;

/// Row-major `dim × dim` orthonormal matrix.
#[derive(Debug)]
pub struct RandomRotation {
    pub dim: usize,
    /// Row-major: row `i` lives at `rows[i*dim .. (i+1)*dim]`.
    rows: Vec<f32>,
}

impl RandomRotation {
    /// Build the rotation. Cost is `O(dim³)` for Gram-Schmidt — fine
    /// at column-build time, but cache the result if called repeatedly.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).expect("valid stddev");
        let mut rows = vec![0.0f32; dim * dim];
        for x in rows.iter_mut() {
            *x = normal.sample(&mut rng);
        }
        // Gram-Schmidt: subtract projections of all earlier rows from
        // each new row, then unit-normalize.
        for i in 0..dim {
            for j in 0..i {
                let (hi_row, lo_row) = rows.split_at_mut(i * dim);
                let row_j = &hi_row[j * dim..(j + 1) * dim];
                let row_i = &mut lo_row[..dim];
                let proj = dot(row_j, row_i);
                for (xi, xj) in row_i.iter_mut().zip(row_j) {
                    *xi -= proj * xj;
                }
            }
            let row_i = &mut rows[i * dim..(i + 1) * dim];
            let mag: f32 = row_i.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > GRAM_SCHMIDT_MIN_ROW_MAGNITUDE {
                let inv = 1.0 / mag;
                for x in row_i.iter_mut() {
                    *x *= inv;
                }
            }
        }
        RandomRotation { dim, rows }
    }

    /// Compute `out = R · x`. Both slices must have length `dim`.
    #[inline]
    pub fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.dim);
        debug_assert_eq!(out.len(), self.dim);
        for (i, slot) in out.iter_mut().enumerate() {
            let row = &self.rows[i * self.dim..(i + 1) * self.dim];
            *slot = dot(row, x);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn row(rot: &RandomRotation, i: usize) -> &[f32] {
        &rot.rows[i * rot.dim..(i + 1) * rot.dim]
    }

    // --- structural ----------------------------------------------------

    #[test]
    fn new_with_dim_8_succeeds() {
        let r = RandomRotation::new(8, 42);
        assert_eq!(r.dim, 8);
        assert_eq!(r.rows.len(), 64);
    }

    #[test]
    fn new_with_realistic_dim_succeeds() {
        for &dim in &[16, 64, 128, 384] {
            let r = RandomRotation::new(dim, 7);
            assert_eq!(r.dim, dim);
            assert_eq!(r.rows.len(), dim * dim);
        }
    }

    // --- orthonormality ------------------------------------------------

    #[test]
    fn rows_are_unit_vectors() {
        let r = RandomRotation::new(64, 7);
        for i in 0..r.dim {
            let mag_sq = dot(row(&r, i), row(&r, i));
            assert!(approx(mag_sq, 1.0, 1e-4), "row {i} mag² = {mag_sq}");
        }
    }

    #[test]
    fn rows_are_pairwise_orthogonal() {
        let r = RandomRotation::new(32, 11);
        for i in 0..r.dim {
            for j in (i + 1)..r.dim {
                let p = dot(row(&r, i), row(&r, j));
                assert!(approx(p, 0.0, 1e-4), "rows {i}, {j} dot = {p}");
            }
        }
    }

    // --- determinism ---------------------------------------------------

    #[test]
    fn same_seed_yields_same_matrix() {
        let r1 = RandomRotation::new(64, 12345);
        let r2 = RandomRotation::new(64, 12345);
        assert_eq!(r1.rows, r2.rows);
    }

    #[test]
    fn different_seed_yields_different_matrix() {
        let r1 = RandomRotation::new(64, 1);
        let r2 = RandomRotation::new(64, 2);
        assert_ne!(r1.rows, r2.rows);
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
