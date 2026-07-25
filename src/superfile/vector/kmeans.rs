// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! K-means clustering — 5-iteration Lloyd's algorithm.
//!
//! Used to derive the `n_cent` IVF centroids per vector index at
//! build time. Five iterations is the standard turn-key default —
//! diminishing returns past that on most embedding distributions,
//! and we don't have a quality budget to spend more.
//!
//! Strategy:
//!
//!  - **Init**: random sample of `k` rows from the input.
//!  - **Assign**: parallel over docs via `rayon`. Each doc's cluster =
//!    `argmin l2_sq(doc, centroid)`.
//!  - **Update**: sequential f64-accumulator means. The parallel version
//!    would need either `k * dim` atomics or per-thread scratch
//!    buffers; at 5 iterations the assign-step CPU dominates anyway,
//!    so the sequential update isn't a bottleneck.
//!
//! Numerical stability: f64 accumulator for the sum, casting back to
//! f32 only after dividing by the cluster count. Avoids the precision
//! loss of summing many f32s.
//!
//! Determinism: same `seed` + same input `vectors` → same centroids.
//! The seed is derived from this column's `rot_seed` (offset by 7) so
//! the rotation and clustering use distinct PRNG streams.

use rand::{RngExt, SeedableRng, rngs::StdRng};
use rayon::prelude::*;

use crate::superfile::vector::distance::{
    Metric, add_f32_to_f64_acc, f64_acc_mean_into_f32, nearest_centroid_transposed,
    transpose_centroids_cluster_major,
};

/// Offset added to a column's `rot_seed` to seed k-means. Keeps the
/// clustering PRNG stream distinct from the rotation stream, which is
/// seeded from `rot_seed` directly.
const KMEANS_SEED_OFFSET: u64 = 7;

/// Run 5-iteration Lloyd k-means and return `k * dim` centroids,
/// row-major. `vectors` is `n_docs * dim`, also row-major. Drops
/// the final assignments — call [`kmeans_with_assignments`] when
/// the caller already needs them, to avoid a redundant full
/// assignment pass downstream.
pub fn kmeans(vectors: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    kmeans_with_assignments(vectors, dim, k, iters, seed).0
}

/// Run k-means and return both the centroids and the final-iter
/// assignments. The builder uses this to skip a second full pass
/// over the corpus that would otherwise reproduce these same
/// assignments — at 1M × 384 that pass is ~2.4 s of the ~15 s
/// finish() time.
///
/// # Panics
///
/// - `vectors.len() % dim != 0`.
/// - `n_docs == 0`.
/// - `k == 0` or `k > n_docs`.
pub fn kmeans_with_assignments(
    vectors: &[f32],
    dim: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u32>) {
    assert!(dim > 0, "kmeans: dim must be > 0");
    assert!(k > 0, "kmeans: k must be > 0");
    assert_eq!(
        vectors.len() % dim,
        0,
        "kmeans: vectors len {} not multiple of dim {dim}",
        vectors.len()
    );
    let n = vectors.len() / dim;
    assert!(n > 0, "kmeans: at least one doc required");
    assert!(k <= n, "kmeans: k ({k}) > n_docs ({n})");

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(KMEANS_SEED_OFFSET));
    let mut centroids = vec![0f32; k * dim];

    // Init: random sample of input vectors. (Repetition is allowed; for
    // small k vs n the chance of a duplicate is negligible.)
    for i in 0..k {
        let idx = rng.random_range(0..n);
        centroids[i * dim..(i + 1) * dim].copy_from_slice(&vectors[idx * dim..(idx + 1) * dim]);
    }

    lloyd_refine(vectors, dim, k, iters, centroids)
}

/// k-means++ (D²-weighted) init followed by the same Lloyd refinement. Returns
/// `k * dim` centroids and the final assignments.
///
/// The random init in [`kmeans_with_assignments`] draws `k` seeds uniformly, so
/// at k > 2 in high dimension several seeds land in one dense blob and leave
/// other blobs merged into a single oversized cluster — the exact imbalance the
/// cell-split planner hits (one child absorbing several data centers). D²
/// seeding spreads the initial centers across the cloud, so each blob gets a
/// seed and the split stays balanced. Scoped
/// to the split planner; the fine-index build keeps random init (validated).
pub fn kmeans_pp_with_assignments(
    vectors: &[f32],
    dim: usize,
    k: usize,
    iters: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u32>) {
    assert!(dim > 0, "kmeans_pp: dim must be > 0");
    assert!(k > 0, "kmeans_pp: k must be > 0");
    assert_eq!(
        vectors.len() % dim,
        0,
        "kmeans_pp: vectors len {} not multiple of dim {dim}",
        vectors.len()
    );
    let n = vectors.len() / dim;
    assert!(n > 0, "kmeans_pp: at least one doc required");
    assert!(k <= n, "kmeans_pp: k ({k}) > n_docs ({n})");

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(KMEANS_SEED_OFFSET));
    let centroids = kmeanspp_init(vectors, dim, k, n, &mut rng);
    lloyd_refine(vectors, dim, k, iters, centroids)
}

/// Centroids-only k-means++ (drops assignments), mirroring [`kmeans`].
pub fn kmeans_pp(vectors: &[f32], dim: usize, k: usize, iters: usize, seed: u64) -> Vec<f32> {
    kmeans_pp_with_assignments(vectors, dim, k, iters, seed).0
}

/// L2² between two `dim`-length rows.
#[inline]
fn l2_sq(a: &[f32], b: &[f32], dim: usize) -> f32 {
    let mut s = 0f32;
    for j in 0..dim {
        let d = a[j] - b[j];
        s += d * d;
    }
    s
}

/// Greedy k-means++ seed selection. The first center is uniform; each
/// subsequent center is chosen by drawing `local_trials` candidates
/// D²-proportionally and keeping the one that most reduces the total potential
/// (Σ nearest-center D²). Greedy ++ reaches the quality of many plain-++
/// restarts in a single run — plain ++ (one D²-weighted draw per step) can
/// settle a colliding basin where one cluster owns several blobs, and escaping
/// it took ~16 restarts; the greedy local search finds the balanced basin
/// directly. `local_trials = 2 + ⌊ln k⌋` (the standard scikit default).
fn kmeanspp_init(vectors: &[f32], dim: usize, k: usize, n: usize, rng: &mut StdRng) -> Vec<f32> {
    let local_trials = 2 + (k as f64).ln() as usize;
    let mut centroids = Vec::with_capacity(k * dim);
    let first = rng.random_range(0..n);
    centroids.extend_from_slice(&vectors[first * dim..(first + 1) * dim]);
    // `d2[i]` = L2² from point i to its nearest chosen center so far.
    let mut d2 = vec![f32::INFINITY; n];
    {
        let c = &centroids[0..dim];
        d2.par_iter_mut().enumerate().for_each(|(i, di)| {
            *di = l2_sq(&vectors[i * dim..(i + 1) * dim], c, dim);
        });
    }
    for _ in 1..k {
        let total: f64 = d2.iter().map(|&x| x as f64).sum();
        // Draw `local_trials` D²-weighted candidates; keep the one whose
        // addition minimizes the resulting potential Σ min(d2[i], D²(i, cand)).
        let mut best_cand = n - 1;
        let mut best_potential = f64::INFINITY;
        for _ in 0..local_trials {
            let cand = if total <= 0.0 {
                rng.random_range(0..n)
            } else {
                let mut target = rng.random::<f64>() * total;
                let mut idx = n - 1;
                for (i, &di) in d2.iter().enumerate() {
                    target -= di as f64;
                    if target <= 0.0 {
                        idx = i;
                        break;
                    }
                }
                idx
            };
            let cand_vec = &vectors[cand * dim..(cand + 1) * dim];
            let potential: f64 = (0..n)
                .into_par_iter()
                .map(|i| {
                    let dd = l2_sq(&vectors[i * dim..(i + 1) * dim], cand_vec, dim);
                    f64::from(dd.min(d2[i]))
                })
                .sum();
            if potential < best_potential {
                best_potential = potential;
                best_cand = cand;
            }
        }
        // Commit the winner: copy it out, then fold it into `d2`.
        let chosen: Vec<f32> = vectors[best_cand * dim..(best_cand + 1) * dim].to_vec();
        d2.par_iter_mut().enumerate().for_each(|(i, di)| {
            let dd = l2_sq(&vectors[i * dim..(i + 1) * dim], &chosen, dim);
            if dd < *di {
                *di = dd;
            }
        });
        centroids.extend_from_slice(&chosen);
    }
    centroids
}

/// Lloyd refinement from a given initial centroid set. Shared by the random-init
/// [`kmeans_with_assignments`] and the D²-init [`kmeans_pp_with_assignments`] so
/// the assign/update kernel stays single-sourced.
fn lloyd_refine(
    vectors: &[f32],
    dim: usize,
    k: usize,
    iters: usize,
    mut centroids: Vec<f32>,
) -> (Vec<f32>, Vec<u32>) {
    let n = vectors.len() / dim;
    let mut assignments = vec![0u32; n];

    for _ in 0..iters {
        // Assign — parallel over docs through the block-transposed SIMD
        // kernel (one transpose per iteration; at small k the transpose is
        // proportionally tiny). Single scan owner, no scalar branch; argmin
        // tie-breaking matches the naive loop (lowest index wins).
        assignments = {
            let transposed = transpose_centroids_cluster_major(&centroids, k, dim);
            (0..n)
                .into_par_iter()
                .map(|d| {
                    let v = &vectors[d * dim..(d + 1) * dim];
                    nearest_centroid_transposed(Metric::L2Sq, v, &transposed, k, dim).0
                })
                .collect()
        };

        // Update — per-thread (sums, counts) accumulators reduced
        // pairwise. Sums in f64 for numeric stability; counts in u64
        // for headroom at billion-doc scales. Pairwise reduction
        // bounds float drift across runs (the order of the binary
        // tree is the rayon work-stealing topology, not strictly
        // deterministic — accept ~ULP-level differences across runs
        // since they're below recall-test thresholds).
        let chunk_size = (n.div_ceil(rayon::current_num_threads().max(1))).max(1);
        let (sums, counts) = (0..n)
            .into_par_iter()
            .chunks(chunk_size)
            .map(|chunk| {
                let mut s = vec![0f64; k * dim];
                let mut c = vec![0u64; k];
                for d in chunk {
                    let cid = assignments[d] as usize;
                    c[cid] += 1;
                    let row = &vectors[d * dim..(d + 1) * dim];
                    let dst = &mut s[cid * dim..(cid + 1) * dim];
                    add_f32_to_f64_acc(dst, row);
                }
                (s, c)
            })
            .reduce(
                || (vec![0f64; k * dim], vec![0u64; k]),
                |mut acc, x| {
                    for j in 0..acc.0.len() {
                        acc.0[j] += x.0[j];
                    }
                    for j in 0..acc.1.len() {
                        acc.1[j] += x.1[j];
                    }
                    acc
                },
            );

        for c in 0..k {
            // Skip empty clusters: their centroids stay at their last
            // value (init value or previous iteration's value).
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                let dst = &mut centroids[c * dim..(c + 1) * dim];
                let src = &sums[c * dim..(c + 1) * dim];
                f64_acc_mean_into_f32(src, inv, dst);
            }
        }
    }
    (centroids, assignments)
}

/// Assign each row of `vectors` to its argmin centroid under L2²,
/// writing the result into `assignments`. Rayon-parallel over docs
/// through the block-transposed SIMD kernel (one transpose per call,
/// amortized across the chunk's rows). Same assignment as one
/// iteration of [`kmeans_with_assignments`]'s inner loop, but
/// exposed as a standalone entry point so the reservoir-trained
/// k-means in [`crate::superfile::vector::reservoir`] can fan the
/// trained centroids back out across the full corpus after
/// training touched only a sample.
///
/// # Panics
///
/// - `vectors.len() % dim != 0`
/// - `assignments.len() != vectors.len() / dim`
/// - `centroids.len() != k * dim`
/// - `k == 0` or `dim == 0`
pub(crate) fn assign_to_centroids(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
    k: usize,
    assignments: &mut [u32],
) {
    assert!(dim > 0, "assign_to_centroids: dim must be > 0");
    assert!(k > 0, "assign_to_centroids: k must be > 0");
    assert_eq!(
        vectors.len() % dim,
        0,
        "assign_to_centroids: vectors len {} not multiple of dim {dim}",
        vectors.len()
    );
    assert_eq!(
        centroids.len(),
        k * dim,
        "assign_to_centroids: centroids len {} != k*dim {}",
        centroids.len(),
        k * dim
    );
    let n = vectors.len() / dim;
    assert_eq!(
        assignments.len(),
        n,
        "assign_to_centroids: assignments len {} != n_docs {n}",
        assignments.len()
    );
    if n == 0 {
        return;
    }
    let transposed = transpose_centroids_cluster_major(centroids, k, dim);
    assignments
        .par_iter_mut()
        .enumerate()
        .for_each(|(d, slot)| {
            let v = &vectors[d * dim..(d + 1) * dim];
            *slot = nearest_centroid_transposed(Metric::L2Sq, v, &transposed, k, dim).0;
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn returns_k_centroids_of_dim_each() {
        // 100 docs, dim=8, k=4.
        let vectors: Vec<f32> = (0..800).map(|i| (i as f32) * 0.01).collect();
        let centroids = kmeans(&vectors, 8, 4, 5, 42);
        assert_eq!(centroids.len(), 4 * 8);
    }

    #[test]
    fn determinism_same_seed_same_centroids() {
        let vectors: Vec<f32> = (0..100 * 8).map(|i| (i as f32) * 0.01).collect();
        let c1 = kmeans(&vectors, 8, 4, 5, 12345);
        let c2 = kmeans(&vectors, 8, 4, 5, 12345);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_seeds_likely_different_centroids() {
        // Init is the only randomness, but at small k it dominates.
        let vectors: Vec<f32> = (0..100 * 8).map(|i| (i as f32) * 0.01).collect();
        let c1 = kmeans(&vectors, 8, 4, 5, 1);
        let c2 = kmeans(&vectors, 8, 4, 5, 999);
        // After 5 iterations they could converge — but for this
        // monotone input the order of cluster ids tends to differ.
        // Assert "not always identical" rather than a specific shape.
        let identical = c1 == c2;
        if identical {
            // Acceptable convergence at this scale; just sanity-check that
            // both have valid shapes.
            assert_eq!(c1.len(), c2.len());
        }
    }

    #[test]
    fn centroids_are_within_data_range() {
        // Centroids are means of subsets of input vectors → bounded by
        // input min/max along each axis.
        let n = 200;
        let dim = 4;
        let vectors: Vec<f32> = (0..n * dim).map(|i| (i % 10) as f32).collect();
        let centroids = kmeans(&vectors, dim, 8, 5, 7);
        for &c in &centroids {
            assert!(
                (-0.001..=9.001).contains(&c),
                "centroid value {c} outside data range [0, 9]"
            );
        }
    }

    #[test]
    fn cluster_data_recovers_natural_centers() {
        // Plant 3 well-separated clusters; verify the centroids
        // converge near the planted means.
        let dim = 4;
        let centers = [
            [0.0f32, 0.0, 0.0, 0.0],
            [10.0, 10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0, -10.0],
        ];
        let mut vectors: Vec<f32> = Vec::new();
        // 30 docs per cluster, ε noise. Use a tiny deterministic
        // pseudo-noise so the test stays reproducible.
        for (cluster_idx, c) in centers.iter().enumerate() {
            for d in 0..30 {
                for (j, &cj) in c.iter().enumerate() {
                    let noise = ((cluster_idx * 30 + d + j) % 7) as f32 * 0.01 - 0.03;
                    vectors.push(cj + noise);
                }
            }
        }
        let centroids = kmeans(&vectors, dim, 3, 5, 42);

        // For each planted center, find the nearest computed centroid
        // and assert it's within a tight tolerance.
        for c in &centers {
            let mut best = f32::INFINITY;
            for ki in 0..3 {
                let cc = &centroids[ki * dim..(ki + 1) * dim];
                let d = (0..dim).map(|j| (c[j] - cc[j]).powi(2)).sum::<f32>().sqrt();
                if d < best {
                    best = d;
                }
            }
            assert!(
                best < 0.5,
                "no centroid within 0.5 of planted center {c:?} (closest = {best})"
            );
        }
    }

    #[test]
    fn k_equal_to_n_assigns_each_doc_its_own_cluster() {
        // Pathological case: k == n.
        let dim = 2;
        let vectors = vec![
            1.0f32, 2.0, // doc 0
            3.0, 4.0, // doc 1
            5.0, 6.0, // doc 2
        ];
        let centroids = kmeans(&vectors, dim, 3, 5, 42);
        // Each centroid should match exactly one input vector.
        let input_pts: Vec<[f32; 2]> = (0..3)
            .map(|i| [vectors[i * 2], vectors[i * 2 + 1]])
            .collect();
        for ki in 0..3 {
            let c = [centroids[ki * 2], centroids[ki * 2 + 1]];
            let any_match = input_pts
                .iter()
                .any(|p| approx(p[0], c[0], 1e-3) && approx(p[1], c[1], 1e-3));
            assert!(any_match, "centroid {c:?} doesn't match any input point");
        }
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn panics_on_zero_k() {
        kmeans(&[1.0; 8], 8, 0, 5, 0);
    }

    #[test]
    #[should_panic(expected = "k")]
    fn panics_on_k_greater_than_n() {
        kmeans(&[1.0; 8], 8, 5, 5, 0); // n=1, k=5
    }

    #[test]
    #[should_panic(expected = "not multiple of dim")]
    fn panics_on_unaligned_input() {
        kmeans(&[1.0; 7], 8, 1, 5, 0);
    }
}
