// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Reservoir sampling for k-means training in the streaming
//! `VectorBuilder`.
//!
//! The build path is k-means-bound, not corpus-bound — the
//! information k-means needs to converge is a representative
//! sample of the corpus, not every vector. Drawing a bounded
//! uniform-random sample (typically `64 × n_cent`, capped at
//! 500 K vectors) and training on that produces centroids that
//! are recall-indistinguishable from training on the full
//! corpus, while making memory consumption a function of
//! `(n_cent, dim)` rather than `(n_docs, dim)`.
//!
//! ## Algorithm
//!
//! Vitter Algorithm L (1985, "Random Sampling with a Reservoir"):
//! O(k(1 + log(n/k))) total work, O(log n) amortized per
//! `update` call once the reservoir is full. Algorithm L wins
//! over Algorithm R (O(1) per add but n Bernoulli draws) at
//! large n/k ratios — at 10M docs with k=100K, the per-call
//! work is ~7× less.
//!
//! Sketch:
//!
//! 1. Fill the reservoir with the first `k` items.
//! 2. Compute `W = U^(1/k)` where `U ~ Uniform(0, 1)`.
//! 3. Compute a skip distance `S = floor(ln(U') / ln(1 − W))`;
//!    that many items pass without touching the reservoir.
//! 4. Replace a uniformly random slot with the item at index
//!    `i + S + 1`.
//! 5. Update `W = W × U''^(1/k)`. Repeat from (3).
//!
//! The replacement slot is uniform over `[0, k)`; the skip
//! distance follows a geometric distribution whose mean grows
//! as the stream advances, which is exactly what makes the
//! per-item amortized cost drop to O(log n).
//!
//! ## Determinism
//!
//! Seeded from a caller-supplied `u64` so identical builds
//! produce identical reservoirs (and therefore identical
//! centroids, since k-means itself is deterministic given
//! initialization). `StdRng` rather than `ChaCha8Rng` — both
//! are seedable + deterministic + reproducible across `rand`
//! patch versions, and `StdRng` is already a transitive dep
//! via `rand 0.10` so we avoid pulling in `rand_chacha`.

use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::config;

/// Multiplier on a column's IVF centroid count to size its k-means
/// training sample. Slightly above the FAISS-empirical 30–60× sweet
/// spot for IVF training, picked for recall headroom.
const KMEANS_SAMPLE_NCENT_MULT: usize = 64;

/// Floor on the k-means training sample size, so builds with a small
/// `n_cent` still see enough corpus variance to converge.
const KMEANS_SAMPLE_SIZE_FLOOR: usize = 100_000;

/// Cap on the k-means training sample size. Bounds reservoir memory
/// (≈ 730 MB at dim=384); past this the recall gain is below gate
/// noise.
const KMEANS_SAMPLE_SIZE_CAP: usize = 500_000;

/// Default k-means training sample size for a column with
/// `n_cent` IVF centroids:
///
/// ```text
///   sample_size = max(100_000, min(500_000, 64 × n_cent))
/// ```
///
/// `64 × n_cent` is slightly above the FAISS-empirical sweet
/// spot of 30–60 × n_cent for IVF training, with a 100 K floor
/// so small-n_cent builds still see enough variance to converge
/// and a 500 K cap so the reservoir never gets pathological at
/// large n_cent. The cap saturates at `n_cent = 7812`; above
/// that the sample stays at 500 K (≈ 730 MB at dim=384) and the
/// gain from more training data is well below recall-gate noise.
pub fn default_kmeans_sample_size(n_cent: usize) -> usize {
    let target = KMEANS_SAMPLE_NCENT_MULT.saturating_mul(n_cent);
    target.clamp(KMEANS_SAMPLE_SIZE_FLOOR, KMEANS_SAMPLE_SIZE_CAP)
}

/// Partition row count past which the per-cell training sample switches
/// from points-per-centroid to a row fraction. At the 10M design point a
/// hidden cell holds ~39K rows and `64 × n_cent` (~6.7% of the cell)
/// trains balanced fine runs (measured p50 566 rows vs ~950 target). At
/// 100M cell sizes (~98K rows) the same flat sample collapses the
/// run-size distribution (measured p50 69 / max 23,918) and post-drain
/// recall with it (0.199). Below this threshold the flat sample is both
/// sufficient and *better* — boosting it at 1M-scale cells measured a
/// 0.02 recall drop (0.995 → 0.975). Deliberately LOWER than the
/// builder's cap/split boundary (`CONSOLIDATED_CELL_ROWS_THRESHOLD` in
/// `builder.rs`): the 2-GET 10M baseline was measured with this boost
/// active on its biggest cells (max 65,918 rows), so the boost is part
/// of the known-good layout there while the cap/split must stay out.
const PARTITION_SAMPLE_ROW_FRACTION_THRESHOLD_ROWS: usize = 65_536;

/// Row fraction of an oversized partition the training sample must cover
/// once past [`PARTITION_SAMPLE_ROW_FRACTION_THRESHOLD_ROWS`]: a quarter
/// of the rows, keeping the trained shape scale-invariant as cells grow.
const PARTITION_SAMPLE_ROW_FRACTION_DIVISOR: usize = 4;

/// K-means training sample for a **per-partition sub-build** (the drain's
/// per-cell IVF build): *points per centroid* (`mult × n_cent`), floored
/// at one `mult` for tiny `n_cent` and sharing the upper cap; the caller
/// bounds it by the row count (`.min(n_docs)`). Partitions past
/// [`PARTITION_SAMPLE_ROW_FRACTION_THRESHOLD_ROWS`] raise the sample to a
/// quarter of their rows so the fine-run distribution stays balanced at
/// 100M-scale cell sizes (see the threshold's doc for both measurements).
///
/// Unlike [`default_kmeans_sample_size`] there is no absolute
/// representativeness floor — that floor suits the global build over the
/// whole corpus, whereas a per-cell sub-build trains sub-centroids over
/// one cell. `mult` is sourced from `vector.kmeans_pts_per_centroid`
/// (falling back to the built-in default when it is set to zero).
pub fn partition_kmeans_sample_size(n_cent: usize, n_rows: usize) -> usize {
    let configured = config::global().vector.kmeans_pts_per_centroid;
    let mult = if configured > 0 {
        configured
    } else {
        KMEANS_SAMPLE_NCENT_MULT
    };
    let per_centroid = mult.saturating_mul(n_cent);
    let sample = if n_rows > PARTITION_SAMPLE_ROW_FRACTION_THRESHOLD_ROWS {
        per_centroid.max(n_rows / PARTITION_SAMPLE_ROW_FRACTION_DIVISOR)
    } else {
        per_centroid
    };
    sample.clamp(mult, KMEANS_SAMPLE_SIZE_CAP)
}

/// Online reservoir for f32 vector samples.
///
/// One instance per logical vector index in `VectorBuilder`. Holds at
/// most `sample_size` vectors as contiguous f32s in a single
/// `Vec<f32>` of capacity `sample_size × dim`. Update cost is
/// O(1) while the reservoir is filling and O(1) amortized once
/// it is full (the underlying probability of a replacement on
/// any given item is `sample_size / n_seen`, which drops quickly
/// past the fill phase).
///
/// The struct owns its RNG state so callers don't need to
/// thread one through; seeding is done at construction.
pub(crate) struct Reservoir {
    sample_size: usize,
    dim: usize,
    rng: StdRng,
    /// Flat reservoir buffer. Length is either
    /// `n_seen * dim` (during fill, `n_seen < sample_size`) or
    /// `sample_size * dim` (after fill).
    buf: Vec<f32>,
    /// Total vectors observed via [`Self::update`], including
    /// those that were accepted into the reservoir and those
    /// that were skipped.
    n_seen: u64,
    /// Algorithm L's `W` accumulator. `0.0` until the reservoir
    /// is full; thereafter strictly in `(0, 1)`.
    w: f64,
    /// Index (0-based) of the next item that will trigger a
    /// replacement. `u64::MAX` until the reservoir is full.
    next_replace_at: u64,
}

impl Reservoir {
    /// Create a fresh reservoir of capacity `sample_size`
    /// vectors of dimension `dim`, seeded from `seed`. Both
    /// dimensions are stored so [`Self::update`] can validate
    /// each incoming vector and so [`Self::sample`] knows how
    /// to slice the buffer.
    ///
    /// `sample_size == 0` is rejected; callers should always
    /// derive sample size from [`default_kmeans_sample_size`].
    pub fn new(sample_size: usize, dim: usize, seed: u64) -> Self {
        assert!(sample_size > 0, "Reservoir: sample_size must be > 0");
        assert!(dim > 0, "Reservoir: dim must be > 0");
        Self {
            sample_size,
            dim,
            rng: StdRng::seed_from_u64(seed),
            buf: Vec::with_capacity(sample_size * dim),
            n_seen: 0,
            w: 0.0,
            next_replace_at: u64::MAX,
        }
    }

    /// Observe one vector. The vector is either appended (during
    /// the fill phase, `n_seen < sample_size`) or evaluated by
    /// the Vitter Algorithm L skip counter (after fill) — at
    /// most one comparison and, with probability `≤ 1/n_seen`,
    /// one `copy_from_slice` of `dim × 4` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `vec.len() != self.dim`.
    pub fn update(&mut self, vec: &[f32]) {
        assert_eq!(
            vec.len(),
            self.dim,
            "Reservoir::update: vec.len() {} != dim {}",
            vec.len(),
            self.dim
        );
        let k = self.sample_size as u64;
        let i = self.n_seen;
        self.n_seen += 1;

        if i < k {
            // Fill phase.
            self.buf.extend_from_slice(vec);
            if self.n_seen == k {
                // Reservoir just filled — seed the skip
                // counter from W = U^(1/k).
                self.w = (Self::nonzero_uniform(&mut self.rng).ln() / k as f64).exp();
                self.next_replace_at = i + 1 + Self::skip(&mut self.rng, self.w);
            }
            return;
        }

        // Full phase. Replace at the precomputed skip
        // boundary; otherwise this item passes by untouched.
        if i == self.next_replace_at {
            let slot = self.rng.random_range(0..self.sample_size);
            self.buf[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(vec);
            self.w *= (Self::nonzero_uniform(&mut self.rng).ln() / k as f64).exp();
            self.next_replace_at = i + 1 + Self::skip(&mut self.rng, self.w);
        }
    }

    /// Number of vectors observed via [`Self::update`].
    #[cfg(test)]
    pub(crate) fn n_seen(&self) -> u64 {
        self.n_seen
    }

    /// Current reservoir contents as a contiguous `&[f32]` of
    /// length `min(n_seen, sample_size) × dim`. Caller passes
    /// this directly to k-means training.
    pub fn sample(&self) -> &[f32] {
        &self.buf
    }

    /// Same as [`Self::sample`] but consumes the reservoir,
    /// handing back the owned buffer.
    #[cfg(test)]
    pub(crate) fn into_sample(self) -> Vec<f32> {
        self.buf
    }

    /// Number of rows actually held in the reservoir
    /// (`min(n_seen, sample_size)`). Useful for tests + the
    /// degenerate-tiny-corpus case where the reservoir never
    /// fully fills.
    pub fn n_rows(&self) -> usize {
        (self.n_seen as usize).min(self.sample_size)
    }

    /// Algorithm L's skip distance: `floor(ln(U) / ln(1 − W))`.
    /// `U` is drawn from `(0, 1]` to keep `ln(U)` finite;
    /// `1 − W` is in `(0, 1)` so the log is negative and the
    /// quotient is non-negative.
    fn skip(rng: &mut StdRng, w: f64) -> u64 {
        let u = Self::nonzero_uniform(rng);
        let denom = (1.0 - w).ln();
        // `denom` is strictly negative for w ∈ (0, 1). A
        // pathological w == 1.0 (mathematically unreachable)
        // would produce `-inf`, giving a `0.0 / -inf = 0` skip,
        // which degrades to "replace every item" — non-fatal,
        // just defeats the optimization. Guard explicitly.
        if !denom.is_finite() || denom == 0.0 {
            return 0;
        }
        (u.ln() / denom).floor().max(0.0) as u64
    }

    /// Draw a uniform `(0, 1]` sample. `rand`'s
    /// `random::<f64>()` returns `[0, 1)` for the standard
    /// distribution; we shift to `(0, 1]` so `ln()` is finite.
    fn nonzero_uniform(rng: &mut StdRng) -> f64 {
        1.0 - rng.random::<f64>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(reservoir_size: usize, dim: usize, n: u64, seed: u64) -> Reservoir {
        let mut r = Reservoir::new(reservoir_size, dim, seed);
        for i in 0..n {
            // Each row encodes its source index in the first
            // f32 — the rest is filler. Lets tests recover
            // "which source items survived in the reservoir".
            let mut row = vec![0.0f32; dim];
            row[0] = i as f32;
            r.update(&row);
        }
        r
    }

    /// The row-fraction floor engages only past the oversized-partition
    /// threshold: 10M-scale cells keep the flat per-centroid sample
    /// (boosting them measured a 1M recall drop), 100M-scale cells train
    /// on a quarter of their rows, and the cap still binds.
    #[test]
    fn partition_sample_row_fraction_gated_by_threshold() {
        // 10M design point: ~39K-row cell stays on the flat sample.
        assert_eq!(partition_kmeans_sample_size(41, 39_062), 41 * 64);
        // 1M shape: ~3.9K-row cell, flat sample, floored at one mult.
        assert_eq!(partition_kmeans_sample_size(4, 3_906), 4 * 64);
        // 100M shape: ~98K-row cell crosses the threshold — a quarter of
        // the rows beats 64×92.
        assert_eq!(partition_kmeans_sample_size(92, 97_656), 97_656 / 4);
        // Just under the threshold: still flat.
        assert_eq!(partition_kmeans_sample_size(64, 65_536), 64 * 64);
        // Enormous partitions still cap.
        assert_eq!(
            partition_kmeans_sample_size(4_096, 40_000_000),
            KMEANS_SAMPLE_SIZE_CAP
        );
    }

    #[test]
    fn fill_phase_appends_exactly_n_seen_rows() {
        let dim = 4;
        let mut r = Reservoir::new(10, dim, 1);
        for i in 0..5 {
            let mut row = vec![0.0f32; dim];
            row[0] = i as f32;
            r.update(&row);
            assert_eq!(r.n_rows(), (i + 1) as usize);
            assert_eq!(r.sample().len(), (i + 1) as usize * dim);
        }
    }

    #[test]
    fn at_fill_boundary_buffer_holds_first_k_rows_in_order() {
        let dim = 3;
        let r = run(/*sample=*/ 5, dim, /*n=*/ 5, /*seed=*/ 7);
        let s = r.sample();
        assert_eq!(s.len(), 5 * dim);
        for i in 0..5 {
            assert_eq!(
                s[i * dim],
                i as f32,
                "fill phase didn't preserve insertion order"
            );
        }
    }

    #[test]
    fn n_seen_counts_every_update_regardless_of_acceptance() {
        let r = run(
            /*sample=*/ 10, /*dim=*/ 2, /*n=*/ 1000, /*seed=*/ 42,
        );
        assert_eq!(r.n_seen(), 1000);
        assert_eq!(r.n_rows(), 10);
    }

    #[test]
    fn determinism_same_seed_same_reservoir() {
        let a = run(50, 4, 10_000, 12345);
        let b = run(50, 4, 10_000, 12345);
        assert_eq!(a.sample(), b.sample());
    }

    #[test]
    fn different_seeds_yield_different_reservoirs() {
        let a = run(50, 4, 10_000, 1);
        let b = run(50, 4, 10_000, 2);
        // Same-seed equality is guaranteed; different-seed
        // inequality is overwhelmingly likely (probability of
        // collision ≈ (50/10000)^50, astronomically small).
        assert_ne!(
            a.sample(),
            b.sample(),
            "two seeds yielded identical reservoirs"
        );
    }

    /// Each source item should appear in the reservoir with
    /// probability `sample_size / n` (k/n). At sample_size=100,
    /// n=10_000, the expected count per item is 0.01; aggregated
    /// across many trials we can check uniformity.
    ///
    /// This is the actual recall-correctness gate for the
    /// sampler: if the distribution isn't uniform, k-means
    /// training will be biased toward whichever subset of the
    /// stream the reservoir over-represents.
    #[test]
    fn distribution_is_approximately_uniform_across_seeds() {
        let n = 1000usize;
        let sample_size = 100usize;
        let trials = 200usize;
        let dim = 1;
        let mut counts = vec![0u64; n];
        for trial in 0..trials {
            let r = run(sample_size, dim, n as u64, trial as u64 + 1);
            let s = r.sample();
            assert_eq!(s.len(), sample_size * dim);
            for row in 0..sample_size {
                let idx = s[row * dim] as usize;
                assert!(idx < n, "reservoir held out-of-range item {idx}");
                counts[idx] += 1;
            }
        }
        // Expected count per item: trials * (sample_size / n) =
        // 200 * 0.1 = 20.0. With n=1000 trials=200 sample=100,
        // counts should cluster around 20. Allow a wide
        // tolerance — we're checking gross bias, not tightness.
        let total: u64 = counts.iter().sum();
        let expected_total = (trials * sample_size) as u64;
        assert_eq!(total, expected_total, "expected exact total");
        let mean = expected_total as f64 / n as f64;
        let max = *counts.iter().max().expect("counts non-empty") as f64;
        let min = *counts.iter().min().expect("counts non-empty") as f64;
        // For a binomial with p=0.1 over 200 trials, sigma ≈
        // sqrt(200 * 0.1 * 0.9) ≈ 4.24. ±5σ from the mean is
        // ≈ [0, 41]; check max - min stays well inside ±20
        // from mean for any single seed-sweep run. If this
        // fires we have a real bias bug, not a flake.
        assert!(
            (max - mean).abs() < 20.0 && (mean - min).abs() < 20.0,
            "non-uniform sampling: mean={mean:.2}, min={min}, max={max} \
             (trial={trials}, n={n}, sample_size={sample_size})"
        );
    }

    #[test]
    fn handles_n_smaller_than_sample_size() {
        // 5 items into a reservoir of 100 → final reservoir holds
        // all 5 in order.
        let dim = 2;
        let mut r = Reservoir::new(100, dim, 999);
        for i in 0..5u32 {
            let mut row = vec![0.0f32; dim];
            row[0] = i as f32;
            r.update(&row);
        }
        assert_eq!(r.n_seen(), 5);
        assert_eq!(r.n_rows(), 5);
        let s = r.sample();
        assert_eq!(s.len(), 5 * dim);
        for i in 0..5 {
            assert_eq!(s[i * dim], i as f32);
        }
    }

    #[test]
    fn handles_n_equal_to_sample_size() {
        let r = run(
            /*sample=*/ 50, /*dim=*/ 3, /*n=*/ 50, /*seed=*/ 7,
        );
        assert_eq!(r.n_seen(), 50);
        assert_eq!(r.n_rows(), 50);
        let s = r.sample();
        // First 50 items, in order.
        for i in 0..50 {
            assert_eq!(s[i * 3], i as f32, "expected pure fill phase");
        }
    }

    #[test]
    fn into_sample_consumes_reservoir() {
        let r = run(10, 4, 10_000, 1);
        let owned = r.into_sample();
        assert_eq!(owned.len(), 10 * 4);
    }

    #[test]
    fn default_sample_size_clamps() {
        // Floor: small n_cent saturates at 100K.
        assert_eq!(default_kmeans_sample_size(0), 100_000);
        assert_eq!(default_kmeans_sample_size(64), 100_000);
        assert_eq!(default_kmeans_sample_size(1_000), 100_000);
        // Mid: 64 × n_cent in band.
        assert_eq!(default_kmeans_sample_size(2_000), 128_000);
        assert_eq!(default_kmeans_sample_size(4_096), 4_096 * 64);
        // Cap: large n_cent saturates at 500K.
        assert_eq!(default_kmeans_sample_size(8_192), 500_000);
        assert_eq!(default_kmeans_sample_size(16_384), 500_000);
        // Overflow guard: usize::MAX × 64 must not panic.
        assert_eq!(default_kmeans_sample_size(usize::MAX), 500_000);
    }

    /// Re-asserting the well-known property that for a stream
    /// shorter than the reservoir, every item must end up in
    /// the sample — uniformity is trivially satisfied since
    /// every item is accepted with probability 1.
    #[test]
    fn every_item_present_when_stream_shorter_than_sample() {
        let dim = 1;
        let r = run(/*sample=*/ 100, dim, /*n=*/ 30, /*seed=*/ 1);
        let s = r.sample();
        let mut indices: Vec<u32> = s.chunks(dim).map(|row| row[0] as u32).collect();
        indices.sort();
        assert_eq!(indices, (0..30).collect::<Vec<_>>());
    }
}
