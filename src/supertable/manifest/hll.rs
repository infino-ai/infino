// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Minimal HyperLogLog sketch for the manifest's per-column
//! distinct-count estimates.
//!
//! Carried per scalar column per segment (and merged into part-level
//! rollups), consumed by the SQL planner as a cardinality *estimate*
//! (`ColumnStatistics::distinct_count`, always reported Inexact) —
//! never for user-visible results. In-tree implementation over the
//! crate's existing `xxhash_rust` dependency; the standard published
//! HLL crates would add a dependency for ~a hundred lines of code.
//!
//! Shape: `2^10 = 1024` one-byte registers (1 KiB per sketch,
//! ~±3.25% standard error). Merging is register-wise max, so part
//! rollups and cross-segment folds are exact unions of the sketches.

/// Width of the hash the sketch indexes over (`xxh3_64`).
const HASH_BITS: u32 = 64;
/// log2 of the register count.
const HLL_P: u32 = 10;
/// Register count `m = 2^P`.
pub const HLL_REGISTERS: usize = 1 << HLL_P;
/// Standard HLL bias-correction constant `α_m` for m ≥ 128.
const HLL_ALPHA: f64 = 0.7213 / (1.0 + 1.079 / (HLL_REGISTERS as f64));
/// Small-range correction threshold: below `2.5 m` the raw estimator
/// is biased and linear counting over empty registers is used.
const SMALL_RANGE_FACTOR: f64 = 2.5;

/// A fixed-size HyperLogLog sketch.
#[derive(Clone, PartialEq, Eq)]
pub struct HllSketch {
    registers: Box<[u8]>,
}

impl Default for HllSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HllSketch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HllSketch")
            .field("estimate", &(self.estimate() as u64))
            .finish()
    }
}

impl HllSketch {
    pub fn new() -> Self {
        Self {
            registers: vec![0u8; HLL_REGISTERS].into_boxed_slice(),
        }
    }

    /// Insert a pre-hashed value (callers hash with `xxh3_64`).
    pub fn insert_hash(&mut self, hash: u64) {
        // High P bits pick the register; the rank is the position of
        // the first set bit in the remaining 64-P bits (capped so an
        // all-zero remainder still yields a valid rank).
        let idx = (hash >> (HASH_BITS - HLL_P)) as usize;
        let rest = hash << HLL_P;
        // Rank = leading-zero run of the remainder bits + 1, capped so
        // an all-zero remainder still yields a valid value (max
        // `HASH_BITS - HLL_P + 1`).
        let rank = (rest.leading_zeros().min(HASH_BITS - HLL_P) + 1) as u8;
        if rank > self.registers[idx] {
            self.registers[idx] = rank;
        }
    }

    /// Distinct-count estimate (standard HLL with the small-range
    /// linear-counting correction).
    pub fn estimate(&self) -> f64 {
        let m = HLL_REGISTERS as f64;
        // Harmonic-mean term per register: 2^-rank. `powi` avoids the
        // `1 << rank` integer-shift overflow (rank reaches MAX_RANK,
        // far past a u32/u64 shift's valid range).
        let sum: f64 = self
            .registers
            .iter()
            .map(|&r| 2.0_f64.powi(-i32::from(r)))
            .sum();
        let raw = HLL_ALPHA * m * m / sum;
        if raw <= SMALL_RANGE_FACTOR * m {
            let zeros = self.registers.iter().filter(|&&r| r == 0).count();
            if zeros > 0 {
                return m * (m / zeros as f64).ln();
            }
        }
        raw
    }

    /// Union with `other` (register-wise max) — the merged sketch is
    /// exactly the sketch of the union of the inserted sets.
    pub fn merge(&mut self, other: &Self) {
        for (a, b) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *b > *a {
                *a = *b;
            }
        }
    }

    /// Raw register bytes (always [`HLL_REGISTERS`] long).
    pub fn as_bytes(&self) -> &[u8] {
        &self.registers
    }

    /// Rebuild from raw register bytes; `None` on a length mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != HLL_REGISTERS {
            return None;
        }
        Some(Self {
            registers: bytes.to_vec().into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Distinct values inserted by the accuracy test.
    const TEST_DISTINCT: usize = 50_000;
    /// Loose relative-error bound: ~6 standard errors at m=1024 so the
    /// test never flakes on an unlucky hash draw.
    const TEST_MAX_REL_ERR: f64 = 0.20;

    #[test]
    fn estimate_tracks_distinct_count() {
        let mut sketch = HllSketch::new();
        for i in 0..TEST_DISTINCT as u64 {
            // Duplicate inserts must not move the estimate.
            for _ in 0..2 {
                sketch.insert_hash(xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes()));
            }
        }
        let est = sketch.estimate();
        let rel = (est - TEST_DISTINCT as f64).abs() / TEST_DISTINCT as f64;
        assert!(
            rel < TEST_MAX_REL_ERR,
            "estimate {est} vs {TEST_DISTINCT} (rel err {rel:.3})"
        );
    }

    #[test]
    fn merge_is_union() {
        let mut a = HllSketch::new();
        let mut b = HllSketch::new();
        let mut both = HllSketch::new();
        for i in 0..10_000u64 {
            let h = xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes());
            if i % 2 == 0 {
                a.insert_hash(h);
            } else {
                b.insert_hash(h);
            }
            both.insert_hash(h);
        }
        a.merge(&b);
        assert_eq!(a, both, "merged sketch must equal the union sketch");
    }

    #[test]
    fn bytes_round_trip() {
        let mut sketch = HllSketch::new();
        for i in 0..1000u64 {
            sketch.insert_hash(xxhash_rust::xxh3::xxh3_64(&i.to_le_bytes()));
        }
        let restored = HllSketch::from_bytes(sketch.as_bytes()).expect("valid length");
        assert_eq!(restored, sketch);
        assert!(HllSketch::from_bytes(&[0u8; 3]).is_none());
    }
}
