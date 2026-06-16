// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-column rerank codec.
//!
//! Each vector column picks one codec at build time:
//!
//! - [`RerankCodec::Fp32`]: little-endian fp32, `dim × 4` bytes
//!   per vector. Zero-copy on the rerank distance kernel.
//! - [`RerankCodec::Sq8ResidualEpsilon`]: `Sq8` codes plus a signed
//!   8-bit residual sidecar, `dim × 2` bytes per vector
//!   (row-interleaved `[code dim u8 ‖ residual dim i8]`). The
//!   Sq8 score selects a small final-refine set; the residual
//!   correction is applied only to that set. Default codec.
//! - [`RerankCodec::RabitqOnly`]: no rerank column at all. The
//!   1-bit RaBitQ shortlist is the final ranking — opt-in,
//!   recall-degraded, shrinks the superfile by ~30× at 1M × 384.
//!   Named `RabitqOnly` rather than `None` to (a) avoid shadowing
//!   `Option::None` at every call site and (b) describe the search
//!   behaviour rather than the absence of a codec.
//!
//! ## On-disk discriminator
//!
//! The codec choice rides as a single byte in the per-column
//! subsection-directory entry at offset 52 (bytes 53..55 stay
//! reserved). A zero byte at slot 52 deserializes to
//! [`RerankCodec::Fp32`], so fp32-only superfiles that left the
//! slot zero round-trip identically.
//!
//! ## `codec_meta` region
//!
//! For codecs that need per-column auxiliary data (today:
//! `Sq8ResidualEpsilon`'s scale + offset arrays), the subsection carries a
//! `codec_meta` region between the `codes` region and the
//! `full[]` region. The region's relative offset within the
//! subsection is recorded in sub-header bytes 12..16 as
//! `codec_meta_off: u32`. `Fp32` / `RabitqOnly` superfiles
//! write `codec_meta_off = 0`.

use serde::{Deserialize, Serialize};

use crate::superfile::vector::distance::Metric;

/// `dim` at and below which a column counts as "low-dim" for the
/// rerank-floor calibration table in
/// [`RerankCodec::recommended_rerank_mult_floor`]. Set at 384 to
/// match the dominant embedding-model bucket (e5, MiniLM, etc.).
const LOW_DIM_RERANK_FLOOR_THRESHOLD: usize = 384;

/// Recommended floor on `rerank_mult` for `Fp32` columns at
/// `dim ≤ 384`.
const FP32_LOW_DIM_RERANK_FLOOR: usize = 20;

/// Recommended floor on `rerank_mult` for `Fp32` columns at
/// `dim > 384`. Higher dim widens the gap between the 1-bit
/// shortlist score and the true distance; more candidates are
/// needed to recover the same recall.
const FP32_HIGH_DIM_RERANK_FLOOR: usize = 50;

/// Recommended floor on `rerank_mult` for `Sq8ResidualEpsilon` columns at
/// `dim ≤ 384`. The compressed first-pass score needs more
/// candidates than fp32 to
/// recover equivalent recall because the dequant noise floor is
/// higher.
const SQ8_LOW_DIM_RERANK_FLOOR: usize = 50;

/// Recommended floor on `rerank_mult` for `Sq8ResidualEpsilon` columns at
/// `dim > 384`. See [`SQ8_LOW_DIM_RERANK_FLOOR`] and
/// [`FP32_HIGH_DIM_RERANK_FLOOR`] for the underlying
/// calibration rationale.
const SQ8_HIGH_DIM_RERANK_FLOOR: usize = 100;

/// Per-column rerank codec. Picks the on-disk byte layout of the
/// per-vector rerank values inside the subsection's `full[]`
/// region.
///
/// See the module docs for the on-disk discriminator + lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RerankCodec {
    /// fp32 little-endian, `dim` contiguous f32s per vector.
    /// The rerank distance kernel reads it via
    /// `bytemuck::try_cast_slice` → zero-copy SIMD.
    Fp32,
    /// `Sq8` plus a signed 8-bit residual sidecar. Per-vector
    /// body is `dim` u8 Sq8 codes followed by `dim` i8 residual
    /// codes (residual step = `scale_c[d] / SQ8_RESIDUAL_DIVISOR`).
    /// Search uses the normal Sq8 score to choose a small
    /// final-refine set, then applies the residual correction to
    /// that set — closing the tight top-K cosine recall gap plain
    /// Sq8 exhibits on production-shaped 384D corpora.
    Sq8ResidualEpsilon,
    /// No rerank column at all. The 1-bit RaBitQ shortlist is
    /// the final ranking. Opt-in — recall drops 0.05–0.15 on
    /// typical normalized-Gaussian / image-embedding corpora;
    /// trade-off is a ~30× superfile-size shrink at 1M × 384.
    ///
    /// Spelled `RabitqOnly` rather than `None` so call sites
    /// don't collide with `Option::None` and the variant name
    /// describes the search behaviour rather than the absence
    /// of a codec.
    RabitqOnly,
}

impl Default for RerankCodec {
    /// `Sq8ResidualEpsilon` keeps the compressed Sq8 path as the default
    /// while correcting the tight top-K swaps that plain Sq8
    /// exhibited in the residual-selection diagnostics. Callers that
    /// need bit-exact fp32 (oracles, regression fixtures,
    /// recall-floor reference runs) opt in to [`RerankCodec::Fp32`].
    fn default() -> Self {
        Self::Sq8ResidualEpsilon
    }
}

impl RerankCodec {
    /// On-disk discriminator byte. Lives at offset 52 inside the
    /// 64-byte per-column directory entry. `0` is reserved for
    /// [`Self::Fp32`] so fp32-only superfiles that left the slot
    /// zero round-trip identically.
    #[inline]
    pub const fn codec_id(self) -> u8 {
        match self {
            Self::Fp32 => 0,
            Self::Sq8ResidualEpsilon => 1,
            Self::RabitqOnly => 2,
        }
    }

    /// Inverse of [`Self::codec_id`]. Returns `None` for unknown
    /// discriminator bytes — the reader treats that as a
    /// `MalformedVersion` failure so a corrupted / future superfile
    /// fails loud rather than mis-decoding.
    #[inline]
    pub const fn from_codec_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::Fp32),
            1 => Some(Self::Sq8ResidualEpsilon),
            2 => Some(Self::RabitqOnly),
            _ => None,
        }
    }

    /// Stable human-readable name, used in JSON-config + error
    /// strings.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Sq8ResidualEpsilon => "sq8_residual",
            Self::RabitqOnly => "rabitq_only",
        }
    }

    /// Per-vector body size in bytes inside the `full[]` region.
    /// `0` for [`Self::RabitqOnly`] (no rerank bytes at all).
    #[inline]
    pub const fn per_vector_bytes(self, dim: usize) -> usize {
        match self {
            Self::Fp32 => dim * 4,
            Self::Sq8ResidualEpsilon => dim * 2,
            Self::RabitqOnly => 0,
        }
    }

    /// Whether this codec writes a per-vector `full[]` region
    /// to disk. `false` only for [`Self::RabitqOnly`], which
    /// drops the rerank column entirely. Build + open paths use
    /// this to skip the `full[]` allocation, the per-row spill
    /// in pass 2, and the bucket-read load in pass 3.
    #[inline]
    pub const fn writes_full(self) -> bool {
        !matches!(self, Self::RabitqOnly)
    }

    /// Whether the build + search paths implement this codec.
    /// All enum variants are currently implemented; this
    /// hook exists so future codecs can be added to the enum
    /// (and the on-disk discriminator table) before their build
    /// path lands — call sites use it to fail fast with a
    /// targeted `Unimplemented` error rather than silently
    /// writing a byte format that the reader can't decode.
    #[inline]
    pub const fn is_implemented(self) -> bool {
        matches!(
            self,
            Self::Fp32 | Self::Sq8ResidualEpsilon | Self::RabitqOnly
        )
    }

    /// Recommended **lower bound** on `rerank_mult` for this
    /// codec at the given `dim`. Returns `None` for codecs
    /// where rerank is meaningless (today: just
    /// [`Self::RabitqOnly`], which skips the rerank step
    /// entirely).
    ///
    /// Sq8ResidualEpsilon needs more candidates to recover fp32-equivalent
    /// recall because the first-pass dequant noise floor is higher
    /// than fp32. The bench harness uses this as the calibration-grid
    /// lower bound; direct `search(.., rerank_mult)` callers are
    /// unaffected.
    ///
    /// Numbers calibrated against FAISS-doc peer benchmarks.
    #[inline]
    pub const fn recommended_rerank_mult_floor(self, dim: usize) -> Option<usize> {
        let high_dim = dim > LOW_DIM_RERANK_FLOOR_THRESHOLD;
        match self {
            Self::Fp32 => Some(if high_dim {
                FP32_HIGH_DIM_RERANK_FLOOR
            } else {
                FP32_LOW_DIM_RERANK_FLOOR
            }),
            Self::Sq8ResidualEpsilon => Some(if high_dim {
                SQ8_HIGH_DIM_RERANK_FLOOR
            } else {
                SQ8_LOW_DIM_RERANK_FLOOR
            }),
            Self::RabitqOnly => None,
        }
    }

    /// Returns the per-column `codec_meta` region size in bytes
    /// for this codec at the given dim + n_docs + n_cent + metric.
    /// Stored immediately before the subsection's `full[]` region.
    ///
    /// - `Fp32` / `RabitqOnly`: `0` (no codec metadata).
    /// - `Sq8ResidualEpsilon`: **per-cluster** per-dim `(scale, offset)` arrays
    ///   (`2 × n_cent × dim × 4` bytes) plus, for `L2Sq`/`Cosine`-metric
    ///   columns, a per-doc `sum_x_decoded² : f32` table
    ///   (`n_docs × 4` bytes) used to short-circuit the `Σx²`
    ///   term in the L2Sq distance formula or normalize the decoded
    ///   vector for Cosine at search time. NegDot columns drop the
    ///   per-doc norms.
    ///
    /// **Why per-cluster, not per-column.** A naive design uses
    /// one `(scale[dim], offset[dim])` pair for the whole
    /// column. On highly clustered cosine corpora (real sentence
    /// embeddings, the bench's planted-cluster generator) the
    /// per-column min/max spans the cross-cluster spread — but the
    /// rerank step's ranking signal lives in the *intra-cluster*
    /// spread, which is several times narrower. With 256 buckets
    /// stretched across the wider global range, only a small slice
    /// of them is used within any one cluster; the quantization
    /// noise dominates intra-cluster cosine differences and recall
    /// collapses (the planted-cluster diagnostic in `reader.rs`
    /// reproduces the failure mode at small scale). Per-cluster
    /// quantizer recovers full recall by giving each cluster's docs
    /// the finest possible buckets over their local range. Cost is
    /// `n_cent × dim × 8` codec_meta bytes — small relative to
    /// the Sq8 `full[]` region at typical IVF shapes.
    #[inline]
    pub const fn codec_meta_bytes(
        self,
        dim: usize,
        n_docs: usize,
        n_cent: usize,
        metric: Metric,
    ) -> usize {
        match self {
            Self::Fp32 | Self::RabitqOnly => 0,
            Self::Sq8ResidualEpsilon => {
                let scale_offset_bytes = 2 * n_cent * dim * 4;
                let norms_bytes = match metric {
                    Metric::L2Sq | Metric::Cosine => n_docs * 4,
                    Metric::NegDot => 0,
                };
                scale_offset_bytes + norms_bytes
            }
        }
    }
}

impl std::fmt::Display for RerankCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default codec is `Sq8ResidualEpsilon`. Any change here is a
    /// load-bearing format choice — every caller that uses
    /// `RerankCodec::default()` silently follows this pick, so
    /// the test pins the contract.
    #[test]
    fn default_is_sq8_residual() {
        assert_eq!(RerankCodec::default(), RerankCodec::Sq8ResidualEpsilon);
    }

    /// `Fp32`'s codec_id is zero. Older superfiles have all-zero
    /// reserved bytes in the directory-entry slot we squat on
    /// for the codec discriminator; the zero match keeps them
    /// readable as `Fp32` without a format bump.
    #[test]
    fn fp32_codec_id_is_zero() {
        assert_eq!(RerankCodec::Fp32.codec_id(), 0u8);
    }

    /// Round-trip every defined variant through `codec_id` /
    /// `from_codec_id`. Catches accidental enum reordering — the
    /// discriminator is on-disk so the numeric mapping is part of
    /// the format contract.
    #[test]
    fn codec_id_roundtrips_every_variant() {
        for c in [
            RerankCodec::Fp32,
            RerankCodec::Sq8ResidualEpsilon,
            RerankCodec::RabitqOnly,
        ] {
            assert_eq!(
                RerankCodec::from_codec_id(c.codec_id()),
                Some(c),
                "round-trip mismatch for {c:?}"
            );
        }
    }

    /// Unknown discriminator bytes (any value not currently
    /// assigned, e.g. `5`, `255`) return `None`. The reader
    /// upgrades that into a `MalformedVersion` error rather than
    /// guessing.
    #[test]
    fn unknown_codec_id_is_none() {
        for id in [3u8, 4, 5, 16, 200, 255] {
            assert_eq!(
                RerankCodec::from_codec_id(id),
                None,
                "unknown id {id} must not map to a codec"
            );
        }
    }

    /// Per-vector body sizes match the on-disk spec. `RabitqOnly`'s
    /// zero is what lets that codec drop the entire `full[]`
    /// region.
    #[test]
    fn per_vector_bytes_matches_spec() {
        assert_eq!(RerankCodec::Fp32.per_vector_bytes(384), 1536);
        assert_eq!(RerankCodec::Sq8ResidualEpsilon.per_vector_bytes(384), 768);
        assert_eq!(RerankCodec::RabitqOnly.per_vector_bytes(384), 0);
    }

    /// `writes_full` is the inverse of "this codec is
    /// `RabitqOnly`" — pins the build/open fast-path predicate
    /// to the codec's identity rather than scattered
    /// `matches!(_, RabitqOnly)` checks.
    #[test]
    fn writes_full_matches_per_vector_bytes() {
        for c in [
            RerankCodec::Fp32,
            RerankCodec::Sq8ResidualEpsilon,
            RerankCodec::RabitqOnly,
        ] {
            assert_eq!(
                c.writes_full(),
                c.per_vector_bytes(384) > 0,
                "writes_full disagrees with per_vector_bytes for {c:?}"
            );
        }
    }

    /// All three codecs are wired end-to-end (build + open + search).
    #[test]
    fn all_codecs_implemented() {
        assert!(RerankCodec::Fp32.is_implemented());
        assert!(RerankCodec::Sq8ResidualEpsilon.is_implemented());
        assert!(RerankCodec::RabitqOnly.is_implemented());
    }

    /// Calibration-floor table the bench harness threads into
    /// its calibration grid. The hard contract is the values +
    /// the `None`-returns-`None` behaviour; the dim split
    /// (`> 384`) is one of two load-bearing knobs the bench
    /// harness reads.
    #[test]
    fn recommended_rerank_mult_floor_matches_calibration_table() {
        // dim ≤ 384 column.
        assert_eq!(
            RerankCodec::Fp32.recommended_rerank_mult_floor(384),
            Some(20)
        );
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.recommended_rerank_mult_floor(384),
            Some(50)
        );
        assert_eq!(
            RerankCodec::RabitqOnly.recommended_rerank_mult_floor(384),
            None
        );
        // 384 < dim ≤ 1024 column.
        assert_eq!(
            RerankCodec::Fp32.recommended_rerank_mult_floor(1024),
            Some(50)
        );
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.recommended_rerank_mult_floor(1024),
            Some(100)
        );
        assert_eq!(
            RerankCodec::RabitqOnly.recommended_rerank_mult_floor(1024),
            None
        );
        // Split point: dim == 384 is the low-dim cell; dim == 385
        // crosses into high-dim.
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.recommended_rerank_mult_floor(385),
            Some(100)
        );
    }

    /// Sq8ResidualEpsilon's codec_meta size: `8·n_cent·dim` for negdot,
    /// `8·n_cent·dim + 4·n_docs` for L2Sq/Cosine (per-doc decoded-norm
    /// cache). Fp32 / RabitqOnly always contribute zero
    /// bytes. Per-cluster scale/offset is the recall-recovery
    /// fix landed in the Sq8PerCluster patch (see fn-doc above).
    #[test]
    fn codec_meta_bytes_matches_layout_spec() {
        // Fp32 + RabitqOnly never carry codec_meta.
        for c in [RerankCodec::Fp32, RerankCodec::RabitqOnly] {
            for m in [Metric::L2Sq, Metric::Cosine, Metric::NegDot] {
                assert_eq!(
                    c.codec_meta_bytes(384, 1_000_000, 1024, m),
                    0,
                    "{c:?} / {m:?}"
                );
            }
        }
        // Sq8ResidualEpsilon negdot: per-cluster scale + offset arrays.
        let so_bytes = 2 * 1024 * 384 * 4;
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.codec_meta_bytes(384, 1_000_000, 1024, Metric::NegDot),
            so_bytes
        );
        // Sq8ResidualEpsilon L2Sq/Cosine: per-cluster scale + offset + per-doc norms.
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.codec_meta_bytes(384, 1_000_000, 1024, Metric::Cosine),
            so_bytes + 1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.codec_meta_bytes(384, 1_000_000, 1024, Metric::L2Sq),
            so_bytes + 1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq8ResidualEpsilon.codec_meta_bytes(384, 1_000_000, 1024, Metric::NegDot),
            so_bytes
        );
    }
}
