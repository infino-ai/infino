// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-vector-index rerank codec.
//!
//! Each vector index picks one codec at build time:
//!
//! - [`RerankCodec::Fp32`]: little-endian fp32, `dim × 4` bytes
//!   per vector. Zero-copy on the rerank distance kernel.
//! - [`RerankCodec::Sq8Residual`]: `Sq8` codes plus a signed
//!   8-bit residual sidecar, `dim × 2` bytes per vector
//!   (row-interleaved `[code dim u8 ‖ residual dim i8]`). Both bytes
//!   score every RaBitQ shortlist survivor.
//! - [`RerankCodec::Sq8FixedResidual`]: the same two-byte layout on a
//!   fixed cosine-only grid (`offset=-1`, `scale=2/255`, residual
//!   divisor `256`). The payload is portable across cluster changes.
//! - [`RerankCodec::Sq16`]: a flat uniform 16-bit scalar quantizer on
//!   a fixed cosine-only grid (`offset=-1`, `scale=2/65535`). Stores
//!   one little-endian `u16` code per dimension (`dim × 2` bytes per
//!   vector — the same footprint as `Sq8FixedResidual`'s
//!   `[u8 code ‖ i8 residual]`, but a single plane scored in one pass
//!   instead of two). No residual plane. Because the grid is fixed
//!   constants, `Sq16` needs no per-cluster scale/offset arrays; its
//!   only `codec_meta` is the per-doc dequantized-norm table
//!   (`n_docs × 4` bytes for cosine/L2Sq), matching the Sq8 family so
//!   the cosine kernel divides by `‖d̂‖`. The default cosine codec:
//!   same footprint as the split-plane `Sq8FixedResidual` but scored in
//!   one pass, and leaner on disk (norms only, no fixed scale/offset
//!   arrays).
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
//! For codecs that need per-index auxiliary data (today:
//! `Sq8Residual`'s scale + offset arrays), the subsection carries a
//! `codec_meta` region between the `codes` region and the
//! `full[]` region. The region's relative offset within the
//! subsection is recorded in sub-header bytes 12..16 as
//! `codec_meta_off: u32`. `Fp32` / `RabitqOnly` superfiles
//! write `codec_meta_off = 0`.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::superfile::{
    BuildError,
    vector::{
        cell_posting::{
            EncodedCellRow, note_transcode_clamped_components,
            residual_family_materialize_into_cluster_quant,
        },
        distance::{
            Metric, SQ8_RESIDUAL_DIVISOR, dequantize_sq8_residual_into,
            dequantize_sq16_adaptive_into, dequantize_sq16_into, encode_sq16_adaptive_row,
            sq8_residual_norm_sq, sq16_adaptive_norm_sq, sq16_decoded_norm_sq,
        },
    },
};

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

/// Recommended floor on `rerank_mult` for `Sq8Residual` columns at
/// `dim ≤ 384`. The compressed first-pass score needs more
/// candidates than fp32 to
/// recover equivalent recall because the dequant noise floor is
/// higher.
const SQ8_LOW_DIM_RERANK_FLOOR: usize = 50;

/// Recommended floor on `rerank_mult` for `Sq8Residual` columns at
/// `dim > 384`. See [`SQ8_LOW_DIM_RERANK_FLOOR`] and
/// [`FP32_HIGH_DIM_RERANK_FLOOR`] for the underlying
/// calibration rationale.
const SQ8_HIGH_DIM_RERANK_FLOOR: usize = 100;

/// Absolute offset for the portable cosine-only Sq8 grid.
pub(crate) const SQ8_FIXED_OFFSET: f32 = -1.0;
/// Absolute scale for the portable cosine-only Sq8 grid.
pub(crate) const SQ8_FIXED_SCALE: f32 = 2.0 / 255.0;
/// Residual divisor for the portable cosine-only Sq8 grid.
pub(crate) const SQ8_FIXED_RESIDUAL_DIVISOR: f32 = 256.0;

/// Largest code value of a single `u16` rerank plane. The single point of
/// truth for the 16-bit range, shared by [`SQ16_FIXED_SCALE`], the codec's
/// [`RerankCodec::code_max`], and the encode/dequant math in `distance.rs`, so
/// the ruler and the plane width can't silently disagree.
pub(crate) const SQ16_CODE_MAX: f32 = 65535.0;
/// Absolute offset for the flat cosine-only Sq16 grid.
pub(crate) const SQ16_FIXED_OFFSET: f32 = -1.0;
/// Absolute scale for the flat cosine-only Sq16 grid: the full
/// `u16` range (`0..=SQ16_CODE_MAX`) spans `[-1, 1]` in even steps.
pub(crate) const SQ16_FIXED_SCALE: f32 = 2.0 / SQ16_CODE_MAX;

/// Per-vector-index rerank codec. Picks the on-disk byte layout of the
/// per-vector rerank values inside the subsection's `full[]`
/// region.
///
/// See the module docs for the on-disk discriminator + lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RerankCodec {
    /// fp32 little-endian, `dim` contiguous f32s per vector.
    /// The rerank distance kernel reads it via
    /// `bytemuck::try_cast_slice` → zero-copy SIMD.
    Fp32,
    /// `Sq8` plus a signed 8-bit residual sidecar. Per-vector
    /// body is `dim` u8 Sq8 codes followed by `dim` i8 residual
    /// codes (residual step = `scale_c[d] / SQ8_RESIDUAL_DIVISOR`).
    /// Search applies both bytes to every RaBitQ shortlist survivor, closing
    /// the tight top-K cosine recall gap plain Sq8 exhibits on
    /// production-shaped 384D corpora.
    Sq8Residual,
    /// `Sq8Residual`'s two-byte row layout with one fixed quantizer for
    /// every cluster and file. Cosine-only. The residual divisor is 256,
    /// yielding approximately 16-bit scalar precision while keeping bytes
    /// portable across drain, compaction, and split.
    Sq8FixedResidual,
    /// Flat uniform 16-bit scalar quantizer on a fixed cosine-only
    /// grid (`offset = -1`, `scale = 2/65535`). One little-endian
    /// `u16` code per dimension (`dim × 2` bytes per vector), scored
    /// in a single pass — no residual plane. Reconstruction is
    /// `x = code × scale + offset`. The grid is fixed constants shared
    /// by every cluster and file, so `Sq16` stores no per-cluster
    /// scale/offset arrays; its only `codec_meta` is the per-doc
    /// dequantized-norm table (`n_docs × 4` for cosine/L2Sq), matching
    /// the Sq8 family so the cosine kernel divides by `‖d̂‖`.
    ///
    /// Cosine-only: the fixed `[-1, 1]` grid assumes unit-normalized input and
    /// clamps any out-of-range component, so callers must normalize (the engine
    /// and bindings do not normalize on your behalf).
    Sq16,
    /// `Sq16`'s single-plane `u16` body over a **per-cluster, data-fitted**
    /// range instead of the fixed `[-1, 1]` grid — i.e. the 16-bit,
    /// variable-ruler counterpart of [`Self::Sq8Residual`], with no residual
    /// plane (the 16-bit grid is fine enough that the residual leg is
    /// unnecessary). One little-endian `u16` per dim (`dim × 2` bytes — the same
    /// storage as `Sq8Residual`), reconstructed as `x = code × scale_c[d] +
    /// offset_c[d]` from the cluster's stored quantizer. On merge the
    /// destination cluster reuses the first contributing input's ruler and
    /// re-encodes the rest onto it, clamping any component that falls outside
    /// that range (the same first-input-ruler behaviour as the residual family;
    /// a wider destination ruler that covers every input's range is future
    /// work). Metric-agnostic (not pinned to `[-1, 1]`); the default rerank
    /// codec for L2Sq / NegDot.
    Sq16Adaptive,
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
    /// `Sq16` is the cosine default — a single 16-bit plane on the fixed
    /// `[-1, 1]` grid: finer per-component precision than `Sq8FixedResidual`
    /// at the same 2 bytes/dim and faster rerank. The strictly finer grid gives
    /// a provably ≤ per-component quantization error, so recall is ≥ in
    /// expectation (and ≥ in the codec-isolated measurements) — not a strict
    /// per-query guarantee, since ranking can flip on a near-tie. Metric-aware
    /// constructors use the data-fitted `Sq16Adaptive` ruler for
    /// non-cosine metrics, whose values are not bounded to `[-1, 1]`.
    fn default() -> Self {
        Self::Sq16
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
            Self::Sq8Residual => 1,
            Self::RabitqOnly => 2,
            Self::Sq8FixedResidual => 3,
            Self::Sq16 => 4,
            Self::Sq16Adaptive => 5,
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
            1 => Some(Self::Sq8Residual),
            2 => Some(Self::RabitqOnly),
            3 => Some(Self::Sq8FixedResidual),
            4 => Some(Self::Sq16),
            5 => Some(Self::Sq16Adaptive),
            _ => None,
        }
    }

    /// Stable human-readable name, used in JSON-config + error
    /// strings.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Sq8Residual => "sq8_residual",
            Self::RabitqOnly => "rabitq_only",
            Self::Sq8FixedResidual => "sq8_fixed_residual",
            Self::Sq16 => "sq16",
            Self::Sq16Adaptive => "sq16_adaptive",
        }
    }

    /// Per-vector body size in bytes inside the `full[]` region.
    /// `0` for [`Self::RabitqOnly`] (no rerank bytes at all).
    #[inline]
    pub const fn per_vector_bytes(self, dim: usize) -> usize {
        match self {
            Self::Fp32 => dim * 4,
            Self::Sq8Residual | Self::Sq8FixedResidual | Self::Sq16 | Self::Sq16Adaptive => dim * 2,
            Self::RabitqOnly => 0,
        }
    }

    /// The vector `dim` implied by a materialized row's `codes` byte length.
    /// The residual family stores a `dim`-byte u8 coarse plane, so the code
    /// length *is* `dim`; the single-plane Sq16 stores a `dim*2`-byte u16
    /// plane, so `dim` is half the code length. The drain spill sizes a
    /// per-cell writer from the first row it materializes and must derive
    /// `dim` this way — assuming `codes.len() == dim` doubles it for Sq16 and
    /// trips the spill's row-shape check.
    #[inline]
    pub(crate) fn dim_from_codes_len(self, codes_len: usize) -> usize {
        if self.is_sq8_residual_family() {
            codes_len
        } else {
            codes_len / 2
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
            Self::Fp32
                | Self::Sq8Residual
                | Self::Sq8FixedResidual
                | Self::Sq16
                | Self::Sq16Adaptive
                | Self::RabitqOnly
        )
    }

    /// Whether the codec uses the shared `[u8 code | i8 residual]` layout.
    /// `Sq16` is deliberately **not** a member — it is a single `u16`
    /// plane with its own scoring path, not the two-plane residual layout.
    #[inline]
    pub const fn is_sq8_residual_family(self) -> bool {
        matches!(self, Self::Sq8Residual | Self::Sq8FixedResidual)
    }

    /// Whether this codec participates in the IVF drain / merge / compaction
    /// maintenance paths that re-quantize rows into a destination cluster
    /// quantizer. True for the residual family (`Sq8Residual`,
    /// `Sq8FixedResidual`) and for the single-plane codecs (`Sq16`,
    /// `Sq16Adaptive`), all of which carry a cold-path `ops()` impl and so can be
    /// built-from-source, drained, merged, and compacted. `Fp32` / `RabitqOnly`
    /// have their own (or no) rerank plane and are never IVF-merged here.
    ///
    /// This is the gate predicate for the maintenance sites; it replaces the
    /// former direct use of [`Self::is_sq8_residual_family`] at those sites so
    /// the mergeable set is named for what it means (eligible for IVF merge)
    /// rather than for the byte layout that happens to coincide with it today.
    #[inline]
    pub const fn is_ivf_mergeable(self) -> bool {
        matches!(
            self,
            Self::Sq8Residual | Self::Sq8FixedResidual | Self::Sq16 | Self::Sq16Adaptive
        )
    }

    /// Whether the codec's per-vector body is a single little-endian `u16`
    /// plane (`Sq16`, `Sq16Adaptive`) rather than the two-plane
    /// `[u8 code ‖ i8 residual]` layout of the residual family. Both single
    /// -plane codecs share the same body/parse/dequant code — they differ only
    /// in the ruler (fixed grid vs per-cluster fitted).
    #[inline]
    pub const fn writes_single_u16_plane(self) -> bool {
        matches!(self, Self::Sq16 | Self::Sq16Adaptive)
    }

    /// Largest quantizer code value for this codec's coarse plane: `65535` for
    /// the single-`u16`-plane codecs, `255` for the `u8`-coarse residual family.
    /// The per-cluster ruler is fit as `scale = (max - min) / code_max`, so this
    /// MUST match the plane's integer width — fitting a `u16` plane with the
    /// `u8` `255` confines every code to `0..=255` and silently throws away 8
    /// bits of the 16-bit grid.
    #[inline]
    pub const fn code_max(self) -> f32 {
        if self.writes_single_u16_plane() {
            SQ16_CODE_MAX
        } else {
            255.0
        }
    }

    /// Whether the codec stores per-cluster `scale`/`offset` quantizer arrays in
    /// `codec_meta` (so open/merge must load them). True for the residual family
    /// and for `Sq16Adaptive`; **not** `Sq16` (fixed grid, no arrays). This is
    /// the ruler-storage axis, distinct from the body-layout axis
    /// ([`Self::is_sq8_residual_family`]) — `Sq16Adaptive` carries the arrays
    /// but is single-plane, so the two axes no longer coincide.
    #[inline]
    pub const fn carries_cluster_quant_meta(self) -> bool {
        matches!(
            self,
            Self::Sq8Residual | Self::Sq8FixedResidual | Self::Sq16Adaptive
        )
    }

    /// Whether the codec fits its per-cluster ruler from the data (rather than
    /// pinning a fixed grid): the build path scans each cluster's `[min,max]`,
    /// and a merge reuses the first contributing input's ruler for the
    /// destination cluster. True for `Sq8Residual` and `Sq16Adaptive`; the
    /// fixed-grid codecs (`Sq8FixedResidual`, `Sq16`) are excluded.
    #[inline]
    pub const fn fits_per_cluster_ruler(self) -> bool {
        matches!(self, Self::Sq8Residual | Self::Sq16Adaptive)
    }

    /// Whether this is the flat single-plane `u16` codec. Scored via
    /// [`crate::superfile::vector::distance::Sq16Kernel`] — the `Fp32`
    /// distance path with a `u16 → f32` dequant front.
    #[inline]
    pub const fn is_sq16(self) -> bool {
        matches!(self, Self::Sq16)
    }

    /// Residual divisor implied by the on-disk codec discriminator.
    #[inline]
    pub const fn residual_divisor(self) -> Option<f32> {
        match self {
            Self::Sq8Residual => Some(SQ8_RESIDUAL_DIVISOR),
            Self::Sq8FixedResidual => Some(SQ8_FIXED_RESIDUAL_DIVISOR),
            // `Sq16`/`Sq16Adaptive` are single planes — no residual step, no divisor.
            Self::Fp32 | Self::Sq16 | Self::Sq16Adaptive | Self::RabitqOnly => None,
        }
    }

    /// Whether the codec quantizes onto a fixed absolute `[-1, 1]`
    /// grid shared by every cluster and file (rather than a per-cluster
    /// fitted quantizer). True for both fixed-grid cosine codecs —
    /// `Sq8FixedResidual` and `Sq16`.
    ///
    /// Note this describes the *grid*, not the on-disk metadata layout:
    /// `Sq16` is fixed-quantizer **and** carries no `codec_meta`, so
    /// per-cluster scale/offset-array sites must additionally gate on
    /// [`Self::is_sq8_residual_family`] to stay off the single-plane
    /// `Sq16` path.
    #[inline]
    pub const fn uses_fixed_quantizer(self) -> bool {
        matches!(self, Self::Sq8FixedResidual | Self::Sq16)
    }

    /// Whether this codec supports the requested metric. The
    /// fixed-grid cosine codecs (`Sq8FixedResidual`, `Sq16`) pin the
    /// quantizer to the `[-1, 1]` cosine range and so are Cosine-only.
    #[inline]
    pub const fn supports_metric(self, metric: Metric) -> bool {
        !matches!(self, Self::Sq8FixedResidual | Self::Sq16) || matches!(metric, Metric::Cosine)
    }

    /// Recommended **lower bound** on `rerank_mult` for this
    /// codec at the given `dim`. Returns `None` for codecs
    /// where rerank is meaningless (today: just
    /// [`Self::RabitqOnly`], which skips the rerank step
    /// entirely).
    ///
    /// Sq8Residual needs more candidates to recover fp32-equivalent
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
            // `Sq16`/`Sq16Adaptive` are ~16-bit clean, so their rerank floor
            // matches `Fp32` rather than the lossier Sq8 first-pass.
            Self::Fp32 | Self::Sq16 | Self::Sq16Adaptive => Some(if high_dim {
                FP32_HIGH_DIM_RERANK_FLOOR
            } else {
                FP32_LOW_DIM_RERANK_FLOOR
            }),
            Self::Sq8Residual => Some(if high_dim {
                SQ8_HIGH_DIM_RERANK_FLOOR
            } else {
                SQ8_LOW_DIM_RERANK_FLOOR
            }),
            Self::Sq8FixedResidual => Some(if high_dim {
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
    /// - `Sq16`: per-doc `sum_x_decoded² : f32` table (`n_docs × 4`
    ///   bytes) for `L2Sq`/`Cosine`, and **nothing else** — the grid is
    ///   fixed constants, so there are no per-cluster scale/offset
    ///   arrays. The per-doc norm lets the cosine kernel divide by the
    ///   dequantized vector norm (`base − dot/‖d̂‖`), matching the Sq8
    ///   family so the recall comparison is apples-to-apples. `NegDot`
    ///   drops the table (0 bytes).
    /// - `Sq8Residual`: **per-cluster** per-dim `(scale, offset)` arrays
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
            // `Sq16`'s grid is fixed constants — no scale/offset arrays.
            // It carries only the per-doc dequantized-norm table for the
            // norm-corrected cosine (and L2Sq) kernel.
            Self::Sq16 => match metric {
                Metric::L2Sq | Metric::Cosine => n_docs * 4,
                Metric::NegDot => 0,
            },
            // Residual family and `Sq16Adaptive` all carry per-cluster
            // scale/offset arrays; `Sq16Adaptive` differs only in the body
            // (single u16 plane), not the codec_meta shape.
            Self::Sq8Residual | Self::Sq8FixedResidual | Self::Sq16Adaptive => {
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

impl fmt::Display for RerankCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The codes / residuals / per-doc norm parsed out of one materialized on-disk
/// rerank row. `residuals` is empty for single-plane codecs (`Sq16`).
pub(crate) struct EncodedRowParts {
    pub codes: Vec<u8>,
    pub residuals: Vec<u8>,
    pub norm_sq: Option<f32>,
}

/// Absolute in-subsection byte offsets of a codec's `codec_meta` sub-regions,
/// computed from the region's base offset. `None` marks a sub-region a codec
/// does not carry: `Sq16` has no per-cluster `scale`/`offset` arrays (fixed
/// grid), so `scale_off`/`offset_off` are `None`; `norms_off` is `None` for
/// `NegDot` (no per-doc dequantized-norm table). Sites that write scale/offset
/// arrays must skip when `scale_off == None` — a stale write corrupts the
/// subsection.
pub(crate) struct CodecMetaLayout {
    pub scale_off: Option<usize>,
    pub offset_off: Option<usize>,
    pub norms_off: Option<usize>,
}

/// Per-codec operations for the cold paths (build-from-source, drain read-back,
/// IVF merge, compaction) so those paths stop branching on the codec: a new
/// codec is one `impl` + one arm in [`RerankCodec::ops`]. Implemented only by
/// the quantized-rerank family (`Sq8Residual`, `Sq8FixedResidual`, `Sq16`);
/// `ops()` returns `None` for `Fp32` / `RabitqOnly`, which carry no quantized
/// rerank plane and keep their own paths. The HOT per-candidate scoring kernel
/// is deliberately NOT part of this trait — it stays statically dispatched.
pub(crate) trait RerankCodecOps: Sync {
    /// Split one materialized on-disk rerank row into its codes/residuals planes
    /// plus the per-doc dequantized norm (when `store_norm`). `scale`/`offset`
    /// are the row's per-cluster quantizer (ignored by fixed-grid codecs).
    /// `row.len()` must equal `RerankCodec::per_vector_bytes(dim)`.
    fn parse_materialized_row(
        &self,
        row: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        store_norm: bool,
    ) -> EncodedRowParts;

    /// Dequantize one row's codes (plus `residuals`, ignored by single-plane
    /// codecs) into fp32 `out` (`out.len() == dim`). `scale`/`offset` are the
    /// row's per-cluster quantizer (ignored by fixed-grid codecs). The
    /// residual family threads its codec's residual divisor internally so
    /// callers no longer `residual_divisor().expect(...)` at the site.
    fn dequantize_row_into(
        &self,
        codes: &[u8],
        residuals: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        out: &mut [f32],
    );

    /// Dequantized-vector squared norm `Σ_d x[d]²` for one materialized
    /// on-disk row. `code` is the contiguous per-vector body
    /// (`RerankCodec::per_vector_bytes(dim)` bytes: `[codes ‖ residuals]` for
    /// the residual family, a single `u16` plane for `Sq16`). `scale`/`offset`
    /// are the row's per-cluster quantizer (ignored by fixed-grid codecs). The
    /// residual family threads its residual divisor internally.
    fn decoded_norm_sq(&self, code: &[u8], dim: usize, scale: &[f32], offset: &[f32]) -> f32;

    /// Copy or transcode one source `row` into the destination cluster
    /// quantizer, writing `out` (`RerankCodec::per_vector_bytes(dim)` bytes).
    /// `self` is the destination codec's ops. Returns residual-corrected
    /// `‖x‖²` when `store_norm` (L2Sq/Cosine). Errors if `row`'s codec does not
    /// match this destination codec.
    fn materialize_row_into_cluster_quant(
        &self,
        row: &EncodedCellRow,
        dst_scale: &[f32],
        dst_offset: &[f32],
        dim: usize,
        out: &mut [u8],
        store_norm: bool,
    ) -> Result<Option<f32>, BuildError>;

    /// Absolute byte offsets of this codec's `codec_meta` sub-regions given the
    /// region base `meta_off`. Mirrors `RerankCodec::codec_meta_bytes`'s layout:
    /// residual family lays out `[scale ‖ offset ‖ norms?]`; `Sq16` carries only
    /// `[norms?]` (no scale/offset arrays). `norms_off` is present only for
    /// `L2Sq`/`Cosine`.
    fn codec_meta_layout(
        &self,
        meta_off: usize,
        n_cent: usize,
        dim: usize,
        metric: Metric,
    ) -> CodecMetaLayout;
}

pub(crate) struct Sq8ResidualOps;
pub(crate) struct Sq8FixedResidualOps;
pub(crate) struct Sq16Ops;
pub(crate) struct Sq16AdaptiveOps;

/// Shared residual-family row split: `dim` u8 coarse codes + `dim` i8 residual
/// codes; norm via [`sq8_residual_norm_sq`] with the codec's `divisor`.
fn parse_residual_family_row(
    row: &[u8],
    dim: usize,
    scale: &[f32],
    offset: &[f32],
    divisor: f32,
    store_norm: bool,
) -> EncodedRowParts {
    let codes = row[..dim].to_vec();
    let residuals = row[dim..dim * 2].to_vec();
    let norm_sq =
        store_norm.then(|| sq8_residual_norm_sq(scale, offset, &codes, &residuals, divisor));
    EncodedRowParts {
        codes,
        residuals,
        norm_sq,
    }
}

/// Shared residual-family dequant: delegate to [`dequantize_sq8_residual_into`]
/// with the codec's residual `divisor`. `out.len()` fixes `dim`.
fn dequantize_residual_family_into(
    codes: &[u8],
    residuals: &[u8],
    scale: &[f32],
    offset: &[f32],
    divisor: f32,
    out: &mut [f32],
) {
    dequantize_sq8_residual_into(scale, offset, codes, residuals, divisor, out);
}

/// Shared residual-family decoded-norm: split the contiguous per-vector body
/// `[codes(dim) ‖ residuals(dim)]` and reduce via [`sq8_residual_norm_sq`]
/// with the codec's residual `divisor`.
fn residual_family_norm_sq(
    code: &[u8],
    dim: usize,
    scale: &[f32],
    offset: &[f32],
    divisor: f32,
) -> f32 {
    sq8_residual_norm_sq(scale, offset, &code[..dim], &code[dim..dim * 2], divisor)
}

/// Shared residual-family `codec_meta` layout: `[scale ‖ offset ‖ norms?]`,
/// mirroring [`RerankCodec::codec_meta_bytes`]. Norms present for L2Sq/Cosine.
fn residual_family_codec_meta_layout(
    meta_off: usize,
    n_cent: usize,
    dim: usize,
    metric: Metric,
) -> CodecMetaLayout {
    let scale_off = meta_off;
    let offset_off = scale_off + n_cent * dim * size_of::<f32>();
    let norms_off = matches!(metric, Metric::L2Sq | Metric::Cosine)
        .then_some(offset_off + n_cent * dim * size_of::<f32>());
    CodecMetaLayout {
        scale_off: Some(scale_off),
        offset_off: Some(offset_off),
        norms_off,
    }
}

impl RerankCodecOps for Sq8ResidualOps {
    fn parse_materialized_row(
        &self,
        row: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        store_norm: bool,
    ) -> EncodedRowParts {
        parse_residual_family_row(row, dim, scale, offset, SQ8_RESIDUAL_DIVISOR, store_norm)
    }

    fn dequantize_row_into(
        &self,
        codes: &[u8],
        residuals: &[u8],
        _dim: usize,
        scale: &[f32],
        offset: &[f32],
        out: &mut [f32],
    ) {
        dequantize_residual_family_into(codes, residuals, scale, offset, SQ8_RESIDUAL_DIVISOR, out);
    }

    fn decoded_norm_sq(&self, code: &[u8], dim: usize, scale: &[f32], offset: &[f32]) -> f32 {
        residual_family_norm_sq(code, dim, scale, offset, SQ8_RESIDUAL_DIVISOR)
    }

    fn materialize_row_into_cluster_quant(
        &self,
        row: &EncodedCellRow,
        dst_scale: &[f32],
        dst_offset: &[f32],
        dim: usize,
        out: &mut [u8],
        store_norm: bool,
    ) -> Result<Option<f32>, BuildError> {
        residual_family_materialize_into_cluster_quant(
            row,
            RerankCodec::Sq8Residual,
            dst_scale,
            dst_offset,
            dim,
            out,
            store_norm,
        )
    }

    fn codec_meta_layout(
        &self,
        meta_off: usize,
        n_cent: usize,
        dim: usize,
        metric: Metric,
    ) -> CodecMetaLayout {
        residual_family_codec_meta_layout(meta_off, n_cent, dim, metric)
    }
}

impl RerankCodecOps for Sq8FixedResidualOps {
    fn parse_materialized_row(
        &self,
        row: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        store_norm: bool,
    ) -> EncodedRowParts {
        parse_residual_family_row(
            row,
            dim,
            scale,
            offset,
            SQ8_FIXED_RESIDUAL_DIVISOR,
            store_norm,
        )
    }

    fn dequantize_row_into(
        &self,
        codes: &[u8],
        residuals: &[u8],
        _dim: usize,
        scale: &[f32],
        offset: &[f32],
        out: &mut [f32],
    ) {
        dequantize_residual_family_into(
            codes,
            residuals,
            scale,
            offset,
            SQ8_FIXED_RESIDUAL_DIVISOR,
            out,
        );
    }

    fn decoded_norm_sq(&self, code: &[u8], dim: usize, scale: &[f32], offset: &[f32]) -> f32 {
        residual_family_norm_sq(code, dim, scale, offset, SQ8_FIXED_RESIDUAL_DIVISOR)
    }

    fn materialize_row_into_cluster_quant(
        &self,
        row: &EncodedCellRow,
        dst_scale: &[f32],
        dst_offset: &[f32],
        dim: usize,
        out: &mut [u8],
        store_norm: bool,
    ) -> Result<Option<f32>, BuildError> {
        residual_family_materialize_into_cluster_quant(
            row,
            RerankCodec::Sq8FixedResidual,
            dst_scale,
            dst_offset,
            dim,
            out,
            store_norm,
        )
    }

    fn codec_meta_layout(
        &self,
        meta_off: usize,
        n_cent: usize,
        dim: usize,
        metric: Metric,
    ) -> CodecMetaLayout {
        residual_family_codec_meta_layout(meta_off, n_cent, dim, metric)
    }
}

impl RerankCodecOps for Sq16Ops {
    fn parse_materialized_row(
        &self,
        row: &[u8],
        dim: usize,
        _scale: &[f32],
        _offset: &[f32],
        store_norm: bool,
    ) -> EncodedRowParts {
        // Single u16 plane (dim*2 bytes); no residual plane. Norm is computed
        // against the fixed grid, matching the streaming builder byte-for-byte.
        let codes = row[..dim * 2].to_vec();
        let norm_sq = store_norm.then(|| sq16_decoded_norm_sq(&codes, dim));
        EncodedRowParts {
            codes,
            residuals: Vec::new(),
            norm_sq,
        }
    }

    fn dequantize_row_into(
        &self,
        codes: &[u8],
        _residuals: &[u8],
        _dim: usize,
        _scale: &[f32],
        _offset: &[f32],
        out: &mut [f32],
    ) {
        // Single u16 plane; residuals/scale/offset are unused (fixed grid).
        dequantize_sq16_into(codes, out);
    }

    fn decoded_norm_sq(&self, code: &[u8], dim: usize, _scale: &[f32], _offset: &[f32]) -> f32 {
        // Single u16 plane (dim*2 bytes); norm against the fixed grid.
        sq16_decoded_norm_sq(&code[..dim * 2], dim)
    }

    fn materialize_row_into_cluster_quant(
        &self,
        row: &EncodedCellRow,
        _dst_scale: &[f32],
        _dst_offset: &[f32],
        dim: usize,
        out: &mut [u8],
        store_norm: bool,
    ) -> Result<Option<f32>, BuildError> {
        if row.rerank_codec != RerankCodec::Sq16 {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "cannot transcode Sq16 row from {} to {}",
                row.rerank_codec.name(),
                RerankCodec::Sq16.name()
            )));
        }
        // Fixed `[-1, 1]` grid: the `u16` plane is portable across cluster
        // changes, so a merge is a verbatim byte copy (no transcode ever).
        out[..dim * 2].copy_from_slice(&row.codes);
        Ok(store_norm.then(|| {
            row.norm_sq
                .unwrap_or_else(|| sq16_decoded_norm_sq(&row.codes, dim))
        }))
    }

    fn codec_meta_layout(
        &self,
        meta_off: usize,
        _n_cent: usize,
        _dim: usize,
        metric: Metric,
    ) -> CodecMetaLayout {
        // Fixed grid — no per-cluster scale/offset arrays. The only codec_meta
        // is the per-doc dequantized-norm table (L2Sq/Cosine), which sits at
        // the head of the region.
        let norms_off = matches!(metric, Metric::L2Sq | Metric::Cosine).then_some(meta_off);
        CodecMetaLayout {
            scale_off: None,
            offset_off: None,
            norms_off,
        }
    }
}

/// `Sq16Adaptive` = `Sq16`'s single-`u16`-plane body over a per-cluster fitted
/// ruler. It reuses the body/parse layout of [`Sq16Ops`] and overrides only the
/// ruler: the arithmetic reads the passed `scale`/`offset` (via the
/// `*_adaptive_*` distance functions) instead of the fixed grid, its
/// `codec_meta` carries the per-cluster scale/offset arrays like the residual
/// family, and a merge transcodes rather than byte-copies (source and
/// destination cluster grids differ, so each row is decoded against its source
/// ruler and re-encoded against the destination ruler).
impl RerankCodecOps for Sq16AdaptiveOps {
    fn parse_materialized_row(
        &self,
        row: &[u8],
        dim: usize,
        scale: &[f32],
        offset: &[f32],
        store_norm: bool,
    ) -> EncodedRowParts {
        let codes = row[..dim * 2].to_vec();
        let norm_sq = store_norm.then(|| sq16_adaptive_norm_sq(&codes, dim, scale, offset));
        EncodedRowParts {
            codes,
            residuals: Vec::new(),
            norm_sq,
        }
    }

    fn dequantize_row_into(
        &self,
        codes: &[u8],
        _residuals: &[u8],
        _dim: usize,
        scale: &[f32],
        offset: &[f32],
        out: &mut [f32],
    ) {
        dequantize_sq16_adaptive_into(codes, scale, offset, out);
    }

    fn decoded_norm_sq(&self, code: &[u8], dim: usize, scale: &[f32], offset: &[f32]) -> f32 {
        sq16_adaptive_norm_sq(&code[..dim * 2], dim, scale, offset)
    }

    fn materialize_row_into_cluster_quant(
        &self,
        row: &EncodedCellRow,
        dst_scale: &[f32],
        dst_offset: &[f32],
        dim: usize,
        out: &mut [u8],
        store_norm: bool,
    ) -> Result<Option<f32>, BuildError> {
        if row.rerank_codec != RerankCodec::Sq16Adaptive {
            return Err(BuildError::VectorSchemaMismatch(format!(
                "cannot transcode Sq16Adaptive row from {} to {}",
                row.rerank_codec.name(),
                RerankCodec::Sq16Adaptive.name()
            )));
        }
        // Per-cluster grids differ across source and destination, so transcode:
        // decode against the source ruler, re-encode against the destination
        // ruler. The destination reuses the first input's ruler, so a component
        // outside that range is clamped by encode_sq16_adaptive_row; feed the
        // clamp count to the maintenance tripwire so a destination ruler that
        // fails to cover its inputs shouts rather than silently losing recall.
        let mut decoded = vec![0.0f32; dim];
        dequantize_sq16_adaptive_into(&row.codes, &row.scale, &row.offset, &mut decoded);
        let clamped = encode_sq16_adaptive_row(&decoded, dst_scale, dst_offset, out);
        note_transcode_clamped_components(clamped);
        Ok(store_norm.then(|| sq16_adaptive_norm_sq(out, dim, dst_scale, dst_offset)))
    }

    fn codec_meta_layout(
        &self,
        meta_off: usize,
        n_cent: usize,
        dim: usize,
        metric: Metric,
    ) -> CodecMetaLayout {
        // Same `[scale ‖ offset ‖ norms?]` layout as the residual family — the
        // ruler storage is identical; only the row body differs.
        residual_family_codec_meta_layout(meta_off, n_cent, dim, metric)
    }
}

impl RerankCodec {
    /// Cold-path ops for the quantized-rerank family; `None` for `Fp32` /
    /// `RabitqOnly` (no quantized rerank plane — they keep their own paths).
    pub(crate) fn ops(&self) -> Option<&'static dyn RerankCodecOps> {
        match self {
            Self::Sq8Residual => Some(&Sq8ResidualOps),
            Self::Sq8FixedResidual => Some(&Sq8FixedResidualOps),
            Self::Sq16 => Some(&Sq16Ops),
            Self::Sq16Adaptive => Some(&Sq16AdaptiveOps),
            Self::Fp32 | Self::RabitqOnly => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Default codec is `Sq16` (the cosine default). Any change here is a
    /// load-bearing format choice — every caller that uses
    /// `RerankCodec::default()` silently follows this pick, so
    /// the test pins the contract.
    #[test]
    fn default_is_sq16() {
        assert_eq!(RerankCodec::default(), RerankCodec::Sq16);
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
            RerankCodec::Sq8Residual,
            RerankCodec::Sq8FixedResidual,
            RerankCodec::Sq16,
            RerankCodec::Sq16Adaptive,
            RerankCodec::RabitqOnly,
        ] {
            assert_eq!(
                RerankCodec::from_codec_id(c.codec_id()),
                Some(c),
                "round-trip mismatch for {c:?}"
            );
        }
    }

    /// `Sq16Adaptive` sits on both axes the codebase used to conflate: it shares
    /// the single-`u16` body with `Sq16` but the per-cluster fitted ruler with
    /// `Sq8Residual`. Pins each predicate so a future refactor can't silently
    /// re-route it (e.g. onto the fixed grid, or the two-plane body).
    #[test]
    fn sq16_adaptive_sits_on_both_axes() {
        let c = RerankCodec::Sq16Adaptive;
        // additive id, 2 bytes/dim (storage-neutral vs Sq8Residual)
        assert_eq!(c.codec_id(), 5);
        assert_eq!(c.per_vector_bytes(384), 384 * 2);
        // u16 plane ⇒ the ruler MUST be fit to the full 16-bit range, not 255
        // (fitting with 255 confines codes to 0..=255 and drops 8 bits — the
        // recall bug this guards).
        assert_eq!(c.code_max(), 65535.0);
        assert_eq!(RerankCodec::Sq8Residual.code_max(), 255.0);
        // single-u16 body (like Sq16), NOT the two-plane residual body
        assert!(c.writes_single_u16_plane());
        assert!(!c.is_sq8_residual_family());
        // per-cluster fitted ruler (like Sq8Residual), NOT a fixed grid
        assert!(c.fits_per_cluster_ruler());
        assert!(c.carries_cluster_quant_meta());
        assert!(!c.uses_fixed_quantizer());
        // single plane ⇒ no residual divisor; IVF-mergeable; metric-agnostic
        assert_eq!(c.residual_divisor(), None);
        assert!(c.is_ivf_mergeable());
        assert!(c.supports_metric(Metric::L2Sq) && c.supports_metric(Metric::NegDot));
        // dim recovered as half the u16 code length
        assert_eq!(c.dim_from_codes_len(384 * 2), 384);
    }

    /// Unknown discriminator bytes (any value not currently
    /// assigned, e.g. `6`, `255`) return `None`. The reader
    /// upgrades that into a `MalformedVersion` error rather than
    /// guessing. Id `5` now maps to `Sq16Adaptive`.
    #[test]
    fn unknown_codec_id_is_none() {
        assert_eq!(
            RerankCodec::from_codec_id(5),
            Some(RerankCodec::Sq16Adaptive)
        );
        for id in [6u8, 16, 200, 255] {
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
        assert_eq!(RerankCodec::Sq8Residual.per_vector_bytes(384), 768);
        assert_eq!(RerankCodec::Sq8FixedResidual.per_vector_bytes(384), 768);
        assert_eq!(RerankCodec::Sq16.per_vector_bytes(384), 768);
        assert_eq!(RerankCodec::RabitqOnly.per_vector_bytes(384), 0);
    }

    /// Regression guard for the drain spill: the per-cell spill writer is
    /// sized from the first materialized row's `codes` length, which is `dim`
    /// for the residual family (u8 coarse plane) but `dim*2` for Sq16 (u16
    /// plane). Assuming `codes.len() == dim` doubled the Sq16 spill dim and
    /// tripped the row-shape check ("codes N vs expected dim 2N"). This pins
    /// the inverse so a single-plane codec resolves back to its true `dim`.
    #[test]
    fn dim_from_codes_len_inverts_code_plane_size() {
        let dim = 1024usize;
        // Residual family: code plane is `dim` u8 bytes.
        assert_eq!(RerankCodec::Sq8Residual.dim_from_codes_len(dim), dim);
        assert_eq!(RerankCodec::Sq8FixedResidual.dim_from_codes_len(dim), dim);
        // Sq16: code plane is `dim*2` bytes (u16), so dim is half the length.
        assert_eq!(RerankCodec::Sq16.dim_from_codes_len(dim * 2), dim);
    }

    /// `writes_full` is the inverse of "this codec is
    /// `RabitqOnly`" — pins the build/open fast-path predicate
    /// to the codec's identity rather than scattered
    /// `matches!(_, RabitqOnly)` checks.
    #[test]
    fn writes_full_matches_per_vector_bytes() {
        for c in [
            RerankCodec::Fp32,
            RerankCodec::Sq8Residual,
            RerankCodec::Sq8FixedResidual,
            RerankCodec::Sq16,
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
        assert!(RerankCodec::Sq8Residual.is_implemented());
        assert!(RerankCodec::Sq8FixedResidual.is_implemented());
        assert!(RerankCodec::Sq16.is_implemented());
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
            RerankCodec::Sq8Residual.recommended_rerank_mult_floor(384),
            Some(50)
        );
        assert_eq!(
            RerankCodec::Sq8FixedResidual.recommended_rerank_mult_floor(384),
            Some(50)
        );
        // Sq16 tracks the Fp32 floor (it is near-lossless).
        assert_eq!(
            RerankCodec::Sq16.recommended_rerank_mult_floor(384),
            Some(20)
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
            RerankCodec::Sq8Residual.recommended_rerank_mult_floor(1024),
            Some(100)
        );
        assert_eq!(
            RerankCodec::Sq8FixedResidual.recommended_rerank_mult_floor(1024),
            Some(100)
        );
        assert_eq!(
            RerankCodec::Sq16.recommended_rerank_mult_floor(1024),
            Some(50)
        );
        assert_eq!(
            RerankCodec::RabitqOnly.recommended_rerank_mult_floor(1024),
            None
        );
        // Split point: dim == 384 is the low-dim cell; dim == 385
        // crosses into high-dim.
        assert_eq!(
            RerankCodec::Sq8Residual.recommended_rerank_mult_floor(385),
            Some(100)
        );
    }

    /// `Display` renders the stable [`RerankCodec::name`] for every
    /// variant — the same string used in JSON config + error messages.
    #[test]
    fn display_renders_stable_name() {
        assert_eq!(RerankCodec::Fp32.to_string(), "fp32");
        assert_eq!(RerankCodec::Sq8Residual.to_string(), "sq8_residual");
        assert_eq!(
            RerankCodec::Sq8FixedResidual.to_string(),
            "sq8_fixed_residual"
        );
        assert_eq!(RerankCodec::Sq16.to_string(), "sq16");
        assert_eq!(RerankCodec::RabitqOnly.to_string(), "rabitq_only");
        // `Display` must agree with `name` byte-for-byte.
        for c in [
            RerankCodec::Fp32,
            RerankCodec::Sq8Residual,
            RerankCodec::Sq8FixedResidual,
            RerankCodec::Sq16,
            RerankCodec::RabitqOnly,
        ] {
            assert_eq!(c.to_string(), c.name());
        }
    }

    /// Sq8Residual's codec_meta size: `8·n_cent·dim` for negdot,
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
        // Sq8Residual negdot: per-cluster scale + offset arrays.
        let so_bytes = 2 * 1024 * 384 * 4;
        assert_eq!(
            RerankCodec::Sq8Residual.codec_meta_bytes(384, 1_000_000, 1024, Metric::NegDot),
            so_bytes
        );
        // Sq8Residual L2Sq/Cosine: per-cluster scale + offset + per-doc norms.
        assert_eq!(
            RerankCodec::Sq8Residual.codec_meta_bytes(384, 1_000_000, 1024, Metric::Cosine),
            so_bytes + 1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq8FixedResidual.codec_meta_bytes(384, 1_000_000, 1024, Metric::Cosine),
            so_bytes + 1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq8Residual.codec_meta_bytes(384, 1_000_000, 1024, Metric::L2Sq),
            so_bytes + 1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq8Residual.codec_meta_bytes(384, 1_000_000, 1024, Metric::NegDot),
            so_bytes
        );
    }

    #[test]
    fn fixed_residual_contract_is_cosine_only() {
        assert!(RerankCodec::Sq8FixedResidual.supports_metric(Metric::Cosine));
        assert!(!RerankCodec::Sq8FixedResidual.supports_metric(Metric::L2Sq));
        assert!(!RerankCodec::Sq8FixedResidual.supports_metric(Metric::NegDot));
        assert_eq!(
            RerankCodec::Sq8FixedResidual.residual_divisor(),
            Some(SQ8_FIXED_RESIDUAL_DIVISOR)
        );
        assert!(RerankCodec::Sq8FixedResidual.uses_fixed_quantizer());
        assert!(RerankCodec::Sq8FixedResidual.is_sq8_residual_family());
    }

    /// `Sq16` is the flat single-plane codec: cosine-only, no residual
    /// divisor, `dim × 2` bytes per vector, and — unlike the residual
    /// family — it carries **no** `codec_meta` (fixed grid, no per-doc
    /// norms). It must also stay out of `is_sq8_residual_family()` so
    /// the reader routes it to its own scoring path.
    #[test]
    fn sq16_contract_is_flat_cosine_only_norms_meta() {
        assert!(RerankCodec::Sq16.supports_metric(Metric::Cosine));
        assert!(!RerankCodec::Sq16.supports_metric(Metric::L2Sq));
        assert!(!RerankCodec::Sq16.supports_metric(Metric::NegDot));
        assert_eq!(RerankCodec::Sq16.residual_divisor(), None);
        assert!(RerankCodec::Sq16.is_sq16());
        assert!(!RerankCodec::Sq16.is_sq8_residual_family());
        // Sq16 is a fixed `[-1, 1]` grid, semantically like
        // Sq8FixedResidual, so callers see it as fixed-quantizer — but
        // (unlike the residual family) it still carries no codec_meta,
        // asserted below.
        assert!(RerankCodec::Sq16.uses_fixed_quantizer());
        assert!(RerankCodec::Sq16.writes_full());
        assert_eq!(RerankCodec::Sq16.per_vector_bytes(1024), 1024 * 2);
        // codec_meta = per-doc norms only (no scale/offset arrays):
        // n_docs*4 for cosine/L2Sq, 0 for NegDot. Crucially it excludes
        // the residual family's 2*n_cent*dim*4 array bytes.
        assert_eq!(
            RerankCodec::Sq16.codec_meta_bytes(384, 1_000_000, 1024, Metric::Cosine),
            1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq16.codec_meta_bytes(384, 1_000_000, 1024, Metric::L2Sq),
            1_000_000 * 4
        );
        assert_eq!(
            RerankCodec::Sq16.codec_meta_bytes(384, 1_000_000, 1024, Metric::NegDot),
            0
        );
    }
}
