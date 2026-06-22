// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Sq8+ε IVF centroid block — the one shared on-disk representation of an IVF
//! subsection's summary centroid and per-cluster centroids.
//!
//! The invariant is that nothing stored internally is fp32: cell payloads AND
//! centroids are Sq8+residual. This module owns the centroid leg of that
//! invariant. Both the summary centroid and the `n_cent` IVF centroids are
//! quantized together under **one shared quantizer** (the same
//! [`encode_rows`](super::cell_posting::encode_rows) used for cell payloads) and
//! spliced into the subsection as a single contiguous block:
//!
//! ```text
//!   [scale]      — dim × f32
//!   [offset]     — dim × f32
//!   [rows]       — (n_cent + 1) × dim × 2  (codes(dim) u8 ‖ residuals(dim) i8)
//!   [norms]      — (n_cent + 1) × f32  (L2Sq / Cosine only; absent for NegDot)
//! ```
//!
//! Row 0 is the summary centroid; rows `1..=n_cent` are the IVF centroids in
//! cluster-major order. The query is scored against any centroid with
//! [`Sq8ResidualKernel`] — never an fp32 decode — exactly as cell rows are.

use crate::superfile::vector::{
    cell_posting::{EncodedCellRow, ROW_BYTES_PER_DIM, encode_rows, encoded_component_at},
    distance::{Metric, SQ8_RESIDUAL_DIVISOR, Sq8ResidualKernel},
};

/// Width of a little-endian `f32` field.
const F32_BYTES: usize = 4;

/// Whether this metric stores a per-row decoded norm (L2Sq / Cosine fold the
/// candidate's `‖x‖²`; NegDot does not).
#[inline]
pub(crate) fn metric_stores_norm(metric: Metric) -> bool {
    matches!(metric, Metric::L2Sq | Metric::Cosine)
}

/// Number of rows in the centroid block: the summary centroid plus the
/// `n_cent` per-cluster centroids.
#[inline]
pub(crate) fn centroid_block_rows(n_cent: usize) -> usize {
    n_cent + 1
}

/// Byte offset of the `offset` sub-block relative to the block start. The block
/// starts with `scale` at relative offset 0.
#[inline]
pub(crate) fn offset_rel_off(dim: usize) -> usize {
    dim * F32_BYTES
}

/// Byte offset of the `rows` sub-block relative to the block start.
#[inline]
pub(crate) fn rows_rel_off(dim: usize) -> usize {
    offset_rel_off(dim) + dim * F32_BYTES
}

/// Byte offset of the `norms` sub-block relative to the block start (valid only
/// when the metric stores norms).
#[inline]
pub(crate) fn norms_rel_off(dim: usize, n_cent: usize) -> usize {
    rows_rel_off(dim) + centroid_block_rows(n_cent) * dim * ROW_BYTES_PER_DIM
}

/// Total byte size of the centroid block for `(dim, n_cent, metric)`.
#[inline]
pub(crate) fn centroid_block_bytes(dim: usize, n_cent: usize, metric: Metric) -> usize {
    let rows = centroid_block_rows(n_cent);
    let base = 2 * dim * F32_BYTES + rows * dim * ROW_BYTES_PER_DIM;
    if metric_stores_norm(metric) {
        base + rows * F32_BYTES
    } else {
        base
    }
}

/// Encode the summary centroid and the `n_cent` IVF centroids (cluster-major)
/// into one Sq8+ε block under a single shared quantizer, then write it into
/// `bytes` at `block_off`. `summary` is `dim` fp32 components; `centroids` is
/// `n_cent × dim` fp32 components in cluster-major order. The fp32 inputs are
/// the ingest-boundary form (k-means output / averaged centroids); nothing fp32
/// survives into the written bytes.
pub(crate) fn write_centroid_block(
    bytes: &mut [u8],
    block_off: usize,
    metric: Metric,
    dim: usize,
    n_cent: usize,
    summary: &[f32],
    centroids: &[f32],
) {
    debug_assert_eq!(summary.len(), dim);
    debug_assert_eq!(centroids.len(), n_cent * dim);
    let n_rows = centroid_block_rows(n_cent);
    // Lay out [summary ‖ centroids] as one fp32 buffer so the shared quantizer
    // (encode_rows) trains over every centroid at once — row 0 = summary.
    let mut flat = Vec::with_capacity(n_rows * dim);
    flat.extend_from_slice(summary);
    flat.extend_from_slice(centroids);
    let ids: Vec<u32> = (0..n_rows as u32).collect();
    let row_idx: Vec<usize> = (0..n_rows).collect();
    let encoded = encode_rows(metric, &flat, &ids, dim, &row_idx);

    let scale_off = block_off;
    let offset_off = block_off + offset_rel_off(dim);
    let rows_off = block_off + rows_rel_off(dim);
    bytes[scale_off..scale_off + dim * F32_BYTES]
        .copy_from_slice(bytemuck::cast_slice(&encoded.scale));
    bytes[offset_off..offset_off + dim * F32_BYTES]
        .copy_from_slice(bytemuck::cast_slice(&encoded.offset));
    bytes[rows_off..rows_off + encoded.rows.len()].copy_from_slice(&encoded.rows);
    if let Some(norms) = &encoded.per_doc_norms {
        let norms_off = block_off + norms_rel_off(dim, n_cent);
        debug_assert_eq!(norms.len(), n_rows);
        bytes[norms_off..norms_off + n_rows * F32_BYTES]
            .copy_from_slice(bytemuck::cast_slice(norms));
    }
}

/// Read-only view over a centroid block resident in `block` (a slice starting
/// at the block's first byte and at least [`centroid_block_bytes`] long). Holds
/// borrows into the subsection; all accessors score / decode straight off the
/// Sq8+ε bytes with no fp32 centroid materialization in the hot path.
pub(crate) struct CentroidBlock<'a> {
    metric: Metric,
    dim: usize,
    scale: &'a [u8],
    offset: &'a [u8],
    rows: &'a [u8],
    norms: Option<&'a [u8]>,
}

impl<'a> CentroidBlock<'a> {
    /// Wrap a resident block slice. `block` must start at the block's first
    /// byte; only the first [`centroid_block_bytes`] are read.
    pub(crate) fn new(block: &'a [u8], metric: Metric, dim: usize, n_cent: usize) -> Self {
        let scale = &block[..dim * F32_BYTES];
        let offset = &block[offset_rel_off(dim)..offset_rel_off(dim) + dim * F32_BYTES];
        let rows_off = rows_rel_off(dim);
        let n_rows = centroid_block_rows(n_cent);
        let rows = &block[rows_off..rows_off + n_rows * dim * ROW_BYTES_PER_DIM];
        let norms = if metric_stores_norm(metric) {
            let norms_off = norms_rel_off(dim, n_cent);
            Some(&block[norms_off..norms_off + n_rows * F32_BYTES])
        } else {
            None
        };
        Self {
            metric,
            dim,
            scale,
            offset,
            rows,
            norms,
        }
    }

    /// Shared `scale` array (fp32 LE bytes), `dim` entries.
    #[inline]
    pub(crate) fn scale(&self) -> Vec<f32> {
        parse_f32_le(self.scale)
    }

    /// Shared `offset` array (fp32 LE bytes), `dim` entries.
    #[inline]
    pub(crate) fn offset(&self) -> Vec<f32> {
        parse_f32_le(self.offset)
    }

    /// Codes (`dim` u8) for storage row `r` (0 = summary, `1..=n_cent` =
    /// cluster centroids).
    #[inline]
    fn codes(&self, r: usize) -> &[u8] {
        let base = r * self.dim * ROW_BYTES_PER_DIM;
        &self.rows[base..base + self.dim]
    }

    /// Residuals (`dim` i8 LE bytes) for storage row `r`.
    #[inline]
    fn residuals(&self, r: usize) -> &[u8] {
        let base = r * self.dim * ROW_BYTES_PER_DIM + self.dim;
        &self.rows[base..base + self.dim]
    }

    /// Decoded norm for storage row `r` (`None` for NegDot).
    #[inline]
    fn norm(&self, r: usize) -> Option<f32> {
        self.norms.map(|n| {
            let b = r * F32_BYTES;
            f32::from_le_bytes([n[b], n[b + 1], n[b + 2], n[b + 3]])
        })
    }

    /// Build a per-query kernel bound to this block's shared scale/offset. Used
    /// to score the query against every centroid (cluster selection / nprobe).
    pub(crate) fn kernel(&self, query: &[f32]) -> Sq8ResidualKernel {
        Sq8ResidualKernel::new(
            self.metric,
            query,
            &self.scale(),
            &self.offset(),
            SQ8_RESIDUAL_DIVISOR,
        )
    }

    /// Distance from `query` to cluster centroid `c` (`0..n_cent`) under
    /// `kernel` (built once via [`Self::kernel`]). Smaller = closer.
    #[inline]
    pub(crate) fn cluster_distance(&self, kernel: &Sq8ResidualKernel, c: usize) -> f32 {
        let r = c + 1; // row 0 is the summary centroid
        kernel.distance_with_norm(self.codes(r), self.residuals(r), self.norm(r))
    }

    /// Decode cluster centroid `c` (`0..n_cent`) back to `dim` fp32 components.
    pub(crate) fn cluster_components(&self, c: usize) -> Vec<f32> {
        self.row_components(c + 1)
    }

    /// The cluster centroids' Sq8+residual rows (excludes the summary row 0) —
    /// a byte copy of the stored quantized form, for the manifest's per-cluster
    /// routing centroids. No fp32 decode.
    pub(crate) fn cluster_rows(&self) -> &[u8] {
        &self.rows[self.dim * ROW_BYTES_PER_DIM..]
    }

    /// Per-cluster decoded norms (excludes the summary norm); `None` for NegDot.
    pub(crate) fn cluster_norms(&self) -> Option<Vec<f32>> {
        self.norms.map(|n| parse_f32_le(n)[1..].to_vec())
    }

    /// The summary centroid's Sq8+residual row (row 0): `[codes(dim) ‖ residuals(dim)]`.
    pub(crate) fn summary_row(&self) -> &[u8] {
        &self.rows[..self.dim * ROW_BYTES_PER_DIM]
    }

    /// The summary centroid's decoded norm (row 0); `None` for NegDot.
    pub(crate) fn summary_norm(&self) -> Option<f32> {
        self.norm(0)
    }

    /// Decode storage row `r` to fp32 via the shared scale/offset and the row's
    /// codes/residuals — the inverse of [`encode_rows`].
    fn row_components(&self, r: usize) -> Vec<f32> {
        let scale = self.scale();
        let offset = self.offset();
        let row = EncodedCellRow {
            stable_id: 0,
            scale,
            offset,
            codes: self.codes(r).to_vec(),
            residuals: self.residuals(r).to_vec(),
            norm_sq: None,
        };
        (0..self.dim)
            .map(|d| encoded_component_at(&row, d))
            .collect()
    }
}

/// Parse a little-endian fp32 byte slice into a `Vec<f32>`.
fn parse_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(F32_BYTES)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
