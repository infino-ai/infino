// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Flat-scan probe over the 4-bit resident plane.
//!
//! A measurement seam, not a serving path. It answers one question the
//! engine cannot currently be configured to answer: **what does our
//! 4-bit codec score, and how fast, when it ranks terminally instead of
//! navigating a graph?**
//!
//! The distinction matters because the graph walk and a flat scan fail
//! differently. A walk can miss a neighbourhood outright — its recall
//! mixes codec error with routing error. A scan visits every vector, so
//! whatever it loses is quantization error alone. That is the regime a
//! compressed flat index competes in, so it is the only honest way to
//! compare our codec against one.
//!
//! Nothing here is new arithmetic: the encoder ([`Sq4Scorer`]) and the
//! SIMD nibble kernel it scores with are the shipping ones, reached
//! through the same [`NodeScorer`] interface the walk uses. The only
//! addition is the loop that visits every node instead of a beam.
//!
//! Byte accounting note: the plane rotates through the *blocked*
//! transform, which keeps the rotated space at exactly `dim`, so stored
//! bytes are `dim/2` per plane per row with no power-of-two padding.
//! That matters because a flat scan's per-query cost is bytes-read ÷
//! bandwidth, so stored padding would be paid twice — once in residency
//! and once in latency. [`Sq4FlatIndex::minimum_bytes`] recomputes the
//! floor independently and should equal
//! [`Sq4FlatIndex::resident_bytes`]; a divergence means padding crept
//! back into the plane.

use std::{cmp::Ordering, collections::BinaryHeap};

use rayon::prelude::*;

use crate::superfile::vector::{
    distance::{SQ4_ROW_BLOCK, encode_sq16_row},
    hnsw::{NodeScorer, Sq4Scorer},
};

/// Bytes a 4-bit code occupies per dimension, expressed as its
/// reciprocal: two coordinates share one byte.
const COORDS_PER_BYTE: usize = 2;
/// Bytes per `f32` ruler entry (offset and step each store one per
/// rotated coordinate).
const RULER_ENTRY_BYTES: usize = 4;
/// Minimum rows a rayon task claims. Large enough that per-task setup
/// and the fold's heap allocation stay negligible against the scan, small
/// enough that the tail does not idle threads at the corpus sizes a flat
/// index is used at.
const SCAN_BLOCK_ROWS: usize = 4_096;
/// Rotated coordinates below which the scan scores a BLOCK of rows per
/// pass rather than one row at a time.
///
/// The two strategies trade the same two things against each other.
/// Row-at-a-time reads each row as one sequential run, which is what a
/// DRAM-streamed plane wants. Row-blocked amortizes the per-row query
/// load and horizontal reduction, but reads `SQ4_ROW_BLOCK` interleaved
/// strided streams instead of one sequential one.
///
/// Which wins is set by how many vector blocks a row spans: the per-row
/// fixed cost is roughly one reduction, so it stays under ~10% of the row
/// once a row spans ~10 blocks — i.e. ~640 coordinates. Measured at 100K
/// rows: at dim 200 (3 blocks) blocking took the scan 0.326 → 0.190 ms,
/// while at dim 1536 (24 blocks) it cost 1.455 → 1.652 ms.
const ROW_BLOCK_MAX_COORDS: usize = 640;

/// One scored candidate, ordered so a [`BinaryHeap`] of bounded size
/// evicts the current worst (largest score, since lower is nearer).
#[derive(PartialEq)]
struct Candidate {
    score: f32,
    node: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Scores are finite by construction (fitted ruler, finite
        // codes); `total_cmp` keeps the ordering total regardless.
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A resident 4-bit plane scanned exhaustively per query.
pub struct Sq4FlatIndex {
    scorer: Sq4Scorer,
    dim: usize,
    len: usize,
}

impl Sq4FlatIndex {
    /// Encode `vectors` (row-major, `len × dim` fp32) into the 4-bit
    /// plane.
    ///
    /// The plane's encoder consumes the stored Sq16 representation, so
    /// the rows go through the same [`encode_sq16_row`] the builder
    /// writes with before being fitted and packed to nibbles — i.e. this
    /// reproduces the exact bytes a drain would produce, rather than a
    /// parallel encode path that could drift from it.
    ///
    /// `with_residual` selects the 1 byte/dim construction (coarse plane
    /// plus a residual nibble) over the bare 0.5 byte/dim one.
    pub fn build(vectors: &[f32], dim: usize, rot_seed: u64, with_residual: bool) -> Self {
        assert!(dim > 0, "dim must be non-zero");
        assert!(
            vectors.len().is_multiple_of(dim),
            "vector buffer must be a whole number of rows"
        );
        let len = vectors.len() / dim;
        let mut sq16 = vec![0u8; len * dim * 2];
        for (row, codes) in vectors
            .chunks_exact(dim)
            .zip(sq16.chunks_exact_mut(dim * 2))
        {
            encode_sq16_row(row, codes);
        }
        let scorer = Sq4Scorer::from_sq16_plane(&sq16, dim, len, with_residual, rot_seed, None);
        Self { scorer, dim, len }
    }

    /// Stored rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes held resident to serve: the code plane, the residual plane
    /// when present, and the two ruler vectors. This is the number to
    /// set against a competing index's resident footprint.
    pub fn resident_bytes(&self) -> usize {
        let (codes, residual, offset, step) = self.scorer.parts();
        codes.len()
            + residual.map_or(0, <[u8]>::len)
            + (offset.len() + step.len()) * RULER_ENTRY_BYTES
    }

    /// The byte floor for these codes, recomputed from `dim` and the
    /// plane count rather than read off the buffers. Equal to
    /// [`Self::resident_bytes`] while the rotation stays unpadded; a
    /// divergence is padding creeping back in.
    pub fn minimum_bytes(&self) -> usize {
        let (_, residual, _, _) = self.scorer.parts();
        let planes = if residual.is_some() { 2 } else { 1 };
        let per_row = self.dim.div_ceil(COORDS_PER_BYTE) * planes;
        per_row * self.len + self.dim * COORDS_PER_BYTE * RULER_ENTRY_BYTES
    }

    /// Exhaustive top-`k` for one query. Returns `(node, score)` with
    /// **lower score nearer**, matching the engine's `NegDot` convention,
    /// sorted nearest-first.
    ///
    /// Single query per call by design: a batched form would amortize
    /// per-query setup and layout reuse across a batch, which is exactly
    /// the accounting that makes a published batch figure incomparable
    /// to a served one.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query dimensionality mismatch");
        if k == 0 || self.len == 0 {
            return Vec::new();
        }
        let prepared = self.scorer.prepare(query);
        // Parallel over rows either way (rayon for the CPU wave, as the
        // engine's own scan does; a single-threaded per-node loop measures
        // the loop rather than the codec).
        //
        // Row-blocking is chosen by shape, not always: see
        // [`ROW_BLOCK_MAX_COORDS`]. Below the threshold a row is too few
        // bytes to hide the per-row query load and reduction, so blocking
        // wins; above it the sequential per-row read is worth more than
        // the amortization. Rows past the last whole block fall back to
        // per-node scoring in both cases.
        let block = if self.dim <= ROW_BLOCK_MAX_COORDS {
            Sq4Scorer::row_block()
        } else {
            // Zero whole blocks ⇒ every row takes the per-node path.
            self.len + 1
        };
        let blocks = self.len / block;
        let heap = (0..blocks)
            .into_par_iter()
            .with_min_len(SCAN_BLOCK_ROWS / block)
            .fold(
                || BinaryHeap::<Candidate>::with_capacity(k + 1),
                |mut heap, b| {
                    let first = (b * block) as u32;
                    let mut scores = [0.0f32; SQ4_ROW_BLOCK];
                    self.scorer.score_rows(&prepared, first, &mut scores);
                    for (r, &score) in scores.iter().enumerate() {
                        let node = first + r as u32;
                        // The root is the worst kept candidate, so a
                        // bounded push/pop keeps the k nearest without
                        // sorting N.
                        if heap.len() < k {
                            heap.push(Candidate { score, node });
                        } else if heap.peek().is_some_and(|worst| score < worst.score) {
                            heap.pop();
                            heap.push(Candidate { score, node });
                        }
                    }
                    heap
                },
            )
            .chain(
                // Trailing partial block, scored per node.
                (blocks * block..self.len).into_par_iter().fold(
                    || BinaryHeap::<Candidate>::with_capacity(k + 1),
                    |mut heap, node| {
                        let node = node as u32;
                        let score = self.scorer.score(&prepared, node);
                        if heap.len() < k {
                            heap.push(Candidate { score, node });
                        } else if heap.peek().is_some_and(|worst| score < worst.score) {
                            heap.pop();
                            heap.push(Candidate { score, node });
                        }
                        heap
                    },
                ),
            )
            .reduce(
                || BinaryHeap::<Candidate>::with_capacity(k + 1),
                |mut a, b| {
                    for c in b {
                        if a.len() < k {
                            a.push(c);
                        } else if a.peek().is_some_and(|worst| c.score < worst.score) {
                            a.pop();
                            a.push(c);
                        }
                    }
                    a
                },
            );
        let mut out: Vec<(u32, f32)> = heap.into_iter().map(|c| (c.node, c.score)).collect();
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rotation seed for the fixtures. Any seeded rotation works; fixing
    /// it keeps a failure reproducible.
    const TEST_ROT_SEED: u64 = 0x5EED_4F1A;
    /// Dimensions covering the shapes that exercise different code paths:
    /// a whole number of VNNI blocks, a non-multiple that needs the masked
    /// partial block, a dim below one block, and one ABOVE
    /// [`ROW_BLOCK_MAX_COORDS`] so the row-at-a-time path is covered too
    /// (the equivalence has to hold on both sides of that switch).
    const TEST_DIMS: &[usize] = &[128, 200, 32, 1024];

    fn planted(dim: usize, rows: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        };
        (0..rows * dim).map(|_| next()).collect()
    }

    /// The blocked scan must return exactly what per-node scoring would.
    ///
    /// This is the load-bearing property of row-blocking: it reorders
    /// which query loads and reductions happen when, and nothing else. A
    /// row/query misalignment in the blocked kernel would still produce
    /// plausible scores and a plausible top-k -- it would just be scoring
    /// the wrong vectors -- so an aggregate recall number would not
    /// reliably catch it, and this comparison does.
    #[test]
    fn blocked_scan_matches_per_node_scoring() {
        for &dim in TEST_DIMS {
            for with_residual in [false, true] {
                // Rows deliberately not a multiple of the block, so the
                // trailing partial-block path is covered too.
                let rows = 5 * Sq4Scorer::row_block() + 3;
                let vectors = planted(dim, rows, 0x1234_5678);
                let index = Sq4FlatIndex::build(&vectors, dim, TEST_ROT_SEED, with_residual);
                let query = planted(dim, 1, 0x9ABC_DEF0);
                let prepared = index.scorer.prepare(&query);
                let want: Vec<f32> = (0..rows)
                    .map(|n| index.scorer.score(&prepared, n as u32))
                    .collect();
                let got = index.search(&query, rows);
                assert_eq!(
                    got.len(),
                    rows,
                    "dim {dim}: scan returned {} rows",
                    got.len()
                );
                for (node, score) in got {
                    let expected = want[node as usize];
                    assert!(
                        (expected - score).abs() <= 1e-4 * expected.abs().max(1.0),
                        "dim {dim} residual={with_residual} node {node}: blocked \
                         scan scored {score}, per-node scoring {expected}"
                    );
                }
            }
        }
    }
}
