// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Flat 4-bit vector index: an exhaustive scan over a resident nibble
//! plane, ranking terminally.
//!
//! This is a peer of the graph index, not a mode of it. The two fail
//! differently and are sized differently:
//!
//! - The graph walk visits a beam. Its recall mixes codec error with
//!   routing error, and it re-ranks that beam on the full Sq16 plane —
//!   which therefore has to be resident, at 2 bytes/dim.
//! - This scan visits every row and returns the codes' own ranking.
//!   Whatever it loses is quantization error alone, and **nothing but the
//!   nibble plane is resident**: 0.5 bytes/dim. There is no Sq16 section in
//!   the persisted form at all. (The type and the format also carry the
//!   residual construction at 1.0 bytes/dim, but the drain does not build
//!   it: a single uniform 8-bit plane is expected to dominate coarse+residual
//!   at equal bytes — the Sq16-vs-Sq8Residual theorem, one rung down — so the
//!   1.0 B/dim slot stays open until a matched-bytes comparison fills it.)
//!
//! That second point is the whole reason this index exists. Holding Sq16
//! resident to refine a 4-bit scan would cost 2 B/dim on top of 0.5 and
//! give back the entire memory advantage — at 1536 dims, 3840 bytes/row
//! instead of 768. A flat index that did that would be strictly worse than
//! the graph it was meant to undercut.
//!
//! The trade it accepts in exchange is per-query work linear in the corpus.
//! That is the right trade at embedded scale and the wrong one at ten
//! million rows, so the serving path gates on a document ceiling and falls
//! back to `ivf` above it, exactly as the graph does.
//!
//! Nothing here re-implements the codec. The encoder ([`Sq4Scorer`]) and
//! the SIMD nibble kernels are the shipping ones, reached through the same
//! [`NodeScorer`] interface the walk uses; the only addition is the loop
//! that visits every node instead of a beam.

use std::{cmp::Ordering, collections::BinaryHeap};

use bytes::Bytes;
use rayon::prelude::*;

#[cfg(test)]
use crate::superfile::vector::distance::encode_sq16_row;
use crate::superfile::vector::{
    distance::SQ4_ROW_BLOCK,
    hnsw::{
        Cursor, NodeScorer, Plane, Sq4Scorer, Sq16Scorer, WalkCodec, calibration_queries,
        exhaustive_topk, read_f32_le,
    },
};

/// Magic for the flat index's persisted form.
///
/// A format of its own rather than a graph bundle with the graph omitted.
/// The layouts genuinely differ — this one carries no graph section and, more
/// importantly, no Sq16 plane — so encoding it as a degenerate graph bundle
/// would mean every reader of that format had to know which of its mandatory
/// sections were secretly optional. A distinct magic makes an old reader
/// decline the bytes outright instead of mis-slicing them.
const FLAT_MAGIC_V1: &[u8; 8] = b"INFDFL01";

/// Byte size of the fixed frame: magic(8) + n(u64) + dim(u32) + codec(u8)
/// + col_len(u32) + rot_seed(u64). The doc-id map, ruler, and nibble planes
/// are added on top; naming it keeps the encode capacity hint exact so the
/// final plane extend cannot trigger a realloc of a multi-hundred-MiB buffer.
const FLAT_FIXED_BYTES: usize = FLAT_MAGIC_V1.len() + 8 + 4 + 1 + 4 + 8;

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
/// load and horizontal reduction, but reads [`SQ4_ROW_BLOCK`] interleaved
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

/// A resident 4-bit plane scanned exhaustively per query, with the
/// `node -> stable doc id` map the serving path resolves through.
pub(crate) struct Sq4FlatIndex {
    scorer: Sq4Scorer,
    /// `node_index -> stable doc id`, in node order. Present so a hit can
    /// be answered without touching a superfile: the scan's node index is
    /// meaningless outside this plane.
    doc_ids: Vec<i128>,
    /// Vector column this index was built for. A table can carry several
    /// same-dim vector columns; the serving path must decline a query on a
    /// different column rather than answer it from this one's rows.
    column: String,
    dim: usize,
    len: usize,
}

impl Sq4FlatIndex {
    /// Build from a node-ordered Sq16 code plane — the drain path.
    ///
    /// Superfiles store Sq16 rows and no fp32 source exists anywhere, so the
    /// 4-bit plane is a re-quantization of the decoded 16-bit reconstruction.
    /// The Sq16 codes are consumed here and NOT retained: that is the point
    /// of this index.
    ///
    /// `with_residual` selects the 1.0 byte/dim construction (coarse plane
    /// plus a residual nibble) over the bare 0.5 byte/dim one.
    pub(crate) fn from_sq16_plane(
        sq16_codes: &[u8],
        doc_ids: Vec<i128>,
        column: &str,
        dim: usize,
        with_residual: bool,
        rot_seed: u64,
    ) -> Self {
        let len = doc_ids.len();
        debug_assert_eq!(sq16_codes.len(), len * dim * 2);
        let scorer =
            Sq4Scorer::from_sq16_plane(sq16_codes, dim, len, with_residual, rot_seed, None);
        Self {
            scorer,
            doc_ids,
            column: column.to_string(),
            dim,
            len,
        }
    }

    /// Encode `vectors` (row-major, `len × dim` fp32) into the plane, with
    /// node-ordered synthetic ids.
    ///
    /// Test-only. The serving path never has fp32 in hand — superfiles store
    /// Sq16 and [`Self::from_sq16_plane`] is the real constructor — so this
    /// exists to spare the unit tests a full drain.
    ///
    /// It is NOT a measurement entry point. An earlier revision exposed it to
    /// benches under the `test-helpers` feature, and the 4-bit numbers that
    /// came out described a construction path no table could be configured to
    /// use. A codec comparison sets `vector.search_mode` in the config and
    /// runs the engine normally; it does not reach in here.
    #[cfg(test)]
    fn build(vectors: &[f32], dim: usize, rot_seed: u64, with_residual: bool) -> Self {
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
        Self::from_sq16_plane(
            &sq16,
            (0..len as i128).collect(),
            "",
            dim,
            with_residual,
            rot_seed,
        )
    }

    /// Stored rows.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Whether the index holds no rows.
    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rotated dimensionality — equal to the input `dim`, since the blocked
    /// rotation is unpadded.
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }

    /// The column this index serves.
    pub(crate) fn column(&self) -> &str {
        &self.column
    }

    /// Stable doc id for a scan node, or `None` if the node is out of range.
    pub(crate) fn doc_id(&self, node: u32) -> Option<i128> {
        self.doc_ids.get(node as usize).copied()
    }

    /// Whether the plane carries the residual nibble leg.
    pub(crate) fn has_residual(&self) -> bool {
        self.scorer.has_residual()
    }

    /// Bytes held resident to serve: the code plane, the residual plane
    /// when present, and the two ruler vectors. This is the number to
    /// set against a competing index's resident footprint.
    ///
    /// The `node -> doc id` map is excluded deliberately: every index has to
    /// carry stable ids somewhere, so counting them here would make the
    /// codec's footprint incomparable to another index's plane.
    pub(crate) fn resident_bytes(&self) -> usize {
        let (codes, residual, offset, step) = self.scorer.parts();
        codes.len()
            + residual.map_or(0, <[u8]>::len)
            + (offset.len() + step.len()) * RULER_ENTRY_BYTES
    }

    /// The byte floor for these codes, recomputed from `dim` and the
    /// plane count rather than read off the buffers. Equal to
    /// [`Self::resident_bytes`] while the rotation stays unpadded; a
    /// divergence is padding creeping back in.
    ///
    /// Test-only: it exists to be compared against the real figure, not to be
    /// reported instead of it.
    #[cfg(test)]
    fn minimum_bytes(&self) -> usize {
        let (_, residual, _, _) = self.scorer.parts();
        let planes = if residual.is_some() { 2 } else { 1 };
        let per_row = self.dim.div_ceil(COORDS_PER_BYTE) * planes;
        per_row * self.len + self.dim * COORDS_PER_BYTE * RULER_ENTRY_BYTES
    }

    /// Serialize to the persistable [`FLAT_MAGIC_V1`] form.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let (codes, residual, offset, step) = self.scorer.parts();
        let codec = if self.has_residual() {
            WalkCodec::Sq4Residual
        } else {
            WalkCodec::Sq4
        };
        let col = self.column.as_bytes();
        // The ruler and both nibble planes are sized from the header's
        // `dim`/`n` on read, so a plane whose own dimensions disagree with
        // the header would mis-slice every later section and decode without
        // error. Four integer compares once per drain against a silently
        // corrupt index is not a trade worth thinking about.
        let stride = self.dim.div_ceil(COORDS_PER_BYTE);
        assert_eq!(offset.len(), self.dim, "ruler offset length vs header dim");
        assert_eq!(step.len(), self.dim, "ruler step length vs header dim");
        assert_eq!(
            codes.len(),
            self.len * stride,
            "code plane length vs header"
        );
        assert_eq!(
            residual.map(<[u8]>::len),
            self.has_residual().then_some(self.len * stride),
            "residual plane presence/length vs header codec"
        );
        let mut out = Vec::with_capacity(
            FLAT_FIXED_BYTES
                + col.len()
                + self.len * 16
                + self.dim * 2 * RULER_ENTRY_BYTES
                + codes.len()
                + residual.map_or(0, <[u8]>::len),
        );
        out.extend_from_slice(FLAT_MAGIC_V1);
        out.extend_from_slice(&(self.len as u64).to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        out.push(codec.tag());
        out.extend_from_slice(&(col.len() as u32).to_le_bytes());
        out.extend_from_slice(col);
        for &id in &self.doc_ids {
            out.extend_from_slice(&id.to_le_bytes());
        }
        out.extend_from_slice(&self.scorer.rot_seed().to_le_bytes());
        for v in offset.iter().chain(step) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(codes);
        if let Some(res) = residual {
            out.extend_from_slice(res);
        }
        out
    }

    /// Rebuild from [`Self::encode`]. `None` on any malformation, so a
    /// corrupt index degrades to the `ivf` fallback rather than failing the
    /// query.
    ///
    /// When `bundle` is the memory-mapped index the nibble planes are served
    /// as zero-copy [`Bytes::slice_ref`] slices of it, so opening a
    /// multi-hundred-MiB plane copies nothing and the returned index keeps
    /// the mapping alive.
    pub(crate) fn decode(bundle: &Bytes) -> Option<Self> {
        let bytes: &[u8] = bundle.as_ref();
        let mut c = Cursor::new(bytes);
        if c.take(FLAT_MAGIC_V1.len())? != FLAT_MAGIC_V1 {
            return None;
        }
        let len = c.u64()? as usize;
        let dim = c.u32()? as usize;
        if dim == 0 {
            return None;
        }
        let codec = WalkCodec::from_tag(c.u8()?)?;
        // A flat index is defined by its 4-bit plane; any other codec tag is
        // a bundle this reader cannot serve.
        if !codec.is_sq4() {
            return None;
        }
        let col_len = c.u32()? as usize;
        if col_len > c.remaining() {
            return None;
        }
        let column = String::from_utf8(c.take(col_len)?.to_vec()).ok()?;
        // Cross-check the doc-id block against the bytes present BEFORE
        // reserving, so a corrupt `len` (e.g. ~2^60) cannot drive a huge
        // `with_capacity` that aborts under `handle_alloc_error`.
        if len.checked_mul(16)? > c.remaining() {
            return None;
        }
        let mut doc_ids = Vec::with_capacity(len);
        for _ in 0..len {
            doc_ids.push(c.i128()?);
        }
        let rot_seed = c.u64()?;
        if dim.checked_mul(2)?.checked_mul(RULER_ENTRY_BYTES)? > c.remaining() {
            return None;
        }
        let offset = read_f32_le(c.take(dim.checked_mul(RULER_ENTRY_BYTES)?)?);
        let step = read_f32_le(c.take(dim.checked_mul(RULER_ENTRY_BYTES)?)?);
        let stride = dim.div_ceil(COORDS_PER_BYTE);
        let plane_len = len.checked_mul(stride)?;
        let codes = bundle.slice_ref(c.take(plane_len)?);
        let residual = if codec.with_residual() {
            Some(Plane::Shared(bundle.slice_ref(c.take(plane_len)?)))
        } else {
            None
        };
        let scorer = Sq4Scorer::from_parts(
            Plane::Shared(codes),
            residual,
            offset,
            step,
            rot_seed,
            dim,
            len,
        )?;
        Some(Self {
            scorer,
            doc_ids,
            column,
            dim,
            len,
        })
    }

    /// Exhaustive top-`k` for one query. Returns `(node, score)` with
    /// **lower score nearer**, matching the engine's `NegDot` convention,
    /// sorted nearest-first.
    ///
    /// Single query per call by design: a batched form would amortize
    /// per-query setup and layout reuse across a batch, which is exactly
    /// the accounting that makes a published batch figure incomparable
    /// to a served one.
    pub(crate) fn search(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
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

    /// Recall@`k` of this index's scan against an exhaustive Sq16 scan of the
    /// same rows, over `n_queries` held-out perturbed queries.
    ///
    /// The drain calls this to decide whether to register the index at all.
    /// There is nothing here to tune — no `(m0, ef)` to sweep, since a scan
    /// visits every row — so this is not a calibration. It is a gate: a plane
    /// too coarse for the corpus it was fitted on would otherwise serve
    /// silently below the table's bar, returning wrong neighbours at the right
    /// latency, which nothing downstream could detect.
    ///
    /// `reference` must be the Sq16 plane these codes were fitted from. Grading
    /// against an exhaustive scan of the 4-bit plane itself would measure
    /// nothing: if the codes misrank, the ground truth misranks identically.
    pub(crate) fn probe_recall(
        &self,
        reference: &Sq16Scorer,
        k: usize,
        n_queries: usize,
        seed: u64,
    ) -> f64 {
        if self.len == 0 || k == 0 {
            return 0.0;
        }
        let queries = calibration_queries(reference, n_queries, seed);
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth = exhaustive_topk(reference, q, k);
            let got: Vec<u32> = self.search(q, k).into_iter().map(|(n, _)| n).collect();
            hit += truth.iter().filter(|t| got.contains(t)).count();
            total += truth.len();
        }
        if total == 0 {
            0.0
        } else {
            hit as f64 / total as f64
        }
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

    /// The persisted form must round-trip byte-for-byte and serve
    /// identically, on both codecs and on dims either side of the
    /// row-blocking switch.
    ///
    /// A version bump is where an index can begin decoding into a subtly
    /// different plane, so the comparison is on the SCAN RESULT and not
    /// just on the bytes: a ruler that survived encode but was mis-sliced
    /// on read would still produce a plausible top-k.
    #[test]
    fn persisted_form_round_trips_and_serves_identically() {
        for &dim in TEST_DIMS {
            for with_residual in [false, true] {
                let rows = 3 * Sq4Scorer::row_block() + 5;
                let vectors = planted(dim, rows, 0x000F_F1CE);
                let doc_ids: Vec<i128> = (0..rows as i128).map(|i| 7_000_000 + i * 3).collect();
                let mut sq16 = vec![0u8; rows * dim * 2];
                for (row, codes) in vectors
                    .chunks_exact(dim)
                    .zip(sq16.chunks_exact_mut(dim * 2))
                {
                    encode_sq16_row(row, codes);
                }
                let built = Sq4FlatIndex::from_sq16_plane(
                    &sq16,
                    doc_ids.clone(),
                    "emb",
                    dim,
                    with_residual,
                    TEST_ROT_SEED,
                );
                let bytes = built.encode();
                assert_eq!(
                    &bytes[..FLAT_MAGIC_V1.len()],
                    FLAT_MAGIC_V1,
                    "flat encode must stamp its own magic, not a graph bundle's"
                );
                let decoded =
                    Sq4FlatIndex::decode(&Bytes::from(bytes.clone())).expect("decode flat index");
                assert_eq!(decoded.len(), rows);
                assert_eq!(decoded.dim(), dim);
                assert_eq!(decoded.column(), "emb");
                assert_eq!(decoded.has_residual(), with_residual);
                assert_eq!(decoded.doc_ids, doc_ids, "the node -> id map round-trips");
                assert_eq!(
                    decoded.resident_bytes(),
                    built.resident_bytes(),
                    "residency must not change across a round trip"
                );
                // Re-encoding the decoded index must reproduce the bytes:
                // anything else means a section was dropped or reordered.
                assert_eq!(decoded.encode(), bytes, "dim {dim}: re-encode diverged");

                let query = planted(dim, 1, 0x0D15_EA5E);
                assert_eq!(
                    built.search(&query, 10),
                    decoded.search(&query, 10),
                    "dim {dim} residual={with_residual}: decoded index ranked differently"
                );
            }
        }
    }

    /// The index holds the scorer, the id map and its two descriptors — and
    /// no fourth buffer.
    ///
    /// [`Sq4FlatIndex::resident_bytes`] reports what the SCORER exposes
    /// through `parts()`, so it is blind by construction to anything owned
    /// beside the scorer — and "retain the Sq16 plane to refine with" is
    /// exactly a buffer owned beside the scorer. That regression would keep
    /// [`residency_is_the_nibble_plane_only`] green while residency
    /// quadrupled.
    ///
    /// A retained buffer has to be reachable from the struct, and an owning
    /// handle to one (`Vec`, `Bytes`, `Box<[u8]>`, `Plane`) costs inline
    /// bytes. So pinning the struct's own width closes the hole that the
    /// byte-rate assertion cannot see: adding a field fails here, and the
    /// failure names the reason.
    #[test]
    fn the_index_owns_nothing_beside_the_scorer_and_the_id_map() {
        assert_eq!(
            size_of::<Sq4FlatIndex>(),
            size_of::<Sq4Scorer>()
                + size_of::<Vec<i128>>()
                + size_of::<String>()
                + 2 * size_of::<usize>(),
            "Sq4FlatIndex has gained a field. If it owns a buffer, \
             `resident_bytes` does not count it and the residency assertions \
             are measuring the wrong thing — this index exists to hold the \
             nibble plane and nothing else."
        );
    }

    /// The resident footprint must be the nibble plane and its ruler, and
    /// nothing else — in particular no Sq16 plane.
    ///
    /// This is the index's entire reason for existing. Paired with
    /// [`the_index_owns_nothing_beside_the_scorer_and_the_id_map`], which
    /// covers the buffer this one cannot see.
    #[test]
    fn residency_is_the_nibble_plane_only() {
        let dim = 1536;
        let rows = 1_000;
        let vectors = planted(dim, rows, 0xBEEF);
        for (with_residual, bytes_per_dim) in [(false, 0.5f64), (true, 1.0f64)] {
            let index = Sq4FlatIndex::build(&vectors, dim, TEST_ROT_SEED, with_residual);
            assert_eq!(
                index.resident_bytes(),
                index.minimum_bytes(),
                "residual={with_residual}: residency exceeds the byte floor, \
                 which means padding crept back into the plane"
            );
            // The ruler is O(dim), not O(rows), so per-row residency
            // converges on the codec's rate.
            let ruler = dim * 2 * RULER_ENTRY_BYTES;
            let per_row = (index.resident_bytes() - ruler) as f64 / rows as f64;
            let want = dim as f64 * bytes_per_dim;
            assert!(
                (per_row - want).abs() <= 1.0,
                "residual={with_residual}: {per_row} bytes/row against the \
                 codec's {want} — an Sq16 plane would show as ~{} here",
                dim * 2
            );
        }
    }

    /// A `k` far smaller than the corpus must return the same `k` nearest an
    /// unbounded scan would, and in the same order.
    ///
    /// This is the path a real query takes and the one the other scan tests
    /// miss: they ask for every row, so the per-task heaps never fill and the
    /// EVICTION arms — in the blocked fold, in the trailing per-node fold, and
    /// in the cross-task reduce — never run. An eviction that dropped the
    /// wrong end would still return `k` plausible rows.
    #[test]
    fn bounded_top_k_matches_an_unbounded_scan() {
        const K: usize = 5;
        for &dim in TEST_DIMS {
            // Deliberately not a whole number of blocks: the trailing rows
            // take the per-node fold, so both eviction arms are exercised.
            let rows = 7 * Sq4Scorer::row_block() + 11;
            let vectors = planted(dim, rows, 0x5EA5_04A1);
            let index = Sq4FlatIndex::build(&vectors, dim, TEST_ROT_SEED, true);
            let query = planted(dim, 1, 0x7E57_0001);

            let bounded = index.search(&query, K);
            let full = index.search(&query, rows);
            assert_eq!(bounded.len(), K, "dim {dim}: a bounded scan returns k");
            // Compared on SCORES, not on `(node, score)` pairs: 4-bit codes
            // are coarse enough to tie, and a tie at the k-th place is a free
            // choice the heap and the full sort may make differently. The k
            // smallest scores are the property either way.
            let bounded_scores: Vec<f32> = bounded.iter().map(|&(_, s)| s).collect();
            let full_scores: Vec<f32> = full[..K].iter().map(|&(_, s)| s).collect();
            assert_eq!(
                bounded_scores, full_scores,
                "dim {dim}: the bounded heap kept a different k than the full \
                 ranking's head"
            );
        }
    }

    /// Malformed bytes decline rather than panicking or mis-slicing.
    #[test]
    fn decode_declines_malformed_input() {
        let dim = 128;
        let rows = 40;
        let vectors = planted(dim, rows, 0xC0FFEE);
        let index = Sq4FlatIndex::build(&vectors, dim, TEST_ROT_SEED, false);
        let good = index.encode();

        assert!(
            Sq4FlatIndex::decode(&Bytes::from_static(b"short")).is_none(),
            "a truncated buffer must decline"
        );
        let mut wrong_magic = good.clone();
        wrong_magic[..FLAT_MAGIC_V1.len()].copy_from_slice(b"INFDDG05");
        assert!(
            Sq4FlatIndex::decode(&Bytes::from(wrong_magic)).is_none(),
            "a graph bundle must not decode as a flat index"
        );
        let mut truncated = good.clone();
        truncated.truncate(good.len() - 1);
        assert!(
            Sq4FlatIndex::decode(&Bytes::from(truncated)).is_none(),
            "a plane one byte short must decline, not serve a short plane"
        );
        // A corrupt row count must be rejected against the bytes present
        // rather than driving a huge allocation.
        let mut huge_n = good.clone();
        huge_n[FLAT_MAGIC_V1.len()..FLAT_MAGIC_V1.len() + 8]
            .copy_from_slice(&(1u64 << 60).to_le_bytes());
        assert!(
            Sq4FlatIndex::decode(&Bytes::from(huge_n)).is_none(),
            "an implausible row count must decline before reserving"
        );
    }
}
