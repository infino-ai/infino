// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Hierarchical navigable small-world (HNSW) proximity graph over the
//! vector rerank codecs.
//!
//! The graph is generic over a [`NodeScorer`]: the per-node distance is
//! the *only* thing the codec-specific layer exposes, so [`Hnsw::build`]
//! and [`Hnsw::search`] never see codes, dequant grids, or f32 planes —
//! only `prepare` (fold a query once) and `score` (distance from that
//! folded query to a stored node, lower = nearer). Two scorers ship:
//!
//! - [`Sq16Scorer`] — the flat 16-bit scalar codec on the fixed
//!   `[-1, 1]` cosine grid. It is a thin adapter over the existing
//!   [`Sq16Kernel`] fused `u16 → f32` dequant dot, so there is a single
//!   source of truth for the SIMD-tiered scoring math; the graph never
//!   materializes a decoded vector to score a candidate. This is the
//!   impl used in practice.
//! - [`Fp32Scorer`] — raw f32 vectors scored with a plain dot. A
//!   reference impl that proves the graph is codec-agnostic: the same
//!   [`Hnsw::build`] / [`Hnsw::search`] drive it unchanged.
//!
//! Scores are dot-*distances* (`−dot` on unit vectors, so smaller is
//! nearer, equivalent to `1 − cos` up to a constant).
//!
//! Layer assignment is deterministic (seeded SplitMix64), so the tower a
//! node lands on never depends on insert order. [`Hnsw::build`] then
//! inserts nodes concurrently over a rayon pool: each node's adjacency
//! sits behind its own lock, a beam reader clones a neighbor list under
//! that lock and scores outside it, and edge splices take the lock only
//! to write. Concurrency reorders inserts, so the graph is not
//! bit-identical run to run, but the seeded tower plus the diversity
//! heuristic keep walk recall stable. The finished graph is immutable and
//! searched single-threaded.
//!
//! Some items (e.g. [`Fp32Scorer`]) are exercised only by the unit tests,
//! so the module allows dead code rather than sprinkling per-item guards.
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    sync::{Mutex, RwLock},
};

use bytes::Bytes;
use rayon::prelude::*;

use crate::superfile::vector::{
    distance::{
        Metric, SQ4_CODE_MAX, SQ4_LOADING_SIGMAS, SQ4_RESIDUAL_CENTER, SQ4_RESIDUAL_DIVISOR,
        SQ4_ROW_BLOCK, Sq4Kernel, Sq16Kernel, dequantize_sq16_into, dot, encode_sq16_row,
        quantize_query_i8, sq8_walk_dot,
    },
    rotation::RandomRotation,
};

/// Per-node distance the graph is generic over. Lower = nearer.
///
/// `build` and `search` see only this trait — never the codec. A scorer
/// folds a query once via [`prepare`](NodeScorer::prepare) (or an
/// already-stored node via [`prepare_node`](NodeScorer::prepare_node),
/// the node-to-node primitive graph construction needs) and then scores
/// many candidate nodes cheaply against that folded query.
pub(crate) trait NodeScorer {
    /// Query folded into whatever form makes per-candidate scoring cheap
    /// (e.g. the Sq16 kernel's `q_prime` + offset precompute).
    type Prepared;

    /// Number of stored nodes.
    fn len(&self) -> usize;

    /// Whether the scorer holds no nodes.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality.
    fn dim(&self) -> usize;

    /// Fold an external query into the per-candidate scoring form.
    fn prepare(&self, query: &[f32]) -> Self::Prepared;

    /// Fold an already-stored node into the scoring form, so the graph
    /// can measure node-to-node distance during build without ever
    /// decoding the codec itself.
    fn prepare_node(&self, node: u32) -> Self::Prepared;

    /// Distance from the folded query `q` to stored node `node`. Lower
    /// = nearer.
    fn score(&self, q: &Self::Prepared, node: u32) -> f32;
}

/// Backing store for a serving byte plane (the Sq16 code plane or the derived
/// SQ8 walk plane): either owned heap (`Vec`) or a zero-copy slice of the
/// memory-mapped graph bundle (`Bytes`). [`Plane::bytes`] hands out a
/// contiguous `&[u8]` either way, so the scoring / VNNI kernels are unchanged.
///
/// The mapped variant keeps its backing `Bytes` alive: when the bundle is
/// served via `mmap` (the default for local backends, see
/// `slow_vector_state::fetch_resident_index_blob`), the Sq16 and SQ8 planes are
/// `slice_ref` views of that one mapping — one physical page-cache copy shared
/// across every process, and no per-open heap copy of the multi-GiB planes.
pub(crate) enum Plane {
    Owned(Vec<u8>),
    Shared(Bytes),
}

impl Plane {
    #[inline]
    pub(crate) fn bytes(&self) -> &[u8] {
        match self {
            Plane::Owned(v) => v,
            Plane::Shared(b) => b,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes().len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.bytes().is_empty()
    }
}

/// Sq16 node scorer: one `u16` code per dimension on the fixed cosine
/// grid, scored with the existing fused-dequant [`Sq16Kernel`] under the
/// [`Metric::NegDot`] convention (`score = −dot`, so smaller is nearer).
///
/// The codes are stored row-major (`dim × 2` bytes per node) and scored
/// straight from the code bytes — no per-candidate decode buffer.
pub(crate) struct Sq16Scorer {
    /// `len × dim × 2` little-endian `u16` codes, row-major — owned heap, or a
    /// zero-copy slice of the mapped graph bundle.
    codes: Plane,
    dim: usize,
    len: usize,
}

impl Sq16Scorer {
    /// Encode `vectors` (each length `dim`, unit-normalized for the
    /// cosine grid) into Sq16 codes via the engine's own
    /// [`encode_sq16_row`], the exact inverse of the kernel's dequant.
    pub(crate) fn from_unit_vectors(vectors: &[Vec<f32>], dim: usize) -> Self {
        let stride = dim * 2;
        let mut codes = vec![0u8; vectors.len() * stride];
        for (i, v) in vectors.iter().enumerate() {
            debug_assert_eq!(v.len(), dim);
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        Self {
            codes: Plane::Owned(codes),
            dim,
            len: vectors.len(),
        }
    }

    /// Adopt already-encoded Sq16 code bytes verbatim: `codes` is
    /// `len × dim × 2` little-endian `u16` (row-major), exactly the
    /// on-disk `full[]` Sq16 plane. No decode/re-encode round trip.
    pub(crate) fn from_codes(codes: Vec<u8>, dim: usize, len: usize) -> Self {
        debug_assert_eq!(codes.len(), len * dim * 2);
        Self {
            codes: Plane::Owned(codes),
            dim,
            len,
        }
    }

    /// Adopt an already-backed Sq16 code plane verbatim: the plane holds
    /// exactly the `len × dim × 2` on-disk bytes, whether that is a zero-copy
    /// slice of the mapped bundle (`Plane::Shared`) or an owned buffer. No
    /// decode/re-encode round trip and, for the shared variant, no heap copy.
    pub(crate) fn from_plane(codes: Plane, dim: usize, len: usize) -> Self {
        debug_assert_eq!(codes.len(), len * dim * 2);
        Self { codes, dim, len }
    }

    /// The raw node-ordered Sq16 code plane — so an incremental build can
    /// concatenate the prior codes with a freshly-drained delta into one
    /// combined scorer.
    pub(crate) fn codes(&self) -> &[u8] {
        self.codes.bytes()
    }

    #[inline]
    fn row(&self, node: u32) -> &[u8] {
        let stride = self.dim * 2;
        let start = node as usize * stride;
        &self.codes.bytes()[start..start + stride]
    }
}

impl NodeScorer for Sq16Scorer {
    /// The per-query fused-dequant kernel: `q_prime[d] = query[d]·scale`
    /// plus the folded grid offset, reused across every candidate.
    type Prepared = Sq16Kernel;

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn prepare(&self, query: &[f32]) -> Sq16Kernel {
        Sq16Kernel::new(Metric::NegDot, query)
    }

    fn prepare_node(&self, node: u32) -> Sq16Kernel {
        // Decode this node once (the only decode buffer in play, and only
        // at build time) so it can act as the query for node-to-node
        // distance; candidate scoring below stays fused-from-codes.
        let mut decoded = vec![0.0f32; self.dim];
        dequantize_sq16_into(self.row(node), &mut decoded);
        Sq16Kernel::new(Metric::NegDot, &decoded)
    }

    #[inline]
    fn score(&self, q: &Sq16Kernel, node: u32) -> f32 {
        // NegDot: `distance_with_norm` returns `−dot`, computed by the
        // fused `u16 → f32` dequant cross kernel straight off the code
        // bytes — no per-candidate decode.
        q.distance_with_norm(self.row(node), None)
    }
}

/// Raw-f32 reference scorer: plain dot, `score = −dot`. Proves the graph
/// abstracts the codec — the same build/search run over this and
/// [`Sq16Scorer`] with no changes.
pub(crate) struct Fp32Scorer {
    /// `len × dim` contiguous f32s, row-major.
    data: Vec<f32>,
    dim: usize,
    len: usize,
}

impl Fp32Scorer {
    pub(crate) fn from_vectors(vectors: &[Vec<f32>], dim: usize) -> Self {
        let mut data = Vec::with_capacity(vectors.len() * dim);
        for v in vectors {
            debug_assert_eq!(v.len(), dim);
            data.extend_from_slice(v);
        }
        Self {
            data,
            dim,
            len: vectors.len(),
        }
    }

    #[inline]
    fn row(&self, node: u32) -> &[f32] {
        let start = node as usize * self.dim;
        &self.data[start..start + self.dim]
    }
}

impl NodeScorer for Fp32Scorer {
    /// A boxed copy of the query. (`Box<[f32]>` rather than `Vec<f32>`
    /// so the trait's `&Self::Prepared` param is a plain slice ref.)
    type Prepared = Box<[f32]>;

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn prepare(&self, query: &[f32]) -> Box<[f32]> {
        query.to_vec().into_boxed_slice()
    }

    fn prepare_node(&self, node: u32) -> Box<[f32]> {
        self.row(node).to_vec().into_boxed_slice()
    }

    #[inline]
    fn score(&self, q: &Box<[f32]>, node: u32) -> f32 {
        -dot(q, self.row(node))
    }
}

/// Sq4 node scorer: one 4-bit code per ROTATED coordinate on a fitted
/// per-coordinate ruler, two codes packed per byte (low nibble = even
/// coordinate), with an optional second nibble plane carrying a sub-step
/// residual.
///
/// Two properties make 4 bits survivable, and both are load-bearing:
///
/// * **Rotation first.** The stored Sq16 rerank rows are UNROTATED (only
///   the 1-bit codes are rotated — `builder.rs` pass 2), and raw
///   embedding axes carry wildly uneven energy, so quantizing them at 4
///   bits wastes most of the 16 levels per axis. The structured rotation
///   (the same seeded spinner the 1-bit codes use) spreads variance
///   near-uniformly across coordinates, which is exactly what makes a
///   per-coordinate ruler meaningful at this width. Queries rotate once
///   in `prepare` (`O(dim·log dim)`, trivial next to the walk).
/// * **Fitted, not fixed.** A rotated unit vector's components are of
///   order `1/sqrt(dim)` (±0.036 at 768d) while the fixed `[-1, 1]`
///   grid's 4-bit step is `2/15 ≈ 0.133` — wider than the whole occupied
///   range, so every component would collapse onto the two central
///   codes. The same mechanism one rung up is why the original
///   fixed-grid Sq8 lost top-K cosine recall on production corpora.
///
/// The ruler is per-coordinate over the whole plane (not per-cluster):
/// the resident plane is node-ordered across every cell, so there is no
/// cluster to key a ruler on without carrying an assignment per node —
/// and after rotation the coordinates are near-identically distributed,
/// which is what a global per-coordinate fit assumes.
///
/// Scoring is the same fused fold [`Sq16Kernel`] uses, under the
/// [`Metric::NegDot`] convention (`score = −dot`, smaller is nearer):
/// the per-query fold precomputes `q[d]·step[d]` (and its residual-scaled
/// twin) plus the constant offset term, so each candidate costs one
/// multiply-add per nibble straight off the packed bytes with no decode
/// buffer. The kernel is deliberately scalar for the phase-1 recall/RSS
/// measurement; a SIMD nibble kernel is follow-up work if the codec is
/// adopted.
pub(crate) struct Sq4Scorer {
    /// `len × ceil(dim/2)` bytes: coarse plane in ROTATED space, two 4-bit
    /// codes per byte. A [`Plane`] rather than a `Vec` so an Sq4 bundle
    /// section serves zero-copy out of the mapped bundle, exactly as the Sq16
    /// and SQ8 planes do — owning it would copy the whole plane on every open.
    codes: Plane,
    /// Same packing; present only for the residual construction.
    residual: Option<Plane>,
    /// Per-ROTATED-coordinate ruler: reconstruction is
    /// `offset[d] + code·step[d] (+ (res − 7.5)·step[d]/15)`.
    offset: Vec<f32>,
    step: Vec<f32>,
    /// The seeded structured rotation the plane lives in. Queries rotate
    /// on `prepare`; nodes come back through the inverse on
    /// [`NodeScorer::decode_node`]. Reconstructed from `(dim, rot_seed)`
    /// at decode — deterministic, nothing about it is persisted beyond
    /// the seed.
    rot: RandomRotation,
    rot_seed: u64,
    /// Rotated component count. The blocked rotation is unpadded, so this
    /// is ALWAYS `dim`; it stays a named field because the kernels and the
    /// scan read it as the stride of a rotated row.
    ///
    /// The `v04` layout is correct only because of that equality: decode
    /// sizes the ruler as exactly `dim` f32s and each nibble plane as
    /// `n · ceil(dim/2)`. Reintroducing power-of-two padding on the write
    /// side without widening those reads would mis-slice every section
    /// after the ruler on any non-power-of-two `dim` — the header-vs-layout
    /// divergence the versioned format exists to prevent.
    padded: usize,
    dim: usize,
    len: usize,
}

impl Sq4Scorer {
    /// Bytes per node per nibble plane: two codes a byte, odd tail padded.
    #[inline]
    fn stride(dim: usize) -> usize {
        dim.div_ceil(2)
    }

    /// Build a plane by re-encoding an existing node-ordered Sq16 plane —
    /// the drain path: superfiles hold Sq16 rows, and no fp32 source
    /// exists anywhere, so the 4-bit plane is a re-quantization of the
    /// decoded 16-bit reconstruction. `ruler` supplies a prior plane's
    /// ruler for the incremental path (delta rows must land on the ruler
    /// the resident nodes already use — the first-input-ruler rule the
    /// adaptive codecs follow on merge); `None` fits min/max per
    /// coordinate over these rows.
    pub(crate) fn from_sq16_plane(
        sq16_codes: &[u8],
        dim: usize,
        len: usize,
        with_residual: bool,
        rot_seed: u64,
        ruler: Option<(&[f32], &[f32])>,
    ) -> Self {
        debug_assert_eq!(sq16_codes.len(), len * dim * 2);
        let rot = RandomRotation::new(dim, rot_seed);
        let padded = rot.dim;
        let mut raw = vec![0.0f32; dim];
        let mut row = vec![0.0f32; padded];
        // Fit pass (skipped when inheriting a prior ruler), then encode
        // pass; the rotation runs per row per pass — drain-time CPU on
        // the reader pool, never query-time.
        let (offset, step) = match ruler {
            Some((o, st)) => (o.to_vec(), st.to_vec()),
            None => {
                // Load the 16 levels over mean ± Z·sigma, NOT over
                // min/max. A min/max ruler is set by the single most
                // extreme value in each coordinate, so on a heavy-tailed
                // coordinate most of the 16 levels cover range that
                // almost no row occupies, and the rows that do cluster
                // near the mean share only a handful of codes. Fitting
                // the second moment instead puts the levels where the
                // mass is and clamps the tails, which is what the encode
                // pass below already does for out-of-range values.
                let mut sum = vec![0.0f64; padded];
                let mut sumsq = vec![0.0f64; padded];
                let mut lo = vec![f32::INFINITY; padded];
                let mut hi = vec![f32::NEG_INFINITY; padded];
                for i in 0..len {
                    dequantize_sq16_into(&sq16_codes[i * dim * 2..(i + 1) * dim * 2], &mut raw);
                    rot.apply_blocked(&raw, &mut row);
                    for (d, &x) in row.iter().enumerate() {
                        sum[d] += x as f64;
                        sumsq[d] += (x as f64) * (x as f64);
                        lo[d] = lo[d].min(x);
                        hi[d] = hi[d].max(x);
                    }
                }
                let n = len.max(1) as f64;
                let mut offset = vec![0.0f32; padded];
                let mut step = vec![1.0f32; padded];
                for d in 0..padded {
                    let mean = sum[d] / n;
                    let var = (sumsq[d] / n - mean * mean).max(0.0);
                    let sigma = var.sqrt();
                    // A degenerate coordinate (all rows equal, or an
                    // empty plane) keeps the unit step so the stored
                    // ruler stays finite and round-trips.
                    // Only the BARE plane loads over sigma. With a
                    // residual leg the effective resolution is 8 bits, so
                    // clamping a coarse code is what hurts — the residual
                    // nibble then refines an interval the value is not in
                    // — and the span must cover the data. Measured on
                    // dbpedia-1536: sigma loading moved the bare plane's
                    // recall@10 up 0.032 and the residual construction's
                    // DOWN 0.012. It is also unsafe on multimodal data,
                    // where sigma reflects the spread BETWEEN modes and a
                    // sigma-scaled step lands far coarser than the modes
                    // themselves (the planted-cluster walk test pins
                    // exactly that).
                    if with_residual {
                        if lo[d].is_finite() && hi[d] > lo[d] {
                            offset[d] = lo[d];
                            step[d] = (hi[d] - lo[d]) / SQ4_CODE_MAX;
                        }
                    } else if sigma > 0.0 && len > 0 {
                        let half = SQ4_LOADING_SIGMAS as f64 * sigma;
                        offset[d] = (mean - half) as f32;
                        step[d] = (2.0 * half / SQ4_CODE_MAX as f64) as f32;
                    } else {
                        offset[d] = mean as f32;
                    }
                }
                (offset, step)
            }
        };
        let stride = Self::stride(padded);
        let mut codes = vec![0u8; len * stride];
        let mut residual = with_residual.then(|| vec![0u8; len * stride]);
        for i in 0..len {
            dequantize_sq16_into(&sq16_codes[i * dim * 2..(i + 1) * dim * 2], &mut raw);
            rot.apply_blocked(&raw, &mut row);
            for (d, &x) in row.iter().enumerate() {
                let c = ((x - offset[d]) / step[d]).round().clamp(0.0, SQ4_CODE_MAX);
                pack_nibble(&mut codes[i * stride..(i + 1) * stride], d, c as u8);
                if let Some(res) = residual.as_mut() {
                    let recon = offset[d] + c * step[d];
                    let r = ((x - recon) / step[d] * SQ4_RESIDUAL_DIVISOR + SQ4_RESIDUAL_CENTER)
                        .round()
                        .clamp(0.0, SQ4_CODE_MAX);
                    pack_nibble(&mut res[i * stride..(i + 1) * stride], d, r as u8);
                }
            }
        }
        Self {
            codes: Plane::Owned(codes),
            residual: residual.map(Plane::Owned),
            offset,
            step,
            rot,
            rot_seed,
            padded,
            dim,
            len,
        }
    }

    /// Adopt already-packed planes and their stored ruler verbatim (the
    /// bundle decode path). `None` on any shape mismatch, or on a ruler
    /// that is not finite and positive, so a malformed section is rejected
    /// rather than panicking or scoring against nonsense.
    ///
    /// `None` does NOT fail the surrounding decode: the bundle still opens
    /// and serves, on the SQ8 plane [`decode_hnsw`] derives in place of the
    /// rejected 4-bit one. That keeps a torn 4-bit section from taking the
    /// whole graph offline, but it does mean the walk silently runs wider
    /// than configured — [`decode_hnsw`] logs a warning for exactly that
    /// reason.
    pub(crate) fn from_parts(
        codes: Plane,
        residual: Option<Plane>,
        offset: Vec<f32>,
        step: Vec<f32>,
        rot_seed: u64,
        dim: usize,
        len: usize,
    ) -> Option<Self> {
        let rot = RandomRotation::new(dim, rot_seed);
        let padded = rot.dim;
        let plane = len.checked_mul(Self::stride(padded))?;
        if offset.len() != padded
            || step.len() != padded
            || codes.len() != plane
            || residual.as_ref().is_some_and(|r| r.len() != plane)
            || step.iter().any(|s| !s.is_finite() || *s <= 0.0)
            || offset.iter().any(|o| !o.is_finite())
        {
            return None;
        }
        Some(Self {
            codes,
            residual,
            offset,
            step,
            rot,
            rot_seed,
            padded,
            dim,
            len,
        })
    }

    /// The packed planes and ruler, for the bundle encode and for an
    /// incremental drain to extend onto the same ruler.
    pub(crate) fn parts(&self) -> (&[u8], Option<&[u8]>, &[f32], &[f32]) {
        (
            self.codes.bytes(),
            self.residual.as_ref().map(Plane::bytes),
            &self.offset,
            &self.step,
        )
    }

    /// Whether the residual nibble plane is present.
    pub(crate) fn has_residual(&self) -> bool {
        self.residual.is_some()
    }

    /// Rows a [`Self::score_rows`] pass covers.
    pub(crate) fn row_block() -> usize {
        Sq4Kernel::row_block()
    }

    /// Score [`Self::row_block`] consecutive nodes starting at `first`.
    ///
    /// Same results as [`NodeScorer::score`] per node, but the query load
    /// and the horizontal reduction are amortized across the block —
    /// which is what a terminal flat scan is bound by at low dimension,
    /// where a row is too few bytes to hide them. `first + row_block()`
    /// must be within the plane.
    pub(crate) fn score_rows(
        &self,
        prepared: &Sq4Kernel,
        first: u32,
        out: &mut [f32; SQ4_ROW_BLOCK],
    ) {
        prepared.distance_negdot_rows(
            self.codes.bytes(),
            self.residual.as_ref().map(Plane::bytes),
            Self::stride(self.padded),
            first as usize,
            out,
        );
    }

    /// The rotation seed the plane's space derives from — an incremental
    /// drain must encode its delta with the SAME rotation, or the
    /// concatenated planes would live in different spaces.
    pub(crate) fn rot_seed(&self) -> u64 {
        self.rot_seed
    }

    #[inline]
    fn row(plane: &[u8], padded: usize, node: u32) -> &[u8] {
        let stride = Self::stride(padded);
        let start = node as usize * stride;
        &plane[start..start + stride]
    }

    /// Reconstruct one node in ROTATED (padded) space — build- and
    /// calibration-time only; the walk never decodes a candidate. Also the
    /// flat index's norm pass: correcting the estimator needs each row's
    /// reconstruction norm, computed once at drain from these decodes.
    pub(crate) fn decode_rotated_into(&self, node: u32, out: &mut [f32]) {
        let row = Self::row(self.codes.bytes(), self.padded, node);
        let res = self.residual.as_ref().map(Plane::bytes);
        for (d, o) in out.iter_mut().enumerate().take(self.padded) {
            let c = unpack_nibble(row, d) as f32;
            let mut x = self.offset[d] + c * self.step[d];
            if let Some(res) = res {
                let r = unpack_nibble(Self::row(res, self.padded, node), d) as f32;
                x += (r - SQ4_RESIDUAL_CENTER) * self.step[d] / SQ4_RESIDUAL_DIVISOR;
            }
            *o = x;
        }
    }
}

/// Write 4-bit `code` for dimension `d` into a packed row (low nibble =
/// even dimension).
#[inline]
fn pack_nibble(row: &mut [u8], d: usize, code: u8) {
    let byte = &mut row[d / 2];
    if d.is_multiple_of(2) {
        *byte = (*byte & 0xF0) | code;
    } else {
        *byte = (*byte & 0x0F) | (code << 4);
    }
}

/// Read 4-bit code for dimension `d` from a packed row.
#[inline]
fn unpack_nibble(row: &[u8], d: usize) -> u8 {
    let byte = row[d / 2];
    if d.is_multiple_of(2) {
        byte & 0x0F
    } else {
        byte >> 4
    }
}

impl NodeScorer for Sq4Scorer {
    /// The per-query fused kernel from `distance.rs` — ruler folded into
    /// the rotated query once, packed nibble rows scored in-register.
    type Prepared = Sq4Kernel;

    fn len(&self) -> usize {
        self.len
    }

    fn dim(&self) -> usize {
        self.dim
    }

    /// Rotate the query into the plane's space (`O(dim·log dim)`, trivial
    /// next to the walk), then hand the ruler fold to the kernel.
    fn prepare(&self, query: &[f32]) -> Sq4Kernel {
        let mut rq = vec![0.0f32; self.padded];
        self.rot.apply_blocked(query, &mut rq);
        Sq4Kernel::new(&rq, &self.offset, &self.step, self.residual.is_some())
    }

    fn prepare_node(&self, node: u32) -> Sq4Kernel {
        // Fold straight from the node's ROTATED reconstruction — no round
        // trip through the inverse rotation and back. Build-time only.
        let mut rq = vec![0.0f32; self.padded];
        self.decode_rotated_into(node, &mut rq);
        Sq4Kernel::new(&rq, &self.offset, &self.step, self.residual.is_some())
    }

    #[inline]
    fn score(&self, q: &Sq4Kernel, node: u32) -> f32 {
        q.distance_negdot(
            Self::row(self.codes.bytes(), self.padded, node),
            self.residual
                .as_ref()
                .map(|res| Self::row(res.bytes(), self.padded, node)),
        )
    }
}

impl Sq4Scorer {
    /// Query-space reconstruction: dequantize in rotated space, then back
    /// through the inverse rotation.
    ///
    /// Not a [`NodeScorer`] method — calibration derives its probes from the
    /// Sq16 reference rather than through any walk codec, and the serving walk
    /// never decodes a candidate. Kept for tests and for checking a stored row
    /// against a finer plane.
    pub(crate) fn decode_node(&self, node: u32, out: &mut [f32]) {
        let mut rotated = vec![0.0f32; self.padded];
        self.decode_rotated_into(node, &mut rotated);
        self.rot.apply_inverse_blocked(&rotated, out);
    }
}

/// Build-time knobs. Defaults track the common HNSW sweet spot.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HnswParams {
    /// Max neighbors per node on layers above 0.
    pub m: usize,
    /// Max neighbors per node on layer 0 (denser base layer).
    pub m0: usize,
    /// Beam width during construction.
    pub ef_construction: usize,
    /// Seed for the deterministic layer-assignment RNG. Fixed input →
    /// fixed graph; no system randomness or wall-clock is consulted.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            seed: 0x51ED_270B_2E67_6DA5,
        }
    }
}

/// Hard cap on the layer tower so a pathological RNG draw can't allocate
/// an absurd number of empty adjacency levels for one node.
const MAX_LEVEL: u32 = 63;

/// A built HNSW graph. Node-major adjacency: `neighbors[node][level]` is
/// node `node`'s neighbor list at `level`, present for
/// `level <= node_level[node]`.
pub(crate) struct Hnsw {
    neighbors: Vec<Vec<Vec<u32>>>,
    node_level: Vec<u32>,
    entry: u32,
    m: usize,
    m0: usize,
    ef_construction: usize,
    len: usize,
}

/// A `(node, distance)` pair ordered by distance (ties broken by id for
/// determinism). `Ord` via `f32::total_cmp`, so it is safe in the heaps.
#[derive(Clone, Copy, PartialEq)]
struct Scored {
    dist: f32,
    node: u32,
}

impl Eq for Scored {}

impl Ord for Scored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then(self.node.cmp(&other.node))
    }
}

impl PartialOrd for Scored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Epoch-stamped visited set — O(1) reset by bumping the epoch, no
/// per-search allocation and no hashing.
struct VisitedSet {
    stamp: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    fn new(n: usize) -> Self {
        Self {
            stamp: vec![0u32; n],
            epoch: 0,
        }
    }

    fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            // Wrapped: repaint so stale stamps can't alias the new epoch.
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }

    /// Mark `node` visited; return whether it was already visited.
    #[inline]
    fn test_and_set(&mut self, node: u32) -> bool {
        let i = node as usize;
        if self.stamp[i] == self.epoch {
            true
        } else {
            self.stamp[i] = self.epoch;
            false
        }
    }
}

/// SplitMix64 increment (the odd golden-ratio constant `⌊2⁶⁴/φ⌋`), also mixed
/// into calibration/layer seeds to decorrelate their streams.
const SPLITMIX64_INCREMENT: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64 — a tiny, fully deterministic mixer for layer assignment.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(SPLITMIX64_INCREMENT);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic layer for `node`: `floor(−ln(U) · ml)` with `U` a
/// seeded uniform in `(0, 1]`, the standard exponential HNSW tower.
fn assign_level(seed: u64, node: u32, ml: f64) -> u32 {
    let mut st = seed ^ (node as u64).wrapping_mul(SPLITMIX64_INCREMENT);
    let r = splitmix64(&mut st);
    // Top 53 bits → uniform in [0, 1).
    let unif = (r >> 11) as f64 / ((1u64 << 53) as f64);
    if unif <= 0.0 {
        return 0;
    }
    ((-unif.ln()) * ml).floor().min(MAX_LEVEL as f64) as u32
}

impl Hnsw {
    /// Build a graph over every node the scorer holds, inserting nodes
    /// concurrently over the rayon pool. The per-node layer tower is
    /// assigned first (seeded, order-independent); node 0 seeds the entry
    /// point; every other node is then inserted in parallel against the
    /// shared, lock-guarded adjacency (see [`ParBuild`]). The result is a
    /// plain immutable graph — identical in shape/semantics to a serial
    /// build, just not bit-identical across runs.
    pub(crate) fn build<S: NodeScorer + Sync>(scorer: &S, params: HnswParams) -> Hnsw {
        let n = scorer.len();
        if n == 0 {
            return Hnsw {
                neighbors: Vec::new(),
                node_level: Vec::new(),
                entry: 0,
                m: params.m,
                m0: params.m0,
                ef_construction: params.ef_construction,
                len: 0,
            };
        }

        // Deterministic per-node layer tower: independent of insert order,
        // so the parallel build lands each node on the same level a serial
        // build would.
        let ml = 1.0 / (params.m.max(2) as f64).ln();
        let node_level: Vec<u32> = (0..n as u32)
            .map(|node| assign_level(params.seed, node, ml))
            .collect();
        let level0 = node_level[0];

        // One lock per node guards that node's whole adjacency (all its
        // levels). Readers clone the small `Vec<u32>` out under the lock and
        // score outside it; writers hold it only to splice ids.
        let adj: Vec<Mutex<Vec<Vec<u32>>>> = node_level
            .iter()
            .map(|&lvl| Mutex::new(vec![Vec::new(); lvl as usize + 1]))
            .collect();

        let builder = ParBuild {
            adj,
            node_level,
            // Node 0 is the seed entry point: present at all its own levels
            // with empty lists, so every other node has somewhere to descend
            // from. A taller node promotes itself past it during insert.
            entry: RwLock::new(EntryState {
                node: 0,
                top_level: level0,
            }),
            m: params.m,
            m0: params.m0,
            ef_construction: params.ef_construction,
        };

        // Insert nodes 1..n concurrently. `for_each_init` calls `init` once per
        // job (a contiguous run of items a worker processes), not once per
        // element, so the O(n) epoch buffer is amortized across many inserts
        // rather than allocated per insert.
        (1..n as u32).into_par_iter().for_each_init(
            || VisitedSet::new(n),
            |visited, node| builder.insert(scorer, node, visited),
        );

        let entry = builder
            .entry
            .into_inner()
            .expect("invariant: hnsw entry lock never poisoned")
            .node;
        let neighbors: Vec<Vec<Vec<u32>>> = builder
            .adj
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .expect("invariant: hnsw adjacency lock never poisoned")
            })
            .collect();
        Hnsw {
            neighbors,
            node_level: builder.node_level,
            entry,
            m: params.m,
            m0: params.m0,
            ef_construction: params.ef_construction,
            len: n,
        }
    }

    /// Extend an existing graph with newly-appended nodes WITHOUT rebuilding
    /// it: seed a mutable [`ParBuild`] from this graph's adjacency + entry
    /// point, assign the new nodes their (seeded, deterministic) levels, and
    /// insert ONLY nodes `[self.len(), scorer.len())` concurrently. `scorer`
    /// must cover all `scorer.len()` nodes — the prior code plane followed by
    /// the appended delta. Work is ∝ the number of new nodes, not the whole
    /// corpus, so an append updates the graph in seconds where a rebuild
    /// takes minutes.
    ///
    /// Node levels use the same seeded [`assign_level`] as [`build`], so node
    /// `k` lands on the same layer whether it arrives in a fresh build or an
    /// incremental one. The prior nodes' adjacency is preserved as-is and
    /// grows only where a new node links back into it (bounded by the reverse
    /// -link cap + heuristic shrink).
    pub(crate) fn extend<S: NodeScorer + Sync>(self, scorer: &S, params: HnswParams) -> Hnsw {
        let prior = self.len;
        let total = scorer.len();
        if total <= prior {
            return self;
        }
        let prior_entry = self.entry;
        let ml = 1.0 / (params.m.max(2) as f64).ln();

        // Prior levels kept; new nodes get their seeded levels.
        let mut node_level = self.node_level;
        node_level.reserve(total - prior);
        for node in prior..total {
            node_level.push(assign_level(params.seed, node as u32, ml));
        }

        // Seed adjacency: move the prior lists in, give new nodes empty ones.
        let mut adj: Vec<Mutex<Vec<Vec<u32>>>> = Vec::with_capacity(total);
        for lists in self.neighbors {
            adj.push(Mutex::new(lists));
        }
        for &lvl in &node_level[prior..] {
            adj.push(Mutex::new(vec![Vec::new(); lvl as usize + 1]));
        }

        let entry_top = node_level[prior_entry as usize];
        let builder = ParBuild {
            adj,
            node_level,
            entry: RwLock::new(EntryState {
                node: prior_entry,
                top_level: entry_top,
            }),
            m: params.m,
            m0: params.m0,
            ef_construction: params.ef_construction,
        };

        (prior as u32..total as u32).into_par_iter().for_each_init(
            || VisitedSet::new(total),
            |visited, node| builder.insert(scorer, node, visited),
        );

        let entry = builder
            .entry
            .into_inner()
            .expect("invariant: hnsw entry lock never poisoned")
            .node;
        let neighbors: Vec<Vec<Vec<u32>>> = builder
            .adj
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .expect("invariant: hnsw adjacency lock never poisoned")
            })
            .collect();
        Hnsw {
            neighbors,
            node_level: builder.node_level,
            entry,
            m: params.m,
            m0: params.m0,
            ef_construction: params.ef_construction,
            len: total,
        }
    }

    /// Walk greedily downhill at `level` from `entry`, hopping to the
    /// nearest improving neighbor until none is closer.
    fn greedy_nearest<S: NodeScorer>(
        &self,
        scorer: &S,
        prepared: &S::Prepared,
        entry: u32,
        level: u32,
    ) -> u32 {
        let mut best = entry;
        let mut best_d = scorer.score(prepared, entry);
        loop {
            let mut improved = false;
            for &nb in &self.neighbors[best as usize][level as usize] {
                let d = scorer.score(prepared, nb);
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// `ef`-width beam search at one `level`. Returns the surviving
    /// candidates sorted ascending by distance (nearest first).
    fn search_layer<S: NodeScorer>(
        &self,
        scorer: &S,
        prepared: &S::Prepared,
        entry_points: &[u32],
        ef: usize,
        level: u32,
        visited: &mut VisitedSet,
    ) -> Vec<Scored> {
        visited.clear();
        // `cand`: min-heap (nearest popped first). `result`: max-heap
        // capped at `ef` (farthest on top, so it is cheap to evict).
        let mut cand: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut result: BinaryHeap<Scored> = BinaryHeap::new();
        for &ep in entry_points {
            if visited.test_and_set(ep) {
                continue;
            }
            let d = scorer.score(prepared, ep);
            let s = Scored { dist: d, node: ep };
            cand.push(Reverse(s));
            result.push(s);
            if result.len() > ef {
                result.pop();
            }
        }
        while let Some(Reverse(c)) = cand.pop() {
            let farthest = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && result.len() >= ef {
                break;
            }
            for &nb in &self.neighbors[c.node as usize][level as usize] {
                if visited.test_and_set(nb) {
                    continue;
                }
                let d = scorer.score(prepared, nb);
                let farthest = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                if result.len() < ef || d < farthest {
                    let s = Scored { dist: d, node: nb };
                    cand.push(Reverse(s));
                    result.push(s);
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        let mut out: Vec<Scored> = result.into_vec();
        out.sort_unstable();
        out
    }

    /// Search the graph for the `k` nearest nodes to `query`, using an
    /// `ef`-width beam on layer 0. Returns `(node, distance)` ascending.
    /// Allocates a fresh visited set; prefer [`search_scratch`](Self::search_scratch)
    /// on a hot loop (e.g. calibration) to reuse one across many searches.
    pub(crate) fn search<S: NodeScorer>(
        &self,
        scorer: &S,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let mut visited = VisitedSet::new(self.len);
        self.search_scratch(scorer, query, k, ef, &mut visited)
    }

    /// [`search`](Self::search) reusing a caller-owned visited set. The set is
    /// reset in O(1) here, so a caller running many searches (calibration runs
    /// thousands per drain) allocates the O(n) epoch buffer once instead of
    /// per search. `visited` must be sized for at least `self.len` nodes.
    fn search_scratch<S: NodeScorer>(
        &self,
        scorer: &S,
        query: &[f32],
        k: usize,
        ef: usize,
        visited: &mut VisitedSet,
    ) -> Vec<(u32, f32)> {
        if self.len == 0 || k == 0 {
            return Vec::new();
        }
        let prepared = scorer.prepare(query);
        let mut ep = self.entry;
        let top = self.node_level[self.entry as usize];
        let mut l = top;
        while l >= 1 {
            ep = self.greedy_nearest(scorer, &prepared, ep, l);
            l -= 1;
        }
        // `search_layer` resets `visited` (O(1) epoch bump) before use, so a
        // reused scratch set needs no clear here.
        let ef = ef.max(k);
        let found = self.search_layer(scorer, &prepared, &[ep], ef, 0, visited);
        found
            .into_iter()
            .take(k)
            .map(|s| (s.node, s.dist))
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn base_degree(&self) -> usize {
        self.m0
    }

    /// A copy with the layer-0 (base) adjacency reduced to `m0` neighbors per
    /// node — a cheap way to evaluate a smaller base-layer degree without a
    /// native rebuild. Upper layers are untouched. The pruned graph is BOTH the
    /// calibration proxy and what gets persisted for the chosen `m0`.
    ///
    /// The reduction re-runs [`select_neighbors_heuristic`] per node at the
    /// target `m0` — the SAME distance-aware selection [`link_into`] applies
    /// when a list overflows its cap during a build. A positional truncation
    /// (`lst[..m0]`) would be unsound here: an un-overflowed base list is laid
    /// out `[distance-sorted own selection | reverse links in arrival order]`,
    /// so slicing preferentially drops the unsorted reverse-link tail
    /// regardless of distance, leaving a run-varying set of in-degree-zero
    /// nodes permanently unreachable and making small-`m0` recall measure worse
    /// than a native build. Re-selecting by the heuristic keeps the closest
    /// diverse neighbors and matches a native `m0` build closely.
    pub(crate) fn pruned_base_layer<S: NodeScorer + Sync>(&self, scorer: &S, m0: usize) -> Hnsw {
        // Each node's pruned neighbor list is computed independently, so the
        // re-selection fans across rayon like the base build's per-node work
        // (this already runs on the reader pool). `par_iter().enumerate()`
        // keeps node order, so `collect` reassembles the adjacency in place.
        let neighbors = self
            .neighbors
            .par_iter()
            .enumerate()
            .map(|(node, levels)| {
                levels
                    .iter()
                    .enumerate()
                    .map(|(lvl, lst)| {
                        if lvl == 0 && lst.len() > m0 {
                            let prep = scorer.prepare_node(node as u32);
                            let cands: Vec<Scored> = lst
                                .iter()
                                .map(|&x| Scored {
                                    node: x,
                                    dist: scorer.score(&prep, x),
                                })
                                .collect();
                            select_neighbors_heuristic(scorer, cands, m0)
                        } else {
                            lst.clone()
                        }
                    })
                    .collect()
            })
            .collect();
        Hnsw {
            neighbors,
            node_level: self.node_level.clone(),
            entry: self.entry,
            m: self.m,
            m0,
            ef_construction: self.ef_construction,
            len: self.len,
        }
    }
}

/// Outcome of graph calibration: the base-layer degree to build at, the
/// query beam to stamp, the recall it achieves, and whether to register the
/// graph at all (`registered = false` ⇒ serve ivf).
#[derive(Debug, Clone, Copy)]
pub(crate) struct CalibChoice {
    /// Base-layer degree to build the full graph at.
    pub m0: usize,
    /// Query beam (`ef`) to stamp in the bundle header.
    pub ef: usize,
    /// Recall of the winning `(m0, ef)` on the calibration sample.
    pub recall: f64,
    /// Register the graph? `false` ⇒ recall below the graceful floor, serve ivf.
    pub registered: bool,
    /// Recall cleared the full target (vs the `0.9×target` graceful band only).
    pub at_target: bool,
}

/// Exhaustive top-`k` node ids under `scorer` for one query — the calibration
/// ground truth. Sq16-exhaustive matches served fp32 recall to within the
/// codec's own exhaustive ceiling, so it needs no fp32 plane.
pub(crate) fn exhaustive_topk<S: NodeScorer>(scorer: &S, query: &[f32], k: usize) -> Vec<u32> {
    let prepared = scorer.prepare(query);
    let mut all: Vec<Scored> = (0..scorer.len() as u32)
        .map(|node| Scored {
            node,
            dist: scorer.score(&prepared, node),
        })
        .collect();
    all.sort_unstable();
    all.into_iter().take(k).map(|s| s.node).collect()
}

/// Odd Knuth multiplier that spreads calibration query source nodes evenly
/// across the plane (multiplicative hashing) without clustering.
const CALIB_QUERY_STRIDE_MULT: usize = 2_654_435_761;
/// Fraction each calibration query is nudged off its exact source node (then
/// renormalized) so measured recall reflects true off-node search rather than
/// a node's trivial self-hit.
const CALIB_QUERY_JITTER: f32 = 0.05;
/// Recall-`k` anchors the calibrator stamps an `ef` for — a compact k→ef curve
/// so each query's requested `k` gets the minimal `ef` that clears the recall
/// target at that `k`. A single stamped `ef` cannot serve every `k`: an `ef`
/// sized for k=10 under-serves recall@100, and the wide `ef` that clears
/// recall@100 over-serves (needlessly slows) k=10. Ascending; the largest is
/// the ground-truth depth the calibrator computes exhaustively.
const HNSW_CALIB_K_ANCHORS: [usize; 4] = [1, 10, 50, 100];

/// Held-out, perturbed (off-node) calibration queries drawn from the plane —
/// evenly spread source nodes, each jittered off its exact position and
/// renormalized. Shared by the calibrator and the incremental recall re-check.
pub(crate) fn calibration_queries(
    scorer: &Sq16Scorer,
    n_queries: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = scorer.len();
    let dim = scorer.dim();
    let stride = dim * 2;
    let mut rng = seed ^ SPLITMIX64_INCREMENT;
    let nq = n_queries.min(n);
    (0..nq)
        .map(|i| {
            let node = i.wrapping_mul(CALIB_QUERY_STRIDE_MULT) % n;
            let mut v = vec![0.0f32; dim];
            // Straight from the Sq16 codes: a query derived through a
            // coarse plane would carry that plane's error into the probe.
            dequantize_sq16_into(&scorer.codes()[node * stride..(node + 1) * stride], &mut v);
            for x in &mut v {
                let u = (splitmix64(&mut rng) >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
                *x += (u * 2.0 - 1.0) * CALIB_QUERY_JITTER;
            }
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect()
}

/// Measured recall@`k` of an already-built `graph` walked at `ef`, against
/// exhaustive ground truth on held-out perturbed queries. Lets a drain
/// re-check that a graph GROWN by incremental insert still clears its recall
/// bar (the base-layer degree requirement rises with N, so inherited `(m0,
/// ef)` calibrated at a smaller scale can drift below target).
pub(crate) fn measure_recall<S: NodeScorer + Sync>(
    graph: &Hnsw,
    serving: &S,
    reference: &Sq16Scorer,
    ef: usize,
    k: usize,
    n_queries: usize,
    seed: u64,
) -> f64 {
    if graph.is_empty() {
        return 0.0;
    }
    let queries = calibration_queries(reference, n_queries, seed);
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| exhaustive_topk(reference, q, k))
        .collect();
    graph_recall(graph, serving, &queries, &gt, k, ef)
}

/// Recall@k of `graph` walked at `ef` against exhaustive `gt`. `gt` truth lists
/// may be deeper than `k` (the calibrator computes ground truth to the widest
/// anchor); only the top-`k` prefix of each is scored, so recall@k is measured
/// against the top-`k` truth regardless of how deep `gt` was computed.
///
/// `scorer` is the plane the walk runs on, which is not necessarily the plane
/// `gt` was computed from — that separation is the point: grading a walk
/// against an exhaustive scan of its OWN representation cannot see that
/// representation's error.
fn graph_recall<S: NodeScorer>(
    graph: &Hnsw,
    scorer: &S,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    // One visited set reused across every query (calibration runs thousands of
    // searches per drain — a fresh O(n) buffer each would dominate).
    let mut visited = VisitedSet::new(graph.len());
    for (q, truth) in queries.iter().zip(gt) {
        let truth = &truth[..k.min(truth.len())];
        let got: HashSet<u32> = graph
            .search_scratch(scorer, q, k, ef, &mut visited)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        hit += truth.iter().filter(|t| got.contains(t)).count();
        total += truth.len();
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}

/// For a fixed `graph`/`m0`, stamp the compact k→ef curve: the minimal `ef`
/// (from ascending `efs`) that clears `target_recall` at each anchor in
/// `anchors`. One graph search per (query, ef) at the widest anchor depth
/// yields recall at every anchor as a prefix of the same candidate list — no
/// extra walks — reusing the exhaustive `gt` (computed to the widest anchor)
/// the caller already has. An anchor no `ef` clears gets the ceiling `ef`
/// (`efs.last()`); the curve is forced monotonic non-decreasing in `k` so a
/// larger `k` never asks for a narrower beam than a smaller one (measurement
/// noise can otherwise invert two adjacent anchors).
///
/// `scorer` is the SERVING plane — the one the walk will score at query time —
/// while `gt` comes from the Sq16 reference. Sweeping the beam on the serving
/// plane is the whole point: the beam width each `k` needs depends on the
/// codec's error, so calibrating on Sq16 and serving on a coarser plane would
/// stamp a curve for a search that never runs.
fn calibrate_ef_curve<S: NodeScorer>(
    graph: &Hnsw,
    scorer: &S,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    anchors: &[usize],
    efs: &[usize],
    target_recall: f64,
) -> Vec<(u32, u32)> {
    let kmax = anchors.iter().copied().max().unwrap_or(0);
    let ceiling = efs.last().copied().unwrap_or(0);
    // `search_scratch` walks at `ef.max(k)` with `k = kmax`, so the recall an
    // anchor is credited with is measured at beam `max(ef, kmax)` while `chosen`
    // records the un-clamped `ef`. That only agrees when every candidate `ef` is
    // at least the widest anchor; otherwise a small cleared `ef` would be stamped
    // yet served at a narrower beam than it was measured at.
    debug_assert!(
        efs.first().copied().unwrap_or(usize::MAX) >= kmax,
        "ef candidates must be >= the widest anchor ({kmax}) so a stamped ef is served \
         at the beam its recall was measured at"
    );
    // Minimal clearing ef per anchor, filled the first time an ascending ef
    // clears that anchor's target.
    let mut chosen: Vec<Option<usize>> = vec![None; anchors.len()];
    let mut visited = VisitedSet::new(graph.len());
    for &ef in efs {
        if chosen.iter().all(Option::is_some) {
            break;
        }
        let mut hit = vec![0usize; anchors.len()];
        let mut total = vec![0usize; anchors.len()];
        for (q, truth) in queries.iter().zip(gt) {
            // Top-`kmax` candidates at this beam; recall@anchor is the top-anchor
            // prefix of this same list intersected with the top-anchor truth.
            let got: Vec<u32> = graph
                .search_scratch(scorer, q, kmax, ef, &mut visited)
                .into_iter()
                .map(|(n, _)| n)
                .collect();
            for (ai, &a) in anchors.iter().enumerate() {
                let got_a: HashSet<u32> = got.iter().copied().take(a).collect();
                let truth_a = &truth[..a.min(truth.len())];
                hit[ai] += truth_a.iter().filter(|t| got_a.contains(t)).count();
                total[ai] += truth_a.len();
            }
        }
        for (ai, slot) in chosen.iter_mut().enumerate() {
            if slot.is_none() {
                let recall = if total[ai] == 0 {
                    0.0
                } else {
                    hit[ai] as f64 / total[ai] as f64
                };
                if recall >= target_recall {
                    *slot = Some(ef);
                }
            }
        }
    }
    // Emit (k, ef) pairs, defaulting an uncleared anchor to the ceiling ef and
    // clamping each ef up to the running max so the curve is non-decreasing.
    let mut running = 0usize;
    anchors
        .iter()
        .zip(&chosen)
        .map(|(&a, slot)| {
            running = slot.unwrap_or(ceiling).max(running);
            (a as u32, running as u32)
        })
        .collect()
}

/// Calibrate `(m0, ef)` to `target_recall` on `scorer` (the drained Sq16 plane,
/// or a subsample of it). Builds ONE graph at `max(m0_candidates)`, evaluates
/// smaller `m0` by pruning the base layer (cheap) and `ef` by re-search (free),
/// and returns the **fastest** clearing pair (min `ef`, then min `m0` — latency
/// is the graph's whole point). If none clears within the candidates, returns
/// the best achieved with `registered` gated by the `target_recall −
/// `register_floor` graceful bar. Queries are held-out, perturbed (off-node) so
/// recall is realistic.
///
/// `want_curve` gates the per-`k` calibration: callers that only need the
/// `(m0, ef)` choice (the corpus-size probe, the incremental-append recall
/// check) pass `false` to skip the extra anchor sweep, whose result they would
/// discard anyway — the authoritative curve is stamped on the full-corpus build.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate_graph<S: NodeScorer + Sync>(
    serving: &S,
    reference: &Sq16Scorer,
    m0_candidates: &[usize],
    ef_candidates: &[usize],
    target_recall: f64,
    register_floor: f64,
    ef_construction: usize,
    n_queries: usize,
    k: usize,
    seed: u64,
    want_curve: bool,
) -> (CalibChoice, Vec<(u32, u32)>, Option<Hnsw>) {
    let register_floor = register_floor.clamp(0.0, 1.0);
    let n = serving.len();
    let fallback = CalibChoice {
        m0: *m0_candidates.iter().min().unwrap_or(&32),
        ef: *ef_candidates.iter().min().unwrap_or(&128),
        recall: 0.0,
        registered: false,
        at_target: false,
    };
    if n == 0 || m0_candidates.is_empty() || ef_candidates.is_empty() {
        return (fallback, Vec::new(), None);
    }
    // Queries and ground truth both come from `reference`, never from
    // `serving`: an exhaustive scan of the walk's own representation cannot see
    // that representation's error, so a plane too coarse to serve would still
    // calibrate as passing.
    let queries = calibration_queries(reference, n_queries, seed);
    // Ground truth is computed to the widest anchor (or the primary `k`,
    // whichever is deeper) so the m0-selection recall@`k` and every curve
    // anchor's recall are prefixes of the SAME exhaustive lists — no re-derive.
    let gt_depth = HNSW_CALIB_K_ANCHORS
        .iter()
        .copied()
        .max()
        .unwrap_or(k)
        .max(k);
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| exhaustive_topk(reference, q, gt_depth))
        .collect();

    let mut m0s: Vec<usize> = m0_candidates.to_vec();
    m0s.sort_unstable();
    m0s.dedup();
    let mut efs: Vec<usize> = ef_candidates.to_vec();
    efs.sort_unstable();
    efs.dedup();
    let m0_max = *m0s
        .last()
        .expect("invariant: m0 candidates non-empty (guarded above)");
    // Built on the serving plane so its neighbour lists reflect the distances
    // the walk will actually see.
    let base = Hnsw::build(
        serving,
        HnswParams {
            m0: m0_max,
            ef_construction,
            ..HnswParams::default()
        },
    );
    // Fill a recall[m0][ef] matrix by pruning each m0 ONCE, sweeping every ef
    // against that single pruned copy, then dropping it before the next m0.
    // Peak resident stays at `base` + one pruned copy — never all candidates at
    // once (each pruned copy is ~a full graph, multi-GB at scale).
    let recall_matrix: Vec<Vec<f64>> = m0s
        .iter()
        .map(|&m0| {
            let g = base.pruned_base_layer(serving, m0);
            efs.iter()
                .map(|&ef| graph_recall(&g, serving, &queries, &gt, k, ef))
                .collect()
        })
        .collect();

    // Latency-first pick: smallest ef (outer), then smallest m0 (inner), that
    // clears the target; else the best-recall pair seen.
    let mut best = fallback;
    let mut chosen: Option<CalibChoice> = None;
    'search: for (ei, &ef) in efs.iter().enumerate() {
        for (mi, &m0) in m0s.iter().enumerate() {
            let recall = recall_matrix[mi][ei];
            let c = CalibChoice {
                m0,
                ef,
                recall,
                registered: recall >= register_floor,
                at_target: recall >= target_recall,
            };
            if recall > best.recall {
                best = c;
            }
            if recall >= target_recall {
                chosen = Some(c);
                break 'search;
            }
        }
    }
    let choice = chosen.unwrap_or(best);
    // Persist the graph pruned to the chosen m0 — one prune of the base, no
    // second full build; the pruned max-graph IS what serves. `None` when not
    // registered. When the chosen m0 IS the max (the common case for hard
    // high-dim tables), the base graph already has that degree — move it
    // instead of a byte-for-byte deep copy (a full graph is multi-GB at scale).
    let graph = if !choice.registered {
        None
    } else if choice.m0 == m0_max {
        Some(base)
    } else {
        Some(base.pruned_base_layer(serving, choice.m0))
    };
    // Stamp the k→ef curve for the chosen m0's graph (the one that serves).
    // Per-`k` widening only earns its keep on a graph that clears the target:
    // uncleared anchors widen to the ceiling to chase recall, which is what a
    // larger `k` wants. A graceful-band graph (registered but below target at
    // every ef) has no anchor to clear, so a curve would just serve every `k`
    // at the ceiling — strictly slower than the single stamped `choice.ef` it
    // used before this curve existed. Stamp an empty curve there so serving
    // degrades to `ef_search` (= `choice.ef`) for every `k`, exactly the
    // pre-curve behavior. Empty too when the caller does not want a curve, or
    // when nothing registered (no bundle is written in that case).
    let curve = match graph.as_ref() {
        Some(g) if want_curve && choice.at_target => calibrate_ef_curve(
            g,
            serving,
            &queries,
            &gt,
            &HNSW_CALIB_K_ANCHORS,
            &efs,
            target_recall,
        ),
        _ => Vec::new(),
    };
    (choice, curve, graph)
}

/// Sentinel filling unused fixed-stride layer-0 adjacency slots. Node ids
/// are `< n <= u32::MAX`, so this never collides with a real id.
const ADJ_SENTINEL: u32 = u32::MAX;

/// On-disk magic for a serialized [`Hnsw`] graph section.
const HNSW_GRAPH_MAGIC: &[u8; 8] = b"INFHNSW1";

// ---------------- graph serialization ----------------
//
// A little cursor over a byte slice: every read is bounds-checked and
// returns `None` on underrun, so a truncated or corrupt section decodes to
// `None` and the caller falls back rather than panicking.
pub(crate) struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    pub(crate) fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    /// Bytes left unread — used to bound wire-driven allocations before
    /// reserving, so a corrupt length word can't request a huge `Vec`.
    pub(crate) fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }
    pub(crate) fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }
    fn u16(&mut self) -> Option<u16> {
        Some(u16::from_le_bytes(self.take(2)?.try_into().ok()?))
    }
    pub(crate) fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    pub(crate) fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    pub(crate) fn i128(&mut self) -> Option<i128> {
        Some(i128::from_le_bytes(self.take(16)?.try_into().ok()?))
    }
}

impl Hnsw {
    /// Serialize the graph to a self-describing byte section: a small
    /// header, the per-node top level, the layer-0 adjacency at a **fixed
    /// `M0` stride** (unused slots filled with [`ADJ_SENTINEL`] — the bulk
    /// of the bytes, laid out for `base + node*M0*4` addressing), then the
    /// sparse upper-layer lists (few nodes reach level ≥ 1). Paired with
    /// [`from_bytes`](Self::from_bytes).
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let n = self.len;
        let m0 = self.m0.max(1);
        let mut out = Vec::with_capacity(48 + n * (4 + m0 * 4));
        out.extend_from_slice(HNSW_GRAPH_MAGIC);
        out.extend_from_slice(&(n as u64).to_le_bytes());
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        out.extend_from_slice(&(self.m0 as u32).to_le_bytes());
        out.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
        out.extend_from_slice(&self.entry.to_le_bytes());

        for &lvl in &self.node_level {
            out.extend_from_slice(&lvl.to_le_bytes());
        }
        // Layer 0, fixed stride m0.
        for node in 0..n {
            let l0 = &self.neighbors[node][0];
            for slot in 0..m0 {
                let id = l0.get(slot).copied().unwrap_or(ADJ_SENTINEL);
                out.extend_from_slice(&id.to_le_bytes());
            }
        }
        // Upper layers: [count u64] then (node u32, level u32, len u32, ids…).
        let mut upper: Vec<u8> = Vec::new();
        let mut upper_records: u64 = 0;
        for node in 0..n {
            let levels = self.neighbors[node].len();
            for level in 1..levels {
                let list = &self.neighbors[node][level];
                upper.extend_from_slice(&(node as u32).to_le_bytes());
                upper.extend_from_slice(&(level as u32).to_le_bytes());
                upper.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for &id in list {
                    upper.extend_from_slice(&id.to_le_bytes());
                }
                upper_records += 1;
            }
        }
        out.extend_from_slice(&upper_records.to_le_bytes());
        out.extend_from_slice(&upper);
        out
    }

    /// Reconstruct a graph from [`to_bytes`](Self::to_bytes). Returns `None`
    /// on a bad magic, truncation, or an out-of-range node id, so a corrupt
    /// section degrades to a fallback rather than a panic.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Hnsw> {
        let mut c = Cursor::new(bytes);
        if c.take(HNSW_GRAPH_MAGIC.len())? != HNSW_GRAPH_MAGIC {
            return None;
        }
        let n = c.u64()? as usize;
        let m = c.u32()? as usize;
        let m0 = c.u32()? as usize;
        let ef_construction = c.u32()? as usize;
        let entry = c.u32()?;
        if n == 0 || entry as usize >= n || m0 == 0 {
            return None;
        }
        // Cross-check the wire lengths against the bytes actually present
        // BEFORE reserving, so a corrupt `n`/`m0` word cannot drive a huge
        // `with_capacity` (an `n` of u32::MAX would otherwise abort under
        // `handle_alloc_error`). The node-level block is `n * 4` bytes and the
        // fixed-stride layer-0 block is `n * m0 * 4`; both must fit.
        let node_level_bytes = n.checked_mul(4)?;
        let l0_bytes = n.checked_mul(m0)?.checked_mul(4)?;
        if node_level_bytes.checked_add(l0_bytes)? > c.remaining() {
            return None;
        }
        let mut node_level = Vec::with_capacity(n);
        for _ in 0..n {
            let lvl = c.u32()?;
            // A tower taller than the graph ever builds is corruption; reject
            // rather than allocate a `MAX_LEVEL`-plus adjacency vec.
            if lvl > MAX_LEVEL {
                return None;
            }
            node_level.push(lvl);
        }
        // Allocate per-node adjacency sized by its top level.
        let mut neighbors: Vec<Vec<Vec<u32>>> = node_level
            .iter()
            .map(|&lvl| vec![Vec::new(); lvl as usize + 1])
            .collect();
        // Layer 0, fixed stride m0.
        for slot in neighbors.iter_mut() {
            let mut l0 = Vec::with_capacity(m0);
            for _ in 0..m0 {
                let id = c.u32()?;
                if id != ADJ_SENTINEL {
                    if id as usize >= n {
                        return None;
                    }
                    l0.push(id);
                }
            }
            slot[0] = l0;
        }
        // Upper layers.
        let records = c.u64()?;
        for _ in 0..records {
            let node = c.u32()? as usize;
            let level = c.u32()? as usize;
            let len = c.u32()? as usize;
            if node >= n || level >= neighbors[node].len() {
                return None;
            }
            // Bound the per-record allocation by the bytes left (each id is 4).
            if len.checked_mul(4)? > c.remaining() {
                return None;
            }
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                let id = c.u32()? as usize;
                if id >= n {
                    return None;
                }
                // Tower-coverage guard: an edge at `level` is followed into
                // `neighbors[id][level]` during the walk, so `id`'s tower must
                // reach `level`. Without this, a level edge to a shorter tower
                // is an out-of-bounds panic in `greedy_nearest` at query time —
                // exactly the corruption we must degrade to a fallback, not
                // panic inside a query or drain worker.
                if (node_level[id] as usize) < level {
                    return None;
                }
                list.push(id as u32);
            }
            neighbors[node][level] = list;
        }
        Some(Hnsw {
            neighbors,
            node_level,
            entry,
            m,
            m0,
            ef_construction,
            len: n,
        })
    }
}

/// On-disk magic for a persisted `hnsw` data bundle (graph + node→doc-id map +
/// node-ordered Sq16 plane), the self-contained payload a resident data index
/// is rebuilt from at open.
///
/// `v03` is the current format: it appends the derived SQ8 walk plane (`n ×
/// dim` high bytes) as a section right after the Sq16 plane, so a mapped bundle
/// serves *both* planes as zero-copy slices — no per-open derive of the ~0.77
/// GiB SQ8 plane. `v02` (pre-existing bundles) carries only the Sq16 plane; it
/// is still decoded, with the SQ8 plane derived on read as before — the
/// backward-compatible fallback, so no forced rebuild. `v03` and `v02` share
/// an identical header/doc-id/Sq16/graph framing; they differ only by the
/// presence of the SQ8 section (and thus the magic). An older `01` bundle (no
/// column) is neither, so it decodes to `None` and the query falls back to ivf
/// until the next drain rebuilds it.
const HNSW_DATA_MAGIC_V2: &[u8; 8] = b"INFDDG02";
const HNSW_DATA_MAGIC_V3: &[u8; 8] = b"INFDDG03";
/// `v04` appends the compact k→ef calibration curve as a trailing section after
/// the graph — a `u16` pair count then that many `(u32 k, u32 ef)` pairs — so a
/// query's requested `k` is served at its own calibrated beam (`v03` added the
/// persisted SQ8 walk plane; `v04` adds the curve). Older bundles carry a
/// single stamped `ef` and no curve; `decode_hnsw` reads the curve on `v04` and
/// synthesizes a degenerate 1-point curve (`ef_for_k(k) = ef_search` for every
/// `k`) on `v03`/`v02` — today's exact behavior, no forced rebuild.
const HNSW_DATA_MAGIC_V4: &[u8; 8] = b"INFDDG04";
/// All magics are 8 bytes; the shared byte width of the leading tag.
const HNSW_DATA_MAGIC_LEN: usize = HNSW_DATA_MAGIC_V4.len();
/// Byte size of the fixed frame of a data bundle: magic(8) + n(u64) + dim(u32)
/// + ef(u32) + col_len(u32) + graph_len(u64). The variable-length column name,
/// doc-id map, Sq16/SQ8 planes, and graph bytes are added on top; naming it
/// keeps the `encode_hnsw` capacity hint exact so the final `graph_len` extend
/// after the multi-GiB planes cannot trigger a full-buffer realloc at drain.
const HNSW_DATA_FIXED_BYTES: usize = HNSW_DATA_MAGIC_LEN + 8 + 4 + 4 + 4 + 8;

/// `v05` names the walk plane in the header (one `u8` after `ef`) and writes
/// only the sections that plane needs, rather than `v03`/`v04`'s always-present
/// SQ8 section. That is what lets a denser walk codec exist without every
/// bundle paying for a plane it never walks. It keeps `v04`'s trailing k→ef
/// curve unchanged, so the two format axes — which plane the walk scores, and
/// which beam each `k` is served at — compose rather than exclude each other.
///
/// The version is `05` and not a second `04` because both changes shipped
/// against `v03` independently: a `v04` bundle has no walk-codec byte and
/// always carries the SQ8 section, so reading one as though it named its plane
/// would take the column-name length out of the middle of the header and
/// mis-slice every section after it.
const HNSW_DATA_MAGIC_V5: &[u8; 8] = b"INFDDG05";

/// Which resident plane the graph walk scores candidates on.
///
/// A peer set, not a flag: each variant names a real stored representation,
/// the bundle header records which one its sections were written for, and
/// every one of them re-ranks its final beam on Sq16 (always present). So the
/// walk codec decides which candidates reach the beam and what each costs —
/// never the order returned. That is why a coarser walk plane buys latency
/// instead of costing recall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WalkCodec {
    /// No extra plane: walk the Sq16 codes directly (2 bytes/dim, and the only
    /// option that adds nothing to residency).
    Sq16,
    /// Derived int8 plane — the high byte of each Sq16 code (+1 byte/dim).
    /// Derivable from Sq16 on read, so selecting it never needs a rebuild.
    Sq8,
    /// Fitted 4-bit nibbles in rotated space (+0.5 bytes/dim). Not derivable
    /// on read — the fit needs a rotation and a moment pass over the corpus —
    /// so it is written at drain and applies at the next full rebuild.
    Sq4,
    /// [`Self::Sq4`] plus a second nibble plane carrying the sub-step residual
    /// (+1 byte/dim total).
    Sq4Residual,
}

impl WalkCodec {
    /// The codec a config selection asks for. Lives here rather than in
    /// `config` so the wire form and the knob cannot drift apart.
    pub(crate) fn from_config(plane: crate::config::VectorHnswPlane) -> Self {
        match plane {
            crate::config::VectorHnswPlane::Sq16 => WalkCodec::Sq16,
            crate::config::VectorHnswPlane::Sq8 => WalkCodec::Sq8,
            crate::config::VectorHnswPlane::Sq4 => WalkCodec::Sq4,
            crate::config::VectorHnswPlane::Sq4Residual => WalkCodec::Sq4Residual,
        }
    }

    /// Wire tag stored in the `v04` header.
    pub(crate) fn tag(self) -> u8 {
        match self {
            WalkCodec::Sq16 => 0,
            WalkCodec::Sq8 => 1,
            WalkCodec::Sq4 => 2,
            WalkCodec::Sq4Residual => 3,
        }
    }

    /// Inverse of [`Self::tag`]; `None` for a tag written by a newer build,
    /// which decodes to `None` and serves ivf rather than misreading a plane.
    pub(crate) fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            0 => Some(WalkCodec::Sq16),
            1 => Some(WalkCodec::Sq8),
            2 => Some(WalkCodec::Sq4),
            3 => Some(WalkCodec::Sq4Residual),
            _ => None,
        }
    }

    /// Whether this codec stores a 4-bit section (and so needs the rotation
    /// seed and ruler alongside it).
    pub(crate) fn is_sq4(self) -> bool {
        matches!(self, WalkCodec::Sq4 | WalkCodec::Sq4Residual)
    }

    /// Whether the 4-bit form carries the residual nibble plane.
    pub(crate) fn with_residual(self) -> bool {
        matches!(self, WalkCodec::Sq4Residual)
    }
}

/// A `hnsw` resident index rebuilt from a persisted bundle: the scorer
/// over the node-ordered code plane (whichever codec the bundle stamped),
/// the walkable graph, and the `node_index -> stable doc id` map.
pub(crate) struct HnswIndex {
    /// Sq16 plane: the refine plane every walk codec re-ranks its beam on, and
    /// the reference calibration grades against. Always present.
    pub scorer: Sq16Scorer,
    pub graph: Hnsw,
    pub doc_ids: Vec<i128>,
    pub dim: usize,
    /// Calibrated query beam stamped at drain — the recall@10 `ef` (a query
    /// knob, so it rides in the bundle header, not the graph structure). Always
    /// non-zero from the drain; a 0 (which cannot occur) degrades to `k`. Also
    /// the value the degenerate 1-point `ef_curve` carries for a pre-`v04`
    /// bundle (no per-`k` curve stamped).
    pub ef_search: usize,
    /// Compact k→ef calibration curve: ascending `(k_anchor, ef)` pairs,
    /// monotonic non-decreasing in `ef`. Stamped on a `v04` bundle so each
    /// query's requested `k` is served at its own calibrated beam; a pre-`v04`
    /// bundle decodes to a degenerate 1-point curve `[(u32::MAX, ef_search)]`
    /// (every `k` maps to the single stamped `ef`). Read via [`Self::ef_for_k`].
    pub ef_curve: Vec<(u32, u32)>,
    /// Vector column this graph was built for. A table can carry several
    /// same-dim vector columns; the serving path must reject a query on a
    /// different column (→ ivf) rather than silently answer it from this
    /// column's neighbors.
    pub column: String,
    /// Resident contiguous SQ8 walk plane (`n × dim` bytes) — the high byte of
    /// each Sq16 code. On a `v03` bundle it is a zero-copy slice of the mapped
    /// bundle (persisted as a section); on a `v02` bundle it is derived on read
    /// into owned heap. Empty when SQ8-walk serving is off, so it costs no
    /// memory when disabled and serving falls back to the Sq16 walk. Used by
    /// [`HnswIndex::search_sq8_refine`] for a cheap int8-VNNI walk; the exact
    /// ranking comes from the Sq16 refine over `scorer`.
    pub sq8_plane: Plane,
    /// Resident 4-bit walk plane with its fitted ruler, when the bundle was
    /// written for [`WalkCodec::Sq4`] / [`WalkCodec::Sq4Residual`]. A peer of
    /// [`Self::sq8_plane`]: the walk scores on it and the beam is re-ranked on
    /// `scorer`, so it trades latency-per-candidate, never returned order.
    /// `None` for any other codec, so it costs nothing when unused.
    pub sq4: Option<Sq4Scorer>,
    /// The codec this bundle's sections were WRITTEN for, straight from the
    /// header — not the one this particular decode asked for.
    ///
    /// The two differ whenever a caller requests a narrower view than the
    /// bundle stores, and the maintenance path does exactly that. It must
    /// not be inferred from the decoded planes: a decode that filtered them
    /// out is indistinguishable from a bundle that never had them, and an
    /// incremental drain that guesses wrong re-encodes the bundle without
    /// the plane it stored — destroying a fitted 4-bit plane and its ruler,
    /// which no later open can reconstruct.
    pub stored_walk: WalkCodec,
}

/// Serialize a `hnsw` index to a persistable byte bundle (`v04`): header
/// (including the walk codec), the `node -> stable doc id` map, the
/// node-ordered Sq16 code plane, the walk plane the codec names, and the graph
/// section. Every section is inline so the bundle is self-contained —
/// reopening needs nothing but these bytes, and a mapped bundle serves each
/// plane zero-copy.
///
/// Only the sections the codec needs are written. `v03` always wrote the SQ8
/// plane, which is affordable at 1 byte/dim but stops being so once denser
/// codecs exist: a bundle should not carry a plane it never walks. The Sq16
/// plane is always present because every codec re-ranks its beam on it.
///
/// `sq4` must be `Some` exactly when `walk` names a 4-bit codec; the caller
/// builds it from the same `sq16_codes` written here, so the two cannot
/// describe different rows.
pub(crate) fn encode_hnsw(
    sq16_codes: &[u8],
    doc_ids: &[i128],
    graph: &Hnsw,
    dim: usize,
    ef_search: usize,
    ef_curve: &[(u32, u32)],
    column: &str,
    walk: WalkCodec,
    sq4: Option<&Sq4Scorer>,
) -> Vec<u8> {
    let n = doc_ids.len();
    debug_assert_eq!(sq16_codes.len(), n * dim * 2);
    debug_assert_eq!(
        walk.is_sq4(),
        sq4.is_some(),
        "the 4-bit plane must be supplied exactly when the codec names it"
    );
    let graph_bytes = graph.to_bytes();
    let col = column.as_bytes();
    // Size the section this codec writes: SQ8 is `n × dim`; a 4-bit plane is
    // `n × ceil(dim/2)` per nibble plane plus the rotation seed and two f32
    // ruler vectors; an Sq16 walk writes nothing extra.
    let walk_len = match walk {
        WalkCodec::Sq16 => 0,
        WalkCodec::Sq8 => n * dim,
        WalkCodec::Sq4 | WalkCodec::Sq4Residual => {
            let planes = if walk.with_residual() { 2 } else { 1 };
            8 + dim * 2 * 4 + n * dim.div_ceil(2) * planes
        }
    };
    // The trailing k→ef curve: a u16 count then `(u32 k, u32 ef)` per pair.
    let curve_len = 2 + ef_curve.len() * 8;
    let mut out = Vec::with_capacity(
        HNSW_DATA_FIXED_BYTES
            + 1
            + col.len()
            + n * 16
            + sq16_codes.len()
            + walk_len
            + graph_bytes.len()
            + curve_len,
    );
    out.extend_from_slice(HNSW_DATA_MAGIC_V5);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    // Was reserved / alignment; now the stamped recall@10 query beam (u32).
    out.extend_from_slice(&(ef_search as u32).to_le_bytes());
    // Walk codec: names which section follows the Sq16 plane.
    out.push(walk.tag());
    // Stamped column name: length-prefixed UTF-8.
    out.extend_from_slice(&(col.len() as u32).to_le_bytes());
    out.extend_from_slice(col);
    for &id in doc_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out.extend_from_slice(sq16_codes);
    match walk {
        // Nothing extra: the walk scores the Sq16 plane already written.
        WalkCodec::Sq16 => {}
        // Derived from the Sq16 plane just written. Persisting it (rather than
        // deriving on read) lets a mapped bundle serve it zero-copy; the
        // derivation is shared with the read-time fallback so they can't drift.
        WalkCodec::Sq8 => extend_sq8_plane(&mut out, sq16_codes),
        // Seed and ruler first, then the nibble planes, so a reader can size
        // every plane read from `dim` alone before touching them.
        WalkCodec::Sq4 | WalkCodec::Sq4Residual => {
            let sq4 = sq4.expect("checked above: a 4-bit codec supplies its plane");
            let (codes, residual, offset, step) = sq4.parts();
            // The v04 reader sizes the ruler and both nibble planes from the
            // header's `dim`/`n` alone, so a plane whose own dimensions differ
            // from the header we just stamped produces a bundle where every
            // LATER section — residual plane, graph length, graph — is sliced
            // at the wrong offset, and decode reports no error at all. These
            // stay live in release: the cost is four integer compares once per
            // drain, and the failure they catch is a silently corrupt bundle.
            let stride = dim.div_ceil(2);
            assert_eq!(offset.len(), dim, "4-bit ruler offset length vs header dim");
            assert_eq!(step.len(), dim, "4-bit ruler step length vs header dim");
            assert_eq!(codes.len(), n * stride, "4-bit code plane length vs header");
            assert_eq!(
                residual.map(<[u8]>::len),
                walk.with_residual().then_some(n * stride),
                "4-bit residual plane presence/length vs header codec"
            );
            out.extend_from_slice(&sq4.rot_seed().to_le_bytes());
            for v in offset.iter().chain(step) {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(codes);
            if let Some(res) = residual {
                out.extend_from_slice(res);
            }
        }
    }
    out.extend_from_slice(&(graph_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&graph_bytes);
    // k→ef curve, appended AFTER the variable sections so the fixed-header
    // byte count is unaffected. Bounded to u16 pairs (a handful of anchors).
    out.extend_from_slice(&(ef_curve.len() as u16).to_le_bytes());
    for &(k, ef) in ef_curve {
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&ef.to_le_bytes());
    }
    out
}

/// Append the derived SQ8 walk plane — the high byte of each little-endian
/// `u16` Sq16 code — to `out`. The single source of truth for the derivation,
/// shared by [`encode_hnsw`] (persisting the v03 section) and
/// [`derive_sq8_plane`] (the v02 read-time fallback) so the two can't drift.
/// `chunks_exact(2)` elides the per-element bounds check on this hydration loop.
fn extend_sq8_plane(out: &mut Vec<u8>, sq16_plane: &[u8]) {
    out.extend(sq16_plane.chunks_exact(2).map(|w| w[1]));
}

/// The derived SQ8 walk plane as an owned buffer (`n × dim`). A pure function
/// of the Sq16 plane — used to serve a `v02` bundle that has no persisted SQ8
/// section.
fn derive_sq8_plane(sq16_plane: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sq16_plane.len() / 2);
    extend_sq8_plane(&mut out, sq16_plane);
    out
}

/// Rebuild a resident [`HnswIndex`] from [`encode_hnsw`].
/// Returns `None` on any malformation so the caller falls back to the lazy
/// build or scan path rather than failing the query.
///
/// `bundle` owns the encoded bytes. When it is the memory-mapped graph bundle
/// (the default local serving path), the Sq16 plane — and, on a `v03` bundle,
/// the SQ8 plane — are served as zero-copy [`Bytes::slice_ref`] slices of it,
/// with no heap copy, and the returned index keeps the mapping alive. A `v02`
/// bundle has no SQ8 section, so the SQ8 plane is derived on read into owned
/// heap when SQ8-walk serving is on.
///
/// `want` selects which walk plane becomes resident: `Some(codec)` for a
/// serving hydration, which is free to ask for a narrower view than the bundle
/// holds, or `None` for "whatever this bundle stored". Maintenance must pass
/// `None`: it re-encodes the bundle, so it has to see the plane that is
/// actually on disk rather than impose the running config's choice. Either way
/// [`HnswIndex::stored_walk`] reports what the header named, so a caller can
/// tell a filtered view from a bundle that never carried the section.
pub(crate) fn decode_hnsw(bundle: &Bytes, want: Option<WalkCodec>) -> Option<HnswIndex> {
    let bytes: &[u8] = bundle.as_ref();
    let mut c = Cursor::new(bytes);
    let magic = c.take(HNSW_DATA_MAGIC_LEN)?;
    // Two independent format axes, so three facts per magic:
    //   - `walk_tag`: does the header carry the walk-codec byte? `v05` only.
    //   - `implied_walk`: for the versions that predate that byte, the codec
    //     their layout implies — `v03`/`v04` always wrote the SQ8 section,
    //     `v02` wrote none, so SQ8 is derived on read.
    //   - `has_curve`: is the trailing k→ef curve present? `v04`/`v05`. Older
    //     bundles synthesize a degenerate 1-point curve from the single stamped
    //     `ef` below.
    // Any other tag (e.g. a legacy `01`) is unsupported → `None`, and the query
    // serves ivf until the next drain rebuilds.
    let (walk_tag, implied_walk, has_curve) = if magic == HNSW_DATA_MAGIC_V5 {
        (true, WalkCodec::Sq16, true)
    } else if magic == HNSW_DATA_MAGIC_V4 {
        (false, WalkCodec::Sq8, true)
    } else if magic == HNSW_DATA_MAGIC_V3 {
        (false, WalkCodec::Sq8, false)
    } else if magic == HNSW_DATA_MAGIC_V2 {
        (false, WalkCodec::Sq16, false)
    } else {
        return None;
    };
    let n = c.u64()? as usize;
    let dim = c.u32()? as usize;
    let ef_search = c.u32()? as usize; // 0 on older bundles (was reserved)
    if dim == 0 {
        return None;
    }
    // Which sections this bundle holds. Only `v05` states it; everything older
    // is implied by its layout, decided above.
    let stored = if walk_tag {
        WalkCodec::from_tag(c.u8()?)?
    } else {
        implied_walk
    };
    // A caller that passes `None` is asking for the bundle's own codec — see
    // the `want` paragraph above. Resolving it here, once, keeps every section
    // decision below reading a concrete codec.
    let want = want.unwrap_or(stored);
    let col_len = c.u32()? as usize;
    // Bound the column-name read against the bytes present before taking it.
    if col_len > c.remaining() {
        return None;
    }
    let column = String::from_utf8(c.take(col_len)?.to_vec()).ok()?;
    // Cross-check the doc-id block length (16 B/id, one i128 per node) against
    // the bytes present BEFORE reserving, so a corrupt `n` (e.g. ~2^60) cannot
    // drive a huge `with_capacity` that aborts under `handle_alloc_error` —
    // mirroring the guard `Hnsw::from_bytes` applies to its own wire lengths.
    if n.checked_mul(16)? > c.remaining() {
        return None;
    }
    let mut doc_ids = Vec::with_capacity(n);
    for _ in 0..n {
        doc_ids.push(c.i128()?);
    }
    // Sq16 plane as a zero-copy slice of `bundle`: `c.take` yields a subslice of
    // `bytes` (= `bundle.as_ref()`), so `slice_ref` recovers the owning `Bytes`
    // view without copying — a mapped bundle never copies the ~1.5 GiB plane
    // into this process's heap. Every walk codec re-ranks its beam on this
    // plane, so it is read unconditionally.
    let sq16_slice = c.take(n.checked_mul(dim)?.checked_mul(2)?)?;
    let sq16_plane = bundle.slice_ref(sq16_slice);
    // The walk section named by the header, sliced the same zero-copy way.
    //
    // Note the asymmetry in what a caller can ask for without a rebuild: SQ8 is
    // the Sq16 high byte, so wanting it always works — derived on read when the
    // bundle has no section. A 4-bit plane is not derivable (its fit needs a
    // rotation and a moment pass over the corpus), so wanting it on a bundle
    // written for another codec falls back to SQ8 until the next drain writes
    // one, rather than paying a full re-quantization on every open.
    let mut sq4: Option<Sq4Scorer> = None;
    let mut sq8_plane = Plane::Owned(Vec::new());
    match stored {
        WalkCodec::Sq16 => {
            if want != WalkCodec::Sq16 {
                sq8_plane = Plane::Owned(derive_sq8_plane(sq16_slice));
            }
        }
        WalkCodec::Sq8 => {
            // Consumed either way, so the graph section that follows stays
            // reachable; kept only when a walk plane is actually wanted.
            let sq8_slice = c.take(n.checked_mul(dim)?)?;
            if want != WalkCodec::Sq16 {
                sq8_plane = Plane::Shared(bundle.slice_ref(sq8_slice));
            }
        }
        WalkCodec::Sq4 | WalkCodec::Sq4Residual => {
            let rot_seed = c.u64()?;
            // Bound the ruler read (two f32 vectors) before taking it.
            if dim.checked_mul(2)?.checked_mul(4)? > c.remaining() {
                return None;
            }
            let offset = read_f32_le(c.take(dim.checked_mul(4)?)?);
            let step = read_f32_le(c.take(dim.checked_mul(4)?)?);
            let stride = dim.div_ceil(2);
            let plane_len = n.checked_mul(stride)?;
            let codes = bundle.slice_ref(c.take(plane_len)?);
            let residual = if stored.with_residual() {
                Some(Plane::Shared(bundle.slice_ref(c.take(plane_len)?)))
            } else {
                None
            };
            if want.is_sq4() {
                sq4 = Sq4Scorer::from_parts(
                    Plane::Shared(codes),
                    residual,
                    offset,
                    step,
                    rot_seed,
                    dim,
                    n,
                );
                if sq4.is_none() {
                    // A rejected 4-bit section (torn write, non-finite ruler)
                    // must not leave BOTH walk planes empty: that serves a
                    // full-width Sq16 walk — correct results at several times
                    // the configured walk bandwidth, and indistinguishable
                    // from healthy. Derive the SQ8 plane so the degradation is
                    // one rung rather than all the way, and say so, because
                    // nothing else about the served answers would reveal it.
                    tracing::warn!(
                        column = column.as_str(),
                        n,
                        dim,
                        "hnsw: stored 4-bit walk section rejected (shape or ruler); \
                         walking the derived SQ8 plane instead. The next full \
                         rebuild rewrites the section."
                    );
                    sq8_plane = Plane::Owned(derive_sq8_plane(sq16_slice));
                }
            } else if want != WalkCodec::Sq16 {
                sq8_plane = Plane::Owned(derive_sq8_plane(sq16_slice));
            }
        }
    }
    let graph_len = c.u64()? as usize;
    let graph_bytes = c.take(graph_len)?;
    let graph = Hnsw::from_bytes(graph_bytes)?;
    if graph.len() != n {
        return None;
    }
    // k→ef curve. On `v04` it is the trailing section after the graph; on an
    // older bundle (no curve) synthesize a degenerate 1-point curve mapping
    // every `k` to the single stamped `ef_search` — today's exact behavior.
    let ef_curve = if has_curve {
        let count = c.u16()? as usize;
        // Bound the pair read against the bytes present (8 B/pair) before
        // reserving, so a corrupt count cannot drive a huge allocation.
        if count.checked_mul(8)? > c.remaining() {
            return None;
        }
        let mut curve = Vec::with_capacity(count);
        for _ in 0..count {
            let k = c.u32()?;
            let ef = c.u32()?;
            curve.push((k, ef));
        }
        curve
    } else {
        Vec::new()
    };
    // An absent or empty curve degrades to the degenerate 1-point fallback so
    // `ef_for_k` always has a value to return.
    let ef_curve = if ef_curve.is_empty() {
        vec![(u32::MAX, ef_search as u32)]
    } else {
        ef_curve
    };
    let scorer = Sq16Scorer::from_plane(Plane::Shared(sq16_plane), dim, n);
    Some(HnswIndex {
        scorer,
        graph,
        doc_ids,
        dim,
        ef_search,
        ef_curve,
        column,
        sq8_plane,
        sq4,
        stored_walk: stored,
    })
}

/// Parse a little-endian `f32` vector from raw bytes. The caller bounds the
/// read first, so a short slice is a caller bug rather than a corrupt-input
/// path.
pub(crate) fn read_f32_le(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Prepared query for [`Sq8WalkScorer`]: the int8 query plus the constant
/// `128·Σq_i8` baseline subtracted at score time.
pub(crate) struct Sq8Query {
    q: Vec<i8>,
    /// `128·Σq_i8`. The stored code byte is unsigned in `[0, 255]`, centered
    /// near 128, so subtracting this baseline turns the raw dot `Σ code·q` into
    /// the centered dot `Σ(code−128)·q` — the same ranking (a per-query
    /// constant shift), but small enough to stay in f32's exact-integer range.
    /// The raw u8·i8 dot can reach `dim·255·127` (> 2^24 for `dim ≳ 519`, e.g.
    /// 768-dim), where distinct dots would collapse to the same f32 in the beam
    /// heap and mis-order candidates before the Sq16 refine sees them.
    baseline: i32,
}

/// Walk scorer over the resident SQ8 plane: ranks candidates by the int8-VNNI
/// dot `-Σ(code−128)·q_i8` (NegDot — lower is nearer). A coarse *navigation*
/// proxy; exact ranking is restored by the Sq16 refine in
/// [`HnswIndex::search_sq8_refine`].
pub(crate) struct Sq8WalkScorer<'a> {
    plane: &'a [u8],
    dim: usize,
    len: usize,
}

impl Sq8WalkScorer<'_> {
    #[inline]
    fn row(&self, node: u32) -> &[u8] {
        let s = node as usize * self.dim;
        &self.plane[s..s + self.dim]
    }
}

impl NodeScorer for Sq8WalkScorer<'_> {
    type Prepared = Sq8Query;
    fn len(&self) -> usize {
        self.len
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn prepare(&self, query: &[f32]) -> Sq8Query {
        let q = quantize_query_i8(query);
        let baseline = 128 * q.iter().map(|&x| x as i32).sum::<i32>();
        Sq8Query { q, baseline }
    }
    fn prepare_node(&self, node: u32) -> Sq8Query {
        // Node-as-query for build-time node-to-node distance; unused by the
        // search-only walk. Center the stored bytes into signed int8.
        let q: Vec<i8> = self
            .row(node)
            .iter()
            .map(|&c| (c as i16 - 128) as i8)
            .collect();
        let baseline = 128 * q.iter().map(|&x| x as i32).sum::<i32>();
        Sq8Query { q, baseline }
    }
    #[inline]
    fn score(&self, q: &Sq8Query, node: u32) -> f32 {
        // Centered dot: raw `Σ code·q` minus the `128·Σq` baseline — see
        // `Sq8Query::baseline` for why the centering is needed for f32 exactness.
        -((sq8_walk_dot(self.row(node), &q.q) - q.baseline) as f32)
    }
}

impl HnswIndex {
    /// The calibrated query beam for a requested `k`, read from the stamped
    /// k→ef curve: round `k` UP to the next anchor and return that anchor's
    /// `ef` (the minimal beam that cleared the recall target there). A `k`
    /// above the top anchor clamps to the top anchor's `ef` (the widest
    /// calibrated beam — no measured-recall promise beyond the anchors). A
    /// degenerate 1-point curve
    /// (a pre-`v04` bundle) returns its single stamped `ef` for every `k`.
    pub(crate) fn ef_for_k(&self, k: usize) -> usize {
        for &(anchor_k, ef) in &self.ef_curve {
            if anchor_k as usize >= k {
                return ef as usize;
            }
        }
        // `k` past the widest anchor: clamp to the widest anchor's ef.
        self.ef_curve
            .last()
            .map(|&(_, ef)| ef as usize)
            .unwrap_or(self.ef_search)
    }

    /// SQ8 walk + Sq16 refine: navigate the graph scoring the cheap int8 plane
    /// (returning the top `refine_k` by SQ8), then re-score those on the exact
    /// Sq16 plane and return the true top-`k`. Reader-side; no bundle change.
    /// Walk on `walk`, then re-rank the beam on Sq16.
    ///
    /// The codec-independent half of a coarse-plane search: the walk decides
    /// which candidates are considered, and this restores their exact order,
    /// so a coarser plane costs candidates examined rather than ranking
    /// quality. Every walk codec other than Sq16 goes through here.
    pub(crate) fn search_walk_refine<S: NodeScorer>(
        &self,
        walk: &S,
        query: &[f32],
        k: usize,
        ef: usize,
        refine_k: usize,
    ) -> Vec<(u32, f32)> {
        // Refine the top `refine_k` of the beam on Sq16, clamped to `[k, ef]`:
        // at least `k` (so there are enough to return) and at most the beam
        // width (nothing beyond it was explored).
        let shortlist = refine_k.max(k).min(ef);
        let beam = self.graph.search(walk, query, shortlist, ef);
        let prepared = self.scorer.prepare(query);
        let mut refined: Vec<(u32, f32)> = beam
            .into_iter()
            .map(|(node, _)| (node, self.scorer.score(&prepared, node)))
            .collect();
        refined.sort_by(|a, b| a.1.total_cmp(&b.1));
        refined.truncate(k);
        refined
    }

    pub(crate) fn search_sq8_refine(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        refine_k: usize,
    ) -> Vec<(u32, f32)> {
        let sq8 = Sq8WalkScorer {
            plane: self.sq8_plane.bytes(),
            dim: self.dim,
            len: self.graph.len(),
        };
        // Refine the top `refine_k` of the SQ8 beam on Sq16, clamped to
        // `[k, ef]`: at least `k` (so there are enough to return) and at most
        // the beam width `ef` (nothing beyond it was explored).
        let shortlist = refine_k.max(k).min(ef);
        let beam = self.graph.search(&sq8, query, shortlist, ef);
        let prepared = self.scorer.prepare(query);
        let mut refined: Vec<(u32, f32)> = beam
            .into_iter()
            .map(|(node, _)| (node, self.scorer.score(&prepared, node)))
            .collect();
        refined.sort_by(|a, b| a.1.total_cmp(&b.1));
        refined.truncate(k);
        refined
    }
}

/// On-disk magic for the combined graph bundle (one slow-state section
/// object holding the centroid graph and, at ≤ N scale, the data bundle).
const GRAPH_BUNDLE_MAGIC: &[u8; 8] = b"INFVGB01";

/// Fixed byte offset of the population key inside a graph bundle: right
/// after the 8-byte magic. One `u64` digest of the live doc-id population
/// the graph covers (repack-invariant, delete-sensitive — computed by the
/// supertable layer).
const GRAPH_BUNDLE_KEY_OFF: usize = GRAPH_BUNDLE_MAGIC.len();
/// Fixed byte offset of the high-water stable id: right after the key. The
/// largest doc id the graph covers, so the next drain knows where the
/// append delta starts (`stable_id > high_water`) for an incremental
/// insert instead of a full rebuild.
const GRAPH_BUNDLE_HIGH_WATER_OFF: usize = GRAPH_BUNDLE_KEY_OFF + 8;
/// Byte length of the header a settle read needs: magic + key(u64) +
/// high-water(i128). A single small range GET recovers both without
/// fetching the multi-GiB body.
pub(crate) const GRAPH_BUNDLE_HEADER_BYTES: usize = GRAPH_BUNDLE_MAGIC.len() + 8 + 16;

/// The graph sections carried in one slow-state blob, as raw bytes, plus
/// the population key and high-water id they cover. `centroid_graph` is a
/// bare [`Hnsw::to_bytes`] over the fp32 fine centroids (present at any
/// scale). `data_bundle` is an [`encode_hnsw`] payload (graph +
/// Sq16 plane + node→stable-doc-id map), present only when the table's doc
/// count is within the data-graph scale ceiling. Full-projection queries
/// resolve each hit's stable id to its live `(superfile, local)` through
/// the engine's existing id→placement resolver, so no per-node physical
/// provenance is baked in (which would go stale on a compaction repack).
pub(crate) struct ResidentIndexEnvelope {
    /// One `u64` digest of the covered doc-id population (opaque here; the
    /// supertable layer defines it).
    pub population_key: u64,
    /// Largest stable doc id the graph covers (the append-delta boundary).
    pub high_water_id: i128,
    pub centroid_graph: Vec<u8>,
    /// The [`encode_hnsw`] payload, as a zero-copy slice of the input `Bytes`
    /// (`raw.slice_ref`). When the input is the mapped bundle this is the
    /// multi-GiB data section with no heap copy — the open-time transient the
    /// mmap serving path removes; [`decode_hnsw`] then slices the Sq16 and SQ8
    /// planes straight out of it.
    pub data_bundle: Option<(PayloadKind, Bytes)>,
}

/// Read the `(population_key, high_water_id)` header from a bundle's first
/// [`GRAPH_BUNDLE_HEADER_BYTES`] bytes. `None` on a bad magic or a short
/// read. Lets the settle path key on the covered population — and find the
/// append boundary — via a tiny range GET instead of the whole object.
pub(crate) fn resident_envelope_header(header: &[u8]) -> Option<(u64, i128)> {
    if header.len() < GRAPH_BUNDLE_HEADER_BYTES
        || &header[..GRAPH_BUNDLE_MAGIC.len()] != GRAPH_BUNDLE_MAGIC
    {
        return None;
    }
    let key = u64::from_le_bytes(
        header[GRAPH_BUNDLE_KEY_OFF..GRAPH_BUNDLE_KEY_OFF + 8]
            .try_into()
            .ok()?,
    );
    let high_water = i128::from_le_bytes(
        header[GRAPH_BUNDLE_HIGH_WATER_OFF..GRAPH_BUNDLE_HIGH_WATER_OFF + 16]
            .try_into()
            .ok()?,
    );
    Some((key, high_water))
}

/// Length-prefixed opaque section (`0` len flag when absent).
/// Which index the envelope's payload section holds.
///
/// The envelope STATES this rather than leaving a reader to sniff the payload's
/// magic. Sniffing works but inverts the responsibility: every reader would
/// have to try each format in turn and treat "none matched" as corruption,
/// which gets worse with each kind and is silent when it guesses wrong.
///
/// The tag lives in the section's existing presence byte — `0` was already
/// "absent" and `1` "present", and `1` has only ever meant a graph — so an
/// envelope written before this tag existed reads back as [`Self::Graph`],
/// which is what it is. A binary that predates a kind sees a present section,
/// fails to decode it as a graph, and serves the ivf scan: a downgrade, not a
/// mis-parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PayloadKind {
    /// An `encode_hnsw` payload: graph + Sq16 plane + walk plane.
    Graph = 1,
    /// A flat 4-bit index payload: nibble plane + ruler, no graph, no Sq16.
    Flat = 2,
}

impl PayloadKind {
    fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(PayloadKind::Graph),
            2 => Some(PayloadKind::Flat),
            _ => None,
        }
    }

    /// The kind's name for a log line, so a message about a published or
    /// hydrated section says which index it was.
    pub(crate) fn label(self) -> &'static str {
        match self {
            PayloadKind::Graph => "hnsw",
            PayloadKind::Flat => "flat",
        }
    }
}

fn put_opt_section(out: &mut Vec<u8>, section: Option<(PayloadKind, &[u8])>) {
    match section {
        Some((kind, bytes)) => {
            out.push(kind as u8);
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        None => out.push(0),
    }
}

/// Frame the graph sections into one slow-state blob, stamping the
/// `(high_water_id, count)` watermark into the fixed-offset header. The
/// data bundle and its provenance are omitted (a `0` flag) above the
/// data-graph scale ceiling.
pub(crate) fn encode_resident_envelope(
    population_key: u64,
    high_water_id: i128,
    centroid_graph: &[u8],
    payload: Option<(PayloadKind, &[u8])>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(GRAPH_BUNDLE_HEADER_BYTES + 16 + centroid_graph.len());
    out.extend_from_slice(GRAPH_BUNDLE_MAGIC);
    out.extend_from_slice(&population_key.to_le_bytes());
    out.extend_from_slice(&high_water_id.to_le_bytes());
    out.extend_from_slice(&(centroid_graph.len() as u64).to_le_bytes());
    out.extend_from_slice(centroid_graph);
    put_opt_section(&mut out, payload);
    out
}

/// Parse an [`encode_resident_envelope`] blob into its raw sections + header.
/// `None` on a bad magic or truncation, so a corrupt bundle degrades to a
/// fallback.
///
/// `mmap_backed` picks how the large `data_bundle` section is returned:
/// - `true` (the local serving path, `raw` is the shared file mmap): a
///   zero-copy `raw.slice_ref` — the multi-GiB payload never touches heap and
///   the returned index keeps the one mapping alive.
/// - `false` (remote/object-store, `raw` is a striped *heap* blob): the data
///   section is copied out into its own `Bytes`, so `raw` — which also holds
///   the centroid section and framing — is freed once this returns instead of
///   being pinned for the index's whole lifetime.
///
/// The small `centroid_graph` (present at every scale, modest size) is always
/// copied out into owned heap.
pub(crate) fn decode_resident_envelope(
    raw: &Bytes,
    mmap_backed: bool,
) -> Option<ResidentIndexEnvelope> {
    let bytes: &[u8] = raw.as_ref();
    let mut c = Cursor::new(bytes);
    if c.take(GRAPH_BUNDLE_MAGIC.len())? != GRAPH_BUNDLE_MAGIC {
        return None;
    }
    let population_key = c.u64()?;
    let high_water_id = c.i128()?;
    let centroid_len = c.u64()? as usize;
    let centroid_graph = c.take(centroid_len)?.to_vec();
    // Optional length-prefixed data-bundle section (`0` flag ⇒ absent). `c.take`
    // yields a subslice of `bytes` (= `raw`): on the mmap path `slice_ref`
    // shares that mapping zero-copy; on the heap path we copy it out so `raw`
    // (the full striped blob) frees.
    let tag = c.take(1)?[0];
    let data_bundle = if tag == 0 {
        None
    } else {
        // An unknown kind is a payload written by a newer build. Decline the
        // SECTION rather than the whole envelope: the centroid graph above it
        // is still valid and still worth serving.
        let kind = PayloadKind::from_tag(tag);
        let len = c.u64()? as usize;
        let section = c.take(len)?;
        kind.map(|kind| {
            let bytes = if mmap_backed {
                raw.slice_ref(section)
            } else {
                Bytes::copy_from_slice(section)
            };
            (kind, bytes)
        })
    };
    Some(ResidentIndexEnvelope {
        population_key,
        high_water_id,
        centroid_graph,
        data_bundle,
    })
}

/// The mutable graph entry point during a concurrent build: the current
/// tallest node and its top level. Read at the start of every insert (to
/// pick a descent origin) and promoted only when a taller node lands.
#[derive(Clone, Copy)]
struct EntryState {
    node: u32,
    top_level: u32,
}

/// Shared, lock-guarded scratch graph for a concurrent [`Hnsw::build`].
/// Each node's adjacency is behind its own `Mutex`, so independent inserts
/// touching different nodes never contend; the entry point is an `RwLock`
/// (read on every insert, written only on a rare promotion). Finalized
/// into a plain immutable [`Hnsw`] once every insert completes.
struct ParBuild {
    adj: Vec<Mutex<Vec<Vec<u32>>>>,
    /// Immutable after the pre-pass — read without locking.
    node_level: Vec<u32>,
    entry: RwLock<EntryState>,
    m: usize,
    m0: usize,
    ef_construction: usize,
}

impl ParBuild {
    /// Clone `node`'s neighbor list at `level` out from under its lock, so
    /// the (expensive) scoring of those neighbors happens lock-free.
    #[inline]
    fn snapshot(&self, node: u32, level: u32) -> Vec<u32> {
        let guard = self.adj[node as usize]
            .lock()
            .expect("invariant: hnsw adjacency lock never poisoned");
        let l = level as usize;
        if l < guard.len() {
            guard[l].clone()
        } else {
            Vec::new()
        }
    }

    /// Width-1 greedy descent at `level`, reading neighbor lists through
    /// [`snapshot`](Self::snapshot).
    fn greedy_nearest<S: NodeScorer>(
        &self,
        scorer: &S,
        prepared: &S::Prepared,
        entry: u32,
        level: u32,
    ) -> u32 {
        let mut best = entry;
        let mut best_d = scorer.score(prepared, entry);
        loop {
            let mut improved = false;
            for nb in self.snapshot(best, level) {
                let d = scorer.score(prepared, nb);
                if d < best_d {
                    best_d = d;
                    best = nb;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        best
    }

    /// `ef`-width beam at one `level`, reading neighbor lists through
    /// [`snapshot`](Self::snapshot). Same beam discipline as
    /// [`Hnsw::search_layer`]; returns candidates sorted nearest-first.
    fn search_layer<S: NodeScorer>(
        &self,
        scorer: &S,
        prepared: &S::Prepared,
        entry_points: &[u32],
        ef: usize,
        level: u32,
        visited: &mut VisitedSet,
    ) -> Vec<Scored> {
        visited.clear();
        let mut cand: BinaryHeap<Reverse<Scored>> = BinaryHeap::new();
        let mut result: BinaryHeap<Scored> = BinaryHeap::new();
        for &ep in entry_points {
            if visited.test_and_set(ep) {
                continue;
            }
            let d = scorer.score(prepared, ep);
            let s = Scored { dist: d, node: ep };
            cand.push(Reverse(s));
            result.push(s);
            if result.len() > ef {
                result.pop();
            }
        }
        while let Some(Reverse(c)) = cand.pop() {
            let farthest = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
            if c.dist > farthest && result.len() >= ef {
                break;
            }
            for nb in self.snapshot(c.node, level) {
                if visited.test_and_set(nb) {
                    continue;
                }
                let d = scorer.score(prepared, nb);
                let farthest = result.peek().map(|s| s.dist).unwrap_or(f32::INFINITY);
                if result.len() < ef || d < farthest {
                    let s = Scored { dist: d, node: nb };
                    cand.push(Reverse(s));
                    result.push(s);
                    if result.len() > ef {
                        result.pop();
                    }
                }
            }
        }
        let mut out: Vec<Scored> = result.into_vec();
        out.sort_unstable();
        out
    }

    /// Wire `node <-> selected` at `level` under the fine-grained locks.
    /// Each side takes one node lock at a time (never two at once, so no
    /// lock-order deadlock).
    ///
    /// Both the forward list and each reverse link are **merged** into the
    /// existing adjacency under the lock — never overwritten. A concurrent
    /// insert may already have spliced a reverse edge onto this node's
    /// forward list, so blindly assigning `selected` would silently drop
    /// those edges and shred graph connectivity (measured as recall
    /// collapse at scale). On overflow the list is re-pruned with the SAME
    /// diversity heuristic, not a plain keep-closest-M truncation — plain
    /// keep-M collapses hub diversity on clustered data and strands
    /// small-beam walks. The scorer is read-only (no graph locks), so
    /// scoring while holding a node lock cannot re-enter another lock.
    fn connect<S: NodeScorer>(
        &self,
        scorer: &S,
        node: u32,
        selected: &[u32],
        level: u32,
        cap: usize,
    ) {
        let li = level as usize;
        self.link_into(scorer, node, selected, li, cap);
        for &nb in selected {
            self.link_into(scorer, nb, &[node], li, cap);
        }
    }

    /// Merge `additions` into `target`'s neighbor list at level `li`
    /// (dedup), then heuristic-shrink if the merged list exceeds `cap`. All
    /// under `target`'s lock, so it composes safely with concurrent merges
    /// onto the same node.
    fn link_into<S: NodeScorer>(
        &self,
        scorer: &S,
        target: u32,
        additions: &[u32],
        li: usize,
        cap: usize,
    ) {
        let mut g = self.adj[target as usize]
            .lock()
            .expect("invariant: hnsw adjacency lock never poisoned");
        for &a in additions {
            if a != target && !g[li].contains(&a) {
                g[li].push(a);
            }
        }
        if g[li].len() > cap {
            let current = g[li].clone();
            let prep = scorer.prepare_node(target);
            let cands: Vec<Scored> = current
                .iter()
                .map(|&x| Scored {
                    node: x,
                    dist: scorer.score(&prep, x),
                })
                .collect();
            g[li] = select_neighbors_heuristic(scorer, cands, cap);
        }
    }

    /// Insert one node into the shared graph: snapshot the entry point,
    /// descend the upper layers with a width-1 beam, then run the
    /// `ef_construction` beam and connect on each layer at/below the node's
    /// top level. Promotes the node to entry point if it is taller than the
    /// one seen at snapshot time.
    fn insert<S: NodeScorer>(&self, scorer: &S, node: u32, visited: &mut VisitedSet) {
        let level = self.node_level[node as usize];
        let prepared = scorer.prepare_node(node);
        let EntryState {
            node: mut ep,
            top_level: entry_level,
        } = *self
            .entry
            .read()
            .expect("invariant: hnsw entry lock never poisoned");

        let mut l = entry_level;
        while l > level {
            ep = self.greedy_nearest(scorer, &prepared, ep, l);
            l -= 1;
        }

        let mut entry_points = vec![ep];
        let top = level.min(entry_level);
        for l in (0..=top).rev() {
            let found = self.search_layer(
                scorer,
                &prepared,
                &entry_points,
                self.ef_construction,
                l,
                visited,
            );
            let cap = if l == 0 { self.m0 } else { self.m };
            let selected = select_neighbors_heuristic(scorer, found.clone(), cap);
            self.connect(scorer, node, &selected, l, cap);
            entry_points = found.into_iter().map(|s| s.node).collect();
            if entry_points.is_empty() {
                entry_points.push(ep);
            }
        }

        if level > entry_level {
            let mut e = self
                .entry
                .write()
                .expect("invariant: hnsw entry lock never poisoned");
            // Re-check under the write lock: another worker may have promoted
            // a still-taller node between the snapshot and here.
            if level > e.top_level {
                e.node = node;
                e.top_level = level;
            }
        }
    }
}

/// Malkov/Yashunin diversity heuristic (Algorithm 4, core form). Walk
/// candidates nearest-first; keep `e` only if it is closer to the target
/// than to every already-kept node, so the kept set spreads across
/// directions instead of clumping on the single nearest cluster. This is
/// what preserves long-range hub edges that a pure nearest-M would drop.
fn select_neighbors_heuristic<S: NodeScorer>(
    scorer: &S,
    mut candidates: Vec<Scored>,
    m: usize,
) -> Vec<u32> {
    candidates.sort_unstable();
    let mut selected: Vec<u32> = Vec::with_capacity(m);
    for cand in candidates {
        if selected.len() >= m {
            break;
        }
        let prep_e = scorer.prepare_node(cand.node);
        let mut keep = true;
        for &r in &selected {
            // `cand.dist` is e→target; `d_er` is e→already-kept r.
            let d_er = scorer.score(&prep_e, r);
            if d_er < cand.dist {
                keep = false;
                break;
            }
        }
        if keep {
            selected.push(cand.node);
        }
    }
    selected
}

/// Sequential reference build, retained only to anchor the timed
/// serial-vs-parallel comparison test. Same insertion algorithm the
/// parallel [`Hnsw::build`] runs, without the per-node locking — so it is
/// also the deterministic build the equality-sensitive tests use.
#[cfg(test)]
impl Hnsw {
    fn build_serial<S: NodeScorer>(scorer: &S, params: HnswParams) -> Hnsw {
        let n = scorer.len();
        let mut g = Hnsw {
            neighbors: Vec::with_capacity(n),
            node_level: Vec::with_capacity(n),
            entry: 0,
            m: params.m,
            m0: params.m0,
            ef_construction: params.ef_construction,
            len: n,
        };
        if n == 0 {
            return g;
        }
        let ml = 1.0 / (params.m.max(2) as f64).ln();
        let mut visited = VisitedSet::new(n);
        for node in 0..n as u32 {
            let level = assign_level(params.seed, node, ml);
            g.insert_serial(scorer, node, level, &mut visited);
        }
        g
    }

    fn insert_serial<S: NodeScorer>(
        &mut self,
        scorer: &S,
        node: u32,
        level: u32,
        visited: &mut VisitedSet,
    ) {
        self.neighbors.push(vec![Vec::new(); level as usize + 1]);
        self.node_level.push(level);
        if self.node_level.len() == 1 {
            self.entry = node;
            return;
        }
        let prepared = scorer.prepare_node(node);
        let entry_level = self.node_level[self.entry as usize];
        let mut ep = self.entry;
        let mut l = entry_level;
        while l > level {
            ep = self.greedy_nearest(scorer, &prepared, ep, l);
            l -= 1;
        }
        let mut entry_points = vec![ep];
        let top = level.min(entry_level);
        for l in (0..=top).rev() {
            let found = self.search_layer(
                scorer,
                &prepared,
                &entry_points,
                self.ef_construction,
                l,
                visited,
            );
            let cap = if l == 0 { self.m0 } else { self.m };
            let selected = select_neighbors_heuristic(scorer, found.clone(), cap);
            self.connect_serial(scorer, node, &selected, l, cap);
            entry_points = found.into_iter().map(|s| s.node).collect();
            if entry_points.is_empty() {
                entry_points.push(ep);
            }
        }
        if level > entry_level {
            self.entry = node;
        }
    }

    fn connect_serial<S: NodeScorer>(
        &mut self,
        scorer: &S,
        node: u32,
        selected: &[u32],
        level: u32,
        cap: usize,
    ) {
        let li = level as usize;
        self.neighbors[node as usize][li] = selected.to_vec();
        for &nb in selected {
            let over = {
                let list = &mut self.neighbors[nb as usize][li];
                list.push(node);
                list.len() > cap
            };
            if over {
                let current = self.neighbors[nb as usize][li].clone();
                let prep_nb = scorer.prepare_node(nb);
                let cands: Vec<Scored> = current
                    .iter()
                    .map(|&x| Scored {
                        node: x,
                        dist: scorer.score(&prep_nb, x),
                    })
                    .collect();
                self.neighbors[nb as usize][li] = select_neighbors_heuristic(scorer, cands, cap);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed rotation seed for every Sq4 test plane.
    const TEST_ROT_SEED: u64 = 0x7E57_5EED;

    /// Deterministic uniform in [0, 1) from a mutable SplitMix64 state.
    fn next_unit(state: &mut u64) -> f32 {
        (splitmix64(state) >> 40) as f32 / ((1u64 << 24) as f32)
    }

    /// A batch of deterministic unit vectors of dimension `dim`.
    fn random_unit_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| next_unit(&mut state) * 2.0 - 1.0)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                for x in &mut v {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    /// Exhaustive nearest-`k` node ids under a scorer, for recall truth.
    fn brute_force<S: NodeScorer>(scorer: &S, query: &[f32], k: usize) -> Vec<u32> {
        let prepared = scorer.prepare(query);
        let mut all: Vec<Scored> = (0..scorer.len() as u32)
            .map(|n| Scored {
                node: n,
                dist: scorer.score(&prepared, n),
            })
            .collect();
        all.sort_unstable();
        all.into_iter().take(k).map(|s| s.node).collect()
    }

    /// Generic top-`k` over any scorer — its existence is the proof the
    /// graph is codec-agnostic (it is instantiated with both scorers).
    fn graph_topk<S: NodeScorer + Sync>(
        scorer: &S,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Vec<(u32, f32)> {
        let hnsw = Hnsw::build(scorer, HnswParams::default());
        hnsw.search(scorer, query, k, ef)
    }

    /// Build an Sq16 graph over ~2000 unit vectors and check graph
    /// recall@10 against exhaustive Sq16 search (same distance, so this
    /// isolates graph quality from quantization) is at least 0.9.
    #[test]
    fn sq16_graph_recall_at_10() {
        let dim = 32;
        let n = 2000;
        let vectors = random_unit_vectors(n, dim, 0xA11CE);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let hnsw = Hnsw::build(&scorer, HnswParams::default());
        assert_eq!(hnsw.len(), n);

        let queries = random_unit_vectors(50, dim, 0xB0B);
        let k = 10;
        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: std::collections::HashSet<u32> =
                brute_force(&scorer, q, k).into_iter().collect();
            let got = hnsw.search(&scorer, q, k, 64);
            for (node, _) in got {
                if truth.contains(&node) {
                    hit += 1;
                }
            }
            total += k;
        }
        let recall = hit as f64 / total as f64;
        eprintln!("sq16 graph recall@10 = {recall:.4}");
        assert!(recall >= 0.9, "sq16 recall@10 = {recall:.3} (< 0.9)");
    }

    /// Deterministic clustered corpus: `n_cent` near-orthogonal unit
    /// centers, each doc = a center plus small per-dim noise, renormalized.
    /// Mirrors the synthetic vector bench's planted-cluster structure so we
    /// can study graph quality on well-separated clusters.
    fn clustered_unit_vectors(
        n: usize,
        n_cent: usize,
        dim: usize,
        noise: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let mut state = seed;
        let renorm = |v: &mut Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            for x in v.iter_mut() {
                *x /= norm;
            }
        };
        let centers: Vec<Vec<f32>> = (0..n_cent)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim)
                    .map(|_| next_unit(&mut state) * 2.0 - 1.0)
                    .collect();
                renorm(&mut v);
                v
            })
            .collect();
        (0..n)
            .map(|i| {
                let c = &centers[i % n_cent];
                let mut v: Vec<f32> = c
                    .iter()
                    .map(|&cv| cv + (next_unit(&mut state) * 2.0 - 1.0) * noise)
                    .collect();
                renorm(&mut v);
                v
            })
            .collect()
    }

    /// The calibrator returns a registering `(m0, ef)` that clears the target
    /// on a corpus where the graph can, and picks from the candidate sets.
    #[test]
    fn calibrate_graph_picks_registering_choice() {
        let dim = 128;
        let n = 5000;
        let vectors = clustered_unit_vectors(n, 32, dim, 0.3, 0x0CA_11B);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let (choice, curve, graph) = calibrate_graph(
            &scorer,
            &scorer,
            &[32, 64, 128],
            &[128, 256, 512],
            0.90,
            0.01,
            200,
            100,
            10,
            0x5EED,
            /* want_curve */ true,
        );
        eprintln!(
            "[calib] m0={} ef={} recall={:.3} registered={} at_target={} curve={curve:?}",
            choice.m0, choice.ef, choice.recall, choice.registered, choice.at_target
        );
        assert!(
            choice.registered,
            "should register; got recall {:.3}",
            choice.recall
        );
        let graph = graph.expect("registered ⇒ pruned graph returned");
        assert_eq!(
            graph.base_degree(),
            choice.m0,
            "persisted graph pruned to chosen m0"
        );
        assert_eq!(graph.len(), n, "graph covers all rows");
        assert!(
            [32, 64, 128].contains(&choice.m0),
            "m0 {} not a candidate",
            choice.m0
        );
        assert!(
            [128, 256, 512].contains(&choice.ef),
            "ef {} not a candidate",
            choice.ef
        );
        // A dim-128 clustered corpus should be reachable ⇒ at_target.
        assert!(
            choice.at_target,
            "expected to clear 0.90; got {:.3}",
            choice.recall
        );
        // A registered graph stamps a k→ef curve: one pair per anchor, its
        // anchors ARE the calibrator's anchors, ascending in `k`, and its `ef`
        // is monotonic non-decreasing (a larger `k` never asks for a narrower
        // beam than a smaller one). In particular k=100's ef ≥ k=10's ef.
        assert_eq!(
            curve.len(),
            HNSW_CALIB_K_ANCHORS.len(),
            "one curve pair per anchor"
        );
        for (i, &(k_anchor, _)) in curve.iter().enumerate() {
            assert_eq!(
                k_anchor as usize, HNSW_CALIB_K_ANCHORS[i],
                "curve anchor matches the calibrator's anchor"
            );
        }
        for w in curve.windows(2) {
            assert!(w[0].0 < w[1].0, "curve anchors strictly ascending in k");
            assert!(
                w[0].1 <= w[1].1,
                "curve ef monotonic non-decreasing in k: {:?} then {:?}",
                w[0],
                w[1]
            );
        }
        let ef_at = |k: usize| {
            curve
                .iter()
                .find(|&&(a, _)| a as usize == k)
                .map(|&(_, e)| e)
        };
        assert!(
            ef_at(100) >= ef_at(10),
            "k=100 ef {:?} must be ≥ k=10 ef {:?}",
            ef_at(100),
            ef_at(10)
        );
    }

    /// The same generic build/search satisfies the trait for both the
    /// Sq16 and the Fp32 reference scorer, and each finds an exact stored
    /// vector as its own nearest neighbor.
    #[test]
    fn both_scorers_satisfy_trait() {
        let dim = 16;
        let n = 500;
        let vectors = random_unit_vectors(n, dim, 0xC0FFEE);

        let sq16 = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let fp32 = Fp32Scorer::from_vectors(&vectors, dim);

        // Query with a stored vector: it must come back as node 0's rank.
        let probe = &vectors[123];

        let sq16_top = graph_topk(&sq16, probe, 5, 64);
        let fp32_top = graph_topk(&fp32, probe, 5, 64);

        assert_eq!(sq16_top.len(), 5);
        assert_eq!(fp32_top.len(), 5);

        // Both codecs recover the exact stored vector for a self-query. The
        // parallel build isn't bit-identical run to run, so assert membership
        // in the top handful rather than a strict rank-0 (recall-stable, not
        // order-exact).
        assert!(
            fp32_top.iter().any(|(node, _)| *node == 123),
            "fp32 top-5 for a stored vector should contain it: {fp32_top:?}"
        );
        assert!(
            sq16_top.iter().any(|(node, _)| *node == 123),
            "sq16 top-5 for a stored vector should contain it: {sq16_top:?}"
        );

        // Distances come back sorted ascending for both codecs.
        for top in [&sq16_top, &fp32_top] {
            assert!(
                top.windows(2).all(|w| w[0].1 <= w[1].1),
                "not ascending: {top:?}"
            );
        }
    }

    /// The `from_codes` path — adopting an already-encoded flat Sq16 code
    /// buffer (exactly what `build_hnsw_index` feeds from the on-disk
    /// `full[]` plane) — must produce a graph identical to encoding the same
    /// vectors through `from_unit_vectors`. This pins the resident-index
    /// build's code path: raw Sq16 bytes in, same search out.
    #[test]
    fn from_codes_matches_from_unit_vectors() {
        use crate::superfile::vector::distance::encode_sq16_row;
        let dim = 24;
        let n = 800;
        let vectors = random_unit_vectors(n, dim, 0xD1_5EA5E);

        // Path A: encode inside the scorer.
        let a = Sq16Scorer::from_unit_vectors(&vectors, dim);

        // Path B: pre-encode a flat `n × dim × 2` buffer (as the on-disk
        // plane is laid out) and adopt it verbatim.
        let stride = dim * 2;
        let mut codes = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        let b = Sq16Scorer::from_codes(codes, dim, n);

        // The parallel build is not bit-identical run to run, so compare the
        // two scorers by their deterministic exhaustive rankings instead of
        // two graphs: identical brute-force top-k for every query means the
        // adopted-bytes scorer scores byte-for-byte like the encode-inside
        // scorer, which is the actual `from_codes` contract.
        let queries = random_unit_vectors(20, dim, 0xF00D);
        for q in &queries {
            let ra = brute_force(&a, q, 10);
            let rb = brute_force(&b, q, 10);
            assert_eq!(ra, rb, "from_codes scorer diverged from from_unit_vectors");
        }
    }

    /// A graph survives `to_bytes` → `from_bytes` byte-for-byte in
    /// behavior: the restored graph gives identical search results (the
    /// adjacency, entry, and levels are reconstructed exactly).
    #[test]
    fn graph_bytes_roundtrip() {
        let dim = 32;
        let n = 1500;
        let vectors = random_unit_vectors(n, dim, 0x6A47);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let graph = Hnsw::build(&scorer, HnswParams::default());

        let bytes = graph.to_bytes();
        let restored = Hnsw::from_bytes(&bytes).expect("decode graph");
        assert_eq!(restored.len(), graph.len());

        let queries = random_unit_vectors(25, dim, 0x9B2E);
        for q in &queries {
            assert_eq!(
                graph.search(&scorer, q, 10, 64),
                restored.search(&scorer, q, 10, 64),
                "restored graph search diverged"
            );
        }
        // A corrupt/short section decodes to None (caller falls back).
        assert!(Hnsw::from_bytes(&bytes[..bytes.len() / 2]).is_none());
        assert!(Hnsw::from_bytes(b"not a graph").is_none());
    }

    /// Pruning a max-degree base layer down to a small `m0` must track a
    /// NATIVE build at that `m0` — the property that makes the pruned graph a
    /// sound calibration proxy AND a servable persisted graph. A positional
    /// `lst[..m0]` slice (the prior bug) drops the unsorted reverse-link tail
    /// regardless of distance, so small-`m0` recall falls well short of a
    /// native build and leaves nodes unreachable. Serial builds keep this
    /// deterministic.
    #[test]
    fn pruned_base_layer_tracks_native_small_m0() {
        let dim = 24;
        let n = 1200;
        let vectors = random_unit_vectors(n, dim, 0x9F17);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let (m0_small, m0_max, efc) = (8usize, 64usize, 200usize);

        let native = Hnsw::build_serial(
            &scorer,
            HnswParams {
                m0: m0_small,
                ef_construction: efc,
                ..HnswParams::default()
            },
        );
        let base = Hnsw::build_serial(
            &scorer,
            HnswParams {
                m0: m0_max,
                ef_construction: efc,
                ..HnswParams::default()
            },
        );
        let pruned = base.pruned_base_layer(&scorer, m0_small);
        assert_eq!(pruned.base_degree(), m0_small);

        let queries = random_unit_vectors(60, dim, 0x2C4);
        let k = 10;
        let recall = |g: &Hnsw| -> f64 {
            let mut hit = 0usize;
            let mut total = 0usize;
            for q in &queries {
                let truth: HashSet<u32> = brute_force(&scorer, q, k).into_iter().collect();
                let got: HashSet<u32> = g
                    .search(&scorer, q, k, 64)
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect();
                hit += truth.iter().filter(|t| got.contains(t)).count();
                total += k;
            }
            hit as f64 / total as f64
        };
        let r_native = recall(&native);
        let r_pruned = recall(&pruned);
        assert!(
            r_pruned >= r_native - 0.03,
            "distance-aware prune should track native small-m0 recall: pruned {r_pruned:.3} vs native {r_native:.3}"
        );
    }

    /// `measure_recall` reflects graph quality: an under-provisioned base
    /// layer measures below a well-provisioned one. This is the primitive the
    /// incremental drain uses to catch a graph whose inherited `m0` has drifted
    /// below the recall bar as the table grew, and force a full rebuild.
    #[test]
    fn measure_recall_reflects_graph_quality() {
        let dim = 128;
        let n = 4000;
        let vectors = clustered_unit_vectors(n, 32, dim, 0.3, 0xD1F7);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let ef = 256;
        let strong = Hnsw::build(
            &scorer,
            HnswParams {
                m0: 128,
                ef_construction: 200,
                ..HnswParams::default()
            },
        );
        let weak = Hnsw::build(
            &scorer,
            HnswParams {
                m0: 4,
                ef_construction: 200,
                ..HnswParams::default()
            },
        );
        let r_strong = measure_recall(&strong, &scorer, &scorer, ef, 10, 100, 0x5EED);
        let r_weak = measure_recall(&weak, &scorer, &scorer, ef, 10, 100, 0x5EED);
        assert!(
            r_strong >= 0.9,
            "a well-provisioned graph should measure high recall, got {r_strong:.3}"
        );
        assert!(
            r_strong > r_weak,
            "a denser base layer must measure higher recall: strong {r_strong:.3} vs weak {r_weak:.3}"
        );
    }

    /// `from_bytes` degrades a corrupt section to `None` (→ ivf fallback)
    /// rather than decoding a graph that panics at query time. Two hardening
    /// guards beyond the prior `id < n` check: a tower taller than the graph
    /// ever builds, and an upper-layer edge into a shorter tower (an
    /// out-of-bounds index in `greedy_nearest`).
    #[test]
    fn from_bytes_rejects_tower_violations() {
        let dim = 8;
        let n = 300;
        let vectors = random_unit_vectors(n, dim, 0x77A1);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let graph = Hnsw::build(&scorer, HnswParams::default());
        let good = graph.to_bytes();
        assert!(Hnsw::from_bytes(&good).is_some(), "baseline decodes");

        let rd_u32 =
            |b: &[u8], off: usize| u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"));
        let rd_u64 =
            |b: &[u8], off: usize| u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"));

        // Layout: MAGIC(8) n(8) m(4) m0(4) efc(4) entry(4) = 32-byte header,
        // then node_level[n]*4, then layer0 n*m0*4, then records u64, then
        // records of (node u32, level u32, len u32, ids…).
        let n_hdr = rd_u64(&good, 8) as usize;
        assert_eq!(n_hdr, n);
        let m0 = rd_u32(&good, 20) as usize;
        let node_level_off = 32;
        let layer0_off = node_level_off + n * 4;
        let records_off = layer0_off + n * m0 * 4;

        // Guard 1: a node_level word above MAX_LEVEL is rejected.
        let mut over_tower = good.clone();
        over_tower[node_level_off..node_level_off + 4]
            .copy_from_slice(&(MAX_LEVEL + 1).to_le_bytes());
        assert!(
            Hnsw::from_bytes(&over_tower).is_none(),
            "a tower above MAX_LEVEL must be rejected"
        );

        // Guard 2: point an upper-layer edge at a node whose tower is too
        // short for that level. Find a node with tower level 0 to aim at.
        let short = (0..n)
            .find(|&i| rd_u32(&good, node_level_off + i * 4) == 0)
            .expect("some node sits at level 0");
        let records = rd_u64(&good, records_off) as usize;
        assert!(records > 0, "graph has at least one upper-layer node");
        // First record: node(4) level(4) len(4) then ids.
        let rec_body = records_off + 8;
        let level = rd_u32(&good, rec_body + 4);
        assert!(level >= 1, "upper record sits at level >= 1");
        let ids_off = rec_body + 12;
        let mut bad_edge = good.clone();
        bad_edge[ids_off..ids_off + 4].copy_from_slice(&(short as u32).to_le_bytes());
        assert!(
            Hnsw::from_bytes(&bad_edge).is_none(),
            "a level-{level} edge into a level-0 tower must be rejected"
        );
    }

    /// `Hnsw::extend` (incremental batch-insert) grows a prior graph with a
    /// delta and keeps recall in the same ballpark as a from-scratch build at
    /// the same final scale — the property that makes drain-time incremental
    /// insert viable. Also checks the new nodes are actually findable.
    #[test]
    fn extend_matches_full_build_recall() {
        let dim = 32;
        let (n0, delta) = (1500usize, 500usize);
        let total = n0 + delta;
        let vectors = random_unit_vectors(total, dim, 0xE47E7D);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);

        // Prior graph over the first n0, then extend by the delta.
        let prior_vecs: Vec<Vec<f32>> = vectors[..n0].to_vec();
        let prior_scorer = Sq16Scorer::from_unit_vectors(&prior_vecs, dim);
        let prior = Hnsw::build(&prior_scorer, HnswParams::default());
        let incremental = prior.extend(&scorer, HnswParams::default());
        assert_eq!(incremental.len(), total);

        // Full build over all `total` for the recall baseline.
        let full = Hnsw::build(&scorer, HnswParams::default());

        let queries = random_unit_vectors(60, dim, 0xC0FFEE2);
        let k = 10;
        let recall = |g: &Hnsw| -> f64 {
            let mut hit = 0usize;
            for q in &queries {
                let truth: std::collections::HashSet<u32> =
                    brute_force(&scorer, q, k).into_iter().collect();
                for (node, _) in g.search(&scorer, q, k, 64) {
                    if truth.contains(&node) {
                        hit += 1;
                    }
                }
            }
            hit as f64 / (queries.len() * k) as f64
        };
        let inc_recall = recall(&incremental);
        let full_recall = recall(&full);
        // Incremental must stay close to the full-build baseline (small graphs
        // are noisy; the drain-scale gate is measured end-to-end separately).
        assert!(
            inc_recall >= full_recall - 0.05,
            "incremental recall {inc_recall:.3} lags full {full_recall:.3}"
        );
        // A query for a brand-new node's own vector finds it — proof the
        // delta is wired into the graph, not orphaned.
        let new_node = (n0 + delta / 2) as u32;
        let found = incremental.search(&scorer, &vectors[new_node as usize], 1, 64);
        assert_eq!(found[0].0, new_node, "appended node must be reachable");
    }

    /// A full `hnsw` bundle (graph + node→doc-id map + Sq16 plane)
    /// round-trips: the rebuilt index searches identically and maps nodes
    /// back to the same stable doc ids.
    #[test]
    fn hnsw_bundle_roundtrip() {
        use crate::superfile::vector::distance::encode_sq16_row;
        let dim = 24;
        let n = 1200;
        let vectors = random_unit_vectors(n, dim, 0xD00D);
        let stride = dim * 2;
        let mut codes = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        // Distinct, non-trivial stable ids so a node→id mixup would show.
        let doc_ids: Vec<i128> = (0..n as i128).map(|i| 1_000_000 + i * 7).collect();
        let scorer = Sq16Scorer::from_codes(codes.clone(), dim, n);
        let graph = Hnsw::build(&scorer, HnswParams::default());

        // A representative k→ef curve to round-trip through the trailing
        // section.
        let curve: Vec<(u32, u32)> = vec![(1, 128), (10, 128), (50, 256), (100, 512)];
        let bytes = encode_hnsw(
            &codes,
            &doc_ids,
            &graph,
            dim,
            256,
            &curve,
            "emb",
            WalkCodec::Sq8,
            None,
        );
        assert_eq!(
            &bytes[..HNSW_DATA_MAGIC_LEN],
            HNSW_DATA_MAGIC_V5,
            "encode_hnsw stamps the v05 data magic"
        );
        let idx =
            decode_hnsw(&Bytes::from(bytes.clone()), Some(WalkCodec::Sq8)).expect("decode bundle");
        assert_eq!(idx.dim, dim);
        assert_eq!(idx.doc_ids, doc_ids);
        assert_eq!(idx.graph.len(), n);
        assert_eq!(
            idx.ef_search, 256,
            "stamped ef round-trips through the bundle"
        );
        assert_eq!(idx.column, "emb", "stamped column round-trips");
        assert_eq!(idx.ef_curve, curve, "k→ef curve round-trips through v04");
        // The accessor rounds a requested k UP to the next anchor and clamps
        // above the top anchor to the ceiling ef.
        assert_eq!(idx.ef_for_k(1), 128, "k=1 → anchor 1");
        assert_eq!(idx.ef_for_k(10), 128, "k=10 → anchor 10");
        assert_eq!(idx.ef_for_k(11), 256, "k=11 rounds up to anchor 50");
        assert_eq!(idx.ef_for_k(100), 512, "k=100 → anchor 100");
        assert_eq!(
            idx.ef_for_k(1000),
            512,
            "k above top anchor clamps to ceiling"
        );

        let queries = random_unit_vectors(20, dim, 0xFEED);
        for q in &queries {
            let orig = graph.search(&scorer, q, 10, 64);
            let restored = idx.graph.search(&idx.scorer, q, 10, 64);
            assert_eq!(orig, restored, "bundle search diverged");
            // Node → doc id maps through the persisted map.
            for (node, _) in &restored {
                assert_eq!(idx.doc_ids[*node as usize], doc_ids[*node as usize]);
            }
        }
        assert!(decode_hnsw(&Bytes::from_static(b"short"), Some(WalkCodec::Sq8)).is_none());

        // A corrupt node count must degrade to None, not drive a huge
        // `with_capacity` alloc-abort. Overwrite the `n` word (right after the
        // 8-byte magic) with an absurd value and confirm the decode declines.
        let mut poisoned = bytes.clone();
        poisoned[HNSW_DATA_MAGIC_LEN..HNSW_DATA_MAGIC_LEN + 8]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            decode_hnsw(&Bytes::from(poisoned), Some(WalkCodec::Sq8)).is_none(),
            "a corrupt node count must decode to None, not attempt a giant alloc"
        );
    }

    /// The `v04` bundle round-trips both Sq4 variants: the rotation seed, the
    /// fitted ruler, the packed nibble plane(s), and — the property that
    /// matters — the exact search results. The 4-bit section is the one that
    /// cannot be re-derived on read, so a byte lost here is a plane that
    /// scores against a different space than it was fitted in, which shows up
    /// as diffuse recall loss rather than a decode failure.
    #[test]
    fn hnsw_bundle_roundtrip_sq4_planes() {
        let dim = 24usize;
        let n = 300usize;
        let vectors = random_unit_vectors(n, dim, 0xD00D);
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).map(|i| 5_000_000 + i * 3).collect();

        for with_residual in [false, true] {
            let sq4 = Sq4Scorer::from_sq16_plane(&sq16, dim, n, with_residual, TEST_ROT_SEED, None);
            assert_eq!(sq4.has_residual(), with_residual);
            let graph = Hnsw::build(&sq4, HnswParams::default());
            let walk = if with_residual {
                WalkCodec::Sq4Residual
            } else {
                WalkCodec::Sq4
            };
            let bytes = encode_hnsw(
                &sq16,
                &doc_ids,
                &graph,
                dim,
                192,
                &[(10, 192)],
                "emb",
                walk,
                Some(&sq4),
            );
            assert_eq!(
                &bytes[..HNSW_DATA_MAGIC_LEN],
                HNSW_DATA_MAGIC_V5,
                "a 4-bit walk plane must stamp the v04 magic"
            );
            let idx =
                decode_hnsw(&Bytes::from(bytes.clone()), Some(walk)).expect("decode v04 bundle");
            assert_eq!(idx.ef_search, 192);
            assert_eq!(idx.doc_ids, doc_ids);
            let restored = idx
                .sq4
                .as_ref()
                .expect("a 4-bit bundle decodes its 4-bit plane");
            let original = &sq4;
            assert_eq!(restored.has_residual(), with_residual);
            let (rc, rr, ro, rs) = restored.parts();
            let (oc, or_, oo, os) = original.parts();
            assert_eq!(rc, oc, "coarse plane bytes round-trip");
            assert_eq!(rr, or_, "residual plane bytes round-trip");
            assert_eq!(ro, oo, "ruler offsets round-trip");
            assert_eq!(rs, os, "ruler steps round-trip");
            for q in &random_unit_vectors(10, dim, 0xBEEF) {
                assert_eq!(
                    graph.search(original, q, 10, 64),
                    idx.graph.search(restored, q, 10, 64),
                    "restored Sq4 bundle search diverged"
                );
            }
        }
    }

    /// A NARROWER decode must still report the codec the bundle stores.
    ///
    /// This is the property an incremental drain rests on. That path
    /// re-encodes the bundle, so if it reads "which plane is resident?" as
    /// "which plane was stored?" it writes the bundle back without the
    /// section it actually held — and for a fitted 4-bit plane, the ruler is
    /// gone for good: refitting needs a moment pass over the whole corpus
    /// that the incremental path does not do. A filtered decode and a bundle
    /// that never carried the plane look identical from the decoded planes
    /// alone, which is why `stored_walk` comes from the header instead.
    #[test]
    fn stored_walk_reports_the_header_not_the_decoded_planes() {
        let (dim, n) = (40usize, 64usize);
        let vectors = random_unit_vectors(n, dim, 0x5D04);
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();

        for walk in [
            WalkCodec::Sq16,
            WalkCodec::Sq8,
            WalkCodec::Sq4,
            WalkCodec::Sq4Residual,
        ] {
            let sq4 = walk.is_sq4().then(|| {
                Sq4Scorer::from_sq16_plane(&sq16, dim, n, walk.with_residual(), TEST_ROT_SEED, None)
            });
            let bytes = encode_hnsw(
                &sq16,
                &doc_ids,
                &Hnsw::build(
                    &Sq16Scorer::from_codes(sq16.clone(), dim, n),
                    HnswParams::default(),
                ),
                dim,
                192,
                &[(10, 192)],
                "emb",
                walk,
                sq4.as_ref(),
            );
            let bundle = Bytes::from(bytes);

            // The narrowest possible view: no extra plane at all.
            let narrow = decode_hnsw(&bundle, Some(WalkCodec::Sq16)).expect("narrow decode");
            assert_eq!(
                narrow.stored_walk, walk,
                "{walk:?}: a narrower decode must still report the stored codec"
            );
            assert!(
                narrow.sq4.is_none() && narrow.sq8_plane.is_empty(),
                "{walk:?}: the narrow view really did drop the planes — \
                 which is why inferring the codec from them cannot work"
            );

            // "As stored" — what maintenance asks for.
            let as_stored = decode_hnsw(&bundle, None).expect("as-stored decode");
            assert_eq!(as_stored.stored_walk, walk);
            assert_eq!(
                as_stored.sq4.is_some(),
                walk.is_sq4(),
                "{walk:?}: an as-stored decode must materialize the 4-bit plane \
                 so the delta can extend it"
            );
            assert_eq!(
                as_stored.sq4.as_ref().is_some_and(Sq4Scorer::has_residual),
                walk.with_residual(),
                "{walk:?}: the residual leg must survive an as-stored decode"
            );
            assert_eq!(
                !as_stored.sq8_plane.is_empty(),
                walk == WalkCodec::Sq8,
                "{walk:?}: only the SQ8 codec makes the SQ8 plane resident"
            );
        }
    }

    /// A re-encode driven by `stored_walk` must preserve the walk plane.
    ///
    /// This is the incremental drain's write step with the manifest machinery
    /// stripped out: hydrate the prior bundle the way maintenance does (as
    /// stored), then write it back using the codec the header reported. The
    /// bug this pins had the drain hydrate a NARROW view and infer the codec
    /// from it, which resolved to Sq16 every time — so one append-only drain
    /// replaced a fitted 4-bit plane with nothing, and the ruler it needed to
    /// extend was unrecoverable afterwards.
    #[test]
    fn an_as_stored_reencode_preserves_the_walk_plane() {
        let (dim, n) = (40usize, 64usize);
        let vectors = random_unit_vectors(n, dim, 0x5D06);
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();

        for walk in [
            WalkCodec::Sq16,
            WalkCodec::Sq8,
            WalkCodec::Sq4,
            WalkCodec::Sq4Residual,
        ] {
            let sq4 = walk.is_sq4().then(|| {
                Sq4Scorer::from_sq16_plane(&sq16, dim, n, walk.with_residual(), TEST_ROT_SEED, None)
            });
            let graph = Hnsw::build(
                &Sq16Scorer::from_codes(sq16.clone(), dim, n),
                HnswParams::default(),
            );
            // A multi-point curve, so the round-trip is stable: an EMPTY curve
            // decodes to the degenerate 1-point fallback, which would then
            // re-encode to one pair rather than none and fail the byte
            // comparison below for a reason that has nothing to do with the
            // walk plane.
            let curve: Vec<(u32, u32)> = vec![(1, 192), (10, 192), (100, 256)];
            let first = Bytes::from(encode_hnsw(
                &sq16,
                &doc_ids,
                &graph,
                dim,
                192,
                &curve,
                "emb",
                walk,
                sq4.as_ref(),
            ));

            // What maintenance does: hydrate as stored, re-encode on the
            // header's codec.
            let prior = decode_hnsw(&first, None).expect("as-stored hydrate");
            let again = Bytes::from(encode_hnsw(
                prior.scorer.codes(),
                &prior.doc_ids,
                &prior.graph,
                prior.dim,
                prior.ef_search,
                &prior.ef_curve,
                &prior.column,
                prior.stored_walk,
                prior.sq4.as_ref(),
            ));
            assert_eq!(
                first, again,
                "{walk:?}: an as-stored re-encode must reproduce the bundle byte \
                 for byte — a codec that changed here silently drops a section"
            );

            // And the rewritten bundle still serves the same plane.
            let reopened = decode_hnsw(&again, None).expect("reopen rewritten bundle");
            assert_eq!(reopened.stored_walk, walk);
            assert_eq!(reopened.sq4.is_some(), walk.is_sq4());
        }
    }

    /// A rejected 4-bit section must fall back one rung, not all the way.
    ///
    /// With a corrupt ruler `Sq4Scorer::from_parts` declines, and the decode
    /// still succeeds — by design, so a torn section does not take the graph
    /// offline. But leaving BOTH walk planes empty silently promotes the walk
    /// to full-width Sq16: correct answers at several times the configured
    /// bandwidth, and nothing in the results to reveal it. The SQ8 plane is
    /// derivable from Sq16, so that is the rung to land on.
    #[test]
    fn a_rejected_4bit_section_falls_back_to_the_derived_sq8_plane() {
        let (dim, n) = (32usize, 48usize);
        let vectors = random_unit_vectors(n, dim, 0x5D05);
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();
        let column = "emb";
        let sq4 = Sq4Scorer::from_sq16_plane(&sq16, dim, n, false, TEST_ROT_SEED, None);
        let graph = Hnsw::build(&sq4, HnswParams::default());
        let mut bytes = encode_hnsw(
            &sq16,
            &doc_ids,
            &graph,
            dim,
            192,
            &[(10, 192)],
            column,
            WalkCodec::Sq4,
            Some(&sq4),
        );

        // Walk the header to the ruler's `step` vector and zero its first
        // entry: `from_parts` requires every step finite and positive, so this
        // is the minimal poison that models a torn write rather than a
        // structurally short section.
        let step_at = HNSW_DATA_MAGIC_LEN
            + 8  // n
            + 4  // dim
            + 4  // ef
            + 1  // walk tag
            + 4  // column length
            + column.len()
            + n * 16          // doc-id map
            + n * dim * 2     // Sq16 plane
            + 8               // rotation seed
            + dim * 4; // ruler offsets
        bytes[step_at..step_at + 4].copy_from_slice(&0.0f32.to_le_bytes());

        let idx = decode_hnsw(&Bytes::from(bytes), Some(WalkCodec::Sq4))
            .expect("a corrupt 4-bit ruler must not fail the whole decode");
        assert!(
            idx.sq4.is_none(),
            "a non-positive ruler step must be rejected, not scored against"
        );
        assert_eq!(
            idx.sq8_plane.len(),
            n * dim,
            "the rejected 4-bit section must degrade to the derived SQ8 plane, \
             not to a full-width Sq16 walk"
        );
        assert_eq!(
            idx.stored_walk,
            WalkCodec::Sq4,
            "the header still says what this bundle was written for"
        );
    }

    /// The Sq4 walk must find the same planted neighborhood the Sq16 walk
    /// finds. Planted clusters (not uniform noise) so the codec has real
    /// structure to preserve; the residual variant must be at least as
    /// faithful as the bare one on the coarse ruler it refines.
    #[test]
    fn sq4_walk_matches_sq16_on_planted_clusters() {
        let dim = 32usize;
        let n_clusters = 12usize;
        let per = 60usize;
        let n = n_clusters * per;
        // Cluster centers: distinct unit axes pairs; members: center + small
        // deterministic jitter, renormalized.
        let mut rng = 0x5EED_5EEDu64;
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for c in 0..n_clusters {
            for _ in 0..per {
                let mut v = vec![0.0f32; dim];
                v[c % dim] = 1.0;
                v[(c * 7 + 3) % dim] = 0.5;
                for x in v.iter_mut() {
                    let u = (splitmix64(&mut rng) >> 40) as f32 / (1u64 << 24) as f32;
                    *x += (u - 0.5) * 0.08;
                }
                let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt();
                for x in v.iter_mut() {
                    *x /= norm;
                }
                vectors.push(v);
            }
        }
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let sq16_scorer = Sq16Scorer::from_codes(sq16.clone(), dim, n);
        let g16 = Hnsw::build(&sq16_scorer, HnswParams::default());

        // The two variants promise DIFFERENT invariants, and the test
        // asserts each rung's own claim rather than one bar for both:
        //
        //  * bare Sq4 (0.5 B/dim) is a NAVIGATION plane. Within a cluster
        //    its coarse step quantizes all members to near-identical codes,
        //    so intra-cluster ordering genuinely dissolves (measured:
        //    id-overlap with Sq16 collapses to ~0.17 ≈ the tie fraction) —
        //    but the walk must still land in the right cluster. The
        //    assertable property is cluster membership of the top-k.
        //  * Sq4+residual (1 B/dim) refines each coarse step 15×, restoring
        //    fine ordering — so it must ALSO recover most of Sq16's actual
        //    top-k identities.
        //
        // Both floors are loose wiring tripwires (a broken ruler or nibble
        // pack collapses membership toward the cluster fraction ~0.08), not
        // recall benchmarks; those run in the bench harness on real corpora.
        const K: usize = 10;
        let queries: Vec<Vec<f32>> = (0..n_clusters)
            .map(|c| {
                let mut q = vec![0.0f32; dim];
                q[c % dim] = 1.0;
                q[(c * 7 + 3) % dim] = 0.5;
                let norm = q.iter().map(|a| a * a).sum::<f32>().sqrt();
                for x in q.iter_mut() {
                    *x /= norm;
                }
                q
            })
            .collect();
        let mut overlaps = [0.0f64; 2];
        for (vi, with_residual) in [false, true].into_iter().enumerate() {
            let sq4 = Sq4Scorer::from_sq16_plane(&sq16, dim, n, with_residual, TEST_ROT_SEED, None);
            let g4 = Hnsw::build(&sq4, HnswParams::default());
            let (mut same_cluster, mut agree, mut total) = (0usize, 0usize, 0usize);
            for (c, q) in queries.iter().enumerate() {
                let want: Vec<u32> = g16
                    .search(&sq16_scorer, q, K, 64)
                    .into_iter()
                    .map(|(node, _)| node)
                    .collect();
                for (node, _) in g4.search(&sq4, q, K, 64) {
                    same_cluster += usize::from(node as usize / per == c);
                    agree += usize::from(want.contains(&node));
                    total += 1;
                }
            }
            let membership = same_cluster as f64 / total as f64;
            overlaps[vi] = agree as f64 / total as f64;
            assert!(
                membership >= 0.9,
                "Sq4(residual={with_residual}) top-{K} cluster membership \
                 {membership:.3} — the walk is not navigating, the plane \
                 wiring is broken"
            );
        }
        // The residual's assertable property is RELATIVE: near-tied
        // same-cluster members sit within Sq4+residual's reconstruction
        // error, so exact id-agreement with Sq16 is not a codec invariant
        // (measured ~0.5 here, and that is the physics of near-ties, not a
        // defect). What IS an invariant: the residual plane must refine the
        // bare plane's ordering materially — a broken residual leaves the
        // overlap at the bare plane's tie-collapse level (~0.17).
        assert!(
            overlaps[1] >= (overlaps[0] * 2.0).max(0.4),
            "Sq4+residual id-overlap {:.3} does not refine the bare plane's \
             {:.3} — the residual nibbles are not being applied",
            overlaps[1],
            overlaps[0]
        );
    }

    /// Incremental extends inherit the PRIOR ruler: delta rows encoded onto
    /// it clamp rather than refit, so resident nodes' reconstructions never
    /// move. A delta row outside the prior range must land on the ruler's
    /// edge, not shift the ruler.
    #[test]
    fn sq4_delta_rows_inherit_the_prior_ruler_and_clamp() {
        let dim = 8usize;
        let n = 64usize;
        let vectors = random_unit_vectors(n, dim, 0xABBA);
        let stride = dim * 2;
        let mut sq16 = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut sq16[i * stride..(i + 1) * stride]);
        }
        let prior = Sq4Scorer::from_sq16_plane(&sq16, dim, n, true, TEST_ROT_SEED, None);
        let (_, _, offset, step) = prior.parts();
        let (offset, step) = (offset.to_vec(), step.to_vec());

        // A delta row deliberately outside the fitted range on dim 0.
        let mut wild = vec![0.0f32; dim];
        wild[0] = 1.0;
        let mut delta16 = vec![0u8; stride];
        encode_sq16_row(&wild, &mut delta16);
        let delta = Sq4Scorer::from_sq16_plane(
            &delta16,
            dim,
            1,
            true,
            TEST_ROT_SEED,
            Some((&offset, &step)),
        );
        let (_, _, doff, dstep) = delta.parts();
        assert_eq!(doff, offset.as_slice(), "delta must adopt the prior ruler");
        assert_eq!(dstep, step.as_slice(), "delta must adopt the prior ruler");
        // The encoder clamps out-of-range components to the ruler's edge
        // codes by construction; what the test must pin is that the delta
        // NEVER refits (asserted above via ruler equality) and that the
        // clamped row still decodes to finite query-space values.
        let mut recon = vec![0.0f32; dim];
        delta.decode_node(0, &mut recon);
        assert!(
            recon.iter().all(|x| x.is_finite()),
            "clamped delta row must decode to finite components"
        );
    }

    /// A `v03` bundle serves the SQ8 plane as a persisted section that is a
    /// zero-copy slice of the same backing bytes as the Sq16 plane, and a
    /// pre-existing `v02` bundle (Sq16 only) still decodes with the SQ8 plane
    /// derived on read — the backward-compatible fallback. Both paths must
    /// yield byte-identical planes and identical search results.
    #[test]
    fn sq8_section_v03_matches_derived_v02() {
        use crate::superfile::vector::distance::encode_sq16_row;
        let dim = 24;
        let n = 400;
        let vectors = random_unit_vectors(n, dim, 0x5EC7);
        let stride = dim * 2;
        let mut codes = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();
        let scorer = Sq16Scorer::from_codes(codes.clone(), dim, n);
        let graph = Hnsw::build(&scorer, HnswParams::default());

        // Current encoder → v05: codec named in the header, its section, and
        // the trailing curve.
        let curve: Vec<(u32, u32)> = vec![(1, 128), (10, 128), (50, 256), (100, 512)];
        let v5 = encode_hnsw(
            &codes,
            &doc_ids,
            &graph,
            dim,
            128,
            &curve,
            "emb",
            WalkCodec::Sq8,
            None,
        );
        assert_eq!(&v5[..HNSW_DATA_MAGIC_LEN], HNSW_DATA_MAGIC_V5);

        // Both older SQ8-carrying formats are laid out BY HAND rather than
        // re-encoded. A fixture the current encoder produced cannot catch a
        // framing change that moved reader and writer together, which is the
        // failure a version bump exists to prevent. v03 is the v05 framing minus
        // the codec byte and minus the curve; v04 is v03 plus the curve.
        let sq8_framed = |magic: &[u8; 8], with_curve: bool| -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(magic);
            out.extend_from_slice(&(n as u64).to_le_bytes());
            out.extend_from_slice(&(dim as u32).to_le_bytes());
            out.extend_from_slice(&128u32.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(b"emb");
            for &id in &doc_ids {
                out.extend_from_slice(&id.to_le_bytes());
            }
            out.extend_from_slice(&codes);
            out.extend_from_slice(&derive_sq8_plane(&codes));
            let gb = graph.to_bytes();
            out.extend_from_slice(&(gb.len() as u64).to_le_bytes());
            out.extend_from_slice(&gb);
            if with_curve {
                out.extend_from_slice(&(curve.len() as u16).to_le_bytes());
                for &(k, ef) in &curve {
                    out.extend_from_slice(&k.to_le_bytes());
                    out.extend_from_slice(&ef.to_le_bytes());
                }
            }
            out
        };
        let v4 = sq8_framed(HNSW_DATA_MAGIC_V4, true);
        let v3 = sq8_framed(HNSW_DATA_MAGIC_V3, false);

        // Synthesize a pre-existing v02 bundle: identical framing but no SQ8
        // section (Sq16 plane runs straight into the graph section) and the v02
        // magic. Built by re-laying the wire so the fixture matches exactly what
        // an old drain wrote.
        let mut v2 = Vec::new();
        v2.extend_from_slice(HNSW_DATA_MAGIC_V2);
        v2.extend_from_slice(&(n as u64).to_le_bytes());
        v2.extend_from_slice(&(dim as u32).to_le_bytes());
        v2.extend_from_slice(&128u32.to_le_bytes());
        let col = b"emb";
        v2.extend_from_slice(&(col.len() as u32).to_le_bytes());
        v2.extend_from_slice(col);
        for &id in &doc_ids {
            v2.extend_from_slice(&id.to_le_bytes());
        }
        v2.extend_from_slice(&codes);
        let graph_bytes = graph.to_bytes();
        v2.extend_from_slice(&(graph_bytes.len() as u64).to_le_bytes());
        v2.extend_from_slice(&graph_bytes);

        let idx_v5 = decode_hnsw(&Bytes::from(v5), Some(WalkCodec::Sq8)).expect("decode v05");
        let idx_v4 = decode_hnsw(&Bytes::from(v4), Some(WalkCodec::Sq8)).expect("decode v04");
        let idx_v3 = decode_hnsw(&Bytes::from(v3), Some(WalkCodec::Sq8)).expect("decode v03");
        let idx_v2 = decode_hnsw(&Bytes::from(v2), Some(WalkCodec::Sq8)).expect("decode v02");

        // The SQ8 plane the persisted section carries must equal the one derived
        // from the Sq16 high byte.
        let derived = derive_sq8_plane(&codes);
        assert_eq!(idx_v5.sq8_plane.bytes(), derived.as_slice());
        assert_eq!(idx_v4.sq8_plane.bytes(), derived.as_slice());
        assert_eq!(idx_v3.sq8_plane.bytes(), derived.as_slice());
        assert_eq!(idx_v2.sq8_plane.bytes(), derived.as_slice());
        assert_eq!(idx_v5.sq8_plane.len(), n * dim);
        // Every one of these stores the SQ8 codec (v02 by implication — it has
        // no section, so its plane is derived) and none carries a 4-bit one.
        assert_eq!(idx_v5.stored_walk, WalkCodec::Sq8);
        assert_eq!(idx_v4.stored_walk, WalkCodec::Sq8);
        assert_eq!(idx_v3.stored_walk, WalkCodec::Sq8);
        assert_eq!(idx_v2.stored_walk, WalkCodec::Sq16);
        assert!(
            idx_v5.sq4.is_none()
                && idx_v4.sq4.is_none()
                && idx_v3.sq4.is_none()
                && idx_v2.sq4.is_none()
        );

        // The curve rides the two versions that carry it and degenerates on the
        // two that do not — the axis is independent of the walk plane.
        assert_eq!(idx_v5.ef_curve, curve, "v05 carries the curve verbatim");
        assert_eq!(idx_v4.ef_curve, curve, "v04 carries the curve verbatim");
        assert_eq!(idx_v3.ef_curve.len(), 1, "v03 degenerates to one point");
        assert_eq!(idx_v2.ef_curve.len(), 1, "v02 degenerates to one point");

        // Sq16 plane and searches identical across all four bundle versions.
        assert_eq!(idx_v5.scorer.codes(), idx_v2.scorer.codes());
        assert_eq!(idx_v4.scorer.codes(), idx_v2.scorer.codes());
        assert_eq!(idx_v3.scorer.codes(), idx_v2.scorer.codes());
        let queries = random_unit_vectors(30, dim, 0x1CE);
        for q in &queries {
            let a = idx_v5.search_sq8_refine(q, 10, 64, 32);
            let b = idx_v2.search_sq8_refine(q, 10, 64, 32);
            let c = idx_v3.search_sq8_refine(q, 10, 64, 32);
            let d = idx_v4.search_sq8_refine(q, 10, 64, 32);
            assert_eq!(a, c, "v05 section vs v03 section diverged");
            assert_eq!(a, d, "v05 section vs v04 section diverged");
            assert_eq!(a, b, "persisted SQ8 section vs v02 derive diverged");
        }

        // Asked for the Sq16 walk, a v05 bundle still CONSUMES the SQ8 section
        // (so the graph and the curve behind it stay reachable) and serves with
        // an empty SQ8 plane.
        let v5_again = encode_hnsw(
            &codes,
            &doc_ids,
            &graph,
            dim,
            128,
            &curve,
            "emb",
            WalkCodec::Sq8,
            None,
        );
        let idx_off =
            decode_hnsw(&Bytes::from(v5_again), Some(WalkCodec::Sq16)).expect("decode v05 sq8-off");
        assert!(idx_off.sq8_plane.is_empty());
        assert_eq!(idx_off.graph.len(), n);
        assert_eq!(
            idx_off.ef_curve, curve,
            "a narrower walk request must not cost the curve behind the section"
        );
    }

    /// A pre-`v04` bundle (`v03` with the SQ8 section, or `v02` without) carries
    /// a single stamped `ef` and no k→ef curve. Decoding must synthesize the
    /// degenerate 1-point curve so `ef_for_k(k) == ef_search` for EVERY `k` —
    /// exactly today's single-`ef` behavior, no forced rebuild.
    #[test]
    fn pre_v04_bundle_decodes_to_one_point_curve() {
        use crate::superfile::vector::distance::encode_sq16_row;
        let dim = 16;
        let n = 300;
        let stamped_ef = 200usize;
        let vectors = random_unit_vectors(n, dim, 0xB0BB1E);
        let stride = dim * 2;
        let mut codes = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();
        let scorer = Sq16Scorer::from_codes(codes.clone(), dim, n);
        let graph = Hnsw::build(&scorer, HnswParams::default());
        let graph_bytes = graph.to_bytes();
        let col = b"emb";

        // Shared framing writer: magic + header + doc-ids + Sq16 plane, then
        // (v03 only) the SQ8 section, then the graph section — NO curve section.
        let frame = |magic: &[u8; 8], with_sq8: bool| -> Vec<u8> {
            let mut b = Vec::new();
            b.extend_from_slice(magic);
            b.extend_from_slice(&(n as u64).to_le_bytes());
            b.extend_from_slice(&(dim as u32).to_le_bytes());
            b.extend_from_slice(&(stamped_ef as u32).to_le_bytes());
            b.extend_from_slice(&(col.len() as u32).to_le_bytes());
            b.extend_from_slice(col);
            for &id in &doc_ids {
                b.extend_from_slice(&id.to_le_bytes());
            }
            b.extend_from_slice(&codes);
            if with_sq8 {
                extend_sq8_plane(&mut b, &codes);
            }
            b.extend_from_slice(&(graph_bytes.len() as u64).to_le_bytes());
            b.extend_from_slice(&graph_bytes);
            b
        };

        let v3 = frame(HNSW_DATA_MAGIC_V3, true);
        let v2 = frame(HNSW_DATA_MAGIC_V2, false);

        for (label, fixture) in [("v03", v3), ("v02", v2)] {
            let idx = decode_hnsw(&Bytes::from(fixture), Some(WalkCodec::Sq8))
                .unwrap_or_else(|| panic!("{label} fixture must decode"));
            assert_eq!(idx.ef_search, stamped_ef, "{label} ef round-trips");
            // Degenerate 1-point curve: constant ef for every k, small and large.
            assert_eq!(idx.ef_curve.len(), 1, "{label} → degenerate 1-point curve");
            for k in [1usize, 10, 50, 100, 1000, 100_000] {
                assert_eq!(
                    idx.ef_for_k(k),
                    stamped_ef,
                    "{label}: ef_for_k({k}) must equal the single stamped ef"
                );
            }
        }
    }

    /// SQ8 walk + Sq16 refine returns essentially the same top-k as the Sq16
    /// walk. Both are measured against the brute-force exact-Sq16 top-k so the
    /// test asserts the PR's core claim directly: navigating on the cheap int8
    /// plane then refining on Sq16 costs no meaningful recall versus walking on
    /// Sq16 throughout.
    #[test]
    fn search_sq8_refine_matches_sq16_walk() {
        use crate::superfile::vector::distance::encode_sq16_row;
        let dim = 32;
        let n = 1000;
        let k = 10;
        let ef = 64;
        let vectors = random_unit_vectors(n, dim, 0xA11CE);
        let stride = dim * 2;
        let mut codes = vec![0u8; n * stride];
        for (i, v) in vectors.iter().enumerate() {
            encode_sq16_row(v, &mut codes[i * stride..(i + 1) * stride]);
        }
        let doc_ids: Vec<i128> = (0..n as i128).collect();
        let scorer = Sq16Scorer::from_codes(codes.clone(), dim, n);
        let graph = Hnsw::build(&scorer, HnswParams::default());
        let bytes = encode_hnsw(
            &codes,
            &doc_ids,
            &graph,
            dim,
            ef,
            &[(10, ef as u32)],
            "emb",
            WalkCodec::Sq8,
            None,
        );
        let idx = decode_hnsw(&Bytes::from(bytes), Some(WalkCodec::Sq8)).expect("decode bundle");
        assert!(
            !idx.sq8_plane.is_empty(),
            "the SQ8 plane must be resident when the SQ8 walk is asked for"
        );

        let queries = random_unit_vectors(50, dim, 0xB0B);
        let mut sq16_hits = 0usize;
        let mut sq8_hits = 0usize;
        for q in &queries {
            // Brute-force exact-Sq16 top-k ground truth (NegDot: lower nearer).
            let prep = scorer.prepare(q);
            let mut all: Vec<(u32, f32)> = (0..n as u32)
                .map(|node| (node, scorer.score(&prep, node)))
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gt: std::collections::HashSet<u32> =
                all.iter().take(k).map(|(node, _)| *node).collect();

            let sq16: Vec<u32> = graph
                .search(&scorer, q, k, ef)
                .into_iter()
                .map(|(node, _)| node)
                .collect();
            // refine_k == ef refines the whole beam — the widest refine.
            let sq8: Vec<u32> = idx
                .search_sq8_refine(q, k, ef, ef)
                .into_iter()
                .map(|(node, _)| node)
                .collect();
            sq16_hits += sq16.iter().filter(|node| gt.contains(node)).count();
            sq8_hits += sq8.iter().filter(|node| gt.contains(node)).count();
        }
        let denom = (queries.len() * k) as f32;
        let sq16_recall = sq16_hits as f32 / denom;
        let sq8_recall = sq8_hits as f32 / denom;
        // The SQ8 walk navigates a slightly different beam, but refining on
        // Sq16 recovers the ranking: its recall tracks the Sq16 walk within a
        // small margin (not a lower fixed floor, which would pass even on a
        // broken walk that always trailed).
        assert!(
            sq16_recall > 0.9,
            "sanity: Sq16 walk recall {sq16_recall:.4} unexpectedly low"
        );
        assert!(
            sq8_recall >= sq16_recall - 0.03,
            "SQ8 refine recall {sq8_recall:.4} trails Sq16 walk {sq16_recall:.4} by too much"
        );
    }

    /// The combined graph bundle frames its sections losslessly, including
    /// the absent-section flags (data/provenance omitted above the scale
    /// ceiling) and an empty centroid section.
    #[test]
    fn graph_bundle_frames_sections() {
        // Full bundle: centroid graph + data bundle + population key + high water.
        let centroid = vec![1u8, 2, 3, 4, 5];
        let data = vec![9u8; 300];
        let blob = encode_resident_envelope(
            0xDEAD_BEEF_1234,
            987_654_321,
            &centroid,
            Some((PayloadKind::Graph, &data)),
        );
        // Both modes recover identical sections; only the data-section backing
        // differs (zero-copy slice of `raw` vs an owned copy).
        for mmap_backed in [true, false] {
            let raw = Bytes::from(blob.clone());
            let got = decode_resident_envelope(&raw, mmap_backed).expect("decode full");
            assert_eq!(got.population_key, 0xDEAD_BEEF_1234);
            assert_eq!(got.high_water_id, 987_654_321);
            assert_eq!(got.centroid_graph, centroid);
            let (kind, bytes) = got.data_bundle.as_ref().expect("data present");
            assert_eq!(
                *kind,
                PayloadKind::Graph,
                "the envelope must report the kind it was written with"
            );
            assert_eq!(&bytes[..], &data[..]);
            // On the heap path the data section must be an independent copy, not
            // a view into `raw`, so `raw` can be freed.
            if !mmap_backed {
                assert!(
                    !raw.as_ref().as_ptr_range().contains(&bytes.as_ptr()),
                    "heap-path payload must not alias raw"
                );
            }
        }
        // The header reads from the fixed-offset prefix alone (a tiny range
        // GET at settle time — no need for the multi-GiB body).
        assert_eq!(
            resident_envelope_header(&blob[..GRAPH_BUNDLE_HEADER_BYTES]),
            Some((0xDEAD_BEEF_1234, 987_654_321))
        );

        // Data-less bundle (above the scale ceiling): empty centroid, no data.
        let blob = encode_resident_envelope(0, 0, &[], None);
        let got = decode_resident_envelope(&Bytes::from(blob), true).expect("decode empty");
        assert!(got.centroid_graph.is_empty());
        assert!(got.data_bundle.is_none());

        assert!(decode_resident_envelope(&Bytes::from_static(b"bad"), true).is_none());
        assert!(resident_envelope_header(b"short").is_none());
    }

    /// The kind tag round-trips for BOTH kinds, and an unknown one declines
    /// the section without taking the envelope down with it.
    ///
    /// The compat story was asserted only for `Graph`, which is the arm that
    /// cannot regress: `1` is what the presence byte has always meant. `Flat`
    /// is the arm a future edit to `put_opt_section` or the decode cursor
    /// would break, and it would break silently — every flat table falls back
    /// to ivf and every test stays green, because nothing else pins it.
    #[test]
    fn envelope_kind_tag_round_trips_and_declines_the_unknown() {
        let centroid = vec![7u8; 40];
        let data = vec![3u8; 128];
        for kind in [PayloadKind::Graph, PayloadKind::Flat] {
            for mmap_backed in [true, false] {
                let blob = encode_resident_envelope(11, 22, &centroid, Some((kind, &data)));
                let got = decode_resident_envelope(&Bytes::from(blob), mmap_backed)
                    .expect("decode a tagged envelope");
                let (got_kind, bytes) = got.data_bundle.as_ref().expect("payload present");
                assert_eq!(*got_kind, kind, "the tag must survive the round trip");
                assert_eq!(&bytes[..], &data[..]);
                assert_eq!(got.centroid_graph, centroid);
            }
        }

        // A payload written by a NEWER build. The section is declined and the
        // centroid graph — which this reader still understands — survives. The
        // cursor must consume the payload's length prefix and body even though
        // the kind is unknown, or everything after it mis-slices.
        let mut blob =
            encode_resident_envelope(11, 22, &centroid, Some((PayloadKind::Flat, &data)));
        // Header (magic + population key + high water), then the centroid
        // section's own u64 length prefix, then its bytes, then the kind tag.
        let tag_at = GRAPH_BUNDLE_HEADER_BYTES + size_of::<u64>() + centroid.len();
        assert_eq!(
            blob[tag_at],
            PayloadKind::Flat as u8,
            "the tag sits directly after the centroid section"
        );
        for unknown in [3u8, 9, u8::MAX] {
            blob[tag_at] = unknown;
            let got = decode_resident_envelope(&Bytes::from(blob.clone()), true)
                .expect("an unknown kind must decline the SECTION, not the envelope");
            assert!(
                got.data_bundle.is_none(),
                "tag {unknown} is not a kind this build can serve"
            );
            assert_eq!(
                got.centroid_graph, centroid,
                "the centroid graph must survive a payload this reader cannot decode"
            );
            assert_eq!((got.population_key, got.high_water_id), (11, 22));
        }
    }

    /// Empty and singleton graphs don't panic and answer sanely.
    #[test]
    fn degenerate_graphs() {
        let dim = 8;
        let empty: Vec<Vec<f32>> = Vec::new();
        let scorer = Fp32Scorer::from_vectors(&empty, dim);
        let hnsw = Hnsw::build(&scorer, HnswParams::default());
        assert!(hnsw.is_empty());
        assert!(hnsw.search(&scorer, &vec![0.0; dim], 5, 16).is_empty());

        let one = random_unit_vectors(1, dim, 7);
        let scorer = Fp32Scorer::from_vectors(&one, dim);
        let hnsw = Hnsw::build(&scorer, HnswParams::default());
        let got = hnsw.search(&scorer, &one[0], 5, 16);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 0);
    }

    /// Manual build-time signal: serial vs parallel wall time on Sq16 nodes.
    /// `#[ignore]`d (too slow for the default run); node count is
    /// `HNSW_BENCH_N` (default 50_000). Run with:
    ///
    /// ```text
    /// HNSW_BENCH_N=200000 cargo test --release --lib \
    ///   superfile::vector::hnsw::tests::build_speedup_serial_vs_parallel \
    ///   -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn build_speedup_serial_vs_parallel() {
        use std::time::Instant;
        let dim = 128;
        let n: usize = std::env::var("HNSW_BENCH_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000);
        let vectors = random_unit_vectors(n, dim, 0x5EED);
        let scorer = Sq16Scorer::from_unit_vectors(&vectors, dim);
        let threads = rayon::current_num_threads();

        let t = Instant::now();
        let serial = Hnsw::build_serial(&scorer, HnswParams::default());
        let serial_s = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let parallel = Hnsw::build(&scorer, HnswParams::default());
        let parallel_s = t.elapsed().as_secs_f64();

        assert_eq!(serial.len(), n);
        assert_eq!(parallel.len(), n);
        eprintln!(
            "hnsw build n={n} dim={dim} threads={threads}: serial {serial_s:.2}s, \
             parallel {parallel_s:.2}s, speedup {:.2}x",
            serial_s / parallel_s
        );

        // The guard is PARITY, not an absolute floor: random-uniform vectors
        // in high dim are adversarial for any HNSW (recall is low even
        // serially), so what proves the parallel build didn't wreck graph
        // quality is that its recall tracks the serial build's on the same
        // data/params.
        let queries = random_unit_vectors(50, dim, 0xBEEF);
        let recall = |g: &Hnsw| -> f64 {
            let k = 10;
            let mut hit = 0usize;
            for q in &queries {
                let truth: std::collections::HashSet<u32> =
                    brute_force(&scorer, q, k).into_iter().collect();
                for (node, _) in g.search(&scorer, q, k, 64) {
                    if truth.contains(&node) {
                        hit += 1;
                    }
                }
            }
            hit as f64 / (queries.len() * k) as f64
        };
        let serial_recall = recall(&serial);
        let parallel_recall = recall(&parallel);
        eprintln!("recall@10: serial {serial_recall:.4}, parallel {parallel_recall:.4}");
        assert!(
            parallel_recall >= serial_recall - 0.05,
            "parallel recall {parallel_recall:.3} regressed vs serial {serial_recall:.3}"
        );
    }
}
