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
    collections::BinaryHeap,
    sync::{Mutex, RwLock},
};

use rayon::prelude::*;

use crate::superfile::vector::distance::{
    Metric, Sq16Kernel, dequantize_sq16_into, dot, encode_sq16_row,
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

/// Sq16 node scorer: one `u16` code per dimension on the fixed cosine
/// grid, scored with the existing fused-dequant [`Sq16Kernel`] under the
/// [`Metric::NegDot`] convention (`score = −dot`, so smaller is nearer).
///
/// The codes are stored row-major (`dim × 2` bytes per node) and scored
/// straight from the code bytes — no per-candidate decode buffer.
pub(crate) struct Sq16Scorer {
    /// `len × dim × 2` little-endian `u16` codes, row-major.
    codes: Vec<u8>,
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
            codes,
            dim,
            len: vectors.len(),
        }
    }

    /// Adopt already-encoded Sq16 code bytes verbatim: `codes` is
    /// `len × dim × 2` little-endian `u16` (row-major), exactly the
    /// on-disk `full[]` Sq16 plane. No decode/re-encode round trip.
    pub(crate) fn from_codes(codes: Vec<u8>, dim: usize, len: usize) -> Self {
        debug_assert_eq!(codes.len(), len * dim * 2);
        Self { codes, dim, len }
    }

    /// The raw node-ordered Sq16 code plane — so an incremental build can
    /// concatenate the prior codes with a freshly-drained delta into one
    /// combined scorer.
    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes
    }

    #[inline]
    fn row(&self, node: u32) -> &[u8] {
        let stride = self.dim * 2;
        let start = node as usize * stride;
        &self.codes[start..start + stride]
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

/// SplitMix64 — a tiny, fully deterministic mixer for layer assignment.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic layer for `node`: `floor(−ln(U) · ml)` with `U` a
/// seeded uniform in `(0, 1]`, the standard exponential HNSW tower.
fn assign_level(seed: u64, node: u32, ml: f64) -> u32 {
    let mut st = seed ^ (node as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
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

        // Insert nodes 1..n concurrently. `for_each_init` hands each worker a
        // reusable `VisitedSet` scratch so the O(n) epoch buffer is allocated
        // once per thread, not once per insert.
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
    pub(crate) fn search<S: NodeScorer>(
        &self,
        scorer: &S,
        query: &[f32],
        k: usize,
        ef: usize,
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
        let mut visited = VisitedSet::new(self.len);
        let ef = ef.max(k);
        let found = self.search_layer(scorer, &prepared, &[ep], ef, 0, &mut visited);
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

    /// A copy with the layer-0 (base) adjacency truncated to `m0` neighbors
    /// per node — a cheap way to evaluate a smaller base-layer degree without a
    /// native rebuild. Upper layers are untouched. The pruned graph is BOTH the
    /// calibration proxy and what gets persisted for the chosen `m0`: truncating
    /// the max-`m0` build's base layer approximates a native `m0` build closely
    /// enough to rank candidates, and it serves directly — no second full build.
    pub(crate) fn pruned_base_layer(&self, m0: usize) -> Hnsw {
        let neighbors = self
            .neighbors
            .iter()
            .map(|levels| {
                levels
                    .iter()
                    .enumerate()
                    .map(|(lvl, lst)| {
                        if lvl == 0 && lst.len() > m0 {
                            lst[..m0].to_vec()
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
fn exhaustive_topk<S: NodeScorer>(scorer: &S, query: &[f32], k: usize) -> Vec<u32> {
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

/// Recall@k of `graph` walked at `ef` against exhaustive `gt`.
fn graph_recall(
    graph: &Hnsw,
    scorer: &Sq16Scorer,
    queries: &[Vec<f32>],
    gt: &[Vec<u32>],
    k: usize,
    ef: usize,
) -> f64 {
    let mut hit = 0usize;
    let mut total = 0usize;
    for (q, truth) in queries.iter().zip(gt) {
        let got: std::collections::HashSet<u32> = graph
            .search(scorer, q, k, ef)
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

/// Calibrate `(m0, ef)` to `target_recall` on `scorer` (the drained Sq16 plane,
/// or a subsample of it). Builds ONE graph at `max(m0_candidates)`, evaluates
/// smaller `m0` by pruning the base layer (cheap) and `ef` by re-search (free),
/// and returns the **fastest** clearing pair (min `ef`, then min `m0` — latency
/// is the graph's whole point). If none clears within the candidates, returns
/// the best achieved with `registered` gated by the `0.9×target` graceful
/// floor. Queries are held-out, perturbed (off-node) so recall is realistic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate_graph(
    scorer: &Sq16Scorer,
    m0_candidates: &[usize],
    ef_candidates: &[usize],
    target_recall: f64,
    recall_slack: f64,
    ef_construction: usize,
    n_queries: usize,
    k: usize,
    seed: u64,
) -> (CalibChoice, Option<Hnsw>) {
    let register_floor = (target_recall - recall_slack).max(0.0);
    let n = scorer.len();
    let dim = scorer.dim();
    let fallback = CalibChoice {
        m0: *m0_candidates.iter().min().unwrap_or(&32),
        ef: *ef_candidates.iter().min().unwrap_or(&128),
        recall: 0.0,
        registered: false,
        at_target: false,
    };
    if n == 0 || m0_candidates.is_empty() || ef_candidates.is_empty() {
        return (fallback, None);
    }
    let stride = dim * 2;
    let mut rng = seed ^ 0x9E37_79B9_7F4A_7C15;
    let nq = n_queries.min(n);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| {
            let node = i.wrapping_mul(2_654_435_761) % n;
            let mut v = vec![0.0f32; dim];
            dequantize_sq16_into(&scorer.codes()[node * stride..(node + 1) * stride], &mut v);
            for x in &mut v {
                let u = (splitmix64(&mut rng) >> 40) as f32 / (1u64 << 24) as f32; // [0,1)
                *x += (u * 2.0 - 1.0) * 0.05;
            }
            let norm = v.iter().map(|a| a * a).sum::<f32>().sqrt().max(1e-12);
            for x in &mut v {
                *x /= norm;
            }
            v
        })
        .collect();
    let gt: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| exhaustive_topk(scorer, q, k))
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
    let base = Hnsw::build(
        scorer,
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
            let g = base.pruned_base_layer(m0);
            efs.iter()
                .map(|&ef| graph_recall(&g, scorer, &queries, &gt, k, ef))
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
    // registered.
    let graph = choice.registered.then(|| base.pruned_base_layer(choice.m0));
    (choice, graph)
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
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
    fn i128(&mut self) -> Option<i128> {
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
        let mut node_level = Vec::with_capacity(n);
        for _ in 0..n {
            node_level.push(c.u32()?);
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
            let mut list = Vec::with_capacity(len);
            for _ in 0..len {
                let id = c.u32()?;
                if id as usize >= n {
                    return None;
                }
                list.push(id);
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

/// On-disk magic for a persisted `hnsw` bundle (graph + node→doc-id
/// map + node-ordered Sq16 plane), the self-contained payload a resident
/// data index is rebuilt from at open.
const HNSW_DATA_MAGIC: &[u8; 8] = b"INFDDG01";

/// A `hnsw` resident index rebuilt from a persisted bundle: the Sq16
/// scorer over the node-ordered code plane, the walkable graph, and the
/// `node_index -> stable doc id` map.
pub(crate) struct HnswIndex {
    pub scorer: Sq16Scorer,
    pub graph: Hnsw,
    pub doc_ids: Vec<i128>,
    pub dim: usize,
    /// Calibrated query beam stamped at drain — the served `ef` default (a
    /// query knob, so it rides in the bundle header, not the graph structure).
    /// 0 ⇒ older bundle with no stamp; caller falls back to the ef law.
    pub ef_search: usize,
}

/// Serialize a `hnsw` index to a persistable byte bundle: header,
/// the `node -> stable doc id` map, the node-ordered Sq16 code plane, and
/// the graph section. The Sq16 plane is carried inline so the bundle is
/// self-contained — reopening needs nothing but these bytes.
pub(crate) fn encode_hnsw(
    sq16_codes: &[u8],
    doc_ids: &[i128],
    graph: &Hnsw,
    dim: usize,
    ef_search: usize,
) -> Vec<u8> {
    let n = doc_ids.len();
    debug_assert_eq!(sq16_codes.len(), n * dim * 2);
    let graph_bytes = graph.to_bytes();
    let mut out = Vec::with_capacity(32 + n * 16 + sq16_codes.len() + graph_bytes.len());
    out.extend_from_slice(HNSW_DATA_MAGIC);
    out.extend_from_slice(&(n as u64).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    // Was reserved / alignment; now the stamped query beam (u32).
    out.extend_from_slice(&(ef_search as u32).to_le_bytes());
    for &id in doc_ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out.extend_from_slice(sq16_codes);
    out.extend_from_slice(&(graph_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&graph_bytes);
    out
}

/// Rebuild a resident [`HnswIndex`] from [`encode_hnsw`].
/// Returns `None` on any malformation so the caller falls back to the lazy
/// build or scan path rather than failing the query.
pub(crate) fn decode_hnsw(bytes: &[u8]) -> Option<HnswIndex> {
    let mut c = Cursor::new(bytes);
    if c.take(HNSW_DATA_MAGIC.len())? != HNSW_DATA_MAGIC {
        return None;
    }
    let n = c.u64()? as usize;
    let dim = c.u32()? as usize;
    let ef_search = c.u32()? as usize; // 0 on older bundles (was reserved)
    if dim == 0 {
        return None;
    }
    let mut doc_ids = Vec::with_capacity(n);
    for _ in 0..n {
        doc_ids.push(c.i128()?);
    }
    let plane = c.take(n.checked_mul(dim)?.checked_mul(2)?)?.to_vec();
    let graph_len = c.u64()? as usize;
    let graph_bytes = c.take(graph_len)?;
    let graph = Hnsw::from_bytes(graph_bytes)?;
    if graph.len() != n {
        return None;
    }
    let scorer = Sq16Scorer::from_codes(plane, dim, n);
    Some(HnswIndex {
        scorer,
        graph,
        doc_ids,
        dim,
        ef_search,
    })
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
pub(crate) struct GraphBundle {
    /// One `u64` digest of the covered doc-id population (opaque here; the
    /// supertable layer defines it).
    pub population_key: u64,
    /// Largest stable doc id the graph covers (the append-delta boundary).
    pub high_water_id: i128,
    pub centroid_graph: Vec<u8>,
    pub data_bundle: Option<Vec<u8>>,
}

/// Read the `(population_key, high_water_id)` header from a bundle's first
/// [`GRAPH_BUNDLE_HEADER_BYTES`] bytes. `None` on a bad magic or a short
/// read. Lets the settle path key on the covered population — and find the
/// append boundary — via a tiny range GET instead of the whole object.
pub(crate) fn graph_bundle_header(header: &[u8]) -> Option<(u64, i128)> {
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
fn put_opt_section(out: &mut Vec<u8>, section: Option<&[u8]>) {
    match section {
        Some(bytes) => {
            out.push(1);
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        None => out.push(0),
    }
}

fn take_opt_section(c: &mut Cursor<'_>) -> Option<Option<Vec<u8>>> {
    if c.take(1)?[0] == 0 {
        return Some(None);
    }
    let len = c.u64()? as usize;
    Some(Some(c.take(len)?.to_vec()))
}

/// Frame the graph sections into one slow-state blob, stamping the
/// `(high_water_id, count)` watermark into the fixed-offset header. The
/// data bundle and its provenance are omitted (a `0` flag) above the
/// data-graph scale ceiling.
pub(crate) fn encode_graph_bundle(
    population_key: u64,
    high_water_id: i128,
    centroid_graph: &[u8],
    data_bundle: Option<&[u8]>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(GRAPH_BUNDLE_HEADER_BYTES + 16 + centroid_graph.len());
    out.extend_from_slice(GRAPH_BUNDLE_MAGIC);
    out.extend_from_slice(&population_key.to_le_bytes());
    out.extend_from_slice(&high_water_id.to_le_bytes());
    out.extend_from_slice(&(centroid_graph.len() as u64).to_le_bytes());
    out.extend_from_slice(centroid_graph);
    put_opt_section(&mut out, data_bundle);
    out
}

/// Parse an [`encode_graph_bundle`] blob into its raw sections + header.
/// `None` on a bad magic or truncation, so a corrupt bundle degrades to a
/// fallback.
pub(crate) fn decode_graph_bundle(bytes: &[u8]) -> Option<GraphBundle> {
    let mut c = Cursor::new(bytes);
    if c.take(GRAPH_BUNDLE_MAGIC.len())? != GRAPH_BUNDLE_MAGIC {
        return None;
    }
    let population_key = c.u64()?;
    let high_water_id = c.i128()?;
    let centroid_len = c.u64()? as usize;
    let centroid_graph = c.take(centroid_len)?.to_vec();
    let data_bundle = take_opt_section(&mut c)?;
    Some(GraphBundle {
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
        let (choice, graph) = calibrate_graph(
            &scorer,
            &[32, 64, 128],
            &[128, 256, 512],
            0.90,
            0.01,
            200,
            100,
            10,
            0x5EED,
        );
        eprintln!(
            "[calib] m0={} ef={} recall={:.3} registered={} at_target={}",
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

        let bytes = encode_hnsw(&codes, &doc_ids, &graph, dim, 256);
        let idx = decode_hnsw(&bytes).expect("decode bundle");
        assert_eq!(idx.dim, dim);
        assert_eq!(idx.doc_ids, doc_ids);
        assert_eq!(idx.graph.len(), n);
        assert_eq!(
            idx.ef_search, 256,
            "stamped ef round-trips through the bundle"
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
        assert!(decode_hnsw(b"short").is_none());
    }

    /// The combined graph bundle frames its sections losslessly, including
    /// the absent-section flags (data/provenance omitted above the scale
    /// ceiling) and an empty centroid section.
    #[test]
    fn graph_bundle_frames_sections() {
        // Full bundle: centroid graph + data bundle + population key + high water.
        let centroid = vec![1u8, 2, 3, 4, 5];
        let data = vec![9u8; 300];
        let blob = encode_graph_bundle(0xDEAD_BEEF_1234, 987_654_321, &centroid, Some(&data));
        let got = decode_graph_bundle(&blob).expect("decode full");
        assert_eq!(got.population_key, 0xDEAD_BEEF_1234);
        assert_eq!(got.high_water_id, 987_654_321);
        assert_eq!(got.centroid_graph, centroid);
        assert_eq!(got.data_bundle.as_deref(), Some(&data[..]));
        // The header reads from the fixed-offset prefix alone (a tiny range
        // GET at settle time — no need for the multi-GiB body).
        assert_eq!(
            graph_bundle_header(&blob[..GRAPH_BUNDLE_HEADER_BYTES]),
            Some((0xDEAD_BEEF_1234, 987_654_321))
        );

        // Data-less bundle (above the scale ceiling): empty centroid, no data.
        let blob = encode_graph_bundle(0, 0, &[], None);
        let got = decode_graph_bundle(&blob).expect("decode empty");
        assert!(got.centroid_graph.is_empty());
        assert!(got.data_bundle.is_none());

        assert!(decode_graph_bundle(b"bad").is_none());
        assert!(graph_bundle_header(b"short").is_none());
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
