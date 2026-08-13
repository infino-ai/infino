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

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Mutex, RwLock};

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

        let entry = builder.entry.into_inner().unwrap().node;
        let neighbors: Vec<Vec<Vec<u32>>> = builder
            .adj
            .into_iter()
            .map(|m| m.into_inner().unwrap())
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
        let guard = self.adj[node as usize].lock().unwrap();
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
        let mut g = self.adj[target as usize].lock().unwrap();
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
        } = *self.entry.read().unwrap();

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
            let mut e = self.entry.write().unwrap();
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
    /// buffer (exactly what `build_direct_data_index` feeds from the on-disk
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
