// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic vector driver.
//!
//! Mirrors the FTS driver: one measured 1-writer build produces the
//! canonical queryable artifact, an optional `N writers` build-throughput
//! probe is measured separately, and queries run against the canonical
//! artifact. `run_vector_with_index` returns the index so in-tree benches
//! can run correctness and cold upload against the exact bytes that were
//! just measured.

use std::time::{Duration, Instant};

use super::{VectorEngine, VectorHit};
use crate::{
    corpus,
    markdown::fmt_count,
    rss::{PeakSampler, RssStats},
};

/// Metric requested by the benchmark harness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorMetric {
    L2Sq,
    Cosine,
    NegDot,
}

/// Search parameters for a vector query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorSearch {
    pub nprobe: usize,
    pub rerank_mult: usize,
}

/// One named vector query and its search parameters.
#[derive(Clone, Copy, Debug)]
pub struct VectorQuery<'a> {
    pub name: &'static str,
    pub vector: &'a [f32],
    pub search: VectorSearch,
}

#[derive(Clone, Copy, Debug)]
pub struct VectorRunConfig<'a> {
    pub column: &'a str,
    pub dim: usize,
    pub metric: VectorMetric,
    pub k: usize,
    pub iters: usize,
    pub parallel: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct VectorBuildStat {
    pub writers: usize,
    pub wall: Duration,
    pub rss: RssStats,
}

#[derive(Clone, Debug)]
pub struct VectorQueryStats {
    pub name: &'static str,
    pub p50: Duration,
    pub rss: RssStats,
    pub hit_ids: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct EngineVectorResult {
    pub engine: &'static str,
    pub builds: Vec<VectorBuildStat>,
    pub queries: Vec<VectorQueryStats>,
}

pub fn run_vector<E: VectorEngine>(
    cfg: VectorRunConfig<'_>,
    vectors: &[f32],
    queries: &[VectorQuery<'_>],
) -> EngineVectorResult {
    let (result, mut index) = run_vector_with_index::<E>(cfg, vectors, queries);
    E::close(&mut index);
    E::delete(index);
    result
}

pub fn run_vector_with_index<E: VectorEngine>(
    cfg: VectorRunConfig<'_>,
    vectors: &[f32],
    queries: &[VectorQuery<'_>],
) -> (EngineVectorResult, E::Index) {
    let n_docs = vectors.len() / cfg.dim;
    let n_cent = corpus::n_cent(n_docs);

    eprintln!(
        "[harness/vector] {}: building 1-writer index over {} docs × dim={}...",
        E::name(),
        fmt_count(n_docs),
        cfg.dim,
    );
    let mut index = E::open(cfg.column, cfg.dim, cfg.metric, n_cent);
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    E::write(&mut index, vectors);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();
    let mut builds = vec![VectorBuildStat {
        writers: 1,
        wall,
        rss,
    }];

    if cfg.parallel > 1 {
        eprintln!(
            "[harness/vector] {}: parallel build probe ({} writers)...",
            E::name(),
            cfg.parallel,
        );
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        E::parallel_write(cfg.column, vectors, cfg.dim, cfg.metric, cfg.parallel);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();
        builds.push(VectorBuildStat {
            writers: cfg.parallel,
            wall,
            rss,
        });
    }

    // One battery-level progress line; per-query results land in the
    // report table, so per-query progress lines are just noise.
    if !queries.is_empty() {
        eprintln!(
            "[harness/vector] {}: warm search battery ({} queries × {} timed iters)...",
            E::name(),
            queries.len(),
            cfg.iters,
        );
    }
    let mut queries_out = Vec::with_capacity(queries.len());
    for q in queries {
        let sampler = PeakSampler::start_default();
        let warm = E::read(&index, q.vector, cfg.k, q.search);
        let hit_ids: Vec<u64> = warm.iter().map(|h: &VectorHit| h.doc_id).collect();

        let mut samples = Vec::with_capacity(cfg.iters.max(1));
        for _ in 0..cfg.iters.max(1) {
            let t0 = Instant::now();
            let hits = E::read(&index, q.vector, cfg.k, q.search);
            samples.push(t0.elapsed());
            std::hint::black_box(hits);
        }
        let rss = sampler.stop_stats();
        queries_out.push(VectorQueryStats {
            name: q.name,
            p50: percentile_duration(&mut samples, 50),
            rss,
            hit_ids,
        });
    }

    (
        EngineVectorResult {
            engine: E::name(),
            builds,
            queries: queries_out,
        },
        index,
    )
}

fn percentile_duration(samples: &mut [Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = ((percentile as f64 / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}
