// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic FTS driver.
//!
//! [`run_fts`] executes the measured `open` → `write` → `read`
//! lifecycle for any [`FtsEngine`] over a shared corpus: the build
//! phase is timed + RSS-sampled in isolation, then each query is warmed
//! and timed (p50) with its own RSS window. The same code drives every
//! engine, so latency / memory differences come from the engines, not
//! the harness.

use std::time::{Duration, Instant};

use super::rss::{PeakSampler, RssStats};
use super::{BoolMode, FtsEngine, Hit};

/// One named query in the battery.
#[derive(Clone, Copy, Debug)]
pub struct FtsQuery {
    pub name: &'static str,
    pub terms: &'static [&'static str],
    pub mode: BoolMode,
}

/// Timing + memory for a single measured phase.
#[derive(Clone, Copy, Debug)]
pub struct PhaseStats {
    pub wall: Duration,
    pub rss: RssStats,
}

/// Per-query result: median latency, its RSS window, and the hit ids
/// (so a caller can grade recall across engines).
#[derive(Clone, Debug)]
pub struct QueryStats {
    pub name: &'static str,
    pub p50: Duration,
    pub rss: RssStats,
    pub hit_ids: Vec<u64>,
}

/// Everything one engine produced for the FTS modality.
#[derive(Clone, Debug)]
pub struct EngineFtsResult {
    pub engine: &'static str,
    pub build: PhaseStats,
    pub queries: Vec<QueryStats>,
}

/// Drive one engine through the full FTS lifecycle.
///
/// `docs` is the shared corpus (`(doc_id, text)`), built once by the
/// caller and reused across engines so its small footprint is outside
/// every engine's measured window. `k` is the top-k; `iters` is the
/// number of timed query repetitions (after one warmup).
pub fn run_fts<E: FtsEngine>(
    column: &str,
    docs: &[(u64, &str)],
    queries: &[FtsQuery],
    k: usize,
    iters: usize,
) -> EngineFtsResult {
    // ── write (build + seal), isolated and measured ──────────────────
    let mut index = E::open(column);
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    E::write(&mut index, docs);
    let build_wall = t0.elapsed();
    let build_rss = sampler.stop_stats();

    // ── read: per-query warmup + timed iters ─────────────────────────
    let mut queries_out = Vec::with_capacity(queries.len());
    for q in queries {
        let sampler = PeakSampler::start_default();
        // Warmup (fault pages, prime caches) — not timed.
        let warm = E::read(&index, q.terms, k, q.mode);
        let hit_ids: Vec<u64> = warm.iter().map(|h: &Hit| h.doc_id).collect();

        let mut samples = Vec::with_capacity(iters.max(1));
        for _ in 0..iters.max(1) {
            let t = Instant::now();
            let hits = E::read(&index, q.terms, k, q.mode);
            samples.push(t.elapsed());
            std::hint::black_box(hits);
        }
        let rss = sampler.stop_stats();
        queries_out.push(QueryStats {
            name: q.name,
            p50: percentile_duration(&mut samples, 50),
            rss,
            hit_ids,
        });
    }

    EngineFtsResult {
        engine: E::name(),
        build: PhaseStats {
            wall: build_wall,
            rss: build_rss,
        },
        queries: queries_out,
    }
}

/// Nearest-rank percentile of a duration sample set (sorts in place).
fn percentile_duration(samples: &mut [Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    let rank = ((percentile as f64 / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.saturating_sub(1).min(samples.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench_harness::{InfinoFtsEngine, MmapTextCorpus};

    /// 1M-scale validation: drive infino through the shared `run_fts`
    /// driver and print build + per-query stats, to confirm the driver
    /// reproduces infino's known superfile-FTS numbers. Ignored by
    /// default (heavy); run explicitly:
    /// `cargo test --features bench-harness --release -- --ignored \
    ///  --nocapture run_fts_infino_superfile_scale`
    #[test]
    #[ignore = "1M-scale; run explicitly in --release"]
    fn run_fts_infino_superfile_scale() {
        use crate::bench_harness::corpus::SUPERFILE_DOCS;
        use crate::bench_harness::fmt_bytes;

        let corpus = MmapTextCorpus::generate(SUPERFILE_DOCS, 1);
        let docs = corpus.rows();
        let queries = [
            FtsQuery {
                name: "single_rare",
                terms: &["term09999"],
                mode: BoolMode::Or,
            },
            FtsQuery {
                name: "single_common",
                terms: &["term00001"],
                mode: BoolMode::Or,
            },
            FtsQuery {
                name: "two_term_or",
                terms: &["term00001", "term00050"],
                mode: BoolMode::Or,
            },
            FtsQuery {
                name: "two_term_and",
                terms: &["term00001", "term00050"],
                mode: BoolMode::And,
            },
        ];
        let res = run_fts::<InfinoFtsEngine>("title", &docs, &queries, 10, 50);
        eprintln!(
            "[run_fts infino @{SUPERFILE_DOCS}] build wall={:.2}s peak_rss={}",
            res.build.wall.as_secs_f64(),
            fmt_bytes(res.build.rss.peak_rss_bytes),
        );
        for q in &res.queries {
            eprintln!(
                "  {:14} p50={:>10?}  rss={}  hits={}",
                q.name,
                q.p50,
                fmt_bytes(q.rss.peak_rss_bytes),
                q.hit_ids.len(),
            );
        }
        assert!(res.build.wall > Duration::ZERO);
        assert!(res.queries.iter().all(|q| !q.hit_ids.is_empty()));
    }

    #[test]
    fn run_fts_drives_infino_end_to_end() {
        let corpus = MmapTextCorpus::generate(2_000, 7);
        let docs = corpus.rows();
        let queries = [FtsQuery {
            name: "common",
            terms: &["term00001"],
            mode: BoolMode::Or,
        }];
        let res = run_fts::<InfinoFtsEngine>("title", &docs, &queries, 10, 3);

        assert_eq!(res.engine, "infino");
        assert!(res.build.wall > Duration::ZERO, "build should be timed");
        assert_eq!(res.queries.len(), 1);
        let q = &res.queries[0];
        assert!(!q.hit_ids.is_empty(), "common term should return hits");
        assert!(q.hit_ids.len() <= 10, "top-k respected");
    }
}
