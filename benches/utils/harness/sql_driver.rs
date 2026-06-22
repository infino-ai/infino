// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic SQL driver.
//!
//! Builds one canonical 1-writer queryable artifact, optionally measures
//! an N-writer build-throughput row, and times SQL queries against the
//! canonical artifact. `run_sql_with_index` returns the artifact so
//! in-tree benches can run additional correctness/warm/cold checks before
//! calling `close`/`delete`.

use std::time::{Duration, Instant};

use super::{SqlEngine, SqlRow};
use crate::{
    markdown::fmt_count,
    rss::{PeakSampler, RssStats},
};

#[derive(Clone, Copy, Debug)]
pub struct SqlQuery {
    pub name: &'static str,
    pub sql: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlRunConfig {
    pub iters: usize,
    pub parallel: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SqlBuildStat {
    pub writers: usize,
    pub wall: Duration,
    pub rss: RssStats,
}

#[derive(Clone, Debug)]
pub struct SqlQueryStats {
    pub name: &'static str,
    pub p50: Duration,
    pub rss: RssStats,
    pub rows: usize,
}

#[derive(Clone, Debug)]
pub struct EngineSqlResult {
    pub engine: &'static str,
    pub builds: Vec<SqlBuildStat>,
    pub queries: Vec<SqlQueryStats>,
}

pub fn run_sql<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> EngineSqlResult {
    let (result, mut index) = run_sql_with_index::<E>(cfg, rows, queries);
    E::close(&mut index);
    E::delete(index);
    result
}

pub fn run_sql_with_index<E: SqlEngine>(
    cfg: SqlRunConfig,
    rows: &[SqlRow<'_>],
    queries: &[SqlQuery],
) -> (EngineSqlResult, E::Index) {
    eprintln!(
        "[harness/sql] {}: building 1-writer table over {} rows...",
        E::name(),
        fmt_count(rows.len()),
    );
    let mut index = E::open();
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    E::write(&mut index, rows);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();
    let mut builds = vec![SqlBuildStat {
        writers: 1,
        wall,
        rss,
    }];

    if cfg.parallel > 1 {
        eprintln!(
            "[harness/sql] {}: parallel build probe ({} writers)...",
            E::name(),
            cfg.parallel,
        );
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        E::parallel_write(rows, cfg.parallel);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();
        builds.push(SqlBuildStat {
            writers: cfg.parallel,
            wall,
            rss,
        });
    }

    // One battery-level progress line; per-query results land in the
    // report table, so per-query progress lines are just noise.
    if !queries.is_empty() {
        eprintln!(
            "[harness/sql] {}: warm query battery ({} queries × {} timed iters)...",
            E::name(),
            queries.len(),
            cfg.iters,
        );
    }
    let mut queries_out = Vec::with_capacity(queries.len());
    for q in queries {
        let sampler = PeakSampler::start_default();
        let warm = E::read(&index, q.sql);
        let mut samples = Vec::with_capacity(cfg.iters.max(1));
        for _ in 0..cfg.iters.max(1) {
            let t0 = Instant::now();
            let out = E::read(&index, q.sql);
            samples.push(t0.elapsed());
            std::hint::black_box(out);
        }
        let rss = sampler.stop_stats();
        queries_out.push(SqlQueryStats {
            name: q.name,
            p50: percentile_duration(&mut samples, 50),
            rss,
            rows: warm.rows,
        });
    }

    (
        EngineSqlResult {
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
