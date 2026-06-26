// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Concurrent ingest + query contention harness.
//!
//! Measures sustained reader latency and throughput under two conditions:
//! - **baseline**: N readers firing queries in a tight loop, no writer.
//! - **contention**: same N readers + 1 writer committing continuously.
//!
//! **Duration-based** (not iteration-based): each condition runs for a fixed
//! wall-clock window (default 15 s; 3 s warmup discarded). Every query that
//! completes during the measurement window is recorded. This guarantees
//! sustained overlap between readers and the writer — the failure mode of
//! iteration-based benches is that readers finish in milliseconds before the
//! writer commits even once.
//!
//! Runs on a `multi_thread` tokio runtime so `bridge_sync_to_async` takes
//! the `block_in_place` path — the same code path as axum/SaaS production.
//!
//! Knobs (env vars):
//!   INFINO_BENCH_CONCURRENT_DOCS      corpus size (default 200_000)
//!   INFINO_BENCH_CONCURRENT_READERS   concurrent reader tasks (default 8)
//!   INFINO_BENCH_CONCURRENT_TENANTS   tables open simultaneously (default 1)
//!   INFINO_BENCH_CONCURRENT_DURATION  measurement window in seconds (default 15)
//!   INFINO_BENCH_CONCURRENT_WARMUP    warmup seconds to discard (default 3)
//!
//! Invoked as `cargo bench -- concurrent`.

use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use futures::future::join_all;
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::fts::reader::BoolMode,
    supertable::Supertable,
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

use crate::{
    markdown::fmt_time,
    report::{Better, Block, Cell, Report, Section, metric, text},
};

const DEFAULT_DOCS: usize = 200_000;
const DEFAULT_READERS: usize = 8;
const DEFAULT_TENANTS: usize = 1;
const DEFAULT_DURATION_SECS: u64 = 15;
const DEFAULT_WARMUP_SECS: u64 = 3;
const QUERY_FIELD: &str = "title";
const QUERY_TERM: &str = "alpha";
const TOP_K: usize = 10;
const WRITER_BATCH: usize = 1_024;
const CORPUS_CHUNKS: usize = 8;
const FALLBACK_SIM_WORKERS: usize = 4;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn n_docs() -> usize {
    env_usize("INFINO_BENCH_CONCURRENT_DOCS", DEFAULT_DOCS)
}

fn n_readers() -> usize {
    env_usize("INFINO_BENCH_CONCURRENT_READERS", DEFAULT_READERS)
}

fn n_tenants() -> usize {
    env_usize("INFINO_BENCH_CONCURRENT_TENANTS", DEFAULT_TENANTS)
}

fn duration_secs() -> u64 {
    env_u64("INFINO_BENCH_CONCURRENT_DURATION", DEFAULT_DURATION_SECS)
}

fn warmup_secs() -> u64 {
    env_u64("INFINO_BENCH_CONCURRENT_WARMUP", DEFAULT_WARMUP_SECS)
}

// ─── Runtime ──────────────────────────────────────────────────────────────────

// Simulates the SaaS/axum process-level runtime.
fn build_sim_runtime() -> tokio::runtime::Runtime {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(FALLBACK_SIM_WORKERS);
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .expect("sim runtime")
}

// ─── Fixture ──────────────────────────────────────────────────────────────────

struct Fixture {
    st: Supertable,
    _dir: TempDir,
}

fn build_fixture(n_docs: usize) -> Fixture {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
    let st = Supertable::create(default_supertable_options().with_storage(storage))
        .expect("create supertable");

    let chunk_size = n_docs.div_ceil(CORPUS_CHUNKS);
    let mut w = st.writer().expect("writer");
    for chunk in 0..CORPUS_CHUNKS {
        let start = chunk * chunk_size;
        let end = ((chunk + 1) * chunk_size).min(n_docs);
        if start >= end {
            break;
        }
        let titles_owned: Vec<String> = (start..end).map(|i| format!("alpha row{i:08}")).collect();
        let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
        let batch = build_title_batch(&titles);
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    Fixture { st, _dir: dir }
}

// ─── Measurement ──────────────────────────────────────────────────────────────

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

struct PhaseStat {
    p50: Duration,
    p95: Duration,
    p99: Duration,
    n: usize,
    qps: f64,
}

fn stat_from(mut latencies: Vec<Duration>, measure_secs: f64) -> PhaseStat {
    latencies.sort_unstable();
    let n = latencies.len();
    let qps = n as f64 / measure_secs;
    PhaseStat {
        p50: percentile(&latencies, 50.0),
        p95: percentile(&latencies, 95.0),
        p99: percentile(&latencies, 99.0),
        n,
        qps,
    }
}

// Each reader task fires queries in a tight loop for the entire phase window.
// Latencies recorded only after the warmup period — warmup opens lazy readers
// and populates caches without inflating measured numbers.
async fn reader_loop(
    st: Supertable,
    stop: Arc<AtomicBool>,
    phase_start: Instant,
    warmup: Duration,
) -> Vec<Duration> {
    let mut latencies = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        let _ = black_box(
            st.reader()
                .bm25_search(QUERY_FIELD, QUERY_TERM, TOP_K, BoolMode::Or, None)
                .expect("bm25_search"),
        );
        if phase_start.elapsed() > warmup {
            latencies.push(t0.elapsed());
        }
    }
    latencies
}

// Writer loop: continuous append+commit for the entire phase window.
// Single-writer slot is fine — in production each table has one writer.
async fn writer_loop(st: Supertable, stop: Arc<AtomicBool>) -> usize {
    let mut commits = 0usize;
    let mut batch_start = 0usize;
    while !stop.load(Ordering::Relaxed) {
        if let Ok(mut w) = st.writer() {
            let start = batch_start;
            let titles_owned: Vec<String> = (start..start + WRITER_BATCH)
                .map(|i| format!("alpha extra{i:08}"))
                .collect();
            let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
            let batch = build_title_batch(&titles);
            let _ = w.append(&batch);
            let _ = w.commit();
            batch_start += WRITER_BATCH;
            commits += 1;
        }
    }
    commits
}

fn run_phase(
    st: &Supertable,
    n_readers: usize,
    with_writer: bool,
    total: Duration,
    warmup: Duration,
) -> (PhaseStat, usize) {
    let rt = build_sim_runtime();
    let stop = Arc::new(AtomicBool::new(false));
    let phase_start = Instant::now();

    let writer = if with_writer {
        let st_w = st.clone();
        let stop_w = Arc::clone(&stop);
        Some(rt.spawn(async move { writer_loop(st_w, stop_w).await }))
    } else {
        None
    };

    let readers: Vec<_> = (0..n_readers)
        .map(|_| {
            let st_r = st.clone();
            let stop_r = Arc::clone(&stop);
            rt.spawn(async move { reader_loop(st_r, stop_r, phase_start, warmup).await })
        })
        .collect();

    // Sleep on the calling thread; the rt drives tasks concurrently.
    std::thread::sleep(total);
    stop.store(true, Ordering::Relaxed);

    let all: Vec<Duration> = rt.block_on(async {
        join_all(readers)
            .await
            .into_iter()
            .flat_map(|r| r.expect("reader task"))
            .collect()
    });

    let commits = match writer {
        Some(w) => rt.block_on(w).expect("writer task"),
        None => 0,
    };

    let measure_secs = (total - warmup).as_secs_f64();
    (stat_from(all, measure_secs), commits)
}

// ─── Entry point ──────────────────────────────────────────────────────────────

pub fn run() {
    let docs = n_docs();
    let readers = n_readers();
    let tenants = n_tenants();
    let dur = Duration::from_secs(duration_secs());
    let warmup = Duration::from_secs(warmup_secs());

    let measure_secs = (dur - warmup).as_secs_f64();

    eprintln!(
        "[concurrent] {tenants} table(s), {docs} docs/{CORPUS_CHUNKS} superfiles, \
         {readers} reader tasks, {:.0}s window ({:.0}s warmup discarded)",
        dur.as_secs_f64(),
        warmup.as_secs_f64(),
    );

    let mut report = Report::load("concurrent");
    let mut rows: Vec<Vec<Cell>> = Vec::new();

    for tenant in 0..tenants {
        let fixture = build_fixture(docs);

        eprintln!("[concurrent] table {tenant}: baseline ({readers} readers, no writer)...");
        let (base, _) = run_phase(&fixture.st, readers, false, dur, warmup);

        eprintln!("[concurrent] table {tenant}: contention ({readers}r+1w)...");
        let (contend, commits) = run_phase(&fixture.st, readers, true, dur, warmup);

        let label = if tenants == 1 {
            "single table".to_string()
        } else {
            format!("table {tenant}")
        };

        let p99_delta_pct = if base.p99 > Duration::ZERO {
            100.0 * (contend.p99.as_secs_f64() - base.p99.as_secs_f64()) / base.p99.as_secs_f64()
        } else {
            0.0
        };

        let qps_delta_pct = if base.qps > 0.0 {
            100.0 * (contend.qps - base.qps) / base.qps
        } else {
            0.0
        };

        let base_p50 = base.p50.as_nanos() as f64;
        let base_p95 = base.p95.as_nanos() as f64;
        let base_p99 = base.p99.as_nanos() as f64;
        let contend_p50 = contend.p50.as_nanos() as f64;
        let contend_p95 = contend.p95.as_nanos() as f64;
        let contend_p99 = contend.p99.as_nanos() as f64;

        rows.push(vec![
            text(label.clone()),
            text("baseline"),
            metric(base_p50, fmt_time(base_p50), Better::Lower),
            metric(base_p95, fmt_time(base_p95), Better::Lower),
            metric(base_p99, fmt_time(base_p99), Better::Lower),
            metric(base.qps, format!("{:.0} q/s", base.qps), Better::Higher),
            text(format!("{}", base.n)),
        ]);
        rows.push(vec![
            text(label),
            text(format!("{readers}r+1w")),
            metric(contend_p50, fmt_time(contend_p50), Better::Lower),
            metric(contend_p95, fmt_time(contend_p95), Better::Lower),
            metric(contend_p99, fmt_time(contend_p99), Better::Lower),
            metric(
                contend.qps,
                format!("{:.0} q/s ({:+.1}%)", contend.qps, qps_delta_pct),
                Better::Higher,
            ),
            text(format!(
                "{} / {} commits (p99 {:+.1}%)",
                contend.n, commits, p99_delta_pct
            )),
        ]);

        eprintln!(
            "[concurrent] table {tenant}: baseline {:.0} q/s | contention {:.0} q/s ({:+.1}%) | writer {:.1} commits/s | p99 {:+.1}%",
            base.qps,
            contend.qps,
            qps_delta_pct,
            commits as f64 / measure_secs,
            p99_delta_pct,
        );
    }

    report.emit(&Section {
        anchor: "bench/concurrent/contention".into(),
        title: format!(
            "Concurrent ingest+query — {tenants} table(s), {docs} docs, {readers} readers, {:.0}s window",
            dur.as_secs_f64()
        ),
        note: format!(
            "Duration-based ({:.0}s total, {:.0}s warmup discarded). \
             Readers fire in tight loops; writer commits {WRITER_BATCH}-row batches continuously. \
             Runs on multi_thread tokio runtime (bridge_sync_to_async → block_in_place). \
             QPS delta and p99 delta measure contention overhead. \
             INFINO_BENCH_CONCURRENT_DOCS/READERS/TENANTS/DURATION/WARMUP to adjust. Δ vs previous run.",
            dur.as_secs_f64(),
            warmup.as_secs_f64(),
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Table".into(),
                "Condition".into(),
                "p50".into(),
                "p95".into(),
                "p99".into(),
                "q/s".into(),
                "n / commits".into(),
            ],
            rows,
        }],
    });
    report.save();
}
