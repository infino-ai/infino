// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Concurrent ingest + query contention harness.
//!
//! Measures sustained reader latency and throughput under two conditions:
//! - **baseline**: N readers per table firing queries in a tight loop, no writers.
//! - **contention**: same readers + 1 writer per table committing continuously.
//!
//! All `TENANTS` tables are built up front and stay open, with default pool
//! construction, so the process thread inventory grows with the tenant count
//! exactly as in a multi-tenant server. Both phases load **all tables at
//! once**; latencies aggregate across tables per modality, and peak OS
//! thread count is sampled per phase — the headline metric for pool /
//! runtime consolidation.
//!
//! Duration-based, not iteration-based: each condition runs a fixed
//! wall-clock window (default 15 s, 3 s warmup discarded) so readers and
//! writers genuinely overlap. Runs on a `multi_thread` tokio runtime so
//! `bridge_sync_to_async` takes the `block_in_place` path, as in production.
//!
//! Knobs (env vars):
//!   INFINO_BENCH_CONCURRENT_DOCS      corpus size per table (default 200_000)
//!   INFINO_BENCH_CONCURRENT_READERS   reader tasks per modality per table (default 8)
//!   INFINO_BENCH_CONCURRENT_TENANTS   tables open + loaded simultaneously (default 1)
//!   INFINO_BENCH_CONCURRENT_DURATION  measurement window in seconds (default 15)
//!   INFINO_BENCH_CONCURRENT_WARMUP    warmup seconds to discard (default 3)
//!   INFINO_BENCH_CONCURRENT_BASELINE  set to 0 to skip the no-writer phase
//!
//! Invoked as `cargo bench -- concurrent`.

use std::{
    hint::black_box,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use arrow_array::{FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use futures::future::join_all;
use infino::{
    VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::{FtsConfig, VectorConfig},
        fts::reader::BoolMode,
        vector::{distance::Metric, rerank_codec::RerankCodec},
    },
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_tokenizer,
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
const VEC_COLUMN: &str = "emb";
/// dim=128 is large enough for the shortlist+rerank CPU to show pool-routing
/// impact under contention, but small enough for fast fixture builds.
const VEC_DIM: usize = 128;
const VEC_N_CENT: usize = 32;
const VEC_ROT_SEED: u64 = 7;
const TOP_K: usize = 10;
const WRITER_BATCH: usize = 1_024;
const CORPUS_CHUNKS: usize = 8;
const FALLBACK_SIM_WORKERS: usize = 4;
/// Thread-count poll cadence — pools live for whole phases, so 200 ms
/// catches the peak.
const THREAD_SAMPLE_INTERVAL: Duration = Duration::from_millis(200);

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

fn run_baseline() -> bool {
    std::env::var("INFINO_BENCH_CONCURRENT_BASELINE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

// ─── OS thread count ──────────────────────────────────────────────────────────

/// Current OS thread count of this process (procfs on Linux, `ps -M`
/// on macOS).
fn current_thread_count() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        return s
            .lines()
            .find_map(|l| l.strip_prefix("Threads:"))
            .and_then(|rest| rest.trim().parse().ok());
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("ps")
            .args(["-M", "-p", &std::process::id().to_string()])
            .output()
            .ok()?;
        let rows = String::from_utf8_lossy(&out.stdout).lines().count();
        return (rows > 1).then(|| rows - 1);
    }
    #[allow(unreachable_code)]
    None
}

/// Background sampler recording peak OS thread count over a phase.
struct ThreadSampler {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<usize>,
}

impl ThreadSampler {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_t = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("thread-sampler".into())
            .spawn(move || {
                let mut peak = current_thread_count().unwrap_or(0);
                while !stop_t.load(Ordering::Relaxed) {
                    if let Some(n) = current_thread_count() {
                        peak = peak.max(n);
                    }
                    thread::sleep(THREAD_SAMPLE_INTERVAL);
                }
                peak
            })
            .expect("spawn thread-sampler");
        Self { stop, handle }
    }

    fn stop(self) -> usize {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.join().expect("thread-sampler join")
    }
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

fn concurrent_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            VEC_COLUMN,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                VEC_DIM as i32,
            ),
            false,
        ),
    ]))
}

fn build_batch(start: usize, n: usize) -> RecordBatch {
    let titles_owned: Vec<String> = (start..start + n)
        .map(|i| format!("alpha row{i:08}"))
        .collect();
    let titles: Vec<&str> = titles_owned.iter().map(|s| s.as_str()).collect();
    let title_arr = LargeStringArray::from(titles);

    // Deterministic non-zero vectors: coord j = (row_idx + j) as f32 % 1.0
    let floats: Vec<f32> = (start..start + n)
        .flat_map(|i| (0..VEC_DIM).map(move |j| ((i + j) % 97) as f32 + 0.1))
        .collect();
    let values = Arc::new(Float32Array::from(floats)) as Arc<dyn arrow_array::Array>;
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let emb_arr = FixedSizeListArray::try_new(item_field, VEC_DIM as i32, values, None)
        .expect("FixedSizeListArray from f32 values");

    RecordBatch::try_new(
        concurrent_schema(),
        vec![Arc::new(title_arr), Arc::new(emb_arr)],
    )
    .expect("RecordBatch shape matches concurrent_schema")
}

// Default pool construction — per-table pool/runtime growth is part of
// what this harness measures.
fn build_supertable_options(storage: Arc<dyn StorageProvider>) -> SupertableOptions {
    SupertableOptions::new(
        concurrent_schema(),
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![VectorConfig {
            column: VEC_COLUMN.into(),
            dim: VEC_DIM,
            n_cent: VEC_N_CENT,
            rot_seed: VEC_ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
            provided_centroids: None,
        }],
        Some(default_tokenizer()),
    )
    .expect("SupertableOptions with FTS + vector")
    .with_storage(storage)
}

fn build_fixture(n_docs: usize) -> Fixture {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("localfs"));
    let st = Supertable::create(build_supertable_options(storage)).expect("create supertable");

    let chunk_size = n_docs.div_ceil(CORPUS_CHUNKS);
    let mut w = st.writer().expect("writer");
    for chunk in 0..CORPUS_CHUNKS {
        let start = chunk * chunk_size;
        let end = ((chunk + 1) * chunk_size).min(n_docs);
        if start >= end {
            break;
        }
        let batch = build_batch(start, end - start);
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
    let mut batch_start = 1_000_000usize;
    while !stop.load(Ordering::Relaxed) {
        if let Ok(mut w) = st.writer() {
            let batch = build_batch(batch_start, WRITER_BATCH);
            let _ = w.append(&batch);
            let _ = w.commit();
            batch_start += WRITER_BATCH;
            commits += 1;
        }
    }
    commits
}

// Vector reader loop: rides the supertable fan-out, so its shortlist+rerank
// CPU lands on each table's reader_pool.
async fn vector_reader_loop(
    st: Supertable,
    stop: Arc<AtomicBool>,
    phase_start: Instant,
    warmup: Duration,
) -> Vec<Duration> {
    // Fixed unit-ish query vector — direction matters for cosine, magnitude doesn't.
    let query: Vec<f32> = (0..VEC_DIM).map(|j| (j % 13) as f32 + 0.5).collect();
    let mut latencies = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let t0 = Instant::now();
        let _ = black_box(
            st.vector_search(
                VEC_COLUMN,
                &query,
                TOP_K,
                VectorSearchOptions::new(),
                None,
                None,
            )
            .expect("vector_search"),
        );
        if phase_start.elapsed() > warmup {
            latencies.push(t0.elapsed());
        }
    }
    latencies
}

struct PhaseResult {
    fts: PhaseStat,
    vec: PhaseStat,
    /// Total commits across all tables' writers.
    commits: usize,
    /// Peak OS thread count observed during the phase (0 if unavailable).
    peak_threads: usize,
    /// Per-table query counts (fts, vec) — fairness check across tenants.
    per_table_n: Vec<(usize, usize)>,
}

// Drives n_readers fts + n_readers vec tasks per table on all tables at
// once, plus one writer per table when `with_writer`.
fn run_phase(
    tables: &[Supertable],
    n_readers: usize,
    with_writer: bool,
    total: Duration,
    warmup: Duration,
) -> PhaseResult {
    let rt = build_sim_runtime();
    let stop = Arc::new(AtomicBool::new(false));
    let phase_start = Instant::now();
    let threads = ThreadSampler::start();

    let writers: Vec<_> = if with_writer {
        tables
            .iter()
            .map(|st| {
                let st_w = st.clone();
                let stop_w = Arc::clone(&stop);
                rt.spawn(async move { writer_loop(st_w, stop_w).await })
            })
            .collect()
    } else {
        Vec::new()
    };

    // Grouped per table so per-table counts survive aggregation.
    let fts_readers: Vec<Vec<_>> = tables
        .iter()
        .map(|st| {
            (0..n_readers)
                .map(|_| {
                    let st_r = st.clone();
                    let stop_r = Arc::clone(&stop);
                    rt.spawn(async move { reader_loop(st_r, stop_r, phase_start, warmup).await })
                })
                .collect()
        })
        .collect();

    let vec_readers: Vec<Vec<_>> = tables
        .iter()
        .map(|st| {
            (0..n_readers)
                .map(|_| {
                    let st_r = st.clone();
                    let stop_r = Arc::clone(&stop);
                    rt.spawn(
                        async move { vector_reader_loop(st_r, stop_r, phase_start, warmup).await },
                    )
                })
                .collect()
        })
        .collect();

    // Sleep on the calling thread; the rt drives tasks concurrently.
    std::thread::sleep(total);
    stop.store(true, Ordering::Relaxed);

    let (fts_by_table, vec_by_table): (Vec<Vec<Duration>>, Vec<Vec<Duration>>) =
        rt.block_on(async {
            let mut fts = Vec::with_capacity(fts_readers.len());
            for group in fts_readers {
                let lats: Vec<Duration> = join_all(group)
                    .await
                    .into_iter()
                    .flat_map(|r| r.expect("fts reader task"))
                    .collect();
                fts.push(lats);
            }
            let mut vec_l = Vec::with_capacity(vec_readers.len());
            for group in vec_readers {
                let lats: Vec<Duration> = join_all(group)
                    .await
                    .into_iter()
                    .flat_map(|r| r.expect("vec reader task"))
                    .collect();
                vec_l.push(lats);
            }
            (fts, vec_l)
        });

    let commits: usize = rt.block_on(async {
        join_all(writers)
            .await
            .into_iter()
            .map(|r| r.expect("writer task"))
            .sum()
    });

    let peak_threads = threads.stop();

    let per_table_n: Vec<(usize, usize)> = fts_by_table
        .iter()
        .zip(&vec_by_table)
        .map(|(f, v)| (f.len(), v.len()))
        .collect();
    let fts_all: Vec<Duration> = fts_by_table.into_iter().flatten().collect();
    let vec_all: Vec<Duration> = vec_by_table.into_iter().flatten().collect();

    let measure_secs = (total - warmup).as_secs_f64();
    PhaseResult {
        fts: stat_from(fts_all, measure_secs),
        vec: stat_from(vec_all, measure_secs),
        commits,
        peak_threads,
        per_table_n,
    }
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
        "[concurrent] {tenants} table(s), {docs} docs/{CORPUS_CHUNKS} superfiles each, \
         {readers} fts + {readers} vec readers per table, {:.0}s window ({:.0}s warmup discarded)",
        dur.as_secs_f64(),
        warmup.as_secs_f64(),
    );

    // All tables built up front and kept alive, as in a multi-tenant server.
    let fixtures: Vec<Fixture> = (0..tenants)
        .map(|t| {
            eprintln!("[concurrent] building table {t}...");
            build_fixture(docs)
        })
        .collect();
    let tables: Vec<Supertable> = fixtures.iter().map(|f| f.st.clone()).collect();

    let idle_threads = current_thread_count().unwrap_or(0);
    eprintln!("[concurrent] {idle_threads} OS threads after build, before load");

    let mut report = Report::load("concurrent");
    let mut rows: Vec<Vec<Cell>> = Vec::new();

    let with_baseline = run_baseline();

    let base = if with_baseline {
        eprintln!("[concurrent] baseline: readers on all {tenants} table(s), no writers...");
        Some(run_phase(&tables, readers, false, dur, warmup))
    } else {
        eprintln!("[concurrent] skipping baseline (INFINO_BENCH_CONCURRENT_BASELINE=0)");
        None
    };

    eprintln!("[concurrent] contention: readers + 1 writer per table on all {tenants} table(s)...");
    let contend = run_phase(&tables, readers, true, dur, warmup);
    let commits = contend.commits;

    for (t, (f_n, v_n)) in contend.per_table_n.iter().enumerate() {
        eprintln!(
            "[concurrent] contention table {t}: {:.0} fts q/s, {:.0} vec q/s",
            *f_n as f64 / measure_secs,
            *v_n as f64 / measure_secs,
        );
    }

    let label = if tenants == 1 {
        "single table".to_string()
    } else {
        format!("{tenants} tables agg")
    };

    // Emit one row per modality (FTS, vector) per condition (baseline, contention).
    #[allow(clippy::type_complexity)]
    let modalities: &[(&str, fn(&PhaseResult) -> &PhaseStat)] =
        &[("fts", |r| &r.fts), ("vec", |r| &r.vec)];

    for (modality, stat_fn) in modalities {
        let contend_stat = stat_fn(&contend);

        if let Some(ref b) = base {
            let base_stat = stat_fn(b);
            let base_p50 = base_stat.p50.as_nanos() as f64;
            let base_p95 = base_stat.p95.as_nanos() as f64;
            let base_p99 = base_stat.p99.as_nanos() as f64;
            rows.push(vec![
                text(format!("{label} / {modality}")),
                text("baseline".to_string()),
                metric(base_p50, fmt_time(base_p50), Better::Lower),
                metric(base_p95, fmt_time(base_p95), Better::Lower),
                metric(base_p99, fmt_time(base_p99), Better::Lower),
                metric(
                    base_stat.qps,
                    format!("{:.0} q/s", base_stat.qps),
                    Better::Higher,
                ),
                text(format!("{}", base_stat.n)),
            ]);
        }

        let p99_delta_pct = base.as_ref().map(|b| {
            let base_stat = stat_fn(b);
            if base_stat.p99 > Duration::ZERO {
                100.0 * (contend_stat.p99.as_secs_f64() - base_stat.p99.as_secs_f64())
                    / base_stat.p99.as_secs_f64()
            } else {
                0.0
            }
        });
        let qps_delta_pct = base.as_ref().map(|b| {
            let base_stat = stat_fn(b);
            if base_stat.qps > 0.0 {
                100.0 * (contend_stat.qps - base_stat.qps) / base_stat.qps
            } else {
                0.0
            }
        });

        let cp50 = contend_stat.p50.as_nanos() as f64;
        let cp95 = contend_stat.p95.as_nanos() as f64;
        let cp99 = contend_stat.p99.as_nanos() as f64;
        let n_note = match p99_delta_pct {
            Some(d) => format!("{} / {} commits (p99 {:+.1}%)", contend_stat.n, commits, d),
            None => format!("{} / {} commits", contend_stat.n, commits),
        };
        let qps_label = match qps_delta_pct {
            Some(d) => format!("{:.0} q/s ({:+.1}%)", contend_stat.qps, d),
            None => format!("{:.0} q/s", contend_stat.qps),
        };
        rows.push(vec![
            text(format!("{label} / {modality}")),
            text(format!("{readers}r+1w")),
            metric(cp50, fmt_time(cp50), Better::Lower),
            metric(cp95, fmt_time(cp95), Better::Lower),
            metric(cp99, fmt_time(cp99), Better::Lower),
            metric(contend_stat.qps, qps_label, Better::Higher),
            text(n_note),
        ]);

        eprintln!(
            "[concurrent] {label} / {modality}: baseline {:.0} q/s | contention {:.0} q/s | p99 {} | writers {:.1} commits/s",
            base.as_ref().map(|b| stat_fn(b).qps).unwrap_or(0.0),
            contend_stat.qps,
            match p99_delta_pct {
                Some(d) => format!("{:+.1}%", d),
                None => "—".into(),
            },
            commits as f64 / measure_secs,
        );
    }

    // OS thread inventory — the consolidation headline.
    let mut thread_rows: Vec<Vec<Cell>> = vec![vec![
        text("idle after build".to_string()),
        metric(
            idle_threads as f64,
            format!("{idle_threads}"),
            Better::Lower,
        ),
    ]];
    if let Some(ref b) = base {
        thread_rows.push(vec![
            text("baseline peak".to_string()),
            metric(
                b.peak_threads as f64,
                format!("{}", b.peak_threads),
                Better::Lower,
            ),
        ]);
    }
    thread_rows.push(vec![
        text("contention peak".to_string()),
        metric(
            contend.peak_threads as f64,
            format!("{}", contend.peak_threads),
            Better::Lower,
        ),
    ]);
    eprintln!(
        "[concurrent] OS threads: idle {idle_threads} | baseline peak {} | contention peak {}",
        base.as_ref().map(|b| b.peak_threads).unwrap_or(0),
        contend.peak_threads,
    );

    report.emit(&Section {
        anchor: "bench/concurrent/contention".into(),
        title: format!(
            "Concurrent ingest+query — {tenants} table(s), {docs} docs each, {readers} fts+vec readers/table, {:.0}s window",
            dur.as_secs_f64()
        ),
        note: format!(
            "Duration-based ({:.0}s total, {:.0}s warmup discarded). All tables open and \
             loaded simultaneously with default pool construction; latencies aggregate across tables. \
             Each table gets {readers} FTS (bm25_search) + {readers} vector (vector_search dim={VEC_DIM}) \
             reader tasks in tight loops; each writer commits {WRITER_BATCH}-row batches continuously. \
             Runs on multi_thread tokio runtime (bridge_sync_to_async → block_in_place). \
             QPS delta and p99 delta measure contention overhead vs baseline; the OS-thread table \
             tracks per-table pool/runtime growth. \
             INFINO_BENCH_CONCURRENT_DOCS/READERS/TENANTS/DURATION/WARMUP to adjust. Δ vs previous run.",
            dur.as_secs_f64(),
            warmup.as_secs_f64(),
        ),
        blocks: vec![
            Block {
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
            },
            Block {
                subtitle: "OS threads".into(),
                headers: vec!["Phase".into(), "peak".into()],
                rows: thread_rows,
            },
        ],
    });
    report.save();
}
