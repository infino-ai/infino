// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable object-store bench (infino-only entry point).
//!
//! Multi-segment ingest to object storage at the supertable scale
//! (`INFINO_BENCH_SUPERTABLE_DOCS`, default 10M), built through the
//! production `SupertableWriter::append` + `commit` path. Three index
//! shapes are measured for apples-to-apples comparison against
//! single-modality peers: FTS-only, vector-only, SQL, and combined FTS +
//! vector.
//!
//! **Real object store only** (`INFINO_BENCH_STORE=s3` or `azure`). The
//! multi-commit build relies on conditional `If-Match` PUTs that the
//! `s3s-fs` emulator does not implement, so this bench rejects `s3s_fs` (the
//! default) and exits with a message otherwise. Every object the run writes
//! lands under one unique prefix per shape, all deleted before the runner
//! returns (unless `INFINO_BENCH_KEEP_TABLE` is set).
//!
//! ## Per-shape process isolation
//!
//! Each shape is built in its **own subprocess** (the parent re-execs this
//! same bench binary with `INFINO_BENCH_SUPERTABLE_SHAPE=<shape>`). RSS is
//! sampled inside that child, so each shape's Peak/Median/P90 are measured
//! from a clean address space. Within a single process `VmRSS` is a
//! monotonic high-water mark — the allocator does not return freed pages to
//! the OS — so running all three shapes in one process would let whichever
//! ran first poison the memory numbers of the ones after it. Isolation makes
//! the three rows independent and comparable.
//!
//! ## Invocation
//!
//! ```text
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket cargo bench --bench supertable_all
//! INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
//!   AZURE_STORAGE_ACCOUNT_NAME=... AZURE_STORAGE_ACCOUNT_KEY=... cargo bench --bench supertable_all
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_SUPERTABLE_DOCS=100000 cargo bench --bench supertable_all
//! ```

use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use infino::superfile::fts::reader::BoolMode as InfinoBoolMode;
use infino::superfile::reader::VectorSearchOptions;
use infino::supertable::Supertable;
use tempfile::TempDir;

use crate::corpus::DIM;
use crate::harness::BoolMode;
use crate::ingest::supertable::{self, Modality};
use crate::markdown::{fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, PeakSampler};
use crate::tiers;

/// Env var the parent sets to make a child build exactly one shape and
/// print its metrics instead of emitting the report.
const SHAPE_ENV: &str = "INFINO_BENCH_SUPERTABLE_SHAPE";
/// Line prefix a child writes to stdout carrying its measured metrics.
const RESULT_PREFIX: &str = "__SUPERTABLE_SHAPE_RESULT__ ";

/// The three measured shapes: (display label, child-env key, modality).
const SHAPES: [(&str, &str, Modality); 4] = [
    ("FTS-only", "fts", Modality::Fts),
    ("vector-only", "vector", Modality::Vector),
    ("SQL", "sql", Modality::Sql),
    ("combined FTS + vector", "combined", Modality::Combined),
];

/// Plain measured numbers for one shape, marshalled across the
/// parent/child process boundary as a single `key=value` line.
pub struct ShapeMetrics {
    pub wall_ns: f64,
    pub n_superfiles: usize,
    pub peak_rss_bytes: u64,
    pub median_rss_bytes: u64,
    pub p90_rss_bytes: u64,
}

pub struct SupertableShapeResult {
    pub label: &'static str,
    pub key: &'static str,
    pub metrics: ShapeMetrics,
}

impl ShapeMetrics {
    /// Render as the single stdout line the parent parses.
    fn to_result_line(&self) -> String {
        format!(
            "{RESULT_PREFIX}wall_ns={} n_superfiles={} peak={} median={} p90={}",
            self.wall_ns,
            self.n_superfiles,
            self.peak_rss_bytes,
            self.median_rss_bytes,
            self.p90_rss_bytes,
        )
    }

    /// Parse the line emitted by [`to_result_line`]. Returns `None` if a
    /// field is missing or unparseable.
    fn from_result_line(line: &str) -> Option<Self> {
        let body = line.strip_prefix(RESULT_PREFIX)?;
        let mut wall_ns = None;
        let mut n_superfiles = None;
        let mut peak = None;
        let mut median = None;
        let mut p90 = None;
        for tok in body.split_whitespace() {
            let (k, v) = tok.split_once('=')?;
            match k {
                "wall_ns" => wall_ns = v.parse().ok(),
                "n_superfiles" => n_superfiles = v.parse().ok(),
                "peak" => peak = v.parse().ok(),
                "median" => median = v.parse().ok(),
                "p90" => p90 = v.parse().ok(),
                _ => {}
            }
        }
        Some(ShapeMetrics {
            wall_ns: wall_ns?,
            n_superfiles: n_superfiles?,
            peak_rss_bytes: peak?,
            median_rss_bytes: median?,
            p90_rss_bytes: p90?,
        })
    }
}

fn modality_for_key(key: &str) -> Option<Modality> {
    SHAPES
        .iter()
        .find(|(_, k, _)| *k == key)
        .map(|(_, _, m)| *m)
}

/// Child entry point: build exactly one shape, sample its RSS in this
/// fresh process, clean up the real-S3 prefix it wrote, and print the
/// metrics line. Does not emit the report.
fn run_child_shape(key: &str) {
    let modality = match modality_for_key(key) {
        Some(m) => m,
        None => {
            eprintln!("[supertable] unknown shape key {key:?}");
            std::process::exit(2);
        }
    };

    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = supertable::build_on_storage(modality);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();

    // This child wrote its own unique prefix; delete it before exiting so the
    // real-backend run accrues no ongoing cost (ingest-only bench — the
    // artifact is not reused after the build is measured).
    if let Some(cleanup) = &built.cleanup {
        crate::tiers::cleanup_prefix(cleanup);
    }

    let metrics = ShapeMetrics {
        wall_ns: wall.as_secs_f64() * 1e9,
        n_superfiles: built.n_superfiles,
        peak_rss_bytes: rss.peak_rss_bytes,
        median_rss_bytes: rss.median_rss_bytes,
        p90_rss_bytes: rss.p90_rss_bytes,
    };
    println!("{}", metrics.to_result_line());
}

/// Spawn one isolated child to build `key` and return its metrics.
/// stderr is inherited so the child's `[tiers]` logs stream live; stdout
/// is captured to read back the single result line.
fn build_shape_isolated(key: &str) -> Option<ShapeMetrics> {
    let exe = std::env::current_exe().expect("current_exe for supertable child");
    let output = Command::new(exe)
        .env(SHAPE_ENV, key)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .expect("spawn supertable shape child");
    if !output.status.success() {
        eprintln!(
            "[supertable] shape {key:?} child exited with {} — skipping its row",
            output.status
        );
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let metrics = stdout.lines().find_map(ShapeMetrics::from_result_line);
    if metrics.is_none() {
        eprintln!("[supertable] shape {key:?} child produced no result line — skipping its row");
    }
    metrics
}

pub fn handle_shape_child_from_env() -> bool {
    if let Ok(key) = std::env::var(SHAPE_ENV) {
        run_child_shape(&key);
        true
    } else {
        false
    }
}

pub fn run_ingest_shapes_isolated() -> Vec<SupertableShapeResult> {
    let mut results = Vec::with_capacity(SHAPES.len());
    for (label, key, _) in SHAPES {
        eprintln!("[supertable] === shape {label} (isolated process) ===");
        if let Some(metrics) = build_shape_isolated(key) {
            results.push(SupertableShapeResult {
                label,
                key,
                metrics,
            });
        }
    }
    results
}

pub fn ingest_row(n_docs: usize, label: &str, m: &ShapeMetrics) -> Vec<Cell> {
    let secs = m.wall_ns / 1e9;
    let thr = if secs > 0.0 {
        n_docs as f64 / secs
    } else {
        0.0
    };
    vec![
        text(label),
        metric(m.wall_ns, fmt_time(m.wall_ns), Better::Lower),
        metric(thr, fmt_throughput(thr), Better::Higher),
        text(fmt_count(m.n_superfiles)),
        metric(
            m.peak_rss_bytes as f64,
            rss::fmt_bytes(m.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            m.median_rss_bytes as f64,
            rss::fmt_bytes(m.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            m.p90_rss_bytes as f64,
            rss::fmt_bytes(m.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

pub fn run() {
    // Pre-flight: this bench only runs against a real object store (S3 or
    // Azure; see `tiers::supertable_storage_fixture`). Fail fast with a clear
    // message instead of a panic deep inside the first build. Checked in both
    // the parent and any spawned child (env is inherited).
    if let Err(reason) = crate::tiers::supertable_backend_check() {
        eprintln!("[supertable] skipped: {reason}");
        return;
    }

    // Child mode: build exactly one shape in this fresh process, then exit.
    if handle_shape_child_from_env() {
        return;
    }

    // Parent mode: build each shape in its own isolated subprocess so the
    // per-shape RSS numbers are independent (see the module docs).
    let n_docs = supertable::n_docs();
    eprintln!(
        "[supertable] ingesting {} docs ({} commits) per shape to object storage, \
         one isolated process per shape...",
        fmt_count(n_docs),
        supertable::N_COMMIT_CHUNKS
    );

    let shape_results = run_ingest_shapes_isolated();
    let rows: Vec<Vec<Cell>> = shape_results
        .iter()
        .map(|r| ingest_row(n_docs, r.label, &r.metrics))
        .collect();

    if rows.is_empty() {
        eprintln!("[supertable] no shapes produced metrics — not emitting a report");
        return;
    }

    let mut report = Report::load("supertable");
    report.emit(&Section {
        anchor: "bench/supertable/ingest".into(),
        title: format!(
            "Supertable — ingest, multi-segment / object-store ({} docs × dim={}, {} commits)",
            fmt_count(n_docs),
            crate::corpus::DIM,
            supertable::N_COMMIT_CHUNKS
        ),
        note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). \
               Each shape is built in its own subprocess, so Peak/Median/P90 RSS are measured from a \
               clean address space and are comparable across shapes. Rows are the three index shapes \
               built from the same seeded corpus, so each is directly comparable to its single-modality \
               peer. Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the \
               previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Shape".into(),
                "Time".into(),
                "Throughput".into(),
                "Superfiles".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows,
        }],
    });
    report.save();
}

// ─── Per-modality query runners ───────────────────────────────────────────

const HOT_ITERS: usize = 20;
const COLD_ITERS: usize = 5;
const TOP_K: usize = 10;
const VECTOR_NPROBE: usize = 8;
const VECTOR_RERANK_MULT: usize = 20;

/// Selected phases for a per-modality supertable runner.
///
/// Read phases (`hot`, `cold`) still build the object-store table because
/// they need the committed artifact; `build` controls whether the ingest
/// section is emitted.
#[derive(Clone, Copy)]
pub struct Phases {
    pub build: bool,
    pub hot: bool,
    pub cold: bool,
}

impl Phases {
    pub const ALL: Phases = Phases {
        build: true,
        hot: true,
        cold: true,
    };
}

fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

fn open_consumer(
    modality: Modality,
    built: &supertable::IngestResult,
) -> (TempDir, Supertable) {
    let (cache_dir, cache) = tiers::fresh_supertable_search_cache(
        Arc::clone(&built.storage),
        Some(built.total_index_bytes),
    );
    let opts = tiers::consumer_options(
        supertable::options_for(modality, None),
        Arc::clone(&built.storage),
        cache,
    );
    (cache_dir, tiers::open_consumer(opts))
}

fn to_infino_mode(mode: BoolMode) -> InfinoBoolMode {
    match mode {
        BoolMode::Or => InfinoBoolMode::Or,
        BoolMode::And => InfinoBoolMode::And,
    }
}

fn emit_query_rows(
    report: &mut Report,
    anchor: &str,
    title: String,
    note: &str,
    col: &str,
    rows: Vec<(&'static str, Duration)>,
) {
    report.emit(&Section {
        anchor: anchor.into(),
        title,
        note: note.into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec!["Query".into(), col.into()],
            rows: rows
                .into_iter()
                .map(|(name, d)| {
                    let ns = d.as_secs_f64() * 1e9;
                    vec![text(name), metric(ns, fmt_time(ns), Better::Lower)]
                })
                .collect(),
        }],
    });
}

pub mod fts {
    use super::*;

    /// Build an FTS-only supertable, then measure hot and cold BM25 reads.
    ///
    /// This is not wired to a Cargo selector yet; it is the per-modality
    /// runner that the eventual one-binary dispatcher will call.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_fts] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Fts);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();

        let mut report = Report::load("supertable_fts");
        let metrics = ShapeMetrics {
            wall_ns: wall.as_secs_f64() * 1e9,
            n_superfiles: built.n_superfiles,
            peak_rss_bytes: rss.peak_rss_bytes,
            median_rss_bytes: rss.median_rss_bytes,
            p90_rss_bytes: rss.p90_rss_bytes,
        };
        if phases.build {
            report.emit(&Section {
            anchor: "bench/fts/supertable/ingest".into(),
            title: format!(
                "Supertable FTS — ingest, multi-segment / object-store ({} docs, {} commits)",
                fmt_count(n_docs),
                supertable::N_COMMIT_CHUNKS
            ),
            note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: vec![
                    "Shape".into(),
                    "Time".into(),
                    "Throughput".into(),
                    "Superfiles".into(),
                    "Peak RSS".into(),
                    "Median RSS".into(),
                    "P90 RSS".into(),
                ],
                rows: vec![ingest_row(n_docs, "FTS-only", &metrics)],
            }],
            });
        }

        if phases.hot {
            let (cache_dir, consumer) = open_consumer(Modality::Fts, &built);
            let hot = measure_hot(&consumer);
            drop(consumer);
            drop(cache_dir);
            emit_query_rows(
                &mut report,
                "bench/fts/supertable/hot",
                format!(
                    "Supertable FTS — hot search, warm cache / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                "Hot = committed table reopened with a disk cache sized to the index; p50 over repeated BM25 reads. Δ is vs the previous run.",
                "hot",
                hot,
            );
        }

        if phases.cold {
            let cold = measure_cold(&built);
            emit_query_rows(
                &mut report,
                "bench/fts/supertable/cold",
                format!(
                    "Supertable FTS — cold search, fresh cache / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                "Cold = fresh disk cache + fresh consumer per iteration, so each read pays the object-store cold open. Δ is vs the previous run.",
                "cold",
                cold,
            );
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn measure_hot(consumer: &Supertable) -> Vec<(&'static str, Duration)> {
        crate::superfile::fts::FTS_BATTERY
            .iter()
            .map(|q| {
                let query = q.terms.join(" ");
                let mode = to_infino_mode(q.mode);
                let reader = consumer.reader();
                let _ = reader
                    .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                    .expect("warmup bm25");
                let mut samples = Vec::with_capacity(HOT_ITERS);
                for _ in 0..HOT_ITERS {
                    let t = Instant::now();
                    let hits = reader
                        .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                        .expect("hot bm25");
                    std::hint::black_box(hits);
                    samples.push(t.elapsed());
                }
                (q.name, p50(&mut samples))
            })
            .collect()
    }

    fn measure_cold(built: &supertable::IngestResult) -> Vec<(&'static str, Duration)> {
        crate::superfile::fts::FTS_BATTERY
            .iter()
            .map(|q| {
                let query = q.terms.join(" ");
                let mode = to_infino_mode(q.mode);
                let mut samples = Vec::with_capacity(COLD_ITERS);
                for _ in 0..COLD_ITERS {
                    let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
                    let t = Instant::now();
                    let hits = consumer
                        .reader()
                        .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                        .expect("cold bm25");
                    std::hint::black_box(hits);
                    samples.push(t.elapsed());
                    drop(consumer);
                    drop(cache_dir);
                }
                (q.name, p50(&mut samples))
            })
            .collect()
    }
}

pub mod vector {
    use super::*;

    /// Build a vector-only supertable, then measure hot and cold kNN reads.
    ///
    /// This is not wired to a Cargo selector yet; it is the per-modality
    /// runner that the eventual one-binary dispatcher will call.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_vector] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Vector);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();

        let mut report = Report::load("supertable_vector");
        let metrics = ShapeMetrics {
            wall_ns: wall.as_secs_f64() * 1e9,
            n_superfiles: built.n_superfiles,
            peak_rss_bytes: rss.peak_rss_bytes,
            median_rss_bytes: rss.median_rss_bytes,
            p90_rss_bytes: rss.p90_rss_bytes,
        };
        if phases.build {
            report.emit(&Section {
                anchor: "bench/vector/supertable/ingest".into(),
                title: format!(
                    "Supertable vector — ingest, multi-segment / object-store ({} docs × dim={}, {} commits)",
                    fmt_count(n_docs),
                    DIM,
                    supertable::N_COMMIT_CHUNKS
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: vec![
                        "Shape".into(),
                        "Time".into(),
                        "Throughput".into(),
                        "Superfiles".into(),
                        "Peak RSS".into(),
                        "Median RSS".into(),
                        "P90 RSS".into(),
                    ],
                    rows: vec![ingest_row(n_docs, "vector-only", &metrics)],
                }],
            });
        }

        let query = query_vector();
        if phases.hot {
            let (cache_dir, consumer) = open_consumer(Modality::Vector, &built);
            let hot = measure_hot(&consumer, &query);
            drop(consumer);
            drop(cache_dir);
            emit_query_rows(
                &mut report,
                "bench/vector/supertable/hot",
                format!(
                    "Supertable vector — hot search, warm cache / object-store ({} docs × dim={})",
                    fmt_count(n_docs),
                    DIM
                ),
                "Hot = committed table reopened with a disk cache sized to the index; p50 over repeated vector reads. Δ is vs the previous run.",
                "hot",
                hot,
            );
        }

        if phases.cold {
            let cold = measure_cold(&built, &query);
            emit_query_rows(
                &mut report,
                "bench/vector/supertable/cold",
                format!(
                    "Supertable vector — cold search, fresh cache / object-store ({} docs × dim={})",
                    fmt_count(n_docs),
                    DIM
                ),
                "Cold = fresh disk cache + fresh consumer per iteration, so each read pays the object-store cold open. Δ is vs the previous run.",
                "cold",
                cold,
            );
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn query_vector() -> Vec<f32> {
        vec![1.0 / (DIM as f32).sqrt(); DIM]
    }

    fn search_options() -> VectorSearchOptions {
        VectorSearchOptions::new()
            .with_nprobe(VECTOR_NPROBE)
            .with_rerank_mult(VECTOR_RERANK_MULT)
    }

    fn measure_hot(consumer: &Supertable, query: &[f32]) -> Vec<(&'static str, Duration)> {
        let reader = consumer.reader();
        let _ = reader
            .vector_search(supertable::VEC_COLUMN, query, TOP_K, search_options())
            .expect("warmup vector_search");
        let mut samples = Vec::with_capacity(HOT_ITERS);
        for _ in 0..HOT_ITERS {
            let t = Instant::now();
            let hits = reader
                .vector_search(supertable::VEC_COLUMN, query, TOP_K, search_options())
                .expect("hot vector_search");
            std::hint::black_box(hits);
            samples.push(t.elapsed());
        }
        vec![("knn_default", p50(&mut samples))]
    }

    fn measure_cold(
        built: &supertable::IngestResult,
        query: &[f32],
    ) -> Vec<(&'static str, Duration)> {
        let mut samples = Vec::with_capacity(COLD_ITERS);
        for _ in 0..COLD_ITERS {
            let (cache_dir, consumer) = open_consumer(Modality::Vector, built);
            let t = Instant::now();
            let hits = consumer
                .reader()
                .vector_search(supertable::VEC_COLUMN, query, TOP_K, search_options())
                .expect("cold vector_search");
            std::hint::black_box(hits);
            samples.push(t.elapsed());
            drop(consumer);
            drop(cache_dir);
        }
        vec![("knn_default", p50(&mut samples))]
    }
}

pub mod sql {
    use super::*;

    /// Build a SQL-only supertable, then measure hot and cold `query_sql`
    /// reads over the scalar SQL battery.
    ///
    /// This is not wired to a Cargo selector yet; it is the per-modality
    /// runner that the eventual one-binary dispatcher will call.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_sql] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Sql);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();

        let mut report = Report::load("supertable_sql");
        let metrics = ShapeMetrics {
            wall_ns: wall.as_secs_f64() * 1e9,
            n_superfiles: built.n_superfiles,
            peak_rss_bytes: rss.peak_rss_bytes,
            median_rss_bytes: rss.median_rss_bytes,
            p90_rss_bytes: rss.p90_rss_bytes,
        };
        if phases.build {
            report.emit(&Section {
                anchor: "bench/sql/supertable/ingest".into(),
                title: format!(
                    "Supertable SQL — ingest, multi-segment / object-store ({} rows, {} commits)",
                    fmt_count(n_docs),
                    supertable::N_COMMIT_CHUNKS
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Superfiles` is the committed segment count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: vec![
                        "Shape".into(),
                        "Time".into(),
                        "Throughput".into(),
                        "Superfiles".into(),
                        "Peak RSS".into(),
                        "Median RSS".into(),
                        "P90 RSS".into(),
                    ],
                    rows: vec![ingest_row(n_docs, "SQL", &metrics)],
                }],
            });
        }

        if phases.hot {
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            let hot = measure_hot(&consumer);
            drop(consumer);
            drop(cache_dir);
            emit_query_rows(
                &mut report,
                "bench/sql/supertable/hot",
                format!(
                    "Supertable SQL — hot queries, warm cache / object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Hot = committed table reopened with a disk cache sized to the index; p50 over repeated `reader().query_sql(...)` calls. Δ is vs the previous run.",
                "hot",
                hot,
            );
        }

        if phases.cold {
            let cold = measure_cold(&built);
            emit_query_rows(
                &mut report,
                "bench/sql/supertable/cold",
                format!(
                    "Supertable SQL — cold queries, fresh cache / object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Cold = fresh disk cache + fresh consumer per iteration, so each query pays the object-store cold open. Δ is vs the previous run.",
                "cold",
                cold,
            );
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn measure_hot(consumer: &Supertable) -> Vec<(&'static str, Duration)> {
        crate::superfile::sql::SQL_BATTERY
            .iter()
            .map(|q| {
                let reader = consumer.reader();
                let _ = reader.query_sql(q.sql).expect("warmup query_sql");
                let mut samples = Vec::with_capacity(HOT_ITERS);
                for _ in 0..HOT_ITERS {
                    let t = Instant::now();
                    let batches = reader.query_sql(q.sql).expect("hot query_sql");
                    std::hint::black_box(batches);
                    samples.push(t.elapsed());
                }
                (q.name, p50(&mut samples))
            })
            .collect()
    }

    fn measure_cold(built: &supertable::IngestResult) -> Vec<(&'static str, Duration)> {
        crate::superfile::sql::SQL_BATTERY
            .iter()
            .map(|q| {
                let mut samples = Vec::with_capacity(COLD_ITERS);
                for _ in 0..COLD_ITERS {
                    let (cache_dir, consumer) = open_consumer(Modality::Sql, built);
                    let t = Instant::now();
                    let batches = consumer.reader().query_sql(q.sql).expect("cold query_sql");
                    std::hint::black_box(batches);
                    samples.push(t.elapsed());
                    drop(consumer);
                    drop(cache_dir);
                }
                (q.name, p50(&mut samples))
            })
            .collect()
    }
}
