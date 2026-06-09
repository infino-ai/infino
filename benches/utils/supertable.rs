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
use crate::harness::BoolMode;
use infino::supertable::Supertable;
use tempfile::TempDir;

use crate::corpus::DIM;
use crate::ingest::supertable::{self, Modality, modality_label};
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

    eprintln!(
        "[supertable] child process: ingesting {} shape ({} docs)...",
        modality_label(modality),
        fmt_count(supertable::n_docs()),
    );
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = supertable::build_on_storage(modality);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();

    // This child wrote its own unique prefix; delete it before exiting so the
    // real-backend run accrues no ongoing cost (ingest-only bench — the
    // artifact is not reused after the build is measured).
    if let Some(cleanup) = &built.cleanup {
        eprintln!("[supertable] child process: cleaning up object-store prefix...");
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
    eprintln!("[supertable] spawning isolated subprocess for shape {key:?}...");
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

const WARM_ITERS: usize = 20;
const COLD_ITERS: usize = 5;
const TOP_K: usize = 10;
const VECTOR_NPROBE: usize = 8;
const VECTOR_RERANK_MULT: usize = 20;

/// Selected phases for a per-modality supertable runner.
///
/// Read phases (`warm`, `cold`) still build the object-store table because
/// they need the committed artifact; `build` controls whether the ingest
/// section is emitted.
#[derive(Clone, Copy)]
pub struct Phases {
    pub build: bool,
    pub warm: bool,
    pub cold: bool,
}

impl Phases {
    pub const ALL: Phases = Phases {
        build: true,
        warm: true,
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

    /// Build an FTS-only supertable, then measure warm and cold BM25 reads.
    ///
    /// This is not wired to a Cargo selector yet; it is the per-modality
    /// runner that the eventual one-binary dispatcher will call.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_fts] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_fts");

        // Build-only matches main `supertable_all`: one isolated subprocess with
        // a clean RSS sample. Warm/cold need the committed artifact in-process.
        if phases.build && !phases.warm && !phases.cold {
            eprintln!(
                "[supertable_fts] build-only: isolated ingest of {} docs to object storage...",
                fmt_count(n_docs),
            );
            if let Some(metrics) = build_shape_isolated("fts") {
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
                report.save();
            }
            return;
        }

        if phases.build {
            eprintln!(
                "[supertable_fts] ingesting {} docs to object storage...",
                fmt_count(n_docs),
            );
        } else {
            eprintln!(
                "[supertable_fts] building search artifact ({} docs) for warm/cold phases...",
                fmt_count(n_docs),
            );
        }
        let sampler = PeakSampler::start_default();
        let t0 = Instant::now();
        let built = supertable::build_on_storage(Modality::Fts);
        let wall = t0.elapsed();
        let rss = sampler.stop_stats();
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

        if phases.warm {
            eprintln!(
                "[supertable_fts] warm search: {} queries × {WARM_ITERS} timed iters (shared consumer, bm25_search)...",
                crate::superfile::fts::FTS_BATTERY.len(),
            );
            let warm = measure_warm(&built);
            emit_query_rows(
                &mut report,
                "bench/fts/supertable/warm",
                format!(
                    "Supertable FTS — warm search, warm cache / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                "Warm = one shared consumer + disk cache: untimed prewarm bm25_search, wait_until_warm once, then per-query prewarm + p50 over repeated bm25_search (full row materialization). Δ is vs the previous run.",
                "warm",
                warm,
            );
        }

        if phases.cold {
            eprintln!(
                "[supertable_fts] cold search: {} queries × {COLD_ITERS} fresh-cache iters...",
                crate::superfile::fts::FTS_BATTERY.len(),
            );
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
            eprintln!("[supertable_fts] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn measure_warm(built: &supertable::IngestResult) -> Vec<(&'static str, Duration)> {
        eprintln!(
            "[supertable_fts] warm: opening shared consumer, prewarm + wait_until_warm once..."
        );
        let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
        let reader = consumer.reader();
        let first = &crate::superfile::fts::FTS_BATTERY[0];
        let first_query = first.terms.join(" ");
        let first_mode = to_infino_mode(first.mode);
        let _ = reader
            .bm25_search(supertable::TEXT_COLUMN, &first_query, TOP_K, first_mode)
            .expect("warm prewarm bm25_search");
        consumer
            .wait_until_warm(Duration::from_secs(600))
            .expect("supertable warm promotion");
        eprintln!(
            "[supertable_fts] warm: cache hot — timing {} queries × {WARM_ITERS} iters via bm25_search...",
            crate::superfile::fts::FTS_BATTERY.len(),
        );
        let results = crate::superfile::fts::FTS_BATTERY
            .iter()
            .map(|q| {
                eprintln!("[supertable_fts] warm: query {}...", q.name);
                let query = q.terms.join(" ");
                let mode = to_infino_mode(q.mode);
                let _ = reader
                    .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                    .expect("warm prewarm bm25_search");
                let mut samples = Vec::with_capacity(WARM_ITERS);
                for _ in 0..WARM_ITERS {
                    let t = Instant::now();
                    let batches = reader
                        .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                        .expect("warm bm25_search");
                    std::hint::black_box(batches);
                    samples.push(t.elapsed());
                }
                (q.name, p50(&mut samples))
            })
            .collect();
        drop(consumer);
        drop(cache_dir);
        results
    }

    fn measure_cold(built: &supertable::IngestResult) -> Vec<(&'static str, Duration)> {
        crate::superfile::fts::FTS_BATTERY
            .iter()
            .map(|q| {
                eprintln!(
                    "[supertable_fts] cold: query {} — {COLD_ITERS} fresh-cache iters...",
                    q.name,
                );
                let query = q.terms.join(" ");
                let mode = to_infino_mode(q.mode);
                let mut samples = Vec::with_capacity(COLD_ITERS);
                for _ in 0..COLD_ITERS {
                    let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
                    let t = Instant::now();
                    let batches = consumer
                        .reader()
                        .bm25_search(supertable::TEXT_COLUMN, &query, TOP_K, mode)
                        .expect("cold bm25_search");
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

pub mod vector {
    use super::*;
    use crate::corpus::{self, Calibrated, MmapVectorCorpus};

    /// Correctness gate: recall@TOP_K must clear this at the high-recall
    /// probe/rerank config or the bench fails.
    const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;
    const CORRECTNESS_NPROBE: usize = 64;
    const CORRECTNESS_RERANK_MULT: usize = 256;
    const N_CORRECTNESS_QUERIES: usize = 20;
    /// Calibration query battery + p50 repetitions per timed point.
    const N_CALIBRATION_QUERIES: usize = 100;
    const CALIBRATION_P50_ITERS: usize = 7;
    /// Recall targets the calibrated rows report (lowest-p50 point clearing
    /// each), plus the user-facing `default` config.
    const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];
    const DEFAULT_NPROBE: usize = VECTOR_NPROBE;
    const DEFAULT_RERANK_MULT: usize = VECTOR_RERANK_MULT;
    /// (probe, refine) calibration grid — same shape as the superfile
    /// vector runner so the two are directly comparable.
    const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
    const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];
    /// Vector-corpus seed. Must match `ingest::supertable`'s `CORPUS_VEC_SEED`
    /// so the regenerated ground-truth vectors are bit-identical to the rows
    /// that were ingested (asserted by `stream_matches_mmap_vector_corpus`).
    const CORPUS_VEC_SEED: u64 = 1;
    const QUERY_CORRECTNESS_SEED: u64 = 17;
    const QUERY_CALIBRATION_SEED: u64 = 99;
    const QUERY_SIGMA: f32 = 0.05;
    const NS_PER_US: f64 = 1e3;

    /// Build a vector-only supertable, then measure warm + cold kNN search
    /// at calibrated recall targets (and a default config), with a
    /// correctness recall gate — the same measurement the superfile vector
    /// runner produces, over the multi-segment object-store consumer.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_vector] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        if phases.build {
            eprintln!(
                "[supertable_vector] ingesting {} docs × dim={DIM} to object storage...",
                fmt_count(n_docs),
            );
        } else {
            eprintln!(
                "[supertable_vector] building search artifact ({} docs) for warm/cold phases...",
                fmt_count(n_docs),
            );
        }
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

        if phases.warm || phases.cold {
            // Regenerate the ingested vectors (same seed → bit-identical to
            // what was committed) to compute brute-force ground truth.
            eprintln!(
                "[supertable_vector] regenerating {}×{DIM} corpus + brute-force ground truth...",
                fmt_count(n_docs),
            );
            let vectors =
                MmapVectorCorpus::generate(n_docs, corpus::n_cent(n_docs), CORPUS_VEC_SEED, true);
            let vslice = vectors.as_slice();
            let q_correct = corpus::generate_realistic_queries(
                vslice,
                n_docs,
                N_CORRECTNESS_QUERIES,
                QUERY_CORRECTNESS_SEED,
                true,
                QUERY_SIGMA,
            );
            let gt_correct = corpus::ground_truth(vslice, n_docs, &q_correct, TOP_K);
            let q_cal = corpus::generate_realistic_queries(
                vslice,
                n_docs,
                N_CALIBRATION_QUERIES,
                QUERY_CALIBRATION_SEED,
                true,
                QUERY_SIGMA,
            );
            let gt_cal = corpus::ground_truth(vslice, n_docs, &q_cal, TOP_K);

            // One hot consumer drives correctness + warm calibration.
            eprintln!("[supertable_vector] opening warm consumer, prewarm + wait_until_warm...");
            let (cache_dir, consumer) = open_consumer(Modality::Vector, &built);
            let _ = consumer
                .reader()
                .vector_search(supertable::VEC_COLUMN, &q_cal[0], TOP_K, search_opts(DEFAULT_NPROBE, DEFAULT_RERANK_MULT))
                .expect("warm prewarm vector_search");
            consumer
                .wait_until_warm(Duration::from_secs(600))
                .expect("supertable warm promotion");

            // Correctness gate on the hot consumer.
            eprintln!(
                "[supertable_vector] correctness: recall@{TOP_K} on {N_CORRECTNESS_QUERIES} queries (nprobe={CORRECTNESS_NPROBE}, rerank={CORRECTNESS_RERANK_MULT})..."
            );
            let recall = mean_recall(
                &consumer,
                &q_correct,
                &gt_correct,
                CORRECTNESS_NPROBE,
                CORRECTNESS_RERANK_MULT,
            );
            assert!(
                recall >= CORRECTNESS_RECALL_FLOOR,
                "supertable vector recall@{TOP_K} {recall:.3} < floor {CORRECTNESS_RECALL_FLOOR:.2}"
            );
            eprintln!("[supertable_vector] correctness OK: recall@{TOP_K} = {recall:.3}");

            // Calibrate each recall target on the hot consumer (warm p50 is
            // measured here, on the same hot cache).
            let cal: Vec<Option<Calibrated>> = RECALL_TARGETS
                .iter()
                .map(|&target| {
                    eprintln!(
                        "[supertable_vector] calibrating recall@{target:.2}: grid over probes/refines ({N_CALIBRATION_QUERIES} queries)..."
                    );
                    calibrate(&consumer, &q_cal, &gt_cal, target)
                })
                .collect();

            let q0 = &q_cal[0];

            // Warm default-config p50 on the hot consumer.
            let warm_default = phases.warm.then(|| {
                warm_p50_ns(&consumer, q0, DEFAULT_NPROBE, DEFAULT_RERANK_MULT)
            });
            drop(consumer);
            drop(cache_dir);

            // Cold: fresh consumer per iteration per config.
            let cold = |nprobe: usize, rerank: usize| {
                phases.cold.then(|| cold_p50_ns(&built, q0, nprobe, rerank))
            };

            let mut rows: Vec<Vec<Cell>> = Vec::new();
            for (i, &target) in RECALL_TARGETS.iter().enumerate() {
                match cal[i] {
                    Some(c) => {
                        let warm = phases.warm.then_some(c.p50_micros as f64 * NS_PER_US);
                        let cold = cold(c.probe, c.refine);
                        rows.push(recall_row(
                            format!("{target:.2}"),
                            format!("p={}, r={}", c.probe, c.refine),
                            format!("{:.3}", c.recall),
                            warm,
                            cold,
                            phases,
                        ));
                    }
                    None => rows.push(recall_row(
                        format!("{target:.2}"),
                        "—".into(),
                        "—".into(),
                        phases.warm.then_some(f64::NAN),
                        None,
                        phases,
                    )),
                }
            }
            rows.push(recall_row(
                "default".into(),
                format!("p={DEFAULT_NPROBE}, r={DEFAULT_RERANK_MULT}"),
                "—".into(),
                warm_default,
                cold(DEFAULT_NPROBE, DEFAULT_RERANK_MULT),
                phases,
            ));

            emit_recall_table(&mut report, n_docs, rows, phases);
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_vector] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
        VectorSearchOptions::new()
            .with_nprobe(nprobe)
            .with_rerank_mult(rerank_mult)
    }

    /// Global-id top-k hits (segment prefix-offset + local id, carrying
    /// score) for recall comparison against brute-force ground truth.
    fn global_hits(
        consumer: &Supertable,
        query: &[f32],
        nprobe: usize,
        rerank: usize,
    ) -> Vec<(u32, f32)> {
        let hits = consumer
            .reader()
            .vector_hits(supertable::VEC_COLUMN, query, TOP_K, search_opts(nprobe, rerank))
            .expect("vector_hits");
        let reader = consumer.reader();
        let manifest = reader.manifest();
        let mut offsets: Vec<u32> = Vec::with_capacity(manifest.superfiles.len());
        let mut acc: u32 = 0;
        for entry in manifest.superfiles.iter() {
            offsets.push(acc);
            acc = acc.saturating_add(entry.n_docs as u32);
        }
        hits.into_iter()
            .map(|h| {
                let seg_idx = manifest
                    .superfiles
                    .iter()
                    .position(|e| e.uri == h.segment)
                    .expect("superfile in manifest");
                (offsets[seg_idx] + h.local_doc_id, h.score)
            })
            .collect()
    }

    fn mean_recall(
        consumer: &Supertable,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        nprobe: usize,
        rerank: usize,
    ) -> f32 {
        let mut sum = 0f32;
        for (q, t) in queries.iter().zip(truths) {
            let hits = global_hits(consumer, q, nprobe, rerank);
            sum += corpus::recall_at_k(&hits, t);
        }
        sum / queries.len() as f32
    }

    /// Lowest-p50 (probe, refine) point clearing `target_recall`; warm p50
    /// is timed on the (hot) consumer. `None` if no grid point reaches it.
    fn calibrate(
        consumer: &Supertable,
        queries: &[Vec<f32>],
        truths: &[Vec<u32>],
        target_recall: f32,
    ) -> Option<Calibrated> {
        let mut best: Option<Calibrated> = None;
        let mut peak = 0f32;
        for &probe in PROBES {
            for &refine in REFINES {
                let recall = mean_recall(consumer, queries, truths, probe, refine);
                peak = peak.max(recall);
                if recall < target_recall {
                    continue;
                }
                let q0 = &queries[0];
                let p50 = corpus::p50_micros(
                    || {
                        let _ = consumer
                            .reader()
                            .vector_search(supertable::VEC_COLUMN, q0, TOP_K, search_opts(probe, refine))
                            .expect("vector_search");
                    },
                    CALIBRATION_P50_ITERS,
                );
                let cand = Calibrated {
                    probe,
                    refine,
                    recall,
                    p50_micros: p50,
                };
                best = match best {
                    None => Some(cand),
                    Some(b) if cand.p50_micros < b.p50_micros => Some(cand),
                    Some(b) => Some(b),
                };
            }
        }
        if best.is_none() {
            eprintln!(
                "    [supertable_vector] no point hit recall ≥ {target_recall:.2}; peak = {peak:.3}"
            );
        }
        best
    }

    fn warm_p50_ns(consumer: &Supertable, query: &[f32], nprobe: usize, rerank: usize) -> f64 {
        corpus::p50_micros(
            || {
                let _ = consumer
                    .reader()
                    .vector_search(supertable::VEC_COLUMN, query, TOP_K, search_opts(nprobe, rerank))
                    .expect("warm vector_search");
            },
            CALIBRATION_P50_ITERS,
        ) as f64
            * NS_PER_US
    }

    fn cold_p50_ns(
        built: &supertable::IngestResult,
        query: &[f32],
        nprobe: usize,
        rerank: usize,
    ) -> f64 {
        let mut samples = Vec::with_capacity(COLD_ITERS);
        for _ in 0..COLD_ITERS {
            let (cache_dir, consumer) = open_consumer(Modality::Vector, built);
            let t = Instant::now();
            let hits = consumer
                .reader()
                .vector_search(supertable::VEC_COLUMN, query, TOP_K, search_opts(nprobe, rerank))
                .expect("cold vector_search");
            std::hint::black_box(hits);
            samples.push(t.elapsed());
            drop(consumer);
            drop(cache_dir);
        }
        p50(&mut samples).as_secs_f64() * 1e9
    }

    fn time_cell(ns: Option<f64>) -> Cell {
        match ns {
            Some(v) if v.is_finite() => metric(v, fmt_time(v), Better::Lower),
            _ => text("—"),
        }
    }

    fn recall_row(
        target: String,
        params: String,
        recall: String,
        warm_ns: Option<f64>,
        cold_ns: Option<f64>,
        phases: Phases,
    ) -> Vec<Cell> {
        let mut cells = vec![text(&target), text(&params), text(&recall)];
        if phases.warm {
            cells.push(time_cell(warm_ns));
        }
        if phases.cold {
            cells.push(time_cell(cold_ns));
        }
        cells
    }

    fn emit_recall_table(report: &mut Report, n_docs: usize, rows: Vec<Vec<Cell>>, phases: Phases) {
        let mut headers = vec!["Recall target".into(), "(p, r)".into(), "recall".into()];
        if phases.warm {
            headers.push("warm".into());
        }
        if phases.cold {
            headers.push("cold".into());
        }
        report.emit(&Section {
            anchor: "bench/vector/supertable/search".into(),
            title: format!(
                "Supertable vector — search, multi-segment / object-store ({} docs × dim={})",
                fmt_count(n_docs),
                DIM
            ),
            note: "Recall rows use the lowest-p50 calibrated `(p, r)` clearing each target (recall vs brute-force ground truth on the regenerated corpus); `default` is the user-facing config. Warm = hot disk cache sized to the index; cold = fresh disk cache + consumer per iteration. Δ is vs the previous run."
                .into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers,
                rows,
            }],
        });
    }
}

pub mod sql {
    use super::*;

    /// Build a SQL-only supertable, then measure warm and cold `query_sql`
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
        if phases.build {
            eprintln!(
                "[supertable_sql] ingesting {} rows to object storage...",
                fmt_count(n_docs),
            );
        } else {
            eprintln!(
                "[supertable_sql] building query artifact ({} rows) for warm/cold phases...",
                fmt_count(n_docs),
            );
        }
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

        if phases.warm {
            eprintln!(
                "[supertable_sql] warm queries: {} queries × {WARM_ITERS} timed iters...",
                crate::superfile::sql::SQL_BATTERY.len(),
            );
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            // Promote segments to the disk cache (mmap) before timing, so
            // warm reads hit local cached bytes instead of re-fetching from
            // object storage on every query — same warm contract as the
            // FTS/vector runners.
            eprintln!("[supertable_sql] warm: prewarm + wait_until_warm...");
            let _ = consumer
                .reader()
                .query_sql(crate::superfile::sql::SQL_BATTERY[0].sql)
                .expect("warm prewarm query_sql");
            consumer
                .wait_until_warm(Duration::from_secs(600))
                .expect("supertable warm promotion");
            let warm = measure_warm(&consumer);
            drop(consumer);
            drop(cache_dir);
            emit_query_rows(
                &mut report,
                "bench/sql/supertable/warm",
                format!(
                    "Supertable SQL — warm queries, warm cache / object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Warm = committed table reopened with a disk cache sized to the index; p50 over repeated `reader().query_sql(...)` calls. Δ is vs the previous run.",
                "warm",
                warm,
            );
        }

        if phases.cold {
            eprintln!(
                "[supertable_sql] cold queries: {} queries × {COLD_ITERS} fresh-cache iters...",
                crate::superfile::sql::SQL_BATTERY.len(),
            );
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
            eprintln!("[supertable_sql] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn measure_warm(consumer: &Supertable) -> Vec<(&'static str, Duration)> {
        crate::superfile::sql::SQL_BATTERY
            .iter()
            .map(|q| {
                eprintln!(
                    "[supertable_sql] warm: query {} — prewarm + {WARM_ITERS} timed iters...",
                    q.name,
                );
                let reader = consumer.reader();
                let _ = reader.query_sql(q.sql).expect("warmup query_sql");
                let mut samples = Vec::with_capacity(WARM_ITERS);
                for _ in 0..WARM_ITERS {
                    let t = Instant::now();
                    let batches = reader.query_sql(q.sql).expect("warm query_sql");
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
                eprintln!(
                    "[supertable_sql] cold: query {} — {COLD_ITERS} fresh-cache iters...",
                    q.name,
                );
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
