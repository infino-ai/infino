// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable object-store bench (infino-only entry point).
//!
//! Multi-segment ingest to object storage at the supertable scale
//! (`INFINO_BENCH_DOC_COUNT`, default 10M), built through the production
//! `SupertableWriter::append` + `commit` path. Three index shapes are
//! measured for apples-to-apples comparison against single-modality
//! peers: FTS-only, vector-only, and combined FTS + vector. Pick a
//! subset with `INFINO_BENCH_SUPERTABLE_SHAPES` (e.g. `fts,vector`);
//! unset builds all three.
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
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_DOC_COUNT=100000 cargo bench --bench supertable_all
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_SUPERTABLE_SHAPES=vector cargo bench --bench supertable_all
//! ```

use std::process::{Command, Stdio};
use std::time::Instant;

use crate::ingest::supertable::{self, Modality};
use crate::markdown::{fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, PeakSampler};

/// Env var the parent sets to make a child build exactly one shape and
/// print its metrics instead of emitting the report.
const SHAPE_ENV: &str = "INFINO_BENCH_SUPERTABLE_SHAPE";
/// Line prefix a child writes to stdout carrying its measured metrics.
const RESULT_PREFIX: &str = "__SUPERTABLE_SHAPE_RESULT__ ";

/// The three measured shapes: (display label, child-env key, modality).
const SHAPES: [(&str, &str, Modality); 3] = [
    ("FTS-only", "fts", Modality::Fts),
    ("vector-only", "vector", Modality::Vector),
    ("combined FTS + vector", "combined", Modality::Combined),
];

/// Plain measured numbers for one shape, marshalled across the
/// parent/child process boundary as a single `key=value` line.
struct ShapeMetrics {
    wall_ns: f64,
    n_superfiles: usize,
    peak_rss_bytes: u64,
    median_rss_bytes: u64,
    p90_rss_bytes: u64,
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

/// `(label, key)` of the shapes to build, in `SHAPES` order. Reads
/// `INFINO_BENCH_SUPERTABLE_SHAPES` (e.g. `fts,vector`); unset selects
/// all. An unknown key exits — silently running the wrong subset would
/// waste a full run.
fn selected_shapes() -> Vec<(&'static str, &'static str)> {
    let requested = match std::env::var("INFINO_BENCH_SUPERTABLE_SHAPES") {
        Ok(v) if !v.trim().is_empty() => v,
        _ => return SHAPES.iter().map(|(l, k, _)| (*l, *k)).collect(),
    };
    let keys: Vec<&str> = requested
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .collect();
    for key in &keys {
        if !SHAPES.iter().any(|(_, k, _)| k == key) {
            let valid: Vec<&str> = SHAPES.iter().map(|(_, k, _)| *k).collect();
            eprintln!(
                "[supertable] unknown shape {key:?}; valid: {}",
                valid.join(", ")
            );
            std::process::exit(2);
        }
    }
    SHAPES
        .iter()
        .filter(|(_, k, _)| keys.contains(k))
        .map(|(l, k, _)| (*l, *k))
        .collect()
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

fn ingest_row(n_docs: usize, label: &str, m: &ShapeMetrics) -> Vec<Cell> {
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
    if let Ok(key) = std::env::var(SHAPE_ENV) {
        run_child_shape(&key);
        return;
    }

    // Parent mode: build each selected shape in its own isolated
    // subprocess so the per-shape RSS numbers are independent (see the
    // module docs).
    let shapes = selected_shapes();
    let n_docs = supertable::n_docs();
    eprintln!(
        "[supertable] ingesting {} docs ({} commits) per shape to object storage, \
         one isolated process per shape ({} shape(s) selected)...",
        fmt_count(n_docs),
        supertable::N_COMMIT_CHUNKS,
        shapes.len()
    );

    let mut rows: Vec<Vec<Cell>> = Vec::with_capacity(shapes.len());
    for (label, key) in shapes {
        eprintln!("[supertable] === shape {label} (isolated process) ===");
        if let Some(metrics) = build_shape_isolated(key) {
            rows.push(ingest_row(n_docs, label, &metrics));
        }
    }

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
