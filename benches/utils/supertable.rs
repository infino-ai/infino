// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Supertable object-store bench (infino-only entry point).
//!
//! Multi-superfile ingest to object storage at the supertable scale
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
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket cargo bench -- supertable
//! INFINO_BENCH_STORE=azure INFINO_REAL_AZURE_CONTAINER=my-container \
//!   AZURE_STORAGE_ACCOUNT_NAME=... AZURE_STORAGE_ACCOUNT_KEY=... cargo bench -- supertable
//! INFINO_BENCH_STORE=s3 INFINO_REAL_S3_BUCKET=my-bucket INFINO_BENCH_SUPERTABLE_DOCS=100000 cargo bench -- supertable
//! ```

use std::process::{Command, Stdio};
use std::sync::Arc;
#[allow(unused_imports)] // `Instant` is consumed by the child mods via `use super::*`
use std::time::{Duration, Instant};

use infino::supertable::Supertable;
use tempfile::TempDir;

use crate::corpus::DIM;
use crate::cost;
use crate::ingest::supertable::{self, Modality, modality_label};
use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss::{self, PeakSampler};
use crate::storage_meter;
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
    /// Index bytes written to object storage during ingest. The
    /// supertable's "upload bandwidth" is this over the wall time —
    /// the bytes-to-object-store rate, the analogue of the superfile
    /// build's input-payload bandwidth.
    pub index_bytes: u64,
    /// Raw input corpus size (text + vector bytes) — the source data
    /// fed to ingest, distinct from `index_bytes` (what's written out).
    pub corpus_bytes: u64,
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
            "{RESULT_PREFIX}wall_ns={} n_superfiles={} peak={} median={} p90={} index_bytes={} corpus_bytes={}",
            self.wall_ns,
            self.n_superfiles,
            self.peak_rss_bytes,
            self.median_rss_bytes,
            self.p90_rss_bytes,
            self.index_bytes,
            self.corpus_bytes,
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
        let mut index_bytes = None;
        let mut corpus_bytes = None;
        for tok in body.split_whitespace() {
            let (k, v) = tok.split_once('=')?;
            match k {
                "wall_ns" => wall_ns = v.parse().ok(),
                "n_superfiles" => n_superfiles = v.parse().ok(),
                "peak" => peak = v.parse().ok(),
                "median" => median = v.parse().ok(),
                "p90" => p90 = v.parse().ok(),
                "index_bytes" => index_bytes = v.parse().ok(),
                "corpus_bytes" => corpus_bytes = v.parse().ok(),
                _ => {}
            }
        }
        Some(ShapeMetrics {
            wall_ns: wall_ns?,
            n_superfiles: n_superfiles?,
            peak_rss_bytes: peak?,
            median_rss_bytes: median?,
            p90_rss_bytes: p90?,
            index_bytes: index_bytes?,
            corpus_bytes: corpus_bytes?,
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
    // Corpus is generated to disk + mmapped BEFORE the sampler so the
    // measured window covers the engine only.
    let corpus = supertable::prepare_corpus(modality);
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = supertable::build_on_storage(modality, &corpus);
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
        index_bytes: built.total_index_bytes,
        corpus_bytes: corpus.byte_size(),
    };
    println!("{}", metrics.to_result_line());
}

/// Spawn one isolated child to build `key` and return its metrics.
/// stderr is inherited so the child's `[tiers]` logs stream live; stdout
/// is captured to read back the single result line.
fn build_shape_isolated(key: &str) -> Option<ShapeMetrics> {
    eprintln!("[supertable] spawning isolated subprocess for shape {key:?}...");
    let exe = std::env::current_exe().expect("current_exe for supertable child");
    let mut cmd = Command::new(exe);
    cmd.env(SHAPE_ENV, key);
    // Forward a CLI-set dataset prefix; the child only inherits the env.
    if let Some(prefix) = crate::dataset::dataset_prefix() {
        cmd.env(crate::dataset::PREFIX_ENV, prefix);
    }
    let output = cmd
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

/// Shared column headers for every supertable ingest table (the
/// combined `run()` table and the per-modality fts/vector/sql tables),
/// so the four call sites can't drift apart. `Stored` is the total
/// on-storage footprint of the committed superfiles — full Parquet
/// (data pages + embedded BM25/vector indexes), not just the index
/// subsections — printed next to the raw `Corpus` it was built from.
pub fn ingest_headers() -> Vec<String> {
    vec![
        "Shape".into(),
        "Time".into(),
        "Throughput".into(),
        "Bandwidth".into(),
        "Corpus".into(),
        "Stored".into(),
        "Superfiles".into(),
        "Peak RSS".into(),
        "Median RSS".into(),
        "P90 RSS".into(),
    ]
}

pub fn ingest_row(n_docs: usize, label: &str, m: &ShapeMetrics) -> Vec<Cell> {
    let secs = m.wall_ns / 1e9;
    let thr = if secs > 0.0 {
        n_docs as f64 / secs
    } else {
        0.0
    };
    // Upload bandwidth: stored bytes written to object storage per
    // second over the ingest wall time.
    let bw = if secs > 0.0 {
        m.index_bytes as f64 / secs
    } else {
        0.0
    };
    // Stored footprint as a fraction of the raw corpus it was built
    // from — the headline compression/expansion ratio per modality.
    let stored_pct = if m.corpus_bytes > 0 {
        100.0 * m.index_bytes as f64 / m.corpus_bytes as f64
    } else {
        0.0
    };
    vec![
        text(label),
        metric(m.wall_ns, fmt_time(m.wall_ns), Better::Lower),
        metric(thr, fmt_throughput(thr), Better::Higher),
        metric(bw, fmt_bandwidth(bw), Better::Higher),
        text(rss::fmt_bytes(m.corpus_bytes)),
        metric(
            m.index_bytes as f64,
            format!("{} ({stored_pct:.0}%)", rss::fmt_bytes(m.index_bytes)),
            Better::Lower,
        ),
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

#[allow(clippy::too_many_arguments)]
fn emit_cost_warm(
    report: &mut Report,
    anchor: &str,
    title: String,
    built: &supertable::IngestResult,
    metrics: Option<&ShapeMetrics>,
    n_docs: usize,
    warm: &[(String, f64)],
    cold: Option<&[cost::ColdQuery]>,
    cold_store: Option<storage_meter::ObjectStoreMeter>,
) {
    if warm.is_empty() && cold.is_none() {
        return;
    }
    let resident = rss::current_anon_rss_bytes().unwrap_or(0);
    let (wall_s, corpus_bytes) = match metrics {
        Some(m) => (m.wall_ns / 1e9, m.corpus_bytes),
        None => (0.0, 0),
    };
    cost::emit(
        report,
        anchor,
        title,
        &cost::CellCost {
            ingest_wall_s: wall_s,
            writers: supertable::n_writers() as u32,
            put_count: cost::supertable_ingest_puts(built.n_superfiles),
            stored_bytes: built.total_index_bytes,
            corpus_bytes,
            n_docs,
            resident_anon_bytes: resident,
            warm,
            cold,
            cold_store,
            storage_months: None,
            cold_open_amortized: true,
        },
    );
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
        "[supertable] ingesting {} docs ({} commits, {} writers) per shape to object storage, \
         one isolated process per shape...",
        fmt_count(n_docs),
        supertable::n_commits(),
        supertable::n_writers()
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
            "Supertable — ingest, multi-superfile / object-store ({} docs × dim={}, {} commits, {} writers)",
            fmt_count(n_docs),
            crate::corpus::DIM,
            supertable::n_commits(),
            supertable::n_writers()
        ),
        note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). \
               Each shape is built in its own subprocess, so Peak/Median/P90 RSS are measured from a \
               clean address space and are comparable across shapes. Rows are the three index shapes \
               built from the same seeded corpus, so each is directly comparable to its single-modality \
               peer. Throughput is rows/s; `Stored` is the total on-storage footprint of the committed \
               superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; \
               `Superfiles` is the committed superfile count. Δ is vs the previous run."
            .into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: ingest_headers(),
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

/// Ingest a prepared corpus, sampling RSS over the build window. Returns the
/// ingest measurements only for the build phase (it emits them).
fn build_measured(
    modality: Modality,
    corpus: &supertable::PreparedCorpus,
    phases: Phases,
) -> (supertable::IngestResult, Option<ShapeMetrics>) {
    let sampler = PeakSampler::start_default();
    let t0 = Instant::now();
    let built = supertable::build_on_storage(modality, corpus);
    let wall = t0.elapsed();
    let rss = sampler.stop_stats();
    let metrics = phases.build.then_some(ShapeMetrics {
        wall_ns: wall.as_secs_f64() * 1e9,
        n_superfiles: built.n_superfiles,
        peak_rss_bytes: rss.peak_rss_bytes,
        median_rss_bytes: rss.median_rss_bytes,
        p90_rss_bytes: rss.p90_rss_bytes,
        index_bytes: built.total_index_bytes,
        corpus_bytes: corpus.byte_size(),
    });
    (built, metrics)
}

/// Obtain the search artifact for modalities that don't need the corpus after
/// build (FTS, SQL): in dataset mode open the pre-uploaded dataset (no corpus,
/// no ingest); otherwise generate the corpus and ingest it. Vector keeps its
/// corpus for recall ground truth and calls [`build_measured`] directly.
fn build_or_open(
    modality: Modality,
    phases: Phases,
) -> (supertable::IngestResult, Option<ShapeMetrics>) {
    // Dataset mode opens the pre-uploaded dataset only for read phases; a
    // build phase is the prepare step, which still ingests (to the fixed
    // prefix).
    if crate::dataset::dataset_mode() && !phases.build {
        return (supertable::open_dataset(modality), None);
    }
    // Corpus to disk + mmap BEFORE the sampler — engine-only window.
    let corpus = supertable::prepare_corpus(modality);
    build_measured(modality, &corpus, phases)
}

fn open_consumer(modality: Modality, built: &supertable::IngestResult) -> (TempDir, Supertable) {
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

pub mod fts {
    use super::*;
    use crate::executors::fts as exec_fts;
    use crate::executors::fts::{FTS_BATTERY, FtsRead};

    /// Build an FTS-only supertable, then measure warm and cold BM25
    /// reads through the shared FTS executor (same code superfile runs).
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_fts] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_fts");

        // Build-only matches main `supertable_all`: one isolated subprocess
        // with a clean RSS sample. Warm/cold need the artifact in-process.
        if phases.build && !phases.warm && !phases.cold {
            eprintln!(
                "[supertable_fts] build-only: isolated ingest of {} docs to object storage...",
                fmt_count(n_docs),
            );
            if let Some(metrics) = build_shape_isolated("fts") {
                emit_ingest(&mut report, n_docs, &metrics);
                report.save();
            }
            return;
        }

        let (built, ingest_metrics) = build_or_open(Modality::Fts, phases);
        if let Some(metrics) = &ingest_metrics {
            emit_ingest(&mut report, n_docs, metrics);
        }

        if phases.warm || phases.cold {
            let (cache_dir, consumer) = open_consumer(Modality::Fts, &built);
            let reader = consumer.reader();
            exec_fts::assert_correct(&reader, supertable::TEXT_COLUMN, n_docs, "supertable_fts");
            drop(consumer);
            drop(cache_dir);
        }

        let warm = phases.warm.then(|| measure_warm(&built));
        let cold = phases.cold.then(|| measure_cold(&built));
        if phases.warm || phases.cold {
            exec_fts::emit_search(
                &mut report,
                "bench/fts/supertable/search",
                format!(
                    "Supertable FTS — search, multi-superfile / object-store ({} docs)",
                    fmt_count(n_docs)
                ),
                "Warm = shared consumer + disk cache (untimed prewarm + wait_until_warm, then per-query \
                 p50 over repeated bm25_search). Cold = fresh disk cache + consumer per iteration, so \
                 each read pays the object-store cold open. Δ is vs the previous run.",
                warm.as_deref(),
                cold.as_ref(),
                None,
            );
        }

        if phases.warm || phases.cold {
            let warm_vec = warm.as_deref().map(cost::warm_from_fts).unwrap_or_default();
            let cold_vec = cold.as_ref().map(cost::cold_from_fts).unwrap_or_default();
            let cold_store = phases.cold.then(|| measure_cold_store(&built)).flatten();
            if !warm_vec.is_empty() || !cold_vec.is_empty() {
                emit_cost_warm(
                    &mut report,
                    "bench/fts/supertable/cost",
                    format!("Supertable FTS — cost model ({} docs)", fmt_count(n_docs)),
                    &built,
                    ingest_metrics.as_ref(),
                    n_docs,
                    &warm_vec,
                    if cold_vec.is_empty() {
                        None
                    } else {
                        Some(&cold_vec)
                    },
                    cold_store,
                );
            }
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_fts] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    fn emit_ingest(report: &mut Report, n_docs: usize, metrics: &ShapeMetrics) {
        report.emit(&Section {
            anchor: "bench/fts/supertable/ingest".into(),
            title: format!(
                "Supertable FTS — ingest, multi-superfile / object-store ({} docs, {} commits, {} writers)",
                fmt_count(n_docs),
                supertable::n_commits(),
                supertable::n_writers()
            ),
            note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
            blocks: vec![Block {
                subtitle: String::new(),
                headers: ingest_headers(),
                rows: vec![ingest_row(n_docs, "FTS-only", metrics)],
            }],
        });
    }

    fn measure_warm(built: &supertable::IngestResult) -> Vec<exec_fts::FtsQueryStat> {
        eprintln!(
            "[supertable_fts] warm: opening shared consumer, prewarm + wait_until_warm once..."
        );
        // Phase-boundary RSS splits (anonymous heap vs mmap'd files):
        // the discriminator for "where do the warm-phase GiBs live" —
        // ingest leftovers show up as anonymous bloat already present
        // before the consumer opens; promotion double-residency shows
        // up as anonymous ≈ file_backed after warm-up.
        crate::rss::log_rss_breakdown("supertable_fts before consumer open");
        let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
        let reader = consumer.reader();
        let first = &FTS_BATTERY[0];
        let first_query = first.terms.join(" ");
        let _ = reader
            .bm25_search(
                supertable::TEXT_COLUMN,
                &first_query,
                TOP_K,
                exec_fts::to_infino_mode(first.mode),
                None,
            )
            .expect("warm prewarm bm25_search");
        consumer
            .wait_until_warm(Duration::from_secs(600))
            .expect("supertable warm promotion");
        crate::rss::log_rss_breakdown("supertable_fts after wait_until_warm");
        eprintln!(
            "[supertable_fts] warm: cache hot — timing {} queries × {WARM_ITERS} iters via bm25_search...",
            FTS_BATTERY.len(),
        );
        let out = exec_fts::measure_warm(
            &reader,
            FTS_BATTERY,
            supertable::TEXT_COLUMN,
            TOP_K,
            WARM_ITERS,
            "supertable_fts",
        );
        crate::rss::log_rss_breakdown("supertable_fts after warm battery");
        drop(consumer);
        drop(cache_dir);
        out
    }

    fn measure_cold(
        built: &supertable::IngestResult,
    ) -> std::collections::HashMap<&'static str, crate::executors::ColdTiming> {
        exec_fts::measure_cold(
            || SupertableColdGuard::open(built),
            FTS_BATTERY,
            supertable::TEXT_COLUMN,
            TOP_K,
            COLD_ITERS,
            "supertable_fts",
        )
    }

    /// One metered cold iteration (`ten_term_or`) on a real object-store backend.
    fn measure_cold_store(
        built: &supertable::IngestResult,
    ) -> Option<storage_meter::ObjectStoreMeter> {
        built.cleanup.as_ref()?;
        let query = FTS_BATTERY.iter().find(|q| q.name == "ten_term_or")?;
        let meter = storage_meter::wrap(std::sync::Arc::clone(&built.storage));
        let (cache_dir, cache) =
            tiers::fresh_supertable_search_cache(meter.provider(), Some(built.total_index_bytes));
        let opts = tiers::consumer_options(
            supertable::options_for(Modality::Fts, None),
            meter.provider(),
            cache,
        );
        let consumer = tiers::open_consumer(opts);
        crate::executors::open_all_superfiles(&consumer);
        let reader = consumer.reader();
        let terms = query.terms.join(" ");
        let mode = exec_fts::to_infino_mode(query.mode);
        let _ = reader
            .bm25_search(supertable::TEXT_COLUMN, &terms, TOP_K, mode, None)
            .expect("metered cold bm25_search");
        drop(consumer);
        drop(cache_dir);
        Some(meter.snapshot())
    }

    /// Cold-tier guard: a fresh disk cache + consumer per open. The
    /// constructor performs the full cold open (consumer + manifest +
    /// every superfile reader), so the timed `bm25_rows` pays only the
    /// cold search work — open and search are reported separately.
    struct SupertableColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
    }

    impl SupertableColdGuard {
        fn open(built: &supertable::IngestResult) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Fts, built);
            crate::executors::open_all_superfiles(&consumer);
            Self {
                _cache_dir: cache_dir,
                consumer,
            }
        }
    }

    impl FtsRead for SupertableColdGuard {
        fn bm25_rows(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: infino::superfile::fts::reader::BoolMode,
        ) -> usize {
            self.consumer
                .reader()
                .bm25_search(column, query, k, mode, None)
                .expect("cold bm25_search")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }

        fn bm25_rows_fetched(
            &self,
            column: &str,
            query: &str,
            k: usize,
            mode: infino::superfile::fts::reader::BoolMode,
        ) -> usize {
            self.consumer
                .reader()
                .bm25_search(column, query, k, mode, Some(&["_id", column, "score"]))
                .expect("cold bm25_search fetched")
                .iter()
                .map(|b| b.num_rows())
                .sum()
        }
    }
}

pub mod vector {
    use super::*;
    use crate::corpus;
    use crate::executors::vector as exec_vec;
    use crate::executors::vector::VectorRead;

    // Correctness gate, recall targets, calibration grid, and p50 iters
    // live in `crate::executors::vector` (shared by both tiers).
    const N_CORRECTNESS_QUERIES: usize = 20;
    const N_CALIBRATION_QUERIES: usize = 100;
    const DEFAULT_NPROBE: usize = VECTOR_NPROBE;
    const DEFAULT_RERANK_MULT: usize = VECTOR_RERANK_MULT;
    const QUERY_CORRECTNESS_SEED: u64 = 17;
    const QUERY_CALIBRATION_SEED: u64 = 99;
    const QUERY_SIGMA: f32 = 0.05;

    /// `INFINO_BENCH_SKIP_CALIBRATION=1` measures only the fixed
    /// `(nprobe, rerank)` config — no correctness gate, no recall-target
    /// grid, no brute-force ground truth. Gives a fast, prod-shaped
    /// cold-only run without the 54-config calibration sweep.
    fn skip_calibration() -> bool {
        std::env::var_os("INFINO_BENCH_SKIP_CALIBRATION").is_some()
    }
    /// Fixed probe count for the `default` row, overridable with
    /// `INFINO_BENCH_VECTOR_NPROBE` (defaults to [`DEFAULT_NPROBE`]).
    fn fixed_nprobe() -> usize {
        std::env::var("INFINO_BENCH_VECTOR_NPROBE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_NPROBE)
    }
    /// Fixed rerank multiplier for the `default` row, overridable with
    /// `INFINO_BENCH_VECTOR_RERANK` (defaults to [`DEFAULT_RERANK_MULT`]).
    fn fixed_rerank_mult() -> usize {
        std::env::var("INFINO_BENCH_VECTOR_RERANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RERANK_MULT)
    }

    /// Build a vector-only supertable, then measure warm + cold kNN search
    /// at calibrated recall targets (and a default config), with a
    /// correctness recall gate — the same measurement the superfile vector
    /// runner produces, over the multi-superfile object-store consumer.
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_vector] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_vector");

        // Existing-prefix mode: read directly against an already-built,
        // retained supertable (`INFINO_BENCH_EXISTING_PREFIX`) — no corpus, no
        // ingest. With no corpus to back recall, calibration + the brute-force
        // gate are forced off below and queries are corpus-free.
        let existing = tiers::block_on(tiers::existing_supertable_storage_fixture());

        // Corpus to disk + mmap (engine-only window), EXCEPT in existing-prefix
        // mode. Kept alive for the search phase: the same vectors back the
        // brute-force ground truth, so dataset mode regenerates it too
        // (skipping only the ingest).
        let corpus = existing
            .is_none()
            .then(|| supertable::prepare_corpus(Modality::Vector));

        let (built, ingest_metrics) = if let Some(fixture) = existing {
            (supertable::open_existing(Modality::Vector, fixture), None)
        } else if crate::dataset::dataset_mode() && !phases.build {
            (supertable::open_dataset(Modality::Vector), None)
        } else {
            build_measured(
                Modality::Vector,
                corpus
                    .as_ref()
                    .expect("non-existing path prepared a corpus"),
                phases,
            )
        };
        if let Some(metrics) = &ingest_metrics {
            report.emit(&Section {
                anchor: "bench/vector/supertable/ingest".into(),
                title: format!(
                    "Supertable vector — ingest, multi-superfile / object-store ({} docs × dim={}, {} commits, {} writers)",
                    fmt_count(n_docs),
                    DIM,
                    supertable::n_commits(),
                    supertable::n_writers()
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: ingest_headers(),
                    rows: vec![ingest_row(n_docs, "vector-only", metrics)],
                }],
            });
        }

        if phases.warm || phases.cold {
            // No corpus (existing-prefix) ⇒ no ground truth possible ⇒ force
            // skip-calibration: this path measures latency + memory only.
            let skip_cal = skip_calibration() || corpus.is_none();
            let nprobe = fixed_nprobe();
            let rerank = fixed_rerank_mult();

            #[allow(clippy::type_complexity)]
            let (q_correct, q_cal, gt_correct, gt_cal): (
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
                Vec<Vec<u32>>,
                Vec<Vec<u32>>,
            ) = if let Some(corpus) = &corpus {
                // The ingested vectors are still mmapped from the prepared
                // corpus — queries and ground truth come from them instead
                // of a regeneration. Skip-calibration needs no ground truth
                // (no recall gate / grid), so the brute-force pass is elided
                // there; otherwise both query batches share ONE streamed
                // oracle pass: the pass is I/O-bound over a corpus several
                // times RAM, so its cost is corpus bytes, not query count.
                let vslice = corpus
                    .vectors()
                    .expect("vector modality prepared a vector corpus")
                    .as_slice();
                let q_correct = corpus::generate_realistic_queries(
                    vslice,
                    n_docs,
                    N_CORRECTNESS_QUERIES,
                    QUERY_CORRECTNESS_SEED,
                    true,
                    QUERY_SIGMA,
                );
                let q_cal = corpus::generate_realistic_queries(
                    vslice,
                    n_docs,
                    N_CALIBRATION_QUERIES,
                    QUERY_CALIBRATION_SEED,
                    true,
                    QUERY_SIGMA,
                );
                let (gt_correct, gt_cal): (Vec<Vec<u32>>, Vec<Vec<u32>>) = if skip_cal {
                    (Vec::new(), Vec::new())
                } else {
                    eprintln!(
                        "[supertable_vector] brute-force ground truth: one streamed pass, {} queries...",
                        q_correct.len() + q_cal.len(),
                    );
                    let all_queries: Vec<Vec<f32>> =
                        q_correct.iter().chain(q_cal.iter()).cloned().collect();
                    let mut gt_all = corpus::ground_truth(vslice, n_docs, &all_queries, TOP_K);
                    let gt_cal = gt_all.split_off(q_correct.len());
                    (gt_all, gt_cal)
                };
                (q_correct, q_cal, gt_correct, gt_cal)
            } else {
                // Existing-prefix: no corpus → corpus-free Gaussian queries
                // and no ground truth. Recall is meaningless here; this path
                // reuses the normal vector search implementation to measure
                // cold/warm latency and RSS against a retained large table.
                eprintln!(
                    "[supertable_vector] existing-prefix: corpus-free queries, calibration + recall gate disabled",
                );
                (
                    corpus::generate_queries(N_CORRECTNESS_QUERIES, QUERY_CORRECTNESS_SEED),
                    corpus::generate_queries(N_CALIBRATION_QUERIES, QUERY_CALIBRATION_SEED),
                    Vec::new(),
                    Vec::new(),
                )
            };
            // Queries + ground truth extracted; free the corpus pages
            // + temp file so the warm/cold samplers measure the engine
            // only.
            drop(corpus);

            // One consumer drives correctness + calibration. Full cache
            // promotion (prewarm + wait_until_warm) only matters for the
            // warm timing rows — a cold-only run skips it (fts/sql gate
            // the same way) so it doesn't pull every superfile into the
            // cache just to throw it away.
            let (cache_dir, consumer) = open_consumer(Modality::Vector, &built);
            if phases.warm {
                eprintln!(
                    "[supertable_vector] opening warm consumer, prewarm + wait_until_warm..."
                );
                let _ = consumer
                    .reader()
                    .vector_search(
                        supertable::VEC_COLUMN,
                        &q_cal[0],
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                    )
                    .expect("warm prewarm vector_search");
                consumer
                    .wait_until_warm(Duration::from_secs(600))
                    .expect("supertable warm promotion");
            }

            let title = format!(
                "Supertable vector — search, multi-superfile / object-store ({} docs × dim={})",
                fmt_count(n_docs),
                DIM
            );
            let recall_rows = exec_vec::run_search(
                &mut report,
                &consumer,
                || SupertableVecColdGuard::open(&built),
                supertable::VEC_COLUMN,
                n_docs,
                TOP_K,
                nprobe,
                rerank,
                &q_correct,
                &gt_correct,
                &q_cal,
                &gt_cal,
                phases.warm,
                phases.cold,
                COLD_ITERS,
                skip_cal,
                "supertable_vector",
                "bench/vector/supertable/search",
                title,
                "Recall rows use the lowest-p50 calibrated (p, r) clearing each target (recall vs brute-force ground truth on the regenerated corpus); `default` is the user-facing config. Warm = hot disk cache sized to the index; cold = fresh disk cache + consumer per iteration. Δ is vs the previous run.",
            );
            if phases.warm {
                emit_cost_warm(
                    &mut report,
                    "bench/vector/supertable/cost",
                    format!(
                        "Supertable vector — cost model ({} docs × dim={})",
                        fmt_count(n_docs),
                        DIM
                    ),
                    &built,
                    ingest_metrics.as_ref(),
                    n_docs,
                    &cost::warm_from_vector(&recall_rows),
                    None,
                    None,
                );
            }
            drop(consumer);
            drop(cache_dir);
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_vector] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    struct SupertableVecColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
    }

    impl SupertableVecColdGuard {
        fn open(built: &supertable::IngestResult) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Vector, built);
            crate::executors::open_all_superfiles(&consumer);
            Self {
                _cache_dir: cache_dir,
                consumer,
            }
        }
    }

    impl VectorRead for SupertableVecColdGuard {
        fn topk_global(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            nprobe: usize,
            rerank: usize,
        ) -> Vec<(u32, f32)> {
            self.consumer.topk_global(column, query, k, nprobe, rerank)
        }
    }
}

pub mod sql {
    use super::*;
    use crate::executors::sql as exec_sql;
    use crate::executors::sql::SqlRead;
    use crate::harness::sample_query_csv;

    /// Build a SQL supertable, then measure warm + cold `query_sql` through
    /// the shared SQL executor (same code + same query shapes as superfile).
    pub fn run(phases: Phases) {
        if let Err(reason) = tiers::supertable_backend_check() {
            eprintln!("[supertable_sql] skipped: {reason}");
            return;
        }

        let n_docs = supertable::n_docs();
        let mut report = Report::load("supertable_sql");
        let (built, ingest_metrics) = build_or_open(Modality::Sql, phases);
        if let Some(metrics) = &ingest_metrics {
            report.emit(&Section {
                anchor: "bench/sql/supertable/ingest".into(),
                title: format!(
                    "Supertable SQL — ingest, multi-superfile / object-store ({} rows, {} commits, {} writers)",
                    fmt_count(n_docs),
                    supertable::n_commits(),
                    supertable::n_writers()
                ),
                note: "Build path: `SupertableWriter::append` + `commit` to object storage (production path). Throughput is rows/s; `Stored` is the total on-storage footprint of the committed superfiles (full Parquet + embedded indexes) and its share of the raw `Corpus`; `Superfiles` is the committed superfile count. Δ is vs the previous run.".into(),
                blocks: vec![Block {
                    subtitle: String::new(),
                    headers: ingest_headers(),
                    rows: vec![ingest_row(n_docs, "SQL", metrics)],
                }],
            });
        }

        let inputs = exec_sql::QueryInputs {
            qv: sample_query_csv(),
            sample_title: built
                .sql_sample_title
                .clone()
                .expect("sql ingest sets sample_title"),
            sample_key: built
                .sql_sample_key
                .clone()
                .expect("sql ingest sets sample_key"),
        };

        if phases.warm || phases.cold {
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            exec_sql::assert_correct(&consumer, n_docs, "supertable_sql");
            drop(consumer);
            drop(cache_dir);
        }

        if phases.warm {
            eprintln!("[supertable_sql] warm: opening consumer, prewarm + wait_until_warm...");
            let (cache_dir, consumer) = open_consumer(Modality::Sql, &built);
            let _ = consumer
                .reader()
                .query_sql("SELECT COUNT(*) AS n FROM supertable")
                .expect("warm prewarm query_sql");
            consumer
                .wait_until_warm(Duration::from_secs(600))
                .expect("supertable warm promotion");
            let sets =
                exec_sql::measure_query_sets(&consumer, &inputs, exec_sql::ITERS, "supertable_sql");
            drop(consumer);
            drop(cache_dir);
            exec_sql::emit_query(
                &mut report,
                "bench/sql/supertable/warm",
                format!(
                    "Supertable SQL — warm queries, warm cache / object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Warm = committed table reopened with a disk cache sized to the index; p50 over repeated `query_sql` calls, all through infino's own path (the DataFusion-only control arms are not run here). Δ is vs the previous run.",
                &sets,
            );
            emit_cost_warm(
                &mut report,
                "bench/sql/supertable/cost",
                format!("Supertable SQL — cost model ({} rows)", fmt_count(n_docs)),
                &built,
                ingest_metrics.as_ref(),
                n_docs,
                &cost::warm_from_sql(&sets),
                None,
                None,
            );
        }

        if phases.cold {
            let cold = exec_sql::measure_cold(
                || SupertableSqlColdGuard::open(&built),
                COLD_ITERS,
                "supertable_sql",
            );
            exec_sql::emit_cold(
                &mut report,
                "bench/sql/supertable/cold",
                format!(
                    "Supertable SQL — cold queries, fresh cache / object-store ({} rows)",
                    fmt_count(n_docs)
                ),
                "Cold = fresh disk cache + consumer per iteration, so each query pays the object-store cold open. Δ is vs the previous run.",
                &cold,
            );
        }

        report.save();

        if let Some(cleanup) = &built.cleanup {
            eprintln!("[supertable_sql] cleaning up object-store prefix...");
            tiers::cleanup_prefix(cleanup);
        }
    }

    /// Cold-tier guard: fresh disk cache + consumer per open; the timed
    /// `query_rows` pays the object-store cold open on the empty cache.
    struct SupertableSqlColdGuard {
        _cache_dir: TempDir,
        consumer: Supertable,
    }
    impl SupertableSqlColdGuard {
        fn open(built: &supertable::IngestResult) -> Self {
            let (cache_dir, consumer) = open_consumer(Modality::Sql, built);
            crate::executors::open_all_superfiles(&consumer);
            Self {
                _cache_dir: cache_dir,
                consumer,
            }
        }
    }
    impl SqlRead for SupertableSqlColdGuard {
        fn query_rows(&self, sql: &str) -> usize {
            self.consumer.query_rows(sql)
        }
        fn query_count(&self, sql: &str) -> i64 {
            self.consumer.query_count(sql)
        }
    }
}
