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

#[allow(unused_imports)] // `Instant` is consumed by the child mods via `use super::*`
use std::time::{Duration, Instant};
use std::{
    collections::HashMap,
    process::{Command, Stdio},
    sync::Arc,
};

use infino::supertable::Supertable;
use tempfile::TempDir;

use crate::{
    corpus::DIM,
    cost,
    ingest::supertable::{self, Modality, modality_label},
    markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::{self, PeakSampler},
    storage_meter, tiers,
};

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
                        exec_vec::search_opts(nprobe, rerank),
                        None,
                        None,                    )
                    .expect("warm prewarm vector_hits");
                consumer
                    .wait_until_warm(Duration::from_secs(600))
                    .expect("supertable warm promotion");
                if let Some((total, max_per_cell)) = consumer.hidden_vector_superfile_stats() {
                    eprintln!(
                        "[supertable_vector] hidden vector index at warm open: {total} superfiles, max {max_per_cell} per cell"
                    );
                }
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
                default_opts.nprobe,
                default_opts.rerank_mult(),
                &q_correct,
                &gt_correct,
                &q_cal,
                &gt_cal,
                phases.warm,
                phases.cold,
                COLD_ITERS,
                skip_cal,
                if phases.warm && !skip_cal {
                    Some(&consumer)
                } else {
                    None
                },
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
            // Filtered vector recall + latency mirrors the superfile tier:
            // same every-Nth-row allow-set, same brute-force filtered ground
            // truth, same default config.
            if phases.warm
                && let Some(filtered_gt) = filtered_gt.as_ref()
            {
                let mut allow = RoaringBitmap::new();
                for i in (0..n_docs as u32).step_by(FILTER_KEEP_EVERY) {
                    allow.insert(i);
                }
                let allow = Arc::new(allow);

                let consumer_reader = consumer.reader();
                let manifest = consumer_reader.manifest();
                let mut offsets: HashMap<_, u32> = HashMap::new();
                let mut base = 0u32;
                for entry in manifest.superfiles.iter() {
                    offsets.insert(entry.uri, base);
                    base = base.saturating_add(entry.n_docs as u32);
                }
                let mut recalls = Vec::new();
                let mut latencies = Vec::new();
                for (q, gt) in q_correct.iter().zip(filtered_gt) {
                    let t0 = Instant::now();
                    let hits = tiers::block_on(consumer_reader.vector_hits_global_allow_async(
                        supertable::VEC_COLUMN,
                        q,
                        TOP_K,
                        exec_vec::search_opts(nprobe, rerank),
                        Arc::clone(&allow),
                    ))
                    .expect("filtered recall query");
                    latencies.push(t0.elapsed());
                    let global_hits: Vec<(u32, f32)> = hits
                        .iter()
                        .map(|h| {
                            let base = offsets.get(&h.superfile).unwrap_or_else(|| {
                                panic!("missing manifest offset for superfile {:?}", h.superfile,)
                            });
                            (base.saturating_add(h.local_doc_id), h.score)
                        })
                        .collect();
                    recalls.push(corpus::recall_at_k(&global_hits, gt));
                }
                if recalls.is_empty() || latencies.is_empty() {
                    eprintln!(
                        "[supertable_vector] filtered recall skipped: no correctness queries"
                    );
                } else {
                    let mean_recall: f32 = recalls.iter().sum::<f32>() / recalls.len() as f32;
                    latencies.sort_unstable();
                    let p50_ns = latencies[latencies.len() / 2].as_secs_f64() * 1e9;
                    let selectivity = 1.0 / FILTER_KEEP_EVERY as f64;
                    let effective_rerank = rerank.saturating_mul(FILTER_KEEP_EVERY);

                    eprintln!(
                        "[supertable_vector] filtered recall@{TOP_K} ({} queries, ~10% selectivity): {mean_recall:.3}, p50={:.2}ms",
                        q_correct.len(),
                        p50_ns / 1e6,
                    );

                    report.emit(&Section {
                        anchor: "bench/vector/supertable/filtered".into(),
                        title: format!(
                            "Supertable vector — filtered search ({} docs × dim={})",
                            fmt_count(n_docs),
                            DIM
                        ),
                        note: format!(
                            "Filtered kNN (~10% selectivity, every {}th row). recall@{TOP_K} = {mean_recall:.3}. Δ is vs the previous run.",
                            FILTER_KEEP_EVERY
                        ),
                        blocks: vec![Block {
                            subtitle: String::new(),
                            headers: vec![
                                "Filter".into(),
                                "(p, r)".into(),
                                "effective (p, r)".into(),
                                "selectivity".into(),
                                "recall@10".into(),
                                "p50".into(),
                            ],
                            rows: vec![vec![
                                text("filtered (~10%)"),
                                text(format!("p={nprobe}, r={rerank}")),
                                text(format!("p={nprobe}, r={effective_rerank}")),
                                text(format!("{:.1}%", selectivity * 100.0)),
                                text(format!("{mean_recall:.3}")),
                                metric(p50_ns, fmt_time(p50_ns), Better::Lower),
                            ]],
                        }],
                    });
                }
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
    use crate::{
        executors::{sql as exec_sql, sql::SqlRead},
        harness::sample_query_csv,
    };

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
