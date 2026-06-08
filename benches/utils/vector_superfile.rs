//! Infino-only vector bench for the superfile layer:
//!
//!   ingest timing (1M × 384 Gaussian planted clusters, cosine)
//! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
//! + nprobe/rerank sweeps
//! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
//!
//! Every phase uses the production path: [`SuperfileBuilder`] →
//! [`SuperfileReader`] → [`SuperfileReader::vector_search`]. Hot
//! opens the finished `.parquet` in memory; cold commits the same bytes
//! to object storage and reads through [`DiskCacheStore::reader`].
//!
//! Pinned to 1M × 384. Supertable scale (10M × 384, sharded into N
//! superfiles) lives in `benches/vector/supertable.rs`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench superfile_vector -- superfile_vec_build      # ingest only
//! cargo bench --bench superfile_vector -- superfile_vec_search     # search only
//! ```

use std::hint::black_box;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crate::corpus::{self, Calibrated, DIM};
use crate::harness::{
    InfinoVectorEngine, InfinoVectorIndex, VectorEngine, VectorMetric, VectorRunConfig,
    VectorSearch, run_vector_with_index,
};
use crate::markdown::{fmt_bandwidth, fmt_count, fmt_throughput, fmt_time};
use crate::report::{Better, Block, Cell, Report, Section, metric, text};
use crate::rss;
use crate::tiers;
use bytes::Bytes;
use infino::superfile::SuperfileReader;
use infino::superfile::reader::VectorSearchOptions;

// ─── Constants ────────────────────────────────────────────────────────

const TOP_K: usize = 10;
const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;
const CALIBRATION_P50_ITERS: usize = 7;

/// Recall floor for the correctness gate. Any infino regression that
/// drops below this fails the bench.
const CORRECTNESS_RECALL_FLOOR: f32 = 0.80;

/// High-recall config used as the correctness probe.
const CORRECTNESS_NPROBE: usize = 64;
const CORRECTNESS_RERANK_MULT: usize = 256;

/// Default options for the user-facing "what does it cost in
/// production?" baseline reported in the search markdown.
const DEFAULT_NPROBE: usize = 8;
const DEFAULT_RERANK_MULT: usize = 20;

const RECALL_TARGETS: &[f32] = &[0.90, 0.95, 0.99];

/// Nanoseconds per second, for latency markdown.
const NS_PER_SEC: f64 = 1e9;
/// Deterministic rotation seed for the vector corpus fixture.
const CORPUS_ROT_SEED: u64 = 1;

/// (probe, refine) calibration grids. The lowest-p50 point clearing
/// each recall target is what the search table reports.
const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];

const VEC_COLUMN: &str = "v";

fn n_docs() -> usize {
    corpus::superfile_docs()
}

// ─── Fixtures ────────────────────────────────────────────────────────

static VECTORS: OnceLock<corpus::MmapVectorCorpus> = OnceLock::new();
static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
fn vectors() -> &'static [f32] {
    VECTORS
        .get_or_init(|| {
            // Raw corpus fixture only. Build/search still exercise Infino's
            // normal vector builder/reader paths; the mmap avoids pinning the
            // synthetic source corpus as heap RAM.
            let n = n_docs();
            corpus::MmapVectorCorpus::generate(n, corpus::n_cent(n), CORPUS_ROT_SEED, true)
        })
        .as_slice()
}

fn queries_correctness() -> &'static [Vec<f32>] {
    QUERIES_CORRECTNESS.get_or_init(|| {
        corpus::generate_realistic_queries(
            vectors(),
            n_docs(),
            N_CORRECTNESS_QUERIES,
            17,
            true,
            0.05,
        )
    })
}

fn queries_calibration() -> &'static [Vec<f32>] {
    QUERIES_CALIBRATION.get_or_init(|| {
        corpus::generate_realistic_queries(
            vectors(),
            n_docs(),
            N_CALIBRATION_QUERIES,
            99,
            true,
            0.05,
        )
    })
}

fn ground_truth_correctness() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CORRECTNESS
        .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_correctness(), TOP_K))
}

fn ground_truth_calibration() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CALIBRATION
        .get_or_init(|| corpus::ground_truth(vectors(), n_docs(), queries_calibration(), TOP_K))
}

fn search_opts(nprobe: usize, rerank_mult: usize) -> VectorSearchOptions {
    VectorSearchOptions::new()
        .with_nprobe(nprobe)
        .with_rerank_mult(rerank_mult)
}

// ─── Correctness ──────────────────────────────────────────────────────

fn assert_infino_self_consistent(reader: &SuperfileReader) -> f32 {
    let qs = queries_correctness();
    let gt = ground_truth_correctness();
    let opts = search_opts(CORRECTNESS_NPROBE, CORRECTNESS_RERANK_MULT);
    let mut total_recall = 0.0_f32;
    for (q, truth) in qs.iter().zip(gt.iter()) {
        let hits = corpus::block_on_inmem(async {
            reader.vector_search(VEC_COLUMN, q, TOP_K, opts).await
        })
        .expect("vector_search");
        assert_eq!(
            hits.len(),
            TOP_K,
            "infino kNN should fill top-{TOP_K}; got {}",
            hits.len()
        );
        total_recall += corpus::recall_at_k(&hits, truth);
    }
    let mean_recall = total_recall / (qs.len() as f32);
    assert!(
        mean_recall >= CORRECTNESS_RECALL_FLOOR,
        "infino mean recall@{TOP_K} at correctness config \
         (p={CORRECTNESS_NPROBE}, r={CORRECTNESS_RERANK_MULT}) \
         below floor: {mean_recall:.3} < {CORRECTNESS_RECALL_FLOOR:.3}"
    );
    mean_recall
}

// ─── Custom-harness runner ────────────────────────────────────────────

#[derive(Clone, Copy)]
struct Timed {
    p50: Duration,
    rss: rss::RssStats,
}

fn writer_label(writers: usize) -> String {
    if writers == 1 {
        "1 writer".to_string()
    } else {
        format!("{writers} writers")
    }
}

fn p50(samples: &mut [Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort_unstable();
    samples[(samples.len() - 1) / 2]
}

fn local_calibrations(reader: &SuperfileReader) -> [Option<Calibrated>; 3] {
    let qs = queries_calibration();
    let gt = ground_truth_calibration();
    let mut out: [Option<Calibrated>; 3] = [None; 3];
    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        out[i] = corpus::calibrate_superfile(
            reader,
            VEC_COLUMN,
            qs,
            gt,
            target,
            PROBES,
            REFINES,
            CALIBRATION_P50_ITERS,
            TOP_K,
        );
    }
    out
}

fn timed_hot(index: &InfinoVectorIndex, query: &[f32], search: VectorSearch) -> Timed {
    let sampler = rss::PeakSampler::start_default();
    let _ = InfinoVectorEngine::read(index, query, TOP_K, search);
    let mut samples = Vec::with_capacity(CALIBRATION_P50_ITERS);
    for _ in 0..CALIBRATION_P50_ITERS {
        let t0 = Instant::now();
        let hits = InfinoVectorEngine::read(index, query, TOP_K, search);
        samples.push(t0.elapsed());
        black_box(hits);
    }
    let rss = sampler.stop_stats();
    Timed {
        p50: p50(&mut samples),
        rss,
    }
}

fn timed_cold(
    committed: &tiers::SuperfileCommitted,
    query: &[f32],
    search: VectorSearch,
) -> Duration {
    let storage = Arc::clone(&committed.storage);
    let uri = committed.uri;
    let mut samples = Vec::with_capacity(3);
    for _ in 0..3 {
        let (cache_dir, cache) = tiers::fresh_superfile_cache(Arc::clone(&storage));
        let opts = search_opts(search.nprobe, search.rerank_mult);
        let t0 = Instant::now();
        tiers::block_on(async {
            let reader = cache.reader(&uri).await.expect("cold reader");
            let _ = reader
                .vector_search(VEC_COLUMN, query, TOP_K, opts)
                .await
                .expect("cold vector_search");
        });
        samples.push(t0.elapsed());
        drop(cache);
        drop(cache_dir);
    }
    p50(&mut samples)
}

fn build_row(label: &str, n_docs: usize, wall: Duration, stats: rss::RssStats) -> Vec<Cell> {
    let secs = wall.as_secs_f64();
    let ns = secs * NS_PER_SEC;
    let input_bytes = (n_docs * DIM * std::mem::size_of::<f32>()) as f64;
    let thr = n_docs as f64 / secs;
    let bw = input_bytes / secs;
    vec![
        text(label),
        metric(ns, fmt_time(ns), Better::Lower),
        metric(thr, fmt_throughput(thr), Better::Higher),
        metric(bw, fmt_bandwidth(bw), Better::Higher),
        metric(
            stats.peak_rss_bytes as f64,
            rss::fmt_bytes(stats.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.median_rss_bytes as f64,
            rss::fmt_bytes(stats.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            stats.p90_rss_bytes as f64,
            rss::fmt_bytes(stats.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

fn search_row(label: String, params: String, hot: Timed, cold: Duration) -> Vec<Cell> {
    let hot_ns = hot.p50.as_secs_f64() * NS_PER_SEC;
    let cold_ns = cold.as_secs_f64() * NS_PER_SEC;
    vec![
        text(label),
        text(params),
        metric(hot_ns, fmt_time(hot_ns), Better::Lower),
        metric(cold_ns, fmt_time(cold_ns), Better::Lower),
        metric(
            hot.rss.peak_rss_bytes as f64,
            rss::fmt_bytes(hot.rss.peak_rss_bytes),
            Better::Lower,
        ),
        metric(
            hot.rss.median_rss_bytes as f64,
            rss::fmt_bytes(hot.rss.median_rss_bytes),
            Better::Lower,
        ),
        metric(
            hot.rss.p90_rss_bytes as f64,
            rss::fmt_bytes(hot.rss.p90_rss_bytes),
            Better::Lower,
        ),
    ]
}

pub fn run() {
    let n_docs = n_docs();
    eprintln!(
        "[superfile_vec] generating {}×{DIM} vector corpus...",
        fmt_count(n_docs)
    );
    let vectors = vectors();

    let empty_queries: [crate::harness::VectorQuery<'_>; 0] = [];
    let (build_result, index) = run_vector_with_index::<InfinoVectorEngine>(
        VectorRunConfig {
            column: VEC_COLUMN,
            dim: DIM,
            metric: VectorMetric::Cosine,
            k: TOP_K,
            iters: CALIBRATION_P50_ITERS,
            parallel: corpus::parallel_writers(),
        },
        vectors,
        &empty_queries,
    );

    eprintln!("[superfile_vec] correctness: using measured 1-writer artifact...");
    let recall = assert_infino_self_consistent(index.reader());
    eprintln!(
        "[superfile_vec] correctness OK: recall@{TOP_K} = {recall:.3} (≥ {CORRECTNESS_RECALL_FLOOR:.2})"
    );

    let cal = local_calibrations(index.reader());
    eprintln!("[superfile_vec] committing measured 1-writer artifact to object storage...");
    let committed = tiers::block_on(tiers::commit_superfile(&Bytes::copy_from_slice(
        index.bytes(),
    )));
    let q = &queries_calibration()[0];

    let mut build_rows = Vec::new();
    for b in &build_result.builds {
        build_rows.push(build_row(&writer_label(b.writers), n_docs, b.wall, b.rss));
    }

    let mut search_rows = Vec::new();
    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        if let Some(c) = cal[i] {
            let search = VectorSearch {
                nprobe: c.probe,
                rerank_mult: c.refine,
            };
            let hot = timed_hot(&index, q, search);
            let cold = timed_cold(&committed, q, search);
            search_rows.push(search_row(
                format!("{target:.2}"),
                format!("p={}, r={}", c.probe, c.refine),
                hot,
                cold,
            ));
        }
    }
    let default_search = VectorSearch {
        nprobe: DEFAULT_NPROBE,
        rerank_mult: DEFAULT_RERANK_MULT,
    };
    let default_hot = timed_hot(&index, q, default_search);
    let default_cold = timed_cold(&committed, q, default_search);
    search_rows.push(search_row(
        "default".into(),
        format!("p={DEFAULT_NPROBE}, r={DEFAULT_RERANK_MULT}"),
        default_hot,
        default_cold,
    ));

    let mut report = Report::load("superfile_vector");
    report.emit(&Section {
        anchor: "bench/vector/superfile/ingest".into(),
        title: format!(
            "Superfile vector — ingest, single-segment / in-memory ({} docs × dim={DIM})",
            fmt_count(n_docs)
        ),
        note: "Build path: `SuperfileBuilder` → unified `.parquet`, through `VectorEngine`. Rows are by writer count; `1 writer` is the canonical artifact used by correctness/search/cold upload. Δ is vs the previous run.".into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Build".into(),
                "Time".into(),
                "Throughput".into(),
                "Bandwidth".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows: build_rows,
        }],
    });
    report.emit(&Section {
        anchor: "bench/vector/superfile/search".into(),
        title: format!(
            "Superfile vector — search, single-segment / in-memory ({} docs × dim={DIM})",
            fmt_count(n_docs)
        ),
        note: "Correctness, hot search, and cold upload reuse the measured 1-writer artifact. Recall rows use the lowest-p50 calibrated point meeting each target; `default` is the user-facing option baseline. Δ is vs the previous run.".into(),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Recall target".into(),
                "(p, r)".into(),
                "hot".into(),
                "cold".into(),
                "Peak RSS".into(),
                "Median RSS".into(),
                "P90 RSS".into(),
            ],
            rows: search_rows,
        }],
    });
    report.save();
}
