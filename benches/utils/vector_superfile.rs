//! Infino-only vector bench for the superfile layer:
//!
//!   ingest timing (1M × 384 Gaussian planted clusters, cosine)
//! + calibrated kNN search at recall targets {0.90, 0.95, 0.99}
//! + nprobe/rerank sweeps
//! + correctness gate (`recall@10 ≥ 0.80` at high-recall config)
//!
//! Pinned to 1M × 384. Supertable scale (10M × 384, sharded into N
//! superfiles) lives in `benches/vector/supertable.rs`.
//!
//! ## Invocation
//!
//! ```text
//! cargo bench --bench vector -- superfile_vec_build      # ingest only
//! cargo bench --bench vector -- superfile_vec_search     # search only
//! ```

use std::hint::black_box;
use std::sync::OnceLock;

use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
// Calibrated `(probe, refine)` is intentionally not in the criterion ID:
// criterion's improved/regressed/noise panel only fires on exact-ID matches
// against the previous run's baseline, and `(probe, refine)` drifts as
// calibration re-picks the lowest-p50 point. The chosen point is reported
// in the markdown table instead.
use crate::corpus::{self, Calibrated, DIM};
use crate::{markdown, rss};
use infino::superfile::vector::distance::Metric;
use infino::superfile::vector::reader::VectorReader;

// ─── Constants ────────────────────────────────────────────────────────

const N_DOCS: usize = 1_000_000;
const TOP_K: usize = 10;
const N_CORRECTNESS_QUERIES: usize = 20;
const N_CALIBRATION_QUERIES: usize = 100;

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

/// (probe, refine) calibration grids. The lowest-p50 point clearing
/// each recall target is what the search table reports.
const PROBES: &[usize] = &[1, 5, 10, 25, 50, 100, 200, 400, 800];
const REFINES: &[usize] = &[1, 4, 16, 64, 256, 1024];

// ─── Fixtures ────────────────────────────────────────────────────────

static VECTORS: OnceLock<corpus::MmapCorpus> = OnceLock::new();
static QUERIES_CORRECTNESS: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static QUERIES_CALIBRATION: OnceLock<Vec<Vec<f32>>> = OnceLock::new();
static GROUND_TRUTH_CORRECTNESS: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static GROUND_TRUTH_CALIBRATION: OnceLock<Vec<Vec<u32>>> = OnceLock::new();
static INFINO_BLOB: OnceLock<Vec<u8>> = OnceLock::new();
static CALIBRATIONS: OnceLock<Calibrations> = OnceLock::new();

fn vectors() -> &'static [f32] {
    VECTORS
        .get_or_init(|| {
            // Raw corpus fixture only. Build/search still exercise Infino's
            // normal vector builder/reader paths; the mmap avoids pinning the
            // synthetic source corpus as heap RAM.
            corpus::MmapCorpus::generate(N_DOCS, corpus::n_cent(N_DOCS), 1, true)
        })
        .as_slice()
}

fn queries_correctness() -> &'static [Vec<f32>] {
    QUERIES_CORRECTNESS.get_or_init(|| {
        corpus::generate_realistic_queries(vectors(), N_DOCS, N_CORRECTNESS_QUERIES, 17, true, 0.05)
    })
}

fn queries_calibration() -> &'static [Vec<f32>] {
    QUERIES_CALIBRATION.get_or_init(|| {
        corpus::generate_realistic_queries(vectors(), N_DOCS, N_CALIBRATION_QUERIES, 99, true, 0.05)
    })
}

fn ground_truth_correctness() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CORRECTNESS
        .get_or_init(|| corpus::ground_truth(vectors(), N_DOCS, queries_correctness(), TOP_K))
}

fn ground_truth_calibration() -> &'static [Vec<u32>] {
    GROUND_TRUTH_CALIBRATION
        .get_or_init(|| corpus::ground_truth(vectors(), N_DOCS, queries_calibration(), TOP_K))
}

fn infino_reader() -> VectorReader {
    let blob = INFINO_BLOB.get_or_init(|| build_infino_blob(vectors()));
    open_infino_reader(blob.clone())
}

// ─── Builder ──────────────────────────────────────────────────────────

fn build_infino_blob(vectors: &[f32]) -> Vec<u8> {
    let n_cent = corpus::n_cent(N_DOCS);
    let builder = corpus::build_vector_index(vectors, N_DOCS, n_cent, Metric::Cosine);
    builder.finish()
}

fn open_infino_reader(blob: Vec<u8>) -> VectorReader {
    let n_cent = corpus::n_cent(N_DOCS);
    let json =
        format!(r#"[{{"name":"v","dim":{DIM},"n_cent":{n_cent},"rot_seed":7,"metric":"cosine"}}]"#);
    VectorReader::open(Bytes::from(blob), &json).expect("open VectorReader")
}

// ─── Correctness ──────────────────────────────────────────────────────

fn assert_infino_self_consistent(reader: &VectorReader) -> f32 {
    let qs = queries_correctness();
    let gt = ground_truth_correctness();
    let mut total_recall = 0.0_f32;
    for (q, truth) in qs.iter().zip(gt.iter()) {
        let hits = reader
            .search("v", q, TOP_K, CORRECTNESS_NPROBE, CORRECTNESS_RERANK_MULT)
            .expect("vector search");
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

// ─── Calibration ──────────────────────────────────────────────────────

struct Calibrations {
    infino: [Option<Calibrated>; 3],
}

fn calibrations() -> &'static Calibrations {
    CALIBRATIONS.get_or_init(|| {
        let reader = infino_reader();
        let qs = queries_calibration();
        let gt = ground_truth_calibration();

        eprintln!(
            "[superfile_vec_search] calibrating infino at recall targets {RECALL_TARGETS:?}..."
        );
        let mut inf: [Option<Calibrated>; 3] = [None; 3];
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            inf[i] = corpus::calibrate_infino(&reader, qs, gt, target, PROBES, REFINES, 21, TOP_K);
            eprintln!("  recall ≥ {target:.2} | infino: {:?}", inf[i]);
        }
        Calibrations { infino: inf }
    })
}

// ─── Bench entry ──────────────────────────────────────────────────────

fn bench(c: &mut Criterion) {
    // ---- Correctness phase (runs regardless of criterion filter) ---
    eprintln!("[superfile_vec] correctness: building infino ({N_DOCS} docs)...");
    let reader = infino_reader();
    let recall = assert_infino_self_consistent(&reader);
    eprintln!(
        "[superfile_vec] correctness OK: infino recall@{TOP_K} = {recall:.3} (≥ {:.2})",
        CORRECTNESS_RECALL_FLOOR
    );

    artifact_report(N_DOCS, corpus::n_cent(N_DOCS), vectors());

    // ---- Ingest sub-bench (group: superfile_vec_build) -------------
    {
        let v = vectors();
        let mut g = c.benchmark_group("superfile_vec_build");
        g.sample_size(10);
        g.throughput(Throughput::Elements(N_DOCS as u64));

        // Peak-RSS sampler — bounds the build closure so the recorded
        // RSS reflects this group, not earlier setup or later groups.
        let rss_sample = rss::PeakSampler::start_default();
        let bench_id = format!("infino_build_{N_DOCS}docs");
        g.bench_function(bench_id.clone(), |b| {
            b.iter_with_large_drop(|| build_infino_blob(black_box(v)));
        });
        g.finish();
        let peak = rss_sample.stop();
        let _ = rss::write_peak_rss(group_name::SUPERFILE_VEC_BUILD, &bench_id, peak);

        emit_ingest_markdown();
    }

    // ---- Search sub-bench (group: superfile_vec_search) ------------
    {
        let cal = calibrations();
        let qs = queries_calibration();

        let mut g = c.benchmark_group("superfile_vec_search");
        g.sample_size(10);
        let rss_sample = rss::PeakSampler::start_default();

        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            if let Some(c_inf) = cal.infino[i] {
                // Stable bench id: the calibrated (probe, refine) lives in
                // the markdown table, not the criterion id, so criterion can
                // match this row against its prior baseline and print its
                // own improved/regressed delta on subsequent runs.
                let (p, r) = (c_inf.probe, c_inf.refine);
                g.bench_function(format!("infino_{label}"), |b| {
                    let q = &qs[0];
                    b.iter(|| {
                        let hits = reader.search("v", black_box(q), TOP_K, p, r).expect("kNN");
                        black_box(hits)
                    });
                });
            }
        }

        // Default-options baseline (what users get with no tuning).
        let q = &qs[0];
        g.bench_function("infino_default_options_top10", |b| {
            b.iter(|| {
                let hits = reader
                    .search(
                        black_box("v"),
                        black_box(q),
                        TOP_K,
                        DEFAULT_NPROBE,
                        DEFAULT_RERANK_MULT,
                    )
                    .expect("kNN");
                black_box(hits)
            });
        });

        // nprobe sweep (rerank fixed at default)
        let n_cent = corpus::n_cent(N_DOCS);
        for &nprobe in &[1, 4, 8, 16, 32, 64, 128] {
            if nprobe > n_cent {
                continue;
            }
            g.bench_with_input(
                BenchmarkId::new("infino_nprobe_sweep_rerank20", nprobe),
                &nprobe,
                |b, &np| {
                    b.iter(|| {
                        let hits = reader
                            .search("v", black_box(q), TOP_K, np, DEFAULT_RERANK_MULT)
                            .expect("kNN");
                        black_box(hits)
                    });
                },
            );
        }

        // rerank_mult sweep (nprobe fixed at default)
        for &rerank in &[1, 5, 10, 20, 50, 100] {
            g.bench_with_input(
                BenchmarkId::new("infino_rerank_sweep_nprobe8", rerank),
                &rerank,
                |b, &rm| {
                    b.iter(|| {
                        let hits = reader
                            .search("v", black_box(q), TOP_K, DEFAULT_NPROBE, rm)
                            .expect("kNN");
                        black_box(hits)
                    });
                },
            );
        }

        g.finish();
        let peak = rss_sample.stop();
        // Single sampler covers every (probe, refine) point in this
        // group; record the same peak against each criterion id so the
        // markdown lookup matches the bench id verbatim.
        for (i, &target) in RECALL_TARGETS.iter().enumerate() {
            let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
            if cal.infino[i].is_some() {
                let bid = format!("infino_{label}");
                let _ = rss::write_peak_rss(group_name::SUPERFILE_VEC_SEARCH, &bid, peak);
            }
        }
        let _ = rss::write_peak_rss(
            group_name::SUPERFILE_VEC_SEARCH,
            "infino_default_options_top10",
            peak,
        );

        emit_search_markdown();
    }
}

// ─── Markdown summary emitters ────────────────────────────────────────

mod group_name {
    pub const SUPERFILE_VEC_BUILD: &str = "superfile_vec_build";
    pub const SUPERFILE_VEC_SEARCH: &str = "superfile_vec_search";
}

fn emit_ingest_markdown() {
    use markdown::{MarkdownSection, fmt_throughput, fmt_time, read_mean_ns};

    let group = group_name::SUPERFILE_VEC_BUILD;
    let bench = format!("infino_build_{N_DOCS}docs");
    let ns = read_mean_ns(group, &bench);
    let peak_rss = rss::read_peak_rss_bytes(group, &bench);

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile vector — ingest ({N_DOCS} docs × dim={DIM}, Gaussian planted clusters, cosine)\n\n"
    ));
    body.push_str("| Engine | Time | Throughput | Peak RSS | Peak RSS Δ |\n");
    body.push_str("|--------|------|------------|----------|------------|\n");
    let time = ns.map(fmt_time).unwrap_or_else(|| "—".into());
    let thrpt = ns
        .map(|n| fmt_throughput((N_DOCS as f64) / (n / 1e9)))
        .unwrap_or_else(|| "—".into());
    let rss_cell = peak_rss.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
    let rss_delta = rss::fmt_peak_rss_delta(group, &bench);
    body.push_str(&format!(
        "| infino | {time} | {thrpt} | {rss_cell} | {rss_delta} |\n"
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/superfile/ingest".into(),
        body,
    });
}

fn emit_search_markdown() {
    use markdown::{MarkdownSection, fmt_time, read_mean_ns};

    let group = group_name::SUPERFILE_VEC_SEARCH;
    let cal = calibrations();

    let mut body = String::new();
    body.push_str(&format!(
        "### Superfile vector — search ({N_DOCS} docs × dim={DIM}, calibrated at recall targets)\n\n"
    ));
    body.push_str(
        "| Recall target | infino (probe, refine) | infino p50 | Peak RSS | Peak RSS Δ |\n",
    );
    body.push_str(
        "|---------------|------------------------|------------|----------|------------|\n",
    );

    for (i, &target) in RECALL_TARGETS.iter().enumerate() {
        let label = format!("recall_at_least_{:02}", (target * 100.0) as u32);
        let row_target = format!("{target:.2}");
        if let Some(c_inf) = cal.infino[i] {
            let id = format!("infino_{label}");
            let ns = read_mean_ns(group, &id);
            let peak = rss::read_peak_rss_bytes(group, &id);
            let p50 = ns.map(fmt_time).unwrap_or_else(|| "—".into());
            let rss_cell = peak.map(rss::fmt_bytes).unwrap_or_else(|| "—".into());
            let rss_delta = rss::fmt_peak_rss_delta(group, &id);
            body.push_str(&format!(
                "| {row_target:13} | (p={}, r={}) | {p50:10} | {rss_cell} | {rss_delta} |\n",
                c_inf.probe, c_inf.refine
            ));
        } else {
            body.push_str(&format!("| {row_target:13} | — | — | — | — |\n"));
        }
    }

    body.push('\n');
    body.push_str(
    "**infino default options** (`nprobe=8, rerank_mult=20` — user-facing latency baseline):\n\n",
  );
    body.push_str("| Metric | Value |\n");
    body.push_str("|--------|-------|\n");
    let def = read_mean_ns(group, "infino_default_options_top10");
    let def_s = def.map(fmt_time).unwrap_or_else(|| "—".into());
    let def_rss = rss::read_peak_rss_bytes(group, "infino_default_options_top10")
        .map(rss::fmt_bytes)
        .unwrap_or_else(|| "—".into());
    body.push_str(&format!("| infino_default_options_top10 | {def_s} |\n"));
    body.push_str(&format!(
        "| infino_default_options_top10_peak_rss | {def_rss} |\n"
    ));

    markdown::emit(&MarkdownSection {
        anchor_id: "bench/vector/superfile/search".into(),
        body,
    });
}

// ─── Artifact size + first-query report ──────────────────────────────

fn artifact_report(n: usize, n_cent: usize, vectors: &[f32]) {
    // Build once and time the cold open + first query so the
    // user-visible "first-query latency" number isn't hidden inside
    // criterion's warm-up loop.
    use std::time::Instant;

    let t0 = Instant::now();
    let blob = build_infino_blob(vectors);
    let build_elapsed = t0.elapsed();

    let size_mib = blob.len() as f64 / (1024.0 * 1024.0);

    let t0 = Instant::now();
    let reader = open_infino_reader(blob);
    let open_elapsed = t0.elapsed();

    let q = &queries_calibration()[0];
    let t0 = Instant::now();
    let _ = reader
        .search("v", q, TOP_K, DEFAULT_NPROBE, DEFAULT_RERANK_MULT)
        .expect("kNN");
    let first_q_elapsed = t0.elapsed();

    eprintln!(
        "\n--- artifact-size + cold-load report ({n} docs, {n_cent} clusters, dim={DIM}) ---"
    );
    eprintln!(
        "infino:  build {:>7.2}s  size {size_mib:>6.2} MiB  open {:>6.2} ms  first-query {:>5.2} ms",
        build_elapsed.as_secs_f64(),
        open_elapsed.as_secs_f64() * 1e3,
        first_q_elapsed.as_secs_f64() * 1e3,
    );
}

criterion_group!(benches, bench);
