// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Measured vector recall on a realistic-shape 10K × 384 corpus.
//!
//! Recall@k is the fraction of true top-k neighbors (by exact
//! brute-force distance) that our IVF + RaBitQ + rerank pipeline
//! actually returns. The pinned thresholds catch any regression in
//! clustering quality, quantization fidelity, or rerank shortlist
//! sizing.
//!
//! All searches go through [`SuperfileReader::vector_search`] with
//! [`VectorSearchOptions`] — the same production path callers use.
//!
//! Measured at `rerank_mult = BENCH_RERANK_MULT` (16): at the bare
//! `VectorSearchOptions` default of 4 the `k*4 = 40`-candidate shortlist
//! drops true top-10 neighbors (recall@10 ≈ 0.44 here) so only shortlist
//! depth — not clustering or quantization — is being measured; recall
//! saturates by 16 (≈ 0.99), so 16 isolates the quality signal the
//! thresholds gate on.
//!
//! Runs in the bench-scale lane (release profile) so the 10K-doc
//! brute-force ground truth completes in ~2 s rather than ~3-4 min
//! in debug. Results render through the custom report harness (terminal
//! +, when `INFINO_BENCH_UPDATE_README=1`, the `bench/scale/vector_recall`
//! README anchor) with run-to-run deltas.

use std::collections::HashSet;

use infino::superfile::VectorSearchOptions;
use infino::superfile::reader::SuperfileReader;
use infino::superfile::vector::distance::Metric;
use infino_bench_utils::corpus::{
    brute_force_topk, build_superfile_with_metric, generate_realistic_queries,
    generate_vector_corpus, open_superfile,
};
use infino_bench_utils::report::{Better, Block, Cell, Report, Section, metric, text};
use infino_bench_utils::rss::{self, PeakSampler, RssStats};

const N_DOCS: usize = 10_000;
const N_CENT: usize = 64;
const N_QUERIES: usize = 50;

/// Rerank shortlist depth (`k * mult` candidates from the 1-bit RaBitQ
/// pass enter exact/Sq8 rerank). Deep enough that the shortlist holds the
/// true neighbors, so the numbers gate clustering + quantization quality
/// rather than shortlist depth. See the module docs.
const BENCH_RERANK_MULT: usize = 16;
/// Per-dim Gaussian perturbation for "near-doc" realistic queries.
const QUERY_SIGMA: f32 = 0.05;
/// recall@K used by the gate (top-10) and the strict recall@1 check.
const RECALL_AT_K: usize = 10;
const RECALL_AT_ONE_K: usize = 1;
/// Low / high nprobe operating points for the recall gates.
const NPROBE_LOW: usize = 8;
const NPROBE_HIGH: usize = 32;
/// Recall floors (regression thresholds) per metric / operating point.
const RECALL10_NPROBE_LOW_MIN: f32 = 0.90;
const RECALL10_NPROBE_HIGH_MIN: f32 = 0.95;
const RECALL1_NPROBE_LOW_MIN: f32 = 0.95;
/// Corpus/query seeds per metric fixture (distinct so fixtures differ).
const L2SQ_FIXTURE_SEED: u64 = 1;
const L2SQ_QUERY_SEED: u64 = 100;
const COSINE_FIXTURE_SEED: u64 = 2;
const COSINE_QUERY_SEED: u64 = 200;
const MONOTONIC_FIXTURE_SEED: u64 = 3;
const MONOTONIC_QUERY_SEED: u64 = 300;
/// Sentinel "previous recall" so the first nprobe in the monotonic
/// sweep always passes the non-decreasing check.
const MONOTONIC_PREV_SENTINEL: f32 = -1.0;
/// Ordered nprobe ladder for the monotonicity regression.
const NPROBE_MONOTONIC_SWEEP: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
/// Allowed recall drop between adjacent nprobe steps (noise band).
const NPROBE_MONOTONIC_TOLERANCE: f32 = 0.02;

fn search_blocking(
    reader: &SuperfileReader,
    query: &[f32],
    k: usize,
    opts: VectorSearchOptions,
) -> Vec<(u32, f32)> {
    infino_bench_utils::corpus::block_on_inmem(reader.vector_search("emb", query, k, opts))
        .expect("vector_search")
}

fn measure_recall(
    reader: &SuperfileReader,
    vectors: &[f32],
    metric: Metric,
    queries: &[Vec<f32>],
    k: usize,
    nprobe: usize,
) -> f32 {
    let opts = VectorSearchOptions::new()
        .with_nprobe(nprobe)
        .with_rerank_mult(BENCH_RERANK_MULT);
    let mut total: f32 = 0.0;
    for q in queries {
        let truth: HashSet<u32> = brute_force_topk(vectors, N_DOCS, q, metric, k)
            .into_iter()
            .collect();
        let approx: HashSet<u32> = search_blocking(reader, q, k, opts)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let hit_count = truth.intersection(&approx).count();
        total += (hit_count as f32) / (k as f32);
    }
    total / (queries.len() as f32)
}

/// Run `measure_recall` under an RSS sampler, returning the recall plus
/// the peak/median/p90 VmRSS observed during the measurement.
fn sampled_recall(
    reader: &SuperfileReader,
    vectors: &[f32],
    metric: Metric,
    queries: &[Vec<f32>],
    k: usize,
    nprobe: usize,
) -> (f32, RssStats) {
    let sampler = PeakSampler::start_default();
    let r = measure_recall(reader, vectors, metric, queries, k, nprobe);
    (r, sampler.stop_stats())
}

fn build_fixture(seed: u64, normalize_each: bool, metric: Metric) -> (Vec<f32>, SuperfileReader) {
    let vectors = generate_vector_corpus(N_DOCS, N_CENT, seed, normalize_each);
    let docs: Vec<String> = (0..N_DOCS).map(|i| format!("doc {i}")).collect();
    let bytes = build_superfile_with_metric(&docs, &vectors, N_CENT, metric);
    let reader = open_superfile(bytes);
    (vectors, reader)
}

fn rss_cells(stats: RssStats) -> Vec<Cell> {
    vec![
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

fn recall_row(label: &str, recall: f32, rss: RssStats) -> Vec<Cell> {
    let mut cells = vec![
        text(label),
        metric(recall as f64, format!("{recall:.3}"), Better::Higher),
    ];
    cells.extend(rss_cells(rss));
    cells
}

fn recall_headers() -> Vec<String> {
    vec![
        "Config".into(),
        "Recall".into(),
        "Peak RSS".into(),
        "Median RSS".into(),
        "P90 RSS".into(),
    ]
}

/// Pinned recall@k points for L2Sq + Cosine, with the regression floors
/// the bench has always asserted (now measured at [`BENCH_RERANK_MULT`]).
fn pinned_rows() -> Vec<Vec<Cell>> {
    let (l2_vecs, l2_reader) = build_fixture(L2SQ_FIXTURE_SEED, false, Metric::L2Sq);
    let l2_q = generate_realistic_queries(
        &l2_vecs,
        N_DOCS,
        N_QUERIES,
        L2SQ_QUERY_SEED,
        false,
        QUERY_SIGMA,
    );
    let (l2_r10_np8, rss_a) = sampled_recall(
        &l2_reader,
        &l2_vecs,
        Metric::L2Sq,
        &l2_q,
        RECALL_AT_K,
        NPROBE_LOW,
    );
    let (l2_r10_np32, rss_b) = sampled_recall(
        &l2_reader,
        &l2_vecs,
        Metric::L2Sq,
        &l2_q,
        RECALL_AT_K,
        NPROBE_HIGH,
    );
    let (l2_r1_np8, rss_c) = sampled_recall(
        &l2_reader,
        &l2_vecs,
        Metric::L2Sq,
        &l2_q,
        RECALL_AT_ONE_K,
        NPROBE_LOW,
    );
    assert!(
        l2_r10_np8 >= RECALL10_NPROBE_LOW_MIN,
        "L2Sq recall@10 nprobe={NPROBE_LOW} {l2_r10_np8:.3} < {RECALL10_NPROBE_LOW_MIN}"
    );
    assert!(
        l2_r10_np32 >= RECALL10_NPROBE_HIGH_MIN,
        "L2Sq recall@10 nprobe={NPROBE_HIGH} {l2_r10_np32:.3} < {RECALL10_NPROBE_HIGH_MIN}"
    );
    assert!(
        l2_r1_np8 >= RECALL1_NPROBE_LOW_MIN,
        "L2Sq recall@1 nprobe={NPROBE_LOW} {l2_r1_np8:.3} < {RECALL1_NPROBE_LOW_MIN}"
    );

    let (cos_vecs, cos_reader) = build_fixture(COSINE_FIXTURE_SEED, true, Metric::Cosine);
    let cos_q = generate_realistic_queries(
        &cos_vecs,
        N_DOCS,
        N_QUERIES,
        COSINE_QUERY_SEED,
        true,
        QUERY_SIGMA,
    );
    let (cos_r10_np8, rss_d) = sampled_recall(
        &cos_reader,
        &cos_vecs,
        Metric::Cosine,
        &cos_q,
        RECALL_AT_K,
        NPROBE_LOW,
    );
    let (cos_r10_np32, rss_e) = sampled_recall(
        &cos_reader,
        &cos_vecs,
        Metric::Cosine,
        &cos_q,
        RECALL_AT_K,
        NPROBE_HIGH,
    );
    assert!(
        cos_r10_np8 >= RECALL10_NPROBE_LOW_MIN,
        "Cosine recall@10 nprobe={NPROBE_LOW} {cos_r10_np8:.3} < {RECALL10_NPROBE_LOW_MIN}"
    );
    assert!(
        cos_r10_np32 >= RECALL10_NPROBE_HIGH_MIN,
        "Cosine recall@10 nprobe={NPROBE_HIGH} {cos_r10_np32:.3} < {RECALL10_NPROBE_HIGH_MIN}"
    );

    vec![
        recall_row("L2Sq · recall@10 · nprobe=8", l2_r10_np8, rss_a),
        recall_row("L2Sq · recall@10 · nprobe=32", l2_r10_np32, rss_b),
        recall_row("L2Sq · recall@1 · nprobe=8", l2_r1_np8, rss_c),
        recall_row("Cosine · recall@10 · nprobe=8", cos_r10_np8, rss_d),
        recall_row("Cosine · recall@10 · nprobe=32", cos_r10_np32, rss_e),
    ]
}

/// recall@10 vs nprobe sweep (L2Sq), asserting monotonic-within-noise.
fn nprobe_sweep_rows() -> Vec<Vec<Cell>> {
    let (vectors, reader) = build_fixture(MONOTONIC_FIXTURE_SEED, false, Metric::L2Sq);
    let queries = generate_realistic_queries(
        &vectors,
        N_DOCS,
        N_QUERIES,
        MONOTONIC_QUERY_SEED,
        false,
        QUERY_SIGMA,
    );
    let mut rows = Vec::new();
    let mut prev: f32 = MONOTONIC_PREV_SENTINEL;
    for &nprobe in NPROBE_MONOTONIC_SWEEP {
        let (r, rss) = sampled_recall(
            &reader,
            &vectors,
            Metric::L2Sq,
            &queries,
            RECALL_AT_K,
            nprobe,
        );
        assert!(
            r >= prev - NPROBE_MONOTONIC_TOLERANCE,
            "recall regressed with more nprobe: nprobe={nprobe}, recall={r:.3}, prev={prev:.3}"
        );
        prev = r;
        rows.push(recall_row(&format!("nprobe={nprobe}"), r, rss));
    }
    rows
}

pub fn run() {
    eprintln!(
        "[scale] vector_recall: measuring recall@k over {N_DOCS} × 384 (IVF + RaBitQ + rerank, rerank_mult={BENCH_RERANK_MULT})..."
    );
    let pinned = pinned_rows();
    let sweep = nprobe_sweep_rows();

    let mut report = Report::load("scale");
    report.emit(&Section {
        anchor: "bench/scale/vector_recall".into(),
        title: format!(
            "Scale — vector recall ({N_DOCS} × 384, IVF + RaBitQ + rerank, {N_CENT} centroids)"
        ),
        note: "Recall@k is the fraction of the exact brute-force top-k that the approximate IVF \
               pipeline returns, averaged over planted realistic queries, measured at \
               rerank_mult=16. Pinned points assert regression floors (L2Sq r@10 ≥ 0.90 / \
               r@1 ≥ 0.95; Cosine r@10 ≥ 0.90); the sweep asserts recall is monotonic in nprobe \
               within noise. Δ is vs the previous run."
            .into(),
        blocks: vec![
            Block {
                subtitle: "Pinned recall@k".into(),
                headers: recall_headers(),
                rows: pinned,
            },
            Block {
                subtitle: "recall@10 vs nprobe (L2Sq)".into(),
                headers: recall_headers(),
                rows: sweep,
            },
        ],
    });
    report.save();
}
