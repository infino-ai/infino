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
    let (l2_vecs, l2_reader) = build_fixture(1, false, Metric::L2Sq);
    let l2_q = generate_realistic_queries(&l2_vecs, N_DOCS, N_QUERIES, 100, false, 0.05);
    let (l2_r10_np8, rss_a) = sampled_recall(&l2_reader, &l2_vecs, Metric::L2Sq, &l2_q, 10, 8);
    let (l2_r10_np32, rss_b) = sampled_recall(&l2_reader, &l2_vecs, Metric::L2Sq, &l2_q, 10, 32);
    let (l2_r1_np8, rss_c) = sampled_recall(&l2_reader, &l2_vecs, Metric::L2Sq, &l2_q, 1, 8);
    assert!(
        l2_r10_np8 >= 0.90,
        "L2Sq recall@10 nprobe=8 {l2_r10_np8:.3} < 0.90"
    );
    assert!(
        l2_r10_np32 >= 0.95,
        "L2Sq recall@10 nprobe=32 {l2_r10_np32:.3} < 0.95"
    );
    assert!(
        l2_r1_np8 >= 0.95,
        "L2Sq recall@1 nprobe=8 {l2_r1_np8:.3} < 0.95"
    );

    let (cos_vecs, cos_reader) = build_fixture(2, true, Metric::Cosine);
    let cos_q = generate_realistic_queries(&cos_vecs, N_DOCS, N_QUERIES, 200, true, 0.05);
    let (cos_r10_np8, rss_d) =
        sampled_recall(&cos_reader, &cos_vecs, Metric::Cosine, &cos_q, 10, 8);
    let (cos_r10_np32, rss_e) =
        sampled_recall(&cos_reader, &cos_vecs, Metric::Cosine, &cos_q, 10, 32);
    assert!(
        cos_r10_np8 >= 0.90,
        "Cosine recall@10 nprobe=8 {cos_r10_np8:.3} < 0.90"
    );
    assert!(
        cos_r10_np32 >= 0.95,
        "Cosine recall@10 nprobe=32 {cos_r10_np32:.3} < 0.95"
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
    let (vectors, reader) = build_fixture(3, false, Metric::L2Sq);
    let queries = generate_realistic_queries(&vectors, N_DOCS, N_QUERIES, 300, false, 0.05);
    let mut rows = Vec::new();
    let mut prev: f32 = -1.0;
    for &nprobe in &[1, 2, 4, 8, 16, 32, 64] {
        let (r, rss) = sampled_recall(&reader, &vectors, Metric::L2Sq, &queries, 10, nprobe);
        assert!(
            r >= prev - 0.02,
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
