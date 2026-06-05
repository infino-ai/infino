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
//! `rerank_mult` is fixed internally at `RERANK_MULT = 4`.
//!
//! Runs in the bench-scale lane (release profile) so the 10K-doc
//! brute-force ground truth completes in ~2 s rather than ~3-4 min
//! in debug. Invoked via
//! `cargo bench --features bench-diagnostics --bench scale -- vector_recall`.

use std::collections::HashSet;

use infino::superfile::VectorSearchOptions;
use infino::superfile::reader::SuperfileReader;
use infino::superfile::vector::distance::Metric;
use infino_bench_utils::corpus::{
    brute_force_topk, build_superfile_with_metric, generate_realistic_queries,
    generate_vector_corpus, open_superfile,
};

const N_DOCS: usize = 10_000;
const N_CENT: usize = 64;
const N_QUERIES: usize = 50;

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
    let opts = VectorSearchOptions::new().with_nprobe(nprobe);
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

fn build_fixture(seed: u64, normalize_each: bool, metric: Metric) -> (Vec<f32>, SuperfileReader) {
    let vectors = generate_vector_corpus(N_DOCS, N_CENT, seed, normalize_each);
    let docs: Vec<String> = (0..N_DOCS).map(|i| format!("doc {i}")).collect();
    let bytes = build_superfile_with_metric(&docs, &vectors, N_CENT, metric);
    let reader = open_superfile(bytes);
    (vectors, reader)
}

fn recall_l2sq_at_10k_dim384_meets_threshold() {
    let (vectors, reader) = build_fixture(1, false, Metric::L2Sq);
    let queries = generate_realistic_queries(&vectors, N_DOCS, N_QUERIES, 100, false, 0.05);

    let r10 = measure_recall(&reader, &vectors, Metric::L2Sq, &queries, 10, 8);
    assert!(
        r10 >= 0.90,
        "L2Sq recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &vectors, Metric::L2Sq, &queries, 10, 32);
    assert!(
        r10_high >= 0.95,
        "L2Sq recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    let r1 = measure_recall(&reader, &vectors, Metric::L2Sq, &queries, 1, 8);
    assert!(
        r1 >= 0.95,
        "L2Sq recall@1 at nprobe=8 below threshold: {r1:.3} < 0.95"
    );

    println!(
        "L2Sq @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}; recall@1/nprobe=8 = {r1:.3}"
    );
}

fn recall_cosine_at_10k_dim384_meets_threshold() {
    let (vectors, reader) = build_fixture(2, true, Metric::Cosine);
    let queries = generate_realistic_queries(&vectors, N_DOCS, N_QUERIES, 200, true, 0.05);

    let r10 = measure_recall(&reader, &vectors, Metric::Cosine, &queries, 10, 8);
    assert!(
        r10 >= 0.90,
        "Cosine recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &vectors, Metric::Cosine, &queries, 10, 32);
    assert!(
        r10_high >= 0.95,
        "Cosine recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    println!("Cosine @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}");
}

fn recall_increases_monotonically_with_nprobe() {
    let (vectors, reader) = build_fixture(3, false, Metric::L2Sq);
    let queries = generate_realistic_queries(&vectors, N_DOCS, N_QUERIES, 300, false, 0.05);

    let mut prev: f32 = -1.0;
    for &nprobe in &[1, 2, 4, 8, 16, 32, 64] {
        let r = measure_recall(&reader, &vectors, Metric::L2Sq, &queries, 10, nprobe);
        assert!(
            r >= prev - 0.02,
            "recall regressed with more nprobe: nprobe={nprobe}, recall={r:.3}, prev={prev:.3}"
        );
        prev = r;
    }
}

pub fn run() {
    println!("vector_recall: running 3 pinned-threshold checks (10K × 384)");
    recall_l2sq_at_10k_dim384_meets_threshold();
    recall_cosine_at_10k_dim384_meets_threshold();
    recall_increases_monotonically_with_nprobe();
    println!("vector_recall: all 3 pinned-threshold checks passed");
}
