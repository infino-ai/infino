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
    let (vectors, reader) = build_fixture(L2SQ_FIXTURE_SEED, false, Metric::L2Sq);
    let queries = generate_realistic_queries(
        &vectors,
        N_DOCS,
        N_QUERIES,
        L2SQ_QUERY_SEED,
        false,
        QUERY_SIGMA,
    );

    let r10 = measure_recall(
        &reader,
        &vectors,
        Metric::L2Sq,
        &queries,
        RECALL_AT_K,
        NPROBE_LOW,
    );
    assert!(
        r10 >= RECALL10_NPROBE_LOW_MIN,
        "L2Sq recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(
        &reader,
        &vectors,
        Metric::L2Sq,
        &queries,
        RECALL_AT_K,
        NPROBE_HIGH,
    );
    assert!(
        r10_high >= RECALL10_NPROBE_HIGH_MIN,
        "L2Sq recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    let r1 = measure_recall(
        &reader,
        &vectors,
        Metric::L2Sq,
        &queries,
        RECALL_AT_ONE_K,
        NPROBE_LOW,
    );
    assert!(
        r1 >= RECALL1_NPROBE_LOW_MIN,
        "L2Sq recall@1 at nprobe=8 below threshold: {r1:.3} < 0.95"
    );

    println!(
        "L2Sq @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}; recall@1/nprobe=8 = {r1:.3}"
    );
}

fn recall_cosine_at_10k_dim384_meets_threshold() {
    let (vectors, reader) = build_fixture(COSINE_FIXTURE_SEED, true, Metric::Cosine);
    let queries = generate_realistic_queries(
        &vectors,
        N_DOCS,
        N_QUERIES,
        COSINE_QUERY_SEED,
        true,
        QUERY_SIGMA,
    );

    let r10 = measure_recall(
        &reader,
        &vectors,
        Metric::Cosine,
        &queries,
        RECALL_AT_K,
        NPROBE_LOW,
    );
    assert!(
        r10 >= RECALL10_NPROBE_LOW_MIN,
        "Cosine recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(
        &reader,
        &vectors,
        Metric::Cosine,
        &queries,
        RECALL_AT_K,
        NPROBE_HIGH,
    );
    assert!(
        r10_high >= RECALL10_NPROBE_HIGH_MIN,
        "Cosine recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    println!("Cosine @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}");
}

fn recall_increases_monotonically_with_nprobe() {
    let (vectors, reader) = build_fixture(MONOTONIC_FIXTURE_SEED, false, Metric::L2Sq);
    let queries = generate_realistic_queries(
        &vectors,
        N_DOCS,
        N_QUERIES,
        MONOTONIC_QUERY_SEED,
        false,
        QUERY_SIGMA,
    );

    let mut prev: f32 = MONOTONIC_PREV_SENTINEL;
    for &nprobe in NPROBE_MONOTONIC_SWEEP {
        let r = measure_recall(
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
    }
}

pub fn run() {
    println!("vector_recall: running 3 pinned-threshold checks (10K × 384)");
    recall_l2sq_at_10k_dim384_meets_threshold();
    recall_cosine_at_10k_dim384_meets_threshold();
    recall_increases_monotonically_with_nprobe();
    println!("vector_recall: all 3 pinned-threshold checks passed");
}
