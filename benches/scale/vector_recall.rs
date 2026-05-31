//! Measured vector recall on a realistic-shape 10K × 384 corpus.
//!
//! Recall@k is the fraction of true top-k neighbors (by exact
//! brute-force distance) that our IVF + RaBitQ + rerank pipeline
//! actually returns. The pinned thresholds catch any regression in
//! clustering quality, quantization fidelity, or rerank shortlist
//! sizing.
//!
//! Runs in the bench-scale lane (release profile) so the 10K-doc
//! brute-force ground truth completes in ~2 s rather than ~3-4 min
//! in debug. Invoked via `cargo bench --bench scale -- vector_recall`.
//!
//! ## Sizing
//!
//! - n = 10,000 docs (large enough that recall isn't trivial; small
//!   enough that brute-force ground truth runs in <1s and the build
//!   completes in <2s at default test profile).
//! - dim = 384 (matches modern sentence-embedding models —
//!   all-MiniLM-L6-v2 is 384, BGE-small is 384, OpenAI ada-002 is
//!   1536 but 384 is the realistic-tested baseline per the project's
//!   benchmark policy).
//! - n_cent = 64 (~sqrt(n), the conventional IVF setting).
//!
//! ## Thresholds
//!
//! Pinned at conservative levels that the current implementation
//! comfortably exceeds. If a refactor drops below threshold, the
//! test fails and forces a deliberate decision (raise nprobe,
//! re-tune rerank_mult, or accept the regression).

use bytes::Bytes;
use infino::superfile::VectorSearchOptions;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, distance, normalize};
use infino::superfile::vector::reader::{OpenOptions, VectorReader};
use infino::superfile::vector::rerank_codec::RerankCodec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// `rerank_mult` value used by every recall test. Pinned to the
/// public default exposed via [`VectorSearchOptions::DEFAULT_RERANK_MULT`]
/// so a future bump of the default automatically refreshes the
/// thresholds these tests defend.
const RERANK_MULT: usize = VectorSearchOptions::DEFAULT_RERANK_MULT;
use std::collections::HashSet;

const N_DOCS: usize = 10_000;
const DIM: usize = 384;
const N_CENT: usize = 64;
const N_QUERIES: usize = 50;

/// `VectorReader::search` is async on this branch; the recall body is
/// a synchronous sweep, so block on each query in place.
fn search_blocking(
    reader: &VectorReader,
    query: &[f32],
    k: usize,
    nprobe: usize,
    rerank_mult: usize,
) -> Vec<(u32, f32)> {
    futures::executor::block_on(reader.search("v", query, k, nprobe, rerank_mult)).expect("search")
}

fn generate_planted_corpus(seed: u64, normalize_each: bool) -> Vec<Vec<f32>> {
    // Generate `n_cent` random "centers" then sample each doc near
    // a random center. Result: planted-cluster geometry so IVF
    // clustering has a real signal to recover.
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    let centers: Vec<Vec<f32>> = (0..N_CENT)
        .map(|_| {
            (0..DIM)
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * 3.0
                })
                .collect()
        })
        .collect();

    let mut corpus = Vec::with_capacity(N_DOCS);
    for i in 0..N_DOCS {
        let cluster = i % N_CENT;
        let mut v: Vec<f32> = centers[cluster]
            .iter()
            .map(|&c| {
                let s: f64 = dist.sample(&mut rng);
                c + (s as f32) * 0.3
            })
            .collect();
        if normalize_each {
            normalize(&mut v);
        }
        corpus.push(v);
    }
    corpus
}

fn brute_force_top_k(corpus: &[Vec<f32>], query: &[f32], metric: Metric, k: usize) -> Vec<u32> {
    let mut hits: Vec<(u32, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, distance(metric, query, v)))
        .collect();
    hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.into_iter().take(k).map(|(d, _)| d).collect()
}

fn build_reader(corpus: &[Vec<f32>], metric: Metric) -> VectorReader {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim: DIM,
        n_cent: N_CENT,
        rot_seed: 7,
        metric,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");
    for v in corpus {
        b.add(0, v).expect("add to vector builder");
    }
    let bytes = b.finish().expect("finish vector builder");
    let metric_str = match metric {
        Metric::L2Sq => "l2sq",
        Metric::Cosine => "cosine",
        Metric::NegDot => "negdot",
    };
    let json = format!(
        r#"[{{"column":"v","dim":{DIM},"n_cent":{N_CENT},"rot_seed":7,"metric":"{metric_str}"}}]"#
    );
    VectorReader::open_with(
        Bytes::from(bytes),
        &json,
        OpenOptions {
            verify_crc: true,
            ..OpenOptions::default()
        },
    )
    .expect("open VectorReader")
}

/// Returns mean recall@k over `queries` against the planted ground
/// truth.
fn measure_recall(
    reader: &VectorReader,
    corpus: &[Vec<f32>],
    metric: Metric,
    queries: &[Vec<f32>],
    k: usize,
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let mut total: f32 = 0.0;
    for q in queries {
        let truth: HashSet<u32> = brute_force_top_k(corpus, q, metric, k)
            .into_iter()
            .collect();
        let approx: HashSet<u32> = search_blocking(reader, q, k, nprobe, rerank_mult)
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let hit_count = truth.intersection(&approx).count();
        total += (hit_count as f32) / (k as f32);
    }
    total / (queries.len() as f32)
}

fn build_query_set(corpus: &[Vec<f32>], seed: u64, normalize_each: bool) -> Vec<Vec<f32>> {
    // Use corpus members + perturbed midpoints as queries. Self-query
    // would inflate recall because the query is exactly indexed; we
    // perturb to make it a realistic "near a doc" query.
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    let mut queries = Vec::with_capacity(N_QUERIES);
    for i in 0..N_QUERIES {
        let base = &corpus[(i * 173) % corpus.len()];
        let mut q: Vec<f32> = base
            .iter()
            .map(|&v| {
                let s: f64 = dist.sample(&mut rng);
                v + (s as f32) * 0.05
            })
            .collect();
        if normalize_each {
            normalize(&mut q);
        }
        queries.push(q);
    }
    queries
}

fn recall_l2sq_at_10k_dim384_meets_threshold() {
    let corpus = generate_planted_corpus(1, false);
    let reader = build_reader(&corpus, Metric::L2Sq);
    let queries = build_query_set(&corpus, 100, false);

    // rerank_mult = 20 → rerank top 200 candidates by 1-bit
    // estimate, full-precision distance for those 200, keep top-10.
    // 1-bit RaBitQ at dim=384 has noise stddev ~1 in dot-product
    // units, so a small rerank_mult drops too much recall.
    let r10 = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, 8, RERANK_MULT);
    assert!(
        r10 >= 0.90,
        "L2Sq recall@10 at nprobe=8 rerank_mult=20 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, 32, RERANK_MULT);
    assert!(
        r10_high >= 0.95,
        "L2Sq recall@10 at nprobe=32 rerank_mult=20 below threshold: {r10_high:.3} < 0.95"
    );

    let r1 = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 1, 8, RERANK_MULT);
    assert!(
        r1 >= 0.95,
        "L2Sq recall@1 at nprobe=8 rerank_mult=20 below threshold: {r1:.3} < 0.95"
    );

    println!(
        "L2Sq @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}; recall@1/nprobe=8 = {r1:.3}"
    );
}

fn recall_cosine_at_10k_dim384_meets_threshold() {
    let corpus = generate_planted_corpus(2, true);
    let reader = build_reader(&corpus, Metric::Cosine);
    let queries = build_query_set(&corpus, 200, true);

    let r10 = measure_recall(&reader, &corpus, Metric::Cosine, &queries, 10, 8, RERANK_MULT);
    assert!(
        r10 >= 0.90,
        "Cosine recall@10 at nprobe=8 rerank_mult=20 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &corpus, Metric::Cosine, &queries, 10, 32, RERANK_MULT);
    assert!(
        r10_high >= 0.95,
        "Cosine recall@10 at nprobe=32 rerank_mult=20 below threshold: {r10_high:.3} < 0.95"
    );

    println!("Cosine @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}");
}

fn recall_increases_monotonically_with_nprobe() {
    // Sanity property: more nprobe = at least as much recall. A
    // non-monotone result suggests a quantization or rerank bug.
    let corpus = generate_planted_corpus(3, false);
    let reader = build_reader(&corpus, Metric::L2Sq);
    let queries = build_query_set(&corpus, 300, false);

    let mut prev: f32 = -1.0;
    for &nprobe in &[1, 2, 4, 8, 16, 32, 64] {
        let r = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, nprobe, 5);
        // Allow a small epsilon for stochastic tie-breaking; recall
        // must not *significantly* drop with more nprobe.
        assert!(
            r >= prev - 0.02,
            "recall regressed with more nprobe: nprobe={nprobe}, recall={r:.3}, prev={prev:.3}"
        );
        prev = r;
    }
}

fn recall_increases_monotonically_with_rerank_mult() {
    // Same property for rerank_mult: more candidates reranked = at
    // least as much recall. Catches a rerank-shortlist bug (e.g.,
    // off-by-one in the select_nth call).
    let corpus = generate_planted_corpus(4, false);
    let reader = build_reader(&corpus, Metric::L2Sq);
    let queries = build_query_set(&corpus, 400, false);

    let mut prev: f32 = -1.0;
    for &rerank_mult in &[1, 2, 5, 10, 20, 40] {
        let r = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, 16, rerank_mult);
        assert!(
            r >= prev - 0.02,
            "recall regressed with more rerank_mult: rerank_mult={rerank_mult}, recall={r:.3}, prev={prev:.3}"
        );
        prev = r;
    }
}

pub fn run() {
    println!("vector_recall: running 4 pinned-threshold checks (10K × 384)");
    recall_l2sq_at_10k_dim384_meets_threshold();
    recall_cosine_at_10k_dim384_meets_threshold();
    recall_increases_monotonically_with_nprobe();
    recall_increases_monotonically_with_rerank_mult();
    println!("vector_recall: all 4 pinned-threshold checks passed");
}
