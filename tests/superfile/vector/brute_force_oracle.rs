// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Vector kNN oracle: IVF + RaBitQ + rerank vs O(N) exact brute force.
//!
//! For each query and metric, we compute the exact top-k by scanning
//! every original full-precision vector and asserting our pipeline
//! recovers the same top-k. With sufficient nprobe coverage (= scan
//! all clusters) the recall must be 100%; with reduced nprobe we
//! check that the *most-similar* doc is still recovered (top-1
//! recall).
//!
//! ## What this oracle catches
//!
//! Bugs in any of the four pipeline stages — clustering (k-means
//! convergence + cluster-contiguous storage), random-rotation
//! determinism, 1-bit quantization estimate, full-precision rerank —
//! can produce internally-consistent results that disagree with the
//! exact ground truth. Brute force is the algorithm-isolating
//! reference; if our IVF pipeline disagrees, the bug is in the
//! pipeline, not in the corpus.
//!
//! ## Coverage
//!
//! Tests run for all three metrics (L2Sq, Cosine, NegDot) at a small
//! corpus size where O(N) brute force is cheap (n=200, dim=32).
//! Larger-scale recall tests live in `tests/recall.rs`.

use bytes::Bytes;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, distance, normalize};
use infino::superfile::vector::reader::VectorReader;
use infino::superfile::vector::rerank_codec::RerankCodec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

/// Standard small-corpus oracle shape (brute force stays cheap).
const ORACLE_DIM: usize = 32;
const ORACLE_N_DOCS: usize = 200;
const ORACLE_N_CENT: usize = 4;
/// Top-k and rerank pool for the full-nprobe oracle comparisons.
const ORACLE_TOP_K: usize = 5;
const ORACLE_RERANK_MULT: usize = 40;
/// Smaller corpus for the distance-tolerance / ordering tests.
const ORACLE_SMALL_N_DOCS: usize = 100;
/// Distance-agreement tolerances vs brute force (abs OR rel passes).
const DISTANCE_ABS_TOLERANCE: f32 = 1e-3;
const DISTANCE_REL_TOLERANCE: f32 = 1e-4;
/// Relative-error denominator floor (avoids divide-by-zero).
const DISTANCE_REL_FLOOR: f32 = 1e-6;

// Per-test corpus + rotation seeds (distinct so fixtures differ).
const L2SQ_CORPUS_SEED: u64 = 11;
const L2SQ_ROT_SEED: u64 = 7;
const COSINE_CORPUS_SEED: u64 = 13;
const COSINE_ROT_SEED: u64 = 17;
const NEGDOT_CORPUS_SEED: u64 = 19;
const NEGDOT_ROT_SEED: u64 = 23;
const PARTIAL_CORPUS_SEED: u64 = 29;
const PARTIAL_ROT_SEED: u64 = 31;
const TOLERANCE_CORPUS_SEED: u64 = 37;
const TOLERANCE_ROT_SEED: u64 = 41;
const NONSELF_CORPUS_SEED: u64 = 43;
const NONSELF_ROT_SEED: u64 = 47;
const ORDERING_CORPUS_SEED: u64 = 53;
const ORDERING_ROT_SEED: u64 = 59;

/// Partial-nprobe top-1 test: more clusters, reduced probe/rerank.
const PARTIAL_N_CENT: usize = 8;
const PARTIAL_NPROBE: usize = 1;
const PARTIAL_RERANK_MULT: usize = 10;
/// Ordering test: top-10 at full nprobe, modest rerank.
const ORDERING_TOP_K: usize = 10;
const ORDERING_RERANK_MULT: usize = 10;
/// Blend factor for the synthetic non-self midpoint query.
const NONSELF_BLEND: f32 = 0.5;

/// Generate `n` deterministic vectors at `dim` dimensions. For
/// cosine, normalizes each vector to unit norm.
fn generate_corpus(n: usize, dim: usize, seed: u64, normalize_each: bool) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim)
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    s as f32
                })
                .collect();
            if normalize_each {
                normalize(&mut v);
            }
            v
        })
        .collect()
}

/// Compute exact top-k by brute force: distance to every doc, sort,
/// take first k. Returns (doc_id, distance) pairs in distance-
/// ascending order (smaller = closer for every metric — see
/// `distance::distance`).
fn brute_force_top_k(
    corpus: &[Vec<f32>],
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<(u32, f32)> {
    let mut hits: Vec<(u32, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, distance(metric, query, v)))
        .collect();
    hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits
}

/// Build a vector blob from the corpus with given metric, return a
/// VectorReader plus the original full-precision vectors (used for
/// brute force).
fn build_reader(
    corpus: &[Vec<f32>],
    dim: usize,
    n_cent: usize,
    metric: Metric,
    rot_seed: u64,
) -> VectorReader {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim,
        n_cent,
        rot_seed,
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
        r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":{rot_seed},"metric":"{metric_str}"}}]"#
    );
    VectorReader::open(Bytes::from(bytes), &json).expect("open VectorReader")
}

#[tokio::test]
async fn oracle_l2sq_full_nprobe_recovers_exact_topk() {
    let dim = ORACLE_DIM;
    let n = ORACLE_N_DOCS;
    let n_cent = ORACLE_N_CENT;
    let corpus = generate_corpus(n, dim, L2SQ_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, L2SQ_ROT_SEED);

    // Use 5 different queries (corpus members + a synthetic one).
    for q_idx in [0usize, 47, 99, 142, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, ORACLE_TOP_K);
        // nprobe = n_cent ⇒ scan everything; rerank_mult plenty.
        let approx = reader
            .search("v", query, ORACLE_TOP_K, n_cent, ORACLE_RERANK_MULT)
            .await
            .expect("FTS search");
        // Exact should fully match approx: same doc set, top-1 must
        // be the query itself (distance 0).
        assert_eq!(approx[0].0 as usize, q_idx, "self-NN must be top-1");
        let exact_set: std::collections::HashSet<u32> = exact.iter().map(|(d, _)| *d).collect();
        let approx_set: std::collections::HashSet<u32> = approx.iter().map(|(d, _)| *d).collect();
        assert_eq!(
            exact_set, approx_set,
            "L2Sq full-nprobe top-5 set diverges from brute force; query={q_idx}"
        );
    }
}

#[tokio::test]
async fn oracle_cosine_full_nprobe_recovers_exact_topk() {
    let dim = ORACLE_DIM;
    let n = ORACLE_N_DOCS;
    let n_cent = ORACLE_N_CENT;
    // Cosine requires unit-norm inputs.
    let corpus = generate_corpus(n, dim, COSINE_CORPUS_SEED, true);
    let reader = build_reader(&corpus, dim, n_cent, Metric::Cosine, COSINE_ROT_SEED);

    for q_idx in [0usize, 50, 100, 150, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::Cosine, ORACLE_TOP_K);
        let approx = reader
            .search("v", query, ORACLE_TOP_K, n_cent, ORACLE_RERANK_MULT)
            .await
            .expect("FTS search");
        assert_eq!(approx[0].0 as usize, q_idx);
        let exact_set: std::collections::HashSet<u32> = exact.iter().map(|(d, _)| *d).collect();
        let approx_set: std::collections::HashSet<u32> = approx.iter().map(|(d, _)| *d).collect();
        assert_eq!(
            exact_set, approx_set,
            "Cosine full-nprobe top-5 set diverges; query={q_idx}"
        );
    }
}

#[tokio::test]
async fn oracle_negdot_full_nprobe_recovers_exact_topk() {
    let dim = ORACLE_DIM;
    let n = ORACLE_N_DOCS;
    let n_cent = ORACLE_N_CENT;
    let corpus = generate_corpus(n, dim, NEGDOT_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::NegDot, NEGDOT_ROT_SEED);

    for q_idx in [0usize, 33, 77, 145, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::NegDot, ORACLE_TOP_K);
        let approx = reader
            .search("v", query, ORACLE_TOP_K, n_cent, ORACLE_RERANK_MULT)
            .await
            .expect("FTS search");
        // For NegDot, self-NN is *most negative dot* — for non-unit
        // vectors that's not necessarily the query itself. So we
        // only assert set agreement.
        let exact_set: std::collections::HashSet<u32> = exact.iter().map(|(d, _)| *d).collect();
        let approx_set: std::collections::HashSet<u32> = approx.iter().map(|(d, _)| *d).collect();
        assert_eq!(
            exact_set, approx_set,
            "NegDot full-nprobe top-5 set diverges; query={q_idx}"
        );
    }
}

#[tokio::test]
async fn oracle_partial_nprobe_top1_preserved() {
    // With reduced nprobe we may miss tail of the top-k, but the
    // single most-similar doc (= the query itself for self-query) is
    // still in the cluster the query lands in, so top-1 must
    // survive.
    let dim = ORACLE_DIM;
    let n = ORACLE_N_DOCS;
    let n_cent = PARTIAL_N_CENT;
    let corpus = generate_corpus(n, dim, PARTIAL_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, PARTIAL_ROT_SEED);

    for q_idx in [10usize, 50, 100, 150] {
        let query = &corpus[q_idx];
        let approx = reader
            .search(
                "v",
                query,
                ORACLE_TOP_K,
                PARTIAL_NPROBE,
                PARTIAL_RERANK_MULT,
            )
            .await
            .expect("FTS search");
        assert_eq!(
            approx[0].0 as usize, q_idx,
            "top-1 self-recall failed at nprobe=1, query={q_idx}"
        );
    }
}

#[tokio::test]
async fn oracle_distances_match_brute_force_within_tolerance() {
    // For full-nprobe + max rerank, our reported distance should
    // equal the brute-force distance to within float noise.
    let dim = ORACLE_DIM;
    let n = ORACLE_SMALL_N_DOCS;
    let n_cent = ORACLE_N_CENT;
    let corpus = generate_corpus(n, dim, TOLERANCE_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, TOLERANCE_ROT_SEED);
    let query = &corpus[42];
    let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, ORACLE_TOP_K);
    let approx = reader
        .search("v", query, ORACLE_TOP_K, n_cent, ORACLE_RERANK_MULT)
        .await
        .expect("FTS search");
    // Build doc_id → exact_distance map.
    let exact_map: std::collections::HashMap<u32, f32> = exact.iter().copied().collect();
    for (d, approx_dist) in &approx {
        let exact_dist = exact_map[d];
        let abs_err = (approx_dist - exact_dist).abs();
        let rel_err = abs_err / exact_dist.abs().max(DISTANCE_REL_FLOOR);
        assert!(
            abs_err < DISTANCE_ABS_TOLERANCE || rel_err < DISTANCE_REL_TOLERANCE,
            "doc {d}: approx_dist={approx_dist} exact_dist={exact_dist}"
        );
    }
}

#[tokio::test]
async fn oracle_nonself_query_topk_recovered() {
    // Query is *not* a corpus member; both engines must agree on
    // top-k under full-nprobe. This isolates "is rerank correct"
    // from "do you find yourself".
    let dim = ORACLE_DIM;
    let n = ORACLE_N_DOCS;
    let n_cent = ORACLE_N_CENT;
    let corpus = generate_corpus(n, dim, NONSELF_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, NONSELF_ROT_SEED);

    // Synthesize a query as midpoint of two corpus vectors.
    let q: Vec<f32> = corpus[5]
        .iter()
        .zip(corpus[150].iter())
        .map(|(a, b)| (a + b) * NONSELF_BLEND)
        .collect();
    let exact = brute_force_top_k(&corpus, &q, Metric::L2Sq, ORACLE_TOP_K);
    // rerank_mult chosen so k * rerank_mult ≥ n; covers the whole
    // corpus through rerank, isolating the test from 1-bit estimate
    // tail loss (which is expected behavior, just not what this
    // oracle checks).
    let approx = reader
        .search("v", &q, ORACLE_TOP_K, n_cent, ORACLE_RERANK_MULT)
        .await
        .expect("FTS search");
    let exact_set: std::collections::HashSet<u32> = exact.iter().map(|(d, _)| *d).collect();
    let approx_set: std::collections::HashSet<u32> = approx.iter().map(|(d, _)| *d).collect();
    assert_eq!(
        exact_set, approx_set,
        "non-self full-nprobe top-5 set diverges from brute force"
    );
}

#[tokio::test]
async fn oracle_topk_distance_ordering_matches_exact() {
    // The order of (doc, distance) pairs from our reader, after
    // full-nprobe, should agree with brute-force ordering modulo
    // tied scores. Test the strict-monotonicity invariant: distances
    // are non-decreasing.
    let dim = ORACLE_DIM;
    let n = ORACLE_SMALL_N_DOCS;
    let corpus = generate_corpus(n, dim, ORDERING_CORPUS_SEED, false);
    let reader = build_reader(&corpus, dim, ORACLE_N_CENT, Metric::L2Sq, ORDERING_ROT_SEED);
    let query = &corpus[7];
    let approx = reader
        .search(
            "v",
            query,
            ORDERING_TOP_K,
            ORACLE_N_CENT,
            ORDERING_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    for w in approx.windows(2) {
        assert!(w[0].1 <= w[1].1, "distances must be non-decreasing");
    }
    // And the bottom of our 10 should not be closer than the brute-
    // force 10th.
    let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, ORDERING_TOP_K);
    let approx_max = approx.last().expect("last element").1;
    let exact_max = exact.last().expect("last element").1;
    let abs_err = (approx_max - exact_max).abs();
    let rel_err = abs_err / exact_max.abs().max(DISTANCE_REL_FLOOR);
    assert!(
        abs_err < DISTANCE_ABS_TOLERANCE || rel_err < DISTANCE_REL_TOLERANCE,
        "approx top-10 boundary diverges: approx={approx_max} exact={exact_max}"
    );
}
