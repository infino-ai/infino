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
    let dim = 32;
    let n = 200;
    let n_cent = 4;
    let corpus = generate_corpus(n, dim, 11, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, 7);

    // Use 5 different queries (corpus members + a synthetic one).
    for q_idx in [0usize, 47, 99, 142, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, 5);
        // nprobe = n_cent ⇒ scan everything; rerank_mult plenty.
        let approx = reader
            .search("v", query, 5, n_cent, 40)
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
    let dim = 32;
    let n = 200;
    let n_cent = 4;
    // Cosine requires unit-norm inputs.
    let corpus = generate_corpus(n, dim, 13, true);
    let reader = build_reader(&corpus, dim, n_cent, Metric::Cosine, 17);

    for q_idx in [0usize, 50, 100, 150, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::Cosine, 5);
        let approx = reader
            .search("v", query, 5, n_cent, 40)
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
    let dim = 32;
    let n = 200;
    let n_cent = 4;
    let corpus = generate_corpus(n, dim, 19, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::NegDot, 23);

    for q_idx in [0usize, 33, 77, 145, 199] {
        let query = &corpus[q_idx];
        let exact = brute_force_top_k(&corpus, query, Metric::NegDot, 5);
        let approx = reader
            .search("v", query, 5, n_cent, 40)
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
    let dim = 32;
    let n = 200;
    let n_cent = 8;
    let corpus = generate_corpus(n, dim, 29, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, 31);

    for q_idx in [10usize, 50, 100, 150] {
        let query = &corpus[q_idx];
        let approx = reader.search("v", query, 5, 1, 10).expect("FTS search");
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
    let dim = 32;
    let n = 100;
    let n_cent = 4;
    let corpus = generate_corpus(n, dim, 37, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, 41);
    let query = &corpus[42];
    let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, 5);
    let approx = reader
        .search("v", query, 5, n_cent, 40)
        .expect("FTS search");
    // Build doc_id → exact_distance map.
    let exact_map: std::collections::HashMap<u32, f32> = exact.iter().copied().collect();
    for (d, approx_dist) in &approx {
        let exact_dist = exact_map[d];
        let abs_err = (approx_dist - exact_dist).abs();
        let rel_err = abs_err / exact_dist.abs().max(1e-6);
        assert!(
            abs_err < 1e-3 || rel_err < 1e-4,
            "doc {d}: approx_dist={approx_dist} exact_dist={exact_dist}"
        );
    }
}

#[tokio::test]
async fn oracle_nonself_query_topk_recovered() {
    // Query is *not* a corpus member; both engines must agree on
    // top-k under full-nprobe. This isolates "is rerank correct"
    // from "do you find yourself".
    let dim = 32;
    let n = 200;
    let n_cent = 4;
    let corpus = generate_corpus(n, dim, 43, false);
    let reader = build_reader(&corpus, dim, n_cent, Metric::L2Sq, 47);

    // Synthesize a query as midpoint of two corpus vectors.
    let q: Vec<f32> = corpus[5]
        .iter()
        .zip(corpus[150].iter())
        .map(|(a, b)| (a + b) * 0.5)
        .collect();
    let exact = brute_force_top_k(&corpus, &q, Metric::L2Sq, 5);
    // rerank_mult chosen so k * rerank_mult ≥ n; covers the whole
    // corpus through rerank, isolating the test from 1-bit estimate
    // tail loss (which is expected behavior, just not what this
    // oracle checks).
    let approx = reader.search("v", &q, 5, n_cent, 40).expect("FTS search");
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
    let dim = 32;
    let n = 100;
    let corpus = generate_corpus(n, dim, 53, false);
    let reader = build_reader(&corpus, dim, 4, Metric::L2Sq, 59);
    let query = &corpus[7];
    let approx = reader.search("v", query, 10, 4, 10).expect("FTS search");
    for w in approx.windows(2) {
        assert!(w[0].1 <= w[1].1, "distances must be non-decreasing");
    }
    // And the bottom of our 10 should not be closer than the brute-
    // force 10th.
    let exact = brute_force_top_k(&corpus, query, Metric::L2Sq, 10);
    let approx_max = approx.last().expect("last element").1;
    let exact_max = exact.last().expect("last element").1;
    let abs_err = (approx_max - exact_max).abs();
    let rel_err = abs_err / exact_max.abs().max(1e-6);
    assert!(
        abs_err < 1e-3 || rel_err < 1e-4,
        "approx top-10 boundary diverges: approx={approx_max} exact={exact_max}"
    );
}
