// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Recall test that exercises the reservoir sampling path
//! end-to-end with `n_docs > sample_size`.
//!
//! Default reservoir size is `max(100K, min(500K, 64 × n_cent))`, so
//! the normal-scale tests in `brute_force_oracle.rs` and
//! `against_lance.rs` run with `n_docs ≤ sample_size` — the
//! reservoir holds the full corpus, and k-means training is
//! exactly equivalent to the full-corpus training path. That's necessary for
//! "no regression on small corpora" but doesn't probe the
//! actual sampling logic.
//!
//! This test uses [`VectorBuilder::set_kmeans_sample_size`] to
//! override the reservoir to be deliberately smaller than the
//! corpus (`sample_size = n_docs / 10`) and asserts that recall
//! against brute-force ground truth stays above a conservative
//! threshold. If the reservoir were biased or the
//! `assign_to_centroids` plumbing were broken, recall would
//! collapse here long before it would in the bench harness.

use bytes::Bytes;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, distance, normalize};
use infino::superfile::vector::reader::VectorReader;
use infino::superfile::vector::rerank_codec::RerankCodec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::collections::HashSet;

/// Random-rotation seed shared by both reservoir-recall fixtures.
const ROT_SEED: u64 = 7;
/// Vector dimension for both fixtures.
const RESERVOIR_DIM: usize = 32;
// Undersized-reservoir stress fixture (`n_docs > sample_size`).
const RESERVOIR_N_CENT: usize = 8;
const RESERVOIR_N_DOCS: usize = 2_000;
const RESERVOIR_SAMPLE_SIZE: usize = 200;
const RESERVOIR_TOP_K: usize = 10;
const RESERVOIR_CORPUS_SEED: u64 = 42;
/// Floor on rerank_mult for maximal-coverage retrieval.
const RESERVOIR_RERANK_MULT_MIN: usize = 64;
/// Conservative mean-recall pass threshold for the undersized run.
const RESERVOIR_RECALL_FLOOR: f32 = 0.85;
/// Query row indices probed for recall measurement.
const RESERVOIR_QUERY_INDICES: [usize; 8] = [0, 137, 251, 503, 911, 1234, 1567, 1999];
// Default-reservoir fixture (reservoir holds the whole corpus).
const DEFAULT_RESERVOIR_N_CENT: usize = 4;
const DEFAULT_RESERVOIR_N_DOCS: usize = 200;
const DEFAULT_RESERVOIR_TOP_K: usize = 5;
const DEFAULT_RESERVOIR_CORPUS_SEED: u64 = 19;
const DEFAULT_RESERVOIR_RERANK_MULT: usize = 40;

/// Planted-cluster corpus: `N_CLUSTERS` Gaussian centers, each
/// with `n_docs / N_CLUSTERS` near-by samples. Centers are
/// generated independently so the cluster geometry is rich
/// enough to challenge IVF; the reservoir sample needs to
/// recover representatives of each center, otherwise k-means
/// will collapse multiple planted clusters into one IVF
/// partition and recall will degrade visibly.
fn corpus(n_docs: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * 3.0
                })
                .collect()
        })
        .collect();
    let mut out = Vec::with_capacity(n_docs * dim);
    for i in 0..n_docs {
        let center = &centers[i % n_clusters];
        let mut v: Vec<f32> = center
            .iter()
            .map(|&c| {
                let s: f64 = dist.sample(&mut rng);
                c + (s as f32) * 0.3
            })
            .collect();
        normalize(&mut v);
        out.extend_from_slice(&v);
    }
    out
}

fn brute_force_top_k(
    corpus_flat: &[f32],
    dim: usize,
    n_docs: usize,
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<u32> {
    let mut hits: Vec<(u32, f32)> = (0..n_docs)
        .map(|i| {
            let v = &corpus_flat[i * dim..(i + 1) * dim];
            (i as u32, distance(metric, query, v))
        })
        .collect();
    hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits.into_iter().map(|(d, _)| d).collect()
}

fn build_reader_with_sample_size(
    corpus_flat: &[f32],
    dim: usize,
    n_docs: usize,
    n_cent: usize,
    sample_size: usize,
    rerank_codec: RerankCodec,
) -> VectorReader {
    let mut b = VectorBuilder::new();
    let cid = b
        .register_column(VectorConfig {
            column: "v".into(),
            dim,
            n_cent,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec,
        })
        .expect("register column");
    b.set_kmeans_sample_size(cid, sample_size)
        .expect("override sample size");
    for i in 0..n_docs {
        b.add(0, &corpus_flat[i * dim..(i + 1) * dim])
            .expect("add to vector builder");
    }
    let bytes = b.finish().expect("finish vector builder");
    let json = format!(
        r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":{ROT_SEED},"metric":"cosine"}}]"#
    );
    VectorReader::open(Bytes::from(bytes), &json).expect("open VectorReader")
}

#[tokio::test]
async fn recall_under_undersized_reservoir_matches_brute_force() {
    let dim = RESERVOIR_DIM;
    let n_cent = RESERVOIR_N_CENT;
    let n_docs = RESERVOIR_N_DOCS;
    // 1/10 the corpus — well below the default 100K floor and a
    // valid stress for the sampler. At n_clusters == n_cent the
    // reservoir must capture representatives of every planted
    // center; if it doesn't, recall will drop visibly.
    let sample_size = RESERVOIR_SAMPLE_SIZE;
    let top_k = RESERVOIR_TOP_K;

    let flat = corpus(n_docs, dim, n_cent, /*seed=*/ RESERVOIR_CORPUS_SEED);

    // Maximal-coverage retrieval: full nprobe sweep and a wide
    // rerank pool. Any recall loss here is k-means-related, not
    // IVF-pruning-related — exactly what we want to validate.
    let nprobe = n_cent;
    let rerank_mult = (n_docs / top_k + 1).max(RESERVOIR_RERANK_MULT_MIN);

    let queries: [usize; 8] = RESERVOIR_QUERY_INDICES;

    // Parameterize across every reranking codec. All three share
    // the same 0.85 recall floor on this corpus: Fp32 is bit-exact,
    // and Sq8's per-cluster quantizer recovers fp32-equivalent
    // recall at this dim/cluster shape. `RabitqOnly` is excluded
    // because it skips the rerank step entirely and is covered
    // separately by `rabitq_only_self_query_ranks_self_first` in
    // reader.rs.
    for codec in [RerankCodec::Fp32, RerankCodec::Sq8ResidualEpsilon] {
        let reader = build_reader_with_sample_size(&flat, dim, n_docs, n_cent, sample_size, codec);
        let mut total_recall = 0.0f32;
        for q_idx in queries {
            let query = &flat[q_idx * dim..(q_idx + 1) * dim];
            let approx: Vec<u32> = reader
                .search("v", query, top_k, nprobe, rerank_mult)
                .await
                .expect("search")
                .into_iter()
                .map(|(d, _)| d)
                .collect();
            let exact = brute_force_top_k(&flat, dim, n_docs, query, Metric::Cosine, top_k);

            // Self-NN invariant: query is exactly a corpus row, so
            // its own doc id must be top-1 regardless of codec.
            // True for every reranking codec because the rerank
            // step distinguishes the exact-match candidate.
            assert_eq!(
                approx[0] as usize, q_idx,
                "self-NN broken at query {q_idx} under codec {codec:?}: top-1={}",
                approx[0]
            );

            let a: HashSet<u32> = approx.iter().copied().collect();
            let e: HashSet<u32> = exact.iter().copied().collect();
            let intersect = a.intersection(&e).count() as f32;
            let recall = intersect / (top_k as f32);
            total_recall += recall;
        }
        let mean_recall = total_recall / queries.len() as f32;
        // 0.85 is comfortably below what a properly-sampled
        // reservoir delivers in this regime (empirically ≥ 0.95).
        // A failure here means either the reservoir is biased,
        // `assign_to_centroids` produced wrong assignments, or
        // the codec's rerank kernel regressed — all
        // implementation bugs.
        assert!(
            mean_recall >= RESERVOIR_RECALL_FLOOR,
            "reservoir-trained recall@{top_k} = {mean_recall:.3} \
             under sample_size={sample_size} codec={codec:?}; expected ≥ 0.85"
        );
    }
}

#[tokio::test]
async fn recall_with_default_reservoir_equivalent_to_full_corpus_training() {
    // Sanity check: when the corpus fits inside the default
    // reservoir (which is the normal regime for unit tests), the
    // result should be self-NN-perfect just like the full-corpus path.
    let dim = RESERVOIR_DIM;
    let n_cent = DEFAULT_RESERVOIR_N_CENT;
    let n_docs = DEFAULT_RESERVOIR_N_DOCS;
    let top_k = DEFAULT_RESERVOIR_TOP_K;

    let flat = corpus(
        n_docs,
        dim,
        n_cent,
        /*seed=*/ DEFAULT_RESERVOIR_CORPUS_SEED,
    );
    // Same call site as the test above but without the override,
    // so the default sample size (100K) is in effect; reservoir
    // holds the full 200-doc corpus.
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim,
        n_cent,
        rot_seed: ROT_SEED,
        metric: Metric::Cosine,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");
    for i in 0..n_docs {
        b.add(0, &flat[i * dim..(i + 1) * dim])
            .expect("add to vector builder");
    }
    let bytes = b.finish().expect("finish vector builder");
    let json = format!(
        r#"[{{"column":"v","dim":{dim},"n_cent":{n_cent},"rot_seed":{ROT_SEED},"metric":"cosine"}}]"#
    );
    let reader = VectorReader::open(Bytes::from(bytes), &json).expect("open VectorReader");

    for q_idx in [0usize, 73, 142, 199] {
        let query = &flat[q_idx * dim..(q_idx + 1) * dim];
        let approx: Vec<u32> = reader
            .search("v", query, top_k, n_cent, DEFAULT_RESERVOIR_RERANK_MULT)
            .await
            .expect("search")
            .into_iter()
            .map(|(d, _)| d)
            .collect();
        let exact = brute_force_top_k(&flat, dim, n_docs, query, Metric::Cosine, top_k);
        let a: HashSet<u32> = approx.iter().copied().collect();
        let e: HashSet<u32> = exact.iter().copied().collect();
        assert_eq!(
            a, e,
            "default-reservoir top-{top_k} set diverges at q={q_idx}"
        );
    }
}
