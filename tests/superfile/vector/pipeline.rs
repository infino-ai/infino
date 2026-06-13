// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end vector kNN pipeline integration test.
//!
//! Builds a multi-column vector blob, opens it, exercises kNN search.
//! Mirrors the planted-ground-truth correctness pattern.

use bytes::Bytes;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, normalize};
use infino::superfile::vector::reader::VectorReader;
use infino::superfile::vector::rerank_codec::RerankCodec;

/// `text_emb` column shape (cosine, unit-norm).
const TEXT_EMB_DIM: usize = 16;
const TEXT_EMB_ROT_SEED: u64 = 11;
/// `image_emb` column shape (L2Sq).
const IMAGE_EMB_DIM: usize = 24;
const IMAGE_EMB_ROT_SEED: u64 = 22;
/// IVF centroid count for both columns.
const TWO_COL_N_CENT: usize = 4;
/// Deterministic per-doc vector hash recipe (shared by build + query
/// reconstruction): `((i*MUL + j*AXIS_MUL) % MOD) * SCALE [+ BIAS]`.
const TEXT_HASH_MUL: u32 = 31;
const TEXT_AXIS_MUL: u32 = 3;
const IMAGE_HASH_MUL: u32 = 17;
const IMAGE_AXIS_MUL: u32 = 7;
const HASH_MOD: u32 = 100;
const HASH_SCALE: f32 = 0.01;
const TEXT_BIAS: f32 = 0.1;
/// Default doc count for the two-column fixture.
const TWO_COL_DEFAULT_N_DOCS: u32 = 80;
/// Doc count for the routing / dim-mismatch tests.
const ROUTING_TEST_N_DOCS: u32 = 60;
/// Doc count for the summary-centroid test.
const SUMMARY_TEST_N_DOCS: u32 = 40;
/// Self-query target doc index.
const SELF_QUERY_TARGET: u32 = 17;
/// Search parameters for the pipeline vector searches.
const VECTOR_PIPELINE_K: usize = 5;
const VECTOR_PIPELINE_TOPK_LIMIT_K: usize = 3;
const VECTOR_PIPELINE_NPROBE: usize = 4;
const VECTOR_PIPELINE_RERANK_MULT: usize = 5;
/// Self-distance tolerance (a doc queried with its own vector ≈ 0).
const SELF_DIST_TOLERANCE: f32 = 1e-3;
/// BM25/vector top-k with headroom for routing/summary/cluster tests.
const VECTOR_PIPELINE_HEADROOM_K: usize = 10;
/// Planted-cluster test fixture parameters.
const CLUSTER_TEST_N_CENT: usize = 3;
const CLUSTER_TEST_ROT_SEED: u64 = 42;
const DOCS_PER_CLUSTER: usize = 20;
const CLUSTER_NOISE_MOD: usize = 7;

/// Build a 2-column vector blob: text_emb (dim=16, cosine) and
/// image_emb (dim=24, l2sq), each with `n_docs` deterministic vectors.
fn build_two_column_blob(n_docs: u32) -> (Bytes, String) {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "text_emb".into(),
        dim: TEXT_EMB_DIM,
        n_cent: TWO_COL_N_CENT,
        rot_seed: TEXT_EMB_ROT_SEED,
        metric: Metric::Cosine,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");
    b.register_column(VectorConfig {
        column: "image_emb".into(),
        dim: IMAGE_EMB_DIM,
        n_cent: TWO_COL_N_CENT,
        rot_seed: IMAGE_EMB_ROT_SEED,
        metric: Metric::L2Sq,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");

    for i in 0..n_docs {
        // Deterministic per-doc vectors with simple structure so we can
        // make planted-ground-truth assertions.
        let mut v_text: Vec<f32> = (0..TEXT_EMB_DIM)
            .map(|j| {
                ((i.wrapping_mul(TEXT_HASH_MUL) + j as u32 * TEXT_AXIS_MUL) % HASH_MOD) as f32
                    * HASH_SCALE
                    + TEXT_BIAS
            })
            .collect();
        // Cosine metric requires unit-norm inputs.
        normalize(&mut v_text);
        let v_img: Vec<f32> = (0..IMAGE_EMB_DIM)
            .map(|j| {
                ((i.wrapping_mul(IMAGE_HASH_MUL) + j as u32 * IMAGE_AXIS_MUL) % HASH_MOD) as f32
                    * HASH_SCALE
            })
            .collect();
        b.add(0, &v_text).expect("add to vector builder");
        b.add(1, &v_img).expect("add to vector builder");
    }

    let bytes = b.finish().expect("finish vector builder");
    let json = r#"[
        {"column":"text_emb","dim":16,"n_cent":4,"rot_seed":11,"metric":"cosine"},
        {"column":"image_emb","dim":24,"n_cent":4,"rot_seed":22,"metric":"l2sq"}
    ]"#;
    (Bytes::from(bytes), json.to_string())
}

#[tokio::test]
async fn end_to_end_self_query_recovers_self() {
    let n_docs = TWO_COL_DEFAULT_N_DOCS;
    let (blob, json) = build_two_column_blob(n_docs);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    assert_eq!(r.n_docs(), n_docs as u64);

    // Reconstruct doc 17's (normalized) text_emb vector for the query.
    let target = SELF_QUERY_TARGET;
    let mut q_text: Vec<f32> = (0..TEXT_EMB_DIM)
        .map(|j| {
            ((target.wrapping_mul(TEXT_HASH_MUL) + j as u32 * TEXT_AXIS_MUL) % HASH_MOD) as f32
                * HASH_SCALE
                + TEXT_BIAS
        })
        .collect();
    normalize(&mut q_text);
    let hits = r
        .search(
            "text_emb",
            &q_text,
            VECTOR_PIPELINE_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    assert_eq!(hits[0].0, target, "self should be top-1");
    // Cosine distance to self for unit-norm vector = 1 - 1 = 0.
    assert!(
        hits[0].1 < SELF_DIST_TOLERANCE,
        "cosine self-distance should be ~0, got {}",
        hits[0].1
    );
}

#[tokio::test]
async fn end_to_end_l2sq_self_query_distance_is_zero() {
    let (blob, json) = build_two_column_blob(TWO_COL_DEFAULT_N_DOCS);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let target = 5u32;
    let q_img: Vec<f32> = (0..IMAGE_EMB_DIM)
        .map(|j| {
            ((target.wrapping_mul(IMAGE_HASH_MUL) + j as u32 * IMAGE_AXIS_MUL) % HASH_MOD) as f32
                * HASH_SCALE
        })
        .collect();
    let hits = r
        .search(
            "image_emb",
            &q_img,
            VECTOR_PIPELINE_TOPK_LIMIT_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    assert_eq!(hits[0].0, target);
    // L2² of v with itself is exactly 0.
    assert!(
        hits[0].1 < SELF_DIST_TOLERANCE,
        "self L2² should be ~0, got {}",
        hits[0].1
    );
}

#[tokio::test]
async fn end_to_end_multi_column_routing_isolated() {
    let (blob, json) = build_two_column_blob(ROUTING_TEST_N_DOCS);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");

    // text_emb is dim=16; querying with a dim=24 image vector must error.
    let v_img: Vec<f32> = vec![0.5; IMAGE_EMB_DIM];
    let err = r
        .search(
            "text_emb",
            &v_img,
            VECTOR_PIPELINE_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await;
    assert!(err.is_err(), "dim mismatch must error");

    // And vice versa.
    let v_text: Vec<f32> = vec![0.5; TEXT_EMB_DIM];
    let err = r
        .search(
            "image_emb",
            &v_text,
            VECTOR_PIPELINE_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await;
    assert!(err.is_err());
}

#[tokio::test]
async fn end_to_end_top_k_limits_results() {
    let (blob, json) = build_two_column_blob(TWO_COL_DEFAULT_N_DOCS);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let q: Vec<f32> = vec![0.3; TEXT_EMB_DIM];
    let hits = r
        .search(
            "text_emb",
            &q,
            VECTOR_PIPELINE_TOPK_LIMIT_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    assert!(hits.len() <= 3);
}

#[test]
fn end_to_end_summary_per_column() {
    let (blob, json) = build_two_column_blob(SUMMARY_TEST_N_DOCS);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");

    let (text_centroid, text_radius) = r.summary("text_emb").expect("vector summary");
    assert_eq!(text_centroid.len(), TEXT_EMB_DIM);
    assert!(text_radius >= 0.0);

    let (img_centroid, img_radius) = r.summary("image_emb").expect("vector summary");
    assert_eq!(img_centroid.len(), IMAGE_EMB_DIM);
    assert!(img_radius >= 0.0);

    // Different columns should have different summary centroids
    // (different data, different dim). Just sanity-check shapes.
    assert!(r.summary("nonexistent").is_none());
}

#[tokio::test]
async fn end_to_end_planted_clusters_recovered() {
    // Plant 3 well-separated clusters in dim=16; verify nearest-neighbor
    // for a query at one center pulls back docs from that cluster.
    let dim = TEXT_EMB_DIM;
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim,
        n_cent: CLUSTER_TEST_N_CENT,
        rot_seed: CLUSTER_TEST_ROT_SEED,
        metric: Metric::L2Sq,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");

    let centers = [
        [
            10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [
            0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    ];
    let mut planted_cluster: Vec<u32> = Vec::new();
    let mut next_doc_id: u32 = 0;
    for (cluster_idx, c) in centers.iter().enumerate() {
        for d in 0..DOCS_PER_CLUSTER {
            let mut v = c.to_vec();
            // Tiny per-doc noise so docs aren't identical.
            for (j, slot) in v.iter_mut().enumerate() {
                *slot += ((cluster_idx * DOCS_PER_CLUSTER + d + j) % CLUSTER_NOISE_MOD) as f32
                    * HASH_SCALE;
            }
            b.add(0, &v).expect("add to vector builder");
            planted_cluster.push(cluster_idx as u32);
            next_doc_id += 1;
        }
    }
    assert_eq!(next_doc_id, (CLUSTER_TEST_N_CENT * DOCS_PER_CLUSTER) as u32);

    let bytes = b.finish().expect("finish vector builder");
    let json = r#"[{"column":"v","dim":16,"n_cent":3,"rot_seed":42,"metric":"l2sq"}]"#;
    let r = VectorReader::open(Bytes::from(bytes), json).expect("open VectorReader");

    // Query at exactly the first cluster's center → top-k should all
    // come from cluster 0.
    let q = centers[0].to_vec();
    let hits = r
        .search(
            "v",
            &q,
            VECTOR_PIPELINE_HEADROOM_K,
            CLUSTER_TEST_N_CENT,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    assert!(!hits.is_empty());
    for (doc, _) in &hits {
        assert_eq!(
            planted_cluster[*doc as usize], 0,
            "top-k for query at cluster-0 center should be from cluster 0; doc {} in cluster {}",
            doc, planted_cluster[*doc as usize]
        );
    }
}

#[tokio::test]
async fn end_to_end_results_sorted_by_distance() {
    let (blob, json) = build_two_column_blob(ROUTING_TEST_N_DOCS);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let q = vec![0.5; TEXT_EMB_DIM];
    let hits = r
        .search(
            "text_emb",
            &q,
            VECTOR_PIPELINE_HEADROOM_K,
            VECTOR_PIPELINE_NPROBE,
            VECTOR_PIPELINE_RERANK_MULT,
        )
        .await
        .expect("FTS search");
    for w in hits.windows(2) {
        assert!(w[0].1 <= w[1].1, "distances ascending");
    }
}
