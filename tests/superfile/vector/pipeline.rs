//! End-to-end vector kNN pipeline integration test.
//!
//! Builds a multi-column vector blob, opens it, exercises kNN search.
//! Mirrors the planted-ground-truth correctness pattern.

use bytes::Bytes;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, normalize};
use infino::superfile::vector::reader::VectorReader;
use infino::superfile::vector::rerank_codec::RerankCodec;

/// Build a 2-column vector blob: text_emb (dim=16, cosine) and
/// image_emb (dim=24, l2sq), each with `n_docs` deterministic vectors.
fn build_two_column_blob(n_docs: u32) -> (Bytes, String) {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "text_emb".into(),
        dim: 16,
        n_cent: 4,
        rot_seed: 11,
        metric: Metric::Cosine,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");
    b.register_column(VectorConfig {
        column: "image_emb".into(),
        dim: 24,
        n_cent: 4,
        rot_seed: 22,
        metric: Metric::L2Sq,
        rerank_codec: RerankCodec::Fp32,
    })
    .expect("register column");

    for i in 0..n_docs {
        // Deterministic per-doc vectors with simple structure so we can
        // make planted-ground-truth assertions.
        let mut v_text: Vec<f32> = (0..16)
            .map(|j| ((i.wrapping_mul(31) + j as u32 * 3) % 100) as f32 * 0.01 + 0.1)
            .collect();
        // Cosine metric requires unit-norm inputs.
        normalize(&mut v_text);
        let v_img: Vec<f32> = (0..24)
            .map(|j| ((i.wrapping_mul(17) + j as u32 * 7) % 100) as f32 * 0.01)
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
    let n_docs = 80u32;
    let (blob, json) = build_two_column_blob(n_docs);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    assert_eq!(r.n_docs(), n_docs as u64);

    // Reconstruct doc 17's (normalized) text_emb vector for the query.
    let target = 17u32;
    let mut q_text: Vec<f32> = (0..16)
        .map(|j| ((target.wrapping_mul(31) + j as u32 * 3) % 100) as f32 * 0.01 + 0.1)
        .collect();
    normalize(&mut q_text);
    let hits = r.search("text_emb", &q_text, 5, 4, 5).expect("FTS search");
    assert_eq!(hits[0].0, target, "self should be top-1");
    // Cosine distance to self for unit-norm vector = 1 - 1 = 0.
    assert!(
        hits[0].1 < 1e-3,
        "cosine self-distance should be ~0, got {}",
        hits[0].1
    );
}

#[tokio::test]
async fn end_to_end_l2sq_self_query_distance_is_zero() {
    let (blob, json) = build_two_column_blob(80);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let target = 5u32;
    let q_img: Vec<f32> = (0..24)
        .map(|j| ((target.wrapping_mul(17) + j as u32 * 7) % 100) as f32 * 0.01)
        .collect();
    let hits = r.search("image_emb", &q_img, 3, 4, 5).expect("FTS search");
    assert_eq!(hits[0].0, target);
    // L2² of v with itself is exactly 0.
    assert!(hits[0].1 < 1e-3, "self L2² should be ~0, got {}", hits[0].1);
}

#[tokio::test]
async fn end_to_end_multi_column_routing_isolated() {
    let (blob, json) = build_two_column_blob(60);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");

    // text_emb is dim=16; querying with a dim=24 image vector must error.
    let v_img: Vec<f32> = vec![0.5; 24];
    let err = r.search("text_emb", &v_img, 5, 4, 5);
    assert!(err.is_err(), "dim mismatch must error");

    // And vice versa.
    let v_text: Vec<f32> = vec![0.5; 16];
    let err = r.search("image_emb", &v_text, 5, 4, 5);
    assert!(err.is_err());
}

#[tokio::test]
async fn end_to_end_top_k_limits_results() {
    let (blob, json) = build_two_column_blob(80);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let q: Vec<f32> = vec![0.3; 16];
    let hits = r.search("text_emb", &q, 3, 4, 5).expect("FTS search");
    assert!(hits.len() <= 3);
}

#[test]
fn end_to_end_summary_per_column() {
    let (blob, json) = build_two_column_blob(40);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");

    let (text_centroid, text_radius) = r.summary("text_emb").expect("vector summary");
    assert_eq!(text_centroid.len(), 16);
    assert!(text_radius >= 0.0);

    let (img_centroid, img_radius) = r.summary("image_emb").expect("vector summary");
    assert_eq!(img_centroid.len(), 24);
    assert!(img_radius >= 0.0);

    // Different columns should have different summary centroids
    // (different data, different dim). Just sanity-check shapes.
    assert!(r.summary("nonexistent").is_none());
}

#[tokio::test]
async fn end_to_end_planted_clusters_recovered() {
    // Plant 3 well-separated clusters in dim=16; verify nearest-neighbor
    // for a query at one center pulls back docs from that cluster.
    let dim = 16;
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim,
        n_cent: 3,
        rot_seed: 42,
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
        for d in 0..20 {
            let mut v = c.to_vec();
            // Tiny per-doc noise so docs aren't identical.
            for (j, slot) in v.iter_mut().enumerate() {
                *slot += ((cluster_idx * 20 + d + j) % 7) as f32 * 0.01;
            }
            b.add(0, &v).expect("add to vector builder");
            planted_cluster.push(cluster_idx as u32);
            next_doc_id += 1;
        }
    }
    assert_eq!(next_doc_id, 60);

    let bytes = b.finish().expect("finish vector builder");
    let json = r#"[{"column":"v","dim":16,"n_cent":3,"rot_seed":42,"metric":"l2sq"}]"#;
    let r = VectorReader::open(Bytes::from(bytes), json).expect("open VectorReader");

    // Query at exactly the first cluster's center → top-k should all
    // come from cluster 0.
    let q = centers[0].to_vec();
    let hits = r.search("v", &q, 10, 3, 5).expect("FTS search");
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
    let (blob, json) = build_two_column_blob(60);
    let r = VectorReader::open(blob, &json).expect("open VectorReader");
    let q = vec![0.5; 16];
    let hits = r.search("text_emb", &q, 10, 4, 5).expect("FTS search");
    for w in hits.windows(2) {
        assert!(w[0].1 <= w[1].1, "distances ascending");
    }
}
