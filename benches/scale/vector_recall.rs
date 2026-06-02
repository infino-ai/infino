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
use std::sync::Arc;

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::VectorSearchOptions;
use infino::superfile::builder::{BuilderOptions, SuperfileBuilder, VectorConfig};
use infino::superfile::reader::SuperfileReader;
use infino::superfile::vector::distance::{Metric, distance, normalize};
use infino::superfile::vector::rerank_codec::RerankCodec;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

const N_DOCS: usize = 10_000;
const DIM: usize = 384;
const N_CENT: usize = 64;
const N_QUERIES: usize = 50;

fn search_blocking(
    reader: &SuperfileReader,
    query: &[f32],
    k: usize,
    opts: VectorSearchOptions,
) -> Vec<(u32, f32)> {
    futures::executor::block_on(reader.vector_search("emb", query, k, opts)).expect("vector_search")
}

fn generate_planted_corpus(seed: u64, normalize_each: bool) -> Vec<Vec<f32>> {
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

fn build_superfile_reader(corpus: &[Vec<f32>], metric: Metric) -> SuperfileReader {
    let _metric_str = match metric {
        Metric::L2Sq => "l2sq",
        Metric::Cosine => "cosine",
        Metric::NegDot => "negdot",
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "_id",
        vec![],
        vec![VectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent: N_CENT,
            rot_seed: 7,
            metric,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let flat: Vec<f32> = corpus.iter().flatten().copied().collect();
    let ids = arrow_array::Decimal128Array::from((0..corpus.len() as i128).collect::<Vec<_>>())
        .with_precision_and_scale(38, 0)
        .expect("decimal128");
    let titles = LargeStringArray::from(
        (0..corpus.len())
            .map(|i| format!("doc {i}"))
            .collect::<Vec<_>>(),
    );
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");

    let bytes = Bytes::from(b.finish().expect("finish builder"));
    SuperfileReader::open(bytes).expect("open SuperfileReader")
}

fn measure_recall(
    reader: &SuperfileReader,
    corpus: &[Vec<f32>],
    metric: Metric,
    queries: &[Vec<f32>],
    k: usize,
    nprobe: usize,
) -> f32 {
    let opts = VectorSearchOptions::new().with_nprobe(nprobe);
    let mut total: f32 = 0.0;
    for q in queries {
        let truth: HashSet<u32> = brute_force_top_k(corpus, q, metric, k)
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

fn build_query_set(corpus: &[Vec<f32>], seed: u64, normalize_each: bool) -> Vec<Vec<f32>> {
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
    let reader = build_superfile_reader(&corpus, Metric::L2Sq);
    let queries = build_query_set(&corpus, 100, false);

    let r10 = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, 8);
    assert!(
        r10 >= 0.90,
        "L2Sq recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, 32);
    assert!(
        r10_high >= 0.95,
        "L2Sq recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    let r1 = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 1, 8);
    assert!(
        r1 >= 0.95,
        "L2Sq recall@1 at nprobe=8 below threshold: {r1:.3} < 0.95"
    );

    println!(
        "L2Sq @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}; recall@1/nprobe=8 = {r1:.3}"
    );
}

fn recall_cosine_at_10k_dim384_meets_threshold() {
    let corpus = generate_planted_corpus(2, true);
    let reader = build_superfile_reader(&corpus, Metric::Cosine);
    let queries = build_query_set(&corpus, 200, true);

    let r10 = measure_recall(&reader, &corpus, Metric::Cosine, &queries, 10, 8);
    assert!(
        r10 >= 0.90,
        "Cosine recall@10 at nprobe=8 below threshold: {r10:.3} < 0.90"
    );

    let r10_high = measure_recall(&reader, &corpus, Metric::Cosine, &queries, 10, 32);
    assert!(
        r10_high >= 0.95,
        "Cosine recall@10 at nprobe=32 below threshold: {r10_high:.3} < 0.95"
    );

    println!("Cosine @10k×384: recall@10/nprobe=8 = {r10:.3}; recall@10/nprobe=32 = {r10_high:.3}");
}

fn recall_increases_monotonically_with_nprobe() {
    let corpus = generate_planted_corpus(3, false);
    let reader = build_superfile_reader(&corpus, Metric::L2Sq);
    let queries = build_query_set(&corpus, 300, false);

    let mut prev: f32 = -1.0;
    for &nprobe in &[1, 2, 4, 8, 16, 32, 64] {
        let r = measure_recall(&reader, &corpus, Metric::L2Sq, &queries, 10, nprobe);
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
