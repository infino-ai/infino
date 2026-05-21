//! Shared bench fixtures: deterministic corpora, query batteries,
//! brute-force ground truth, recall calibration, and thin builder
//! wrappers around infino's public API.
//!
//! `infino/benches/` consumes these directly. Centralizing the
//! generators here means a single deterministic source of truth for
//! the corpus, queries, and ground truth — without that, every
//! re-run would silently risk mixing measurements against drifted
//! data.
//!
//! ## Scale knob
//!
//! [`n_docs`] returns `10_000_000` when `INFINO_BENCH_FULL=1` is set
//! in the environment, otherwise `1_000_000`. Vector at 10M × 384 (f32)
//! = 14.6 GB resident — needs a 32 GB+ machine. Superfile-shape benches
//! pin themselves to 1M (one-segment scale); supertable-shape benches
//! pin to 10M (sharding scale). Each topic's bench files name the
//! pinned constant directly rather than calling this helper at
//! benchmark time.

#![allow(clippy::too_many_arguments)]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{LargeStringArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use memmap2::Mmap;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use tempfile::TempDir;

use infino::superfile::SuperfileReader;
use infino::superfile::builder::{
    BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig as SfVectorConfig,
};
use infino::superfile::fts::builder::FtsBuilder;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, normalize};
use infino::superfile::vector::reader::VectorReader;
use infino::test_helpers::default_tokenizer;

// ─── Scale constants ──────────────────────────────────────────────────

/// Tokens per doc — chosen to land in the same magnitude as a typical
/// short article (~200 words). The product `n_docs * tokens_per_doc`
/// drives FTS posting volume.
pub const TOKENS_PER_DOC: usize = 200;

/// Vocabulary size — controls term-frequency distribution. Small
/// enough that common terms appear in many docs (exercising long
/// posting lists); large enough that rare terms exist (exercising the
/// FST + skip-table cold path).
pub const VOCAB_SIZE: usize = 10_000;

/// Vector dimension — matches modern sentence-embedding models
/// (all-MiniLM-L6-v2 = 384, BGE-small = 384).
pub const DIM: usize = 384;

/// One `(local_doc_id, distance)` hit — same shape `VectorReader::search`
/// returns. Re-exported here so recall helpers stay engine-agnostic.
pub type Hit = (u32, f32);

/// Resolved doc count for the current bench run, gated by the
/// `INFINO_BENCH_FULL` env var (10M when set, 1M otherwise).
pub fn n_docs() -> usize {
    if std::env::var("INFINO_BENCH_FULL").is_ok() {
        10_000_000
    } else {
        1_000_000
    }
}

/// IVF cluster count. Conventionally `~sqrt(n_docs)`, snapped to a
/// fixed value per scale band so 1M and 10M runs share a stable
/// `n_cent`.
pub fn n_cent(n_docs: usize) -> usize {
    if n_docs >= 5_000_000 {
        4096
    } else if n_docs >= 100_000 {
        1024
    } else {
        64
    }
}

// ─── Text corpus ──────────────────────────────────────────────────────

/// Deterministic Zipfian sampler over `[1, n]`. Inverse-CDF; O(log n)
/// per draw. Avoids `rand_distr::Zipf`'s f64-parameter overhead.
pub struct ZipfDistribution {
    /// Cumulative `1/i` weights up to rank `n`. Index 0 == rank 1.
    cum_weights: Vec<f64>,
}

impl ZipfDistribution {
    pub fn new(n: usize) -> Self {
        let mut cum = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for i in 1..=n {
            acc += 1.0 / (i as f64);
            cum.push(acc);
        }
        Self { cum_weights: cum }
    }

    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> usize {
        use rand::RngExt;
        let total = *self.cum_weights.last().expect("non-empty");
        let target: f64 = rng.random::<f64>() * total;
        match self
            .cum_weights
            .binary_search_by(|p| p.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) | Err(i) => i.min(self.cum_weights.len() - 1) + 1,
        }
    }
}

/// Generate a Zipfian token corpus. Returns `n_docs` strings, each
/// `TOKENS_PER_DOC` body tokens drawn from a closed [`VOCAB_SIZE`]
/// vocabulary prefixed by one doc-unique identifier token
/// (`doc<7-digit-id>`).
///
/// The closed-vocab body alone has no singletons — the rarest body
/// term still has df ≈ N / (V · H_V) ≈ 2000 at 1M docs × 200 tokens ×
/// 10K vocab — which underexercises the format's `df=1` paths (per-term
/// metadata, BMW upper bound on one-doc terms, the inline-encoding
/// short-circuit). The per-doc identifier creates a singleton long
/// tail proportional to `n_docs`, matching production text where every
/// real doc carries some unique token (URL hash, ISBN, headline number).
pub fn generate_text_corpus(n_docs: usize, seed: u64) -> Vec<String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let zipf = ZipfDistribution::new(VOCAB_SIZE);
    let mut out = Vec::with_capacity(n_docs);
    for doc_id in 0..n_docs {
        let mut doc = String::with_capacity((TOKENS_PER_DOC + 1) * 8);
        doc.push_str(&format!("doc{doc_id:07}"));
        for _ in 0..TOKENS_PER_DOC {
            let idx = zipf.sample(&mut rng);
            doc.push(' ');
            doc.push_str(&format!("term{idx:05}"));
        }
        out.push(doc);
    }
    out
}

// ─── Vector corpus ────────────────────────────────────────────────────

/// Generate `n_docs` planted-cluster vectors of [`DIM`] dimensions,
/// optionally per-doc normalized for cosine. `n_cent` planted centers
/// drawn from `3·N(0, 1)` per dim; each doc lives near a center with
/// `sigma = 0.3` per-dim Gaussian noise.
///
/// **Centers are intentionally NOT normalized.** At `DIM=384` the
/// un-normalized center magnitude is ~58 and per-doc noise norm is
/// ~5.9 (about 10% of center magnitude), so docs sit tightly around
/// their planted center direction. If centers were unit-normalized
/// first, the same noise would dominate (`||noise|| ≈ 5.9 ≫ 1`) and
/// per-doc normalization would destroy the cluster signal entirely —
/// IVF + RaBitQ trained on that data can't recover any meaningful
/// cluster structure even at full sweep + maximal rerank.
pub fn generate_vector_corpus(
    n_docs: usize,
    n_cent: usize,
    seed: u64,
    normalize_each: bool,
) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;

    let centers: Vec<Vec<f32>> = (0..n_cent)
        .map(|_| {
            (0..DIM)
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * 3.0
                })
                .collect()
        })
        .collect();

    let mut out: Vec<f32> = Vec::with_capacity(n_docs * DIM);
    for i in 0..n_docs {
        let center = &centers[i % n_cent];
        let mut v: Vec<f32> = center
            .iter()
            .map(|&c| {
                let s: f64 = dist.sample(&mut rng);
                c + (s as f32) * 0.3
            })
            .collect();
        if normalize_each {
            normalize(&mut v);
        }
        out.extend_from_slice(&v);
    }
    out
}

/// Disk-backed raw vector corpus for the large vector benches.
///
/// At 10M x 384, storing the corpus as a `Vec<f32>` pins about 14.6 GiB
/// of anonymous RAM before the builder under test starts. The mmap-backed
/// path keeps the same `&[f32]` call sites while letting the kernel reclaim
/// corpus pages as page cache under pressure.
///
/// This is not an alternate Infino ingestion path. It is only the raw input
/// fixture: benches still build Arrow arrays, call `SupertableWriter::append`,
/// and commit through the same path production callers use. The mmap lets
/// ingestion, query generation, and brute-force recall share one deterministic
/// corpus without keeping the whole corpus on the heap.
pub struct MmapCorpus {
    _tmp: TempDir,
    map: Mmap,
    n_docs: usize,
    dim: usize,
}

impl MmapCorpus {
    pub fn generate(n_docs: usize, n_cent: usize, seed: u64, normalize_each: bool) -> Self {
        let tmp = TempDir::new().expect("create MmapCorpus tempdir");
        let path = tmp.path().join("corpus.bin");
        let file = File::create(&path).expect("create corpus file");
        let mut writer = BufWriter::with_capacity(8 << 20, file);
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = StandardNormal;
        let centers: Vec<Vec<f32>> = (0..n_cent)
            .map(|_| {
                (0..DIM)
                    .map(|_| {
                        let s: f64 = dist.sample(&mut rng);
                        (s as f32) * 3.0
                    })
                    .collect()
            })
            .collect();
        let mut row = vec![0.0f32; DIM];
        for i in 0..n_docs {
            let center = &centers[i % n_cent];
            for (j, slot) in row.iter_mut().enumerate() {
                let s: f64 = dist.sample(&mut rng);
                *slot = center[j] + (s as f32) * 0.3;
            }
            if normalize_each {
                normalize(&mut row);
            }
            writer
                .write_all(bytemuck::cast_slice(&row))
                .expect("write corpus row");
        }
        let file = writer.into_inner().expect("flush corpus writer");
        file.sync_all().expect("sync corpus");
        drop(file);

        let file = File::open(&path).expect("reopen corpus");
        // SAFETY: this helper owns the temp file and never writes to it after
        // the fsync above, so the read-only mmap cannot observe mutation.
        let map = unsafe { Mmap::map(&file).expect("mmap corpus") };
        Self {
            _tmp: tmp,
            map,
            n_docs,
            dim: DIM,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(&self.map)
    }

    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

// ─── Query batteries ──────────────────────────────────────────────────

/// `n_queries` deterministic Gaussian queries (no corpus dependency),
/// normalized. Useful only for smoke wiring — real benches should use
/// [`generate_realistic_queries`] so recall is meaningful at modest
/// nprobe.
pub fn generate_queries(n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    (0..n_queries)
        .map(|_| {
            let mut q: Vec<f32> = (0..DIM)
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * 3.0
                })
                .collect();
            normalize(&mut q);
            q
        })
        .collect()
}

/// Pick `n_queries` corpus members and perturb each by small Gaussian
/// noise. A pure-Gaussian query lands far from any doc in embedding
/// space, so the top-10 NN are spread across many planted clusters and
/// IVF recall stays low even at high nprobe. Perturbed corpus members
/// match the same workload `tests/recall.rs` uses.
pub fn generate_realistic_queries(
    vectors: &[f32],
    n_docs: usize,
    n_queries: usize,
    seed: u64,
    normalize_each: bool,
    sigma: f32,
) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = StandardNormal;
    let mut out = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        // Coprime stride so consecutive queries don't all sit in the
        // first planted cluster.
        let base_idx = (i * 7919) % n_docs;
        let off = base_idx * DIM;
        let mut q: Vec<f32> = (0..DIM)
            .map(|d| {
                let s: f64 = dist.sample(&mut rng);
                vectors[off + d] + (s as f32) * sigma
            })
            .collect();
        if normalize_each {
            normalize(&mut q);
        }
        out.push(q);
    }
    out
}

// ─── Brute-force ground truth + recall ────────────────────────────────

/// Brute-force kNN ground truth for cosine distance on L2-normalized
/// vectors. Returns top-k local doc_ids (no distances — recall only
/// needs the id set).
pub fn brute_force_topk_cosine(
    vectors: &[f32],
    n_docs: usize,
    query: &[f32],
    k: usize,
) -> Vec<u32> {
    assert_eq!(vectors.len(), n_docs * DIM);
    assert_eq!(query.len(), DIM);
    // For L2-normalized inputs cosine distance is monotone in -dot.
    let mut scored: Vec<(u32, f32)> = (0..n_docs as u32)
        .map(|i| {
            let off = (i as usize) * DIM;
            let mut dot = 0f32;
            for d in 0..DIM {
                dot += vectors[off + d] * query[d];
            }
            (i, -dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Brute-force top-k per query for a whole batch.
pub fn ground_truth(
    vectors: &[f32],
    n_docs: usize,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| brute_force_topk_cosine(vectors, n_docs, q, k))
        .collect()
}

/// Recall@k between a predicted top-k id list and ground truth.
pub fn recall_at_k(predicted: &[Hit], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return 1.0;
    }
    let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
    let hits = predicted
        .iter()
        .filter(|(id, _)| truth_set.contains(id))
        .count();
    hits as f32 / truth.len() as f32
}

/// Mean recall for one (engine, config) point across a query batch.
pub fn mean_recall_infino(
    reader: &VectorReader,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let mut sum = 0f32;
    for (q, t) in queries.iter().zip(truths) {
        let hits = reader
            .search("v", q, k, nprobe, rerank_mult)
            .expect("vector search");
        sum += recall_at_k(&hits, t);
    }
    sum / queries.len() as f32
}

// ─── Recall-floor calibration ─────────────────────────────────────────

/// p50 wall time (microseconds) over `n_iter` repetitions of one closure.
/// Generic over `FnMut()` so calibration can wrap any search call
/// with one timing implementation.
pub fn p50_micros<F: FnMut()>(mut f: F, n_iter: usize) -> f32 {
    let mut samples = Vec::with_capacity(n_iter);
    for _ in 0..n_iter {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f32() * 1e6);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).expect("partial_cmp"));
    samples[samples.len() / 2]
}

/// Calibration result for one engine at one recall target.
#[derive(Debug, Clone, Copy)]
pub struct Calibrated {
    pub probe: usize,
    pub refine: usize,
    pub recall: f32,
    pub p50_micros: f32,
}

/// Sweep a `(probe, refine)` grid for infino; return the lowest-p50
/// point that hits `recall ≥ target_recall`. `None` if no grid point
/// meets the target.
pub fn calibrate_infino(
    reader: &VectorReader,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    target_recall: f32,
    probes: &[usize],
    refines: &[usize],
    p50_iter: usize,
    k: usize,
) -> Option<Calibrated> {
    let mut best: Option<Calibrated> = None;
    let mut peak_recall = 0f32;
    for &probe in probes {
        for &refine in refines {
            let recall = mean_recall_infino(reader, queries, truths, k, probe, refine);
            if recall > peak_recall {
                peak_recall = recall;
            }
            if recall < target_recall {
                continue;
            }
            // Single-query timing fixture; Gaussian queries are
            // statistically interchangeable so p50 over n_iter on one
            // query approximates the mean shape across the battery.
            let q = &queries[0];
            let p50 = p50_micros(
                || {
                    let _ = reader.search("v", q, k, probe, refine).expect("search");
                },
                p50_iter,
            );
            let cand = Calibrated {
                probe,
                refine,
                recall,
                p50_micros: p50,
            };
            best = match best {
                None => Some(cand),
                Some(b) if cand.p50_micros < b.p50_micros => Some(cand),
                Some(b) => Some(b),
            };
        }
    }
    if best.is_none() {
        eprintln!(
            "    [infino] no point hit recall ≥ {target_recall:.2}; peak observed = {peak_recall:.3}"
        );
    }
    best
}

// ─── Thin builder wrappers ────────────────────────────────────────────

/// Build a stand-alone FTS index from a token corpus. Wrapper exists so
/// both bench harnesses construct the index identically.
pub fn build_fts_index(docs: &[String]) -> FtsBuilder {
    let mut b = FtsBuilder::new(default_tokenizer());
    b.register_column("title".to_string())
        .expect("register column");
    for (i, text) in docs.iter().enumerate() {
        b.add_doc(0, i as u32, text).expect("add doc");
    }
    b
}

/// Build a stand-alone vector index. `vectors` is flat `n_docs * DIM`.
pub fn build_vector_index(
    vectors: &[f32],
    n_docs: usize,
    n_cent: usize,
    metric: Metric,
) -> VectorBuilder {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        name: "v".into(),
        dim: DIM,
        n_cent,
        rot_seed: 7,
        metric,
    })
    .expect("register column");
    for i in 0..n_docs {
        let off = i * DIM;
        b.add(0, &vectors[off..off + DIM])
            .expect("add to vector builder");
    }
    b
}

/// Open a built vector blob as a reader. Encodes the directory JSON
/// inline so callers don't reinvent it.
pub fn open_vector_reader(blob: Vec<u8>, n_cent: usize, metric: Metric) -> VectorReader {
    let metric_str = match metric {
        Metric::L2Sq => "l2sq",
        Metric::Cosine => "cosine",
        Metric::NegDot => "negdot",
    };
    let json = format!(
        r#"[{{"name":"v","dim":{DIM},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_str}"}}]"#
    );
    VectorReader::open(Bytes::from(blob), &json).expect("open VectorReader")
}

/// Build a full superfile (FTS + vector) for end-to-end benches.
pub fn build_superfile(docs: &[String], vectors: &[f32], n_cent: usize) -> Vec<u8> {
    let n = docs.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::UInt64, false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent,
            rot_seed: 7,
            metric: Metric::Cosine,
        }],
        Some(default_tokenizer()),
    );

    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = UInt64Array::from((0..n as u64).collect::<Vec<_>>());
    let titles = LargeStringArray::from(docs.iter().map(String::as_str).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[vectors]).expect("add_batch");
    b.finish().expect("finish builder")
}

/// Open a finished superfile blob.
pub fn open_superfile(bytes: Vec<u8>) -> SuperfileReader {
    SuperfileReader::open(Bytes::from(bytes)).expect("open superfile")
}
