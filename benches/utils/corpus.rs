// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

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
//! ## Scale policy
//!
//! Scale is fixed by *shape*, not by an environment variable:
//! superfile-shape benches use [`SUPERFILE_DOCS`] (1M, one-superfile
//! scale), supertable-shape benches use [`SUPERTABLE_DOCS`] (10M,
//! sharding scale). Vector at 10M × 384 (f32) = 14.6 GB resident —
//! needs a 32 GB+ machine. There is deliberately no `INFINO_BENCH_FULL`
//! knob: a bench's scale is a property of the shape it measures, so it
//! lives in a `const` next to that bench, not behind an env toggle that
//! silently means different things in different files.

#![allow(clippy::too_many_arguments)]

use std::{
    cmp::Ordering,
    env,
    fs::File,
    io::{BufWriter, Error, ErrorKind, Result as IoResult, Write},
    mem::size_of,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
    time::{Duration, Instant},
};

use arrow_array::{
    Array, Decimal128Array, FixedSizeListArray, Float32Array, Float64Array, LargeListArray,
    LargeStringArray, ListArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    roaring::RoaringBitmap,
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig as SfVectorConfig},
        fts::builder::FtsBuilder,
        reader::VectorSearchOptions,
        vector::{
            builder::{VectorBuilder, VectorConfig},
            distance::{Metric, distance, normalize},
            reader::{OpenOptions, VectorReader},
            rerank_codec::RerankCodec,
        },
    },
    test_helpers::default_tokenizer,
};
use memmap2::Mmap;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;
use tempfile::TempDir;

use crate::rss;

// ─── Async bridge for in-memory bench helpers ─────────────────────────

/// Drive an in-memory (no object-store I/O) async search to
/// completion from sync bench code.
///
/// The query/search API is `async`. In-memory `VectorReader` /
/// `FtsReader` / `Supertable` readers never touch the object store, so
/// their futures resolve without object-store I/O — but the search
/// kernels bridge their CPU scans onto the rayon pool and `await` a
/// oneshot for the result, which requires an ambient tokio runtime. So
/// bench helpers drive them on one shared multi-thread runtime (built
/// once, reused across every call — never a throwaway per-call
/// runtime).
pub fn block_on_inmem<F: std::future::Future>(fut: F) -> F::Output {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let rt = RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("bench in-memory tokio runtime")
    });
    rt.block_on(fut)
}

// ─── Scale constants ──────────────────────────────────────────────────

/// Codec benches build vector columns with. Mirrors the engine default
/// (`default_rerank_codec_for`): the fixed cosine grid (`Sq16`) for cosine, and
/// the per-cluster-fitted 16-bit `Sq16Adaptive` for the unbounded metrics
/// (L2Sq / NegDot). Codec choice is engine configuration
/// (`vector.rerank_codec` in YAML), not a bench env knob.
pub fn bench_rerank_codec(metric: Metric) -> RerankCodec {
    // Diagnostic (env-gated): force a codec to isolate its recall/latency from
    // routing/coverage, or to A/B the new default against the prior one.
    // INFINO_BENCH_RERANK_CODEC=fp32 → if the recall ceiling jumps to ~1.0, the
    // gap was quantization; =sq8_residual → A/B the pre-Sq16Adaptive default.
    let codec = match std::env::var("INFINO_BENCH_RERANK_CODEC").ok().as_deref() {
        Some("fp32") => RerankCodec::Fp32,
        Some("sq16") => RerankCodec::Sq16,
        Some("sq16_adaptive") => RerankCodec::Sq16Adaptive,
        Some("sq8_residual") => RerankCodec::Sq8Residual,
        Some("sq8_fixed_residual") => RerankCodec::Sq8FixedResidual,
        _ if metric == Metric::Cosine => RerankCodec::default(),
        _ => RerankCodec::Sq16Adaptive,
    };
    assert!(
        codec.supports_metric(metric),
        "codec {} does not support metric {metric:?}",
        codec.name()
    );
    codec
}

/// Tokens per doc — chosen to land in the same magnitude as a typical
/// short article (~200 words). The product `n_docs * tokens_per_doc`
/// drives FTS posting volume.
pub const TOKENS_PER_DOC: usize = 200;

/// Vocabulary size — controls term-frequency distribution. Small
/// enough that common terms appear in many docs (exercising long
/// posting lists); large enough that rare terms exist (exercising the
/// FST + skip-table cold path).
pub const VOCAB_SIZE: usize = 10_000;

// ─── Parallel corpus-generation constants ─────────────────────────────
//
// The text corpus has a deterministic, RNG-independent byte layout, which
// lets the offset table be computed up front and each chunk write to its
// own disjoint byte range in parallel. These name the pieces of that
// layout so the offset math carries no bare literals.

/// Bytes in the per-doc `"doc"` prefix of `"doc{doc_id:07}"`.
const DOC_ID_PREFIX_BYTES: usize = 3;
/// Minimum digit width the doc id is zero-padded to (`"{:07}"`). A doc id
/// with more digits than this widens the doc token by exactly that many
/// bytes; fewer digits still occupy this width.
const DOC_ID_PAD_WIDTH: usize = 7;
/// Bytes in one body token `" term{:05}"` — 5 for `" term"` plus 5 digits.
/// Fixed because `VOCAB_SIZE = 10_000 ≤ 100_000` keeps the index ≤ 5 digits.
const TERM_BYTES: usize = 10;
/// Docs assigned to one text corpus scheduling chunk. Each worker streams its
/// chunk through [`PARALLEL_CORPUS_WRITE_BUF_CAPACITY`] rather than buffering
/// the whole chunk in memory. `pub(crate)`: the sequential stream
/// ([`combined::SequentialSyntheticCorpus`]) reseeds at these boundaries to
/// stay bit-identical with the parallel writer.
pub(crate) const TEXT_CORPUS_CHUNK_DOCS: usize = 1 << 20;
/// Docs assigned to one vector corpus scheduling chunk. Each worker streams
/// its chunk through [`PARALLEL_CORPUS_WRITE_BUF_CAPACITY`] rather than
/// buffering the whole chunk in memory. `pub(crate)` for the same
/// boundary-reseed reason as [`TEXT_CORPUS_CHUNK_DOCS`].
pub(crate) const VECTOR_CORPUS_CHUNK_DOCS: usize = 1 << 19;
/// Per-worker output buffer before a positioned write flushes to the corpus
/// file. This keeps memory bounded by roughly `rayon_threads × 8 MiB` while
/// still issuing large writes to NVMe.
const PARALLEL_CORPUS_WRITE_BUF_CAPACITY: usize = 8 << 20;
/// Odd 64-bit multiplier (fractional golden ratio) used to derive a
/// distinct, deterministic RNG seed per chunk from `(seed, chunk_index)`,
/// so a parallelized corpus is still reproducible for a given `seed`.
const CHUNK_SEED_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// Deterministic per-chunk RNG seed: mixes the base `seed` with the chunk
/// index so independent chunks draw disjoint, reproducible streams.
/// `pub(crate)`: the sequential stream reseeds with the same function so
/// streamed docs match the parallel-written corpus bytes.
pub(crate) fn chunk_seed(seed: u64, chunk_index: usize) -> u64 {
    seed.wrapping_add(
        (chunk_index as u64)
            .wrapping_add(1)
            .wrapping_mul(CHUNK_SEED_MIX),
    )
}

/// Synthetic-corpus vector dimension — matches modern large embedding
/// models (OpenAI text-embedding-3-small = 1536 truncates to 1024,
/// BGE-large = 1024, E5-large = 1024). Runtime code reads [`dim`], which
/// resolves to this for the synthetic corpus and to the dataset's own
/// dimension for a real corpus.
pub const SYNTHETIC_DIM: usize = 1024;

/// Vector corpus source, selected by the `corpus=<spec>` bench argument
/// (see [`set_source`]). Deliberately NOT an env var: the bench tier's
/// only env-tunable knobs are the two doc counts, and a corpus that can
/// silently change from the environment is exactly the class of drift
/// that makes two runs incomparable.
///
/// Specs:
///
/// * `synthetic` — the seeded planted-cluster generator (the default).
/// * `annb:<slug>` — a published ann-benchmarks dataset
///   (`annb:glove-100-angular`). One HDF5 file carries corpus rows, the
///   official query set, and official top-k neighbour ids, so ground
///   truth needs no recompute and recall is comparable to published
///   numbers.
/// * `hf:<repo>` — a Hugging Face parquet dataset
///   (`hf:KShivendu/dbpedia-entities-openai-1M`). Real embeddings at
///   scale (that one is 1M x 1536) and, when the repo carries text
///   columns, the only source that can feed FTS and SQL as well as
///   vectors. No shipped ground truth: truth is computed over the rows
///   actually ingested.
/// * `parquet:<dir>` — every `*.parquet` in a local directory, read in
///   sorted name order. No download and no naming convention: a directory
///   of shards from any producer works, including one a `hf:` run already
///   staged.
///
/// Real corpora are vector-only today except where a source carries text
/// columns; FTS / SQL / combined modalities otherwise keep the synthetic
/// corpus and say so rather than silently mixing corpora.
#[derive(Debug, Clone)]
pub enum CorpusSource {
    /// Seeded planted-cluster generator (the historical default).
    Synthetic,
    /// Published ann-benchmarks dataset `slug`, staged under `dir`.
    AnnBenchmarks { dir: PathBuf, slug: String },
    /// Hugging Face parquet dataset `repo`, staged under `dir`.
    HuggingFace { dir: PathBuf, repo: String },
    /// Local directory of parquet shards.
    LocalParquet { dir: PathBuf },
}

/// Where downloaded corpora are staged, set alongside the spec.
static SOURCE: OnceLock<CorpusSource> = OnceLock::new();

/// Parse and install the corpus source for this process. First call wins,
/// mirroring [`crate::dataset::set_prefix`]. `dir` is where downloadable
/// sources stage their files. Returns an error string for an unparseable
/// spec so the caller can fail with usage rather than panicking deep in a
/// build.
pub fn set_source(spec: &str, dir: Option<&str>) -> Result<(), String> {
    let staged = |what: &str| -> Result<PathBuf, String> {
        dir.map(PathBuf::from)
            .ok_or_else(|| format!("corpus={spec} needs corpus-dir=<path> to stage {what}"))
    };
    let source = match spec.split_once(':') {
        None if spec == "synthetic" => CorpusSource::Synthetic,
        Some(("annb", slug)) if !slug.is_empty() => CorpusSource::AnnBenchmarks {
            dir: staged("the dataset file")?,
            slug: slug.to_string(),
        },
        Some(("hf", repo)) if repo.contains('/') => CorpusSource::HuggingFace {
            dir: staged("the dataset shards")?,
            repo: repo.to_string(),
        },
        Some(("parquet", d)) if !d.is_empty() => CorpusSource::LocalParquet {
            dir: PathBuf::from(d),
        },
        _ => {
            return Err(format!(
                "unknown corpus spec {spec:?} (expected synthetic | annb:<slug> | \
                 hf:<owner/repo> | parquet:<dir>)"
            ));
        }
    };
    let _ = SOURCE.set(source);
    Ok(())
}

/// The process-wide corpus source; [`CorpusSource::Synthetic`] until set.
pub fn corpus_source() -> &'static CorpusSource {
    SOURCE.get_or_init(|| CorpusSource::Synthetic)
}

/// Short corpus name for reports and dataset sidecars.
pub fn corpus_label() -> &'static str {
    match corpus_source() {
        CorpusSource::Synthetic => "synthetic",
        CorpusSource::AnnBenchmarks { slug, .. } => slug,
        CorpusSource::HuggingFace { repo, .. } => repo,
        CorpusSource::LocalParquet { dir } => dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("parquet"),
    }
}

/// Vector dimension of the active corpus: the synthetic constant for
/// the generator, the dataset's own embedding width for a real corpus
/// (read once from the parquet schema).
pub fn dim() -> usize {
    static DIM_ONCE: OnceLock<usize> = OnceLock::new();
    *DIM_ONCE.get_or_init(|| match corpus_source() {
        CorpusSource::Synthetic => SYNTHETIC_DIM,
        CorpusSource::AnnBenchmarks { dir, slug } => real_corpus_dim(dir, slug),
        source => parquet_dim(&parquet_shards_for(source)),
    })
}

/// One `(local_doc_id, distance)` hit — same shape `VectorReader::search`
/// returns. Re-exported here so recall helpers stay engine-agnostic.
pub type Hit = (u32, f32);

/// Doc count for superfile-shape benches (one-superfile scale). 1M ×
/// 1024 (f32) ≈ 4 GB — mmap-backed corpus streams from disk, so only
/// the engine's resident footprint (not the corpus) bounds RAM. It is
/// the single-superfile cold-open unit for the warm/cold tiers.
pub const SUPERFILE_DOCS: usize = 1_000_000;

/// Doc count for supertable-shape benches (scale-out / sharding
/// scale). 10M × 1024 (f32) ≈ 40 GB on disk — the mmap-backed corpus
/// streams from disk (never fully resident), so disk space, not RAM,
/// is the constraint at this dim. This is the headline supertable
/// scale that the warm/cold tiers run over the object store.
pub const SUPERTABLE_DOCS: usize = 10_000_000;

/// Document count for the **superfile** test — a single-superfile index
/// built and queried entirely **in memory**. Defaults to
/// [`SUPERFILE_DOCS`] (1M); override with `INFINO_BENCH_SUPERFILE_DOCS`
/// for a quicker local loop or a larger stress run.
pub fn superfile_docs() -> usize {
    clamp_docs_to_corpus(docs_from_env("INFINO_BENCH_SUPERFILE_DOCS", SUPERFILE_DOCS))
}

/// Document count for the **supertable** test — a multi-superfile table
/// committed to and queried from **object storage**. Defaults to
/// [`SUPERTABLE_DOCS`] (10M); override with
/// `INFINO_BENCH_SUPERTABLE_DOCS`.
pub fn supertable_docs() -> usize {
    clamp_docs_to_corpus(docs_from_env(
        "INFINO_BENCH_SUPERTABLE_DOCS",
        SUPERTABLE_DOCS,
    ))
}

/// Cap a requested doc count at what a real corpus actually holds (the
/// synthetic generator has no cap). Loud when it bites: a clamped run
/// measures a different scale than the caller asked for.
fn clamp_docs_to_corpus(n: usize) -> usize {
    match corpus_source() {
        CorpusSource::Synthetic => n,
        CorpusSource::AnnBenchmarks { dir, slug } => {
            let rows = real_corpus_rows(dir, slug);
            if n > rows {
                eprintln!("[corpus] requested {n} docs but the real corpus holds {rows}; clamping");
            }
            n.min(rows)
        }
        source => {
            // Queries are the rows past the ingested prefix, so the ingest
            // must stop short of the dataset's end or there is nothing held
            // out to query with.
            // Budget: ingest + delta commit + held-out queries must all fit
            // inside the dataset. The delta is a fraction of the ingest, so
            // solve for the ingest rather than subtracting a constant — a
            // shortfall here previously surfaced as an opaque ground-truth
            // size assertion after a full ingest had already run.
            let rows = parquet_rows(&parquet_shards_for(source));
            let usable = rows.saturating_sub(PARQUET_QUERY_RESERVE_ROWS);
            let ingestable = (usable as f64 / (1.0 + PARQUET_DELTA_HEADROOM)) as usize;
            assert!(
                ingestable > 0,
                "dataset holds {rows} rows, fewer than the {PARQUET_QUERY_RESERVE_ROWS}-row \
                 query reserve"
            );
            if n > ingestable {
                eprintln!(
                    "[corpus] requested {n} docs; dataset holds {rows}, reserving \
                     {PARQUET_QUERY_RESERVE_ROWS} rows for held-out queries plus delta \
                     headroom — clamping to {ingestable}"
                );
            }
            n.min(ingestable)
        }
    }
}

/// Human name of a source kind, for the not-yet-wired reports above.
fn corpus_kind(source: &CorpusSource) -> &'static str {
    match source {
        CorpusSource::Synthetic => "synthetic",
        CorpusSource::AnnBenchmarks { .. } => "ann-benchmarks",
        CorpusSource::HuggingFace { .. } => "hugging-face parquet",
        CorpusSource::LocalParquet { .. } => "local parquet",
    }
}

/// Parse a positive doc-count override from `var`, falling back to
/// `default` when unset, empty, unparseable, or zero.
fn docs_from_env(var: &str, default: usize) -> usize {
    env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Parallel-writer count for the "N writers" build row — how many
/// writers build the corpus concurrently. Applied identically to every
/// engine (infino shards across this many builders; Tantivy uses this
/// many indexing threads). Defaults to the machine's logical core count
/// so runs on the same box are comparable; `INFINO_BENCH_WRITERS`
/// overrides for shard-shape experiments.
pub fn parallel_writers() -> usize {
    std::env::var("INFINO_BENCH_WRITERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or_else(num_cpus::get)
}

/// IVF cluster count. Conventionally `~sqrt(n_docs)`, snapped to a
/// fixed value per scale band so 1M and 10M runs share a stable
/// `n_cent`.
pub fn n_cent(n_docs: usize) -> usize {
    if n_docs >= N_CENT_LARGE_DOC_THRESHOLD {
        N_CENT_LARGE
    } else if n_docs >= N_CENT_MEDIUM_DOC_THRESHOLD {
        N_CENT_MEDIUM
    } else {
        N_CENT_SMALL
    }
}

/// Doc-count threshold (≥) at/above which the large `n_cent` band is used.
const N_CENT_LARGE_DOC_THRESHOLD: usize = 5_000_000;
/// IVF centroid count for the large scale band.
const N_CENT_LARGE: usize = 4096;
/// Doc-count threshold (≥) for the medium `n_cent` band.
const N_CENT_MEDIUM_DOC_THRESHOLD: usize = 100_000;
/// IVF centroid count for the medium scale band.
const N_CENT_MEDIUM: usize = 1024;
/// IVF centroid count for small corpora (below the medium threshold).
const N_CENT_SMALL: usize = 64;

/// Average bytes-per-token estimate used to pre-size a doc's `String`
/// (`(TOKENS_PER_DOC + 1) * AVG_BYTES_PER_TOKEN`).
const AVG_BYTES_PER_TOKEN: usize = 8;
/// Gaussian scale of a planted cluster center (controls cluster signal
/// strength relative to per-doc noise).
const CENTER_GAUSSIAN_SCALE: f32 = 3.0;
/// Per-dimension Gaussian noise added around a cluster center.
const DOC_NOISE_SIGMA: f32 = 0.3;
/// Gaussian scale for pure-noise smoke queries (no planted cluster).
const QUERY_GAUSSIAN_SCALE: f32 = 3.0;
/// Coprime stride used to spread generated queries across the corpus
/// (and thus across clusters).
const QUERY_BASE_DOC_STRIDE: usize = 7919;
/// Recall returned for an empty ground-truth set (vacuously perfect).
const EMPTY_TRUTH_RECALL: f32 = 1.0;
/// Seconds-to-microseconds factor for p50 latency reporting.
const SEC_TO_MICROS: f32 = 1e6;
/// Random-rotation RNG seed for bench vector-index builders.
const ROT_SEED: u64 = 7;
/// Decimal128 precision for the injected `_id` column in bench fixtures.
const ID_DECIMAL_PRECISION: u8 = 38;
/// Decimal128 scale for the injected `_id` column (integer ids).
const ID_DECIMAL_SCALE: i8 = 0;

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
        let mut doc = String::with_capacity((TOKENS_PER_DOC + 1) * AVG_BYTES_PER_TOKEN);
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

/// Disk-backed Zipfian text corpus for large FTS supertable benches.
///
/// At 10M docs, `Vec<String>` pins the full corpus on the heap before the
/// writer under test starts. This mirrors [`MmapVectorCorpus`]: store UTF-8
/// bytes in a temp file, keep only an offset table in memory, and materialize
/// Arrow string arrays one append chunk at a time.
pub struct MmapTextCorpus {
    _tmp: TempDir,
    map: Mmap,
    offsets: Vec<u64>,
}

impl MmapTextCorpus {
    pub fn generate(n_docs: usize, seed: u64) -> Self {
        let tmp = TempDir::new().expect("create MmapTextCorpus tempdir");
        let path = tmp.path().join("corpus.txt");

        // Each doc's on-disk byte length is RNG-independent: the body is
        // exactly `TOKENS_PER_DOC` tokens of " term{:05}" (10 bytes each,
        // since VOCAB_SIZE = 10_000 keeps the index ≤ 5 digits), and the
        // leading "doc{doc_id:07}" is 3 + max(7, digits(doc_id)) bytes.
        // So the offset table is deterministic and every chunk can write
        // to its own disjoint byte range — letting all cores generate the
        // (RNG-driven) token *content* in parallel via positioned writes.
        #[inline]
        fn doc_len(doc_id: usize) -> u64 {
            let digits = if doc_id == 0 {
                1
            } else {
                doc_id.ilog10() as usize + 1
            };
            let doc_token = DOC_ID_PREFIX_BYTES + digits.max(DOC_ID_PAD_WIDTH);
            (doc_token + TOKENS_PER_DOC * TERM_BYTES) as u64
        }

        let mut offsets = Vec::with_capacity(n_docs + 1);
        let mut pos = 0u64;
        offsets.push(pos);
        for doc_id in 0..n_docs {
            pos += doc_len(doc_id);
            offsets.push(pos);
        }
        let total = pos;

        let file = File::create(&path).expect("create text corpus file");
        file.set_len(total).expect("set_len text corpus");

        // rayon fans the chunks across all cores; each pwrites to its own
        // disjoint byte range (offsets are RNG-independent, computed above)
        // and draws from its own deterministic per-chunk RNG.
        let n_chunks = n_docs.div_ceil(TEXT_CORPUS_CHUNK_DOCS).max(1);
        let offsets_ref = &offsets;
        let file_ref = &file;
        (0..n_chunks).into_par_iter().for_each(|c| {
            let start = c * TEXT_CORPUS_CHUNK_DOCS;
            if start >= n_docs {
                return;
            }
            let end = ((c + 1) * TEXT_CORPUS_CHUNK_DOCS).min(n_docs);
            let base = offsets_ref[start];
            let cap = (offsets_ref[end] - base) as usize;
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, c));
            let zipf = ZipfDistribution::new(VOCAB_SIZE);
            let mut buf: Vec<u8> = Vec::with_capacity(PARALLEL_CORPUS_WRITE_BUF_CAPACITY);
            let mut written = 0usize;
            let flush = |buf: &mut Vec<u8>, written: &mut usize| {
                if buf.is_empty() {
                    return;
                }
                file_ref
                    .write_all_at(buf, base + (*written as u64))
                    .expect("pwrite text corpus buffer");
                *written += buf.len();
                buf.clear();
            };
            for doc_id in start..end {
                write!(buf, "doc{doc_id:07}").expect("fmt doc token");
                for _ in 0..TOKENS_PER_DOC {
                    write!(buf, " term{:05}", zipf.sample(&mut rng)).expect("fmt term");
                }
                if buf.len() >= PARALLEL_CORPUS_WRITE_BUF_CAPACITY {
                    flush(&mut buf, &mut written);
                }
            }
            flush(&mut buf, &mut written);
            assert_eq!(written, cap, "deterministic doc-length mismatch");
        });

        file.sync_all().expect("sync text corpus");
        drop(file);

        let file = File::open(&path).expect("reopen text corpus");
        // SAFETY: this helper owns the temp file and never writes to it after
        // the fsync above, so the read-only mmap cannot observe mutation.
        let map = unsafe { Mmap::map(&file).expect("mmap text corpus") };
        Self {
            _tmp: tmp,
            map,
            offsets,
        }
    }

    pub fn n_docs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total logical text bytes across all docs — the ingest input
    /// payload size, used to report build bandwidth in MB/s.
    pub fn total_bytes(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0) - self.offsets.first().copied().unwrap_or(0)
    }

    pub fn doc(&self, idx: usize) -> &str {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        std::str::from_utf8(&self.map[start..end]).expect("generated corpus is valid UTF-8")
    }

    pub fn chunk_strs(&self, start: usize, len: usize) -> Vec<&str> {
        let end = (start + len).min(self.n_docs());
        (start..end).map(|idx| self.doc(idx)).collect()
    }

    /// Drop the resident pages backing docs `[start, start + len)`
    /// from this process's RSS (`MADV_DONTNEED`, best-effort). The
    /// streamed build loop calls this after committing each chunk so
    /// the whole-process RSS sampler measures the engine, not the
    /// harness's already-consumed corpus pages. Page-rounding may also
    /// drop a neighbouring chunk's boundary page — harmless; clean
    /// file-backed pages transparently re-fault from the file.
    pub fn advise_consumed(&self, start: usize, len: usize) {
        let end = (start + len).min(self.n_docs());
        if start >= end {
            return;
        }
        let lo = page_floor(self.offsets[start] as usize);
        let hi = self.offsets[end] as usize;
        // SAFETY: read-only shared file mapping — `MADV_DONTNEED` can
        // only discard clean pages, which re-fault from the backing
        // file on the next touch; no data is mutated or lost. The
        // byte range lies within the map by construction of `offsets`.
        unsafe {
            let _ =
                self.map
                    .unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, lo, hi - lo);
        }
    }

    /// Materialize the whole corpus as `(doc_id, text)` rows borrowing
    /// from the mmap — the input shape the engine-generic FTS driver
    /// feeds to every engine. `doc_id` is the dense row index, so it
    /// doubles as the cross-engine recall id.
    pub fn rows(&self) -> Vec<(u64, &str)> {
        (0..self.n_docs())
            .map(|i| (i as u64, self.doc(i)))
            .collect()
    }
}

/// Page size assumed for `madvise` range alignment. 4 KiB on every
/// Linux bench host; a larger real page size only makes the floor
/// coarser, which is still correct (more bytes advised away).
const PAGE_BYTES: usize = 4096;

/// Round a byte offset down to the containing page boundary —
/// `madvise` requires a page-aligned start address.
fn page_floor(off: usize) -> usize {
    off & !(PAGE_BYTES - 1)
}

pub mod combined;
pub mod grading;
pub mod sql;
pub mod sql_battery;

pub use combined::SequentialSyntheticCorpus;

// ─── Vector corpus ────────────────────────────────────────────────────

/// Generate `n_docs` planted-cluster vectors of [`dim()`] dimensions,
/// optionally per-doc normalized for cosine. `n_cent` planted centers
/// drawn from `3·N(0, 1)` per dim; each doc lives near a center with
/// `sigma = 0.3` per-dim Gaussian noise.
///
/// **Centers are intentionally NOT normalized.** At `dim()=384` the
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
            (0..dim())
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * CENTER_GAUSSIAN_SCALE
                })
                .collect()
        })
        .collect();

    let mut out: Vec<f32> = Vec::with_capacity(n_docs * dim());
    for i in 0..n_docs {
        let center = &centers[i % n_cent];
        let mut v: Vec<f32> = center
            .iter()
            .map(|&c| {
                let s: f64 = dist.sample(&mut rng);
                c + (s as f32) * DOC_NOISE_SIGMA
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
pub struct MmapVectorCorpus {
    _tmp: Option<TempDir>,
    map: Mmap,
    n_docs: usize,
    dim: usize,
}

impl MmapVectorCorpus {
    pub fn generate(n_docs: usize, n_cent: usize, seed: u64, normalize_each: bool) -> Self {
        Self::generate_range(0, n_docs, n_cent, seed, normalize_each)
    }

    /// Generate only `[start_doc, start_doc + n_docs)` while preserving the
    /// same global chunk seeds and RNG positions as a full corpus.
    pub fn generate_range(
        start_doc: usize,
        n_docs: usize,
        n_cent: usize,
        seed: u64,
        normalize_each: bool,
    ) -> Self {
        let tmp = TempDir::new().expect("create MmapVectorCorpus tempdir");
        let path = tmp.path().join("corpus.bin");
        let end_doc = start_doc
            .checked_add(n_docs)
            .expect("vector corpus document range overflow");

        // Centers are derived from `seed` exactly as the sequential
        // builder did, so the planted IVF cluster structure — and hence
        // the brute-force recall ground truth — is unchanged. Each row's
        // cluster (`i % n_cent`) is a pure function of its index, also
        // preserved. Only the per-doc Gaussian *noise* stream is drawn
        // from a per-chunk RNG so generation parallelizes; recall is
        // recomputed from the actual corpus, so an equivalent (not
        // identical) noise realization is fine.
        let mut crng = StdRng::seed_from_u64(seed);
        let cdist = StandardNormal;
        let centers: Vec<Vec<f32>> = (0..n_cent)
            .map(|_| {
                (0..dim())
                    .map(|_| {
                        let s: f64 = cdist.sample(&mut crng);
                        (s as f32) * CENTER_GAUSSIAN_SCALE
                    })
                    .collect()
            })
            .collect();

        let row_bytes = dim() * size_of::<f32>();
        let total = vector_corpus_byte_len(n_docs).expect("vector corpus byte length");
        let file = File::create(&path).expect("create corpus file");
        file.set_len(total).expect("set_len vector corpus");

        // rayon fans the chunks across all cores; each writes a fixed-stride
        // row range via a positioned write and draws noise from its own
        // deterministic per-chunk RNG (shared centers keep clusters intact).
        let first_chunk = start_doc / VECTOR_CORPUS_CHUNK_DOCS;
        let end_chunk = end_doc.div_ceil(VECTOR_CORPUS_CHUNK_DOCS);
        let centers_ref = &centers;
        let file_ref = &file;
        (first_chunk..end_chunk).into_par_iter().for_each(|c| {
            let chunk_start = c * VECTOR_CORPUS_CHUNK_DOCS;
            let start = chunk_start.max(start_doc);
            let end = ((c + 1) * VECTOR_CORPUS_CHUNK_DOCS).min(end_doc);
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, c));
            let dist = StandardNormal;
            let mut buf: Vec<u8> = Vec::with_capacity(PARALLEL_CORPUS_WRITE_BUF_CAPACITY);
            let base = ((start - start_doc) as u64) * (row_bytes as u64);
            let expected = (end - start) * row_bytes;
            let mut written = 0usize;
            let flush = |buf: &mut Vec<u8>, written: &mut usize| {
                if buf.is_empty() {
                    return;
                }
                file_ref
                    .write_all_at(buf, base + (*written as u64))
                    .expect("pwrite vector corpus buffer");
                *written += buf.len();
                buf.clear();
            };
            // The first requested row may begin partway through a global
            // chunk. Advance that chunk's noise stream without materializing
            // any preceding vectors so the requested rows remain bit-identical
            // to a full corpus generated with the same knobs.
            for _ in chunk_start..start {
                for _ in 0..dim() {
                    let _: f64 = dist.sample(&mut rng);
                }
            }
            let mut row = vec![0.0f32; dim()];
            for i in start..end {
                let center = &centers_ref[i % n_cent];
                for (j, slot) in row.iter_mut().enumerate() {
                    let s: f64 = dist.sample(&mut rng);
                    *slot = center[j] + (s as f32) * DOC_NOISE_SIGMA;
                }
                if normalize_each {
                    normalize(&mut row);
                }
                buf.extend_from_slice(bytemuck::cast_slice(&row));
                if buf.len() >= PARALLEL_CORPUS_WRITE_BUF_CAPACITY {
                    flush(&mut buf, &mut written);
                }
            }
            flush(&mut buf, &mut written);
            assert_eq!(written, expected, "vector corpus chunk byte mismatch");
        });

        file.sync_all().expect("sync corpus");
        drop(file);

        let file = File::open(&path).expect("reopen corpus");
        Self::from_file(file, n_docs, Some(tmp)).expect("mmap generated vector corpus")
    }

    /// Open an existing raw f32 corpus without copying or modifying it.
    ///
    /// The file must contain exactly `n_docs * dim()` native-endian f32 values.
    pub fn open(path: &Path, n_docs: usize) -> IoResult<Self> {
        let file = File::open(path)?;
        Self::from_file(file, n_docs, None)
    }

    fn from_file(file: File, n_docs: usize, tmp: Option<TempDir>) -> IoResult<Self> {
        Self::from_file_with_dim(file, n_docs, dim(), tmp)
    }

    /// As [`Self::from_file`] but with the row width passed in rather than
    /// read from the process-wide [`dim`]. Real-dataset loads use this: the
    /// dataset's own width is known from its file, and `dim()` resolves once
    /// per process from env — so validating against it would reject a
    /// correctly-sized corpus whenever the two disagree (which is precisely
    /// what happens in tests, and would happen for any second dataset).
    fn from_file_with_dim(
        file: File,
        n_docs: usize,
        dim: usize,
        tmp: Option<TempDir>,
    ) -> IoResult<Self> {
        let expected = vector_corpus_byte_len_for_dim(n_docs, dim)?;
        let actual = file.metadata()?.len();
        if actual != expected {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("vector corpus size is {actual} bytes; expected exactly {expected} bytes"),
            ));
        }
        // SAFETY: the mapping is read-only. Generated files are never written
        // after their fsync; persisted benchmark inputs must likewise remain
        // immutable while the benchmark owns this mapping.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self {
            _tmp: tmp,
            map,
            n_docs,
            dim,
        })
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

    /// Drop the resident pages backing rows `[start, start + len)`
    /// from this process's RSS — same contract and safety argument as
    /// [`MmapTextCorpus::advise_consumed`].
    pub fn advise_consumed(&self, start: usize, len: usize) {
        let end = (start + len).min(self.n_docs);
        if start >= end {
            return;
        }
        let row_bytes = self.dim * size_of::<f32>();
        let lo = page_floor(start * row_bytes);
        let hi = end * row_bytes;
        // SAFETY: read-only shared file mapping — `MADV_DONTNEED` only
        // discards clean pages, which re-fault from the backing file;
        // the range lies within the map (`end <= n_docs`).
        unsafe {
            let _ =
                self.map
                    .unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, lo, hi - lo);
        }
    }
}

fn vector_corpus_byte_len(n_docs: usize) -> IoResult<u64> {
    vector_corpus_byte_len_for_dim(n_docs, dim())
}

fn vector_corpus_byte_len_for_dim(n_docs: usize, dim: usize) -> IoResult<u64> {
    let row_bytes = dim
        .checked_mul(size_of::<f32>())
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "vector corpus row size overflow"))?;
    let total = n_docs.checked_mul(row_bytes).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            "vector corpus byte length overflow",
        )
    })?;
    u64::try_from(total).map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            "vector corpus byte length exceeds u64",
        )
    })
}

// ─── Real-corpus (ann-benchmarks HDF5) source ─────────────────────────

/// Row-slab size for streaming a real corpus into the mmap: bounded peak
/// memory (slab x dim x 4 bytes) rather than the whole dataset, which is
/// multi-GB for the larger datasets.
const H5_SLAB_ROWS: usize = 8_192;

/// Connect timeout for a dataset download.
const CORPUS_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Whole-download deadline: generous enough for the largest published
/// dataset on a slow link, finite so a half-open connection cannot hang a
/// bench run forever.
const CORPUS_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(3 * 3600);

/// Canonical ann-benchmarks dataset host: the academic, vendor-neutral
/// mirror serving the `*-angular.hdf5` / `*-euclidean.hdf5` files the
/// published ANN benchmarks are run against.
const DEFAULT_CORPUS_URL_BASE: &str = "https://ann-benchmarks.com";

/// Corpus rows dataset inside the HDF5 file.
const H5_TRAIN: &str = "train";
/// Query rows dataset.
const H5_TEST: &str = "test";
/// Official top-k neighbour ids per query (dataset row indices) — the
/// published ground truth, so recall needs no brute-force recompute and
/// the numbers are directly comparable to ann-benchmarks results.
const H5_NEIGHBORS: &str = "neighbors";

/// Resolve the dataset file, downloading it once when absent. The file name
/// is the dataset slug plus `.hdf5` (`glove-25-angular` ->
/// `glove-25-angular.hdf5`), matching the host's own layout, so a slug is
/// the only thing a caller configures. Present files are used as-is.
/// Streams to `<file>.part` and renames, so an interrupted download never
/// leaves a truncated file under the real name, and verifies the received
/// length against `Content-Length` when the server provides one.
fn real_corpus_file(dir: &Path, slug: &str) -> PathBuf {
    let name = format!("{slug}.hdf5");
    let url = format!("{DEFAULT_CORPUS_URL_BASE}/{name}");
    download_if_absent(dir, &name, &url)
}

/// Shared blocking HTTP client for corpus downloads: bounded connect and
/// whole-transfer deadlines, so a dead host fails fast and a half-open
/// connection to a multi-GB file cannot hang a run forever.
fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .connect_timeout(CORPUS_CONNECT_TIMEOUT)
        .timeout(CORPUS_DOWNLOAD_TIMEOUT)
        .build()
        .expect("build http client")
}

/// Fetch `url` into `dir/name` unless already present. Streams to
/// `<name>.part` and renames, so an interrupted transfer never leaves a
/// truncated file under the real name, and verifies the received length
/// against `Content-Length` when the server sends one — a
/// truncated-but-200 response would otherwise be accepted silently and
/// then surface as a confusing parse error deep in a run.
fn download_if_absent(dir: &Path, name: &str, url: &str) -> PathBuf {
    let path = dir.join(name);
    if path.exists() {
        return path;
    }
    eprintln!(
        "[corpus] {name} missing from {}; downloading {url}...",
        dir.display()
    );
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("create {}: {e}", dir.display()));
    let mut resp = http_client()
        .get(url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .unwrap_or_else(|e| {
            panic!(
                "download {url} failed: {e}\nstage {name} in {} manually instead",
                dir.display()
            )
        });
    let expected = resp.content_length();
    if let Some(len) = expected {
        eprintln!("[corpus]   size: {}", rss::fmt_bytes(len));
    }
    let tmp = dir.join(format!("{name}.part"));
    let mut out = BufWriter::new(
        File::create(&tmp).unwrap_or_else(|e| panic!("create {}: {e}", tmp.display())),
    );
    let written = std::io::copy(&mut resp, &mut out)
        .and_then(|n| {
            out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
            Ok(n)
        })
        .unwrap_or_else(|e| {
            let _ = std::fs::remove_file(&tmp);
            panic!("stream {url} to {}: {e}", tmp.display())
        });
    // Integrity: a truncated-but-200 response is otherwise accepted
    // silently and then fails as a confusing parse error deep in a run.
    if let Some(len) = expected
        && written != len
    {
        let _ = std::fs::remove_file(&tmp);
        panic!(
            "{url}: received {written} bytes, Content-Length said {len} — refusing partial data"
        );
    }
    std::fs::rename(&tmp, &path).unwrap_or_else(|e| {
        let _ = std::fs::remove_file(&tmp);
        panic!("finalize {}: {e}", path.display())
    });
    eprintln!("[corpus]   downloaded {}", path.display());
    path
}

/// Open the dataset file for `slug` under `dir`, fetching it if needed.
fn open_h5(dir: &Path, slug: &str) -> hdf5::File {
    let path = real_corpus_file(dir, slug);
    hdf5::File::open(&path).unwrap_or_else(|e| panic!("open HDF5 dataset {}: {e}", path.display()))
}

/// Shape of one 2-D dataset: `(rows, cols)`.
fn h5_shape(file: &hdf5::File, name: &str) -> (usize, usize) {
    let ds = file
        .dataset(name)
        .unwrap_or_else(|e| panic!("dataset {name:?} missing from the HDF5 file: {e}"));
    let shape = ds.shape();
    assert_eq!(
        shape.len(),
        2,
        "dataset {name:?} must be 2-D, got shape {shape:?}"
    );
    (shape[0], shape[1])
}

/// Corpus row count of the real dataset (metadata only, no data read).
pub fn real_corpus_rows(dir: &Path, slug: &str) -> usize {
    h5_shape(&open_h5(dir, slug), H5_TRAIN).0
}

/// Embedding width of the real dataset.
pub fn real_corpus_dim(dir: &Path, slug: &str) -> usize {
    h5_shape(&open_h5(dir, slug), H5_TRAIN).1
}

impl MmapVectorCorpus {
    /// Materialize the first `n_docs` corpus rows of a real dataset into the
    /// standard corpus mmap. Rows stream in row-slabs, so peak memory is one
    /// slab rather than the dataset. `dim` is passed explicitly (not read
    /// from the process-wide [`dim`]) so the write/normalize path is
    /// exercisable offline in tests.
    pub fn from_hdf5(
        dir: &Path,
        slug: &str,
        n_docs: usize,
        dim: usize,
        normalize_each: bool,
    ) -> IoResult<Self> {
        let file = open_h5(dir, slug);
        let (rows, cols) = h5_shape(&file, H5_TRAIN);
        if cols != dim {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("dataset {slug} is {cols}-dim; {dim} requested"),
            ));
        }
        if rows < n_docs {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("dataset {slug} holds {rows} rows; {n_docs} requested"),
            ));
        }
        let ds = file
            .dataset(H5_TRAIN)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
        let tmp = TempDir::new().expect("create MmapVectorCorpus tempdir");
        let path = tmp.path().join("corpus.bin");
        let mut writer = BufWriter::new(File::create(&path)?);
        let mut done = 0usize;
        while done < n_docs {
            let take = (n_docs - done).min(H5_SLAB_ROWS);
            let slab: ndarray::Array2<f32> =
                ds.read_slice_2d(ndarray::s![done..done + take, ..])
                    .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
            for mut row in slab.outer_iter().map(|r| r.to_vec()) {
                if normalize_each {
                    normalize(&mut row);
                }
                let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                writer.write_all(&bytes)?;
            }
            done += take;
        }
        writer.flush()?;
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        Self::from_file_with_dim(File::open(&path)?, n_docs, dim, Some(tmp))
    }
}

// ─── Real-corpus (parquet) source ─────────────────────────────────────

/// Text column names accepted in a parquet dataset, in priority order.
/// A dataset carrying one of these can drive the FTS and SQL tiers on the
/// same rows the vector tier ingests — the reason real corpora belong in
/// `benches/` rather than in a vector-only harness.
const PARQUET_TEXT_COLUMNS: [&str; 3] = ["text", "title", "body"];

/// Refuse a real corpus for a bench cell that only generates synthetic
/// data. Falling back silently would measure generated data under the
/// real corpus's label — and worse, `clamp_docs_to_corpus` sizes the run
/// to the real dataset's row count, so the report would be shaped by rows
/// the cell never read. Real-corpus text lives on the supertable tier;
/// the superfile SQL cell reads real corpora through its schema-driven
/// path, and the superfile FTS cell stays synthetic until it grows a
/// reader of its own.
pub fn require_synthetic(cell: &str) {
    if !matches!(corpus_source(), CorpusSource::Synthetic) {
        panic!(
            "corpus {} is not supported by the `{cell}` cell: it measures its \
             synthetic corpus only, so running it under a real corpus label \
             would report numbers for data it never read — drop `corpus=` for \
             this cell, or run the supertable tier, which consumes real corpora",
            corpus_label()
        );
    }
}

/// Whether the active corpus can supply text (and therefore feed FTS /
/// SQL). Synthetic always can; a parquet dataset can when it carries one
/// of [`PARQUET_TEXT_COLUMNS`]; an ann-benchmarks dataset never can — its
/// HDF5 files hold vectors only.
pub fn corpus_has_text() -> bool {
    match corpus_source() {
        CorpusSource::Synthetic => true,
        CorpusSource::AnnBenchmarks { .. } => false,
        source => {
            let shards = parquet_shards_for(source);
            let first = shards.first().expect("at least one shard");
            let f = File::open(first).unwrap_or_else(|e| panic!("open {}: {e}", first.display()));
            let reader = ParquetRecordBatchReaderBuilder::try_new(f)
                .unwrap_or_else(|e| panic!("read {}: {e}", first.display()));
            let schema = reader.schema();
            PARQUET_TEXT_COLUMNS
                .iter()
                .any(|n| schema.column_with_name(n).is_some())
        }
    }
}

/// The text column of one batch as owned strings.
fn parquet_text_rows(batch: &RecordBatch) -> Vec<String> {
    let col = PARQUET_TEXT_COLUMNS
        .iter()
        .find_map(|name| batch.column_by_name(name))
        .unwrap_or_else(|| {
            panic!("parquet batch has none of the text columns {PARQUET_TEXT_COLUMNS:?}")
        });
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        (0..a.len()).map(|i| a.value(i).to_string()).collect()
    } else if let Some(a) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
        (0..a.len()).map(|i| a.value(i).to_string()).collect()
    } else {
        panic!(
            "text column must be Utf8 or LargeUtf8, got {:?}",
            col.data_type()
        )
    }
}

impl MmapTextCorpus {
    /// Materialize the first `n_docs` text rows of a parquet dataset into
    /// the standard text-corpus mmap, streaming batch by batch. Newlines in
    /// the source are replaced with spaces: the mmap format is one document
    /// per line, so an embedded newline would split one document into two.
    pub fn from_parquet_shards(shards: &[PathBuf], n_docs: usize) -> IoResult<Self> {
        let tmp = TempDir::new().expect("create MmapTextCorpus tempdir");
        let path = tmp.path().join("corpus.txt");
        let mut writer = BufWriter::new(File::create(&path)?);
        let mut offsets: Vec<u64> = Vec::with_capacity(n_docs + 1);
        let mut pos = 0u64;
        offsets.push(0);
        let mut written = 0usize;
        'outer: for shard in shards {
            let f = File::open(shard)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(f)
                .and_then(|b| b.with_batch_size(PARQUET_BATCH_ROWS).build())
                .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
            for batch in reader {
                let batch = batch.map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
                for doc in parquet_text_rows(&batch) {
                    let line = doc.replace(['\n', '\r'], " ");
                    writer.write_all(line.as_bytes())?;
                    writer.write_all(b"\n")?;
                    pos += line.len() as u64 + 1;
                    offsets.push(pos);
                    written += 1;
                    if written == n_docs {
                        break 'outer;
                    }
                }
            }
        }
        if written < n_docs {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("parquet dataset holds {written} text rows; {n_docs} requested"),
            ));
        }
        writer.flush()?;
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        let file = File::open(&path)?;
        // SAFETY: read-only mapping of a file this function just wrote and
        // fsynced, and which nothing mutates while the corpus owns it.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self {
            _tmp: tmp,
            map,
            offsets,
        })
    }
}

/// Hugging Face resolve base. Only the file endpoint is used — the `/api/`
/// metadata endpoint requires credentials, so shard names come from a
/// manifest read (below) rather than a directory listing.
const HF_RESOLVE_BASE: &str = "https://huggingface.co";

/// Vector column names accepted in a parquet dataset, in priority order.
/// Datasets name the embedding column differently (`openai` in
/// dbpedia-entities-openai-1M, `emb` elsewhere); both are read.
const PARQUET_VECTOR_COLUMNS: [&str; 2] = ["openai", "emb"];

/// Rows held back from the end of a parquet dataset to serve as queries.
/// Generous relative to any battery's query count, and small relative to
/// the datasets this source is for.
const PARQUET_QUERY_RESERVE_ROWS: usize = 8_192;

/// Headroom the ingest needs BEYOND its doc count, as a fraction: the
/// supertable battery's delta phase appends exactly ONE more commit's
/// worth of rows (`n_docs / n_commits`), and its lifecycle ground truth
/// requires the corpus to contain them. The commit count never drops below
/// `ingest::supertable::MIN_COMMIT_CHUNKS` (16) and only rises with scale,
/// so one sixteenth is the exact worst case, not an estimate. Sized for the smallest commit count the
/// bench uses, so the reserve is never short.
const PARQUET_DELTA_HEADROOM: f64 = 1.0 / 16.0;

/// Rows kept in memory while converting a parquet batch.
const PARQUET_BATCH_ROWS: usize = 1_024;

/// One parquet dataset's shard files, in ingest order. A `hf:` source
/// downloads any that are missing first; both then read the directory, so
/// there is no naming convention to satisfy and no producer-specific
/// layout in this code.
pub fn parquet_shards_for(source: &CorpusSource) -> Vec<PathBuf> {
    if let CorpusSource::HuggingFace { dir, repo } = source {
        hf_download_shards(dir, repo);
    }
    let dir = match source {
        CorpusSource::LocalParquet { dir } | CorpusSource::HuggingFace { dir, .. } => dir,
        other => panic!("{} is not a parquet source", corpus_kind(other)),
    };
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "parquet"))
        .collect();
    shards.sort();
    assert!(
        !shards.is_empty(),
        "{} holds no *.parquet shards",
        dir.display()
    );
    shards
}

/// Resolve a Hugging Face dataset's shard files, downloading any that are
/// absent. Shard names follow the Hub's own convention
/// (`data/train-<i>-of-<n>-<hash>.parquet`), which cannot be derived — so
/// the first shard's name is required once via [`HF_SHARDS_ENV`]-free
/// discovery: we read the local directory when it already holds shards,
/// and otherwise ask the Hub's tree endpoint for the file list. The tree
/// endpoint is the one metadata route that answers unauthenticated.
fn hf_download_shards(dir: &Path, repo: &str) {
    // Reconcile against the Hub's shard list rather than treating any
    // parquet file as "already staged": a partially downloaded directory
    // would otherwise be accepted, silently shrinking the corpus to
    // whatever happened to be present (measured: one shard's 38,462 rows
    // standing in for a 1M-row dataset).
    let url = format!("{HF_RESOLVE_BASE}/api/datasets/{repo}/tree/main/data");
    let client = http_client();
    let listing = client
        .get(&url)
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .and_then(|r| r.text())
        .unwrap_or_else(|e| {
            panic!(
                "list {repo} shards via {url} failed: {e}\nstage the dataset's \
                 data/*.parquet files in {} manually instead",
                dir.display()
            )
        });
    // Minimal extraction: the listing is JSON objects each carrying a
    // "path" field; no JSON dependency needed for one field.
    let mut names: Vec<String> = listing
        .split("\"path\":\"")
        .skip(1)
        .filter_map(|rest| rest.split('"').next())
        .filter(|n| n.ends_with(".parquet"))
        .map(str::to_string)
        .collect();
    names.sort();
    assert!(
        !names.is_empty(),
        "{repo} listing returned no parquet shards: {}",
        &listing[..listing.len().min(200)]
    );
    let missing = names
        .iter()
        .filter(|n| !dir.join(n.rsplit('/').next().unwrap_or(n)).exists())
        .count();
    if missing > 0 {
        eprintln!(
            "[corpus] {repo}: {}/{} shards staged, fetching {missing}...",
            names.len() - missing,
            names.len()
        );
    }
    for name in &names {
        let leaf = name.rsplit('/').next().unwrap_or(name);
        download_if_absent(
            dir,
            leaf,
            &format!("{HF_RESOLVE_BASE}/datasets/{repo}/resolve/main/{name}"),
        );
    }
}

/// Row count of a parquet dataset, from footer metadata only.
fn parquet_rows(shards: &[PathBuf]) -> usize {
    shards
        .iter()
        .map(|p| {
            let f = File::open(p).unwrap_or_else(|e| panic!("open {}: {e}", p.display()));
            ParquetRecordBatchReaderBuilder::try_new(f)
                .unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
                .metadata()
                .file_metadata()
                .num_rows() as usize
        })
        .sum()
}

/// Vector width of a parquet dataset, from its first row.
fn parquet_dim(shards: &[PathBuf]) -> usize {
    let first = shards.first().expect("at least one shard");
    let f = File::open(first).unwrap_or_else(|e| panic!("open {}: {e}", first.display()));
    let mut reader = ParquetRecordBatchReaderBuilder::try_new(f)
        .and_then(|b| b.with_batch_size(1).build())
        .unwrap_or_else(|e| panic!("read {}: {e}", first.display()));
    let batch = reader
        .next()
        .expect("shard has at least one row")
        .unwrap_or_else(|e| panic!("decode {}: {e}", first.display()));
    parquet_vector_rows(&batch)
        .first()
        .map(Vec::len)
        .expect("non-empty batch")
}

/// The vector column of one batch as owned fp32 rows. Accepts the list
/// encodings these datasets use and downcasts f64 values (the
/// dbpedia-entities-openai-1M column is `list<double>`, 12 KB a row on the
/// wire) to the f32 the engine ingests.
fn parquet_vector_rows(batch: &RecordBatch) -> Vec<Vec<f32>> {
    let col = PARQUET_VECTOR_COLUMNS
        .iter()
        .find_map(|name| batch.column_by_name(name))
        .unwrap_or_else(|| {
            panic!(
                "parquet batch has none of the vector columns {PARQUET_VECTOR_COLUMNS:?}; \
                 columns are {:?}",
                batch
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
            )
        });
    let values_at = |values: &dyn Array, start: usize, len: usize| -> Vec<f32> {
        if let Some(f) = values.as_any().downcast_ref::<Float32Array>() {
            (start..start + len).map(|i| f.value(i)).collect()
        } else if let Some(d) = values.as_any().downcast_ref::<Float64Array>() {
            (start..start + len).map(|i| d.value(i) as f32).collect()
        } else {
            panic!(
                "vector values must be f32 or f64, got {:?}",
                values.data_type()
            )
        }
    };
    if let Some(l) = col.as_any().downcast_ref::<ListArray>() {
        (0..l.len())
            .map(|i| {
                let (s, e) = (
                    l.value_offsets()[i] as usize,
                    l.value_offsets()[i + 1] as usize,
                );
                values_at(l.values().as_ref(), s, e - s)
            })
            .collect()
    } else if let Some(l) = col.as_any().downcast_ref::<LargeListArray>() {
        (0..l.len())
            .map(|i| {
                let (s, e) = (
                    l.value_offsets()[i] as usize,
                    l.value_offsets()[i + 1] as usize,
                );
                values_at(l.values().as_ref(), s, e - s)
            })
            .collect()
    } else if let Some(l) = col.as_any().downcast_ref::<FixedSizeListArray>() {
        let w = l.value_length() as usize;
        (0..l.len())
            .map(|i| values_at(l.values().as_ref(), i * w, w))
            .collect()
    } else {
        panic!("unsupported vector encoding {:?}", col.data_type())
    }
}

impl MmapVectorCorpus {
    /// Materialize the first `n_docs` rows of a parquet dataset into the
    /// standard corpus mmap, streaming batch by batch so peak memory is one
    /// batch. `dim` is explicit for the same reason as [`Self::from_hdf5`].
    pub fn from_parquet_shards(
        shards: &[PathBuf],
        n_docs: usize,
        dim: usize,
        normalize_each: bool,
    ) -> IoResult<Self> {
        let tmp = TempDir::new().expect("create MmapVectorCorpus tempdir");
        let path = tmp.path().join("corpus.bin");
        let mut writer = BufWriter::new(File::create(&path)?);
        let mut written = 0usize;
        'outer: for shard in shards {
            let f = File::open(shard)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(f)
                .and_then(|b| b.with_batch_size(PARQUET_BATCH_ROWS).build())
                .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
            for batch in reader {
                let batch = batch.map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;
                for mut row in parquet_vector_rows(&batch) {
                    if row.len() != dim {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            format!("row width {} != {dim} in {}", row.len(), shard.display()),
                        ));
                    }
                    if normalize_each {
                        normalize(&mut row);
                    }
                    let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
                    writer.write_all(&bytes)?;
                    written += 1;
                    if written == n_docs {
                        break 'outer;
                    }
                }
            }
        }
        if written < n_docs {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("parquet dataset holds {written} rows; {n_docs} requested"),
            ));
        }
        writer.flush()?;
        writer
            .into_inner()
            .map_err(|e| e.into_error())?
            .sync_all()?;
        Self::from_file_with_dim(File::open(&path)?, n_docs, dim, Some(tmp))
    }
}

/// Held-out query rows for a parquet dataset: the rows immediately PAST
/// the ingested prefix. They are in the dataset and never in the table, so
/// they are genuinely held out rather than self-queries, and no dataset
/// needs to ship a separate query file.
pub fn parquet_queries(
    source: &CorpusSource,
    n_docs: usize,
    n_queries: usize,
    normalize_each: bool,
) -> Vec<Vec<f32>> {
    let shards = parquet_shards_for(source);
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_queries);
    let mut seen = 0usize;
    for shard in &shards {
        let f = File::open(shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
        let reader = ParquetRecordBatchReaderBuilder::try_new(f)
            .and_then(|b| b.with_batch_size(PARQUET_BATCH_ROWS).build())
            .unwrap_or_else(|e| panic!("read {}: {e}", shard.display()));
        for batch in reader {
            for mut row in parquet_vector_rows(&batch.expect("decode batch")) {
                if seen >= n_docs {
                    if out.len() == n_queries {
                        return out;
                    }
                    if normalize_each {
                        normalize(&mut row);
                    }
                    out.push(row);
                }
                seen += 1;
            }
        }
    }
    assert!(
        !out.is_empty(),
        "no rows past {n_docs} to query with; ingest fewer rows or use a deeper dataset"
    );
    out
}

/// The dataset's own query vectors (`test`), first `n_queries` in file
/// order — the published query set, so the neighbour ids below line up with
/// them by index.
pub fn real_queries(
    dir: &Path,
    slug: &str,
    n_queries: usize,
    normalize_each: bool,
) -> Vec<Vec<f32>> {
    let file = open_h5(dir, slug);
    let (rows, _) = h5_shape(&file, H5_TEST);
    let take = n_queries.min(rows);
    let ds = file.dataset(H5_TEST).expect("test dataset");
    let slab: ndarray::Array2<f32> = ds
        .read_slice_2d(ndarray::s![0..take, ..])
        .unwrap_or_else(|e| panic!("read {H5_TEST}: {e}"));
    slab.outer_iter()
        .map(|r| {
            let mut v = r.to_vec();
            if normalize_each {
                normalize(&mut v);
            }
            v
        })
        .collect()
}

/// The dataset's OFFICIAL top-`k` neighbour ids for the first `n_queries`
/// queries, as corpus row indices. Published ground truth: no brute-force
/// recompute, and recall computed against it is directly comparable to
/// ann-benchmarks numbers.
///
/// **Only valid for a FULL-corpus ingest.** The published neighbours are the
/// true top-k over every row of the dataset; on a truncated corpus a query's
/// true neighbours are generally different rows, so the published list is
/// simply the wrong answer — dropping the out-of-range ids would leave a
/// short list that silently overstates recall. Callers ingesting a subset
/// must compute ground truth over the rows they actually ingested
/// ([`brute_force_topk`]). This asserts rather than degrading, because a
/// wrong ground truth produces a plausible number that is not measuring
/// anything.
pub fn real_ground_truth(
    dir: &Path,
    slug: &str,
    n_queries: usize,
    k: usize,
    n_docs: usize,
) -> Vec<Vec<u32>> {
    let rows = real_corpus_rows(dir, slug);
    assert_eq!(
        n_docs, rows,
        "official ground truth for {slug} covers all {rows} rows; it cannot grade a \
         {n_docs}-row ingest — recompute truth over the ingested rows instead"
    );
    let file = open_h5(dir, slug);
    let (rows, cols) = h5_shape(&file, H5_NEIGHBORS);
    let take = n_queries.min(rows);
    assert!(
        k <= cols,
        "dataset {slug} ships {cols} neighbours per query; k={k} requested"
    );
    let ds = file.dataset(H5_NEIGHBORS).expect("neighbors dataset");
    let slab: ndarray::Array2<i32> = ds
        .read_slice_2d(ndarray::s![0..take, 0..k])
        .unwrap_or_else(|e| panic!("read {H5_NEIGHBORS}: {e}"));
    slab.outer_iter()
        .map(|r| {
            r.iter()
                .filter(|&&id| id >= 0 && (id as usize) < n_docs)
                .map(|&id| id as u32)
                .collect()
        })
        .collect()
}

/// The bench query battery for the MAIN vector corpus: dataset test
/// queries for a real corpus, perturbed corpus members for the
/// synthetic one. Diagnostics that plant their own private corpora keep
/// calling [`generate_realistic_queries`] directly.
pub fn bench_queries(
    vectors: &[f32],
    n_docs: usize,
    n_queries: usize,
    seed: u64,
    normalize_each: bool,
    sigma: f32,
) -> Vec<Vec<f32>> {
    match corpus_source() {
        CorpusSource::Synthetic => {
            generate_realistic_queries(vectors, n_docs, n_queries, seed, normalize_each, sigma)
        }
        CorpusSource::AnnBenchmarks { dir, slug } => {
            real_queries(dir, slug, n_queries, normalize_each)
        }
        // Parquet sources hold back the rows past the ingested prefix, so
        // `n_docs` must be the ingested count, not the dataset's size.
        source => parquet_queries(source, n_docs, n_queries, normalize_each),
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
            let mut q: Vec<f32> = (0..dim())
                .map(|_| {
                    let s: f64 = dist.sample(&mut rng);
                    (s as f32) * QUERY_GAUSSIAN_SCALE
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
        let base_idx = (i * QUERY_BASE_DOC_STRIDE) % n_docs;
        let off = base_idx * dim();
        let mut q: Vec<f32> = (0..dim())
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

/// Brute-force kNN ground truth for any [`Metric`]. Returns top-k local
/// doc_ids (no distances — recall only needs the id set).
pub fn brute_force_topk(
    vectors: &[f32],
    n_docs: usize,
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<u32> {
    assert_eq!(vectors.len(), n_docs * dim());
    assert_eq!(query.len(), dim());
    let mut scored: Vec<(u32, f32)> = (0..n_docs as u32)
        .map(|i| {
            let off = (i as usize) * dim();
            (i, distance(metric, query, &vectors[off..off + dim()]))
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Brute-force kNN ground truth for cosine distance on L2-normalized
/// vectors. Returns top-k local doc_ids (no distances — recall only
/// needs the id set).
pub fn brute_force_topk_cosine(
    vectors: &[f32],
    n_docs: usize,
    query: &[f32],
    k: usize,
) -> Vec<u32> {
    assert_eq!(vectors.len(), n_docs * dim());
    assert_eq!(query.len(), dim());
    // For L2-normalized inputs cosine distance is monotone in -dot.
    let mut scored: Vec<(u32, f32)> = (0..n_docs as u32)
        .map(|i| {
            let off = (i as usize) * dim();
            let mut dot = 0f32;
            for d in 0..dim() {
                dot += vectors[off + d] * query[d];
            }
            (i, -dot)
        })
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored.into_iter().map(|(i, _)| i).collect()
}

/// Docs per parallel work unit in the transposed ground-truth pass —
/// big enough to amortize per-chunk heap setup, small enough to
/// load-balance the tail across the rayon pool.
const GT_DOC_CHUNK: usize = 8192;
/// Number of progress updates emitted by a large exact-oracle pass.
const GT_PROGRESS_STEPS: usize = 20;
/// Small oracles finish quickly enough that progress output is noise.
const GT_PROGRESS_MIN_DOCS: usize = 1_000_000;

type GroundTruthCandidate = (f32, u32);

/// Exact top-k labels for the three logical corpus views exercised by the
/// supertable vector lifecycle.
#[derive(Debug)]
pub struct LifecycleGroundTruth {
    pub base: Vec<Vec<u32>>,
    pub filtered: Vec<Vec<u32>>,
    pub augmented: Vec<Vec<u32>>,
}

struct LifecycleTopLists {
    base: Vec<Vec<GroundTruthCandidate>>,
    filtered: Vec<Vec<GroundTruthCandidate>>,
    augmented: Vec<Vec<GroundTruthCandidate>>,
}

impl LifecycleTopLists {
    fn empty(n_queries: usize, n_correctness_queries: usize) -> Self {
        Self {
            base: vec![Vec::new(); n_queries],
            filtered: vec![Vec::new(); n_correctness_queries],
            augmented: vec![Vec::new(); n_correctness_queries],
        }
    }

    fn merge(self, other: Self, k: usize) -> Self {
        Self {
            base: merge_ground_truth_tops(self.base, other.base, k),
            filtered: merge_ground_truth_tops(self.filtered, other.filtered, k),
            augmented: merge_ground_truth_tops(self.augmented, other.augmented, k),
        }
    }
}

fn compare_ground_truth_candidates(a: &GroundTruthCandidate, b: &GroundTruthCandidate) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

fn insert_ground_truth_candidate(
    top: &mut Vec<GroundTruthCandidate>,
    candidate: GroundTruthCandidate,
    k: usize,
) {
    if top.len() == k
        && compare_ground_truth_candidates(
            &candidate,
            top.last().expect("ground-truth top-k is non-empty"),
        )
        .is_ge()
    {
        return;
    }
    let position =
        top.partition_point(|entry| compare_ground_truth_candidates(entry, &candidate).is_lt());
    top.insert(position, candidate);
    top.truncate(k);
}

fn merge_ground_truth_tops(
    mut accumulated: Vec<Vec<GroundTruthCandidate>>,
    partial: Vec<Vec<GroundTruthCandidate>>,
    k: usize,
) -> Vec<Vec<GroundTruthCandidate>> {
    for (top, candidates) in accumulated.iter_mut().zip(partial) {
        top.extend(candidates);
        top.sort_unstable_by(compare_ground_truth_candidates);
        top.truncate(k);
    }
    accumulated
}

fn ground_truth_ids(tops: Vec<Vec<GroundTruthCandidate>>) -> Vec<Vec<u32>> {
    tops.into_iter()
        .map(|top| top.into_iter().map(|(_, id)| id).collect())
        .collect()
}

fn transpose_queries(queries: &[Vec<f32>]) -> Vec<f32> {
    let mut transposed = vec![0.0; dim() * queries.len()];
    for (query_index, query) in queries.iter().enumerate() {
        assert_eq!(query.len(), dim());
        for (dimension, value) in query.iter().enumerate() {
            transposed[dimension * queries.len() + query_index] = *value;
        }
    }
    transposed
}

fn report_ground_truth_progress(
    processed: &AtomicUsize,
    next_report: &AtomicUsize,
    total: usize,
    stride: usize,
    chunk_docs: usize,
) {
    let done = processed.fetch_add(chunk_docs, AtomicOrdering::Relaxed) + chunk_docs;
    loop {
        let threshold = next_report.load(AtomicOrdering::Relaxed);
        if done < threshold {
            return;
        }
        let next = threshold.saturating_add(stride);
        if next_report
            .compare_exchange(
                threshold,
                next,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            )
            .is_ok()
        {
            let bounded = done.min(total);
            eprintln!(
                "[vector ground truth] {bounded}/{total} docs ({:.0}%)",
                bounded as f64 * 100.0 / total as f64
            );
            return;
        }
    }
}

/// Brute-force exact top-k for a whole query batch in ONE streaming
/// pass over the corpus.
///
/// The loop is transposed (doc-major): every doc's vector is scored
/// against all queries while its bytes are hot, with one bounded
/// top-k list per query. At bench scale the corpus is an mmap many
/// times larger than RAM, so the naive per-query loop costs
/// O(queries × corpus_bytes) of page traffic — 7.7 TB of reads for
/// 100 queries over a 50M×384 corpus, hours of wall time. The
/// transpose makes it O(corpus_bytes) total, regardless of batch
/// size. Ties break toward the lower doc id (the per-query reference
/// kernel leaves tie order unspecified); equality with the reference
/// is pinned by `transposed_ground_truth_matches_reference`.
pub fn ground_truth(
    vectors: &[f32],
    n_docs: usize,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<u32>> {
    assert_eq!(vectors.len(), n_docs * dim());
    if queries.is_empty() || n_docs == 0 || k == 0 {
        return vec![Vec::new(); queries.len()];
    }

    let tops = vectors
        .par_chunks(GT_DOC_CHUNK * dim())
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base = (chunk_idx * GT_DOC_CHUNK) as u32;
            let mut tops: Vec<Vec<GroundTruthCandidate>> =
                vec![Vec::with_capacity(k + 1); queries.len()];
            for (j, doc) in chunk.chunks_exact(dim()).enumerate() {
                let id = base + j as u32;
                for (top, q) in tops.iter_mut().zip(queries) {
                    let mut dot = 0f32;
                    for d in 0..dim() {
                        dot += doc[d] * q[d];
                    }
                    insert_ground_truth_candidate(top, (-dot, id), k);
                }
            }
            tops
        })
        .reduce(
            || vec![Vec::new(); queries.len()],
            |accumulated, partial| merge_ground_truth_tops(accumulated, partial, k),
        );
    ground_truth_ids(tops)
}

/// Compute exact base, filtered, and post-delta top-k labels in one
/// document-major pass.
///
/// Every base row is scored once against all `queries`; delta-only rows are
/// scored only against the first `n_correctness_queries`, because calibration
/// never runs against the augmented corpus. The same score updates the
/// relevant base, filtered, and augmented top-k lists, avoiding the previous
/// two full corpus passes plus one filtered pass.
pub fn lifecycle_ground_truth(
    vectors: &[f32],
    n_docs: usize,
    augmented_docs: usize,
    queries: &[Vec<f32>],
    n_correctness_queries: usize,
    filter_keep_every: usize,
    k: usize,
) -> LifecycleGroundTruth {
    assert_eq!(vectors.len(), augmented_docs * dim());
    assert!(n_docs <= augmented_docs);
    assert!(augmented_docs <= u32::MAX as usize);
    assert!(n_correctness_queries <= queries.len());
    assert!(filter_keep_every > 0);
    if queries.is_empty() || augmented_docs == 0 || k == 0 {
        return LifecycleGroundTruth {
            base: vec![Vec::new(); queries.len()],
            filtered: vec![Vec::new(); n_correctness_queries],
            augmented: vec![Vec::new(); n_correctness_queries],
        };
    }

    let transposed_queries = transpose_queries(queries);
    let processed = AtomicUsize::new(0);
    let progress_stride = (augmented_docs >= GT_PROGRESS_MIN_DOCS)
        .then(|| augmented_docs.div_ceil(GT_PROGRESS_STEPS));
    let next_report = AtomicUsize::new(progress_stride.unwrap_or(usize::MAX));
    let tops = vectors
        .par_chunks(GT_DOC_CHUNK * dim())
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let base = chunk_index * GT_DOC_CHUNK;
            let mut tops = LifecycleTopLists::empty(queries.len(), n_correctness_queries);
            let mut dots = vec![0.0f32; queries.len()];
            for (local_doc, doc) in chunk.chunks_exact(dim()).enumerate() {
                let doc_id = base + local_doc;
                let query_count = if doc_id < n_docs {
                    queries.len()
                } else {
                    n_correctness_queries
                };
                dots[..query_count].fill(0.0);
                for (dimension, value) in doc.iter().enumerate() {
                    let query_offset = dimension * queries.len();
                    for (dot, query_value) in dots[..query_count]
                        .iter_mut()
                        .zip(&transposed_queries[query_offset..query_offset + query_count])
                    {
                        *dot += *value * *query_value;
                    }
                }
                let id = doc_id as u32;
                for (query_index, dot) in dots[..query_count].iter().enumerate() {
                    let candidate = (-*dot, id);
                    if doc_id < n_docs {
                        insert_ground_truth_candidate(&mut tops.base[query_index], candidate, k);
                        if query_index < n_correctness_queries
                            && doc_id.is_multiple_of(filter_keep_every)
                        {
                            insert_ground_truth_candidate(
                                &mut tops.filtered[query_index],
                                candidate,
                                k,
                            );
                        }
                    }
                    if query_index < n_correctness_queries {
                        insert_ground_truth_candidate(
                            &mut tops.augmented[query_index],
                            candidate,
                            k,
                        );
                    }
                }
            }
            if let Some(stride) = progress_stride {
                report_ground_truth_progress(
                    &processed,
                    &next_report,
                    augmented_docs,
                    stride,
                    chunk.len() / dim(),
                );
            }
            tops
        })
        .reduce(
            || LifecycleTopLists::empty(queries.len(), n_correctness_queries),
            |accumulated, partial| accumulated.merge(partial, k),
        );

    LifecycleGroundTruth {
        base: ground_truth_ids(tops.base),
        filtered: ground_truth_ids(tops.filtered),
        augmented: ground_truth_ids(tops.augmented),
    }
}

/// Brute-force *filtered* top-k: exact nearest neighbors by NegDot, restricted
/// to the rows in `allow`. Shared by the superfile and supertable vector benches
/// so both compute the filtered ground truth identically (same sort key + tie
/// break as [`ground_truth`]).
pub fn filtered_ground_truth(
    vectors: &[f32],
    allow: &RoaringBitmap,
    queries: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u32)> = allow
                .iter()
                .map(|id| {
                    let row = &vectors[id as usize * dim()..(id as usize + 1) * dim()];
                    let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                    (-dot, id)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            dists.truncate(k);
            dists.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

/// Recall@k between a predicted top-k id list and ground truth.
pub fn recall_at_k(predicted: &[Hit], truth: &[u32]) -> f32 {
    if truth.is_empty() {
        return EMPTY_TRUTH_RECALL;
    }
    let truth_set: std::collections::HashSet<u32> = truth.iter().copied().collect();
    let hits = predicted
        .iter()
        .filter(|(id, _)| truth_set.contains(id))
        .count();
    hits as f32 / truth.len() as f32
}

/// Stable `_id` + score pairs from public [`SupertableReader::vector_search`]
/// batches (`projection = None` → `_id` + `score`).
pub fn id_scores_from_vector_search(batches: &[RecordBatch]) -> Vec<(i128, f32)> {
    let mut out = Vec::new();
    for batch in batches {
        let id_idx = batch.schema().index_of("_id").unwrap_or(0);
        let score_idx = batch.schema().index_of("score").unwrap_or(1);
        let ids = batch
            .column(id_idx)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id column is Decimal128");
        let scores = batch
            .column(score_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score column is Float32");
        for i in 0..batch.num_rows() {
            out.push((ids.value(i), scores.value(i)));
        }
    }
    out
}

/// Recall@k on stable table `_id`s (minted `i128` values).
pub fn recall_at_k_stable(predicted: &[(i128, f32)], truth: &[i128]) -> f32 {
    if truth.is_empty() {
        return EMPTY_TRUTH_RECALL;
    }
    let truth_set: std::collections::HashSet<i128> = truth.iter().copied().collect();
    let hits = predicted
        .iter()
        .filter(|(id, _)| truth_set.contains(id))
        .count();
    hits as f32 / truth.len() as f32
}

/// Map `vector_search` `_id` values to dense oracle row indices.
///
/// `_id`s are minted sequentially at `append()` (one snowflake generator per
/// handle, buffer order), so ascending `_id` order IS ingest order: the row at
/// position `d` of `ORDER BY _id` is corpus row `d`. The ORDER BY is load-
/// bearing — an unordered scan returns DataFusion partitions interleaved.
pub fn engine_id_to_dense(
    table: &infino::supertable::Supertable,
    n_docs: usize,
) -> std::collections::HashMap<i128, u32> {
    use arrow_array::Decimal128Array;

    let batches = table
        .reader()
        .expect("reader")
        .query_sql("SELECT _id FROM supertable ORDER BY _id")
        .expect("SELECT _id FROM supertable ORDER BY _id");
    let mut ids = Vec::with_capacity(n_docs);
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id column is Decimal128");
        ids.extend(col.values().iter().copied());
    }
    assert_eq!(
        ids.len(),
        n_docs,
        "SELECT _id row count must match ingest doc count"
    );
    ids.into_iter()
        .enumerate()
        .map(|(d, id)| (id, d as u32))
        .collect()
}

/// Brute-force oracle rows are corpus row indices; translate to engine `_id`s.
/// `row_ids[d]` is the engine `_id` for corpus row `d`.
pub fn oracle_to_engine_ids(gt: &[Vec<u32>], row_ids: &[i128]) -> Vec<Vec<i128>> {
    gt.iter()
        .map(|row| row.iter().map(|&d| row_ids[d as usize]).collect())
        .collect()
}

/// Superfile oracle rows are already engine-local ids (`0..n-1`).
pub fn oracle_to_i128(gt: &[Vec<u32>]) -> Vec<Vec<i128>> {
    gt.iter()
        .map(|row| row.iter().map(|&d| i128::from(d)).collect())
        .collect()
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
        let hits =
            block_on_inmem(reader.search("v", q, k, nprobe, rerank_mult)).expect("vector search");
        sum += recall_at_k(&hits, t);
    }
    sum / queries.len() as f32
}

/// Mean recall via production [`SuperfileReader::vector_search`].
pub fn mean_recall_superfile(
    reader: &SuperfileReader,
    column: &str,
    queries: &[Vec<f32>],
    truths: &[Vec<u32>],
    k: usize,
    nprobe: usize,
    rerank_mult: usize,
) -> f32 {
    let opts = VectorSearchOptions::new()
        .with_nprobe(nprobe)
        .with_rerank_mult(rerank_mult);
    let mut sum = 0f32;
    for (q, t) in queries.iter().zip(truths) {
        let hits =
            block_on_inmem(reader.vector_hits_async(column, q, k, opts)).expect("vector_search");
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
        samples.push(t0.elapsed().as_secs_f32() * SEC_TO_MICROS);
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
                    let _ =
                        block_on_inmem(reader.search("v", q, k, probe, refine)).expect("search");
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

/// Sweep `(nprobe, rerank_mult)` values via [`SuperfileReader::vector_search`].
pub fn calibrate_superfile(
    reader: &SuperfileReader,
    column: &str,
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
            let recall = mean_recall_superfile(reader, column, queries, truths, k, probe, refine);
            if recall > peak_recall {
                peak_recall = recall;
            }
            if recall < target_recall {
                continue;
            }
            let q = &queries[0];
            let opts = VectorSearchOptions::new()
                .with_nprobe(probe)
                .with_rerank_mult(refine);
            let p50 = p50_micros(
                || {
                    let _ = block_on_inmem(reader.vector_hits_async(column, q, k, opts))
                        .expect("vector_search");
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
            "    [superfile] no point hit recall ≥ {target_recall:.2}; peak observed = {peak_recall:.3}"
        );
    }
    best
}

// ─── Thin builder wrappers ────────────────────────────────────────────

/// Build a stand-alone FTS index from a token corpus. Wrapper exists so
/// both bench harnesses construct the index identically.
pub fn build_fts_index(docs: &[String]) -> FtsBuilder {
    let mut b = FtsBuilder::new(default_tokenizer());
    b.register_column("title".to_string(), true)
        .expect("register column");
    for (i, text) in docs.iter().enumerate() {
        b.add_doc(0, i as u32, text).expect("add doc");
    }
    b
}

/// Build a stand-alone vector index. `vectors` is flat `n_docs * dim()`.
///
/// Bench harness picks `Sq8` by default to match the on-disk
/// default for production superfiles. Per-cluster scale/offset
/// quantizer is the active codec (drop ≤ 0.04 on the
/// pathological planted-cluster synthetic; expected near-zero on
/// real embeddings). Callers measuring the Fp32 baseline (recall
/// oracles, bit-exact regression tests) construct their own
/// `VectorConfig` with `RerankCodec::Fp32`.
pub fn build_vector_index(vectors: &[f32], n_docs: usize, metric: Metric) -> VectorBuilder {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        provided_centroids: None,
        column: "v".into(),
        dim: dim(),
        rot_seed: ROT_SEED,
        metric,
        rerank_codec: bench_rerank_codec(metric),
    })
    .expect("register column");
    for i in 0..n_docs {
        let off = i * dim();
        b.add(0, &vectors[off..off + dim()])
            .expect("add to vector builder");
    }
    b
}

/// Open a built vector blob as a reader. Encodes the directory JSON
/// inline so callers don't reinvent it.
pub fn open_vector_reader(blob: Vec<u8>, metric: Metric) -> VectorReader {
    let metric_str = match metric {
        Metric::L2Sq => "l2sq",
        Metric::Cosine => "cosine",
        Metric::NegDot => "negdot",
    };
    let json = format!(
        r#"[{{"column":"v","dim":{},"rot_seed":7,"metric":"{metric_str}"}}]"#,
        dim()
    );
    VectorReader::open_with(Bytes::from(blob), &json, OpenOptions { verify_crc: true })
        .expect("open VectorReader")
}

/// Build a full superfile (FTS + vector) for end-to-end benches.
pub fn build_superfile(docs: &[String], vectors: &[f32]) -> Vec<u8> {
    let n = docs.len();
    // `SuperfileBuilder` requires the id column to be
    // `Decimal128(38, 0)` (the supertable's snowflake id type), not
    // `UInt64` — match it so this helper actually builds.
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![SfVectorConfig {
            provided_centroids: None,
            column: "emb".into(),
            dim: dim(),
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: bench_rerank_codec(Metric::Cosine),
        }],
        Some(default_tokenizer()),
    );

    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids: Decimal128Array = (0..n as u64)
        .map(|i| Some(i as i128))
        .collect::<Decimal128Array>()
        .with_precision_and_scale(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE)
        .expect("decimal128 with_precision_and_scale");
    let titles = LargeStringArray::from(docs.iter().map(String::as_str).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[vectors]).expect("add_batch");
    b.finish().expect("finish builder")
}

/// Build a full superfile (FTS + vector) with an explicit metric.
pub fn build_superfile_with_metric(docs: &[String], vectors: &[f32], metric: Metric) -> Vec<u8> {
    let n = docs.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![SfVectorConfig {
            provided_centroids: None,
            column: "emb".into(),
            dim: dim(),
            rot_seed: ROT_SEED,
            metric,
            rerank_codec: bench_rerank_codec(metric),
        }],
        Some(default_tokenizer()),
    );

    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids: Decimal128Array = (0..n as u64)
        .map(|i| Some(i as i128))
        .collect::<Decimal128Array>()
        .with_precision_and_scale(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE)
        .expect("decimal128 with_precision_and_scale");
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

#[cfg(test)]
mod tests {
    use std::{fs, io::ErrorKind};

    use rand::RngExt;

    use super::*;

    /// Corpus size for the oracle-equivalence test — a few parallel
    /// chunks' worth so the chunked/merged path is exercised.
    const GT_TEST_DOCS: usize = 3 * GT_DOC_CHUNK + 17;
    /// Query batch size for the oracle-equivalence test.
    const GT_TEST_QUERIES: usize = 7;
    /// Top-k for the oracle-equivalence test.
    const GT_TEST_K: usize = 10;
    /// Seed for the test's corpus + queries.
    const GT_TEST_SEED: u64 = 42;
    /// Base corpus rows for the lifecycle-oracle equivalence test.
    const LIFECYCLE_GT_TEST_DOCS: usize = 512;
    /// Post-delta corpus rows for the lifecycle-oracle equivalence test.
    const LIFECYCLE_GT_TEST_AUGMENTED_DOCS: usize = 576;
    /// Correctness-query prefix graded against filtered and augmented views.
    const LIFECYCLE_GT_TEST_CORRECTNESS_QUERIES: usize = 3;
    /// Deterministic allow-set stride for the filtered lifecycle view.
    const LIFECYCLE_GT_TEST_FILTER_KEEP_EVERY: usize = 7;
    /// Rows in the persisted mmap open/size validation fixture.
    const MMAP_OPEN_TEST_DOCS: usize = 3;
    /// First global row generated by the range-equivalence fixture.
    const MMAP_RANGE_TEST_START: usize = 13;
    /// Rows generated by the range-equivalence fixture.
    const MMAP_RANGE_TEST_DOCS: usize = 19;
    /// Planted centers in the range-equivalence fixture.
    const MMAP_RANGE_TEST_CENTERS: usize = 8;
    /// Corpus seed in the range-equivalence fixture.
    const MMAP_RANGE_TEST_SEED: u64 = 91;

    #[test]
    fn mmap_vector_corpus_opens_only_the_exact_expected_size() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = directory.path().join("vectors.bin");
        let values: Vec<f32> = (0..MMAP_OPEN_TEST_DOCS * dim())
            .map(|value| value as f32)
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&values);
        fs::write(&path, bytes).expect("write exact corpus");

        let corpus = MmapVectorCorpus::open(&path, MMAP_OPEN_TEST_DOCS).expect("open exact corpus");
        assert_eq!(corpus.n_docs(), MMAP_OPEN_TEST_DOCS);
        assert_eq!(corpus.dim(), dim());
        assert_eq!(corpus.as_slice(), values);

        let wrong_path = directory.path().join("wrong-size.bin");
        fs::write(&wrong_path, &bytes[..bytes.len() - 1]).expect("write wrong-size corpus");
        let wrong_size = MmapVectorCorpus::open(&wrong_path, MMAP_OPEN_TEST_DOCS)
            .err()
            .expect("wrong-size corpus must fail");
        assert_eq!(wrong_size.kind(), ErrorKind::InvalidData);

        let missing =
            MmapVectorCorpus::open(&directory.path().join("missing.bin"), MMAP_OPEN_TEST_DOCS)
                .err()
                .expect("missing corpus must fail");
        assert_eq!(missing.kind(), ErrorKind::NotFound);
    }

    #[test]
    fn mmap_vector_corpus_generated_range_matches_full_corpus_slice() {
        let full_docs = MMAP_RANGE_TEST_START + MMAP_RANGE_TEST_DOCS;
        let full = MmapVectorCorpus::generate(
            full_docs,
            MMAP_RANGE_TEST_CENTERS,
            MMAP_RANGE_TEST_SEED,
            true,
        );
        let tail = MmapVectorCorpus::generate_range(
            MMAP_RANGE_TEST_START,
            MMAP_RANGE_TEST_DOCS,
            MMAP_RANGE_TEST_CENTERS,
            MMAP_RANGE_TEST_SEED,
            true,
        );
        let start = MMAP_RANGE_TEST_START * dim();
        let end = full_docs * dim();
        let tail_bytes: &[u8] = bytemuck::cast_slice(tail.as_slice());
        let expected_bytes: &[u8] = bytemuck::cast_slice(&full.as_slice()[start..end]);

        assert_eq!(tail.n_docs(), MMAP_RANGE_TEST_DOCS);
        assert_eq!(tail_bytes, expected_bytes);
    }

    #[test]
    fn transposed_ground_truth_matches_reference() {
        let mut rng = StdRng::seed_from_u64(GT_TEST_SEED);
        let mut vectors = vec![0f32; GT_TEST_DOCS * dim()];
        for v in vectors.iter_mut() {
            *v = rng.random::<f32>() - 0.5;
        }
        let queries: Vec<Vec<f32>> = (0..GT_TEST_QUERIES)
            .map(|_| (0..dim()).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();

        let transposed = ground_truth(&vectors, GT_TEST_DOCS, &queries, GT_TEST_K);
        for (q, got) in queries.iter().zip(&transposed) {
            let reference = brute_force_topk_cosine(&vectors, GT_TEST_DOCS, q, GT_TEST_K);
            assert_eq!(
                got, &reference,
                "transposed oracle diverged from the per-query reference"
            );
        }
    }

    #[test]
    fn lifecycle_ground_truth_matches_three_reference_oracles() {
        let mut rng = StdRng::seed_from_u64(GT_TEST_SEED);
        let mut vectors = vec![0f32; LIFECYCLE_GT_TEST_AUGMENTED_DOCS * dim()];
        for value in &mut vectors {
            *value = rng.random::<f32>() - 0.5;
        }
        let queries: Vec<Vec<f32>> = (0..GT_TEST_QUERIES)
            .map(|_| (0..dim()).map(|_| rng.random::<f32>() - 0.5).collect())
            .collect();
        let base_vectors = &vectors[..LIFECYCLE_GT_TEST_DOCS * dim()];
        let base = ground_truth(base_vectors, LIFECYCLE_GT_TEST_DOCS, &queries, GT_TEST_K);
        let mut allow = RoaringBitmap::new();
        for id in (0..LIFECYCLE_GT_TEST_DOCS as u32).step_by(LIFECYCLE_GT_TEST_FILTER_KEEP_EVERY) {
            allow.insert(id);
        }
        let filtered = filtered_ground_truth(
            base_vectors,
            &allow,
            &queries[..LIFECYCLE_GT_TEST_CORRECTNESS_QUERIES],
            GT_TEST_K,
        );
        let augmented = ground_truth(
            &vectors,
            LIFECYCLE_GT_TEST_AUGMENTED_DOCS,
            &queries[..LIFECYCLE_GT_TEST_CORRECTNESS_QUERIES],
            GT_TEST_K,
        );

        let combined = lifecycle_ground_truth(
            &vectors,
            LIFECYCLE_GT_TEST_DOCS,
            LIFECYCLE_GT_TEST_AUGMENTED_DOCS,
            &queries,
            LIFECYCLE_GT_TEST_CORRECTNESS_QUERIES,
            LIFECYCLE_GT_TEST_FILTER_KEEP_EVERY,
            GT_TEST_K,
        );
        assert_eq!(combined.base, base);
        assert_eq!(combined.filtered, filtered);
        assert_eq!(combined.augmented, augmented);
    }

    /// `from_hdf5` and the query / ground-truth readers round-trip a fixture
    /// this test writes itself, so the write + normalize path is exercised
    /// offline — no network, no staged dataset. `dim` is passed explicitly
    /// for exactly this reason: the process-wide `dim()` resolves once from
    /// env and would pin whatever the first caller in the process asked for.
    #[test]
    fn real_corpus_hdf5_round_trip() {
        const ROWS: usize = 40;
        const TEST_DIM: usize = 4;
        const QUERIES: usize = 6;
        const NEIGHBORS: usize = 5;

        let dir = TempDir::new().expect("fixture dir");
        let slug = "fixture-4-angular";
        let train: Vec<f32> = (0..ROWS * TEST_DIM).map(|i| 1.0 + i as f32).collect();
        let test: Vec<f32> = (0..QUERIES * TEST_DIM).map(|i| 100.0 + i as f32).collect();
        // Includes one out-of-range id to pin the truncated-corpus filter.
        let neigh: Vec<i32> = (0..QUERIES * NEIGHBORS)
            .map(|i| {
                if i == 0 {
                    ROWS as i32 + 1
                } else {
                    (i % ROWS) as i32
                }
            })
            .collect();
        {
            let f = hdf5::File::create(dir.path().join(format!("{slug}.hdf5"))).expect("create");
            f.new_dataset::<f32>()
                .shape([ROWS, TEST_DIM])
                .create(H5_TRAIN)
                .expect("train ds")
                .write(
                    &ndarray::Array2::from_shape_vec((ROWS, TEST_DIM), train.clone())
                        .expect("shape"),
                )
                .expect("write train");
            f.new_dataset::<f32>()
                .shape([QUERIES, TEST_DIM])
                .create(H5_TEST)
                .expect("test ds")
                .write(
                    &ndarray::Array2::from_shape_vec((QUERIES, TEST_DIM), test.clone())
                        .expect("shape"),
                )
                .expect("write test");
            f.new_dataset::<i32>()
                .shape([QUERIES, NEIGHBORS])
                .create(H5_NEIGHBORS)
                .expect("neighbors ds")
                .write(
                    &ndarray::Array2::from_shape_vec((QUERIES, NEIGHBORS), neigh).expect("shape"),
                )
                .expect("write neighbors");
        }

        assert_eq!(real_corpus_rows(dir.path(), slug), ROWS);
        assert_eq!(real_corpus_dim(dir.path(), slug), TEST_DIM);

        // Un-normalized load preserves row order and values exactly.
        let raw =
            MmapVectorCorpus::from_hdf5(dir.path(), slug, ROWS, TEST_DIM, false).expect("load raw");
        assert_eq!(raw.as_slice(), train.as_slice());

        // Normalized load makes every row unit, same order.
        let unit = MmapVectorCorpus::from_hdf5(dir.path(), slug, ROWS, TEST_DIM, true)
            .expect("load normalized");
        for (i, row) in unit.as_slice().chunks_exact(TEST_DIM).enumerate() {
            let norm: f32 = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {i} norm {norm}");
            let want = &train[i * TEST_DIM..(i + 1) * TEST_DIM];
            let want_norm: f32 = want.iter().map(|v| v * v).sum::<f32>().sqrt();
            for (d, v) in row.iter().enumerate() {
                assert!((v - want[d] / want_norm).abs() < 1e-5, "row {i} dim {d}");
            }
        }

        // Row shortfall and dim mismatch are errors, never silent loads.
        assert!(
            MmapVectorCorpus::from_hdf5(dir.path(), slug, ROWS + 1, TEST_DIM, false).is_err(),
            "row shortfall must be an error"
        );
        assert!(
            MmapVectorCorpus::from_hdf5(dir.path(), slug, ROWS, TEST_DIM + 1, false).is_err(),
            "dim mismatch must be an error"
        );

        // Queries in file order, normalized on request.
        let qs = real_queries(dir.path(), slug, 3, false);
        assert_eq!(qs.len(), 3);
        assert_eq!(qs[0], test[..TEST_DIM].to_vec());
        let qn = real_queries(dir.path(), slug, 1, true);
        let norm: f32 = qn[0].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "query norm {norm}");

        // Official ground truth: out-of-corpus ids dropped, rest in range.
        let gt = real_ground_truth(dir.path(), slug, QUERIES, NEIGHBORS, ROWS);
        assert_eq!(gt.len(), QUERIES);
        assert_eq!(
            gt[0].len(),
            NEIGHBORS - 1,
            "the planted out-of-range id must be filtered"
        );
        assert!(gt.iter().flatten().all(|&id| (id as usize) < ROWS));
    }
}
