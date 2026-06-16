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

use std::fs::File;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use arrow_array::{Decimal128Array, LargeStringArray, RecordBatch};
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
use infino::superfile::reader::VectorSearchOptions;
use infino::superfile::vector::builder::{VectorBuilder, VectorConfig};
use infino::superfile::vector::distance::{Metric, distance, normalize};
use infino::superfile::vector::reader::{OpenOptions, VectorReader};
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::test_helpers::default_tokenizer;

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
/// the whole chunk in memory.
const TEXT_CORPUS_CHUNK_DOCS: usize = 1 << 20;
/// Docs assigned to one vector corpus scheduling chunk. Each worker streams
/// its chunk through [`PARALLEL_CORPUS_WRITE_BUF_CAPACITY`] rather than
/// buffering the whole chunk in memory.
const VECTOR_CORPUS_CHUNK_DOCS: usize = 1 << 19;
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
fn chunk_seed(seed: u64, chunk_index: usize) -> u64 {
    seed.wrapping_add(
        (chunk_index as u64)
            .wrapping_add(1)
            .wrapping_mul(CHUNK_SEED_MIX),
    )
}

/// Vector dimension — matches modern large embedding models
/// (OpenAI text-embedding-3-small = 1536 truncates to 1024,
/// BGE-large = 1024, E5-large = 1024).
pub const DIM: usize = 1024;

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
    docs_from_env("INFINO_BENCH_SUPERFILE_DOCS", SUPERFILE_DOCS)
}

/// Document count for the **supertable** test — a multi-superfile table
/// committed to and queried from **object storage**. Defaults to
/// [`SUPERTABLE_DOCS`] (10M); override with
/// `INFINO_BENCH_SUPERTABLE_DOCS`.
pub fn supertable_docs() -> usize {
    docs_from_env("INFINO_BENCH_SUPERTABLE_DOCS", SUPERTABLE_DOCS)
}

/// Parse a positive doc-count override from `var`, falling back to
/// `default` when unset, empty, unparseable, or zero.
fn docs_from_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Parallel-writer count for the "N writers" build row — how many
/// writers build the corpus concurrently. Applied identically to every
/// engine (infino shards across this many builders; Tantivy uses this
/// many indexing threads). Defaults to the machine's logical core count;
/// override with `INFINO_BENCH_WRITERS`.
pub fn parallel_writers() -> usize {
    docs_from_env("INFINO_BENCH_WRITERS", num_cpus::get())
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

pub use combined::SequentialSyntheticCorpus;

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
                    (s as f32) * CENTER_GAUSSIAN_SCALE
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
    _tmp: TempDir,
    map: Mmap,
    n_docs: usize,
    dim: usize,
}

impl MmapVectorCorpus {
    pub fn generate(n_docs: usize, n_cent: usize, seed: u64, normalize_each: bool) -> Self {
        let tmp = TempDir::new().expect("create MmapVectorCorpus tempdir");
        let path = tmp.path().join("corpus.bin");

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
                (0..DIM)
                    .map(|_| {
                        let s: f64 = cdist.sample(&mut crng);
                        (s as f32) * CENTER_GAUSSIAN_SCALE
                    })
                    .collect()
            })
            .collect();

        let row_bytes = DIM * std::mem::size_of::<f32>();
        let total = (n_docs as u64) * (row_bytes as u64);
        let file = File::create(&path).expect("create corpus file");
        file.set_len(total).expect("set_len vector corpus");

        // rayon fans the chunks across all cores; each writes a fixed-stride
        // row range via a positioned write and draws noise from its own
        // deterministic per-chunk RNG (shared centers keep clusters intact).
        let n_chunks = n_docs.div_ceil(VECTOR_CORPUS_CHUNK_DOCS).max(1);
        let centers_ref = &centers;
        let file_ref = &file;
        (0..n_chunks).into_par_iter().for_each(|c| {
            let start = c * VECTOR_CORPUS_CHUNK_DOCS;
            if start >= n_docs {
                return;
            }
            let end = ((c + 1) * VECTOR_CORPUS_CHUNK_DOCS).min(n_docs);
            let mut rng = StdRng::seed_from_u64(chunk_seed(seed, c));
            let dist = StandardNormal;
            let mut buf: Vec<u8> = Vec::with_capacity(PARALLEL_CORPUS_WRITE_BUF_CAPACITY);
            let base = (start as u64) * (row_bytes as u64);
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
            let mut row = vec![0.0f32; DIM];
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

    /// Drop the resident pages backing rows `[start, start + len)`
    /// from this process's RSS — same contract and safety argument as
    /// [`MmapTextCorpus::advise_consumed`].
    pub fn advise_consumed(&self, start: usize, len: usize) {
        let end = (start + len).min(self.n_docs);
        if start >= end {
            return;
        }
        let row_bytes = self.dim * std::mem::size_of::<f32>();
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

/// Brute-force kNN ground truth for any [`Metric`]. Returns top-k local
/// doc_ids (no distances — recall only needs the id set).
pub fn brute_force_topk(
    vectors: &[f32],
    n_docs: usize,
    query: &[f32],
    metric: Metric,
    k: usize,
) -> Vec<u32> {
    assert_eq!(vectors.len(), n_docs * DIM);
    assert_eq!(query.len(), DIM);
    let mut scored: Vec<(u32, f32)> = (0..n_docs as u32)
        .map(|i| {
            let off = (i as usize) * DIM;
            (i, distance(metric, query, &vectors[off..off + DIM]))
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

/// Docs per parallel work unit in the transposed ground-truth pass —
/// big enough to amortize per-chunk heap setup, small enough to
/// load-balance the tail across the rayon pool.
const GT_DOC_CHUNK: usize = 8192;

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
    use rayon::prelude::*;

    assert_eq!(vectors.len(), n_docs * DIM);
    if queries.is_empty() || n_docs == 0 || k == 0 {
        return vec![Vec::new(); queries.len()];
    }

    // Per-query candidate lists sorted ascending by (neg_dot, id) —
    // best first, worst last, at most k entries.
    let better = |a: &(f32, u32), b: &(f32, u32)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
    let merge = |mut acc: Vec<Vec<(f32, u32)>>, part: Vec<Vec<(f32, u32)>>| {
        for (a, p) in acc.iter_mut().zip(part) {
            a.extend(p);
            a.sort_unstable_by(better);
            a.truncate(k);
        }
        acc
    };

    vectors
        .par_chunks(GT_DOC_CHUNK * DIM)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let base = (chunk_idx * GT_DOC_CHUNK) as u32;
            let mut tops: Vec<Vec<(f32, u32)>> = vec![Vec::with_capacity(k + 1); queries.len()];
            for (j, doc) in chunk.chunks_exact(DIM).enumerate() {
                let id = base + j as u32;
                for (top, q) in tops.iter_mut().zip(queries) {
                    let mut dot = 0f32;
                    for d in 0..DIM {
                        dot += doc[d] * q[d];
                    }
                    let cand = (-dot, id);
                    if top.len() == k && better(&cand, top.last().expect("non-empty at k")).is_ge()
                    {
                        continue;
                    }
                    let pos = top.partition_point(|e| better(e, &cand).is_lt());
                    top.insert(pos, cand);
                    top.truncate(k);
                }
            }
            tops
        })
        .reduce(|| vec![Vec::new(); queries.len()], merge)
        .into_iter()
        .map(|top| top.into_iter().map(|(_, id)| id).collect())
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
    b.register_column("title".to_string())
        .expect("register column");
    for (i, text) in docs.iter().enumerate() {
        b.add_doc(0, i as u32, text).expect("add doc");
    }
    b
}

/// Build a stand-alone vector index. `vectors` is flat `n_docs * DIM`.
///
/// Bench harness picks `Sq8` by default to match the on-disk
/// default for production superfiles. Per-cluster scale/offset
/// quantizer is the active codec (drop ≤ 0.04 on the
/// pathological planted-cluster synthetic; expected near-zero on
/// real embeddings). Callers measuring the Fp32 baseline (recall
/// oracles, bit-exact regression tests) construct their own
/// `VectorConfig` with `RerankCodec::Fp32`.
pub fn build_vector_index(
    vectors: &[f32],
    n_docs: usize,
    n_cent: usize,
    metric: Metric,
) -> VectorBuilder {
    let mut b = VectorBuilder::new();
    b.register_column(VectorConfig {
        column: "v".into(),
        dim: DIM,
        n_cent,
        rot_seed: ROT_SEED,
        metric,
        rerank_codec: RerankCodec::Sq8ResidualEpsilon,
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
        r#"[{{"column":"v","dim":{DIM},"n_cent":{n_cent},"rot_seed":7,"metric":"{metric_str}"}}]"#
    );
    VectorReader::open_with(Bytes::from(blob), &json, OpenOptions { verify_crc: true })
        .expect("open VectorReader")
}

/// Build a full superfile (FTS + vector) for end-to-end benches.
pub fn build_superfile(docs: &[String], vectors: &[f32], n_cent: usize) -> Vec<u8> {
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
        }],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
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
pub fn build_superfile_with_metric(
    docs: &[String],
    vectors: &[f32],
    n_cent: usize,
    metric: Metric,
) -> Vec<u8> {
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
        }],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: DIM,
            n_cent,
            rot_seed: ROT_SEED,
            metric,
            rerank_codec: RerankCodec::Sq8ResidualEpsilon,
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

    #[test]
    fn transposed_ground_truth_matches_reference() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(GT_TEST_SEED);
        let mut vectors = vec![0f32; GT_TEST_DOCS * DIM];
        for v in vectors.iter_mut() {
            *v = rng.random::<f32>() - 0.5;
        }
        let queries: Vec<Vec<f32>> = (0..GT_TEST_QUERIES)
            .map(|_| (0..DIM).map(|_| rng.random::<f32>() - 0.5).collect())
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
}
