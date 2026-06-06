// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Deterministic, memory-mapped text corpus for the bench harness.
//!
//! The corpus is generated once into a temp file and memory-mapped, so
//! it lives in file-backed (reclaimable) pages rather than anonymous
//! heap. That keeps the shared input out of each engine's measured
//! resident footprint — an engine's RSS then reflects its own index,
//! not the corpus every engine indexes identically.

use std::fs::File;
use std::io::{BufWriter, Write};

use memmap2::Mmap;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tempfile::TempDir;

/// Body tokens per document (≈ a short article).
pub const TOKENS_PER_DOC: usize = 200;

/// Closed vocabulary size — common terms get long posting lists; rare
/// terms exercise the cold FST + skip-table path.
pub const VOCAB_SIZE: usize = 10_000;

/// Single-segment (superfile) scale.
pub const SUPERFILE_DOCS: usize = 1_000_000;

/// Multi-segment (supertable) scale.
pub const SUPERTABLE_DOCS: usize = 10_000_000;

/// Deterministic Zipfian sampler over `[1, n]` (inverse-CDF, O(log n)
/// per draw).
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
        let total = *self.cum_weights.last().expect("non-empty vocabulary");
        let target: f64 = rng.random::<f64>() * total;
        match self
            .cum_weights
            .binary_search_by(|p| p.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) | Err(i) => i.min(self.cum_weights.len() - 1) + 1,
        }
    }
}

/// Memory-mapped Zipfian text corpus. Each doc is `TOKENS_PER_DOC` body
/// terms from a closed [`VOCAB_SIZE`] vocabulary, prefixed by one
/// doc-unique identifier token (`doc<7-digit-id>`) so a singleton long
/// tail exists for the `df = 1` paths.
pub struct MmapTextCorpus {
    _tmp: TempDir,
    map: Mmap,
    offsets: Vec<u64>,
}

impl MmapTextCorpus {
    /// Generate `n_docs` documents deterministically from `seed`.
    pub fn generate(n_docs: usize, seed: u64) -> Self {
        let tmp = TempDir::new().expect("create MmapTextCorpus tempdir");
        let path = tmp.path().join("corpus.txt");
        let file = File::create(&path).expect("create text corpus file");
        let mut writer = BufWriter::with_capacity(8 << 20, file);
        let mut rng = StdRng::seed_from_u64(seed);
        let zipf = ZipfDistribution::new(VOCAB_SIZE);
        let mut offsets = Vec::with_capacity(n_docs + 1);
        let mut pos = 0u64;
        offsets.push(pos);

        for doc_id in 0..n_docs {
            let token = format!("doc{doc_id:07}");
            writer.write_all(token.as_bytes()).expect("write doc token");
            pos += token.len() as u64;
            for _ in 0..TOKENS_PER_DOC {
                let term = format!(" term{:05}", zipf.sample(&mut rng));
                writer.write_all(term.as_bytes()).expect("write term");
                pos += term.len() as u64;
            }
            offsets.push(pos);
        }

        let file = writer.into_inner().expect("flush text corpus writer");
        file.sync_all().expect("sync text corpus");
        drop(file);

        let file = File::open(&path).expect("reopen text corpus");
        // SAFETY: this helper owns the temp file and never writes to it
        // after the fsync above, so the read-only mmap cannot observe
        // mutation.
        let map = unsafe { Mmap::map(&file).expect("mmap text corpus") };
        Self {
            _tmp: tmp,
            map,
            offsets,
        }
    }

    /// Number of documents.
    pub fn n_docs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Total logical text bytes — the ingest input payload, for build
    /// bandwidth (MB/s) reporting.
    pub fn total_bytes(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0) - self.offsets.first().copied().unwrap_or(0)
    }

    /// Borrow document `idx` as a `&str` into the mmap (no copy).
    pub fn doc(&self, idx: usize) -> &str {
        let start = self.offsets[idx] as usize;
        let end = self.offsets[idx + 1] as usize;
        std::str::from_utf8(&self.map[start..end]).expect("generated corpus is valid UTF-8")
    }

    /// Materialize the whole corpus as `(doc_id, text)` rows borrowing
    /// from the mmap — the input shape the FTS driver feeds to engines.
    /// `doc_id` is the dense row index, so it doubles as the recall id.
    pub fn rows(&self) -> Vec<(u64, &str)> {
        (0..self.n_docs()).map(|i| (i as u64, self.doc(i))).collect()
    }
}
