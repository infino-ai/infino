// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Stream synthetic text + vector rows for supertable ingest (no full-dataset file).

use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};

use crate::corpus::{DIM, TOKENS_PER_DOC, VOCAB_SIZE, ZipfDistribution, normalize};

/// Gaussian scale of a planted cluster center (matches `corpus.rs`).
const CENTER_GAUSSIAN_SCALE: f32 = 3.0;
/// Per-dimension Gaussian noise around a cluster center.
const DOC_NOISE_SIGMA: f32 = 0.3;
/// Average bytes-per-token estimate for pre-sizing a doc `String`.
const AVG_BYTES_PER_TOKEN: usize = 8;

/// Stream the same deterministic synthetic docs as [`super::MmapTextCorpus`] +
/// [`super::MmapVectorCorpus`], one append chunk at a time.
///
/// Advance docs strictly in order (doc 0, 1, 2, …).
pub struct SequentialSyntheticCorpus {
    doc_id: usize,
    vec_rng: StdRng,
    text_rng: StdRng,
    centers: Vec<Vec<f32>>,
    zipf: ZipfDistribution,
    normalize_vectors: bool,
}

impl SequentialSyntheticCorpus {
    pub fn new(n_cent: usize, vec_seed: u64, text_seed: u64, normalize_vectors: bool) -> Self {
        let mut vec_rng = StdRng::seed_from_u64(vec_seed);
        let dist = StandardNormal;
        let centers: Vec<Vec<f32>> = (0..n_cent)
            .map(|_| {
                (0..DIM)
                    .map(|_| {
                        let s: f64 = dist.sample(&mut vec_rng);
                        (s as f32) * CENTER_GAUSSIAN_SCALE
                    })
                    .collect()
            })
            .collect();
        Self {
            doc_id: 0,
            vec_rng,
            text_rng: StdRng::seed_from_u64(text_seed),
            centers,
            zipf: ZipfDistribution::new(VOCAB_SIZE),
            normalize_vectors,
        }
    }

    /// Fill `titles` and `flat` (`len * DIM` elements) for the next `len` docs.
    pub fn fill_chunk(&mut self, len: usize, titles: &mut Vec<String>, flat: &mut Vec<f32>) {
        self.fill_chunk_modality(len, titles, flat, true, true);
    }

    /// Modality-aware fill: generate only the columns the build actually
    /// ingests. A vector-only build does not need the (~2 KB/doc) title
    /// strings, and an FTS-only build does not need the (DIM·4 B/doc) vector
    /// payload. Generating an unused column would (a) burn CPU and (b) sit
    /// resident in the bench process so the whole-process RSS sampler counts
    /// it — neither of which a production server ingesting over the API pays.
    ///
    /// The two RNG streams are independent (`text_rng` vs `vec_rng`), so
    /// skipping one column leaves the other column's bytes bit-identical to a
    /// `true, true` run with the same seeds.
    pub fn fill_chunk_modality(
        &mut self,
        len: usize,
        titles: &mut Vec<String>,
        flat: &mut Vec<f32>,
        gen_text: bool,
        gen_vec: bool,
    ) {
        titles.clear();
        flat.clear();
        if gen_text {
            titles.reserve(len);
        }
        if gen_vec {
            flat.reserve(len.saturating_mul(DIM));
        }
        let dist = StandardNormal;
        let mut row = vec![0.0f32; DIM];
        for _ in 0..len {
            let doc_id = self.doc_id;
            if gen_text {
                let mut doc = String::with_capacity((TOKENS_PER_DOC + 1) * AVG_BYTES_PER_TOKEN);
                doc.push_str(&format!("doc{doc_id:07}"));
                for _ in 0..TOKENS_PER_DOC {
                    let idx = self.zipf.sample(&mut self.text_rng);
                    doc.push(' ');
                    doc.push_str(&format!("term{idx:05}"));
                }
                titles.push(doc);
            }

            if gen_vec {
                let center = &self.centers[doc_id % self.centers.len()];
                for (j, slot) in row.iter_mut().enumerate() {
                    let s: f64 = dist.sample(&mut self.vec_rng);
                    *slot = center[j] + (s as f32) * DOC_NOISE_SIGMA;
                }
                if self.normalize_vectors {
                    normalize(&mut row);
                }
                flat.extend_from_slice(&row);
            }
            self.doc_id += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{MmapTextCorpus, MmapVectorCorpus, n_cent};

    /// Document count for the stream-vs-mmap equivalence fixtures.
    const TEST_N_DOCS: usize = 256;
    /// RNG seed shared by the stream and mmap corpora under test.
    const TEST_CORPUS_SEED: u64 = 1;

    /// Streamed ingest vectors must match [`super::MmapVectorCorpus`].
    #[test]
    fn stream_matches_mmap_vector_corpus() {
        let n_docs = TEST_N_DOCS;
        let n_cent = n_cent(n_docs);
        let mmap = MmapVectorCorpus::generate(n_docs, n_cent, TEST_CORPUS_SEED, true);
        let mut stream =
            SequentialSyntheticCorpus::new(n_cent, TEST_CORPUS_SEED, TEST_CORPUS_SEED, true);
        let mut titles = Vec::new();
        let mut flat = Vec::new();
        stream.fill_chunk(n_docs, &mut titles, &mut flat);
        assert_eq!(flat, mmap.as_slice());
        assert_eq!(titles.len(), n_docs);
        assert!(titles[0].starts_with("doc0000000"));
    }

    /// Streamed ingest text must match [`super::MmapTextCorpus`].
    #[test]
    fn stream_matches_mmap_text_corpus() {
        let n_docs = TEST_N_DOCS;
        let mmap = MmapTextCorpus::generate(n_docs, TEST_CORPUS_SEED);
        let mut stream = SequentialSyntheticCorpus::new(
            n_cent(n_docs),
            TEST_CORPUS_SEED,
            TEST_CORPUS_SEED,
            true,
        );
        let mut titles = Vec::new();
        let mut flat = Vec::new();
        stream.fill_chunk(n_docs, &mut titles, &mut flat);
        assert_eq!(titles.len(), n_docs);
        for (i, doc) in titles.iter().enumerate() {
            assert_eq!(doc.as_str(), mmap.doc(i), "doc {i}");
        }
    }
}
