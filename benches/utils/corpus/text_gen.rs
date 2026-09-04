// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Deterministic text-document generators, shared by every path that
//! needs the bench text corpus: the parallel mmap writer
//! ([`super::MmapTextCorpus`]), the streaming ingest corpus
//! ([`super::SequentialSyntheticCorpus`]), and the FTS quality oracle,
//! which re-derives the corpus doc by doc rather than keeping it.
//!
//! Two flavours share one token vocabulary (`term{rank:05}`, rank 1 the
//! most frequent) and one per-doc unique token (`doc{id:07}`), so the FTS
//! query batteries apply to both:
//!
//! * [`TextFlavor::Uniform`] — the historical corpus: exactly
//!   [`TOKENS_PER_DOC`] iid Zipf draws from a closed [`VOCAB_SIZE`]
//!   vocabulary. Byte length is RNG-independent, which is what lets the
//!   mmap writer pre-compute its offset table and write chunks in parallel.
//!   The perf tables are measured on this flavour.
//! * [`TextFlavor::Realistic`] — calibrated to English Wikipedia so that
//!   the quantities BM25 quality depends on actually vary: log-normal doc
//!   lengths (median 89 tokens, 13% of docs under 16 tokens, a 1% tail
//!   past 3000), an open Zipf vocabulary over [`REALISTIC_VOCAB_SIZE`]
//!   ranks (rank 1 lands in ~84% of docs, like `the`; the tail supplies
//!   the singleton mass), within-doc burstiness (a token repeats one the
//!   doc already used with probability [`REPEAT_PROB`], giving ~56%
//!   repeated tokens), and a sprinkle of surface variation — capitalized
//!   tokens, punctuation glued to a token, year-like digit tokens — so
//!   the analyzer is on the measured path. Everything stays ASCII, so the
//!   `ascii_lower` rule is the only tokenization rule in play.
//!
//! Both flavours are seeded per scheduling chunk with [`chunk_seed`] so a
//! parallel writer and a sequential stream produce identical bytes; the
//! chunk size differs per flavour ([`TextDocGen::chunk_docs`]) because a
//! realistic chunk's bytes must be buffered before they can be written.

use std::fmt::Write as _;

use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, LogNormal};
use rayon::prelude::*;

use super::{TEXT_CORPUS_CHUNK_DOCS, TOKENS_PER_DOC, VOCAB_SIZE, ZipfDistribution, chunk_seed};

/// Which text generator the synthetic corpus uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextFlavor {
    /// Fixed-length, closed-vocabulary Zipf docs (the historical corpus).
    Uniform,
    /// Variable-length, open-vocabulary, bursty docs calibrated to real text.
    Realistic,
}

impl TextFlavor {
    /// The label reports and dataset sidecars carry for this flavour.
    pub fn label(self) -> &'static str {
        match self {
            TextFlavor::Uniform => "synthetic",
            TextFlavor::Realistic => "realistic",
        }
    }
}

/// Docs per scheduling chunk for the realistic flavour. Its byte length
/// is RNG-dependent, so a chunk's bytes are buffered before the writer
/// appends them in order; at ~1.6 KB/doc this bounds a chunk at ~50 MB,
/// and the writer keeps at most two chunks per worker in flight.
pub const REALISTIC_CHUNK_DOCS: usize = 1 << 15;

/// Rank space of the realistic vocabulary. Zipf(s = 1) over one million
/// ranks puts rank 1 at ~7% of tokens (a stopword) and leaves a long tail
/// of ranks that appear in a handful of docs at 10M — the singleton mass
/// real vocabularies carry.
pub const REALISTIC_VOCAB_SIZE: usize = 1_000_000;

/// Log-normal doc-length parameters: `exp(mu)` is the median (89 tokens,
/// Wikipedia's), `sigma` sets the spread (p10 ≈ 12, p90 ≈ 650, p99 ≈ 3300,
/// mean ≈ 295 — all within the calibration bands measured on 300K
/// articles: 7 / 89 / 653 / 2636 / 268).
const LEN_LOG_MEDIAN: f64 = 4.4886; // ln(89)
const LEN_LOG_SIGMA: f64 = 1.55;
/// Doc-length clamp. The lower bound keeps every doc non-empty; the upper
/// bound trims the log-normal's unbounded tail near the longest real
/// articles (35K tokens) so one draw cannot dominate a chunk's bytes.
const MIN_DOC_TOKENS: usize = 1;
const MAX_DOC_TOKENS: usize = 20_000;

/// Probability that the next token repeats one the doc already emitted
/// (a Pólya-urn cache). 0.4 lands the within-doc repeat fraction at 0.56
/// and the rank-1 df at 84%, both matching the measured corpus.
const REPEAT_PROB: f64 = 0.4;

/// Surface-variation thresholds on one uniform draw per token, cumulative:
/// `[0, YEAR)` emits a year-like digit token instead of the term,
/// `[YEAR, CAP)` capitalizes the term, `[CAP, COMMA)` glues a trailing
/// comma, `[COMMA, PERIOD)` a trailing period, the rest is the bare term.
/// Every variant tokenizes back to the same term under `ascii_lower`
/// (case folds, punctuation splits), so df/tf are unchanged while the
/// analyzer does real work.
const YEAR_TOKEN_CUTOFF: f64 = 0.01;
const CAPITALIZED_CUTOFF: f64 = 0.04;
const TRAILING_COMMA_CUTOFF: f64 = 0.06;
const TRAILING_PERIOD_CUTOFF: f64 = 0.07;
/// Year-token range (`1900..=2029`).
const YEAR_MIN: u32 = 1900;
const YEAR_SPAN: u32 = 130;

/// Rough bytes-per-token for pre-sizing a doc buffer.
const AVG_BYTES_PER_TOKEN: usize = 8;

/// One flavour's document generator. Cheap to share across threads
/// (`&self`); all per-stream state lives in the caller's RNG.
pub struct TextDocGen {
    flavor: TextFlavor,
    zipf: ZipfDistribution,
    len_dist: LogNormal<f64>,
}

impl TextDocGen {
    pub fn new(flavor: TextFlavor) -> Self {
        let vocab = match flavor {
            TextFlavor::Uniform => VOCAB_SIZE,
            TextFlavor::Realistic => REALISTIC_VOCAB_SIZE,
        };
        Self {
            flavor,
            zipf: ZipfDistribution::new(vocab),
            len_dist: LogNormal::new(LEN_LOG_MEDIAN, LEN_LOG_SIGMA).expect("log-normal params"),
        }
    }

    pub fn flavor(&self) -> TextFlavor {
        self.flavor
    }

    /// Docs per reseeded scheduling chunk for this flavour. Every producer
    /// of this corpus reseeds its RNG with `chunk_seed(seed, doc_id /
    /// chunk_docs())` at each multiple of this, and nowhere else.
    pub fn chunk_docs(&self) -> usize {
        match self.flavor {
            TextFlavor::Uniform => TEXT_CORPUS_CHUNK_DOCS,
            TextFlavor::Realistic => REALISTIC_CHUNK_DOCS,
        }
    }

    /// The RNG for the chunk containing `doc_id`, positioned at the chunk's
    /// first doc.
    pub fn chunk_rng(&self, seed: u64, chunk_index: usize) -> StdRng {
        StdRng::seed_from_u64(chunk_seed(seed, chunk_index))
    }

    /// Append doc `doc_id`'s text to `out` (which the caller clears),
    /// drawing from `rng`. `rng` must be the chunk RNG advanced through
    /// every earlier doc of the same chunk.
    pub fn write_doc(&self, rng: &mut StdRng, doc_id: usize, out: &mut String) {
        match self.flavor {
            TextFlavor::Uniform => self.write_uniform(rng, doc_id, out),
            TextFlavor::Realistic => self.write_realistic(rng, doc_id, out),
        }
    }

    fn write_uniform(&self, rng: &mut StdRng, doc_id: usize, out: &mut String) {
        out.reserve((TOKENS_PER_DOC + 1) * AVG_BYTES_PER_TOKEN);
        write!(out, "doc{doc_id:07}").expect("fmt doc token");
        for _ in 0..TOKENS_PER_DOC {
            let idx = self.zipf.sample(rng);
            write!(out, " term{idx:05}").expect("fmt term");
        }
    }

    fn write_realistic(&self, rng: &mut StdRng, doc_id: usize, out: &mut String) {
        let drawn = self.len_dist.sample(rng).round();
        let len = (drawn as usize).clamp(MIN_DOC_TOKENS, MAX_DOC_TOKENS);
        out.reserve((len + 1) * AVG_BYTES_PER_TOKEN);
        write!(out, "doc{doc_id:07}").expect("fmt doc token");
        let mut emitted: Vec<u32> = Vec::with_capacity(len);
        for _ in 0..len {
            let rank = if !emitted.is_empty() && rng.random::<f64>() < REPEAT_PROB {
                emitted[rng.random_range(0..emitted.len())]
            } else {
                self.zipf.sample(rng) as u32
            };
            let surface: f64 = rng.random();
            out.push(' ');
            if surface < YEAR_TOKEN_CUTOFF {
                let year = YEAR_MIN + rng.random_range(0..YEAR_SPAN);
                write!(out, "{year}").expect("fmt year");
                continue;
            }
            emitted.push(rank);
            if surface < CAPITALIZED_CUTOFF {
                write!(out, "Term{rank:05}").expect("fmt term");
            } else if surface < TRAILING_COMMA_CUTOFF {
                write!(out, "term{rank:05},").expect("fmt term");
            } else if surface < TRAILING_PERIOD_CUTOFF {
                write!(out, "term{rank:05}.").expect("fmt term");
            } else {
                write!(out, "term{rank:05}").expect("fmt term");
            }
        }
    }
}

/// Number of scheduling chunks an `n_docs` corpus of `flavor` spans.
pub fn generated_chunk_count(n_docs: usize, flavor: TextFlavor) -> usize {
    n_docs.div_ceil(TextDocGen::new(flavor).chunk_docs())
}

/// Visit the docs of one scheduling chunk, in doc order, without
/// materializing them: `f(doc_id, text)` for every doc `chunk` covers.
/// Chunks are independent (each reseeds from [`chunk_seed`]), so callers
/// fan out over `0..generated_chunk_count(..)` and accumulate per chunk.
pub fn for_each_doc_in_chunk<F>(
    n_docs: usize,
    seed: u64,
    flavor: TextFlavor,
    chunk: usize,
    mut f: F,
) where
    F: FnMut(usize, &str),
{
    let generator = TextDocGen::new(flavor);
    let chunk_docs = generator.chunk_docs();
    let start = chunk * chunk_docs;
    let end = ((chunk + 1) * chunk_docs).min(n_docs);
    let mut rng = generator.chunk_rng(seed, chunk);
    let mut buf = String::new();
    for doc_id in start..end {
        buf.clear();
        generator.write_doc(&mut rng, doc_id, &mut buf);
        f(doc_id, &buf);
    }
}

/// Visit every doc of the `flavor` corpus seeded with `seed`, in parallel
/// by scheduling chunk, without materializing the corpus. `f(doc_id,
/// text)` runs on rayon workers; docs within a chunk arrive in order,
/// chunks in any order. Produces exactly the bytes
/// [`super::MmapTextCorpus::generate`] writes for the same arguments.
pub fn for_each_generated_doc<F>(n_docs: usize, seed: u64, flavor: TextFlavor, f: F)
where
    F: Fn(usize, &str) + Sync,
{
    (0..generated_chunk_count(n_docs, flavor))
        .into_par_iter()
        .for_each(|c| {
            for_each_doc_in_chunk(n_docs, seed, flavor, c, |doc_id, text| f(doc_id, text))
        });
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        sync::Mutex,
    };

    use infino::test_helpers::default_tokenizer;

    use super::*;
    use crate::corpus::MmapTextCorpus;

    /// Docs sampled for the calibration bands. Large enough that the
    /// length quantiles and the rank-1 df settle to two digits; a few
    /// seconds of generation.
    const CALIBRATION_DOCS: usize = 20_000;
    const TEST_SEED: u64 = 1;

    struct Stats {
        lens: Vec<usize>,
        tokens: u64,
        repeats: u64,
        rank1_docs: usize,
        rank30_docs: usize,
        distinct_terms: usize,
        singleton_terms: usize,
    }

    fn realistic_stats(n_docs: usize) -> Stats {
        let tk = default_tokenizer();
        let per_doc: Mutex<Vec<(usize, Vec<String>)>> = Mutex::new(Vec::with_capacity(n_docs));
        for_each_generated_doc(n_docs, TEST_SEED, TextFlavor::Realistic, |doc_id, text| {
            let mut toks = Vec::new();
            tk.tokenize_each(text, &mut |t| toks.push(t.to_owned()));
            per_doc.lock().unwrap().push((doc_id, toks));
        });
        let per_doc = per_doc.into_inner().unwrap();
        let mut lens = Vec::with_capacity(n_docs);
        let mut tokens = 0u64;
        let mut repeats = 0u64;
        let mut rank1_docs = 0;
        let mut rank30_docs = 0;
        let mut df: HashMap<&str, u32> = HashMap::new();
        for (_, toks) in &per_doc {
            // The `doc…` token is not part of the body length.
            let body: Vec<&str> = toks.iter().skip(1).map(String::as_str).collect();
            lens.push(body.len());
            tokens += body.len() as u64;
            let distinct: HashSet<&str> = body.iter().copied().collect();
            repeats += (body.len() - distinct.len()) as u64;
            rank1_docs += usize::from(distinct.contains("term00001"));
            rank30_docs += usize::from(distinct.contains("term00030"));
            for t in distinct {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        lens.sort_unstable();
        let singleton_terms = df.values().filter(|&&c| c == 1).count();
        Stats {
            lens,
            tokens,
            repeats,
            rank1_docs,
            rank30_docs,
            distinct_terms: df.len(),
            singleton_terms,
        }
    }

    fn quantile(sorted: &[usize], p: f64) -> usize {
        sorted[((sorted.len() - 1) as f64 * p) as usize]
    }

    /// The realistic flavour stays inside the bands measured on English
    /// Wikipedia (300K-article sample). A generator change that moves a
    /// statistic out of its band changes what the quality bench measures
    /// and must be a deliberate recalibration.
    #[test]
    fn realistic_flavor_matches_calibration_bands() {
        let s = realistic_stats(CALIBRATION_DOCS);
        let n = CALIBRATION_DOCS as f64;
        let p50 = quantile(&s.lens, 0.5);
        let p90 = quantile(&s.lens, 0.9);
        let p99 = quantile(&s.lens, 0.99);
        let mean = s.tokens as f64 / n;
        let short = s.lens.iter().filter(|&&l| l < 16).count() as f64 / n;
        let repeat_frac = s.repeats as f64 / s.tokens as f64;
        let rank1_df = s.rank1_docs as f64 / n;
        let rank30_df = s.rank30_docs as f64 / n;
        let singleton_frac = s.singleton_terms as f64 / s.distinct_terms as f64;
        let report = format!(
            "p50={p50} p90={p90} p99={p99} mean={mean:.0} short={short:.3} repeat={repeat_frac:.3} \
             df1={rank1_df:.3} df30={rank30_df:.3} distinct={} singletons={singleton_frac:.3}",
            s.distinct_terms
        );
        // Real: 89 / 653 / 2636 / 268 / 0.156 / 0.56 / 0.84 / ~0.2-0.4 / 0.58.
        assert!((75..=105).contains(&p50), "{report}");
        assert!((520..=800).contains(&p90), "{report}");
        assert!((2200..=4200).contains(&p99), "{report}");
        assert!((240.0..=340.0).contains(&mean), "{report}");
        assert!((0.10..=0.17).contains(&short), "{report}");
        assert!((0.50..=0.62).contains(&repeat_frac), "{report}");
        assert!((0.78..=0.90).contains(&rank1_df), "{report}");
        assert!((0.15..=0.35).contains(&rank30_df), "{report}");
        assert!((0.45..=0.75).contains(&singleton_frac), "{report}");
    }

    /// Every surface variant folds back to a vocabulary term, a year, or
    /// the doc token under the production tokenizer — the analyzer is
    /// exercised without changing df/tf.
    #[test]
    fn realistic_surface_variants_tokenize_to_vocabulary() {
        let tk = default_tokenizer();
        let seen_variant = Mutex::new([false; 4]);
        for_each_generated_doc(512, TEST_SEED, TextFlavor::Realistic, |doc_id, text| {
            let mut flags = [false; 4];
            for raw in text.split(' ').skip(1) {
                flags[0] |= raw.starts_with('T');
                flags[1] |= raw.ends_with(',');
                flags[2] |= raw.ends_with('.');
                flags[3] |= raw.bytes().all(|b| b.is_ascii_digit());
            }
            let mut toks = Vec::new();
            tk.tokenize_each(text, &mut |t| toks.push(t.to_owned()));
            assert_eq!(toks[0], format!("doc{doc_id:07}"));
            for t in &toks[1..] {
                let is_term = t
                    .strip_prefix("term")
                    .is_some_and(|r| r.len() >= 5 && r.bytes().all(|b| b.is_ascii_digit()));
                let is_year = t.len() == 4 && t.bytes().all(|b| b.is_ascii_digit());
                assert!(is_term || is_year, "unexpected token {t:?} in doc {doc_id}");
            }
            let mut seen = seen_variant.lock().unwrap();
            for (s, f) in seen.iter_mut().zip(flags) {
                *s |= f;
            }
        });
        assert_eq!(
            seen_variant.into_inner().unwrap(),
            [true; 4],
            "capitalized / comma / period / year variants all occur"
        );
    }

    /// The streaming visitor and the mmap writer are the same corpus, for
    /// both flavours and across a chunk boundary.
    #[test]
    fn for_each_generated_doc_matches_mmap_corpus() {
        for flavor in [TextFlavor::Uniform, TextFlavor::Realistic] {
            let n_docs = TextDocGen::new(flavor).chunk_docs().min(4096) + 37;
            let mmap = MmapTextCorpus::generate_flavor(n_docs, TEST_SEED, flavor);
            assert_eq!(mmap.n_docs(), n_docs);
            let seen = Mutex::new(vec![false; n_docs]);
            for_each_generated_doc(n_docs, TEST_SEED, flavor, |doc_id, text| {
                assert_eq!(text, mmap.doc(doc_id), "{flavor:?} doc {doc_id}");
                seen.lock().unwrap()[doc_id] = true;
            });
            assert!(seen.into_inner().unwrap().iter().all(|&s| s));
        }
    }

    /// The realistic writer spans several reseeded chunks and stays
    /// byte-identical to the streaming visitor across every boundary.
    #[test]
    fn realistic_multi_chunk_writer_matches_stream() {
        let n_docs = REALISTIC_CHUNK_DOCS * 2 + 11;
        let mmap = MmapTextCorpus::generate_flavor(n_docs, TEST_SEED, TextFlavor::Realistic);
        assert_eq!(mmap.n_docs(), n_docs);
        for_each_generated_doc(n_docs, TEST_SEED, TextFlavor::Realistic, |doc_id, text| {
            assert_eq!(text, mmap.doc(doc_id), "doc {doc_id}");
        });
    }
}
