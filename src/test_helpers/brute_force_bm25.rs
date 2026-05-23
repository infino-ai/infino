//! Textbook BM25 reference implementation. Computes top-k by
//! scoring every doc directly from the BM25 formula; no inverted
//! index, no skip table, no WAND. Used as the correctness oracle
//! for the FTS pipeline's optimized BMW / BMM walks.
//!
//! Formula matches infino's production scoring (Lucene-form IDF +
//! standard BM25 tf factor):
//!
//! ```text
//!   idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
//!   tf_factor   = tf · (K1 + 1) / (tf + K1 · (1 - B + B · dl/avgdl))
//!   score(d, q) = Σ_{t ∈ q} idf(t) · tf_factor(tf(t, d), dl(d), avgdl)
//!
//!   K1 = 1.2,  B = 0.75            (Lucene defaults)
//! ```
//!
//! Tokenization runs through whatever `Tokenizer` the caller
//! supplies — pass [`crate::superfile::fts::tokenize::AsciiLowerTokenizer`]
//! to match the production pipeline.
//!
//! Result invariants match the optimized search path:
//! top-k by descending score, ties broken by ascending doc_id.

use std::collections::HashMap;

use crate::superfile::fts::tokenize::Tokenizer;

/// Lucene-default BM25 parameters. Match the constants used by
/// the production scoring path.
const K1: f32 = 1.2;
const B: f32 = 0.75;

/// Per-doc statistics derived from the corpus once and reused
/// across queries.
struct DocStats {
    /// Doc id, mirrored from the input.
    doc_id: u64,
    /// Token count (doc length).
    dl: u32,
    /// Term frequencies for this doc: term → count.
    tf: HashMap<String, u32>,
}

/// Pre-tokenized corpus + per-term df + corpus avgdl. Construct
/// once per fixture, query many times.
pub struct BruteForceBm25 {
    docs: Vec<DocStats>,
    /// Document-frequency: how many docs contain a given term at
    /// least once.
    df: HashMap<String, u32>,
    avgdl: f32,
    n: u32,
}

impl BruteForceBm25 {
    /// Tokenize the corpus once and capture per-doc + per-term
    /// statistics for later scoring. `tokenizer` MUST match what
    /// the production pipeline indexed the same corpus under —
    /// otherwise dl, df, and tf will diverge and recall comparisons
    /// will be meaningless.
    pub fn index(corpus: &[(u64, &str)], tokenizer: &dyn Tokenizer) -> Self {
        let mut docs: Vec<DocStats> = Vec::with_capacity(corpus.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total_tokens: u64 = 0;

        for (doc_id, text) in corpus {
            let mut tf: HashMap<String, u32> = HashMap::new();
            let mut dl: u32 = 0;
            tokenizer.tokenize_each(text, &mut |tok| {
                dl += 1;
                *tf.entry(tok.to_owned()).or_insert(0) += 1;
            });
            for term in tf.keys() {
                *df.entry(term.clone()).or_insert(0) += 1;
            }
            total_tokens += dl as u64;
            docs.push(DocStats {
                doc_id: *doc_id,
                dl,
                tf,
            });
        }

        let n = docs.len() as u32;
        let avgdl = if n == 0 {
            0.0
        } else {
            total_tokens as f32 / n as f32
        };

        Self { docs, df, avgdl, n }
    }

    /// Brute-force top-k for a multi-term OR-mode BM25 query.
    /// `query` is tokenized by the same tokenizer the index was
    /// built with (the caller passes both; we do not capture
    /// the tokenizer to keep the struct `Send + Sync` regardless
    /// of the tokenizer's bounds).
    ///
    /// Returns up to `k` `(doc_id, score)` pairs in descending
    /// score order; tie-breaks ascending by `doc_id`.
    pub fn top_k(&self, query: &str, k: usize, tokenizer: &dyn Tokenizer) -> Vec<(u64, f32)> {
        if k == 0 || self.n == 0 {
            return Vec::new();
        }
        let mut q_terms: Vec<String> = Vec::new();
        tokenizer.tokenize_each(query, &mut |tok| q_terms.push(tok.to_owned()));
        self.top_k_terms(&q_terms, k)
    }

    /// As [`Self::top_k`] but skips the query-side tokenization
    /// step — the caller has already tokenized.
    pub fn top_k_terms(&self, terms: &[String], k: usize) -> Vec<(u64, f32)> {
        if k == 0 || terms.is_empty() || self.n == 0 {
            return Vec::new();
        }

        let n = self.n as f32;
        let avgdl = self.avgdl;

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(self.docs.len());
        for doc in &self.docs {
            let mut score: f32 = 0.0;
            let dl = doc.dl as f32;
            let dl_norm = K1 * (1.0 - B + B * dl / avgdl.max(f32::MIN_POSITIVE));
            for term in terms {
                let Some(&tf) = doc.tf.get(term) else {
                    continue;
                };
                let df = *self.df.get(term).unwrap_or(&0) as f32;
                if df == 0.0 {
                    continue;
                }
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let tf_f = tf as f32;
                score += idf * tf_f * (K1 + 1.0) / (tf_f + dl_norm);
            }
            if score > 0.0 {
                scored.push((doc.doc_id, score));
            }
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    /// Same as [`Self::top_k_terms`] but with AND semantics: only docs
    /// that contain *every* query term contribute a score. Used by the
    /// AND-mode oracle tests against the reader's leapfrog
    /// intersection. Each term still contributes its own BM25 score to
    /// the per-doc sum; tie-break is ascending doc_id, identical to
    /// the OR helper.
    pub fn top_k_terms_and(&self, terms: &[String], k: usize) -> Vec<(u64, f32)> {
        if k == 0 || terms.is_empty() || self.n == 0 {
            return Vec::new();
        }

        let n = self.n as f32;
        let avgdl = self.avgdl;

        let mut scored: Vec<(u64, f32)> = Vec::with_capacity(self.docs.len());
        'docs: for doc in &self.docs {
            let dl = doc.dl as f32;
            let dl_norm = K1 * (1.0 - B + B * dl / avgdl.max(f32::MIN_POSITIVE));
            let mut score: f32 = 0.0;
            for term in terms {
                let Some(&tf) = doc.tf.get(term) else {
                    continue 'docs;
                };
                let df = *self.df.get(term).unwrap_or(&0) as f32;
                if df == 0.0 {
                    continue 'docs;
                }
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let tf_f = tf as f32;
                score += idf * tf_f * (K1 + 1.0) / (tf_f + dl_norm);
            }
            scored.push((doc.doc_id, score));
        }

        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        scored
    }

    /// Number of indexed docs.
    pub fn n_docs(&self) -> u32 {
        self.n
    }
}
