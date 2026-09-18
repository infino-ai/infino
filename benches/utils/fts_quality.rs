// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! FTS quality — BM25 top-k parity of the search path against a textbook
//! oracle, at bench scale, on both tiers: the supertable ([`run`]) and a
//! single superfile ([`run_superfile`]).
//!
//! The speed cells time `bm25_search` and discard what it returns. This
//! module grades what it returns: for every shape in [`QUALITY_BATTERY`]
//! and every `k` in [`QUALITY_KS`], the engine's top-k is compared with a
//! reference computed directly from the BM25 formula over the whole
//! corpus. The vector cells have had this (recall against brute-force
//! truth) from the start; this is the FTS equivalent.
//!
//! ## Two references, one corpus pass
//!
//! The engine deliberately differs from textbook BM25 in two places, and a
//! quality number has to separate those design costs from kernel bugs:
//!
//! * **Length quantization.** A document's length is stored in one byte
//!   (`stored_len`): exact below 16 tokens, truncated downward by up to
//!   one bucket above. On a corpus with realistic length variation this
//!   reorders near-ties.
//! * **Statistics scope.** Under [`Bm25Stats::PerSuperfile`] each
//!   superfile scores with its own document count and term frequencies.
//!   Under [`Bm25Stats::Global`] idf is table-wide, and each superfile
//!   normalizes lengths with the average it *declares*: the table-wide
//!   average as of its commit (every superfile already committed plus its
//!   own documents), rounded to the fixed point the file stores. A file
//!   from the first commit therefore scores at its own average; a file
//!   from the last at nearly the corpus-wide one.
//!
//! So the oracle scores every matching document twice from the same
//! statistics: **T**, textbook BM25 with the exact length and the
//! corpus-wide average, and **Q**, the BM25 the engine is specified to
//! compute, with the stored length and the declared average of the
//! superfile that holds the document. The table then reports, per query
//! and `k`:
//!
//! | column | definition |
//! |---|---|
//! | recall vs BM25 | engine top-k under `Global` graded against T — the user-facing quality, including the quantization and average-length costs |
//! | recall (per-superfile stats) | the same under `PerSuperfile` — the opt-in mode's sharded-idf drift |
//! | recall vs engine BM25 | engine top-k under `Global` graded against Q — must be 1.0; a drop is a kernel or pruning bug. **Gated.** |
//! | max score Δ | largest relative gap between an engine score and Q for the same document (`Global`). **Gated** at f32 noise. |
//!
//! Q is exact, so its gates are tight. Making it exact needs the row-to-
//! superfile mapping and each superfile's declared average: [`Layout`]
//! recovers both from the table itself and verifies the mapping against
//! the stored document lengths (§ [`Layout::recover`]).
//!
//! Recall is **tie-aware**: a returned document counts as a hit when its
//! oracle score is at least the oracle's k-th score (within a rounding
//! tolerance), so the engine picking a different member of a tied group at
//! the boundary is not penalized. On a bursty corpus single-term queries
//! tie heavily at the boundary; doc-id set overlap would report noise.
//!
//! ## Why a streaming oracle
//!
//! The test-suite reference (`BruteForceBm25`) keeps every token of every
//! document and is unusable at 10M docs. This oracle re-derives the corpus
//! from its seed one scheduling chunk at a time and keeps only what the
//! battery needs: each document's length and its term frequencies for the
//! few dozen terms (and phrases) the battery mentions — a sparse table of
//! a few tens of millions of entries at 10M docs. Both sides tokenize and
//! parse with the table's own tokenizer, so tokenization and clause
//! semantics cannot diverge by construction.

use std::{borrow::Cow, cmp::Ordering, collections::HashMap, sync::Arc, time::Instant};

use arrow_array::{Array, Float32Array, LargeStringArray, RecordBatch, StringArray};
use infino::{
    Bm25SearchOptions,
    superfile::{
        SuperfileReader,
        format::fts::DOC_LENGTH_STORED_MAX,
        fts::{
            bm25::{stored_avgdl, stored_len},
            reader::{Bm25Stats, BoolMode, ColumnLengthStats, FtsReader},
            tokenize::Tokenizer,
        },
    },
    supertable::{SupertableReader, manifest::SuperfileEntry},
    test_helpers::default_tokenizer,
};
use rayon::prelude::*;

use crate::{
    corpus::{self, TextFlavor, block_on_inmem, for_each_doc_in_chunk, generated_chunk_count},
    markdown::fmt_count,
    report::{Better, Block, Cell, Report, Section, context, metric, text},
    tiers,
};

/// The `k` values graded. `10` is the speed cell's top-k, `1000` its
/// large-k gate; `1000` is also where sharded-idf drift and ties show.
pub const QUALITY_KS: &[usize] = &[10, 100, 1000];

/// Standard BM25 parameters — the same constants the engine scores with.
/// Restated here so the oracle is the formula by construction, sharing no
/// code with the kernels it grades.
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Relative tolerance when comparing a document's oracle score against
/// the k-th oracle score for the tie test. Covers f32 accumulation order
/// on the engine side and f64→f32 rounding; well below any real score
/// gap.
const TIE_TOLERANCE: f64 = 1e-4;

/// Gate: every query × k must reach this recall against the engine-model
/// reference (Q) under `Global` statistics. Q is exact, so every expected
/// slot must be filled; a miss is a document the formula the kernels
/// implement would not have returned.
const MIN_ENGINE_MODEL_RECALL: f64 = 1.0;

/// Gate: the largest relative gap between an engine score and Q over the
/// returned documents, under `Global`. Q reproduces every input the engine
/// scores with (table-wide idf, the stored length, the declared average),
/// so what remains is f32 arithmetic: idf and the length normalizer held
/// as `f32`, the per-term sum accumulated in `f32`. Those are parts per
/// million; the ceiling leaves three orders of magnitude of headroom and
/// still fails on any mismatched input.
const MAX_SCORE_DELTA: f64 = 1e-3;

/// Relative tolerance when checking a superfile's declared average against
/// the value the commit rule predicts. Both sides round to the same fixed
/// point from the same totals; this only absorbs the `f64` → `f32` step.
const DECLARED_AVGDL_TOLERANCE: f64 = 1e-6;

/// Guard against a degenerate oracle score in a relative-delta denominator.
const MIN_SCORE_FOR_RELATIVE_DELTA: f64 = 1e-9;

/// One graded query shape. `query` is the literal string handed to both
/// the engine and the oracle's parser; `mode` is the default operator.
#[derive(Clone, Copy, Debug)]
pub struct QualityQuery {
    pub name: &'static str,
    pub query: &'static str,
    pub mode: BoolMode,
}

/// Query shapes chosen by document-frequency tier rather than by fixed
/// rank so they mean the same thing on every generated corpus. Terms are
/// Zipf ranks: rank 1 is the stopword band (~84% of docs on the realistic
/// corpus, every doc on the uniform one), rank 30 is common (~20%), rank
/// 2000 is mid (~0.6%), rank 200 000 is rare (~1e-4; absent from the
/// uniform corpus's 10K vocabulary, where both sides return nothing), and
/// `doc0000001` is a singleton on every generated corpus.
pub const QUALITY_BATTERY: &[QualityQuery] = &[
    QualityQuery {
        name: "stopword",
        query: "term00001",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "common",
        query: "term00030",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "mid",
        query: "term02000",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "rare",
        query: "term200000",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "singleton",
        query: "doc0000001",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "stopword_common_or",
        query: "term00001 term00030",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "stopword_rare_or",
        query: "term00001 term200000",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "three_stopword_or",
        query: "term00001 term00002 term00003",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "three_mid_or",
        query: "term02000 term02001 term02002",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "ten_common_or",
        query: "term00030 term00031 term00032 term00033 term00034 term00035 term00036 term00037 \
                term00038 term00039",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "stopword_rare_and",
        query: "term00001 term200000",
        mode: BoolMode::And,
    },
    QualityQuery {
        name: "common_mid_and",
        query: "term00030 term02000",
        mode: BoolMode::And,
    },
    QualityQuery {
        name: "three_mid_and",
        query: "term02000 term02001 term02002",
        mode: BoolMode::And,
    },
    QualityQuery {
        name: "must_rare_should_common",
        query: "+term200000 term00030",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "must_two_should_two",
        query: "+term00030 +term00031 term00001 term02000",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "common_not_stopword",
        query: "term00030 -term00001",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "phrase_stopwords",
        query: "\"term00001 term00002\"",
        mode: BoolMode::Or,
    },
    QualityQuery {
        name: "phrase_common_mid",
        query: "\"term00030 term02000\"",
        mode: BoolMode::Or,
    },
];

/// Atom index: `< n_terms` is a term, `>= n_terms` a phrase (`atom -
/// n_terms` indexes `phrases`).
type Atom = u16;

/// One nonzero `(document, atom, tf)` cell of the sparse statistics table.
#[derive(Clone, Copy)]
struct Entry {
    row: u32,
    atom: Atom,
    tf: u32,
}

/// A query's clauses resolved to atom indices.
struct ResolvedQuery {
    musts: Vec<Atom>,
    shoulds: Vec<Atom>,
    negatives: Vec<Atom>,
}

/// One superfile's slice of the corpus and the average length it scores at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuperfileSpan {
    /// First corpus row this superfile holds; its rows are contiguous.
    pub first_row: u32,
    pub n_docs: u32,
    /// The average document length the superfile declares, which is what
    /// its norm table decodes with.
    pub avgdl: f64,
}

/// The row-to-superfile mapping of the graded table, with each
/// superfile's declared average length — the one input to Q that varies
/// by superfile.
#[derive(Clone, Debug)]
pub struct Layout {
    /// Ascending by `first_row`, covering `0..n_docs` without gaps.
    spans: Vec<SuperfileSpan>,
    /// How many commits the superfiles fell into (see [`Self::recover`]).
    n_commits: usize,
}

impl Layout {
    /// One superfile holding every row, scoring at `avgdl`: the superfile
    /// tier, and the oracle's default before a table layout is recovered.
    pub fn single(n_docs: usize, avgdl: f64) -> Self {
        Self {
            spans: vec![SuperfileSpan {
                first_row: 0,
                n_docs: u32::try_from(n_docs).expect("row count fits u32"),
                avgdl,
            }],
            n_commits: 1,
        }
    }

    /// Recover the layout of the table behind `reader` for FTS column
    /// `column`, given the oracle's exact per-row token counts `dl`.
    ///
    /// Rows reach a superfile in append order: the writer splits a
    /// commit's buffered rows into contiguous, row-ordered shards, and the
    /// id column is assigned monotonically at append, so ordering the
    /// manifest by `id_min` orders superfiles by first row and each holds
    /// its `n_docs` rows contiguously. That is an inference about the
    /// writer, so it is **verified**, not assumed: every superfile's
    /// stored document lengths are read back and must equal the oracle's
    /// lengths for the rows assigned to it, element for element (both
    /// saturated at the format's stored maximum). A mismatch is a hard
    /// error naming the superfile and row.
    ///
    /// Each superfile's declared average is read from its reader. It is
    /// then checked against the commit rule the writer implements — the
    /// table-wide totals of every superfile committed before it, plus its
    /// own, rounded to the stored fixed point. Superfiles built by one
    /// commit share the same "committed before" totals, and the manifest
    /// does not record commit boundaries, so they are derived: a superfile
    /// whose declared average matches the rule with every earlier
    /// superfile counted as committed starts a new commit; one that
    /// matches only with the current commit's earlier files excluded
    /// continues it. Neither matching is an error.
    pub fn recover(reader: &SupertableReader, column: &str, dl: &[u32]) -> Self {
        let manifest = reader.manifest();
        let mut entries: Vec<Arc<SuperfileEntry>> = manifest.get_all_superfiles().to_vec();
        if entries.is_empty() {
            for part_entry in manifest.get_all_list_entries() {
                let part = tiers::block_on(manifest.get_part_by_id(part_entry.part_id))
                    .expect("load manifest part for the quality layout");
                entries.extend(part.superfiles.iter().cloned());
            }
        }
        entries.sort_by_key(|e| e.id_min);
        let n_rows: u64 = entries.iter().map(|e| e.n_docs).sum();
        assert_eq!(
            n_rows,
            dl.len() as u64,
            "manifest row count differs from the oracle's corpus"
        );

        let mut spans = Vec::with_capacity(entries.len());
        let mut committed = ColumnLengthStats::default();
        let mut commit = ColumnLengthStats::default();
        let mut n_commits = 0usize;
        let mut first_row = 0usize;
        for (i, entry) in entries.iter().enumerate() {
            let n_docs = usize::try_from(entry.n_docs).expect("superfile row count fits usize");

            let superfile = reader.open_superfile(entry).unwrap_or_else(|e| {
                panic!(
                    "open superfile {i} ({:?}) for the quality layout: {e}",
                    entry.uri
                )
            });
            let fts = superfile.fts().unwrap_or_else(|| {
                panic!("superfile {i} ({:?}) has no full-text index", entry.uri)
            });
            let stored = fts
                .column_doc_lengths(column)
                .unwrap_or_else(|e| panic!("superfile {i}: read doc lengths of {column:?}: {e}"));
            assert_eq!(
                stored.len(),
                n_docs,
                "superfile {i}: doc-length array holds {} rows, manifest says {n_docs}",
                stored.len()
            );
            let label = format!("superfile {i} ({:?})", entry.uri);
            verify_doc_lengths(&stored, dl, first_row, &label);
            let declared = declared_avgdl(fts, column);

            // The commit rule, under both hypotheses about where this
            // superfile's commit begins.
            let own = ColumnLengthStats::from_lengths(stored.iter().copied());
            let mut all_earlier = committed;
            all_earlier.merge_with(&commit);
            let as_new_commit = declared_under(&all_earlier, &own);
            let as_same_commit = declared_under(&committed, &own);
            if close_enough(declared, as_new_commit) {
                committed = all_earlier;
                commit = own;
                n_commits += 1;
            } else if close_enough(declared, as_same_commit) {
                commit.merge_with(&own);
            } else {
                panic!(
                    "superfile {i} ({:?}) declares avgdl {declared} but the commit rule predicts \
                     {as_new_commit} (first file of a commit) or {as_same_commit} (later file of \
                     the current commit)",
                    entry.uri
                );
            }

            spans.push(SuperfileSpan {
                first_row: u32::try_from(first_row).expect("row fits u32"),
                n_docs: u32::try_from(n_docs).expect("row count fits u32"),
                avgdl: declared,
            });
            first_row += n_docs;
        }
        Self { spans, n_commits }
    }

    pub fn n_superfiles(&self) -> usize {
        self.spans.len()
    }

    pub fn n_commits(&self) -> usize {
        self.n_commits
    }

    /// The declared average of the superfile holding `row`.
    fn avgdl_of(&self, row: u32) -> f64 {
        let i = self.spans.partition_point(|s| s.first_row <= row);
        debug_assert!(i > 0, "row {row} precedes the first span");
        self.spans[i - 1].avgdl
    }
}

/// The average a superfile declares when `own` joins a table whose
/// committed totals are `committed`: the merged average, rounded as the
/// writer stores it.
fn declared_under(committed: &ColumnLengthStats, own: &ColumnLengthStats) -> f64 {
    let mut table = *committed;
    table.merge_with(own);
    f64::from(stored_avgdl(table.avgdl()))
}

fn close_enough(declared: f64, predicted: f64) -> bool {
    (declared - predicted).abs()
        <= DECLARED_AVGDL_TOLERANCE * predicted.abs().max(f64::MIN_POSITIVE)
}

/// Check that `stored`, one file's document-length array, equals the
/// oracle's lengths for the rows starting at `first_row`, element for
/// element (the oracle's saturated at the format's stored maximum, as
/// the build saturates). `label` names the file in the failure.
fn verify_doc_lengths(stored: &[u32], dl: &[u32], first_row: usize, label: &str) {
    let end = first_row + stored.len();
    assert!(
        end <= dl.len(),
        "{label}: holds {} rows from row {first_row}, past the corpus ({} rows)",
        stored.len(),
        dl.len()
    );
    let expected = &dl[first_row..end];
    if let Some(bad) =
        (0..stored.len()).find(|&j| stored[j] != expected[j].min(DOC_LENGTH_STORED_MAX))
    {
        panic!(
            "{label} does not hold rows {first_row}..{end}: local doc {bad} has stored length {} \
             but corpus row {} has {}. The row-to-file mapping the oracle infers is wrong for \
             this table.",
            stored[bad],
            first_row + bad,
            expected[bad]
        );
    }
}

/// The average document length `column` is scored at in one file: what
/// its norm table decodes with.
fn declared_avgdl(fts: &FtsReader, column: &str) -> f64 {
    f64::from(
        fts.fts_columns_config()
            .find(|c| c.name == column)
            .unwrap_or_else(|| panic!("column {column:?} has no full-text index in this file"))
            .avgdl(),
    )
}

/// Textbook BM25 statistics over the whole corpus, restricted to the
/// battery's atoms.
pub struct Oracle {
    n_docs: usize,
    n_terms: usize,
    /// Term → atom, the battery's vocabulary.
    term_index: HashMap<String, Atom>,
    /// Phrase atoms as sequences of term atoms.
    phrases: Vec<Vec<Atom>>,
    /// Token count per document (index = corpus row).
    dl: Vec<u32>,
    /// Nonzero tf cells, sorted by `(row, atom)`.
    entries: Vec<Entry>,
    /// `entries[row_start[r]..row_start[r + 1]]` are row `r`'s cells.
    row_start: Vec<u32>,
    /// Documents containing each atom at least once.
    df: Vec<u32>,
    /// Corpus-wide average length: T normalizes with it.
    avgdl: f64,
    /// Where each row lives and the average its superfile scores at: Q
    /// normalizes with it. One span at the corpus average until
    /// [`Self::set_layout`].
    layout: Layout,
}

/// One document's scores under both references.
#[derive(Clone, Copy)]
struct Scored {
    row: u32,
    textbook: f64,
    engine_model: f64,
}

/// A query's oracle result: the match count and, per graded `k`, the
/// k-th score under each reference (the threshold a returned document
/// must reach to count as a hit).
struct OracleTopK {
    n_matches: usize,
    /// Indexed like [`QUALITY_KS`]; `None` when fewer than `k` documents match
    /// (then every match is expected and the threshold is the lowest match).
    textbook_kth: Vec<f64>,
    engine_model_kth: Vec<f64>,
}

/// Per-chunk accumulation of the streaming pass.
struct ChunkStats {
    start_row: u32,
    dl: Vec<u32>,
    entries: Vec<Entry>,
}

impl Oracle {
    /// Stream the `flavor` corpus (`n_docs` docs, `seed`) once and collect
    /// the statistics `battery` needs, tokenizing and parsing with
    /// `tokenizer` — the table's own, so both sides see identical tokens
    /// and clauses.
    pub fn build(
        n_docs: usize,
        seed: u64,
        flavor: TextFlavor,
        tokenizer: &dyn Tokenizer,
        battery: &[QualityQuery],
    ) -> Self {
        // Atom vocabulary: every term any clause or phrase mentions, then
        // every distinct phrase.
        let mut term_index: HashMap<String, Atom> = HashMap::new();
        let mut phrases: Vec<Vec<Atom>> = Vec::new();
        for q in battery {
            let clauses = tokenizer.parse(q.query).into_clauses(q.mode);
            for term in clauses
                .musts
                .iter()
                .chain(&clauses.shoulds)
                .chain(&clauses.negatives)
            {
                intern_term(&mut term_index, term);
            }
            for phrase in clauses
                .must_phrases
                .iter()
                .chain(&clauses.should_phrases)
                .chain(&clauses.negative_phrases)
            {
                let atoms: Vec<Atom> = phrase
                    .iter()
                    .map(|t| intern_term(&mut term_index, t))
                    .collect();
                if !phrases.contains(&atoms) {
                    phrases.push(atoms);
                }
            }
        }
        let n_terms = term_index.len();
        let n_atoms = n_terms + phrases.len();

        let n_chunks = generated_chunk_count(n_docs, flavor);
        let mut chunks: Vec<ChunkStats> = (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let mut stats = ChunkStats {
                    start_row: u32::MAX,
                    dl: Vec::new(),
                    entries: Vec::new(),
                };
                let mut positions: Vec<(u32, Atom)> = Vec::new();
                let mut tf = vec![0u32; n_atoms];
                for_each_doc_in_chunk(n_docs, seed, flavor, c, |doc_id, text| {
                    let row = u32::try_from(doc_id).expect("row fits u32");
                    if stats.start_row == u32::MAX {
                        stats.start_row = row;
                    }
                    positions.clear();
                    let mut dl = 0u32;
                    tokenizer.tokenize_each(text, &mut |tok| {
                        if let Some(&a) = term_index.get(tok) {
                            positions.push((dl, a));
                        }
                        dl += 1;
                    });
                    stats.dl.push(dl);
                    for &(_, a) in &positions {
                        tf[a as usize] += 1;
                    }
                    for (p, phrase) in phrases.iter().enumerate() {
                        tf[n_terms + p] = phrase_tf(&positions, phrase);
                    }
                    for (a, count) in tf.iter_mut().enumerate() {
                        if *count > 0 {
                            stats.entries.push(Entry {
                                row,
                                atom: a as Atom,
                                tf: *count,
                            });
                            *count = 0;
                        }
                    }
                });
                stats
            })
            .collect();
        chunks.sort_by_key(|c| c.start_row);

        let mut dl = Vec::with_capacity(n_docs);
        let mut entries = Vec::with_capacity(chunks.iter().map(|c| c.entries.len()).sum());
        for chunk in chunks {
            dl.extend(chunk.dl);
            entries.extend(chunk.entries);
        }
        assert_eq!(dl.len(), n_docs, "oracle streamed every document");
        let mut row_start = vec![0u32; n_docs + 1];
        let mut df = vec![0u32; n_atoms];
        for (i, e) in entries.iter().enumerate() {
            row_start[e.row as usize + 1] = u32::try_from(i + 1).expect("entry count fits u32");
            df[e.atom as usize] += 1;
        }
        // Rows without cells inherit the previous boundary so each row's
        // range is well-formed (and empty).
        for r in 1..=n_docs {
            if row_start[r] < row_start[r - 1] {
                row_start[r] = row_start[r - 1];
            }
        }
        let total_tokens: u64 = dl.iter().map(|&l| u64::from(l)).sum();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            total_tokens as f64 / n_docs as f64
        };
        Self {
            n_docs,
            n_terms,
            term_index,
            phrases,
            dl,
            entries,
            row_start,
            df,
            avgdl,
            layout: Layout::single(n_docs, avgdl),
        }
    }

    pub fn n_docs(&self) -> usize {
        self.n_docs
    }

    /// The exact token count of every row, in row order.
    pub fn doc_lengths(&self) -> &[u32] {
        &self.dl
    }

    /// Install the graded table's layout, which decides the average
    /// length Q normalizes each row with.
    pub fn set_layout(&mut self, layout: Layout) {
        self.layout = layout;
    }

    /// `ln(1 + (N - df + 0.5) / (df + 0.5))`; a phrase's idf is the sum of
    /// its members', as in the engine.
    fn idf(&self, atom: Atom) -> f64 {
        let a = atom as usize;
        if a >= self.n_terms {
            return self.phrases[a - self.n_terms]
                .iter()
                .map(|&m| self.idf(m))
                .sum();
        }
        let n = self.n_docs as f64;
        let df = f64::from(self.df[a]);
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    fn tf_factor(tf: u32, dl: u32, avgdl: f64) -> f64 {
        let tf = f64::from(tf);
        let norm = 1.0 - B + B * f64::from(dl) / avgdl.max(f64::MIN_POSITIVE);
        tf / (tf + K1 * norm)
    }

    /// A battery query's clauses as atom indices. Every term and phrase was
    /// interned at build time from the same parser, so a miss is a
    /// programming error, not a data case.
    fn resolve(&self, tokenizer: &dyn Tokenizer, q: &QualityQuery) -> ResolvedQuery {
        let clauses = tokenizer.parse(q.query).into_clauses(q.mode);
        let term_atom = |t: &Cow<'_, str>| -> Atom {
            *self
                .term_index
                .get(t.as_ref())
                .unwrap_or_else(|| panic!("battery term {t:?} missing from the oracle vocabulary"))
        };
        let phrase_atom = |p: &[Cow<'_, str>]| -> Atom {
            let atoms: Vec<Atom> = p.iter().map(term_atom).collect();
            let idx = self
                .phrases
                .iter()
                .position(|ph| *ph == atoms)
                .expect("battery phrase missing from the oracle vocabulary");
            Atom::try_from(self.n_terms + idx).expect("atom index")
        };
        let mut musts: Vec<Atom> = clauses.musts.iter().map(term_atom).collect();
        musts.extend(clauses.must_phrases.iter().map(|p| phrase_atom(p)));
        let mut shoulds: Vec<Atom> = clauses.shoulds.iter().map(term_atom).collect();
        shoulds.extend(clauses.should_phrases.iter().map(|p| phrase_atom(p)));
        let mut negatives: Vec<Atom> = clauses.negatives.iter().map(term_atom).collect();
        negatives.extend(clauses.negative_phrases.iter().map(|p| phrase_atom(p)));
        ResolvedQuery {
            musts,
            shoulds,
            negatives,
        }
    }

    /// Score one row's cells under both references, or `None` when the row
    /// does not match the query. T normalizes with the corpus average and
    /// the exact length; Q with the declared average of the row's
    /// superfile and the length as the file stores it.
    fn score_cells(&self, cells: &[Entry], row: u32, q: &ResolvedQuery) -> Option<(f64, f64)> {
        let tf_of = |atom: Atom| -> Option<u32> {
            cells
                .binary_search_by_key(&atom, |e| e.atom)
                .ok()
                .map(|i| cells[i].tf)
        };
        if q.negatives.iter().any(|&a| tf_of(a).is_some()) {
            return None;
        }
        let dl = self.dl[row as usize];
        let dl_q = stored_len(dl.min(DOC_LENGTH_STORED_MAX));
        let avgdl_q = self.layout.avgdl_of(row);
        let mut textbook = 0.0f64;
        let mut engine_model = 0.0f64;
        let mut add = |atom: Atom, tf: u32| {
            let idf = self.idf(atom);
            textbook += idf * Self::tf_factor(tf, dl, self.avgdl);
            engine_model += idf * Self::tf_factor(tf, dl_q, avgdl_q);
        };
        for &a in &q.musts {
            add(a, tf_of(a)?);
        }
        let mut matched_should = false;
        for &a in &q.shoulds {
            if let Some(tf) = tf_of(a) {
                matched_should = true;
                add(a, tf);
            }
        }
        if q.musts.is_empty() && !matched_should {
            return None;
        }
        Some((textbook, engine_model))
    }

    /// Both scores for `row`, or `None` if it does not match.
    fn score_row(&self, row: u32, q: &ResolvedQuery) -> Option<(f64, f64)> {
        let r = row as usize;
        if r >= self.n_docs {
            return None;
        }
        let cells = &self.entries[self.row_start[r] as usize..self.row_start[r + 1] as usize];
        self.score_cells(cells, row, q)
    }

    /// Every matching document, scored under both references.
    fn matches(&self, q: &ResolvedQuery) -> Vec<Scored> {
        // Walk cells grouped by row; rows without cells cannot match (a
        // match needs at least one positive atom present).
        (0..self.n_docs)
            .into_par_iter()
            .filter_map(|r| {
                let (lo, hi) = (self.row_start[r] as usize, self.row_start[r + 1] as usize);
                if lo == hi {
                    return None;
                }
                let row = r as u32;
                let (textbook, engine_model) = self.score_cells(&self.entries[lo..hi], row, q)?;
                Some(Scored {
                    row,
                    textbook,
                    engine_model,
                })
            })
            .collect()
    }

    fn top_k(&self, q: &ResolvedQuery) -> OracleTopK {
        let mut scored = self.matches(q);
        let n_matches = scored.len();
        let kth = |scored: &mut Vec<Scored>, key: fn(&Scored) -> f64, k: usize| -> f64 {
            if scored.is_empty() {
                return 0.0;
            }
            let k = k.min(scored.len());
            // Descending by score, ascending by row — the engine's own tie
            // order, though the tie test makes the order immaterial.
            scored.select_nth_unstable_by(k - 1, |a, b| {
                key(b)
                    .partial_cmp(&key(a))
                    .unwrap_or(Ordering::Equal)
                    .then(a.row.cmp(&b.row))
            });
            key(&scored[k - 1])
        };
        let textbook_kth = QUALITY_KS
            .iter()
            .map(|&k| kth(&mut scored, |s| s.textbook, k))
            .collect();
        let engine_model_kth = QUALITY_KS
            .iter()
            .map(|&k| kth(&mut scored, |s| s.engine_model, k))
            .collect();
        OracleTopK {
            n_matches,
            textbook_kth,
            engine_model_kth,
        }
    }
}

/// Intern `term` into the atom vocabulary, returning its atom.
fn intern_term(index: &mut HashMap<String, Atom>, term: &str) -> Atom {
    if let Some(&a) = index.get(term) {
        return a;
    }
    let a = Atom::try_from(index.len()).expect("battery vocabulary fits an atom index");
    index.insert(term.to_owned(), a);
    a
}

/// Occurrence-start count of `phrase` in a document's recorded
/// `(position, atom)` list (ascending positions). Every phrase member is
/// in the vocabulary, so an adjacent member must have been recorded at
/// `position + 1`; overlapping occurrences count separately, as in the
/// engine's phrase walk.
fn phrase_tf(positions: &[(u32, Atom)], phrase: &[Atom]) -> u32 {
    if phrase.is_empty() || positions.len() < phrase.len() {
        return 0;
    }
    let mut count = 0u32;
    'starts: for i in 0..=(positions.len() - phrase.len()) {
        let (start_pos, first) = positions[i];
        if first != phrase[0] {
            continue;
        }
        for (offset, &member) in phrase.iter().enumerate().skip(1) {
            let (pos, atom) = positions[i + offset];
            if atom != member || pos != start_pos + offset as u32 {
                continue 'starts;
            }
        }
        count += 1;
    }
    count
}

/// One engine hit mapped back to a corpus row.
struct EngineHit {
    row: u32,
    score: f64,
}

/// How a tier answers one battery query: its hits at `k`, scored under
/// `stats`. The supertable tier maps each hit to its row through the doc
/// token of the returned text; on the superfile tier the local doc id is
/// the row and the statistics scope is moot.
type HitSource<'a> = dyn Fn(&QualityQuery, usize, Bm25Stats) -> Vec<EngineHit> + 'a;

/// Run `query` through the supertable's public search path and map each
/// hit to its corpus row via the per-doc unique `doc{id:07}` token the
/// generated corpora plant as the first token — cheaper than a 10M-row
/// `_id` scan and independent of `_id` assignment order.
fn supertable_hits(
    reader: &SupertableReader,
    column: &str,
    q: &QualityQuery,
    k: usize,
    stats: Bm25Stats,
) -> Vec<EngineHit> {
    let batches = reader
        .bm25_search(
            column,
            q.query,
            k,
            Bm25SearchOptions::new().with_mode(q.mode).with_stats(stats),
            Some(&[column, "score"]),
        )
        .expect("quality bm25_search");
    let mut hits = Vec::with_capacity(k);
    for batch in &batches {
        let text_idx = batch
            .schema()
            .index_of(column)
            .expect("text column projected");
        let score_idx = batch
            .schema()
            .index_of("score")
            .expect("score column projected");
        let scores = batch
            .column(score_idx)
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("score is Float32");
        for i in 0..batch.num_rows() {
            let title = row_text(batch, text_idx, i);
            let row = title
                .split(' ')
                .next()
                .and_then(|t| t.strip_prefix("doc"))
                .and_then(|d| d.parse::<u32>().ok())
                .unwrap_or_else(|| panic!("hit text does not start with a doc token: {title:?}"));
            hits.push(EngineHit {
                row,
                score: f64::from(scores.value(i)),
            });
        }
    }
    hits
}

fn row_text(batch: &RecordBatch, idx: usize, i: usize) -> &str {
    let col = batch.column(idx);
    if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
        return a.value(i);
    }
    if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        return a.value(i);
    }
    panic!("text column is neither LargeUtf8 nor Utf8");
}

/// Run `query` through one superfile's string search path. The fixture
/// is built in corpus order, so the local doc id is the corpus row.
fn superfile_hits(
    reader: &SuperfileReader,
    column: &str,
    q: &QualityQuery,
    k: usize,
) -> Vec<EngineHit> {
    block_on_inmem(reader.bm25_hits_async(column, q.query, k, q.mode))
        .expect("quality bm25_hits_async")
        .into_iter()
        .map(|(doc, score)| EngineHit {
            row: doc,
            score: f64::from(score),
        })
        .collect()
}

/// Tie-aware recall: the fraction of the `expected` slots filled by a
/// returned document whose oracle score reaches the k-th oracle score,
/// less a relative `tolerance` for rounding. A query with no matches
/// scores 1.0 only if the engine also returned nothing.
fn tie_aware_recall(
    hits: &[EngineHit],
    expected: usize,
    kth: f64,
    tolerance: f64,
    score_of: impl Fn(u32) -> Option<f64>,
) -> f64 {
    if expected == 0 {
        return if hits.is_empty() { 1.0 } else { 0.0 };
    }
    let ok = hits
        .iter()
        .filter(|h| score_of(h.row).is_some_and(|s| s >= kth * (1.0 - tolerance)))
        .count();
    ok.min(expected) as f64 / expected as f64
}

/// One graded `(query, k)` cell of the report.
pub struct GradedRow {
    pub name: &'static str,
    pub k: usize,
    pub n_matches: usize,
    pub recall_textbook: f64,
    /// `None` on the superfile tier, where statistics have one scope.
    pub recall_per_superfile_stats: Option<f64>,
    pub recall_engine_model: f64,
    pub max_delta: f64,
}

impl GradedRow {
    /// Whether this cell passes both gates.
    pub fn passes(&self) -> bool {
        self.recall_engine_model >= MIN_ENGINE_MODEL_RECALL && self.max_delta <= MAX_SCORE_DELTA
    }

    fn failure(&self) -> String {
        format!(
            "{} k={}: recall vs engine BM25 {:.4} (floor {MIN_ENGINE_MODEL_RECALL}), max score Δ {:.4}% (ceiling {:.4}%)",
            self.name,
            self.k,
            self.recall_engine_model,
            self.max_delta * 100.0,
            MAX_SCORE_DELTA * 100.0
        )
    }
}

/// Grade every `battery` shape at every `k` against `oracle`, whose
/// layout must already be the graded table's, taking hits from `hits`.
/// `per_superfile_stats` adds the opt-in statistics scope's column.
fn grade_with(
    oracle: &Oracle,
    tokenizer: &dyn Tokenizer,
    battery: &[QualityQuery],
    hits: &HitSource<'_>,
    per_superfile_stats: bool,
) -> Vec<GradedRow> {
    let mut rows: Vec<GradedRow> = Vec::new();
    for q in battery {
        let rq = oracle.resolve(tokenizer, q);
        let top = oracle.top_k(&rq);
        for (ki, &k) in QUALITY_KS.iter().enumerate() {
            let expected = k.min(top.n_matches);
            let global = hits(q, k, Bm25Stats::Global);
            let textbook_of = |row| oracle.score_row(row, &rq).map(|(t, _)| t);
            let engine_model_of = |row| oracle.score_row(row, &rq).map(|(_, e)| e);
            let recall_textbook = tie_aware_recall(
                &global,
                expected,
                top.textbook_kth[ki],
                TIE_TOLERANCE,
                textbook_of,
            );
            let recall_per_superfile_stats = per_superfile_stats.then(|| {
                let per_superfile = hits(q, k, Bm25Stats::PerSuperfile);
                tie_aware_recall(
                    &per_superfile,
                    expected,
                    top.textbook_kth[ki],
                    TIE_TOLERANCE,
                    textbook_of,
                )
            });
            let recall_engine_model = tie_aware_recall(
                &global,
                expected,
                top.engine_model_kth[ki],
                TIE_TOLERANCE,
                engine_model_of,
            );
            let max_delta = global
                .iter()
                .map(|h| match engine_model_of(h.row) {
                    Some(e) => (h.score - e).abs() / e.max(MIN_SCORE_FOR_RELATIVE_DELTA),
                    None => f64::INFINITY,
                })
                .fold(0.0, f64::max);
            rows.push(GradedRow {
                name: q.name,
                k,
                n_matches: top.n_matches,
                recall_textbook,
                recall_per_superfile_stats,
                recall_engine_model,
                max_delta,
            });
        }
    }
    rows
}

/// Grade through a supertable reader's public search path, under both
/// statistics scopes.
pub fn grade(
    oracle: &Oracle,
    reader: &SupertableReader,
    column: &str,
    tokenizer: &dyn Tokenizer,
    battery: &[QualityQuery],
) -> Vec<GradedRow> {
    grade_with(
        oracle,
        tokenizer,
        battery,
        &|q, k, stats| supertable_hits(reader, column, q, k, stats),
        true,
    )
}

/// Grade through one superfile's search path.
pub fn grade_superfile(
    oracle: &Oracle,
    reader: &SuperfileReader,
    column: &str,
    tokenizer: &dyn Tokenizer,
    battery: &[QualityQuery],
) -> Vec<GradedRow> {
    grade_with(
        oracle,
        tokenizer,
        battery,
        &|q, k, _| superfile_hits(reader, column, q, k),
        false,
    )
}

/// The oracle for this process's synthetic corpus at `n_docs`, with the
/// flavour and tokenizer both sides use. Logs the build.
fn build_oracle(
    n_docs: usize,
    text_seed: u64,
    log_prefix: &str,
) -> (Oracle, TextFlavor, Arc<dyn Tokenizer>) {
    corpus::require_synthetic(&format!("{log_prefix} quality"));
    let flavor = corpus::text_flavor();
    let tokenizer = default_tokenizer();
    eprintln!(
        "[{log_prefix}] quality: streaming {} {} docs into the BM25 oracle...",
        fmt_count(n_docs),
        flavor.label()
    );
    let t = Instant::now();
    let oracle = Oracle::build(
        n_docs,
        text_seed,
        flavor,
        tokenizer.as_ref(),
        QUALITY_BATTERY,
    );
    eprintln!(
        "[{log_prefix}] quality: oracle ready in {:.1}s ({} cells, avgdl {:.1})",
        t.elapsed().as_secs_f64(),
        fmt_count(oracle.entries.len()),
        oracle.avgdl
    );
    (oracle, flavor, tokenizer)
}

/// Fail loudly on any graded cell outside the gates.
fn assert_gates(rows: &[GradedRow], log_prefix: &str) {
    let failures: Vec<String> = rows
        .iter()
        .filter(|r| !r.passes())
        .map(GradedRow::failure)
        .collect();
    assert!(
        failures.is_empty(),
        "[{log_prefix}] quality gate failed:\n  {}",
        failures.join("\n  ")
    );
    eprintln!(
        "[{log_prefix}] quality OK: {} shapes × {} k within gates",
        QUALITY_BATTERY.len(),
        QUALITY_KS.len()
    );
}

/// Build the oracle for the configured corpus, recover the table's
/// layout, grade every battery shape through `reader`, emit the section
/// under `bench/fts/supertable/quality` and fail loudly if a gate is
/// missed.
pub fn run(
    report: &mut Report,
    reader: &SupertableReader,
    column: &str,
    n_docs: usize,
    text_seed: u64,
    log_prefix: &str,
) {
    let (mut oracle, flavor, tokenizer) = build_oracle(n_docs, text_seed, log_prefix);

    let t = Instant::now();
    let layout = Layout::recover(reader, column, oracle.doc_lengths());
    eprintln!(
        "[{log_prefix}] quality: layout verified in {:.1}s: {} superfiles from {} commits, \
         every stored doc length matches the oracle's",
        t.elapsed().as_secs_f64(),
        layout.n_superfiles(),
        layout.n_commits()
    );
    let n_superfiles = layout.n_superfiles();
    oracle.set_layout(layout);

    let rows = grade(&oracle, reader, column, tokenizer.as_ref(), QUALITY_BATTERY);
    emit(
        report,
        "bench/fts/supertable/quality",
        format!(
            "Supertable FTS — BM25 top-k parity vs textbook oracle ({} docs, {} corpus, {n_superfiles} superfiles)",
            fmt_count(n_docs),
            flavor.label()
        ),
        "each superfile's declared average, both read back from the table and verified against \
         the oracle's own tokenization",
        &rows,
        true,
    );
    assert_gates(&rows, log_prefix);
}

/// The superfile tier's quality phase: one file holding every row of the
/// configured corpus, built in corpus order. Verifies the file's stored
/// document lengths and declared average against the oracle, grades the
/// battery through the string search path, emits the section under
/// `bench/fts/superfile/quality` and fails loudly if a gate is missed.
/// With one file the declared average is the file's own, so Q is exact
/// by construction; this is where a kernel change is graded first.
pub fn run_superfile(
    report: &mut Report,
    reader: &SuperfileReader,
    column: &str,
    n_docs: usize,
    text_seed: u64,
    log_prefix: &str,
) {
    let (mut oracle, flavor, tokenizer) = build_oracle(n_docs, text_seed, log_prefix);

    let fts = reader
        .fts()
        .expect("the graded superfile has a full-text index");
    let stored = fts
        .column_doc_lengths(column)
        .unwrap_or_else(|e| panic!("read doc lengths of {column:?}: {e}"));
    verify_doc_lengths(&stored, oracle.doc_lengths(), 0, "the superfile");
    let declared = declared_avgdl(fts, column);
    let own = ColumnLengthStats::from_lengths(stored.iter().copied());
    let predicted = declared_under(&ColumnLengthStats::default(), &own);
    assert!(
        close_enough(declared, predicted),
        "the superfile declares avgdl {declared} but its own lengths average to {predicted}"
    );
    eprintln!(
        "[{log_prefix}] quality: superfile verified: {} stored doc lengths match the oracle's, \
         declared avgdl {declared:.3}",
        fmt_count(n_docs)
    );
    oracle.set_layout(Layout::single(n_docs, declared));

    let rows = grade_superfile(&oracle, reader, column, tokenizer.as_ref(), QUALITY_BATTERY);
    emit(
        report,
        "bench/fts/superfile/quality",
        format!(
            "Superfile FTS — BM25 top-k parity vs textbook oracle ({} docs, {} corpus)",
            fmt_count(n_docs),
            flavor.label()
        ),
        "the file's own declared average, both read back from the file and verified against the \
         oracle's own tokenization",
        &rows,
        false,
    );
    assert_gates(&rows, log_prefix);
}

/// Emit one tier's quality section. `engine_avgdl` names the average
/// length Q normalizes with on that tier, for the note; `per_superfile`
/// adds the opt-in statistics scope's column.
fn emit(
    report: &mut Report,
    anchor: &str,
    title: String,
    engine_avgdl: &str,
    rows: &[GradedRow],
    per_superfile: bool,
) {
    let mut headers = vec![
        "Query".to_string(),
        "matches".to_string(),
        "recall vs BM25".to_string(),
    ];
    if per_superfile {
        headers.push("recall (per-superfile stats)".to_string());
    }
    headers.push("recall vs engine BM25".to_string());
    headers.push("max score Δ".to_string());
    let blocks = QUALITY_KS
        .iter()
        .map(|&k| Block {
            subtitle: format!("k = {k}"),
            headers: headers.clone(),
            rows: rows
                .iter()
                .filter(|r| r.k == k)
                .map(|r| {
                    let mut cells = vec![
                        text(r.name),
                        text(fmt_count(r.n_matches)),
                        recall_cell(r.recall_textbook, false),
                    ];
                    if let Some(per_superfile_stats) = r.recall_per_superfile_stats {
                        cells.push(recall_cell(per_superfile_stats, false));
                    }
                    cells.push(recall_cell(r.recall_engine_model, true));
                    cells.push(metric(
                        r.max_delta,
                        format!("{:.4}%", r.max_delta * 100.0),
                        Better::Lower,
                    ));
                    cells
                })
                .collect(),
        })
        .collect();
    let per_superfile_note = if per_superfile {
        " `recall (per-superfile stats)` = the same under segment-local `PerSuperfile` idf (the \
         pre-0.7 default)."
    } else {
        ""
    };
    report.emit(&Section {
        anchor: anchor.into(),
        title,
        note: format!(
            "Engine top-k graded against a streaming BM25 oracle over the whole corpus, \
             tie-aware (a hit is any returned doc scoring at least the oracle's k-th score). \
             `recall vs BM25` = textbook BM25 with exact doc lengths and the corpus-wide average \
             length — the user-facing quality, which pays for the one-byte length quantization \
             and for each file normalizing with the average it declares.{per_superfile_note} \
             `recall vs engine BM25` = BM25 with the engine's stored (quantized) lengths and \
             {engine_avgdl} — gated at {floor:.1}: a miss is a document the kernels' own formula \
             would not have returned. `max score Δ` = largest relative gap between an engine \
             score and that reference for the same doc — gated at {ceiling:.2}%, f32 arithmetic \
             noise.",
            floor = MIN_ENGINE_MODEL_RECALL,
            ceiling = MAX_SCORE_DELTA * 100.0
        ),
        blocks,
    });
}

fn recall_cell(recall: f64, gate: bool) -> Cell {
    let shown = format!("{recall:.4}");
    if gate {
        metric(recall, shown, Better::Higher)
    } else {
        context(recall, shown, Better::Higher)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use infino::{
        storage::{LocalFsStorageProvider, StorageProvider},
        superfile::{builder::FtsConfig, fts::tokenize::Phrase},
        supertable::{Supertable, SupertableOptions},
        test_helpers::{brute_force_bm25::BruteForceBm25, build_title_batch, schema_id_title},
    };
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;

    use super::*;
    use crate::corpus::for_each_generated_doc;

    const TEST_SEED: u64 = 1;
    /// Small enough for the retained-token reference, large enough that
    /// every battery tier except `rare` has matches on the uniform flavour.
    const TEST_DOCS: usize = 600;
    const TEST_K: usize = 10;
    const SCORE_TOLERANCE: f64 = 1e-4;
    /// Commits the fragmented test table is ingested in.
    const TEST_COMMITS: usize = 3;
    /// Writer threads: with the byte split target disabled each commit
    /// splits into this many shard superfiles, so the layout recovery sees
    /// files of one commit that share their "committed before" totals.
    const TEST_WRITER_THREADS: usize = 2;

    fn corpus_rows(flavor: TextFlavor) -> Vec<(u64, String)> {
        let rows = Mutex::new(Vec::with_capacity(TEST_DOCS));
        for_each_generated_doc(TEST_DOCS, TEST_SEED, flavor, |doc_id, text| {
            rows.lock().unwrap().push((doc_id as u64, text.to_owned()));
        });
        let mut rows = rows.into_inner().unwrap();
        rows.sort_by_key(|(id, _)| *id);
        rows
    }

    /// The streaming oracle's textbook scores agree with the retained-token
    /// reference on every battery shape, for both flavours: same match
    /// count and the same top-k score multiset.
    #[test]
    fn oracle_textbook_matches_brute_force_reference() {
        let tokenizer = default_tokenizer();
        for flavor in [TextFlavor::Uniform, TextFlavor::Realistic] {
            let rows = corpus_rows(flavor);
            let borrowed: Vec<(u64, &str)> = rows.iter().map(|(id, t)| (*id, t.as_str())).collect();
            let reference = BruteForceBm25::index(&borrowed, tokenizer.as_ref());
            let oracle = Oracle::build(
                TEST_DOCS,
                TEST_SEED,
                flavor,
                tokenizer.as_ref(),
                QUALITY_BATTERY,
            );
            let resolved: Vec<ResolvedQuery> = QUALITY_BATTERY
                .iter()
                .map(|q| oracle.resolve(tokenizer.as_ref(), q))
                .collect();
            let mut any_matches = 0;
            for (q, rq) in QUALITY_BATTERY.iter().zip(&resolved) {
                let clauses = tokenizer.parse(q.query).into_clauses(q.mode);
                let owned = |v: &[Cow<'_, str>]| -> Vec<String> {
                    v.iter().map(|c| c.to_string()).collect()
                };
                let owned_phrases = |v: &[Phrase<Cow<'_, str>>]| -> Vec<Phrase<String>> {
                    v.iter().map(|p| p.map(|c| c.to_string())).collect()
                };
                let expected = reference.top_k_atoms(
                    &owned(&clauses.musts),
                    &owned_phrases(&clauses.must_phrases),
                    &owned(&clauses.shoulds),
                    &owned_phrases(&clauses.should_phrases),
                    &owned(&clauses.negatives),
                    &owned_phrases(&clauses.negative_phrases),
                    usize::MAX,
                );
                let mut got = oracle.matches(rq);
                got.sort_by(|a, b| {
                    b.textbook
                        .partial_cmp(&a.textbook)
                        .unwrap_or(Ordering::Equal)
                        .then(a.row.cmp(&b.row))
                });
                assert_eq!(
                    got.len(),
                    expected.len(),
                    "{flavor:?} {}: match count",
                    q.name
                );
                any_matches += got.len();
                for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        (g.textbook - f64::from(e.1)).abs() <= SCORE_TOLERANCE,
                        "{flavor:?} {} rank {i}: oracle {} vs reference {}",
                        q.name,
                        g.textbook,
                        e.1
                    );
                }
                // Per-row lookup agrees with the bulk walk.
                for g in got.iter().take(TEST_K) {
                    let (t, _) = oracle.score_row(g.row, rq).expect("matched row scores");
                    assert!((t - g.textbook).abs() <= SCORE_TOLERANCE);
                }
            }
            assert!(any_matches > 0, "{flavor:?}: battery matched nothing");
        }
    }

    /// With a single-span layout at the corpus average, the engine-model
    /// reference differs from textbook only through the stored length:
    /// identical for short docs, lower length (higher score) for long ones.
    #[test]
    fn engine_model_uses_stored_length() {
        let tokenizer = default_tokenizer();
        let oracle = Oracle::build(
            TEST_DOCS,
            TEST_SEED,
            TextFlavor::Realistic,
            tokenizer.as_ref(),
            QUALITY_BATTERY,
        );
        let stopword = oracle.resolve(tokenizer.as_ref(), &QUALITY_BATTERY[0]);
        let stopword = &stopword;
        let mut saw_exact = false;
        let mut saw_quantized = false;
        for s in oracle.matches(stopword) {
            let dl = oracle.dl[s.row as usize];
            if stored_len(dl) == dl {
                assert!((s.textbook - s.engine_model).abs() <= SCORE_TOLERANCE);
                saw_exact = true;
            } else {
                assert!(
                    s.engine_model > s.textbook,
                    "shorter stored length scores higher"
                );
                saw_quantized = true;
            }
        }
        assert!(
            saw_exact && saw_quantized,
            "realistic corpus spans both length regions"
        );
    }

    /// Ingest the realistic test corpus into a table in several commits
    /// with a multi-thread writer, so the manifest holds several
    /// superfiles per commit, each declaring the running table-wide
    /// average as of its commit.
    fn fragmented_table() -> (TempDir, Supertable) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(TEST_WRITER_THREADS)
                .build()
                .expect("writer pool"),
        );
        let options = SupertableOptions::new(
            schema_id_title(),
            vec![FtsConfig::new("title").positions(true)],
            vec![],
        )
        .expect("options")
        .with_writer_pool(pool)
        // Shard every commit across the whole pool regardless of its
        // size; the default byte target would keep a 200-row commit whole.
        .with_superfile_buffer_split_mb(0)
        .with_storage(storage);
        let st = Supertable::create(options).expect("create");
        let rows = corpus_rows(TextFlavor::Realistic);
        let chunk = TEST_DOCS.div_ceil(TEST_COMMITS);
        let mut w = st.writer().expect("writer");
        for commit in rows.chunks(chunk) {
            let titles: Vec<&str> = commit.iter().map(|(_, t)| t.as_str()).collect();
            w.append(&build_title_batch(&titles)).expect("append");
            w.commit().expect("commit");
        }
        drop(w);
        (dir, st)
    }

    /// The layout recovery assigns every row to the superfile that holds
    /// it (verified against the stored lengths), derives the commit
    /// grouping from the declared averages, and with that layout Q is
    /// exact: every battery shape reaches full recall against it and the
    /// engine's scores match it to f32 noise. The corpus-wide average
    /// would not: on a three-commit table the first commit's files score
    /// at a third of the corpus, and the gap shows in the score deltas.
    #[test]
    fn recovered_layout_makes_the_engine_model_exact() {
        let tokenizer = default_tokenizer();
        let (_dir, st) = fragmented_table();
        let reader = st.reader().expect("reader");
        let mut oracle = Oracle::build(
            TEST_DOCS,
            TEST_SEED,
            TextFlavor::Realistic,
            tokenizer.as_ref(),
            QUALITY_BATTERY,
        );

        // The corpus-wide average is measurably wrong for this table.
        let single = grade(
            &oracle,
            &reader,
            "title",
            tokenizer.as_ref(),
            QUALITY_BATTERY,
        );
        let worst_single = single.iter().map(|r| r.max_delta).fold(0.0, f64::max);
        assert!(
            worst_single > MAX_SCORE_DELTA,
            "a fragmented table must expose the corpus-wide average as inexact (worst Δ {worst_single})"
        );

        let layout = Layout::recover(&reader, "title", oracle.doc_lengths());
        assert_eq!(layout.n_commits(), TEST_COMMITS);
        assert!(
            layout.n_superfiles() > TEST_COMMITS,
            "the multi-thread writer splits a commit into several superfiles"
        );
        let covered: u64 = layout.spans.iter().map(|s| u64::from(s.n_docs)).sum();
        assert_eq!(covered, TEST_DOCS as u64);
        oracle.set_layout(layout);

        let rows = grade(
            &oracle,
            &reader,
            "title",
            tokenizer.as_ref(),
            QUALITY_BATTERY,
        );
        let failures: Vec<String> = rows
            .iter()
            .filter(|r| !r.passes())
            .map(GradedRow::failure)
            .collect();
        assert!(failures.is_empty(), "{}", failures.join("\n"));
        assert!(
            rows.iter().any(|r| r.n_matches > 0),
            "the battery matched nothing on the test table"
        );
    }

    #[test]
    fn layout_lookup_maps_rows_to_their_span() {
        let layout = Layout {
            spans: vec![
                SuperfileSpan {
                    first_row: 0,
                    n_docs: 3,
                    avgdl: 10.0,
                },
                SuperfileSpan {
                    first_row: 3,
                    n_docs: 2,
                    avgdl: 20.0,
                },
                SuperfileSpan {
                    first_row: 5,
                    n_docs: 1,
                    avgdl: 30.0,
                },
            ],
            n_commits: 2,
        };
        assert_eq!(layout.avgdl_of(0), 10.0);
        assert_eq!(layout.avgdl_of(2), 10.0);
        assert_eq!(layout.avgdl_of(3), 20.0);
        assert_eq!(layout.avgdl_of(4), 20.0);
        assert_eq!(layout.avgdl_of(5), 30.0);
    }

    #[test]
    fn tie_aware_recall_counts_boundary_ties_as_hits() {
        let scores = |row: u32| -> Option<f64> {
            match row {
                0 => Some(3.0),
                1..=3 => Some(1.0),
                _ => None,
            }
        };
        let hits = |rows: &[u32]| -> Vec<EngineHit> {
            rows.iter()
                .map(|&row| EngineHit { row, score: 0.0 })
                .collect()
        };
        // k = 2, k-th score 1.0: any of the tied rows 1..3 fills the second slot.
        let t = TIE_TOLERANCE;
        assert_eq!(tie_aware_recall(&hits(&[0, 3]), 2, 1.0, t, scores), 1.0);
        assert_eq!(tie_aware_recall(&hits(&[0, 9]), 2, 1.0, t, scores), 0.5);
        assert_eq!(tie_aware_recall(&hits(&[0]), 2, 1.0, t, scores), 0.5);
        assert_eq!(tie_aware_recall(&hits(&[]), 0, 0.0, t, scores), 1.0);
        assert_eq!(tie_aware_recall(&hits(&[0]), 0, 0.0, t, scores), 0.0);
        // A wider tolerance admits a near-miss at the boundary.
        let near = |r: u32| if r == 4 { Some(0.97) } else { scores(r) };
        assert_eq!(tie_aware_recall(&hits(&[0, 4]), 2, 1.0, t, near), 0.5);
        assert_eq!(tie_aware_recall(&hits(&[0, 4]), 2, 1.0, 0.05, near), 1.0);
    }

    #[test]
    fn phrase_tf_counts_adjacent_starts() {
        // positions: a b a b b a
        let pos = [(0, 0), (1, 1), (2, 0), (3, 1), (4, 1), (5, 0)];
        assert_eq!(phrase_tf(&pos, &[0, 1]), 2);
        assert_eq!(phrase_tf(&pos, &[1, 0]), 2);
        assert_eq!(phrase_tf(&pos, &[1, 1]), 1);
        assert_eq!(phrase_tf(&pos, &[0, 0]), 0);
        // A gap (position 7, not 6) breaks adjacency.
        assert_eq!(phrase_tf(&[(5, 0), (7, 1)], &[0, 1]), 0);
    }
}
