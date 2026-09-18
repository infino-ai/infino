// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Property-based BM25 correctness oracle for the superfile FTS
//! pipeline.
//!
//! The hand-planted oracles in [`super::brute_force_oracle`] pin
//! individual kernels with carefully chosen mod-arithmetic corpora —
//! coverage is only as good as the cases someone thought to write, and
//! the router's branch *boundaries* (the exact df/k where dispatch
//! flips kernel) are lightly exercised. This module instead generates
//! random corpora and random clause/phrase queries and diffs the
//! reader against the textbook [`BruteForceBm25`] reference on every
//! case, so coverage tracks the router's actual behaviour space rather
//! than an enumerated list.
//!
//! ## Why this stays exact (no quantization slack)
//!
//! Production scores read a *quantized* length-norm table, so the
//! kernel oracles carry a `1e-3` tolerance. Here every generated doc
//! is capped at [`MAX_DOC_LEN_HARD`] (< `LEN_QUANT_EXACT_MAX = 16`)
//! tokens, which lands entirely in the norm table's exact region — the
//! reader and the oracle then compute the *same* length norm, and the
//! only residual is f64-vs-f32 idf rounding plus BM25 sum operand
//! order, both far under the `1e-3` bar. Norm quantization on long
//! docs is a separate concern, pinned by
//! [`super::brute_force_oracle::quantized_norms_preserve_topk_ranking_on_long_docs`].
//!
//! ## Comparison contract
//!
//! The query is generated as a *string* and both sides interpret it
//! through the same parser (`parse` + `into_clauses(mode)`), so clause
//! semantics can never diverge by construction. Then:
//!
//! * the returned match set is always a subset of the oracle's full
//!   match set, and its size is exactly `min(k, match_count)`;
//! * when `k` covers every match, the match sets are equal and each
//!   doc's score matches the oracle within tolerance;
//! * when `k` truncates, the *multiset* of returned scores equals the
//!   oracle's top-`k` scores within tolerance — robust to tie
//!   reordering at the head and at the k-boundary, which is the only
//!   place the reader and a doc-id-tie-breaking oracle may legitimately
//!   pick different docs for the same score.
//!
//! ## Scale (env-tunable)
//!
//! The default corpus/case sizes are capped so the suite stays fast in
//! CI; a deeper run is on-demand via env overrides (there is no nightly
//! lane): `PROPTEST_CASES`, `INFINO_FTS_FUZZ_MAX_DOCS`,
//! `INFINO_FTS_FUZZ_MAX_DOC_LEN` (hard-capped at 15 to keep norms
//! exact), `INFINO_FTS_FUZZ_MAX_ATOMS`. The one opt-in test past the
//! seed floor takes `INFINO_FTS_FUZZ_LARGE_CASES`.

use std::collections::HashSet;

use infino::{
    superfile::{SuperfileReader, fts::reader::BoolMode},
    test_helpers::{brute_force_bm25::BruteForceBm25, default_tokenizer},
};
use proptest::{prelude::*, test_runner::TestCaseError};

use crate::fts::brute_force_oracle::{build_infino_superfile_positional, oracle_top_k_atoms};

/// Small shared vocabulary. Kept short so terms co-occur, intersect,
/// and (across enough docs) form dense bitset blocks — the shapes the
/// router branches on — instead of every doc being disjoint.
const VOCAB: &[&str] = &[
    "alpha", "beta", "gamma", "delta", "rust", "async", "web", "tokio", "go", "data",
];

/// Absolute ceiling on generated doc length. Must stay `<
/// LEN_QUANT_EXACT_MAX` (16) so every doc's length norm is stored
/// exactly and the reader/oracle norms agree bit-for-bit.
const MAX_DOC_LEN_HARD: usize = 15;

/// Top-k values every fuzz case may draw besides a uniform one: `1` is
/// the sharpest ranking check (one wrong admission fails it), and `128`
/// / `129` straddle the two-term union's WAND-to-MaxScore cutoff, so the
/// router's two kernels are both graded on the same corpora.
const PINNED_KS: [usize; 3] = [1, 128, 129];

/// Tier weights of the tiered vocabulary, per token draw. Relative to
/// each other they set the document frequency each tier lands at; at
/// the default corpus size the first tier is in nearly every document
/// (bitset blocks), the second in most (several packed blocks), the
/// third in about a third (one long-form block past the short-form
/// cap), the fourth in a few (a short-form list), and the fifth token
/// is planted in one document only (the inline df=1 form).
const TIER_WEIGHT_DENSE: u32 = 40;
const TIER_WEIGHT_MULTI_BLOCK: u32 = 12;
const TIER_WEIGHT_ONE_BLOCK: u32 = 4;
const TIER_WEIGHT_SHORT: u32 = 1;
/// Vocabulary index of each tier's token.
const TIER_DENSE: usize = 0;
const TIER_MULTI_BLOCK: usize = 1;
const TIER_ONE_BLOCK: usize = 2;
const TIER_SHORT: usize = 3;
const TIER_SINGLETON: usize = 4;

/// Smallest corpus of the large tiered tier: past the single-term
/// walk's seed floor of 256 blocks (32 768 postings), so the dense
/// tier's list is seeded from its coarse table and skipped span by
/// span, which no default-size case reaches.
const LARGE_TIER_MIN_DOCS: usize = 33_000;
/// Documents the large tier may add past its minimum.
const LARGE_TIER_DOC_SPREAD: usize = 256;
/// Default cases of the large tier (`INFINO_FTS_FUZZ_LARGE_CASES`
/// overrides). Each builds a positional superfile over tens of
/// thousands of documents, so the tier is opt-in.
const LARGE_TIER_DEFAULT_CASES: usize = 4;

/// Score-equality tolerance. The two scorers share the identical BM25
/// formula and (with exact norms) identical inputs; this only absorbs
/// f64-vs-f32 idf rounding and sum operand order.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

/// Read a `usize` cap from an env var, falling back to `default`.
fn env_cap(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Default proptest case count — modest for CI, overridable via the
/// standard `PROPTEST_CASES` env var.
fn cases() -> u32 {
    env_cap("PROPTEST_CASES", 96) as u32
}

/// Max docs per generated corpus. Default crosses several 128-doc PFOR
/// blocks so block-crossing kernels are reached organically.
fn max_docs() -> usize {
    env_cap("INFINO_FTS_FUZZ_MAX_DOCS", 400).clamp(1, 4096)
}

/// Max tokens per generated doc — hard-clamped to keep norms exact.
fn max_doc_len() -> usize {
    env_cap("INFINO_FTS_FUZZ_MAX_DOC_LEN", 12).clamp(1, MAX_DOC_LEN_HARD)
}

/// Max clause atoms per generated query.
fn max_atoms() -> usize {
    env_cap("INFINO_FTS_FUZZ_MAX_ATOMS", 4).clamp(1, 6)
}

/// One generated query atom: a polarity (`+`/bare/`-`) and 1..=3 vocab
/// tokens (one token ⇒ a term, more ⇒ a phrase).
#[derive(Clone, Debug)]
struct Atom {
    /// 0 = must (`+`), 1 = should (bare), 2 = negative (`-`).
    polarity: u8,
    /// Indices into [`VOCAB`].
    tokens: Vec<usize>,
}

/// A corpus is `n_docs` docs, each a bag of vocab-token indices.
fn corpus_strategy() -> impl Strategy<Value = Vec<Vec<usize>>> {
    let doc = prop::collection::vec(0..VOCAB.len(), 1..=max_doc_len());
    prop::collection::vec(doc, 1..=max_docs())
}

/// Like [`corpus_strategy`] but with a **skewed** token distribution:
/// the first two vocab terms dominate (dense, bitset-encoded posting
/// lists) while the other eight are rare. A uniform vocabulary produces
/// near-equal document frequencies, so it rarely exercises the
/// df-ratio-gated router branches — the 2-term WAND rare-anchor
/// (`hi_df ≥ lo_df·16`) and the anchored OR count (`max_df ≥
/// others·8`). Skewing the frequencies makes those branches fire.
fn skewed_corpus_strategy() -> impl Strategy<Value = Vec<Vec<usize>>> {
    let token = prop_oneof![
        12 => 0usize..2,
        1 => 2usize..VOCAB.len(),
    ];
    let doc = prop::collection::vec(token, 1..=max_doc_len());
    prop::collection::vec(doc, 1..=max_docs())
}

/// A corpus whose vocabulary spans every posting encoding at once:
/// per token draw the first four vocabulary terms are weighted by
/// [`TIER_WEIGHT_DENSE`] down to [`TIER_WEIGHT_SHORT`], and the fifth
/// is planted once, in the first document, after generation. The rest
/// of the vocabulary never appears, so a query naming it hits the
/// absent-term paths. `n_docs` sets the scale: at the CI default the
/// tiers land on bitset, packed multi-block, single long-form block,
/// short form and inline df=1; at the large tier the dense list runs
/// past the seed floor and spans several coarse tables.
fn tiered_corpus_strategy(
    n_docs: impl Strategy<Value = usize>,
) -> impl Strategy<Value = Vec<Vec<usize>>> {
    let token = prop_oneof![
        TIER_WEIGHT_DENSE => Just(TIER_DENSE),
        TIER_WEIGHT_MULTI_BLOCK => Just(TIER_MULTI_BLOCK),
        TIER_WEIGHT_ONE_BLOCK => Just(TIER_ONE_BLOCK),
        TIER_WEIGHT_SHORT => Just(TIER_SHORT),
    ];
    let doc = prop::collection::vec(token, 1..=max_doc_len());
    n_docs
        .prop_flat_map(move |n| prop::collection::vec(doc.clone(), n))
        .prop_map(|mut docs| {
            // The singleton goes in the first document, replacing its
            // last draw when the document is already at the length cap.
            if let Some(first) = docs.first_mut() {
                if first.len() >= max_doc_len() {
                    first.pop();
                }
                first.push(TIER_SINGLETON);
            }
            docs
        })
}

/// Top-k for a case: one of the pinned values or a uniform draw.
fn k_strategy(max_docs: usize) -> impl Strategy<Value = usize> {
    prop_oneof![
        1 => prop::sample::select(PINNED_KS.to_vec()),
        3 => 1usize..=(max_docs + 16),
    ]
}

/// Cases of the large tiered tier.
fn large_tier_cases() -> u32 {
    env_cap("INFINO_FTS_FUZZ_LARGE_CASES", LARGE_TIER_DEFAULT_CASES) as u32
}

fn atoms_strategy() -> impl Strategy<Value = Vec<Atom>> {
    let atom = (0u8..3u8, prop::collection::vec(0..VOCAB.len(), 1..=3))
        .prop_map(|(polarity, tokens)| Atom { polarity, tokens });
    prop::collection::vec(atom, 1..=max_atoms())
}

/// Render an atom to its query-string form (`+term`, `term`, `-term`,
/// or the quoted-phrase variants).
fn render_atom(a: &Atom) -> String {
    let body = if a.tokens.len() == 1 {
        VOCAB[a.tokens[0]].to_string()
    } else {
        let words: Vec<&str> = a.tokens.iter().map(|&i| VOCAB[i]).collect();
        format!("\"{}\"", words.join(" "))
    };
    match a.polarity {
        0 => format!("+{body}"),
        2 => format!("-{body}"),
        _ => body,
    }
}

/// Build a well-formed query string from generated atoms: dedup
/// identical rendered atoms (repeated-term scoring is a deliberately
/// separate concern), and guarantee at least one positive clause so
/// the query never trips the `NegationOnly` error.
fn build_query(atoms: &[Atom]) -> String {
    let mut atoms = atoms.to_vec();
    // Guarantee a positive: if every atom is a negative, flip the first
    // to a bare should.
    if atoms.iter().all(|a| a.polarity == 2) {
        atoms[0].polarity = 1;
    }
    let mut seen = HashSet::new();
    let mut rendered = Vec::new();
    for a in &atoms {
        let r = render_atom(a);
        if seen.insert(r.clone()) {
            rendered.push(r);
        }
    }
    rendered.join(" ")
}

/// Shared tokio runtime — one per test-fn invocation, reused across all
/// proptest cases (each case only awaits I/O-free in-memory reads).
fn rt() -> &'static tokio::runtime::Runtime {
    use std::sync::OnceLock;
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("build fuzz runtime")
    })
}

/// Run one generated (corpus, query, mode, k) case and assert the
/// reader agrees with the brute-force oracle.
fn run_case(
    corpus_idx: &[Vec<usize>],
    atoms: &[Atom],
    and_mode: bool,
    k: usize,
) -> Result<(), TestCaseError> {
    // Materialize the corpus text; doc_id == row index so the reader's
    // local_doc_id is the user id (the invariant the oracle assumes).
    let owned: Vec<(u64, String)> = corpus_idx
        .iter()
        .enumerate()
        .map(|(i, toks)| {
            let text = toks.iter().map(|&t| VOCAB[t]).collect::<Vec<_>>().join(" ");
            (i as u64, text)
        })
        .collect();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();

    let reader: SuperfileReader = build_infino_superfile_positional(&refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&refs, tok.as_ref());

    let query = build_query(atoms);
    let mode = if and_mode {
        BoolMode::And
    } else {
        BoolMode::Or
    };

    // Reader result.
    let got: Vec<(u64, f32)> = rt()
        .block_on(reader.bm25_hits_async("title", &query, k, mode))
        .map_err(|e| {
            TestCaseError::fail(format!(
                "reader error on {query:?} (mode={mode:?}, k={k}): {e}"
            ))
        })?
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect();

    // Oracle full match set (k = n) via the same parsed clauses, so the
    // clause interpretation can never diverge from the reader's.
    let want_full = oracle_top_k_atoms(&oracle, tok.as_ref(), &query, mode, owned.len());

    let want_ids: HashSet<u64> = want_full.iter().map(|(d, _)| *d).collect();
    let got_ids: HashSet<u64> = got.iter().map(|(d, _)| *d).collect();

    // Cardinality: the reader returns exactly the k highest matches (no
    // floor here), i.e. min(k, match_count).
    let expected_len = k.min(want_full.len());
    prop_assert_eq!(
        got.len(),
        expected_len,
        "hit count: query={:?} mode={:?} k={} matches={}",
        query,
        mode,
        k,
        want_full.len()
    );

    // Every returned doc is a real match.
    prop_assert!(
        got_ids.is_subset(&want_ids),
        "returned docs not all matches: query={:?} mode={:?} k={} extra={:?}",
        query,
        mode,
        k,
        got_ids.difference(&want_ids).collect::<Vec<_>>()
    );

    if k >= want_full.len() {
        // Full result: exact set + per-doc score.
        prop_assert_eq!(
            &got_ids,
            &want_ids,
            "full-result set mismatch: query={:?} mode={:?}",
            query,
            mode
        );
        let want_scores: std::collections::HashMap<u64, f32> = want_full.iter().copied().collect();
        for (d, s) in &got {
            let w = want_scores[d];
            prop_assert!(
                (s - w).abs() <= SCORE_ABS_TOLERANCE,
                "score mismatch on doc {}: reader={} oracle={} query={:?} mode={:?}",
                d,
                s,
                w,
                query,
                mode
            );
        }
    }

    // Score multiset (both regimes): the returned scores, sorted
    // descending, equal the oracle's top-k scores sorted descending,
    // within tolerance. Ties may reorder docs but never scores.
    let mut got_scores: Vec<f32> = got.iter().map(|(_, s)| *s).collect();
    got_scores.sort_unstable_by(|a, b| b.partial_cmp(a).expect("finite scores"));
    let mut want_scores: Vec<f32> = want_full.iter().map(|(_, s)| *s).collect();
    want_scores.sort_unstable_by(|a, b| b.partial_cmp(a).expect("finite scores"));
    want_scores.truncate(k);
    prop_assert_eq!(
        got_scores.len(),
        want_scores.len(),
        "score-vec length: query={:?} mode={:?} k={}",
        query,
        mode,
        k
    );
    for (g, w) in got_scores.iter().zip(&want_scores) {
        prop_assert!(
            (g - w).abs() <= SCORE_ABS_TOLERANCE,
            "top-k score mismatch: reader={} oracle={} query={:?} mode={:?} k={}",
            g,
            w,
            query,
            mode,
            k
        );
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(), ..ProptestConfig::default() })]

    /// The reader's BM25 search agrees with textbook brute-force BM25
    /// on every generated (corpus, clause/phrase query, mode, k).
    #[test]
    fn fuzz_bm25_matches_brute_force(
        corpus in corpus_strategy(),
        atoms in atoms_strategy(),
        and_mode in any::<bool>(),
        k in k_strategy(max_docs()),
    ) {
        run_case(&corpus, &atoms, and_mode, k)?;
    }

    /// Same agreement contract as [`fuzz_bm25_matches_brute_force`], but
    /// over a skewed corpus so the rare-anchor and dominant-term router
    /// branches — which a uniform vocabulary rarely triggers — are
    /// exercised against the same brute-force reference.
    #[test]
    fn fuzz_bm25_skewed_vocab_matches_brute_force(
        corpus in skewed_corpus_strategy(),
        atoms in atoms_strategy(),
        and_mode in any::<bool>(),
        k in k_strategy(max_docs()),
    ) {
        run_case(&corpus, &atoms, and_mode, k)?;
    }

    /// Same contract over the tiered corpus, so every posting encoding
    /// — bitset, packed multi-block, one long-form block, short form,
    /// inline df=1 — is walked by the same generated clause and phrase
    /// shapes and graded against the same reference.
    #[test]
    fn fuzz_bm25_tiered_vocab_matches_brute_force(
        corpus in tiered_corpus_strategy(1usize..=max_docs()),
        atoms in atoms_strategy(),
        and_mode in any::<bool>(),
        k in k_strategy(max_docs()),
    ) {
        run_case(&corpus, &atoms, and_mode, k)?;
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: large_tier_cases(), ..ProptestConfig::default() })]

    /// The tiered contract past the single-term seed floor and across
    /// several coarse tables: the dense tier's list has more than 256
    /// blocks, so the threshold-first seed, the coarse-span skip and the
    /// per-block skip all run on every single-term case. Opt-in
    /// (`cargo test ... -- --ignored`): each case builds a superfile
    /// over tens of thousands of documents.
    #[test]
    #[ignore = "large corpora; run explicitly"]
    fn fuzz_bm25_tiered_large_matches_brute_force(
        corpus in tiered_corpus_strategy(
            LARGE_TIER_MIN_DOCS..=LARGE_TIER_MIN_DOCS + LARGE_TIER_DOC_SPREAD
        ),
        atoms in atoms_strategy(),
        and_mode in any::<bool>(),
        k in k_strategy(LARGE_TIER_MIN_DOCS),
    ) {
        run_case(&corpus, &atoms, and_mode, k)?;
    }
}
