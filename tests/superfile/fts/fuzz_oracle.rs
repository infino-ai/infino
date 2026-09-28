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
/// each other they set the document frequency each *drawn* tier lands
/// at: at the default corpus size the first is in most documents, the
/// second in about a third, and the third in a few.
///
/// Every drawn tier dense enough to fill a block is also dense enough
/// for its block to store presence bits more cheaply than deltas, so the
/// denser ones are all **bitset** and only the sparsest is short-form. A
/// weight cannot separate them: lowering one until its deltas pack
/// narrower than a bitset drops its document frequency below the
/// short-form cap on the way. Random placement is the reason — the
/// widest gap in a block sets the delta width for all 128 lanes, and at
/// these densities that gap is wide. The delta-coded formats therefore
/// come from planted strides instead ([`TIER_PACKED`],
/// [`TIER_PATCHED`]), whose gaps are regular by construction.
const TIER_WEIGHT_MULTI_BLOCK: u32 = 12;
const TIER_WEIGHT_ONE_BLOCK: u32 = 4;
const TIER_WEIGHT_SHORT: u32 = 1;
/// Vocabulary index of each tier's token.
///
/// Planted in **every** document rather than drawn, so its posting count
/// is exactly the corpus size. Drawing it left the count a few percent
/// short of the documents — enough that in the large tier its list came
/// out just under the single-term walk's seed floor and the tier missed
/// the path it exists to reach. A planted term makes the block count
/// arithmetic, so [`LARGE_TIER_MIN_DOCS`] can be sized against the floor
/// instead of hoping a draw clears it.
const TIER_DENSE: usize = 0;
const TIER_MULTI_BLOCK: usize = 1;
const TIER_ONE_BLOCK: usize = 2;
const TIER_SHORT: usize = 3;
const TIER_SINGLETON: usize = 4;
/// Planted at a regular [`TIER_PACKED_STRIDE`], so every delta is the
/// same small number and the block packs them narrower than a presence
/// bitset over the same span: the **packed** form.
const TIER_PACKED: usize = 5;
/// Planted at [`TIER_PATCHED_STRIDE`] with a wider jump every
/// [`TIER_PATCHED_OUTLIER_EVERY`] postings, so a few lanes per block
/// need more bits than the rest: the **patched** form, which packs the
/// narrow majority and lists the outliers as exceptions.
const TIER_PATCHED: usize = 6;

/// Stride of [`TIER_PACKED`]. Three is the narrowest stride a block of
/// it packs into fewer bytes than a presence bitset over the same span:
/// at stride two the 128 postings span 254 documents, which is four
/// 64-bit presence words — exactly what two-bit deltas cost — and the
/// encoder keeps the bitset on a tie.
const TIER_PACKED_STRIDE: usize = 3;
/// Base stride of [`TIER_PATCHED`], between its wider jumps.
const TIER_PATCHED_STRIDE: usize = 3;
/// Postings between [`TIER_PATCHED`]'s wider jumps: a handful of
/// exception lanes per 128-lane block, which is where patching pays.
const TIER_PATCHED_OUTLIER_EVERY: usize = 24;
/// [`TIER_PATCHED`]'s wider jump. Wide enough that a block's span costs
/// more as presence words than its deltas do packed, so the bitset does
/// not claim the block, while the lanes needing that width stay few
/// enough for the exception list to beat widening all 128.
const TIER_PATCHED_JUMP: usize = 200;

/// Tokens [`tiered_corpus_strategy`] plants into a document on top of
/// its drawn ones — the dense tier in every document, and at most the
/// singleton, the packed and the patched tiers on top. Drawn lengths
/// leave room for these, so planting never evicts a token another tier
/// just planted.
const PLANTED_TOKENS_PER_DOC: usize = 4;

/// Smallest corpus the tiered lane draws. [`TIER_PACKED`] is long form
/// only once its stride has laid down more than the short-form cap of
/// 128 postings, so a corpus under `3 x 129` documents cannot show the
/// packed form at all. The other lanes vary the corpus size; this one
/// exists to span the posting formats, so it draws from the range where
/// they are all present.
const TIERED_MIN_DOCS: usize = TIER_PACKED_STRIDE * (BLOCK_POSTINGS + 1);
/// Postings a full block holds, mirrored from the format so the sizing
/// note above reads on its own.
const BLOCK_POSTINGS: usize = 128;

/// Postings the single-term ranked walk wants before it seeds its
/// threshold from the coarse block-max table: 256 blocks' worth. Stated
/// here rather than imported so that moving the walk's floor fails this
/// lane loudly instead of silently un-covering the seeded path.
const SEED_FLOOR_POSTINGS: usize = 256 * BLOCK_POSTINGS;

/// Blocks of headroom the large tier keeps over the seed floor. The
/// count is exact now that the dense tier is planted, so one block would
/// technically do; several keeps the lane covering the seeded path
/// through a modest change to the floor or to how a partial block is
/// counted, instead of falling off it silently again.
const SEED_FLOOR_HEADROOM_BLOCKS: usize = 8;

/// Smallest corpus of the large tiered tier. [`TIER_DENSE`] is planted
/// in every document, so its posting count is the corpus size: this
/// clears [`SEED_FLOOR_POSTINGS`] by [`SEED_FLOOR_HEADROOM_BLOCKS`]
/// rather than landing near it, and the walk seeds from the coarse table
/// and skips span by span, which no default-size case reaches.
const LARGE_TIER_MIN_DOCS: usize =
    SEED_FLOOR_POSTINGS + SEED_FLOOR_HEADROOM_BLOCKS * BLOCK_POSTINGS;
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

/// A corpus whose vocabulary spans every posting form at once. Three
/// terms are drawn per token, weighted by [`TIER_WEIGHT_MULTI_BLOCK`]
/// down to [`TIER_WEIGHT_SHORT`]; four more are planted after generation
/// — the dense tier in every document, the singleton in the first, and
/// the two strided tiers across the corpus. The remaining vocabulary
/// never appears, so a query naming it hits the absent-term paths.
///
/// What each form comes from, at the default corpus size:
///
/// | form | term | how |
/// |---|---|---|
/// | bitset, several blocks | [`TIER_DENSE`], and the denser drawn tiers | dense enough that presence bits cost no more than deltas |
/// | short (one bodiless block) | the sparsest drawn tier, and [`TIER_PATCHED`] | under the 128-posting cap |
/// | packed, two blocks | [`TIER_PACKED`] | planted at a uniform stride, so its deltas pack narrower than the span's presence words |
/// | inline df=1 | [`TIER_SINGLETON`] | planted in one document |
///
/// **Patched blocks are out of reach at this size**, and no weighting
/// fixes that. Patching pays when a few lanes are far wider than the
/// rest, but a block whose lanes are that uneven spans enough documents
/// for its presence bits to be cheaper still, so the bitset claims it
/// first. Escaping that needs a span of several hundred documents per
/// block *and* 128 postings to fill it, which a corpus of a few hundred
/// documents cannot supply at once — [`TIER_PATCHED`] is short form here
/// and only reaches the patched form in the large tier. The planted
/// corpora in `boundaries.rs` cover patched blocks exactly, including
/// the exception-count limit, so the fuzz lane is not the only coverage.
///
/// [`tiered_vocabulary_spans_the_posting_forms`] asserts this table
/// against the built index rather than trusting it.
///
/// At the large tier the dense list additionally runs past the
/// single-term walk's seed floor and spans several coarse tables, and
/// [`TIER_PATCHED`]'s jumps finally buy it the patched form.
fn tiered_corpus_strategy(
    n_docs: impl Strategy<Value = usize>,
) -> impl Strategy<Value = Vec<Vec<usize>>> {
    let token = prop_oneof![
        TIER_WEIGHT_MULTI_BLOCK => Just(TIER_MULTI_BLOCK),
        TIER_WEIGHT_ONE_BLOCK => Just(TIER_ONE_BLOCK),
        TIER_WEIGHT_SHORT => Just(TIER_SHORT),
    ];
    // Drawn lengths leave room for the planted tokens, so a document at
    // the cap does not lose one tier's token to the next tier's plant.
    let drawn_len = max_doc_len().saturating_sub(PLANTED_TOKENS_PER_DOC).max(1);
    let doc = prop::collection::vec(token, 1..=drawn_len);
    n_docs
        .prop_flat_map(move |n| prop::collection::vec(doc.clone(), n))
        .prop_map(|mut docs| {
            // The singleton goes in the first document; the strided
            // tiers at their own regular intervals. Planting rather than
            // drawing is what makes their gaps uniform, and uniform gaps
            // are what the delta-coded formats need — see the tier
            // weights' note.
            for doc in docs.iter_mut() {
                doc.push(TIER_DENSE);
            }
            if let Some(first) = docs.first_mut() {
                first.push(TIER_SINGLETON);
            }
            for doc in docs.iter_mut().step_by(TIER_PACKED_STRIDE) {
                doc.push(TIER_PACKED);
            }
            let mut at = 0usize;
            let mut planted = 0usize;
            while at < docs.len() {
                docs[at].push(TIER_PATCHED);
                planted += 1;
                at += match planted.is_multiple_of(TIER_PATCHED_OUTLIER_EVERY) {
                    true => TIER_PATCHED_JUMP,
                    false => TIER_PATCHED_STRIDE,
                };
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

/// Corpus sizes the default tiered lane draws from: at least
/// [`TIERED_MIN_DOCS`], so every case shows every form the lane claims.
/// Clamped when `INFINO_FTS_FUZZ_MAX_DOCS` is set below that, which
/// trades the packed form for the smaller corpus the caller asked for.
fn tiered_docs() -> impl Strategy<Value = usize> {
    let max = max_docs();
    TIERED_MIN_DOCS.min(max)..=max
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
        .map(|(d, s)| (u64::from(d.get()), s))
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
        corpus in tiered_corpus_strategy(tiered_docs()),
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

/// Corpora sampled when checking the tiered lane's posting forms.
const TIER_FORM_SAMPLES: usize = 4;

/// The tiered vocabulary lands on the posting forms
/// [`tiered_corpus_strategy`]'s table claims, checked against the built
/// index rather than argued from the weights.
///
/// The table is easy to get wrong and was: an earlier version claimed
/// packed and patched blocks from weighting alone, and the lane built
/// neither — every drawn tier dense enough to fill a block was dense
/// enough for the bitset to claim it, so the lane covered exactly what
/// the two older lanes already did. This test is what makes the claim
/// worth reading.
#[tokio::test]
async fn tiered_vocabulary_spans_the_posting_forms() {
    use proptest::{strategy::ValueTree, test_runner::TestRunner};

    let mut runner = TestRunner::deterministic();
    for sample in 0..TIER_FORM_SAMPLES {
        let docs = tiered_corpus_strategy(tiered_docs())
            .new_tree(&mut runner)
            .expect("tiered corpus")
            .current();
        let owned: Vec<(u64, String)> = docs
            .iter()
            .enumerate()
            .map(|(i, tokens)| {
                let text = tokens
                    .iter()
                    .map(|&t| VOCAB[t])
                    .collect::<Vec<_>>()
                    .join(" ");
                (i as u64, text)
            })
            .collect();
        let refs: Vec<(u64, &str)> = owned.iter().map(|(i, t)| (*i, t.as_str())).collect();
        let reader = build_infino_superfile_positional(&refs);
        let fts = reader.fts().expect("full-text index");
        let layout = async |tier: usize| {
            fts.term_layout("title", VOCAB[tier])
                .await
                .expect("term layout")
                .unwrap_or_else(|| panic!("sample {sample}: {} is absent", VOCAB[tier]))
        };

        let dense = layout(TIER_DENSE).await;
        assert!(
            dense.bitset_blocks > 0,
            "sample {sample}: the densest drawn tier should store presence bits, got {dense:?}"
        );
        let packed = layout(TIER_PACKED).await;
        assert!(
            packed.packed_blocks > 0 && !packed.short,
            "sample {sample}: the strided tier should pack its deltas, got {packed:?}"
        );
        let short = layout(TIER_SHORT).await;
        assert!(
            short.short,
            "sample {sample}: the sparsest drawn tier should take the short form, got {short:?}"
        );
        let singleton = layout(TIER_SINGLETON).await;
        assert!(
            singleton.inline && singleton.df == 1,
            "sample {sample}: the singleton should be inline, got {singleton:?}"
        );
        // Patched blocks need a span this corpus size cannot supply —
        // see the strategy's note. Pinning the fallback keeps the note
        // honest: if a format change ever makes patching reachable here,
        // this fails and the note gets rewritten.
        let patched = layout(TIER_PATCHED).await;
        assert!(
            patched.patched_blocks == 0 && patched.short,
            "sample {sample}: the jumped tier is short form at this size, got {patched:?}"
        );
    }
}

/// The jumped tier reaches the patched form once the corpus is large
/// enough to give its blocks a wide span, which is the other half of
/// [`tiered_corpus_strategy`]'s claim and the reason the tier exists at
/// all. Opt-in with the rest of the large tier: it indexes tens of
/// thousands of documents.
#[tokio::test]
#[ignore = "large corpus; run explicitly"]
async fn the_jumped_tier_is_patched_once_its_blocks_span_far_enough() {
    let docs: Vec<Vec<usize>> = (0..LARGE_TIER_MIN_DOCS).map(|_| Vec::new()).collect();
    let mut docs = docs;
    let mut at = 0usize;
    let mut planted = 0usize;
    while at < docs.len() {
        docs[at].push(TIER_PATCHED);
        planted += 1;
        at += match planted.is_multiple_of(TIER_PATCHED_OUTLIER_EVERY) {
            true => TIER_PATCHED_JUMP,
            false => TIER_PATCHED_STRIDE,
        };
    }
    let owned: Vec<(u64, String)> = docs
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let text = tokens
                .iter()
                .map(|&t| VOCAB[t])
                .collect::<Vec<_>>()
                .join(" ");
            (i as u64, text)
        })
        .collect();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, t)| (*i, t.as_str())).collect();
    let reader = build_infino_superfile_positional(&refs);
    let layout = reader
        .fts()
        .expect("full-text index")
        .term_layout("title", VOCAB[TIER_PATCHED])
        .await
        .expect("term layout")
        .expect("the jumped tier is present");
    assert!(
        layout.patched_blocks > 0,
        "the jumped tier should patch its outlier lanes at this size, got {layout:?}"
    );
}

/// The large tier's dense list crosses the single-term walk's seed
/// floor, which is the path that tier exists to reach.
///
/// It did not before: the dense term was drawn rather than planted, so
/// it missed a few percent of documents and its list came out at ~248
/// blocks against a floor of 256 — close enough to read as covered and
/// never actually seeded. Planting makes the count arithmetic, and this
/// asserts the arithmetic against the built index.
#[tokio::test]
#[ignore = "large corpus; run explicitly"]
async fn the_dense_tier_crosses_the_walks_seed_floor() {
    let docs: Vec<Vec<usize>> = (0..LARGE_TIER_MIN_DOCS).map(|_| vec![TIER_DENSE]).collect();
    let owned: Vec<(u64, String)> = docs
        .iter()
        .enumerate()
        .map(|(i, tokens)| {
            let text = tokens
                .iter()
                .map(|&t| VOCAB[t])
                .collect::<Vec<_>>()
                .join(" ");
            (i as u64, text)
        })
        .collect();
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, t)| (*i, t.as_str())).collect();
    let reader = build_infino_superfile_positional(&refs);
    let layout = reader
        .fts()
        .expect("full-text index")
        .term_layout("title", VOCAB[TIER_DENSE])
        .await
        .expect("term layout")
        .expect("the dense tier is present");
    let floor_blocks = SEED_FLOOR_POSTINGS / BLOCK_POSTINGS;
    assert!(
        layout.num_blocks > floor_blocks,
        "the dense tier should span more than the walk's {floor_blocks}-block seed floor, \
         got {} blocks ({:?})",
        layout.num_blocks,
        layout
    );
}
