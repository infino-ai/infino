// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 correctness oracle for the superfile FTS pipeline.
//!
//! Indexes a planted 60-doc corpus and asserts that infino's
//! optimized BMW / BMM walks return the same top-k as the
//! textbook BM25 reference implementation in
//! [`infino::test_helpers::brute_force_bm25`].
//!
//! ## What this oracle catches
//!
//! Planted-ground-truth tests verify that the pipeline returns
//! the *expected* docs but not that the *scoring math* is right —
//! a self-consistent BM25 bug (e.g. wrong avgdl handling) can
//! produce correct relative ranking on the planted set while
//! disagreeing with the actual BM25 formula. Comparing against
//! a textbook brute-force scorer catches this class: brute-force
//! is the BM25 math by direct construction, with no shared code
//! with the optimized walks.
//!
//! ## Tolerances
//!
//! Top-k *sets* must agree exactly on the head. Order within a
//! tied score may vary because brute-force breaks ties by
//! ascending doc-id while the optimized walks may break the same
//! tie differently. We assert "set equality" on the head, not
//! "ordered equality".

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    test_helpers::{brute_force_bm25::BruteForceBm25, decimal128_ids, default_tokenizer},
};

/// 60-doc planted corpus with mixed term frequencies. Enough to
/// make BM25's tf + idf + dl-norm interaction non-trivial, small
/// enough to keep the test fast.
pub fn corpus() -> Vec<(u64, &'static str)> {
    vec![
        (0, "rust async runtime tokio"),
        (1, "rust embedded systems firmware"),
        (2, "python data pipeline pandas"),
        (3, "python machine learning numpy"),
        (4, "javascript web frontend react"),
        (5, "javascript node backend server"),
        (6, "go concurrency goroutines channels"),
        (7, "go web framework gin echo"),
        (8, "rust web framework actix axum"),
        (9, "rust systems programming low level"),
        (10, "kubernetes pods deployment helm"),
        (11, "docker containers images registry"),
        (12, "postgresql replication wal logical"),
        (13, "mysql innodb redo log"),
        (14, "redis sorted sets pub sub"),
        (15, "kafka topics partitions consumers"),
        (16, "elasticsearch inverted index"),
        (17, "rare-token-zzz lucene rust search engine"),
        (18, "search engine bm25 ranking inverted"),
        (19, "vector search ann hnsw ivf"),
        (20, "rust async await futures"),
        (21, "rust ownership borrow checker lifetimes"),
        (22, "rust trait dyn impl async"),
        (23, "rust unsafe pointer manipulation"),
        (24, "linux kernel scheduler cfs"),
        (25, "linux network namespace netns"),
        (26, "windows powershell scripting"),
        (27, "macos darwin xcode swift"),
        (28, "ios swift uikit swiftui"),
        (29, "android kotlin jetpack compose"),
        (30, "tcp ip osi seven layers"),
        (31, "udp datagram unreliable fast"),
        (32, "http2 multiplexing streams binary"),
        (33, "http3 quic udp encrypted"),
        (34, "tls handshake certificate chain"),
        (35, "ssh key exchange rsa ed25519"),
        (36, "git rebase merge cherry pick"),
        (37, "git stash pop apply"),
        (38, "github pull request review approve"),
        (39, "ci cd pipeline github actions"),
        (40, "rust cargo build release profile"),
        (41, "rust crate publish workspace"),
        (42, "rust testing cfg test mod"),
        (43, "rust benchmark harnesses measure"),
        (44, "compiler optimization llvm ir"),
        (45, "compiler frontend parser ast"),
        (46, "interpreter virtual machine bytecode"),
        (47, "garbage collector mark sweep"),
        (48, "memory allocator slab arena"),
        (49, "memory mapped file mmap madvise"),
        (50, "concurrency lock free atomic"),
        (51, "concurrency mutex condvar wait"),
        (52, "rust send sync auto traits"),
        (53, "database transaction isolation"),
        (54, "database query optimizer plan"),
        (55, "data warehouse columnar storage"),
        (56, "parquet rowgroup metadata footer"),
        (57, "arrow record batch zero copy"),
        (58, "rust simd portable wide x86"),
        (59, "rust performance profiling perf"),
    ]
}

/// Build an infino superfile from the corpus.
pub fn build_infino_superfile(corpus: &[(u64, &str)]) -> SuperfileReader {
    build_infino_superfile_with(corpus, false)
}

/// Positional variant for the phrase oracle tests.
pub fn build_infino_superfile_positional(corpus: &[(u64, &str)]) -> SuperfileReader {
    build_infino_superfile_with(corpus, true)
}

fn build_infino_superfile_with(corpus: &[(u64, &str)], positions: bool) -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
            positions,
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(corpus.iter().map(|(i, _)| *i));
    let titles = LargeStringArray::from(corpus.iter().map(|(_, t)| *t).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(titles)])
        .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    SuperfileReader::open(bytes).expect("open superfile")
}

/// Run infino's BM25 search and return doc_ids in score-descending
/// order. The superfile is built so user `doc_id` matches the row
/// index 0..N-1, so the reader's `local_doc_id` IS the user id.
async fn infino_top_k(reader: &SuperfileReader, query: &str, k: usize) -> Vec<u64> {
    let hits = reader
        .bm25_hits_async("title", query, k, BoolMode::Or)
        .await
        .expect("BM25 search");
    hits.into_iter().map(|(d, _)| d as u64).collect()
}

/// Compare top-k *sets* between infino and brute-force for a query.
/// Asserts agreement on the head; allows tail divergence for ties.
async fn assert_top_k_head_agrees(
    infino: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    head_size: usize,
    k: usize,
) {
    let tok = default_tokenizer();
    let infino_hits = infino_top_k(infino, query, k).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k(query, k, tok.as_ref())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    assert!(
        infino_hits.len() >= head_size && oracle_hits.len() >= head_size,
        "query {query:?}: not enough hits — infino={infino_hits:?} oracle={oracle_hits:?}"
    );
    let infino_head: HashSet<u64> = infino_hits.into_iter().take(head_size).collect();
    let oracle_head: HashSet<u64> = oracle_hits.into_iter().take(head_size).collect();
    assert_eq!(
        infino_head, oracle_head,
        "query {query:?}: top-{head_size} sets disagree"
    );
}

#[tokio::test]
async fn oracle_rare_term_top1_matches() {
    // Single-term, single-doc match: "rare-token-zzz" is unique to
    // doc 17. Both engines must return [17] as top-1.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "rare-token-zzz", 1, 5).await;
}

#[tokio::test]
async fn oracle_common_term_top1_in_correct_set() {
    // "rust" appears in many same-length docs at mathematically tied
    // BM25 scores. We can't assert exact top-K agreement because
    // tie-breaking diverges, but BOTH engines must pick top-1 from
    // the docs that actually contain "rust".
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_top: u64 = infino_top_k(&infino, "rust", 1).await[0];
    let oracle_top: u64 = oracle
        .top_k("rust", 1, tok.as_ref())
        .first()
        .expect("oracle returns at least one hit")
        .0;
    let rust_docs: HashSet<u64> = corp
        .iter()
        .filter(|(_, t)| t.split_whitespace().any(|w| w == "rust"))
        .map(|(i, _)| *i)
        .collect();
    assert!(
        rust_docs.contains(&infino_top),
        "infino top-1 doc {infino_top} doesn't contain 'rust'"
    );
    assert!(
        rust_docs.contains(&oracle_top),
        "oracle top-1 doc {oracle_top} doesn't contain 'rust'"
    );
}

#[tokio::test]
async fn oracle_two_term_or_top1_matches() {
    // "redis kafka" — doc 14 has "redis", doc 15 has "kafka". Both
    // single-occurrence docs; either could be top-1. Top-2 set must
    // agree.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "redis kafka", 2, 5).await;
}

#[tokio::test]
async fn oracle_two_term_overlap_top3_matches() {
    // "rust async" — docs 0 and 20 contain both terms, so they should
    // rank highest under any sensible BM25.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_hits = infino_top_k(&infino, "rust async", 5).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k("rust async", 5, tok.as_ref())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    let infino_head: HashSet<u64> = infino_hits.into_iter().take(2).collect();
    let oracle_head: HashSet<u64> = oracle_hits.into_iter().take(2).collect();
    assert!(
        infino_head.contains(&0) && infino_head.contains(&20),
        "infino top-2 should contain docs 0+20 (both 'rust' and 'async'); got {infino_head:?}"
    );
    assert!(
        oracle_head.contains(&0) && oracle_head.contains(&20),
        "oracle top-2 should contain docs 0+20; got {oracle_head:?}"
    );
    assert_eq!(infino_head, oracle_head);
}

#[tokio::test]
async fn oracle_three_term_query_top5_set_matches() {
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "rust web framework", 3, 10).await;
}

#[tokio::test]
async fn oracle_no_match_query_returns_empty() {
    // "xyzzy" is in none of the docs; both engines must return empty.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_hits = infino_top_k(&infino, "xyzzy", 5).await;
    let oracle_hits = oracle.top_k("xyzzy", 5, tok.as_ref());
    assert!(
        infino_hits.is_empty(),
        "infino should return [] for unknown term"
    );
    assert!(
        oracle_hits.is_empty(),
        "oracle should return [] for unknown term"
    );
}

// ─── AND-mode oracles ─────────────────────────────────────────────────

async fn infino_top_k_and(reader: &SuperfileReader, query: &str, k: usize) -> Vec<u64> {
    // The reader's `bm25_search` consumes a pre-built query string,
    // tokenizes it column-internally, and runs the AND intersection.
    // Returned `local_doc_id` == user `doc_id` thanks to the planted
    // 0..N row layout.
    let hits = reader
        .bm25_hits_async("title", query, k, BoolMode::And)
        .await
        .expect("AND BM25 search");
    hits.into_iter().map(|(d, _)| d as u64).collect()
}

async fn assert_top_k_and_set_matches(
    infino: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    head_size: usize,
    k: usize,
) {
    let tok = default_tokenizer();
    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each(query, &mut |t| terms.push(t.to_owned()));
    let infino_hits = infino_top_k_and(infino, query, k).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k_terms_and(&terms, k)
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    assert!(
        infino_hits.len() >= head_size && oracle_hits.len() >= head_size,
        "AND query {query:?}: not enough hits — infino={infino_hits:?} oracle={oracle_hits:?}"
    );
    let infino_head: HashSet<u64> = infino_hits.into_iter().take(head_size).collect();
    let oracle_head: HashSet<u64> = oracle_hits.into_iter().take(head_size).collect();
    assert_eq!(
        infino_head, oracle_head,
        "AND query {query:?}: top-{head_size} sets disagree"
    );
}

#[tokio::test]
async fn oracle_and_two_term_overlap_top3_matches() {
    // "rust" and "async" co-occur only in docs 0, 20, 22. Both engines
    // must return exactly that set as the AND result.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_set: HashSet<u64> = infino_top_k_and(&infino, "rust async", 10)
        .await
        .into_iter()
        .collect();
    assert_eq!(
        infino_set,
        HashSet::from([0u64, 20, 22]),
        "AND(rust, async) must be exactly {{0, 20, 22}}; got {infino_set:?}"
    );
    assert_top_k_and_set_matches(&infino, &oracle, "rust async", 3, 10).await;
}

#[tokio::test]
async fn oracle_and_three_term_singleton_match() {
    // "rust async tokio" all co-occur only in doc 0. Tightens the
    // intersection to one doc and verifies the leapfrog over three
    // cursors reduces correctly.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let infino_hits = infino_top_k_and(&infino, "rust async tokio", 10).await;
    assert_eq!(
        infino_hits,
        vec![0u64],
        "AND(rust, async, tokio) must be exactly [0]; got {infino_hits:?}"
    );
}

#[tokio::test]
async fn oracle_and_missing_term_returns_empty() {
    // A term that's absent from the entire corpus must short-circuit
    // AND to empty — even though "rust" alone has many hits.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let hits = infino_top_k_and(&infino, "rust definitely-not-a-token", 10).await;
    assert!(
        hits.is_empty(),
        "AND with missing term must return []; got {hits:?}"
    );
}

#[tokio::test]
async fn oracle_and_disjoint_terms_return_empty() {
    // Two terms that both appear in the corpus but never co-occur
    // ("python" in docs 2-3; "kafka" in doc 15). AND yields no docs.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let hits = infino_top_k_and(&infino, "python kafka", 10).await;
    assert!(
        hits.is_empty(),
        "AND with disjoint posting lists must return []; got {hits:?}"
    );
}

#[tokio::test]
async fn oracle_and_scores_match_brute_force_ordering() {
    // For docs in the AND intersection of "rust" and "framework"
    // (only doc 8), the per-doc BM25 score must match brute force
    // bit-exactly — there's no rank ambiguity, so we can compare
    // values directly.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each("rust framework", &mut |t| terms.push(t.to_owned()));

    let infino_hits: Vec<(u64, f32)> = infino
        .bm25_hits_async("title", "rust framework", 10, BoolMode::And)
        .await
        .expect("AND search")
        .into_iter()
        .map(|(d, s)| (d as u64, s))
        .collect();
    let oracle_hits = oracle.top_k_terms_and(&terms, 10);
    assert_eq!(
        infino_hits.len(),
        oracle_hits.len(),
        "AND(rust, framework) hit counts disagree: infino={infino_hits:?} oracle={oracle_hits:?}"
    );
    for ((i_doc, i_score), (o_doc, o_score)) in infino_hits.iter().zip(oracle_hits.iter()) {
        assert_eq!(*i_doc, *o_doc, "doc-id mismatch");
        // f32 BM25 sums diverge by ~1e-4 between the two scorers due
        // to operand ordering (infino precomputes idf_x_k1p1 and
        // dl_norm_k1; the oracle multiplies term-by-term). 1e-3 is
        // tighter than any meaningful BM25 score gap on this corpus.
        let delta = (i_score - o_score).abs();
        assert!(
            delta < BM25_SCORE_ABS_TOLERANCE,
            "score divergence on doc {i_doc}: infino={i_score} oracle={o_score} delta={delta}"
        );
    }
}

#[tokio::test]
async fn oracle_and_single_term_routed_consistently() {
    // BoolMode::And with a single term must route the same as
    // BoolMode::Or (both fall through to the single-term BMW path).
    // Asserting symmetry catches the case where AND's branch
    // accidentally skips the early single-term short-circuit.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let and_hits = infino_top_k_and(&infino, "rare-token-zzz", 5).await;
    let or_hits = infino_top_k(&infino, "rare-token-zzz", 5).await;
    assert_eq!(and_hits, or_hits);
    assert_eq!(and_hits, vec![17u64]);
}

// ─── (resume existing OR oracles) ─────────────────────────────────────

#[tokio::test]
async fn oracle_long_doc_vs_short_doc_dl_norm() {
    // BM25's dl-norm should make short docs that contain a term rank
    // higher than long docs containing the same term once. Doc 7
    // ("go web framework gin echo", 5 tokens) and doc 8 ("rust web
    // framework actix axum", 5 tokens) both contain "framework"
    // exactly once at the same dl. Top-1 may tie-break either way but
    // top-2 set must include both.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_hits = infino_top_k(&infino, "framework", 5).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k("framework", 5, tok.as_ref())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    let infino_top2: HashSet<u64> = infino_hits.into_iter().take(2).collect();
    let oracle_top2: HashSet<u64> = oracle_hits.into_iter().take(2).collect();
    assert_eq!(infino_top2, oracle_top2, "framework top-2 sets disagree");
}

// ─── Multi-block AND oracles ──────────────────────────────────────────
//
// The 60-doc corpus above holds every term in a single PFOR block
// (BLOCK_LEN = 128). Block-crossing paths in `run_and_intersect_*` —
// the inner-loop `next()` cross-block walk, the alignment step that
// fires when a non-leader cursor lands in a new block, and the
// block-max-AND pruning that skips a whole leader block — only fire
// when terms span multiple blocks. This section plants a 1000-doc
// corpus with deterministic-frequency terms chosen so the common
// terms span 2–4 blocks each, then runs AND across 2/3/4 terms and
// compares against the brute-force oracle.

const MULTI_BLOCK_N_DOCS: u64 = 1_000;

/// Planting periods for the multi-block corpus terms (every Nth doc),
/// shared by the corpus builder and the AND-truth predicate.
const TERM_ALPHA_PERIOD: u64 = 3;
const TERM_BETA_PERIOD: u64 = 4;
const TERM_GAMMA_PERIOD: u64 = 5;
const TERM_DELTA_PERIOD: u64 = 7;
const TERM_EPSILON_PERIOD: u64 = 20;
/// Planting period for equal-frequency terms in the multi-block OR oracle.
const UNIFORM_OR_TERM_PERIOD: u64 = 2;
/// Number of document-length variants in the multi-block OR oracle.
const OR_DOC_LENGTH_VARIANTS: u64 = 5;
/// Filler-token bucket count (`no00`..`no49`) for doc-length variation.
const FILLER_TERM_MODULUS: u64 = 50;
/// AND top-k retrieving the full large intersection.
const MULTI_BLOCK_AND_K: usize = 200;
/// AND top-k for the rarer `alpha ∧ epsilon` intersection.
const MULTI_BLOCK_RARE_AND_K: usize = 100;
/// Score-equality tolerance comparing the two BM25 scorers.
const BM25_SCORE_ABS_TOLERANCE: f32 = 1e-3;

/// Deterministic-frequency planted corpus. Each doc is identified
/// by its position 0..N-1 and seeded with terms based on simple
/// mod predicates so the resulting posting list lengths are
/// predictable:
///
/// * `alpha` — every 3rd doc        → ~334 postings → 3 blocks
/// * `beta`  — every 4th doc        → ~250 postings → 2 blocks
/// * `gamma` — every 5th doc        → ~200 postings → 2 blocks
/// * `delta` — every 7th doc        → ~143 postings → 2 blocks
/// * `epsilon` — every 20th doc     → ~50 postings  → 1 block
/// * `noXXX` — per-doc filler tokens to vary doc lengths
pub fn build_multi_block_corpus() -> Vec<(u64, String)> {
    let mut out: Vec<(u64, String)> = Vec::with_capacity(MULTI_BLOCK_N_DOCS as usize);
    for d in 0..MULTI_BLOCK_N_DOCS {
        let mut toks: Vec<&'static str> = Vec::new();
        if d.is_multiple_of(TERM_ALPHA_PERIOD) {
            toks.push("alpha");
        }
        if d.is_multiple_of(TERM_BETA_PERIOD) {
            toks.push("beta");
        }
        if d.is_multiple_of(TERM_GAMMA_PERIOD) {
            toks.push("gamma");
        }
        if d.is_multiple_of(TERM_DELTA_PERIOD) {
            toks.push("delta");
        }
        if d.is_multiple_of(TERM_EPSILON_PERIOD) {
            toks.push("epsilon");
        }
        // Filler keeps every doc non-empty and gives a few different
        // doc lengths so dl-norm isn't a constant. Using mod-50 yields
        // 50 distinct filler terms across 1000 docs.
        let filler = format!("no{:02}", d % FILLER_TERM_MODULUS);
        let mut s = toks.join(" ");
        if !s.is_empty() {
            s.push(' ');
        }
        s.push_str(&filler);
        out.push((d, s));
    }
    out
}

pub fn build_multi_block_reader(owned: &[(u64, String)]) -> SuperfileReader {
    let refs: Vec<(u64, &str)> = owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    build_infino_superfile(&refs)
}

#[tokio::test]
async fn oracle_or_multi_block_scores_match_brute_force() {
    // Equal posting lists give these terms identical upper bounds, so
    // the public OR dispatcher selects windowed union by construction.
    // The lists span several blocks with full SIMD batches and a scalar
    // final-block tail; extra filler varies document-length norms.
    let mut corp_owned = Vec::with_capacity(MULTI_BLOCK_N_DOCS as usize);
    for doc in 0..MULTI_BLOCK_N_DOCS {
        let mut text = if doc.is_multiple_of(UNIFORM_OR_TERM_PERIOD) {
            String::from("alpha beta gamma delta")
        } else {
            String::from("filler")
        };
        for _ in 0..doc % OR_DOC_LENGTH_VARIANTS {
            text.push_str(" filler");
        }
        corp_owned.push((doc, text));
    }
    let corp_refs: Vec<(u64, &str)> = corp_owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let reader = build_infino_superfile(&corp_refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp_refs, tok.as_ref());
    let query = "alpha beta gamma delta";

    let mut infino_hits = reader
        .bm25_hits_async("title", query, MULTI_BLOCK_N_DOCS as usize, BoolMode::Or)
        .await
        .expect("OR search");
    let mut oracle_hits = oracle.top_k(query, MULTI_BLOCK_N_DOCS as usize, tok.as_ref());
    infino_hits.sort_unstable_by_key(|(doc, _)| *doc);
    oracle_hits.sort_unstable_by_key(|(doc, _)| *doc);

    assert_eq!(
        infino_hits.len(),
        oracle_hits.len(),
        "OR hit counts disagree"
    );
    for ((infino_doc, infino_score), (oracle_doc, oracle_score)) in
        infino_hits.iter().zip(&oracle_hits)
    {
        assert_eq!(*infino_doc as u64, *oracle_doc, "OR doc-id mismatch");
        let delta = (infino_score - oracle_score).abs();
        assert!(
            delta < BM25_SCORE_ABS_TOLERANCE,
            "score divergence on doc {infino_doc}: infino={infino_score} oracle={oracle_score} delta={delta}"
        );
    }
}

/// Compute the expected AND intersection for the multi-block corpus
/// using the same planting predicates as `build_multi_block_corpus`.
/// Returns the set of doc-ids in the intersection.
fn multi_block_and_truth(terms: &[&str]) -> HashSet<u64> {
    let predicate = |d: u64, t: &str| -> bool {
        match t {
            "alpha" => d.is_multiple_of(TERM_ALPHA_PERIOD),
            "beta" => d.is_multiple_of(TERM_BETA_PERIOD),
            "gamma" => d.is_multiple_of(TERM_GAMMA_PERIOD),
            "delta" => d.is_multiple_of(TERM_DELTA_PERIOD),
            "epsilon" => d.is_multiple_of(TERM_EPSILON_PERIOD),
            _ => false,
        }
    };
    (0..MULTI_BLOCK_N_DOCS)
        .filter(|d| terms.iter().all(|t| predicate(*d, t)))
        .collect()
}

#[tokio::test]
async fn oracle_and_multi_block_two_term_matches_brute_force() {
    // alpha ∧ beta: both span >1 block (3 + 2). Intersection is
    // docs where d % lcm(3,4) == 0, i.e., d % 12 == 0 → 84 matches
    // distributed across the corpus, forcing the 2-term flat-merge
    // path to cross blocks on both cursors.
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let infino_set: HashSet<u64> = infino_top_k_and(&r, "alpha beta", MULTI_BLOCK_AND_K)
        .await
        .into_iter()
        .collect();
    let truth = multi_block_and_truth(&["alpha", "beta"]);
    assert_eq!(
        infino_set, truth,
        "AND(alpha, beta) over multi-block corpus disagrees with planted truth"
    );
}

#[tokio::test]
async fn oracle_and_multi_block_three_term_matches_brute_force() {
    // alpha ∧ beta ∧ gamma: all span >1 block. Intersection is docs
    // where d % lcm(3,4,5) == 0, i.e., d % 60 == 0. Exercises the
    // n>=3 flat-merge `for o in others.iter_mut()` inner loop with
    // both branches of the match/no-match split and the block
    // crossings on three cursors simultaneously.
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let infino_set: HashSet<u64> = infino_top_k_and(&r, "alpha beta gamma", MULTI_BLOCK_AND_K)
        .await
        .into_iter()
        .collect();
    let truth = multi_block_and_truth(&["alpha", "beta", "gamma"]);
    assert_eq!(
        infino_set, truth,
        "AND(alpha, beta, gamma) over multi-block corpus disagrees with planted truth"
    );
}

#[tokio::test]
async fn oracle_and_multi_block_four_term_matches_brute_force() {
    // alpha ∧ beta ∧ gamma ∧ delta: all four span >1 block.
    // Intersection is d % lcm(3,4,5,7) == 0, i.e., d % 420 == 0 →
    // 3 matches at most {0, 420, 840} in a 1000-doc corpus.
    // Drives the cursor-alignment + flat-merge over four cursors
    // and tests the `block_exhausted` early-break path.
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let infino_set: HashSet<u64> =
        infino_top_k_and(&r, "alpha beta gamma delta", MULTI_BLOCK_AND_K)
            .await
            .into_iter()
            .collect();
    let truth = multi_block_and_truth(&["alpha", "beta", "gamma", "delta"]);
    assert_eq!(
        infino_set, truth,
        "AND(alpha, beta, gamma, delta) over multi-block corpus disagrees with planted truth"
    );
}

#[tokio::test]
async fn oracle_and_multi_block_with_rare_term_short_circuits() {
    // alpha (common, multi-block) ∧ epsilon (rare, single block).
    // The leapfrog picks the rarer (epsilon) as leader and walks
    // its single block; the alpha cursor must cross several blocks
    // as alignment proceeds, exercising the leader-side alignment
    // path that crosses block_last_doc_id.
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let infino_set: HashSet<u64> = infino_top_k_and(&r, "alpha epsilon", MULTI_BLOCK_RARE_AND_K)
        .await
        .into_iter()
        .collect();
    let truth = multi_block_and_truth(&["alpha", "epsilon"]);
    assert_eq!(
        infino_set, truth,
        "AND(alpha, epsilon) over multi-block corpus disagrees with planted truth"
    );
}

#[tokio::test]
async fn oracle_and_multi_block_top_k_smaller_than_match_count() {
    // top-k=5 against an AND that has ~84 matches. Once the heap
    // fills, the block-max-AND pruning check at the top of the
    // outer loop fires on every subsequent leader block whose UB
    // can't beat the kth-best score. Verifies the top-K matches
    // are a subset of the planted truth (every returned doc is a
    // real match; ranking-tie tail may differ from any specific
    // brute-force order).
    let corp = build_multi_block_corpus();
    let r = build_multi_block_reader(&corp);
    let infino_hits = infino_top_k_and(&r, "alpha beta", 5).await;
    assert_eq!(infino_hits.len(), 5, "top-k=5 should fill");
    let truth = multi_block_and_truth(&["alpha", "beta"]);
    for d in &infino_hits {
        assert!(
            truth.contains(d),
            "top-5 AND returned doc {d} that isn't in the planted intersection {truth:?}"
        );
    }
}

#[tokio::test]
async fn oracle_and_multi_block_score_matches_brute_force() {
    // Cross-check scores against the brute-force scorer on the
    // multi-block corpus. The two-term AND has 84 matches and the
    // top-10 list must agree on doc-id sets with brute force (ties
    // may reorder within a single score class). Catches scoring
    // drift introduced by the block-crossing code paths in the
    // flat-merge (e.g. wrong `block_tfs[pos]` index after a block
    // boundary, or a stale `idf_x_k1p1` if the cursor was
    // reconstructed mid-walk).
    let corp_owned = build_multi_block_corpus();
    let corp_refs: Vec<(u64, &str)> = corp_owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile(&corp_refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp_refs, tok.as_ref());

    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each("alpha beta", &mut |t| terms.push(t.to_owned()));
    let infino_hits = infino_top_k_and(&r, "alpha beta", 10).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k_terms_and(&terms, 10)
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    let infino_set: HashSet<u64> = infino_hits.iter().copied().collect();
    let oracle_set: HashSet<u64> = oracle_hits.iter().copied().collect();
    assert_eq!(
        infino_set, oracle_set,
        "multi-block AND top-10 sets disagree: infino={infino_hits:?} oracle={oracle_hits:?}"
    );
}

#[tokio::test]
async fn oracle_and_multi_block_two_term_score_values_match_brute_force() {
    // Scores (not just doc-id sets) for a two-term AND whose match set
    // spans multiple posting blocks — cross-checked against the exact
    // BM25 oracle so a block-crossing scoring bug (wrong tf index after
    // a block boundary, stale norm) can't slip through.
    //
    // alpha (÷3, spans 3 blocks) ∧ gamma (÷5, spans 2 blocks) →
    // d ÷ 15 → docs {0, 15, …, 990} = 67 matches, crossing blocks on
    // both cursors mid-walk.
    let corp_owned = build_multi_block_corpus();
    let corp_refs: Vec<(u64, &str)> = corp_owned.iter().map(|(i, s)| (*i, s.as_str())).collect();
    let r = build_infino_superfile(&corp_refs);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp_refs, tok.as_ref());
    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each("alpha gamma", &mut |t| terms.push(t.to_owned()));

    // Oracle scores for the full intersection, indexed by doc-id and
    // also as a descending score vector for the top-k comparison.
    let oracle_full = oracle.top_k_terms_and(&terms, MULTI_BLOCK_N_DOCS as usize);
    let oracle_by_doc: HashMap<u64, f32> = oracle_full.iter().copied().collect();
    assert_eq!(
        oracle_full.len(),
        67,
        "corpus assumption changed: alpha∧gamma should be 67 docs"
    );

    // Regime 1 — no pruning: k >= match count, so the heap never fills,
    // the block-max bar stays -inf, and every match is batch-scored.
    // Each returned score must match the oracle for that exact doc.
    let infino_full = r
        .bm25_hits_async(
            "title",
            "alpha gamma",
            MULTI_BLOCK_N_DOCS as usize,
            BoolMode::And,
        )
        .await
        .expect("AND search (full)");
    assert_eq!(infino_full.len(), 67, "full AND(alpha, gamma) count");
    for (doc, score) in &infino_full {
        let expected = oracle_by_doc
            .get(&(*doc as u64))
            .expect("returned doc not in oracle intersection");
        let delta = (score - expected).abs();
        assert!(
            delta < BM25_SCORE_ABS_TOLERANCE,
            "score divergence on doc {doc}: infino={score} oracle={expected} delta={delta}"
        );
    }

    // Regime 2 — pruning active: k < match count, so the heap fills and
    // the block-max-AND skip fires between blocks while the batched
    // scorer flushes each block's tail before the bar is re-read. The
    // ten best scores must equal the oracle's ten best (compared by
    // value, so a score-tie at the boundary can't make this flaky).
    let mut infino_top: Vec<f32> = r
        .bm25_hits_async("title", "alpha gamma", 10, BoolMode::And)
        .await
        .expect("AND search (top-k)")
        .into_iter()
        .map(|(_, s)| s)
        .collect();
    assert_eq!(infino_top.len(), 10, "top-k=10 should fill");
    infino_top.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    let mut oracle_top: Vec<f32> = oracle_full.iter().map(|(_, s)| *s).collect();
    oracle_top.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
    oracle_top.truncate(10);
    for (got, want) in infino_top.iter().zip(&oracle_top) {
        let delta = (got - want).abs();
        assert!(
            delta < BM25_SCORE_ABS_TOLERANCE,
            "top-10 score divergence: infino={got} oracle={want} delta={delta}"
        );
    }
}

#[tokio::test]
async fn quantized_norms_preserve_topk_ranking_on_long_docs() {
    // The length-norm table is byte-quantized, which is lossy for
    // documents long enough to leave the exact-encoding region. This
    // guards the ranking-quality claim: on long docs with well-separated
    // lengths, the quantized top-k must still be the exact BM25 top-k.
    //
    // Each candidate doc contains "qterm" once and is padded with
    // per-doc-unique filler to a length that grows ~1.5× per doc
    // (20, 30, 45, …). BM25 for "qterm" (tf=1, same idf everywhere)
    // ranks purely by length-norm — shortest first — and the 1.5× gap
    // exceeds one quantization bucket, so the exact order is preserved.
    const N_CANDIDATES: usize = 8;
    let mut corpus_owned: Vec<(u64, String)> = Vec::new();
    let mut len = 20usize;
    for d in 0..N_CANDIDATES {
        let mut text = String::from("qterm");
        // Pad with tokens unique to this doc so filler never matches the
        // query and every doc's length is deliberate.
        for f in 0..(len - 1) {
            text.push_str(&format!(" fill{d}x{f}"));
        }
        corpus_owned.push((d as u64, text));
        len = (len * 3) / 2;
    }
    // A few long filler-only docs to make avgdl realistic and ensure the
    // norm table spans multiple quantization buckets.
    for d in N_CANDIDATES..(N_CANDIDATES + 4) {
        let mut text = String::new();
        for f in 0..300 {
            text.push_str(&format!("noise{d}x{f} "));
        }
        corpus_owned.push((d as u64, text));
    }
    let corpus: Vec<(u64, &str)> = corpus_owned.iter().map(|(i, s)| (*i, s.as_str())).collect();

    let infino = build_infino_superfile(&corpus);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corpus, tok.as_ref());

    let infino_hits = infino_top_k(&infino, "qterm", N_CANDIDATES).await;
    let oracle_hits: Vec<u64> = oracle
        .top_k("qterm", N_CANDIDATES, tok.as_ref())
        .into_iter()
        .map(|(d, _)| d)
        .collect();

    // Order preserved: shortest doc (id 0) first, longest last. The 1.5×
    // length spacing keeps every candidate in its own bucket, so the
    // quantized ranking matches the exact ranking element-for-element.
    assert_eq!(
        infino_hits, oracle_hits,
        "quantized norms reordered the top-k vs exact BM25"
    );
}
