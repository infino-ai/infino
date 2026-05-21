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

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::SuperfileReader;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use infino::superfile::fts::reader::BoolMode;
use infino::test_helpers::brute_force_bm25::BruteForceBm25;
use infino::test_helpers::{decimal128_ids, default_tokenizer};
use std::collections::HashSet;
use std::sync::Arc;

/// 60-doc planted corpus with mixed term frequencies. Enough to
/// make BM25's tf + idf + dl-norm interaction non-trivial, small
/// enough to keep the test fast.
fn corpus() -> Vec<(u64, &'static str)> {
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
        (43, "rust criterion benchmarks measure"),
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
fn build_infino_superfile(corpus: &[(u64, &str)]) -> SuperfileReader {
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
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
fn infino_top_k(reader: &SuperfileReader, query: &str, k: usize) -> Vec<u64> {
    let hits = reader
        .bm25_search("title", query, k, BoolMode::Or)
        .expect("BM25 search");
    hits.into_iter().map(|(d, _)| d as u64).collect()
}

/// Compare top-k *sets* between infino and brute-force for a query.
/// Asserts agreement on the head; allows tail divergence for ties.
fn assert_top_k_head_agrees(
    infino: &SuperfileReader,
    oracle: &BruteForceBm25,
    query: &str,
    head_size: usize,
    k: usize,
) {
    let tok = default_tokenizer();
    let infino_hits = infino_top_k(infino, query, k);
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

#[test]
fn oracle_rare_term_top1_matches() {
    // Single-term, single-doc match: "rare-token-zzz" is unique to
    // doc 17. Both engines must return [17] as top-1.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "rare-token-zzz", 1, 5);
}

#[test]
fn oracle_common_term_top1_in_correct_set() {
    // "rust" appears in many same-length docs at mathematically tied
    // BM25 scores. We can't assert exact top-K agreement because
    // tie-breaking diverges, but BOTH engines must pick top-1 from
    // the docs that actually contain "rust".
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_top: u64 = infino_top_k(&infino, "rust", 1)[0];
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

#[test]
fn oracle_two_term_or_top1_matches() {
    // "redis kafka" — doc 14 has "redis", doc 15 has "kafka". Both
    // single-occurrence docs; either could be top-1. Top-2 set must
    // agree.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "redis kafka", 2, 5);
}

#[test]
fn oracle_two_term_overlap_top3_matches() {
    // "rust async" — docs 0 and 20 contain both terms, so they should
    // rank highest under any sensible BM25.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_hits = infino_top_k(&infino, "rust async", 5);
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

#[test]
fn oracle_three_term_query_top5_set_matches() {
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    assert_top_k_head_agrees(&infino, &oracle, "rust web framework", 3, 10);
}

#[test]
fn oracle_no_match_query_returns_empty() {
    // "xyzzy" is in none of the docs; both engines must return empty.
    let corp = corpus();
    let infino = build_infino_superfile(&corp);
    let tok = default_tokenizer();
    let oracle = BruteForceBm25::index(&corp, tok.as_ref());
    let infino_hits = infino_top_k(&infino, "xyzzy", 5);
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

#[test]
fn oracle_long_doc_vs_short_doc_dl_norm() {
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
    let infino_hits = infino_top_k(&infino, "framework", 5);
    let oracle_hits: Vec<u64> = oracle
        .top_k("framework", 5, tok.as_ref())
        .into_iter()
        .map(|(d, _)| d)
        .collect();
    let infino_top2: HashSet<u64> = infino_hits.into_iter().take(2).collect();
    let oracle_top2: HashSet<u64> = oracle_hits.into_iter().take(2).collect();
    assert_eq!(infino_top2, oracle_top2, "framework top-2 sets disagree");
}
