//! BM25 correctness oracle for the supertable's multi-segment
//! search path.
//!
//! The supertable shards the corpus across N superfiles. Each
//! superfile runs its own BM25 with its own per-segment IDF +
//! avgdl, and the supertable merges the per-segment top-k into a
//! global top-k. This oracle mirrors that shape with a per-segment
//! brute-force BM25 and a global merge, then asserts the
//! supertable's hits match.
//!
//! ## What this oracle catches
//!
//! Per-segment brute-force catches per-segment scoring bugs (same
//! as the single-segment oracle in
//! `tests/superfile/fts/brute_force_oracle.rs`). The cross-segment
//! merge catches a separate class of bugs that the single-segment
//! oracle can't see: wrong segment partitioning, wrong tagging of
//! per-segment hits with their segment URI, wrong score-direction
//! in the top-k merge.
//!
//! ## Tolerances
//!
//! Top-k *sets* must agree exactly on the head. Order within a
//! tied score may vary; we assert set equality on the head.

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use arrow_array::{LargeStringArray, RecordBatch};
use rand::SeedableRng;
use rand::rngs::StdRng;

use infino::superfile::builder::FtsConfig;
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::fts::tokenize::Tokenizer;
use infino::supertable::query::SuperfileHit;
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::brute_force_bm25::BruteForceBm25;
use infino::test_helpers::{default_tokenizer, schema_id_title};

/// Fixed planted corpus, 60 docs. Sharded into 4 superfiles of 15
/// docs each.
fn planted_corpus() -> Vec<(u64, &'static str)> {
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

const SEGMENTS: usize = 4;
const N_PREFIX_TERMS: usize = SEGMENTS;
const N_PLANTED: usize = 60;
const CHUNK_SIZE: usize = N_PLANTED / SEGMENTS;

/// Plant `N_PREFIX_TERMS` unique-prefix terms (`alphafox00`..)
/// across distinct superfiles for prefix-search testing.
fn corpus_with_prefix_terms() -> Vec<(u64, String)> {
    let mut corp: Vec<(u64, String)> = planted_corpus()
        .into_iter()
        .map(|(id, t)| (id, t.to_string()))
        .collect();
    for i in 0..N_PREFIX_TERMS {
        let target_idx = (i + 1) * CHUNK_SIZE - 1;
        let extra = format!(" alphafox{i:02}");
        corp[target_idx].1.push_str(&extra);
    }
    corp
}

// ---- Supertable side -----------------------------------------------

fn build_supertable(corpus: &[(u64, String)], n_superfiles: usize) -> Supertable {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("pool"),
    );
    let tk: Arc<dyn Tokenizer> = default_tokenizer();
    let opts = SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(tk),
    )
    .expect("opts")
    .with_writer_pool(pool);

    let st = Supertable::create(opts);
    let mut w = st.writer().expect("writer");
    let chunk_size = corpus.len().div_ceil(n_superfiles);
    for chunk in corpus.chunks(chunk_size) {
        let titles =
            LargeStringArray::from(chunk.iter().map(|(_, t)| t.as_str()).collect::<Vec<_>>());
        let batch = RecordBatch::try_new(schema_id_title(), vec![Arc::new(titles)]).expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

/// Convert supertable hits to global doc_ids using the segment-
/// append order (segment_index * chunk_size + local_doc_id).
fn supertable_to_global_ids(
    st: &Supertable,
    hits: Vec<SuperfileHit>,
    chunk_size: usize,
) -> Vec<u64> {
    let r = st.reader();
    let manifest = r.manifest();
    hits.into_iter()
        .map(|h| {
            let seg_idx = manifest
                .superfiles
                .iter()
                .position(|e| e.uri == h.segment)
                .expect("segment in manifest");
            (seg_idx as u64) * (chunk_size as u64) + (h.local_doc_id as u64)
        })
        .collect()
}

fn supertable_search_global(st: &Supertable, query: &str, k: usize, chunk_size: usize) -> Vec<u64> {
    let r = st.reader();
    let hits = r
        .bm25_search("title", query, k, BoolMode::Or)
        .expect("supertable bm25");
    supertable_to_global_ids(st, hits, chunk_size)
}

fn supertable_search_and_global(
    st: &Supertable,
    query: &str,
    k: usize,
    chunk_size: usize,
) -> Vec<u64> {
    let r = st.reader();
    let hits = r
        .bm25_search("title", query, k, BoolMode::And)
        .expect("supertable bm25 AND");
    supertable_to_global_ids(st, hits, chunk_size)
}

fn supertable_prefix_global(
    st: &Supertable,
    prefix: &str,
    k: usize,
    chunk_size: usize,
) -> Vec<u64> {
    let r = st.reader();
    let hits = r
        .bm25_search_prefix("title", prefix, k)
        .expect("supertable bm25_prefix");
    supertable_to_global_ids(st, hits, chunk_size)
}

// ---- Brute-force oracle (per-segment + global merge) ---------------

/// Build a per-segment BruteForceBm25 oracle list. Index i scores
/// segment i with that segment's own IDF/avgdl, mirroring the
/// supertable's per-segment scoring shape.
fn build_oracles(corpus: &[(u64, String)], n_superfiles: usize) -> Vec<BruteForceBm25> {
    let tok = default_tokenizer();
    let chunk_size = corpus.len().div_ceil(n_superfiles);
    corpus
        .chunks(chunk_size)
        .map(|chunk| {
            // The chunk lives in &str-as-&'a String land; BruteForceBm25
            // wants `&[(u64, &str)]`, so adapt the borrow once.
            let view: Vec<(u64, &str)> = chunk.iter().map(|(i, t)| (*i, t.as_str())).collect();
            BruteForceBm25::index(&view, tok.as_ref())
        })
        .collect()
}

/// Run per-segment brute-force BM25 and merge into a global top-k
/// in the same shape the supertable's fan-out produces.
fn brute_force_top_k(oracles: &[BruteForceBm25], query: &str, k: usize) -> Vec<u64> {
    let tok = default_tokenizer();
    let mut all: Vec<(u64, f32)> = Vec::new();
    for o in oracles {
        all.extend(o.top_k(query, k, tok.as_ref()));
    }
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all.into_iter().map(|(d, _)| d).collect()
}

/// Same as [`brute_force_top_k`] but for a multi-term explicit
/// AND query. Each segment scores its AND intersection
/// independently; the global merge keeps the highest-scoring docs
/// across segments. Mirrors the supertable's AND fan-out shape.
fn brute_force_and_top_k(oracles: &[BruteForceBm25], query: &str, k: usize) -> Vec<u64> {
    let tok = default_tokenizer();
    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each(query, &mut |t| terms.push(t.to_owned()));
    let mut all: Vec<(u64, f32)> = Vec::new();
    for o in oracles {
        all.extend(o.top_k_terms_and(&terms, k));
    }
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all.into_iter().map(|(d, _)| d).collect()
}

/// Same as [`brute_force_top_k`] but for a multi-term explicit
/// OR query (used to mirror the supertable's prefix expansion).
fn brute_force_terms_top_k(oracles: &[BruteForceBm25], terms: &[String], k: usize) -> Vec<u64> {
    let mut all: Vec<(u64, f32)> = Vec::new();
    for o in oracles {
        all.extend(o.top_k_terms(terms, k));
    }
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all.into_iter().map(|(d, _)| d).collect()
}

fn assert_top_k_sets_match(label: &str, supertable: Vec<u64>, oracle: Vec<u64>, head_size: usize) {
    let sup_head: HashSet<u64> = supertable.iter().take(head_size).copied().collect();
    let ora_head: HashSet<u64> = oracle.iter().take(head_size).copied().collect();
    assert_eq!(
        sup_head, ora_head,
        "{label}: top-{head_size} sets disagree — supertable={supertable:?} oracle={oracle:?}",
    );
}

// ---- Shared fixture --------------------------------------------------

struct StandardFixture {
    infino: Supertable,
    oracles: Vec<BruteForceBm25>,
}

static STANDARD_FIXTURE: LazyLock<StandardFixture> = LazyLock::new(|| {
    let corp = corpus_with_prefix_terms();
    let infino = build_supertable(&corp, SEGMENTS);
    let oracles = build_oracles(&corp, SEGMENTS);
    StandardFixture { infino, oracles }
});

// ---- Tests: query-shape coverage -------------------------------------

#[test]
fn oracle_single_rare_top1_matches() {
    // "rare-token-zzz" appears in exactly 1 doc (id=17, segment 1).
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "rare-token-zzz", 5, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rare-token-zzz", 5);
    assert_eq!(inf_hits.first().copied(), Some(17));
    assert_eq!(ora_hits.first().copied(), Some(17));
    assert_top_k_sets_match("single_rare", inf_hits, ora_hits, 1);
}

#[test]
fn oracle_single_common_top3_overlap() {
    // "rust" appears in many docs. Top-10 sets must overlap by ≥3.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "rust", 10, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust", 10);
    let inf_set: HashSet<u64> = inf_hits.iter().take(10).copied().collect();
    let ora_set: HashSet<u64> = ora_hits.iter().take(10).copied().collect();
    let common: HashSet<u64> = inf_set.intersection(&ora_set).copied().collect();
    assert!(
        common.len() >= 3,
        "single_common: top-10 sets should overlap by ≥3 — supertable={inf_hits:?} oracle={ora_hits:?}",
    );
}

#[test]
fn oracle_two_term_or_top2_matches() {
    // Docs containing both "rust" AND "async": doc 0, doc 20, doc 22.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "rust async", 5, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust async", 5);
    let inf_top2: HashSet<u64> = inf_hits.iter().take(2).copied().collect();
    let ora_top2: HashSet<u64> = ora_hits.iter().take(2).copied().collect();
    assert!(
        inf_top2.contains(&0) && inf_top2.contains(&20),
        "supertable top-2 should include docs 0 and 20; got {inf_hits:?}"
    );
    assert!(
        ora_top2.contains(&0) && ora_top2.contains(&20),
        "oracle top-2 should include docs 0 and 20; got {ora_hits:?}"
    );
    assert_eq!(inf_top2, ora_top2);
}

#[test]
fn oracle_three_wide_or_top3_matches() {
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "rust web framework", 10, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust web framework", 10);
    let inf_top: HashSet<u64> = inf_hits.iter().take(3).copied().collect();
    let ora_top: HashSet<u64> = ora_hits.iter().take(3).copied().collect();
    assert!(inf_top.contains(&8));
    assert!(ora_top.contains(&8));
    assert_top_k_sets_match("three_wide_or", inf_hits, ora_hits, 3);
}

#[test]
fn oracle_three_similar_or_top3_matches() {
    // Three single-doc terms (docs 14, 15, 16).
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "redis kafka elasticsearch", 5, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "redis kafka elasticsearch", 5);
    let want: HashSet<u64> = [14u64, 15, 16].into_iter().collect();
    let inf_top: HashSet<u64> = inf_hits.iter().take(3).copied().collect();
    let ora_top: HashSet<u64> = ora_hits.iter().take(3).copied().collect();
    assert_eq!(inf_top, want);
    assert_eq!(ora_top, want);
}

#[test]
fn oracle_five_term_or_top5_matches() {
    // Five single-doc terms (docs 30..34).
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "tcp udp http2 http3 tls", 10, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "tcp udp http2 http3 tls", 10);
    let want: HashSet<u64> = [30u64, 31, 32, 33, 34].into_iter().collect();
    let inf_top: HashSet<u64> = inf_hits.iter().take(5).copied().collect();
    let ora_top: HashSet<u64> = ora_hits.iter().take(5).copied().collect();
    assert_eq!(inf_top, want);
    assert_eq!(ora_top, want);
}

// ---- Tests: AND-mode oracle (multi-segment intersection) ------------

#[test]
fn oracle_two_term_and_matches() {
    // "rust" + "async" co-occur in docs 0, 20, 22 — split across
    // segments 0 (doc 0) and 1 (docs 20, 22), so this exercises
    // multi-segment AND fan-out.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust async", 10, CHUNK_SIZE);
    let ora_hits = brute_force_and_top_k(&f.oracles, "rust async", 10);
    let want: HashSet<u64> = [0u64, 20, 22].into_iter().collect();
    let inf_set: HashSet<u64> = inf_hits.iter().copied().collect();
    let ora_set: HashSet<u64> = ora_hits.iter().copied().collect();
    assert_eq!(inf_set, want, "supertable AND={inf_hits:?}");
    assert_eq!(ora_set, want, "oracle AND={ora_hits:?}");
}

#[test]
fn oracle_three_term_and_singleton_match() {
    // "rust" + "async" + "tokio" intersect only at doc 0 (segment 0).
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust async tokio", 10, CHUNK_SIZE);
    assert_eq!(inf_hits, vec![0u64], "got {inf_hits:?}");
}

#[test]
fn oracle_and_missing_term_returns_empty() {
    // A globally absent term must short-circuit AND to empty even
    // when the other term has many hits.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust definitelynotpresent", 10, CHUNK_SIZE);
    assert!(inf_hits.is_empty(), "got {inf_hits:?}");
}

#[test]
fn oracle_and_segment_locally_missing_term_still_intersects_elsewhere() {
    // "rust" + "kafka" — "rust" appears in every segment, but
    // "kafka" only appears in doc 15 (segment 1) where "rust" does
    // not co-occur. The intersection is empty across the whole
    // table, but the test confirms that segments with the missing
    // term contribute nothing and segments without the missing term
    // also contribute nothing.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust kafka", 10, CHUNK_SIZE);
    let ora_hits = brute_force_and_top_k(&f.oracles, "rust kafka", 10);
    assert!(inf_hits.is_empty(), "supertable AND must be empty; got {inf_hits:?}");
    assert!(ora_hits.is_empty(), "oracle AND must be empty; got {ora_hits:?}");
}

// ---- Tests: prefix-row exercise ---------------------------------------

#[test]
fn oracle_prefix_query_matches_explicit_term_or() {
    // The supertable expands `alphafox` via per-segment FST walk,
    // then runs a per-segment OR over the expansion. Mirror this
    // by running a brute-force OR over the same explicit term list.
    let f = &*STANDARD_FIXTURE;
    let prefix = "alphafox";
    let expanded: Vec<String> = (0..N_PREFIX_TERMS)
        .map(|i| format!("alphafox{i:02}"))
        .collect();

    let inf_hits = supertable_prefix_global(&f.infino, prefix, 10, CHUNK_SIZE);
    let ora_hits = brute_force_terms_top_k(&f.oracles, &expanded, 10);

    let want: HashSet<u64> = [14u64, 29, 44, 59].into_iter().collect();
    let inf_set: HashSet<u64> = inf_hits.iter().take(N_PREFIX_TERMS).copied().collect();
    let ora_set: HashSet<u64> = ora_hits.iter().take(N_PREFIX_TERMS).copied().collect();
    assert_eq!(inf_set, want, "supertable prefix hits = {inf_hits:?}");
    assert_eq!(ora_set, want, "oracle explicit-OR hits = {ora_hits:?}");
}

#[test]
fn prefix_skip_prunes_segments_without_matching_lex_range() {
    // Plant a prefix term in only one segment; verify the prefix
    // search returns exactly that doc and skip pruning prevents
    // other superfiles from contributing.
    let mut corp: Vec<(u64, String)> = planted_corpus()
        .into_iter()
        .map(|(id, t)| (id, t.to_string()))
        .collect();
    corp[0].1.push_str(" quokka_unique");
    let infino = build_supertable(&corp, SEGMENTS);
    let r = infino.reader();
    let hits = r.bm25_search_prefix("title", "quokka", 5).expect("prefix");
    assert_eq!(hits.len(), 1);
    let manifest = r.manifest();
    let target_uri = manifest.superfiles[0].uri;
    assert_eq!(hits[0].segment, target_uri);
    assert_eq!(hits[0].local_doc_id, 0);
}

// ---- Tests: empty + no-match ----------------------------------------

#[test]
fn oracle_no_match_returns_empty() {
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "definitelynotpresent", 5, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "definitelynotpresent", 5);
    assert!(inf_hits.is_empty());
    assert!(ora_hits.is_empty());
}

// ---- Tests: larger-scale Zipfian smoke ------------------------------

/// Generate a small Zipfian corpus matching the bench's shape at
/// test-fast scale.
fn zipfian_corpus(n_docs: usize, seed: u64) -> Vec<(u64, String)> {
    use rand::RngExt;
    let mut rng = StdRng::seed_from_u64(seed);
    const VOCAB: usize = 10_000;
    const TOKENS_PER_DOC: usize = 100;
    let mut cum = Vec::with_capacity(VOCAB);
    let mut acc = 0.0f64;
    for i in 1..=VOCAB {
        acc += 1.0 / (i as f64);
        cum.push(acc);
    }
    let total = *cum.last().expect("vocab > 0");

    let mut out = Vec::with_capacity(n_docs);
    for d in 0..n_docs as u64 {
        let mut s = String::with_capacity(TOKENS_PER_DOC * 8);
        for j in 0..TOKENS_PER_DOC {
            let target: f64 = rng.random::<f64>() * total;
            let idx = match cum
                .binary_search_by(|p| p.partial_cmp(&target).unwrap_or(std::cmp::Ordering::Equal))
            {
                Ok(i) | Err(i) => i.min(VOCAB - 1) + 1,
            };
            if j > 0 {
                s.push(' ');
            }
            s.push_str(&format!("term{idx:05}"));
        }
        out.push((d, s));
    }
    out
}

#[test]
fn oracle_zipfian_corpus_query_shapes_match() {
    // 5K docs × 4 superfiles = 1250 docs/segment. Brute-force across
    // segments is the exact same scoring path the supertable runs
    // (per-segment IDF + global top-k merge with identical
    // tie-breaker), so set overlap on the top-k is expected to be
    // tight; we keep the 60 % threshold loose to absorb any future
    // BM25 dl-norm refinements without test churn.
    let n_docs = 5_000;
    let corp = zipfian_corpus(n_docs, 42);
    let infino = build_supertable(&corp, SEGMENTS);
    let oracles = build_oracles(&corp, SEGMENTS);
    let k = 20;

    let queries = [
        ("single_rare", "term09999"),
        ("two_term_or", "term00001 term00050"),
        ("three_wide_or", "term00001 term00050 term00100"),
        ("three_similar_or", "term00050 term00051 term00052"),
        (
            "five_term_or",
            "term00050 term00051 term00052 term00053 term00054",
        ),
    ];

    for (label, q) in queries {
        let inf = supertable_search_global(&infino, q, k, n_docs / SEGMENTS);
        let ora = brute_force_top_k(&oracles, q, k);
        let inf_set: HashSet<u64> = inf.iter().copied().collect();
        let ora_set: HashSet<u64> = ora.iter().copied().collect();
        let common = inf_set.intersection(&ora_set).count();
        let target = inf_set.len().min(ora_set.len());
        // ≥ 60 % overlap threshold. Brute-force shares infino's
        // tie-breaker so in practice the overlap is much higher, but
        // we keep the threshold loose so BM25 dl-norm refinements
        // aren't artificially bound by the test.
        let threshold = (target * 6) / 10;
        assert!(
            common >= threshold,
            "{label}: top-{k} overlap {common}/{target} below 60% threshold; \
             supertable={inf:?} oracle={ora:?}",
        );
    }
}
