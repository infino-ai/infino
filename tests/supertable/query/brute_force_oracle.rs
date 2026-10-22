// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! BM25 correctness oracle for the supertable's multi-superfile
//! search path.
//!
//! The supertable shards the corpus across N superfiles. Each
//! superfile scores with table-wide IDF and the average document length
//! it declares (the table-wide average as of its commit), and the
//! supertable merges the per-superfile top-k into a global top-k. This
//! oracle mirrors that shape: one brute-force BM25 per superfile, each
//! indexed over the whole corpus and re-pointed at that file's declared
//! average, restricted to the file's rows and merged globally.
//!
//! ## What this oracle catches
//!
//! Scoring bugs (same as the single-superfile oracle in
//! `tests/superfile/fts/brute_force_oracle.rs`), wrong table-wide
//! statistics, and the cross-superfile bugs the single-superfile oracle
//! can't see: wrong superfile partitioning, wrong tagging of
//! per-superfile hits with their superfile URI, wrong score-direction
//! in the top-k merge.
//!
//! ## Tolerances
//!
//! Top-k *sets* must agree exactly on the head. Order within a
//! tied score may vary; we assert set equality on the head.

#![deny(clippy::unwrap_used)]

use std::{
    cmp::{Ordering, Reverse},
    collections::{HashMap, HashSet},
    iter::repeat_n,
    ops::Range,
    slice::from_ref,
    sync::{Arc, LazyLock},
};

use arrow_array::{LargeStringArray, RecordBatch};
use infino::{
    Bm25SearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::FtsConfig,
        fts::{
            bm25::stored_avgdl,
            reader::{BoolMode, ColumnLengthStats},
            tokenize::Phrase,
        },
    },
    supertable::{
        Supertable, SupertableOptions,
        manifest::{SuperfileEntry, SuperfileUri},
        query::SuperfileHit,
    },
    test_helpers::{
        brute_force_bm25::{BruteForceBm25, OracleBm25Params},
        build_title_batch, default_tokenizer, schema_id_title,
    },
};
use rand::{SeedableRng, rngs::StdRng};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

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

const SUPERFILES: usize = 4;
const N_PREFIX_TERMS: usize = SUPERFILES;
const N_PLANTED: usize = 60;
const CHUNK_SIZE: usize = N_PLANTED / SUPERFILES;
/// Standard oracle top-k (with headroom) for the comparison queries.
const ORACLE_TOP_K: usize = 10;
/// Smaller oracle top-k for the single-/few-match queries.
const ORACLE_TOP_K_SMALL: usize = 5;
/// Single-thread rayon pool for deterministic oracle comparisons.
const RAYON_POOL_THREADS: usize = 1;
/// Top-k for the larger zipfian-corpus agreement test.
const ZIPFIAN_TOP_K: usize = 20;

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

/// The fixture at the standard BM25 pair — the shape most of these
/// tests want, so they do not have to name a pair they do not care
/// about.
fn build_supertable(corpus: &[(u64, String)], n_superfiles: usize) -> Supertable {
    build_supertable_with_params(corpus, n_superfiles, OracleBm25Params::default())
}

/// The fixture, with every FTS column declaring `params` — so the build
/// bakes its block-max bounds at that pair and the reader scores with
/// it.
fn build_supertable_with_params(
    corpus: &[(u64, String)],
    n_superfiles: usize,
    params: OracleBm25Params,
) -> Supertable {
    build_supertable_with(corpus, n_superfiles, params, |o| o)
}

/// The fixture with `params`, its options passed through `configure`
/// before the table is created.
fn build_supertable_with(
    corpus: &[(u64, String)],
    n_superfiles: usize,
    params: OracleBm25Params,
    configure: impl FnOnce(SupertableOptions) -> SupertableOptions,
) -> Supertable {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let opts = SupertableOptions::new(
        schema_id_title(),
        // Positions are indexed so the phrase shape below is answerable.
        // They change what the column can be asked, not how a term
        // scores, so every non-phrase case grades the same as before.
        vec![
            FtsConfig::new("title")
                .bm25(params.k1, params.b)
                .positions(true),
        ],
        vec![],
    )
    .expect("opts")
    .with_writer_pool(pool);
    let opts = configure(opts);

    let st = Supertable::create(opts).expect("create");
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

/// Convert supertable hits to global doc_ids using the superfile-
/// append order (superfile_index * chunk_size + local_doc_id).
fn supertable_to_global_ids(
    st: &Supertable,
    hits: Vec<SuperfileHit>,
    chunk_size: usize,
) -> Vec<u64> {
    let r = st.reader().expect("reader");
    let manifest = r.manifest();
    hits.into_iter()
        .map(|h| {
            let seg_idx = manifest
                .superfiles
                .iter()
                .position(|e| e.uri == h.superfile)
                .expect("superfile in manifest");
            (seg_idx as u64) * (chunk_size as u64) + (h.local_doc_id as u64)
        })
        .collect()
}

fn supertable_search_global(st: &Supertable, query: &str, k: usize, chunk_size: usize) -> Vec<u64> {
    let hits = st
        .reader()
        .expect("reader")
        .bm25_hits("title", query, k, Bm25SearchOptions::new())
        .expect("supertable bm25");
    supertable_to_global_ids(st, hits, chunk_size)
}

fn supertable_search_and_global(
    st: &Supertable,
    query: &str,
    k: usize,
    chunk_size: usize,
) -> Vec<u64> {
    let hits = st
        .reader()
        .expect("reader")
        .bm25_hits(
            "title",
            query,
            k,
            Bm25SearchOptions::new().with_mode(BoolMode::And),
        )
        .expect("supertable bm25 AND");
    supertable_to_global_ids(st, hits, chunk_size)
}

fn supertable_prefix_global(
    st: &Supertable,
    prefix: &str,
    k: usize,
    chunk_size: usize,
) -> Vec<u64> {
    let hits = st
        .reader()
        .expect("reader")
        .bm25_search_prefix("title", prefix, k)
        .expect("supertable bm25_prefix");
    supertable_to_global_ids(st, hits, chunk_size)
}

// ---- Brute-force oracle (per-superfile + global merge) ---------------

/// One superfile's reference scorer: indexed over the whole corpus (so
/// idf is table-wide), normalizing lengths at the table's average
/// whatever average the file was built at, and answering only for the rows
/// the file holds.
struct FileOracle {
    scorer: BruteForceBm25,
    rows: Range<u64>,
}

impl FileOracle {
    /// This file's rows out of a whole-corpus ranking.
    fn own(&self, hits: Vec<(u64, f32)>) -> impl Iterator<Item = (u64, f32)> + '_ {
        hits.into_iter().filter(|(d, _)| self.rows.contains(d))
    }
}

/// File oracles that score with `params` rather than the standard
/// pair — the reference side of a declared-parameter fixture.
fn build_oracles_with_params(
    corpus: &[(u64, String)],
    n_superfiles: usize,
    params: OracleBm25Params,
) -> Vec<FileOracle> {
    build_oracles(corpus, n_superfiles)
        .into_iter()
        .map(|o| FileOracle {
            scorer: o.scorer.with_params(params),
            rows: o.rows,
        })
        .collect()
}

/// One [`FileOracle`] per superfile [`build_supertable`] writes, each
/// scoring at the whole corpus's average, rounded as a file stores it.
fn build_oracles(corpus: &[(u64, String)], n_superfiles: usize) -> Vec<FileOracle> {
    let tok = default_tokenizer();
    let view: Vec<(u64, &str)> = corpus.iter().map(|(i, t)| (*i, t.as_str())).collect();
    let table_avgdl = corpus_avgdl(corpus);
    let chunk_size = corpus.len().div_ceil(n_superfiles);
    corpus
        .chunks(chunk_size)
        .map(|chunk| {
            let first = chunk.first().expect("non-empty chunk").0;
            let last = chunk.last().expect("non-empty chunk").0;
            FileOracle {
                scorer: BruteForceBm25::index(&view, tok.as_ref()).with_avgdl(table_avgdl),
                rows: first..last + 1,
            }
        })
        .collect()
}

/// The average document length over all of `corpus`, rounded as a file
/// stores it: what the engine scores every superfile at.
fn corpus_avgdl(corpus: &[(u64, String)]) -> f32 {
    let tok = default_tokenizer();
    let stats = ColumnLengthStats::from_lengths(corpus.iter().map(|(_, t)| {
        let mut n = 0u32;
        tok.tokenize_each(t, &mut |_| n += 1);
        n
    }));
    stored_avgdl(stats.avgdl())
}

/// Merge per-superfile brute-force hits into a global top-k, in the
/// same shape the supertable's fan-out produces: score descending,
/// ties broken by ascending doc id.
///
/// Every oracle wrapper below differs only in which per-superfile
/// scorer it calls, so the merge lives here once — four copies of a
/// tie-break rule is four places for it to drift.
fn merge_global_top_k(mut all: Vec<(u64, f32)>, k: usize) -> Vec<u64> {
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all.into_iter().map(|(d, _)| d).collect()
}

/// Run per-superfile brute-force BM25 and merge into a global top-k.
fn brute_force_top_k(oracles: &[FileOracle], query: &str, k: usize) -> Vec<u64> {
    let tok = default_tokenizer();
    let all = oracles
        .iter()
        .flat_map(|o| o.own(o.scorer.top_k(query, usize::MAX, tok.as_ref())))
        .collect();
    merge_global_top_k(all, k)
}

/// Same as [`brute_force_top_k`] but for a multi-term explicit
/// AND query. Each superfile scores its AND intersection
/// independently; the global merge keeps the highest-scoring docs
/// across superfiles. Mirrors the supertable's AND fan-out shape.
fn brute_force_and_top_k(oracles: &[FileOracle], query: &str, k: usize) -> Vec<u64> {
    let tok = default_tokenizer();
    let mut terms: Vec<String> = Vec::new();
    tok.tokenize_each(query, &mut |t| terms.push(t.to_owned()));
    let all = oracles
        .iter()
        .flat_map(|o| o.own(o.scorer.top_k_terms_and(&terms, usize::MAX)))
        .collect();
    merge_global_top_k(all, k)
}

/// Same as [`brute_force_top_k`] but for a multi-term explicit
/// OR query (used to mirror the supertable's prefix expansion).
fn brute_force_terms_top_k(oracles: &[FileOracle], terms: &[String], k: usize) -> Vec<u64> {
    let all = oracles
        .iter()
        .flat_map(|o| o.own(o.scorer.top_k_terms(terms, usize::MAX)))
        .collect();
    merge_global_top_k(all, k)
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
    oracles: Vec<FileOracle>,
}

static STANDARD_FIXTURE: LazyLock<StandardFixture> = LazyLock::new(|| {
    let corp = corpus_with_prefix_terms();
    let infino = build_supertable(&corp, SUPERFILES);
    let oracles = build_oracles(&corp, SUPERFILES);
    StandardFixture { infino, oracles }
});

/// The same corpus with a declared, non-standard BM25 pair on both
/// sides. The engine bakes its block-max bounds at this pair and scores
/// with it; the oracle uses the identical formula, so a divergence is a
/// real disagreement rather than two different similarity functions.
const DECLARED_PARAMS: OracleBm25Params = OracleBm25Params { k1: 1.6, b: 0.4 };

static DECLARED_PARAM_FIXTURE: LazyLock<StandardFixture> = LazyLock::new(|| {
    let corp = corpus_with_prefix_terms();
    let infino = build_supertable_with_params(&corp, SUPERFILES, DECLARED_PARAMS);
    let oracles = build_oracles_with_params(&corp, SUPERFILES, DECLARED_PARAMS);
    StandardFixture { infino, oracles }
});

/// A query shape, carrying both the string the engine parses and the
/// decomposition the oracle scores.
///
/// The two are derived from one value on purpose. Feeding the oracle
/// from the engine's own parser — the convention elsewhere in the
/// suite — makes a parsing change invisible here, because both sides
/// would move together. These cases exist to grade *scoring* at a
/// non-standard pair, so the clause structure is stated outright and a
/// parser that stopped honouring a sigil would fail rather than agree
/// with itself.
enum QueryShape {
    /// Bare terms, OR-combined — the default operator.
    Or(&'static str),
    /// Every term required (`+a +b`): the ranked-AND membership walk.
    AllOf(&'static [&'static str]),
    /// One exact phrase (`"a b"`): position-verified, and a different
    /// kernel again — the phrase cursor composes its members' idfs and
    /// its own term-level bound, so it reads a stored bound the single
    /// and union paths never touch.
    Phrase(&'static [&'static str]),
}

impl QueryShape {
    /// The query string handed to the engine.
    fn engine_query(&self) -> String {
        match self {
            Self::Or(q) => (*q).to_owned(),
            Self::AllOf(terms) => terms
                .iter()
                .map(|t| format!("+{t}"))
                .collect::<Vec<_>>()
                .join(" "),
            Self::Phrase(terms) => format!("\"{}\"", terms.join(" ")),
        }
    }

    /// The reference top-k for the same shape.
    fn oracle_top_k(&self, oracles: &[FileOracle], k: usize) -> Vec<u64> {
        match self {
            Self::Or(q) => brute_force_top_k(oracles, q, k),
            Self::AllOf(terms) => {
                let musts: Vec<String> = terms.iter().map(|t| (*t).to_owned()).collect();
                let all = oracles
                    .iter()
                    .flat_map(|o| {
                        o.own(
                            o.scorer
                                .top_k_atoms(&musts, &[], &[], &[], &[], &[], usize::MAX),
                        )
                    })
                    .collect();
                merge_global_top_k(all, k)
            }
            Self::Phrase(terms) => {
                // A bare quoted run is a *should* phrase, matching how
                // the engine's clause split reads it with no sigil.
                // Adjacent, which is every phrase on a column with no
                // analysis chain.
                let phrase =
                    Phrase::adjacent(terms.iter().map(|t| (*t).to_owned()).collect::<Vec<_>>());
                let all = oracles
                    .iter()
                    .flat_map(|o| {
                        o.own(o.scorer.top_k_atoms(
                            &[],
                            &[],
                            &[],
                            from_ref(&phrase),
                            &[],
                            &[],
                            usize::MAX,
                        ))
                    })
                    .collect();
                merge_global_top_k(all, k)
            }
        }
    }
}

/// The shapes both parameter tests grade, so the declared-pair arm and
/// the query-time-override arm cover exactly the same ground. Each
/// entry is `(label, shape, k, head)`, where `head` is how many leading
/// positions must match the reference exactly.
///
/// `+rust +async` and `"rust async"` are deliberately the same two
/// terms: the phrase matches only where they are adjacent, while the
/// must-query also takes `rust trait dyn impl async`. A phrase kernel
/// that quietly degraded to an intersection would return that extra
/// document and fail here.
const QUERY_SHAPES: [(&str, QueryShape, usize, usize); 5] = [
    (
        "single_rare",
        QueryShape::Or("rare-token-zzz"),
        ORACLE_TOP_K_SMALL,
        1,
    ),
    ("single_common", QueryShape::Or("rust"), ORACLE_TOP_K, 3),
    ("two_term_or", QueryShape::Or("rust async"), ORACLE_TOP_K, 2),
    // Both match sets are small and exact, so the head spans the whole
    // result rather than a prefix of it. A prefix would compare only
    // the top two, which both shapes agree on — the document that
    // separates them (`rust trait dyn impl async`, matching the
    // intersection but not the phrase) ranks third, and a head of 2
    // would never look at it.
    (
        "two_term_and",
        QueryShape::AllOf(&["rust", "async"]),
        ORACLE_TOP_K,
        ORACLE_TOP_K,
    ),
    (
        "phrase",
        QueryShape::Phrase(&["rust", "async"]),
        ORACLE_TOP_K,
        ORACLE_TOP_K,
    ),
];

// ---- Tests: declared BM25 parameters ---------------------------------

/// A column that declares a non-standard pair must rank exactly as the
/// reference implementation does at the same pair — across the query
/// shapes that exercise different kernels (single rare, single common,
/// multi-term union).
#[test]
fn oracle_declared_bm25_params_match_across_query_shapes() {
    let f = &*DECLARED_PARAM_FIXTURE;
    for (label, shape, k, head) in QUERY_SHAPES {
        let inf_hits = supertable_search_global(&f.infino, &shape.engine_query(), k, CHUNK_SIZE);
        let ora_hits = shape.oracle_top_k(&f.oracles, k);
        assert_top_k_sets_match(&format!("declared_{label}"), inf_hits, ora_hits, head);
    }
}

/// A query-time override on a *standard* table must rank as the
/// reference does at the override's pair. This is the pruning-correction
/// path: the stored bounds belong to 1.2 / 0.75, so every bound is
/// inflated by the correction factor, and a factor that came out too
/// small would drop a qualifying doc and show up here as a set
/// mismatch.
#[test]
fn oracle_query_time_override_matches_the_reference_at_that_pair() {
    let f = &*STANDARD_FIXTURE;
    let corp = corpus_with_prefix_terms();
    for params in [
        OracleBm25Params { k1: 1.6, b: 0.4 },
        OracleBm25Params { k1: 0.6, b: 0.9 },
        OracleBm25Params { k1: 1.2, b: 0.0 },
    ] {
        let oracles = build_oracles_with_params(&corp, SUPERFILES, params);
        for (label, shape, k, head) in QUERY_SHAPES {
            let query = shape.engine_query();
            let hits = f
                .infino
                .reader()
                .expect("reader")
                .bm25_hits(
                    "title",
                    &query,
                    k,
                    Bm25SearchOptions::new().with_bm25(params.k1, params.b),
                )
                .expect("bm25 with override");
            let inf_hits = supertable_to_global_ids(&f.infino, hits, CHUNK_SIZE);
            let ora_hits = shape.oracle_top_k(&oracles, k);
            assert_top_k_sets_match(
                &format!("override_k1={}_b={}_{label}", params.k1, params.b),
                inf_hits,
                ora_hits,
                head,
            );
        }
    }
}

// ---- Tests: query-shape coverage -------------------------------------

#[test]
fn oracle_single_rare_top1_matches() {
    // "rare-token-zzz" appears in exactly 1 doc (id=17, superfile 1).
    let f = &*STANDARD_FIXTURE;
    let inf_hits =
        supertable_search_global(&f.infino, "rare-token-zzz", ORACLE_TOP_K_SMALL, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rare-token-zzz", ORACLE_TOP_K_SMALL);
    assert_eq!(inf_hits.first().copied(), Some(17));
    assert_eq!(ora_hits.first().copied(), Some(17));
    assert_top_k_sets_match("single_rare", inf_hits, ora_hits, 1);
}

#[test]
fn oracle_single_common_top3_overlap() {
    // "rust" appears in many docs. Top-10 sets must overlap by ≥3.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(&f.infino, "rust", ORACLE_TOP_K, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust", ORACLE_TOP_K);
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
    let inf_hits =
        supertable_search_global(&f.infino, "rust async", ORACLE_TOP_K_SMALL, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust async", ORACLE_TOP_K_SMALL);
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
    let inf_hits =
        supertable_search_global(&f.infino, "rust web framework", ORACLE_TOP_K, CHUNK_SIZE);
    let ora_hits = brute_force_top_k(&f.oracles, "rust web framework", ORACLE_TOP_K);
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
    let inf_hits = supertable_search_global(
        &f.infino,
        "redis kafka elasticsearch",
        ORACLE_TOP_K_SMALL,
        CHUNK_SIZE,
    );
    let ora_hits = brute_force_top_k(&f.oracles, "redis kafka elasticsearch", ORACLE_TOP_K_SMALL);
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
    let inf_hits = supertable_search_global(
        &f.infino,
        "tcp udp http2 http3 tls",
        ORACLE_TOP_K,
        CHUNK_SIZE,
    );
    let ora_hits = brute_force_top_k(&f.oracles, "tcp udp http2 http3 tls", ORACLE_TOP_K);
    let want: HashSet<u64> = [30u64, 31, 32, 33, 34].into_iter().collect();
    let inf_top: HashSet<u64> = inf_hits.iter().take(5).copied().collect();
    let ora_top: HashSet<u64> = ora_hits.iter().take(5).copied().collect();
    assert_eq!(inf_top, want);
    assert_eq!(ora_top, want);
}

// ---- Tests: AND-mode oracle (multi-superfile intersection) ------------

#[test]
fn oracle_two_term_and_matches() {
    // "rust" + "async" co-occur in docs 0, 20, 22 — split across
    // superfiles 0 (doc 0) and 1 (docs 20, 22), so this exercises
    // multi-superfile AND fan-out.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust async", ORACLE_TOP_K, CHUNK_SIZE);
    let ora_hits = brute_force_and_top_k(&f.oracles, "rust async", ORACLE_TOP_K);
    let want: HashSet<u64> = [0u64, 20, 22].into_iter().collect();
    let inf_set: HashSet<u64> = inf_hits.iter().copied().collect();
    let ora_set: HashSet<u64> = ora_hits.iter().copied().collect();
    assert_eq!(inf_set, want, "supertable AND={inf_hits:?}");
    assert_eq!(ora_set, want, "oracle AND={ora_hits:?}");
}

#[test]
fn oracle_three_term_and_singleton_match() {
    // "rust" + "async" + "tokio" intersect only at doc 0 (superfile 0).
    let f = &*STANDARD_FIXTURE;
    let inf_hits =
        supertable_search_and_global(&f.infino, "rust async tokio", ORACLE_TOP_K, CHUNK_SIZE);
    assert_eq!(inf_hits, vec![0u64], "got {inf_hits:?}");
}

#[test]
fn oracle_and_missing_term_returns_empty() {
    // A globally absent term must short-circuit AND to empty even
    // when the other term has many hits.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(
        &f.infino,
        "rust definitelynotpresent",
        ORACLE_TOP_K,
        CHUNK_SIZE,
    );
    assert!(inf_hits.is_empty(), "got {inf_hits:?}");
}

#[test]
fn oracle_and_superfile_locally_missing_term_still_intersects_elsewhere() {
    // "rust" + "kafka" — "rust" appears in every superfile, but
    // "kafka" only appears in doc 15 (superfile 1) where "rust" does
    // not co-occur. The intersection is empty across the whole
    // table, but the test confirms that superfiles with the missing
    // term contribute nothing and superfiles without the missing term
    // also contribute nothing.
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_and_global(&f.infino, "rust kafka", ORACLE_TOP_K, CHUNK_SIZE);
    let ora_hits = brute_force_and_top_k(&f.oracles, "rust kafka", ORACLE_TOP_K);
    assert!(
        inf_hits.is_empty(),
        "supertable AND must be empty; got {inf_hits:?}"
    );
    assert!(
        ora_hits.is_empty(),
        "oracle AND must be empty; got {ora_hits:?}"
    );
}

// ---- Tests: prefix-row exercise ---------------------------------------

#[test]
fn oracle_prefix_query_matches_explicit_term_or() {
    // The supertable expands `alphafox` via per-superfile FST walk,
    // then runs a per-superfile OR over the expansion. Mirror this
    // by running a brute-force OR over the same explicit term list.
    let f = &*STANDARD_FIXTURE;
    let prefix = "alphafox";
    let expanded: Vec<String> = (0..N_PREFIX_TERMS)
        .map(|i| format!("alphafox{i:02}"))
        .collect();

    let inf_hits = supertable_prefix_global(&f.infino, prefix, ORACLE_TOP_K, CHUNK_SIZE);
    let ora_hits = brute_force_terms_top_k(&f.oracles, &expanded, 10);

    let want: HashSet<u64> = [14u64, 29, 44, 59].into_iter().collect();
    let inf_set: HashSet<u64> = inf_hits.iter().take(N_PREFIX_TERMS).copied().collect();
    let ora_set: HashSet<u64> = ora_hits.iter().take(N_PREFIX_TERMS).copied().collect();
    assert_eq!(inf_set, want, "supertable prefix hits = {inf_hits:?}");
    assert_eq!(ora_set, want, "oracle explicit-OR hits = {ora_hits:?}");
}

#[test]
fn prefix_skip_prunes_superfiles_without_matching_lex_range() {
    // Plant a prefix term in only one superfile; verify the prefix
    // search returns exactly that doc and skip pruning prevents
    // other superfiles from contributing.
    let mut corp: Vec<(u64, String)> = planted_corpus()
        .into_iter()
        .map(|(id, t)| (id, t.to_string()))
        .collect();
    corp[0].1.push_str(" quokka_unique");
    let infino = build_supertable(&corp, SUPERFILES);
    let r = infino.reader().expect("reader");
    let hits = r
        .bm25_search_prefix("title", "quokka", ORACLE_TOP_K_SMALL)
        .expect("prefix");
    assert_eq!(hits.len(), 1);
    let manifest = r.manifest();
    let target_uri = manifest.superfiles[0].uri;
    assert_eq!(hits[0].superfile, target_uri);
    assert_eq!(hits[0].local_doc_id, 0);
}

/// Superfiles the shared-prefix corpus is split into, so each holds a
/// different subset of a prefix's expansion.
const PREFIX_SUPERFILES: usize = 3;
/// Top-k that returns every prefix match.
const PREFIX_K_ALL: usize = 64;
/// Top-k below the broad prefix's match count, so the pruning bar is live.
const PREFIX_K_TRUNCATED: usize = 2;
/// Largest allowed gap between an engine score and the reference's.
const PREFIX_SCORE_TOLERANCE: f32 = 1e-3;

/// Docs with shared prefixes, so an expansion is non-trivial: `run`,
/// `runtime`, `running`, `rust` and `rustacean` share `ru`, and `rust*` is
/// its own sub-group. `python` and `pandas` share `p` but never `ru`.
fn shared_prefix_corpus() -> Vec<(u64, String)> {
    [
        "run the loop",
        "runtime scheduler tokio",
        "running fast async",
        "rust systems language",
        "rustacean community crate",
        "rust runtime bindings",
        "python data pipeline",
        "pandas dataframe numpy",
        "go concurrency channels",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, text)| (i as u64, text.to_owned()))
    .collect()
}

/// Distinct terms of `corpus` that start with `prefix`: what the engine's
/// expansion must reproduce across the whole table.
fn prefix_expansion(corpus: &[(u64, String)], prefix: &str) -> Vec<String> {
    let terms: HashSet<&str> = corpus
        .iter()
        .flat_map(|(_, text)| text.split_whitespace())
        .filter(|term| term.starts_with(prefix))
        .collect();
    terms.into_iter().map(str::to_owned).collect()
}

/// The engine's prefix top-`k` agrees with a brute-force `Or` over the
/// prefix's table-wide expansion: the same docs when `k` covers every
/// match (a subset of the reference top-`k` otherwise), each at the
/// reference's score.
fn assert_prefix_scores_match(
    st: &Supertable,
    oracles: &[FileOracle],
    corpus: &[(u64, String)],
    prefix: &str,
    k: usize,
) {
    let chunk_size = corpus.len().div_ceil(PREFIX_SUPERFILES);
    let hits = st
        .reader()
        .expect("reader")
        .bm25_search_prefix("title", prefix, k)
        .expect("prefix search");
    let scores: Vec<f32> = hits.iter().map(|hit| hit.score).collect();
    let got: Vec<(u64, f32)> = supertable_to_global_ids(st, hits, chunk_size)
        .into_iter()
        .zip(scores)
        .collect();

    let terms = prefix_expansion(corpus, prefix);
    let reference: Vec<(u64, f32)> = oracles
        .iter()
        .flat_map(|o| o.own(o.scorer.top_k_terms(&terms, usize::MAX)))
        .collect();
    let reference_scores: HashMap<u64, f32> = reference.iter().copied().collect();
    let want: HashSet<u64> = merge_global_top_k(reference, k).into_iter().collect();
    let got_ids: HashSet<u64> = got.iter().map(|(doc, _)| *doc).collect();
    if k >= reference_scores.len() {
        assert_eq!(got_ids, want, "prefix {prefix:?}: match sets disagree");
    } else {
        assert_eq!(got.len(), k, "prefix {prefix:?}: {got:?}");
        assert!(
            got_ids.is_subset(&want),
            "prefix {prefix:?}: {got:?} is not the reference top-{k} {want:?}"
        );
    }
    for (doc, score) in &got {
        let expected = reference_scores[doc];
        assert!(
            (score - expected).abs() <= PREFIX_SCORE_TOLERANCE,
            "prefix {prefix:?} doc {doc}: engine={score} reference={expected}"
        );
    }
}

#[test]
fn oracle_prefix_scores_match_brute_force_or() {
    let corpus = shared_prefix_corpus();
    let st = build_supertable(&corpus, PREFIX_SUPERFILES);
    let oracles = build_oracles(&corpus, PREFIX_SUPERFILES);
    // Broad (`ru*`), narrower (`rust*`), single-term (`runt*`), and one
    // whose matches sit in a single superfile (`p*`).
    for prefix in ["ru", "run", "rust", "runt", "p"] {
        assert_prefix_scores_match(&st, &oracles, &corpus, prefix, PREFIX_K_ALL);
    }
    assert_prefix_scores_match(&st, &oracles, &corpus, "ru", PREFIX_K_TRUNCATED);
}

// ---- Tests: empty + no-match ----------------------------------------

#[test]
fn oracle_no_match_returns_empty() {
    let f = &*STANDARD_FIXTURE;
    let inf_hits = supertable_search_global(
        &f.infino,
        "definitelynotpresent",
        ORACLE_TOP_K_SMALL,
        CHUNK_SIZE,
    );
    let ora_hits = brute_force_top_k(&f.oracles, "definitelynotpresent", ORACLE_TOP_K_SMALL);
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
    // 5K docs × 4 superfiles = 1250 docs/superfile. The file oracles
    // model the engine's scoring exactly (table-wide idf, each file's
    // declared average); only the one-byte doc-length quantization and
    // tie order remain, which the 90 % overlap threshold absorbs.
    let n_docs = 5_000;
    let corp = zipfian_corpus(n_docs, 42);
    let infino = build_supertable(&corp, SUPERFILES);
    let oracles = build_oracles(&corp, SUPERFILES);
    let k = ZIPFIAN_TOP_K;

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
        let inf = supertable_search_global(&infino, q, k, n_docs / SUPERFILES);
        let ora = brute_force_top_k(&oracles, q, k);
        let inf_set: HashSet<u64> = inf.iter().copied().collect();
        let ora_set: HashSet<u64> = ora.iter().copied().collect();
        let common = inf_set.intersection(&ora_set).count();
        let target = inf_set.len().min(ora_set.len());
        let threshold = (target * 9) / 10;
        assert!(
            common >= threshold,
            "{label}: top-{k} overlap {common}/{target} below 90% threshold; \
             supertable={inf:?} oracle={ora:?}",
        );
    }
}

// ---- the tiers' progressive merge ----------------------------------

/// Superfiles in the tier-merge arm.
const TIER_SUPERFILES: usize = 8;
/// Documents in the tier-merge corpus: 500 per superfile, so a common
/// term has hundreds of documents in each.
const TIER_DOCS: usize = 4_000;
/// Documents each superfile keeps per term: two blocks, so a common
/// term's head is read block by block and bounded by its tails, while a
/// rare term's head is whole.
const TIER_HEAD_DEPTH: usize = 128;
/// The `k` values the tier-merge arm compares at.
const TIER_KS: [usize; 4] = [1, 10, 100, 1000];

/// The tiers' merge, with heads cut to two blocks, returns hit for hit
/// and score for score what the ceiling path returns on the same table
/// — for one, two, three and four terms, as disjunctions and
/// conjunctions, at every `k` — and both agree with the brute-force
/// reference. The merge is exercised, not bypassed: the index reads tier
/// entries for every query, and answers a rare conjunction outright.
#[test]
fn oracle_tier_merge_with_cut_heads_matches_the_ceiling_path_and_brute_force() {
    let corp = zipfian_corpus(TIER_DOCS, 7);
    let params = OracleBm25Params::default();
    // A term index needs storage: both arms are tables on disk. Opens
    // are one at a time on both, so the counts below are the bounds'
    // doing and not a race between the opens in a window.
    let storage_in = |dir: &TempDir| -> Arc<dyn StorageProvider> {
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"))
    };
    let with_dir = TempDir::new().expect("tempdir");
    let without_dir = TempDir::new().expect("tempdir");
    let with_tiers = build_supertable_with(&corp, TIER_SUPERFILES, params, |o| {
        o.with_storage(storage_in(&with_dir))
            .with_term_head_depth(TIER_HEAD_DEPTH)
            .with_bound_ordered_open_window(1)
            // The merge under test, for every query the tiers can serve.
            .with_tier_merge_gate(false)
    });
    let without_tiers = build_supertable_with(&corp, TIER_SUPERFILES, params, |o| {
        o.with_storage(storage_in(&without_dir))
            .with_term_tier_entries_per_term(0)
            .with_bound_ordered_open_window(1)
    });
    let oracles = build_oracles(&corp, TIER_SUPERFILES);
    let chunk_size = TIER_DOCS / TIER_SUPERFILES;

    let queries: [(&str, &str, BoolMode); 10] = [
        ("single_common", "term00001", BoolMode::Or),
        ("single_mid", "term00200", BoolMode::Or),
        ("single_rare", "term05000", BoolMode::Or),
        ("two_common_or", "term00001 term00002", BoolMode::Or),
        ("two_common_and", "term00001 term00002", BoolMode::And),
        ("common_mid_and", "term00003 term00300", BoolMode::And),
        ("common_rare_and", "term00001 term03000", BoolMode::And),
        ("three_or", "term00005 term00050 term00500", BoolMode::Or),
        ("three_and", "term00001 term00002 term00003", BoolMode::And),
        (
            "four_or",
            "term00010 term00020 term00030 term00040",
            BoolMode::Or,
        ),
    ];
    let hits_of = |st: &Supertable, q: &str, mode: BoolMode, k: usize| {
        let (hits, stats) = infino::runtime_metrics::op_stats::with_op_stats(|| {
            st.reader()
                .expect("reader")
                .bm25_hits("title", q, k, Bm25SearchOptions::new().with_mode(mode))
                .expect("search")
        });
        let ids = supertable_to_global_ids(st, hits.clone(), chunk_size);
        let scored: Vec<(u64, f32)> = ids.into_iter().zip(hits.iter().map(|h| h.score)).collect();
        (scored, stats)
    };
    for (label, q, mode) in queries {
        for k in TIER_KS {
            let (tiered, stats) = hits_of(&with_tiers, q, mode, k);
            let (plain, _) = hits_of(&without_tiers, q, mode, k);
            assert_eq!(
                tiered, plain,
                "{label} k={k}: the tiers' merge must return what the ceiling path returns"
            );
            // Every query here is served by the merge, which reads tiers.
            let mut terms: Vec<String> = Vec::new();
            default_tokenizer().tokenize_each(q, &mut |t| terms.push(t.to_owned()));
            assert!(
                stats.tier_entries > 0,
                "{label} k={k}: {} tier entries read",
                stats.tier_entries
            );
            // The reference: the same ranking, up to the one-byte length
            // quantization at the boundary of the top k.
            let reference = match mode {
                BoolMode::And => brute_force_and_top_k(&oracles, q, k),
                _ => brute_force_terms_top_k(&oracles, &terms, k),
            };
            let got: HashSet<u64> = tiered.iter().map(|h| h.0).collect();
            let want: HashSet<u64> = reference.iter().copied().collect();
            let common = got.intersection(&want).count();
            let target = got.len().min(want.len());
            assert!(
                common * 10 >= target * 9,
                "{label} k={k}: top-{k} overlap {common}/{target} below 90%"
            );
            assert_eq!(
                got.len(),
                want.len(),
                "{label} k={k}: the engine and the reference return different counts"
            );
        }
    }
    // A rare conjunction's heads are whole, so the index answers it
    // without opening a superfile.
    let (_, stats) = hits_of(&with_tiers, "term03000 term03001", BoolMode::And, 10);
    assert_eq!(
        stats.score_pruning_survived, 0,
        "a conjunction of rare terms is answered from the index"
    );
}

// ---- single-term global-idf skip path ------------------------------
//
// A lone scored term takes the BlockMaxWAND walk, which prunes against
// per-block upper bounds that were stored with the superfile's LOCAL
// idf. Under global statistics the walk scores with the corpus idf
// instead, so it rescales those bounds by `global_idf / local_idf` —
// the score is linear in idf, which makes the rescale exact. If it were
// wrong in either direction the walk would drop documents that belong
// in the top-k or admit ones that do not, and only a fragmented table
// shows it: on a single superfile local idf *is* the global idf and the
// rescale is the identity.
//
// The fixture below therefore compares a fragmented table against a
// single-superfile table holding the same corpus, and is built so the
// comparison is exact and the rescale is never trivial:
//
//   * `common`'s local frequency differs in every superfile (all rows,
//     an eighth, a half), so no superfile's local idf equals the global
//     one and every walk rescales by a different ratio.
//   * every document is the same token length, so per-superfile `avgdl`
//     is identical across the fragmented layout and equal to the single
//     superfile's — leaving idf as the only thing layout could change.
//   * the term's frequency within a handful of documents is distinct
//     (2..=9 against a baseline of 1), so the head of the ranking is
//     ordered strictly by score, with no ties to make the comparison
//     ambiguous, and those documents sit in different superfiles so the
//     cross-superfile merge is part of what is being checked.
//   * each superfile holds enough matching rows to span several posting
//     blocks, so the skip table is genuinely consulted rather than the
//     whole list being scored anyway.

/// Superfiles in the fragmented arm.
const SKIP_SUPERFILES: usize = 3;
/// Rows per superfile — several 128-row posting blocks' worth, so the
/// BlockMaxWAND skip table is exercised rather than bypassed.
const SKIP_DOCS_PER_SUPERFILE: usize = 384;
/// Tokens per document, uniform so `avgdl` is layout-independent.
const SKIP_DOC_LEN: usize = 12;
/// Top-k for the skip-path comparison: small enough that the walk's
/// threshold rises early and prunes most blocks.
const SKIP_TOP_K: usize = 5;
/// `(global doc id, term frequency)` for the documents that carry a
/// distinct `common` frequency; every other matching document has
/// frequency 1.
///
/// Placement is the point of this fixture, not decoration. The walk
/// skips a posting BLOCK whose stored upper bound cannot beat the
/// running threshold, so a bound that is too low is only observable
/// when a wrongly skipped block held a document that belongs in the
/// top-k. Half of the top-k therefore sits deliberately LATE in its
/// superfile's posting list — id 300 is in the third of the first
/// superfile's three blocks, id 1068 in the second of the third
/// superfile's two — while the rest sit in the first block, which fills
/// the threshold early and makes those later blocks skippable. An
/// earlier version of this fixture put every high-frequency document in
/// the first block and passed even with the rescale removed.
const SKIP_HOT_DOCS: &[(u64, usize)] = &[
    // superfile 0 (every row matches, three posting blocks)
    (300, 9),
    (200, 6),
    (10, 4),
    // superfile 1 (every eighth row matches, one posting block)
    (704, 7),
    (392, 3),
    // superfile 2 (every second row matches, two posting blocks)
    (1068, 8),
    (780, 5),
    (900, 2),
];

/// The fixture corpus: `SKIP_SUPERFILES` blocks of rows in which
/// `common` appears in every row, every eighth row, and every second
/// row respectively.
fn skip_path_corpus() -> Vec<(u64, String)> {
    let hot: std::collections::HashMap<u64, usize> = SKIP_HOT_DOCS.iter().copied().collect();
    let mut corpus = Vec::with_capacity(SKIP_SUPERFILES * SKIP_DOCS_PER_SUPERFILE);
    for segment in 0..SKIP_SUPERFILES {
        for row in 0..SKIP_DOCS_PER_SUPERFILE {
            let id = (segment * SKIP_DOCS_PER_SUPERFILE + row) as u64;
            let matches = match segment {
                0 => true,
                1 => row % 8 == 0,
                _ => row % 2 == 0,
            };
            let tf = match matches {
                false => 0,
                true => hot.get(&id).copied().unwrap_or(1),
            };
            // `tf` copies of the term, filler to a fixed length, then a
            // unique token so every row is identifiable.
            let mut tokens: Vec<String> = std::iter::repeat_n("common".to_string(), tf).collect();
            tokens.extend(std::iter::repeat_n(
                "pad".to_string(),
                SKIP_DOC_LEN - tf - 1,
            ));
            tokens.push(format!("d{id:05}"));
            debug_assert_eq!(tokens.len(), SKIP_DOC_LEN);
            corpus.push((id, tokens.join(" ")));
        }
    }
    corpus
}

/// `(global id, score)` for a single-term global-stats search.
fn skip_path_hits(st: &Supertable, k: usize, chunk_size: usize) -> Vec<(u64, f32)> {
    let hits = st
        .reader()
        .expect("reader")
        .bm25_hits(
            "title",
            "common",
            k,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
        )
        .expect("single-term global search");
    let scores: Vec<f32> = hits.iter().map(|h| h.score).collect();
    supertable_to_global_ids(st, hits, chunk_size)
        .into_iter()
        .zip(scores)
        .collect()
}

#[test]
fn single_term_global_idf_skip_matches_a_single_superfile() {
    let corpus = skip_path_corpus();
    let fragmented = build_supertable(&corpus, SKIP_SUPERFILES);
    let single = build_supertable(&corpus, 1);
    assert_eq!(
        fragmented.reader().expect("reader").n_superfiles(),
        SKIP_SUPERFILES,
        "fragmented arm must actually be fragmented"
    );
    assert_eq!(
        single.reader().expect("reader").n_superfiles(),
        1,
        "single arm must be one superfile"
    );

    // The rescaled walk over three superfiles must return exactly what
    // the un-rescaled walk over one superfile returns — same documents,
    // same scores, same order.
    let fragmented_top = skip_path_hits(&fragmented, SKIP_TOP_K, SKIP_DOCS_PER_SUPERFILE);
    let single_top = skip_path_hits(&single, SKIP_TOP_K, corpus.len());
    assert_eq!(
        fragmented_top.len(),
        SKIP_TOP_K,
        "fixture must fill the top-k"
    );
    assert_eq!(
        fragmented_top, single_top,
        "the single-term global-idf walk ranked a fragmented table differently from one \
         superfile holding the same corpus — the rescale of the stored block-max bounds \
         does not match the idf the walk scores with"
    );

    // The head is ordered strictly by the planted frequencies, which is
    // what makes the equality above unambiguous rather than a tie
    // ordering that happened to agree.
    let mut expected: Vec<(u64, usize)> = SKIP_HOT_DOCS.to_vec();
    expected.sort_by_key(|(_, tf)| Reverse(*tf));
    let expected_ids: Vec<u64> = expected
        .iter()
        .take(SKIP_TOP_K)
        .map(|(id, _)| *id)
        .collect();
    assert_eq!(
        fragmented_top.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        expected_ids,
        "top-k should be the documents with the highest planted term frequency"
    );
    for pair in fragmented_top.windows(2) {
        assert!(
            pair[0].1 > pair[1].1,
            "planted frequencies must give strictly decreasing scores, got {:?}",
            fragmented_top
        );
    }
}

// ---- the running table-wide average, one declared value per file ------
//
// Under global statistics idf is table-wide, but the average document
// length a superfile normalizes with is the one it declares: the table's
// token and document totals as of the file's commit — every file already
// committed, plus its own rows — rounded to the stored fixed point. Files
// flushed earlier in the same commit are not yet committed and do not
// count. A reference scoring the whole table at one average cannot grade
// this; the one below re-points a whole-table oracle (table-wide idf) at
// each file's declared average and scores that file's rows with it.

/// Commits of the running-average fixture.
const RUNNING_COMMITS: usize = 4;
/// Rows per commit. With two files per commit each holds 450 rows, so
/// `every` spans four posting blocks in every file.
const RUNNING_DOCS_PER_COMMIT: usize = 900;
/// Filler tokens added per commit index, so each commit's own average
/// differs from every earlier one and the running average moves at
/// every commit.
const RUNNING_PAD_PER_COMMIT: usize = 2;
/// Rows per length band inside a commit: the band index cycles through
/// `0..RUNNING_LEN_BANDS` filler tokens, so lengths vary inside a
/// commit *and* the two halves of a commit average differently — which
/// is what makes "an earlier file of the same commit is not counted"
/// observable in the second file's declared average.
const RUNNING_LEN_BAND_ROWS: u64 = 150;
const RUNNING_LEN_BANDS: u64 = 5;
/// Planting periods of the fixture's terms.
const RUNNING_ALPHA_PERIOD: u64 = 3;
const RUNNING_BETA_PERIOD: u64 = 4;
const RUNNING_GAMMA_PERIOD: u64 = 7;
const RUNNING_NEG_PERIOD: u64 = 5;
/// Score tolerance between the two scorers (f32 accumulation order).
const RUNNING_SCORE_TOLERANCE: f32 = 1e-3;
/// Relative tolerance on a declared average against the commit rule
/// (the `f64` to `f32` step; both sides round to the same fixed point).
const RUNNING_AVGDL_TOLERANCE: f32 = 1e-6;
/// Truncated top-k, graded by score multiset.
const RUNNING_TOP_K: usize = 10;
/// Query shapes graded under `Or`: single term, union, intersection by
/// sigil, negation of a term and of a phrase, must with should, phrase.
const RUNNING_OR_QUERIES: [&str; 9] = [
    "alpha",
    "alpha beta",
    "+alpha +beta",
    "alpha -neg",
    "+gamma alpha",
    "\"alpha beta\"",
    "beta -\"alpha beta\"",
    "every -neg",
    "+every gamma -neg",
];
/// Query shapes graded under `And`.
const RUNNING_AND_QUERIES: [&str; 2] = ["alpha beta -neg", "gamma every"];

/// The fixture corpus. `alpha beta` is adjacent wherever both occur.
/// Longest document: five planted terms, four filler, six pad — under
/// the length quantizer's exact region, so scores compare by value.
fn running_average_corpus() -> Vec<(u64, String)> {
    let n = (RUNNING_COMMITS * RUNNING_DOCS_PER_COMMIT) as u64;
    (0..n)
        .map(|d| {
            let commit = d as usize / RUNNING_DOCS_PER_COMMIT;
            let band =
                ((d % RUNNING_DOCS_PER_COMMIT as u64) / RUNNING_LEN_BAND_ROWS) % RUNNING_LEN_BANDS;
            let mut t = vec!["every"];
            if d.is_multiple_of(RUNNING_ALPHA_PERIOD) {
                t.push("alpha");
            }
            if d.is_multiple_of(RUNNING_BETA_PERIOD) {
                t.push("beta");
            }
            if d.is_multiple_of(RUNNING_GAMMA_PERIOD) {
                t.push("gamma");
            }
            if d.is_multiple_of(RUNNING_NEG_PERIOD) {
                t.push("neg");
            }
            t.extend(repeat_n("fill", band as usize));
            t.extend(repeat_n("pad", commit * RUNNING_PAD_PER_COMMIT));
            (d, t.join(" "))
        })
        .collect()
}

/// Ingest the corpus in [`RUNNING_COMMITS`] commits with a writer pool
/// of `files_per_commit` threads and the byte split target disabled, so
/// every commit is sharded across the whole pool into that many files.
fn build_running_average_table(corpus: &[(u64, String)], files_per_commit: usize) -> Supertable {
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(files_per_commit)
            .build()
            .expect("pool"),
    );
    let opts = SupertableOptions::new(
        schema_id_title(),
        vec![FtsConfig::new("title").positions(true)],
        vec![],
    )
    .expect("opts")
    .with_writer_pool(pool)
    .with_superfile_buffer_split_mb(0);
    let st = Supertable::create(opts).expect("create");
    let mut w = st.writer().expect("writer");
    for chunk in corpus.chunks(RUNNING_DOCS_PER_COMMIT) {
        let titles: Vec<&str> = chunk.iter().map(|(_, t)| t.as_str()).collect();
        w.append(&build_title_batch(&titles)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

/// One file of the fixture: which rows it holds and the average it
/// declares.
struct RunningSpan {
    uri: SuperfileUri,
    rows: Range<u64>,
    declared_avgdl: f32,
}

/// The average a file declares when it joins a table whose committed
/// totals are `committed`: the merged totals' average, rounded as the
/// writer stores it. Both sides of the comparison run on the engine's own
/// [`ColumnLengthStats`], so the test cannot drift from the writer's
/// arithmetic the way a private copy of these two numbers would.
fn declared_under(committed: ColumnLengthStats, own: &ColumnLengthStats) -> f32 {
    let mut merged = committed;
    merged.merge_with(own);
    stored_avgdl(merged.avgdl())
}

/// Recover the fixture's files in row order and check each one against
/// the corpus and the commit rule.
///
/// Rows reach files in append order (a commit's buffer is split into
/// row-ordered shards and ids are assigned monotonically), so the
/// manifest ordered by first id gives contiguous row ranges. The
/// mapping is verified rather than assumed: each file's stored document
/// lengths must equal the corpus's for the rows assigned to it. Each
/// file's declared average is then checked against the rule — the
/// totals of every file of *earlier commits* plus its own — and, for a
/// file that is not the first of its commit, checked to differ from
/// what counting the same commit's earlier file would have given, so
/// the exclusion is pinned by a value that would fail, not by an
/// arithmetic identity.
fn running_spans(st: &Supertable, corpus: &[(u64, String)]) -> Vec<RunningSpan> {
    let tok = default_tokenizer();
    let dl: Vec<u32> = corpus
        .iter()
        .map(|(_, t)| {
            let mut n = 0u32;
            tok.tokenize_each(t, &mut |_| n += 1);
            n
        })
        .collect();
    let reader = st.reader().expect("reader");
    let mut entries: Vec<Arc<SuperfileEntry>> = reader.manifest().superfiles.to_vec();
    entries.sort_by_key(|e| e.id_min);

    let mut spans = Vec::with_capacity(entries.len());
    let mut committed = ColumnLengthStats::default();
    let mut same_commit = ColumnLengthStats::default();
    let mut start = 0u64;
    for entry in &entries {
        let rows = start..start + entry.n_docs;
        let file = reader.open_superfile(entry).expect("open superfile");
        let fts = file.fts().expect("full-text index");
        let stored = fts.column_doc_lengths("title").expect("stored doc lengths");
        assert_eq!(
            stored,
            dl[rows.start as usize..rows.end as usize],
            "file {:?} does not hold rows {rows:?}: manifest order is not row order",
            entry.uri
        );
        let declared = fts
            .fts_columns_config()
            .find(|c| c.name == "title")
            .expect("title column")
            .avgdl();

        let first_of_commit = (start as usize).is_multiple_of(RUNNING_DOCS_PER_COMMIT);
        if first_of_commit {
            committed.merge_with(&same_commit);
            same_commit = ColumnLengthStats::default();
        }
        let own = ColumnLengthStats::from_lengths(stored.iter().copied());
        let expected = declared_under(committed, &own);
        assert!(
            (declared - expected).abs() <= RUNNING_AVGDL_TOLERANCE * expected,
            "file at rows {rows:?} declares avgdl {declared}, the commit rule predicts {expected}"
        );
        if !first_of_commit {
            let mut with_earlier_flush = committed;
            with_earlier_flush.merge_with(&same_commit);
            let counting_earlier_flush = declared_under(with_earlier_flush, &own);
            assert!(
                (declared - counting_earlier_flush).abs() > RUNNING_AVGDL_TOLERANCE * expected,
                "fixture cannot tell whether an earlier file of the same commit was counted \
                 ({declared} vs {counting_earlier_flush}); vary the lengths across a commit"
            );
        }
        same_commit.merge_with(&own);
        spans.push(RunningSpan {
            uri: entry.uri,
            rows,
            declared_avgdl: declared,
        });
        start += entry.n_docs;
    }
    assert_eq!(
        start as usize,
        corpus.len(),
        "files do not cover the corpus"
    );
    spans
}

/// The reference for `query`: every row scored with table-wide idf at the
/// table's average, whatever average its file declares, merged by score
/// then id.
fn running_expected(
    corpus: &[(u64, String)],
    query: &str,
    mode: BoolMode,
    k: usize,
) -> Vec<(u64, f32)> {
    let refs: Vec<(u64, &str)> = corpus.iter().map(|(i, t)| (*i, t.as_str())).collect();
    let tok = default_tokenizer();
    let mut all = BruteForceBm25::index(&refs, tok.as_ref())
        .with_avgdl(corpus_avgdl(corpus))
        .top_k_query(query, mode, tok.as_ref(), usize::MAX);
    all.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    all.truncate(k);
    all
}

/// The engine's hits for `query` under global statistics, as
/// `(global row, score)`.
fn running_engine_hits(
    st: &Supertable,
    spans: &[RunningSpan],
    query: &str,
    mode: BoolMode,
    k: usize,
) -> Vec<(u64, f32)> {
    st.reader()
        .expect("reader")
        .bm25_hits("title", query, k, Bm25SearchOptions::new().with_mode(mode))
        .expect("bm25 search")
        .into_iter()
        .map(|h| {
            let span = spans
                .iter()
                .find(|s| s.uri == h.superfile)
                .expect("hit from a known file");
            (span.rows.start + u64::from(h.local_doc_id), h.score)
        })
        .collect()
}

/// Grade one shape: with `k` covering every match, identical match
/// sets and per-row scores; with a truncated `k`, identical score
/// multisets (ties may reorder rows across files, never scores).
fn assert_running_shape(
    st: &Supertable,
    corpus: &[(u64, String)],
    spans: &[RunningSpan],
    query: &str,
    mode: BoolMode,
    k: usize,
) {
    let want = running_expected(corpus, query, mode, k);
    let got = running_engine_hits(st, spans, query, mode, k);
    assert_eq!(
        got.len(),
        want.len(),
        "{query:?} ({mode:?}, k={k}): hit count"
    );
    if k >= corpus.len() {
        let want_scores: HashMap<u64, f32> = want.iter().copied().collect();
        for (d, s) in &got {
            let w = want_scores
                .get(d)
                .unwrap_or_else(|| panic!("{query:?} ({mode:?}): row {d} is not a match"));
            assert!(
                (s - w).abs() <= RUNNING_SCORE_TOLERANCE,
                "{query:?} ({mode:?}): row {d} scored {s}, reference {w}"
            );
        }
    }
    let mut got_scores: Vec<f32> = got.iter().map(|(_, s)| *s).collect();
    let mut want_scores: Vec<f32> = want.iter().map(|(_, s)| *s).collect();
    got_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    want_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    for (g, w) in got_scores.iter().zip(&want_scores) {
        assert!(
            (g - w).abs() <= RUNNING_SCORE_TOLERANCE,
            "{query:?} ({mode:?}, k={k}): score {g} in the top-k, reference {w}"
        );
    }
}

/// One arm of the running-average fixture: `files_per_commit` files per
/// commit, every shape at full and truncated depth, plus the check that
/// a single-average reference would have disagreed.
fn run_running_average_arm(files_per_commit: usize) {
    let corpus = running_average_corpus();
    let st = build_running_average_table(&corpus, files_per_commit);
    let spans = running_spans(&st, &corpus);
    assert_eq!(
        spans.len(),
        RUNNING_COMMITS * files_per_commit,
        "every commit must produce {files_per_commit} file(s)"
    );

    for q in RUNNING_OR_QUERIES {
        assert_running_shape(&st, &corpus, &spans, q, BoolMode::Or, corpus.len());
        assert_running_shape(&st, &corpus, &spans, q, BoolMode::Or, RUNNING_TOP_K);
    }
    for q in RUNNING_AND_QUERIES {
        assert_running_shape(&st, &corpus, &spans, q, BoolMode::And, corpus.len());
        assert_running_shape(&st, &corpus, &spans, q, BoolMode::And, RUNNING_TOP_K);
    }

    // The fixture's files declare averages other than the table's, so the
    // agreement above shows every file scored at the table's average and
    // not at what it declares.
    let table = corpus_avgdl(&corpus);
    assert!(
        spans
            .iter()
            .any(|s| (s.declared_avgdl - table).abs() > RUNNING_AVGDL_TOLERANCE * table),
        "every file declares the table's average; the fixture's commits do not differ \
         enough in length to tell the two apart"
    );
}

/// One file per commit: each file declares the running table-wide
/// average as of its commit, and the engine scores it at the table's.
#[test]
fn running_average_with_one_file_per_commit_matches_the_reference() {
    run_running_average_arm(1);
}

/// Two files per commit: the second file of a commit declares the
/// average *without* the first file's rows — the manifest the writer
/// reads has not been advanced yet — and both are scored at the table's
/// average. A writer that folded the earlier flush in fails the
/// declared-average check; a reader that scored a file at what it declares
/// fails the score comparison.
#[test]
fn running_average_with_two_files_per_commit_matches_the_reference() {
    run_running_average_arm(2);
}

// ---- term-index ceilings across build-time averages --------------------
//
// The term index stores each file's score ceilings at the average it was
// built at. A file built while the table's average was shorter scores
// higher once the average grows, so its ceiling must grow with it, or
// bound-ordered opening skips a file that holds the best match.

/// Tokens in a long doc of the ceiling fixture: enough to lift the table's
/// average far above the short commit's.
const CEILING_LONG_DOC_TOKENS: usize = 40;
/// Superfiles open at a time: one, so the long commit sets the floor
/// before the short commit's ceiling is checked.
const CEILING_OPEN_WINDOW: usize = 1;
/// Top-k of the ceiling query: the single best match.
const CEILING_TOP_K: usize = 1;
/// Largest allowed gap between the engine's score and the reference's.
const CEILING_SCORE_TOLERANCE: f32 = 1e-3;

/// Two commits: short docs, one of them a strong `zeta` match, then long
/// docs holding a weaker one. The second commit's file opens first, and
/// its match sets the floor the first file's ceiling is checked against.
fn ceiling_commits() -> Vec<Vec<String>> {
    let filler = |n: usize| vec!["f"; n].join(" ");
    let weaker = format!("zeta zeta {}", filler(CEILING_LONG_DOC_TOKENS / 2 - 2));
    let long = filler(CEILING_LONG_DOC_TOKENS);
    vec![
        vec!["zeta a".to_owned(), "b c".to_owned()],
        vec![weaker, long.clone(), long],
    ]
}

#[test]
fn oracle_ceilings_follow_the_table_average() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("pool"),
    );
    let opts = SupertableOptions::new(schema_id_title(), vec![FtsConfig::new("title")], vec![])
        .expect("opts")
        .with_writer_pool(pool)
        .with_storage(storage)
        .with_bound_ordered_open_window(CEILING_OPEN_WINDOW);
    let st = Supertable::create(opts).expect("create");
    let commits = ceiling_commits();
    let mut w = st.writer().expect("writer");
    for titles in &commits {
        let titles: Vec<&str> = titles.iter().map(String::as_str).collect();
        w.append(&build_title_batch(&titles)).expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    let corpus: Vec<(u64, String)> = commits
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(i, text)| (i as u64, text))
        .collect();
    let tok = default_tokenizer();
    let view: Vec<(u64, &str)> = corpus.iter().map(|(i, t)| (*i, t.as_str())).collect();
    let want = BruteForceBm25::index(&view, tok.as_ref())
        .with_avgdl(corpus_avgdl(&corpus))
        .top_k("zeta", CEILING_TOP_K, tok.as_ref());

    let reader = st.reader().expect("reader");
    let manifest = reader.manifest();
    assert_eq!(manifest.superfiles.len(), 2, "one superfile per commit");
    let first_rows: HashMap<SuperfileUri, u64> = manifest
        .superfiles
        .iter()
        .scan(0u64, |next, entry| {
            let first = *next;
            *next += entry.n_docs;
            Some((entry.uri, first))
        })
        .collect();
    let got: Vec<(u64, f32)> = reader
        .bm25_hits(
            "title",
            "zeta",
            CEILING_TOP_K,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
        )
        .expect("search")
        .into_iter()
        .map(|hit| {
            (
                first_rows[&hit.superfile] + u64::from(hit.local_doc_id),
                hit.score,
            )
        })
        .collect();

    assert_eq!(
        got.iter().map(|(doc, _)| *doc).collect::<Vec<_>>(),
        want.iter().map(|(doc, _)| *doc).collect::<Vec<_>>(),
        "engine {got:?}, reference {want:?}"
    );
    for ((_, score), (_, expected)) in got.iter().zip(&want) {
        assert!(
            (score - expected).abs() <= CEILING_SCORE_TOLERANCE,
            "engine {score}, reference {expected}"
        );
    }
}
