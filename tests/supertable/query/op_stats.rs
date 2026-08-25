// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query work stats over the sync BM25 surface: a query wrapped in
//! [`with_op_stats`] reports the posting bytes its kernels indexed into,
//! deterministically — the same query against the same committed corpus
//! reports the same number on a cold first run and a warm repeat, and a
//! reader minted outside a scope records nothing.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, thread};

use arrow_array::{
    Array, ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, Int64Array,
    LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{Expr, col, lit};
use infino::{
    ConnectOptions, Connection, IndexSpec, Metric, connect, connect_with,
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::{FtsConfig, VectorConfig},
        fts::reader::{Bm25Stats, BoolMode},
        vector::rerank_codec::RerankCodec,
    },
    supertable::{
        Supertable, SupertableOptions,
        query::vector::{VectorFilter, VectorSearchOptions},
    },
    test_helpers::{default_tokenizer, default_vector_config},
};
use tempfile::TempDir;

/// Docs per committed segment. Commits row-shard across the writer
/// pool, so this must keep every term's per-superfile df ≥ 2 —
/// df=1 terms store inline in the FST with genuinely zero
/// postings-region bytes, which would make the assertions vacuous.
const DOCS_PER_SEGMENT: usize = 40;
/// Rayon pool size for deterministic builds (two shards per commit).
const RAYON_POOL_THREADS: usize = 2;
/// Top-k ≥ the corpus size so no assertion depends on ranking cutoffs.
const TOP_K: usize = 128;

/// Schema `[title (FTS, no positions)]` — the smallest corpus that
/// exercises the full multi-superfile BM25 fan-out.
fn options_title_only() -> SupertableOptions {
    let writer_pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(RAYON_POOL_THREADS)
            .build()
            .expect("writer pool"),
    );
    let schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        Vec::new(),
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(writer_pool)
}

/// One segment's titles: `rust` in every other doc, `async` and `web`
/// sprinkled every 4th/5th doc so each stays df ≥ 2 in every shard, and
/// a unique filler token per doc keeps the corpus realistic.
fn segment_titles(segment: usize) -> Vec<String> {
    (0..DOCS_PER_SEGMENT)
        .map(|i| {
            let mut title = format!("filler{segment}x{i}");
            if i % 2 == 0 {
                title.push_str(" rust");
            }
            if i % 4 == 1 {
                title.push_str(" async");
            }
            if i % 5 == 3 {
                title.push_str(" web");
            }
            title
        })
        .collect()
}

fn title_batch(titles: &[String], schema: Arc<Schema>) -> RecordBatch {
    let arr: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    RecordBatch::try_new(schema, vec![arr]).expect("batch")
}

/// Two committed segments (each row-sharded into superfiles by the
/// 2-thread writer pool), so the query fans out across superfiles.
fn demo_two_superfiles() -> Supertable {
    let st = Supertable::create(options_title_only()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&title_batch(&segment_titles(0), schema.clone()))
        .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&title_batch(&segment_titles(1), schema))
        .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

/// Zero the fields the determinism contract exempts: kernel CPU is
/// measured time (varies run to run), and reranked rows are actual
/// execution counts that the deferred path's cold arm can legitimately
/// shift (they ARE deterministic at a fixed temperature, but this
/// helper serves the cross-temperature comparisons).
fn deterministic(mut stats: OpStats) -> OpStats {
    stats.kernel_cpu_ns = 0;
    stats.vector_rows_reranked = 0;
    stats
}

/// One BM25 query's full work stats, run inside a fresh scope, minting
/// the reader inside the scope (the pickup point).
fn scoped_fts_stats(st: &Supertable, query: &str) -> OpStats {
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", query, TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert!(!hits.is_empty(), "fixture query {query:?} must match");
    stats
}

/// Posting bytes for one BM25 query run inside a fresh scope.
fn scoped_query_bytes(st: &Supertable, query: &str) -> u64 {
    scoped_fts_stats(st, query).fts_postings_bytes
}

#[test]
fn a_scoped_bm25_query_reports_its_posting_bytes() {
    let st = demo_two_superfiles();
    let bytes = scoped_query_bytes(&st, "rust");
    assert!(
        bytes > 0,
        "a matching query indexes into posting bytes; got 0"
    );
}

#[test]
fn a_scoped_bm25_query_reports_its_planned_ranges() {
    let st = demo_two_superfiles();
    let (_, one_term) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    let (_, three_terms) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust async web", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert!(
        one_term.planned_read_ranges > 0,
        "a PFOR term requests its posting range"
    );
    assert!(
        three_terms.planned_read_ranges > one_term.planned_read_ranges,
        "three terms request more ranges than one (three {}, one {})",
        three_terms.planned_read_ranges,
        one_term.planned_read_ranges
    );
}

#[test]
fn fts_planned_ranges_pin_one_range_per_term_per_superfile() {
    // Every fixture term is PFOR (df >= 2) in every superfile, so the
    // plan is EXACTLY one posting range per term per superfile. An exact
    // pin: a double flush (2x), a missed superfile, or a phantom extra
    // range all fail an equality that the >0 assertions would pass.
    let st = demo_two_superfiles();
    let n_superfiles = st.reader().expect("reader").n_superfiles() as u64;
    assert!(
        n_superfiles >= 2,
        "fixture must fan out; got {n_superfiles}"
    );
    let (_, one_term) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    // Per superfile: one dictionary fetch + one PFOR posting range.
    assert_eq!(
        one_term.planned_read_ranges,
        2 * n_superfiles,
        "single term = dict + posting range per superfile"
    );
    let (_, three_terms) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust async web", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    // Per superfile: one dictionary fetch + three PFOR posting ranges.
    assert_eq!(
        three_terms.planned_read_ranges,
        4 * n_superfiles,
        "three PFOR terms = dict + three posting ranges per superfile"
    );
}

#[test]
fn fts_work_stats_repeat_identically_on_the_same_table_state() {
    // Named for what it guards. It used to claim the first run decodes
    // from a cold state and the repeat hits warm structures, but this
    // fixture has no cache-temperature axis at all: `demo_two_superfiles`
    // attaches no storage, so every published superfile's bytes sit in
    // the table's in-memory reader cache for its lifetime and both runs
    // are served from the same resident reader. What it really pins is
    // run-to-run repeatability across the full masked snapshot, which is
    // worth having — the genuine cold-open axis is covered by
    // `sql_work_stats_do_not_depend_on_reader_open_shape` for SQL and by
    // the reader-lifetime transposition in the vector sibling.
    let st = demo_two_superfiles();
    let first = deterministic(scoped_fts_stats(&st, "rust"));
    let second = deterministic(scoped_fts_stats(&st, "rust"));
    assert_eq!(first, second, "same plan, same table state, same work");
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
)]
fn a_scoped_bm25_query_reports_kernel_cpu() {
    // The thread-CPU clock (schedstat) advances at scheduler events, so a
    // single microsecond kernel can legitimately read zero; a batch of
    // queries crosses enough context switches that the cumulative
    // bracketed time must register.
    const KERNEL_CPU_BATCH: usize = 200;
    let st = demo_two_superfiles();
    let (_, stats) = with_op_stats(|| {
        for i in 0..KERNEL_CPU_BATCH {
            // Alternate the single-term shape (whose walk finishes inside
            // `prepare_clauses` and rides the `Done` result) with the
            // multi-term OR (bracketed at `run_prepared`), so both kernel
            // accounting paths contribute.
            let query = if i % 2 == 0 { "rust" } else { "rust async web" };
            st.reader()
                .expect("reader")
                .bm25_hits("title", query, TOP_K, BoolMode::Or)
                .expect("bm25");
        }
    });
    assert!(
        stats.kernel_cpu_ns > 0,
        "the bracketed FTS kernels report on-CPU time over {KERNEL_CPU_BATCH} queries; got 0"
    );
}

#[test]
fn more_clauses_index_more_posting_bytes() {
    let st = demo_two_superfiles();
    let narrow = scoped_query_bytes(&st, "async");
    let wide = scoped_query_bytes(&st, "rust async web");
    assert!(
        wide > narrow,
        "a three-term OR must index at least the one-term query's bytes \
         (wide {wide} vs narrow {narrow})"
    );
}

#[test]
fn a_reader_minted_outside_the_scope_records_nothing() {
    let st = demo_two_superfiles();
    // Mint first, then open the scope: the collector is picked up at
    // reader mint, so this query deliberately reports zero work.
    let reader = st.reader().expect("reader");
    let (hits, stats) = with_op_stats(|| {
        reader
            .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert!(!hits.is_empty());
    assert_eq!(
        stats.fts_postings_bytes, 0,
        "pickup happens at reader mint, not at query time"
    );
}

#[test]
fn an_inline_df1_term_plans_no_posting_range() {
    // "filler0x0" is unique to one doc: inline in the FST of its home
    // superfile (zero fetches planned), absent everywhere else. Adding
    // it to a query must not change the planned range count — a phantom
    // range per inline term was the old behavior.
    let st = demo_two_superfiles();
    let (_, base) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    let (_, with_inline) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust filler0x0", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert_eq!(
        with_inline.planned_read_ranges, base.planned_read_ranges,
        "an inline df=1 term plans no fetch, so it must add no range"
    );
    assert!(
        with_inline.fts_postings_bytes >= base.fts_postings_bytes,
        "byte counts never shrink when a term is added"
    );
}

#[test]
fn a_scoped_token_match_reports_posting_work() {
    let st = demo_two_superfiles();
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .token_match("title", "rust async", BoolMode::Or)
            .expect("token_match")
    });
    assert!(!hits.is_empty());
    assert!(
        stats.fts_postings_bytes > 0,
        "an unranked match walks posting bytes; got 0"
    );
    assert!(
        stats.planned_read_ranges > 0,
        "each PFOR term is one planned range; got 0"
    );
}

#[test]
fn a_scoped_count_reports_posting_work() {
    let st = demo_two_superfiles();
    // Single term: the df fast path reads one header range per superfile.
    let (n_single, single) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .count("title", "rust", BoolMode::Or)
            .expect("count")
    });
    assert!(n_single > 0);
    assert!(
        single.planned_read_ranges > 0 && single.fts_postings_bytes > 0,
        "the df fast path reads real header ranges (ranges {}, bytes {})",
        single.planned_read_ranges,
        single.fts_postings_bytes
    );
    // Multi-term: the counting walk indexes full posting lists.
    let (n_multi, multi) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .count("title", "rust async", BoolMode::Or)
            .expect("count")
    });
    assert!(n_multi > 0);
    assert!(
        multi.fts_postings_bytes > single.fts_postings_bytes,
        "the counting walk indexes posting lists, not just headers"
    );
}

#[test]
fn a_scoped_exact_match_reports_posting_work() {
    let st = demo_two_superfiles();
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .exact_match("title", "filler0x0 rust")
            .expect("exact_match")
    });
    // The exact string exists as doc 0 of segment 0.
    assert!(!hits.is_empty(), "fixture exact value must match");
    assert!(
        stats.fts_postings_bytes > 0,
        "the prune pass walks posting bytes; got 0"
    );
}

#[test]
fn a_scoped_prefix_search_reports_posting_work() {
    // Prefix expansion used to flush nothing at all — a prefix widening
    // to many terms billed as zero FTS work.
    let st = demo_two_superfiles();
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_search_prefix("title", "ru", TOP_K)
            .expect("prefix search")
    });
    assert!(!hits.is_empty(), "fixture prefix must match");
    assert!(
        stats.fts_postings_bytes > 0,
        "the prefix expansion walks posting bytes; got 0"
    );
    assert!(
        stats.planned_read_ranges > 0,
        "the prefix expansion plans posting ranges; got 0"
    );
}

#[test]
fn a_scalar_projection_reports_materialized_rows() {
    // Materializing named columns decodes stored rows — real work the
    // counters must carry; the id+score default decodes nothing.
    let st = demo_two_superfiles();
    let (batches, projected) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                TOP_K,
                BoolMode::Or,
                Bm25Stats::PerSuperfile,
                Some(&["title"]),
            )
            .expect("projected search")
    });
    let returned: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    assert!(returned > 0);
    assert_eq!(
        projected.rows_materialized, returned,
        "every returned row was decoded from stored columns"
    );
    let (_, bare) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_search(
                "title",
                "rust",
                TOP_K,
                BoolMode::Or,
                Bm25Stats::PerSuperfile,
                None,
            )
            .expect("bare search")
    });
    assert_eq!(
        bare.rows_materialized, 0,
        "the id+score fast path decodes no stored columns"
    );
}

// ---- Vector: the drained (hidden-index) deferred-rerank path ----

/// `default_vector_config` is dim=16.
const DIM: usize = 16;
/// Rows in the vector fixture — enough for a non-degenerate drain.
const VECTOR_ROWS: usize = 64;
/// Top-k for the vector work-stats queries.
const VECTOR_K: usize = 4;
/// Probe width > 1 so the drained table takes the deferred-rerank path
/// (immediate-rerank widths report no scan tallies yet).
const VECTOR_NPROBE: usize = 2;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 13;

/// Deterministic non-degenerate vector for row `i`.
fn row_vec(i: usize) -> Vec<f32> {
    (0..DIM)
        .map(|d| ((i * 31 + d * 17 + 7) % 97) as f32 / 97.0 + 0.05)
        .collect()
}

fn vector_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

fn vector_batch(schema: Arc<Schema>) -> RecordBatch {
    let mut flat = Vec::<f32>::with_capacity(VECTOR_ROWS * DIM);
    let mut titles = Vec::with_capacity(VECTOR_ROWS);
    for i in 0..VECTOR_ROWS {
        flat.extend(row_vec(i));
        // "vec" appears in every row (df >= 2 ⇒ PFOR postings), so a
        // text predicate on it exercises real posting-walk work.
        titles.push(format!("vec row{i:03}"));
    }
    let fsl = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        DIM as i32,
        Arc::new(Float32Array::from(flat)) as ArrayRef,
        None,
    )
    .expect("FSL");
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(titles)) as ArrayRef,
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// A committed + drained vector table (hidden cell index built), so
/// queries run the deferred-rerank serving path the counters cover.
fn drained_vector_table(dir: &TempDir) -> Supertable {
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(vector_options().with_storage(storage)).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&vector_batch(schema)).expect("append");
    w.commit().expect("commit");
    drop(w);
    st.drain_vectors_to_cells_sync().expect("drain");
    st
}

/// One scoped vector query over the drained table.
fn scoped_vector_stats(st: &Supertable) -> OpStats {
    let query = row_vec(3);
    let (hits, stats) = with_op_stats(|| {
        st.vector_search(
            "emb",
            &query,
            VECTOR_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            None,
            None,
        )
        .expect("vector search")
    });
    assert!(!hits.is_empty(), "fixture vector query must match");
    stats
}

#[test]
fn a_scoped_vector_query_reports_scan_and_rerank_work() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let stats = scoped_vector_stats(&st);
    assert!(
        stats.vector_cells_scanned >= VECTOR_NPROBE as u64,
        "a width-{VECTOR_NPROBE} probe scans at least that many cells; got {}",
        stats.vector_cells_scanned
    );
    assert!(
        stats.vector_candidates_scanned >= stats.vector_cells_scanned,
        "every scanned cell holds at least one code (candidates {}, cells {})",
        stats.vector_candidates_scanned,
        stats.vector_cells_scanned
    );
    assert!(
        stats.vector_rows_reranked > 0,
        "the global shortlist reranks a non-empty winner set"
    );
    assert!(
        stats.vector_rows_reranked <= stats.vector_candidates_scanned,
        "rerank rows are shortlist survivors of the scanned candidates"
    );
    assert!(
        stats.planned_read_ranges >= 2 * stats.vector_cells_scanned,
        "each scanned cell requests its cluster index plus at least one \
         prefix/block range (ranges {}, cells {})",
        stats.planned_read_ranges,
        stats.vector_cells_scanned
    );
    assert!(
        stats.planned_read_ranges < stats.vector_rows_reranked + 64 * stats.vector_cells_scanned,
        "the range counter stays request-shaped: rerank rows must not be \
         folded into it (ranges {}, rows {})",
        stats.planned_read_ranges,
        stats.vector_rows_reranked
    );
}

/// A committed + drained L2Sq vector table whose rerank codec is the
/// non-cosine default, `Sq16Adaptive`. The other vector fixtures build the
/// cosine/`Fp32` path, so this is the only op-stats coverage of the new
/// default codec's rerank kernel.
fn drained_sq16_adaptive_table(dir: &TempDir) -> Supertable {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let vector = VectorConfig {
        metric: Metric::L2Sq,
        ..default_vector_config("emb", VECTOR_ROT_SEED).with_rerank_codec(RerankCodec::Sq16Adaptive)
    };
    let opts = SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
            positions: false,
        }],
        vec![vector],
        Some(default_tokenizer()),
    )
    .expect("valid options");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(opts.with_storage(storage)).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&vector_batch(schema)).expect("append");
    w.commit().expect("commit");
    drop(w);
    st.drain_vectors_to_cells_sync().expect("drain");
    st
}

/// #550 kernel-CPU metering must fire for the new default codec, not only the
/// residual/cosine paths: a drained `Sq16Adaptive` (L2Sq) table reports a
/// non-zero rerank kernel CPU time. Regression guard for a rebase or refactor
/// that drops the metering from the generic scorer's Sq16Adaptive arms.
#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
)]
fn a_scoped_sq16_adaptive_vector_query_reports_kernel_cpu() {
    // Same thread-CPU-clock granularity handling as the BM25 kernel-CPU test: a
    // single tiny rerank can legitimately read zero, so batch enough queries
    // that the cumulative bracketed time registers.
    const KERNEL_CPU_BATCH: usize = 200;
    let dir = TempDir::new().expect("tempdir");
    let st = drained_sq16_adaptive_table(&dir);
    let query = row_vec(3);
    let (_, stats) = with_op_stats(|| {
        for _ in 0..KERNEL_CPU_BATCH {
            st.vector_search(
                "emb",
                &query,
                VECTOR_K,
                VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                None,
                None,
            )
            .expect("vector search");
        }
    });
    assert!(
        stats.vector_rows_reranked > 0,
        "the Sq16Adaptive shortlist must rerank a non-empty winner set"
    );
    assert!(
        stats.kernel_cpu_ns > 0,
        "the Sq16Adaptive rerank kernel reports on-CPU time over {KERNEL_CPU_BATCH} queries; got 0"
    );
}

#[test]
fn a_full_width_probe_scans_every_row_exactly_once() {
    // nprobe >= the hidden cell count chooses every cluster, so the scan
    // estimates every stored code exactly once: candidates == rows in
    // the table. An exact pin — any double flush would report 2x.
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let (hits, stats) = with_op_stats(|| {
        st.vector_search(
            "emb",
            &query,
            VECTOR_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_ROWS),
            None,
            None,
        )
        .expect("vector search")
    });
    assert!(!hits.is_empty());
    assert_eq!(
        stats.vector_candidates_scanned, VECTOR_ROWS as u64,
        "a full-width probe estimates each stored code exactly once"
    );
}

#[test]
fn a_filtered_vector_query_meters_its_predicate_leg() {
    // Filtered kNN first resolves the text predicate on the user table
    // (posting walks), then ranks among matching rows. Both legs are
    // real work and both must land in the counters — the predicate leg
    // used to be invisible.
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let (hits, stats) = with_op_stats(|| {
        st.vector_search(
            "emb",
            &query,
            VECTOR_K,
            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
            Some(VectorFilter {
                column: "title",
                query: "vec",
                mode: BoolMode::Or,
            }),
            None,
        )
        .expect("filtered vector search")
    });
    assert!(!hits.is_empty(), "the fixture predicate matches every row");
    assert!(
        stats.fts_postings_bytes > 0,
        "the predicate leg walks posting bytes; got 0"
    );
    assert!(
        stats.vector_candidates_scanned > 0,
        "the vector leg still scans candidates; got 0"
    );
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
)]
fn a_scoped_vector_query_reports_kernel_cpu() {
    // Same schedstat-resolution caveat as the FTS kernel test: batch the
    // queries so the bracketed scan + rerank sections cross scheduler
    // ticks.
    const VECTOR_KERNEL_BATCH: usize = 200;
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let (_, stats) = with_op_stats(|| {
        for _ in 0..VECTOR_KERNEL_BATCH {
            st.vector_search(
                "emb",
                &query,
                VECTOR_K,
                VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                None,
                None,
            )
            .expect("vector search");
        }
    });
    assert!(
        stats.kernel_cpu_ns > 0,
        "the bracketed vector kernels report on-CPU time over {VECTOR_KERNEL_BATCH} queries; got 0"
    );
}

#[test]
fn vector_work_stats_are_deterministic_across_cache_temperature() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let cold = deterministic(scoped_vector_stats(&st));
    let warm = deterministic(scoped_vector_stats(&st));
    assert_eq!(cold, warm, "same plan, same table state, same work");
}

/// The reader-level SQL surface (test/bench only) meters CPU through the
/// plan root, but its session context is cached and deliberately carries
/// no collector — so the scan-level wrappers are inert there and the
/// row count has to come from DataFusion's own leaf metrics. Without
/// that harvest the benches driving this surface got a CPU number with no
/// row denominator to divide it by.
#[test]
fn the_reader_level_sql_surface_reports_materialized_rows() {
    let st = demo_two_superfiles();
    let (batches, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .query_sql("SELECT title FROM supertable")
            .expect("reader-level query_sql")
    });
    let returned: u64 = batches.iter().map(|b| b.num_rows() as u64).sum();
    assert!(returned > 0, "the fixture must return rows");
    assert_eq!(
        stats.rows_materialized, returned,
        "every row the scan decoded must reach the counter"
    );
}

// ---- SQL: the per-query channel through the catalog `query_sql` path ----

/// Rows in the SQL fixture.
const SQL_ROWS: usize = 64;

/// A storage-backed catalog connection with one FTS-indexed table whose
/// scans and search TVFs exercise the per-query SQL channel.
fn sql_fixture(dir: &TempDir) -> Connection {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
    ]));
    let docs = db
        .create_table("docs", schema.clone(), IndexSpec::new().fts("title"))
        .expect("create_table");
    let titles: Vec<String> = (0..SQL_ROWS)
        .map(|i| {
            let mut t = format!("filler{i}");
            if i % 2 == 0 {
                t.push_str(" rust");
            }
            t
        })
        .collect();
    let title_arr: ArrayRef = Arc::new(LargeStringArray::from(
        titles.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    let ratings: ArrayRef = Arc::new(Int64Array::from((0..SQL_ROWS as i64).collect::<Vec<_>>()));
    let batch = RecordBatch::try_new(schema, vec![title_arr, ratings]).expect("batch");
    docs.append(&batch).expect("append");
    db
}

/// One scoped SQL statement's work stats.
fn scoped_sql_stats(db: &Connection, sql: &str) -> OpStats {
    let (batches, stats) = with_op_stats(|| db.query_sql(sql).expect("query_sql"));
    assert!(!batches.is_empty(), "fixture SQL {sql:?} must return");
    stats
}

#[test]
fn a_scoped_sql_scan_reports_page_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let db = sql_fixture(&dir);
    // A row-returning range scan cannot fold to manifest statistics, so it
    // must decode Parquet pages through the DataFusion store.
    let stats = scoped_sql_stats(&db, "SELECT rating FROM docs WHERE rating > 5");
    assert!(
        stats.sql_page_bytes > 0,
        "a scan-backed SQL query requests Parquet bytes; got 0"
    );
    assert!(
        stats.planned_read_ranges > 0,
        "each Parquet request is a planned range"
    );
    assert!(
        stats.rows_materialized > 0,
        "the scan's decoded rows come from DataFusion's own metrics; got 0"
    );
}

/// A whole-table aggregate is answerable from statistics the provider
/// attaches to the scan, so the planner folds it to a constant and never
/// opens a page. Measuring must not change that. `MeteredExec` sits
/// directly between the aggregate and the scan, and an `ExecutionPlan`
/// wrapper that takes the trait defaults reports unknown statistics —
/// which silently turns an O(1) manifest read into a full columnar scan
/// that the customer is then billed for. This asserts the billing-visible
/// consequence rather than a plan string, so it holds whichever rule does
/// the folding.
#[test]
fn a_whole_table_aggregate_folds_instead_of_scanning() {
    let dir = TempDir::new().expect("tempdir");
    let db = sql_fixture(&dir);
    for sql in [
        "SELECT COUNT(*) FROM docs",
        "SELECT MIN(rating), MAX(rating) FROM docs",
    ] {
        let stats = scoped_sql_stats(&db, sql);
        assert_eq!(
            stats.sql_page_bytes, 0,
            "{sql} folds from statistics; it must not read Parquet pages"
        );
        assert_eq!(
            stats.planned_read_ranges, 0,
            "{sql} folds from statistics; it must not plan a read range"
        );
        assert_eq!(
            stats.rows_materialized, 0,
            "{sql} folds from statistics; it must not decode rows"
        );
    }
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "per-thread CPU clock is Linux procfs (schedstat); off Linux kernel_cpu_ns is always 0"
)]
fn a_scoped_sql_scan_reports_kernel_cpu() {
    // SQL CPU is bracketed per scan poll on the thread clock, like every
    // other kernel — not DataFusion's `elapsed_compute`, which is wall
    // time and omits Parquet decode. Same granularity handling as the
    // BM25 and vector kernel-CPU tests: schedstat advances at scheduler
    // events, so one sub-tick scan can legitimately read zero and only a
    // batch is guaranteed to register.
    const KERNEL_CPU_BATCH: usize = 200;
    let dir = TempDir::new().expect("tempdir");
    let db = sql_fixture(&dir);
    let mut total = 0u64;
    for _ in 0..KERNEL_CPU_BATCH {
        total += scoped_sql_stats(&db, "SELECT rating FROM docs WHERE rating > 5").kernel_cpu_ns;
    }
    assert!(
        total > 0,
        "the bracketed SQL scan reports on-CPU time over {KERNEL_CPU_BATCH} queries; got 0"
    );
}

#[test]
fn sql_work_stats_are_deterministic_across_cache_temperature() {
    let dir = TempDir::new().expect("tempdir");
    let db = sql_fixture(&dir);
    let sql = "SELECT rating FROM docs WHERE rating > 5";
    let cold = deterministic(scoped_sql_stats(&db, sql));
    let warm = deterministic(scoped_sql_stats(&db, sql));
    assert_eq!(cold, warm, "same plan, same table state, same work");
}

#[test]
fn sql_work_stats_do_not_depend_on_reader_open_shape() {
    // A predicated scan makes DataFusion consult the Parquet page index.
    // An eagerly-opened reader's footer parse already carries it; a cold
    // disk-cache open serves the query through a lazy reader whose
    // open-time parse is footer-only. The provider must serve DataFusion
    // an index-complete parse loaded through the reader's own byte
    // source — otherwise the opener fetches the index bytes through the
    // metered store on the lazy shape only, and the same query prices
    // differently depending on how its readers happened to be opened.
    //
    // The writer connection populates only its own in-memory reader
    // tier, so the cache-backed connection's first query really takes
    // the cold lazy-open path.
    let dir = TempDir::new().expect("tempdir");
    let cache = TempDir::new().expect("cache tempdir");
    let sql = "SELECT rating FROM docs WHERE rating > 5";
    let eager = {
        let db = sql_fixture(&dir);
        deterministic(scoped_sql_stats(&db, sql))
    };
    let lazy_db = connect_with(
        dir.path().to_str().expect("utf-8 path"),
        ConnectOptions::new().with_cache_dir(cache.path()),
    )
    .expect("connect with cold disk cache");
    let lazy_cold = deterministic(scoped_sql_stats(&lazy_db, sql));
    let lazy_warm = deterministic(scoped_sql_stats(&lazy_db, sql));
    assert_eq!(
        lazy_cold, lazy_warm,
        "same plan, same table state, same work"
    );
    assert_eq!(
        lazy_cold, eager,
        "reader open shape (cold disk cache vs eager) must not change reported work"
    );
}

/// A storage-backed catalog connection whose table also carries a
/// drained vector index, so SQL vector TVFs run the hidden-index path.
fn sql_vector_fixture(dir: &TempDir) -> Connection {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let docs = db
        .create_table(
            "docs",
            schema.clone(),
            IndexSpec::new()
                .fts("title")
                .vector("emb", DIM, Metric::L2Sq),
        )
        .expect("create_table");
    docs.append(&vector_batch(schema)).expect("append");
    docs.local_handle()
        .drain_vectors_to_cells_sync()
        .expect("drain");
    db
}

#[test]
fn a_scope_minted_reader_meters_hidden_work_from_any_thread() {
    // The collector is picked up at reader mint and must TRAVEL with the
    // reader — never be re-read from a thread-local mid-query. The
    // drained path mints the hidden vector-index reader mid-query, and
    // real drivers (DataFusion partitions, spawned fan-out bodies) poll
    // kernels on runtime threads where the caller's scope is invisible.
    // Regression: run the query on a scope-less thread; the hidden mint
    // used to consult that thread's empty slot and report zero vector
    // work for exactly the query class the token meter is anchored on.
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let (hits, stats) = with_op_stats(|| {
        let reader = st.reader().expect("reader minted inside the scope");
        thread::scope(|scope| {
            scope
                .spawn(|| {
                    reader
                        .vector_hits(
                            "emb",
                            &query,
                            VECTOR_K,
                            VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                            None,
                        )
                        .expect("vector hits off-thread")
                })
                .join()
                .expect("query thread")
        })
    });
    assert!(!hits.is_empty(), "fixture vector query must match");
    assert!(
        stats.vector_cells_scanned > 0,
        "the hidden-index leg meters cells from a scope-less thread; got 0"
    );
    assert!(
        stats.vector_candidates_scanned > 0,
        "the hidden-index leg meters candidates from a scope-less thread; got 0"
    );
    assert!(
        stats.planned_read_ranges > 0,
        "the hidden-index leg meters planned ranges from a scope-less thread; got 0"
    );
}

#[test]
fn a_vector_tvf_inside_sql_reports_vector_work() {
    // The hidden-index reader is minted MID-QUERY, on a DataFusion
    // runtime thread where the caller's `with_op_stats` thread-local is
    // invisible — the collector must arrive by inheritance from the
    // TVF's reader. Regression: this path used to report zero vector
    // work for exactly the query class the token is anchored on.
    let dir = TempDir::new().expect("tempdir");
    let db = sql_vector_fixture(&dir);
    let csv = row_vec(3)
        .iter()
        .map(f32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let stats = scoped_sql_stats(
        &db,
        &format!("SELECT _id, score FROM vector_search('docs', 'emb', '{csv}', {VECTOR_K})"),
    );
    assert!(
        stats.vector_cells_scanned > 0,
        "the hidden-index leg inside SQL scans cells; got 0"
    );
    assert!(
        stats.vector_candidates_scanned > 0,
        "the hidden-index leg inside SQL estimates candidates; got 0"
    );
    assert!(
        stats.planned_read_ranges > 0,
        "the hidden-index leg inside SQL plans ranges; got 0"
    );
}

#[test]
fn a_search_tvf_inside_sql_reports_fts_work() {
    let dir = TempDir::new().expect("tempdir");
    let db = sql_fixture(&dir);
    // The TVF resolves its reader on a runtime thread; the collector must
    // arrive via the registration-time capture, not the thread-local.
    let stats = scoped_sql_stats(
        &db,
        "SELECT _id, score FROM bm25_search('docs', 'title', 'rust', 10)",
    );
    assert!(
        stats.fts_postings_bytes > 0,
        "the BM25 leg inside SQL indexes posting bytes; got 0"
    );
}

/// Projecting only `_id` must cost no more than the engine-native
/// (`None`) result: both columns are produced by the search wave — ids
/// stamped on the hits, scores from the kernel — so neither needs a
/// placement resolve or a Parquet decode. Regression for the fast path
/// that matched only the exact `[_id, score]` pair and sent a bare
/// `["_id"]` down the scalar-projection path, which resolves placements
/// (a whole-`_id`-column read per gapped superfile on first touch) and
/// then decodes rows. `rows_materialized` is the invariant signal: the
/// native path decodes nothing, so anything above 0 here means the
/// query fell off the fast path.
#[test]
fn id_only_projection_costs_no_more_than_the_native_result() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let search = |projection: Option<&[&str]>| {
        let (hits, stats) = with_op_stats(|| {
            st.vector_search(
                "emb",
                &query,
                VECTOR_K,
                VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                None,
                projection,
            )
            .expect("vector search")
        });
        assert!(!hits.is_empty(), "fixture vector query must match");
        stats
    };

    let native = search(None);
    let id_only = search(Some(&["_id"]));
    let id_and_score = search(Some(&["_id", "score"]));
    let score_only = search(Some(&["score"]));

    assert_eq!(
        native.rows_materialized, 0,
        "the engine-native result decodes no stored rows"
    );
    for (label, stats) in [
        ("_id", &id_only),
        ("_id + score", &id_and_score),
        ("score", &score_only),
    ] {
        assert_eq!(
            stats.rows_materialized, 0,
            "projection [{label}] must stay on the id/score fast path \
             (decoded {} rows)",
            stats.rows_materialized
        );
        assert_eq!(
            stats.planned_read_ranges, native.planned_read_ranges,
            "projection [{label}] must plan the same reads as the native result"
        );
    }
}

/// The gapped-placement memo must not pin connection-budget bytes. The
/// budget gates MANDATORY work — ingest and compaction both hard-fail
/// when refused — so a discretionary, rebuildable read cache that holds
/// bytes for as long as its superfile stays live can push those over the
/// ceiling and keep them there: the only thing that evicts an entry is
/// its superfile being superseded, which is what compaction does, which
/// would be the operation denied.
///
/// Measured as a DELTA against a warmed native query, because the
/// transposed-code cache legitimately pins bytes for its reader's
/// lifetime (`TransposedCluster::_reservation`) — this asserts only that
/// building the placement memo adds nothing permanent on top.
#[test]
fn placement_memo_releases_its_budget_bytes() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let budget = st.options().connection_budget();
    let query = row_vec(3);
    let search = |projection: Option<&[&str]>| {
        let hits = st
            .vector_search(
                "emb",
                &query,
                VECTOR_K,
                VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE),
                None,
                projection,
            )
            .expect("vector search");
        assert!(!hits.is_empty(), "fixture vector query must match");
    };

    // Warm every reader-lifetime cache the query path legitimately pins,
    // so the only new resident state below is the placement memo.
    search(None);
    search(None);
    let warmed = budget.used_bytes();
    assert!(
        budget.peak() > 0,
        "the query must have exercised the budget at all"
    );

    // A user-column projection forces placement resolution over the
    // drained (gapped) superfiles — this is what builds the memo.
    search(Some(&["title"]));

    assert_eq!(
        budget.used_bytes(),
        warmed,
        "building the placement memo must leave no reserved bytes behind; \
         {} held beyond the warmed baseline",
        budget.used_bytes().saturating_sub(warmed)
    );

    // Liveness: mandatory work still reserves after the memo is warm.
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&vector_batch(schema)).expect("append after memo");
    w.commit().expect("commit after memo");
}

/// A FILTERED vector query projecting `_id`/`score` must not return
/// hidden-deleted rows — and this pins WHY it doesn't.
///
/// The fast path returns straight from search-wave stamps, skipping
/// `user_placement_for_scalar_resolve`, which is where the identity-level
/// delete filter lives. Unlike the global route, the filtered route has no
/// retain of its own, so the omission looks unsafe. It is not, for a reason
/// worth pinning: the filtered route derives its hidden allow-set from the
/// USER-table allow bitmaps (`stable_ids_from_user_allow_async`), and those
/// have already had `subtract_tombstones` applied in
/// `fanout_candidate_bitmaps`. A deleted row is therefore dropped at the
/// predicate leg and never becomes a candidate id, so no deleted row can
/// reach the fast path in the first place.
///
/// That upstream subtraction is load-bearing and invisible from the
/// projection code. This test fails if it is ever removed or bypassed.
#[test]
fn a_filtered_id_score_query_excludes_deleted_rows() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let query = row_vec(3);
    let opts = VectorSearchOptions::new().with_nprobe(VECTOR_NPROBE);
    let filter = || VectorFilter {
        column: "title",
        query: "vec",
        mode: BoolMode::Or,
    };

    // Identify the top-k by BOTH id and title in one pass: the title drives
    // the delete predicate, the id is what the fast path must stop serving.
    // This arm projects a scalar column, so it goes through placement — the
    // arm under test is the `["_id", "score"]` one below.
    let before = st
        .vector_search(
            "emb",
            &query,
            VECTOR_K,
            opts,
            Some(filter()),
            Some(&["_id", "title", "score"]),
        )
        .expect("filtered search before delete");
    let (ids_before, titles) = ids_and_titles(&before);
    assert_eq!(
        ids_before.len(),
        VECTOR_K,
        "fixture must fill k before any delete"
    );

    let preds: Vec<Expr> = titles.iter().map(|t| lit(t.as_str())).collect();
    let stats = st
        .delete(col("title").in_list(preds, false))
        .expect("delete");
    assert_eq!(
        stats.n_tombstoned() as usize,
        ids_before.len(),
        "every top-k row must tombstone"
    );

    // The arm under test: `["_id", "score"]` takes the fast path.
    let after = st
        .vector_search(
            "emb",
            &query,
            VECTOR_K,
            opts,
            Some(filter()),
            Some(&["_id", "score"]),
        )
        .expect("filtered search after delete");
    let (ids_after, _) = ids_and_titles(&after);
    for id in &ids_after {
        assert!(
            !ids_before.contains(id),
            "filtered id/score fast path returned deleted _id {id}"
        );
    }
    assert_eq!(
        ids_after.len(),
        VECTOR_K,
        "filtered result underflowed instead of backfilling past tombstones"
    );
}

/// `_id`s, and titles when the projection carried them.
fn ids_and_titles(batches: &[RecordBatch]) -> (Vec<i128>, Vec<String>) {
    let mut ids = Vec::new();
    let mut titles = Vec::new();
    for b in batches {
        let id_col = b
            .column_by_name("_id")
            .expect("_id column")
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("_id is decimal128");
        ids.extend((0..id_col.len()).map(|i| id_col.value(i)));
        if let Some(col) = b.column_by_name("title") {
            let t = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("title is large utf8");
            titles.extend((0..t.len()).map(|i| t.value(i).to_owned()));
        }
    }
    (ids, titles)
}
