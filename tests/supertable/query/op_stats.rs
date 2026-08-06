// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query work stats over the sync BM25 surface: a query wrapped in
//! [`with_op_stats`] reports the posting bytes its kernels indexed into,
//! deterministically — the same query against the same committed corpus
//! reports the same number on a cold first run and a warm repeat, and a
//! reader minted outside a scope records nothing.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;
use std::thread;

use arrow_array::{
    ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    Connection, IndexSpec, Metric, connect,
    runtime_metrics::op_stats::{OpStats, with_op_stats},
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{builder::FtsConfig, fts::reader::BoolMode},
    supertable::{Supertable, SupertableOptions, query::vector::VectorSearchOptions},
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

/// Zero the one measured-time field so determinism assertions compare only
/// the plan counts (kernel CPU legitimately varies run to run).
fn deterministic(mut stats: OpStats) -> OpStats {
    stats.kernel_cpu_ns = 0;
    stats
}

/// Posting bytes for one BM25 query run inside a fresh scope, minting the
/// reader inside the scope (the pickup point).
fn scoped_query_bytes(st: &Supertable, query: &str) -> u64 {
    let (hits, stats) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", query, TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert!(!hits.is_empty(), "fixture query {query:?} must match");
    stats.fts_postings_bytes
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
    assert_eq!(
        one_term.planned_read_ranges, n_superfiles,
        "one PFOR term = one posting range per superfile"
    );
    let (_, three_terms) = with_op_stats(|| {
        st.reader()
            .expect("reader")
            .bm25_hits("title", "rust async web", TOP_K, BoolMode::Or)
            .expect("bm25")
    });
    assert_eq!(
        three_terms.planned_read_ranges,
        3 * n_superfiles,
        "three PFOR terms = three posting ranges per superfile"
    );
}

#[test]
fn work_stats_are_deterministic_across_cache_temperature() {
    // The first run decodes from a cold state, the repeat hits every
    // warm structure — the whole point of the counter is that the
    // reported work is identical either way.
    let st = demo_two_superfiles();
    let cold = scoped_query_bytes(&st, "rust");
    let warm = scoped_query_bytes(&st, "rust");
    assert_eq!(cold, warm, "same plan, same table state, same work");
}

#[test]
fn a_scoped_bm25_query_reports_kernel_cpu() {
    // The thread-CPU clock (schedstat) advances at scheduler events, so a
    // single microsecond kernel can legitimately read zero; a batch of
    // queries crosses enough context switches that the cumulative
    // bracketed time must register.
    const KERNEL_CPU_BATCH: usize = 200;
    let st = demo_two_superfiles();
    let (_, stats) = with_op_stats(|| {
        for _ in 0..KERNEL_CPU_BATCH {
            st.reader()
                .expect("reader")
                .bm25_hits("title", "rust async web", TOP_K, BoolMode::Or)
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
        titles.push(format!("row{i:03}"));
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
        stats.planned_read_ranges >= stats.vector_cells_scanned + stats.vector_rows_reranked,
        "each scanned cell requests at least its cluster index and each          reranked row its survivor range (ranges {}, cells {}, rows {})",
        stats.planned_read_ranges,
        stats.vector_cells_scanned,
        stats.vector_rows_reranked
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
fn vector_work_stats_are_deterministic_across_cache_temperature() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let cold = deterministic(scoped_vector_stats(&st));
    let warm = deterministic(scoped_vector_stats(&st));
    assert_eq!(cold, warm, "same plan, same table state, same work");
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
