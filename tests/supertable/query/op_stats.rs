// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query work stats over the sync BM25 surface: a query wrapped in
//! [`with_op_stats`] reports the posting bytes its kernels indexed into,
//! deterministically — the same query against the same committed corpus
//! reports the same number on a cold first run and a warm repeat, and a
//! reader minted outside a scope records nothing.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
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
}

#[test]
fn vector_work_stats_are_deterministic_across_cache_temperature() {
    let dir = TempDir::new().expect("tempdir");
    let st = drained_vector_table(&dir);
    let cold = scoped_vector_stats(&st);
    let warm = scoped_vector_stats(&st);
    assert_eq!(cold, warm, "same plan, same table state, same work");
}
