// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Per-query work stats over the sync BM25 surface: a query wrapped in
//! [`with_op_stats`] reports the posting bytes its kernels indexed into,
//! deterministically — the same query against the same committed corpus
//! reports the same number on a cold first run and a warm repeat, and a
//! reader minted outside a scope records nothing.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{ArrayRef, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    runtime_metrics::op_stats::with_op_stats,
    superfile::{builder::FtsConfig, fts::reader::BoolMode},
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_tokenizer,
};

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
