// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Breadth coverage for the SQL query surface exposed by
//! [`Supertable::query_sql`](infino::supertable::Supertable).
//!
//! The other `query/*` files pin behavioural contracts of individual
//! TVFs (RRF fusion, union semantics, covered aggregates). This file
//! deliberately walks the *plan* surface end-to-end instead: it runs
//! `EXPLAIN` (which formats every physical node via `DisplayAs`),
//! `SELECT *` (the scalar-materialization path), `ORDER BY score` +
//! `LIMIT` (plan rewrites that swap children), and plain base-table
//! aggregates — across all five search TVFs plus the base table. The
//! goal is to exercise the custom `ExecutionPlan` / `TableProvider`
//! wiring (display, schema, statistics, execute) the way a SQL consumer
//! drives it, not to re-assert per-TVF ranking math.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::builder::FtsConfig,
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config},
};

/// `default_vector_config` is dim=16, cosine, n_cent=4.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 11;
/// Docs per commit; two commits keep the corpus multi-superfile so the
/// plans actually fan out.
const DOCS_PER_COMMIT: usize = 8;
/// Top-k used by the TVF surface walks. Larger than the corpus so no
/// query truncates away a node's work.
const SURFACE_TOP_K: usize = 32;

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Schema `[title (FTS), rating (scalar), emb (vector)]`. The vector
/// column is stripped from the SQL scalar schema at commit, so SQL only
/// ever sees `_id`, `title`, `rating`.
fn options_title_rating_emb() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
        Field::new("emb", fixed_list_f32(DIM), false),
    ]));
    SupertableOptions::new(
        schema,
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", VECTOR_ROT_SEED)],
        Some(default_tokenizer()),
    )
    .expect("valid options")
}

/// Doc `i` (within the batch) gets `titles[i]`, rating `base + i`, and a
/// one-hot embedding at global dim `base + i`.
fn build_batch(titles: &[&str], base: usize, schema: Arc<Schema>) -> RecordBatch {
    let n = titles.len();
    let title_arr = LargeStringArray::from(titles.to_vec());
    let ratings: Vec<i64> = (0..n).map(|i| (base + i) as i64).collect();
    let mut flat = Vec::<f32>::with_capacity(n * DIM);
    for i in 0..n {
        let active = (base + i) % DIM;
        for d in 0..DIM {
            flat.push(if d == active { 1.0 } else { 0.0 });
        }
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
            Arc::new(title_arr),
            Arc::new(Int64Array::from(ratings)),
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Two-superfile corpus (docs 0-7, then 8-15). `rust` recurs across both
/// superfiles; doc 0's title is unique on `async`.
fn demo_table() -> Supertable {
    let st = Supertable::create(options_title_rating_emb()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(
        &[
            "rust async",
            "python data",
            "java spring",
            "go rust",
            "ruby rails",
            "scala akka",
            "kotlin flow",
            "rust systems",
        ],
        0,
        schema.clone(),
    ))
    .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&build_batch(
        &[
            "swift ui",
            "rust web",
            "elixir otp",
            "haskell lazy",
            "rust embedded",
            "perl regex",
            "lua script",
            "rust async runtime",
        ],
        DOCS_PER_COMMIT,
        schema.clone(),
    ))
    .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

/// One-hot query vector at `active`, as the CSV literal the vector TVFs
/// parse.
fn csv_one_hot(active: usize) -> String {
    (0..DIM)
        .map(|d| if d == active { "1" } else { "0" })
        .collect::<Vec<_>>()
        .join(",")
}

/// Flatten an `EXPLAIN` result's string columns into one blob.
fn explain_text(st: &Supertable, sql: &str) -> String {
    let batches = st
        .reader()
        .query_sql(&format!("EXPLAIN {sql}"))
        .expect("explain");
    let mut out = String::new();
    for batch in &batches {
        for column in batch.columns() {
            if let Some(strings) = column.as_any().downcast_ref::<arrow_array::StringArray>() {
                for i in 0..strings.len() {
                    if !strings.is_null(i) {
                        out.push_str(strings.value(i));
                        out.push('\n');
                    }
                }
            }
        }
    }
    out
}

fn row_count(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Every search TVF must format a physical plan via `EXPLAIN` without
/// erroring. This drives the `DisplayAs::fmt_as` impl on each custom
/// `ExecutionPlan`, which the behavioural tests never reach.
#[test]
fn explain_formats_every_search_tvf_plan() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    let queries = [
        "SELECT _id, score FROM bm25_search('title', 'rust', 8)".to_string(),
        "SELECT _id, score FROM bm25_search_prefix('title', 'rus', 8)".to_string(),
        format!("SELECT _id, score FROM vector_search('emb', '{qv}', 8)"),
        "SELECT _id FROM token_match('title', 'rust')".to_string(),
        "SELECT _id FROM token_match('title', 'rust systems', 'and')".to_string(),
        "SELECT _id FROM exact_match('title', 'rust async')".to_string(),
        format!("SELECT _id, score FROM hybrid_search('title', 'rust', 'emb', '{qv}', 8)"),
    ];
    for q in &queries {
        let plan = explain_text(&st, q);
        assert!(
            !plan.trim().is_empty(),
            "EXPLAIN produced no plan text for: {q}"
        );
    }
}

/// `EXPLAIN ANALYZE` additionally runs the plan and formats the runtime
/// metrics — a distinct display path from plain `EXPLAIN`.
#[test]
fn explain_analyze_runs_and_formats_metrics() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    for q in [
        "SELECT _id FROM bm25_search('title', 'rust', 8)".to_string(),
        format!("SELECT _id FROM vector_search('emb', '{qv}', 8)"),
        format!("SELECT _id FROM hybrid_search('title', 'rust', 'emb', '{qv}', 8)"),
    ] {
        let plan = explain_text(&st, &format!("ANALYZE {q}"));
        assert!(!plan.trim().is_empty(), "EXPLAIN ANALYZE empty for: {q}");
    }
}

/// `SELECT *` over each TVF returns the scalar schema plus `score`,
/// driving the full scalar-materialization path in `execute`.
#[test]
fn star_projection_materializes_scalar_columns_for_every_tvf() {
    let st = demo_table();
    let qv = csv_one_hot(0);
    let queries = [
        format!("SELECT * FROM bm25_search('title', 'rust', {SURFACE_TOP_K})"),
        format!("SELECT * FROM bm25_search_prefix('title', 'rus', {SURFACE_TOP_K})"),
        format!("SELECT * FROM vector_search('emb', '{qv}', {SURFACE_TOP_K})"),
        "SELECT * FROM token_match('title', 'rust')".to_string(),
        "SELECT * FROM exact_match('title', 'rust async')".to_string(),
        format!("SELECT * FROM hybrid_search('title', 'rust', 'emb', '{qv}', {SURFACE_TOP_K})"),
    ];
    for q in &queries {
        let batches = st.reader().query_sql(q).expect("star query");
        let b = &batches[0];
        // _id, title, rating, score — the vector column is never exposed.
        assert_eq!(b.schema().field(0).name(), "_id", "{q}");
        assert!(b.schema().index_of("title").is_ok(), "{q}");
        assert!(b.schema().index_of("rating").is_ok(), "{q}");
        assert!(b.schema().index_of("score").is_ok(), "{q}");
        assert!(
            b.schema().index_of("emb").is_err(),
            "vector column must never reach SQL: {q}"
        );
    }
}

/// A SQL-level `ORDER BY score DESC` + `LIMIT` wraps the TVF scan in
/// sort/limit nodes — the optimizer calls `with_new_children` and
/// `statistics` on the custom exec while doing so.
#[test]
fn order_by_and_limit_wrap_the_tvf_scan() {
    let st = demo_table();
    let batches = st
        .reader()
        .query_sql(&format!(
            "SELECT _id, score FROM bm25_search('title', 'rust', {SURFACE_TOP_K}) \
             ORDER BY score DESC LIMIT 3"
        ))
        .expect("order+limit query");
    assert!(row_count(&batches) <= 3, "LIMIT 3 must cap the row count");
    assert!(row_count(&batches) >= 1, "'rust' should match something");
}

/// A scalar column from the TVF output feeds a `WHERE` predicate
/// evaluated above the custom scan.
#[test]
fn filter_on_materialized_scalar_above_tvf() {
    let st = demo_table();
    let batches = st
        .reader()
        .query_sql(&format!(
            "SELECT _id, rating FROM bm25_search('title', 'rust', {SURFACE_TOP_K}) \
             WHERE rating >= 8"
        ))
        .expect("filter query");
    // Only second-commit docs (ratings 8..16) survive; first-commit
    // 'rust' docs (ratings 0,3,7) are filtered out.
    let idx = batches[0].schema().index_of("rating").expect("rating col");
    for b in &batches {
        let col = b
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i64 rating");
        for i in 0..col.len() {
            assert!(col.value(i) >= 8, "WHERE rating >= 8 leaked a row");
        }
    }
}

/// Base-table aggregates and projections (no TVF) exercise the plain
/// `SupertableProvider` scan path: COUNT, SUM, MIN/MAX, ORDER BY, LIMIT.
#[test]
fn base_table_scan_supports_aggregates_and_projection() {
    let st = demo_table();
    let total = st
        .reader()
        .query_sql("SELECT COUNT(*) AS n FROM supertable")
        .expect("count");
    let n = total[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("i64 count")
        .value(0);
    assert_eq!(n, (DOCS_PER_COMMIT * 2) as i64, "all rows visible");

    // SUM over the full table equals the closed form 0+1+..+15.
    let sum = st
        .reader()
        .query_sql("SELECT SUM(rating) AS s FROM supertable")
        .expect("sum");
    let s = sum[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("i64 sum")
        .value(0);
    let expected: i64 = (0..(DOCS_PER_COMMIT * 2) as i64).sum();
    assert_eq!(s, expected, "SUM(rating) over the whole table");

    // Projection + ORDER BY + LIMIT over the base table.
    let top = st
        .reader()
        .query_sql("SELECT title, rating FROM supertable ORDER BY rating DESC LIMIT 2")
        .expect("ordered projection");
    assert!(row_count(&top) <= 2, "LIMIT 2 caps the projection");
}

/// `EXPLAIN` over the base-table scan formats the `SupertableProvider`
/// plan, a separate display path from the search TVFs.
#[test]
fn explain_base_table_scan() {
    let st = demo_table();
    let plan = explain_text(&st, "SELECT title FROM supertable WHERE rating > 4");
    assert!(!plan.trim().is_empty(), "base-table EXPLAIN was empty");
}
