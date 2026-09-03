// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-surface contract for index-only FTS columns
//! (`FtsConfig { stored: false, .. }`): searchable, never readable.
//!
//! * `bm25_search` over the column returns ranked `_id` + `score`;
//! * naming the column in a search projection is rejected up front
//!   (it resolves against the stored schema, like any unknown column);
//! * SQL never sees the column: `SELECT *` omits it, and selecting or
//!   filtering on it by name fails at plan time;
//! * the error text stays at the caller's level — it may name the
//!   column, but no storage-format or execution-engine tokens leak.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::{
        builder::FtsConfig,
        fts::reader::{Bm25Stats, BoolMode},
    },
    supertable::{Supertable, SupertableOptions},
};

/// Docs per commit; two commits keep the corpus multi-superfile.
const DOCS_PER_COMMIT: usize = 4;
/// Top-k larger than the corpus so nothing truncates.
const K: usize = 32;

/// Schema `[title (stored FTS), body (index-only FTS), rating]`.
fn options_with_index_only_body() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
    ]));
    SupertableOptions::new(
        schema,
        vec![
            FtsConfig::new("title"),
            FtsConfig::new("body").stored(false),
        ],
        vec![],
    )
    .expect("valid options")
}

fn build_batch(rows: &[(&str, &str)], base: usize, schema: Arc<Schema>) -> RecordBatch {
    let titles = LargeStringArray::from(rows.iter().map(|(t, _)| *t).collect::<Vec<_>>());
    let bodies = LargeStringArray::from(rows.iter().map(|(_, b)| *b).collect::<Vec<_>>());
    let ratings: Vec<i64> = (0..rows.len()).map(|i| (base + i) as i64).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(Int64Array::from(ratings)),
        ],
    )
    .expect("batch")
}

/// Two-superfile corpus; `signal` appears in exactly one body per segment.
fn demo_table() -> Supertable {
    let st = Supertable::create(options_with_index_only_body()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    w.append(&build_batch(
        &[
            ("rust systems", "signal in the noise"),
            ("python data", "frames and plots"),
            ("go network", "goroutines everywhere"),
            ("java spring", "beans and factories"),
        ],
        0,
        schema.clone(),
    ))
    .expect("append seg1");
    w.commit().expect("commit seg1");
    w.append(&build_batch(
        &[
            ("ruby rails", "conventions abound"),
            ("scala akka", "actors send signal fast"),
            ("kotlin flow", "coroutines flowing"),
            ("swift ui", "views and state"),
        ],
        DOCS_PER_COMMIT,
        schema,
    ))
    .expect("append seg2");
    w.commit().expect("commit seg2");
    drop(w);
    st
}

#[test]
fn index_only_column_searches_but_rejects_projection() {
    let st = demo_table();
    let reader = st.reader().expect("reader");

    // Ranked search over the index-only column spans both segments.
    let batches = reader
        .bm25_search(
            "body",
            "signal",
            K,
            BoolMode::Or,
            Bm25Stats::PerSuperfile,
            None,
        )
        .expect("bm25 over index-only column");
    let n: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(n, 2, "one signal doc per segment");

    // Projecting stored columns works.
    let batches = reader
        .bm25_search(
            "body",
            "signal",
            K,
            BoolMode::Or,
            Bm25Stats::PerSuperfile,
            Some(&["_id", "title", "rating", "score"]),
        )
        .expect("stored projection");
    assert_eq!(batches[0].num_columns(), 4);

    // Naming the index-only column is rejected up front, with a
    // caller-level message: the column may be named, engine internals
    // may not.
    let err = reader
        .bm25_search(
            "body",
            "signal",
            K,
            BoolMode::Or,
            Bm25Stats::PerSuperfile,
            Some(&["_id", "body", "score"]),
        )
        .expect_err("index-only column must not be projectable");
    let msg = err.to_string();
    assert!(msg.contains("body"), "error names the column: {msg}");
    for leak in ["DataFusion", "parquet", "inf.fts"] {
        assert!(!msg.contains(leak), "error leaks {leak}: {msg}");
    }
}

#[test]
fn sql_never_sees_an_index_only_column() {
    let st = demo_table();
    let reader = st.reader().expect("reader");

    // SELECT * omits the column entirely: _id, title, rating.
    let batches = reader
        .query_sql("SELECT * FROM supertable ORDER BY rating")
        .expect("select star");
    assert_eq!(
        batches[0]
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>(),
        vec!["_id", "title", "rating"],
        "the index-only column is not part of the SQL view"
    );

    // Stored columns stay selectable and filterable.
    let batches = reader
        .query_sql("SELECT title FROM supertable WHERE rating >= 4")
        .expect("stored select");
    let n: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(n, DOCS_PER_COMMIT);

    // Selecting or filtering on the index-only column fails at plan
    // time, like any unknown column.
    for q in [
        "SELECT body FROM supertable",
        "SELECT title FROM supertable WHERE body LIKE '%signal%'",
        "SELECT _id, body FROM bm25_search('body', 'signal', 8)",
    ] {
        assert!(
            reader.query_sql(q).is_err(),
            "index-only column must be invisible to SQL: {q}"
        );
    }

    // The search TVF itself still ranks over the column — only reading
    // its text is off the table.
    let batches = reader
        .query_sql("SELECT _id, score FROM bm25_search('body', 'signal', 8)")
        .expect("bm25 TVF over index-only column");
    let n: usize = batches.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(n, 2);
}
