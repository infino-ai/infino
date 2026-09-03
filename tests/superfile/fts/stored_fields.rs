// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Index-only FTS columns (`FtsConfig { stored: false, .. }`): the text
//! feeds the FTS blob at ingest but never lands in the Parquet body.
//!
//! Pins the superfile-level contract:
//! * the stored schema drops the column while search (ranked, phrase)
//!   still works over it;
//! * both compaction merge kinds that FTS/scalar files take — the
//!   generic reader merge and the streaming postings merge — produce a
//!   file whose search behavior is score-identical to a fresh build of
//!   the same surviving rows, for stored and index-only columns alike
//!   (merges carry prebuilt postings; nothing is re-tokenized);
//! * tombstoned rows drop out of the carried index.

use std::sync::Arc;

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{
    roaring::RoaringBitmap,
    superfile::{
        SuperfileReader,
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        fts::reader::BoolMode,
    },
    test_helpers::decimal128_ids,
};

/// k large enough to capture every match on these tiny corpora.
const K_ALL: usize = 64;
/// Score-equality tolerance between a fresh build and a merged build.
const SCORE_ABS_TOLERANCE: f32 = 1e-3;

/// Corpus rows: `(id, title, body)`. `title` is a stored FTS column,
/// `body` an index-only positional one.
type Row<'a> = (u64, &'a str, &'a str);

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]))
}

fn options() -> BuilderOptions {
    BuilderOptions::new(
        schema(),
        "doc_id",
        vec![
            FtsConfig::new("title"),
            FtsConfig::new("body").positions(true).stored(false),
        ],
        vec![],
    )
}

fn build(rows: &[Row<'_>]) -> Arc<SuperfileReader> {
    let mut b = SuperfileBuilder::new(options()).expect("builder");
    let ids = decimal128_ids(rows.iter().map(|(id, _, _)| *id));
    let titles = LargeStringArray::from(rows.iter().map(|(_, t, _)| *t).collect::<Vec<_>>());
    let bodies = LargeStringArray::from(rows.iter().map(|(_, _, b)| *b).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(
        schema(),
        vec![Arc::new(ids), Arc::new(titles), Arc::new(bodies)],
    )
    .expect("batch");
    b.add_batch(&batch, &[]).expect("add_batch");
    Arc::new(SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open"))
}

const CORPUS_A: &[Row<'static>] = &[
    (1, "rust systems", "fast async runtime for services"),
    (2, "python data", "pandas frames and fast plots"),
    (3, "go network", "goroutines make fast async easy"),
];
const CORPUS_B: &[Row<'static>] = &[
    (4, "java spring", "enterprise beans everywhere"),
    (5, "rust web", "async runtime with tower services"),
    (6, "ruby rails", "convention over configuration"),
];

/// Probe queries spanning both columns, ranked + phrase.
const PROBES: &[(&str, &str)] = &[
    ("title", "rust"),
    ("title", "rust web"),
    ("body", "fast"),
    ("body", "async runtime"),
    ("body", "\"fast async\""),
    ("body", "\"async runtime\""),
];

async fn hits(r: &SuperfileReader, column: &str, query: &str) -> Vec<(u32, f32)> {
    r.bm25_hits_async(column, query, K_ALL, BoolMode::Or)
        .await
        .expect("bm25 probe")
}

/// Assert `got` and `want` agree on both membership order and scores.
fn assert_same_hits(got: &[(u32, f32)], want: &[(u32, f32)], label: &str) {
    let g: Vec<u32> = got.iter().map(|(d, _)| *d).collect();
    let w: Vec<u32> = want.iter().map(|(d, _)| *d).collect();
    assert_eq!(g, w, "{label}: doc membership/order diverged");
    for ((d, gs), (_, ws)) in got.iter().zip(want.iter()) {
        assert!(
            (gs - ws).abs() < SCORE_ABS_TOLERANCE,
            "{label}: doc {d} score {gs} vs fresh {ws}"
        );
    }
}

/// Run every probe against `merged` and a fresh single build over
/// `expect_rows`, asserting identical hits + scores.
async fn assert_merge_equals_fresh(merged: &SuperfileReader, expect_rows: &[Row<'_>]) {
    let fresh = build(expect_rows);
    assert_eq!(merged.n_docs(), expect_rows.len() as u64);
    for (column, query) in PROBES {
        let got = hits(merged, column, query).await;
        let want = hits(&fresh, column, query).await;
        assert_same_hits(&got, &want, &format!("{column} / {query}"));
    }
}

#[tokio::test]
async fn index_only_column_is_searchable_but_not_stored() {
    let r = build(CORPUS_A);
    let names: Vec<&str> = r
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    assert_eq!(
        names,
        vec!["doc_id", "title"],
        "index-only body stays out of the Parquet body"
    );
    // Ranked search and phrase both work over the index-only column.
    let got = hits(&r, "body", "fast").await;
    assert_eq!(
        got.iter().map(|(d, _)| *d).collect::<Vec<_>>().len(),
        3,
        "every doc mentions fast"
    );
    let got = hits(&r, "body", "\"fast async\"").await;
    let ids: Vec<u32> = got.iter().map(|(d, _)| *d).collect();
    assert_eq!(ids, vec![0, 2], "phrase matches contiguous docs only");
}

#[tokio::test]
async fn generic_reader_merge_matches_fresh_build() {
    let (a, b) = (build(CORPUS_A), build(CORPUS_B));
    let mut mb = SuperfileBuilder::new(BuilderOptions::new_from_reader(&a)).expect("merge builder");
    mb.add_batch_from_reader(&a, None).expect("merge a");
    mb.add_batch_from_reader(&b, None).expect("merge b");
    let merged =
        SuperfileReader::open(Bytes::from(mb.finish().expect("finish"))).expect("open merged");
    let all: Vec<Row<'_>> = CORPUS_A.iter().chain(CORPUS_B).copied().collect();
    assert_merge_equals_fresh(&merged, &all).await;
}

#[tokio::test]
async fn streaming_fts_merge_matches_fresh_build() {
    let (a, b) = (build(CORPUS_A), build(CORPUS_B));
    let (bytes, stats) =
        SuperfileBuilder::build_from_readers_fts_merge(&[(a, None), (b, None)]).expect("fts merge");
    assert_eq!(stats.n_docs, 6);
    let merged = SuperfileReader::open(Bytes::from(bytes)).expect("open merged");
    let all: Vec<Row<'_>> = CORPUS_A.iter().chain(CORPUS_B).copied().collect();
    assert_merge_equals_fresh(&merged, &all).await;
}

#[tokio::test]
async fn merges_drop_tombstoned_rows_from_the_carried_index() {
    let (a, b) = (build(CORPUS_A), build(CORPUS_B));
    // Delete local doc 0 of A ("rust systems") and local doc 1 of B
    // ("rust web") — both match probe terms, so a leak is visible.
    let mut dead_a = RoaringBitmap::new();
    dead_a.insert(0);
    let mut dead_b = RoaringBitmap::new();
    dead_b.insert(1);
    let survivors: Vec<Row<'_>> = vec![CORPUS_A[1], CORPUS_A[2], CORPUS_B[0], CORPUS_B[2]];

    let (bytes, _) = SuperfileBuilder::build_from_readers_fts_merge(&[
        (a.clone(), Some(Arc::new(dead_a.clone()))),
        (b.clone(), Some(Arc::new(dead_b.clone()))),
    ])
    .expect("fts merge with tombstones");
    let merged = SuperfileReader::open(Bytes::from(bytes)).expect("open merged");
    assert_merge_equals_fresh(&merged, &survivors).await;

    // Same deletions through the generic reader merge.
    let mut mb = SuperfileBuilder::new(BuilderOptions::new_from_reader(&a)).expect("merge builder");
    mb.add_batch_from_reader(&a, Some(Arc::new(dead_a)))
        .expect("merge a");
    mb.add_batch_from_reader(&b, Some(Arc::new(dead_b)))
        .expect("merge b");
    let merged =
        SuperfileReader::open(Bytes::from(mb.finish().expect("finish"))).expect("open merged");
    assert_merge_equals_fresh(&merged, &survivors).await;
}
