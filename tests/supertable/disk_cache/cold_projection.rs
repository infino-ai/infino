// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Row materialization through COLD lazy readers.
//!
//! Under `ColdFetchMode::LazyForegroundWithBackgroundFill` a fresh consumer
//! serves searches from `open_lazy` byte-source readers — parquet metadata,
//! id pages, and projected scalar columns are all fetched as ranges from
//! storage rather than read from a resident buffer. Projection queries on
//! that path exercise the async take pipeline (`ByteSourceAsyncReader` +
//! `take_rows_async`) and the lazy reader's record-batch/id-lookup fetch
//! paths, which warm-path tests never touch. The assertions here are about
//! row INTEGRITY: every projected cell must belong to the hit row it is
//! returned with, across superfile boundaries.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int64Array, LargeStringArray, RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    VectorSearchOptions,
    storage::{LocalFsStorageProvider, StorageProvider},
    superfile::{
        builder::FtsConfig,
        fts::reader::{Bm25Stats, BoolMode},
    },
    supertable::{Supertable, SupertableOptions},
    test_helpers::{default_tokenizer, default_vector_config, lazy_foreground_disk_cache},
};
use tempfile::TempDir;

/// Matches `default_vector_config`'s dimension.
const DIM: usize = 16;
/// Random-rotation seed for the fixture's vector index.
const VECTOR_ROT_SEED: u64 = 23;
/// Larger than the corpus, so ranking — not truncation — decides hits.
const TOP_K: usize = 32;

/// First commit's titles (superfile 1). "fox" recurs in the second commit;
/// "async" is unique to doc 0.
const FIRST_BATCH: &[&str] = &["rust async fox", "lazy sleeping dog"];
/// Second commit's titles (superfile 2).
const SECOND_BATCH: &[&str] = &["clever red fox", "old grey wolf"];

fn fixed_list_f32(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// Schema `[title (FTS), rating (scalar), emb (vector)]`. The `rating`
/// scalar exists so projections force a real column decode through the
/// byte source, not just the id pages.
fn cold_options() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
        Field::new("emb", fixed_list_f32(DIM), false),
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

/// Doc `base + i` gets `titles[i]`, rating `base + i`, and a one-hot
/// embedding at dim `base + i` — every column derivable from the global
/// row number, so integrity checks need no side tables.
fn build_batch(titles: &[&str], base: usize, schema: Arc<Schema>) -> RecordBatch {
    let n = titles.len();
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
            Arc::new(LargeStringArray::from(titles.to_vec())),
            Arc::new(Int64Array::from(ratings)),
            Arc::new(fsl),
        ],
    )
    .expect("batch")
}

/// Commit the two-superfile corpus with a producer handle, drop it, and
/// reopen COLD: fresh consumer, lazy-foreground disk cache, nothing
/// resident. Returns the consumer plus the guards keeping the dirs alive.
fn cold_consumer() -> (Supertable, TempDir, TempDir) {
    let dir = TempDir::new().expect("storage tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    {
        let producer =
            Supertable::create(cold_options().with_storage(Arc::clone(&storage))).expect("create");
        let schema = producer.options().schema.clone();
        let mut w = producer.writer().expect("writer");
        w.append(&build_batch(FIRST_BATCH, 0, schema.clone()))
            .expect("append 1");
        w.commit().expect("commit 1");
        w.append(&build_batch(SECOND_BATCH, FIRST_BATCH.len(), schema))
            .expect("append 2");
        w.commit().expect("commit 2");
    }

    let cache_dir = TempDir::new().expect("cache tempdir");
    let cache = lazy_foreground_disk_cache(Arc::clone(&storage), cache_dir.path());
    let consumer = Supertable::open(
        cold_options()
            .with_storage(Arc::clone(&storage))
            .with_disk_cache(cache),
    )
    .expect("cold open");
    (consumer, dir, cache_dir)
}

/// `(title, rating)` pairs across the returned batches, in hit order.
fn projected_rows(batches: &[RecordBatch]) -> Vec<(String, i64)> {
    let mut rows = Vec::new();
    for batch in batches {
        let title_idx = batch.schema().index_of("title").expect("title projected");
        let rating_idx = batch.schema().index_of("rating").expect("rating projected");
        let titles = batch
            .column(title_idx)
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column")
            .clone();
        let ratings = batch
            .column(rating_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("rating column")
            .clone();
        for row in 0..batch.num_rows() {
            rows.push((titles.value(row).to_string(), ratings.value(row)));
        }
    }
    rows
}

/// The global row number whose title is `title` — the oracle for the
/// row-integrity checks.
fn expected_rating(title: &str) -> i64 {
    FIRST_BATCH
        .iter()
        .chain(SECOND_BATCH)
        .position(|t| *t == title)
        .expect("known title") as i64
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_bm25_projection_pairs_each_hit_with_its_own_row() {
    let (st, _dir, _cache) = cold_consumer();

    // "fox" hits one row in each superfile, so materialization crosses a
    // superfile boundary within one query.
    let hits = st
        .reader()
        .expect("reader")
        .bm25_search(
            "title",
            "fox",
            TOP_K,
            BoolMode::Or,
            Bm25Stats::PerSuperfile,
            Some(&["_id", "title", "rating", "score"]),
        )
        .expect("cold bm25");
    let rows = projected_rows(&hits);
    assert_eq!(rows.len(), 2, "one fox per superfile");
    for (title, rating) in &rows {
        assert!(title.contains("fox"));
        assert_eq!(
            *rating,
            expected_rating(title),
            "projected rating must belong to the hit row"
        );
    }

    // Multi-term OR drives the union cursor over the lazy posting reader;
    // "async" only exists in doc 0, "wolf" only in doc 3.
    let union = st
        .reader()
        .expect("reader")
        .bm25_search(
            "title",
            "async wolf",
            TOP_K,
            BoolMode::Or,
            Bm25Stats::PerSuperfile,
            Some(&["_id", "title", "rating", "score"]),
        )
        .expect("cold bm25 or");
    let mut union_titles: Vec<String> =
        projected_rows(&union).into_iter().map(|(t, _)| t).collect();
    union_titles.sort();
    assert_eq!(union_titles, vec![SECOND_BATCH[1], FIRST_BATCH[0]]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_unranked_matches_project_rows_correctly() {
    let (st, _dir, _cache) = cold_consumer();

    let exact = st
        .exact_match("title", SECOND_BATCH[0], Some(&["_id", "title", "rating"]))
        .expect("cold exact_match");
    let rows = projected_rows(&exact);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, SECOND_BATCH[0]);
    assert_eq!(rows[0].1, expected_rating(SECOND_BATCH[0]));

    let tokens = st
        .token_match(
            "title",
            "grey wolf",
            BoolMode::And,
            Some(&["_id", "title", "rating"]),
        )
        .expect("cold token_match");
    let rows = projected_rows(&tokens);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, SECOND_BATCH[1]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cold_vector_projection_ranks_and_materializes() {
    let (st, _dir, _cache) = cold_consumer();

    // Query with row 2's own one-hot embedding; it must rank first and
    // carry its own title + rating through the cold take path.
    let target = FIRST_BATCH.len();
    let mut q = vec![0.0f32; DIM];
    q[target % DIM] = 1.0;
    let hits = st
        .vector_search(
            "emb",
            &q,
            TOP_K,
            VectorSearchOptions::new(),
            None,
            Some(&["_id", "title", "rating", "score"]),
        )
        .expect("cold vector search");
    let rows = projected_rows(&hits);
    assert!(!rows.is_empty());
    assert_eq!(rows[0].0, SECOND_BATCH[0], "top hit is the queried row");
    assert_eq!(rows[0].1, target as i64);
    for (title, rating) in &rows {
        assert_eq!(
            *rating,
            expected_rating(title),
            "every materialized row must be internally consistent"
        );
    }
}
