// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Regression coverage for the bare-projection `_id` fast path.
//!
//! `bm25_search(.., None)` returns the engine-native `_id` + `score`
//! pair, and resolves `_id` by **manifest arithmetic** (`id_min +
//! local_doc_id`) whenever a segment's id span is contiguous — no
//! Parquet read. A projection that names a scalar column rides the
//! id-page read path instead. Both must agree on every id, in rank
//! order, across a multi-commit / multi-segment table; this pins the
//! arithmetic path against the storage-backed one.

#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use arrow_array::{Decimal128Array, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    superfile::{builder::FtsConfig, fts::reader::BoolMode},
    supertable::{Supertable, SupertableOptions},
    test_helpers::default_tokenizer,
};

/// Commits — enough that the writer's row-sharding produces a real
/// multi-segment fan-out and hits land in several segments.
const COMMITS: usize = 4;
/// Docs per commit — small; this is a correctness gate, not a bench.
const DOCS_PER_COMMIT: usize = 512;
/// Rayon pool width for the fixture.
const POOL_THREADS: usize = 4;
/// Top-k large enough to span multiple segments.
const K: usize = 32;
/// Doc index whose unique `token{:06}` the single-hit probe queries.
const PROBE_DOC: usize = 7;
/// Top-k for the single-hit probe — anything ≥ 1 works; small keeps
/// the assertion focused on identity, not ranking.
const PROBE_K: usize = 5;

fn options_title_only() -> SupertableOptions {
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(POOL_THREADS)
            .build()
            .expect("pool"),
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
        vec![],
        Some(default_tokenizer()),
    )
    .expect("valid options")
    .with_writer_pool(Arc::clone(&pool))
    .with_reader_pool(pool)
}

/// Every doc carries `common`; each doc also carries a unique token so
/// scores differ and rank order is non-trivial.
fn build_batch(commit: usize, schema: Arc<Schema>) -> RecordBatch {
    let titles: Vec<String> = (0..DOCS_PER_COMMIT)
        .map(|i| format!("common token{:06}", commit * DOCS_PER_COMMIT + i))
        .collect();
    let arr = LargeStringArray::from(titles.iter().map(String::as_str).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch")
}

fn decimal_column(batch: &RecordBatch, name: &str) -> Vec<i128> {
    let idx = batch.schema().index_of(name).expect("column present");
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("Decimal128 id column");
    (0..arr.len()).map(|i| arr.value(i)).collect()
}

fn float_column(batch: &RecordBatch, name: &str) -> Vec<f32> {
    let idx = batch.schema().index_of(name).expect("column present");
    let arr = batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32 score column");
    (0..arr.len()).map(|i| arr.value(i)).collect()
}

#[test]
fn bare_projection_ids_match_id_page_read_path() {
    let st = Supertable::create(options_title_only()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for commit in 0..COMMITS {
        w.append(&build_batch(commit, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    let reader = st.reader();

    // `common` is in every doc → hits span segments; per-doc unique
    // tokens keep scores distinct so rank order is meaningful.
    let bare = reader
        .bm25_search("title", "common", K, BoolMode::Or, None)
        .expect("bare search");
    assert_eq!(bare.len(), 1, "single merged batch");
    let bare = &bare[0];
    // Bare = engine-native pair: `_id` + `score`, nothing else.
    assert_eq!(bare.num_columns(), 2, "bare call returns _id + score");
    assert_eq!(bare.num_rows(), K);

    // Naming a scalar column forces the id-page/take path; its `_id`
    // column is the storage-backed ground truth.
    let projected = reader
        .bm25_search(
            "title",
            "common",
            K,
            BoolMode::Or,
            Some(&["_id", "title", "score"]),
        )
        .expect("projected search");
    let projected = &projected[0];
    assert_eq!(projected.num_columns(), 3);
    assert_eq!(projected.num_rows(), K);

    assert_eq!(
        decimal_column(bare, "_id"),
        decimal_column(projected, "_id"),
        "arithmetic _id resolve must equal the id-page read, in rank order"
    );
    assert_eq!(
        float_column(bare, "score"),
        float_column(projected, "score"),
        "same kernel ranking on both paths"
    );

    // Unique-token probe: exactly one hit, deterministic identity on
    // both paths.
    let probe_token = format!("token{PROBE_DOC:06}");
    let bare_one = reader
        .bm25_search("title", &probe_token, PROBE_K, BoolMode::Or, None)
        .expect("bare unique");
    let proj_one = reader
        .bm25_search(
            "title",
            &probe_token,
            PROBE_K,
            BoolMode::Or,
            Some(&["_id", "title", "score"]),
        )
        .expect("projected unique");
    assert_eq!(bare_one[0].num_rows(), 1);
    assert_eq!(proj_one[0].num_rows(), 1);
    assert_eq!(
        decimal_column(&bare_one[0], "_id"),
        decimal_column(&proj_one[0], "_id"),
    );
}
