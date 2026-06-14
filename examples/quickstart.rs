// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Public-API tour of Infino — the smallest end-to-end program that
//! exercises the shipped `infino::*` surface only (no internal
//! `test-helpers`), so it runs straight out of a fresh clone:
//!
//! ```text
//! cargo run --example quickstart
//! ```
//!
//! It opens an in-memory connection, then shows the three retrieval
//! modes a single table supports: BM25 full-text search, vector kNN,
//! and SQL — all over the same committed superfiles.

use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use arrow_array::{Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{BoolMode, IndexSpec, Metric, VectorSearchOptions, connect};

/// Embedding width for the demo vector column. Small on purpose (the
/// engine's minimum is 16) — the point is the API shape, not recall.
const EMB_DIM: usize = 16;
/// IVF centroid count. A handful of rows is one cluster.
const DEMO_N_CENT: usize = 1;
/// Top-K for every search in this tour.
const SEARCH_TOP_K: usize = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // `memory://` keeps everything in-process; swap for "./data" or
    // "s3://bucket/prefix" and nothing else in this file changes.
    let db = connect("memory://")?;

    // One table, two payload columns: a text column indexed for BM25
    // and a `FixedSizeList<Float32, EMB_DIM>` column indexed for vector
    // search. The `_id` primary key is auto-injected by the table.
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", vector_field(EMB_DIM), false),
    ]));
    let docs = db.create_table(
        "docs",
        schema.clone(),
        IndexSpec::new()
            .fts("title")
            .vector("emb", EMB_DIM, DEMO_N_CENT, Metric::Cosine),
    )?;

    // One `append` == one commit == one sealed, immutable superfile.
    let titles = ["the quick brown fox", "a lazy sleeping dog"];
    docs.append(&build_batch(schema, &titles)?)?;

    // A — BM25 keyword search. Arrow rows carry the auto-injected `_id`,
    // the projected columns, and a trailing `score`.
    println!("== A. BM25 full-text search for \"fox\" ==");
    let hits = docs.bm25_search(
        "title",
        "fox",
        SEARCH_TOP_K,
        BoolMode::Or,
        Some(&["_id", "title", "score"]),
    )?;
    print_batches(&hits);

    // B — vector kNN. Query with the first row's own embedding, so it
    // ranks first under cosine distance.
    println!("== B. vector kNN (query == row 0's embedding) ==");
    let query = unit_embedding(0);
    let knn = docs.vector_search(
        "emb",
        &query,
        SEARCH_TOP_K,
        VectorSearchOptions::new(),
        Some(&["_id", "title", "score"]),
    )?;
    print_batches(&knn);

    // C — SQL over the catalog. Every superfile is also a valid Parquet
    // file, so the same data answers plain SQL.
    println!("== C. SQL over the same table ==");
    let rows = db.query_sql("SELECT _id, title FROM docs ORDER BY _id")?;
    print_batches(&rows);

    Ok(())
}

/// Arrow type for an `EMB_DIM`-wide vector column.
fn vector_field(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
    )
}

/// A one-hot unit embedding with its single non-zero at `row % EMB_DIM`,
/// giving each row a distinct direction.
fn unit_embedding(row: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; EMB_DIM];
    v[row % EMB_DIM] = 1.0;
    v
}

/// Build a `(title, emb)` batch for the given titles.
fn build_batch(
    schema: Arc<Schema>,
    titles: &[&str],
) -> Result<RecordBatch, Box<dyn std::error::Error>> {
    let title_col = LargeStringArray::from(titles.to_vec());
    let mut flat = Vec::with_capacity(titles.len() * EMB_DIM);
    for row in 0..titles.len() {
        flat.extend_from_slice(&unit_embedding(row));
    }
    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let emb_col = FixedSizeListArray::try_new(
        item_field,
        EMB_DIM as i32,
        Arc::new(Float32Array::from(flat)) as Arc<dyn Array>,
        None,
    )?;
    Ok(RecordBatch::try_new(
        schema,
        vec![Arc::new(title_col), Arc::new(emb_col)],
    )?)
}

fn print_batches(batches: &[RecordBatch]) {
    match pretty_format_batches(batches) {
        Ok(table) => {
            for line in table.to_string().lines() {
                println!("  {line}");
            }
        }
        Err(e) => println!("  <failed to format batches: {e}>"),
    }
    println!();
}
