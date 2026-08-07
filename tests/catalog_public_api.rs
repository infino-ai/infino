// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end tour of the public catalog surface on a local-filesystem
//! connection — the shipped `infino::*` API only, no engine internals.
//!
//! The public `Supertable` wrapper delegates every operation through the
//! `Table` trait seam to the engine handle; these tests pin that seam for
//! the operations no other test reaches through the catalog layer (schema,
//! vector + hybrid search, optimize, gc, SQL over a registered table, drop
//! with purge) so a wiring regression in the wrapper — not just in the
//! engine — fails loudly.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, time::Duration};

use infino::{
    BoolMode, IndexSpec, Metric, OptimizeOptions,
    arrow_array::{
        Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch, StringArray,
    },
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use tempfile::TempDir;

/// Embedding width for the vector column — the engine's minimum.
const EMB_DIM: usize = 16;
/// Top-K for every search here; larger than the corpus so ranking, not
/// truncation, decides the assertions.
const SEARCH_TOP_K: usize = 10;

/// Titles committed in the first batch (one commit == one superfile).
const FIRST_BATCH: &[&str] = &["the quick brown fox", "a lazy sleeping dog"];
/// Titles committed in the second batch, so maintenance has two
/// superfiles to compact.
const SECOND_BATCH: &[&str] = &["a red clever fox", "an old grey wolf"];

fn vector_field(dim: usize) -> DataType {
    DataType::FixedSizeList(
        Arc::new(Field::new("item", DataType::Float32, false)),
        dim as i32,
    )
}

fn table_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("emb", vector_field(EMB_DIM), false),
    ]))
}

/// Deterministic unit embedding: one-hot at `row % EMB_DIM`, so a query
/// with row `r`'s own embedding ranks row `r` first under cosine.
fn unit_embedding(row: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; EMB_DIM];
    v[row % EMB_DIM] = 1.0;
    v
}

fn build_batch(schema: SchemaRef, titles: &[&str], first_row_offset: usize) -> RecordBatch {
    let title_arr = LargeStringArray::from(titles.to_vec());
    let flat: Vec<f32> = (0..titles.len())
        .flat_map(|i| unit_embedding(first_row_offset + i))
        .collect();
    let values = Float32Array::from(flat);
    let emb_arr = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, false)),
        EMB_DIM as i32,
        Arc::new(values),
        None,
    );
    RecordBatch::try_new(schema, vec![Arc::new(title_arr), Arc::new(emb_arr)]).expect("valid batch")
}

/// First projected `title` value across the returned batches.
fn first_title(batches: &[RecordBatch]) -> String {
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("a hit");
    let idx = batch.schema().index_of("title").expect("title projected");
    let column = batch.column(idx);
    if let Some(arr) = column.as_any().downcast_ref::<LargeStringArray>() {
        arr.value(0).to_string()
    } else {
        column
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("title is utf8")
            .value(0)
            .to_string()
    }
}

#[test]
fn public_surface_search_maintain_query_and_drop() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");

    let schema = table_schema();
    let docs = db
        .create_table(
            "docs",
            schema.clone(),
            IndexSpec::new()
                .fts("title")
                .vector("emb", EMB_DIM, Metric::Cosine),
        )
        .expect("create_table");

    // Two appends == two commits == two superfiles, so optimize below has
    // real work and gc has real orphans afterwards.
    docs.append(&build_batch(schema.clone(), FIRST_BATCH, 0))
        .expect("append 1");
    docs.append(&build_batch(
        schema.clone(),
        SECOND_BATCH,
        FIRST_BATCH.len(),
    ))
    .expect("append 2");

    // Schema round-trips through the wrapper: exactly the declared payload
    // columns. The auto-injected `_id` primary key is not part of the
    // declared schema — it materializes in query projections only.
    let table_schema = docs.schema();
    assert!(table_schema.index_of("title").is_ok());
    assert!(table_schema.index_of("emb").is_ok());
    assert!(
        table_schema.index_of("_id").is_err(),
        "the engine-injected id column stays out of the declared schema"
    );

    // Vector kNN: querying with row 0's own embedding ranks row 0 first
    // under cosine.
    let knn = docs
        .vector_search(
            "emb",
            &unit_embedding(0),
            SEARCH_TOP_K,
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("vector_search");
    assert_eq!(first_title(&knn), FIRST_BATCH[0]);

    // Hybrid: the text leg pins "fox" rows, the vector leg (row 2's
    // embedding) ranks the second-batch fox above the first.
    let hybrid = docs
        .hybrid_search(
            "title",
            "fox",
            BoolMode::Or,
            "emb",
            &unit_embedding(FIRST_BATCH.len()),
            SEARCH_TOP_K,
            Some(&["_id", "title", "score"]),
        )
        .expect("hybrid_search");
    assert_eq!(first_title(&hybrid), SECOND_BATCH[0]);

    // The public wrapper's Debug is hand-written (dyn Table is not Debug);
    // it must render without reaching the engine.
    assert!(format!("{docs:?}").contains("Supertable"));

    // SQL over the registered table goes through the connection's
    // DataFusion path.
    let counted = db
        .query_sql("SELECT COUNT(*) AS n FROM docs")
        .expect("query_sql");
    let n: i64 = counted
        .iter()
        .filter(|b| b.num_rows() > 0)
        .map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<infino::arrow_array::Int64Array>()
                .expect("count is int64")
                .value(0)
        })
        .sum();
    assert_eq!(n as usize, FIRST_BATCH.len() + SECOND_BATCH.len());

    // Storage accounting sees the committed superfiles.
    let bytes = db.table_storage_bytes("docs").expect("storage bytes");
    assert!(bytes > 0, "committed table must have a nonzero footprint");

    // Maintenance through the wrapper: optimize compacts the two small
    // superfiles, gc reclaims what the swap orphaned. Every row must
    // survive the rewrite.
    docs.optimize(&OptimizeOptions::default())
        .expect("optimize");
    let report = docs.gc(Duration::ZERO).expect("gc");
    assert_eq!(report.delete_errors, 0);
    let post_maintenance = docs
        .vector_search(
            "emb",
            &unit_embedding(0),
            SEARCH_TOP_K,
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("vector_search after optimize+gc");
    assert_eq!(first_title(&post_maintenance), FIRST_BATCH[0]);

    // Drop with purge deletes the table's storage subtree and unregisters
    // the name.
    db.drop_table("docs", true).expect("drop_table purge");
    assert!(
        db.list_tables().expect("list_tables").is_empty(),
        "dropped table must leave the catalog empty"
    );
    assert!(
        db.open_table("docs").is_err(),
        "dropped table must not reopen"
    );
}

#[test]
fn open_table_returns_a_live_handle_on_a_fresh_connection() {
    // A second connection to the same directory must see the table by
    // catalog lookup alone (no shared in-process state) and search it.
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let schema = table_schema();
        let docs = db
            .create_table(
                "docs",
                schema.clone(),
                IndexSpec::new()
                    .fts("title")
                    .vector("emb", EMB_DIM, Metric::Cosine),
            )
            .expect("create_table");
        docs.append(&build_batch(schema, FIRST_BATCH, 0))
            .expect("append");
    }

    let db = connect(&uri).expect("reconnect");
    assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    let docs = db.open_table("docs").expect("open_table");
    let hits = docs
        .vector_search(
            "emb",
            &unit_embedding(1),
            SEARCH_TOP_K,
            None,
            Some(&["_id", "title", "score"]),
        )
        .expect("vector_search");
    assert_eq!(first_title(&hits), FIRST_BATCH[1]);
}
