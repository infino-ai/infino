// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Open-format compatibility oracle: a superfile, read via vanilla
//! DataFusion, must return exact expected counts and column values
//! for the planted corpus.
//!
//! Every superfile is a valid Parquet file, and vanilla Parquet
//! engines (DataFusion, DuckDB, pyarrow) read it as a regular
//! table without any infino-aware adapter. We pick **DataFusion**
//! as the verification engine because:
//!
//!   1. It's pure Rust — fits the test env without a Python or C
//!      toolchain.
//!   2. It uses the same `parquet-rs` metadata reader our
//!      `SuperfileReader` is built on, so any divergence in the
//!      footer KV / row-group bookkeeping that breaks third-party
//!      readers will manifest here.
//!   3. The shared metadata path means: if DataFusion can read it,
//!      DuckDB and pyarrow (which use independent Parquet
//!      implementations) almost certainly can too — those round-
//!      trips aren't part of this in-tree harness because the
//!      marginal signal (parquet-rs writer producing a file only
//!      parquet-rs can read) is small relative to the Python /
//!      C++ toolchain cost they would impose on CI.
//!
//! The check the test enforces is functional, not aesthetic:
//! `SELECT COUNT(*)`, `SELECT ... WHERE` predicates, and direct
//! column-value extraction must all produce the planted ground
//! truth. If we ever break the Parquet body (e.g. by an off-by-one
//! in the splice's truncation), this test catches it.

use std::sync::Arc;

use arrow_array::{Array, Decimal128Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::prelude::*;
use infino::{
    superfile::{
        builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        vector::distance::normalize,
    },
    test_helpers::{decimal128_ids, default_tokenizer, default_vector_config},
};
use tempfile::NamedTempFile;

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Random-rotation seed for the planted superfile's vector index.
const PARQUET_COMPAT_ROT_SEED: u64 = 7;
/// Planted-corpus document count.
const PARQUET_COMPAT_N_DOCS: u32 = 6;
/// Embedding dimension (matches `default_vector_config`).
const PARQUET_COMPAT_EMB_DIM: usize = 16;
/// Secondary one-hot axis weight planted in each doc vector.
const PARQUET_COMPAT_SECONDARY_WEIGHT: f32 = 0.1;
/// Planted category distribution counts.
const COUNT_RUST: i64 = 3;
const COUNT_PYTHON: i64 = 2;
const COUNT_GO: i64 = 1;
/// Inclusive `doc_id` BETWEEN bounds for the predicate-pushdown test.
const FILTER_LO: i128 = 102;
const FILTER_HI: i128 = 104;

/// Build a planted superfile with FTS + vector indexes alongside
/// scalar columns. Returns the bytes ready to write to a temp file.
///
/// Planted distribution:
///   - 6 docs with `doc_id` 100..=105
///   - `category` column has 3 "rust" + 2 "python" + 1 "go"
///   - `score` column has known values for direct extraction
fn build_planted_superfile() -> Bytes {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("category", DataType::LargeUtf8, false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![default_vector_config("emb", PARQUET_COMPAT_ROT_SEED)],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102, 103, 104, 105]);
    let categories = LargeStringArray::from(vec!["rust", "rust", "python", "rust", "python", "go"]);
    let titles = LargeStringArray::from(vec![
        "rust async runtime",
        "rust embedded systems",
        "python data pipeline",
        "rust web framework",
        "python ml numpy",
        "go concurrency model",
    ]);
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(categories), Arc::new(titles)],
    )
    .expect("build RecordBatch");

    // Build deterministic unit-norm vectors so cosine doesn't NaN.
    let mut flat =
        Vec::<f32>::with_capacity(PARQUET_COMPAT_N_DOCS as usize * PARQUET_COMPAT_EMB_DIM);
    for i in 0..PARQUET_COMPAT_N_DOCS {
        let mut v = vec![0.0f32; PARQUET_COMPAT_EMB_DIM];
        v[(i as usize) % PARQUET_COMPAT_EMB_DIM] = 1.0;
        v[((i as usize) + 1) % PARQUET_COMPAT_EMB_DIM] = PARQUET_COMPAT_SECONDARY_WEIGHT;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    Bytes::from(b.finish().expect("finish builder"))
}

/// Spill the superfile bytes to a temp file and return the wrapper.
/// DataFusion's `register_parquet` takes a path, not bytes.
fn spill_to_tempfile(bytes: &Bytes) -> NamedTempFile {
    let f = NamedTempFile::with_suffix(".parquet").expect("tempfile");
    std::fs::write(f.path(), bytes).expect("write tempfile");
    f
}

/// Register the superfile as a DataFusion table and return the context.
async fn datafusion_ctx_for(superfile: &NamedTempFile) -> SessionContext {
    let ctx = SessionContext::new();
    ctx.register_parquet(
        "docs",
        superfile.path().to_str().expect("utf8 path"),
        ParquetReadOptions::default(),
    )
    .await
    .expect("DataFusion must register the superfile as a Parquet table");
    ctx
}

#[tokio::test]
async fn datafusion_reads_superfile_as_plain_parquet_count_matches() {
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql("SELECT COUNT(*) AS n FROM docs")
        .await
        .expect("count query parses + plans");
    let batches = df.collect().await.expect("count query executes");
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count is Int64")
        .value(0);
    assert_eq!(
        n, PARQUET_COMPAT_N_DOCS as i64,
        "DataFusion sees 6 rows in the superfile"
    );
}

#[tokio::test]
async fn datafusion_filter_predicate_returns_planted_rust_count() {
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql("SELECT COUNT(*) AS n FROM docs WHERE category = 'rust'")
        .await
        .expect("await async result");
    let batches = df.collect().await.expect("collect record batches");
    let n = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("downcast")
        .value(0);
    assert_eq!(n, 3, "3 rust-categorized docs in the planted corpus");
}

#[tokio::test]
async fn datafusion_groupby_yields_correct_per_category_counts() {
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql(
            "SELECT category, COUNT(*) AS n FROM docs \
             GROUP BY category ORDER BY category",
        )
        .await
        .expect("await async result");
    let batches = df.collect().await.expect("collect record batches");

    // DataFusion may materialize the GROUP BY key as Utf8 even
    // though the source column is LargeUtf8 (it picks the cheapest
    // type for the hash-aggregate path). Handle both.
    let cat_col = batches[0].column(0);
    let counts = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("count is Int64");

    let extract_cat = |i: usize| -> String {
        if let Some(a) = cat_col
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
        {
            a.value(i).to_string()
        } else if let Some(a) = cat_col.as_any().downcast_ref::<arrow_array::StringArray>() {
            a.value(i).to_string()
        } else if let Some(a) = cat_col
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
        {
            a.value(i).to_string()
        } else {
            panic!(
                "DataFusion returned unexpected type for the category column: {:?}",
                cat_col.data_type()
            )
        }
    };
    let mut got: Vec<(String, i64)> = (0..cat_col.len())
        .map(|i| (extract_cat(i), counts.value(i)))
        .collect();
    got.sort();

    assert_eq!(
        got,
        vec![
            ("go".to_string(), COUNT_GO),
            ("python".to_string(), COUNT_PYTHON),
            ("rust".to_string(), COUNT_RUST),
        ],
        "GROUP BY counts must match planted distribution"
    );
}

#[tokio::test]
async fn datafusion_extracts_planted_doc_ids_in_order() {
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql("SELECT doc_id FROM docs ORDER BY doc_id")
        .await
        .expect("await async result");
    let batches = df.collect().await.expect("collect record batches");
    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("doc_id is Decimal128");
    let collected: Vec<i128> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(
        collected,
        vec![100, 101, 102, 103, 104, 105],
        "exact planted doc_id sequence must round-trip"
    );
}

#[tokio::test]
async fn datafusion_sees_all_three_columns_in_schema() {
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql("SELECT * FROM docs LIMIT 1")
        .await
        .expect("await async result");
    let schema = df.schema();
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec!["doc_id", "category", "title"],
        "DataFusion sees the user's full schema; the inf.* KV \
         metadata is invisible (correct for opaque KV keys)"
    );
}

#[tokio::test]
async fn datafusion_predicate_pushdown_does_not_break_on_inf_kv_metadata() {
    // Stress test: a query that exercises Parquet's predicate
    // pushdown machinery (which reads stats from the row-group
    // metadata). If our footer-rewrite corrupts those stats, this
    // is where it'd manifest.
    let bytes = build_planted_superfile();
    let f = spill_to_tempfile(&bytes);
    let ctx = datafusion_ctx_for(&f).await;

    let df = ctx
        .sql(&format!(
            "SELECT category, doc_id FROM docs \
             WHERE doc_id BETWEEN {FILTER_LO} AND {FILTER_HI} \
             ORDER BY doc_id",
        ))
        .await
        .expect("await async result");
    let batches = df.collect().await.expect("collect record batches");
    let ids = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("downcast");
    let collected: Vec<i128> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(collected, (FILTER_LO..=FILTER_HI).collect::<Vec<i128>>());
}
