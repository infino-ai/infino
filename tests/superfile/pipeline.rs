// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! End-to-end superfile pipeline: build a real superfile (Parquet
//! body + FTS blob + vector blob), reopen it via `SuperfileReader`,
//! exercise BM25 + vector search, and verify the bytes are still a
//! valid Parquet file readable by parquet-rs.

use arrow_array::{Array, Decimal128Array, Float32Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::builder::{
    BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig as SfVectorConfig,
};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::{Metric, normalize};
use infino::superfile::vector::rerank_codec::RerankCodec;
use infino::superfile::{SuperfileReader, VectorSearchOptions};
use infino::test_helpers::{decimal128_ids, default_tokenizer};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::sync::Arc;

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Vector-column dimension for the pipeline fixture.
const EMB_DIM: usize = 16;
/// IVF centroid count for the pipeline fixture.
const N_CENT: usize = 4;
/// Random-rotation seed for the pipeline fixture's vector index.
const ROT_SEED: u64 = 17;
/// Planted-corpus document count.
const N_DOCS: usize = 6;
/// Schema field count (doc_id + title + body + score).
const SCHEMA_FIELD_COUNT: usize = 4;
/// BM25 / vector top-k used by the pipeline searches.
const SEARCH_K: usize = 5;
/// nprobe (full cluster sweep == N_CENT) for the self-query.
const NPROBE: usize = 4;
/// Secondary one-hot axis weight planted in each doc vector.
const SECONDARY_AXIS_WEIGHT: f32 = 0.1;
/// Document count for the "naked" (no-index) superfile fixture.
const NAKED_N_DOCS: u64 = 2;
/// Number of `add_batch` chunks in the multi-batch continuity test.
const MULTI_BATCH_CHUNK_COUNT: u64 = 3;
/// Per-chunk external-id stride in the multi-batch test.
const MULTI_BATCH_ID_STRIDE: u64 = 10;
/// BM25 top-k with headroom for the multi-batch / fts-only queries.
const MULTI_TERM_K: usize = 10;

fn pipeline_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("score", DataType::Float32, true),
    ]))
}

/// Build a superfile with FTS on `title`+`body` and a single vector
/// column `emb`. 6 docs; cosine similarity, dim=16.
fn build_pipeline_superfile() -> Bytes {
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![
            FtsConfig {
                column: "title".into(),
            },
            FtsConfig {
                column: "body".into(),
            },
        ],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102, 103, 104, 105]);
    let titles = LargeStringArray::from(vec![
        "rust async runtime",
        "python data pipeline",
        "rust embedded system",
        "javascript web frontend",
        "go concurrency model",
        "rust web framework",
    ]);
    let bodies = LargeStringArray::from(vec![
        "tokio fast",
        "pandas slow",
        "embedded firmware low level",
        "react node browser",
        "channels goroutines fast",
        "actix axum tide",
    ]);
    let scores = Float32Array::from(vec![0.9, 0.5, 0.7, 0.6, 0.8, 0.95]);

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    // Build 6 deterministic unit-norm vectors with planted structure:
    // docs 0/2/5 ("rust" titles) cluster on axis 0; doc 1 on axis 1;
    // doc 3 on axis 2; doc 4 on axis 3.
    let mut flat = Vec::<f32>::with_capacity(N_DOCS * EMB_DIM);
    let axes: [usize; N_DOCS] = [0, 1, 0, 2, 3, 0];
    for &a in &axes {
        let mut v = vec![0.0f32; EMB_DIM];
        v[a] = 1.0;
        v[(a + 1) % EMB_DIM] = SECONDARY_AXIS_WEIGHT;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    Bytes::from(b.finish().expect("finish builder"))
}

#[test]
fn end_to_end_open_reports_correct_metadata() {
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes).expect("open superfile");
    assert_eq!(r.n_docs(), N_DOCS as u64);
    assert_eq!(r.id_column(), "doc_id");
    assert_eq!(r.fts_columns(), vec!["title", "body"]);
    assert_eq!(r.vector_columns(), vec!["emb"]);
    assert_eq!(r.schema().fields().len(), SCHEMA_FIELD_COUNT);
}

#[tokio::test]
async fn end_to_end_bm25_finds_rust_docs() {
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes).expect("open superfile");
    let hits = r
        .bm25_search("title", "rust", SEARCH_K, BoolMode::Or)
        .await
        .expect("BM25 search");
    let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    // docs 0, 2, 5 have "rust" in title
    assert!(doc_ids.contains(&0));
    assert!(doc_ids.contains(&2));
    assert!(doc_ids.contains(&5));
}

#[tokio::test]
async fn end_to_end_bm25_multi_combines_columns() {
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes).expect("open superfile");
    let hits = r
        .bm25_search_multi(
            &[("title", 1.0), ("body", 1.0)],
            "rust embedded",
            SEARCH_K,
            BoolMode::Or,
        )
        .await
        .expect("BM25 multi-column search");
    // doc 2 has both "rust" (title) and "embedded" (body) → should rank well.
    assert!(!hits.is_empty());
    let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert!(doc_ids.contains(&2));
}

#[tokio::test]
async fn end_to_end_vector_search_recovers_self() {
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes).expect("open superfile");
    // Reconstruct doc 4's vector (axis 3 + tiny axis 4).
    let mut q = vec![0.0f32; EMB_DIM];
    q[3] = 1.0;
    q[4] = SECONDARY_AXIS_WEIGHT;
    normalize(&mut q);
    let hits = r
        .vector_search("emb", &q, 1, VectorSearchOptions::new().with_nprobe(NPROBE))
        .await
        .expect("vector search");
    assert_eq!(hits[0].0, 4, "self-query should recover doc 4");
}

#[test]
fn end_to_end_parquet_round_trip() {
    // The superfile bytes are also a valid Parquet file; vanilla
    // parquet-rs must read them and recover all rows + columns.
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes.clone()).expect("open superfile");
    let parquet = r
        .parquet_bytes()
        .expect("eager open retains parquet bytes")
        .clone();
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet)
        .expect("try_new ParquetRecordBatchReaderBuilder");
    let mut reader = builder.build().expect("build parquet reader");
    let batch = reader
        .next()
        .expect("at least one batch")
        .expect("decode batch");
    assert_eq!(batch.num_rows(), N_DOCS);
    assert_eq!(batch.num_columns(), SCHEMA_FIELD_COUNT);
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("downcast");
    let collected: Vec<i128> = (0..ids.len()).map(|i| ids.value(i)).collect();
    assert_eq!(collected, vec![100, 101, 102, 103, 104, 105]);
}

#[test]
fn end_to_end_no_indexes_still_valid_parquet() {
    // A "naked" superfile (no FTS, no vectors) should still open and
    // be readable as Parquet.
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64, 2]);
    let titles = LargeStringArray::from(vec!["a", "b"]);
    let bodies = LargeStringArray::from(vec!["x", "y"]);
    let scores = Float32Array::from(vec![1.0, 2.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));

    let r = SuperfileReader::open(bytes).expect("open superfile");
    assert_eq!(r.n_docs(), NAKED_N_DOCS);
    assert!(r.fts().is_none());
    assert!(r.vec().is_none());
    assert!(r.fts_columns().is_empty());
    assert!(r.vector_columns().is_empty());
    let p = r
        .parquet_bytes()
        .expect("eager open retains parquet bytes")
        .clone();
    let builder = ParquetRecordBatchReaderBuilder::try_new(p)
        .expect("try_new ParquetRecordBatchReaderBuilder");
    let mut reader = builder.build().expect("build parquet reader");
    let read = reader.next().expect("batch").expect("decode batch");
    assert_eq!(read.num_rows(), NAKED_N_DOCS as usize);
}

#[tokio::test]
async fn end_to_end_fts_only_blob_offsets_within_file() {
    // Sanity: when FTS is present and vectors absent, the vec keys
    // are absent and FTS keys point inside the file.
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let ids = decimal128_ids(vec![1u64, 2]);
    let titles = LargeStringArray::from(vec!["alpha", "beta"]);
    let bodies = LargeStringArray::from(vec!["x", "y"]);
    let scores = Float32Array::from(vec![1.0, 2.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let r = SuperfileReader::open(bytes).expect("open superfile");
    assert!(r.fts().is_some());
    assert!(r.vec().is_none());
    let hits = r
        .bm25_search("title", "alpha", SEARCH_K, BoolMode::Or)
        .await
        .expect("BM25 search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].0, 0);
}

#[tokio::test]
async fn end_to_end_three_batches_doc_ids_continuous() {
    // Splitting input into multiple add_batch calls must keep
    // local_doc_id sequential across batches.
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    for chunk in 0..MULTI_BATCH_CHUNK_COUNT {
        let ids = decimal128_ids(vec![
            chunk * MULTI_BATCH_ID_STRIDE,
            chunk * MULTI_BATCH_ID_STRIDE + 1,
        ]);
        let titles = LargeStringArray::from(vec![
            format!("t{} alpha", chunk),
            format!("t{} beta", chunk),
        ]);
        let bodies = LargeStringArray::from(vec!["x", "y"]);
        let scores = Float32Array::from(vec![Some(1.0), Some(2.0)]);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(ids),
                Arc::new(titles),
                Arc::new(bodies),
                Arc::new(scores),
            ],
        )
        .expect("build RecordBatch");
        b.add_batch(&batch, &[]).expect("add_batch");
    }
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let r = SuperfileReader::open(bytes).expect("open superfile");
    assert_eq!(r.n_docs(), N_DOCS as u64);
    let hits = r
        .bm25_search("title", "alpha", MULTI_TERM_K, BoolMode::Or)
        .await
        .expect("BM25 search");
    // alpha appears at local_doc_ids 0, 2, 4 (one per chunk).
    let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert!(doc_ids.contains(&0));
    assert!(doc_ids.contains(&2));
    assert!(doc_ids.contains(&4));
}

#[test]
fn add_batch_from_reader_mergeability_compatible_superfiles() {
    // Both superfiles have identical configurations; merge succeeds.
    let bytes1 = build_pipeline_superfile();
    let r1 = SuperfileReader::open(bytes1).expect("open superfile 1");

    let bytes2 = build_pipeline_superfile();
    let r2 = SuperfileReader::open(bytes2).expect("open superfile 2");

    let opts = BuilderOptions::new(
        r1.schema().clone(),
        r1.id_column(),
        vec![
            FtsConfig {
                column: "title".into(),
            },
            FtsConfig {
                column: "body".into(),
            },
        ],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    // Should not error on compatible configuration
    b.add_batch_from_reader(&r1, None)
        .expect("add_batch_from_reader succeeds for compatible superfiles");
    b.add_batch_from_reader(&r2, None)
        .expect("add_batch_from_reader succeeds for compatible superfiles");
}

#[test]
fn add_batch_from_reader_mergeability_id_column_mismatch() {
    // Build superfile with "different_id" as id column
    let schema_with_alt_id = Arc::new(Schema::new(vec![
        Field::new("different_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("score", DataType::Float32, true),
    ]));
    let opts = BuilderOptions::new(
        schema_with_alt_id.clone(),
        "different_id",
        vec![],
        vec![],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema_with_alt_id,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Build a builder with "doc_id" as id column using a matching schema
    // Map the reader's schema: rename "different_id" to "doc_id" for the builder
    let builder_schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
        Field::new("score", DataType::Float32, true),
    ]));
    let orig_opts = BuilderOptions::new(builder_schema, "doc_id", vec![], vec![], None);
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for id column mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_schema_mismatch() {
    // Build superfile with schema missing a column
    let schema_short = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
        // missing "body" column compared to pipeline_schema
        Field::new("score", DataType::Float32, true),
    ]));
    let opts = BuilderOptions::new(schema_short.clone(), "doc_id", vec![], vec![], None);
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema_short,
        vec![Arc::new(ids), Arc::new(titles), Arc::new(scores)],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with full pipeline schema
    let orig_schema = pipeline_schema();
    let orig_opts = BuilderOptions::new(orig_schema, "doc_id", vec![], vec![], None);
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for schema mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_fts_column_count_mismatch() {
    // Build superfile with FTS on only one column
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with FTS on two columns
    let orig_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![
            FtsConfig {
                column: "title".into(),
            },
            FtsConfig {
                column: "body".into(),
            },
        ],
        vec![],
        Some(default_tokenizer()),
    );
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for FTS column count mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_fts_column_name_mismatch() {
    // Build superfile with FTS on "body"
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "body".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");
    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with FTS on "title"
    let orig_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for FTS column name mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_vector_column_count_mismatch() {
    // Build superfile with one vector column
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    let mut emb = vec![0.0f32; 16];
    emb[0] = 1.0;
    normalize(&mut emb);
    b.add_batch(&batch, &[emb.as_slice()]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with no vector columns
    let orig_opts = BuilderOptions::new(pipeline_schema(), "doc_id", vec![], vec![], None);
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for vector column count mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_vector_column_name_mismatch() {
    // Build superfile with vector column "emb"
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    let mut emb = vec![0.0f32; 16];
    emb[0] = 1.0;
    normalize(&mut emb);
    b.add_batch(&batch, &[emb.as_slice()]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with different vector column name
    let orig_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "other_vec".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for vector column name mismatch"
    );
}

#[test]
fn add_batch_from_reader_mergeability_vector_dimension_mismatch() {
    // Build superfile with vector dim=16
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![1u64]);
    let titles = LargeStringArray::from(vec!["test"]);
    let bodies = LargeStringArray::from(vec!["test"]);
    let scores = Float32Array::from(vec![1.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    let mut emb = vec![0.0f32; 16];
    emb[0] = 1.0;
    normalize(&mut emb);
    b.add_batch(&batch, &[emb.as_slice()]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Try to merge into builder with different dimension
    let orig_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: 32, // different dimension
            n_cent: 4,
            rot_seed: 17,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut orig_builder = SuperfileBuilder::new(orig_opts).expect("new SuperfileBuilder");

    let err = orig_builder.add_batch_from_reader(&reader, None);
    assert!(
        err.is_err(),
        "expected mergeability error for vector dimension mismatch"
    );
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_excludes_records() {
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102]);
    let titles = LargeStringArray::from(vec!["doc0", "doc1", "doc2"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1", "body2"]);
    let scores = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Create bitmap to delete documents at local_doc_id 1 (doc_id=101)
    let mut deleted = roaring::RoaringBitmap::new();
    deleted.insert(1);

    // Merge with deleted_docs_bitmap
    let builder2_opts = BuilderOptions::new(pipeline_schema(), "doc_id", vec![], vec![], None);
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with deleted docs");

    let bytes2 = Bytes::from(builder2.finish().expect("finish builder"));
    let reader2 = SuperfileReader::open(bytes2).expect("open merged superfile");

    // The merged superfile should only have 2 docs (doc 0 and doc 2)
    assert_eq!(
        reader2.n_docs(),
        2,
        "Expected 2 documents after filtering 1 deletion"
    );

    let parquet = reader2
        .parquet_bytes()
        .expect("eager open retains parquet bytes")
        .clone();
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet)
        .expect("try_new ParquetRecordBatchReaderBuilder");
    let mut parquet_reader = builder.build().expect("build parquet reader");
    let read_batch = parquet_reader
        .next()
        .expect("at least one batch")
        .expect("decode batch");

    assert_eq!(read_batch.num_rows(), 2, "Parquet batch should have 2 rows");

    // Verify the correct documents remain
    let ids_array = read_batch
        .column(0)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .expect("downcast to Decimal128Array");
    let id_vals: Vec<i128> = (0..ids_array.len()).map(|i| ids_array.value(i)).collect();
    assert_eq!(id_vals, vec![100, 102], "Expected doc_ids 100 and 102");
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_excludes_fts() {
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102]);
    let titles = LargeStringArray::from(vec!["rust async", "python data", "rust embedded"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1", "body2"]);
    let scores = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Create bitmap to delete document at local_doc_id 1 (python data)
    let mut deleted = roaring::RoaringBitmap::new();
    deleted.insert(1);

    let builder2_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![],
        Some(default_tokenizer()),
    );
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with deleted docs");

    let bytes2 = Bytes::from(builder2.finish().expect("finish builder"));
    let reader2 = SuperfileReader::open(bytes2).expect("open merged superfile");

    assert_eq!(
        reader2.n_docs(),
        2,
        "Expected 2 documents after FTS filtering"
    );

    // Search for "rust" should find 2 docs (original doc 0 and 2, now local_doc_ids 0 and 1)
    let hits = futures::executor::block_on(async {
        reader2
            .bm25_search("title", "rust", SEARCH_K, BoolMode::Or)
            .await
            .expect("BM25 search")
    });

    assert_eq!(hits.len(), 2, "Expected 2 rust docs");
    let local_doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    // After filtering and merging, local_doc_ids are 0 and 1
    assert_eq!(
        local_doc_ids,
        vec![0, 1].into_iter().collect(),
        "Expected local_doc_ids 0 and 1"
    );

    // Search for "python" should find 0 docs (it was deleted)
    let python_hits = futures::executor::block_on(async {
        reader2
            .bm25_search("title", "python", SEARCH_K, BoolMode::Or)
            .await
            .expect("BM25 search")
    });

    assert_eq!(
        python_hits.len(),
        0,
        "Expected 0 python docs (it was deleted)"
    );
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_excludes_vectors() {
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102]);
    let titles = LargeStringArray::from(vec!["doc0", "doc1", "doc2"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1", "body2"]);
    let scores = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    // Create 3 unit-norm vectors with distinct axes
    let mut flat = Vec::<f32>::with_capacity(3 * EMB_DIM);
    let axes: [usize; 3] = [0, 1, 2];
    for &a in &axes {
        let mut v = vec![0.0f32; EMB_DIM];
        v[a] = 1.0;
        v[(a + 1) % EMB_DIM] = SECONDARY_AXIS_WEIGHT;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Create bitmap to delete document at local_doc_id 1 (axis 1)
    let mut deleted = roaring::RoaringBitmap::new();
    deleted.insert(1);

    let builder2_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        None,
    );
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with deleted docs");

    let bytes2 = Bytes::from(builder2.finish().expect("finish builder"));
    let reader2 = SuperfileReader::open(bytes2).expect("open merged superfile");

    assert_eq!(
        reader2.n_docs(),
        2,
        "Expected 2 documents after vector filtering"
    );

    // Query with a vector aligned to axis 0 (doc 0's axis)
    let mut q = vec![0.0f32; EMB_DIM];
    q[0] = 1.0;
    q[1] = SECONDARY_AXIS_WEIGHT;
    normalize(&mut q);

    let hits = futures::executor::block_on(async {
        reader2
            .vector_search("emb", &q, 2, VectorSearchOptions::new().with_nprobe(NPROBE))
            .await
            .expect("vector search")
    });

    assert_eq!(hits.len(), 2, "Expected 2 results from vector search");
    // After filtering, doc 0 is local_doc_id 0, doc 2 is local_doc_id 1
    assert_eq!(
        hits[0].0, 0,
        "Top result should be local_doc_id 0 (original doc 0)"
    );
    let local_doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert_eq!(
        local_doc_ids,
        vec![0, 1].into_iter().collect(),
        "Expected local_doc_ids 0 and 1"
    );
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_all_deletes() {
    // Edge case: delete all documents
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102]);
    let titles = LargeStringArray::from(vec!["doc0", "doc1", "doc2"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1", "body2"]);
    let scores = Float32Array::from(vec![1.0, 2.0, 3.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Create bitmap to delete all documents
    let mut deleted = roaring::RoaringBitmap::new();
    deleted.insert(0);
    deleted.insert(1);
    deleted.insert(2);

    let builder2_opts = BuilderOptions::new(pipeline_schema(), "doc_id", vec![], vec![], None);
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with all docs deleted");

    let bytes2 = builder2.finish().expect("finish builder");
    // Empty superfile should return empty bytes
    assert!(bytes2.is_empty(), "All-deleted superfile should be empty");
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_no_deletes() {
    // Edge case: bitmap is provided but empty (no deletions)
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![], vec![], None);
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101]);
    let titles = LargeStringArray::from(vec!["doc0", "doc1"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1"]);
    let scores = Float32Array::from(vec![1.0, 2.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    b.add_batch(&batch, &[]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Create empty bitmap (no deletions)
    let deleted = roaring::RoaringBitmap::new();

    let builder2_opts = BuilderOptions::new(pipeline_schema(), "doc_id", vec![], vec![], None);
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with empty bitmap");

    let bytes2 = Bytes::from(builder2.finish().expect("finish builder"));
    let reader2 = SuperfileReader::open(bytes2).expect("open merged superfile");

    // All documents should be present
    assert_eq!(
        reader2.n_docs(),
        2,
        "Expected all 2 documents with empty deletion bitmap"
    );
}

#[test]
fn add_batch_from_reader_with_deleted_docs_bitmap_partial_deletes_mixed_indexes() {
    // Test with both FTS and vectors, deleting documents in between
    let schema = pipeline_schema();
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");

    let ids = decimal128_ids(vec![100u64, 101, 102, 103]);
    let titles = LargeStringArray::from(vec!["rust one", "python two", "rust three", "go four"]);
    let bodies = LargeStringArray::from(vec!["body0", "body1", "body2", "body3"]);
    let scores = Float32Array::from(vec![1.0, 2.0, 3.0, 4.0]);
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(titles),
            Arc::new(bodies),
            Arc::new(scores),
        ],
    )
    .expect("build RecordBatch");

    // Create 4 distinct vectors
    let mut flat = Vec::<f32>::with_capacity(4 * EMB_DIM);
    for a in 0..4 {
        let mut v = vec![0.0f32; EMB_DIM];
        v[a] = 1.0;
        normalize(&mut v);
        flat.extend_from_slice(&v);
    }
    b.add_batch(&batch, &[flat.as_slice()]).expect("add_batch");
    let bytes = Bytes::from(b.finish().expect("finish builder"));
    let reader = SuperfileReader::open(bytes).expect("open superfile");

    // Delete docs 1 and 3 (indices 1 and 3)
    let mut deleted = roaring::RoaringBitmap::new();
    deleted.insert(1);
    deleted.insert(3);

    let builder2_opts = BuilderOptions::new(
        pipeline_schema(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![SfVectorConfig {
            column: "emb".into(),
            dim: EMB_DIM,
            n_cent: N_CENT,
            rot_seed: ROT_SEED,
            metric: Metric::Cosine,
            rerank_codec: RerankCodec::Fp32,
        }],
        Some(default_tokenizer()),
    );
    let mut builder2 = SuperfileBuilder::new(builder2_opts).expect("new SuperfileBuilder");

    builder2
        .add_batch_from_reader(&reader, Some(Arc::new(deleted)))
        .expect("add_batch_from_reader with partial deletes");

    let bytes2 = Bytes::from(builder2.finish().expect("finish builder"));
    let reader2 = SuperfileReader::open(bytes2).expect("open merged superfile");

    // Should have 2 documents (indices 0 and 2)
    assert_eq!(
        reader2.n_docs(),
        2,
        "Expected 2 documents after deleting 1 and 3"
    );

    // FTS: search for "rust" should find 2 docs (original docs 0 and 2)
    let hits = futures::executor::block_on(async {
        reader2
            .bm25_search("title", "rust", SEARCH_K, BoolMode::Or)
            .await
            .expect("BM25 search")
    });
    assert_eq!(hits.len(), 2, "Expected 2 rust docs");
    let local_doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    // After filtering out docs 1 and 3, only docs 0 and 2 remain (reassigned as 0 and 1)
    assert_eq!(
        local_doc_ids,
        vec![0, 1].into_iter().collect(),
        "Expected local_doc_ids 0 and 1"
    );

    // Vector search: search with axis-0 vector should find doc 0 (highest relevance)
    let mut q = vec![0.0f32; EMB_DIM];
    q[0] = 1.0;
    normalize(&mut q);
    let vec_hits = futures::executor::block_on(async {
        reader2
            .vector_search("emb", &q, 2, VectorSearchOptions::new().with_nprobe(NPROBE))
            .await
            .expect("vector search")
    });
    assert_eq!(vec_hits.len(), 2, "Expected 2 vector results");
    assert_eq!(
        vec_hits[0].0, 0,
        "Top result should be local_doc_id 0 (original doc 0)"
    );
}
