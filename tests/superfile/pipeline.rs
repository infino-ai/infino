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

fn pipeline_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
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
            dim: 16,
            n_cent: 4,
            rot_seed: 17,
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
    let mut flat = Vec::<f32>::with_capacity(6 * 16);
    let axes: [usize; 6] = [0, 1, 0, 2, 3, 0];
    for &a in &axes {
        let mut v = vec![0.0f32; 16];
        v[a] = 1.0;
        v[(a + 1) % 16] = 0.1;
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
    assert_eq!(r.n_docs(), 6);
    assert_eq!(r.id_column(), "doc_id");
    assert_eq!(r.fts_columns(), vec!["title", "body"]);
    assert_eq!(r.vector_columns(), vec!["emb"]);
    assert_eq!(r.schema().fields().len(), 4);
}

#[tokio::test]
async fn end_to_end_bm25_finds_rust_docs() {
    let bytes = build_pipeline_superfile();
    let r = SuperfileReader::open(bytes).expect("open superfile");
    let hits = r
        .bm25_search("title", "rust", 5, BoolMode::Or)
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
            5,
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
    let mut q = vec![0.0f32; 16];
    q[3] = 1.0;
    q[4] = 0.1;
    normalize(&mut q);
    let hits = r
        .vector_search("emb", &q, 1, VectorSearchOptions::new().with_nprobe(4))
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
    assert_eq!(batch.num_rows(), 6);
    assert_eq!(batch.num_columns(), 4);
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
    assert_eq!(r.n_docs(), 2);
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
    assert_eq!(read.num_rows(), 2);
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
        .bm25_search("title", "alpha", 5, BoolMode::Or)
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
    for chunk in 0..3u64 {
        let ids = decimal128_ids(vec![chunk * 10, chunk * 10 + 1]);
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
    assert_eq!(r.n_docs(), 6);
    let hits = r
        .bm25_search("title", "alpha", 10, BoolMode::Or)
        .await
        .expect("BM25 search");
    // alpha appears at local_doc_ids 0, 2, 4 (one per chunk).
    let doc_ids: std::collections::HashSet<u32> = hits.iter().map(|(d, _)| *d).collect();
    assert!(doc_ids.contains(&0));
    assert!(doc_ids.contains(&2));
    assert!(doc_ids.contains(&4));
}
