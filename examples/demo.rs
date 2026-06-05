//! Smallest possible end-to-end tour of the stack:
//!
//!   A. superfile  — feed one document, then BM25 *and* vector kNN
//!      against the indexes embedded in the same bytes.
//!   B. open-format — those same bytes are a valid Parquet file;
//!      vanilla DataFusion reads them as a plain table.
//!   C. supertable — the cross-segment layer that auto-injects a
//!      real `_id`, queried across two committed segments.
//!
//! Run with:
//! ```text
//! cargo run --example demo
//! ```

use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::prelude::*;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder, VectorConfig};
use infino::superfile::fts::reader::BoolMode;
use infino::superfile::vector::distance::{Metric, normalize};
use infino::superfile::{SuperfileReader, VectorSearchOptions};
use infino::supertable::Supertable;
use infino::test_helpers::{
    build_title_batch, decimal128_ids, default_supertable_options, default_tokenizer,
};
use tempfile::NamedTempFile;

const DOCSTRING_1: &str = "the quick brown fox";
const DOCSTRING_2: &str = "a lazy sleeping fox";
const DOC_ID_1: u64 = 42;
const EMB_DIM: usize = 16;

fn main() {
    let bytes = demo_superfile();
    demo_datafusion(&bytes);
    demo_supertable();
}

/// A — build a one-doc superfile with both an FTS column and a
/// vector column, then query each embedded index.
fn demo_superfile() -> Bytes {
    println!("== A. superfile: one doc, BM25 + vector kNN ==");

    // doc_id (the required Decimal128 primary key) + one text column.
    let schema = Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::Decimal128(38, 0), false),
        Field::new("title", DataType::LargeUtf8, false),
    ]));

    // `emb` is a vector column: a logical name only — its f32s are
    // passed to add_batch separately and never enter the Parquet
    // schema. n_cent=1 because a single doc yields one IVF cluster.
    let opts = BuilderOptions::new(
        schema.clone(),
        "doc_id",
        vec![FtsConfig {
            column: "title".into(),
        }],
        vec![VectorConfig::new(
            "emb".into(),
            EMB_DIM,
            1,
            7,
            Metric::Cosine,
        )],
        Some(default_tokenizer()),
    );

    let mut builder = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(decimal128_ids(vec![DOC_ID_1])),
            Arc::new(LargeStringArray::from(vec![DOCSTRING_1])),
        ],
    )
    .expect("build RecordBatch");

    // One unit-norm embedding for the single doc.
    let mut emb = vec![0.0f32; EMB_DIM];
    emb[0] = 1.0;
    normalize(&mut emb);

    builder
        .add_batch(&batch, &[emb.as_slice()])
        .expect("add_batch");
    let bytes: Bytes = builder.finish().expect("finish").into();
    println!("built superfile: {} bytes", bytes.len());

    let reader = SuperfileReader::open(bytes.clone()).expect("open SuperfileReader");

    // The per-segment `SuperfileReader` query kernels are async; drive
    // them on a throwaway runtime here (the supertable layer below
    // exposes the sync public API instead).
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        // BM25 over the embedded FTS blob. Hits are (local_doc_id, score).
        for q in ["brown", "fox", "missing"] {
            let hits = reader
                .bm25_search("title", q, 10, BoolMode::Or)
                .await
                .expect("bm25_search");
            println!("  bm25 {q:>8?} -> {} hit(s): {hits:?}", hits.len());
        }

        // kNN over the embedded vector blob. Query with the doc's own
        // embedding -> distance ~0 under cosine.
        let knn = reader
            .vector_search("emb", &emb, 10, VectorSearchOptions::default())
            .await
            .expect("vector_search");
        println!("  knn  self-query -> {} hit(s): {knn:?}", knn.len());
    });
    println!();

    bytes
}

/// B — the very same bytes open as a plain Parquet table. DataFusion
/// uses an independent reader and ignores the embedded blobs + the
/// `inf.*` KV metadata, per the Parquet spec.
fn demo_datafusion(bytes: &Bytes) {
    println!("== B. open-format: DataFusion reads the same bytes ==");

    let f = NamedTempFile::with_suffix(".parquet").expect("tempfile");
    std::fs::write(f.path(), bytes).expect("write tempfile");

    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let ctx = SessionContext::new();
        ctx.register_parquet(
            "docs",
            f.path().to_str().expect("utf8 path"),
            ParquetReadOptions::default(),
        )
        .await
        .expect("register superfile as a Parquet table");

        let df = ctx
            .sql("SELECT doc_id, title FROM docs")
            .await
            .expect("plan SQL");
        let batches = df.collect().await.expect("execute SQL");
        let table = pretty_format_batches(&batches).expect("format");
        for line in table.to_string().lines() {
            println!("  {line}");
        }
    });
    println!();
}

/// C — the supertable layer. It auto-injects a snowflake `_id`, so
/// the user schema carries only payload columns. Two commits produce
/// two segments; BM25 fans out across both, and SQL surfaces the
/// real `_id` values.
fn demo_supertable() {
    println!("== C. supertable: cross-segment, auto-injected _id ==");

    // default_supertable_options(): schema is `title: LargeUtf8`,
    // FTS on title, no vectors, in-memory store. The `_id` column is
    // added by the supertable, not declared here.
    let st = Supertable::create(default_supertable_options()).expect("create supertable");

    // Each commit seals one segment. The writer holds an exclusive
    // slot on the supertable, so we scope it so it drops before the
    // next one is taken.
    for title in [DOCSTRING_1, DOCSTRING_2] {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(&[title])).expect("append");
        w.commit().expect("commit");
    }

    // BM25 across both segments. SuperfileHit carries the source
    // segment + local_doc_id + score.
    let hits = st
        .bm25_search("title", "fox", 10, BoolMode::Or)
        .expect("bm25 fan-out");
    println!("  bm25 \"fox\" across segments -> {} hit(s)", hits.len());
    for h in &hits {
        println!("    local_doc_id={} score={:.4}", h.local_doc_id, h.score);
    }

    // SQL surfaces the real auto-injected `_id` alongside the payload.
    let batches = st
        .query_sql("SELECT _id, title FROM supertable ORDER BY _id")
        .expect("query_sql");
    let table = pretty_format_batches(&batches).expect("format");
    for line in table.to_string().lines() {
        println!("  {line}");
    }
}
