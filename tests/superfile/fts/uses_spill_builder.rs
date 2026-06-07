// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Production-wiring integration test for the spill-backed FTS
//! builder. Confirms that `SuperfileBuilder::finish` flows through
//! `FtsBuilder::finish` → `finish_to`, exercising the partition-
//! spill path on a corpus large enough to populate multiple
//! partitions.
//!
//! The bar this guards: the on-disk superfile must be byte-for-
//! byte readable by `SuperfileReader` with BM25 search returning
//! the expected docs. If a refactor accidentally bypasses the
//! spill pipeline (e.g. reintroduces an in-memory shortcut at the
//! `SuperfileBuilder` layer), the build will still likely
//! succeed, but this test will catch behavioural regressions in
//! the production path used by `SupertableWriter`.

use arrow_array::{LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::superfile::SuperfileReader;
use infino::superfile::builder::{BuilderOptions, FtsConfig, SuperfileBuilder};
use infino::superfile::fts::reader::BoolMode;
use infino::test_helpers::{decimal128_ids, default_tokenizer};
use std::sync::Arc;

/// Decimal128 precision / scale for the `doc_id` column.
const ID_DECIMAL_PRECISION: u8 = 38;
const ID_DECIMAL_SCALE: i8 = 0;
/// Corpus size, chosen to populate multiple hash-partition spill files.
const SPILL_TEST_N_DOCS: usize = 1024;
/// FTS spill threshold override (1 byte) forcing every column to spill
/// on the first doc, exercising the production spill pipeline.
const SPILL_FORCE_THRESHOLD_BYTES: usize = 1;
/// BM25 top-k for the spill-path searches.
const SPILL_TEST_BM25_K: usize = 10;

fn schema_for_test() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "doc_id",
            DataType::Decimal128(ID_DECIMAL_PRECISION, ID_DECIMAL_SCALE),
            false,
        ),
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("body", DataType::LargeUtf8, false),
    ]))
}

/// 1024-doc superfile with two FTS columns. Each doc has a unique
/// `term{i}` plus the shared `common` token. 1024 docs across two
/// columns is enough that hash-partitioned spill files (default
/// 128 per column) all have multiple records on average, so the
/// production `add_doc → spill → finish_to` path is exercised end to
/// end rather than collapsing into a single-partition shortcut.
fn build_test_superfile() -> Bytes {
    let n_docs: usize = SPILL_TEST_N_DOCS;

    let schema = schema_for_test();
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
        Vec::new(),
        Some(default_tokenizer()),
    );
    let mut b = SuperfileBuilder::new(opts).expect("new SuperfileBuilder");
    // Force the spill + streaming-FST finish path. 1024 docs at
    // ~50 bytes/doc/column never crosses the default 256 MiB
    // threshold, so without this override the test name
    // ("uses_spill_builder") would be misleading — every column
    // would stay InRam and the production spill pipeline would
    // never be exercised. `1` is the minimum value the setter
    // accepts (`> 0` is enforced in `FtsBuilder`) and is well below
    // the first `add_doc`'s accumulator size, so every column
    // transitions to Spilled on the first doc.
    b.set_fts_spill_threshold_bytes(SPILL_FORCE_THRESHOLD_BYTES);

    let ids = decimal128_ids(0..n_docs as u64);
    let titles_owned: Vec<String> = (0..n_docs)
        .map(|i| format!("common title term{i:04}"))
        .collect();
    let bodies_owned: Vec<String> = (0..n_docs)
        .map(|i| format!("body common payload{i:04} extra word{i:04}"))
        .collect();
    let titles = LargeStringArray::from(
        titles_owned
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>(),
    );
    let bodies = LargeStringArray::from(
        bodies_owned
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<&str>>(),
    );

    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(titles), Arc::new(bodies)],
    )
    .expect("build record batch");
    b.add_batch(&batch, &[]).expect("add batch");

    Bytes::from(b.finish().expect("finish superfile"))
}

#[tokio::test]
async fn superfile_build_routes_through_spill_backed_fts_builder() {
    let blob = build_test_superfile();

    // The blob is a valid Parquet file with embedded FTS + (empty)
    // vector pointers; SuperfileReader is the production read path.
    let r = SuperfileReader::open(blob).expect("open SuperfileReader");

    // The reader exposes BM25 search directly via `bm25_search`;
    // its presence confirms the FTS blob round-tripped.
    assert!(r.fts().is_some(), "superfile must have FTS blob");

    // BM25 on a term that appears in every doc must hit at least
    // `k` docs. With k=10 across 1024 docs the floor is exactly 10.
    let common_hits = r
        .bm25_search("title", "common", SPILL_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("search common");
    assert_eq!(
        common_hits.len(),
        SPILL_TEST_BM25_K,
        "common term must hit at least k=10 docs"
    );

    // A unique term that appears in exactly one body must return
    // exactly that doc. This is the cross-column-isolation gate:
    // the body column gets `payload0500` only at row 500; the title
    // column never has it.
    let payload_hits = r
        .bm25_search("body", "payload0500", SPILL_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("search payload0500");
    assert_eq!(
        payload_hits.len(),
        1,
        "payload0500 is unique to body row 500"
    );

    // Negative: a body-only term must not appear in a title search,
    // and vice versa. Production path keeps per-column FST keys
    // (`<col>\x1F<term>`) so this scopes correctly.
    let payload_in_title = r
        .bm25_search("title", "payload0500", SPILL_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("search payload0500 in title");
    assert!(
        payload_in_title.is_empty(),
        "payload0500 lives only in body column"
    );

    let term_in_body = r
        .bm25_search("body", "term0500", SPILL_TEST_BM25_K, BoolMode::Or)
        .await
        .expect("search term0500 in body");
    assert!(
        term_in_body.is_empty(),
        "term0500 lives only in title column"
    );
}
