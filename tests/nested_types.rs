// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Nested Arrow column types through the local engine: a table carrying a
//! struct, a list of structs, and a timestamp alongside a searchable text
//! column, exercised across append, SQL, reopen, and maintenance.
//!
//! The nested columns are payload: no index targets them (full-text needs a
//! large-string column, vector search needs a fixed-size float list), so these
//! tests pin that nested data is stored, queryable, and survives a rewrite --
//! not that it is searchable.

#![deny(clippy::unwrap_used)]

use std::{io::Cursor, sync::Arc, time::Duration};

use arrow::json::ReaderBuilder;
use infino::{
    Bm25SearchOptions, BoolMode, Connection, IndexSpec, OptimizeOptions,
    arrow_array::{Array, Int64Array, ListArray, RecordBatch, StructArray},
    arrow_schema::{DataType, Field, Fields, Schema, SchemaRef, TimeUnit},
    connect,
};
use tempfile::TempDir;

/// The table name every test in this file creates.
const TABLE: &str = "docs";

/// First append payload, newline-delimited JSON. One row has a null struct so
/// nested nullability is covered from the start.
const FIRST_ROWS: &str = r#"{"title":"the quick brown fox","image":{"content_url":"https://example.test/fox.png","height":480},"entities":[{"identifier":"Q1","url":"https://example.test/Q1"},{"identifier":"Q2","url":"https://example.test/Q2"}],"updated":"2026-01-02T03:04:05Z"}
{"title":"a lazy sleeping dog","image":null,"entities":[],"updated":null}"#;

/// Second append payload -- a separate commit, so table maintenance has two
/// superfiles to merge.
const SECOND_ROWS: &str = r#"{"title":"a red clever fox","image":{"content_url":"https://example.test/red.png","height":720},"entities":[{"identifier":"Q3","url":"https://example.test/Q3"}],"updated":"2026-02-03T04:05:06Z"}"#;

/// Rows committed across both appends.
const TOTAL_ROWS: usize = 3;

/// Rows in the fixtures whose `image` struct is null.
const FIXTURE_NULL_IMAGES: usize = 1;

/// Tallest `image.height` across the fixtures.
const TALLEST_IMAGE_HEIGHT: i64 = 720;

/// Entities across both fixtures: two in the first row, none in the second,
/// one in the third.
const TOTAL_ENTITIES: i64 = 3;

/// Fixture rows whose title contains "fox".
const FOX_ROWS: usize = 2;

/// Height threshold that admits only the taller image row.
const MIN_IMAGE_HEIGHT: i64 = 500;

/// Fixture rows with an image taller than `MIN_IMAGE_HEIGHT`.
const TALL_IMAGE_ROWS: usize = 1;

/// Top-K for the text search; larger than the corpus so ranking, not
/// truncation, decides the assertion.
const SEARCH_TOP_K: usize = 10;

/// Child fields of the `image` struct column.
fn image_fields() -> Fields {
    Fields::from(vec![
        Field::new("content_url", DataType::Utf8, true),
        Field::new("height", DataType::Int64, true),
    ])
}

/// Child fields of each entity in the `entities` list.
fn entity_fields() -> Fields {
    Fields::from(vec![
        Field::new("identifier", DataType::Utf8, true),
        Field::new("url", DataType::Utf8, true),
    ])
}

/// The table schema: a searchable text column beside nested payload.
fn nested_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("image", DataType::Struct(image_fields()), true),
        Field::new(
            "entities",
            DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(entity_fields()),
                true,
            ))),
            true,
        ),
        Field::new(
            "updated",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
    ]))
}

/// Create the table and commit both batches, returning the open connection.
/// Two appends means two commits, so table maintenance has real work to do.
fn seeded_table(dir: &TempDir) -> Connection {
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = nested_schema();
    let docs = db
        .create_table(TABLE, schema.clone(), IndexSpec::new().fts("title"))
        .expect("create_table");
    docs.append(&batch_from_json(schema.clone(), FIRST_ROWS))
        .expect("append 1");
    docs.append(&batch_from_json(schema, SECOND_ROWS))
        .expect("append 2");
    db
}

/// The single scalar value in a one-row, one-column result.
fn only_i64(batches: &[RecordBatch]) -> i64 {
    batches
        .iter()
        .filter(|b| b.num_rows() > 0)
        .map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("column is int64")
                .value(0)
        })
        .next()
        .expect("a row")
}

/// One record batch from newline-delimited JSON, decoded against `schema`.
fn batch_from_json(schema: SchemaRef, rows: &str) -> RecordBatch {
    let mut reader = ReaderBuilder::new(schema)
        .build(Cursor::new(rows.as_bytes()))
        .expect("json reader");
    reader
        .next()
        .expect("at least one batch")
        .expect("decodable batch")
}

#[test]
fn struct_column_round_trips_through_append_and_sql() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");

    let schema = nested_schema();
    let docs = db
        .create_table(TABLE, schema.clone(), IndexSpec::new().fts("title"))
        .expect("create_table accepts a nested schema");

    let first = batch_from_json(schema.clone(), FIRST_ROWS);

    // Guard the fixture itself. If the JSON decoder does not actually produce a
    // null struct, the null assertion at the end of this test would pass or
    // fail for a reason that has nothing to do with the engine.
    let fixture_idx = first.schema().index_of("image").expect("image in fixture");
    assert_eq!(
        first.column(fixture_idx).null_count(),
        FIXTURE_NULL_IMAGES,
        "the fixture must carry a null struct before it is appended"
    );

    docs.append(&first)
        .expect("append batch with a struct column");
    docs.append(&batch_from_json(schema.clone(), SECOND_ROWS))
        .expect("append second batch");

    // The declared schema comes back with the struct intact, not flattened or
    // rewritten to text.
    let described = docs.schema();
    let image = described
        .field_with_name("image")
        .expect("image column present");
    assert!(
        matches!(image.data_type(), DataType::Struct(_)),
        "image must stay a struct, got {:?}",
        image.data_type()
    );

    // A full scan returns the nested column as structured data.
    let scanned = db
        .query_sql(&format!("SELECT title, image FROM {TABLE}"))
        .expect("query_sql over a nested column");
    let rows: usize = scanned.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, TOTAL_ROWS, "every appended row must come back");

    let batch = scanned
        .iter()
        .find(|b| b.num_rows() > 0)
        .expect("a non-empty batch");
    let idx = batch.schema().index_of("image").expect("image projected");
    let column = batch.column(idx);
    assert!(
        column.as_any().downcast_ref::<StructArray>().is_some(),
        "image must decode as a StructArray, got {:?}",
        column.data_type()
    );

    // Nulls are counted across every returned batch: one commit becomes one
    // superfile, so the scan can hand back the rows in more than one batch and
    // the null row is not necessarily in the first.
    let nulls: usize = scanned
        .iter()
        .filter(|b| b.num_rows() > 0)
        .map(|b| {
            let i = b.schema().index_of("image").expect("image projected");
            b.column(i).null_count()
        })
        .sum();
    assert_eq!(
        nulls, FIXTURE_NULL_IMAGES,
        "the null struct row must survive as null"
    );
}

#[test]
fn list_of_structs_and_timestamp_round_trip() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");

    let schema = nested_schema();
    let docs = db
        .create_table(TABLE, schema.clone(), IndexSpec::new().fts("title"))
        .expect("create_table accepts a list-of-structs and a timestamp");

    docs.append(&batch_from_json(schema.clone(), FIRST_ROWS))
        .expect("append batch with a list of structs");
    docs.append(&batch_from_json(schema, SECOND_ROWS))
        .expect("append second batch");

    let described = docs.schema();
    let entities = described
        .field_with_name("entities")
        .expect("entities column present");
    match entities.data_type() {
        DataType::List(item) => assert!(
            matches!(item.data_type(), DataType::Struct(_)),
            "entities items must stay structs, got {:?}",
            item.data_type()
        ),
        other => panic!("entities must stay a list, got {other:?}"),
    }
    assert!(
        matches!(
            described
                .field_with_name("updated")
                .expect("updated column present")
                .data_type(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        ),
        "updated must stay a microsecond timestamp"
    );

    let scanned = db
        .query_sql(&format!("SELECT entities, updated FROM {TABLE}"))
        .expect("query_sql over a list of structs");
    let rows: usize = scanned.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, TOTAL_ROWS);

    let batch = scanned
        .iter()
        .find(|b| b.num_rows() > 0)
        .expect("a non-empty batch");
    let idx = batch.schema().index_of("entities").expect("projected");
    assert!(
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<ListArray>()
            .is_some(),
        "entities must decode as a ListArray, got {:?}",
        batch.column(idx).data_type()
    );
}

#[test]
fn sql_reaches_into_nested_values() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);

    // Struct field access. Spelled with brackets: the dotted form parses as a
    // qualified column reference (table `image`, column `height`) and fails to
    // plan, which looks like a nested-data failure but is not one.
    let heights = db
        .query_sql(&format!(
            "SELECT max(image['height']) AS tallest FROM {TABLE}"
        ))
        .expect("struct field access in SQL");
    assert_eq!(only_i64(&heights), TALLEST_IMAGE_HEIGHT);

    // Unnesting the list yields one row per entity across the table.
    let unnested = db
        .query_sql(&format!(
            "SELECT count(*) AS n FROM (SELECT unnest(entities) AS e FROM {TABLE})"
        ))
        .expect("unnest a list of structs in SQL");
    assert_eq!(only_i64(&unnested), TOTAL_ENTITIES);
}

#[test]
fn predicates_prune_correctly_with_nested_columns_present() {
    let dir = TempDir::new().expect("tempdir");
    let db = seeded_table(&dir);

    // A predicate on a scalar column, with nested columns in the same table.
    // This exercises the skip-summary path over a schema whose statistics the
    // collector has not seen before.
    let filtered = db
        .query_sql(&format!(
            "SELECT title FROM {TABLE} WHERE title LIKE '%fox%'"
        ))
        .expect("scalar predicate with nested columns present");
    let rows: usize = filtered.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, FOX_ROWS, "both fox rows must match");

    // A predicate on a nested field. If the engine cannot push this down it
    // should still answer correctly by scanning, so a wrong count is a pruning
    // bug and an error is a capability gap.
    let by_height = db
        .query_sql(&format!(
            "SELECT title FROM {TABLE} WHERE image['height'] > {MIN_IMAGE_HEIGHT}"
        ))
        .expect("predicate on a nested field");
    let tall: usize = by_height.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(tall, TALL_IMAGE_ROWS);

    // Full-text search still works on the top-level column while nested
    // columns sit beside it in the same superfiles.
    let docs = db.open_table(TABLE).expect("open_table");
    let hits = docs
        .bm25_search(
            "title",
            "fox",
            SEARCH_TOP_K,
            Bm25SearchOptions::new().with_mode(BoolMode::Or),
            None,
        )
        .expect("bm25_search with nested columns present");
    let hit_rows: usize = hits.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(
        hit_rows, FOX_ROWS,
        "the text index is unaffected by nesting"
    );
}

#[test]
fn nested_schema_survives_reopen_compaction_and_gc() {
    let dir = TempDir::new().expect("tempdir");
    {
        let db = seeded_table(&dir);
        drop(db);
    }

    // Fresh-process equivalent: a new connection recovers the schema from
    // storage rather than from the handle that created it.
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("reconnect");
    let docs = db.open_table(TABLE).expect("open_table after reopen");
    let described = docs.schema();
    assert_eq!(
        described.fields(),
        nested_schema().fields(),
        "the reopened schema must equal the declared one, nesting included"
    );

    // Compaction merges the two committed superfiles, rewriting the Parquet
    // body. Every row and every nested value must survive.
    docs.optimize(&OptimizeOptions::default())
        .expect("optimize a table with nested columns");
    let report = docs.gc(Duration::ZERO).expect("gc after optimize");
    assert_eq!(report.delete_errors, 0);

    let scanned = db
        .query_sql(&format!(
            "SELECT title, image, entities, updated FROM {TABLE}"
        ))
        .expect("query_sql after maintenance");
    let rows: usize = scanned.iter().map(RecordBatch::num_rows).sum();
    assert_eq!(rows, TOTAL_ROWS, "compaction must preserve every row");

    let tallest = db
        .query_sql(&format!("SELECT max(image['height']) AS t FROM {TABLE}"))
        .expect("nested values readable after compaction");
    assert_eq!(
        only_i64(&tallest),
        TALLEST_IMAGE_HEIGHT,
        "nested values must survive the rewrite unchanged"
    );
}
