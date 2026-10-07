// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Integration coverage for the public [`Supertable::hydrate`] bulk-load path.
//!
//! These stream a real Parquet file on disk straight into the public method and
//! check two things: the hydrated table answers SQL correctly, and it answers
//! identically to a table built the regular way (append, then optimize + gc).

#![deny(clippy::unwrap_used)]

use std::{fs::File, path::Path, sync::Arc, time::Duration};

use infino::{
    IndexSpec, OptimizeOptions,
    arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch, StringArray},
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use parquet::arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder};
use tempfile::TempDir;

/// Target rows per coalesced superfile for the tests.
const HYDRATE_TARGET_ROWS: usize = 64;
/// Rows per batch the Parquet reader hands out: half the target, so two reader
/// batches coalesce into each superfile.
const READ_BATCH_ROWS: usize = 32;

/// The user schema, no `_id` (the supertable mints and prepends it).
fn user_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]))
}

/// A batch of `n = lo..=hi` and `s = "r<n>"`.
fn rows_batch(lo: i64, hi: i64) -> RecordBatch {
    let n = Int64Array::from((lo..=hi).collect::<Vec<_>>());
    let s = StringArray::from((lo..=hi).map(|i| format!("r{i}")).collect::<Vec<_>>());
    RecordBatch::try_new(user_schema(), vec![Arc::new(n), Arc::new(s)]).expect("valid batch")
}

/// Write one batch as a single-row-group Parquet file.
fn write_parquet(path: &Path, batch: &RecordBatch) {
    let file = File::create(path).expect("create parquet");
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).expect("arrow writer");
    writer.write(batch).expect("write row group");
    writer.close().expect("close parquet");
}

/// Hydrate a Parquet file into table `name` through the public API: create a
/// no-index table and stream the file's reader straight into
/// [`Supertable::hydrate`] (no optimize, no GC). Returns the rows committed.
fn hydrate_parquet(db: &infino::Connection, name: &str, parquet_path: &Path) -> usize {
    let table = db
        .create_table(name, user_schema(), IndexSpec::new())
        .expect("create_table");
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path).expect("open"))
        .expect("parquet reader builder")
        .with_batch_size(READ_BATCH_ROWS)
        .build()
        .expect("parquet reader");
    table.hydrate(reader, HYDRATE_TARGET_ROWS).expect("hydrate")
}

/// Render a query result as `|`-joined rows, in result order, so two results
/// compare exactly. Only the types these tests produce are handled.
fn render(batches: &[RecordBatch]) -> Vec<String> {
    let mut rows = Vec::new();
    for batch in batches {
        for r in 0..batch.num_rows() {
            let cells: Vec<String> = (0..batch.num_columns())
                .map(|c| {
                    let col = batch.column(c);
                    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
                        if a.is_null(r) {
                            "NULL".into()
                        } else {
                            a.value(r).to_string()
                        }
                    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
                        if a.is_null(r) {
                            "NULL".into()
                        } else {
                            a.value(r).to_string()
                        }
                    } else if let Some(a) = col.as_any().downcast_ref::<LargeStringArray>() {
                        // The scan returns Utf8 columns as LargeUtf8.
                        if a.is_null(r) {
                            "NULL".into()
                        } else {
                            a.value(r).to_string()
                        }
                    } else {
                        panic!("unhandled result column type: {:?}", col.data_type())
                    }
                })
                .collect();
            rows.push(cells.join("|"));
        }
    }
    rows
}

/// Run `sql` and render the result.
fn query(db: &infino::Connection, sql: &str) -> Vec<String> {
    render(&db.query_sql(sql).expect("query_sql"))
}

#[test]
fn hydrate_parquet_file_is_queryable() {
    let dir = TempDir::new().expect("tempdir");
    let parquet_path = dir.path().join("hits.parquet");
    write_parquet(&parquet_path, &rows_batch(1, 5));

    let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");
    let committed = hydrate_parquet(&db, "hits", &parquet_path);

    assert_eq!(committed, 5, "hydrate reports every row committed");
    assert_eq!(query(&db, "SELECT COUNT(*) FROM hits"), vec!["5"]);
    assert_eq!(query(&db, "SELECT SUM(n) FROM hits"), vec!["15"]);
}

/// A hydrated table and a normally-ingested one (two appends, then optimize +
/// GC) answer every query identically, with hydrate building several
/// superfiles rather than one.
#[test]
fn hydrate_matches_normal_ingest() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().join("db").to_str().expect("utf-8 path")).expect("connect");

    // Hydrated: a 200-row Parquet file read in 32-row batches at a 64-row
    // target, so it lands as 4 superfiles (64 + 64 + 64 + 8).
    let parquet_path = dir.path().join("rows.parquet");
    write_parquet(&parquet_path, &rows_batch(1, 200));
    let committed = hydrate_parquet(&db, "hydrated", &parquet_path);
    assert_eq!(committed, 200);

    // Ingested the regular way: two appends (two superfiles), then optimize + GC.
    let ingested = db
        .create_table("ingested", user_schema(), IndexSpec::new())
        .expect("create_table (ingest)");
    ingested.append(&rows_batch(1, 100)).expect("append 1");
    ingested.append(&rows_batch(101, 200)).expect("append 2");
    ingested
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    ingested.gc(Duration::ZERO).expect("gc");

    // Count, sum, filter, min/max, and ordered row content. `_id` is excluded:
    // it is minted per path and differs.
    let queries = [
        "SELECT COUNT(*) FROM {t}",
        "SELECT SUM(n) FROM {t}",
        "SELECT COUNT(*) FROM {t} WHERE n > 150",
        "SELECT MIN(n), MAX(n) FROM {t}",
        "SELECT n, s FROM {t} WHERE n <= 3 ORDER BY n",
        "SELECT n, s FROM {t} WHERE n BETWEEN 90 AND 110 ORDER BY n",
    ];
    for q in queries {
        let hydrated = query(&db, &q.replace("{t}", "hydrated"));
        let ingested = query(&db, &q.replace("{t}", "ingested"));
        assert_eq!(hydrated, ingested, "mismatch for query: {q}");
    }
}
