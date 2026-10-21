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

use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use datafusion::prelude::{col, lit};
use infino::{
    BoolMode, IndexSpec, InfinoError, Metric, OptimizeOptions, Supertable,
    arrow_array::{
        Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Int64Array,
        LargeStringArray, RecordBatch, StringArray, UInt32Array, UInt64Array,
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
const SECOND_BATCH: &[&str] = &["a red clever fox fox", "an old grey wolf"];

// Source-file list fixtures.
/// Directory-name suffix of the hidden source-file list under a table.
const SOURCE_FILES_DIR_SUFFIX: &str = "_sync";
/// Directory-name suffix of the hidden vector index under a table.
const VECTOR_INDEX_DIR_SUFFIX: &str = "_vector_index";
/// Prefix every hidden table's directory name starts with.
const HIDDEN_DIR_PREFIX: &str = "_infino_";
/// The manifest key naming the source-file list.
const SOURCE_FILES_MANIFEST_KEY: &str = "source_files_storage_prefix";
/// Two source files the list records, by key.
const FILE_KEYS: [&str; 2] = [
    "events/2026-01-01/part-0.json",
    "events/2026-01-02/part-0.json",
];
/// The numbers the list gives those files.
const FILE_NOS: [u32; 2] = [7, 8];
/// The version markers the files were read at.
const FILE_MARKERS: [&str; 2] = ["1700000000000001", "1700000000000002"];
/// Bytes, modification time, rows and load time recorded for each file.
const FILE_SIZE: u64 = 4_096;
const FILE_MTIME_MS: i64 = 1_700_000_000_000;
const FILE_ROWS: u64 = 690;
const FILE_LOADED_MS: i64 = 1_700_000_100_000;

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

    // Hybrid: the text leg pins "fox" rows — the second-batch fox
    // carries tf=2 so BM25 ranks it first under corpus-wide (global)
    // statistics — and the vector leg (row 2's own embedding) agrees,
    // so the combined ranking puts the second-batch fox on top.
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

/// A title-only schema: a table with no vector column, so no vector index.
fn title_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

/// One file row per entry of the fixtures, shaped by the list's own schema.
fn file_rows(list: &Supertable) -> RecordBatch {
    let n = FILE_KEYS.len();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from(vec!["file"; n])),
        Arc::new(UInt32Array::from(FILE_NOS.to_vec())),
        Arc::new(LargeStringArray::from(FILE_KEYS.to_vec())),
        Arc::new(LargeStringArray::from(FILE_MARKERS.to_vec())),
        Arc::new(UInt64Array::from(vec![FILE_SIZE; n])),
        Arc::new(Int64Array::from(vec![FILE_MTIME_MS; n])),
        Arc::new(UInt64Array::from(vec![FILE_ROWS; n])),
        Arc::new(Int64Array::from(vec![FILE_LOADED_MS; n])),
        Arc::new(Int64Array::from(vec![None::<i64>; n])),
        Arc::new(BooleanArray::from(vec![None::<bool>; n])),
    ];
    RecordBatch::try_new(list.schema(), columns).expect("file rows match the list schema")
}

/// The `file_no` of every row an exact match on `key` returns.
fn file_nos_for(list: &Supertable, key: &str) -> Vec<u32> {
    list.exact_match("key", key, Some(&["file_no"]))
        .expect("exact_match on key")
        .iter()
        .flat_map(|b| {
            let arr = b
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("file_no is u32");
            (0..arr.len()).map(|i| arr.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

/// Every directory under `root` named like a hidden table with `suffix`.
fn hidden_dirs(root: &Path, suffix: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if !path.is_dir() {
                continue;
            }
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with(HIDDEN_DIR_PREFIX) && name.ends_with(suffix) {
                found.push(path.clone());
            }
            stack.push(path);
        }
    }
    found
}

/// Whether any `.json` file under `root`, outside the hidden tables, names
/// the source-file list.
fn user_manifests_name_the_list(root: &Path) -> bool {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() {
                if !name.starts_with(HIDDEN_DIR_PREFIX) {
                    stack.push(path);
                }
            } else if name.ends_with(".json")
                && fs::read_to_string(&path)
                    .expect("read manifest")
                    .contains(SOURCE_FILES_MANIFEST_KEY)
            {
                return true;
            }
        }
    }
    false
}

/// A table created with a source-file list has one: written and read
/// through the parent's handle, named on the parent's manifest, reopened
/// from it by a fresh connection, never listed, and purged with the table.
#[test]
fn source_file_list_is_hidden_reopened_and_purged_with_its_table() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table(
                "docs",
                title_schema(),
                IndexSpec::new().fts("title").source_file_list(),
            )
            .expect("create_table with a source-file list");
        let list = docs.source_file_list().expect("the list exists");
        list.append(&file_rows(&list)).expect("append file rows");
        assert_eq!(file_nos_for(&list, FILE_KEYS[0]), vec![FILE_NOS[0]]);
        assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    }
    let list_dirs = hidden_dirs(dir.path(), SOURCE_FILES_DIR_SUFFIX);
    assert_eq!(list_dirs.len(), 1, "one source-file list under the table");
    assert!(files_under(&list_dirs[0]) > 0, "the list holds objects");
    assert!(
        hidden_dirs(dir.path(), VECTOR_INDEX_DIR_SUFFIX).is_empty(),
        "no vector columns, no vector index"
    );
    assert!(user_manifests_name_the_list(dir.path()));

    // A fresh connection finds the list from the manifest alone, and a
    // file's row can be replaced: tombstone it, append the new one.
    let db = connect(&uri).expect("reconnect");
    assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    let list = db
        .open_table("docs")
        .expect("open_table")
        .source_file_list()
        .expect("reopened from the manifest");
    assert_eq!(file_nos_for(&list, FILE_KEYS[1]), vec![FILE_NOS[1]]);
    let stats = list
        .delete(col("file_no").eq(lit(FILE_NOS[1])))
        .expect("delete a file's row");
    assert_eq!(stats.n_tombstoned(), 1);
    assert!(file_nos_for(&list, FILE_KEYS[1]).is_empty());
    assert_eq!(file_nos_for(&list, FILE_KEYS[0]), vec![FILE_NOS[0]]);

    db.drop_table("docs", true).expect("drop_table purge");
    // Purge deletes objects; a local directory may outlive its last file.
    assert_eq!(
        hidden_dirs(dir.path(), SOURCE_FILES_DIR_SUFFIX)
            .iter()
            .map(|d| files_under(d))
            .sum::<usize>(),
        0,
        "purge removes the list's objects with the table"
    );
}

/// Files anywhere under `root`.
fn files_under(root: &Path) -> usize {
    let mut n = 0;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                stack.push(path);
            } else {
                n += 1;
            }
        }
    }
    n
}

/// A table created without the flag has no list and its manifest never
/// names one.
#[test]
fn a_table_without_the_flag_has_no_source_file_list() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let docs = db
        .create_table("docs", title_schema(), IndexSpec::new().fts("title"))
        .expect("create_table");
    let batch = RecordBatch::try_new(
        title_schema(),
        vec![Arc::new(LargeStringArray::from(FIRST_BATCH.to_vec()))],
    )
    .expect("batch");
    docs.append(&batch).expect("append");
    assert!(docs.source_file_list().is_none());
    assert!(
        db.open_table("docs")
            .expect("open_table")
            .source_file_list()
            .is_none()
    );
    assert!(hidden_dirs(dir.path(), SOURCE_FILES_DIR_SUFFIX).is_empty());
    assert!(!user_manifests_name_the_list(dir.path()));
}

/// A table with vector columns and the flag carries both hidden tables, and
/// vector search still runs over its own index.
#[test]
fn a_table_with_vectors_and_the_flag_has_both_hidden_tables() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = table_schema();
    let docs = db
        .create_table(
            "docs",
            schema.clone(),
            IndexSpec::new()
                .fts("title")
                .vector("emb", EMB_DIM, Metric::Cosine)
                .source_file_list(),
        )
        .expect("create_table");
    docs.append(&build_batch(schema, FIRST_BATCH, 0))
        .expect("append");
    let list = docs.source_file_list().expect("the list exists");
    list.append(&file_rows(&list)).expect("append file rows");
    assert_eq!(hidden_dirs(dir.path(), SOURCE_FILES_DIR_SUFFIX).len(), 1);
    assert_eq!(hidden_dirs(dir.path(), VECTOR_INDEX_DIR_SUFFIX).len(), 1);
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

/// A connection with no storage refuses the flag rather than creating a
/// table without the list it was asked for.
#[test]
fn a_memory_connection_refuses_a_source_file_list() {
    let db = connect("memory://").expect("connect");
    let err = db
        .create_table(
            "docs",
            title_schema(),
            IndexSpec::new().fts("title").source_file_list(),
        )
        .expect_err("no storage, no list");
    assert!(matches!(err, InfinoError::Unsupported(_)), "got {err:?}");
    assert!(db.list_tables().expect("list").is_empty());
}
