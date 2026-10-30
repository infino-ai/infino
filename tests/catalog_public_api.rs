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
    BoolMode, Connection, FieldPatch, IndexSpec, InfinoError, Metric, OptimizeOptions, SchemaPatch,
    Supertable,
    arrow_array::{
        Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch, StringArray,
        StringViewArray, UInt64Array,
    },
    arrow_schema::{DataType, Field, Schema, SchemaRef},
    connect,
};
use serde_json::Value;
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

// Companion fixtures.
/// The purpose of the companion most tests declare: a ledger of what has
/// been loaded into the table.
const LEDGER: &str = "ledger";
/// A second companion's purpose.
const AUDIT: &str = "audit";
/// What every hidden table's directory name under its table starts with.
const HIDDEN_DIR_PREFIX: &str = "_infino_";
/// Directory-name suffix of the hidden vector index under a table.
const VECTOR_INDEX_DIR_SUFFIX: &str = "_vector_index";
/// The manifest-list key recording a table's companions.
const COMPANIONS_KEY: &str = "companions";
/// The directory holding a table's manifest pointer, and the pointer's name.
const POINTER_DIR: &str = "_supertable";
const POINTER_FILE: &str = "current";
/// The pointer line naming the current manifest list.
const MANIFEST_URI_LINE: &str = "manifest_uri=";
/// Keys the ledger records, and the rows each yielded.
const LEDGER_KEYS: [&str; 2] = [
    "events/2026-01-01/part-0.json",
    "events/2026-01-02/part-0.json",
];
const LEDGER_ROWS: [u64; 2] = [690, 12];
/// Events the audit companion records.
const AUDIT_EVENTS: [&str; 2] = ["loaded", "replaced"];
/// The column a schema patch adds to the ledger, and the value written to it.
const NOTE: &str = "note";
const NOTE_VALUE: &str = "rewritten upstream";
/// A purpose length one past the 64 characters a purpose may have.
const TOO_LONG_PURPOSE: usize = 65;
/// A uuid in the shape the engine names a companion's directory with, for a
/// second, unrecorded copy of one.
const STRAY_UUID: &str = "00000000-0000-4000-8000-000000000000";
/// What the error for an ambiguous re-attach says.
const AMBIGUOUS_MARK: &str = "none is attached";

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

fn title_batch(titles: &[&str]) -> RecordBatch {
    RecordBatch::try_new(
        title_schema(),
        vec![Arc::new(LargeStringArray::from(titles.to_vec()))],
    )
    .expect("title batch")
}

/// The ledger companion's schema, as a caller declares it: a key, full-text
/// indexed so exact matches run on it, and a row count.
fn ledger_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::LargeUtf8, false),
        Field::new("rows", DataType::UInt64, true),
    ]))
}

fn ledger_indexes() -> IndexSpec {
    IndexSpec::new().fts("key")
}

/// The audit companion's schema: one indexed event name.
fn audit_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "event",
        DataType::LargeUtf8,
        false,
    )]))
}

/// The docs table declared with the ledger companion.
fn docs_with_ledger() -> IndexSpec {
    IndexSpec::new()
        .fts("title")
        .companion(LEDGER, ledger_schema(), ledger_indexes())
}

/// One ledger row per fixture key.
fn ledger_batch(ledger: &Supertable) -> RecordBatch {
    RecordBatch::try_new(
        ledger.schema(),
        vec![
            Arc::new(LargeStringArray::from(LEDGER_KEYS.to_vec())),
            Arc::new(UInt64Array::from(LEDGER_ROWS.to_vec())),
        ],
    )
    .expect("ledger rows match the ledger schema")
}

/// The `rows` of every ledger row an exact match on `key` returns.
fn rows_for(ledger: &Supertable, key: &str) -> Vec<u64> {
    ledger
        .exact_match("key", key, Some(&["rows"]))
        .expect("exact_match on key")
        .iter()
        .flat_map(|b| {
            let arr = b
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .expect("rows is u64");
            (0..arr.len()).map(|i| arr.value(i)).collect::<Vec<_>>()
        })
        .collect()
}

/// Rows of `table` a token match for `word` on `column` finds.
fn matches(table: &Supertable, column: &str, word: &str) -> usize {
    table
        .token_match(column, word, BoolMode::Or, None)
        .expect("token_match")
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

/// A connection's ledger handle, which the test expects to exist.
fn ledger_of(docs: &Supertable) -> Supertable {
    docs.companion(LEDGER)
        .expect("the ledger opens")
        .expect("the table has a ledger")
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

/// The directory-name suffix of a companion of `purpose`.
fn companion_suffix(purpose: &str) -> String {
    format!("_{purpose}")
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

/// The user table's directory under `root`: the one holding a manifest
/// pointer outside every hidden table's directory.
fn table_dir(root: &Path) -> PathBuf {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        if dir.join(POINTER_DIR).join(POINTER_FILE).is_file() {
            found.push(dir.clone());
        }
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("dir entry").path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if path.is_dir() && !name.starts_with(HIDDEN_DIR_PREFIX) {
                stack.push(path);
            }
        }
    }
    assert_eq!(found.len(), 1, "one user table under {root:?}: {found:?}");
    found.remove(0)
}

/// The manifest list the pointer of the table at `dir` names.
fn current_list(dir: &Path) -> PathBuf {
    let pointer =
        fs::read_to_string(dir.join(POINTER_DIR).join(POINTER_FILE)).expect("read pointer");
    let uri = pointer
        .lines()
        .find_map(|line| line.strip_prefix(MANIFEST_URI_LINE))
        .expect("the pointer names a list");
    dir.join(uri)
}

/// The table's current manifest list, as JSON.
fn current_list_json(root: &Path) -> Value {
    let bytes = fs::read(current_list(&table_dir(root))).expect("read list");
    serde_json::from_slice(&bytes).expect("the list is JSON")
}

/// The prefix the table's current manifest list records for `purpose`.
fn recorded_companion(root: &Path, purpose: &str) -> Option<String> {
    current_list_json(root)
        .get(COMPANIONS_KEY)
        .and_then(|companions| companions.get(purpose))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// Rewrite the table's current manifest list without its companions, as an
/// engine older than companions writes it on its next commit.
fn drop_companions_from_list(root: &Path) {
    let path = current_list(&table_dir(root));
    let mut list: Value =
        serde_json::from_slice(&fs::read(&path).expect("read list")).expect("the list is JSON");
    let removed = list
        .as_object_mut()
        .expect("the list is an object")
        .remove(COMPANIONS_KEY);
    assert!(removed.is_some(), "the list recorded companions");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&list).expect("encode list"),
    )
    .expect("write list");
}

/// Copy the directory `from` to `to`, files and all.
fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create_dir");
    for entry in fs::read_dir(from).expect("read_dir") {
        let path = entry.expect("dir entry").path();
        let target = to.join(path.file_name().expect("a name"));
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy");
        }
    }
}

/// A table created with a companion has one: written and read through the
/// handle the table gives, recorded on the table's manifest, reopened from
/// it by a fresh connection, never listed, and purged with the table.
#[test]
fn a_companion_is_hidden_reopened_and_purged_with_its_table() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table("docs", title_schema(), docs_with_ledger())
            .expect("create_table with a companion");
        let ledger = ledger_of(&docs);
        ledger.append(&ledger_batch(&ledger)).expect("append");
        assert_eq!(rows_for(&ledger, LEDGER_KEYS[0]), vec![LEDGER_ROWS[0]]);
        assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    }
    let ledger_dirs = hidden_dirs(dir.path(), &companion_suffix(LEDGER));
    assert_eq!(ledger_dirs.len(), 1, "one companion under the table");
    assert!(
        files_under(&ledger_dirs[0]) > 0,
        "the companion holds objects"
    );
    assert!(
        hidden_dirs(dir.path(), VECTOR_INDEX_DIR_SUFFIX).is_empty(),
        "no vector columns, no vector index"
    );
    let recorded = recorded_companion(dir.path(), LEDGER).expect("the manifest records it");
    assert!(ledger_dirs[0].ends_with(&recorded));

    // A fresh connection finds the companion from the manifest alone, and
    // a row can be replaced: delete it, append the new one.
    let db = connect(&uri).expect("reconnect");
    assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    let ledger = ledger_of(&db.open_table("docs").expect("open_table"));
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[1]), vec![LEDGER_ROWS[1]]);
    let stats = ledger
        .delete(col("rows").eq(lit(LEDGER_ROWS[1])))
        .expect("delete a row");
    assert_eq!(stats.n_tombstoned(), 1);
    assert!(rows_for(&ledger, LEDGER_KEYS[1]).is_empty());
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[0]), vec![LEDGER_ROWS[0]]);

    db.drop_table("docs", true).expect("drop_table purge");
    // Purge deletes objects; a local directory may outlive its last file.
    assert_eq!(
        hidden_dirs(dir.path(), &companion_suffix(LEDGER))
            .iter()
            .map(|d| files_under(d))
            .sum::<usize>(),
        0,
        "purge removes the companion's objects with the table"
    );
}

/// A table created without companions has none, finds none under its
/// location, and its manifest list carries no key for them: it is the list
/// an engine without companions writes.
#[test]
fn a_table_without_companions_has_none_and_an_unchanged_list() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    let db = connect(&uri).expect("connect");
    let docs = db
        .create_table("docs", title_schema(), IndexSpec::new().fts("title"))
        .expect("create_table");
    docs.append(&title_batch(FIRST_BATCH)).expect("append");
    assert!(docs.companion(LEDGER).expect("no error").is_none());
    let reopened = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("open_table");
    assert!(reopened.companion(LEDGER).expect("no error").is_none());
    assert!(hidden_dirs(dir.path(), "").is_empty());
    assert!(
        current_list_json(dir.path()).get(COMPANIONS_KEY).is_none(),
        "no companions writes no key"
    );
}

/// Two companions of different purposes live side by side, each with its
/// own schema and its own full-text index, and both reopen.
#[test]
fn two_companions_each_keep_their_own_schema_and_index() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table(
                "docs",
                title_schema(),
                docs_with_ledger().companion(AUDIT, audit_schema(), IndexSpec::new().fts("event")),
            )
            .expect("create_table with two companions");
        let ledger = ledger_of(&docs);
        ledger
            .append(&ledger_batch(&ledger))
            .expect("append ledger");
        let audit = docs
            .companion(AUDIT)
            .expect("audit opens")
            .expect("audit exists");
        audit
            .append(
                &RecordBatch::try_new(
                    audit.schema(),
                    vec![Arc::new(LargeStringArray::from(AUDIT_EVENTS.to_vec()))],
                )
                .expect("audit rows"),
            )
            .expect("append audit");
        assert_eq!(db.list_tables().expect("list"), vec!["docs".to_string()]);
    }
    assert_eq!(hidden_dirs(dir.path(), &companion_suffix(LEDGER)).len(), 1);
    assert_eq!(hidden_dirs(dir.path(), &companion_suffix(AUDIT)).len(), 1);

    let docs = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("open_table");
    let ledger = ledger_of(&docs);
    let audit = docs
        .companion(AUDIT)
        .expect("audit opens")
        .expect("audit exists");
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[0]), vec![LEDGER_ROWS[0]]);
    assert_eq!(matches(&audit, "event", AUDIT_EVENTS[1]), 1);
    assert!(ledger.schema().index_of("event").is_err());
    assert!(audit.schema().index_of("key").is_err());
    assert!(docs.companion("unknown").expect("no error").is_none());
}

/// A companion declaration the engine cannot honor is refused at create,
/// and nothing is registered.
#[test]
fn an_invalid_companion_declaration_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let ledger = |purpose: &str| {
        IndexSpec::new()
            .fts("title")
            .companion(purpose, ledger_schema(), ledger_indexes())
    };
    let refused = [
        ("an empty purpose", ledger("")),
        ("an uppercase purpose", ledger("Ledger")),
        ("a purpose with a dash", ledger("source-files")),
        ("the vector index's suffix", ledger("vector_index")),
        (
            "a purpose past the bound",
            ledger(&"a".repeat(TOO_LONG_PURPOSE)),
        ),
        (
            "a purpose declared twice",
            ledger(LEDGER).companion(LEDGER, ledger_schema(), ledger_indexes()),
        ),
        (
            "a companion with a companion",
            IndexSpec::new().fts("title").companion(
                LEDGER,
                ledger_schema(),
                ledger_indexes().companion(AUDIT, audit_schema(), IndexSpec::new()),
            ),
        ),
    ];
    for (what, spec) in refused {
        let err = db
            .create_table("docs", title_schema(), spec)
            .expect_err(what);
        assert!(matches!(err, InfinoError::Config(_)), "{what}: {err:?}");
    }
    assert!(db.list_tables().expect("list").is_empty());

    let docs = db
        .create_table("docs", title_schema(), IndexSpec::new().fts("title"))
        .expect("create_table");
    let err = docs.companion("Ledger").expect_err("an invalid purpose");
    assert!(matches!(err, InfinoError::Config(_)), "got {err:?}");
}

/// A companion whose pointer is gone does not stop its table opening: the
/// table reads and writes as before, and only the caller asking for the
/// companion sees its error.
#[test]
fn a_companion_that_fails_to_open_leaves_its_table_usable() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table("docs", title_schema(), docs_with_ledger())
            .expect("create_table");
        docs.append(&title_batch(FIRST_BATCH)).expect("append");
        let ledger = ledger_of(&docs);
        ledger.append(&ledger_batch(&ledger)).expect("append");
    }
    let ledger_dir = hidden_dirs(dir.path(), &companion_suffix(LEDGER)).remove(0);
    fs::remove_file(ledger_dir.join(POINTER_DIR).join(POINTER_FILE)).expect("remove pointer");

    let docs = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("the table opens without its companion");
    let err = docs
        .companion(LEDGER)
        .expect_err("the companion does not open");
    assert!(
        err.to_string().contains(LEDGER),
        "names the companion: {err}"
    );
    assert_eq!(matches(&docs, "title", "fox"), 1);
    docs.append(&title_batch(SECOND_BATCH))
        .expect("the table still takes writes");
    assert_eq!(matches(&docs, "title", "fox"), 2);
    // The table's commits keep the record, so a repaired companion is found
    // again.
    assert!(recorded_companion(dir.path(), LEDGER).is_some());
}

/// The manifest's record of a companion, dropped as an engine without
/// companions drops it on commit, is recovered from the companion's
/// storage location: the companion re-attaches with its rows, and the
/// table's next commit records it again.
#[test]
fn a_lost_companion_record_is_reattached_and_written_back() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table("docs", title_schema(), docs_with_ledger())
            .expect("create_table");
        let ledger = ledger_of(&docs);
        ledger.append(&ledger_batch(&ledger)).expect("append");
    }
    drop_companions_from_list(dir.path());
    assert!(recorded_companion(dir.path(), LEDGER).is_none());

    let docs = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("open_table");
    let ledger = ledger_of(&docs);
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[0]), vec![LEDGER_ROWS[0]]);
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[1]), vec![LEDGER_ROWS[1]]);

    docs.append(&title_batch(FIRST_BATCH)).expect("append");
    let recorded = recorded_companion(dir.path(), LEDGER).expect("the commit records it");
    assert!(hidden_dirs(dir.path(), &companion_suffix(LEDGER))[0].ends_with(&recorded));
}

/// When the record is lost and two companions of the purpose are stored
/// under the table, neither is a safe guess: none is attached.
#[test]
fn two_unrecorded_companions_of_one_purpose_attach_neither() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        db.create_table("docs", title_schema(), docs_with_ledger())
            .expect("create_table");
    }
    let ledger_dir = hidden_dirs(dir.path(), &companion_suffix(LEDGER)).remove(0);
    let stray = ledger_dir
        .parent()
        .expect("under the table")
        .join(format!("{HIDDEN_DIR_PREFIX}{STRAY_UUID}_{LEDGER}"));
    copy_dir(&ledger_dir, &stray);
    drop_companions_from_list(dir.path());

    let docs = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("open_table");
    let err = docs.companion(LEDGER).expect_err("ambiguous");
    let message = err.to_string();
    assert!(message.contains(LEDGER), "names the purpose: {message}");
    assert!(
        message.contains(AMBIGUOUS_MARK) && message.contains(STRAY_UUID),
        "says why and names the candidates: {message}"
    );
}

/// A companion's schema evolves through its own handle, and the companion
/// reopens with the evolved schema.
#[test]
fn a_companion_schema_evolves_through_its_handle() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    {
        let db = connect(&uri).expect("connect");
        let docs = db
            .create_table("docs", title_schema(), docs_with_ledger())
            .expect("create_table");
        let ledger = ledger_of(&docs);
        ledger.append(&ledger_batch(&ledger)).expect("append");
        let patch = SchemaPatch::new(vec![
            FieldPatch::named(NOTE)
                .with_type(DataType::LargeUtf8)
                .with_nullable(true),
        ]);
        let doc = ledger.apply_schema(&patch, None).expect("apply_schema");
        assert!(doc.fields().iter().any(|f| f.name == NOTE));
        let schema = ledger.schema();
        assert!(schema.index_of(NOTE).is_ok());
        let row = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(LargeStringArray::from(vec![
                    "events/2026-01-03/part-0.json",
                ])),
                Arc::new(UInt64Array::from(vec![LEDGER_ROWS[0]])),
                Arc::new(LargeStringArray::from(vec![NOTE_VALUE])),
            ],
        )
        .expect("a row in the evolved schema");
        ledger.append(&row).expect("append evolved");
    }

    let docs = connect(&uri)
        .expect("reconnect")
        .open_table("docs")
        .expect("open_table");
    let ledger = ledger_of(&docs);
    assert!(ledger.schema().index_of(NOTE).is_ok());
    let notes = ledger
        .exact_match("key", "events/2026-01-03/part-0.json", Some(&[NOTE]))
        .expect("exact_match");
    let note = notes
        .iter()
        .find(|b| b.num_rows() > 0)
        .expect("the evolved row")
        .column(0)
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .expect("note is utf8")
        .value(0)
        .to_string();
    assert_eq!(note, NOTE_VALUE);
    assert_eq!(rows_for(&ledger, LEDGER_KEYS[0]), vec![LEDGER_ROWS[0]]);
}

/// A table with vector columns and a companion carries both hidden tables,
/// and vector search still runs over its own index.
#[test]
fn a_table_with_vectors_and_a_companion_has_both_hidden_tables() {
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
                .companion(LEDGER, ledger_schema(), ledger_indexes()),
        )
        .expect("create_table");
    docs.append(&build_batch(schema, FIRST_BATCH, 0))
        .expect("append");
    let ledger = ledger_of(&docs);
    ledger.append(&ledger_batch(&ledger)).expect("append");
    assert_eq!(hidden_dirs(dir.path(), &companion_suffix(LEDGER)).len(), 1);
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

/// A connection with no storage refuses companions rather than creating a
/// table without the companion it was asked for.
#[test]
fn a_memory_connection_refuses_companions() {
    let db = connect("memory://").expect("connect");
    let err = db
        .create_table("docs", title_schema(), docs_with_ledger())
        .expect_err("no storage, no companion");
    assert!(matches!(err, InfinoError::Unsupported(_)), "got {err:?}");
    assert!(db.list_tables().expect("list").is_empty());
}

// ---- Table snapshots ----

/// The rows a snapshot is taken over, all in one commit (one superfile).
const HELD_TITLES: &[&str] = &["alpha", "bravo", "charlie", "echo"];
/// Deleted before the snapshot, so the superfile has tombstones when it is
/// taken.
const DELETED_BEFORE: &str = "echo";
/// Deleted after the snapshot, from the same superfile: the delete rewrites
/// the tombstones the snapshot read.
const DELETED_AFTER: &str = "alpha";
/// Updated after the snapshot, and what it is updated to.
const UPDATED: &str = "bravo";
const UPDATED_TO: &str = "bravo rewritten";
/// Appended after the snapshot.
const APPENDED_AFTER: &str = "delta";
/// A gc safety gap that keeps every file the snapshot reads.
const KEEP_SNAPSHOT_FILES: Duration = Duration::from_secs(3600);

/// Every `title` across `batches`, sorted.
fn titles_of(batches: &[RecordBatch]) -> Vec<String> {
    let mut titles = Vec::new();
    for batch in batches {
        let column = batch.column(batch.schema().index_of("title").expect("title projected"));
        for i in 0..batch.num_rows() {
            let title = if let Some(arr) = column.as_any().downcast_ref::<LargeStringArray>() {
                arr.value(i)
            } else if let Some(arr) = column.as_any().downcast_ref::<StringViewArray>() {
                arr.value(i)
            } else {
                column
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("title is utf8")
                    .value(i)
            };
            titles.push(title.to_string());
        }
    }
    titles.sort();
    titles
}

/// Sorted owned copies of `titles`.
fn sorted(titles: &[&str]) -> Vec<String> {
    let mut titles: Vec<String> = titles.iter().map(|t| t.to_string()).collect();
    titles.sort();
    titles
}

/// A title-only table `docs` holding [`HELD_TITLES`] less [`DELETED_BEFORE`].
fn held_docs(db: &Connection) -> Supertable {
    let docs = db
        .create_table("docs", title_schema(), IndexSpec::new().fts("title"))
        .expect("create_table");
    docs.append(&title_batch(HELD_TITLES)).expect("append");
    docs.delete(col("title").eq(lit(DELETED_BEFORE)))
        .expect("delete before the snapshot");
    docs
}

/// A snapshot holds its version while the table moves on: SQL, the search
/// functions and the snapshot's own searches all see the rows the table
/// held when it was taken, not rows appended, deleted or updated since,
/// and the live table sees the new state.
#[test]
fn a_snapshot_reads_its_version_through_later_writes() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let docs = held_docs(&db);
    let held = sorted(&HELD_TITLES[..3]);

    let snapshot = db.snapshot("docs").expect("snapshot");
    assert_eq!(snapshot.table_name(), "docs");
    assert_eq!(
        db.snapshot("docs").expect("second snapshot").version(),
        snapshot.version(),
        "nothing committed in between, so the same version"
    );

    docs.append(&title_batch(&[APPENDED_AFTER]))
        .expect("append after");
    docs.delete(col("title").eq(lit(DELETED_AFTER)))
        .expect("delete after");
    docs.update(col("title").eq(lit(UPDATED)), &title_batch(&[UPDATED_TO]))
        .expect("update after");

    let live = titles_of(&db.query_sql("SELECT title FROM docs").expect("live"));
    assert_eq!(live, sorted(&["charlie", APPENDED_AFTER, UPDATED_TO]));
    let read = |sql: &str| titles_of(&db.query_sql_at(sql, &[&snapshot]).expect(sql));
    assert_eq!(read("SELECT title FROM docs"), held);
    assert_eq!(
        read(&format!(
            "SELECT title FROM token_match('docs', 'title', '{DELETED_AFTER}')"
        )),
        vec![DELETED_AFTER.to_string()]
    );
    assert!(
        read(&format!(
            "SELECT title FROM token_match('docs', 'title', '{APPENDED_AFTER}')"
        ))
        .is_empty()
    );

    let rows = |batches: Vec<RecordBatch>| batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    assert_eq!(
        rows(
            snapshot
                .token_match("title", DELETED_AFTER, BoolMode::Or, None)
                .expect("token_match")
        ),
        1
    );
    assert_eq!(matches(&docs, "title", DELETED_AFTER), 0);
    assert_eq!(
        snapshot
            .count("title", APPENDED_AFTER, BoolMode::Or)
            .expect("count"),
        0
    );
    assert_eq!(
        rows(
            snapshot
                .exact_match("title", UPDATED, Some(&["title"]))
                .expect("exact_match")
        ),
        1
    );
    assert_eq!(
        snapshot
            .count("title", DELETED_BEFORE, BoolMode::Or)
            .expect("count"),
        0,
        "a delete the snapshot's version holds stays applied"
    );
    assert!(db.snapshot("docs").expect("later snapshot").version() > snapshot.version());
}

/// Another connection's deletes, a compaction and a gc that keeps the
/// snapshot's files do not reach a snapshot.
#[test]
fn a_snapshot_reads_its_version_through_another_writer_and_a_compaction() {
    let dir = TempDir::new().expect("tempdir");
    let uri = dir.path().to_str().expect("utf-8 path").to_string();
    let db = connect(&uri).expect("connect");
    let docs = held_docs(&db);
    docs.append(&title_batch(&[APPENDED_AFTER]))
        .expect("a second superfile");
    let snapshot = db.snapshot("docs").expect("snapshot");

    let writer = connect(&uri).expect("second connection");
    let other = writer.open_table("docs").expect("open_table");
    other
        .delete(col("title").eq(lit(DELETED_AFTER)))
        .expect("delete from the other connection");
    other
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    let report = other.gc(KEEP_SNAPSHOT_FILES).expect("gc");
    assert_eq!(report.delete_errors, 0);

    assert_eq!(
        titles_of(&writer.query_sql("SELECT title FROM docs").expect("live")),
        sorted(&["bravo", "charlie", APPENDED_AFTER])
    );
    assert_eq!(
        titles_of(
            &db.query_sql_at("SELECT title FROM docs", &[&snapshot])
                .expect("snapshot")
        ),
        sorted(&["alpha", "bravo", "charlie", APPENDED_AFTER])
    );
}

/// A snapshot does not serve vector search (the vector index is not held
/// with it), and one query takes one snapshot per table.
#[test]
fn a_snapshot_refuses_vector_search_and_a_second_snapshot_of_its_table() {
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
    docs.append(&build_batch(schema, FIRST_BATCH, 0))
        .expect("append");
    let snapshot = db.snapshot("docs").expect("snapshot");

    let query: Vec<String> = unit_embedding(0).iter().map(f32::to_string).collect();
    let err = db
        .query_sql_at(
            &format!(
                "SELECT _id FROM vector_search('docs', 'emb', '{}', {SEARCH_TOP_K})",
                query.join(",")
            ),
            &[&snapshot],
        )
        .expect_err("vector search over a snapshot");
    assert!(matches!(err, InfinoError::Unsupported(_)), "got {err:?}");

    let again = db.snapshot("docs").expect("second snapshot");
    let err = db
        .query_sql_at("SELECT _id FROM docs", &[&snapshot, &again])
        .expect_err("two snapshots of one table");
    assert!(matches!(err, InfinoError::Query(_)), "got {err:?}");
}
