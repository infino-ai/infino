// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The schema grows from the data and changes through one declarative
//! write, and every read follows: appends that add a column, batches that
//! omit one, the schema patch's add/rename/drop, updates under the same
//! rules, two writers racing on the schema, and compaction over the
//! generations the growth leaves behind.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    Bm25Params, ColumnIndex, CompactionSettings, Connection, FieldId, FieldPatch, IndexSpec,
    InfinoError, Metric, OptimizeOptions, SchemaError, SchemaPatch, Stemmer, Stopwords,
    TableSchema, connect, serde_json,
};
use tempfile::TempDir;

const TABLE: &str = "docs";
/// Small enough that two one-row superfiles are below it and merge.
const COMPACT_TARGET_MB: u64 = 1;
const COMPACT_MIN_FILL_PERCENT: u8 = 1;
/// Rows beside each generation's marker row in the compaction test.
const FILLER_ROWS: usize = 200;

fn title_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

fn title_score_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("score", DataType::Int64, true),
    ]))
}

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns
        .iter()
        .map(|(name, array)| Field::new(*name, array.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns.into_iter().map(|(_, a)| a).collect(),
    )
    .expect("batch")
}

fn titles(values: &[&str]) -> ArrayRef {
    Arc::new(LargeStringArray::from(values.to_vec()))
}

fn ints(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn add(name: &str, data_type: DataType) -> FieldPatch {
    FieldPatch::named(name).with_type(data_type)
}

fn patch(fields: Vec<FieldPatch>) -> SchemaPatch {
    SchemaPatch::new(fields)
}

/// `sql`'s rows as strings, one per row, cells joined by `|`, sorted.
fn rows(db: &Connection, sql: &str) -> Vec<String> {
    let batches = db.query_sql(sql).expect("query");
    let mut out = Vec::new();
    for b in &batches {
        for row in 0..b.num_rows() {
            let cells: Vec<String> = (0..b.num_columns())
                .map(|c| {
                    let array = b.column(c);
                    if array.is_null(row) {
                        return "null".to_string();
                    }
                    if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
                        return a.value(row).to_string();
                    }
                    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
                        return a.value(row).to_string();
                    }
                    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
                        return a.value(row).to_string();
                    }
                    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                        return a.value(row).to_string();
                    }
                    panic!("unexpected column type {:?}", array.data_type());
                })
                .collect();
            out.push(cells.join("|"));
        }
    }
    out.sort();
    out
}

fn names(schema: &TableSchema) -> Vec<(&str, FieldId)> {
    schema
        .fields()
        .iter()
        .map(|f| (f.name.as_str(), f.id))
        .collect()
}

#[test]
fn an_unseen_column_joins_the_schema_and_earlier_rows_read_null() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let before = db.schema(TABLE).expect("schema");
    assert_eq!(names(&before), vec![("title", FieldId(1))]);

    docs.append(&batch(vec![("title", titles(&["a"]))]))
        .expect("first append");
    docs.append(&batch(vec![
        ("title", titles(&["b"])),
        ("score", ints(vec![Some(7)])),
    ]))
    .expect("append with a new column");

    let after = db.schema(TABLE).expect("schema");
    assert_eq!(
        names(&after),
        vec![("title", FieldId(1)), ("score", FieldId(2))]
    );
    let score = &after.fields()[1];
    assert_eq!(score.data_type, DataType::Int64);
    assert!(score.nullable, "a column added from data admits nulls");
    assert_eq!(after.schema_id(), before.schema_id() + 1);
    assert_eq!(
        rows(&db, "SELECT title, score FROM docs"),
        vec!["a|null", "b|7"]
    );
    // The Arrow view a client appends against grew too.
    assert_eq!(docs.schema().fields().len(), 2);
}

#[test]
fn an_absent_nullable_column_is_null_filled_and_a_missing_required_one_refused() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append(&batch(vec![("title", titles(&["a"]))]))
        .expect("score may be omitted");
    assert_eq!(rows(&db, "SELECT title, score FROM docs"), vec!["a|null"]);

    let err = docs
        .append(&batch(vec![("score", ints(vec![Some(1)]))]))
        .expect_err("title is required");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::MissingColumn { column }) if column == "title"),
        "{err}"
    );
    let err = docs
        .append(&batch(vec![(
            "title",
            Arc::new(LargeStringArray::from(vec![None::<&str>])) as ArrayRef,
        )]))
        .expect_err("a null in a required column");
    assert!(
        matches!(
            &err,
            InfinoError::Schema(SchemaError::NullInNonNullable { .. })
        ),
        "{err}"
    );
    assert_eq!(db.schema(TABLE).expect("schema").schema_id(), 1);
}

/// How the narrow batches of [`narrow_batches_on_a_wide_vector_table`]
/// reach the table.
#[derive(Clone, Copy)]
enum Commits {
    /// One `append` per batch, each committed on its own.
    EachAppend,
    /// Every batch buffered by one writer, then committed together.
    OnceForAll,
}

/// Each narrow batch committed on its own: every commit fills the absent
/// columns with one shared null array per type and gathers each shard's
/// rows only from the columns that hold values.
#[test]
fn narrow_batches_on_a_wide_vector_table_read_back_exactly() {
    narrow_batches_on_a_wide_vector_table(Commits::EachAppend);
}

/// The narrow batches buffered by one writer and committed once: the
/// commit joins the buffered batches, keeping the columns none of them
/// carries as one shared null array, and the rows read back as when each
/// batch is committed alone.
#[test]
fn narrow_batches_buffered_into_one_commit_read_back_exactly() {
    narrow_batches_on_a_wide_vector_table(Commits::OnceForAll);
}

/// A wide vector table written by batches that each carry a few of its
/// columns, committed as `commits` says: every carried value reads back
/// where it was written, every other cell reads null, and the vector and
/// text indexes answer over the rows.
fn narrow_batches_on_a_wide_vector_table(commits: Commits) {
    /// Nullable columns beside the text and the vector.
    const WIDE_COLUMNS: usize = 300;
    /// Rows per append; enough for the vector index to train cells.
    const ROWS: usize = 64;
    /// The vector's width: the narrowest a vector index takes.
    const DIM: usize = 16;
    /// The columns each append carries, by index into the wide ones.
    const CARRIED: [[usize; 2]; 2] = [[3, 150], [150, 299]];
    /// Neighbours asked of the vector search.
    const K: usize = 5;

    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8 path")).expect("connect");
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let mut fields = vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(Arc::clone(&item), DIM as i32),
            false,
        ),
    ];
    fields
        .extend((0..WIDE_COLUMNS).map(|i| Field::new(format!("c{i}"), DataType::LargeUtf8, true)));
    let docs = db
        .create_table(
            TABLE,
            Arc::new(Schema::new(fields)),
            IndexSpec::new()
                .fts("title")
                .vector("emb", DIM, Metric::L2Sq),
        )
        .expect("create");

    let mut writer = match commits {
        Commits::EachAppend => None,
        Commits::OnceForAll => Some(docs.local_handle().writer().expect("writer")),
    };
    for (append, carried) in CARRIED.iter().enumerate() {
        let values = Float32Array::from(
            (0..ROWS * DIM)
                .map(|i| ((append * ROWS * DIM + i) % 97) as f32)
                .collect::<Vec<_>>(),
        );
        let emb =
            FixedSizeListArray::try_new(Arc::clone(&item), DIM as i32, Arc::new(values), None)
                .expect("vectors");
        let mut columns: Vec<(String, ArrayRef)> = vec![
            (
                "title".into(),
                Arc::new(LargeStringArray::from_iter_values(
                    (0..ROWS).map(|r| format!("word{append} row{r}")),
                )),
            ),
            ("emb".into(), Arc::new(emb)),
        ];
        for &c in carried {
            // Every other row holds a value, the rest are null.
            let cells = LargeStringArray::from_iter(
                (0..ROWS).map(|r| (r % 2 == 0).then(|| format!("a{append} c{c} r{r}"))),
            );
            columns.push((format!("c{c}"), Arc::new(cells)));
        }
        let narrow = batch(
            columns
                .iter()
                .map(|(n, a)| (n.as_str(), Arc::clone(a)))
                .collect(),
        );
        match writer.as_mut() {
            None => {
                docs.append(&narrow).expect("append a narrow batch");
            }
            Some(writer) => {
                writer.append(&narrow).expect("buffer a narrow batch");
            }
        }
    }
    if let Some(mut writer) = writer {
        assert_eq!(
            writer.buffered_batches(),
            CARRIED.len(),
            "every batch waits for the one commit"
        );
        writer.commit().expect("commit the buffered batches");
    }

    let count = |sql: &str| -> String { rows(&db, sql).concat() };
    let total = (ROWS * CARRIED.len()).to_string();
    assert_eq!(count("SELECT count(*) FROM docs"), total);
    for (append, carried) in CARRIED.iter().enumerate() {
        for &c in carried {
            // `rows` sorts what it reads, so the expectation is sorted too.
            let mut expected: Vec<String> = (0..ROWS)
                .step_by(2)
                .map(|r| format!("a{append} c{c} r{r}"))
                .collect();
            expected.sort();
            let read = rows(
                &db,
                &format!("SELECT c{c} FROM docs WHERE c{c} LIKE 'a{append} %'"),
            );
            assert_eq!(read, expected, "c{c} of append {append}");
        }
    }
    // A column no append carried is null in every row; one carried by the
    // first append only is null in the second's rows and its own odd ones.
    assert_eq!(count("SELECT count(c0) FROM docs"), "0");
    assert_eq!(count("SELECT count(*) FROM docs WHERE c0 IS NULL"), total);
    assert_eq!(count("SELECT count(c3) FROM docs"), (ROWS / 2).to_string());
    assert_eq!(count("SELECT count(c150) FROM docs"), ROWS.to_string());

    let hits = docs
        .vector_search("emb", &[0.0f32; DIM], K, None, None)
        .expect("vector search");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), K);
    let text_hits = docs
        .bm25_search("title", "word1", K, Default::default(), None)
        .expect("text search");
    assert_eq!(text_hits.iter().map(|b| b.num_rows()).sum::<usize>(), K);
}

#[test]
fn a_type_that_disagrees_with_the_frozen_one_is_refused() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let err = docs
        .append(&batch(vec![
            ("title", titles(&["a"])),
            ("score", Arc::new(StringArray::from(vec!["7"])) as ArrayRef),
        ]))
        .expect_err("score is frozen at Int64");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::TypeMismatch { column, .. }) if column == "score"),
        "{err}"
    );
}

#[test]
fn a_buffer_of_two_appends_adds_both_columns_in_one_commit() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new())
        .expect("create");
    let mut writer = docs.local_handle().writer().expect("writer");
    writer
        .append(&batch(vec![
            ("title", titles(&["a"])),
            ("x", ints(vec![Some(1)])),
        ]))
        .expect("append x");
    writer
        .append(&batch(vec![
            ("title", titles(&["b"])),
            ("y", ints(vec![Some(2)])),
        ]))
        .expect("append y");
    assert_eq!(
        db.schema(TABLE).expect("schema").schema_id(),
        1,
        "nothing is committed yet"
    );
    writer.commit().expect("commit");
    let schema = db.schema(TABLE).expect("schema");
    assert_eq!(
        names(&schema),
        vec![("title", FieldId(1)), ("x", FieldId(2)), ("y", FieldId(3))]
    );
    assert_eq!(schema.schema_id(), 2, "one commit, one schema change");
    assert_eq!(
        rows(&db, "SELECT title, x, y FROM docs"),
        vec!["a|1|null", "b|null|2"]
    );
}

#[test]
fn a_retype_commits_the_rows_the_writer_is_holding_before_it_lands() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let mut writer = docs.local_handle().writer().expect("writer");
    writer
        .append(&batch(vec![
            ("title", titles(&["a"])),
            ("score", ints(vec![Some(7)])),
        ]))
        .expect("append under the Int64 score");
    // These rows are acknowledged but un-flushed. Retyping `score` out
    // from under them would leave them unable to commit under any schema,
    // and dropping the writer would discard them.
    let doc = writer
        .apply_schema(
            &patch(vec![add("score", DataType::LargeUtf8).with_id(FieldId(2))]),
            None,
        )
        .expect("retype score");
    assert_eq!(doc.fields()[1].data_type, DataType::LargeUtf8);
    writer.commit().expect("commit");
    drop(writer);

    assert_eq!(
        rows(&db, "SELECT title FROM docs"),
        vec!["a"],
        "the acknowledged row is durable"
    );
}

#[test]
fn a_rename_follows_the_rows_the_writer_is_holding_instead_of_minting_a_second_column() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let mut writer = docs.local_handle().writer().expect("writer");
    writer
        .append(&batch(vec![
            ("title", titles(&["a"])),
            ("score", ints(vec![Some(7)])),
        ]))
        .expect("append under the old name");
    let doc = writer
        .apply_schema(
            &patch(vec![add("points", DataType::Int64).with_id(FieldId(2))]),
            None,
        )
        .expect("rename score to points");
    assert_eq!(
        names(&doc),
        vec![("title", FieldId(1)), ("points", FieldId(2))]
    );
    writer.commit().expect("commit");
    drop(writer);

    // One logical column, not two: the buffered rows landed under the old
    // name before the rename, so they read back under the new one rather
    // than bringing `score` back as a column of its own.
    let after = db.schema(TABLE).expect("schema");
    assert_eq!(
        names(&after),
        vec![("title", FieldId(1)), ("points", FieldId(2))]
    );
    assert_eq!(rows(&db, "SELECT title, points FROM docs"), vec!["a|7"]);
    assert!(
        db.query_sql("SELECT score FROM docs").is_err(),
        "the old name is gone"
    );
}

#[test]
fn a_column_cannot_stop_admitting_nulls_over_rows_a_peer_committed() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf8");
    let stale = connect(path).expect("connect");
    stale
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    // The document this handle holds, read while the table is still empty.
    let doc = stale.schema(TABLE).expect("schema");

    let peer = connect(path).expect("connect");
    peer.open_table(TABLE)
        .expect("open")
        .append(&batch(vec![
            ("title", titles(&["a"])),
            ("score", ints(vec![None])),
        ]))
        .expect("the peer commits a null score");

    // An append over known columns leaves `schema_id` where it was, so the
    // expectation the write carries is still satisfied — only the rows
    // moved, and they are what forbids the change.
    let err = stale
        .apply_schema(
            TABLE,
            &patch(vec![
                add("score", DataType::Int64)
                    .with_id(FieldId(2))
                    .with_nullable(false),
            ]),
            Some(doc.schema_id()),
        )
        .expect_err("score cannot stop admitting nulls over a committed null");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::NotEmpty { column }) if column == "score"),
        "{err}"
    );

    let fresh = connect(path).expect("connect");
    let after = fresh.schema(TABLE).expect("schema");
    assert!(
        after.fields()[1].nullable,
        "the document still admits nulls"
    );
    assert_eq!(after.schema_id(), doc.schema_id(), "nothing was published");
}

#[test]
fn the_schema_write_adds_renames_and_drops_and_reads_follow() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append(&batch(vec![
        ("title", titles(&["a"])),
        ("score", ints(vec![Some(1)])),
    ]))
    .expect("append");

    let doc = db
        .apply_schema(TABLE, &patch(vec![add("tag", DataType::LargeUtf8)]), None)
        .expect("add tag");
    assert_eq!(
        names(&doc),
        vec![
            ("title", FieldId(1)),
            ("score", FieldId(2)),
            ("tag", FieldId(3))
        ]
    );
    docs.append(&batch(vec![
        ("title", titles(&["b"])),
        ("score", ints(vec![Some(2)])),
        ("tag", titles(&["t"])),
    ]))
    .expect("append with tag");

    let renamed = db
        .apply_schema(
            TABLE,
            &patch(vec![add("points", DataType::Int64).with_id(FieldId(2))]),
            None,
        )
        .expect("rename score");
    assert_eq!(renamed.name_of(FieldId(2)), Some("points"));
    assert_eq!(
        rows(&db, "SELECT title, points, tag FROM docs"),
        vec!["a|1|null", "b|2|t"]
    );
    assert!(
        db.query_sql("SELECT score FROM docs").is_err(),
        "the old name is gone"
    );

    let dropped = db
        .apply_schema(
            TABLE,
            &patch(vec![add("tag", DataType::LargeUtf8).dropped()]),
            None,
        )
        .expect("drop tag");
    assert_eq!(dropped.tombstoned(), &[FieldId(3)]);
    assert!(db.query_sql("SELECT tag FROM docs").is_err());
    assert_eq!(
        rows(&db, "SELECT title, points FROM docs"),
        vec!["a|1", "b|2"]
    );

    // The name comes back under a new id: the dropped values stay gone.
    let again = db
        .apply_schema(TABLE, &patch(vec![add("tag", DataType::LargeUtf8)]), None)
        .expect("add tag again");
    assert_eq!(again.id_of("tag"), Some(FieldId(4)));
    assert_eq!(rows(&db, "SELECT tag FROM docs"), vec!["null", "null"]);
    assert_eq!(docs.schema().fields().len(), 3);
}

/// Swaps the names of columns `a` and `b`, each keeping its field id.
fn swap_a_and_b(db: &Connection) {
    let before = db
        .apply_schema(TABLE, &patch(vec![]), None)
        .expect("read the schema");
    let a_id = before.id_of("a").expect("a");
    let b_id = before.id_of("b").expect("b");
    for (name, id) in [("tmp", a_id), ("a", b_id), ("b", a_id)] {
        db.apply_schema(
            TABLE,
            &patch(vec![FieldPatch::named(name).with_id(id)]),
            None,
        )
        .expect("rename");
    }
}

/// Two full-text columns that swap names: a superfile written before the
/// swap keys each dictionary by the old name, so every index-backed
/// `WHERE` must find a column's terms by its id, not its current name.
#[test]
fn a_where_on_full_text_columns_follows_a_name_swap() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(
            TABLE,
            Arc::new(Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, false),
                Field::new("a", DataType::LargeUtf8, false),
                Field::new("b", DataType::LargeUtf8, false),
            ])),
            IndexSpec::new().fts("title").fts("a").fts("b"),
        )
        .expect("create");
    docs.append(&batch(vec![
        ("title", titles(&["rust one", "rust two", "rust three"])),
        ("a", titles(&["p", "q", "p"])),
        ("b", titles(&["v", "w", "w"])),
    ]))
    .expect("append");

    swap_a_and_b(&db);
    assert_eq!(
        rows(&db, "SELECT title, a, b FROM docs"),
        vec!["rust one|v|p", "rust three|w|p", "rust two|w|q"]
    );

    let cases: &[(&str, &[&str])] = &[
        ("SELECT title FROM docs WHERE a = 'v'", &["rust one"]),
        ("SELECT title FROM docs WHERE b = 'q'", &["rust two"]),
        (
            "SELECT title FROM docs WHERE a IN ('v', 'x')",
            &["rust one"],
        ),
        ("SELECT title FROM docs WHERE a LIKE 'v'", &["rust one"]),
        ("SELECT title FROM docs WHERE b LIKE 'q%'", &["rust two"]),
        (
            "SELECT title FROM docs WHERE a ILIKE '%w%'",
            &["rust three", "rust two"],
        ),
        (
            "SELECT title FROM bm25_search('docs', 'title', 'rust', 10) WHERE a = 'v'",
            &["rust one"],
        ),
        (
            "SELECT title FROM bm25_search('docs', 'title', 'rust', 10) WHERE b LIKE 'q%'",
            &["rust two"],
        ),
    ];
    for (sql, want) in cases {
        assert_eq!(rows(&db, sql), *want, "{sql}");
    }
}

/// The rows an exact `ILIKE` is checked on — the few a bound leaves, and
/// the doubtful ones a dotted capital I leaves — are read from their
/// stored text, which a superfile written before a name swap holds under
/// the old names: the check must read each column by its id.
#[test]
fn an_exact_where_checks_rows_by_id_after_a_name_swap() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(
            TABLE,
            Arc::new(Schema::new(vec![
                Field::new("title", DataType::LargeUtf8, false),
                Field::new("a", DataType::LargeUtf8, false),
                Field::new("b", DataType::LargeUtf8, false),
            ])),
            IndexSpec::new().fts("title").fts("a").fts("b"),
        )
        .expect("create");
    docs.append(&batch(vec![
        (
            "title",
            titles(&["one", "two", "three", "four", "five", "six"]),
        ),
        ("a", titles(&["qx", "p", "p", "p", "p", "p"])),
        ("b", titles(&["v", "w", "w", "w", "w", "w"])),
    ]))
    .expect("append");
    docs.append(&batch(vec![
        ("title", titles(&["seven"])),
        ("a", titles(&["x\u{130}"])),
        ("b", titles(&["xi"])),
    ]))
    .expect("append");
    swap_a_and_b(&db);

    let cases: &[(&str, &[&str])] = &[
        (
            "SELECT title FROM docs WHERE b LIKE 'q%' AND a ILIKE '%x%'",
            &[],
        ),
        (
            "SELECT title FROM docs WHERE b LIKE 'q%' AND b ILIKE '%x%'",
            &["one"],
        ),
        ("SELECT title FROM docs WHERE b ILIKE '%xi%'", &[]),
        ("SELECT title FROM docs WHERE a ILIKE '%xi%'", &["seven"]),
    ];
    for (sql, want) in cases {
        assert_eq!(rows(&db, sql), *want, "{sql}");
    }
}

#[test]
fn the_read_document_applies_as_a_no_op_and_the_expected_id_guards_the_write() {
    let db = connect("memory://").expect("connect");
    db.create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let doc = db.schema(TABLE).expect("schema");
    let same = db
        .apply_schema(TABLE, &SchemaPatch::from(&doc), None)
        .expect("no-op");
    assert_eq!(same, doc);
    let one = SchemaPatch::new(vec![SchemaPatch::from(&doc).fields.remove(1)]);
    assert_eq!(db.apply_schema(TABLE, &one, None).expect("subset"), doc);

    let err = db
        .apply_schema(
            TABLE,
            &patch(vec![add("tag", DataType::LargeUtf8)]),
            Some(doc.schema_id() + 5),
        )
        .expect_err("stale expectation");
    assert!(matches!(err, InfinoError::Conflict(_)), "{err}");
    assert_eq!(db.schema(TABLE).expect("schema"), doc, "nothing changed");

    let next = db
        .apply_schema(
            TABLE,
            &patch(vec![add("tag", DataType::LargeUtf8)]),
            Some(doc.schema_id()),
        )
        .expect("matching expectation");
    assert_eq!(next.schema_id(), doc.schema_id() + 1);

    let capped = db
        .apply_schema(TABLE, &SchemaPatch::new(vec![]).with_max_fields(3), None)
        .expect("set the cap");
    assert_eq!(capped.max_fields(), 3);
    let err = db
        .apply_schema(TABLE, &patch(vec![add("more", DataType::Int64)]), None)
        .expect_err("over the cap");
    assert!(
        matches!(
            &err,
            InfinoError::Schema(SchemaError::FieldCapExceeded { .. })
        ),
        "{err}"
    );
}

#[test]
fn the_schema_write_creates_an_absent_table_and_create_table_refuses_a_present_one() {
    let db = connect("memory://").expect("connect");
    let doc = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![
                add("title", DataType::LargeUtf8).with_nullable(false),
                add("score", DataType::Int64),
            ])
            .with_max_fields(50),
            None,
        )
        .expect("create through the schema write");
    assert_eq!(
        names(&doc),
        vec![("title", FieldId(1)), ("score", FieldId(2))]
    );
    assert!(!doc.fields()[0].nullable);
    assert_eq!(doc.max_fields(), 50);
    assert_eq!(db.list_tables().expect("tables"), vec![TABLE.to_string()]);

    let err = db
        .create_table(TABLE, title_schema(), IndexSpec::new())
        .expect_err("fail-if-exists");
    assert!(matches!(err, InfinoError::AlreadyExists(_)), "{err}");

    let err = db
        .apply_schema("absent", &patch(vec![add("x", DataType::Int64)]), Some(1))
        .expect_err("an expectation against a table that is not there");
    assert!(matches!(err, InfinoError::Conflict(_)), "{err}");
    let err = db
        .apply_schema("absent", &patch(vec![FieldPatch::named("x")]), None)
        .expect_err("a new table's columns need types");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::TypeRequired { .. })),
        "{err}"
    );
}

#[test]
fn updates_obey_the_append_rules() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let docs = db
        .create_table(TABLE, title_score_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append(&batch(vec![
        ("title", titles(&["a", "b"])),
        ("score", ints(vec![Some(1), Some(2)])),
    ]))
    .expect("append");

    // An omitted nullable column is null for the replaced row: replacement
    // rows are whole rows.
    let stats = docs
        .update(
            col("title").eq(lit("a")),
            &batch(vec![("title", titles(&["a"]))]),
        )
        .expect("update without score");
    assert_eq!(stats.matched(), 1);
    assert_eq!(
        rows(&db, "SELECT title, score FROM docs"),
        vec!["a|null", "b|2"]
    );

    // A column the table has not seen is added by the update.
    docs.update(
        col("title").eq(lit("b")),
        &batch(vec![
            ("title", titles(&["b"])),
            ("score", ints(vec![Some(3)])),
            ("note", titles(&["n"])),
        ]),
    )
    .expect("update with a new column");
    assert_eq!(
        db.schema(TABLE).expect("schema").id_of("note"),
        Some(FieldId(3))
    );
    assert_eq!(
        rows(&db, "SELECT title, score, note FROM docs"),
        vec!["a|null|null", "b|3|n"]
    );

    let err = docs
        .update(
            col("title").eq(lit("b")),
            &batch(vec![
                ("title", titles(&["b"])),
                ("score", Arc::new(StringArray::from(vec!["x"])) as ArrayRef),
            ]),
        )
        .expect_err("a frozen type");
    assert!(matches!(err, InfinoError::Schema(_)), "{err}");
}

#[test]
fn two_writers_adding_different_columns_both_land() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf8");
    let first = connect(path).expect("connect");
    first
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    let second = connect(path).expect("connect");
    let a = first.open_table(TABLE).expect("open");
    let b = second.open_table(TABLE).expect("open");
    // Both handles hold the table at schema 1.
    assert_eq!(b.schema().fields().len(), 1);

    a.append(&batch(vec![
        ("title", titles(&["a"])),
        ("x", ints(vec![Some(1)])),
    ]))
    .expect("first writer adds x");
    // The second handle still believes the schema is at 1: its commit
    // loses the pointer race, reloads the winner's schema, and rebuilds.
    b.append(&batch(vec![
        ("title", titles(&["b"])),
        ("y", ints(vec![Some(2)])),
    ]))
    .expect("second writer adds y");

    let fresh = connect(path).expect("connect");
    let doc = fresh.schema(TABLE).expect("schema");
    assert_eq!(
        names(&doc),
        vec![("title", FieldId(1)), ("x", FieldId(2)), ("y", FieldId(3))]
    );
    assert_eq!(doc.schema_id(), 3);
    assert_eq!(
        rows(&fresh, "SELECT title, x, y FROM docs"),
        vec!["a|1|null", "b|null|2"]
    );

    // The same unseen name in two types: the loser is refused by the
    // winner's schema, after its own append had validated.
    a.append(&batch(vec![
        ("title", titles(&["c"])),
        ("z", ints(vec![Some(3)])),
    ]))
    .expect("first writer adds z as Int64");
    let err = b
        .append(&batch(vec![
            ("title", titles(&["d"])),
            ("z", titles(&["three"])),
        ]))
        .expect_err("z is frozen at Int64 by the winner");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::TypeMismatch { column, .. }) if column == "z"),
        "{err}"
    );
    let fresh = connect(path).expect("connect");
    assert_eq!(
        rows(&fresh, "SELECT title FROM docs"),
        vec!["a", "b", "c"],
        "the refused batch left nothing behind"
    );
}

#[test]
fn compaction_merges_the_generations_growth_leaves_behind() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    // Filler rows beside each generation's marker row: enough bytes for
    // the fill rule to merge the three generations.
    let with_fillers = |marker: &str| -> ArrayRef {
        let mut values = vec![marker.to_string()];
        values.extend((0..FILLER_ROWS).map(|i| format!("filler {i}")));
        Arc::new(LargeStringArray::from(values))
    };
    let marker_then_nulls = |v: i64| -> ArrayRef {
        let mut values = vec![Some(v)];
        values.extend(std::iter::repeat_n(None, FILLER_ROWS));
        Arc::new(Int64Array::from(values))
    };
    docs.append(&batch(vec![("title", with_fillers("a"))]))
        .expect("generation one");
    docs.append(&batch(vec![
        ("title", with_fillers("b")),
        ("score", marker_then_nulls(2)),
    ]))
    .expect("generation two");
    db.apply_schema(
        TABLE,
        &patch(vec![add("points", DataType::Int64).with_id(FieldId(2))]),
        None,
    )
    .expect("rename");
    docs.append(&batch(vec![
        ("title", with_fillers("c")),
        ("points", marker_then_nulls(3)),
        ("note", with_fillers("n")),
    ]))
    .expect("generation three");
    let before = rows(
        &db,
        "SELECT title, points, note FROM docs WHERE title NOT LIKE 'filler%'",
    );
    assert_eq!(before, vec!["a|null|null", "b|2|null", "c|3|n"]);

    docs.optimize(&OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: COMPACT_TARGET_MB,
        min_fill_percent: COMPACT_MIN_FILL_PERCENT,
        ..CompactionSettings::default()
    }))
    .expect("compact");
    assert_eq!(
        docs.local_handle().reader().expect("reader").n_superfiles(),
        1
    );
    assert_eq!(
        rows(
            &db,
            "SELECT title, points, note FROM docs WHERE title NOT LIKE 'filler%'"
        ),
        before
    );
    let hits = docs
        .bm25_search("title", "b", 10, Default::default(), None)
        .expect("search");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

/// Columns the wide, sparse table ends with.
const WIDE_COLUMNS: usize = 300;
/// Commits that write the wide table; each adds a slice of its columns.
const WIDE_GENERATIONS: usize = 6;
/// Rows each commit writes.
const WIDE_ROWS_PER_GENERATION: usize = 400;
/// Columns each row of the wide table fills.
const WIDE_FILLED_PER_ROW: usize = 3;
/// Every this-many-th wide column holds integers; the rest hold strings.
const WIDE_INT_EVERY: usize = 5;

/// The `i`-th column of the wide table.
fn wide_field(i: usize) -> Field {
    match i.is_multiple_of(WIDE_INT_EVERY) {
        true => Field::new(format!("p{i:03}"), DataType::Int64, true),
        false => Field::new(format!("p{i:03}"), DataType::LargeUtf8, true),
    }
}

/// Generation `g`'s rows over the columns it knows: each row fills
/// [`WIDE_FILLED_PER_ROW`] of them, spread so most columns of a commit
/// hold a value somewhere and most of a row's columns are null.
fn wide_generation(g: usize) -> RecordBatch {
    let width = WIDE_COLUMNS * (g + 1) / WIDE_GENERATIONS;
    let filled = |row: usize, column: usize| {
        (0..WIDE_FILLED_PER_ROW).any(|k| (row * 7 + k * 31 + g) % width == column)
    };
    let fields: Vec<Field> = (0..width).map(wide_field).collect();
    let columns: Vec<ArrayRef> = (0..width)
        .map(|c| -> ArrayRef {
            let rows = 0..WIDE_ROWS_PER_GENERATION;
            match c.is_multiple_of(WIDE_INT_EVERY) {
                true => {
                    Arc::new(Int64Array::from_iter(rows.map(|r| {
                        filled(r, c).then_some((g * 1_000_000 + r * 1_000 + c) as i64)
                    })))
                }
                false => Arc::new(LargeStringArray::from_iter(
                    rows.map(|r| filled(r, c).then(|| format!("g{g} r{r} c{c}"))),
                )),
            }
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("wide batch")
}

/// A wide, sparse table — hundreds of nullable columns, each row filling a
/// few, the schema growing commit by commit so the early files predate the
/// late columns — reads back every value unchanged once compaction has
/// merged its generations into one superfile.
#[test]
fn compaction_of_a_wide_sparse_table_keeps_every_value() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let first = Arc::new(Schema::new(
        (0..WIDE_COLUMNS / WIDE_GENERATIONS)
            .map(wide_field)
            .collect::<Vec<_>>(),
    ));
    let wide = db
        .create_table(TABLE, first, IndexSpec::new())
        .expect("create");
    for g in 0..WIDE_GENERATIONS {
        wide.append(&wide_generation(g)).expect("append");
    }
    let projection: Vec<String> = (0..WIDE_COLUMNS)
        .map(|i| wide_field(i).name().clone())
        .collect();
    let sql = format!("SELECT {} FROM docs", projection.join(", "));
    let before = rows(&db, &sql);
    assert_eq!(before.len(), WIDE_GENERATIONS * WIDE_ROWS_PER_GENERATION);
    assert!(
        wide.local_handle().reader().expect("reader").n_superfiles() >= WIDE_GENERATIONS,
        "every generation wrote its own superfiles"
    );

    wide.optimize(&OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: COMPACT_TARGET_MB,
        min_fill_percent: COMPACT_MIN_FILL_PERCENT,
        ..CompactionSettings::default()
    }))
    .expect("compact");
    assert_eq!(
        wide.local_handle().reader().expect("reader").n_superfiles(),
        1
    );
    assert_eq!(rows(&db, &sql), before);
}

/// The whole life of a schema-optional table, in the order a caller meets
/// it: a table with no columns at all, grown by the documents it is given,
/// then changed deliberately through schema writes, with every refusal
/// along the way checked for the reason it gives rather than merely for
/// failing.
#[test]
fn a_table_with_no_schema_grows_from_its_documents_and_changes_by_hand() {
    let db = connect("memory://").expect("connect");

    // ---- 1. A table with no columns -------------------------------------
    // The schema write creates an absent table, so a caller with nothing to
    // declare declares nothing.
    let empty = db
        .apply_schema(TABLE, &SchemaPatch::new(vec![]), None)
        .expect("a table may start with no columns");
    assert!(names(&empty).is_empty(), "no columns yet");
    assert_eq!(empty.schema_id(), 1);
    assert_eq!(db.list_tables().expect("tables"), vec![TABLE.to_string()]);

    let docs = db.open_table(TABLE).expect("open");

    // ---- 2. The documents define the columns -----------------------------
    docs.append_rows(&[
        serde_json::json!({"title": "first", "views": 10, "ratio": 1.5, "live": true}),
        serde_json::json!({"title": "second", "views": 20}),
    ])
    .expect("the first documents define the table");

    let grown = db.schema(TABLE).expect("schema");
    let by_name = |doc: &TableSchema, name: &str| {
        doc.fields()
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.data_type.clone())
    };
    assert_eq!(by_name(&grown, "title"), Some(DataType::LargeUtf8));
    assert_eq!(by_name(&grown, "views"), Some(DataType::Int64));
    assert_eq!(by_name(&grown, "ratio"), Some(DataType::Float64));
    assert_eq!(by_name(&grown, "live"), Some(DataType::Boolean));
    // Ids are minted as the mapper walks the document, and a JSON object's
    // keys arrive sorted, so one document's new columns are numbered
    // alphabetically rather than in the order they were written. Ids are
    // opaque identity, so the order is not a promise to a caller; it is
    // pinned here because a change to it would change which id a rename or
    // a drop addresses.
    assert_eq!(
        names(&grown),
        vec![
            ("live", FieldId(1)),
            ("ratio", FieldId(2)),
            ("title", FieldId(3)),
            ("views", FieldId(4)),
        ]
    );
    let id_of = |doc: &TableSchema, name: &str| {
        doc.fields()
            .iter()
            .find(|f| f.name == name)
            .map(|f| f.id)
            .expect("a live column")
    };
    // A row that omitted a column reads null there, not a default.
    assert_eq!(
        rows(&db, "SELECT title, views, ratio FROM docs ORDER BY title"),
        vec!["first|10|1.5", "second|20|null"]
    );

    // ---- 3. The columns are frozen once seen -----------------------------
    let err = docs
        .append_rows(&[serde_json::json!({"title": "third", "views": "many"})])
        .expect_err("a string does not belong in an integer column");
    assert!(
        matches!(
            &err,
            InfinoError::Schema(SchemaError::TypeMismatch { column, .. }) if column == "views"
        ),
        "{err:?}"
    );
    // The refusal is whole-batch: nothing of that append landed.
    assert_eq!(rows(&db, "SELECT title FROM docs").len(), 2);

    // A float where the column is an integer is refused on the same rule.
    let err = docs
        .append_rows(&[serde_json::json!({"title": "third", "views": 1.5})])
        .expect_err("nor does a float");
    assert!(
        matches!(&err, InfinoError::Schema(SchemaError::TypeMismatch { .. })),
        "{err:?}"
    );

    // ---- 4. Adding a field by hand, with a full-text index ---------------
    // An index may be declared when the column is added.
    let with_body = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![
                FieldPatch::named("body")
                    .with_type(DataType::LargeUtf8)
                    .with_index(ColumnIndex::Fts {
                        stopwords: Stopwords::None,
                        stemmer: Stemmer::None,
                        positions: false,
                        stored: true,
                        bm25: Bm25Params::default(),
                    }),
            ]),
            Some(grown.schema_id()),
        )
        .expect("a new column may carry an index");
    assert_eq!(by_name(&with_body, "body"), Some(DataType::LargeUtf8));
    assert_eq!(with_body.schema_id(), grown.schema_id() + 1);

    docs.append_rows(&[serde_json::json!({"title": "third", "body": "the quick fox"})])
        .expect("append into the indexed column");
    let hits = docs
        .bm25_search("body", "fox", 10, Default::default(), None)
        .expect("the new column is searchable");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    // ---- 5. An index cannot be added to a column that already exists -----
    // The index is part of a column's identity: it is fixed when the column
    // is created, so `title` cannot be made full-text after the fact.
    let err = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![
                FieldPatch::named("title")
                    .with_id(id_of(&with_body, "title"))
                    .with_index(ColumnIndex::Fts {
                        stopwords: Stopwords::None,
                        stemmer: Stemmer::None,
                        positions: false,
                        stored: true,
                        bm25: Bm25Params::default(),
                    }),
            ]),
            None,
        )
        .expect_err("an index is identity, not a setting");
    assert!(
        matches!(
            &err,
            InfinoError::Schema(SchemaError::IdentityChange { attribute })
                if attribute.contains("title")
        ),
        "{err:?}"
    );

    // ---- 6. Rename by id, and drop -------------------------------------
    let renamed = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![
                FieldPatch::named("impressions").with_id(id_of(&with_body, "views")),
            ]),
            None,
        )
        .expect("rename by id");
    assert_eq!(by_name(&renamed, "impressions"), Some(DataType::Int64));
    assert_eq!(by_name(&renamed, "views"), None, "the old label is gone");
    // The rows written under the old label are still there.
    assert_eq!(
        rows(&db, "SELECT impressions FROM docs ORDER BY impressions"),
        vec!["10", "20", "null"]
    );

    let dropped = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![FieldPatch::named("live").dropped()]),
            None,
        )
        .expect("drop a column");
    assert_eq!(by_name(&dropped, "live"), None);
    assert_eq!(
        dropped.tombstoned(),
        &[id_of(&grown, "live")],
        "the id is retired"
    );

    // A dropped id is never reused: the next column minted takes a new one.
    let after_drop = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![
                FieldPatch::named("tag").with_type(DataType::LargeUtf8),
            ]),
            None,
        )
        .expect("add after a drop");
    let tag_id = after_drop
        .fields()
        .iter()
        .find(|f| f.name == "tag")
        .map(|f| f.id)
        .expect("tag");
    assert!(
        tag_id > FieldId(5) && !after_drop.tombstoned().contains(&tag_id),
        "a retired id is not handed out again: {tag_id:?}"
    );

    // ---- 7. The expected-id guard ---------------------------------------
    let stale = after_drop.schema_id() - 1;
    let err = db
        .apply_schema(
            TABLE,
            &SchemaPatch::new(vec![FieldPatch::named("late").with_type(DataType::Int64)]),
            Some(stale),
        )
        .expect_err("the schema has moved since that id");
    assert!(
        matches!(
            &err,
            InfinoError::Conflict(_) | InfinoError::Schema(SchemaError::SchemaConflict { .. })
        ),
        "{err:?}"
    );

    // ---- 8. The document read back applies as a no-op ---------------------
    let doc = db.schema(TABLE).expect("schema");
    let again = db
        .apply_schema(
            TABLE,
            &SchemaPatch::from_json(&doc.to_json()).expect("the document is a patch"),
            None,
        )
        .expect("a document read back changes nothing");
    assert_eq!(
        again.schema_id(),
        doc.schema_id(),
        "a patch that changes nothing commits nothing"
    );

    // ---- 9. Everything still reads ---------------------------------------
    assert_eq!(
        rows(
            &db,
            "SELECT title, impressions, body FROM docs ORDER BY title"
        ),
        vec![
            "first|10|null",
            "second|20|null",
            "third|null|the quick fox"
        ]
    );
}

/// A projection that names only columns the hit's file predates resolves to
/// no stored column in that file, so the read returns no columns at all. It
/// must still report how many rows it covered, because that count is what
/// the caller null-fills from: a batch carrying neither columns nor a stated
/// row count cannot say. Mixed projections never hit this; an all-missing
/// one failed the whole search.
#[test]
fn a_search_projecting_only_a_newer_column_reads_it_as_null() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(
            TABLE,
            Arc::new(Schema::new(vec![Field::new(
                "body",
                DataType::LargeUtf8,
                true,
            )])),
            IndexSpec::new().fts("body"),
        )
        .expect("create");

    // This file predates `title` entirely.
    docs.append_rows(&[serde_json::json!({"body": "the quick fox"})])
        .expect("append");

    db.apply_schema(
        TABLE,
        &SchemaPatch::new(vec![
            FieldPatch::named("title").with_type(DataType::LargeUtf8),
        ]),
        None,
    )
    .expect("add a column the written file cannot hold");

    let hits = docs
        .bm25_search("body", "fox", 10, Default::default(), Some(&["title"]))
        .expect("a projection of only the newer column still reads");
    assert_eq!(
        hits.iter().map(|b| b.num_rows()).sum::<usize>(),
        1,
        "the hit is returned"
    );
    for b in &hits {
        let column = b.column_by_name("title").expect("the projected column");
        for row in 0..b.num_rows() {
            assert!(column.is_null(row), "a column the file predates reads null");
        }
    }
}

/// How many columns the writer adds while the reader runs. Enough that the
/// reader lands mid-flight many times over, few enough to stay quick.
const RACE_GENERATIONS: usize = 24;

/// A reader running *while* the schema moves only ever sees a coherent
/// snapshot: the columns the manifest names and the data behind them land
/// under one pointer, so a reader selecting exactly what the schema reports
/// never finds a column with no data or data under a column the schema has
/// not got.
///
/// The other tests here drive the race by hand — two handles appending in
/// turn — which pins the resolution but never the window. This one has a
/// second thread reading continuously while the first commits, so the reader
/// genuinely observes whatever intermediate states exist.
#[test]
fn a_reader_running_while_the_schema_grows_sees_only_coherent_snapshots() {
    use std::{
        sync::atomic::{AtomicBool, Ordering},
        thread,
    };

    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf8 path");
    let db = connect(path).expect("connect");
    db.create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");

    let writing = Arc::new(AtomicBool::new(true));
    let writer_done = Arc::clone(&writing);
    let writer_path = path.to_owned();
    let writer = thread::spawn(move || {
        let db = connect(&writer_path).expect("connect");
        let table = db.open_table(TABLE).expect("open");
        for generation in 0..RACE_GENERATIONS {
            // Each append adds a column the table has never had, so every
            // commit publishes a new schema and the rows that introduced it
            // together.
            let column = format!("c{generation}");
            table
                .append(&batch(vec![
                    ("title", titles(&["row"])),
                    (column.as_str(), ints(vec![Some(generation as i64)])),
                ]))
                .expect("the writer's append");
        }
        writer_done.store(false, Ordering::SeqCst);
    });

    let mut reads = 0usize;
    let mut widest = 0usize;
    let mut rows_seen = 0usize;
    while writing.load(Ordering::SeqCst) {
        // A fresh connection each pass, so the reader sees the pointer move
        // rather than a handle's cached manifest.
        let reader = connect(path).expect("connect");
        let document = reader.schema(TABLE).expect("schema");
        let columns: Vec<&str> = document.fields().iter().map(|f| f.name.as_str()).collect();
        assert!(
            columns.first() == Some(&"title"),
            "the declared column is always there: {columns:?}"
        );

        // Selecting exactly what the schema reports must work. A torn
        // snapshot shows up here: a column the manifest names whose data
        // never landed cannot be read, and a file holding a column the
        // schema has not got would not be projectable by name.
        let sql = format!("SELECT {} FROM {TABLE}", columns.join(", "));
        let out = rows(&reader, &sql);

        // Every row carries one value per column the schema names, so the
        // manifest and the files agree about the table's width.
        for row in &out {
            assert_eq!(
                row.split('|').count(),
                columns.len(),
                "row {row:?} against columns {columns:?}"
            );
        }

        // Neither the width nor the row count ever goes backwards: a reader
        // moves from one committed generation to a later one, never to a
        // partial one.
        assert!(
            columns.len() >= widest,
            "the schema narrowed from {widest} to {}",
            columns.len()
        );
        assert!(
            out.len() >= rows_seen,
            "the table lost rows: {} after {rows_seen}",
            out.len()
        );
        widest = columns.len();
        rows_seen = out.len();
        reads += 1;
    }
    writer.join().expect("the writer thread");

    assert!(reads > 0, "the reader never got a pass in");
    // The reader must have actually observed the schema moving, or it proved
    // nothing about the race.
    assert!(
        widest > 1,
        "the reader only ever saw the table at its declared width"
    );

    // Everything the writer committed is there at the end.
    let fresh = connect(path).expect("connect");
    let document = fresh.schema(TABLE).expect("schema");
    assert_eq!(
        document.fields().len(),
        RACE_GENERATIONS + 1,
        "title plus one column per generation"
    );
    assert_eq!(
        rows(&fresh, &format!("SELECT title FROM {TABLE}")).len(),
        RACE_GENERATIONS,
        "one row per generation"
    );
}

/// A text column added after the table was created is not among the FTS
/// columns the table was created with, yet `text_match` on it is answered
/// from its postings, and a superfile written before the column existed,
/// which holds no postings for it, has its rows checked against their
/// (null) text. Every answer is `token_match`'s.
#[test]
fn a_text_match_on_a_text_column_added_later_answers_as_token_match_does() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append(&batch(vec![("title", titles(&["rust one", "go two"]))]))
        .expect("append before the column");
    db.apply_schema(
        TABLE,
        &SchemaPatch::new(vec![
            FieldPatch::named("body")
                .with_type(DataType::LargeUtf8)
                .with_index(ColumnIndex::Fts {
                    stopwords: Stopwords::None,
                    stemmer: Stemmer::None,
                    positions: true,
                    stored: true,
                    bm25: Bm25Params::default(),
                }),
        ]),
        None,
    )
    .expect("add an indexed text column");
    docs.append_rows(&[
        serde_json::json!({"title": "three", "body": "rust and Go"}),
        serde_json::json!({"title": "four", "body": "nothing here"}),
    ])
    .expect("append with the column");

    for (query, mode, want) in [
        ("rust", "or", &["three"][..]),
        ("go rust", "and", &["three"][..]),
        ("nothing missing", "or", &["four"][..]),
        ("zzq", "or", &[][..]),
    ] {
        let predicate =
            format!("SELECT title FROM {TABLE} WHERE text_match(body, '{query}', '{mode}')");
        assert_eq!(rows(&db, &predicate), want, "{predicate}");
        let function = format!(
            "SELECT title FROM {TABLE} WHERE _id IN \
             (SELECT _id FROM token_match('{TABLE}', 'body', '{query}', '{mode}'))"
        );
        assert_eq!(rows(&db, &function), want, "{function}");
    }
    // Beside a leaf on the column the table was created with: both files.
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT title FROM {TABLE} WHERE text_match(title, 'rust') OR text_match(body, 'rust')"
            )
        ),
        vec!["rust one", "three"]
    );
    assert_eq!(
        rows(
            &db,
            &format!(
                "SELECT COUNT(*) FROM {TABLE} WHERE text_match(title, 'go') OR text_match(body, 'go')"
            )
        ),
        vec!["2"]
    );
    // Exact: nothing in the plan evaluates it.
    let plan = rows(
        &db,
        &format!(
            "EXPLAIN SELECT title FROM {TABLE} WHERE text_match(title, 'rust') OR text_match(body, 'rust')"
        ),
    );
    let physical: Vec<&String> = plan
        .iter()
        .filter(|r| r.starts_with("physical_plan"))
        .collect();
    assert!(!physical.is_empty(), "{plan:?}");
    assert!(
        physical.iter().all(|r| !r.contains("text_match")),
        "{plan:?}"
    );
}
