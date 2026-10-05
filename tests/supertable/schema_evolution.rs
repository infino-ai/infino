// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The schema grows from the data and changes through one declarative
//! write, and every read follows: appends that add a column, batches that
//! omit one, the schema patch's add/rename/drop, updates under the same
//! rules, two writers racing on the schema, and compaction over the
//! generations the growth leaves behind.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    CompactionSettings, Connection, FieldId, FieldPatch, IndexSpec, InfinoError, OptimizeOptions,
    SchemaError, SchemaPatch, TableSchema, connect,
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
    FieldPatch {
        id: None,
        name: name.into(),
        data_type: Some(data_type),
        nullable: None,
        index: None,
        dropped: false,
    }
}

fn patch(fields: Vec<FieldPatch>) -> SchemaPatch {
    SchemaPatch {
        fields,
        max_fields: None,
        max_depth: None,
        templates: None,
    }
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
            &patch(vec![FieldPatch {
                id: Some(FieldId(2)),
                ..add("score", DataType::LargeUtf8)
            }]),
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
            &patch(vec![FieldPatch {
                id: Some(FieldId(2)),
                ..add("points", DataType::Int64)
            }]),
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
            &patch(vec![FieldPatch {
                id: Some(FieldId(2)),
                nullable: Some(false),
                ..add("score", DataType::Int64)
            }]),
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
            &patch(vec![FieldPatch {
                id: Some(FieldId(2)),
                ..add("points", DataType::Int64)
            }]),
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
            &patch(vec![FieldPatch {
                dropped: true,
                ..add("tag", DataType::LargeUtf8)
            }]),
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
    let one = SchemaPatch {
        fields: vec![SchemaPatch::from(&doc).fields.remove(1)],
        max_fields: None,
        max_depth: None,
        templates: None,
    };
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
        .apply_schema(
            TABLE,
            &SchemaPatch {
                fields: vec![],
                max_fields: Some(3),
                max_depth: None,
                templates: None,
            },
            None,
        )
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
            &SchemaPatch {
                fields: vec![
                    FieldPatch {
                        nullable: Some(false),
                        ..add("title", DataType::LargeUtf8)
                    },
                    add("score", DataType::Int64),
                ],
                max_fields: Some(50),
                max_depth: None,
                templates: None,
            },
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
        .apply_schema(
            "absent",
            &patch(vec![FieldPatch {
                data_type: None,
                ..add("x", DataType::Null)
            }]),
            None,
        )
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
        &patch(vec![FieldPatch {
            id: Some(FieldId(2)),
            ..add("points", DataType::Int64)
        }]),
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
