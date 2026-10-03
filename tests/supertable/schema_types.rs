// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A column's type changes through the schema write in one of two ways: a
//! lossless widening is metadata the files catch up with, a lossy rewrite
//! flips the type at once and compaction converts the files. Reads give
//! the same rows throughout, nulls included.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch,
};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    CompactionSettings, Connection, FieldId, FieldPatch, IndexSpec, InfinoError, Metric,
    OptimizeOptions, SchemaPatch, connect,
};
use tempfile::TempDir;

const TABLE: &str = "t";
const COMPACT_TARGET_MB: u64 = 1;
const COMPACT_MIN_FILL_PERCENT: u8 = 1;
const VECTOR_DIM: usize = 16;

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

fn strings(values: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(LargeStringArray::from(values))
}

fn ints(values: Vec<Option<i64>>) -> ArrayRef {
    Arc::new(Int64Array::from(values))
}

fn retype(name: &str, to: DataType) -> SchemaPatch {
    SchemaPatch {
        fields: vec![FieldPatch {
            id: None,
            name: name.into(),
            data_type: Some(to),
            nullable: None,
            index: None,
            dropped: false,
        }],
        max_fields: None,
        max_depth: None,
        templates: None,
    }
}

fn compact() -> OptimizeOptions {
    OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: COMPACT_TARGET_MB,
        min_fill_percent: COMPACT_MIN_FILL_PERCENT,
        ..CompactionSettings::default()
    })
}

/// `column`'s values as strings, in id order; nulls as `null`.
fn column(db: &Connection, column: &str) -> Vec<String> {
    let batches = db
        .query_sql(&format!("SELECT \"{column}\" FROM {TABLE} ORDER BY _id"))
        .expect("query");
    let mut out = Vec::new();
    for b in &batches {
        let array = b.column(0);
        for row in 0..b.num_rows() {
            if array.is_null(row) {
                out.push("null".to_string());
            } else if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
                out.push(a.value(row).to_string());
            } else if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
                out.push(a.value(row).to_string());
            } else if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
                out.push(a.value(row).to_string());
            } else {
                panic!("unexpected column type {:?}", array.data_type());
            }
        }
    }
    out
}

/// The type every committed file holds column `id` in, read through a
/// fresh connection so the writing handle's snapshot does not answer.
fn physical_types(path: &str, id: FieldId) -> Vec<DataType> {
    let fresh = connect(path).expect("connect");
    let table = fresh.open_table(TABLE).expect("open");
    let reader = table.local_handle().reader().expect("reader");
    let mut types: Vec<DataType> = reader
        .manifest()
        .get_all_superfiles()
        .iter()
        .map(|entry| {
            entry
                .physical_schema
                .as_ref()
                .expect("a committed file records its physical schema")
                .column_by_id(id)
                .expect("the file holds the column")
                .data_type
                .clone()
        })
        .collect();
    types.dedup();
    types
}

fn storage_table(
    schema: Arc<Schema>,
    indexes: IndexSpec,
) -> (TempDir, Connection, infino::Supertable) {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let table = db.create_table(TABLE, schema, indexes).expect("create");
    (dir, db, table)
}

#[test]
fn a_widening_is_metadata_and_the_files_catch_up_on_optimize() {
    let (dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)])),
        IndexSpec::new(),
    );
    let path = dir.path().to_str().expect("utf8");
    table
        .append(&batch(vec![(
            "n",
            Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrayRef,
        )]))
        .expect("append");

    let doc = db
        .apply_schema(TABLE, &retype("n", DataType::Int64), None)
        .expect("widen");
    let n = doc.id_of("n").expect("n");
    assert_eq!(doc.fields()[0].data_type, DataType::Int64);
    assert!(
        doc.fields()[0].converting_from.is_none(),
        "nothing is lost, so nothing is converting"
    );
    assert_eq!(column(&db, "n"), vec!["1", "2"]);
    assert_eq!(physical_types(path, n), vec![DataType::Int32]);

    let err = table
        .append(&batch(vec![(
            "n",
            Arc::new(Int32Array::from(vec![Some(3)])) as ArrayRef,
        )]))
        .expect_err("the frozen type is now Int64");
    assert!(
        matches!(&err, InfinoError::Schema(m) if m.contains("Int64")),
        "{err}"
    );
    table
        .append(&batch(vec![("n", ints(vec![Some(3)]))]))
        .expect("the new type is accepted");
    assert_eq!(column(&db, "n"), vec!["1", "2", "3"]);

    table.optimize(&compact()).expect("optimize");
    assert_eq!(column(&db, "n"), vec!["1", "2", "3"]);
    assert_eq!(physical_types(path, n), vec![DataType::Int64]);
    assert_eq!(
        db.schema(TABLE).expect("schema").schema_id(),
        doc.schema_id(),
        "a widening needs no clearing commit"
    );
}

#[test]
fn a_rewrite_flips_at_once_and_compaction_converts_the_files() {
    let (dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::LargeUtf8,
            true,
        )])),
        IndexSpec::new(),
    );
    let path = dir.path().to_str().expect("utf8");
    table
        .append(&batch(vec![(
            "s",
            strings(vec![Some("1"), Some("x"), Some("3")]),
        )]))
        .expect("append strings");

    let flipped = db
        .apply_schema(TABLE, &retype("s", DataType::Int64), None)
        .expect("flip");
    let s = flipped.id_of("s").expect("s");
    assert_eq!(flipped.fields()[0].data_type, DataType::Int64);
    assert_eq!(
        flipped.fields()[0].converting_from,
        Some(DataType::LargeUtf8)
    );
    // Unconverted files are cast as they are read; what cannot cast is null.
    assert_eq!(column(&db, "s"), vec!["1", "null", "3"]);

    let err = table
        .append(&batch(vec![("s", strings(vec![Some("4")]))]))
        .expect_err("strings are refused from the flip");
    assert!(
        matches!(&err, InfinoError::Schema(m) if m.contains("Int64")),
        "{err}"
    );
    table
        .append(&batch(vec![("s", ints(vec![Some(7)]))]))
        .expect("ints are accepted from the flip");
    assert_eq!(column(&db, "s"), vec!["1", "null", "3", "7"]);
    assert_eq!(
        physical_types(path, s),
        vec![DataType::LargeUtf8, DataType::Int64]
    );

    let err = db
        .apply_schema(TABLE, &retype("s", DataType::Float64), None)
        .expect_err("a second change while converting");
    assert!(
        matches!(&err, InfinoError::Schema(m) if m.contains("converting")),
        "{err}"
    );

    table.optimize(&compact()).expect("optimize converts");
    assert_eq!(column(&db, "s"), vec!["1", "null", "3", "7"]);
    assert_eq!(physical_types(path, s), vec![DataType::Int64]);
    let cleared = db.schema(TABLE).expect("schema");
    assert!(cleared.fields()[0].converting_from.is_none());
    assert_eq!(cleared.schema_id(), flipped.schema_id() + 1);
    db.apply_schema(TABLE, &retype("s", DataType::Float64), None)
        .expect("the column can change again once converted");
}

#[test]
fn retyping_back_before_optimize_reads_the_originals() {
    let (_dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::LargeUtf8,
            true,
        )])),
        IndexSpec::new(),
    );
    table
        .append(&batch(vec![(
            "s",
            strings(vec![Some("1"), Some("x"), Some("3")]),
        )]))
        .expect("append strings");
    db.apply_schema(TABLE, &retype("s", DataType::Int64), None)
        .expect("flip");
    table
        .append(&batch(vec![("s", ints(vec![Some(7)]))]))
        .expect("an int file under the new type");
    assert_eq!(column(&db, "s"), vec!["1", "null", "3", "7"]);

    let back = db
        .apply_schema(TABLE, &retype("s", DataType::LargeUtf8), None)
        .expect("back to the old type is the one change allowed");
    assert_eq!(back.fields()[0].data_type, DataType::LargeUtf8);
    assert_eq!(back.fields()[0].converting_from, Some(DataType::Int64));
    // The unconverted file reads its originals; the int file is cast.
    assert_eq!(column(&db, "s"), vec!["1", "x", "3", "7"]);

    table.optimize(&compact()).expect("optimize");
    assert_eq!(column(&db, "s"), vec!["1", "x", "3", "7"]);
    assert!(
        db.schema(TABLE).expect("schema").fields()[0]
            .converting_from
            .is_none()
    );
}

#[test]
fn a_file_converted_in_between_keeps_the_cast_result() {
    let (_dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new(
            "s",
            DataType::LargeUtf8,
            true,
        )])),
        IndexSpec::new(),
    );
    table
        .append(&batch(vec![(
            "s",
            strings(vec![Some("1"), Some("x"), Some("3")]),
        )]))
        .expect("append strings");
    db.apply_schema(TABLE, &retype("s", DataType::Int64), None)
        .expect("flip");
    table.optimize(&compact()).expect("convert");
    db.apply_schema(TABLE, &retype("s", DataType::LargeUtf8), None)
        .expect("flip back");
    assert_eq!(
        column(&db, "s"),
        vec!["1", "null", "3"],
        "the value that did not cast is gone"
    );
}

#[test]
fn an_fts_column_retyped_to_integers_loses_its_index() {
    let (_dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new(
            "title",
            DataType::LargeUtf8,
            false,
        )])),
        IndexSpec::new().fts("title"),
    );
    table
        .append(&batch(vec![(
            "title",
            strings(vec![Some("alpha"), Some("42")]),
        )]))
        .expect("append");
    let hits = table
        .bm25_search("title", "alpha", 10, Default::default(), None)
        .expect("search");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    let flipped = db
        .apply_schema(TABLE, &retype("title", DataType::Int64), None)
        .expect("flip");
    assert!(flipped.fields()[0].index.is_none());
    assert!(flipped.fields()[0].nullable, "a lossy rewrite admits nulls");
    assert!(
        table
            .bm25_search("title", "alpha", 10, Default::default(), None)
            .is_err(),
        "no full-text index on an integer column"
    );
    assert_eq!(column(&db, "title"), vec!["null", "42"]);
    table.optimize(&compact()).expect("optimize");
    assert_eq!(column(&db, "title"), vec!["null", "42"]);
}

#[test]
fn the_column_behind_the_vector_index_cannot_be_retyped_or_dropped() {
    let item = Arc::new(Field::new("item", DataType::Float32, true));
    let (_dir, db, table) = storage_table(
        Arc::new(Schema::new(vec![Field::new(
            "emb",
            DataType::FixedSizeList(Arc::clone(&item), VECTOR_DIM as i32),
            false,
        )])),
        IndexSpec::new().vector("emb", VECTOR_DIM, Metric::L2Sq),
    );
    let values = Float32Array::from((0..VECTOR_DIM * 2).map(|i| i as f32).collect::<Vec<_>>());
    let emb = FixedSizeListArray::try_new(item, VECTOR_DIM as i32, Arc::new(values), None)
        .expect("vectors");
    table
        .append(&batch(vec![("emb", Arc::new(emb) as ArrayRef)]))
        .expect("append vectors");

    let err = db
        .apply_schema(TABLE, &retype("emb", DataType::LargeUtf8), None)
        .expect_err("the vector index is built on it");
    assert!(
        matches!(&err, InfinoError::Schema(m) if m.contains("vector index")),
        "{err}"
    );
    let drop = SchemaPatch {
        fields: vec![FieldPatch {
            dropped: true,
            ..retype("emb", DataType::LargeUtf8).fields.remove(0)
        }],
        max_fields: None,
        max_depth: None,
        templates: None,
    };
    let err = db
        .apply_schema(TABLE, &drop, None)
        .expect_err("nor dropped");
    assert!(
        matches!(&err, InfinoError::Schema(m) if m.contains("vector index")),
        "{err}"
    );
}
