// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Rows as JSON documents: the mapper gives each path one type under the
//! number rule, templates decide what a new path becomes, the caps refuse a
//! document whole, and every refusal names the schema write that lets the
//! same rows in.

use std::{fs, path::Path, sync::Arc};

use arrow_array::{Array, Float64Array, Int64Array, LargeStringArray};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    Connection, FieldPatch, IndexSpec, InfinoError, SchemaPatch, connect,
    serde_json::{self, Value, json},
};
use tempfile::TempDir;

const TABLE: &str = "docs";
/// Where the shared corpus and its frozen schema document live; the Python
/// and Node suites read the same files.
const FIXTURES: &str = "tests/fixtures/dynamic";
/// Set to rewrite the frozen schema document from this engine's output.
const UPDATE_FIXTURES: &str = "INFINO_UPDATE_FIXTURES";

fn title_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]))
}

fn corpus() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURES)
        .join("rows.jsonl");
    fs::read_to_string(&path)
        .expect("corpus")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON document per line"))
        .collect()
}

/// `column`'s values as strings in id order, `null` for nulls.
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
            } else if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
                out.push(a.value(row).to_string());
            } else {
                out.push(format!("{:?}", array.slice(row, 1)));
            }
        }
    }
    out
}

fn schema_error(result: Result<(), InfinoError>, expected: &str) {
    match result {
        Err(InfinoError::Schema(m)) if m.contains(expected) => {}
        other => panic!("expected a schema error mentioning {expected:?}, got {other:?}"),
    }
}

#[test]
fn documents_grow_the_schema_and_read_back() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append_rows(&corpus()).expect("append rows");

    let doc = db.schema(TABLE).expect("schema");
    let names: Vec<&str> = doc.fields().iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "title",
            "author.age",
            "author.name",
            "published",
            "score",
            "tags",
            "views",
            "meta.attempt",
            "meta.source",
            "author.handles",
            "ratings",
            "comments.stars",
            "comments.user",
        ],
        "paths join as documents introduce them, sorted within a document; `notes` and `empty` add nothing"
    );
    let typed = |name: &str| {
        doc.fields()
            .iter()
            .find(|f| f.name == name)
            .expect(name)
            .data_type
            .clone()
    };
    assert_eq!(typed("views"), DataType::Int64);
    assert_eq!(
        typed("score"),
        DataType::Float64,
        "4.5 then 3: a number path is a float"
    );
    assert_eq!(typed("published"), DataType::Boolean);
    assert!(
        matches!(typed("tags"), DataType::List(item) if item.data_type() == &DataType::LargeUtf8)
    );
    assert!(
        matches!(typed("ratings"), DataType::List(item) if item.data_type() == &DataType::Float64)
    );
    assert!(
        matches!(typed("comments.stars"), DataType::List(item) if item.data_type() == &DataType::Int64)
    );

    assert_eq!(column(&db, "views"), vec!["10", "25", "null", "7"]);
    assert_eq!(column(&db, "score"), vec!["4.5", "null", "3", "null"]);
    assert_eq!(column(&db, "author.name"), vec!["ann", "bob", "cy", "null"]);
    let hits = docs
        .bm25_search("title", "post", 10, Default::default(), None)
        .expect("search");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
}

#[test]
fn the_frozen_document_matches_the_shared_fixture() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append_rows(&corpus()).expect("append rows");
    let canonical = db.schema(TABLE).expect("schema").to_json().to_string();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURES)
        .join("schema.json");
    if std::env::var_os(UPDATE_FIXTURES).is_some() {
        fs::write(&path, format!("{canonical}\n")).expect("write fixture");
    }
    let frozen = fs::read_to_string(&path).expect("frozen schema document");
    assert_eq!(
        canonical,
        frozen.trim(),
        "the document this engine freezes for the corpus; rerun with {UPDATE_FIXTURES}=1 to refreeze"
    );
}

#[test]
fn every_refusal_names_the_schema_write_that_admits_the_rows() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new())
        .expect("create");
    docs.append_rows(&[json!({"title": "a", "n": 1})])
        .expect("n is Int64 from here");

    // A fraction, `5.0` and a string are all refused on an Int64 path, and
    // a retype to Float64 admits the numbers.
    schema_error(
        docs.append_rows(&[json!({"title": "b", "n": 1.5})]),
        "Int64",
    );
    schema_error(
        docs.append_rows(&[json!({"title": "b", "n": 5.0})]),
        "Int64",
    );
    schema_error(
        docs.append_rows(&[json!({"title": "b", "n": "42"})]),
        "Int64",
    );
    db.apply_schema(
        TABLE,
        &SchemaPatch {
            fields: vec![FieldPatch {
                id: None,
                name: "n".into(),
                data_type: Some(DataType::Float64),
                nullable: None,
                index: None,
                dropped: false,
            }],
            max_fields: None,
            max_depth: None,
            templates: None,
        },
        None,
    )
    .expect("retype");
    docs.append_rows(&[
        json!({"title": "b", "n": 1.5}),
        json!({"title": "c", "n": 5}),
    ])
    .expect("floats and integral literals fit a Float64 column");
    assert_eq!(column(&db, "n"), vec!["1", "1.5", "5"]);

    // A mixed array is refused outright.
    schema_error(
        docs.append_rows(&[json!({"title": "d", "x": [1, "a"]})]),
        "more than one type",
    );

    // Caps refuse the document whole, and raising the cap admits it.
    let caps = |max_fields: Option<u32>, max_depth: Option<u32>| SchemaPatch {
        fields: vec![],
        max_fields,
        max_depth,
        templates: None,
    };
    db.apply_schema(TABLE, &caps(Some(3), None), None)
        .expect("cap");
    schema_error(
        docs.append_rows(&[json!({"title": "e", "p": 1, "q": 2})]),
        "over its cap",
    );
    assert_eq!(
        db.schema(TABLE).expect("schema").fields().len(),
        2,
        "nothing was added"
    );
    db.apply_schema(TABLE, &caps(Some(10), None), None)
        .expect("raise");
    docs.append_rows(&[json!({"title": "e", "p": 1, "q": 2})])
        .expect("within the cap");

    db.apply_schema(TABLE, &caps(None, Some(2)), None)
        .expect("depth");
    schema_error(
        docs.append_rows(&[json!({"title": "f", "a": {"b": {"c": 1}}})]),
        "nests deeper",
    );
    db.apply_schema(TABLE, &caps(None, Some(3)), None)
        .expect("deeper");
    docs.append_rows(&[json!({"title": "f", "a": {"b": {"c": 1}}})])
        .expect("within the depth");
    assert!(db.schema(TABLE).expect("schema").id_of("a.b.c").is_some());
}

#[test]
fn templates_decide_what_a_new_path_becomes() {
    let db = connect("memory://").expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new())
        .expect("create");
    let templates = serde_json::from_value(json!({
        "templates": [
            {"name": "prices", "match": "integer", "path": "*_price", "type": "f64"},
            {"name": "text", "match": "string", "path": "body", "index": {"kind": "fts"}},
            {"name": "when", "path": "*_at", "type": "timestamp_us", "tz": "UTC"}
        ]
    }))
    .expect("json");
    let patch = SchemaPatch::from_json(&templates).expect("patch");
    let doc = db.apply_schema(TABLE, &patch, None).expect("templates");
    assert_eq!(doc.templates().len(), 3);

    docs.append_rows(&[json!({
        "title": "t",
        "list_price": 5,
        "body": "the quick brown fox",
        "created_at": "2026-10-03T12:00:00Z"
    })])
    .expect("append");
    let doc = db.schema(TABLE).expect("schema");
    let field = |name: &str| doc.fields().iter().find(|f| f.name == name).expect(name);
    assert_eq!(field("list_price").data_type, DataType::Float64);
    assert!(field("body").index.is_some(), "the template's index");
    assert!(matches!(
        field("created_at").data_type,
        DataType::Timestamp(_, _)
    ));
    let hits = docs
        .bm25_search("body", "fox", 10, Default::default(), None)
        .expect("the new column is searchable from its first file");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    docs.append_rows(&[json!({"title": "u", "body": "a lazy dog"})])
        .expect("append");
    let hits = docs
        .bm25_search("body", "dog", 10, Default::default(), None)
        .expect("search");
    assert_eq!(hits.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
}

#[test]
fn update_rows_obey_the_same_rules() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let docs = db
        .create_table(TABLE, title_schema(), IndexSpec::new().fts("title"))
        .expect("create");
    docs.append_rows(&[json!({"title": "a", "n": 1}), json!({"title": "b", "n": 2})])
        .expect("append");
    let stats = docs
        .update_rows(
            col("title").eq(lit("a")),
            &[json!({"title": "a", "note": "edited"})],
        )
        .expect("update with a new column");
    assert_eq!(stats.matched(), 1);
    // The replaced row lands as a new row, so it reads after `b`.
    assert_eq!(column(&db, "note"), vec!["null", "edited"]);
    assert_eq!(
        column(&db, "n"),
        vec!["2", "null"],
        "replacement rows are whole rows"
    );
    let err = docs
        .update_rows(
            col("title").eq(lit("b")),
            &[json!({"title": "b", "n": 2.5})],
        )
        .expect_err("a frozen type");
    assert!(matches!(err, InfinoError::Schema(_)), "{err}");
}
