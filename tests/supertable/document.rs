// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Document columns end to end, through the public API: documents of every
//! shape a writer meets are appended as typed columns plus a skeleton and a
//! residual, and the rebuilt `source` equals each original compact JSON
//! byte for byte through SQL (`SELECT *`, a filtered page, a filter on the
//! document itself, a subquery), the search methods (`exact_match`,
//! `token_match`, `sorted_search`), after compaction, after a rename of a
//! column the documents draw on, and after the connection is reopened.
//!
//! A statement whose plan only filters, sorts and limits on other columns
//! reads the document after its limit, for the rows the limit keeps
//! (`LateDocumentExec`); one that reads the document below its limit, or
//! has none, rebuilds it over the scan (`DocumentExec`). Both give the same
//! documents, and the late read finds an updated row's new copy, never a
//! deleted one, through compaction.
//!
//! The test is its own writer: it builds each skeleton by the rule
//! [`infino::document`] gives a writer — a value becomes a slot when
//! `render_value` of what the column holds equals the token as written,
//! and goes to the residual otherwise.

use std::sync::Arc;

use arrow::{compute::cast, util::pretty::pretty_format_batches};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, RecordBatch,
    StringArray, TimestampMillisecondArray,
    builder::{Int64Builder, ListBuilder, StringBuilder},
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::prelude::{col, lit};
use infino::{
    BoolMode, Connection, FieldPatch, IndexSpec, OptimizeOptions, SchemaPatch, TableSchema,
    connect,
    document::{Slot, render_value, residual, residual_column, skeleton_column},
};
use tempfile::TempDir;

/// The table under test.
const TABLE: &str = "logs";
/// The document column.
const SOURCE: &str = "source";
/// Rows in the first commit; the rest go in a second, so the table is two
/// superfiles until it is compacted.
const FIRST_COMMIT_ROWS: usize = 4;
/// Rows a filtered page asks for.
const PAGE: usize = 3;

/// A document as the test writes it: its shape, and for each value either
/// the literal text kept in the skeleton or the column it was stored in
/// with the token text the document carried.
enum J {
    Obj(Vec<(&'static str, J)>),
    Arr(Vec<J>),
    /// Kept as written in the skeleton: `null`, `{}`, `[]`, or a value no
    /// column holds.
    Lit(&'static str),
    /// The row's value in a column, written as `token` in the document.
    Col(&'static str, &'static str),
    /// Element `index` of the row's value in a list column, written as
    /// `token`.
    Elem(&'static str, usize, &'static str),
}

/// One row: its typed columns and its document (`None`: the row has none).
struct Row {
    doc_id: &'static str,
    ts: i64,
    host_name: Option<&'static str>,
    path: Option<&'static str>,
    bytes: Option<i64>,
    ratio: Option<f64>,
    ok: Option<bool>,
    tags: Option<Vec<&'static str>>,
    user_names: Option<Vec<Option<&'static str>>>,
    user_ages: Option<Vec<Option<i64>>>,
    created: Option<i64>,
    message: Option<&'static str>,
    doc: Option<J>,
}

impl Row {
    /// A row with only its id and time set.
    fn new(doc_id: &'static str, ts: i64) -> Self {
        Self {
            doc_id,
            ts,
            host_name: None,
            path: None,
            bytes: None,
            ratio: None,
            ok: None,
            tags: None,
            user_names: None,
            user_ages: None,
            created: None,
            message: None,
            doc: None,
        }
    }
}

/// `2023-01-01T00:00:00Z` in milliseconds since the epoch.
const JAN_1: i64 = 1_672_531_200_000;
/// Milliseconds in an hour and in a day.
const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;

/// The rows: dotted keys beside nested objects, arrays of objects with
/// members missing from some entries, a one-element array beside a scalar
/// in the same list column, nulls, empty objects and arrays, keys out of
/// order, escapes, a float the column cannot render as written and a date
/// written in another format (both residual), and a row with no document.
fn rows() -> Vec<Row> {
    use J::*;
    vec![
        Row {
            host_name: Some("web-1"),
            path: Some("/var/log/a.log"),
            bytes: Some(512),
            ratio: Some(0.25),
            ok: Some(true),
            tags: Some(vec!["a", "b"]),
            message: Some("hello \"world\"\n"),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d0""#)),
                ("@timestamp", Col("ts", r#""2023-01-01T09:37:20.000Z""#)),
                ("host", Obj(vec![("name", Col("host.name", r#""web-1""#))])),
                ("log.file.path", Col("log.file.path", r#""/var/log/a.log""#)),
                ("bytes", Col("bytes", "512")),
                ("ratio", Col("ratio", "0.25")),
                ("ok", Col("ok", "true")),
                (
                    "tags",
                    Arr(vec![Elem("tags", 0, r#""a""#), Elem("tags", 1, r#""b""#)]),
                ),
                ("message", Col("message", r#""hello \"world\"\n""#)),
            ])),
            ..Row::new("d0", JAN_1 + 9 * HOUR + 37 * 60_000 + 20_000)
        },
        Row {
            path: Some("/var/log/b.log"),
            bytes: Some(0),
            ratio: Some(1.0),
            ok: Some(false),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d1""#)),
                ("@timestamp", Col("ts", r#""2023-01-01T10:00:00.000Z""#)),
                (
                    "log",
                    Obj(vec![(
                        "file",
                        Obj(vec![("path", Col("log.file.path", r#""/var/log/b.log""#))]),
                    )]),
                ),
                ("bytes", Col("bytes", "0")),
                // `1` is how the column renders 1.0, so this goes to the residual.
                ("ratio", Col("ratio", "1.0")),
                ("ok", Col("ok", "false")),
                ("meta", Lit("{}")),
                ("list", Lit("[]")),
                ("extra", Lit("null")),
            ])),
            ..Row::new("d1", JAN_1 + 10 * HOUR)
        },
        Row {
            tags: Some(vec!["solo"]),
            user_names: Some(vec![Some("ann"), None, Some("bob")]),
            user_ages: Some(vec![Some(31), Some(40), None]),
            bytes: Some(2048),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d2""#)),
                ("@timestamp", Col("ts", r#""2023-01-02T00:00:00.000Z""#)),
                (
                    "users",
                    Arr(vec![
                        Obj(vec![
                            ("name", Elem("users.name", 0, r#""ann""#)),
                            ("age", Elem("users.age", 0, "31")),
                        ]),
                        Obj(vec![("age", Elem("users.age", 1, "40"))]),
                        Obj(vec![("name", Elem("users.name", 2, r#""bob""#))]),
                    ]),
                ),
                // A scalar beside other documents' arrays in one list column.
                ("tags", Elem("tags", 0, r#""solo""#)),
                ("bytes", Col("bytes", "2048")),
            ])),
            ..Row::new("d2", JAN_1 + DAY)
        },
        Row {
            tags: Some(vec!["solo"]),
            created: Some(JAN_1 + DAY),
            ok: Some(true),
            doc: Some(Obj(vec![
                ("z", Lit("1")),
                ("doc_id", Col("doc_id", r#""d3""#)),
                ("@timestamp", Col("ts", r#""2023-01-02T12:30:00.500Z""#)),
                // A one-element array, where `d2` has the scalar.
                ("tags", Arr(vec![Elem("tags", 0, r#""solo""#)])),
                // A date in another format: only the residual holds it.
                (
                    "event",
                    Obj(vec![("created", Col("event.created", r#""01/02/2023""#))]),
                ),
                (
                    "a",
                    Obj(vec![("b.c", Lit("true")), ("ok", Col("ok", "true"))]),
                ),
            ])),
            ..Row::new("d3", JAN_1 + DAY + 12 * HOUR + 30 * 60_000 + 500)
        },
        Row {
            host_name: Some("web-2"),
            bytes: Some(-7),
            ratio: Some(1e-7),
            message: Some("café – ünïcode"),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d4""#)),
                ("@timestamp", Col("ts", r#""2023-01-03T00:00:00.000Z""#)),
                // An escape the column's rendering does not use: residual.
                ("message", Col("message", r#""café – ünïcode""#)),
                ("bytes", Col("bytes", "-7")),
                ("ratio", Col("ratio", "1e-7")),
                ("host", Obj(vec![("name", Col("host.name", r#""web-2""#))])),
            ])),
            ..Row::new("d4", JAN_1 + 2 * DAY)
        },
        Row {
            bytes: Some(1),
            ..Row::new("d5", JAN_1 + 3 * DAY)
        },
        Row {
            tags: Some(vec![]),
            user_names: Some(vec![]),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d6""#)),
                ("@timestamp", Col("ts", r#""2023-01-04T00:00:00.000Z""#)),
                ("ok", Lit("null")),
                ("tags", Lit("[]")),
                ("users", Arr(vec![])),
            ])),
            ..Row::new("d6", JAN_1 + 4 * DAY)
        },
        Row {
            bytes: Some(9_007_199_254_740_993),
            ratio: Some(3.5),
            host_name: Some("web-1"),
            path: Some("/x"),
            doc: Some(Obj(vec![
                ("doc_id", Col("doc_id", r#""d7""#)),
                ("@timestamp", Col("ts", r#""2023-01-05T00:00:00.000Z""#)),
                ("bytes", Col("bytes", "9007199254740993")),
                ("ratio", Col("ratio", "3.5")),
                ("host.name", Col("host.name", r#""web-1""#)),
                (
                    "log",
                    Obj(vec![("file.path", Col("log.file.path", r#""/x""#))]),
                ),
            ])),
            ..Row::new("d7", JAN_1 + 5 * DAY)
        },
    ]
}

/// The typed columns, as the table is created with them. `source` and its
/// two stored columns are added by the document declaration.
fn schema() -> Arc<Schema> {
    let list = |item: DataType| DataType::List(Arc::new(Field::new("item", item, true)));
    let ms = DataType::Timestamp(TimeUnit::Millisecond, None);
    Arc::new(Schema::new(vec![
        Field::new("doc_id", DataType::LargeUtf8, false),
        Field::new("ts", ms.clone(), false),
        Field::new("host.name", DataType::Utf8, true),
        Field::new("log.file.path", DataType::Utf8, true),
        Field::new("bytes", DataType::Int64, true),
        Field::new("ratio", DataType::Float64, true),
        Field::new("ok", DataType::Boolean, true),
        Field::new("tags", list(DataType::Utf8), true),
        Field::new("users.name", list(DataType::Utf8), true),
        Field::new("users.age", list(DataType::Int64), true),
        Field::new("event.created", ms, true),
        Field::new("message", DataType::LargeUtf8, true),
    ]))
}

/// The typed columns of `rows`, as one batch of [`schema`].
fn typed_batch(rows: &[Row]) -> RecordBatch {
    let mut tags = ListBuilder::new(StringBuilder::new());
    let mut names = ListBuilder::new(StringBuilder::new());
    let mut ages = ListBuilder::new(Int64Builder::new());
    for row in rows {
        match &row.tags {
            Some(values) => {
                values.iter().for_each(|v| tags.values().append_value(v));
                tags.append(true);
            }
            None => tags.append(false),
        }
        match &row.user_names {
            Some(values) => {
                values.iter().for_each(|v| names.values().append_option(*v));
                names.append(true);
            }
            None => names.append(false),
        }
        match &row.user_ages {
            Some(values) => {
                values.iter().for_each(|v| ages.values().append_option(*v));
                ages.append(true);
            }
            None => ages.append(false),
        }
    }
    let columns: Vec<ArrayRef> = vec![
        Arc::new(LargeStringArray::from_iter_values(
            rows.iter().map(|r| r.doc_id),
        )),
        Arc::new(TimestampMillisecondArray::from_iter_values(
            rows.iter().map(|r| r.ts),
        )),
        Arc::new(StringArray::from_iter(rows.iter().map(|r| r.host_name))),
        Arc::new(StringArray::from_iter(rows.iter().map(|r| r.path))),
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.bytes))),
        Arc::new(Float64Array::from_iter(rows.iter().map(|r| r.ratio))),
        Arc::new(BooleanArray::from_iter(rows.iter().map(|r| r.ok))),
        Arc::new(tags.finish()),
        Arc::new(names.finish()),
        Arc::new(ages.finish()),
        Arc::new(TimestampMillisecondArray::from_iter(
            rows.iter().map(|r| r.created),
        )),
        Arc::new(LargeStringArray::from_iter(rows.iter().map(|r| r.message))),
    ];
    RecordBatch::try_new(schema(), columns).expect("typed batch")
}

/// What the test's writer makes of one document: the original compact
/// text, the skeleton and the residual entries.
#[derive(Default)]
struct Written {
    original: String,
    skeleton: String,
    residual: Vec<String>,
}

/// Write `j`, row `row` of `batch`, into `out`: the writer's rule, slot
/// where the column renders the token exactly, residual otherwise.
fn write(j: &J, batch: &RecordBatch, row: usize, ids: &TableSchema, out: &mut Written) {
    let value = |column: &str, element: Option<usize>, token: &str, out: &mut Written| {
        out.original.push_str(token);
        let array = batch.column(batch.schema().index_of(column).expect("column"));
        let id = ids.id_of(column).expect("column id");
        let slot = match render_value(array.as_ref(), row, element) {
            Some(rendered) if rendered == token => match element {
                Some(index) => Slot::Element(id, index),
                None => Slot::Value(id),
            },
            _ => {
                out.residual.push(token.to_owned());
                Slot::Residual(out.residual.len() - 1)
            }
        };
        out.skeleton.push_str(&slot.to_string());
    };
    match j {
        J::Obj(members) => {
            out.original.push('{');
            out.skeleton.push('{');
            for (i, (key, member)) in members.iter().enumerate() {
                let sep = if i > 0 { "," } else { "" };
                let key = format!("{sep}{}:", serde_json::to_string(key).expect("key"));
                out.original.push_str(&key);
                out.skeleton.push_str(&key);
                write(member, batch, row, ids, out);
            }
            out.original.push('}');
            out.skeleton.push('}');
        }
        J::Arr(items) => {
            out.original.push('[');
            out.skeleton.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.original.push(',');
                    out.skeleton.push(',');
                }
                write(item, batch, row, ids, out);
            }
            out.original.push(']');
            out.skeleton.push(']');
        }
        J::Lit(text) => {
            out.original.push_str(text);
            out.skeleton.push_str(text);
        }
        J::Col(column, token) => value(column, None, token, out),
        J::Elem(column, index, token) => value(column, Some(*index), token, out),
    }
}

/// `rows` as the batch appended: the typed columns, then each row's
/// skeleton and residual. Returns the batch and each row's original text.
fn append_batch(rows: &[Row], ids: &TableSchema) -> (RecordBatch, Vec<Option<String>>) {
    let typed = typed_batch(rows);
    let written: Vec<Option<Written>> = rows
        .iter()
        .enumerate()
        .map(|(row, r)| {
            r.doc.as_ref().map(|doc| {
                let mut out = Written::default();
                write(doc, &typed, row, ids, &mut out);
                out
            })
        })
        .collect();
    let skeletons = StringArray::from_iter(
        written
            .iter()
            .map(|w| w.as_ref().map(|w| w.skeleton.clone())),
    );
    let residuals = LargeStringArray::from_iter(written.iter().map(|w| {
        w.as_ref()
            .filter(|w| !w.residual.is_empty())
            .map(|w| residual(w.residual.iter().map(String::as_str)))
    }));
    let mut fields: Vec<Field> = typed
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(skeleton_column(SOURCE), DataType::Utf8, true));
    fields.push(Field::new(
        residual_column(SOURCE),
        DataType::LargeUtf8,
        true,
    ));
    let mut columns = typed.columns().to_vec();
    columns.push(Arc::new(skeletons));
    columns.push(Arc::new(residuals));
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("batch");
    let originals = written.into_iter().map(|w| w.map(|w| w.original)).collect();
    (batch, originals)
}

/// The table, written in two commits; with each row's id and original
/// document.
fn create(db: &Connection) -> Vec<(String, Option<String>)> {
    let spec = IndexSpec::new().fts("doc_id").range("ts").document(SOURCE);
    db.create_table(TABLE, schema(), spec).expect("create");
    let ids = db.schema(TABLE).expect("schema document");
    let rows = rows();
    let table = db.open_table(TABLE).expect("open");
    let mut expected = Vec::new();
    for part in [&rows[..FIRST_COMMIT_ROWS], &rows[FIRST_COMMIT_ROWS..]] {
        let (batch, originals) = append_batch(part, &ids);
        table.append(&batch).expect("append");
        expected.extend(part.iter().map(|r| r.doc_id.to_owned()).zip(originals));
    }
    expected
}

/// `column` of `batches` as optional strings, in row order.
fn strings(batches: &[RecordBatch], column: &str) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            let array = batch
                .column(batch.schema().index_of(column).expect("column in result"))
                .clone();
            let array = cast(&array, &DataType::LargeUtf8).expect("as text");
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("large strings")
                .clone();
            (0..array.len())
                .map(|i| (!array.is_null(i)).then(|| array.value(i).to_owned()))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// `(doc_id, source)` of each row of `batches`.
fn pairs(batches: &[RecordBatch]) -> Vec<(String, Option<String>)> {
    strings(batches, "doc_id")
        .into_iter()
        .map(|id| id.expect("doc_id"))
        .zip(strings(batches, SOURCE))
        .collect()
}

/// The expected rows whose id is in `ids`, in that order.
fn pick(expected: &[(String, Option<String>)], ids: &[&str]) -> Vec<(String, Option<String>)> {
    ids.iter()
        .map(|id| {
            expected
                .iter()
                .find(|(d, _)| d == id)
                .expect("known id")
                .clone()
        })
        .collect()
}

/// How a statement's plan rebuilds the document column.
#[derive(Debug, PartialEq, Eq)]
enum Rebuild {
    /// After the limit, for the rows it keeps (`LateDocumentExec`), over a
    /// scan that reads none of the document's stored columns.
    Late,
    /// Over the scan, for every row it reads (`DocumentExec`).
    OverScan,
}

/// How `db` plans `query`'s rebuild of `source`.
fn rebuild_of(db: &Connection, query: &str) -> Rebuild {
    let batches = db
        .query_sql(&format!("EXPLAIN {query}"))
        .unwrap_or_else(|e| panic!("EXPLAIN {query}: {e}"));
    let plan = pretty_format_batches(&batches).expect("format").to_string();
    let late = plan.contains("LateDocumentExec: rebuild=[source]");
    let over_scan = plan.contains(" DocumentExec: rebuild=[source]");
    match (late, over_scan) {
        (true, false) => {
            assert!(
                !plan.contains(&skeleton_column(SOURCE)),
                "{query}: the late plan's scan reads no skeleton:\n{plan}"
            );
            Rebuild::Late
        }
        (false, true) => Rebuild::OverScan,
        _ => panic!("{query}: one rebuild of source:\n{plan}"),
    }
}

/// `query`'s `(doc_id, source)` rows, which `db` plans with a `rebuild`.
fn rows_of(
    db: &Connection,
    query: &str,
    rebuild: Rebuild,
    label: &str,
) -> Vec<(String, Option<String>)> {
    assert_eq!(rebuild_of(db, query), rebuild, "{label}: {query}");
    pairs(
        &db.query_sql(query)
            .unwrap_or_else(|e| panic!("{label}: {query}: {e}")),
    )
}

/// The statements that read `source` after their limit, and the ones that
/// must not, each giving exactly the documents of `expected`.
fn assert_late(db: &Connection, expected: &[(String, Option<String>)], label: &str) {
    let mut by_ts: Vec<(i64, &str)> = rows()
        .iter()
        .filter(|r| expected.iter().any(|(id, _)| id == r.doc_id))
        .map(|r| (r.ts, r.doc_id))
        .collect();
    by_ts.sort();
    let oldest: Vec<&str> = by_ts.iter().take(4).map(|(_, id)| *id).collect();
    let newest: Vec<&str> = by_ts.iter().rev().take(PAGE).map(|(_, id)| *id).collect();

    // A filtered `LIMIT`: any rows the filter keeps, each with its document.
    let got = rows_of(
        db,
        &format!("SELECT doc_id, source FROM {TABLE} WHERE bytes >= 0 LIMIT {PAGE}"),
        Rebuild::Late,
        label,
    );
    assert_eq!(got.len(), PAGE, "{label}: filtered LIMIT");
    for (id, doc) in &got {
        assert_eq!(
            Some(doc),
            expected.iter().find(|(e, _)| e == id).map(|(_, d)| d),
            "{label}: filtered LIMIT, {id}"
        );
    }

    // `ORDER BY` another column with a `LIMIT`: a top-k.
    let got = rows_of(
        db,
        &format!("SELECT doc_id, source FROM {TABLE} ORDER BY ts DESC LIMIT {PAGE}"),
        Rebuild::Late,
        label,
    );
    assert_eq!(got, pick(expected, &newest), "{label}: top-k");

    // The document renamed and repeated by the projection, with an offset.
    let got = rows_of(
        db,
        &format!(
            "SELECT doc_id, source AS doc, source FROM {TABLE} \
             ORDER BY ts LIMIT 2 OFFSET 1"
        ),
        Rebuild::Late,
        label,
    );
    assert_eq!(
        got,
        pick(expected, &oldest[1..3]),
        "{label}: renamed, offset"
    );
    let batches = db
        .query_sql(&format!(
            "SELECT doc_id, source AS doc, source FROM {TABLE} ORDER BY ts LIMIT 2 OFFSET 1"
        ))
        .expect("renamed");
    assert_eq!(
        strings(&batches, "doc"),
        strings(&batches, SOURCE),
        "{label}: a document projected twice"
    );

    // A subquery with the limit, sorted again outside it.
    let mut inner: Vec<&str> = oldest.clone();
    inner.sort();
    let got = rows_of(
        db,
        &format!(
            "SELECT doc_id, source FROM \
             (SELECT doc_id, source FROM {TABLE} ORDER BY ts LIMIT 4) s ORDER BY doc_id"
        ),
        Rebuild::Late,
        label,
    );
    assert_eq!(
        got,
        pick(expected, &inner),
        "{label}: subquery with a limit"
    );

    // A filter on the document above the subquery's limit: the documents
    // are read after the limit, below the filter, which sees them rebuilt.
    let solo: Vec<&str> = inner
        .iter()
        .copied()
        .filter(|id| {
            expected
                .iter()
                .any(|(e, d)| e == id && d.as_ref().is_some_and(|d| d.contains("\"solo\"")))
        })
        .collect();
    let got = rows_of(
        db,
        &format!(
            "SELECT doc_id, source FROM \
             (SELECT doc_id, source FROM {TABLE} ORDER BY ts LIMIT 4) s \
             WHERE source LIKE '%\"solo\"%' ORDER BY doc_id"
        ),
        Rebuild::Late,
        label,
    );
    assert_eq!(
        got,
        pick(expected, &solo),
        "{label}: filter above the limit"
    );

    // A filter or a sort on the document below the limit reads it rebuilt.
    let all_solo: Vec<&str> = expected
        .iter()
        .filter(|(_, d)| d.as_ref().is_some_and(|d| d.contains("\"solo\"")))
        .map(|(id, _)| id.as_str())
        .collect();
    let got = rows_of(
        db,
        &format!(
            "SELECT doc_id, source FROM {TABLE} WHERE source LIKE '%\"solo\"%' \
             ORDER BY doc_id LIMIT 5"
        ),
        Rebuild::OverScan,
        label,
    );
    assert_eq!(got, pick(expected, &all_solo), "{label}: filter on source");
    let mut by_source: Vec<&(String, Option<String>)> = expected.iter().collect();
    by_source.sort_by(|a, b| (a.1.is_none(), &a.1).cmp(&(b.1.is_none(), &b.1)));
    let got = rows_of(
        db,
        &format!("SELECT doc_id, source FROM {TABLE} ORDER BY source LIMIT 2"),
        Rebuild::OverScan,
        label,
    );
    let first: Vec<(String, Option<String>)> = by_source.into_iter().take(2).cloned().collect();
    assert_eq!(got, first, "{label}: sort on source");

    // No limit: every scanned row is returned, rebuilt over the scan.
    let got = rows_of(
        db,
        &format!("SELECT doc_id, source FROM {TABLE} ORDER BY doc_id"),
        Rebuild::OverScan,
        label,
    );
    assert_eq!(got, expected, "{label}: no limit");
}

/// Every read path gives back each original document exactly.
fn assert_rebuilt(db: &Connection, expected: &[(String, Option<String>)], label: &str) {
    let sql = |q: &str| {
        db.query_sql(q)
            .unwrap_or_else(|e| panic!("{label}: {q}: {e}"))
    };
    // `SELECT *` carries the document column with the rest.
    let all = sql(&format!("SELECT * FROM {TABLE} ORDER BY doc_id"));
    assert_eq!(pairs(&all), expected, "{label}: SELECT *");

    // A filtered page, newest first.
    let mut page: Vec<(i64, &str)> = rows()
        .iter()
        .filter(|r| r.bytes.is_some_and(|b| b >= 0))
        .map(|r| (r.ts, r.doc_id))
        .collect();
    page.sort_by(|a, b| b.cmp(a));
    let page_ids: Vec<&str> = page.iter().take(PAGE).map(|(_, id)| *id).collect();
    let got = sql(&format!(
        "SELECT doc_id, source FROM {TABLE} WHERE bytes >= 0 ORDER BY ts DESC LIMIT {PAGE}"
    ));
    assert_eq!(pairs(&got), pick(expected, &page_ids), "{label}: page");

    // A filter and a sort on the document itself run over the rebuilt text.
    let got = sql(&format!(
        "SELECT doc_id, source FROM {TABLE} WHERE source LIKE '%\"solo\"%' ORDER BY source"
    ));
    assert_eq!(
        pairs(&got),
        pick(expected, &["d2", "d3"]),
        "{label}: filter on source"
    );

    // Through a subquery, the document column projected from the inner scan.
    let got = sql(&format!(
        "SELECT doc_id, source FROM (SELECT * FROM {TABLE} WHERE ok) s ORDER BY doc_id"
    ));
    assert_eq!(
        pairs(&got),
        pick(expected, &["d0", "d3"]),
        "{label}: subquery"
    );

    // The search methods resolve the document for their hits.
    let table = db.open_table(TABLE).expect("open");
    let projection: &[&str] = &["doc_id", SOURCE];
    for id in ["d1", "d3", "d5"] {
        let hits = table
            .exact_match("doc_id", id, Some(projection))
            .expect("exact_match");
        assert_eq!(
            pairs(&hits),
            pick(expected, &[id]),
            "{label}: exact_match {id}"
        );
    }
    let hits = table
        .token_match("doc_id", "d2", BoolMode::Or, Some(projection))
        .expect("token_match");
    assert_eq!(
        pairs(&hits),
        pick(expected, &["d2"]),
        "{label}: token_match"
    );
    let newest = table
        .sorted_search("ts", true, PAGE, Some(projection))
        .expect("sorted_search");
    assert_eq!(
        pairs(&newest),
        pick(expected, &["d7", "d6", "d5"]),
        "{label}: sorted_search"
    );
    // The search table functions resolve it for their hits too.
    let got = sql(&format!(
        "SELECT doc_id, source FROM bm25_search('{TABLE}', 'doc_id', 'd4', 10)"
    ));
    assert_eq!(pairs(&got), pick(expected, &["d4"]), "{label}: bm25_search");
    let got = sql(&format!(
        "SELECT doc_id, source FROM token_match('{TABLE}', 'doc_id', 'd6', 'or')"
    ));
    assert_eq!(
        pairs(&got),
        pick(expected, &["d6"]),
        "{label}: token_match()"
    );
    // The document alone, with no other column beside it.
    let alone = table
        .exact_match("doc_id", "d0", Some(&[SOURCE]))
        .expect("exact_match source only");
    assert_eq!(
        strings(&alone, SOURCE),
        vec![pick(expected, &["d0"])[0].1.clone()],
        "{label}: source alone"
    );

    assert_late(db, expected, label);
}

#[test]
fn a_document_column_rebuilds_every_document_exactly_through_compaction_rename_and_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf-8 path");
    let db = connect(path).expect("connect");
    let expected = create(&db);
    // The fixture exercises both sides of the writer's rule.
    let ids = db.schema(TABLE).expect("schema");
    assert!(
        expected.iter().any(|(_, doc)| doc.is_none()),
        "a row without a document"
    );
    assert!(
        ids.id_of(SOURCE).is_some(),
        "the document column is in the schema"
    );
    assert_rebuilt(&db, &expected, "after load");

    db.open_table(TABLE)
        .expect("open")
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    assert_rebuilt(&db, &expected, "after compaction");

    // A column the documents draw on is renamed: slots name it by id.
    let renamed = FieldPatch::named("size").with_id(ids.id_of("bytes").expect("bytes id"));
    db.apply_schema(TABLE, &SchemaPatch::new(vec![renamed]), None)
        .expect("rename");
    let all = db
        .query_sql(&format!("SELECT * FROM {TABLE} ORDER BY doc_id"))
        .expect("select after rename");
    assert_eq!(pairs(&all), expected, "after a rename");
    let mut sized: Vec<(i64, &str)> = rows()
        .iter()
        .filter(|r| r.bytes.is_some_and(|b| b >= 0))
        .map(|r| (r.ts, r.doc_id))
        .collect();
    sized.sort_by(|a, b| b.cmp(a));
    let sized: Vec<&str> = sized.iter().take(PAGE).map(|(_, id)| *id).collect();
    let got = rows_of(
        &db,
        &format!(
            "SELECT doc_id, source FROM {TABLE} WHERE size >= 0 ORDER BY ts DESC LIMIT {PAGE}"
        ),
        Rebuild::Late,
        "after a rename",
    );
    assert_eq!(got, pick(&expected, &sized), "after a rename: late page");
    let back = FieldPatch::named("bytes").with_id(ids.id_of("bytes").expect("bytes id"));
    db.apply_schema(TABLE, &SchemaPatch::new(vec![back]), None)
        .expect("rename back");

    drop(db);
    let reopened = connect(path).expect("reconnect");
    assert_rebuilt(&reopened, &expected, "after a reopen");
}

/// `d1` as an update writes it: other values, the same id.
fn updated_d1() -> Row {
    use J::*;
    Row {
        path: Some("/var/log/b2.log"),
        bytes: Some(7),
        doc: Some(Obj(vec![
            ("doc_id", Col("doc_id", r#""d1""#)),
            ("@timestamp", Col("ts", r#""2023-01-01T10:00:00.000Z""#)),
            ("bytes", Col("bytes", "7")),
            ("path", Col("log.file.path", r#""/var/log/b2.log""#)),
            ("updated", Lit("true")),
        ])),
        ..Row::new("d1", JAN_1 + 10 * HOUR)
    }
}

#[test]
fn a_late_read_finds_the_live_copy_of_an_updated_row_and_skips_a_deleted_one() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let mut expected = create(&db);
    let ids = db.schema(TABLE).expect("schema");
    let table = db.open_table(TABLE).expect("open");

    // `d1` is rewritten into a superfile of its own, its old copy deleted
    // where it was; `d4` is deleted.
    let (batch, originals) = append_batch(&[updated_d1()], &ids);
    table
        .update(col("doc_id").eq(lit("d1")), &batch)
        .expect("update d1");
    table
        .delete(col("doc_id").eq(lit("d4")))
        .expect("delete d4");
    let d1 = expected.iter_mut().find(|(id, _)| id == "d1").expect("d1");
    d1.1 = originals[0].clone();
    assert!(
        d1.1.as_ref().is_some_and(|d| d.contains("updated")),
        "the update changes the document"
    );
    expected.retain(|(id, _)| id != "d4");

    assert_late(&db, &expected, "after an update and a delete");
    // Compaction drops the deleted rows, so the merged superfile's ids have
    // gaps and each row is found by its id in the superfile's `_id` column.
    table
        .optimize(&OptimizeOptions::default())
        .expect("optimize");
    assert_late(&db, &expected, "after an update, a delete and compaction");
}

#[test]
fn an_append_writes_the_stored_columns_and_never_the_document() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let expected = create(&db);
    let table = db.open_table(TABLE).expect("open");
    let ids = db.schema(TABLE).expect("schema");
    let (batch, _) = append_batch(&rows()[..1], &ids);

    // The document itself is refused.
    let mut fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.push(Field::new(SOURCE, DataType::LargeUtf8, true));
    let mut columns = batch.columns().to_vec();
    columns.push(Arc::new(LargeStringArray::from(vec![Some("{}")])));
    let with_source = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("batch");
    let err = table.append(&with_source).expect_err("source refused");
    assert!(err.to_string().contains("rebuilt on read"), "{err}");

    // So are residuals without the skeletons that name their entries.
    let skeleton_at = batch
        .schema()
        .index_of(&skeleton_column(SOURCE))
        .expect("skeleton");
    let without_skeleton = batch
        .project(
            &(0..batch.num_columns())
                .filter(|&i| i != skeleton_at)
                .collect::<Vec<_>>(),
        )
        .expect("project");
    let err = table
        .append(&without_skeleton)
        .expect_err("residual alone refused");
    assert!(
        err.to_string().contains("whose slots name its entries"),
        "{err}"
    );

    // Neither is a row with no document.
    let typed = typed_batch(&rows()[..1]);
    table.append(&typed).expect("no document");
    let got = db
        .query_sql(&format!("SELECT source FROM {TABLE} WHERE doc_id = 'd0'"))
        .expect("select");
    let mut sources = strings(&got, SOURCE);
    sources.sort();
    assert_eq!(
        sources,
        vec![None, pick(&expected, &["d0"])[0].1.clone()],
        "the row appended without a document, and the one appended with it"
    );
}

#[test]
fn a_schema_write_keeps_the_columns_a_document_is_rebuilt_from() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    create(&db);
    for patch in [
        FieldPatch::named(skeleton_column(SOURCE)).dropped(),
        FieldPatch::named(residual_column(SOURCE)).with_type(DataType::Utf8),
        FieldPatch::named(SOURCE).with_type(DataType::Utf8),
    ] {
        let err = db
            .apply_schema(TABLE, &SchemaPatch::new(vec![patch]), None)
            .expect_err("refused");
        assert!(
            err.to_string().contains("document column `source`"),
            "{err}"
        );
    }
    // Dropping the document drops nothing it was rebuilt from, and the two
    // stored columns can go with it.
    db.apply_schema(
        TABLE,
        &SchemaPatch::new(vec![
            FieldPatch::named(SOURCE).dropped(),
            FieldPatch::named(skeleton_column(SOURCE)).dropped(),
            FieldPatch::named(residual_column(SOURCE)).dropped(),
        ]),
        None,
    )
    .expect("drop the document with its columns");
    let ids = db.schema(TABLE).expect("schema");
    assert!(ids.id_of(SOURCE).is_none());
    assert!(ids.id_of("bytes").is_some());
}
