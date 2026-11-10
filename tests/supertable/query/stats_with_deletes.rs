// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Aggregates answered from statistics on tables with deleted rows.
//!
//! The oracle is the same statement made unanswerable from statistics: an
//! extra `random() >= 0` conjunct holds for every row but is no range, so
//! the statement scans every superfile and drops the deleted rows there.
//! Every aggregate the statistics may answer has to match it exactly, on
//! random tables whose superfiles hold random deletes: clustered, scattered,
//! every row, none; columns null in every row, in none, in some; empty
//! strings, booleans, integers of two widths, strings of both offset widths,
//! floats, ties, and columns past the per-superfile cap of counted values.

#![deny(clippy::unwrap_used)]

use std::{
    cmp::Ordering,
    collections::BTreeMap,
    ops::{Range, RangeInclusive},
    slice::from_ref,
    sync::Arc,
};

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    common::ScalarValue,
    prelude::{col, lit},
};
use infino::{Connection, IndexSpec, Supertable, connect};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

/// Seeds of the random tables.
const SEEDS: u64 = 12;
/// Superfiles in a random table, at least and at most.
const MIN_FILES: usize = 3;
const MAX_FILES: usize = 7;
/// Rows of a superfile, at most; a "wide" one holds more distinct values of
/// `j` than a superfile counts exactly.
const MAX_ROWS: usize = 40;
const WIDE_ROWS: usize = 300;
/// Relative tolerance between two floating averages: statistics and the
/// scan add the same values in different orders.
const FLOAT_TOLERANCE: f64 = 1e-9;
/// The table's name in SQL.
const TABLE: &str = "t";
/// How often a superfile's column is null: never, sometimes, always.
const NULL_SHARES: [f64; 3] = [0.0, 0.3, 1.0];
/// Chance a superfile holds more distinct values of `j` than it counts.
const WIDE_CHANCE: f64 = 0.2;
/// Chance a superfile holds one value of `g` throughout.
const CONSTANT_CHANCE: f64 = 0.2;
/// The values of `g` and `u`; of `i`; of a narrow and a wide `j`; and of
/// `f`'s numerator over [`QUARTER`].
const G_VALUES: [&str; 5] = ["", "a", "b", "c", "dd"];
const U_VALUES: [&str; 4] = ["x", "y", "", "zz"];
const I_VALUES: RangeInclusive<i64> = -3..=3;
const NARROW_J: Range<i32> = 0..5;
const WIDE_J: Range<i32> = -1000..1000;
const F_QUARTERS: RangeInclusive<i32> = -40..=40;
/// `f`'s values are quarters: they add up exactly whatever the order, so a
/// sum from statistics and one from the scan agree to the bit.
const QUARTER: f64 = 4.0;
/// Of [`DELETE_MODES`] equally likely draws, how many delete no row of a
/// superfile, a few rows apart, and a run; the rest delete every row.
const DELETE_MODES: u32 = 10;
const DELETE_NONE: u32 = 4;
const DELETE_SCATTERED: u32 = 2;
const DELETE_RUN: u32 = 3;
/// Chance a row goes in a scattered delete.
const SCATTERED_CHANCE: f64 = 0.2;
/// The columns aggregated: name, and whether SUM/AVG apply.
const COLUMNS: [(&str, bool); 6] = [
    ("g", false),
    ("u", false),
    ("i", true),
    ("j", true),
    ("b", false),
    ("f", true),
];

/// One row of a random table.
#[derive(Clone, Debug)]
struct Row {
    k: i64,
    g: Option<String>,
    u: Option<String>,
    i: Option<i64>,
    j: Option<i32>,
    b: Option<bool>,
    f: Option<f64>,
}

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("g", DataType::LargeUtf8, true),
        Field::new("u", DataType::Utf8, true),
        Field::new("i", DataType::Int64, true),
        Field::new("j", DataType::Int32, true),
        Field::new("b", DataType::Boolean, true),
        Field::new("f", DataType::Float64, true),
    ]))
}

fn batch(rows: &[Row]) -> RecordBatch {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.k))),
        Arc::new(LargeStringArray::from_iter(
            rows.iter().map(|r| r.g.clone()),
        )),
        Arc::new(StringArray::from_iter(rows.iter().map(|r| r.u.clone()))),
        Arc::new(Int64Array::from_iter(rows.iter().map(|r| r.i))),
        Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.j))),
        Arc::new(BooleanArray::from_iter(rows.iter().map(|r| r.b))),
        Arc::new(Float64Array::from_iter(rows.iter().map(|r| r.f))),
    ];
    RecordBatch::try_new(schema(), columns).expect("batch")
}

/// How often a superfile's column is null: never, sometimes, or always.
fn null_share(rng: &mut StdRng) -> f64 {
    NULL_SHARES[rng.random_range(0..NULL_SHARES.len())]
}

/// One of `values`, drawn by `rng`.
fn pick(rng: &mut StdRng, values: &[&'static str]) -> &'static str {
    values[rng.random_range(0..values.len())]
}

/// A value of `share` chance of being null, else `value()`.
fn maybe<T>(rng: &mut StdRng, share: f64, value: impl FnOnce(&mut StdRng) -> T) -> Option<T> {
    (!rng.random_bool(share)).then(|| value(rng))
}

/// One superfile's rows, keys from `first_key`.
fn random_file(rng: &mut StdRng, first_key: i64) -> Vec<Row> {
    let wide = rng.random_bool(WIDE_CHANCE);
    let n = match wide {
        true => WIDE_ROWS,
        false => rng.random_range(1..=MAX_ROWS),
    };
    let shares: Vec<f64> = COLUMNS.iter().map(|_| null_share(rng)).collect();
    // A superfile holding one value of `g` throughout, beside the others.
    let constant = rng.random_bool(CONSTANT_CHANCE);
    (0..n)
        .map(|r| Row {
            k: first_key + r as i64,
            g: maybe(rng, shares[0], |rng| match constant {
                true => G_VALUES[1].to_string(),
                false => pick(rng, &G_VALUES).to_string(),
            }),
            u: maybe(rng, shares[1], |rng| pick(rng, &U_VALUES).to_string()),
            i: maybe(rng, shares[2], |rng| rng.random_range(I_VALUES)),
            j: maybe(rng, shares[3], |rng| match wide {
                true => rng.random_range(WIDE_J),
                false => rng.random_range(NARROW_J),
            }),
            b: maybe(rng, shares[4], |rng| rng.random()),
            f: maybe(rng, shares[5], |rng| {
                f64::from(rng.random_range(F_QUARTERS)) / QUARTER
            }),
        })
        .collect()
}

/// The keys of `rows` one round of deletes removes: none, a few apart, a
/// run, or every one.
fn random_deletes(rng: &mut StdRng, rows: &[Row]) -> Vec<i64> {
    let keys: Vec<i64> = rows.iter().map(|r| r.k).collect();
    let mode = rng.random_range(0..DELETE_MODES);
    if mode < DELETE_NONE {
        Vec::new()
    } else if mode < DELETE_NONE + DELETE_SCATTERED {
        keys.into_iter()
            .filter(|_| rng.random_bool(SCATTERED_CHANCE))
            .collect()
    } else if mode < DELETE_NONE + DELETE_SCATTERED + DELETE_RUN {
        let start = rng.random_range(0..keys.len());
        let len = rng.random_range(1..=keys.len() - start);
        keys[start..start + len].to_vec()
    } else {
        keys
    }
}

/// A random table and what it holds.
struct Fixture {
    _dir: TempDir,
    conn: Connection,
    table: Supertable,
    /// Each superfile's rows, as appended.
    files: Vec<Vec<Row>>,
    rng: StdRng,
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let dir = TempDir::new().expect("tempdir");
        let conn = connect(dir.path().to_str().expect("utf8 path")).expect("connect");
        let table = conn
            .create_table(TABLE, schema(), IndexSpec::new())
            .expect("create");
        let mut rng = StdRng::seed_from_u64(seed);
        let n_files = rng.random_range(MIN_FILES..=MAX_FILES);
        let mut files = Vec::with_capacity(n_files);
        let mut next_key = 0i64;
        for _ in 0..n_files {
            let rows = random_file(&mut rng, next_key);
            next_key += rows.len() as i64;
            table.append(&batch(&rows)).expect("append");
            files.push(rows);
        }
        Self {
            _dir: dir,
            conn,
            table,
            files,
            rng,
        }
    }

    /// One round of random deletes across the superfiles.
    fn delete_some(&mut self) {
        let mut keys = Vec::new();
        for rows in &self.files {
            keys.extend(random_deletes(&mut self.rng, rows));
        }
        if keys.is_empty() {
            return;
        }
        let list = keys.into_iter().map(lit).collect();
        self.table
            .delete(col("k").in_list(list, false))
            .expect("delete");
    }

    /// Keys at which a `k` range starts or ends: superfile edges, so a
    /// range covers whole superfiles, and points inside one, so it cuts one.
    fn key_edges(&mut self) -> Vec<(i64, i64)> {
        let mut starts = vec![0i64];
        for rows in &self.files {
            starts.push(starts.last().copied().unwrap_or(0) + rows.len() as i64);
        }
        let last = starts.last().copied().unwrap_or(0);
        let mut ranges = vec![(0, last), (starts[1], last)];
        let a = self.rng.random_range(0..last);
        let b = self.rng.random_range(a..=last);
        ranges.push((a, b));
        if starts.len() > 3 {
            ranges.push((starts[1], starts[3] - 1));
        }
        ranges
    }
}

/// A result cell, compared exactly except for floats.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
enum Cell {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
}

fn cell(array: &dyn Array, row: usize) -> Cell {
    let value = ScalarValue::try_from_array(array, row).expect("value");
    if value.is_null() {
        return Cell::Null;
    }
    match value {
        ScalarValue::Boolean(Some(v)) => Cell::Bool(v),
        ScalarValue::Int8(Some(v)) => Cell::Int(v.into()),
        ScalarValue::Int16(Some(v)) => Cell::Int(v.into()),
        ScalarValue::Int32(Some(v)) => Cell::Int(v.into()),
        ScalarValue::Int64(Some(v)) => Cell::Int(v.into()),
        ScalarValue::UInt64(Some(v)) => Cell::Int(v.into()),
        ScalarValue::Float64(Some(v)) => Cell::Float(v),
        ScalarValue::Utf8(Some(v))
        | ScalarValue::LargeUtf8(Some(v))
        | ScalarValue::Utf8View(Some(v)) => Cell::Str(v),
        other => panic!("unexpected result type {other:?}"),
    }
}

/// Every row of `batches`, sorted, so grouped results compare whatever
/// order the groups come in.
fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<Cell>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|column| cell(column.as_ref(), row))
                    .collect::<Vec<_>>(),
            );
        }
    }
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    rows
}

fn same_cell(a: &Cell, b: &Cell) -> bool {
    match (a, b) {
        (Cell::Float(x), Cell::Float(y)) => {
            x == y || (x - y).abs() <= FLOAT_TOLERANCE * x.abs().max(y.abs())
        }
        _ => a == b,
    }
}

fn assert_same_rows(answered: &[Vec<Cell>], scanned: &[Vec<Cell>], what: &str) {
    let same = answered.len() == scanned.len()
        && answered
            .iter()
            .zip(scanned)
            .all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_cell(x, y)));
    assert!(
        same,
        "{what}\n  answered: {answered:?}\n  scanned:  {scanned:?}"
    );
}

/// A statement, and the same statement with a conjunct no statistics can
/// answer for: every row passes it, so it scans every superfile.
struct Statement {
    select: String,
    filter: Option<String>,
    group: Option<String>,
}

impl Statement {
    fn sql(&self, forced: bool) -> String {
        let mut sql = format!("SELECT {} FROM {TABLE}", self.select);
        match (&self.filter, forced) {
            (Some(filter), false) => sql.push_str(&format!(" WHERE {filter}")),
            (Some(filter), true) => sql.push_str(&format!(" WHERE ({filter}) AND random() >= 0")),
            (None, true) => sql.push_str(" WHERE random() >= 0"),
            (None, false) => {}
        }
        if let Some(group) = &self.group {
            sql.push_str(&format!(" GROUP BY {group}"));
        }
        sql
    }
}

fn statement(select: &str, filter: Option<&str>, group: Option<&str>) -> Statement {
    Statement {
        select: select.to_string(),
        filter: filter.map(str::to_string),
        group: group.map(str::to_string),
    }
}

/// Every statement the statistics may answer, over `fixture`'s columns.
fn statements(fixture: &mut Fixture) -> Vec<Statement> {
    let mut out = vec![statement("COUNT(*)", None, None)];
    let ranges = fixture.key_edges();
    for (name, numeric) in COLUMNS {
        let mut select = format!("COUNT(*), COUNT({name}), MIN({name}), MAX({name})");
        if numeric {
            select.push_str(&format!(", SUM({name}), AVG({name})"));
        }
        out.push(statement(&select, None, None));
        for (lo, hi) in &ranges {
            out.push(statement(
                &select,
                Some(&format!("k BETWEEN {lo} AND {hi}")),
                None,
            ));
        }
        out.push(statement(&format!("{name}, COUNT(*)"), None, Some(name)));
        out.push(statement(
            &format!("{name}, COUNT(*)"),
            Some(&format!("{name} IS NOT NULL")),
            Some(name),
        ));
    }
    for filter in [
        "g = 'a'",
        "g = ''",
        "g >= 'b'",
        "u = ''",
        "i = 2",
        "i BETWEEN -1 AND 1",
        "j >= 3",
        "b = true",
    ] {
        out.push(statement("COUNT(*)", Some(filter), None));
    }
    out
}

/// The text of `sql`'s `EXPLAIN` through `conn`.
fn explain(conn: &Connection, sql: &str) -> String {
    rows_of(&conn.query_sql(&format!("EXPLAIN {sql}")).expect("explain"))
        .into_iter()
        .flatten()
        .filter_map(|cell| match cell {
            Cell::Str(s) => Some(s),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether a plan answers some superfile from statistics: it reads no
/// table, or its scan says it stands for only some superfiles.
fn uses_statistics(explained: &str) -> bool {
    !explained.contains("TableScan") || explained.contains("answered from statistics")
}

/// Every statement of [`statements`] matches its scan on `fixture`, through
/// the connection and through a reader of the table; and each column's
/// value counts match its grouped scan. Returns how many statements were
/// answered at least in part from statistics.
fn assert_answers_match_scans(fixture: &mut Fixture, when: &str) -> usize {
    let reader = fixture.table.local_handle().reader().expect("reader");
    let mut from_statistics = 0;
    for statement in statements(fixture) {
        let (answered_sql, scanned_sql) = (statement.sql(false), statement.sql(true));
        let scanned = rows_of(&fixture.conn.query_sql(&scanned_sql).expect("scan"));
        let answered = rows_of(&fixture.conn.query_sql(&answered_sql).expect("answer"));
        assert_same_rows(&answered, &scanned, &format!("{when}: {answered_sql}"));
        assert!(
            !uses_statistics(&explain(&fixture.conn, &scanned_sql)),
            "{when}: the oracle must scan: {scanned_sql}"
        );
        if uses_statistics(&explain(&fixture.conn, &answered_sql)) {
            from_statistics += 1;
        }
        let through_reader = answered_sql.replace(&format!("FROM {TABLE}"), "FROM supertable");
        let read = rows_of(&reader.query_sql(&through_reader).expect("reader answer"));
        assert_same_rows(&read, &scanned, &format!("{when}, reader: {answered_sql}"));
    }
    for (name, _) in COLUMNS {
        let counts = fixture.table.value_counts(name).expect("value counts");
        let grouped = statement(&format!("{name}, COUNT(*)"), None, Some(name)).sql(true);
        assert_same_rows(
            &rows_of(from_ref(&counts)),
            &rows_of(&fixture.conn.query_sql(&grouped).expect("grouped scan")),
            &format!("{when}: value_counts({name})"),
        );
    }
    from_statistics
}

/// On random tables, before any delete and after each of two rounds of
/// random deletes, every aggregate the statistics may answer equals the
/// scan of the same statement.
#[test]
fn aggregates_from_statistics_match_the_scan_under_random_deletes() {
    // Statements answered at least in part from statistics, before and
    // after deletes, across every seed.
    let (mut clean, mut deleted) = (0, 0);
    for seed in 0..SEEDS {
        let mut fixture = Fixture::new(seed);
        clean += assert_answers_match_scans(&mut fixture, &format!("seed {seed}, no deletes"));
        fixture.delete_some();
        deleted += assert_answers_match_scans(&mut fixture, &format!("seed {seed}, first deletes"));
        fixture.delete_some();
        deleted +=
            assert_answers_match_scans(&mut fixture, &format!("seed {seed}, second deletes"));
    }
    eprintln!("from statistics: {clean} before deletes, {deleted} after");
    assert!(
        clean > 0 && deleted > 0,
        "the statistics answer some statements"
    );
}

/// `(group, count)` rows of a grouped count over `g`, the null group as
/// `None`.
fn group_counts(batches: &[RecordBatch]) -> BTreeMap<Option<String>, i64> {
    rows_of(batches)
        .into_iter()
        .map(|row| {
            let group = match &row[0] {
                Cell::Str(s) => Some(s.clone()),
                Cell::Null => None,
                other => panic!("unexpected group {other:?}"),
            };
            let Cell::Int(count) = row[1] else {
                panic!("unexpected count {:?}", row[1]);
            };
            (group, i64::try_from(count).expect("count fits"))
        })
        .collect()
}

/// Two superfiles, one with a deleted row: `g`'s grouped count, the null
/// group included, answers the clean superfile from its value counts and
/// scans only the other, and `EXPLAIN` says so.
#[test]
fn grouped_count_scans_only_the_superfile_with_deletes() {
    let dir = TempDir::new().expect("tempdir");
    let conn = connect(dir.path().to_str().expect("utf8 path")).expect("connect");
    let table = conn
        .create_table(TABLE, schema(), IndexSpec::new())
        .expect("create");
    let row = |k: i64, g: Option<&str>| Row {
        k,
        g: g.map(str::to_string),
        u: None,
        i: None,
        j: None,
        b: None,
        f: None,
    };
    table
        .append(&batch(&[
            row(0, Some("a")),
            row(1, Some("b")),
            row(2, None),
        ]))
        .expect("append clean");
    table
        .append(&batch(&[
            row(3, Some("a")),
            row(4, Some("c")),
            row(5, None),
        ]))
        .expect("append the one with deletes");
    table.delete(col("k").eq(lit(4i64))).expect("delete");

    let sql = format!("SELECT g, COUNT(*) FROM {TABLE} GROUP BY g");
    let expected: BTreeMap<Option<String>, i64> = [
        (None, 2),
        (Some("a".to_string()), 2),
        (Some("b".to_string()), 1),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        group_counts(&conn.query_sql(&sql).expect("grouped")),
        expected
    );
    let explained = explain(&conn, &sql);
    assert!(
        explained.contains("scans 1 superfiles (1 with deleted rows); 1 answered from statistics"),
        "the scan stands for the one superfile with deletes:\n{explained}"
    );
    let counts = table.value_counts("g").expect("value counts");
    assert_eq!(group_counts(from_ref(&counts)), expected);

    // Columns null in every row: no sum, no average, no bound, no count.
    for name in ["i", "j", "f"] {
        let aggregates = statement(
            &format!("COUNT({name}), MIN({name}), MAX({name}), SUM({name}), AVG({name})"),
            None,
            None,
        );
        let answered = rows_of(&conn.query_sql(&aggregates.sql(false)).expect("answer"));
        assert_eq!(
            answered,
            vec![vec![
                Cell::Int(0),
                Cell::Null,
                Cell::Null,
                Cell::Null,
                Cell::Null
            ]],
            "{name}"
        );
    }
}
