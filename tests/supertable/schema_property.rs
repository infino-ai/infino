// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A random sequence of appends, schema writes and compactions against an
//! oracle that holds the rows as `field id → value`: after every step the
//! schema document, every live column's values and the full-text hits
//! agree with the oracle, and every batch is stored or refused exactly as
//! the oracle predicts. Case count follows `PROPTEST_CASES` (the release
//! gate runs it at 10,000); the seed corpus is under
//! `tests/proptest-regressions/`.

use std::{collections::BTreeMap, sync::Arc};

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    CompactionSettings, Connection, FieldId, FieldPatch, IndexSpec, InfinoError, OptimizeOptions,
    SchemaPatch, Supertable, connect,
};
use proptest::prelude::*;
use tempfile::TempDir;

const TABLE: &str = "t";
/// Column names a batch or a schema write may use, each with the type its
/// values take when it is added.
const POOL: [(&str, Kind); 6] = [
    ("c0", Kind::Int),
    ("c1", Kind::Int),
    ("c2", Kind::Str),
    ("c3", Kind::Str),
    ("c4", Kind::Float),
    ("c5", Kind::Float),
];
/// Words a title is drawn from; each is also a full-text query.
const WORDS: [&str; 4] = ["alpha", "beta", "gamma", "delta"];
const MAX_OPS: usize = 10;
const MAX_ROWS_PER_BATCH: usize = 4;
const DEFAULT_CASES: u32 = 24;
const FTS_TOP_K: usize = 1000;
const COMPACT_TARGET_MB: u64 = 1;
const COMPACT_MIN_FILL_PERCENT: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Int,
    Str,
    Float,
}

impl Kind {
    fn data_type(self) -> DataType {
        match self {
            Kind::Int => DataType::Int64,
            Kind::Str => DataType::LargeUtf8,
            Kind::Float => DataType::Float64,
        }
    }

    /// How a value of this kind reads back through [`cell`].
    fn render(self, v: i64) -> String {
        match self {
            Kind::Int => v.to_string(),
            Kind::Str => format!("s{v}"),
            Kind::Float => (v as f64).to_string(),
        }
    }

    fn array(self, values: &[Option<i64>]) -> ArrayRef {
        match self {
            Kind::Int => Arc::new(Int64Array::from(values.to_vec())),
            Kind::Str => Arc::new(LargeStringArray::from(
                values
                    .iter()
                    .map(|v| v.map(|v| format!("s{v}")))
                    .collect::<Vec<_>>(),
            )),
            Kind::Float => Arc::new(Float64Array::from(
                values
                    .iter()
                    .map(|v| v.map(|v| v as f64))
                    .collect::<Vec<_>>(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
enum Op {
    /// A batch carrying the pool columns in `present`, one value per row.
    Append {
        present: Vec<bool>,
        rows: Vec<(Option<usize>, Vec<Option<i64>>)>,
    },
    /// Add pool column `col` through the schema write.
    Add {
        col: usize,
    },
    /// Rename the live column named after pool entry `from` to pool entry
    /// `to`'s name.
    Rename {
        from: usize,
        to: usize,
    },
    /// Drop the live column named after pool entry `col`.
    Drop {
        col: usize,
    },
    Compact,
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let col = 0..POOL.len();
    let row = (
        proptest::option::of(0..WORDS.len()),
        proptest::collection::vec(proptest::option::of(0..20i64), POOL.len()),
    );
    prop_oneof![
        5 => (
            proptest::collection::vec(any::<bool>(), POOL.len()),
            proptest::collection::vec(row, 1..=MAX_ROWS_PER_BATCH),
        )
            .prop_map(|(present, rows)| Op::Append { present, rows }),
        2 => col.clone().prop_map(|col| Op::Add { col }),
        2 => (col.clone(), col.clone()).prop_map(|(from, to)| Op::Rename { from, to }),
        2 => col.prop_map(|col| Op::Drop { col }),
        1 => Just(Op::Compact),
    ]
}

/// A live column as the oracle knows it.
#[derive(Debug, Clone)]
struct Column {
    id: FieldId,
    name: String,
    kind: Kind,
}

#[derive(Debug, Default)]
struct Oracle {
    live: Vec<Column>,
    next_id: u32,
    /// Each row: its title and its values by field id.
    rows: Vec<(Option<String>, BTreeMap<FieldId, String>)>,
}

impl Oracle {
    fn new() -> Self {
        Self {
            live: vec![Column {
                id: FieldId(1),
                name: "title".into(),
                kind: Kind::Str,
            }],
            next_id: 2,
            rows: Vec::new(),
        }
    }

    fn by_name(&self, name: &str) -> Option<&Column> {
        self.live.iter().find(|c| c.name == name)
    }

    fn add(&mut self, name: &str, kind: Kind) -> FieldId {
        let id = FieldId(self.next_id);
        self.next_id += 1;
        self.live.push(Column {
            id,
            name: name.into(),
            kind,
        });
        id
    }

    /// Whether the batch is stored; `None` means the oracle refuses it
    /// because a present column's pool type disagrees with the live type.
    fn append(&mut self, present: &[bool], rows: &[(Option<usize>, Vec<Option<i64>>)]) -> bool {
        for (i, (name, kind)) in POOL.iter().enumerate() {
            if present[i]
                && let Some(live) = self.by_name(name)
                && live.kind != *kind
            {
                return false;
            }
        }
        let mut ids = Vec::new();
        for (i, (name, kind)) in POOL.iter().enumerate() {
            if !present[i] {
                ids.push(None);
                continue;
            }
            let id = match self.by_name(name) {
                Some(live) => live.id,
                None => self.add(name, *kind),
            };
            ids.push(Some(id));
        }
        for (title, values) in rows {
            let mut cells = BTreeMap::new();
            for (i, (_, kind)) in POOL.iter().enumerate() {
                if let (Some(id), Some(v)) = (ids[i], values[i]) {
                    cells.insert(id, kind.render(v));
                }
            }
            self.rows.push((title.map(|w| WORDS[w].to_string()), cells));
        }
        true
    }

    /// The table as the SQL projection of its live columns renders it,
    /// sorted.
    fn table(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .rows
            .iter()
            .map(|(title, cells)| {
                let mut parts = vec![title.clone().unwrap_or_else(|| "null".into())];
                for c in self.live.iter().skip(1) {
                    parts.push(cells.get(&c.id).cloned().unwrap_or_else(|| "null".into()));
                }
                parts.join("|")
            })
            .collect();
        out.sort();
        out
    }

    fn hits(&self, word: &str) -> usize {
        self.rows
            .iter()
            .filter(|(t, _)| t.as_deref() == Some(word))
            .count()
    }
}

fn cell(array: &ArrayRef, row: usize) -> String {
    if array.is_null(row) {
        return "null".into();
    }
    if let Some(a) = array.as_any().downcast_ref::<LargeStringArray>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return a.value(row).to_string();
    }
    panic!("unexpected column type {:?}", array.data_type());
}

fn table_rows(db: &Connection, oracle: &Oracle) -> Vec<String> {
    let columns: Vec<String> = oracle
        .live
        .iter()
        .map(|c| format!("\"{}\"", c.name))
        .collect();
    let sql = format!("SELECT {} FROM {TABLE}", columns.join(", "));
    let mut out = Vec::new();
    for b in db.query_sql(&sql).expect("select") {
        for row in 0..b.num_rows() {
            out.push(
                (0..b.num_columns())
                    .map(|c| cell(b.column(c), row))
                    .collect::<Vec<_>>()
                    .join("|"),
            );
        }
    }
    out.sort();
    out
}

fn check(db: &Connection, table: &Supertable, oracle: &Oracle) -> Result<(), TestCaseError> {
    let doc = db.schema(TABLE).expect("schema");
    let engine: Vec<(FieldId, &str, DataType)> = doc
        .fields()
        .iter()
        .map(|f| (f.id, f.name.as_str(), f.data_type.clone()))
        .collect();
    let expected: Vec<(FieldId, &str, DataType)> = oracle
        .live
        .iter()
        .map(|c| (c.id, c.name.as_str(), c.kind.data_type()))
        .collect();
    prop_assert_eq!(engine, expected);
    prop_assert_eq!(table_rows(db, oracle), oracle.table());
    for word in WORDS {
        let hits: usize = table
            .bm25_search("title", word, FTS_TOP_K, Default::default(), None)
            .expect("search")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        prop_assert_eq!(hits, oracle.hits(word), "hits for {}", word);
    }
    Ok(())
}

fn field(name: &str, kind: Kind) -> FieldPatch {
    FieldPatch {
        id: None,
        name: name.into(),
        data_type: Some(kind.data_type()),
        nullable: None,
        index: None,
        dropped: false,
    }
}

fn run(ops: Vec<Op>) -> Result<(), TestCaseError> {
    // Storage-backed: compaction needs a store, and the schema document
    // travels through the manifest list the way it does for every durable
    // table.
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let table = db
        .create_table(
            TABLE,
            Arc::new(Schema::new(vec![Field::new(
                "title",
                DataType::LargeUtf8,
                true,
            )])),
            IndexSpec::new().fts("title"),
        )
        .expect("create");
    let mut oracle = Oracle::new();
    for op in ops {
        match op {
            Op::Append { present, rows } => {
                let mut columns: Vec<(&str, ArrayRef)> = vec![(
                    "title",
                    Arc::new(LargeStringArray::from(
                        rows.iter()
                            .map(|(t, _)| t.map(|w| WORDS[w]))
                            .collect::<Vec<_>>(),
                    )),
                )];
                for (i, (name, kind)) in POOL.iter().enumerate() {
                    if present[i] {
                        let values: Vec<Option<i64>> = rows.iter().map(|(_, v)| v[i]).collect();
                        columns.push((name, kind.array(&values)));
                    }
                }
                let fields: Vec<Field> = columns
                    .iter()
                    .map(|(n, a)| Field::new(*n, a.data_type().clone(), true))
                    .collect();
                let batch = RecordBatch::try_new(
                    Arc::new(Schema::new(fields)),
                    columns.into_iter().map(|(_, a)| a).collect(),
                )
                .expect("batch");
                let stored = oracle.append(&present, &rows);
                match table.append(&batch) {
                    Ok(()) => prop_assert!(stored, "the oracle refuses this batch"),
                    Err(InfinoError::Schema(_)) => {
                        prop_assert!(!stored, "the oracle stores this batch")
                    }
                    Err(other) => return Err(TestCaseError::fail(other.to_string())),
                }
            }
            Op::Add { col } => {
                let (name, kind) = POOL[col];
                let patch = SchemaPatch {
                    fields: vec![field(name, kind)],
                    max_fields: None,
                    max_depth: None,
                    templates: None,
                };
                match oracle.by_name(name) {
                    Some(live) if live.kind != kind => {
                        // The write would retype the column; outside this
                        // tier's scope, so it is not applied.
                    }
                    Some(_) => {
                        db.apply_schema(TABLE, &patch, None).expect("no-op add");
                    }
                    None => {
                        db.apply_schema(TABLE, &patch, None).expect("add");
                        oracle.add(name, kind);
                    }
                }
            }
            Op::Rename { from, to } => {
                let Some(live) = oracle.by_name(POOL[from].0).cloned() else {
                    continue;
                };
                let target = POOL[to].0;
                let patch = SchemaPatch {
                    fields: vec![FieldPatch {
                        id: Some(live.id),
                        ..field(target, live.kind)
                    }],
                    max_fields: None,
                    max_depth: None,
                    templates: None,
                };
                let taken = from != to && oracle.by_name(target).is_some();
                match db.apply_schema(TABLE, &patch, None) {
                    Ok(_) => {
                        prop_assert!(!taken, "the name is taken");
                        if let Some(c) = oracle.live.iter_mut().find(|c| c.id == live.id) {
                            c.name = target.into();
                        }
                    }
                    Err(InfinoError::Schema(_)) => prop_assert!(taken, "the name is free"),
                    Err(other) => return Err(TestCaseError::fail(other.to_string())),
                }
            }
            Op::Drop { col } => {
                let Some(live) = oracle.by_name(POOL[col].0).cloned() else {
                    continue;
                };
                let patch = SchemaPatch {
                    fields: vec![FieldPatch {
                        dropped: true,
                        ..field(&live.name, live.kind)
                    }],
                    max_fields: None,
                    max_depth: None,
                    templates: None,
                };
                db.apply_schema(TABLE, &patch, None).expect("drop");
                oracle.live.retain(|c| c.id != live.id);
            }
            Op::Compact => {
                table
                    .optimize(&OptimizeOptions::compact(CompactionSettings {
                        target_superfile_size_mb: COMPACT_TARGET_MB,
                        min_fill_percent: COMPACT_MIN_FILL_PERCENT,
                        ..CompactionSettings::default()
                    }))
                    .expect("compact");
            }
        }
        check(&db, &table, &oracle)?;
    }
    Ok(())
}

fn cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_CASES)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: cases(), ..ProptestConfig::default() })]
    #[test]
    fn the_table_agrees_with_the_oracle_after_every_step(
        ops in proptest::collection::vec(op_strategy(), 1..=MAX_OPS)
    ) {
        run(ops)?;
    }
}
