// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A superfile stores the columns its rows fill. On a wide, sparse table —
//! thousands of nullable columns, each row setting a few — a write stores
//! and describes only the columns it sets, a column it leaves out reads
//! as null, pruning treats a file without a column as null throughout, a
//! merge stores the union of what its inputs hold, and what the table
//! records per superfile follows the columns written, not the width.

use std::{collections::HashSet, fs, path::Path, sync::Arc};

use arrow_array::{Array, ArrayRef, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{CompactionSettings, Connection, IndexSpec, OptimizeOptions, Supertable, connect};
use tempfile::TempDir;

const TABLE: &str = "wide";
/// Columns of the wide table.
const WIDTH: usize = 2_000;
/// Columns one write fills.
const FILLED: usize = 50;
/// Rows of one write.
const ROWS: usize = 100;
/// Small enough that the test's superfiles merge.
const COMPACT_TARGET_MB: u64 = 1;
const COMPACT_MIN_FILL_PERCENT: u8 = 1;
/// Writes in the manifest-growth comparison.
const GROWTH_WRITES: usize = 4;
/// Where the table keeps its manifest parts, one per commit.
const MANIFEST_PARTS_DIR: &str = "manifest-parts";
/// Where the table keeps the per-superfile summaries a part points at.
const MANIFEST_SUMMARIES_DIR: &str = "manifest-summaries";

fn column(i: usize) -> String {
    format!("c{i:04}")
}

fn wide_schema(width: usize) -> Arc<Schema> {
    Arc::new(Schema::new(
        (0..width)
            .map(|i| Field::new(column(i), DataType::LargeUtf8, true))
            .collect::<Vec<_>>(),
    ))
}

/// `ROWS` rows over every column of `schema`, setting `value` in the
/// columns `filled` names (every row) and null in every other.
fn rows_setting(schema: &Arc<Schema>, filled: &[usize], value: &str) -> RecordBatch {
    let set: HashSet<String> = filled.iter().map(|&i| column(i)).collect();
    let columns: Vec<ArrayRef> = schema
        .fields()
        .iter()
        .map(|f| -> ArrayRef {
            match set.contains(f.name()) {
                true => Arc::new(LargeStringArray::from(vec![Some(value); ROWS])),
                false => Arc::new(LargeStringArray::from(vec![None::<&str>; ROWS])),
            }
        })
        .collect();
    RecordBatch::try_new(Arc::clone(schema), columns).expect("batch")
}

fn table(dir: &TempDir, width: usize) -> (Connection, Supertable) {
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let t = db
        .create_table(TABLE, wide_schema(width), IndexSpec::new())
        .expect("create");
    (db, t)
}

/// The names of the columns each live superfile records it holds, the id
/// column aside, in manifest order.
fn stored_columns(t: &Supertable) -> Vec<HashSet<String>> {
    let reader = t.local_handle().reader().expect("reader");
    reader
        .manifest()
        .get_all_superfiles()
        .iter()
        .map(|entry| {
            entry
                .physical_schema
                .as_ref()
                .expect("a new file records its columns")
                .columns()
                .iter()
                .map(|c| c.name.clone())
                .filter(|name| name.starts_with('c'))
                .collect()
        })
        .collect()
}

/// Stats each live superfile records, by column count.
fn stat_columns(t: &Supertable) -> Vec<usize> {
    let reader = t.local_handle().reader().expect("reader");
    reader
        .manifest()
        .get_all_superfiles()
        .iter()
        .map(|entry| entry.scalar_stats.columns().len())
        .collect()
}

fn count(db: &Connection, sql: &str) -> i64 {
    let batches = db.query_sql(sql).expect("query");
    let total = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count");
    total.value(0)
}

/// Bytes of the files under `dir` whose path passes `keep`, recursively.
fn bytes_under(dir: &Path, keep: &dyn Fn(&str) -> bool) -> u64 {
    fs::read_dir(dir)
        .expect("read dir")
        .flatten()
        .map(|e| {
            let path = e.path();
            if path.is_dir() {
                bytes_under(&path, keep)
            } else if keep(&path.to_string_lossy()) {
                e.metadata().map(|m| m.len()).unwrap_or(0)
            } else {
                0
            }
        })
        .sum()
}

/// Bytes of what the table records per superfile — its manifest parts and
/// their summaries — and of the superfiles themselves.
fn per_superfile_bytes(dir: &Path) -> (u64, u64) {
    let records = bytes_under(dir, &|p| {
        p.contains(MANIFEST_PARTS_DIR) || p.contains(MANIFEST_SUMMARIES_DIR)
    });
    let superfiles = bytes_under(dir, &|p| p.ends_with(".sf.parquet"));
    (records, superfiles)
}

#[test]
fn a_write_stores_only_the_columns_its_rows_fill() {
    let dir = TempDir::new().expect("tempdir");
    let (db, t) = table(&dir, WIDTH);
    let filled: Vec<usize> = (0..FILLED).map(|i| i * (WIDTH / FILLED)).collect();
    t.append(&rows_setting(&t.schema(), &filled, "x"))
        .expect("append");

    let want: HashSet<String> = filled.iter().map(|&i| column(i)).collect();
    for stored in stored_columns(&t) {
        assert_eq!(stored, want, "a file stores the columns its rows set");
    }
    for stats in stat_columns(&t) {
        // The filled columns and the id column.
        assert_eq!(stats, FILLED + 1, "stats only for what the file stores");
    }
    // A column no row set reads null in every row; one that was set reads
    // its value.
    let unfilled = column(1);
    assert_eq!(
        count(
            &db,
            &format!("SELECT count(*) FROM {TABLE} WHERE {unfilled} IS NULL")
        ),
        ROWS as i64
    );
    assert_eq!(
        count(&db, &format!("SELECT count({unfilled}) FROM {TABLE}")),
        0
    );
    assert_eq!(
        count(
            &db,
            &format!("SELECT count(*) FROM {TABLE} WHERE {} = 'x'", column(0))
        ),
        ROWS as i64
    );
}

/// Two writes setting disjoint columns: `=` and `IS NOT NULL` on a column
/// find only the rows of the write that set it, `IS NULL` finds the rest,
/// before and after a merge, and the merged file stores the union.
#[test]
fn files_of_disjoint_columns_answer_and_merge_into_their_union() {
    let dir = TempDir::new().expect("tempdir");
    let (db, t) = table(&dir, WIDTH);
    let schema = t.schema();
    let (a, b) = (column(10), column(20));
    t.append(&rows_setting(&schema, &[10], "a"))
        .expect("append a");
    t.append(&rows_setting(&schema, &[20], "b"))
        .expect("append b");

    let answers = |db: &Connection| {
        [
            count(db, &format!("SELECT count(*) FROM {TABLE} WHERE {a} = 'a'")),
            count(
                db,
                &format!("SELECT count(*) FROM {TABLE} WHERE {a} IS NULL"),
            ),
            count(
                db,
                &format!("SELECT count(*) FROM {TABLE} WHERE {a} IS NOT NULL"),
            ),
            count(db, &format!("SELECT count(*) FROM {TABLE} WHERE {b} = 'a'")),
            count(
                db,
                &format!("SELECT count(*) FROM {TABLE} WHERE {b} IS NULL"),
            ),
            count(
                db,
                &format!("SELECT count(*) FROM {TABLE} WHERE {a} IS NULL AND {b} IS NULL"),
            ),
        ]
    };
    let rows = ROWS as i64;
    let expected = [rows, rows, rows, 0, rows, 0];
    assert_eq!(answers(&db), expected);

    t.optimize(&OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: COMPACT_TARGET_MB,
        min_fill_percent: COMPACT_MIN_FILL_PERCENT,
        min_superfiles_for_merge: 2,
        ..CompactionSettings::default()
    }))
    .expect("compact");
    let stored = stored_columns(&t);
    assert_eq!(stored.len(), 1, "the two files merged");
    assert_eq!(stored[0], HashSet::from([a.clone(), b.clone()]));
    assert_eq!(answers(&db), expected);
}

/// What a write costs per superfile follows the columns it sets, not how
/// wide the table is: the same rows written to a table forty times wider
/// make superfiles, manifest parts and summaries of the same size.
#[test]
fn what_a_write_records_follows_its_columns_not_the_width() {
    let filled: Vec<usize> = (0..FILLED).collect();
    let growth = |width: usize| -> (u64, u64) {
        let dir = TempDir::new().expect("tempdir");
        let (_db, t) = table(&dir, width);
        let schema = t.schema();
        t.append(&rows_setting(&schema, &filled, "x"))
            .expect("first append");
        let (records, superfiles) = per_superfile_bytes(dir.path());
        for _ in 0..GROWTH_WRITES {
            t.append(&rows_setting(&schema, &filled, "x"))
                .expect("append");
        }
        let (records_after, superfiles_after) = per_superfile_bytes(dir.path());
        (records_after - records, superfiles_after - superfiles)
    };
    let narrow = growth(FILLED);
    let wide = growth(WIDTH);
    assert!(
        narrow.0 > 0 && narrow.1 > 0,
        "the writes recorded something"
    );
    assert_eq!(
        wide, narrow,
        "(manifest records, superfile bytes) over {GROWTH_WRITES} writes"
    );
}
