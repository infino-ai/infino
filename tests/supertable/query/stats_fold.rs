// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Plan-shape and correctness gates for manifest-statistics aggregate
//! folding.
//!
//! On a tombstone-free table, `COUNT(*)` / `MIN` / `MAX` must be
//! answered from manifest statistics — the physical plan contains no
//! scan node at all. With tombstones, `COUNT(*)` may still fold (the
//! bitmap cardinalities are exact) but value-derived stats degrade to
//! a real scan; results must stay correct either way.

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use tempfile::TempDir;

use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::{Supertable, SupertableOptions};
use infino::test_helpers::{build_title_batch, default_supertable_options};

/// Commits in the fold fixture — multiple segments so the statistics
/// fold exercises the cross-segment merge, not a single-segment
/// shortcut.
const COMMITS: usize = 3;
/// Rows per commit.
const ROWS_PER_COMMIT: usize = 64;
/// Disk-cache budget for the tombstone fixture.
const DISK_CACHE_BUDGET_BYTES: u64 = 1 << 30;
/// Mmap promotion timers disabled in tests.
const MMAP_TIMER_DISABLED_SECS: u64 = 0;

fn options_cat_rating() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::LargeUtf8, false),
        Field::new("rating", DataType::Int64, false),
    ]));
    SupertableOptions::new(schema, vec![], vec![], None).expect("valid options")
}

/// Commit `idx` carries categories `cat{idx}_{row}` and ratings
/// `idx*1000 + row` — distinct per row, so MIN/MAX/SUM have known
/// closed forms.
fn build_batch(idx: usize, schema: Arc<Schema>) -> RecordBatch {
    let cats: Vec<String> = (0..ROWS_PER_COMMIT)
        .map(|r| format!("cat{idx}_{r:03}"))
        .collect();
    let ratings: Vec<i64> = (0..ROWS_PER_COMMIT)
        .map(|r| (idx * 1000 + r) as i64)
        .collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(LargeStringArray::from(
                cats.iter().map(String::as_str).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(ratings)),
        ],
    )
    .expect("batch")
}

/// Flatten an `EXPLAIN` result into one searchable string.
fn explain(st: &Supertable, sql: &str) -> String {
    let batches = st
        .reader()
        .query_sql(&format!("EXPLAIN {sql}"))
        .expect("explain");
    let mut out = String::new();
    for batch in &batches {
        for column in batch.columns() {
            if let Some(strings) = column.as_any().downcast_ref::<arrow_array::StringArray>() {
                for i in 0..strings.len() {
                    if !strings.is_null(i) {
                        out.push_str(strings.value(i));
                        out.push('\n');
                    }
                }
            }
        }
    }
    out
}

/// Single-cell i64 result of an aggregate query.
fn scalar_i64(st: &Supertable, sql: &str) -> i64 {
    let batches = st.reader().query_sql(sql).expect("sql");
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("i64 result")
        .value(0)
}

/// Single-cell string result of an aggregate query.
fn scalar_string(st: &Supertable, sql: &str) -> String {
    let batches = st.reader().query_sql(sql).expect("sql");
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    let column = batch.column(0);
    if let Some(s) = column.as_any().downcast_ref::<LargeStringArray>() {
        return s.value(0).to_string();
    }
    column
        .as_any()
        .downcast_ref::<arrow_array::StringArray>()
        .expect("string result")
        .value(0)
        .to_string()
}

#[test]
fn unfiltered_aggregates_fold_without_scanning() {
    let st = Supertable::create(options_cat_rating()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for idx in 0..COMMITS {
        w.append(&build_batch(idx, schema.clone())).expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    let total = (COMMITS * ROWS_PER_COMMIT) as i64;
    let max_rating = ((COMMITS - 1) * 1000 + ROWS_PER_COMMIT - 1) as i64;

    // Values first — folding must never change results.
    assert_eq!(scalar_i64(&st, "SELECT COUNT(*) FROM supertable"), total);
    assert_eq!(
        scalar_i64(&st, "SELECT MAX(rating) FROM supertable"),
        max_rating
    );
    assert_eq!(scalar_i64(&st, "SELECT MIN(rating) FROM supertable"), 0);
    assert_eq!(
        scalar_string(&st, "SELECT MAX(category) FROM supertable"),
        format!("cat{}_{:03}", COMMITS - 1, ROWS_PER_COMMIT - 1)
    );

    // Plan shape: a tombstone-free table answers these from manifest
    // statistics; the physical plan must not contain a scan.
    for sql in [
        "SELECT COUNT(*) FROM supertable",
        "SELECT MAX(rating) FROM supertable",
        "SELECT MIN(rating) FROM supertable",
        "SELECT MAX(category) FROM supertable",
    ] {
        let plan = explain(&st, sql);
        assert!(
            !plan.contains("DataSourceExec"),
            "{sql}: expected statistics fold (no scan); plan was:\n{plan}"
        );
    }

    // SUM has no built-in statistics fold in this DataFusion version —
    // correctness only (closed form: Σ over commits of Σ row + 1000·idx).
    let expected_sum: i64 = (0..COMMITS as i64)
        .map(|idx| {
            (0..ROWS_PER_COMMIT as i64)
                .map(|r| idx * 1000 + r)
                .sum::<i64>()
        })
        .sum();
    assert_eq!(
        scalar_i64(&st, "SELECT SUM(rating) FROM supertable"),
        expected_sum
    );
}

#[test]
fn tombstoned_tables_degrade_but_stay_correct() {
    let dir = TempDir::new().expect("tempdir");
    let cache_dir = TempDir::new().expect("cache");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let cfg = DiskCacheConfig {
        cache_root: cache_dir.path().to_path_buf(),
        disk_budget_bytes: DISK_CACHE_BUDGET_BYTES,
        cold_fetch_mode: ColdFetchMode::HybridWithPrefetch,
        mmap_cold_threshold_secs: MMAP_TIMER_DISABLED_SECS,
        mmap_sweep_interval_secs: MMAP_TIMER_DISABLED_SECS,
        eviction: Box::new(LruPolicy::new()),
        verify_crc_on_open: true,
        ..Default::default()
    };
    let pinned: Arc<dyn Fn() -> HashSet<_> + Send + Sync> = Arc::new(HashSet::new);
    let disk_cache = DiskCacheStore::new(Arc::clone(&storage), cfg, pinned).expect("cache");

    let st = Supertable::create(
        default_supertable_options()
            .with_storage(storage)
            .with_disk_cache(disk_cache),
    )
    .expect("create");

    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(&[
        "alpha", "bravo", "charlie", "delta", "echo",
    ]))
    .expect("append");
    w.commit().expect("commit");

    // Pre-delete: clean table, exact folds.
    assert_eq!(scalar_i64(&st, "SELECT COUNT(*) FROM supertable"), 5);
    assert_eq!(
        scalar_string(&st, "SELECT MAX(title) FROM supertable"),
        "echo"
    );

    // Delete the lexical maximum — the manifest max ("echo") is now a
    // dead row, exactly the case where value stats must degrade.
    let pending = w.delete(col("title").eq(lit("echo"))).expect("delete");
    assert_eq!(pending.matched, 1);
    w.commit().expect("commit delete");
    drop(w);

    // COUNT(*) must reflect the delete (folded or scanned — bitmap
    // cardinalities are exact either way).
    assert_eq!(scalar_i64(&st, "SELECT COUNT(*) FROM supertable"), 4);
    // MAX must NOT report the deleted extremum: a fold from manifest
    // stats would say "echo"; the degraded scan must say "delta".
    assert_eq!(
        scalar_string(&st, "SELECT MAX(title) FROM supertable"),
        "delta"
    );
}
