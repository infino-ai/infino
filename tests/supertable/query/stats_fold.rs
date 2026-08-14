// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Plan-shape and correctness gates for manifest statistics.
//!
//! On a tombstone-free table, `COUNT(*)` / `MIN` / `MAX` must be
//! answered from manifest statistics — the physical plan contains no
//! scan node at all. With tombstones, `COUNT(*)` may still fold (the
//! bitmap cardinalities are exact) but value-derived stats degrade to
//! a real scan; results must stay correct either way.
//!
//! The same statistics reach the planner a second way, as the column
//! bounds behind its row estimate for a filter. Bounds covering a type's
//! entire domain are withheld there (see `spans_full_domain` in
//! `supertable::query::provider`), so the last group below pins both
//! halves of that: range filters on such a column return rows, and
//! withholding the bounds does not cost the aggregate fold.

#![deny(clippy::unwrap_used)]

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Date32Array, Int64Array, LargeStringArray, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use rayon::{ThreadPool, ThreadPoolBuilder};
use tempfile::TempDir;

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
        .expect("reader")
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
    let batches = st.reader().expect("reader").query_sql(sql).expect("sql");
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
    let batches = st.reader().expect("reader").query_sql(sql).expect("sql");
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

// ---- temporal columns (Date32) ------------------------------------
//
// The manifest now records min/max for temporal types, so `MIN`/`MAX`
// on a date column fold like an integer. This is the ClickBench
// `EventDate` shape. `id` is a monotonic Int64 aligned with `day`, so a
// delete can target the extremum row by a plain integer literal.

/// Schema `(day: Date32, id: Int64)`; `id` tracks `day` so the max id is
/// the max day.
fn options_day_id() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("day", DataType::Date32, false),
        Field::new("id", DataType::Int64, false),
    ]));
    SupertableOptions::new(schema, vec![], vec![], None).expect("valid options")
}

/// Base day (days-since-epoch) for commit 0; later commits and rows step
/// strictly upward so extrema have closed forms.
const DAY_BASE: i32 = 15000;

fn build_day_batch(idx: usize, schema: Arc<Schema>) -> RecordBatch {
    let days: Vec<i32> = (0..ROWS_PER_COMMIT)
        .map(|r| DAY_BASE + (idx * 100 + r) as i32)
        .collect();
    let ids: Vec<i64> = (0..ROWS_PER_COMMIT)
        .map(|r| (idx * 1000 + r) as i64)
        .collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Date32Array::from(days)),
            Arc::new(Int64Array::from(ids)),
        ],
    )
    .expect("batch")
}

/// Single-cell Date32 (days-since-epoch) result of an aggregate query.
fn scalar_date32(st: &Supertable, sql: &str) -> i32 {
    let batches = st.reader().expect("reader").query_sql(sql).expect("sql");
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("date32 result")
        .value(0)
}

#[test]
fn temporal_aggregates_fold_without_scanning() {
    let st = Supertable::create(options_day_id()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for idx in 0..COMMITS {
        w.append(&build_day_batch(idx, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
    }
    drop(w);

    let min_day = DAY_BASE;
    let max_day = DAY_BASE + ((COMMITS - 1) * 100 + ROWS_PER_COMMIT - 1) as i32;

    // Values first — the fold must never change results.
    assert_eq!(
        scalar_date32(&st, "SELECT MIN(day) FROM supertable"),
        min_day
    );
    assert_eq!(
        scalar_date32(&st, "SELECT MAX(day) FROM supertable"),
        max_day
    );

    // Plan shape: a tombstone-free date column folds from manifest stats,
    // so the physical plan has no scan node. This is the regression that
    // fails before temporal min/max is recorded (the column had no bounds,
    // so `MIN`/`MAX(day)` fell back to a full `DataSourceExec` scan).
    for sql in [
        "SELECT MIN(day) FROM supertable",
        "SELECT MAX(day) FROM supertable",
    ] {
        let plan = explain(&st, sql);
        assert!(
            !plan.contains("DataSourceExec"),
            "{sql}: expected temporal statistics fold (no scan); plan was:\n{plan}"
        );
    }
}

#[test]
fn temporal_fold_excludes_deleted_extremum() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(options_day_id().with_storage(storage)).expect("create");
    let schema = st.options().schema.clone();

    let mut w = st.writer().expect("writer");
    for idx in 0..COMMITS {
        w.append(&build_day_batch(idx, schema.clone()))
            .expect("append");
        w.commit().expect("commit");
    }

    let max_day = DAY_BASE + ((COMMITS - 1) * 100 + ROWS_PER_COMMIT - 1) as i32;
    let second_max_day = max_day - 1;
    let max_id = ((COMMITS - 1) * 1000 + ROWS_PER_COMMIT - 1) as i64;

    // Clean table: max folds to the true extremum.
    assert_eq!(
        scalar_date32(&st, "SELECT MAX(day) FROM supertable"),
        max_day
    );

    // Delete the row holding the max day (by its aligned id). The manifest
    // max is now a dead row: a fold would report the stale `max_day`; the
    // clean-view gate must decline the fold and the scan must report the
    // true survivor max.
    let pending = w.delete(col("id").eq(lit(max_id))).expect("delete");
    assert_eq!(pending.matched, 1);
    w.commit().expect("commit delete");
    drop(w);

    assert_eq!(
        scalar_date32(&st, "SELECT MAX(day) FROM supertable"),
        second_max_day
    );
}

/// One writer thread means one superfile per commit, so these fixtures
/// control exactly which rows share a min/max range. The default pool
/// shards a small batch across superfiles by row range, which would
/// scatter the endpoints and make the fold depend on the host's core
/// count.
const FULL_DOMAIN_WRITER_THREADS: usize = 1;

/// Row ids for the `UInt64` fixture, one per row, in append order.
const U64_IDS: [i64; 5] = [1, 2, 3, 4, 5];
/// The `u` column: both domain endpoints plus small and mid values, so a
/// single superfile's min/max cover `0 ..= u64::MAX` exactly.
const US: [u64; 5] = [0, 1, 1000, 1 << 63, u64::MAX];
/// Mid-range literal for the `u` filters. Nowhere near the type's upper
/// bound; only the column's committed range is.
const SMALL_LITERAL: u64 = 1000;

/// Row ids and `n` values for the split-domain `Int64` fixture. The first
/// commit contributes the lower endpoint, the second the upper one, so
/// neither superfile spans the domain on its own and only the fold does.
const SPLIT_IDS_FIRST: [i64; 2] = [1, 2];
const SPLIT_NS_FIRST: [i64; 2] = [i64::MIN, 5];
const SPLIT_IDS_SECOND: [i64; 2] = [3, 4];
const SPLIT_NS_SECOND: [i64; 2] = [10, i64::MAX];
/// Threshold for the split-domain query, chosen so every superfile
/// survives pruning: the first still holds `5`, the second only larger
/// values. A threshold that pruned either would narrow the folded range
/// and stop exercising the withholding at all.
const SPLIT_THRESHOLD: i64 = 0;

fn single_superfile_writer_pool() -> Arc<ThreadPool> {
    Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(FULL_DOMAIN_WRITER_THREADS)
            .build()
            .expect("writer pool"),
    )
}

/// Schema `(i: Int64, u: UInt64)`; `i` labels the row, `u` carries the
/// values under test.
fn options_full_domain_u64() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("u", DataType::UInt64, false),
    ]));
    SupertableOptions::new(schema, vec![], vec![], None)
        .expect("valid options")
        .with_writer_pool(single_superfile_writer_pool())
}

/// Schema `(i: Int64, n: Int64)`, the signed counterpart of
/// [`options_full_domain_u64`].
fn options_split_domain_i64() -> SupertableOptions {
    let schema = Arc::new(Schema::new(vec![
        Field::new("i", DataType::Int64, false),
        Field::new("n", DataType::Int64, false),
    ]));
    SupertableOptions::new(schema, vec![], vec![], None)
        .expect("valid options")
        .with_writer_pool(single_superfile_writer_pool())
}

/// Single-superfile table over [`U64_IDS`] / [`US`]: one superfile holds
/// both `UInt64` endpoints.
fn full_domain_u64_table() -> Supertable {
    let st = Supertable::create(options_full_domain_u64()).expect("create");
    let schema = st.options().schema.clone();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(U64_IDS.to_vec())),
            Arc::new(UInt64Array::from(US.to_vec())),
        ],
    )
    .expect("batch");
    let mut w = st.writer().expect("writer");
    w.append(&batch).expect("append");
    w.commit().expect("commit");
    drop(w);
    st
}

/// Two-superfile table where each superfile holds one `Int64` endpoint,
/// so only the fold across survivors spans the domain.
fn split_domain_i64_table() -> Supertable {
    let st = Supertable::create(options_split_domain_i64()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for (ids, ns) in [
        (SPLIT_IDS_FIRST, SPLIT_NS_FIRST),
        (SPLIT_IDS_SECOND, SPLIT_NS_SECOND),
    ] {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids.to_vec())),
                Arc::new(Int64Array::from(ns.to_vec())),
            ],
        )
        .expect("batch");
        w.append(&batch).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

/// The `i` column of `sql`'s result, sorted, so assertions don't depend
/// on fan-out order.
fn select_ids(st: &Supertable, sql: &str) -> Vec<i64> {
    let batches = st
        .reader()
        .expect("reader")
        .query_sql(sql)
        .unwrap_or_else(|e| panic!("query failed: {sql}: {e}"));
    let mut ids = Vec::new();
    for batch in &batches {
        let idx = batch.schema().index_of("i").expect("i column");
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("i is Int64");
        for r in 0..arr.len() {
            if !arr.is_null(r) {
                ids.push(arr.value(r));
            }
        }
    }
    ids.sort_unstable();
    ids
}

/// Single-cell u64 result of an aggregate query.
fn scalar_u64(st: &Supertable, sql: &str) -> u64 {
    let batches = st.reader().expect("reader").query_sql(sql).expect("sql");
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    batch
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .expect("u64 result")
        .value(0)
}

/// Every comparison shape over a whole-domain column returns the right
/// rows. `=` and the boundary `>=` always worked, because they collapse
/// the column's range to a single point and take a distinct-count path
/// that never divides; they are here so a change to the withholding rule
/// can't quietly break the side that was fine.
#[test]
fn full_domain_column_answers_every_comparison_shape() {
    let st = full_domain_u64_table();
    let max = u64::MAX;

    for (sql, expected) in [
        (format!("SELECT i FROM supertable WHERE u = {max}"), vec![5]),
        (
            format!("SELECT i FROM supertable WHERE u >= {max}"),
            vec![5],
        ),
        (
            format!("SELECT i FROM supertable WHERE u = {SMALL_LITERAL}"),
            vec![3],
        ),
        (
            format!("SELECT i FROM supertable WHERE u > {SMALL_LITERAL}"),
            vec![4, 5],
        ),
        (
            format!("SELECT i FROM supertable WHERE u < {max}"),
            vec![1, 2, 3, 4],
        ),
        (
            format!("SELECT i FROM supertable WHERE u BETWEEN 1 AND {max}"),
            vec![2, 3, 4, 5],
        ),
    ] {
        assert_eq!(select_ids(&st, &sql), expected, "{sql}");
    }
}

/// The domain is spanned only after folding two superfiles' bounds
/// together, and the column is `Int64` rather than `UInt64`: the same
/// failure reached along the two axes the single-superfile fixture
/// doesn't cover. The threshold excludes only the `i64::MIN` row.
#[test]
fn range_filter_holds_when_domain_is_split_across_superfiles() {
    let st = split_domain_i64_table();
    assert_eq!(
        select_ids(
            &st,
            &format!("SELECT i FROM supertable WHERE n > {SPLIT_THRESHOLD}")
        ),
        vec![2, 3, 4]
    );
}

/// Withholding whole-domain bounds costs the aggregate fold nothing.
/// Both extremes are still the right answer, and the query is still
/// answered without a scan: the fold reads the per-superfile stats
/// directly, so it never saw the withheld bounds.
#[test]
fn full_domain_column_still_folds_min_max_without_scanning() {
    let st = full_domain_u64_table();

    assert_eq!(scalar_u64(&st, "SELECT MIN(u) FROM supertable"), 0);
    assert_eq!(scalar_u64(&st, "SELECT MAX(u) FROM supertable"), u64::MAX);
    assert_eq!(
        scalar_i64(&st, "SELECT COUNT(*) FROM supertable"),
        U64_IDS.len() as i64
    );

    for sql in [
        "SELECT MIN(u) FROM supertable",
        "SELECT MAX(u) FROM supertable",
        "SELECT COUNT(*) FROM supertable",
    ] {
        let plan = explain(&st, sql);
        assert!(
            !plan.contains("DataSourceExec"),
            "{sql}: expected statistics fold (no scan); plan was:\n{plan}"
        );
    }
}
