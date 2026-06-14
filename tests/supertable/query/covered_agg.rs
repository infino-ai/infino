// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Gates for the covered/residual aggregate rewrite: filter-aligned
//! range aggregates answer covered segments from manifest statistics
//! and scan only boundary segments.
//!
//! Observable: the rewritten plan aggregates `__resid_*` partials.
//! Correctness gates compare every supported aggregate against closed
//! forms on planted data — aligned ranges (rewrite fires), misaligned
//! ranges (falls back), and tombstoned segments (demote to residual).

#![deny(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use tempfile::TempDir;

use infino::storage::{LocalFsStorageProvider, StorageProvider};
use infino::supertable::reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy};
use infino::supertable::{Supertable, SupertableOptions};

/// Commits in the fixture; each commit's ratings occupy a disjoint
/// thousand-block so range filters can align with (or deliberately
/// cut) commit boundaries.
const COMMITS: usize = 4;
/// Rows per commit.
const ROWS_PER_COMMIT: usize = 50;
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

/// Commit `idx`: ratings `idx*1000 .. idx*1000 + ROWS_PER_COMMIT`.
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

fn build_table() -> Supertable {
    let st = Supertable::create(options_cat_rating()).expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for idx in 0..COMMITS {
        w.append(&build_batch(idx, schema.clone())).expect("append");
        w.commit().expect("commit");
    }
    drop(w);
    st
}

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

fn scalar_f64(st: &Supertable, sql: &str) -> f64 {
    let batches = st.reader().query_sql(sql).expect("sql");
    let batch = batches.iter().find(|b| b.num_rows() > 0).expect("one row");
    batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("f64 result")
        .value(0)
}

/// Closed forms over commits `lo..=hi` (whole commits).
fn commit_range_count(lo: usize, hi: usize) -> i64 {
    ((hi - lo + 1) * ROWS_PER_COMMIT) as i64
}
fn commit_range_sum(lo: usize, hi: usize) -> i64 {
    (lo..=hi)
        .map(|idx| {
            (0..ROWS_PER_COMMIT as i64)
                .map(|r| (idx as i64) * 1000 + r)
                .sum::<i64>()
        })
        .sum()
}

#[test]
fn aligned_range_aggregates_use_covered_residual_and_stay_exact() {
    let st = build_table();

    // Range covering commits 1..=2 entirely, with slack inside the
    // empty thousand-block gaps — commits 1 and 2 are fully covered,
    // commits 0 and 3 disjoint, so the residual is EMPTY.
    let where_range = "rating BETWEEN 1000 AND 2999";

    let count_sql = format!("SELECT COUNT(*) FROM supertable WHERE {where_range}");
    let sum_sql = format!("SELECT SUM(rating) FROM supertable WHERE {where_range}");
    let min_sql = format!("SELECT MIN(rating) FROM supertable WHERE {where_range}");
    let max_sql = format!("SELECT MAX(rating) FROM supertable WHERE {where_range}");
    let avg_sql = format!("SELECT AVG(rating) FROM supertable WHERE {where_range}");

    assert_eq!(scalar_i64(&st, &count_sql), commit_range_count(1, 2));
    assert_eq!(scalar_i64(&st, &sum_sql), commit_range_sum(1, 2));
    assert_eq!(scalar_i64(&st, &min_sql), 1000);
    assert_eq!(scalar_i64(&st, &max_sql), 2000 + ROWS_PER_COMMIT as i64 - 1);
    let expected_avg = commit_range_sum(1, 2) as f64 / commit_range_count(1, 2) as f64;
    let got_avg = scalar_f64(&st, &avg_sql);
    assert!(
        (got_avg - expected_avg).abs() < 1e-9,
        "avg {got_avg} vs {expected_avg}"
    );

    // The rewrite fired: plans aggregate the residual partials.
    for sql in [&count_sql, &sum_sql, &min_sql, &max_sql, &avg_sql] {
        let plan = explain(&st, sql);
        assert!(
            plan.contains("__resid_0"),
            "{sql}: expected the covered/residual rewrite; plan was:\n{plan}"
        );
    }
}

#[test]
fn boundary_cutting_range_mixes_covered_and_residual_exactly() {
    let st = build_table();

    // Cuts INTO commits 0 and 3 (boundary) while fully covering
    // commits 1 and 2 (covered): the mixed covered + residual path.
    let where_range = "rating >= 25 AND rating <= 3025";
    let count_sql = format!("SELECT COUNT(*) FROM supertable WHERE {where_range}");
    let sum_sql = format!("SELECT SUM(rating) FROM supertable WHERE {where_range}");

    // Closed forms: commit 0 contributes rows 25.., commits 1 and 2
    // everything, commit 3 rows ..=25.
    let expected_count = (ROWS_PER_COMMIT - 25) as i64 + 2 * ROWS_PER_COMMIT as i64 + 26;
    let expected_sum: i64 = (25..ROWS_PER_COMMIT as i64).sum::<i64>()
        + commit_range_sum(1, 2)
        + (0..=25).map(|r| 3000 + r).sum::<i64>();

    assert_eq!(scalar_i64(&st, &count_sql), expected_count);
    assert_eq!(scalar_i64(&st, &sum_sql), expected_sum);

    let plan = explain(&st, &count_sql);
    assert!(
        plan.contains("__resid_0"),
        "boundary-cutting range should still rewrite (covered middle); plan was:\n{plan}"
    );
}

#[test]
fn non_range_conjunct_declines_rewrite_and_stays_exact() {
    let st = build_table();

    // The range alone would cover commits 1..=2, but the extra
    // non-range conjunct means manifest bounds can't answer any
    // segment — the rewrite must see the WHOLE predicate or decline.
    let where_clause = "rating BETWEEN 1000 AND 2999 AND category <> 'cat1_005'";
    let count_sql = format!("SELECT COUNT(*) FROM supertable WHERE {where_clause}");
    let sum_sql = format!("SELECT SUM(rating) FROM supertable WHERE {where_clause}");

    assert_eq!(scalar_i64(&st, &count_sql), commit_range_count(1, 2) - 1);
    assert_eq!(scalar_i64(&st, &sum_sql), commit_range_sum(1, 2) - 1005);

    let plan = explain(&st, &count_sql);
    assert!(
        !plan.contains("__resid_0"),
        "a predicate with a non-range conjunct must not rewrite; plan was:\n{plan}"
    );
}

#[test]
fn tombstoned_covered_segment_demotes_to_residual() {
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
        options_cat_rating()
            .with_storage(storage)
            .with_disk_cache(disk_cache),
    )
    .expect("create");
    let schema = st.options().schema.clone();
    let mut w = st.writer().expect("writer");
    for idx in 0..COMMITS {
        w.append(&build_batch(idx, schema.clone())).expect("append");
        w.commit().expect("commit");
    }

    // Delete one row INSIDE the would-be-covered range: rating 1005.
    let pending = w.delete(col("rating").eq(lit(1005i64))).expect("delete");
    assert_eq!(pending.matched, 1);
    w.commit().expect("commit delete");
    drop(w);

    let where_range = "rating BETWEEN 1000 AND 2999";
    let count_sql = format!("SELECT COUNT(*) FROM supertable WHERE {where_range}");
    let sum_sql = format!("SELECT SUM(rating) FROM supertable WHERE {where_range}");

    // Answering the tombstoned segment from stats would report the
    // pre-delete numbers; the demotion must surface the post-delete
    // truth.
    assert_eq!(scalar_i64(&st, &count_sql), commit_range_count(1, 2) - 1);
    assert_eq!(scalar_i64(&st, &sum_sql), commit_range_sum(1, 2) - 1005);
}
