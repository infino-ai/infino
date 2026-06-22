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

use std::{collections::HashSet, sync::Arc};

use arrow_array::{Array, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use datafusion::prelude::{col, lit};
use infino::{
    storage::{LocalFsStorageProvider, StorageProvider},
    supertable::{
        Supertable, SupertableOptions,
        reader_cache::{ColdFetchMode, DiskCacheConfig, DiskCacheStore, LruPolicy},
    },
};
use tempfile::TempDir;

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

/// `IN`-list scalar pruning must keep every superfile holding a listed
/// value (the disjoint rating blocks put each value in its own superfile):
///  - a wrong prune would drop a superfile and undercount.
///  - `FilterExec` verifies rows, so the count is the exact oracle.
#[test]
fn in_list_filter_returns_every_matching_row_across_superfiles() {
    let st = build_table();

    // 5 → commit 0, 1005 → commit 1, 3005 → commit 3: three superfiles.
    let spanning = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (5, 1005, 3005)",
    );
    assert_eq!(spanning, 3, "one row per value, across three superfiles");

    // A value outside every superfile's range matches nothing.
    let absent = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (99999)",
    );
    assert_eq!(absent, 0);

    // Mixed present/absent → only the present values contribute.
    let mixed = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (5, 99999, 2005)",
    );
    assert_eq!(mixed, 2);
}

// ---- How DataFusion reshapes a numeric IN before the provider ----
//
// Verified against DataFusion 53.1.0 source (`datafusion-optimizer`):
// `ShortenInListSimplifier` in `simplify_expressions/inlist_simplifier.rs`
// rewrites `expr IN (list)` when:
//   * `list.len() == 1`            → `expr = A` (any expr), OR
//   * `list.len() <= THRESHOLD_INLINE_INLIST` (=3) AND `expr` is a bare
//     column → `expr = A OR expr = B OR …` (negated → AND of `!=`).
// Otherwise the node is kept as `Expr::InList` (len ≥ 4, or a 2–3 list
// over a non-column expression).
//
// So, for a plain `col IN (...)` the scan sees:
//   IN (x)         → `col = x`               (BinaryExpr Eq)
//   IN (a,b,c) ≤3  → OR-of-equalities        (top-level Operator::Or)
//   IN (a,b,c,d)≥4 → `Expr::InList`          (kept)
//   IN (SELECT …)  → `Expr::InSubquery`, decorrelated to a LeftSemi join
//                    by `DecorrelatePredicateSubquery` → the OUTER scan
//                    gets ZERO pushed filters.
//
// Correctness pins — the count is identical regardless of shape, since
// `FilterExec` verifies above the scan.

#[test]
fn numeric_in_single_value_rewrites_to_equality() {
    let st = build_table();
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (5)",
    );
    assert_eq!(n, 1);
}

#[test]
fn numeric_in_small_list_lowers_to_or() {
    // 2 elements → `rating=5 OR rating=1005`.
    let st = build_table();
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (5, 1005)",
    );
    assert_eq!(n, 2);
}

#[test]
fn numeric_in_large_list_stays_inlist() {
    // 6 elements → kept as `Expr::InList` (≥ 4). 1,2,3 in commit 0;
    // 1001,1002 in commit 1; 2001 in commit 2.
    let st = build_table();
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating IN (1, 2, 3, 1001, 1002, 2001)",
    );
    assert_eq!(n, 6);
}

#[test]
fn subquery_in_becomes_a_semi_join_not_a_pushed_filter() {
    // `rating IN (SELECT rating FROM supertable WHERE rating < 10)` →
    // a semi-join: the inner scan filters `rating < 10` (commit 0's
    // ratings 0..9 = 10 rows), the outer scan gets no pushed filter, and
    // they join. The IN never reaches the provider as a prunable filter.
    let st = build_table();
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable \
         WHERE rating IN (SELECT rating FROM supertable WHERE rating < 10)",
    );
    assert_eq!(n, 10);
}

#[test]
fn function_wrapped_small_in_stays_inlist_not_or() {
    // A 2-value IN over a *function* expression stays `Expr::InList`:
    //  - the 2-3 element OR rewrite requires a bare column
    //    (`expr.try_as_col()` in ShortenInListSimplifier).
    //  - `lower(category)` fails that, so no OR rewrite.
    // category = 'cat{idx}_{r:03}'; 'cat0_005' + 'cat1_005' → 2 rows.
    let st = build_table();
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable \
         WHERE lower(category) IN ('cat0_005', 'cat1_005')",
    );
    assert_eq!(n, 2);
}

#[test]
fn not_in_returns_the_complement() {
    // NOT IN never prunes here:
    //  - small (≤3) → AND of `!=`; ≥4 → a negated InList.
    //  - `collect_in_list_leaves` bails on `negated`, and `!=` doesn't
    //    prune meaningfully → full scan + FilterExec.
    // So just assert the exact complement (no rows wrongly dropped).
    // 4 commits × 50 = 200 rows; ratings are unique.
    let st = build_table();
    let total = (COMMITS * ROWS_PER_COMMIT) as i64;

    // Small NOT IN (2 values) → `rating != 5 AND rating != 1005`.
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating NOT IN (5, 1005)",
    );
    assert_eq!(n, total - 2);

    // Large NOT IN (≥4) → negated InList (kept as-is, no leaf emitted).
    let n = scalar_i64(
        &st,
        "SELECT COUNT(*) AS n FROM supertable WHERE rating NOT IN (1, 2, 3, 1001)",
    );
    assert_eq!(n, total - 4);
}
