// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors
//! The filters a parent hands a scan of kept decoded columns, which tighten
//! as the statement runs, skip work and change no answer. With no `WHERE`
//! between the top-k and the scan, the scan reads its row groups best first
//! by the sort column's statistics; a top-k's bound then skips the row
//! groups it rules out and drops the rows it rules out from each batch.
//! Checked against a full sort, ascending and descending: on a column whose
//! values follow the row groups, on one whose values spread across every
//! row group, on one older row groups hold no statistics for (its nulls
//! ordered first and last), and on one widened after some row groups were
//! written. A hash join's filter skips row groups, and the join and a
//! `MIN`/`MAX` aggregate answer as without them. And, under a `WHERE` no
//! index answers and with deleted rows left out, a top-k keyed on the id and
//! on a column no row holds a value in answers as a full sort too. Its own
//! binary, because the host configuration (and with it the process's
//! decoded-column cache) is read once per process.
#![deny(clippy::unwrap_used)]

use std::sync::Arc;

use datafusion::prelude::{col, lit};
use infino::{
    FieldPatch, IndexSpec, SchemaPatch,
    arrow_array::{
        Array, Decimal128Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    },
    arrow_schema::{DataType, Field, Schema},
    connect,
};
use tempfile::TempDir;

/// Bytes of decoded columns the process keeps: far more than the tables
/// hold, so every column the first statement decodes stays.
const KEPT_BYTES: u64 = 64 * 1024 * 1024;

/// Rows in each commit of `t`, each a superfile of one row group: more than
/// DataFusion's default batch of 8,192 rows, so a row group goes out as two
/// batches.
const ROWS: i64 = 10_000;

/// Row groups of `t` per partition of its scan.
const GROUPS_PER_PARTITION: i64 = 3;

/// Fewest partitions `t` is sized for.
const MIN_PARTITIONS: i64 = 3;

/// A prime no row count of `t` is a multiple of: `seq * SPREAD % rows`
/// spreads `seq`'s values across every row group.
const SPREAD: i64 = 7_919;

/// Commits of `w` before its column is widened, and again after.
const WIDE_COMMITS: i64 = 20;

/// Rows in each commit of `w`.
const WIDE_ROWS: i64 = 100;

/// The `LIMIT` the skipping is measured with.
const SMALL_K: usize = 10;

/// How `EXPLAIN ANALYZE` names the rows of the row groups the scan read.
const ROWS_READ: &str = "rows_read";

/// How `EXPLAIN ANALYZE` names the row groups the bound ruled out unread.
const ROW_GROUPS_SKIPPED: &str = "row_groups_skipped";

/// How `EXPLAIN ANALYZE` names the rows the bound dropped from batches.
const ROWS_DROPPED: &str = "rows_dropped";

/// The scale of each suffix `EXPLAIN ANALYZE` prints a count with.
const COUNT_SUFFIXES: [(&str, f64); 3] = [(" K", 1e3), (" M", 1e6), (" B", 1e9)];

/// Column `at` of `batches`, an `Int64` column, in order.
fn int64s(batches: &[RecordBatch], at: usize) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|b| {
            let c = b
                .column(at)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("an Int64 column")
                .clone();
            (0..c.len())
                .map(move |i| c.is_valid(i).then(|| c.value(i)))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The `_id` column across `batches`, in order.
fn ids(batches: &[RecordBatch]) -> Vec<i128> {
    batches
        .iter()
        .flat_map(|b| {
            let ids = b
                .column_by_name("_id")
                .and_then(|c| c.as_any().downcast_ref::<Decimal128Array>())
                .expect("_id")
                .clone();
            (0..ids.len()).map(move |i| ids.value(i))
        })
        .collect()
}

/// The first column of what `sql` returns.
fn answer(conn: &infino::Connection, sql: &str) -> Vec<Option<i64>> {
    int64s(
        &conn.query_sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}")),
        0,
    )
}

/// The lines of the plans `EXPLAIN` (or `EXPLAIN ANALYZE`) gives for `sql`.
fn plan_lines(conn: &infino::Connection, explain: &str, sql: &str) -> Vec<String> {
    conn.query_sql(&format!("{explain} {sql}"))
        .unwrap_or_else(|e| panic!("{explain} {sql}: {e}"))
        .iter()
        .flat_map(|b| {
            let plans = b
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("plan column")
                .clone();
            (0..plans.len())
                .filter(|i| plans.is_valid(*i))
                .flat_map(|i| {
                    plans
                        .value(i)
                        .lines()
                        .map(str::to_owned)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// The scan's metric `name` as `EXPLAIN ANALYZE` prints it running `sql`,
/// read back from its rounded form (`3.07 K`).
fn scan_metric(conn: &infino::Connection, sql: &str, name: &str) -> f64 {
    let lines = plan_lines(conn, "EXPLAIN ANALYZE", sql);
    let line = lines
        .iter()
        .find(|line| line.contains("CachedScanExec"))
        .unwrap_or_else(|| panic!("no CachedScanExec in {lines:#?}"));
    let value = line
        .split(&format!("{name}="))
        .nth(1)
        .and_then(|rest| rest.split([',', ']']).next())
        .unwrap_or_else(|| panic!("no {name} in {line}"));
    let (number, scale) = COUNT_SUFFIXES
        .iter()
        .find_map(|&(suffix, scale)| value.strip_suffix(suffix).map(|n| (n, scale)))
        .unwrap_or((value, 1.0));
    number.parse::<f64>().expect("a count") * scale
}

/// The sorted statement `full` with `LIMIT k`, for each `k`, run twice
/// (decoding, then from kept columns): the first `k` rows `full` returns.
fn same_as_a_full_sort(conn: &infino::Connection, full: &str, ks: &[usize]) {
    let all = answer(conn, full);
    for &k in ks {
        let sql = format!("{full} LIMIT {k}");
        for run in ["first", "again"] {
            assert_eq!(answer(conn, &sql), all[..k.min(all.len())], "{run}: {sql}");
        }
    }
}

/// Points the process's configuration at a temp dir that keeps [`KEPT_BYTES`]
/// of decoded columns, before any test reads it. The configuration (and with
/// it the cache) is read once per process, so the tests in this binary share
/// this one setup and each builds its own table under its own root.
fn kept_columns_config() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // Leaked so the config file outlives the read below and every test.
        let config = Box::leak(Box::new(TempDir::new().expect("tempdir")));
        std::fs::create_dir_all(config.path().join("infino")).expect("config dir");
        std::fs::write(
            config.path().join("infino/config.yaml"),
            format!("memory:\n  decoded_column_cache_bytes: {KEPT_BYTES}\n"),
        )
        .expect("config file");
        // SAFETY: inside the `Once`, and every test calls this before it
        // touches the environment or the configuration, so no other thread
        // reads either while it is set.
        unsafe { std::env::set_var("XDG_CONFIG_HOME", config.path()) };
        assert_eq!(
            infino::config::global().memory.decoded_column_cache_bytes,
            KEPT_BYTES
        );
    });
}

#[test]
fn a_bound_from_above_skips_work_and_changes_no_answer() {
    kept_columns_config();

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");

    // `t`: a row group per commit, several per partition, written highest
    // `seq` first, so only the best-first order reads the lowest first.
    // Each row group's `seq` ascends within it. The later half of the
    // commits also hold `late`, a column the earlier row groups have no
    // statistics for.
    let partitions = std::thread::available_parallelism()
        .map_or(MIN_PARTITIONS, |n| {
            i64::try_from(n.get()).unwrap_or(MIN_PARTITIONS)
        })
        .max(MIN_PARTITIONS);
    let groups = partitions * GROUPS_PER_PARTITION;
    let total = groups * ROWS;
    let plain = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("spread", DataType::Int64, false),
    ]));
    let with_late = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("spread", DataType::Int64, false),
        Field::new("late", DataType::Int64, true),
    ]));
    let t = conn
        .create_table("t", Arc::clone(&plain), IndexSpec::new())
        .expect("create t");
    for commit in 0..groups {
        let block = groups - 1 - commit;
        let seq: Vec<i64> = (block * ROWS..(block + 1) * ROWS).collect();
        let spread: Vec<i64> = seq.iter().map(|s| s * SPREAD % total).collect();
        let mut columns: Vec<Arc<dyn Array>> = vec![
            Arc::new(Int64Array::from(seq.clone())),
            Arc::new(Int64Array::from(spread)),
        ];
        let schema = if commit * 2 < groups {
            Arc::clone(&plain)
        } else {
            columns.push(Arc::new(Int64Array::from(seq)));
            Arc::clone(&with_late)
        };
        t.append(&RecordBatch::try_new(schema, columns).expect("batch"))
            .expect("append");
    }

    for order in ["ASC", "DESC"] {
        let ks = [1, SMALL_K, 100, 15_000];
        // Values that follow the row groups: the bound settles in the first
        // row group each partition reads and skips the rest.
        same_as_a_full_sort(
            &conn,
            &format!("SELECT seq FROM t ORDER BY seq {order}"),
            &ks,
        );
        // Values spread across every row group.
        same_as_a_full_sort(
            &conn,
            &format!("SELECT spread FROM t ORDER BY spread {order}"),
            &ks,
        );
        // A column the earlier row groups have no statistics for. `NULLS
        // LAST`: they go last, and a `k` past the values reaches into their
        // nulls. `NULLS FIRST`: their nulls lead, so the best-first order
        // reads the row groups that hold them first and a small `k` is all
        // nulls.
        let reach = usize::try_from(total / 2 + 5).expect("k");
        same_as_a_full_sort(
            &conn,
            &format!("SELECT late FROM t ORDER BY late {order} NULLS LAST"),
            &[1, SMALL_K, reach],
        );
        same_as_a_full_sort(
            &conn,
            &format!("SELECT late FROM t ORDER BY late {order} NULLS FIRST"),
            &[1, SMALL_K, reach],
        );

        let sql = format!("SELECT seq FROM t ORDER BY seq {order} LIMIT {SMALL_K}");
        let plan = plan_lines(&conn, "EXPLAIN", &sql).join("\n");
        assert!(plan.contains("CachedScanExec"), "{plan}");
        // Read best first, each partition's first row group settles the
        // bound, which rules out every other row group it holds: about a
        // third of the rows are read. Read in the order written, ascending,
        // every partition reads all its row groups; without the skip, so does
        // every partition either way.
        let read = scan_metric(&conn, &sql, ROWS_READ);
        assert!(
            read * 2.0 < total as f64,
            "read {read} of {total} rows: {sql}"
        );
        let skipped = scan_metric(&conn, &sql, ROW_GROUPS_SKIPPED);
        assert!(skipped >= 1.0, "skipped {skipped} row groups: {sql}");
        // Ascending, each row group's first batch holds its lowest values,
        // and the bound they settle drops every row of its second batch.
        if order == "ASC" {
            let dropped = scan_metric(&conn, &sql, ROWS_DROPPED);
            assert!(dropped >= 1.0, "dropped {dropped} rows: {sql}");
        }
    }

    // A hash join hands its probe side a filter built from its build side's
    // rows; the scan skips the row groups it rules out, and the join answers
    // the same. With no `LIMIT`, the scan drops no rows from a batch, so the
    // filter proves it reached the scan by the row groups it skips.
    let last = total - 1;
    let join = format!(
        "SELECT t.seq FROM t JOIN (VALUES (5), (12345), ({last})) AS u(v) \
         ON t.seq = u.v ORDER BY t.seq"
    );
    assert_eq!(
        answer(&conn, &join),
        vec![Some(5), Some(12_345), Some(last)],
        "{join}"
    );
    // Three values among millions of rows: whole row groups fall outside
    // them and are skipped by their statistics.
    let skipped = scan_metric(&conn, &join, ROW_GROUPS_SKIPPED);
    assert!(skipped >= 1.0, "skipped {skipped} row groups: {join}");

    // A `MIN`/`MAX` aggregate hands down its best value so far, and the scan
    // skips the row groups whose statistics can't beat it. A filter passes its
    // rows on in batches of 8,192, so under `spread % 5 = 0`, a fifth of each
    // row group, the aggregate sets no bound until its partition's scan ends,
    // and a skip turns on another partition's bound arriving in time: those are
    // checked for their answers only.
    let kept: Vec<i64> = (0..total).filter(|s| s * SPREAD % total % 5 == 0).collect();
    for (aggregate, want) in [("MIN", kept.iter().min()), ("MAX", kept.iter().max())] {
        let sql = format!("SELECT {aggregate}(seq) FROM t WHERE spread % 5 = 0");
        assert_eq!(answer(&conn, &sql), vec![want.copied()], "{sql}");
    }
    // Under `spread % 5 <> 7`, which keeps every row but which statistics
    // cannot settle, a row group's first batch goes on whole, and each partition
    // reads its highest `seq` first, so the `MAX` that batch settles rules out
    // the partition's other row groups: the bound reached the scan.
    let max = "SELECT MAX(seq) FROM t WHERE spread % 5 <> 7";
    assert_eq!(answer(&conn, max), vec![Some(total - 1)], "{max}");
    let skipped = scan_metric(&conn, max, ROW_GROUPS_SKIPPED);
    assert!(skipped >= 1.0, "skipped {skipped} row groups: {max}");
    let both = "SELECT MIN(seq), MAX(seq) FROM t WHERE spread % 5 = 0";
    let got = conn.query_sql(both).expect("min and max");
    assert_eq!(
        (int64s(&got, 0), int64s(&got, 1)),
        (
            vec![kept.iter().min().copied()],
            vec![kept.iter().max().copied()]
        ),
        "{both}"
    );

    // `w`: `n` written as `Int32`, widened to `Int64`, written again. The
    // two generations' row groups interleave in `n`, and their statistics
    // come in both types; the best-first order sorts them as one type.
    let narrow = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, false)]));
    let wide = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
    let w = conn
        .create_table("w", Arc::clone(&narrow), IndexSpec::new())
        .expect("create w");
    for commit in 0..WIDE_COMMITS {
        let first = 2 * commit * WIDE_ROWS;
        let n: Vec<i32> = (first..first + WIDE_ROWS)
            .map(|v| i32::try_from(v).expect("fits"))
            .collect();
        w.append(
            &RecordBatch::try_new(Arc::clone(&narrow), vec![Arc::new(Int32Array::from(n))])
                .expect("batch"),
        )
        .expect("append");
    }
    conn.apply_schema(
        "w",
        &SchemaPatch::new(vec![FieldPatch::named("n").with_type(DataType::Int64)]),
        None,
    )
    .expect("widen n");
    let w = conn.open_table("w").expect("open w");
    for commit in 0..WIDE_COMMITS {
        let first = (2 * commit + 1) * WIDE_ROWS;
        let n: Vec<i64> = (first..first + WIDE_ROWS).collect();
        w.append(
            &RecordBatch::try_new(Arc::clone(&wide), vec![Arc::new(Int64Array::from(n))])
                .expect("batch"),
        )
        .expect("append");
    }
    let rows = 2 * WIDE_COMMITS * WIDE_ROWS;
    for order in ["ASC", "DESC"] {
        let full = format!("SELECT n FROM w ORDER BY n {order}");
        let all = answer(&conn, &full);
        assert_eq!(all.len(), usize::try_from(rows).expect("rows"), "{full}");
        same_as_a_full_sort(&conn, &full, &[1, SMALL_K, 250]);
    }
}

#[test]
fn a_top_k_over_kept_columns_answers_as_a_full_sort() {
    /// Rows in each of the table's commits.
    const ROWS: i64 = 3_000;
    /// Commits, each a superfile.
    const COMMITS: i64 = 3;

    kept_columns_config();

    let root = TempDir::new().expect("tempdir");
    let conn = connect(format!("file://{}", root.path().display())).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("note", DataType::LargeUtf8, true),
    ]));
    let t = conn
        .create_table("t", Arc::clone(&schema), IndexSpec::new())
        .expect("create t");
    for commit in 0..COMMITS {
        let seq: Vec<i64> = (commit * ROWS..(commit + 1) * ROWS).collect();
        let note: Vec<Option<&str>> = seq.iter().map(|_| None).collect();
        t.append(
            &RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(seq)),
                    Arc::new(LargeStringArray::from(note)),
                ],
            )
            .expect("batch"),
        )
        .expect("append");
    }
    t.delete(col("seq").lt(lit(30_i64))).expect("delete");

    let every = {
        let mut all = ids(&conn
            .query_sql("SELECT _id FROM t WHERE seq % 7 = 0")
            .expect("sql"));
        all.sort_unstable();
        all
    };
    for k in [1_usize, 10, 500, 5_000] {
        let mut want = every.clone();
        want.truncate(k);
        for keys in ["_id ASC", "note ASC NULLS LAST, _id ASC"] {
            let sql = format!(
                "SELECT seq, \"_id\" AS \"_id\" FROM t WHERE seq % 7 = 0 ORDER BY {keys} LIMIT {k}"
            );
            for run in ["decoding", "from kept columns"] {
                let got = ids(&conn
                    .query_sql(&sql)
                    .unwrap_or_else(|e| panic!("{run}: {e}")));
                assert_eq!(got, want, "{run}: {sql}");
            }
            let plan = plan_lines(&conn, "EXPLAIN", &sql).join("\n");
            assert!(plan.contains("CachedScanExec"), "{plan}");
        }
    }

    // A top-k on another column reads row groups best first and skips the
    // ones its bound rules out as it tightens: the same rows as a full sort.
    let seqs = |batches: &[RecordBatch]| -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| {
                let c = b
                    .column_by_name("seq")
                    .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                    .expect("seq")
                    .clone();
                (0..c.len()).map(move |i| c.value(i))
            })
            .collect()
    };
    let mut kept: Vec<i64> = (30..COMMITS * ROWS).filter(|s| s % 7 == 0).collect();
    for (order, descending) in [("ASC", false), ("DESC", true)] {
        if descending {
            kept.reverse();
        }
        for k in [1_usize, 10, 500, 5_000] {
            let sql = format!("SELECT seq FROM t WHERE seq % 7 = 0 ORDER BY seq {order} LIMIT {k}");
            for run in ["first", "again"] {
                let got = seqs(
                    &conn
                        .query_sql(&sql)
                        .unwrap_or_else(|e| panic!("{run}: {e}")),
                );
                assert_eq!(got, kept[..k.min(kept.len())], "{run}: {sql}");
            }
        }
    }
}
