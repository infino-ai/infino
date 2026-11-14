// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Range index end to end, through the public API: a table that
//! range-indexes its numeric columns answers every range shape exactly as
//! the same rows in an unindexed table do — across several superfiles,
//! after a delete, after compaction, and after the connection is reopened.
//!
//! The unindexed table is the oracle: it answers by scanning. Every shape
//! the range index serves is here — a count (answered from the index alone),
//! a bucketed count (`COUNT(*) FILTER (WHERE range)` per bucket, likewise),
//! a range's rows, `BETWEEN`, two conjuncts folded into one interval, an
//! `ORDER BY ... LIMIT` the index serves in order, integer, float and date
//! columns — beside shapes it must leave to the scan: two columns, an `OR`.
//! A range-indexed column is found by its id, so a rename keeps its index
//! and a name dropped and reused for a new column does not inherit it.

use std::sync::Arc;

use arrow::{compute::cast, util::pretty::pretty_format_batches};
use arrow_array::{
    Array, Date32Array, Float64Array, Int32Array, Int64Array, LargeStringArray, RecordBatch,
    TimestampNanosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use datafusion::{
    logical_expr::LogicalPlanBuilder,
    prelude::{col, in_subquery, lit},
};
use infino::{
    ColumnIndex, Connection, FieldPatch, IndexSpec, OptimizeOptions, SchemaPatch, connect,
};
use tempfile::TempDir;

/// The `seq` values a subquery predicate lists: some deleted, one past the
/// last row.
const SUBQUERY_SEQS: [i64; 4] = [3, 777, 4321, 999_999];
/// The name DataFusion gives a `VALUES` list's one column.
const VALUES_COLUMN: &str = "column1";
/// Rows in each committed superfile.
const ROWS_PER_FILE: i64 = 3000;
/// Superfiles each table is written as, one commit each.
const FILES: i64 = 4;
/// Superfiles of one row each in the one-row fixture.
const ONE_ROW_FILES: usize = 3;
/// Every this-many rows `qty` is null.
const QTY_NULL_EVERY: i64 = 17;
/// The first `day`: 2022-01-08 as days since the epoch.
const FIRST_DAY: i32 = 19_000;
/// Distinct days the rows spread over.
const DAYS: i64 = 365;
/// Rows the delete removes: those with `seq` below this.
const DELETED_BELOW: i64 = 500;
/// The physical operator a scan's rows pass through when a filter is
/// checked on them; a count the range index answers has none.
const FILTER_OPERATOR: &str = "FilterExec";

/// The shapes both tables answer, `{t}` standing for the table. Each is
/// ordered so its rows compare as text, and a top-k selects only the sort
/// column, as rows tied on it may come back in either order — unless a
/// later key (`seq`, unique) breaks the ties, which then must pick the same
/// rows from among them.
const QUERIES: &[&str] = &[
    "SELECT count(*) AS n FROM {t} WHERE qty > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty >= -10 AND qty < 10",
    "SELECT count(*) AS n FROM {t} WHERE qty BETWEEN -5 AND 5",
    "SELECT count(*) AS n FROM {t} WHERE qty = 7",
    "SELECT count(*) AS n FROM {t} WHERE qty < -1000",
    "SELECT count(*) AS n FROM {t} WHERE 100 < qty",
    "SELECT count(*) AS n FROM {t} WHERE seq >= 1000 AND seq < 1100",
    "SELECT count(*) AS n FROM {t} WHERE price > 990.5",
    "SELECT seq FROM {t} WHERE price > 990.5 ORDER BY seq",
    "SELECT seq FROM {t} WHERE price BETWEEN 100 AND 101 ORDER BY seq",
    "SELECT seq, qty FROM {t} WHERE qty = 0 ORDER BY seq",
    "SELECT qty FROM {t} WHERE qty > 200 ORDER BY qty LIMIT 10",
    "SELECT qty FROM {t} WHERE qty < -200 ORDER BY qty DESC LIMIT 10",
    "SELECT price FROM {t} WHERE price < 50 ORDER BY price DESC LIMIT 5",
    "SELECT qty FROM {t} ORDER BY qty LIMIT 3",
    "SELECT seq FROM {t} WHERE day >= DATE '2022-06-01' AND day < DATE '2022-06-03' ORDER BY seq",
    "SELECT sum(seq) AS s FROM {t} WHERE qty > 0 AND price > 500",
    "SELECT count(*) AS n FROM {t} WHERE qty > 240 OR price < 1",
    "SELECT seq FROM {t} WHERE qty > 240 AND seq < 3000 ORDER BY seq",
    BUCKETED_QTY,
    "SELECT count(*) FILTER (WHERE qty < 0) AS low, count(*) FILTER (WHERE qty >= 0 AND qty < 200) AS high FROM {t} WHERE qty > -200",
    "SELECT count(*) AS n, count(*) FILTER (WHERE qty > 100) AS big FROM {t} WHERE qty > 0",
    BUCKETED_DAY,
    BUCKETED_WITH_TOTAL,
    "SELECT count(*) FILTER (WHERE price < 250) AS a, count(*) FILTER (WHERE price >= 250 AND price < 750.5) AS b, count(*) FILTER (WHERE price >= 750.5) AS c FROM {t}",
    "SELECT count(*) FILTER (WHERE qty > 0) AS a, count(*) FILTER (WHERE price > 500) AS b FROM {t} WHERE seq < 6000",
    "SELECT count(*) FILTER (WHERE qty > 0) AS a, count(*) FILTER (WHERE qty > 100 OR qty < -100) AS b FROM {t}",
    TIE_BROKEN_TOP_K,
    "SELECT seq, qty FROM {t} WHERE qty < -200 ORDER BY qty DESC, seq DESC LIMIT 7",
    "SELECT seq, price AS price FROM {t} WHERE price > 900 ORDER BY price ASC NULLS LAST, seq ASC NULLS LAST LIMIT 10",
    "SELECT seq, qty FROM {t} WHERE qty BETWEEN 240 AND 249 ORDER BY qty DESC, seq LIMIT 30",
];

/// A top-k whose sort column ties on many rows (`qty` repeats every 500
/// rows), the ties broken by `seq`: a paged search's sort.
const TIE_BROKEN_TOP_K: &str =
    "SELECT seq, qty FROM {t} WHERE qty > 200 ORDER BY qty, seq LIMIT 10";
/// Most rows a tie-broken top-k over the range index may read: the first
/// `qty` above 200 holds the 10 answer rows and a few dozen ties.
const TIE_BROKEN_TOP_K_MAX_ROWS: f64 = 100.0;
/// Fewest rows a scan of `qty > 200` reads with no index: every row the
/// range holds (49 values, about 22 live rows each).
const TIE_BROKEN_TOP_K_SCAN_MIN_ROWS: f64 = 500.0;
/// Where an analyzed plan names the scan, and how it counts what it read.
const SCAN_OPERATOR: &str = "DataSourceExec";
const OUTPUT_ROWS: &str = "output_rows=";

/// A bucketed range aggregation in one statement, its buckets overlapping
/// as a range aggregation's may.
const BUCKETED_QTY: &str = "SELECT count(*) FILTER (WHERE qty < -100) AS a, count(*) FILTER (WHERE qty >= -100 AND qty < 0) AS b, count(*) FILTER (WHERE qty >= 0 AND qty < 100) AS c, count(*) FILTER (WHERE qty >= 50) AS d FROM {t}";
/// Date buckets, quarter by quarter.
const BUCKETED_DAY: &str = "SELECT count(*) FILTER (WHERE day < DATE '2022-04-01') AS q1, count(*) FILTER (WHERE day >= DATE '2022-04-01' AND day < DATE '2022-07-01') AS q2, count(*) FILTER (WHERE day >= DATE '2022-07-01') AS rest FROM {t}";
/// Buckets beside the whole table's count, under a `WHERE` that holds every
/// row: a search request's range aggregation and its hit total together.
const BUCKETED_WITH_TOTAL: &str = "SELECT count(*) AS total, count(*) FILTER (WHERE price < 250) AS a, count(*) FILTER (WHERE price >= 250 AND price < 500) AS b, count(*) FILTER (WHERE price >= 500) AS c FROM {t} WHERE 1=1";
/// How a plan shows an aggregate that checks its own filter on each row; a
/// bucketed count the range index answers has none.
const BUCKET_FILTER: &str = "FILTER (WHERE";

/// The shapes both tables answer once `qty` is renamed `amount` and a new
/// `qty` is added: the range index follows `amount`, and the new `qty`
/// holds only the rows written after it.
const RENAMED_QUERIES: &[&str] = &[
    "SELECT count(*) AS n FROM {t} WHERE amount > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty > 100",
    "SELECT count(*) AS n FROM {t} WHERE qty BETWEEN -5 AND 5",
    "SELECT seq FROM {t} WHERE qty BETWEEN -5 AND 5 ORDER BY seq",
    "SELECT amount FROM {t} WHERE amount > 200 ORDER BY amount LIMIT 10",
    "SELECT qty FROM {t} WHERE qty < -150 ORDER BY qty LIMIT 10",
];

/// The shapes both tables answer once `day` is retyped from days to
/// nanoseconds and `qty` widened to `Int64`. A superfile written before
/// keys `day` by days, which a nanosecond bound must not be read against.
const RETYPED_QUERIES: &[&str] = &[
    "SELECT count(*) AS n FROM {t} WHERE day >= TIMESTAMP '2022-06-01 00:00:00'",
    "SELECT seq FROM {t} WHERE day = TIMESTAMP '2022-01-08 00:00:00' ORDER BY seq",
    "SELECT seq FROM {t} WHERE day >= TIMESTAMP '2022-06-01 00:00:00' AND day < TIMESTAMP '2022-06-03 00:00:00' ORDER BY seq",
    "SELECT day FROM {t} WHERE day > TIMESTAMP '2022-03-01 00:00:00' ORDER BY day LIMIT 10",
    "SELECT day FROM {t} ORDER BY day DESC LIMIT 5",
    "SELECT count(*) AS n FROM {t} WHERE qty > 100",
    "SELECT qty FROM {t} WHERE qty < -200 ORDER BY qty DESC LIMIT 10",
];
/// Nanoseconds in a day.
const NANOS_PER_DAY: i64 = 86_400_000_000_000;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("qty", DataType::Int32, true),
        Field::new("price", DataType::Float64, false),
        Field::new("day", DataType::Date32, false),
    ]))
}

/// Rows `first..first + ROWS_PER_FILE`: `qty` spread over `-250..250` with
/// nulls, `price` over `0..1000`, `day` over a year — none of them in the
/// order the rows arrive.
fn rows(first: i64) -> RecordBatch {
    let seq: Vec<i64> = (first..first + ROWS_PER_FILE).collect();
    let qty: Vec<Option<i32>> = seq
        .iter()
        .map(|i| (i % QTY_NULL_EVERY != 0).then(|| ((i * 37) % 500 - 250) as i32))
        .collect();
    let price: Vec<f64> = seq
        .iter()
        .map(|i| ((i * 7919) % 10_000) as f64 / 10.0)
        .collect();
    let day: Vec<i32> = seq
        .iter()
        .map(|i| FIRST_DAY + ((i * 13) % DAYS) as i32)
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from(seq)),
            Arc::new(Int32Array::from(qty)),
            Arc::new(Float64Array::from(price)),
            Arc::new(Date32Array::from(day)),
        ],
    )
    .expect("batch")
}

/// The two tables, each written as [`FILES`] superfiles of the same rows.
fn create(db: &Connection) {
    let plain = db
        .create_table("plain", schema(), IndexSpec::new())
        .expect("create plain");
    let ranged = db
        .create_table(
            "ranged",
            schema(),
            IndexSpec::new()
                .range("seq")
                .range("qty")
                .range("price")
                .range("day"),
        )
        .expect("create ranged");
    for file in 0..FILES {
        let batch = rows(file * ROWS_PER_FILE);
        plain.append(&batch).expect("append plain");
        ranged.append(&batch).expect("append ranged");
    }
}

/// `query` on `table`, as text.
fn answer(db: &Connection, query: &str, table: &str) -> String {
    let batches = db
        .query_sql(&query.replace("{t}", table))
        .unwrap_or_else(|e| panic!("{query} on {table}: {e}"));
    pretty_format_batches(&batches).expect("format").to_string()
}

/// The rows the scan of an analyzed plan read (its `output_rows`, which
/// DataFusion prints as `24` or `1.00 K`).
fn scan_rows(plan: &str) -> f64 {
    let scan = &plan[plan.find(SCAN_OPERATOR).expect("the plan has a scan")..];
    let count =
        &scan[scan.find(OUTPUT_ROWS).expect("the scan counts its rows") + OUTPUT_ROWS.len()..];
    let count = &count[..count.find(',').expect("the count ends")];
    let (number, scale) = match count.split_once(' ') {
        Some((number, "K")) => (number, 1e3),
        Some((number, "M")) => (number, 1e6),
        _ => (count, 1.0),
    };
    number.parse::<f64>().expect("a row count") * scale
}

/// Every one of `queries`' answers from both tables, the same.
fn assert_same_answers(db: &Connection, queries: &[&str], when: &str) {
    for query in queries {
        assert_eq!(
            answer(db, query, "ranged"),
            answer(db, query, "plain"),
            "{when}: {query}"
        );
    }
    assert_same_counts(db, when);
}

/// `count_where` on both tables, and the same counts as SQL `COUNT(*) FILTER`
/// statements on the plain table: ranges the index counts, a `true`, a
/// conjunct that simplifies away, and shapes it must leave to the aggregate.
fn assert_same_counts(db: &Connection, when: &str) {
    let predicates = [
        (lit(true), "true"),
        (col("qty").gt(lit(100)), "qty > 100"),
        (
            lit(1).eq(lit(1)).and(col("price").gt(lit(500.0))),
            "1 = 1 AND price > 500.0",
        ),
        (
            col("price")
                .gt_eq(lit(250))
                .and(col("price").lt(lit(750.5))),
            "price >= 250 AND price < 750.5",
        ),
        (col("seq").lt(lit(1000i64)), "seq < 1000"),
        (
            col("qty").gt(lit(240)).or(col("price").lt(lit(1.0))),
            "qty > 240 OR price < 1.0",
        ),
        (
            col("qty").gt(lit(0)).and(col("price").gt(lit(500.0))),
            "qty > 0 AND price > 500.0",
        ),
    ];
    let exprs: Vec<_> = predicates.iter().map(|(expr, _)| expr.clone()).collect();
    let sql = format!(
        "SELECT {} FROM plain",
        predicates
            .iter()
            .map(|(_, sql)| format!("count(*) FILTER (WHERE {sql})"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let batches = db.query_sql(&sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let expected: Vec<u64> = (0..predicates.len())
        .map(|i| {
            let column = batches[0].column(i);
            let counts = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("an Int64 count");
            u64::try_from(counts.value(0)).expect("a count")
        })
        .collect();
    for table in ["ranged", "plain"] {
        let counts = db
            .open_table(table)
            .expect("open")
            .count_where(&exprs)
            .unwrap_or_else(|e| panic!("{when}: count_where on {table}: {e}"));
        assert_eq!(counts, expected, "{when}: count_where on {table}");
    }

    // A predicate that reads another relation, which no FILTER clause can
    // hold, beside one the index counts.
    // A subquery's plan ends in the projection of its one column, as SQL
    // plans `seq IN (SELECT ...)`.
    let listed =
        LogicalPlanBuilder::values(SUBQUERY_SEQS.iter().map(|seq| vec![lit(*seq)]).collect())
            .and_then(|values| values.project(vec![col(VALUES_COLUMN)]))
            .and_then(|values| values.build())
            .expect("values");
    let with_subquery = [
        in_subquery(col("seq"), Arc::new(listed)),
        col("qty").gt(lit(100)),
    ];
    let listed_sql = SUBQUERY_SEQS
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT count(*) FILTER (WHERE seq IN ({listed_sql})), count(*) FILTER (WHERE qty > 100) FROM plain"
    );
    let batches = db.query_sql(&sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    let expected: Vec<u64> = (0..with_subquery.len())
        .map(|i| {
            let counts = batches[0]
                .column(i)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("an Int64 count");
            u64::try_from(counts.value(0)).expect("a count")
        })
        .collect();
    for table in ["ranged", "plain"] {
        let counts = db
            .open_table(table)
            .expect("open")
            .count_where(&with_subquery)
            .unwrap_or_else(|e| panic!("{when}: count_where with a subquery on {table}: {e}"));
        assert_eq!(
            counts, expected,
            "{when}: count_where with a subquery on {table}"
        );
    }
}

#[test]
fn a_range_indexed_table_answers_as_a_scan_does_through_deletes_compaction_and_reopen() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_str().expect("utf-8 path");
    let db = connect(path).expect("connect");
    create(&db);
    assert_same_answers(&db, QUERIES, "after load");

    // A count over a range-indexed column reads no row to check the range,
    // on an integer and on a float column alike. The integer is `seq`, whose
    // file holds a distinct value per row: `qty`'s 500 values fit the exact
    // value table, so the plain table counts a range of them from its
    // statistics too, and only a column past that table shows the scan the
    // index saves.
    for query in [
        "EXPLAIN SELECT count(*) FROM {t} WHERE seq >= 100 AND seq < 2000",
        "EXPLAIN SELECT count(*) FROM {t} WHERE price > 990.5",
    ] {
        assert!(
            answer(&db, query, "plain").contains(FILTER_OPERATOR),
            "{query}"
        );
        assert!(
            !answer(&db, query, "ranged").contains(FILTER_OPERATOR),
            "{query}"
        );
    }
    // Nor does a bucketed one check any bucket's range on a row.
    for bucketed in [BUCKETED_QTY, BUCKETED_DAY, BUCKETED_WITH_TOTAL] {
        let query = format!("EXPLAIN {bucketed}");
        assert!(
            answer(&db, &query, "plain").contains(BUCKET_FILTER),
            "{query}"
        );
        let ranged = answer(&db, &query, "ranged");
        assert!(
            !ranged.contains(BUCKET_FILTER) && !ranged.contains(FILTER_OPERATOR),
            "{query}: {ranged}"
        );
    }
    // A top-k broken by a later key reads the rows up to the cutoff from the
    // range index, ties included, not every row of the range.
    let query = format!("EXPLAIN ANALYZE {TIE_BROKEN_TOP_K}");
    let plain_rows = scan_rows(&answer(&db, &query, "plain"));
    let ranged_rows = scan_rows(&answer(&db, &query, "ranged"));
    assert!(
        plain_rows >= TIE_BROKEN_TOP_K_SCAN_MIN_ROWS,
        "{query}: the plain scan read {plain_rows} rows"
    );
    assert!(
        ranged_rows <= TIE_BROKEN_TOP_K_MAX_ROWS,
        "{query}: the indexed scan read {ranged_rows} rows"
    );

    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .delete(col("seq").lt(lit(DELETED_BELOW)))
            .expect("delete");
    }
    assert_same_answers(&db, QUERIES, "after a delete");

    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .optimize(&OptimizeOptions::default())
            .expect("optimize");
    }
    assert_same_answers(&db, QUERIES, "after compaction");

    drop(db);
    let reopened = connect(path).expect("reconnect");
    assert_same_answers(&reopened, QUERIES, "after a reopen");
}

#[test]
fn a_range_index_follows_its_column_through_a_rename_and_not_into_a_reused_name() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    create(&db);
    for table in ["plain", "ranged"] {
        let before = db
            .apply_schema(table, &SchemaPatch::new(vec![]), None)
            .expect("read the schema");
        let qty = before.id_of("qty").expect("qty");
        db.apply_schema(
            table,
            &SchemaPatch::new(vec![FieldPatch::named("amount").with_id(qty)]),
            None,
        )
        .expect("rename qty");
        let mut new_qty = FieldPatch::named("qty").with_type(DataType::Int32);
        if table == "ranged" {
            new_qty = new_qty.with_index(ColumnIndex::Range);
        }
        let after = db
            .apply_schema(table, &SchemaPatch::new(vec![new_qty]), None)
            .expect("add a new qty");
        assert_ne!(after.id_of("qty"), Some(qty), "a new column takes a new id");
    }
    // One more superfile, with values in both columns.
    let first = FILES * ROWS_PER_FILE;
    let old = rows(first);
    let new_qty: Int32Array = (first..first + ROWS_PER_FILE)
        .map(|i| Some(((i * 11) % 400 - 200) as i32))
        .collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("amount", DataType::Int32, true),
        Field::new("price", DataType::Float64, false),
        Field::new("day", DataType::Date32, false),
        Field::new("qty", DataType::Int32, true),
    ]));
    let mut columns = old.columns().to_vec();
    columns.push(Arc::new(new_qty) as Arc<dyn Array>);
    let batch = RecordBatch::try_new(schema, columns).expect("batch");
    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .append(&batch)
            .expect("append");
    }
    assert_same_answers(&db, RENAMED_QUERIES, "after a rename and a reused name");
}

#[test]
fn a_range_index_written_before_a_retype_answers_only_where_its_keys_still_hold() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    create(&db);
    let retype = SchemaPatch::new(vec![
        FieldPatch::named("day").with_type(DataType::Timestamp(TimeUnit::Nanosecond, None)),
        FieldPatch::named("qty").with_type(DataType::Int64),
    ]);
    for table in ["plain", "ranged"] {
        db.apply_schema(table, &retype, None).expect("retype");
    }
    assert_same_answers(&db, RETYPED_QUERIES, "after a retype");
    // A wider integer keeps its keys, so the old index still serves `qty`.
    let query = format!("EXPLAIN ANALYZE {TIE_BROKEN_TOP_K}");
    let ranged_rows = scan_rows(&answer(&db, &query, "ranged"));
    assert!(
        ranged_rows <= TIE_BROKEN_TOP_K_MAX_ROWS,
        "{query}: the indexed scan read {ranged_rows} rows"
    );

    // One more superfile, written under the new types.
    let old = rows(FILES * ROWS_PER_FILE);
    let schema = Arc::new(Schema::new(vec![
        Field::new("seq", DataType::Int64, false),
        Field::new("qty", DataType::Int64, true),
        Field::new("price", DataType::Float64, false),
        Field::new(
            "day",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ),
    ]));
    let qty = cast(old.column(1), &DataType::Int64).expect("qty");
    let days = old
        .column(3)
        .as_any()
        .downcast_ref::<Date32Array>()
        .expect("days");
    let day: TimestampNanosecondArray = days
        .values()
        .iter()
        .map(|d| Some(i64::from(*d) * NANOS_PER_DAY))
        .collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::clone(old.column(0)),
            qty,
            Arc::clone(old.column(2)),
            Arc::new(day),
        ],
    )
    .expect("batch");
    for table in ["plain", "ranged"] {
        db.open_table(table)
            .expect("open")
            .append(&batch)
            .expect("append");
    }
    assert_same_answers(&db, RETYPED_QUERIES, "after a retype and an append");
}

/// A search function's `WHERE` on a retyped range-indexed column answers
/// as it does with no range index: a superfile whose index no longer keys
/// the column's values is left to the filter, not reported as a planner
/// bug. Covers a change of key unit and of key kind.
#[test]
fn a_search_where_on_a_retyped_range_column_answers_as_without_the_index() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("day", DataType::Date32, false),
        Field::new("qty", DataType::Int64, false),
    ]));
    let n = 50;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(LargeStringArray::from(
                (0..n).map(|i| format!("rust {i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Date32Array::from(
                (0..n).map(|i| FIRST_DAY + i % 5).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..n).map(|i| i64::from(i % 7)).collect::<Vec<_>>(),
            )),
        ],
    )
    .expect("batch");
    for (table, spec) in [
        ("plain", IndexSpec::new().fts("title")),
        (
            "ranged",
            IndexSpec::new().fts("title").range("day").range("qty"),
        ),
    ] {
        db.create_table(table, Arc::clone(&schema), spec)
            .expect("create")
            .append(&batch)
            .expect("append");
        db.apply_schema(
            table,
            &SchemaPatch::new(vec![
                FieldPatch::named("day").with_type(DataType::Timestamp(TimeUnit::Nanosecond, None)),
                FieldPatch::named("qty").with_type(DataType::Float64),
            ]),
            None,
        )
        .expect("retype");
    }
    let search = "SELECT title FROM bm25_search('{t}', 'title', 'rust', 100)";
    let queries = [
        format!("{search} WHERE day = TIMESTAMP '2022-01-08 00:00:00' ORDER BY title"),
        format!("{search} WHERE qty = 3.0 ORDER BY title"),
    ];
    for query in &queries {
        let plain = answer(&db, query, "plain");
        assert!(
            plain.matches("rust").count() > 1,
            "premise: {query}: {plain}"
        );
        assert_eq!(answer(&db, query, "ranged"), plain, "{query}");
    }
}

/// Superfiles of a single row each, as a writer committing one document at
/// a time leaves them, answer as the scan does.
#[test]
fn one_row_superfiles_answer_as_a_scan_does() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let plain = db
        .create_table("plain", schema(), IndexSpec::new())
        .expect("create plain");
    let ranged = db
        .create_table(
            "ranged",
            schema(),
            IndexSpec::new()
                .range("seq")
                .range("qty")
                .range("price")
                .range("day"),
        )
        .expect("create ranged");
    let batch = rows(0);
    for row in 0..ONE_ROW_FILES {
        let one = batch.slice(row, 1);
        plain.append(&one).expect("append plain");
        ranged.append(&one).expect("append ranged");
    }
    assert_same_answers(&db, QUERIES, "over one-row superfiles");
}

#[test]
fn a_range_index_on_a_column_it_cannot_key_is_refused() {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::LargeUtf8, false),
        Field::new("n", DataType::Int64, false),
    ]));
    for spec in [
        IndexSpec::new().range("name"),
        IndexSpec::new().range("missing"),
        IndexSpec::new().range("n").range("n"),
        IndexSpec::new().fts("name").range("name"),
    ] {
        assert!(
            db.create_table("t", Arc::clone(&schema), spec).is_err(),
            "the table must not be created"
        );
    }
    db.create_table("t", schema, IndexSpec::new().range("n"))
        .expect("an integer column takes a range index");
}
