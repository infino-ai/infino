// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Aggregates the manifest statistics answer, on a table whose superfiles
//! carry deleted rows.
//!
//! A table kept current by updates deletes the old version of each changed
//! row and appends the new one, so its superfiles collect deleted rows. This
//! diagnostic writes a wide table of many superfiles, then updates it in
//! two stages: first a few superfiles carry deletes, then most do. After
//! each stage it times grouped counts (`GROUP BY column, COUNT(*)`, also
//! filtered to the column's non-null values, and the `value_counts` call)
//! and unfiltered `COUNT`/`MIN`/`MAX` and `SUM`/`AVG` over many columns,
//! the first run on a fresh connection and the median of warm runs.
//!
//! Each update deletes a run of rows that were appended together, as an
//! update of a batch of source records does, and appends their new
//! versions as one more commit. Each stage is labeled with how many of the
//! table's superfiles carry deletes, read from its manifest.
//!
//! A last stage replaces a quarter of the rows of some commits at once, then
//! runs `optimize` with a target below every superfile's size, so each is
//! already at target: what compaction reclaims of superfiles that are full
//! shows in how many still carry deletes, and in the times after it.
//!
//! Invoked as `cargo bench -- stats-deletes`. Knobs:
//! `INFINO_BENCH_STATS_DELETES_ROWS` (default 2 000 000),
//! `INFINO_BENCH_STATS_DELETES_COLUMNS` (default 200),
//! `INFINO_DIAG_ITERS` (warm runs per query, default 7).

use std::{
    hint::black_box,
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{ArrayRef, Float64Array, Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    logical_expr::utils::disjunction,
    prelude::{col, lit},
};
use infino::{CompactionSettings, Connection, IndexSpec, OptimizeOptions, connect};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

use crate::{
    diag_common::{env_usize, percentile},
    markdown::fmt_time,
    report::{Better, Block, Cell, Report, Section, metric, text},
};

/// Rows in the table before the updates.
const DEFAULT_ROWS: usize = 2_000_000;
/// Value columns, besides the key.
const DEFAULT_COLUMNS: usize = 200;
/// Rows per commit: one superfile each.
const ROWS_PER_COMMIT: usize = 50_000;
/// Rows each update replaces in one superfile, as one run.
const ROWS_PER_UPDATE: usize = 500;
/// Commits the first stage updates: one in this many.
const FEW_EVERY: usize = 10;
/// Commits the second stage updates, besides the first stage's: all but one
/// in this many.
const MOST_SKIP_EVERY: usize = 10;
/// Commits the last stage replaces a quarter of: one in this many.
const HEAVY_EVERY: usize = 5;
/// Of a commit, the share the last stage's updates replace: one row in
/// this many.
const HEAVY_SHARE: usize = 4;
/// The compaction target of the last stage's optimize, in MiB: below
/// every superfile's size.
const OPTIMIZE_TARGET_MB: u64 = 1;
/// Distinct values of a low-cardinality column.
const LOW_CARDINALITY: i64 = 12;
/// The values of an integer of many values.
const AMOUNT_RANGE: Range<i64> = -1_000_000..1_000_000;
/// A float's values are this many steps of [`RATIO_STEP`].
const RATIO_STEPS: i64 = 1_000;
const RATIO_STEP: f64 = 0.125;
/// Fewest value columns: one of each kind.
const MIN_COLUMNS: usize = 4;
/// Share of a value column's rows that are null.
const NULL_SHARE: f64 = 0.3;
/// Columns each unfiltered `COUNT`/`MIN`/`MAX` statement reads.
const BOUND_COLUMNS: usize = 50;
/// Timed warm runs per query; override with `INFINO_DIAG_ITERS`.
const DEFAULT_ITERS: usize = 7;
const NS_PER_SEC: f64 = 1e9;
/// Median.
const P50: usize = 50;
/// The table's name.
const TABLE: &str = "updated";
/// The seed of the generated rows.
const SEED: u64 = 0x5747;

/// What a value column holds, by its position.
#[derive(Clone, Copy)]
enum Kind {
    /// A low-cardinality string, counted by value in its statistics.
    Label,
    /// A low-cardinality integer, counted by value too.
    Code,
    /// An integer of many values.
    Amount,
    /// A float.
    Ratio,
}

fn kind(column: usize) -> Kind {
    [Kind::Label, Kind::Code, Kind::Amount, Kind::Ratio][column % 4]
}

fn column_name(column: usize) -> String {
    let prefix = match kind(column) {
        Kind::Label => "label",
        Kind::Code => "code",
        Kind::Amount => "amount",
        Kind::Ratio => "ratio",
    };
    format!("{prefix}_{column}")
}

fn schema(columns: usize) -> SchemaRef {
    let mut fields = vec![Field::new("k", DataType::Int64, false)];
    fields.extend((0..columns).map(|c| {
        let data_type = match kind(c) {
            Kind::Label => DataType::LargeUtf8,
            Kind::Code | Kind::Amount => DataType::Int64,
            Kind::Ratio => DataType::Float64,
        };
        Field::new(column_name(c), data_type, true)
    }));
    Arc::new(Schema::new(fields))
}

/// Rows with keys `keys`, values drawn from `rng`.
fn rows(schema: &SchemaRef, columns: usize, keys: &[i64], rng: &mut StdRng) -> RecordBatch {
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(keys.to_vec()))];
    for c in 0..columns {
        // Each row's value, or `None` for a null, drawn by `value`.
        let mut draw = |value: &mut dyn FnMut(&mut StdRng) -> i64| -> Vec<Option<i64>> {
            keys.iter()
                .map(|_| (!rng.random_bool(NULL_SHARE)).then(|| value(rng)))
                .collect()
        };
        let array: ArrayRef = match kind(c) {
            Kind::Label => Arc::new(LargeStringArray::from_iter(
                draw(&mut |rng| rng.random_range(0..LOW_CARDINALITY))
                    .into_iter()
                    .map(|v| v.map(|v| format!("v{v}"))),
            )),
            Kind::Code => Arc::new(Int64Array::from(draw(&mut |rng| {
                rng.random_range(0..LOW_CARDINALITY)
            }))),
            Kind::Amount => Arc::new(Int64Array::from(draw(&mut |rng| {
                rng.random_range(AMOUNT_RANGE)
            }))),
            Kind::Ratio => Arc::new(Float64Array::from_iter(
                draw(&mut |rng| rng.random_range(0..RATIO_STEPS))
                    .into_iter()
                    .map(|v| v.map(|v| v as f64 * RATIO_STEP)),
            )),
        };
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::clone(schema), arrays).expect("batch")
}

/// The statements timed, by label.
fn queries(columns: usize) -> Vec<(String, String)> {
    let first = |k: fn(usize) -> bool| (0..columns).find(|c| k(*c)).expect("a column of the kind");
    let label = column_name(first(|c| matches!(kind(c), Kind::Label)));
    let code = column_name(first(|c| matches!(kind(c), Kind::Code)));
    let bounds = (0..columns.min(BOUND_COLUMNS))
        .map(|c| {
            let name = column_name(c);
            format!("COUNT({name}), MIN({name}), MAX({name})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let sums = (0..columns.min(BOUND_COLUMNS))
        .filter(|c| matches!(kind(*c), Kind::Code | Kind::Amount))
        .map(|c| {
            let name = column_name(c);
            format!("SUM({name}), AVG({name})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    vec![
        (
            format!("GROUP BY {label}"),
            format!("SELECT {label}, COUNT(*) FROM {TABLE} GROUP BY {label}"),
        ),
        (
            format!("GROUP BY {code} (non-null)"),
            format!(
                "SELECT {code}, COUNT(*) FROM {TABLE} WHERE {code} IS NOT NULL GROUP BY {code}"
            ),
        ),
        (
            format!("COUNT/MIN/MAX x{}", columns.min(BOUND_COLUMNS)),
            format!("SELECT {bounds} FROM {TABLE}"),
        ),
        (
            "SUM/AVG of integers".to_string(),
            format!("SELECT {sums} FROM {TABLE}"),
        ),
    ]
}

/// The first run of `run` on a fresh connection to `uri`, and the median of
/// `iters` warm runs on it.
fn time(uri: &str, iters: usize, run: impl Fn(&Connection)) -> (Duration, Duration) {
    let conn = connect(uri).expect("connect");
    let started = Instant::now();
    run(&conn);
    let first = started.elapsed();
    let mut warm: Vec<Duration> = (0..iters)
        .map(|_| {
            let started = Instant::now();
            run(&conn);
            started.elapsed()
        })
        .collect();
    (first, percentile(&mut warm, P50))
}

/// Replace a run of `run_rows` rows of each commit `commits`
/// names: delete them, in one statement, then append their new versions as
/// one more commit.
fn update(
    conn: &Connection,
    schema: &SchemaRef,
    columns: usize,
    commits: &[usize],
    run_rows: usize,
    next_key: &mut i64,
    rng: &mut StdRng,
) {
    let table = conn.open_table(TABLE).expect("open");
    let runs = commits.iter().map(|&commit| {
        let start = (commit * ROWS_PER_COMMIT) as i64
            + rng.random_range(0..=(ROWS_PER_COMMIT - run_rows)) as i64;
        col("k")
            .gt_eq(lit(start))
            .and(col("k").lt(lit(start + run_rows as i64)))
    });
    let runs = disjunction(runs.collect::<Vec<_>>()).expect("at least one run");
    table.delete(runs).expect("delete");
    let keys: Vec<i64> = (0..commits.len() * run_rows)
        .map(|i| *next_key + i as i64)
        .collect();
    *next_key += keys.len() as i64;
    table
        .append(&rows(schema, columns, &keys, rng))
        .expect("append the new versions");
}

/// The table's superfiles, and how many of them carry deleted rows.
fn superfile_counts(conn: &Connection) -> (usize, usize) {
    let table = conn.open_table(TABLE).expect("open");
    let reader = table.local_handle().reader().expect("reader");
    let manifest = reader.manifest();
    let listed = manifest.get_all_superfiles();
    let with_deletes = manifest.get_tombstone_seqs().map_or(0, |seqs| {
        listed
            .iter()
            .filter(|entry| seqs.contains_key(&entry.superfile_id))
            .count()
    });
    (listed.len(), with_deletes)
}

/// A stage's label: its superfiles with deletes, of all.
fn stage(conn: &Connection) -> String {
    let (superfiles, with_deletes) = superfile_counts(conn);
    format!("deletes in {with_deletes} of {superfiles} superfiles")
}

/// One timed query, run on a connection.
type Timed = Box<dyn Fn(&Connection)>;

/// Time every query on the table at `uri`, as rows of the report for
/// `stage`.
fn measure(uri: &str, columns: usize, iters: usize, stage: &str) -> Vec<Vec<Cell>> {
    let mut out = Vec::new();
    let mut timed: Vec<(String, Timed)> = queries(columns)
        .into_iter()
        .map(|(label, sql)| {
            let run: Timed = Box::new(move |conn: &Connection| {
                black_box(conn.query_sql(&sql).expect("query"));
            });
            (label, run)
        })
        .collect();
    let label = column_name(0);
    timed.push((
        format!("value_counts({label})"),
        Box::new(move |conn: &Connection| {
            let table = conn.open_table(TABLE).expect("open");
            black_box(table.value_counts(&label).expect("value counts"));
        }),
    ));
    for (label, run) in timed {
        let (first, warm) = time(uri, iters, run);
        let (first, warm) = (
            first.as_secs_f64() * NS_PER_SEC,
            warm.as_secs_f64() * NS_PER_SEC,
        );
        eprintln!(
            "[stats-deletes] {stage} | {label}: first {}, warm p50 {}",
            fmt_time(first),
            fmt_time(warm)
        );
        out.push(vec![
            text(stage),
            text(label),
            metric(first, fmt_time(first), Better::Lower),
            metric(warm, fmt_time(warm), Better::Lower),
        ]);
    }
    out
}

/// Run the diagnostic.
pub fn run() {
    let iters = env_usize("INFINO_DIAG_ITERS", DEFAULT_ITERS).max(1);
    let total_rows =
        env_usize("INFINO_BENCH_STATS_DELETES_ROWS", DEFAULT_ROWS).max(ROWS_PER_COMMIT);
    let columns = env_usize("INFINO_BENCH_STATS_DELETES_COLUMNS", DEFAULT_COLUMNS).max(MIN_COLUMNS);
    let commits = total_rows / ROWS_PER_COMMIT;
    let dir = TempDir::new().expect("table dir");
    let uri = dir.path().to_str().expect("utf8 path").to_string();
    let schema = schema(columns);
    let mut rng = StdRng::seed_from_u64(SEED);

    eprintln!(
        "[stats-deletes] writing {commits} commits of {ROWS_PER_COMMIT} rows x {columns} columns..."
    );
    let conn = connect(&uri).expect("connect");
    let table = conn
        .create_table(TABLE, Arc::clone(&schema), IndexSpec::new())
        .expect("create");
    for commit in 0..commits {
        let keys: Vec<i64> = (0..ROWS_PER_COMMIT)
            .map(|r| (commit * ROWS_PER_COMMIT + r) as i64)
            .collect();
        table
            .append(&rows(&schema, columns, &keys, &mut rng))
            .expect("append");
    }
    let mut next_key = (commits * ROWS_PER_COMMIT) as i64;

    let mut report_rows = measure(&uri, columns, iters, &stage(&conn));

    let few: Vec<usize> = (0..commits).filter(|f| f % FEW_EVERY == 0).collect();
    eprintln!(
        "[stats-deletes] updating {} of {commits} commits...",
        few.len()
    );
    update(
        &conn,
        &schema,
        columns,
        &few,
        ROWS_PER_UPDATE,
        &mut next_key,
        &mut rng,
    );
    report_rows.extend(measure(&uri, columns, iters, &stage(&conn)));

    let most: Vec<usize> = (0..commits)
        .filter(|f| f % FEW_EVERY != 0 && f % MOST_SKIP_EVERY != 1)
        .collect();
    eprintln!("[stats-deletes] updating {} more commits...", most.len());
    update(
        &conn,
        &schema,
        columns,
        &most,
        ROWS_PER_UPDATE,
        &mut next_key,
        &mut rng,
    );
    report_rows.extend(measure(&uri, columns, iters, &stage(&conn)));

    let heavy: Vec<usize> = (0..commits).filter(|f| f % HEAVY_EVERY == 0).collect();
    eprintln!(
        "[stats-deletes] replacing a quarter of {} commits, then optimizing...",
        heavy.len()
    );
    update(
        &conn,
        &schema,
        columns,
        &heavy,
        ROWS_PER_COMMIT / HEAVY_SHARE,
        &mut next_key,
        &mut rng,
    );
    eprintln!("[stats-deletes] before optimize: {}", stage(&conn));
    let optimize = OptimizeOptions::compact(CompactionSettings {
        target_superfile_size_mb: OPTIMIZE_TARGET_MB,
        ..CompactionSettings::default()
    });
    let started = Instant::now();
    conn.open_table(TABLE)
        .expect("open")
        .optimize(&optimize)
        .expect("optimize");
    eprintln!(
        "[stats-deletes] optimize took {}",
        fmt_time(started.elapsed().as_secs_f64() * NS_PER_SEC)
    );
    let after = format!("optimized: {}", stage(&conn));
    report_rows.extend(measure(&uri, columns, iters, &after));

    let mut report = Report::load("stats-deletes");
    report.emit(&Section {
        anchor: "bench/diagnostic/stats-deletes".into(),
        title: "Statistics-answered aggregates on a table with deleted rows".into(),
        note: format!(
            "{commits} commits of {ROWS_PER_COMMIT} rows x {columns} nullable columns. Each \
             update replaces a run of {ROWS_PER_UPDATE} rows in each of some commits, in one \
             delete, and appends the new versions as one more commit. `first` is the first run on a fresh connection, `warm` the \
             median of {iters} more."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Stage".into(),
                "Query".into(),
                "first".into(),
                "warm p50".into(),
            ],
            rows: report_rows,
        }],
    });
    report.save();
}
