// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! What `IS [NOT] NULL` and `=` on nullable columns cost against a table
//! whose manifest is loaded part by part.
//!
//! A query first prunes the manifest's parts on their aggregate statistics
//! and loads only the parts that may match, then prunes the superfiles of
//! those parts on theirs. This diagnostic writes a many-part table with
//! three nullable columns — one set in every row, one in one row in a
//! hundred, one only in the first commits — opens it cold and lazily for
//! each query, and reports how many parts each predicate loaded, how many
//! rows it returned, and its median latency (open included).
//!
//! Invoked as `cargo bench -- null-prune`.

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use arrow_array::{Int64Array, LargeStringArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use infino::{
    supertable::{
        Supertable, SupertableOptions,
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::default_disk_cache,
};
use rayon::ThreadPoolBuilder;
use tempfile::TempDir;

use crate::{
    diag_common::{env_usize, percentile},
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
};

/// Commits, each one superfile; override with
/// `INFINO_BENCH_NULL_PRUNE_COMMITS`.
const DEFAULT_COMMITS: usize = 128;
/// Rows per commit.
const ROWS_PER_COMMIT: usize = 2_000;
/// Superfiles per manifest part.
const SUPERFILES_PER_PART: u64 = 4;
/// Forces the manifest's parts to load lazily, as a large table's do.
const FORCE_LAZY: u32 = 0;
/// One row in this many sets the sparse column.
const SPARSE_EVERY: usize = 100;
/// The rare column is set only in the first this-many commits.
const RARE_COMMITS: usize = 4;
/// Timed runs per query; override with `INFINO_DIAG_ITERS`.
const DEFAULT_ITERS: usize = 15;
const NS_PER_SEC: f64 = 1e9;
/// Median.
const P50: usize = 50;

/// The queries, by label.
const QUERIES: [(&str, &str); 5] = [
    (
        "dense IS NULL",
        "SELECT count(*) FROM supertable WHERE dense IS NULL",
    ),
    (
        "dense IS NOT NULL",
        "SELECT count(*) FROM supertable WHERE dense IS NOT NULL",
    ),
    (
        "sparse IS NULL",
        "SELECT count(*) FROM supertable WHERE sparse IS NULL",
    ),
    (
        "rare IS NOT NULL",
        "SELECT count(*) FROM supertable WHERE rare IS NOT NULL",
    ),
    (
        "rare = 'r'",
        "SELECT count(*) FROM supertable WHERE rare = 'r'",
    ),
];

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("title", DataType::LargeUtf8, false),
        Field::new("dense", DataType::LargeUtf8, true),
        Field::new("sparse", DataType::LargeUtf8, true),
        Field::new("rare", DataType::LargeUtf8, true),
    ]))
}

fn options(storage: &Arc<dyn StorageProvider>) -> SupertableOptions {
    // One writer thread: one superfile per commit, so parts fill evenly.
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("writer pool"),
    );
    SupertableOptions::new(schema(), vec![], vec![])
        .expect("options")
        .with_storage(Arc::clone(storage))
        .with_writer_pool(pool)
        .with_target_superfiles_per_part(SUPERFILES_PER_PART)
}

fn commit_rows(commit: usize) -> RecordBatch {
    let rows = 0..ROWS_PER_COMMIT;
    let title = LargeStringArray::from_iter_values(rows.clone().map(|r| format!("t{commit}-{r}")));
    let dense = LargeStringArray::from_iter(rows.clone().map(|r| Some(format!("d{}", r % 10))));
    let sparse = LargeStringArray::from_iter(
        rows.clone()
            .map(|r| r.is_multiple_of(SPARSE_EVERY).then(|| format!("s{r}"))),
    );
    let rare = LargeStringArray::from_iter(rows.map(|_| (commit < RARE_COMMITS).then_some("r")));
    RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(title),
            Arc::new(dense),
            Arc::new(sparse),
            Arc::new(rare),
        ],
    )
    .expect("batch")
}

/// One cold run of `sql`: a fresh lazy open over its own empty disk cache.
/// The rows counted, the manifest parts loaded and in all, the storage
/// reads the open and query made, and the wall time.
fn run_once(storage: &Arc<dyn StorageProvider>, sql: &str) -> (i64, usize, usize, u64, Duration) {
    let meter = storage.usage_meter();
    let before = meter.snapshot();
    let cache_dir = TempDir::new().expect("cache dir");
    let cache = default_disk_cache(Arc::clone(storage), cache_dir.path());
    let started = Instant::now();
    let table = Supertable::open(
        options(storage)
            .with_eager_load_threshold(FORCE_LAZY)
            .with_disk_cache(cache),
    )
    .expect("open");
    let reader = table.reader().expect("reader");
    let batches = reader.query_sql(sql).expect("query");
    let elapsed = started.elapsed();
    let gets = meter.snapshot().since(&before).get_count;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("count")
        .value(0);
    let manifest = reader.manifest();
    let entries = manifest.get_all_list_entries();
    let loaded = entries
        .iter()
        .filter(|e| manifest.get_cached_part_by_id(&e.part_id).is_some())
        .count();
    (count, loaded, entries.len(), gets, elapsed)
}

/// Run the diagnostic.
pub fn run() {
    let iters = env_usize("INFINO_DIAG_ITERS", DEFAULT_ITERS).max(1);
    let commits = env_usize("INFINO_BENCH_NULL_PRUNE_COMMITS", DEFAULT_COMMITS).max(1);
    let dir = TempDir::new().expect("table dir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("storage"));
    eprintln!("[null-prune] writing {commits} commits of {ROWS_PER_COMMIT} rows...");
    {
        let table = Supertable::create(options(&storage)).expect("create");
        for commit in 0..commits {
            let mut writer = table.writer().expect("writer");
            writer.append(&commit_rows(commit)).expect("append");
            writer.commit().expect("commit");
        }
    }

    let mut rows: Vec<Vec<Cell>> = Vec::new();
    for (label, sql) in QUERIES {
        let mut times = Vec::with_capacity(iters);
        let mut seen = None;
        for _ in 0..iters {
            let (count, loaded, total, gets, elapsed) = run_once(&storage, sql);
            times.push(elapsed);
            seen = Some((count, loaded, total, gets));
        }
        let (count, loaded, total, gets) = seen.expect("at least one run");
        let p50 = percentile(&mut times, P50).as_secs_f64() * NS_PER_SEC;
        eprintln!(
            "[null-prune] {label}: {count} rows, {loaded}/{total} parts loaded, {gets} storage \
             reads, p50 {}",
            fmt_time(p50)
        );
        rows.push(vec![
            text(label),
            text(fmt_count(count as usize)),
            metric(loaded as f64, format!("{loaded}/{total}"), Better::Lower),
            metric(gets as f64, fmt_count(gets as usize), Better::Lower),
            metric(p50, fmt_time(p50), Better::Lower),
        ]);
    }

    let mut report = Report::load("null-prune");
    report.emit(&Section {
        anchor: "bench/diagnostic/null-prune".into(),
        title: "Null-check pruning over a lazily loaded manifest".into(),
        note: format!(
            "{commits} commits of {ROWS_PER_COMMIT} rows, {SUPERFILES_PER_PART} superfiles per \
             part. `dense` is set in every row, `sparse` in one in {SPARSE_EVERY}, `rare` only in \
             the first {RARE_COMMITS} commits. Each run opens the table cold and lazily over an \
             empty disk cache; storage reads and p50 include the open."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Predicate".into(),
                "Rows".into(),
                "Parts loaded".into(),
                "Storage reads".into(),
                "p50".into(),
            ],
            rows,
        }],
    });
    report.save();
}
