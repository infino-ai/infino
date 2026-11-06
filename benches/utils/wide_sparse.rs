// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Compaction of a wide, sparse table: thousands of nullable columns,
//! each row filling only a handful of them.
//!
//! A table like this stores very little — a null costs a bit and a run
//! of nulls almost nothing — but a merge reads it back through Arrow,
//! where a null string column of `n` rows still holds `n + 1` offsets.
//! This diagnostic builds such a table from a seeded generator, runs the
//! default `optimize`, and reports how much memory the compaction took
//! against how much the table stores.
//!
//! The table grows its schema as it is written, as a table fed documents
//! whose keys keep appearing does: each commit adds the columns its rows
//! are the first to use, so the early superfiles lack the late columns
//! and every superfile stores the columns known when it was written,
//! almost all of them null.
//!
//! Peak memory is sampled from `/proc` (Linux). Elsewhere the sampler
//! reads nothing; run the optimize alone under the operating system's own tool
//! instead: build the table once into `INFINO_BENCH_WIDE_SPARSE_DIR`
//! (which builds and stops), copy it, and point the knob at the copy,
//! which then only optimizes.
//!
//! Invoked as `cargo bench -- wide-sparse`. Knobs:
//! `INFINO_BENCH_WIDE_SPARSE_ROWS` (default 300 000),
//! `INFINO_BENCH_WIDE_SPARSE_COLUMNS` (default 2 500),
//! `INFINO_BENCH_WIDE_SPARSE_DIR` (build the table at this path and stop;
//! a table already there is optimized without being rebuilt).

use std::{
    env, fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use arrow_array::{
    ArrayRef, RecordBatch,
    builder::{Int64Builder, LargeStringBuilder, ListBuilder},
    new_null_array,
};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use infino::{IndexSpec, OptimizeOptions, connect};
use parquet::file::reader::{FileReader, SerializedFileReader};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use tempfile::TempDir;

use crate::{
    diag_common::env_usize,
    markdown::{fmt_count, fmt_time},
    report::{Better, Block, Cell, Report, Section, metric, text},
    rss::{self, PeakSampler},
};

/// Rows in the table.
const DEFAULT_ROWS: usize = 300_000;

/// Columns the table ends with.
const DEFAULT_COLUMNS: usize = 2_500;

/// Rows per commit: each commit writes its own superfiles, so the table
/// reaches the merge as many small files, the shape compaction exists for.
const ROWS_PER_COMMIT: usize = 10_000;

/// Columns every commit's rows draw from, as a few keys appear on every
/// kind of document.
const COMMON_COLUMNS: usize = 20;

/// Common columns each row fills.
const COMMON_PER_ROW: usize = 2;

/// Columns a commit's own kind of document draws the rest of its keys
/// from: a window of the table's columns that moves from commit to commit.
const FAMILY_COLUMNS: usize = 60;

/// Columns of its commit's family each row fills.
const FAMILY_PER_ROW: usize = 4;

/// How far the family window moves between commits: a prime, so the
/// windows spread over the whole width.
const FAMILY_STRIDE: usize = 397;

/// Share of the final width the first commit already knows; the rest is
/// added a slice per commit.
const INITIAL_WIDTH_PERCENT: usize = 25;

/// Every `LIST_EVERY`-th column holds a list of strings, every
/// `INT_EVERY`-th (that is not a list) an integer; the rest hold strings.
const LIST_EVERY: usize = 10;
const INT_EVERY: usize = 7;

/// Strings in a filled list cell.
const LIST_ITEMS: usize = 3;

/// The generator's seed.
const SEED: u64 = 1;

/// The table's name in the catalog.
const TABLE: &str = "wide_sparse";

const NS_PER_SEC: f64 = 1e9;
const KIB: f64 = 1024.0;
const MIB: f64 = 1024.0 * 1024.0;

/// A superfile's file name ends in this.
const SUPERFILE_SUFFIX: &str = ".sf.parquet";

/// Parquet's tail: the footer's length, then the `PAR1` magic.
const PARQUET_TAIL_BYTES: usize = 8;

/// Where the table keeps what it records per superfile.
const MANIFEST_RECORD_DIRS: [&str; 2] = ["manifest-parts", "manifest-summaries"];

/// The `i`-th column's name.
fn column_name(i: usize) -> String {
    format!("prop_{i:05}")
}

/// The `i`-th column's type.
fn column_type(i: usize) -> DataType {
    if i.is_multiple_of(LIST_EVERY) {
        DataType::List(Arc::new(Field::new("item", DataType::LargeUtf8, true)))
    } else if i.is_multiple_of(INT_EVERY) {
        DataType::Int64
    } else {
        DataType::LargeUtf8
    }
}

/// How many columns commit `commit` of `commits` knows: from
/// [`INITIAL_WIDTH_PERCENT`] of `columns` at the first to all of them at
/// the last.
fn width_at(commit: usize, commits: usize, columns: usize) -> usize {
    let first = (columns * INITIAL_WIDTH_PERCENT / 100).max(1);
    if commits <= 1 {
        return columns;
    }
    first + (columns - first) * commit / (commits - 1)
}

/// Commit `commit`'s rows, `rows` of them over the first `width` columns:
/// each fills [`COMMON_PER_ROW`] of the [`COMMON_COLUMNS`] and
/// [`FAMILY_PER_ROW`] of its commit's family of [`FAMILY_COLUMNS`], so a
/// commit sets a few dozen of the table's columns, as a batch of one kind
/// of document does. A column no row fills is one null array shared by
/// every such column of its type, so the generator costs what the rows
/// hold rather than rows × width.
fn sparse_batch(
    rng: &mut StdRng,
    commit: usize,
    rows: usize,
    width: usize,
    first_row: usize,
) -> RecordBatch {
    let common = COMMON_COLUMNS.min(width);
    let rest = width - common;
    let family_start = match rest {
        0 => 0,
        rest => (commit * FAMILY_STRIDE) % rest,
    };
    let mut filled: Vec<Vec<usize>> = vec![Vec::new(); width];
    let mut fill = |row: usize, column: usize| {
        if filled[column].last() != Some(&row) {
            filled[column].push(row);
        }
    };
    for row in 0..rows {
        for _ in 0..COMMON_PER_ROW {
            fill(row, rng.random_range(0..common));
        }
        if rest == 0 {
            continue;
        }
        for _ in 0..FAMILY_PER_ROW {
            let offset = rng.random_range(0..FAMILY_COLUMNS.min(rest));
            fill(row, common + (family_start + offset) % rest);
        }
    }
    let shared_nulls: Vec<(DataType, ArrayRef)> = [0, INT_EVERY, 1]
        .into_iter()
        .map(|i| {
            let t = column_type(i);
            (t.clone(), new_null_array(&t, rows))
        })
        .collect();
    let null_of = |t: &DataType| -> ArrayRef {
        let (_, array) = shared_nulls
            .iter()
            .find(|(st, _)| st == t)
            .expect("every column type has a shared null array");
        Arc::clone(array)
    };
    let mut fields = Vec::with_capacity(width);
    let mut columns = Vec::with_capacity(width);
    for (i, rows_filled) in filled.iter().enumerate() {
        let data_type = column_type(i);
        fields.push(Field::new(column_name(i), data_type.clone(), true));
        if rows_filled.is_empty() {
            columns.push(null_of(&data_type));
            continue;
        }
        columns.push(filled_column(&data_type, i, rows_filled, rows, first_row));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).expect("sparse batch")
}

/// Column `i` of `rows` rows with a value at each of `filled` (ascending)
/// and nulls elsewhere.
fn filled_column(
    data_type: &DataType,
    i: usize,
    filled: &[usize],
    rows: usize,
    first_row: usize,
) -> ArrayRef {
    let mut next = filled.iter().copied().peekable();
    let mut take = |row: usize| -> bool {
        if next.peek() == Some(&row) {
            next.next();
            true
        } else {
            false
        }
    };
    match data_type {
        DataType::Int64 => {
            let mut b = Int64Builder::with_capacity(rows);
            for row in 0..rows {
                match take(row) {
                    true => b.append_value((first_row + row) as i64),
                    false => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
        DataType::List(_) => {
            let mut b = ListBuilder::new(LargeStringBuilder::new());
            for row in 0..rows {
                if take(row) {
                    for item in 0..LIST_ITEMS {
                        b.values().append_value(format!("v{i}_{item}"));
                    }
                    b.append(true);
                } else {
                    b.append_null();
                }
            }
            Arc::new(b.finish())
        }
        _ => {
            let mut b = LargeStringBuilder::new();
            for row in 0..rows {
                match take(row) {
                    true => b.append_value(format!("value {} of {}", first_row + row, i)),
                    false => b.append_null(),
                }
            }
            Arc::new(b.finish())
        }
    }
}

/// Bytes under `dir`, recursively.
fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_bytes(&e.path()),
            _ => e.metadata().map(|m| m.len()).unwrap_or(0),
        })
        .sum()
}

/// Files under `dir`, recursively.
fn files_under(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .flatten()
        .flat_map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => files_under(&e.path()),
            _ => vec![e.path()],
        })
        .collect()
}

/// Whether `path` is a superfile.
fn is_superfile(path: &Path) -> bool {
    path.to_string_lossy().ends_with(SUPERFILE_SUFFIX)
}

/// Superfiles under `dir`, recursively.
fn superfile_count(dir: &Path) -> usize {
    files_under(dir).iter().filter(|p| is_superfile(p)).count()
}

/// What the table holds per superfile, averaged over its superfiles: the
/// Parquet leaf columns a file stores, the bytes of its footer, and the
/// bytes the table records about it in its manifest parts and summaries.
struct Layout {
    superfiles: usize,
    columns: f64,
    footer_bytes: f64,
    record_bytes: f64,
}

impl Layout {
    /// The layout of the table at `root`.
    fn of(root: &Path) -> Self {
        let files = files_under(root);
        let mut superfiles = 0usize;
        let mut columns = 0usize;
        let mut footer_bytes = 0u64;
        for path in files.iter().filter(|p| is_superfile(p)) {
            let bytes = Bytes::from(fs::read(path).expect("read superfile"));
            let tail = &bytes[bytes.len() - PARQUET_TAIL_BYTES..];
            let footer_len = u32::from_le_bytes(tail[..4].try_into().expect("footer length"));
            footer_bytes += u64::from(footer_len);
            let reader = SerializedFileReader::new(bytes).expect("parquet footer");
            columns += reader
                .metadata()
                .file_metadata()
                .schema_descr()
                .num_columns();
            superfiles += 1;
        }
        let record_bytes: u64 = files
            .iter()
            .filter(|p| {
                let p = p.to_string_lossy();
                MANIFEST_RECORD_DIRS.iter().any(|d| p.contains(d))
            })
            .map(|p| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        let per = superfiles.max(1) as f64;
        Self {
            superfiles,
            columns: columns as f64 / per,
            footer_bytes: footer_bytes as f64 / per,
            record_bytes: record_bytes as f64 / per,
        }
    }

    fn print(&self, when: &str) {
        eprintln!(
            "[wide-sparse] {when}: {} superfiles; per superfile {:.0} stored columns, \
             {:.1} KiB footer, {:.1} KiB of manifest records",
            self.superfiles,
            self.columns,
            self.footer_bytes / KIB,
            self.record_bytes / KIB,
        );
    }
}

/// Write the table into `root`: `rows` rows over a schema that grows to
/// `columns` columns, one commit per [`ROWS_PER_COMMIT`] rows.
fn build(root: &Path, rows: usize, columns: usize) {
    let db = connect(root.to_str().expect("utf8 table path")).expect("connect");
    let commits = rows.div_ceil(ROWS_PER_COMMIT);
    let first = width_at(0, commits, columns);
    let seed_schema = Arc::new(Schema::new(
        (0..first)
            .map(|i| Field::new(column_name(i), column_type(i), true))
            .collect::<Vec<_>>(),
    ));
    let table = db
        .create_table(TABLE, seed_schema, IndexSpec::new())
        .expect("create wide table");
    let mut rng = StdRng::seed_from_u64(SEED);
    for commit in 0..commits {
        let first_row = commit * ROWS_PER_COMMIT;
        let n = ROWS_PER_COMMIT.min(rows - first_row);
        let width = width_at(commit, commits, columns);
        let batch = sparse_batch(&mut rng, commit, n, width, first_row);
        table.append(&batch).expect("append");
    }
}

/// Whether `root` already holds a written table.
fn has_table(root: &Path) -> bool {
    superfile_count(root) > 0
}

/// Run the diagnostic.
pub fn run() {
    let rows = env_usize("INFINO_BENCH_WIDE_SPARSE_ROWS", DEFAULT_ROWS);
    let columns = env_usize("INFINO_BENCH_WIDE_SPARSE_COLUMNS", DEFAULT_COLUMNS).max(1);
    let kept: Option<PathBuf> = env::var_os("INFINO_BENCH_WIDE_SPARSE_DIR").map(PathBuf::from);
    let scratch = TempDir::new().expect("tempdir");
    let root = kept.clone().unwrap_or_else(|| scratch.path().to_path_buf());

    let build_ns = if has_table(&root) {
        eprintln!(
            "[wide-sparse] optimizing the table already at {}",
            root.display()
        );
        0.0
    } else {
        eprintln!("[wide-sparse] building {rows} rows over {columns} columns...");
        fs::create_dir_all(&root).expect("table dir");
        let started = Instant::now();
        build(&root, rows, columns);
        let build_ns = started.elapsed().as_secs_f64() * NS_PER_SEC;
        Layout::of(&root).print("written");
        if kept.is_some() {
            eprintln!(
                "[wide-sparse] built at {} in {:.1} s; copy it and point the knob at the copy \
                 to measure the optimize alone",
                root.display(),
                build_ns / NS_PER_SEC
            );
            return;
        }
        build_ns
    };
    let stored = dir_bytes(&root);
    let files_before = superfile_count(&root);
    let layout = Layout::of(&root);

    let db = connect(root.to_str().expect("utf8 table path")).expect("connect");
    let table = db.open_table(TABLE).expect("open wide table");
    let sampler = PeakSampler::start_default();
    let started = Instant::now();
    table
        .optimize(&OptimizeOptions::default())
        .expect("optimize wide table");
    let optimize_ns = started.elapsed().as_secs_f64() * NS_PER_SEC;
    let rss = sampler.stop_stats();
    // The inputs stay on disk until gc's safety gap has passed, so the
    // files the optimize added are its outputs.
    let written = superfile_count(&root).saturating_sub(files_before);
    eprintln!(
        "[wide-sparse] stored {:.1} MiB in {files_before} superfiles; optimize {:.1} s, \
         peak RSS {} ({:.1}x stored); merged into {written} superfiles",
        stored as f64 / MIB,
        optimize_ns / NS_PER_SEC,
        rss::fmt_bytes(rss.peak_rss_bytes),
        rss.peak_rss_bytes as f64 / stored.max(1) as f64,
    );

    let row: Vec<Cell> = vec![
        text(format!("{} × {}", fmt_count(rows), fmt_count(columns))),
        metric(stored as f64, rss::fmt_bytes(stored), Better::Lower),
        metric(build_ns, fmt_time(build_ns), Better::Lower),
        metric(optimize_ns, fmt_time(optimize_ns), Better::Lower),
        metric(
            rss.peak_rss_bytes as f64,
            rss::fmt_bytes(rss.peak_rss_bytes),
            Better::Lower,
        ),
        text(format!("{files_before} → {written}")),
        metric(
            layout.columns,
            format!("{:.0}", layout.columns),
            Better::Lower,
        ),
        metric(
            layout.footer_bytes,
            format!("{:.1} KiB", layout.footer_bytes / KIB),
            Better::Lower,
        ),
    ];
    let mut report = Report::load("wide-sparse");
    report.emit(&Section {
        anchor: "bench/diagnostic/wide-sparse".into(),
        title: "Compaction of a wide, sparse table".into(),
        note: format!(
            "Rows fill {COMMON_PER_ROW} of {COMMON_COLUMNS} common columns and \
             {FAMILY_PER_ROW} of their commit's {FAMILY_COLUMNS}-column family; the \
             schema grows from {INITIAL_WIDTH_PERCENT}% of its width to all of it across \
             commits of {ROWS_PER_COMMIT} rows. Peak RSS is sampled across the default \
             `optimize` (Linux only); columns and footer are per superfile as written."
        ),
        blocks: vec![Block {
            subtitle: String::new(),
            headers: vec![
                "Rows × columns".into(),
                "Stored".into(),
                "Build".into(),
                "Optimize".into(),
                "Peak RSS".into(),
                "Superfiles".into(),
                "Columns / superfile".into(),
                "Footer / superfile".into(),
            ],
            rows: vec![row],
        }],
    });
    report.save();
}
