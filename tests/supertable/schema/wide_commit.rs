// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! What a commit to a wide, sparse table costs in memory: thousands of
//! nullable columns, each row setting a few. A commit's peak follows the
//! cells its rows set, not its rows times the table's columns, a commit is
//! cut into superfiles by those cells, and a handle that commits again and
//! again keeps each commit's statistics, not a copy of its values.
//!
//! Memory is measured on the whole process, so each measurement runs in a
//! child process of this test binary that does nothing else: the test
//! re-enters itself with [`CHILD_ENV`] naming the shape to write, and the
//! child prints one line per commit.

use std::{env, process::Command, sync::Arc};

use arrow_array::{
    ArrayRef, RecordBatch,
    builder::{Int64Builder, LargeStringBuilder},
};
use arrow_schema::{DataType, Field, Schema};
use infino::{IndexSpec, connect, runtime_metrics::rss::purge_allocator};
use libmimalloc_sys::mi_process_info;
use tempfile::TempDir;

/// Names the shape a child writes; unset in the parent.
const CHILD_ENV: &str = "INFINO_TEST_WIDE_COMMIT_SHAPE";

/// Starts each line a child reports a commit on.
const RESULT_PREFIX: &str = "WIDE_COMMIT ";

/// Rows of each commit to an integer table.
const ROWS: usize = 6_000;

/// Columns of the text table: fewer than [`WIDE`], so that each column of a
/// commit holds a couple of hundred values, as many as a superfile counts
/// exactly when they repeat.
const TEXT_COLUMNS: usize = 500;

/// Rows of each commit to the text table.
const TEXT_ROWS: usize = 6_000;

/// Columns of the wide tables.
const WIDE: usize = 2_000;

/// Cells each row sets, in every shape: the narrow table sets all of its
/// columns, the wide one this many of its [`WIDE`].
const SET_PER_ROW: usize = 20;

/// Commits of the shape that commits repeatedly.
const REPEATED_COMMITS: usize = 10;

/// The last commits of [`REPEATED_COMMITS`] whose growth is measured: the
/// allocator keeps what the first commits' peaks committed, so growth is
/// taken once that has settled.
const MEASURED_COMMITS: usize = 5;

/// Bytes of each text value: distinct, so no value repeats in a column. A
/// column of a commit then holds several times what its statistics take.
const TEXT_VALUE_BYTES: usize = 200;

/// What a commit to the wide table may add to its peak beyond the narrow
/// table's, per column of the width: each column the commit stores still
/// costs its statistics, its footer entry and its page index, an encode
/// works through the columns a few at a time, and the allocator commits
/// memory in pages. A writer per column of the width, open at once, would
/// cost a few hundred KiB a column here.
const PEAK_PER_COLUMN: usize = 64 * 1024;

/// What a superfile's exact value tables may take together, as the writer
/// budgets them.
const VALUE_TABLES_BUDGET: usize = 128 * 1024;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// [`SET_PER_ROW`] integer columns, every cell set.
    Narrow,
    /// [`WIDE`] integer columns, [`SET_PER_ROW`] set in each row.
    Wide,
    /// [`TEXT_COLUMNS`] text columns, [`SET_PER_ROW`] set in each row with
    /// values no other row holds, committed [`REPEATED_COMMITS`] times.
    WideText,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Self::Narrow => "narrow",
            Self::Wide => "wide",
            Self::WideText => "wide-text",
        }
    }

    fn from_name(name: &str) -> Self {
        match name {
            "narrow" => Self::Narrow,
            "wide" => Self::Wide,
            "wide-text" => Self::WideText,
            other => panic!("unknown shape {other}"),
        }
    }

    fn columns(self) -> usize {
        match self {
            Self::Narrow => SET_PER_ROW,
            Self::Wide => WIDE,
            Self::WideText => TEXT_COLUMNS,
        }
    }

    fn rows(self) -> usize {
        match self {
            Self::Narrow | Self::Wide => ROWS,
            Self::WideText => TEXT_ROWS,
        }
    }

    fn commits(self) -> usize {
        match self {
            Self::Narrow | Self::Wide => 1,
            Self::WideText => REPEATED_COMMITS,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Narrow | Self::Wide => DataType::Int64,
            Self::WideText => DataType::LargeUtf8,
        }
    }

    fn schema(self) -> Arc<Schema> {
        Arc::new(Schema::new(
            (0..self.columns())
                .map(|c| Field::new(format!("c{c:04}"), self.data_type(), true))
                .collect::<Vec<_>>(),
        ))
    }

    /// Whether row `row` sets column `column`: [`SET_PER_ROW`] columns a
    /// row, spread over the width and moving from row to row.
    fn sets(self, row: usize, column: usize) -> bool {
        let columns = self.columns();
        let stride = columns / SET_PER_ROW;
        column % stride == row % stride
    }

    /// Commit `commit`'s rows, and the bytes of the values they set.
    fn batch(self, commit: usize) -> (RecordBatch, usize) {
        let mut value_bytes = 0;
        let rows = self.rows();
        let columns: Vec<ArrayRef> = (0..self.columns())
            .map(|c| -> ArrayRef {
                match self {
                    Self::Narrow | Self::Wide => {
                        let mut b = Int64Builder::with_capacity(rows);
                        for r in 0..rows {
                            match self.sets(r, c) {
                                true => {
                                    b.append_value(((commit * rows + r) * WIDE + c) as i64);
                                    value_bytes += size_of::<i64>();
                                }
                                false => b.append_null(),
                            }
                        }
                        Arc::new(b.finish())
                    }
                    Self::WideText => {
                        let mut b = LargeStringBuilder::new();
                        for r in 0..rows {
                            match self.sets(r, c) {
                                true => {
                                    let value = format!(
                                        "{:0width$}",
                                        (commit * rows + r) * TEXT_COLUMNS + c,
                                        width = TEXT_VALUE_BYTES
                                    );
                                    value_bytes += value.len();
                                    b.append_value(value);
                                }
                                false => b.append_null(),
                            }
                        }
                        Arc::new(b.finish())
                    }
                }
            })
            .collect();
        let batch = RecordBatch::try_new(self.schema(), columns).expect("batch");
        (batch, value_bytes)
    }
}

/// The process's resident bytes now and at their peak so far, and the
/// bytes the allocator has committed now.
fn process_memory() -> (usize, usize, usize) {
    let (mut elapsed, mut user, mut system) = (0, 0, 0);
    let (mut current, mut peak) = (0, 0);
    let (mut commit, mut peak_commit, mut faults) = (0, 0, 0);
    // SAFETY: every pointer is to a live local the call only writes.
    unsafe {
        mi_process_info(
            &mut elapsed,
            &mut user,
            &mut system,
            &mut current,
            &mut peak,
            &mut commit,
            &mut peak_commit,
            &mut faults,
        );
    }
    (current, peak, commit)
}

/// One commit as a child reports it.
#[derive(Debug)]
struct Commit {
    /// What the commit raised the process's peak by.
    peak_added: usize,
    /// What the allocator holds committed after the commit, its batch
    /// dropped and freed memory returned.
    committed: usize,
    /// Bytes of the values the commit's rows set.
    value_bytes: usize,
    /// The table's superfiles after the commit, the warm-up's included.
    superfiles: usize,
    /// Text columns of those superfiles counted by an exact table of their
    /// values rather than sketched.
    tables: usize,
}

/// Write `shape` as the child, printing a line per commit.
fn run_child(shape: Shape) {
    let dir = TempDir::new().expect("tempdir");
    let db = connect(dir.path().to_str().expect("utf8")).expect("connect");
    let table = db
        .create_table("wide", shape.schema(), IndexSpec::new())
        .expect("create");
    // Start the pools and the runtime with a commit of one row, so the
    // first measured commit pays for none of it.
    let (first, _) = shape.batch(0);
    table.append(&first.slice(0, 1)).expect("warm-up append");
    drop(first);
    for commit in 0..shape.commits() {
        let (batch, value_bytes) = shape.batch(commit + 1);
        purge_allocator();
        let (current, peak, _) = process_memory();
        table.append(&batch).expect("append");
        let (_, peak_after, _) = process_memory();
        drop(batch);
        purge_allocator();
        let (_, _, committed) = process_memory();
        let reader = table.local_handle().reader().expect("reader");
        let superfiles = reader.n_superfiles();
        // Text columns of the table's superfiles whose statistics carry no
        // sketch: an exact table of their values stands in for it. The
        // warm-up's superfile, of one row, is left out: a table of one value
        // is smaller than a sketch.
        let tables = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .filter(|entry| entry.n_docs > 1)
            .flat_map(|entry| entry.scalar_stats.map().into_owned().into_values())
            .filter(|stats| stats.min.data_type() == &DataType::LargeUtf8 && stats.hll.is_none())
            .count();
        println!(
            "{RESULT_PREFIX}{} {committed} {value_bytes} {superfiles} {tables}",
            peak_after.saturating_sub(peak.max(current)),
        );
    }
}

/// Run `shape` in a child process re-entering `test`, and read its commits.
fn measure(test: &str, shape: Shape) -> Vec<Commit> {
    let output = Command::new(env::current_exe().expect("current_exe"))
        .args(["--exact", "--test-threads=1", "--nocapture", test])
        .env(CHILD_ENV, shape.name())
        .output()
        .expect("spawn child");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the {shape:?} child failed: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let commits: Vec<Commit> = stdout
        .lines()
        // The harness prints a test's name on the line its output starts.
        .filter_map(|line| line.split_once(RESULT_PREFIX).map(|(_, rest)| rest))
        .map(|line| {
            let n: Vec<u64> = line
                .split(' ')
                .map(|v| v.parse().expect("a number"))
                .collect();
            Commit {
                peak_added: n[0] as usize,
                committed: n[1] as usize,
                value_bytes: n[2] as usize,
                superfiles: n[3] as usize,
                tables: n[4] as usize,
            }
        })
        .collect();
    assert_eq!(commits.len(), shape.commits(), "{stdout}");
    commits
}

/// Become the child when the parent named a shape.
fn child_if_named() -> bool {
    match env::var(CHILD_ENV) {
        Ok(name) => {
            run_child(Shape::from_name(&name));
            true
        }
        Err(_) => false,
    }
}

/// A commit of the same cells to a table a hundred times as wide peaks
/// within a small allowance per column of the narrow table's peak, and
/// stays one superfile: its encode holds the writers of the columns being
/// encoded, not a codec and a dictionary for every column of the width,
/// and it is cut into superfiles by the cells it sets, not by the slots
/// its nulls keep.
#[test]
fn a_wide_sparse_commit_peaks_by_the_cells_it_sets_not_its_width() {
    if child_if_named() {
        return;
    }
    const TEST: &str =
        "schema::wide_commit::a_wide_sparse_commit_peaks_by_the_cells_it_sets_not_its_width";
    let narrow = &measure(TEST, Shape::Narrow)[0];
    let wide = &measure(TEST, Shape::Wide)[0];
    assert_eq!(narrow.value_bytes, wide.value_bytes, "the same cells set");
    let allowance = WIDE * PEAK_PER_COLUMN;
    assert!(
        wide.peak_added <= narrow.peak_added + allowance,
        "a commit of {} set cells over {WIDE} columns raised the peak by {} bytes; over {} columns, by {} (allowed {allowance} more)",
        ROWS * SET_PER_ROW,
        wide.peak_added,
        SET_PER_ROW,
        narrow.peak_added
    );
    assert_eq!(
        wide.superfiles, 2,
        "the warm-up's superfile and one more: a commit is cut by the cells it sets, and these fit one"
    );
}

/// Commits through one handle add their statistics to what it holds, not
/// their values. A superfile keeps exact tables of its columns' values only
/// within its budget for them, [`VALUE_TABLES_BUDGET`], so on text columns
/// of a couple of hundred values each it keeps a few tables, not one a
/// column, which would be about the bytes written again, kept for the life
/// of the handle: the other columns keep a sketch of a KiB and their
/// bounds. And what the process keeps committed grows, over the last
/// [`MEASURED_COMMITS`] commits, by less than the bytes each commit wrote.
#[test]
fn repeated_commits_hold_their_statistics_not_their_values() {
    if child_if_named() {
        return;
    }
    const TEST: &str =
        "schema::wide_commit::repeated_commits_hold_their_statistics_not_their_values";
    let commits = measure(TEST, Shape::WideText);
    let first = &commits[commits.len() - 1 - MEASURED_COMMITS];
    let last = &commits[commits.len() - 1];
    // A column of a commit holds about `TEXT_ROWS * SET_PER_ROW /
    // TEXT_COLUMNS` values; half that is a floor on what a kept table of
    // them takes.
    let least_table = TEXT_ROWS * SET_PER_ROW / TEXT_COLUMNS / 2 * TEXT_VALUE_BYTES;
    let superfiles = last.superfiles - 1;
    assert!(
        last.tables * least_table <= superfiles * VALUE_TABLES_BUDGET,
        "{} tables over {superfiles} superfiles pass their budget: {commits:?}",
        last.tables
    );
    assert!(
        last.tables < superfiles * TEXT_COLUMNS / 10,
        "most columns are sketched, not copied into a table: {commits:?}"
    );
    let grown = last.committed.saturating_sub(first.committed) / MEASURED_COMMITS;
    assert!(
        grown < last.value_bytes,
        "each commit of {} bytes of values grows what the process holds by {grown} bytes: {commits:?}",
        last.value_bytes
    );
}
