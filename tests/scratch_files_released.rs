// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Ingest and optimize leave no scratch files behind.
//!
//! A build seals each superfile to a file in the scratch directory and maps
//! it back. Once the superfile is published the file's name has no further
//! use, but a table on storage without a disk cache keeps the mapped bytes
//! for as long as it lives — and a background task can keep the table alive
//! past the caller's last handle, to the end of the process, whose exit runs
//! no destructors. Each run of a long ingest left a scratch file per
//! superfile behind that way.
//!
//! An optimize's commits drop what they are done with — the merges' term
//! contributions, files in the scratch directory, and the term-index layer
//! built from them — on threads of their own, and the optimize joins them
//! before it returns. The child watches the scratch directory while the
//! optimize runs, so the check after it is known to have had files to see
//! unlinked.
//!
//! The work runs in a child copy of this binary with the scratch directory
//! pointed at a fresh one, so the check covers the process ending too, and
//! nothing else in the process writes there. One test, its own binary.

#![cfg(unix)]
#![deny(clippy::unwrap_used)]

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};

use infino::{
    CompactionSettings, IndexSpec, OptimizeOptions,
    arrow_array::{LargeStringArray, RecordBatch},
    arrow_schema::{DataType, Field, Schema},
    connect,
};

/// Set in the child: the directory its table is stored in.
const ENV_DATA_DIR: &str = "INFINO_TEST_SCRATCH_DATA_DIR";
/// Appends, each a commit sealing its own superfiles.
const APPENDS: usize = 4;
/// Rows per append.
const ROWS_PER_APPEND: usize = 2000;
/// Distinct filler words, so the full-text index has real terms.
const VOCAB: usize = 97;
/// Superfiles that make a merge: every append's, so the optimize merges.
const MERGE_AT: u64 = 2;
/// How often the child looks at the scratch directory while it optimizes.
const WATCH_EVERY: Duration = Duration::from_micros(200);
/// Suffix of a superfile object in the table's storage directory.
const SUPERFILE_SUFFIX: &str = ".sf.parquet";
/// This test's name, which the child is filtered to.
const TEST_NAME: &str = "ingest_and_optimize_leave_no_scratch_files";

/// The entries in `dir`.
fn entries(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .expect("read the scratch directory")
        .map(|entry| entry.expect("scratch directory entry").path())
        .collect()
}

/// The files under `dir`, at any depth; none for a directory already gone.
fn files_under(dir: &Path) -> usize {
    let Ok(read) = fs::read_dir(dir) else {
        return 0;
    };
    read.filter_map(Result::ok)
        .map(|entry| entry.path())
        .map(|path| if path.is_dir() { files_under(&path) } else { 1 })
        .sum()
}

/// The superfile objects under `dir`, at any depth.
fn superfiles_under(dir: &Path) -> usize {
    fs::read_dir(dir)
        .expect("read the data directory")
        .map(|entry| entry.expect("data directory entry").path())
        .map(|path| {
            if path.is_dir() {
                superfiles_under(&path)
            } else {
                usize::from(path.to_string_lossy().ends_with(SUPERFILE_SUFFIX))
            }
        })
        .sum()
}

/// The child's work: appends, then an optimize, on a table stored in
/// `data`, checking the scratch directory after each while the table is
/// still open.
fn run_child(data: &Path) {
    let scratch = env::temp_dir();
    let db = connect(data.to_str().expect("utf-8 path")).expect("connect");
    let schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::LargeUtf8,
        false,
    )]));
    let table = db
        .create_table("docs", schema.clone(), IndexSpec::new().fts("title"))
        .expect("create table");
    for append in 0..APPENDS {
        let titles: Vec<String> = (0..ROWS_PER_APPEND)
            .map(|row| format!("doc {append} {row} quick brown fox w{}", row % VOCAB))
            .collect();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from(titles))],
        )
        .expect("batch");
        table.append(&batch).expect("append");
    }
    assert_eq!(
        entries(&scratch),
        Vec::<PathBuf>::new(),
        "the appends' published superfiles keep no scratch file"
    );
    assert_eq!(superfiles_under(data), APPENDS, "one superfile per append");
    let settings = CompactionSettings {
        min_superfiles_for_merge: MERGE_AT,
        ..CompactionSettings::default()
    };
    // The most scratch files the optimize held at once.
    let held = AtomicUsize::new(0);
    let optimizing = AtomicBool::new(true);
    thread::scope(|scope| {
        scope.spawn(|| {
            while optimizing.load(Ordering::Acquire) {
                held.fetch_max(files_under(&scratch), Ordering::Relaxed);
                thread::sleep(WATCH_EVERY);
            }
        });
        let optimized = table.optimize(&OptimizeOptions::compact(settings));
        optimizing.store(false, Ordering::Release);
        optimized.expect("optimize");
    });
    assert!(
        held.load(Ordering::Relaxed) > 0,
        "the optimize spilled scratch files, so the check below has some to see unlinked"
    );
    // The merged inputs are reclaimed later; the merge's output is here now.
    assert_eq!(superfiles_under(data), APPENDS + 1, "the optimize merged");
    assert_eq!(
        entries(&scratch),
        Vec::<PathBuf>::new(),
        "the optimize unlinked its scratch files before it returned"
    );
}

#[test]
fn ingest_and_optimize_leave_no_scratch_files() {
    if let Ok(data) = env::var(ENV_DATA_DIR) {
        run_child(Path::new(&data));
        return;
    }
    let data = tempfile::tempdir().expect("data dir");
    let scratch = tempfile::tempdir().expect("scratch dir");
    let output = Command::new(env::current_exe().expect("current_exe"))
        .args(["--exact", "--test-threads=1", "--nocapture", TEST_NAME])
        .env(ENV_DATA_DIR, data.path())
        .env("TMPDIR", scratch.path())
        .output()
        .expect("spawn the child");
    assert!(
        output.status.success(),
        "child failed: {}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        entries(scratch.path()),
        Vec::<PathBuf>::new(),
        "the process ended with scratch files left behind"
    );
}
