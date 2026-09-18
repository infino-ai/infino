// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Killing a reindex mid-run keeps the rewrites it finished, and the next
//! run resumes.
//!
//! That sentence is in `Supertable::reindex`'s own doc comment and
//! nothing executed it. It is also the claim the whole design rests on:
//! staleness is read from the files rather than tracked in a journal
//! precisely so an interrupted run needs no recovery step, and a migration
//! an operator dare not interrupt is not one they will run on a live
//! table.
//!
//! ## Why this crashes differently from the commit tests
//!
//! `supertable_commit_crash_localfs.rs` wraps the storage provider and
//! aborts immediately after a chosen PUT, which places the kill exactly.
//! Neither half of that works here. A corpus table is opened through the
//! catalog, and `connect_with` takes a URI rather than a provider, so
//! there is nowhere to hang a wrapper. And a table this engine wrote is
//! never stale, so a fixture built in-process would plan no jobs at all —
//! the input has to be bytes an older release wrote.
//!
//! So the child watches its own table directory and aborts once a chosen
//! number of rewritten superfiles have appeared. `abort()` raises SIGABRT
//! from whichever thread calls it, running no destructors, which is the
//! same durability question the commit tests ask: the process is gone
//! between a superfile's bytes landing and the manifest swap that makes
//! them visible.
//!
//! The kill therefore lands *somewhere* in the second job rather than at a
//! named byte, and the assertions are written for that: every outcome the
//! window allows is coherent, and the test says which ones it saw.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use infino::{ReindexOptions, connect, superfile::format::fts::VERSION_CURRENT};
use tempfile::TempDir;

use crate::corpus_shapes::{
    N_DOCS, TABLE, assert_scores_equivalent, blob_versions, copy_tree, corpus_dir, hits,
    scores_by_id,
};

/// Directory the child reindexes. Set by the parent; its presence is what
/// makes the child a child.
const ENV_DIR: &str = "INFINO_REINDEX_CRASH_DIR";

/// The shape this runs against: five superfiles, so a run has jobs left to
/// resume after the kill. A single-superfile table would make "interrupted"
/// and "not started" the same state.
const CRASH_SHAPE: &str = "v2_positions_region";
/// Blob version that shape carries before the migration.
const CRASH_SHAPE_VERSION: u32 = 2;
/// Superfiles the corpus shape holds.
const CRASH_SHAPE_SUPERFILES: usize = 5;

/// Rewritten superfiles that must appear before the child aborts.
///
/// Two, so the kill lands with at least one job committed and at least two
/// still to do — the state the resume has to pick up. One would leave the
/// crash indistinguishable from a run that never started; five would leave
/// nothing to resume.
const ABORT_AFTER_REWRITES: usize = 2;

/// How often the watcher counts superfiles on disk.
const WATCH_POLL: Duration = Duration::from_millis(5);

/// How long the child waits for its abort condition before giving up.
///
/// Exceeding it means the reindex finished without the watcher ever seeing
/// the threshold, which is a test that proved nothing — so the child exits
/// zero and the parent fails on the clean exit rather than reporting a
/// pass.
const WATCH_TIMEOUT: Duration = Duration::from_secs(60);

/// Every superfile under `root`, by path.
fn superfile_paths(root: &Path) -> Vec<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    let mut found = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            match path.is_dir() {
                true => stack.push(path),
                false if path.extension().is_some_and(|e| e == "parquet") => found.push(path),
                false => {}
            }
        }
    }
    found
}

/// The child: reindex, and abort once the run is demonstrably underway.
///
/// The superseded bytes stay on disk until they are collected, so each
/// rewrite *adds* a file rather than replacing one — which is what makes a
/// plain file count a usable progress signal without reaching into the
/// manifest.
fn run_reindex_crash_child(dir: PathBuf) -> ! {
    let watch_root = dir.clone();
    thread::spawn(move || {
        let deadline = Instant::now() + WATCH_TIMEOUT;
        let target = CRASH_SHAPE_SUPERFILES + ABORT_AFTER_REWRITES;
        while Instant::now() < deadline {
            if superfile_paths(&watch_root).len() >= target {
                // No unwinding, no destructors, no flush: the process is
                // simply gone, which is the durability question being
                // asked.
                std::process::abort();
            }
            thread::sleep(WATCH_POLL);
        }
    });

    let db = connect(dir.to_str().expect("utf-8 path")).expect("child connects");
    let table = db.open_table(TABLE).expect("child opens the corpus table");
    let _ = table.reindex(&ReindexOptions::default());

    // Reaching here means the whole run landed before the watcher fired.
    // Exiting zero makes the parent fail loudly rather than pass on a
    // crash that never happened.
    std::process::exit(0);
}

/// Become the child when the parent set the directory.
fn dispatch_child_if_set() -> Option<()> {
    if let Ok(dir) = env::var(ENV_DIR) {
        run_reindex_crash_child(PathBuf::from(dir));
    }
    None
}

#[test]
fn an_interrupted_reindex_keeps_its_finished_rewrites_and_resumes() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let Some(src) = corpus_dir(CRASH_SHAPE) else {
        return;
    };

    // The ranking the table must still produce after being killed, taken
    // from an untouched copy of the same bytes rather than from the copy
    // the child is about to damage.
    let pristine = TempDir::new().expect("tempdir");
    copy_tree(&src, pristine.path());
    let db = connect(pristine.path().to_str().expect("utf-8 path")).expect("connect to pristine");
    let baseline = scores_by_id(
        &db.open_table(TABLE).expect("open pristine"),
        "body",
        "common shared",
        N_DOCS,
    );
    assert!(!baseline.is_empty(), "the baseline ranking is empty");

    // `keep` leaks the directory so it survives for the parent's
    // inspection; a guard would drop it before the assertions run.
    let victim = TempDir::new().expect("tempdir").keep();
    copy_tree(&src, &victim);

    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .args([
            "--exact",
            "--test-threads=1",
            "reindex_crash::an_interrupted_reindex_keeps_its_finished_rewrites_and_resumes",
        ])
        .env(ENV_DIR, &victim)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn child");
    assert!(
        !status.success(),
        "the child exited cleanly, so the reindex finished before the kill and \
         this asserted nothing about an interrupted one: {status:?}"
    );

    // The table opens at all. A crash between a superfile's bytes and the
    // manifest swap that publishes them must leave the manifest readable —
    // the bytes nobody committed are orphans, which readers tolerate.
    let db = connect(victim.to_str().expect("utf-8 path")).expect("the killed table reopens");
    let table = db.open_table(TABLE).expect("the killed table opens");

    // Nothing was lost and nothing moved. This is the assertion that would
    // catch a half-committed rewrite being read as live.
    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS,
        "documents went missing across the crash"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS),
        &baseline,
        "the killed table against the same bytes untouched",
    );

    // Collect the orphans the crash left, then look at what is live. Every
    // superfile is either one the run had not reached or one it finished —
    // a third value would mean a partially written file had been published.
    table.gc(Duration::ZERO).expect("collect orphans");
    let after_crash = blob_versions(&victim);
    assert_eq!(
        after_crash.len(),
        CRASH_SHAPE_SUPERFILES,
        "the live superfile count changed across the crash: {after_crash:?}"
    );
    let migrated = after_crash
        .iter()
        .filter(|v| **v == VERSION_CURRENT)
        .count();
    assert!(
        after_crash
            .iter()
            .all(|v| *v == VERSION_CURRENT || *v == CRASH_SHAPE_VERSION),
        "a superfile is at neither the old nor the new version: {after_crash:?}"
    );
    assert!(
        migrated >= 1,
        "the child aborted without committing a rewrite, so nothing was kept \
         to resume from: {after_crash:?}"
    );
    assert!(
        migrated < CRASH_SHAPE_SUPERFILES,
        "the run finished before the kill, so there was nothing left to \
         resume: {after_crash:?}"
    );

    // The claim itself, and the part a crash complicates. The dead process
    // still holds a tombstone-sidecar seal on the superfile it was
    // rewriting, and the seal is what keeps two writers off one file — a
    // live owner and a dead one look identical from here. So the resume
    // honours it, migrates everything else, and says what it had to leave.
    let report = table
        .reindex(&ReindexOptions::default())
        .expect("a resume makes progress rather than failing on the dead run's seal");
    // The rewrites the dead run finished are not redone. `already_current`
    // stays zero throughout and is not the check: it counts files current
    // on *both* axes, and every file here keeps the analysis revision it
    // was written at, so none of them ever qualifies. What the resume owes
    // is that it touches exactly the containers still behind.
    assert_eq!(
        report.rewritten + report.held_by_another_run,
        CRASH_SHAPE_SUPERFILES - migrated,
        "the resume did not account for everything the crash left behind"
    );
    assert_eq!(
        report.awaiting_reanalysis, CRASH_SHAPE_SUPERFILES,
        "every file in this shape predates the analysis revision, so a \
         container rewrite leaves all of them waiting"
    );
    assert!(
        report.held_by_another_run <= 1,
        "one dead process can hold at most the superfile it was rewriting,          got {}",
        report.held_by_another_run
    );

    // And the seal is abandoned, not permanent. Lowering the timeout is
    // what an operator does when the crash is known rather than suspected;
    // zero is that knob taken to its limit, and the run then takes the
    // file over and finishes the table.
    let takeover = ReindexOptions {
        stale_seal_timeout_ms: 0,
        ..ReindexOptions::default()
    };
    table
        .reindex(&takeover)
        .expect("an abandoned seal is taken over once it is stale");

    table.gc(Duration::ZERO).expect("collect superseded bytes");
    let after_resume = blob_versions(&victim);
    assert!(
        after_resume.iter().all(|v| *v == VERSION_CURRENT),
        "the table is not fully migrated after the resume: {after_resume:?}"
    );
    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS,
        "documents went missing across the resume"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS),
        &baseline,
        "the resumed migration",
    );
}
