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
//! That does not work here: the fixture is opened through the catalog,
//! and `connect_with` takes a URI rather than a provider, so there is
//! nowhere to hang a wrapper.
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
    env,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use infino::{ReindexOptions, connect, test_helpers::copy_dir_recursive};
use tempfile::TempDir;

use crate::{
    reindex_fixture::{
        N_DOCS, Staleness, TABLE, assert_scores_equivalent, file_revisions, hits, scores_by_id,
        superfile_paths, write_stale_table,
    },
    reindex_invariance::{DELETED_DOCS, delete_leading_rows, rows_by_id},
};

/// Directory the child reindexes. Set by the parent; its presence is what
/// makes the child a child.
const ENV_DIR: &str = "INFINO_REINDEX_CRASH_DIR";

/// The staleness this runs against. The fixture holds several superfiles,
/// so a run has jobs left to resume after the kill: a single-superfile
/// table would make "interrupted" and "not started" the same state.
const CRASH_STALENESS: Staleness = Staleness::Analysis;

/// Rewritten superfiles that must appear before the child aborts.
///
/// Two, so the kill lands with at least one job committed and more still
/// to do — the state the resume has to pick up. One would leave the crash
/// indistinguishable from a run that never started; all of them would
/// leave nothing to resume.
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

/// The child: reindex, and abort once the run is demonstrably underway.
///
/// The superseded bytes stay on disk until they are collected, so each
/// rewrite *adds* a file rather than replacing one — which is what makes a
/// plain file count a usable progress signal without reaching into the
/// manifest.
fn run_reindex_crash_child(dir: PathBuf) -> ! {
    let watch_root = dir.clone();
    let target = superfile_paths(&watch_root).len() + ABORT_AFTER_REWRITES;
    thread::spawn(move || {
        let deadline = Instant::now() + WATCH_TIMEOUT;
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
    let table = db.open_table(TABLE).expect("child opens the fixture table");
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
    // The ranking the table must still produce after being killed, taken
    // from an untouched copy of the same bytes rather than from the copy
    // the child is about to damage.
    let pristine = TempDir::new().expect("tempdir");
    let superfiles = write_stale_table(pristine.path(), CRASH_STALENESS);
    let stale_revision = file_revisions(pristine.path())[0];
    let db = connect(pristine.path().to_str().expect("utf-8 path")).expect("connect to pristine");
    let baseline = scores_by_id(
        &db.open_table(TABLE).expect("open pristine"),
        "body",
        "common shared",
        N_DOCS as usize,
    );
    assert!(!baseline.is_empty(), "the baseline ranking is empty");

    // `keep` leaks the directory so it survives for the parent's
    // inspection; a guard would drop it before the assertions run.
    let victim = TempDir::new().expect("tempdir").keep();
    copy_dir_recursive(pristine.path(), &victim);

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
        N_DOCS as usize,
        "documents went missing across the crash"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS as usize),
        &baseline,
        "the killed table against the same bytes untouched",
    );

    // Collect the orphans the crash left, then look at what is live. Every
    // superfile is either one the run had not reached or one it finished —
    // a third value would mean a partially written file had been published.
    table.gc(Duration::ZERO).expect("collect orphans");
    let after_crash = file_revisions(&victim);
    assert_eq!(
        after_crash.len(),
        superfiles,
        "the live superfile count changed across the crash: {after_crash:?}"
    );
    let migrated = after_crash.iter().filter(|r| **r > stale_revision).count();
    assert!(
        after_crash
            .iter()
            .all(|r| *r == stale_revision || *r == stale_revision + 1),
        "a superfile records neither the stale nor the current revision: {after_crash:?}"
    );
    assert!(
        migrated >= 1,
        "the child aborted without committing a rewrite, so nothing was kept \
         to resume from: {after_crash:?}"
    );
    assert!(
        migrated < superfiles,
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
    // The rewrites the dead run finished are not redone: the resume
    // touches exactly the files still behind.
    assert_eq!(
        report.already_current, migrated,
        "the resume did not recognise the dead run's finished rewrites: {report:?}"
    );
    assert_eq!(
        report.rewritten + report.held_by_another_run,
        superfiles - migrated,
        "the resume did not account for everything the crash left behind: {report:?}"
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
    let takeover = ReindexOptions::default().with_stale_seal_timeout_ms(0);
    table
        .reindex(&takeover)
        .expect("an abandoned seal is taken over once it is stale");

    table.gc(Duration::ZERO).expect("collect superseded bytes");
    assert!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess the resumed table")
            .is_current(),
        "the table is not fully repaired after the resume: {:?}",
        file_revisions(&victim)
    );
    assert_eq!(
        hits(&table, "body", "common"),
        N_DOCS as usize,
        "documents went missing across the resume"
    );
    assert_scores_equivalent(
        &scores_by_id(&table, "body", "common shared", N_DOCS as usize),
        &baseline,
        "the resumed migration",
    );
}

/// Killing a reindex on a table with deletions does not resurrect a row.
///
/// The sidecar write and the manifest swap are two durable steps; a kill
/// between them must not publish the output with its tombstones absent.
#[test]
fn an_interrupted_reindex_never_resurrects_a_deleted_row() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let victim = TempDir::new().expect("tempdir").keep();
    let superfiles = write_stale_table(&victim, CRASH_STALENESS);
    let stale_revision = file_revisions(&victim)[0];

    // Tombstone rows, and record which, before anything is killed.
    let deleted: Vec<i128> = {
        let db = connect(victim.to_str().expect("utf-8 path")).expect("connect to the victim");
        let table = db.open_table(TABLE).expect("open the victim");
        let mut ids: Vec<i128> = delete_leading_rows(&db, &table).into_iter().collect();
        ids.sort_unstable();
        ids
    };
    let live_before = live_ids(&victim);
    assert_eq!(
        live_before.len(),
        N_DOCS as usize - DELETED_DOCS,
        "the delete did not take before the crash run started"
    );

    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .args([
            "--exact",
            "--test-threads=1",
            "reindex_crash::an_interrupted_reindex_never_resurrects_a_deleted_row",
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

    // The claim, at whatever point in the window the kill landed.
    let after_crash = live_ids(&victim);
    assert!(
        deleted.iter().all(|id| !after_crash.contains(id)),
        "a deleted row came back across the crash"
    );
    assert_eq!(
        after_crash, live_before,
        "the live row set changed across the crash"
    );

    // And once the migration is driven to completion, taking over the dead
    // run's seal the way the sibling test establishes.
    let db = connect(victim.to_str().expect("utf-8 path")).expect("the killed table reopens");
    let table = db.open_table(TABLE).expect("the killed table opens");

    // Confirm the kill landed mid-migration; untouched or fully migrated
    // would let this pass without the window ever opening.
    table.gc(Duration::ZERO).expect("collect orphans");
    let after_crash_revisions = file_revisions(&victim);
    let migrated = after_crash_revisions
        .iter()
        .filter(|r| **r > stale_revision)
        .count();
    assert!(
        (1..superfiles).contains(&migrated),
        "the crash left the table either untouched or fully repaired, so the \
         window this test is about was never open: {after_crash_revisions:?}"
    );

    let takeover = ReindexOptions::default().with_stale_seal_timeout_ms(0);
    table
        .reindex(&takeover)
        .expect("a resume finishes a table with deletions");
    table.gc(Duration::ZERO).expect("collect superseded bytes");

    assert!(
        table
            .index_staleness(&ReindexOptions::default())
            .expect("assess the resumed table")
            .is_current(),
        "the table is not fully repaired after the resume: {:?}",
        file_revisions(&victim)
    );
    let after_resume_ids = live_ids(&victim);
    assert!(
        deleted.iter().all(|id| !after_resume_ids.contains(id)),
        "a deleted row came back across the resume"
    );
    assert_eq!(
        after_resume_ids, live_before,
        "the live row set changed across the resume"
    );
}

/// Every `_id` the table reads as live, in id order, from a fresh
/// connection — what is durably on disk, not what a handle cached.
fn live_ids(root: &Path) -> Vec<i128> {
    let db = connect(root.to_str().expect("utf-8 path")).expect("connect for a live read");
    rows_by_id(&db).into_iter().map(|(id, ..)| id).collect()
}
