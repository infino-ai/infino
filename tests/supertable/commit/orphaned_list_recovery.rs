// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Commit recovery past a crash-orphaned manifest list.
//!
//! A writer that dies between its manifest-list PUT and its pointer CAS
//! leaves the list object durably occupying the next manifest id while the
//! pointer still names the prior commit. Lists are conditional-create and
//! never overwritten, so the next writer's natural attempt at that id can
//! only surface `PreconditionFailed` — and, before the orphan-skip fix, its
//! OCC retry loop refreshed from the (unmoved) pointer, re-derived the same
//! id, and exhausted as `WriteContentionExhausted` until a GC sweep
//! reclaimed the orphan.
//!
//! These tests inject the crash residue directly (a PUT to the next
//! manifest id, no pointer advance) and assert the next commit publishes by
//! skipping past the occupied id, leaving the orphan for GC. The
//! process-level version of the same scenario (a real SIGABRT between the
//! two PUTs) lives in `tests/supertable_commit_crash_localfs.rs`.

#![deny(clippy::unwrap_used)]

use std::{sync::Arc, time::Duration};

use bytes::Bytes;
use infino::{
    supertable::{
        Supertable,
        manifest::commit::{manifest_uri, read_pointer},
        storage::{LocalFsStorageProvider, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;

/// Manifest id of the orphaned list a crashed writer left behind: `create`
/// publishes id 0, the first commit id 1, so the crashed commit was id 2.
const ORPHANED_MANIFEST_ID: u64 = 2;

/// Bytes standing in for the crashed writer's encoded manifest list. Never
/// parsed: no pointer references the object, and the recovery path skips
/// the id without reading it.
const ORPHANED_LIST_BYTES: &[u8] =
    b"manifest list left by a writer that died before its pointer CAS";

fn commit_titles(st: &Supertable, titles: &[&str]) {
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

fn put_orphan(storage: &Arc<dyn StorageProvider>, manifest_id: u64) {
    futures::executor::block_on(storage.put_atomic(
        &manifest_uri(manifest_id),
        Bytes::from_static(ORPHANED_LIST_BYTES),
    ))
    .expect("orphan PUT");
}

#[test]
fn commit_skips_past_orphaned_manifest_list() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    commit_titles(&st, &["first commit alpha"]);
    assert_eq!(st.manifest_id(), 1);

    // Crash residue: the list object occupies id 2, the pointer stays at 1.
    put_orphan(&storage, ORPHANED_MANIFEST_ID);

    // The next commit must publish rather than retry id 2 into
    // WriteContentionExhausted — it skips to id 3.
    commit_titles(&st, &["second commit beta"]);
    assert_eq!(
        st.manifest_id(),
        ORPHANED_MANIFEST_ID + 1,
        "commit publishes at the first id past the orphan"
    );
    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read pointer")
        .expect("pointer present");
    assert_eq!(pointer.get_manifest_id(), ORPHANED_MANIFEST_ID + 1);

    // Both commits' rows are visible through the published manifest.
    assert_eq!(st.reader().expect("reader").n_superfiles(), 2);

    // Recovery never touches the orphan: it stays on disk, unreferenced,
    // until a GC sweep past the safety gap reclaims it.
    let (orphan_bytes, _) =
        futures::executor::block_on(storage.get(&manifest_uri(ORPHANED_MANIFEST_ID)))
            .expect("orphan still present");
    assert_eq!(&orphan_bytes[..], ORPHANED_LIST_BYTES);
    st.gc(Duration::ZERO).expect("gc");
    let gone = futures::executor::block_on(storage.get(&manifest_uri(ORPHANED_MANIFEST_ID)));
    assert!(gone.is_err(), "gc sweeps the unreferenced orphan list");
}

#[test]
fn commit_skips_past_consecutive_orphaned_manifest_lists() {
    // Two writers crashing back-to-back (the second recovered past the
    // first, then died the same way) occupy two consecutive ids. Each
    // occupied id costs one OCC retry; the commit still publishes.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    commit_titles(&st, &["first commit alpha"]);

    put_orphan(&storage, ORPHANED_MANIFEST_ID);
    put_orphan(&storage, ORPHANED_MANIFEST_ID + 1);

    commit_titles(&st, &["second commit beta"]);
    assert_eq!(
        st.manifest_id(),
        ORPHANED_MANIFEST_ID + 2,
        "commit publishes at the first id past both orphans"
    );
    assert_eq!(st.reader().expect("reader").n_superfiles(), 2);
}

#[test]
fn fresh_handle_commits_past_orphaned_manifest_list() {
    // The wild-world shape of the hazard: the crashed writer's process is
    // gone, and a NEW handle (fresh open, no in-memory state carried over)
    // is the one that must commit past the residue.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    {
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        commit_titles(&st, &["first commit alpha"]);
    }
    put_orphan(&storage, ORPHANED_MANIFEST_ID);

    let st = Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("open");
    assert_eq!(st.manifest_id(), 1, "opens at the last published commit");
    commit_titles(&st, &["second commit beta"]);
    assert_eq!(st.manifest_id(), ORPHANED_MANIFEST_ID + 1);
    assert_eq!(st.reader().expect("reader").n_superfiles(), 2);
}
