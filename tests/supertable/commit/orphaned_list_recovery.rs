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

use std::{
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, sync_channel},
    },
    thread,
    time::Duration,
};

use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    supertable::{
        Supertable,
        manifest::commit::{POINTER_PATH, manifest_uri, read_pointer},
        storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    },
    test_helpers::{build_title_batch, default_supertable_options},
};
use tempfile::TempDir;
use tokio::sync::Notify;

/// Manifest id of the orphaned list a crashed writer left behind: `create`
/// publishes id 0, the first commit id 1, so the crashed commit was id 2.
const ORPHANED_MANIFEST_ID: u64 = 2;

/// Bytes standing in for the crashed writer's encoded manifest list. Never
/// parsed: no pointer references the object, and the recovery path skips
/// the id without reading it.
const ORPHANED_LIST_BYTES: &[u8] =
    b"manifest list left by a writer that died before its pointer CAS";

/// Length of the contiguous orphan run in the consecutive-orphans test.
/// Deliberately longer than [`TIGHT_COMMIT_RETRIES`]: recovery must escape
/// the whole run in a single retry (one forward probe to the first free
/// id), not one id per retry — the latter would exhaust the budget here.
const N_CONSECUTIVE_ORPHANS: u64 = 4;

/// Commit-retry budget for the consecutive-orphans test — smaller than the
/// orphan run so per-id skipping cannot pass.
const TIGHT_COMMIT_RETRIES: u32 = 2;

/// Safety gap far longer than any orphan's age within a test run, so a
/// guarded sweep must classify the fresh orphan as too new to reclaim.
const ORPHAN_PROTECTION_GAP: Duration = Duration::from_secs(3600);

/// Rounds of alternating stale-handle commits in the dense-ids tripwire.
const STALE_ALTERNATION_ROUNDS: usize = 3;

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

    // A safety gap longer than the orphan's age must protect it — to a
    // sweeper, a young orphan is indistinguishable from a live writer's
    // just-PUT, not-yet-published list.
    let guarded = st.gc(ORPHAN_PROTECTION_GAP).expect("guarded gc");
    assert!(
        guarded.objects_skipped_too_new >= 1,
        "the young orphan must be among the too-new skips"
    );
    futures::executor::block_on(storage.get(&manifest_uri(ORPHANED_MANIFEST_ID)))
        .expect("young orphan survives a guarded sweep");

    st.gc(Duration::ZERO).expect("gc");
    let gone = futures::executor::block_on(storage.get(&manifest_uri(ORPHANED_MANIFEST_ID)));
    assert!(gone.is_err(), "gc sweeps the unreferenced orphan list");
}

#[test]
fn commit_skips_past_consecutive_orphaned_manifest_lists() {
    // Writers crashing back-to-back (each recovered past the previous run,
    // then died the same way) leave a contiguous run of occupied ids. The
    // run is longer than the retry budget, so recovery must escape it in a
    // single retry — one forward probe to the first free id — rather than
    // skipping one id per retry.
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st = Supertable::create(
        default_supertable_options()
            .with_max_commit_retries(TIGHT_COMMIT_RETRIES)
            .with_storage(Arc::clone(&storage)),
    )
    .expect("create");
    commit_titles(&st, &["first commit alpha"]);

    for i in 0..N_CONSECUTIVE_ORPHANS {
        put_orphan(&storage, ORPHANED_MANIFEST_ID + i);
    }

    commit_titles(&st, &["second commit beta"]);
    assert_eq!(
        st.manifest_id(),
        ORPHANED_MANIFEST_ID + N_CONSECUTIVE_ORPHANS,
        "commit publishes at the first id past the whole orphan run"
    );
    assert_eq!(st.reader().expect("reader").n_superfiles(), 2);
}

/// Pass-through storage that pauses its handle's FIRST pointer CAS. The
/// wrapper signals `reached` from inside `put_if_match` — i.e. after the
/// commit's manifest-list PUT already landed — then holds the CAS until
/// `release` fires: the deterministic reproduction of a live winner frozen
/// inside the list-PUT → pointer-CAS window.
#[derive(Debug)]
struct PauseFirstPointerCas {
    inner: Arc<dyn StorageProvider>,
    armed: AtomicBool,
    reached: SyncSender<()>,
    release: Arc<Notify>,
}

#[async_trait]
impl StorageProvider for PauseFirstPointerCas {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }
    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }
    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.inner.get_range(uri, range).await
    }
    async fn put_atomic(&self, uri: &str, bytes: Bytes) -> Result<Option<String>, StorageError> {
        self.inner.put_atomic(uri, bytes).await
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        if uri == POINTER_PATH && self.armed.swap(false, Ordering::SeqCst) {
            self.reached.send(()).expect("test main alive");
            self.release.notified().await;
        }
        self.inner.put_if_match(uri, bytes, expected_etag).await
    }
    async fn put_multipart(
        &self,
        uri: &str,
    ) -> Result<Box<dyn object_store::MultipartUpload>, StorageError> {
        self.inner.put_multipart(uri).await
    }
    async fn delete(&self, uri: &str) -> Result<(), StorageError> {
        self.inner.delete(uri).await
    }
}

/// The hazardous interleaving behind the orphan-skip gate: winner W has
/// PUT its manifest list and stands before its pointer CAS while loser L
/// collides, HEAD-finds W's list, and floors past it. The etag fence must
/// elect exactly one winner per id — W's frozen CAS loses, its OCC retry
/// republishes, and every batch's rows land exactly once.
#[test]
fn loser_skips_a_live_winner_and_both_commits_land_exactly_once() {
    let dir = TempDir::new().expect("tempdir");
    let plain: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st_l = Supertable::create(default_supertable_options().with_storage(Arc::clone(&plain)))
        .expect("create");
    commit_titles(&st_l, &["first commit alpha"]);

    let (reached_tx, reached_rx) = sync_channel(1);
    let release = Arc::new(Notify::new());
    let wrapped: Arc<dyn StorageProvider> = Arc::new(PauseFirstPointerCas {
        inner: Arc::clone(&plain),
        armed: AtomicBool::new(true),
        reached: reached_tx,
        release: Arc::clone(&release),
    });
    let st_w = Supertable::open(default_supertable_options().with_storage(wrapped))
        .expect("open winner handle");

    let winner = thread::spawn(move || {
        let mut w = st_w.writer().expect("winner writer");
        w.append(&build_title_batch(&["winner bravo"]))
            .expect("append winner");
        w.commit()
            .expect("the frozen winner's commit must still publish");
        st_w.manifest_id()
    });
    reached_rx
        .recv()
        .expect("winner reached its pointer CAS with its list durable");

    // W is frozen with list id 2 durable and the pointer at 1. L's commit
    // collides at id 2, HEAD-finds the in-flight list, floors past it, and
    // publishes at id 3.
    commit_titles(&st_l, &["loser charlie"]);
    assert_eq!(
        st_l.manifest_id(),
        3,
        "loser skips the live winner's in-flight id"
    );

    // Released, W's fenced CAS at id 2 must LOSE (the pointer moved), and
    // its OCC retry republishes everything at id 4.
    release.notify_one();
    let winner_view = winner.join().expect("winner thread");
    assert_eq!(winner_view, 4, "winner republishes after losing the fence");

    // Exactly-once across the whole interleaving: three commits, three
    // superfiles, no double-publish of the winner's rows.
    let fresh = Supertable::open(default_supertable_options().with_storage(Arc::clone(&plain)))
        .expect("fresh open");
    assert_eq!(fresh.manifest_id(), 4);
    assert_eq!(fresh.reader().expect("reader").n_superfiles(), 3);

    // The winner's superseded first list is exactly the steady-state
    // orphan this mechanism trades for crash recovery — present,
    // unreferenced, and left for GC.
    futures::executor::block_on(plain.get(&manifest_uri(2)))
        .expect("the skipped in-flight list remains for GC");
}

/// False-positive tripwire: ordinary staleness (a handle that last
/// refreshed before another handle's commits) must never trip the orphan
/// gate. Every id stays dense — a skip here would turn plain alternating
/// writers into steady orphan production.
#[test]
fn stale_handle_contention_keeps_manifest_ids_dense() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st_a = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");
    let st_b = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("open second handle");

    for round in 0..STALE_ALTERNATION_ROUNDS {
        let title_a = format!("round {round} from a");
        let title_b = format!("round {round} from b");
        commit_titles(&st_a, &[&title_a]);
        commit_titles(&st_b, &[&title_b]);
    }

    let expected_commits = 2 * STALE_ALTERNATION_ROUNDS as u64;
    let (pointer, _) = futures::executor::block_on(read_pointer(&*storage))
        .expect("read pointer")
        .expect("pointer present");
    assert_eq!(
        pointer.get_manifest_id(),
        expected_commits,
        "every commit takes the next id — no skips under plain staleness"
    );
    // Dense ids ⇔ exactly one list object per id 0..=N: any extra object
    // in manifest/ would be an orphan a false-positive skip left behind.
    let n_lists = std::fs::read_dir(dir.path().join("manifest"))
        .expect("manifest dir")
        .count();
    assert_eq!(
        n_lists as u64,
        expected_commits + 1,
        "create's id-0 list plus one list per commit, nothing else"
    );
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
