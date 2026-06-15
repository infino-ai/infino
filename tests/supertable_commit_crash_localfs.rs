// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Crash safety for the persisted supertable commit path on
//! LocalFS.
//!
//! Parent spawns a child copy of this test binary with an
//! env var pointing at a temp directory + the kill point to
//! hit. The child drives one or two commits through a `CrashStorage`
//! wrapper that calls `std::process::abort()` immediately
//! after the underlying PUT lands (raising SIGABRT, which
//! drops the process without running any Drop impls — the
//! semantic equivalent of `kill -9` for our durability
//! claim).
//!
//! The parent then `Supertable::open`s the temp directory
//! and asserts the recovered state is one of two coherent
//! outcomes:
//!
//!   - The pointer file is missing or still references the
//!     prior committed `manifest_id` → open returns the prior
//!     state (or `PointerUnreadable` on a fresh supertable).
//!     Any superfile / manifest-part / manifest-list bytes
//!     written before the crash but never referenced by a
//!     committed pointer are **orphans**: tolerated by
//!     readers and GC'd by compaction.
//!   - The pointer file has been atomically replaced with
//!     the new version → open returns the new state. The
//!     crash happened AFTER the visibility barrier; the
//!     commit is durable.
//!
//! This is the load-bearing property of the
//! atomic-rename pointer commit: the pointer is the *only*
//! object that ever gets renamed, so the question "did the
//! commit succeed?" reduces to "did the pointer's rename
//! complete?" — a single atomic operation on LocalFS.
//!
//! Kill points exercised (one test function each):
//!
//! | Test fn                                                      | Crash point                                | Expected post-crash open state                    |
//! |--------------------------------------------------------------|---------------------------------------------|----------------------------------------------------|
//! | `crash_post_superfile_no_prior_commit_yields_pointer_unreadable` | After 1st superfile PUT, before list/pointer | `OpenError::PointerUnreadable`                     |
//! | `crash_post_list_no_prior_commit_yields_pointer_unreadable`    | After 1st list PUT, before pointer         | `OpenError::PointerUnreadable`                     |
//! | `crash_post_superfile_on_second_commit_yields_v1`                | First commit succeeds; 2nd commit's superfile PUT triggers | `manifest_id == 1` (v_prev), orphan v2 superfile    |
//! | `crash_post_list_on_second_commit_yields_v1`                   | First commit succeeds; 2nd commit's list PUT triggers   | `manifest_id == 1`, orphan v2 list + part         |
//! | `crash_post_pointer_on_second_commit_yields_v2`                | First commit succeeds; 2nd commit's pointer PUT triggers AFTER it lands | `manifest_id == 2` (commit was durable)           |
//!
//! LocalFS-only. The atomic-rename semantics hinge on local
//! filesystem behavior; s3s-fs's crash story is its own
//! concern.

#![deny(clippy::unwrap_used)]

use std::env;
use std::ops::Range;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use bytes::Bytes;

use infino::supertable::storage::{
    LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider,
};
use infino::supertable::{OpenError, Supertable};
use infino::test_helpers::{build_title_batch, default_supertable_options};

const ENV_DIR: &str = "INFINO_M12_CRASH_DIR";
const ENV_KILL_POINT: &str = "INFINO_M12_CRASH_KILL_POINT";

/// One named kill point. The child reads the env var and
/// configures the `CrashStorage` to match.
const KP_SEG_FIRST: &str = "seg-1";
const KP_LIST_FIRST: &str = "list-1";
const KP_SEG_SECOND: &str = "seg-2";
const KP_LIST_SECOND: &str = "list-2";
const KP_POINTER_SECOND: &str = "pointer-2";

/// Exit code used when the crash child finishes WITHOUT aborting —
/// signals a misconfigured kill point (distinct from a clean exit).
const MISCONFIGURED_KILL_POINT_EXIT_CODE: i32 = 2;

/// Storage wrapper that aborts the process after the N-th
/// PUT whose URI starts with `trigger_path_prefix` returns
/// success. Everything else is forwarded verbatim to the
/// inner `LocalFsStorageProvider`.
#[derive(Debug)]
struct CrashStorage {
    inner: LocalFsStorageProvider,
    trigger_path_prefix: String,
    trigger_after_nth_match: usize,
    matches_seen: AtomicUsize,
    abort_label: String,
}

impl CrashStorage {
    fn new(
        inner: LocalFsStorageProvider,
        trigger_path_prefix: impl Into<String>,
        trigger_after_nth_match: usize,
        abort_label: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            trigger_path_prefix: trigger_path_prefix.into(),
            trigger_after_nth_match,
            matches_seen: AtomicUsize::new(0),
            abort_label: abort_label.into(),
        }
    }

    /// Called from put_atomic / put_if_match after the
    /// inner provider returns. Aborts the process iff
    /// `is_match` AND `ok` AND this is the Nth such match.
    fn maybe_abort(&self, uri: &str, is_match: bool, ok: bool) {
        if !(is_match && ok) {
            return;
        }
        let n = self.matches_seen.fetch_add(1, Ordering::SeqCst) + 1;
        if n == self.trigger_after_nth_match {
            eprintln!(
                "CRASH-CHILD: aborting ({label}) after PUT uri={uri} match#={n}",
                label = self.abort_label
            );
            std::process::abort();
        }
    }
}

#[async_trait]
impl StorageProvider for CrashStorage {
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
        let is_match = uri.starts_with(&self.trigger_path_prefix);
        let result = self.inner.put_atomic(uri, bytes).await;
        self.maybe_abort(uri, is_match, result.is_ok());
        result
    }
    async fn put_if_match(
        &self,
        uri: &str,
        bytes: Bytes,
        expected_etag: Option<&str>,
    ) -> Result<Option<String>, StorageError> {
        let is_match = uri.starts_with(&self.trigger_path_prefix);
        let result = self.inner.put_if_match(uri, bytes, expected_etag).await;
        self.maybe_abort(uri, is_match, result.is_ok());
        result
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

/// Translate a kill point name into (trigger_path_prefix,
/// trigger_after_nth_match, n_commits). The child uses this
/// to configure `CrashStorage` and decide how many successful
/// commits to land before the crashing one.
fn kill_point_config(kp: &str) -> (&'static str, usize, usize) {
    match kp {
        KP_SEG_FIRST => ("data/", 1, 1),
        KP_LIST_FIRST => ("manifest-lists/", 1, 1),
        KP_SEG_SECOND => ("data/", 2, 2),
        KP_LIST_SECOND => ("manifest-lists/", 2, 2),
        KP_POINTER_SECOND => ("_supertable/current", 2, 2),
        other => panic!("unknown kill point {other}"),
    }
}

/// Child path: build a Supertable on `CrashStorage` and run
/// up to `n_commits` commits. The wrapper triggers
/// `std::process::abort()` mid-flight in the last commit
/// once the Nth matching PUT lands. The function never
/// returns normally — either it aborts (expected) or, if
/// the test configuration is wrong (Nth match doesn't fire),
/// the commit completes and the function exits cleanly.
/// The parent treats either as failure of expectations.
fn run_crash_child(dir: PathBuf, kill_point: &str) -> ! {
    let (prefix, nth, n_commits) = kill_point_config(kill_point);

    let local = LocalFsStorageProvider::new(&dir).expect("local fs provider");
    let wrapped = Arc::new(CrashStorage::new(local, prefix, nth, kill_point));
    let storage: Arc<dyn StorageProvider> = wrapped;

    let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
        .expect("create");

    for c in 1..=n_commits {
        let mut w = st.writer().expect("writer");
        let titles = match c {
            1 => vec!["first commit alpha"],
            2 => vec!["second commit beta"],
            _ => vec!["nth commit gamma"],
        };
        let batch = build_title_batch(&titles);
        w.append(&batch).expect("append");
        // commit may abort mid-flight; if it returns
        // we either misconfigured the kill point or
        // we're on a successful commit before the
        // crashing one.
        w.commit().expect("commit");
    }

    // If we reach here, the crash never fired. Print + exit
    // with a recognizable non-zero code so the parent can
    // distinguish "no crash fired" from "child aborted as
    // expected".
    eprintln!(
        "CRASH-CHILD: completed {n_commits} commits without aborting (kill_point={kill_point}) — \
         test configuration is wrong"
    );
    std::process::exit(MISCONFIGURED_KILL_POINT_EXIT_CODE);
}

/// Spawn a child copy of this test binary, filtered to a
/// single named test, with the kill-point env var set.
fn spawn_crash_child(test_name: &str, kill_point: &str) -> PathBuf {
    let tmp = tempfile::tempdir().expect("tempdir");
    // `into_path` lets the parent inspect the directory after
    // the child aborts (otherwise the TempDir guard would drop
    // it before our verification runs). It leaks the dir, but
    // that's fine for a single test invocation.
    let dir = tmp.keep();

    let exe = env::current_exe().expect("current_exe");
    let status = Command::new(&exe)
        .args(["--exact", "--test-threads=1", test_name])
        .env(ENV_DIR, &dir)
        .env(ENV_KILL_POINT, kill_point)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("spawn child");

    assert!(
        !status.success(),
        "child should have aborted (SIGABRT); got clean exit {status:?}"
    );

    dir
}

/// Parent-side dispatch: if the env var is set, become the
/// child. Otherwise return so the caller runs as parent.
fn dispatch_child_if_set() -> Option<()> {
    if let Ok(dir) = env::var(ENV_DIR) {
        let kp = env::var(ENV_KILL_POINT).expect("ENV_KILL_POINT must be set with ENV_DIR");
        run_crash_child(PathBuf::from(dir), &kp);
    }
    None
}

#[test]
fn crash_post_superfile_no_prior_commit_yields_pointer_unreadable() {
    if dispatch_child_if_set().is_some() {
        return; // unreachable; child never returns
    }
    let dir = spawn_crash_child(
        "crash_post_superfile_no_prior_commit_yields_pointer_unreadable",
        KP_SEG_FIRST,
    );

    // Parent verifies. The crash fired after the first
    // superfile PUT, before any manifest writes. No pointer
    // exists yet → open must return PointerUnreadable.
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("must reject post-crash state with no pointer");
    assert!(
        matches!(err, OpenError::ManifestLoadError(_)),
        "expected PointerUnreadable, got {err:?}"
    );

    // The orphan superfile file is present and ignored — the
    // superfile is just bytes under data/; readers don't
    // discover it without a committed manifest list.
    let data_dir = dir.join("data");
    let n_orphans = std::fs::read_dir(&data_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_orphans >= 1,
        "orphan superfile must be present on disk; found {n_orphans} in {data_dir:?}"
    );
}

#[test]
fn crash_post_list_no_prior_commit_yields_pointer_unreadable() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_list_no_prior_commit_yields_pointer_unreadable",
        KP_LIST_FIRST,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let err = Supertable::open(default_supertable_options().with_storage(storage))
        .expect_err("must reject post-crash state with no pointer");
    assert!(
        matches!(err, OpenError::ManifestLoadError(_)),
        "expected PointerUnreadable, got {err:?}"
    );

    // The orphan manifest list is on disk but unreferenced.
    let lists_dir = dir.join("manifest-lists");
    let n_orphan_lists = std::fs::read_dir(&lists_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_orphan_lists >= 1,
        "orphan manifest list must be present; found {n_orphan_lists} in {lists_dir:?}"
    );
}

#[test]
fn crash_post_superfile_on_second_commit_yields_v1() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_superfile_on_second_commit_yields_v1",
        KP_SEG_SECOND,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v1");
    assert_eq!(consumer.manifest_id(), 1, "must recover at v1");
    assert_eq!(
        consumer.reader().n_superfiles(),
        1,
        "v1 has exactly the first commit's superfile; v2's orphan superfile is invisible"
    );
}

#[test]
fn crash_post_list_on_second_commit_yields_v1() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child("crash_post_list_on_second_commit_yields_v1", KP_LIST_SECOND);

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v1");
    assert_eq!(consumer.manifest_id(), 1);
    assert_eq!(consumer.reader().n_superfiles(), 1);

    // Orphan v2 manifest list and v2 part are on disk —
    // tolerated here; compaction GCs them later.
    let lists_dir = dir.join("manifest-lists");
    let n_lists = std::fs::read_dir(&lists_dir)
        .map(|rd| rd.count())
        .unwrap_or(0);
    assert!(
        n_lists >= 2,
        "v1 list + orphan v2 list both on disk; found {n_lists}"
    );
}

#[test]
fn crash_post_pointer_on_second_commit_yields_v2() {
    if dispatch_child_if_set().is_some() {
        return;
    }
    let dir = spawn_crash_child(
        "crash_post_pointer_on_second_commit_yields_v2",
        KP_POINTER_SECOND,
    );

    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(&dir).expect("provider"));
    let consumer =
        Supertable::open(default_supertable_options().with_storage(storage)).expect("open at v2");
    assert_eq!(
        consumer.manifest_id(),
        2,
        "pointer rename completed before crash → commit is durable"
    );
    assert_eq!(
        consumer.reader().n_superfiles(),
        2,
        "v2 sees both commits' superfiles"
    );
}
