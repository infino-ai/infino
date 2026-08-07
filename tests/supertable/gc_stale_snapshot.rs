// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A gc sweep must decide liveness from the committed manifest, never from whatever snapshot the
//! calling handle happens to hold. A handle that has not refreshed deletes superfiles another
//! handle committed after its snapshot, and since the manifest still references them the loss goes
//! unnoticed until a later read or compaction fails with `not found`, permanently.
//!
//! The tests below cover the two ways a keep-set goes stale (a handle that never refreshed, and a
//! commit landing mid-sweep), what happens when liveness cannot be resolved at all, and the request
//! shape of a sweep, since the sweep runs per table per grace window against object storage.

use std::{
    fs::FileTimes,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, SystemTime},
};

use async_trait::async_trait;
use bytes::Bytes;
use infino::{
    storage::{LocalFsStorageProvider, ObjectMeta, StorageError, StorageProvider},
    supertable::{
        Supertable,
        manifest::commit::{MANIFEST_PARTS_DIR, POINTER_PATH, manifest_uri},
    },
    test_helpers::{
        build_title_batch, default_supertable_options,
        fault_storage::{FaultOp, FaultStorage},
    },
};
use tempfile::TempDir;
use tokio::sync::oneshot;

/// One superfile per manifest part, so every commit splits a part and a few commits build a
/// many-part manifest without writing a large table.
const SUPERFILES_PER_PART: u64 = 1;

/// Commits in the multi-part fixture, and therefore parts.
const MULTI_PART_COMMITS: usize = 4;

/// How far [`backdate_superfiles`] moves a superfile's mtime into the past. Anything clear of a
/// `Duration::ZERO` cutoff works; an hour leaves no doubt.
const BACKDATE: Duration = Duration::from_secs(60 * 60);

/// Repeats armed on a fault the test needs to stay broken for a whole call rather than heal on a
/// retry.
const FANOUT_FAULTS: usize = 64;

/// Pointer reads one sweep may issue: one per liveness resolve, and it resolves twice.
const MAX_SWEEP_POINTER_READS: usize = 2;

fn commit_titles(st: &Supertable, titles: &[&str]) {
    let mut w = st.writer().expect("writer");
    w.append(&build_title_batch(titles)).expect("append");
    w.commit().expect("commit");
}

/// Move every superfile's mtime `BACKDATE` into the past, so the sweep's
/// age check treats it as an old object rather than a fresh one. LocalFs
/// reports mtime as `ObjectMeta::last_modified`.
fn backdate_superfiles(data_dir: &std::path::Path) {
    let stamp = SystemTime::now() - BACKDATE;
    let times = FileTimes::new().set_accessed(stamp).set_modified(stamp);
    let Ok(entries) = std::fs::read_dir(data_dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        if !entry.file_name().to_string_lossy().ends_with(".sf.parquet") {
            continue;
        }
        let file = std::fs::File::options()
            .write(true)
            .open(entry.path())
            .expect("open superfile to backdate");
        file.set_times(times).expect("backdate superfile mtime");
    }
}

/// Every `*.sf.parquet` under the table's data directory.
fn superfiles(dir: &std::path::Path) -> Vec<String> {
    let data_dir = dir.join("data");
    let Ok(entries) = std::fs::read_dir(&data_dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".sf.parquet"))
        .collect();
    names.sort();
    names
}

#[test]
fn deferred_gc_from_an_earlier_handle_deletes_a_later_handles_superfile() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = || default_supertable_options().with_storage(Arc::clone(&storage));

    // Pass one: open, commit, done. Its manifest swap is what arms the deferred
    // reclaim, which captures this handle and keeps it alive past the pass.
    let pass_one = Supertable::create(options()).expect("create");
    commit_titles(&pass_one, &["alphatoken marker"]);
    assert_eq!(
        superfiles(dir.path()).len(),
        1,
        "one superfile after the first pass"
    );

    // Pass two: a fresh handle, as a process that reconnects per pass gets. It
    // commits a superfile that pass one's captured view will never see. Pass one
    // is no longer writing — only its armed reclaim outlives it.
    let pass_two = Supertable::open(options()).expect("open");
    commit_titles(&pass_two, &["betatoken marker"]);
    let live = superfiles(dir.path());
    assert_eq!(live.len(), 2, "two superfiles after the second pass");

    // Pass one's deferred sweep fires. `Duration::ZERO` stands in for the
    // wall-clock wait: after the real grace the newer superfile is older than
    // the cutoff and equally eligible.
    let report = pass_one.gc(Duration::ZERO).expect("gc");

    let surviving = superfiles(dir.path());
    assert_eq!(
        surviving, live,
        "the earlier handle's sweep deleted a superfile the current manifest \
         still references; deleted {} object(s), kept {} as live",
        report.objects_deleted, report.objects_skipped_live,
    );
}

// Four commits at one superfile per part, a fifth from a second handle, and an orphan planted
// under `data/`.
//  - the sweep lists `data/` only when the manifest's superfile membership is fully resident.
//  - a partial part view drops the prefix silently: nothing fails, reclaim just stops.
//  - so the orphan disappearing is what shows the refreshed view was complete.
// This pins the many-part shape, which the unit fixtures never reach.
#[test]
fn a_refreshed_multi_part_manifest_still_reclaims_under_the_data_prefix() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let options = || {
        default_supertable_options()
            .with_storage(Arc::clone(&storage))
            .with_target_superfiles_per_part(SUPERFILES_PER_PART)
    };

    let first = Supertable::create(options()).expect("create");
    for i in 0..MULTI_PART_COMMITS {
        commit_titles(&first, &[&format!("parttoken marker {i}")]);
    }
    let live = superfiles(dir.path());
    assert_eq!(live.len(), MULTI_PART_COMMITS, "one superfile per commit");

    // A second handle commits, so the sweeper has to load the newest manifest rather than reuse
    // its own view: it inherits the parts it already holds and fetches the rest.
    let second = Supertable::open(options()).expect("open");
    commit_titles(&second, &["parttoken marker last"]);

    let orphan = dir.path().join("data").join("seg-orphan.sf.parquet");
    std::fs::write(&orphan, b"not a superfile").expect("plant orphan");

    let report = first.gc(Duration::ZERO).expect("gc");

    assert!(
        !orphan.exists(),
        "the data prefix was not swept, so the manifest's part view was \
         incomplete after the refresh: {report:?}"
    );
    let surviving = superfiles(dir.path());
    assert_eq!(
        surviving.len(),
        MULTI_PART_COMMITS + 1,
        "every referenced superfile across every part survives: {report:?}"
    );
}

/// Runs a caller-supplied commit from inside a listing call, putting it in the window between the
/// sweep's listing and its deletes without any timing assumption. The commit runs on a plain
/// thread, which has no ambient runtime, so the sync commit path builds its own.
struct CommitDuringList {
    inner: Arc<dyn StorageProvider>,
    /// Fires on the first listing after [`Self::arm`]; later listings pass straight through.
    commit: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    /// Opening a table and committing both list prefixes too, so an always-live hook would fire
    /// before the sweep and leave the commit inside the keep-set, testing nothing.
    armed: AtomicBool,
    lists: AtomicUsize,
}

// `StorageProvider: Debug`, and a boxed closure isn't. The wrapper's identity
// is its inner provider.
impl std::fmt::Debug for CommitDuringList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommitDuringList")
            .field("inner", &self.inner)
            .field("lists", &self.lists)
            .finish()
    }
}

impl CommitDuringList {
    fn wrap(inner: Arc<dyn StorageProvider>, commit: Box<dyn FnOnce() + Send>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            commit: Mutex::new(Some(commit)),
            armed: AtomicBool::new(false),
            lists: AtomicUsize::new(0),
        })
    }

    fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
        self.lists.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl StorageProvider for CommitDuringList {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.inner.get(uri).await
    }

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        self.inner.get_if_none_match(uri, etag).await
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

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        let entries = self.inner.list_with_prefix_metadata(prefix).await?;
        self.lists.fetch_add(1, Ordering::Relaxed);
        if !self.armed.load(Ordering::Relaxed) {
            return Ok(entries);
        }
        let pending = self
            .commit
            .lock()
            .expect("commit slot mutex poisoned")
            .take();
        if let Some(commit) = pending {
            let (tx, rx) = oneshot::channel();
            std::thread::spawn(move || {
                commit();
                let _ = tx.send(());
            });
            rx.await.expect("mid-sweep commit thread");
        }
        Ok(entries)
    }
}

// A second handle commits while the sweep sits between its listing and its deletes.
//  - the keep-set was built before that commit, so the new superfile is a delete candidate.
//  - the hook backdates it past the cutoff, or the too-new guard would keep it and the test would
//    pass with or without the re-resolve.
//  - only the re-resolve before deleting can put it back.
// A sweeping handle and a writing handle over one table is a maintenance pass beside an ingest one.
#[test]
fn a_commit_landing_between_the_listing_and_the_deletes_loses_nothing() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

    let sweeper_storage = Arc::clone(&local);
    let writer = Arc::new(
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&local)))
            .expect("create"),
    );
    commit_titles(&writer, &["alphatoken marker"]);

    let interleaved = Arc::clone(&writer);
    let data_dir = dir.path().join("data");
    let hooked = CommitDuringList::wrap(
        sweeper_storage,
        Box::new(move || {
            commit_titles(&interleaved, &["betatoken marker"]);
            backdate_superfiles(&data_dir);
        }),
    );
    let sweeper = Supertable::open(
        default_supertable_options().with_storage(Arc::<CommitDuringList>::clone(&hooked)),
    )
    .expect("open");

    hooked.arm();
    let report = sweeper.gc(Duration::ZERO).expect("gc");
    assert!(
        hooked.lists.load(Ordering::Relaxed) > 0,
        "the sweep must have listed at least one prefix"
    );
    assert!(
        hooked
            .commit
            .lock()
            .expect("commit slot mutex poisoned")
            .is_none(),
        "the interleaved commit must have fired inside the sweep"
    );

    assert_eq!(
        superfiles(dir.path()).len(),
        2,
        "a superfile committed mid-sweep was deleted: {report:?}"
    );
    let reader = writer.reader().expect("reader");
    assert_eq!(
        reader.n_superfiles(),
        2,
        "the manifest still references both superfiles: {report:?}"
    );
}

// Two commits leave the first manifest list unreferenced, then the pointer read is faulted.
//  - the superseded list is a genuine orphan, so it witnesses whether the aborted sweep deleted
//    anything at all.
//  - a keep-set that cannot be verified must stop the sweep, not fall back to the cached snapshot.
//  - clearing the fault must let the next sweep reclaim, so the refusal cannot latch.
// This is a deliberate change: a transient pointer read used to be invisible to a sweep that never
// read the pointer, and now fails it, which `optimize()` surfaces as `OptimizeError::Gc`. A failed
// maintenance pass is recoverable; a deleted live superfile is not.
#[test]
fn a_failed_pointer_read_aborts_the_sweep_without_deleting_anything() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let faults = FaultStorage::wrap(local);
    let st = Supertable::create(
        default_supertable_options().with_storage(Arc::<FaultStorage>::clone(&faults)),
    )
    .expect("create");

    commit_titles(&st, &["alphatoken marker"]);
    commit_titles(&st, &["betatoken marker"]);
    let orphan = dir.path().join(manifest_uri(1));
    assert!(orphan.exists(), "the superseded manifest list is on disk");

    faults.fail(FaultOp::Get, POINTER_PATH, FANOUT_FAULTS);
    st.gc(Duration::ZERO)
        .expect_err("an unresolvable keep-set must fail the sweep");
    assert!(
        faults.fired() > 0,
        "the sweep never read the pointer, so this proves nothing about \
         resolving liveness"
    );
    assert!(
        orphan.exists(),
        "the aborted sweep deleted an object anyway"
    );
    assert_eq!(
        superfiles(dir.path()).len(),
        2,
        "the aborted sweep deleted a superfile"
    );

    // The failure is transient, not latching: the next sweep reclaims normally.
    faults.clear();
    st.gc(Duration::ZERO).expect("gc once the fault clears");
    assert!(!orphan.exists(), "the healthy sweep reclaims the orphan");
    assert_eq!(
        superfiles(dir.path()).len(),
        2,
        "both referenced superfiles survive the healthy sweep"
    );
}

// A deleted pointer means the table was dropped and purged, so the sweep has no keep-set to work
// from and must abort. Reclaiming what the purge left behind belongs to the purge, which knows the
// whole subtree.
#[test]
fn a_vanished_pointer_aborts_the_sweep() {
    let dir = TempDir::new().expect("tempdir");
    let storage: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let st =
        Supertable::create(default_supertable_options().with_storage(storage)).expect("create");
    commit_titles(&st, &["alphatoken marker"]);

    std::fs::remove_file(dir.path().join(POINTER_PATH)).expect("purge the pointer");

    st.gc(Duration::ZERO)
        .expect_err("a purged table's sweep must abort, not sweep a stale view");
    assert_eq!(
        superfiles(dir.path()).len(),
        1,
        "the aborted sweep deleted a superfile"
    );
}

/// Counts reads by URI fragment, so a sweep's request shape can be asserted rather than assumed.
#[derive(Debug)]
struct CountingProxy {
    inner: Arc<dyn StorageProvider>,
    pointer_reads: AtomicUsize,
    part_reads: AtomicUsize,
}

impl CountingProxy {
    fn wrap(inner: Arc<dyn StorageProvider>) -> Arc<Self> {
        Arc::new(Self {
            inner,
            pointer_reads: AtomicUsize::new(0),
            part_reads: AtomicUsize::new(0),
        })
    }

    fn note(&self, uri: &str) {
        if uri.contains(POINTER_PATH) {
            self.pointer_reads.fetch_add(1, Ordering::Relaxed);
        }
        if uri.contains(MANIFEST_PARTS_DIR) {
            self.part_reads.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.pointer_reads.store(0, Ordering::Relaxed);
        self.part_reads.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl StorageProvider for CountingProxy {
    async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
        self.inner.head(uri).await
    }

    async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
        self.note(uri);
        self.inner.get(uri).await
    }

    async fn get_if_none_match(
        &self,
        uri: &str,
        etag: &str,
    ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
        self.note(uri);
        self.inner.get_if_none_match(uri, etag).await
    }

    async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
        self.note(uri);
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

    async fn list_with_prefix_metadata(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
        self.inner.list_with_prefix_metadata(prefix).await
    }
}

// A many-part table swept through a counting provider, with the pointer unchanged throughout.
//  - refreshing is only affordable because it starts from the handle's snapshot and inherits
//    already-loaded parts, so re-fetching the part fan is the regression to catch.
//  - a sweep runs once per table per grace window, so an amplifier here is invisible to any query
//    bench.
//  - the probe count is bounded on both sides: reading no pointer at all would satisfy an upper
//    bound while being the exact bug this all exists to prevent.
#[test]
fn a_sweep_on_an_unchanged_pointer_reads_the_pointer_and_no_parts() {
    let dir = TempDir::new().expect("tempdir");
    let local: Arc<dyn StorageProvider> =
        Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
    let counting = CountingProxy::wrap(local);
    let st = Supertable::create(
        default_supertable_options()
            .with_storage(Arc::<CountingProxy>::clone(&counting))
            .with_target_superfiles_per_part(SUPERFILES_PER_PART),
    )
    .expect("create");
    for i in 0..MULTI_PART_COMMITS {
        commit_titles(&st, &[&format!("parttoken marker {i}")]);
    }

    // Count only the sweep, not the table setup that precedes it.
    counting.reset();
    let report = st.gc(Duration::ZERO).expect("gc");

    assert_eq!(
        counting.part_reads.load(Ordering::Relaxed),
        0,
        "the sweep re-fetched manifest parts instead of inheriting them: {report:?}"
    );
    let probes = counting.pointer_reads.load(Ordering::Relaxed);
    assert!(
        (1..=MAX_SWEEP_POINTER_READS).contains(&probes),
        "expected one pointer read per liveness resolve, got {probes}",
    );
}
