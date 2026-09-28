// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The deferred sweep a commit schedules: delete exactly what committed manifests dropped.
//!
//! A commit uploads its files first and swaps the manifest pointer last, so until the swap lands
//! those files are in storage and in no committed manifest. A sweep that lists storage and deletes
//! what the committed manifest does not name cannot tell them from garbage, and a commit slower
//! than its grace loses its own files. This sweep never lists. Each commit records the keys its
//! base manifest referenced and its successor dropped; the sweep deletes those, one grace later:
//!
//! ```text
//!   commit lands ── note(base, new, removed) ──► the handle's pending keys
//!                                                    │  a scheduled sweep takes them all
//!                                                    ▼
//!                        sleep the grace (readers pinned to `base` finish their fetches)
//!                                                    │
//!                        re-read the pointer ── fails? ──► delete nothing
//!                                                    ▼
//!                        skip keys the latest manifest names ──► DELETE the rest
//! ```
//!
//! An upload whose commit has not landed is in no committed manifest, so no commit can have
//! recorded it, and this sweep cannot delete it. Objects no commit ever published (a failed or
//! crashed commit, keys lost with a process that died during the grace) are orphans for
//! [`Supertable::gc`](crate::Supertable::gc), whose listing sweep `optimize` runs with a one-day
//! grace.

use std::{collections::HashSet, iter, mem, sync::Arc};

use futures::{StreamExt, stream};
use tokio::task::spawn_blocking;
use tracing::{debug, warn};

use crate::{
    storage::StorageError,
    supertable::{
        ManifestSnapshot,
        error::GcError,
        gc::refresh_to_committed,
        handle::SupertableInner,
        manifest::{SuperfileEntry, SuperfileUri, commit::manifest_uri},
        wal::persistence::WalStore,
    },
};

/// DELETEs one sweep keeps in flight. A compaction can drop thousands of superfiles in one commit;
/// issued one at a time that is minutes of wall clock, and unbounded it would crowd the
/// object-store client the table's queries share.
const RECLAIM_CONCURRENCY: usize = 32;

/// Most keys a handle holds for its next deferred sweep. Some commit paths never schedule one
/// (tombstone and term-index stamps, deletes mirrored onto the vector index), so a handle that
/// only takes those commits would grow the list for the life of the process. Past the cap a
/// commit's keys are not recorded; they become orphans for the listing sweep `optimize` runs,
/// which is where they went before this sweep existed.
const MAX_PENDING_SUPERSEDED: usize = 100_000;

/// Keys committed manifests dropped, waiting for a deferred sweep: removed superfiles and their
/// tombstone sidecars, superseded manifest lists and parts, and replaced blobs (vector state and
/// its sections, graph and router blobs, term stats, the term-index root).
#[derive(Debug, Default)]
pub(in crate::supertable) struct Superseded {
    keys: Vec<String>,
}

/// What a deferred sweep did, for its summary line and for tests.
#[derive(Debug, Default, PartialEq, Eq)]
pub(in crate::supertable) struct ReclaimReport {
    /// Objects the sweep deleted.
    pub(in crate::supertable) deleted: usize,
    /// Kept because the latest manifest references them again: content-addressed blobs can be
    /// published again under the same name.
    pub(in crate::supertable) still_referenced: usize,
    /// DELETEs the store refused. Each object is kept and left to the listing sweep.
    pub(in crate::supertable) failed: usize,
}

impl Superseded {
    /// What the commit from `base` to `new` dropped. `removed` is the superfile entries the commit
    /// took out, which the commit already has in hand; everything else is a diff of the two lists,
    /// so the cost is proportional to the parts, never to the superfiles.
    pub(in crate::supertable) fn between(
        base: &ManifestSnapshot,
        new: &ManifestSnapshot,
        removed: &[Arc<SuperfileEntry>],
    ) -> Self {
        let kept: HashSet<&str> = list_refs(new).collect();
        let mut keys = Vec::new();
        for entry in removed {
            keys.push(entry.storage_path());
            keys.push(WalStore::tombstones_path(entry.superfile_id));
        }
        if base.get_manifest_id() != new.get_manifest_id() {
            keys.push(manifest_uri(base.get_manifest_id()));
        }
        keys.extend(
            list_refs(base)
                .filter(|uri| !kept.contains(uri))
                .map(str::to_owned),
        );
        Self { keys }
    }

    pub(in crate::supertable) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub(in crate::supertable) fn len(&self) -> usize {
        self.keys.len()
    }

    pub(in crate::supertable) fn extend(&mut self, other: Self) {
        self.keys.extend(other.keys);
    }
}

/// Every object a manifest's list references besides the list itself (parts, their routing
/// siblings, and the list-level blobs): the same set the listing sweep keeps from the list, read
/// with no I/O.
fn list_refs(manifest: &ManifestSnapshot) -> impl Iterator<Item = &str> {
    let parts = manifest.get_all_list_entries().iter().flat_map(|entry| {
        iter::once(entry.uri.as_str())
            .chain(entry.routing.as_ref().map(|routing| routing.uri.as_str()))
    });
    let blobs = [
        manifest.slow_vector_state_blob().map(|(uri, _)| uri),
        manifest
            .slow_vector_state_centroids_blob()
            .map(|r| r.uri.as_str()),
        manifest
            .resident_vector_index_blob()
            .map(|r| r.uri.as_str()),
        manifest
            .slow_vector_state_centroid_graph_blob()
            .map(|r| r.uri.as_str()),
        manifest.term_stats_blob().map(|r| r.uri.as_str()),
        manifest.term_index_ref().map(|r| r.uri.as_str()),
    ]
    .into_iter()
    .flatten();
    parts.chain(blobs)
}

/// Delete `superseded` against the table's committed state now. The caller has already waited the
/// grace. Aborts, deleting nothing, if the pointer cannot be re-read: a keep-set that cannot be
/// verified is the input that deletes live data.
pub(in crate::supertable) async fn reclaim(
    inner: &SupertableInner,
    superseded: Superseded,
) -> Result<ReclaimReport, GcError> {
    let storage = inner.options.storage.clone().ok_or(GcError::NoStorage)?;
    refresh_to_committed(inner).await?;
    let latest = inner.manifest.load_full();
    let latest_list = manifest_uri(latest.get_manifest_id());
    let referenced: HashSet<&str> = list_refs(&latest).chain([latest_list.as_str()]).collect();

    let mut report = ReclaimReport::default();
    let (kept, targets): (Vec<String>, Vec<String>) = superseded
        .keys
        .into_iter()
        .partition(|key| referenced.contains(key.as_str()));
    report.still_referenced = kept.len();

    erase_cached_copies(inner, &targets).await;
    let results: Vec<(String, Result<(), StorageError>)> = stream::iter(targets)
        .map(|key| {
            let storage = Arc::clone(&storage);
            async move {
                let result = storage.delete(&key).await;
                (key, result)
            }
        })
        .buffer_unordered(RECLAIM_CONCURRENCY)
        .collect()
        .await;
    for (key, result) in results {
        match result {
            Ok(()) => report.deleted += 1,
            Err(error) => {
                report.failed += 1;
                warn!(object = %key, %error, "gc: failed to delete a superseded object; the next listing sweep reclaims it");
            }
        }
    }

    debug!(
        deleted = report.deleted,
        still_referenced = report.still_referenced,
        failed = report.failed,
        manifest_id = latest.get_manifest_id(),
        "gc: reclaimed superseded objects"
    );
    Ok(report)
}

/// Drop the disk-cache copy of each superfile about to be deleted. Unlinking thousands of files is
/// blocking filesystem work, so it runs on the blocking pool, off the runtime queries share.
async fn erase_cached_copies(inner: &SupertableInner, keys: &[String]) {
    let Some(cache) = inner.options.disk_cache.clone() else {
        return;
    };
    let uris: Vec<SuperfileUri> = keys
        .iter()
        .filter_map(|key| SuperfileUri::from_storage_path(key))
        .collect();
    if uris.is_empty() {
        return;
    }
    if let Err(error) = spawn_blocking(move || {
        for uri in &uris {
            cache.erase_superfile_local_copy(uri);
        }
    })
    .await
    {
        warn!(%error, "gc: disk-cache erase for superseded superfiles did not finish");
    }
}

impl SupertableInner {
    /// Record what the commit from `base` to `new` dropped, for the next deferred sweep on this
    /// handle. Called from the success arm of every loop that swaps the manifest pointer, with that
    /// attempt's own base, so a lost attempt records nothing.
    pub(in crate::supertable) fn note_superseded(
        &self,
        base: &ManifestSnapshot,
        new: &ManifestSnapshot,
        removed: &[Arc<SuperfileEntry>],
    ) {
        let superseded = Superseded::between(base, new, removed);
        if superseded.is_empty() {
            return;
        }
        let mut pending = self.superseded.lock().expect("superseded mutex poisoned");
        if pending.len() + superseded.len() > MAX_PENDING_SUPERSEDED {
            debug!(
                pending = pending.len(),
                dropped = superseded.len(),
                "gc: no deferred sweep has run on this handle for a while; leaving these to the listing sweep"
            );
            return;
        }
        pending.extend(superseded);
    }

    /// Everything recorded since the last deferred sweep was scheduled.
    pub(in crate::supertable) fn take_superseded(&self) -> Superseded {
        mem::take(&mut *self.superseded.lock().expect("superseded mutex poisoned"))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fmt,
        fs::{File, FileTimes, read_dir},
        future::Future,
        ops::Range,
        path::Path,
        sync::atomic::{AtomicBool, AtomicUsize, Ordering},
        thread,
        time::{Duration, SystemTime},
    };

    use async_trait::async_trait;
    use bytes::Bytes;
    use object_store::MultipartUpload;
    use tempfile::{TempDir, tempdir};
    use tokio::{runtime::Builder, sync::Notify};

    use super::*;
    use crate::{
        CompactionSettings, OptimizeOptions,
        storage::{LocalFsStorageProvider, ObjectMeta, StorageProvider},
        supertable::{
            Supertable,
            manifest::{commit::POINTER_PATH, list::RoutingRef, part::ContentHash},
            writer::{CommitListMetadata, persist_commit_async},
        },
        test_helpers::{build_title_batch, default_supertable_options},
    };

    /// How far a test moves an object's write time into the past: clear of any grace a test uses.
    const BACKDATE: Duration = Duration::from_secs(60 * 60);

    /// Superfiles in the compaction fixture: enough small ones for the planner to merge them.
    const COMPACTION_INPUTS: usize = 10;

    /// Local storage wrapped so a test can see and steer what the sweep does: it counts LISTs,
    /// can hold the next pointer swap until released (a commit stuck before its swap, as in the
    /// incident this sweep exists for), and can fail pointer reads.
    struct Hooked {
        inner: Arc<dyn StorageProvider>,
        lists: AtomicUsize,
        stall_next_swap: AtomicBool,
        swap_reached: Notify,
        swap_released: Notify,
        fail_pointer_reads: AtomicBool,
    }

    // `StorageProvider: Debug`; the wrapper is its inner provider.
    impl fmt::Debug for Hooked {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.inner.fmt(f)
        }
    }

    impl Hooked {
        fn over(dir: &Path) -> Arc<Self> {
            Arc::new(Self {
                inner: Arc::new(LocalFsStorageProvider::new(dir).expect("provider")),
                lists: AtomicUsize::new(0),
                stall_next_swap: AtomicBool::new(false),
                swap_reached: Notify::new(),
                swap_released: Notify::new(),
                fail_pointer_reads: AtomicBool::new(false),
            })
        }

        fn pointer_read(&self, uri: &str) -> Result<(), StorageError> {
            if uri == POINTER_PATH && self.fail_pointer_reads.load(Ordering::Relaxed) {
                return Err(StorageError::Permanent {
                    uri: uri.into(),
                    source: "pointer read refused by the test".into(),
                });
            }
            Ok(())
        }
    }

    #[async_trait]
    impl StorageProvider for Hooked {
        async fn head(&self, uri: &str) -> Result<ObjectMeta, StorageError> {
            self.pointer_read(uri)?;
            self.inner.head(uri).await
        }

        async fn get(&self, uri: &str) -> Result<(Bytes, ObjectMeta), StorageError> {
            self.pointer_read(uri)?;
            self.inner.get(uri).await
        }

        async fn get_if_none_match(
            &self,
            uri: &str,
            etag: &str,
        ) -> Result<Option<(Bytes, ObjectMeta)>, StorageError> {
            self.pointer_read(uri)?;
            self.inner.get_if_none_match(uri, etag).await
        }

        async fn get_range(&self, uri: &str, range: Range<u64>) -> Result<Bytes, StorageError> {
            self.inner.get_range(uri, range).await
        }

        async fn put_atomic(
            &self,
            uri: &str,
            bytes: Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: Bytes,
            expected_etag: Option<&str>,
        ) -> Result<Option<String>, StorageError> {
            if uri == POINTER_PATH && self.stall_next_swap.swap(false, Ordering::Relaxed) {
                self.swap_reached.notify_one();
                self.swap_released.notified().await;
            }
            self.inner.put_if_match(uri, bytes, expected_etag).await
        }

        async fn put_multipart(&self, uri: &str) -> Result<Box<dyn MultipartUpload>, StorageError> {
            self.inner.put_multipart(uri).await
        }

        async fn delete(&self, uri: &str) -> Result<(), StorageError> {
            self.inner.delete(uri).await
        }

        async fn list_with_prefix_metadata(
            &self,
            prefix: &str,
        ) -> Result<Vec<(String, ObjectMeta)>, StorageError> {
            self.lists.fetch_add(1, Ordering::Relaxed);
            self.inner.list_with_prefix_metadata(prefix).await
        }
    }

    fn block_on<F: Future>(future: F) -> F::Output {
        Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("runtime")
            .block_on(future)
    }

    fn table(storage: &Arc<Hooked>) -> Supertable {
        let storage: Arc<dyn StorageProvider> = Arc::clone(storage) as _;
        Supertable::create(default_supertable_options().with_storage(storage)).expect("create")
    }

    fn commit_title(st: &Supertable, title: &str) {
        let mut writer = st.writer().expect("writer");
        writer.append(&build_title_batch(&[title])).expect("append");
        writer.commit().expect("commit");
    }

    fn superfile_keys(st: &Supertable) -> Vec<String> {
        st.inner()
            .manifest
            .load_full()
            .get_all_superfiles()
            .iter()
            .map(|entry| entry.storage_path())
            .collect()
    }

    fn exists(dir: &TempDir, key: &str) -> bool {
        dir.path().join(key).exists()
    }

    fn backdate(dir: &TempDir, key: &str) {
        let stamp = SystemTime::now() - BACKDATE;
        File::options()
            .write(true)
            .open(dir.path().join(key))
            .expect("open to backdate")
            .set_times(FileTimes::new().set_accessed(stamp).set_modified(stamp))
            .expect("backdate");
    }

    /// The incident this sweep exists for. One commit is stuck between uploading its superfile
    /// and swapping the pointer, long enough that its upload is older than any grace, while the
    /// previous commit's deferred sweep runs. A listing sweep takes the upload for garbage; this
    /// sweep deletes only what committed manifests dropped, so the upload survives and the stuck
    /// commit lands on a file that is still there.
    #[test]
    fn a_commit_stuck_before_its_pointer_swap_keeps_its_upload() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = Arc::new(table(&storage));
        commit_title(&st, "alphatoken marker");
        commit_title(&st, "betatoken marker");
        let previous_commit = st.inner().take_superseded();
        assert!(
            !previous_commit.is_empty(),
            "each commit supersedes the list before it"
        );
        let committed = superfile_keys(&st);

        storage.stall_next_swap.store(true, Ordering::Relaxed);
        let stuck = Arc::clone(&st);
        let writer = thread::spawn(move || commit_title(&stuck, "gammatoken marker"));

        let report = block_on(async {
            storage.swap_reached.notified().await;
            let upload = read_dir(dir.path().join("data"))
                .expect("data dir")
                .map(|entry| {
                    format!(
                        "data/{}",
                        entry.expect("entry").file_name().to_string_lossy()
                    )
                })
                .find(|key| !committed.contains(key))
                .expect("the stuck commit uploaded its superfile");
            backdate(&dir, &upload);
            let report = reclaim(st.inner(), previous_commit).await;
            // Released before any assertion, so a failure cannot leave the commit stuck.
            let survived = exists(&dir, &upload);
            storage.swap_released.notify_one();
            let report = report.expect("reclaim");
            assert!(
                survived,
                "the stuck commit's upload was deleted: {report:?}"
            );
            report
        });
        writer.join().expect("stuck commit finished");

        for key in superfile_keys(&st) {
            assert!(
                exists(&dir, &key),
                "{key} referenced but missing: {report:?}"
            );
        }
        assert_eq!(superfile_keys(&st).len(), 3, "the stuck commit landed");
    }

    /// A compaction's inputs, their sidecars, and the manifest list and parts it replaced are
    /// deleted one sweep later, with no LIST; the merged output and the latest list stay.
    #[test]
    fn a_compaction_sweep_deletes_what_it_replaced_without_listing() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        for i in 0..COMPACTION_INPUTS {
            commit_title(&st, &format!("token{i} marker"));
        }
        let inputs = superfile_keys(&st);
        let _appends = st.inner().take_superseded();
        st.optimize(&OptimizeOptions::compact(CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        }))
        .expect("optimize");
        let merged = superfile_keys(&st);
        let dropped: Vec<&String> = inputs.iter().filter(|key| !merged.contains(key)).collect();
        assert!(!dropped.is_empty(), "the compaction merged something");
        let superseded = st.inner().take_superseded();
        let lists_before = storage.lists.load(Ordering::Relaxed);

        let report = block_on(reclaim(st.inner(), superseded)).expect("reclaim");

        assert_eq!(
            storage.lists.load(Ordering::Relaxed),
            lists_before,
            "no LIST"
        );
        for key in dropped {
            assert!(!exists(&dir, key), "{key} survived: {report:?}");
        }
        for key in &merged {
            assert!(exists(&dir, key), "{key} was deleted: {report:?}");
        }
        let latest = manifest_uri(st.inner().manifest.load_full().get_manifest_id());
        assert!(exists(&dir, &latest), "the latest list was deleted");
        assert_eq!(report.failed, 0, "{report:?}");
    }

    /// A content-addressed blob can be published again under the same name, so one the latest
    /// manifest names is kept even though a commit recorded it as replaced.
    #[test]
    fn a_superseded_key_the_latest_manifest_names_is_kept() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        commit_title(&st, "alphatoken marker");
        let latest_blob = st
            .inner()
            .manifest
            .load_full()
            .term_index_ref()
            .expect("the table has a term index")
            .uri
            .clone();

        let report = block_on(reclaim(
            st.inner(),
            Superseded {
                keys: vec![latest_blob.clone()],
            },
        ))
        .expect("reclaim");

        assert_eq!(report.still_referenced, 1, "{report:?}");
        assert!(
            exists(&dir, &latest_blob),
            "the latest term-index root was deleted"
        );
    }

    /// A sweep that cannot re-read the pointer cannot verify anything, so it deletes nothing.
    #[test]
    fn a_sweep_that_cannot_read_the_pointer_deletes_nothing() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        commit_title(&st, "alphatoken marker");
        commit_title(&st, "betatoken marker");
        let superseded = st.inner().take_superseded();
        let previous_list = superseded.keys.clone();
        storage.fail_pointer_reads.store(true, Ordering::Relaxed);

        let result = block_on(reclaim(st.inner(), superseded));

        assert!(result.is_err(), "the sweep must abort: {result:?}");
        for key in &previous_list {
            assert!(
                exists(&dir, key),
                "{key} deleted without a verified pointer"
            );
        }
    }

    /// A commit that loses the pointer race and retries records what its winning attempt
    /// replaced, never what its losing attempt would have.
    #[test]
    fn a_retried_commit_records_only_its_winning_attempt() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        commit_title(&st, "alphatoken marker");
        let stale_id = st.inner().manifest.load_full().get_manifest_id();
        let other = Supertable::open(
            default_supertable_options()
                .with_storage(Arc::clone(&storage) as Arc<dyn StorageProvider>),
        )
        .expect("open");
        commit_title(&other, "betatoken marker");
        let winning_base_id = other.inner().manifest.load_full().get_manifest_id();
        let _ = st.inner().take_superseded();

        // `st` still holds the manifest `other` already replaced: its first attempt loses.
        commit_title(&st, "gammatoken marker");
        let recorded = st.inner().take_superseded();

        assert!(
            recorded.keys.contains(&manifest_uri(winning_base_id)),
            "the winning attempt's base list is recorded: {recorded:?}"
        );
        assert!(
            !recorded.keys.contains(&manifest_uri(stale_id)),
            "the losing attempt's base list is recorded: {recorded:?}"
        );
    }

    /// A handle whose commits never schedule a sweep stops recording at the cap instead of
    /// growing without bound; what it skips is left to the listing sweep.
    #[test]
    fn recording_stops_at_the_cap() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        commit_title(&st, "alphatoken marker");
        let _ = st.inner().take_superseded();
        *st.inner().superseded.lock().expect("pending") = Superseded {
            keys: vec![String::new(); MAX_PENDING_SUPERSEDED],
        };

        commit_title(&st, "betatoken marker");

        assert_eq!(
            st.inner().take_superseded().len(),
            MAX_PENDING_SUPERSEDED,
            "a commit past the cap is not recorded"
        );
    }

    /// Commit a list-only stamp of `graph` as the resident vector-index blob, the way a drain
    /// batch stamps the graph it built.
    async fn stamp_graph(st: &Supertable, graph: &RoutingRef) {
        let inner = st.inner();
        let storage = inner.options.storage.clone().expect("storage-backed table");
        let committed = persist_commit_async(
            inner,
            storage,
            Vec::new(),
            &[],
            Vec::new(),
            Vec::new(),
            CommitListMetadata {
                graph_ref: Some(Some(graph.clone())),
                ..CommitListMetadata::empty()
            },
            Vec::new(),
        )
        .await
        .expect("stamp");
        inner.manifest.store(committed);
    }

    fn graph_ref(name: &str) -> RoutingRef {
        RoutingRef {
            uri: format!("graphs/{name}"),
            content_hash: ContentHash::of(name.as_bytes()),
        }
    }

    /// A commit's stamps already name its own new refs, so what it replaced must be measured from
    /// the committed manifest before them: the graph a drain batch replaces is recorded, the one it
    /// stamps is not.
    #[test]
    fn a_stamped_commit_records_the_blob_its_stamp_replaced() {
        let dir = tempdir().expect("tempdir");
        let storage = Hooked::over(dir.path());
        let st = table(&storage);
        commit_title(&st, "alphatoken marker");
        let (first, second) = (graph_ref("first"), graph_ref("second"));
        block_on(stamp_graph(&st, &first));
        let _ = st.inner().take_superseded();

        block_on(stamp_graph(&st, &second));
        let recorded = st.inner().take_superseded();

        assert!(
            recorded.keys.contains(&first.uri),
            "the replaced graph: {recorded:?}"
        );
        assert!(
            !recorded.keys.contains(&second.uri),
            "the stamped graph: {recorded:?}"
        );
    }
}
