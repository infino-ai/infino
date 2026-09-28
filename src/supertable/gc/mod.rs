// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, SystemTime},
};

use tracing::{debug, error, info, warn};

use crate::{
    runtime_bridge::bridge_on_runtime,
    storage::{StorageError, StorageProvider},
    supertable::{
        ManifestSnapshot, Supertable,
        error::GcError,
        handle::SupertableInner,
        manifest::{
            SUPERFILE_DATA_DIR, SUPERFILE_KEY_SUFFIX, SuperfileUri,
            commit::{MANIFEST_DIR, MANIFEST_PARTS_DIR, POINTER_PATH, manifest_uri},
            term_index::{self, STORAGE_PREFIX as TERM_INDEX_STORAGE_PREFIX},
            term_stats::STORAGE_PREFIX as TERM_STATS_STORAGE_PREFIX,
        },
        slow_vector_state::{self, STORAGE_PREFIX as SLOW_VECTOR_STATE_STORAGE_PREFIX},
        wal::persistence::{SUPERFILES_DIR, WalStore},
    },
};

/// Minimum age of a storage object before [`gc_storage_sweep_for_inner`] may
/// delete it. Sized so snapshot-pinned readers can finish cold fetches against
/// superseded superfiles after a manifest swap.
#[cfg_attr(test, allow(dead_code))]
pub(crate) const DEFAULT_SUPERFILE_RECLAIM_GRACE: Duration = Duration::from_secs(5 * 60);

/// Most referenced-but-unlisted superfiles one sweep confirms with a HEAD. A table that has lost
/// more than this is already reported loudly, and a HEAD each for thousands would stall the sweep.
const MAX_MISSING_SUPERFILE_PROBES: usize = 64;

/// A superfile deleted younger than this gets its own `info` line. Routine reclaim removes
/// superseded files long after they were written, so a young one is the rare, risky case: a file
/// whose commit was still in flight, or never landed. Three of the deferred sweep's grace windows
/// keeps the line for a file taken the moment it cleared the grace, without logging every routine
/// delete.
const YOUNG_SUPERFILE_DELETE_AGE: Duration =
    Duration::from_secs(3 * DEFAULT_SUPERFILE_RECLAIM_GRACE.as_secs());

/// Which caller ran a sweep, carried on every line it logs so a deleted object can be traced back
/// to the path that deleted it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum GcTrigger {
    /// [`Supertable::gc`], including the one `optimize` runs. The only trigger that runs the
    /// missing-superfile check.
    Explicit,
    /// The deferred sweep a commit schedules after dropping superfile references. Skips the
    /// missing-superfile check, which would otherwise run once per commit.
    DeferredReclaim,
}

impl GcTrigger {
    fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::DeferredReclaim => "deferred_reclaim",
        }
    }
}

/// Outcome of a [`crate::Supertable::gc`] sweep: what was reclaimed and what was
/// intentionally kept.
#[derive(Debug, Default, Clone)]
pub struct GcReport {
    /// Orphaned objects deleted.
    pub objects_deleted: u64,
    /// Total bytes reclaimed by the deleted objects.
    pub bytes_freed: u64,
    /// Objects kept because they are still referenced by the live set.
    pub objects_skipped_live: u64,
    /// Objects kept because they are younger than the safety gap.
    pub objects_skipped_too_new: u64,
    /// Objects that could not be deleted (left for a later sweep).
    pub delete_errors: u64,
}

/// Every storage key this manifest version references, and whether its superfile membership was
/// fully resident. Anything absent from the returned set is an orphan as far as the caller is
/// concerned, so a key that belongs here and is missed, gets deleted.
fn build_live_set(manifest: &ManifestSnapshot) -> (HashSet<String>, bool) {
    let mut live = HashSet::new();

    // The pointer, and the one manifest list it names. Superseded lists are left out on purpose:
    // being unreferenced here is exactly what makes them reclaimable.
    live.insert(POINTER_PATH.to_string());
    live.insert(manifest_uri(manifest.manifest_id));

    // The part fan this list is built from, plus each part's routing sibling where it has one.
    for entry in manifest.get_all_list_entries() {
        live.insert(entry.uri.clone());
        if let Some(routing) = &entry.routing {
            live.insert(routing.uri.clone());
        }
    }

    // Every superfile, but only when the parts are all loaded. A partial view names some of the
    // superfiles and no more, so the caller must skip `data/` entirely rather than treat the ones
    // it cannot see as orphans. That is what the flag carries.
    let superfiles_complete = if let Some(superfiles) = manifest.complete_flat_superfiles() {
        for sf in superfiles {
            live.insert(sf.storage_path());
        }
        true
    } else {
        false
    };

    // The slow-CAS state blob and its centroid section, read straight off the list refs with no
    // fetch. Older drains are absent from the current list and age out past the safety gap.
    if let Some((uri, _)) = manifest.slow_vector_state_blob() {
        live.insert(uri.to_owned());
    }
    if let Some(centroids) = manifest.slow_vector_state_centroids_blob() {
        live.insert(centroids.uri.clone());
    }
    if let Some(graphs) = manifest.resident_vector_index_blob() {
        live.insert(graphs.uri.clone());
    }
    if let Some(centroid_graph) = manifest.slow_vector_state_centroid_graph_blob() {
        live.insert(centroid_graph.uri.clone());
    }
    // The global term-stats sidecar, same list-ref discipline: the current
    // artifact is live; superseded generations age out past the safety gap.
    if let Some(stats) = manifest.term_stats_blob() {
        live.insert(stats.uri.clone());
    }

    // Each resident superfile's tombstone sidecar. `superfiles/` is swept whatever the flag says,
    // so these have to be named here or a sidecar past the gap is deleted and its deleted rows
    // come back. The superfile paths repeat what the complete view above already inserted.
    for sf in manifest.get_all_superfiles() {
        live.insert(sf.storage_path());
        live.insert(WalStore::tombstones_path(sf.superfile_id));
    }

    (live, superfiles_complete)
}

impl Supertable {
    /// Delete orphaned storage objects left by compaction or interrupted
    /// writes. Only objects older than `safety_gap` are removed, so a
    /// concurrent reader or writer is never raced. Requires durable storage.
    #[doc(alias = "vacuum")]
    pub fn gc(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        bridge_on_runtime(self.gc_async(safety_gap), &self.inner().query_runtime())
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(name = "gc", skip_all, fields(role = self.role().as_str()))
    )]
    pub(crate) async fn gc_async(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        gc_storage_sweep_for_inner(self.inner(), safety_gap, GcTrigger::Explicit).await
    }
}

/// Everything one manifest version references, with the `manifest_id` it came from and whether its
/// superfile membership was fully resident (see [`build_live_set`]).
struct LiveSet {
    uris: HashSet<String>,
    superfiles_complete: bool,
    manifest_id: u64,
}

/// Advance the handle to the committed manifest, so the keep-set built next describes the table
/// rather than one handle's memory of it. A superfile another handle committed after that snapshot
/// is missing from the cached view, and a sweep built on it deletes a file the manifest still
/// references, which nothing notices until a later read or compaction fails with `not found`.
///
/// Costs one conditional pointer GET while the pointer is unchanged, and inherits already-loaded
/// parts when it has moved.
///
/// Any failure aborts the sweep rather than falling back to the cached snapshot, because a keep-set
/// that cannot be verified is the input that deletes live data. That includes `PointerVanished`,
/// where the table was dropped and purged and reclaiming the remains belongs to the purge.
async fn refresh_to_committed(inner: &SupertableInner) -> Result<(), GcError> {
    inner.refresh().await.map(|_advanced| ()).map_err(|error| {
        GcError::Storage(StorageError::Permanent {
            uri: POINTER_PATH.to_string(),
            source: Box::new(error),
        })
    })
}

/// Keep-set for the manifest this handle currently holds. Callers run [`refresh_to_committed`]
/// first; this reads no pointer of its own.
///
/// Not cheap on a table carrying slow-CAS vector state: hydrating the pending drain re-fetches that
/// blob and re-hashes it, which is multi-GiB work on a large table. Build it once per manifest
/// version, never speculatively.
async fn live_set(
    inner: &SupertableInner,
    storage: &Arc<dyn StorageProvider>,
) -> Result<LiveSet, GcError> {
    let manifest = inner.manifest.load_full();
    let (mut uris, superfiles_complete) = build_live_set(&manifest);

    if let Some((uri, hash)) = manifest.slow_vector_state_blob() {
        // An unreadable slow-state blob is a permanent storage-level failure
        // on that URI (missing, corrupt, or hash-mismatched bytes) — surface
        // it through the existing `Storage` variant rather than a dedicated
        // public error variant.
        let state = slow_vector_state::load_full_state(storage.as_ref(), uri, &hash)
            .await
            .map_err(|error| {
                GcError::Storage(StorageError::Permanent {
                    uri: uri.to_string(),
                    source: Box::new(error),
                })
            })?;
        if let Some(pending) = state.pending_drain {
            uris.extend(pending.entries.iter().map(|entry| entry.storage_path()));
        }
    }

    // The term index: the root the list references, and every slice that
    // root names — which takes reading the root, one small object. A root
    // that cannot be read is a permanent failure on that URI, surfaced the
    // same way as an unreadable slow-state blob: deleting slices we could
    // not enumerate would be exactly the loss this sweep exists to prevent.
    if let Some(reference) = manifest.term_index_ref() {
        uris.insert(reference.uri.clone());
        let root = term_index::load_root(storage.as_ref(), reference)
            .await
            .map_err(|error| {
                GcError::Storage(StorageError::Permanent {
                    uri: reference.uri.clone(),
                    source: Box::new(error),
                })
            })?;
        for segment in &root.segments {
            for slice in &segment.slices {
                uris.insert(term_index::slice_uri(&slice.content_hash));
            }
        }
    }
    Ok(LiveSet {
        uris,
        superfiles_complete,
        manifest_id: manifest.manifest_id,
    })
}

/// Delete storage objects not referenced by the current manifest once they are
/// older than `safety_gap`. Supersedes inline post-commit deletes so readers
/// pinned to an older snapshot cannot lose bytes mid-fetch.
///
/// Listing and deleting are not atomic against a commit, so liveness is resolved twice — once
/// before listing and once more before deleting — and an object referenced by either version is
/// kept. Unioning the two keep-sets rather than replacing the first is the point: a commit that
/// lands mid-sweep would otherwise fall between them.
///
/// The same listing doubles as an integrity check: a superfile the committed manifest references
/// but storage does not hold is data already lost, and every read or compaction touching it fails
/// from then on. Only an explicit sweep runs it ([`Supertable::gc`], which `optimize` calls once per
/// table per maintenance pass), and only over a fully resident membership, the one case `data/` is
/// listed. The deferred sweep every commit schedules skips it, so ingest never pays for it, not even
/// on a table already broken. On a healthy table it costs one pass over the keep-set and a counter:
///
/// ```text
///   list every prefix ── count the referenced superfiles `data/` returned
///        │                 (a counter, no allocation; equal to the referenced total when healthy)
///        ▼ came up short
///   re-list `data/`, HEAD each referenced key it lacks          (rare path, probes capped)
///        │
///        ▼
///   re-read the pointer ── manifest moved? keep only keys the newer one still references
///        │
///        ▼
///   log the survivors at `error`
/// ```
///
/// The HEAD runs before that pointer re-read on purpose. A key the HEAD found absent, and that a
/// manifest read afterwards still references, was missing while committed: that is loss. Probing
/// after the re-read would open a window in which a compaction drops the key, a concurrent sweep
/// reclaims it, and this sweep reports the legitimate reclaim as loss.
pub(super) async fn gc_storage_sweep_for_inner(
    inner: &SupertableInner,
    safety_gap: Duration,
    trigger: GcTrigger,
) -> Result<GcReport, GcError> {
    let storage = inner.options.storage.clone().ok_or(GcError::NoStorage)?;

    refresh_to_committed(inner).await?;

    let LiveSet {
        uris: live_uris,
        superfiles_complete,
        manifest_id: live_manifest_id,
    } = live_set(inner, &storage).await?;

    let cutoff = SystemTime::now()
        .checked_sub(safety_gap)
        .unwrap_or(SystemTime::UNIX_EPOCH);

    let mut report = GcReport::default();

    let mut prefixes = vec![
        MANIFEST_DIR,
        MANIFEST_PARTS_DIR,
        SLOW_VECTOR_STATE_STORAGE_PREFIX,
        TERM_STATS_STORAGE_PREFIX,
        TERM_INDEX_STORAGE_PREFIX,
        // Tombstone sidecars under `superfiles/` (live set includes the
        // paths for current superfiles; orphans age out past the safety gap).
        SUPERFILES_DIR,
    ];
    if superfiles_complete {
        prefixes.push(SUPERFILE_DATA_DIR);
    }

    // `data/` is only listed when membership is complete, so the check needs that too.
    let check_integrity = superfiles_complete && matches!(trigger, GcTrigger::Explicit);

    let mut candidates: Vec<(String, u64, SystemTime)> = Vec::new();
    // Referenced superfiles the `data/` listing returned. Only a shortfall against the referenced
    // total sends the sweep down the integrity path below.
    let mut listed_referenced_superfiles = 0usize;
    for prefix in prefixes {
        // Decided once per prefix. Every referenced key `data/` returns is a superfile.
        let count_referenced = check_integrity && prefix == SUPERFILE_DATA_DIR;
        let entries = storage.list_with_prefix_metadata(prefix).await?;
        for (key, meta) in entries {
            if live_uris.contains(&key) {
                if count_referenced {
                    listed_referenced_superfiles += 1;
                }

                report.objects_skipped_live += 1;
                continue;
            }
            if meta.last_modified >= cutoff {
                report.objects_skipped_too_new += 1;
                continue;
            }
            candidates.push((key, meta.size, meta.last_modified));
        }
    }

    // Only superfile keys end in the suffix: the pointer, manifest lists and parts, sidecars and
    // index blobs never do. Were one to, the check would only re-list and HEAD it (finding it
    // present), never report it.
    let referenced_superfiles = if check_integrity {
        live_uris
            .iter()
            .filter(|key| key.ends_with(SUPERFILE_KEY_SUFFIX))
            .count()
    } else {
        0
    };

    let mut absent = if listed_referenced_superfiles < referenced_superfiles {
        absent_superfiles(&storage, &live_uris, trigger).await?
    } else {
        AbsentSuperfiles::default()
    };

    // Nothing is deleted until the listing is complete and the pointer has been re-read once more,
    // so candidates accumulate across every prefix first. That also keeps the re-read below at one
    // probe per sweep rather than one per prefix.
    //
    // The first keep-set is spent at this point and holds a `String` per referenced object, so
    // release it rather than carry it alongside the second.
    drop(live_uris);

    // A commit may have landed while the listing ran, so re-read the pointer and put back anything
    // the newer manifest references. Only the pointer is re-read up front: rebuilding the keep-set
    // costs a slow-state fetch on a vector table, so it happens solely when the manifest actually
    // moved. Candidates can only be removed here, never added, so a re-check that comes back with a
    // partial view keeps more than it should rather than deleting something it cannot see.
    //
    // The same re-read vets the absent superfiles, which is why they alone can trigger it: the HEADs
    // above already ran, so only a key the newer manifest still references is loss.
    let mut checked_manifest_id = live_manifest_id;
    if !candidates.is_empty() || !absent.is_empty() {
        refresh_to_committed(inner).await?;
        if inner.manifest.load().manifest_id != live_manifest_id {
            let recheck = live_set(inner, &storage).await?;
            checked_manifest_id = recheck.manifest_id;
            absent.retain_referenced(&recheck.uris);
            let before = candidates.len();
            candidates.retain(|(key, _, _)| !recheck.uris.contains(key));

            let rescued = before - candidates.len();
            report.objects_skipped_live += rescued as u64;
            if rescued > 0 {
                debug!(
                    rescued,
                    from_manifest = live_manifest_id,
                    to_manifest = recheck.manifest_id,
                    "gc: a commit landed mid-sweep; keeping objects it references"
                );
            }
        }
    }

    absent.report(checked_manifest_id, trigger);

    let now = SystemTime::now();
    let mut young_superfiles_deleted = 0u64;
    for (key, size, last_modified) in candidates {
        // Drop the cache copy first.
        if let (Some(cache), Some(uri)) = (
            inner.options.disk_cache.as_ref(),
            SuperfileUri::from_storage_path(&key),
        ) {
            cache.erase_superfile_local_copy(&uri);
        }

        match storage.delete(&key).await {
            Ok(()) => {
                report.objects_deleted += 1;
                report.bytes_freed += size;
                let age = now.duration_since(last_modified).unwrap_or_default();
                if age < YOUNG_SUPERFILE_DELETE_AGE && key.ends_with(SUPERFILE_KEY_SUFFIX) {
                    young_superfiles_deleted += 1;
                    info!(
                        object = %key,
                        bytes = size,
                        age_secs = age.as_secs(),
                        manifest_id = checked_manifest_id,
                        trigger = trigger.as_str(),
                        "gc: deleted a young superfile unreferenced by the committed manifest"
                    );
                }
            }
            Err(e) => {
                warn!(
                    object = %key,
                    error = %e,
                    trigger = trigger.as_str(),
                    "gc: failed to delete orphan object"
                );
                report.delete_errors += 1;
            }
        }
    }

    // Every commit schedules a sweep, and under ingest most sweeps reclaim a few superseded manifest
    // lists or parts, so a routine sweep stays at `debug`. One that deleted a young superfile or
    // failed a delete is the one worth reading.
    if young_superfiles_deleted > 0 || report.delete_errors > 0 {
        info!(
            deleted = report.objects_deleted,
            young_superfiles_deleted,
            bytes_freed = report.bytes_freed,
            delete_errors = report.delete_errors,
            manifest_id = checked_manifest_id,
            trigger = trigger.as_str(),
            "gc sweep complete"
        );
    } else {
        debug!(
            deleted = report.objects_deleted,
            bytes_freed = report.bytes_freed,
            manifest_id = checked_manifest_id,
            trigger = trigger.as_str(),
            superfiles_complete,
            "gc sweep complete"
        );
    }
    Ok(report)
}

/// Referenced superfiles the sweep's listing did not return, split by whether a HEAD confirmed
/// them absent. Every key is still subject to the caller's manifest re-check before it is reported.
#[derive(Debug, Default)]
struct AbsentSuperfiles {
    /// A HEAD answered `NotFound`.
    confirmed: Vec<String>,
    /// Past [`MAX_MISSING_SUPERFILE_PROBES`], never probed. Reported as a count, not by name.
    unprobed: Vec<String>,
}

impl AbsentSuperfiles {
    fn is_empty(&self) -> bool {
        self.confirmed.is_empty() && self.unprobed.is_empty()
    }

    /// Keep only keys the newer manifest still references. One it dropped was reclaimed by a
    /// concurrent sweep after a compaction removed it: a legitimate delete, not loss. A newer
    /// manifest seen only in part drops keys it cannot see, which under-reports rather than
    /// reports loss on a guess.
    fn retain_referenced(&mut self, referenced: &HashSet<String>) {
        self.confirmed.retain(|key| referenced.contains(key));
        self.unprobed.retain(|key| referenced.contains(key));
    }

    /// One `error` per confirmed key, plus one for the unprobed remainder. Keep the message text
    /// stable: operators alert on it.
    fn report(&self, manifest_id: u64, trigger: GcTrigger) {
        for key in &self.confirmed {
            error!(
                object = %key,
                manifest_id,
                trigger = trigger.as_str(),
                "manifest references a superfile missing from storage"
            );
        }
        if !self.unprobed.is_empty() {
            error!(
                unprobed = self.unprobed.len(),
                manifest_id,
                trigger = trigger.as_str(),
                "manifest references a superfile missing from storage: more than the sweep probes"
            );
        }
    }
}

/// Re-list `data/` and HEAD each referenced superfile it lacks, at most
/// [`MAX_MISSING_SUPERFILE_PROBES`] of them; the rest are kept as unprobed. Only called when the
/// sweep's own listing came up short, so the key set built here costs nothing on a healthy table.
///
/// A HEAD that finds the object means the listing raced a write and nothing is lost. One that fails
/// for another reason is no evidence either way; the next explicit sweep probes the key again.
async fn absent_superfiles(
    storage: &Arc<dyn StorageProvider>,
    referenced: &HashSet<String>,
    trigger: GcTrigger,
) -> Result<AbsentSuperfiles, GcError> {
    let listed: HashSet<String> = storage
        .list_with_prefix(SUPERFILE_DATA_DIR)
        .await?
        .into_iter()
        .collect();

    let mut absent = AbsentSuperfiles::default();
    let mut probes = 0usize;

    for key in referenced
        .iter()
        .filter(|key| key.ends_with(SUPERFILE_KEY_SUFFIX) && !listed.contains(*key))
    {
        if probes == MAX_MISSING_SUPERFILE_PROBES {
            absent.unprobed.push(key.clone());
            continue;
        }

        probes += 1;

        match storage.head(key).await {
            Err(StorageError::NotFound { .. }) => absent.confirmed.push(key.clone()),
            Ok(_) => {}
            Err(e) => warn!(
                object = %key,
                error = %e,
                trigger = trigger.as_str(),
                "gc: could not confirm a superfile the listing did not return"
            ),
        }
    }

    Ok(absent)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use tempfile::tempdir;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, PrefixedStorageProvider, StorageProvider},
        supertable::{
            SupertableOptions,
            manifest::{
                ManifestSnapshot, SuperfileEntry, SuperfileUri,
                list::{
                    FORMAT_VERSION, Manifest, ManifestPartEntry, PartitionStrategy, RoutingRef,
                },
                part::{ContentHash, PartId},
            },
            slow_vector_state,
        },
        test_helpers::default_supertable_options,
    };

    /// The hidden vector index sweeps through a `PrefixedStorageProvider`, which
    /// strips its sub-prefix on list. Its keys therefore reach the cache
    /// drop-through in the same `data/seg-<uuid>.sf.parquet` shape as the user
    /// table's, so the sweep drops the right entry from the shared cache.
    #[tokio::test]
    async fn keys_listed_through_a_prefixed_provider_parse_as_superfile_uris() {
        let dir = tempdir().expect("tempdir");
        let root: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let prefixed: Arc<dyn StorageProvider> = Arc::new(PrefixedStorageProvider::new(
            Arc::clone(&root),
            "hidden-index-prefix/",
        ));
        let uri = SuperfileUri::new_v4();
        prefixed
            .put_atomic(&uri.storage_path(), bytes::Bytes::from_static(b"superfile"))
            .await
            .expect("put");

        let listed = prefixed
            .list_with_prefix_metadata(SUPERFILE_DATA_DIR)
            .await
            .expect("list");
        assert_eq!(listed.len(), 1, "the prefixed listing sees its own object");
        assert_eq!(
            SuperfileUri::from_storage_path(&listed[0].0),
            Some(uri),
            "prefix-stripped key parses back to the URI GC must drop"
        );
    }

    /// Bucket count for a minimal hash-partitioned manifest list fixture.
    const TEST_HASH_BUCKETS: u32 = 1;

    /// ManifestSnapshot id for a single-list live-set fixture.
    const TEST_MANIFEST_ID: u64 = 0;

    fn opts() -> Arc<SupertableOptions> {
        Arc::new(default_supertable_options())
    }

    fn sf_entry(uri: SuperfileUri) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
            stem: None,
            birth_version: 0,
            superfile_id: Uuid::new_v4(),
            uri,
            n_docs: 1,
            id_min: 0,
            id_max: 0,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: vec![],
            partition_hint: None,
            vector_layout: crate::superfile::vector::layout::VectorLayout::Ivf,
            subsection_offsets: None,
        })
    }

    #[test]
    fn build_live_set_contains_pointer_and_manifest_uri() {
        let manifest = ManifestSnapshot::empty(opts());
        let (live, superfiles_complete) = build_live_set(&manifest);
        assert!(superfiles_complete);
        assert!(live.contains(POINTER_PATH));
        assert!(live.contains(&manifest_uri(manifest.manifest_id)));
    }

    #[test]
    fn build_live_set_contains_superfile_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        let (live, superfiles_complete) = build_live_set(&manifest);
        assert!(superfiles_complete);
        assert!(live.contains(&uri.storage_path()));
    }

    /// The keep-set names a source-named superfile by the key it actually
    /// lives at, not the unnamed key its uuid alone would give — the one
    /// mismatch that would make the sweep delete live data. And the cache
    /// drop-through parses that key back to the uri, so the local copy goes
    /// with the object.
    #[test]
    fn build_live_set_names_a_source_named_superfile_by_its_stem_key() {
        let uri = SuperfileUri::new_v4();
        let mut entry = (*sf_entry(uri)).clone();
        entry.stem = Some("customers".into());
        let named_key = entry.storage_path();
        assert_eq!(named_key, format!("data/customers-{}.sf.parquet", uri.0));

        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![Arc::new(entry)]);
        let (live, superfiles_complete) = build_live_set(&manifest);
        assert!(superfiles_complete);
        assert!(live.contains(&named_key), "the named key is what is kept");
        assert!(
            !live.contains(&uri.storage_path()),
            "the unnamed key is not where the bytes are, so it must not be what is kept"
        );
        assert_eq!(
            SuperfileUri::from_storage_path(&named_key),
            Some(uri),
            "eviction parses the named key back to the cache's uri"
        );
    }

    #[test]
    fn build_live_set_marks_lazy_part_membership_incomplete() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let part_id = PartId::new_v4();
        let manifest = ManifestSnapshot::new(
            TEST_MANIFEST_ID,
            opts(),
            Vec::new(),
            Some(storage),
            Some(Manifest {
                tombstone_seqs: Default::default(),
                superseded_cells: Default::default(),
                split_checks: Default::default(),
                format_version: FORMAT_VERSION.into(),
                manifest_id: TEST_MANIFEST_ID,
                options_hash: ContentHash::of(b"options"),
                schema: Vec::new(),
                id_column: "_id".into(),
                fts_columns: Vec::new(),
                vector_columns: Vec::new(),
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: TEST_HASH_BUCKETS,
                },
                vector_index_storage_prefix: None,
                global_vector_index: None,
                drained_ranges: Default::default(),
                deleted_user_ids_inline: None,
                slow_vector_state_uri: None,
                slow_vector_state_content_hash: None,
                slow_vector_state_centroids: None,
                slow_vector_state_graphs: None,
                slow_vector_state_centroid_graph: None,
                term_stats: None,
                term_index: None,
                term_index_complete: false,
                parts: vec![ManifestPartEntry {
                    part_id,
                    uri: format!("manifest-parts/part-{part_id}.avro.zst"),
                    n_superfiles: 1,
                    size_bytes_compressed: 1,
                    size_bytes_uncompressed: 1,
                    content_hash: ContentHash::of(b"part"),
                    routing: None,
                    id_range: (0, 0),
                    scalar_stats_agg: HashMap::new(),
                    fts_summary_agg: Default::default(),
                }],
            }),
        );

        let (_, superfiles_complete) = build_live_set(&manifest);
        assert!(!superfiles_complete);
    }

    #[test]
    fn build_live_set_does_not_contain_older_manifest_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        assert_eq!(manifest.manifest_id, 1);
        let (live, superfiles_complete) = build_live_set(&manifest);
        assert!(superfiles_complete);
        assert!(!live.contains(&manifest_uri(0)));
        assert!(!live.contains(&manifest_uri(2)));
    }

    /// The slow-CAS entry blob referenced from the list is live; anything
    /// else under its prefix (superseded drains, orphans from a crash
    /// between PUT and stamp) is sweepable, and a ref-less manifest keeps
    /// nothing there.
    #[test]
    fn build_live_set_contains_slow_vector_state_blob() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let hash = ContentHash::of(b"slow state");
        let uri = slow_vector_state::storage_path(&hash);
        let section_hash = ContentHash::of(b"slow state centroid section");
        let section_uri = slow_vector_state::storage_path(&section_hash);
        let orphan = slow_vector_state::storage_path(&ContentHash::of(b"orphan"));
        let manifest = ManifestSnapshot::new(
            TEST_MANIFEST_ID,
            opts(),
            Vec::new(),
            Some(storage),
            Some(Manifest {
                tombstone_seqs: Default::default(),
                superseded_cells: Default::default(),
                split_checks: Default::default(),
                format_version: FORMAT_VERSION.into(),
                manifest_id: TEST_MANIFEST_ID,
                options_hash: ContentHash::of(b"options"),
                schema: Vec::new(),
                id_column: "_id".into(),
                fts_columns: Vec::new(),
                vector_columns: Vec::new(),
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: TEST_HASH_BUCKETS,
                },
                vector_index_storage_prefix: None,
                global_vector_index: None,
                drained_ranges: Default::default(),
                deleted_user_ids_inline: None,
                slow_vector_state_uri: Some(uri.clone()),
                slow_vector_state_content_hash: Some(hash),
                slow_vector_state_centroids: Some(RoutingRef {
                    uri: section_uri.clone(),
                    content_hash: section_hash,
                }),
                slow_vector_state_graphs: None,
                slow_vector_state_centroid_graph: None,
                term_stats: None,
                term_index: None,
                term_index_complete: false,
                parts: Vec::new(),
            }),
        );
        let (live, superfiles_complete) = build_live_set(&manifest);
        assert!(superfiles_complete);
        assert!(live.contains(&uri), "referenced blob must be live");
        assert!(
            live.contains(&section_uri),
            "referenced centroid section must be live"
        );
        assert!(
            !live.contains(&orphan),
            "unreferenced blob must be sweepable"
        );

        // A manifest without a ref keeps nothing under the prefix live.
        let bare = ManifestSnapshot::empty(opts());
        let (live, superfiles_complete) = build_live_set(&bare);
        assert!(superfiles_complete);
        assert!(!live.contains(&uri));
    }

    /// A referenced centroid-router section is live and survives a sweep. The
    /// section lives under the swept `slow-vector-state/` prefix, so if it were
    /// omitted from the live set GC would delete a referenced section and every
    /// subsequent query would silently rebuild the router in memory.
    #[test]
    fn build_live_set_contains_centroid_graph_section() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let hash = ContentHash::of(b"slow state");
        let uri = slow_vector_state::storage_path(&hash);
        let centroid_graph_hash = ContentHash::of(b"centroid router section");
        let centroid_graph_uri = slow_vector_state::storage_path(&centroid_graph_hash);
        let orphan = slow_vector_state::storage_path(&ContentHash::of(b"orphan"));
        let manifest = ManifestSnapshot::new(
            TEST_MANIFEST_ID,
            opts(),
            Vec::new(),
            Some(storage),
            Some(Manifest {
                tombstone_seqs: Default::default(),
                superseded_cells: Default::default(),
                split_checks: Default::default(),
                format_version: FORMAT_VERSION.into(),
                manifest_id: TEST_MANIFEST_ID,
                options_hash: ContentHash::of(b"options"),
                schema: Vec::new(),
                id_column: "_id".into(),
                fts_columns: Vec::new(),
                vector_columns: Vec::new(),
                partition_strategy: PartitionStrategy::Hash {
                    column: "_id".into(),
                    n_buckets: TEST_HASH_BUCKETS,
                },
                vector_index_storage_prefix: None,
                global_vector_index: None,
                drained_ranges: Default::default(),
                deleted_user_ids_inline: None,
                slow_vector_state_uri: Some(uri.clone()),
                slow_vector_state_content_hash: Some(hash),
                slow_vector_state_centroids: None,
                slow_vector_state_graphs: None,
                slow_vector_state_centroid_graph: Some(RoutingRef {
                    uri: centroid_graph_uri.clone(),
                    content_hash: centroid_graph_hash,
                }),
                term_stats: None,
                term_index: None,
                term_index_complete: false,
                parts: Vec::new(),
            }),
        );
        let (live, _) = build_live_set(&manifest);
        assert!(
            live.contains(&centroid_graph_uri),
            "referenced centroid-router section must be live"
        );
        assert!(
            !live.contains(&orphan),
            "unreferenced blob must still be sweepable"
        );
    }

    /// Every line a test's subscriber formats, so an assertion can read what the sweep logged.
    #[derive(Clone, Default)]
    struct CapturedLogs(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for CapturedLogs {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log buffer").extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Runs `hook` on a plain thread the first time the sweep lists `data/` after [`Self::arm`],
    /// before the listing reads anything, and hides `hidden` from every `data/` listing. That puts a
    /// concurrent commit, or a listing that missed an object, exactly where the sweep is most exposed
    /// without any timing assumption. The plain thread matters: a sync commit builds its own runtime.
    struct DataListHook {
        inner: Arc<dyn StorageProvider>,
        hook: std::sync::Mutex<Option<Box<dyn FnOnce() + Send>>>,
        hidden: Option<String>,
        armed: std::sync::atomic::AtomicBool,
        /// Requests the sweep issued while armed, so a test can pin its shape.
        heads: std::sync::atomic::AtomicUsize,
        data_lists: std::sync::atomic::AtomicUsize,
    }

    // `StorageProvider: Debug`, and a boxed closure is not; the wrapper is its inner provider.
    impl std::fmt::Debug for DataListHook {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.inner.fmt(f)
        }
    }

    impl DataListHook {
        /// A table handle over `inner` wrapped in the hook, armed once the open's own listings are
        /// done so only the sweep under test sees the hook.
        fn sweeper(
            inner: Arc<dyn StorageProvider>,
            hook: Option<Box<dyn FnOnce() + Send>>,
            hidden: Option<String>,
        ) -> (Arc<Self>, Supertable) {
            let hooked = Arc::new(Self {
                inner,
                hook: std::sync::Mutex::new(hook),
                hidden,
                armed: Default::default(),
                heads: Default::default(),
                data_lists: Default::default(),
            });
            let storage: Arc<dyn StorageProvider> = Arc::clone(&hooked) as _;
            let sweeper =
                Supertable::open(default_supertable_options().with_storage(storage)).expect("open");
            hooked
                .armed
                .store(true, std::sync::atomic::Ordering::Relaxed);
            (hooked, sweeper)
        }

        fn heads(&self) -> usize {
            self.heads.load(std::sync::atomic::Ordering::Relaxed)
        }

        fn data_lists(&self) -> usize {
            self.data_lists.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[async_trait::async_trait]
    impl StorageProvider for DataListHook {
        async fn head(&self, uri: &str) -> Result<crate::storage::ObjectMeta, StorageError> {
            if self.armed.load(std::sync::atomic::Ordering::Relaxed) {
                self.heads
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            self.inner.head(uri).await
        }

        async fn get(
            &self,
            uri: &str,
        ) -> Result<(bytes::Bytes, crate::storage::ObjectMeta), StorageError> {
            self.inner.get(uri).await
        }

        async fn get_if_none_match(
            &self,
            uri: &str,
            etag: &str,
        ) -> Result<Option<(bytes::Bytes, crate::storage::ObjectMeta)>, StorageError> {
            self.inner.get_if_none_match(uri, etag).await
        }

        async fn get_range(
            &self,
            uri: &str,
            range: std::ops::Range<u64>,
        ) -> Result<bytes::Bytes, StorageError> {
            self.inner.get_range(uri, range).await
        }

        async fn put_atomic(
            &self,
            uri: &str,
            bytes: bytes::Bytes,
        ) -> Result<Option<String>, StorageError> {
            self.inner.put_atomic(uri, bytes).await
        }

        async fn put_if_match(
            &self,
            uri: &str,
            bytes: bytes::Bytes,
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
        ) -> Result<Vec<(String, crate::storage::ObjectMeta)>, StorageError> {
            let armed = self.armed.load(std::sync::atomic::Ordering::Relaxed);
            if armed && prefix == SUPERFILE_DATA_DIR {
                self.data_lists
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let hook = self.hook.lock().expect("hook slot").take();
                if let Some(hook) = hook {
                    let (tx, rx) = tokio::sync::oneshot::channel();
                    std::thread::spawn(move || {
                        hook();
                        let _ = tx.send(());
                    });
                    rx.await.expect("hook thread");
                }
            }
            let mut entries = self.inner.list_with_prefix_metadata(prefix).await?;
            if armed && let Some(hidden) = &self.hidden {
                entries.retain(|(key, _)| key != hidden);
            }
            Ok(entries)
        }
    }

    /// A table over `storage`, committed one title at a time so each title is its own superfile.
    fn table_with_superfiles(storage: &Arc<dyn StorageProvider>, titles: &[&str]) -> Supertable {
        let st = Supertable::create(default_supertable_options().with_storage(Arc::clone(storage)))
            .expect("create");
        for title in titles {
            let mut writer = st.writer().expect("writer");
            writer
                .append(&crate::test_helpers::build_title_batch(&[title]))
                .expect("append");
            writer.commit().expect("commit");
        }
        st
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

    /// Sweep `st` with no grace as `trigger`, capturing every line it logs at `info` and above.
    ///
    /// A multi-thread runtime, so `block_on` drives the sweep on this thread and the thread-local
    /// subscriber sees its events.
    fn sweep_logging(st: &Supertable, trigger: GcTrigger) -> (GcReport, String) {
        let logs = CapturedLogs::default();
        let writer = logs.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        let report = tracing::subscriber::with_default(subscriber, || {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("runtime")
                .block_on(gc_storage_sweep_for_inner(
                    st.inner(),
                    Duration::ZERO,
                    trigger,
                ))
                .expect("sweep")
        });
        let text = String::from_utf8_lossy(&logs.0.lock().expect("log buffer")).into_owned();
        (report, text)
    }

    /// Lines reporting loss, under the stable message operators alert on.
    fn loss_lines(logs: &str) -> Vec<&str> {
        logs.lines()
            .filter(|line| line.contains("manifest references a superfile missing from storage"))
            .collect()
    }

    /// A referenced superfile deleted behind the table's back is named at `error`, and the one still
    /// there is neither named nor touched.
    #[test]
    fn a_sweep_reports_a_referenced_superfile_missing_from_storage() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = table_with_superfiles(&storage, &["alphatoken marker", "betatoken marker"]);
        let [lost, kept] = <[String; 2]>::try_from(superfile_keys(&st)).expect("two superfiles");
        std::fs::remove_file(dir.path().join(&lost)).expect("delete behind the table");

        let (report, logs) = sweep_logging(&st, GcTrigger::Explicit);

        let lines = loss_lines(&logs);
        assert_eq!(lines.len(), 1, "exactly the lost superfile: {logs}");
        assert!(
            lines[0].contains("ERROR") && lines[0].contains(&lost),
            "{logs}"
        );
        assert!(
            !logs.contains(&kept),
            "the present superfile is not named: {logs}"
        );
        assert!(dir.path().join(&kept).exists(), "{report:?}");
    }

    /// The race the ordering exists for: between the sweep's keep-set and its listing, a compaction
    /// drops two superfiles and a concurrent sweep reclaims them. They are absent and were
    /// referenced, but the committed manifest no longer names them, so nothing is lost and nothing
    /// may be reported. The merged output is new to the sweep and must survive too.
    #[test]
    fn a_superfile_a_concurrent_compaction_dropped_is_not_reported() {
        let dir = tempdir().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        // Enough small superfiles for the planner to merge them.
        let titles: Vec<String> = (0..10).map(|i| format!("token{i} marker")).collect();
        let titles: Vec<&str> = titles.iter().map(String::as_str).collect();
        let writer = Arc::new(table_with_superfiles(&local, &titles));
        let inputs = superfile_keys(&writer);

        let compactor = Arc::clone(&writer);
        let root = dir.path().to_path_buf();
        let hook: Box<dyn FnOnce() + Send> = Box::new(move || {
            compactor
                .optimize(&crate::OptimizeOptions::compact(
                    crate::CompactionSettings {
                        target_superfile_size_mb: 1,
                        min_fill_percent: 1,
                        ..crate::CompactionSettings::default()
                    },
                ))
                .expect("optimize");
            let live = superfile_keys(&compactor);
            let dropped: Vec<&String> = inputs.iter().filter(|key| !live.contains(key)).collect();
            assert!(!dropped.is_empty(), "the compaction merged something");
            for key in dropped {
                std::fs::remove_file(root.join(key)).expect("concurrent reclaim");
            }
        });
        let (_, sweeper) = DataListHook::sweeper(Arc::clone(&local), Some(hook), None);

        let (report, logs) = sweep_logging(&sweeper, GcTrigger::Explicit);

        assert!(
            loss_lines(&logs).is_empty(),
            "a legitimate reclaim was reported as loss: {logs}"
        );
        for key in superfile_keys(&writer) {
            assert!(
                dir.path().join(&key).exists(),
                "merged output kept: {report:?}"
            );
        }
    }

    /// A listing that missed an object storage does hold is a race, not loss: the HEAD finds it.
    #[test]
    fn a_superfile_the_listing_missed_but_storage_holds_is_not_reported() {
        let dir = tempdir().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let writer = table_with_superfiles(&local, &["alphatoken marker"]);
        let [key] = <[String; 1]>::try_from(superfile_keys(&writer)).expect("one superfile");

        let (_, sweeper) = DataListHook::sweeper(local, None, Some(key.clone()));

        let (_, logs) = sweep_logging(&sweeper, GcTrigger::Explicit);

        assert!(loss_lines(&logs).is_empty(), "{logs}");
        assert!(
            dir.path().join(&key).exists(),
            "the hidden superfile was kept"
        );
    }

    /// A table missing more superfiles than one sweep probes: the first batch is confirmed by name,
    /// the rest carried as unprobed so the report can still count them.
    #[tokio::test]
    async fn probing_stops_at_the_cap_and_keeps_the_rest_unprobed() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let referenced: HashSet<String> = (0..=MAX_MISSING_SUPERFILE_PROBES)
            .map(|_| SuperfileUri::new_v4().storage_path())
            .chain([POINTER_PATH.to_string()])
            .collect();

        let absent = absent_superfiles(&storage, &referenced, GcTrigger::Explicit)
            .await
            .expect("probe");

        assert_eq!(absent.confirmed.len(), MAX_MISSING_SUPERFILE_PROBES);
        assert_eq!(
            absent.unprobed.len(),
            1,
            "only superfile keys, past the cap"
        );
    }

    /// Routine reclaim stays at `debug`; a superfile deleted soon after it was written, the case an
    /// in-flight commit loses, gets its own `info` line and lifts the summary to `info`.
    #[test]
    fn a_sweep_logs_only_young_superfile_deletes_at_info() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = table_with_superfiles(&storage, &[]);

        let young = SuperfileUri::new_v4().storage_path();
        let old = SuperfileUri::new_v4().storage_path();
        std::fs::create_dir_all(dir.path().join(SUPERFILE_DATA_DIR)).expect("data dir");
        for key in [&young, &old] {
            std::fs::write(dir.path().join(key), b"orphan").expect("plant orphan");
        }
        let stamp = SystemTime::now() - 2 * YOUNG_SUPERFILE_DELETE_AGE;
        std::fs::File::options()
            .write(true)
            .open(dir.path().join(&old))
            .expect("open old orphan")
            .set_times(
                std::fs::FileTimes::new()
                    .set_accessed(stamp)
                    .set_modified(stamp),
            )
            .expect("backdate old orphan");

        let (report, logs) = sweep_logging(&st, GcTrigger::Explicit);

        assert!(
            !dir.path().join(&young).exists() && !dir.path().join(&old).exists(),
            "both orphans reclaimed: {report:?}"
        );
        assert!(logs.contains(&young), "the young delete is logged: {logs}");
        assert!(
            !logs.contains(&old),
            "the old delete stays at debug: {logs}"
        );
        assert!(
            logs.contains("young_superfiles_deleted=1"),
            "summary at info: {logs}"
        );
    }

    /// A sweep that only reclaimed routine leftovers logs nothing at `info`.
    #[test]
    fn a_routine_sweep_stays_below_info() {
        let dir = tempdir().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        // Each commit supersedes the manifest list before it, which the sweep then reclaims.
        let st = table_with_superfiles(&storage, &["alphatoken marker", "betatoken marker"]);

        let (report, logs) = sweep_logging(&st, GcTrigger::Explicit);

        assert!(
            report.objects_deleted > 0,
            "something routine was reclaimed"
        );
        assert!(logs.is_empty(), "nothing at info or above: {logs}");
    }

    /// The check must add no request to a healthy sweep: no HEAD, and `data/` listed once, as before.
    #[test]
    fn a_healthy_sweep_issues_no_extra_requests() {
        let dir = tempdir().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        table_with_superfiles(&local, &["alphatoken marker", "betatoken marker"]);
        let (hooked, sweeper) = DataListHook::sweeper(local, None, None);

        let (_, logs) = sweep_logging(&sweeper, GcTrigger::Explicit);

        assert!(loss_lines(&logs).is_empty(), "{logs}");
        assert_eq!(hooked.heads(), 0, "no HEAD on a healthy table");
        assert_eq!(hooked.data_lists(), 1, "data/ listed once");
    }

    /// The sweep every commit schedules skips the check, so ingest pays nothing for it even on a
    /// table that has already lost a superfile.
    #[test]
    fn a_deferred_sweep_skips_the_check() {
        let dir = tempdir().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let writer = table_with_superfiles(&local, &["alphatoken marker", "betatoken marker"]);
        let lost = superfile_keys(&writer).remove(0);
        std::fs::remove_file(dir.path().join(&lost)).expect("delete behind the table");
        let (hooked, sweeper) = DataListHook::sweeper(local, None, None);

        let (_, logs) = sweep_logging(&sweeper, GcTrigger::DeferredReclaim);

        assert!(loss_lines(&logs).is_empty(), "{logs}");
        assert_eq!(hooked.heads(), 0, "no probe from a deferred sweep");
        assert_eq!(hooked.data_lists(), 1, "no re-list from a deferred sweep");
    }

    /// A HEAD that fails for another reason proves nothing either way: it is logged at `warn`,
    /// never reported as loss, and the next explicit sweep probes the key again.
    #[test]
    fn a_superfile_whose_head_fails_is_not_reported() {
        use crate::test_helpers::fault_storage::{FaultOp, FaultStorage};

        let dir = tempdir().expect("tempdir");
        let local: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let faulty = FaultStorage::wrap(local);
        let storage: Arc<dyn StorageProvider> = Arc::clone(&faulty) as _;
        let st = table_with_superfiles(&storage, &["alphatoken marker"]);
        let [lost] = <[String; 1]>::try_from(superfile_keys(&st)).expect("one superfile");
        std::fs::remove_file(dir.path().join(&lost)).expect("delete behind the table");
        faulty.fail(FaultOp::Head, &lost, usize::MAX);

        let (_, logs) = sweep_logging(&st, GcTrigger::Explicit);

        assert!(
            loss_lines(&logs).is_empty(),
            "an unanswered HEAD was reported as loss: {logs}"
        );
        assert!(
            logs.contains("could not confirm"),
            "the failed probe is visible: {logs}"
        );
        assert!(faulty.fired() > 0, "the probe ran and failed");
    }
}
