// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, SystemTime},
};

use tracing::{debug, warn};

use crate::{
    runtime_bridge::bridge_on_runtime,
    storage::{StorageError, StorageProvider},
    supertable::{
        ManifestSnapshot, Supertable,
        error::GcError,
        handle::SupertableInner,
        manifest::{
            SUPERFILE_DATA_DIR, SuperfileUri,
            commit::{MANIFEST_DIR, MANIFEST_PARTS_DIR, POINTER_PATH, manifest_uri},
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
            live.insert(sf.uri.storage_path());
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
    if let Some(graphs) = manifest.slow_vector_state_graphs_blob() {
        live.insert(graphs.uri.clone());
    }

    // Each resident superfile's tombstone sidecar. `superfiles/` is swept whatever the flag says,
    // so these have to be named here or a sidecar past the gap is deleted and its deleted rows
    // come back. The superfile paths repeat what the complete view above already inserted.
    for sf in manifest.get_all_superfiles() {
        live.insert(sf.uri.storage_path());
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

    pub(crate) async fn gc_async(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        gc_storage_sweep_for_inner(self.inner(), safety_gap).await
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
            uris.extend(pending.entries.iter().map(|entry| entry.uri.storage_path()));
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
pub(super) async fn gc_storage_sweep_for_inner(
    inner: &SupertableInner,
    safety_gap: Duration,
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
        // Tombstone sidecars under `superfiles/` (live set includes the
        // paths for current superfiles; orphans age out past the safety gap).
        SUPERFILES_DIR,
    ];
    if superfiles_complete {
        prefixes.push(SUPERFILE_DATA_DIR);
    }

    let mut candidates: Vec<(String, u64)> = Vec::new();
    for prefix in prefixes {
        let entries = storage.list_with_prefix_metadata(prefix).await?;
        for (key, meta) in entries {
            if live_uris.contains(&key) {
                report.objects_skipped_live += 1;
                continue;
            }
            if meta.last_modified >= cutoff {
                report.objects_skipped_too_new += 1;
                continue;
            }
            candidates.push((key, meta.size));
        }
    }

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
    if !candidates.is_empty() {
        refresh_to_committed(inner).await?;
        if inner.manifest.load().manifest_id != live_manifest_id {
            let recheck = live_set(inner, &storage).await?;
            let before = candidates.len();
            candidates.retain(|(key, _)| !recheck.uris.contains(key));

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

    for (key, size) in candidates {
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
            }
            Err(e) => {
                warn!(object = %key, error = %e, "gc: failed to delete orphan object");
                report.delete_errors += 1;
            }
        }
    }

    debug!(
        deleted = report.objects_deleted,
        bytes_freed = report.bytes_freed,
        delete_errors = report.delete_errors,
        superfiles_complete,
        "gc sweep complete"
    );
    Ok(report)
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
}
