// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

use std::{
    collections::HashSet,
    time::{Duration, SystemTime},
};

use tracing::{debug, warn};

use crate::{
    Supertable,
    runtime_bridge::bridge_on_runtime,
    storage::StorageError,
    supertable::{
        ManifestSnapshot,
        error::GcError,
        handle::SupertableInner,
        manifest::{
            SUPERFILE_DATA_DIR,
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

/// Outcome of a [`Supertable::gc`] sweep: what was reclaimed and what was
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

fn build_live_set(manifest: &ManifestSnapshot) -> (HashSet<String>, bool) {
    let mut live = HashSet::new();
    live.insert(POINTER_PATH.to_string());
    live.insert(manifest_uri(manifest.manifest_id));
    for entry in manifest.get_all_list_entries() {
        live.insert(entry.uri.clone());
        if let Some(routing) = &entry.routing {
            live.insert(routing.uri.clone());
        }
    }
    let superfiles_complete = if let Some(superfiles) = manifest.complete_flat_superfiles() {
        for sf in superfiles {
            live.insert(sf.uri.storage_path());
        }
        true
    } else {
        false
    };
    // Slow-CAS objects (routing-shaped state blob + fp32 centroid
    // section): the URIs are read straight off the manifest-list refs —
    // sync, no fetch. Superseded generations (older drains) are absent
    // from the current list and get swept once past the safety gap.
    if let Some((uri, _)) = manifest.slow_vector_state_blob() {
        live.insert(uri.to_owned());
    }
    if let Some(centroids) = manifest.slow_vector_state_centroids_blob() {
        live.insert(centroids.uri.clone());
    }
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

/// Delete storage objects not referenced by the current manifest once they are
/// older than `safety_gap`. Supersedes inline post-commit deletes so readers
/// pinned to an older snapshot cannot lose bytes mid-fetch.
pub(super) async fn gc_storage_sweep_for_inner(
    inner: &SupertableInner,
    safety_gap: Duration,
) -> Result<GcReport, GcError> {
    let storage = inner.options.storage.clone().ok_or(GcError::NoStorage)?;
    let manifest = inner.manifest.load_full();
    let (mut live, superfiles_complete) = build_live_set(&manifest);
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
            live.extend(pending.entries.iter().map(|entry| entry.uri.storage_path()));
        }
    }
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
    for prefix in prefixes {
        let entries = storage.list_with_prefix_metadata(prefix).await?;
        for (key, meta) in entries {
            if live.contains(&key) {
                report.objects_skipped_live += 1;
                continue;
            }
            if meta.last_modified >= cutoff {
                report.objects_skipped_too_new += 1;
                continue;
            }
            match storage.delete(&key).await {
                Ok(()) => {
                    report.objects_deleted += 1;
                    report.bytes_freed += meta.size;
                }
                Err(e) => {
                    warn!(object = %key, error = %e, "gc: failed to delete orphan object");
                    report.delete_errors += 1;
                }
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
        storage::{LocalFsStorageProvider, StorageProvider},
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
