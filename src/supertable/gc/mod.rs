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
    supertable::{
        ManifestSnapshot,
        error::GcError,
        manifest::commit::{MANIFEST_DIR, MANIFEST_PARTS_DIR, POINTER_PATH, manifest_uri},
    },
};

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

fn build_live_set(manifest: &ManifestSnapshot) -> HashSet<String> {
    let mut live = HashSet::new();
    live.insert(POINTER_PATH.to_string());
    live.insert(manifest_uri(manifest.manifest_id));
    for entry in manifest.get_all_list_entries() {
        live.insert(entry.uri.clone());
    }
    for sf in manifest.get_all_superfiles() {
        live.insert(sf.uri.storage_path());
    }
    live
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
        let inner = self.inner();
        let storage = inner.options.storage.clone().ok_or(GcError::NoStorage)?;
        let manifest = inner.manifest.load_full();
        let live = build_live_set(&manifest);
        let cutoff = SystemTime::now()
            .checked_sub(safety_gap)
            .unwrap_or(SystemTime::UNIX_EPOCH);

        let mut report = GcReport::default();

        for prefix in [MANIFEST_DIR, MANIFEST_PARTS_DIR, "data"] {
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
            "gc sweep complete"
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc};

    use uuid::Uuid;

    use super::*;
    use crate::{
        supertable::{
            SupertableOptions,
            manifest::{ManifestSnapshot, SuperfileEntry, SuperfileUri},
        },
        test_helpers::default_supertable_options,
    };

    fn opts() -> Arc<SupertableOptions> {
        Arc::new(default_supertable_options())
    }

    fn sf_entry(uri: SuperfileUri) -> Arc<SuperfileEntry> {
        Arc::new(SuperfileEntry {
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
            subsection_offsets: None,
        })
    }

    #[test]
    fn build_live_set_contains_pointer_and_manifest_uri() {
        let manifest = ManifestSnapshot::empty(opts());
        let live = build_live_set(&manifest);
        assert!(live.contains(POINTER_PATH));
        assert!(live.contains(&manifest_uri(manifest.manifest_id)));
    }

    #[test]
    fn build_live_set_contains_superfile_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        let live = build_live_set(&manifest);
        assert!(live.contains(&uri.storage_path()));
    }

    #[test]
    fn build_live_set_does_not_contain_older_manifest_uris() {
        let uri = SuperfileUri::new_v4();
        let manifest = ManifestSnapshot::empty(opts()).with_appended(vec![sf_entry(uri)]);
        assert_eq!(manifest.manifest_id, 1);
        let live = build_live_set(&manifest);
        assert!(!live.contains(&manifest_uri(0)));
        assert!(!live.contains(&manifest_uri(2)));
    }
}
