// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Picks which superfiles to merge.
//!
//! no I/O. `supertable::compact` gathers the
//! stats, calls [`select`], then merges each [`CompactionJob`].
//! Compaction is single-level — a target-sized superfile is never
//! re-compacted.

use crate::{
    Supertable,
    config::CompactionSettings,
    superfile::builder::SuperfileBuilder,
    supertable::{
        BuildError, SuperfileEntry,
        error::CompactionError,
        query::dispatch::open_reader,
        wal::{WalStore, tombstones_admin, tombstones_admin::TombstonesAdminError},
        writer::{PreparedSuperfile, ShardOutput, prepare_superfile},
    },
};
use bytes::Bytes;
use std::{collections::BTreeMap, sync::Arc};
use uuid::Uuid;

const MIB: u64 = 1024 * 1024;

/// Stats for one superfile. The caller fills these in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuperfileStats {
    pub superfile_id: Uuid,
    /// Partition it belongs to.
    /// never merge across partitions.
    pub partition_key: Vec<u8>,
    pub size_bytes: u64,
    pub n_docs: u64,
    pub tombstoned_docs: u64,
    /// Already owned by another compaction so skip it.
    pub sealed_by_other: bool,
}

impl SuperfileStats {
    fn live_docs(&self) -> u64 {
        self.n_docs.saturating_sub(self.tombstoned_docs)
    }

    /// Bytes left after dropping deleted rows.
    fn live_bytes(&self) -> u64 {
        if self.n_docs == 0 {
            return 0;
        }
        (self.size_bytes as u128 * self.live_docs() as u128 / self.n_docs as u128) as u64
    }
}

/// A set of superfiles to merge into one new superfile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub partition_key: Vec<u8>,
    pub inputs: Vec<Uuid>,
    /// Estimated size of the merged superfile.
    pub estimated_output_bytes: u64,
}

/// Plan compaction: pack each partition's small superfiles into
/// as many target-sized jobs as they fill. Leftovers that can't
/// reach the floor are left for next time.
pub fn select(superfiles: &[SuperfileStats], cfg: &CompactionSettings) -> Vec<CompactionJob> {
    let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
    let min_output_bytes =
        (target_bytes as u128 * cfg.min_fill_percent.clamp(1, 100) as u128 / 100) as u64;

    let mut by_partition: BTreeMap<&[u8], Vec<&SuperfileStats>> = BTreeMap::new();
    for s in superfiles {
        by_partition.entry(&s.partition_key).or_default().push(s);
    }

    let mut jobs = Vec::new();
    for (key, segs) in by_partition {
        pack_partition(key, segs, target_bytes, min_output_bytes, &mut jobs);
    }
    jobs
}

fn pack_partition(
    key: &[u8],
    segs: Vec<&SuperfileStats>,
    target_bytes: u64,
    min_output_bytes: u64,
    jobs: &mut Vec<CompactionJob>,
) {
    // Exclude superfiles already at target size — they are done and
    // re-compacting them gains nothing.
    let mut candidates: Vec<&SuperfileStats> = segs
        .into_iter()
        .filter(|s| !s.sealed_by_other && s.size_bytes < min_output_bytes)
        .collect();

    // Most-deleted first (reclaim space soonest), then smallest, then ID.
    candidates.sort_by(|a, b| {
        let lhs = a.tombstoned_docs as u128 * b.n_docs.max(1) as u128;
        let rhs = b.tombstoned_docs as u128 * a.n_docs.max(1) as u128;
        rhs.cmp(&lhs)
            .then(a.size_bytes.cmp(&b.size_bytes))
            .then(a.superfile_id.cmp(&b.superfile_id))
    });

    let mut pending = PendingJob::default();
    for s in candidates {
        if !pending.fits(s, target_bytes) {
            pending.emit(key, min_output_bytes, jobs);
        }
        pending.push(s);
    }
    pending.emit(key, min_output_bytes, jobs);
}

#[derive(Default)]
struct PendingJob {
    inputs: Vec<Uuid>,
    live_bytes: u64,
}

impl PendingJob {
    fn fits(&self, s: &SuperfileStats, target_bytes: u64) -> bool {
        self.live_bytes + s.live_bytes() <= target_bytes
    }

    fn push(&mut self, s: &SuperfileStats) {
        self.inputs.push(s.superfile_id);
        self.live_bytes += s.live_bytes();
    }

    /// Emit a CompactionJob if ≥ 2 inputs and live bytes reach `min_output_bytes`.
    fn emit(&mut self, key: &[u8], min_output_bytes: u64, jobs: &mut Vec<CompactionJob>) {
        if self.inputs.len() >= 2 && self.live_bytes >= min_output_bytes {
            jobs.push(CompactionJob {
                partition_key: key.to_vec(),
                inputs: std::mem::take(&mut self.inputs),
                estimated_output_bytes: self.live_bytes,
            });
        }
        *self = PendingJob::default();
    }
}

impl Supertable {
    /// Compaction entry point.
    /// Gathers per-superfile stats from the current manifest snapshot,
    /// selects compaction jobs, then for each job seals every input
    /// superfile's tombstone sidecar so no concurrent deletes can land
    /// during the merge window.
    pub(crate) async fn compact(&self, cfg: &CompactionSettings) -> Result<(), CompactionError> {
        let inner = self.inner();
        let manifest = inner.manifest.load_full();

        let storage = manifest
            .options
            .storage
            .as_ref()
            .ok_or(CompactionError::NoStorage)?
            .clone();
        let wal_store = WalStore::new(storage);

        // One LIST to find which superfiles have a tombstone sidecar.
        let tombstone_ids: std::collections::HashSet<Uuid> = wal_store
            .list_tombstone_ids()
            .await
            .map_err(|e| CompactionError::Seal(e.to_string()))?
            .into_iter()
            .collect();

        // Fetch full sidecars (bitmap + seal record) only for superfiles
        // that actually have a file.
        let superfile_ids: Vec<Uuid> = manifest
            .superfile_list
            .superfiles
            .iter()
            .map(|e| e.superfile_id)
            .filter(|id| tombstone_ids.contains(id))
            .collect();
        let sidecar_futs = superfile_ids.iter().map(|id| {
            let ws = wal_store.clone();
            let id = *id;
            async move { (id, ws.get_tombstones(id).await) }
        });
        let sidecar_results = futures::future::join_all(sidecar_futs).await;
        let sidecar_map: std::collections::HashMap<Uuid, _> = sidecar_results
            .into_iter()
            .filter_map(|(id, result)| result.ok().flatten().map(|(sc, _etag)| (id, sc)))
            .collect();

        // Build SuperfileStats for every superfile in the snapshot.
        let stats: Vec<SuperfileStats> = manifest
            .superfile_list
            .superfiles
            .iter()
            .map(|entry| {
                let sidecar = sidecar_map.get(&entry.superfile_id);
                let tombstoned_docs = sidecar.map(|sc| sc.bitmap.len()).unwrap_or(0);
                let sealed_by_other = sidecar.map(|sc| sc.seal.is_some()).unwrap_or(false);
                SuperfileStats {
                    superfile_id: entry.superfile_id,
                    partition_key: entry.partition_key.clone(),
                    size_bytes: entry
                        .subsection_offsets
                        .as_ref()
                        .map(|o| o.total_size)
                        .unwrap_or(0),
                    n_docs: entry.n_docs,
                    tombstoned_docs,
                    sealed_by_other,
                }
            })
            .collect();

        let jobs = select(&stats, cfg);

        for job in jobs {
            // Resolve input Arc<SuperfileEntry> from the snapshot.
            let inputs: Vec<Arc<SuperfileEntry>> = job
                .inputs
                .iter()
                .map(|id| {
                    manifest
                        .superfile_list
                        .superfiles
                        .iter()
                        .find(|e| e.superfile_id == *id)
                        .cloned()
                        .ok_or(CompactionError::SuperfileNotFound(*id))
                })
                .collect::<Result<_, _>>()?;

            // Seal every input sidecar.
            // once sealed, further incoming updates are rejected
            // and this seal flag helps to prevent overlapping compactions
            // on same files
            let compaction_id = Uuid::new_v4();
            let sealed_at = chrono::Utc::now();
            for entry in &inputs {
                loop {
                    match tombstones_admin::seal(
                        &wal_store,
                        entry.superfile_id,
                        compaction_id,
                        sealed_at,
                    )
                    .await
                    {
                        Ok(_) => break,
                        Err(TombstonesAdminError::CasLost { .. }) => {
                            // A writer landed a tombstone bit between our
                            // GET and our PUT. Re-read and retry — the
                            // seal will succeed on the next attempt unless
                            // another compactor raced us.
                            continue;
                        }
                        Err(TombstonesAdminError::AlreadySealed {
                            superfile_id,
                            existing_compaction_id,
                        }) => {
                            return Err(CompactionError::SidecarConflict {
                                superfile_id,
                                existing_compaction_id,
                            });
                        }
                        Err(TombstonesAdminError::WalStore(e)) => {
                            return Err(CompactionError::Seal(e.to_string()));
                        }
                    }
                }
            }

            // TODO(pranav): merge_superfiles(&inputs) + try_commit_attempt
            let _ = inputs;
        }

        Ok(())
    }

    /// Merges the given superfiles into one
    pub(crate) async fn merge_superfiles(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
    ) -> Result<PreparedSuperfile, BuildError> {
        let manifest = { self.inner().manifest.load().clone() };
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();
        let tombstone_cache = self.inner().tombstone_cache.clone();

        let mut superfile_readers_fut = Vec::with_capacity(superfiles.len());
        for entry in superfiles {
            let open_fut = async {
                let r = open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry).await;
                (entry.superfile_id, r)
            };
            superfile_readers_fut.push(open_fut);
        }
        let readers = futures::future::join_all(superfile_readers_fut).await;

        let now = std::time::Instant::now();
        if let Some(tombstone_cache) = &tombstone_cache {
            let superfile_ids = superfiles
                .iter()
                .map(|entry| entry.superfile_id)
                .collect::<Vec<_>>();

            tombstone_cache.prefetch(&superfile_ids, now).await;
        }

        let mut readers_with_tombstones = Vec::with_capacity(readers.len());
        for (superfile_id, reader) in readers {
            let bitmap = tombstone_cache
                .as_ref()
                .map(|t| t.bitmap_for(superfile_id, now))
                .transpose()
                .map_err(|e| BuildError::Store(e.to_string()))?;

            let reader = reader.map_err(|e| BuildError::Store(e.to_string()))?;
            readers_with_tombstones.push((reader.clone(), bitmap));
        }

        let (merged_bytes, superfile_stats) =
            SuperfileBuilder::build_from_readers(&readers_with_tombstones)?;
        let merged_bytes = Bytes::from(merged_bytes);

        let shard = ShardOutput::new_with_params(
            merged_bytes,
            superfile_stats.n_docs,
            superfile_stats.id_min,
            superfile_stats.id_max,
            superfile_stats.scalar_stats,
        );

        let prepared_superfile = prepare_superfile(self.inner().as_ref(), shard)?;

        prepared_superfile.ok_or(BuildError::NoDocsToBuild)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Supertable;
    use crate::supertable::storage::LocalFsStorageProvider;
    use crate::test_helpers::{build_title_batch, default_supertable_options};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn mib(n: u64) -> u64 {
        n * MIB
    }

    fn seg(id: u128, size_mib: u64, n_docs: u64, tombstoned: u64) -> SuperfileStats {
        SuperfileStats {
            superfile_id: Uuid::from_u128(id),
            partition_key: Vec::new(),
            size_bytes: mib(size_mib),
            n_docs,
            tombstoned_docs: tombstoned,
            sealed_by_other: false,
        }
    }

    fn default_cfg() -> CompactionSettings {
        CompactionSettings::default() // 1 GiB target, 80% floor
    }

    #[test]
    fn empty_input_yields_no_jobs() {
        assert!(select(&[], &default_cfg()).is_empty());
    }

    #[test]
    fn below_fill_floor_skips() {
        // 400 MiB total < 80% of 1 GiB.
        let segs = vec![seg(1, 200, 1000, 0), seg(2, 200, 1000, 0)];
        assert!(select(&segs, &default_cfg()).is_empty());
    }

    #[test]
    fn packs_one_job_and_leaves_remainder() {
        // 6 × 200 MiB: one job of 5 (1000 MiB), 6th left over.
        let segs: Vec<_> = (0..6).map(|i| seg(i, 200, 1000, 0)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 5);
        assert_eq!(jobs[0].estimated_output_bytes, mib(1000));
    }

    #[test]
    fn splits_many_superfiles_into_multiple_jobs() {
        // 12 × 200 MiB: two jobs of 5, last 2 left over.
        let segs: Vec<_> = (0..12).map(|i| seg(i, 200, 1000, 0)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.inputs.len() == 5));
    }

    #[test]
    fn already_target_sized_superfile_is_never_re_compacted() {
        let big = seg(99, 1024, 1_000_000, 0);
        let mut segs = vec![big.clone()];
        segs.extend((0..5).map(|i| seg(i, 200, 1000, 0)));
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert!(!jobs[0].inputs.contains(&big.superfile_id));
    }

    #[test]
    fn output_estimate_uses_live_bytes() {
        // 5 × 400 MiB raw, half deleted → 200 MiB live each.
        let segs: Vec<_> = (0..5).map(|i| seg(i, 400, 1000, 500)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 5);
        assert_eq!(jobs[0].estimated_output_bytes, mib(1000));
    }

    #[test]
    fn prefers_most_deleted_first() {
        let mut segs: Vec<_> = (0..9).map(|i| seg(i, 100, 1000, 0)).collect();
        let dead_heavy = seg(100, 100, 1000, 900);
        segs.push(dead_heavy.clone());
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs[0].inputs[0], dead_heavy.superfile_id);
    }

    #[test]
    fn sealed_by_other_is_excluded() {
        let mut owned = seg(1, 200, 1000, 0);
        owned.sealed_by_other = true;
        let segs = vec![owned, seg(2, 200, 1000, 0), seg(3, 200, 1000, 0)];
        for job in select(&segs, &default_cfg()) {
            assert!(!job.inputs.contains(&Uuid::from_u128(1)));
        }
    }

    #[test]
    fn fewer_than_two_candidates_skips() {
        assert!(select(&[seg(1, 200, 1000, 0)], &default_cfg()).is_empty());
    }

    #[test]
    fn partitions_packed_independently() {
        let mut segs = Vec::new();
        for i in 0..5 {
            let mut s = seg(i, 200, 1000, 0);
            s.partition_key = vec![0xA];
            segs.push(s);
        }
        for i in 5..10 {
            let mut s = seg(i, 200, 1000, 0);
            s.partition_key = vec![0xB];
            segs.push(s);
        }
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 2);
        let a = jobs
            .iter()
            .find(|j| j.partition_key == vec![0xA])
            .expect("partition A job");
        assert!(a.inputs.iter().all(|id| id.as_u128() < 5));
    }

    // Tests for merge_superfiles function
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_merges_two_superfiles() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn crate::supertable::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create first superfile with 2 rows
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["first doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Create second superfile with 2 rows
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["third doc", "fourth doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Get the superfiles to merge
        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .superfile_list
            .superfiles
            .iter()
            .take(2)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 2, "should have 2 superfiles");

        // Merge the superfiles - should succeed
        let _merged_superfile = st
            .merge_superfiles(&superfiles)
            .await
            .expect("merge_superfiles should succeed");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_preserves_scalar_stats() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn crate::supertable::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create first superfile with apple/banana
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["apple", "banana"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Create second superfile with cherry/date
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["cherry", "date"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .superfile_list
            .superfiles
            .iter()
            .take(2)
            .cloned()
            .collect();

        // Precompute expected stats from source superfiles
        let expected_n_docs: u64 = superfiles.iter().map(|sf| sf.n_docs).sum();
        let expected_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let expected_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);

        // Merge should succeed and preserve scalar stats
        let merged_superfile = st
            .merge_superfiles(&superfiles)
            .await
            .expect("merge_superfiles should succeed");

        // Verify merged superfile stats match expected values
        assert_eq!(
            merged_superfile.entry.n_docs, expected_n_docs,
            "n_docs should be sum of input superfiles"
        );
        assert_eq!(
            merged_superfile.entry.id_min, expected_id_min,
            "id_min should be minimum across all superfiles"
        );
        assert_eq!(
            merged_superfile.entry.id_max, expected_id_max,
            "id_max should be maximum across all superfiles"
        );

        // Verify scalar stats for title column (lexicographic ordering: apple < banana < cherry < date)
        let title_stats = merged_superfile
            .entry
            .scalar_stats
            .cols
            .get("title")
            .expect("merged entry should have title column stats");

        // Extract min and max string values from the arrays
        let title_min_arr = title_stats
            .0
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .expect("title column should be LargeStringArray");
        let title_max_arr = title_stats
            .1
            .as_any()
            .downcast_ref::<arrow_array::LargeStringArray>()
            .expect("title column should be LargeStringArray");

        // Verify exact min/max values (apple is min across all data, date is max)
        let min_value = title_min_arr.value(0);
        let max_value = title_max_arr.value(0);
        assert_eq!(min_value, "apple", "minimum title should be 'apple'");
        assert_eq!(max_value, "date", "maximum title should be 'date'");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_combines_multiple_superfiles() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn crate::supertable::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create three superfiles with 2 rows each. Each batch gets a
        // unique word that survives tokenization (no underscores/numbers).
        let batch_titles = [
            ["alpha first", "alpha second"],
            ["beta first", "beta second"],
            ["gamma first", "gamma second"],
        ];
        for titles in &batch_titles {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(titles);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .superfile_list
            .superfiles
            .iter()
            .take(3)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 3, "should have 3 superfiles");

        // Merging 3 superfiles should succeed
        let merged_superfile = st
            .merge_superfiles(&superfiles)
            .await
            .expect("merge_superfiles should succeed");

        // Verify merged superfile stats
        assert_eq!(
            merged_superfile.entry.n_docs, 6,
            "merged superfile should have 6 documents (3 files × 2 docs each)"
        );

        let source_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let source_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);
        assert_eq!(merged_superfile.entry.id_min, source_id_min);
        assert_eq!(merged_superfile.entry.id_max, source_id_max);

        // Verify no data loss by querying the merged reader
        let merged_reader = merged_superfile
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        assert_eq!(merged_reader.n_docs(), 6, "reader should report 6 docs");

        // Each batch has 2 docs sharing a unique word — search for each batch's unique term
        for term in &["alpha", "beta", "gamma"] {
            let hits = merged_reader
                .token_match("title", &[*term], crate::BoolMode::And)
                .await
                .unwrap_or_else(|_| panic!("token_match for '{term}'"));
            assert_eq!(hits.len(), 2, "term '{term}' should match exactly 2 docs");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_single_superfile() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn crate::supertable::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Create a single superfile
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["only doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader
            .manifest()
            .superfile_list
            .superfiles
            .iter()
            .take(1)
            .cloned()
            .collect();

        assert_eq!(superfiles.len(), 1, "should have 1 superfile");

        // Merging a single superfile should succeed
        let merged_superfile = st
            .merge_superfiles(&superfiles)
            .await
            .expect("merge_superfiles should succeed");

        // Verify merged superfile stats
        assert_eq!(
            merged_superfile.entry.n_docs, 2,
            "merged superfile should have 2 documents"
        );

        let source_id_min = superfiles
            .iter()
            .map(|sf| sf.id_min)
            .min()
            .unwrap_or(i128::MAX);
        let source_id_max = superfiles
            .iter()
            .map(|sf| sf.id_max)
            .max()
            .unwrap_or(i128::MIN);
        assert_eq!(merged_superfile.entry.id_min, source_id_min);
        assert_eq!(merged_superfile.entry.id_max, source_id_max);

        // Verify no data loss by querying the merged reader
        let merged_reader = merged_superfile
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        assert_eq!(merged_reader.n_docs(), 2, "reader should report 2 docs");

        let only_hits = merged_reader
            .token_match("title", &["only"], crate::BoolMode::And)
            .await
            .expect("token_match for 'only'");
        assert_eq!(
            only_hits.len(),
            1,
            "should find exactly 1 doc matching 'only'"
        );

        let second_hits = merged_reader
            .token_match("title", &["second"], crate::BoolMode::And)
            .await
            .expect("token_match for 'second'");
        assert_eq!(
            second_hits.len(),
            1,
            "should find exactly 1 doc matching 'second'"
        );
    }
}
