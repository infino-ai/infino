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
        wal::{
            WalStore,
            tombstones_admin::{self, TombstonesAdminError},
        },
        writer::{PreparedSuperfile, ShardOutput, prepare_superfile, try_commit_attempt},
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
        .filter(|s| !s.sealed_by_other && s.size_bytes < target_bytes)
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
    pub fn compact(&self, cfg: &CompactionSettings) -> Result<(), CompactionError> {
        crate::runtime_bridge::bridge_on_runtime(
            self.compact_async(cfg),
            &self.inner().query_runtime(),
        )
    }

    pub(crate) async fn compact_async(
        &self,
        cfg: &CompactionSettings,
    ) -> Result<(), CompactionError> {
        let inner = self.inner();
        let manifest = inner.manifest.load_full();

        let storage = manifest
            .options
            .storage
            .as_ref()
            .ok_or(CompactionError::NoStorage)?
            .clone();
        let wal_store = WalStore::new(storage.clone());

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
            self.run_compaction_job(job).await?;
            self.refresh()
                .await
                .map_err(|e| CompactionError::Refresh(e.to_string()))?;
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

    pub(crate) async fn run_compaction_job(
        &self,
        job: CompactionJob,
    ) -> Result<(), CompactionError> {
        let inner = self.inner();
        let manifest = inner.manifest.load_full();
        let storage = manifest
            .options
            .storage
            .as_ref()
            .ok_or(CompactionError::NoStorage)?
            .clone();
        let wal_store = WalStore::new(storage.clone());

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

        let merged_segment = self
            .merge_superfiles(&inputs)
            .await
            .map_err(|e| CompactionError::Build(e.to_string()))?;

        let new_entries = vec![merged_segment.entry];
        let mut pending_storage_writes = vec![
            merged_segment
                .bytes_for_storage
                .ok_or(CompactionError::EmptyMergedSuperfile)?,
        ];

        let opts = Arc::clone(&inner.options);
        let new_manifest_id = manifest.manifest_id + 1;
        try_commit_attempt(
            storage.clone(),
            opts,
            manifest.clone(),
            &new_entries,
            &inputs,
            new_manifest_id,
            &mut pending_storage_writes,
        )
        .await
        .map_err(|e| CompactionError::Commit(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoolMode;
    use crate::Supertable;
    use crate::supertable::storage::LocalFsStorageProvider;
    use crate::test_helpers::{build_title_batch, default_supertable_options};
    use std::collections::HashSet;
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

    // ─── Helpers shared by the end-to-end compact() tests ─────────────────

    fn make_st(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn crate::supertable::storage::StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
            .expect("create supertable")
    }

    /// Compact config designed to trigger on tiny test superfiles.
    /// target = 1 MiB, fill floor = 1 % → min_output_bytes ≈ 10 KiB.
    /// Individual files must be < 10 KiB to be candidates; their
    /// combined live_bytes must reach 10 KiB for a job to be emitted.
    fn small_compact_cfg() -> CompactionSettings {
        CompactionSettings {
            target_superfile_size_mb: 1,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        }
    }

    fn commit_titles(st: &Supertable, titles: &[&str]) {
        let mut w = st.writer().expect("writer");
        w.append(&build_title_batch(titles)).expect("append");
        w.commit().expect("commit");
    }

    // ─── End-to-end compact() tests ────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_reduces_superfile_count() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits, each with a unique first word so the merged bloom is verifiable.
        // 10 × ~1217 bytes ≈ 12 170 bytes > min_output_bytes (~10 485) → job emitted.
        commit_titles(&st, &["alpha cherry", "alpha mango"]);
        commit_titles(&st, &["bravo cherry", "bravo mango"]);
        commit_titles(&st, &["charlie delta", "charlie echo"]);
        commit_titles(&st, &["foxtrot golf", "foxtrot hotel"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["lima first", "lima second"]);
        commit_titles(&st, &["november first", "november second"]);
        commit_titles(&st, &["quebec first", "quebec second"]);
        commit_titles(&st, &["romeo first", "romeo second"]);
        commit_titles(&st, &["sierra first", "sierra second"]);

        let before = st.reader();
        let before_manifest_id = before.manifest_id();
        let before_n_superfiles = before.n_superfiles();
        let input_ids: HashSet<Uuid> = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        let expected_docs = before.n_docs_total();
        let expected_id_min = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.id_min)
            .min()
            .expect("at least one superfile before compaction");
        let expected_id_max = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.id_max)
            .max()
            .expect("at least one superfile before compaction");

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let after = st.reader();
        let sfs = &after.manifest().superfiles;

        assert!(
            after.manifest_id() == before_manifest_id + 1,
            "no compaction jobs ran; adjust small_compact_cfg() if superfiles exceed \
             min_output_bytes"
        );
        assert!(
            sfs.len() < before_n_superfiles,
            "superfile count should decrease after compaction"
        );
        assert!(
            !sfs.iter().any(|s| input_ids.contains(&s.superfile_id)),
            "original superfile IDs must not appear after compaction"
        );

        // Doc count preserved across the merge
        assert_eq!(after.n_docs_total(), expected_docs);

        // Merged entry ID range spans all original inputs
        let merged_min = sfs
            .iter()
            .map(|s| s.id_min)
            .min()
            .expect("at least one superfile after compaction");
        let merged_max = sfs
            .iter()
            .map(|s| s.id_max)
            .max()
            .expect("at least one superfile after compaction");
        assert!(merged_min == expected_id_min);
        assert!(merged_max == expected_id_max);

        // Partition key consistent across all remaining superfiles
        assert!(sfs.iter().all(|s| s.partition_key == sfs[0].partition_key));

        // FTS bloom covers the unique first word from each of the 10 input batches
        let fts = sfs[0]
            .fts_summary
            .get("title")
            .expect("fts summary present");
        for term in &[
            b"alpha" as &[u8],
            b"bravo",
            b"charlie",
            b"foxtrot",
            b"india",
            b"lima",
            b"november",
            b"quebec",
            b"romeo",
            b"sierra",
        ] {
            assert!(
                fts.term_bloom.contains(term),
                "bloom missing term '{}'",
                std::str::from_utf8(term).expect("term literal is valid utf-8")
            );
        }

        // Box::leak(dir);
        std::mem::forget(dir);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_no_op_when_single_superfile() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["only doc", "second doc"]);

        let before_manifest_id = st.manifest_id();
        let before_n = st.reader().n_superfiles();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert_eq!(
            st.manifest_id(),
            before_manifest_id,
            "manifest_id must not change: a single superfile cannot form a merge job"
        );
        assert_eq!(st.reader().n_superfiles(), before_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_no_op_when_below_fill_floor() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["beta first", "beta second"]);

        let before_manifest_id = st.manifest_id();

        // fill floor = 100% of 1 GiB → min_output_bytes = 1 GiB.
        // Both tiny superfiles are candidates (each < 1 GiB) but their
        // combined live_bytes is far below 1 GiB, so no job is emitted.
        let cfg = CompactionSettings {
            target_superfile_size_mb: 1024,
            min_fill_percent: 100,
            ..CompactionSettings::default()
        };
        st.compact_async(&cfg).await.expect("compact");

        assert_eq!(
            st.manifest_id(),
            before_manifest_id,
            "manifest must not change when combined size is below the fill floor"
        );
        assert_eq!(st.reader().n_superfiles(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reader_pinned_before_compact_sees_old_state() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // Pin a snapshot before compaction.
        let reader_before = st.reader();
        let before_n = reader_before.n_superfiles();
        let before_manifest_id = reader_before.manifest_id();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let reader_after = st.reader();

        // The pinned snapshot must be frozen — it still sees the original superfiles.
        assert_eq!(reader_before.n_superfiles(), before_n);
        assert_eq!(reader_before.manifest_id(), before_manifest_id);

        // A freshly-opened reader must reflect the post-compact manifest.
        assert!(
            reader_after.manifest_id() > before_manifest_id,
            "compact must have run for snapshot isolation to be observable; \
             adjust small_compact_cfg() if needed"
        );
        assert!(reader_after.n_superfiles() < before_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fts_search_returns_correct_results_after_compact() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits so combined size exceeds min_output_bytes.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let before_manifest_id = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert!(
            st.manifest_id() == before_manifest_id + 1,
            "compact must have run; adjust small_compact_cfg() if needed"
        );

        // Each batch-unique term should match exactly 2 docs.
        for term in &["alpha", "bravo", "charlie"] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 2, "term '{term}' should match 2 docs after compact");
        }

        // The shared token 'first' appears once per batch: 10 batches → 10 docs.
        let n_first: usize = st
            .token_match("title", "first", BoolMode::And, None)
            .expect("token_match for 'first'")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n_first, 10, "'first' should match 10 docs");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fts_bloom_filter_covers_all_terms_after_compact() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Ten commits (2 docs each) so combined size exceeds min_output_bytes.
        // Each commit has a unique first word; all must survive in the merged bloom.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        let before_manifest_id = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert!(
            st.manifest_id() == before_manifest_id + 1,
            "compact must have run; adjust small_compact_cfg() if needed"
        );

        let r = st.reader();
        let sfs = &r.manifest().superfiles;
        assert!(sfs.len() < 10, "superfile count should have decreased");

        let fts = sfs[0]
            .fts_summary
            .get("title")
            .expect("fts summary present");
        for term in &[
            b"alpha" as &[u8],
            b"bravo",
            b"charlie",
            b"delta",
            b"echo",
            b"foxtrot",
            b"golf",
            b"hotel",
            b"india",
            b"juliet",
        ] {
            assert!(
                fts.term_bloom.contains(term),
                "bloom missing term '{}'",
                std::str::from_utf8(term).expect("term literal is valid utf-8")
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn second_compact_is_no_op_after_full_merge() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // First compact: merges all 10 tiny superfiles into one.
        let before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");
        assert!(
            st.manifest_id() == before_first_compact + 1,
            "first compact must have run; adjust small_compact_cfg() if needed"
        );
        assert_eq!(st.inner().manifest.load_full().superfiles.len(), 1);

        let after_first_manifest_id = st.manifest_id();
        let after_first_n = st.reader().n_superfiles();

        // Second compact on the same data: the merged superfile is the only
        // file in its partition, so pack_partition emits no job (needs ≥ 2 inputs).
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        assert_eq!(
            st.manifest_id(),
            after_first_manifest_id,
            "second compact should produce no jobs"
        );
        assert_eq!(st.reader().n_superfiles(), after_first_n);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_runs_multiple_compactions_on_separate_file_sets() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Batch A: ten superfiles with group-A terms (2 docs each = 20 docs total).
        // 10 × ~1217 bytes ≈ 12 170 bytes > min_output_bytes → job emitted.
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);
        commit_titles(&st, &["echo first", "echo second"]);
        commit_titles(&st, &["foxtrot first", "foxtrot second"]);
        commit_titles(&st, &["golf first", "golf second"]);
        commit_titles(&st, &["hotel first", "hotel second"]);
        commit_titles(&st, &["india first", "india second"]);
        commit_titles(&st, &["juliet first", "juliet second"]);

        // First compact: merges the ten batch-A superfiles into one.
        let before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");

        let manifest_id_after_first_compact = st.manifest_id();
        assert_eq!(manifest_id_after_first_compact, before_first_compact + 1);
        assert_eq!(
            st.reader().n_docs_total(),
            20,
            "batch A should have 20 docs"
        );

        // Batch B: ten more superfiles with group-B terms (2 docs each = 20 docs).
        commit_titles(&st, &["kilo first", "kilo second"]);
        commit_titles(&st, &["lima first", "lima second"]);
        commit_titles(&st, &["mike first", "mike second"]);
        commit_titles(&st, &["november first", "november second"]);
        commit_titles(&st, &["oscar first", "oscar second"]);
        commit_titles(&st, &["papa first", "papa second"]);
        commit_titles(&st, &["quebec first", "quebec second"]);
        commit_titles(&st, &["romeo first", "romeo second"]);
        commit_titles(&st, &["sierra first", "sierra second"]);
        commit_titles(&st, &["tango first", "tango second"]);

        // Second compact: runs a job on the new batch-B superfiles.
        // The merged-A superfile is above min_output_bytes so it is not a
        // candidate; the ten batch-B files combine to exceed the floor.
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        // The manifest must have advanced past the ten batch-B commits.
        assert!(
            st.manifest_id() == manifest_id_after_first_compact + 10 + 1,
            "second compact must have run a job on the batch-B superfiles"
        );

        // All 40 docs must be visible after both compaction rounds.
        let r = st.reader();
        assert_eq!(r.n_docs_total(), 40, "all docs must be preserved");
        assert!(
            r.n_superfiles() < 8,
            "overall superfile count must have decreased from original 20"
        );

        // Manifest consistency: per-entry doc counts sum to 40.
        let sfs = &r.manifest().superfiles;
        let total_from_manifest: u64 = sfs.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_from_manifest, 40);

        // ID range is monotonically ordered within each remaining superfile.
        for sf in sfs.iter() {
            assert!(sf.id_min <= sf.id_max);
        }

        drop(r);

        // FTS: every batch-unique term must be searchable and return exactly 2 docs.
        for term in &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
            "sierra", "tango",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 2, "term '{term}' should match exactly 2 docs");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_runs_multiple_compactions_on_separate_file_sets_in_same_job() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Each superfile must be large enough that 30 combined overflow the 1 MiB
        // target, forcing the selector to emit two jobs. Write 4096 batches per
        // commit so each superfile holds 4096 × 2 = 8192 docs.
        let commit_bulk = |titles: &[&str]| {
            let mut w = st.writer().expect("writer");
            for _ in 0..4096 {
                w.append(&build_title_batch(titles)).expect("append");
            }
            w.commit().expect("commit");
        };

        // Batch A: ten superfiles; 10 × 8192 = 81920 docs total.
        commit_bulk(&["alpha first", "alpha second"]);
        commit_bulk(&["bravo first", "bravo second"]);
        commit_bulk(&["charlie first", "charlie second"]);
        commit_bulk(&["delta first", "delta second"]);
        commit_bulk(&["echo first", "echo second"]);
        commit_bulk(&["foxtrot first", "foxtrot second"]);
        commit_bulk(&["golf first", "golf second"]);
        commit_bulk(&["hotel first", "hotel second"]);
        commit_bulk(&["india first", "india second"]);
        commit_bulk(&["juliet first", "juliet second"]);

        // Batch B: twenty superfiles (2 iterations × 10 terms); 20 × 8192 = 163840 docs total.
        for _ in 0..2 {
            commit_bulk(&["kilo first", "kilo second"]);
            commit_bulk(&["lima first", "lima second"]);
            commit_bulk(&["mike first", "mike second"]);
            commit_bulk(&["november first", "november second"]);
            commit_bulk(&["oscar first", "oscar second"]);
            commit_bulk(&["papa first", "papa second"]);
            commit_bulk(&["quebec first", "quebec second"]);
            commit_bulk(&["romeo first", "romeo second"]);
            commit_bulk(&["sierra first", "sierra second"]);
            commit_bulk(&["tango first", "tango second"]);
        }

        // 30 superfiles total; 81920 + 163840 = 245760 docs.
        let manifest_id_before_first_compact = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact");

        // compact() must have run two jobs (one per file set → manifest +2).
        assert!(
            st.manifest_id() == manifest_id_before_first_compact + 2,
            "compact must have run two jobs, one per file set"
        );

        // All 245760 docs must be visible after compaction.
        let r = st.reader();
        assert_eq!(r.n_docs_total(), 245760, "all docs must be preserved");
        assert!(
            r.n_superfiles() == 2,
            "overall superfile count must have decreased from original 30"
        );

        // Manifest consistency: per-entry doc counts sum to 245760.
        let sfs = &r.manifest().superfiles;
        let total_from_manifest: u64 = sfs.iter().map(|s| s.n_docs).sum();
        assert_eq!(total_from_manifest, 245760);

        // ID range is monotonically ordered within each remaining superfile.
        for sf in sfs.iter() {
            assert!(sf.id_min <= sf.id_max);
        }

        drop(r);

        // FTS: batch-A terms committed once → 1 × 8192 = 8192 hits each.
        for term in &[
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
            "juliet",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 8192, "term '{term}' should match exactly 8192 docs");
        }

        // FTS: batch-B terms committed twice → 2 × 8192 = 16384 hits each.
        for term in &[
            "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra",
            "tango",
        ] {
            let n: usize = st
                .token_match("title", term, BoolMode::And, None)
                .unwrap_or_else(|e| panic!("token_match for '{term}': {e}"))
                .iter()
                .map(|b| b.num_rows())
                .sum();
            assert_eq!(n, 16384, "term '{term}' should match exactly 16384 docs");
        }
    }
}
