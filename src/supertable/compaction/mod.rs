// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Picks which superfiles to merge.
//!
//! no I/O. `supertable::compact` gathers the
//! stats, calls [`select`], then merges each [`CompactionJob`].
//! Compaction is single-level — a target-sized superfile is never
//! re-compacted.

use std::{
    collections::{BTreeMap, HashMap},
    mem,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use bytes::Bytes;
use chrono::Utc;
use futures::{
    future::join_all,
    stream::{self, StreamExt},
};
use roaring::RoaringBitmap;
use tokio::time;
use tracing::warn;
use uuid::Uuid;

use crate::{
    Supertable,
    config::CompactionSettings,
    runtime_bridge::bridge_on_runtime,
    superfile::builder::SuperfileBuilder,
    supertable::{
        BuildError, CommitError, ManifestSnapshot, SuperfileEntry,
        error::CompactionError,
        query::dispatch::open_reader,
        wal::{
            Etag, SealRecord, TombstonesSidecar, WalStore,
            tombstones_admin::{self, TombstonesAdminError},
        },
        writer::{
            PreparedSuperfile, ShardOutput, backoff_delay, prepare_superfile, try_commit_attempt,
        },
    },
};

struct CompactionSlot<'a>(&'a AtomicBool);

impl Drop for CompactionSlot<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

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
    let max_memory_bytes = cfg.max_memory_mb.saturating_mul(MIB);

    let mut by_partition: BTreeMap<&[u8], Vec<&SuperfileStats>> = BTreeMap::new();
    for s in superfiles {
        by_partition.entry(&s.partition_key).or_default().push(s);
    }

    let mut jobs = Vec::new();
    for (key, segs) in by_partition {
        pack_partition(
            key,
            segs,
            target_bytes,
            min_output_bytes,
            max_memory_bytes,
            &mut jobs,
        );
    }
    jobs
}

fn pack_partition(
    key: &[u8],
    segs: Vec<&SuperfileStats>,
    target_bytes: u64,
    min_output_bytes: u64,
    max_memory_bytes: u64,
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
        if !pending.fits(s, target_bytes, max_memory_bytes) {
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
    raw_bytes: u64,
}

impl PendingJob {
    fn fits(&self, s: &SuperfileStats, target_bytes: u64, max_memory_bytes: u64) -> bool {
        self.live_bytes + s.live_bytes() <= target_bytes
            && self.raw_bytes + s.size_bytes <= max_memory_bytes
    }

    fn push(&mut self, s: &SuperfileStats) {
        self.raw_bytes += s.size_bytes;
        self.inputs.push(s.superfile_id);
        self.live_bytes += s.live_bytes();
    }

    /// Emit a CompactionJob if ≥ 2 inputs and live bytes reach `min_output_bytes`.
    fn emit(&mut self, key: &[u8], min_output_bytes: u64, jobs: &mut Vec<CompactionJob>) {
        if self.inputs.len() >= 2 && self.live_bytes >= min_output_bytes {
            jobs.push(CompactionJob {
                partition_key: key.to_vec(),
                inputs: mem::take(&mut self.inputs),
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
    pub(crate) fn compact(&self, cfg: &CompactionSettings) -> Result<(), CompactionError> {
        bridge_on_runtime(self.compact_async(cfg), &self.inner().query_runtime())
    }

    pub(crate) async fn compact_async(
        &self,
        cfg: &CompactionSettings,
    ) -> Result<(), CompactionError> {
        let inner = self.inner();

        match inner.compaction_outstanding.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ) {
            Ok(_) => {}
            Err(_) => return Err(CompactionError::AlreadyCompacting),
        }
        let _slot = CompactionSlot(&inner.compaction_outstanding);

        let manifest = inner.manifest.load_full();

        // Prefetch sidecars using the cache to batch storage GETs.
        // This populates both bitmap and seal information for all superfiles.
        // The cache returns empty bitmaps for superfiles without tombstones.
        let superfile_ids: Vec<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();

        let sidecar_map: HashMap<Uuid, (Arc<RoaringBitmap>, Option<SealRecord>)> =
            if let Some(cache) = &inner.tombstone_cache {
                let now = Instant::now();
                cache.prefetch(&superfile_ids, now).await;

                // Build a map of superfile_id → (bitmap, seal) by checking the cache.
                // Cache hits are O(1); any misses are already prefetched above.
                superfile_ids
                    .iter()
                    .filter_map(|id| match cache.sidecar_for(*id, now) {
                        Ok((bitmap, seal)) => Some((*id, (bitmap, seal))),
                        Err(_) => None,
                    })
                    .collect()
            } else {
                // Fallback for in-memory-only tables (no storage, no tombstone cache).
                HashMap::new()
            };

        // Build SuperfileStats for every superfile in the snapshot.
        let now = Utc::now();
        let stale_seal_timeout = std::time::Duration::from_millis(cfg.stale_seal_timeout_ms);
        let stats: Vec<SuperfileStats> = manifest
            .get_all_superfiles()
            .iter()
            .map(|entry| {
                let (bitmap, seal) = sidecar_map
                    .get(&entry.superfile_id)
                    .cloned()
                    .unwrap_or_else(|| (Arc::new(RoaringBitmap::new()), None));
                let tombstoned_docs = bitmap.len();
                let sealed_by_other = seal.as_ref().is_some_and(|s| {
                    !tombstones_admin::is_seal_stale(s.sealed_at, now, stale_seal_timeout)
                });
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
            self.run_compaction_job(job, stale_seal_timeout).await?;
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

        // This reserves budget for the whole input size since merge still
        // loads it all at once. Real fix is streaming the merge and pooling
        // buffers instead of a flat reservation; picking that up later.
        let input_bytes: u64 = superfiles
            .iter()
            .map(|e| e.subsection_offsets.as_ref().map_or(0, |o| o.total_size))
            .sum();
        // double the input bytes to account for the merge buffer and any overhead
        let estimated_bytes = input_bytes.saturating_mul(2) as usize;
        let _memory_reservation = manifest
            .options
            .connection_memory_budget
            .try_reserve(estimated_bytes)
            .map_err(|e| BuildError::MemoryBudgetExceeded(e.to_string()))?;

        let mut superfile_readers_fut = Vec::with_capacity(superfiles.len());
        for entry in superfiles {
            let open_fut = async {
                let r = open_reader(&store, disk_cache.as_ref(), storage.as_ref(), entry).await;
                (entry.superfile_id, r)
            };
            superfile_readers_fut.push(open_fut);
        }
        let readers = join_all(superfile_readers_fut).await;

        let now = Instant::now();
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
        stale_seal_timeout: std::time::Duration,
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
                    .get_all_superfiles()
                    .iter()
                    .find(|e| e.superfile_id == *id)
                    .cloned()
                    .ok_or(CompactionError::SuperfileNotFound(*id))
            })
            .collect::<Result<_, _>>()?;

        let opts = Arc::clone(&inner.options);
        let max_retries = opts.max_commit_retries.max(1);

        // Seal every input sidecar so no writer can land a tombstone
        // on a file that's about to disappear, and so another
        // compactor doesn't pick up the same inputs. If we die
        // before unsealing (crash, not a caught error), `seal`
        // itself lets a later compactor take over once the seal
        // goes stale.
        let compaction_id = Uuid::new_v4();
        let sealed_at = Utc::now();
        let mut sealed: Vec<SealedInput> = Vec::with_capacity(inputs.len());
        for entry in &inputs {
            let (sidecar, etag) = match seal_with_bounded_retry(
                &wal_store,
                entry.superfile_id,
                compaction_id,
                sealed_at,
                stale_seal_timeout,
                max_retries,
            )
            .await
            {
                Ok(v) => v,
                Err(e) => {
                    unseal_all(&wal_store, sealed).await;
                    return Err(e);
                }
            };
            sealed.push(SealedInput {
                superfile_id: entry.superfile_id,
                bitmap: sidecar.bitmap,
                etag,
            });
        }

        let merged_segment = match self.merge_superfiles(&inputs).await {
            Ok(seg) => seg,
            Err(e) => {
                unseal_all(&wal_store, sealed).await;
                return Err(CompactionError::Build(e.to_string()));
            }
        };

        let PreparedSuperfile {
            entry: merged_entry,
            bytes_for_store,
            bytes_for_storage,
            bytes_for_cache,
        } = merged_segment;
        let merged_superfile_id = merged_entry.superfile_id;
        let new_entries = vec![merged_entry];
        let mut pending_storage_writes =
            vec![bytes_for_storage.ok_or(CompactionError::EmptyMergedSuperfile)?];

        for attempt in 0..max_retries {
            let current = inner.manifest.load_full();

            // Another compactor already merged our inputs — nothing left to commit.
            let entries_to_remove = match resolve_entries_to_remove(&current, &job.inputs) {
                Ok(entries) => entries,
                Err(_missing) => return Ok(()),
            };

            match try_commit_attempt(
                storage.clone(),
                Arc::clone(&opts),
                current,
                &new_entries,
                &entries_to_remove,
                &mut pending_storage_writes,
            )
            .await
            {
                Ok(_) => {
                    // Warm the merged superfile into the cache, same as a
                    // normal writer commit does. Without this every query
                    // against it misses and re-fetches + re-opens from
                    // storage every single time.
                    if let Some((uri, bytes)) = bytes_for_store
                        && let Err(e) = opts.store.insert(uri, bytes)
                    {
                        warn!(
                            superfile_id = %merged_superfile_id,
                            error = %e,
                            "compact: failed to warm reader cache for merged superfile"
                        );
                    }
                    // Same warm-up, disk-cache side: mirrors what a normal
                    // write does when prepopulate_cache_on_commit is on.
                    // Shares the writer's own insert_warm_or_warn so the
                    // two paths can't drift out of sync again.
                    if let Some((uri, bytes)) = bytes_for_cache
                        && let Some(cache) = opts.disk_cache.as_ref()
                    {
                        cache.insert_warm_or_warn(&uri, bytes).await;
                    }
                    // Drop the merged-away inputs so the in-memory cache
                    // doesn't grow forever across repeated compactions.
                    // The disk cache is already size-bounded (LRU), so its
                    // stale entries just age out on their own.
                    for entry in &entries_to_remove {
                        opts.store.remove(&entry.uri);
                    }
                    return Ok(());
                }
                Err(CommitError::WriteContentionExhausted) if attempt + 1 < max_retries => {
                    if let Err(e) = self.refresh().await {
                        unseal_all(&wal_store, sealed).await;
                        return Err(CompactionError::Refresh(e.to_string()));
                    }
                    // Input vanished mid-retry (someone else merged it away).
                    // Our built output no longer matches reality, so abort
                    // instead of retrying the commit.
                    if let Err(missing) =
                        resolve_entries_to_remove(&inner.manifest.load_full(), &job.inputs)
                    {
                        unseal_all(&wal_store, sealed).await;
                        return Err(CompactionError::SuperfileNotFound(missing));
                    }
                    time::sleep(backoff_delay(attempt)).await;
                }
                Err(e) => {
                    unseal_all(&wal_store, sealed).await;
                    return Err(CompactionError::Commit(e.to_string()));
                }
            }
        }

        unseal_all(&wal_store, sealed).await;
        Err(CompactionError::Commit(
            "commit retries exhausted".to_string(),
        ))
    }
}

/// One superfile this attempt sealed: enough to unseal it later with
/// no extra GET (`unseal` uses the etag + bitmap straight from `seal`).
struct SealedInput {
    superfile_id: Uuid,
    bitmap: RoaringBitmap,
    etag: Etag,
}

/// Cap on in-flight unseal calls. Single-writer model: one compactor
/// commits at a time, so there's no throughput reason to fire every
/// unseal at once.
const MAX_CONCURRENT_UNSEALS: usize = 8;

/// Best-effort: clear every seal this attempt placed. Each one is an
/// independent sidecar, so order doesn't matter, but they're bounded
/// to a small number in flight rather than all at once.
async fn unseal_all(wal_store: &WalStore, sealed: Vec<SealedInput>) {
    let results = stream::iter(sealed.into_iter().map(|s| {
        let wal_store = wal_store.clone();
        async move {
            let result =
                tombstones_admin::unseal(&wal_store, s.superfile_id, s.bitmap, &s.etag).await;
            (s.superfile_id, result)
        }
    }))
    .buffer_unordered(MAX_CONCURRENT_UNSEALS)
    .collect::<Vec<_>>()
    .await;
    for (superfile_id, result) in results {
        if let Err(e) = result {
            warn!(superfile_id = %superfile_id, error = %e, "compact: failed to unseal after aborting");
        }
    }
}

/// Look up `job_inputs` in `current`, in order. `Err` carries the first
/// missing id (removed by another compactor).
fn resolve_entries_to_remove(
    current: &ManifestSnapshot,
    job_inputs: &[Uuid],
) -> Result<Vec<Arc<SuperfileEntry>>, Uuid> {
    job_inputs
        .iter()
        .map(|id| {
            current
                .get_all_superfiles()
                .iter()
                .find(|e| e.superfile_id == *id)
                .cloned()
                .ok_or(*id)
        })
        .collect()
}

/// Seal one input, retrying a CAS race with a writer up to `max_retries`
/// times with backoff. `CasLost` just means a writer landed a tombstone
/// bit between our read and write — not an abandoned compaction.
async fn seal_with_bounded_retry(
    wal_store: &WalStore,
    superfile_id: Uuid,
    compaction_id: Uuid,
    sealed_at: chrono::DateTime<Utc>,
    stale_seal_timeout: std::time::Duration,
    max_retries: u32,
) -> Result<(TombstonesSidecar, Etag), CompactionError> {
    for attempt in 0..max_retries {
        match tombstones_admin::seal(
            wal_store,
            superfile_id,
            compaction_id,
            sealed_at,
            stale_seal_timeout,
        )
        .await
        {
            Ok(sealed) => return Ok(sealed),
            Err(TombstonesAdminError::CasLost { .. }) if attempt + 1 < max_retries => {
                time::sleep(backoff_delay(attempt)).await;
            }
            Err(TombstonesAdminError::CasLost { .. }) => {
                return Err(CompactionError::Seal("seal retries exhausted".to_string()));
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
    Err(CompactionError::Seal("seal retries exhausted".to_string()))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, mem, str, sync::Arc};

    use arrow_array::LargeStringArray;
    use tempfile::TempDir;
    use tokio::task;

    use super::*;
    use crate::{
        BoolMode, Supertable,
        config::DEFAULT_STALE_SEAL_TIMEOUT_MS,
        memory::ConnectionMemoryBudget,
        supertable::{
            error::CompactionError,
            storage::{LocalFsStorageProvider, StorageProvider},
        },
        test_helpers::{build_title_batch, default_supertable_options},
    };

    const DEFAULT_STALE_SEAL_TIMEOUT: std::time::Duration =
        std::time::Duration::from_millis(DEFAULT_STALE_SEAL_TIMEOUT_MS);

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

    // ---- SuperfileStats live_docs / live_bytes -----------------------

    #[test]
    fn live_docs_subtracts_tombstones_and_saturates() {
        let s = seg(1, 100, 1000, 250);
        assert_eq!(s.live_docs(), 750);
        // More tombstones than docs saturates to zero rather than
        // underflowing.
        let over = seg(2, 100, 100, 200);
        assert_eq!(over.live_docs(), 0);
    }

    #[test]
    fn live_bytes_scales_by_live_fraction() {
        // 100 MiB, half the docs tombstoned → ~50 MiB live.
        let s = seg(1, 100, 1000, 500);
        assert_eq!(s.live_bytes(), mib(100) / 2);
    }

    #[test]
    fn live_bytes_zero_docs_is_zero() {
        // A 0-doc superfile must report 0 live bytes (guards the
        // division-by-zero branch).
        let s = seg(1, 100, 0, 0);
        assert_eq!(s.live_bytes(), 0);
    }

    // ---- PendingJob fits / push -------------------------------------

    #[test]
    fn pending_job_fits_until_target_exceeded() {
        let target = mib(100);
        let max_memory = mib(1000);
        let mut p = PendingJob::default();
        let a = seg(1, 60, 1000, 0); // 60 MiB live
        assert!(p.fits(&a, target, max_memory));
        p.push(&a);
        assert_eq!(p.live_bytes, mib(60));
        assert_eq!(p.inputs.len(), 1);
        // A second 60 MiB superfile would overflow the 100 MiB target.
        let b = seg(2, 60, 1000, 0);
        assert!(!p.fits(&b, target, max_memory));
        // A 40 MiB superfile fits exactly to the boundary.
        let c = seg(3, 40, 1000, 0);
        assert!(p.fits(&c, target, max_memory));
    }

    #[test]
    fn pending_job_fits_respects_max_memory_even_under_target() {
        // live_bytes fits comfortably under target, but raw size_bytes
        // (pre-tombstone) would blow past a tight memory ceiling.
        let target = mib(1000);
        let max_memory = mib(100);
        let mut p = PendingJob::default();
        let a = seg(1, 60, 1000, 0); // 60 MiB raw, 60 MiB live
        assert!(p.fits(&a, target, max_memory));
        p.push(&a);
        let b = seg(2, 60, 1000, 0); // would push raw to 120 MiB > 100 MiB cap
        assert!(!p.fits(&b, target, max_memory));
    }

    #[test]
    fn pending_job_emit_requires_two_inputs() {
        // A single-input pending job never emits even if it reaches
        // the fill floor.
        let mut jobs = Vec::new();
        let mut p = PendingJob::default();
        p.push(&seg(1, 200, 1000, 0));
        p.emit(&[], 0, &mut jobs);
        assert!(jobs.is_empty(), "single-input job must not emit");
        // Reset to default after emit attempt.
        assert_eq!(p.inputs.len(), 0);
        assert_eq!(p.live_bytes, 0);
    }

    // ---- run_compaction_job error arms ------------------------------

    #[tokio::test(flavor = "multi_thread")]
    async fn run_compaction_job_unknown_input_surfaces_not_found() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);
        // A job referencing a superfile id that isn't in the manifest
        // must surface SuperfileNotFound.
        let bogus = Uuid::from_u128(0xDEAD_BEEF);
        let job = CompactionJob {
            partition_key: Vec::new(),
            inputs: vec![bogus],
            estimated_output_bytes: 0,
        };
        let err = st
            .run_compaction_job(job, DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect_err("must error on unknown input");
        assert!(
            matches!(err, CompactionError::SuperfileNotFound(id) if id == bogus),
            "{err:?}"
        );
    }

    /// Resolves every present input in order; reports the missing one by id.
    #[tokio::test(flavor = "multi_thread")]
    async fn resolve_entries_to_remove_reports_the_missing_input() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let manifest = st.inner().manifest.load_full();
        let ids: Vec<Uuid> = manifest
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(ids.len(), 2);

        // All present.
        let resolved = resolve_entries_to_remove(&manifest, &ids).expect("both inputs are present");
        assert_eq!(
            resolved.iter().map(|e| e.superfile_id).collect::<Vec<_>>(),
            ids
        );

        // One missing.
        let vanished = Uuid::from_u128(0xDEAD_BEEF);
        let mut job_inputs = ids.clone();
        job_inputs.push(vanished);
        let err = resolve_entries_to_remove(&manifest, &job_inputs)
            .expect_err("a missing input must be reported");
        assert_eq!(err, vanished);
    }

    /// If one input is already sealed by a different, still-live
    /// compaction, we abort -- but must unseal whatever we already
    /// sealed ourselves this attempt, not leave it stranded.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_unseals_its_own_inputs_when_a_later_one_conflicts() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let entries = st.reader().manifest().superfiles.clone();
        assert_eq!(entries.len(), 2);
        let (entry_a, entry_b) = (&entries[0], &entries[1]);

        // entry_b is already held by a different, still-live compaction.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let foreign_cid = Uuid::new_v4();
        tombstones_admin::seal(
            &wal_store,
            entry_b.superfile_id,
            foreign_cid,
            Utc::now(),
            DEFAULT_STALE_SEAL_TIMEOUT,
        )
        .await
        .expect("seal entry_b as foreign");

        let job = CompactionJob {
            partition_key: entry_a.partition_key.clone(),
            inputs: vec![entry_a.superfile_id, entry_b.superfile_id],
            estimated_output_bytes: 1,
        };
        let err = st
            .run_compaction_job(job, DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect_err("must conflict on entry_b");
        assert!(matches!(err, CompactionError::SidecarConflict { .. }));

        // entry_a got sealed by us first, then unsealed on the abort.
        let (sidecar_a, _) = wal_store
            .get_tombstones(entry_a.superfile_id)
            .await
            .expect("get")
            .expect("present");
        assert!(sidecar_a.seal.is_none());

        // entry_b's foreign seal is untouched -- it's not ours to clear.
        let (sidecar_b, _) = wal_store
            .get_tombstones(entry_b.superfile_id)
            .await
            .expect("get")
            .expect("present");
        assert_eq!(
            sidecar_b.seal.expect("still sealed").compaction_id,
            foreign_cid
        );
    }

    /// A stale seal (left behind by a crashed compactor, no error
    /// ever caught to clean it up) must not exclude its superfile
    /// from selection forever. Once it's older than
    /// `DEFAULT_STALE_SEAL_TIMEOUT`, a fresh `compact_async` call
    /// must pick it up and actually merge it.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_recovers_a_superfile_stuck_under_a_stale_seal() {
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

        let entries = st.reader().manifest().superfiles.clone();
        let crashed_entry = &entries[0];

        // Simulate a compactor that sealed this file and then died
        // long enough ago that its seal is now stale.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let old_time = Utc::now()
            - chrono::Duration::from_std(DEFAULT_STALE_SEAL_TIMEOUT).unwrap_or_default()
            - chrono::Duration::seconds(1);
        tombstones_admin::seal(
            &wal_store,
            crashed_entry.superfile_id,
            Uuid::new_v4(),
            old_time,
            DEFAULT_STALE_SEAL_TIMEOUT,
        )
        .await
        .expect("simulate a stale seal");

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact must succeed and recover the stale seal");

        // The stuck superfile must not still be sitting in the
        // manifest under its original id -- it has to have actually
        // been picked up and merged, not just left alone while its
        // 9 unsealed siblings merged around it.
        let still_stuck = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .any(|s| s.superfile_id == crashed_entry.superfile_id);
        assert!(
            !still_stuck,
            "the stale-sealed superfile must have been merged, not left behind"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_sync_wrapper_runs_jobs() {
        // Exercise the sync `compact()` entry point (the
        // runtime-bridge wrapper around `compact_async`). Use
        // spawn_blocking so we're not inside a tokio runtime when
        // the bridge tries to block.
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for titles in [
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, &titles);
        }
        let before = st.manifest_id();
        let cfg = small_compact_cfg();
        task::spawn_blocking(move || st.compact(&cfg).map(|_| st.manifest_id()))
            .await
            .expect("join")
            .map(|after| {
                assert!(after > before, "sync compact must have run a job");
            })
            .expect("compact");
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
        let storage: Arc<dyn StorageProvider> =
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
            .get_all_superfiles()
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
        let storage: Arc<dyn StorageProvider> =
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
            .get_all_superfiles()
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
            .get("title")
            .expect("merged entry should have title column stats");

        // Extract min and max string values from the arrays
        let title_min_arr = title_stats
            .min
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("title column should be LargeStringArray");
        let title_max_arr = title_stats
            .max
            .as_any()
            .downcast_ref::<LargeStringArray>()
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
        let storage: Arc<dyn StorageProvider> =
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
            .get_all_superfiles()
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
                .token_match("title", &[*term], BoolMode::And)
                .await
                .unwrap_or_else(|_| panic!("token_match for '{term}'"));
            assert_eq!(hits.len(), 2, "term '{term}' should match exactly 2 docs");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_respects_connection_memory_budget() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

        // Write the data with a normal budget first — ingest draws from the
        // same connection budget, so a tight limit here would starve the
        // setup appends too.
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["first doc", "second doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
        {
            let mut w = st.writer().expect("writer");
            let batch = build_title_batch(&["third doc", "fourth doc"]);
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }

        // Reopen the same committed data under a starved budget to exercise
        // the merge-time reservation.
        let mut opts = default_supertable_options().with_storage(Arc::clone(&storage));
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("reopen supertable");

        let reader = st.reader();
        let superfiles: Vec<Arc<SuperfileEntry>> = reader.manifest().get_all_superfiles().to_vec();

        match st.merge_superfiles(&superfiles).await {
            Err(BuildError::MemoryBudgetExceeded(_)) => {}
            Err(other) => panic!("expected MemoryBudgetExceeded, got {other:?}"),
            Ok(_) => panic!("merge must be refused over budget"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_single_superfile() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
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
            .get_all_superfiles()
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
            .token_match("title", &["only"], BoolMode::And)
            .await
            .expect("token_match for 'only'");
        assert_eq!(
            only_hits.len(),
            1,
            "should find exactly 1 doc matching 'only'"
        );

        let second_hits = merged_reader
            .token_match("title", &["second"], BoolMode::And)
            .await
            .expect("token_match for 'second'");
        assert_eq!(
            second_hits.len(),
            1,
            "should find exactly 1 doc matching 'second'"
        );
    }

    /// An in-memory supertable (no storage, no tombstone cache) takes
    /// the empty-sidecar-map fallback arm in `compact_async`: it still
    /// builds per-superfile stats and runs `select`, and with a single
    /// committed superfile `select` finds nothing to do, so the call
    /// returns `Ok(())` without touching storage.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_in_memory_table_takes_empty_sidecar_fallback() {
        let st =
            Supertable::create(default_supertable_options()).expect("create in-memory supertable");
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&["alpha first", "alpha second"]))
                .expect("append");
            w.commit().expect("commit");
        }
        let before = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("in-memory compact is a no-op, not an error");
        assert_eq!(
            st.manifest_id(),
            before,
            "single superfile yields no compaction job"
        );
    }

    // ─── Helpers shared by the end-to-end compact() tests ─────────────────

    fn make_st(dir: &TempDir) -> Supertable {
        let storage: Arc<dyn StorageProvider> =
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

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_rejects_concurrent_call_while_slot_held() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Manually set the slot as if a compaction is running.
        st.inner()
            .compaction_outstanding
            .store(true, Ordering::Release);

        let err = st
            .compact_async(&small_compact_cfg())
            .await
            .expect_err("must reject while slot held");

        assert!(
            matches!(err, CompactionError::AlreadyCompacting),
            "expected AlreadyCompacting, got {err:?}"
        );

        // Release so the supertable is clean for drop.
        st.inner()
            .compaction_outstanding
            .store(false, Ordering::Release);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_slot_released_after_completion() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("first compact");

        // Slot must be released so a second call succeeds.
        st.compact_async(&small_compact_cfg())
            .await
            .expect("second compact after slot release");
    }

    // OCC retry tests
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_succeeds_when_concurrent_writer_commits_during_compaction() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Enough superfiles to trigger a compaction job.
        for title in &[
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, title);
        }

        let before_docs = st.reader().n_docs_total();
        let st2 = st.clone();

        // Race a writer commit against compaction. The compactor will
        // hit WriteContentionExhausted on its first pointer CAS attempt
        // (or succeed before the writer — either way both must succeed).
        let writer_handle = task::spawn_blocking(move || {
            commit_titles(&st2, &["kilo first", "kilo second"]);
        });

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact must succeed despite concurrent writer");

        writer_handle.await.expect("writer task");

        // All docs from both paths must be visible after refresh.
        st.refresh().await.expect("refresh");
        let after_docs = st.reader().n_docs_total();
        assert_eq!(
            after_docs,
            before_docs + 2,
            "writer's 2 docs must survive alongside compacted data"
        );
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
                fts.may_contain(term),
                "bloom missing term '{}'",
                str::from_utf8(term).expect("term literal is valid utf-8")
            );
        }

        // Box::leak(dir);
        mem::forget(dir);
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
                fts.may_contain(term),
                "bloom missing term '{}'",
                str::from_utf8(term).expect("term literal is valid utf-8")
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

    /// The merged superfile from compaction must be warmed into the
    /// reader cache, and the merged-away inputs must be evicted from it.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_warms_merged_superfile_and_evicts_merged_away_ones_from_cache() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Combined size must clear small_compact_cfg()'s ~10KB floor,
        // or select() emits no job at all.
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

        let old_uris: Vec<_> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.uri)
            .collect();
        assert_eq!(old_uris.len(), 10);
        // Each commit already warmed the cache on its own.
        for uri in &old_uris {
            assert!(
                st.inner().options.store.reader(uri).is_ok(),
                "pre-merge superfile {uri:?} should already be warm from its own commit"
            );
        }

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let merged_uri = st.reader().manifest().superfiles[0].uri;
        assert!(
            st.inner().options.store.reader(&merged_uri).is_ok(),
            "merged superfile must be warmed into the in-memory cache right after compact"
        );
        for uri in &old_uris {
            assert!(
                st.inner().options.store.reader(uri).is_err(),
                "merged-away superfile {uri:?} must be evicted from the in-memory cache"
            );
        }
    }

    /// Same as the in-memory case, but for a disk-cache-attached table:
    /// the merged superfile should already be resident in the disk
    /// cache right after compact, with no cold fetch needed.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_warms_merged_superfile_into_disk_cache() {
        use crate::supertable::reader_cache::{DiskCacheConfig, DiskCacheStore, LruPolicy};

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let cache = DiskCacheStore::new_unpinned(
            Arc::clone(&storage),
            DiskCacheConfig {
                cache_root: dir.path().join("disk-cache"),
                mmap_cold_threshold_secs: 0,
                eviction: Box::new(LruPolicy::new()),
                ..Default::default()
            },
        )
        .expect("disk cache");
        let st = Supertable::create(
            default_supertable_options()
                .with_storage(Arc::clone(&storage))
                .with_disk_cache(Arc::clone(&cache)),
        )
        .expect("create supertable");

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

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let cold_fetches_after_compact = cache.stats().n_cold_fetches;

        // A query against the merged file must not trigger a cold
        // fetch -- it should already be resident from compaction's
        // own warm-up.
        let n: usize = st
            .token_match("title", "alpha", BoolMode::And, None)
            .expect("token_match")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n, 2);
        assert_eq!(
            cache.stats().n_cold_fetches,
            cold_fetches_after_compact,
            "querying the merged superfile should not cold-fetch -- it \
             should already be warm in the disk cache from compaction"
        );
    }

    /// Vocabulary for realistic term-frequency spread (no `rand` dep).
    const LATENCY_BENCH_WORDS: &[&str] = &[
        "system",
        "storage",
        "query",
        "index",
        "engine",
        "object",
        "table",
        "column",
        "vector",
        "search",
        "cluster",
        "replica",
        "cache",
        "buffer",
        "stream",
        "batch",
        "record",
        "field",
        "schema",
        "partition",
    ];

    fn env_usize(key: &str, default: usize) -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Builds one superfile's worth of rows. `shard_tag` is this
    /// superfile's unique narrow term; `broad_term` shows up in 1/3
    /// rows.
    fn latency_bench_shard_batch(
        shard_tag: &str,
        broad_term: &str,
        row_offset: usize,
        n_rows: usize,
    ) -> arrow_array::RecordBatch {
        let titles: Vec<String> = (0..n_rows)
            .map(|local_i| {
                let i = row_offset + local_i;
                // Cheap multiplicative hash, spreads word choice without a rand dep.
                let words: Vec<&str> = (0..5)
                    .map(|k| {
                        let h = (i as u64)
                            .wrapping_mul(2_654_435_761)
                            .wrapping_add(k as u64 * 40_503);
                        LATENCY_BENCH_WORDS[(h % LATENCY_BENCH_WORDS.len() as u64) as usize]
                    })
                    .collect();
                let common = if i.is_multiple_of(3) {
                    format!(" {broad_term}")
                } else {
                    String::new()
                };
                format!("{shard_tag}{common} {} row{i}", words.join(" "))
            })
            .collect();
        let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
        build_title_batch(&refs)
    }

    fn latency_bench_warm_median(
        st: &Supertable,
        query: &str,
        warmup_iters: usize,
        measured_iters: usize,
    ) -> u128 {
        for _ in 0..warmup_iters {
            st.bm25_search("title", query, 10, BoolMode::Or, None)
                .expect("bm25_search warmup");
        }
        let mut samples = Vec::with_capacity(measured_iters);
        for _ in 0..measured_iters {
            let start = Instant::now();
            st.bm25_search("title", query, 10, BoolMode::Or, None)
                .expect("bm25_search measured");
            samples.push(start.elapsed().as_micros());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }

    /// Exact match count (unlike `bm25_search`'s top-k), so it catches
    /// old pre-compact files leaking back into results.
    fn latency_bench_count_hits(st: &Supertable, query: &str) -> u64 {
        st.count("title", query, BoolMode::Or).expect("count")
    }

    /// Warm `bm25_search` latency after merging many small superfiles
    /// into one, on a real local-filesystem corpus (no cloud needed).
    /// Scale via env vars: `INFINO_COMPACT_BENCH_TOTAL_MB` (default 500),
    /// `INFINO_COMPACT_BENCH_N_SUPERFILES` (default 40),
    /// `INFINO_COMPACT_BENCH_TARGET_MB` (default = total).
    #[ignore = "perf diagnostic for issue #372/#378; run with --ignored --nocapture"]
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_latency_at_scale() {
        const APPROX_BYTES_PER_DOC: u64 = 90;
        const BROAD_TERM: &str = "broadterm";
        const WARMUP_ITERS: usize = 20;
        const MEASURED_ITERS: usize = 50;

        let total_mb = env_usize("INFINO_COMPACT_BENCH_TOTAL_MB", 500);
        let n_superfiles = env_usize("INFINO_COMPACT_BENCH_N_SUPERFILES", 40);
        let compact_target_mb = env_usize("INFINO_COMPACT_BENCH_TARGET_MB", total_mb.max(1)) as u64;

        let total_docs = (total_mb as u64 * 1_000_000) / APPROX_BYTES_PER_DOC;
        let docs_per_superfile = (total_docs as usize / n_superfiles).max(1);

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("local fs provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        let narrow_term = format!("shard{}", n_superfiles / 2);

        for i in 0..n_superfiles {
            let shard_tag = format!("shard{i}");
            let mut w = st.writer().expect("writer");
            w.append(&latency_bench_shard_batch(
                &shard_tag,
                BROAD_TERM,
                i * docs_per_superfile,
                docs_per_superfile,
            ))
            .expect("append");
            w.commit().expect("commit");
        }

        let n_before = st.reader().n_superfiles();
        let docs_before = st.reader().n_docs_total();
        let narrow_hits_before = latency_bench_count_hits(&st, &narrow_term);
        let broad_hits_before = latency_bench_count_hits(&st, BROAD_TERM);
        let narrow_before =
            latency_bench_warm_median(&st, &narrow_term, WARMUP_ITERS, MEASURED_ITERS);
        let broad_before = latency_bench_warm_median(&st, BROAD_TERM, WARMUP_ITERS, MEASURED_ITERS);

        st.compact_async(&CompactionSettings {
            target_superfile_size_mb: compact_target_mb,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        })
        .await
        .expect("compact");

        let n_after = st.reader().n_superfiles();
        assert!(n_after < n_before, "compact should reduce superfile count");

        // No old-file double-counting: doc/hit counts must be identical.
        assert_eq!(st.reader().n_docs_total(), docs_before);
        assert_eq!(
            latency_bench_count_hits(&st, &narrow_term),
            narrow_hits_before
        );
        assert_eq!(latency_bench_count_hits(&st, BROAD_TERM), broad_hits_before);

        let narrow_after =
            latency_bench_warm_median(&st, &narrow_term, WARMUP_ITERS, MEASURED_ITERS);
        let broad_after = latency_bench_warm_median(&st, BROAD_TERM, WARMUP_ITERS, MEASURED_ITERS);

        eprintln!(
            "superfiles: {n_before} -> {n_after}, narrow: {narrow_before}us -> {narrow_after}us, \
             broad: {broad_before}us -> {broad_after}us"
        );

        // Narrow only ever touches one relevant superfile (bloom-skips
        // the rest either way), so it must stay flat regardless of
        // merge count.
        assert!(
            narrow_after <= narrow_before * 2,
            "narrow query regressed: {narrow_before}us -> {narrow_after}us"
        );

        mem::forget(dir);
    }

    /// compact() drops the manifest's superfile count right away, but
    /// the merged-away files stay on disk until a gc() sweep past the
    /// safety gap deletes them.
    #[tokio::test(flavor = "multi_thread")]
    async fn compact_reduces_manifest_count_but_gc_safety_gap_leaves_old_files_on_disk() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");

        for titles in [
            ["alpha first", "alpha second"],
            ["bravo first", "bravo second"],
            ["charlie first", "charlie second"],
            ["delta first", "delta second"],
            ["echo first", "echo second"],
            ["foxtrot first", "foxtrot second"],
            ["golf first", "golf second"],
            ["hotel first", "hotel second"],
            ["india first", "india second"],
            ["juliet first", "juliet second"],
        ] {
            commit_titles(&st, &titles);
        }

        let before_n_superfiles = st.reader().n_superfiles();
        let before_data_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ before compact")
            .len();
        assert_eq!(before_data_objects, before_n_superfiles);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let after_n_superfiles = st.reader().n_superfiles();
        assert!(
            after_n_superfiles < before_n_superfiles,
            "manifest superfile count must drop right after compact"
        );

        // Old inputs are orphaned, not deleted, until gc() runs.
        let after_data_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after compact")
            .len();
        assert_eq!(after_data_objects, before_data_objects + 1);

        // Default 1-day safety gap: everything here is brand new, so
        // gc() deletes nothing yet.
        let default_gap_report = st
            .gc(crate::config::DEFAULT_GC_SAFETY_GAP)
            .expect("gc with default safety gap");
        assert_eq!(default_gap_report.objects_deleted, 0);
        let after_default_gc_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after default-gap gc")
            .len();
        assert_eq!(after_default_gc_objects, before_data_objects + 1);

        // A shrunk safety gap reclaims the orphaned inputs, and disk
        // count catches up with the manifest.
        let zero_gap_report = st
            .gc(std::time::Duration::ZERO)
            .expect("gc with zero safety gap");
        assert!(
            zero_gap_report.objects_deleted > 0,
            "a gc() past the safety gap must reclaim the orphaned pre-merge inputs"
        );
        let after_zero_gap_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ after zero-gap gc")
            .len();
        assert_eq!(after_zero_gap_objects, after_n_superfiles);

        mem::forget(dir);
    }

    /// A superfile sealed by an abandoned compaction attempt (a merge
    /// that started but never finished) is never unsealed, so
    /// `pack_partition`'s `!sealed_by_other` filter excludes it from
    /// every future compaction pass, forever.
    #[tokio::test(flavor = "multi_thread")]
    async fn superfiles_sealed_by_an_abandoned_compaction_are_stranded_forever() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        commit_titles(&st, &["alpha first", "alpha second"]);
        commit_titles(&st, &["bravo first", "bravo second"]);

        let stranded_ids: Vec<Uuid> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        assert_eq!(stranded_ids.len(), 2);

        // Simulate a compaction that sealed its inputs then died
        // before committing the merge.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let wal_store = WalStore::new(storage);
        let abandoned_compaction_id = Uuid::new_v4();
        let sealed_at = Utc::now();
        for id in &stranded_ids {
            tombstones_admin::seal(
                &wal_store,
                *id,
                abandoned_compaction_id,
                sealed_at,
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("seal");
        }

        // New data arrives and a generous compaction config runs.
        commit_titles(&st, &["charlie first", "charlie second"]);
        commit_titles(&st, &["delta first", "delta second"]);

        let cfg = CompactionSettings {
            target_superfile_size_mb: 1024,
            min_fill_percent: 1,
            ..CompactionSettings::default()
        };
        st.compact_async(&cfg)
            .await
            .expect("compact must not error");

        // The two stranded superfiles are still sitting untouched —
        // they can never be merged, so they leak permanently.
        let remaining_ids: HashSet<Uuid> = st
            .reader()
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        for id in &stranded_ids {
            assert!(remaining_ids.contains(id));
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
