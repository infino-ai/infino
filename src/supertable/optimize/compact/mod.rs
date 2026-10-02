// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Picks which superfiles to merge.
//!
//! no I/O. `supertable::compact` gathers the
//! stats, calls [`select`], then merges each [`CompactionJob`].
//! Compaction is single-level — a target-sized superfile is never
//! re-compacted.

use std::{
    collections::{BTreeSet, HashMap, HashSet, VecDeque},
    io::{BufWriter, Write},
    mem,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use bytes::Bytes;
use chrono::{DateTime, Utc};
use futures::{
    future::BoxFuture,
    stream::{self, StreamExt},
};
use roaring::RoaringBitmap;
use tempfile::NamedTempFile;
use tokio::{
    sync::Semaphore,
    task::{JoinError, JoinSet},
    time,
};
#[cfg(not(feature = "detailed-tracing"))]
use tracing::Span;
#[cfg(feature = "detailed-tracing")]
use tracing::info_span;
use tracing::{Instrument, info, warn};
use uuid::Uuid;

use crate::{
    config::{CompactionSettings, RecalibratePolicy, global},
    runtime_bridge::{bridge_on_runtime, run_on_pool},
    runtime_metrics::rss::memory_budget,
    superfile::{
        builder::SuperfileBuilder,
        fts::reader::ColumnLengthStats,
        reader::SuperfileReader,
        stats::SuperfileStats as BuiltSuperfileStats,
        vector::{cell_posting::transcode_clamped_components, layout::VectorLayout},
    },
    supertable::{
        BuildError, CommitError, ManifestSnapshot, SuperfileEntry, Supertable,
        error::CompactionError,
        handle::hidden_vector_index_compaction_settings,
        manifest::{
            SuperfileUri, list::PartitionStrategy, listed_once,
            term_index::Contribution as TermContribution,
        },
        opann::rerank_pool_hint,
        query::dispatch::open_compaction_input,
        reader_cache::disk::mmap_readonly_bytes,
        wal::{
            Etag, SealRecord, TombstonesSidecar, WalStore,
            tombstones_admin::{self, TombstonesAdminError},
        },
        writer::{
            CommitFence, NewEntryBirthVersions, PreparedSuperfile, ShardOutput, backoff_delay,
            finalize_compaction_commit, maint_pool, prepare_superfile_named,
            recalibrate_probe_laws, refresh_slow_vector_state, split_overflow_cells,
            try_commit_attempt, write_superfile_list,
        },
    },
    utils::trace::{detail_span, record},
};

/// Held for as long as one process is reshaping superfiles, and released
/// on drop. Compaction and reindex share it: both rewrite superfiles and
/// commit manifest swaps, so running them together would put two planners
/// on the same files.
pub(crate) struct CompactionSlot<'a>(&'a AtomicBool);

impl Drop for CompactionSlot<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Supertable {
    /// Take the reshape slot, or report that something else holds it.
    ///
    /// Process-local by design: across processes it is the per-superfile
    /// tombstone-sidecar seal that serializes writers, and that guard does
    /// not care which kind of job took it.
    pub(crate) fn try_hold_compaction_slot(&self) -> Option<CompactionSlot<'_>> {
        let outstanding = &self.inner().compaction_outstanding;
        match outstanding.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed) {
            Ok(_) => Some(CompactionSlot(outstanding)),
            Err(_) => None,
        }
    }
}

/// What running a job actually did.
///
/// A job whose inputs have already been replaced by another writer is not
/// a failure — there is simply nothing left to do — but it is also not a
/// rewrite, and a caller counting its progress must not count it as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JobOutcome {
    /// The replacement was committed.
    Committed,
    /// The inputs were gone by the time this job reached the manifest, so
    /// another writer had already handled them.
    InputsAlreadyReplaced,
}

/// How a job turns its input superfiles into the one it commits.
///
/// [`Supertable::run_compaction_job`] owns the risky half of a rewrite —
/// sealing each input's tombstone sidecar, committing under OCC with
/// retries, unsealing on the way out — and that is worth exactly one
/// implementation. What a job *builds* is not: compaction merges files
/// together, and a format migration rebuilds one in place. Injecting the
/// build keeps the second from having to be known here.
pub(crate) trait SuperfileMerge: Send + Sync {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<BuiltSuperfileStats, BuildError>;

    /// Whether the build carries every input row, tombstoned ones included.
    ///
    /// True means the inputs' bitmaps still describe the output, so the
    /// runner must carry them onto it; false means the rows renumbered.
    fn preserves_tombstones(&self) -> bool;
}

/// The opened inputs a [`SuperfileMerge`] builds from.
pub(crate) struct MergeInputs<'a> {
    /// Each input reader with the tombstones that apply to it.
    pub(crate) readers: &'a [(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    /// The manifest entries those readers were opened from, in the same
    /// order. A build that carries rows unchanged takes its output stats
    /// from here rather than recomputing them from decoded rows.
    pub(crate) entries: &'a [Arc<SuperfileEntry>],
    /// Per reader, the hidden-index cells its rows have been superseded in.
    pub(crate) superseded: &'a [BTreeSet<u32>],
    /// Table-wide document-length totals excluding the inputs, so the
    /// output bakes the average an unfragmented table would have.
    pub(crate) fts_corpus: &'a HashMap<String, ColumnLengthStats>,
}

/// What compaction does: splice or carry, never re-tokenize.
///
/// Each arm is chosen by what the inputs hold, and every one of them
/// carries the inputs' posting lists across rather than rebuilding them —
/// re-tokenizing a corpus to merge it costs far more and changes nothing.
pub(crate) struct CompactionMerge;

impl SuperfileMerge for CompactionMerge {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<BuiltSuperfileStats, BuildError> {
        let MergeInputs {
            readers,
            superseded,
            fts_corpus,
            ..
        } = inputs;
        let first_vec = readers.first().and_then(|(reader, _)| reader.vec());
        let multi_cell = first_vec.is_some_and(|v| v.is_multi_cell());
        let sq8_merge = first_vec.and_then(|v| {
            v.vector_columns_config()
                .next()
                .map(|c| c.rerank_codec.is_ivf_mergeable())
        });
        let stats = if multi_cell && sq8_merge == Some(true) {
            SuperfileBuilder::build_from_multi_cell_sq8_ivf_readers_to(
                readers, superseded, fts_corpus, output,
            )?
        } else if sq8_merge == Some(true) {
            SuperfileBuilder::build_from_sq8_ivf_readers_to(readers, fts_corpus, output)?
        } else if first_vec.is_none() {
            // FTS/scalar inputs (no vector index): carry each input's
            // already-built posting lists across instead of re-tokenizing
            // the whole corpus.
            SuperfileBuilder::build_from_readers_fts_merge_to(readers, fts_corpus, output)?
        } else {
            // A vector index is present but not IVF-mergeable (e.g. an fp32
            // rerank codec); this path re-encodes both the FTS and the
            // vectors from the decoded rows.
            SuperfileBuilder::build_from_readers_to(readers, fts_corpus, output)?
        };
        Ok(stats)
    }

    fn preserves_tombstones(&self) -> bool {
        false
    }
}

pub(crate) mod plan;

use plan::split_stats_at_drain_watermark;
pub(crate) use plan::{CompactionJob, SuperfileStats, select};

/// Cap on compaction input opens in flight, counted across the whole process
/// rather than per merge. Several merges run at once, and it is their combined
/// fan-out that saturates the object-store connection pool and pushes requests
/// into timeouts, so the budget they share is the one worth bounding.
const MAX_CONCURRENT_INPUT_OPENS: usize = 64;

/// The shared permit pool behind [`MAX_CONCURRENT_INPUT_OPENS`].
static INPUT_OPEN_PERMITS: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn input_open_permits() -> &'static Arc<Semaphore> {
    INPUT_OPEN_PERMITS.get_or_init(|| Arc::new(Semaphore::new(MAX_CONCURRENT_INPUT_OPENS)))
}

impl Supertable {
    /// Compaction entry point.
    /// Gathers per-superfile stats from the current manifest snapshot,
    /// selects compaction jobs, then for each job seals every input
    /// superfile's tombstone sidecar so no concurrent deletes can land
    /// during the merge window.
    /// Compaction with the historical `Auto` recalibration behavior. Used by
    /// tests; production goes through [`compact_with`] from the optimize entry
    /// point so the caller's [`RecalibratePolicy`] is honored.
    #[cfg(test)]
    pub(crate) fn compact(&self, cfg: &CompactionSettings) -> Result<(), CompactionError> {
        self.compact_with(cfg, RecalibratePolicy::Auto)
    }

    /// Like [`compact`], but with an explicit recalibration policy. `compact`
    /// keeps the historical `Auto` behavior for its many call sites; the
    /// optimize entry point threads the caller's `OptimizeOptions.recalibrate`
    /// through here so a repeated-optimize ingest loop can skip the O(N)
    /// recalibration.
    pub(crate) fn compact_with(
        &self,
        cfg: &CompactionSettings,
        recalibrate: RecalibratePolicy,
    ) -> Result<(), CompactionError> {
        bridge_on_runtime(
            self.compact_async_with(cfg, recalibrate),
            &self.inner().query_runtime(),
        )
    }

    /// Async compaction with the historical `Auto` recalibration behavior.
    /// Used by tests; production goes through [`compact_async_with`].
    #[cfg(test)]
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(name = "compact", skip_all, fields(role = self.role().as_str()))
    )]
    pub(crate) async fn compact_async(
        &self,
        cfg: &CompactionSettings,
    ) -> Result<(), CompactionError> {
        self.compact_async_with(cfg, RecalibratePolicy::Auto).await
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(name = "compact", skip_all, fields(role = self.role().as_str()))
    )]
    pub(crate) async fn compact_async_with(
        &self,
        cfg: &CompactionSettings,
        recalibrate: RecalibratePolicy,
    ) -> Result<(), CompactionError> {
        let phase_timers = global().diagnostics.optimize_phase_timers;
        Self::compact_one_table(self, cfg, recalibrate).await?;
        if matches!(
            self.inner().manifest.load().get_partition_strategy(),
            PartitionStrategy::VectorCell { .. }
        ) {
            let __st = Instant::now();
            refresh_slow_vector_state(
                self.inner(),
                !matches!(recalibrate, RecalibratePolicy::Skip),
            )
            .await
            .map_err(|error| CompactionError::Refresh(error.to_string()))?;
            if phase_timers {
                info!(secs = __st.elapsed().as_secs_f64(), "[optphase]   settle");
            }
        } else if let Some(hidden) = self.inner().vector_index_table.as_ref() {
            Self::compact_one_table(
                hidden,
                &hidden_vector_index_compaction_settings(cfg.max_concurrent_jobs),
                recalibrate,
            )
            .await?;
            // The hidden pass settled vector membership (merges + finalize +
            // any cell splits); its `update`s cleared the slow-CAS ref, so
            // republish the entry blob and restamp. Hidden tables have no
            // manifest parts, so publication is required for reopen and a
            // failure must be visible to the caller.
            let __st = Instant::now();
            refresh_slow_vector_state(
                hidden.inner(),
                !matches!(recalibrate, RecalibratePolicy::Skip),
            )
            .await
            .map_err(|error| CompactionError::Refresh(error.to_string()))?;
            if phase_timers {
                info!(secs = __st.elapsed().as_secs_f64(), "[optphase]   settle");
            }
        }
        Ok(())
    }

    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(name = "compact_table", skip_all, fields(role = table.role().as_str()))
    )]
    pub(crate) async fn compact_one_table(
        table: &Supertable,
        cfg: &CompactionSettings,
        recalibrate: RecalibratePolicy,
    ) -> Result<(), CompactionError> {
        let inner = table.inner();

        let _slot = table
            .try_hold_compaction_slot()
            .ok_or(CompactionError::AlreadyCompacting)?;
        // #512 invariant tripwire, mirroring the drain's: merges and splits
        // transcode Sq8 rows between per-cluster quantizers, and a
        // destination grid that fails to cover its inputs saturates
        // components silently. Snapshot the process tally; shout on exit if
        // this pass added any.
        let transcode_clamp_baseline = transcode_clamped_components();

        // Phase 1 (split-then-merge): split every over-cap cell first, from the
        // live grid, before merge-job selection. An over-cap cell is thus never
        // merged just to be re-split (the merge output would be discarded), and
        // the split runs as its own snapshot-consistent phase, so it can't remove
        // a superfile a later merge job in this pass planned to use.
        //
        // Keyed on the manifest's LOCKED strategy, not the handle options: a
        // hidden handle built at table create time has no user manifest to
        // train a grid from, so its options never carry VectorCell — only the
        // first drain locks the strategy into the manifest. An options-keyed
        // gate silently skips every split until the table is reopened.
        // `split_overflow_cells` re-checks the manifest strategy itself, so
        // user tables (never VectorCell-locked) cannot reach the split. The
        // recalibration trigger below shares the same signal.
        let hidden_ivf = matches!(
            inner.manifest.load().partition_strategy(),
            Some(PartitionStrategy::VectorCell { .. })
        );
        // Superfile-id snapshot for the recalibration trigger below: splits
        // and merges both change the id set, and both invalidate a stamped
        // probe law (splits change the cell geometry, merges rebuild the
        // merged cells' fine IVFs).
        let snapshot_ids = || -> HashSet<Uuid> {
            inner
                .manifest
                .load()
                .superfiles
                .iter()
                .map(|e| e.superfile_id)
                .collect()
        };
        let pre_pass_ids = if hidden_ivf {
            snapshot_ids()
        } else {
            HashSet::new()
        };
        // Optimize phase timers ([optphase]); gated, off by default.
        let phase_timers = global().diagnostics.optimize_phase_timers;
        // How many merges this pass keeps in flight. Resolved once: the jobs
        // are planned from one snapshot, so the width must not drift mid-pass.
        let concurrency = global().compaction_concurrency(cfg);
        let mut __pt = Instant::now();
        if hidden_ivf {
            split_overflow_cells(Arc::clone(inner))
                .await
                .map_err(|e| CompactionError::Build(e.to_string()))?;
        }
        if phase_timers {
            info!(secs = __pt.elapsed().as_secs_f64(), "[optphase]   split");
        }

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

        // Build SuperfileStats for every superfile in the snapshot, once per
        // id. Deduped here, not in `select`: the drain-watermark split below
        // could otherwise put two copies of one superfile in different jobs.
        let now = Utc::now();
        let stale_seal_timeout = Duration::from_millis(cfg.stale_seal_timeout_ms);
        let listed = manifest.get_all_superfiles();
        let stats: Vec<SuperfileStats> = listed_once(listed, |entry| entry.superfile_id)
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
                    birth_version: entry.birth_version,
                }
            })
            .collect();

        if stats.len() < listed.len() {
            warn!(
                role = table.role().as_str(),
                repeats = listed.len() - stats.len(),
                "manifest lists some superfiles more than once; compacting each once"
            );
        }

        // A user table with a hidden vector index selects jobs per side of
        // the drain watermark, never across it (see
        // [`split_stats_at_drain_watermark`] for why a mixed merge loses
        // vectors). Tables without a hidden sibling select over everything.
        let stat_groups: Vec<Vec<SuperfileStats>> = match inner.vector_index_table.as_ref() {
            Some(hidden) => {
                let drained = hidden.inner().manifest.load_full().get_drained_ranges();
                let (drained_stats, undrained_stats) =
                    split_stats_at_drain_watermark(stats, &drained);
                vec![drained_stats, undrained_stats]
            }
            None => vec![stats],
        };
        for stats in &stat_groups {
            let jobs = select(stats, cfg);
            info!(
                role = table.role().as_str(),
                jobs = jobs.len(),
                concurrency,
                "compaction jobs planned"
            );
            if phase_timers {
                __pt = Instant::now();
            }
            table
                .run_compaction_jobs(jobs, stale_seal_timeout, concurrency)
                .await?;
            if phase_timers {
                info!(secs = __pt.elapsed().as_secs_f64(), "[optphase]   merge");
            }
        }

        // The pass reshaped the hidden index (split children and/or merge
        // outputs committed): the probe laws were measured against the old
        // geometry, so re-measure and restamp both (width + fine depth)
        // while the compaction slot still serializes hidden reorgs.
        // Repair trigger, independent of reshapes: a width law whose
        // rerank points sit CLEARED (the stamped width outgrew the pool
        // that measured them) never self-heals on a table that doesn't
        // split or merge — the load -> optimize flow would otherwise
        // leave the default path on the constant budget forever.
        let rerank_lags = || match inner.manifest.load().partition_strategy() {
            Some(PartitionStrategy::VectorCell {
                routing, clusters, ..
            }) => {
                let achievable =
                    rerank_pool_hint(&routing.width_for_k, clusters.n_cent as usize) as u32;
                routing.rerank_law_lags_pool(achievable)
            }
            _ => false,
        };
        // Recalibration is the O(N) query-serving calibration — gate it by the
        // caller's policy (default Auto). Skip runs no recalibration; Force always
        // runs it; Auto keeps the changed-or-lagging condition. Storage work above
        // already ran regardless of the policy.
        let run_recalibrate = hidden_ivf
            && match recalibrate {
                RecalibratePolicy::Skip => false,
                RecalibratePolicy::Force => true,
                RecalibratePolicy::Auto => snapshot_ids() != pre_pass_ids || rerank_lags(),
            };
        if run_recalibrate {
            if phase_timers {
                __pt = Instant::now();
            }
            recalibrate_probe_laws(inner)
                .await
                .map_err(|e| CompactionError::Build(e.to_string()))?;
            if phase_timers {
                info!(
                    secs = __pt.elapsed().as_secs_f64(),
                    "[optphase]   recalibrate"
                );
            }
        }

        let clamped_components = transcode_clamped_components() - transcode_clamp_baseline;
        if clamped_components > 0 {
            warn!(
                "[supertable compaction] BUG: {clamped_components} component(s) saturated \
                 their destination quantizer during this pass's merges/splits — a \
                 destination grid failed to cover its inputs; affected rows' recall \
                 degrades.",
            );
        }
        Ok(())
    }

    /// Merges the given superfiles into one, dropping the rows `tombstones`
    /// marks dead.
    ///
    /// `tombstones` maps each input to the bitmap its sidecar held when the
    /// caller sealed it. The caller has to supply them because only it knows
    /// its reads are current: it GETs each sidecar under the seal, so no bit
    /// can land behind that read. Reading them here instead, through the
    /// sidecar cache, would be a read against whatever seq view this handle
    /// last installed, and a handle that has not seen a delete's commit holds
    /// a view that predates it — the merge would then carry a deleted row into
    /// the output and the commit would remove the input that held its bit.
    ///
    /// An input absent from the map is treated as having no tombstones, which
    /// is correct for a superfile with no sidecar and a caller bug otherwise.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(name = "merge_superfiles", skip_all, fields(inputs = superfiles.len()))
    )]
    /// Merge with the build compaction uses. Test-only: production reaches
    /// the same build through `run_compaction_job`, which names the
    /// strategy so a caller cannot get one it did not choose.
    #[cfg(test)]
    pub(crate) async fn merge_superfiles(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        tombstones: &HashMap<Uuid, Arc<RoaringBitmap>>,
    ) -> Result<PreparedSuperfile, BuildError> {
        self.merge_superfiles_with(superfiles, tombstones, Arc::new(CompactionMerge))
            .await
    }

    /// As [`Self::merge_superfiles`], with the build injected.
    ///
    /// Everything around the build — opening inputs, applying tombstones,
    /// the table-wide length totals, streaming to a temp file and mapping
    /// it back — is the same work whatever produces the bytes, and is not
    /// worth a second copy. Only the build differs.
    pub(crate) async fn merge_superfiles_with(
        &self,
        superfiles: &[Arc<SuperfileEntry>],
        tombstones: &HashMap<Uuid, Arc<RoaringBitmap>>,
        merge: Arc<dyn SuperfileMerge>,
    ) -> Result<PreparedSuperfile, BuildError> {
        debug_assert!(
            tombstones.is_empty()
                || superfiles
                    .iter()
                    .all(|e| tombstones.contains_key(&e.superfile_id)),
            "every input needs its sealed tombstone bitmap, or its rows come back"
        );
        let manifest = { self.inner().manifest.load().clone() };
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();

        // This reserves budget for the whole input size since merge still
        // loads it all at once. Real fix is streaming the merge and pooling
        // buffers instead of a flat reservation; picking that up later.
        // Deliberately not reserved from `connection_memory_budget`: that budget
        // bounds what search and ingest may hold on behalf of a connection, and
        // a merge is neither. Charging it here made the optimizer's footprint a
        // function of a knob set for queries — and with several merges running,
        // a pass could be refused for a budget a serial pass fit inside, on a
        // machine with the memory to spare. What bounds a merge is the host:
        // the runner admits one at a time against free memory, and
        // `max_memory_mb` caps the input bytes a single job may pack.

        let semaphore = input_open_permits();
        let mut superfile_readers_tasks = JoinSet::new();
        for (idx, entry) in superfiles.iter().enumerate() {
            #[cfg(feature = "detailed-tracing")]
            let span = info_span!("compaction_input", superfile_id = %entry.superfile_id);
            #[cfg(not(feature = "detailed-tracing"))]
            let span = Span::none();
            let store = store.clone();
            let disk_cache = disk_cache.clone();
            let storage = storage.clone();
            let entry = entry.clone();
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .expect("should not be closed");
            let open_fut = async move {
                let _permit = permit;
                let r = open_compaction_input(
                    &store,
                    disk_cache.as_ref(),
                    storage.as_ref(),
                    entry.as_ref(),
                )
                .await;
                (idx, entry.superfile_id, r)
            }
            .instrument(span);

            superfile_readers_tasks.spawn(open_fut);
        }
        let mut readers = superfile_readers_tasks.join_all().await;
        readers.sort_unstable_by_key(|(idx, ..)| *idx);

        // A build that carries the row set never consults a bitmap at all.
        let carries_rows = merge.preserves_tombstones();

        let superseded_map = manifest.get_superseded_cells();
        let mut readers_with_tombstones = Vec::with_capacity(readers.len());
        let mut superseded_per_reader = Vec::with_capacity(readers.len());
        for (_idx, superfile_id, reader) in readers {
            let bitmap = match carries_rows {
                true => None,
                false => tombstones.get(&superfile_id).map(Arc::clone),
            };

            let reader = reader.map_err(|e| BuildError::Store(e.to_string()))?;
            let superseded = superseded_map
                .and_then(|m| m.get(&superfile_id))
                .cloned()
                .unwrap_or_default();
            superseded_per_reader.push(superseded);
            readers_with_tombstones.push((reader.clone(), bitmap));
        }

        // The merged file replaces its inputs, so it bakes the table-wide
        // average document length over everything else in the table plus
        // its own documents — the same statistic a fresh append bakes, and
        // what lets a compacted table score like an unfragmented one.
        let replaced: HashSet<Uuid> = superfiles.iter().map(|e| e.superfile_id).collect();
        let fts_corpus = manifest.fts_corpus_stats(&replaced);
        // The build is long, synchronous CPU work, so it runs on the
        // maintenance pool rather than the thread driving this future.
        // `run_on_pool` needs a `'static` closure, so everything it reads —
        // the merge included — is owned before it crosses over.
        let entries: Vec<Arc<SuperfileEntry>> = superfiles.to_vec();
        let merge = Arc::clone(&merge);
        let (merged_bytes, superfile_stats) = run_on_pool(
            Some(maint_pool()?),
            "compaction merge",
            move || -> Result<(Bytes, _), BuildError> {
                // Every merge kind streams its output to a temp file and mmaps it
                // back, so the corpus-sized merge output is never held as an anon
                // Vec — the allocation that OOMs compaction on a memory-tight host.
                // Mapped pages are file-backed and reclaimable; downstream publish
                // takes `Bytes` unchanged (large superfiles already stream via
                // put_multipart).
                let mut output = NamedTempFile::new()
                    .map_err(|e| BuildError::Store(format!("merge temp create: {e}")))?;
                let stats = {
                    let mut writer = BufWriter::new(output.as_file_mut());
                    let stats = merge.build(
                        MergeInputs {
                            readers: &readers_with_tombstones,
                            entries: &entries,
                            superseded: &superseded_per_reader,
                            fts_corpus: &fts_corpus,
                        },
                        &mut writer,
                    )?;
                    writer
                        .flush()
                        .map_err(|e| BuildError::Store(format!("merge temp flush: {e}")))?;
                    stats
                };
                let bytes = mmap_readonly_bytes(output.path())
                    .map_err(|e| BuildError::Store(format!("merge mmap: {e}")))?;
                Ok((bytes, stats))
            },
        )
        .await
        .map_err(|e| BuildError::Store(e.to_string()))??;

        let shard = ShardOutput::new_with_params(
            merged_bytes,
            superfile_stats.n_docs,
            superfile_stats.id_min,
            superfile_stats.id_max,
            superfile_stats.scalar_stats,
        );

        // A merge keeps a source stem only when every input carries the same
        // one. Inputs from different sources, or any unnamed input, produce
        // an unnamed superfile rather than a label that names one source
        // for rows from several.
        let stem = superfiles
            .first()
            .and_then(|first| first.stem.as_deref())
            .filter(|stem| superfiles.iter().all(|e| e.stem.as_deref() == Some(*stem)));
        let prepared_superfile = {
            let _span = detail_span!("prepare_merged_superfile").entered();
            prepare_superfile_named(self.inner().as_ref(), shard, stem)?
        };

        prepared_superfile.ok_or(BuildError::NoDocsToBuild)
    }

    /// Seal, merge, and stage one job for commit. Everything up to the
    /// manifest CAS: the caller decides whether to commit this job alone or
    /// batched with its siblings.
    ///
    /// On any failure here the job's own seals are cleared before returning,
    /// so a failed prepare leaves nothing behind for a sibling to trip over.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            name = "prepare_compaction_job",
            skip_all,
            fields(
                role = self.role().as_str(),
                inputs = job.inputs.len(),
                partition_key = ?job.partition_key,
                estimated_output_bytes = job.estimated_output_bytes,
            )
        )
    )]
    pub(crate) async fn prepare_compaction_job(
        &self,
        job: CompactionJob,
        stale_seal_timeout: Duration,
    ) -> Result<PreparedJob, CompactionError> {
        self.prepare_compaction_job_with(job, stale_seal_timeout, Arc::new(CompactionMerge))
            .await
    }

    /// As [`Self::prepare_compaction_job`], with the build injected — the
    /// seal, merge and stage cycle is identical, only the bytes differ.
    pub(crate) async fn prepare_compaction_job_with(
        &self,
        job: CompactionJob,
        stale_seal_timeout: Duration,
        merge: Arc<dyn SuperfileMerge>,
    ) -> Result<PreparedJob, CompactionError> {
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

        // The bitmaps `seal` GETs are the authoritative ones: read from
        // storage, under the seal, so nothing can land behind them.
        let sealed_tombstones: HashMap<Uuid, Arc<RoaringBitmap>> = sealed
            .iter()
            .map(|s| (s.superfile_id, Arc::new(s.bitmap.clone())))
            .collect();
        let merged_segment = match self
            .merge_superfiles_with(&inputs, &sealed_tombstones, Arc::clone(&merge))
            .await
        {
            Ok(seg) => Some(seg),
            // Every input was fully dead — all cells tombstoned, or all
            // superseded by an in-place cell split. There is nothing live to
            // write, so commit the inputs' removal with no replacement entry:
            // a pure reclaim of the dead superfiles.
            Err(BuildError::NoDocsToBuild) => None,
            Err(e) => {
                unseal_all(&wal_store, sealed).await;
                return Err(CompactionError::Build(e.to_string()));
            }
        };

        let (
            new_entries,
            pending_storage_writes,
            bytes_for_store,
            bytes_for_cache,
            merged_superfile_id,
            term_contributions,
        ) = match merged_segment {
            Some(PreparedSuperfile {
                entry: merged_prepared,
                bytes_for_store,
                bytes_for_storage,
                bytes_for_cache,
                term_contribution,
            }) => {
                let merged_entry = Arc::new(SuperfileEntry {
                    // Carry the OLDEST input's birth_version so a merge of
                    // already-drained inputs stays <= the drain watermark
                    // (skipped, not re-drained). See the hidden-index
                    // `drained_ranges` design.
                    birth_version: inputs.iter().map(|e| e.birth_version).min().unwrap_or(0),
                    // Left empty: the manifest's `update()` stamps the
                    // partition key at commit time from `partition_hint`.
                    partition_key: Vec::new(),
                    partition_hint: inputs.first().and_then(|e| e.partition_hint),
                    vector_layout: inputs
                        .first()
                        .map(|e| e.vector_layout)
                        .unwrap_or(VectorLayout::Ivf),
                    ..(*merged_prepared).clone()
                });
                let id = merged_entry.superfile_id;
                let storage_write = match bytes_for_storage {
                    Some(w) => w,
                    None => {
                        unseal_all(&wal_store, sealed).await;
                        return Err(CompactionError::EmptyMergedSuperfile);
                    }
                };
                (
                    vec![merged_entry],
                    vec![storage_write],
                    bytes_for_store,
                    bytes_for_cache,
                    id,
                    term_contribution.into_iter().collect::<Vec<_>>(),
                )
            }
            // Pure reclaim: remove the dead inputs, add no replacement.
            None => (Vec::new(), Vec::new(), None, None, Uuid::nil(), Vec::new()),
        };

        // Before the manifest swap: an orphan sidecar is recoverable, a live
        // output with no tombstones is not. One-in-one-out builds only, so a
        // batch never carries more than one.
        let carried_sidecar = match carry_tombstones_to_output(
            merge.as_ref(),
            &wal_store,
            &inputs,
            &new_entries,
            &sealed,
        )
        .await
        {
            Ok(id) => id,
            Err(e) => {
                unseal_all(&wal_store, sealed).await;
                return Err(e);
            }
        };

        Ok(PreparedJob {
            input_ids: job.inputs,
            compaction_id,
            sealed_at,
            carried_sidecar,
            sealed,
            new_entries,
            pending_storage_writes,
            bytes_for_store,
            bytes_for_cache,
            merged_superfile_id,
            term_contributions,
        })
    }

    /// Commit a batch of prepared merges in ONE manifest CAS: every new entry
    /// added and every input removed together.
    ///
    /// The jobs a pass plans never share an input, so a batch's removals are
    /// disjoint and its additions independent — the manifest cannot tell a
    /// batch of N from N separate commits, except that it produces one
    /// generation instead of N.
    ///
    /// A single-job batch is exactly the historical per-job commit, which is
    /// what keeps a serial pass byte-for-byte what it was.
    ///
    /// `jobs` is what this commit was handed and `committed` what survived, so
    /// a job whose inputs another compactor took shows as a gap between them.
    #[cfg_attr(
        feature = "detailed-tracing",
        tracing::instrument(
            name = "commit_compaction_batch",
            skip_all,
            fields(
                role = self.role().as_str(),
                jobs = batch.len(),
                attempts = tracing::field::Empty,
                committed = tracing::field::Empty,
            )
        )
    )]
    pub(crate) async fn commit_compaction_batch(
        &self,
        mut batch: Vec<PreparedJob>,
    ) -> Result<(), CompactionError> {
        if batch.is_empty() {
            return Ok(());
        }
        let inner = self.inner();
        let manifest = inner.manifest.load_full();
        let storage = manifest
            .options
            .storage
            .as_ref()
            .ok_or(CompactionError::NoStorage)?
            .clone();
        let wal_store = WalStore::new(storage.clone());
        let opts = Arc::clone(&inner.options);
        let max_retries = opts.max_commit_retries.max(1);

        // Raised by a job that lost its inputs to another compactor after we
        // had already merged. The remaining jobs still commit; the error is
        // surfaced once the batch has settled.
        let mut deferred_error: Option<CompactionError> = None;

        // At most one job can have pre-written its output's sidecar: only a
        // one-in-one-out build carries tombstones, and those run one at a
        // time. `try_commit_attempt` registers a single one.
        let carried: Vec<Uuid> = batch.iter().filter_map(|p| p.carried_sidecar).collect();
        if carried.len() > 1 {
            return Err(CompactionError::Build(format!(
                "{} jobs in one batch carried a tombstone sidecar; a commit can register one",
                carried.len()
            )));
        }
        let carried_sidecar = carried.first().copied();

        for attempt in 0..max_retries {
            let current = inner.manifest.load_full();

            // Another compactor already merged some job's inputs — that job has
            // nothing left to commit, so drop it and keep the rest. On a retry
            // this is a lost race rather than a benign no-op, because we had
            // resolved those inputs once already.
            let mut resolved: Vec<(usize, Vec<Arc<SuperfileEntry>>)> = Vec::new();
            let mut vanished: Vec<usize> = Vec::new();
            for (i, prepared) in batch.iter().enumerate() {
                match resolve_entries_to_remove(&current, &prepared.input_ids) {
                    Ok(entries) => resolved.push((i, entries)),
                    Err(missing) => {
                        if attempt > 0 {
                            deferred_error
                                .get_or_insert(CompactionError::SuperfileNotFound(missing));
                        }
                        vanished.push(i);
                    }
                }
            }
            // Drop the vanished jobs back-to-front so the surviving indices
            // stay valid. Their seals go with them: on a retry the inputs are
            // gone, so there is no sidecar left to clear.
            for i in vanished.into_iter().rev() {
                batch.remove(i);
            }
            if batch.is_empty() {
                return match deferred_error {
                    Some(e) => Err(e),
                    None => Ok(()),
                };
            }

            let stale = match restamp_seals(&wal_store, &mut batch, Utc::now()).await {
                Ok(stale) => stale,
                // Storage failed, so nothing is known about who holds these
                // sidecars and the batch cannot be committed on the strength
                // of it.
                Err(e) => {
                    unseal_batch(&wal_store, batch).await;
                    return Err(e);
                }
            };
            // Back-to-front so the surviving indices stay valid. `resolved[k]`
            // is `batch[k]`'s, since both were built walking the batch in
            // order; the index stored in `resolved` predates the vanished
            // removal and must not address either.
            debug_assert_eq!(resolved.len(), batch.len());
            let mut dropped: Vec<PreparedJob> = Vec::with_capacity(stale.len());
            for i in stale.into_iter().rev() {
                dropped.push(batch.remove(i));
                resolved.remove(i);
            }
            // A dropped job's seals must be cleared: only the input that lost
            // the CAS has left our hands, while the ones re-stamped before it
            // carry a FRESH window under a compaction that will never commit,
            // and the ones after it still carry prepare's. Left behind they
            // block every delete resolving there and fail the next pass
            // outright, since `seal` answers `AlreadySealed` without retrying.
            if !dropped.is_empty() {
                unseal_batch(&wal_store, dropped).await;
            }
            if batch.is_empty() {
                return match deferred_error {
                    Some(e) => Err(e),
                    None => Ok(()),
                };
            }

            let entries_to_remove: Vec<Arc<SuperfileEntry>> =
                resolved.into_iter().flat_map(|(_, e)| e).collect();
            let new_entries: Vec<Arc<SuperfileEntry>> = batch
                .iter()
                .flat_map(|p| p.new_entries.iter().cloned())
                .collect();
            // A term contribution owns a spilled file and is not cloneable, so
            // the batch's are borrowed out for the attempt and handed back if
            // it has to be retried. `owners` records which job each came from.
            let mut term_contributions: Vec<TermContribution> = Vec::new();
            let mut contribution_owners: Vec<usize> = Vec::new();
            for (i, prepared) in batch.iter_mut().enumerate() {
                for contribution in prepared.term_contributions.drain(..) {
                    term_contributions.push(contribution);
                    contribution_owners.push(i);
                }
            }
            // Successful PUTs are drained from this vec, so a retry re-writes
            // only what the previous attempt failed to land.
            let mut pending_storage_writes: Vec<(String, Bytes)> = batch
                .iter_mut()
                .flat_map(|p| p.pending_storage_writes.drain(..))
                .collect();
            let mut pending_storage_replaces: Vec<(String, Bytes)> = Vec::new();

            // The CAS on its own, separated from the cache warm and reclaim
            // that follow it: one span over the whole retry loop could not
            // tell a slow pointer write from several fast ones plus backoff.
            // The seals have to still be ours when the swap lands, not merely
            // when the uploads started: `try_commit_attempt` writes every
            // merged superfile before its pointer PUT, and a seal that expired
            // during that lets a delete land a tombstone on an input this
            // commit then removes.
            let mut fence = SealFence {
                wal_store: &wal_store,
                batch: &mut batch,
            };
            let attempt_outcome = try_commit_attempt(
                storage.clone(),
                Arc::clone(&opts),
                Arc::clone(&current),
                &new_entries,
                &entries_to_remove,
                NewEntryBirthVersions::Preserve,
                carried_sidecar,
                &mut pending_storage_writes,
                &mut pending_storage_replaces,
                &term_contributions,
                Some(&mut fence),
            )
            .instrument(detail_span!(
                "compaction_commit_attempt",
                attempt = attempt,
                superfiles_added = new_entries.len(),
                superfiles_removed = entries_to_remove.len(),
            ))
            .await;
            match attempt_outcome {
                Ok(new_manifest) => {
                    record("attempts", attempt + 1);
                    record("committed", batch.len());
                    // This attempt's own removals, resolved against its own base above, so a
                    // retry that found fewer inputs records only those.
                    inner.note_superseded(&current, &new_manifest, &entries_to_remove);
                    inner.manifest.store(Arc::new(new_manifest));
                    // Point the sidecar cache at the manifest just published;
                    // until then a carried sidecar has no seq and reads as absent.
                    inner.reconcile_tombstone_seqs();
                    // Warm each merged superfile into the in-memory reader
                    // cache, same as a normal writer commit does. Without
                    // this every query against it misses and re-fetches +
                    // re-opens from storage every single time.
                    for prepared in &batch {
                        if let Some((uri, bytes)) = prepared.bytes_for_store.clone()
                            && let Err(e) = opts.store.insert(uri, bytes)
                        {
                            warn!(
                                superfile_id = %prepared.merged_superfile_id,
                                error = %e,
                                "compact: failed to warm reader cache for merged superfile"
                            );
                        }
                    }

                    // Drop the merged-away inputs so the in-memory cache
                    // doesn't grow forever across repeated compactions.
                    // The disk cache is already size-bounded (LRU), so its
                    // stale entries just age out on their own.
                    for entry in &entries_to_remove {
                        opts.store.remove(&entry.uri);
                    }

                    // Disk-cache warm + background storage reclaim ride the
                    // shared post-commit finalizer (the same path writer
                    // commits use), so the two paths can't drift.
                    let pending_cache_inserts: Vec<_> = batch
                        .iter()
                        .filter_map(|p| p.bytes_for_cache.clone())
                        .collect();
                    finalize_compaction_commit(
                        Arc::clone(inner),
                        &storage,
                        &new_entries,
                        &entries_to_remove,
                        pending_cache_inserts,
                    )
                    .await;

                    return match deferred_error {
                        Some(e) => Err(e),
                        None => Ok(()),
                    };
                }
                // A lost pointer CAS and a seal taken over are the same kind of
                // failure: nothing was published, and the next attempt
                // re-resolves against what is there now. Without this arm a
                // single stolen seal would take the catch-all below, unsealing
                // and discarding every finished, already-uploaded merge in the
                // batch and ending the pass.
                Err(
                    e @ (CommitError::WriteContentionExhausted | CommitError::InputsChanged { .. }),
                ) if attempt + 1 < max_retries => {
                    warn!(
                        jobs = batch.len(),
                        attempt,
                        max_retries,
                        error = %e,
                        "compaction commit lost a race, retrying"
                    );
                    // Put the undrained writes and the borrowed contributions
                    // back so the next attempt still knows what it owes.
                    redistribute_pending_writes(&mut batch, pending_storage_writes);
                    for (contribution, owner) in
                        term_contributions.into_iter().zip(contribution_owners)
                    {
                        batch[owner].term_contributions.push(contribution);
                    }
                    if let Err(e) = self.refresh().await {
                        unseal_batch(&wal_store, batch).await;
                        return Err(CompactionError::Refresh(e.to_string()));
                    }
                    time::sleep(backoff_delay(attempt)).await;
                }
                Err(e) => {
                    unseal_batch(&wal_store, batch).await;
                    return Err(CompactionError::Commit(e.to_string()));
                }
            }
        }

        unseal_batch(&wal_store, batch).await;
        Err(CompactionError::Commit(
            "commit retries exhausted".to_string(),
        ))
    }

    /// Seal, merge and commit one job on its own, with no refresh after.
    /// Production runs jobs through [`Self::run_compaction_jobs`]; this is the
    /// single-job shorthand the compaction tests drive directly.
    #[cfg(test)]
    pub(crate) async fn run_compaction_job(
        &self,
        job: CompactionJob,
        stale_seal_timeout: Duration,
    ) -> Result<(), CompactionError> {
        self.run_compaction_job_with(job, stale_seal_timeout, Arc::new(CompactionMerge))
            .await
            .map(|_| ())
    }

    /// As [`Self::run_compaction_job`], with the build injected: prepare and
    /// commit one job on its own, reporting whether it landed.
    ///
    /// A job whose inputs another compactor replaced, or whose input changed
    /// under its seal, did not commit and is not an error to its caller: the
    /// work is simply redone later against what is there now.
    pub(crate) async fn run_compaction_job_with(
        &self,
        job: CompactionJob,
        stale_seal_timeout: Duration,
        merge: Arc<dyn SuperfileMerge>,
    ) -> Result<JobOutcome, CompactionError> {
        // Preparing reports its own failures, an input that was never in the
        // manifest among them. Only the manifest swap can find the inputs
        // gone, which is the race this outcome is for rather than an error.
        let prepared = self
            .prepare_compaction_job_with(job, stale_seal_timeout, merge)
            .await?;
        match self.commit_compaction_batch(vec![prepared]).await {
            Ok(()) => Ok(JobOutcome::Committed),
            Err(CompactionError::SuperfileNotFound(_)) => Ok(JobOutcome::InputsAlreadyReplaced),
            Err(e) => Err(e),
        }
    }

    /// Run a pass's planned jobs, up to `concurrency` merges in flight, and
    /// commit each as it finishes.
    ///
    /// Merges are admitted one at a time against the host and committed when
    /// ready, taking along whatever else has finished, so a serial pass (the
    /// default `concurrency` of 1) is prepare-commit-refresh per job, exactly
    /// as before. A merge that fails does not cost the others their work: the
    /// successful merges commit and the first error surfaces after.
    async fn run_compaction_jobs(
        &self,
        jobs: Vec<CompactionJob>,
        stale_seal_timeout: Duration,
        concurrency: usize,
    ) -> Result<(), CompactionError> {
        let concurrency = concurrency.max(1);
        let mut queued: VecDeque<(usize, CompactionJob)> = jobs.into_iter().enumerate().collect();
        let mut merging: JoinSet<(usize, Result<PreparedJob, CompactionError>)> = JoinSet::new();
        let mut first_error: Option<CompactionError> = None;
        let mut fatal: Option<CompactionError> = None;
        let mut stop_admitting = false;
        // When the next admission may be considered. A deadline rather than a
        // fresh timer per iteration: a merge finishing is not a reason to make
        // the next admission wait another full settle, and under merges
        // shorter than the settle a restarting timer would never elapse at all
        // and the width would collapse to one.
        let mut next_admission = time::Instant::now() + MERGE_ADMIT_SETTLE;

        while !queued.is_empty() || !merging.is_empty() {
            // Nothing running: start one without consulting the host. A table
            // whose single job does not fit still has to compact, and stalling
            // it would strand exactly the tables that most need compacting.
            if merging.is_empty() {
                if stop_admitting {
                    break;
                }
                let (plan_index, job) = queued.pop_front().expect("loop condition");
                self.spawn_merge(&mut merging, plan_index, job, stale_seal_timeout);
                next_admission = time::Instant::now() + MERGE_ADMIT_SETTLE;
                continue;
            }

            // Cheap half of the admission test, as a select guard; the half
            // that reads the host runs inside the branch, after the settle.
            // A failure that will repeat stops the pass admitting more: under
            // a systemic one, say storage is down and every seal exhausts its
            // retries, admitting the rest of the plan only burns each job's
            // retries to reach the same error. A job whose inputs another
            // compactor took is not that, and the rest of the plan still runs.
            let may_admit = !stop_admitting
                && admits_another_merge(merging.len(), concurrency, queued.len(), true);
            let joined = tokio::select! {
                // Completions first: a merge that has finished should commit
                // and free its slot rather than wait behind an admission.
                biased;
                joined = merging.join_next() => joined,
                // The settle runs with every admitted merge in flight, so the
                // reading that follows it sees their allocations rather than
                // an idle host. That is the whole feedback loop: admit, let it
                // land, look at what is left, decide again.
                _ = time::sleep_until(next_admission), if may_admit => {
                    if admits_another_merge(
                        merging.len(),
                        concurrency,
                        queued.len(),
                        host_has_room_for_another_merge(),
                    ) {
                        let (plan_index, job) = queued.pop_front().expect("guarded above");
                        self.spawn_merge(&mut merging, plan_index, job, stale_seal_timeout);
                    }
                    // Advanced whether or not the host had room: a deadline
                    // left in the past would re-fire immediately and spin on
                    // `/proc` until a merge finished.
                    next_admission = time::Instant::now() + MERGE_ADMIT_SETTLE;
                    continue;
                }
            };
            let Some(joined) = joined else {
                continue;
            };

            let mut ready: Vec<(usize, PreparedJob)> = Vec::new();
            stop_admitting |= collect_merge(joined, &mut ready, &mut first_error);
            // Whatever else has already finished rides along in the same CAS.
            // Polling once never waits, so committing early costs a sibling
            // nothing and merges that land together still batch.
            while let Some(next) = merging.try_join_next() {
                stop_admitting |= collect_merge(next, &mut ready, &mut first_error);
            }
            if ready.is_empty() {
                continue;
            }
            // Plan order within the batch, which is as much of it as
            // committing merges when they finish can preserve.
            ready.sort_by_key(|(plan_index, _)| *plan_index);
            let batch: Vec<PreparedJob> = ready.into_iter().map(|(_, prepared)| prepared).collect();

            let commit = self.commit_compaction_batch(batch).await;
            // Both failures leave by the same door: merges are still running,
            // and returning here would drop their tasks with their seals
            // placed and nothing to clear them until they went stale.
            if let Err(e) = self.refresh().await {
                fatal = Some(CompactionError::Refresh(e.to_string()));
                break;
            }
            if let Err(e) = commit {
                fatal = Some(e);
                break;
            }
        }

        if let Some(e) = fatal {
            // Merges still running hold seals this pass will not commit. Let
            // them finish and clear their own, rather than dropping the tasks
            // and leaving the seals to age out.
            self.unseal_remaining(merging).await;
            return Err(e);
        }
        if let Some(e) = first_error {
            return Err(e);
        }
        Ok(())
    }

    /// Start one merge, tagged with its position in the plan.
    ///
    /// Spawned rather than held in a future this loop polls: the loop awaits a
    /// commit between admissions, and a merge parked in an unpolled future
    /// would make no progress while that commit ran.
    fn spawn_merge(
        &self,
        merging: &mut JoinSet<(usize, Result<PreparedJob, CompactionError>)>,
        plan_index: usize,
        job: CompactionJob,
        stale_seal_timeout: Duration,
    ) {
        let table = self.clone();
        merging.spawn(async move {
            let prepared = match table.prepare_compaction_job(job, stale_seal_timeout).await {
                Ok(prepared) => prepared,
                Err(e) => return (plan_index, Err(e)),
            };
            let uploaded = table.upload_prepared_output(prepared).await;
            (plan_index, uploaded)
        });
    }

    /// Put this job's merged bytes in storage, inside its own task.
    ///
    /// The commit used to do this, on the loop that admits merges and makes
    /// commits: a multi-gigabyte PUT there stops every other merge from
    /// starting and every finished one from committing, so their inputs stay
    /// sealed for the length of someone else's upload. The bytes still precede
    /// the pointer, which is what crash safety rests on, and a job that never
    /// commits leaves them for gc exactly as a dropped one already does.
    async fn upload_prepared_output(
        &self,
        mut prepared: PreparedJob,
    ) -> Result<PreparedJob, CompactionError> {
        if prepared.pending_storage_writes.is_empty() {
            return Ok(prepared);
        }
        let inner = self.inner();
        let Some(storage) = inner.manifest.load_full().options.storage.clone() else {
            return Err(CompactionError::NoStorage);
        };
        let opts = Arc::clone(&inner.options);
        let mut writes = mem::take(&mut prepared.pending_storage_writes);
        let mut replaces: Vec<(String, Bytes)> = Vec::new();
        let outcome = write_superfile_list(&storage, &opts, &mut writes, &mut replaces).await;
        // Whatever did not land goes back, so a commit retry re-PUTs exactly
        // the outstanding bytes as it did when the upload lived there.
        prepared.pending_storage_writes = writes;
        match outcome {
            Ok(()) => Ok(prepared),
            Err(e) => {
                let wal_store = WalStore::new(storage);
                unseal_batch(&wal_store, vec![prepared]).await;
                Err(CompactionError::Commit(e.to_string()))
            }
        }
    }

    /// Drain merges still running after a pass has given up, clearing the
    /// seals they placed. Their staged outputs are orphans for gc, exactly as
    /// a job dropped from a batch leaves behind.
    async fn unseal_remaining(
        &self,
        mut merging: JoinSet<(usize, Result<PreparedJob, CompactionError>)>,
    ) {
        let Some(storage) = self.inner().manifest.load_full().options.storage.clone() else {
            return;
        };
        let wal_store = WalStore::new(storage);
        while let Some(joined) = merging.join_next().await {
            if let Ok((_, Ok(prepared))) = joined {
                unseal_batch(&wal_store, vec![prepared]).await;
            }
        }
    }
}

/// Share of the host's memory the runner keeps free: it stops admitting
/// merges once less than this is available.
///
/// The width comes from feedback rather than an estimate of what a merge
/// costs. A merge's footprint depends on term cardinality, posting density,
/// document length and which indexes a table carries — none of which is
/// visible in its input byte count, and all of which shows up in
/// `MemAvailable`. So the runner admits one merge, lets the allocation land,
/// looks at what the host has left, and admits another only if there is still
/// room. A corpus twice as expensive per byte gets fewer merges in flight,
/// with nothing to re-tune.
const MERGE_MEMORY_RESERVE_PERCENT: u64 = 40;

/// How long to let an admitted merge's allocation materialize before reading
/// memory again. Resident size lags admission, so deciding immediately would
/// widen against a reading that has not caught up yet. The admitted merges run
/// throughout this wait, which is what makes the next reading mean something.
/// Merges run for minutes; a short settle between admissions costs nothing
/// measurable.
const MERGE_ADMIT_SETTLE: Duration = Duration::from_millis(250);

/// Whether the host still has room for one more concurrent merge: reads the
/// machine once and hands both figures to [`has_room_for_another_merge`].
///
/// Split so the decision itself is a pure function of two numbers. Reading
/// inside the predicate would mean a test could only re-derive the same
/// arithmetic from a second, later reading of the same host.
fn host_has_room_for_another_merge() -> bool {
    match memory_budget() {
        // One call, not one per half: asking twice re-decides cgroup versus
        // host each time, and a read that fails on the first and succeeds on
        // the second gives a host numerator over a cgroup ceiling, which reads
        // as far above 100% free and never throttles.
        Some((available, total)) => has_room_for_another_merge(Some(available), Some(total)),
        None => has_room_for_another_merge(None, None),
    }
}

/// Whether a host reporting `available` of `total` bytes has room for one more
/// concurrent merge.
///
/// `true` where memory cannot be read at all: there is nothing to throttle
/// against, and a derived width has already resolved to 1 on such a host, so
/// the only way to be here is an explicit width the operator asked for. A
/// `total` of zero is a nonsense reading rather than an absent one, and is the
/// one case that denies.
fn has_room_for_another_merge(available: Option<u64>, total: Option<u64>) -> bool {
    let (Some(available), Some(total)) = (available, total) else {
        return true;
    };
    total > 0 && available.saturating_mul(100) / total >= MERGE_MEMORY_RESERVE_PERCENT
}

/// How much of the staleness window a commit keeps in hand. A seal younger
/// than the window minus this cannot have been stolen yet and cannot expire
/// before the CAS lands, so re-stamping it would be a round trip that only
/// confirms what the clock already proves. It also absorbs modest clock skew
/// between this process and whichever one might steal a seal, since the two
/// judge staleness against their own clocks.
const SEAL_RESTAMP_MARGIN: Duration = Duration::from_secs(30);

/// Whether a seal placed at `sealed_at` could have been taken over by now.
///
/// Measured against the threshold a WRITER steals at, read from the one place
/// that answers that question: a compactor judging it by anything else leaves
/// a window where a delete may take a seal over that this skips re-stamping as
/// too young, which is a lost deletion. Not this pass's own
/// `stale_seal_timeout`, which decides when a compactor takes over another
/// compactor's seal.
fn seal_may_have_been_stolen(
    sealed_at: DateTime<Utc>,
    now: DateTime<Utc>,
    writer_steals_after: Duration,
) -> bool {
    let steal_after = writer_steals_after.saturating_sub(SEAL_RESTAMP_MARGIN);
    match (now - sealed_at).to_std() {
        Ok(age) => age >= steal_after,
        // A seal stamped in the future is a clock that moved; re-stamp rather
        // than trust the arithmetic.
        Err(_) => true,
    }
}

/// Whether another merge may start right now.
///
/// The first is unconditional, whatever the host says: a table whose single
/// job does not fit still has to compact, and stalling it would strand exactly
/// the tables that most need compacting. Past that the width knob caps how
/// many run at once, the host has to report headroom, and a pass never starts
/// work it did not plan. A width of zero still admits one, so a misconfigured
/// knob cannot stall compaction outright.
fn admits_another_merge(
    in_flight: usize,
    concurrency: usize,
    queued: usize,
    host_has_room: bool,
) -> bool {
    if queued == 0 {
        return false;
    }
    if in_flight == 0 {
        return true;
    }
    in_flight < concurrency.max(1) && host_has_room
}

/// Sort one finished merge into the batch being assembled, or record why it
/// produced nothing. A failed prepare has already cleared its own seals.
///
/// Returns whether the pass should stop admitting work. A job whose inputs
/// another compactor took is a race rather than a sick table: the rest of the
/// plan is independent of it and still runs, which is what keeps a contended
/// table compacting at all. Every other failure is one the next job is likely
/// to hit too.
#[must_use]
fn collect_merge(
    joined: Result<(usize, Result<PreparedJob, CompactionError>), JoinError>,
    ready: &mut Vec<(usize, PreparedJob)>,
    first_error: &mut Option<CompactionError>,
) -> bool {
    let failure = match joined {
        Ok((plan_index, Ok(prepared))) => {
            ready.push((plan_index, prepared));
            return false;
        }
        Ok((_, Err(e))) => e,
        Err(e) => CompactionError::Build(format!("merge task failed: {e}")),
    };
    let stops_the_pass = !matches!(failure, CompactionError::SuperfileNotFound(_));
    first_error.get_or_insert(failure);
    stops_the_pass
}

/// One merge that has run and is waiting for its manifest commit.
pub(crate) struct PreparedJob {
    /// Inputs this job claimed, in plan order.
    input_ids: Vec<Uuid>,
    /// Owns the seals on those inputs; the commit re-stamps them under it.
    compaction_id: Uuid,
    /// When those seals were placed, which decides whether the commit has to
    /// re-stamp them at all.
    sealed_at: DateTime<Utc>,
    /// A one-in-one-out build writes its output's sidecar before the commit,
    /// which then has to name it. `None` for an ordinary merge, whose output
    /// has no tombstones to carry.
    carried_sidecar: Option<Uuid>,
    /// Seals placed on those inputs, cleared if the job never commits.
    sealed: Vec<SealedInput>,
    /// The merged superfile's entry. Empty on a pure reclaim, where every
    /// input was fully dead and the commit removes them with no replacement.
    new_entries: Vec<Arc<SuperfileEntry>>,
    /// Superfile bytes still owed to object storage, drained as they land.
    pending_storage_writes: Vec<(String, Bytes)>,
    bytes_for_store: Option<(SuperfileUri, Bytes)>,
    bytes_for_cache: Option<(SuperfileUri, Bytes)>,
    merged_superfile_id: Uuid,
    term_contributions: Vec<TermContribution>,
}

/// Re-stamps a batch's seals with only the pointer PUT left to outlive them.
///
/// The early re-stamp before the uploads is a cheap way to drop a job whose sidecar has already
/// moved, so a multi-gigabyte upload is not spent on work that cannot commit. It is not what makes
/// the commit safe: the uploads take minutes at scale, and a seal can expire inside them. This
/// runs after them, leaving one pointer PUT for a freshly stamped seal to outlive.
///
/// A lost CAS here fails the whole attempt rather than dropping one job: the outputs are already
/// uploaded and the manifest is already built against this batch, so there is nothing left to drop
/// a job from. The retry re-resolves, and its early re-stamp drops the job then.
struct SealFence<'a> {
    wal_store: &'a WalStore,
    batch: &'a mut Vec<PreparedJob>,
}

/// Fault injection for the fence: the next check fails as though a writer took a seal over during
/// the upload. The real interleaving needs a delete to land between the early re-stamp and the
/// pointer PUT, which is microseconds apart in a test and minutes apart only on a real upload.
#[cfg(test)]
pub(crate) static FENCE_FAILS_ONCE: AtomicBool = AtomicBool::new(false);

impl CommitFence for SealFence<'_> {
    fn check(&mut self) -> BoxFuture<'_, Result<(), CommitError>> {
        Box::pin(async move {
            #[cfg(test)]
            if FENCE_FAILS_ONCE.swap(false, Ordering::SeqCst) {
                let superfile_id = self
                    .batch
                    .first()
                    .and_then(|p| p.sealed.first())
                    .map(|s| s.superfile_id)
                    .unwrap_or_default();
                return Err(CommitError::InputsChanged { superfile_id });
            }
            let stale = restamp_seals(self.wal_store, self.batch, Utc::now())
                .await
                .map_err(|e| CommitError::Encode(e.to_string()))?;
            let Some(&i) = stale.first() else {
                return Ok(());
            };
            let superfile_id = self.batch[i]
                .sealed
                .first()
                .map(|s| s.superfile_id)
                .unwrap_or_default();
            Err(CommitError::InputsChanged { superfile_id })
        })
    }
}

/// Re-stamp every seal the batch still needs stamped, conditioned on the etag
/// `seal` returned, and report which jobs are no longer ours.
///
/// A seal expires after the staleness threshold, and a long merge outlives it:
/// a writer then treats the seal as abandoned and lands a tombstone, changing
/// the etag. Committing that input away would drop the bit, since the merged
/// superfile was built from the bitmap the merge read. So a job whose re-stamp
/// loses the CAS is named in the result and must leave the batch, to be merged
/// again next pass with the tombstone in view. A won re-stamp also moves
/// `sealed_at` forward, so the seal cannot expire before the CAS lands.
///
/// Seals too young to have been stolen are skipped: the round trip could only
/// confirm what the clock already proves, and this runs on the critical path
/// of every commit, serial single-job ones included.
///
/// `Err` means storage failed rather than a writer winning, which proves
/// nothing about who holds the sidecar. Any etags already re-stamped are
/// written back through `batch` either way, so an unseal on the error path
/// still clears them.
///
/// The indices come back ascending and at most one per job, which is what lets
/// the caller remove them back-to-front.
async fn restamp_seals(
    wal_store: &WalStore,
    batch: &mut [PreparedJob],
    now: DateTime<Utc>,
) -> Result<Vec<usize>, CompactionError> {
    let mut stale = Vec::new();
    for (i, prepared) in batch.iter_mut().enumerate() {
        if !seal_may_have_been_stolen(
            prepared.sealed_at,
            now,
            tombstones_admin::writer_steal_timeout(),
        ) {
            continue;
        }
        let compaction_id = prepared.compaction_id;
        for input in prepared.sealed.iter_mut() {
            match tombstones_admin::refresh_seal(
                wal_store,
                input.superfile_id,
                compaction_id,
                input.bitmap.clone(),
                now,
                &input.etag,
            )
            .await
            {
                Ok(etag) => input.etag = etag,
                // The same condition the one-in-one-out carry raises as
                // `SidecarChangedUnderSeal`, and the same answer: this job
                // does not commit. A many-in-one merge cannot carry the bit
                // to its output, so the inputs are merged again next pass
                // with the tombstone in view.
                Err(TombstonesAdminError::CasLost { .. }) => {
                    warn!(
                        error = %CompactionError::SidecarChangedUnderSeal {
                            superfile_id: input.superfile_id,
                        },
                        "compact: dropping the job"
                    );
                    stale.push(i);
                    break;
                }
                Err(e) => return Err(CompactionError::Seal(e.to_string())),
            }
        }
        // The seals are young again, so a later commit attempt skips them
        // rather than re-stamping what it refreshed seconds ago.
        if stale.last() != Some(&i) {
            prepared.sealed_at = now;
        }
    }
    Ok(stale)
}

/// Hand the writes a failed attempt did not land back to the jobs that owe
/// them, so the next attempt re-PUTs exactly the outstanding bytes. Keyed by
/// storage path, which is unique per superfile.
fn redistribute_pending_writes(batch: &mut [PreparedJob], outstanding: Vec<(String, Bytes)>) {
    for (path, bytes) in outstanding {
        // A job dropped from the batch this attempt owns none of these; its
        // bytes are an orphan for gc, not something to retry.
        if let Some(owner) = batch
            .iter_mut()
            .find(|p| p.new_entries.iter().any(|e| e.storage_path() == path))
        {
            owner.pending_storage_writes.push((path, bytes));
        }
    }
}

/// Clear every seal a batch placed. Called when the batch will not commit.
async fn unseal_batch(wal_store: &WalStore, batch: Vec<PreparedJob>) {
    let sealed: Vec<SealedInput> = batch.into_iter().flat_map(|p| p.sealed).collect();
    unseal_all(wal_store, sealed).await;
}

/// Write the input's tombstone bitmap onto the superfile that replaces it.
///
/// Sound only because a carried row set leaves local doc ids where they
/// were, which the doc-count check below verifies rather than assumes.
/// Restricted to one input and one output: several would need each bitmap
/// shifted by the rows ahead of it, and every carrying job is a
/// one-in-one-out migration.
async fn carry_tombstones_to_output(
    merge: &dyn SuperfileMerge,
    wal_store: &WalStore,
    inputs: &[Arc<SuperfileEntry>],
    new_entries: &[Arc<SuperfileEntry>],
    sealed: &[SealedInput],
) -> Result<Option<Uuid>, CompactionError> {
    // No output means no rows survived, which needs no sidecar.
    if !merge.preserves_tombstones() || new_entries.is_empty() {
        return Ok(None);
    }
    let ([input], [output]) = (inputs, new_entries) else {
        return Err(CompactionError::Build(format!(
            "a tombstone-preserving build must be one-in-one-out, got {} input(s) \
             and {} output(s)",
            inputs.len(),
            new_entries.len()
        )));
    };

    // The bitmap this job holds was read when it sealed the input. The
    // mutation path takes a seal over once it is older than its own
    // timeout, clears it and writes its bit — so on a job that outlived
    // that, the sealed bitmap is missing rows the input now has. The etag
    // is what says so, and it is checked before the empty-bitmap exit
    // below: sealing an empty sidecar and taking over are what a job that
    // silently drops the first delete against its input looks like.
    let held = sealed.iter().find(|s| s.superfile_id == input.superfile_id);
    let current_etag = wal_store
        .get_tombstones(input.superfile_id)
        .await
        .map_err(|e| {
            CompactionError::Build(format!(
                "re-reading the sidecar of {}: {e}",
                input.superfile_id
            ))
        })?
        .map(|(_, etag)| etag);
    if current_etag.as_ref() != held.map(|s| &s.etag) {
        return Err(CompactionError::SidecarChangedUnderSeal {
            superfile_id: input.superfile_id,
        });
    }

    // An absent sidecar *is* the empty state, so writing one would leave an
    // object for GC to collect and every reader to fetch for nothing.
    let Some(bitmap) = held.map(|s| &s.bitmap).filter(|b| !b.is_empty()) else {
        return Ok(None);
    };

    if output.n_docs != input.n_docs {
        return Err(CompactionError::Build(format!(
            "a tombstone-preserving build changed the document count of {} \
             ({} in, {} out), so its local doc ids moved and its tombstones \
             no longer describe it",
            input.superfile_id, input.n_docs, output.n_docs
        )));
    }

    let sidecar = TombstonesSidecar {
        // The output is live and may be tombstoned into; the input's seal
        // would lock it against the mutation path from birth.
        seal: None,
        bitmap: bitmap.clone(),
    };
    wal_store
        .put_tombstones(output.superfile_id, None, &sidecar)
        .await
        .map_err(|e| {
            CompactionError::Build(format!(
                "carrying tombstones onto {}: {e}",
                output.superfile_id
            ))
        })?;
    Ok(Some(output.superfile_id))
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
    stale_seal_timeout: Duration,
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
                return Err(CompactionError::SealRetriesExhausted { superfile_id });
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
    Err(CompactionError::SealRetriesExhausted { superfile_id })
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, env, mem, str, sync::Arc, time::Duration};

    use arrow_array::{
        ArrayRef, Decimal128Array, FixedSizeListArray, Float32Array, LargeStringArray, RecordBatch,
    };
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::prelude::{col, lit};
    use rayon::ThreadPoolBuilder;
    use tempfile::TempDir;
    use tokio::task;

    use super::{
        plan::tests::{default_cfg, seg},
        *,
    };
    use crate::{
        Bm25Stats, BoolMode, VectorSearchOptions,
        config::{DEFAULT_GC_SAFETY_GAP, DEFAULT_STALE_SEAL_TIMEOUT_MS, OptimizeOptions},
        memory::ConnectionMemoryBudget,
        superfile::{
            builder::{FtsConfig, VectorConfig},
            fts::reader::Bm25SearchOptions,
            reader::SuperfileReader,
            vector::{distance::Metric, rerank_codec::RerankCodec},
        },
        supertable::{
            Supertable, SupertableOptions,
            error::CompactionError,
            manifest::commit::{POINTER_PATH, get_current_manifest_etag},
            storage::{LocalFsStorageProvider, StorageProvider},
        },
        test_helpers::{
            build_title_batch, default_supertable_options, default_vector_config,
            fault_storage::{FaultKind, FaultOp, FaultStorage},
        },
    };

    const DEFAULT_STALE_SEAL_TIMEOUT: Duration =
        Duration::from_millis(DEFAULT_STALE_SEAL_TIMEOUT_MS);

    /// A build that carries every row, so the runner must carry the
    /// input's tombstones onto its output.
    struct AlwaysCarriesRows;

    impl SuperfileMerge for AlwaysCarriesRows {
        fn build(
            &self,
            _inputs: MergeInputs<'_>,
            _output: &mut dyn Write,
        ) -> Result<BuiltSuperfileStats, BuildError> {
            unreachable!("the tombstone carry never builds")
        }

        fn preserves_tombstones(&self) -> bool {
            true
        }
    }

    /// A delete that lands while a job is building must not be dropped by
    /// that job's commit.
    ///
    /// The runner seals the input's sidecar and holds the bitmap it read
    /// at seal time. The delete path takes a seal over once it is older
    /// than its own fixed timeout — reachable here because a reindex
    /// targets the largest superfiles — clears it, and writes its bit.
    /// Publishing the seal-time bitmap onto the output then loses that
    /// row, and nothing downstream can tell: the output is live and its
    /// sidecar looks authoritative.
    ///
    /// The input's etag is what says the sidecar moved, so the carry must
    /// check it rather than trust the seal it took.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_delete_that_lands_mid_job_is_not_dropped_by_the_carry() {
        const SEALED_ROW: u32 = 0;
        const ROW_DELETED_MID_JOB: u32 = 1;

        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);

        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let wal_store = WalStore::new(storage);

        let manifest = st.inner().manifest.load_full();
        let input = manifest
            .get_all_superfiles()
            .first()
            .expect("one superfile")
            .clone();
        let output = Arc::new(SuperfileEntry {
            superfile_id: Uuid::from_u128(0xFEED),
            ..(*input).clone()
        });

        // What the runner sees when it seals.
        let at_seal: RoaringBitmap = [SEALED_ROW].into_iter().collect();
        let sealed_etag = wal_store
            .put_tombstones(
                input.superfile_id,
                None,
                &TombstonesSidecar {
                    seal: Some(SealRecord {
                        compaction_id: Uuid::from_u128(1),
                        sealed_at: Utc::now(),
                    }),
                    bitmap: at_seal.clone(),
                },
            )
            .await
            .expect("seal the input");
        let sealed = vec![SealedInput {
            superfile_id: input.superfile_id,
            bitmap: at_seal,
            etag: sealed_etag.clone(),
        }];

        // The delete path finds the seal stale, clears it, adds its bit.
        let taken_over: RoaringBitmap = [SEALED_ROW, ROW_DELETED_MID_JOB].into_iter().collect();
        wal_store
            .put_tombstones(
                input.superfile_id,
                Some(&sealed_etag),
                &TombstonesSidecar {
                    seal: None,
                    bitmap: taken_over,
                },
            )
            .await
            .expect("take the stale seal over");

        let carried = carry_tombstones_to_output(
            &AlwaysCarriesRows,
            &wal_store,
            &[Arc::clone(&input)],
            &[Arc::clone(&output)],
            &sealed,
        )
        .await;

        assert!(
            carried.is_err(),
            "the sidecar moved under the seal, so the carry must refuse rather \
             than publish the bitmap it read at seal time"
        );
        assert!(
            wal_store
                .get_tombstones(output.superfile_id)
                .await
                .expect("read output sidecar")
                .is_none(),
            "and it must not have written a sidecar onto the output"
        );
    }

    /// The same loss, on a superfile that had no tombstones when the job
    /// sealed it — the first delete against it.
    ///
    /// An empty sealed bitmap means "write no sidecar", which looks like a
    /// no-op and is why this case hides: the job publishes an output with
    /// no tombstones at all, so the delete leaves no trace anywhere. The
    /// etag check has to come before that exit, not after it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_first_delete_mid_job_is_not_dropped_by_the_empty_carry() {
        const ROW_DELETED_MID_JOB: u32 = 1;

        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_titles(&st, &["alpha first", "alpha second"]);

        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let wal_store = WalStore::new(storage);

        let manifest = st.inner().manifest.load_full();
        let input = manifest
            .get_all_superfiles()
            .first()
            .expect("one superfile")
            .clone();
        let output = Arc::new(SuperfileEntry {
            superfile_id: Uuid::from_u128(0xFEED),
            ..(*input).clone()
        });

        let sealed_etag = wal_store
            .put_tombstones(
                input.superfile_id,
                None,
                &TombstonesSidecar {
                    seal: Some(SealRecord {
                        compaction_id: Uuid::from_u128(1),
                        sealed_at: Utc::now(),
                    }),
                    bitmap: RoaringBitmap::new(),
                },
            )
            .await
            .expect("seal the input");
        let sealed = vec![SealedInput {
            superfile_id: input.superfile_id,
            bitmap: RoaringBitmap::new(),
            etag: sealed_etag.clone(),
        }];

        wal_store
            .put_tombstones(
                input.superfile_id,
                Some(&sealed_etag),
                &TombstonesSidecar {
                    seal: None,
                    bitmap: [ROW_DELETED_MID_JOB].into_iter().collect(),
                },
            )
            .await
            .expect("take the stale seal over");

        let carried = carry_tombstones_to_output(
            &AlwaysCarriesRows,
            &wal_store,
            &[Arc::clone(&input)],
            &[Arc::clone(&output)],
            &sealed,
        )
        .await;

        assert!(
            carried.is_err(),
            "an empty sealed bitmap is not permission to publish an output \
             with no tombstones when the input has since gained one"
        );
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

        let entries = st.reader().expect("reader").manifest().superfiles.clone();
        assert_eq!(entries.len(), 2);
        let (entry_a, entry_b) = (&entries[0], &entries[1]);

        // entry_b is already held by a different, still-live compaction.
        let wal_store = wal_store_for(&st);
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

        let entries = st.reader().expect("reader").manifest().superfiles.clone();
        let crashed_entry = &entries[0];

        // Simulate a compactor that sealed this file and then died
        // long enough ago that its seal is now stale.
        let wal_store = wal_store_for(&st);
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
            .expect("reader")
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
    fn hidden_profile_select_merges_small_same_cell_files() {
        let mut segs = Vec::new();
        for i in 0..4 {
            let mut s = seg(i, 1, 1000, 0);
            s.partition_key = 3u32.to_le_bytes().to_vec();
            segs.push(s);
        }
        // Exercises same-cell selection grouping independent of the
        // production target; a small target keeps the 1 MiB fixtures under
        // the ceiling while their combined size clears the fill floor.
        let cfg = CompactionSettings {
            target_superfile_size_mb: 8,
            min_fill_percent: 40,
            ..CompactionSettings::default()
        };
        let jobs = select(&segs, &cfg);
        assert!(
            !jobs.is_empty(),
            "expected a merge job for 4×1MiB files in one cell partition"
        );
        assert_eq!(jobs[0].partition_key, 3u32.to_le_bytes().to_vec());
        assert!(jobs[0].inputs.len() >= 2);
    }

    #[test]
    fn zero_fill_floor_merges_tiny_fragments_on_count() {
        // Hidden-index policy: a 0% fill floor drives consolidation on the
        // >= 2 fragment count alone. Two sub-target fragments in one cell must
        // merge even though their combined bytes are a tiny fraction of the
        // target — each unmerged fragment is a drain generation that costs a
        // query a fine-run. Under a byte floor the same fragments never merge.
        let mut segs = Vec::new();
        for i in 0..2 {
            let mut s = seg(i, 1, 1000, 0); // 1 MiB each
            s.partition_key = 7u32.to_le_bytes().to_vec();
            segs.push(s);
        }
        let count_driven = CompactionSettings {
            target_superfile_size_mb: 2048,
            min_fill_percent: 0,
            ..CompactionSettings::default()
        };
        let jobs = select(&segs, &count_driven);
        assert_eq!(
            jobs.len(),
            1,
            "0% floor must merge 2 tiny fragments on count"
        );
        assert_eq!(jobs[0].inputs.len(), 2);

        // 2 MiB is far below 40% of a 2 GiB target → the byte floor blocks it.
        let byte_floored = CompactionSettings {
            min_fill_percent: 40,
            ..count_driven.clone()
        };
        assert!(
            select(&segs, &byte_floored).is_empty(),
            "a byte floor must block consolidation of tiny fragments"
        );
    }

    #[test]
    fn user_table_merges_tiny_fragments_on_count_below_size_floor() {
        // Many tiny appends, each a sub-target superfile, whose combined live
        // bytes stay far under the 80% size floor. Without the fragment-count
        // trigger these never merge, so the superfile (and manifest-part) count
        // grows without bound. The count leg consolidates them on count alone.
        // A low `min_superfiles_for_merge` lets the test trip the trigger with a
        // handful of fragments instead of the default 50.
        let cfg = CompactionSettings {
            min_superfiles_for_merge: 3,
            ..CompactionSettings::default() // 1 GiB target, 80% floor (819 MiB)
        };
        // Two 1 MiB fragments: below the count trigger and far below the floor.
        let two = vec![seg(1, 1, 1000, 0), seg(2, 1, 1000, 0)];
        assert!(
            select(&two, &cfg).is_empty(),
            "2 < min_superfiles_for_merge (3) and 2 MiB << 819 MiB floor: no merge"
        );
        // A third fragment trips the count trigger even though 3 MiB << the floor.
        let three = vec![seg(1, 1, 1000, 0), seg(2, 1, 1000, 0), seg(3, 1, 1000, 0)];
        let jobs = select(&three, &cfg);
        assert_eq!(jobs.len(), 1, "count trigger merges once inputs reach 3");
        assert_eq!(jobs[0].inputs.len(), 3);
    }

    #[test]
    fn min_superfiles_for_merge_below_two_is_clamped() {
        // A degenerate config (< 2) must not fire single-input no-op merges: it
        // is raised to 2, so one fragment never merges but two do — even under a
        // floor that blocks the size leg entirely.
        let cfg = CompactionSettings {
            target_superfile_size_mb: 2048,
            min_fill_percent: 100, // size leg unreachable for tiny fragments
            min_superfiles_for_merge: 1,
            ..CompactionSettings::default()
        };
        assert!(
            select(&[seg(1, 1, 1000, 0)], &cfg).is_empty(),
            "one input never merges (clamped floor is 2)"
        );
        let jobs = select(&[seg(1, 1, 1000, 0), seg(2, 1, 1000, 0)], &cfg);
        assert_eq!(
            jobs.len(),
            1,
            "clamped count floor of 2 merges two fragments"
        );
        assert_eq!(jobs[0].inputs.len(), 2);
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
        let reader = st.reader().expect("reader");
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
            .merge_superfiles(&superfiles, &no_tombstones())
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

        let reader = st.reader().expect("reader");
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
            .merge_superfiles(&superfiles, &no_tombstones())
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

        let reader = st.reader().expect("reader");
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
            .merge_superfiles(&superfiles, &no_tombstones())
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
            let (hits, _) = merged_reader
                .token_match("title", &[*term], BoolMode::And)
                .await
                .unwrap_or_else(|_| panic!("token_match for '{term}'"));
            assert_eq!(hits.len(), 2, "term '{term}' should match exactly 2 docs");
        }
    }

    /// Stable ids of every row in `reader`, in row order.
    fn read_ids(reader: &SuperfileReader) -> Vec<i128> {
        let batch = reader.get_record_batch(None).expect("record batch");
        batch
            .column(
                batch
                    .schema()
                    .index_of(reader.id_column())
                    .expect("id column"),
            )
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal ids")
            .values()
            .to_vec()
    }

    /// Inputs are opened concurrently, but the merged rows must still follow
    /// the input order. Otherwise a merged file with a contiguous id span
    /// maps local rows to the wrong `_id` via `id_min + local`. The first
    /// inputs are the largest so they tend to finish opening last.
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_keeps_input_row_order() {
        const N_INPUTS: usize = 8;
        const ROWS_PER_STEP: usize = 500;

        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for i in 0..N_INPUTS {
            let titles: Vec<String> = (0..(N_INPUTS - i) * ROWS_PER_STEP)
                .map(|r| format!("doc {r}"))
                .collect();
            let titles: Vec<&str> = titles.iter().map(String::as_str).collect();
            commit_titles(&st, &titles);
        }

        let superfiles: Vec<Arc<SuperfileEntry>> = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .to_vec();
        assert_eq!(superfiles.len(), N_INPUTS);

        let merged = st
            .merge_superfiles(&superfiles, &no_tombstones())
            .await
            .expect("merge_superfiles should succeed");
        let merged_reader = merged
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        // Ids are not always contiguous within one commit, so read each
        // input's ids from its own rows.
        let storage = st
            .inner()
            .manifest
            .load_full()
            .options
            .storage
            .clone()
            .expect("storage-backed table");
        let mut expected = Vec::new();
        for entry in &superfiles {
            let (bytes, _) = storage.get(&entry.storage_path()).await.expect("get input");
            let reader = SuperfileReader::open(bytes).expect("open input");
            expected.extend(read_ids(&reader));
        }
        let ids = read_ids(&merged_reader);
        assert_eq!(ids, expected, "merged rows must follow input order");
    }

    /// Ranked BM25 search must survive the k-way compaction merge. Two docs
    /// with the same term frequency and the same document frequency but
    /// different lengths must get *different*, length-normalized scores against
    /// the merged-corpus average document length — the shorter one higher. That
    /// only holds if the merge carried each input's per-doc lengths and token
    /// totals across correctly; a merge that dropped them collapses the
    /// length-normalization table (equal scores, or a panic on an empty table).
    /// `token_match` (unranked) can't see this — it only checks presence — so
    /// this exercises the ranked path through the actual `merge_superfiles`
    /// dispatch + streamed temp-file output, complementing the builder oracle.
    /// A merged superfile replaces its inputs, so it bakes the table-wide
    /// average over everything that remains beside it plus itself — not
    /// its inputs' own average — and carries its own totals for the next
    /// fold. Here commit A (5 + 1 tokens), B (2 tokens), C (10 tokens) are
    /// four documents; merging A and B must declare (6 + 2 + 10) / 4.
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_bakes_the_table_wide_average() {
        use crate::superfile::fts::{bm25, reader::ColumnLengthStats};

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");
        for titles in [
            &["cat bird elephant giraffe hippo", "dog"][..],
            &["cat dog"][..],
            &["one two three four five six seven eight nine ten"][..],
        ] {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(titles)).expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader().expect("reader");
        let mut superfiles: Vec<Arc<SuperfileEntry>> =
            reader.manifest().get_all_superfiles().to_vec();
        assert_eq!(superfiles.len(), 3, "three ingest superfiles");
        superfiles.sort_by_key(|sf| sf.id_min);
        let inputs = &superfiles[..2];

        let merged = st
            .merge_superfiles(inputs, &no_tombstones())
            .await
            .expect("merge_superfiles should succeed");
        let merged_reader = merged
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");
        let fts = merged_reader.fts().expect("fts index");
        assert_eq!(
            fts.column_length_stats("title"),
            Some(ColumnLengthStats {
                total_tokens: 8,
                n_scored_docs: 3,
            }),
            "the merged file's own totals"
        );
        let declared = fts
            .fts_columns_config()
            .next()
            .expect("title column")
            .avgdl();
        assert_eq!(
            declared,
            bm25::stored_avgdl((6.0 + 2.0 + 10.0) / 4.0),
            "declared average spans the remaining superfile too, not just the inputs"
        );
        assert_ne!(
            declared,
            bm25::stored_avgdl(8.0 / 3.0),
            "not the inputs' own average"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_preserves_bm25_length_normalization() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");

        // Superfile 1: a short "cat" doc. Superfile 2: a long "cat" doc. Across
        // the merged corpus tf(cat)=1 and df(cat)=2 for both, so the score gap
        // is purely BM25 length normalization against avgdl.
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&["cat", "dog"]))
                .expect("append");
            w.commit().expect("commit");
        }
        {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&[
                "cat bird elephant giraffe hippo",
                "dog",
            ]))
            .expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader().expect("reader");
        let mut superfiles: Vec<Arc<SuperfileEntry>> =
            reader.manifest().get_all_superfiles().to_vec();
        assert_eq!(superfiles.len(), 2, "two ingest superfiles");
        // Merge input order fixes the output doc-id layout; order by id_min so
        // the short-cat doc lands at merged doc 0 and the long-cat doc at 2.
        superfiles.sort_by_key(|sf| sf.id_min);

        let merged = st
            .merge_superfiles(&superfiles, &no_tombstones())
            .await
            .expect("merge_superfiles should succeed");
        let merged_reader = merged
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");
        assert_eq!(merged_reader.n_docs(), 4);

        let hits = merged_reader
            .bm25_search_pretokenized("title", &["cat"], 10, BoolMode::Or)
            .await
            .expect("ranked bm25 search on the merged superfile");
        assert_eq!(hits.len(), 2, "both 'cat' docs must match after the merge");
        for (doc, score) in &hits {
            assert!(
                score.is_finite() && *score > 0.0,
                "doc {doc} score must be finite and positive, got {score}"
            );
        }
        let score_of = |target: u32| -> f32 {
            hits.iter()
                .find(|(doc, _)| *doc == target)
                .unwrap_or_else(|| panic!("expected a hit for merged doc {target}"))
                .1
        };
        let short = score_of(0); // "cat" (length 1)
        let long = score_of(2); // "cat bird elephant giraffe hippo" (length 5)
        assert!(
            short > long,
            "BM25 length normalization must carry across the merge: \
             short-doc score {short} must exceed long-doc score {long}"
        );
    }

    /// Compaction dispatch: superfiles whose vector column is **not**
    /// IVF-mergeable (an `Fp32` rerank codec) must take the re-index branch
    /// (`build_from_readers_to`), which re-encodes both FTS and vectors — not
    /// the FTS-only k-way merge, which carries no vectors. This guards that
    /// routing: after merging such inputs the merged superfile must still have
    /// a queryable vector index (and its FTS index). If the dispatch had
    /// wrongly picked the FTS merge, `vec()` would be `None` here.
    #[tokio::test(flavor = "multi_thread")]
    async fn merge_superfiles_preserves_vectors_for_non_ivf_mergeable_inputs() {
        const DIM: usize = 16;
        let emb_field = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(Arc::clone(&emb_field), DIM as i32),
                false,
            ),
        ]));

        // `default_vector_config` uses RerankCodec::Fp32 — deliberately the
        // non-IVF-mergeable case, so compaction routes to the re-index branch.
        let opts = SupertableOptions::new(
            Arc::clone(&schema),
            vec![FtsConfig::new("title")],
            vec![default_vector_config("emb", 42)],
        )
        .expect("options with an fp32 vector column")
        // One writer thread ⇒ one superfile per commit (deterministic doc-id
        // layout), matching `default_supertable_options`.
        .with_writer_pool(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("1-thread writer pool"),
        ));

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(opts.with_storage(Arc::clone(&storage))).expect("create");

        // One-hot vectors so nearest-neighbour is unambiguous. `title` gives the
        // FTS side something to index. `axes` are the hot dimension per row.
        let make_batch = |titles: &[&str], axes: &[usize]| -> RecordBatch {
            let mut flat = vec![0.0f32; titles.len() * DIM];
            for (row, &ax) in axes.iter().enumerate() {
                flat[row * DIM + ax] = 1.0;
            }
            let emb = FixedSizeListArray::try_new(
                Arc::clone(&emb_field),
                DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .expect("fixed-size-list");
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(titles.to_vec())) as ArrayRef,
                    Arc::new(emb) as ArrayRef,
                ],
            )
            .expect("batch")
        };

        {
            let mut w = st.writer().expect("writer");
            w.append(&make_batch(
                &["alpha", "alpha", "alpha", "alpha"],
                &[0, 1, 2, 3],
            ))
            .expect("append");
            w.commit().expect("commit");
        }
        {
            let mut w = st.writer().expect("writer");
            w.append(&make_batch(
                &["beta", "beta", "beta", "beta"],
                &[4, 5, 6, 7],
            ))
            .expect("append");
            w.commit().expect("commit");
        }

        let reader = st.reader().expect("reader");
        let mut superfiles: Vec<Arc<SuperfileEntry>> =
            reader.manifest().get_all_superfiles().to_vec();
        assert_eq!(superfiles.len(), 2, "two ingest superfiles");
        // Deterministic output doc-id layout: first superfile's rows land at 0..4.
        superfiles.sort_by_key(|sf| sf.id_min);

        let merged = st
            .merge_superfiles(&superfiles, &no_tombstones())
            .await
            .expect("merge_superfiles should succeed");
        let merged_reader = merged
            .open_reader()
            .expect("merged superfile should have bytes")
            .expect("open reader on merged superfile");

        assert_eq!(merged_reader.n_docs(), 8);
        // The re-index branch must preserve BOTH indexes.
        assert!(
            merged_reader.vec().is_some(),
            "vector index must survive the merge (routing must not use the FTS-only path)"
        );
        assert!(
            merged_reader.fts().is_some(),
            "FTS index must survive the merge"
        );

        // Vectors are queryable end to end. Query the exact one-hot of merged
        // doc 0 (first superfile, row 0, axis 0); with a full-cluster nprobe and
        // exact fp32 rerank it must come back as the nearest.
        let mut query = vec![0.0f32; DIM];
        query[0] = 1.0;
        let hits = merged_reader
            .vector_hits_async("emb", &query, 8, VectorSearchOptions::new().with_nprobe(64))
            .await
            .expect("vector search on the merged superfile");
        assert!(!hits.is_empty(), "vector search must return hits");
        assert_eq!(hits[0].0, 0, "nearest to the axis-0 query is merged doc 0");

        // FTS side re-encoded too: every first-superfile doc carries "alpha".
        let fts_hits = merged_reader
            .token_match("title", &["alpha"], BoolMode::And)
            .await
            .expect("token_match on merged superfile")
            .0;
        assert_eq!(fts_hits.len(), 4, "all four 'alpha' docs must match");
    }

    /// A merge is not charged to the connection memory budget.
    ///
    /// That budget bounds what search and ingest may hold for a connection; an
    /// optimize is neither, and sizing it from a knob set for queries meant a
    /// pass could be refused on a machine with the memory to spare — the more
    /// easily the more merges ran at once. The host is what bounds a merge.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_merge_is_not_charged_to_the_connection_budget() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));

        // Written under a normal budget: ingest draws from the same one, so a
        // tight limit here would starve the setup appends too.
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create supertable");
        for titles in [["first doc", "second doc"], ["third doc", "fourth doc"]] {
            let mut w = st.writer().expect("writer");
            w.append(&build_title_batch(&titles)).expect("append");
            w.commit().expect("commit");
        }

        // Reopened with a budget that cannot admit a single byte.
        let mut opts = default_supertable_options().with_storage(Arc::clone(&storage));
        opts.connection_memory_budget = ConnectionMemoryBudget::with_limit(1);
        let st = Supertable::create(opts).expect("reopen supertable");

        let reader = st.reader().expect("reader");
        let superfiles: Vec<Arc<SuperfileEntry>> = reader.manifest().get_all_superfiles().to_vec();
        st.merge_superfiles(&superfiles, &no_tombstones())
            .await
            .expect("a merge must not be refused by a budget that is not about it");
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

        let reader = st.reader().expect("reader");
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
            .merge_superfiles(&superfiles, &no_tombstones())
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
            .expect("token_match for 'only'")
            .0;
        assert_eq!(
            only_hits.len(),
            1,
            "should find exactly 1 doc matching 'only'"
        );

        let second_hits = merged_reader
            .token_match("title", &["second"], BoolMode::And)
            .await
            .expect("token_match for 'second'")
            .0;
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

    /// Inputs with no tombstone sidecars at all, which is every merge fixture
    /// that never deletes. Named rather than inlined so a merge test that
    /// DOES tombstone cannot pass an empty map by accident.
    fn no_tombstones() -> HashMap<Uuid, Arc<RoaringBitmap>> {
        HashMap::new()
    }

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

        let before_docs = st.reader().expect("reader").n_docs_total();
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
        let after_docs = st.reader().expect("reader").n_docs_total();
        assert_eq!(
            after_docs,
            before_docs + 2,
            "writer's 2 docs must survive alongside compacted data"
        );
    }

    /// Ten small commits, one superfile each, that `small_compact_cfg` merges
    /// into one job.
    fn commit_mergeable_superfiles(st: &Supertable) {
        for titles in &[
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
            commit_titles(st, titles);
        }
    }

    /// Assert the manifest lists each superfile once and each of `rows` once,
    /// and nothing else. Returns the listed superfile ids.
    async fn assert_listed_once(st: &Supertable, rows: &HashSet<i128>) -> Vec<Uuid> {
        let listed: Vec<Uuid> = st
            .inner()
            .manifest
            .load_full()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(
            listed.iter().collect::<HashSet<_>>().len(),
            listed.len(),
            "no superfile is listed twice: {listed:?}"
        );
        let ids = listed_row_ids(st).await;
        let distinct: HashSet<i128> = ids.iter().copied().collect();
        assert_eq!(ids.len(), distinct.len(), "every row id appears once");
        assert_eq!(&distinct, rows, "no row was lost or added");
        listed
    }

    /// Publish a successor manifest listing the table's first superfile a
    /// second time, as an append committed twice left it: same id and uri, a
    /// later `birth_version`. Returns that superfile's id.
    async fn list_first_superfile_twice(st: &Supertable) -> Uuid {
        let inner = st.inner();
        let storage = inner.options.storage.clone().expect("storage-backed table");
        let current = inner.manifest.load_full();
        let first = Arc::clone(&current.get_all_superfiles()[0]);
        // `update` stamps the partition key, so the copy arrives unstamped.
        let again = Arc::new(SuperfileEntry {
            partition_key: Vec::new(),
            ..(*first).clone()
        });
        let (successor, parts) = current
            .update_admitting_duplicates(&[again])
            .await
            .expect("successor listing the superfile twice");
        let prev_etag = get_current_manifest_etag(&storage, Arc::clone(&current))
            .await
            .expect("pointer etag");
        let encoded: Vec<&[u8]> = parts
            .iter()
            .flat_map(|p| [Some(p.encoded.as_slice()), p.routing_encoded.as_deref()])
            .flatten()
            .collect();
        successor
            .write(storage.as_ref(), prev_etag.as_deref(), &encoded)
            .await
            .expect("publish");
        st.refresh().await.expect("refresh");
        first.superfile_id
    }

    /// Row ids of every superfile the manifest lists, read from storage. A
    /// superfile listed twice contributes its ids twice.
    async fn listed_row_ids(st: &Supertable) -> Vec<i128> {
        let inner = st.inner();
        let storage = inner.options.storage.clone().expect("storage-backed table");
        let mut ids = Vec::new();
        for entry in inner.manifest.load_full().get_all_superfiles() {
            let (bytes, _) = storage
                .get(&entry.storage_path())
                .await
                .expect("superfile bytes");
            ids.extend(read_ids(
                &SuperfileReader::open(bytes).expect("open superfile"),
            ));
        }
        ids
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_repairs_a_superfile_listed_twice() {
        // An append committed twice left one superfile listed at two birth
        // versions.
        //  - every read sees that superfile's rows twice.
        //  - compaction plans it once, so the merge reads its rows once.
        //  - the commit removes it by id, which drops both copies.
        // One pass leaves no superfile listed twice and every row id once.
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        commit_mergeable_superfiles(&st);
        let twice = list_first_superfile_twice(&st).await;

        let before = listed_row_ids(&st).await;
        let distinct: HashSet<i128> = before.iter().copied().collect();
        assert!(
            distinct.len() < before.len(),
            "guard: the fixture lists some rows twice"
        );

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact a manifest listing a superfile twice");

        let listed = assert_listed_once(&st, &distinct).await;
        assert!(
            !listed.contains(&twice),
            "the merge replaced the superfile listed twice: {listed:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn compact_whose_pointer_response_is_lost_lists_the_merge_once() {
        // The merge's pointer PUT lands, but its response is lost.
        //  - the storage client re-issues the PUT, which fails its etag check
        //    against the merge's own write.
        //  - the compaction sees a lost race. Its inputs are gone because the
        //    merge replaced them, so it stops instead of committing again, and
        //    reports an error although the merge is published.
        // Whatever it returns, the table lists the merged superfile once and
        // each row once, with retries left and on the last attempt.
        let retry_budgets = [1, default_supertable_options().max_commit_retries];
        for max_commit_retries in retry_budgets {
            let dir = TempDir::new().expect("tempdir");
            let local: Arc<dyn StorageProvider> =
                Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
            let faults = FaultStorage::wrap(local);
            let storage: Arc<dyn StorageProvider> = Arc::<FaultStorage>::clone(&faults);
            let st = Supertable::create(
                default_supertable_options()
                    .with_storage(storage)
                    .with_max_commit_retries(max_commit_retries),
            )
            .expect("create supertable");
            commit_mergeable_superfiles(&st);
            let rows: HashSet<i128> = listed_row_ids(&st).await.into_iter().collect();

            faults.fail_with(
                FaultKind::ResponseLost,
                FaultOp::PutIfMatch,
                POINTER_PATH,
                1,
            );
            // An error here is expected: the published merge looks like lost inputs.
            let _ = st.compact_async(&small_compact_cfg()).await;
            assert_eq!(faults.fired(), 1, "the pointer response was lost once");

            st.refresh().await.expect("refresh");
            let listed = assert_listed_once(&st, &rows).await;
            assert_eq!(listed.len(), 1, "one merged superfile: {listed:?}");
        }
    }

    /// Vector dimension of the multi-cell fixture.
    const MULTI_CELL_DIM: usize = 16;
    /// Rows per commit in the multi-cell fixture.
    const MULTI_CELL_ROWS: usize = 8;
    /// Commits per batch of the multi-cell fixture: enough inputs for a job.
    const MULTI_CELL_COMMITS: usize = 4;

    /// A user table shaped like the one that failed in production: a `title`
    /// column beside an Sq16 `emb` column. Every commit writes one multi-cell
    /// superfile, so its merges take the multi-cell path, which refuses a
    /// repeated row id (the check needs a scalar column besides the id).
    fn make_multi_cell_st(dir: &TempDir) -> Supertable {
        let emb_item = Arc::new(Field::new("item", DataType::Float32, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("title", DataType::LargeUtf8, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(emb_item, MULTI_CELL_DIM as i32),
                false,
            ),
        ]));
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let opts = SupertableOptions::new(
            schema,
            vec![FtsConfig::new("title")],
            vec![VectorConfig {
                column: "emb".into(),
                dim: MULTI_CELL_DIM,
                rot_seed: 7,
                metric: Metric::Cosine,
                rerank_codec: RerankCodec::Sq16,
                provided_centroids: None,
            }],
        )
        .expect("options with a multi-cell vector column")
        .with_writer_pool(Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .expect("1-thread writer pool"),
        ))
        .with_storage(storage);
        Supertable::create(opts).expect("create supertable")
    }

    /// `MULTI_CELL_COMMITS` commits of one-hot rows, numbered from `first`.
    fn commit_multi_cell_superfiles(st: &Supertable, first: usize) {
        let schema = st.options().schema.clone();
        let emb_item = Arc::new(Field::new("item", DataType::Float32, true));
        for commit in first..first + MULTI_CELL_COMMITS {
            let mut flat = vec![0.0f32; MULTI_CELL_ROWS * MULTI_CELL_DIM];
            let titles: Vec<String> = (0..MULTI_CELL_ROWS)
                .map(|row| {
                    flat[row * MULTI_CELL_DIM
                        + (commit * MULTI_CELL_ROWS + row) % MULTI_CELL_DIM] = 1.0;
                    format!("commit {commit} row {row}")
                })
                .collect();
            let emb = FixedSizeListArray::try_new(
                Arc::clone(&emb_item),
                MULTI_CELL_DIM as i32,
                Arc::new(Float32Array::from(flat)),
                None,
            )
            .expect("fixed-size list");
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(LargeStringArray::from(titles)) as ArrayRef,
                    Arc::new(emb) as ArrayRef,
                ],
            )
            .expect("batch");
            let mut w = st.writer().expect("writer");
            w.append(&batch).expect("append");
            w.commit().expect("commit");
        }
    }

    /// Guard: the table's superfiles take the multi-cell merge, the branch
    /// `merge_superfiles` picks for a multi-cell, IVF-mergeable first input.
    async fn assert_merges_are_multi_cell(st: &Supertable) {
        let storage = st.inner().options.storage.clone().expect("storage");
        let first = Arc::clone(&st.inner().manifest.load_full().get_all_superfiles()[0]);
        let (bytes, _) = storage.get(&first.storage_path()).await.expect("bytes");
        let reader = SuperfileReader::open(bytes).expect("open superfile");
        let vec = reader.vec().expect("vector index");
        assert!(vec.is_multi_cell(), "guard: a multi-cell superfile");
        assert!(
            vec.vector_columns_config()
                .next()
                .is_some_and(|c| c.rerank_codec.is_ivf_mergeable()),
            "guard: an IVF-mergeable codec"
        );
    }

    /// Merge any two small superfiles of the multi-cell fixture.
    fn multi_cell_compact_cfg() -> CompactionSettings {
        CompactionSettings {
            min_superfiles_for_merge: 2,
            ..small_compact_cfg()
        }
    }

    #[test]
    fn optimize_repairs_a_multi_cell_superfile_listed_twice() {
        // The production case: an append committed twice left one multi-cell
        // superfile listed at two birth versions, both not yet drained.
        //  - the drain takes the superfile once, so the vector index holds its
        //    rows once. The table's own options drain one superfile per batch,
        //    so the drain's per-batch row dedupe can't catch the copy.
        //  - compaction plans the superfile once, so the multi-cell merge sees
        //    each row id once instead of failing on "duplicate stable_id".
        //  - the commit removes it by id, which drops both copies.
        // One optimize leaves each superfile and each row listed once, in the
        // table and in its vector index, and a search returns each row once.
        let dir = TempDir::new().expect("tempdir");
        let st = make_multi_cell_st(&dir);
        commit_multi_cell_superfiles(&st, 0);
        st.block_on_query(assert_merges_are_multi_cell(&st));
        let twice = st.block_on_query(list_first_superfile_twice(&st));

        let before = st.block_on_query(listed_row_ids(&st));
        let rows: HashSet<i128> = before.iter().copied().collect();
        assert!(
            rows.len() < before.len(),
            "guard: some rows are listed twice"
        );

        st.optimize(&OptimizeOptions::compact(multi_cell_compact_cfg()))
            .expect("optimize a table listing a multi-cell superfile twice");

        let listed = st.block_on_query(assert_listed_once(&st, &rows));
        assert!(
            !listed.contains(&twice),
            "the merge replaced the superfile listed twice: {listed:?}"
        );
        let hidden = st
            .reader()
            .expect("reader")
            .vector_index_table()
            .expect("vector index")
            .clone();
        let indexed = st.block_on_query(listed_row_ids(&hidden));
        let indexed_once: HashSet<i128> = indexed.iter().copied().collect();
        assert_eq!(
            indexed.len(),
            indexed_once.len(),
            "the vector index holds each row once"
        );
        assert_eq!(indexed_once, rows, "the vector index holds every row");

        let mut query = vec![0.0f32; MULTI_CELL_DIM];
        query[0] = 1.0;
        let batches = st
            .vector_search(
                "emb",
                &query,
                rows.len(),
                VectorSearchOptions::new(),
                None,
                None,
            )
            .expect("vector search");
        let mut hits = Vec::new();
        for batch in &batches {
            let ids = batch
                .column(batch.schema().index_of("_id").expect("_id"))
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("decimal ids");
            hits.extend(ids.values().iter().copied());
        }
        let distinct_hits: HashSet<i128> = hits.iter().copied().collect();
        assert_eq!(
            hits.len(),
            distinct_hits.len(),
            "each row is returned once: {hits:?}"
        );
    }

    #[test]
    fn compact_repairs_a_multi_cell_superfile_listed_twice_across_the_drain_watermark() {
        // The second copy of a superfile can land after a drain, so the two
        // copies sit on opposite sides of the drain watermark.
        //  - compaction splits candidates into drained and undrained groups.
        //  - planning each id once, before that split, keeps the earlier copy
        //    in the drained group and drops the later one.
        //  - without it, the drained job removes both copies by id and the
        //    undrained job then fails on an input that is gone.
        // One compaction leaves each superfile and each row listed once.
        let dir = TempDir::new().expect("tempdir");
        let st = make_multi_cell_st(&dir);
        commit_multi_cell_superfiles(&st, 0);
        st.drain_vectors_to_cells_sync().expect("drain");
        commit_multi_cell_superfiles(&st, MULTI_CELL_COMMITS);
        st.block_on_query(assert_merges_are_multi_cell(&st));
        let twice = st.block_on_query(list_first_superfile_twice(&st));

        let before = st.block_on_query(listed_row_ids(&st));
        let rows: HashSet<i128> = before.iter().copied().collect();
        assert!(
            rows.len() < before.len(),
            "guard: some rows are listed twice"
        );

        st.compact(&multi_cell_compact_cfg())
            .expect("compact copies on both sides of the drain watermark");

        let listed = st.block_on_query(assert_listed_once(&st, &rows));
        assert!(
            !listed.contains(&twice),
            "the merge replaced the superfile listed twice: {listed:?}"
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

        let before = st.reader().expect("reader");
        let before_manifest_id = before.manifest_id();
        let before_n_superfiles = before.n_superfiles();
        let input_ids: HashSet<Uuid> = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        let expected_birth_version = before
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.birth_version)
            .min()
            .expect("at least one superfile before compaction");
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

        let after = st.reader().expect("reader");
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
        assert_eq!(
            sfs[0].birth_version, expected_birth_version,
            "compaction must preserve the oldest input birth version"
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
        let before_n = st.reader().expect("reader").n_superfiles();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        assert_eq!(
            st.manifest_id(),
            before_manifest_id,
            "manifest_id must not change: a single superfile cannot form a merge job"
        );
        assert_eq!(st.reader().expect("reader").n_superfiles(), before_n);
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
        assert_eq!(st.reader().expect("reader").n_superfiles(), 2);
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
        let reader_before = st.reader().expect("reader");
        let before_n = reader_before.n_superfiles();
        let before_manifest_id = reader_before.manifest_id();

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let reader_after = st.reader().expect("reader");

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

    /// After a compaction large enough that the merged blob stores its
    /// documents in an order of its own, a search must still name the
    /// row that actually carries the term.
    ///
    /// The existing compaction search test runs twenty documents, far
    /// below the size at which an order is chosen, so it cannot reach
    /// this path at all. Here every document carries a token unique to
    /// it, so a returned row can be checked against the text it should
    /// hold: an untranslated id would come back in range, with a real
    /// score, naming the wrong document.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_compacted_table_names_the_right_rows_when_the_blob_reorders() {
        // Comfortably past the threshold below which arrival order is kept.
        const BATCHES: usize = 60;
        const PER_BATCH: usize = 80;
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        let mut expected: Vec<String> = Vec::with_capacity(BATCHES * PER_BATCH);
        for b in 0..BATCHES {
            let titles: Vec<String> = (0..PER_BATCH)
                .map(|i| {
                    let n = b * PER_BATCH + i;
                    format!("uq{n} shared t{} t{}", n % 37, n % 53)
                })
                .collect();
            expected.extend(titles.iter().cloned());
            let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            commit_titles(&st, &refs);
        }

        let before = st.manifest_id();
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");
        assert!(
            st.manifest_id() > before,
            "compact must have run; adjust small_compact_cfg() if needed"
        );

        // Spread over the corpus so the check does not depend on where a
        // document happened to land.
        for &n in &[0usize, 1, 977, 2500, 4095, 4096, 4799] {
            let want = expected[n].clone();
            let token = format!("uq{n}");

            // Ranked: the row a score is attached to must be the row
            // holding the token.
            let batches = st
                .bm25_search(
                    "title",
                    &token,
                    5,
                    Bm25SearchOptions::new()
                        .with_mode(BoolMode::And)
                        .with_stats(Bm25Stats::Global),
                    Some(&["title"]),
                )
                .unwrap_or_else(|e| panic!("bm25_search for {token}: {e}"));
            assert_eq!(
                titles_of(&batches),
                vec![want.clone()],
                "bm25_search for {token} named the wrong row"
            );

            // Unranked: the same, through the walk that returns bare ids.
            let batches = st
                .token_match("title", &token, BoolMode::And, Some(&["title"]))
                .unwrap_or_else(|e| panic!("token_match for {token}: {e}"));
            assert_eq!(
                titles_of(&batches),
                vec![want],
                "token_match for {token} named the wrong row"
            );
        }

        // A term every document carries still matches all of them, so the
        // translation has not dropped or duplicated anything.
        let n_shared: usize = st
            .token_match("title", "shared", BoolMode::And, None)
            .expect("token_match shared")
            .iter()
            .map(|b| b.num_rows())
            .sum();
        assert_eq!(n_shared, BATCHES * PER_BATCH, "every document carries it");
    }

    /// A `WHERE` on an indexed column pushes a row-keyed allow-set into
    /// the kernel, which walks the blob's own ids. On a reordered blob
    /// the two are different spaces, so the set has to be consulted with
    /// the row a blob id stands for and not with the id itself.
    ///
    /// The failure this guards is quiet rather than wrong: the predicate
    /// is re-applied after the search, so a mismatched set returns an
    /// empty result. In a hybrid query the full-text leg simply drops
    /// out and the ranking degrades with nothing to show for it.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scoped_search_after_a_reordering_compaction_finds_the_row() {
        const BATCHES: usize = 60;
        const PER_BATCH: usize = 80;
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let mut expected: Vec<String> = Vec::new();
        for b in 0..BATCHES {
            let titles: Vec<String> = (0..PER_BATCH)
                .map(|i| {
                    let n = b * PER_BATCH + i;
                    format!("uq{n} shared t{} t{}", n % 37, n % 53)
                })
                .collect();
            expected.extend(titles.iter().cloned());
            let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
            commit_titles(&st, &refs);
        }
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        for n in [0usize, 7, 977, 2500, 4799] {
            let sql = format!(
                "SELECT title FROM bm25_search('title', 'shared', 10) WHERE title = '{}'",
                expected[n]
            );
            let got = st.reader().expect("reader").query_sql(&sql).expect("sql");
            assert_eq!(titles_of(&got), vec![expected[n].clone()], "doc {n}");
        }
    }

    /// Compacting an already-reordered superfile again must keep every
    /// document with its own postings.
    ///
    /// This is ordinary operation, not an edge case: a reordered output
    /// is usually below the target size, so the next pass picks it up
    /// again. The merge reads an input's postings and stored lengths by
    /// the input's own doc ids while its tombstones and rows are keyed by
    /// row, and on a reordered input those are different numbers. Reading
    /// one as the other files a posting under a different document and
    /// writes the result to storage.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_second_compaction_of_a_reordered_superfile_keeps_the_rows() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let mut expected: Vec<String> = Vec::new();
        let mut commit = |st: &Supertable, from: usize, to: usize| {
            for b in from..to {
                let titles: Vec<String> = (0..80)
                    .map(|i| {
                        let n = b * 80 + i;
                        format!("uq{n} shared t{} t{}", n % 37, n % 53)
                    })
                    .collect();
                expected.extend(titles.iter().cloned());
                let refs: Vec<&str> = titles.iter().map(String::as_str).collect();
                commit_titles(st, &refs);
            }
        };
        commit(&st, 0, 60);
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact 1");

        // The reordered output is below target, so the next compaction
        // merges it again.
        let deleted = [3usize, 1000, 2222, 4000];
        for n in deleted {
            let title = format!("uq{n} shared t{} t{}", n % 37, n % 53);
            st.delete(col("title").eq(lit(title))).expect("delete");
        }
        commit(&st, 60, 75);
        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact 2");

        for n in (0..expected.len()).step_by(97) {
            let want = match deleted.contains(&n) {
                true => vec![],
                false => vec![expected[n].clone()],
            };
            let got = st
                .token_match("title", &format!("uq{n}"), BoolMode::And, Some(&["title"]))
                .expect("token_match");
            assert_eq!(titles_of(&got), want, "doc {n}");
        }
        for n in deleted {
            let got = st
                .token_match("title", &format!("uq{n}"), BoolMode::And, Some(&["title"]))
                .expect("token_match");
            assert!(titles_of(&got).is_empty(), "deleted doc {n} still matches");
        }
    }

    /// The `title` column of every row in a result, in order.
    fn titles_of(batches: &[RecordBatch]) -> Vec<String> {
        let mut out = Vec::new();
        for b in batches {
            let col = b.column_by_name("title").expect("title projected");
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("title is LargeUtf8");
            for i in 0..b.num_rows() {
                out.push(arr.value(i).to_string());
            }
        }
        out
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

        let r = st.reader().expect("reader");
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
        let after_first_n = st.reader().expect("reader").n_superfiles();

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
        assert_eq!(st.reader().expect("reader").n_superfiles(), after_first_n);
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
            st.reader().expect("reader").n_docs_total(),
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
        let r = st.reader().expect("reader");
        assert_eq!(r.n_docs_total(), 40, "all docs must be preserved");
        assert!(
            r.n_superfiles() < 8,
            "overall superfile count must have decreased from original 20"
        );

        // ManifestSnapshot consistency: per-entry doc counts sum to 40.
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
            .expect("reader")
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

        let merged_uri = st.reader().expect("reader").manifest().superfiles[0].uri;
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
        env::var(key)
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
            st.bm25_search(
                "title",
                query,
                10,
                Bm25SearchOptions::new()
                    .with_mode(BoolMode::Or)
                    .with_stats(Bm25Stats::Global),
                None,
            )
            .expect("bm25_search warmup");
        }
        let mut samples = Vec::with_capacity(measured_iters);
        for _ in 0..measured_iters {
            let start = Instant::now();
            st.bm25_search(
                "title",
                query,
                10,
                Bm25SearchOptions::new()
                    .with_mode(BoolMode::Or)
                    .with_stats(Bm25Stats::Global),
                None,
            )
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

        let n_before = st.reader().expect("reader").n_superfiles();
        let docs_before = st.reader().expect("reader").n_docs_total();
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

        let n_after = st.reader().expect("reader").n_superfiles();
        assert!(n_after < n_before, "compact should reduce superfile count");

        // No old-file double-counting: doc/hit counts must be identical.
        assert_eq!(st.reader().expect("reader").n_docs_total(), docs_before);
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

        let before_n_superfiles = st.reader().expect("reader").n_superfiles();
        let before_data_objects = storage
            .list_with_prefix_metadata("data")
            .await
            .expect("list data/ before compact")
            .len();
        assert_eq!(before_data_objects, before_n_superfiles);

        st.compact_async(&small_compact_cfg())
            .await
            .expect("compact");

        let after_n_superfiles = st.reader().expect("reader").n_superfiles();
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
            .gc(DEFAULT_GC_SAFETY_GAP)
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
        let zero_gap_report = st.gc(Duration::ZERO).expect("gc with zero safety gap");
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
            .expect("reader")
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        assert_eq!(stranded_ids.len(), 2);

        // Simulate a compaction that sealed its inputs then died
        // before committing the merge.
        let wal_store = wal_store_for(&st);
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
            .expect("reader")
            .manifest()
            .superfiles
            .iter()
            .map(|s| s.superfile_id)
            .collect();
        for id in &stranded_ids {
            assert!(remaining_ids.contains(id));
        }
    }

    // ---- merge admission -------------------------------------------------

    /// A width far above any plausible job count.
    const ABSURD_WIDTH: usize = 4096;

    /// The compactor must re-stamp before a writer may steal, at whatever
    /// threshold is in force.
    ///
    /// Both sides read `writer_steal_timeout`, so a seal the writer could take
    /// over is never one this skips as too young. Reading different numbers —
    /// a hard-coded default here against a configured one there — leaves the
    /// gap between them as a window where a delete lands a bit on a superfile
    /// about to be removed, and the deletion is lost. Checked across a range
    /// of thresholds, because the one in force is configurable and a predicate
    /// that only agrees with the default would say nothing about the rest.
    #[test]
    fn a_re_stamp_never_skips_a_seal_a_writer_could_steal() {
        /// Thresholds either side of the shipped default, including ones below
        /// the re-stamp margin.
        const THRESHOLDS_SECS: [u64; 5] = [10, 45, 90, 120, 600];

        let now = Utc::now();
        for secs in THRESHOLDS_SECS {
            let steals_after = Duration::from_secs(secs);
            let placed_at =
                |age: Duration| now - chrono::Duration::from_std(age).expect("representable");

            assert!(
                seal_may_have_been_stolen(placed_at(steals_after), now, steals_after),
                "a seal old enough for a writer to steal must be re-stamped ({secs}s)"
            );
            assert!(
                seal_may_have_been_stolen(
                    placed_at(steals_after.saturating_sub(SEAL_RESTAMP_MARGIN)),
                    now,
                    steals_after
                ),
                "the margin's worth of life left is already at risk ({secs}s)"
            );
            // Below the margin there is no safe window left to skip in, and
            // the floor at zero makes every seal a candidate — conservative,
            // and the only correct answer when a writer steals sooner than a
            // commit can be relied on to finish.
            let expected = steals_after <= SEAL_RESTAMP_MARGIN;
            assert_eq!(
                seal_may_have_been_stolen(placed_at(Duration::ZERO), now, steals_after),
                expected,
                "a seal just placed, against a {secs}s threshold"
            );
        }
    }

    /// One merge starts whatever the host says. Refusing it would stall
    /// exactly the tables that most need compacting, and there is nothing
    /// running yet for a memory reading to be about.
    #[test]
    fn a_merge_starts_even_when_the_host_is_tight() {
        assert!(admits_another_merge(0, ABSURD_WIDTH, 4, false));
        assert!(
            admits_another_merge(0, 0, 1, false),
            "a zero width still runs one merge at a time"
        );
    }

    /// Past the first, the host decides. This is the half that was dead while
    /// admission ran before any merge had started: every reading saw an idle
    /// machine, so a tight host admitted the full width anyway.
    #[test]
    fn a_tight_host_admits_no_further_merge() {
        assert!(admits_another_merge(1, 4, 3, true));
        assert!(!admits_another_merge(1, 4, 3, false));
    }

    /// The width knob is a hard cap: the runner may admit fewer when the host
    /// is tight, never more than asked for.
    #[test]
    fn merges_in_flight_never_exceed_the_width_knob() {
        assert!(!admits_another_merge(3, 3, 61, true));
        assert!(!admits_another_merge(1, 1, 5, true));
    }

    /// A pass never starts work it did not plan, which is why no separate
    /// ceiling on the width is needed.
    #[test]
    fn no_merge_starts_without_planned_work() {
        assert!(!admits_another_merge(0, ABSURD_WIDTH, 0, true));
        assert!(!admits_another_merge(2, ABSURD_WIDTH, 0, true));
    }

    /// The reserve the cases below encode, written out rather than read from
    /// [`MERGE_MEMORY_RESERVE_PERCENT`]: a test that takes its inputs from the
    /// value it is checking moves with a wrong edit instead of failing on it.
    const SHIPPED_RESERVE_PERCENT: u64 = 40;
    /// One point under the reserve — the widest share that must be denied.
    const UNDER_RESERVE_PERCENT: u64 = 39;
    /// A host with nothing else on it.
    const IDLE_HOST_PERCENT: u64 = 100;
    /// A host with nothing left.
    const EXHAUSTED_HOST_PERCENT: u64 = 0;

    /// Free share of a synthetic 100 GiB host, as the two readings the
    /// decision takes. Fixed inputs: the running machine's own memory is not
    /// an input here, so the test cannot flake when the host itself sits near
    /// the reserve, and nothing re-derives the predicate's arithmetic.
    fn host_at(free_percent: u64) -> (Option<u64>, Option<u64>) {
        const SYNTHETIC_HOST_BYTES: u64 = 100 * 1024 * 1024 * 1024;
        (
            Some(SYNTHETIC_HOST_BYTES / 100 * free_percent),
            Some(SYNTHETIC_HOST_BYTES),
        )
    }

    /// The admission decision follows the host's free share against the
    /// reserve, and admits at exactly the reserve rather than only above it.
    /// The 40/39 pair pins both the constant and the comparison: lower the
    /// reserve and the first assertion fails, tighten `>=` to `>` and the
    /// second does.
    ///
    /// Nothing here depends on what a merge costs per byte, which is a
    /// function of term cardinality, posting density and which indexes a table
    /// carries — none of it knowable from a job's byte count.
    #[test]
    fn room_for_another_merge_is_judged_against_the_host() {
        assert_eq!(
            MERGE_MEMORY_RESERVE_PERCENT, SHIPPED_RESERVE_PERCENT,
            "the shares below are written against a {SHIPPED_RESERVE_PERCENT}% reserve"
        );

        let (available, total) = host_at(SHIPPED_RESERVE_PERCENT);
        assert!(
            has_room_for_another_merge(available, total),
            "a host exactly at the reserve still admits"
        );
        let (available, total) = host_at(UNDER_RESERVE_PERCENT);
        assert!(
            !has_room_for_another_merge(available, total),
            "a host one point under the reserve does not"
        );
        let (available, total) = host_at(IDLE_HOST_PERCENT);
        assert!(has_room_for_another_merge(available, total), "an idle host");
        let (available, total) = host_at(EXHAUSTED_HOST_PERCENT);
        assert!(
            !has_room_for_another_merge(available, total),
            "an exhausted host"
        );
    }

    /// A host whose memory cannot be read does not throttle: there is nothing
    /// to throttle against, and a derived width is already 1 there, so the
    /// only way to reach this path is a width an operator set explicitly.
    /// A zero total is a nonsense reading rather than an absent one, and is
    /// the one case that denies.
    #[test]
    fn an_unreadable_host_does_not_throttle() {
        const SOME_BYTES: u64 = 1024 * 1024 * 1024;
        assert!(has_room_for_another_merge(None, None));
        assert!(has_room_for_another_merge(None, Some(SOME_BYTES)));
        assert!(has_room_for_another_merge(Some(SOME_BYTES), None));
        assert!(!has_room_for_another_merge(Some(SOME_BYTES), Some(0)));
    }

    // ---- concurrent jobs ------------------------------------------------

    /// Merges in flight when a test exercises the concurrent path. Four is
    /// enough to hold several at once without needing a large fixture.
    const TEST_CONCURRENT_JOBS: usize = 4;

    /// Build a table whose superfiles the selector packs into several jobs,
    /// then compact it at `concurrency` and report what the pass produced:
    /// total docs, the per-superfile doc counts, and how many manifest
    /// generations the pass burned.
    async fn compact_a_fragmented_table(concurrency: usize) -> (u64, Vec<u64>, u64) {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);

        // Each superfile must be big enough that a handful overflow the 1 MiB
        // target, so the selector emits more than one job and the concurrent
        // path is exercised rather than degenerating to a single merge.
        let commit_bulk = |titles: &[&str]| {
            let mut w = st.writer().expect("writer");
            for _ in 0..4096 {
                w.append(&build_title_batch(titles)).expect("append");
            }
            w.commit().expect("commit");
        };
        // Thirty-two superfiles: the packer fills several 1 MiB jobs from them,
        // which is what makes running them concurrently meaningful.
        for round in 0..2 {
            for term in [
                "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
                "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
            ] {
                commit_bulk(&[
                    &format!("{term} first {round}"),
                    &format!("{term} second {round}"),
                ]);
            }
        }

        let cfg = CompactionSettings {
            max_concurrent_jobs: Some(concurrency),
            ..small_compact_cfg()
        };
        let before = st.manifest_id();
        st.compact_async(&cfg).await.expect("compact");
        let generations = st.manifest_id() - before;

        let reader = st.reader().expect("reader");
        let total = reader.n_docs_total();
        let mut per_superfile: Vec<u64> = reader
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.n_docs)
            .collect();
        per_superfile.sort_unstable();
        (total, per_superfile, generations)
    }

    /// The gate on running jobs concurrently: a wider pass must land the same
    /// table. Same doc count, same superfile shape — the plan is identical, so
    /// only the order the merges run in changed.
    ///
    /// Commit count is bounded rather than pinned. A merge commits as soon as
    /// it is ready and takes along whatever else has finished, so how many
    /// CASes a pass spends depends on how its merges interleave.
    #[tokio::test(flavor = "multi_thread")]
    async fn concurrent_jobs_land_the_same_table_as_a_serial_pass() {
        let (serial_docs, serial_shape, serial_generations) = compact_a_fragmented_table(1).await;
        let (concurrent_docs, concurrent_shape, concurrent_generations) =
            compact_a_fragmented_table(TEST_CONCURRENT_JOBS).await;

        assert!(
            serial_generations >= 2,
            "fixture must plan more than one job to be a real test of batching, \
             got {serial_generations} generations"
        );
        assert_eq!(
            serial_docs, concurrent_docs,
            "a concurrent pass must preserve every doc"
        );
        assert_eq!(
            serial_shape, concurrent_shape,
            "a concurrent pass must produce the same superfiles as a serial one"
        );
        // Commit count is not asserted beyond this bound. A merge commits as
        // soon as it is ready, taking along whatever else has finished, so how
        // many CASes a pass spends depends on how its merges interleave. What
        // must hold either way: every commit lands at least one job, so a pass
        // never spends more CASes than it planned jobs, and each job produces
        // one superfile.
        assert!(
            concurrent_generations <= concurrent_shape.len() as u64,
            "{concurrent_generations} commits for {} jobs",
            concurrent_shape.len()
        );
        assert!(
            concurrent_generations <= serial_generations,
            "committing early must not cost more CASes than a serial pass: \
             {concurrent_generations} vs {serial_generations}"
        );
    }

    /// One job failing must not cost the others their merges. The failure
    /// here is an input that vanished between planning and preparing, which
    /// `prepare_compaction_job` reports as `SuperfileNotFound`.
    ///
    /// The surviving job still commits, and the error still surfaces.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_failed_job_still_lets_the_others_commit() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);
        assert_eq!(live.len(), 4, "fixture");

        // Two jobs in flight: the first names a superfile that is not in the
        // manifest, the second is real.
        let doomed = CompactionJob {
            partition_key: Vec::new(),
            inputs: vec![Uuid::new_v4(), live[0]],
            estimated_output_bytes: 0,
        };
        let good = CompactionJob {
            partition_key: Vec::new(),
            inputs: vec![live[1], live[2]],
            estimated_output_bytes: 0,
        };

        let err = st
            .run_compaction_jobs(vec![doomed, good], DEFAULT_STALE_SEAL_TIMEOUT, 2)
            .await
            .expect_err("the doomed job must surface its error");
        assert!(
            matches!(err, CompactionError::SuperfileNotFound(_)),
            "unexpected error: {err:?}"
        );

        // The healthy job committed anyway: its two inputs are gone, replaced
        // by one merged superfile, and the untouched fourth is still listed.
        let after: Vec<Uuid> = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert!(
            !after.contains(&live[1]) && !after.contains(&live[2]),
            "the healthy job's inputs must have been merged away"
        );
        assert!(
            after.contains(&live[0]) && after.contains(&live[3]),
            "the failed job must not have touched anything"
        );
    }

    /// Superfile ids the table lists, in manifest order.
    fn listed_ids(st: &Supertable) -> Vec<Uuid> {
        st.reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect()
    }

    /// A `WalStore` over this table's storage, for tests that read or poke
    /// tombstone sidecars directly.
    fn wal_store_for(st: &Supertable) -> WalStore {
        WalStore::new(
            st.inner()
                .manifest
                .load_full()
                .options
                .storage
                .clone()
                .expect("storage-backed table"),
        )
    }

    /// Age every seal these jobs hold past the staleness threshold, as a merge
    /// running longer than the window does.
    ///
    /// Both halves of the clock move together. `prepare` writes one instant
    /// into the sidecar and into the job it returns, so a merge old enough for
    /// a writer to steal from is one whose own record of when it sealed is
    /// equally old; aging only the sidecar would describe a state the code
    /// cannot reach. Aging rewrites the sidecar, which moves its etag, so the
    /// job's held etag moves with it: real elapsed time leaves the etag alone,
    /// and a job holding a stale one would look stolen and prove nothing.
    async fn age_seals_past_stale(wal_store: &WalStore, jobs: &mut [PreparedJob]) {
        /// Comfortably past the threshold, so no test rides the boundary.
        const PAST_STALE_MS: i64 = tombstones_admin::DEFAULT_STALE_SEAL_TIMEOUT_MS as i64 + 60_000;

        let aged = Utc::now() - chrono::Duration::milliseconds(PAST_STALE_MS);
        for prepared in jobs.iter_mut() {
            prepared.sealed_at = aged;
            for input in prepared.sealed.iter_mut() {
                let (mut sidecar, etag) = wal_store
                    .get_tombstones(input.superfile_id)
                    .await
                    .expect("get sidecar")
                    .expect("prepare sealed every input");
                sidecar.seal.as_mut().expect("sealed by prepare").sealed_at = aged;
                input.etag = wal_store
                    .put_tombstones(input.superfile_id, Some(&etag), &sidecar)
                    .await
                    .expect("age the seal");
            }
        }
    }

    /// A tombstone that lands while a merge's seal has gone stale survives the
    /// commit that removes its superfile.
    ///
    /// A seal expires after `DEFAULT_STALE_SEAL_TIMEOUT_MS`, and a writer that
    /// finds one expired treats its owner as dead, steals it and lands its bit
    /// anyway. A merge holds its inputs sealed from prepare until its commit, which on a large table runs longer than that. Committing such
    /// an input away would drop the bit, because the merged superfile was
    /// built from the bitmap the merge read — so the commit re-stamps each
    /// seal against the etag it holds, and a job that lost one leaves the
    /// batch. The delete stands and the merge is redone later.
    ///
    /// The other job is untouched and must still commit: one stolen seal costs
    /// its own job, not the pass.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_delete_that_steals_a_stale_seal_survives_the_commit() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let doomed = "delta first";
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);
        assert_eq!(live.len(), 4, "fixture");

        // Stage two jobs: every input sealed, every merge done, nothing committed.
        let mut prepared_jobs = Vec::new();
        for pair in [[live[0], live[1]], [live[2], live[3]]] {
            prepared_jobs.push(
                st.prepare_compaction_job(
                    CompactionJob {
                        partition_key: Vec::new(),
                        inputs: pair.to_vec(),
                        estimated_output_bytes: 0,
                    },
                    DEFAULT_STALE_SEAL_TIMEOUT,
                )
                .await
                .expect("prepare"),
            );
        }

        // Age every seal past the stale threshold. The merges are still running —
        // only their seals now look abandoned to a writer, which is what a
        // merge longer than the timeout looks like from the delete path.
        //
        // Aging means rewriting the sidecar, which moves its etag, so the
        // held etags are re-synced below. Real elapsed time moves
        // `sealed_at` past the threshold without touching the object, and it
        // is that state — stale seal, etag still the compactor's — the test
        // has to reproduce. Skipping the re-sync would make every input look
        // stolen and prove nothing.
        let wal_store = wal_store_for(&st);
        age_seals_past_stale(&wal_store, &mut prepared_jobs).await;

        // The delete finds a stale seal, steals it, and lands its bit on an
        // input the commit is about to remove.
        let deleting = st.clone();
        let title = doomed.to_string();
        let stats = task::spawn_blocking(move || deleting.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");
        assert_eq!(
            stats.n_tombstoned(),
            1,
            "the delete must steal the stale seal and land its bit"
        );

        // The batch commits, removing the inputs it merged before that landed.
        st.commit_compaction_batch(prepared_jobs)
            .await
            .expect("the batch commits");

        // The row must still be gone. A second delete resolves against live
        // rows, so a match here means the deleted row came back.
        let deleting = st.clone();
        let title = doomed.to_string();
        let again = task::spawn_blocking(move || deleting.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");
        assert_eq!(
            again.matched(),
            0,
            "the deleted row must not survive the commit"
        );

        // The job holding the stolen seal left the batch, so its inputs are
        // still listed; the other job committed as normal.
        let after: Vec<Uuid> = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert!(
            after.contains(&live[2]) && after.contains(&live[3]),
            "the job whose seal was stolen must not have committed"
        );
        assert!(
            !after.contains(&live[0]) && !after.contains(&live[1]),
            "the untouched job must still have merged"
        );
    }

    /// Prepare one job over the whole table, for the fence tests below.
    async fn one_job_over(st: &Supertable, live: &[Uuid]) -> Vec<PreparedJob> {
        vec![
            st.prepare_compaction_job(
                CompactionJob {
                    partition_key: Vec::new(),
                    inputs: live.to_vec(),
                    estimated_output_bytes: 0,
                },
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("prepare"),
        ]
    }

    /// A seal taken over at the fence costs its own job, not the batch.
    ///
    /// The fence runs after every merged superfile is uploaded, so a failure
    /// there is expensive: the catch-all arm would unseal the batch, fail the
    /// commit, and `run_compaction_jobs` would treat that as fatal and end the
    /// pass — throwing away every finished, already-uploaded merge in it for
    /// one stolen seal. `InputsChanged` is a lost race with nothing published,
    /// so it takes the retry arm instead: the attempt re-resolves, its early
    /// re-stamp drops the job whose sidecar moved, and the rest commit.
    ///
    /// The failure is injected. Reaching it for real needs a delete to land
    /// between the early re-stamp and the pointer PUT, which is minutes apart
    /// on a multi-gigabyte upload and microseconds apart here.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_seal_taken_over_at_the_fence_costs_only_its_own_job() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live = listed_ids(&st);
        assert_eq!(live.len(), 4, "fixture");
        let before_docs = st.reader().expect("reader").n_docs_total();

        let mut batch = Vec::new();
        for pair in [[live[0], live[1]], [live[2], live[3]]] {
            batch.push(
                st.prepare_compaction_job(
                    CompactionJob {
                        partition_key: Vec::new(),
                        inputs: pair.to_vec(),
                        estimated_output_bytes: 0,
                    },
                    DEFAULT_STALE_SEAL_TIMEOUT,
                )
                .await
                .expect("prepare"),
            );
        }

        // The first attempt's fence fails, as a seal taken over mid-upload
        // would. Nothing is published by that attempt.
        FENCE_FAILS_ONCE.store(true, Ordering::SeqCst);
        st.commit_compaction_batch(batch)
            .await
            .expect("a seal taken over at the fence must not fail the commit");
        assert!(
            !FENCE_FAILS_ONCE.load(Ordering::SeqCst),
            "the injected failure must have been consumed"
        );

        // The retry committed: both jobs' inputs are merged away, since the
        // injected failure named a seal nothing had actually taken over.
        let after = listed_ids(&st);
        assert!(
            after.len() < live.len(),
            "the retry must have committed, got {after:?}"
        );
        assert_eq!(
            st.reader().expect("reader").n_docs_total(),
            before_docs,
            "no rows may be lost by the retry"
        );
    }

    /// The fence passes a commit whose inputs nobody touched, re-stamping the
    /// seals so they outlive the pointer PUT that follows.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_commit_fence_passes_an_untouched_sidecar() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live = listed_ids(&st);
        let mut batch = one_job_over(&st, &live).await;
        let wal_store = wal_store_for(&st);

        // Old enough that the fence re-stamps rather than skipping, which is
        // the state a long upload leaves the seals in.
        age_seals_past_stale(&wal_store, &mut batch).await;
        let mut fence = SealFence {
            wal_store: &wal_store,
            batch: &mut batch,
        };
        fence
            .check()
            .await
            .expect("a sidecar nobody touched must pass the fence");
    }

    /// The fence refuses a commit whose input sidecar moved under its seal.
    ///
    /// `try_commit_attempt` writes every merged superfile before its pointer
    /// PUT, so a re-stamp proved only before that work leaves a window as wide
    /// as the upload is slow: the seal expires mid-upload, a delete takes it
    /// over and marks its row, and the pointer PUT removes the input anyway.
    /// The fence is checked with only the pointer PUT left to outlive a
    /// freshly stamped seal.
    ///
    /// Driven directly rather than through a slowed upload: making the expiry
    /// fall strictly between the early re-stamp and the pointer PUT needs a
    /// fault-injection point inside `try_commit_attempt` that the harness does
    /// not have. What this pins is the fence's own decision, which is what the
    /// commit path calls at that moment.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_commit_fence_refuses_a_sidecar_that_moved_under_its_seal() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        let doomed = "bravo first";
        for term in ["alpha", "bravo"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live = listed_ids(&st);
        let mut batch = one_job_over(&st, &live).await;
        let wal_store = wal_store_for(&st);

        // The seals age out, as they would inside a long upload, and a delete
        // arriving meanwhile takes one over and marks its row.
        age_seals_past_stale(&wal_store, &mut batch).await;
        let deleting = st.clone();
        let title = doomed.to_string();
        let stats = task::spawn_blocking(move || deleting.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");
        assert_eq!(
            stats.n_tombstoned(),
            1,
            "the delete must take the expired seal over and land its bit"
        );

        // The job still holds the etag from before that bit landed. Committing
        // would remove the input carrying it, so the fence must refuse.
        let mut fence = SealFence {
            wal_store: &wal_store,
            batch: &mut batch,
        };
        let refused = fence.check().await;
        assert!(
            refused.is_err(),
            "a sidecar that moved under the seal must fail the attempt, got {refused:?}"
        );
    }

    /// A job dropped for a stolen seal must leave none of its other inputs
    /// sealed.
    ///
    /// Only the input that lost the CAS has left our hands. The ones
    /// re-stamped before it carry a FRESH window under a compaction that will
    /// never commit, and the ones after it still carry prepare's. Left
    /// behind, they block every delete resolving there and fail the next pass
    /// outright, because `seal` answers `AlreadySealed` without retrying.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_dropped_for_a_stolen_seal_leaves_nothing_sealed() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        // Four inputs in one job, with the stolen seal on the LAST. The three
        // before it are re-stamped first, so their held etags have moved and
        // the unseal below can only clear them if the commit wrote those
        // etags back. A steal on the first input would exercise none of that.
        let doomed = "delta first";
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);
        assert_eq!(live.len(), 4, "fixture");

        let prepared = st
            .prepare_compaction_job(
                CompactionJob {
                    partition_key: Vec::new(),
                    inputs: live.clone(),
                    estimated_output_bytes: 0,
                },
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("prepare");

        let wal_store = wal_store_for(&st);
        let mut prepared_jobs = vec![prepared];
        age_seals_past_stale(&wal_store, &mut prepared_jobs).await;

        // A delete steals one input's seal, which costs the job its commit.
        let deleting = st.clone();
        let title = doomed.to_string();
        task::spawn_blocking(move || deleting.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");

        st.commit_compaction_batch(prepared_jobs)
            .await
            .expect("a batch with nothing left to commit is not an error");

        // Every input must now be unsealed: a seal left behind here is one no
        // compaction will ever clear.
        for id in &live {
            let sealed = wal_store
                .get_tombstones(*id)
                .await
                .expect("get sidecar")
                .and_then(|(sidecar, _)| sidecar.seal);
            assert!(
                sealed.is_none(),
                "superfile {id} is still sealed after its job was dropped"
            );
        }
    }

    /// A compactor whose tombstone view predates a committed delete still
    /// drops the deleted row.
    ///
    /// The sidecar cache refreshes an entry only when its seq differs from the
    /// seq view installed from THIS handle's manifest, so a handle that has
    /// not seen a delete's commit holds a view that predates it. A merge
    /// reading tombstones through that cache would carry the deleted row into
    /// the output and then remove the input holding its bit. The seal is no
    /// help here: the bit landed before the seal, so nothing changes the
    /// sidecar afterwards. What makes it correct is that the merge uses the
    /// bitmaps `seal` read from storage rather than anything cached.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_stale_tombstone_view_does_not_resurrect_a_deleted_row() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let compactor =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");

        let doomed = "alpha first";
        commit_titles(&compactor, &[doomed, "alpha second"]);
        commit_titles(&compactor, &["bravo first", "bravo second"]);
        let live: Vec<Uuid> = compactor
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        assert_eq!(live.len(), 2, "fixture");

        // A second handle deletes the row and commits it. The compactor handle
        // is never refreshed, so its manifest and seq view stay behind.
        let writer_handle =
            Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("open second handle");
        let title = doomed.to_string();
        let stats = task::spawn_blocking(move || writer_handle.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");
        assert_eq!(stats.n_tombstoned(), 1, "the delete landed its bit");

        // The compactor merges both superfiles off its stale view and commits.
        let prepared = compactor
            .prepare_compaction_job(
                CompactionJob {
                    partition_key: Vec::new(),
                    inputs: live.clone(),
                    estimated_output_bytes: 0,
                },
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("prepare");
        compactor
            .commit_compaction_batch(vec![prepared])
            .await
            .expect("commit");

        // Read through a handle that is definitely current.
        let checker =
            Supertable::open(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("open checker");
        let title = doomed.to_string();
        let again = task::spawn_blocking(move || checker.delete(col("title").eq(lit(title))))
            .await
            .expect("delete task")
            .expect("delete");
        assert_eq!(
            again.matched(),
            0,
            "the deleted row must not come back through a stale-view merge"
        );
    }

    /// The refund is what keeps a delete alive across a pass of several merges.
    ///
    /// The delete's targets sit in the LAST merge's inputs, so nothing it does
    /// lands until that merge commits. Each earlier commit removes superfiles
    /// from the manifest, which is the compactor's progress, and that is what
    /// refunds the budget. Lowered to three retries, an unrefunded delete
    /// exhausts after roughly 700 ms of backoff, well before the last merge
    /// publishes at ~1.2 s.
    ///
    /// Every superfile in the table is sealed before the delete starts, so
    /// this also covers the case where a blocked delete has nowhere unsealed
    /// to go at all: it must resolve its row and land the bit on the merged
    /// output, not report success having tombstoned nothing.
    ///
    /// This gates that the refund exists. That it counts a removal rather than
    /// any commit at all is gated separately, by the pipeline's
    /// `an_append_does_not_refund_the_sealed_budget`.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_delete_survives_a_pass_whose_merges_commit_one_by_one() {
        /// Small enough that an unrefunded delete exhausts inside the test.
        const SEALED_RETRY_BUDGET: u32 = 3;
        /// Gap between commits, comfortably under the ~700 ms an
        /// unrefunded delete survives and over nothing else.
        const COMMIT_GAP: Duration = Duration::from_millis(300);

        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st = Supertable::create(
            default_supertable_options()
                .with_storage(storage)
                .with_max_sealed_retries(SEALED_RETRY_BUDGET),
        )
        .expect("create supertable");

        let doomed = "hotel first";
        for term in [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        ] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);
        assert_eq!(live.len(), 8, "fixture");
        let before_docs = st.reader().expect("reader").n_docs_total();

        // Four jobs prepared up front: every superfile in the table is sealed
        // and nothing has committed, so the delete has nowhere unsealed to go.
        let mut prepared_jobs = Vec::new();
        for pair in [
            [live[0], live[1]],
            [live[2], live[3]],
            [live[4], live[5]],
            [live[6], live[7]],
        ] {
            prepared_jobs.push(
                st.prepare_compaction_job(
                    CompactionJob {
                        partition_key: Vec::new(),
                        inputs: pair.to_vec(),
                        estimated_output_bytes: 0,
                    },
                    DEFAULT_STALE_SEAL_TIMEOUT,
                )
                .await
                .expect("prepare"),
            );
        }

        let deleting = st.clone();
        let title = doomed.to_string();
        let delete = task::spawn_blocking(move || {
            deleting
                .delete(col("title").eq(lit(title)))
                .map(|s| (s.matched(), s.n_tombstoned()))
        });

        // Publish the merges one at a time. Only the last frees the target.
        for prepared in prepared_jobs {
            tokio::time::sleep(COMMIT_GAP).await;
            st.commit_compaction_batch(vec![prepared])
                .await
                .expect("the batch commits");
        }

        let (matched, tombstoned) = delete
            .await
            .expect("delete task")
            .expect("earlier merges committing must refund the budget");
        assert_eq!(matched, 1, "the predicate must have resolved its row");
        // A delete that resolved its row and then failed to tombstone it
        // anywhere has silently lost the deletion, so the call returning
        // without an error is not enough.
        assert_eq!(
            tombstoned, 1,
            "the tombstone must land once the last merge publishes"
        );
        assert_eq!(before_docs, st.reader().expect("reader").n_docs_total());
    }

    /// A batch that loses the manifest CAS retries as a batch: the merges are
    /// already done and their outputs already staged, so the second attempt
    /// re-resolves the inputs against the refreshed manifest and commits the
    /// same batch. Nothing is re-merged and nothing is lost.
    ///
    /// The race is a real concurrent writer rather than an injected fault,
    /// because that is the contention the retry loop is built for: a writer
    /// moves the pointer between the batch's manifest read and its CAS.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_batch_commits_alongside_a_racing_writer() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let before_docs = st.reader().expect("reader").n_docs_total();
        let live: Vec<Uuid> = listed_ids(&st);
        assert_eq!(live.len(), 4, "fixture");

        // Two jobs in one batch, racing a writer commit. The batch may lose its
        // pointer CAS to the writer and retry, or win outright; either way
        // both the merge and the append must land.
        let racing = st.clone();
        let writer = task::spawn_blocking(move || {
            commit_titles(&racing, &["echo first", "echo second"]);
        });
        let jobs = vec![
            CompactionJob {
                partition_key: Vec::new(),
                inputs: vec![live[0], live[1]],
                estimated_output_bytes: 0,
            },
            CompactionJob {
                partition_key: Vec::new(),
                inputs: vec![live[2], live[3]],
                estimated_output_bytes: 0,
            },
        ];
        st.run_compaction_jobs(jobs, DEFAULT_STALE_SEAL_TIMEOUT, 2)
            .await
            .expect("the batch must commit despite a racing writer");
        writer.await.expect("writer task");

        st.refresh().await.expect("refresh");
        let listed: Vec<Uuid> = st
            .reader()
            .expect("reader")
            .manifest()
            .get_all_superfiles()
            .iter()
            .map(|e| e.superfile_id)
            .collect();
        for input in &live {
            assert!(
                !listed.contains(input),
                "every merged input must be gone: {input}"
            );
        }
        assert_eq!(
            st.reader().expect("reader").n_docs_total(),
            before_docs + 2,
            "the merge kept its rows and the racing writer's two landed"
        );
    }

    /// The retry path hands a failed attempt's unwritten superfile bytes back
    /// to the jobs that own them, keyed by storage path, so the next attempt
    /// re-PUTs exactly what is still owed and no job re-PUTs a sibling's
    /// bytes. Bytes whose owner was dropped from the batch are orphans for gc
    /// and must not be handed to anyone.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_retry_hands_unwritten_bytes_back_to_the_job_that_owes_them() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);

        let mut batch = Vec::new();
        for pair in [[live[0], live[1]], [live[2], live[3]]] {
            batch.push(
                st.prepare_compaction_job(
                    CompactionJob {
                        partition_key: Vec::new(),
                        inputs: pair.to_vec(),
                        estimated_output_bytes: 0,
                    },
                    DEFAULT_STALE_SEAL_TIMEOUT,
                )
                .await
                .expect("prepare"),
            );
        }

        // Drain what a commit attempt would have taken, then hand it back with
        // one extra entry nobody in the batch owns.
        let outstanding: Vec<(String, Bytes)> = batch
            .iter_mut()
            .flat_map(|p| p.pending_storage_writes.drain(..))
            .collect();
        assert_eq!(outstanding.len(), 2, "one staged superfile per job");
        let owners: Vec<String> = batch
            .iter()
            .map(|p| p.new_entries[0].storage_path())
            .collect();

        let mut handed_back = outstanding.clone();
        handed_back.push(("orphan-of-a-dropped-job".to_string(), Bytes::new()));
        redistribute_pending_writes(&mut batch, handed_back);

        for (i, prepared) in batch.iter().enumerate() {
            let paths: Vec<&str> = prepared
                .pending_storage_writes
                .iter()
                .map(|(path, _)| path.as_str())
                .collect();
            assert_eq!(
                paths,
                vec![owners[i].as_str()],
                "job {i} must get back exactly its own bytes"
            );
        }

        // Nothing is left holding the orphan.
        let total: usize = batch.iter().map(|p| p.pending_storage_writes.len()).sum();
        assert_eq!(total, outstanding.len(), "the orphan must be dropped");

        // Leave no seals behind for the tempdir teardown.
        st.commit_compaction_batch(batch).await.expect("commit");
    }

    /// A job whose inputs another compactor merged away between planning and
    /// commit is dropped from the batch, not treated as a failure: there is
    /// nothing left for it to remove, and its output is an orphan for gc.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_job_whose_inputs_vanished_before_commit_is_dropped_not_failed() {
        let dir = TempDir::new().expect("tempdir");
        let st = make_st(&dir);
        for term in ["alpha", "bravo", "charlie", "delta"] {
            commit_titles(&st, &[&format!("{term} first"), &format!("{term} second")]);
        }
        let live: Vec<Uuid> = listed_ids(&st);

        // Prepare a job over the first two, then merge them away underneath it.
        let staged = st
            .prepare_compaction_job(
                CompactionJob {
                    partition_key: Vec::new(),
                    inputs: vec![live[0], live[1]],
                    estimated_output_bytes: 0,
                },
                DEFAULT_STALE_SEAL_TIMEOUT,
            )
            .await
            .expect("prepare");
        // The racing compactor treats every seal as stale, which is how a
        // second compactor takes over after the first one dies. Here the first
        // one has not died, but from the second's point of view the situation
        // is identical, and it is the only way to reach a committed merge over
        // inputs another job has already staged.
        st.run_compaction_job(
            CompactionJob {
                partition_key: Vec::new(),
                inputs: vec![live[0], live[1]],
                estimated_output_bytes: 0,
            },
            Duration::ZERO,
        )
        .await
        .expect("the racing compactor commits");

        let docs_before = st.reader().expect("reader").n_docs_total();
        st.commit_compaction_batch(vec![staged])
            .await
            .expect("a vanished job is benign, not an error");
        assert_eq!(
            st.reader().expect("reader").n_docs_total(),
            docs_before,
            "the dropped job must not have changed the table"
        );
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

        // The selector packs the 30 small superfiles into several target-sized
        // jobs rather than one oversized merge, and each job writes one output
        // superfile. Count the OUTPUTS, not manifest generations: a pass runs
        // several merges at once and commits whichever have finished in a
        // single CAS, so generations count commits, not jobs. The exact job
        // count tracks
        // per-superfile byte size, which is format-dependent, so assert the
        // invariant — more than one — not a pinned number.
        let commits = st.manifest_id() - manifest_id_before_first_compact;
        assert!(commits >= 1, "compaction must have committed something");

        // All 245760 docs must be visible after compaction.
        let r = st.reader().expect("reader");
        assert_eq!(r.n_docs_total(), 245760, "all docs must be preserved");
        // Fewer than the original 30, and more than one: the inputs
        // consolidated into a handful of target-sized superfiles rather than
        // collapsing into a single oversized one.
        let n_superfiles = r.n_superfiles();
        assert!(
            (2..30).contains(&n_superfiles),
            "30 inputs must consolidate into several (< 30) superfiles, \
             got {n_superfiles}"
        );

        // ManifestSnapshot consistency: per-entry doc counts sum to 245760.
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
