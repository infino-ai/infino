// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Planning which superfiles a migration has to rewrite.
//!
//! A reindex job is a **one-input** compaction job: the same seal, merge
//! and manifest swap a merge already performs, with a single superfile
//! going in and its rewritten replacement coming out. Reusing that path is
//! what makes a migration atomic per superfile and safe across processes
//! for free — the tombstone-sidecar seal is what serializes writers, and
//! it does not care how many inputs a job has.
//!
//! Compaction's own planner cannot be reused, and should not be changed to
//! allow it: it excludes superfiles already at target size, and requires
//! two inputs because merging one is a no-op *for compaction*. Both rules
//! stay true there. For a migration the rewrite is the entire point, and
//! the files it must reach are exactly the large, already-compacted ones
//! compaction skips — so the selection rule is different, and lives here.

use std::time::Duration;

use futures::future::join_all;
use tracing::warn;
use uuid::Uuid;

mod build;

use crate::{
    config::{ReindexMode, ReindexOptions},
    runtime_bridge::bridge_on_runtime,
    superfile::{
        fts::reader::{FtsStaleness, StaleColumn},
        reader::SuperfileReader,
    },
    supertable::{
        Supertable,
        compaction::{CompactionJob, JobOutcome, SuperfileMerge},
        error::{CompactionError, ReindexError},
        query::dispatch::open_compaction_input,
    },
};

/// Rewrites between refreshes of the global term-statistics sidecar.
///
/// The manifest drops its reference to that sidecar on **any** superfile
/// removal, and every rewrite is a removal — so without this a migration
/// would run its whole length with no sidecar, and every scored query
/// would fall back to a gather wave. Refreshing after each rewrite would
/// be correct and wasteful; refreshing never would be cheap and slow. This
/// bounds the window to a handful of rewrites.
const REWRITES_PER_TERM_STATS_REFRESH: usize = 16;

/// Superfiles opened at once while deciding which are stale.
///
/// Opening one materializes its bytes — there is no ranged read that
/// fetches a header alone — so a fan-out over the whole manifest would
/// hold the entire table resident just to read a version field. Assessing
/// in batches keeps the peak at this many superfiles regardless of table
/// size, while still overlapping enough fetches to hide latency on object
/// storage, where this scan is one round trip per file.
const SUPERFILES_ASSESSED_AT_ONCE: usize = 8;

/// One superfile that is behind, with what a job needs to rewrite it.
#[derive(Debug, Clone)]
pub(crate) struct StaleSuperfile {
    pub(crate) superfile_id: Uuid,
    /// Carried onto the job so the rewritten file lands in the partition
    /// its rows already belong to.
    pub(crate) partition_key: Vec<u8>,
    /// Live bytes, for the job's size estimate.
    pub(crate) live_bytes: u64,
    pub(crate) fts: FtsStaleness,
}

impl StaleSuperfile {
    /// Assess one superfile against what this engine writes today.
    ///
    /// A superfile with no FTS index is never stale here — this migration
    /// is about the FTS blob, and a file without one has nothing to say
    /// about it.
    pub(crate) fn assess(
        superfile_id: Uuid,
        partition_key: Vec<u8>,
        live_bytes: u64,
        reader: &SuperfileReader,
    ) -> Self {
        let fts = reader.fts().map(|f| f.staleness()).unwrap_or_default();
        Self {
            superfile_id,
            partition_key,
            live_bytes,
            fts,
        }
    }

    /// Columns a rewrite cannot repair, because their text was never
    /// stored and so their terms cannot be regenerated.
    ///
    /// A migration reports these rather than carrying them silently: the
    /// file comes out with a current container and terms that are still
    /// the old chain's, and the only remaining repair is re-ingesting the
    /// column from its source, which is outside this engine.
    pub(crate) fn unrepairable_columns(&self) -> impl Iterator<Item = &StaleColumn> {
        self.fts.unrepairable_columns()
    }
}

/// What a reindex *would* do, without doing any of it.
///
/// Every field here is derived from the same scan
/// [`crate::Supertable::reindex`] plans from, so a number in this report
/// is the number that run will act on — not an estimate of it.
///
/// It exists because the migration is deliberately on demand, and an
/// operation nobody is told to run is one nobody runs. Every input to the
/// decision — whether anything is behind, which axis, how many bytes move,
/// whether re-analysis would refuse the table outright — was already being
/// computed and then discarded inside `reindex`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StalenessReport {
    /// Superfiles in the table, stale or not — the denominator for
    /// everything below.
    pub superfiles: usize,
    /// Superfiles whose container is behind, which is exactly what
    /// [`crate::ReindexMode::Rewrite`] would rewrite.
    pub needing_rewrite: usize,
    /// Superfiles holding terms an older analysis produced, which only
    /// [`crate::ReindexMode::Reanalyze`] can repair.
    ///
    /// Counts only files with at least one stale column whose text is
    /// stored. A file whose stale columns are all index-only cannot be
    /// repaired and is named through [`Self::unrepairable_columns`]
    /// instead, because re-analyzing it would rewrite the corpus and
    /// change nothing.
    pub awaiting_reanalysis: usize,
    /// Live bytes in the superfiles a [`crate::ReindexMode::Rewrite`]
    /// would read and write again.
    ///
    /// The cost of the cheap mode, for sizing a run. Re-analysis touches
    /// these *and* the files in [`Self::awaiting_reanalysis`], and costs
    /// far more per byte besides, so this does not bound it.
    pub bytes_to_rewrite: u64,
    /// Columns no rewrite and no re-analysis can repair, because their
    /// text was never stored.
    ///
    /// Non-empty means a fully migrated table will still hold terms from
    /// an older analysis in these columns, and the only remaining repair
    /// is re-ingesting them from their source.
    pub unrepairable_columns: Vec<String>,
}

impl StalenessReport {
    /// Whether a reindex would do anything at all.
    pub fn is_current(&self) -> bool {
        self.needing_rewrite == 0 && self.awaiting_reanalysis == 0
    }
}

/// What a reindex did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReindexReport {
    /// Superfiles rewritten into the current format.
    pub rewritten: usize,
    /// Superfiles already in the current format when the run planned.
    pub already_current: usize,
    /// Superfiles holding terms from an older analysis, which a rewrite
    /// cannot repair.
    ///
    /// A rewrite copies postings, so it moves the container and leaves the
    /// terms alone. Clearing these needs re-analysis from the stored text,
    /// a separate and far more expensive operation — so they are counted
    /// here rather than rewritten pointlessly on every run.
    pub awaiting_reanalysis: usize,
    /// Stale superfiles this run could not take, because another run holds
    /// their tombstone sidecar.
    ///
    /// The sidecar seal is the cross-process guard that keeps two writers
    /// off one superfile, and it is held by superfile rather than by
    /// table — so a concurrent compaction, or a *crashed* run whose seal
    /// outlives it, blocks that file and nothing else. Skipping it and
    /// carrying on is what lets a resume make progress on the rest of the
    /// table; failing the run would let one abandoned seal hold a whole
    /// migration for as long as the seal takes to go stale.
    ///
    /// Non-zero means the table is not fully migrated and the work is
    /// still there to do. Run again: a live owner will have finished, and
    /// an abandoned seal is taken over once it is older than
    /// [`crate::ReindexOptions::stale_seal_timeout_ms`].
    pub held_by_another_run: usize,
    /// Columns carried forward still stale, because their text was never
    /// stored and nothing in the file can regenerate their terms.
    ///
    /// Non-empty means the migration is as complete as this engine can
    /// make it and the table still holds terms from an older analysis.
    /// The only remaining repair is re-ingesting those columns from their
    /// source, which is outside the engine — so this is reported rather
    /// than swallowed.
    pub unrepairable_columns: Vec<String>,
}

/// One rewrite job per stale superfile.
///
/// Deliberately not batched. A merge of several stale superfiles would
/// rewrite fewer files, but it also changes which rows share a file — so a
/// migration would reshape the table as a side effect, and a failure
/// halfway would leave a partly-reshaped one. One in, one out keeps a
/// migration a migration: every job is independently committable,
/// independently retryable, and leaves the table's layout exactly as it
/// found it.
///
/// Ordering is by id so a plan is stable across runs, which is what lets
/// an interrupted migration resume by simply re-planning: the superfiles
/// already rewritten are no longer stale and drop out.
///
/// Selects on [`FtsStaleness::needs_rewrite`] rather than on staleness in
/// general, and the difference is what makes a migration terminate. A
/// rewrite carries postings across, so it moves a file's container to the
/// current one and leaves its analysis revision exactly where it was —
/// planning a rewrite for a file that is only analysis-stale would emit
/// the same job on every run, each producing a file as stale as the last.
/// Those files need re-analysis, which is a different operation, and they
/// are reported rather than rewritten.
pub(crate) fn plan_jobs(stale: &[StaleSuperfile], mode: ReindexMode) -> Vec<CompactionJob> {
    let mut stale: Vec<&StaleSuperfile> = stale
        .iter()
        .filter(|s| match mode {
            // A rewrite copies postings, so it cannot clear an analysis
            // revision — planning one for a file that is only
            // analysis-stale would emit the same job forever.
            ReindexMode::Rewrite => s.fts.needs_rewrite(),
            // Re-analysis produces new terms and a current container, so
            // it repairs either axis.
            ReindexMode::Reanalyze => s.fts.needs_rewrite() || s.fts.needs_reanalysis(),
        })
        .collect();
    stale.sort_by_key(|s| s.superfile_id);
    stale
        .into_iter()
        .map(|s| CompactionJob {
            partition_key: s.partition_key.clone(),
            inputs: vec![s.superfile_id],
            estimated_output_bytes: s.live_bytes,
        })
        .collect()
}

impl Supertable {
    /// Every superfile in the current snapshot that is behind what this
    /// engine writes, assessed by opening each one.
    ///
    /// The manifest records no format version, so staleness is read from
    /// the files themselves, which keeps the manifest free of a field
    /// needing its own backward-compatible decode path forever.
    ///
    /// Reading it costs a full open per superfile — the reader has no
    /// header-only path — so the scan runs in batches of
    /// [`SUPERFILES_ASSESSED_AT_ONCE`] and drops each batch's readers
    /// before taking the next. Peak memory is then a function of that
    /// constant rather than of how large the table is. Readers that the
    /// cache already holds cost nothing extra.
    /// Returns the stale superfiles and how many the snapshot held, so a
    /// caller can report both against one point in time.
    pub(crate) async fn stale_superfiles(
        &self,
    ) -> Result<(Vec<StaleSuperfile>, usize), CompactionError> {
        let manifest = self.inner().manifest.load_full();
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();

        let entries = manifest.get_all_superfiles();
        let mut stale = Vec::new();
        for batch in entries.chunks(SUPERFILES_ASSESSED_AT_ONCE) {
            let opens = batch.iter().map(|entry| {
                let entry = entry.clone();
                let (store, disk_cache, storage) =
                    (store.clone(), disk_cache.clone(), storage.clone());
                async move {
                    let reader = open_compaction_input(
                        &store,
                        disk_cache.as_ref(),
                        storage.as_ref(),
                        &entry,
                    )
                    .await;
                    (entry, reader)
                }
            });
            for (entry, reader) in join_all(opens).await {
                let reader = reader.map_err(|e| CompactionError::Build(e.to_string()))?;
                let live_bytes = entry
                    .subsection_offsets
                    .as_ref()
                    .map_or(0, |o| o.total_size);
                let assessed = StaleSuperfile::assess(
                    entry.superfile_id,
                    entry.partition_key.clone(),
                    live_bytes,
                    &reader,
                );
                if !assessed.fts.is_current() {
                    stale.push(assessed);
                }
            }
            // Readers from this batch go out of scope here, so the next
            // batch's opens do not stack on top of them.
        }
        Ok((stale, entries.len()))
    }
}

impl Supertable {
    /// Rewrite every superfile whose FTS index is behind what this engine
    /// writes today, leaving the table's rows, ids and layout unchanged.
    ///
    /// Each superfile is rewritten on its own and committed on its own, so
    /// a query sees either the old file or its replacement and never a
    /// half-migrated table. A run that is interrupted leaves the rewrites
    /// it finished in place; running again picks up exactly what is left,
    /// because "what is left" is read from the files rather than tracked
    /// in a journal.
    ///
    /// Idempotent: a second run over a migrated table plans nothing.
    ///
    /// This rewrites containers — the layout and the bounds. It does not
    /// re-analyze text, so a column whose terms came from an older
    /// analysis stays stale and is named in
    /// [`ReindexReport::unrepairable_columns`] only when nothing *could*
    /// repair it.
    ///
    /// # Errors
    ///
    /// [`ReindexError::NoStorage`] without a durable backend,
    /// [`ReindexError::AlreadyRunning`] while a compaction or another
    /// reindex holds the slot, and [`ReindexError::Rewrite`] naming the
    /// superfile whose rewrite failed.
    /// What a reindex would do, without doing it.
    ///
    /// Reads every superfile's index metadata and reports what is behind
    /// and what repairing it would cost. Writes nothing and takes no
    /// writer slot, so it is safe to run against a live table and safe to
    /// run while a reindex or a compaction is in flight — the numbers are
    /// then a snapshot that run is already changing.
    ///
    /// # Errors
    ///
    /// [`ReindexError::NoStorage`] without a durable backend, and
    /// [`ReindexError::Assess`] if a superfile cannot be opened.
    pub fn index_staleness(&self) -> Result<StalenessReport, ReindexError> {
        bridge_on_runtime(self.index_staleness_async(), &self.inner().query_runtime())
    }

    async fn index_staleness_async(&self) -> Result<StalenessReport, ReindexError> {
        if self.inner().manifest.load_full().options.storage.is_none() {
            return Err(ReindexError::NoStorage);
        }
        // No writer slot: this reads and reports. Taking one would make an
        // assessment fail while a migration it is meant to describe is
        // running, which is precisely when someone asks.
        let (stale, superfiles) = self
            .stale_superfiles()
            .await
            .map_err(|e| ReindexError::Assess(e.to_string()))?;

        let mut report = StalenessReport {
            superfiles,
            ..Default::default()
        };
        for file in &stale {
            // Same predicates the planner filters on, so a count here is
            // the count that run acts on rather than an estimate of it.
            if file.fts.needs_rewrite() {
                report.needing_rewrite += 1;
                report.bytes_to_rewrite += file.live_bytes;
            }
            if file.fts.needs_reanalysis() {
                report.awaiting_reanalysis += 1;
            }
            for column in file.unrepairable_columns() {
                if !report.unrepairable_columns.contains(&column.name) {
                    report.unrepairable_columns.push(column.name.clone());
                }
            }
        }
        Ok(report)
    }

    pub fn reindex(&self, opts: &ReindexOptions) -> Result<ReindexReport, ReindexError> {
        bridge_on_runtime(self.reindex_async(opts), &self.inner().query_runtime())
    }

    async fn reindex_async(&self, opts: &ReindexOptions) -> Result<ReindexReport, ReindexError> {
        let manifest = self.inner().manifest.load_full();
        if manifest.options.storage.is_none() {
            return Err(ReindexError::NoStorage);
        }
        let stale_seal_timeout = Duration::from_millis(opts.stale_seal_timeout_ms);

        // Share compaction's slot rather than adding a second one: both
        // rewrite superfiles and commit manifest swaps, so letting them
        // run together would have two planners racing over the same files
        // and losing each other's commits to the retry loop.
        let _slot = self
            .try_hold_compaction_slot()
            .ok_or(ReindexError::AlreadyRunning)?;

        // Both counts come from the scan's own snapshot. Taking `total`
        // from the manifest loaded above and `all` from inside the scan
        // would mix two points in time, so a commit landing between them
        // would skew the report by however many superfiles it added.
        let (all, total) = self
            .stale_superfiles()
            .await
            .map_err(|e| ReindexError::Assess(e.to_string()))?;

        let mut report = ReindexReport {
            already_current: total.saturating_sub(all.len()),
            awaiting_reanalysis: match opts.mode {
                // Re-analysis is what clears this axis, so a run that
                // performs it leaves nothing waiting — except the columns
                // it could not repair, which are named separately.
                ReindexMode::Reanalyze => 0,
                ReindexMode::Rewrite => all.iter().filter(|s| s.fts.needs_reanalysis()).count(),
            },
            ..Default::default()
        };
        for stale in &all {
            for column in stale.unrepairable_columns() {
                if !report.unrepairable_columns.contains(&column.name) {
                    report.unrepairable_columns.push(column.name.clone());
                }
            }
        }
        if !report.unrepairable_columns.is_empty() {
            warn!(
                "[supertable reindex] {} column(s) hold terms from an older \
                 analysis and are index-only, so no rewrite can repair them: {}",
                report.unrepairable_columns.len(),
                report.unrepairable_columns.join(", "),
            );
        }

        // `Rewrite` is compaction's build with deletions off; `Reanalyze`
        // is this tool's own. Both carry the row set.
        let merge: &dyn SuperfileMerge = match opts.mode {
            ReindexMode::Rewrite => &build::RewriteMerge,
            ReindexMode::Reanalyze => &build::ReanalyzeMerge,
        };
        for (done, job) in plan_jobs(&all, opts.mode).into_iter().enumerate() {
            let superfile_id = job.inputs[0];
            let outcome = match self
                .run_compaction_job_with(job, stale_seal_timeout, merge)
                .await
            {
                Ok(outcome) => outcome,
                // Someone else holds this superfile. That is the seal
                // doing its job, not a failure: the owner may be a live
                // compaction, or a run that died still holding it. Either
                // way this file is untouchable right now and every other
                // stale file is not, so the run continues and says what it
                // had to leave.
                Err(CompactionError::SidecarConflict { .. }) => {
                    report.held_by_another_run += 1;
                    continue;
                }
                Err(e) => {
                    return Err(ReindexError::Rewrite {
                        superfile_id,
                        cause: e.to_string(),
                    });
                }
            };
            // A job whose inputs another writer had already replaced did
            // no work. Counting it would report more rewrites than the
            // table received, which matters because this count is how a
            // caller decides a migration is done.
            if outcome == JobOutcome::Committed {
                report.rewritten += 1;
            }

            // Bound how long the table runs without its term-statistics
            // sidecar; see REWRITES_PER_TERM_STATS_REFRESH.
            if (done + 1) % REWRITES_PER_TERM_STATS_REFRESH == 0 {
                self.refresh_term_stats_best_effort();
            }
        }
        if report.rewritten > 0 {
            self.refresh_term_stats_best_effort();
        }
        Ok(report)
    }

    /// Rebuild the global term-statistics sidecar, logging rather than
    /// failing.
    ///
    /// A missing sidecar costs latency, never correctness — queries fall
    /// back to gathering the statistics live — so a refresh that fails is
    /// not a reason to abandon a migration that is otherwise succeeding.
    fn refresh_term_stats_best_effort(&self) {
        if let Err(e) = self.refresh_term_stats_sync() {
            warn!("[supertable reindex] term-stats refresh failed, queries gather live: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::superfile::fts::reader::StaleColumn;

    fn entry(id: u128, fts: FtsStaleness) -> StaleSuperfile {
        StaleSuperfile {
            superfile_id: Uuid::from_u128(id),
            partition_key: vec![7],
            live_bytes: 1_024,
            fts,
        }
    }

    fn behind_container() -> FtsStaleness {
        FtsStaleness {
            container: Some(4),
            analysis: Vec::new(),
        }
    }

    fn behind_analysis() -> FtsStaleness {
        FtsStaleness {
            container: None,
            analysis: vec![StaleColumn {
                name: "title".into(),
                recorded: 0,
                current: 1,
                stored: true,
            }],
        }
    }

    /// A superfile behind on its container earns a job; one that is
    /// current earns none.
    #[test]
    fn plans_one_job_per_superfile_behind_on_its_container() {
        let jobs = plan_jobs(
            &[
                entry(1, behind_container()),
                entry(2, FtsStaleness::default()),
            ],
            ReindexMode::Rewrite,
        );
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        assert_eq!(jobs[0].inputs, vec![Uuid::from_u128(1)]);
    }

    /// A file that is *only* analysis-stale earns no rewrite, and this is
    /// what makes a migration terminate.
    ///
    /// A rewrite carries postings, so the file it produces records the
    /// same analysis revision the input did. Planning one here would emit
    /// the identical job on the next run and every run after it, with the
    /// table never converging and each pass paying a full rewrite of the
    /// corpus. Re-analysis is the operation that clears these.
    #[test]
    fn an_only_analysis_stale_superfile_earns_no_rewrite() {
        let jobs = plan_jobs(&[entry(3, behind_analysis())], ReindexMode::Rewrite);
        assert!(
            jobs.is_empty(),
            "a rewrite cannot clear an analysis revision, so planning one \
             would never terminate: {jobs:?}"
        );
    }

    /// Behind on both axes: the rewrite is still worth doing for the
    /// container, and the analysis stays for re-analysis to clear.
    #[test]
    fn a_superfile_behind_on_both_axes_is_rewritten() {
        let both = FtsStaleness {
            container: Some(4),
            analysis: behind_analysis().analysis,
        };
        assert_eq!(plan_jobs(&[entry(5, both)], ReindexMode::Rewrite).len(), 1);
    }

    /// Every job takes exactly one input, so a rewrite never merges rows
    /// that were not already together.
    #[test]
    fn every_job_is_one_in_one_out() {
        let jobs = plan_jobs(
            &[entry(9, behind_container()), entry(4, behind_analysis())],
            ReindexMode::Rewrite,
        );
        assert!(jobs.iter().all(|j| j.inputs.len() == 1), "{jobs:?}");
        assert_eq!(jobs[0].partition_key, vec![7], "partition is carried");
    }

    /// A plan is stable across runs, which is what makes an interrupted
    /// migration resumable by re-planning rather than by a journal.
    #[test]
    fn a_plan_is_ordered_independently_of_input_order() {
        let forward = plan_jobs(
            &[entry(2, behind_container()), entry(1, behind_container())],
            ReindexMode::Rewrite,
        );
        let reversed = plan_jobs(
            &[entry(1, behind_container()), entry(2, behind_container())],
            ReindexMode::Rewrite,
        );
        assert_eq!(forward, reversed);
    }

    /// Nothing stale, nothing to do — the state a completed migration
    /// converges to, and the check a caller polls.
    #[test]
    fn a_current_table_plans_no_work() {
        assert!(plan_jobs(&[entry(1, FtsStaleness::default())], ReindexMode::Rewrite).is_empty());
    }
}
