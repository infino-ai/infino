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

use std::{sync::Arc, time::Duration};

use futures::future::join_all;
use tracing::warn;
use uuid::Uuid;

mod build;

use crate::{
    config::{self, ReindexMode, ReindexOptions, ReindexTarget, SuperfileIndex},
    runtime_bridge::bridge_on_runtime,
    superfile::{
        format::footer::{BlobRegion, has_duplicated_region_key, resolved_regions},
        fts::{
            analysis::{UNKNOWN_ANALYSIS_REVISION, analysis_revision_written_by},
            reader::{FtsStaleness, StaleColumn},
        },
        reader::{SuperfileReader, writer_builder_of},
    },
    supertable::{
        Supertable,
        error::{CompactionError, ReindexError},
        manifest::SubsectionOffsets,
        optimize::compact::{CompactionJob, JobOutcome, SuperfileMerge},
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
    /// The footer stores a blob region key more than once. The copy this
    /// engine reads is sound, and a rewrite lays the footer out afresh
    /// from it, so it repairs this whatever the FTS index's state.
    pub(crate) has_duplicated_region_keys: bool,
}

/// Where a superfile's footer says its blobs sit, judged against the file
/// and its manifest entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FooterState {
    /// Each region key stored once, inside the file, agreeing with the
    /// manifest.
    Sound,
    /// A region key stored more than once, while the copy this engine
    /// reads is inside the file and agrees with the manifest: a stale
    /// duplicate a rewrite removes.
    DuplicatedKeys,
    /// The regions this engine reads run past the file or disagree with
    /// the manifest. Nothing says which side is right, so a rewrite could
    /// carry the wrong bytes and erase the evidence; the file is reported
    /// and never rewritten.
    Inconsistent,
}

/// What one scan found: the superfiles a reindex can repair, the ones it
/// must leave for a person, and how many the snapshot held.
pub(crate) struct Assessment {
    pub(crate) stale: Vec<StaleSuperfile>,
    pub(crate) inconsistent_footers: Vec<Uuid>,
    pub(crate) superfiles: usize,
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
        has_duplicated_region_keys: bool,
        trust_writer_analysis: bool,
    ) -> Self {
        // A file recording no revision is an unknown unless the caller has
        // taken responsibility for reading it as its writer's.
        let assumed = match trust_writer_analysis {
            true => writer_builder_of(reader.parquet_metadata())
                .map_or(UNKNOWN_ANALYSIS_REVISION, analysis_revision_written_by),
            false => UNKNOWN_ANALYSIS_REVISION,
        };
        let fts = reader
            .fts()
            .map(|f| f.staleness(assumed))
            .unwrap_or_default();
        Self {
            superfile_id,
            partition_key,
            live_bytes,
            fts,
            has_duplicated_region_keys,
        }
    }

    /// Whether a rewrite would change anything: the container is behind,
    /// or the footer stores a stale duplicate.
    pub(crate) fn needs_rewrite(&self) -> bool {
        self.fts.needs_rewrite() || self.has_duplicated_region_keys
    }

    /// Whether this file is behind on anything a reindex repairs.
    pub(crate) fn is_current(&self) -> bool {
        self.fts.is_current() && !self.has_duplicated_region_keys
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

/// Judge `reader`'s footer regions against the file and the manifest.
///
/// Judged on the copy of each key this engine reads, the last one, since
/// that is what queries are served from; a stale earlier copy only misleads
/// a reader that keeps the first.
fn footer_state(reader: &SuperfileReader, offsets: Option<&SubsectionOffsets>) -> FooterState {
    let metadata = reader.parquet_metadata();
    // A reindex input is always opened whole; one that is not cannot be
    // checked against its bytes, so it is left alone like any other doubt.
    let (Some(resolved), Some(bytes)) = (resolved_regions(metadata), reader.whole_file_bytes())
    else {
        return FooterState::Inconsistent;
    };
    // An older manifest may record no region; only a recorded one can
    // disagree.
    let agrees = |manifest: Option<BlobRegion>, footer: Option<BlobRegion>| {
        manifest.is_none_or(|m| Some(m) == footer)
    };
    let in_file = resolved.fit_within(bytes.len() as u64);
    let matches_manifest =
        offsets.is_none_or(|o| agrees(o.fts, resolved.fts) && agrees(o.vec, resolved.vec));
    match (
        in_file && matches_manifest,
        has_duplicated_region_key(metadata),
    ) {
        (false, _) => FooterState::Inconsistent,
        (true, true) => FooterState::DuplicatedKeys,
        (true, false) => FooterState::Sound,
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
/// which columns nothing can repair — was already being computed and then
/// discarded inside `reindex`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StalenessReport {
    /// Superfiles in the table, stale or not — the denominator for
    /// everything below.
    pub superfiles: usize,
    /// Superfiles whose container is behind or whose footer misplaces a
    /// blob, which is exactly what [`crate::ReindexMode::Rewrite`] would
    /// rewrite.
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
    /// Superfiles whose footer places a blob somewhere the file or its
    /// manifest entry contradicts, which a reindex reports and never
    /// rewrites.
    ///
    /// Nothing in the file says whether the footer or the manifest is
    /// right, so a rewrite could carry the wrong bytes and leave a file
    /// that only looks consistent. Non-empty means a superfile needs a
    /// person, not a migration; these are counted nowhere else.
    pub inconsistent_footers: Vec<Uuid>,
}

impl StalenessReport {
    /// Whether a reindex would do anything at all.
    ///
    /// A table can be current and still hold
    /// [`Self::inconsistent_footers`], which no reindex acts on.
    pub fn is_current(&self) -> bool {
        self.needing_rewrite == 0 && self.awaiting_reanalysis == 0
    }
}

/// One superfile a reindex would repair, and how.
///
/// What [`StalenessReport`] counts, this names. A caller reviewing a
/// migration before running it needs the list rather than the totals —
/// which files move, and which of them pay for a re-analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PlannedRepair {
    /// The superfile this job reads and replaces.
    pub superfile_id: Uuid,
    /// The repair it gets. Never [`ReindexMode::Auto`]: that is the
    /// question a plan answers, so it is already resolved here.
    pub mode: ReindexMode,
    /// Live bytes in the superfile, so a caller can size the run.
    pub live_bytes: u64,
}

/// What a reindex did.
///
/// Counted against the snapshot the run planned from, which is one moment
/// and not a lock. A concurrent compaction can merge a superfile this run
/// planned to repair; the job then finds it gone and does nothing, while
/// the merged output — no newer than its oldest input, so stale in the
/// same way — was not in the plan and is not repaired here. A run that
/// reports nothing left can therefore be followed by one that finds work,
/// which is the price of not holding the table still for a migration.
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
    /// their tombstone sidecar, took it over, or kept it moving.
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
    /// Superfiles left untouched because their footer places a blob
    /// somewhere the file or its manifest entry contradicts; see
    /// [`StalenessReport::inconsistent_footers`].
    pub inconsistent_footers: Vec<Uuid>,
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
/// Under [`ReindexMode::Rewrite`] this selects on
/// [`StaleSuperfile::needs_rewrite`] rather than on staleness in general,
/// and the difference is what makes a migration terminate. A rewrite
/// carries postings across, so it moves a file's container to the current
/// one and leaves its analysis revision exactly where it was — planning a
/// rewrite for a file that is only analysis-stale would emit the same job
/// on every run, each producing a file as stale as the last. Those files
/// need re-analysis, and under that mode they are reported rather than
/// rewritten.
pub(crate) fn plan_jobs(
    stale: &[StaleSuperfile],
    mode: ReindexMode,
) -> Vec<(CompactionJob, Repair)> {
    let mut stale: Vec<&StaleSuperfile> = stale
        .iter()
        .filter(|s| match mode {
            // A rewrite copies postings, so it cannot clear an analysis
            // revision — planning one for a file that is only
            // analysis-stale would emit the same job forever.
            ReindexMode::Rewrite => s.needs_rewrite(),
            // Re-analysis produces new terms and a current container, so
            // it repairs either axis.
            ReindexMode::Auto | ReindexMode::Reanalyze => {
                s.needs_rewrite() || s.fts.needs_reanalysis()
            }
        })
        .collect();
    stale.sort_by_key(|s| s.superfile_id);
    stale
        .into_iter()
        .map(|s| {
            let repair = match mode {
                ReindexMode::Rewrite => Repair::Layout,
                ReindexMode::Reanalyze => Repair::Terms,
                // The cheapest repair that makes *this* file current:
                // copying postings cannot clear a stale revision, so only
                // a file whose terms are current can take the cheap one.
                ReindexMode::Auto => match s.fts.needs_reanalysis() {
                    true => Repair::Terms,
                    false => Repair::Layout,
                },
            };
            let job = CompactionJob {
                partition_key: s.partition_key.clone(),
                inputs: vec![s.superfile_id],
                estimated_output_bytes: s.live_bytes,
            };
            (job, repair)
        })
        .collect()
}

/// The distinct columns no repair can fix, in first-seen order.
///
/// Both reports name these, and the docs promise the two agree — so they
/// read it from here rather than each folding the same list themselves.
fn unrepairable_column_names(stale: &[StaleSuperfile]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for file in stale {
        for column in file.unrepairable_columns() {
            if !names.contains(&column.name) {
                names.push(column.name.clone());
            }
        }
    }
    names
}

/// The repair one superfile gets, after [`ReindexMode::Auto`] has been
/// resolved against what that file is actually behind on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Repair {
    /// Copy the postings into the current layout.
    Layout,
    /// Re-analyze the stored text, which brings the layout current too.
    Terms,
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
    /// Returns what it found against one snapshot, so a caller reports
    /// every count against the same point in time.
    pub(crate) async fn stale_superfiles(
        &self,
        trust_writer_analysis: bool,
    ) -> Result<Assessment, CompactionError> {
        let manifest = self.inner().manifest.load_full();
        let store = manifest.options.store.clone();
        let disk_cache = manifest.options.disk_cache.clone();
        let storage = manifest.options.storage.clone();

        let entries = manifest.get_all_superfiles();
        let mut stale = Vec::new();
        let mut inconsistent_footers = Vec::new();
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
                let offsets = entry.subsection_offsets.as_ref();
                let has_duplicated_region_keys = match footer_state(&reader, offsets) {
                    FooterState::Sound => false,
                    FooterState::DuplicatedKeys => true,
                    // Kept out of the stale list, so no plan or count can
                    // reach it.
                    FooterState::Inconsistent => {
                        inconsistent_footers.push(entry.superfile_id);
                        continue;
                    }
                };
                let assessed = StaleSuperfile::assess(
                    entry.superfile_id,
                    entry.partition_key.clone(),
                    offsets.map_or(0, |o| o.total_size),
                    &reader,
                    has_duplicated_region_keys,
                    trust_writer_analysis,
                );
                if !assessed.is_current() {
                    stale.push(assessed);
                }
            }
            // Readers from this batch go out of scope here, so the next
            // batch's opens do not stack on top of them.
        }
        inconsistent_footers.sort_unstable();
        Ok(Assessment {
            stale,
            inconsistent_footers,
            superfiles: entries.len(),
        })
    }
}

impl Supertable {
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
    /// The superfiles [`Supertable::reindex`] would repair under `opts`,
    /// and the repair each one gets — without repairing anything.
    ///
    /// Planned off one snapshot, so it predicts the run that starts now;
    /// see [`ReindexReport`] for what a concurrent writer does to that.
    /// Writes nothing and takes no writer slot.
    ///
    /// # Errors
    ///
    /// [`ReindexError::NoStorage`] without a durable backend, and
    /// [`ReindexError::Assess`] if a superfile cannot be opened.
    pub fn reindex_plan(&self, opts: &ReindexOptions) -> Result<Vec<PlannedRepair>, ReindexError> {
        bridge_on_runtime(self.reindex_plan_async(opts), &self.inner().query_runtime())
    }

    async fn reindex_plan_async(
        &self,
        opts: &ReindexOptions,
    ) -> Result<Vec<PlannedRepair>, ReindexError> {
        let ReindexTarget::Superfile(SuperfileIndex::Fts) = opts.target;
        if self.inner().manifest.load_full().options.storage.is_none() {
            return Err(ReindexError::NoStorage);
        }
        let assessment = self
            .stale_superfiles(opts.trust_writer_analysis)
            .await
            .map_err(|e| ReindexError::Assess(e.to_string()))?;
        // The same planner the run drives, so the two cannot disagree.
        Ok(plan_jobs(&assessment.stale, opts.mode)
            .into_iter()
            .map(|(job, repair)| PlannedRepair {
                superfile_id: job.inputs[0],
                mode: match repair {
                    Repair::Layout => ReindexMode::Rewrite,
                    Repair::Terms => ReindexMode::Reanalyze,
                },
                live_bytes: job.estimated_output_bytes,
            })
            .collect())
    }

    pub fn index_staleness(&self, opts: &ReindexOptions) -> Result<StalenessReport, ReindexError> {
        bridge_on_runtime(
            self.index_staleness_async(opts),
            &self.inner().query_runtime(),
        )
    }

    async fn index_staleness_async(
        &self,
        opts: &ReindexOptions,
    ) -> Result<StalenessReport, ReindexError> {
        // Matched here as well as in the two entry points that act, so a
        // second target cannot report against the full-text index while a
        // run repairs something else.
        let ReindexTarget::Superfile(SuperfileIndex::Fts) = opts.target;
        if self.inner().manifest.load_full().options.storage.is_none() {
            return Err(ReindexError::NoStorage);
        }
        // No writer slot: this reads and reports. Taking one would make an
        // assessment fail while a migration it is meant to describe is
        // running, which is precisely when someone asks.
        let Assessment {
            stale,
            inconsistent_footers,
            superfiles,
        } = self
            .stale_superfiles(opts.trust_writer_analysis)
            .await
            .map_err(|e| ReindexError::Assess(e.to_string()))?;

        let mut report = StalenessReport {
            superfiles,
            inconsistent_footers,
            ..Default::default()
        };
        for file in &stale {
            // Same predicates the planner filters on, so a count here is
            // the count that run acts on rather than an estimate of it.
            if file.needs_rewrite() {
                report.needing_rewrite += 1;
                report.bytes_to_rewrite += file.live_bytes;
            }
            if file.fts.needs_reanalysis() {
                report.awaiting_reanalysis += 1;
            }
        }
        report.unrepairable_columns = unrepairable_column_names(&stale);
        Ok(report)
    }

    /// Repair the index named by [`ReindexOptions::target`] in every
    /// superfile this table holds that is behind what this engine writes.
    ///
    /// Scope, because a table is more than its superfiles:
    ///
    /// - **Rebuilt** — the full-text index inside each stale superfile.
    /// - **Copied byte for byte** — the Parquet body and the vector blob,
    ///   so a repair never re-encodes a row or decodes a vector.
    /// - **Never opened** — the hidden vector index.
    /// - **Written to publish the result, not migrated** — one manifest
    ///   commit per superfile, each output's tombstone sidecar, and the
    ///   table's term-statistics sidecar. These follow from replacing a
    ///   file; their own formats are untouched.
    ///
    /// Superfiles are brought to the index layout this engine writes. One
    /// already at or above it is left alone, because a newer release may
    /// write a layout this one does not produce.
    ///
    /// Rows, their order and their `_id`s are identical across a repair,
    /// deleted rows included — dropping them would renumber the
    /// survivors.
    ///
    /// # Errors
    ///
    /// [`ReindexError::NoStorage`] when the table has none,
    /// [`ReindexError::AlreadyRunning`] when a compaction or another
    /// reindex holds the slot, and [`ReindexError::Rewrite`] when a
    /// superfile fails to rewrite. Superfiles another run has sealed are
    /// counted in [`ReindexReport::held_by_another_run`] rather than
    /// failing the run.
    pub fn reindex(&self, opts: &ReindexOptions) -> Result<ReindexReport, ReindexError> {
        bridge_on_runtime(self.reindex_async(opts), &self.inner().query_runtime())
    }

    async fn reindex_async(&self, opts: &ReindexOptions) -> Result<ReindexReport, ReindexError> {
        // Everything below assesses and repairs the full-text index. The
        // match is what makes a second target a decision here rather than
        // a field this function quietly ignores.
        let ReindexTarget::Superfile(SuperfileIndex::Fts) = opts.target;

        let manifest = self.inner().manifest.load_full();
        if manifest.options.storage.is_none() {
            return Err(ReindexError::NoStorage);
        }
        // Unset means the table's own compaction setting: the seal is the
        // same guard, so a table that tuned it meant it for this too.
        let stale_seal_timeout = Duration::from_millis(
            opts.stale_seal_timeout_ms
                .unwrap_or_else(|| config::global().compaction.stale_seal_timeout_ms),
        );

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
        let Assessment {
            stale: all,
            inconsistent_footers,
            superfiles: total,
        } = self
            .stale_superfiles(opts.trust_writer_analysis)
            .await
            .map_err(|e| ReindexError::Assess(e.to_string()))?;
        if !inconsistent_footers.is_empty() {
            warn!(
                "[supertable reindex] {} superfile(s) have a footer that places a \
                 blob where the file or its manifest entry contradicts; left \
                 untouched for inspection: {:?}",
                inconsistent_footers.len(),
                inconsistent_footers,
            );
        }

        let mut report = ReindexReport {
            already_current: total.saturating_sub(all.len() + inconsistent_footers.len()),
            awaiting_reanalysis: match opts.mode {
                // Re-analysis is what clears this axis, so a run that
                // performs it leaves nothing waiting — except the columns
                // it could not repair, which are named separately. `Auto`
                // re-analyzes exactly the files on this axis, so it clears
                // it too.
                ReindexMode::Auto | ReindexMode::Reanalyze => 0,
                ReindexMode::Rewrite => all.iter().filter(|s| s.fts.needs_reanalysis()).count(),
            },
            inconsistent_footers,
            ..Default::default()
        };
        report.unrepairable_columns = unrepairable_column_names(&all);
        if !report.unrepairable_columns.is_empty() {
            warn!(
                "[supertable reindex] {} column(s) hold terms from an older \
                 analysis and are index-only, so no rewrite can repair them: {}",
                report.unrepairable_columns.len(),
                report.unrepairable_columns.join(", "),
            );
        }

        // `Layout` is compaction's build with deletions off; `Terms` is
        // this tool's own. Both carry the row set. Built once and shared,
        // because the plan picks between them per superfile.
        let layout: Arc<dyn SuperfileMerge> = Arc::new(build::RepairMerge(Repair::Layout));
        let terms: Arc<dyn SuperfileMerge> = Arc::new(build::RepairMerge(Repair::Terms));
        for (done, (job, repair)) in plan_jobs(&all, opts.mode).into_iter().enumerate() {
            let merge = match repair {
                Repair::Layout => &layout,
                Repair::Terms => &terms,
            };
            let superfile_id = job.inputs[0];
            let outcome = match self
                .run_compaction_job_with(job, stale_seal_timeout, Arc::clone(merge))
                .await
            {
                Ok(outcome) => outcome,
                // Someone else holds this superfile. That is the seal
                // doing its job, not a failure: the owner may be a live
                // compaction, or a run that died still holding it. Either
                // way this file is untouchable right now and every other
                // stale file is not, so the run continues and says what it
                // had to leave.
                // Contention on one superfile's sidecar: another run holds
                // the seal, took ours, or is landing tombstones faster than
                // we can freeze them. The work is still there for a later
                // run, and one contended file must not end a migration.
                Err(
                    CompactionError::SidecarConflict { .. }
                    | CompactionError::SidecarChangedUnderSeal { .. }
                    | CompactionError::SealRetriesExhausted { .. },
                ) => {
                    report.held_by_another_run += 1;
                    continue;
                }
                // Another writer replaced this superfile between the plan
                // and the job. Its staleness went with it, so there is
                // nothing here to repair and nothing to report.
                Err(CompactionError::SuperfileNotFound(_)) => continue,
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
    use std::{fs, slice};

    use bytes::Bytes;
    use datafusion::prelude::{col, lit};
    use tempfile::TempDir;

    use super::*;
    use crate::{
        Bm25SearchOptions,
        storage::StorageProvider,
        superfile::{
            format::{footer::with_forged_footer_kv, kv},
            fts::reader::StaleColumn,
        },
        supertable::{
            Supertable,
            writer::{CommitListMetadata, build_subsection_offsets, persist_commit_async},
        },
        test_helpers::{copy_dir_recursive, old_format_fts_fixture, open_old_format_fts_fixture},
    };

    /// Count the hits a term has, across the whole fixture.
    fn hits(table: &Supertable, term: &str) -> usize {
        table
            .bm25_search("title", term, 100, Bm25SearchOptions::default(), None)
            .expect("search")
            .iter()
            .map(|b| b.num_rows())
            .sum()
    }

    /// A migration of a one-superfile table written before the term index
    /// existed leaves an index that lists that superfile, and marks it so.
    ///
    /// The rewrite publishes the table's first index, from this commit's
    /// postings alone, which is exactly what a full rebuild over the new
    /// membership produces. Marked incomplete, part selection and global idf
    /// would keep ignoring an index that is whole, and no later maintenance
    /// pass would notice, since its rebuild finds the root it would write.
    #[test]
    fn a_migrated_single_superfile_table_marks_its_term_index_complete() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);
        let manifest = table.reader().expect("reader").manifest().clone();
        assert!(
            manifest.term_index_ref().is_none(),
            "the fixture must predate the term index for this to prove anything"
        );

        // Leave one stale superfile: drop the others from the membership.
        let entries = table
            .block_on_query(manifest.get_all_superfiles_loaded())
            .expect("load entries");
        table
            .block_on_query(persist_commit_async(
                table.inner(),
                storage,
                Vec::new(),
                &entries[1..],
                Vec::new(),
                Vec::new(),
                CommitListMetadata::empty(),
                Vec::new(),
            ))
            .expect("drop all but one superfile");
        table.block_on_query(table.refresh()).expect("refresh");

        let report = table
            .reindex(&ReindexOptions::default())
            .expect("reindex the remaining superfile");
        assert_eq!(report.rewritten, 1, "{report:?}");

        let manifest = table.reader().expect("reader").manifest().clone();
        assert!(
            manifest.term_index_ref().is_some(),
            "the rewrite publishes an index"
        );
        assert!(
            manifest.term_index_complete(),
            "the only live superfile is the one the index lists"
        );
    }

    /// A re-analysis that repairs a table with a delete already on it.
    ///
    /// Runs on the committed fixture rather than the generated corpus, so
    /// it exercises the reindex path — the carried body, the re-analysis
    /// and the tombstone carry — wherever the corpus is absent. The
    /// corpus-backed tests cover the older formats this one cannot.
    #[test]
    fn a_reanalysis_repairs_a_stale_table_and_keeps_its_deletes() {
        const DELETED: &str = "alpha shared s0d00";
        const SURVIVOR: &str = "s1d01";

        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        let before = table
            .index_staleness(&ReindexOptions::default())
            .expect("assess");
        assert!(
            before.awaiting_reanalysis > 0,
            "the fixture must be analysis-stale for this to prove anything: {before:?}"
        );

        table
            .delete(col("title").eq(lit(DELETED)))
            .expect("delete one row");
        assert_eq!(hits(&table, "s0d00"), 0, "the row is gone before the run");
        let survivors = hits(&table, "shared");

        let report = table
            .reindex(&ReindexOptions::default())
            .expect("reindex the fixture");
        assert!(
            report.rewritten > 0,
            "nothing was rewritten, so nothing was tested: {report:?}"
        );

        let after = table
            .index_staleness(&ReindexOptions::default())
            .expect("assess again");
        assert!(
            after.is_current(),
            "a default reindex must leave the table current: {after:?}"
        );

        assert_eq!(
            hits(&table, "s0d00"),
            0,
            "the deleted row came back, so its tombstone was not carried"
        );
        assert_eq!(
            hits(&table, "shared"),
            survivors,
            "the rewrite changed how many rows a corpus-wide term matches"
        );
        assert_eq!(hits(&table, SURVIVOR), 1, "a survivor is still findable");
    }

    /// `Reanalyze` forces every stale superfile through a re-analysis,
    /// including ones a delete has already touched.
    ///
    /// `Auto` picks the repair per file; this mode does not, so it is the
    /// one that can pair the expensive rebuild with a carried tombstone on
    /// the same superfile.
    #[test]
    fn a_forced_reanalysis_keeps_the_deletes_it_rebuilds_over() {
        const DELETED: &str = "alpha shared s0d00";

        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        table
            .delete(col("title").eq(lit(DELETED)))
            .expect("delete one row");
        let survivors = hits(&table, "shared");

        let report = table
            .reindex(&ReindexOptions::reanalyzing())
            .expect("force a re-analysis");
        assert!(report.rewritten > 0, "nothing was re-analyzed: {report:?}");

        assert_eq!(
            hits(&table, "s0d00"),
            0,
            "the deleted row survived a forced re-analysis"
        );
        assert_eq!(
            hits(&table, "shared"),
            survivors,
            "the re-analysis changed which rows a corpus-wide term matches"
        );
        assert!(
            table
                .index_staleness(&ReindexOptions::default())
                .expect("assess")
                .is_current(),
            "a forced re-analysis must leave the table current"
        );
    }

    /// A dry run names exactly the superfiles a real run then repairs,
    /// and how each one is repaired.
    ///
    /// The point of it is reviewability: counts say how much work there
    /// is, this says what the work is. So it is held to matching the run
    /// it predicts, not merely to being non-empty.
    #[test]
    fn a_planned_run_names_what_the_real_run_repairs() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        let planned = table
            .reindex_plan(&ReindexOptions::default())
            .expect("plan a default run");
        assert!(
            !planned.is_empty(),
            "the fixture is stale, so there is a plan"
        );
        assert!(
            planned.iter().all(|p| p.mode == ReindexMode::Reanalyze),
            "the fixture's terms are behind, so Auto resolves to re-analysis: {planned:?}"
        );

        // Planning writes nothing, so planning twice must say the same.
        assert_eq!(
            table
                .reindex_plan(&ReindexOptions::default())
                .expect("plan again"),
            planned,
            "planning changed the table"
        );

        let report = table
            .reindex(&ReindexOptions::default())
            .expect("run what was planned");
        assert_eq!(
            report.rewritten,
            planned.len(),
            "the run repaired a different number of superfiles than it planned"
        );
        assert!(
            table
                .reindex_plan(&ReindexOptions::default())
                .expect("plan a third time")
                .is_empty(),
            "a migrated table has nothing left to plan"
        );
    }

    /// `rewriting()` cannot clear an analysis revision, so on a table
    /// whose terms are behind it plans nothing rather than planning a
    /// rewrite that would repeat forever.
    #[test]
    fn a_planned_rewrite_declines_a_table_only_its_terms_are_behind_on() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        let planned = table
            .reindex_plan(&ReindexOptions::rewriting())
            .expect("plan a rewrite");
        assert!(
            planned.iter().all(|p| p.mode == ReindexMode::Rewrite),
            "a rewrite plans only rewrites: {planned:?}"
        );
    }

    /// The committed fixture was written by `infino/0.8.6`, which records
    /// no analysis revision. That is an unknown, so by default its columns
    /// read as stale and a reindex re-analyzes them.
    ///
    /// Told to trust the writer, the same files read as current: 0.8.6
    /// shipped the chains this engine still has. The two answers are the
    /// trade the option exists for, so both are pinned here.
    #[test]
    fn an_unrecorded_revision_is_stale_until_the_writer_is_trusted() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        let conservative = table
            .index_staleness(&ReindexOptions::default())
            .expect("staleness");
        assert_eq!(
            conservative.awaiting_reanalysis, conservative.superfiles,
            "recording no revision, every superfile is an unknown: {conservative:?}"
        );

        let trusting = table
            .index_staleness(&ReindexOptions {
                trust_writer_analysis: true,
                ..ReindexOptions::default()
            })
            .expect("staleness");
        assert_eq!(
            trusting.awaiting_reanalysis, 0,
            "0.8.6 shipped the current chains, so nothing needs re-analysis: {trusting:?}"
        );
        assert!(
            trusting.unrepairable_columns.is_empty(),
            "and no column is reported unrepairable: {trusting:?}"
        );
    }

    fn entry(id: u128, fts: FtsStaleness) -> StaleSuperfile {
        StaleSuperfile {
            superfile_id: Uuid::from_u128(id),
            partition_key: vec![7],
            live_bytes: 1_024,
            fts,
            has_duplicated_region_keys: false,
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

    /// The bytes of the fixture's first superfile.
    fn first_fixture_superfile(table: &Supertable, storage: &Arc<dyn StorageProvider>) -> Bytes {
        let manifest = table.reader().expect("reader").manifest().clone();
        let entries = table
            .block_on_query(manifest.get_all_superfiles_loaded())
            .expect("load entries");
        let (bytes, _) = table
            .block_on_query(storage.get(&entries[0].storage_path()))
            .expect("read superfile");
        bytes
    }

    /// A footer is judged on the copy of each key this engine reads: sound
    /// when the manifest records the same regions, inconsistent the moment
    /// it records one elsewhere — whichever side is wrong.
    #[test]
    fn a_footer_the_manifest_contradicts_is_inconsistent() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);
        let bytes = first_fixture_superfile(&table, &storage);
        let offsets = build_subsection_offsets(&bytes).expect("subsection offsets");
        let reader = SuperfileReader::open(bytes).expect("open superfile");

        assert_eq!(footer_state(&reader, Some(&offsets)), FooterState::Sound);
        assert_eq!(footer_state(&reader, None), FooterState::Sound);

        let (at, len) = offsets.fts.expect("the fixture has an FTS region");
        let elsewhere = SubsectionOffsets {
            fts: Some((at + 1, len)),
            ..offsets
        };
        assert_eq!(
            footer_state(&reader, Some(&elsewhere)),
            FooterState::Inconsistent
        );
    }

    /// A stale copy of a region key ahead of the real one, the shape a
    /// carried rewrite once left, is a duplicate a rewrite can repair: the
    /// copy this engine reads still matches the file and the manifest.
    #[test]
    fn a_stale_region_copy_ahead_of_the_real_one_is_a_duplicate() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);
        let bytes = first_fixture_superfile(&table, &storage);
        let offsets = build_subsection_offsets(&bytes).expect("subsection offsets");
        let (at, _) = offsets.fts.expect("the fixture has an FTS region");

        let stale = (at + 1).to_string();
        let forged = with_forged_footer_kv(&bytes, &[(kv::FTS_OFFSET, &stale)]);
        let reader = SuperfileReader::open(forged).expect("the last copy still opens");
        assert_eq!(
            footer_state(&reader, Some(&offsets)),
            FooterState::DuplicatedKeys
        );
    }

    /// A duplicated region key on a table that is otherwise current is
    /// found and repaired: the footer is the only reason the file is
    /// rewritten, and the rewrite stores each key once.
    #[test]
    fn a_reindex_repairs_a_duplicated_region_key() {
        let dir = TempDir::new().expect("tempdir");
        copy_dir_recursive(&old_format_fts_fixture(), dir.path());
        let (_storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);
        table
            .reindex(&ReindexOptions::default())
            .expect("bring the fixture current");
        let survivors = hits(&table, "shared");

        // Plant the stale copy on one file of the now-current table.
        let manifest = table.reader().expect("reader").manifest().clone();
        let entries = table
            .block_on_query(manifest.get_all_superfiles_loaded())
            .expect("load entries");
        let path = dir.path().join(entries[0].storage_path());
        let bytes = fs::read(&path).expect("read superfile");
        let offsets = build_subsection_offsets(&Bytes::from(bytes.clone())).expect("offsets");
        let (at, _) = offsets.fts.expect("an FTS region");
        let stale = (at + 1).to_string();
        fs::write(
            &path,
            with_forged_footer_kv(&bytes, &[(kv::FTS_OFFSET, &stale)]),
        )
        .expect("write forged superfile");
        drop(table);
        let (storage, table) = open_old_format_fts_fixture(dir.path(), |o| o);

        let before = table
            .index_staleness(&ReindexOptions::default())
            .expect("assess");
        assert_eq!(before.needing_rewrite, 1, "{before:?}");
        assert_eq!(before.awaiting_reanalysis, 0, "{before:?}");
        assert!(before.inconsistent_footers.is_empty(), "{before:?}");

        let report = table
            .reindex(&ReindexOptions::default())
            .expect("repair the footer");
        assert_eq!(report.rewritten, 1, "{report:?}");
        for entry in table
            .block_on_query(
                table
                    .reader()
                    .expect("reader")
                    .manifest()
                    .get_all_superfiles_loaded(),
            )
            .expect("load entries")
        {
            let (bytes, _) = table
                .block_on_query(storage.get(&entry.storage_path()))
                .expect("read superfile");
            let reader = SuperfileReader::open(bytes).expect("open superfile");
            assert!(
                !has_duplicated_region_key(reader.parquet_metadata()),
                "{}: footer still stores a region key twice",
                entry.storage_path()
            );
        }
        assert_eq!(hits(&table, "shared"), survivors, "the repair moved rows");
        assert!(
            table
                .index_staleness(&ReindexOptions::default())
                .expect("assess the repaired table")
                .is_current()
        );
    }

    /// A footer storing a stale duplicate earns a rewrite under every
    /// mode, even with a current FTS index: the rewrite is what lays the
    /// footer out afresh.
    #[test]
    fn a_duplicated_region_key_earns_a_rewrite_under_every_mode() {
        let duplicated = StaleSuperfile {
            has_duplicated_region_keys: true,
            ..entry(1, FtsStaleness::default())
        };
        for (mode, repair) in [
            (ReindexMode::Rewrite, Repair::Layout),
            (ReindexMode::Auto, Repair::Layout),
            (ReindexMode::Reanalyze, Repair::Terms),
        ] {
            let jobs = plan_jobs(slice::from_ref(&duplicated), mode);
            assert_eq!(jobs.len(), 1, "{mode:?}");
            assert_eq!(jobs[0].1, repair, "{mode:?}");
        }
        assert!(!duplicated.is_current());
        assert!(entry(2, FtsStaleness::default()).is_current());
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
        assert_eq!(jobs[0].0.inputs, vec![Uuid::from_u128(1)]);
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

    /// `Auto` gives each superfile the cheapest repair that makes *it*
    /// current, rather than charging the whole run the most expensive one
    /// any file needs.
    ///
    /// This is the distinction the mode exists for: under `Reanalyze` a
    /// file that is only behind on its container is tokenized again for
    /// nothing, which costs far more than copying its postings.
    #[test]
    fn auto_repairs_each_superfile_by_what_that_file_is_behind_on() {
        let both = FtsStaleness {
            container: Some(4),
            analysis: behind_analysis().analysis,
        };
        let plan = plan_jobs(
            &[
                entry(1, behind_container()),
                entry(2, behind_analysis()),
                entry(3, both),
            ],
            ReindexMode::Auto,
        );
        let repairs: Vec<Repair> = plan.iter().map(|(_, r)| *r).collect();
        assert_eq!(
            repairs,
            vec![Repair::Layout, Repair::Terms, Repair::Terms],
            "only the container-stale file may take the cheap repair"
        );
    }

    /// `Reanalyze` re-analyzes even a file whose terms are current, which
    /// is the only thing it offers over `Auto`.
    #[test]
    fn reanalyze_rebuilds_terms_even_where_only_the_container_is_behind() {
        let plan = plan_jobs(&[entry(1, behind_container())], ReindexMode::Reanalyze);
        assert_eq!(
            plan.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
            vec![Repair::Terms]
        );
    }

    /// `Auto` leaves nothing stale: it plans both axes, where `Rewrite`
    /// plans only one.
    #[test]
    fn auto_plans_every_stale_superfile() {
        let stale = [entry(1, behind_container()), entry(2, behind_analysis())];
        assert_eq!(plan_jobs(&stale, ReindexMode::Auto).len(), 2);
        assert_eq!(plan_jobs(&stale, ReindexMode::Rewrite).len(), 1);
    }

    /// Every job takes exactly one input, so a rewrite never merges rows
    /// that were not already together.
    #[test]
    fn every_job_is_one_in_one_out() {
        let jobs = plan_jobs(
            &[entry(9, behind_container()), entry(4, behind_analysis())],
            ReindexMode::Rewrite,
        );
        assert!(jobs.iter().all(|(j, _)| j.inputs.len() == 1), "{jobs:?}");
        assert_eq!(jobs[0].0.partition_key, vec![7], "partition is carried");
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
