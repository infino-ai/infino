// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Which superfiles to merge, and into which jobs.
//!
//! Pure policy: every function here is a decision over
//! [`SuperfileStats`], with no I/O, no manifest and no async. The
//! schedule ([`super::schedule`]) gathers the stats and runs what
//! [`select`] returns.
//!
//! The packing is single-level and single-pass. A superfile already at
//! target size is done and never re-merged, unless enough of its rows are
//! deleted ([`RECLAIM_FULL_DELETED_PERCENT`]) to rewrite it without them.
//! Each candidate lands in **at most one** job, which is what lets the jobs
//! run independently. At most, not exactly: a candidate is dropped when it
//! is already at target size with few deletes or sealed by another
//! compactor, and a trailing group that clears no leg of the merge trigger
//! is left for a later pass. The
//! property the concurrent runner depends on is that no two jobs name the
//! same superfile, which [`prop_jobs_never_share_an_input`] pins.

use std::{collections::BTreeMap, mem};

use uuid::Uuid;

use crate::{
    config::CompactionSettings,
    superfile::builder::MERGE_DECODED_WINDOW_BYTES,
    supertable::manifest::{list::DrainedVersionRanges, listed_once},
};

/// Bytes in a mebibyte, the unit the settings are expressed in.
pub(super) const MIB: u64 = 1024 * 1024;

/// The most input bytes one merge takes under `cfg`: a compaction job's,
/// and a term-index fold's at optimize.
pub(crate) fn max_merge_input_bytes(cfg: &CompactionSettings) -> u64 {
    cfg.max_merge_input_mb.saturating_mul(MIB)
}

/// Bytes an indexed merge keeps per row of its output while it runs: the
/// survivor numbering of every input row, the document order it chooses
/// and that order's inverse. A merge with no index carries its rows
/// straight through and keeps none of it.
pub(super) const INDEXED_MERGE_ROW_BYTES: u64 = 16;

/// What a merge of a table holds while it runs, and the most the planner
/// may give one merge.
///
/// A merge opens its inputs whole and stages its output whole, so those
/// bytes are held from start to finish. Its rows pass through decoded,
/// and a decoded row costs the table's width whatever it holds: a null
/// string is still an offset, so a row of a table thousands of columns
/// wide decodes to tens of kilobytes however few of them it fills, which
/// its stored bytes, a bit per null, say nothing of. The merge holds a
/// window of those rows at a time ([`MERGE_DECODED_WINDOW_BYTES`] at the
/// most), so a wide merge is charged that window and a narrow one the rows
/// it has.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct MergeMemory {
    /// What one row of the table decodes to at the least
    /// (`null_row_bytes` of its scalar schema).
    pub(crate) row_bytes: u64,
    /// Whether merges keep per-row bookkeeping: the table has an index.
    pub(crate) indexed: bool,
    /// The most one planned merge may be expected to hold: what the host
    /// can give a merge. `None` where it cannot be read, which never
    /// bounds a job.
    pub(crate) limit: Option<u64>,
}

impl MergeMemory {
    /// What a merge of `input_bytes` stored, writing `output_bytes`, over
    /// `live_docs` rows is expected to hold once it is running. A floor
    /// rather than a model of everything a merge allocates (term
    /// cardinality and posting density add to an indexed one), and the
    /// host reading, not this, is what finally bounds how many run at once.
    pub(crate) fn footprint(&self, input_bytes: u64, output_bytes: u64, live_docs: u64) -> u64 {
        let bookkeeping = match self.indexed {
            true => live_docs.saturating_mul(INDEXED_MERGE_ROW_BYTES),
            false => 0,
        };
        let decoded = live_docs
            .saturating_mul(self.row_bytes)
            .min(MERGE_DECODED_WINDOW_BYTES);
        input_bytes
            .saturating_add(output_bytes)
            .saturating_add(bookkeeping)
            .saturating_add(decoded)
    }

    /// Whether a merge of that shape is within [`Self::limit`].
    fn admits(&self, input_bytes: u64, output_bytes: u64, live_docs: u64) -> bool {
        self.limit
            .is_none_or(|limit| self.footprint(input_bytes, output_bytes, live_docs) <= limit)
    }
}

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
    /// The smallest document id the superfile holds: the key compaction
    /// packs neighbours by, so merged outputs keep the disjoint, monotone
    /// id ranges ingest gave their inputs.
    pub id_min: i128,
    /// Already owned by another compaction so skip it.
    pub sealed_by_other: bool,
    /// Commit version the superfile was born at. A merged superfile carries
    /// the OLDEST input's `birth_version`, so user-table merge jobs must
    /// never mix inputs from opposite sides of the hidden drain watermark
    /// (see [`split_stats_at_drain_watermark`]).
    pub birth_version: u64,
    /// Whether the file holds some column in a type other than the
    /// table's. Such a file is rewritten by the next compaction whatever
    /// its size or fill, alone if need be, so a type change converges.
    pub stale_type: bool,
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

/// Split merge candidates at the hidden drain watermark: inputs whose
/// `birth_version` the hidden index has already drained versus inputs it has
/// not. A merged superfile is stamped with the OLDEST input `birth_version`
/// (see `run_compaction_job`), so a job mixing the two sides would inherit a
/// drained version and the drain's `!drained.contains(birth_version)` filter
/// would skip it — the undrained inputs' vectors would silently never enter
/// the hidden index (a permanent recall hole). Merging within either side is
/// safe: all-drained stays drained, all-undrained keeps an undrained version
/// and is drained as one source.
pub(super) fn split_stats_at_drain_watermark(
    stats: Vec<SuperfileStats>,
    drained: &DrainedVersionRanges,
) -> (Vec<SuperfileStats>, Vec<SuperfileStats>) {
    stats
        .into_iter()
        .partition(|s| drained.contains(s.birth_version))
}

/// A set of superfiles to merge into one new superfile.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CompactionJob {
    pub partition_key: Vec<u8>,
    pub inputs: Vec<Uuid>,
    /// Estimated size of the merged superfile.
    pub estimated_output_bytes: u64,
    /// Bytes of the inputs as stored. A merge opens every input whole and
    /// holds the bytes until it finishes, so this is the floor of what it
    /// costs the host from the moment it is admitted.
    pub input_bytes: u64,
    /// Rows the output will carry: the inputs' rows less their tombstones.
    /// An indexed merge keeps per-row bookkeeping for them.
    pub live_docs: u64,
    /// Memory this job may use, in MiB, from the settings the run was given.
    /// Carried on the job because the planner is where those settings are in
    /// scope; the merge reads it rather than the global configuration, so a
    /// caller that raises the budget reorders more documents together.
    ///
    /// `None` for a job no compaction run planned, which takes the
    /// configured value because there is no other.
    pub merge_input_mb: Option<u64>,
}

/// Plan compaction: pack each partition's small superfiles into
/// as many target-sized jobs as they fill. Leftovers that can't
/// reach the floor are left for next time. Each `superfile_id` must appear
/// once: `compact_one_table` drops repeats with [`listed_once`] before the
/// drain-watermark split.
#[cfg(test)]
pub fn select(superfiles: &[SuperfileStats], cfg: &CompactionSettings) -> Vec<CompactionJob> {
    select_within(superfiles, cfg, &MergeMemory::default())
}

/// [`select`], with no job of more than one input planned past what
/// `memory` admits: a merge expected to hold more than the host can give
/// it is cut where its inputs reach that, as a job is cut at the target
/// size. One stale-typed superfile that alone exceeds it still starts a
/// job, which is what lets a stale-typed file be rewritten on any host; one
/// rewritten only to drop its deleted rows does not.
pub(crate) fn select_within(
    superfiles: &[SuperfileStats],
    cfg: &CompactionSettings,
    memory: &MergeMemory,
) -> Vec<CompactionJob> {
    debug_assert_eq!(
        listed_once(superfiles, |s| s.superfile_id).count(),
        superfiles.len(),
        "select was given a superfile_id twice"
    );

    let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
    // Size leg of the merge trigger: a job's combined live bytes must reach this
    // fraction of the target. The count leg (`min_superfiles_for_merge`) fires
    // independently, so a partition fragmented into many tiny superfiles still
    // consolidates even when it sits far below this floor.
    let min_output_bytes =
        (target_bytes as u128 * cfg.min_fill_percent.clamp(0, 100) as u128 / 100) as u64;
    // Count leg: merge once a partition has this many sub-target superfiles.
    // Clamped to >= 2 — merging fewer than two inputs is a no-op rewrite, so a
    // misconfigured smaller value is raised rather than rejected.
    let min_superfiles_for_merge = cfg.min_superfiles_for_merge.max(2) as usize;
    let max_input_bytes = max_merge_input_bytes(cfg);

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
            min_superfiles_for_merge,
            max_input_bytes,
            memory,
            &mut jobs,
        );
    }
    jobs
}

/// A superfile at least this share deleted is rewritten by the next pass
/// whatever its neighbours amount to: packing by id would otherwise leave
/// it waiting for neighbours that may never fill a job, and the space its
/// deleted rows hold with it.
const RECLAIM_DELETED_PERCENT: u64 = 50;

/// A superfile already at target size is rewritten, without its deleted
/// rows, once at least this share of them is deleted; below the target the
/// share is [`RECLAIM_DELETED_PERCENT`].
///
/// A table kept current by updates deletes old versions of rows from
/// superfiles that are already full, so their deleted rows would otherwise
/// stay for good: storage keeps paying for them, every scan of the file
/// decodes past them, and aggregates the statistics would answer have to
/// correct for them or scan the file. A rewrite reads and writes the whole
/// file to drop them, so the share sets the cost per byte reclaimed: at a
/// fifth deleted a rewrite moves about five bytes for each one it frees.
/// Lower, a file under steady updates is rewritten ever more often for
/// less; higher, a fifth to a half of a full file's bytes sit dead between
/// rewrites. A sub-target file keeps the higher bar: it is merged with its
/// neighbours as they fill a job, its deleted rows dropped then.
///
/// A rewrite packs the file with the neighbours its live bytes leave room
/// for, so its output, as large in live bytes as the file was, packs with
/// the same neighbours as before and is not merged again by the next pass.
/// It is when the input cap or the memory a merge is given, which count the
/// file's deleted rows too, kept a neighbour out that the smaller output
/// takes it in next pass: the merge the cap had put off.
const RECLAIM_FULL_DELETED_PERCENT: u64 = 20;

/// Whether at least `percent` of `s`'s rows are deleted.
fn deleted_at_least(s: &SuperfileStats, percent: u64) -> bool {
    s.tombstoned_docs as u128 * 100 >= s.n_docs.max(1) as u128 * percent as u128
}

/// Whether enough of `s`'s rows are deleted that the next pass rewrites it,
/// alone if no neighbour joins it: [`RECLAIM_FULL_DELETED_PERCENT`] at the
/// target size or above it, [`RECLAIM_DELETED_PERCENT`] below.
fn reclaim_due(s: &SuperfileStats, target_bytes: u64) -> bool {
    let percent = match s.size_bytes >= target_bytes {
        true => RECLAIM_FULL_DELETED_PERCENT,
        false => RECLAIM_DELETED_PERCENT,
    };
    deleted_at_least(s, percent)
}

fn pack_partition(
    key: &[u8],
    segs: Vec<&SuperfileStats>,
    target_bytes: u64,
    min_output_bytes: u64,
    min_superfiles_for_merge: usize,
    max_input_bytes: u64,
    memory: &MergeMemory,
    jobs: &mut Vec<CompactionJob>,
) {
    // Exclude superfiles already at target size — they are done and
    // re-compacting them gains nothing — unless enough of their rows are
    // deleted to rewrite them without those.
    let mut candidates: Vec<&SuperfileStats> = segs
        .into_iter()
        .filter(|s| {
            !s.sealed_by_other
                && (s.size_bytes < target_bytes || s.stale_type || reclaim_due(s, target_bytes))
        })
        .collect();

    // Neighbours by document id pack together: ingest gives each superfile
    // a contiguous id range disjoint from every other's, and a job over
    // adjacent ranges produces an output whose range is as disjoint and as
    // monotone as its inputs'. A ranked search profits from that — a
    // superfile whose best document can only tie the k-th score is skipped
    // when every id it holds is past the tied document's, which ranges
    // that do not interleave make the common case. Every sub-target
    // superfile is a candidate regardless; a deleted-heavy one no longer
    // leads the pass, and is instead rewritten with whatever neighbours
    // its job has gathered (see [`RECLAIM_DELETED_PERCENT`]).
    candidates.sort_by(|a, b| {
        a.id_min
            .cmp(&b.id_min)
            .then(a.superfile_id.cmp(&b.superfile_id))
    });

    // The same setting the packing limit was built from, back in the unit
    // the merge states it in.
    let merge_input_mb = Some(max_input_bytes / MIB);
    let mut pending = PendingJob::default();
    for s in candidates {
        if !pending.fits(s, target_bytes, max_input_bytes, memory) {
            pending.emit(
                key,
                min_output_bytes,
                min_superfiles_for_merge,
                merge_input_mb,
                jobs,
            );
        }
        // A file that starts a job may be rewritten on its own only if one
        // merge can take it: one larger than the input cap, or than the
        // memory a merge is given, is never reclaimed alone.
        let alone_fits = pending.members.is_empty() && pending.admits(s, max_input_bytes, memory);
        pending.push(s, target_bytes);
        pending.reclaim_alone = alone_fits && reclaim_due(s, target_bytes);
    }
    pending.emit(
        key,
        min_output_bytes,
        min_superfiles_for_merge,
        merge_input_mb,
        jobs,
    );
}

/// What a job keeps of each input to order the merge; see [`PendingJob::emit`].
#[derive(Clone, Copy)]
struct Member {
    id: Uuid,
    tombstoned_docs: u64,
    n_docs: u64,
    size_bytes: u64,
}

#[derive(Default)]
struct PendingJob {
    members: Vec<Member>,
    live_bytes: u64,
    raw_bytes: u64,
    live_docs: u64,
    /// Whether an input holds a column in a type the table has moved on
    /// from; the job then runs however small.
    stale_type: bool,
    /// Whether an input is deleted enough to be rewritten (see
    /// [`reclaim_due`]); the job then runs however small, to reclaim the
    /// space.
    reclaiming: bool,
    /// Whether the job's first input may be rewritten on its own: it is
    /// deleted enough, and one merge can take it.
    reclaim_alone: bool,
}

impl PendingJob {
    fn fits(
        &self,
        s: &SuperfileStats,
        target_bytes: u64,
        max_input_bytes: u64,
        memory: &MergeMemory,
    ) -> bool {
        self.live_bytes + s.live_bytes() <= target_bytes && self.admits(s, max_input_bytes, memory)
    }

    /// Whether one merge can take `s` with the inputs it has: within the
    /// input cap and the memory a merge is given.
    fn admits(&self, s: &SuperfileStats, max_input_bytes: u64, memory: &MergeMemory) -> bool {
        self.raw_bytes + s.size_bytes <= max_input_bytes
            && memory.admits(
                self.raw_bytes + s.size_bytes,
                self.live_bytes + s.live_bytes(),
                self.live_docs + s.live_docs(),
            )
    }

    fn push(&mut self, s: &SuperfileStats, target_bytes: u64) {
        self.raw_bytes += s.size_bytes;
        self.members.push(Member {
            id: s.superfile_id,
            tombstoned_docs: s.tombstoned_docs,
            n_docs: s.n_docs,
            size_bytes: s.size_bytes,
        });
        self.live_bytes += s.live_bytes();
        self.live_docs += s.live_docs();
        self.stale_type |= s.stale_type;
        self.reclaiming |= reclaim_due(s, target_bytes);
    }

    /// Emit a CompactionJob when the pending inputs clear either leg of the
    /// merge trigger — size OR count:
    /// - size: `>= 2` inputs and live bytes reach `min_output_bytes`;
    /// - count: `>= min_superfiles_for_merge` inputs (already `>= 2`), which
    ///   fires even when the live bytes sit far below the size floor.
    ///
    /// A job holding a stale-typed input runs however small, one input
    /// included: the rewrite converges the type. One holding an input
    /// deleted enough runs however small too, one input included when one
    /// merge can take it: the rewrite reclaims the deleted rows' space.
    fn emit(
        &mut self,
        key: &[u8],
        min_output_bytes: u64,
        min_superfiles_for_merge: usize,
        merge_input_mb: Option<u64>,
        jobs: &mut Vec<CompactionJob>,
    ) {
        let size_ready = self.members.len() >= 2 && self.live_bytes >= min_output_bytes;
        let count_ready = self.members.len() >= min_superfiles_for_merge;
        let converging = self.stale_type && !self.members.is_empty();
        let reclaiming = match self.members.len() {
            0 => false,
            1 => self.reclaim_alone,
            _ => self.reclaiming,
        };
        if size_ready || count_ready || converging || reclaiming {
            // A job's inputs were gathered in id order; they merge
            // most-deleted first, then smallest, then by id. The merged
            // superfile's document order starts from its inputs'
            // concatenation before the reorder refines it, and a start in
            // id order — ingest order — leaves the ranked kernels pruning
            // a fifth less on the output: on a 5.3M-document table
            // compacted to one superfile, the same inputs merged in id
            // order answered at 1.20 the time at k=10 and 1.13 at k=100,
            // every query shape alike, against this order. The output's
            // id range is the same either way, so jobs still produce
            // superfiles whose ranges do not interleave.
            let mut members = mem::take(&mut self.members);
            members.sort_by(|a, b| {
                let lhs = a.tombstoned_docs as u128 * b.n_docs.max(1) as u128;
                let rhs = b.tombstoned_docs as u128 * a.n_docs.max(1) as u128;
                rhs.cmp(&lhs)
                    .then(a.size_bytes.cmp(&b.size_bytes))
                    .then(a.id.cmp(&b.id))
            });
            jobs.push(CompactionJob {
                partition_key: key.to_vec(),
                inputs: members.into_iter().map(|m| m.id).collect(),
                estimated_output_bytes: self.live_bytes,
                input_bytes: self.raw_bytes,
                live_docs: self.live_docs,
                merge_input_mb,
            });
        }
        *self = PendingJob::default();
    }
}

#[cfg(test)]
pub(in crate::supertable::optimize::compact) mod tests {
    use std::{
        collections::{HashMap, HashSet},
        slice::from_ref,
    };

    use proptest::{bool::weighted, option::of as optional, prelude::*};
    use uuid::Uuid;

    use super::*;
    use crate::config::CompactionSettings;

    pub(in crate::supertable::optimize::compact) fn mib(n: u64) -> u64 {
        n * MIB
    }

    pub(in crate::supertable::optimize::compact) fn seg(
        id: u128,
        size_mib: u64,
        n_docs: u64,
        tombstoned: u64,
    ) -> SuperfileStats {
        SuperfileStats {
            superfile_id: Uuid::from_u128(id),
            partition_key: Vec::new(),
            size_bytes: mib(size_mib),
            n_docs,
            tombstoned_docs: tombstoned,
            id_min: id as i128,
            sealed_by_other: false,
            birth_version: 0,
            stale_type: false,
        }
    }

    /// Each job carries the memory budget of the run that planned it, so the
    /// merge sizes its reordering from the settings the caller passed rather
    /// than from the configured default.
    #[test]
    fn a_job_carries_the_budget_of_the_run_that_planned_it() {
        let stats = vec![seg(1, 1, 1000, 0), seg(2, 1, 1000, 0)];
        for mb in [512u64, 3072, 16_384] {
            let cfg = CompactionSettings {
                max_merge_input_mb: mb,
                // Two fragments are enough to plan on, so the job exists to
                // be inspected whatever the budget is.
                min_superfiles_for_merge: 2,
                ..CompactionSettings::default()
            };
            let jobs = select(&stats, &cfg);
            assert!(!jobs.is_empty(), "{mb} MiB should still plan a job");
            for job in &jobs {
                assert_eq!(
                    job.merge_input_mb,
                    Some(mb),
                    "a job planned at {mb} MiB should carry it"
                );
            }
        }
    }

    /// Two mergeable fragments on opposite sides of the drain watermark must
    /// land in different selection groups: a single mixed job would stamp the
    /// merged superfile with the drained input's (older) `birth_version` and
    /// the drain would skip the undrained rows forever.
    #[test]
    fn drain_watermark_partition_never_mixes_drained_and_undrained() {
        // Watermark: versions 0..=10 drained.
        let drained = DrainedVersionRanges::from_intervals(vec![(0, 10)]).expect("valid intervals");
        let mut a = seg(1, 1, 1000, 0);
        a.birth_version = 5; // drained
        let mut b = seg(2, 1, 1000, 0);
        b.birth_version = 20; // undrained
        let mut c = seg(3, 1, 1000, 0);
        c.birth_version = 21; // undrained

        // Sanity: without the watermark split, selection would happily merge
        // all three into one job — the exact F1 hazard.
        let all = vec![a.clone(), b.clone(), c.clone()];
        let cfg = CompactionSettings {
            target_superfile_size_mb: 2048,
            min_fill_percent: 0,
            ..CompactionSettings::default()
        };
        let mixed = select(&all, &cfg);
        assert_eq!(mixed.len(), 1);
        assert_eq!(mixed[0].inputs.len(), 3, "guard: unsplit selection mixes");

        let (drained_side, undrained_side) = split_stats_at_drain_watermark(all, &drained);
        assert_eq!(
            drained_side
                .iter()
                .map(|s| s.superfile_id)
                .collect::<Vec<_>>(),
            vec![Uuid::from_u128(1)]
        );
        assert_eq!(undrained_side.len(), 2);
        // Group-wise selection: the drained side alone can't merge (one
        // input); the undrained side merges its two fragments.
        assert!(select(&drained_side, &cfg).is_empty());
        let jobs = select(&undrained_side, &cfg);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs.len(), 2);
        assert!(
            !jobs[0].inputs.contains(&Uuid::from_u128(1)),
            "undrained job must not contain the drained input"
        );
    }

    pub(in crate::supertable::optimize::compact) fn default_cfg() -> CompactionSettings {
        CompactionSettings::default() // 1 GiB target, 80% floor
    }

    #[test]
    fn listed_once_keeps_the_first_copy_of_each_id() {
        // Superfile 1 listed at versions 5 and 9, superfile 2 once.
        //  - the repeat is dropped, so the plan holds each superfile once.
        //  - the kept copy is the earlier one, at version 5, so a merge of it
        //    inherits the earlier `birth_version`.
        let mut first = seg(1, 1, 1000, 0);
        first.birth_version = 5;
        let mut repeat = seg(1, 1, 1000, 0);
        repeat.birth_version = 9;
        let other = seg(2, 1, 1000, 0);
        let kept: Vec<SuperfileStats> =
            listed_once(vec![first.clone(), repeat, other.clone()], |s| {
                s.superfile_id
            })
            .collect();
        assert_eq!(kept, vec![first, other]);
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

    /// Candidates pack by document id, whatever order they are listed in
    /// and however deleted: a job's inputs are id-adjacent, so its output's
    /// range does not interleave with any other output's.
    /// Packing by id must not strand a mostly deleted superfile: four
    /// 300 MiB superfiles fill a job without the 900 MiB one at the
    /// highest id that is 72% deleted, which the reclaim leg then rewrites
    /// with the neighbour left over. Alone it is rewritten alone, so a
    /// superfile no neighbour joins still drops its deleted rows; under the
    /// share it waits for its neighbours as any other does.
    #[test]
    fn a_mostly_deleted_superfile_is_rewritten_by_the_next_pass() {
        let mut segs: Vec<_> = (0..4).map(|i| seg(i, 300, 1000, 0)).collect();
        let dead_heavy = seg(4, 900, 1000, 720);
        segs.push(dead_heavy.clone());
        let jobs = select(&segs, &default_cfg());
        let holder = jobs
            .iter()
            .find(|j| j.inputs.contains(&dead_heavy.superfile_id))
            .expect("the mostly deleted superfile is in a job");
        assert_eq!(holder.inputs.len(), 2, "with the neighbour left over");
        let alone = select(&segs[4..], &default_cfg());
        assert_eq!(alone.len(), 1, "alone");
        assert_eq!(alone[0].inputs, vec![dead_heavy.superfile_id]);
        segs[4] = seg(4, 900, 1000, 100);
        let jobs = select(&segs, &default_cfg());
        assert!(
            jobs.iter()
                .all(|j| !j.inputs.contains(&segs[4].superfile_id)),
            "lightly deleted, it waits with its neighbour: {jobs:?}"
        );
    }

    #[test]
    fn packs_id_adjacent_superfiles_together() {
        // Ten sub-target superfiles listed out of id order, one of them
        // mostly deleted; each job fills to the target from the lowest ids.
        let order = [7u128, 2, 9, 0, 4, 1, 8, 3, 6, 5];
        let segs: Vec<_> = order
            .iter()
            .map(|&i| seg(i, 300, 1000, if i == 9 { 900 } else { 0 }))
            .collect();
        let jobs = select(&segs, &default_cfg());
        assert!(jobs.len() >= 2, "several jobs");
        let mut seen = 0u128;
        for job in &jobs {
            let ids: Vec<u128> = job.inputs.iter().map(|u| u.as_u128()).collect();
            let mut sorted = ids.clone();
            sorted.sort_unstable();
            assert!(
                sorted.windows(2).all(|w| w[0] + 1 == w[1]) && sorted[0] == seen,
                "job inputs are the next ids: {ids:?} after {seen}"
            );
            seen = sorted[sorted.len() - 1] + 1;
        }
    }

    #[test]
    fn a_job_merges_most_deleted_first_then_smallest_then_by_id() {
        // Four id-adjacent inputs of one job: the deleted-heavy one leads,
        // then the two clean ones smallest first, then id breaks the tie.
        let segs = vec![
            seg(0, 300, 1000, 0),
            seg(1, 200, 1000, 0),
            seg(2, 300, 1000, 600),
            seg(3, 200, 1000, 0),
        ];
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 1, "one job: {jobs:?}");
        let ids: Vec<u128> = jobs[0].inputs.iter().map(|u| u.as_u128()).collect();
        assert_eq!(ids, vec![2, 1, 3, 0]);
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
        assert!(p.fits(&a, target, max_memory, &MergeMemory::default()));
        p.push(&a, target);
        assert_eq!(p.live_bytes, mib(60));
        assert_eq!(p.members.len(), 1);
        // A second 60 MiB superfile would overflow the 100 MiB target.
        let b = seg(2, 60, 1000, 0);
        assert!(!p.fits(&b, target, max_memory, &MergeMemory::default()));
        // A 40 MiB superfile fits exactly to the boundary.
        let c = seg(3, 40, 1000, 0);
        assert!(p.fits(&c, target, max_memory, &MergeMemory::default()));
    }

    #[test]
    fn pending_job_fits_respects_max_memory_even_under_target() {
        // live_bytes fits comfortably under target, but raw size_bytes
        // (pre-tombstone) would blow past a tight memory ceiling.
        let target = mib(1000);
        let max_memory = mib(100);
        let mut p = PendingJob::default();
        let a = seg(1, 60, 1000, 0); // 60 MiB raw, 60 MiB live
        assert!(p.fits(&a, target, max_memory, &MergeMemory::default()));
        p.push(&a, target);
        let b = seg(2, 60, 1000, 0); // would push raw to 120 MiB > 100 MiB cap
        assert!(!p.fits(&b, target, max_memory, &MergeMemory::default()));
    }

    /// A decoded row of a table thousands of columns wide, nearly all null.
    const WIDE_ROW_BYTES: u64 = 20_000;

    /// Superfiles of a wide table: small as stored, as a table of mostly
    /// nulls is, with many rows each.
    fn wide_segs() -> Vec<SuperfileStats> {
        (1..=WIDE_SEGS).map(|i| seg(i, 4, 3_000, 0)).collect()
    }

    /// Superfiles in the wide fixture.
    const WIDE_SEGS: u128 = 100;

    /// A wide table's job is cut where its expected footprint would pass
    /// what the host can give a merge, every input still lands in exactly
    /// one job, and with no limit the same superfiles make one job.
    #[test]
    fn a_wide_tables_jobs_fit_the_memory_a_merge_is_given() {
        let cfg = CompactionSettings {
            min_superfiles_for_merge: 2,
            ..CompactionSettings::default()
        };
        let segs = wide_segs();
        let unbounded = MergeMemory {
            row_bytes: WIDE_ROW_BYTES,
            ..MergeMemory::default()
        };
        let whole = select_within(&segs, &cfg, &unbounded);
        assert_eq!(
            whole.len(),
            1,
            "stored bytes alone put all of it in one job"
        );

        let limit = mib(1_024);
        let bounded = MergeMemory {
            limit: Some(limit),
            ..unbounded
        };
        let jobs = select_within(&segs, &cfg, &bounded);
        assert!(jobs.len() > 1, "the limit cuts the table into several jobs");
        let mut seen = HashSet::new();
        for job in &jobs {
            assert!(job.inputs.len() >= 2);
            let footprint =
                bounded.footprint(job.input_bytes, job.estimated_output_bytes, job.live_docs);
            assert!(
                footprint <= limit,
                "a job expected to hold {footprint} bytes"
            );
            for id in &job.inputs {
                assert!(seen.insert(*id), "{id} is in two jobs");
            }
        }
        // Every superfile still merges: the limit cuts, it drops nothing.
        assert_eq!(seen.len() as u128, WIDE_SEGS);
    }

    /// A limit no merge of two inputs fits plans nothing rather than a job
    /// past it; a stale-typed superfile is still rewritten alone.
    #[test]
    fn a_limit_no_pair_fits_plans_only_the_rewrites() {
        let cfg = CompactionSettings {
            min_superfiles_for_merge: 2,
            ..CompactionSettings::default()
        };
        let mut segs = wide_segs();
        segs[0].stale_type = true;
        let memory = MergeMemory {
            row_bytes: WIDE_ROW_BYTES,
            indexed: false,
            limit: Some(1),
        };
        let jobs = select_within(&segs, &cfg, &memory);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs, vec![segs[0].superfile_id]);
    }

    // ---- job disjointness (the invariant concurrent merging rests on) ----

    /// Upper bound on generated superfiles per case. Large enough that the
    /// packer emits several jobs per partition, small enough to keep a
    /// 256-case proptest run under a second.
    const PROP_MAX_SUPERFILES: usize = 40;
    /// Distinct partition keys a generated case spreads its superfiles over,
    /// so the multi-partition path is exercised and not just one packer run.
    const PROP_PARTITIONS: u64 = 3;

    prop_compose! {
        /// One superfile. `index` supplies the id, so a generated vector
        /// never repeats one -- `select` debug-asserts that its input is
        /// already deduped, and duplicate inputs are a separate, upstream
        /// concern (`listed_once` in the schedule).
        fn arb_stats(index: usize)(
            partition in 0u64..PROP_PARTITIONS,
            size_mib in 0u64..3000,
            n_docs in 0u64..10_000,
            tombstone_pct in 0u64..=100,
            sealed_by_other in any::<bool>(),
            birth_version in 0u64..100,
            stale_type in weighted(PROP_STALE_SHARE),
        ) -> SuperfileStats {
            SuperfileStats {
                superfile_id: Uuid::from_u128(index as u128 + 1),
                partition_key: partition.to_le_bytes().to_vec(),
                size_bytes: mib(size_mib),
                n_docs,
                tombstoned_docs: n_docs * tombstone_pct / 100,
                id_min: index as i128,
                sealed_by_other,
                birth_version,
                stale_type,
            }
        }
    }

    /// Share of generated superfiles holding a column in a stale type.
    const PROP_STALE_SHARE: f64 = 0.1;

    fn arb_stats_vec() -> impl Strategy<Value = Vec<SuperfileStats>> {
        (0..=PROP_MAX_SUPERFILES).prop_flat_map(|len| (0..len).map(arb_stats).collect::<Vec<_>>())
    }

    prop_compose! {
        /// Settings across the whole trigger space: fill floors from "merge
        /// anything" to "merge nothing", count legs below and above the
        /// clamp, and memory ceilings tight enough to cut a job short.
        fn arb_cfg()(
            target_superfile_size_mb in 1u64..4096,
            min_fill_percent in 0u8..=100,
            min_superfiles_for_merge in 0u64..12,
            max_merge_input_mb in 1u64..8192,
        ) -> CompactionSettings {
            CompactionSettings {
                target_superfile_size_mb,
                min_fill_percent,
                min_superfiles_for_merge,
                max_merge_input_mb,
                ..CompactionSettings::default()
            }
        }
    }

    proptest! {
        /// The property the concurrent job runner rests on: the jobs a single
        /// `select` returns never name the same superfile twice, so two jobs
        /// can seal, merge and commit without coordinating.
        ///
        /// Coverage is deliberately NOT asserted -- a candidate legitimately
        /// reaches no job at all (already at target size, sealed by another
        /// compactor, or a trailing group below both trigger legs).
        #[test]
        fn prop_jobs_never_share_an_input(
            stats in arb_stats_vec(),
            cfg in arb_cfg(),
        ) {
            let jobs = select(&stats, &cfg);

            let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
            let candidates: HashSet<Uuid> = stats.iter().map(|s| s.superfile_id).collect();
            let by_id: HashMap<Uuid, &SuperfileStats> =
                stats.iter().map(|s| (s.superfile_id, s)).collect();

            let max_input_bytes = max_merge_input_bytes(&cfg);
            let mut claimed: HashSet<Uuid> = HashSet::new();
            for job in &jobs {
                // One input alone is a rewrite that converges a type or
                // drops deleted rows, of a file one merge can take.
                match job.inputs.as_slice() {
                    [] => prop_assert!(false, "a job of no input"),
                    [only] => {
                        let s = by_id.get(only).expect("input is a candidate");
                        prop_assert!(
                            s.stale_type
                                || (reclaim_due(s, target_bytes) && s.size_bytes <= max_input_bytes),
                            "a job of one input rewrites nothing: {:?}",
                            job
                        );
                    }
                    _ => {}
                }
                for id in &job.inputs {
                    prop_assert!(
                        claimed.insert(*id),
                        "superfile {id} appears in two jobs; concurrent jobs would \
                         seal and merge the same input"
                    );
                    prop_assert!(
                        candidates.contains(id),
                        "job names {id}, which was never offered to select"
                    );
                    // Each input must also have survived the packer's filters,
                    // which is the other half of "no two compactors touch it".
                    let s = by_id.get(id).expect("input is a candidate");
                    prop_assert!(!s.sealed_by_other, "sealed input {id} was packed");
                    prop_assert!(
                        s.size_bytes < target_bytes || s.stale_type || reclaim_due(s, target_bytes),
                        "at-target input {id} with few deletes was packed"
                    );
                    prop_assert_eq!(
                        &s.partition_key,
                        &job.partition_key,
                        "job mixes partitions"
                    );
                }
            }
        }
    }

    proptest! {
        /// An at-target superfile is planned exactly when at least
        /// [`RECLAIM_FULL_DELETED_PERCENT`] of it is deleted and one merge
        /// can take it, under the input cap and the memory a merge is given.
        #[test]
        fn prop_full_superfiles_are_reclaimed_past_the_share_within_the_caps(
            stats in arb_stats_vec(),
            cfg in arb_cfg(),
            limit_mib in optional(1u64..8192),
        ) {
            let memory = MergeMemory {
                limit: limit_mib.map(mib),
                ..MergeMemory::default()
            };
            let jobs = select_within(&stats, &cfg, &memory);
            let target_bytes = cfg.target_superfile_size_mb.saturating_mul(MIB);
            let max_input_bytes = max_merge_input_bytes(&cfg);
            let job_of: HashMap<Uuid, &CompactionJob> = jobs
                .iter()
                .flat_map(|job| job.inputs.iter().map(move |id| (*id, job)))
                .collect();
            for job in &jobs {
                if job.inputs.len() > 1 {
                    prop_assert!(job.input_bytes <= max_input_bytes, "over the input cap");
                    prop_assert!(
                        memory.footprint(job.input_bytes, job.estimated_output_bytes, job.live_docs)
                            <= memory.limit.unwrap_or(u64::MAX),
                        "past the memory a merge is given"
                    );
                }
            }
            let full = stats
                .iter()
                .filter(|s| s.size_bytes >= target_bytes && !s.sealed_by_other && !s.stale_type);
            for s in full {
                let due = deleted_at_least(s, RECLAIM_FULL_DELETED_PERCENT)
                    && s.size_bytes <= max_input_bytes
                    && memory.admits(s.size_bytes, s.live_bytes(), s.live_docs());
                prop_assert_eq!(
                    job_of.contains_key(&s.superfile_id),
                    due,
                    "{:?} in {:?}",
                    s,
                    job_of.get(&s.superfile_id)
                );
            }
        }
    }

    fn reclaim_cfg() -> CompactionSettings {
        CompactionSettings {
            target_superfile_size_mb: 100,
            ..CompactionSettings::default()
        }
    }

    /// A superfile at the target size is rewritten once a fifth of its rows
    /// are deleted, and not before: alone, or with the neighbours its live
    /// rows leave room for.
    #[test]
    fn a_full_superfile_is_rewritten_once_a_fifth_is_deleted() {
        let cfg = reclaim_cfg();
        let below = seg(1, 100, 1_000, 199);
        assert!(select(from_ref(&below), &cfg).is_empty());
        let full = seg(1, 100, 1_000, 200);
        let jobs = select(from_ref(&full), &cfg);
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        assert_eq!(jobs[0].inputs, vec![full.superfile_id]);
        assert_eq!(jobs[0].live_docs, 800);
        assert_eq!(jobs[0].estimated_output_bytes, mib(80));

        let neighbour = seg(2, 10, 1_000, 0);
        let jobs = select(&[below, neighbour.clone()], &cfg);
        assert!(jobs.is_empty(), "under the share nothing is due: {jobs:?}");
        let jobs = select(&[full.clone(), neighbour.clone()], &cfg);
        assert_eq!(jobs.len(), 1, "{jobs:?}");
        let mut inputs = jobs[0].inputs.clone();
        inputs.sort();
        assert_eq!(inputs, vec![full.superfile_id, neighbour.superfile_id]);
    }

    /// The superfiles after `jobs` ran over `stats`: each job's inputs
    /// replaced by one superfile of its live rows, with no deletes.
    fn after_pass(stats: &[SuperfileStats], jobs: &[CompactionJob]) -> Vec<SuperfileStats> {
        let merged: HashSet<Uuid> = jobs.iter().flat_map(|job| job.inputs.clone()).collect();
        let mut next: Vec<SuperfileStats> = stats
            .iter()
            .filter(|s| !merged.contains(&s.superfile_id))
            .cloned()
            .collect();
        for (at, job) in jobs.iter().enumerate() {
            let id_min = stats
                .iter()
                .filter(|s| job.inputs.contains(&s.superfile_id))
                .map(|s| s.id_min)
                .min()
                .expect("a job has inputs");
            next.push(SuperfileStats {
                superfile_id: Uuid::from_u128(u128::MAX - at as u128),
                size_bytes: job.estimated_output_bytes,
                n_docs: job.live_docs,
                tombstoned_docs: 0,
                id_min,
                ..seg(0, 0, 0, 0)
            });
        }
        next
    }

    /// Over two passes, a full superfile a quarter deleted is rewritten
    /// once: alone, its output is not merged with the same neighbours the
    /// next pass, and packed with a neighbour, that job's output is left
    /// alone too. Only when the input cap kept the neighbour out does the
    /// smaller output take it in next pass.
    #[test]
    fn a_reclaimed_superfile_is_rewritten_once_over_two_passes() {
        let cfg = reclaim_cfg();
        let full = seg(3, 100, 1_000, 250);
        for stats in [
            vec![seg(1, 30, 1_000, 0), seg(2, 30, 1_000, 0), full.clone()],
            vec![full.clone(), seg(4, 10, 1_000, 0)],
        ] {
            let first = select(&stats, &cfg);
            assert_eq!(first.len(), 1, "{first:?}");
            assert!(first[0].inputs.contains(&full.superfile_id));
            let second = select(&after_pass(&stats, &first), &cfg);
            assert!(second.is_empty(), "rewritten again: {second:?}");
        }

        let mut capped = reclaim_cfg();
        capped.max_merge_input_mb = 105;
        let stats = vec![full.clone(), seg(4, 10, 1_000, 0)];
        let first = select(&stats, &capped);
        assert_eq!(first.len(), 1);
        assert_eq!(
            first[0].inputs,
            vec![full.superfile_id],
            "the cap kept them apart"
        );
        let second = select(&after_pass(&stats, &first), &capped);
        assert_eq!(second.len(), 1, "the smaller output takes the neighbour in");
    }

    /// A full superfile larger than one merge may read, or than the memory
    /// a merge is given, stays as it is however deleted.
    #[test]
    fn a_full_superfile_one_merge_cannot_take_stays() {
        let mut cfg = reclaim_cfg();
        cfg.max_merge_input_mb = 150;
        let huge = seg(1, 200, 1_000, 900);
        assert!(
            select(from_ref(&huge), &cfg).is_empty(),
            "over the input cap"
        );
        cfg.max_merge_input_mb = 400;
        assert_eq!(select(from_ref(&huge), &cfg).len(), 1, "within it");
        let tight = MergeMemory {
            limit: Some(mib(100)),
            ..MergeMemory::default()
        };
        assert!(
            select_within(from_ref(&huge), &cfg, &tight).is_empty(),
            "past the memory a merge is given"
        );
    }

    /// What a reclaim writes has no deleted rows, so the next pass never
    /// selects it again.
    #[test]
    fn a_reclaimed_superfile_is_not_selected_again() {
        let cfg = reclaim_cfg();
        let full = seg(1, 150, 1_000, 300);
        let jobs = select(from_ref(&full), &cfg);
        assert_eq!(jobs.len(), 1);
        let job = &jobs[0];
        let output = SuperfileStats {
            superfile_id: Uuid::from_u128(3),
            size_bytes: job.estimated_output_bytes,
            n_docs: job.live_docs,
            tombstoned_docs: 0,
            ..seg(3, 0, 0, 0)
        };
        assert!(output.size_bytes >= mib(cfg.target_superfile_size_mb));
        assert!(select(from_ref(&output), &cfg).is_empty());
    }

    #[test]
    fn pending_job_emit_requires_two_inputs_without_a_reclaim() {
        // A single input with no deleted rows to reclaim never emits, even if
        // it reaches the fill floor and the count trigger (emit takes a
        // pre-clamped count of 2, so one input clears neither the size nor
        // the count leg). Only a reclaim emits a job of one input.
        let mut jobs = Vec::new();
        let mut p = PendingJob::default();
        p.push(&seg(1, 200, 1000, 0), mib(1024));
        p.emit(&[], 0, 2, None, &mut jobs);
        assert!(jobs.is_empty(), "single-input job must not emit");
        // Reset to default after emit attempt.
        assert_eq!(p.members.len(), 0);
        assert_eq!(p.live_bytes, 0);
    }
    /// A file holding a column in a type the table has moved on from is a
    /// compaction input whatever its size: alone, above the target, below
    /// the fill floor. An ordinary file of the same shape stays out.
    #[test]
    fn a_stale_typed_file_is_rewritten_however_large_or_lonely() {
        let cfg = CompactionSettings {
            target_superfile_size_mb: 100,
            min_fill_percent: 80,
            min_superfiles_for_merge: 4,
            ..CompactionSettings::default()
        };
        let mut lonely = seg(1, 10, 1_000, 0);
        assert!(
            select(&[lonely.clone()], &cfg).is_empty(),
            "a small file alone waits"
        );
        lonely.stale_type = true;
        let jobs = select(&[lonely.clone()], &cfg);
        assert_eq!(jobs.len(), 1);
        assert_eq!(jobs[0].inputs, vec![lonely.superfile_id]);

        let mut huge = seg(2, 500, 1_000, 0);
        assert!(
            select(from_ref(&huge), &cfg).is_empty(),
            "a target-sized file is left alone"
        );
        huge.stale_type = true;
        let jobs = select(from_ref(&huge), &cfg);
        assert_eq!(jobs.len(), 1, "unless its type is stale");
        assert_eq!(jobs[0].inputs, vec![huge.superfile_id]);
    }

    /// A stale file packs with ordinary candidates of its partition; the
    /// job it lands in runs even when it would otherwise be too small.
    #[test]
    fn a_stale_typed_file_packs_with_its_neighbours() {
        let cfg = CompactionSettings {
            target_superfile_size_mb: 100,
            min_fill_percent: 80,
            min_superfiles_for_merge: 4,
            ..CompactionSettings::default()
        };
        let mut stale = seg(1, 10, 1_000, 0);
        stale.stale_type = true;
        let fresh = seg(2, 10, 1_000, 0);
        let jobs = select(&[stale.clone(), fresh.clone()], &cfg);
        assert_eq!(jobs.len(), 1);
        let mut inputs = jobs[0].inputs.clone();
        inputs.sort();
        assert_eq!(inputs, vec![stale.superfile_id, fresh.superfile_id]);
    }
}
