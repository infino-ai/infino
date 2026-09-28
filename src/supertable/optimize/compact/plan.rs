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
//! target size is done and never re-merged, and each candidate lands in
//! **exactly one** job, which is what lets the jobs run independently.

use std::{collections::BTreeMap, mem};

use uuid::Uuid;

use crate::{
    config::CompactionSettings,
    supertable::manifest::{list::DrainedVersionRanges, listed_once},
};

/// Bytes in a mebibyte, the unit the settings are expressed in.
pub(super) const MIB: u64 = 1024 * 1024;

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
    /// Commit version the superfile was born at. A merged superfile carries
    /// the OLDEST input's `birth_version`, so user-table merge jobs must
    /// never mix inputs from opposite sides of the hidden drain watermark
    /// (see [`split_stats_at_drain_watermark`]).
    pub birth_version: u64,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionJob {
    pub partition_key: Vec<u8>,
    pub inputs: Vec<Uuid>,
    /// Estimated size of the merged superfile.
    pub estimated_output_bytes: u64,
}

/// Plan compaction: pack each partition's small superfiles into
/// as many target-sized jobs as they fill. Leftovers that can't
/// reach the floor are left for next time. Each `superfile_id` must appear
/// once: `compact_one_table` drops repeats with [`listed_once`] before the
/// drain-watermark split.
pub fn select(superfiles: &[SuperfileStats], cfg: &CompactionSettings) -> Vec<CompactionJob> {
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
            min_superfiles_for_merge,
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
    min_superfiles_for_merge: usize,
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
            pending.emit(key, min_output_bytes, min_superfiles_for_merge, jobs);
        }
        pending.push(s);
    }
    pending.emit(key, min_output_bytes, min_superfiles_for_merge, jobs);
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

    /// Emit a CompactionJob when the pending inputs clear either leg of the
    /// merge trigger — size OR count:
    /// - size: `>= 2` inputs and live bytes reach `min_output_bytes`;
    /// - count: `>= min_superfiles_for_merge` inputs (already `>= 2`), which
    ///   fires even when the live bytes sit far below the size floor.
    fn emit(
        &mut self,
        key: &[u8],
        min_output_bytes: u64,
        min_superfiles_for_merge: usize,
        jobs: &mut Vec<CompactionJob>,
    ) {
        let size_ready = self.inputs.len() >= 2 && self.live_bytes >= min_output_bytes;
        let count_ready = self.inputs.len() >= min_superfiles_for_merge;
        if size_ready || count_ready {
            jobs.push(CompactionJob {
                partition_key: key.to_vec(),
                inputs: mem::take(&mut self.inputs),
                estimated_output_bytes: self.live_bytes,
            });
        }
        *self = PendingJob::default();
    }
}

#[cfg(test)]
pub(in crate::supertable::optimize::compact) mod tests {
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
            sealed_by_other: false,
            birth_version: 0,
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
        // A single-input pending job never emits even if it reaches the fill
        // floor and the count trigger (emit takes a pre-clamped count of 2, so
        // one input clears neither the size nor the count leg).
        let mut jobs = Vec::new();
        let mut p = PendingJob::default();
        p.push(&seg(1, 200, 1000, 0));
        p.emit(&[], 0, 2, &mut jobs);
        assert!(jobs.is_empty(), "single-input job must not emit");
        // Reset to default after emit attempt.
        assert_eq!(p.inputs.len(), 0);
        assert_eq!(p.live_bytes, 0);
    }
}
