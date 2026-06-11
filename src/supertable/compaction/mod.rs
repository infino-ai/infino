// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Picks which superfiles to merge.
//!
//! no I/O. `supertable::compact` gathers the
//! stats, calls [`select`], then merges each [`CompactionJob`].
//! Compaction is single-level — a target-sized segment is never
//! re-compacted.

use crate::config::CompactionSettings;
use std::collections::BTreeMap;
use uuid::Uuid;

const MIB: u64 = 1024 * 1024;

/// Stats for one superfile. The caller fills these in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentStats {
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

impl SegmentStats {
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

/// Plan compaction: pack each partition's small segments into
/// as many target-sized jobs as they fill. Leftovers that can't
/// reach the floor are left for next time.
pub fn select(segments: &[SegmentStats], cfg: &CompactionSettings) -> Vec<CompactionJob> {
    let target_bytes = cfg.target_segment_size_mb.saturating_mul(MIB);
    let min_output_bytes =
        (target_bytes as u128 * cfg.min_fill_percent.clamp(1, 100) as u128 / 100) as u64;

    let mut by_partition: BTreeMap<&[u8], Vec<&SegmentStats>> = BTreeMap::new();
    for s in segments {
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
    segs: Vec<&SegmentStats>,
    target_bytes: u64,
    min_output_bytes: u64,
    jobs: &mut Vec<CompactionJob>,
) {
    // Exclude segments already at target size — they are done and
    // re-compacting them gains nothing.
    let mut candidates: Vec<&SegmentStats> = segs
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
    fn fits(&self, s: &SegmentStats, target_bytes: u64) -> bool {
        self.live_bytes + s.live_bytes() <= target_bytes
    }

    fn push(&mut self, s: &SegmentStats) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mib(n: u64) -> u64 {
        n * MIB
    }

    fn seg(id: u128, size_mib: u64, n_docs: u64, tombstoned: u64) -> SegmentStats {
        SegmentStats {
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
    fn splits_many_segments_into_multiple_jobs() {
        // 12 × 200 MiB: two jobs of 5, last 2 left over.
        let segs: Vec<_> = (0..12).map(|i| seg(i, 200, 1000, 0)).collect();
        let jobs = select(&segs, &default_cfg());
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.inputs.len() == 5));
    }

    #[test]
    fn already_target_sized_segment_is_never_re_compacted() {
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
}
