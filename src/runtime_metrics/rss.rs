// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident-Set-Size sampling for cost accounting and usage flush.
//!
//! Two surfaces:
//!
//! - [`current_rss_bytes`] — one-shot read of the process's current `VmRSS`
//!   (Linux `/proc/self/status`). Returns `None` on platforms without procfs.
//! - [`PeakSampler`] — background thread that polls VmRSS at a fixed cadence
//!   and records peak / median / p90 over the sampler's lifetime.
//!
//! Process-wide attribution only — not a Stripe billing dimension.

use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use tracing::debug;

/// Force the global allocator (mimalloc, default-on) to return freed-but-
/// retained arenas to the OS. No-op when mimalloc is not the global allocator.
pub fn purge_allocator() {
    #[cfg(all(not(miri), feature = "mimalloc"))]
    {
        // SAFETY: `mi_collect` is documented safe to call from any thread
        // at any time; `true` forces a synchronous collection that
        // releases deferred pages back to the OS.
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}

const DEFAULT_INTERVAL: Duration = Duration::from_millis(50);

/// Bytes per kibibyte — `/proc/self/status` reports `VmRSS` in kB
/// (actually KiB), which we convert to bytes.
const KIB_TO_BYTES: u64 = 1024;
/// Median percentile rank for RSS stats.
const RSS_MEDIAN_PERCENTILE: usize = 50;
/// P90 percentile rank for RSS stats.
const RSS_P90_PERCENTILE: usize = 90;
/// Divisor converting a percentile rank to a `[0, 1]` fraction.
const PERCENT_SCALE: f64 = 100.0;
/// Process status file carrying `VmRSS` and `RssAnon`.
const PROC_SELF_STATUS: &str = "/proc/self/status";
/// System memory summary carrying `MemAvailable`.
const PROC_MEMINFO: &str = "/proc/meminfo";
/// Aggregated smaps rollup (Anonymous / Rss / Shmem).
const PROC_SELF_SMAPS_ROLLUP: &str = "/proc/self/smaps_rollup";
/// Where the cgroup v2 hierarchy is mounted.
const CGROUP_MOUNT: &str = "/sys/fs/cgroup";
/// This process's place in that hierarchy, as `0::<relative path>`.
const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";
/// cgroup v2 memory ceiling, or the literal `max`.
const CGROUP_MEMORY_MAX: &str = "memory.max";
/// cgroup v2 current charge: anonymous, page cache and kernel memory.
const CGROUP_MEMORY_CURRENT: &str = "memory.current";
/// cgroup v2 breakdown of that charge, read for its reclaimable part.
const CGROUP_MEMORY_STAT: &str = "memory.stat";
/// `memory.max` for a cgroup with no ceiling of its own.
const CGROUP_UNLIMITED: &str = "max";

/// One-shot read of the calling process's current VmRSS in bytes.
pub fn current_rss_bytes() -> Option<u64> {
    status_field_bytes("VmRSS:")
}

/// The calling process's anonymous resident bytes (`RssAnon`), read from
/// `/proc/self/status`: the kernel's counter, with no allocator purge and no
/// walk of the mappings, so it is cheap enough to poll while a query runs.
/// Unlike [`current_anon_rss_bytes`] it includes pages the allocator has freed
/// but not yet returned, which is what a memory limit counts.
pub(crate) fn status_anon_rss_bytes() -> Option<u64> {
    status_field_bytes("RssAnon:")
}

/// A `/proc/self/status` field that the kernel reports in kB, in bytes.
fn status_field_bytes(field: &str) -> Option<u64> {
    let s = fs::read_to_string(PROC_SELF_STATUS).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * KIB_TO_BYTES);
        }
    }
    None
}

/// One-shot read of the memory available for a new allocation without
/// swapping, in bytes: this process's cgroup ceiling where it has one, the
/// host's `MemAvailable` otherwise.
///
/// The cgroup comes first because `/proc/meminfo` reports the machine, not
/// the limit the process actually lives under. In a 4 GiB container on a
/// large host it reads back tens of gigabytes free, a sizing decision made
/// on it admits work the cgroup cannot hold, and the OOM killer answers
/// instead of the throttle.
///
/// Returns `None` on platforms with neither, so every caller needs a
/// conservative fallback rather than a guess at the machine's size.
pub fn available_memory_bytes() -> Option<u64> {
    memory_budget().map(|(available, _)| available)
}

/// Total memory a sizing decision may spend, in bytes: the cgroup's ceiling
/// where it has one, the host's `MemTotal` otherwise. Paired with
/// [`available_memory_bytes`] so a reserve can be a share of whichever of the
/// two actually binds. `None` with neither.
pub fn total_memory_bytes() -> Option<u64> {
    memory_budget().map(|(_, total)| total)
}

/// Available and total together, always from the same source.
///
/// Taken as a pair because callers compare them: a cgroup-limited numerator
/// over a host-sized denominator, or the reverse, is not a share of anything.
/// The reverse is the dangerous one — host `MemAvailable` over a container's
/// ceiling reads as far more than 100% free and would admit without limit —
/// and it is reachable whenever the ceiling is readable but the charge is not.
/// So a cgroup answer needs every part of it to come back, or the host's pair
/// is used whole.
pub fn memory_budget() -> Option<(u64, u64)> {
    if let Some(budget) = cgroup_budget() {
        return Some(budget);
    }
    Some((meminfo_field("MemAvailable:")?, meminfo_field("MemTotal:")?))
}

/// What this process's cgroup can still take, and its ceiling. `None` without
/// cgroup v2, or where no cgroup from here to the root sets one, in which case
/// the host's figures are the right ones.
///
/// Walks leaf to root and takes the tightest of the ceilings it finds, because
/// a v2 limit binds every descendant: the ceiling that stops this process may
/// be set an ancestor away, on a systemd slice rather than the unit, or on a
/// Kubernetes pod rather than the container. Reading only the leaf sees `max`
/// there and sizes against the whole machine, which is the failure this path
/// exists to prevent.
fn cgroup_budget() -> Option<(u64, u64)> {
    let root = Path::new(CGROUP_MOUNT);
    let mut dir = cgroup_dir()?;
    let mut tightest: Option<(u64, u64)> = None;
    loop {
        if let Some(budget) = cgroup_level_budget(&dir) {
            tightest = Some(match tightest {
                // Headroom and ceiling are tracked together rather than
                // minimised apart: a level's headroom is only meaningful
                // against its own ceiling, and a share built from two levels
                // would describe neither.
                Some(current) if current.0 <= budget.0 => current,
                _ => budget,
            });
        }
        if dir == root {
            return tightest;
        }
        match dir.parent() {
            Some(parent) if parent.starts_with(root) => dir = parent.to_path_buf(),
            _ => return tightest,
        }
    }
}

/// Headroom and ceiling for one cgroup directory. `None` when it sets no
/// ceiling of its own, or when any of the three files it needs is unreadable.
fn cgroup_level_budget(dir: &Path) -> Option<(u64, u64)> {
    let limit = parse_cgroup_limit(&fs::read_to_string(dir.join(CGROUP_MEMORY_MAX)).ok()?)?;
    let current: u64 = fs::read_to_string(dir.join(CGROUP_MEMORY_CURRENT))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    // Most of a compaction's charge is page cache, which the kernel reclaims
    // under pressure rather than OOM-killing for, so it counts as headroom —
    // the same accounting `MemAvailable` does for the host. Ceiling minus
    // charge alone would throttle a container nowhere near its limit.
    let stat = fs::read_to_string(dir.join(CGROUP_MEMORY_STAT)).ok()?;
    let reclaimable = memory_stat_field(&stat, "inactive_file").unwrap_or(0)
        + memory_stat_field(&stat, "slab_reclaimable").unwrap_or(0);
    Some((cgroup_headroom(limit, current, reclaimable), limit))
}

/// The directory holding this process's own cgroup v2 memory files, the leaf
/// of the walk in [`cgroup_budget`].
///
/// Not simply the mount root. That is the process's own cgroup only when it
/// has a private cgroup namespace, which `docker run -m` gives it; a systemd
/// unit with `MemoryMax=`, or a Kubernetes runtime sharing the host's cgroup
/// namespace, leaves the process in a nested cgroup whose root carries no
/// `memory.max` at all. Reading only the root there finds nothing and falls
/// back to the host's figures, which is the case this path exists to avoid.
/// So the relative path comes from `/proc/self/cgroup`, with the mount root as
/// the fallback.
fn cgroup_dir() -> Option<PathBuf> {
    let root = Path::new(CGROUP_MOUNT);
    if let Some(relative) = fs::read_to_string(PROC_SELF_CGROUP)
        .ok()
        .as_deref()
        .and_then(parse_cgroup_path)
    {
        let nested = root.join(relative);
        if nested.join(CGROUP_MEMORY_MAX).exists() {
            return Some(nested);
        }
    }
    root.join(CGROUP_MEMORY_MAX)
        .exists()
        .then(|| root.to_path_buf())
}

/// This process's cgroup v2 path, relative to the mount root.
///
/// `/proc/self/cgroup` lists one controller per line; v2's is `0::<path>`,
/// with the path absolute-looking but relative to the mount. A process at the
/// root reads `0::/`, which joins to the root itself. cgroup v1 lists numbered
/// controllers and no `0::` line, so it reads as absent and the host's figures
/// are used, which is what a v1 host did before any of this existed.
fn parse_cgroup_path(raw: &str) -> Option<&str> {
    raw.lines()
        .find_map(|line| line.strip_prefix("0::"))
        .map(|path| path.trim().trim_start_matches('/'))
}

/// `memory.max`, as bytes. `None` for the literal `max`, a cgroup with no
/// ceiling of its own.
fn parse_cgroup_limit(raw: &str) -> Option<u64> {
    let raw = raw.trim();
    if raw == CGROUP_UNLIMITED {
        return None;
    }
    raw.parse().ok()
}

/// Headroom left in a cgroup: what the ceiling has not charged, plus what is
/// charged but reclaimable. Saturating, because `current` can exceed the
/// ceiling momentarily and a sizing decision wants zero rather than a wrap.
fn cgroup_headroom(limit: u64, current: u64, reclaimable: u64) -> u64 {
    limit
        .saturating_sub(current)
        .saturating_add(reclaimable.min(current))
        .min(limit)
}

/// One `memory.stat` field, in bytes. The file is `key value` per line, in
/// bytes already, unlike `/proc/meminfo`'s kibibytes.
fn memory_stat_field(raw: &str, key: &str) -> Option<u64> {
    raw.lines()
        .filter_map(|line| line.split_once(' '))
        .find(|(name, _)| *name == key)
        .and_then(|(_, value)| value.trim().parse().ok())
}

/// One `/proc/meminfo` field, in bytes.
fn meminfo_field(prefix: &str) -> Option<u64> {
    let s = fs::read_to_string(PROC_MEMINFO).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * KIB_TO_BYTES);
        }
    }
    None
}

/// One-shot read of anonymous resident set (private heap) in bytes.
pub fn current_anon_rss_bytes() -> Option<u64> {
    purge_allocator();
    anon_rss_bytes_fast()
}

fn anon_rss_bytes_fast() -> Option<u64> {
    let rollup = fs::read_to_string(PROC_SELF_SMAPS_ROLLUP).ok()?;
    rollup
        .lines()
        .find(|l| l.starts_with("Anonymous:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb * KIB_TO_BYTES)
}

/// Background-thread peak-RSS sampler.
pub struct PeakSampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<(u64, u64)>>>,
    /// Seed sample taken at start; reused if the sampler thread never runs.
    initial: (u64, u64),
}

#[derive(Debug, Clone, Copy)]
pub struct RssStats {
    /// Peak total VmRSS — what the cost model's RAM-hold leg bills.
    pub peak_rss_bytes: u64,
    pub median_rss_bytes: u64,
    pub p90_rss_bytes: u64,
    /// Peak anonymous (private heap) RSS — diagnostic only.
    pub peak_anon_rss_bytes: u64,
    /// Peak file-backed resident bytes — diagnostic only.
    pub peak_file_rss_bytes: u64,
}

impl RssStats {
    fn from_samples(mut samples: Vec<(u64, u64)>) -> Self {
        if samples.is_empty() {
            samples.push((
                current_rss_bytes().unwrap_or(0),
                anon_rss_bytes_fast().unwrap_or(0),
            ));
        }
        let peak_anon = samples.iter().map(|(_, a)| *a).max().unwrap_or(0);
        let peak_file = samples
            .iter()
            .map(|(t, a)| t.saturating_sub(*a))
            .max()
            .unwrap_or(0);
        let mut totals: Vec<u64> = samples.iter().map(|(t, _)| *t).collect();
        totals.sort_unstable();
        Self {
            peak_rss_bytes: *totals.last().expect("rss samples is non-empty"),
            median_rss_bytes: percentile_nearest_rank(&totals, RSS_MEDIAN_PERCENTILE),
            p90_rss_bytes: percentile_nearest_rank(&totals, RSS_P90_PERCENTILE),
            peak_anon_rss_bytes: peak_anon,
            peak_file_rss_bytes: peak_file,
        }
    }
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((percentile as f64 / PERCENT_SCALE) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

impl PeakSampler {
    /// Start a sampler with the default cadence (50 ms).
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    /// Start a sampler that polls VmRSS every `interval`.
    pub fn start(interval: Duration) -> Self {
        purge_allocator();
        let stop = Arc::new(AtomicBool::new(false));
        let initial = (
            current_rss_bytes().unwrap_or(0),
            anon_rss_bytes_fast().unwrap_or(0),
        );

        let stop_t = Arc::clone(&stop);
        // Sampling is best-effort: if the OS refuses a thread, degrade to
        // the initial snapshot instead of aborting the process.
        let handle = thread::Builder::new()
            .name("rss-sampler".into())
            .spawn(move || {
                let mut samples = vec![initial];
                while !stop_t.load(Ordering::Acquire) {
                    if let Some(rss) = current_rss_bytes() {
                        samples.push((rss, anon_rss_bytes_fast().unwrap_or(0)));
                    }
                    // Interruptible wait so `stop_stats` can unpark promptly.
                    thread::park_timeout(interval);
                }
                if let Some(rss) = current_rss_bytes() {
                    samples.push((rss, anon_rss_bytes_fast().unwrap_or(0)));
                }
                samples
            })
            .ok();

        Self {
            stop,
            handle,
            initial,
        }
    }

    /// Stop the sampler and return peak VmRSS (bytes).
    pub fn stop(self) -> u64 {
        self.stop_stats().peak_rss_bytes
    }

    /// Stop the sampler and return peak / median / p90 plus anon/file peaks.
    pub fn stop_stats(mut self) -> RssStats {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.as_ref() {
            handle.thread().unpark();
        }
        let samples = self
            .handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_else(|| vec![self.initial]);
        RssStats::from_samples(samples)
    }
}

/// Settled `(rss, anonymous, file_backed, shmem)` after an allocator purge.
pub fn settled_rss_breakdown() -> Option<(u64, u64, u64, u64)> {
    purge_allocator();
    let rollup = fs::read_to_string(PROC_SELF_SMAPS_ROLLUP).ok()?;
    let kb = |key: &str| -> u64 {
        rollup
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let rss = kb("Rss:") * KIB_TO_BYTES;
    let anon = kb("Anonymous:") * KIB_TO_BYTES;
    let shmem = kb("Shmem:") * KIB_TO_BYTES;
    let file_backed = rss.saturating_sub(anon).saturating_sub(shmem);
    Some((rss, anon, file_backed, shmem))
}

/// Log the anonymous-vs-file-backed RSS split with a phase label.
pub fn log_rss_breakdown(label: &str) {
    let Some((rss, anon, file_backed, shmem)) = settled_rss_breakdown() else {
        return;
    };
    debug!(
        "[rss-breakdown] {label}: rss={} anonymous={} file_backed={} shmem={}",
        fmt_bytes(rss),
        fmt_bytes(anon),
        fmt_bytes(file_backed),
        fmt_bytes(shmem),
    );
}

pub fn fmt_bytes(b: u64) -> String {
    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else if b >= MIB {
        format!("{:.2} MiB", b as f64 / MIB as f64)
    } else if b >= KIB {
        format!("{:.1} KiB", b as f64 / KIB as f64)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use std::hint::black_box;

    use super::*;

    const TEST_SAMPLER_INTERVAL_MS: u64 = 1_000;
    const TEST_ALLOC_SIZE_BYTES: usize = 32 * 1024 * 1024;
    const TEST_PAGE_STRIDE_BYTES: usize = 4096;
    const TEST_MIN_RSS_GROWTH_BYTES: u64 = 16 * 1024 * 1024;
    /// Fast poll interval so the growth test can observe the allocation.
    const TEST_GROWTH_SAMPLER_INTERVAL: Duration = Duration::from_millis(5);
    /// Hold the allocation long enough for at least one sampler tick.
    const TEST_GROWTH_HOLD: Duration = Duration::from_millis(50);
    /// Retry budget for the growth test. RSS is process-global, so a
    /// concurrent test in the same binary freeing memory between the
    /// baseline snapshot and the sampler's peak window shrinks the
    /// observed delta and can mask the faulted allocation. Each attempt
    /// takes a fresh baseline, so a false failure requires that
    /// interference to recur on every attempt.
    const TEST_GROWTH_ATTEMPTS: usize = 5;

    #[test]
    fn current_rss_is_nonzero_on_linux() {
        if let Some(rss) = current_rss_bytes() {
            assert!(rss > 0, "VmRSS reported as zero — parse error?");
        }
    }

    #[test]
    fn sampler_returns_at_least_start_rss() {
        purge_allocator();
        let before = current_rss_bytes();
        let s = PeakSampler::start(Duration::from_millis(TEST_SAMPLER_INTERVAL_MS));
        let after_start = current_rss_bytes();
        let peak = s.stop();
        if let (Some(before), Some(after)) = (before, after_start) {
            let floor = before.min(after);
            assert!(peak >= floor, "peak {peak} < floor {floor} — seed missing");
        }
    }

    /// One growth-test attempt: snapshot a fresh baseline, fault
    /// [`TEST_ALLOC_SIZE_BYTES`] under a running sampler, and return
    /// `(baseline, peak)`. `None` when VmRSS is unavailable (no procfs).
    fn fault_alloc_and_sample_peak() -> Option<(u64, u64)> {
        purge_allocator();
        let baseline = current_rss_bytes()?;
        let s = PeakSampler::start(TEST_GROWTH_SAMPLER_INTERVAL);
        let mut v: Vec<u8> = vec![0; TEST_ALLOC_SIZE_BYTES];
        for chunk in v.chunks_mut(TEST_PAGE_STRIDE_BYTES) {
            chunk[0] = 1;
        }
        thread::sleep(TEST_GROWTH_HOLD);
        black_box(&v);
        Some((baseline, s.stop()))
    }

    #[test]
    fn sampler_observes_allocation_growth() {
        let mut last = (0, 0);
        for _ in 0..TEST_GROWTH_ATTEMPTS {
            let Some((baseline, peak)) = fault_alloc_and_sample_peak() else {
                return;
            };
            if peak >= baseline + TEST_MIN_RSS_GROWTH_BYTES {
                return;
            }
            last = (baseline, peak);
        }
        let (baseline, peak) = last;
        panic!(
            "sampler missed the 32 MiB faulted allocation in \
             {TEST_GROWTH_ATTEMPTS} attempts: last baseline={baseline}, \
             last peak={peak}"
        );
    }

    #[test]
    fn rss_stats_use_nearest_rank_percentiles() {
        let stats = RssStats::from_samples(vec![(50, 5), (10, 1), (40, 30), (20, 2), (30, 3)]);
        assert_eq!(stats.peak_rss_bytes, 50);
        assert_eq!(stats.median_rss_bytes, 30);
        assert_eq!(stats.p90_rss_bytes, 50);
        assert_eq!(stats.peak_anon_rss_bytes, 30);
        assert_eq!(stats.peak_file_rss_bytes, 45);
    }

    /// A cgroup with no ceiling of its own reads back as absent, so the
    /// caller falls through to the host's figures rather than treating the
    /// literal `max` as a byte count.
    #[test]
    fn an_unlimited_cgroup_has_no_limit() {
        assert_eq!(parse_cgroup_limit("max\n"), None);
        assert_eq!(parse_cgroup_limit("4294967296\n"), Some(4_294_967_296));
        assert_eq!(parse_cgroup_limit(""), None);
    }

    /// Headroom counts what the ceiling has not charged plus what is charged
    /// but reclaimable, because most of a merge's charge is page cache the
    /// kernel drops under pressure rather than OOM-killing for.
    #[test]
    fn cgroup_headroom_counts_reclaimable_charge() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // 4 GiB ceiling, 3 GiB charged, 2 GiB of it reclaimable file pages.
        assert_eq!(cgroup_headroom(4 * GIB, 3 * GIB, 2 * GIB), 3 * GIB);
        // Nothing reclaimable: only the uncharged remainder is available.
        assert_eq!(cgroup_headroom(4 * GIB, 3 * GIB, 0), GIB);
        // Over the ceiling, and a reclaimable figure larger than the charge:
        // neither may wrap or exceed the ceiling.
        assert_eq!(cgroup_headroom(4 * GIB, 5 * GIB, 0), 0);
        assert_eq!(cgroup_headroom(4 * GIB, GIB, 9 * GIB), 4 * GIB);
    }

    /// A nested cgroup's path is read relative to the mount, and the root
    /// case joins to the mount itself. Without this a process in a nested
    /// cgroup (a systemd unit, or a container sharing the host's cgroup
    /// namespace) reads no ceiling at all and silently sizes against the host.
    #[test]
    fn a_nested_cgroup_path_is_read_relative_to_the_mount() {
        assert_eq!(
            parse_cgroup_path("0::/system.slice/infino.service\n"),
            Some("system.slice/infino.service")
        );
        assert_eq!(parse_cgroup_path("0::/\n"), Some(""));
        // v1 lists numbered controllers and no `0::` line.
        assert_eq!(
            parse_cgroup_path("11:memory:/docker/abc\n4:cpu:/docker/abc\n"),
            None
        );
    }

    /// `memory.stat` is `key value` in bytes, not `/proc/meminfo`'s kibibytes,
    /// and a prefix must not match a longer key.
    #[test]
    fn memory_stat_fields_parse_in_bytes() {
        let raw = "anon 1024\nfile 2048\ninactive_file 512\nslab_reclaimable 256\n";
        assert_eq!(memory_stat_field(raw, "inactive_file"), Some(512));
        assert_eq!(memory_stat_field(raw, "slab_reclaimable"), Some(256));
        assert_eq!(memory_stat_field(raw, "file"), Some(2048));
        assert_eq!(memory_stat_field(raw, "absent"), None);
    }
}
