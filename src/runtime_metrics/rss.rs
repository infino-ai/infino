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
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

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
/// Process status file carrying `VmRSS`.
const PROC_SELF_STATUS: &str = "/proc/self/status";
/// Aggregated smaps rollup (Anonymous / Rss / Shmem).
const PROC_SELF_SMAPS_ROLLUP: &str = "/proc/self/smaps_rollup";

/// One-shot read of the calling process's current VmRSS in bytes.
pub fn current_rss_bytes() -> Option<u64> {
    let s = fs::read_to_string(PROC_SELF_STATUS).ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
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
    eprintln!(
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
    use super::*;

    const TEST_SAMPLER_INTERVAL_MS: u64 = 1_000;
    const TEST_ALLOC_SIZE_BYTES: usize = 32 * 1024 * 1024;
    const TEST_PAGE_STRIDE_BYTES: usize = 4096;
    const TEST_MIN_RSS_GROWTH_BYTES: u64 = 16 * 1024 * 1024;
    /// Fast poll interval so the growth test can observe the allocation.
    const TEST_GROWTH_SAMPLER_INTERVAL: Duration = Duration::from_millis(5);
    /// Hold the allocation long enough for at least one sampler tick.
    const TEST_GROWTH_HOLD: Duration = Duration::from_millis(50);

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

    #[test]
    fn sampler_observes_allocation_growth() {
        purge_allocator();
        let baseline = match current_rss_bytes() {
            Some(b) => b,
            None => return,
        };
        let s = PeakSampler::start(TEST_GROWTH_SAMPLER_INTERVAL);
        let mut v: Vec<u8> = vec![0; TEST_ALLOC_SIZE_BYTES];
        for chunk in v.chunks_mut(TEST_PAGE_STRIDE_BYTES) {
            chunk[0] = 1;
        }
        thread::sleep(TEST_GROWTH_HOLD);
        std::hint::black_box(&v);
        let peak = s.stop();
        assert!(
            peak >= baseline + TEST_MIN_RSS_GROWTH_BYTES,
            "sampler missed the 32 MiB faulted allocation: \
             baseline={baseline}, peak={peak}"
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
}
