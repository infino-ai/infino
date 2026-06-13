// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident-Set-Size sampling helper for the bench harnesses.
//!
//! Two surfaces:
//!
//! - [`current_rss_bytes`] — one-shot read of the process's
//!   current `VmRSS` (Linux `/proc/self/status`). Returns
//!   `None` on platforms without procfs.
//! - [`PeakSampler`] — background thread that polls VmRSS at
//!   a fixed cadence and records peak / median / p90 values
//!   over the sampler's lifetime. Use [`PeakSampler::start`]
//!   (or [`PeakSampler::start_default`]) before the work you
//!   want to bound, [`PeakSampler::stop`] after — returns the
//!   peak observed.
//!
//! Why a sampler thread instead of `getrusage(RUSAGE_SELF)`:
//! `ru_maxrss` is process-lifetime peak. Re-running a build
//! after a huge build doesn't reset it, so back-to-back bench
//! groups read the same number. Per-group peak via a sampler
//! correctly attributes RSS to the group that drove it.
//!
//! Why VmRSS specifically: it's the resident portion of the
//! process address space — what shows up in `top`. Reflects
//! what the bench actually paid in physical memory, not the
//! virtual reservation (which mmap-heavy workloads inflate
//! without paying for it).
//!
//! Sampling at 50 ms is enough resolution to catch any peak
//! a real build / ingest will dwell in for >50 ms (every
//! 1M-doc build is in the multi-second range; the IVF
//! training + assignment plateaus are seconds long). Faster
//! sampling adds noise without adding signal.
//!
//! Run-to-run persistence lives in `report.rs`; this module only
//! samples and formats RSS.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Force the global allocator (mimalloc, default-on in the crate) to
/// return freed-but-retained arenas to the OS. Every sampler start
/// calls this so a phase's reported RSS is the engine's working set,
/// not heap history from earlier bench phases — e.g. ingest scratch
/// and brute-force ground-truth buffers were inflating the vector
/// warm-search rows by the (freed) corpus-scale residual mimalloc had
/// kept around (measured: ~1 GiB at 1M docs, ~10 GiB at 10M).
pub fn purge_allocator() {
    // SAFETY: `mi_collect` is documented safe to call from any thread
    // at any time; `true` forces a synchronous collection that
    // releases deferred pages back to the OS.
    unsafe { libmimalloc_sys::mi_collect(true) };
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

/// One-shot read of the calling process's current VmRSS in
/// bytes. `None` on non-Linux hosts or if `/proc/self/status`
/// is unavailable. The c7i.4xlarge bench host is Linux, so
/// `None` on it indicates a parse failure (which the caller
/// should treat as bench-instrumentation failure, not a
/// regression).
pub fn current_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        // Format: `VmRSS:\t   12345 kB`
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * KIB_TO_BYTES);
        }
    }
    None
}

/// One-shot read of the calling process's **anonymous** resident set
/// (private heap) in bytes, from `/proc/self/smaps_rollup`. File-backed
/// pages (the disk-cache mmap on free local NVMe) are excluded.
pub fn current_anon_rss_bytes() -> Option<u64> {
    purge_allocator();
    let rollup = std::fs::read_to_string("/proc/self/smaps_rollup").ok()?;
    rollup
        .lines()
        .find(|l| l.starts_with("Anonymous:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse::<u64>().ok())
        .map(|kb| kb * KIB_TO_BYTES)
}

/// Background-thread peak-RSS sampler. Start it before the
/// work you want to bound and stop it after; the returned
/// peak is the max VmRSS observed across the sampler's
/// lifetime.
///
/// The thread reads `/proc/self/status` at `interval`
/// cadence. Each read is a ~10 µs syscall — negligible next
/// to the work the sampler watches.
pub struct PeakSampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<u64>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct RssStats {
    pub peak_rss_bytes: u64,
    pub median_rss_bytes: u64,
    pub p90_rss_bytes: u64,
}

impl RssStats {
    fn from_samples(mut samples: Vec<u64>) -> Self {
        if samples.is_empty() {
            samples.push(current_rss_bytes().unwrap_or(0));
        }
        samples.sort_unstable();
        Self {
            peak_rss_bytes: *samples.last().expect("rss samples is non-empty"),
            median_rss_bytes: percentile_nearest_rank(&samples, RSS_MEDIAN_PERCENTILE),
            p90_rss_bytes: percentile_nearest_rank(&samples, RSS_P90_PERCENTILE),
        }
    }
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((percentile as f64 / PERCENT_SCALE) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

impl PeakSampler {
    /// Start a sampler with the default bench cadence (50 ms).
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    /// Start a sampler that polls VmRSS every `interval`.
    /// Seeds the peak with the current reading so callers
    /// who stop the sampler before any background sample
    /// lands still see at least the start-time RSS.
    pub fn start(interval: Duration) -> Self {
        // Drop allocator-retained pages first so the measurement
        // window opens on the process's true working set.
        purge_allocator();
        let stop = Arc::new(AtomicBool::new(false));
        let initial = current_rss_bytes().unwrap_or(0);

        let stop_t = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name("rss-sampler".into())
            .spawn(move || {
                let mut samples = vec![initial];
                while !stop_t.load(Ordering::Acquire) {
                    if let Some(rss) = current_rss_bytes() {
                        samples.push(rss);
                    }
                    thread::sleep(interval);
                }
                if let Some(rss) = current_rss_bytes() {
                    samples.push(rss);
                }
                samples
            })
            .expect("spawn rss-sampler thread");

        Self {
            stop,
            handle: Some(handle),
        }
    }

    /// Stop the sampler, join the background thread, return
    /// the peak VmRSS observed (in bytes). Consumes the
    /// sampler.
    pub fn stop(self) -> u64 {
        self.stop_stats().peak_rss_bytes
    }

    /// Stop the sampler, join the background thread, and return
    /// peak / median / p90 VmRSS observed over the sampler's lifetime.
    pub fn stop_stats(mut self) -> RssStats {
        self.stop.store(true, Ordering::Release);
        let samples = self
            .handle
            .take()
            .and_then(|h| h.join().ok())
            .unwrap_or_else(|| vec![current_rss_bytes().unwrap_or(0)]);
        RssStats::from_samples(samples)
    }
}

/// Format a byte count as a right-justified human string —
/// `"12.34 GiB"` / `"456.78 MiB"` / `"123.4 KiB"` — for the
/// bench markdown tables.
/// Log the anonymous-vs-file-backed RSS split from
/// `/proc/self/smaps_rollup` with a phase label. The split is the
/// discriminator for RSS investigations: anonymous = heap (allocator
/// slack, retained `Bytes`, builder state); file-backed = mmap'd
/// cache files (disk-cache segments, corpus files). Purges the
/// allocator first so retained-but-free heap doesn't masquerade as a
/// live working set.
pub fn log_rss_breakdown(label: &str) {
    purge_allocator();
    let Ok(rollup) = std::fs::read_to_string("/proc/self/smaps_rollup") else {
        return;
    };
    let kb = |key: &str| -> u64 {
        rollup
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    };
    let rss = kb("Rss:");
    let anon = kb("Anonymous:");
    let shmem = kb("Shmem:");
    // Everything resident that is neither anonymous heap nor shmem is
    // file-backed: mmap'd disk-cache segments, corpus files, binaries.
    let file_backed = rss.saturating_sub(anon).saturating_sub(shmem);
    eprintln!(
        "[rss-breakdown] {label}: rss={} anonymous={} file_backed={} shmem={}",
        fmt_bytes(rss * KIB_TO_BYTES),
        fmt_bytes(anon * KIB_TO_BYTES),
        fmt_bytes(file_backed * KIB_TO_BYTES),
        fmt_bytes(shmem * KIB_TO_BYTES),
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

    /// Sampler poll interval used by the "seed at start" test (long
    /// enough that no poll fires before `stop()`).
    const TEST_SAMPLER_INTERVAL_MS: u64 = 1_000;
    /// Size of the faulted-in buffer the allocation-growth test
    /// allocates (32 MiB).
    const TEST_ALLOC_SIZE_BYTES: usize = 32 * 1024 * 1024;
    /// Page-touch stride used to defeat lazy fault-in.
    const TEST_PAGE_STRIDE_BYTES: usize = 4096;
    /// Lower bound on observed RSS growth the allocation test asserts
    /// (16 MiB — half the allocation, leaving slack for the sampler).
    const TEST_MIN_RSS_GROWTH_BYTES: u64 = 16 * 1024 * 1024;

    /// VmRSS must be non-zero on Linux during a normal test
    /// run — the test process itself has resident pages.
    /// Skipped silently on non-Linux hosts where procfs is
    /// absent (returns `None`).
    #[test]
    fn current_rss_is_nonzero_on_linux() {
        if let Some(rss) = current_rss_bytes() {
            assert!(rss > 0, "VmRSS reported as zero — parse error?");
        }
    }

    /// Sampler must observe at least the start-time RSS even
    /// if `stop()` is called before the first poll fires.
    /// Pins the seed-with-current behavior in [`PeakSampler::start`].
    #[test]
    fn sampler_returns_at_least_start_rss() {
        let start_rss = current_rss_bytes();
        let s = PeakSampler::start(Duration::from_millis(TEST_SAMPLER_INTERVAL_MS));
        let peak = s.stop();
        if let Some(start) = start_rss {
            assert!(peak >= start, "peak {peak} < start {start} — seed missing");
        }
    }

    /// Allocating a sizeable buffer mid-sampling must move
    /// the observed peak above the pre-allocation reading.
    /// Touches every page to defeat lazy fault-in (otherwise
    /// the allocation reserves virtual address space without
    /// actually paying RSS).
    #[test]
    fn sampler_observes_allocation_growth() {
        let baseline = match current_rss_bytes() {
            Some(b) => b,
            None => return,
        };
        let s = PeakSampler::start(Duration::from_millis(5));
        // 32 MiB faulted-in buffer.
        let mut v: Vec<u8> = vec![0; TEST_ALLOC_SIZE_BYTES];
        for chunk in v.chunks_mut(TEST_PAGE_STRIDE_BYTES) {
            chunk[0] = 1;
        }
        std::thread::sleep(Duration::from_millis(50));
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
        let stats = RssStats::from_samples(vec![50, 10, 40, 20, 30]);
        assert_eq!(stats.peak_rss_bytes, 50);
        assert_eq!(stats.median_rss_bytes, 30);
        assert_eq!(stats.p90_rss_bytes, 50);
    }
}
