// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Resident-Set-Size sampling for the bench harness.
//!
//! [`PeakSampler`] polls the process `VmRSS` on a background thread and
//! reports peak / median / p90 over its lifetime. Wrap it around the
//! work for *one* engine in isolation (build, then query) so the
//! sampled window contains only that engine plus the mmap corpus —
//! which, being file-backed, doesn't count as the engine's anonymous
//! footprint.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_INTERVAL: Duration = Duration::from_millis(50);

/// One-shot read of the calling process's current `VmRSS` in bytes.
/// `None` on non-Linux hosts or if `/proc/self/status` is unavailable.
pub fn current_rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Peak / median / p90 `VmRSS` (bytes) observed over a sampler's life.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
            median_rss_bytes: percentile_nearest_rank(&samples, 50),
            p90_rss_bytes: percentile_nearest_rank(&samples, 90),
        }
    }
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((percentile as f64 / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

/// Background-thread `VmRSS` peak sampler. Start before the work to
/// bound, stop after.
pub struct PeakSampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<u64>>>,
}

impl PeakSampler {
    /// Start a sampler at the default 50 ms cadence.
    pub fn start_default() -> Self {
        Self::start(DEFAULT_INTERVAL)
    }

    /// Start a sampler polling `VmRSS` every `interval`. Seeds with the
    /// current reading so a sampler stopped before the first poll still
    /// reports at least the start-time RSS.
    pub fn start(interval: Duration) -> Self {
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

    /// Stop the sampler and return peak / median / p90 `VmRSS`.
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

/// Format a byte count as a right-justified human string for tables.
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
