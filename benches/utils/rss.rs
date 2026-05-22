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
//! [`write_rss_stats`] / [`read_peak_rss_bytes`] persist + read
//! a per-bench `rss.json` next to criterion's `estimates.json`
//! so the markdown emitters can pick memory stats up by the
//! same `(group, bench)` lookup shape they use for timings.

use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const DEFAULT_INTERVAL: Duration = Duration::from_millis(50);

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
            let kb: u64 = rest.trim().split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
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
            median_rss_bytes: percentile_nearest_rank(&samples, 50),
            p90_rss_bytes: percentile_nearest_rank(&samples, 90),
        }
    }

    fn peak_only(peak_rss_bytes: u64) -> Self {
        Self {
            peak_rss_bytes,
            median_rss_bytes: peak_rss_bytes,
            p90_rss_bytes: peak_rss_bytes,
        }
    }
}

fn percentile_nearest_rank(sorted: &[u64], percentile: usize) -> u64 {
    debug_assert!(!sorted.is_empty());
    let rank = ((percentile as f64 / 100.0) * sorted.len() as f64).ceil() as usize;
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

/// Persist a peak RSS sample next to criterion's artifacts:
///
/// `target/criterion/<group>/<bench>/new/rss.json`
///
/// Before writing, the previous `new/rss.json` is moved to
/// `base/rss.json`, mirroring criterion's own `new`/`base`
/// rotation for `estimates.json`. Keeping the artifact beside
/// `estimates.json` makes the markdown emitters use the same
/// `(group, bench)` lookup shape for both latency and memory.
pub fn write_peak_rss(group: &str, bench: &str, peak_rss_bytes: u64) -> std::io::Result<()> {
    write_rss_stats(group, bench, RssStats::peak_only(peak_rss_bytes))
}

pub fn write_rss_stats(group: &str, bench: &str, stats: RssStats) -> std::io::Result<()> {
    let dir = criterion_bench_dir(group, bench);
    let new_dir = dir.join("new");
    let base_dir = dir.join("base");
    std::fs::create_dir_all(&new_dir)?;
    if let Ok(existing) = std::fs::read(new_dir.join("rss.json")) {
        std::fs::create_dir_all(&base_dir)?;
        std::fs::write(base_dir.join("rss.json"), existing)?;
    }
    let body = serde_json::json!({
        "peak_rss_bytes": stats.peak_rss_bytes,
        "median_rss_bytes": stats.median_rss_bytes,
        "p90_rss_bytes": stats.p90_rss_bytes,
    });
    std::fs::write(
        new_dir.join("rss.json"),
        serde_json::to_vec_pretty(&body).expect("serialize rss json"),
    )
}

/// Read a locally recorded peak RSS sample. `None` if the
/// file doesn't exist (bench was filtered out or hasn't run
/// yet) or the JSON can't be parsed.
pub fn read_peak_rss_bytes(group: &str, bench: &str) -> Option<u64> {
    read_rss_field(group, bench, "peak_rss_bytes")
}

pub fn read_median_rss_bytes(group: &str, bench: &str) -> Option<u64> {
    read_rss_field(group, bench, "median_rss_bytes")
}

pub fn read_p90_rss_bytes(group: &str, bench: &str) -> Option<u64> {
    read_rss_field(group, bench, "p90_rss_bytes")
}

pub fn fmt_median_rss(group: &str, bench: &str) -> String {
    read_median_rss_bytes(group, bench)
        .map(fmt_bytes)
        .unwrap_or_else(|| "—".into())
}

pub fn fmt_p90_rss(group: &str, bench: &str) -> String {
    read_p90_rss_bytes(group, bench)
        .map(fmt_bytes)
        .unwrap_or_else(|| "—".into())
}

fn read_rss_field(group: &str, bench: &str, field: &str) -> Option<u64> {
    let dir = criterion_bench_dir(group, bench);
    let path = dir.join("new").join("rss.json");
    let text = std::fs::read_to_string(&path)
        .or_else(|_| std::fs::read_to_string(dir.join("rss.json")))
        .ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v.get(field)
        .and_then(Value::as_u64)
        .or_else(|| v.get("peak_rss_bytes")?.as_u64())
}

/// Read the previous run's peak RSS sample (`base/rss.json`).
pub fn read_base_peak_rss_bytes(group: &str, bench: &str) -> Option<u64> {
    let path = criterion_bench_dir(group, bench)
        .join("base")
        .join("rss.json");
    let text = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v.get("peak_rss_bytes")?.as_u64()
}

/// Format the RSS delta for markdown tables. Uses a 5% noise band,
/// matching criterion's default practical-significance threshold.
pub fn fmt_peak_rss_delta(group: &str, bench: &str) -> String {
    let Some(new) = read_peak_rss_bytes(group, bench) else {
        return "—".into();
    };
    let Some(base) = read_base_peak_rss_bytes(group, bench) else {
        return "—".into();
    };
    if base == 0 {
        return "—".into();
    }
    let pct = ((new as f64 - base as f64) / base as f64) * 100.0;
    let label = if pct <= -5.0 {
        "improved"
    } else if pct >= 5.0 {
        "regressed"
    } else {
        "no change"
    };
    format!("{pct:+.1}% {label}")
}

/// `$CARGO_TARGET_DIR/criterion/<group>/<bench>` if `CARGO_TARGET_DIR`
/// is set (criterion writes there when the env var is exported), else
/// workspace-relative `target/criterion/<group>/<bench>`. Tracking
/// criterion's own behavior keeps `rss.json` next to `estimates.json`
/// on every host, including CI where the target dir is redirected
/// outside the workspace.
fn criterion_bench_dir(group: &str, bench: &str) -> PathBuf {
    let base = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    base.join("criterion").join(group).join(bench)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let s = PeakSampler::start(Duration::from_millis(1_000));
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
        let mut v: Vec<u8> = vec![0; 32 * 1024 * 1024];
        for chunk in v.chunks_mut(4096) {
            chunk[0] = 1;
        }
        std::thread::sleep(Duration::from_millis(50));
        std::hint::black_box(&v);
        let peak = s.stop();
        assert!(
            peak >= baseline + 16 * 1024 * 1024,
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
