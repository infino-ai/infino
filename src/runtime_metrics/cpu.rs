// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Process CPU-time sampling for cost accounting and usage flush.
//!
//! Source: process-wide user + system CPU ticks from `/proc/self/stat`
//! (`utime` + `stime`), converted with `getconf CLK_TCK`. Unlike summing only
//! currently-live task schedstats, this counter retains CPU from worker threads
//! that exit during a measured window and is therefore monotonic.
//!
//!   - **All-thread aggregate.** Includes the rayon compute-pool workers and
//!     tokio I/O workers, so it captures the whole process's compute — the
//!     foreground+background total we want.
//!   - **Excludes I/O wait.** A thread awaiting an object-store GET is *not*
//!     on-CPU, so its wait does not accrue here. This is exactly why wall time
//!     must not be used for compute cost: wall includes the wait, on-CPU time
//!     does not.
//!   - **ns resolution, no dependency.** Two `/proc` reads per measured
//!     window; nothing added to the dependency graph.
//!
//! Latency stays wall-clock (`Instant`); only *compute cost* uses these
//! on-CPU deltas.

use std::{
    fs,
    process::Command,
    sync::OnceLock,
    time::{Duration, Instant},
};

/// Process stat file containing cumulative user/system CPU ticks.
const PROC_SELF_STAT: &str = "/proc/self/stat";
/// POSIX clock-tick query used once per process.
const GETCONF: &str = "getconf";
/// `getconf` variable for scheduler ticks per second.
const CLK_TCK: &str = "CLK_TCK";
/// Nanoseconds per second.
const NS_PER_SEC: u128 = 1_000_000_000;
/// Index of `utime` in the post-`)` whitespace fields of `/proc/self/stat`
/// (kernel field 14 → suffix index 11).
const UTIME_FIELD_IDX: usize = 11;
/// Index of `stime` in the same suffix (kernel field 15 → suffix index 12).
const STIME_FIELD_IDX: usize = 12;

fn clock_ticks_per_second() -> Option<u128> {
    // Cache only a successful tick rate. A transient `getconf` failure must
    // not permanently disable CPU metering for the process.
    static TICKS: OnceLock<u128> = OnceLock::new();
    if let Some(ticks) = TICKS.get() {
        return Some(*ticks);
    }
    let output = Command::new(GETCONF).arg(CLK_TCK).output().ok()?;
    let ticks = output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok()?.trim().parse().ok())
        .flatten()
        .filter(|ticks| *ticks != 0)?;
    let _ = TICKS.set(ticks);
    Some(ticks)
}

/// Sum of on-CPU nanoseconds across every thread of this process, or `None`
/// on a platform without Linux procfs / `getconf`.
pub fn process_cpu_ns() -> Option<u128> {
    let raw = fs::read_to_string(PROC_SELF_STAT).ok()?;
    // Field 2 (`comm`) is parenthesized and may contain spaces. Fields after
    // its closing ')' start at field 3 (`state`), so utime/stime (14/15) are
    // indexes 11/12 in this suffix.
    let close = raw.rfind(')')?;
    let fields: Vec<&str> = raw[close + 1..].split_whitespace().collect();
    let user_ticks = fields.get(UTIME_FIELD_IDX)?.parse::<u128>().ok()?;
    let system_ticks = fields.get(STIME_FIELD_IDX)?.parse::<u128>().ok()?;
    // Reject zero so a bogus `getconf` value cannot panic on divide.
    let ticks_per_second = clock_ticks_per_second().filter(|ticks| *ticks != 0)?;
    Some((user_ticks + system_ticks) * NS_PER_SEC / ticks_per_second)
}

/// Non-negative CPU seconds elapsed since a `process_cpu_ns()` snapshot.
/// `None` if either snapshot is unavailable.
pub fn cpu_seconds_since(start_ns: Option<u128>) -> Option<f64> {
    let start = start_ns?;
    let end = process_cpu_ns()?;
    Some(end.saturating_sub(start) as f64 / NS_PER_SEC as f64)
}

/// Run `f`, returning `(result, wall_duration, measured_on_cpu_seconds)`.
///
/// Wall and on-CPU time are bracketed identically around the same work.
/// `cpu` is `None` only when procfs sampling is unavailable.
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration, Option<f64>) {
    let cpu0 = process_cpu_ns();
    let t0 = Instant::now();
    let out = f();
    let wall = t0.elapsed();
    (out, wall, cpu_seconds_since(cpu0))
}

#[cfg(test)]
mod tests {
    use std::{hint::black_box, thread};

    use super::*;

    /// Short-lived workers used to prove process CPU remains monotonic.
    const EXITING_WORKERS: usize = 4;
    /// Busy / sleep window used by the on-CPU vs sleep comparison.
    const TEST_WORK_WINDOW: Duration = Duration::from_millis(50);
    /// Inner loop iterations per busy spin.
    const TEST_BUSY_INNER_ITERS: u64 = 4_096;
    /// Knuth multiplicative hash constant for the busy spin.
    const TEST_BUSY_MIX_CONST: u64 = 2_654_435_761;
    /// Minimum on-CPU seconds expected from a busy window.
    const TEST_MIN_BUSY_CPU_S: f64 = 0.010;
    /// Sleep must accrue less than busy / this factor.
    const TEST_SLEEP_VS_BUSY_FACTOR: f64 = 5.0;
    /// Wall sleep used by `timed` coverage.
    const TEST_TIMED_SLEEP: Duration = Duration::from_millis(20);
    /// Per-thread busy window for the exited-worker retention test.
    const TEST_WORKER_BUSY_WINDOW: Duration = Duration::from_millis(50);
    /// Minimum process CPU seconds from exited workers.
    const TEST_MIN_EXITED_WORKER_CPU_S: f64 = 0.01;
    /// Thread-local schedstat path (Linux).
    const PROC_THREAD_SELF_SCHEDSTAT: &str = "/proc/thread-self/schedstat";

    /// The calling thread's own on-CPU nanoseconds.
    fn thread_cpu_ns() -> Option<u128> {
        let raw = fs::read_to_string(PROC_THREAD_SELF_SCHEDSTAT).ok()?;
        raw.split_whitespace().next()?.parse().ok()
    }

    #[test]
    fn on_cpu_time_excludes_sleep() {
        let Some(start_busy) = thread_cpu_ns() else {
            return;
        };
        let t = Instant::now();
        let mut acc = 0u64;
        while t.elapsed() < TEST_WORK_WINDOW {
            for i in 0..TEST_BUSY_INNER_ITERS {
                acc = acc.wrapping_add(i.wrapping_mul(TEST_BUSY_MIX_CONST));
            }
        }
        black_box(acc);
        let busy = (thread_cpu_ns().expect("cpu") - start_busy) as f64 / NS_PER_SEC as f64;

        let start_sleep = thread_cpu_ns().expect("cpu");
        thread::sleep(TEST_WORK_WINDOW);
        let slept = (thread_cpu_ns().expect("cpu") - start_sleep) as f64 / NS_PER_SEC as f64;

        assert!(
            busy > TEST_MIN_BUSY_CPU_S,
            "50ms of busy work must accrue on-CPU time, got {busy}s"
        );
        assert!(
            slept < busy / TEST_SLEEP_VS_BUSY_FACTOR,
            "a sleep ({slept}s) must accrue far less on-CPU time than busy work ({busy}s)"
        );
    }

    #[test]
    fn timed_returns_value_wall_and_cpu() {
        let (value, wall, cpu) = timed(|| {
            thread::sleep(TEST_TIMED_SLEEP);
            42u64
        });
        assert_eq!(value, 42);
        assert!(wall >= TEST_TIMED_SLEEP);
        if process_cpu_ns().is_some() {
            assert!(cpu.expect("procfs host measures cpu") >= 0.0);
        }
    }

    #[test]
    fn process_cpu_retains_exited_worker_time() {
        let Some(before) = process_cpu_ns() else {
            return;
        };
        let workers: Vec<_> = (0..EXITING_WORKERS)
            .map(|_| {
                thread::spawn(|| {
                    let started = Instant::now();
                    let mut value = 0u64;
                    while started.elapsed() < TEST_WORKER_BUSY_WINDOW {
                        value = value.wrapping_mul(31).wrapping_add(17);
                        black_box(value);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().expect("worker");
        }
        let elapsed = cpu_seconds_since(Some(before)).expect("procfs CPU");
        assert!(
            elapsed > TEST_MIN_EXITED_WORKER_CPU_S,
            "exited workers must remain in process CPU total, got {elapsed}s"
        );
    }
}
