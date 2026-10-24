// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The process memory limit for SQL statements.
//!
//! The connection budget counts reservations, and the batches a SQL plan
//! streams between its operators are never reserved. This reads what a
//! container or cgroup memory limit kills on instead — the process's anonymous
//! resident memory — and ends a running statement once it passes the
//! process's limit.
//!
//! The limit is resolved once per process: `memory.process_limit_bytes` from
//! config when it is set (`0` sets none), otherwise [`CGROUP_LIMIT_PERCENT`]
//! of the process's cgroup memory limit — the lowest `memory.high` or
//! `memory.max` from its cgroup v2 directory up to the root
//! (`runtime_metrics::rss::cgroup_memory_limit_bytes`). With neither, there
//! is no limit.
//!
//! One sampler thread serves the whole process. While at least one statement is
//! watching it reads `RssAnon` every [`SAMPLE_EVERY`] and publishes the reading
//! on a `watch` channel; with none watching it parks. A plain thread, not a
//! runtime timer: a statement's CPU-bound partitions can hold every runtime
//! worker for a whole batch, and a timer needs a free worker to fire.

use std::{
    future::pending,
    sync::{
        Arc, Mutex, OnceLock, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, Thread},
    time::Duration,
};

use datafusion::error::DataFusionError;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::{
    config,
    runtime_metrics::rss::{cgroup_memory_limit_bytes, status_anon_rss_bytes},
};

/// How often the sampler reads the process's anonymous memory while a
/// statement is watching. Growth that lasts longer than this is seen; a
/// shorter spike may not be.
const SAMPLE_EVERY: Duration = Duration::from_millis(100);

/// The sampler thread's name, as a stack dump shows it.
const SAMPLER_THREAD_NAME: &str = "infino-rss-sampler";

/// The share of the cgroup memory limit the process limit is set at: the rest
/// is left for the page cache and for what is already in flight when a
/// statement is refused.
const CGROUP_LIMIT_PERCENT: u64 = 90;

/// Divisor turning a percent into a fraction.
const PERCENT: u64 = 100;

/// The stored limit meaning "none". No real limit is 0 bytes: config's `0`
/// means none as well.
const NO_LIMIT: u64 = 0;

/// The process's limit, resolved on first use (see [`resolve_limit`]).
static LIMIT: OnceLock<AtomicU64> = OnceLock::new();

fn limit_cell() -> &'static AtomicU64 {
    LIMIT.get_or_init(|| AtomicU64::new(resolve_limit().unwrap_or(NO_LIMIT)))
}

/// The process's SQL memory limit in bytes, if it has one.
pub(crate) fn process_limit() -> Option<u64> {
    let limit = limit_cell().load(Ordering::Relaxed);
    (limit != NO_LIMIT).then_some(limit)
}

/// Replace the resolved limit; `None` sets none.
pub(crate) fn set_process_limit(limit: Option<u64>) {
    limit_cell().store(limit.unwrap_or(NO_LIMIT), Ordering::Relaxed);
}

/// The limit from config, else from the cgroup, logged once so an operator can
/// see what SQL is held to. A limit the process cannot be measured against
/// would look enforced and never be, so where `RssAnon` cannot be read there
/// is none, and a warning says so.
fn resolve_limit() -> Option<u64> {
    let limit = configured_or_cgroup_limit();
    if limit.is_some() && status_anon_rss_bytes().is_none() {
        warn!(
            ?limit,
            "SQL process memory limit disabled: this platform does not report the process's \
             anonymous resident memory"
        );
        return None;
    }
    limit
}

fn configured_or_cgroup_limit() -> Option<u64> {
    if let Some(bytes) = config::global().memory.process_limit_bytes {
        let limit = (bytes > 0).then_some(bytes);
        info!(?limit, "SQL process memory limit from config");
        return limit;
    }
    let cgroup = cgroup_memory_limit_bytes();
    let limit = cgroup.map(share_of_cgroup_limit);
    info!(?cgroup, ?limit, "SQL process memory limit from the cgroup");
    limit
}

/// The SQL limit for a cgroup limit of `bytes`: [`CGROUP_LIMIT_PERCENT`] of
/// it, exact across the whole range, and never 0, which would read as no
/// limit. The whole hundreds and the remainder are scaled apart: scaling all
/// of `bytes` first would overflow near `u64::MAX`, and dividing first would
/// round a limit under 100 bytes away.
fn share_of_cgroup_limit(bytes: u64) -> u64 {
    let share =
        bytes / PERCENT * CGROUP_LIMIT_PERCENT + bytes % PERCENT * CGROUP_LIMIT_PERCENT / PERCENT;
    share.max(1)
}

/// The shared sampler: the channel its readings go out on, and its thread, to
/// wake when a statement starts watching.
struct Sampler {
    readings: Arc<watch::Sender<u64>>,
    thread: Thread,
}

/// Started on first use. A spawn can fail for a moment (a thread limit, memory
/// pressure), so a failure is not remembered: that statement gets the check at
/// its start only, and the next one tries again.
static SAMPLER: OnceLock<Sampler> = OnceLock::new();

/// Held while the sampler starts, so two statements racing to start it cannot
/// each spawn a thread; the one that lost would park forever.
static SAMPLER_STARTING: Mutex<()> = Mutex::new(());

fn sampler() -> Option<&'static Sampler> {
    start_once(&SAMPLER, &SAMPLER_STARTING, start_sampler)
}

/// The value in `cell`, starting it with `start` if it is empty. A `start`
/// that returns `None` leaves `cell` empty for the next call to retry, and
/// `starting` keeps two calls from starting it at once.
fn start_once<T>(
    cell: &'static OnceLock<T>,
    starting: &Mutex<()>,
    start: impl FnOnce() -> Option<T>,
) -> Option<&'static T> {
    if let Some(started) = cell.get() {
        return Some(started);
    }
    let _starting = starting.lock().unwrap_or_else(PoisonError::into_inner);
    // Another call may have started it while this one waited for the lock.
    if let Some(started) = cell.get() {
        return Some(started);
    }
    let started = start()?;
    Some(cell.get_or_init(|| started))
}

fn start_sampler() -> Option<Sampler> {
    let (sender, _) = watch::channel(0);
    let readings = Arc::new(sender);
    let publish = Arc::clone(&readings);
    let spawned = thread::Builder::new()
        .name(SAMPLER_THREAD_NAME.to_string())
        .spawn(move || {
            loop {
                // Nobody watching: sleep until a statement subscribes and
                // unparks this thread. An unpark that lands before the park
                // makes the park return at once, so no wake is lost.
                if publish.receiver_count() == 0 {
                    thread::park();
                    continue;
                }
                if let Some(anon) = status_anon_rss_bytes() {
                    publish.send_replace(anon);
                }
                thread::sleep(SAMPLE_EVERY);
            }
        });
    match spawned {
        Ok(handle) => Some(Sampler {
            readings,
            thread: handle.thread().clone(),
        }),
        Err(error) => {
            warn!(
                %error,
                "the SQL memory sampler did not start; this statement is checked against \
                 the process memory limit only at its start, and the next one retries"
            );
            None
        }
    }
}

/// The process's anonymous resident bytes, if they are over `limit` now: one
/// read, no sampler. `None` when they are not, or cannot be read.
pub(crate) fn process_over_limit(limit: u64) -> Option<u64> {
    status_anon_rss_bytes().filter(|&anon| anon > limit)
}

/// Resolves with the process's anonymous resident bytes once a sample passes
/// `limit`, and never otherwise. The caller checks the process once before it
/// starts ([`process_over_limit`]); this follows the sampler from there. Where
/// the reading is unavailable (not Linux) it never resolves.
pub(crate) async fn process_limit_exceeded(limit: u64) -> u64 {
    let Some(sampler) = sampler() else {
        return pending().await;
    };
    // A new receiver has already seen the current reading, which may be old,
    // so only readings taken from here on are compared.
    let mut readings = sampler.readings.subscribe();
    sampler.thread.unpark();
    while readings.changed().await.is_ok() {
        let anon = *readings.borrow_and_update();
        if anon > limit {
            return anon;
        }
    }
    pending().await
}

/// The refusal a statement ends with when the process passes its limit. It
/// goes out as `ResourcesExhausted`, the channel the memory pool refuses
/// through, so it reaches the caller as an over-budget error.
pub(crate) fn over_process_limit(anon: u64, limit: u64) -> DataFusionError {
    DataFusionError::ResourcesExhausted(format!(
        "during SQL query, process anonymous resident memory {anon} B is over the \
         {limit} B process memory limit"
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::{Barrier, atomic::AtomicUsize};

    use tokio::{runtime::Runtime, time::timeout};

    use super::*;

    /// Longer than several samples, so a limit that was going to trip would
    /// have.
    const SETTLE: Duration = Duration::from_millis(500);
    /// What the test starters return once they succeed.
    const STARTED: u32 = 7;
    /// Calls racing to start the same value at once.
    const RACERS: usize = 8;
    /// How long a racing start takes: long enough that, without the lock,
    /// every racer would find the value missing and start it too.
    const START_TAKES: Duration = Duration::from_millis(50);
    /// The worker unit's throttle (`memory.high`).
    const WORKER_HIGH: u64 = 7_516_192_768;
    /// A limit that is not a whole number of hundreds, so the remainder counts.
    const NOT_WHOLE_HUNDREDS: u64 = 199;

    /// The share computed in 128 bits, where nothing can overflow: the
    /// reference the 64-bit split is checked against.
    fn wide_share(bytes: u64) -> u64 {
        let share = u128::from(bytes) * u128::from(CGROUP_LIMIT_PERCENT) / u128::from(PERCENT);
        u64::try_from(share).expect("a share of a u64 fits in a u64")
    }

    #[test]
    fn the_sql_limit_is_ninety_percent_of_the_cgroups_and_never_rounds_to_none() {
        for bytes in [WORKER_HIGH, NOT_WHOLE_HUNDREDS, u64::MAX - 1, u64::MAX] {
            assert_eq!(share_of_cgroup_limit(bytes), wide_share(bytes), "{bytes}");
        }
        // Below 100 bytes the exact share is 0, which would read as no limit.
        assert_eq!(share_of_cgroup_limit(1), 1);
    }

    #[test]
    fn a_start_that_fails_is_retried_and_one_that_succeeds_is_kept() {
        static CELL: OnceLock<u32> = OnceLock::new();
        static STARTING: Mutex<()> = Mutex::new(());
        assert_eq!(start_once(&CELL, &STARTING, || None), None);
        assert_eq!(
            start_once(&CELL, &STARTING, || Some(STARTED)),
            Some(&STARTED)
        );
        assert_eq!(
            start_once(&CELL, &STARTING, || panic!("started a second time")),
            Some(&STARTED)
        );
    }

    #[test]
    fn racing_starts_start_once() {
        static CELL: OnceLock<u32> = OnceLock::new();
        static STARTING: Mutex<()> = Mutex::new(());
        let starts = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(Barrier::new(RACERS));
        let racers: Vec<_> = (0..RACERS)
            .map(|_| {
                let starts = Arc::clone(&starts);
                let ready = Arc::clone(&ready);
                thread::spawn(move || {
                    ready.wait();
                    start_once(&CELL, &STARTING, || {
                        starts.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(START_TAKES);
                        Some(STARTED)
                    })
                    .copied()
                })
            })
            .collect();
        for racer in racers {
            assert_eq!(racer.join().expect("racer"), Some(STARTED));
        }
        assert_eq!(starts.load(Ordering::SeqCst), 1);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_limit_the_process_is_over_resolves_at_the_first_sample() {
        // Any live process holds more than one byte of anonymous memory.
        assert!(process_over_limit(1).is_some());
        let runtime = Runtime::new().expect("tokio runtime");
        let anon = runtime
            .block_on(async { timeout(SETTLE, process_limit_exceeded(1)).await })
            .expect("the first sample passes a 1-byte limit");
        assert!(anon > 1, "the reading that tripped is returned: {anon}");
    }

    #[test]
    fn a_limit_above_any_process_never_resolves() {
        assert_eq!(process_over_limit(u64::MAX), None);
        let runtime = Runtime::new().expect("tokio runtime");
        let waited =
            runtime.block_on(async { timeout(SETTLE, process_limit_exceeded(u64::MAX)).await });
        assert!(waited.is_err(), "no reading passes u64::MAX: {waited:?}");
    }

    #[test]
    fn the_refusal_is_a_resources_exhausted_error() {
        assert!(matches!(
            over_process_limit(2, 1),
            DataFusionError::ResourcesExhausted(msg) if msg.contains("process memory limit")
        ));
    }
}
