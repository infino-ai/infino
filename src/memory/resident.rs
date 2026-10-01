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
//! `memory.max` from its cgroup v2 directory up to the root. With neither,
//! there is no limit.
//!
//! One sampler thread serves the whole process. While at least one statement is
//! watching it reads `RssAnon` every [`SAMPLE_EVERY`] and publishes the reading
//! on a `watch` channel; with none watching it parks. A plain thread, not a
//! runtime timer: a statement's CPU-bound partitions can hold every runtime
//! worker for a whole batch, and a timer needs a free worker to fire.

use std::{
    fs,
    future::pending,
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    thread::{self, Thread},
    time::Duration,
};

use datafusion::error::DataFusionError;
use tokio::sync::watch;
use tracing::{info, warn};

use crate::{config, runtime_metrics::rss::status_anon_rss_bytes};

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

/// Where the cgroup v2 hierarchy is mounted.
const CGROUP_ROOT: &str = "/sys/fs/cgroup";

/// The process's cgroup membership. Its cgroup v2 entry reads `0::<path>`.
const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";

/// The prefix of the cgroup v2 entry in [`PROC_SELF_CGROUP`].
const CGROUP_V2_PREFIX: &str = "0::";

/// The cgroup v2 files that hold a memory limit: the throttle and the kill.
/// Either reads `max` when unset.
const CGROUP_LIMIT_FILES: [&str; 2] = ["memory.high", "memory.max"];

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
/// see what SQL is held to.
fn resolve_limit() -> Option<u64> {
    if let Some(bytes) = config::global().memory.process_limit_bytes {
        let limit = (bytes > 0).then_some(bytes);
        info!(?limit, "SQL process memory limit from config");
        return limit;
    }
    let cgroup = fs::read_to_string(PROC_SELF_CGROUP)
        .ok()
        .and_then(|membership| cgroup_memory_limit(Path::new(CGROUP_ROOT), &membership));
    let limit = cgroup.map(|bytes| bytes / PERCENT * CGROUP_LIMIT_PERCENT);
    info!(?cgroup, ?limit, "SQL process memory limit from the cgroup");
    limit
}

/// The lowest memory limit set on the cgroup v2 path `membership` names or on
/// any of its ancestors, read under `root`. `None` when the membership has no
/// cgroup v2 entry or no level sets a limit.
fn cgroup_memory_limit(root: &Path, membership: &str) -> Option<u64> {
    let path = membership
        .lines()
        .find_map(|line| line.strip_prefix(CGROUP_V2_PREFIX))?;
    let mut dir = PathBuf::from(path.trim());
    let mut lowest: Option<u64> = None;
    loop {
        let level = root.join(dir.strip_prefix("/").unwrap_or(dir.as_path()));
        for file in CGROUP_LIMIT_FILES {
            // `max` (no limit) does not parse, and a missing file sets nothing.
            let bytes = fs::read_to_string(level.join(file))
                .ok()
                .and_then(|value| value.trim().parse::<u64>().ok());
            if let Some(bytes) = bytes {
                lowest = Some(lowest.map_or(bytes, |lower| lower.min(bytes)));
            }
        }
        if !dir.pop() {
            return lowest;
        }
    }
}

/// The shared sampler: the channel its readings go out on, and its thread, to
/// wake when a statement starts watching.
struct Sampler {
    readings: Arc<watch::Sender<u64>>,
    thread: Thread,
}

/// Started on first use. `None` when the thread could not be spawned; a
/// statement then gets the check at its start only.
static SAMPLER: OnceLock<Option<Sampler>> = OnceLock::new();

fn sampler() -> Option<&'static Sampler> {
    SAMPLER.get_or_init(start_sampler).as_ref()
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
                "the SQL memory sampler did not start; the process memory limit is \
                 checked only when a statement starts"
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
    use tempfile::TempDir;
    use tokio::{runtime::Runtime, time::timeout};

    use super::*;

    /// Longer than several samples, so a limit that was going to trip would
    /// have.
    const SETTLE: Duration = Duration::from_millis(500);
    /// A worker's cgroup path, as `/proc/self/cgroup` names it.
    const WORKER_CGROUP: &str = "0::/system.slice/worker.service\n";
    /// The worker unit's throttle (`memory.high`).
    const WORKER_HIGH: &str = "7516192768";
    /// The worker unit's kill (`memory.max`).
    const WORKER_MAX: &str = "8589934592";
    /// A parent slice's limit, tighter than the worker's own.
    const PARENT_MAX: &str = "1000";

    /// A fake cgroup v2 hierarchy: `files` are (path under the root, content).
    fn hierarchy(files: &[(&str, &str)]) -> TempDir {
        let root = TempDir::new().expect("tempdir");
        for (path, content) in files {
            let file = root.path().join(path);
            fs::create_dir_all(file.parent().expect("a file under the root"))
                .expect("create cgroup dir");
            fs::write(file, content).expect("write cgroup file");
        }
        root
    }

    #[test]
    fn the_cgroup_limit_is_the_lowest_set_on_the_path() {
        let root = hierarchy(&[
            ("system.slice/memory.high", "max"),
            ("system.slice/memory.max", "max"),
            ("system.slice/worker.service/memory.high", WORKER_HIGH),
            ("system.slice/worker.service/memory.max", WORKER_MAX),
        ]);
        assert_eq!(
            cgroup_memory_limit(root.path(), WORKER_CGROUP),
            Some(WORKER_HIGH.parse().expect("bytes"))
        );
    }

    #[test]
    fn a_parent_limit_tighter_than_the_cgroups_own_wins() {
        let root = hierarchy(&[
            ("system.slice/memory.max", PARENT_MAX),
            ("system.slice/worker.service/memory.max", WORKER_MAX),
        ]);
        assert_eq!(
            cgroup_memory_limit(root.path(), WORKER_CGROUP),
            Some(PARENT_MAX.parse().expect("bytes"))
        );
    }

    #[test]
    fn no_limit_anywhere_or_no_cgroup_v2_entry_sets_none() {
        let root = hierarchy(&[
            ("system.slice/memory.max", "max"),
            ("system.slice/worker.service/memory.high", "max"),
            ("system.slice/worker.service/memory.max", "max"),
        ]);
        assert_eq!(cgroup_memory_limit(root.path(), WORKER_CGROUP), None);
        // cgroup v1 only: no `0::` entry to follow.
        assert_eq!(cgroup_memory_limit(root.path(), "4:memory:/worker\n"), None);
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
