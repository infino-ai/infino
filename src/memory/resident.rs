// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A process-wide anonymous-memory ceiling for SQL statements.
//!
//! The connection budget counts reservations, and the batches a SQL plan
//! streams between its operators are never reserved. This reads what a
//! container or cgroup memory limit kills on instead — the process's anonymous
//! resident memory — and ends a running statement once it passes the
//! connection's ceiling
//! ([`ConnectOptions::with_process_memory_limit_bytes`](crate::ConnectOptions::with_process_memory_limit_bytes)).
//!
//! One sampler thread serves the whole process. While at least one statement is
//! watching it reads `RssAnon` every [`SAMPLE_EVERY`] and publishes the reading
//! on a `watch` channel; with none watching it parks. A plain thread, not a
//! runtime timer: a statement's CPU-bound partitions can hold every runtime
//! worker for a whole batch, and a timer needs a free worker to fire.

use std::{
    future::pending,
    sync::{Arc, OnceLock},
    thread::{self, Thread},
    time::Duration,
};

use datafusion::error::DataFusionError;
use tokio::sync::watch;

use crate::runtime_metrics::rss::status_anon_rss_bytes;

/// How often the sampler reads the process's anonymous memory while a
/// statement is watching. Growth that lasts longer than this is seen; a
/// shorter spike may not be.
const SAMPLE_EVERY: Duration = Duration::from_millis(100);

/// The sampler thread's name, as a stack dump shows it.
const SAMPLER_THREAD_NAME: &str = "infino-rss-sampler";

/// The process ceiling a SQL session carries, as a DataFusion session-config
/// extension: set from the connection's budget by `budgeted_session_context`,
/// read where the plan is collected.
#[derive(Debug)]
pub(crate) struct ProcessMemoryLimit(pub(crate) u64);

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
    let handle = thread::Builder::new()
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
        })
        .ok()?;
    Some(Sampler {
        readings,
        thread: handle.thread().clone(),
    })
}

/// The process's anonymous resident bytes, if they are over `limit` now: one
/// read, no sampler. `None` when they are not, or cannot be read.
pub(crate) fn process_over_limit(limit: u64) -> Option<u64> {
    status_anon_rss_bytes().filter(|&anon| anon > limit)
}

/// Resolves with the process's anonymous resident bytes once they pass
/// `limit`, and never otherwise. It checks once immediately, then follows the
/// sampler. Where the reading is unavailable (not Linux) it never resolves.
pub(crate) async fn process_limit_exceeded(limit: u64) -> u64 {
    if let Some(anon) = process_over_limit(limit) {
        return anon;
    }
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

/// The refusal a statement ends with when the process passes its ceiling. It
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
    use tokio::{runtime::Runtime, time::timeout};

    use super::*;

    /// Longer than several samples, so a limit that was going to trip would
    /// have.
    const SETTLE: Duration = Duration::from_millis(500);

    #[cfg(target_os = "linux")]
    #[test]
    fn a_limit_the_process_is_already_over_resolves_at_once() {
        // Any live process holds more than one byte of anonymous memory.
        let runtime = Runtime::new().expect("tokio runtime");
        let anon = runtime
            .block_on(async { timeout(SETTLE, process_limit_exceeded(1)).await })
            .expect("the check at the start resolves without waiting for a sample");
        assert!(anon > 1, "the reading that tripped is returned: {anon}");
    }

    #[test]
    fn a_limit_above_any_process_never_resolves() {
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
