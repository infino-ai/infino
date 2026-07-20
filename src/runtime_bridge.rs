// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Sync→async bridge for sync public API on top of async storage.
//!
//! The supertable's public surface (writer.commit, reader queries,
//! tombstone-cache refresh, lazy byte-source range fetches) is sync,
//! but the storage trait + downstream object_store calls are async.
//! Every call site that crosses that boundary lands here.
//!
//! ## Two modes
//!
//! - **Ambient `multi_thread` tokio runtime present** — `block_in_place`
//!   tells the scheduler "I'm about to block this worker; rearrange,"
//!   then `Handle::block_on` drives the future on the current thread.
//!   Other tasks keep making progress on sibling workers.
//! - **No ambient runtime** — build a one-shot `current_thread` runtime
//!   and drive the future on it. Sync callers (CLI tools, rayon
//!   workers, Python bindings via PyO3) land here.
//!
//! ## Unsupported: `current_thread` ambient runtime
//!
//! `tokio::task::block_in_place` requires `multi_thread`. If a caller
//! invokes this from inside a `current_thread` tokio runtime,
//! `Handle::try_current()` returns `Ok(...)`, we take the
//! `block_in_place` branch, and tokio panics. There is no good
//! detection primitive in tokio's public API for "this handle is from
//! a current_thread runtime"; surfacing a typed error would require
//! parsing `format!("{handle:?}")` or shipping our own probe. For now
//! this is a documented requirement: async callers must run on a
//! `multi_thread` runtime (the default for `#[tokio::main]`, axum,
//! actix, etc.).

use std::{
    future::Future,
    sync::{Arc, OnceLock},
    thread,
};

use tokio::{
    runtime::{self, Handle, Runtime},
    task::block_in_place,
};

/// Fallback worker count for [`build_query_runtime`] when the host's available
/// parallelism can't be determined.
const FALLBACK_QUERY_RUNTIME_WORKERS: usize = 4;

/// Drive `fut` to completion from a sync context. Uses the ambient
/// tokio runtime if present (via `block_in_place + Handle::block_on`),
/// otherwise builds a tiny `current_thread` runtime for the call.
///
/// Panics if called from inside a `current_thread` tokio runtime
/// (`block_in_place` requires `multi_thread`). See the module-level
/// docs.
pub(crate) fn bridge_sync_to_async<F, T>(fut: F) -> T
where
    F: Future<Output = T>,
{
    match runtime::Handle::try_current() {
        Ok(handle) => block_in_place(|| handle.block_on(fut)),
        Err(_) => build_current_thread_runtime().block_on(fut),
    }
}

/// Drive `fut` to completion from a sync context using a
/// caller-supplied runtime for the no-ambient-runtime case (instead of
/// a throwaway). The supertable query path passes its pooled
/// `query_runtime` here so a sync query issued from a plain thread
/// reuses the shared multi-thread runtime rather than spinning up a
/// one-shot one per call.
///
/// Same `current_thread`-ambient caveat as [`bridge_sync_to_async`].
pub(crate) fn bridge_on_runtime<F: Future>(fut: F, runtime: &Runtime) -> F::Output {
    // Always drive on the passed `runtime` — callers hand us the runtime the
    // future's async resources are bound to (e.g. `query_runtime`, where the
    // disk cache's coordination lives). Driving on a *different* ambient
    // runtime instead awaits those resources cross-runtime and can lose the
    // wakeup → deadlock (e.g. a cold disk-cache fetch during search). When an
    // ambient runtime is present, escape its worker via `block_in_place` so the
    // nested `block_on` is legal.
    match Handle::try_current() {
        Ok(_ambient) => block_in_place(|| runtime.handle().block_on(fut)),
        Err(_) => runtime.block_on(fut),
    }
}

/// Robust sync→async bridge for `Send + 'static` futures. Unlike
/// [`bridge_sync_to_async`], this also handles being called from inside
/// a `current_thread` runtime (where `block_in_place` is illegal and a
/// nested runtime would panic) by driving the future on a dedicated
/// thread. Used by the lazy byte-source range fetches, which can be
/// called from any context — a multi-thread query worker, a rayon
/// reader-pool thread (no ambient runtime), or a `current_thread`
/// test runtime.
///
/// Three contexts:
/// - **multi_thread ambient** — `block_in_place` + `Handle::block_on`.
/// - **`current_thread` ambient** — drive on a spawned thread with its
///   own one-shot `current_thread` runtime (can't block_in_place / nest).
/// - **no ambient runtime** — build a one-shot `current_thread` runtime
///   inline (no extra thread; this is the rayon-worker / CLI path).
pub(crate) fn bridge_sync_to_async_send<F, T>(fut: F) -> T
where
    F: Future<Output = T> + Send + 'static,
    T: Send + 'static,
{
    match runtime::Handle::try_current() {
        Ok(handle) if matches!(handle.runtime_flavor(), runtime::RuntimeFlavor::MultiThread) => {
            block_in_place(|| handle.block_on(fut))
        }
        Ok(_) => thread::spawn(move || build_current_thread_runtime().block_on(fut))
            .join()
            .expect("sync→async bridge worker thread panicked"),
        Err(_) => build_current_thread_runtime().block_on(fut),
    }
}

/// Process-wide query runtime, shared by every `Connection` and
/// `Supertable` so open handles add zero tokio workers. Never shut
/// down — the static keeps a reference until process exit, so handle
/// drops from inside a caller's async context have nothing to tear down.
static SHARED_IO_RUNTIME: OnceLock<Arc<runtime::Runtime>> = OnceLock::new();

pub(crate) fn shared_io_runtime() -> Arc<runtime::Runtime> {
    Arc::clone(SHARED_IO_RUNTIME.get_or_init(|| build_query_runtime("infino-io")))
}

/// Shared multi-thread runtime for driving the sync query API's async I/O.
///
/// Multi-thread is required, not just preferred: the bridges above take the
/// `block_in_place` branch once this is the ambient runtime, and
/// `block_in_place` panics on a `current_thread` runtime. Workers scale to
/// the CPU count so a cold query's per-superfile fan-out overlaps instead
/// of serializing.
fn build_query_runtime(thread_name: &str) -> Arc<runtime::Runtime> {
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(FALLBACK_QUERY_RUNTIME_WORKERS);
    Arc::new(
        runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .thread_name(thread_name)
            .build()
            .expect(
                "invariant: tokio Runtime build only fails on \
                 catastrophic OS resource exhaustion",
            ),
    )
}

fn build_current_thread_runtime() -> runtime::Runtime {
    runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect(
            "invariant: tokio Runtime build only fails on \
             catastrophic OS resource exhaustion",
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this pins: `bridge_on_runtime` once preferred the
    /// AMBIENT runtime when one was present, driving the future on a
    /// runtime other than the one its async resources are bound to.
    /// Awaiting those resources cross-runtime can lose the wakeup — the
    /// cold disk-cache fetch during a sync search deadlocked exactly
    /// this way. The contract: always drive on the PASSED runtime; an
    /// ambient runtime only decides whether `block_in_place` is needed
    /// to make the nested `block_on` legal.
    #[test]
    fn bridge_on_runtime_drives_on_the_passed_runtime() {
        let owned = build_query_runtime("bridge-test");
        let owned_id = format!("{:?}", owned.handle().id());

        // No ambient runtime: drives on the passed runtime.
        let seen = bridge_on_runtime(async { format!("{:?}", Handle::current().id()) }, &owned);
        assert_eq!(
            seen, owned_id,
            "no-ambient drive must use the passed runtime"
        );

        // Ambient multi-thread runtime present: the drive must STILL land on
        // the passed runtime — under the old ambient-preferring behavior this
        // assertion fails with the ambient's id.
        let ambient = build_query_runtime("bridge-test-ambient");
        let owned_for_task = Arc::clone(&owned);
        let seen = ambient.block_on(async move {
            tokio::spawn(async move {
                bridge_on_runtime(
                    async { format!("{:?}", Handle::current().id()) },
                    &owned_for_task,
                )
            })
            .await
            .expect("bridge task")
        });
        assert_eq!(
            seen, owned_id,
            "an ambient runtime must not capture the drive"
        );
    }
}
