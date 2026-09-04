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
    fmt,
    future::Future,
    sync::{Arc, OnceLock},
    thread,
};

use rayon::ThreadPool;
use tokio::{
    runtime::{self, Handle, Runtime},
    sync::oneshot,
    task::block_in_place,
};
use tracing::{Instrument, Span};

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
        Ok(_) => {
            let fut = fut.in_current_span();
            thread::spawn(move || build_current_thread_runtime().block_on(fut))
                .join()
                .expect("sync→async bridge worker thread panicked")
        }
        Err(_) => build_current_thread_runtime().block_on(fut),
    }
}

/// Async→rayon bridge: dispatch CPU work onto the configured reader pool
pub(crate) fn spawn_on<F: FnOnce() + Send + 'static>(pool: Option<&ThreadPool>, f: F) {
    let f = carry_span(f);
    match pool {
        Some(pool) => pool.spawn(f),
        None => rayon::spawn(f),
    }
}

/// Wrap `f` so it runs inside the caller's tracing span on whatever thread ends
/// up executing it.
///
/// A pool or thread hop does not carry the span with it, so without this the log
/// lines a unit of work emits lose the context of the operation that scheduled
/// it. Use it at every hand-off to another thread.
pub(crate) fn carry_span<T, F>(f: F) -> impl FnOnce() -> T + Send
where
    F: FnOnce() -> T + Send,
{
    let span = Span::current();
    move || {
        let _entered = span.enter();
        f()
    }
}

/// The pool task's sender was dropped before sending a result surfaced as
/// an error rather than a second panic.
#[derive(Debug)]
pub(crate) struct PoolDropped(pub(crate) &'static str);

impl fmt::Display for PoolDropped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for PoolDropped {}

/// Run `f` on `pool` (or the global rayon pool if `None`) and bridge the
/// result back to the awaiting async task via a oneshot, so no tokio worker
/// blocks under the compute. `what` becomes the [`PoolDropped`] message if
/// the pool task dies before sending
pub(crate) async fn run_on_pool<R, F>(
    pool: Option<&ThreadPool>,
    what: &'static str,
    f: F,
) -> Result<R, PoolDropped>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    spawn_on(pool, move || {
        let _ = tx.send(f());
    });
    rx.await.map_err(|_| PoolDropped(what))
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
    use std::sync::mpsc;

    use super::*;

    /// A span is only resolvable from another thread when the subscriber is the
    /// process-wide default — a thread-local one is invisible to a pool worker,
    /// which is also why these assertions would pass vacuously without it.
    /// Installed once for the binary; nothing else here sets one.
    fn assign_span_ids() {
        static INSTALLED: OnceLock<()> = OnceLock::new();
        INSTALLED.get_or_init(|| {
            let _ = tracing::subscriber::set_global_default(tracing_subscriber::registry());
        });
    }

    #[test]
    fn pool_work_runs_inside_the_span_that_dispatched_it() {
        assign_span_ids();
        let span = tracing::info_span!("dispatching");
        let _entered = span.enter();
        let dispatcher = Span::current().id();
        assert!(dispatcher.is_some(), "span ids must be assigned");

        let (tx, rx) = mpsc::channel();
        spawn_on(None, move || {
            let _ = tx.send(Span::current().id());
        });

        assert_eq!(
            rx.recv().expect("pool task ran"),
            dispatcher,
            "a rayon hop must not drop the caller's span"
        );
    }

    #[test]
    fn carry_span_restores_the_span_on_a_plain_thread() {
        assign_span_ids();
        let span = tracing::info_span!("scheduling");
        let _entered = span.enter();
        let scheduler = Span::current().id();
        assert!(scheduler.is_some(), "span ids must be assigned");

        let observe = carry_span(|| Span::current().id());
        assert_eq!(
            thread::spawn(observe).join().expect("thread ran"),
            scheduler
        );
    }

    #[test]
    fn carry_span_is_harmless_with_no_span_entered() {
        assign_span_ids();
        let observe = carry_span(|| Span::current().id());
        assert_eq!(thread::spawn(observe).join().expect("thread ran"), None);
    }

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
