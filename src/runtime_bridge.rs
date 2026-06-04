//! Sync→async bridge for the sync public API on top of async storage.
//!
//! The supertable's public surface (writer.commit, reader queries,
//! tombstone-cache refresh, lazy byte-source range fetches) is sync,
//! but the storage trait + downstream object_store calls are async.
//! **Every** call site that crosses that boundary routes through
//! [`bridge_sync_to_async`] — there is exactly one bridge, so the
//! policy below holds uniformly.
//!
//! ## Two modes
//!
//! - **Ambient `multi_thread` runtime present** — `block_in_place`
//!   tells the scheduler "I'm about to block this worker; rearrange,"
//!   then `Handle::block_on` drives the future on the current thread.
//!   Sibling workers keep making progress.
//! - **No ambient runtime** — drive on a per-thread owned runtime (see
//!   below). Sync callers (CLI tools, rayon reader/writer pool threads,
//!   Python bindings via PyO3) land here.
//!
//! ## The owned runtime: thread-local `current_thread`
//!
//! Each thread that reaches the no-ambient path lazily builds and
//! caches its **own `current_thread`** runtime in a `thread_local!`,
//! reused for every later bridged call on that thread. Two properties
//! matter:
//!
//! - **`current_thread`, not `multi_thread`.** The bridged work is a
//!   short async drive (load N manifest parts, fetch a byte range, run
//!   a commit) on the *calling* thread. A `current_thread` runtime polls
//!   it inline — no worker thread, no cross-thread scheduler hand-off.
//!   A `multi_thread` runtime's `block_on` adds per-poll coordination
//!   with its worker that, on these sub-millisecond reads, measured
//!   **+6–17%** on multi-term FTS search at 10M docs (the cost grows
//!   with the per-query async work, e.g. the `join_all` over kept manifest
//!   parts). Storage fan-out doesn't need worker threads anyway —
//!   concurrent `join_all` GETs run on the reactor (remote) or the
//!   blocking pool (local), neither bounded by the runtime's worker count.
//! - **Thread-local, not shared.** A `current_thread` runtime can't be
//!   driven by concurrent `block_on` calls from multiple threads, and
//!   the reader pool calls the bridge from many rayon threads at once.
//!   One runtime per thread keeps them independent (no shared-worker
//!   contention) and amortizes the build cost across that thread's
//!   calls — so this is not the per-call runtime construction the bridge
//!   exists to avoid.
//!
//! ## Unsupported: re-entrancy and `current_thread` ambient runtimes
//!
//! `tokio::task::block_in_place` requires `multi_thread`. Two
//! consequences, both documented requirements rather than handled cases:
//!
//! - A bridged future must **not synchronously re-enter the bridge** on
//!   the same thread: inside `block_on`, `Handle::try_current()` returns
//!   `Ok(...)`, so the nested call takes the `block_in_place` branch and
//!   tokio panics (the owned runtime is `current_thread`). No current
//!   path nests — the bridged futures are async all the way down.
//! - Calling the sync API from inside a **`current_thread` ambient**
//!   runtime panics for the same reason. Async callers must run on a
//!   `multi_thread` runtime (the default for `#[tokio::main]`, axum,
//!   actix, etc.) or wrap the call in `spawn_blocking`.

use tokio::runtime::Runtime;

thread_local! {
    /// Per-thread `current_thread` runtime for the no-ambient path,
    /// built on first use and reused for the thread's lifetime. See the
    /// module docs for why it's thread-local `current_thread`.
    static OWNED_RT: Runtime = new_owned_runtime();
}

fn new_owned_runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect(
            "invariant: tokio current_thread Runtime build only fails on \
             catastrophic OS resource exhaustion",
        )
}

/// Drive `fut` to completion from a sync context. Uses the ambient
/// tokio runtime if present (via `block_in_place + Handle::block_on`),
/// otherwise this thread's owned `current_thread` runtime.
///
/// Panics if called from inside a `current_thread` tokio runtime, or if
/// a bridged future synchronously re-enters the bridge on the same
/// thread (`block_in_place` requires `multi_thread`). See the module docs.
pub(crate) fn bridge_sync_to_async<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => OWNED_RT.with(|rt| rt.block_on(fut)),
    }
}
