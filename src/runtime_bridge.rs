//! Sync→async bridge for the sync public API on top of async storage.
//!
//! The supertable's public surface (writer.commit, reader queries,
//! tombstone-cache refresh, lazy byte-source range fetches) is sync,
//! but the storage trait + downstream object_store calls are async.
//! **Every** call site that crosses that boundary routes through
//! [`bridge_sync_to_async`] — there is exactly one bridge and one
//! owned runtime, so the policy below holds uniformly.
//!
//! ## The owned runtime
//!
//! When a sync caller is *not* already inside a tokio runtime, futures
//! are driven on a single process-wide owned runtime: a `multi_thread`
//! runtime pinned to **one worker thread**. The flavor is deliberate:
//!
//! - **Not `current_thread`** — storage work fans out (parallel range
//!   GETs, multi-segment query, `spawn`ed tasks) and may re-enter the
//!   bridge; a current-thread runtime serializes that and can't be
//!   `block_in_place`d. Building one per call also pays a setup cost.
//! - **Not `multi_thread` with N workers** — the work is await/IO-bound
//!   (CPU work lives on rayon), so extra OS threads buy nothing.
//! - **`multi_thread` + 1 worker** — a real runtime (so `spawn` and
//!   `block_in_place` are legal and tasks make progress) at one thread
//!   of overhead. Revisit only if a CPU-bound async path appears.
//!
//! One worker does **not** serialize parallel I/O. A `join_all` over N
//! storage GETs is single-task *concurrency*, not multi-thread
//! parallelism — it polls all N futures on one task and never fans them
//! across workers (no `spawn`), so worker count is irrelevant to it. The
//! actual GET parallelism lives below the worker pool and is unbounded
//! by it: `object_store`'s remote backends are reactor-async (one worker
//! drives many in-flight requests), and the local backend offloads reads
//! to `spawn_blocking` (the separate blocking-thread pool). The single
//! worker only bottlenecks CPU-bound *spawned* async tasks — which is
//! why CPU work stays on rayon.
//!
//! ## Two modes
//!
//! - **Ambient `multi_thread` runtime present** — `block_in_place`
//!   tells the scheduler "I'm about to block this worker; rearrange,"
//!   then `Handle::block_on` drives the future on the current thread.
//!   Sibling workers keep making progress.
//! - **No ambient runtime** — drive on the owned runtime. Sync callers
//!   (CLI tools, rayon workers, Python bindings via PyO3) land here.
//!
//! ## Unsupported: `current_thread` ambient runtime
//!
//! `tokio::task::block_in_place` requires `multi_thread`. If a caller
//! invokes this from inside a `current_thread` tokio runtime,
//! `Handle::try_current()` returns `Ok(...)`, we take the
//! `block_in_place` branch, and tokio panics. tokio exposes no clean
//! way to detect a current-thread handle, so this stays a documented
//! requirement: async callers must run on a `multi_thread` runtime
//! (the default for `#[tokio::main]`, axum, actix, etc.). Lifting it
//! requires offloading the future to the owned runtime via `spawn`,
//! which needs `Send + 'static` futures — deferred to the async-core
//! work, where the futures get reshaped anyway.

use std::sync::OnceLock;

use tokio::runtime::Runtime;

/// The single process-wide runtime that drives infino's async I/O when
/// a sync caller is not already inside a tokio runtime. See the module
/// docs for why it's `multi_thread` pinned to one worker.
fn owned_runtime() -> &'static Runtime {
    static RT: OnceLock<Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("supertable-bridge")
            .build()
            .expect(
                "invariant: tokio Runtime build only fails on \
                 catastrophic OS resource exhaustion",
            )
    })
}

/// Drive `fut` to completion from a sync context. Uses the ambient
/// tokio runtime if present (via `block_in_place + Handle::block_on`),
/// otherwise the process-wide owned runtime.
///
/// Panics if called from inside a `current_thread` tokio runtime
/// (`block_in_place` requires `multi_thread`). See the module docs.
pub(crate) fn bridge_sync_to_async<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => owned_runtime().block_on(fut),
    }
}
