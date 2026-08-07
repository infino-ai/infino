// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Concurrent cold opens of the same table must not deadlock.
//!
//! `Connection::open_table_handle` takes the per-name single-flight build
//! gate (a `std::sync::Mutex`) and then blocks inside it, driving the
//! catalog read and the supertable open through the sync→async bridge.
//! That bridge used to run the future on the *ambient* runtime, so anything
//! the future spawned landed on the caller's runtime too.
//!
//! With `WORKERS` runtime workers and `WORKERS + 1` concurrent cold opens
//! of one name: the winner releases its worker via `block_in_place` (tokio
//! spawns a replacement) and the losers hard-block on the gate. A plain
//! mutex block is invisible to the scheduler, so no further replacements
//! appear — every worker is consumed and the winner's future can never be
//! polled to completion. Nothing ever releases the gate.
//!
//! The fix: drive that bridged work on a *dedicated* I/O runtime instead of
//! the caller's. The losers still hard-block the caller's workers, but the
//! winner's future — and anything it spawns — is polled by the dedicated
//! runtime's own workers, which nothing is contending for.
//!
//! Two tests here pin that, plus one ignored negative control:
//!
//! - `mechanism_*` needs no infrastructure and pins the code shape itself:
//!   lock → bridge onto a dedicated runtime → await a spawned task, with
//!   N+1 contenders on N caller workers.
//! - `cold_open_*` drives the real `Connection` path over the local RustFS
//!   daemon. An object store is required because that is what supplies the
//!   spawn dependency (hyper spawns a driver task per pooled connection).
//!   A `file://` backend does NOT reproduce: `tokio::fs` routes through
//!   `spawn_blocking`, a separate pool that never starves. Point the test
//!   at any other S3-compatible endpoint with `INFINO_REPRO_S3_ENDPOINT`
//!   (plus `_BUCKET` / `_KEY` / `_SECRET`).
//! - `regression_*` is the pre-fix shape, kept as executable documentation
//!   of the bug. It wedges by construction and aborts the process, so it is
//!   `#[ignore]`d; run it on its own when you want to see the failure:
//!
//! ```sh
//! cargo test --features test-helpers --test catalog_cold_open_deadlock \
//!     -- --ignored --exact --nocapture \
//!     regression_mutex_across_ambient_bridge_awaiting_spawned_task_deadlocks
//! ```
//!
//! Every watchdog lives on a plain OS thread on purpose — once every worker
//! is wedged, a `tokio::time::timeout` on that runtime cannot fire either.

#![deny(clippy::unwrap_used)]

use std::{
    env,
    future::Future,
    sync::{Arc, Mutex, OnceLock, mpsc},
    thread,
    time::Duration,
};

use infino::{ConnectOptions, IndexSpec, connect_with, test_helpers::schema_id_title};
use infino_bench_utils::rustfs_server;
use tokio::{
    runtime::{Builder, Handle, Runtime},
    sync::Barrier,
    task::block_in_place,
};

/// Matches the 2-vCPU deployment where this was first seen. The bug needs a
/// worker count small enough that the contending opens can consume all of it.
const WORKERS: usize = 2;
/// One task to win the gate and enter `block_in_place`, then one per remaining
/// live worker to block on the gate and starve the winner's future.
const CONTENDERS: usize = WORKERS + 1;
/// Generous enough that a healthy cold open (catalog GET + manifest load) is
/// never mistaken for a hang.
const WATCHDOG: Duration = Duration::from_secs(25);
const TABLE: &str = "docs";

/// Abort the process if `WATCHDOG` elapses before the returned sender fires.
/// Runs on an OS thread — see the module docs.
fn arm_watchdog(what: &'static str) -> mpsc::Sender<()> {
    let (tx, rx) = mpsc::channel::<()>();
    thread::spawn(move || {
        if rx.recv_timeout(WATCHDOG).is_err() {
            eprintln!("DEADLOCK: {what} did not complete within {WATCHDOG:?}");
            std::process::abort();
        }
    });
    tx
}

fn runtime(name: &str) -> Runtime {
    Builder::new_multi_thread()
        .worker_threads(WORKERS)
        .enable_all()
        .thread_name(name)
        .build()
        .expect("build multi_thread runtime")
}

/// Run `CONTENDERS` tasks on a fresh `WORKERS`-wide runtime, each of which
/// lines up on a barrier, takes `gate`, and then bridges into `bridge` while
/// still holding it. `bridge` receives the ambient handle and stands in for
/// the engine's sync→async bridge; the future it drives awaits a spawned
/// task, which is the dependency that starves.
fn contend<F>(caller_name: &str, bridge: F)
where
    F: Fn(Handle) + Send + Sync + 'static,
{
    let bridge = Arc::new(bridge);
    runtime(caller_name).block_on(async move {
        let gate = Arc::new(Mutex::new(()));
        let barrier = Arc::new(Barrier::new(CONTENDERS));

        let tasks: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let gate = Arc::clone(&gate);
                let barrier = Arc::clone(&barrier);
                let bridge = Arc::clone(&bridge);
                tokio::spawn(async move {
                    // Line every task up so they all contend on one gate.
                    barrier.wait().await;
                    let _held = gate.lock().expect("gate");
                    bridge(Handle::current());
                })
            })
            .collect();

        for task in tasks {
            task.await.expect("contender panicked");
        }
    });
}

/// Stands in for the engine's process-wide `shared_io_runtime()`: separate
/// workers, uncontended by the gate, and — like the engine's — held by a
/// static so it is never dropped. A `Runtime` dropped inside an async context
/// panics, and every drop site here is inside the caller runtime's `block_on`.
fn io_runtime() -> &'static Runtime {
    static IO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
    IO_RUNTIME.get_or_init(|| runtime("infino-io"))
}

/// Drive `fut` on `runtime` from a sync context, escaping the ambient worker
/// via `block_in_place` so the nested `block_on` is legal. This is the shape
/// of the engine's `bridge_on_runtime` — that function is `pub(crate)`, so an
/// integration test has to restate it.
fn bridge_on_runtime<T>(fut: impl Future<Output = T>, runtime: &Runtime) -> T {
    block_in_place(|| runtime.handle().block_on(fut))
}

/// The fixed code shape on its own, with no engine involved: a
/// `std::sync::Mutex` held across a bridge that drives its future on a
/// dedicated I/O runtime, and that future awaits a spawned task. This mirrors
/// `open_table_handle` bridging through `bridge_on_runtime(…,
/// &shared_io_runtime())` into a path that spawns (the disk reader cache, or
/// hyper's per-connection driver).
///
/// The losers still hard-block every worker of the *caller's* runtime; the
/// point is that the winner's future no longer needs one.
#[test]
fn mechanism_mutex_across_bridge_onto_dedicated_runtime_completes() {
    let done = arm_watchdog("mechanism: lock -> bridge onto dedicated runtime");

    contend("caller", |_ambient| {
        bridge_on_runtime(
            async {
                tokio::spawn(async { tokio::task::yield_now().await })
                    .await
                    .expect("spawned helper")
            },
            io_runtime(),
        );
    });

    let _ = done.send(());
}

/// The pre-fix shape, kept as executable documentation of the bug: the same
/// contention, but bridged onto the *ambient* runtime. The winner's spawned
/// task needs a caller worker; the losers have hard-blocked all of them on the
/// gate, and a plain mutex block is invisible to the scheduler, so tokio never
/// grows a replacement. Nothing ever releases the gate.
///
/// This wedges by construction — no engine change can make it pass, and the
/// watchdog aborts the process — so it stays out of the default run. See the
/// module docs for how to invoke it.
#[test]
#[ignore = "documents the bug: wedges by construction and aborts the process"]
fn regression_mutex_across_ambient_bridge_awaiting_spawned_task_deadlocks() {
    let done = arm_watchdog("regression: lock -> ambient bridge -> await spawned task");

    contend("caller", |ambient| {
        block_in_place(|| {
            ambient.block_on(async {
                tokio::spawn(async { tokio::task::yield_now().await })
                    .await
                    .expect("spawned helper")
            })
        });
    });

    let _ = done.send(());
}

/// Endpoint + credentials for the run: an explicit `INFINO_REPRO_S3_*` target
/// if set, otherwise a bucket on the shared local RustFS daemon. `None` when
/// RustFS is unavailable and no explicit target was given.
fn object_store_target() -> Option<(String, ConnectOptions)> {
    let mut opts = ConnectOptions::new().with_storage_option("region", "us-east-1");

    if let Ok(endpoint) = env::var("INFINO_REPRO_S3_ENDPOINT") {
        let bucket =
            env::var("INFINO_REPRO_S3_BUCKET").unwrap_or_else(|_| "infino-repro".to_string());
        opts = opts
            .with_storage_option("endpoint", endpoint)
            .with_storage_option(
                "access_key_id",
                env::var("INFINO_REPRO_S3_KEY").unwrap_or_else(|_| "minioadmin".to_string()),
            )
            .with_storage_option(
                "secret_access_key",
                env::var("INFINO_REPRO_S3_SECRET").unwrap_or_else(|_| "minioadmin".to_string()),
            )
            .with_storage_option("allow_http", "true")
            .with_storage_option("allow_invalid_certificates", "true");
        return Some((format!("s3://{bucket}"), opts));
    }

    if !rustfs_server::begin_rustfs_test("cold_open_same_table_concurrently") {
        return None;
    }
    let session = rustfs_server::session().ok()?;
    let lease = session.open_test_bucket("repro").ok()?;
    opts = opts
        .with_storage_option("endpoint", session.endpoint())
        .with_storage_option("access_key_id", session.access_key())
        .with_storage_option("secret_access_key", session.secret_key())
        // The daemon's CA is per-session and not reachable through the
        // `storage_options` surface; this is a loopback test daemon.
        .with_storage_option("allow_invalid_certificates", "true");
    Some((format!("s3://{}", lease.bucket), opts))
}

/// The real `Connection` path over an object store.
/// See the module docs for why `file://` cannot reproduce this.
#[test]
fn cold_open_same_table_concurrently_over_object_store() {
    let Some((uri, opts)) = object_store_target() else {
        eprintln!("skipping: no object store available (see module docs)");
        return;
    };

    // Seed the table and prove connectivity uncontended, so a later hang is
    // unambiguously the deadlock and not a broken endpoint.
    {
        let conn = connect_with(&uri, opts.clone()).expect("connect (seed)");
        conn.create_table(TABLE, schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");
        let bytes = conn.table_storage_bytes(TABLE).expect("uncontended open");
        eprintln!("connectivity probe ok: table_storage_bytes = {bytes}");
    }

    let done = arm_watchdog("cold open: N+1 concurrent table_storage_bytes");

    // A fresh connection, so `handles` is empty and every open below takes
    // the cold path through the single-flight gate.
    runtime("caller").block_on(async {
        let conn = Arc::new(connect_with(&uri, opts).expect("connect (cold)"));
        let barrier = Arc::new(Barrier::new(CONTENDERS));

        let tasks: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let conn = Arc::clone(&conn);
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    // Line every task up so they contend on one cold name.
                    barrier.wait().await;
                    conn.table_storage_bytes(TABLE)
                })
            })
            .collect();

        for task in tasks {
            task.await
                .expect("open task panicked")
                .expect("table_storage_bytes");
        }
    });

    let _ = done.send(());
}

#[test]
fn create_table_same_name_concurrently_over_object_store() {
    let Some((uri, opts)) = object_store_target() else {
        eprintln!("skipping: no object store available (see module docs)");
        return;
    };

    let done = arm_watchdog("create_table: N+1 concurrent same-name creates");

    runtime("caller").block_on(async {
        let conn = Arc::new(connect_with(&uri, opts).expect("connect"));
        let barrier = Arc::new(Barrier::new(CONTENDERS));

        let tasks: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let conn = Arc::clone(&conn);
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    conn.create_table(TABLE, schema_id_title(), IndexSpec::new().fts("title"))
                })
            })
            .collect();

        for task in tasks {
            let _ = task.await.expect("create_table task panicked");
        }
    });

    let _ = done.send(());
}

#[test]
fn drop_table_same_name_concurrently_over_object_store() {
    let Some((uri, opts)) = object_store_target() else {
        eprintln!("skipping: no object store available (see module docs)");
        return;
    };

    {
        let conn = connect_with(&uri, opts.clone()).expect("connect (seed)");
        conn.create_table(TABLE, schema_id_title(), IndexSpec::new().fts("title"))
            .expect("create_table");
    }

    let done = arm_watchdog("drop_table: N+1 concurrent same-name drops");

    runtime("caller").block_on(async {
        let conn = Arc::new(connect_with(&uri, opts).expect("connect"));
        let barrier = Arc::new(Barrier::new(CONTENDERS));

        let tasks: Vec<_> = (0..CONTENDERS)
            .map(|_| {
                let conn = Arc::clone(&conn);
                let barrier = Arc::clone(&barrier);
                tokio::spawn(async move {
                    barrier.wait().await;
                    conn.drop_table(TABLE, true)
                })
            })
            .collect();

        for task in tasks {
            let _ = task.await.expect("drop_table task panicked");
        }
    });

    let _ = done.send(());
}
