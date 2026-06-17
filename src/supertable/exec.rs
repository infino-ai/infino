// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared execution resources for a connection's tables.
//!
//! The host-level resources a query/build needs — the tokio runtime that
//! drives async I/O and the rayon pools that run the CPU waves — are sized
//! to the machine, not to a table. Building them per table oversubscribes
//! the CPU and wastes threads, so a [`Connection`](crate::Connection) builds
//! one `ExecContext` and shares it (`Arc`) across every table it opens.

use std::sync::{Arc, OnceLock};

use rayon::ThreadPool;
use tokio::runtime::Runtime;

use super::error::BuildError;
use crate::runtime_bridge::build_query_runtime;

/// Tokio runtime + rayon reader/writer pools shared across a connection's
/// tables. Cheap to clone (`Arc`); all clones share the same threads.
pub(crate) struct ExecContext {
    /// Drives async I/O (object-store GETs, range fetches). A `OnceLock`
    /// (always set in `new`) rather than a plain `Arc` so [`Drop`] can take
    /// it back out and shut it down without blocking.
    query_runtime: OnceLock<Arc<Runtime>>,
    /// CPU pool for the read path: page decode, scoring, rerank.
    pub reader_pool: Arc<ThreadPool>,
    /// CPU pool for the build/encode path.
    pub writer_pool: Arc<ThreadPool>,
}

impl ExecContext {
    /// Build a context with `reader_threads` / `writer_threads` rayon
    /// workers and a host-sized multi-thread runtime (see
    /// [`build_query_runtime`]).
    pub(crate) fn new(
        reader_threads: usize,
        writer_threads: usize,
    ) -> Result<Arc<Self>, BuildError> {
        let query_runtime = OnceLock::new();
        let _ = query_runtime.set(build_query_runtime("supertable-query"));
        Ok(Arc::new(Self {
            query_runtime,
            reader_pool: Arc::new(build_pool("supertable-reader", reader_threads)?),
            writer_pool: Arc::new(build_pool("supertable-writer", writer_threads)?),
        }))
    }

    /// The shared multi-thread runtime.
    pub(crate) fn query_runtime(&self) -> &Arc<Runtime> {
        self.query_runtime
            .get()
            .expect("query_runtime is set in ExecContext::new")
    }
}

impl Drop for ExecContext {
    /// Shut the runtime down off-thread so dropping the last reference from
    /// inside the caller's own runtime can't trip tokio's
    /// drop-runtime-in-async-context guard. `try_unwrap` only fires when
    /// this is the last owner.
    fn drop(&mut self) {
        if let Some(rt) = self.query_runtime.take()
            && let Ok(rt) = Arc::try_unwrap(rt)
        {
            rt.shutdown_background();
        }
    }
}

fn build_pool(prefix: &'static str, threads: usize) -> Result<ThreadPool, BuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(move |i| format!("{prefix}-{i}"))
        .build()
        .map_err(|e| BuildError::ThreadPoolCreation(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_with_requested_pool_sizes() {
        let exec = ExecContext::new(3, 2).expect("build");
        assert_eq!(exec.reader_pool.current_num_threads(), 3);
        assert_eq!(exec.writer_pool.current_num_threads(), 2);
        let _ = exec.query_runtime();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drops_cleanly_inside_async_runtime() {
        // Dropping the last reference from inside a runtime must not trip
        // tokio's drop-runtime-in-async-context guard.
        let exec = ExecContext::new(2, 2).expect("build");
        drop(exec); // must not panic
    }
}
