//! Shared build-side dispatch for supertable commit fan-out.
//!
//! Query paths use `query::dispatch` so FTS, vector, and SQL segment
//! work share one runtime and one fan-out primitive. Builds have the
//! same outer shape at the supertable layer: a commit partitions
//! buffered rows into segment shards, then builds one superfile per
//! shard. This module owns that fan-out so FTS-only, vector-only, and
//! combined commits all go through the same scheduler — no per-modality
//! branching.
//!
//! The work function decides what a shard means; for the current writer
//! it is one `SuperfileBuilder` producing one immutable segment. The
//! fan-out is a single rayon level: every shard is dispatched onto
//! `pool` at once via `par_iter`. Each shard's kernel is free to expose
//! its own intra-shard rayon work (e.g. the vector builder's row-parallel
//! rotation/encode, or `finish_index_blobs`' FTS‖vector `join`) on the
//! same pool — rayon's work-stealing schedules both nesting levels onto
//! the pool's threads without oversubscription. Shard count is capped at
//! the pool width (`n_shards ≤ n_threads`); intra-shard work fills cores
//! as shards drain.

use rayon::ThreadPool;
use rayon::prelude::*;

/// Run shard build tasks on `pool`, preserving input order.
///
/// All shards are dispatched at once: rayon owns the CPU and balances
/// the inter-shard fan-out against whatever intra-shard parallelism each
/// `build_one` exposes on the same pool. No concurrency cap — the only
/// bound is the pool's thread count.
pub(crate) fn fanout_shards<T, O, E, F>(
    pool: &ThreadPool,
    tasks: &[T],
    build_one: F,
) -> Result<Vec<O>, E>
where
    T: Sync,
    O: Send,
    E: Send,
    F: Fn(&T) -> Result<O, E> + Sync,
{
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    pool.install(|| tasks.par_iter().map(&build_one).collect())
}
