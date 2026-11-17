// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Shared build-side dispatch for supertable commit fan-out.
//!
//! Query paths use `query::dispatch` so FTS, vector, and SQL superfile
//! work share one runtime and one fan-out primitive. Builds have the
//! same outer shape at the supertable layer: a commit partitions
//! buffered rows into superfile shards, then builds one superfile per
//! shard. This module owns that fan-out so FTS-only, vector-only, and
//! combined commits all go through the same scheduler — no per-modality
//! branching.
//!
//! The work function decides what a shard means; for the current writer
//! it is one `SuperfileBuilder` producing one immutable superfile. The
//! fan-out is a single rayon level: every shard is dispatched onto
//! `pool` at once via `par_iter`. Each shard's kernel is free to expose
//! its own intra-shard rayon work (e.g. the vector builder's row-parallel
//! rotation/encode, or `finish_index_blobs`' FTS‖vector `join`) on the
//! same pool — rayon's work-stealing schedules both nesting levels onto
//! the pool's threads without oversubscription. Shard count is capped at
//! the pool width (`n_shards ≤ n_threads`); intra-shard work fills cores
//! as shards drain.

use std::sync::{
    Arc, Condvar, Mutex, PoisonError,
    atomic::{AtomicBool, Ordering},
};

use rayon::{ThreadPool, prelude::*};

use crate::runtime_metrics::op_stats::{OpStatsCollector, timed_kernel};

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
    // Borrow `build_one` into the `install` closure (it is `Sync` but not
    // necessarily `Send`); `&F` is itself `Fn + Sync`, so the in-scope variant
    // accepts it and the empty-check + par_iter live in exactly one place.
    pool.install(|| fanout_shards_in_pool_scope(tasks, &build_one))
}

/// Like [`fanout_shards`], but assumes the caller already holds
/// `pool.install` — uses `par_iter` directly to avoid nested install deadlock.
pub(crate) fn fanout_shards_in_pool_scope<T, O, E, F>(
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
    tasks.par_iter().map(&build_one).collect()
}

/// Run tasks on `pool` like [`fanout_shards`], but only as many at a time as
/// `budget` bytes of memory hold, by what `cost` says each task needs.
/// Results come back in input order; the first error stops further tasks
/// from starting and is returned once those running finish.
///
/// For work whose memory, not its thread count, decides how much can run at
/// once: tasks of very different sizes, where a full pool of large ones does
/// not fit. A task costing more than the whole budget runs alone, and a
/// budget of zero runs every task alone.
///
/// The caller's thread waits for room and hands each task to the pool once
/// it fits; a task never waits inside the pool. A pool thread that blocked
/// for memory could be the one a running task's own parallel work is waiting
/// on, and neither would finish.
pub(crate) fn fanout_within_memory<T, O, E, C, F>(
    pool: &ThreadPool,
    tasks: &[T],
    budget: u64,
    cost: C,
    run_one: F,
) -> Result<Vec<O>, E>
where
    T: Sync,
    O: Send,
    E: Send,
    C: Fn(&T) -> u64,
    F: Fn(&T) -> Result<O, E> + Sync,
{
    // Every task holds at least a byte of a budget of at least a byte, so a
    // zero budget or a free task still runs one at a time, never all at once.
    let budget = budget.max(1);
    let gate = MemoryGate::default();
    let failed = AtomicBool::new(false);
    let results: Vec<Mutex<Option<Result<O, E>>>> =
        tasks.iter().map(|_| Mutex::new(None)).collect();
    pool.in_place_scope(|scope| {
        for (task, slot) in tasks.iter().zip(&results) {
            if failed.load(Ordering::Acquire) {
                break;
            }
            let bytes = cost(task).clamp(1, budget);
            gate.acquire(bytes, budget);
            let (gate, failed, run_one) = (&gate, &failed, &run_one);
            scope.spawn(move |_| {
                let result = run_one(task);
                if result.is_err() {
                    failed.store(true, Ordering::Release);
                }
                *slot.lock().unwrap_or_else(PoisonError::into_inner) = Some(result);
                gate.release(bytes);
            });
        }
    });
    let mut out = Vec::with_capacity(tasks.len());
    for slot in results {
        match slot.into_inner().unwrap_or_else(PoisonError::into_inner) {
            Some(Ok(value)) => out.push(value),
            Some(Err(error)) => return Err(error),
            // Not started: an earlier task failed, and its error is returned
            // above before any unstarted slot is reached.
            None => {}
        }
    }
    Ok(out)
}

/// Bytes held by running tasks, and a wake-up when some are given back.
#[derive(Default)]
struct MemoryGate {
    held: Mutex<u64>,
    released: Condvar,
}

impl MemoryGate {
    /// Wait until `bytes` more fit in `budget`, then hold them. With nothing
    /// held anything fits, so a task over the budget runs alone.
    fn acquire(&self, bytes: u64, budget: u64) {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        while *held > 0 && held.saturating_add(bytes) > budget {
            held = self
                .released
                .wait(held)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *held += bytes;
    }

    fn release(&self, bytes: u64) {
        let mut held = self.held.lock().unwrap_or_else(PoisonError::into_inner);
        *held = held.saturating_sub(bytes);
        self.released.notify_all();
    }
}

/// [`fanout_shards`] with each task's on-CPU time folded into `op_stats`.
///
/// The bracket sits *inside* the per-task closure, on the pool thread that
/// runs it — the calling thread blocks in `install` for the whole fan-out,
/// so a bracket around the call would measure nothing. Concurrent tasks
/// fold concurrently; the collector's counters are atomics.
///
/// Use this for shard builds that are a caller op's own work (an append's
/// commit build, a vector commit's drain-pack). Maintenance fan-outs
/// (hidden-index drain merges, compaction) keep the unmetered variant —
/// their cost is deliberately not attributed to any op.
pub(crate) fn fanout_shards_metered<T, O, E, F>(
    pool: &ThreadPool,
    op_stats: &Option<Arc<OpStatsCollector>>,
    tasks: &[T],
    build_one: F,
) -> Result<Vec<O>, E>
where
    T: Sync,
    O: Send,
    E: Send,
    F: Fn(&T) -> Result<O, E> + Sync,
{
    let stats = op_stats.clone();
    fanout_shards(pool, tasks, move |task| {
        timed_kernel(&stats, || build_one(task))
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, AtomicUsize},
        thread::sleep,
        time::Duration,
    };

    use rayon::ThreadPoolBuilder;

    use super::*;

    /// How long each test task holds its memory, so tasks overlap.
    const TASK_HOLD: Duration = Duration::from_millis(20);

    fn pool() -> ThreadPool {
        ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("pool")
    }

    /// Runs `costs` through [`fanout_within_memory`] and returns the most
    /// memory held at once.
    fn peak_held(costs: &[u64], budget: u64) -> u64 {
        let (held, peak) = (AtomicU64::new(0), AtomicU64::new(0));
        let out: Result<Vec<u64>, ()> = fanout_within_memory(
            &pool(),
            costs,
            budget,
            |&c| c,
            |&c| {
                let now = held.fetch_add(c, Ordering::AcqRel) + c;
                peak.fetch_max(now, Ordering::AcqRel);
                sleep(TASK_HOLD);
                held.fetch_sub(c, Ordering::AcqRel);
                Ok(c)
            },
        );
        assert_eq!(
            out.expect("no task fails"),
            costs,
            "results keep input order"
        );
        peak.load(Ordering::Acquire)
    }

    #[test]
    fn tasks_never_hold_more_than_the_budget() {
        let costs = [30, 30, 30, 30, 30, 30, 30, 30];
        let peak = peak_held(&costs, 100);
        assert!(peak <= 100, "held {peak} at once");
        assert!(
            peak >= 60,
            "tasks that fit together still run together ({peak})"
        );
    }

    #[test]
    fn a_task_over_the_budget_runs_alone() {
        // The big task is admitted only with nothing else held, and nothing
        // is admitted beside it, so the most held at once is its own size.
        let peak = peak_held(&[10, 500, 10, 10], 100);
        assert_eq!(peak, 500, "something ran beside the task over the budget");
    }

    #[test]
    fn an_error_stops_tasks_from_starting() {
        let started = AtomicUsize::new(0);
        let costs: Vec<u64> = (0..50).collect();
        let out: Result<Vec<u64>, u64> = fanout_within_memory(
            &pool(),
            &costs,
            1,
            |_| 1,
            |&i| {
                started.fetch_add(1, Ordering::AcqRel);
                if i == 3 { Err(i) } else { Ok(i) }
            },
        );
        assert_eq!(out, Err(3));
        assert!(
            started.load(Ordering::Acquire) < costs.len(),
            "later tasks were not started"
        );
    }

    #[test]
    fn a_zero_budget_runs_one_task_at_a_time() {
        let running = AtomicUsize::new(0);
        let most = AtomicUsize::new(0);
        let tasks: Vec<u64> = (0..8).collect();
        let out: Result<Vec<u64>, ()> = fanout_within_memory(
            &pool(),
            &tasks,
            0,
            |_| 0,
            |&t| {
                let now = running.fetch_add(1, Ordering::AcqRel) + 1;
                most.fetch_max(now, Ordering::AcqRel);
                sleep(TASK_HOLD);
                running.fetch_sub(1, Ordering::AcqRel);
                Ok(t)
            },
        );
        assert_eq!(out, Ok(tasks));
        assert_eq!(most.load(Ordering::Acquire), 1);
    }

    #[test]
    fn no_tasks_return_nothing() {
        let out: Result<Vec<u64>, ()> =
            fanout_within_memory(&pool(), &[] as &[u64], 10, |&c| c, |&c| Ok(c));
        assert_eq!(out, Ok(Vec::new()));
    }
}
