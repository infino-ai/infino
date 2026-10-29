// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Joining many fallible futures without running them all at once.

use std::future::Future;

use futures::stream::{FuturesUnordered, StreamExt};

/// Run `futures` at most `limit` at a time and return their outputs in input
/// order.
///
/// - The next future starts as soon as any running one finishes, so a slow
///   future holds one slot, not the ones queued behind it.
/// - `futures` is pulled one item per free slot, so the work of making a
///   future (spawning a task, say) is bounded by `limit` too.
/// - The first error, in time, ends the join: the running futures are
///   dropped and the rest are never made.
/// - A `limit` of zero runs one at a time.
pub(crate) async fn try_join_bounded<I, F, T, E>(futures: I, limit: usize) -> Result<Vec<T>, E>
where
    I: IntoIterator<Item = F>,
    F: Future<Output = Result<T, E>>,
{
    // Not `stream::iter(..).buffer_unordered(..)`: inside an `async_trait`
    // future (the SQL scan's), rustc cannot prove that combinator `Send` over
    // futures that borrow from the caller ("`FnOnce` is not general enough"),
    // even with no closure in the chain.
    let limit = limit.max(1);
    let mut queued = futures.into_iter().enumerate();
    let mut running = FuturesUnordered::new();
    // One slot per future made, filled as it finishes, so outputs land in
    // input order without a sort.
    let mut slots: Vec<Option<T>> = Vec::with_capacity(queued.size_hint().0);
    loop {
        while running.len() < limit
            && let Some((index, future)) = queued.next()
        {
            slots.push(None);
            running.push(async move { (index, future.await) });
        }
        let Some((index, out)) = running.next().await else {
            break;
        };
        slots[index] = Some(out?);
    }
    // The loop ends only once every future made has finished, so every slot
    // is filled.
    Ok(slots.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use std::{
        future::pending,
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use futures::{FutureExt, poll};
    use tokio::time::{sleep, timeout};

    use super::*;

    /// Futures in each join, more than any limit below.
    const UNITS: usize = 20;
    /// A limit well under [`UNITS`], so the bound has work to do.
    const LIMIT: usize = 4;
    /// How long a unit takes on the paused clock.
    const UNIT_TIME: Duration = Duration::from_millis(10);

    /// Counts the units running now, the most ever running at once, and the
    /// units started.
    #[derive(Default)]
    struct Gauge {
        now: AtomicUsize,
        peak: AtomicUsize,
        started: AtomicUsize,
    }

    impl Gauge {
        /// Mark a unit started; the guard marks it finished when dropped,
        /// whether it completed or was dropped mid-way.
        fn enter(&self) -> Running<'_> {
            self.started.fetch_add(1, Ordering::SeqCst);
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            Running(self)
        }
    }

    struct Running<'a>(&'a Gauge);

    impl Drop for Running<'_> {
        fn drop(&mut self) {
            self.0.now.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn at_most_the_limit_run_at_once_and_the_limit_is_reached() {
        let gauge = Gauge::default();
        let out = try_join_bounded(
            (0..UNITS).map(|i| {
                let gauge = &gauge;
                async move {
                    let _running = gauge.enter();
                    sleep(UNIT_TIME).await;
                    Ok::<usize, ()>(i)
                }
            }),
            LIMIT,
        )
        .await
        .expect("join");
        assert_eq!(out.len(), UNITS);
        assert_eq!(
            gauge.peak.load(Ordering::SeqCst),
            LIMIT,
            "never more than the limit at once, and the fan-out reaches it"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn outputs_come_back_in_input_order_whatever_order_they_finish_in() {
        // The first unit finishes last and the last finishes first.
        let out = try_join_bounded(
            (0..UNITS).map(|i| async move {
                sleep(UNIT_TIME * (UNITS - i) as u32).await;
                Ok::<usize, ()>(i)
            }),
            LIMIT,
        )
        .await
        .expect("join");
        assert_eq!(out, (0..UNITS).collect::<Vec<_>>());
    }

    #[tokio::test(start_paused = true)]
    async fn a_slow_unit_holds_one_slot_not_the_units_behind_it() {
        // Unit 0 never finishes; every other unit must still run, the
        // running set never exceeding the limit.
        let gauge = Gauge::default();
        let join = try_join_bounded(
            (0..UNITS).map(|i| {
                let gauge = &gauge;
                async move {
                    let _running = gauge.enter();
                    match i {
                        0 => pending::<()>().await,
                        _ => sleep(UNIT_TIME).await,
                    }
                    Ok::<usize, ()>(i)
                }
            }),
            LIMIT,
        );
        let finished = timeout(UNIT_TIME * UNITS as u32, join).await;
        assert!(finished.is_err(), "the join waits for the stuck unit");
        assert_eq!(
            gauge.started.load(Ordering::SeqCst),
            UNITS,
            "every unit behind the stuck one still ran"
        );
        assert_eq!(
            gauge.peak.load(Ordering::SeqCst),
            LIMIT,
            "never past the limit"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_first_error_ends_the_join_and_drops_the_running_units() {
        // Unit 1 fails at once; the others would never finish, so a join
        // that waited for them would time out here rather than return.
        let gauge = Gauge::default();
        let join = try_join_bounded(
            (0..UNITS).map(|i| {
                let gauge = &gauge;
                async move {
                    let _running = gauge.enter();
                    match i {
                        1 => Err(i),
                        _ => pending().await,
                    }
                }
            }),
            LIMIT,
        );
        let out = timeout(UNIT_TIME, join)
            .await
            .expect("the error ends the join without waiting for the rest");
        assert_eq!(out, Err::<Vec<()>, usize>(1));
        assert_eq!(gauge.now.load(Ordering::SeqCst), 0, "nothing left running");
        assert!(
            gauge.started.load(Ordering::SeqCst) <= LIMIT + 1,
            "units past the error never start"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn dropping_the_join_drops_every_running_unit() {
        let gauge = Gauge::default();
        let mut join = try_join_bounded(
            (0..UNITS).map(|_| {
                let gauge = &gauge;
                async move {
                    let _running = gauge.enter();
                    pending::<Result<(), ()>>().await
                }
            }),
            LIMIT,
        )
        .boxed_local();
        assert!(poll!(&mut join).is_pending());
        assert_eq!(
            gauge.now.load(Ordering::SeqCst),
            LIMIT,
            "the limit's worth run"
        );
        drop(join);
        assert_eq!(
            gauge.now.load(Ordering::SeqCst),
            0,
            "dropping the join drops them"
        );
        assert_eq!(
            gauge.started.load(Ordering::SeqCst),
            LIMIT,
            "no unit past the limit was started"
        );
    }

    /// A caller that spawns a task as it makes each future relies on this:
    /// a future made is work started, so only the limit's worth may be made.
    #[tokio::test(start_paused = true)]
    async fn futures_are_made_only_as_slots_free() {
        let made = AtomicUsize::new(0);
        let mut join = try_join_bounded(
            (0..UNITS).map(|_| {
                made.fetch_add(1, Ordering::SeqCst);
                pending::<Result<(), ()>>()
            }),
            LIMIT,
        )
        .boxed_local();
        assert!(poll!(&mut join).is_pending());
        assert_eq!(
            made.load(Ordering::SeqCst),
            LIMIT,
            "only as many futures made as there are slots"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_zero_limit_runs_one_at_a_time() {
        let gauge = Gauge::default();
        let out = try_join_bounded(
            (0..UNITS).map(|i| {
                let gauge = &gauge;
                async move {
                    let _running = gauge.enter();
                    sleep(UNIT_TIME).await;
                    Ok::<usize, ()>(i)
                }
            }),
            0,
        )
        .await
        .expect("join");
        assert_eq!(out.len(), UNITS);
        assert_eq!(gauge.peak.load(Ordering::SeqCst), 1, "one unit at a time");
    }
}
