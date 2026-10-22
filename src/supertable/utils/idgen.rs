// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! 128-bit Snowflake-style id generator for the supertable's
//! auto-injected `_id` column.
//!
//! Layout (most-significant-bit first):
//!
//! ```text
//! 127                              64 63              24 23      0
//! ┌────────────────────────────────┬─────────────────┬─────────┐
//! │     64-bit ms timestamp        │   40 worker     │ 24 ctr  │
//! └────────────────────────────────┴─────────────────┴─────────┘
//! ```
//!
//! `next_id()` returns `i128`, matching the Arrow / Parquet
//! `Decimal128(38, 0)` storage type. The high bit (i128's
//! sign) stays 0 for any plausible lifetime — a 64-bit
//! Unix-ms timestamp exhausts in year ~292M — so signed
//! `i128` comparison matches time order, giving cheap
//! skip-pruning by id range at the manifest layer.
//!
//! Ids are handed out in claims: one clock read takes as many
//! sequence numbers as the caller needs, up to what is left in
//! the current millisecond. A single id is a claim of one; a
//! batch of `n` rows ([`IdGenerator::reserve_range`]) is one
//! claim per millisecond of sequence it needs, so the caller
//! holds the generator for a handful of clock reads rather than
//! for `n` of them and expands the spans after letting go.
//!
//! The generator is single-threaded by construction (its state
//! is a `Cell`, so it's `!Sync`). One generator per supertable
//! handle is the intended usage; the handle keeps it behind a
//! mutex, which concurrent appends on that handle take.

use std::{cell::Cell, fmt, hint};

use ferroid::time::{MonotonicClock, TimeSource, UNIX_EPOCH};
use rand::random;

const WORKER_BITS: u32 = 40;
const WORKER_MASK: u64 = (1u64 << WORKER_BITS) - 1;
const SEQUENCE_BITS: u32 = 24;
const SEQUENCE_MASK: u128 = (1u128 << SEQUENCE_BITS) - 1;
const TIMESTAMP_SHIFT: u32 = WORKER_BITS + SEQUENCE_BITS;

/// Single-threaded id generator. One per supertable handle.
///
/// Construction is cheap: one `rand::random::<u64>()` call for
/// [`Self::new`], zero for [`Self::with_worker_id`].
/// [`Self::next_id`] is `&self` (interior-mutable) and runs
/// at ~3 ns/id single-threaded; [`Self::reserve_range`] costs
/// one clock read per millisecond of sequence it spans.
pub struct IdGenerator {
    worker_id: u64,
    /// `worker_id` in its place in the id, below the timestamp.
    worker_bits: u128,
    /// Milliseconds since the Unix epoch, published by a shared
    /// ticker thread, so reading it is a load rather than a syscall
    /// and it never steps backwards.
    clock: MonotonicClock<1>,
    /// The last id handed out, or 0 before the first. Its timestamp
    /// and sequence fields are where the next claim continues from.
    last: Cell<u128>,
}

impl IdGenerator {
    /// Construct with a 40-bit random worker_id. Each process
    /// should call this exactly once per supertable handle.
    ///
    /// At 40 random bits, birthday-collision probability
    /// stays below 1% for fleets up to ~148k concurrent
    /// writer processes per supertable — well past any
    /// realistic deployment without coordination.
    pub fn new() -> Self {
        let worker_id = random::<u64>() & WORKER_MASK;
        Self::with_worker_id(worker_id)
    }

    /// Construct with an explicit worker_id (truncated to 40
    /// bits). Useful for tests that need a stable id sequence
    /// and for callers driving multiple generators with
    /// known-disjoint worker_ids in a single process.
    pub fn with_worker_id(worker_id: u64) -> Self {
        let worker40 = worker_id & WORKER_MASK;
        Self {
            worker_id: worker40,
            worker_bits: u128::from(worker40) << SEQUENCE_BITS,
            clock: MonotonicClock::<1>::with_epoch(UNIX_EPOCH),
            last: Cell::new(0),
        }
    }

    /// The 40-bit worker_id stamped into every produced id.
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Mint one id.
    ///
    /// Returns `i128` directly — the natural type for Arrow
    /// `Decimal128Array::value()`. The high bit is always 0
    /// for current-era timestamps, so the `as i128` cast is
    /// lossless and the resulting value's signed sort order
    /// matches time order.
    ///
    /// **Single-threaded contract.** Calling this from
    /// multiple threads concurrently is a logic error — the
    /// generator is `!Sync` and the borrow checker will
    /// refuse. Appends on one handle run concurrently, so
    /// the supertable keeps its generator behind a mutex;
    /// that mutex is required.
    #[inline]
    pub fn next_id(&self) -> i128 {
        self.claim(1).0
    }

    /// Reserve `n` ids in advance and return them as an ordered
    /// list of contiguous spans. Each span is `(first, last)`
    /// inclusive; the flatten of all spans yields exactly `n`
    /// distinct, monotonically-increasing ids.
    ///
    /// **Why spans and not a single `(first, last)`:** the
    /// sequence field is 24 bits wide, so one millisecond holds
    /// at most 2^24 ids. A reservation that needs more than the
    /// current millisecond has left continues in the next one,
    /// and two ids straddling a ms boundary are not numerically
    /// contiguous (the timestamp field changes). The spans shape
    /// makes that boundary visible: under typical use the result
    /// is a `Vec` of length 1; it grows by one per ms boundary
    /// the reservation crosses.
    ///
    /// Each span is one claim on the generator, so the cost is
    /// one clock read per span, not one per id. Callers hold the
    /// generator's lock only for this call and expand the spans
    /// after releasing it.
    ///
    /// **Caller contract (update):** persist the full `Vec` into
    /// a durable artifact (the WAL state doc) BEFORE doing any
    /// work that depends on the ids. Recovery reads the spans
    /// verbatim and never re-runs `reserve_range` — a recovering
    /// process has a different `worker_id` and would mint
    /// different ids, which would break the determinism a
    /// replay-safe append phase needs.
    ///
    /// `n == 0` returns an empty `Vec` without any minting.
    pub fn reserve_range(&self, n: usize) -> Vec<(i128, i128)> {
        let mut spans: Vec<(i128, i128)> = Vec::with_capacity(1);
        let mut left = n;
        while left > 0 {
            let (first, count) = self.claim(left);
            spans.push((first, first + (count as i128 - 1)));
            left -= count;
        }
        spans
    }

    /// Take up to `max` (> 0) consecutive ids from one millisecond:
    /// the rest of the current millisecond's sequence if the clock has
    /// not moved, else a fresh millisecond starting at sequence 0.
    /// Returns the first id and how many were taken (at least 1). Waits
    /// for the next tick when the current millisecond is used up.
    #[inline]
    fn claim(&self, max: usize) -> (i128, usize) {
        debug_assert!(max > 0, "a claim takes at least one id");
        loop {
            let now: u64 = self.clock.current_millis();
            let last = self.last.get();
            let last_ms = (last >> TIMESTAMP_SHIFT) as u64;
            let (first, room) = if now > last_ms {
                (
                    (u128::from(now) << TIMESTAMP_SHIFT) | self.worker_bits,
                    SEQUENCE_MASK + 1,
                )
            } else if now == last_ms && last & SEQUENCE_MASK < SEQUENCE_MASK {
                (last + 1, SEQUENCE_MASK - (last & SEQUENCE_MASK))
            } else {
                // This millisecond's sequence is used up (or, never
                // in practice, the clock reads behind the last id).
                // The ticker publishes the next millisecond shortly.
                hint::spin_loop();
                continue;
            };
            let count = room.min(max as u128);
            self.last.set(first + (count - 1));
            // High bit is 0 for current-era Unix-ms timestamps
            // (today ≈ 1.7×10¹² ms = 41 bits; the high bit is bit
            // 127, ~86 bits past the timestamp field). The `as
            // i128` cast is lossless under that invariant.
            return (first as i128, count as usize);
        }
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for IdGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IdGenerator")
            .field("worker_id", &format_args!("0x{:010x}", self.worker_id))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, thread::sleep, time};

    /// The generator's identity surface: `Default` builds a working
    /// generator, `worker_id()` reports the stamped 40-bit id, and
    /// `Debug` prints it as fixed-width hex without leaking generator
    /// internals.
    #[test]
    fn generator_identity_surface() {
        let generator = IdGenerator::default();
        let id = generator.next_id();
        assert!(id > 0, "default generator mints time-ordered ids");
        assert!(generator.worker_id() < (1 << 40), "worker id is 40-bit");
        let debug = format!("{generator:?}");
        assert!(
            debug.contains("IdGenerator") && debug.contains("0x"),
            "Debug is the hex identity form: {debug}"
        );
    }

    use super::*;

    /// Extract the `worker_id` field bits from a produced id.
    fn worker_id_of(id: i128) -> u64 {
        (((id as u128) >> 24) & (WORKER_MASK as u128)) as u64
    }

    /// Extract the `timestamp` field bits from a produced id.
    fn timestamp_of(id: i128) -> u64 {
        ((id as u128) >> 64) as u64
    }

    /// Extract the `sequence` field bits from a produced id.
    fn sequence_of(id: i128) -> u32 {
        ((id as u128) & ((1u128 << 24) - 1)) as u32
    }

    #[test]
    fn strict_monotonicity_within_one_generator() {
        let g = IdGenerator::with_worker_id(0x0012_3456_789A);
        let mut last = i128::MIN;
        for _ in 0..100_000 {
            let id = g.next_id();
            assert!(
                id > last,
                "expected strict monotonic; got {id} after {last}"
            );
            last = id;
        }
    }

    #[test]
    fn high_bit_stays_zero_for_current_era_timestamps() {
        // `Decimal128(38, 0)` storage relies on the i128
        // value being non-negative for our intended sort
        // semantics. A current Unix-ms timestamp is well
        // under i64::MAX, so the i128 high bit is 0.
        let g = IdGenerator::new();
        let id = g.next_id();
        assert!(id >= 0, "id={id} unexpectedly negative");
    }

    #[test]
    fn timestamp_field_matches_now() {
        // The minted id's 64-bit timestamp field should be
        // within a few seconds of wall-clock now.
        let g = IdGenerator::with_worker_id(0xABCD);
        let now_ms = time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .expect("clock pre-1970?")
            .as_millis() as u64;
        let id = g.next_id();
        let ts = timestamp_of(id);
        let drift_ms = ts.abs_diff(now_ms);
        assert!(
            drift_ms < 5_000,
            "id timestamp {ts} drifted {drift_ms}ms from now_ms {now_ms}"
        );
    }

    #[test]
    fn worker_id_truncates_to_40_bits_and_is_recoverable() {
        let g = IdGenerator::with_worker_id(u64::MAX);
        assert_eq!(g.worker_id(), WORKER_MASK);
        let id = g.next_id();
        assert_eq!(worker_id_of(id), WORKER_MASK);
    }

    #[test]
    fn worker_id_zero_is_valid() {
        let g = IdGenerator::with_worker_id(0);
        let id = g.next_id();
        assert_eq!(worker_id_of(id), 0);
    }

    #[test]
    fn different_worker_ids_appear_in_their_id_field() {
        let g1 = IdGenerator::with_worker_id(0x01);
        let g2 = IdGenerator::with_worker_id(0xFE_DCBA);
        assert_eq!(worker_id_of(g1.next_id()), 0x01);
        assert_eq!(worker_id_of(g2.next_id()), 0xFE_DCBA);
    }

    #[test]
    fn sequence_resets_per_ms_advances_within_ms() {
        // Two adjacent ids in the same ms have consecutive
        // sequence numbers; two ids across an ms boundary
        // both start the new ms with sequence 0.
        let g = IdGenerator::with_worker_id(0);
        let mut prev_ts = 0u64;
        let mut prev_seq = u32::MAX;
        // Mint a small burst, check the invariant on every
        // adjacent pair.
        for _ in 0..10_000 {
            let id = g.next_id();
            let ts = timestamp_of(id);
            let seq = sequence_of(id);
            if ts == prev_ts {
                assert!(
                    seq == prev_seq.wrapping_add(1),
                    "same-ms seq must increment: ts={ts} prev_seq={prev_seq} seq={seq}"
                );
            } else {
                // New ms — seq should reset to 0.
                assert!(
                    ts > prev_ts || prev_ts == 0,
                    "ts must be non-decreasing: prev_ts={prev_ts} ts={ts}"
                );
                assert_eq!(
                    seq, 0,
                    "new-ms seq must be 0; got {seq} after ts={prev_ts} → {ts}"
                );
            }
            prev_ts = ts;
            prev_seq = seq;
        }
    }

    #[test]
    fn new_picks_random_worker_id_per_instance() {
        // Two `IdGenerator::new()` calls in the same process
        // should produce distinct worker_ids with overwhelming
        // probability. Birthday-collision over 2 picks from a
        // 2^40 space is ~2^-40 ≈ 10⁻¹². If this ever fires,
        // either the RNG is broken or you ran this test on
        // ~10¹² CI builds.
        let g1 = IdGenerator::new();
        let g2 = IdGenerator::new();
        assert_ne!(
            g1.worker_id(),
            g2.worker_id(),
            "two IdGenerator::new() collided on worker_id — \
             this is a ~10⁻¹² event"
        );
    }

    #[test]
    fn debug_format_includes_worker_id_in_hex() {
        let g = IdGenerator::with_worker_id(0xDEAD_BEEF);
        let s = format!("{g:?}");
        assert!(s.contains("0x00deadbeef"), "got: {s}");
    }

    // ---- reserve_range ------------------------------------------------

    /// Flatten a span list to the implied sequence of ids. Used
    /// by the tests below to assert the per-span shape produces
    /// the right total count + ordering.
    fn flatten_spans(spans: &[(i128, i128)]) -> Vec<i128> {
        let mut out = Vec::new();
        for (first, last) in spans {
            for v in *first..=*last {
                out.push(v);
            }
        }
        out
    }

    #[test]
    fn reserve_range_zero_returns_empty() {
        let g = IdGenerator::with_worker_id(0x1234);
        let spans = g.reserve_range(0);
        assert!(spans.is_empty());
    }

    #[test]
    fn reserve_range_one_returns_singleton_span() {
        let g = IdGenerator::with_worker_id(0x1234);
        let spans = g.reserve_range(1);
        assert_eq!(spans.len(), 1);
        let (first, last) = spans[0];
        assert_eq!(first, last);
    }

    #[test]
    fn reserve_range_small_n_flattens_to_n_monotonic_ids() {
        // Mints fit comfortably in a single ms (~2 ns/mint),
        // so we expect exactly one span here. The structural
        // assertion is on the flatten — multi-span behavior
        // gets exercised by the cross-ms test below.
        let g = IdGenerator::with_worker_id(0x1234);
        let spans = g.reserve_range(100);
        let flat = flatten_spans(&spans);
        assert_eq!(flat.len(), 100);
        for w in flat.windows(2) {
            assert!(w[1] > w[0], "reserve_range ids must be strictly monotonic");
        }
    }

    #[test]
    fn reserve_range_large_n_still_flattens_correctly() {
        // 10K ids is small enough to almost certainly fit in
        // one ms on a fast machine but large enough to catch a
        // regression where the loop accidentally drops ids at
        // span boundaries. We assert flat.len() == n and no
        // duplicates regardless of whether we span 1 or N ms.
        let g = IdGenerator::with_worker_id(0xCAFE);
        let n = 10_000;
        let spans = g.reserve_range(n);
        let flat = flatten_spans(&spans);
        assert_eq!(flat.len(), n);
        let mut seen = HashSet::with_capacity(n);
        for id in &flat {
            assert!(
                seen.insert(*id),
                "duplicate id in reserve_range output: {id}"
            );
        }
        for w in flat.windows(2) {
            assert!(w[1] > w[0]);
        }
        // Per-span shape: every span is non-empty and
        // numerically contiguous within itself.
        for (first, last) in &spans {
            assert!(last >= first, "span ({first}, {last}) is inverted");
        }
    }

    #[test]
    fn reserve_range_back_to_back_calls_produce_disjoint_ids() {
        // Two calls on the same generator must produce
        // non-overlapping id sets — the in-process equivalent of
        // the multi-mutation pattern. (Cross-generator
        // disjointness is covered by
        // `cross_worker_ids_remain_distinct_within_same_ms`
        // below.)
        let g = IdGenerator::with_worker_id(0x10);
        let a = flatten_spans(&g.reserve_range(50));
        let b = flatten_spans(&g.reserve_range(50));
        let a_set: HashSet<i128> = a.iter().copied().collect();
        for id in &b {
            assert!(!a_set.contains(id), "id {id} in both reservations");
        }
    }

    #[test]
    fn reserve_range_across_ms_boundary_produces_multi_span() {
        // `reserve_range` starts a new span whenever two consecutive ids
        // are not numerically adjacent, which happens exactly when the ms
        // timestamp field advances mid-reservation. The straight-line API
        // gives no sleep-injection hook, so we exercise that same
        // predicate directly: mint ids until the clock ticks to a new ms
        // and confirm the id landing in the new ms is NOT `prev + 1`.
        //
        // We poll until the clock advances rather than sleeping a fixed
        // amount: the monotonic clock's ms counter is published by a
        // background ticker thread that can lag a short fixed sleep under
        // load (e.g. coverage instrumentation on a busy runner). A fixed
        // sleep would then read the same stale ms for both ids and see
        // them as adjacent — flaky. Sleeping a little between attempts
        // also keeps us from starving the very ticker we're waiting on.
        const TICK_TIMEOUT: time::Duration = time::Duration::from_secs(5);

        let g = IdGenerator::with_worker_id(0x20);
        let mut prev = g.next_id();
        let deadline = time::Instant::now() + TICK_TIMEOUT;
        loop {
            let id = g.next_id();
            assert!(id > prev, "ids must be strictly monotonic — {prev} → {id}");
            if id != prev + 1 {
                // The ms field advanced: the discontinuity that
                // `reserve_range` turns into a new span. Property
                // confirmed.
                break;
            }
            // Same ms — only the sequence advanced. Yield so the ticker
            // can publish the next ms, then retry.
            prev = id;
            assert!(
                time::Instant::now() < deadline,
                "clock did not advance to a new ms within {TICK_TIMEOUT:?}"
            );
            sleep(time::Duration::from_millis(1));
        }
    }

    #[test]
    fn reserve_range_past_one_ms_of_sequence_splits_at_the_ms_boundary() {
        // One millisecond holds 2^24 ids, so a reservation of more
        // than that must take the rest of the current millisecond and
        // continue in a later one. Check the boundary exactly: the
        // first span runs to the last sequence number, every later
        // span starts a later millisecond at sequence 0, no span
        // straddles a millisecond, and the spans add up to `n`.
        let worker = 0x00AB_CDEF_0123;
        let g = IdGenerator::with_worker_id(worker);
        let before = g.next_id();
        let n = (1usize << SEQUENCE_BITS) + 1_000;
        let spans = g.reserve_range(n);

        assert!(spans.len() >= 2, "{n} ids cannot fit one ms: {spans:?}");
        let total: usize = spans.iter().map(|&(f, l)| (l - f + 1) as usize).sum();
        assert_eq!(total, n);
        assert!(
            spans[0].0 > before,
            "a reservation continues after the last id"
        );
        assert_eq!(
            sequence_of(spans[0].1),
            SEQUENCE_MASK as u32,
            "the first span takes the rest of its millisecond"
        );
        for &(first, last) in &spans {
            assert!(last >= first, "span ({first}, {last}) is inverted");
            assert_eq!(timestamp_of(first), timestamp_of(last), "a span is one ms");
            assert_eq!(worker_id_of(first), worker);
            assert_eq!(worker_id_of(last), worker);
        }
        for pair in spans.windows(2) {
            let ((_, prev_last), (next_first, _)) = (pair[0], pair[1]);
            assert!(timestamp_of(next_first) > timestamp_of(prev_last));
            assert_eq!(sequence_of(next_first), 0, "a new ms starts at sequence 0");
        }
    }

    #[test]
    fn next_id_after_reserve_range_continues_past_it() {
        // A reservation advances the generator to its last id, so the
        // next single id cannot land inside any reserved span.
        let g = IdGenerator::with_worker_id(0x77);
        let spans = g.reserve_range(1_000);
        let reserved_last = spans.last().expect("non-empty").1;
        assert!(g.next_id() > reserved_last);
    }

    // ---- cross-worker isolation ----------------------------------------

    #[test]
    fn cross_worker_ids_remain_distinct_within_same_ms() {
        // Two generators with different worker_ids minting in
        // the same ms produce ids that differ at minimum in
        // the worker_id field, even if their ts and seq match.
        // This is the core "no coordination needed across
        // writer processes" property.
        let g1 = IdGenerator::with_worker_id(0xAAAA);
        let g2 = IdGenerator::with_worker_id(0xBBBB);
        let id1 = g1.next_id();
        let id2 = g2.next_id();
        assert_ne!(id1, id2);
        assert_eq!(worker_id_of(id1), 0xAAAA);
        assert_eq!(worker_id_of(id2), 0xBBBB);
    }

    /// `Default` builds a usable generator equivalent to `new()` — it
    /// yields strictly-increasing ids.
    #[test]
    fn default_constructs_a_usable_generator() {
        let g = IdGenerator::default();
        let a = g.next_id();
        let b = g.next_id();
        assert!(b > a, "generator must yield monotonically increasing ids");
    }
}
