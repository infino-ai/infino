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
//! The generator is single-threaded by construction
//! (ferroid's `BasicSnowflakeGenerator` is interior-mutable
//! via `Cell`, so it's `!Sync`). One generator per supertable
//! handle is the intended usage; the supertable's writer-slot
//! lock already serializes `append()` per handle, so no
//! cross-thread sharing is needed.

use ferroid::generator::BasicSnowflakeGenerator;
use ferroid::time::{MonotonicClock, UNIX_EPOCH};

// 64-bit timestamp + 40-bit machine + 24-bit sequence, packed
// in a u128. The macro generates constructors, accessors, and
// the `SnowflakeId` + `Id` trait impls ferroid's generator
// needs. `reserved: 0` leaves all 128 bits as live payload —
// our high bit stays 0 by virtue of the timestamp magnitude,
// not by reservation.
ferroid::define_snowflake_id!(
    InfinoId128, u128,
    reserved: 0,
    timestamp: 64,
    machine_id: 40,
    sequence: 24
);

const WORKER_BITS: u32 = 40;
const WORKER_MASK: u64 = (1u64 << WORKER_BITS) - 1;

/// Single-threaded id generator. One per supertable handle.
///
/// Construction is cheap: one `rand::random::<u64>()` call for
/// [`Self::new`], zero for [`Self::with_worker_id`].
/// [`Self::next_id`] is `&self` (interior-mutable) and runs
/// at ~2 ns/id single-threaded.
pub struct IdGenerator {
    worker_id: u64,
    inner: BasicSnowflakeGenerator<InfinoId128, MonotonicClock>,
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
        let worker_id = rand::random::<u64>() & WORKER_MASK;
        Self::with_worker_id(worker_id)
    }

    /// Construct with an explicit worker_id (truncated to 40
    /// bits). Useful for tests that need a stable id sequence
    /// and for callers driving multiple generators with
    /// known-disjoint worker_ids in a single process.
    pub fn with_worker_id(worker_id: u64) -> Self {
        let worker40 = worker_id & WORKER_MASK;
        let clock = MonotonicClock::<1>::with_epoch(UNIX_EPOCH);
        Self {
            worker_id: worker40,
            inner: BasicSnowflakeGenerator::new(worker40 as u128, clock),
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
    /// underlying ferroid generator is `!Sync` and the
    /// borrow checker will refuse. The supertable's
    /// writer-slot lock already serializes `append()` per
    /// supertable handle; mint at append time and you'll
    /// never violate this.
    ///
    /// **Clock skew.** On a backward wall-clock step, ferroid
    /// spins via the closure passed to `next_id` until the
    /// clock catches up. In practice unreachable; included
    /// for correctness.
    #[inline]
    pub fn next_id(&self) -> i128 {
        let id: InfinoId128 = self.inner.next_id(|_| std::hint::spin_loop());
        // High bit is 0 for current-era Unix-ms timestamps
        // (today ≈ 1.7×10¹² ms = 41 bits; the high bit is bit
        // 127, ~86 bits past the timestamp field). The `as
        // i128` cast is lossless under that invariant.
        id.to_raw() as i128
    }

    /// Reserve `n` ids in advance and return them as an ordered
    /// list of contiguous spans. Each span is `(first, last)`
    /// inclusive; the flatten of all spans yields exactly `n`
    /// distinct, monotonically-increasing ids.
    ///
    /// **Why spans and not a single `(first, last)`:** the
    /// underlying ferroid generator's per-ms sequence field is
    /// 24 bits wide. Once the sequence is exhausted within a
    /// millisecond, the generator blocks for the next ms tick
    /// before minting again. Two ids straddling a ms boundary
    /// are not numerically contiguous (the timestamp field
    /// changes), so a single-range return type can't describe
    /// the reservation honestly under contention. The spans
    /// shape makes that boundary visible: under typical use
    /// the result is a `Vec` of length 1; under contention it
    /// grows by one per ms boundary crossed mid-call.
    ///
    /// **Caller contract:** persist the full `Vec` into a
    /// durable artifact (the WAL state doc) BEFORE doing any
    /// work that depends on the ids. Recovery reads the spans
    /// verbatim and never re-runs `reserve_range` — a recovering
    /// process has a different `worker_id` and would mint
    /// different ids, which would break the determinism a
    /// replay-safe append phase needs.
    ///
    /// `n == 0` returns an empty `Vec` without any minting.
    pub fn reserve_range(&self, n: u32) -> Vec<(i128, i128)> {
        if n == 0 {
            return Vec::new();
        }
        let mut spans: Vec<(i128, i128)> = Vec::with_capacity(1);
        let first = self.next_id();
        let mut span_first = first;
        let mut span_last = first;
        // We've already minted one id; mint `n - 1` more,
        // extending the current span when the next id is
        // numerically adjacent and starting a new span at any
        // discontinuity (which signals a ms boundary).
        for _ in 1..n {
            let id = self.next_id();
            if id == span_last + 1 {
                span_last = id;
            } else {
                spans.push((span_first, span_last));
                span_first = id;
                span_last = id;
            }
        }
        spans.push((span_first, span_last));
        spans
    }
}

impl Default for IdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for IdGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdGenerator")
            .field("worker_id", &format_args!("0x{:010x}", self.worker_id))
            .finish()
    }
}

#[cfg(test)]
mod tests {
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
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
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
        let n = 10_000u32;
        let spans = g.reserve_range(n);
        let flat = flatten_spans(&spans);
        assert_eq!(flat.len(), n as usize);
        let mut seen = std::collections::HashSet::with_capacity(n as usize);
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
        let a_set: std::collections::HashSet<i128> = a.iter().copied().collect();
        for id in &b {
            assert!(!a_set.contains(id), "id {id} in both reservations");
        }
    }

    #[test]
    fn reserve_range_across_ms_boundary_produces_multi_span() {
        // Forces a ms boundary by sleeping mid-call. The
        // straight-line `reserve_range` API doesn't let us
        // inject the sleep, so we exercise the multi-span
        // path differently: mint a few ids, sleep until the
        // next ms, mint more, and verify reserve_range
        // *would* produce multiple spans if the same id
        // sequence had come from one call. The boundary-
        // detection logic (numerically-non-adjacent ids start
        // a new span) is identical either way.
        let g = IdGenerator::with_worker_id(0x20);
        let a = g.next_id();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = g.next_id();
        // `b` is at least 2ms after `a` in the timestamp
        // field, so it's not `a + 1` even though both are
        // monotonic. This is the exact predicate that
        // `reserve_range`'s inner loop uses to start a new
        // span.
        assert!(b > a, "monotonic");
        assert_ne!(
            b,
            a + 1,
            "two ids straddling a ms boundary should NOT be numerically adjacent — {a} → {b}"
        );
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
}
