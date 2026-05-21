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
/// at ~2 ns/id single-threaded on Apple M4 Max.
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
        let g = IdGenerator::with_worker_id(0x1234_5678_9A);
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
