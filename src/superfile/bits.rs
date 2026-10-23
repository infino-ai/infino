// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fixed-width bit packing over a little-endian bit stream: value `i` of
//! a `width`-bit stream occupies bits `i·width .. (i+1)·width`, least
//! significant bit first. Shared by the frame-of-reference id sidecar
//! and the position groups; both need random access to one value at a
//! fixed cost, which this gives with one unaligned load per value.

use std::mem::swap;

/// Widest value the stream carries: a full `u64`.
pub(crate) const MAX_WIDTH: u8 = 64;

/// Bits needed to hold `v` (zero for zero).
#[inline]
pub(crate) fn width_of(v: u64) -> u8 {
    (u64::BITS - v.leading_zeros()) as u8
}

/// Bytes a stream of `n` values at `width` bits occupies.
#[inline]
pub(crate) fn payload_bytes(n: usize, width: u8) -> usize {
    (n * width as usize).div_ceil(8)
}

/// OR `v` (`width` bits) into `buf` at bit position `bit`. `buf` must
/// already be zeroed over the bytes the value lands on.
#[inline]
pub(crate) fn put_bits(buf: &mut [u8], bit: usize, v: u64, width: u8) {
    if width == 0 {
        return;
    }
    let first = bit / 8;
    let shift = (bit % 8) as u32;
    // Up to 71 bits land across at most 9 bytes.
    let acc: u128 = (v as u128) << shift;
    let n_bytes = (shift as usize + width as usize).div_ceil(8);
    for (i, slot) in buf[first..first + n_bytes].iter_mut().enumerate() {
        *slot |= (acc >> (8 * i)) as u8;
    }
}

/// Value `i` of a `width`-bit stream in `payload`; `None` past its end.
#[inline]
pub(crate) fn get_bits(payload: &[u8], i: usize, width: u8) -> Option<u64> {
    if width == 0 {
        return Some(0);
    }
    let bit = i * width as usize;
    let byte = bit / 8;
    let shift = (bit % 8) as u32;
    // One unaligned `u64` load covers any value of up to 57 bits that
    // does not sit in the stream's last bytes; the tail and the widest
    // values take the byte loop below.
    if shift as usize + width as usize <= u64::BITS as usize
        && let Some(chunk) = payload.get(byte..byte + 8)
    {
        let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        let mask: u64 = if width == MAX_WIDTH {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        return Some((word >> shift) & mask);
    }
    let n_bytes = (shift as usize + width as usize).div_ceil(8);
    let slice = payload.get(byte..byte + n_bytes)?;
    let mut acc: u128 = 0;
    for (k, &b) in slice.iter().enumerate() {
        acc |= (b as u128) << (8 * k);
    }
    let mask: u128 = if width == MAX_WIDTH {
        u64::MAX as u128
    } else {
        (1u128 << width) - 1
    };
    Some(((acc >> shift) & mask) as u64)
}

/// A patched packing: the width most lanes fit, the lanes that do not
/// with their high bits (ascending lane order), and the bytes the whole
/// takes under the caller's cost model.
#[derive(Default)]
pub(crate) struct ExceptionPlan {
    pub(crate) width: u8,
    pub(crate) exceptions: Vec<(u32, u32)>,
    pub(crate) bytes: usize,
}

/// The buffers the patched encoders reuse from block to block: two
/// plans (a block's deltas and tfs, or a group's first positions and
/// gaps), the candidate list the width search fills, and two lane
/// buffers for splitting a group into its streams. One per writer, so
/// encoding a block allocates nothing.
#[derive(Default)]
pub struct PackScratch {
    pub(crate) plan_a: ExceptionPlan,
    pub(crate) plan_b: ExceptionPlan,
    pub(crate) candidates: Vec<(u32, u32)>,
    pub(crate) lanes_a: Vec<u32>,
    pub(crate) lanes_b: Vec<u32>,
}

/// The cheapest patched packing of `lanes`, written into `plan`: every
/// width below the plain one is tried and the smallest total kept,
/// where a width costs `base(width)` for its packed low bits plus
/// `per_exception(lane, hi)` for each lane whose high bits do not fit —
/// at most `max_exceptions` of them. The plain width's plan when nothing
/// beats it. `candidates` is scratch for the width under trial; a
/// winning width's list is swapped into `plan`, so neither grows past
/// the block's lane count.
pub(crate) fn plan_exceptions(
    lanes: &[u32],
    max_exceptions: usize,
    base: impl Fn(u8) -> usize,
    per_exception: impl Fn(u32, u32) -> usize,
    plan: &mut ExceptionPlan,
    candidates: &mut Vec<(u32, u32)>,
) {
    let plain = width_of(lanes.iter().copied().max().unwrap_or(0).into());
    plan.width = plain;
    plan.exceptions.clear();
    plan.bytes = base(plain);
    for width in 0..plain {
        candidates.clear();
        let mut bytes = base(width);
        let mut beaten = bytes >= plan.bytes;
        for (i, &v) in lanes.iter().enumerate() {
            if beaten {
                break;
            }
            let hi = if width == 0 { v } else { v >> width };
            if hi != 0 {
                candidates.push((i as u32, hi));
                bytes += per_exception(i as u32, hi);
                beaten = candidates.len() > max_exceptions || bytes >= plan.bytes;
            }
        }
        if !beaten {
            plan.width = width;
            plan.bytes = bytes;
            swap(&mut plan.exceptions, candidates);
        }
    }
}

/// Lanes `from..from + n` of a `width`-bit stream, one 64-bit load per
/// lane after a single bounds check for the run, to `sink`. `None` when
/// the run ends past the payload. Shared by the position streams and
/// the id sidecar's block decode.
#[inline]
pub(crate) fn for_each_lane(
    payload: &[u8],
    from: usize,
    n: usize,
    width: u8,
    mut sink: impl FnMut(u64),
) -> Option<()> {
    if width == 0 {
        for _ in 0..n {
            sink(0);
        }
        return Some(());
    }
    let width = width as usize;
    let end_bit = (from + n) * width;
    if end_bit.div_ceil(8) > payload.len() {
        return None;
    }
    let mask: u64 = if width == MAX_WIDTH as usize {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let mut bit = from * width;
    for _ in 0..n {
        let byte = bit / 8;
        let word = match payload.get(byte..byte + 8) {
            Some(chunk) => u64::from_le_bytes(chunk.try_into().expect("8 bytes")),
            None => {
                // The stream's last bytes: pad the load.
                let mut tail = [0u8; 8];
                let rest = &payload[byte..];
                tail[..rest.len()].copy_from_slice(rest);
                u64::from_le_bytes(tail)
            }
        };
        sink((word >> (bit % 8)) & mask);
        bit += width;
    }
    Some(())
}

/// Lanes `from..from + out.len()` of a `width`-bit stream (`width <= 32`)
/// written into `out`: the values [`for_each_lane`] yields, without its
/// per-lane sink call or tail test. Every lane whose 8-byte load lies inside
/// the payload is read in one straight loop; only the stream's last few
/// take the padded load. `None` when the run ends past the payload.
#[inline]
pub(crate) fn unpack_lanes_u32(
    payload: &[u8],
    from: usize,
    width: u8,
    out: &mut [u32],
) -> Option<()> {
    let n = out.len();
    if width == 0 {
        out.fill(0);
        return Some(());
    }
    debug_assert!(width <= 32, "a u32 lane is at most 32 bits wide");
    let w = width as usize;
    if ((from + n) * w).div_ceil(8) > payload.len() {
        return None;
    }
    let mask: u64 = (1u64 << w) - 1;
    // Lane `l` loads bytes `l·w/8 .. +8`, inside the payload while
    // `l ≤ ((len − 8)·8 + 7) / w`.
    let fast = match payload.len().checked_sub(8) {
        Some(room) => ((room * 8 + 7) / w + 1).saturating_sub(from).min(n),
        None => 0,
    };
    let mut bit = from * w;
    let (head, tail) = out.split_at_mut(fast);
    // Whole groups of eight lanes first: eight lanes of `w` bits span
    // exactly `w` bytes, so each lane's byte offset and shift within its
    // group are the same in every group and are worked out once per run.
    let (grouped, head) = head.split_at_mut(head.len() / LANE_GROUP * LANE_GROUP);
    unpack_groups(payload, bit, width, grouped);
    bit += grouped.len() * w;
    for slot in head {
        let byte = bit / 8;
        let word = u64::from_le_bytes(
            payload[byte..byte + 8]
                .try_into()
                .expect("invariant: a fast lane's load is in bounds"),
        );
        *slot = ((word >> (bit % 8)) & mask) as u32;
        bit += w;
    }
    for slot in tail {
        let byte = bit / 8;
        let mut padded = [0u8; 8];
        let rest = &payload[byte..payload.len().min(byte + 8)];
        padded[..rest.len()].copy_from_slice(rest);
        *slot = ((u64::from_le_bytes(padded) >> (bit % 8)) & mask) as u32;
        bit += w;
    }
    Some(())
}

/// Lanes a width-specialized group unpacks together; eight lanes of any
/// width end on a byte boundary.
const LANE_GROUP: usize = 8;

/// [`unpack_lanes_u32`]'s whole groups, dispatched once to the copy built
/// for `width`, so the inner loop's shifts and masks are constants.
/// `start_bit` is the first lane's bit position; every lane's 8-byte load
/// must lie inside `payload`.
fn unpack_groups(payload: &[u8], start_bit: usize, width: u8, out: &mut [u32]) {
    macro_rules! dispatch {
        ($($w:literal)*) => {
            match width {
                $($w => unpack_groups_w::<$w>(payload, start_bit, out),)*
                _ => unreachable!("a u32 lane is 1 to 32 bits wide"),
            }
        };
    }
    dispatch!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32);
}

/// Unpack `out.len() / 8` groups of eight `W`-bit lanes starting at
/// `start_bit`. The run's bit phase fixes each lane's byte offset and shift
/// within its group for the whole run; each group then advances `W` bytes.
#[inline(always)]
fn unpack_groups_w<const W: usize>(payload: &[u8], start_bit: usize, out: &mut [u32]) {
    let mask: u64 = (1u64 << W) - 1;
    let phase = start_bit % 8;
    let offs: [usize; LANE_GROUP] = std::array::from_fn(|l| (phase + l * W) / 8);
    let shifts: [u32; LANE_GROUP] = std::array::from_fn(|l| ((phase + l * W) % 8) as u32);
    let span = offs[LANE_GROUP - 1] + 8;
    let mut byte = start_bit / 8;
    for group in out.chunks_exact_mut(LANE_GROUP) {
        let chunk = &payload[byte..byte + span];
        for (l, slot) in group.iter_mut().enumerate() {
            let word = u64::from_le_bytes(
                chunk[offs[l]..offs[l] + 8]
                    .try_into()
                    .expect("invariant: eight bytes"),
            );
            *slot = ((word >> shifts[l]) & mask) as u32;
        }
        byte += W;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Plan with fresh scratch and hand the plan back.
    fn planned(
        lanes: &[u32],
        max_exceptions: usize,
        base: impl Fn(u8) -> usize,
        per_exception: impl Fn(u32, u32) -> usize,
    ) -> ExceptionPlan {
        let mut out = ExceptionPlan::default();
        let mut candidates = Vec::new();
        plan_exceptions(
            lanes,
            max_exceptions,
            base,
            per_exception,
            &mut out,
            &mut candidates,
        );
        out
    }

    #[test]
    fn unpack_lanes_agrees_with_for_each_lane_at_every_width_and_offset() {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        for width in 0..=32u8 {
            for n_vals in [1usize, 2, 7, 8, 9, 31, 64, 129] {
                let vals: Vec<u64> = (0..n_vals)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        if width == 0 {
                            0
                        } else {
                            x & ((1u64 << width) - 1)
                        }
                    })
                    .collect();
                let mut buf = vec![0u8; payload_bytes(n_vals, width)];
                for (i, &v) in vals.iter().enumerate() {
                    put_bits(&mut buf, i * width as usize, v, width);
                }
                for from in 0..n_vals {
                    for n in 0..=n_vals - from {
                        let mut want = Vec::new();
                        for_each_lane(&buf, from, n, width, |v| want.push(v as u32))
                            .expect("in range");
                        let mut got = vec![u32::MAX; n];
                        unpack_lanes_u32(&buf, from, width, &mut got).expect("in range");
                        assert_eq!(got, want, "width {width} from {from} n {n}");
                    }
                    // Past the payload both refuse alike (a partial last
                    // byte may still hold one more whole lane).
                    let n = n_vals - from + 9;
                    let mut over = vec![0u32; n];
                    assert_eq!(
                        unpack_lanes_u32(&buf, from, width, &mut over).is_none(),
                        for_each_lane(&buf, from, n, width, |_| ()).is_none(),
                        "width {width} from {from}: overrun"
                    );
                }
            }
        }
    }

    #[test]
    fn for_each_lane_reads_runs_at_every_width_and_refuses_overruns() {
        for width in [0u8, 1, 5, 8, 13, 32, 64] {
            let n = 41;
            let values: Vec<u64> = (0..n as u64)
                .map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & mask(width))
                .collect();
            let mut buf = vec![0u8; payload_bytes(n, width)];
            for (i, &v) in values.iter().enumerate() {
                put_bits(&mut buf, i * width as usize, v, width);
            }
            for (from, count) in [(0usize, n), (7, 9), (n - 1, 1), (n, 0)] {
                let mut got = Vec::new();
                for_each_lane(&buf, from, count, width, |v| got.push(v)).expect("in range");
                assert_eq!(got, values[from..from + count], "width {width} from {from}");
            }
            // The check is byte-granular — lanes inside the last byte's
            // padding read as zeros — so overrun by a couple of bytes' worth.
            if width > 0 {
                assert!(for_each_lane(&buf, n, 16, width, |_| {}).is_none());
            }
        }
    }

    fn mask(width: u8) -> u64 {
        match width {
            64 => u64::MAX,
            w => (1u64 << w) - 1,
        }
    }

    #[test]
    fn plan_exceptions_picks_the_cheapest_width_and_keeps_lane_order() {
        // 128 lanes of 1..=7 (3 bits) with three 20-bit outliers: plain
        // packing costs 20 bits per lane; the plan drops to 3 bits and
        // lists the three outliers in ascending lane order with their
        // high bits, at the caller's cost of 2 bytes per exception.
        let mut lanes = vec![0u32; 128];
        for (i, l) in lanes.iter_mut().enumerate() {
            *l = 1 + (i as u32 % 7);
        }
        lanes[5] = 1 << 19;
        lanes[70] = 3 << 18;
        lanes[127] = 1 << 12;
        let base = |width: u8| 128 * width as usize / 8;
        let plan = planned(&lanes, 31, base, |_, _| 2);
        assert_eq!(plan.width, 3);
        assert_eq!(plan.bytes, base(3) + 3 * 2);
        assert_eq!(
            plan.exceptions,
            vec![
                (5, (1 << 19) >> 3),
                (70, (3 << 18) >> 3),
                (127, (1 << 12) >> 3)
            ]
        );
        // Uniform lanes: nothing beats plain.
        let uniform = vec![5u32; 128];
        let plan = planned(&uniform, 31, base, |_, _| 2);
        assert_eq!(
            (plan.width, plan.exceptions.len(), plan.bytes),
            (3, 0, base(3))
        );
        // All-zero lanes: width 0 and no exceptions.
        let plan = planned(&[0u32; 16], 31, |w| 2 * w as usize, |_, _| 2);
        assert_eq!((plan.width, plan.bytes), (0, 0));
    }

    #[test]
    fn plan_exceptions_honours_the_cap_and_the_cost_model() {
        // 40 outliers among 128 lanes: with a cap of 31 the narrow width
        // is out of reach and the plan stays plain; with a cap of 64 it
        // is taken. A dear per-exception cost also keeps it plain.
        let mut lanes = vec![1u32; 128];
        for i in 0..40 {
            lanes[i * 3] = 1 << 15;
        }
        let base = |width: u8| 128 * width as usize / 8;
        let capped = planned(&lanes, 31, base, |_, _| 2);
        assert_eq!((capped.width, capped.exceptions.len()), (16, 0));
        let roomy = planned(&lanes, 64, base, |_, _| 2);
        assert_eq!((roomy.width, roomy.exceptions.len()), (1, 40));
        assert!(roomy.bytes < capped.bytes);
        let dear = planned(&lanes, 64, base, |_, _| 100);
        assert_eq!(dear.exceptions.len(), 0);
    }

    #[test]
    fn every_width_round_trips_at_every_alignment() {
        for width in 0..=MAX_WIDTH {
            let n = 37;
            let values: Vec<u64> = (0..n as u64)
                .map(|i| {
                    let mask = if width == 64 {
                        u64::MAX
                    } else {
                        (1u64 << width) - 1
                    };
                    (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & mask
                })
                .collect();
            let mut buf = vec![0u8; payload_bytes(n, width)];
            for (i, &v) in values.iter().enumerate() {
                put_bits(&mut buf, i * width as usize, v, width);
            }
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(get_bits(&buf, i, width), Some(v), "width {width} value {i}");
            }
            // A value a whole byte past the stream is out of range (a
            // zero-width stream has no end to run past).
            assert_eq!(get_bits(&buf, n + 8, width).is_some(), width == 0);
        }
        assert_eq!(width_of(0), 0);
        assert_eq!(width_of(1), 1);
        assert_eq!(width_of(255), 8);
        assert_eq!(width_of(u64::MAX), 64);
    }
}
