// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fixed-width bit packing over a little-endian bit stream: value `i` of
//! a `width`-bit stream occupies bits `i·width .. (i+1)·width`, least
//! significant bit first. Shared by the frame-of-reference id sidecar
//! and the position groups; both need random access to one value at a
//! fixed cost, which this gives with one unaligned 8-byte load per value
//! (plus the bytes past it for a value wider than [`ONE_LOAD_MAX_WIDTH`]
//! bits that starts mid-byte).

use std::mem::swap;

/// Widest value the stream carries: a full `u64`.
pub(crate) const MAX_WIDTH: u8 = 64;

/// Widest value one 8-byte load holds wherever it starts in its first
/// byte: the load keeps `64 - bit % 8` of the value's bits, at worst 57.
pub(crate) const ONE_LOAD_MAX_WIDTH: u8 = 57;

/// Bits needed to hold `v` (zero for zero).
#[inline]
pub(crate) fn width_of(v: u64) -> u8 {
    (u64::BITS - v.leading_zeros()) as u8
}

/// The low `width` bits set (`width <= 64`): the mask a `width`-bit value
/// is read out of a wider load with.
#[inline(always)]
pub(crate) const fn low_mask(width: u8) -> u64 {
    if width == MAX_WIDTH {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
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
    // One unaligned `u64` load covers any value of up to
    // `ONE_LOAD_MAX_WIDTH` bits that does not sit in the stream's last
    // bytes; the tail and the widest values take the byte loop below.
    if shift as usize + width as usize <= u64::BITS as usize
        && let Some(chunk) = payload.get(byte..byte + 8)
    {
        let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        return Some((word >> shift) & low_mask(width));
    }
    let n_bytes = (shift as usize + width as usize).div_ceil(8);
    let slice = payload.get(byte..byte + n_bytes)?;
    let mut acc: u128 = 0;
    for (k, &b) in slice.iter().enumerate() {
        acc |= (b as u128) << (8 * k);
    }
    Some(((acc >> shift) & u128::from(low_mask(width))) as u64)
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
/// the run ends past the payload. A run wider than [`ONE_LOAD_MAX_WIDTH`]
/// bits also reads the ninth byte a lane spills into (see [`sink_lanes`]);
/// a narrower run never tests for it. The id sidecar's block decode reads
/// its streams this way; the position streams take [`unpack_lanes_u32`].
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
    let mask = low_mask(width);
    let width = width as usize;
    let end_bit = (from + n) * width;
    if end_bit.div_ceil(8) > payload.len() {
        return None;
    }
    let bit = from * width;
    if width <= ONE_LOAD_MAX_WIDTH as usize {
        sink_lanes::<false>(payload, bit, n, width, mask, &mut sink);
    } else {
        sink_lanes::<true>(payload, bit, n, width, mask, &mut sink);
    }
    Some(())
}

/// [`for_each_lane`]'s loop over `n` lanes of `width` bits (`mask` their
/// low bits set) from bit `bit`, a run the caller has checked ends inside
/// `payload`.
///
/// One unaligned 8-byte load from a lane's first byte, shifted down, holds
/// the lane's first `64 - bit % 8` bits: all of it when it is at most
/// [`ONE_LOAD_MAX_WIDTH`] bits wide. A wider lane that starts mid-byte ends
/// in the ninth byte, which `SPILL` reads too:
///
/// ```text
///   a 63-bit lane starting at bit 3 of its first byte:
///     byte 0      bits 3..8    5 bits  ┐
///     bytes 1..8  all         56 bits  ├ the 8-byte load: 61 bits
///     byte 8      bits 0..2    2 bits  ─ the ninth byte
/// ```
///
/// The loop is built twice so a narrow run, the common case, carries no
/// test for the ninth byte: in a shared loop that test slows every width,
/// not only the wide ones. In the payload's last 8 bytes the load is
/// zero-padded; a lane that ends there never reaches a ninth byte.
#[inline(always)]
fn sink_lanes<const SPILL: bool>(
    payload: &[u8],
    mut bit: usize,
    n: usize,
    width: usize,
    mask: u64,
    sink: &mut impl FnMut(u64),
) {
    for _ in 0..n {
        let byte = bit / 8;
        let shift = bit % 8;
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
        let mut v = word >> shift;
        // The lane's last bit lies past the load, in byte `byte + 8`, which
        // the caller's check put inside the payload.
        if SPILL && shift + width > u64::BITS as usize {
            v |= u64::from(payload[byte + 8]) << (u64::BITS as usize - shift);
        }
        sink(v & mask);
        bit += width;
    }
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
    let mask = low_mask(width);
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
    #[cfg(target_arch = "x86_64")]
    let (start_bit, out) =
        if width <= SIMD_UNPACK_MAX_WIDTH && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: avx2 is present (checked above); see the function's notes
            // for the bounds every load and store keeps to.
            let done = unsafe { unpack_groups_avx2(payload, start_bit, width, out) };
            (start_bit + done * usize::from(width), &mut out[done..])
        } else {
            (start_bit, out)
        };
    unpack_groups_scalar(payload, start_bit, width, out);
}

/// [`unpack_groups`] without the vector kernel: every whole group by the
/// copy of [`unpack_groups_w`] built for `width`. The whole run on a target
/// without AVX2, the groups the vector kernel leaves on one with it.
#[inline]
fn unpack_groups_scalar(payload: &[u8], start_bit: usize, width: u8, out: &mut [u32]) {
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

/// Widest lane [`unpack_groups_avx2`] reads: a lane's bits, shifted by up to
/// 7 within its first byte, fit the 32-bit load it is gathered into.
#[cfg(target_arch = "x86_64")]
const SIMD_UNPACK_MAX_WIDTH: u8 = 25;

/// [`unpack_groups`] eight lanes per step: two 16-byte loads (lanes 0–3 and
/// 4–7 each span at most 13 bytes at this width), one byte shuffle that
/// gathers each lane's four bytes into its 32-bit slot, a per-lane shift and
/// a mask. The run's bit phase fixes the shuffle and the shifts, so both are
/// built once. Returns how many lanes it wrote (whole groups, from the
/// front); it stops where a load would pass the payload's end and the
/// scalar copy finishes the rest.
///
/// # Safety
///
/// The caller must have detected AVX2 at runtime. Given that, the function
/// reads and writes only in bounds for any input: each step's two 16-byte
/// loads lie inside `payload` (the loop runs while `byte + half + 16` fits),
/// each store lies inside `out` (while `done + 8` fits), and `width` is at
/// most [`SIMD_UNPACK_MAX_WIDTH`], so every shuffle index stays inside its
/// 16-byte lane and every shift plus width inside 32 bits.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_groups_avx2(
    payload: &[u8],
    start_bit: usize,
    width: u8,
    out: &mut [u32],
) -> usize {
    use std::arch::x86_64::*;
    debug_assert!((1..=SIMD_UNPACK_MAX_WIDTH).contains(&width));
    let w = usize::from(width);
    let phase = start_bit % 8;
    let offs: [usize; LANE_GROUP] = std::array::from_fn(|l| (phase + l * w) / 8);
    let half = offs[4];
    let mut shuffle = [0u8; 32];
    let mut shifts = [0u32; LANE_GROUP];
    for l in 0..LANE_GROUP {
        let (slot, from) = match l < 4 {
            true => (4 * l, offs[l]),
            false => (16 + 4 * (l - 4), offs[l] - half),
        };
        for b in 0..4 {
            shuffle[slot + b] = (from + b) as u8;
        }
        shifts[l] = ((phase + l * w) % 8) as u32;
    }
    let mut byte = start_bit / 8;
    let mut done = 0usize;
    // SAFETY, per obligation:
    // - AVX2 intrinsics: reachable only through the dispatcher's
    //   `is_x86_feature_detected!("avx2")` guard.
    // - Loads: both 16-byte loads start at `byte` and `byte + half` and the
    //   loop runs only while `byte + half + 16 <= payload.len()`, `half >= 0`.
    //   Unaligned variants, so no alignment precondition.
    // - Store: eight `u32`s at `out[done..done + 8]`, inside `out` while
    //   `done + 8 <= out.len()`.
    // - Shuffle indices are below 16 within each 128-bit half: a lane's
    //   offset from its half's start is at most `(7 + 3·25) / 8 = 10`, plus 3.
    unsafe {
        let shuffle_v = _mm256_loadu_si256(shuffle.as_ptr().cast());
        let shift_v = _mm256_loadu_si256(shifts.as_ptr().cast());
        let mask_v = _mm256_set1_epi32(low_mask(width) as i32);
        let src = payload.as_ptr();
        let dst = out.as_mut_ptr();
        while done + LANE_GROUP <= out.len() && byte + half + 16 <= payload.len() {
            let lo = _mm_loadu_si128(src.add(byte).cast());
            let hi = _mm_loadu_si128(src.add(byte + half).cast());
            let lanes = _mm256_shuffle_epi8(_mm256_set_m128i(hi, lo), shuffle_v);
            let values = _mm256_and_si256(_mm256_srlv_epi32(lanes, shift_v), mask_v);
            _mm256_storeu_si256(dst.add(done).cast(), values);
            done += LANE_GROUP;
            byte += w;
        }
    }
    done
}

/// Unpack `out.len() / 8` groups of eight `W`-bit lanes starting at
/// `start_bit`. The run's bit phase fixes each lane's byte offset and shift
/// within its group for the whole run; each group then advances `W` bytes.
#[inline(always)]
fn unpack_groups_w<const W: usize>(payload: &[u8], start_bit: usize, out: &mut [u32]) {
    let mask = const { low_mask(W as u8) };
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

/// Set bits [`expand_bitset_scalar`] writes per step.
const EXPAND_STEP: usize = 8;

/// Slots past its last doc that [`expand_bitset_scalar`] may write: its
/// output needs this much room beyond the docs.
pub(crate) const EXPAND_SLACK: usize = EXPAND_STEP - 1;

/// The doc ids a presence bitset's set bits stand for, `base` plus each bit's
/// index, ascending, into `out`; returns how many. Each word's set bits are
/// written [`EXPAND_STEP`] at a time whatever the word holds: a step writes
/// past the word's last bit into slack the next word overwrites, so the only
/// branch is the step count, not one exit per set bit, which the predictor
/// misses once a word. The last step overruns the docs by at most
/// [`EXPAND_SLACK`] slots, so `out` needs that room past them; a bitset whose
/// set bits would pass it panics on the slice bound instead.
pub(crate) fn expand_bitset_scalar(words: &[u8], base: u32, out: &mut [u32]) -> usize {
    let mut j = 0usize;
    for (wi, chunk) in words.chunks_exact(8).enumerate() {
        let mut word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        let at = base.wrapping_add(wi as u32 * 64);
        let n = word.count_ones() as usize;
        let mut done = 0usize;
        while done < n {
            for slot in &mut out[j + done..j + done + EXPAND_STEP] {
                *slot = at.wrapping_add(word.trailing_zeros());
                word &= word.wrapping_sub(1);
            }
            done += EXPAND_STEP;
        }
        j += n;
    }
    j
}

/// [`expand_bitset_avx512`] where the CPU has AVX-512F, which needs no slack
/// in `out`, else `None` and the caller runs [`expand_bitset_scalar`].
#[inline]
pub(crate) fn expand_bitset(words: &[u8], base: u32, out: &mut [u32]) -> Option<usize> {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: AVX-512F is present (checked above), the expansion's one
        // precondition; it bounds its own stores.
        return Some(unsafe { expand_bitset_avx512(words, base, out) });
    }
    let _ = (words, base, out);
    None
}

/// The doc ids a presence bitset's set bits stand for, `base` plus each bit's
/// index, ascending, into `out`: sixteen bits at a time, each slice's ids
/// built in one register, compressed and stored, so a word costs four steps
/// whatever it holds. Returns how many were written.
///
/// # Safety
///
/// The caller must have detected AVX-512F at runtime. Every store is
/// bounds-checked first: a masked store writes exactly the selected lanes,
/// and a slice whose set bits would pass `out`'s end panics instead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn expand_bitset_avx512(words: &[u8], base: u32, out: &mut [u32]) -> usize {
    use std::arch::x86_64::*;
    let iota = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let mut j = 0usize;
    for (wi, chunk) in words.chunks_exact(8).enumerate() {
        let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        // No branch on an empty slice: a dense term's slices are seldom
        // empty and a sparse one's unpredictably so, and an empty slice
        // stores nothing through its all-clear mask.
        for quarter in 0..4u32 {
            let mask = (word >> (16 * quarter)) as u16;
            let n = mask.count_ones() as usize;
            assert!(j + n <= out.len(), "bitset holds more docs than its block");
            let at = base.wrapping_add(wi as u32 * 64 + 16 * quarter) as i32;
            let ids = _mm512_add_epi32(_mm512_set1_epi32(at), iota);
            // Compressed in a register, then stored through a mask of its
            // first `n` lanes: a compress straight to memory is microcoded,
            // and many times slower, on some AVX-512 cores.
            let packed = _mm512_maskz_compress_epi32(mask, ids);
            let first_n = ((1u32 << n) - 1) as __mmask16;
            // SAFETY: the target feature is enabled on this function (see the
            // contract above). The masked store writes `n` consecutive u32s
            // from `out[j]`, inside `out` by the assert above; masked-off
            // lanes are neither written nor faulted on.
            unsafe { _mm512_mask_storeu_epi32(out.as_mut_ptr().add(j).cast(), first_n, packed) };
            j += n;
        }
    }
    j
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

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
                        x & low_mask(width)
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

    /// Bytes of slack past a group stream's payload, so every 8- and
    /// 16-byte load the group kernels make stays inside the buffer.
    const GROUP_SLACK: usize = 32;

    /// `n` pseudo-random `width`-bit lanes packed from bit 0, followed by
    /// `slack` zero bytes.
    fn random_stream(seed: &mut u64, n: usize, width: u8, slack: usize) -> Vec<u8> {
        let mut buf = vec![0u8; payload_bytes(n, width) + slack];
        for i in 0..n {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            put_bits(&mut buf, i * width as usize, *seed & low_mask(width), width);
        }
        buf
    }

    #[test]
    fn scalar_groups_agree_with_for_each_lane_at_every_width_and_phase() {
        // Called directly, so the scalar kernel is graded on every target,
        // including one whose dispatcher would hand the run to AVX2.
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        for width in 1..=32u8 {
            for from in 0..LANE_GROUP {
                for groups in [1usize, 2, 5, 16] {
                    let n = from + groups * LANE_GROUP;
                    let buf = random_stream(&mut seed, n, width, GROUP_SLACK);
                    let mut want = Vec::new();
                    for_each_lane(&buf, from, n - from, width, |v| want.push(v as u32))
                        .expect("in range");
                    let mut got = vec![u32::MAX; n - from];
                    unpack_groups_scalar(&buf, from * width as usize, width, &mut got);
                    assert_eq!(got, want, "width {width} from {from} groups {groups}");
                }
            }
        }
    }

    /// Bitsets of one to four words at random word-aligned origins — empty,
    /// full, and random — each with the doc ids its set bits stand for.
    fn bitset_cases() -> Vec<(Vec<u8>, u32, Vec<u32>)> {
        let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        (0..200)
            .map(|case| {
                let n_words = 1 + (next() % 4) as usize;
                let words: Vec<u64> = (0..n_words)
                    .map(|w| match case % 4 {
                        0 => 0,
                        1 => u64::MAX,
                        _ => next() & next() >> (w % 3),
                    })
                    .collect();
                let base = (next() % 1_000_000) as u32 & !63;
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let want: Vec<u32> = words
                    .iter()
                    .enumerate()
                    .flat_map(|(wi, &w)| {
                        (0..64)
                            .filter(move |b| (w >> b) & 1 == 1)
                            .map(move |b| base + wi as u32 * 64 + b)
                    })
                    .collect();
                (bytes, base, want)
            })
            .collect()
    }

    /// The scalar bitset expansion gives exactly the set bits' doc ids, in
    /// order, writing no further than its slack past them, and refuses a
    /// bitset holding more docs than its output has room for.
    #[test]
    fn scalar_bitset_expansion_agrees_with_the_set_bits() {
        for (case, (bytes, base, want)) in bitset_cases().into_iter().enumerate() {
            let mut got = vec![u32::MAX; 256 + EXPAND_SLACK + 1];
            let n = expand_bitset_scalar(&bytes, base, &mut got);
            assert_eq!(&got[..n], want.as_slice(), "case {case}");
            assert!(
                got[n + EXPAND_SLACK..].iter().all(|&g| g == u32::MAX),
                "case {case}: no write past the slack"
            );
        }
        let full = [0xFFu8; 8];
        let mut short = vec![0u32; 63];
        let refused = catch_unwind(AssertUnwindSafe(|| {
            expand_bitset_scalar(&full, 0, &mut short)
        }));
        assert!(
            refused.is_err(),
            "a bitset past its output's room must panic, not overrun"
        );
    }

    /// The AVX-512 bitset expansion gives exactly the set bits' doc ids, in
    /// order — so the scalar expansion's ids, bit for bit — for empty, full
    /// and random words at any origin, writing nothing past them, and
    /// refuses a bitset holding more docs than its output has room for.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_bitset_expansion_agrees_with_the_set_bits() {
        if !std::arch::is_x86_feature_detected!("avx512f") {
            return;
        }
        for (case, (bytes, base, want)) in bitset_cases().into_iter().enumerate() {
            let mut got = vec![u32::MAX; 256];
            // SAFETY: avx512f was detected above.
            let n = unsafe { expand_bitset_avx512(&bytes, base, &mut got) };
            assert_eq!(&got[..n], want.as_slice(), "case {case}");
            assert!(
                got[n..].iter().all(|&g| g == u32::MAX),
                "case {case}: no write past the docs"
            );
        }
        let full = [0xFFu8; 8];
        let mut short = vec![0u32; 63];
        // SAFETY: avx512f was detected above.
        let refused = catch_unwind(AssertUnwindSafe(|| unsafe {
            expand_bitset_avx512(&full, 0, &mut short)
        }));
        assert!(
            refused.is_err(),
            "a bitset past its output's room must panic, not overrun"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_groups_agree_with_the_scalar_groups() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        for width in 1..=SIMD_UNPACK_MAX_WIDTH {
            for from in 0..LANE_GROUP {
                for groups in [1usize, 2, 5, 16] {
                    // With slack every group fits the vector loads; without
                    // it the kernel stops short and leaves the tail.
                    for slack in [0, GROUP_SLACK] {
                        let n = from + groups * LANE_GROUP;
                        let buf = random_stream(&mut seed, n, width, slack);
                        let start_bit = from * usize::from(width);
                        let mut simd = vec![u32::MAX; n - from];
                        // SAFETY: avx2 was detected above.
                        let done = unsafe { unpack_groups_avx2(&buf, start_bit, width, &mut simd) };
                        assert!(done.is_multiple_of(LANE_GROUP) && done <= simd.len());
                        if slack > 0 {
                            assert_eq!(done, simd.len(), "width {width}: every group fits");
                        }
                        // The scalar kernel needs its loads inside the buffer
                        // too: give it the same lanes with the slack.
                        let mut padded = buf.clone();
                        padded.resize(payload_bytes(n, width) + GROUP_SLACK, 0);
                        let mut scalar = vec![0u32; n - from];
                        unpack_groups_scalar(&padded, start_bit, width, &mut scalar);
                        assert_eq!(
                            simd[..done],
                            scalar[..done],
                            "width {width} from {from} groups {groups} slack {slack}"
                        );
                        assert!(
                            simd[done..].iter().all(|&v| v == u32::MAX),
                            "no write past done"
                        );
                    }
                }
            }
        }
    }

    /// A stream long enough that narrow lanes are read through the plain
    /// 8-byte load, not only the zero-padded one at the payload's end.
    const LONG_STREAM_LANES: usize = 41;

    /// Stream lengths the lane readers are checked at: 1 to 16 lanes end
    /// the payload at every bit phase (eight lanes end on a byte boundary),
    /// so a wide lane spilling into the payload's last byte is read; one
    /// long stream covers the plain load.
    fn stream_lengths() -> impl Iterator<Item = usize> {
        (1..=2 * LANE_GROUP).chain([LONG_STREAM_LANES])
    }

    /// `n` values of `width` bits, high bits set in most, and their stream.
    fn packed_stream(n: usize, width: u8) -> (Vec<u64>, Vec<u8>) {
        let values: Vec<u64> = (0..n as u64)
            .map(|i| (i.wrapping_mul(0x9E37_79B9_7F4A_7C15)) & low_mask(width))
            .collect();
        let mut buf = vec![0u8; payload_bytes(n, width)];
        for (i, &v) in values.iter().enumerate() {
            put_bits(&mut buf, i * width as usize, v, width);
        }
        (values, buf)
    }

    #[test]
    fn for_each_lane_reads_runs_at_every_width_and_refuses_overruns() {
        for width in 0..=MAX_WIDTH {
            for n in stream_lengths() {
                let (values, buf) = packed_stream(n, width);
                for from in 0..=n {
                    for count in 0..=n - from {
                        let mut got = Vec::new();
                        for_each_lane(&buf, from, count, width, |v| got.push(v)).expect("in range");
                        assert_eq!(
                            got,
                            values[from..from + count],
                            "width {width} lanes {n} from {from} count {count}"
                        );
                    }
                }
                // The check is byte-granular — lanes inside the last byte's
                // padding read as zeros — so overrun by a couple of bytes' worth.
                if width > 0 {
                    assert!(for_each_lane(&buf, n, 16, width, |_| {}).is_none());
                }
            }
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
            for n in stream_lengths() {
                let (values, buf) = packed_stream(n, width);
                for (i, &v) in values.iter().enumerate() {
                    assert_eq!(
                        get_bits(&buf, i, width),
                        Some(v),
                        "width {width} lanes {n} value {i}"
                    );
                }
                // A value a whole byte past the stream is out of range (a
                // zero-width stream has no end to run past).
                assert_eq!(get_bits(&buf, n + 8, width).is_some(), width == 0);
            }
        }
        assert_eq!(width_of(0), 0);
        assert_eq!(width_of(1), 1);
        assert_eq!(width_of(255), 8);
        assert_eq!(width_of(u64::MAX), 64);
    }
}
