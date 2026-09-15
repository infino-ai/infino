// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Fixed-width bit packing over a little-endian bit stream: value `i` of
//! a `width`-bit stream occupies bits `i·width .. (i+1)·width`, least
//! significant bit first. Shared by the frame-of-reference id sidecar
//! and the position groups; both need random access to one value at a
//! fixed cost, which this gives with one unaligned load per value.

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
pub(crate) struct ExceptionPlan {
    pub(crate) width: u8,
    pub(crate) exceptions: Vec<(u32, u32)>,
    pub(crate) bytes: usize,
}

/// The cheapest patched packing of `lanes`: every width below the plain
/// one is tried and the smallest total kept, where a width costs
/// `base(width)` for its packed low bits plus `per_exception(lane, hi)`
/// for each lane whose high bits do not fit — at most `max_exceptions`
/// of them. Returns the plain width's plan when nothing beats it.
pub(crate) fn plan_exceptions(
    lanes: &[u32],
    max_exceptions: usize,
    base: impl Fn(u8) -> usize,
    per_exception: impl Fn(u32, u32) -> usize,
) -> ExceptionPlan {
    let plain = width_of(lanes.iter().copied().max().unwrap_or(0).into());
    let mut best = ExceptionPlan {
        width: plain,
        exceptions: Vec::new(),
        bytes: base(plain),
    };
    for width in 0..plain {
        let mut exceptions = Vec::new();
        let mut bytes = base(width);
        let mut beaten = bytes >= best.bytes;
        for (i, &v) in lanes.iter().enumerate() {
            if beaten {
                break;
            }
            let hi = if width == 0 { v } else { v >> width };
            if hi != 0 {
                exceptions.push((i as u32, hi));
                bytes += per_exception(i as u32, hi);
                beaten = exceptions.len() > max_exceptions || bytes >= best.bytes;
            }
        }
        if !beaten {
            best = ExceptionPlan {
                width,
                exceptions,
                bytes,
            };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

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
