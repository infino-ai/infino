// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Position-run encoding for positional FTS columns.
//!
//! A **run** is one document's token positions for one term, stored as
//! LEB128 varints: the first position absolute, each subsequent value
//! the gap to the previous position (positions within a doc are
//! strictly increasing, so gaps are ≥ 1 and small for clustered
//! terms). A run's varint count equals the posting's `tf`, so runs
//! need no length framing — the decoder reads exactly `tf` values.
//!
//! The positions region of the FTS blob is, per term, the
//! concatenation of its runs in posting (doc-id) order; the skip
//! table records each 128-doc block's starting byte so a block's runs
//! are randomly addressable without decoding its predecessors.

use crate::superfile::bits::{payload_bytes, put_bits, unpack_all, width_of};

/// Largest byte length one encoded `u32` can occupy (LEB128: 5 × 7
/// bits ≥ 32 bits). Used to reserve scratch capacity.
/// (Consumed by the read path that follows in this series.)
#[allow(dead_code)]
pub(crate) const MAX_VARINT_BYTES: usize = 5;

/// LEB128 continuation flag: high bit set ⇒ another byte follows.
const CONTINUATION_BIT: u8 = 0x80;
/// Payload bits per LEB128 byte.
const PAYLOAD_BITS: u32 = 7;
/// Payload mask per LEB128 byte.
const PAYLOAD_MASK: u8 = 0x7f;

/// Append one `u32` as LEB128 to `out`.
#[inline]
pub(crate) fn push_varint(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let byte = (v as u8) & PAYLOAD_MASK;
        v >>= PAYLOAD_BITS;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | CONTINUATION_BIT);
    }
}

/// Decode one LEB128 `u32` from `bytes` starting at `*at`, advancing
/// `*at` past it. Returns `None` on truncated input or a value that
/// overflows `u32` — both only reachable on corrupt bytes, which the
/// caller surfaces as a read error.
#[inline]
pub(crate) fn read_varint(bytes: &[u8], at: &mut usize) -> Option<u32> {
    let mut v: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        let &b = bytes.get(*at)?;
        *at += 1;
        let payload = (b & PAYLOAD_MASK) as u32;
        v |= payload.checked_shl(shift)?;
        if shift > 0 && payload >> (32 - shift.min(32)) != 0 {
            // Payload bits past the 32-bit boundary ⇒ overflow.
            return None;
        }
        if b & CONTINUATION_BIT == 0 {
            return Some(v);
        }
        shift += PAYLOAD_BITS;
        if shift >= 32 + PAYLOAD_BITS {
            return None;
        }
    }
}

/// Append one document's position run — first value absolute, then
/// gaps. `positions` must be strictly increasing (token positions
/// within one doc always are).
pub(crate) fn encode_run(out: &mut Vec<u8>, positions: &[u32]) {
    let mut prev: u32 = 0;
    for (i, &p) in positions.iter().enumerate() {
        debug_assert!(i == 0 || p > prev, "positions must be strictly increasing");
        let delta = if i == 0 { p } else { p - prev };
        push_varint(out, delta);
        prev = p;
    }
}

/// Decode one run of exactly `tf` positions from `bytes` at `*at`,
/// appending the absolute positions to `out` and advancing `*at`.
/// `None` on corrupt (truncated / overflowing) bytes.
#[allow(dead_code)]
pub(crate) fn decode_run(bytes: &[u8], at: &mut usize, tf: u32, out: &mut Vec<u32>) -> Option<()> {
    let mut prev: u32 = 0;
    for i in 0..tf {
        let delta = read_varint(bytes, at)?;
        let p = if i == 0 {
            delta
        } else {
            prev.checked_add(delta)?
        };
        out.push(p);
        prev = p;
    }
    Some(())
}

/// Advance `*at` past one run of `tf` positions without materializing
/// them. `None` on truncated bytes.
pub(crate) fn skip_run(bytes: &[u8], at: &mut usize, tf: u32) -> Option<()> {
    for _ in 0..tf {
        read_varint(bytes, at)?;
    }
    Some(())
}

/// Group header value for a group stored as LEB128 runs (the layout
/// every blob before `VERSION_V7` used for all of its runs).
pub(crate) const GROUP_LEB128: u8 = 0;
/// Widest packed position value: a `u32`.
const GROUP_MAX_WIDTH: u8 = 32;

/// Append one **position group** — the run values (first position
/// absolute per doc, then gaps, in posting order) of one posting block,
/// or of a whole short-form term — behind a one-byte header. The header
/// is the bit width the values are packed at, or [`GROUP_LEB128`] when
/// the values are LEB128 runs because that is smaller (an outlier gap
/// would otherwise widen every value). The reader knows how many values
/// the group holds (the block's tf sum), so neither form carries a count.
///
/// From `VERSION_V7` every group takes this shape; a phrase decode reads
/// a packed group whole and indexes it by the block's tf prefix sums,
/// where a LEB128 group is walked run by run as before.
pub(crate) fn encode_group(out: &mut Vec<u8>, values: &[u32]) {
    let width = width_of(values.iter().copied().max().unwrap_or(0).into());
    let packed_len = payload_bytes(values.len(), width);
    let leb_len: usize = values.iter().map(|&v| varint_len(v)).sum();
    if width == 0 || leb_len <= packed_len {
        out.push(GROUP_LEB128);
        for &v in values {
            push_varint(out, v);
        }
        return;
    }
    debug_assert!(width <= GROUP_MAX_WIDTH);
    out.push(width);
    let start = out.len();
    out.resize(start + packed_len, 0);
    for (i, &v) in values.iter().enumerate() {
        put_bits(&mut out[start..], i * width as usize, u64::from(v), width);
    }
}

/// Encoded LEB128 length of `v`.
#[inline]
pub(crate) fn varint_len(v: u32) -> usize {
    match v {
        0..=0x7f => 1,
        0x80..=0x3fff => 2,
        0x4000..=0x1f_ffff => 3,
        0x20_0000..=0xfff_ffff => 4,
        _ => 5,
    }
}

/// Decode a whole group of `n` values starting at `*at` (its header
/// byte), appending the run values to `out` and advancing `*at` past the
/// group. `None` on a truncated or malformed group.
pub(crate) fn decode_group(
    bytes: &[u8],
    at: &mut usize,
    n: usize,
    out: &mut Vec<u32>,
) -> Option<()> {
    let header = *bytes.get(*at)?;
    *at += 1;
    if header == GROUP_LEB128 {
        for _ in 0..n {
            out.push(read_varint(bytes, at)?);
        }
        return Some(());
    }
    if header > GROUP_MAX_WIDTH {
        return None;
    }
    let len = payload_bytes(n, header);
    let payload = bytes.get(*at..*at + len)?;
    unpack_all(payload, n, header, out)?;
    *at += len;
    Some(())
}

/// Turn a run's `tf` values (first absolute, then gaps) into absolute
/// positions, appending to `out`. `None` on an overflowing gap.
#[inline]
pub(crate) fn positions_from_run_values(values: &[u32], out: &mut Vec<u32>) -> Option<()> {
    let mut prev: u32 = 0;
    for (i, &delta) in values.iter().enumerate() {
        let p = match i {
            0 => delta,
            _ => prev.checked_add(delta)?,
        };
        out.push(p);
        prev = p;
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_picks_the_smaller_form_and_round_trips() {
        // Small gaps pack: 200 values of 5 bits = 125 B + 1 vs 200 B LEB128.
        let small: Vec<u32> = (0..200u32).map(|i| 3 + i % 20).collect();
        let mut out = Vec::new();
        encode_group(&mut out, &small);
        assert_ne!(out[0], GROUP_LEB128);
        assert_eq!(out.len(), 1 + payload_bytes(small.len(), 5));
        let mut at = 0;
        let mut back = Vec::new();
        decode_group(&out, &mut at, small.len(), &mut back).expect("decodes");
        assert_eq!(back, small);
        assert_eq!(at, out.len());

        // One huge outlier would widen every value: LEB128 wins and is chosen.
        let mut outlier = small.clone();
        outlier[7] = 1 << 30;
        let mut out = Vec::new();
        encode_group(&mut out, &outlier);
        assert_eq!(out[0], GROUP_LEB128);
        let mut at = 0;
        let mut back = Vec::new();
        decode_group(&out, &mut at, outlier.len(), &mut back).expect("decodes");
        assert_eq!(back, outlier);
        assert_eq!(at, out.len());

        // A single value and a zero value.
        for v in [vec![0u32], vec![u32::MAX], vec![5, 0, 0, 7]] {
            let mut out = Vec::new();
            encode_group(&mut out, &v);
            let mut at = 0;
            let mut back = Vec::new();
            decode_group(&out, &mut at, v.len(), &mut back).expect("decodes");
            assert_eq!(back, v);
        }
        // Truncation is refused, not a panic.
        let mut out = Vec::new();
        encode_group(&mut out, &small);
        for cut in 0..out.len() {
            let mut at = 0;
            assert!(decode_group(&out[..cut], &mut at, small.len(), &mut Vec::new()).is_none());
        }
        assert!(
            decode_group(&[33], &mut 0, 1, &mut Vec::new()).is_none(),
            "width past 32"
        );
    }

    #[test]
    fn varint_round_trips_boundaries() {
        for v in [0u32, 1, 127, 128, 16383, 16384, u32::MAX - 1, u32::MAX] {
            let mut buf = Vec::new();
            push_varint(&mut buf, v);
            assert!(buf.len() <= MAX_VARINT_BYTES);
            let mut at = 0;
            assert_eq!(read_varint(&buf, &mut at), Some(v));
            assert_eq!(at, buf.len());
        }
    }

    #[test]
    fn read_varint_rejects_truncation() {
        let mut buf = Vec::new();
        push_varint(&mut buf, 300);
        let mut at = 0;
        assert_eq!(read_varint(&buf[..1], &mut at), None);
    }

    #[test]
    fn read_varint_rejects_overflow() {
        // Six continuation bytes exceed a u32's 5-byte maximum.
        let buf = [0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        let mut at = 0;
        assert_eq!(read_varint(&buf, &mut at), None);
    }

    #[test]
    fn run_round_trips() {
        let positions = [3u32, 4, 9, 100, 1_000_000];
        let mut buf = Vec::new();
        encode_run(&mut buf, &positions);
        let mut at = 0;
        let mut got = Vec::new();
        decode_run(&buf, &mut at, positions.len() as u32, &mut got).expect("decode");
        assert_eq!(got, positions);
        assert_eq!(at, buf.len());
    }

    #[test]
    fn runs_concatenate_and_skip() {
        // Two docs' runs back to back; skip the first, decode the second.
        let a = [5u32, 6];
        let b = [0u32, 2, 4];
        let mut buf = Vec::new();
        encode_run(&mut buf, &a);
        let a_end = buf.len();
        encode_run(&mut buf, &b);
        let mut at = 0;
        skip_run(&buf, &mut at, a.len() as u32).expect("skip");
        assert_eq!(at, a_end);
        let mut got = Vec::new();
        decode_run(&buf, &mut at, b.len() as u32, &mut got).expect("decode");
        assert_eq!(got, b);
    }

    #[test]
    fn decode_run_rejects_truncated_tail() {
        let mut buf = Vec::new();
        encode_run(&mut buf, &[1u32, 2, 3]);
        let mut at = 0;
        let mut got = Vec::new();
        assert_eq!(
            decode_run(&buf[..buf.len() - 1], &mut at, 3, &mut got),
            None
        );
    }
}
