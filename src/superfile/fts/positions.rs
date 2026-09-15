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

use crate::superfile::bits::{get_bits, payload_bytes, put_bits, width_of};

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
/// A packed group's first header byte is the first-position stream's
/// width plus this, so that a zero width (every doc's first position is
/// 0) is told apart from [`GROUP_LEB128`].
const GROUP_FIRST_WIDTH_BIAS: u8 = 1;

/// Append one **position group** — the run values (first position
/// absolute per doc, then gaps, in posting order) of one posting block,
/// or of a whole short-form term — behind a short header. `tfs` are the
/// group's per-doc term frequencies, so `values.len() == Σ tfs`.
///
/// A packed group splits the values into two streams, each bit-packed
/// at its own width: every doc's **first** position (bounded by the
/// document length — tens of thousands at most, so ten to fifteen bits)
/// and every **gap** between a doc's positions (a few bits for a term
/// that recurs within a document). Packing them together would let the
/// first positions set the width for every gap and lose most of the
/// saving. Header: `first_width + 1`, `gap_width`, then the two
/// payloads; the reader knows both counts from the tfs. When the LEB128
/// runs come out smaller (an outlier in either stream), the header is
/// the single byte [`GROUP_LEB128`] and the runs follow as before, so no
/// group grows past what it cost before grouping.
///
/// From `VERSION_V7` every group takes this shape; a phrase decode reads
/// a packed group whole and indexes it by the block's tf prefix sums,
/// where a LEB128 group is walked run by run as before.
pub(crate) fn encode_group(out: &mut Vec<u8>, tfs: &[u32], values: &[u32]) {
    debug_assert_eq!(
        tfs.iter().map(|&t| t as usize).sum::<usize>(),
        values.len(),
        "values are the runs of tfs"
    );
    let n_first = tfs.len();
    let n_gap = values.len() - n_first;
    let (mut max_first, mut max_gap) = (0u32, 0u32);
    let mut vi = 0usize;
    for &tf in tfs {
        max_first = max_first.max(values[vi]);
        for &g in &values[vi + 1..vi + tf as usize] {
            max_gap = max_gap.max(g);
        }
        vi += tf as usize;
    }
    let first_width = width_of(max_first.into());
    let gap_width = width_of(max_gap.into());
    let packed_len = 2 + payload_bytes(n_first, first_width) + payload_bytes(n_gap, gap_width);
    let leb_len: usize = 1 + values.iter().map(|&v| varint_len(v)).sum::<usize>();
    if leb_len <= packed_len {
        out.push(GROUP_LEB128);
        for &v in values {
            push_varint(out, v);
        }
        return;
    }
    debug_assert!(first_width <= GROUP_MAX_WIDTH && gap_width <= GROUP_MAX_WIDTH);
    out.push(first_width + GROUP_FIRST_WIDTH_BIAS);
    out.push(gap_width);
    let first_start = out.len();
    let gap_start = first_start + payload_bytes(n_first, first_width);
    out.resize(gap_start + payload_bytes(n_gap, gap_width), 0);
    let (mut fi, mut gi, mut vi) = (0usize, 0usize, 0usize);
    for &tf in tfs {
        put_bits(
            &mut out[first_start..gap_start],
            fi * first_width as usize,
            u64::from(values[vi]),
            first_width,
        );
        fi += 1;
        for &g in &values[vi + 1..vi + tf as usize] {
            put_bits(
                &mut out[gap_start..],
                gi * gap_width as usize,
                u64::from(g),
                gap_width,
            );
            gi += 1;
        }
        vi += tf as usize;
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

/// Decode a whole group starting at `*at` (its header), appending the
/// run values — `Σ tfs` of them, in run order — to `out` and advancing
/// `*at` past the group. `None` on a truncated or malformed group.
pub(crate) fn decode_group(
    bytes: &[u8],
    at: &mut usize,
    tfs: &[u32],
    out: &mut Vec<u32>,
) -> Option<()> {
    let n: usize = tfs.iter().map(|&t| t as usize).sum();
    let header = *bytes.get(*at)?;
    *at += 1;
    if header == GROUP_LEB128 {
        for _ in 0..n {
            out.push(read_varint(bytes, at)?);
        }
        return Some(());
    }
    let first_width = header - GROUP_FIRST_WIDTH_BIAS;
    let gap_width = *bytes.get(*at)?;
    *at += 1;
    if first_width > GROUP_MAX_WIDTH || gap_width > GROUP_MAX_WIDTH {
        return None;
    }
    let n_first = tfs.len();
    let n_gap = n - n_first;
    let first_len = payload_bytes(n_first, first_width);
    let gap_len = payload_bytes(n_gap, gap_width);
    let firsts = bytes.get(*at..*at + first_len)?;
    let gaps = bytes.get(*at + first_len..*at + first_len + gap_len)?;
    out.reserve(n);
    let mut gi = 0usize;
    for (fi, &tf) in tfs.iter().enumerate() {
        out.push(get_bits(firsts, fi, first_width)? as u32);
        for _ in 1..tf {
            out.push(get_bits(gaps, gi, gap_width)? as u32);
            gi += 1;
        }
    }
    *at += first_len + gap_len;
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
        // 100 docs, tf 3 each: first positions up to ~2000 (11 bits), gaps
        // of 1..=20 (5 bits). Packed: 2 + 138 + 125 = 265 B; LEB128 would
        // be 1 + 100 × 2 + 200 × 1 = 401 B. Packed wins, and splitting the
        // streams is what makes it win: at one shared width the gaps would
        // cost 11 bits each.
        let tfs = vec![3u32; 100];
        let mut vals = Vec::new();
        for d in 0..100u32 {
            vals.extend_from_slice(&[20 * d + 7, 1 + d % 20, 3 + d % 17]);
        }
        let mut out = Vec::new();
        encode_group(&mut out, &tfs, &vals);
        assert_ne!(out[0], GROUP_LEB128);
        assert_eq!(
            out.len(),
            2 + payload_bytes(100, 11) + payload_bytes(200, 5)
        );
        let mut at = 0;
        let mut back = Vec::new();
        decode_group(&out, &mut at, &tfs, &mut back).expect("decodes");
        assert_eq!(back, vals);
        assert_eq!(at, out.len());

        // One huge gap widens every gap: LEB128 wins and is chosen.
        let mut outlier = vals.clone();
        outlier[7] = 1 << 30;
        let mut out = Vec::new();
        encode_group(&mut out, &tfs, &outlier);
        assert_eq!(out[0], GROUP_LEB128);
        let mut at = 0;
        let mut back = Vec::new();
        decode_group(&out, &mut at, &tfs, &mut back).expect("decodes");
        assert_eq!(back, outlier);
        assert_eq!(at, out.len());

        // Single posting, tf 1 (no gaps), zero first positions, wide values.
        for (tfs, v) in [
            (vec![1u32], vec![0u32]),
            (vec![1], vec![u32::MAX]),
            (vec![2, 2], vec![0, 5, 0, 7]),
            (vec![4], vec![5, 0, 0, 7]),
            (vec![1; 64], vec![0; 64]),
        ] {
            let mut out = Vec::new();
            encode_group(&mut out, &tfs, &v);
            let mut at = 0;
            let mut back = Vec::new();
            decode_group(&out, &mut at, &tfs, &mut back).expect("decodes");
            assert_eq!(back, v, "tfs {tfs:?}");
            assert_eq!(at, out.len());
        }
        // Truncation is refused, not a panic.
        let mut out = Vec::new();
        encode_group(&mut out, &tfs, &vals);
        for cut in 0..out.len() {
            let mut at = 0;
            assert!(decode_group(&out[..cut], &mut at, &tfs, &mut Vec::new()).is_none());
        }
        assert!(
            decode_group(&[34, 0], &mut 0, &[1], &mut Vec::new()).is_none(),
            "first width past 32"
        );
        assert!(
            decode_group(&[1, 33], &mut 0, &[2], &mut Vec::new()).is_none(),
            "gap width past 32"
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
