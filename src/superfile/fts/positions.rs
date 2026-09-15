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

use std::ops::Range;

use crate::superfile::bits::{get_bits, payload_bytes, put_bits, width_of};

/// Largest byte length one encoded `u32` can occupy (LEB128: 5 × 7
/// bits ≥ 32 bits). Used to reserve scratch capacity.
/// (Consumed by the read path that follows in this series.)
#[allow(dead_code)]
pub(crate) const MAX_VARINT_BYTES: usize = 5;

/// LEB128 continuation flag: high bit set ⇒ another byte follows.
pub(crate) const CONTINUATION_BIT: u8 = 0x80;
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
/// every blob before `VERSION_V7` used for all of its runs; from `V7`
/// only a short-form term's group may still take it).
pub(crate) const GROUP_LEB128: u8 = 0;
/// Group header value for a group stored as two patched streams.
pub(crate) const GROUP_PACKED: u8 = 1;
/// Widest packed position value: a `u32`.
const GROUP_MAX_WIDTH: u8 = 32;
/// Most exception lanes a packed stream may carry; bounds the patch loop.
const GROUP_MAX_EXCEPTIONS: usize = 64;

/// Append one **position group** — the run values (first position
/// absolute per doc, then gaps, in posting order) of one posting block,
/// or of a whole short-form term — behind a one-byte header. `tfs` are
/// the group's per-doc term frequencies, so `values.len() == Σ tfs`.
///
/// A [`GROUP_PACKED`] group splits the values into two **streams**,
/// every doc's **first** position and every **gap** between a doc's
/// positions, because the two have very different ranges (a first
/// position is bounded by the document length, a gap for a recurring
/// term is a few bits). Each stream is bit-packed at the width most of
/// its lanes fit, and the lanes that do not — one long document, one
/// long gap — store their high bits as **exceptions** `(lane, high
/// bits)` so a single outlier never widens the whole stream. Stream
/// layout: `width (u8) | n_exceptions (varint) | packed low bits |
/// exceptions (varint lane, varint high bits)`.
///
/// A packed group can be decoded whole and indexed by the block's tf
/// prefix sums, so a positional term needs no run offsets past its
/// block starts: `V7` carries no position sub-index. That only holds if
/// every long-form group is packed, so `allow_leb128` is `false` for
/// them; a short-form term's group is decoded whole in any case and may
/// take the LEB128 form when that is smaller.
pub(crate) fn encode_group(out: &mut Vec<u8>, tfs: &[u32], values: &[u32], allow_leb128: bool) {
    debug_assert_eq!(
        tfs.iter().map(|&t| t as usize).sum::<usize>(),
        values.len(),
        "values are the runs of tfs"
    );
    let mut firsts: Vec<u32> = Vec::with_capacity(tfs.len());
    let mut gaps: Vec<u32> = Vec::with_capacity(values.len() - tfs.len());
    let mut vi = 0usize;
    for &tf in tfs {
        firsts.push(values[vi]);
        gaps.extend_from_slice(&values[vi + 1..vi + tf as usize]);
        vi += tf as usize;
    }
    let first_plan = plan_stream(&firsts);
    let gap_plan = plan_stream(&gaps);
    let packed_len = 1 + first_plan.bytes + gap_plan.bytes;
    if allow_leb128 {
        let leb_len: usize = 1 + values.iter().map(|&v| varint_len(v)).sum::<usize>();
        if leb_len <= packed_len {
            out.push(GROUP_LEB128);
            for &v in values {
                push_varint(out, v);
            }
            return;
        }
    }
    out.push(GROUP_PACKED);
    write_stream(out, &firsts, &first_plan);
    write_stream(out, &gaps, &gap_plan);
}

/// A stream's packing: the width most lanes fit and the lanes that do
/// not, with their high bits, plus the bytes the whole stream takes.
struct StreamPlan {
    width: u8,
    exceptions: Vec<(u32, u32)>,
    bytes: usize,
}

/// The cheapest `(width, exceptions)` for `lanes`: every width below the
/// plain one is tried and the smallest total kept, subject to
/// [`GROUP_MAX_EXCEPTIONS`].
fn plan_stream(lanes: &[u32]) -> StreamPlan {
    let plain = width_of(lanes.iter().copied().max().unwrap_or(0).into());
    let cost = |width: u8, exceptions: &[(u32, u32)]| -> usize {
        1 + varint_len(exceptions.len() as u32)
            + payload_bytes(lanes.len(), width)
            + exceptions
                .iter()
                .map(|&(lane, hi)| varint_len(lane) + varint_len(hi))
                .sum::<usize>()
    };
    let mut best = StreamPlan {
        width: plain,
        exceptions: Vec::new(),
        bytes: cost(plain, &[]),
    };
    for width in 0..plain {
        let mut exceptions = Vec::new();
        for (i, &v) in lanes.iter().enumerate() {
            let hi = if width == 0 { v } else { v >> width };
            if hi != 0 {
                exceptions.push((i as u32, hi));
                if exceptions.len() > GROUP_MAX_EXCEPTIONS {
                    break;
                }
            }
        }
        if exceptions.len() > GROUP_MAX_EXCEPTIONS {
            continue;
        }
        let bytes = cost(width, &exceptions);
        if bytes < best.bytes {
            best = StreamPlan {
                width,
                exceptions,
                bytes,
            };
        }
    }
    best
}

/// Emit one stream per its plan.
fn write_stream(out: &mut Vec<u8>, lanes: &[u32], plan: &StreamPlan) {
    debug_assert!(plan.width <= GROUP_MAX_WIDTH);
    out.push(plan.width);
    push_varint(out, plan.exceptions.len() as u32);
    let start = out.len();
    out.resize(start + payload_bytes(lanes.len(), plan.width), 0);
    if plan.width > 0 {
        let mask: u64 = (1u64 << plan.width) - 1;
        for (i, &v) in lanes.iter().enumerate() {
            put_bits(
                &mut out[start..],
                i * plan.width as usize,
                u64::from(v) & mask,
                plan.width,
            );
        }
    }
    for &(lane, hi) in &plan.exceptions {
        push_varint(out, lane);
        push_varint(out, hi);
    }
}

/// Decode one stream of `n` lanes at `*at`, appending to `out`.
fn read_stream(bytes: &[u8], at: &mut usize, n: usize, out: &mut Vec<u32>) -> Option<()> {
    let width = *bytes.get(*at)?;
    *at += 1;
    if width > GROUP_MAX_WIDTH {
        return None;
    }
    let n_exc = read_varint(bytes, at)? as usize;
    if n_exc > GROUP_MAX_EXCEPTIONS.max(n) {
        return None;
    }
    let len = payload_bytes(n, width);
    let payload = bytes.get(*at..*at + len)?;
    let base = out.len();
    out.reserve(n);
    for i in 0..n {
        out.push(get_bits(payload, i, width)? as u32);
    }
    *at += len;
    for _ in 0..n_exc {
        let lane = read_varint(bytes, at)? as usize;
        let hi = read_varint(bytes, at)?;
        if lane >= n {
            return None;
        }
        out[base + lane] |= hi.checked_shl(u32::from(width)).unwrap_or(0);
    }
    Some(())
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
    match header {
        GROUP_LEB128 => {
            for _ in 0..n {
                out.push(read_varint(bytes, at)?);
            }
            Some(())
        }
        GROUP_PACKED => {
            let n_first = tfs.len();
            let mut firsts = Vec::with_capacity(n_first);
            let mut gaps = Vec::with_capacity(n - n_first);
            read_stream(bytes, at, n_first, &mut firsts)?;
            read_stream(bytes, at, n - n_first, &mut gaps)?;
            out.reserve(n);
            let mut gi = 0usize;
            for (fi, &tf) in tfs.iter().enumerate() {
                out.push(firsts[fi]);
                let take = tf as usize - 1;
                out.extend_from_slice(&gaps[gi..gi + take]);
                gi += take;
            }
            Some(())
        }
        _ => None,
    }
}

/// One stream of a packed group located for random access: its lane
/// width, where its payload lies in the positions bytes, and its
/// exceptions in ascending lane order.
#[derive(Default)]
struct StreamIndex {
    width: u8,
    payload: Range<usize>,
    exceptions: Vec<(u32, u32)>,
}

impl StreamIndex {
    /// Parse the stream of `n` lanes at `*at` into `self`, reusing its
    /// exception buffer, advancing past it. The exception lanes must
    /// ascend (the writer emits them in lane order), so a run's
    /// exceptions are one binary search away.
    fn parse_into(&mut self, bytes: &[u8], at: &mut usize, n: usize) -> Option<()> {
        let width = *bytes.get(*at)?;
        *at += 1;
        if width > GROUP_MAX_WIDTH {
            return None;
        }
        let n_exc = read_varint(bytes, at)? as usize;
        if n_exc > GROUP_MAX_EXCEPTIONS.max(n) {
            return None;
        }
        let payload = *at..*at + payload_bytes(n, width);
        bytes.get(payload.clone())?;
        *at = payload.end;
        self.exceptions.clear();
        let mut prev_lane: Option<u32> = None;
        for _ in 0..n_exc {
            let lane = read_varint(bytes, at)?;
            let hi = read_varint(bytes, at)?;
            if lane as usize >= n || prev_lane.is_some_and(|p| lane <= p) {
                return None;
            }
            self.exceptions.push((lane, hi));
            prev_lane = Some(lane);
        }
        self.width = width;
        self.payload = payload;
        Some(())
    }

    /// Lanes `from..from + n`, exceptions patched in, appended to `out`.
    fn read_lanes(&self, bytes: &[u8], from: usize, n: usize, out: &mut Vec<u32>) -> Option<()> {
        let payload = &bytes[self.payload.clone()];
        let base = out.len();
        out.reserve(n);
        for i in from..from + n {
            out.push(get_bits(payload, i, self.width)? as u32);
        }
        let first = self
            .exceptions
            .partition_point(|&(lane, _)| (lane as usize) < from);
        for &(lane, hi) in &self.exceptions[first..] {
            let lane = lane as usize;
            if lane >= from + n {
                break;
            }
            out[base + lane - from] |= hi.checked_shl(u32::from(self.width)).unwrap_or(0);
        }
        Some(())
    }
}

/// Which layout the located group has.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum GroupKind {
    /// Decoded whole into `values`: LEB128 runs allow no random access,
    /// and only a short-form term (read once, whole) still writes them.
    #[default]
    Leb128,
    /// Two packed streams read lane by lane.
    Packed,
}

/// A position group located for **per-run** access: one pair's run is
/// decoded without touching the block's other runs. A phrase visits few
/// of a common term's pairs per block when its other members are rare,
/// so decoding a whole group — every position of 128 docs — per
/// candidate was the cost that dominated; here a candidate costs its
/// own `tf` lanes plus one binary search over the stream's exceptions.
/// One instance is reused across blocks so relocating allocates nothing.
#[derive(Default)]
pub(crate) struct GroupIndex {
    kind: GroupKind,
    /// A LEB128 group's values, whole.
    values: Vec<u32>,
    /// A packed group's first-position and gap streams.
    first: StreamIndex,
    gap: StreamIndex,
    /// Where pair `p`'s lanes begin: in the gap stream (`Σ (tf - 1)` over
    /// the pairs before it) for a packed group, in the values (`Σ tf`)
    /// for a LEB128 one. `tfs.len() + 1` entries.
    starts: Vec<u32>,
    /// A run's raw values before the gaps are summed into positions.
    run: Vec<u32>,
}

impl GroupIndex {
    /// Locate the group at `*at` (its header) for the block whose per-doc
    /// term frequencies are `tfs`, advancing `*at` past it. `None` on a
    /// truncated or malformed group, after which the index must be
    /// relocated before use.
    pub(crate) fn locate(&mut self, bytes: &[u8], at: &mut usize, tfs: &[u32]) -> Option<()> {
        let n: usize = tfs.iter().map(|&t| t as usize).sum();
        let header = *bytes.get(*at)?;
        *at += 1;
        let lanes_per_pair: fn(u32) -> u32 = match header {
            GROUP_LEB128 => {
                self.kind = GroupKind::Leb128;
                self.values.clear();
                self.values.reserve(n);
                for _ in 0..n {
                    self.values.push(read_varint(bytes, at)?);
                }
                |tf| tf
            }
            GROUP_PACKED => {
                self.kind = GroupKind::Packed;
                self.first.parse_into(bytes, at, tfs.len())?;
                self.gap.parse_into(bytes, at, n - tfs.len())?;
                |tf| tf.saturating_sub(1)
            }
            _ => return None,
        };
        self.starts.clear();
        self.starts.reserve(tfs.len() + 1);
        let mut acc = 0u32;
        self.starts.push(acc);
        for &tf in tfs {
            acc = acc.checked_add(lanes_per_pair(tf))?;
            self.starts.push(acc);
        }
        Some(())
    }

    /// The absolute positions of pair `pair` (whose term frequency is
    /// `tf`), appended to `out`. `None` on an overflowing gap.
    pub(crate) fn run_positions(
        &mut self,
        bytes: &[u8],
        pair: usize,
        tf: u32,
        out: &mut Vec<u32>,
    ) -> Option<()> {
        let start = *self.starts.get(pair)? as usize;
        self.run.clear();
        match self.kind {
            GroupKind::Leb128 => {
                self.run
                    .extend_from_slice(self.values.get(start..start + tf as usize)?);
            }
            GroupKind::Packed => {
                self.first.read_lanes(bytes, pair, 1, &mut self.run)?;
                self.gap
                    .read_lanes(bytes, start, tf as usize - 1, &mut self.run)?;
            }
        }
        positions_from_run_values(&self.run, out)
    }
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
    fn a_group_packs_two_streams_with_exceptions_and_round_trips() {
        // 100 docs, tf 3 each: first positions up to ~2000 (11 bits), gaps
        // of 1..=20 (5 bits), plus one doc far out (an exception in the
        // first stream) and one huge gap (an exception in the gap
        // stream). Neither outlier widens its stream.
        let tfs = vec![3u32; 100];
        let mut vals = Vec::new();
        for d in 0..100u32 {
            vals.extend_from_slice(&[20 * d + 7, 1 + d % 20, 3 + d % 17]);
        }
        vals[3 * 41] = 1 << 24;
        vals[3 * 77 + 2] = 1 << 30;
        let mut out = Vec::new();
        encode_group(&mut out, &tfs, &vals, true);
        assert_eq!(out[0], GROUP_PACKED);
        // firsts: 1 + 1 + 138 (11 bits) + one exception; gaps: 1 + 1 + 125 (5 bits) + one exception.
        assert!(out.len() < 1 + 145 + 133, "got {} bytes", out.len());
        let mut at = 0;
        let mut back = Vec::new();
        decode_group(&out, &mut at, &tfs, &mut back).expect("decodes");
        assert_eq!(back, vals);
        assert_eq!(at, out.len());

        // A short term with a lone value: LEB128 is smaller and allowed.
        let mut out = Vec::new();
        encode_group(&mut out, &[1], &[5], true);
        assert_eq!(out[0], GROUP_LEB128);
        // The same values with LEB128 disallowed pack anyway.
        let mut out = Vec::new();
        encode_group(&mut out, &[1], &[5], false);
        assert_eq!(out[0], GROUP_PACKED);
        let mut back = Vec::new();
        decode_group(&out, &mut 0, &[1], &mut back).expect("decodes");
        assert_eq!(back, vec![5]);

        // Edge shapes: tf 1 everywhere (no gaps), zero first positions, the
        // top of the range, a stream that is all exceptions but one.
        for (tfs, v) in [
            (vec![1u32], vec![0u32]),
            (vec![1], vec![u32::MAX]),
            (vec![2, 2], vec![0, 5, 0, 7]),
            (vec![4], vec![5, 0, 0, 7]),
            (vec![1; 64], vec![0; 64]),
            (vec![1; 5], vec![1, 1 << 20, 1, 1 << 31, 1]),
        ] {
            for allow in [true, false] {
                let mut out = Vec::new();
                encode_group(&mut out, &tfs, &v, allow);
                let mut at = 0;
                let mut back = Vec::new();
                decode_group(&out, &mut at, &tfs, &mut back).expect("decodes");
                assert_eq!(back, v, "tfs {tfs:?} allow {allow}");
                assert_eq!(at, out.len());
            }
        }
        // Truncation and bad headers are refused, not a panic.
        let mut out = Vec::new();
        encode_group(&mut out, &tfs, &vals, false);
        for cut in 0..out.len() {
            let mut at = 0;
            assert!(decode_group(&out[..cut], &mut at, &tfs, &mut Vec::new()).is_none());
        }
        assert!(
            decode_group(&[2], &mut 0, &[1], &mut Vec::new()).is_none(),
            "unknown header"
        );
        assert!(
            decode_group(&[GROUP_PACKED, 33, 0], &mut 0, &[1], &mut Vec::new()).is_none(),
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
    fn a_group_index_reads_each_run_like_the_whole_decode() {
        // Mixed tfs, exceptions in both streams, and the first pair.
        let tfs: Vec<u32> = (0..100u32).map(|d| 1 + d % 5).collect();
        let mut vals = Vec::new();
        for (d, &tf) in tfs.iter().enumerate() {
            vals.push(30 * d as u32 + 3);
            vals.extend((1..tf).map(|g| 1 + (g + d as u32) % 9));
        }
        vals[0] = 1 << 25;
        let last = vals.len() - 1;
        vals[last] = 1 << 28;
        for allow in [false, true] {
            let mut out = vec![0xAA; 7];
            let start = out.len();
            encode_group(&mut out, &tfs, &vals, allow);
            let mut at = start;
            let mut index = GroupIndex::default();
            index.locate(&out, &mut at, &tfs).expect("parses");
            assert_eq!(at, out.len());
            let mut whole = Vec::new();
            decode_group(&out, &mut start.clone(), &tfs, &mut whole).expect("decodes");
            let mut vi = 0usize;
            for (pair, &tf) in tfs.iter().enumerate() {
                let mut want = Vec::new();
                positions_from_run_values(&whole[vi..vi + tf as usize], &mut want).expect("sums");
                vi += tf as usize;
                let mut got = Vec::new();
                index
                    .run_positions(&out, pair, tf, &mut got)
                    .expect("run decodes");
                assert_eq!(got, want, "pair {pair} allow {allow}");
            }
            assert!(
                index
                    .run_positions(&out, tfs.len(), 1, &mut Vec::new())
                    .is_none()
            );
            for cut in start..out.len() {
                assert!(
                    GroupIndex::default()
                        .locate(&out[..cut], &mut start.clone(), &tfs)
                        .is_none()
                );
            }
        }
        // A LEB128 group must still be allowed to have a run overflow refused.
        let mut out = Vec::new();
        encode_group(&mut out, &[2], &[u32::MAX, 1], true);
        let mut index = GroupIndex::default();
        index.locate(&out, &mut 0, &[2]).expect("parses");
        assert!(index.run_positions(&out, 0, 2, &mut Vec::new()).is_none());
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
