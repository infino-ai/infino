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

use crate::{
    superfile::bits::{
        ExceptionPlan, PackScratch, for_each_lane, get_bits, payload_bytes, plan_exceptions,
        put_bits,
    },
    utils::varint::{push_varint, read_varint, varint_len},
};

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

/// A term's position runs in posting order, as the encoder takes them.
/// A run is a posting's first position, then the gaps between the rest.
#[derive(Clone, Copy)]
pub(crate) enum TermRuns<'a> {
    /// LEB128 runs back to back, as the build accumulator holds them.
    Encoded(&'a [u8]),
    /// Run values, as the compaction merge decodes them: posting `i`'s
    /// `tf` values start at `starts[i]`. Keeping them decoded saves the
    /// merge a LEB128 round trip per position.
    Values {
        values: &'a [u32],
        starts: &'a [usize],
    },
}

impl TermRuns<'_> {
    /// Append posting `i`'s run of `tf` values to `out`. `at` walks the
    /// encoded form and must start at 0 for posting 0; the decoded form
    /// ignores it.
    #[inline]
    pub(crate) fn push_run(&self, i: usize, tf: u32, at: &mut usize, out: &mut Vec<u32>) {
        match *self {
            Self::Encoded(bytes) => {
                for _ in 0..tf {
                    out.push(read_varint(bytes, at).expect("builder-encoded run"));
                }
            }
            Self::Values { values, starts } => {
                let start = starts[i];
                out.extend_from_slice(&values[start..start + tf as usize]);
            }
        }
    }

    /// The runs' LEB128 bytes, when that is the form they are in.
    pub(crate) fn encoded(&self) -> Option<&[u8]> {
        match *self {
            Self::Encoded(bytes) => Some(bytes),
            Self::Values { .. } => None,
        }
    }

    /// The first posting's first value.
    pub(crate) fn first_value(&self) -> u32 {
        match *self {
            Self::Encoded(bytes) => read_varint(bytes, &mut 0).expect("builder-encoded run"),
            Self::Values { values, starts } => values[starts[0]],
        }
    }

    /// The runs as LEB128 bytes, which the layouts before grouped
    /// positions store verbatim. `tfs` gives each posting's run length.
    pub(crate) fn encode_into(&self, tfs: impl Iterator<Item = u32>, out: &mut Vec<u8>) {
        match *self {
            Self::Encoded(bytes) => out.extend_from_slice(bytes),
            Self::Values { values, starts } => {
                for (i, tf) in tfs.enumerate() {
                    for &v in &values[starts[i]..starts[i] + tf as usize] {
                        push_varint(out, v);
                    }
                }
            }
        }
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
pub(crate) fn encode_group(
    out: &mut Vec<u8>,
    tfs: &[u32],
    values: &[u32],
    allow_leb128: bool,
    scratch: &mut PackScratch,
) {
    debug_assert_eq!(
        tfs.iter().map(|&t| t as usize).sum::<usize>(),
        values.len(),
        "values are the runs of tfs"
    );
    let firsts = &mut scratch.lanes_a;
    let gaps = &mut scratch.lanes_b;
    firsts.clear();
    gaps.clear();
    let mut vi = 0usize;
    for &tf in tfs {
        firsts.push(values[vi]);
        gaps.extend_from_slice(&values[vi + 1..vi + tf as usize]);
        vi += tf as usize;
    }
    plan_stream(firsts, &mut scratch.plan_a, &mut scratch.candidates);
    plan_stream(gaps, &mut scratch.plan_b, &mut scratch.candidates);
    let (first_plan, gap_plan) = (&scratch.plan_a, &scratch.plan_b);
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
    write_stream(out, firsts, first_plan);
    write_stream(out, gaps, gap_plan);
}

/// The cheapest `(width, exceptions)` for one stream: a byte of width, a
/// byte of exception count, the packed lanes, and each exception's lane
/// and high bits as varints.
fn plan_stream(lanes: &[u32], plan: &mut ExceptionPlan, candidates: &mut Vec<(u32, u32)>) {
    plan_exceptions(
        lanes,
        GROUP_MAX_EXCEPTIONS,
        |width| 2 + payload_bytes(lanes.len(), width),
        |lane, hi| varint_len(lane) + varint_len(hi),
        plan,
        candidates,
    );
}

/// Emit one stream per its plan.
fn write_stream(out: &mut Vec<u8>, lanes: &[u32], plan: &ExceptionPlan) {
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
    /// One lane of the stream, exception applied.
    #[inline]
    fn lane(&self, bytes: &[u8], i: usize) -> Option<u32> {
        let low = u32::try_from(get_bits(&bytes[self.payload.clone()], i, self.width)?).ok()?;
        if self.exceptions.is_empty() {
            return Some(low);
        }
        let hi = match self
            .exceptions
            .binary_search_by_key(&(i as u32), |&(lane, _)| lane)
        {
            Ok(k) => self.exception_hi(self.exceptions[k].1),
            Err(_) => 0,
        };
        Some(low | hi)
    }

    /// An exception's high bits placed above the stream's packed width.
    #[inline]
    fn exception_hi(&self, hi: u32) -> u32 {
        hi.checked_shl(u32::from(self.width)).unwrap_or(0)
    }

    fn read_lanes(&self, bytes: &[u8], from: usize, n: usize, out: &mut Vec<u32>) -> Option<()> {
        let payload = &bytes[self.payload.clone()];
        let base = out.len();
        out.reserve(n);
        for_each_lane(payload, from, n, self.width, |v| out.push(v as u32))?;
        if self.exceptions.is_empty() {
            return Some(());
        }
        let first = self
            .exceptions
            .partition_point(|&(lane, _)| (lane as usize) < from);
        for &(lane, hi) in &self.exceptions[first..] {
            let lane = lane as usize;
            if lane >= from + n {
                break;
            }
            out[base + lane - from] |= self.exception_hi(hi);
        }
        Some(())
    }
}

/// Which layout the located group has.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
enum GroupKind {
    /// Decoded whole into `values`: LEB128 runs allow no random access,
    /// and only a short-form term (read once, whole) still writes them.
    #[default]
    Leb128,
    /// Two packed streams read lane by lane, one run at a time.
    Packed,
    /// The packed streams read whole into `firsts` and `gaps` — the block
    /// is being visited densely, and one sequential pass over each stream
    /// is cheaper than a random read per run.
    Bulk,
}

/// Runs read one at a time from a packed group before the block may be
/// decoded in bulk, and the density that then triggers it: at least one
/// run in [`BULK_MAX_STRIDE`] pairs consumed so far. A phrase driven by
/// a rare member asks for one or two runs per block of a common member,
/// scattered, and never reaches this; a phrase of common members asks
/// for most of them in order, and paid two to three times the
/// sequential decode's cost per run before this switch. The density
/// test keeps a block with a handful of scattered candidates on the
/// per-run path, where decoding all 128 runs would cost more than they
/// do.
const BULK_AFTER_RUNS: u32 = 2;
/// Runs up to this many positions bypass the generic lane reader.
const SMALL_TF: usize = 4;
const BULK_MAX_STRIDE: usize = 4;

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
    /// The streams decoded whole once the block proves dense.
    firsts: Vec<u32>,
    gaps: Vec<u32>,
    /// Runs served from this block so far.
    served: u32,
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
        self.served = 0;
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
        if self.kind == GroupKind::Packed
            && self.served >= BULK_AFTER_RUNS
            && pair < self.served as usize * BULK_MAX_STRIDE
        {
            self.firsts.clear();
            self.gaps.clear();
            let n_pairs = self.starts.len() - 1;
            let n_gaps = *self.starts.last()? as usize;
            self.first.read_lanes(bytes, 0, n_pairs, &mut self.firsts)?;
            self.gap.read_lanes(bytes, 0, n_gaps, &mut self.gaps)?;
            self.kind = GroupKind::Bulk;
        }
        self.served += 1;
        self.run.clear();
        match self.kind {
            GroupKind::Leb128 => {
                self.run
                    .extend_from_slice(self.values.get(start..start + tf as usize)?);
            }
            GroupKind::Packed => {
                if tf as usize <= SMALL_TF {
                    // Most runs on a real corpus are one to three positions:
                    // read the lanes straight out of the two payloads and
                    // write the prefix sums, skipping the generic lane
                    // reader and its scratch. An exception is looked up per
                    // lane only when the stream has any.
                    let mut p = self.first.lane(bytes, pair)?;
                    out.push(p);
                    for k in 0..tf as usize - 1 {
                        p = p.checked_add(self.gap.lane(bytes, start + k)?)?;
                        out.push(p);
                    }
                    return Some(());
                }
                self.first.read_lanes(bytes, pair, 1, &mut self.run)?;
                self.gap
                    .read_lanes(bytes, start, tf as usize - 1, &mut self.run)?;
            }
            GroupKind::Bulk => {
                self.run.push(*self.firsts.get(pair)?);
                self.run
                    .extend_from_slice(self.gaps.get(start..start + tf as usize - 1)?);
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

    /// The decoded form of a term's runs reads, and re-encodes, exactly as
    /// the LEB128 form does, whatever order the postings' runs sit in.
    #[test]
    fn decoded_runs_agree_with_encoded_runs() {
        // Three postings' positions, and the order they are emitted in.
        let docs: [&[u32]; 3] = [&[4, 9, 300], &[0], &[7, 200_000]];
        let emitted = [2usize, 0, 1];
        let mut values = Vec::new();
        let mut starts = vec![0; docs.len()];
        for (i, positions) in docs.iter().enumerate() {
            starts[i] = values.len();
            let mut prev = 0;
            for &p in *positions {
                values.push(p - prev);
                prev = p;
            }
        }
        let mut bytes = Vec::new();
        for &i in &emitted {
            encode_run(&mut bytes, docs[i]);
        }
        let emitted_starts: Vec<usize> = emitted.iter().map(|&i| starts[i]).collect();
        let decoded = TermRuns::Values {
            values: &values,
            starts: &emitted_starts,
        };
        let encoded = TermRuns::Encoded(&bytes);
        let tfs: Vec<u32> = emitted.iter().map(|&i| docs[i].len() as u32).collect();

        let (mut from_decoded, mut from_encoded) = (Vec::new(), Vec::new());
        let (mut at_decoded, mut at_encoded) = (0, 0);
        for (i, &tf) in tfs.iter().enumerate() {
            decoded.push_run(i, tf, &mut at_decoded, &mut from_decoded);
            encoded.push_run(i, tf, &mut at_encoded, &mut from_encoded);
        }
        assert_eq!(from_decoded, from_encoded, "run values");
        assert_eq!(at_encoded, bytes.len(), "the encoded walk covers every run");
        assert_eq!(decoded.first_value(), encoded.first_value(), "first value");

        let mut reencoded = Vec::new();
        decoded.encode_into(tfs.iter().copied(), &mut reencoded);
        assert_eq!(reencoded, bytes, "re-encoded runs");
        assert_eq!(decoded.encoded(), None);
        assert_eq!(encoded.encoded(), Some(bytes.as_slice()));
    }

    /// Encode a group with fresh scratch.
    fn group(out: &mut Vec<u8>, tfs: &[u32], values: &[u32], allow_leb128: bool) {
        encode_group(out, tfs, values, allow_leb128, &mut PackScratch::default());
    }

    /// Decode every pair of a group through the per-run index into
    /// absolute positions, checking the index consumed the whole group.
    fn decode_all(bytes: &[u8], start: usize, tfs: &[u32]) -> Option<Vec<u32>> {
        let mut at = start;
        let mut index = GroupIndex::default();
        index.locate(bytes, &mut at, tfs)?;
        assert_eq!(at, bytes.len());
        let mut out = Vec::new();
        for (pair, &tf) in tfs.iter().enumerate() {
            index.run_positions(bytes, pair, tf, &mut out)?;
        }
        Some(out)
    }

    /// The positions `vals` (runs of first-then-gaps) stand for.
    fn positions_of(tfs: &[u32], vals: &[u32]) -> Vec<u32> {
        let mut out = Vec::new();
        let mut vi = 0;
        for &tf in tfs {
            positions_from_run_values(&vals[vi..vi + tf as usize], &mut out).expect("sums");
            vi += tf as usize;
        }
        out
    }

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
        vals[3 * 77 + 2] = 1 << 20;
        let mut out = Vec::new();
        group(&mut out, &tfs, &vals, true);
        assert_eq!(out[0], GROUP_PACKED);
        // firsts: 1 + 1 + 138 (11 bits) + one exception; gaps: 1 + 1 + 125 (5 bits) + one exception.
        assert!(out.len() < 1 + 145 + 133, "got {} bytes", out.len());
        assert_eq!(decode_all(&out, 0, &tfs), Some(positions_of(&tfs, &vals)));

        // A short term with a lone value: LEB128 is smaller and allowed.
        let mut out = Vec::new();
        group(&mut out, &[1], &[5], true);
        assert_eq!(out[0], GROUP_LEB128);
        // The same values with LEB128 disallowed pack anyway.
        let mut out = Vec::new();
        group(&mut out, &[1], &[5], false);
        assert_eq!(out[0], GROUP_PACKED);
        assert_eq!(decode_all(&out, 0, &[1]), Some(vec![5]));

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
                group(&mut out, &tfs, &v, allow);
                assert_eq!(
                    decode_all(&out, 0, &tfs),
                    Some(positions_of(&tfs, &v)),
                    "tfs {tfs:?} allow {allow}"
                );
            }
        }
        // Truncation and bad headers are refused, not a panic.
        let mut out = Vec::new();
        group(&mut out, &tfs, &vals, false);
        for cut in 0..out.len() {
            let mut index = GroupIndex::default();
            assert!(index.locate(&out[..cut], &mut 0, &tfs).is_none());
        }
        assert!(
            GroupIndex::default().locate(&[2], &mut 0, &[1]).is_none(),
            "unknown header"
        );
        assert!(
            GroupIndex::default()
                .locate(&[GROUP_PACKED, 33, 0], &mut 0, &[1])
                .is_none(),
            "width past 32"
        );
    }

    #[test]
    fn a_dense_walk_switches_to_the_bulk_decode_and_agrees() {
        // Reading every pair of a packed group in order crosses the bulk
        // threshold after a few runs; the answers before and after the
        // switch are the same positions, and a fresh index reading pairs
        // out of order (never dense) agrees too.
        let tfs: Vec<u32> = (0..128u32).map(|d| 1 + d % 4).collect();
        let mut vals = Vec::new();
        for (d, &tf) in tfs.iter().enumerate() {
            vals.push(11 * d as u32 + 2);
            vals.extend((1..tf).map(|g| 2 + (g * 7 + d as u32) % 13));
        }
        vals[4] = 1 << 22;
        let want = positions_of(&tfs, &vals);
        let mut out = Vec::new();
        group(&mut out, &tfs, &vals, false);
        let mut dense = GroupIndex::default();
        dense.locate(&out, &mut 0, &tfs).expect("locates");
        assert_eq!(dense.kind, GroupKind::Packed);
        let mut got = Vec::new();
        for (pair, &tf) in tfs.iter().enumerate() {
            dense.run_positions(&out, pair, tf, &mut got).expect("run");
            if pair as u32 > BULK_AFTER_RUNS {
                assert_eq!(dense.kind, GroupKind::Bulk, "pair {pair}");
            }
        }
        assert_eq!(got, want);
        let mut sparse = GroupIndex::default();
        sparse.locate(&out, &mut 0, &tfs).expect("locates");
        let mut wi: Vec<usize> = std::iter::once(0)
            .chain(tfs.iter().scan(0usize, |a, &t| {
                *a += t as usize;
                Some(*a)
            }))
            .collect();
        wi.pop();
        for pair in [127usize, 0, 64] {
            let mut one = Vec::new();
            sparse
                .run_positions(&out, pair, tfs[pair], &mut one)
                .expect("run");
            assert_eq!(
                one,
                want[wi[pair]..wi[pair] + tfs[pair] as usize],
                "pair {pair}"
            );
        }
        // Relocating resets the count: the next block starts per-run.
        dense.locate(&out, &mut 0, &tfs).expect("relocates");
        assert_eq!(dense.kind, GroupKind::Packed);
        // Scattered candidates — one pair in forty — never trip the bulk
        // decode, however many of them a block has.
        let mut scattered = GroupIndex::default();
        scattered.locate(&out, &mut 0, &tfs).expect("locates");
        for pair in [0usize, 40, 80, 120] {
            scattered
                .run_positions(&out, pair, tfs[pair], &mut Vec::new())
                .expect("run");
            assert_eq!(scattered.kind, GroupKind::Packed, "pair {pair}");
        }
    }

    #[test]
    fn short_runs_read_directly_agree_with_the_lane_reader() {
        // Runs of one to four positions take the direct path; the same
        // runs with exceptions in the first stream, in the gap stream, and
        // in both must read the same as the generic reader, including a
        // run that is itself the exception.
        let tfs: Vec<u32> = (0..64u32).map(|d| 1 + d % 4).collect();
        let mut vals = Vec::new();
        for (d, &tf) in tfs.iter().enumerate() {
            vals.push(12 * d as u32 + 5);
            vals.extend((1..tf).map(|g| 2 + (g + d as u32) % 7));
        }
        let base = positions_of(&tfs, &vals);
        let first_of = |pair: usize| tfs[..pair].iter().map(|&t| t as usize).sum::<usize>();
        let mut variants: Vec<(&str, Vec<u32>)> = vec![("no exceptions", vals.clone())];
        let mut v = vals.clone();
        v[first_of(9)] = 1 << 26; // tf-2 pair, first value is an exception
        variants.push(("first stream", v));
        let mut v = vals.clone();
        v[first_of(14) + 2] = 1 << 22; // tf-3 pair, a gap is an exception
        variants.push(("gap stream", v));
        let mut v = vals.clone();
        v[first_of(21)] = 1 << 27;
        v[first_of(21) + 1] = 1 << 21; // tf-2 pair, both lanes exceptional
        v[first_of(0)] = 1 << 25; // the first pair of the block too
        variants.push(("both streams", v));
        for (label, vals) in variants {
            let want = if label == "no exceptions" {
                base.clone()
            } else {
                positions_of(&tfs, &vals)
            };
            let mut out = Vec::new();
            group(&mut out, &tfs, &vals, false);
            assert_eq!(out[0], GROUP_PACKED, "{label}");
            let mut index = GroupIndex::default();
            index.locate(&out, &mut 0, &tfs).expect("parses");
            let mut wi = 0usize;
            for (pair, &tf) in tfs.iter().enumerate() {
                let mut got = Vec::new();
                index
                    .run_positions(&out, pair, tf, &mut got)
                    .expect("run decodes");
                assert_eq!(got, want[wi..wi + tf as usize], "{label} pair {pair}");
                wi += tf as usize;
            }
        }
    }

    #[test]
    fn a_group_index_reads_each_run_on_its_own() {
        // Mixed tfs, exceptions in both streams, and the first pair; a
        // pair read alone equals its slice of the whole decode, and a
        // pair past the block or an overflowing gap is refused.
        let tfs: Vec<u32> = (0..100u32).map(|d| 1 + d % 5).collect();
        let mut vals = Vec::new();
        for (d, &tf) in tfs.iter().enumerate() {
            vals.push(30 * d as u32 + 3);
            vals.extend((1..tf).map(|g| 1 + (g + d as u32) % 9));
        }
        vals[0] = 1 << 25;
        let last = vals.len() - 1;
        vals[last] = 1 << 28;
        let want = positions_of(&tfs, &vals);
        for allow in [false, true] {
            let mut out = vec![0xAA; 7];
            let start = out.len();
            group(&mut out, &tfs, &vals, allow);
            let mut at = start;
            let mut index = GroupIndex::default();
            index.locate(&out, &mut at, &tfs).expect("parses");
            assert_eq!(at, out.len());
            let mut wi = 0usize;
            for (pair, &tf) in tfs.iter().enumerate() {
                let mut got = Vec::new();
                index
                    .run_positions(&out, pair, tf, &mut got)
                    .expect("run decodes");
                assert_eq!(got, want[wi..wi + tf as usize], "pair {pair} allow {allow}");
                wi += tf as usize;
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
        let mut out = Vec::new();
        group(&mut out, &[2], &[u32::MAX, 1], true);
        let mut index = GroupIndex::default();
        index.locate(&out, &mut 0, &[2]).expect("parses");
        assert!(index.run_positions(&out, 0, 2, &mut Vec::new()).is_none());
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
