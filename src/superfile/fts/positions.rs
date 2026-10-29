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
//! table records each posting block's starting byte so a block's runs
//! are randomly addressable without decoding its predecessors.

use std::ops::Range;

use crate::{
    superfile::bits::{
        ExceptionPlan, PackScratch, get_bits, payload_bytes, plan_exceptions, put_bits,
        unpack_lanes_u32,
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
}

/// Decode one run of exactly `tf` positions from `bytes` at `*at`,
/// appending the absolute positions to `out` and advancing `*at`.
/// `None` on corrupt (truncated / overflowing) bytes.
#[cfg(test)]
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
/// Group header value for a group stored as two patched streams whose
/// exceptions are varints (every blob before `VERSION_V10`).
pub(crate) const GROUP_PACKED: u8 = 1;
/// Group header value for a group stored as two patched streams whose
/// exception tables are fixed-width little-endian arrays, which a reader
/// takes in place instead of decoding a varint pair per exception on every
/// group it locates. Each table's field width (1, 2 or 4 bytes) is the
/// narrowest its values fit, so the tables take about what the varints did.
/// The streams and their lanes are those of [`GROUP_PACKED`]. Written from
/// `VERSION_V10`.
pub(crate) const GROUP_PACKED_FIXED: u8 = 2;

/// Bytes a little-endian field needs to hold `v`: 1, 2 or 4.
fn field_bytes(v: u32) -> usize {
    match v {
        0..=0xFF => 1,
        0x100..=0xFFFF => 2,
        _ => 4,
    }
}

/// The exception tables' leading byte: the lane field's byte width in the
/// low nibble, the high-bits field's in the high one.
const EXC_LANE_BYTES_MASK: u8 = 0xF;
const EXC_HI_BYTES_SHIFT: u32 = 4;

/// Field `i` of a table of `W`-byte little-endian fields.
#[inline(always)]
fn field<const W: usize>(table: &[u8], i: usize) -> u32 {
    let f: [u8; W] = table[i * W..i * W + W]
        .try_into()
        .expect("invariant: a W-byte field");
    let mut word = [0u8; 4];
    word[..W].copy_from_slice(&f);
    u32::from_le_bytes(word)
}

/// Both exception tables of one stream read into `out`, `n_exc` lanes of
/// `LB` bytes then as many high bits of `HB` bytes, the widths constant
/// so each field is one load. `None` when the tables run past `bytes`.
fn read_tables<const LB: usize, const HB: usize>(
    bytes: &[u8],
    at: usize,
    n_exc: usize,
    out: &mut Vec<(u32, u32)>,
) -> Option<usize> {
    let lanes = bytes.get(at..at + n_exc * LB)?;
    let his = bytes.get(at + n_exc * LB..at + n_exc * (LB + HB))?;
    out.extend((0..n_exc).map(|e| (field::<LB>(lanes, e), field::<HB>(his, e))));
    Some(at + n_exc * (LB + HB))
}
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
/// layout: `width (u8) | n_exceptions (u8) | packed low bits |` and, with
/// any exceptions, `field widths (u8: lane bytes, high-bits bytes in the
/// high nibble) | exception lanes | exception high bits`, each table one
/// little-endian field per exception
/// ([`GROUP_PACKED_FIXED`]; the [`GROUP_PACKED`] groups of older blobs
/// carry a varint count and varint `(lane, high bits)` pairs instead).
///
/// A packed group can be decoded whole and indexed by the block's tf
/// prefix sums, so a positional term needs no run offsets past its
/// block starts. That only holds if
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
    encode_group_as(out, tfs, values, allow_leb128, scratch, GROUP_PACKED_FIXED);
}

/// [`encode_group`] with the packed form `packed` ([`GROUP_PACKED`] or
/// [`GROUP_PACKED_FIXED`]); the tests write both, as blobs of either
/// version must read.
fn encode_group_as(
    out: &mut Vec<u8>,
    tfs: &[u32],
    values: &[u32],
    allow_leb128: bool,
    scratch: &mut PackScratch,
    packed: u8,
) {
    let fixed = packed == GROUP_PACKED_FIXED;
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
    plan_stream(firsts, fixed, &mut scratch.plan_a, &mut scratch.candidates);
    plan_stream(gaps, fixed, &mut scratch.plan_b, &mut scratch.candidates);
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
    out.push(packed);
    write_stream(out, firsts, first_plan, fixed);
    write_stream(out, gaps, gap_plan, fixed);
}

/// The cheapest `(width, exceptions)` for one stream: a byte of width, a
/// byte of exception count, the packed lanes, and each exception's lane
/// and high bits, fixed-width or as varints.
fn plan_stream(
    lanes: &[u32],
    fixed: bool,
    plan: &mut ExceptionPlan,
    candidates: &mut Vec<(u32, u32)>,
) {
    plan_exceptions(
        lanes,
        GROUP_MAX_EXCEPTIONS,
        |width| 2 + payload_bytes(lanes.len(), width),
        |lane, hi| match fixed {
            true => field_bytes(lane) + field_bytes(hi),
            false => varint_len(lane) + varint_len(hi),
        },
        plan,
        candidates,
    );
}

/// Emit one stream per its plan.
fn write_stream(out: &mut Vec<u8>, lanes: &[u32], plan: &ExceptionPlan, fixed: bool) {
    debug_assert!(plan.width <= GROUP_MAX_WIDTH);
    debug_assert!(plan.exceptions.len() <= GROUP_MAX_EXCEPTIONS);
    out.push(plan.width);
    match fixed {
        true => out.push(plan.exceptions.len() as u8),
        false => push_varint(out, plan.exceptions.len() as u32),
    }
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
    if fixed {
        let Some(&(last_lane, _)) = plan.exceptions.last() else {
            return;
        };
        // Lanes ascend, so the last is the widest.
        let lane_bytes = field_bytes(last_lane);
        let hi_bytes = plan
            .exceptions
            .iter()
            .map(|&(_, hi)| field_bytes(hi))
            .max()
            .unwrap_or(1);
        out.push((lane_bytes as u8) | ((hi_bytes as u8) << EXC_HI_BYTES_SHIFT));
        for &(lane, _) in &plan.exceptions {
            out.extend_from_slice(&lane.to_le_bytes()[..lane_bytes]);
        }
        for &(_, hi) in &plan.exceptions {
            out.extend_from_slice(&hi.to_le_bytes()[..hi_bytes]);
        }
        return;
    }
    for &(lane, hi) in &plan.exceptions {
        push_varint(out, lane);
        push_varint(out, hi);
    }
}

/// One stream of a packed group located for random access: its lane
/// width and count, where its payload lies in the positions bytes, and
/// its exceptions in ascending lane order.
#[derive(Default)]
struct StreamIndex {
    width: u8,
    lanes: usize,
    payload: Range<usize>,
    exceptions: Vec<(u32, u32)>,
}

impl StreamIndex {
    /// Parse the stream of `n` lanes at `*at` into `self`, reusing its
    /// exception buffer, advancing past it. The exception lanes must
    /// ascend (the writer emits them in lane order), so a run's
    /// exceptions are one binary search away.
    fn parse_into(&mut self, bytes: &[u8], at: &mut usize, n: usize, fixed: bool) -> Option<()> {
        let width = *bytes.get(*at)?;
        *at += 1;
        if width > GROUP_MAX_WIDTH {
            return None;
        }
        let n_exc = match fixed {
            true => {
                let count = usize::from(*bytes.get(*at)?);
                *at += 1;
                count
            }
            false => read_varint(bytes, at)? as usize,
        };
        if n_exc > GROUP_MAX_EXCEPTIONS.max(n) {
            return None;
        }
        let payload = *at..*at + payload_bytes(n, width);
        bytes.get(payload.clone())?;
        *at = payload.end;
        self.exceptions.clear();
        let tables = &mut self.exceptions;
        if fixed && n_exc > 0 {
            // Both tables in place: lanes, then high bits, each `n_exc`
            // little-endian fields of the width the leading byte gives.
            let widths = *bytes.get(*at)?;
            *at += 1;
            *at = match (widths & EXC_LANE_BYTES_MASK, widths >> EXC_HI_BYTES_SHIFT) {
                (1, 1) => read_tables::<1, 1>(bytes, *at, n_exc, tables),
                (1, 2) => read_tables::<1, 2>(bytes, *at, n_exc, tables),
                (1, 4) => read_tables::<1, 4>(bytes, *at, n_exc, tables),
                (2, 1) => read_tables::<2, 1>(bytes, *at, n_exc, tables),
                (2, 2) => read_tables::<2, 2>(bytes, *at, n_exc, tables),
                (2, 4) => read_tables::<2, 4>(bytes, *at, n_exc, tables),
                (4, 1) => read_tables::<4, 1>(bytes, *at, n_exc, tables),
                (4, 2) => read_tables::<4, 2>(bytes, *at, n_exc, tables),
                (4, 4) => read_tables::<4, 4>(bytes, *at, n_exc, tables),
                _ => None,
            }?;
        } else if !fixed {
            for _ in 0..n_exc {
                let lane = read_varint(bytes, at)?;
                let hi = read_varint(bytes, at)?;
                tables.push((lane, hi));
            }
        }
        // Lanes must ascend and fall inside the stream.
        let mut prev: Option<u32> = None;
        for &(lane, _) in tables.iter() {
            if lane as usize >= n || prev.is_some_and(|p| lane <= p) {
                return None;
            }
            prev = Some(lane);
        }
        self.width = width;
        self.lanes = n;
        self.payload = payload;
        Some(())
    }

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

    /// Lanes `from..from + n`, exceptions patched in, appended to `out`.
    /// `None`, with nothing appended, when they run past the stream.
    fn read_lanes(&self, bytes: &[u8], from: usize, n: usize, out: &mut Vec<u32>) -> Option<()> {
        // Checked before `out` grows: `n` comes from a posting's tf, which
        // a corrupt block can make arbitrarily large.
        if from + n > self.lanes {
            return None;
        }
        let base = out.len();
        out.resize(base + n, 0);
        let read = self.read_lanes_into(bytes, from, &mut out[base..]);
        if read.is_none() {
            out.truncate(base);
        }
        read
    }

    /// [`Self::read_lanes`] into a slice of exactly the lanes wanted, which
    /// every lane overwrites: a buffer reused across groups is then sized,
    /// not cleared and zero-filled again before the unpack.
    fn read_lanes_into(&self, bytes: &[u8], from: usize, out: &mut [u32]) -> Option<()> {
        let payload = &bytes[self.payload.clone()];
        let n = out.len();
        unpack_lanes_u32(payload, from, self.width, out)?;
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
            out[lane - from] |= self.exception_hi(hi);
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
/// per-run path, where decoding every run of the block would cost more
/// than they do.
const BULK_AFTER_RUNS: u32 = 2;
/// Runs up to this many positions bypass the generic lane reader.
const SMALL_TF: usize = 4;
const BULK_MAX_STRIDE: usize = 4;

/// A position group located for **per-run** access: one pair's run is
/// decoded without touching the block's other runs. A phrase visits few
/// of a common term's pairs per block when its other members are rare,
/// so decoding a whole group — every position of a block's docs — per
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
}

impl GroupIndex {
    /// Locate the group at `*at` (its header) for the block whose per-doc
    /// term frequencies are `tfs`, advancing `*at` past it. `None` on a
    /// truncated or malformed group, after which the index must be
    /// relocated before use.
    pub(crate) fn locate(&mut self, bytes: &[u8], at: &mut usize, tfs: &[u32]) -> Option<()> {
        // Where each pair's values begin, as a running sum over the tfs, and
        // the group's value count as that sum's end: one pass over the tfs,
        // the sum in a register, into a buffer sized once. A wide total kept
        // beside it says whether the running sum wrapped — every partial sum
        // is at most the total, so one check covers them all. (A packed
        // pair's lanes are its gaps alone; that is applied to the sums below
        // once the header says which encoding the group has.)
        self.starts.clear();
        self.starts.resize(tfs.len() + 1, 0);
        let (mut acc, mut total) = (0u32, 0u64);
        for (slot, &tf) in self.starts[1..].iter_mut().zip(tfs) {
            acc = acc.wrapping_add(tf);
            total += u64::from(tf);
            *slot = acc;
        }
        let n = usize::try_from(u32::try_from(total).ok()?).ok()?;
        self.served = 0;
        let header = *bytes.get(*at)?;
        *at += 1;
        match header {
            GROUP_LEB128 => {
                self.kind = GroupKind::Leb128;
                // Each value takes at least a byte: a count the rest of the
                // bytes cannot hold is corrupt, and is refused before it
                // sizes the buffer.
                if n > bytes.len() - *at {
                    return None;
                }
                self.values.clear();
                self.values.reserve(n);
                for _ in 0..n {
                    self.values.push(read_varint(bytes, at)?);
                }
            }
            GROUP_PACKED | GROUP_PACKED_FIXED => {
                let fixed = header == GROUP_PACKED_FIXED;
                self.kind = GroupKind::Packed;
                let n_gaps = n.checked_sub(tfs.len())?;
                self.first.parse_into(bytes, at, tfs.len(), fixed)?;
                self.gap.parse_into(bytes, at, n_gaps, fixed)?;
                // Every gap is at least one (a doc's positions strictly
                // increase), so a zero-width gap stream carries each gap as
                // an exception. One that does not is corrupt; refusing it
                // bounds the gap count by the bytes holding the exceptions,
                // as a payload bounds it at any other width.
                if self.gap.width == 0 && self.gap.exceptions.len() != n_gaps {
                    return None;
                }
            }
            _ => return None,
        }
        // A packed pair's lanes are its gaps: its first position sits in the
        // other stream, so pair `p` starts `p` lanes earlier than its value
        // sum says. A sum that falls short of `p` means a pair with no
        // occurrences, which no block writes: the group is malformed.
        if self.kind == GroupKind::Packed {
            for (p, slot) in self.starts.iter_mut().enumerate() {
                *slot = slot.checked_sub(p as u32)?;
            }
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
            let n_pairs = self.starts.len() - 1;
            let n_gaps = *self.starts.last()? as usize;
            // Sized, not cleared: every lane is overwritten, so only growth
            // past the last group's length is zero-filled.
            self.firsts.resize(n_pairs, 0);
            self.gaps.resize(n_gaps, 0);
            self.first.read_lanes_into(bytes, 0, &mut self.firsts)?;
            self.gap.read_lanes_into(bytes, 0, &mut self.gaps)?;
            self.kind = GroupKind::Bulk;
        }
        self.served += 1;
        match self.kind {
            GroupKind::Leb128 => {
                positions_from_run_values(self.values.get(start..start + tf as usize)?, out)
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
                // A long run: its raw values go straight into `out` and are
                // summed there, with no staging buffer in between.
                let base = out.len();
                self.first.read_lanes(bytes, pair, 1, out)?;
                self.gap.read_lanes(bytes, start, tf as usize - 1, out)?;
                prefix_sum_in_place(&mut out[base..])
            }
            GroupKind::Bulk => {
                let base = out.len();
                out.push(*self.firsts.get(pair)?);
                out.extend_from_slice(self.gaps.get(start..start + tf as usize - 1)?);
                prefix_sum_in_place(&mut out[base..])
            }
        }
    }
}

/// Turn a run's values, already in place (first absolute, then gaps), into
/// absolute positions. `None` on an overflowing gap.
#[inline]
fn prefix_sum_in_place(run: &mut [u32]) -> Option<()> {
    // The running sum stays in a register: reading back the slot just
    // written would put a store-to-load round trip on every position. The
    // values are non-negative, so the run overflows exactly when its total
    // does: one widened sum, which vectorizes, checks that up front and the
    // running sum then wraps freely instead of branching per position.
    let Some((first, rest)) = run.split_first_mut() else {
        return Some(());
    };
    let total = u64::from(*first) + rest.iter().map(|&g| u64::from(g)).sum::<u64>();
    u32::try_from(total).ok()?;
    let mut acc = *first;
    for slot in rest {
        acc = acc.wrapping_add(*slot);
        *slot = acc;
    }
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
        assert_eq!(out[0], GROUP_PACKED_FIXED);
        // firsts: 1 + 1 + 138 (11 bits) + one exception; gaps: 1 + 1 + 125 (5 bits) + one;
        // each table adds one byte of field widths.
        assert!(out.len() < 1 + 145 + 133 + 2, "got {} bytes", out.len());
        assert_eq!(decode_all(&out, 0, &tfs), Some(positions_of(&tfs, &vals)));

        // A short term with a lone value: LEB128 is smaller and allowed.
        let mut out = Vec::new();
        group(&mut out, &[1], &[5], true);
        assert_eq!(out[0], GROUP_LEB128);
        // The same values with LEB128 disallowed pack anyway.
        let mut out = Vec::new();
        group(&mut out, &[1], &[5], false);
        assert_eq!(out[0], GROUP_PACKED_FIXED);
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
            GroupIndex::default().locate(&[3], &mut 0, &[1]).is_none(),
            "unknown header"
        );
        assert!(
            GroupIndex::default()
                .locate(&[GROUP_PACKED, 33, 0], &mut 0, &[1])
                .is_none(),
            "width past 32"
        );
        assert!(
            GroupIndex::default()
                .locate(&[GROUP_PACKED_FIXED, 33, 0], &mut 0, &[1])
                .is_none(),
            "width past 32, fixed tables"
        );
    }

    #[test]
    fn every_exception_field_width_reads_back() {
        // One doc whose gaps are all one (width 1) but for three outliers in
        // its last lanes. The lane count sets the lane field's bytes (lanes
        // past 255 need two, past 65535 four) and the outliers' size the
        // high-bits field's, so each pair of widths is written and read.
        for (n_gaps, lane_bytes) in [(200usize, 1u8), (1_000, 2), (70_000, 4)] {
            for (outlier, hi_bytes) in [(200u32, 1u8), (1 << 12, 2), (1 << 20, 4)] {
                let mut vals = vec![1u32; n_gaps + 1];
                vals[0] = 0;
                for k in 1..=3 {
                    vals[n_gaps + 1 - k] = outlier + k as u32;
                }
                let tfs = [vals.len() as u32];
                let mut out = Vec::new();
                group(&mut out, &tfs, &vals, false);
                // Header, an empty first stream (width 0, no exceptions),
                // then the gap stream's width, count and packed lanes.
                let widths_at = 1 + 2 + 2 + payload_bytes(n_gaps, 1);
                assert_eq!(out[1 + 2], 1, "gap width");
                assert_eq!(out[1 + 2 + 1], 3, "three exceptions");
                assert_eq!(
                    out[widths_at],
                    lane_bytes | (hi_bytes << EXC_HI_BYTES_SHIFT),
                    "{n_gaps} gaps, outlier {outlier}"
                );
                assert_eq!(
                    decode_all(&out, 0, &tfs),
                    Some(positions_of(&tfs, &vals)),
                    "{n_gaps} gaps, outlier {outlier}"
                );
            }
        }
    }

    #[test]
    fn exception_lanes_out_of_order_or_range_are_refused() {
        // A first stream of two lanes at width 0 whose exceptions name the
        // lanes given, in both table forms. Ascending in-range lanes parse;
        // a repeat, a descent, or a lane past the stream is corrupt.
        for (lanes, ok) in [
            ([0u8, 1], true),
            ([1, 1], false),
            ([1, 0], false),
            ([0, 2], false),
        ] {
            let fixed = [
                GROUP_PACKED_FIXED,
                0,
                2,
                1 | (1 << EXC_HI_BYTES_SHIFT),
                lanes[0],
                lanes[1],
                5,
                6,
                0,
                0,
            ];
            let varint = [GROUP_PACKED, 0, 2, lanes[0], 5, lanes[1], 6, 0, 0];
            for bytes in [&fixed[..], &varint[..]] {
                let mut index = GroupIndex::default();
                assert_eq!(
                    index.locate(bytes, &mut 0, &[1, 1]).is_some(),
                    ok,
                    "lanes {lanes:?}, header {}",
                    bytes[0]
                );
            }
        }
    }

    /// A pair with no occurrences cannot be written, so tfs that say one
    /// has are malformed. A packed pair starts as many lanes before its
    /// value sum as there are pairs ahead of it, so a zero tf ahead of
    /// another pair would put that pair before its own lanes: refused,
    /// not summed below zero.
    #[test]
    fn a_zero_tf_ahead_of_a_pair_is_refused() {
        let tfs = [2u32, 2];
        let vals = [5u32, 3, 40, 2];
        let mut out = Vec::new();
        group(&mut out, &tfs, &vals, false);
        let mut index = GroupIndex::default();
        assert_eq!(
            index.locate(&out, &mut 0, &tfs),
            Some(()),
            "premise: the group reads"
        );
        assert_eq!(index.locate(&out, &mut 0, &[0, 2]), None);
    }

    #[test]
    fn corrupt_counts_are_refused_before_they_size_a_buffer() {
        // A LEB128 group claiming more values than its bytes could hold:
        // refused before reserving room for them.
        let huge = [u32::MAX; 128];
        assert!(
            GroupIndex::default()
                .locate(&[GROUP_LEB128, 1, 2, 3], &mut 0, &huge)
                .is_none(),
            "LEB128 count past the bytes"
        );
        // A zero-width gap stream with no exceptions: every gap would be
        // zero, which no doc's strictly increasing positions give, and its
        // lane count would be bounded by nothing in the bytes.
        for header in [GROUP_PACKED, GROUP_PACKED_FIXED] {
            let bytes = [header, 0, 0, 0, 0];
            assert!(
                GroupIndex::default()
                    .locate(&bytes, &mut 0, &[u32::MAX])
                    .is_none(),
                "zero-width gaps without exceptions, header {header}"
            );
            // A zero tf leaves fewer values than docs: refused, not an
            // underflow.
            assert!(
                GroupIndex::default()
                    .locate(&bytes, &mut 0, &[0, 0])
                    .is_none(),
                "zero tfs, header {header}"
            );
        }
        // A run asked for past its group: refused before its gap lanes
        // are appended (its first position may already be).
        let tfs = [6u32, 6];
        let vals: Vec<u32> = (0..12).map(|i| 3 + i % 5).collect();
        let mut out = Vec::new();
        group(&mut out, &tfs, &vals, false);
        let mut index = GroupIndex::default();
        index.locate(&out, &mut 0, &tfs).expect("parses");
        let mut got = vec![9u32];
        assert!(index.run_positions(&out, 1, 1 << 30, &mut got).is_none());
        assert!(got.len() <= 2, "no gap lanes appended: {}", got.len());
    }

    #[test]
    fn both_packed_forms_decode_to_the_same_positions() {
        // A blob from before the fixed exception tables must read as it did:
        // the same groups written in both packed forms decode to the same
        // positions, run by run and through the bulk switch, and a cut group
        // of either form is refused.
        let mut x: u64 = 0x5DEE_CE66_D1CE_4E5B;
        let mut rng = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for case in 0..60 {
            let n_docs = 1 + rng(128) as usize;
            let tfs: Vec<u32> = (0..n_docs)
                .map(|_| 1 + rng(if case % 3 == 0 { 40 } else { 5 }) as u32)
                .collect();
            let mut vals = Vec::new();
            for &tf in &tfs {
                vals.push(rng(3000) as u32);
                vals.extend((1..tf).map(|_| 1 + rng(30) as u32));
            }
            // Outliers in both streams, so both carry exceptions.
            for _ in 0..rng(6) {
                let i = rng(vals.len() as u64) as usize;
                vals[i] = 1 << (16 + rng(12));
            }
            let want = positions_of(&tfs, &vals);
            for packed in [GROUP_PACKED, GROUP_PACKED_FIXED] {
                let mut out = Vec::new();
                encode_group_as(
                    &mut out,
                    &tfs,
                    &vals,
                    false,
                    &mut PackScratch::default(),
                    packed,
                );
                assert_eq!(out[0], packed);
                assert_eq!(
                    decode_all(&out, 0, &tfs).as_ref(),
                    Some(&want),
                    "case {case} kind {packed}"
                );
                // Pairs read one at a time from a fresh index, every other
                // pair, never dense enough for the bulk decode.
                let mut index = GroupIndex::default();
                index.locate(&out, &mut 0, &tfs).expect("parses");
                let mut wi = 0usize;
                for (pair, &tf) in tfs.iter().enumerate() {
                    if pair % 7 == 3 {
                        let mut got = Vec::new();
                        index.run_positions(&out, pair, tf, &mut got).expect("run");
                        assert_eq!(
                            got,
                            want[wi..wi + tf as usize],
                            "case {case} kind {packed} pair {pair}"
                        );
                    }
                    wi += tf as usize;
                }
                for cut in 0..out.len() {
                    assert!(
                        GroupIndex::default()
                            .locate(&out[..cut], &mut 0, &tfs)
                            .is_none(),
                        "case {case} kind {packed} cut {cut}"
                    );
                }
            }
        }
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
            assert_eq!(out[0], GROUP_PACKED_FIXED, "{label}");
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
    fn a_long_or_bulk_run_that_overflows_is_refused() {
        // Runs longer than the small-run path, read one at a time and then
        // in bulk once the reads prove dense. The last run overflows; in the
        // middle one a huge gap wraps the sum back to a plausible position,
        // which a check per step or on the total must both refuse.
        let tfs = [6u32, 6, 6, 6, 6];
        let mut vals = Vec::new();
        for d in 0..4u32 {
            vals.push(100 * d);
            vals.extend([1, 2, 3, 4, 5]);
        }
        vals.push(u32::MAX - 10);
        vals.extend([2, 2, 2, 2, 10]);
        vals[2 * 6 + 3] = u32::MAX;
        for allow in [false, true] {
            let mut out = Vec::new();
            group(&mut out, &tfs, &vals, allow);
            let mut index = GroupIndex::default();
            index.locate(&out, &mut 0, &tfs).expect("parses");
            for (pair, &tf) in tfs.iter().enumerate() {
                let mut got = Vec::new();
                let read = index.run_positions(&out, pair, tf, &mut got);
                match pair {
                    2 | 4 => assert!(read.is_none(), "pair {pair} allow {allow}"),
                    _ => {
                        read.expect("run decodes");
                        let p = 100 * pair as u32;
                        assert_eq!(got, [p, p + 1, p + 3, p + 6, p + 10, p + 15]);
                    }
                }
            }
        }
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
