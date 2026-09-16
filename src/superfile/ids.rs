// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Packed stable-id sidecar: the `_id` of every local doc, addressable
//! in O(1), at a fraction of the sixteen bytes a raw `i128` costs.
//!
//! Ids are Snowflake-style — a millisecond timestamp in the high half, a
//! worker id and counter in the low half — minted by one generator per
//! writer and appended in order, so within a run of consecutive docs the
//! high halves sit in a narrow band and the low halves in another. The
//! sidecar splits every id into its two `u64` halves and stores each
//! block of [`BLOCK_DOCS`] docs as two frame-of-reference streams: a
//! base (the block's minimum) plus each doc's offset from it, bit-packed
//! at the width the block's largest offset needs. A block of sequential
//! ids costs a few bytes per doc; a block of arbitrary ids degrades to
//! the raw sixteen (two 64-bit widths) and never more.
//!
//! Resolving a doc stays a fixed-cost read: block index and slot by
//! division, one directory entry, one block header, one bit extraction
//! per half. That is the property the sidecar exists for (a hit → `_id`
//! resolve that never decodes a Parquet page), kept at ~4× fewer bytes.
//!
//! ```text
//!   header    u32 n_docs | u32 block_docs | u32 n_blocks | u32 reserved
//!   directory n_blocks × u64  byte offset of each block from the sidecar start
//!   block     u64 hi_base | u64 lo_base | u8 hi_width | u8 lo_width |
//!             u16 n_in_block | u32 reserved |
//!             hi payload: ceil(n_in_block × hi_width / 8) bytes |
//!             lo payload: ceil(n_in_block × lo_width / 8) bytes
//! ```
//!
//! Every multi-byte field is little-endian; a payload's values are a
//! little-endian bit stream (value `i` occupies bits `i·w .. (i+1)·w`).
//! The layout is named by the `inf.ids.layout` footer key; a sidecar
//! without that key is the older raw `i128` array.

use std::ops::Range;

use crate::superfile::{
    bits::{MAX_WIDTH, for_each_lane, get_bits, payload_bytes, put_bits, width_of},
    format::{ID_SIDECAR_ENTRY_BYTES, u32_le_at, u64_le_at},
};

/// Docs per frame-of-reference block. Large enough that the 24-byte
/// block header and 8-byte directory entry are noise per doc, small
/// enough that a block of sequential ids spans a narrow band of
/// timestamps and counters.
pub(crate) const BLOCK_DOCS: usize = 1024;
/// Header size in bytes (four `u32` fields).
const HEADER_BYTES: usize = 16;
/// Per-block header size in bytes: two `u64` bases, two `u8` widths, a
/// `u16` count and a `u32` reserved word.
const BLOCK_HEADER_BYTES: usize = 24;
/// Directory entry size: one `u64` block offset.
const DIR_ENTRY_BYTES: usize = 8;
/// Encode a raw sidecar (`n_docs` little-endian `i128`s, local doc order)
/// into the packed layout. An empty input yields an empty output, the
/// "no sidecar" shape both ends already understand.
pub(crate) fn encode_packed(raw: &[u8]) -> Vec<u8> {
    debug_assert!(raw.len().is_multiple_of(ID_SIDECAR_ENTRY_BYTES));
    let n_docs = raw.len() / ID_SIDECAR_ENTRY_BYTES;
    if n_docs == 0 {
        return Vec::new();
    }
    let n_blocks = n_docs.div_ceil(BLOCK_DOCS);
    let mut out =
        Vec::with_capacity(HEADER_BYTES + n_blocks * (DIR_ENTRY_BYTES + BLOCK_HEADER_BYTES));
    out.extend_from_slice(&(n_docs as u32).to_le_bytes());
    out.extend_from_slice(&(BLOCK_DOCS as u32).to_le_bytes());
    out.extend_from_slice(&(n_blocks as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    let dir_start = out.len();
    out.resize(dir_start + n_blocks * DIR_ENTRY_BYTES, 0);

    let mut his: Vec<u64> = Vec::with_capacity(BLOCK_DOCS);
    let mut los: Vec<u64> = Vec::with_capacity(BLOCK_DOCS);
    for (b, chunk) in raw.chunks(BLOCK_DOCS * ID_SIDECAR_ENTRY_BYTES).enumerate() {
        his.clear();
        los.clear();
        for id in chunk.chunks_exact(ID_SIDECAR_ENTRY_BYTES) {
            let v = u128::from_le_bytes(id.try_into().expect("16-byte id"));
            his.push((v >> 64) as u64);
            los.push(v as u64);
        }
        let n = his.len();
        let hi_base = his.iter().copied().min().unwrap_or(0);
        let lo_base = los.iter().copied().min().unwrap_or(0);
        let hi_width = width_of(his.iter().map(|&h| h - hi_base).max().unwrap_or(0));
        let lo_width = width_of(los.iter().map(|&l| l - lo_base).max().unwrap_or(0));
        let block_off = out.len() as u64;
        out[dir_start + b * DIR_ENTRY_BYTES..dir_start + (b + 1) * DIR_ENTRY_BYTES]
            .copy_from_slice(&block_off.to_le_bytes());
        out.extend_from_slice(&hi_base.to_le_bytes());
        out.extend_from_slice(&lo_base.to_le_bytes());
        out.push(hi_width);
        out.push(lo_width);
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let hi_start = out.len();
        out.resize(hi_start + payload_bytes(n, hi_width), 0);
        for (i, &h) in his.iter().enumerate() {
            put_bits(
                &mut out[hi_start..],
                i * hi_width as usize,
                h - hi_base,
                hi_width,
            );
        }
        let lo_start = out.len();
        out.resize(lo_start + payload_bytes(n, lo_width), 0);
        for (i, &l) in los.iter().enumerate() {
            put_bits(
                &mut out[lo_start..],
                i * lo_width as usize,
                l - lo_base,
                lo_width,
            );
        }
    }
    out
}

/// A validated packed sidecar over borrowed bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedIds<'a> {
    bytes: &'a [u8],
    n_docs: usize,
    n_blocks: usize,
}

impl<'a> PackedIds<'a> {
    /// Parse and validate every header, directory entry and block bound
    /// against `bytes` and the expected `n_docs`. A mismatch is an `Err`
    /// with the reason; nothing panics on a malformed sidecar. This is
    /// the superfile-open check; the per-query path takes [`Self::open`].
    pub(crate) fn parse(bytes: &'a [u8], n_docs: usize) -> Result<Self, String> {
        let ids = Self::open(bytes, n_docs)?;
        for b in 0..ids.n_blocks {
            let off = u64_le_at(bytes, HEADER_BYTES + b * DIR_ENTRY_BYTES)
                .ok_or("packed id sidecar directory truncated")? as usize;
            let n_in_block = match b + 1 == ids.n_blocks {
                true => n_docs - b * BLOCK_DOCS,
                false => BLOCK_DOCS,
            };
            let header = bytes
                .get(off..off + BLOCK_HEADER_BYTES)
                .ok_or("packed id sidecar block header truncated")?;
            let hi_width = header[16];
            let lo_width = header[17];
            let n_declared = u16::from_le_bytes([header[18], header[19]]) as usize;
            if hi_width > MAX_WIDTH || lo_width > MAX_WIDTH || n_declared != n_in_block {
                return Err(format!("packed id sidecar block {b} header is malformed"));
            }
            let end = off
                + BLOCK_HEADER_BYTES
                + payload_bytes(n_in_block, hi_width)
                + payload_bytes(n_in_block, lo_width);
            if end > bytes.len() {
                return Err(format!("packed id sidecar block {b} runs past the sidecar"));
            }
        }
        Ok(ids)
    }

    /// Open a sidecar by its header alone — no walk of the block
    /// directory — for the per-query id resolve of a superfile whose
    /// sidecar [`Self::parse`] validated at open. The walk is a few
    /// thousand blocks on a large superfile, and it was being repeated
    /// on every query. [`Self::get`] bounds-checks each read it makes,
    /// so bytes this is misapplied to answer `None` rather than panic.
    pub(crate) fn open(bytes: &'a [u8], n_docs: usize) -> Result<Self, String> {
        let declared =
            u32_le_at(bytes, 0).ok_or("packed id sidecar shorter than its header")? as usize;
        if declared != n_docs {
            return Err(format!(
                "packed id sidecar declares {declared} docs, superfile has {n_docs}"
            ));
        }
        if u32_le_at(bytes, 4) != Some(BLOCK_DOCS as u32) {
            return Err("packed id sidecar block size is not the reader's".into());
        }
        let n_blocks = u32_le_at(bytes, 8).ok_or("packed id sidecar truncated")? as usize;
        if n_blocks != n_docs.div_ceil(BLOCK_DOCS) {
            return Err("packed id sidecar block count does not match n_docs".into());
        }
        if bytes.len() < HEADER_BYTES + n_blocks * DIR_ENTRY_BYTES {
            return Err("packed id sidecar directory truncated".into());
        }
        Ok(Self {
            bytes,
            n_docs,
            n_blocks,
        })
    }

    /// Block `b`'s header and stream ranges, `None` if the bytes cannot
    /// hold it.
    #[inline]
    fn block(&self, b: usize) -> Option<IdBlock> {
        let off = u64_le_at(self.bytes, HEADER_BYTES + b * DIR_ENTRY_BYTES)? as usize;
        let header = self.bytes.get(off..off + BLOCK_HEADER_BYTES)?;
        let hi_width = header[16];
        let lo_width = header[17];
        let n_in_block = u16::from_le_bytes([header[18], header[19]]) as usize;
        if hi_width > MAX_WIDTH || lo_width > MAX_WIDTH {
            return None;
        }
        let hi_start = off + BLOCK_HEADER_BYTES;
        let lo_start = hi_start + payload_bytes(n_in_block, hi_width);
        let lo_end = lo_start + payload_bytes(n_in_block, lo_width);
        self.bytes.get(hi_start..lo_end)?;
        Some(IdBlock {
            hi_base: u64::from_le_bytes(header[..8].try_into().expect("8 bytes")),
            lo_base: u64::from_le_bytes(header[8..16].try_into().expect("8 bytes")),
            hi_width,
            lo_width,
            n_in_block,
            hi: hi_start..lo_start,
            lo: lo_start..lo_end,
        })
    }

    /// One doc's id, through the same block path `get_many` takes.
    #[cfg(test)]
    pub(crate) fn get(&self, doc: u32) -> Option<i128> {
        let mut out = Vec::with_capacity(1);
        self.get_many(&[doc], &mut out)?;
        out.pop()
    }

    pub(crate) fn get_many(&self, docs: &[u32], out: &mut Vec<i128>) -> Option<()> {
        let base = out.len();
        let resolved = if docs.is_sorted() {
            out.reserve(docs.len());
            self.resolve_sorted(docs, |_, id| out.push(id))
        } else {
            // Ranked hits arrive in score order. Resolve them in doc order,
            // so a block's directory entry and header are parsed once for
            // every doc it holds, then scatter back to the caller's order.
            let mut order: Vec<(u32, u32)> = docs
                .iter()
                .enumerate()
                .map(|(i, &d)| (d, i as u32))
                .collect();
            order.sort_unstable();
            let sorted: Vec<u32> = order.iter().map(|&(d, _)| d).collect();
            out.resize(base + docs.len(), 0);
            self.resolve_sorted(&sorted, |k, id| out[base + order[k].1 as usize] = id)
        };
        if resolved.is_none() {
            out.truncate(base);
        }
        resolved
    }

    /// Resolve ascending `docs`, calling `sink(index_in_docs, id)` for each.
    /// A block's docs are served from one parsed header: unpacked wholesale
    /// once the run asks for at least `1 / BULK_DENSITY` of the block, lane
    /// by lane otherwise.
    fn resolve_sorted(&self, docs: &[u32], mut sink: impl FnMut(usize, i128)) -> Option<()> {
        let mut his: Vec<u64> = Vec::new();
        let mut los: Vec<u64> = Vec::new();
        let mut i = 0usize;
        while i < docs.len() {
            let b = docs[i] as usize / BLOCK_DOCS;
            let mut j = i;
            while j < docs.len() && docs[j] as usize / BLOCK_DOCS == b {
                j += 1;
            }
            if docs[j - 1] as usize >= self.n_docs {
                return None;
            }
            let blk = self.block(b)?;
            let hi_bytes = &self.bytes[blk.hi.clone()];
            let lo_bytes = &self.bytes[blk.lo.clone()];
            if (j - i) * BULK_DENSITY < blk.n_in_block {
                for (k, &d) in docs.iter().enumerate().take(j).skip(i) {
                    let lane = d as usize % BLOCK_DOCS;
                    if lane >= blk.n_in_block {
                        return None;
                    }
                    let hi = get_bits(hi_bytes, lane, blk.hi_width)?;
                    let lo = get_bits(lo_bytes, lane, blk.lo_width)?;
                    sink(k, blk.id(hi, lo));
                }
            } else {
                his.clear();
                los.clear();
                for_each_lane(hi_bytes, 0, blk.n_in_block, blk.hi_width, |v| his.push(v))?;
                for_each_lane(lo_bytes, 0, blk.n_in_block, blk.lo_width, |v| los.push(v))?;
                for (k, &d) in docs.iter().enumerate().take(j).skip(i) {
                    let lane = d as usize % BLOCK_DOCS;
                    if lane >= blk.n_in_block {
                        return None;
                    }
                    sink(k, blk.id(his[lane], los[lane]));
                }
            }
            i = j;
        }
        Some(())
    }
}

struct IdBlock {
    hi_base: u64,
    lo_base: u64,
    hi_width: u8,
    lo_width: u8,
    n_in_block: usize,
    hi: Range<usize>,
    lo: Range<usize>,
}

impl IdBlock {
    /// The id whose halves are `hi` and `lo` above the block's bases.
    #[inline]
    fn id(&self, hi: u64, lo: u64) -> i128 {
        let hi = self.hi_base.wrapping_add(hi);
        let lo = self.lo_base.wrapping_add(lo);
        (((hi as u128) << 64) | lo as u128) as i128
    }
}

/// A block is unpacked whole for a bulk resolve when at least one doc in
/// this many of its docs is asked for; below that, per-doc reads win.
const BULK_DENSITY: usize = 4;

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(ids: &[i128]) -> Vec<u8> {
        ids.iter().flat_map(|id| id.to_le_bytes()).collect()
    }

    fn round_trip(ids: &[i128]) -> Vec<u8> {
        let packed = encode_packed(&raw(ids));
        let p = PackedIds::parse(&packed, ids.len()).expect("valid");
        for (d, &id) in ids.iter().enumerate() {
            assert_eq!(p.get(d as u32), Some(id), "doc {d}");
        }
        assert_eq!(p.get(ids.len() as u32), None);
        packed
    }

    #[test]
    fn sequential_snowflake_ids_pack_to_a_few_bytes_per_doc() {
        // Timestamp high, worker | counter low; counter increments, the
        // millisecond ticks every 40 ids.
        let worker: u64 = 0x1234 << 24;
        let ids: Vec<i128> = (0..10_000u64)
            .map(|i| {
                let ts = 1_700_000_000_000u64 + i / 40;
                (((ts as u128) << 64) | (worker | (i % 40)) as u128) as i128
            })
            .collect();
        let packed = round_trip(&ids);
        let per_doc = packed.len() as f64 / ids.len() as f64;
        assert!(per_doc < 2.0, "{per_doc:.2} B/doc for sequential ids");
    }

    #[test]
    fn arbitrary_ids_never_cost_more_than_raw_plus_headers() {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let ids: Vec<i128> = (0..3_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (((x as u128) << 64) | (x.rotate_left(29) as u128)) as i128
            })
            .collect();
        let packed = round_trip(&ids);
        let headers = HEADER_BYTES + 3 * (DIR_ENTRY_BYTES + BLOCK_HEADER_BYTES);
        assert!(packed.len() <= ids.len() * ID_SIDECAR_ENTRY_BYTES + headers);
    }

    #[test]
    fn edge_values_and_block_boundaries_round_trip() {
        let mut ids = vec![i128::MIN, -1, 0, 1, i128::MAX, 7, 7, 7];
        ids.extend((0..BLOCK_DOCS as i128 * 2 + 3).map(|i| i * 3 - 5));
        round_trip(&ids);
        round_trip(&[42]);
        assert!(encode_packed(&[]).is_empty());
    }

    #[test]
    fn bulk_resolve_matches_single_lookups_in_every_order_and_density() {
        let ids: Vec<i128> = (0..BLOCK_DOCS as i128 * 3 + 17)
            .map(|i| (i * 7919) ^ (i << 40))
            .collect();
        let packed = round_trip(&ids);
        let p = PackedIds::parse(&packed, ids.len()).expect("parses");
        let n = ids.len() as u32;
        let cases: Vec<Vec<u32>> = vec![
            (0..n).collect(),               // every doc, ascending
            (0..n).step_by(3).collect(),    // dense enough for bulk
            (0..n).step_by(97).collect(),   // sparse: per-doc path
            vec![5, 3, 3, 2_000, 1_024, 1], // unsorted, duplicates
            // Score-ordered hits: every block touched, out of order, dense
            // in one block and sparse in the others.
            (0..n)
                .rev()
                .step_by(11)
                .chain((0..64).map(|k| k * 2))
                .collect(),
            vec![n - 1, n - 2], // last block, tail
            Vec::new(),
        ];
        for docs in cases {
            let mut got = Vec::new();
            p.get_many(&docs, &mut got).expect("in range");
            let want: Vec<i128> = docs.iter().map(|&d| ids[d as usize]).collect();
            assert_eq!(got, want, "docs {docs:?}");
        }
        assert!(
            p.get_many(&[0, n], &mut Vec::new()).is_none(),
            "past n_docs"
        );
        assert!(
            p.get_many(&[n, 0], &mut Vec::new()).is_none(),
            "past n_docs, unsorted"
        );
    }

    #[test]
    fn header_only_open_answers_none_on_a_block_it_cannot_hold() {
        let ids: Vec<i128> = (0..BLOCK_DOCS as i128 * 2 + 5)
            .map(|i| i * 11 + 3)
            .collect();
        let mut packed = round_trip(&ids);
        // Point the last block's directory entry past the sidecar.
        let dir = HEADER_BYTES + 2 * DIR_ENTRY_BYTES;
        let past_end = packed.len() as u64;
        packed[dir..dir + 8].copy_from_slice(&past_end.to_le_bytes());
        assert!(
            PackedIds::parse(&packed, ids.len()).is_err(),
            "open-time parse refuses it"
        );
        let p = PackedIds::open(&packed, ids.len()).expect("header is intact");
        assert_eq!(p.get(0), Some(ids[0]));
        assert_eq!(
            p.get(2 * BLOCK_DOCS as u32 - 1),
            Some(ids[2 * BLOCK_DOCS - 1])
        );
        assert_eq!(p.get(2 * BLOCK_DOCS as u32), None, "corrupt block");
        assert_eq!(p.get(ids.len() as u32), None, "past n_docs");
        assert!(PackedIds::open(&packed[..HEADER_BYTES + 8], ids.len()).is_err());
    }

    #[test]
    fn malformed_sidecars_are_refused() {
        let ids: Vec<i128> = (0..1_500).map(|i| i as i128).collect();
        let packed = encode_packed(&raw(&ids));
        assert!(PackedIds::parse(&packed, 1_499).is_err(), "n_docs mismatch");
        for cut in [0usize, 8, HEADER_BYTES + 3, packed.len() - 1] {
            assert!(
                PackedIds::parse(&packed[..cut], ids.len()).is_err(),
                "cut {cut}"
            );
        }
        let mut bad_width = packed.clone();
        let block0 = u64::from_le_bytes(
            packed[HEADER_BYTES..HEADER_BYTES + 8]
                .try_into()
                .expect("8 bytes"),
        ) as usize;
        bad_width[block0 + 16] = 65;
        assert!(
            PackedIds::parse(&bad_width, ids.len()).is_err(),
            "width past 64"
        );
    }
}
