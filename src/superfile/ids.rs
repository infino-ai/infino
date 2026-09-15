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

use crate::superfile::{
    bits::{MAX_WIDTH, get_bits, payload_bytes, put_bits, width_of},
    format::ID_SIDECAR_ENTRY_BYTES,
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
    /// against `bytes` and the expected `n_docs`, so [`Self::get`] can
    /// index without further checks. A mismatch is an `Err` with the
    /// reason; nothing panics on a malformed sidecar.
    pub(crate) fn parse(bytes: &'a [u8], n_docs: usize) -> Result<Self, String> {
        let u32_at = |at: usize| -> Option<u32> {
            bytes
                .get(at..at + 4)
                .map(|s| u32::from_le_bytes(s.try_into().expect("4 bytes")))
        };
        let u64_at = |at: usize| -> Option<u64> {
            bytes
                .get(at..at + 8)
                .map(|s| u64::from_le_bytes(s.try_into().expect("8 bytes")))
        };
        let declared = u32_at(0).ok_or("packed id sidecar shorter than its header")? as usize;
        if declared != n_docs {
            return Err(format!(
                "packed id sidecar declares {declared} docs, superfile has {n_docs}"
            ));
        }
        if u32_at(4) != Some(BLOCK_DOCS as u32) {
            return Err("packed id sidecar block size is not the reader's".into());
        }
        let n_blocks = u32_at(8).ok_or("packed id sidecar truncated")? as usize;
        if n_blocks != n_docs.div_ceil(BLOCK_DOCS) {
            return Err("packed id sidecar block count does not match n_docs".into());
        }
        for b in 0..n_blocks {
            let off = u64_at(HEADER_BYTES + b * DIR_ENTRY_BYTES)
                .ok_or("packed id sidecar directory truncated")? as usize;
            let n_in_block = match b + 1 == n_blocks {
                true => n_docs - b * BLOCK_DOCS,
                false => BLOCK_DOCS,
            };
            let hi_width = *bytes
                .get(off + 16)
                .ok_or("packed id sidecar block header truncated")?;
            let lo_width = *bytes
                .get(off + 17)
                .ok_or("packed id sidecar block header truncated")?;
            let n_declared = bytes
                .get(off + 18..off + 20)
                .map(|s| u16::from_le_bytes([s[0], s[1]]) as usize)
                .ok_or("packed id sidecar block header truncated")?;
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
        Ok(Self {
            bytes,
            n_docs,
            n_blocks,
        })
    }

    /// The `_id` of local doc `doc`; `None` past `n_docs`.
    #[inline]
    pub(crate) fn get(&self, doc: u32) -> Option<i128> {
        let doc = doc as usize;
        if doc >= self.n_docs {
            return None;
        }
        let b = doc / BLOCK_DOCS;
        let i = doc % BLOCK_DOCS;
        debug_assert!(b < self.n_blocks);
        let dir = HEADER_BYTES + b * DIR_ENTRY_BYTES;
        let off =
            u64::from_le_bytes(self.bytes[dir..dir + 8].try_into().expect("8 bytes")) as usize;
        let hi_base = u64::from_le_bytes(self.bytes[off..off + 8].try_into().expect("8 bytes"));
        let lo_base =
            u64::from_le_bytes(self.bytes[off + 8..off + 16].try_into().expect("8 bytes"));
        let hi_width = self.bytes[off + 16];
        let lo_width = self.bytes[off + 17];
        let n_in_block = u16::from_le_bytes([self.bytes[off + 18], self.bytes[off + 19]]) as usize;
        let hi_start = off + BLOCK_HEADER_BYTES;
        let lo_start = hi_start + payload_bytes(n_in_block, hi_width);
        let lo_end = lo_start + payload_bytes(n_in_block, lo_width);
        let hi = hi_base.wrapping_add(get_bits(&self.bytes[hi_start..lo_start], i, hi_width)?);
        let lo = lo_base.wrapping_add(get_bits(&self.bytes[lo_start..lo_end], i, lo_width)?);
        Some((((hi as u128) << 64) | lo as u128) as i128)
    }
}

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
