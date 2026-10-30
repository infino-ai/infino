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

use std::{
    io::{self, Write},
    ops::Range,
};

#[cfg(any(test, feature = "test-helpers"))]
use crate::superfile::format::{footer::read_kv_metadata, kv};
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
/// Where a block header keeps the bit width of its ids' high halves.
const HI_WIDTH_AT: usize = 16;
/// Where a block header keeps the bit width of its ids' low halves.
const LO_WIDTH_AT: usize = 17;
/// Where a block header keeps how many docs the block holds.
const N_IN_BLOCK_AT: usize = 18;

/// Encode a raw sidecar (`n_docs` little-endian `i128`s, local doc order)
/// into the packed layout. An empty input yields an empty output, the
/// "no sidecar" shape both ends already understand.
pub(crate) fn encode_packed(raw: &[u8]) -> Vec<u8> {
    let mut encoder = PackedIdEncoder::new(Vec::new());
    encoder
        .push_raw(raw)
        .expect("a Vec sink never fails to write");
    let PackedIdOutput {
        mut directory,
        blocks,
        ..
    } = encoder.finish().expect("a Vec sink never fails to write");
    directory.extend_from_slice(&blocks);
    directory
}

/// Encodes a sidecar block by block as its ids arrive, writing each
/// finished block to `blocks` and keeping only the open block and one
/// directory entry per block in memory. A merge of a hundred million rows
/// then holds a few megabytes for its sidecar instead of sixteen bytes per
/// row; [`encode_packed`] is the same encoder over a `Vec`.
///
/// The header and directory are known only once the last block is in, so
/// they come back separately from [`Self::finish`], ahead of the blocks.
pub(crate) struct PackedIdEncoder<W: Write> {
    blocks: W,
    /// Bytes written to `blocks` so far: the next block's offset within it.
    blocks_len: u64,
    /// Each finished block's offset within `blocks`.
    block_offsets: Vec<u64>,
    n_docs: usize,
    /// The open block's ids, split into halves.
    his: Vec<u64>,
    los: Vec<u64>,
    /// One block's bytes, reused across blocks.
    block: Vec<u8>,
}

/// A finished packed sidecar in two pieces: `directory` (header plus block
/// directory, with offsets from the sidecar's start) followed by the blocks
/// in the sink the encoder was given. Empty when no ids were pushed.
pub(crate) struct PackedIdOutput<W> {
    pub(crate) directory: Vec<u8>,
    pub(crate) blocks: W,
    pub(crate) blocks_len: u64,
}

impl<W> PackedIdOutput<W> {
    /// Bytes of the whole sidecar, directory and blocks together.
    pub(crate) fn len(&self) -> u64 {
        self.directory.len() as u64 + self.blocks_len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The same sidecar with its blocks behind another handle: a scratch
    /// file's write handle swapped for a read handle, or a `Vec` for a
    /// cursor over it.
    pub(crate) fn map_blocks<R>(self, f: impl FnOnce(W) -> R) -> PackedIdOutput<R> {
        PackedIdOutput {
            directory: self.directory,
            blocks: f(self.blocks),
            blocks_len: self.blocks_len,
        }
    }
}

impl<W: Write> PackedIdEncoder<W> {
    pub(crate) fn new(blocks: W) -> Self {
        Self {
            blocks,
            blocks_len: 0,
            block_offsets: Vec::new(),
            n_docs: 0,
            his: Vec::with_capacity(BLOCK_DOCS),
            los: Vec::with_capacity(BLOCK_DOCS),
            block: Vec::new(),
        }
    }

    /// Append the next doc's id. A block that fills is encoded and written
    /// out at once.
    pub(crate) fn push(&mut self, id: i128) -> io::Result<()> {
        let v = id as u128;
        self.his.push((v >> 64) as u64);
        self.los.push(v as u64);
        self.n_docs += 1;
        if self.his.len() == BLOCK_DOCS {
            self.flush_block()?;
        }
        Ok(())
    }

    /// Append `raw` ids (whole little-endian `i128`s, in local doc order).
    pub(crate) fn push_raw(&mut self, raw: &[u8]) -> io::Result<()> {
        debug_assert!(raw.len().is_multiple_of(ID_SIDECAR_ENTRY_BYTES));
        for id in raw.chunks_exact(ID_SIDECAR_ENTRY_BYTES) {
            self.push(i128::from_le_bytes(id.try_into().expect("16-byte id")))?;
        }
        Ok(())
    }

    /// Encode the open block into `self.block` and write it to the sink.
    fn flush_block(&mut self) -> io::Result<()> {
        let n = self.his.len();
        if n == 0 {
            return Ok(());
        }
        let hi_base = self.his.iter().copied().min().unwrap_or(0);
        let lo_base = self.los.iter().copied().min().unwrap_or(0);
        let hi_width = width_of(self.his.iter().map(|&h| h - hi_base).max().unwrap_or(0));
        let lo_width = width_of(self.los.iter().map(|&l| l - lo_base).max().unwrap_or(0));
        let out = &mut self.block;
        out.clear();
        out.extend_from_slice(&hi_base.to_le_bytes());
        out.extend_from_slice(&lo_base.to_le_bytes());
        out.push(hi_width);
        out.push(lo_width);
        out.extend_from_slice(&(n as u16).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        let hi_start = out.len();
        out.resize(hi_start + payload_bytes(n, hi_width), 0);
        for (i, &h) in self.his.iter().enumerate() {
            put_bits(
                &mut out[hi_start..],
                i * hi_width as usize,
                h - hi_base,
                hi_width,
            );
        }
        let lo_start = out.len();
        out.resize(lo_start + payload_bytes(n, lo_width), 0);
        for (i, &l) in self.los.iter().enumerate() {
            put_bits(
                &mut out[lo_start..],
                i * lo_width as usize,
                l - lo_base,
                lo_width,
            );
        }
        self.blocks.write_all(out)?;
        self.block_offsets.push(self.blocks_len);
        self.blocks_len += out.len() as u64;
        self.his.clear();
        self.los.clear();
        Ok(())
    }

    /// Flush the open block and assemble the header and directory.
    pub(crate) fn finish(mut self) -> io::Result<PackedIdOutput<W>> {
        self.flush_block()?;
        self.blocks.flush()?;
        if self.n_docs == 0 {
            return Ok(PackedIdOutput {
                directory: Vec::new(),
                blocks: self.blocks,
                blocks_len: 0,
            });
        }
        let n_blocks = self.block_offsets.len();
        debug_assert_eq!(n_blocks, self.n_docs.div_ceil(BLOCK_DOCS));
        let directory_len = HEADER_BYTES + n_blocks * DIR_ENTRY_BYTES;
        let mut directory = Vec::with_capacity(directory_len);
        directory.extend_from_slice(&(self.n_docs as u32).to_le_bytes());
        directory.extend_from_slice(&(BLOCK_DOCS as u32).to_le_bytes());
        directory.extend_from_slice(&(n_blocks as u32).to_le_bytes());
        directory.extend_from_slice(&0u32.to_le_bytes());
        // Block offsets are from the sidecar's start, which is where the
        // reader's directory entries point.
        for off in &self.block_offsets {
            directory.extend_from_slice(&(directory_len as u64 + off).to_le_bytes());
        }
        Ok(PackedIdOutput {
            directory,
            blocks: self.blocks,
            blocks_len: self.blocks_len,
        })
    }
}

/// Bytes of a packed sidecar's header and block directory for `n_docs`
/// documents: what a read of a few rows fetches first, to find their blocks.
pub(crate) fn directory_len(n_docs: usize) -> usize {
    HEADER_BYTES + n_docs.div_ceil(BLOCK_DOCS) * DIR_ENTRY_BYTES
}

/// Check a packed sidecar's header against `n_docs` and return its block
/// count. `bytes` must hold the header and the directory.
fn check_header(bytes: &[u8], n_docs: usize) -> Result<usize, String> {
    let declared = u32_le_at(bytes, 0).ok_or("packed id sidecar shorter than its header")? as usize;
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
    Ok(n_blocks)
}

/// A packed sidecar's header and directory alone, over the bytes
/// [`directory_len`] names: enough to place every block, so a read of a few
/// rows fetches those blocks and nothing else of the sidecar.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedDirectory<'a> {
    bytes: &'a [u8],
    n_blocks: usize,
}

impl<'a> PackedDirectory<'a> {
    pub(crate) fn open(bytes: &'a [u8], n_docs: usize) -> Result<Self, String> {
        let n_blocks = check_header(bytes, n_docs)?;
        Ok(Self { bytes, n_blocks })
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.n_blocks
    }

    /// Where block `b` lies in the sidecar, as offsets from its start: from
    /// its directory entry to the next block's, or to the sidecar's end for
    /// the last. `None` for a block past the end or a directory that does not
    /// ascend.
    pub(crate) fn block_range(&self, b: usize, sidecar_len: usize) -> Option<Range<usize>> {
        if b >= self.n_blocks {
            return None;
        }
        let start = u64_le_at(self.bytes, HEADER_BYTES + b * DIR_ENTRY_BYTES)? as usize;
        let end = match b + 1 < self.n_blocks {
            true => u64_le_at(self.bytes, HEADER_BYTES + (b + 1) * DIR_ENTRY_BYTES)? as usize,
            false => sidecar_len,
        };
        (start < end && end <= sidecar_len).then_some(start..end)
    }
}

/// The ids of `docs` — ascending, every one in block `b` — decoded from the
/// block's own bytes, appended to `out` in `docs` order. `None` when the block
/// is malformed or a doc is not its.
pub(crate) fn ids_in_block(
    block: &[u8],
    b: usize,
    n_docs: usize,
    n_blocks: usize,
    docs: &[u32],
    out: &mut Vec<i128>,
) -> Option<()> {
    let expected = match b + 1 == n_blocks {
        true => n_docs - b * BLOCK_DOCS,
        false => BLOCK_DOCS,
    };
    let blk = parse_block(block, 0, expected)?;
    if docs.iter().any(|&d| d as usize / BLOCK_DOCS != b) {
        return None;
    }
    let mut his = Vec::new();
    let mut los = Vec::new();
    decode_docs(block, &blk, docs, &mut his, &mut los, |_, id| out.push(id))
}

/// Block `b`'s header and where its ids sit when the block starts at `off`
/// in `bytes`, or `None` if the header is bad or the block runs past the
/// bytes. `expected` is how many docs the block must hold.
fn parse_block(bytes: &[u8], off: usize, expected: usize) -> Option<IdBlock> {
    let header = bytes.get(off..off + BLOCK_HEADER_BYTES)?;
    let hi_width = header[HI_WIDTH_AT];
    let lo_width = header[LO_WIDTH_AT];
    let n_in_block =
        u16::from_le_bytes([header[N_IN_BLOCK_AT], header[N_IN_BLOCK_AT + 1]]) as usize;
    if hi_width > MAX_WIDTH || lo_width > MAX_WIDTH || n_in_block != expected {
        return None;
    }
    let hi_start = off + BLOCK_HEADER_BYTES;
    let lo_start = hi_start + payload_bytes(n_in_block, hi_width);
    let lo_end = lo_start + payload_bytes(n_in_block, lo_width);
    bytes.get(hi_start..lo_end)?;
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

/// The ids of `docs`, all of one block `blk` in `bytes`, through
/// `sink(index_in_docs, id)`: unpacked wholesale once the run asks for at
/// least `1 / BULK_DENSITY` of the block, lane by lane otherwise.
fn decode_docs(
    bytes: &[u8],
    blk: &IdBlock,
    docs: &[u32],
    his: &mut Vec<u64>,
    los: &mut Vec<u64>,
    mut sink: impl FnMut(usize, i128),
) -> Option<()> {
    let hi_bytes = &bytes[blk.hi.clone()];
    let lo_bytes = &bytes[blk.lo.clone()];
    if docs.len() * BULK_DENSITY < blk.n_in_block {
        for (k, &d) in docs.iter().enumerate() {
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
        for (k, &d) in docs.iter().enumerate() {
            let lane = d as usize % BLOCK_DOCS;
            if lane >= blk.n_in_block {
                return None;
            }
            sink(k, blk.id(his[lane], los[lane]));
        }
    }
    Some(())
}

/// A packed id sidecar over borrowed bytes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedIds<'a> {
    bytes: &'a [u8],
    n_docs: usize,
    n_blocks: usize,
}

impl<'a> PackedIds<'a> {
    /// Opens a sidecar by checking only its header. Checking every block here
    /// would read the whole sidecar from disk on a cold open, even for a query
    /// that never asks for an `_id`. A bad block shows up as `None` on read.
    pub(crate) fn open(bytes: &'a [u8], n_docs: usize) -> Result<Self, String> {
        let n_blocks = check_header(bytes, n_docs)?;
        Ok(Self {
            bytes,
            n_docs,
            n_blocks,
        })
    }

    /// Checks every block and names the first bad one. A rewrite calls this
    /// before copying the sidecar, so a bad block is not copied into a new file.
    pub(crate) fn check_blocks(&self) -> Result<(), String> {
        (0..self.n_blocks).try_for_each(|b| match self.block(b) {
            Some(_) => Ok(()),
            None => Err(format!("packed id sidecar block {b} is malformed")),
        })
    }

    /// Block `b`'s header and where its ids sit, or `None` if the header is
    /// bad or the block runs past the sidecar.
    #[inline]
    fn block(&self, b: usize) -> Option<IdBlock> {
        let off = u64_le_at(self.bytes, HEADER_BYTES + b * DIR_ENTRY_BYTES)? as usize;
        // Every block holds BLOCK_DOCS docs but the last, which holds the rest.
        let expected = match b + 1 == self.n_blocks {
            true => self.n_docs - b * BLOCK_DOCS,
            false => BLOCK_DOCS,
        };
        parse_block(self.bytes, off, expected)
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
            decode_docs(
                self.bytes,
                &blk,
                &docs[i..j],
                &mut his,
                &mut los,
                |k, id| sink(i + k, id),
            )?;
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

/// Test-only: where block `b`'s header starts in a packed sidecar.
#[cfg(any(test, feature = "test-helpers"))]
fn block_offset(sidecar: &[u8], b: usize) -> usize {
    u64_le_at(sidecar, HEADER_BYTES + b * DIR_ENTRY_BYTES).expect("directory entry") as usize
}

/// Test-only: break block `b`'s header (a bit width past 64), so a test has
/// one bad block among good ones.
#[cfg(any(test, feature = "test-helpers"))]
fn damage_block(sidecar: &mut [u8], b: usize) {
    sidecar[block_offset(sidecar, b) + HI_WIDTH_AT] = MAX_WIDTH + 1;
}

/// Test-only: where `superfile`'s id sidecar sits, read from its footer.
#[cfg(any(test, feature = "test-helpers"))]
fn id_sidecar_range(superfile: &[u8]) -> Range<usize> {
    let kvs = read_kv_metadata(superfile).expect("superfile footer");
    let at = |key: &str| -> usize {
        kvs.get(key)
            .expect("the superfile carries an id sidecar")
            .parse()
            .expect("a byte offset")
    };
    let off = at(kv::IDS_OFFSET);
    off..off + at(kv::IDS_LENGTH)
}

#[cfg(any(test, feature = "test-helpers"))]
test_visible! {
    /// Test-only: a copy of `superfile` with block `b` of its id sidecar
    /// broken, and every other byte unchanged.
    fn with_damaged_id_block(superfile: &[u8], b: usize) -> Vec<u8> {
        let mut out = superfile.to_vec();
        let range = id_sidecar_range(superfile);
        damage_block(&mut out[range], b);
        out
    }
}

#[cfg(any(test, feature = "test-helpers"))]
test_visible! {
    /// Test-only: a copy of `superfile` whose id directory starts block `b`
    /// where block `at` starts, so `b` begins before an earlier block ends.
    fn with_out_of_order_id_block(superfile: &[u8], b: usize, at: usize) -> Vec<u8> {
        let mut out = superfile.to_vec();
        let range = id_sidecar_range(superfile);
        let sidecar = &mut out[range];
        let start = block_offset(sidecar, at) as u64;
        let entry = HEADER_BYTES + b * DIR_ENTRY_BYTES;
        sidecar[entry..entry + DIR_ENTRY_BYTES].copy_from_slice(&start.to_le_bytes());
        out
    }
}

#[cfg(any(test, feature = "test-helpers"))]
test_visible! {
    /// Test-only: a copy of `superfile` whose id sidecar block `b` still
    /// decodes but to the wrong ids: its high base is one more, which moves
    /// every id in the block out of the file's id range. The damage a block
    /// check cannot see and only a check against the footer can.
    fn with_shifted_id_block(superfile: &[u8], b: usize) -> Vec<u8> {
        let mut out = superfile.to_vec();
        let range = id_sidecar_range(superfile);
        let sidecar = &mut out[range];
        let at = block_offset(sidecar, b);
        let base = u64_le_at(sidecar, at).expect("block header");
        sidecar[at..at + size_of::<u64>()].copy_from_slice(&base.wrapping_add(1).to_le_bytes());
        out
    }
}

#[cfg(any(test, feature = "test-helpers"))]
test_visible! {
    /// Test-only: checks every block of `superfile`'s packed id sidecar, the
    /// way a rewrite does before copying it.
    fn check_id_sidecar(superfile: &[u8]) -> Result<(), String> {
        let sidecar = &superfile[id_sidecar_range(superfile)];
        let n_docs = u32_le_at(sidecar, 0).ok_or("packed id sidecar shorter than its header")?;
        PackedIds::open(sidecar, n_docs as usize)?.check_blocks()
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
        let p = PackedIds::open(&packed, ids.len()).expect("valid");
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

    /// The packed layout byte by byte, for ids that cross one block
    /// boundary, built by hand from the module's layout comment: pins the
    /// header, the directory's sidecar-relative offsets, the block header and
    /// the bit-packed payload independently of the encoder, so a rewrite of
    /// the encoder cannot move the format without failing here.
    #[test]
    fn a_packed_sidecar_is_the_documented_bytes() {
        const HI: u64 = 0x1122_3344_5566_7788;
        const LO: u64 = 0x99AA_BBCC_DDEE_FF00;
        let id = |lo: u64| (((HI as u128) << 64) | lo as u128) as i128;
        // Block 0: one id repeated, so both halves pack at width 0 and the
        // block is its header alone. Block 1: three ids whose low halves
        // are base + 0, 1, 2, width 2.
        let mut ids = vec![id(LO); BLOCK_DOCS];
        ids.extend((0..3).map(|i| id(LO + i)));
        let n_blocks = 2usize;
        let block0_at = HEADER_BYTES + n_blocks * DIR_ENTRY_BYTES;
        let block1_at = block0_at + BLOCK_HEADER_BYTES;

        let mut expected = Vec::new();
        expected.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        expected.extend_from_slice(&(BLOCK_DOCS as u32).to_le_bytes());
        expected.extend_from_slice(&(n_blocks as u32).to_le_bytes());
        expected.extend_from_slice(&0u32.to_le_bytes());
        expected.extend_from_slice(&(block0_at as u64).to_le_bytes());
        expected.extend_from_slice(&(block1_at as u64).to_le_bytes());
        expected.extend_from_slice(&HI.to_le_bytes());
        expected.extend_from_slice(&LO.to_le_bytes());
        expected.extend_from_slice(&[0u8, 0]);
        expected.extend_from_slice(&(BLOCK_DOCS as u16).to_le_bytes());
        expected.extend_from_slice(&0u32.to_le_bytes());
        expected.extend_from_slice(&HI.to_le_bytes());
        expected.extend_from_slice(&LO.to_le_bytes());
        expected.extend_from_slice(&[0u8, 2]);
        expected.extend_from_slice(&3u16.to_le_bytes());
        expected.extend_from_slice(&0u32.to_le_bytes());
        // Offsets 0, 1, 2 at two bits each, as a little-endian bit stream.
        expected.push(0b10_01_00);

        assert_eq!(encode_packed(&raw(&ids)), expected);
        assert_eq!(expected.len(), block1_at + BLOCK_HEADER_BYTES + 1);
    }

    /// The streaming encoder, fed in uneven pieces and writing its blocks to
    /// a separate sink, produces the bytes the one-shot encode does: where a
    /// piece ends has no bearing on the sidecar.
    #[test]
    fn a_streamed_encode_matches_the_whole_buffer_encode() {
        let ids: Vec<i128> = (0..BLOCK_DOCS as i128 * 3 + 17)
            .map(|i| ((i * 7919) << 64) | (i * 31 + 5))
            .collect();
        let raw: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
        let whole = encode_packed(&raw);

        let mut encoder = PackedIdEncoder::new(Vec::new());
        // Pieces that straddle block boundaries, one id at a time at the end.
        let mut at = 0;
        for piece in [700usize, 1500, 1, 1, 2000] {
            let end = (at + piece * ID_SIDECAR_ENTRY_BYTES).min(raw.len());
            encoder.push_raw(&raw[at..end]).expect("vec sink");
            at = end;
        }
        encoder.push_raw(&raw[at..]).expect("vec sink");
        let streamed = encoder.finish().expect("vec sink");
        assert_eq!(streamed.len() as usize, whole.len());
        let mut joined = streamed.directory.clone();
        joined.extend_from_slice(&streamed.blocks);
        assert_eq!(
            joined, whole,
            "streamed and whole-buffer encodes agree byte for byte"
        );

        let empty = PackedIdEncoder::new(Vec::new()).finish().expect("vec sink");
        assert!(empty.is_empty(), "no ids pushed is no sidecar");
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
        let p = PackedIds::open(&packed, ids.len()).expect("opens");
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
    fn a_malformed_header_is_refused_and_a_malformed_block_answers_none() {
        // A sidecar of 2,053 docs in three blocks, broken one way at a time.
        //  - a wrong doc count, or the header cut short: `open` refuses it.
        //  - the last block broken (cut short, its start past the end, a bit
        //    width past 64, or the wrong doc count): `open` still takes it,
        //    reads in that block return `None`, and `check_blocks` names it.
        //  - docs in the other blocks still return their ids.
        // This test pins that `open` checks only the header, and each read
        // still checks its block.
        let ids: Vec<i128> = (0..BLOCK_DOCS as i128 * 2 + 5)
            .map(|i| i * 11 + 3)
            .collect();
        let packed = encode_packed(&raw(&ids));
        let n = ids.len();
        let last_block = n.div_ceil(BLOCK_DOCS) - 1;
        let (bad_doc, good_docs) = (
            (last_block * BLOCK_DOCS) as u32,
            [0, (last_block * BLOCK_DOCS - 1) as u32],
        );

        assert!(PackedIds::open(&packed, n - 1).is_err(), "n_docs mismatch");
        for cut in [0, 8, HEADER_BYTES + 3, HEADER_BYTES + DIR_ENTRY_BYTES] {
            assert!(PackedIds::open(&packed[..cut], n).is_err(), "cut {cut}");
        }
        assert_eq!(
            PackedIds::open(&packed, n).expect("intact").check_blocks(),
            Ok(()),
            "an intact sidecar passes the full check"
        );

        let at = block_offset(&packed, last_block);
        let damage = |edit: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = packed.clone();
            edit(&mut bytes);
            bytes
        };
        let cases: [(&str, Vec<u8>); 4] = [
            ("block cut short", packed[..packed.len() - 1].to_vec()),
            (
                "directory entry past the sidecar",
                damage(&|b| {
                    let dir = HEADER_BYTES + last_block * DIR_ENTRY_BYTES;
                    let past_end = b.len() as u64;
                    b[dir..dir + 8].copy_from_slice(&past_end.to_le_bytes());
                }),
            ),
            ("width past 64", damage(&|b| damage_block(b, last_block))),
            (
                "doc count",
                damage(&|b| {
                    let count_at = at + N_IN_BLOCK_AT;
                    let wrong = (n - last_block * BLOCK_DOCS - 1) as u16;
                    b[count_at..count_at + 2].copy_from_slice(&wrong.to_le_bytes());
                }),
            ),
        ];
        for (what, bytes) in &cases {
            let p = PackedIds::open(bytes, n).expect("header intact");
            assert_eq!(p.get(bad_doc), None, "{what}");
            assert_eq!(
                p.check_blocks(),
                Err(format!("packed id sidecar block {last_block} is malformed")),
                "{what}: the full check names the block"
            );
            for good in good_docs {
                assert_eq!(p.get(good), Some(ids[good as usize]), "{what}: doc {good}");
            }
        }
    }
}

#[cfg(test)]
mod block_tests {
    use super::*;

    /// Rows in the fixture: three full blocks and a partial fourth.
    const ROWS: usize = 3 * BLOCK_DOCS + 17;

    fn raw_ids() -> Vec<u8> {
        // Ids with both halves varying, not monotonic, so a block's bases and
        // widths are all exercised.
        (0..ROWS)
            .flat_map(|i| {
                let hi = (i as u128 % 5) << 64;
                let lo = ((i as u128 * 7919) % 100_003) + (1u128 << 40);
                ((hi | lo) as i128).to_le_bytes()
            })
            .collect()
    }

    /// A block read on its own, through the directory's range for it,
    /// decodes the same ids the whole sidecar does.
    #[test]
    fn a_block_decodes_alone_as_in_the_whole_sidecar() {
        let raw = raw_ids();
        let packed = encode_packed(&raw);
        let whole = PackedIds::open(&packed, ROWS).expect("whole sidecar");
        let dir = PackedDirectory::open(&packed[..directory_len(ROWS)], ROWS).expect("directory");
        assert_eq!(dir.n_blocks(), ROWS.div_ceil(BLOCK_DOCS));
        for b in 0..dir.n_blocks() {
            let range = dir.block_range(b, packed.len()).expect("block in place");
            let docs: Vec<u32> = [0usize, 1, 2, 500, 1023]
                .iter()
                .map(|&lane| (b * BLOCK_DOCS + lane) as u32)
                .filter(|&d| (d as usize) < ROWS)
                .collect();
            let mut alone = Vec::new();
            ids_in_block(&packed[range], b, ROWS, dir.n_blocks(), &docs, &mut alone)
                .expect("block decodes");
            let mut together = Vec::new();
            whole.get_many(&docs, &mut together).expect("whole decodes");
            assert_eq!(alone, together, "block {b}");
        }
        // A doc of another block is refused, as is a block out of range.
        let mut out = Vec::new();
        let range = dir.block_range(0, packed.len()).expect("block 0");
        assert!(
            ids_in_block(
                &packed[range],
                0,
                ROWS,
                dir.n_blocks(),
                &[BLOCK_DOCS as u32],
                &mut out
            )
            .is_none()
        );
        assert!(dir.block_range(dir.n_blocks(), packed.len()).is_none());
    }
}
