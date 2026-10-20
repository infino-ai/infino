// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term's *tier*: its best documents table-wide, by impact, so a ranked
//! query can find the documents that hold several of its terms — and
//! bound every superfile — before opening any superfile.
//!
//! A tier is a sequence of entries `(superfile ordinal, local doc id,
//! term frequency, quantized length)` in descending *key* — the length-
//! normalized term frequency `tf / (tf + k1·(1 − b + b·dl/avgdl))`, the
//! factor a BM25 score multiplies the term's idf by — cut into blocks of
//! [`BLOCK_ENTRIES`]
//! and the blocks into *chunks* of at most [`CHUNK_TARGET_BYTES`], read
//! in order and usually only in part: a query stops once the blocks it
//! has read bound everything it has not. The chunks of every term in a
//! slice sit back to back in that slice's tier object, so a run names its
//! chunks by offset. A query reads a chunk's directory with one ranged
//! fetch — sized from the entry count the run records, every block but
//! the last being full — and then only the blocks it consumes, each with
//! its own fetch; the directory and each block carry their own CRC, so a
//! read of either verifies on its own.
//!
//! ```text
//! chunk: n_blocks u32 | directory crc32c u32
//!        | per block: max_impact f32 | n_entries u16 | byte_len u32 | crc32c u32
//!        | block bodies back to back
//! block: ordinal_bits u8 | lid_bits u8 | tf_bits u8 | len_bits u8
//!        | ordinals, bit-packed sorted deltas | local ids, bit-packed
//!        | term frequencies, bit-packed | quantized lengths, bit-packed
//! ```
//!
//! A chunk's layout is versioned by the slice that names it, so it carries
//! no magic of its own. Each stream is packed in lanes of [`LANE`] values
//! by the SIMD packer, and the block's final partial lane densely at the
//! same width: most terms' tiers are a handful of entries, and a lane
//! padded to full width would cost them a hundred times their size.
//!
//! Inside a block the entries are stored sorted by `(ordinal, local id)`
//! — the block is always consumed whole, so its internal order carries
//! no information and the sorted ordinals pack to a few bits.
//!
//! An entry stores the facts a score is computed from, not a score: a
//! term frequency and the one-byte length bucket the superfile itself
//! scores the document at. A reader then computes the document's score
//! for a query exactly as the superfile's kernel would — the table's idf
//! times `tf / (tf + norm)` with the same quantized length and average —
//! so a candidate the tier names needs no opening to be scored, and a
//! merge re-cuts tiers at any average length without loosening a bound.
//! The ordering key, and the directory's per-block maximum that bounds
//! every entry of its block and of the blocks after it, are computed at
//! the average length and parameters the slice records for the column
//! ([`ColumnScoring`]); a reader scales a maximum to the table's average
//! as it scales any stored ceiling.

use std::ops::Range;

use bitpacking::{BitPacker, BitPacker4x};

use super::TermIndexError;
use crate::{
    superfile::{
        format::checksum::crc32c,
        fts::bm25::{Bm25Params, dequantize_len, tf_factor},
    },
    utils::{
        bytes::{u16_le_at, u32_le_at},
        varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
    },
};

/// The average length a column's keys are computed at when the build was
/// told none: any fixed value orders a tier consistently, since the slice
/// records what it used, but a reader's bound scales by the ratio to the
/// table's average, so a build is expected to pass the real one.
const DEFAULT_KEY_AVGDL: f32 = 1.0;

/// Entries per block, the unit a reader consumes at a time. A multiple of
/// the bit-packer's lane width.
pub(crate) const BLOCK_ENTRIES: usize = 4096;
/// Bytes a chunk grows to before the next block starts a new one. A query
/// at small `k` reads one or two chunks per term.
pub(crate) const CHUNK_TARGET_BYTES: usize = 4 * 1024 * 1024;
const LANE: usize = BitPacker4x::BLOCK_LEN;
const CHUNK_HEADER_LEN: usize = 4 + 4;
/// Where the directory's CRC sits in the header: over the directory
/// entries that follow it.
const CRC_AT: usize = 4;
const DIRECTORY_ENTRY_LEN: usize = 4 + 2 + 4 + 4;
const BLOCK_HEADER_LEN: usize = 4;

/// One tier entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TierEntry {
    /// Ordinal into the root's superfile list.
    pub(crate) superfile: u32,
    /// Row within that superfile.
    pub(crate) doc_id: u32,
    /// The term's frequency in that document.
    pub(crate) tf: u32,
    /// The document's length as the superfile scores it: the one-byte
    /// bucket its norm table holds (see [`dequantize_len`]).
    pub(crate) len: u8,
}

/// The average length and parameters a column's tier keys are computed
/// at: recorded by the slice, so a reader and a merge compute the same
/// key for the same entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColumnScoring {
    pub(crate) avgdl: f32,
    pub(crate) params: Bm25Params,
}

impl Default for ColumnScoring {
    fn default() -> Self {
        Self {
            avgdl: DEFAULT_KEY_AVGDL,
            params: Bm25Params::default(),
        }
    }
}

impl ColumnScoring {
    /// The length-normalized term frequency of an entry: what the term's
    /// idf multiplies into its score. Orders a tier and bounds its blocks.
    pub(crate) fn key(&self, tf: u32, len: u8) -> f32 {
        tf_factor(tf, self.params.dl_norm_k1(dequantize_len(len), self.avgdl))
    }
}

/// Where one chunk of a term's tier sits in its slice's tier object and
/// how much of the tier it holds, so a prefix read fetches only the chunks
/// it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierChunkRef {
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) n_entries: u64,
}

/// One encoded chunk before it is placed in a tier object.
pub(crate) struct EncodedChunk {
    pub(crate) n_entries: u64,
    pub(crate) bytes: Vec<u8>,
}

/// A chunk's block directory: where each block sits in the chunk, its
/// entry count, its exact largest impact and its CRC. What a query reads
/// first, to pick blocks by their maxima and fetch only those.
#[derive(Debug)]
pub(crate) struct TierDirectory {
    blocks: Vec<BlockDir>,
}

/// A whole chunk in hand — directory and bodies — as a build or a merge
/// has it.
pub(crate) struct TierChunk<'a> {
    directory: TierDirectory,
    bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
struct BlockDir {
    max_key: f32,
    n_entries: u32,
    /// Byte offset of the body from the chunk's start.
    offset: usize,
    len: usize,
    crc: u32,
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(format!("tier chunk: {what}"))
}

/// Cut `entries` — in descending key, `keys` alongside — into blocks and
/// chunks. Returns each chunk's bytes with its entry count, in tier order.
pub(crate) fn encode_tier(entries: &[TierEntry], keys: &[f32]) -> Vec<EncodedChunk> {
    debug_assert_eq!(entries.len(), keys.len());
    debug_assert!(
        keys.windows(2).all(|w| w[0] >= w[1]),
        "tier entries must be in descending key"
    );
    let mut chunks = Vec::new();
    let mut blocks: Vec<(BlockDir, Vec<u8>)> = Vec::new();
    let mut chunk_bytes = 0usize;
    let mut scratch = vec![0u8; LANE * 4];
    let flush = |blocks: &mut Vec<(BlockDir, Vec<u8>)>, chunks: &mut Vec<EncodedChunk>| {
        if blocks.is_empty() {
            return;
        }
        let n_entries: u64 = blocks.iter().map(|(d, _)| d.n_entries as u64).sum();
        let mut out = Vec::with_capacity(
            CHUNK_HEADER_LEN
                + blocks.len() * DIRECTORY_ENTRY_LEN
                + blocks.iter().map(|(_, b)| b.len()).sum::<usize>(),
        );
        out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for (d, body) in blocks.iter() {
            out.extend_from_slice(&d.max_key.to_le_bytes());
            out.extend_from_slice(&(d.n_entries as u16).to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(&crc32c(body).to_le_bytes());
        }
        let crc = crc32c(&out[CHUNK_HEADER_LEN..]);
        out[CRC_AT..CRC_AT + 4].copy_from_slice(&crc.to_le_bytes());
        for (_, body) in blocks.drain(..) {
            out.extend_from_slice(&body);
        }
        chunks.push(EncodedChunk {
            n_entries,
            bytes: out,
        });
    };
    for (block, block_keys) in entries
        .chunks(BLOCK_ENTRIES)
        .zip(keys.chunks(BLOCK_ENTRIES))
    {
        let (dir, body) = encode_block(block, block_keys, &mut scratch);
        if chunk_bytes > 0 && chunk_bytes + body.len() > CHUNK_TARGET_BYTES {
            flush(&mut blocks, &mut chunks);
            chunk_bytes = 0;
        }
        chunk_bytes += body.len() + DIRECTORY_ENTRY_LEN;
        blocks.push((dir, body));
    }
    flush(&mut blocks, &mut chunks);
    chunks
}

fn encode_block(block: &[TierEntry], keys: &[f32], scratch: &mut [u8]) -> (BlockDir, Vec<u8>) {
    debug_assert!(!block.is_empty() && block.len() <= BLOCK_ENTRIES);
    let max_key = keys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sorted: Vec<TierEntry> = block.to_vec();
    sorted.sort_unstable_by_key(|e| (e.superfile, e.doc_id));
    let n = sorted.len();
    let full = n / LANE * LANE;
    let mut ordinals = vec![0u32; n];
    let mut doc_ids = vec![0u32; n];
    let mut tfs = vec![0u32; n];
    let mut lens = vec![0u32; n];
    for (i, e) in sorted.iter().enumerate() {
        ordinals[i] = e.superfile;
        doc_ids[i] = e.doc_id;
        tfs[i] = e.tf;
        lens[i] = u32::from(e.len);
    }
    let bp = BitPacker4x::new();
    // Each lane's deltas start from the previous lane's last ordinal, so a
    // lane costs only the spread of its own ordinals, not their magnitude.
    let lane_initial = |i: usize| if i == 0 { 0 } else { ordinals[i * LANE - 1] };
    // The partial lane is stored as deltas too, at a width the SIMD packer
    // would have chosen for them.
    let tail_deltas: Vec<u32> = (full..n)
        .map(|i| ordinals[i] - if i == 0 { 0 } else { ordinals[i - 1] })
        .collect();
    let width = |values: &[u32]| values.iter().map(|v| bits_for(*v)).max().unwrap_or(0);
    let ordinal_bits = ordinals[..full]
        .chunks(LANE)
        .enumerate()
        .map(|(i, lane)| bp.num_bits_sorted(lane_initial(i), lane))
        .max()
        .unwrap_or(0)
        .max(width(&tail_deltas));
    let lid_bits = doc_ids[..full]
        .chunks(LANE)
        .map(|lane| bp.num_bits(lane))
        .max()
        .unwrap_or(0)
        .max(width(&doc_ids[full..]));
    let lane_width = |values: &[u32]| {
        values[..full]
            .chunks(LANE)
            .map(|lane| bp.num_bits(lane))
            .max()
            .unwrap_or(0)
            .max(width(&values[full..]))
    };
    let tf_bits = lane_width(&tfs);
    let len_bits = lane_width(&lens);
    let mut body = Vec::with_capacity(
        BLOCK_HEADER_LEN
            + n * (ordinal_bits as usize
                + lid_bits as usize
                + tf_bits as usize
                + len_bits as usize)
                / 8
            + 4,
    );
    body.push(ordinal_bits);
    body.push(lid_bits);
    body.push(tf_bits);
    body.push(len_bits);
    for (i, lane) in ordinals[..full].chunks(LANE).enumerate() {
        let written = bp.compress_sorted(lane_initial(i), lane, scratch, ordinal_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    pack_dense(&tail_deltas, ordinal_bits, &mut body);
    for lane in doc_ids[..full].chunks(LANE) {
        let written = bp.compress(lane, scratch, lid_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    pack_dense(&doc_ids[full..], lid_bits, &mut body);
    for lane in tfs[..full].chunks(LANE) {
        let written = bp.compress(lane, scratch, tf_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    pack_dense(&tfs[full..], tf_bits, &mut body);
    for lane in lens[..full].chunks(LANE) {
        let written = bp.compress(lane, scratch, len_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    pack_dense(&lens[full..], len_bits, &mut body);
    (
        BlockDir {
            max_key,
            n_entries: n as u32,
            offset: 0,
            len: body.len(),
            crc: 0,
        },
        body,
    )
}

/// Bits that hold `v`.
fn bits_for(v: u32) -> u8 {
    (32 - v.leading_zeros()) as u8
}

/// Bytes a dense run of `n` values at `bits` each takes.
fn dense_len(n: usize, bits: u8) -> usize {
    (n * bits as usize).div_ceil(8)
}

/// Append `values` at `bits` each, least significant bit first, no
/// padding between values; the run ends on a byte boundary.
fn pack_dense(values: &[u32], bits: u8, out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + dense_len(values.len(), bits), 0);
    let mut bit = 0usize;
    for &v in values {
        for b in 0..bits as usize {
            if v >> b & 1 == 1 {
                out[start + (bit + b) / 8] |= 1 << ((bit + b) % 8);
            }
        }
        bit += bits as usize;
    }
}

/// Read `n` values packed by [`pack_dense`] from the start of `bytes`.
fn unpack_dense(bytes: &[u8], n: usize, bits: u8, out: &mut [u32]) {
    let mut bit = 0usize;
    for slot in out.iter_mut().take(n) {
        let mut v = 0u32;
        for b in 0..bits as usize {
            if bytes[(bit + b) / 8] >> ((bit + b) % 8) & 1 == 1 {
                v |= 1 << b;
            }
        }
        *slot = v;
        bit += bits as usize;
    }
}

impl TierDirectory {
    /// Bytes the directory of a chunk holding `n_entries` takes, header
    /// included: every block but the last is full, so the block count
    /// follows from the entry count the run records, and a reader sizes
    /// the directory fetch before it has read a byte of the chunk.
    pub(crate) fn len_for(n_entries: u64) -> usize {
        CHUNK_HEADER_LEN + (n_entries as usize).div_ceil(BLOCK_ENTRIES) * DIRECTORY_ENTRY_LEN
    }

    /// Parse a directory from the start of a chunk; `bytes` may be the
    /// directory alone or the whole chunk. Verifies the directory's CRC.
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, TermIndexError> {
        if bytes.len() < CHUNK_HEADER_LEN {
            return Err(malformed("short header"));
        }
        let n_blocks = u32_le_at(bytes, 0).ok_or_else(|| malformed("block count"))? as usize;
        let dir_end = CHUNK_HEADER_LEN
            .checked_add(
                n_blocks
                    .checked_mul(DIRECTORY_ENTRY_LEN)
                    .ok_or_else(|| malformed("directory"))?,
            )
            .ok_or_else(|| malformed("directory"))?;
        if bytes.len() < dir_end {
            return Err(malformed("short directory"));
        }
        let stored = u32_le_at(bytes, CRC_AT).ok_or_else(|| malformed("crc"))?;
        if stored != crc32c(&bytes[CHUNK_HEADER_LEN..dir_end]) {
            return Err(TermIndexError::HashMismatch);
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut offset = dir_end;
        for i in 0..n_blocks {
            let at = CHUNK_HEADER_LEN + i * DIRECTORY_ENTRY_LEN;
            let f = |o: usize| u32_le_at(bytes, at + o).ok_or_else(|| malformed("directory entry"));
            let max_key = f32::from_bits(f(0)?);
            let n_entries = u16_le_at(bytes, at + 4).ok_or_else(|| malformed("directory entry"))?;
            let len = f(6)? as usize;
            let crc = f(10)?;
            if n_entries == 0 || n_entries as usize > BLOCK_ENTRIES || len < BLOCK_HEADER_LEN {
                return Err(malformed("directory entry out of range"));
            }
            // Every block but the last is full, which is what lets a reader
            // size the directory from the entry count alone.
            if i + 1 < n_blocks && n_entries as usize != BLOCK_ENTRIES {
                return Err(malformed("a block before the last is not full"));
            }
            blocks.push(BlockDir {
                max_key,
                n_entries: u32::from(n_entries),
                offset,
                len,
                crc,
            });
            offset = offset
                .checked_add(len)
                .ok_or_else(|| malformed("block offsets"))?;
        }
        Ok(Self { blocks })
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Exact largest key in block `i`, at the slice's [`ColumnScoring`].
    pub(crate) fn block_max(&self, i: usize) -> f32 {
        self.blocks[i].max_key
    }

    #[cfg(test)]
    pub(crate) fn block_entries(&self, i: usize) -> usize {
        self.blocks[i].n_entries as usize
    }

    /// Where block `i`'s body sits, as byte offsets from the chunk's start.
    pub(crate) fn block_range(&self, i: usize) -> Range<usize> {
        let d = &self.blocks[i];
        d.offset..d.offset + d.len
    }

    /// Where the chunk's bodies end: the chunk's byte length.
    pub(crate) fn chunk_len(&self) -> usize {
        self.blocks
            .last()
            .map_or(CHUNK_HEADER_LEN, |d| d.offset + d.len)
    }

    /// Check block `i`'s body against its CRC: what a body fetched by
    /// range is verified by. Once per fetch, not per decode.
    pub(crate) fn verify_block(&self, i: usize, body: &[u8]) -> Result<(), TermIndexError> {
        let d = self.blocks.get(i).ok_or_else(|| malformed("block index"))?;
        if body.len() != d.len {
            return Err(malformed("block body length"));
        }
        if crc32c(body) != d.crc {
            return Err(TermIndexError::HashMismatch);
        }
        Ok(())
    }

    /// Decode block `i` from its body into `out` (cleared first); entries
    /// are in `(ordinal, doc id)` order.
    pub(crate) fn decode_block(
        &self,
        i: usize,
        body: &[u8],
        out: &mut Vec<TierEntry>,
    ) -> Result<(), TermIndexError> {
        out.clear();
        let d = self.blocks.get(i).ok_or_else(|| malformed("block index"))?;
        if body.len() != d.len || body.len() < BLOCK_HEADER_LEN {
            return Err(malformed("block body length"));
        }
        let (ordinal_bits, lid_bits, tf_bits, len_bits) = (body[0], body[1], body[2], body[3]);
        let n = d.n_entries as usize;
        let full = n / LANE * LANE;
        let rest = n - full;
        let lanes = full / LANE;
        let stream = |bits: u8| lanes * LANE * bits as usize / 8 + dense_len(rest, bits);
        let need = BLOCK_HEADER_LEN
            + stream(ordinal_bits)
            + stream(lid_bits)
            + stream(tf_bits)
            + stream(len_bits);
        if body.len() != need
            || ordinal_bits > 32
            || lid_bits > 32
            || tf_bits > 32
            || len_bits > u8::BITS as u8
        {
            return Err(malformed("block body length"));
        }
        let bp = BitPacker4x::new();
        let mut ordinals = vec![0u32; n];
        let mut doc_ids = vec![0u32; n];
        let mut tfs = vec![0u32; n];
        let mut lens = vec![0u32; n];
        let mut at = BLOCK_HEADER_LEN;
        let mut initial = 0u32;
        for lane in ordinals[..full].chunks_mut(LANE) {
            at += bp.decompress_sorted(initial, &body[at..], lane, ordinal_bits);
            initial = lane[LANE - 1];
        }
        unpack_dense(&body[at..], rest, ordinal_bits, &mut ordinals[full..]);
        at += dense_len(rest, ordinal_bits);
        for v in &mut ordinals[full..] {
            initial = initial.wrapping_add(*v);
            *v = initial;
        }
        for lane in doc_ids[..full].chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, lid_bits);
        }
        unpack_dense(&body[at..], rest, lid_bits, &mut doc_ids[full..]);
        at += dense_len(rest, lid_bits);
        for lane in tfs[..full].chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, tf_bits);
        }
        unpack_dense(&body[at..], rest, tf_bits, &mut tfs[full..]);
        at += dense_len(rest, tf_bits);
        for lane in lens[..full].chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, len_bits);
        }
        unpack_dense(&body[at..], rest, len_bits, &mut lens[full..]);
        at += dense_len(rest, len_bits);
        debug_assert_eq!(at, body.len());
        out.reserve(n);
        for i in 0..n {
            out.push(TierEntry {
                superfile: ordinals[i],
                doc_id: doc_ids[i],
                tf: tfs[i],
                len: lens[i] as u8,
            });
        }
        Ok(())
    }
}

impl<'a> TierChunk<'a> {
    /// Check a whole chunk: its directory's CRC and every block's.
    #[cfg(test)]
    pub(crate) fn verify(bytes: &'a [u8]) -> Result<(), TermIndexError> {
        let chunk = Self::open(bytes)?;
        for i in 0..chunk.n_blocks() {
            chunk
                .directory
                .verify_block(i, &bytes[chunk.directory.block_range(i)])?;
        }
        Ok(())
    }

    /// Parse a whole chunk's directory and check its bodies fit it.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
        let directory = TierDirectory::parse(bytes)?;
        if bytes.len() != directory.chunk_len() {
            return Err(malformed("body length does not match the directory"));
        }
        Ok(Self { directory, bytes })
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.directory.n_blocks()
    }

    /// Exact largest impact in block `i`.
    #[cfg(test)]
    pub(crate) fn block_max(&self, i: usize) -> f32 {
        self.directory.block_max(i)
    }

    #[cfg(test)]
    pub(crate) fn block_entries(&self, i: usize) -> usize {
        self.directory.block_entries(i)
    }

    /// Decode block `i` into `out` (cleared first).
    pub(crate) fn decode_block(
        &self,
        i: usize,
        out: &mut Vec<TierEntry>,
    ) -> Result<(), TermIndexError> {
        if i >= self.n_blocks() {
            return Err(malformed("block index"));
        }
        let body = &self.bytes[self.directory.block_range(i)];
        self.directory.decode_block(i, body, out)
    }
}

/// Serialize a chunk list into a run header.
pub(crate) fn push_chunk_refs(out: &mut Vec<u8>, chunks: &[TierChunkRef]) {
    push_varint(out, chunks.len() as u32);
    for c in chunks {
        push_u64_varint(out, c.offset);
        push_u64_varint(out, c.len);
        push_u64_varint(out, c.n_entries);
    }
}

/// Parse a chunk list written by [`push_chunk_refs`].
pub(crate) fn read_chunk_refs(
    bytes: &[u8],
    at: &mut usize,
) -> Result<Vec<TierChunkRef>, TermIndexError> {
    let n = read_varint(bytes, at).ok_or_else(|| malformed("chunk count"))? as usize;
    let mut out = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let offset = read_u64_varint(bytes, at).ok_or_else(|| malformed("chunk offset"))?;
        let len = read_u64_varint(bytes, at).ok_or_else(|| malformed("chunk len"))?;
        let n_entries = read_u64_varint(bytes, at).ok_or_else(|| malformed("chunk entries"))?;
        out.push(TierChunkRef {
            offset,
            len,
            n_entries,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scoring the tests key their samples at.
    fn scoring() -> ColumnScoring {
        ColumnScoring {
            avgdl: 20.0,
            params: Bm25Params::default(),
        }
    }

    fn key_of(e: &TierEntry) -> f32 {
        scoring().key(e.tf, e.len)
    }

    fn descending(entries: &mut [TierEntry]) {
        entries.sort_by(|a, b| {
            key_of(b)
                .total_cmp(&key_of(a))
                .then(a.superfile.cmp(&b.superfile))
                .then(a.doc_id.cmp(&b.doc_id))
        });
    }

    fn keys(entries: &[TierEntry]) -> Vec<f32> {
        entries.iter().map(key_of).collect()
    }

    /// A deterministic sample: `n` entries over `superfiles` superfiles of
    /// `docs` documents each, small term frequencies and lengths across the
    /// whole bucket range.
    fn sample(n: usize, superfiles: u32, docs: u32, seed: u64) -> Vec<TierEntry> {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut out: Vec<TierEntry> = (0..n)
            .map(|_| TierEntry {
                superfile: (next() % superfiles as u64) as u32,
                doc_id: (next() % docs as u64) as u32,
                tf: 1 + (next() % 4) as u32,
                len: (next() % 256) as u8,
            })
            .collect();
        descending(&mut out);
        out
    }

    fn decode_all(chunks: &[EncodedChunk]) -> Vec<TierEntry> {
        let mut all = Vec::new();
        let mut buf = Vec::new();
        for c in chunks {
            TierChunk::verify(&c.bytes).expect("crc");
            let chunk = TierChunk::open(&c.bytes).expect("open");
            for i in 0..chunk.n_blocks() {
                chunk.decode_block(i, &mut buf).expect("decode");
                all.extend_from_slice(&buf);
            }
        }
        all
    }

    fn sorted(entries: &[TierEntry]) -> Vec<(u32, u32, u32, u8)> {
        let mut v: Vec<(u32, u32, u32, u8)> = entries
            .iter()
            .map(|e| (e.superfile, e.doc_id, e.tf, e.len))
            .collect();
        v.sort_unstable();
        v
    }

    #[test]
    fn round_trips_every_entry_exactly() {
        let entries = sample(10_000, 37, 1_000_000, 7);
        let chunks = encode_tier(&entries, &keys(&entries));
        let decoded = decode_all(&chunks);
        assert_eq!(decoded.len(), entries.len());
        assert_eq!(sorted(&entries), sorted(&decoded), "entries are exact");
    }

    #[test]
    fn directory_bounds_are_exact_and_monotone() {
        let entries = sample(20_000, 100, 500_000, 11);
        let chunks = encode_tier(&entries, &keys(&entries));
        let mut prev_min = f32::INFINITY;
        let mut at = 0usize;
        for c in &chunks {
            let chunk = TierChunk::open(&c.bytes).expect("open");
            for i in 0..chunk.n_blocks() {
                let n = chunk.block_entries(i);
                let block = &entries[at..at + n];
                assert_eq!(chunk.block_max(i), key_of(&block[0]));
                assert!(
                    chunk.block_max(i) <= prev_min,
                    "blocks are in descending key"
                );
                prev_min = key_of(&block[n - 1]);
                at += n;
            }
        }
        assert_eq!(at, entries.len());
    }

    #[test]
    fn a_constant_key_block_round_trips_exactly() {
        let mut entries: Vec<TierEntry> = (0..300)
            .map(|i| TierEntry {
                superfile: i % 3,
                doc_id: i * 7,
                tf: 2,
                len: 9,
            })
            .collect();
        descending(&mut entries);
        let chunks = encode_tier(&entries, &keys(&entries));
        assert_eq!(chunks.len(), 1);
        let decoded = decode_all(&chunks);
        assert!(decoded.iter().all(|e| e.tf == 2 && e.len == 9));
        assert_eq!(decoded.len(), 300);
    }

    #[test]
    fn chunks_cut_at_the_target_and_count_their_entries() {
        let entries = sample(2_000_000, 1_080, 1_000_000, 3);
        let chunks = encode_tier(&entries, &keys(&entries));
        assert!(chunks.len() > 1, "2M entries exceed one 4 MiB chunk");
        let total: u64 = chunks.iter().map(|c| c.n_entries).sum();
        assert_eq!(total, entries.len() as u64);
        for c in &chunks {
            assert!(
                c.bytes.len() <= CHUNK_TARGET_BYTES + 64 * 1024,
                "chunk {} over target",
                c.bytes.len()
            );
        }
        let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
        let per_entry = bytes as f64 / entries.len() as f64;
        assert!(
            per_entry <= 5.8,
            "{per_entry:.2} bytes per entry at 1,080 superfiles x 1M docs"
        );
        eprintln!("tier: {per_entry:.2} bytes per entry at 1,080 superfiles x 1M docs");
    }

    #[test]
    fn chunk_refs_round_trip_in_a_run_header() {
        let refs: Vec<TierChunkRef> = (0..3)
            .map(|i| TierChunkRef {
                offset: i * 1_000,
                len: 1_000 + i,
                n_entries: 40_000 * (i + 1),
            })
            .collect();
        let mut out = Vec::new();
        push_chunk_refs(&mut out, &refs);
        let mut at = 0;
        assert_eq!(read_chunk_refs(&out, &mut at).expect("read"), refs);
        assert_eq!(at, out.len());
        assert!(read_chunk_refs(&out[..out.len() - 1], &mut 0).is_err());
    }

    #[test]
    fn rejects_a_bad_crc_and_truncation() {
        let entries = sample(5_000, 10, 1_000, 5);
        let chunks = encode_tier(&entries, &keys(&entries));
        let bytes = &chunks[0].bytes;
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(
            matches!(
                TierChunk::verify(&flipped),
                Err(TermIndexError::HashMismatch)
            ),
            "a flipped body byte fails the crc"
        );
        TierChunk::verify(bytes).expect("intact bytes pass");
        assert!(TierChunk::open(&bytes[..bytes.len() - 1]).is_err());
        assert!(TierChunk::open(&bytes[..CHUNK_HEADER_LEN - 1]).is_err());
        let mut dir_flipped = bytes.clone();
        dir_flipped[CHUNK_HEADER_LEN] ^= 1;
        assert!(
            matches!(
                TierDirectory::parse(&dir_flipped),
                Err(TermIndexError::HashMismatch)
            ),
            "a flipped directory byte fails the directory crc"
        );
        let dir = TierDirectory::parse(bytes).expect("directory");
        let body = &bytes[dir.block_range(0)];
        let mut flipped_body = body.to_vec();
        flipped_body[BLOCK_HEADER_LEN] ^= 1;
        assert!(matches!(
            dir.verify_block(0, &flipped_body),
            Err(TermIndexError::HashMismatch)
        ));
        dir.verify_block(0, body).expect("intact body");
    }

    /// A reader sizes a chunk's directory from the entry count its run
    /// records and parses it from that prefix alone; the blocks it then
    /// fetches by the directory's ranges decode to the chunk's entries.
    #[test]
    fn a_directory_prefix_sized_from_the_entry_count_reads_the_blocks() {
        let entries = sample(30_000, 50, 100_000, 9);
        let chunks = encode_tier(&entries, &keys(&entries));
        let mut all = Vec::new();
        let mut buf = Vec::new();
        for c in &chunks {
            let dir_len = TierDirectory::len_for(c.n_entries);
            let dir = TierDirectory::parse(&c.bytes[..dir_len]).expect("directory from the prefix");
            assert_eq!(dir.chunk_len(), c.bytes.len());
            assert_eq!(
                dir.n_blocks(),
                (c.n_entries as usize).div_ceil(BLOCK_ENTRIES)
            );
            for i in 0..dir.n_blocks() {
                let body = &c.bytes[dir.block_range(i)];
                dir.verify_block(i, body).expect("crc");
                dir.decode_block(i, body, &mut buf).expect("decode");
                all.extend_from_slice(&buf);
            }
        }
        assert_eq!(all.len(), entries.len());
    }

    /// Most terms' tiers are a handful of entries; they must cost a handful
    /// of bytes, not a padded lane of each stream, and round-trip exactly.
    #[test]
    fn small_tiers_are_small_and_exact() {
        for n in [1usize, 2, 3, 7, 127, 129, 4095, 4097] {
            let entries = sample(n, 1_080, 1_000_000, n as u64);
            let chunks = encode_tier(&entries, &keys(&entries));
            let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
            let fixed = CHUNK_HEADER_LEN + DIRECTORY_ENTRY_LEN * chunks.len() + BLOCK_HEADER_LEN;
            assert!(
                bytes <= fixed * entries.len().div_ceil(BLOCK_ENTRIES) + 8 * n,
                "{n} entries took {bytes} bytes"
            );
            assert_eq!(
                sorted(&entries),
                sorted(&decode_all(&chunks)),
                "{n} entries round-trip"
            );
        }
    }

    /// The key is the factor a score multiplies the idf by, at the
    /// recorded average: a longer document scores lower, a higher term
    /// frequency higher, and the bucketed length is the one the superfile
    /// scores at.
    #[test]
    fn the_key_orders_like_the_score() {
        let s = scoring();
        assert!(s.key(2, 9) > s.key(1, 9), "more occurrences score higher");
        assert!(s.key(1, 9) > s.key(1, 40), "a longer document scores lower");
        let dl = dequantize_len(40);
        assert_eq!(
            s.key(1, 40),
            tf_factor(1, s.params.dl_norm_k1(dl, s.avgdl)),
            "the key is the length-normalized term frequency at the bucket"
        );
    }
}
