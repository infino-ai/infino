// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term's *tier*: per superfile, the term's best documents there — its
//! *head* — as `(local doc id, term frequency, quantized length)`, the
//! facts a score is computed from, in descending *key* (the length-
//! normalized term frequency `tf / (tf + k1·(1 − b + b·dl/avgdl))`, the
//! factor a BM25 score multiplies the term's idf by). A head is cut into
//! blocks of [`BLOCK_ENTRIES`]; the chunks of a term's tier are *regions*:
//! region `r` holds block `r` of every superfile whose head reaches that
//! deep, in ascending superfile ordinal, so a query reads the first block
//! of every superfile with one ranged fetch, and deepens one superfile at a
//! time with a fetch of its next block alone. Each block's directory entry
//! carries the superfile's ordinal, the block's largest key and its *tail*:
//! the key of the first entry past the block (or, past the head, the key
//! of the best document the head left out), so what a superfile can still
//! hold beyond what was read is bounded from the directory alone.
//!
//! The chunks of every term in a slice sit back to back in that slice's
//! tier object, so a run names its chunks by offset. A query reads a
//! chunk's directory with one ranged fetch — sized from the block count the
//! run records — and then only the blocks it consumes; the directory and
//! each block carry their own CRC, so a read of either verifies on its own.
//!
//! ```text
//! chunk: n_blocks u32 | directory crc32c u32
//!        | per block: ordinal u32 | max_key f32 | tail f32 | n_entries u16 | byte_len u32 | crc32c u32
//!        | block bodies back to back
//! block: ordinal_bits u8 | lid_bits u8 | tf_bits u8 | len_bits u8
//!        | ordinals, bit-packed sorted deltas | local ids, bit-packed
//!        | term frequencies, bit-packed | quantized lengths, bit-packed
//! ```
//!
//! A chunk's layout is versioned by the slice that names it, so it carries
//! no magic of its own. Each stream is packed in lanes of [`LANE`] values
//! by the SIMD packer, and the block's final partial lane densely at the
//! same width: most heads are a handful of entries, and a lane padded to
//! full width would cost them a hundred times their size. A block holds
//! one superfile's entries, so its ordinal stream packs to nothing.
//!
//! Inside a block the entries are stored sorted by local id — the block is
//! always consumed whole, and a sorted block merges with another term's by
//! a linear walk.
//!
//! An entry stores facts, not a score: a reader computes the document's
//! score for a query exactly as the superfile's kernel would — the table's
//! idf times `tf / (tf + norm)` with the same quantized length and average
//! — so a candidate the tier names needs no opening to be scored, and a
//! merge re-cuts tiers at any average length without loosening a bound.
//! Keys, maxima and tails are at the entry's superfile's own average length
//! and the column's parameters ([`ColumnScoring`]); a reader scales them to
//! the table's average as it scales any stored ceiling.

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

/// Entries per block: one superfile's slice of its head, the unit a reader
/// consumes at a time. A query's first round reads one block per superfile
/// per term, so this sets that round's cost; its later rounds deepen one
/// block at a time.
pub(crate) const BLOCK_ENTRIES: usize = 64;
const LANE: usize = BitPacker4x::BLOCK_LEN;
const CHUNK_HEADER_LEN: usize = 4 + 4;
/// Where the directory's CRC sits in the header: over the directory
/// entries that follow it.
const CRC_AT: usize = 4;
const DIRECTORY_ENTRY_LEN: usize = 4 + 4 + 4 + 2 + 4 + 4;
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

/// Where one chunk — one region — of a term's tier sits in its slice's
/// tier object, how many entries and blocks it holds, so a read of its
/// directory is sized before a byte of it is fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierChunkRef {
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) n_entries: u64,
    pub(crate) n_blocks: u32,
}

/// One encoded chunk before it is placed in a tier object.
pub(crate) struct EncodedChunk {
    pub(crate) n_entries: u64,
    pub(crate) n_blocks: u32,
    pub(crate) bytes: Vec<u8>,
}

/// One superfile's head of a term, as the build hands it to the encoder:
/// entries in descending key with their keys, and the tail — the key of the
/// best document of the superfile the head leaves out, zero when none.
pub(crate) struct HeadGroup {
    pub(crate) ordinal: u32,
    pub(crate) entries: Vec<TierEntry>,
    pub(crate) keys: Vec<f32>,
    pub(crate) tail: f32,
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
    /// The superfile whose head the block is a slice of.
    ordinal: u32,
    max_key: f32,
    /// Bound on every entry of the superfile past this block.
    tail: f32,
    n_entries: u32,
    /// Byte offset of the body from the chunk's start.
    offset: usize,
    len: usize,
    crc: u32,
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(format!("tier chunk: {what}"))
}

/// Cut each superfile's head into blocks and lay the blocks out in regions:
/// region `r` holds block `r` of every group that reaches that deep, in
/// ascending ordinal. `groups` must be in ascending ordinal, each in
/// descending key. Returns one encoded chunk per region, in order.
pub(crate) fn encode_tier(groups: &[HeadGroup]) -> Vec<EncodedChunk> {
    debug_assert!(
        groups.windows(2).all(|w| w[0].ordinal < w[1].ordinal),
        "head groups must be in ascending ordinal"
    );
    debug_assert!(
        groups
            .iter()
            .all(|g| g.keys.len() == g.entries.len() && g.keys.windows(2).all(|w| w[0] >= w[1])),
        "a head is in descending key"
    );
    let mut chunks = Vec::new();
    let mut scratch = vec![0u8; LANE * 4];
    let deepest = groups.iter().map(|g| g.entries.len()).max().unwrap_or(0);
    for r in 0..deepest.div_ceil(BLOCK_ENTRIES) {
        let from = r * BLOCK_ENTRIES;
        let mut blocks: Vec<(BlockDir, Vec<u8>)> = Vec::new();
        for g in groups.iter().filter(|g| g.entries.len() > from) {
            let to = (from + BLOCK_ENTRIES).min(g.entries.len());
            let (mut dir, body) =
                encode_block(&g.entries[from..to], &g.keys[from..to], &mut scratch);
            dir.ordinal = g.ordinal;
            dir.tail = g.keys.get(to).copied().unwrap_or(g.tail);
            blocks.push((dir, body));
        }
        chunks.push(encode_chunk(blocks));
    }
    chunks
}

/// One region: its directory, then its bodies.
fn encode_chunk(mut blocks: Vec<(BlockDir, Vec<u8>)>) -> EncodedChunk {
    let n_entries: u64 = blocks.iter().map(|(d, _)| d.n_entries as u64).sum();
    let mut out = Vec::with_capacity(
        CHUNK_HEADER_LEN
            + blocks.len() * DIRECTORY_ENTRY_LEN
            + blocks.iter().map(|(_, b)| b.len()).sum::<usize>(),
    );
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for (d, body) in blocks.iter() {
        out.extend_from_slice(&d.ordinal.to_le_bytes());
        out.extend_from_slice(&d.max_key.to_le_bytes());
        out.extend_from_slice(&d.tail.to_le_bytes());
        out.extend_from_slice(&(d.n_entries as u16).to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32c(body).to_le_bytes());
    }
    let crc = crc32c(&out[CHUNK_HEADER_LEN..]);
    out[CRC_AT..CRC_AT + 4].copy_from_slice(&crc.to_le_bytes());
    let n_blocks = blocks.len() as u32;
    for (_, body) in blocks.drain(..) {
        out.extend_from_slice(&body);
    }
    EncodedChunk {
        n_entries,
        n_blocks,
        bytes: out,
    }
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
            ordinal: 0,
            max_key,
            tail: 0.0,
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
    /// Bytes the directory of a chunk of `n_blocks` takes, header included:
    /// what a reader fetches before it has read a byte of the chunk.
    pub(crate) fn len_for(n_blocks: u32) -> usize {
        CHUNK_HEADER_LEN + n_blocks as usize * DIRECTORY_ENTRY_LEN
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
        let mut prev_ordinal: Option<u32> = None;
        for i in 0..n_blocks {
            let at = CHUNK_HEADER_LEN + i * DIRECTORY_ENTRY_LEN;
            let f = |o: usize| u32_le_at(bytes, at + o).ok_or_else(|| malformed("directory entry"));
            let ordinal = f(0)?;
            let max_key = f32::from_bits(f(4)?);
            let tail = f32::from_bits(f(8)?);
            let n_entries =
                u16_le_at(bytes, at + 12).ok_or_else(|| malformed("directory entry"))?;
            let len = f(14)? as usize;
            let crc = f(18)?;
            if n_entries == 0 || n_entries as usize > BLOCK_ENTRIES || len < BLOCK_HEADER_LEN {
                return Err(malformed("directory entry out of range"));
            }
            // Ascending ordinals, so a superfile's block is found by search.
            if prev_ordinal.is_some_and(|p| p >= ordinal) {
                return Err(malformed("directory ordinals out of order"));
            }
            prev_ordinal = Some(ordinal);
            blocks.push(BlockDir {
                ordinal,
                max_key,
                tail,
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

    /// Exact largest key in block `i`, at its superfile's [`ColumnScoring`].
    #[cfg(test)]
    pub(crate) fn block_max(&self, i: usize) -> f32 {
        self.blocks[i].max_key
    }

    /// The superfile ordinal block `i` belongs to.
    #[cfg(test)]
    pub(crate) fn block_ordinal(&self, i: usize) -> u32 {
        self.blocks[i].ordinal
    }

    /// Bound on every entry of block `i`'s superfile past the block.
    pub(crate) fn block_tail(&self, i: usize) -> f32 {
        self.blocks[i].tail
    }

    /// The block of superfile `ordinal` in this chunk, if it has one.
    pub(crate) fn block_of(&self, ordinal: u32) -> Option<usize> {
        self.blocks
            .binary_search_by_key(&ordinal, |b| b.ordinal)
            .ok()
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
        push_varint(out, c.n_blocks);
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
        let n_blocks = read_varint(bytes, at).ok_or_else(|| malformed("chunk blocks"))?;
        out.push(TierChunkRef {
            offset,
            len,
            n_entries,
            n_blocks,
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

    /// A deterministic sample: heads of `per_sf` entries for each of
    /// `superfiles` superfiles of `docs` documents, small term frequencies
    /// and lengths across the whole bucket range, each head in descending
    /// key with a tail below its last key.
    fn sample(per_sf: usize, superfiles: u32, docs: u32, seed: u64) -> Vec<HeadGroup> {
        let mut x = seed;
        let mut next = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        (0..superfiles)
            .map(|sf| {
                let mut entries: Vec<TierEntry> = (0..per_sf)
                    .map(|_| TierEntry {
                        superfile: sf,
                        doc_id: (next() % docs as u64) as u32,
                        tf: 1 + (next() % 4) as u32,
                        len: (next() % 256) as u8,
                    })
                    .collect();
                entries.sort_by(|a, b| {
                    key_of(b)
                        .total_cmp(&key_of(a))
                        .then(a.doc_id.cmp(&b.doc_id))
                });
                entries.dedup_by_key(|e| e.doc_id);
                let keys: Vec<f32> = entries.iter().map(key_of).collect();
                let tail = keys.last().map_or(0.0, |k| k * 0.5);
                HeadGroup {
                    ordinal: sf,
                    entries,
                    keys,
                    tail,
                }
            })
            .collect()
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

    fn all_entries(groups: &[HeadGroup]) -> Vec<TierEntry> {
        groups
            .iter()
            .flat_map(|g| g.entries.iter().copied())
            .collect()
    }

    #[test]
    fn round_trips_every_entry_exactly() {
        let groups = sample(200, 37, 1_000_000, 7);
        let chunks = encode_tier(&groups);
        let decoded = decode_all(&chunks);
        assert_eq!(
            sorted(&all_entries(&groups)),
            sorted(&decoded),
            "entries are exact"
        );
    }

    /// Region `r` holds block `r` of every superfile deep enough, in
    /// ascending ordinal; a block's maximum is its first key and its tail
    /// the key after it, or the head's own tail past the last block.
    #[test]
    fn regions_hold_one_block_per_superfile_with_exact_maxima_and_tails() {
        let groups = sample(150, 20, 500_000, 11);
        let chunks = encode_tier(&groups);
        assert_eq!(
            chunks.len(),
            150usize.div_ceil(BLOCK_ENTRIES),
            "one region per block depth"
        );
        for (r, c) in chunks.iter().enumerate() {
            let dir = TierDirectory::parse(&c.bytes).expect("directory");
            let from = r * BLOCK_ENTRIES;
            let deep: Vec<&HeadGroup> = groups.iter().filter(|g| g.entries.len() > from).collect();
            assert_eq!(dir.n_blocks(), deep.len(), "region {r}");
            for (i, g) in deep.iter().enumerate() {
                assert_eq!(dir.block_ordinal(i), g.ordinal);
                assert_eq!(dir.block_of(g.ordinal), Some(i));
                assert_eq!(
                    dir.block_max(i),
                    g.keys[from],
                    "region {r} ordinal {}",
                    g.ordinal
                );
                let to = (from + BLOCK_ENTRIES).min(g.entries.len());
                let want_tail = g.keys.get(to).copied().unwrap_or(g.tail);
                assert_eq!(
                    dir.block_tail(i),
                    want_tail,
                    "region {r} ordinal {}",
                    g.ordinal
                );
                assert!(
                    dir.block_tail(i) <= dir.block_max(i),
                    "a tail never exceeds its block's maximum"
                );
            }
            assert_eq!(dir.block_of(u32::MAX), None, "an ordinal with no block");
        }
    }

    #[test]
    fn a_constant_key_block_round_trips_exactly() {
        let entries: Vec<TierEntry> = (0..300)
            .map(|i| TierEntry {
                superfile: 3,
                doc_id: i * 7,
                tf: 2,
                len: 9,
            })
            .collect();
        let keys = vec![key_of(&entries[0]); entries.len()];
        let groups = vec![HeadGroup {
            ordinal: 3,
            entries,
            keys,
            tail: 0.0,
        }];
        let chunks = encode_tier(&groups);
        assert_eq!(chunks.len(), 300usize.div_ceil(BLOCK_ENTRIES));
        let decoded = decode_all(&chunks);
        assert!(
            decoded
                .iter()
                .all(|e| e.tf == 2 && e.len == 9 && e.superfile == 3)
        );
        assert_eq!(decoded.len(), 300);
    }

    /// Bytes per entry on a billion-document shape: a head block of one
    /// superfile packs its ordinal to nothing and its rows and facts to a
    /// few bits each.
    #[test]
    fn heads_cost_a_few_bytes_per_entry() {
        let groups = sample(512, 1_080, 1_000_000, 3);
        let chunks = encode_tier(&groups);
        let entries: usize = groups.iter().map(|g| g.entries.len()).sum();
        let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
        let per_entry = bytes as f64 / entries as f64;
        assert!(
            per_entry <= 5.5,
            "{per_entry:.2} bytes per entry at 1,080 superfiles x 512 deep"
        );
        eprintln!("heads: {per_entry:.2} bytes per entry at 1,080 superfiles x 512 deep");
    }

    #[test]
    fn chunk_refs_round_trip_in_a_run_header() {
        let refs: Vec<TierChunkRef> = (0..3)
            .map(|i| TierChunkRef {
                offset: i * 1_000,
                len: 1_000 + i,
                n_entries: 40_000 * (i + 1),
                n_blocks: 1_078 - i as u32,
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
        let chunks = encode_tier(&sample(80, 10, 1_000, 5));
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

    /// A reader sizes a region's directory from the block count its run
    /// records and parses it from that prefix alone; the blocks it then
    /// fetches by the directory's ranges decode to the region's entries.
    #[test]
    fn a_directory_prefix_sized_from_the_block_count_reads_the_blocks() {
        let groups = sample(130, 50, 100_000, 9);
        let chunks = encode_tier(&groups);
        let mut all = Vec::new();
        let mut buf = Vec::new();
        for c in &chunks {
            let dir_len = TierDirectory::len_for(c.n_blocks);
            let dir = TierDirectory::parse(&c.bytes[..dir_len]).expect("directory from the prefix");
            assert_eq!(dir.chunk_len(), c.bytes.len());
            assert_eq!(dir.n_blocks(), c.n_blocks as usize);
            for i in 0..dir.n_blocks() {
                let body = &c.bytes[dir.block_range(i)];
                dir.verify_block(i, body).expect("crc");
                dir.decode_block(i, body, &mut buf).expect("decode");
                all.extend_from_slice(&buf);
            }
        }
        assert_eq!(sorted(&all), sorted(&all_entries(&groups)));
    }

    /// Most heads are a handful of entries; they must cost a handful of
    /// bytes, not a padded lane of each stream, and round-trip exactly.
    #[test]
    fn small_heads_are_small_and_exact() {
        for n in [1usize, 2, 3, 7, 63, 64, 65, 127] {
            let groups = sample(n, 3, 1_000_000, n as u64);
            let chunks = encode_tier(&groups);
            let entries: usize = groups.iter().map(|g| g.entries.len()).sum();
            let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
            let blocks: usize = chunks.iter().map(|c| c.n_blocks as usize).sum();
            let fixed =
                CHUNK_HEADER_LEN * chunks.len() + (DIRECTORY_ENTRY_LEN + BLOCK_HEADER_LEN) * blocks;
            assert!(
                bytes <= fixed + 8 * entries,
                "{n} entries per head took {bytes} bytes"
            );
            assert_eq!(
                sorted(&all_entries(&groups)),
                sorted(&decode_all(&chunks)),
                "{n} round-trip"
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
