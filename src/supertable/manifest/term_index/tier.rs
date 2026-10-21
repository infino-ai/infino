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
//! carries the superfile's ordinal and the block's *tail*: the key of the
//! first entry past the block (or, past the head, the key of the best
//! document the head left out), so what a superfile can still hold beyond
//! what was read is bounded from the directory alone.
//!
//! The chunks of every term in a slice sit back to back in that slice's
//! tier object, so a run names its chunks by offset and block count. A
//! query reads a chunk's directory with one ranged fetch — sized from the
//! block count — and then only the blocks it consumes; the directory and
//! each block carry their own CRC, so a read of either verifies on its own.
//!
//! ```text
//! chunk: directory crc32c u32
//!        | per block: ordinal u32 | tail f32 | n_entries u8 | byte_len u16 | crc32c u32
//!        | block bodies back to back
//! block: lid_bits u8 | tf_bits u8 | len_bits u8
//!        | local ids, bit-packed | term frequencies, bit-packed | quantized lengths, bit-packed
//! ```
//!
//! A chunk's layout is versioned by the slice that names it, so it carries
//! no magic of its own. A block holds one superfile's entries, so the
//! superfile is named once, in the directory, and the body carries only
//! what differs per entry, each stream packed at the width its widest
//! value needs: most heads are a handful of entries, and a stream padded
//! to a fixed lane would cost them many times their size.
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
//! Keys and tails are at the entry's superfile's own average length and
//! the column's parameters ([`ColumnScoring`]); a reader scales them to
//! the table's average as it scales any stored ceiling.

use std::ops::Range;

use super::TermIndexError;
use crate::{
    superfile::{
        bits::{for_each_lane, payload_bytes, put_bits, width_of},
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
/// block at a time. Fits the directory's one-byte entry count.
pub(crate) const BLOCK_ENTRIES: usize = 64;
/// The chunk header: the directory's CRC. The block count is carried by
/// the run that names the chunk.
const CHUNK_HEADER_LEN: usize = 4;
const DIRECTORY_ENTRY_LEN: usize = 4 + 4 + 1 + 2 + 4;
/// The block header: the three stream widths.
const BLOCK_HEADER_LEN: usize = 3;
/// Widest a local id or term frequency stream may be.
const MAX_U32_BITS: u8 = u32::BITS as u8;
/// Widest a length-bucket stream may be.
const MAX_LEN_BITS: u8 = u8::BITS as u8;

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
/// tier object and how many blocks it holds, so a read of its directory is
/// sized before a byte of it is fetched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierChunkRef {
    pub(crate) offset: u64,
    pub(crate) len: u64,
    pub(crate) n_blocks: u32,
}

/// One encoded chunk before it is placed in a tier object.
pub(crate) struct EncodedChunk {
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

/// A chunk's block directory: which superfile each block belongs to, where
/// it sits in the chunk, its entry count, its tail and its CRC. What a
/// query reads first, to find its superfiles' blocks and fetch only those.
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
    /// Bound on every entry of the superfile past this block.
    tail: f32,
    n_entries: usize,
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
    let deepest = groups.iter().map(|g| g.entries.len()).max().unwrap_or(0);
    let mut chunks = Vec::with_capacity(deepest.div_ceil(BLOCK_ENTRIES));
    let mut bodies: Vec<u8> = Vec::new();
    let mut dirs: Vec<BlockDir> = Vec::new();
    for r in 0..deepest.div_ceil(BLOCK_ENTRIES) {
        let from = r * BLOCK_ENTRIES;
        bodies.clear();
        dirs.clear();
        for g in groups.iter().filter(|g| g.entries.len() > from) {
            let to = (from + BLOCK_ENTRIES).min(g.entries.len());
            let start = bodies.len();
            encode_block(&g.entries[from..to], &mut bodies);
            dirs.push(BlockDir {
                ordinal: g.ordinal,
                tail: g.keys.get(to).copied().unwrap_or(g.tail),
                n_entries: to - from,
                offset: start,
                len: bodies.len() - start,
                crc: crc32c(&bodies[start..]),
            });
        }
        chunks.push(encode_chunk(&dirs, &bodies));
    }
    chunks
}

/// One region: its directory, then its bodies.
fn encode_chunk(dirs: &[BlockDir], bodies: &[u8]) -> EncodedChunk {
    let mut out = Vec::with_capacity(TierDirectory::len_for(dirs.len() as u32) + bodies.len());
    out.extend_from_slice(&0u32.to_le_bytes());
    for d in dirs {
        out.extend_from_slice(&d.ordinal.to_le_bytes());
        out.extend_from_slice(&d.tail.to_le_bytes());
        out.push(d.n_entries as u8);
        out.extend_from_slice(&(d.len as u16).to_le_bytes());
        out.extend_from_slice(&d.crc.to_le_bytes());
    }
    let crc = crc32c(&out[CHUNK_HEADER_LEN..]);
    out[..CHUNK_HEADER_LEN].copy_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(bodies);
    EncodedChunk {
        n_blocks: dirs.len() as u32,
        bytes: out,
    }
}

/// Append one block's body — the entries of one superfile, sorted by local
/// id, as three packed streams — to `out`.
fn encode_block(block: &[TierEntry], out: &mut Vec<u8>) {
    debug_assert!(!block.is_empty() && block.len() <= BLOCK_ENTRIES);
    let n = block.len();
    let mut sorted = [TierEntry {
        superfile: 0,
        doc_id: 0,
        tf: 0,
        len: 0,
    }; BLOCK_ENTRIES];
    sorted[..n].copy_from_slice(block);
    let sorted = &mut sorted[..n];
    sorted.sort_unstable_by_key(|e| e.doc_id);
    let lid_bits = width_of(
        sorted
            .iter()
            .map(|e| u64::from(e.doc_id))
            .max()
            .unwrap_or(0),
    );
    let tf_bits = width_of(sorted.iter().map(|e| u64::from(e.tf)).max().unwrap_or(0));
    let len_bits = width_of(sorted.iter().map(|e| u64::from(e.len)).max().unwrap_or(0));
    out.push(lid_bits);
    out.push(tf_bits);
    out.push(len_bits);
    pack_stream(out, lid_bits, sorted.iter().map(|e| u64::from(e.doc_id)));
    pack_stream(out, tf_bits, sorted.iter().map(|e| u64::from(e.tf)));
    pack_stream(out, len_bits, sorted.iter().map(|e| u64::from(e.len)));
}

/// Append `values` at `bits` each, least significant bit first, no padding
/// between values; the stream ends on a byte boundary.
fn pack_stream(out: &mut Vec<u8>, bits: u8, values: impl ExactSizeIterator<Item = u64>) {
    let start = out.len();
    out.resize(start + payload_bytes(values.len(), bits), 0);
    for (i, v) in values.enumerate() {
        put_bits(&mut out[start..], i * bits as usize, v, bits);
    }
}

impl TierDirectory {
    /// Bytes the directory of a chunk of `n_blocks` takes, header included:
    /// what a reader fetches before it has read a byte of the chunk.
    pub(crate) fn len_for(n_blocks: u32) -> usize {
        CHUNK_HEADER_LEN + n_blocks as usize * DIRECTORY_ENTRY_LEN
    }

    /// Parse the directory of a chunk of `n_blocks` from its start; `bytes`
    /// may be the directory alone or the whole chunk. Verifies the
    /// directory's CRC.
    pub(crate) fn parse(bytes: &[u8], n_blocks: u32) -> Result<Self, TermIndexError> {
        let n_blocks = n_blocks as usize;
        let dir_end = Self::len_for(n_blocks as u32);
        if bytes.len() < dir_end {
            return Err(malformed("short directory"));
        }
        let stored = u32_le_at(bytes, 0).ok_or_else(|| malformed("crc"))?;
        if stored != crc32c(&bytes[CHUNK_HEADER_LEN..dir_end]) {
            return Err(TermIndexError::HashMismatch);
        }
        let mut blocks = Vec::with_capacity(n_blocks);
        let mut offset = dir_end;
        let mut prev_ordinal: Option<u32> = None;
        for i in 0..n_blocks {
            let at = CHUNK_HEADER_LEN + i * DIRECTORY_ENTRY_LEN;
            let entry = bytes
                .get(at..at + DIRECTORY_ENTRY_LEN)
                .ok_or_else(|| malformed("directory entry"))?;
            let ordinal = u32_le_at(entry, 0).ok_or_else(|| malformed("directory entry"))?;
            let tail =
                f32::from_bits(u32_le_at(entry, 4).ok_or_else(|| malformed("directory entry"))?);
            let n_entries = entry[8] as usize;
            let len = u16_le_at(entry, 9).ok_or_else(|| malformed("directory entry"))? as usize;
            let crc = u32_le_at(entry, 11).ok_or_else(|| malformed("directory entry"))?;
            if n_entries == 0 || n_entries > BLOCK_ENTRIES || len < BLOCK_HEADER_LEN {
                return Err(malformed("directory entry out of range"));
            }
            // Ascending ordinals, so a superfile's block is found by search.
            if prev_ordinal.is_some_and(|p| p >= ordinal) {
                return Err(malformed("directory ordinals out of order"));
            }
            prev_ordinal = Some(ordinal);
            blocks.push(BlockDir {
                ordinal,
                tail,
                n_entries,
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
    /// are in ascending doc id, all of the block's superfile.
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
        let (lid_bits, tf_bits, len_bits) = (body[0], body[1], body[2]);
        let n = d.n_entries;
        let need = BLOCK_HEADER_LEN
            + payload_bytes(n, lid_bits)
            + payload_bytes(n, tf_bits)
            + payload_bytes(n, len_bits);
        if body.len() != need
            || lid_bits > MAX_U32_BITS
            || tf_bits > MAX_U32_BITS
            || len_bits > MAX_LEN_BITS
        {
            return Err(malformed("block body length"));
        }
        let base = out.len();
        out.resize(
            base + n,
            TierEntry {
                superfile: d.ordinal,
                doc_id: 0,
                tf: 0,
                len: 0,
            },
        );
        let entries = &mut out[base..];
        let mut at = BLOCK_HEADER_LEN;
        let mut j = 0usize;
        for_each_lane(&body[at..], 0, n, lid_bits, |v| {
            entries[j].doc_id = v as u32;
            j += 1;
        })
        .ok_or_else(|| malformed("local id stream"))?;
        at += payload_bytes(n, lid_bits);
        j = 0;
        for_each_lane(&body[at..], 0, n, tf_bits, |v| {
            entries[j].tf = v as u32;
            j += 1;
        })
        .ok_or_else(|| malformed("term frequency stream"))?;
        at += payload_bytes(n, tf_bits);
        j = 0;
        for_each_lane(&body[at..], 0, n, len_bits, |v| {
            entries[j].len = v as u8;
            j += 1;
        })
        .ok_or_else(|| malformed("length stream"))?;
        Ok(())
    }
}

impl<'a> TierChunk<'a> {
    /// Check a whole chunk: its directory's CRC and every block's.
    #[cfg(test)]
    pub(crate) fn verify(bytes: &'a [u8], n_blocks: u32) -> Result<(), TermIndexError> {
        let chunk = Self::open(bytes, n_blocks)?;
        for i in 0..chunk.n_blocks() {
            chunk
                .directory
                .verify_block(i, &bytes[chunk.directory.block_range(i)])?;
        }
        Ok(())
    }

    /// Parse a whole chunk's directory and check its bodies fit it.
    pub(crate) fn open(bytes: &'a [u8], n_blocks: u32) -> Result<Self, TermIndexError> {
        let directory = TierDirectory::parse(bytes, n_blocks)?;
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

    /// Every entry of the chunk, block by block, appended to `out`.
    pub(crate) fn entries(&self, out: &mut Vec<TierEntry>) -> Result<(), TermIndexError> {
        let mut block = Vec::new();
        for i in 0..self.n_blocks() {
            self.decode_block(i, &mut block)?;
            out.extend_from_slice(&block);
        }
        Ok(())
    }
}

/// Serialize a chunk list into a run header.
pub(crate) fn push_chunk_refs(out: &mut Vec<u8>, chunks: &[TierChunkRef]) {
    push_varint(out, chunks.len() as u32);
    for c in chunks {
        push_u64_varint(out, c.offset);
        push_u64_varint(out, c.len);
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
        let n_blocks = read_varint(bytes, at).ok_or_else(|| malformed("chunk blocks"))?;
        out.push(TierChunkRef {
            offset,
            len,
            n_blocks,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

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
                // One entry per document, in descending key.
                entries.sort_unstable_by_key(|e| e.doc_id);
                entries.dedup_by_key(|e| e.doc_id);
                entries.sort_by(|a, b| {
                    key_of(b)
                        .total_cmp(&key_of(a))
                        .then(a.doc_id.cmp(&b.doc_id))
                });
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
        for c in chunks {
            TierChunk::verify(&c.bytes, c.n_blocks).expect("crc");
            TierChunk::open(&c.bytes, c.n_blocks)
                .expect("open")
                .entries(&mut all)
                .expect("decode");
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

    /// A decoded block is one superfile's entries in ascending doc id:
    /// what a merge of two terms' blocks walks linearly.
    #[test]
    fn a_block_decodes_to_its_superfile_in_ascending_doc_id() {
        let groups = sample(100, 9, 50_000, 13);
        let chunks = encode_tier(&groups);
        let mut block = Vec::new();
        for c in &chunks {
            let chunk = TierChunk::open(&c.bytes, c.n_blocks).expect("open");
            for i in 0..chunk.n_blocks() {
                chunk.decode_block(i, &mut block).expect("decode");
                let ordinal = chunk.directory.block_ordinal(i);
                assert!(block.iter().all(|e| e.superfile == ordinal));
                assert!(block.windows(2).all(|w| w[0].doc_id < w[1].doc_id));
            }
        }
    }

    /// Region `r` holds block `r` of every superfile deep enough, in
    /// ascending ordinal; a block's tail is the key after it, or the head's
    /// own tail past the last block, and never exceeds the keys before it.
    #[test]
    fn regions_hold_one_block_per_superfile_with_exact_tails() {
        let groups = sample(150, 20, 500_000, 11);
        let chunks = encode_tier(&groups);
        assert_eq!(
            chunks.len(),
            150usize.div_ceil(BLOCK_ENTRIES),
            "one region per block depth"
        );
        for (r, c) in chunks.iter().enumerate() {
            let dir = TierDirectory::parse(&c.bytes, c.n_blocks).expect("directory");
            let from = r * BLOCK_ENTRIES;
            let deep: Vec<&HeadGroup> = groups.iter().filter(|g| g.entries.len() > from).collect();
            assert_eq!(dir.n_blocks(), deep.len(), "region {r}");
            for (i, g) in deep.iter().enumerate() {
                assert_eq!(dir.block_ordinal(i), g.ordinal);
                assert_eq!(dir.block_of(g.ordinal), Some(i));
                let to = (from + BLOCK_ENTRIES).min(g.entries.len());
                let want_tail = g.keys.get(to).copied().unwrap_or(g.tail);
                assert_eq!(
                    dir.block_tail(i),
                    want_tail,
                    "region {r} ordinal {}",
                    g.ordinal
                );
                assert!(
                    dir.block_tail(i) <= g.keys[to - 1],
                    "a tail never exceeds the keys before it"
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

    /// Bytes per entry on a billion-document shape: a head block names its
    /// superfile once and packs its rows and facts to a few bits each.
    #[test]
    fn heads_cost_a_few_bytes_per_entry() {
        let groups = sample(512, 1_080, 1_000_000, 3);
        let chunks = encode_tier(&groups);
        let entries: usize = groups.iter().map(|g| g.entries.len()).sum();
        let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
        let per_entry = bytes as f64 / entries as f64;
        assert!(
            per_entry <= 4.3,
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
        let (bytes, n_blocks) = (&chunks[0].bytes, chunks[0].n_blocks);
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(
            matches!(
                TierChunk::verify(&flipped, n_blocks),
                Err(TermIndexError::HashMismatch)
            ),
            "a flipped body byte fails the crc"
        );
        TierChunk::verify(bytes, n_blocks).expect("intact bytes pass");
        assert!(TierChunk::open(&bytes[..bytes.len() - 1], n_blocks).is_err());
        assert!(TierChunk::open(&bytes[..CHUNK_HEADER_LEN - 1], n_blocks).is_err());
        assert!(
            TierDirectory::parse(bytes, n_blocks + 1).is_err(),
            "a block count the directory does not hold"
        );
        let mut dir_flipped = bytes.clone();
        dir_flipped[CHUNK_HEADER_LEN] ^= 1;
        assert!(
            matches!(
                TierDirectory::parse(&dir_flipped, n_blocks),
                Err(TermIndexError::HashMismatch)
            ),
            "a flipped directory byte fails the directory crc"
        );
        let dir = TierDirectory::parse(bytes, n_blocks).expect("directory");
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
            let dir = TierDirectory::parse(&c.bytes[..dir_len], c.n_blocks)
                .expect("directory from the prefix");
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        /// Any shape of heads — a handful of superfiles or many, shallow or
        /// several blocks deep, rows anywhere in a superfile — round-trips
        /// entry for entry, and every directory entry names its superfile,
        /// its block's entry count and the exact tail the build gave it.
        #[test]
        fn any_heads_round_trip_with_exact_directories(
            per_sf in 1usize..300,
            superfiles in 1u32..40,
            docs in 1u32..2_000_000,
            seed in any::<u64>(),
        ) {
            let groups = sample(per_sf, superfiles, docs, seed | 1);
            let chunks = encode_tier(&groups);
            let deepest = groups.iter().map(|g| g.entries.len()).max().unwrap_or(0);
            prop_assert_eq!(chunks.len(), deepest.div_ceil(BLOCK_ENTRIES));
            prop_assert_eq!(sorted(&decode_all(&chunks)), sorted(&all_entries(&groups)));
            for (r, c) in chunks.iter().enumerate() {
                let from = r * BLOCK_ENTRIES;
                let dir = TierDirectory::parse(&c.bytes[..TierDirectory::len_for(c.n_blocks)], c.n_blocks)
                    .expect("directory from its prefix");
                prop_assert_eq!(dir.chunk_len(), c.bytes.len());
                let deep: Vec<&HeadGroup> = groups.iter().filter(|g| g.entries.len() > from).collect();
                prop_assert_eq!(dir.n_blocks(), deep.len());
                let mut block = Vec::new();
                for (i, g) in deep.iter().enumerate() {
                    let to = (from + BLOCK_ENTRIES).min(g.entries.len());
                    prop_assert_eq!(dir.block_ordinal(i), g.ordinal);
                    prop_assert_eq!(dir.block_of(g.ordinal), Some(i));
                    prop_assert_eq!(dir.block_tail(i), g.keys.get(to).copied().unwrap_or(g.tail));
                    let body = &c.bytes[dir.block_range(i)];
                    dir.verify_block(i, body).expect("crc");
                    dir.decode_block(i, body, &mut block).expect("decode");
                    prop_assert_eq!(block.len(), to - from);
                    prop_assert!(block.iter().all(|e| e.superfile == g.ordinal));
                    prop_assert!(block.windows(2).all(|w| w[0].doc_id < w[1].doc_id));
                    prop_assert_eq!(sorted(&block), sorted(&g.entries[from..to]));
                }
            }
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
