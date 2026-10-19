// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term's *tier*: its best documents table-wide, by impact, so a ranked
//! query can find the documents that hold several of its terms — and
//! bound every superfile — before opening any superfile.
//!
//! A tier is a sequence of entries `(superfile ordinal, local doc id,
//! impact)` in descending impact, cut into blocks of [`BLOCK_ENTRIES`]
//! and the blocks into *chunks* of at most [`CHUNK_TARGET_BYTES`], read
//! in order and usually only in part: a query stops once the blocks it
//! has read bound everything it has not. The chunks of every term in a
//! slice sit back to back in that slice's tier object, so a run names its
//! chunks by offset and a query reads one with a ranged fetch; the CRC in
//! the chunk header is what verifies a chunk read on its own.
//!
//! ```text
//! chunk: n_blocks u32 | crc32c u32
//!        | per block: max_impact f32 | n_entries u16 | byte_len u32
//!        | block bodies back to back
//! block: ordinal_bits u8 | lid_bits u8 | impact_bits u8
//!        | ordinals, bit-packed sorted deltas | local ids, bit-packed
//!        | impacts as `max_impact_bits - impact_bits`, bit-packed
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
//! no information and the sorted ordinals pack to a few bits. Impacts are
//! exact: a block spans a narrow range of positive floats, whose bit
//! patterns order like the values, so each is stored as its distance
//! below the block's maximum pattern in the bits that distance needs.
//! Exactness is what lets a merge re-cut tiers without loosening a bound
//! or changing a byte, and what lets the directory's `max_impact` — which
//! bounds every entry of its block and of the blocks after it — stay
//! exact across any number of merges.

use bitpacking::{BitPacker, BitPacker4x};

use super::TermIndexError;
use crate::{
    superfile::format::checksum::crc32c,
    utils::{
        bytes::{u16_le_at, u32_le_at},
        varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
    },
};

/// Entries per block, the unit a reader consumes at a time. A multiple of
/// the bit-packer's lane width.
pub(crate) const BLOCK_ENTRIES: usize = 4096;
/// Bytes a chunk grows to before the next block starts a new one. A query
/// at small `k` reads one or two chunks per term.
pub(crate) const CHUNK_TARGET_BYTES: usize = 4 * 1024 * 1024;
const LANE: usize = BitPacker4x::BLOCK_LEN;
const CHUNK_HEADER_LEN: usize = 4 + 4;
/// Where the chunk's CRC sits in its header: over everything after it.
const CRC_AT: usize = 4;
const DIRECTORY_ENTRY_LEN: usize = 4 + 2 + 4;
const BLOCK_HEADER_LEN: usize = 3;

/// One tier entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TierEntry {
    /// Ordinal into the root's superfile list.
    pub(crate) superfile: u32,
    /// Row within that superfile.
    pub(crate) doc_id: u32,
    /// BM25 impact of the term in that document, at the superfile's own
    /// statistics. Exact.
    pub(crate) impact: f32,
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

/// A chunk's bytes with its block directory parsed.
pub(crate) struct TierChunk<'a> {
    directory: Vec<BlockDir>,
    bodies: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
struct BlockDir {
    max_impact: f32,
    n_entries: u32,
    offset: usize,
    len: usize,
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(format!("tier chunk: {what}"))
}

/// Cut `entries` — in descending impact — into blocks and chunks. Returns
/// each chunk's bytes with its entry count, in tier order.
pub(crate) fn encode_tier(entries: &[TierEntry]) -> Vec<EncodedChunk> {
    debug_assert!(
        entries.windows(2).all(|w| w[0].impact >= w[1].impact),
        "tier entries must be in descending impact"
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
            out.extend_from_slice(&d.max_impact.to_le_bytes());
            out.extend_from_slice(&(d.n_entries as u16).to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        }
        for (_, body) in blocks.drain(..) {
            out.extend_from_slice(&body);
        }
        let crc = crc32c(&out[CHUNK_HEADER_LEN..]);
        out[CRC_AT..CRC_AT + 4].copy_from_slice(&crc.to_le_bytes());
        chunks.push(EncodedChunk {
            n_entries,
            bytes: out,
        });
    };
    for block in entries.chunks(BLOCK_ENTRIES) {
        let (dir, body) = encode_block(block, &mut scratch);
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

fn encode_block(block: &[TierEntry], scratch: &mut [u8]) -> (BlockDir, Vec<u8>) {
    debug_assert!(!block.is_empty() && block.len() <= BLOCK_ENTRIES);
    let max_impact = block
        .iter()
        .map(|e| e.impact)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sorted: Vec<TierEntry> = block.to_vec();
    sorted.sort_unstable_by_key(|e| (e.superfile, e.doc_id));
    let n = sorted.len();
    let full = n / LANE * LANE;
    let mut ordinals = vec![0u32; n];
    let mut doc_ids = vec![0u32; n];
    let mut codes = vec![0u32; n];
    debug_assert!(
        block.iter().all(|e| e.impact >= 0.0),
        "impacts are non-negative, so their bit patterns order like their values"
    );
    let max_bits = max_impact.to_bits();
    for (i, e) in sorted.iter().enumerate() {
        ordinals[i] = e.superfile;
        doc_ids[i] = e.doc_id;
        codes[i] = max_bits - e.impact.to_bits();
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
    let impact_bits = codes[..full]
        .chunks(LANE)
        .map(|lane| bp.num_bits(lane))
        .max()
        .unwrap_or(0)
        .max(width(&codes[full..]));
    let mut body = Vec::with_capacity(
        BLOCK_HEADER_LEN
            + n * (ordinal_bits as usize + lid_bits as usize + impact_bits as usize) / 8
            + 3,
    );
    body.push(ordinal_bits);
    body.push(lid_bits);
    body.push(impact_bits);
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
    for lane in codes[..full].chunks(LANE) {
        let written = bp.compress(lane, scratch, impact_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    pack_dense(&codes[full..], impact_bits, &mut body);
    (
        BlockDir {
            max_impact,
            n_entries: n as u32,
            offset: 0,
            len: body.len(),
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

impl<'a> TierChunk<'a> {
    /// Check a chunk's CRC: what a chunk fetched by range, without its
    /// object's hash, is verified by. Once per fetch, not per open.
    pub(crate) fn verify(bytes: &[u8]) -> Result<(), TermIndexError> {
        if bytes.len() < CHUNK_HEADER_LEN {
            return Err(malformed("short header"));
        }
        let stored = u32_le_at(bytes, CRC_AT).ok_or_else(|| malformed("crc"))?;
        if stored != crc32c(&bytes[CHUNK_HEADER_LEN..]) {
            return Err(TermIndexError::HashMismatch);
        }
        Ok(())
    }

    /// Parse a chunk's directory.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
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
        let mut directory = Vec::with_capacity(n_blocks);
        let mut offset = 0usize;
        for i in 0..n_blocks {
            let at = CHUNK_HEADER_LEN + i * DIRECTORY_ENTRY_LEN;
            let f = |o: usize| u32_le_at(bytes, at + o).ok_or_else(|| malformed("directory entry"));
            let max_impact = f32::from_bits(f(0)?);
            let n_entries = u16_le_at(bytes, at + 4).ok_or_else(|| malformed("directory entry"))?;
            let len = f(6)? as usize;
            if n_entries == 0 || n_entries as usize > BLOCK_ENTRIES || len < BLOCK_HEADER_LEN {
                return Err(malformed("directory entry out of range"));
            }
            directory.push(BlockDir {
                max_impact,
                n_entries: u32::from(n_entries),
                offset,
                len,
            });
            offset = offset
                .checked_add(len)
                .ok_or_else(|| malformed("block offsets"))?;
        }
        let bodies = &bytes[dir_end..];
        if bodies.len() != offset {
            return Err(malformed("body length does not match the directory"));
        }
        Ok(Self { directory, bodies })
    }

    pub(crate) fn n_blocks(&self) -> usize {
        self.directory.len()
    }

    /// Exact largest impact in block `i`.
    pub(crate) fn block_max(&self, i: usize) -> f32 {
        self.directory[i].max_impact
    }

    #[cfg(test)]
    pub(crate) fn block_entries(&self, i: usize) -> usize {
        self.directory[i].n_entries as usize
    }

    /// Decode block `i` into `out` (cleared first); entries are in
    /// `(ordinal, doc id)` order.
    pub(crate) fn decode_block(
        &self,
        i: usize,
        out: &mut Vec<TierEntry>,
    ) -> Result<(), TermIndexError> {
        out.clear();
        let d = self
            .directory
            .get(i)
            .ok_or_else(|| malformed("block index"))?;
        let body = &self.bodies[d.offset..d.offset + d.len];
        let (ordinal_bits, lid_bits, impact_bits) = (body[0], body[1], body[2]);
        let max_bits = d.max_impact.to_bits();
        let n = d.n_entries as usize;
        let full = n / LANE * LANE;
        let rest = n - full;
        let lanes = full / LANE;
        let stream = |bits: u8| lanes * LANE * bits as usize / 8 + dense_len(rest, bits);
        let need = BLOCK_HEADER_LEN + stream(ordinal_bits) + stream(lid_bits) + stream(impact_bits);
        if body.len() != need || ordinal_bits > 32 || lid_bits > 32 || impact_bits > 32 {
            return Err(malformed("block body length"));
        }
        let bp = BitPacker4x::new();
        let mut ordinals = vec![0u32; n];
        let mut doc_ids = vec![0u32; n];
        let mut codes = vec![0u32; n];
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
        for lane in codes[..full].chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, impact_bits);
        }
        unpack_dense(&body[at..], rest, impact_bits, &mut codes[full..]);
        at += dense_len(rest, impact_bits);
        debug_assert_eq!(at, body.len());
        out.reserve(n);
        for i in 0..n {
            out.push(TierEntry {
                superfile: ordinals[i],
                doc_id: doc_ids[i],
                impact: f32::from_bits(max_bits.wrapping_sub(codes[i])),
            });
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

    fn descending(entries: &mut [TierEntry]) {
        entries.sort_by(|a, b| {
            b.impact
                .total_cmp(&a.impact)
                .then(a.superfile.cmp(&b.superfile))
                .then(a.doc_id.cmp(&b.doc_id))
        });
    }

    /// A deterministic sample: `n` entries over `superfiles` superfiles of
    /// `docs` documents each, impacts from a smooth distribution.
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
                impact: 3.0 + (next() % 10_000) as f32 / 1_000.0,
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

    #[test]
    fn round_trips_every_entry_exactly() {
        let entries = sample(10_000, 37, 1_000_000, 7);
        let chunks = encode_tier(&entries);
        let decoded = decode_all(&chunks);
        assert_eq!(decoded.len(), entries.len());
        let mut want: Vec<(u32, u32, f32)> = entries
            .iter()
            .map(|e| (e.superfile, e.doc_id, e.impact))
            .collect();
        let mut got: Vec<(u32, u32, f32)> = decoded
            .iter()
            .map(|e| (e.superfile, e.doc_id, e.impact))
            .collect();
        want.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));
        got.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.total_cmp(&b.2)));
        assert_eq!(want, got, "entries and impacts are exact");
    }

    #[test]
    fn directory_bounds_are_exact_and_monotone() {
        let entries = sample(20_000, 100, 500_000, 11);
        let chunks = encode_tier(&entries);
        let mut prev_min = f32::INFINITY;
        let mut at = 0usize;
        for c in &chunks {
            let chunk = TierChunk::open(&c.bytes).expect("open");
            for i in 0..chunk.n_blocks() {
                let n = chunk.block_entries(i);
                let block = &entries[at..at + n];
                assert_eq!(chunk.block_max(i), block[0].impact);
                assert!(
                    chunk.block_max(i) <= prev_min,
                    "blocks are in descending impact"
                );
                prev_min = block[n - 1].impact;
                at += n;
            }
        }
        assert_eq!(at, entries.len());
    }

    #[test]
    fn a_constant_impact_block_round_trips_exactly() {
        let mut entries: Vec<TierEntry> = (0..300)
            .map(|i| TierEntry {
                superfile: i % 3,
                doc_id: i * 7,
                impact: 2.5,
            })
            .collect();
        descending(&mut entries);
        let chunks = encode_tier(&entries);
        assert_eq!(chunks.len(), 1);
        let decoded = decode_all(&chunks);
        assert!(decoded.iter().all(|e| e.impact == 2.5));
        assert_eq!(decoded.len(), 300);
    }

    #[test]
    fn chunks_cut_at_the_target_and_count_their_entries() {
        let entries = sample(1_500_000, 1_080, 1_000_000, 3);
        let chunks = encode_tier(&entries);
        assert!(chunks.len() > 1, "1.5M entries exceed one 4 MiB chunk");
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
        let chunks = encode_tier(&sample(5_000, 10, 1_000, 5));
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
    }

    /// Most terms' tiers are a handful of entries; they must cost a handful
    /// of bytes, not a padded lane of each stream, and round-trip exactly.
    #[test]
    fn small_tiers_are_small_and_exact() {
        for n in [1usize, 2, 3, 7, 127, 129, 4095, 4097] {
            let entries = sample(n, 1_080, 1_000_000, n as u64);
            let chunks = encode_tier(&entries);
            let bytes: usize = chunks.iter().map(|c| c.bytes.len()).sum();
            let fixed = CHUNK_HEADER_LEN + DIRECTORY_ENTRY_LEN * chunks.len() + BLOCK_HEADER_LEN;
            assert!(
                bytes <= fixed * entries.len().div_ceil(BLOCK_ENTRIES) + 8 * n,
                "{n} entries took {bytes} bytes"
            );
            let decoded = decode_all(&chunks);
            let mut want: Vec<(u32, u32, u32)> = entries
                .iter()
                .map(|e| (e.superfile, e.doc_id, e.impact.to_bits()))
                .collect();
            let mut got: Vec<(u32, u32, u32)> = decoded
                .iter()
                .map(|e| (e.superfile, e.doc_id, e.impact.to_bits()))
                .collect();
            want.sort_unstable();
            got.sort_unstable();
            assert_eq!(want, got, "{n} entries round-trip");
        }
    }
}
