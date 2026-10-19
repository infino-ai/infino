// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term's *tier*: its best documents table-wide, by impact, so a ranked
//! query can find the documents that hold several of its terms — and
//! bound every superfile — before opening any superfile.
//!
//! A tier is a sequence of entries `(superfile ordinal, local doc id,
//! impact)` in descending impact, cut into blocks of [`BLOCK_ENTRIES`]
//! and the blocks into content-addressed *chunks* of at most
//! [`CHUNK_TARGET_BYTES`], read in order and usually only in part: a
//! query stops once the blocks it has read bound everything it has not.
//!
//! ```text
//! chunk: magic "INFTTIR1" | version u32 | n_blocks u32
//!        | per block: max_impact f32 | min_impact f32 | n_entries u32 | byte_len u32
//!        | block bodies back to back
//! block: ordinal_bits u8 | lid_bits u8 | impact_bits u8 | max_impact_bits u32
//!        | ordinals, bit-packed sorted deltas | local ids, bit-packed
//!        | impacts as `max_impact_bits - impact_bits`, bit-packed
//! ```
//!
//! Inside a block the entries are stored sorted by `(ordinal, local id)`
//! — the block is always consumed whole, so its internal order carries
//! no information and the sorted ordinals pack to a few bits. Impacts are
//! exact: a block spans a narrow range of positive floats, whose bit
//! patterns order like the values, so each is stored as its distance
//! below the block's maximum pattern in the bits that distance needs.
//! Exactness is what lets a merge re-cut tiers without loosening a bound
//! or changing a byte, and what lets the directory's `max_impact` /
//! `min_impact` — the minimum of the last block read bounds every entry
//! not yet read — stay exact across any number of merges.

use bitpacking::{BitPacker, BitPacker4x};

use super::TermIndexError;
use crate::{
    supertable::manifest::part::ContentHash,
    utils::bytes::{u32_le_at, u64_le_at},
};

/// Identifies a tier chunk and its layout family.
pub(crate) const TIER_MAGIC: &[u8; 8] = b"INFTTIR1";
/// Layout version of a tier chunk.
pub(crate) const TIER_FORMAT_VERSION: u32 = 1;
/// Entries per block, the unit a reader consumes at a time. A multiple of
/// the bit-packer's lane width.
pub(crate) const BLOCK_ENTRIES: usize = 4096;
/// Bytes a chunk grows to before the next block starts a new one. A query
/// at small `k` reads one or two chunks per term.
pub(crate) const CHUNK_TARGET_BYTES: usize = 4 * 1024 * 1024;
const LANE: usize = BitPacker4x::BLOCK_LEN;
const MAGIC_LEN: usize = 8;
const CHUNK_HEADER_LEN: usize = MAGIC_LEN + 4 + 4;
const DIRECTORY_ENTRY_LEN: usize = 4 + 4 + 4 + 4;
const BLOCK_HEADER_LEN: usize = 3 + 4;

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

/// Where one chunk of a term's tier lives and how much of the tier it
/// holds, so a prefix read fetches only the chunks it needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TierChunkRef {
    pub(crate) content_hash: ContentHash,
    pub(crate) len: u64,
    pub(crate) n_entries: u64,
}

/// A chunk's bytes with its block directory parsed.
pub(crate) struct TierChunk<'a> {
    directory: Vec<BlockDir>,
    bodies: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
struct BlockDir {
    max_impact: f32,
    min_impact: f32,
    n_entries: u32,
    offset: usize,
    len: usize,
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(format!("tier chunk: {what}"))
}

/// Cut `entries` — in descending impact — into blocks and chunks. Returns
/// each chunk's bytes with its hash and entry count, in tier order.
pub(crate) fn encode_tier(entries: &[TierEntry]) -> Vec<(TierChunkRef, Vec<u8>)> {
    debug_assert!(
        entries.windows(2).all(|w| w[0].impact >= w[1].impact),
        "tier entries must be in descending impact"
    );
    let mut chunks = Vec::new();
    let mut blocks: Vec<(BlockDir, Vec<u8>)> = Vec::new();
    let mut chunk_bytes = 0usize;
    let mut scratch = vec![0u8; LANE * 4];
    let flush = |blocks: &mut Vec<(BlockDir, Vec<u8>)>,
                 chunks: &mut Vec<(TierChunkRef, Vec<u8>)>| {
        if blocks.is_empty() {
            return;
        }
        let n_entries: u64 = blocks.iter().map(|(d, _)| d.n_entries as u64).sum();
        let mut out = Vec::with_capacity(
            CHUNK_HEADER_LEN
                + blocks.len() * DIRECTORY_ENTRY_LEN
                + blocks.iter().map(|(_, b)| b.len()).sum::<usize>(),
        );
        out.extend_from_slice(TIER_MAGIC);
        out.extend_from_slice(&TIER_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
        for (d, body) in blocks.iter() {
            out.extend_from_slice(&d.max_impact.to_le_bytes());
            out.extend_from_slice(&d.min_impact.to_le_bytes());
            out.extend_from_slice(&d.n_entries.to_le_bytes());
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        }
        for (_, body) in blocks.drain(..) {
            out.extend_from_slice(&body);
        }
        let content_hash = ContentHash::of(&out);
        chunks.push((
            TierChunkRef {
                content_hash,
                len: out.len() as u64,
                n_entries,
            },
            out,
        ));
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
    let min_impact = block.iter().map(|e| e.impact).fold(f32::INFINITY, f32::min);
    let mut sorted: Vec<TierEntry> = block.to_vec();
    sorted.sort_unstable_by_key(|e| (e.superfile, e.doc_id));
    let n = sorted.len();
    let padded = n.div_ceil(LANE) * LANE;
    let mut ordinals = vec![0u32; padded];
    let mut doc_ids = vec![0u32; padded];
    let mut codes = vec![0u32; padded];
    debug_assert!(
        min_impact >= 0.0,
        "impacts are non-negative, so their bit patterns order like their values"
    );
    let max_bits = max_impact.to_bits();
    for (i, e) in sorted.iter().enumerate() {
        ordinals[i] = e.superfile;
        doc_ids[i] = e.doc_id;
        codes[i] = max_bits - e.impact.to_bits();
    }
    // Pad the sorted lane with the last ordinal so the sorted packer's
    // non-decreasing contract holds.
    let last = ordinals[n - 1];
    for slot in &mut ordinals[n..] {
        *slot = last;
    }
    let bp = BitPacker4x::new();
    // Each lane's deltas start from the previous lane's last ordinal, so a
    // lane costs only the spread of its own ordinals, not their magnitude.
    let lane_initial = |lanes: &[u32], i: usize| if i == 0 { 0 } else { lanes[i * LANE - 1] };
    let ordinal_bits = ordinals
        .chunks(LANE)
        .enumerate()
        .map(|(i, lane)| bp.num_bits_sorted(lane_initial(&ordinals, i), lane))
        .max()
        .unwrap_or(0);
    let lid_bits = doc_ids
        .chunks(LANE)
        .map(|lane| bp.num_bits(lane))
        .max()
        .unwrap_or(0);
    let impact_bits = codes
        .chunks(LANE)
        .map(|lane| bp.num_bits(lane))
        .max()
        .unwrap_or(0);
    let mut body = Vec::with_capacity(
        BLOCK_HEADER_LEN
            + padded * (ordinal_bits as usize + lid_bits as usize + impact_bits as usize) / 8,
    );
    body.push(ordinal_bits);
    body.push(lid_bits);
    body.push(impact_bits);
    body.extend_from_slice(&max_bits.to_le_bytes());
    for (i, lane) in ordinals.chunks(LANE).enumerate() {
        let written = bp.compress_sorted(lane_initial(&ordinals, i), lane, scratch, ordinal_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    for lane in doc_ids.chunks(LANE) {
        let written = bp.compress(lane, scratch, lid_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    for lane in codes.chunks(LANE) {
        let written = bp.compress(lane, scratch, impact_bits);
        body.extend_from_slice(&scratch[..written]);
    }
    (
        BlockDir {
            max_impact,
            min_impact,
            n_entries: n as u32,
            offset: 0,
            len: body.len(),
        },
        body,
    )
}

impl<'a> TierChunk<'a> {
    /// Parse a chunk's directory, refusing an unknown magic or version.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
        if bytes.len() < CHUNK_HEADER_LEN {
            return Err(malformed("short header"));
        }
        if &bytes[..MAGIC_LEN] != TIER_MAGIC {
            return Err(malformed("bad magic"));
        }
        let version = u32_le_at(bytes, MAGIC_LEN).ok_or_else(|| malformed("version"))?;
        if version != TIER_FORMAT_VERSION {
            return Err(TermIndexError::UnsupportedVersion {
                what: "tier chunk",
                found: version,
                expected: TIER_FORMAT_VERSION,
            });
        }
        let n_blocks =
            u32_le_at(bytes, MAGIC_LEN + 4).ok_or_else(|| malformed("block count"))? as usize;
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
            let min_impact = f32::from_bits(f(4)?);
            let n_entries = f(8)?;
            let len = f(12)? as usize;
            if n_entries == 0 || n_entries as usize > BLOCK_ENTRIES || len < BLOCK_HEADER_LEN {
                return Err(malformed("directory entry out of range"));
            }
            directory.push(BlockDir {
                max_impact,
                min_impact,
                n_entries,
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

    /// Exact smallest impact in block `i`; bounds every entry after it.
    pub(crate) fn block_min(&self, i: usize) -> f32 {
        self.directory[i].min_impact
    }

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
        let max_bits = u32::from_le_bytes(body[3..7].try_into().expect("4 bytes"));
        let n = d.n_entries as usize;
        let lanes = n.div_ceil(LANE);
        let lane_bytes = |bits: u8| LANE * bits as usize / 8;
        let need = BLOCK_HEADER_LEN
            + lanes * (lane_bytes(ordinal_bits) + lane_bytes(lid_bits) + lane_bytes(impact_bits));
        if body.len() != need || ordinal_bits > 32 || lid_bits > 32 || impact_bits > 32 {
            return Err(malformed("block body length"));
        }
        let bp = BitPacker4x::new();
        let mut ordinals = vec![0u32; lanes * LANE];
        let mut doc_ids = vec![0u32; lanes * LANE];
        let mut codes = vec![0u32; lanes * LANE];
        let mut at = BLOCK_HEADER_LEN;
        let mut initial = 0u32;
        for lane in ordinals.chunks_mut(LANE) {
            at += bp.decompress_sorted(initial, &body[at..], lane, ordinal_bits);
            initial = lane[LANE - 1];
        }
        for lane in doc_ids.chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, lid_bits);
        }
        for lane in codes.chunks_mut(LANE) {
            at += bp.decompress(&body[at..], lane, impact_bits);
        }
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
    crate::utils::varint::push_varint(out, chunks.len() as u32);
    for c in chunks {
        out.extend_from_slice(&c.content_hash.0);
        out.extend_from_slice(&c.len.to_le_bytes());
        out.extend_from_slice(&c.n_entries.to_le_bytes());
    }
}

/// Parse a chunk list written by [`push_chunk_refs`].
pub(crate) fn read_chunk_refs(
    bytes: &[u8],
    at: &mut usize,
) -> Result<Vec<TierChunkRef>, TermIndexError> {
    let n = crate::utils::varint::read_varint(bytes, at).ok_or_else(|| malformed("chunk count"))?
        as usize;
    let mut out = Vec::with_capacity(n.min(1024));
    for _ in 0..n {
        let hash = bytes
            .get(*at..*at + 32)
            .ok_or_else(|| malformed("chunk hash"))?;
        *at += 32;
        let len = u64_le_at(bytes, *at).ok_or_else(|| malformed("chunk len"))?;
        *at += 8;
        let n_entries = u64_le_at(bytes, *at).ok_or_else(|| malformed("chunk entries"))?;
        *at += 8;
        out.push(TierChunkRef {
            content_hash: ContentHash(hash.try_into().expect("32 bytes")),
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

    fn decode_all(chunks: &[(TierChunkRef, Vec<u8>)]) -> Vec<TierEntry> {
        let mut all = Vec::new();
        let mut buf = Vec::new();
        for (_, bytes) in chunks {
            let chunk = TierChunk::open(bytes).expect("open");
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
        for (_, bytes) in &chunks {
            let chunk = TierChunk::open(bytes).expect("open");
            for i in 0..chunk.n_blocks() {
                let n = chunk.block_entries(i);
                let block = &entries[at..at + n];
                assert_eq!(chunk.block_max(i), block[0].impact);
                assert_eq!(chunk.block_min(i), block[n - 1].impact);
                assert!(
                    chunk.block_max(i) <= prev_min,
                    "blocks are in descending impact"
                );
                prev_min = chunk.block_min(i);
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
        let total: u64 = chunks.iter().map(|(r, _)| r.n_entries).sum();
        assert_eq!(total, entries.len() as u64);
        for (r, bytes) in &chunks {
            assert_eq!(r.len as usize, bytes.len());
            assert_eq!(r.content_hash, ContentHash::of(bytes));
            assert!(
                bytes.len() <= CHUNK_TARGET_BYTES + 64 * 1024,
                "chunk {} over target",
                bytes.len()
            );
        }
        let bytes: usize = chunks.iter().map(|(_, b)| b.len()).sum();
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
                content_hash: ContentHash::of(&[i as u8; 4]),
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
    fn rejects_bad_magic_version_and_truncation() {
        let chunks = encode_tier(&sample(5_000, 10, 1_000, 5));
        let bytes = &chunks[0].1;
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(
            TierChunk::open(&bad),
            Err(TermIndexError::Malformed(_))
        ));
        let mut bad = bytes.clone();
        bad[MAGIC_LEN] = 9;
        assert!(matches!(
            TierChunk::open(&bad),
            Err(TermIndexError::UnsupportedVersion {
                what: "tier chunk",
                found: 9,
                ..
            })
        ));
        assert!(TierChunk::open(&bytes[..bytes.len() - 1]).is_err());
    }
}
