// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A term's *level head*: one head per `(term, layer)`, ranking the term's
//! documents across every superfile of the layer, so a query can read a
//! term's best documents for the whole layer from the top and stop once
//! the rest cannot reach its k-th score — whatever the number of
//! superfiles.
//!
//! **Order.** Entries are in descending *key* — the length-normalized term
//! frequency `tf / (tf + norm)`, the factor a BM25 score multiplies the
//! term's idf by, at the entry's superfile's own average length and the
//! column's parameters — then ascending superfile ordinal, then ascending
//! row. Ties are ordered physically, not by `_id`: a query whose k-th
//! score ties the frontier reads through the tie band and resolves `_id`
//! for that band alone.
//!
//! **Depth.** [`head_depth`]: a term with at most 2048 documents in the
//! layer keeps them all; a larger one keeps the top eighth of them, never
//! fewer than 2048.
//!
//! **Floor.** The largest key of a document of the term in the layer that
//! the head leaves out, `0.0` when it holds them all: what an unread
//! document can still contribute, once the head is read through.
//!
//! **Levels.** Level 0 is the first [`LEVEL0_ENTRIES`] entries; each level
//! after is [`LEVEL_GROWTH`] times the one before; the last ends at the
//! depth. A query that needs `d` entries reads about `log₈(d / 64)` levels
//! and fetches fewer than `8·d` entries. Each level is stored as one
//! *segment*, read and verified whole:
//!
//! ```text
//! segment: crc32c u32 (over what follows) | n_blocks varint | blocks
//! block:   first key f32 | last key f32 | n_entries u8 | base ordinal varint
//!          | ordinal bits u8 | row bits u8 | tf bits u8 | length bits u8
//!          | ordinal deltas, bit-packed | rows, bit-packed
//!          | term frequencies, bit-packed | length buckets, bit-packed
//! ```
//!
//! A block holds [`BLOCK_ENTRIES`] consecutive entries of the head (fewer
//! in the last), stored sorted by `(ordinal, row)` so both streams code as
//! small deltas: an ordinal as its distance from the block's smallest, a
//! row as its distance from the previous row of the same superfile (the
//! row itself for a superfile's first). The header's first and last keys
//! are the block's largest and smallest, so a reader bounds what lies past
//! a block without decoding it; it recomputes each entry's key from its
//! term frequency and length bucket, as the kernel scores it.
//!
//! **Per-bucket max tf.** For each length bucket the term occurs in, the
//! largest term frequency over the term's whole posting list in the layer:
//! what the length-coherent bound on an unread document is built from.

use std::{cmp::Ordering, ops::Range};

use super::{
    TermIndexError,
    tier::{BLOCK_ENTRIES, TierEntry, pack_stream},
};
use crate::{
    superfile::{
        bits::{for_each_lane, payload_bytes, width_of},
        format::checksum::crc32c,
        fts::builder::proportional_head_depth,
    },
    utils::{
        bytes::u32_le_at,
        varint::{
            push_doc_facts, push_u64_varint, push_varint, read_doc_facts, read_u64_varint,
            read_varint,
        },
    },
};

/// Entries in level 0: one block, enough for every single-term top-10.
pub(crate) const LEVEL0_ENTRIES: usize = BLOCK_ENTRIES;
/// Each level holds this many times the entries of the one before.
pub(crate) const LEVEL_GROWTH: usize = 8;

const CRC_LEN: usize = 4;
const KEY_LEN: usize = 4;
/// A block header after its keys and entry count: the four stream widths.
const N_WIDTHS: usize = 4;
/// Widest an ordinal delta, row or term-frequency stream may be.
const MAX_U32_BITS: u8 = u32::BITS as u8;
/// Widest a length-bucket stream may be.
const MAX_LEN_BITS: u8 = u8::BITS as u8;

/// How deep a term's head is in a layer where it has `df` documents: the
/// rule a superfile's own head follows, so a layer's cut finds the
/// documents it keeps in its superfiles' heads.
pub(crate) fn head_depth(df: u64) -> u64 {
    proportional_head_depth(df)
}

/// The entry ranges of a head of `depth` entries' levels, in order.
pub(crate) fn level_ranges(depth: usize) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut size) = (0usize, LEVEL0_ENTRIES);
    while start < depth {
        let end = start.saturating_add(size).min(depth);
        out.push(start..end);
        start = end;
        size = size.saturating_mul(LEVEL_GROWTH);
    }
    out
}

/// One head entry with its key.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct KeyedEntry {
    pub(crate) entry: TierEntry,
    pub(crate) key: f32,
}

/// The head's order: key descending, then ordinal, then row ascending.
pub(crate) fn head_order(a: &KeyedEntry, b: &KeyedEntry) -> Ordering {
    b.key
        .total_cmp(&a.key)
        .then_with(|| a.entry.superfile.cmp(&b.entry.superfile))
        .then_with(|| a.entry.doc_id.cmp(&b.entry.doc_id))
}

/// A head as cut: its entries in head order and its floor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CutHead {
    pub(crate) entries: Vec<KeyedEntry>,
    pub(crate) floor: f32,
}

/// Cut a head of at most `depth` entries from `candidates`, any order.
/// `outside` bounds the key of every document of the term not among the
/// candidates (the largest of the sources' tails or floors; `0.0` when the
/// candidates are the whole list). The floor is the larger of `outside`
/// and the best key the cut leaves out.
pub(crate) fn cut_head(mut candidates: Vec<KeyedEntry>, outside: f32, depth: usize) -> CutHead {
    let mut floor = outside;
    if candidates.len() > depth {
        if depth > 0 {
            candidates.select_nth_unstable_by(depth, head_order);
        }
        let dropped_best = candidates[depth..]
            .iter()
            .map(|c| c.key)
            .fold(0.0f32, f32::max);
        floor = floor.max(dropped_best);
        candidates.truncate(depth);
    }
    candidates.sort_unstable_by(head_order);
    CutHead {
        entries: candidates,
        floor,
    }
}

/// The largest term frequency per length bucket, ascending by bucket, one
/// pair per bucket the term occurs in.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct BucketMaxTf(Vec<(u8, u32)>);

impl BucketMaxTf {
    /// Record a document of `tf` in bucket `len`.
    pub(crate) fn observe(&mut self, len: u8, tf: u32) {
        match self.0.binary_search_by_key(&len, |p| p.0) {
            Ok(i) => self.0[i].1 = self.0[i].1.max(tf),
            Err(i) => self.0.insert(i, (len, tf)),
        }
    }

    /// Fold another source's maxima in: the per-bucket maximum of both.
    pub(crate) fn merge(&mut self, other: &Self) {
        for &(len, tf) in &other.0 {
            self.observe(len, tf);
        }
    }

    /// The largest term frequency in bucket `len`; `None` when the term
    /// has no document of that length.
    pub(crate) fn get(&self, len: u8) -> Option<u32> {
        self.0
            .binary_search_by_key(&len, |p| p.0)
            .ok()
            .map(|i| self.0[i].1)
    }

    /// The `(bucket, max tf)` pairs, ascending by bucket.
    #[cfg(test)]
    pub(crate) fn pairs(&self) -> &[(u8, u32)] {
        &self.0
    }

    /// Append as `n varint | per pair: bucket delta varint | max tf varint`.
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        Self::encode_pairs(&self.0, out);
    }

    /// [`Self::encode`] of `pairs`, ascending by bucket, one per bucket.
    pub(crate) fn encode_pairs(pairs: &[(u8, u32)], out: &mut Vec<u8>) {
        debug_assert!(pairs.windows(2).all(|w| w[0].0 < w[1].0));
        push_u64_varint(out, pairs.len() as u64);
        let mut prev = 0u8;
        for &(len, tf) in pairs {
            push_varint(out, u32::from(len - prev));
            push_varint(out, tf);
            prev = len;
        }
    }

    /// Read what [`Self::encode`] wrote at `*at`, advancing it.
    pub(crate) fn decode(bytes: &[u8], at: &mut usize) -> Result<Self, TermIndexError> {
        let n = read_u64_varint(bytes, at).ok_or_else(|| malformed("bucket count"))?;
        let mut pairs = Vec::new();
        let mut prev = 0u32;
        for i in 0..n {
            let delta = read_varint(bytes, at).ok_or_else(|| malformed("bucket"))?;
            let tf = read_varint(bytes, at).ok_or_else(|| malformed("bucket max tf"))?;
            let len = prev
                .checked_add(delta)
                .filter(|&l| l <= u32::from(u8::MAX) && (i == 0 || delta > 0))
                .ok_or_else(|| malformed("bucket order"))?;
            pairs.push((len as u8, tf));
            prev = len;
        }
        Ok(Self(pairs))
    }
}

/// Where one level of a term's head sits in its slice's head object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LevelRef {
    /// Byte offset of the level's segment in the head object.
    pub(crate) offset: u64,
    /// The segment's byte length.
    pub(crate) len: u64,
    /// Entries the level holds.
    pub(crate) n_entries: u32,
}

/// A term's level head, as its run names it: the floor, the per-bucket
/// maxima, and either the levels' places in the slice's head object or,
/// for a head of a few entries, the entries themselves in head order.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct HeadRef {
    /// See the module docs; `f32::INFINITY` when what the head leaves out
    /// is not bounded (a source that kept no head).
    pub(crate) floor: f32,
    /// `None` when a source did not record them, which leaves the
    /// length-coherent bound to the frontier and the length budget alone.
    pub(crate) buckets: Option<BucketMaxTf>,
    pub(crate) levels: Vec<LevelRef>,
    pub(crate) inline: Vec<TierEntry>,
}

/// Bucket maxima flag in a head reference: absent or present.
const BUCKETS_ABSENT: u8 = 0;
const BUCKETS_PRESENT: u8 = 1;

impl HeadRef {
    /// Entries the head holds.
    #[cfg(test)]
    pub(crate) fn n_entries(&self) -> u64 {
        self.inline.len() as u64
            + self
                .levels
                .iter()
                .map(|l| u64::from(l.n_entries))
                .sum::<u64>()
    }

    /// Append as `floor f32 | buckets flag u8 [| buckets] | n_inline varint
    /// | per inline entry: ordinal varint, doc facts | n_levels varint | per
    /// level: offset varint, length varint, entries varint`.
    pub(crate) fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.floor.to_le_bytes());
        match &self.buckets {
            None => out.push(BUCKETS_ABSENT),
            Some(b) => {
                out.push(BUCKETS_PRESENT);
                b.encode(out);
            }
        }
        push_u64_varint(out, self.inline.len() as u64);
        for e in &self.inline {
            push_varint(out, e.superfile);
            push_doc_facts(out, e.doc_id, e.tf, e.len);
        }
        push_u64_varint(out, self.levels.len() as u64);
        for l in &self.levels {
            push_u64_varint(out, l.offset);
            push_u64_varint(out, l.len);
            push_varint(out, l.n_entries);
        }
    }

    /// Read what [`Self::encode`] wrote at `*at`, advancing it.
    pub(crate) fn decode(bytes: &[u8], at: &mut usize) -> Result<Self, TermIndexError> {
        let floor = bytes
            .get(*at..*at + KEY_LEN)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .ok_or_else(|| malformed("head floor"))?;
        *at += KEY_LEN;
        let flag = *bytes
            .get(*at)
            .ok_or_else(|| malformed("head buckets flag"))?;
        *at += 1;
        let buckets = match flag {
            BUCKETS_ABSENT => None,
            BUCKETS_PRESENT => Some(BucketMaxTf::decode(bytes, at)?),
            _ => return Err(malformed("head buckets flag")),
        };
        let n_inline = read_u64_varint(bytes, at).ok_or_else(|| malformed("head inline count"))?;
        let mut inline = Vec::new();
        for _ in 0..n_inline {
            let superfile =
                read_varint(bytes, at).ok_or_else(|| malformed("head inline ordinal"))?;
            let (doc_id, tf, len) =
                read_doc_facts(bytes, at).ok_or_else(|| malformed("head inline entry"))?;
            inline.push(TierEntry {
                superfile,
                doc_id,
                tf,
                len,
            });
        }
        let n_levels = read_u64_varint(bytes, at).ok_or_else(|| malformed("head level count"))?;
        let mut levels = Vec::new();
        for _ in 0..n_levels {
            let offset = read_u64_varint(bytes, at).ok_or_else(|| malformed("level offset"))?;
            let len = read_u64_varint(bytes, at).ok_or_else(|| malformed("level length"))?;
            let n_entries = read_varint(bytes, at).ok_or_else(|| malformed("level entries"))?;
            levels.push(LevelRef {
                offset,
                len,
                n_entries,
            });
        }
        Ok(Self {
            floor,
            buckets,
            levels,
            inline,
        })
    }
}

/// One decoded block of a segment: its entries in `(ordinal, row)` order
/// and the keys its header records.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LevelBlock {
    /// The block's largest key.
    pub(crate) first_key: f32,
    /// The block's smallest key: a bound on every entry after the block.
    pub(crate) last_key: f32,
    pub(crate) entries: Vec<TierEntry>,
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(format!("level head: {what}"))
}

/// Encode one level of a head — `entries`, consecutive in head order — as
/// a segment.
pub(crate) fn encode_segment(entries: &[KeyedEntry]) -> Vec<u8> {
    debug_assert!(
        entries
            .windows(2)
            .all(|w| head_order(&w[0], &w[1]) != Ordering::Greater),
        "a segment's entries are in head order"
    );
    let mut out = vec![0u8; CRC_LEN];
    push_u64_varint(&mut out, entries.len().div_ceil(BLOCK_ENTRIES) as u64);
    for block in entries.chunks(BLOCK_ENTRIES) {
        encode_block(block, &mut out);
    }
    let crc = crc32c(&out[CRC_LEN..]);
    out[..CRC_LEN].copy_from_slice(&crc.to_le_bytes());
    out
}

fn encode_block(block: &[KeyedEntry], out: &mut Vec<u8>) {
    let first_key = block[0].key;
    let last_key = block[block.len() - 1].key;
    let mut sorted: Vec<TierEntry> = block.iter().map(|k| k.entry).collect();
    sorted.sort_unstable_by_key(|e| (e.superfile, e.doc_id));
    let base = sorted[0].superfile;
    let ordinal_deltas: Vec<u64> = sorted
        .iter()
        .map(|e| u64::from(e.superfile - base))
        .collect();
    let rows: Vec<u64> = sorted
        .iter()
        .enumerate()
        .map(|(i, e)| match i {
            0 => u64::from(e.doc_id),
            _ if sorted[i - 1].superfile == e.superfile => {
                u64::from(e.doc_id - sorted[i - 1].doc_id)
            }
            _ => u64::from(e.doc_id),
        })
        .collect();
    let widest = |vs: &mut dyn Iterator<Item = u64>| width_of(vs.max().unwrap_or(0));
    let ordinal_bits = widest(&mut ordinal_deltas.iter().copied());
    let row_bits = widest(&mut rows.iter().copied());
    let tf_bits = widest(&mut sorted.iter().map(|e| u64::from(e.tf)));
    let len_bits = widest(&mut sorted.iter().map(|e| u64::from(e.len)));
    out.extend_from_slice(&first_key.to_le_bytes());
    out.extend_from_slice(&last_key.to_le_bytes());
    out.push(block.len() as u8);
    push_varint(out, base);
    out.extend_from_slice(&[ordinal_bits, row_bits, tf_bits, len_bits]);
    pack_stream(out, ordinal_bits, ordinal_deltas.into_iter());
    pack_stream(out, row_bits, rows.into_iter());
    pack_stream(out, tf_bits, sorted.iter().map(|e| u64::from(e.tf)));
    pack_stream(out, len_bits, sorted.iter().map(|e| u64::from(e.len)));
}

/// Decode a whole segment, verifying its CRC, into `out` (cleared first).
pub(crate) fn decode_segment(
    bytes: &[u8],
    out: &mut Vec<LevelBlock>,
) -> Result<(), TermIndexError> {
    out.clear();
    if bytes.len() < CRC_LEN {
        return Err(malformed("segment truncated"));
    }
    let stored = u32_le_at(bytes, 0).ok_or_else(|| malformed("segment truncated"))?;
    // Altered bytes are what a hash mismatch reports elsewhere in the
    // index: the call for a rebuild, not a parse failure.
    if crc32c(&bytes[CRC_LEN..]) != stored {
        return Err(TermIndexError::HashMismatch);
    }
    let mut at = CRC_LEN;
    let n_blocks = read_u64_varint(bytes, &mut at).ok_or_else(|| malformed("block count"))?;
    for _ in 0..n_blocks {
        out.push(decode_block(bytes, &mut at)?);
    }
    if at != bytes.len() {
        return Err(malformed("segment length"));
    }
    Ok(())
}

fn decode_block(bytes: &[u8], at: &mut usize) -> Result<LevelBlock, TermIndexError> {
    let header = bytes
        .get(*at..*at + 2 * KEY_LEN + 1)
        .ok_or_else(|| malformed("block header"))?;
    let first_key = f32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let last_key = f32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let n = header[2 * KEY_LEN] as usize;
    *at += 2 * KEY_LEN + 1;
    if n == 0 || n > BLOCK_ENTRIES || first_key < last_key {
        return Err(malformed("block header"));
    }
    let base = read_varint(bytes, at).ok_or_else(|| malformed("base ordinal"))?;
    let widths = bytes
        .get(*at..*at + N_WIDTHS)
        .ok_or_else(|| malformed("block widths"))?;
    let [ordinal_bits, row_bits, tf_bits, len_bits] = [widths[0], widths[1], widths[2], widths[3]];
    *at += N_WIDTHS;
    if ordinal_bits > MAX_U32_BITS
        || row_bits > MAX_U32_BITS
        || tf_bits > MAX_U32_BITS
        || len_bits > MAX_LEN_BITS
    {
        return Err(malformed("block widths"));
    }
    let mut streams = [
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    ];
    for (stream, bits) in streams
        .iter_mut()
        .zip([ordinal_bits, row_bits, tf_bits, len_bits])
    {
        let len = payload_bytes(n, bits);
        let payload = bytes
            .get(*at..*at + len)
            .ok_or_else(|| malformed("block stream"))?;
        for_each_lane(payload, 0, n, bits, |v| stream.push(v))
            .ok_or_else(|| malformed("block stream"))?;
        *at += len;
    }
    let [ordinal_deltas, rows, tfs, lens] = streams;
    let mut entries: Vec<TierEntry> = Vec::with_capacity(n);
    for i in 0..n {
        let superfile = u64::from(base)
            .checked_add(ordinal_deltas[i])
            .and_then(|o| u32::try_from(o).ok())
            .ok_or_else(|| malformed("ordinal"))?;
        let doc_id = match entries.last() {
            Some(prev) if prev.superfile == superfile => u64::from(prev.doc_id)
                .checked_add(rows[i])
                .and_then(|r| u32::try_from(r).ok())
                .ok_or_else(|| malformed("row"))?,
            Some(prev) if prev.superfile > superfile => return Err(malformed("ordinal order")),
            _ => rows[i] as u32,
        };
        entries.push(TierEntry {
            superfile,
            doc_id,
            tf: tfs[i] as u32,
            len: lens[i] as u8,
        });
    }
    Ok(LevelBlock {
        first_key,
        last_key,
        entries,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use proptest::prelude::*;

    use super::*;
    use crate::superfile::fts::builder::WHOLE_HEAD_MAX_DF;

    /// Proptest cases per property.
    const CASES: u32 = 128;
    /// Largest generated candidate set.
    const MAX_CANDIDATES: usize = 600;

    fn keyed(superfile: u32, doc_id: u32, tf: u32, len: u8) -> KeyedEntry {
        // Any key monotone in tf at fixed length orders the head; the codec
        // never looks at how a key was computed.
        let key = tf as f32 / (tf as f32 + 1.0 + f32::from(len) / 16.0);
        KeyedEntry {
            entry: TierEntry {
                superfile,
                doc_id,
                tf,
                len,
            },
            key,
        }
    }

    fn arb_candidates() -> impl Strategy<Value = Vec<KeyedEntry>> {
        // Few superfiles, rows, tfs and lengths: many ties, repeated
        // superfiles within a block, rows far apart and close together.
        prop::collection::vec((0u32..6, 0u32..5000, 1u32..4, 0u8..4), 0..MAX_CANDIDATES).prop_map(
            |v| {
                let mut seen = BTreeSet::new();
                v.into_iter()
                    .filter(|&(s, r, ..)| seen.insert((s, r)))
                    .map(|(s, r, tf, len)| keyed(s, r, tf, len))
                    .collect()
            },
        )
    }

    fn round_trip(entries: &[KeyedEntry]) -> Vec<LevelBlock> {
        let bytes = encode_segment(entries);
        let mut blocks = Vec::new();
        decode_segment(&bytes, &mut blocks).expect("decode");
        blocks
    }

    #[test]
    fn depth_keeps_rare_terms_whole_and_an_eighth_of_common_ones() {
        assert_eq!(head_depth(0), 0);
        assert_eq!(head_depth(1), 1);
        assert_eq!(head_depth(WHOLE_HEAD_MAX_DF), WHOLE_HEAD_MAX_DF);
        // Just past the whole-list size the floor of the cut is the same size.
        assert_eq!(head_depth(WHOLE_HEAD_MAX_DF + 1), WHOLE_HEAD_MAX_DF);
        assert_eq!(head_depth(8 * WHOLE_HEAD_MAX_DF), WHOLE_HEAD_MAX_DF);
        assert_eq!(head_depth(8 * WHOLE_HEAD_MAX_DF + 1), WHOLE_HEAD_MAX_DF + 1);
        assert_eq!(head_depth(4_600_000_000), 575_000_000);
    }

    #[test]
    fn levels_grow_eightfold_from_one_block() {
        assert!(level_ranges(0).is_empty());
        assert_eq!(level_ranges(1), vec![0..1]);
        assert_eq!(level_ranges(64), vec![0..64]);
        assert_eq!(level_ranges(65), vec![0..64, 64..65]);
        assert_eq!(
            level_ranges(64 + 512 + 4096 + 1),
            vec![0..64, 64..576, 576..4672, 4672..4673]
        );
    }

    #[test]
    fn an_empty_and_a_single_entry_segment_round_trip() {
        assert!(round_trip(&[]).is_empty());
        let one = [keyed(7, 123_456, 3, 9)];
        let blocks = round_trip(&one);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].entries, vec![one[0].entry]);
        assert_eq!(
            (blocks[0].first_key, blocks[0].last_key),
            (one[0].key, one[0].key)
        );
    }

    #[test]
    fn extreme_values_round_trip() {
        let entries = [
            keyed(u32::MAX, u32::MAX, u32::MAX, u8::MAX),
            keyed(0, 0, 1, 0),
        ];
        let mut entries = entries.to_vec();
        entries.sort_by(head_order);
        let blocks = round_trip(&entries);
        let mut got = blocks[0].entries.clone();
        let mut want: Vec<_> = entries.iter().map(|k| k.entry).collect();
        got.sort_by_key(|e| (e.superfile, e.doc_id));
        want.sort_by_key(|e| (e.superfile, e.doc_id));
        assert_eq!(got, want);
    }

    #[test]
    fn a_corrupt_or_truncated_segment_is_refused() {
        let mut entries: Vec<_> = (0..100)
            .map(|i| keyed(i % 3, i * 7, 1 + i % 5, 2))
            .collect();
        entries.sort_by(head_order);
        let bytes = encode_segment(&entries);
        let mut out = Vec::new();
        for cut in [0, 3, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                decode_segment(&bytes[..cut], &mut out).is_err(),
                "cut at {cut}"
            );
        }
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 1;
        assert!(decode_segment(&flipped, &mut out).is_err());
        // Bytes past the blocks are refused even behind a valid CRC.
        let mut long = bytes;
        long.push(0);
        let crc = crc32c(&long[CRC_LEN..]);
        long[..CRC_LEN].copy_from_slice(&crc.to_le_bytes());
        assert!(decode_segment(&long, &mut out).is_err());
    }

    #[test]
    fn bucket_maxima_merge_and_round_trip() {
        let mut a = BucketMaxTf::default();
        a.observe(5, 2);
        a.observe(0, 1);
        a.observe(5, 7);
        a.observe(255, 3);
        let mut b = BucketMaxTf::default();
        b.observe(5, 4);
        b.observe(9, 9);
        a.merge(&b);
        assert_eq!(a.pairs(), &[(0, 1), (5, 7), (9, 9), (255, 3)]);
        assert_eq!((a.get(5), a.get(6)), (Some(7), None));
        let mut bytes = Vec::new();
        a.encode(&mut bytes);
        let mut at = 0;
        assert_eq!(BucketMaxTf::decode(&bytes, &mut at).expect("decode"), a);
        assert_eq!(at, bytes.len());
        // A repeated bucket is refused.
        let bad = [2u8, 4, 1, 0, 1];
        assert!(BucketMaxTf::decode(&bad, &mut 0).is_err());
    }

    #[test]
    fn a_head_reference_round_trips() {
        let mut buckets = BucketMaxTf::default();
        buckets.observe(3, 9);
        let heads = [
            HeadRef::default(),
            HeadRef {
                floor: f32::INFINITY,
                buckets: Some(buckets),
                levels: vec![
                    LevelRef {
                        offset: 0,
                        len: 300,
                        n_entries: 64,
                    },
                    LevelRef {
                        offset: 300,
                        len: 2000,
                        n_entries: 512,
                    },
                ],
                inline: Vec::new(),
            },
            HeadRef {
                floor: 0.25,
                buckets: None,
                levels: Vec::new(),
                inline: vec![TierEntry {
                    superfile: 4,
                    doc_id: 77,
                    tf: 2,
                    len: 5,
                }],
            },
        ];
        for head in heads {
            let mut bytes = Vec::new();
            head.encode(&mut bytes);
            let mut at = 0;
            assert_eq!(HeadRef::decode(&bytes, &mut at).expect("decode"), head);
            assert_eq!(at, bytes.len());
            for cut in 0..bytes.len() {
                assert!(
                    HeadRef::decode(&bytes[..cut], &mut 0).is_err(),
                    "cut at {cut}"
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(CASES))]

        /// Every entry comes back, in its block, with the block's extreme
        /// keys; blocks hold consecutive head entries.
        #[test]
        fn any_head_round_trips(candidates in arb_candidates()) {
            let mut entries = candidates;
            entries.sort_by(head_order);
            let blocks = round_trip(&entries);
            prop_assert_eq!(blocks.len(), entries.len().div_ceil(BLOCK_ENTRIES));
            for (block, want) in blocks.iter().zip(entries.chunks(BLOCK_ENTRIES)) {
                let mut want_entries: Vec<_> = want.iter().map(|k| k.entry).collect();
                want_entries.sort_by_key(|e| (e.superfile, e.doc_id));
                prop_assert_eq!(&block.entries, &want_entries);
                prop_assert_eq!(block.first_key, want[0].key);
                prop_assert_eq!(block.last_key, want[want.len() - 1].key);
            }
        }

        /// The cut equals the brute-force top `depth` in head order, and
        /// its floor is the best key it left out, or `outside` if larger.
        #[test]
        fn the_cut_is_the_brute_force_top(
            candidates in arb_candidates(),
            depth in 0usize..700,
            outside in prop_oneof![Just(0.0f32), 0.0f32..1.0],
        ) {
            let mut all = candidates.clone();
            all.sort_by(head_order);
            let cut = cut_head(candidates, outside, depth);
            let kept = depth.min(all.len());
            prop_assert_eq!(&cut.entries[..], &all[..kept]);
            let dropped_best = all[kept..].iter().map(|c| c.key).fold(0.0f32, f32::max);
            prop_assert_eq!(cut.floor, dropped_best.max(outside));
            for omitted in &all[kept..] {
                prop_assert!(cut.floor >= omitted.key);
            }
        }

        /// Per-bucket maxima equal the brute force over any documents.
        #[test]
        fn bucket_maxima_equal_the_brute_force(docs in prop::collection::vec((0u8..=255, 1u32..1000), 0..300)) {
            let mut got = BucketMaxTf::default();
            for &(len, tf) in &docs {
                got.observe(len, tf);
            }
            for len in 0..=255u8 {
                let want = docs.iter().filter(|d| d.0 == len).map(|d| d.1).max();
                prop_assert_eq!(got.get(len), want);
            }
        }
    }
}
