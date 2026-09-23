// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Term dictionary of the FTS blob: `(column, term)` keys to the term's
//! posting metadata, in two layouts chosen by the blob version.
//!
//! Every key is `<column_name>\x1F<term>` (ASCII Unit Separator, below
//! every printable byte), so the terms of one column are exactly the keys
//! with the `<column>\x1F` prefix, in sorted order.
//!
//! # Term blocks (`V7`+)
//!
//! Keys are sorted and cut into blocks of [`TERM_BLOCK_SIZE`]. Inside a
//! block every entry is **front-coded** against the entry before it: a
//! varint shared-prefix length, a varint suffix length, the suffix bytes,
//! then a form byte and the form's fields — an inline df=1 term carries
//! its doc id and tf as varints; a short- or long-form term carries its
//! metadata offset as a varint delta from the previous such offset and
//! its postings length as a varint. After the blocks comes an **index**:
//! every block's first full key, concatenated, then a fixed-width table
//! with each block's byte offset and where its first key begins. A
//! trailing footer holds the term and block counts, the block size and
//! the offset of the key area. Fixed-width entries mean the index is
//! binary-searched in place — opening a dictionary reads only the
//! footer, however many terms it holds — for the last first-key `<=`
//! the probe, and that one block is decoded sequentially; a prefix scan
//! starts the same way and walks blocks until the prefix stops matching.
//! This
//! is a third smaller than the FST for the same keys: an FST spends
//! bytes on transitions and on the value at every leaf, front-coding
//! spends them only on each key's unshared tail.
//!
//! # FST (`V1`–`V6`)
//!
//! One `fst::Map` over the same keys, values packed as `fst_value`
//! describes. Still written for the term-stats sidecar ([`DictBuilder`])
//! and read for legacy blobs; [`TermDict`] hides which layout a blob uses.

use std::{cmp::Ordering, collections::BTreeMap, io::Write, mem::take, ops::Range};

use fst::{IntoStreamer, Map, MapBuilder, Streamer};

use crate::utils::{
    bytes::{u32_le_at, u64_le_at},
    varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
};

pub(crate) mod value;
pub(crate) use value::{FstValue, INLINE_TF_MAX, PFOR_LENGTH_UNKNOWN};

/// Reserved separator byte inside dictionary keys (`<column>\x1F<term>`).
/// User column names must not contain this byte. ASCII Unit Separator
/// (U+001F) is below every printable ASCII char, so prefix iteration over a
/// column's terms works via a plain range scan.
pub const FST_SEPARATOR: u8 = 0x1F;

/// How a term dictionary lays its terms out — by blob version for a
/// superfile, by choice for any other caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DictLayout {
    /// One FST keyed `column <SEP> term`, values packed as [`value`]
    /// describes. Superfile blobs `V1`–`V6`.
    Fst,
    /// Front-coded term blocks behind a fixed-width first-key table (the
    /// block layout in this module). Superfile blobs `V7`+.
    Blocks,
}

/// Build a canonical FST key from `(column_name, term)`.
///
/// Encoding: `<column_name_utf8> | 0x1F | <term_utf8>`. The separator
/// byte ([`FST_SEPARATOR`], ASCII Unit Separator) is below every printable
/// ASCII byte, so prefix iteration `column_name\x1F` cleanly captures
/// every term in that column.
///
/// Callers must ensure `column_name` does not itself contain the
/// separator byte — see [`validate_column_name`]. `term` may be any
/// bytes a tokenizer produced; the separator `0x1F` is a C0 control
/// byte, which no shipped tokenizer emits inside a token
/// (`AsciiLowerTokenizer` keeps only `[a-z0-9]+`; `StandardTokenizer`
/// emits UAX #29 word segments, which never contain a control byte),
/// so the separator can never appear in a term.
pub fn make_key(column_name: &str, term: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(column_name.len() + 1 + term.len());
    k.extend_from_slice(column_name.as_bytes());
    k.push(FST_SEPARATOR);
    k.extend_from_slice(term.as_bytes());
    k
}

/// Returns `true` if `column_name` is safe to use as the column part of
/// an FST key: it must not contain the FST separator byte (otherwise
/// prefix iteration could return cross-column matches).
///
/// All other bytes are allowed; format-level naming rules (no `inf.`
/// prefix, etc.) are enforced elsewhere.
#[inline]
pub fn validate_column_name(column_name: &str) -> bool {
    !column_name.as_bytes().contains(&FST_SEPARATOR)
}

/// Stages keys for FST construction.
///
/// `fst::MapBuilder` requires keys to be inserted in sorted order;
/// `BTreeMap` absorbs that constraint while also deduplicating on
/// repeated key inserts (last write wins, matching the FST invariant
/// of one value per key).
#[derive(Debug, Default)]
pub struct DictBuilder {
    sorted_buffer: BTreeMap<Vec<u8>, u64>,
}

impl DictBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage a `(key, value)` pair. Inserts can arrive in any order;
    /// repeated inserts of the same key keep the most-recent value.
    pub fn insert(&mut self, key: &[u8], value: u64) {
        self.sorted_buffer.insert(key.to_vec(), value);
    }

    /// Number of distinct keys staged so far.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.sorted_buffer.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.sorted_buffer.is_empty()
    }

    /// Finalize the FST and return its serialized bytes.
    ///
    /// Panics on internal FST errors — these can only occur if the
    /// `BTreeMap` invariant (sorted, unique keys) is broken, which is
    /// impossible without `unsafe`.
    pub fn finish(self) -> Vec<u8> {
        let mut builder = MapBuilder::memory();
        for (k, v) in self.sorted_buffer {
            builder
                .insert(&k, v)
                .expect("BTreeMap guarantees sorted, unique keys");
        }
        builder
            .into_inner()
            .expect("in-memory FST writer cannot fail at finalize")
    }
}

/// Reads a serialized `u64`-valued FST — the shape [`DictBuilder`]
/// writes (the term-stats sidecar); the FTS blob's own dictionary is
/// read through [`TermDict`]. Test-only: the sidecar has its own reader.
#[cfg(test)]
pub struct DictReader<'a> {
    fst: Map<&'a [u8]>,
}

#[cfg(test)]
impl<'a> DictReader<'a> {
    /// Open from already-serialized FST bytes (the output of
    /// `DictBuilder::finish`).
    pub fn open(bytes: &'a [u8]) -> Result<Self, fst::Error> {
        Ok(Self {
            fst: Map::new(bytes)?,
        })
    }

    /// Exact lookup. Returns `None` if `key` is not in the FST.
    #[inline]
    pub fn lookup(&self, key: &[u8]) -> Option<u64> {
        self.fst.get(key)
    }

    /// Number of `(key, value)` pairs in the FST.
    #[inline]
    pub fn len(&self) -> usize {
        self.fst.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fst.is_empty()
    }

    /// Collect every `(key, value)` pair whose key starts with `prefix`,
    /// in lexicographic order. Returns owned data so the caller doesn't
    /// have to manage stream lifetimes.
    ///
    /// Uses an FST range scan that starts at `prefix` and stops as soon
    /// as a key without `prefix` appears — O(matching keys), not O(N).
    /// Suitable for "list every term in column X" diagnostics; for hot
    /// query paths use [`Self::lookup`].
    pub fn iter_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, u64)> {
        let mut out = Vec::new();
        self.for_each_prefix(prefix, |key, value| {
            out.push((key.to_vec(), value));
            true
        });
        out
    }

    /// Visit every `(key, value)` pair whose key starts with `prefix`, in
    /// lexicographic order, until `visit` returns `false`. The same range
    /// scan as [`Self::iter_prefix`] without materializing the keys, for a
    /// caller that keeps only a filtered subset or stops early — a `LIKE`
    /// expansion walks a column's whole vocabulary this way and gives up
    /// once too many terms qualify.
    pub fn for_each_prefix(&self, prefix: &[u8], mut visit: impl FnMut(&[u8], u64) -> bool) {
        let mut stream = self.fst.range().ge(prefix).into_stream();
        while let Some((key, value)) = stream.next() {
            if !key.starts_with(prefix) || !visit(key, value) {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Term-block dictionary (blob V7): front-coded blocks instead of an FST.
// ---------------------------------------------------------------------------

/// Terms per block. Small enough that a lookup's scan after the index
/// probe touches a few hundred bytes; large enough that the index (one
/// full key per block) is a rounding error.
pub(crate) const TERM_BLOCK_SIZE: usize = 32;
/// Trailing footer of a term-block region: `n_terms`, `n_blocks`,
/// `block_size` (`u32` each) and the offset of the key area (`u64`),
/// which is also where the blocks end.
const TERM_BLOCKS_FOOTER_BYTES: usize = 3 * 4 + 8;
/// One index-table entry: the block's byte offset (`u64`) and the start
/// of its first key within the key area (`u32`).
const INDEX_ENTRY_BYTES: usize = 8 + 4;
/// Term forms in a block entry.
const FORM_INLINE: u8 = 0;
const FORM_SHORT: u8 = 1;
const FORM_LONG: u8 = 2;

/// Streams sorted `(key, entry)` pairs into the term-block layout:
/// blocks as they fill, then the index, then the footer — so the
/// spilled build writes it to scratch without holding the dictionary,
/// and the in-RAM build writes it to a `Vec`.
pub(crate) struct TermBlockWriter<W: Write> {
    out: W,
    written: u64,
    /// The current block's encoded terms.
    block: Vec<u8>,
    /// First key of the current block (also the index entry).
    block_first_key: Vec<u8>,
    /// Previous key within the block, for front-coding.
    prev_key: Vec<u8>,
    /// Offset of the previous postings-form term in the block.
    prev_offset: u64,
    in_block: usize,
    /// `(first key, byte offset)` of every finished block.
    index: Vec<(Vec<u8>, u64)>,
    n_terms: u64,
}

impl<W: Write> TermBlockWriter<W> {
    pub(crate) fn new(out: W) -> Self {
        Self {
            out,
            written: 0,
            block: Vec::new(),
            block_first_key: Vec::new(),
            prev_key: Vec::new(),
            prev_offset: 0,
            in_block: 0,
            index: Vec::new(),
            n_terms: 0,
        }
    }

    /// Append one term. Keys must arrive strictly ascending.
    pub(crate) fn insert_sorted(&mut self, key: &[u8], entry: FstValue) -> std::io::Result<()> {
        debug_assert!(
            self.in_block == 0 || self.prev_key.as_slice() < key,
            "term-block dictionary keys must be strictly ascending"
        );
        if self.in_block == 0 {
            self.block_first_key.clear();
            self.block_first_key.extend_from_slice(key);
            self.prev_offset = 0;
        }
        let lcp = match self.in_block {
            0 => 0,
            _ => self
                .prev_key
                .iter()
                .zip(key)
                .take_while(|(a, b)| a == b)
                .count(),
        };
        push_varint(&mut self.block, lcp as u32);
        push_varint(&mut self.block, (key.len() - lcp) as u32);
        self.block.extend_from_slice(&key[lcp..]);
        match entry {
            FstValue::Inline { doc_id, tf } => {
                self.block.push(FORM_INLINE);
                push_varint(&mut self.block, doc_id);
                push_varint(&mut self.block, tf);
            }
            FstValue::Pfor {
                metadata_offset,
                postings_length_hint,
                short,
            } => {
                self.block.push(if short { FORM_SHORT } else { FORM_LONG });
                push_u64_varint(
                    &mut self.block,
                    metadata_offset.wrapping_sub(self.prev_offset),
                );
                push_varint(
                    &mut self.block,
                    postings_length_hint.expect("a term-block entry always carries its length"),
                );
                self.prev_offset = metadata_offset;
            }
        }
        self.prev_key.clear();
        self.prev_key.extend_from_slice(key);
        self.in_block += 1;
        self.n_terms += 1;
        if self.in_block == TERM_BLOCK_SIZE {
            self.flush_block()?;
        }
        Ok(())
    }

    fn flush_block(&mut self) -> std::io::Result<()> {
        if self.in_block == 0 {
            return Ok(());
        }
        self.index
            .push((take(&mut self.block_first_key), self.written));
        self.out.write_all(&self.block)?;
        self.written += self.block.len() as u64;
        self.block.clear();
        self.in_block = 0;
        Ok(())
    }

    /// Write the last block, the index and the footer; return the sink.
    pub(crate) fn finish(mut self) -> std::io::Result<W> {
        self.flush_block()?;
        let keys_offset = self.written;
        let mut tail = Vec::new();
        for (key, _) in &self.index {
            tail.extend_from_slice(key);
        }
        let mut key_start = 0u32;
        for (key, offset) in &self.index {
            tail.extend_from_slice(&offset.to_le_bytes());
            tail.extend_from_slice(&key_start.to_le_bytes());
            key_start += key.len() as u32;
        }
        tail.extend_from_slice(&(self.n_terms as u32).to_le_bytes());
        tail.extend_from_slice(&(self.index.len() as u32).to_le_bytes());
        tail.extend_from_slice(&(TERM_BLOCK_SIZE as u32).to_le_bytes());
        tail.extend_from_slice(&keys_offset.to_le_bytes());
        self.out.write_all(&tail)?;
        Ok(self.out)
    }
}

/// A term-block dictionary over borrowed bytes. Opening reads the
/// footer only; the index table is consulted in place per lookup, and a
/// malformed entry answers as an absent key rather than a panic (the
/// region is CRC-checked at open, so this is defensive).
pub(crate) struct TermBlocks<'a> {
    bytes: &'a [u8],
    n_blocks: usize,
    /// End of the block area (start of the key area).
    blocks_end: usize,
    /// End of the key area (start of the index table).
    keys_end: usize,
}

impl<'a> TermBlocks<'a> {
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() < TERM_BLOCKS_FOOTER_BYTES {
            return Err("term-block dictionary shorter than its footer".into());
        }
        let f = bytes.len() - TERM_BLOCKS_FOOTER_BYTES;
        let footer = "term-block dictionary footer is malformed";
        let n_terms = u32_le_at(bytes, f).ok_or(footer)? as usize;
        let n_blocks = u32_le_at(bytes, f + 4).ok_or(footer)? as usize;
        let block_size = u32_le_at(bytes, f + 8).ok_or(footer)? as usize;
        let keys_offset = u64_le_at(bytes, f + 12).ok_or(footer)? as usize;
        let table_bytes = n_blocks
            .checked_mul(INDEX_ENTRY_BYTES)
            .ok_or("term-block dictionary footer is malformed")?;
        let keys_end = f
            .checked_sub(table_bytes)
            .ok_or("term-block index does not fit before the footer")?;
        if block_size != TERM_BLOCK_SIZE || keys_offset > keys_end || n_terms < n_blocks {
            return Err("term-block dictionary footer is malformed".into());
        }
        let dict = Self {
            bytes,
            n_blocks,
            blocks_end: keys_offset,
            keys_end,
        };
        // The first block starts the region and its first entry is the
        // first indexed key; the table and footer sit at the end, so a
        // region whose bytes shifted still parses them yet misreads the
        // blocks — this one decode catches that.
        let anchored = match n_blocks {
            0 => keys_offset == keys_end,
            _ => {
                dict.entry(0) == Some((0, 0))
                    && dict.block_range(0).is_some_and(|range| {
                        let mut cur = BlockCursor::new(&bytes[range]);
                        cur.next().is_some() && Some(cur.key.as_slice()) == dict.first_key(0)
                    })
            }
        };
        if !anchored {
            return Err("term-block index is misaligned".into());
        }
        Ok(dict)
    }

    /// Block `b`'s byte offset and the start of its first key within the
    /// key area, straight from the table; `None` past the last block.
    fn entry(&self, b: usize) -> Option<(usize, usize)> {
        if b >= self.n_blocks {
            return None;
        }
        let at = self.keys_end + b * INDEX_ENTRY_BYTES;
        let block_start = u64_le_at(self.bytes, at)? as usize;
        let key_start = u32_le_at(self.bytes, at + 8)? as usize;
        Some((block_start, key_start))
    }

    /// Block `b`'s first key.
    fn first_key(&self, b: usize) -> Option<&'a [u8]> {
        let (_, start) = self.entry(b)?;
        let end = match self.entry(b + 1) {
            Some((_, next)) => next,
            None => self.keys_end - self.blocks_end,
        };
        self.bytes
            .get(self.blocks_end + start..self.blocks_end + end)
    }

    /// Index of the last block whose first key is `<= key`, if any. A
    /// malformed entry sorts as greater, so it is never chosen.
    fn block_for(&self, key: &[u8]) -> Option<usize> {
        let (mut lo, mut hi) = (0usize, self.n_blocks);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.first_key(mid) {
                Some(first) if first <= key => lo = mid + 1,
                _ => hi = mid,
            }
        }
        lo.checked_sub(1)
    }

    fn block_range(&self, b: usize) -> Option<Range<usize>> {
        let (start, _) = self.entry(b)?;
        let end = match self.entry(b + 1) {
            Some((next, _)) => next,
            None => self.blocks_end,
        };
        (start <= end && end <= self.blocks_end).then_some(start..end)
    }

    /// Exact lookup.
    pub(crate) fn lookup(&self, key: &[u8]) -> Option<FstValue> {
        let b = self.block_for(key)?;
        let mut cur = BlockCursor::new(&self.bytes[self.block_range(b)?]);
        while let Some(entry) = cur.next() {
            match cur.key.as_slice().cmp(key) {
                Ordering::Less => continue,
                Ordering::Equal => return Some(entry),
                Ordering::Greater => return None,
            }
        }
        None
    }

    /// Visit every `(key, entry)` whose key starts with `prefix`, in
    /// order, until `visit` returns `false`.
    pub(crate) fn for_each_prefix(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], FstValue) -> bool,
    ) {
        let mut b = self.block_for(prefix).unwrap_or(0);
        while b < self.n_blocks {
            let Some(range) = self.block_range(b) else {
                return;
            };
            let mut cur = BlockCursor::new(&self.bytes[range]);
            while let Some(entry) = cur.next() {
                let key = cur.key.as_slice();
                if key < prefix {
                    continue;
                }
                if !key.starts_with(prefix) || !visit(key, entry) {
                    return;
                }
            }
            b += 1;
        }
    }
}

/// Sequential decoder over one block's front-coded entries.
struct BlockCursor<'a> {
    bytes: &'a [u8],
    at: usize,
    key: Vec<u8>,
    prev_offset: u64,
}

impl<'a> BlockCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            at: 0,
            key: Vec::new(),
            prev_offset: 0,
        }
    }

    /// Decode the next entry into `self.key` and return its value;
    /// `None` at the block's end. A malformed block ends the walk early
    /// (the region is CRC-checked at open, so this is defensive).
    fn next(&mut self) -> Option<FstValue> {
        if self.at >= self.bytes.len() {
            return None;
        }
        let lcp = read_varint(self.bytes, &mut self.at)? as usize;
        let suffix_len = read_varint(self.bytes, &mut self.at)? as usize;
        let suffix = self.bytes.get(self.at..self.at + suffix_len)?;
        self.at += suffix_len;
        if lcp > self.key.len() {
            return None;
        }
        self.key.truncate(lcp);
        self.key.extend_from_slice(suffix);
        let form = *self.bytes.get(self.at)?;
        self.at += 1;
        match form {
            FORM_INLINE => {
                let doc_id = read_varint(self.bytes, &mut self.at)?;
                let tf = read_varint(self.bytes, &mut self.at)?;
                Some(FstValue::Inline { doc_id, tf })
            }
            FORM_SHORT | FORM_LONG => {
                let delta = read_u64_varint(self.bytes, &mut self.at)?;
                let length = read_varint(self.bytes, &mut self.at)?;
                let offset = self.prev_offset.wrapping_add(delta);
                self.prev_offset = offset;
                Some(FstValue::Pfor {
                    metadata_offset: offset,
                    postings_length_hint: Some(length),
                    short: form == FORM_SHORT,
                })
            }
            _ => None,
        }
    }
}

/// Builds the term dictionary of an FTS blob in either layout: staged
/// in a `BTreeMap` (any insertion order) and finished to bytes.
pub(crate) struct TermDictBuilder {
    layout: DictLayout,
    sorted: BTreeMap<Vec<u8>, FstValue>,
}

impl TermDictBuilder {
    pub(crate) fn new(layout: DictLayout) -> Self {
        Self {
            layout,
            sorted: BTreeMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, key: &[u8], entry: FstValue) {
        self.sorted.insert(key.to_vec(), entry);
    }

    pub(crate) fn finish(self) -> Vec<u8> {
        match self.layout {
            DictLayout::Fst => {
                let mut builder = DictBuilder::new();
                for (k, v) in self.sorted {
                    builder.insert(&k, pack_value(v));
                }
                builder.finish()
            }
            DictLayout::Blocks => {
                let mut w = TermBlockWriter::new(Vec::new());
                for (k, v) in self.sorted {
                    w.insert_sorted(&k, v).expect("Vec sink cannot fail");
                }
                w.finish().expect("Vec sink cannot fail")
            }
        }
    }
}

/// Streaming counterpart of [`TermDictBuilder`] for a producer that
/// already emits keys in order (the spilled build's k-way merge).
pub(crate) enum StreamingTermDictBuilder<W: Write> {
    Fst(MapBuilder<W>),
    Blocks(TermBlockWriter<W>),
}

impl<W: Write> StreamingTermDictBuilder<W> {
    pub(crate) fn new(layout: DictLayout, w: W) -> Result<Self, fst::Error> {
        Ok(match layout {
            DictLayout::Fst => Self::Fst(MapBuilder::new(w)?),
            DictLayout::Blocks => Self::Blocks(TermBlockWriter::new(w)),
        })
    }

    pub(crate) fn insert_sorted(&mut self, key: &[u8], entry: FstValue) -> Result<(), fst::Error> {
        match self {
            Self::Fst(inner) => inner.insert(key, pack_value(entry)),
            Self::Blocks(w) => w.insert_sorted(key, entry).map_err(fst::Error::Io),
        }
    }

    pub(crate) fn finish(self) -> Result<W, fst::Error> {
        match self {
            Self::Fst(inner) => inner.into_inner(),
            Self::Blocks(w) => w.finish().map_err(fst::Error::Io),
        }
    }
}

/// An FST dictionary holds long-form terms only; the short form exists
/// from `V7`, whose dictionary is term blocks.
fn pack_value(entry: FstValue) -> u64 {
    match entry {
        FstValue::Inline { doc_id, tf } => FstValue::pack_inline(doc_id, tf),
        FstValue::Pfor {
            metadata_offset,
            postings_length_hint,
            short,
        } => {
            assert!(!short, "an FST dictionary has no short-form terms");
            FstValue::pack_pfor(
                metadata_offset,
                postings_length_hint.unwrap_or(PFOR_LENGTH_UNKNOWN),
            )
        }
    }
}

/// The term dictionary of an FTS blob, whichever layout the blob's
/// version uses: decoded entries out, so no caller unpacks a value.
pub(crate) enum TermDict<'a> {
    Fst(Map<&'a [u8]>),
    Blocks(TermBlocks<'a>),
}

impl<'a> TermDict<'a> {
    pub(crate) fn open(bytes: &'a [u8], layout: DictLayout) -> Result<Self, String> {
        Ok(match layout {
            DictLayout::Fst => Self::Fst(Map::new(bytes).map_err(|e| e.to_string())?),
            DictLayout::Blocks => Self::Blocks(TermBlocks::open(bytes)?),
        })
    }

    #[inline]
    pub(crate) fn lookup(&self, key: &[u8]) -> Option<FstValue> {
        match self {
            Self::Fst(map) => map.get(key).map(FstValue::unpack),
            Self::Blocks(b) => b.lookup(key),
        }
    }

    pub(crate) fn iter_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, FstValue)> {
        let mut out = Vec::new();
        self.for_each_prefix(prefix, |key, value| {
            out.push((key.to_vec(), value));
            true
        });
        out
    }

    pub(crate) fn for_each_prefix(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], FstValue) -> bool,
    ) {
        match self {
            Self::Fst(map) => {
                let mut stream = map.range().ge(prefix).into_stream();
                while let Some((key, packed)) = stream.next() {
                    if !key.starts_with(prefix) || !visit(key, FstValue::unpack(packed)) {
                        break;
                    }
                }
            }
            Self::Blocks(b) => b.for_each_prefix(prefix, visit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(n: u32) -> Vec<(Vec<u8>, FstValue)> {
        // Sorted keys with realistic shapes: shared prefixes, a block
        // boundary in the middle of a run, inline and both postings forms.
        let mut v: Vec<(Vec<u8>, FstValue)> = (0..n)
            .map(|i| {
                let key = make_key("body", &format!("term{i:05}x{}", i % 7));
                let value = match i % 3 {
                    0 => FstValue::Inline {
                        doc_id: i * 7,
                        tf: 1 + i % 4,
                    },
                    1 => FstValue::Pfor {
                        metadata_offset: u64::from(i) * 13,
                        postings_length_hint: Some(9 + i % 5),
                        short: true,
                    },
                    _ => FstValue::Pfor {
                        metadata_offset: u64::from(i) * 13 + 4,
                        postings_length_hint: Some(1000 + i),
                        short: false,
                    },
                };
                (key, value)
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }

    #[test]
    fn term_blocks_look_up_every_key_and_refuse_absent_ones() {
        for n in [1u32, 5, 31, 32, 33, 64, 65, 1000] {
            let items = entries(n);
            let mut w = TermBlockWriter::new(Vec::new());
            for (k, v) in &items {
                w.insert_sorted(k, *v).expect("vec sink");
            }
            let bytes = w.finish().expect("vec sink");
            let d = TermBlocks::open(&bytes).expect("opens");
            for (k, v) in &items {
                assert_eq!(
                    d.lookup(k),
                    Some(*v),
                    "n={n} key {:?}",
                    String::from_utf8_lossy(k)
                );
            }
            assert_eq!(d.lookup(b"body\x1Fa"), None, "before the first key");
            assert_eq!(d.lookup(b"body\x1Fzzz"), None, "after the last key");
            assert_eq!(
                d.lookup(&make_key("body", "term00000x0!")),
                None,
                "between keys"
            );
            assert_eq!(
                d.lookup(&make_key("other", "term00000x0")),
                None,
                "other column"
            );
            // Prefix walks: whole column, a prefix spanning blocks, none.
            let all: Vec<_> = TermDict::Blocks(TermBlocks::open(&bytes).expect("opens"))
                .iter_prefix(&make_key("body", ""));
            assert_eq!(all, items);
            let some: Vec<_> = TermDict::Blocks(TermBlocks::open(&bytes).expect("opens"))
                .iter_prefix(&make_key("body", "term0003"));
            let want: Vec<_> = items
                .iter()
                .filter(|(k, _)| k.starts_with(&make_key("body", "term0003")))
                .cloned()
                .collect();
            assert_eq!(some, want);
            assert!(
                TermDict::Blocks(TermBlocks::open(&bytes).expect("opens"))
                    .iter_prefix(&make_key("body", "zzz"))
                    .is_empty()
            );
        }
    }

    #[test]
    fn term_blocks_are_smaller_than_the_fst_for_the_same_terms() {
        let items = entries(20_000);
        // The FST arm has no short form; compare against long-form entries.
        let mut fst = TermDictBuilder::new(DictLayout::Fst);
        let items: Vec<(Vec<u8>, FstValue)> = items
            .into_iter()
            .map(|(k, v)| match v {
                FstValue::Pfor {
                    metadata_offset,
                    postings_length_hint,
                    ..
                } => (
                    k,
                    FstValue::Pfor {
                        metadata_offset,
                        postings_length_hint,
                        short: false,
                    },
                ),
                inline => (k, inline),
            })
            .collect();
        let mut w = TermBlockWriter::new(Vec::new());
        for (k, v) in &items {
            w.insert_sorted(k, *v).expect("vec sink");
        }
        let blocks = w.finish().expect("vec sink");
        for (k, v) in &items {
            fst.insert(k, *v);
        }
        let fst = fst.finish();
        assert!(
            blocks.len() < fst.len(),
            "blocks {} vs fst {}",
            blocks.len(),
            fst.len()
        );
        // Both layouts answer identically through the unified reader.
        let a = TermDict::open(&blocks, DictLayout::Blocks).expect("blocks");
        let b = TermDict::open(&fst, DictLayout::Fst).expect("fst");
        for (k, v) in items.iter().step_by(97) {
            assert_eq!(a.lookup(k), Some(*v));
            assert_eq!(b.lookup(k), Some(*v));
        }
    }

    #[test]
    fn term_blocks_refuse_malformed_regions() {
        let items = entries(100);
        let mut w = TermBlockWriter::new(Vec::new());
        for (k, v) in &items {
            w.insert_sorted(k, *v).expect("vec sink");
        }
        let bytes = w.finish().expect("vec sink");
        assert!(
            TermBlocks::open(&bytes[..bytes.len() - 1]).is_err(),
            "truncated footer"
        );
        assert!(TermBlocks::open(&bytes[1..]).is_err(), "shifted bytes");
        assert!(TermBlocks::open(&[]).is_err(), "empty");
        // A table entry pointing past the blocks answers as absent, not
        // a panic: corrupt the last entry's block offset.
        let mut bad = bytes.clone();
        let f = bad.len() - TERM_BLOCKS_FOOTER_BYTES;
        let last = f - INDEX_ENTRY_BYTES;
        bad[last..last + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        let d = TermBlocks::open(&bad).expect("footer intact");
        let (k, _) = items.last().expect("items");
        assert_eq!(d.lookup(k), None);
        assert!(
            TermDict::Blocks(TermBlocks::open(&bad).expect("footer intact"))
                .iter_prefix(k)
                .is_empty()
        );
    }

    // --- make_key -------------------------------------------------------

    #[test]
    fn make_key_encodes_with_unit_separator() {
        assert_eq!(make_key("title", "rust"), b"title\x1Frust");
    }

    #[test]
    fn make_key_handles_empty_term() {
        // The bare prefix used by iter_prefix to "list all terms in column".
        assert_eq!(make_key("title", ""), b"title\x1F");
    }

    #[test]
    fn make_key_handles_empty_column() {
        // Edge case; not produced in practice but must not panic.
        assert_eq!(make_key("", "rust"), b"\x1Frust");
    }

    #[test]
    fn make_key_handles_both_empty() {
        assert_eq!(make_key("", ""), b"\x1F");
    }

    #[test]
    fn make_key_preserves_term_bytes() {
        // AsciiLowerTokenizer drops non-ASCII tokens, but make_key itself
        // is byte-transparent: multi-byte UTF-8 in a term (e.g. from the
        // standard tokenizer) comes through exactly.
        let key = make_key("body", "café");
        assert_eq!(&key[0..4], b"body");
        assert_eq!(key[4], FST_SEPARATOR);
        assert_eq!(&key[5..], "café".as_bytes());
    }

    #[test]
    fn make_key_capacity_avoids_realloc() {
        // Sanity: the Vec is sized exactly. Not a behaviour requirement
        // but a perf one — protects the hot-path call against
        // accidental realloc.
        let key = make_key("title", "rust");
        assert_eq!(key.len(), key.capacity());
    }

    // --- validate_column_name -------------------------------------------

    #[test]
    fn validate_column_name_rejects_separator_byte() {
        // A column name containing 0x1F would let prefix iteration leak
        // across columns. Format-level validation rejects this at the
        // builder boundary.
        let bad = "ti\x1Ftle";
        assert!(!validate_column_name(bad));
    }

    #[test]
    fn validate_column_name_accepts_normal_names() {
        for name in [
            "title",
            "body",
            "headline",
            "field_1",
            "field-2",
            "MyField",
            "snake_case",
            "camelCase",
            "PascalCase",
            "with spaces",
            "with.dots",
        ] {
            assert!(validate_column_name(name), "expected {name:?} to be valid");
        }
    }

    #[test]
    fn validate_column_name_accepts_empty_string() {
        // Length restrictions are enforced at the builder level, not here.
        assert!(validate_column_name(""));
    }

    #[test]
    fn validate_column_name_accepts_high_unicode() {
        // No restriction on non-ASCII bytes. Format-level rules can add
        // restrictions if needed; the dict layer is byte-transparent.
        assert!(validate_column_name("café"));
        assert!(validate_column_name("日本語"));
    }

    // --- DictBuilder + DictReader roundtrip -----------------------------

    #[test]
    fn lookup_roundtrip_basic() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 100);
        b.insert(&make_key("title", "async"), 200);
        b.insert(&make_key("body", "tokio"), 300);
        let bytes = b.finish();

        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.lookup(&make_key("title", "rust")), Some(100));
        assert_eq!(r.lookup(&make_key("title", "async")), Some(200));
        assert_eq!(r.lookup(&make_key("body", "tokio")), Some(300));
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn lookup_missing_returns_none() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("title", "java")), None);
        assert_eq!(r.lookup(&make_key("body", "rust")), None);
        assert_eq!(r.lookup(b""), None);
    }

    #[test]
    fn out_of_order_inserts_work() {
        // Direct fst::MapBuilder rejects out-of-order keys; our
        // DictBuilder absorbs the sort via BTreeMap.
        let mut b = DictBuilder::new();
        let keys = ["z", "a", "m", "b", "y", "c"];
        for (i, k) in keys.iter().enumerate() {
            b.insert(&make_key("col", k), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        for (i, k) in keys.iter().enumerate() {
            assert_eq!(r.lookup(&make_key("col", k)), Some(i as u64));
        }
    }

    #[test]
    fn duplicate_inserts_keep_last_value() {
        // BTreeMap dedups; the FST gets the latest value. Documents
        // the "last write wins" semantic for DictBuilder::insert.
        let mut b = DictBuilder::new();
        b.insert(&make_key("col", "key"), 1);
        b.insert(&make_key("col", "key"), 2);
        b.insert(&make_key("col", "key"), 999);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("col", "key")), Some(999));
        assert_eq!(r.len(), 1);
    }

    // --- Prefix iteration ----------------------------------------------

    #[test]
    fn iter_prefix_returns_matching_keys_only() {
        let mut b = DictBuilder::new();
        // Three columns, distinct vocabularies.
        b.insert(&make_key("title", "alpha"), 1);
        b.insert(&make_key("title", "beta"), 2);
        b.insert(&make_key("title", "gamma"), 3);
        b.insert(&make_key("body", "alpha"), 10);
        b.insert(&make_key("body", "delta"), 20);
        b.insert(&make_key("tag", "epsilon"), 100);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Listing all terms in `title`.
        let title_terms = r.iter_prefix(b"title\x1F");
        let title_only: Vec<&[u8]> = title_terms.iter().map(|(k, _)| k.as_slice()).collect();
        assert_eq!(
            title_only,
            vec![
                b"title\x1Falpha".as_slice(),
                b"title\x1Fbeta".as_slice(),
                b"title\x1Fgamma".as_slice(),
            ],
            "iter_prefix on title\\x1F returns only title-prefixed keys"
        );
        assert_eq!(
            title_terms.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );

        // body has different vocabulary.
        let body_terms = r.iter_prefix(b"body\x1F");
        assert_eq!(body_terms.len(), 2);

        // tag has just one.
        let tag_terms = r.iter_prefix(b"tag\x1F");
        assert_eq!(tag_terms.len(), 1);
    }

    #[test]
    fn iter_prefix_lexicographic_order() {
        // Insert in arbitrary order, expect lex order out.
        let mut b = DictBuilder::new();
        let terms = ["zebra", "ant", "monkey", "apple", "banana", "aardvark"];
        for (i, t) in terms.iter().enumerate() {
            b.insert(&make_key("col", t), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        let listed: Vec<Vec<u8>> = r
            .iter_prefix(b"col\x1F")
            .into_iter()
            .map(|(k, _)| k[4..].to_vec()) // strip "col\x1F"
            .collect();
        let mut expected: Vec<&str> = terms.to_vec();
        expected.sort();
        let expected: Vec<Vec<u8>> = expected.iter().map(|s| s.as_bytes().to_vec()).collect();
        assert_eq!(listed, expected);
    }

    #[test]
    fn iter_prefix_no_match_returns_empty() {
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Column that doesn't exist in the FST.
        assert_eq!(r.iter_prefix(b"missing\x1F"), Vec::new());
        // Empty prefix on non-empty FST returns everything (defining
        // behavior — useful for a "dump dict" diagnostic).
        assert_eq!(r.iter_prefix(b"").len(), 1);
    }

    #[test]
    fn iter_prefix_stops_at_first_non_match() {
        // Implementation correctness: iter_prefix must stop scanning
        // as soon as a non-matching key appears. We can't observe the
        // stopping directly, but we can confirm the result is correct
        // when many post-prefix keys exist.
        let mut b = DictBuilder::new();
        // Lots of `body\x1F*` keys after the (alphabetically earlier)
        // `title\x1F*` block.
        for i in 0..1000 {
            b.insert(&make_key("body", &format!("term{i:04}")), i as u64);
        }
        b.insert(&make_key("title", "rust"), 9999);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        // Title scan must yield exactly one entry, regardless of how
        // many body entries follow (`body` < `title` alphabetically,
        // so by the time the scan reaches `title\x1F` it has already
        // skipped past every `body\x1F*` key).
        let title = r.iter_prefix(b"title\x1F");
        assert_eq!(title.len(), 1);
        assert_eq!(title[0].1, 9999);
    }

    // --- Empty-FST handling --------------------------------------------

    #[test]
    fn empty_builder_produces_valid_empty_fst() {
        let bytes = DictBuilder::new().finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.lookup(b"anything"), None);
        assert_eq!(r.iter_prefix(b""), Vec::new());
        assert_eq!(r.iter_prefix(b"prefix"), Vec::new());
    }

    // --- Stress / scale -------------------------------------------------

    #[test]
    fn handles_thousand_keys_roundtrip() {
        let mut b = DictBuilder::new();
        for i in 0..1000 {
            b.insert(&make_key("body", &format!("term{i:04}")), i as u64);
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.len(), 1000);

        for i in 0..1000 {
            let k = make_key("body", &format!("term{i:04}"));
            assert_eq!(r.lookup(&k), Some(i as u64));
        }

        // Prefix iter returns all 1000 in lex order.
        let listed = r.iter_prefix(b"body\x1F");
        assert_eq!(listed.len(), 1000);
        for w in listed.windows(2) {
            assert!(w[0].0 < w[1].0, "prefix iter not in lex order");
        }
    }

    #[test]
    fn serialization_is_deterministic() {
        // Same inputs (regardless of insertion order) produce
        // byte-identical FST output. Important for reproducible builds
        // and for content-addressed superfile hashing.
        let mut b1 = DictBuilder::new();
        b1.insert(&make_key("a", "x"), 1);
        b1.insert(&make_key("b", "y"), 2);
        b1.insert(&make_key("c", "z"), 3);
        let bytes1 = b1.finish();

        let mut b2 = DictBuilder::new();
        b2.insert(&make_key("c", "z"), 3);
        b2.insert(&make_key("a", "x"), 1);
        b2.insert(&make_key("b", "y"), 2);
        let bytes2 = b2.finish();

        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn keys_with_same_prefix_share_state() {
        // FST tail-merging makes shared prefixes effectively free.
        // We don't probe FST internals; just verify lookups are still
        // correct when many keys share a long common prefix.
        let mut b = DictBuilder::new();
        for i in 0..100 {
            b.insert(
                &make_key("very_long_column_name", &format!("term{i}")),
                i as u64,
            );
        }
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");
        assert_eq!(r.len(), 100);
        for i in 0..100 {
            let k = make_key("very_long_column_name", &format!("term{i}"));
            assert_eq!(r.lookup(&k), Some(i as u64));
        }
    }

    #[test]
    fn lookup_distinguishes_columns_with_same_term() {
        // The whole point of `<col>\x1F<term>` keying: same term in
        // different columns must look up to different values.
        let mut b = DictBuilder::new();
        b.insert(&make_key("title", "rust"), 1);
        b.insert(&make_key("body", "rust"), 2);
        b.insert(&make_key("tag", "rust"), 3);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open DictReader");

        assert_eq!(r.lookup(&make_key("title", "rust")), Some(1));
        assert_eq!(r.lookup(&make_key("body", "rust")), Some(2));
        assert_eq!(r.lookup(&make_key("tag", "rust")), Some(3));
    }

    /// `DictBuilder` tracks key counts + emptiness, which round-trip
    /// through `DictReader::{len, is_empty}`; the streaming builder
    /// tracks `n_keys` as sorted keys arrive.
    #[test]
    fn builder_and_reader_track_key_counts() {
        let mut b = DictBuilder::new();
        assert!(b.is_empty(), "fresh builder is empty");
        assert_eq!(b.len(), 0);
        b.insert(b"alpha", 1);
        b.insert(b"beta", 2);
        assert!(!b.is_empty());
        assert_eq!(b.len(), 2);
        let bytes = b.finish();
        let r = DictReader::open(&bytes).expect("open dict");
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());

        // The streaming term dictionary accepts keys fed in strictly
        // sorted order and produces bytes the unified reader opens, in
        // both layouts.
        for layout in [DictLayout::Fst, DictLayout::Blocks] {
            let mut sb = StreamingTermDictBuilder::new(layout, Vec::new()).expect("streaming");
            sb.insert_sorted(b"a", FstValue::Inline { doc_id: 1, tf: 1 })
                .expect("sorted insert");
            sb.insert_sorted(
                b"b",
                FstValue::Pfor {
                    metadata_offset: 40,
                    postings_length_hint: Some(9),
                    short: false,
                },
            )
            .expect("sorted insert");
            let bytes = sb.finish().expect("finish streaming builder");
            let d = TermDict::open(&bytes, layout).expect("opens");
            assert_eq!(d.lookup(b"a"), Some(FstValue::Inline { doc_id: 1, tf: 1 }));
            assert_eq!(
                d.lookup(b"b"),
                Some(FstValue::Pfor {
                    metadata_offset: 40,
                    postings_length_hint: Some(9),
                    short: false,
                })
            );
        }
    }
}
