// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! On-disk layout of the term index: a resident *root* over
//! content-addressed *slices*.
//!
//! **Root** — small enough to stay resident at any table size:
//!
//! ```text
//! magic "INFTIDX1" | version u32 | n_superfiles u32
//! | per superfile: uuid | smallest doc id i128 LE
//! | n_segments u32 | per segment: n_slices u32
//!     | per slice: first_key (u32 len + bytes) | last_key (u32 len + bytes)
//!                | content hash (32 B) | slice byte length u64
//! ```
//!
//! Superfiles are listed once; every posting names its superfile by
//! *ordinal* into that list, so a posting is a few bytes rather than a
//! uuid. Each carries its smallest doc id: a manifest part's list entry
//! records the id range its superfiles span, so a query that has routed
//! to a set of superfiles can pick the parts to load without loading any
//! — the id is the join key between the two. Segments exist so a later
//! commit can append a delta (its own slice list, appending to the
//! superfile list) without rewriting the base.
//!
//! **Slice** — one contiguous key range. Two layouts are read:
//!
//! ```text
//! version 1, read whole:
//! magic "INFTSLC1" | 1 u32 | dict_len u64 | postings_len u64
//! | dictionary (front-coded term blocks, `utils::terms`) | postings region
//!
//! version 2, read whole or by range:
//! magic "INFTSLC1" | 2 u32 | dict_len u64 | table_len u64 | postings_len u64
//! | postings region: per term, run | run CRC32C
//! | dictionary (front-coded term blocks, `utils::terms`)
//! | block table: CRC32C per dictionary block | head CRC32C
//! ```
//!
//! A whole slice is checked against its content hash. Version 2 adds CRCs so
//! a reader can fetch just the *head* (the header plus everything from the
//! dictionary's block index to the end), then one block and one run. The
//! head sits at the end so one read of the slice's tail usually gets it.
//!
//! The dictionary's value for a term is a byte range into the postings
//! region, which holds that term's *run*:
//!
//! ```text
//! n_postings varint | per posting: superfile ordinal varint | df varint
//!                   | bound f32 LE | location tag u8 [| location payload]
//! ```
//!
//! `location` mirrors the superfile dictionary's own value for the term —
//! a PFOR or short-form byte range relative to that superfile's postings
//! region, or the inline `(doc id, tf)` of a single-document term — so a
//! reader holding the artifact can go straight to the bytes without
//! opening the superfile's dictionary. `None` when the build's field
//! policy chose not to carry it (a term in so many superfiles that a query
//! opens most of the table regardless).
//!
//! Every decoder here refuses an unknown magic or version *loudly*: the
//! block dictionary returns "absent" for an entry it cannot decode, and an
//! artifact read as "terms absent" would silently route to nothing.

use std::ops::Range;

use uuid::Uuid;

use super::TermIndexError;
use crate::{
    superfile::format::{
        CRC_BYTES,
        checksum::{crc32c, crc32c_append},
    },
    supertable::manifest::{part::ContentHash, term_range::prefix_upper_bound},
    utils::{
        bytes::{u32_le_at, u64_le_at},
        terms::{DictEntry, TERM_BLOCK_SIZE, TERM_BLOCKS_FOOTER_BYTES, TermBlockIndex, TermBlocks},
        varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
    },
};

/// Identifies the root file and its major layout family.
pub(crate) const ROOT_MAGIC: &[u8; 8] = b"INFTIDX1";
/// Identifies a slice file and its major layout family.
pub(crate) const SLICE_MAGIC: &[u8; 8] = b"INFTSLC1";
/// Layout version of the root. `2` added each superfile's smallest doc id;
/// `3` keys terms by the column's field id instead of its name. A root at
/// an older version does not decode: the index is rebuilt, not translated.
pub(crate) const ROOT_FORMAT_VERSION: u32 = 3;
/// Slice layout read whole only.
const SLICE_VERSION_WHOLE: u32 = 1;
/// Slice layout that can also be read by range.
const SLICE_VERSION_RANGED: u32 = 2;

const MAGIC_LEN: usize = 8;
const U32_LEN: usize = 4;
const U64_LEN: usize = 8;
const UUID_LEN: usize = 16;
const I128_LEN: usize = 16;
/// blake3 digest width; asserted against `ContentHash` in tests.
const HASH_LEN: usize = 32;
const F32_LEN: usize = 4;
/// Root fixed header: magic, version, superfile count.
const ROOT_HEADER_LEN: usize = MAGIC_LEN + U32_LEN + U32_LEN;
/// Version 1 slice header: magic, version, dictionary and postings lengths.
const WHOLE_HEADER_LEN: usize = MAGIC_LEN + U32_LEN + U64_LEN + U64_LEN;
/// Version 2 slice header: magic, version, dictionary, block-table and
/// postings lengths.
pub(crate) const RANGED_HEADER_LEN: usize = MAGIC_LEN + U32_LEN + 3 * U64_LEN;

/// Location tags, one byte each in a posting.
const LOC_NONE: u8 = 0;
const LOC_PFOR: u8 = 1;
const LOC_SHORT: u8 = 2;
const LOC_INLINE: u8 = 3;

/// Where a term's postings sit inside one superfile — the superfile
/// dictionary's own value for the term, carried here so a routed query
/// need not read that dictionary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Location {
    /// Not carried (field policy), or not applicable.
    None,
    /// Long form: `len` bytes at `offset` within the superfile's postings
    /// region — header, skip table, PFOR blocks.
    Pfor {
        /// Offset within the superfile's FTS postings region.
        offset: u64,
        /// Byte length of the term's postings.
        len: u32,
    },
    /// Short form (`df ≤ 128`): `len` bytes at `offset`, decoded whole.
    Short {
        /// Offset within the superfile's FTS postings region.
        offset: u64,
        /// Byte length of the short body.
        len: u32,
    },
    /// Single-document term: the whole posting.
    Inline {
        /// Local doc id within the superfile.
        doc_id: u32,
        /// Term frequency in that document.
        tf: u32,
    },
}

impl Location {
    /// The location a superfile dictionary entry describes.
    pub(crate) fn from_dict_value(value: DictEntry) -> Self {
        match value {
            DictEntry::Inline { doc_id, tf } => Self::Inline { doc_id, tf },
            DictEntry::Pfor {
                metadata_offset,
                postings_length: len,
                short: true,
            } => Self::Short {
                offset: metadata_offset,
                len,
            },
            DictEntry::Pfor {
                metadata_offset,
                postings_length: len,
                short: false,
            } => Self::Pfor {
                offset: metadata_offset,
                len,
            },
        }
    }
}

impl Location {
    /// The superfile dictionary value this location stands for, or `None`
    /// when no location was carried.
    pub(crate) fn to_dict_value(self) -> Option<DictEntry> {
        match self {
            Self::None => None,
            Self::Pfor { offset, len } => Some(DictEntry::Pfor {
                metadata_offset: offset,
                postings_length: len,
                short: false,
            }),
            Self::Short { offset, len } => Some(DictEntry::Pfor {
                metadata_offset: offset,
                postings_length: len,
                short: true,
            }),
            Self::Inline { doc_id, tf } => Some(DictEntry::Inline { doc_id, tf }),
        }
    }
}

/// One `(term, superfile)` fact.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Posting {
    /// Ordinal into the root's superfile list.
    pub(crate) superfile: u32,
    /// Gross document frequency of the term in that superfile.
    pub(crate) df: u64,
    /// Upper bound on the BM25 score the term can reach in that
    /// superfile, at the superfile's declared scoring parameters.
    /// `f32::INFINITY` is a valid (useless) bound and is what a build
    /// that did not collect bounds writes.
    pub(crate) bound: f32,
    /// Where the postings sit in that superfile, if carried.
    pub(crate) location: Location,
}

/// A slice's place in the key space and in storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SliceRef {
    /// Smallest key the slice holds.
    pub(crate) first_key: Vec<u8>,
    /// Largest key the slice holds.
    pub(crate) last_key: Vec<u8>,
    /// Content hash of the slice bytes; also names the object.
    pub(crate) content_hash: ContentHash,
    /// Slice byte length, so a fetch can be sized before it is issued.
    pub(crate) len: u64,
}

/// One build's slice list. The base is segment 0; deltas append.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Segment {
    /// Slices in ascending key order, non-overlapping.
    pub(crate) slices: Vec<SliceRef>,
}

/// The resident root.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Root {
    /// Superfiles with postings in some segment; postings name them by
    /// ordinal. Never reordered — a delta appends. May hold nil padding,
    /// which no posting names (see `append_reserved`).
    pub(crate) superfiles: Vec<Uuid>,
    /// Each superfile's smallest doc id, parallel to `superfiles`.
    pub(crate) id_mins: Vec<i128>,
    /// Base first, then deltas in commit order.
    pub(crate) segments: Vec<Segment>,
}

fn push_bytes_u32_len(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn malformed(what: &str) -> TermIndexError {
    TermIndexError::Malformed(what.to_owned())
}

/// A bounds-checked cursor over an encoded root or slice.
struct Cursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn u32(&mut self, what: &str) -> Result<u32, TermIndexError> {
        let v = u32_le_at(self.bytes, self.at).ok_or_else(|| malformed(what))?;
        self.at += U32_LEN;
        Ok(v)
    }
    fn u64(&mut self, what: &str) -> Result<u64, TermIndexError> {
        let v = u64_le_at(self.bytes, self.at).ok_or_else(|| malformed(what))?;
        self.at += U64_LEN;
        Ok(v)
    }
    fn take(&mut self, n: usize, what: &str) -> Result<&'a [u8], TermIndexError> {
        let end = self.at.checked_add(n).ok_or_else(|| malformed(what))?;
        let s = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| malformed(what))?;
        self.at = end;
        Ok(s)
    }
    fn bytes_u32_len(&mut self, what: &str) -> Result<&'a [u8], TermIndexError> {
        let n = self.u32(what)? as usize;
        self.take(n, what)
    }
}

fn check_magic_version(
    c: &mut Cursor<'_>,
    magic: &[u8; 8],
    expected: u32,
    what: &'static str,
) -> Result<(), TermIndexError> {
    if c.take(MAGIC_LEN, what)? != magic {
        return Err(malformed(&format!("{what}: bad magic")));
    }
    let found = c.u32(what)?;
    if found != expected {
        return Err(TermIndexError::UnsupportedVersion {
            what,
            found,
            expected,
        });
    }
    Ok(())
}

impl Root {
    /// Serialize.
    pub(crate) fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            ROOT_HEADER_LEN
                + self.superfiles.len() * UUID_LEN
                + U32_LEN
                + self.segments.len() * U32_LEN,
        );
        debug_assert_eq!(self.superfiles.len(), self.id_mins.len());
        out.extend_from_slice(ROOT_MAGIC);
        out.extend_from_slice(&ROOT_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.superfiles.len() as u32).to_le_bytes());
        for (id, id_min) in self.superfiles.iter().zip(&self.id_mins) {
            out.extend_from_slice(id.as_bytes());
            out.extend_from_slice(&id_min.to_le_bytes());
        }
        out.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        for seg in &self.segments {
            out.extend_from_slice(&(seg.slices.len() as u32).to_le_bytes());
            for s in &seg.slices {
                push_bytes_u32_len(&mut out, &s.first_key);
                push_bytes_u32_len(&mut out, &s.last_key);
                out.extend_from_slice(&s.content_hash.0);
                out.extend_from_slice(&s.len.to_le_bytes());
            }
        }
        out
    }

    /// Parse, refusing an unknown magic or version.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, TermIndexError> {
        let mut c = Cursor { bytes, at: 0 };
        check_magic_version(&mut c, ROOT_MAGIC, ROOT_FORMAT_VERSION, "root")?;
        let n_superfiles = c.u32("root superfile count")? as usize;
        let mut superfiles = Vec::with_capacity(n_superfiles);
        let mut id_mins = Vec::with_capacity(n_superfiles);
        for _ in 0..n_superfiles {
            let raw = c.take(UUID_LEN, "root superfile id")?;
            superfiles.push(Uuid::from_bytes(raw.try_into().expect("16 bytes")));
            let raw = c.take(I128_LEN, "root superfile id_min")?;
            id_mins.push(i128::from_le_bytes(raw.try_into().expect("16 bytes")));
        }
        let n_segments = c.u32("root segment count")? as usize;
        let mut segments = Vec::with_capacity(n_segments);
        for _ in 0..n_segments {
            let n_slices = c.u32("segment slice count")? as usize;
            let mut slices = Vec::with_capacity(n_slices);
            for _ in 0..n_slices {
                let first_key = c.bytes_u32_len("slice first key")?.to_vec();
                let last_key = c.bytes_u32_len("slice last key")?.to_vec();
                let hash = c.take(HASH_LEN, "slice hash")?;
                let content_hash = ContentHash(hash.try_into().expect("32 bytes"));
                let len = c.u64("slice length")?;
                slices.push(SliceRef {
                    first_key,
                    last_key,
                    content_hash,
                    len,
                });
            }
            segments.push(Segment { slices });
        }
        if c.at != bytes.len() {
            return Err(malformed("root: trailing bytes"));
        }
        Ok(Self {
            superfiles,
            id_mins,
            segments,
        })
    }

    /// The slices that may hold `key`, one per segment at most — the
    /// last slice whose first key is `≤ key`, if `key ≤` its last key.
    pub(crate) fn slices_for_key<'a>(&'a self, key: &[u8]) -> impl Iterator<Item = &'a SliceRef> {
        self.segments.iter().filter_map(move |seg| {
            let i = seg
                .slices
                .partition_point(|s| s.first_key.as_slice() <= key);
            let s = seg.slices.get(i.checked_sub(1)?)?;
            (key <= s.last_key.as_slice()).then_some(s)
        })
    }

    /// The slices whose key range intersects `[prefix, prefix_upper)`,
    /// in key order per segment. A prefix with no upper bound (all
    /// `0xFF`) runs to the end.
    pub(crate) fn slices_for_prefix<'a>(
        &'a self,
        prefix: &[u8],
    ) -> impl Iterator<Item = &'a SliceRef> {
        let upper = prefix_upper_bound(prefix);
        self.segments.iter().flat_map(move |seg| {
            let start = seg
                .slices
                .partition_point(|s| s.last_key.as_slice() < prefix);
            let upper = upper.clone();
            seg.slices[start..]
                .iter()
                .take_while(move |s| match &upper {
                    Some(u) => s.first_key.as_slice() < u.as_slice(),
                    None => true,
                })
        })
    }
}

/// Append one posting to a run.
pub(crate) fn push_posting(out: &mut Vec<u8>, p: &Posting) {
    push_varint(out, p.superfile);
    push_u64_varint(out, p.df);
    out.extend_from_slice(&p.bound.to_le_bytes());
    match p.location {
        Location::None => out.push(LOC_NONE),
        Location::Pfor { offset, len } => {
            out.push(LOC_PFOR);
            push_u64_varint(out, offset);
            push_varint(out, len);
        }
        Location::Short { offset, len } => {
            out.push(LOC_SHORT);
            push_u64_varint(out, offset);
            push_varint(out, len);
        }
        Location::Inline { doc_id, tf } => {
            out.push(LOC_INLINE);
            push_varint(out, doc_id);
            push_varint(out, tf);
        }
    }
}

/// Encode a term's run: count, then its postings in ascending superfile
/// ordinal.
pub(crate) fn encode_run(postings: &[Posting]) -> Vec<u8> {
    let mut out = Vec::with_capacity(postings.len() * (1 + 2 + F32_LEN + 1 + 8));
    push_varint(&mut out, postings.len() as u32);
    for p in postings {
        push_posting(&mut out, p);
    }
    out
}

/// Decode one posting at `at`, advancing it.
pub(crate) fn read_posting(bytes: &[u8], at: &mut usize) -> Result<Posting, TermIndexError> {
    let superfile = read_varint(bytes, at).ok_or_else(|| malformed("posting superfile"))?;
    let df = read_u64_varint(bytes, at).ok_or_else(|| malformed("posting df"))?;
    let end = at
        .checked_add(F32_LEN)
        .ok_or_else(|| malformed("posting bound"))?;
    let bound_bytes = bytes
        .get(*at..end)
        .ok_or_else(|| malformed("posting bound"))?;
    let bound = f32::from_le_bytes(bound_bytes.try_into().expect("4 bytes"));
    *at = end;
    let tag = *bytes
        .get(*at)
        .ok_or_else(|| malformed("posting location tag"))?;
    *at += 1;
    let location = match tag {
        LOC_NONE => Location::None,
        LOC_PFOR | LOC_SHORT => {
            let offset = read_u64_varint(bytes, at).ok_or_else(|| malformed("location offset"))?;
            let len = read_varint(bytes, at).ok_or_else(|| malformed("location len"))?;
            match tag {
                LOC_PFOR => Location::Pfor { offset, len },
                _ => Location::Short { offset, len },
            }
        }
        LOC_INLINE => {
            let doc_id = read_varint(bytes, at).ok_or_else(|| malformed("inline doc id"))?;
            let tf = read_varint(bytes, at).ok_or_else(|| malformed("inline tf"))?;
            Location::Inline { doc_id, tf }
        }
        other => {
            return Err(TermIndexError::Malformed(format!(
                "posting: unknown location tag {other}"
            )));
        }
    };
    Ok(Posting {
        superfile,
        df,
        bound,
        location,
    })
}

/// Decode a term's run.
pub(crate) fn decode_run(bytes: &[u8]) -> Result<Vec<Posting>, TermIndexError> {
    let mut at = 0usize;
    let n = read_varint(bytes, &mut at).ok_or_else(|| malformed("run count"))? as usize;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_posting(bytes, &mut at)?);
    }
    if at != bytes.len() {
        return Err(malformed("run: trailing bytes"));
    }
    Ok(out)
}

/// A slice layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SliceFormat {
    /// Version 1: read whole.
    Whole,
    /// Version 2: read whole or by range.
    Ranged,
}

impl SliceFormat {
    fn version(self) -> u32 {
        match self {
            Self::Whole => SLICE_VERSION_WHOLE,
            Self::Ranged => SLICE_VERSION_RANGED,
        }
    }

    fn header_len(self) -> usize {
        match self {
            Self::Whole => WHOLE_HEADER_LEN,
            Self::Ranged => RANGED_HEADER_LEN,
        }
    }

    /// Append one encoded run to a postings region.
    pub(crate) fn push_run(self, postings: &mut Vec<u8>, run: &[u8]) {
        postings.extend_from_slice(run);
        if self == Self::Ranged {
            postings.extend_from_slice(&crc32c(run).to_le_bytes());
        }
    }

    /// Block-table bytes for a slice of `n_terms` terms.
    pub(crate) fn table_len(self, n_terms: usize) -> usize {
        match self {
            Self::Whole => 0,
            Self::Ranged => (n_terms.div_ceil(TERM_BLOCK_SIZE) + 1) * CRC_BYTES,
        }
    }
}

/// A slice's fixed header.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SliceHeader {
    format: SliceFormat,
    dict_len: u64,
    table_len: u64,
    postings_len: u64,
}

impl SliceHeader {
    /// Parse a slice's leading bytes; an unknown version is an error.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, TermIndexError> {
        let mut c = Cursor { bytes, at: 0 };
        if c.take(MAGIC_LEN, "slice")? != SLICE_MAGIC {
            return Err(malformed("slice: bad magic"));
        }
        match c.u32("slice")? {
            SLICE_VERSION_WHOLE => Ok(Self {
                format: SliceFormat::Whole,
                dict_len: c.u64("slice dict length")?,
                table_len: 0,
                postings_len: c.u64("slice postings length")?,
            }),
            SLICE_VERSION_RANGED => Ok(Self {
                format: SliceFormat::Ranged,
                dict_len: c.u64("slice dict length")?,
                table_len: c.u64("slice table length")?,
                postings_len: c.u64("slice postings length")?,
            }),
            found => Err(TermIndexError::UnsupportedVersion {
                what: "slice",
                found,
                expected: SLICE_VERSION_RANGED,
            }),
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.format.header_len());
        out.extend_from_slice(SLICE_MAGIC);
        out.extend_from_slice(&self.format.version().to_le_bytes());
        out.extend_from_slice(&self.dict_len.to_le_bytes());
        if self.format == SliceFormat::Ranged {
            out.extend_from_slice(&self.table_len.to_le_bytes());
        }
        out.extend_from_slice(&self.postings_len.to_le_bytes());
        out
    }

    pub(crate) fn format(&self) -> SliceFormat {
        self.format
    }

    fn header_len(&self) -> u64 {
        self.format.header_len() as u64
    }

    fn postings_start(&self) -> u64 {
        match self.format {
            SliceFormat::Whole => self.header_len() + self.dict_len,
            SliceFormat::Ranged => self.header_len(),
        }
    }

    fn dict_start(&self) -> u64 {
        match self.format {
            SliceFormat::Whole => self.header_len(),
            SliceFormat::Ranged => self.header_len() + self.postings_len,
        }
    }

    fn table_start(&self) -> u64 {
        self.dict_start() + self.dict_len
    }

    /// The slice length the header implies. Saturates, so a corrupt header
    /// never matches the real length.
    pub(crate) fn total_len(&self) -> u64 {
        self.header_len()
            .saturating_add(self.dict_len)
            .saturating_add(self.table_len)
            .saturating_add(self.postings_len)
    }

    /// Where the dictionary's footer sits in the slice.
    pub(crate) fn dict_footer_range(&self) -> Range<u64> {
        let end = self.table_start();
        end.saturating_sub(TERM_BLOCKS_FOOTER_BYTES as u64)..end
    }

    /// Where the head's dictionary part starts in the slice, read from the
    /// dictionary's footer.
    pub(crate) fn head_tail_start(&self, footer: &[u8]) -> Result<u64, TermIndexError> {
        TermBlockIndex::tail_start(footer)
            .filter(|at| *at as u64 <= self.dict_len)
            .map(|at| self.dict_start() + at as u64)
            .ok_or_else(|| malformed("slice dictionary footer"))
    }
}

/// A verified version 2 slice head: finds the block that can hold a key,
/// and checks that block and its run.
pub(crate) struct SliceHead {
    header: SliceHeader,
    /// Header, then the slice from the dictionary's block index to the end.
    bytes: Vec<u8>,
}

impl SliceHead {
    /// Parse and verify the head of a slice `slice_len` bytes long.
    pub(crate) fn decode(bytes: Vec<u8>, slice_len: u64) -> Result<Self, TermIndexError> {
        let header = SliceHeader::decode(&bytes)?;
        if header.format != SliceFormat::Ranged || header.total_len() != slice_len {
            return Err(malformed("slice head: header does not match the slice"));
        }
        let rest_len = bytes.len() - RANGED_HEADER_LEN;
        let tail_len = (rest_len as u64)
            .checked_sub(header.table_len)
            .filter(|len| *len <= header.dict_len && header.table_len >= CRC_BYTES as u64)
            .ok_or_else(|| malformed("slice head: length"))? as usize;
        let (covered, stored) = bytes.split_at(bytes.len() - CRC_BYTES);
        if u32_le_at(stored, 0) != Some(crc32c(covered)) {
            return Err(malformed("slice head: CRC mismatch"));
        }
        let tail_offset = header.dict_len as usize - tail_len;
        let head = Self { header, bytes };
        let index = head.index()?;
        if index.tail_offset() != tail_offset
            || header.table_len as usize != (index.n_blocks() + 1) * CRC_BYTES
        {
            return Err(malformed(
                "slice head: block table does not match the dictionary",
            ));
        }
        Ok(head)
    }

    /// The bytes [`Self::decode`] took.
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The dictionary's block index.
    pub(crate) fn index(&self) -> Result<TermBlockIndex<'_>, TermIndexError> {
        let tail_end = self.bytes.len() - self.header.table_len as usize;
        TermBlockIndex::open(
            &self.bytes[RANGED_HEADER_LEN..tail_end],
            self.header.dict_len as usize,
        )
        .map_err(|e| TermIndexError::Malformed(format!("slice dictionary: {e}")))
    }

    /// Where block `b` sits in the slice.
    pub(crate) fn block_range(
        &self,
        index: &TermBlockIndex<'_>,
        b: usize,
    ) -> Result<Range<u64>, TermIndexError> {
        let range = index
            .block_range(b)
            .ok_or_else(|| malformed("slice block past the index"))?;
        let start = self.header.dict_start();
        Ok(start + range.start as u64..start + range.end as u64)
    }

    /// Check block `b`'s bytes against its CRC.
    pub(crate) fn check_block(&self, b: usize, block: &[u8]) -> Result<(), TermIndexError> {
        let table_start = self.bytes.len() - self.header.table_len as usize;
        let at = table_start + b * CRC_BYTES;
        // The last table entry is the head CRC, not a block's.
        match u32_le_at(&self.bytes[..self.bytes.len() - CRC_BYTES], at) {
            Some(crc) if crc == crc32c(block) => Ok(()),
            Some(_) => Err(malformed("slice block: CRC mismatch")),
            None => Err(malformed("slice block: past the block table")),
        }
    }

    /// Where an entry's run and its CRC sit in the slice.
    pub(crate) fn run_range(&self, value: DictEntry) -> Result<Range<u64>, TermIndexError> {
        let run = run_range(value)?;
        let end = run.end as u64 + CRC_BYTES as u64;
        if end > self.header.postings_len {
            return Err(malformed("slice entry range past postings region"));
        }
        let start = self.header.postings_start();
        Ok(start + run.start as u64..start + end)
    }
}

/// Decode a run followed by its CRC; a mismatch is an error.
pub(crate) fn decode_checked_run(bytes: &[u8]) -> Result<Vec<Posting>, TermIndexError> {
    let run_len = bytes
        .len()
        .checked_sub(CRC_BYTES)
        .ok_or_else(|| malformed("run: shorter than its CRC"))?;
    let (run, stored) = bytes.split_at(run_len);
    if u32_le_at(stored, 0) != Some(crc32c(run)) {
        return Err(malformed("run: CRC mismatch"));
    }
    decode_run(run)
}

/// An entry's run within the postings region, without its CRC.
fn run_range(value: DictEntry) -> Result<Range<usize>, TermIndexError> {
    let DictEntry::Pfor {
        metadata_offset,
        postings_length: len,
        ..
    } = value
    else {
        return Err(malformed("slice entry is not a postings range"));
    };
    let start = metadata_offset as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| malformed("slice entry range"))?;
    Ok(start..end)
}

/// Assemble a slice from a block dictionary and a postings region built
/// with [`SliceFormat::push_run`].
pub(crate) fn encode_slice(format: SliceFormat, dict: &[u8], postings: &[u8]) -> Vec<u8> {
    let index = TermBlockIndex::open(dict, dict.len()).expect("a dictionary this build encoded");
    let header = SliceHeader {
        format,
        dict_len: dict.len() as u64,
        table_len: match format {
            SliceFormat::Whole => 0,
            SliceFormat::Ranged => ((index.n_blocks() + 1) * CRC_BYTES) as u64,
        },
        postings_len: postings.len() as u64,
    };
    let mut out = header.encode();
    out.reserve(header.total_len() as usize - out.len());
    match format {
        SliceFormat::Whole => {
            out.extend_from_slice(dict);
            out.extend_from_slice(postings);
        }
        SliceFormat::Ranged => {
            out.extend_from_slice(postings);
            out.extend_from_slice(dict);
            for b in 0..index.n_blocks() {
                let range = index.block_range(b).expect("a block this build encoded");
                out.extend_from_slice(&crc32c(&dict[range]).to_le_bytes());
            }
            let tail = &dict[index.tail_offset()..];
            let table = &out[out.len() - index.n_blocks() * CRC_BYTES..];
            let head_crc = crc32c_append(
                crc32c_append(crc32c(&out[..RANGED_HEADER_LEN]), tail),
                table,
            );
            out.extend_from_slice(&head_crc.to_le_bytes());
        }
    }
    out
}

/// An opened slice: its dictionary and postings region, borrowed.
pub(crate) struct Slice<'a> {
    dict: TermBlocks<'a>,
    postings: &'a [u8],
}

impl<'a> Slice<'a> {
    /// Parse a slice of either layout. CRCs are skipped: a whole slice is
    /// checked by its content hash.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
        let header = SliceHeader::decode(bytes)?;
        if header.total_len() != bytes.len() as u64 {
            return Err(malformed("slice: trailing or missing bytes"));
        }
        let region = |start: u64, len: u64| &bytes[start as usize..(start + len) as usize];
        let dict = TermBlocks::open(region(header.dict_start(), header.dict_len))
            .map_err(|e| TermIndexError::Malformed(format!("slice dictionary: {e}")))?;
        Ok(Self {
            dict,
            postings: region(header.postings_start(), header.postings_len),
        })
    }

    fn run_at(&self, value: DictEntry) -> Result<Vec<Posting>, TermIndexError> {
        let run = self
            .postings
            .get(run_range(value)?)
            .ok_or_else(|| malformed("slice entry range past postings region"))?;
        decode_run(run)
    }

    /// The postings for `key`, `None` when the slice holds no such term.
    pub(crate) fn postings(&self, key: &[u8]) -> Result<Option<Vec<Posting>>, TermIndexError> {
        match self.dict.lookup(key) {
            None => Ok(None),
            Some(v) => self.run_at(v).map(Some),
        }
    }

    /// Up to `limit` terms after `after` (from the first term when `None`),
    /// in key order, with their postings. A walk resumes by passing the
    /// last key it got back.
    pub(crate) fn terms_after(
        &self,
        after: Option<&[u8]>,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<Posting>)>, TermIndexError> {
        let mut out = Vec::new();
        let mut failed: Option<TermIndexError> = None;
        self.dict
            .for_each_from(b"", after.unwrap_or_default(), |key, value| {
                if Some(key) == after {
                    return true;
                }
                match self.run_at(value) {
                    Ok(run) => {
                        out.push((key.to_vec(), run));
                        out.len() < limit
                    }
                    Err(e) => {
                        failed = Some(e);
                        false
                    }
                }
            });
        match failed {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }

    /// Visit every term with `prefix`, in key order, until `visit`
    /// returns `false`. The first decode error stops the walk and is
    /// returned.
    pub(crate) fn for_each_prefix(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], Vec<Posting>) -> bool,
    ) -> Result<(), TermIndexError> {
        let mut failed: Option<TermIndexError> = None;
        self.dict
            .for_each_prefix(prefix, |key, value| match self.run_at(value) {
                Ok(run) => visit(key, run),
                Err(e) => {
                    failed = Some(e);
                    false
                }
            });
        match failed {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::terms::{TermDictBuilder, lookup_in_block, make_key};

    #[test]
    fn hash_width_matches_content_hash() {
        assert_eq!(ContentHash::of(b"").0.len(), HASH_LEN);
    }

    fn sample_postings() -> Vec<Posting> {
        vec![
            Posting {
                superfile: 0,
                df: 1,
                bound: 0.5,
                location: Location::Inline { doc_id: 7, tf: 3 },
            },
            Posting {
                superfile: 3,
                df: 40,
                bound: f32::INFINITY,
                location: Location::Short {
                    offset: 1 << 33,
                    len: 900,
                },
            },
            Posting {
                superfile: 1_000_000,
                df: 1 << 40,
                bound: 12.25,
                location: Location::Pfor {
                    offset: 123_456_789_012,
                    len: u32::MAX - 1,
                },
            },
            Posting {
                superfile: 5,
                df: 2,
                bound: 0.0,
                location: Location::None,
            },
        ]
    }

    #[test]
    fn run_round_trips_every_location_kind() {
        let postings = sample_postings();
        let bytes = encode_run(&postings);
        assert_eq!(decode_run(&bytes).expect("decode"), postings);
    }

    #[test]
    fn run_rejects_truncation_and_unknown_tag() {
        let bytes = encode_run(&sample_postings());
        assert!(decode_run(&bytes[..bytes.len() - 1]).is_err(), "truncated");
        let mut bad = encode_run(&sample_postings()[3..]);
        // The last byte of a `None` posting is its tag.
        *bad.last_mut().expect("tag") = 200;
        assert!(
            matches!(decode_run(&bad), Err(TermIndexError::Malformed(m)) if m.contains("unknown location tag"))
        );
    }

    #[test]
    fn location_mirrors_dictionary_value() {
        assert_eq!(
            Location::from_dict_value(DictEntry::Inline { doc_id: 1, tf: 2 }),
            Location::Inline { doc_id: 1, tf: 2 }
        );
        assert_eq!(
            Location::from_dict_value(DictEntry::Pfor {
                metadata_offset: 10,
                postings_length: 20,
                short: true
            }),
            Location::Short {
                offset: 10,
                len: 20
            }
        );
        assert_eq!(
            Location::from_dict_value(DictEntry::Pfor {
                metadata_offset: 10,
                postings_length: 20,
                short: false
            }),
            Location::Pfor {
                offset: 10,
                len: 20
            }
        );
    }

    fn slice_ref(first: &str, last: &str, tag: u8) -> SliceRef {
        SliceRef {
            first_key: first.as_bytes().to_vec(),
            last_key: last.as_bytes().to_vec(),
            content_hash: ContentHash([tag; HASH_LEN]),
            len: tag as u64 * 1000,
        }
    }

    #[test]
    fn root_round_trips_and_rejects_bad_version() {
        let root = Root {
            superfiles: vec![Uuid::from_u128(1), Uuid::from_u128(2)],
            id_mins: vec![10, -20],
            segments: vec![
                Segment {
                    slices: vec![slice_ref("a", "m", 1), slice_ref("n", "z", 2)],
                },
                Segment {
                    slices: vec![slice_ref("c", "d", 3)],
                },
            ],
        };
        let bytes = root.encode();
        assert_eq!(Root::decode(&bytes).expect("decode"), root);
        assert_eq!(Root::decode(&bytes).expect("decode").id_mins, vec![10, -20]);
        let mut wrong = bytes.clone();
        wrong[MAGIC_LEN] = 9;
        assert!(matches!(
            Root::decode(&wrong),
            Err(TermIndexError::UnsupportedVersion {
                what: "root",
                found: 9,
                expected: ROOT_FORMAT_VERSION
            })
        ));
        let mut magic = bytes.clone();
        magic[0] = b'X';
        assert!(
            matches!(Root::decode(&magic), Err(TermIndexError::Malformed(m)) if m.contains("bad magic"))
        );
        assert!(
            Root::decode(&bytes[..bytes.len() - 3]).is_err(),
            "truncated"
        );
        // The previous layout carried no smallest doc ids: a root written
        // by it names its version, and is refused on that alone.
        let mut previous = bytes.clone();
        previous[MAGIC_LEN..MAGIC_LEN + U32_LEN].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            Root::decode(&previous),
            Err(TermIndexError::UnsupportedVersion { found: 1, .. })
        ));
        // The layout before this one keyed terms by column name; a root
        // written by it is refused the same way and the index rebuilt.
        let mut named = bytes.clone();
        named[MAGIC_LEN..MAGIC_LEN + U32_LEN].copy_from_slice(&2u32.to_le_bytes());
        assert!(matches!(
            Root::decode(&named),
            Err(TermIndexError::UnsupportedVersion { found: 2, .. })
        ));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(
            matches!(Root::decode(&trailing), Err(TermIndexError::Malformed(m)) if m.contains("trailing")),
            "bytes past the last segment are malformed, not ignored"
        );
    }

    /// A prefix with no upper bound (all `0xFF`) runs to the end of each
    /// segment: every slice from the first that can hold it onward.
    #[test]
    fn a_prefix_with_no_upper_bound_scans_to_the_end_of_each_segment() {
        let hi = |a: &[u8], b: &[u8], n: u8| SliceRef {
            first_key: a.to_vec(),
            last_key: b.to_vec(),
            content_hash: ContentHash([n; 32]),
            len: 1,
        };
        let root = Root {
            superfiles: vec![],
            id_mins: vec![],
            segments: vec![Segment {
                slices: vec![
                    slice_ref("a", "m", 1),
                    hi(&[0xFF, 0x01], &[0xFF, 0x05], 2),
                    hi(&[0xFF, 0x06], &[0xFF, 0xFF], 3),
                ],
            }],
        };
        let hashes: Vec<u8> = root
            .slices_for_prefix(&[0xFF])
            .map(|s| s.content_hash.0[0])
            .collect();
        assert_eq!(
            hashes,
            vec![2, 3],
            "from the first slice that can hold it to the end"
        );
    }

    #[test]
    fn root_routes_a_key_to_the_one_slice_per_segment_that_can_hold_it() {
        let root = Root {
            superfiles: vec![],
            id_mins: vec![],
            segments: vec![
                Segment {
                    slices: vec![slice_ref("a", "m", 1), slice_ref("n", "z", 2)],
                },
                Segment {
                    slices: vec![slice_ref("c", "d", 3)],
                },
            ],
        };
        let hits: Vec<u8> = root
            .slices_for_key(b"cat")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert_eq!(hits, vec![1, 3]);
        let hits: Vec<u8> = root
            .slices_for_key(b"q")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert_eq!(hits, vec![2]);
        // Between two slices' ranges: no slice can hold it.
        let hits: Vec<u8> = root
            .slices_for_key(b"mm")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert!(hits.is_empty());
        // Below every slice.
        assert_eq!(root.slices_for_key(b"0").count(), 0);
    }

    #[test]
    fn root_routes_a_prefix_to_every_intersecting_slice() {
        let root = Root {
            superfiles: vec![],
            id_mins: vec![],
            segments: vec![Segment {
                slices: vec![
                    slice_ref("aa", "ab", 1),
                    slice_ref("ac", "b", 2),
                    slice_ref("ba", "c", 3),
                ],
            }],
        };
        let hits: Vec<u8> = root
            .slices_for_prefix(b"a")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert_eq!(
            hits,
            vec![1, 2],
            "a prefix spans the slices whose ranges it touches"
        );
        let hits: Vec<u8> = root
            .slices_for_prefix(b"b")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert_eq!(
            hits,
            vec![2, 3],
            "`b` sits at the end of slice 2 and the start of 3"
        );
        let hits: Vec<u8> = root
            .slices_for_prefix(b"z")
            .map(|s| s.content_hash.0[0])
            .collect();
        assert!(hits.is_empty());
    }

    fn build_slice(format: SliceFormat, entries: &[(&str, &str, Vec<Posting>)]) -> Vec<u8> {
        let mut dict = TermDictBuilder::new();
        let mut postings = Vec::new();
        for (col, term, run) in entries {
            let bytes = encode_run(run);
            dict.insert(
                &make_key(col, term),
                DictEntry::Pfor {
                    metadata_offset: postings.len() as u64,
                    postings_length: bytes.len() as u32,
                    short: false,
                },
            );
            format.push_run(&mut postings, &bytes);
        }
        encode_slice(format, &dict.finish(), &postings)
    }

    fn one(sf: u32) -> Vec<Posting> {
        vec![Posting {
            superfile: sf,
            df: 3,
            bound: 1.0,
            location: Location::None,
        }]
    }

    #[test]
    fn slice_looks_up_terms_and_scans_prefixes_in_both_layouts() {
        for format in [SliceFormat::Whole, SliceFormat::Ranged] {
            slice_looks_up_terms_and_scans_prefixes(format);
        }
    }

    fn slice_looks_up_terms_and_scans_prefixes(format: SliceFormat) {
        let bytes = build_slice(
            format,
            &[
                ("body", "alpha", one(1)),
                ("body", "alphabet", one(2)),
                ("body", "beta", one(3)),
                ("title", "alpha", one(4)),
            ],
        );
        let slice = Slice::open(&bytes).expect("open");
        assert_eq!(
            slice.postings(&make_key("body", "alpha")).expect("ok"),
            Some(one(1))
        );
        assert_eq!(
            slice.postings(&make_key("body", "gamma")).expect("ok"),
            None
        );
        assert_eq!(
            slice.postings(&make_key("title", "beta")).expect("ok"),
            None,
            "column is part of the key"
        );
        let mut seen = Vec::new();
        slice
            .for_each_prefix(&make_key("body", "alph"), |key, run| {
                seen.push((key.to_vec(), run[0].superfile));
                true
            })
            .expect("scan");
        assert_eq!(
            seen,
            vec![
                (make_key("body", "alpha"), 1),
                (make_key("body", "alphabet"), 2)
            ]
        );
        let mut n = 0;
        slice
            .for_each_prefix(&make_key("body", ""), |_, _| {
                n += 1;
                n < 2
            })
            .expect("scan");
        assert_eq!(n, 2, "the visitor can stop the walk");
    }

    #[test]
    fn slice_rejects_unknown_version_loudly_not_as_absent() {
        let mut bytes = build_slice(SliceFormat::Ranged, &[("body", "alpha", vec![])]);
        bytes[MAGIC_LEN] = 3;
        assert!(matches!(
            Slice::open(&bytes),
            Err(TermIndexError::UnsupportedVersion {
                what: "slice",
                found: 3,
                expected: SLICE_VERSION_RANGED
            })
        ));
    }

    /// Enough terms for several dictionary blocks.
    const HEAD_TEST_TERMS: u32 = 5 * TERM_BLOCK_SIZE as u32 + 7;

    /// A version 2 slice of [`HEAD_TEST_TERMS`] terms, and its keys.
    fn ranged_slice() -> (Vec<u8>, Vec<Vec<u8>>) {
        let terms: Vec<String> = (0..HEAD_TEST_TERMS).map(|i| format!("t{i:05}")).collect();
        let entries: Vec<(&str, &str, Vec<Posting>)> = terms
            .iter()
            .zip(0..)
            .map(|(t, i)| ("body", t.as_str(), one(i)))
            .collect();
        let keys = terms.iter().map(|t| make_key("body", t)).collect();
        (build_slice(SliceFormat::Ranged, &entries), keys)
    }

    /// The head of `bytes`, as a range reader assembles it.
    fn head_bytes(bytes: &[u8]) -> Vec<u8> {
        let header = SliceHeader::decode(bytes).expect("header");
        let footer = header.dict_footer_range();
        let from = header
            .head_tail_start(&bytes[footer.start as usize..footer.end as usize])
            .expect("tail start") as usize;
        [&bytes[..RANGED_HEADER_LEN], &bytes[from..]].concat()
    }

    /// Head, block and run reads answer what the whole slice does.
    #[test]
    fn head_block_and_run_reads_match_the_whole_slice() {
        let (bytes, keys) = ranged_slice();
        let slice = Slice::open(&bytes).expect("open");
        let head = SliceHead::decode(head_bytes(&bytes), bytes.len() as u64).expect("head");
        let index = head.index().expect("index");
        assert!(index.n_blocks() > 1, "test spans several blocks");
        let absent = [make_key("body", "t00000a"), make_key("body", "zzz")];
        for key in keys.iter().chain(&absent) {
            let by_range = index.block_for(key).and_then(|b| {
                let range = head.block_range(&index, b).expect("block range");
                let block = &bytes[range.start as usize..range.end as usize];
                head.check_block(b, block).expect("block CRC");
                lookup_in_block(block, key).map(|value| {
                    let run = head.run_range(value).expect("run range");
                    decode_checked_run(&bytes[run.start as usize..run.end as usize]).expect("run")
                })
            });
            assert_eq!(by_range, slice.postings(key).expect("whole"));
        }
        assert!(
            index.block_for(&make_key("a", "")).is_none(),
            "below every key"
        );
    }

    /// A flipped byte in a head, block or run is an error.
    #[test]
    fn head_block_and_run_corruption_is_refused() {
        let (bytes, keys) = ranged_slice();
        let len = bytes.len() as u64;
        let head = head_bytes(&bytes);
        for at in [0, MAGIC_LEN + U32_LEN, RANGED_HEADER_LEN, head.len() - 1] {
            let mut bad = head.clone();
            bad[at] ^= 0xFF;
            assert!(SliceHead::decode(bad, len).is_err(), "head byte {at}");
        }
        assert!(
            SliceHead::decode(head.clone(), len + 1).is_err(),
            "wrong length"
        );
        let head = SliceHead::decode(head, len).expect("head");
        let index = head.index().expect("index");
        let key = &keys[keys.len() / 2];
        let b = index.block_for(key).expect("block");
        let range = head.block_range(&index, b).expect("block range");
        let mut block = bytes[range.start as usize..range.end as usize].to_vec();
        let value = lookup_in_block(&block, key).expect("present");
        block[0] ^= 0xFF;
        assert!(head.check_block(b, &block).is_err(), "corrupt block");
        assert!(
            head.check_block(index.n_blocks(), &block).is_err(),
            "past the table"
        );
        let run = head.run_range(value).expect("run range");
        let mut run = bytes[run.start as usize..run.end as usize].to_vec();
        run[0] ^= 0x01;
        assert!(
            matches!(decode_checked_run(&run), Err(TermIndexError::Malformed(m)) if m.contains("CRC")),
            "corrupt run"
        );
    }

    /// A version 1 slice has no head to read by range.
    #[test]
    fn a_version_1_slice_has_no_head() {
        let bytes = build_slice(SliceFormat::Whole, &[("body", "alpha", one(1))]);
        let header = SliceHeader::decode(&bytes).expect("header");
        assert_eq!(header.format(), SliceFormat::Whole);
        assert!(SliceHead::decode(bytes.clone(), bytes.len() as u64).is_err());
    }
}
