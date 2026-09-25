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
//! **Slice** — one contiguous key range, fetched whole:
//!
//! ```text
//! magic "INFTSLC1" | version u32 | dict_len u64 | postings_len u64
//! | dictionary (front-coded term blocks, `utils::terms`) | postings region
//! ```
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

use uuid::Uuid;

use super::TermIndexError;
use crate::{
    supertable::manifest::{part::ContentHash, term_range::prefix_upper_bound},
    utils::{
        bytes::{u32_le_at, u64_le_at},
        terms::{DictLayout, FstValue, TermDict},
        varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
    },
};

/// Identifies the root file and its major layout family.
pub(crate) const ROOT_MAGIC: &[u8; 8] = b"INFTIDX1";
/// Identifies a slice file and its major layout family.
pub(crate) const SLICE_MAGIC: &[u8; 8] = b"INFTSLC1";
/// Layout version of the root. `2` added each superfile's smallest doc id.
pub(crate) const ROOT_FORMAT_VERSION: u32 = 2;
/// Layout version of a slice.
pub(crate) const SLICE_FORMAT_VERSION: u32 = 1;

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
/// Slice fixed header: magic, version, dictionary length, postings length.
const SLICE_HEADER_LEN: usize = MAGIC_LEN + U32_LEN + U64_LEN + U64_LEN;

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
    pub(crate) fn from_dict_value(value: FstValue) -> Self {
        match value {
            FstValue::Inline { doc_id, tf } => Self::Inline { doc_id, tf },
            FstValue::Pfor {
                metadata_offset,
                postings_length_hint,
                short,
            } => match (postings_length_hint, short) {
                (Some(len), true) => Self::Short {
                    offset: metadata_offset,
                    len,
                },
                (Some(len), false) => Self::Pfor {
                    offset: metadata_offset,
                    len,
                },
                // An FST slot that could not hold the length: the range
                // is not known without reading the header, so it is not
                // carried. Only pre-V7 blobs can produce this.
                (None, _) => Self::None,
            },
        }
    }
}

impl Location {
    /// The superfile dictionary value this location stands for, or `None`
    /// when no location was carried.
    pub(crate) fn to_dict_value(self) -> Option<FstValue> {
        match self {
            Self::None => None,
            Self::Pfor { offset, len } => Some(FstValue::Pfor {
                metadata_offset: offset,
                postings_length_hint: Some(len),
                short: false,
            }),
            Self::Short { offset, len } => Some(FstValue::Pfor {
                metadata_offset: offset,
                postings_length_hint: Some(len),
                short: true,
            }),
            Self::Inline { doc_id, tf } => Some(FstValue::Inline { doc_id, tf }),
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
    /// ordinal. Never reordered — a delta appends.
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
    what: &str,
) -> Result<(), TermIndexError> {
    if c.take(MAGIC_LEN, what)? != magic {
        return Err(malformed(&format!("{what}: bad magic")));
    }
    let version = c.u32(what)?;
    if version != expected {
        return Err(TermIndexError::Malformed(format!(
            "{what}: unsupported version {version} (expected {expected})"
        )));
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

/// Assemble a slice from an encoded block dictionary and its postings
/// region.
pub(crate) fn encode_slice(dict: &[u8], postings: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(SLICE_HEADER_LEN + dict.len() + postings.len());
    out.extend_from_slice(SLICE_MAGIC);
    out.extend_from_slice(&SLICE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(dict.len() as u64).to_le_bytes());
    out.extend_from_slice(&(postings.len() as u64).to_le_bytes());
    out.extend_from_slice(dict);
    out.extend_from_slice(postings);
    out
}

/// An opened slice: its dictionary and postings region, borrowed.
pub(crate) struct Slice<'a> {
    dict: TermDict<'a>,
    postings: &'a [u8],
}

impl<'a> Slice<'a> {
    /// Parse a slice, refusing an unknown magic or version.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
        let mut c = Cursor { bytes, at: 0 };
        check_magic_version(&mut c, SLICE_MAGIC, SLICE_FORMAT_VERSION, "slice")?;
        let dict_len = c.u64("slice dict length")? as usize;
        let postings_len = c.u64("slice postings length")? as usize;
        let dict_bytes = c.take(dict_len, "slice dictionary")?;
        let postings = c.take(postings_len, "slice postings")?;
        if c.at != bytes.len() {
            return Err(malformed("slice: trailing bytes"));
        }
        let dict = TermDict::open(dict_bytes, DictLayout::Blocks)
            .map_err(|e| TermIndexError::Malformed(format!("slice dictionary: {e}")))?;
        Ok(Self { dict, postings })
    }

    fn run_at(&self, value: FstValue) -> Result<Vec<Posting>, TermIndexError> {
        let FstValue::Pfor {
            metadata_offset,
            postings_length_hint: Some(len),
            ..
        } = value
        else {
            return Err(malformed("slice entry is not a postings range"));
        };
        let start = metadata_offset as usize;
        let end = start
            .checked_add(len as usize)
            .ok_or_else(|| malformed("slice entry range"))?;
        let run = self
            .postings
            .get(start..end)
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
    use crate::utils::terms::{TermDictBuilder, make_key};

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
            Location::from_dict_value(FstValue::Inline { doc_id: 1, tf: 2 }),
            Location::Inline { doc_id: 1, tf: 2 }
        );
        assert_eq!(
            Location::from_dict_value(FstValue::Pfor {
                metadata_offset: 10,
                postings_length_hint: Some(20),
                short: true
            }),
            Location::Short {
                offset: 10,
                len: 20
            }
        );
        assert_eq!(
            Location::from_dict_value(FstValue::Pfor {
                metadata_offset: 10,
                postings_length_hint: Some(20),
                short: false
            }),
            Location::Pfor {
                offset: 10,
                len: 20
            }
        );
        assert_eq!(
            Location::from_dict_value(FstValue::Pfor {
                metadata_offset: 10,
                postings_length_hint: None,
                short: false
            }),
            Location::None,
            "an unknown length cannot be carried"
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
        assert!(
            matches!(Root::decode(&wrong), Err(TermIndexError::Malformed(m)) if m.contains("unsupported version 9"))
        );
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
        assert!(
            matches!(Root::decode(&previous), Err(TermIndexError::Malformed(m)) if m.contains("unsupported version 1"))
        );
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

    fn build_slice(entries: &[(&str, &str, Vec<Posting>)]) -> Vec<u8> {
        let mut dict = TermDictBuilder::new(DictLayout::Blocks);
        let mut postings = Vec::new();
        for (col, term, run) in entries {
            let bytes = encode_run(run);
            dict.insert(
                &make_key(col, term),
                FstValue::Pfor {
                    metadata_offset: postings.len() as u64,
                    postings_length_hint: Some(bytes.len() as u32),
                    short: false,
                },
            );
            postings.extend_from_slice(&bytes);
        }
        encode_slice(&dict.finish(), &postings)
    }

    #[test]
    fn slice_looks_up_terms_and_scans_prefixes() {
        let one = |sf: u32| {
            vec![Posting {
                superfile: sf,
                df: 3,
                bound: 1.0,
                location: Location::None,
            }]
        };
        let bytes = build_slice(&[
            ("body", "alpha", one(1)),
            ("body", "alphabet", one(2)),
            ("body", "beta", one(3)),
            ("title", "alpha", one(4)),
        ]);
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
        let mut bytes = build_slice(&[("body", "alpha", vec![])]);
        bytes[MAGIC_LEN] = 2;
        assert!(
            matches!(Slice::open(&bytes), Err(TermIndexError::Malformed(m)) if m.contains("unsupported version 2"))
        );
    }
}
