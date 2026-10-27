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
//!                | content hash (32 B)
//!                | tier object content hash (32 B, all zero when no run of the slice has one)
//! ```
//!
//! A slice's *tier object* holds its terms' tier chunks (`tier.rs`) back
//! to back; a run names its chunks by offset into it. Both the slice and
//! its tier object are content-addressed, so garbage collection enumerates
//! them from the root alone.
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
//! n_postings varint
//! | tier tag u8: absent | chunked [| n_chunks varint
//!                                  | per chunk: offset varint | byte length varint | n_blocks varint]
//!                      | inline [| n_entries varint | per entry: ordinal varint | row varint | tf varint | len u8]
//! | per posting: superfile ordinal varint (<< 1 | has tail) | df varint
//!                   | bound f32 LE [| tail f32 LE] | location tag u8 [| location payload]
//! ```
//!
//! `tier` names the term's tier chunks (`tier.rs`): per superfile, its
//! best documents, read ahead of any superfile open; absent when the
//! build's policy keeps no tier for the term, inline in the run when it
//! is a handful of entries. `bound` is the largest *key* — the
//! length-normalized term frequency, the factor a score multiplies the
//! idf by — of a document of the term in that superfile, and `tail` the
//! largest key of one the tier does *not* hold — zero when it holds them
//! all — so a reader that has consumed the tier knows what the superfile
//! can still contribute. Keys are at the superfile's own average length
//! and the column's parameters, so a superfile's entries, bound and tail
//! order consistently, and a reader scales them to the table's statistics
//! as it rescales any ceiling.
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

use super::{
    TermIndexError,
    tier::{TierChunkRef, TierEntry, push_chunk_refs, read_chunk_refs},
};
use crate::{
    supertable::manifest::{part::ContentHash, term_range::prefix_upper_bound},
    utils::{
        bytes::{u32_le_at, u64_le_at},
        terms::{DictEntry, TermBlocks},
        varint::{push_u64_varint, push_varint, read_u64_varint, read_varint},
    },
};

/// Identifies the root file and its major layout family.
pub(crate) const ROOT_MAGIC: &[u8; 8] = b"INFTIDX1";
/// Identifies a slice file and its major layout family.
pub(crate) const SLICE_MAGIC: &[u8; 8] = b"INFTSLC1";
/// Layout version of the root. `2` added each superfile's smallest doc id;
/// `3` keys terms by the column's field id instead of its name; `4` names
/// each slice's tier object beside the slice; `5` records each slice's
/// byte length again beside its tier, which the fold at commit sizes
/// segments by. A root at version `4` decodes with every length zero, so
/// its segments fold at the first commit that gathers enough of them; one
/// at version `3` still decodes — its slices are at version 1 and serve
/// ceilings until the next maintenance rebuild writes the current layout;
/// anything older does not, and the index is rebuilt, not translated.
pub(crate) const ROOT_FORMAT_VERSION: u32 = 5;
/// The root version that named tiers but not slice lengths.
pub(crate) const UNSIZED_ROOT_FORMAT_VERSION: u32 = 4;
/// The root version before tiers: slices named with their byte length and
/// no tier object.
pub(crate) const LEGACY_ROOT_FORMAT_VERSION: u32 = 3;
/// Layout version of a slice. `2` adds the per-posting tail and the run's
/// tier reference (a small tier in the run itself, a larger one in the
/// slice's tier object, laid out per superfile in regions of one block
/// per superfile, each block with its own CRC), and stores a posting's
/// bound as a key rather than a score.
pub(crate) const SLICE_FORMAT_VERSION: u32 = 2;
/// The slice version before tiers: a run is its postings alone, each a
/// superfile, a document frequency, a bound that is a BM25 score at the
/// superfile's own idf, and a location.
pub(crate) const LEGACY_SLICE_FORMAT_VERSION: u32 = 1;
/// Run header tag: the term has no tier.
const TIER_ABSENT: u8 = 0;
/// Run header tag: a tier reference follows.
const TIER_PRESENT: u8 = 1;
/// Run header tag: the tier's entries follow in the run itself. A term of a
/// few postings keeps its whole tier here — a handful of bytes — rather
/// than in a chunk of its own, whose fixed cost would outweigh them.
const TIER_INLINE: u8 = 2;

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
    /// Upper bound on the tier key — the length-normalized term frequency
    /// at the superfile's own average length and the column's parameters,
    /// the factor a score multiplies the term's idf by — of any document of
    /// the term in that superfile: the key of its best document.
    /// `f32::INFINITY` is a valid (useless) bound and is what a build that
    /// did not collect bounds writes.
    pub(crate) bound: f32,
    /// Upper bound on the tier key — the length-normalized term frequency
    /// at the superfile's own average length and the column's parameters —
    /// of any document of the term in that superfile that the term's tier
    /// does not hold; `0.0` when the tier holds every one (or the
    /// superfile has none), `f32::INFINITY` when no head was collected for
    /// it.
    pub(crate) tail: f32,
    /// Where the postings sit in that superfile, if carried.
    pub(crate) location: Location,
}

/// A term's tier, as its run names it.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct TierRef {
    /// The chunks in tier order; empty when the entries are inline.
    pub(crate) chunks: Vec<TierChunkRef>,
    /// The entries themselves, in tier order, when the tier is small
    /// enough to ride in the run.
    pub(crate) inline: Vec<TierEntry>,
}

/// A term's run: its tier, if any, and one posting per superfile.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Run {
    pub(crate) tier: Option<TierRef>,
    pub(crate) postings: Vec<Posting>,
    /// Whether the run came from a slice of the layout before tiers:
    /// its postings' bounds are BM25 scores at each superfile's own idf
    /// rather than keys, and carry no tail. A reader divides such a bound
    /// by that idf, which needs the superfile's scored document count.
    pub(crate) legacy_bound: bool,
}

/// A bound stored by the layout before tiers — the term's best BM25 score
/// in a superfile at that superfile's own idf — as the key the current
/// layout stores: the factor the score multiplies the idf by. `n` is the
/// superfile's scored document count; with none known, nothing is bounded.
pub(crate) fn legacy_bound_as_key(bound: f32, df: u64, n: u64) -> f32 {
    if n == 0 {
        return f32::INFINITY;
    }
    let local_idf = crate::superfile::fts::bm25::idf(n, df.min(n));
    if local_idf > 0.0 {
        bound / local_idf
    } else {
        bound
    }
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
    /// The content hash of the slice's tier object — its terms' tier chunks
    /// back to back, in key order — when any of its runs has a tier.
    pub(crate) tier: Option<ContentHash>,
    /// The slice object's byte length, as the root records it; zero in a
    /// root written before lengths were recorded beside tiers.
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

/// Check the magic and read the version, which must be the current one
/// (`accepted[0]`) or a legacy one still read; the version found.
fn check_magic_version(
    c: &mut Cursor<'_>,
    magic: &[u8; 8],
    accepted: &[u32],
    what: &'static str,
) -> Result<u32, TermIndexError> {
    if c.take(MAGIC_LEN, what)? != magic {
        return Err(malformed(&format!("{what}: bad magic")));
    }
    let found = c.u32(what)?;
    if !accepted.contains(&found) {
        return Err(TermIndexError::UnsupportedVersion {
            what,
            found,
            expected: accepted[0],
        });
    }
    Ok(found)
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
                let tier = s.tier.unwrap_or(ContentHash([0u8; HASH_LEN]));
                out.extend_from_slice(&tier.0);
                out.extend_from_slice(&s.len.to_le_bytes());
            }
        }
        out
    }

    /// Parse, refusing an unknown magic or a version no longer read. A
    /// root of the layout before tiers names each slice with its byte
    /// length and no tier object.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, TermIndexError> {
        let mut c = Cursor { bytes, at: 0 };
        let version = check_magic_version(
            &mut c,
            ROOT_MAGIC,
            &[
                ROOT_FORMAT_VERSION,
                UNSIZED_ROOT_FORMAT_VERSION,
                LEGACY_ROOT_FORMAT_VERSION,
            ],
            "root",
        )?;
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
                let (tier, len) = if version == LEGACY_ROOT_FORMAT_VERSION {
                    (None, c.u64("slice length")?)
                } else {
                    let hash = c.take(HASH_LEN, "slice tier hash")?;
                    let tier = match hash.iter().all(|b| *b == 0) {
                        true => None,
                        false => Some(ContentHash(hash.try_into().expect("32 bytes"))),
                    };
                    let len = match version == UNSIZED_ROOT_FORMAT_VERSION {
                        true => 0,
                        false => c.u64("slice length")?,
                    };
                    (tier, len)
                };
                slices.push(SliceRef {
                    first_key,
                    last_key,
                    content_hash,
                    tier,
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
    // The tail rides a flag bit on the ordinal: most terms' tiers hold
    // their whole list in every superfile, and a zero tail costs nothing.
    let has_tail = p.tail != 0.0;
    push_varint(out, (p.superfile << 1) | u32::from(has_tail));
    push_u64_varint(out, p.df);
    out.extend_from_slice(&p.bound.to_le_bytes());
    if has_tail {
        out.extend_from_slice(&p.tail.to_le_bytes());
    }
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

/// Append a document's facts — row, term frequency, length bucket — as a
/// head or contribution record carries them.
pub(crate) fn push_doc_facts(out: &mut Vec<u8>, row: u32, tf: u32, len: u8) {
    push_varint(out, row);
    push_varint(out, tf);
    out.push(len);
}

/// Read what [`push_doc_facts`] wrote at `at`, advancing it; `None` short.
pub(crate) fn read_doc_facts(bytes: &[u8], at: &mut usize) -> Option<(u32, u32, u8)> {
    let row = read_varint(bytes, at)?;
    let tf = read_varint(bytes, at)?;
    let len = *bytes.get(*at)?;
    *at += 1;
    Some((row, tf, len))
}

/// Encode a term's run: count, the tier reference, then its postings in
/// ascending superfile ordinal.
pub(crate) fn encode_run(tier: Option<&TierRef>, postings: &[Posting]) -> Vec<u8> {
    let mut out = Vec::with_capacity(postings.len() * (1 + 2 + 2 * F32_LEN + 1 + 8));
    push_varint(&mut out, postings.len() as u32);
    match tier {
        None => out.push(TIER_ABSENT),
        Some(t) if t.chunks.is_empty() => {
            out.push(TIER_INLINE);
            push_varint(&mut out, t.inline.len() as u32);
            for e in &t.inline {
                push_varint(&mut out, e.superfile);
                push_doc_facts(&mut out, e.doc_id, e.tf, e.len);
            }
        }
        Some(t) => {
            out.push(TIER_PRESENT);
            push_chunk_refs(&mut out, &t.chunks);
        }
    }
    for p in postings {
        push_posting(&mut out, p);
    }
    out
}

/// Decode a posting's location at `at`, advancing it.
fn read_location(bytes: &[u8], at: &mut usize) -> Result<Location, TermIndexError> {
    let tag = *bytes
        .get(*at)
        .ok_or_else(|| malformed("posting location tag"))?;
    *at += 1;
    Ok(match tag {
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
    })
}

/// Decode one posting at `at`, advancing it.
pub(crate) fn read_posting(bytes: &[u8], at: &mut usize) -> Result<Posting, TermIndexError> {
    let tagged = read_varint(bytes, at).ok_or_else(|| malformed("posting superfile"))?;
    let (superfile, has_tail) = (tagged >> 1, tagged & 1 == 1);
    let df = read_u64_varint(bytes, at).ok_or_else(|| malformed("posting df"))?;
    let end = at
        .checked_add(F32_LEN)
        .ok_or_else(|| malformed("posting bound"))?;
    let bound_bytes = bytes
        .get(*at..end)
        .ok_or_else(|| malformed("posting bound"))?;
    let bound = f32::from_le_bytes(bound_bytes.try_into().expect("4 bytes"));
    *at = end;
    let tail = if has_tail {
        let end = at
            .checked_add(F32_LEN)
            .ok_or_else(|| malformed("posting tail"))?;
        let tail_bytes = bytes
            .get(*at..end)
            .ok_or_else(|| malformed("posting tail"))?;
        *at = end;
        f32::from_le_bytes(tail_bytes.try_into().expect("4 bytes"))
    } else {
        0.0
    };
    let location = read_location(bytes, at)?;
    Ok(Posting {
        superfile,
        df,
        bound,
        tail,
        location,
    })
}

/// Decode a term's run, postings only.
#[cfg(test)]
pub(crate) fn decode_run(bytes: &[u8]) -> Result<Vec<Posting>, TermIndexError> {
    decode_run_with_tier(bytes).map(|r| r.postings)
}

/// Decode a term's run: its tier reference and its postings.
pub(crate) fn decode_run_with_tier(bytes: &[u8]) -> Result<Run, TermIndexError> {
    let mut at = 0usize;
    let n = read_varint(bytes, &mut at).ok_or_else(|| malformed("run count"))? as usize;
    let tag = *bytes.get(at).ok_or_else(|| malformed("run tier tag"))?;
    at += 1;
    let tier = match tag {
        TIER_ABSENT => None,
        TIER_PRESENT => {
            let chunks = read_chunk_refs(bytes, &mut at)?;
            Some(TierRef {
                chunks,
                inline: Vec::new(),
            })
        }
        TIER_INLINE => {
            let n =
                read_varint(bytes, &mut at).ok_or_else(|| malformed("run tier entries"))? as usize;
            let mut inline = Vec::with_capacity(n.min(64));
            for _ in 0..n {
                let superfile =
                    read_varint(bytes, &mut at).ok_or_else(|| malformed("inline entry ordinal"))?;
                let (doc_id, tf, len) =
                    read_doc_facts(bytes, &mut at).ok_or_else(|| malformed("inline entry"))?;
                inline.push(TierEntry {
                    superfile,
                    doc_id,
                    tf,
                    len,
                });
            }
            Some(TierRef {
                chunks: Vec::new(),
                inline,
            })
        }
        other => {
            return Err(TermIndexError::Malformed(format!(
                "run: unknown tier tag {other}"
            )));
        }
    };
    let mut postings = Vec::with_capacity(n);
    for _ in 0..n {
        postings.push(read_posting(bytes, &mut at)?);
    }
    if at != bytes.len() {
        return Err(malformed("run: trailing bytes"));
    }
    Ok(Run {
        tier,
        postings,
        legacy_bound: false,
    })
}

/// Decode a run of the layout before tiers: a count, then postings of a
/// superfile ordinal, a document frequency, a bound (a BM25 score at the
/// superfile's own idf) and a location; no tail, no tier.
fn decode_legacy_run(bytes: &[u8]) -> Result<Run, TermIndexError> {
    let mut at = 0usize;
    let n = read_varint(bytes, &mut at).ok_or_else(|| malformed("run count"))? as usize;
    let mut postings = Vec::with_capacity(n);
    for _ in 0..n {
        let superfile =
            read_varint(bytes, &mut at).ok_or_else(|| malformed("posting superfile"))?;
        let df = read_u64_varint(bytes, &mut at).ok_or_else(|| malformed("posting df"))?;
        let end = at
            .checked_add(F32_LEN)
            .ok_or_else(|| malformed("posting bound"))?;
        let bound_bytes = bytes
            .get(at..end)
            .ok_or_else(|| malformed("posting bound"))?;
        let bound = f32::from_le_bytes(bound_bytes.try_into().expect("4 bytes"));
        at = end;
        let location = read_location(bytes, &mut at)?;
        postings.push(Posting {
            superfile,
            df,
            bound,
            tail: f32::INFINITY,
            location,
        });
    }
    if at != bytes.len() {
        return Err(malformed("run: trailing bytes"));
    }
    Ok(Run {
        tier: None,
        postings,
        legacy_bound: true,
    })
}

/// Encode `root` as the layout before tiers wrote it: each slice named
/// with its byte length (`lens`, parallel to the slices in order) and no
/// tier object. What a table indexed by that layout holds.
/// A root at [`UNSIZED_ROOT_FORMAT_VERSION`]: tiers named, no lengths.
#[cfg(test)]
pub(crate) fn encode_unsized_root(root: &Root) -> Vec<u8> {
    let sized = root.encode();
    let mut at = ROOT_HEADER_LEN + root.superfiles.len() * (UUID_LEN + I128_LEN) + U32_LEN;
    let mut out = sized[..at].to_vec();
    out[MAGIC_LEN..MAGIC_LEN + U32_LEN].copy_from_slice(&UNSIZED_ROOT_FORMAT_VERSION.to_le_bytes());
    for seg in &root.segments {
        out.extend_from_slice(&sized[at..at + U32_LEN]);
        at += U32_LEN;
        for s in &seg.slices {
            let n = U32_LEN + s.first_key.len() + U32_LEN + s.last_key.len() + HASH_LEN + HASH_LEN;
            out.extend_from_slice(&sized[at..at + n]);
            at += n + 8;
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn encode_legacy_root(root: &Root, lens: &[u64]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(ROOT_MAGIC);
    out.extend_from_slice(&LEGACY_ROOT_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(root.superfiles.len() as u32).to_le_bytes());
    for (id, id_min) in root.superfiles.iter().zip(&root.id_mins) {
        out.extend_from_slice(id.as_bytes());
        out.extend_from_slice(&id_min.to_le_bytes());
    }
    out.extend_from_slice(&(root.segments.len() as u32).to_le_bytes());
    let mut lens = lens.iter();
    for seg in &root.segments {
        out.extend_from_slice(&(seg.slices.len() as u32).to_le_bytes());
        for s in &seg.slices {
            push_bytes_u32_len(&mut out, &s.first_key);
            push_bytes_u32_len(&mut out, &s.last_key);
            out.extend_from_slice(&s.content_hash.0);
            out.extend_from_slice(&lens.next().copied().unwrap_or(0).to_le_bytes());
        }
    }
    out
}

/// Encode a slice as the layout before tiers wrote it: per term, a run of
/// `(superfile, df, bound, location)` postings, the bound a BM25 score at
/// the superfile's own idf.
#[cfg(test)]
pub(crate) fn encode_legacy_slice(terms: &[(Vec<u8>, Vec<(u32, u64, f32, Location)>)]) -> Vec<u8> {
    use crate::utils::terms::TermDictBuilder;
    let mut dict = TermDictBuilder::new();
    let mut postings = Vec::new();
    for (key, run) in terms {
        let mut bytes = Vec::new();
        push_varint(&mut bytes, run.len() as u32);
        for &(superfile, df, bound, location) in run {
            push_varint(&mut bytes, superfile);
            push_u64_varint(&mut bytes, df);
            bytes.extend_from_slice(&bound.to_le_bytes());
            match location {
                Location::None => bytes.push(LOC_NONE),
                Location::Pfor { offset, len } => {
                    bytes.push(LOC_PFOR);
                    push_u64_varint(&mut bytes, offset);
                    push_varint(&mut bytes, len);
                }
                Location::Short { offset, len } => {
                    bytes.push(LOC_SHORT);
                    push_u64_varint(&mut bytes, offset);
                    push_varint(&mut bytes, len);
                }
                Location::Inline { doc_id, tf } => {
                    bytes.push(LOC_INLINE);
                    push_varint(&mut bytes, doc_id);
                    push_varint(&mut bytes, tf);
                }
            }
        }
        dict.insert(
            key,
            DictEntry::Pfor {
                metadata_offset: postings.len() as u64,
                postings_length: bytes.len() as u32,
                short: false,
            },
        );
        postings.extend_from_slice(&bytes);
    }
    let dict = dict.finish();
    let mut out = Vec::with_capacity(SLICE_HEADER_LEN + dict.len() + postings.len());
    out.extend_from_slice(SLICE_MAGIC);
    out.extend_from_slice(&LEGACY_SLICE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(dict.len() as u64).to_le_bytes());
    out.extend_from_slice(&(postings.len() as u64).to_le_bytes());
    out.extend_from_slice(&dict);
    out.extend_from_slice(&postings);
    out
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

/// Where one dictionary block of a slice starts; see [`Slice::block_marks`].
#[derive(Debug, Clone)]
pub(crate) struct BlockMark {
    /// The block's first key.
    pub(crate) key: Vec<u8>,
    /// Postings offset of that key's run.
    pub(crate) postings_at: u64,
    /// Offset of the run's first tier chunk in the slice's tier object,
    /// when it has chunks.
    pub(crate) tier_at: Option<u64>,
}

/// An opened slice: its dictionary and postings region, borrowed.
pub(crate) struct Slice<'a> {
    dict: TermBlocks<'a>,
    postings: &'a [u8],
    /// Whether the slice is of the layout before tiers (see
    /// [`Run::legacy_bound`]).
    legacy: bool,
}

impl<'a> Slice<'a> {
    /// Parse a slice, refusing an unknown magic or a version no longer
    /// read; a slice of the layout before tiers is read with its runs'
    /// bounds marked as the scores they are.
    pub(crate) fn open(bytes: &'a [u8]) -> Result<Self, TermIndexError> {
        let mut c = Cursor { bytes, at: 0 };
        let version = check_magic_version(
            &mut c,
            SLICE_MAGIC,
            &[SLICE_FORMAT_VERSION, LEGACY_SLICE_FORMAT_VERSION],
            "slice",
        )?;
        let dict_len = c.u64("slice dict length")? as usize;
        let postings_len = c.u64("slice postings length")? as usize;
        let dict_bytes = c.take(dict_len, "slice dictionary")?;
        let postings = c.take(postings_len, "slice postings")?;
        if c.at != bytes.len() {
            return Err(malformed("slice: trailing bytes"));
        }
        let dict = TermBlocks::open(dict_bytes)
            .map_err(|e| TermIndexError::Malformed(format!("slice dictionary: {e}")))?;
        Ok(Self {
            dict,
            postings,
            legacy: version == LEGACY_SLICE_FORMAT_VERSION,
        })
    }

    fn run_at(&self, value: DictEntry) -> Result<Run, TermIndexError> {
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
        let run = self
            .postings
            .get(start..end)
            .ok_or_else(|| malformed("slice entry range past postings region"))?;
        if self.legacy {
            decode_legacy_run(run)
        } else {
            decode_run_with_tier(run)
        }
    }

    /// Where each dictionary block starts: its first key, the postings
    /// offset of that key's run, and the offset of its first tier chunk when
    /// the run has chunks. Runs and chunks sit in key order, so the bytes
    /// between two consecutive marks are the bytes of the terms between
    /// them: what a merge weighs when it cuts the key space into shards.
    pub(crate) fn block_marks(&self) -> Result<Vec<BlockMark>, TermIndexError> {
        let mut out = Vec::with_capacity(self.dict.n_blocks());
        for b in 0..self.dict.n_blocks() {
            let Some((key, value)) = self.dict.block_first(b) else {
                return Err(malformed("slice dictionary block"));
            };
            let DictEntry::Pfor {
                metadata_offset, ..
            } = value
            else {
                return Err(malformed("slice entry is not a postings range"));
            };
            let tier_at = self
                .run_at(value)?
                .tier
                .and_then(|t| t.chunks.iter().map(|c| c.offset).min());
            out.push(BlockMark {
                key: key.to_vec(),
                postings_at: metadata_offset,
                tier_at,
            });
        }
        Ok(out)
    }

    /// Bytes in the slice's postings region.
    pub(crate) fn postings_len(&self) -> u64 {
        self.postings.len() as u64
    }

    /// The postings for `key`, `None` when the slice holds no such term.
    #[cfg(test)]
    pub(crate) fn postings(&self, key: &[u8]) -> Result<Option<Vec<Posting>>, TermIndexError> {
        Ok(self.run(key)?.map(|r| r.postings))
    }

    /// The run for `key` — tier reference and postings — `None` when the
    /// slice holds no such term.
    pub(crate) fn run(&self, key: &[u8]) -> Result<Option<Run>, TermIndexError> {
        match self.dict.lookup(key) {
            None => Ok(None),
            Some(v) => self.run_at(v).map(Some),
        }
    }

    /// Up to `limit` terms from `start` on, `start` included (from the
    /// first term when `None`), in key order, with their runs.
    pub(crate) fn terms_from(
        &self,
        start: Option<&[u8]>,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, Run)>, TermIndexError> {
        let mut out = Vec::new();
        let mut failed: Option<TermIndexError> = None;
        self.dict
            .for_each_from(b"", start.unwrap_or_default(), |key, value| {
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

    /// Up to `limit` terms after `after` (from the first term when `None`),
    /// in key order, with their runs. A walk resumes by passing the last
    /// key it got back.
    pub(crate) fn terms_after(
        &self,
        after: Option<&[u8]>,
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, Run)>, TermIndexError> {
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
                Ok(run) => visit(key, run.postings),
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
                tail: 0.0,
                location: Location::Inline { doc_id: 7, tf: 3 },
            },
            Posting {
                superfile: 3,
                df: 40,
                bound: f32::INFINITY,
                tail: f32::INFINITY,
                location: Location::Short {
                    offset: 1 << 33,
                    len: 900,
                },
            },
            Posting {
                superfile: 1_000_000,
                df: 1 << 40,
                bound: 12.25,
                tail: 11.5,
                location: Location::Pfor {
                    offset: 123_456_789_012,
                    len: u32::MAX - 1,
                },
            },
            Posting {
                superfile: 5,
                df: 2,
                bound: 0.0,
                tail: 0.0,
                location: Location::None,
            },
        ]
    }

    fn sample_tier() -> TierRef {
        TierRef {
            chunks: (0..2)
                .map(|i| TierChunkRef {
                    offset: i * 4_000_000,
                    len: 4_000_000 - i,
                    n_blocks: 1_078,
                })
                .collect(),
            inline: Vec::new(),
        }
    }

    /// A tier small enough to ride in the run.
    fn sample_inline_tier() -> TierRef {
        TierRef {
            chunks: Vec::new(),
            inline: vec![
                TierEntry {
                    superfile: 0,
                    doc_id: 7,
                    tf: 3,
                    len: 12,
                },
                TierEntry {
                    superfile: 3,
                    doc_id: 100_000,
                    tf: 2,
                    len: 12,
                },
                TierEntry {
                    superfile: 3,
                    doc_id: 100_001,
                    tf: 1,
                    len: 200,
                },
            ],
        }
    }

    #[test]
    fn run_round_trips_its_tier_reference() {
        let postings = sample_postings();
        let bytes = encode_run(Some(&sample_tier()), &postings);
        let run = decode_run_with_tier(&bytes).expect("decode");
        assert_eq!(run.tier, Some(sample_tier()));
        assert_eq!(run.postings, postings);
        assert_eq!(decode_run(&bytes).expect("postings only"), postings);
        let inline = encode_run(Some(&sample_inline_tier()), &postings);
        let run = decode_run_with_tier(&inline).expect("decode");
        assert_eq!(run.tier, Some(sample_inline_tier()));
        assert_eq!(run.postings, postings);
        assert_eq!(decode_run(&inline).expect("postings only"), postings);
        assert!(
            decode_run_with_tier(&inline[..inline.len() - 1]).is_err(),
            "a truncated inline tier is refused"
        );
        let without = encode_run(None, &postings);
        assert_eq!(decode_run_with_tier(&without).expect("decode").tier, None);
        let mut bad = without.clone();
        bad[1] = 7;
        assert!(
            matches!(decode_run_with_tier(&bad), Err(TermIndexError::Malformed(m)) if m.contains("unknown tier tag"))
        );
    }

    #[test]
    fn run_round_trips_every_location_kind() {
        let postings = sample_postings();
        let bytes = encode_run(None, &postings);
        assert_eq!(decode_run(&bytes).expect("decode"), postings);
    }

    #[test]
    fn run_rejects_truncation_and_unknown_tag() {
        let bytes = encode_run(None, &sample_postings());
        assert!(decode_run(&bytes[..bytes.len() - 1]).is_err(), "truncated");
        let mut bad = encode_run(None, &sample_postings()[3..]);
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
            tier: tag
                .is_multiple_of(2)
                .then_some(ContentHash([tag ^ 0xFF; HASH_LEN])),
            len: u64::from(tag) * 1000,
        }
    }

    #[test]
    fn a_root_records_each_slice_length_and_an_unsized_root_reads_as_zero() {
        let root = Root {
            superfiles: vec![Uuid::from_u128(1)],
            id_mins: vec![7],
            segments: vec![
                Segment {
                    slices: vec![slice_ref("a", "b", 1), slice_ref("c", "d", 2)],
                },
                Segment {
                    slices: vec![slice_ref("c", "d", 3)],
                },
            ],
        };
        let back = Root::decode(&root.encode()).expect("decodes");
        assert_eq!(back, root);
        let lens: Vec<u64> = back
            .segments
            .iter()
            .flat_map(|s| s.slices.iter().map(|r| r.len))
            .collect();
        assert_eq!(lens, vec![1000, 2000, 3000]);
        let without = Root::decode(&encode_unsized_root(&root)).expect("an unsized root decodes");
        let mut expect = root.clone();
        for r in expect.segments.iter_mut().flat_map(|s| s.slices.iter_mut()) {
            r.len = 0;
        }
        assert_eq!(
            without, expect,
            "lengths read as zero, everything else as written"
        );
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
            tier: None,
            len: 0,
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
        let mut dict = TermDictBuilder::new();
        let mut postings = Vec::new();
        for (col, term, run) in entries {
            let bytes = encode_run(None, run);
            dict.insert(
                &make_key(col, term),
                DictEntry::Pfor {
                    metadata_offset: postings.len() as u64,
                    postings_length: bytes.len() as u32,
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
                tail: 1.0,
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

    /// A root written by the layout before tiers decodes: its slices carry
    /// no tier object, and the length it named each with is dropped.
    #[test]
    fn a_root_of_the_layout_before_tiers_decodes_without_tier_objects() {
        let untiered = |a: &str, b: &str, n: u8| SliceRef {
            tier: None,
            ..slice_ref(a, b, n)
        };
        let root = Root {
            superfiles: vec![Uuid::from_u128(1), Uuid::from_u128(2)],
            id_mins: vec![10, -20],
            segments: vec![
                Segment {
                    slices: vec![untiered("a", "b", 1), untiered("c", "d", 2)],
                },
                Segment {
                    slices: vec![untiered("c", "d", 3)],
                },
            ],
        };
        let mut root = root;
        for (r, len) in root
            .segments
            .iter_mut()
            .flat_map(|s| s.slices.iter_mut())
            .zip([100u64, 200, 300])
        {
            r.len = len;
        }
        let legacy = encode_legacy_root(&root, &[100, 200, 300]);
        assert_eq!(
            u32::from_le_bytes(
                legacy[MAGIC_LEN..MAGIC_LEN + U32_LEN]
                    .try_into()
                    .expect("4")
            ),
            LEGACY_ROOT_FORMAT_VERSION
        );
        assert_eq!(Root::decode(&legacy).expect("decodes"), root);
        assert_ne!(
            legacy,
            root.encode(),
            "the current layout writes its own version"
        );
    }

    /// A slice written by the layout before tiers opens; its runs carry no
    /// tier and no tail, and are marked for what their bounds are: scores
    /// at the superfile's own idf, which become keys through that idf.
    #[test]
    fn a_slice_of_the_layout_before_tiers_opens_with_its_bounds_as_scores() {
        let key = make_key("body", "alpha");
        let bytes = encode_legacy_slice(&[(
            key.clone(),
            vec![
                (0, 5, 2.5, Location::None),
                (3, 40, 1.25, Location::Pfor { offset: 77, len: 9 }),
            ],
        )]);
        let slice = Slice::open(&bytes).expect("a legacy slice opens");
        let run = slice.run(&key).expect("run").expect("indexed");
        assert!(run.legacy_bound && run.tier.is_none());
        assert_eq!(run.postings.len(), 2);
        assert_eq!(
            (
                run.postings[0].superfile,
                run.postings[0].df,
                run.postings[0].bound
            ),
            (0, 5, 2.5)
        );
        assert_eq!(
            run.postings[1].location,
            Location::Pfor { offset: 77, len: 9 }
        );
        assert!(
            run.postings.iter().all(|p| p.tail == f32::INFINITY),
            "no tail was stored: nothing below the bound is known"
        );
        assert!(slice.run(&make_key("body", "zzz")).expect("run").is_none());
        // The conversion main's ceilings made: score over the local idf.
        let local_idf = crate::superfile::fts::bm25::idf(100, 5);
        assert_eq!(legacy_bound_as_key(2.5, 5, 100), 2.5 / local_idf);
        assert_eq!(
            legacy_bound_as_key(2.5, 5, 0),
            f32::INFINITY,
            "with no document count the bound is unknown"
        );
        // The current layout's runs are not marked.
        let current = build_slice(&[("body", "alpha", vec![])]);
        let run = Slice::open(&current)
            .expect("open")
            .run(&key)
            .expect("run")
            .expect("indexed");
        assert!(!run.legacy_bound);
    }

    #[test]
    fn slice_rejects_unknown_version_loudly_not_as_absent() {
        let mut bytes = build_slice(&[("body", "alpha", vec![])]);
        bytes[MAGIC_LEN] = 9;
        assert!(matches!(
            Slice::open(&bytes),
            Err(TermIndexError::UnsupportedVersion {
                what: "slice",
                found: 9,
                expected: SLICE_FORMAT_VERSION
            })
        ));
    }
}
