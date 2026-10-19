// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Building a term index from per-superfile *contributions*.
//!
//! A contribution is one superfile's dictionary walked in key order: for
//! each `(column, term)` its `df`, its score bound and where its postings
//! sit. Contributions are spilled to files as they are produced, so the
//! build never holds more than one superfile's terms plus one slice in
//! memory, however many superfiles the table has: the maintenance pass
//! opens one reader at a time, writes its contribution, drops the reader,
//! and only then merges. The merge is a k-way heap over the sorted files.
//!
//! Slices are cut at term boundaries once the pending slice reaches its
//! target size, so every slice is one contiguous key range and a lookup
//! fetches exactly one.
//!
//! An index can also be rebuilt from its own segments
//! ([`merge_segments`]) rather than from the superfiles: a posting depends
//! only on its superfile, which never changes, so copying it out of an
//! existing segment gives exactly what walking the superfile again would.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, VecDeque},
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    mem,
    path::{Path, PathBuf},
    sync::Arc,
};

use rayon::{ThreadPool, prelude::*};
use tempfile::TempDir;
use uuid::Uuid;

use super::{
    TermIndexError,
    format::{
        Location, ObjectRef, Posting, Root, Run, Segment, Slice, SliceRef, TierRef, encode_run,
        encode_run_with_tier, encode_slice, push_posting, read_posting,
    },
    tier::{EncodedChunk, TierChunk, TierChunkRef, TierEntry, encode_tier},
};
use crate::{
    supertable::manifest::part::ContentHash,
    utils::{
        terms::{DictEntry, TermBlockWriter},
        varint::{CONTINUATION_BIT, push_varint, read_u64_varint, read_varint},
    },
};

/// Target slice size. A cold lookup fetches one slice, so this is the
/// cold cost of a term; a smaller slice means a larger root. **Pending
/// measurement** — the plan's first milestone replaces this with a
/// measured value.
pub(crate) const SLICE_TARGET_BYTES: usize = 8 * 1024 * 1024;

/// Share of a table's documents a term's tier keeps, table-wide. The skip
/// rate of a ranked query is set by `k` against the budget, not by the
/// per-superfile depth, and the budget that serves a given `k` grows with
/// the table: the k-th joint score rises as fast as the candidate pool
/// does. Two documents in ten thousand is what a long-text table affords:
/// on a ten-million-document corpus the tiers then hold about six percent
/// of the postings (the complete lists of the terms under the budget,
/// which are a thin tail, plus the budget's worth of each common term),
/// and the same share serves the same `k` on a larger table.
pub(crate) const TIER_SHARE_OF_DOCS: f64 = 2e-4;
/// Smallest tier budget the share resolves to. A small table has few
/// superfiles, so its opens are cheap whatever the budget, while a large
/// floor would copy most of its postings into tiers.
pub(crate) const TIER_ENTRIES_MIN: u64 = 1 << 11;
/// Largest tier budget the share resolves to: past this the per-term read
/// outgrows what a query at any supported `k` consumes.
pub(crate) const TIER_ENTRIES_MAX: u64 = 1 << 22;

/// The tier budget for a table of `n_docs` documents under the default
/// policy.
pub(crate) fn tier_entries_for(n_docs: u64) -> u64 {
    ((n_docs as f64 * TIER_SHARE_OF_DOCS).round() as u64).clamp(TIER_ENTRIES_MIN, TIER_ENTRIES_MAX)
}
/// Most documents a superfile contributes per term to its head, and the
/// default when the table's shape is not known. The depth a table wants is
/// a few times the budget's even share per superfile (see
/// [`head_depth_for`]), so a superfile richer than average in a term still
/// contributes; the rebuild truncates table-wide.
pub(crate) const HEAD_DEPTH_MAX: usize = 4096;
/// Fewest documents a superfile contributes per term: enough that a term's
/// best documents in one superfile are always on hand.
pub(crate) const HEAD_DEPTH_MIN: usize = 64;
/// How many times the budget's even share a superfile's head holds.
const HEAD_DEPTH_OVER_SHARE: u64 = 4;

/// The per-superfile head depth for a table of `n_superfiles` with a tier
/// budget of `tier_entries` per term.
pub(crate) fn head_depth_for(tier_entries: u64, n_superfiles: u64) -> usize {
    let share = tier_entries.saturating_mul(HEAD_DEPTH_OVER_SHARE) / n_superfiles.max(1);
    (share as usize).clamp(HEAD_DEPTH_MIN, HEAD_DEPTH_MAX)
}

/// Knobs the build takes from its caller.
#[derive(Debug, Clone)]
pub(crate) struct BuildPolicy {
    /// See [`SLICE_TARGET_BYTES`].
    pub(crate) slice_target_bytes: usize,
    /// Entries a term's tier keeps table-wide; `0` keeps no tiers. See
    /// [`tier_entries_for`].
    pub(crate) tier_entries_per_term: u64,
    /// See [`head_depth_for`].
    pub(crate) head_depth_per_superfile: usize,
    /// Where the build encodes tiers in parallel; the process-wide pool
    /// when the caller has none to offer.
    pub(crate) pool: Option<Arc<ThreadPool>>,
}

impl Default for BuildPolicy {
    fn default() -> Self {
        Self {
            slice_target_bytes: SLICE_TARGET_BYTES,
            tier_entries_per_term: TIER_ENTRIES_MAX,
            head_depth_per_superfile: HEAD_DEPTH_MAX,
            pool: None,
        }
    }
}

/// Tier entries a batch of terms gathers before the batch's tiers are
/// sorted, cut and encoded in parallel; a batch also closes at
/// [`ENCODE_BATCH_TERMS`]. Bounds the entries held decoded at once.
const ENCODE_BATCH_ENTRIES: usize = 1 << 21;
/// Most terms in one encoding batch.
const ENCODE_BATCH_TERMS: usize = 1 << 13;

/// One term as the merge hands it over: its postings and the tier entries
/// gathered for it, in any order.
struct TermInput {
    key: Vec<u8>,
    postings: Vec<Posting>,
    entries: Vec<TierEntry>,
    /// Every head that fed `entries` was its term's whole list.
    complete: bool,
}

/// A term's tier, cut to the budget and encoded, before it is placed.
struct EncodedTier {
    complete: bool,
    n_entries: u64,
    chunks: Vec<EncodedChunk>,
}

/// A term ready to be written: postings with their tails raised, and the
/// encoded tier when it keeps one.
struct EncodedTerm {
    key: Vec<u8>,
    postings: Vec<Posting>,
    tier: Option<EncodedTier>,
}

/// Sort, cut and encode one term's tier. The tier is truncated to
/// `budget`; a posting whose superfile loses entries to the truncation has
/// its tail raised to the best impact dropped, so the tail stays an upper
/// bound on what the tier does not hold. The budget-th entry is found by
/// selection, so only the kept entries are sorted.
fn encode_term(mut term: TermInput, budget: u64) -> EncodedTerm {
    if budget == 0 || term.entries.is_empty() {
        return EncodedTerm {
            key: term.key,
            postings: term.postings,
            tier: None,
        };
    }
    let order = |a: &TierEntry, b: &TierEntry| {
        b.impact
            .total_cmp(&a.impact)
            .then(a.superfile.cmp(&b.superfile))
            .then(a.doc_id.cmp(&b.doc_id))
    };
    let budget = budget as usize;
    let truncated = term.entries.len() > budget;
    if truncated {
        term.entries.select_nth_unstable_by(budget, order);
        let (kept, dropped) = term.entries.split_at_mut(budget);
        for d in dropped.iter() {
            if let Ok(i) = term
                .postings
                .binary_search_by_key(&d.superfile, |p| p.superfile)
                && term.postings[i].tail < d.impact
            {
                term.postings[i].tail = d.impact;
            }
        }
        kept.sort_unstable_by(order);
        term.entries.truncate(budget);
    } else {
        term.entries.sort_unstable_by(order);
    }
    let chunks = encode_tier(&term.entries);
    EncodedTerm {
        key: term.key,
        postings: term.postings,
        tier: Some(EncodedTier {
            complete: term.complete && !truncated,
            n_entries: term.entries.len() as u64,
            chunks,
        }),
    }
}

/// One term's head from one superfile, as a contribution carries it.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Head {
    /// `(row, impact)` in descending impact.
    pub(crate) docs: Vec<(u32, f32)>,
    /// The head is the term's whole posting list in that superfile.
    pub(crate) complete: bool,
}

/// One superfile's terms, spilled to a file in key order.
///
/// Record: `record_len varint | key_len varint | key | posting | complete u8
/// | n_head varint | per head doc: row varint, impact f32 LE`, where the
/// posting's superfile ordinal is left zero — the merge assigns ordinals
/// by the order contributions are handed to it. The outer length lets the
/// reader take exactly one record from the stream.
pub(crate) struct ContributionWriter {
    out: BufWriter<File>,
    path: PathBuf,
    /// A scratch directory this writer owns, removed when the finished
    /// contribution is dropped. `None` when the caller supplied the
    /// directory.
    scratch: Option<TempDir>,
    superfile_id: Uuid,
    id_min: i128,
    prev_key: Vec<u8>,
    n_terms: u64,
    record: Vec<u8>,
}

impl ContributionWriter {
    /// Start a contribution for `superfile_id`, whose smallest doc id is
    /// `id_min`, spilling under `dir`.
    pub(crate) fn create(
        dir: &Path,
        superfile_id: Uuid,
        id_min: i128,
    ) -> Result<Self, TermIndexError> {
        fs::create_dir_all(dir)?;
        let path = dir.join(format!("{superfile_id}.terms"));
        let out = BufWriter::new(File::create(&path)?);
        Ok(Self {
            out,
            path,
            scratch: None,
            superfile_id,
            id_min,
            prev_key: Vec::new(),
            n_terms: 0,
            record: Vec::new(),
        })
    }

    /// Like [`Self::create`], spilling into a scratch directory of its own
    /// under the system temporary directory; the directory lives as long as
    /// the finished contribution does.
    pub(crate) fn create_in_scratch(
        superfile_id: Uuid,
        id_min: i128,
    ) -> Result<Self, TermIndexError> {
        let scratch = tempfile::Builder::new()
            .prefix("infino-term-index-")
            .tempdir()?;
        let mut writer = Self::create(scratch.path(), superfile_id, id_min)?;
        writer.scratch = Some(scratch);
        Ok(writer)
    }

    /// Append one term. Keys must arrive in strictly ascending order.
    pub(crate) fn push(
        &mut self,
        key: &[u8],
        df: u64,
        bound: f32,
        location: Location,
        tail: f32,
        head: &Head,
    ) -> Result<(), TermIndexError> {
        if self.n_terms > 0 && key <= self.prev_key.as_slice() {
            return Err(TermIndexError::Build(format!(
                "contribution for {} is not in ascending key order",
                self.superfile_id
            )));
        }
        self.record.clear();
        push_varint(&mut self.record, key.len() as u32);
        self.record.extend_from_slice(key);
        push_posting(
            &mut self.record,
            &Posting {
                superfile: 0,
                df,
                bound,
                tail,
                location,
            },
        );
        self.record.push(head.complete as u8);
        push_varint(&mut self.record, head.docs.len() as u32);
        for &(row, impact) in &head.docs {
            push_varint(&mut self.record, row);
            self.record.extend_from_slice(&impact.to_le_bytes());
        }
        let mut frame = Vec::with_capacity(self.record.len() + 4);
        push_varint(&mut frame, self.record.len() as u32);
        self.out.write_all(&frame)?;
        self.out.write_all(&self.record)?;
        self.prev_key.clear();
        self.prev_key.extend_from_slice(key);
        self.n_terms += 1;
        Ok(())
    }

    /// Flush and hand back the finished contribution.
    pub(crate) fn finish(mut self) -> Result<Contribution, TermIndexError> {
        self.out.flush()?;
        Ok(Contribution {
            superfile_id: self.superfile_id,
            id_min: self.id_min,
            path: self.path,
            _scratch: self.scratch,
        })
    }
}

/// A finished, spilled contribution.
#[derive(Debug)]
pub(crate) struct Contribution {
    /// The superfile these terms came from.
    pub(crate) superfile_id: Uuid,
    /// That superfile's smallest doc id.
    pub(crate) id_min: i128,
    /// The spill file.
    pub(crate) path: PathBuf,
    /// The scratch directory holding it, when this contribution owns one;
    /// dropping the contribution removes both.
    _scratch: Option<TempDir>,
}

/// Sequential reader over one spilled contribution.
struct ContributionReader {
    rd: BufReader<File>,
    buf: Vec<u8>,
}

/// One decoded record: the key, the posting (ordinal not yet set) and the
/// head.
struct Record {
    key: Vec<u8>,
    posting: Posting,
    head: Head,
}

/// Largest byte length one encoded `u64` can occupy (10 × 7 bits).
const MAX_U64_VARINT_BYTES: usize = 10;

/// Read one LEB128 varint from a stream; `None` at a clean end of file
/// *before* the first byte. The bytes are decoded by the crate's one
/// varint reader, so a value the slice reader would refuse is refused
/// here too.
fn read_varint_stream(rd: &mut impl Read) -> io::Result<Option<u64>> {
    let mut buf = [0u8; MAX_U64_VARINT_BYTES];
    let mut n = 0usize;
    loop {
        let mut b = [0u8; 1];
        if rd.read(&mut b)? == 0 {
            return match n {
                0 => Ok(None),
                _ => Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "varint cut short",
                )),
            };
        }
        if n == buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "varint too long",
            ));
        }
        buf[n] = b[0];
        n += 1;
        if b[0] & CONTINUATION_BIT == 0 {
            break;
        }
    }
    let mut at = 0usize;
    read_u64_varint(&buf[..n], &mut at)
        .map(Some)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "varint overflows u64"))
}

impl ContributionReader {
    fn open(path: &Path) -> Result<Self, TermIndexError> {
        Ok(Self {
            rd: BufReader::new(File::open(path)?),
            buf: Vec::new(),
        })
    }

    /// The next record, `None` at end of file.
    fn next(&mut self) -> Result<Option<Record>, TermIndexError> {
        let Some(record_len) = read_varint_stream(&mut self.rd)? else {
            return Ok(None);
        };
        self.buf.clear();
        self.buf.resize(record_len as usize, 0);
        self.rd.read_exact(&mut self.buf)?;
        let mut at = 0usize;
        let key_len = read_varint(&self.buf, &mut at)
            .ok_or_else(|| TermIndexError::Malformed("contribution key length".into()))?
            as usize;
        let key = self
            .buf
            .get(at..at + key_len)
            .ok_or_else(|| TermIndexError::Malformed("contribution key".into()))?
            .to_vec();
        at += key_len;
        let posting = read_posting(&self.buf, &mut at)?;
        let complete = *self
            .buf
            .get(at)
            .ok_or_else(|| TermIndexError::Malformed("contribution head flag".into()))?
            != 0;
        at += 1;
        let n = read_varint(&self.buf, &mut at)
            .ok_or_else(|| TermIndexError::Malformed("contribution head count".into()))?
            as usize;
        let mut docs = Vec::with_capacity(n);
        for _ in 0..n {
            let row = read_varint(&self.buf, &mut at)
                .ok_or_else(|| TermIndexError::Malformed("contribution head row".into()))?;
            let raw = self
                .buf
                .get(at..at + 4)
                .ok_or_else(|| TermIndexError::Malformed("contribution head impact".into()))?;
            docs.push((row, f32::from_le_bytes(raw.try_into().expect("4 bytes"))));
            at += 4;
        }
        if at != self.buf.len() {
            return Err(TermIndexError::Malformed(
                "contribution record: trailing bytes".into(),
            ));
        }
        Ok(Some(Record {
            key,
            posting,
            head: Head { docs, complete },
        }))
    }
}

/// A heap entry: the record's key, and which contribution it came from.
/// Ordered by key then ordinal so equal keys pop in ascending ordinal.
struct MergeHead {
    key: Vec<u8>,
    ordinal: u32,
}

impl PartialEq for MergeHead {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.ordinal == other.ordinal
    }
}
impl Eq for MergeHead {}
impl PartialOrd for MergeHead {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MergeHead {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then(self.ordinal.cmp(&other.ordinal))
    }
}

/// The slice under construction.
struct SliceBuilder {
    /// Keys arrive in order from the merge, so the dictionary streams into
    /// its final bytes and its size is known exactly as it grows.
    dict: TermBlockWriter<Vec<u8>>,
    postings: Vec<u8>,
    first_key: Option<Vec<u8>>,
    last_key: Vec<u8>,
    n_terms: usize,
}

impl SliceBuilder {
    fn new() -> Self {
        Self {
            dict: TermBlockWriter::new(Vec::new()),
            postings: Vec::new(),
            first_key: None,
            last_key: Vec::new(),
            n_terms: 0,
        }
    }

    fn add(&mut self, key: &[u8], run: &[u8]) {
        self.dict
            .insert_sorted(
                key,
                DictEntry::Pfor {
                    metadata_offset: self.postings.len() as u64,
                    postings_length: run.len() as u32,
                    short: false,
                },
            )
            .expect("Vec sink cannot fail");
        self.postings.extend_from_slice(run);
        if self.first_key.is_none() {
            self.first_key = Some(key.to_vec());
        }
        self.last_key.clear();
        self.last_key.extend_from_slice(key);
        self.n_terms += 1;
    }

    /// The slice's size so far, its fixed-size header aside. Measured, not
    /// estimated: how much of a key front-coding keeps depends on the
    /// column, and a column of random keys shares almost no prefix, so a
    /// fixed guess undercounted those slices to about twice the target.
    fn encoded_bytes(&self) -> usize {
        self.postings.len() + self.dict.encoded_len()
    }

    /// The slice and its tier object, when its terms wrote any tier chunks.
    fn finish(self, tier: Option<ObjectRef>) -> Option<(SliceRef, Vec<u8>)> {
        let first_key = self.first_key?;
        let dict = self.dict.finish().expect("Vec sink cannot fail");
        let bytes = encode_slice(&dict, &self.postings);
        let content_hash = ContentHash::of(&bytes);
        Some((
            SliceRef {
                first_key,
                last_key: self.last_key,
                content_hash,
                len: bytes.len() as u64,
                tier,
            },
            bytes,
        ))
    }
}

/// Cuts a stream of `(key, encoded run)` in key order into slices. The
/// build and the merge both write through it, so equal input gives equal
/// bytes.
struct SegmentWriter {
    current: SliceBuilder,
    /// The open slice's tier object so far: its terms' chunks back to back.
    current_tier: Vec<u8>,
    slices: Vec<SliceRef>,
    slice_bytes: Vec<(ContentHash, Vec<u8>)>,
    /// The tier objects of the slices cut so far, keyed by content hash.
    tier_bytes: Vec<(ContentHash, Vec<u8>)>,
    target_bytes: usize,
    tier_entries_per_term: u64,
    /// Terms gathered for the next parallel encoding batch, in key order.
    pending: Vec<TermInput>,
    /// Tier entries across `pending`.
    pending_entries: usize,
    pool: Option<Arc<ThreadPool>>,
}

impl SegmentWriter {
    fn new(policy: &BuildPolicy) -> Self {
        Self {
            current: SliceBuilder::new(),
            current_tier: Vec::new(),
            slices: Vec::new(),
            slice_bytes: Vec::new(),
            tier_bytes: Vec::new(),
            target_bytes: policy.slice_target_bytes,
            tier_entries_per_term: policy.tier_entries_per_term,
            pending: Vec::new(),
            pending_entries: 0,
            pool: policy.pool.clone(),
        }
    }

    /// Append one term, first cutting the slice when the run would push it
    /// past the target.
    fn add(&mut self, key: &[u8], run: &[u8]) {
        if self.cuts_before(run.len()) {
            self.cut();
        }
        self.current.add(key, run);
    }

    /// Whether a run of `run_len` bytes starts a new slice.
    fn cuts_before(&self, run_len: usize) -> bool {
        self.current.n_terms > 0 && self.current.encoded_bytes() + run_len > self.target_bytes
    }

    /// Place `chunks` at the end of the open slice's tier object and name
    /// them from there.
    fn place_chunks(&mut self, chunks: &[EncodedChunk]) -> Vec<TierChunkRef> {
        let mut refs = Vec::with_capacity(chunks.len());
        for c in chunks {
            refs.push(TierChunkRef {
                offset: self.current_tier.len() as u64,
                len: c.bytes.len() as u64,
                n_entries: c.n_entries,
            });
            self.current_tier.extend_from_slice(&c.bytes);
        }
        refs
    }

    /// The chunk list `chunks` would get in the open slice's tier object.
    fn refs_at(&self, chunks: &[EncodedChunk]) -> Vec<TierChunkRef> {
        let mut at = self.current_tier.len() as u64;
        chunks
            .iter()
            .map(|c| {
                let r = TierChunkRef {
                    offset: at,
                    len: c.bytes.len() as u64,
                    n_entries: c.n_entries,
                };
                at += r.len;
                r
            })
            .collect()
    }

    /// Queue one term with its postings (ascending superfile) and the tier
    /// entries gathered for it, in any order; `complete` says every head
    /// that fed `entries` was its term's whole list. Terms are encoded in
    /// parallel a batch at a time and written in the order queued.
    fn add_term(
        &mut self,
        key: &[u8],
        postings: Vec<Posting>,
        entries: Vec<TierEntry>,
        complete: bool,
    ) {
        self.pending_entries += entries.len();
        self.pending.push(TermInput {
            key: key.to_vec(),
            postings,
            entries,
            complete,
        });
        if self.pending_entries >= ENCODE_BATCH_ENTRIES || self.pending.len() >= ENCODE_BATCH_TERMS
        {
            self.flush_pending();
        }
    }

    /// Encode the queued terms' tiers in parallel, then write each in order.
    fn flush_pending(&mut self) {
        let batch = mem::take(&mut self.pending);
        self.pending_entries = 0;
        let budget = self.tier_entries_per_term;
        let encode = || -> Vec<EncodedTerm> {
            batch
                .into_par_iter()
                .map(|term| encode_term(term, budget))
                .collect()
        };
        let encoded = match &self.pool {
            Some(pool) => pool.install(encode),
            None => encode(),
        };
        for term in encoded {
            self.add_encoded(term);
        }
    }

    /// Write one encoded term: its chunks go in the tier object of the slice
    /// the term lands in, so a cut moves them to the fresh slice's object.
    fn add_encoded(&mut self, term: EncodedTerm) {
        let Some(encoded) = term.tier else {
            self.add(&term.key, &encode_run(&term.postings));
            return;
        };
        let mut tier = TierRef {
            complete: encoded.complete,
            n_entries: encoded.n_entries,
            chunks: self.refs_at(&encoded.chunks),
        };
        let mut run = encode_run_with_tier(Some(&tier), &term.postings);
        if self.cuts_before(run.len()) {
            self.cut();
            tier.chunks = self.refs_at(&encoded.chunks);
            run = encode_run_with_tier(Some(&tier), &term.postings);
        }
        let placed = self.place_chunks(&encoded.chunks);
        debug_assert_eq!(placed, tier.chunks);
        self.current.add(&term.key, &run);
    }

    fn cut(&mut self) {
        let tier = (!self.current_tier.is_empty()).then(|| {
            let bytes = mem::take(&mut self.current_tier);
            let reference = ObjectRef {
                content_hash: ContentHash::of(&bytes),
                len: bytes.len() as u64,
            };
            self.tier_bytes.push((reference.content_hash, bytes));
            reference
        });
        if let Some((reference, bytes)) =
            mem::replace(&mut self.current, SliceBuilder::new()).finish(tier)
        {
            self.slice_bytes.push((reference.content_hash, bytes));
            self.slices.push(reference);
        }
    }

    /// The segment, its slices and their tier objects, each written beside
    /// the slice that names it.
    fn finish(
        mut self,
    ) -> (
        Segment,
        Vec<(ContentHash, Vec<u8>)>,
        Vec<(ContentHash, Vec<u8>)>,
    ) {
        self.flush_pending();
        self.cut();
        (
            Segment {
                slices: self.slices,
            },
            self.slice_bytes,
            self.tier_bytes,
        )
    }
}

/// A finished build: the root and every slice, ready to be written.
pub(crate) struct Built {
    /// The root, not yet written.
    pub(crate) root: Root,
    /// Slice bytes keyed by content hash, in key order.
    pub(crate) slices: Vec<(ContentHash, Vec<u8>)>,
    /// The slices' tier objects, keyed by content hash.
    pub(crate) tiers: Vec<(ContentHash, Vec<u8>)>,
}

/// One merged segment: the superfiles it covers (in ordinal order, starting
/// at the base the caller passed), its slice list, and the slice bytes.
pub(crate) struct BuiltSegment {
    /// Superfiles this segment's postings name, in ordinal order.
    pub(crate) superfiles: Vec<Uuid>,
    /// Their smallest doc ids, parallel to `superfiles`.
    pub(crate) id_mins: Vec<i128>,
    /// The segment's slices, in key order.
    pub(crate) segment: Segment,
    /// Slice bytes keyed by content hash, in key order.
    pub(crate) slices: Vec<(ContentHash, Vec<u8>)>,
    /// Tier chunks and the segment's tier list, keyed by content hash.
    pub(crate) tiers: Vec<(ContentHash, Vec<u8>)>,
}

/// Merge contributions into a base root: one segment, ordinals from zero.
#[cfg(test)]
pub(crate) fn build(
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<Built, TermIndexError> {
    build_onto(Root::default(), contributions, policy)
}

/// `base` with `contributions` merged into one more segment, whose
/// ordinals continue from `base`'s superfile count. Onto an empty root this
/// is a full build: one segment, ordinals from zero.
///
/// Optimize skips the rebuild when the current index already has a full
/// build's shape, so a change to what this writes reaches existing tables
/// only if it also bumps `ROOT_FORMAT_VERSION`: an old root then fails to
/// load and is rebuilt.
#[cfg_attr(
    feature = "detailed-tracing",
    tracing::instrument(skip_all, fields(inputs = contributions.len()))
)]
pub(crate) fn build_onto(
    mut base: Root,
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<Built, TermIndexError> {
    let built = build_segment(contributions, policy, base.superfiles.len() as u32)?;
    base.superfiles.extend(built.superfiles);
    base.id_mins.extend(built.id_mins);
    base.segments.push(built.segment);
    Ok(Built {
        root: base,
        slices: built.slices,
        tiers: built.tiers,
    })
}

/// Merge contributions into one segment of slices. Superfile ordinals
/// follow the order of `contributions`, offset by `ordinal_base` — the
/// number of superfiles the root already names when this segment is a
/// delta appended to it.
pub(crate) fn build_segment(
    contributions: &[Contribution],
    policy: &BuildPolicy,
    ordinal_base: u32,
) -> Result<BuiltSegment, TermIndexError> {
    let mut readers = Vec::with_capacity(contributions.len());
    let mut heap: BinaryHeap<Reverse<MergeHead>> = BinaryHeap::new();
    let mut pending: Vec<Option<(Posting, Head)>> = Vec::with_capacity(contributions.len());
    for (ordinal, c) in contributions.iter().enumerate() {
        let mut rd = ContributionReader::open(&c.path)?;
        match rd.next()? {
            Some(rec) => {
                heap.push(Reverse(MergeHead {
                    key: rec.key,
                    ordinal: ordinal as u32,
                }));
                pending.push(Some((rec.posting, rec.head)));
            }
            None => pending.push(None),
        }
        readers.push(rd);
    }

    let mut writer = SegmentWriter::new(policy);
    let mut run: Vec<Posting> = Vec::new();
    let mut entries: Vec<TierEntry> = Vec::new();

    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        entries.clear();
        let mut complete = true;
        let mut take = |ordinal: u32,
                        run: &mut Vec<Posting>,
                        pending: &mut Vec<Option<(Posting, Head)>>,
                        readers: &mut Vec<ContributionReader>,
                        heap: &mut BinaryHeap<Reverse<MergeHead>>|
         -> Result<(), TermIndexError> {
            let (mut p, h) = pending[ordinal as usize]
                .take()
                .expect("pending posting for heap head");
            p.superfile = ordinal_base + ordinal;
            complete &= h.complete;
            entries.extend(h.docs.iter().map(|&(doc_id, impact)| TierEntry {
                superfile: p.superfile,
                doc_id,
                impact,
            }));
            run.push(p);
            if let Some(next) = readers[ordinal as usize].next()? {
                heap.push(Reverse(MergeHead {
                    key: next.key,
                    ordinal,
                }));
                pending[ordinal as usize] = Some((next.posting, next.head));
            }
            Ok(())
        };
        take(
            head.ordinal,
            &mut run,
            &mut pending,
            &mut readers,
            &mut heap,
        )?;
        while let Some(Reverse(peek)) = heap.peek() {
            if peek.key != key {
                break;
            }
            let Reverse(next) = heap.pop().expect("peeked");
            take(
                next.ordinal,
                &mut run,
                &mut pending,
                &mut readers,
                &mut heap,
            )?;
        }
        // Every posting keeps its location, however many superfiles the
        // term is in. A common term opens most of the table, and each open
        // that lacks a location reads the superfile's whole dictionary —
        // megabytes per superfile, per query, where the postings it wants
        // are kilobytes — so the location is worth most exactly where it
        // was once dropped.
        writer.add_term(&key, mem::take(&mut run), mem::take(&mut entries), complete);
    }
    let (segment, slices, tiers) = writer.finish();

    Ok(BuiltSegment {
        superfiles: contributions.iter().map(|c| c.superfile_id).collect(),
        id_mins: contributions.iter().map(|c| c.id_min).collect(),
        segment,
        slices,
        tiers,
    })
}

/// Terms a merge cursor decodes from a slice at a time, so a cursor holds
/// one slice's bytes and a batch of decoded runs rather than the whole
/// slice decoded.
const MERGE_BATCH_TERMS: usize = 4096;

/// Reads one segment's slices, already on local disk, term by term in key
/// order.
struct SegmentCursor {
    /// Slice files not opened yet, in key order.
    files: VecDeque<PathBuf>,
    /// The open slice's bytes; empty before the first file.
    bytes: Vec<u8>,
    /// The open slice's tier object, read on the first tier the slice
    /// yields; `None` until then and after every slice change.
    tier: Option<Vec<u8>>,
    /// Where the open slice's tier object sits (see [`tier_path`]).
    tier_file: Option<PathBuf>,
    /// Decoded terms not handed out yet.
    batch: VecDeque<(Vec<u8>, Run)>,
    /// The last key decoded from the open slice, where the next batch
    /// resumes.
    last_key: Option<Vec<u8>>,
}

impl SegmentCursor {
    fn new(files: &[PathBuf]) -> Self {
        Self {
            files: files.iter().cloned().collect(),
            bytes: Vec::new(),
            tier: None,
            tier_file: None,
            batch: VecDeque::new(),
            last_key: None,
        }
    }

    /// The next term and its run, `None` once every slice is read.
    fn next(&mut self) -> Result<Option<(Vec<u8>, Run)>, TermIndexError> {
        loop {
            if let Some(term) = self.batch.pop_front() {
                return Ok(Some(term));
            }
            if !self.bytes.is_empty() {
                let batch = Slice::open(&self.bytes)?
                    .terms_after(self.last_key.as_deref(), MERGE_BATCH_TERMS)?;
                if let Some((key, _)) = batch.last() {
                    self.last_key = Some(key.clone());
                    self.batch = batch.into();
                    continue;
                }
            }
            let Some(path) = self.files.pop_front() else {
                return Ok(None);
            };
            self.bytes = fs::read(&path)?;
            self.tier = None;
            self.tier_file = Some(path.with_extension(TIER_FILE_EXTENSION));
            self.last_key = None;
        }
    }

    /// The open slice's tier object, read on first use. A run with a tier
    /// always comes from a slice that has one, so a missing file is an
    /// error, not an empty tier.
    fn tier_bytes(&mut self) -> Result<&[u8], TermIndexError> {
        if self.tier.is_none() {
            let path = self
                .tier_file
                .as_ref()
                .ok_or_else(|| TermIndexError::Build("tier read before any slice".into()))?;
            self.tier = Some(fs::read(path)?);
        }
        Ok(self.tier.as_deref().expect("read above"))
    }
}

/// Extension of a slice's tier object on local disk, beside the slice file
/// that shares its stem.
const TIER_FILE_EXTENSION: &str = "tier";

/// Where a slice's tier object sits on local disk once downloaded or
/// spilled: beside the slice file, named by the slice's hash.
pub(crate) fn tier_path(dir: &Path, slice_hash: &ContentHash) -> PathBuf {
    dir.join(format!("{}.{TIER_FILE_EXTENSION}", slice_hash.to_hex()))
}

/// Every entry of a run's tier, read from its chunks in `object` (the
/// slice's tier object), renumbered through `ordinals`; entries of
/// superfiles that are gone are dropped. Impacts come back as the stored
/// lower bounds, which is what a merged tier may carry: the directory
/// bounds are recomputed from them, so they stay exact for the entries the
/// merged blocks hold.
fn tier_entries(
    tier: &TierRef,
    object: &[u8],
    ordinals: &[Option<u32>],
    out: &mut Vec<TierEntry>,
) -> Result<(), TermIndexError> {
    let mut block = Vec::new();
    for chunk in &tier.chunks {
        let end = chunk
            .offset
            .checked_add(chunk.len)
            .filter(|end| *end <= object.len() as u64)
            .ok_or_else(|| TermIndexError::Malformed("tier chunk past its object".into()))?;
        let bytes = &object[chunk.offset as usize..end as usize];
        let chunk = TierChunk::open(bytes)?;
        for i in 0..chunk.n_blocks() {
            chunk.decode_block(i, &mut block)?;
            out.extend(block.iter().filter_map(|e| {
                Some(TierEntry {
                    superfile: (*ordinals.get(e.superfile as usize)?)?,
                    doc_id: e.doc_id,
                    impact: e.impact,
                })
            }));
        }
    }
    Ok(())
}

/// Merge segments into one, renumbering every posting through
/// `ordinals`: a posting naming old ordinal `i` names `ordinals[i]` in the
/// result, and is dropped when that is `None` (a superfile that is gone, or
/// padding). `sources` holds each segment's slice files in key order, each
/// with its tier object beside it (see [`tier_path`]). Tiers merge the same
/// way: their entries are pooled, renumbered and re-cut at the policy's
/// budget.
///
/// Runs come out in ascending new ordinal and are cut into slices exactly
/// as [`build_segment`] cuts them, so a merge whose ordinals follow the
/// order a build would give its contributions writes the same bytes as that
/// build.
pub(crate) fn merge_segments(
    sources: &[Vec<PathBuf>],
    ordinals: &[Option<u32>],
    policy: &BuildPolicy,
) -> Result<MergedSegment, TermIndexError> {
    let mut cursors: Vec<SegmentCursor> = sources.iter().map(|s| SegmentCursor::new(s)).collect();
    let mut pending: Vec<Option<Run>> = Vec::with_capacity(cursors.len());
    let mut heap: BinaryHeap<Reverse<MergeHead>> = BinaryHeap::new();
    for (source, cursor) in cursors.iter_mut().enumerate() {
        match cursor.next()? {
            Some((key, run)) => {
                heap.push(Reverse(MergeHead {
                    key,
                    ordinal: source as u32,
                }));
                pending.push(Some(run));
            }
            None => pending.push(None),
        }
    }

    let mut writer = SegmentWriter::new(policy);
    let mut run: Vec<Posting> = Vec::new();
    let mut entries: Vec<TierEntry> = Vec::new();
    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        entries.clear();
        let mut complete = true;
        let mut next = Some(head.ordinal);
        while let Some(source) = next {
            let source_run = pending[source as usize]
                .take()
                .expect("pending run for heap head");
            match &source_run.tier {
                Some(tier) => {
                    complete &= tier.complete;
                    let object = cursors[source as usize].tier_bytes()?;
                    tier_entries(tier, object, ordinals, &mut entries)?;
                }
                // A run without a tier bounds its postings by their tails
                // alone; the merged tier holds nothing for them and is not
                // complete.
                None => complete = false,
            }
            run.extend(source_run.postings.into_iter().filter_map(|mut p| {
                p.superfile = (*ordinals.get(p.superfile as usize)?)?;
                Some(p)
            }));
            if let Some((key, next_run)) = cursors[source as usize].next()? {
                heap.push(Reverse(MergeHead {
                    key,
                    ordinal: source,
                }));
                pending[source as usize] = Some(next_run);
            }
            next = match heap.peek() {
                Some(Reverse(peek)) if peek.key == key => {
                    heap.pop().map(|Reverse(head)| head.ordinal)
                }
                _ => None,
            };
        }
        if run.is_empty() {
            continue;
        }
        run.sort_unstable_by_key(|p| p.superfile);
        writer.add_term(&key, mem::take(&mut run), mem::take(&mut entries), complete);
    }
    let (segment, slices, tiers) = writer.finish();
    Ok(MergedSegment {
        segment,
        slices,
        tiers,
    })
}

/// What [`merge_segments`] produces: the segment and the bytes it names.
pub(crate) struct MergedSegment {
    pub(crate) segment: Segment,
    pub(crate) slices: Vec<(ContentHash, Vec<u8>)>,
    pub(crate) tiers: Vec<(ContentHash, Vec<u8>)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_budget_is_a_share_of_the_table_within_its_floors() {
        assert_eq!(tier_entries_for(0), TIER_ENTRIES_MIN);
        assert_eq!(
            tier_entries_for(1_000_000),
            TIER_ENTRIES_MIN,
            "1M docs: the floor"
        );
        assert_eq!(
            tier_entries_for(10_000_000),
            TIER_ENTRIES_MIN,
            "10M docs: two in ten thousand is under the floor"
        );
        assert_eq!(
            tier_entries_for(100_000_000),
            20_000,
            "100M docs: two in ten thousand"
        );
        assert_eq!(
            tier_entries_for(1_000_000_000),
            200_000,
            "1B docs: two in ten thousand"
        );
        assert_eq!(
            tier_entries_for(100_000_000_000),
            TIER_ENTRIES_MAX,
            "100B docs: the cap"
        );
        assert_eq!(
            head_depth_for(1_000_000, 1_080),
            3_703,
            "1B at 1 GiB superfiles"
        );
        assert_eq!(
            head_depth_for(10_000, 22),
            1_818,
            "10M at 512 MiB superfiles"
        );
        assert_eq!(
            head_depth_for(10_000, 1),
            HEAD_DEPTH_MAX,
            "one superfile: the cap"
        );
        assert_eq!(
            head_depth_for(0, 1_000),
            HEAD_DEPTH_MIN,
            "no budget: the floor"
        );
    }
}
