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
};

use tempfile::TempDir;
use uuid::Uuid;

use super::{
    TermIndexError,
    format::{
        Location, Posting, Root, Segment, Slice, SliceFormat, SliceRef, encode_run, encode_slice,
        push_posting, read_posting,
    },
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

/// Layout new slices are written in. Version 1 until every engine reading
/// a table can read version 2; an older engine fails on it.
pub(crate) const SLICE_WRITE_FORMAT: SliceFormat = SliceFormat::Whole;

/// Knobs the build takes from its caller.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BuildPolicy {
    /// See [`SLICE_TARGET_BYTES`].
    pub(crate) slice_target_bytes: usize,
    /// See [`SLICE_WRITE_FORMAT`].
    pub(crate) slice_format: SliceFormat,
}

impl Default for BuildPolicy {
    fn default() -> Self {
        Self {
            slice_target_bytes: SLICE_TARGET_BYTES,
            slice_format: SLICE_WRITE_FORMAT,
        }
    }
}

/// One superfile's terms, spilled to a file in key order.
///
/// Record: `record_len varint | key_len varint | key | posting`, where the
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
                location,
            },
        );
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

/// One decoded record: the key and the posting (ordinal not yet set).
struct Record {
    key: Vec<u8>,
    posting: Posting,
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
        if at != self.buf.len() {
            return Err(TermIndexError::Malformed(
                "contribution record: trailing bytes".into(),
            ));
        }
        Ok(Some(Record { key, posting }))
    }
}

/// A heap entry: the record's key, and which contribution it came from.
/// Ordered by key then ordinal so equal keys pop in ascending ordinal.
struct Head {
    key: Vec<u8>,
    ordinal: u32,
}

impl PartialEq for Head {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.ordinal == other.ordinal
    }
}
impl Eq for Head {}
impl PartialOrd for Head {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Head {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then(self.ordinal.cmp(&other.ordinal))
    }
}

/// The slice under construction.
struct SliceBuilder {
    format: SliceFormat,
    /// Keys arrive in order from the merge, so the dictionary streams into
    /// its final bytes and its size is known exactly as it grows.
    dict: TermBlockWriter<Vec<u8>>,
    postings: Vec<u8>,
    first_key: Option<Vec<u8>>,
    last_key: Vec<u8>,
    n_terms: usize,
}

impl SliceBuilder {
    fn new(format: SliceFormat) -> Self {
        Self {
            format,
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
        self.format.push_run(&mut self.postings, run);
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
        self.postings.len() + self.dict.encoded_len() + self.format.table_len(self.n_terms)
    }

    fn finish(self) -> Option<(SliceRef, Vec<u8>)> {
        let first_key = self.first_key?;
        let dict = self.dict.finish().expect("Vec sink cannot fail");
        let bytes = encode_slice(self.format, &dict, &self.postings);
        let content_hash = ContentHash::of(&bytes);
        Some((
            SliceRef {
                first_key,
                last_key: self.last_key,
                content_hash,
                len: bytes.len() as u64,
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
    slices: Vec<SliceRef>,
    slice_bytes: Vec<(ContentHash, Vec<u8>)>,
    policy: BuildPolicy,
}

impl SegmentWriter {
    fn new(policy: &BuildPolicy) -> Self {
        Self {
            current: SliceBuilder::new(policy.slice_format),
            slices: Vec::new(),
            slice_bytes: Vec::new(),
            policy: *policy,
        }
    }

    /// Append one term, first cutting the slice when the run would push it
    /// past the target.
    fn add(&mut self, key: &[u8], run: &[u8]) {
        if self.current.n_terms > 0
            && self.current.encoded_bytes() + run.len() > self.policy.slice_target_bytes
        {
            self.cut();
        }
        self.current.add(key, run);
    }

    fn cut(&mut self) {
        if let Some((reference, bytes)) = mem::replace(
            &mut self.current,
            SliceBuilder::new(self.policy.slice_format),
        )
        .finish()
        {
            self.slice_bytes.push((reference.content_hash, bytes));
            self.slices.push(reference);
        }
    }

    fn finish(mut self) -> (Segment, Vec<(ContentHash, Vec<u8>)>) {
        self.cut();
        (
            Segment {
                slices: self.slices,
            },
            self.slice_bytes,
        )
    }
}

/// A finished build: the root and every slice, ready to be written.
pub(crate) struct Built {
    /// The root, not yet written.
    pub(crate) root: Root,
    /// Slice bytes keyed by content hash, in key order.
    pub(crate) slices: Vec<(ContentHash, Vec<u8>)>,
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
    let mut heap: BinaryHeap<Reverse<Head>> = BinaryHeap::new();
    let mut pending: Vec<Option<Posting>> = Vec::with_capacity(contributions.len());
    for (ordinal, c) in contributions.iter().enumerate() {
        let mut rd = ContributionReader::open(&c.path)?;
        match rd.next()? {
            Some(rec) => {
                heap.push(Reverse(Head {
                    key: rec.key,
                    ordinal: ordinal as u32,
                }));
                pending.push(Some(rec.posting));
            }
            None => pending.push(None),
        }
        readers.push(rd);
    }

    let mut writer = SegmentWriter::new(policy);
    let mut run: Vec<Posting> = Vec::new();

    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        let take = |ordinal: u32,
                    run: &mut Vec<Posting>,
                    pending: &mut Vec<Option<Posting>>,
                    readers: &mut Vec<ContributionReader>,
                    heap: &mut BinaryHeap<Reverse<Head>>|
         -> Result<(), TermIndexError> {
            let mut p = pending[ordinal as usize]
                .take()
                .expect("pending posting for heap head");
            p.superfile = ordinal_base + ordinal;
            run.push(p);
            if let Some(next) = readers[ordinal as usize].next()? {
                heap.push(Reverse(Head {
                    key: next.key,
                    ordinal,
                }));
                pending[ordinal as usize] = Some(next.posting);
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
        writer.add(&key, &encode_run(&run));
    }
    let (segment, slices) = writer.finish();

    Ok(BuiltSegment {
        superfiles: contributions.iter().map(|c| c.superfile_id).collect(),
        id_mins: contributions.iter().map(|c| c.id_min).collect(),
        segment,
        slices,
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
    /// Decoded terms not handed out yet.
    batch: VecDeque<(Vec<u8>, Vec<Posting>)>,
    /// The last key decoded from the open slice, where the next batch
    /// resumes.
    last_key: Option<Vec<u8>>,
}

impl SegmentCursor {
    fn new(files: &[PathBuf]) -> Self {
        Self {
            files: files.iter().cloned().collect(),
            bytes: Vec::new(),
            batch: VecDeque::new(),
            last_key: None,
        }
    }

    /// The next term and its postings, `None` once every slice is read.
    fn next(&mut self) -> Result<Option<(Vec<u8>, Vec<Posting>)>, TermIndexError> {
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
            self.bytes = fs::read(path)?;
            self.last_key = None;
        }
    }
}

/// Merge segments into one, renumbering every posting through
/// `ordinals`: a posting naming old ordinal `i` names `ordinals[i]` in the
/// result, and is dropped when that is `None` (a superfile that is gone, or
/// padding). `sources` holds each segment's slice files in key order.
///
/// Runs come out in ascending new ordinal and are cut into slices exactly
/// as [`build_segment`] cuts them, so a merge whose ordinals follow the
/// order a build would give its contributions writes the same bytes as that
/// build.
pub(crate) fn merge_segments(
    sources: &[Vec<PathBuf>],
    ordinals: &[Option<u32>],
    policy: &BuildPolicy,
) -> Result<(Segment, Vec<(ContentHash, Vec<u8>)>), TermIndexError> {
    let mut cursors: Vec<SegmentCursor> = sources.iter().map(|s| SegmentCursor::new(s)).collect();
    let mut pending: Vec<Option<Vec<Posting>>> = Vec::with_capacity(cursors.len());
    let mut heap: BinaryHeap<Reverse<Head>> = BinaryHeap::new();
    for (source, cursor) in cursors.iter_mut().enumerate() {
        match cursor.next()? {
            Some((key, run)) => {
                heap.push(Reverse(Head {
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
    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        let mut next = Some(head.ordinal);
        while let Some(source) = next {
            let postings = pending[source as usize]
                .take()
                .expect("pending run for heap head");
            run.extend(postings.into_iter().filter_map(|mut p| {
                p.superfile = (*ordinals.get(p.superfile as usize)?)?;
                Some(p)
            }));
            if let Some((key, postings)) = cursors[source as usize].next()? {
                heap.push(Reverse(Head {
                    key,
                    ordinal: source,
                }));
                pending[source as usize] = Some(postings);
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
        writer.add(&key, &encode_run(&run));
    }
    Ok(writer.finish())
}
