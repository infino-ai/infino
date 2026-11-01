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
//! An index can also be rebuilt from its own layers
//! ([`merge_layers`]) rather than from the superfiles: a posting depends
//! only on its superfile, which never changes, so copying it out of an
//! existing layer gives exactly what walking the superfile again would.

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap, HashMap, VecDeque, btree_map::Entry},
    fmt,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

use bytes::Bytes;
use memmap2::Mmap;
use rayon::{ThreadPool, prelude::*};
use tempfile::TempDir;
use tracing::debug;
use uuid::Uuid;

use super::{
    TermIndexError,
    format::{
        BlockMark, Layer, Location, Posting, Root, Run, Slice, SliceRef, TierRef, encode_run,
        encode_slice, legacy_bound_as_key, push_doc_facts, push_posting, read_doc_facts,
        read_posting,
    },
    tier::{
        BLOCK_ENTRIES, ColumnScoring, EncodedChunk, HeadGroup, SplicedBlock, TierChunk,
        TierChunkRef, TierEntry, encode_block, encode_spliced_chunk, encode_tier,
    },
};
use crate::{
    config::scratch_root,
    superfile::{
        error::BuildError,
        format::checksum::crc32c,
        fts::builder::{TermHeadFacts, TermHeadSink},
    },
    supertable::{manifest::part::ContentHash, schema::FieldId},
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

/// A tier of at most this many entries rides in its run instead of a
/// chunk: a few bytes against a chunk's fixed header, directory and block
/// header, which on a titles table — where most terms have a posting or
/// two — came to more than the entries themselves.
pub(crate) const TIER_INLINE_MAX: usize = 8;
/// Documents a superfile keeps per term in its head: deep enough that a
/// superfile's tail — the key of the first document it leaves out — sits
/// well below its best, which is what lets a query rule the superfile out
/// without opening it. Measured on a billion-document table: a two-term
/// top-10 opened 26 superfiles of 1,080 at a depth of 128 and 10 at 512.
pub(crate) const HEAD_DEPTH: usize = 512;
/// Most documents a superfile keeps per term, whatever the caller asks.
pub(crate) const HEAD_DEPTH_MAX: usize = 4096;

/// Knobs the build takes from its caller.
#[derive(Debug, Clone)]
pub(crate) struct BuildPolicy {
    /// See [`SLICE_TARGET_BYTES`].
    pub(crate) slice_target_bytes: usize,
    /// Most entries a term's tier keeps across all superfiles: when the
    /// superfiles' heads together exceed it, every head is cut to the
    /// same shallower depth. `0` keeps no tiers.
    pub(crate) tier_entries_per_term: u64,
    /// Entries a superfile keeps per term in its head; see [`HEAD_DEPTH`].
    pub(crate) head_depth_per_superfile: usize,
    /// Where the build encodes tiers in parallel; the process-wide pool
    /// when the caller has none to offer.
    pub(crate) pool: Option<Arc<ThreadPool>>,
    /// Write each finished slice and tier object to a scratch file as it is
    /// cut rather than keep its bytes: a table-wide rebuild produces more
    /// index than a host holds in memory.
    pub(crate) spill_to_disk: bool,
    /// Where scratch goes — contributions, spilled slices, a merge's
    /// downloads — when set: the table's disk, not the system temporary
    /// directory, which on a host sized for a large table may be small or
    /// in memory.
    pub(crate) scratch_root: Option<PathBuf>,
    /// Key ranges a merge of layers, or a build from contributions, is cut
    /// into and run in parallel; `None` chooses by the input's size and the
    /// pool's width (see [`merge_layers`] and [`build_layer`]).
    pub(crate) merge_ranges: Option<usize>,
    /// For a merge: per superfile of the merged root, by ordinal, the
    /// average length and parameters its tier keys are computed at — the
    /// superfile's own, as the manifest records them. A build takes these
    /// from its contributions instead. A superfile or column absent here
    /// keys at [`ColumnScoring::default`].
    pub(crate) scoring_by_ordinal: Arc<Vec<BTreeMap<FieldId, ColumnScoring>>>,
    /// The same facts by superfile id, for a commit: its root's ordinals
    /// are assigned as the commit's layer is reserved, so the fold that
    /// follows the commit keys the layers it merges by ordinal from this
    /// map and the root it folds. Empty, a fold keys at
    /// [`ColumnScoring::default`] like any merge without the table.
    pub(crate) scoring_by_superfile: Arc<HashMap<Uuid, BTreeMap<FieldId, ColumnScoring>>>,
    /// Bytes a merge cursor reads from a tier object at a time; see
    /// [`TIER_READ_WINDOW_BYTES`].
    pub(crate) tier_read_window_bytes: u64,
    /// Whether a merge carries tier blocks over from its sources' chunks
    /// (see `splice_term`); off, every term takes the decoding path. Tests
    /// compare the two paths byte for byte.
    #[cfg(test)]
    pub(crate) splice_tiers: bool,
}

impl BuildPolicy {
    /// Whether a merge may splice tier blocks.
    fn splices_tiers(&self) -> bool {
        #[cfg(test)]
        {
            self.splice_tiers
        }
        #[cfg(not(test))]
        {
            true
        }
    }
}

/// How a layer writer keys an entry: by the superfile ordinal and column
/// it belongs to.
type ScoringOf = Arc<dyn Fn(u32, FieldId) -> ColumnScoring + Send + Sync>;

/// Bytes a build produced: resident, or in a scratch file the build spilled
/// them to, whose directory lives as long as any reference to it.
#[derive(Clone)]
pub(crate) enum BuiltBytes {
    Resident(Vec<u8>),
    Spilled {
        /// Held only to keep the scratch directory, and the file, alive.
        _dir: Arc<TempDir>,
        path: PathBuf,
    },
}

#[cfg(test)]
impl BuiltBytes {
    /// The bytes, read back when spilled.
    pub(crate) fn read(&self) -> io::Result<Vec<u8>> {
        match self {
            Self::Resident(bytes) => Ok(bytes.clone()),
            Self::Spilled { path, .. } => fs::read(path),
        }
    }
}

impl fmt::Debug for BuiltBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Resident(bytes) => write!(f, "Resident({} bytes)", bytes.len()),
            Self::Spilled { path, .. } => write!(f, "Spilled({})", path.display()),
        }
    }
}

/// Equal when the bytes are, wherever they sit.
#[cfg(test)]
impl PartialEq for BuiltBytes {
    fn eq(&self, other: &Self) -> bool {
        match (self.read(), other.read()) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
}

impl Default for BuildPolicy {
    fn default() -> Self {
        Self {
            slice_target_bytes: SLICE_TARGET_BYTES,
            tier_entries_per_term: u64::MAX,
            head_depth_per_superfile: HEAD_DEPTH_MAX,
            pool: None,
            spill_to_disk: false,
            scratch_root: None,
            merge_ranges: None,
            scoring_by_ordinal: Arc::new(Vec::new()),
            scoring_by_superfile: Arc::new(HashMap::new()),
            tier_read_window_bytes: TIER_READ_WINDOW_BYTES,
            #[cfg(test)]
            splice_tiers: true,
        }
    }
}

/// Tier entries a batch of terms gathers before the batch's tiers are
/// sorted, cut and encoded in parallel; a batch also closes at
/// [`ENCODE_BATCH_TERMS`]. Bounds the entries held decoded at once.
const ENCODE_BATCH_ENTRIES: usize = 1 << 21;
/// Most terms in one encoding batch.
const ENCODE_BATCH_TERMS: usize = 1 << 13;
/// Fewest terms one parallel job of a batch takes: a spliced term is
/// microseconds of work, less than a job's scheduling.
const ENCODE_JOB_MIN_TERMS: usize = 64;
/// Bytes a merge cursor reads from a tier object at a time, so the terms
/// of a window share one read rather than each seeking for its own chunks.
/// A cursor keeps its current window, and a term whose chunks are still
/// being encoded keeps the window they were read from, so a merge holds a
/// few windows per cursor on top of the slice's bytes.
const TIER_READ_WINDOW_BYTES: u64 = 4 * 1024 * 1024;

/// One term as the merge hands it over: its postings and the tier entries
/// gathered for it, in any order.
struct TermInput {
    key: Vec<u8>,
    postings: Vec<Posting>,
    /// Entries already decoded and numbered (a build's heads).
    entries: Vec<TierEntry>,
    /// Tiers still to decode and renumber (a merge's sources), so the
    /// decoding runs in the parallel batch rather than on the merge thread.
    sources: Vec<TierSource>,
}

/// A scratch directory for a build, removed when dropped: under `root`
/// when one is set, else under `storage.scratch_root`.
pub(crate) fn scratch_dir(root: Option<&Path>) -> io::Result<TempDir> {
    let builder = {
        let mut b = tempfile::Builder::new();
        b.prefix("infino-term-index-");
        b
    };
    match root {
        Some(root) => {
            fs::create_dir_all(root)?;
            builder.tempdir_in(root)
        }
        None => builder.tempdir_in(scratch_root()),
    }
}

/// One source run's tier in a merge: its chunks, and the bytes of the
/// tier object from `base` through the last of them — the span the chunks
/// lie in, read from the object rather than the object whole.
struct TierSource {
    /// A window of the tier object the chunks lie in, shared with the
    /// cursor that read it.
    bytes: Arc<Vec<u8>>,
    /// Offset of the window in the tier object.
    base: u64,
    chunks: Vec<TierChunkRef>,
}

/// A term's tier, cut to the budget and encoded, before it is placed: in
/// chunks, or inline when small.
struct EncodedTier {
    chunks: Vec<EncodedChunk>,
    inline: Vec<TierEntry>,
}

/// A term ready to be written: postings with their tails raised, and the
/// encoded tier when it keeps one.
struct EncodedTerm {
    key: Vec<u8>,
    postings: Vec<Posting>,
    tier: Option<EncodedTier>,
}

/// Group, cut and encode one term's tier: each superfile keeps its best
/// `depth` entries (keyed at its own scoring, `scoring_of` by ordinal), a
/// posting whose superfile loses entries to the cut has its tail raised to
/// the best key dropped, and when the superfiles' heads together exceed
/// `budget` the depth is lowered for all of them alike until they fit.
fn encode_term(
    mut term: TermInput,
    budget: u64,
    depth: usize,
    ordinals: &[Option<u32>],
    scoring_of: &dyn Fn(u32) -> ColumnScoring,
    splice: bool,
) -> Result<EncodedTerm, TermIndexError> {
    if splice
        && budget > 0
        && depth > 0
        && !term.sources.is_empty()
        && let Some(spliced) = splice_term(&mut term, budget, depth, ordinals)?
    {
        return Ok(spliced);
    }
    for source in &term.sources {
        tier_entries(
            &source.chunks,
            &source.bytes,
            source.base,
            ordinals,
            &mut term.entries,
        )?;
    }
    if budget == 0 || depth == 0 || term.entries.is_empty() {
        // No tier: the postings stand on their bounds. A contribution
        // writes a tail of zero when its head held every document, and
        // zero reads as "the tier holds them all" once a later merge gives
        // the term a tier, so the tail is raised past any key here.
        for p in &mut term.postings {
            p.tail = f32::INFINITY;
        }
        return Ok(EncodedTerm {
            key: term.key,
            postings: term.postings,
            tier: None,
        });
    }
    // One group per superfile, each keyed at that superfile's scoring and
    // put in descending key.
    term.entries.sort_unstable_by_key(|e| e.superfile);
    let mut keyed: Vec<(f32, TierEntry)> = Vec::with_capacity(term.entries.len());
    for g in term.entries.chunk_by(|a, b| a.superfile == b.superfile) {
        let scoring = scoring_of(g[0].superfile);
        let start = keyed.len();
        keyed.extend(g.iter().map(|e| (scoring.key(e.tf, e.len), *e)));
        keyed[start..]
            .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.doc_id.cmp(&b.1.doc_id)));
    }
    let n_groups = keyed
        .chunk_by(|a, b| a.1.superfile == b.1.superfile)
        .count();
    let mut depth = depth;
    let held = |depth: usize| -> u64 {
        keyed
            .chunk_by(|a, b| a.1.superfile == b.1.superfile)
            .map(|g| g.len().min(depth) as u64)
            .sum()
    };
    if held(depth) > budget {
        depth = ((budget / n_groups.max(1) as u64) as usize).max(1);
    }
    let mut groups: Vec<HeadGroup> = Vec::with_capacity(n_groups);
    for g in keyed.chunk_by(|a, b| a.1.superfile == b.1.superfile) {
        let ordinal = g[0].1.superfile;
        let keep = g.len().min(depth);
        // The group's tail: the key of the best document it leaves out, or
        // the posting's own tail — what its head left out before this build
        // — when the group is kept whole; the posting records the larger.
        let posting = term
            .postings
            .binary_search_by_key(&ordinal, |p| p.superfile)
            .ok();
        let mut tail = posting.map_or(0.0, |i| term.postings[i].tail);
        if keep < g.len() {
            tail = tail.max(g[keep].0);
            if let Some(i) = posting {
                term.postings[i].tail = tail;
            }
        }
        groups.push(HeadGroup {
            ordinal,
            entries: g[..keep].iter().map(|(_, e)| *e).collect(),
            keys: g[..keep].iter().map(|(k, _)| *k).collect(),
            tail,
        });
    }
    // A superfile with no entry in the tier — its layer kept no heads —
    // has nothing in the head its posting's tail would bound: the tail is
    // raised past any key, where zero would read as a head that holds
    // every document and drop the superfile's documents.
    let mut with_heads: Vec<u32> = groups.iter().map(|g| g.ordinal).collect();
    with_heads.sort_unstable();
    for p in &mut term.postings {
        if with_heads.binary_search(&p.superfile).is_err() {
            p.tail = f32::INFINITY;
        }
    }
    let n_entries: usize = groups.iter().map(|g| g.entries.len()).sum();
    let (chunks, inline) = match n_entries <= TIER_INLINE_MAX {
        true => (
            Vec::new(),
            groups
                .iter()
                .flat_map(|g| g.entries.iter().copied())
                .collect(),
        ),
        false => (encode_tier(&groups), Vec::new()),
    };
    Ok(EncodedTerm {
        key: term.key,
        postings: term.postings,
        tier: Some(EncodedTier { chunks, inline }),
    })
}

/// A merge's tier for one term carried over from its sources' chunks
/// without decoding an entry, when that gives the bytes [`encode_term`]'s
/// decode, re-key, cut and re-encode would give: every group arrives
/// whole and within `depth`, in the blocks [`encode_tier`] cuts (full
/// blocks but the last), the groups together fit `budget`, and the result
/// is too large to be kept inline. A block's body depends only on its
/// entries, and its directory facts — tail, count, CRC — on the group it
/// came from, so the merged chunks are the sources' blocks, regrouped by
/// rank and in ascending ordinal. `None` when some group needs the full
/// path, which then runs instead.
///
/// The group's order and tails were computed at the scoring its source
/// keyed the superfile at. The full path keys at the scoring the merge is
/// given for the superfile, which is the same: a column's parameters are
/// fixed when it is created and a superfile's average length is baked
/// into it, and both paths read them from the superfile. A superfile the
/// merge has no scoring for (one no longer in the manifest, folded before
/// a rebuild drops it) keeps its own keys here where the full path would
/// re-key it at the default; its postings are dropped at the next rebuild.
fn splice_term(
    term: &mut TermInput,
    budget: u64,
    depth: usize,
    ordinals: &[Option<u32>],
) -> Result<Option<EncodedTerm>, TermIndexError> {
    // Every chunk of every source, opened once: its blocks are carried by
    // reference into the chunks written below.
    let mut opened: Vec<Vec<TierChunk<'_>>> = Vec::with_capacity(term.sources.len());
    for source in &term.sources {
        let mut chunks = Vec::with_capacity(source.chunks.len());
        for chunk in &source.chunks {
            let start = chunk
                .offset
                .checked_sub(source.base)
                .ok_or_else(|| TermIndexError::Malformed("tier chunk before its span".into()))?;
            let end = start
                .checked_add(chunk.len)
                .filter(|end| *end <= source.bytes.len() as u64)
                .ok_or_else(|| TermIndexError::Malformed("tier chunk past its object".into()))?;
            chunks.push(TierChunk::open(
                &source.bytes[start as usize..end as usize],
                chunk.n_blocks,
            )?);
        }
        opened.push(chunks);
    }
    // One group per superfile, by ascending new ordinal: blocks carried
    // from a source, in rank order (a source's chunk `r` holds the rank-`r`
    // block of each of its groups), or entries that sat inline in a run
    // and are encoded fresh. A group split across sources, out of rank
    // order, or with both kinds is not spliced.
    enum Group<'a> {
        Carried(Vec<SplicedBlock<'a>>),
        Fresh(Vec<TierEntry>),
    }
    let mut groups: BTreeMap<u32, Group<'_>> = BTreeMap::new();
    for chunks in &opened {
        for (r, chunk) in chunks.iter().enumerate() {
            for b in 0..chunk.n_blocks() {
                let block = chunk.spliced_block(b);
                let Some(ordinal) = renumber(ordinals, block.ordinal) else {
                    continue;
                };
                match groups.entry(ordinal) {
                    Entry::Vacant(slot) if r == 0 => {
                        slot.insert(Group::Carried(vec![block]));
                    }
                    Entry::Occupied(mut slot) => match slot.get_mut() {
                        Group::Carried(blocks) if blocks.len() == r => blocks.push(block),
                        _ => return Ok(None),
                    },
                    Entry::Vacant(_) => return Ok(None),
                }
            }
        }
    }
    for e in &term.entries {
        match groups.entry(e.superfile) {
            Entry::Vacant(slot) => {
                slot.insert(Group::Fresh(vec![*e]));
            }
            Entry::Occupied(mut slot) => match slot.get_mut() {
                Group::Fresh(entries) => entries.push(*e),
                Group::Carried(_) => return Ok(None),
            },
        }
    }
    // Sizes and shapes: every group within depth and cut as a build cuts
    // it, all of them within budget, and too many entries to inline.
    let mut held = 0u64;
    for group in groups.values() {
        let n = match group {
            Group::Carried(blocks) => {
                let canonical = blocks
                    .iter()
                    .enumerate()
                    .all(|(i, b)| b.n_entries == BLOCK_ENTRIES || i + 1 == blocks.len());
                if !canonical {
                    return Ok(None);
                }
                blocks.iter().map(|b| b.n_entries).sum()
            }
            Group::Fresh(entries) => entries.len(),
        };
        if n > depth {
            return Ok(None);
        }
        held += n as u64;
    }
    if held > budget || held as usize <= TIER_INLINE_MAX {
        return Ok(None);
    }
    // A fresh group is one block; its tail is the posting's, as a group
    // kept whole gets.
    let tail_of = |ordinal: u32| -> f32 {
        term.postings
            .binary_search_by_key(&ordinal, |p| p.superfile)
            .map_or(0.0, |i| term.postings[i].tail)
    };
    let fresh: Vec<(u32, Vec<u8>, usize)> = groups
        .iter()
        .filter_map(|(ordinal, group)| match group {
            Group::Fresh(entries) => {
                let mut body = Vec::new();
                encode_block(entries, &mut body);
                Some((*ordinal, body, entries.len()))
            }
            Group::Carried(_) => None,
        })
        .collect();
    // Chunk `r` takes every group's rank-`r` block, in ascending ordinal.
    let deepest = groups
        .values()
        .map(|g| match g {
            Group::Carried(blocks) => blocks.len(),
            Group::Fresh(_) => 1,
        })
        .max()
        .unwrap_or(0);
    let mut chunks: Vec<EncodedChunk> = Vec::with_capacity(deepest);
    let mut blocks: Vec<SplicedBlock<'_>> = Vec::new();
    for r in 0..deepest {
        blocks.clear();
        let mut fresh = fresh.iter();
        for (ordinal, group) in &groups {
            match group {
                Group::Carried(carried) => {
                    if let Some(block) = carried.get(r) {
                        // The block names its superfile by the source's
                        // ordinal; the merged chunk names it by the new one.
                        blocks.push(SplicedBlock {
                            ordinal: *ordinal,
                            ..*block
                        });
                    }
                }
                Group::Fresh(_) => {
                    let (_, body, n) = fresh.next().expect("one fresh block per fresh group");
                    if r == 0 {
                        blocks.push(SplicedBlock {
                            ordinal: *ordinal,
                            tail: tail_of(*ordinal),
                            n_entries: *n,
                            crc: crc32c(body),
                            body,
                        });
                    }
                }
            }
        }
        chunks.push(encode_spliced_chunk(&blocks));
    }
    // A superfile with no entry in the tier has nothing in the head its
    // posting's tail would bound: the tail is raised past any key.
    for p in &mut term.postings {
        if !groups.contains_key(&p.superfile) {
            p.tail = f32::INFINITY;
        }
    }
    Ok(Some(EncodedTerm {
        key: mem::take(&mut term.key),
        postings: mem::take(&mut term.postings),
        tier: Some(EncodedTier {
            chunks,
            inline: Vec::new(),
        }),
    }))
}

/// One superfile's terms, spilled to a file in key order.
///
/// Record: `record_len varint | key_len varint | key | posting
/// | n_head varint | per head doc: row varint, tf varint, len u8`, where the
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
    scoring: BTreeMap<FieldId, ColumnScoring>,
    prev_key: Vec<u8>,
    n_terms: u64,
    record: Vec<u8>,
    /// The record's length prefix.
    frame: Vec<u8>,
    /// Bytes written so far: where the next record starts.
    offset: u64,
    /// Some records' keys with the offsets they start at (see
    /// [`CONTRIBUTION_MARK_BYTES`]).
    marks: Vec<(Vec<u8>, u64)>,
}

/// Bytes between the records a contribution marks, so a build cut into
/// key ranges starts each range's reader near its first key instead of
/// reading the file from the start. At 1 MB a 1.26 TB build keeps about
/// 1.3M marks, tens of MB.
const CONTRIBUTION_MARK_BYTES: u64 = if cfg!(test) { 256 } else { 1024 * 1024 };

impl ContributionWriter {
    /// Record the scoring `column`'s heads are keyed at.
    pub(crate) fn set_scoring(&mut self, column: FieldId, scoring: ColumnScoring) {
        self.scoring.insert(column, scoring);
    }

    /// Start a contribution for `superfile_id`, whose smallest doc id is
    /// `id_min`, spilling under `dir`.
    pub(crate) fn create(
        dir: &Path,
        superfile_id: Uuid,
        id_min: i128,
    ) -> Result<Self, TermIndexError> {
        fs::create_dir_all(dir)?;
        Self::create_at(
            dir.join(format!("{superfile_id}.terms")),
            superfile_id,
            id_min,
        )
    }

    /// A writer for one part of a contribution — one column's terms —
    /// under `dir`, named `part`; the parts are joined in key order once
    /// every column is written.
    fn create_part(dir: &Path, part: usize) -> Result<Self, TermIndexError> {
        Self::create_at(dir.join(format!("part-{part}.terms")), Uuid::nil(), 0)
    }

    fn create_at(path: PathBuf, superfile_id: Uuid, id_min: i128) -> Result<Self, TermIndexError> {
        let out = BufWriter::new(File::create(&path)?);
        Ok(Self {
            out,
            path,
            scratch: None,
            superfile_id,
            id_min,
            scoring: BTreeMap::new(),
            prev_key: Vec::new(),
            n_terms: 0,
            record: Vec::new(),
            frame: Vec::new(),
            offset: 0,
            marks: Vec::new(),
        })
    }

    /// Like [`Self::create`], spilling into a scratch directory of its own
    /// under `root` (see [`scratch_dir`]); the directory lives as long as
    /// the finished contribution does.
    pub(crate) fn create_in_scratch(
        superfile_id: Uuid,
        id_min: i128,
        root: Option<&Path>,
    ) -> Result<Self, TermIndexError> {
        let scratch = scratch_dir(root)?;
        let mut writer = Self::create(scratch.path(), superfile_id, id_min)?;
        writer.scratch = Some(scratch);
        Ok(writer)
    }

    /// Append one term with its head — `(row, term frequency, length
    /// bucket)` in descending key — and the tail the head leaves. Keys must
    /// arrive in strictly ascending order.
    pub(crate) fn push(
        &mut self,
        key: &[u8],
        df: u64,
        bound: f32,
        location: Location,
        tail: f32,
        head: &[(u32, u32, u8)],
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
        push_varint(&mut self.record, head.len() as u32);
        for &(row, tf, len) in head {
            push_doc_facts(&mut self.record, row, tf, len);
        }
        self.frame.clear();
        push_varint(&mut self.frame, self.record.len() as u32);
        if self
            .marks
            .last()
            .is_none_or(|(_, at)| self.offset - at >= CONTRIBUTION_MARK_BYTES)
        {
            self.marks.push((key.to_vec(), self.offset));
        }
        self.out.write_all(&self.frame)?;
        self.out.write_all(&self.record)?;
        self.offset += (self.frame.len() + self.record.len()) as u64;
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
            scoring: self.scoring,
            path: self.path,
            last_key: (self.n_terms > 0).then_some(self.prev_key),
            marks: self.marks,
            _scratch: self.scratch,
        })
    }
}

/// Gathers a superfile's term-index contribution from the builder that
/// writes the superfile, term by term as the postings are written, so no
/// second pass reads them back. The builder hands terms over column by
/// column in its own column order; each column's terms land in a part of
/// their own, and the parts are joined in ascending key order at the end.
/// Shared between the builder, which drives it, and the caller, which
/// takes the contribution once the build is done.
#[derive(Clone)]
pub(crate) struct HeadSink {
    inner: Arc<Mutex<HeadSinkState>>,
}

struct HeadSinkState {
    scratch: TempDir,
    /// Column name to the id its keys are written under; a column not
    /// here is not indexed and contributes nothing.
    columns: HashMap<String, FieldId>,
    /// `(key prefix, writer)` per column seen so far.
    parts: Vec<(Vec<u8>, ContributionWriter)>,
    /// The column the last term belonged to, so a run of terms finds its
    /// part without a lookup.
    current: Option<(String, usize)>,
    /// Per column, the scoring the builder's heads are keyed at.
    scoring: BTreeMap<FieldId, ColumnScoring>,
    /// The key of the term being written: the part's prefix and the term.
    key: Vec<u8>,
    /// The builder said it was done: without that the sink holds part of
    /// a contribution, not one.
    finished: bool,
}

impl HeadSink {
    /// A sink for `columns`, spilling under `root` (see [`scratch_dir`]).
    pub(crate) fn new(
        columns: HashMap<String, FieldId>,
        root: Option<&Path>,
    ) -> Result<Self, TermIndexError> {
        let scratch = scratch_dir(root)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(HeadSinkState {
                scratch,
                columns,
                parts: Vec::new(),
                current: None,
                scoring: BTreeMap::new(),
                key: Vec::new(),
                finished: false,
            })),
        })
    }

    /// The finished contribution for `superfile_id`, once the builder that
    /// fed this sink has finished; `None` when no builder drove the sink to
    /// its end — a superfile without a text index has nothing to
    /// contribute, and a caller must not take an empty contribution for one.
    pub(crate) fn into_contribution(
        self,
        superfile_id: Uuid,
        id_min: i128,
    ) -> Result<Option<Contribution>, TermIndexError> {
        let state = Arc::try_unwrap(self.inner)
            .map_err(|_| TermIndexError::Build("term head sink still in use".into()))?
            .into_inner()
            .map_err(|_| TermIndexError::Build("term head sink poisoned".into()))?;
        if !state.finished {
            return Ok(None);
        }
        let HeadSinkState {
            scratch,
            mut parts,
            scoring,
            ..
        } = state;
        parts.sort_by(|a, b| a.0.cmp(&b.0));
        let path = scratch.path().join(format!("{superfile_id}.terms"));
        let mut out = BufWriter::new(File::create(&path)?);
        let mut marks = Vec::new();
        let mut last_key = None;
        let mut base = 0u64;
        for (_, writer) in parts {
            let part = writer.finish()?;
            marks.extend(part.marks.into_iter().map(|(key, at)| (key, base + at)));
            last_key = part.last_key.or(last_key);
            let mut file = File::open(&part.path)?;
            base += io::copy(&mut file, &mut out)?;
            fs::remove_file(&part.path)?;
        }
        out.flush()?;
        // A range's reader finds its start by binary search over the marks.
        debug_assert!(
            marks.windows(2).all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1),
            "joined marks ascend by key and offset"
        );
        Ok(Some(Contribution {
            superfile_id,
            id_min,
            scoring,
            path,
            marks,
            last_key,
            _scratch: Some(scratch),
        }))
    }
}

impl TermHeadSink for HeadSink {
    fn term(
        &mut self,
        column: &str,
        term: &str,
        facts: &TermHeadFacts<'_>,
    ) -> Result<(), BuildError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| BuildError::Io(io::Error::other("term head sink poisoned")))?;
        let state = &mut *state;
        let part = match &state.current {
            Some((name, part)) if name == column => *part,
            _ => {
                let Some(id) = state.columns.get(column).copied() else {
                    return Ok(());
                };
                // The scoring is the column's, so a column change records
                // it once for the run of terms that follows.
                state.scoring.insert(
                    id,
                    ColumnScoring {
                        avgdl: facts.avgdl,
                        params: facts.params,
                        scored_docs: 0,
                    },
                );
                let prefix = id.term_key("");
                let part = match state.parts.iter().position(|(p, _)| *p == prefix) {
                    Some(part) => part,
                    None => {
                        let writer = ContributionWriter::create_part(
                            state.scratch.path(),
                            state.parts.len(),
                        )
                        .map_err(|e| BuildError::Io(io::Error::other(e.to_string())))?;
                        state.parts.push((prefix, writer));
                        state.parts.len() - 1
                    }
                };
                state.current = Some((column.to_owned(), part));
                part
            }
        };
        let (prefix, writer) = &mut state.parts[part];
        state.key.clear();
        state.key.extend_from_slice(prefix);
        state.key.extend_from_slice(term.as_bytes());
        writer
            .push(
                &state.key,
                facts.df,
                facts.bound,
                Location::from_dict_value(facts.entry),
                facts.tail,
                facts.head,
            )
            .map_err(|e| BuildError::Io(io::Error::other(e.to_string())))
    }

    fn finish(&mut self) -> Result<(), BuildError> {
        let mut state = self
            .inner
            .lock()
            .map_err(|_| BuildError::Io(io::Error::other("term head sink poisoned")))?;
        for (_, writer) in &mut state.parts {
            writer
                .out
                .flush()
                .map_err(|e| BuildError::Io(io::Error::other(e.to_string())))?;
        }
        state.finished = true;
        Ok(())
    }
}

/// The first record on which two contributions differ, described, or
/// `None` when they are the same record for record.
#[cfg(test)]
pub(crate) fn first_contribution_difference(
    a: &Path,
    b: &Path,
) -> Result<Option<String>, TermIndexError> {
    let (mut ra, mut rb) = (
        ContributionReader::open_at(a, 0)?,
        ContributionReader::open_at(b, 0)?,
    );
    let mut n = 0usize;
    loop {
        let (x, y) = (ra.next_in(None, None)?, rb.next_in(None, None)?);
        match (x, y) {
            (None, None) => return Ok(None),
            (Some(x), None) | (None, Some(x)) => {
                return Ok(Some(format!(
                    "record {n}: only one side has {:?}",
                    String::from_utf8_lossy(&x.key)
                )));
            }
            (Some(x), Some(y)) => {
                let same = x.key == y.key
                    && x.posting.df == y.posting.df
                    && x.posting.bound.to_bits() == y.posting.bound.to_bits()
                    && x.posting.tail.to_bits() == y.posting.tail.to_bits()
                    && x.posting.location == y.posting.location
                    && x.head == y.head;
                if !same {
                    return Ok(Some(format!(
                        "record {n} {:?} / {:?}: df {} vs {}, bound {:?} vs {:?}, tail {:?} vs {:?}, location {:?} vs {:?}, head {:?} vs {:?}",
                        String::from_utf8_lossy(&x.key),
                        String::from_utf8_lossy(&y.key),
                        x.posting.df,
                        y.posting.df,
                        x.posting.bound,
                        y.posting.bound,
                        x.posting.tail,
                        y.posting.tail,
                        x.posting.location,
                        y.posting.location,
                        &x.head[..x.head.len().min(6)],
                        &y.head[..y.head.len().min(6)],
                    )));
                }
            }
        }
        n += 1;
    }
}

/// A finished, spilled contribution.
#[derive(Debug)]
pub(crate) struct Contribution {
    /// The superfile these terms came from.
    pub(crate) superfile_id: Uuid,
    /// That superfile's smallest doc id.
    pub(crate) id_min: i128,
    /// Per column, the average length and parameters the superfile scores
    /// at: what its entries' keys and tails are computed at.
    pub(crate) scoring: BTreeMap<FieldId, ColumnScoring>,
    /// The spill file.
    pub(crate) path: PathBuf,
    /// Some records' keys with the offsets they start at, ascending, the
    /// first record's among them.
    pub(crate) marks: Vec<(Vec<u8>, u64)>,
    /// The last record's key; `None` when there are no records.
    pub(crate) last_key: Option<Vec<u8>>,
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
/// head as `(row, term frequency, length bucket)` in descending key.
struct Record {
    key: Vec<u8>,
    posting: Posting,
    head: Vec<(u32, u32, u8)>,
}

/// Largest byte length one encoded `u64` can occupy (10 × 7 bits).
const MAX_U64_VARINT_BYTES: usize = 10;

/// Read one LEB128 varint from a stream, with the bytes it took; `None`
/// at a clean end of file *before* the first byte. The bytes are decoded by
/// the crate's one varint reader, so a value the slice reader would refuse
/// is refused here too.
fn read_varint_stream(rd: &mut impl Read) -> io::Result<Option<(u64, usize)>> {
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
        .map(|v| Some((v, n)))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "varint overflows u64"))
}

impl ContributionReader {
    /// A reader whose first record starts at byte `offset`.
    fn open_at(path: &Path, offset: u64) -> Result<Self, TermIndexError> {
        let mut file = File::open(path)?;
        if offset > 0 {
            file.seek(SeekFrom::Start(offset))?;
        }
        Ok(Self {
            rd: BufReader::new(file),
            buf: Vec::new(),
        })
    }

    /// The next record whose key is in `start..end`, `None` past `end` or at
    /// end of file. A record before `start` is read up to its key and the
    /// rest skipped undecoded.
    fn next_in(
        &mut self,
        start: Option<&[u8]>,
        end: Option<&[u8]>,
    ) -> Result<Option<Record>, TermIndexError> {
        let malformed = |what: &str| TermIndexError::Malformed(format!("contribution {what}"));
        loop {
            let Some((record_len, _)) = read_varint_stream(&mut self.rd)? else {
                return Ok(None);
            };
            let (key_len, key_len_bytes) =
                read_varint_stream(&mut self.rd)?.ok_or_else(|| malformed("key length"))?;
            let rest = record_len
                .checked_sub(key_len_bytes as u64 + key_len)
                .ok_or_else(|| malformed("key"))?;
            let mut key = vec![0u8; key_len as usize];
            self.rd.read_exact(&mut key)?;
            if start.is_some_and(|s| key.as_slice() < s) {
                self.rd.seek_relative(rest as i64)?;
                continue;
            }
            if end.is_some_and(|e| key.as_slice() >= e) {
                return Ok(None);
            }
            self.buf.clear();
            self.buf.resize(rest as usize, 0);
            self.rd.read_exact(&mut self.buf)?;
            let mut at = 0usize;
            let posting = read_posting(&self.buf, &mut at)?;
            let n =
                read_varint(&self.buf, &mut at).ok_or_else(|| malformed("head count"))? as usize;
            let mut head = Vec::with_capacity(n);
            for _ in 0..n {
                head.push(
                    read_doc_facts(&self.buf, &mut at).ok_or_else(|| malformed("head entry"))?,
                );
            }
            if at != self.buf.len() {
                return Err(malformed("record: trailing bytes"));
            }
            return Ok(Some(Record { key, posting, head }));
        }
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
    fn finish(self, tier: Option<ContentHash>) -> Option<(SliceRef, Vec<u8>)> {
        let first_key = self.first_key?;
        let dict = self.dict.finish().expect("Vec sink cannot fail");
        let bytes = encode_slice(&dict, &self.postings);
        let content_hash = ContentHash::of(&bytes);
        Some((
            SliceRef {
                first_key,
                last_key: self.last_key,
                content_hash,
                tier,
                len: bytes.len() as u64,
            },
            bytes,
        ))
    }
}

/// Cuts a stream of `(key, encoded run)` in key order into slices. The
/// build and the merge both write through it, so equal input gives equal
/// bytes.
struct LayerWriter {
    current: SliceBuilder,
    /// The open slice's tier object so far: its terms' chunks back to back.
    current_tier: Vec<u8>,
    slices: Vec<SliceRef>,
    slice_bytes: Vec<(ContentHash, BuiltBytes)>,
    /// The tier objects of the slices cut so far, keyed by content hash.
    tier_bytes: Vec<(ContentHash, BuiltBytes)>,
    /// Where cut slices and tier objects are spilled, when they are.
    spill: Option<Arc<TempDir>>,
    target_bytes: usize,
    tier_entries_per_term: u64,
    /// Entries a superfile keeps per term.
    head_depth: usize,
    /// The scoring an entry is keyed at, by superfile ordinal and column.
    scoring: ScoringOf,
    /// Terms gathered for the next parallel encoding batch, in key order.
    pending: Vec<TermInput>,
    /// Tier entries across `pending`, decoded or still to decode.
    pending_entries: usize,
    pool: Option<Arc<ThreadPool>>,
    /// How a merge renumbers the ordinals of the tiers it decodes; empty for
    /// a build, whose entries arrive numbered.
    ordinals: Arc<[Option<u32>]>,
    /// Seconds spent in the parallel encoding of batches (the writer's
    /// thread waits on each), in cutting slices, and the terms and tier
    /// entries written: reported per range by the merge.
    stats: WriterStats,
    /// Whether terms may splice their sources' tier blocks.
    splice: bool,
}

/// What a layer writer spent its time on, for the merge's per-range
/// trace event.
#[derive(Debug, Default, Clone, Copy)]
struct WriterStats {
    encode_secs: f64,
    cut_secs: f64,
    terms: u64,
    entries: u64,
}

impl LayerWriter {
    fn new(policy: &BuildPolicy, scoring: ScoringOf) -> Result<Self, TermIndexError> {
        let spill = match policy.spill_to_disk {
            true => Some(Arc::new(scratch_dir(policy.scratch_root.as_deref())?)),
            false => None,
        };
        Ok(Self {
            current: SliceBuilder::new(),
            current_tier: Vec::new(),
            slices: Vec::new(),
            slice_bytes: Vec::new(),
            tier_bytes: Vec::new(),
            spill,
            target_bytes: policy.slice_target_bytes,
            tier_entries_per_term: policy.tier_entries_per_term,
            head_depth: policy.head_depth_per_superfile.min(HEAD_DEPTH_MAX),
            scoring,
            pending: Vec::new(),
            pending_entries: 0,
            pool: policy.pool.clone(),
            ordinals: Arc::from(Vec::new()),
            stats: WriterStats::default(),
            splice: policy.splices_tiers(),
        })
    }

    /// A writer for a merge, renumbering decoded tiers through `ordinals`
    /// and keying entries by the policy's per-ordinal scoring.
    fn for_merge(policy: &BuildPolicy, ordinals: &[Option<u32>]) -> Result<Self, TermIndexError> {
        let scoring = scoring_lookup(Arc::clone(&policy.scoring_by_ordinal), 0);
        Ok(Self {
            ordinals: Arc::from(ordinals.to_vec()),
            ..Self::new(policy, scoring)?
        })
    }

    /// Keep `bytes` resident, or spill them to `name` under the scratch
    /// directory when the build spills.
    fn keep(&self, name: &str, bytes: Vec<u8>) -> Result<BuiltBytes, TermIndexError> {
        match &self.spill {
            None => Ok(BuiltBytes::Resident(bytes)),
            Some(dir) => {
                let path = dir.path().join(name);
                fs::write(&path, &bytes)?;
                Ok(BuiltBytes::Spilled {
                    _dir: Arc::clone(dir),
                    path,
                })
            }
        }
    }

    /// Append one term, first cutting the slice when the run would push it
    /// past the target.
    fn add(&mut self, key: &[u8], run: &[u8]) -> Result<(), TermIndexError> {
        if self.cuts_before(run.len()) {
            self.cut()?;
        }
        self.current.add(key, run);
        Ok(())
    }

    /// Whether a run of `run_len` bytes starts a new slice.
    fn cuts_before(&self, run_len: usize) -> bool {
        self.current.n_terms > 0 && self.current.encoded_bytes() + run_len > self.target_bytes
    }

    /// The chunk list `chunks` get at the end of the open slice's tier
    /// object.
    fn refs_at(&self, chunks: &[EncodedChunk]) -> Vec<TierChunkRef> {
        let mut at = self.current_tier.len() as u64;
        chunks
            .iter()
            .map(|c| {
                let r = TierChunkRef {
                    offset: at,
                    len: c.bytes.len() as u64,
                    n_blocks: c.n_blocks,
                };
                at += r.len;
                r
            })
            .collect()
    }

    /// Place `chunks` at the end of the open slice's tier object, where
    /// [`Self::refs_at`] named them.
    fn place_chunks(&mut self, chunks: &[EncodedChunk]) {
        for c in chunks {
            self.current_tier.extend_from_slice(&c.bytes);
        }
    }

    /// Queue one term with its postings (ascending superfile) and the tier
    /// entries gathered for it, in any order. Terms are encoded in parallel
    /// a batch at a time and written in the order queued.
    fn add_term(
        &mut self,
        key: &[u8],
        postings: Vec<Posting>,
        entries: Vec<TierEntry>,
        sources: Vec<TierSource>,
    ) -> Result<(), TermIndexError> {
        self.stats.terms += 1;
        self.pending_entries += entries.len();
        for source in &sources {
            self.pending_entries += source
                .chunks
                .iter()
                .map(|c| c.n_blocks as usize * BLOCK_ENTRIES)
                .sum::<usize>();
        }
        self.pending.push(TermInput {
            key: key.to_vec(),
            postings,
            entries,
            sources,
        });
        if self.pending_entries >= ENCODE_BATCH_ENTRIES || self.pending.len() >= ENCODE_BATCH_TERMS
        {
            self.flush_pending()?;
        }
        Ok(())
    }

    /// Decode and encode the queued terms' tiers in parallel, then write
    /// each in order.
    fn flush_pending(&mut self) -> Result<(), TermIndexError> {
        let started = Instant::now();
        let batch = mem::take(&mut self.pending);
        self.stats.entries += self.pending_entries as u64;
        self.pending_entries = 0;
        let budget = self.tier_entries_per_term;
        let depth = self.head_depth;
        let splice = self.splice;
        let ordinals = Arc::clone(&self.ordinals);
        let scoring = Arc::clone(&self.scoring);
        let encode = || -> Result<Vec<EncodedTerm>, TermIndexError> {
            batch
                .into_par_iter()
                .with_min_len(ENCODE_JOB_MIN_TERMS)
                .map(|term| {
                    let column = FieldId::of_term_key(&term.key).unwrap_or(FieldId(u32::MAX));
                    encode_term(
                        term,
                        budget,
                        depth,
                        &ordinals,
                        &|ordinal| scoring(ordinal, column),
                        splice,
                    )
                })
                .collect()
        };
        let encoded = match &self.pool {
            Some(pool) => pool.install(encode),
            None => encode(),
        }?;
        self.stats.encode_secs += started.elapsed().as_secs_f64();
        for term in encoded {
            self.add_encoded(term)?;
        }
        Ok(())
    }

    /// Write one encoded term: its chunks go in the tier object of the slice
    /// the term lands in, so a cut moves them to the fresh slice's object.
    fn add_encoded(&mut self, term: EncodedTerm) -> Result<(), TermIndexError> {
        let Some(encoded) = term.tier else {
            return self.add(&term.key, &encode_run(None, &term.postings));
        };
        let mut tier = TierRef {
            chunks: self.refs_at(&encoded.chunks),
            inline: encoded.inline,
        };
        let mut run = encode_run(Some(&tier), &term.postings);
        if self.cuts_before(run.len()) {
            self.cut()?;
            tier.chunks = self.refs_at(&encoded.chunks);
            run = encode_run(Some(&tier), &term.postings);
        }
        self.place_chunks(&encoded.chunks);
        self.current.add(&term.key, &run);
        Ok(())
    }

    /// Cut the open slice, and its tier object when it has one. A spilled
    /// slice is named by its hash and its tier object beside it by the same
    /// hash, as [`tier_path`] places them for a merge.
    fn cut(&mut self) -> Result<(), TermIndexError> {
        let started = Instant::now();
        let tier_bytes = (!self.current_tier.is_empty()).then(|| mem::take(&mut self.current_tier));
        let tier = tier_bytes.as_ref().map(|bytes| ContentHash::of(bytes));
        if let Some((reference, bytes)) =
            mem::replace(&mut self.current, SliceBuilder::new()).finish(tier)
        {
            let stem = reference.content_hash.to_hex();
            if let (Some(hash), Some(bytes)) = (tier, tier_bytes) {
                let kept = self.keep(&format!("{stem}.{TIER_FILE_EXTENSION}"), bytes)?;
                self.tier_bytes.push((hash, kept));
            }
            let kept = self.keep(&format!("{stem}.slice"), bytes)?;
            self.slice_bytes.push((reference.content_hash, kept));
            self.slices.push(reference);
        }
        self.stats.cut_secs += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// The layer, its slices and their tier objects, each written beside
    /// the slice that names it, and what the writer spent its time on.
    #[allow(clippy::type_complexity)]
    fn finish(
        mut self,
    ) -> Result<
        (
            Layer,
            Vec<(ContentHash, BuiltBytes)>,
            Vec<(ContentHash, BuiltBytes)>,
            WriterStats,
        ),
        TermIndexError,
    > {
        self.flush_pending()?;
        self.cut()?;
        Ok((
            Layer {
                slices: self.slices,
            },
            self.slice_bytes,
            self.tier_bytes,
            self.stats,
        ))
    }
}

/// A finished build: the root and every slice, ready to be written.
pub(crate) struct Built {
    /// The root, not yet written.
    pub(crate) root: Root,
    /// Slice bytes keyed by content hash, in key order.
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    /// The slices' tier objects, keyed by content hash.
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
}

/// One merged layer: the superfiles it covers (in ordinal order, starting
/// at the base the caller passed), its slice list, and the slice bytes.
pub(crate) struct BuiltLayer {
    /// Superfiles this layer's postings name, in ordinal order.
    pub(crate) superfiles: Vec<Uuid>,
    /// Their smallest doc ids, parallel to `superfiles`.
    pub(crate) id_mins: Vec<i128>,
    /// The layer's slices, in key order.
    pub(crate) layer: Layer,
    /// Slice bytes keyed by content hash, in key order.
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    /// The slices' tier objects, keyed by content hash.
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
}

/// Merge contributions into a base root: one layer, ordinals from zero.
pub(crate) fn build(
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<Built, TermIndexError> {
    build_onto(Root::default(), contributions, policy)
}

/// `base` with `contributions` merged into one more layer, whose
/// ordinals continue from `base`'s superfile count. Onto an empty root this
/// is a full build: one layer, ordinals from zero.
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
    let built = build_layer(contributions, policy, base.superfiles.len() as u32)?;
    base.superfiles.extend(built.superfiles);
    base.id_mins.extend(built.id_mins);
    base.layers.push(built.layer);
    Ok(Built {
        root: base,
        slices: built.slices,
        tiers: built.tiers,
    })
}

/// Merge contributions into one layer of slices. Superfile ordinals
/// follow the order of `contributions`, offset by `ordinal_base` — the
/// number of superfiles the root already names when this layer is a
/// delta appended to it.
pub(crate) fn build_layer(
    contributions: &[Contribution],
    policy: &BuildPolicy,
    ordinal_base: u32,
) -> Result<BuiltLayer, TermIndexError> {
    let scoring = scoring_lookup(
        Arc::new(contributions.iter().map(|c| c.scoring.clone()).collect()),
        ordinal_base,
    );
    let ranges = build_ranges(contributions, policy)?;
    let built = run_ranges(&ranges, policy, |start, end| {
        build_range(contributions, start, end, policy, &scoring, ordinal_base)
    })?;
    Ok(BuiltLayer {
        superfiles: contributions.iter().map(|c| c.superfile_id).collect(),
        id_mins: contributions.iter().map(|c| c.id_min).collect(),
        layer: built.layer,
        slices: built.slices,
        tiers: built.tiers,
    })
}

/// Contributions of at least this many bytes in all are built in key
/// ranges. A commit's own contributions stay below it and build on one
/// writer, byte for byte as before; a rebuild over thousands of
/// superfiles goes over it.
const BUILD_RANGE_MIN_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// The key ranges a build of `contributions` is cut into (see
/// [`range_count`] and [`cut_ranges`]). A mark weighs the bytes up to its
/// contribution's next mark.
fn build_ranges(
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<Vec<KeyRange>, TermIndexError> {
    let mut lens = Vec::with_capacity(contributions.len());
    for c in contributions {
        lens.push(fs::metadata(&c.path)?.len());
    }
    let want = range_count(policy, BUILD_RANGE_MIN_BYTES, || Ok(lens.iter().sum()))?;
    if want <= 1 {
        return Ok(vec![(None, None)]);
    }
    let mut marks: Vec<(&[u8], u64)> = contributions
        .iter()
        .zip(&lens)
        .flat_map(|(c, &len)| {
            c.marks.iter().enumerate().map(move |(i, (key, at))| {
                let next = c.marks.get(i + 1).map_or(len, |m| m.1);
                (key.as_slice(), next.saturating_sub(*at))
            })
        })
        .collect();
    marks.sort_by(|a, b| a.0.cmp(b.0));
    Ok(cut_ranges(&marks, want))
}

/// Merge the keys in `start..end` of `contributions` into one run of
/// slices. A contribution whose keys miss the range is not opened; the
/// others start reading at their last mark at or before `start`.
fn build_range(
    contributions: &[Contribution],
    start: Option<&[u8]>,
    end: Option<&[u8]>,
    policy: &BuildPolicy,
    scoring: &ScoringOf,
    ordinal_base: u32,
) -> Result<MergedLayer, TermIndexError> {
    let mut readers = Vec::with_capacity(contributions.len());
    let mut heap: BinaryHeap<Reverse<MergeHead>> = BinaryHeap::new();
    let mut pending: Vec<Option<(Posting, Vec<(u32, u32, u8)>)>> =
        Vec::with_capacity(contributions.len());
    for (ordinal, c) in contributions.iter().enumerate() {
        let reaches = c
            .marks
            .first()
            .is_some_and(|(first, _)| end.is_none_or(|e| first.as_slice() < e))
            && c.last_key
                .as_ref()
                .is_some_and(|last| start.is_none_or(|s| last.as_slice() >= s));
        if !reaches {
            readers.push(None);
            pending.push(None);
            continue;
        }
        let offset = start.map_or(0, |s| {
            let at = c.marks.partition_point(|(key, _)| key.as_slice() <= s);
            at.checked_sub(1).map_or(0, |i| c.marks[i].1)
        });
        let mut rd = ContributionReader::open_at(&c.path, offset)?;
        match rd.next_in(start, end)? {
            Some(rec) => {
                heap.push(Reverse(MergeHead {
                    key: rec.key,
                    ordinal: ordinal as u32,
                }));
                pending.push(Some((rec.posting, rec.head)));
            }
            None => pending.push(None),
        }
        readers.push(Some(rd));
    }

    let mut writer = LayerWriter::new(policy, scoring.clone())?;
    let mut run: Vec<Posting> = Vec::new();
    let mut entries: Vec<TierEntry> = Vec::new();

    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        entries.clear();
        let mut take = |ordinal: u32,
                        run: &mut Vec<Posting>,
                        pending: &mut Vec<Option<(Posting, Vec<(u32, u32, u8)>)>>,
                        readers: &mut Vec<Option<ContributionReader>>,
                        heap: &mut BinaryHeap<Reverse<MergeHead>>|
         -> Result<(), TermIndexError> {
            let (mut p, h) = pending[ordinal as usize]
                .take()
                .expect("pending posting for heap head");
            p.superfile = ordinal_base + ordinal;
            entries.extend(h.iter().map(|&(doc_id, tf, len)| TierEntry {
                superfile: p.superfile,
                doc_id,
                tf,
                len,
            }));
            run.push(p);
            if let Some(next) = readers[ordinal as usize]
                .as_mut()
                .expect("a reader for each pending record")
                .next_in(None, end)?
            {
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
        writer.add_term(
            &key,
            mem::take(&mut run),
            mem::take(&mut entries),
            Vec::new(),
        )?;
    }
    let (layer, slices, tiers, _) = writer.finish()?;
    Ok(MergedLayer {
        layer,
        slices,
        tiers,
    })
}

/// Terms a merge cursor decodes from a slice at a time, so a cursor holds
/// one slice's bytes and a batch of decoded runs rather than the whole
/// slice decoded.
const MERGE_BATCH_TERMS: usize = 4096;

/// A slice on local disk, with the key range its layer records for it,
/// so a merge can be range-cut and its sources selected without opening
/// a slice.
#[derive(Debug, Clone)]
pub(crate) struct SliceFile {
    /// The slice's bytes, as downloaded or spilled.
    pub(crate) path: PathBuf,
    /// Smallest key the slice holds.
    pub(crate) first_key: Vec<u8>,
    /// Largest key the slice holds.
    pub(crate) last_key: Vec<u8>,
}

/// `path`'s bytes, mapped read-only; empty for an empty file, which cannot
/// be mapped.
fn map_file(path: &Path) -> io::Result<Bytes> {
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        return Ok(Bytes::new());
    }
    // SAFETY: the slice files are this merge's own scratch copies, written
    // before the merge starts and not changed or removed while it reads them.
    let map = unsafe { Mmap::map(&file)? };
    Ok(Bytes::from_owner(map))
}

/// Reads one layer's slices, already on local disk, term by term in key
/// order.
struct LayerCursor {
    /// Slice files not opened yet, in key order.
    files: VecDeque<PathBuf>,
    /// The open slice's bytes, mapped rather than read, so the many cursors
    /// a merge of many layers keeps open hold their slices in page cache,
    /// which the kernel can drop and read back, not in the heap; empty
    /// before the first file.
    bytes: Bytes,
    /// The open slice's tier object and its length, opened on the first
    /// tier the slice yields; `None` until then and after every slice
    /// change.
    tier: Option<(File, u64)>,
    /// Where the open slice's tier object sits (see [`tier_path`]).
    tier_file: Option<PathBuf>,
    /// The window of the tier object last read, and where it starts. A
    /// slice's chunks lie in key order, so a merge reads the object front
    /// to back and one window serves the terms that fall in it.
    window: Arc<Vec<u8>>,
    window_base: u64,
    /// Least bytes a window holds.
    window_bytes: u64,
    /// Decoded terms not handed out yet.
    batch: VecDeque<(Vec<u8>, Run)>,
    /// The last key decoded from the open slice, where the next batch
    /// resumes.
    last_key: Option<Vec<u8>>,
    /// The keys the cursor yields: from `start` on, when set, and short of
    /// `end`, when set.
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    /// Seconds spent reading slice files and decoding batches, and
    /// reading tier windows, for the merge's per-range trace event.
    decode_secs: f64,
    read_secs: f64,
}

impl LayerCursor {
    /// A cursor over `files` in key order, yielding the keys in
    /// `start..end` (either bound open when `None`), reading tier objects
    /// in windows of at least `window_bytes`.
    fn new(
        files: &[SliceFile],
        start: Option<&[u8]>,
        end: Option<&[u8]>,
        window_bytes: u64,
    ) -> Self {
        Self {
            files: files.iter().map(|f| f.path.clone()).collect(),
            bytes: Bytes::new(),
            tier: None,
            tier_file: None,
            window: Arc::new(Vec::new()),
            window_base: 0,
            window_bytes,
            batch: VecDeque::new(),
            last_key: None,
            start: start.map(<[u8]>::to_vec),
            end: end.map(<[u8]>::to_vec),
            decode_secs: 0.0,
            read_secs: 0.0,
        }
    }

    /// The next term and its run, `None` once every slice is read or the
    /// range's end is reached.
    fn next(&mut self) -> Result<Option<(Vec<u8>, Run)>, TermIndexError> {
        loop {
            if let Some(term) = self.batch.pop_front() {
                if self
                    .end
                    .as_deref()
                    .is_some_and(|end| term.0.as_slice() >= end)
                {
                    // Past the range: nothing later in this file or the
                    // files after it belongs to it.
                    self.batch.clear();
                    self.files.clear();
                    self.bytes = Bytes::new();
                    return Ok(None);
                }
                return Ok(Some(term));
            }
            let started = Instant::now();
            if !self.bytes.is_empty() {
                let slice = Slice::open(&self.bytes)?;
                let batch = match (&self.last_key, &self.start) {
                    (None, Some(start)) => slice.terms_from(Some(start), MERGE_BATCH_TERMS)?,
                    _ => slice.terms_after(self.last_key.as_deref(), MERGE_BATCH_TERMS)?,
                };
                self.decode_secs += started.elapsed().as_secs_f64();
                if let Some((key, _)) = batch.last() {
                    self.last_key = Some(key.clone());
                    self.batch = batch.into();
                    continue;
                }
            }
            let Some(path) = self.files.pop_front() else {
                return Ok(None);
            };
            self.bytes = map_file(&path)?;
            self.decode_secs += started.elapsed().as_secs_f64();
            self.tier = None;
            self.window = Arc::new(Vec::new());
            self.window_base = 0;
            self.tier_file = Some(path.with_extension(TIER_FILE_EXTENSION));
            self.last_key = None;
        }
    }

    /// The bytes of the open slice's tier object that `chunks` lie in, and
    /// where they start: served from the window last read when it holds
    /// them, else from a fresh window of at least `window_bytes` starting
    /// at them. A run with a tier always comes from a slice that has one,
    /// so a missing file is an error, not an empty tier.
    fn tier_span(
        &mut self,
        chunks: &[TierChunkRef],
    ) -> Result<(Arc<Vec<u8>>, u64), TermIndexError> {
        let base = chunks.iter().map(|c| c.offset).min().unwrap_or(0);
        let end = chunks
            .iter()
            .try_fold(0u64, |end, c| Some(end.max(c.offset.checked_add(c.len)?)))
            .ok_or_else(|| TermIndexError::Malformed("tier chunk range".into()))?;
        let window_end = self.window_base + self.window.len() as u64;
        if base >= self.window_base && end <= window_end {
            return Ok((Arc::clone(&self.window), self.window_base));
        }
        let started = Instant::now();
        if self.tier.is_none() {
            let path = self
                .tier_file
                .as_ref()
                .ok_or_else(|| TermIndexError::Build("tier read before any slice".into()))?;
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            self.tier = Some((file, len));
        }
        let (file, len) = self.tier.as_mut().expect("opened above");
        if end > *len {
            return Err(TermIndexError::Malformed(
                "tier chunk past its object".into(),
            ));
        }
        let want = (end - base).max(self.window_bytes).min(*len - base);
        let mut bytes = vec![0u8; want as usize];
        file.seek(SeekFrom::Start(base))?;
        file.read_exact(&mut bytes)?;
        self.window = Arc::new(bytes);
        self.window_base = base;
        self.read_secs += started.elapsed().as_secs_f64();
        Ok((Arc::clone(&self.window), base))
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

/// The new ordinal of old ordinal `old` under `ordinals`; `None` for a
/// superfile that is gone, or padding.
fn renumber(ordinals: &[Option<u32>], old: u32) -> Option<u32> {
    *ordinals.get(old as usize)?
}

/// The per-ordinal scoring of a build's contributions, ordinals counted
/// from `base`; a superfile with none scores at the defaults.
fn scoring_lookup(table: Arc<Vec<BTreeMap<FieldId, ColumnScoring>>>, base: u32) -> ScoringOf {
    Arc::new(move |ordinal: u32, column| {
        ordinal
            .checked_sub(base)
            .and_then(|i| table.get(i as usize))
            .and_then(|columns| columns.get(&column).copied())
            .unwrap_or_default()
    })
}

/// Every entry of `chunks` in `object`, renumbered through `ordinals`;
/// entries of superfiles that are gone are dropped.
fn tier_entries(
    chunks: &[TierChunkRef],
    object: &[u8],
    base: u64,
    ordinals: &[Option<u32>],
    out: &mut Vec<TierEntry>,
) -> Result<(), TermIndexError> {
    let mut block = Vec::new();
    for chunk in chunks {
        let start = chunk
            .offset
            .checked_sub(base)
            .ok_or_else(|| TermIndexError::Malformed("tier chunk before its span".into()))?;
        let end = start
            .checked_add(chunk.len)
            .filter(|end| *end <= object.len() as u64)
            .ok_or_else(|| TermIndexError::Malformed("tier chunk past its object".into()))?;
        let bytes = &object[start as usize..end as usize];
        block.clear();
        TierChunk::open(bytes, chunk.n_blocks)?.entries(&mut block)?;
        out.extend(block.iter().filter_map(|e| {
            Some(TierEntry {
                superfile: renumber(ordinals, e.superfile)?,
                ..*e
            })
        }));
    }
    Ok(())
}

/// Merge layers into one, in key ranges on the pool when the
/// sources are large (see [`key_ranges`]), renumbering every posting through
/// `ordinals`: a posting naming old ordinal `i` names `ordinals[i]` in the
/// result, and is dropped when that is `None` (a superfile that is gone, or
/// padding). `sources` holds each layer's slices in key order, each with
/// its tier object beside it (see [`tier_path`]). Tiers merge the same way:
/// their entries are pooled, renumbered and re-cut at the policy's budget.
///
/// Runs come out in ascending new ordinal and are cut into slices as
/// [`build_layer`] cuts them, with a cut at every range bound as well.
/// A merge in one range whose ordinals follow the order a build would give
/// its contributions writes the same bytes as that build; a range-cut one
/// writes the same terms, each with the same postings and tier entries,
/// in slices cut at the range bounds too.
pub(crate) fn merge_layers(
    sources: &[Vec<SliceFile>],
    ordinals: &[Option<u32>],
    policy: &BuildPolicy,
) -> Result<MergedLayer, TermIndexError> {
    let ranges = key_ranges(sources, policy)?;
    // Each range merges the slices that reach into it, on its own writer.
    run_ranges(&ranges, policy, |start, end| {
        let in_range: Vec<Vec<SliceFile>> = sources
            .iter()
            .map(|files| {
                files
                    .iter()
                    .filter(|slice| slice_reaches(slice, start, end))
                    .cloned()
                    .collect()
            })
            .collect();
        merge_range(&in_range, start, end, ordinals, policy)
    })
}

/// A key range: from its start, when set, short of its end, when set.
type KeyRange = (Option<Vec<u8>>, Option<Vec<u8>>);

/// Run `one` on each of `ranges` on the policy's pool, or the one range in
/// place, and join their outputs: they follow each other in key order,
/// which is what one writer over everything would have produced, slice cuts
/// aside.
fn run_ranges(
    ranges: &[KeyRange],
    policy: &BuildPolicy,
    one: impl Fn(Option<&[u8]>, Option<&[u8]>) -> Result<MergedLayer, TermIndexError> + Sync,
) -> Result<MergedLayer, TermIndexError> {
    if let [(start, end)] = ranges {
        return one(start.as_deref(), end.as_deref());
    }
    let run_all = || -> Result<Vec<MergedLayer>, TermIndexError> {
        ranges
            .par_iter()
            .map(|(start, end)| one(start.as_deref(), end.as_deref()))
            .collect()
    };
    let parts = match policy.pool.as_deref() {
        Some(pool) => pool.install(run_all)?,
        None => run_all()?,
    };
    let mut layer = Layer::default();
    let mut slices = Vec::new();
    let mut tiers = Vec::new();
    for part in parts {
        layer.slices.extend(part.layer.slices);
        slices.extend(part.slices);
        tiers.extend(part.tiers);
    }
    Ok(MergedLayer {
        layer,
        slices,
        tiers,
    })
}

/// How many key ranges to cut work of `bytes` into: the policy's count when
/// set, one when the work is under `min_bytes`, else the pool's width, at
/// most [`MERGE_RANGES_MAX`].
fn range_count(
    policy: &BuildPolicy,
    min_bytes: u64,
    bytes: impl FnOnce() -> Result<u64, TermIndexError>,
) -> Result<usize, TermIndexError> {
    Ok(match policy.merge_ranges {
        Some(n) => n.max(1),
        None if bytes()? < min_bytes => 1,
        None => policy
            .pool
            .as_deref()
            .map_or_else(rayon::current_num_threads, ThreadPool::current_num_threads)
            .clamp(1, MERGE_RANGES_MAX),
    })
}

/// Cut `marks` — `(key, weight)` in ascending key — into up to `want`
/// ranges of about equal weight, cutting where the running weight crosses
/// each range's share. The smallest key is never a cut (every input may
/// start there), and no key is cut twice. One mark holding nearly all the
/// weight leaves no cut, and the work runs in one range.
fn cut_ranges<K: AsRef<[u8]>>(marks: &[(K, u64)], want: usize) -> Vec<KeyRange> {
    let total: u64 = marks.iter().map(|m| m.1).sum();
    if want <= 1 || marks.len() < 2 || total == 0 {
        return vec![(None, None)];
    }
    let n = want.min(marks.len());
    let share = total.div_ceil(n as u64);
    let mut cuts: Vec<&[u8]> = Vec::with_capacity(n);
    let mut acc = 0u64;
    let mut next_at = share;
    let first = marks[0].0.as_ref();
    for (key, weight) in marks {
        let key = key.as_ref();
        if key > first && acc >= next_at && cuts.last().is_none_or(|c| *c < key) {
            cuts.push(key);
            next_at = (acc / share + 1) * share;
            if cuts.len() + 1 >= n {
                break;
            }
        }
        acc += weight;
    }
    let mut ranges = Vec::with_capacity(cuts.len() + 1);
    let mut start: Option<Vec<u8>> = None;
    for cut in cuts {
        ranges.push((start.take(), Some(cut.to_vec())));
        start = Some(cut.to_vec());
    }
    ranges.push((start, None));
    ranges
}

/// Sources of at least this many bytes are merged in key ranges.
const MERGE_RANGE_MIN_BYTES: u64 = 64 * 1024 * 1024;
/// Ranges a merge is cut into at most: each holds a cursor per layer
/// with a slice's bytes and a few tier windows resident, so the width is
/// bounded by memory as well as by threads.
const MERGE_RANGES_MAX: usize = 32;

/// Whether `slice` holds a key in `start..end`.
fn slice_reaches(slice: &SliceFile, start: Option<&[u8]>, end: Option<&[u8]>) -> bool {
    start.is_none_or(|s| slice.last_key.as_slice() >= s)
        && end.is_none_or(|e| slice.first_key.as_slice() < e)
}

/// The key ranges a merge of `sources` is cut into: one when the sources
/// are small or the policy says so, else up to the pool's width (at most
/// [`MERGE_RANGES_MAX`]), cut so that each range holds about the same
/// bytes of runs and tier chunks. The weights come from the slices'
/// dictionary block marks (see [`Slice::block_marks`]): a block's bytes
/// are what lies between its mark and the next, and a cut falls on a
/// block's first key. Cutting by slice count instead left the common
/// terms' ranges — a few thousand terms carrying most of the tier bytes —
/// in a handful of ranges that took the whole merge's wall time.
pub(crate) fn key_ranges(
    sources: &[Vec<SliceFile>],
    policy: &BuildPolicy,
) -> Result<Vec<KeyRange>, TermIndexError> {
    let files: Vec<&SliceFile> = sources.iter().flatten().collect();
    let want = range_count(policy, MERGE_RANGE_MIN_BYTES, || {
        let mut bytes = 0u64;
        for file in &files {
            bytes += fs::metadata(&file.path)?.len();
        }
        Ok(bytes)
    })?;
    if want <= 1 {
        return Ok(vec![(None, None)]);
    }
    // Every dictionary block of every slice, weighed by the bytes of its
    // runs and tier chunks, read on the pool a slice at a time: one pass
    // over the slices before the merge reads them again. A block whose
    // first run has no chunks is weighed by its runs alone and its chunk
    // bytes fall to the next block that has them; a few blocks' worth of
    // tier bytes can shift a cut, which is well within a range.
    let weigh = |file: &SliceFile| -> Result<Vec<(Vec<u8>, u64)>, TermIndexError> {
        let bytes = fs::read(&file.path)?;
        let slice = Slice::open(&bytes)?;
        let marks: Vec<BlockMark> = slice.block_marks()?;
        let tier_len = fs::metadata(file.path.with_extension(TIER_FILE_EXTENSION))
            .map(|m| m.len())
            .unwrap_or(0);
        let mut out = Vec::with_capacity(marks.len());
        let mut tier_at = 0u64;
        for (i, mark) in marks.iter().enumerate() {
            let next = marks.get(i + 1);
            let run_bytes = next
                .map_or(slice.postings_len(), |n| n.postings_at)
                .saturating_sub(mark.postings_at);
            tier_at = mark.tier_at.unwrap_or(tier_at);
            let tier_end = next.and_then(|n| n.tier_at).unwrap_or(if next.is_some() {
                tier_at
            } else {
                tier_len
            });
            let tier_bytes = tier_end.saturating_sub(tier_at);
            out.push((mark.key.clone(), run_bytes + tier_bytes));
        }
        Ok(out)
    };
    let weigh_all = || -> Result<Vec<Vec<(Vec<u8>, u64)>>, TermIndexError> {
        files.par_iter().map(|file| weigh(file)).collect()
    };
    let mut marks: Vec<(Vec<u8>, u64)> = match policy.pool.as_deref() {
        Some(pool) => pool.install(weigh_all)?,
        None => weigh_all()?,
    }
    .into_iter()
    .flatten()
    .collect();
    marks.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(cut_ranges(&marks, want))
}

/// Merge the keys in `start..end` of `sources` into one run of slices.
fn merge_range(
    sources: &[Vec<SliceFile>],
    start: Option<&[u8]>,
    end: Option<&[u8]>,
    ordinals: &[Option<u32>],
    policy: &BuildPolicy,
) -> Result<MergedLayer, TermIndexError> {
    let mut cursors: Vec<LayerCursor> = sources
        .iter()
        .map(|s| LayerCursor::new(s, start, end, policy.tier_read_window_bytes))
        .collect();
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

    let mut writer = LayerWriter::for_merge(policy, ordinals)?;
    let mut run: Vec<Posting> = Vec::new();
    let mut sources: Vec<TierSource> = Vec::new();
    let mut entries: Vec<TierEntry> = Vec::new();
    let started = Instant::now();
    let mut tier_bytes = 0u64;
    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        sources.clear();
        entries.clear();
        let mut next = Some(head.ordinal);
        while let Some(source) = next {
            let source_run = pending[source as usize]
                .take()
                .expect("pending run for heap head");
            // A run without a tier contributes no entries; its postings
            // stand on their bounds in the merged term (see `encode_term`).
            if let Some(tier) = source_run.tier {
                if tier.chunks.is_empty() {
                    entries.extend(tier.inline.iter().filter_map(|e| {
                        Some(TierEntry {
                            superfile: renumber(ordinals, e.superfile)?,
                            ..*e
                        })
                    }));
                } else {
                    let (bytes, base) = cursors[source as usize].tier_span(&tier.chunks)?;
                    tier_bytes += tier.chunks.iter().map(|c| c.len).sum::<u64>();
                    sources.push(TierSource {
                        bytes,
                        base,
                        chunks: tier.chunks,
                    });
                }
            }
            // A layer of the layout before tiers bounds a posting by a
            // score at the superfile's own idf; it is written on as the
            // key the current layout stores.
            let legacy = source_run.legacy_bound;
            let column = FieldId::of_term_key(&key).unwrap_or(FieldId(u32::MAX));
            run.extend(source_run.postings.into_iter().filter_map(|mut p| {
                p.superfile = renumber(ordinals, p.superfile)?;
                if legacy {
                    let n = (writer.scoring)(p.superfile, column).scored_docs;
                    p.bound = legacy_bound_as_key(p.bound, p.df, n);
                }
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
        writer.add_term(
            &key,
            mem::take(&mut run),
            mem::take(&mut entries),
            mem::take(&mut sources),
        )?;
    }
    let (layer, slices, tiers, stats) = writer.finish()?;
    debug!(
        target: "term_index::fold",
        start = %String::from_utf8_lossy(start.unwrap_or(b"")),
        end = %String::from_utf8_lossy(end.unwrap_or(b"")),
        secs = started.elapsed().as_secs_f64(),
        decode_secs = cursors.iter().map(|c| c.decode_secs).sum::<f64>(),
        read_secs = cursors.iter().map(|c| c.read_secs).sum::<f64>(),
        tier_mib = tier_bytes as f64 / (1024.0 * 1024.0),
        encode_secs = stats.encode_secs,
        cut_secs = stats.cut_secs,
        terms = stats.terms,
        entries = stats.entries,
        out_slices = slices.len(),
        "fold range"
    );
    Ok(MergedLayer {
        layer,
        slices,
        tiers,
    })
}

/// What [`merge_layers`] produces: the layer and the bytes it names.
pub(crate) struct MergedLayer {
    pub(crate) layer: Layer,
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
}

#[cfg(test)]
mod contribution_mark_tests {
    use super::*;
    use crate::superfile::fts::bm25::Bm25Params;

    /// A contribution gathered through a [`HeadSink`] from two columns, ids
    /// 3 and 12, with 3's terms handed over first: `"12\x1f…"` sorts before
    /// `"3\x1f…"`, so the parts are reordered when they are joined.
    fn two_column_contribution() -> Contribution {
        let columns = HashMap::from([
            ("three".to_owned(), FieldId(3)),
            ("twelve".to_owned(), FieldId(12)),
        ]);
        let mut sink = HeadSink::new(columns, None).expect("sink");
        let head: Vec<(u32, u32, u8)> = (0..8).map(|row| (row, 1 + row % 3, 4)).collect();
        for column in ["three", "twelve"] {
            for i in 0..300 {
                sink.term(
                    column,
                    &format!("term{i:04}"),
                    &TermHeadFacts {
                        df: head.len() as u64,
                        bound: 0.5,
                        entry: DictEntry::Pfor {
                            metadata_offset: i * 64,
                            postings_length: 64,
                            short: true,
                        },
                        head: &head,
                        tail: 0.1,
                        avgdl: 4.0,
                        params: Bm25Params::default(),
                    },
                )
                .expect("term");
            }
        }
        sink.finish().expect("finish");
        sink.into_contribution(Uuid::from_u128(7), 0)
            .expect("join")
            .expect("the sink was finished")
    }

    /// Every mark of a contribution joined from parts names the record at
    /// its offset: the parts are joined in key order and each part's marks
    /// shifted by the bytes before it.
    #[test]
    fn a_joined_contributions_marks_name_the_records_at_their_offsets() {
        let c = two_column_contribution();
        assert!(c.marks.len() >= 8, "{} marks", c.marks.len());
        assert!(
            c.marks[0].0.starts_with(b"12\x1f"),
            "column 12's terms come first in key order"
        );
        assert!(
            c.marks.iter().any(|(key, _)| key.starts_with(b"3\x1f")),
            "the second part is marked too"
        );
        assert!(
            c.marks
                .windows(2)
                .all(|w| w[0].0 < w[1].0 && w[0].1 < w[1].1),
            "marks ascend by key and offset"
        );
        for (key, at) in &c.marks {
            let rec = ContributionReader::open_at(&c.path, *at)
                .expect("open")
                .next_in(None, None)
                .expect("read")
                .expect("a record at the mark");
            assert_eq!(&rec.key, key, "the record at offset {at}");
        }
        assert_eq!(
            c.last_key.as_deref(),
            Some(FieldId(3).term_key("term0299").as_slice()),
            "the last key is the last part's"
        );
    }

    /// A reader asked for `start..end` skips the records before `start`
    /// and stops at `end`, from any mark at or before `start`.
    #[test]
    fn a_range_reader_yields_exactly_the_keys_in_its_range() {
        let c = two_column_contribution();
        let all: Vec<Vec<u8>> = {
            let mut rd = ContributionReader::open_at(&c.path, 0).expect("open");
            std::iter::from_fn(|| rd.next_in(None, None).expect("read").map(|r| r.key)).collect()
        };
        let (start, end) = (all[150].clone(), all[450].clone());
        for (_, at) in c
            .marks
            .iter()
            .filter(|(key, _)| key.as_slice() <= start.as_slice())
        {
            let mut rd = ContributionReader::open_at(&c.path, *at).expect("open");
            let got: Vec<Vec<u8>> = std::iter::from_fn(|| {
                rd.next_in(Some(&start), Some(&end))
                    .expect("read")
                    .map(|r| r.key)
            })
            .collect();
            assert_eq!(got, all[150..450], "from the mark at {at}");
        }
    }
}
