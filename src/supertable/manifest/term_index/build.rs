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
    collections::HashMap,
    collections::{BinaryHeap, VecDeque},
    fmt,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    mem,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
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
    superfile::{
        error::BuildError,
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
/// The share of a table's postings its tiers may hold, by default. The
/// per-term budget is the largest count at which the sum over terms of
/// `min(df, budget)` fits this share, found by a pass over the terms
/// before they are encoded. Normalizing by postings rather than documents
/// is what keeps short text in bounds: a titles table has a few postings
/// per document and a fat tail of mid-frequency terms, and a budget sized
/// by documents copied nearly every posting into a tier. At six percent a
/// long-text table's tiers come to about nine percent of its bytes.
pub(crate) const TIER_SHARE_OF_POSTINGS: f64 = 0.06;
/// Budgets the pre-pass chooses among: a half-octave ladder from the
/// floor to the ceiling.
const BUDGET_LADDER_STEPS_PER_OCTAVE: u32 = 2;

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
    /// Entries a term's tier keeps table-wide; `0` keeps no tiers. With
    /// `tier_share_of_postings` set this is the ceiling, and the build
    /// chooses the budget under it. See [`tier_entries_for`].
    pub(crate) tier_entries_per_term: u64,
    /// When set, the budget is the largest count under the ceiling at
    /// which the tiers hold at most this share of the postings; see
    /// [`TIER_SHARE_OF_POSTINGS`].
    pub(crate) tier_share_of_postings: Option<f64>,
    /// See [`head_depth_for`].
    pub(crate) head_depth_per_superfile: usize,
    /// Where the build encodes tiers in parallel; the process-wide pool
    /// when the caller has none to offer.
    pub(crate) pool: Option<Arc<ThreadPool>>,
    /// Write each finished slice and tier object to a scratch file as it is
    /// cut rather than keep its bytes: a table-wide rebuild produces more
    /// index than a host holds in memory.
    pub(crate) spill_to_disk: bool,
}

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
            tier_entries_per_term: TIER_ENTRIES_MAX,
            tier_share_of_postings: None,
            head_depth_per_superfile: HEAD_DEPTH_MAX,
            pool: None,
            spill_to_disk: false,
        }
    }
}

/// Chooses the tier budget from the terms' document frequencies: the
/// largest budget on a ladder at which the sum over terms of
/// `min(df, budget)` — the entries the tiers would hold — fits the share
/// of all postings. Fed one table-wide `df` per term.
struct BudgetScan {
    ladder: Vec<u64>,
    /// Entries the tiers would hold at each rung.
    held: Vec<u64>,
    postings: u64,
}

impl BudgetScan {
    fn new(floor: u64, ceiling: u64) -> Self {
        let mut ladder = Vec::new();
        let mut rung = floor.max(1) as f64;
        let step = 2f64.powf(1.0 / f64::from(BUDGET_LADDER_STEPS_PER_OCTAVE));
        while (rung as u64) < ceiling {
            ladder.push(rung as u64);
            rung *= step;
        }
        ladder.push(ceiling);
        ladder.dedup();
        Self {
            held: vec![0; ladder.len()],
            ladder,
            postings: 0,
        }
    }

    fn add(&mut self, df: u64) {
        self.postings += df;
        for (rung, held) in self.ladder.iter().zip(&mut self.held) {
            *held += df.min(*rung);
        }
    }

    /// The largest rung whose entries fit `share` of the postings; the
    /// floor when none does.
    fn pick(&self, share: f64) -> u64 {
        let cap = (self.postings as f64 * share) as u64;
        self.ladder
            .iter()
            .zip(&self.held)
            .rev()
            .find(|(_, held)| **held <= cap)
            .map(|(rung, _)| *rung)
            .unwrap_or(self.ladder[0])
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
    /// Entries already decoded and numbered (a build's heads).
    entries: Vec<TierEntry>,
    /// Tiers still to decode and renumber (a merge's sources), so the
    /// decoding runs in the parallel batch rather than on the merge thread.
    sources: Vec<TierSource>,
    /// Every head or tier that fed the term was its whole list.
    complete: bool,
}

/// One source run's tier in a merge: its chunks and the slice's tier
/// object they sit in.
struct TierSource {
    object: Arc<Vec<u8>>,
    chunks: Vec<TierChunkRef>,
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
fn encode_term(
    mut term: TermInput,
    budget: u64,
    ordinals: &[Option<u32>],
) -> Result<EncodedTerm, TermIndexError> {
    for source in &term.sources {
        tier_entries(&source.chunks, &source.object, ordinals, &mut term.entries)?;
    }
    if budget == 0 || term.entries.is_empty() {
        return Ok(EncodedTerm {
            key: term.key,
            postings: term.postings,
            tier: None,
        });
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
    Ok(EncodedTerm {
        key: term.key,
        postings: term.postings,
        tier: Some(EncodedTier {
            complete: term.complete && !truncated,
            n_entries: term.entries.len() as u64,
            chunks,
        }),
    })
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

    /// A writer for one part of a contribution — one column's terms —
    /// under `dir`, named `part`; the parts are joined in key order once
    /// every column is written.
    fn create_part(dir: &Path, part: usize) -> Result<Self, TermIndexError> {
        let path = dir.join(format!("part-{part}.terms"));
        let out = BufWriter::new(File::create(&path)?);
        Ok(Self {
            out,
            path,
            scratch: None,
            superfile_id: Uuid::nil(),
            id_min: 0,
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
}

impl HeadSink {
    pub(crate) fn new(columns: HashMap<String, FieldId>) -> Result<Self, TermIndexError> {
        let scratch = tempfile::Builder::new()
            .prefix("infino-term-index-")
            .tempdir()?;
        Ok(Self {
            inner: Arc::new(Mutex::new(HeadSinkState {
                scratch,
                columns,
                parts: Vec::new(),
                current: None,
            })),
        })
    }

    /// The finished contribution for `superfile_id`, once the builder that
    /// fed this sink has finished.
    pub(crate) fn into_contribution(
        self,
        superfile_id: Uuid,
        id_min: i128,
    ) -> Result<Contribution, TermIndexError> {
        let state = Arc::try_unwrap(self.inner)
            .map_err(|_| TermIndexError::Build("term head sink still in use".into()))?
            .into_inner()
            .map_err(|_| TermIndexError::Build("term head sink poisoned".into()))?;
        let HeadSinkState {
            scratch, mut parts, ..
        } = state;
        parts.sort_by(|a, b| a.0.cmp(&b.0));
        let path = scratch.path().join(format!("{superfile_id}.terms"));
        let mut out = BufWriter::new(File::create(&path)?);
        for (_, writer) in parts {
            let part = writer.finish()?;
            let mut file = File::open(&part.path)?;
            io::copy(&mut file, &mut out)?;
            fs::remove_file(&part.path)?;
        }
        out.flush()?;
        Ok(Contribution {
            superfile_id,
            id_min,
            path,
            _scratch: Some(scratch),
        })
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
        let mut key = prefix.clone();
        key.extend_from_slice(term.as_bytes());
        writer
            .push(
                &key,
                facts.df,
                facts.bound,
                Location::from_dict_value(facts.entry),
                facts.tail,
                &Head {
                    docs: facts.head.to_vec(),
                    complete: facts.complete,
                },
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
    let (mut ra, mut rb) = (ContributionReader::open(a)?, ContributionReader::open(b)?);
    let mut n = 0usize;
    loop {
        let (x, y) = (ra.next()?, rb.next()?);
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
                    && x.head.complete == y.head.complete
                    && x.head.docs.len() == y.head.docs.len()
                    && x.head
                        .docs
                        .iter()
                        .zip(&y.head.docs)
                        .all(|(p, q)| p.0 == q.0 && p.1.to_bits() == q.1.to_bits());
                if !same {
                    return Ok(Some(format!(
                        "record {n} {:?} / {:?}: df {} vs {}, bound {:?} vs {:?}, tail {:?} vs {:?}, location {:?} vs {:?}, complete {} vs {}, head {:?} vs {:?}",
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
                        x.head.complete,
                        y.head.complete,
                        &x.head.docs[..x.head.docs.len().min(6)],
                        &y.head.docs[..y.head.docs.len().min(6)],
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
    /// The next record's key and the posting's document frequency, the
    /// head left undecoded: what the budget pre-pass needs.
    fn next_df(&mut self) -> Result<Option<(Vec<u8>, u64)>, TermIndexError> {
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
        Ok(Some((key, posting.df)))
    }

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
    slice_bytes: Vec<(ContentHash, BuiltBytes)>,
    /// The tier objects of the slices cut so far, keyed by content hash.
    tier_bytes: Vec<(ContentHash, BuiltBytes)>,
    /// Where cut slices and tier objects are spilled, when they are.
    spill: Option<Arc<TempDir>>,
    target_bytes: usize,
    tier_entries_per_term: u64,
    /// Terms gathered for the next parallel encoding batch, in key order.
    pending: Vec<TermInput>,
    /// Tier entries across `pending`, decoded or still to decode.
    pending_entries: usize,
    pool: Option<Arc<ThreadPool>>,
    /// How a merge renumbers the ordinals of the tiers it decodes; empty for
    /// a build, whose entries arrive numbered.
    ordinals: Arc<[Option<u32>]>,
}

impl SegmentWriter {
    fn new(policy: &BuildPolicy) -> Result<Self, TermIndexError> {
        let spill = match policy.spill_to_disk {
            true => Some(Arc::new(
                tempfile::Builder::new()
                    .prefix("infino-term-index-")
                    .tempdir()?,
            )),
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
            pending: Vec::new(),
            pending_entries: 0,
            pool: policy.pool.clone(),
            ordinals: Arc::from(Vec::new()),
        })
    }

    /// A writer for a merge, renumbering decoded tiers through `ordinals`.
    fn for_merge(policy: &BuildPolicy, ordinals: &[Option<u32>]) -> Result<Self, TermIndexError> {
        Ok(Self {
            ordinals: Arc::from(ordinals.to_vec()),
            ..Self::new(policy)?
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
        sources: Vec<TierSource>,
        complete: bool,
    ) -> Result<(), TermIndexError> {
        self.pending_entries += entries.len();
        for source in &sources {
            self.pending_entries += source
                .chunks
                .iter()
                .map(|c| c.n_entries as usize)
                .sum::<usize>();
        }
        self.pending.push(TermInput {
            key: key.to_vec(),
            postings,
            entries,
            sources,
            complete,
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
        let batch = mem::take(&mut self.pending);
        self.pending_entries = 0;
        let budget = self.tier_entries_per_term;
        let ordinals = Arc::clone(&self.ordinals);
        let encode = || -> Result<Vec<EncodedTerm>, TermIndexError> {
            batch
                .into_par_iter()
                .map(|term| encode_term(term, budget, &ordinals))
                .collect()
        };
        let encoded = match &self.pool {
            Some(pool) => pool.install(encode),
            None => encode(),
        }?;
        for term in encoded {
            self.add_encoded(term)?;
        }
        Ok(())
    }

    /// Write one encoded term: its chunks go in the tier object of the slice
    /// the term lands in, so a cut moves them to the fresh slice's object.
    fn add_encoded(&mut self, term: EncodedTerm) -> Result<(), TermIndexError> {
        let Some(encoded) = term.tier else {
            return self.add(&term.key, &encode_run(&term.postings));
        };
        let mut tier = TierRef {
            complete: encoded.complete,
            n_entries: encoded.n_entries,
            chunks: self.refs_at(&encoded.chunks),
        };
        let mut run = encode_run_with_tier(Some(&tier), &term.postings);
        if self.cuts_before(run.len()) {
            self.cut()?;
            tier.chunks = self.refs_at(&encoded.chunks);
            run = encode_run_with_tier(Some(&tier), &term.postings);
        }
        let placed = self.place_chunks(&encoded.chunks);
        debug_assert_eq!(placed, tier.chunks);
        self.current.add(&term.key, &run);
        Ok(())
    }

    /// Cut the open slice, and its tier object when it has one. A spilled
    /// slice is named by its hash and its tier object beside it by the same
    /// hash, as [`tier_path`] places them for a merge.
    fn cut(&mut self) -> Result<(), TermIndexError> {
        let tier_bytes = (!self.current_tier.is_empty()).then(|| mem::take(&mut self.current_tier));
        let tier = tier_bytes.as_ref().map(|bytes| ObjectRef {
            content_hash: ContentHash::of(bytes),
            len: bytes.len() as u64,
        });
        if let Some((reference, bytes)) =
            mem::replace(&mut self.current, SliceBuilder::new()).finish(tier)
        {
            let stem = reference.content_hash.to_hex();
            if let (Some(object), Some(bytes)) = (tier, tier_bytes) {
                let kept = self.keep(&format!("{stem}.{TIER_FILE_EXTENSION}"), bytes)?;
                self.tier_bytes.push((object.content_hash, kept));
            }
            let kept = self.keep(&format!("{stem}.slice"), bytes)?;
            self.slice_bytes.push((reference.content_hash, kept));
            self.slices.push(reference);
        }
        Ok(())
    }

    /// The segment, its slices and their tier objects, each written beside
    /// the slice that names it.
    #[allow(clippy::type_complexity)]
    fn finish(
        mut self,
    ) -> Result<
        (
            Segment,
            Vec<(ContentHash, BuiltBytes)>,
            Vec<(ContentHash, BuiltBytes)>,
        ),
        TermIndexError,
    > {
        self.flush_pending()?;
        self.cut()?;
        Ok((
            Segment {
                slices: self.slices,
            },
            self.slice_bytes,
            self.tier_bytes,
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
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    /// The slices' tier objects, keyed by content hash.
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
}

/// Merge contributions into a base root: one segment, ordinals from zero.
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

/// Sum one key's frequency across `streams` in key order, feeding each
/// term's table-wide `df` to a budget scan: the k-way key merge the
/// builds do, over frequencies alone.
fn budget_over_streams<S>(
    mut streams: Vec<S>,
    mut next: impl FnMut(&mut S) -> Result<Option<(Vec<u8>, u64)>, TermIndexError>,
    policy: &BuildPolicy,
) -> Result<u64, TermIndexError> {
    let Some(share) = policy.tier_share_of_postings else {
        return Ok(policy.tier_entries_per_term);
    };
    if policy.tier_entries_per_term == 0 {
        return Ok(0);
    }
    let mut heap: BinaryHeap<Reverse<MergeHead>> = BinaryHeap::new();
    let mut pending: Vec<u64> = vec![0; streams.len()];
    for (i, stream) in streams.iter_mut().enumerate() {
        if let Some((key, df)) = next(stream)? {
            heap.push(Reverse(MergeHead {
                key,
                ordinal: i as u32,
            }));
            pending[i] = df;
        }
    }
    let mut scan = BudgetScan::new(
        TIER_ENTRIES_MIN.min(policy.tier_entries_per_term),
        policy.tier_entries_per_term,
    );
    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        let mut df = 0u64;
        let mut source = Some(head.ordinal);
        while let Some(i) = source {
            df += pending[i as usize];
            if let Some((next_key, next_df)) = next(&mut streams[i as usize])? {
                heap.push(Reverse(MergeHead {
                    key: next_key,
                    ordinal: i,
                }));
                pending[i as usize] = next_df;
            }
            source = match heap.peek() {
                Some(Reverse(peek)) if peek.key == key => heap.pop().map(|Reverse(h)| h.ordinal),
                _ => None,
            };
        }
        scan.add(df);
    }
    Ok(scan.pick(share))
}

/// The budget for a build over `contributions`.
fn contribution_budget(
    contributions: &[Contribution],
    policy: &BuildPolicy,
) -> Result<u64, TermIndexError> {
    if policy.tier_share_of_postings.is_none() {
        return Ok(policy.tier_entries_per_term);
    }
    let readers = contributions
        .iter()
        .map(|c| ContributionReader::open(&c.path))
        .collect::<Result<Vec<_>, _>>()?;
    budget_over_streams(readers, |r| r.next_df(), policy)
}

/// The budget for a merge over `sources`, counting only postings of
/// superfiles that survive `ordinals`.
fn source_budget(
    sources: &[Vec<PathBuf>],
    ordinals: &[Option<u32>],
    policy: &BuildPolicy,
) -> Result<u64, TermIndexError> {
    if policy.tier_share_of_postings.is_none() {
        return Ok(policy.tier_entries_per_term);
    }
    let cursors: Vec<SegmentCursor> = sources.iter().map(|s| SegmentCursor::new(s)).collect();
    budget_over_streams(
        cursors,
        |c| {
            Ok(c.next()?.map(|(key, run)| {
                let df = run
                    .postings
                    .iter()
                    .filter(|p| {
                        ordinals
                            .get(p.superfile as usize)
                            .is_some_and(Option::is_some)
                    })
                    .map(|p| p.df)
                    .sum();
                (key, df)
            }))
        },
        policy,
    )
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
    // The budget first, from a pass over the keys and frequencies alone;
    // the heads stay undecoded until the build proper.
    let policy = &BuildPolicy {
        tier_entries_per_term: contribution_budget(contributions, policy)?,
        tier_share_of_postings: None,
        ..policy.clone()
    };
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

    let mut writer = SegmentWriter::new(policy)?;
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
        writer.add_term(
            &key,
            mem::take(&mut run),
            mem::take(&mut entries),
            Vec::new(),
            complete,
        )?;
    }
    let (segment, slices, tiers) = writer.finish()?;

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
    tier: Option<Arc<Vec<u8>>>,
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
    fn tier_bytes(&mut self) -> Result<Arc<Vec<u8>>, TermIndexError> {
        if self.tier.is_none() {
            let path = self
                .tier_file
                .as_ref()
                .ok_or_else(|| TermIndexError::Build("tier read before any slice".into()))?;
            self.tier = Some(Arc::new(fs::read(path)?));
        }
        Ok(Arc::clone(self.tier.as_ref().expect("read above")))
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
    chunks: &[TierChunkRef],
    object: &[u8],
    ordinals: &[Option<u32>],
    out: &mut Vec<TierEntry>,
) -> Result<(), TermIndexError> {
    let mut block = Vec::new();
    for chunk in chunks {
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
    // The budget first, from a pass over the runs; a posting of a superfile
    // that is gone counts for nothing.
    let policy = &BuildPolicy {
        tier_entries_per_term: source_budget(sources, ordinals, policy)?,
        tier_share_of_postings: None,
        ..policy.clone()
    };
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

    let mut writer = SegmentWriter::for_merge(policy, ordinals)?;
    let mut run: Vec<Posting> = Vec::new();
    let mut sources: Vec<TierSource> = Vec::new();
    while let Some(Reverse(head)) = heap.pop() {
        let key = head.key;
        run.clear();
        sources.clear();
        let mut complete = true;
        let mut next = Some(head.ordinal);
        while let Some(source) = next {
            let source_run = pending[source as usize]
                .take()
                .expect("pending run for heap head");
            match &source_run.tier {
                Some(tier) => {
                    complete &= tier.complete;
                    sources.push(TierSource {
                        object: cursors[source as usize].tier_bytes()?,
                        chunks: tier.chunks.clone(),
                    });
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
        writer.add_term(
            &key,
            mem::take(&mut run),
            Vec::new(),
            mem::take(&mut sources),
            complete,
        )?;
    }
    let (segment, slices, tiers) = writer.finish()?;
    Ok(MergedSegment {
        segment,
        slices,
        tiers,
    })
}

/// What [`merge_segments`] produces: the segment and the bytes it names.
pub(crate) struct MergedSegment {
    pub(crate) segment: Segment,
    pub(crate) slices: Vec<(ContentHash, BuiltBytes)>,
    pub(crate) tiers: Vec<(ContentHash, BuiltBytes)>,
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

    /// The scan picks the largest rung at which the tiers' entries fit the
    /// share of postings: with a fat tail every rung costs the tail in
    /// full, and the rung is set by the common terms it caps.
    #[test]
    fn the_budget_scan_picks_the_largest_rung_under_the_share() {
        // 1,000 rare terms of df 10 (10,000 postings), 10 common terms of
        // df 100,000 (1,000,000 postings): 1,010,000 postings in all.
        let mut scan = BudgetScan::new(64, 1 << 20);
        for _ in 0..1_000 {
            scan.add(10);
        }
        for _ in 0..10 {
            scan.add(100_000);
        }
        // At 6%: 60,600 entries; the tail takes 10,000, leaving 5,060 per
        // common term — the ladder's rung at or under that is 4,096.
        assert_eq!(scan.pick(0.06), 4_096);
        // At 1%: 10,100 entries, barely the tail; only the floor fits.
        assert_eq!(scan.pick(0.01), 64);
        // At 100%: everything fits, up to the ceiling.
        assert_eq!(scan.pick(1.0), 1 << 20);
    }

    /// A fixed budget resolves to itself; a share resolves through the scan
    /// over the merged keys, summing a term's frequency across sources.
    #[test]
    fn a_share_policy_sums_each_term_across_sources() {
        let policy = BuildPolicy {
            tier_entries_per_term: 1 << 20,
            tier_share_of_postings: Some(0.5),
            ..BuildPolicy::default()
        };
        // Two sources each holding "a" with df 3,000 and one holding "b"
        // with df 10: "a" is one term of df 6,000. Half the postings is
        // 3,005 entries; the ladder's half-octave rung at or under
        // 3,005 - 10 is 2,896 (2,048 times the square root of two).
        let sources: Vec<Vec<(Vec<u8>, u64)>> = vec![
            vec![(b"a".to_vec(), 3_000), (b"b".to_vec(), 10)],
            vec![(b"a".to_vec(), 3_000)],
        ];
        let streams: Vec<std::vec::IntoIter<(Vec<u8>, u64)>> =
            sources.into_iter().map(Vec::into_iter).collect();
        let budget = budget_over_streams(streams, |s| Ok(s.next()), &policy).expect("budget");
        assert_eq!(budget, 2_896);
        let fixed = BuildPolicy {
            tier_entries_per_term: 77,
            tier_share_of_postings: None,
            ..BuildPolicy::default()
        };
        let budget = budget_over_streams(
            Vec::<std::vec::IntoIter<(Vec<u8>, u64)>>::new(),
            |s| Ok(s.next()),
            &fixed,
        )
        .expect("budget");
        assert_eq!(budget, 77);
    }
}
