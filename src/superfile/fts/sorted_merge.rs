// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Merge the FTS indexes of already-built superfiles term by term.
//!
//! Every input's dictionary is in term order, so a k-way merge over the
//! inputs' dictionaries yields the output terms in final order, with no
//! corpus-sized accumulator; peak memory is one term's postings across all
//! inputs. When every remap rises, a term's postings joined in input order
//! are already sorted by output doc id. When one does not (the merge chose
//! its own doc order, or an input stores its docs under one), that one
//! term's postings are sorted before they are emitted.
//!
//! The walk over the dictionaries and the writes stay in term order on one
//! thread. Between them, each batch of terms is read, sorted and encoded in
//! parallel: a term's work depends on nothing outside the term.

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    io::Error,
    iter::once,
    ops::Range,
    str::from_utf8,
    sync::{Arc, Mutex, PoisonError},
    vec,
};

use bytes::Bytes;
use rayon::{current_num_threads, prelude::*};

use crate::{
    superfile::{
        BuildError, FtsError, SuperfileReader,
        fts::{positions::TermRuns, reader::FtsReader},
        id_space::FtsDocId,
    },
    utils::terms::FstValue,
};

/// Terms each input cursor reads from its dictionary at a time.
pub(crate) const TERMS_PER_CHUNK: usize = 4096;

/// Most terms in one parallel batch. Tiny under test, so the merge tests
/// cross a batch boundary at every kind of term they build.
const BATCH_TERMS: usize = if cfg!(test) { 3 } else { 1 << 16 };

/// Most postings in one batch, so its decoded postings stay bounded in
/// memory whatever the thread count: about 40 bytes each while merged,
/// plus 4 per position. A term with more postings is a batch of its own.
const BATCH_POSTINGS: u64 = if cfg!(test) { 1_200 } else { 1 << 24 };

/// Most terms one parallel task takes, so it takes a worker's scratch once
/// per few hundred terms rather than once per term. Small under test, so
/// a task on one thread takes several terms of a tiny batch.
const TASK_TERMS: usize = if cfg!(test) { 2 } else { 256 };

/// Tasks each thread gets from a batch, at least, when the batch has
/// enough terms. A small batch, such as one of very common terms, is still
/// spread over every thread. One under test, with [`TASK_TERMS`].
const TASKS_PER_THREAD: usize = if cfg!(test) { 1 } else { 4 };

/// Most bytes a worker's merge buffers may keep between terms. A worker
/// that merged a larger term drops them, so a few very common terms do not
/// leave every worker holding buffers of their size. The caller's state
/// is not counted; it trims its own. Tiny under test, so
/// the merge tests drop them after their long terms.
const KEEP_WORK_BYTES: usize = if cfg!(test) { 1 << 10 } else { 16 << 20 };

/// One merge input: a superfile and where its rows land in the output.
pub(crate) struct SortedInput {
    pub(crate) reader: Arc<SuperfileReader>,
    /// The output blob doc id each of this input's own doc ids becomes;
    /// `None` drops the row. Output ids must be unique across all inputs.
    pub(crate) remap: Vec<Option<FtsDocId>>,
}

/// Merge `column_id` across `inputs`: for every term that still has a
/// surviving posting, `encode(state, postings, runs)` builds its output
/// and `write(term, output)` places it. `postings` are
/// `(output_doc_id, tf)`, doc ids ascending; `runs` holds each posting's
/// position run, decoded (empty for a non-positional column).
///
/// `encode` runs on many threads at once, each call with a worker state
/// made by `new_state`; `write` runs on this thread, once per term, in term
/// order. Returns the worker states, so the caller can fold what they
/// gathered.
pub(crate) fn merge_column<S: Send, T: Send>(
    inputs: &[SortedInput],
    column_id: u32,
    new_state: impl Fn() -> S + Sync,
    encode: impl Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError> + Sync,
    mut write: impl FnMut(&str, T) -> Result<(), BuildError>,
) -> Result<Vec<S>, BuildError> {
    let readers = inputs
        .iter()
        .map(|input| input.reader.fts().ok_or(BuildError::BatchReadError))
        .collect::<Result<Vec<_>, _>>()?;
    let mut cursors: Vec<TermChunks> = readers
        .iter()
        .map(|fts| TermChunks::new(fts, column_id))
        .collect::<Result<_, _>>()
        .map_err(read_error)?;

    // Min-heap of each input's current term; ties pop in input order.
    let mut heap = BinaryHeap::new();
    let mut values = Vec::with_capacity(inputs.len());
    for (i, cursor) in cursors.iter_mut().enumerate() {
        let next = cursor.next().map_err(read_error)?;
        values.push(next.as_ref().map(|(_, value)| *value));
        if let Some((term, _)) = next {
            heap.push(Reverse((term, i)));
        }
    }

    let work = BatchWork {
        readers: &readers,
        inputs,
        column_id,
        workers: Mutex::new(Vec::new()),
        new_state: &new_state,
        encode: &encode,
    };
    let mut batch = Batch::default();
    // The current term's inputs, gathered before it joins a batch.
    let mut term_inputs: Vec<(usize, FstValue)> = Vec::new();
    while let Some(Reverse((term, first))) = heap.pop() {
        term_inputs.clear();
        let mut term_postings = 0u64;
        let mut next_input = Some(first);
        while let Some(i) = next_input {
            let value = values[i].ok_or(BuildError::BatchReadError)?;
            term_inputs.push((i, value));
            term_postings += u64::from(
                readers[i]
                    .term_postings_at_most(value)
                    .map_err(read_error)?,
            );
            let next = cursors[i].next().map_err(read_error)?;
            values[i] = next.as_ref().map(|(_, value)| *value);
            if let Some((next_term, _)) = next {
                heap.push(Reverse((next_term, i)));
            }
            next_input = match heap.peek() {
                Some(Reverse((t, _))) if *t == term => heap.pop().map(|Reverse((_, i))| i),
                _ => None,
            };
        }
        // A term that would take the batch past its postings starts a new one.
        if !batch.terms.is_empty() && batch.postings + term_postings > BATCH_POSTINGS {
            work.run(&batch, &mut write)?;
            batch.clear();
        }
        let term_start = batch.term_bytes.len();
        batch.term_bytes.extend_from_slice(&term);
        let inputs_start = batch.inputs.len();
        batch.inputs.extend_from_slice(&term_inputs);
        batch.postings += term_postings;
        batch.terms.push(PendingTerm {
            term: term_start..batch.term_bytes.len(),
            inputs: inputs_start..batch.inputs.len(),
        });
        if batch.terms.len() >= BATCH_TERMS || batch.postings >= BATCH_POSTINGS {
            work.run(&batch, &mut write)?;
            batch.clear();
        }
    }
    work.run(&batch, &mut write)?;

    Ok(work
        .workers
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .map(|(_, s)| s)
        .collect())
}

/// Terms per task for a batch of `n_terms`.
fn task_terms(n_terms: usize) -> usize {
    (n_terms / (current_num_threads() * TASKS_PER_THREAD)).clamp(1, TASK_TERMS)
}

/// One term waiting in a batch: where its bytes and its inputs'
/// dictionary values sit in the batch's shared buffers.
struct PendingTerm {
    term: Range<usize>,
    inputs: Range<usize>,
}

/// Terms walked off the dictionaries, waiting to be merged together.
#[derive(Default)]
struct Batch {
    terms: Vec<PendingTerm>,
    term_bytes: Vec<u8>,
    /// `(input, that input's dictionary value)` for every term's inputs.
    inputs: Vec<(usize, FstValue)>,
    postings: u64,
}

impl Batch {
    fn clear(&mut self) {
        self.terms.clear();
        self.term_bytes.clear();
        self.inputs.clear();
        self.postings = 0;
    }
}

/// One worker's buffers for merging a term.
#[derive(Default)]
struct TermWork {
    postings: Vec<(u32, u32)>,
    runs: Vec<u32>,
    /// Where each posting's run starts in `runs`.
    run_starts: Vec<usize>,
    positions_buf: Vec<u32>,
    sort_scratch: SortScratch,
}

impl TermWork {
    /// Bytes the buffers hold on to.
    fn capacity_bytes(&self) -> usize {
        self.postings.capacity() * size_of::<(u32, u32)>()
            + self.runs.capacity() * size_of::<u32>()
            + self.run_starts.capacity() * size_of::<usize>()
            + self.positions_buf.capacity() * size_of::<u32>()
            + self.sort_scratch.capacity_bytes()
    }
}

/// What every batch of one column's merge shares.
struct BatchWork<'a, S, N, E> {
    readers: &'a [&'a FtsReader],
    inputs: &'a [SortedInput],
    column_id: u32,
    /// Idle workers; a task takes one and puts it back.
    workers: Mutex<Vec<(TermWork, S)>>,
    new_state: &'a N,
    encode: &'a E,
}

impl<S: Send, N: Fn() -> S + Sync, E> BatchWork<'_, S, N, E> {
    /// Merge a batch's terms in parallel, then write them in term order.
    fn run<T: Send>(
        &self,
        batch: &Batch,
        write: &mut impl FnMut(&str, T) -> Result<(), BuildError>,
    ) -> Result<(), BuildError>
    where
        E: Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError> + Sync,
    {
        let merged = batch
            .terms
            .par_chunks(task_terms(batch.terms.len()))
            .map(|chunk| self.merge_chunk(batch, chunk))
            .collect::<Result<Vec<_>, _>>()?;
        for (pending, output) in batch.terms.iter().zip(merged.into_iter().flatten()) {
            let term = from_utf8(&batch.term_bytes[pending.term.clone()])
                .map_err(|_| BuildError::Io(Error::other("fts sorted merge: non-utf8 term")))?;
            // A term whose every posting was deleted leaves the output.
            if let Some(output) = output {
                write(term, output)?;
            }
        }
        Ok(())
    }

    /// Merge some consecutive terms of a batch on one worker.
    fn merge_chunk<T>(
        &self,
        batch: &Batch,
        chunk: &[PendingTerm],
    ) -> Result<Vec<Option<T>>, BuildError>
    where
        E: Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError>,
    {
        let taken = self
            .workers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop();
        let (mut work, mut state) =
            taken.unwrap_or_else(|| (TermWork::default(), (self.new_state)()));
        let merged = chunk
            .iter()
            .map(|pending| {
                let term = &batch.term_bytes[pending.term.clone()];
                let inputs = &batch.inputs[pending.inputs.clone()];
                let merged = self.merge_term(term, inputs, &mut work, &mut state);
                if work.capacity_bytes() > KEEP_WORK_BYTES {
                    work = TermWork::default();
                }
                merged
            })
            .collect();
        self.workers
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((work, state));
        merged
    }

    /// Read one term's postings from its inputs, remapped to output doc
    /// ids, sort them if they arrive out of order, and encode them. `None`
    /// when every posting was deleted.
    fn merge_term<T>(
        &self,
        term: &[u8],
        term_inputs: &[(usize, FstValue)],
        work: &mut TermWork,
        state: &mut S,
    ) -> Result<Option<T>, BuildError>
    where
        E: Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError>,
    {
        let TermWork {
            postings,
            runs,
            run_starts,
            positions_buf,
            sort_scratch,
        } = work;
        postings.clear();
        runs.clear();
        run_starts.clear();
        let mut ascending = true;
        for &(i, value) in term_inputs {
            let remap = &self.inputs[i].remap;
            self.readers[i]
                .for_each_posting_in(
                    self.column_id,
                    once((term, value)),
                    positions_buf,
                    |_, doc, tf, pos| {
                        if let Some(out_doc) = remap[doc as usize] {
                            debug_assert!(
                                pos.is_empty() || pos.len() == tf as usize,
                                "sorted merge: a position run must hold tf positions"
                            );
                            let out_doc = out_doc.get();
                            ascending &= postings.last().is_none_or(|&(d, _)| d < out_doc);
                            postings.push((out_doc, tf));
                            run_starts.push(runs.len());
                            push_run_values(runs, pos);
                        }
                        Ok(())
                    },
                )
                .map_err(read_error)?;
        }
        if postings.is_empty() {
            return Ok(None);
        }
        let starts = match ascending {
            true => run_starts.as_slice(),
            false => sort_scratch.sort(postings, run_starts),
        };
        debug_assert!(
            postings.is_sorted_by(|a, b| a.0 < b.0),
            "sorted merge: output doc ids must be unique"
        );
        let term_runs = TermRuns::Values {
            values: runs,
            starts,
        };
        (self.encode)(state, postings, term_runs).map(Some)
    }
}

/// Reused buffers for sorting one term's postings by output doc id.
#[derive(Default)]
struct SortScratch {
    /// `(output_doc_id, tf, run_start)` per posting.
    entries: Vec<(u32, u32, usize)>,
    /// The run starts in sorted order.
    starts: Vec<usize>,
}

impl SortScratch {
    /// Sort `postings` in place by doc id and return `run_starts` in the
    /// same order. Only the starts move; the run values stay where they are.
    fn sort(&mut self, postings: &mut [(u32, u32)], run_starts: &[usize]) -> &[usize] {
        self.entries.clear();
        self.entries.extend(
            postings
                .iter()
                .zip(run_starts)
                .map(|(&(doc, tf), &start)| (doc, tf, start)),
        );
        self.entries.sort_unstable_by_key(|e| e.0);
        self.starts.clear();
        for (slot, &(doc, tf, start)) in postings.iter_mut().zip(&self.entries) {
            *slot = (doc, tf);
            self.starts.push(start);
        }
        &self.starts
    }

    /// Bytes the buffers hold on to.
    fn capacity_bytes(&self) -> usize {
        self.entries.capacity() * size_of::<(u32, u32, usize)>()
            + self.starts.capacity() * size_of::<usize>()
    }
}

/// Append one document's positions to `out` as run values: the first
/// position, then the gap to each next one. The same values a LEB128 run
/// holds, without the encoding.
fn push_run_values(out: &mut Vec<u32>, positions: &[u32]) {
    let mut prev = 0u32;
    for (i, &p) in positions.iter().enumerate() {
        debug_assert!(i == 0 || p > prev, "positions must be strictly increasing");
        out.push(p - prev);
        prev = p;
    }
}

/// Walks one input column's dictionary in term order, a chunk at a time.
struct TermChunks<'a> {
    fts: &'a FtsReader,
    /// The input's dictionary, fetched once rather than per chunk.
    fst_bytes: Bytes,
    column_id: u32,
    chunk: vec::IntoIter<(Vec<u8>, FstValue)>,
    /// Last term of the latest chunk; the next chunk starts after it.
    resume: Vec<u8>,
    started: bool,
    done: bool,
}

impl<'a> TermChunks<'a> {
    fn new(fts: &'a FtsReader, column_id: u32) -> Result<Self, FtsError> {
        Ok(Self {
            fts,
            fst_bytes: fts.dict_bytes()?,
            column_id,
            chunk: Vec::new().into_iter(),
            resume: Vec::new(),
            started: false,
            done: false,
        })
    }

    fn next(&mut self) -> Result<Option<(Vec<u8>, FstValue)>, FtsError> {
        loop {
            if let Some(entry) = self.chunk.next() {
                return Ok(Some(entry));
            }
            if self.done {
                return Ok(None);
            }
            let terms = self.fts.column_terms_from(
                &self.fst_bytes,
                self.column_id,
                &self.resume,
                TERMS_PER_CHUNK,
            )?;
            self.done = terms.len() < TERMS_PER_CHUNK;
            // A chunk starts at `resume` itself, which was already handed out.
            let skip_first =
                self.started && terms.first().is_some_and(|(term, _)| *term == self.resume);
            if let Some((last, _)) = terms.last() {
                self.resume = last.clone();
            }
            self.started = true;
            self.chunk = terms.into_iter();
            if skip_first {
                self.chunk.next();
            }
        }
    }
}

fn read_error(e: FtsError) -> BuildError {
    BuildError::Io(Error::other(format!("fts sorted merge: {e}")))
}
