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
    mem,
    ops::Range,
    str::from_utf8,
    sync::{
        Arc, Mutex, PoisonError,
        mpsc::{channel, sync_channel},
    },
    thread,
    time::{Duration, Instant},
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
    utils::{terms::DictEntry, trace::record},
};

/// Terms each input cursor reads from its dictionary at a time.
pub(crate) const TERMS_PER_CHUNK: usize = 4096;

/// Key ranges a large column's dictionaries are walked in, side by side.
/// The walk is a heap over every input's cursor, one term at a time, and on
/// a merge of 68 inputs and 9M terms it ran 11 s on its one thread while the
/// encodes it fed took 7: the finish waited on it. Each range's walk runs
/// on a thread of its own, and its batches are taken in range order, so the
/// terms are written in the same order. A later range's batches wait whole
/// until the ranges before them are taken: they hold the terms and their
/// inputs' dictionary values, not postings, about a hundred bytes a term.
const WALK_RANGES: usize = 4;

/// Terms a column's sampled input must hold before its walk is split into
/// [`WALK_RANGES`]: a small merge walks in one range as before.
const WALK_SPLIT_MIN_TERMS: usize = if cfg!(test) { 8 } else { 1 << 16 };

/// Encoded batches the writer may fall behind the encodes by. One keeps
/// a batch's writes beside the next batch's encodes, which is the whole
/// overlap there is: a merge of 68 inputs and 9M terms spent 6 s of its
/// 11 s dispatching encodes and 4.5 s writing, one after the other per
/// batch. More would hold more encoded postings for no further overlap.
const ENCODED_BATCHES_AHEAD: usize = 1;

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
/// made by `new_state`; `write` runs on a thread of its own, once per
/// term, in term order, a batch behind the encodes. Returns the worker
/// states, so the caller can fold what they gathered.
pub(crate) fn merge_column<S: Send, T: Send>(
    inputs: &[SortedInput],
    column_id: u32,
    new_state: impl Fn() -> S + Sync,
    encode: impl Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError> + Sync,
    mut write: impl FnMut(&str, T) -> Result<(), BuildError> + Send,
) -> Result<Vec<S>, BuildError> {
    let readers = inputs
        .iter()
        .map(|input| input.reader.fts().ok_or(BuildError::BatchReadError))
        .collect::<Result<Vec<_>, _>>()?;
    // Each input's dictionary, fetched once and shared by every range's
    // cursor: on a lazy source each fetch is a full-dictionary read.
    let dicts = readers
        .iter()
        .map(|fts| fts.dict_bytes())
        .collect::<Result<Vec<_>, _>>()
        .map_err(read_error)?;
    let ranges = walk_ranges(&readers, &dicts, column_id)?;

    let work = BatchWork {
        readers: &readers,
        inputs,
        column_id,
        workers: Mutex::new(Vec::new()),
        new_state: &new_state,
        encode: &encode,
    };
    // The walk over the dictionaries runs ahead on threads of its own, one
    // per key range, so the next batch is gathered while this one is merged
    // and written. A range's walk runs on without bound, since nothing takes
    // a later range's batches until the ranges before it are done; a batch
    // holds terms and dictionary values, not postings. Batches are taken
    // range by range in the order they were walked, so the terms are
    // written in order.
    let (senders, receivers): (Vec<_>, Vec<_>) = (0..ranges.len())
        .map(|_| channel::<Result<Batch, BuildError>>())
        .unzip();
    // Three stages on three threads: the walk gathers batches, this thread
    // dispatches each batch's encodes to the pool, and the writer writes
    // the encoded terms out in order, a batch behind. Where this thread's
    // time goes: waiting on the walk, dispatching, and at the end waiting
    // for the writer.
    let (mut waited, mut encoded, mut written) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let (encoded_tx, encoded_rx) = sync_channel::<(Batch, Vec<Option<T>>)>(ENCODED_BATCHES_AHEAD);
    thread::scope(|scope| -> Result<(), BuildError> {
        let (readers, dicts) = (&readers, &dicts);
        for ((start, end), sender) in ranges.into_iter().zip(senders) {
            scope.spawn(move || {
                let walk = walk_range(readers, dicts, column_id, start, end, |batch| {
                    sender.send(Ok(batch)).is_ok()
                });
                if let Err(e) = walk {
                    let _ = sender.send(Err(e));
                }
            });
        }
        let mut walked = receivers.into_iter().flatten();
        let writer = scope.spawn(move || -> Result<Duration, BuildError> {
            let mut written = Duration::ZERO;
            for (batch, merged) in encoded_rx {
                let started = Instant::now();
                write_batch(&batch, merged, &mut write)?;
                written += started.elapsed();
            }
            Ok(written)
        });
        loop {
            let started = Instant::now();
            let Some(batch) = walked.next() else {
                break;
            };
            waited += started.elapsed();
            let batch = batch?;
            let started = Instant::now();
            let merged = work.encode(&batch)?;
            encoded += started.elapsed();
            // The writer is gone only after an error of its own, which the
            // join below reports.
            if encoded_tx.send((batch, merged)).is_err() {
                break;
            }
        }
        drop(encoded_tx);
        written = writer.join().map_err(|_| {
            BuildError::Io(Error::other(
                "fts sorted merge: the writer thread ended without a result",
            ))
        })??;
        Ok(())
    })?;
    record("walk_wait_ms", waited.as_millis() as u64);
    record("encode_ms", encoded.as_millis() as u64);
    record("write_ms", written.as_millis() as u64);

    Ok(work
        .workers
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .into_iter()
        .map(|(_, s)| s)
        .collect())
}

/// A key range of a column's walk: from `start` (none: the first term)
/// up to but not including `end` (none: past the last).
type WalkRange = (Option<Vec<u8>>, Option<Vec<u8>>);

/// The key ranges a column's walk runs in: [`WALK_RANGES`] cut at even
/// quantiles of the largest input's terms, or one when that input is small.
fn walk_ranges(
    readers: &[&FtsReader],
    dicts: &[Bytes],
    column_id: u32,
) -> Result<Vec<WalkRange>, BuildError> {
    let Some(largest) = (0..readers.len()).max_by_key(|&i| dicts[i].len()) else {
        return Ok(vec![(None, None)]);
    };
    let mut cursor = TermChunks::new(
        readers[largest],
        dicts[largest].clone(),
        column_id,
        None,
        None,
    );
    let mut terms: Vec<Vec<u8>> = Vec::new();
    while let Some((term, _)) = cursor.next().map_err(read_error)? {
        terms.push(term);
    }
    if terms.len() < WALK_SPLIT_MIN_TERMS {
        return Ok(vec![(None, None)]);
    }
    let cuts: Vec<Vec<u8>> = (1..WALK_RANGES)
        .map(|r| terms[terms.len() * r / WALK_RANGES].clone())
        .collect();
    let mut ranges = Vec::with_capacity(WALK_RANGES);
    let mut start = None;
    for cut in cuts {
        ranges.push((start.take(), Some(cut.clone())));
        start = Some(cut);
    }
    ranges.push((start, None));
    Ok(ranges)
}

/// Walk the terms of `start..end` across every input, in term order,
/// gathering them into batches for `emit`.
fn walk_range(
    readers: &[&FtsReader],
    dicts: &[Bytes],
    column_id: u32,
    start: Option<Vec<u8>>,
    end: Option<Vec<u8>>,
    emit: impl FnMut(Batch) -> bool,
) -> Result<(), BuildError> {
    let mut cursors: Vec<TermChunks> = readers
        .iter()
        .zip(dicts)
        .map(|(fts, dict)| {
            TermChunks::new(fts, dict.clone(), column_id, start.clone(), end.clone())
        })
        .collect();
    // Min-heap of each input's current term; ties pop in input order.
    let mut heap = BinaryHeap::new();
    let mut values = Vec::with_capacity(readers.len());
    for (i, cursor) in cursors.iter_mut().enumerate() {
        let next = cursor.next().map_err(read_error)?;
        values.push(next.as_ref().map(|(_, value)| *value));
        if let Some((term, _)) = next {
            heap.push(Reverse((term, i)));
        }
    }
    walk_batches(readers, &mut cursors, &mut heap, &mut values, emit)
}
/// Walk the inputs' dictionaries in term order, gathering terms into
/// batches and handing each to `emit` in order; the last, possibly empty,
/// batch too. Stops early once `emit` returns false.
fn walk_batches(
    readers: &[&FtsReader],
    cursors: &mut [TermChunks<'_>],
    heap: &mut BinaryHeap<Reverse<(Vec<u8>, usize)>>,
    values: &mut [Option<DictEntry>],
    mut emit: impl FnMut(Batch) -> bool,
) -> Result<(), BuildError> {
    let mut batch = Batch::default();
    // The current term's inputs, gathered before it joins a batch.
    let mut term_inputs: Vec<(usize, DictEntry)> = Vec::new();
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
        if !batch.terms.is_empty()
            && batch.postings + term_postings > BATCH_POSTINGS
            && !emit(mem::take(&mut batch))
        {
            return Ok(());
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
        if (batch.terms.len() >= BATCH_TERMS || batch.postings >= BATCH_POSTINGS)
            && !emit(mem::take(&mut batch))
        {
            return Ok(());
        }
    }
    emit(batch);
    Ok(())
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
    inputs: Vec<(usize, DictEntry)>,
    postings: u64,
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

/// Write a batch's encoded terms out, in term order. A term whose every
/// posting was deleted leaves the output.
fn write_batch<T>(
    batch: &Batch,
    merged: Vec<Option<T>>,
    write: &mut impl FnMut(&str, T) -> Result<(), BuildError>,
) -> Result<(), BuildError> {
    for (pending, output) in batch.terms.iter().zip(merged) {
        let term = from_utf8(&batch.term_bytes[pending.term.clone()])
            .map_err(|_| BuildError::Io(Error::other("fts sorted merge: non-utf8 term")))?;
        if let Some(output) = output {
            write(term, output)?;
        }
    }
    Ok(())
}

impl<S: Send, N: Fn() -> S + Sync, E> BatchWork<'_, S, N, E> {
    /// Merge a batch's terms in parallel, then write them in term order.
    /// Encode a batch's terms on the pool, in the batch's term order.
    fn encode<T: Send>(&self, batch: &Batch) -> Result<Vec<Option<T>>, BuildError>
    where
        E: Fn(&mut S, &[(u32, u32)], TermRuns<'_>) -> Result<T, BuildError> + Sync,
    {
        Ok(batch
            .terms
            .par_chunks(task_terms(batch.terms.len()))
            .map(|chunk| self.merge_chunk(batch, chunk))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect())
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
        term_inputs: &[(usize, DictEntry)],
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
    dict_bytes: Bytes,
    column_id: u32,
    chunk: vec::IntoIter<(Vec<u8>, DictEntry)>,
    /// Last term of the latest chunk; the next chunk starts after it. Before
    /// the first chunk, the range's first key.
    resume: Vec<u8>,
    /// The first term past the range, if it ends before the dictionary.
    end: Option<Vec<u8>>,
    started: bool,
    done: bool,
}

impl<'a> TermChunks<'a> {
    /// A cursor over the column's terms from `start` (none: the first) up
    /// to but not including `end` (none: the last).
    fn new(
        fts: &'a FtsReader,
        dict_bytes: Bytes,
        column_id: u32,
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Self {
        Self {
            fts,
            dict_bytes,
            column_id,
            chunk: Vec::new().into_iter(),
            resume: start.unwrap_or_default(),
            end,
            started: false,
            done: false,
        }
    }

    fn next(&mut self) -> Result<Option<(Vec<u8>, DictEntry)>, FtsError> {
        loop {
            if let Some(entry) = self.chunk.next() {
                if self.end.as_ref().is_some_and(|end| entry.0 >= *end) {
                    self.done = true;
                    self.chunk = Vec::new().into_iter();
                    return Ok(None);
                }
                return Ok(Some(entry));
            }
            if self.done {
                return Ok(None);
            }
            let terms = self.fts.column_terms_from(
                &self.dict_bytes,
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;

    use super::*;
    use crate::{
        superfile::{
            SuperfileReader,
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
        },
        test_helpers::decimal128_ids,
    };

    /// Inputs in the walk test, each holding a different stretch of terms
    /// with some shared by all.
    const INPUTS: u64 = 5;
    /// Documents per input.
    const DOCS: u64 = 400;

    fn input(n: u64) -> SuperfileReader {
        let schema = Arc::new(Schema::new(vec![
            Field::new("doc_id", DataType::Decimal128(38, 0), false),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            Arc::clone(&schema),
            "doc_id",
            vec![FtsConfig::new("title")],
            vec![],
        );
        let mut builder = SuperfileBuilder::new(opts).expect("builder");
        let ids: Vec<u64> = (n * DOCS..(n + 1) * DOCS).collect();
        let titles: Vec<String> = ids
            .iter()
            .map(|&i| format!("shared w{} w{} own{n}x{}", i % 97, (i * 7) % 211, i % 13))
            .collect();
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(decimal128_ids(ids)),
                Arc::new(LargeStringArray::from(titles)),
            ],
        )
        .expect("batch");
        builder.add_batch(&batch, &[]).expect("add_batch");
        SuperfileReader::open(Bytes::from(builder.finish().expect("finish"))).expect("open")
    }

    /// Every term of a walk, in order, with the inputs that hold it.
    fn walked(
        readers: &[&FtsReader],
        dicts: &[Bytes],
        start: Option<Vec<u8>>,
        end: Option<Vec<u8>>,
    ) -> Vec<(Vec<u8>, Vec<usize>)> {
        let mut out = Vec::new();
        walk_range(readers, dicts, 0, start, end, |batch| {
            for term in &batch.terms {
                out.push((
                    batch.term_bytes[term.term.clone()].to_vec(),
                    batch.inputs[term.inputs.clone()]
                        .iter()
                        .map(|(i, _)| *i)
                        .collect(),
                ));
            }
            true
        })
        .expect("walk");
        out
    }

    /// The ranges a column is walked in cover its terms once each, in
    /// order: their walks, one after another, are the one walk.
    #[test]
    fn a_walk_in_ranges_is_the_whole_walk_in_order() {
        let inputs: Vec<SuperfileReader> = (0..INPUTS).map(input).collect();
        let readers: Vec<&FtsReader> = inputs.iter().map(|r| r.fts().expect("fts")).collect();
        let dicts: Vec<Bytes> = readers
            .iter()
            .map(|f| f.dict_bytes().expect("dict"))
            .collect();
        let ranges = walk_ranges(&readers, &dicts, 0).expect("ranges");
        assert_eq!(
            ranges.len(),
            WALK_RANGES,
            "a column past the floor is split"
        );
        for pair in ranges.windows(2) {
            assert_eq!(
                pair[0].1, pair[1].0,
                "each range starts where the last ends"
            );
            assert!(pair[1].0 > pair[0].0, "cut keys rise");
        }
        assert_eq!(
            (ranges[0].0.as_ref(), ranges[WALK_RANGES - 1].1.as_ref()),
            (None, None)
        );
        let whole = walked(&readers, &dicts, None, None);
        let pieced: Vec<_> = ranges
            .into_iter()
            .flat_map(|(start, end)| walked(&readers, &dicts, start, end))
            .collect();
        assert!(whole.len() > WALK_SPLIT_MIN_TERMS);
        assert_eq!(pieced, whole);
    }
}
