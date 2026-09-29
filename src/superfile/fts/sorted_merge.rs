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

use std::{
    cmp::Reverse, collections::BinaryHeap, io::Error, iter::once, str::from_utf8, sync::Arc, vec,
};

use bytes::Bytes;

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

/// One merge input: a superfile and where its rows land in the output.
pub(crate) struct SortedInput {
    pub(crate) reader: Arc<SuperfileReader>,
    /// The output blob doc id each of this input's own doc ids becomes;
    /// `None` drops the row. Output ids must be unique across all inputs.
    pub(crate) remap: Vec<Option<FtsDocId>>,
}

/// Merge `column_id` across `inputs` and call `emit(term, postings, runs)`
/// once per term that still has a surviving posting, in term order.
/// `postings` are `(output_doc_id, tf)`, doc ids ascending; `runs` holds
/// each posting's position run, decoded (empty for a non-positional
/// column).
pub(crate) fn merge_column(
    inputs: &[SortedInput],
    column_id: u32,
    mut emit: impl FnMut(&str, &[(u32, u32)], TermRuns<'_>) -> Result<(), BuildError>,
) -> Result<(), BuildError> {
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

    let mut contributors: Vec<usize> = Vec::new();
    let mut postings: Vec<(u32, u32)> = Vec::new();
    let mut runs: Vec<u32> = Vec::new();
    // Where each posting's run starts in `runs`.
    let mut run_starts: Vec<usize> = Vec::new();
    let mut positions_buf: Vec<u32> = Vec::new();
    let mut sort_scratch = SortScratch::default();
    while let Some(Reverse((term, first))) = heap.pop() {
        contributors.clear();
        contributors.push(first);
        while heap.peek().is_some_and(|Reverse((t, _))| *t == term) {
            if let Some(Reverse((_, i))) = heap.pop() {
                contributors.push(i);
            }
        }

        postings.clear();
        runs.clear();
        run_starts.clear();
        let mut ascending = true;
        for &i in &contributors {
            let value = values[i].ok_or(BuildError::BatchReadError)?;
            let remap = &inputs[i].remap;
            readers[i]
                .for_each_posting_in(
                    column_id,
                    once((term.as_slice(), value)),
                    &mut positions_buf,
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
                            push_run_values(&mut runs, pos);
                        }
                        Ok(())
                    },
                )
                .map_err(read_error)?;
            let next = cursors[i].next().map_err(read_error)?;
            values[i] = next.as_ref().map(|(_, value)| *value);
            if let Some((next_term, _)) = next {
                heap.push(Reverse((next_term, i)));
            }
        }

        let term = from_utf8(&term)
            .map_err(|_| BuildError::Io(Error::other("fts sorted merge: non-utf8 term")))?;
        // A term whose every posting was deleted leaves the output.
        if postings.is_empty() {
            continue;
        }
        if ascending {
            let term_runs = TermRuns::Values {
                values: &runs,
                starts: &run_starts,
            };
            emit(term, &postings, term_runs)?;
            continue;
        }
        let starts = sort_scratch.sort(&mut postings, &run_starts);
        debug_assert!(
            postings.is_sorted_by(|a, b| a.0 < b.0),
            "sorted merge: output doc ids must be unique"
        );
        let term_runs = TermRuns::Values {
            values: &runs,
            starts,
        };
        emit(term, &postings, term_runs)?;
    }
    Ok(())
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
