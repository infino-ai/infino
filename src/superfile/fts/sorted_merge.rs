// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Merge the FTS indexes of already-built superfiles term by term.
//!
//! Every input's dictionary is in term order, and every input's rows land
//! in the output in input order through a remap that never goes down. So
//! the merged postings of a term are each input's postings for it, joined
//! in input order — already sorted by output doc id. A k-way merge over the
//! inputs' dictionaries yields the output in final order, with no
//! corpus-sized accumulator and no sort; peak memory is one term's
//! postings across all inputs.

use std::{
    cmp::Reverse, collections::BinaryHeap, io::Error, iter::once, str::from_utf8, sync::Arc, vec,
};

use bytes::Bytes;

use crate::{
    superfile::{
        BuildError, FtsError, SuperfileReader,
        fts::{positions::encode_run, reader::FtsReader},
    },
    utils::terms::FstValue,
};

/// Terms each input cursor reads from its dictionary at a time.
pub(crate) const TERMS_PER_CHUNK: usize = 4096;

/// One merge input: a superfile and where its rows land in the output.
pub(crate) struct SortedInput {
    pub(crate) reader: Arc<SuperfileReader>,
    /// Input-local doc id → output doc id; `None` drops the row. Must
    /// never go down, and must sit above every earlier input's ids.
    pub(crate) remap: Vec<Option<u32>>,
}

/// Merge `column_id` across `inputs` and call `emit(term, postings, runs)`
/// once per term that still has a surviving posting, in term order.
/// `postings` are `(output_doc_id, tf)`, doc ids ascending; `runs` holds
/// each posting's encoded positions back to back (empty for a
/// non-positional column).
pub(crate) fn merge_column(
    inputs: &[SortedInput],
    column_id: u32,
    mut emit: impl FnMut(&str, &[(u32, u32)], &[u8]) -> Result<(), BuildError>,
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
    let mut runs: Vec<u8> = Vec::new();
    let mut positions_buf: Vec<u32> = Vec::new();
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
                            // Nothing sorts these postings, so a remap that
                            // goes down would write a wrong posting list.
                            debug_assert!(
                                postings.last().is_none_or(|&(d, _)| d < out_doc),
                                "sorted merge: output doc ids must ascend"
                            );
                            postings.push((out_doc, tf));
                            encode_run(&mut runs, pos);
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
        emit(term, &postings, &runs)?;
    }
    Ok(())
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
