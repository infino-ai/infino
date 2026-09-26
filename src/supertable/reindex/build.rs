// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How a reindex builds the superfile it commits.
//!
//! The migration drives compaction's seal → build → commit → unseal cycle,
//! because that cycle is what makes a rewrite safe across processes and is
//! worth exactly one implementation. Only what it *builds* differs.
//!
//! Both builds carry every row: dropping the dead ones would renumber the
//! survivors, and the job runner carries their tombstones onto the output.

use std::{
    collections::{BTreeSet, HashMap},
    io::Write,
    sync::Arc,
};

use roaring::RoaringBitmap;

use crate::{
    superfile::{
        builder::{CarryScope, SuperfileBuilder, merge_builder_opts},
        error::BuildError as SuperfileBuildError,
        fts::reader::ColumnLengthStats,
        reader::SuperfileReader,
        stats::SuperfileStats,
    },
    supertable::{
        BuildError,
        compaction::{CompactionMerge, MergeInputs, SuperfileMerge},
        manifest::SuperfileEntry,
    },
};

/// Compaction's build, carrying the rows it would otherwise drop.
///
/// Restating the merge here would be a second copy that could drift from
/// the one the table is actually compacted with.
pub(crate) struct RewriteMerge;

impl SuperfileMerge for RewriteMerge {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<SuperfileStats, BuildError> {
        match carry_body(&inputs) {
            Some((reader, entry)) => {
                rewrite_carrying_body_to(reader, entry, inputs.fts_corpus, output)
            }
            None => CompactionMerge.build(inputs, output),
        }
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
}

/// The single resident input a carrying rewrite needs, or `None` when the
/// merge path has to run instead.
///
/// Three things have to hold: one input (the migration's job shape), a
/// reader over whole valid bytes (a lazily-opened one has no body to
/// copy), and no tombstones (the carried body would describe rows the
/// output no longer has).
fn carry_body<'a>(
    inputs: &'a MergeInputs<'a>,
) -> Option<(&'a Arc<SuperfileReader>, &'a Arc<SuperfileEntry>)> {
    let ([(reader, deleted)], [entry]) = (inputs.readers, inputs.entries) else {
        return None;
    };
    let carries_every_row = deleted.as_ref().is_none_or(|b| b.is_empty())
        && inputs.superseded.iter().all(BTreeSet::is_empty);
    (carries_every_row && reader.is_fully_resident()).then_some((reader, entry))
}

/// Rebuild the FTS index, and copy every other byte of `source` across.
///
/// The output's stats are the input's: a carried body holds the same rows
/// in the same order, so recomputing them from a decode this build does
/// not perform would only be a chance to get them wrong.
fn rewrite_carrying_body_to(
    source: &Arc<SuperfileReader>,
    entry: &Arc<SuperfileEntry>,
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: &mut dyn Write,
) -> Result<SuperfileStats, BuildError> {
    let readers = [(Arc::clone(source), None)];
    let first = &readers[0];
    let builder_opts = merge_builder_opts(&readers, first, fts_corpus);
    let mut builder = SuperfileBuilder::new(builder_opts)?;
    builder.carry_fts_from_reader_scoped(source, None, CarryScope::AllColumns)?;
    builder.set_carried_doc_count(entry.n_docs);
    builder.finish_carrying_body_to(source, output)?;
    Ok(SuperfileStats {
        n_docs: entry.n_docs,
        id_min: entry.id_min,
        id_max: entry.id_max,
        scalar_stats: entry.scalar_stats.clone(),
    })
}

/// Rebuilds each input's FTS index from the text it stored.
///
/// The only build that changes a file's *terms*: every other rebuild
/// copies postings, which is why a container rewrite leaves an older
/// analyzer's output where it was.
pub(crate) struct ReanalyzeMerge;

impl SuperfileMerge for ReanalyzeMerge {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<SuperfileStats, BuildError> {
        match carry_body(&inputs) {
            Some((reader, entry)) => {
                reanalyze_carrying_body_to(reader, entry, inputs.fts_corpus, output)
            }
            None => reanalyze_to(inputs.readers, inputs.fts_corpus, output),
        }
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
}

/// Re-analyze `source`'s terms, and copy every other byte across.
///
/// Rows are unchanged by a re-analysis — only terms are — so the body and
/// the vector subsection carry, and the vectors are never decoded. That is
/// what the append path cannot do for a quantized codec.
fn reanalyze_carrying_body_to(
    source: &Arc<SuperfileReader>,
    entry: &Arc<SuperfileEntry>,
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: &mut dyn Write,
) -> Result<SuperfileStats, BuildError> {
    let readers = [(Arc::clone(source), None)];
    let first = &readers[0];
    let builder_opts = merge_builder_opts(&readers, first, fts_corpus).reanalyze_stored_columns();
    let mut builder = SuperfileBuilder::new(builder_opts)?;
    builder.reanalyze_fts_from_reader(source)?;
    builder.set_carried_doc_count(entry.n_docs);
    builder.finish_carrying_body_to(source, output)?;
    Ok(SuperfileStats {
        n_docs: entry.n_docs,
        id_min: entry.id_min,
        id_max: entry.id_max,
        scalar_stats: entry.scalar_stats.clone(),
    })
}

/// Rebuild `readers` into one superfile, re-analyzing every stored column.
///
/// A column whose text was never stored cannot be re-analyzed, so its
/// postings are carried and it keeps the revision it was built at. Vectors
/// are decoded and re-encoded rather than spliced, so only a codec that
/// round-trips exactly can take this path — the caller refuses the rest
/// before anything commits.
fn reanalyze_to<W: Write>(
    readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: W,
) -> Result<SuperfileStats, BuildError> {
    let first = readers.first().ok_or(SuperfileBuildError::BatchReadError)?;
    let builder_opts = merge_builder_opts(readers, first, fts_corpus).reanalyze_stored_columns();
    let mut builder = SuperfileBuilder::new(builder_opts)?;

    let mut stats = Vec::with_capacity(readers.len());
    for (reader, deleted) in readers {
        stats.push(builder.add_batch_from_reader_scoped(
            reader,
            deleted.clone(),
            CarryScope::UnstoredOnly,
        )?);
    }

    builder.finish_to(output)?;
    Ok(SuperfileStats::from_children(stats.as_slice()))
}
