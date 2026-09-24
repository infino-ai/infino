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

use std::{collections::HashMap, io::Write, sync::Arc};

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
        CompactionMerge.build(inputs, output)
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
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
        reanalyze_to(inputs.readers, inputs.fts_corpus, output)
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
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
