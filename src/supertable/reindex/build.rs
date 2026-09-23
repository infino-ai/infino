// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! How a reindex builds the superfile it commits.
//!
//! The migration drives the same seal → build → commit → unseal cycle
//! compaction does, because that cycle is what makes a rewrite safe across
//! processes and is worth exactly one implementation. What it *builds* is
//! its own business, and lives here rather than as a branch inside the
//! merge path — a tool that rewrites committed data on demand should not
//! be something the everyday compaction path has to know about.
//!
//! Two builds, picked by mode:
//!
//! - [`super::ReindexMode::Rewrite`] reuses compaction's build with its
//!   deletions turned off. Carrying postings into a current container is
//!   exactly what a merge already does, and restating it here would be a
//!   second copy that could drift from the one the table is actually
//!   compacted with.
//! - [`super::ReindexMode::Reanalyze`] is this module's own: it rebuilds
//!   terms from stored text, which no merge does or should do.
//!
//! Both carry every row, tombstoned ones included. A compaction drops dead
//! rows to reclaim their space; a migration reclaims nothing, and dropping
//! them would renumber the survivors — the one change that would turn
//! carrying a vector subsection from a byte copy into a remapping. The
//! dead rows stay dead because the job runner carries their tombstones
//! onto the output, which is what `preserves_tombstones` asks for.

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

/// Carries every row, tombstoned ones included, into the current layout.
///
/// Compaction's build with its deletions turned off. A compaction drops
/// dead rows because reclaiming their space is the point of it; a
/// migration has no business reclaiming anything, and dropping them would
/// renumber every surviving row — which is the one thing that makes
/// carrying the vector subsection across a byte copy rather than a
/// remapping exercise. Keeping the row set is what keeps a migration's
/// output identical to its input everywhere the FTS index is not.
///
/// The dead rows stay dead: the job runner carries their tombstones onto
/// the output because this build answers `preserves_tombstones`. They are
/// reclaimed by ordinary compaction, whenever that next runs.
pub(crate) struct RewriteMerge;

impl SuperfileMerge for RewriteMerge {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<SuperfileStats, BuildError> {
        let carried: Vec<(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)> = inputs
            .readers
            .iter()
            .map(|(reader, _deleted)| (Arc::clone(reader), None))
            .collect();
        CompactionMerge.build(
            MergeInputs {
                readers: &carried,
                superseded: inputs.superseded,
                fts_corpus: inputs.fts_corpus,
            },
            output,
        )
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
}

/// Rebuilds each input's FTS index from the text it stored.
///
/// The only build that changes a file's *terms*. Every other rebuild
/// copies postings, which is why a container rewrite leaves an older
/// analyzer's output exactly where it was: the terms are the stale thing,
/// and only re-tokenizing the source text replaces them.
///
/// Columns whose text was never stored cannot be re-analyzed, so their
/// postings are carried and they keep the revision they were built at. The
/// output is then honestly mixed — current terms where there was text,
/// older terms where there was not — and the run reports which.
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

/// Rebuild `readers` into one superfile, re-analyzing every column whose
/// text is stored.
///
/// Vectors are decoded from the input and re-encoded by the normal append
/// path rather than spliced, so this costs more than a merge — the trade
/// for changing terms at all — and only a codec that round-trips exactly
/// can take it. The caller refuses the rest before anything commits.
fn reanalyze_to<W: Write>(
    readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: W,
) -> Result<SuperfileStats, BuildError> {
    let first = readers.first().ok_or(SuperfileBuildError::BatchReadError)?;
    let builder_opts = merge_builder_opts(readers, first, fts_corpus).reanalyze_stored_columns();
    let mut builder = SuperfileBuilder::new(builder_opts)?;

    let mut stats = Vec::with_capacity(readers.len());
    // Each input's deletion bitmap is deliberately ignored: a reindex
    // carries the row set, so local doc ids stay put and the tombstones
    // carried onto the output still name the same rows. See `RewriteMerge`.
    for (reader, _deleted) in readers {
        stats.push(builder.add_batch_from_reader_scoped(reader, None, CarryScope::UnstoredOnly)?);
    }

    builder.finish_to(output)?;
    Ok(SuperfileStats::from_children(stats.as_slice()))
}
