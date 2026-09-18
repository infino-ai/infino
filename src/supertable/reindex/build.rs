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
//! - [`super::ReindexMode::Rewrite`] reuses compaction's build unchanged.
//!   Carrying postings into a current container is exactly what a merge
//!   already does, and restating it here would be a second copy that could
//!   drift from the one the table is actually compacted with.
//! - [`super::ReindexMode::Reanalyze`] is this module's own: it rebuilds
//!   terms from stored text, which no merge does or should do.

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
        compaction::{MergeInputs, SuperfileMerge},
    },
};

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
