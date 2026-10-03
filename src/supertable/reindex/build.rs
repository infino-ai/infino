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
    io::{Error, Write},
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
        manifest::SuperfileEntry,
        optimize::compact::{CompactionMerge, MergeInputs, SuperfileMerge},
        reindex::Repair,
    },
};

/// The single resident input a carrying rewrite needs, or `None` when the
/// merge path has to run instead.
///
/// Three things have to hold: one input (the migration's job shape), a
/// reader over whole valid bytes (a lazily-opened one has no body to
/// copy), and no rows dropped (the carried body would describe rows the
/// output no longer has).
///
/// A reindex job opens its input with no tombstone bitmap, so the
/// `deleted` half is true today. It is kept because the cost is one
/// compare and the alternative is a silently wrong body the day a caller
/// does pass one.
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

/// The row count a carried body will hold, checked against the manifest.
///
/// A carrying build reports `entry.n_docs` as its output count, so a
/// downstream comparison of output against entry compares a value with
/// itself. The file itself is the independent number: if the manifest and
/// the body disagree, the tombstones carried onto the output describe
/// different rows than it holds.
fn carried_doc_count(
    source: &Arc<SuperfileReader>,
    entry: &Arc<SuperfileEntry>,
) -> Result<(), BuildError> {
    if source.n_docs() != entry.n_docs {
        return Err(BuildError::Superfile(SuperfileBuildError::Io(
            Error::other(format!(
                "superfile {} holds {} rows but its manifest entry says {}",
                entry.superfile_id,
                source.n_docs(),
                entry.n_docs
            )),
        )));
    }
    Ok(())
}

/// A reindex's build, parameterised by the repair the plan chose.
///
/// One type rather than two: the two repairs differ in which builder call
/// produces the terms and whether the stored columns are re-analyzed, and
/// nothing else. Spelling that as two structs and two carrying builds left
/// four places for the parts they share to drift apart.
pub(crate) struct RepairMerge(pub(crate) Repair);

impl SuperfileMerge for RepairMerge {
    fn build(
        &self,
        inputs: MergeInputs<'_>,
        output: &mut dyn Write,
    ) -> Result<SuperfileStats, BuildError> {
        match carry_body(&inputs) {
            Some((reader, entry)) => {
                repair_carrying_body_to(self.0, reader, entry, inputs.fts_corpus, output)
            }
            // Restating compaction's merge here would be a second copy that
            // could drift from the one the table is actually compacted with.
            None => match self.0 {
                Repair::Layout => CompactionMerge.build(inputs, output),
                Repair::Terms => reanalyze_to(inputs.readers, inputs.fts_corpus, output),
            },
        }
    }

    fn preserves_tombstones(&self) -> bool {
        true
    }
}

/// Apply `repair` to `source`'s FTS index, and copy every other byte
/// across.
///
/// Rows are unchanged by either repair — only the index is — so the body
/// and the vector subsection carry and the vectors are never decoded. That
/// is what the append path cannot do for a quantized codec.
///
/// The output's stats are the input's: a carried body holds the same rows
/// in the same order, so recomputing them from a decode this build does
/// not perform would only be a chance to get them wrong.
fn repair_carrying_body_to(
    repair: Repair,
    source: &Arc<SuperfileReader>,
    entry: &Arc<SuperfileEntry>,
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: &mut dyn Write,
) -> Result<SuperfileStats, BuildError> {
    let readers = [(Arc::clone(source), None)];
    let opts = merge_builder_opts(&readers, fts_corpus)?;
    let mut builder = SuperfileBuilder::new(match repair {
        Repair::Layout => opts,
        Repair::Terms => opts.reanalyze_stored_columns(),
    })?;
    match repair {
        Repair::Layout => {
            builder.carry_fts_from_reader_scoped(source, None, CarryScope::AllColumns)?
        }
        Repair::Terms => builder.reanalyze_fts_from_reader(source)?,
    }
    carried_doc_count(source, entry)?;
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
/// postings are carried and it keeps the revision it was built at.
///
/// Vectors are decoded and re-encoded rather than spliced, so the output
/// holds whatever that round trip produces. Nothing gates which codec may
/// take this path, so a codec that does not round-trip exactly would move
/// its vectors here.
fn reanalyze_to<W: Write>(
    readers: &[(Arc<SuperfileReader>, Option<Arc<RoaringBitmap>>)],
    fts_corpus: &HashMap<String, ColumnLengthStats>,
    output: W,
) -> Result<SuperfileStats, BuildError> {
    let builder_opts = merge_builder_opts(readers, fts_corpus)?.reanalyze_stored_columns();
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow_array::RecordBatch;
    use bytes::Bytes;
    use uuid::Uuid;

    use super::*;
    use crate::{
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            reader::SuperfileReader,
            vector::layout::VectorLayout,
        },
        supertable::manifest::SuperfileUri,
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    /// A carrying build reports the manifest's row count as its own, so
    /// the only number that can disagree is the file's. A manifest entry
    /// that has drifted from its superfile must stop the build: the
    /// tombstones carried onto the output are indexed by row position, so
    /// a different row count means they mark different rows.
    #[test]
    fn a_manifest_row_count_that_disagrees_with_the_file_refuses_the_carry() {
        use arrow_array::LargeStringArray;
        use arrow_schema::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let opts = BuilderOptions::new(
            schema.clone(),
            "doc_id",
            vec![FtsConfig::new("title")],
            vec![],
        );
        let mut b = SuperfileBuilder::new(opts).expect("new builder");
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(decimal128_ids(vec![1u64, 2])),
                Arc::new(LargeStringArray::from(vec!["hello", "world"])),
            ],
        )
        .expect("batch matches schema");
        b.add_batch(&batch, &[]).expect("add");
        let source = Arc::new(
            SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open"),
        );

        let id = Uuid::from_u128(1);
        let honest = Arc::new(SuperfileEntry {
            physical_schema: None,
            superfile_id: id,
            uri: SuperfileUri(id),
            stem: None,
            n_docs: source.n_docs(),
            id_min: 0,
            id_max: 0,
            scalar_stats: HashMap::new(),
            fts_summary: HashMap::new(),
            vector_summary: HashMap::new(),
            partition_key: Vec::new(),
            partition_hint: None,
            subsection_offsets: None,
            birth_version: 0,
            vector_layout: VectorLayout::Ivf,
        });
        assert!(
            carried_doc_count(&source, &honest).is_ok(),
            "an entry that matches its file carries"
        );

        let drifted = Arc::new(SuperfileEntry {
            physical_schema: None,
            n_docs: source.n_docs() + 1,
            ..(*honest).clone()
        });
        assert!(
            carried_doc_count(&source, &drifted).is_err(),
            "an entry claiming a row the file does not hold must refuse"
        );
    }
}
