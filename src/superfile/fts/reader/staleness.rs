// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Whether a superfile's FTS index is what this engine would write today.
//!
//! Two independent things can be behind, and they need different repairs,
//! so they are reported separately rather than as one "stale" flag:
//!
//! - The **container** — the blob's layout and how its block-max bounds
//!   encode. An older container is read correctly but scored with looser
//!   bounds, and costs the faster kernels its version does not advertise.
//!   Copying the postings into a current container fixes it, because the
//!   terms themselves are fine.
//! - The **analysis** — which version of a named chain produced the terms.
//!   Copying postings cannot fix this: the terms are the problem, so only
//!   re-analyzing the source text does, and a column whose text was never
//!   stored cannot be repaired at all.
//!
//! Reporting them apart is what lets a migration choose the cheap repair
//! where it suffices and reserve the expensive one for where it does not.
//!

use crate::superfile::{
    format,
    fts::{analysis::chain_revision, reader::FtsReader},
};

/// What is behind in one superfile's FTS index, if anything.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct FtsStaleness {
    /// The blob's version when it is below [`format::fts::VERSION_CURRENT`],
    /// else `None`.
    pub(crate) container: Option<u32>,
    /// Columns whose terms were produced by an older revision of their
    /// own analysis chain, with the revision each records.
    pub(crate) analysis: Vec<StaleColumn>,
}

/// One column whose terms predate its chain's current revision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StaleColumn {
    pub(crate) name: String,
    /// The revision the column records; `0` when it predates the field.
    pub(crate) recorded: u32,
    /// The revision this engine's chain emits for the same analysis.
    pub(crate) current: u32,
    /// Whether the column's text is in the Parquet body.
    ///
    /// `false` means the terms cannot be regenerated from anything this
    /// file holds, so no rewrite repairs the column and a migration has
    /// to say so rather than silently carry it forward as if it had.
    pub(crate) stored: bool,
}

impl FtsStaleness {
    /// Nothing is behind, on either axis.
    pub(crate) fn is_current(&self) -> bool {
        self.container.is_none() && self.analysis.is_empty()
    }

    /// Whether copying this index into a current container would change
    /// anything.
    ///
    /// This is the question a rewrite must be planned on, and it is *not*
    /// [`Self::is_current`]. A rewrite carries postings; it cannot
    /// re-analyze them, so it never clears an analysis revision — and
    /// planning rewrites for a file that is only analysis-stale would
    /// rewrite it on every run, forever, each one producing a file as
    /// stale as the last.
    pub(crate) fn needs_rewrite(&self) -> bool {
        self.container.is_some()
    }

    /// Whether re-analyzing this file would change any of its terms.
    ///
    /// Counts only columns whose text is stored, and that restriction is
    /// what makes a re-analysis terminate. An index-only column's terms
    /// cannot be regenerated, so it comes out of a rebuild exactly as
    /// stale as it went in — planning on it would re-tokenize the whole
    /// corpus on every run and never converge, which is the same trap
    /// [`Self::needs_rewrite`] documents for the container axis.
    ///
    /// Columns this skips are not forgotten: they are reported through
    /// [`Self::unrepairable_columns`], because a table that cannot be
    /// fully repaired should say so rather than look finished.
    pub(crate) fn needs_reanalysis(&self) -> bool {
        self.analysis.iter().any(|c| c.stored)
    }

    /// Stale columns no rebuild can repair, because their text was never
    /// stored.
    pub(crate) fn unrepairable_columns(&self) -> impl Iterator<Item = &StaleColumn> {
        self.analysis.iter().filter(|c| !c.stored)
    }
}

impl FtsReader {
    /// What this index is behind on, if anything.
    pub(crate) fn staleness(&self) -> FtsStaleness {
        let container = match self.version < format::fts::VERSION_CURRENT {
            true => Some(self.version),
            false => None,
        };
        let analysis = self
            .fts_columns_config()
            .filter_map(|c| {
                // Compare against the revision of *this column's own*
                // chain, not a single engine-wide number: a column
                // analyzed with `ascii_lower` is not stale because
                // `standard` moved.
                let current = chain_revision(c.base, c.stopwords, c.stemmer);
                (c.analysis_rev < current).then(|| StaleColumn {
                    name: c.name.clone(),
                    recorded: c.analysis_rev,
                    current,
                    stored: c.stored,
                })
            })
            .collect();
        FtsStaleness {
            container,
            analysis,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{LargeStringArray, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use bytes::Bytes;

    use crate::{
        superfile::{
            builder::{BuilderOptions, FtsConfig, SuperfileBuilder},
            reader::SuperfileReader,
        },
        test_helpers::{decimal128_id_field, decimal128_ids},
    };

    /// Build a one-column superfile, optionally standing in for a file
    /// whose postings came from an older analysis.
    fn reader(carried: Option<u32>, stored: bool) -> Arc<SuperfileReader> {
        let schema = Arc::new(Schema::new(vec![
            decimal128_id_field("doc_id"),
            Field::new("title", DataType::LargeUtf8, false),
        ]));
        let mut col = FtsConfig::new("title").stored(stored);
        if let Some(rev) = carried {
            col = col.carried_analysis_rev(rev);
        }
        let opts = BuilderOptions::new(schema.clone(), "doc_id", vec![col], vec![]);
        let mut b = SuperfileBuilder::new(opts).expect("new builder");
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(decimal128_ids(vec![1u64])),
                Arc::new(LargeStringArray::from(vec!["hello world"])),
            ],
        )
        .expect("batch matches schema");
        b.add_batch(&batch, &[]).expect("add");
        Arc::new(SuperfileReader::open(Bytes::from(b.finish().expect("finish"))).expect("open"))
    }

    /// A file this engine just wrote is behind on nothing.
    #[test]
    fn a_fresh_build_is_current() {
        let r = reader(None, true);
        let stale = r.fts().expect("fts").staleness();
        assert!(stale.is_current(), "{stale:?}");
    }

    /// Terms carried from an older analysis are stale even though the
    /// container this engine just wrote is current — which is the whole
    /// reason the two are reported separately. A migration that looked
    /// only at the blob version would call this file done.
    #[test]
    fn current_container_does_not_imply_current_analysis() {
        let r = reader(Some(0), true);
        let stale = r.fts().expect("fts").staleness();
        assert_eq!(stale.container, None, "the container is this engine's");
        assert_eq!(stale.analysis.len(), 1, "{stale:?}");
        assert_eq!(stale.analysis[0].recorded, 0);
        assert_eq!(stale.analysis[0].current, 1);
        assert!(!stale.is_current());
    }

    /// An index-only column's text is not in the file, so no rebuild can
    /// regenerate its terms — and planning one anyway would never
    /// terminate, since the rebuilt file is as stale as its input.
    ///
    /// The column is still reported, so a table that cannot be fully
    /// repaired says so rather than looking finished.
    #[test]
    fn an_index_only_stale_column_is_reported_but_not_planned() {
        let stored = reader(Some(0), true).fts().expect("fts").staleness();
        assert!(stored.needs_reanalysis(), "stored text can be re-analyzed");
        assert_eq!(stored.unrepairable_columns().count(), 0);

        let index_only = reader(Some(0), false).fts().expect("fts").staleness();
        assert_eq!(index_only.analysis.len(), 1, "it is stale");
        assert!(
            !index_only.needs_reanalysis(),
            "re-analyzing it would change nothing, so it must not be planned: \
             {index_only:?}"
        );
        assert_eq!(
            index_only.unrepairable_columns().count(),
            1,
            "and it must still be reported"
        );
    }
}
