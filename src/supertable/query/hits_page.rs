// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! A page of rows found as hits without planning a statement, as
//! [`Supertable::sorted_search`] and [`Supertable::head`] answer it: the
//! page size held to the rows the table holds, the hits found, and only
//! those resolved to rows.

use arrow::record_batch::RecordBatch;
use tracing::Instrument;

use crate::{
    InfinoError,
    supertable::{
        error::QueryError,
        handle::{Supertable, SupertableReader},
        query::{SuperfileHit, exec::common::resolve_hits_named},
        reader_cache::disk::ForegroundQueryGuard,
    },
    utils::trace::detail_span,
};

impl SupertableReader {
    /// The rows this snapshot's superfiles hold, deleted ones included: no
    /// page can hold more.
    pub(crate) fn rows_held(&self) -> u64 {
        self.manifest().superfiles.iter().map(|e| e.n_docs).sum()
    }
}

/// What a page's `find` answers: hits to resolve, or rows it read some
/// other way (a statement, where the hits cannot be found).
pub(crate) enum Page {
    /// The page's rows as hits, in page order.
    Hits(Vec<SuperfileHit>),
    /// The page's rows, already read.
    Rows(Vec<RecordBatch>),
}

impl Supertable {
    /// A page of at most `k` rows, found by `find` and, when it answers
    /// hits, resolved to `projection`'s columns, which follow
    /// [`Supertable::bm25_search`]'s rules. `k` is first held to the rows
    /// the table holds, so a page size past them reserves nothing for rows
    /// that cannot come back; `find` gets the held `k` and runs as a
    /// foreground query. Errors name `call`.
    pub(crate) fn hits_page(
        &self,
        call: &'static str,
        k: usize,
        projection: Option<&[&str]>,
        find: impl FnOnce(&SupertableReader, usize) -> Result<Page, QueryError>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let close_out = self.close_out();
        let reader = self.reader()?;
        let context = |e: QueryError| InfinoError::from(e).with_context(call, None);
        reader.check_projection(projection).map_err(context)?;
        let k = k.min(usize::try_from(reader.rows_held()).unwrap_or(usize::MAX));
        let page = {
            let _foreground = ForegroundQueryGuard::enter();
            find(&reader, k).map_err(context)?
        };
        let batches = match page {
            Page::Hits(hits) => vec![
                self.block_on_query(
                    resolve_hits_named(&reader, &hits, projection)
                        .instrument(detail_span!("search.resolve", hits = hits.len())),
                )
                .map_err(context)?,
            ],
            Page::Rows(batches) => batches,
        };
        close_out.finish_batches(&batches);
        Ok(batches)
    }
}
