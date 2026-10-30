// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`TableSnapshot`] — one version of a table, held for reading.

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;

use crate::{
    Bm25SearchOptions, BoolMode, InfinoError,
    runtime_metrics::op_stats,
    supertable::{Supertable as SupertableHandle, handle::SupertableReader},
};

/// One version of a table, held for reading: the rows the table held when
/// the snapshot was taken, whatever is appended, updated, deleted or
/// compacted after. A scan that pages through a table over many calls reads
/// one snapshot, so every page agrees with the others.
///
/// Take one with [`Connection::snapshot`](crate::Connection::snapshot),
/// search it with the methods here, and name it in SQL through
/// [`Connection::query_sql_at`](crate::Connection::query_sql_at). Cheap to
/// clone; clones share one snapshot.
///
/// A snapshot reads the files of its version from the table's storage, so it
/// stays readable while garbage collection keeps them: until they have been
/// out of the table's newest version for longer than the gc safety gap.
/// Taking it reads the deletes its version records that this process has not
/// read yet, so it costs a read per data file with deletes the first time.
///
/// Vector and hybrid search are not offered on a snapshot: the vector index
/// is maintained beside the table and is not held with it.
#[derive(Clone)]
pub struct TableSnapshot {
    name: String,
    table: SupertableHandle,
    reader: Arc<SupertableReader>,
}

impl TableSnapshot {
    /// Take a snapshot of `table`, the catalog table `name`.
    pub(crate) fn take(name: &str, table: SupertableHandle) -> Result<Self, InfinoError> {
        let reader = table
            .snapshot_reader()
            .map_err(|e| e.with_context("snapshot", Some(name)))?;
        Ok(Self {
            name: name.to_string(),
            table,
            reader: Arc::new(reader),
        })
    }

    /// The name of the table the snapshot was taken of.
    pub fn table_name(&self) -> &str {
        &self.name
    }

    /// The version of the table the snapshot holds. Two snapshots of a table
    /// that nothing was committed to in between hold the same version.
    pub fn version(&self) -> u64 {
        self.reader.manifest().manifest_id
    }

    /// The table's Arrow schema at this version.
    pub fn schema(&self) -> SchemaRef {
        self.reader.manifest().user_schema()
    }

    /// [`Supertable::bm25_search`](crate::Supertable::bm25_search) over this
    /// version.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let close_out = self.table.close_out();
        let batches = self
            .query_reader()
            .bm25_search(column, query, k, opts, projection)
            .map_err(|e| InfinoError::from(e).with_context("bm25_search", Some(&self.name)))?;
        close_out.finish_batches(&batches);
        Ok(batches)
    }

    /// [`Supertable::token_match`](crate::Supertable::token_match) over this
    /// version.
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let close_out = self.table.close_out();
        let batch =
            self.table
                .token_match_rows(&self.query_reader(), column, query, mode, projection)?;
        close_out.finish(batch.num_rows() as u64);
        Ok(vec![batch])
    }

    /// [`Supertable::exact_match`](crate::Supertable::exact_match) over this
    /// version.
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let close_out = self.table.close_out();
        let batch = self
            .table
            .exact_match_rows(&self.query_reader(), column, value, projection)?;
        close_out.finish(batch.num_rows() as u64);
        Ok(vec![batch])
    }

    /// [`Supertable::count`](crate::Supertable::count) over this version.
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        self.query_reader()
            .count(column, query, mode)
            .map_err(|e| InfinoError::from(e).with_context("count", Some(&self.name)))
    }

    /// The table handle the snapshot was taken from, whose runtime its
    /// queries run on.
    pub(crate) fn table(&self) -> &SupertableHandle {
        &self.table
    }

    /// The snapshot's reader for one query, counting its work into the
    /// calling thread's per-query collector.
    pub(crate) fn query_reader(&self) -> SupertableReader {
        let mut reader = (*self.reader).clone();
        reader.op_stats = op_stats::current();
        reader
    }
}
