// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RemoteTable`] — a table handle backed by the hosted service.
//!
//! Implements the internal [`Table`] trait by forwarding each operation over
//! the wire through its [`RemoteCatalog`]. The vertical slice (`append`,
//! `bm25_search`, `schema`) is wired; the remaining operations return a clear
//! error until they are added.

#[cfg(any(test, feature = "test-helpers"))]
use std::any::Any;
use std::{sync::Arc, time::Duration};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::prelude::Expr;
use serde_json::json;

use super::{RemoteCatalog, read_arrow, wire};
use crate::{
    BoolMode, GcError, GcReport, InfinoError, MutationStats, OptimizeError, OptimizeOptions,
    VectorFilter, VectorSearchOptions, catalog::table::Table,
};

/// A hosted table handle. Holds its `RemoteCatalog`, the table name, and the
/// schema fetched at create/open (so `schema()` is infallible, matching the
/// local handle).
pub(crate) struct RemoteTable {
    catalog: Arc<RemoteCatalog>,
    table_name: String,
    schema: SchemaRef,
}

impl RemoteTable {
    pub(crate) fn new(catalog: Arc<RemoteCatalog>, table_name: String, schema: SchemaRef) -> Self {
        Self {
            catalog,
            table_name,
            schema,
        }
    }
}

/// Wire spelling for a boolean search mode (`"and"` / `"or"`).
fn mode_str(mode: BoolMode) -> &'static str {
    match mode {
        BoolMode::And => "and",
        BoolMode::Or => "or",
    }
}

/// Error for an operation not yet wired over the remote transport.
fn unsupported(op: &str) -> InfinoError {
    InfinoError::Backend(format!(
        "{op} is not yet supported over the remote transport"
    ))
}

impl Table for RemoteTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        let body = wire::batches_to_ipc(std::slice::from_ref(batch))?;
        self.catalog
            .post_arrow("append", &[("table", self.table_name.as_str())], body)?;
        Ok(())
    }

    fn update(&self, _predicate: Expr, _batch: &RecordBatch) -> Result<MutationStats, InfinoError> {
        Err(unsupported("update"))
    }

    fn delete(&self, _predicate: Expr) -> Result<MutationStats, InfinoError> {
        Err(unsupported("delete"))
    }

    fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "k": k,
            "mode": mode_str(mode),
            "projection": projection,
        });
        let response = self.catalog.post_json("bm25_search", body)?;
        read_arrow("bm25_search", response)
    }

    fn token_match(
        &self,
        _column: &str,
        _query: &str,
        _mode: BoolMode,
        _projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        Err(unsupported("token_match"))
    }

    fn exact_match(
        &self,
        _column: &str,
        _value: &str,
        _projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        Err(unsupported("exact_match"))
    }

    fn count(&self, _column: &str, _query: &str, _mode: BoolMode) -> Result<u64, InfinoError> {
        Err(unsupported("count"))
    }

    fn vector_search(
        &self,
        _column: &str,
        _query: &[f32],
        _k: usize,
        _opts: VectorSearchOptions,
        _filter: Option<VectorFilter<'_>>,
        _projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        Err(unsupported("vector_search"))
    }

    fn hybrid_search(
        &self,
        _text_column: &str,
        _text_query: &str,
        _mode: BoolMode,
        _vector_column: &str,
        _vector_query: &[f32],
        _opts: VectorSearchOptions,
        _k: usize,
        _projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        Err(unsupported("hybrid_search"))
    }

    fn optimize(&self, _opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        // Compaction is a server-side (optimizer) concern for a hosted table;
        // there is no local storage to optimize from the client.
        Err(OptimizeError::NoStorage)
    }

    fn gc(&self, _safety_gap: Duration) -> Result<GcReport, GcError> {
        // Retention is server-side for a hosted table; no local storage here.
        Err(GcError::NoStorage)
    }

    #[cfg(any(test, feature = "test-helpers"))]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
