// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`RemoteTable`] — a table handle backed by the hosted service.
//!
//! Implements the internal [`Table`] trait by forwarding each operation over
//! the wire through its [`RemoteCatalog`]. Every query/mutation is supported;
//! `optimize` and `gc` are deliberately not — on a hosted table those are the
//! platform's (server-side) concern, so they report that rather than forwarding.

#[cfg(any(test, feature = "test-helpers"))]
use std::any::Any;
use std::{sync::Arc, time::Duration};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::{prelude::Expr, sql::unparser::expr_to_sql};
use serde_json::{Value, json};

use super::{RemoteCatalog, read_arrow, read_json, wire};
use crate::{
    Bm25SearchOptions, Bm25Stats, BoolMode, GcError, GcReport, InfinoError, MutationStats,
    OptimizeError, OptimizeOptions, VectorFilter, catalog::table::Table,
    superfile::VectorSearchOptions,
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

/// Wire spelling for the BM25 statistics mode
/// (`"per_superfile"` / `"global"`).
fn stats_str(stats: Bm25Stats) -> &'static str {
    match stats {
        Bm25Stats::PerSuperfile => "per_superfile",
        Bm25Stats::Global => "global",
    }
}

/// Render a mutation predicate to SQL for the wire. The public API takes a
/// DataFusion `Expr`, but the endpoint takes a SQL string (the server parses it
/// back against the table's schema), so unparse it here.
fn predicate_to_sql(predicate: &Expr) -> Result<String, InfinoError> {
    expr_to_sql(predicate)
        .map(|sql| sql.to_string())
        .map_err(|e| {
            InfinoError::Query(format!(
                "cannot express this predicate over the remote transport: {e}"
            ))
        })
}

/// Refuse test-only vector tuning overrides on the remote transport. The
/// public search methods always pass the default options, so this can only
/// fire in a `test-helpers` build using the `_with_options` variants.
fn reject_remote_overrides(opts: &VectorSearchOptions) -> Result<(), InfinoError> {
    if opts.nprobe.is_some() || opts.rerank_mult().is_some() {
        return Err(InfinoError::Query(
            "vector tuning overrides are not supported over the remote transport; \
             hosted tables serve engine-decided settings"
                .to_string(),
        ));
    }
    Ok(())
}

/// Parse an update/delete response (`{matched, n_tombstoned, n_not_found}`)
/// into [`MutationStats`].
fn parse_mutation_stats(op: &str, response: ureq::Response) -> Result<MutationStats, InfinoError> {
    let value = read_json(op, response)?;
    let field = |name: &str| -> Result<usize, InfinoError> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .ok_or_else(|| InfinoError::Backend(format!("{op} response missing `{name}`")))
    };
    Ok(MutationStats::from_remote(
        field("matched")?,
        field("n_tombstoned")?,
        field("n_not_found")?,
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

    fn update(&self, predicate: Expr, batch: &RecordBatch) -> Result<MutationStats, InfinoError> {
        let predicate = predicate_to_sql(&predicate)?;
        let body = wire::batches_to_ipc(std::slice::from_ref(batch))?;
        let response = self.catalog.post_arrow(
            "update",
            &[
                ("table", self.table_name.as_str()),
                ("predicate", &predicate),
            ],
            body,
        )?;
        parse_mutation_stats("update", response)
    }

    fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        let predicate = predicate_to_sql(&predicate)?;
        let response = self.catalog.post_arrow(
            "delete",
            &[
                ("table", self.table_name.as_str()),
                ("predicate", &predicate),
            ],
            Vec::new(),
        )?;
        parse_mutation_stats("delete", response)
    }

    fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "k": k,
            "mode": mode_str(opts.mode),
            "stats": stats_str(opts.stats),
            "projection": projection,
        });
        let response = self.catalog.post_json("bm25_search", body)?;
        read_arrow("bm25_search", response)
    }

    fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "mode": mode_str(mode),
            "projection": projection,
        });
        let response = self.catalog.post_json("token_match", body)?;
        read_arrow("token_match", response)
    }

    fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "value": value,
            "projection": projection,
        });
        let response = self.catalog.post_json("exact_match", body)?;
        read_arrow("exact_match", response)
    }

    fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        let body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "mode": mode_str(mode),
        });
        let response = self.catalog.post_json("count", body)?;
        read_json("count", response)?
            .get("count")
            .and_then(Value::as_u64)
            .ok_or_else(|| InfinoError::Backend("count response missing `count`".to_string()))
    }

    fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        opts: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        // Options never cross the wire: the public path always passes the
        // default (serving is engine-decided), and the hosted API carries
        // no tuning fields. A diagnostics-build caller reaching for the
        // test-only overrides against a hosted table hears "no" loudly —
        // silently serving defaults would corrupt whatever sweep or oracle
        // measurement asked for the override.
        reject_remote_overrides(&opts)?;
        let mut body = json!({
            "table_name": self.table_name,
            "field_name": column,
            "query": query,
            "k": k,
            "projection": projection,
        });
        if let Some(filter) = filter {
            body["filter"] = json!({
                "field_name": filter.column,
                "query": filter.query,
                "mode": mode_str(filter.mode),
            });
        }
        let response = self.catalog.post_json("vector_search", body)?;
        read_arrow("vector_search", response)
    }

    #[allow(clippy::too_many_arguments)]
    fn hybrid_search(
        &self,
        text_column: &str,
        text_query: &str,
        mode: BoolMode,
        vector_column: &str,
        vector_query: &[f32],
        opts: VectorSearchOptions,
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        // See `vector_search`: overrides never cross the wire; loud, not silent.
        reject_remote_overrides(&opts)?;
        let body = json!({
            "table_name": self.table_name,
            "text_field": text_column,
            "text_query": text_query,
            "mode": mode_str(mode),
            "vector_field": vector_column,
            "vector_query": vector_query,
            "k": k,
            "projection": projection,
        });
        let response = self.catalog.post_json("hybrid_search", body)?;
        read_arrow("hybrid_search", response)
    }

    fn optimize(&self, _opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        // Compaction on a hosted table is the platform optimizer's job, not a
        // client's; it is deliberately not exposed over the remote transport.
        Err(OptimizeError::NoStorage)
    }

    fn gc(&self, _safety_gap: Duration) -> Result<GcReport, GcError> {
        // Retention/GC on a hosted table is server-side; not a client operation.
        Err(GcError::NoStorage)
    }

    #[cfg(any(test, feature = "test-helpers"))]
    fn as_any(&self) -> &dyn Any {
        self
    }
}
