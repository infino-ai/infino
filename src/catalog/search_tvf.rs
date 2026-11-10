// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Catalog-level search table-valued functions.
//!
//! The single-table search TVFs (`bm25_search` / `bm25_search_prefix` /
//! `vector_search` / `hybrid_search` / `token_match` / `exact_match`)
//! capture one supertable reader and take column-first arguments — they
//! bind to the lone `FROM supertable` provider. Under the catalog,
//! several tables share one session, so the TVF can't know which table
//! it targets from a bare column.
//!
//! These adapters add a **leading table-name argument**
//! (`bm25_search('users', 'body', 'q', 10)`): each resolves that table's
//! reader through the [`Connection`] at call time, then delegates the
//! remaining (column-first) arguments to the existing single-table
//! function. The table name is a catalog table (a string literal), not a
//! `FROM` alias — the TVF is a relation source, so joins / self-joins
//! compose on its output.

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

use arrow_schema::SchemaRef;
use datafusion::{
    catalog::{TableFunctionArgs, TableFunctionImpl, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::context::SessionContext,
    logical_expr::Expr,
};

use super::Connection;
#[cfg(feature = "graph-index")]
use crate::supertable::query::exec::graph_exec::{GraphFunc, Traversal};
use crate::{
    InfinoError,
    runtime_metrics::op_stats::{self, OpStatsCollector},
    supertable::{
        error::QueryError,
        handle::SupertableReader,
        query::exec::{
            common::{Arity, arg_to_string},
            fts_exec::{
                BM25_PREFIX_SEARCH_ARITY, BM25_PREFIX_UDTF, BM25_SEARCH_ARITY, BM25_SEARCH_UDTF,
                Bm25PrefixFunc, Bm25SearchFunc,
            },
            hybrid_exec::{HYBRID_SEARCH_ARITY, HYBRID_SEARCH_UDTF, HybridSearchFunc},
            match_exec::{
                EXACT_MATCH_ARITY, EXACT_MATCH_UDTF, ExactMatchFunc, TOKEN_MATCH_ARITY,
                TOKEN_MATCH_UDTF, TokenMatchFunc,
            },
            vector_exec::{VECTOR_SEARCH_ARITY, VECTOR_SEARCH_UDTF, VectorSearchFunc},
        },
        schema::sql_schema,
    },
};

/// Why the table a search names could not be opened, as a DataFusion error that
/// keeps its cause: a name that resolves to nothing is the caller's mistake (a
/// plan error); a storage or credential failure while opening it is not, and
/// crosses the plan typed.
fn open_table_error(name: &str, e: InfinoError) -> DataFusionError {
    let opening = |detail: &str| format!("opening table {name:?} for a search: {detail}");
    match e {
        InfinoError::NotFound(detail) => {
            DataFusionError::Plan(format!("search over unknown table {name:?}: {detail}"))
        }
        InfinoError::Io(detail) => QueryError::Store(opening(&detail)).into(),
        InfinoError::PermissionDenied(detail) => {
            QueryError::PermissionDenied(opening(&detail)).into()
        }
        other => QueryError::Internal(opening(&other.to_string())).into(),
    }
}

/// A resolved table's pinned snapshot: the reader the search kernels run
/// against plus its scalar schema (the TVF's output columns).
#[derive(Clone)]
struct ResolvedTable {
    reader: Arc<SupertableReader>,
    scalar_schema: SchemaRef,
}

/// Opens catalog tables by name (once per query) for the search TVFs.
/// Shared across the four TVF adapters registered for one query.
struct TableResolver {
    conn: Connection,
    cache: Mutex<HashMap<String, ResolvedTable>>,
    /// Per-query work collector, captured when the TVFs are registered —
    /// registration happens per query on the caller's thread inside
    /// `query_sql`, while resolution runs later on runtime threads where
    /// the scope's thread-local is invisible.
    op_stats: Option<Arc<OpStatsCollector>>,
}

impl fmt::Debug for TableResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableResolver").finish_non_exhaustive()
    }
}

impl TableResolver {
    fn new(conn: Connection) -> Self {
        Self {
            conn,
            cache: Mutex::new(HashMap::new()),
            op_stats: op_stats::current(),
        }
    }

    /// Resolve `name` to its pinned snapshot, opening + caching it on
    /// first use within this query.
    fn resolve(&self, name: &str) -> DfResult<ResolvedTable> {
        if let Some(t) = self
            .cache
            .lock()
            .expect("resolver cache poisoned")
            .get(name)
        {
            return Ok(t.clone());
        }
        let table = self
            .conn
            .open_table_handle(name)
            .map_err(|e| open_table_error(name, e))?;
        // `reader()` applies the read-consistency freshness check itself (and,
        // under Strong, fails rather than serving a stale snapshot), so there
        // is no separate `ensure_fresh` call here. A failure is classified
        // like the open's, so a table purged in between is an unknown table
        // here too, not a separate `NotFound`.
        let mut reader = table
            .reader()
            .map_err(|e| open_table_error(name, InfinoError::from(e)))?;
        // The mint ran on a runtime thread where the scope's thread-local
        // is invisible; hand it the collector captured at registration.
        reader.op_stats = self.op_stats.clone();
        let resolved = ResolvedTable {
            // Stored shape: search TVF output columns resolve against
            // what Parquet actually holds (index-only FTS columns are
            // not projectable). Built from this reader's own snapshot, so a
            // schema change landing after `reader()` cannot name columns the
            // reader lacks, and with field ids kept out of the plan.
            scalar_schema: sql_schema(reader.manifest().stored_schema()),
            reader: Arc::new(reader),
        };
        self.cache
            .lock()
            .expect("resolver cache poisoned")
            .insert(name.to_string(), resolved.clone());
        Ok(resolved)
    }

    /// Pop the leading table-name argument and resolve it, returning the
    /// snapshot + the remaining (column-first) args for the inner TVF.
    fn split_leading<'a>(
        &self,
        args: &'a [Expr],
        fn_name: &str,
    ) -> DfResult<(ResolvedTable, &'a [Expr])> {
        let first = args.first().ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{fn_name} expects a leading table-name argument: \
                 {fn_name}('table', ...)"
            ))
        })?;
        let name = arg_to_string(first, &format!("{fn_name} table"))?;
        let resolved = self.resolve(&name)?;
        Ok((resolved, &args[1..]))
    }

    /// [`split_leading`](Self::split_leading), then the remaining arguments
    /// checked against `arity` in the form the caller wrote: with the table
    /// counted. Left to the single-table function, the count would leave the
    /// table out ("expects 5 arguments (text_col, ...), got 4" for a call
    /// that passed five), which reads as if the leading table were the
    /// mistake. The refusal names the table's indexed columns, the
    /// arguments a wrong count most often left out.
    fn split_checked<'a>(
        &self,
        args: &'a [Expr],
        fn_name: &str,
        arity: Arity,
    ) -> DfResult<(ResolvedTable, &'a [Expr])> {
        let (table, rest) = self.split_leading(args, fn_name)?;
        if arity.takes(rest.len()) {
            return Ok((table, rest));
        }
        let manifest = table.reader.manifest();
        let fts: Vec<String> = manifest
            .fts_configs()
            .into_iter()
            .map(|c| c.column)
            .collect();
        let vectors: Vec<String> = manifest
            .vector_configs()
            .into_iter()
            .map(|c| c.column)
            .collect();
        let listed = |columns: &[String]| {
            if columns.is_empty() {
                "none".to_string()
            } else {
                columns.join(", ")
            }
        };
        // `args[0]` is the table name `split_leading` just read.
        let table_name = arg_to_string(&args[0], &format!("{fn_name} table"))?;
        Err(DataFusionError::Plan(format!(
            "{fn_name} expects {} arguments (table, {}), got {}; table {table_name:?} has \
             full-text columns: {}; vector columns: {}",
            arity.count(1),
            arity.arguments,
            args.len(),
            listed(&fts),
            listed(&vectors),
        )))
    }
}

/// Register the catalog search TVFs (table-name-first form) on `ctx`,
/// resolving tables through `conn`.
pub(crate) fn register_search_tvfs(ctx: &SessionContext, conn: Connection) {
    let resolver = Arc::new(TableResolver::new(conn));
    ctx.register_udtf(
        BM25_SEARCH_UDTF,
        Arc::new(Bm25SearchCatalogFunc {
            resolver: Arc::clone(&resolver),
        }),
    );
    ctx.register_udtf(
        BM25_PREFIX_UDTF,
        Arc::new(Bm25PrefixCatalogFunc {
            resolver: Arc::clone(&resolver),
        }),
    );
    ctx.register_udtf(
        VECTOR_SEARCH_UDTF,
        Arc::new(VectorSearchCatalogFunc {
            resolver: Arc::clone(&resolver),
        }),
    );
    ctx.register_udtf(
        TOKEN_MATCH_UDTF,
        Arc::new(TokenMatchCatalogFunc {
            resolver: Arc::clone(&resolver),
        }),
    );
    ctx.register_udtf(
        EXACT_MATCH_UDTF,
        Arc::new(ExactMatchCatalogFunc {
            resolver: Arc::clone(&resolver),
        }),
    );
    ctx.register_udtf(
        HYBRID_SEARCH_UDTF,
        Arc::new(HybridSearchCatalogFunc { resolver }),
    );
}

/// Register `graph_walk` / `graph_rank` on `ctx`: the rows of a table of
/// `tables` within some hops of seed rows, walked over the edge table of
/// `graph`. Without `edge_table` the functions take the edge table's name
/// first (`graph_walk('edges', table, seed_table, seed_ids, hops, k)`),
/// like every search function names its table; with it they take
/// `(table, seed_table, seed_ids, hops, k)` over that one edge table — the
/// form a graph attached from another catalog is served in
/// (`Connection::attach_graph`), where the edge table is not one the
/// statement could name. Under one catalog `graph` and `tables` are the
/// same connection.
#[cfg(feature = "graph-index")]
pub(crate) fn register_graph_tvfs(
    ctx: &SessionContext,
    graph: Connection,
    tables: Connection,
    edge_table: Option<String>,
) {
    let graph = Arc::new(TableResolver::new(graph));
    let tables = Arc::new(TableResolver::new(tables));
    for traversal in [Traversal::Walk, Traversal::Rank] {
        ctx.register_udtf(
            traversal.name(),
            Arc::new(GraphCatalogFunc {
                graph: Arc::clone(&graph),
                tables: Arc::clone(&tables),
                edge_table: edge_table.clone(),
                traversal,
            }),
        );
    }
}

#[cfg(feature = "graph-index")]
#[derive(Debug)]
struct GraphCatalogFunc {
    /// Resolves the edge table.
    graph: Arc<TableResolver>,
    /// Resolves the table whose rows a walk returns.
    tables: Arc<TableResolver>,
    /// The edge table, when fixed at registration; else the leading argument.
    edge_table: Option<String>,
    traversal: Traversal,
}

#[cfg(feature = "graph-index")]
impl TableFunctionImpl for GraphCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let name = self.traversal.name();
        let (edges, rest) = match &self.edge_table {
            Some(edge_table) => (self.graph.resolve(edge_table)?, args.exprs()),
            None => self.graph.split_leading(args.exprs(), name)?,
        };
        let first = rest.first().ok_or_else(|| {
            DataFusionError::Plan(format!(
                "{name} expects the table whose rows to return: {name}('table', ...)"
            ))
        })?;
        let target_name = arg_to_string(first, &format!("{name} table"))?;
        let target = self.tables.resolve(&target_name)?;

        GraphFunc::new(
            edges.reader,
            target.reader,
            target_name,
            target.scalar_schema,
            self.traversal,
        )
        .call_with_args(TableFunctionArgs::new(&rest[1..], args.session()))
    }
}

#[derive(Debug)]
struct Bm25SearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for Bm25SearchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) =
            self.resolver
                .split_checked(args.exprs(), BM25_SEARCH_UDTF, BM25_SEARCH_ARITY)?;

        Bm25SearchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct Bm25PrefixCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for Bm25PrefixCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_checked(
            args.exprs(),
            BM25_PREFIX_UDTF,
            BM25_PREFIX_SEARCH_ARITY,
        )?;

        Bm25PrefixFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct VectorSearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for VectorSearchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) =
            self.resolver
                .split_checked(args.exprs(), VECTOR_SEARCH_UDTF, VECTOR_SEARCH_ARITY)?;

        VectorSearchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct HybridSearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for HybridSearchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) =
            self.resolver
                .split_checked(args.exprs(), HYBRID_SEARCH_UDTF, HYBRID_SEARCH_ARITY)?;

        HybridSearchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct TokenMatchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for TokenMatchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) =
            self.resolver
                .split_checked(args.exprs(), TOKEN_MATCH_UDTF, TOKEN_MATCH_ARITY)?;

        TokenMatchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct ExactMatchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for ExactMatchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) =
            self.resolver
                .split_checked(args.exprs(), EXACT_MATCH_UDTF, EXACT_MATCH_ARITY)?;

        ExactMatchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::datafusion_error;

    /// Only a table name that resolves to nothing is the caller's mistake. A
    /// table that exists but failed to open is ours, and keeps its cause.
    #[test]
    fn a_search_over_a_table_that_failed_to_open_keeps_its_cause() {
        let public = |e| datafusion_error(&open_table_error("docs", e));
        assert!(matches!(
            public(InfinoError::NotFound("docs".into())),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            public(InfinoError::Io("bucket timed out".into())),
            InfinoError::Io(_)
        ));
        assert!(matches!(
            public(InfinoError::PermissionDenied("refused".into())),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            public(InfinoError::Backend("manifest list unreadable".into())),
            InfinoError::Backend(_)
        ));
    }
}
