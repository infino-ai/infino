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
//!
//! With a [`SearchReranker`] on the connection, `bm25_search` and
//! `hybrid_search` are called for the reranker's pool and wrapped so the
//! statement gets the `k` rows the reranker chooses (see [`super::rerank`]).

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
    prelude::lit,
};

use super::{
    Connection,
    rerank::{RerankedSearch, SearchReranker},
};
#[cfg(feature = "graph-index")]
use crate::supertable::query::exec::graph_exec::{GraphFunc, Traversal};
use crate::{
    runtime_metrics::op_stats::{self, OpStatsCollector},
    supertable::{
        handle::SupertableReader,
        query::exec::{
            common::{arg_to_string, arg_to_usize},
            fts_exec::{BM25_PREFIX_UDTF, BM25_SEARCH_UDTF, Bm25PrefixFunc, Bm25SearchFunc},
            hybrid_exec::{HYBRID_SEARCH_UDTF, HybridSearchFunc},
            match_exec::{EXACT_MATCH_UDTF, ExactMatchFunc, TOKEN_MATCH_UDTF, TokenMatchFunc},
            vector_exec::{VECTOR_SEARCH_UDTF, VectorSearchFunc},
        },
    },
};

/// Where `bm25_search(table, column, query, k[, mode])` carries its query
/// text and its `k`, in the arguments after the table name.
const BM25_QUERY_ARG: usize = 1;
const BM25_K_ARG: usize = 2;
/// Where `hybrid_search(table, text_col, q_text, vec_col, q_vec, k)` carries
/// its text query and its `k`, in the arguments after the table name.
const HYBRID_QUERY_ARG: usize = 1;
const HYBRID_K_ARG: usize = 4;

/// A ranked search's arguments widened to the reranker's pool: the query
/// text and the statement's `k` read off `rest`, and `rest` with its `k`
/// replaced by the pool. `None` when the arguments do not reach `k`, for
/// the function's own count check to refuse.
fn widened_for_rerank(
    rest: &[Expr],
    query_arg: usize,
    k_arg: usize,
    reranker: &dyn SearchReranker,
    fn_name: &str,
) -> DfResult<Option<(String, usize, Vec<Expr>)>> {
    if rest.len() <= k_arg {
        return Ok(None);
    }
    let query_text = arg_to_string(&rest[query_arg], &format!("{fn_name} query"))?;
    let k = arg_to_usize(&rest[k_arg], &format!("{fn_name} k"))?;
    let pool = reranker.pool(k).max(k);
    let pool = i64::try_from(pool)
        .map_err(|_| DataFusionError::Plan(format!("{fn_name}: the reranker's pool {pool} is too wide")))?;
    let mut widened = rest.to_vec();
    widened[k_arg] = lit(pool);
    Ok(Some((query_text, k, widened)))
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
    /// The host's order on a ranked search's rows, when the connection
    /// carries one at registration.
    reranker: Option<Arc<dyn SearchReranker>>,
}

impl fmt::Debug for TableResolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TableResolver").finish_non_exhaustive()
    }
}

impl TableResolver {
    fn new(conn: Connection, reranker: Option<Arc<dyn SearchReranker>>) -> Self {
        Self {
            conn,
            cache: Mutex::new(HashMap::new()),
            op_stats: op_stats::current(),
            reranker,
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
        let table = self.conn.open_table_handle(name).map_err(|e| {
            DataFusionError::Plan(format!("search over unknown table {name:?}: {e}"))
        })?;
        // `reader()` applies the read-consistency freshness check itself (and,
        // under Strong, fails rather than serving a stale snapshot), so there
        // is no separate `ensure_fresh` call here.
        let mut reader = table.reader().map_err(|e| {
            DataFusionError::Plan(format!("search over unknown table {name:?}: {e}"))
        })?;
        // The mint ran on a runtime thread where the scope's thread-local
        // is invisible; hand it the collector captured at registration.
        reader.op_stats = self.op_stats.clone();
        let resolved = ResolvedTable {
            reader: Arc::new(reader),
            // Stored shape: search TVF output columns resolve against
            // what Parquet actually holds (index-only FTS columns are
            // not projectable).
            scalar_schema: table.options().stored_schema(),
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
}

/// Register the catalog search TVFs (table-name-first form) on `ctx`,
/// resolving tables through `conn`, the ranked ones under `reranker`'s
/// order when the connection carries one.
pub(crate) fn register_search_tvfs(
    ctx: &SessionContext,
    conn: Connection,
    reranker: Option<Arc<dyn SearchReranker>>,
) {
    let resolver = Arc::new(TableResolver::new(conn, reranker));
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
    let graph = Arc::new(TableResolver::new(graph, None));
    let tables = Arc::new(TableResolver::new(tables, None));
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
        let (t, rest) = self.resolver.split_leading(args.exprs(), "bm25_search")?;
        let func = Bm25SearchFunc::new(t.reader, t.scalar_schema);
        if let Some(reranker) = &self.resolver.reranker
            && let Some((query_text, k, widened)) = widened_for_rerank(
                rest,
                BM25_QUERY_ARG,
                BM25_K_ARG,
                reranker.as_ref(),
                "bm25_search",
            )?
        {
            let table = arg_to_string(&args.exprs()[0], "bm25_search table")?;
            let inner = func.call_with_args(TableFunctionArgs::new(&widened, args.session()))?;
            return Ok(Arc::new(RerankedSearch {
                inner,
                reranker: Arc::clone(reranker),
                table,
                query_text,
                k,
            }));
        }
        func.call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct Bm25PrefixCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for Bm25PrefixCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self
            .resolver
            .split_leading(args.exprs(), "bm25_search_prefix")?;

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
        let (t, rest) = self.resolver.split_leading(args.exprs(), "vector_search")?;

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
        let (t, rest) = self.resolver.split_leading(args.exprs(), "hybrid_search")?;
        let func = HybridSearchFunc::new(t.reader, t.scalar_schema);
        if let Some(reranker) = &self.resolver.reranker
            && let Some((query_text, k, widened)) = widened_for_rerank(
                rest,
                HYBRID_QUERY_ARG,
                HYBRID_K_ARG,
                reranker.as_ref(),
                "hybrid_search",
            )?
        {
            let table = arg_to_string(&args.exprs()[0], "hybrid_search table")?;
            let inner = func.call_with_args(TableFunctionArgs::new(&widened, args.session()))?;
            return Ok(Arc::new(RerankedSearch {
                inner,
                reranker: Arc::clone(reranker),
                table,
                query_text,
                k,
            }));
        }
        func.call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}

#[derive(Debug)]
struct TokenMatchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for TokenMatchCatalogFunc {
    fn call_with_args(&self, args: TableFunctionArgs) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args.exprs(), "token_match")?;

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
        let (t, rest) = self.resolver.split_leading(args.exprs(), "exact_match")?;

        ExactMatchFunc::new(t.reader, t.scalar_schema)
            .call_with_args(TableFunctionArgs::new(rest, args.session()))
    }
}
