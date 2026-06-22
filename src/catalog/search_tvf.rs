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
    catalog::{TableFunctionImpl, TableProvider},
    error::{DataFusionError, Result as DfResult},
    execution::context::SessionContext,
    logical_expr::Expr,
};

use super::Connection;
use crate::supertable::{
    handle::SupertableReader,
    query::exec::{
        common::arg_to_string,
        fts_exec::{BM25_PREFIX_UDTF, BM25_SEARCH_UDTF, Bm25PrefixFunc, Bm25SearchFunc},
        hybrid_exec::{HYBRID_SEARCH_UDTF, HybridSearchFunc},
        match_exec::{EXACT_MATCH_UDTF, ExactMatchFunc, TOKEN_MATCH_UDTF, TokenMatchFunc},
        vector_exec::{VECTOR_SEARCH_UDTF, VectorSearchFunc},
    },
};

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
        let table = self.conn.open_table(name).map_err(|e| {
            DataFusionError::Plan(format!("search over unknown table {name:?}: {e}"))
        })?;
        table.ensure_fresh();
        let resolved = ResolvedTable {
            reader: Arc::new(table.reader()),
            scalar_schema: table.options().scalar_schema(),
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

#[derive(Debug)]
struct Bm25SearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for Bm25SearchCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "bm25_search")?;
        Bm25SearchFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}

#[derive(Debug)]
struct Bm25PrefixCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for Bm25PrefixCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "bm25_search_prefix")?;
        Bm25PrefixFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}

#[derive(Debug)]
struct VectorSearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for VectorSearchCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "vector_search")?;
        VectorSearchFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}

#[derive(Debug)]
struct HybridSearchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for HybridSearchCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "hybrid_search")?;
        HybridSearchFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}

#[derive(Debug)]
struct TokenMatchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for TokenMatchCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "token_match")?;
        TokenMatchFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}

#[derive(Debug)]
struct ExactMatchCatalogFunc {
    resolver: Arc<TableResolver>,
}
impl TableFunctionImpl for ExactMatchCatalogFunc {
    fn call(&self, args: &[Expr]) -> DfResult<Arc<dyn TableProvider>> {
        let (t, rest) = self.resolver.split_leading(args, "exact_match")?;
        ExactMatchFunc::new(t.reader, t.scalar_schema).call(rest)
    }
}
