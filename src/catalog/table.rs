// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! [`Supertable`] — the public table handle.
//!
//! The handle is a thin wrapper over an `Arc<dyn Table>`, so one public type
//! serves both a local (embedded) table and a hosted (remote) one: the
//! connection target picks the implementation at `connect` time and everything
//! above this seam calls the same methods. The local implementation is the
//! engine's own table handle; a hosted implementation forwards each operation
//! over the wire. The [`Table`] trait is the shared operation surface.

#[cfg(any(test, feature = "test-helpers"))]
use std::any::Any;
use std::{fmt, sync::Arc, time::Duration};

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use datafusion::prelude::Expr;

use crate::{
    Bm25SearchOptions, BoolMode, GcError, GcReport, InfinoError, MutationStats, OptimizeError,
    OptimizeOptions, VectorFilter, superfile::VectorSearchOptions,
    supertable::Supertable as SupertableHandle,
};

/// The operation surface shared by every table implementation (local or
/// hosted). One method per public table operation; the public [`Supertable`]
/// delegates to it. Kept `pub(crate)` — it is the internal seam, not part of
/// the stable API (which is the inherent methods on [`Supertable`]).
pub(crate) trait Table: Send + Sync {
    fn schema(&self) -> SchemaRef;
    fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError>;
    fn update(&self, predicate: Expr, batch: &RecordBatch) -> Result<MutationStats, InfinoError>;
    fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError>;
    fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError>;
    fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError>;
    fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError>;
    fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError>;
    fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        opts: VectorSearchOptions,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError>;
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
    ) -> Result<Vec<RecordBatch>, InfinoError>;
    fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError>;
    fn gc(&self, safety_gap: Duration) -> Result<GcReport, GcError>;

    /// Test-only: expose the concrete handle behind the trait object so tests
    /// can reach engine internals (`options`, `stats`, `reader`, …) through the
    /// public [`Supertable`]. Not part of any shipped surface.
    #[cfg(any(test, feature = "test-helpers"))]
    fn as_any(&self) -> &dyn Any;
}

// The local (embedded) implementation forwards each operation to the engine
// handle's inherent method. `SupertableHandle::method(self, …)` resolves to the
// inherent method (inherent wins over the trait method of the same name), so
// there is no recursion into the trait.
impl Table for SupertableHandle {
    fn schema(&self) -> SchemaRef {
        SupertableHandle::schema(self)
    }
    fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        SupertableHandle::append(self, batch)
    }
    fn update(&self, predicate: Expr, batch: &RecordBatch) -> Result<MutationStats, InfinoError> {
        SupertableHandle::update(self, predicate, batch)
    }
    fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        SupertableHandle::delete(self, predicate)
    }
    fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        SupertableHandle::bm25_search(self, column, query, k, opts.mode, opts.stats, projection)
    }
    fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        SupertableHandle::token_match(self, column, query, mode, projection)
    }
    fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        SupertableHandle::exact_match(self, column, value, projection)
    }
    fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        SupertableHandle::count(self, column, query, mode)
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
        SupertableHandle::vector_search(self, column, query, k, opts, filter, projection)
    }
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
        SupertableHandle::hybrid_search(
            self,
            text_column,
            text_query,
            mode,
            vector_column,
            vector_query,
            opts,
            k,
            projection,
        )
    }
    fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        SupertableHandle::optimize(self, opts)
    }
    fn gc(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        SupertableHandle::gc(self, safety_gap)
    }
    #[cfg(any(test, feature = "test-helpers"))]
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A single-table handle: `append` / `update` / `delete`, the search surface
/// (`bm25_search` / `vector_search` / `hybrid_search` / `token_match` /
/// `exact_match`), `count`, `schema`, `optimize`, and `gc`. Cheap to clone
/// (one `Arc`); clones share the same table.
#[derive(Clone)]
pub struct Supertable {
    pub(crate) inner: Arc<dyn Table>,
}

impl Supertable {
    /// Wrap the engine's local table handle.
    pub(crate) fn from_local(handle: SupertableHandle) -> Self {
        Self::from_table(Arc::new(handle))
    }

    /// Wrap any table implementation (local or hosted).
    pub(crate) fn from_table(inner: Arc<dyn Table>) -> Self {
        Self { inner }
    }

    /// The table's Arrow schema.
    pub fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    /// Append a batch of rows.
    pub fn append(&self, batch: &RecordBatch) -> Result<(), InfinoError> {
        self.inner.append(batch)
    }

    /// Update rows matching `predicate` with values from `batch`.
    pub fn update(
        &self,
        predicate: Expr,
        batch: &RecordBatch,
    ) -> Result<MutationStats, InfinoError> {
        self.inner.update(predicate, batch)
    }

    /// Delete rows matching `predicate`.
    pub fn delete(&self, predicate: Expr) -> Result<MutationStats, InfinoError> {
        self.inner.delete(predicate)
    }

    /// Ranked BM25 full-text search over one FTS column.
    ///
    /// `opts` ([`Bm25SearchOptions`]) carries the boolean `mode` and the
    /// corpus-statistics selector: [`Bm25Stats::PerSuperfile`](crate::Bm25Stats::PerSuperfile)
    /// (the default, each segment scored against its own local statistics) or
    /// [`Bm25Stats::Global`](crate::Bm25Stats::Global) (one table-wide idf
    /// across all segments, so a fragmented table ranks like a single unified
    /// corpus). `Bm25SearchOptions::new()` is `Or` mode + per-superfile stats.
    pub fn bm25_search(
        &self,
        column: &str,
        query: &str,
        k: usize,
        opts: Bm25SearchOptions,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.inner.bm25_search(column, query, k, opts, projection)
    }

    /// Unranked token match over one FTS column.
    pub fn token_match(
        &self,
        column: &str,
        query: &str,
        mode: BoolMode,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.inner.token_match(column, query, mode, projection)
    }

    /// Unranked exact match over one column.
    pub fn exact_match(
        &self,
        column: &str,
        value: &str,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.inner.exact_match(column, value, projection)
    }

    /// Count rows matching a token query over one FTS column.
    pub fn count(&self, column: &str, query: &str, mode: BoolMode) -> Result<u64, InfinoError> {
        self.inner.count(column, query, mode)
    }

    /// Vector (IVF kNN) search over one vector column.
    ///
    /// Probe width and rerank budget are decided by the engine — the
    /// drain-time calibration stamps them per table and per `k`, and
    /// serving extends them only on the query's own evidence. There is
    /// no caller tuning surface; manual overrides are a test-and-bench
    /// instrument behind `test-helpers` (`vector_search_with_options`).
    pub fn vector_search(
        &self,
        column: &str,
        query: &[f32],
        k: usize,
        filter: Option<VectorFilter<'_>>,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.inner.vector_search(
            column,
            query,
            k,
            VectorSearchOptions::default(),
            filter,
            projection,
        )
    }

    test_visible! {
        /// Test-and-bench-only [`Self::vector_search`] with explicit
        /// probe-width / rerank overrides — recall sweeps and the
        /// exact-scan oracle (all cells at `rerank_mult = ceil(rows/k)`).
        /// Off the public surface: the `cargo-public-api` snapshot is
        /// generated without `test-helpers`.
        fn vector_search_with_options(
            &self,
            column: &str,
            query: &[f32],
            k: usize,
            opts: VectorSearchOptions,
            filter: Option<VectorFilter<'_>>,
            projection: Option<&[&str]>,
        ) -> Result<Vec<RecordBatch>, InfinoError> {
            self.inner
                .vector_search(column, query, k, opts, filter, projection)
        }
    }

    /// Hybrid (BM25 + vector) search. As with [`Self::vector_search`],
    /// vector probe width and rerank budget are engine-decided.
    pub fn hybrid_search(
        &self,
        text_column: &str,
        text_query: &str,
        mode: BoolMode,
        vector_column: &str,
        vector_query: &[f32],
        k: usize,
        projection: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>, InfinoError> {
        self.inner.hybrid_search(
            text_column,
            text_query,
            mode,
            vector_column,
            vector_query,
            VectorSearchOptions::default(),
            k,
            projection,
        )
    }

    test_visible! {
        /// Test-and-bench-only [`Self::hybrid_search`] with explicit
        /// vector probe-width / rerank overrides. Off the public
        /// surface, exactly as [`Self::vector_search_with_options`].
        #[allow(clippy::too_many_arguments)]
        fn hybrid_search_with_options(
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
            self.inner.hybrid_search(
                text_column,
                text_query,
                mode,
                vector_column,
                vector_query,
                opts,
                k,
                projection,
            )
        }
    }

    /// Optimize (compact) the table.
    pub fn optimize(&self, opts: &OptimizeOptions) -> Result<(), OptimizeError> {
        self.inner.optimize(opts)
    }

    /// Garbage-collect orphaned superfiles older than `safety_gap`.
    pub fn gc(&self, safety_gap: Duration) -> Result<GcReport, GcError> {
        self.inner.gc(safety_gap)
    }

    /// Test-only: the underlying local engine handle. Panics for a hosted
    /// (remote) table. Lets tests reach engine internals (`options`, `stats`,
    /// `reader`, …) that are not part of the public surface.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn local_handle(&self) -> &SupertableHandle {
        self.inner
            .as_any()
            .downcast_ref::<SupertableHandle>()
            .expect("local_handle called on a non-local table")
    }
}

// Matches the concrete handle's `Debug` in the public surface. `dyn Table` is
// not `Debug`, so this is hand-written rather than derived.
impl fmt::Debug for Supertable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Supertable").finish_non_exhaustive()
    }
}
