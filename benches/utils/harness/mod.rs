// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic benchmark harness.
//!
//! Defines the [`FtsEngine`] trait so one driver ([`run_fts`]) can
//! measure infino and other retrieval engines through identical code.
//! infino ships the reference implementation ([`InfinoFtsEngine`]); the
//! external comparison crate implements the trait for
//! other engines and drives them all
//! the same way, against a byte-identical [`crate::corpus::MmapTextCorpus`].
//!
//! The three verbs the driver measures are:
//!
//!   - [`FtsEngine::open`]  — prepare an empty index for one column.
//!   - [`FtsEngine::write`] — ingest every document and seal the index
//!     ready to query (the build phase).
//!   - [`FtsEngine::read`]  — run a BM25 query (the search phase).
//!
//! Memory (RSS) and timing reuse the same [`crate::rss`] sampler the
//! in-tree infino benches use, so internal and comparison numbers are
//! produced by one measurement path.

pub mod driver;
mod infino_engine;
mod infino_sql_engine;
mod infino_vector_engine;
pub mod sql_driver;
pub mod vector_driver;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
pub use driver::{
    BuildStat, EngineFtsResult, FtsQuery, PhaseStats, QueryStats, run_fts, run_fts_with_index,
};
use infino::superfile::vector::distance::Metric;
pub use infino_engine::{
    InfinoFtsEngine, InfinoFtsIndex, build_positionless, parallel_build_positionless,
};
pub use infino_sql_engine::{
    InfinoSqlEngine, InfinoSqlIndex, SQL_DIM, build_supertable_with_options, emb_for,
    sample_query_csv, scatter_key, sql_options, sql_schema,
};
pub use infino_vector_engine::{InfinoVectorEngine, InfinoVectorIndex};
pub use sql_driver::{
    EngineSqlResult, SqlBuildStat, SqlQuery, SqlQueryStats, SqlRunConfig, run_sql,
    run_sql_batches_with_index, run_sql_with_index,
};
pub use vector_driver::{
    EngineVectorResult, VectorBuildStat, VectorMetric, VectorQuery, VectorQueryStats,
    VectorRunConfig, VectorSearch, run_vector, run_vector_with_index,
};

// Re-export the shared corpus + byte formatter so a comparison binary
// has one import root for everything it needs to run `run_fts`.
pub use crate::corpus::MmapTextCorpus;
pub use crate::rss::{RssStats, fmt_bytes};

/// Boolean combination mode for a multi-term full-text query.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolMode {
    /// Match documents containing any term.
    Or,
    /// Match documents containing all terms.
    And,
}

/// One ranked search hit: a stable document id and its relevance score
/// (higher is better). `doc_id` is engine-agnostic so the driver can
/// grade recall by comparing id sets across engines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub doc_id: u64,
    pub score: f32,
}

/// Which modalities an engine supports, so the comparison driver never
/// asks an engine for a capability it lacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub fts: bool,
    pub vector: bool,
    pub sql: bool,
    pub hybrid: bool,
    /// Engine can add vectors to an existing index without a full rebuild.
    /// See [`VectorEngine::insert`] — an immutable-artifact engine may
    /// still advertise this if it implements `insert` as an honest, if
    /// heavier, incremental build; the capability flag alone does not
    /// promise a cheap in-place add.
    pub vector_insert: bool,
    /// Engine can remove vectors from an existing index by id, without a
    /// full rebuild. See [`VectorEngine::remove`].
    pub vector_remove: bool,
    /// Engine can serialize its index to bytes and reopen it — i.e. has a
    /// real save/load boundary distinct from "keep the in-process handle".
    /// See [`VectorEngine::save`] / [`VectorEngine::load`].
    pub vector_save_load: bool,
}

/// A full-text retrieval engine under comparison.
///
/// `create` → `write` → `read` → `close` → `delete` is the full
/// lifecycle. `write` performs the ingest *and* seals the index so it is
/// ready to query, so the build/ingest cost is attributed to `write`
/// (not split across a later first read). The corpus is supplied by the
/// driver, so every engine indexes byte-identical documents.
pub trait FtsEngine {
    /// Sealed, queryable index handle produced by `write`.
    type Index;

    /// Engine name — the column/row label in the comparison output.
    fn name() -> &'static str;

    /// Which modalities this engine implements.
    fn capabilities() -> Capabilities;

    /// Create an empty index/artifact for a single text column.
    fn create(column: &str) -> Self::Index;

    /// Open/prepare the handle used by the benchmark lifecycle. For
    /// in-memory engines this usually delegates to [`FtsEngine::create`];
    /// engines with persisted artifacts can make this distinct.
    fn open(column: &str) -> Self::Index {
        Self::create(column)
    }

    /// Ingest all `(doc_id, text)` rows with a single writer and seal
    /// the index ready to `read`. This is the canonical, queryable build
    /// — the "1 writer" build row and the index every query runs against.
    fn write(index: &mut Self::Index, docs: &[(u64, &str)]);

    /// Build the corpus from scratch with `writers` concurrent writers,
    /// for the build-throughput row only — nothing queryable is kept.
    /// `writers == 1` is the single-writer build; `> 1` is the engine's
    /// parallel build (infino shards across builders; Tantivy uses that
    /// many indexing threads). Lets the driver compare ingest at 1 vs N
    /// writers apples-to-apples without favoring any engine.
    fn parallel_write(column: &str, docs: &[(u64, &str)], writers: usize);

    /// BM25 top-`k` over already-tokenized `terms`, returning hits
    /// sorted by descending score. The measured query phase.
    fn read(index: &Self::Index, terms: &[&str], k: usize, mode: BoolMode) -> Vec<Hit>;

    /// Close reader/search handles while retaining enough state for
    /// `delete`. In-memory implementations can drop transient readers.
    fn close(index: &mut Self::Index);

    /// Delete/cleanup the engine artifact. For in-memory engines this is
    /// usually just dropping the index; object-backed engines should
    /// remove temporary files/objects here.
    fn delete(index: Self::Index);
}

/// One nearest-neighbor hit returned by a vector engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorHit {
    pub doc_id: u64,
    /// Distance-like score where smaller is better for all metrics after
    /// engine normalization (`NegDot` returns `-dot`).
    pub distance: f32,
}

/// A vector retrieval engine under comparison.
///
/// `create` → `write` → `read` → `close` → `delete` is the full vector
/// lifecycle. `write` builds the canonical 1-writer queryable artifact
/// and must retain any bytes/handles needed by later correctness, warm
/// search, and cold upload. `parallel_write` is the build-throughput-only axis
/// for `N writers`.
pub trait VectorEngine {
    type Index;

    fn name() -> &'static str;

    fn capabilities() -> Capabilities;

    fn create(column: &str, dim: usize, metric: VectorMetric) -> Self::Index;

    fn open(column: &str, dim: usize, metric: VectorMetric) -> Self::Index {
        Self::create(column, dim, metric)
    }

    fn write(index: &mut Self::Index, vectors: &[f32]);

    fn parallel_write(
        column: &str,
        vectors: &[f32],
        dim: usize,
        metric: VectorMetric,
        writers: usize,
    );

    fn read(index: &Self::Index, query: &[f32], k: usize, search: VectorSearch) -> Vec<VectorHit>;

    fn close(index: &mut Self::Index);

    fn delete(index: Self::Index);

    /// Add `vectors` (a flat buffer of `vectors.len() / dim` new rows) to
    /// an already-built, queryable index. Ids are assigned starting at
    /// `next_id` (the caller tracks the running id counter so ids stay
    /// unique across insert calls). Returns `false` if unsupported —
    /// callers must check [`Capabilities::vector_insert`] before calling
    /// and treat a `false` return as a harness bug, not a normal outcome.
    ///
    /// Deliberately vague about *how* the engine achieves "add to a
    /// queryable index": an immutable-artifact engine may implement this
    /// as an incremental rebuild rather than a true append-to-served
    /// -index. Implementors must document which they do.
    fn insert(_index: &mut Self::Index, _vectors: &[f32], _next_id: u64) -> bool {
        false
    }

    /// Remove the rows named by `ids` from the index. Returns `false` if
    /// unsupported (see [`VectorEngine::insert`]'s return-value contract).
    fn remove(_index: &mut Self::Index, _ids: &[u64]) -> bool {
        false
    }

    /// Serialize the index to bytes. `None` if unsupported.
    fn save(_index: &Self::Index) -> Option<Vec<u8>> {
        None
    }

    /// Reopen an index from bytes previously produced by [`VectorEngine::save`].
    /// `None` if unsupported.
    fn load(
        _column: &str,
        _dim: usize,
        _metric: VectorMetric,
        _bytes: &[u8],
    ) -> Option<Self::Index> {
        None
    }
}

/// A scalar row ingested by SQL engines.
#[derive(Clone, Copy, Debug)]
pub struct SqlRow<'a> {
    pub doc_id: u64,
    pub title: &'a str,
    pub category: &'a str,
    pub score: i64,
}

/// Engine-normalized SQL query output. For the benchmark tables and
/// correctness checks we only need row count; engine-specific batches
/// stay inside the implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SqlOutput {
    pub rows: usize,
}

/// The embedding column of a schema-driven corpus, when it has one.
#[derive(Clone, Debug)]
pub struct SqlVectorSpec {
    pub column: String,
    pub dim: usize,
    pub metric: Metric,
}

/// What a schema-driven corpus tells an engine about its own shape.
///
/// `fts_columns` is empty for every corpus this ships with: choosing which
/// of a dataset's text columns deserve a BM25 index needs a per-dataset
/// declaration, and indexing all of them would dominate build time for
/// queries that never use them.
#[derive(Clone, Debug)]
pub struct SqlCorpusSpec {
    pub schema: SchemaRef,
    pub fts_columns: Vec<String>,
    pub vector: Option<SqlVectorSpec>,
}

/// A SQL engine that can ingest a dataset's own schema, rather than the
/// fixed [`SqlRow`] fixture. Opt-in: an engine that does not implement
/// this is unaffected by its existence.
pub trait SchemaDrivenSqlEngine: SqlEngine {
    fn create_with_spec(spec: &SqlCorpusSpec) -> Self::Index;

    fn write_batches(index: &mut Self::Index, batches: &[RecordBatch]);

    fn parallel_write_batches(spec: &SqlCorpusSpec, batches: &[RecordBatch], writers: usize);
}

/// A SQL engine under comparison.
pub trait SqlEngine {
    type Index;

    fn name() -> &'static str;

    fn capabilities() -> Capabilities;

    fn create() -> Self::Index;

    fn open() -> Self::Index {
        Self::create()
    }

    fn write(index: &mut Self::Index, rows: &[SqlRow<'_>]);

    fn parallel_write(rows: &[SqlRow<'_>], writers: usize);

    fn read(index: &Self::Index, sql: &str) -> SqlOutput;

    fn close(index: &mut Self::Index);

    fn delete(index: Self::Index);
}
