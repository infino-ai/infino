// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Engine-generic benchmark harness.
//!
//! Defines the [`FtsEngine`] trait so one driver ([`run_fts`]) can
//! measure infino and other retrieval engines through identical code.
//! infino ships the reference implementation ([`InfinoFtsEngine`]); the
//! external comparison crate (`retrievalbench`) implements the trait for
//! other engines (Tantivy, DuckDB, LanceDB, CoreDB) and drives them all
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

pub use driver::{EngineFtsResult, FtsQuery, PhaseStats, QueryStats, run_fts};
pub use infino_engine::{InfinoFtsEngine, InfinoFtsIndex};

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
}

/// A full-text retrieval engine under comparison.
///
/// `open` → `write` → `read` is the measured lifecycle. `write`
/// performs the full ingest *and* seals the index so it is ready to
/// query, so the build/ingest cost is attributed to `write` (not split
/// across a later first read). The corpus is supplied by the driver, so
/// every engine indexes byte-identical documents.
pub trait FtsEngine {
    /// Sealed, queryable index handle produced by `write`.
    type Index;

    /// Engine name — the column/row label in the comparison output.
    fn name() -> &'static str;

    /// Which modalities this engine implements.
    fn capabilities() -> Capabilities;

    /// Prepare an empty index for a single text column.
    fn open(column: &str) -> Self::Index;

    /// Ingest all `(doc_id, text)` rows and seal the index ready to
    /// `read`. This is the measured build/ingest phase.
    fn write(index: &mut Self::Index, docs: &[(u64, &str)]);

    /// BM25 top-`k` over already-tokenized `terms`, returning hits
    /// sorted by descending score. The measured query phase.
    fn read(index: &Self::Index, terms: &[&str], k: usize, mode: BoolMode) -> Vec<Hit>;
}
