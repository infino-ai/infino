// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Query-time machinery for the supertable.
//!
//! Each submodule owns one query shape:
//!
//! - [`sql`] — DataFusion SQL via `Supertable::query_sql`.
//! - [`fts`] — BM25 + prefix BM25 fan-out methods on
//!   [`super::SupertableReader`].
//! - [`vector`] — cluster-aware kNN fan-out method on
//!   [`super::SupertableReader`].
//!
//! All non-SQL paths return [`SuperfileHit`] tuples — `(segment_uri,
//! local_doc_id, score)`. Doc-id space is local to a segment in
//! v1, so global identity resolution is the caller's
//! responsibility.
//!
//! [`skip`] holds the manifest-only skip helpers (bloom +
//! term-range + centroid) shared across the query paths.

pub mod candidate;
pub mod df_object_store;
pub mod dispatch;
pub mod exec;
pub mod fts;
pub mod hierarchical_iter;
pub mod provider;
pub mod prune;
pub mod skip;
pub mod sql;
pub mod superfile_reader;
pub mod vector;

pub use vector::VectorSearchOptions;

use super::manifest::SuperfileUri;

/// One scored result from a fan-out query (BM25 or vector).
///
/// `local_doc_id` is the row offset *within* `segment`; doc-id
/// space is local to a segment in v1. Resolving to a global
/// identity goes through the caller's primary-key column —
/// typically a
/// `Supertable::query_sql("SELECT pk FROM supertable WHERE
/// segment = ? AND doc_id = ?")` follow-up, or by carrying the
/// caller's own surrogate key as a scalar column.
///
/// Cheap to copy: 16 bytes for `SuperfileUri` (Uuid) + 4 + 4 = 24 B.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuperfileHit {
    /// Source segment.
    pub segment: SuperfileUri,
    /// Row offset within `segment`.
    pub local_doc_id: u32,
    /// Score. Direction is method-dependent — see the originating
    /// method's docs:
    ///
    /// - [`Supertable::bm25_search`](super::super::Supertable::bm25_search) /
    ///   [`Supertable::bm25_search_prefix`](super::super::Supertable::bm25_search_prefix)
    ///   — BM25 relevance, higher is better. Result vector is sorted
    ///   descending.
    /// - [`Supertable::vector_search`](super::super::Supertable::vector_search)
    ///   — distance under the column's metric (cosine: `1 - dot(a, b)`,
    ///   L2-sq: squared L2). Smaller is better. Result vector is sorted
    ///   ascending.
    pub score: f32,
}

/// A resolved public search hit: the stable public `_id` plus the score.
///
/// `id` is the supertable's auto-injected `_id`, **not** an internal,
/// segment-local row offset. Returned by the lightweight public
/// `Supertable::bm25_hits` / `vector_hits` / `token_match` /
/// `exact_match`, best-scoring first (ranked methods) or unordered
/// (the unranked match methods, where `score` is `0.0`). The
/// row-returning `Supertable::bm25_search` / `vector_search` instead
/// materialize full Arrow rows.
///
/// `#[non_exhaustive]`: constructed only by the engine, so fields can be
/// added later without breaking downstream `match`/struct-literal code.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchHit {
    /// The row's public `_id`.
    pub id: i128,
    /// Score for this hit (see the type-level note on direction).
    pub score: f32,
}
