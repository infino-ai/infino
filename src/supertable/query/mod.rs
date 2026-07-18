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
//! All non-SQL paths return [`SuperfileHit`] tuples — `(superfile_uri,
//! local_doc_id, score)`. Doc-id space is local to a superfile in
//! v1, so global identity resolution is the caller's
//! responsibility.
//!
//! [`skip`] holds the manifest-only skip helpers (bloom +
//! term-range + centroid) shared across the query paths.

pub mod candidate;
pub mod covered_agg;
pub mod df_object_store;
pub mod dispatch;
pub mod exec;
pub mod fts;
pub mod hierarchical_iter;
pub mod provider;
pub mod prune;
pub(crate) mod scalar_cache;
pub mod skip;
pub mod sql;
pub mod superfile_reader;
pub mod vector;

pub use vector::VectorSearchOptions;

use super::manifest::SuperfileUri;

/// One scored result from a fan-out query (BM25 or vector).
///
/// `local_doc_id` is the row offset *within* `superfile`; doc-id
/// space is local to a superfile in v1. Resolving to a global
/// identity goes through the caller's primary-key column —
/// typically a
/// `Supertable::query_sql("SELECT pk FROM supertable WHERE
/// superfile = ? AND doc_id = ?")` follow-up, or by carrying the
/// caller's own surrogate key as a scalar column.
///
/// Cheap to copy: `SuperfileUri` (16) + 4 + 4 + the optional inline `_id`
/// (`Option<i128>`, 16-aligned) — a small, `Copy` value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SuperfileHit {
    /// Source superfile.
    pub superfile: SuperfileUri,
    /// Row offset within `superfile`.
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
    /// Inline stable `_id` resolved during search, when available. Hidden
    /// vector-index hits carry the user `_id` here — resolved at the fan-out
    /// tag site from the in-wave-prefetched `_id` region — so the remap step
    /// reuses it instead of issuing a trailing region GET. `None` on every
    /// other path (FTS, user-table, hits without an inline region); the remap
    /// then falls back to the region/scalar read.
    pub stable_id: Option<i128>,
}
