// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! The single public error type for the curated infino API.
//!
//! Public methods return `Result<T, InfinoError>`. The internal
//! per-stage error enums (`OpenError`, `BuildError`, `ReadError`,
//! `QueryError`, `MutationError`, `CommitError`, `StorageError`)
//! convert inward via `From`. The mappings are intentionally **coarse**
//! — they collapse many internal variants onto a small, stable public
//! set. `InfinoError` is `#[non_exhaustive]`, so finer variants (or
//! structured source chaining) can be added later without a breaking
//! change. Named `InfinoError` (not `Error`) to avoid colliding with
//! the `std::error::Error` trait at call sites and to read consistently
//! alongside `DataFusionError` / `ArrowError`.

use crate::storage::StorageError;
use crate::superfile::BuildError as SuperfileBuildError;
use crate::superfile::ReadError as SuperfileReadError;
use crate::supertable::error::{
    BuildError as SupertableBuildError, CommitError as SupertableCommitError, OpenError, QueryError,
};
use crate::supertable::mutations::{CommitError as MutationCommitError, MutationError};

/// Coarse, stable error type returned by every public infino method.
///
/// Each variant carries a human-readable message (the originating
/// error's `Display`). The set is deliberately small; `#[non_exhaustive]`
/// keeps it open to growth without breaking downstream `match`es.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum InfinoError {
    /// A named table, object, or column was not found.
    #[error("not found: {0}")]
    NotFound(String),

    /// A create conflicted with an existing name / object.
    #[error("already exists: {0}")]
    AlreadyExists(String),

    /// Schema or column validation failed.
    #[error("schema: {0}")]
    Schema(String),

    /// A predicate matched a different row count than required, or
    /// exceeded the mutation cap.
    #[error("cardinality: {0}")]
    Cardinality(String),

    /// Storage / I/O failure.
    #[error("io: {0}")]
    Io(String),

    /// SQL planning or execution failure.
    #[error("query: {0}")]
    Query(String),

    /// Backend / internal failure that doesn't map to a more specific
    /// variant.
    #[error("backend: {0}")]
    Backend(String),
}

impl From<StorageError> for InfinoError {
    fn from(e: StorageError) -> Self {
        let msg = e.to_string();
        match e {
            StorageError::NotFound { .. } => InfinoError::NotFound(msg),
            StorageError::PreconditionFailed { .. } => InfinoError::AlreadyExists(msg),
            StorageError::TransientExhausted { .. } | StorageError::Permanent { .. } => {
                InfinoError::Io(msg)
            }
        }
    }
}

impl From<QueryError> for InfinoError {
    fn from(e: QueryError) -> Self {
        InfinoError::Query(e.to_string())
    }
}

impl From<SuperfileReadError> for InfinoError {
    fn from(e: SuperfileReadError) -> Self {
        InfinoError::Query(e.to_string())
    }
}

impl From<SuperfileBuildError> for InfinoError {
    fn from(e: SuperfileBuildError) -> Self {
        InfinoError::Schema(e.to_string())
    }
}

impl From<SupertableBuildError> for InfinoError {
    fn from(e: SupertableBuildError) -> Self {
        InfinoError::Schema(e.to_string())
    }
}

impl From<SupertableCommitError> for InfinoError {
    fn from(e: SupertableCommitError) -> Self {
        InfinoError::Backend(e.to_string())
    }
}

impl From<OpenError> for InfinoError {
    fn from(e: OpenError) -> Self {
        InfinoError::Backend(e.to_string())
    }
}

impl From<MutationError> for InfinoError {
    fn from(e: MutationError) -> Self {
        let msg = e.to_string();
        match e {
            MutationError::PredicateEval(q) => InfinoError::from(q),
            MutationError::Storage(s) => InfinoError::from(s),
            MutationError::CardinalityMismatch { .. }
            | MutationError::MatchCountExceedsCap { .. } => InfinoError::Cardinality(msg),
            MutationError::SchemaMismatch(_) => InfinoError::Schema(msg),
            _ => InfinoError::Backend(msg),
        }
    }
}

impl From<MutationCommitError> for InfinoError {
    fn from(e: MutationCommitError) -> Self {
        InfinoError::Backend(e.to_string())
    }
}
