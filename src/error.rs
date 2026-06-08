//! The single public error type for the curated infino API.
//!
//! Public methods return `Result<T, Error>`. The internal per-stage
//! error enums (`OpenError`, `BuildError`, `ReadError`, `QueryError`,
//! `MutationError`, `CommitError`, `StorageError`) convert inward via
//! `From`. The mappings are intentionally **coarse** — they collapse
//! many internal variants onto a small, stable public set. `Error` is
//! `#[non_exhaustive]`, so finer variants (or structured source
//! chaining) can be added later without a breaking change.

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
pub enum Error {
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

impl From<StorageError> for Error {
    fn from(e: StorageError) -> Self {
        let msg = e.to_string();
        match e {
            StorageError::NotFound { .. } => Error::NotFound(msg),
            StorageError::PreconditionFailed { .. } => Error::AlreadyExists(msg),
            StorageError::TransientExhausted { .. } | StorageError::Permanent { .. } => {
                Error::Io(msg)
            }
        }
    }
}

impl From<QueryError> for Error {
    fn from(e: QueryError) -> Self {
        Error::Query(e.to_string())
    }
}

impl From<SuperfileReadError> for Error {
    fn from(e: SuperfileReadError) -> Self {
        Error::Query(e.to_string())
    }
}

impl From<SuperfileBuildError> for Error {
    fn from(e: SuperfileBuildError) -> Self {
        Error::Schema(e.to_string())
    }
}

impl From<SupertableBuildError> for Error {
    fn from(e: SupertableBuildError) -> Self {
        Error::Schema(e.to_string())
    }
}

impl From<SupertableCommitError> for Error {
    fn from(e: SupertableCommitError) -> Self {
        Error::Backend(e.to_string())
    }
}

impl From<OpenError> for Error {
    fn from(e: OpenError) -> Self {
        Error::Backend(e.to_string())
    }
}

impl From<MutationError> for Error {
    fn from(e: MutationError) -> Self {
        let msg = e.to_string();
        match e {
            MutationError::PredicateEval(q) => Error::from(q),
            MutationError::Storage(s) => Error::from(s),
            MutationError::CardinalityMismatch { .. }
            | MutationError::MatchCountExceedsCap { .. } => Error::Cardinality(msg),
            MutationError::SchemaMismatch(_) => Error::Schema(msg),
            _ => Error::Backend(msg),
        }
    }
}

impl From<MutationCommitError> for Error {
    fn from(e: MutationCommitError) -> Self {
        Error::Backend(e.to_string())
    }
}
