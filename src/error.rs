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
//!
//! ## Boundary context
//!
//! Public API methods prefix the message with the operation (and catalog
//! table name when known), e.g. `not found: open_table(posts): posts`,
//! via [`InfinoError::with_context`]. Structured payload / `source()`
//! chaining can follow in later PRs.

use crate::{
    storage::StorageError,
    superfile::{BuildError as SuperfileBuildError, ReadError as SuperfileReadError},
    supertable::{
        error::{
            BuildError as SupertableBuildError, CommitError as SupertableCommitError, OpenError,
            QueryError,
        },
        mutations::{CommitError as MutationCommitError, MutationError},
    },
};

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

    /// A query exceeded the connection's memory budget (see
    /// [`ConnectOptions::with_connection_memory_budget_bytes`]). For SQL the
    /// engine spills first and only raises this when it still can't fit.
    ///
    /// [`ConnectOptions::with_connection_memory_budget_bytes`]: crate::ConnectOptions::with_connection_memory_budget_bytes
    #[error("over budget: {0}")]
    OverBudget(String),

    /// Backend / internal failure that doesn't map to a more specific
    /// variant.
    #[error("backend: {0}")]
    Backend(String),

    /// An invalid or conflicting configuration was supplied.
    #[error("config: {0}")]
    Config(String),
}

impl InfinoError {
    /// Prefix this error's message with `operation` or `operation(table)`.
    ///
    /// Used at public API boundaries so Display carries enough context
    /// without changing the variant shape. Example:
    /// `not found: open_table(posts): posts`.
    ///  not found: Kind of failure (the InfinoError variant)
    ///  open_table(posts): Public operation that failed (Operation open_table, catalog table posts)
    ///  posts: Detail / original message.
    pub(crate) fn with_context(self, operation: &'static str, table: Option<&str>) -> Self {
        let prefix = match table {
            Some(t) => format!("{operation}({t})"),
            None => operation.to_string(),
        };
        match self {
            Self::NotFound(m) => Self::NotFound(format!("{prefix}: {m}")),
            Self::AlreadyExists(m) => Self::AlreadyExists(format!("{prefix}: {m}")),
            Self::Schema(m) => Self::Schema(format!("{prefix}: {m}")),
            Self::Cardinality(m) => Self::Cardinality(format!("{prefix}: {m}")),
            Self::Io(m) => Self::Io(format!("{prefix}: {m}")),
            Self::Query(m) => Self::Query(format!("{prefix}: {m}")),
            Self::OverBudget(m) => Self::OverBudget(format!("{prefix}: {m}")),
            Self::Backend(m) => Self::Backend(format!("{prefix}: {m}")),
            Self::Config(m) => Self::Config(format!("{prefix}: {m}")),
        }
    }
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
        match e {
            QueryError::OverBudget(msg) => InfinoError::OverBudget(msg),
            other => InfinoError::Query(other.to_string()),
        }
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
        match e {
            // A budget refusal during ingest keeps its own variant so it
            // surfaces as the public over-budget error, not a schema error.
            SupertableBuildError::OverBudget(msg) => InfinoError::OverBudget(msg),
            other => InfinoError::Schema(other.to_string()),
        }
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
        match e {
            // An ingest budget refusal fails the append-flush phase; preserve it
            // as over-budget rather than flattening to a generic backend error.
            MutationCommitError::AppendFlush(SupertableBuildError::OverBudget(msg)) => {
                InfinoError::OverBudget(msg)
            }
            other => InfinoError::Backend(other.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::StorageError;

    #[test]
    fn display_messages_are_prefixed() {
        assert_eq!(
            InfinoError::NotFound("t".into()).to_string(),
            "not found: t"
        );
        assert_eq!(
            InfinoError::AlreadyExists("t".into()).to_string(),
            "already exists: t"
        );
        assert_eq!(InfinoError::Schema("t".into()).to_string(), "schema: t");
        assert_eq!(
            InfinoError::Cardinality("t".into()).to_string(),
            "cardinality: t"
        );
        assert_eq!(InfinoError::Io("t".into()).to_string(), "io: t");
        assert_eq!(InfinoError::Query("t".into()).to_string(), "query: t");
        assert_eq!(InfinoError::Backend("t".into()).to_string(), "backend: t");
        assert_eq!(InfinoError::Config("t".into()).to_string(), "config: t");
    }

    #[test]
    fn with_context_prefixes_operation_and_table() {
        let err = InfinoError::NotFound("posts".into()).with_context("open_table", Some("posts"));
        assert_eq!(err.to_string(), "not found: open_table(posts): posts");

        let err = InfinoError::Cardinality("mismatch".into()).with_context("update", None);
        assert_eq!(err.to_string(), "cardinality: update: mismatch");
    }

    #[test]
    fn from_storage_error_maps_each_variant() {
        assert!(matches!(
            InfinoError::from(StorageError::NotFound { uri: "u".into() }),
            InfinoError::NotFound(_)
        ));
        assert!(matches!(
            InfinoError::from(StorageError::PreconditionFailed { uri: "u".into() }),
            InfinoError::AlreadyExists(_)
        ));
        assert!(matches!(
            InfinoError::from(StorageError::TransientExhausted {
                uri: "u".into(),
                source: "x".into()
            }),
            InfinoError::Io(_)
        ));
        assert!(matches!(
            InfinoError::from(StorageError::Permanent {
                uri: "u".into(),
                source: "x".into()
            }),
            InfinoError::Io(_)
        ));
    }

    #[test]
    fn from_query_read_and_build_errors() {
        assert!(matches!(
            InfinoError::from(QueryError::Plan("p".into())),
            InfinoError::Query(_)
        ));
        // A budget refusal keeps its own variant rather than collapsing to Query.
        assert!(matches!(
            InfinoError::from(QueryError::OverBudget("b".into())),
            InfinoError::OverBudget(_)
        ));
        assert!(matches!(
            InfinoError::from(SuperfileReadError::MissingKv("k")),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            InfinoError::from(SuperfileBuildError::MissingIdColumn("c".into())),
            InfinoError::Schema(_)
        ));
        assert!(matches!(
            InfinoError::from(SupertableBuildError::NoDocsToBuild),
            InfinoError::Schema(_)
        ));
    }

    #[test]
    fn from_commit_and_open_errors_are_backend() {
        assert!(matches!(
            InfinoError::from(SupertableCommitError::Encode("e".into())),
            InfinoError::Backend(_)
        ));
        assert!(matches!(
            InfinoError::from(OpenError::ManifestListParse("m".into())),
            InfinoError::Backend(_)
        ));
    }

    #[test]
    fn from_mutation_error_maps_each_arm() {
        assert!(matches!(
            InfinoError::from(MutationError::PredicateEval(QueryError::Plan("p".into()))),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::Storage(StorageError::NotFound {
                uri: "u".into()
            })),
            InfinoError::NotFound(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::CardinalityMismatch {
                matched: 1,
                new_rows: 2
            }),
            InfinoError::Cardinality(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::MatchCountExceedsCap { matched: 9, cap: 5 }),
            InfinoError::Cardinality(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::SchemaMismatch("s".into())),
            InfinoError::Schema(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::NoStorageAttached),
            InfinoError::Backend(_)
        ));
    }
}
