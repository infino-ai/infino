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

use std::{error::Error, io};

use arrow_schema::ArrowError;
use datafusion::error::DataFusionError;
use object_store::Error as ObjectStoreError;

use crate::{
    storage::{StorageError, error_chain, permission_denied_in_chain},
    superfile::BuildError as SuperfileBuildError,
    supertable::{
        error::{
            BuildError as SupertableBuildError, CommitError as SupertableCommitError, OpenError,
            QueryError,
        },
        manifest::ManifestLoadError,
        mutations::{CommitError as MutationCommitError, MutationError},
        schema::error::SchemaError,
    },
};

/// The text DataFusion's parquet row filter wraps a pushed-down predicate's
/// failure in: `Error evaluating filter predicate: {e:?}`, the predicate's
/// own `DataFusionError` in `Debug` form inside an `ArrowError::ComputeError`.
/// The type does not survive that hop, so this text is the one way to tell a
/// predicate that failed on the caller's data from a scan that failed.
const PUSHED_DOWN_PREDICATE_FAILED: &str = "Error evaluating filter predicate: ";

/// How the wrapped error starts when an arrow kernel (a cast, a divide)
/// rejected the caller's values, in that `Debug` form.
const FAILED_ON_THE_DATA: &str = "ArrowError(";

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

    /// The storage backend refused the credentials in use (HTTP 403 / 401).
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// The query or search is wrong: it does not parse or plan, names a
    /// column or function that does not exist, or fails on the caller's own
    /// data (a bad cast). A failure inside the engine is `Io` or `Backend`;
    /// valid SQL the engine does not implement is `Unsupported`.
    #[error("query: {0}")]
    Query(String),

    /// A query exceeded the connection's memory budget (see
    /// [`ConnectOptions::with_connection_memory_budget_bytes`]). For SQL the
    /// engine spills first and only raises this when it still can't fit. Also
    /// raised when a SQL statement runs while the process is over its memory
    /// limit: 90% of the process's cgroup memory limit, or
    /// `memory.process_limit_bytes` in config.
    ///
    /// [`ConnectOptions::with_connection_memory_budget_bytes`]: crate::ConnectOptions::with_connection_memory_budget_bytes
    #[error("over budget: {0}")]
    OverBudget(String),

    /// A concurrent writer won the race: an optimistic-concurrency
    /// (compare-and-set) precondition failed and the operation's own retry
    /// budget was exhausted, or another writer in this process holds the
    /// table's single writer slot.
    ///
    /// **Retryable.** Nothing partial is left visible — the losing writer's
    /// manifest swap never published, and a mutation whose WAL did become
    /// durable is completed idempotently by the recovery sweep. Reissuing
    /// `append` / `update` / `delete` (ideally with backoff) resolves the
    /// predicate against fresh state and can succeed. Persistent conflicts
    /// mean genuine multi-writer contention on one table, not a fault.
    #[error("conflict: {0}")]
    Conflict(String),

    /// Backend / internal failure that doesn't map to a more specific
    /// variant.
    #[error("backend: {0}")]
    Backend(String),

    /// An invalid or conflicting configuration was supplied.
    #[error("config: {0}")]
    Config(String),

    /// The query is valid but uses something the engine does not support
    /// yet, such as a SQL feature DataFusion does not implement.
    #[error("unsupported: {0}")]
    Unsupported(String),
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
            Self::PermissionDenied(m) => Self::PermissionDenied(format!("{prefix}: {m}")),
            Self::Query(m) => Self::Query(format!("{prefix}: {m}")),
            Self::OverBudget(m) => Self::OverBudget(format!("{prefix}: {m}")),
            Self::Conflict(m) => Self::Conflict(format!("{prefix}: {m}")),
            Self::Backend(m) => Self::Backend(format!("{prefix}: {m}")),
            Self::Config(m) => Self::Config(format!("{prefix}: {m}")),
            Self::Unsupported(m) => Self::Unsupported(format!("{prefix}: {m}")),
        }
    }

    /// [`From<QueryError>`] for a query error held by reference, as one found
    /// inside a DataFusion error is. The match names every variant, so a new
    /// one has to choose its public variant rather than fall into `Query`.
    fn from_query_ref(e: &QueryError) -> Self {
        let variant = match e {
            QueryError::DataFusion(df) => return datafusion_error(df),
            QueryError::InvalidQuery(_) => InfinoError::Query,
            QueryError::Store(_) | QueryError::Parquet(_) => InfinoError::Io,
            QueryError::ManifestLoad(load) => manifest_load_variant(load),
            QueryError::Internal(_) => InfinoError::Backend,
            QueryError::OverBudget(_) => InfinoError::OverBudget,
            QueryError::PermissionDenied(_) => InfinoError::PermissionDenied,
        };
        variant(e.to_string())
    }
}

impl From<StorageError> for InfinoError {
    fn from(e: StorageError) -> Self {
        let msg = e.to_string();
        match e {
            StorageError::NotFound { .. } => InfinoError::NotFound(msg),
            StorageError::PreconditionFailed { .. } => InfinoError::Conflict(msg),
            StorageError::PermissionDenied { .. } => InfinoError::PermissionDenied(msg),
            StorageError::TransientExhausted { .. } | StorageError::Permanent { .. } => {
                InfinoError::Io(msg)
            }
        }
    }
}

/// `Query` is only for the caller's own mistakes: a query that does not parse
/// or plan, a search over a column the table does not index. Everything the
/// engine failed at on its own, mid-query, is something else, so a caller can
/// tell "fix the query" from "the engine failed":
///
/// | `QueryError` | public | why |
/// |---|---|---|
/// | `InvalidQuery` | `Query` | the request itself is wrong |
/// | `DataFusion` | by its cause | see [`datafusion_error`] |
/// | `Store`, `Parquet` | `Io` | a read failed; retrying can succeed |
/// | `ManifestLoad` | as a [`ManifestLoadError`] would | same failure, same answer |
/// | `Internal` | `Backend` | the engine broke its own invariant: a bug |
/// | `OverBudget`, `PermissionDenied` | the same names | |
impl From<QueryError> for InfinoError {
    fn from(e: QueryError) -> Self {
        InfinoError::from_query_ref(&e)
    }
}

/// Map a failure DataFusion returned from planning or running a query to the
/// public error. Our own errors cross a plan typed (see `From<QueryError> for
/// DataFusionError`), so the cause decides:
///
/// Checked in this order, the first match deciding:
///
/// | found | public |
/// |---|---|
/// | our [`QueryError`] anywhere in the chain | its own mapping and message |
/// | `ResourcesExhausted` at the root, however wrapped | `OverBudget`, the pool's message |
/// | refused credentials | `PermissionDenied` |
/// | a read that failed: storage, object store or io, in the chain | `Io` |
/// | `NotImplemented` | `Unsupported` |
/// | `SQL`, `Plan`, `SchemaError`, `Configuration`, `ArrowError` | `Query`: the query or its data |
/// | `External` holding someone else's error (a regex that does not parse) | `Query` |
/// | a pushed-down predicate that failed on the data | `Query` |
/// | `Execution` while turning SQL into a logical plan (see [`datafusion_planning_error`]) | `Query` |
/// | anything else: `Execution`, `Internal`, a failed task | `Backend` |
///
/// An `External` error at the root that is neither ours nor storage's comes
/// from a DataFusion function rejecting its arguments, which only the caller
/// wrote. DataFusion's own `Execution` errors while a query runs are mixed (a
/// value the caller's function cannot take next to a missing partition), so
/// they count as ours until shown otherwise: a false `Backend` reads as an
/// engine fault the caller can report, a false `Query` blames the caller and
/// hides the bug.
pub(crate) fn datafusion_error(e: &DataFusionError) -> InfinoError {
    classify_datafusion_error(e, false)
}

/// [`datafusion_error`] for a failure turning SQL into a logical plan. No
/// optimizer has run and no data has been scanned yet, so DataFusion's own
/// `Execution` there is almost always a function rejecting the caller's
/// arguments (`arrow_cast(x, 'NotAType')`), and is the caller's. Anything we
/// read while planning (a search table function opening its table) fails with
/// our own error, typed, and keeps its own answer. The cost: the few planner
/// checks DataFusion raises as `Execution` for its own impossible states would
/// read as `Query` here.
pub(crate) fn datafusion_planning_error(e: &DataFusionError) -> InfinoError {
    classify_datafusion_error(e, true)
}

fn classify_datafusion_error(e: &DataFusionError, planning: bool) -> InfinoError {
    if let Some(cause) = error_chain(e).find_map(|link| link.downcast_ref::<QueryError>()) {
        return InfinoError::from_query_ref(cause);
    }
    // A budget refusal keeps the pool's own message, however an operator
    // wrapped it (an external sort adds "Not enough memory to continue").
    if let DataFusionError::ResourcesExhausted(msg) = e.find_root() {
        return InfinoError::OverBudget(msg.clone());
    }
    let variant = if permission_denied_in_chain(e) {
        InfinoError::PermissionDenied
    } else if error_chain(e).any(is_failed_read) {
        InfinoError::Io
    } else {
        // Nothing of ours or the store's under it: DataFusion's own variant.
        match e.find_root() {
            DataFusionError::NotImplemented(_) => InfinoError::Unsupported,
            DataFusionError::SQL(..)
            | DataFusionError::Plan(_)
            | DataFusionError::SchemaError(..)
            | DataFusionError::Configuration(_)
            | DataFusionError::ArrowError(..)
            | DataFusionError::External(_) => InfinoError::Query,
            DataFusionError::ParquetError(_) if pushed_down_predicate_failed_on_the_data(e) => {
                InfinoError::Query
            }
            DataFusionError::Execution(_) if planning => InfinoError::Query,
            _ => InfinoError::Backend,
        }
    };
    variant(e.to_string())
}

/// Whether `link` is a read that failed: a storage, object store or io error.
/// A store saying it cannot do an operation at all (not implemented, not
/// supported) is not one: no retry changes that answer, and a scan only asks
/// a store for what it serves, so reaching one is our bug.
fn is_failed_read(link: &(dyn Error + 'static)) -> bool {
    if let Some(store) = link.downcast_ref::<ObjectStoreError>() {
        return !matches!(
            store,
            ObjectStoreError::NotImplemented { .. } | ObjectStoreError::NotSupported { .. }
        );
    }
    link.is::<StorageError>() || link.is::<io::Error>()
}

/// True when a parquet scan failed because a predicate pushed into it
/// rejected the caller's values, not because the read or the engine failed.
/// See [`PUSHED_DOWN_PREDICATE_FAILED`].
fn pushed_down_predicate_failed_on_the_data(e: &DataFusionError) -> bool {
    error_chain(e).any(|link| {
        matches!(
            link.downcast_ref::<ArrowError>(),
            Some(ArrowError::ComputeError(message))
                if message
                    .strip_prefix(PUSHED_DOWN_PREDICATE_FAILED)
                    .is_some_and(|inner| inner.starts_with(FAILED_ON_THE_DATA))
        )
    })
}

impl From<ManifestLoadError> for InfinoError {
    fn from(e: ManifestLoadError) -> Self {
        manifest_load_variant(&e)(e.to_string())
    }
}

/// The public variant a manifest load failure maps to, wherever it is met:
/// opening a table, or in the middle of a query. Returned as the variant's
/// constructor, so each caller wraps its own message (a mid-query failure
/// keeps its `manifest load error:` label).
fn manifest_load_variant(e: &ManifestLoadError) -> fn(String) -> InfinoError {
    if e.is_permission_denied() {
        return InfinoError::PermissionDenied;
    }
    match e {
        // The table this handle was reading has been dropped and purged, so
        // the name it was opened under no longer resolves to anything:
        // `NotFound`, not a backend fault, is what a caller must react to.
        ManifestLoadError::PointerVanished => InfinoError::NotFound,
        // A storage fault reading the manifest (the pointer probe or a part
        // load) is a transient I/O hiccup, not a permanent failure.
        // Surface it as `Io` so a caller can retry (e.g. against another
        // copy of the data) rather than treat it as a hard backend fault.
        ManifestLoadError::Storage(_) => InfinoError::Io,
        _ => InfinoError::Backend,
    }
}

impl From<SuperfileBuildError> for InfinoError {
    fn from(e: SuperfileBuildError) -> Self {
        InfinoError::Schema(e.to_string())
    }
}

impl From<SupertableBuildError> for InfinoError {
    fn from(e: SupertableBuildError) -> Self {
        if let Some(msg) = e.over_budget() {
            return InfinoError::OverBudget(msg.to_string());
        }
        if e.is_permission_denied() {
            return InfinoError::PermissionDenied(e.to_string());
        }
        if e.is_conflict() {
            return InfinoError::Conflict(e.to_string());
        }
        // A commit that found its table dropped and purged is not a schema
        // problem; it is the name no longer resolving. Same answer the read
        // path gives, so a caller can match one condition, not three.
        if matches!(e, SupertableBuildError::TableGone) {
            return InfinoError::NotFound(e.to_string());
        }
        if matches!(
            e,
            SupertableBuildError::Schema(SchemaError::TableExists { .. })
        ) {
            return InfinoError::AlreadyExists(e.to_string());
        }
        // A bad analyzer name is a configuration mistake, not a schema
        // shape problem — surface it as the same class a bad connect
        // option gets.
        if matches!(e, SupertableBuildError::UnknownAnalyzer { .. }) {
            return InfinoError::Config(e.to_string());
        }
        InfinoError::Schema(e.to_string())
    }
}

impl From<SupertableCommitError> for InfinoError {
    fn from(e: SupertableCommitError) -> Self {
        let msg = e.to_string();
        if e.is_permission_denied() {
            return InfinoError::PermissionDenied(msg);
        }
        match e {
            // Reached by commit paths that surface the typed error directly
            // (the append path converts to `BuildError::TableGone` first).
            SupertableCommitError::PointerVanished => InfinoError::NotFound(msg),
            // The OCC retry budget ran out on a contended pointer / part CAS.
            e if e.is_conflict() => InfinoError::Conflict(msg),
            _ => InfinoError::Backend(msg),
        }
    }
}

impl From<OpenError> for InfinoError {
    fn from(e: OpenError) -> Self {
        if e.is_permission_denied() {
            return InfinoError::PermissionDenied(e.to_string());
        }
        if e.is_conflict() {
            return InfinoError::Conflict(e.to_string());
        }
        InfinoError::Backend(e.to_string())
    }
}

impl From<MutationError> for InfinoError {
    fn from(e: MutationError) -> Self {
        let msg = e.to_string();
        if e.is_conflict() {
            return InfinoError::Conflict(msg);
        }
        if e.is_permission_denied() {
            return InfinoError::PermissionDenied(msg);
        }
        match e {
            // Routes over-budget through From<QueryError> when the predicate
            // eval was the budget refusal.
            MutationError::PredicateEval(q) => InfinoError::from(q),
            MutationError::Storage(s) => InfinoError::from(s),
            MutationError::CardinalityMismatch { .. }
            | MutationError::MatchCountExceedsCap { .. } => InfinoError::Cardinality(msg),
            // Classifies exactly as the same rows would through `append`.
            MutationError::InvalidNewRows(b) => InfinoError::from(b),
            // Matches the read path: a purged table's name resolves to nothing.
            MutationError::TableGone => InfinoError::NotFound(msg),
            _ => InfinoError::Backend(msg),
        }
    }
}

impl From<MutationCommitError> for InfinoError {
    fn from(e: MutationCommitError) -> Self {
        if let Some(msg) = e.over_budget() {
            return InfinoError::OverBudget(msg.to_string());
        }
        if e.is_conflict() {
            return InfinoError::Conflict(e.to_string());
        }
        if e.is_permission_denied() {
            return InfinoError::PermissionDenied(e.to_string());
        }
        // `Supertable::append` lands here, so this is the arm that decides what
        // appending to a purged table reports. Narrow on purpose: every other
        // append-flush failure keeps its existing `Backend` shape.
        if matches!(
            &e,
            MutationCommitError::AppendFlush(SupertableBuildError::TableGone)
        ) {
            return InfinoError::NotFound(e.to_string());
        }
        // A buffer the table's schema refuses at commit (a peer froze a
        // column in another type first) classifies as the same batch would
        // through a synchronous append.
        if let MutationCommitError::AppendFlush(SupertableBuildError::Schema(schema)) = &e {
            return InfinoError::from(SupertableBuildError::Schema(schema.clone()));
        }
        InfinoError::Backend(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use parquet::errors::ParquetError;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::StorageError,
        supertable::wal::{
            WalStoreError,
            pipeline::{AppendPhaseError, TombstonePhaseError},
        },
    };

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
        assert_eq!(InfinoError::Conflict("t".into()).to_string(), "conflict: t");
        assert_eq!(InfinoError::Backend("t".into()).to_string(), "backend: t");
        assert_eq!(InfinoError::Config("t".into()).to_string(), "config: t");
    }

    #[test]
    fn with_context_prefixes_operation_and_table() {
        let err = InfinoError::NotFound("posts".into()).with_context("open_table", Some("posts"));
        assert_eq!(err.to_string(), "not found: open_table(posts): posts");

        let err = InfinoError::Cardinality("mismatch".into()).with_context("update", None);
        assert_eq!(err.to_string(), "cardinality: update: mismatch");

        let err = InfinoError::Conflict("lost the CAS".into()).with_context("delete", None);
        assert_eq!(err.to_string(), "conflict: delete: lost the CAS");
    }

    #[test]
    fn from_storage_error_maps_each_variant() {
        assert!(matches!(
            InfinoError::from(StorageError::NotFound { uri: "u".into() }),
            InfinoError::NotFound(_)
        ));
        assert!(matches!(
            InfinoError::from(StorageError::PreconditionFailed { uri: "u".into() }),
            InfinoError::Conflict(_)
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
    fn from_query_and_build_errors() {
        assert!(matches!(
            InfinoError::from(QueryError::InvalidQuery("p".into())),
            InfinoError::Query(_)
        ));
        // A budget refusal keeps its own variant rather than collapsing to Query.
        assert!(matches!(
            InfinoError::from(QueryError::OverBudget("b".into())),
            InfinoError::OverBudget(_)
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
    fn refused_credentials_route_to_permission_denied_through_every_wrapper() {
        // The condition a caller reacts to by supplying fresh credentials, so
        // it must not arrive as a generic Io/Backend/Query fault on any path.
        let denied = || StorageError::PermissionDenied { uri: "u".into() };

        // Direct storage op.
        assert!(matches!(
            InfinoError::from(denied()),
            InfinoError::PermissionDenied(_)
        ));
        // Manifest / part load — otherwise a retryable Io.
        assert!(matches!(
            InfinoError::from(ManifestLoadError::Storage(denied())),
            InfinoError::PermissionDenied(_)
        ));
        // Open, build, and commit paths — otherwise Backend or Schema.
        assert!(matches!(
            InfinoError::from(OpenError::Storage(denied())),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            InfinoError::from(SupertableBuildError::StorageConstruction(denied())),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            InfinoError::from(SupertableCommitError::Storage(denied())),
            InfinoError::PermissionDenied(_)
        ));
        // Mutation and its commit wrapper — otherwise Backend.
        assert!(matches!(
            InfinoError::from(MutationError::Storage(denied())),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationCommitError::AppendFlush(
                SupertableBuildError::StorageConstruction(denied())
            )),
            InfinoError::PermissionDenied(_)
        ));
        // Query path — otherwise a caller-fault Query (a 400 at an HTTP
        // boundary), which is the one mislabel that hides the real cause.
        assert!(matches!(
            InfinoError::from(QueryError::PermissionDenied("q".into())),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            InfinoError::from(QueryError::ManifestLoad(ManifestLoadError::Storage(
                denied()
            ))),
            InfinoError::PermissionDenied(_)
        ));
    }

    #[test]
    fn an_ordinary_storage_fault_is_still_io_not_permission_denied() {
        // The near miss: a permanent storage fault is not a credential
        // problem, and fresh credentials would not fix it.
        assert!(matches!(
            InfinoError::from(StorageError::Permanent {
                uri: "u".into(),
                source: "bad region".into(),
            }),
            InfinoError::Io(_)
        ));
    }

    /// `Query` is the caller's mistake alone. What the engine fails at in the
    /// middle of a query maps elsewhere, so a caller can tell "fix the query"
    /// from "the engine failed" without reading the message.
    #[test]
    fn a_query_error_is_the_callers_only_when_the_request_is_wrong() {
        let ordinary_storage_fault = || StorageError::Permanent {
            uri: "u".into(),
            source: "bad region".into(),
        };
        // The request itself is wrong.
        assert!(matches!(
            InfinoError::from(QueryError::InvalidQuery("unknown vector column".into())),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            InfinoError::from(QueryError::InvalidQuery("p".into())),
            InfinoError::Query(_)
        ));
        // A read failed: retrying can succeed.
        assert!(matches!(
            InfinoError::from(QueryError::Store("s".into())),
            InfinoError::Io(_)
        ));
        assert!(matches!(
            InfinoError::from(QueryError::Parquet("p".into())),
            InfinoError::Io(_)
        ));
        // A manifest load answers the same mid-query as it does on open.
        assert!(matches!(
            InfinoError::from(QueryError::ManifestLoad(ManifestLoadError::Storage(
                ordinary_storage_fault()
            ))),
            InfinoError::Io(_)
        ));
        assert!(matches!(
            InfinoError::from(QueryError::ManifestLoad(ManifestLoadError::PointerVanished)),
            InfinoError::NotFound(_)
        ));
        // The engine's own invariants.
        assert!(matches!(
            InfinoError::from(QueryError::Internal("_id column missing".into())),
            InfinoError::Backend(_)
        ));
        // The budget refusal keeps its own, already labelled message.
        assert_eq!(
            InfinoError::from(QueryError::OverBudget("during scan, over".into())).to_string(),
            "over budget: during scan, over"
        );
        // The message is the internal error's, unchanged.
        assert_eq!(
            InfinoError::from(QueryError::Store("s".into())).to_string(),
            "io: superfile store error during query: s"
        );
    }

    /// DataFusion's own `Execution` is the caller's only while planning; a
    /// broken invariant and our own errors keep their answer either way.
    #[test]
    fn an_execution_error_is_the_callers_only_while_planning() {
        let execution = DataFusionError::Execution("bad argument".into());
        assert!(matches!(
            datafusion_planning_error(&execution),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            datafusion_error(&execution),
            InfinoError::Backend(_)
        ));
        let invariant = DataFusionError::Internal("bug".into());
        assert!(matches!(
            datafusion_planning_error(&invariant),
            InfinoError::Backend(_)
        ));
        let ours = DataFusionError::from(QueryError::Internal("bug".into()));
        assert!(matches!(
            datafusion_planning_error(&ours),
            InfinoError::Backend(_)
        ));
    }

    /// A DataFusion failure maps by what caused it, not by DataFusion's own
    /// variant alone: our errors cross a plan typed and keep their answer.
    #[test]
    fn a_datafusion_failure_maps_by_its_cause() {
        let ours = |e: QueryError| DataFusionError::from(e);
        // Our own error inside the plan decides, with its own message.
        let err = datafusion_error(&ours(QueryError::Store("bucket timed out".into())));
        assert!(
            matches!(&err, InfinoError::Io(m) if m == "superfile store error during query: bucket timed out"),
            "{err:?}"
        );
        assert!(matches!(
            datafusion_error(&ours(QueryError::InvalidQuery("no full-text index".into()))),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            datafusion_error(&ours(QueryError::Internal("_id column missing".into()))),
            InfinoError::Backend(_)
        ));
        // Wrapped by DataFusion on the way out, it still decides.
        assert!(matches!(
            datafusion_error(&ours(QueryError::Store("s".into())).context("scan")),
            InfinoError::Io(_)
        ));
        // Storage under DataFusion's own variants is a failed read.
        assert!(matches!(
            datafusion_error(&DataFusionError::IoError(io::Error::other("reset"))),
            InfinoError::Io(_)
        ));
        // DataFusion's own classes.
        let plan = |m: &str| DataFusionError::Plan(m.into());
        assert!(matches!(
            datafusion_error(&plan("No field named ghost")),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            datafusion_error(&DataFusionError::ArrowError(
                Box::new(ArrowError::DivideByZero),
                None
            )),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            datafusion_error(&DataFusionError::ResourcesExhausted("spill".into())),
            InfinoError::OverBudget(_)
        ));
        assert!(matches!(
            datafusion_error(&DataFusionError::NotImplemented("LATERAL".into())),
            InfinoError::Unsupported(_)
        ));
        // Mixed or ours: counted as ours, so it is logged and looked at.
        for e in [
            DataFusionError::Execution("Partition 3 not found".into()),
            DataFusionError::Internal("bug".into()),
        ] {
            assert!(
                matches!(datafusion_error(&e), InfinoError::Backend(_)),
                "{e:?}"
            );
        }
    }

    /// The chain branches: refused credentials, a failed read, a store that
    /// cannot do an operation at all, our error under arrow and parquet
    /// wrappers, and another crate's error a DataFusion function returned.
    #[test]
    fn a_datafusion_failure_is_classified_through_its_whole_chain() {
        let external = |e: Box<dyn Error + Send + Sync>| DataFusionError::External(e);
        assert!(matches!(
            datafusion_error(&external(Box::new(StorageError::PermissionDenied {
                uri: "u".into()
            }))),
            InfinoError::PermissionDenied(_)
        ));
        assert!(matches!(
            datafusion_error(&DataFusionError::ObjectStore(Box::new(
                ObjectStoreError::Generic {
                    store: "s3",
                    source: "connection reset".into(),
                }
            ))),
            InfinoError::Io(_)
        ));
        // A store refusing an operation outright is not a read that failed.
        assert!(matches!(
            datafusion_error(&DataFusionError::ObjectStore(Box::new(
                ObjectStoreError::NotImplemented {
                    operation: "put".into(),
                    implementer: "store".into(),
                }
            ))),
            InfinoError::Backend(_)
        ));
        // Our error still decides under arrow's and parquet's own wrappers.
        let ours = || Box::new(QueryError::Store("s".into()));
        assert!(matches!(
            datafusion_error(&DataFusionError::ArrowError(
                Box::new(ArrowError::ExternalError(ours())),
                None
            )),
            InfinoError::Io(_)
        ));
        assert!(matches!(
            datafusion_error(&DataFusionError::ParquetError(Box::new(
                ParquetError::External(ours())
            ))),
            InfinoError::Io(_)
        ));
        // Another crate's error, returned by a DataFusion function rejecting
        // the caller's argument (an invalid regex), is the caller's.
        assert!(matches!(
            datafusion_error(&external("regex parse error".into())),
            InfinoError::Query(_)
        ));
    }

    /// A predicate DataFusion pushed into the parquet scan comes back as text.
    /// One that failed on the caller's values is theirs; any other is ours.
    #[test]
    fn a_pushed_down_predicate_that_failed_on_the_data_is_the_callers() {
        let pushed_down = |inner: &str| {
            DataFusionError::ParquetError(Box::new(ParquetError::External(Box::new(
                ArrowError::ComputeError(format!("{PUSHED_DOWN_PREDICATE_FAILED}{inner}")),
            ))))
        };
        assert!(matches!(
            datafusion_error(&pushed_down(
                r#"ArrowError(CastError("Cannot cast string 'alpha'"), None)"#
            )),
            InfinoError::Query(_)
        ));
        assert!(matches!(
            datafusion_error(&pushed_down(r#"Execution("Partition 3 not found")"#)),
            InfinoError::Backend(_)
        ));
        // A parquet failure that is not a predicate is a corrupt or unread file.
        assert!(matches!(
            datafusion_error(&DataFusionError::ParquetError(Box::new(
                ParquetError::General("bad footer".into())
            ))),
            InfinoError::Backend(_)
        ));
    }

    /// Another writer holding the table's writer slot is the same retryable
    /// condition as a lost commit race, not a schema problem.
    #[test]
    fn a_taken_writer_slot_is_a_conflict() {
        assert!(matches!(
            InfinoError::from(SupertableBuildError::SupertableInUse),
            InfinoError::Conflict(_)
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
    fn manifest_pointer_vanished_is_not_found_but_a_storage_fault_is_retryable_io() {
        // A dropped-and-purged pointer is a hard "gone" — NotFound.
        assert!(matches!(
            InfinoError::from(ManifestLoadError::PointerVanished),
            InfinoError::NotFound(_)
        ));
        // A storage fault reading the manifest is transient I/O, so a caller
        // can retry — Io (a retryable status at the serving layer), not a hard
        // backend fault.
        assert!(matches!(
            InfinoError::from(ManifestLoadError::Storage(
                StorageError::TransientExhausted {
                    uri: "p".into(),
                    source: "blip".into(),
                }
            )),
            InfinoError::Io(_)
        ));
    }

    #[test]
    fn over_budget_routes_through_wrappers() {
        // A budget refusal nested under a wrapper (here the commit's
        // append-flush phase) still routes to OverBudget: each wrapper's
        // over_budget() delegates to the inner error's.
        let nested =
            MutationCommitError::AppendFlush(SupertableBuildError::OverBudget("deep".into()));
        assert!(matches!(
            InfinoError::from(nested),
            InfinoError::OverBudget(_)
        ));
        // A non-budget error in the same wrapper stays a generic backend error.
        assert!(matches!(
            InfinoError::from(MutationCommitError::AppendFlush(
                SupertableBuildError::NoDocsToBuild
            )),
            InfinoError::Backend(_)
        ));
    }

    /// Every CAS-loss shape a public mutation can hit must arrive as the
    /// retryable `Conflict`, not as an opaque `Backend`. One assertion per
    /// path a caller can actually reach:
    ///
    /// - `append`  → append flush → manifest OCC exhausted;
    /// - `delete`  → WAL state-doc CAS lost;
    /// - `delete`  → tombstone-sidecar CAS budget exhausted;
    /// - `update`  → append phase's manifest commit lost the race.
    #[test]
    fn cas_loss_maps_to_conflict_on_every_mutation_path() {
        // The commit layer's own OCC exhaustion.
        assert!(matches!(
            InfinoError::from(SupertableCommitError::WriteContentionExhausted),
            InfinoError::Conflict(_)
        ));
        // Commit → build conversion keeps the contention typed rather than
        // stringifying it into `Store`, which is what let it read as a
        // backend fault before.
        assert!(matches!(
            SupertableBuildError::from(SupertableCommitError::WriteContentionExhausted),
            SupertableBuildError::WriteContention
        ));
        // `append`: writer flush → commit → OCC exhausted.
        assert!(matches!(
            InfinoError::from(MutationCommitError::AppendFlush(
                SupertableBuildError::WriteContention
            )),
            InfinoError::Conflict(_)
        ));
        // `delete`: the WAL state doc lost its CAS mid-commit.
        assert!(matches!(
            InfinoError::from(MutationCommitError::PartialCommit {
                committed_wal_ids: Vec::new(),
                committed: 0,
                total: 1,
                cause: Box::new(MutationError::WalStore(WalStoreError::CasFailed {
                    path: "wal/mutations/1.json".into()
                })),
            }),
            InfinoError::Conflict(_)
        ));
        // `delete`: the per-superfile tombstone sidecar CAS budget ran out.
        assert!(matches!(
            InfinoError::from(MutationError::TombstonePhase(
                TombstonePhaseError::CasRetryExhausted {
                    superfile_id: Uuid::nil(),
                    attempts: 8,
                }
            )),
            InfinoError::Conflict(_)
        ));
        // `update`: the append phase's manifest commit lost the race.
        assert!(matches!(
            InfinoError::from(MutationError::AppendPhase(
                AppendPhaseError::ManifestCommit(Box::new(
                    SupertableCommitError::WriteContentionExhausted
                ))
            )),
            InfinoError::Conflict(_)
        ));
        // A raw storage precondition failure anywhere under a mutation.
        assert!(matches!(
            InfinoError::from(MutationError::Storage(StorageError::PreconditionFailed {
                uri: "u".into()
            })),
            InfinoError::Conflict(_)
        ));
        // Open bootstraps through the same CAS-fenced commit.
        assert!(matches!(
            InfinoError::from(OpenError::Commit(
                SupertableCommitError::WriteContentionExhausted
            )),
            InfinoError::Conflict(_)
        ));
    }

    /// The classifier has to stay narrow: failures that retrying cannot fix
    /// must keep their existing variants.
    #[test]
    fn non_cas_failures_are_not_conflicts() {
        // A duplicate WAL id is a create collision, not a lost race.
        assert!(matches!(
            InfinoError::from(MutationError::WalStore(WalStoreError::AlreadyExists {
                path: "wal/mutations/1.json".into()
            })),
            InfinoError::Backend(_)
        ));
        assert!(matches!(
            InfinoError::from(MutationError::TombstonePhase(
                TombstonePhaseError::IdLookupFailed {
                    targets: "7".into(),
                    message: "boom".into(),
                }
            )),
            InfinoError::Backend(_)
        ));
        assert!(matches!(
            InfinoError::from(SupertableCommitError::Encode("e".into())),
            InfinoError::Backend(_)
        ));
        // A vanished pointer still outranks the conflict check.
        assert!(matches!(
            InfinoError::from(SupertableCommitError::PointerVanished),
            InfinoError::NotFound(_)
        ));
    }

    #[test]
    fn from_mutation_error_maps_each_arm() {
        assert!(matches!(
            InfinoError::from(MutationError::PredicateEval(QueryError::InvalidQuery(
                "p".into()
            ))),
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
            InfinoError::from(MutationError::NoStorageAttached),
            InfinoError::Backend(_)
        ));
    }
}
