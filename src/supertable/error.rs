// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Typed errors for the supertable layer.
//!
//! Mirrors `superfile::error::BuildError` in shape — the
//! supertable's options-validation rules are a strict superset of
//! the superfile's, so most variants either parallel a superfile
//! variant or convert from one. The only genuinely supertable-
//! specific shapes are the `VectorColumnNotFixedSizeList` /
//! `VectorColumnDimMismatch` / `VectorColumnHasNulls` variants
//! that arise because supertable's schema includes vector columns
//! as `FixedSizeList<Float32>` (vs superfile, where vectors are
//! out-of-band entirely).

use std::{error::Error, fmt::Display, path::PathBuf};

use datafusion::error::DataFusionError;
use thiserror::Error;

use crate::{
    storage::{StorageError, error_chain, permission_denied_in_chain},
    superfile::error::{BuildError as SuperfileBuildError, FtsError, ReadError, VectorError},
    supertable::{ManifestLoadError, manifest::part},
};

/// Errors raised when constructing or operating against a
/// `SupertableOptions` / `SupertableWriter`.
#[derive(Debug, Error)]
pub enum BuildError {
    #[error("no documents to build")]
    NoDocsToBuild,

    #[error("schema is missing the declared id_column {0:?}")]
    MissingIdColumn(String),

    #[error("id_column {0:?} must be Decimal128(38, 0); found {1}")]
    IdColumnWrongType(String, String),

    #[error(
        "user schema must not contain a column named {0:?} — \
         that name is reserved for the supertable-managed id column"
    )]
    IdColumnReserved(String),

    #[error("FTS column {column:?} not found in schema")]
    FtsColumnMissing { column: String },

    #[error("FTS column {column:?} must be LargeUtf8; found {actual}")]
    FtsColumnMustBeLargeUtf8 { column: String, actual: String },

    /// A column declared BM25 parameters outside their valid ranges.
    /// Caught before anything is written: the build bakes the
    /// block-max bounds with these, so a nonsense pair would otherwise
    /// surface much later as strange scores rather than as a rejected
    /// table.
    #[error(
        "FTS column {column:?}: BM25 k1 must be finite and > 0, b must be finite and in [0, 1]; \
         got k1={k1}, b={b}"
    )]
    FtsBm25ParamsOutOfRange { column: String, k1: f32, b: f32 },

    #[error("vector column {column:?} not found in schema")]
    VectorColumnMissing { column: String },

    #[error("vector column {column:?} must be FixedSizeList<Float32, {dim}>; found {actual}")]
    VectorColumnNotFixedSizeList {
        column: String,
        dim: usize,
        actual: String,
    },

    #[error(
        "vector column {column:?} declares dim={expected}; \
         schema FixedSizeList list_size is {actual}"
    )]
    VectorColumnDimMismatch {
        column: String,
        expected: usize,
        actual: usize,
    },

    #[error(
        "vector column {column:?} contains null entries at row offsets {first_nulls:?}; \
         null vectors are not permitted in v1"
    )]
    VectorColumnHasNulls {
        column: String,
        first_nulls: Vec<usize>,
    },

    #[error("vector column {column:?} declares dim={dim}; must be in [16, 4096]")]
    VectorDimOutOfRange { column: String, dim: usize },

    #[error("logical name {0:?} duplicated across fts_columns and vector_columns")]
    DuplicateLogicalName(String),

    #[error("user column name {0:?} contains reserved \\x1F separator")]
    ReservedSeparatorInColumnName(String),

    #[error("user column name {0:?} starts with reserved prefix 'inf.'")]
    ReservedPrefixInColumnName(String),

    #[error(
        "FTS column {column:?}: unknown analyzer {analyzer:?} (valid: \
         \"ascii_lower\", \"standard\")"
    )]
    UnknownAnalyzer { column: String, analyzer: String },

    #[error("input RecordBatch schema does not match the supertable's declared schema")]
    BatchSchemaMismatch,

    #[error("error from underlying superfile layer: {0}")]
    Superfile(#[from] SuperfileBuildError),

    /// Ingest build refused: would cross the connection memory budget. The
    /// string is already labelled ("during ingest, ..."); routes to
    /// `InfinoError::OverBudget` via [`BuildError::over_budget`].
    #[error("{0}")]
    OverBudget(String),

    #[error(
        "another SupertableWriter is already outstanding for this Supertable; \
         drop it before acquiring a new one"
    )]
    SupertableInUse,

    #[error("superfile store: {0}")]
    Store(String),

    /// The storage backend refused the credentials in use. Carried as its own
    /// variant rather than folded into [`Self::Store`] for the same reason as
    /// [`Self::TableGone`] and [`Self::WriteContention`]: a stringified error
    /// can't be matched on, and the public mapping must report refused
    /// credentials rather than a backend fault. See
    /// `From<CommitError> for BuildError`.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// The table was dropped and purged while this handle was open, so the
    /// commit had no pointer to fence against. Carried as its own variant
    /// rather than folded into [`Self::Store`] so the public mapping can
    /// report a missing table instead of a backend fault — see
    /// [`CommitError::PointerVanished`] and `From<BuildError> for InfinoError`.
    #[error("table was dropped and purged while this handle was open")]
    TableGone,

    /// A concurrent writer won the manifest CAS and the commit's retry
    /// budget ran out. Carried as its own variant rather than folded into
    /// [`Self::Store`] — a stringified error can't be matched on, and the
    /// public mapping needs to report a retryable conflict rather than a
    /// backend fault. See [`CommitError::WriteContentionExhausted`] and
    /// `From<BuildError> for InfinoError`.
    #[error("write contention: a concurrent writer won the commit race")]
    WriteContention,

    #[error("rayon thread pool creation failed: {0}")]
    ThreadPoolCreation(String),

    #[error("error reading the just-built superfile during commit: {0}")]
    ReadAfterCommit(String),

    /// Storage backend construction failed (auth handshake on
    /// S3, invalid endpoint, region mismatch, LocalFS root not
    /// writable). Source chain preserved so callers can match
    /// on `StorageError::Permanent` vs `::TransientExhausted`
    /// for retry semantics.
    #[error("storage construction failed: {0}")]
    StorageConstruction(#[from] StorageError),

    /// Disk-cache root directory exists but isn't writable, or
    /// can't be created. Distinct from `StorageConstruction`
    /// because the disk cache is a local-only concern that
    /// doesn't go through the storage provider.
    #[error("disk cache root unwritable: {0}")]
    DiskCacheRootUnwritable(PathBuf),

    /// `partition_strategy` names a column the schema doesn't
    /// have. Construction-time check — never silently falls
    /// back. Caller fixes config or schema.
    #[error("partition column missing in schema: {0}")]
    PartitionColumnMissing(String),
}

impl BuildError {
    /// The over-budget message if this is a budget refusal, else `None`.
    pub(crate) fn over_budget(&self) -> Option<&str> {
        match self {
            BuildError::OverBudget(m) => Some(m),
            _ => None,
        }
    }

    /// True when the build failed because a concurrent writer won a
    /// compare-and-set race, so reissuing against fresh state can succeed.
    pub(crate) fn is_conflict(&self) -> bool {
        match self {
            // Another writer holds this table's single writer slot: the same
            // retry-after-the-other-writer answer as a lost commit race.
            BuildError::WriteContention | BuildError::SupertableInUse => true,
            BuildError::StorageConstruction(e) => e.is_conflict(),
            _ => false,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            BuildError::PermissionDenied(_) => true,
            BuildError::StorageConstruction(e) => e.is_permission_denied(),
            _ => false,
        }
    }
}

impl From<CommitError> for BuildError {
    /// Commit failures reach the build path as `Store` carrying the message —
    /// except a vanished pointer and a lost commit race, which keep their own
    /// variants so the public mapping can report a missing table or a
    /// retryable conflict. A stringified error cannot be matched on,
    /// and the append path converts here before any caller sees it.
    fn from(e: CommitError) -> Self {
        match e {
            CommitError::PointerVanished => BuildError::TableGone,
            other if other.is_conflict() => BuildError::WriteContention,
            other if other.is_permission_denied() => {
                BuildError::PermissionDenied(other.to_string())
            }
            other => BuildError::Store(other.to_string()),
        }
    }
}

/// Errors raised by the supertable's commit path — building +
/// publishing a new manifest version. Stable public surface;
/// downstream callers may match on specific variants for
/// recovery (e.g., `WriteContentionExhausted` from the OCC
/// retry loop, `SuperfileSpansPartition` from the
/// partition-assignment validation).
#[derive(Debug, Error)]
pub enum CommitError {
    /// Storage backend returned an error during commit.
    #[error("storage error during commit: {0}")]
    Storage(#[from] crate::storage::StorageError),

    /// Below-storage validation (options + schema) failed.
    #[error("build error during commit")]
    Build(#[from] BuildError),

    /// ManifestSnapshot error
    #[error("manifest error: {0}")]
    ManifestError(#[from] ManifestError),

    /// Failed to encode a manifest part or list to its wire
    /// format. Indicates a programmer error (e.g., a
    /// non-serializable scalar value in a manifest list), not
    /// a transient failure.
    #[error("manifest encode failed: {0}")]
    Encode(String),

    /// Pointer file on storage is malformed (truncated,
    /// missing required fields, unexpected key).
    #[error("pointer file parse failed: {0}")]
    PointerParse(String),

    /// OCC retry budget exhausted on a contended commit.
    /// Reserved variant — the current writer doesn't retry,
    /// but the public surface carries this so adding the retry
    /// loop later is non-breaking.
    #[error("write contention exhausted retries")]
    WriteContentionExhausted,

    /// The pointer this commit would have fenced against is gone: the table
    /// was dropped and purged while this handle stayed open. Not retryable.
    #[error("manifest pointer was deleted while this handle was open")]
    PointerVanished,

    /// An input's tombstone sidecar changed under the seal this commit holds,
    /// so a writer landed a bit on a superfile the commit is about to remove.
    /// Retryable in the same sense as a lost pointer CAS: nothing was
    /// published, and the next attempt re-resolves — dropping the job whose
    /// seal moved and committing the rest.
    #[error("input {superfile_id} changed under this commit's seal")]
    InputsChanged { superfile_id: uuid::Uuid },
}

impl CommitError {
    /// True when the commit failed because a concurrent writer won the
    /// pointer / part CAS, so reissuing against fresh state can succeed.
    ///
    /// A raw [`StorageError::PreconditionFailed`] can still reach here from a
    /// sub-write that skipped the commit module's `translate_contention`, so
    /// both shapes are classified together.
    pub(crate) fn is_conflict(&self) -> bool {
        match self {
            // Both are a race lost to another writer with nothing published.
            CommitError::WriteContentionExhausted | CommitError::InputsChanged { .. } => true,
            CommitError::Storage(e) => e.is_conflict(),
            CommitError::Build(b) => b.is_conflict(),
            _ => false,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            CommitError::Storage(e) => e.is_permission_denied(),
            CommitError::Build(b) => b.is_permission_denied(),
            _ => false,
        }
    }
}

#[derive(Debug, Error)]
pub enum ManifestError {
    /// A superfile's column range spans multiple
    /// partitions under the configured `PartitionStrategy`.
    /// For `TimeRange` / `ColumnRange`, the superfile's
    /// `(min, max)` straddles a bucket boundary. For `Hash`,
    /// the superfile's `partition_hint` is unset — the writer
    /// didn't pre-shard.
    ///
    /// Single-bucket Hash strategies (`n_buckets == 1`) are
    /// special-cased to bypass this check, since every
    /// possible value hashes to bucket 0.
    #[error("superfile spans partition boundary: {detail}")]
    SuperfileSpansPartition { detail: String },
    /// A superfile entry reached `update()` already carrying a
    /// `partition_key`. Entries must arrive unstamped: the key is
    /// derived from the strategy at commit time. A non-empty key means
    /// an earlier stage already stamped it, and committing would
    /// silently overwrite that assignment.
    #[error("superfile entry already partitioned: {detail}")]
    EntryAlreadyPartitioned { detail: String },
    /// A commit tried to add a superfile the manifest already lists. A
    /// superfile listed twice has its rows read twice, by queries and by
    /// merges.
    #[error("superfile {superfile_id} is already listed in the manifest")]
    SuperfileAlreadyListed { superfile_id: uuid::Uuid },
    /// Manifest load error
    #[error("manifest load error: {0}")]
    ManifestLoadError(#[from] ManifestLoadError),
    /// Unknown part id
    #[error("unknown part id: {0}")]
    UnknownPartId(part::PartId),
}

/// Errors raised by [`crate::supertable::Supertable::open`] and
/// [`crate::supertable::Supertable::refresh`].
///
/// Stable public surface; downstream callers may match on
/// specific variants for recovery (e.g., `PointerUnreadable`
/// for the open-or-create pattern: caller falls back to
/// `Supertable::create`).
#[derive(Debug, Error)]
pub enum OpenError {
    /// Pointer file at `_supertable/current` doesn't exist or
    /// can't be read. Matches the "open-or-create" trigger:
    /// callers wanting that semantic catch this variant and
    /// fall back to [`crate::supertable::Supertable::create`].
    #[error("pointer file missing or unreadable")]
    PointerUnreadable(#[source] crate::storage::StorageError),

    /// ManifestSnapshot list parse failed.
    #[error("manifest list parse failed")]
    ManifestListParse(String),

    /// ManifestSnapshot load error.
    #[error("manifest load error: {0}")]
    ManifestLoadError(#[from] ManifestLoadError),

    /// ManifestSnapshot part load or parse failed during open or
    /// refresh.
    #[error("manifest part load failed: {part_id}")]
    ManifestPartLoad {
        part_id: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Content-hash mismatch on a loaded manifest part — the
    /// bytes returned by storage don't match the hash recorded
    /// in the manifest list. Either storage corruption or a
    /// serious bug; never auto-refetched (treated as a
    /// caller-visible failure so the inconsistency can't be
    /// papered over silently).
    #[error("content-hash mismatch: expected {expected}, got {actual}")]
    ContentHashMismatch { expected: String, actual: String },

    /// Storage backend returned an unexpected error during
    /// open or refresh.
    #[error("storage error during open")]
    Storage(#[from] crate::storage::StorageError),

    /// Configuration error — e.g., calling
    /// `Supertable::open` on options with no storage backend
    /// attached.
    #[error("build error during open")]
    Build(#[from] BuildError),

    /// Pointer-file or commit-error surfaced through the open
    /// path.
    #[error("commit error during open")]
    Commit(#[from] CommitError),
}

impl OpenError {
    /// True when the open lost a race against a concurrent writer — the
    /// bootstrap commit an open-or-create performs is CAS-fenced like any
    /// other, so a peer creating the same table first lands here.
    pub(crate) fn is_conflict(&self) -> bool {
        match self {
            OpenError::PointerUnreadable(e) | OpenError::Storage(e) => e.is_conflict(),
            OpenError::Build(b) => b.is_conflict(),
            OpenError::Commit(c) => c.is_conflict(),
            _ => false,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            OpenError::PointerUnreadable(e) | OpenError::Storage(e) => e.is_permission_denied(),
            OpenError::Build(b) => b.is_permission_denied(),
            OpenError::Commit(c) => c.is_permission_denied(),
            OpenError::ManifestLoadError(e) => e.is_permission_denied(),
            // The part loader boxes its source, so read the chain.
            OpenError::ManifestPartLoad { source, .. } => {
                permission_denied_in_chain(source.as_ref())
            }
            _ => false,
        }
    }
}

/// Failures from [`crate::Supertable::reindex`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ReindexError {
    /// No durable storage backend is configured (e.g. `memory://`);
    /// a reindex rewrites committed files, so it needs one.
    #[error("reindex requires a storage backend")]
    NoStorage,
    /// Another compaction or reindex is already running in this process.
    ///
    /// Both reshape the same superfiles through the same slot, so they are
    /// serialized rather than allowed to race. This is a signal to retry
    /// later, not a failure of the migration.
    #[error("a compaction or reindex is already running")]
    AlreadyRunning,
    /// Reading a superfile to decide whether it is stale failed.
    #[error("failed to assess superfiles: {0}")]
    Assess(String),
    /// Rewriting one superfile failed. The migration stops here; the
    /// superfiles already rewritten stay rewritten, and re-running picks
    /// up what is left.
    ///
    /// The cause is carried as text rather than as the underlying error:
    /// that type is internal, and exposing it here would make every
    /// compaction failure mode part of the public surface for the sake of
    /// one message.
    #[error("failed to rewrite superfile {superfile_id}: {cause}")]
    Rewrite {
        /// The superfile whose rewrite failed.
        superfile_id: uuid::Uuid,
        /// What went wrong underneath.
        cause: String,
    },
}

/// Errors raised by [`crate::Supertable::optimize`].
#[derive(Debug, thiserror::Error)]
pub enum OptimizeError {
    /// No durable storage backend is configured (e.g. `memory://`); optimize
    /// needs one.
    #[error("optimize requires a storage backend")]
    NoStorage,
    /// A superfile selected for compaction was absent from the manifest
    /// snapshot.
    #[error("superfile {0} not found in manifest snapshot")]
    SuperfileNotFound(uuid::Uuid),
    /// Compaction produced an empty merged superfile.
    #[error("empty merged superfile")]
    EmptyMergedSuperfile,
    /// The tombstone sidecar for a superfile was already sealed by another
    /// compaction.
    #[error(
        "tombstone sidecar for {superfile_id} already sealed by compaction {existing_compaction_id}"
    )]
    SidecarConflict {
        /// The superfile whose sidecar conflicted.
        superfile_id: uuid::Uuid,
        /// The compaction that had already sealed the sidecar.
        existing_compaction_id: uuid::Uuid,
    },
    /// Sealing the compaction output failed.
    #[error("seal failed: {0}")]
    Seal(String),
    /// Building a merged superfile failed.
    #[error("failed to build superfile: {0}")]
    Build(String),
    /// Committing the compaction to the manifest failed.
    #[error("failed to commit: {0}")]
    Commit(String),
    /// Refreshing the in-memory manifest after the commit failed.
    #[error("post-commit manifest refresh failed: {0}")]
    Refresh(String),
    /// Another optimize is already running on this handle.
    #[error("optimize already in progress on this handle")]
    AlreadyRunning,
    /// The post-compaction garbage-collection step failed.
    #[error("gc failed during optimize: {0}")]
    Gc(#[from] GcError),
    /// The post-compaction WAL sweep failed.
    #[error("wal sweep failed during optimize: {0}")]
    WalGc(#[from] crate::supertable::wal::gc::GcError),
}

impl From<CompactionError> for OptimizeError {
    fn from(e: CompactionError) -> Self {
        match e {
            CompactionError::NoStorage => OptimizeError::NoStorage,
            CompactionError::SuperfileNotFound(id) => OptimizeError::SuperfileNotFound(id),
            CompactionError::EmptyMergedSuperfile => OptimizeError::EmptyMergedSuperfile,
            CompactionError::SidecarConflict {
                superfile_id,
                existing_compaction_id,
            } => OptimizeError::SidecarConflict {
                superfile_id,
                existing_compaction_id,
            },
            CompactionError::SidecarChangedUnderSeal { superfile_id } => OptimizeError::Seal(
                format!("tombstone sidecar for {superfile_id} changed under this job's seal"),
            ),
            CompactionError::SealRetriesExhausted { superfile_id } => {
                OptimizeError::Seal(format!("seal retries exhausted for {superfile_id}"))
            }
            CompactionError::Seal(s) => OptimizeError::Seal(s),
            CompactionError::Build(s) => OptimizeError::Build(s),
            CompactionError::Commit(s) => OptimizeError::Commit(s),
            CompactionError::Refresh(s) => OptimizeError::Refresh(s),
            CompactionError::AlreadyCompacting => OptimizeError::AlreadyRunning,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CompactionError {
    /// Compaction requires durable storage
    /// (needs to seal sidecars and publish the merged superfile).
    #[error("compaction requires a storage backend")]
    NoStorage,

    /// A superfile listed in a `CompactionJob` is not present in the
    /// current manifest snapshot.
    #[error("superfile {0} not found in manifest snapshot")]
    SuperfileNotFound(uuid::Uuid),

    #[error("empty merged superfile")]
    EmptyMergedSuperfile,

    /// The tombstone sidecar for `superfile_id` is already sealed by
    /// a different compaction run. Caller must drive the abandoned
    /// compaction to completion (or unwind it) before retrying.
    #[error(
        "tombstone sidecar for {superfile_id} already sealed by compaction {existing_compaction_id}"
    )]
    SidecarConflict {
        superfile_id: uuid::Uuid,
        existing_compaction_id: uuid::Uuid,
    },

    /// An input's tombstone sidecar changed after this job sealed it, so
    /// the bitmap the job holds no longer describes that superfile.
    ///
    /// Reachable because the mutation path takes over a seal it considers
    /// abandoned. Carrying the seal-time bitmap onto the output would drop
    /// whatever landed in between, so the job gives up its input instead
    /// and a later run repeats it against the current sidecar.
    #[error("tombstone sidecar for {superfile_id} changed under this job's seal")]
    SidecarChangedUnderSeal { superfile_id: uuid::Uuid },

    /// Sealing lost its CAS race to a writer on every attempt.
    ///
    /// Contention, not failure: a writer kept landing tombstone bits while
    /// this job tried to freeze the sidecar. Distinct from [`Self::Seal`],
    /// which is a storage error and means something is actually wrong.
    #[error("seal retries exhausted for {superfile_id}")]
    SealRetriesExhausted { superfile_id: uuid::Uuid },

    /// A WAL-store I/O error occurred while sealing a sidecar.
    #[error("seal failed: {0}")]
    Seal(String),

    /// Error when building the compacted superfile. Carries the
    /// rendered cause as a string so the public error does not leak the
    /// crate-internal `BuildError` type.
    #[error("failed to build superfile: {0}")]
    Build(String),

    /// Error when committing the compacted superfile. Carries the
    /// rendered cause as a string (see `Build`).
    #[error("failed to commit compaction: {0}")]
    Commit(String),

    /// Refreshing the in-memory manifest after a successful commit failed.
    #[error("post-commit manifest refresh failed: {0}")]
    Refresh(String),

    /// Another compaction is already running on this supertable handle.
    #[error("compaction already in progress on this supertable handle")]
    AlreadyCompacting,
}

/// Errors raised by [`crate::Supertable::gc`].
#[derive(Debug, thiserror::Error)]
pub enum GcError {
    /// No durable storage backend is configured (e.g. `memory://`); gc needs
    /// one.
    #[error("gc requires a storage backend")]
    NoStorage,

    /// A storage operation failed while listing or deleting objects.
    #[error("storage error during gc: {0}")]
    Storage(#[from] crate::storage::StorageError),
}

/// Errors raised by the query and search kernels.
///
/// Each variant names a cause, which decides the public [`crate::InfinoError`]
/// (see its `From<QueryError>`): the caller's mistake, a failed read, or the
/// engine breaking its own invariant. Inside a DataFusion plan a `QueryError`
/// travels as `DataFusionError::External` (see `From<QueryError> for
/// DataFusionError`), so it comes out the other side with its cause intact.
#[derive(Debug, Error)]
pub enum QueryError {
    #[error("superfile store error during query: {0}")]
    Store(String),

    #[error("error reading parquet bytes during scan: {0}")]
    Parquet(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    /// The engine broke one of its own invariants: a superfile without the
    /// `_id` column every superfile has, a hit the pipeline failed to stamp,
    /// a build that left out what it must carry. Neither the caller nor a
    /// retry can fix it; it is a bug, and maps to the public `Backend`. A
    /// caller's mistake is [`Self::InvalidQuery`], a failed read
    /// [`Self::Store`] or [`Self::Parquet`].
    #[error("failed to run the query: {0}")]
    Internal(String),

    /// DataFusion failed to plan or run a query, typed as it returned it, so
    /// the public mapping can tell a bad query from a failed read or an
    /// engine fault (`crate::error::datafusion_error`).
    #[error(transparent)]
    DataFusion(DataFusionError),

    /// A query crossed the connection memory budget. The string is already
    /// labelled with the operation; routes to `InfinoError::OverBudget` via
    /// [`QueryError::over_budget`].
    #[error("{0}")]
    OverBudget(String),

    #[error("manifest load error: {0}")]
    ManifestLoad(ManifestLoadError),

    /// The storage backend refused the credentials in use; routes to
    /// `InfinoError::PermissionDenied`. Classified where a source is about to
    /// be stringified into [`Self::Store`] or [`Self::Parquet`] (see
    /// [`Self::build`]); a typed [`Self::DataFusion`] source is checked in
    /// place, by walking its chain.
    #[error("permission denied during query: {0}")]
    PermissionDenied(String),
}

/// A `QueryError` raised inside a DataFusion plan (a table scan, a search
/// table function) crosses back to the caller as `External`, not flattened to
/// a string, so the error keeps the cause it was raised with.
impl From<QueryError> for DataFusionError {
    fn from(e: QueryError) -> Self {
        DataFusionError::External(Box::new(e))
    }
}

/// A superfile read that failed during a query, classified once for every
/// caller (the scan, the search kernels, the id lookups):
///
/// | `ReadError` | `QueryError` |
/// |---|---|
/// | over the connection's memory budget | `OverBudget` |
/// | an FTS query the column cannot answer: a phrase without positions, nothing positive to rank | `InvalidQuery`: the caller's |
/// | the store refused our credentials | `PermissionDenied` |
/// | a local doc id past the superfile's end: our bug, retrying cannot help | `Internal` |
/// | anything else | `Parquet`: a read failed |
impl From<ReadError> for QueryError {
    fn from(e: ReadError) -> Self {
        if let Some(msg) = e.over_budget() {
            return QueryError::OverBudget(msg.to_string());
        }
        if let ReadError::Fts(fts) = &e
            && matches!(
                fts.as_ref(),
                FtsError::PositionsUnavailable { .. } | FtsError::NegationOnly
            )
        {
            return QueryError::InvalidQuery(e.to_string());
        }
        if permission_denied_in_chain(&e) {
            return QueryError::PermissionDenied(e.to_string());
        }
        if matches!(e, ReadError::DocIdOutOfRange { .. }) {
            return QueryError::Internal(e.to_string());
        }
        QueryError::Parquet(e.to_string())
    }
}

/// The vector reader's own error, before it is wrapped in a [`ReadError`]:
/// classified the same way.
impl From<VectorError> for QueryError {
    fn from(e: VectorError) -> Self {
        QueryError::from(ReadError::Vector(Box::new(e)))
    }
}

impl QueryError {
    /// The engine broke its own invariant; `error` says how.
    pub(crate) fn internal(error: impl Display) -> Self {
        QueryError::Internal(error.to_string())
    }

    /// A storage or cache failure under a query: [`Self::Store`], or
    /// [`Self::PermissionDenied`] when the store refused our credentials. A
    /// superfile [`ReadError`] goes through `From<ReadError>` instead, which
    /// also catches a budget refusal and the caller's own mistakes.
    pub(crate) fn store(error: impl Error + 'static) -> Self {
        QueryError::build(error.to_string(), &error)
    }

    /// The over-budget message if this is a budget refusal, else `None`.
    pub(crate) fn over_budget(&self) -> Option<&str> {
        match self {
            QueryError::OverBudget(m) => Some(m),
            // Ours, carried through the plan, or the plan's own memory pool.
            QueryError::DataFusion(e) => error_chain(e)
                .find_map(|link| link.downcast_ref::<QueryError>())
                .and_then(QueryError::over_budget)
                .or(match e.find_root() {
                    DataFusionError::ResourcesExhausted(m) => Some(m.as_str()),
                    _ => None,
                }),
            _ => None,
        }
    }

    /// True when the backend refused the credentials in use.
    pub(crate) fn is_permission_denied(&self) -> bool {
        match self {
            QueryError::PermissionDenied(_) => true,
            QueryError::ManifestLoad(e) => e.is_permission_denied(),
            QueryError::DataFusion(e) => permission_denied_in_chain(e),
            _ => false,
        }
    }

    /// The tombstone cache failed to load a superfile's deletes; classified
    /// like any storage failure, labelled so the message says where.
    pub(crate) fn tombstone_cache(error: impl Error + 'static) -> Self {
        QueryError::build(format!("tombstone cache: {error}"), &error)
    }

    /// Classify a storage-backed query failure whose source is about to be
    /// stringified: refused credentials get their own variant, everything else
    /// stays a [`Self::Store`]. `message` is the text the caller would have
    /// used either way, so no message changes shape.
    pub(crate) fn build(message: String, source: &(dyn Error + 'static)) -> Self {
        if permission_denied_in_chain(source) {
            return QueryError::PermissionDenied(message);
        }
        QueryError::Store(message)
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;
    use crate::supertable::reader_cache::disk::DiskCacheError;

    /// A superfile read carries its storage error inside an `io::Error`, whose
    /// `source()` skips the error it wraps. Refused credentials must still be
    /// found there, and a budget refusal and the caller's own FTS mistake kept
    /// apart from a read that failed.
    #[test]
    fn a_read_error_is_classified_through_its_io_wrapper() {
        let in_io = |storage| ReadError::Io(io::Error::other(storage));
        assert!(matches!(
            QueryError::from(in_io(StorageError::PermissionDenied { uri: "u".into() })),
            QueryError::PermissionDenied(_)
        ));
        assert!(matches!(
            QueryError::from(in_io(StorageError::TransientExhausted {
                uri: "u".into(),
                source: "boom".into(),
            })),
            QueryError::Parquet(_)
        ));
        assert!(matches!(
            QueryError::from(VectorError::OverBudget("gate".into())),
            QueryError::OverBudget(_)
        ));
        assert!(matches!(
            QueryError::from(ReadError::Fts(Box::new(FtsError::NegationOnly))),
            QueryError::InvalidQuery(_)
        ));
        // A local doc id past the end is our bug: retrying the read cannot help.
        assert!(matches!(
            QueryError::from(ReadError::DocIdOutOfRange {
                doc_id: 9,
                n_docs: 4
            }),
            QueryError::Internal(_)
        ));
    }

    /// A budget refusal that crossed DataFusion is still one.
    #[test]
    fn a_budget_refusal_inside_datafusion_is_still_over_budget() {
        let refused = QueryError::DataFusion(DataFusionError::ResourcesExhausted("cap".into()));
        assert_eq!(refused.over_budget(), Some("cap"));
        let other = QueryError::DataFusion(DataFusionError::Execution("boom".into()));
        assert_eq!(other.over_budget(), None);
        let ours =
            QueryError::DataFusion(DataFusionError::from(QueryError::OverBudget("ours".into())));
        assert_eq!(ours.over_budget(), Some("ours"));
    }

    #[test]
    fn a_refused_credential_is_classified_through_the_disk_cache_wrapper() {
        // The read path's real shape: the query layer stringifies whatever the
        // disk cache hands it, so the classification has to read the typed
        // chain through that wrapper. Anything else on the same path stays a
        // store error.
        let refused = DiskCacheError::Storage(StorageError::PermissionDenied { uri: "u".into() });
        assert!(matches!(
            QueryError::build(refused.to_string(), &refused),
            QueryError::PermissionDenied(_)
        ));

        let transient = DiskCacheError::Storage(StorageError::TransientExhausted {
            uri: "u".into(),
            source: "boom".into(),
        });
        assert!(matches!(
            QueryError::build(transient.to_string(), &transient),
            QueryError::Store(_)
        ));
    }
}
