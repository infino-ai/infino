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

use std::path::PathBuf;

use thiserror::Error;

use crate::storage::StorageError;
use crate::superfile::error::BuildError as SuperfileBuildError;
use crate::supertable::ManifestLoadError;

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
        "FTS columns declared but no tokenizer supplied; tokenizer is required iff fts_columns is non-empty"
    )]
    MissingTokenizer,

    #[error("input RecordBatch schema does not match the supertable's declared schema")]
    BatchSchemaMismatch,

    #[error("error from underlying superfile layer: {0}")]
    Superfile(#[from] SuperfileBuildError),

    #[error(
        "another SupertableWriter is already outstanding for this Supertable; \
         drop it before acquiring a new one"
    )]
    SupertableInUse,

    #[error("superfile store: {0}")]
    Store(String),

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

    /// Manifest error
    #[error("manifest load error: {0}")]
    ManifestLoadError(#[from] ManifestLoadError),

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

    /// Manifest list parse failed.
    #[error("manifest list parse failed")]
    ManifestListParse(String),

    /// Manifest load error.
    #[error("manifest load error")]
    ManifestLoadError(#[from] ManifestLoadError),

    /// Manifest part load or parse failed during open or
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

/// Errors raised by [`crate::supertable::Supertable::compact`].
#[derive(Debug, thiserror::Error)]
pub enum CompactionError {
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
}

/// Errors raised by query-time methods on [`crate::supertable::Supertable`]
/// (`query_sql`; future: `bm25_search`, `vector_search`).
///
/// Each variant carries a stringified source — DataFusion's error
/// types are not in the supertable's public dependency surface, so
/// we don't propagate them as `#[from]`. Callers get the formatted
/// message; structured introspection isn't a v1 concern. When the
/// SQL surface gains a manifest-level skip planner, it'll get its
/// own variant to distinguish "DataFusion failed" from "store
/// failed mid-scan".
#[derive(Debug, Error)]
pub enum QueryError {
    #[error("superfile store error during query: {0}")]
    Store(String),

    #[error("error reading parquet bytes during scan: {0}")]
    Parquet(String),

    #[error("DataFusion failed to plan the query: {0}")]
    Plan(String),

    #[error("DataFusion failed to execute the query: {0}")]
    Execute(String),
}
