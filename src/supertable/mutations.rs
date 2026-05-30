//! Public types + entry points for update / delete operations.
//!
//! The buffer + commit shape called out in the plan is a
//! follow-up; this commit ships immediate-drive deletes (and
//! updates land alongside) so end-to-end mutation behaviour is
//! observable from the public API.
//!
//! ## What's here
//!
//! - [`OperationOutcome`] — per-op tally returned from each
//!   `delete()` / `update()` call. Mirrors the shape the future
//!   `CommitResult.outcomes` will carry.
//! - [`MutationError`] — typed failures surfaced at call time
//!   (schema mismatch, cardinality, cap exceeded, storage).
//!
//! The buffer-shaped `PendingDelete` / `PendingUpdate` /
//! `CommitResult` types arrive when the writer's commit-time
//! flush gets the same treatment as the append buffer.

use thiserror::Error;
use uuid::Uuid;

use crate::storage::StorageError;
use crate::supertable::QueryError;
use crate::supertable::wal::persistence::WalStoreError;
use crate::supertable::wal::pipeline::{AppendPhaseError, TombstonePhaseError};
use crate::supertable::wal::state_doc::WalId;

/// Per-call outcome from one `delete` / `update`. Same field
/// shape the eventual `CommitResult.outcomes` will carry, so
/// callers writing against this API don't need to change when
/// the buffer + commit flush lands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationOutcome {
    /// `wal_id` of the WAL that drove this mutation. The WAL is
    /// the recovery boundary: any partial-commit scenario surfaces
    /// the same id in the recovery sweep's report.
    pub wal_id: WalId,
    /// Rows the predicate resolved to at call time. For a
    /// delete this is the number of rows whose tombstone the
    /// engine will try to land; for an update, the count of
    /// rows that must equal `new_rows.num_rows()`.
    pub matched: usize,
    /// Rows whose tombstone bit landed in a per-superfile
    /// sidecar.
    pub n_tombstoned: usize,
    /// Rows the engine couldn't find at commit time — either a
    /// peer beat us to the tombstone, or compaction removed the
    /// row's superfile between resolve and tombstone. Not an
    /// error; surfaced for observability.
    pub n_not_found: usize,
}

/// Cap on the number of rows one mutation call can target.
/// Bounds memory usage in the WAL state doc (tombstone_progress
/// grows linearly with this) and bounds per-call latency.
///
/// Callers whose predicate exceeds this should narrow it and
/// reissue.
pub const MAX_TARGETS_PER_MUTATION: usize = 100_000;

/// Typed failures from `delete` / `update`. Each variant is
/// surfaced at call time; no partial state is left behind on
/// any of these paths.
#[derive(Debug, Error)]
pub enum MutationError {
    /// Predicate evaluation failed — most commonly a reference
    /// to an unknown column, but also covers DataFusion-level
    /// type errors.
    #[error("predicate evaluation failed: {0}")]
    PredicateEval(#[from] QueryError),

    /// Predicate matched more rows than [`MAX_TARGETS_PER_MUTATION`].
    /// Caller narrows the predicate and reissues.
    #[error("predicate matched {matched} rows; mutation cap is {cap}")]
    MatchCountExceedsCap { matched: usize, cap: usize },

    /// `update()` only: predicate matched a different number of
    /// rows than `new_rows` supplies. 1:1-cardinality replacement.
    #[error("cardinality mismatch: predicate matched {matched} rows; new_rows has {new_rows}")]
    CardinalityMismatch { matched: usize, new_rows: usize },

    /// `update()` only: `new_rows`'s schema doesn't match the
    /// supertable's user-facing schema.
    #[error("new_rows schema does not match the supertable's user schema: {0}")]
    SchemaMismatch(String),

    /// Supertable has no storage attached; WAL pipeline requires
    /// durable storage. In-memory-only supertables can't be
    /// mutated through this API.
    #[error("supertable has no storage attached; delete / update requires durable storage")]
    NoStorageAttached,

    /// Underlying storage error from a sidecar PUT or state-doc
    /// write.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// WAL state-doc I/O failure.
    #[error("WAL store error: {0}")]
    WalStore(#[from] WalStoreError),

    /// Append-phase failure when the engine writes the new rows
    /// into a fresh superfile (update only). Surfaced as a
    /// typed wrapper so callers can pattern-match the underlying
    /// reason.
    #[error("append phase failed: {0}")]
    AppendPhase(#[from] AppendPhaseError),

    /// Tombstone-phase failure when the engine lands the
    /// per-target bits in the sidecars.
    #[error("tombstone phase failed: {0}")]
    TombstonePhase(#[from] TombstonePhaseError),
}

/// One target reservation by the writer's update path: a fresh
/// superfile UUID + minted `_id` spans. Carried into the WAL
/// state doc so the recovery sweep can re-build the same
/// superfile on replay.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct UpdateReservation {
    pub preallocated_superfile_id: Uuid,
    pub minted_id_spans: Vec<crate::supertable::wal::state_doc::IdSpan>,
}
