// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Public types + entry points for update / delete operations.
//!
//! Mutations follow the buffer + commit shape that `append()`
//! already uses:
//!
//! 1. `update()` / `delete()` resolve the predicate against the
//!    current manifest snapshot, capture the matching `_id` set,
//!    pre-reserve any resources the WAL will need (an `_id`
//!    range + a fresh superfile UUID for updates), and stash a
//!    pending entry on the writer.
//! 2. `commit()` flushes the buffered work atomically from the
//!    caller's perspective: pending appends are written first,
//!    then each buffered update drives its WAL pipeline through
//!    append + tombstone phases, then each buffered delete
//!    drives its tombstone phase.
//!
//! Durability is the commit barrier: a writer dropped without
//! `commit()` returning `Ok` discards every buffered entry. Same
//! shape as `append()`'s buffer.
//!
//! ## What's here
//!
//! - [`PendingUpdate`] / [`PendingDelete`] — values returned from
//!   the corresponding writer entry points. Carry `matched` so the
//!   caller can decide whether to proceed; the actual `MutationStats`
//!   surfaces on the next `commit()` call.
//! - [`CommitResult`] — aggregate returned from a successful
//!   `commit()`. Contains one [`MutationStats`] per buffered
//!   mutation, in buffer order.
//! - [`CommitError`] — typed failures from `commit()`, including
//!   `PartialCommit { committed_wal_ids, cause }` for the
//!   recoverable mid-flush case.
//! - [`MutationError`] — typed failures surfaced at
//!   `update()` / `delete()` call time (schema mismatch,
//!   cardinality, cap exceeded, storage).

use thiserror::Error;

use crate::storage::StorageError;
use crate::supertable::QueryError;
use crate::supertable::error::BuildError;
use crate::supertable::wal::persistence::WalStoreError;
use crate::supertable::wal::pipeline::{AppendPhaseError, TombstonePhaseError};
use crate::supertable::wal::state_doc::WalId;

/// Per-call outcome from one `delete` / `update`. Returned by the
/// public `Supertable::update` / `Supertable::delete` (which fold
/// writer + commit) and carried, one per buffered mutation, in
/// `CommitResult.outcomes`.
///
/// The public surface is the three count accessors (`matched`,
/// `n_tombstoned`, `n_not_found`); the recovery-only `wal_id` stays
/// `pub(crate)`, so it never enters the public API.
/// `#[non_exhaustive]` keeps the type open to growth.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationStats {
    /// `wal_id` of the WAL that drove this mutation. The WAL is
    /// the recovery boundary: any partial-commit scenario surfaces
    /// the same id in the recovery sweep's report.
    pub(crate) wal_id: WalId,
    /// Rows the predicate resolved to at call time. For a
    /// delete this is the number of rows whose tombstone the
    /// engine will try to land; for an update, the count of
    /// rows that must equal `new_rows.num_rows()`.
    pub(crate) matched: usize,
    /// Rows whose tombstone bit landed in a per-superfile
    /// sidecar.
    pub(crate) n_tombstoned: usize,
    /// Rows the engine couldn't find at commit time — either a
    /// peer beat us to the tombstone, or compaction removed the
    /// row's superfile between resolve and tombstone. Not an
    /// error; surfaced for observability.
    pub(crate) n_not_found: usize,
}

impl MutationStats {
    /// Rows the predicate resolved to at call time.
    pub fn matched(&self) -> usize {
        self.matched
    }

    /// Rows whose tombstone bit landed (a delete, or an update's
    /// tombstone phase).
    pub fn n_tombstoned(&self) -> usize {
        self.n_tombstoned
    }

    /// Rows the predicate resolved but that no superfile claimed at
    /// commit time (a peer or compaction tombstoned them first). Not an
    /// error; surfaced for observability.
    pub fn n_not_found(&self) -> usize {
        self.n_not_found
    }
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

/// Value returned from [`SupertableWriter::update`]. Carries the
/// count of rows the predicate resolved to at call time so the
/// caller can decide whether to proceed to `commit()`. Captured
/// by value rather than reference because `update()` returns
/// after stashing the pending entry on the writer — the caller
/// doesn't otherwise hold a handle to that entry.
///
/// [`SupertableWriter::update`]: crate::supertable::writer::SupertableWriter::update
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingUpdate {
    /// Rows the predicate resolved to at call time. Exactly
    /// `new_rows.num_rows()` (the engine enforced the 1:1
    /// cardinality before returning).
    pub matched: usize,
}

/// Value returned from [`SupertableWriter::delete`].
///
/// [`SupertableWriter::delete`]: crate::supertable::writer::SupertableWriter::delete
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingDelete {
    /// Rows the predicate resolved to at call time. The
    /// commit-time pipeline will try to tombstone each of these;
    /// rows that no superfile claims at commit time are reported
    /// as `n_not_found` in the corresponding [`MutationStats`].
    pub matched: usize,
}

/// Aggregate result of a successful [`SupertableWriter::commit`].
/// One [`MutationStats`] per buffered update / delete, in
/// buffer order. Pending appends don't appear as outcome entries
/// — they're a separate concern from the WAL-driven mutation
/// path and surface only through the manifest swap.
///
/// [`SupertableWriter::commit`]: crate::supertable::writer::SupertableWriter::commit
#[derive(Debug, Clone)]
pub struct CommitResult {
    /// WAL ids minted for each buffered mutation, in buffer
    /// order. Equivalent to `outcomes.iter().map(|o| o.wal_id)`
    /// — exposed separately so callers can pin "did THIS WAL
    /// complete" without scanning the outcome list.
    pub wal_ids: Vec<WalId>,
    /// Per-operation outcomes, in buffer order.
    pub outcomes: Vec<MutationStats>,
}

/// Typed failures from [`SupertableWriter::commit`]. The buffered
/// append phase is one transaction (commit fails atomically if a
/// shard build fails); each buffered mutation is its own
/// recoverable boundary, so a mid-buffer failure surfaces
/// `PartialCommit` listing the WALs that did land durably.
///
/// [`SupertableWriter::commit`]: crate::supertable::writer::SupertableWriter::commit
#[derive(Debug, Error)]
pub enum CommitError {
    /// The pending-appends flush failed. No mutation WALs have
    /// been driven yet; the buffer (mutations + remaining
    /// appends) is preserved on the writer so the caller can
    /// retry.
    #[error("append-phase commit failed: {0}")]
    AppendFlush(BuildError),

    /// At least one buffered mutation failed to drive to
    /// `Complete`. WALs that landed durably before the failure
    /// are listed in `committed_wal_ids`; the recovery sweep on
    /// the next supertable open completes any operation whose
    /// WAL was written before the failure. The remaining
    /// buffered ops stay on the writer for retry.
    #[error("partial commit: {committed} of {total} mutations completed before {cause}")]
    PartialCommit {
        committed_wal_ids: Vec<WalId>,
        committed: usize,
        total: usize,
        cause: Box<MutationError>,
    },
}
