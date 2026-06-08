// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! On-demand WAL recovery sweep.
//!
//! There is **no background scan**. Recovery runs only when a
//! caller invokes it: at `Supertable::open(...)` time, at
//! `Supertable::create(...).expect("create")` time when storage is attached, or
//! via the explicit operator hatch.
//!
//! ## What a sweep does
//!
//! 1. List every WAL state-doc at `wal/mutations/*.json`.
//! 2. For each WAL, sorted oldest-first by `wal_id`:
//!     - Read its state doc.
//!     - `Complete` → skip (the GC sweep cleans these up).
//!     - `Intent` or `Appended` → try to acquire (or preempt
//!       expired) the lease. On acquisition, drive the WAL
//!       through the remainder of its pipeline:
//!         - `Intent` UPDATE → run append phase, then tombstone
//!           phase.
//!         - `Appended` UPDATE → tombstone phase only.
//!         - `Intent` DELETE → tombstone phase only.
//! 3. Return a [`RecoveryReport`] tallying per-outcome counts so
//!    callers (tests, operators) can verify the sweep made the
//!    expected progress.
//!
//! ## Safety notes
//!
//! Every step is gated by the etag-CAS layer: a peer process
//! that takes the lease between our list and our acquire makes
//! our acquire fail with `Conflict` or `CasLost`; we skip and
//! move on. The cooperative-lease pattern means at most one
//! process drives a given WAL at a time, but if two manage to
//! squeeze through simultaneously the CAS-PUTs on the state doc
//! and the manifest pointer linearize the work — duplicate work
//! is wasted, never corrupting.
//!
//! ## Time budget
//!
//! The sweep doesn't enforce its own deadline yet; callers that
//! need bounded latency wrap with `tokio::time::timeout`. The
//! sweep is itself a sequence of independent per-WAL state
//! machines so partial progress is always safe.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use thiserror::Error;

use crate::supertable::handle::Supertable;
use crate::supertable::wal::lease::{self, LeaseError};
use crate::supertable::wal::persistence::{WalStore, WalStoreError};
use crate::supertable::wal::pipeline::{
    self, AppendPhaseError, AppendPhaseOutcome, TombstonePhaseError, TombstonePhaseOutcome,
};
use crate::supertable::wal::state_doc::{OpKind, SupertableHandleId, WalId, WalState, WalStateDoc};

/// Aggregate counts from one recovery sweep. Stable shape so
/// integration tests + operator scripts can pin assertions
/// against it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Total WALs the sweep walked (including the ones it
    /// skipped due to lease conflicts or `Complete` state).
    pub n_scanned: usize,
    /// `Complete` WALs the sweep walked past without touching.
    pub n_already_complete: usize,
    /// WALs whose lease was held by a live peer at scan time.
    /// The peer will drive these to completion; this process
    /// moves on.
    pub n_held_by_peer: usize,
    /// WALs this process drove from `Intent → Appended` then on
    /// to `Complete`.
    pub n_full_pipeline_completed: usize,
    /// WALs this process drove from `Appended → Complete` (the
    /// append phase had already landed before this sweep).
    pub n_tombstone_only_completed: usize,
    /// WALs we skipped because their state-doc read failed (most
    /// likely a peer deleted the WAL between our list + our read,
    /// e.g. as part of post-`Complete` cleanup). Logged for
    /// observability.
    pub n_vanished_during_scan: usize,
    /// CAS losses during lease acquire / preempt. Recoverable;
    /// the next sweep tries again.
    pub n_cas_lost: usize,
    /// Total docs tombstoned across all WALs the sweep drove.
    pub total_targets_tombstoned: usize,
    /// Total `NotFound` outcomes across all WALs the sweep drove.
    pub total_targets_not_found: usize,
}

/// Typed sweep failures. Per-WAL errors aren't terminal — the
/// sweep logs them in the report and continues to the next WAL.
/// Errors surface here only when the sweep itself can't proceed
/// (LIST failure or supertable misconfiguration).
#[derive(Debug, Error)]
pub enum RecoveryError {
    /// The supertable has no storage attached. Recovery has
    /// nowhere to look for WALs; the caller's setup is wrong.
    #[error("recovery sweep requires storage; supertable has none attached")]
    NoStorageAttached,

    /// The LIST against `wal/mutations/*` failed. Without the
    /// listing the sweep can't know which WALs exist; surface
    /// the underlying storage error.
    #[error("failed to list WAL state docs: {0}")]
    ListFailed(#[from] WalStoreError),
}

/// Drive one recovery sweep against `supertable`. Sync-bridges
/// nothing — callers that want a sync entry point go through
/// `Supertable::run_recovery_sweep_once`.
///
/// `owner` is the lease-owner id this sweep stamps on every WAL
/// it acquires. `Supertable` mints one at handle construction;
/// passing it through here keeps the recovery module free of
/// id-allocation concerns.
pub async fn scan_and_recover(
    supertable: &Supertable,
    owner: SupertableHandleId,
    lease_duration: Duration,
) -> Result<RecoveryReport, RecoveryError> {
    let inner = supertable.inner();
    if inner.options.storage.is_none() {
        return Err(RecoveryError::NoStorageAttached);
    }
    let storage = inner
        .options
        .storage
        .as_ref()
        .expect("checked above")
        .clone();
    let wal_store = WalStore::new(Arc::clone(&storage));

    let mut report = RecoveryReport::default();
    let wal_ids = wal_store.list_wal_ids().await?;
    for wal_id in wal_ids {
        report.n_scanned += 1;
        match recover_one(supertable, &wal_store, wal_id, owner, lease_duration).await {
            Ok(outcome) => {
                outcome.fold_into(&mut report);
            }
            Err(SweepStep::Vanished) => {
                report.n_vanished_during_scan += 1;
            }
            Err(SweepStep::HeldByPeer) => {
                report.n_held_by_peer += 1;
            }
            Err(SweepStep::CasLost) => {
                report.n_cas_lost += 1;
            }
        }
    }
    Ok(report)
}

/// Outcome of driving one WAL through the recovery state machine.
enum OneWalOutcome {
    AlreadyComplete,
    FullPipeline {
        n_tombstoned: usize,
        n_not_found: usize,
    },
    TombstoneOnly {
        n_tombstoned: usize,
        n_not_found: usize,
    },
}

impl OneWalOutcome {
    fn fold_into(self, report: &mut RecoveryReport) {
        match self {
            OneWalOutcome::AlreadyComplete => {
                report.n_already_complete += 1;
            }
            OneWalOutcome::FullPipeline {
                n_tombstoned,
                n_not_found,
            } => {
                report.n_full_pipeline_completed += 1;
                report.total_targets_tombstoned += n_tombstoned;
                report.total_targets_not_found += n_not_found;
            }
            OneWalOutcome::TombstoneOnly {
                n_tombstoned,
                n_not_found,
            } => {
                report.n_tombstone_only_completed += 1;
                report.total_targets_tombstoned += n_tombstoned;
                report.total_targets_not_found += n_not_found;
            }
        }
    }
}

/// Internal flow control: per-WAL "this sweep skips it" signals.
/// These aren't typed errors that propagate to the caller — they're
/// counted in the report and the sweep moves on.
enum SweepStep {
    /// The state-doc read failed with NotFound. A peer most
    /// likely deleted it as part of post-`Complete` cleanup
    /// (the `Complete` race in the WAL pipeline). Move on.
    Vanished,
    /// Acquire returned `Conflict` — a live lease is held by a
    /// peer; we move on.
    HeldByPeer,
    /// CAS-loss anywhere — somebody else acted first. Move on.
    CasLost,
}

async fn recover_one(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_id: WalId,
    owner: SupertableHandleId,
    lease_duration: Duration,
) -> Result<OneWalOutcome, SweepStep> {
    // Read the state doc up front. If it vanished between our
    // LIST and this read, classify as `Vanished`.
    let doc = match wal_store.read(wal_id).await {
        Ok((d, _etag)) => d,
        Err(WalStoreError::Storage {
            source: crate::storage::StorageError::NotFound { .. },
            ..
        }) => return Err(SweepStep::Vanished),
        Err(_) => return Err(SweepStep::Vanished),
    };

    // Complete WALs need no work.
    if doc.state == WalState::Complete {
        return Ok(OneWalOutcome::AlreadyComplete);
    }

    // Acquire (or preempt-expired) the lease. The acquired etag
    // is the one the pipeline drivers will CAS against on their
    // first state advance; from here on contention surfaces as
    // CAS-loss in the pipeline (not here).
    let now = Utc::now();
    let (doc, etag) = match lease::try_acquire(wal_store, wal_id, owner, now, lease_duration).await
    {
        Ok((d, e)) => (d, e),
        Err(LeaseError::Conflict { .. }) => return Err(SweepStep::HeldByPeer),
        Err(LeaseError::CasLost) => return Err(SweepStep::CasLost),
        Err(LeaseError::InvalidPreState { .. }) => {
            // Raced with a Complete transition between our
            // first read and the acquire. Treat as already-
            // complete; the GC sweep will reap the state doc
            // later.
            return Ok(OneWalOutcome::AlreadyComplete);
        }
        Err(_) => return Err(SweepStep::CasLost),
    };

    drive_to_complete(supertable, wal_store, doc, etag).await
}

/// Run the remaining pipeline steps based on the WAL's current
/// state + op-kind. Returns the outcome of the work; per-step
/// errors propagate as `SweepStep::CasLost` so the sweep moves
/// on without aborting.
async fn drive_to_complete(
    supertable: &Supertable,
    wal_store: &WalStore,
    doc: WalStateDoc,
    etag: crate::supertable::wal::persistence::Etag,
) -> Result<OneWalOutcome, SweepStep> {
    let (post_doc, post_etag, append_ran) = match (doc.op_kind, doc.state) {
        (OpKind::Update, WalState::Intent) => {
            match pipeline::run_append_phase(supertable, wal_store, &doc, &etag).await {
                Ok((_, d, e)) => (d, e, true),
                Err(AppendPhaseError::WalStore(WalStoreError::CasFailed { .. })) => {
                    return Err(SweepStep::CasLost);
                }
                Err(_) => return Err(SweepStep::CasLost),
            }
        }
        (OpKind::Update, WalState::Appended) | (OpKind::Delete, WalState::Intent) => {
            (doc, etag, false)
        }
        // `Complete` was filtered out above; any other state-doc
        // shape is a builder bug that surfaced after lease
        // acquisition. Treat as skip; future sweeps can pick it
        // up if the underlying inconsistency is repaired.
        _ => return Err(SweepStep::CasLost),
    };

    let outcome =
        match pipeline::run_tombstone_phase(supertable, wal_store, &post_doc, &post_etag).await {
            Ok((
                TombstonePhaseOutcome::Applied {
                    n_tombstoned,
                    n_not_found,
                },
                _,
                _,
            ))
            | Ok((
                TombstonePhaseOutcome::AlreadyComplete {
                    n_tombstoned,
                    n_not_found,
                },
                _,
                _,
            )) => (n_tombstoned, n_not_found),
            Err(TombstonePhaseError::WalStore(WalStoreError::CasFailed { .. })) => {
                return Err(SweepStep::CasLost);
            }
            Err(_) => return Err(SweepStep::CasLost),
        };

    // Unused but pinned; clippy won't otherwise see that
    // append_ran is consumed below.
    let _ = AppendPhaseOutcome::Applied;
    if append_ran {
        Ok(OneWalOutcome::FullPipeline {
            n_tombstoned: outcome.0,
            n_not_found: outcome.1,
        })
    } else {
        Ok(OneWalOutcome::TombstoneOnly {
            n_tombstoned: outcome.0,
            n_not_found: outcome.1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use crate::supertable::Supertable;
    use crate::supertable::wal::state_doc::{
        OpKind, RowId, SCHEMA_VERSION, SupertableHandleId, TombstoneEntry, TombstoneOutcome,
    };
    use crate::test_helpers::default_supertable_options;
    use chrono::Utc;
    use tempfile::TempDir;
    use uuid::Uuid;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scan_empty_supertable_reports_zero_work() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let report = scan_and_recover(&st, SupertableHandleId(0xCAFE), Duration::from_secs(30))
            .await
            .expect("sweep");
        assert_eq!(report, RecoveryReport::default());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_drives_intent_delete_wal_to_complete() {
        // Pre-seed an Intent DELETE WAL whose target doesn't exist
        // anywhere in the (empty) supertable. Recovery's tombstone
        // pass resolves to NotFound and advances to Complete.
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));
        let wal_doc = WalStateDoc {
            wal_id: WalId(1234),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "test".into(),
            target_ids: vec![RowId(99)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(99),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        ws.create(&wal_doc).await.expect("seed");
        let owner = SupertableHandleId(0xC0DE);

        let report = scan_and_recover(&st, owner, Duration::from_secs(30))
            .await
            .expect("sweep");
        assert_eq!(report.n_scanned, 1);
        assert_eq!(report.n_tombstone_only_completed, 1);
        assert_eq!(report.total_targets_not_found, 1);
        assert_eq!(report.total_targets_tombstoned, 0);

        // The WAL is now Complete on disk.
        let (doc_after, _etag) = ws.read(WalId(1234)).await.expect("read");
        assert_eq!(doc_after.state, WalState::Complete);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_skips_complete_walls_without_touching_state() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));
        let mut wal_doc = WalStateDoc {
            wal_id: WalId(5678),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Complete,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "test".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::Tombstoned,
                tombstoned_in_superfile: Some(Uuid::from_u128(0xCAFE)),
            }],
        };
        // ensure doc_id 1 doesn't collide if we re-run
        wal_doc.predicate_repr = "noop".into();
        let etag_before = ws.create(&wal_doc).await.expect("seed");

        let report = scan_and_recover(&st, SupertableHandleId(0xABCD), Duration::from_secs(30))
            .await
            .expect("sweep");
        assert_eq!(report.n_scanned, 1);
        assert_eq!(report.n_already_complete, 1);
        assert_eq!(report.n_tombstone_only_completed, 0);

        // Etag unchanged → sweep didn't touch the state doc.
        let (_, etag_after) = ws.read(WalId(5678)).await.expect("read");
        assert_eq!(etag_after, etag_before);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_skips_wal_held_by_live_peer() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));
        let now = Utc::now();
        let wal_doc = WalStateDoc {
            wal_id: WalId(9999),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: now,
            lease: Some(crate::supertable::wal::state_doc::Lease {
                owner: SupertableHandleId(0xDEAD_DEAD),
                acquired_at: now,
                expires_at: now + chrono::Duration::seconds(120),
            }),
            predicate_repr: "held".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        ws.create(&wal_doc).await.expect("seed");
        let report = scan_and_recover(&st, SupertableHandleId(0xBEEF), Duration::from_secs(30))
            .await
            .expect("sweep");
        assert_eq!(report.n_scanned, 1);
        assert_eq!(report.n_held_by_peer, 1);
        assert_eq!(report.n_tombstone_only_completed, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_against_in_memory_supertable_errors_cleanly() {
        let st = Supertable::create(default_supertable_options()).expect("create");
        let err = scan_and_recover(&st, SupertableHandleId(0xC0DE), Duration::from_secs(30))
            .await
            .expect_err("must error");
        assert!(matches!(err, RecoveryError::NoStorageAttached));
    }

    // Operator-hatch wrapper coverage. `Supertable::run_recovery_sweep_once`
    // is a crate internal (`pub(crate)`), not public API, so these live
    // in-crate rather than in the integration `tests/` crate.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_recovery_sweep_once_hatch_reports_zero_work_on_empty_storage() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let report = st.run_recovery_sweep_once().await.expect("sweep");
        assert_eq!(report.n_scanned, 0);
        assert_eq!(report.n_already_complete, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn run_recovery_sweep_once_hatch_errors_on_in_memory_supertable() {
        let st = Supertable::create(default_supertable_options()).expect("create");
        let err = st.run_recovery_sweep_once().await.expect_err("must error");
        assert!(matches!(err, RecoveryError::NoStorageAttached));
    }
}
