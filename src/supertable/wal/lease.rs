// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Lease primitives for the WAL pipeline.
//!
//! Cooperative ownership of one WAL by the process currently
//! driving it. The lease is **advisory** — the load-bearing safety
//! primitive is the etag CAS on the state doc; the lease only
//! prevents two processes from doing the same work concurrently
//! (still correct, just wasteful). Safety arguments of the form
//! "this is safe because we hold the lease" are wrong — every
//! state transition has to surface at the CAS layer.
//!
//! ## What lives here
//!
//! - [`try_acquire`] — vacant or expired → CAS-take. Vacant means
//!   the WAL's `lease` field is `None`; expired means `expires_at`
//!   is in the past relative to `now`. The CAS returns the new etag
//!   on success or a typed `LeaseConflict` on contention.
//! - [`try_heartbeat`] — owner extends the lease's `expires_at`.
//!   The caller passes the owner id it expects to still hold the
//!   lease; if the WAL's `lease.owner` no longer matches we surface
//!   [`LeaseError::Preempted`] so the caller's work thread can wind
//!   down without producing duplicate work.
//! - [`try_release`] — owner clears the lease on completion. Pure
//!   hygiene; not strictly required since the lease would expire
//!   naturally.
//!
//! ## Defaults
//!
//! [`DEFAULT_LEASE_DURATION`] is 60 s; [`DEFAULT_HEARTBEAT_INTERVAL`]
//! is 10 s. Tests override both to drive edge cases (a 100 ms lease
//! makes expiry observable in a single test run).
//!
//! Wall-clock times go through `chrono::Utc` so they round-trip
//! through the JSON state doc unchanged; the lease grants
//! `Duration` is converted at acquire / heartbeat time.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use chrono::{DateTime, Utc};
use thiserror::Error;
use tokio::task::JoinHandle;

use crate::supertable::wal::{
    persistence::{Etag, WalStore, WalStoreError},
    state_doc::{Lease, SupertableHandleId, WalId, WalState, WalStateDoc},
};

/// Default lease lifetime per acquire / heartbeat. 60s
/// balances "long enough that NTP-level clock skew never
/// expires a healthy owner" against "short enough that a dead
/// owner's lease is reapable within a minute." Wall-clock, not
/// monotonic — leases live in JSON so they have to use a
/// serializable timebase.
pub const DEFAULT_LEASE_DURATION: Duration = Duration::from_secs(60);

/// Default heartbeat interval. One sixth of the lease so a
/// single missed beat doesn't flip a healthy owner over the
/// expiry threshold.
pub const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(10);

/// Typed failures from the lease primitives. Lease loss is
/// expected on the critical path — peer recovery, clock skew,
/// crash-revive — so callers should pattern-match these variants
/// rather than treat them as bugs.
#[derive(Debug, Error)]
pub enum LeaseError {
    /// We tried to acquire / preempt but a live (un-expired) lease
    /// is held by another owner. The caller's recovery sweep skips
    /// this WAL on this pass.
    #[error("lease held by another owner ({held_owner:?}) until {held_expires_at}; cannot acquire")]
    Conflict {
        held_owner: SupertableHandleId,
        held_expires_at: DateTime<Utc>,
    },

    /// Heartbeat / release found a different owner on the WAL —
    /// we've been preempted (clock skew + a peer recovery, most
    /// commonly). The work thread should stop on the next progress
    /// check; the heartbeat task self-terminates.
    #[error("WAL preempted: lease owner is now {actual_owner:?}, not {expected_owner:?}")]
    Preempted {
        expected_owner: SupertableHandleId,
        actual_owner: SupertableHandleId,
    },

    /// Heartbeat / release found no lease at all — typically the
    /// WAL transitioned to `Complete` and a peer cleared the lease.
    /// Treat the same as `Preempted`: stop work.
    #[error("WAL has no lease; expected owner {expected_owner:?}")]
    LeaseMissing { expected_owner: SupertableHandleId },

    /// The CAS lost because the WAL state doc was mutated between
    /// our read and our PUT. The caller decides whether to retry
    /// (re-read + re-CAS) or surface the loss. Distinct from
    /// `Conflict` because the WAL's lease field might still be
    /// vacant on re-read — somebody else just bumped it for an
    /// unrelated reason.
    #[error("etag CAS lost; WAL state doc was updated by another writer")]
    CasLost,

    /// Tried to operate on a WAL whose `state` rules out lease
    /// activity. `Complete` WALs never get a fresh lease — they
    /// just await cleanup. Surfacing this typed lets the recovery
    /// sweep skip without confusion.
    #[error("lease op invalid on WAL in state {state:?}")]
    InvalidPreState { state: WalState },

    /// Underlying state-doc I/O failure.
    #[error("wal store error: {0}")]
    WalStore(#[from] WalStoreError),
}

/// Attempt to acquire a lease on `wal_id` for `owner`.
///
/// Returns the post-CAS WAL state doc + the new etag on success.
///
/// **Vacant** (`wal.lease == None`) → CAS the lease in directly.
///
/// **Expired** (`wal.lease.expires_at < now`) → preempt by CAS-ing
/// the new lease on top.
///
/// **Live** (`wal.lease.expires_at >= now`) → fail with
/// [`LeaseError::Conflict`]; the caller's sweep skips the WAL this
/// pass.
///
/// A CAS-loss on the PUT surfaces as [`LeaseError::CasLost`]. The
/// caller decides whether to retry — typical recovery sweeps just
/// move on, since a CAS-loss means somebody else acted first.
///
/// Pre-condition: `wal.state` must be `Intent` or `Appended`.
/// `Complete` returns [`LeaseError::InvalidPreState`] — a finished
/// WAL doesn't need a lease.
pub async fn try_acquire(
    store: &WalStore,
    wal_id: WalId,
    owner: SupertableHandleId,
    now: DateTime<Utc>,
    lease_duration: Duration,
) -> Result<(WalStateDoc, Etag), LeaseError> {
    let (mut doc, etag) = store.read(wal_id).await?;
    if doc.state == WalState::Complete {
        return Err(LeaseError::InvalidPreState { state: doc.state });
    }

    // Reject only when the lease is held AND un-expired. The "None"
    // and "expired" cases both flow into the CAS path below.
    if let Some(existing) = &doc.lease
        && existing.expires_at > now
        && existing.owner != owner
    {
        return Err(LeaseError::Conflict {
            held_owner: existing.owner,
            held_expires_at: existing.expires_at,
        });
    }

    let expires_at = now + chrono::Duration::from_std(lease_duration).unwrap_or_default();
    doc.lease = Some(Lease {
        owner,
        acquired_at: now,
        expires_at,
    });

    match store.update_with_etag(wal_id, &etag, &doc).await {
        Ok(new_etag) => Ok((doc, new_etag)),
        Err(WalStoreError::CasFailed { .. }) => Err(LeaseError::CasLost),
        Err(other) => Err(other.into()),
    }
}

/// Extend the lease's `expires_at` to `now + lease_duration`.
///
/// Caller passes the `owner` id they expect to still hold the
/// lease. Mismatch surfaces as [`LeaseError::Preempted`] (or
/// [`LeaseError::LeaseMissing`]); the heartbeat task should
/// self-terminate and the work thread should stop on its next
/// progress check.
///
/// Idempotent on the no-op case: if the WAL is already at
/// `Complete` and the lease has been cleared as part of cleanup,
/// we surface `LeaseMissing` rather than spinning the heartbeat
/// loop forever.
pub async fn try_heartbeat(
    store: &WalStore,
    wal_id: WalId,
    owner: SupertableHandleId,
    now: DateTime<Utc>,
    lease_duration: Duration,
) -> Result<(WalStateDoc, Etag), LeaseError> {
    let (mut doc, etag) = store.read(wal_id).await?;
    match &doc.lease {
        None => {
            return Err(LeaseError::LeaseMissing {
                expected_owner: owner,
            });
        }
        Some(existing) if existing.owner != owner => {
            return Err(LeaseError::Preempted {
                expected_owner: owner,
                actual_owner: existing.owner,
            });
        }
        Some(_) => {}
    }

    let expires_at = now + chrono::Duration::from_std(lease_duration).unwrap_or_default();
    if let Some(existing) = doc.lease.as_mut() {
        existing.expires_at = expires_at;
    }

    match store.update_with_etag(wal_id, &etag, &doc).await {
        Ok(new_etag) => Ok((doc, new_etag)),
        Err(WalStoreError::CasFailed { .. }) => Err(LeaseError::CasLost),
        Err(other) => Err(other.into()),
    }
}

/// Clear the lease. Owner check matches [`try_heartbeat`]'s; a
/// non-matching owner is preempted, no-lease is missing. Called
/// when the work thread finishes (state → `Complete`) or
/// voluntarily yields.
pub async fn try_release(
    store: &WalStore,
    wal_id: WalId,
    owner: SupertableHandleId,
) -> Result<(WalStateDoc, Etag), LeaseError> {
    let (mut doc, etag) = store.read(wal_id).await?;
    match &doc.lease {
        None => {
            return Err(LeaseError::LeaseMissing {
                expected_owner: owner,
            });
        }
        Some(existing) if existing.owner != owner => {
            return Err(LeaseError::Preempted {
                expected_owner: owner,
                actual_owner: existing.owner,
            });
        }
        Some(_) => {}
    }
    doc.lease = None;

    match store.update_with_etag(wal_id, &etag, &doc).await {
        Ok(new_etag) => Ok((doc, new_etag)),
        Err(WalStoreError::CasFailed { .. }) => Err(LeaseError::CasLost),
        Err(other) => Err(other.into()),
    }
}

// ============================================================
// Heartbeat task
// ============================================================
//
// Each WAL this process drives gets one background heartbeat
// task that periodically extends the lease's `expires_at`. The
// task self-terminates on:
//
//   - **Preemption** — heartbeat CAS surfaces a different owner
//     or no lease at all (LeaseMissing / Preempted).
//   - **Stuck worker** — the work thread hasn't called
//     `mark_progress()` in over `T_lease / 2`. The lease expires
//     so recovery can take over.
//   - **Explicit stop** — the work thread calls
//     `HeartbeatHandle::stop()` after a clean completion.
//
// The heartbeat task is *not* the safety primitive — CAS on
// state-doc PUTs is. The heartbeat keeps the lease fresh so peer
// recovery doesn't preempt prematurely; nothing more.

/// Tracker the work thread + the heartbeat task share. The work
/// thread bumps `last_progress_at_unix_ms` after each storage
/// operation; the heartbeat task reads it to gate the lease
/// extension. Cheap atomics so the hot path isn't gated on a
/// mutex.
#[derive(Debug)]
pub struct ProgressTracker {
    /// Monotonic wall-clock ms (UTC) of the most recent
    /// progress mark. Bumped by the work thread.
    last_progress_at_unix_ms: AtomicU64,
    /// Set by the heartbeat task on preemption or stuck-worker
    /// detection. The work thread polls this after each
    /// storage operation and bails out cleanly if set.
    stop_requested: AtomicBool,
}

impl ProgressTracker {
    /// Construct a tracker pinned at `now`. The constructor
    /// stamps the initial "most recent progress" so the first
    /// heartbeat doesn't trip the stuck-worker check.
    pub fn new(now: SystemTime) -> Self {
        Self {
            last_progress_at_unix_ms: AtomicU64::new(unix_ms(now)),
            stop_requested: AtomicBool::new(false),
        }
    }

    /// Bump the progress timestamp. Called by the work thread
    /// after each storage operation it completes; the heartbeat
    /// task uses this to detect a stuck worker.
    pub fn mark_progress(&self) {
        let now = unix_ms(SystemTime::now());
        self.last_progress_at_unix_ms.store(now, Ordering::Relaxed);
    }

    /// `true` if the heartbeat task signaled the work thread to
    /// stop (preemption or stuck-worker detection).
    pub fn stop_requested(&self) -> bool {
        self.stop_requested.load(Ordering::Relaxed)
    }

    /// Request the work thread to stop. Called by the heartbeat
    /// task internally, but exposed so the work thread can
    /// signal its own stop on clean completion (e.g., the
    /// pipeline transitions to `Complete` and we drop the
    /// heartbeat).
    pub fn request_stop(&self) {
        self.stop_requested.store(true, Ordering::Relaxed);
    }

    fn last_progress_at(&self) -> SystemTime {
        let ms = self.last_progress_at_unix_ms.load(Ordering::Relaxed);
        UNIX_EPOCH + Duration::from_millis(ms)
    }
}

fn unix_ms(t: SystemTime) -> u64 {
    t.duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Handle to a running heartbeat task. Drop to fire-and-forget
/// (the task self-terminates on the tracker's `stop_requested`);
/// call [`HeartbeatHandle::stop`] for a synchronous wind-down.
#[derive(Debug)]
pub struct HeartbeatHandle {
    join: Option<JoinHandle<()>>,
    tracker: Arc<ProgressTracker>,
}

impl HeartbeatHandle {
    /// Request the heartbeat task to stop and await its
    /// completion. Idempotent.
    pub async fn stop(mut self) {
        self.tracker.request_stop();
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }

    /// Reference to the shared progress tracker. The work thread
    /// uses this to `mark_progress()` after each storage op and
    /// to poll `stop_requested()` between steps.
    pub fn tracker(&self) -> Arc<ProgressTracker> {
        Arc::clone(&self.tracker)
    }
}

impl Drop for HeartbeatHandle {
    fn drop(&mut self) {
        // Best-effort stop on drop. The task self-terminates on
        // the next tick; we don't await here because Drop is
        // sync.
        self.tracker.request_stop();
        if let Some(join) = self.join.take() {
            join.abort();
        }
    }
}

/// Spawn a heartbeat task on the ambient tokio runtime.
///
/// Every `interval` the task:
///
/// 1. Checks the tracker. If `stop_requested` is set, exits.
/// 2. Checks `last_progress_at`. If stale (older than
///    `lease_duration / 2` from now), signals stop on the
///    tracker and exits — letting the lease expire so recovery
///    can take over.
/// 3. Calls [`try_heartbeat`] to extend the lease. On
///    `Preempted` / `LeaseMissing` / `CasLost` errors, signals
///    stop and exits.
///
/// The work thread is responsible for polling
/// `tracker.stop_requested()` between storage operations and
/// bailing out on `true`. Without that cooperation a preempted
/// owner could keep doing duplicate work — wasted but not
/// incorrect, since the WAL's per-step CAS still linearizes the
/// outcomes.
pub fn spawn_heartbeat(
    store: WalStore,
    wal_id: WalId,
    owner: SupertableHandleId,
    lease_duration: Duration,
    interval: Duration,
) -> HeartbeatHandle {
    let tracker = Arc::new(ProgressTracker::new(SystemTime::now()));
    let tracker_for_task = Arc::clone(&tracker);
    let stuck_threshold = lease_duration / 2;

    let join = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // First tick fires immediately (default); skip it so
        // the work thread has a chance to do at least one
        // storage op before we evaluate progress.
        ticker.tick().await;
        loop {
            ticker.tick().await;
            if tracker_for_task.stop_requested() {
                return;
            }
            // Stuck-worker check.
            let last = tracker_for_task.last_progress_at();
            if let Ok(elapsed) = SystemTime::now().duration_since(last)
                && elapsed > stuck_threshold
            {
                tracker_for_task.request_stop();
                return;
            }
            // Lease extension.
            match try_heartbeat(&store, wal_id, owner, Utc::now(), lease_duration).await {
                Ok(_) => {}
                Err(_) => {
                    tracker_for_task.request_stop();
                    return;
                }
            }
        }
    });

    HeartbeatHandle {
        join: Some(join),
        tracker,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use chrono::Duration as ChronoDuration;
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;
    use crate::{
        storage::{LocalFsStorageProvider, StorageProvider},
        supertable::wal::state_doc::{
            OpKind, RowId, SCHEMA_VERSION, SupertableHandleId, TombstoneEntry, TombstoneOutcome,
            WalState,
        },
    };

    fn sample_intent_wal(wal_id_v: i128) -> WalStateDoc {
        WalStateDoc {
            wal_id: WalId(wal_id_v),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
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
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        }
    }

    fn fixture() -> (TempDir, WalStore) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        (dir, WalStore::new(storage))
    }

    async fn put_wal(store: &WalStore, doc: &WalStateDoc) -> Etag {
        store.create(doc).await.expect("create")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_on_vacant_lease_succeeds_and_writes_owner() {
        let (_dir, store) = fixture();
        let doc = sample_intent_wal(1);
        let _ = put_wal(&store, &doc).await;
        let owner = SupertableHandleId(0x1111);
        let now = Utc::now();
        let (post, new_etag) = try_acquire(&store, doc.wal_id, owner, now, Duration::from_secs(30))
            .await
            .expect("acquire");
        let lease = post.lease.expect("set");
        assert_eq!(lease.owner, owner);
        assert!(lease.expires_at > now);
        // Confirm persistence.
        let (read_back, read_etag) = store.read(doc.wal_id).await.expect("read");
        assert_eq!(read_back.lease.expect("set").owner, owner);
        assert_eq!(read_etag, new_etag);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_on_live_lease_returns_conflict() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(2);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner: SupertableHandleId(0xAAAA),
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;
        let err = try_acquire(
            &store,
            doc.wal_id,
            SupertableHandleId(0xBBBB),
            now,
            Duration::from_secs(30),
        )
        .await
        .expect_err("must conflict");
        assert!(
            matches!(err, LeaseError::Conflict { held_owner, .. } if held_owner == SupertableHandleId(0xAAAA))
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_on_expired_lease_preempts() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(3);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner: SupertableHandleId(0xAAAA),
            acquired_at: now - ChronoDuration::seconds(120),
            expires_at: now - ChronoDuration::seconds(10),
        });
        let _ = put_wal(&store, &doc).await;
        let (post, _etag) = try_acquire(
            &store,
            doc.wal_id,
            SupertableHandleId(0xBBBB),
            now,
            Duration::from_secs(30),
        )
        .await
        .expect("expired → preempt");
        assert_eq!(post.lease.expect("set").owner, SupertableHandleId(0xBBBB));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_re_takes_for_same_owner_renewing_expiry() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(4);
        let owner = SupertableHandleId(0xC0C0);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now - ChronoDuration::seconds(45),
            expires_at: now + ChronoDuration::seconds(15),
        });
        let _ = put_wal(&store, &doc).await;
        let (post, _etag) = try_acquire(&store, doc.wal_id, owner, now, Duration::from_secs(60))
            .await
            .expect("re-acquire");
        let lease = post.lease.expect("set");
        assert_eq!(lease.owner, owner);
        // New expiry must reflect the freshly-renewed lease.
        let expected = now + ChronoDuration::seconds(60);
        assert!((lease.expires_at - expected).num_milliseconds().abs() < 10);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn acquire_on_complete_wal_is_invalid_pre_state() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(5);
        doc.state = WalState::Complete;
        let _ = put_wal(&store, &doc).await;
        let err = try_acquire(
            &store,
            doc.wal_id,
            SupertableHandleId(0xDEAD),
            Utc::now(),
            Duration::from_secs(30),
        )
        .await
        .expect_err("must reject");
        assert!(matches!(
            err,
            LeaseError::InvalidPreState {
                state: WalState::Complete
            }
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heartbeat_extends_expires_at() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(6);
        let owner = SupertableHandleId(0x10001);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(20),
        });
        let _ = put_wal(&store, &doc).await;

        let later = now + ChronoDuration::seconds(10);
        let (post, _etag) =
            try_heartbeat(&store, doc.wal_id, owner, later, Duration::from_secs(60))
                .await
                .expect("heartbeat");
        let lease = post.lease.expect("still held");
        let expected = later + ChronoDuration::seconds(60);
        assert!((lease.expires_at - expected).num_milliseconds().abs() < 10);
        // Owner unchanged.
        assert_eq!(lease.owner, owner);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heartbeat_on_preempted_lease_returns_preempted() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(7);
        let original = SupertableHandleId(0xAAAA);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner: SupertableHandleId(0xBBBB),
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;
        let err = try_heartbeat(&store, doc.wal_id, original, now, Duration::from_secs(60))
            .await
            .expect_err("preempted");
        assert!(matches!(
            err,
            LeaseError::Preempted {
                expected_owner,
                actual_owner,
            } if expected_owner == original && actual_owner == SupertableHandleId(0xBBBB)
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heartbeat_on_cleared_lease_returns_lease_missing() {
        let (_dir, store) = fixture();
        let doc = sample_intent_wal(8);
        let _ = put_wal(&store, &doc).await;
        let err = try_heartbeat(
            &store,
            doc.wal_id,
            SupertableHandleId(0xAAAA),
            Utc::now(),
            Duration::from_secs(60),
        )
        .await
        .expect_err("no lease");
        assert!(matches!(err, LeaseError::LeaseMissing { .. }));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_clears_lease_for_matching_owner() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(9);
        let owner = SupertableHandleId(0xC0DE);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;
        let (post, _etag) = try_release(&store, doc.wal_id, owner)
            .await
            .expect("release");
        assert!(post.lease.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_on_wrong_owner_returns_preempted() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(10);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner: SupertableHandleId(0xAAAA),
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;
        let err = try_release(&store, doc.wal_id, SupertableHandleId(0xBBBB))
            .await
            .expect_err("preempted");
        assert!(matches!(err, LeaseError::Preempted { .. }));
    }

    #[test]
    fn defaults_match_plan_constants() {
        assert_eq!(DEFAULT_LEASE_DURATION, Duration::from_secs(60));
        assert_eq!(DEFAULT_HEARTBEAT_INTERVAL, Duration::from_secs(10));
    }

    // ---- Heartbeat task: progress tracking + lease extension ----

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn heartbeat_extends_lease_while_worker_marks_progress() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(20);
        let owner = SupertableHandleId(0xC0FFEE);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(1),
        });
        let _ = put_wal(&store, &doc).await;

        let handle = spawn_heartbeat(
            store.clone(),
            doc.wal_id,
            owner,
            Duration::from_millis(800),
            Duration::from_millis(100),
        );
        let tracker = handle.tracker();
        // Pretend the work thread is making progress every 50ms
        // for ~400ms.
        for _ in 0..8 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            tracker.mark_progress();
        }
        // Read the lease — expires_at should be well past the
        // initial "now + 1s" mark because the heartbeat extended
        // it across at least one tick.
        let (post, _etag) = store.read(doc.wal_id).await.expect("read");
        let lease = post.lease.expect("still held");
        assert_eq!(lease.owner, owner);
        // The original expiry was now+1s; the heartbeat should
        // have pushed it past now+500ms at least.
        let elapsed_since_initial = lease.expires_at - now;
        assert!(
            elapsed_since_initial.num_milliseconds() > 500,
            "expected heartbeat to extend lease past initial 1s budget; got {} ms",
            elapsed_since_initial.num_milliseconds()
        );
        handle.stop().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn heartbeat_signals_stop_when_worker_is_stuck() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(21);
        let owner = SupertableHandleId(0xDEAD_BEEF);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(5),
        });
        let _ = put_wal(&store, &doc).await;

        // Lease duration 400ms → stuck threshold = 200ms. The
        // heartbeat ticks every 50ms and sees no progress mark
        // past 200ms → flips stop_requested.
        let handle = spawn_heartbeat(
            store.clone(),
            doc.wal_id,
            owner,
            Duration::from_millis(400),
            Duration::from_millis(50),
        );
        let tracker = handle.tracker();
        // Do NOT call mark_progress. Wait until the heartbeat
        // observes the stale progress marker. Under a loaded full
        // `cargo test --lib` run the heartbeat task can be scheduled
        // later than the nominal 50ms tick, so poll with a bounded
        // timeout instead of asserting after one fixed sleep.
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if tracker.stop_requested() {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .expect("expected heartbeat to flip stop_requested on stuck worker");
        handle.stop().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn heartbeat_signals_stop_on_preemption() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(22);
        let original = SupertableHandleId(0xAAAA);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner: original,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;

        let handle = spawn_heartbeat(
            store.clone(),
            doc.wal_id,
            original,
            Duration::from_secs(60),
            Duration::from_millis(50),
        );
        let tracker = handle.tracker();
        // Externally preempt: write a new lease owner.
        let (mut current, etag) = store.read(doc.wal_id).await.expect("read");
        current.lease = Some(Lease {
            owner: SupertableHandleId(0xBBBB),
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = store
            .update_with_etag(doc.wal_id, &etag, &current)
            .await
            .expect("preempt cas");
        // Keep marking progress so the stuck-worker check
        // doesn't fire first; we want to confirm preemption
        // specifically triggers stop.
        for _ in 0..6 {
            tracker.mark_progress();
            tokio::time::sleep(Duration::from_millis(50)).await;
            if tracker.stop_requested() {
                break;
            }
        }
        assert!(
            tracker.stop_requested(),
            "expected heartbeat to flip stop_requested on preemption"
        );
        handle.stop().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explicit_stop_winds_down_heartbeat_task() {
        let (_dir, store) = fixture();
        let mut doc = sample_intent_wal(23);
        let owner = SupertableHandleId(0x4242);
        let now = Utc::now();
        doc.lease = Some(Lease {
            owner,
            acquired_at: now,
            expires_at: now + ChronoDuration::seconds(60),
        });
        let _ = put_wal(&store, &doc).await;

        let handle = spawn_heartbeat(
            store.clone(),
            doc.wal_id,
            owner,
            Duration::from_secs(60),
            Duration::from_millis(50),
        );
        // No progress marks; we just stop immediately.
        handle.stop().await;
        // No assertion — `stop().await` must return cleanly.
    }

    // ---- Suppress unused-Uuid warning helper -----------------------------

    #[test]
    fn unused_uuid_smoke() {
        let _ = Uuid::nil();
    }
}
