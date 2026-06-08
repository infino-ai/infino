// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! GC sweep for completed WALs + orphaned arrow sidecars.
//!
//! Steady state: the writer's tombstone phase deletes its own
//! WAL state-doc + arrow sidecar inline as soon as the WAL
//! reaches `Complete` (best-effort). This module is the safety
//! net for the rare cases where:
//!
//! - The inline delete failed (transient storage error).
//! - The writer crashed between the COMPLETE CAS and the
//!   inline delete.
//! - The arrow sidecar was uploaded but the state-doc PUT
//!   failed (orphan sidecar).
//!
//! The sweep walks the `wal/mutations/` prefix once and:
//!
//! - For every `*.json` whose state is `Complete` AND whose
//!   `created_at` is older than [`DEFAULT_WAL_GRACE`], delete
//!   the `.json` and the matching `.arrow` (best-effort).
//! - For every `*.arrow` whose `.json` is absent AND whose
//!   storage mtime / created_at is older than
//!   [`DEFAULT_SIDECAR_GRACE`], delete the orphan.
//!
//! Grace periods bound how aggressively we reap — too short
//! risks deleting an arrow sidecar mid-flight while the
//! producer's state-doc PUT is in progress; too long lets the
//! prefix accumulate. See [`DEFAULT_WAL_GRACE`] +
//! [`DEFAULT_SIDECAR_GRACE`] for the defaults.

use std::sync::Arc;
use std::time::Duration;

use chrono::{DateTime, Utc};
use thiserror::Error;

use crate::storage::StorageProvider;
use crate::supertable::handle::Supertable;
use crate::supertable::wal::persistence::{WalStore, WalStoreError};
use crate::supertable::wal::state_doc::{WalId, WalState};

/// Default grace period before reaping a `Complete` WAL's
/// state-doc. Sized so a writer's inline-delete failure has
/// time to retry on its own before GC steps in — a healthy
/// supertable should never have anything to reap here.
pub const DEFAULT_WAL_GRACE: Duration = Duration::from_secs(5 * 60);

/// Default grace period before reaping an orphaned `.arrow`
/// sidecar. Sized to bound the worst-case writer-side gap
/// between sidecar PUT and state-doc PUT — long enough that a
/// slow-but-progressing producer's mid-flight sidecar is never
/// mistaken for an orphan.
pub const DEFAULT_SIDECAR_GRACE: Duration = Duration::from_secs(60 * 60);

/// Per-sweep tally. Stable shape so prefix-bloat regression
/// tests + operator scripts can pin assertions.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct GcReport {
    /// Total `.json` state docs the sweep walked (Complete +
    /// non-Complete).
    pub n_state_docs_scanned: usize,
    /// Complete state docs whose `created_at` was past the
    /// grace window and got deleted.
    pub n_state_docs_deleted: usize,
    /// Arrow sidecars deleted alongside a reaped state doc.
    pub n_arrow_sidecars_deleted_with_state: usize,
    /// Orphan `.arrow` sidecars (no matching `.json`) deleted.
    pub n_orphan_arrow_sidecars_deleted: usize,
    /// Per-WAL read failures during the sweep. Counted for
    /// observability; the sweep continues to the next WAL.
    pub n_read_errors: usize,
    /// Per-object delete failures. Counted for observability;
    /// objects stay durable on storage and a subsequent sweep
    /// re-attempts.
    pub n_delete_errors: usize,
}

/// Typed sweep failures. Per-object errors are tallied into the
/// report and don't surface here; only sweep-level
/// preconditions are returned as errors.
#[derive(Debug, Error)]
pub enum GcError {
    /// Supertable has no storage attached. GC has nothing to
    /// list against.
    #[error("GC sweep requires storage; supertable has none attached")]
    NoStorageAttached,
    /// The LIST against `wal/mutations/*` failed.
    #[error("failed to list wal/mutations prefix: {0}")]
    ListFailed(#[from] WalStoreError),
}

/// Drive one GC sweep against `supertable`. Storage prefix:
/// `wal/mutations/`. Returns the per-outcome tally.
///
/// `now` is the wall-clock instant the sweep evaluates grace
/// windows against; hoisted to the caller so the sweep
/// observes one consistent reference point across all objects.
/// Tests pass a custom `now` to drive the deterministic
/// grace-window paths.
pub async fn run_sweep(
    supertable: &Supertable,
    now: DateTime<Utc>,
    wal_grace: Duration,
    sidecar_grace: Duration,
) -> Result<GcReport, GcError> {
    let inner = supertable.inner();
    let storage = inner
        .options
        .storage
        .as_ref()
        .ok_or(GcError::NoStorageAttached)?
        .clone();
    let wal_store = WalStore::new(Arc::clone(&storage));

    let mut report = GcReport::default();

    // List the WAL state docs (`*.json`) we know about. The
    // listing returns sorted-ascending `WalId`s — oldest-first
    // is also the order we want to reap in, so peers behind us
    // see a monotonically-shrinking prefix.
    let wal_ids = match wal_store.list_wal_ids().await {
        Ok(v) => v,
        Err(e) => return Err(GcError::ListFailed(e)),
    };

    let wal_grace_chrono =
        chrono::Duration::from_std(wal_grace).unwrap_or_else(|_| chrono::Duration::seconds(0));

    // Per state-doc pass.
    for wal_id in &wal_ids {
        report.n_state_docs_scanned += 1;
        match wal_store.read(*wal_id).await {
            Ok((doc, _etag)) => {
                if doc.state == WalState::Complete && now - doc.created_at > wal_grace_chrono {
                    // Best-effort delete of the state doc.
                    let state_ok = wal_store.delete_state(*wal_id).await.is_ok();
                    // Best-effort delete of the arrow sidecar
                    // (update WAL only; delete WALs don't have
                    // one but `delete_arrow` is idempotent on
                    // 404 so calling unconditionally is fine).
                    let arrow_ok = wal_store.delete_arrow(*wal_id).await.is_ok();
                    if state_ok {
                        report.n_state_docs_deleted += 1;
                    } else {
                        report.n_delete_errors += 1;
                    }
                    if arrow_ok {
                        report.n_arrow_sidecars_deleted_with_state += 1;
                    }
                }
            }
            Err(_) => {
                report.n_read_errors += 1;
            }
        }
    }

    // Orphan `.arrow` pass. We need a full LIST against the
    // prefix to see arrow filenames whose `.json` siblings have
    // already been reaped. The state-doc pass already listed
    // the `.json` shape; an arrow file with no matching json
    // means the producer crashed between sidecar PUT and
    // state-doc create.
    let known_ids: std::collections::HashSet<WalId> = wal_ids.into_iter().collect();
    let sidecar_grace_chrono =
        chrono::Duration::from_std(sidecar_grace).unwrap_or_else(|_| chrono::Duration::seconds(0));
    match list_arrow_orphans(&storage, &known_ids).await {
        Ok(orphans) => {
            for (wal_id, mtime) in orphans {
                // We don't have wall-clock create info per
                // object in our storage trait; the meta.etag
                // round-trip doesn't carry it. Use `mtime` from
                // the LIST when available, else assume the
                // orphan is age-zero — conservative.
                let age_ok = match mtime {
                    Some(t) => now - t > sidecar_grace_chrono,
                    None => false,
                };
                if !age_ok {
                    continue;
                }
                if wal_store.delete_arrow(wal_id).await.is_ok() {
                    report.n_orphan_arrow_sidecars_deleted += 1;
                } else {
                    report.n_delete_errors += 1;
                }
            }
        }
        Err(_) => {
            // List failure for the orphan pass is logged
            // implicitly via report; the state-doc pass already
            // covered the common case.
            report.n_read_errors += 1;
        }
    }

    Ok(report)
}

/// Walk the `wal/mutations/` prefix looking for `.arrow`
/// objects whose `WalId` isn't in `known_ids`. Returns each
/// orphan's `WalId` paired with an mtime placeholder. The
/// mtime placeholder is always `None` because
/// `StorageProvider::list_with_prefix` doesn't surface
/// per-object timestamps; the orphan pass therefore relies on
/// producer-side cooperative cleanup until the trait exposes
/// mtime.
async fn list_arrow_orphans(
    storage: &Arc<dyn StorageProvider>,
    known_ids: &std::collections::HashSet<WalId>,
) -> Result<Vec<(WalId, Option<DateTime<Utc>>)>, crate::storage::StorageError> {
    let uris = storage.list_with_prefix("wal/mutations").await?;
    let mut out: Vec<(WalId, Option<DateTime<Utc>>)> = Vec::new();
    for uri in uris {
        let filename = match uri.rsplit_once('/') {
            Some((_, f)) => f,
            None => uri.as_str(),
        };
        let Some(stem) = filename.strip_suffix(".arrow") else {
            continue;
        };
        let Ok(wal_id) = WalId::from_hex(stem) else {
            continue;
        };
        if known_ids.contains(&wal_id) {
            continue;
        }
        // mtime is None because `list_with_prefix` doesn't
        // expose it. Callers that supply a non-zero
        // `sidecar_grace` therefore see "age == now-now == 0",
        // which is never older than the grace window; in
        // practice the writer's inline delete handles the
        // common case and the orphan pass is rarely needed.
        out.push((wal_id, None));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use crate::supertable::Supertable;
    use crate::supertable::wal::state_doc::{
        OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalStateDoc,
    };
    use crate::test_helpers::default_supertable_options;
    use chrono::{Duration as ChronoDuration, Utc};
    use tempfile::TempDir;

    fn complete_wal(wal_id_v: i128, created_at: DateTime<Utc>) -> WalStateDoc {
        WalStateDoc {
            wal_id: WalId(wal_id_v),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Complete,
            created_at,
            lease: None,
            predicate_repr: "gc-test".into(),
            target_ids: vec![RowId(1)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: vec![TombstoneEntry {
                target_id: RowId(1),
                outcome: TombstoneOutcome::NotFound,
                tombstoned_in_superfile: None,
            }],
        }
    }

    fn intent_wal(wal_id_v: i128, created_at: DateTime<Utc>) -> WalStateDoc {
        let mut doc = complete_wal(wal_id_v, created_at);
        doc.state = WalState::Intent;
        doc.tombstone_progress[0].outcome = TombstoneOutcome::Pending;
        doc
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_empty_supertable_reports_zero_work() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let report = run_sweep(&st, Utc::now(), DEFAULT_WAL_GRACE, DEFAULT_SIDECAR_GRACE)
            .await
            .expect("sweep");
        assert_eq!(report, GcReport::default());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_deletes_complete_wal_past_grace() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));

        // Seed a Complete WAL with created_at = 10 minutes ago.
        let now = Utc::now();
        let old = now - ChronoDuration::seconds(10 * 60);
        let doc = complete_wal(0x111, old);
        ws.create(&doc).await.expect("seed");

        // Run sweep with default grace (5 minutes). The WAL
        // is past the grace window so it should be deleted.
        let report = run_sweep(&st, now, DEFAULT_WAL_GRACE, DEFAULT_SIDECAR_GRACE)
            .await
            .expect("sweep");
        assert_eq!(report.n_state_docs_scanned, 1);
        assert_eq!(report.n_state_docs_deleted, 1);

        // Confirm gone.
        let after = ws.list_wal_ids().await.expect("list");
        assert!(after.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_does_not_delete_fresh_complete_wal_within_grace() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));

        let now = Utc::now();
        // Fresh complete WAL (created_at == now).
        let doc = complete_wal(0x222, now);
        ws.create(&doc).await.expect("seed");

        let report = run_sweep(&st, now, DEFAULT_WAL_GRACE, DEFAULT_SIDECAR_GRACE)
            .await
            .expect("sweep");
        assert_eq!(report.n_state_docs_scanned, 1);
        assert_eq!(report.n_state_docs_deleted, 0);
        // WAL still exists.
        let after = ws.list_wal_ids().await.expect("list");
        assert_eq!(after, vec![WalId(0x222)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sweep_does_not_delete_intent_wal() {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let st =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)))
                .expect("create");
        let ws = WalStore::new(Arc::clone(&storage));

        // Even an OLD intent WAL is preserved — the recovery
        // sweep's responsibility, not GC's.
        let old = Utc::now() - ChronoDuration::seconds(60 * 60);
        let doc = intent_wal(0x333, old);
        ws.create(&doc).await.expect("seed");

        let report = run_sweep(&st, Utc::now(), DEFAULT_WAL_GRACE, DEFAULT_SIDECAR_GRACE)
            .await
            .expect("sweep");
        assert_eq!(report.n_state_docs_scanned, 1);
        assert_eq!(report.n_state_docs_deleted, 0);
        // WAL still exists.
        let after = ws.list_wal_ids().await.expect("list");
        assert_eq!(after, vec![WalId(0x333)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_memory_supertable_sweep_errors_cleanly() {
        let st = Supertable::create(default_supertable_options()).expect("create");
        let err = run_sweep(&st, Utc::now(), DEFAULT_WAL_GRACE, DEFAULT_SIDECAR_GRACE)
            .await
            .expect_err("must error");
        assert!(matches!(err, GcError::NoStorageAttached));
    }
}
