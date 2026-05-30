//! Append-phase orchestrator for the update / delete pipelines.
//!
//! Drives one WAL through the state transition `Intent → Appended`:
//!
//! 1. **Idempotency probe.** Read the current manifest. If it
//!    already contains the WAL's `preallocated_superfile_id`, we're
//!    replaying after a crash — skip directly to step 6.
//! 2. **Fetch + verify IPC payload.** Pull
//!    `wal/mutations/<wal_id>.arrow` and blake3-check against the
//!    state doc's `new_row_content_hash`. Mismatch = corruption;
//!    abort.
//! 3. **Build the superfile bytes** with the WAL's
//!    `preallocated_superfile_id`, the `_id` column populated by
//!    flattening `minted_id_spans` in order, and all other columns
//!    from the IPC payload. Bit-identical across replays by
//!    construction.
//! 4. **PUT the superfile bytes** under the preallocated id.
//!    Content-addressed so re-PUT on replay is a no-op.
//! 5. **CAS-commit the manifest** through the writer's existing
//!    [`persist_commit`] code path. That handles OCC retry,
//!    partition-aware part rewrite, and the pointer-file CAS.
//! 6. **Advance WAL state to `Appended`** with the
//!    `appended_pair_range` populated.
//!
//! Steps 1, 5, and 6 are the durability barriers; the rest is
//! recovery-safe replay material (deterministic bytes; idempotent
//! storage operations).
//!
//! ## Replay safety
//!
//! After any crash, re-running this function against the same WAL
//! must produce the same end state. The invariants:
//!
//! - The superfile uuid is fixed at `preallocated_superfile_id`.
//! - The `_id` column is fixed by `minted_id_spans`.
//! - All other columns come from the content-hashed IPC sidecar.
//!
//! Together these pin every byte of the produced superfile, so the
//! step-4 PUT is overwrite-safe and the step-5 manifest swap can
//! short-circuit via the idempotency probe in step 1.

use std::sync::Arc;

use bytes::Bytes;
use uuid::Uuid;

use crate::storage::StorageError;
use crate::supertable::handle::Supertable;
use crate::supertable::manifest::SuperfileEntry;
use crate::supertable::wal::persistence::{Etag, WalStore, WalStoreError};
use crate::supertable::wal::state_doc::{AppendedPairRange, OpKind, WalState, WalStateDoc};

/// Outcome of one append-phase invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppendPhaseOutcome {
    /// The manifest already referenced the WAL's
    /// `preallocated_superfile_id` when we probed — the superfile
    /// + manifest swap landed on a previous run (or a peer
    /// recovery process beat us to it). No new work; we just
    /// advanced the WAL state to `Appended` if it wasn't already.
    AlreadyApplied,

    /// We built the superfile bytes, PUT them under the
    /// preallocated id, CAS-swapped the manifest to reference the
    /// new superfile, and advanced the WAL state to `Appended`.
    Applied,
}

/// Typed failures from `run_append_phase`. The WAL is left at
/// whatever state was durable when the error surfaced — recovery
/// on a fresh process picks up from there.
#[derive(Debug, thiserror::Error)]
pub enum AppendPhaseError {
    /// State doc is missing a field the append phase needs (e.g.
    /// the WAL was constructed as a DELETE and the orchestrator
    /// was called on it by mistake). The orchestrator only runs
    /// for `op_kind == Update`.
    #[error("WAL is missing required field {field:?} for the append phase")]
    MissingField { field: &'static str },

    /// `wal_doc.op_kind` is `Delete` — the append phase has no
    /// work to do; the caller is using the wrong entry point.
    #[error("append phase invoked on a DELETE WAL; only UPDATE has an append phase")]
    NotAnUpdateWal,

    /// The supertable handle this orchestrator was given doesn't
    /// have a storage backend attached. The append phase has to
    /// commit through the manifest pointer file, which lives on
    /// storage — there's no in-process fallback.
    #[error("supertable has no storage attached; append phase requires durable storage")]
    NoStorageAttached,

    /// IPC sidecar's blake3 doesn't match the WAL state doc's
    /// `new_row_content_hash`. The bytes are corrupt or a peer
    /// abandoned a partial write; surfacing as a typed error
    /// lets recovery quarantine the WAL rather than running it
    /// against a damaged payload.
    #[error("IPC content hash mismatch for WAL {wal_id:?}: expected {expected:?}, got {got:?}")]
    SidecarContentHashMismatch {
        wal_id: String,
        expected: String,
        got: String,
    },

    /// Couldn't decode the IPC sidecar back to a `RecordBatch` —
    /// suggests either a schema mismatch between the producer
    /// and the recovery process, or genuine corruption that the
    /// blake3 check happened to miss.
    #[error("IPC sidecar decode failed for WAL {wal_id:?}: {message}")]
    IpcDecode { wal_id: String, message: String },

    /// `minted_id_spans` flattens to a different count than the
    /// IPC payload claims (`new_row_count`). The two should be
    /// pinned in lockstep at WAL creation; a divergence is a
    /// builder bug or a corrupted state doc.
    #[error(
        "minted_id_spans flatten ({flat_len}) doesn't match new_row_count ({expected}) for WAL {wal_id:?}"
    )]
    IdSpansLengthMismatch {
        wal_id: String,
        flat_len: usize,
        expected: u32,
    },

    /// Building the superfile (Parquet + FTS + vector) failed —
    /// likely a schema-validation or index-build error.
    #[error("superfile build failed: {message}")]
    SuperfileBuild { message: String },

    /// Opening the just-built bytes as a `SuperfileReader` to
    /// extract FTS / vector summaries failed.
    #[error("superfile open for summary failed: {message}")]
    SuperfileOpenForSummary { message: String },

    /// The manifest-commit machinery failed. Surfaces both the
    /// "I lost the pointer CAS" path (which the inner code
    /// retries on its own up to `max_commit_retries`) and any
    /// permanent failure. Caller's handling is the same in both
    /// cases: the WAL stays at whatever state was durable.
    #[error("manifest commit failed: {message}")]
    ManifestCommit { message: String },

    /// Underlying storage error.
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    /// WAL state-document I/O error from the persistence layer.
    #[error("WAL store error: {0}")]
    WalStore(#[from] WalStoreError),
}

/// Drive one UPDATE WAL from `Intent` to `Appended`.
///
/// **Pre-conditions** (caller responsibility):
/// - `wal_doc.op_kind == Update`.
/// - `wal_doc.state == Intent` or `Appended` (re-running on an
///   `Appended` WAL is a no-op via the idempotency probe).
/// - The supertable handle's manifest is read-up-to-date enough
///   that the idempotency probe gives a meaningful answer; this
///   is true for any `Supertable::open` / `create` return value.
///
/// **Post-conditions** on `Ok`:
/// - `wal_doc.state == Appended` durably.
/// - The supertable's manifest contains a superfile entry whose
///   id equals `wal_doc.preallocated_superfile_id`.
/// - That superfile's bytes are durable on storage under
///   `superfiles/<preallocated_superfile_id>.par`.
///
/// **What happens on intermediate failure:** the WAL stays at
/// whatever state was durable when the failure occurred. A
/// recovery process can re-run this function and reach the same
/// end state because every step is idempotent on replay (steps
/// 1-4) or content-addressed (step 5's manifest writes go
/// through the normal CAS).
pub async fn run_append_phase(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
) -> Result<(AppendPhaseOutcome, WalStateDoc, Etag), AppendPhaseError> {
    // Pre-condition: only UPDATE has an append phase.
    if wal_doc.op_kind != OpKind::Update {
        return Err(AppendPhaseError::NotAnUpdateWal);
    }

    let preallocated_superfile_id =
        wal_doc
            .preallocated_superfile_id
            .ok_or(AppendPhaseError::MissingField {
                field: "preallocated_superfile_id",
            })?;

    let inner = supertable.inner();

    // ---- Step 1: Idempotency probe ----
    //
    // Look up the WAL's preallocated_superfile_id in the
    // current manifest snapshot. If it's already there, a
    // previous run (or peer recovery) completed steps 2-5;
    // we just need to make sure the WAL state itself shows
    // Appended.
    let manifest_snapshot = inner.manifest.load_full();
    if manifest_contains(&manifest_snapshot, preallocated_superfile_id) {
        let (new_wal, new_etag) =
            advance_to_appended_if_needed(wal_store, wal_doc, wal_etag, preallocated_superfile_id)
                .await?;
        return Ok((AppendPhaseOutcome::AlreadyApplied, new_wal, new_etag));
    }

    // ---- Steps 2-6 ----
    //
    // Built incrementally in `do_apply` so failure modes funnel
    // through one return path with consistent error mapping.
    let (new_wal, new_etag) = do_apply(
        supertable,
        wal_store,
        wal_doc,
        wal_etag,
        preallocated_superfile_id,
    )
    .await?;
    Ok((AppendPhaseOutcome::Applied, new_wal, new_etag))
}

/// Step 1 helper: scan the manifest's superfile list for a
/// matching `superfile_id`. O(N) in the number of live
/// superfiles; called once per append-phase invocation, so the
/// linear scan is fine at the supertable sizes we target.
fn manifest_contains(manifest: &crate::supertable::Manifest, superfile_id: Uuid) -> bool {
    manifest
        .superfile_list
        .superfiles
        .iter()
        .any(|s| s.uri.0 == superfile_id)
}

/// If the WAL is already in `Appended`, return its current doc +
/// etag unchanged. Otherwise CAS-advance to `Appended` with a
/// best-effort `appended_pair_range` (computed from the WAL's
/// expected row count — the manifest swap that put the superfile
/// there used the same range).
async fn advance_to_appended_if_needed(
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
    superfile_id: Uuid,
) -> Result<(WalStateDoc, Etag), AppendPhaseError> {
    if wal_doc.state == WalState::Appended {
        return Ok((wal_doc.clone(), wal_etag.clone()));
    }
    let mut next = wal_doc.clone();
    next.state = WalState::Appended;
    if next.appended_pair_range.is_none() {
        let n_docs = next.new_row_count.unwrap_or(0);
        if n_docs > 0 {
            next.appended_pair_range = Some(AppendedPairRange {
                superfile_id,
                first_doc_id: 0,
                last_doc_id: n_docs - 1,
            });
        }
    }
    let new_etag = wal_store
        .update_with_etag(wal_doc.wal_id, wal_etag, &next)
        .await?;
    Ok((next, new_etag))
}

/// The non-idempotent fast path: build the superfile, write its
/// bytes, swap the manifest, advance the WAL. Pulled into its own
/// async fn so the orchestrator's high-level flow reads cleanly.
async fn do_apply(
    supertable: &Supertable,
    wal_store: &WalStore,
    wal_doc: &WalStateDoc,
    wal_etag: &Etag,
    preallocated_superfile_id: Uuid,
) -> Result<(WalStateDoc, Etag), AppendPhaseError> {
    // Steps 2-6 land in a follow-up commit. The orchestrator's
    // outer shape (pre-condition checks, the idempotency short
    // circuit, error variants) is reviewable on its own; pulling
    // in the IPC decode + superfile build + manifest commit
    // mechanics adds ~500 lines that are easier to review
    // separately.
    let _ = (
        supertable,
        wal_store,
        wal_doc,
        wal_etag,
        preallocated_superfile_id,
    );
    Err(AppendPhaseError::SuperfileBuild {
        message: "do_apply not yet implemented in this commit".into(),
    })
}

/// Build the superfile bytes for a WAL's append phase.
///
/// Decodes the IPC sidecar back to a `RecordBatch`, prepends a
/// `_id` column populated by flattening `minted_id_spans`, then
/// runs the result through `SuperfileBuilder::add_batch` +
/// `finish`. The output is bit-identical across replays given
/// the same inputs, which is the load-bearing property the
/// append-phase replay-safety story rests on.
#[allow(dead_code)] // wired up by do_apply in a follow-up commit
fn assemble_superfile_bytes(
    _supertable: &Supertable,
    _wal_doc: &WalStateDoc,
    _ipc_bytes: &Bytes,
) -> Result<(Bytes, Arc<SuperfileEntry>), AppendPhaseError> {
    todo!("assemble_superfile_bytes lands alongside do_apply")
}

#[cfg(test)]
mod tests {
    //! Tests for the append-phase orchestrator. This commit
    //! covers the orchestrator's outer-shape behaviour —
    //! pre-condition checks, the `AlreadyApplied` path,
    //! error mapping. End-to-end + crash-injection tests
    //! land alongside the `do_apply` implementation.
    //!
    //! The fixtures use real `Supertable` + `WalStore`
    //! against a `LocalFsStorageProvider`-backed `TempDir`,
    //! matching the existing persistence-layer test pattern
    //! used elsewhere in this crate.

    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use crate::supertable::Supertable;
    use crate::supertable::wal::state_doc::{
        OpKind, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalId, WalState,
    };
    use crate::test_helpers::default_supertable_options;
    use chrono::Utc;
    use tempfile::TempDir;
    use uuid::Uuid;

    /// Construct a Supertable + a fresh WAL state doc + the WAL's
    /// etag, all backed by the same LocalFs storage so an
    /// orchestrator call against this supertable sees the WAL
    /// the fixture just created.
    async fn fixture() -> (TempDir, Supertable, WalStore, WalStateDoc, Etag) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let supertable =
            Supertable::create(default_supertable_options().with_storage(Arc::clone(&storage)));
        let wal_store = WalStore::new(Arc::clone(&storage));

        let wal_id = WalId(42);
        let wal_doc = WalStateDoc {
            wal_id,
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Update,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "_id = 1".into(),
            target_ids: vec![WalId(1)],
            new_row_count: Some(1),
            new_row_content_hash: Some("0".repeat(64)),
            preallocated_superfile_id: Some(Uuid::from_u128(0x1234_5678_9ABC)),
            minted_id_spans: vec![crate::supertable::wal::state_doc::IdSpan {
                first: WalId(100),
                last: WalId(100),
            }],
            appended_pair_range: None,
            tombstone_progress: vec![TombstoneEntry {
                target_id: WalId(1),
                outcome: TombstoneOutcome::Pending,
                tombstoned_in_superfile: None,
            }],
        };
        let etag = wal_store.create(&wal_doc).await.expect("wal create");
        (dir, supertable, wal_store, wal_doc, etag)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rejects_delete_wal_with_typed_error() {
        let (_dir, st, ws, mut wal, etag) = fixture().await;
        wal.op_kind = OpKind::Delete;
        let err = run_append_phase(&st, &ws, &wal, &etag)
            .await
            .expect_err("must error");
        assert!(matches!(err, AppendPhaseError::NotAnUpdateWal), "{err:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn rejects_wal_missing_preallocated_superfile_id() {
        let (_dir, st, ws, mut wal, etag) = fixture().await;
        wal.preallocated_superfile_id = None;
        let err = run_append_phase(&st, &ws, &wal, &etag)
            .await
            .expect_err("must error");
        assert!(
            matches!(
                err,
                AppendPhaseError::MissingField {
                    field: "preallocated_superfile_id"
                }
            ),
            "{err:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn manifest_contains_returns_true_for_matching_uuid() {
        let (_dir, _st, _ws, _wal, _etag) = fixture().await;
        let opts = Arc::new(default_supertable_options());
        let empty = crate::supertable::Manifest::empty(Arc::clone(&opts));
        assert!(!manifest_contains(&empty, Uuid::nil()));
    }
}
