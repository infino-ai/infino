// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Storage-level CAS primitives for WAL state documents.
//!
//! The whole crate's WAL-state I/O goes through `WalStore`.
//! That funnel is load-bearing: the etag CAS contract is
//! enforced in one place, so a grep confirms no other code
//! path can race on the state doc.
//!
//! ## What this module owns
//!
//! - `create(state)` — write a brand-new WAL state doc atomically;
//!   fails if the `wal_id`'s path already exists.
//! - `read(wal_id)` — fetch the current state doc + its etag.
//! - `update_with_etag(wal_id, expected_etag, new_state)` —
//!   CAS update against the etag captured at `read` time.
//! - `delete(wal_id)` — best-effort cleanup at COMPLETE.
//! - Sidecar helpers (`put_sidecar` / `get_sidecar`) for the
//!   per-WAL `*.arrow` payload and the per-superfile `.tombstones`
//!   object — both overwrite-safe; content-addressing handled by
//!   callers via blake3 verification.
//!
//! ## What this module does NOT own
//!
//! No state-machine logic, no lease handling, no recovery
//! orchestration. Those live in the pipeline layer that sits
//! on top. `WalStore` is purely the "talk to storage" layer.
//!
//! ## ETag capture strategy
//!
//! Reads and writes pick up the etag in the same round trip:
//! `StorageProvider::get` returns `(Bytes, ObjectMeta)` from
//! the backend's `GetResult` directly, and `put_atomic` /
//! `put_if_match` return the new etag from the `PutResult`.
//! `Ok(None)` on the put paths collapses to the empty `Etag`,
//! a legal "create-only-if-absent" input on the next CAS hop.
//! The state machine never mints two WALs at the same path,
//! so a missing etag doesn't break the chain.

use std::sync::Arc;

use bytes::Bytes;
use thiserror::Error;

use crate::{
    storage::{StorageError, StorageProvider},
    supertable::wal::{
        state_doc::{WalId, WalStateDoc},
        tombstones_codec::{self, SidecarCodecError, TombstonesSidecar},
    },
};

/// Storage backend's opaque version identifier. Treated as a
/// type alias rather than a newtype because:
///
/// - Existing code (manifest pointer commit, disk cache) uses
///   the bare `String` form against the same backend trait.
/// - The etag never leaves the supertable layer — there's no
///   FFI surface where confusing it with another `String` would
///   be plausible.
pub type Etag = String;

/// Errors surfaced by `WalStore`. CAS-loss carries the
/// observed etag (`None` when storage returned no etag, e.g.
/// the LocalFs path on the fast path) so callers that
/// transition through multiple states can refresh their view
/// in one call.
#[derive(Debug, Error)]
pub enum WalStoreError {
    /// The CAS precondition failed: someone else updated this
    /// WAL between our read and our PUT, or the WAL was
    /// concurrently deleted by recovery's GC. Caller's correct
    /// response is to re-read and re-evaluate (or surrender if
    /// the conflict isn't recoverable). The WAL path is
    /// included for log clarity.
    #[error("CAS failed for {path:?}: storage etag has advanced past expected")]
    CasFailed { path: String },

    /// `create()` collided with an existing object at the same
    /// path. Distinct from `CasFailed` so callers can tell apart
    /// "I lost a CAS race" (CasFailed) from "I tried to create
    /// a duplicate WAL" (AlreadyExists — typically only fires
    /// under a `wal_id` collision, which is astronomically
    /// unlikely given a 40-bit random worker_id + 24-bit
    /// per-ms sequence in the id generator).
    #[error("WAL state doc already exists at {path:?}")]
    AlreadyExists { path: String },

    /// `read()` against a path that has no object. Recovery
    /// treats this as "the WAL was completed and GC'd" rather
    /// than a hard error.
    #[error("WAL state doc not found at {path:?}")]
    NotFound { path: String },

    /// Storage layer surfaced an error we didn't translate above.
    /// Passthrough for log + alert.
    #[error("storage error at {path:?}: {source}")]
    Storage {
        path: String,
        #[source]
        source: StorageError,
    },

    /// JSON encode/decode failure on the state-doc payload.
    /// Should never fire under normal conditions — the doc shape
    /// is internal and unit-tested.
    #[error("state-doc serde error at {path:?}: {source}")]
    Serde {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    /// Sidecar codec error surfaced on tombstone-sidecar GET.
    #[error("sidecar codec error at {path:?}: {source}")]
    SidecarCodec {
        path: String,
        #[source]
        source: SidecarCodecError,
    },

    /// Sidecar content-hash mismatch surfaced on the `.arrow`
    /// payload path: caller pinned a blake3, but the bytes we
    /// fetched hash to something else. Suggests storage
    /// corruption or a partial-write that another peer abandoned.
    #[error("sidecar content hash mismatch at {path:?}: expected {expected:?}, got {got:?}")]
    SidecarContentHashMismatch {
        path: String,
        expected: String,
        got: String,
    },
}

/// Storage prefix for WAL state-doc + sidecar objects.
const WAL_DIR: &str = "wal/mutations";

/// File extension for the state-doc JSON.
const STATE_EXT: &str = "json";

/// File extension for the IPC sidecar (UPDATE only).
const ARROW_EXT: &str = "arrow";

/// Storage prefix for per-superfile tombstone sidecars.
const SUPERFILES_DIR: &str = "superfiles";

/// File extension for tombstone sidecars.
const TOMBSTONES_EXT: &str = "tombstones";

/// CAS-funnel for WAL state-doc I/O. Construct once per
/// supertable; all WAL state transitions go through this type.
///
/// `WalStore` is cheap to clone (an `Arc<dyn StorageProvider>`
/// behind a thin wrapper) and is `Send + Sync` so writers,
/// recovery sweeps, and GC sweeps can share one instance.
#[derive(Debug, Clone)]
pub struct WalStore {
    storage: Arc<dyn StorageProvider>,
}

impl WalStore {
    /// Construct a `WalStore` that talks to the given storage
    /// provider. No I/O at construction time.
    pub fn new(storage: Arc<dyn StorageProvider>) -> Self {
        Self { storage }
    }

    /// Path helpers — local to the module so any future
    /// directory-layout change happens in exactly one place.
    fn state_path(wal_id: WalId) -> String {
        format!("{WAL_DIR}/{}.{STATE_EXT}", wal_id.to_hex())
    }

    fn arrow_path(wal_id: WalId) -> String {
        format!("{WAL_DIR}/{}.{ARROW_EXT}", wal_id.to_hex())
    }

    fn tombstones_path(superfile_id: uuid::Uuid) -> String {
        // UUID's default `Display` is the hyphenated 36-char hex
        // string we use everywhere else in the codebase for
        // superfile identifiers.
        format!("{SUPERFILES_DIR}/{superfile_id}.{TOMBSTONES_EXT}")
    }

    /// Enumerate every WAL state-doc currently in `wal/mutations/`
    /// and return their parsed [`WalId`]s, sorted ascending.
    ///
    /// Sorted-ascending is significant: `WalId` is Snowflake-shaped
    /// (64-bit ms timestamp prefix), so ascending == oldest-first.
    /// The recovery sweep walks the result in that order so older
    /// leftover WALs get drained before fresher ones — bounded
    /// per-sweep latency on backlogged supertables.
    ///
    /// Objects whose filename doesn't parse as `<hex>.json` are
    /// silently skipped. `.arrow` sidecars live under the same
    /// prefix and are excluded the same way.
    pub async fn list_wal_ids(&self) -> Result<Vec<WalId>, WalStoreError> {
        let uris = self
            .storage
            .list_with_prefix(WAL_DIR)
            .await
            .map_err(|source| WalStoreError::Storage {
                path: WAL_DIR.into(),
                source,
            })?;
        let suffix = format!(".{STATE_EXT}");
        let mut out: Vec<WalId> = Vec::new();
        for uri in uris {
            let filename = match uri.rsplit_once('/') {
                Some((_, fname)) => fname,
                None => uri.as_str(),
            };
            let Some(stem) = filename.strip_suffix(&suffix) else {
                continue;
            };
            let Ok(id) = WalId::from_hex(stem) else {
                continue;
            };
            out.push(id);
        }
        out.sort_unstable_by_key(|w| w.0);
        Ok(out)
    }

    /// Return the [`uuid::Uuid`]s of every superfile that currently
    /// has a tombstone sidecar on storage (`superfiles/<id>.tombstones`).
    pub async fn list_tombstone_ids(&self) -> Result<Vec<uuid::Uuid>, WalStoreError> {
        let uris = self
            .storage
            .list_with_prefix(SUPERFILES_DIR)
            .await
            .map_err(|source| WalStoreError::Storage {
                path: SUPERFILES_DIR.into(),
                source,
            })?;
        let suffix = format!(".{TOMBSTONES_EXT}");
        let mut out = Vec::new();
        for uri in uris {
            let filename = match uri.rsplit_once('/') {
                Some((_, fname)) => fname,
                None => uri.as_str(),
            };
            let Some(stem) = filename.strip_suffix(&suffix) else {
                continue;
            };
            let Ok(id) = uuid::Uuid::parse_str(stem) else {
                continue;
            };
            out.push(id);
        }
        Ok(out)
    }

    /// Write a brand-new WAL state doc atomically. Fails with
    /// `AlreadyExists` if the `wal_id`'s path is occupied —
    /// which is how a `wal_id` collision surfaces. Probability
    /// is vanishingly small given the 128-bit id space, but
    /// it's a real fault mode we surface cleanly.
    ///
    /// Returns the etag of the newly-written object — surfaced
    /// directly by [`StorageProvider::put_atomic`]. Backends
    /// that don't carry an etag (LocalFs without xattr support)
    /// surface `None`, which collapses to the empty [`Etag`] —
    /// a legal input to `put_if_match` (interpreted as
    /// "create-only-if-absent"). The state machine never mints
    /// two WALs at the same path, so a missing etag doesn't
    /// break the CAS chain.
    pub async fn create(&self, state: &WalStateDoc) -> Result<Etag, WalStoreError> {
        let path = Self::state_path(state.wal_id);
        let body = serde_json::to_vec(state).map_err(|e| WalStoreError::Serde {
            path: path.clone(),
            source: e,
        })?;
        match self.storage.put_atomic(&path, Bytes::from(body)).await {
            Ok(etag) => Ok(etag.unwrap_or_default()),
            Err(StorageError::PreconditionFailed { .. }) => {
                Err(WalStoreError::AlreadyExists { path })
            }
            Err(other) => Err(WalStoreError::Storage {
                path,
                source: other,
            }),
        }
    }

    /// Fetch the current state doc + its etag. `NotFound`
    /// surfaces as a typed variant so recovery can treat it as
    /// "WAL completed and GC'd" without lifting `StorageError`
    /// out of its abstraction.
    pub async fn read(&self, wal_id: WalId) -> Result<(WalStateDoc, Etag), WalStoreError> {
        let path = Self::state_path(wal_id);
        let (bytes, meta) = match self.storage.get(&path).await {
            Ok(pair) => pair,
            Err(StorageError::NotFound { .. }) => {
                return Err(WalStoreError::NotFound { path });
            }
            Err(other) => {
                return Err(WalStoreError::Storage {
                    path,
                    source: other,
                });
            }
        };
        let state: WalStateDoc =
            serde_json::from_slice(&bytes).map_err(|e| WalStoreError::Serde {
                path: path.clone(),
                source: e,
            })?;
        Ok((state, meta.etag.unwrap_or_default()))
    }

    /// CAS-update the state doc against the etag captured at
    /// the previous `read` (or `create`). The etag of the
    /// newly-written object is returned for the next link in
    /// the CAS chain.
    ///
    /// `CasFailed` means the storage's current etag has
    /// advanced past `expected_etag` — typically a peer
    /// recovery process beat us to a step. Callers handle
    /// CAS-loss according to the state-machine rules at the
    /// pipeline layer.
    pub async fn update_with_etag(
        &self,
        wal_id: WalId,
        expected_etag: &Etag,
        new_state: &WalStateDoc,
    ) -> Result<Etag, WalStoreError> {
        let path = Self::state_path(wal_id);
        let body = serde_json::to_vec(new_state).map_err(|e| WalStoreError::Serde {
            path: path.clone(),
            source: e,
        })?;
        let expected_opt = if expected_etag.is_empty() {
            // Empty etag string = "we never observed an etag for
            // this object." Send `None` so the backend interprets
            // as create-only. This path covers the LocalFs case
            // where `head` returned no etag.
            None
        } else {
            Some(expected_etag.as_str())
        };
        match self
            .storage
            .put_if_match(&path, Bytes::from(body), expected_opt)
            .await
        {
            Ok(etag) => Ok(etag.unwrap_or_default()),
            Err(StorageError::PreconditionFailed { .. }) => Err(WalStoreError::CasFailed { path }),
            Err(StorageError::NotFound { .. }) => {
                // `put_if_match` against a deleted object can
                // surface NotFound depending on backend.
                // Recovery + GC may have raced; same logical
                // outcome as CAS-loss from the caller's view.
                Err(WalStoreError::CasFailed { path })
            }
            Err(other) => Err(WalStoreError::Storage {
                path,
                source: other,
            }),
        }
    }

    /// Best-effort DELETE of a state doc. Idempotent in the
    /// storage trait (a missing path returns Ok), so this never
    /// fails on the missing-target case. Used as the cleanup
    /// step once a WAL has reached COMPLETE, and by the
    /// background sweep that catches WALs whose inline cleanup
    /// failed.
    pub async fn delete_state(&self, wal_id: WalId) -> Result<(), WalStoreError> {
        let path = Self::state_path(wal_id);
        self.storage
            .delete(&path)
            .await
            .map_err(|source| WalStoreError::Storage {
                path: path.clone(),
                source,
            })?;
        Ok(())
    }

    /// Best-effort DELETE of an Arrow IPC payload sidecar.
    /// Idempotent same as `delete_state`.
    pub async fn delete_arrow(&self, wal_id: WalId) -> Result<(), WalStoreError> {
        let path = Self::arrow_path(wal_id);
        self.storage
            .delete(&path)
            .await
            .map_err(|source| WalStoreError::Storage {
                path: path.clone(),
                source,
            })?;
        Ok(())
    }

    // ---- Arrow IPC sidecar (per-WAL `new_rows` payload) -----------------

    /// Write the IPC payload sidecar (UPDATE only). **Idempotent
    /// on bit-identical content** — recovery replay must be able
    /// to re-PUT the sidecar without breaking, but the storage
    /// trait only offers create-only (`put_atomic`) or CAS
    /// (`put_if_match`); it has no `PutMode::Overwrite`. So we
    /// route through `put_atomic` and swallow the
    /// `PreconditionFailed` that fires on a second write.
    ///
    /// The replay-safety argument: the sidecar's bytes are a
    /// pure function of `new_rows`, which is itself fixed at the
    /// caller's `update()` call. The state doc's
    /// `new_row_content_hash` field pins those bytes; a recovery
    /// process that re-PUTs is writing content with the same
    /// blake3, so the existing object IS the right object. The
    /// PUT being a no-op is fine.
    ///
    /// What this does NOT defend against: a caller writing
    /// *different* bytes to the same `wal_id`. That can't happen
    /// in the production flow (each `update()` mints a fresh
    /// `wal_id`); a test that does it deliberately will silently
    /// keep the first write. The pipeline's later
    /// `new_row_content_hash` check catches the divergence.
    pub async fn put_arrow(&self, wal_id: WalId, bytes: Bytes) -> Result<(), WalStoreError> {
        let path = Self::arrow_path(wal_id);
        match self.storage.put_atomic(&path, bytes).await {
            Ok(_) => Ok(()),
            // Second-write-of-same-bytes path (recovery replay).
            // Caller guarantees the bytes are bit-identical via
            // the WAL's content-hash invariant; the existing
            // object is correct as-is.
            Err(StorageError::PreconditionFailed { .. }) => Ok(()),
            Err(source) => Err(WalStoreError::Storage { path, source }),
        }
    }

    /// Fetch the IPC payload sidecar. If `expected_blake3_hex`
    /// is `Some`, the fetched bytes' blake3 is checked against
    /// it and a mismatch surfaces as
    /// `SidecarContentHashMismatch`. Passing `None` skips the
    /// check (used by tests that don't have a pinned hash).
    pub async fn get_arrow(
        &self,
        wal_id: WalId,
        expected_blake3_hex: Option<&str>,
    ) -> Result<Bytes, WalStoreError> {
        let path = Self::arrow_path(wal_id);
        let (bytes, _) =
            self.storage
                .get(&path)
                .await
                .map_err(|source| WalStoreError::Storage {
                    path: path.clone(),
                    source,
                })?;
        if let Some(want) = expected_blake3_hex {
            let got = blake3::hash(&bytes).to_hex().to_string();
            if got != want {
                return Err(WalStoreError::SidecarContentHashMismatch {
                    path,
                    expected: want.to_string(),
                    got,
                });
            }
        }
        Ok(bytes)
    }

    // ---- Tombstone sidecar (per-superfile bitmap) -----------------------

    /// Read the tombstone sidecar for one superfile, returning
    /// both the parsed shape AND the etag for subsequent CAS
    /// writes. `NotFound` is mapped to `Ok(None)` because an
    /// absent sidecar is the legal "no tombstones yet" state
    /// rather than an error.
    ///
    /// Returns `Ok(None)` when no sidecar exists, otherwise
    /// `Ok(Some((sidecar, etag)))`.
    pub async fn get_tombstones(
        &self,
        superfile_id: uuid::Uuid,
    ) -> Result<Option<(TombstonesSidecar, Etag)>, WalStoreError> {
        let path = Self::tombstones_path(superfile_id);
        let (bytes, meta) = match self.storage.get(&path).await {
            Ok(pair) => pair,
            Err(StorageError::NotFound { .. }) => return Ok(None),
            Err(other) => {
                return Err(WalStoreError::Storage {
                    path,
                    source: other,
                });
            }
        };
        let sidecar = tombstones_codec::decode_sidecar(&bytes).map_err(|source| {
            WalStoreError::SidecarCodec {
                path: path.clone(),
                source,
            }
        })?;
        Ok(Some((sidecar, meta.etag.unwrap_or_default())))
    }

    /// CAS-PUT a tombstone sidecar. `expected_etag = None`
    /// requests create-only (used on the first write to a
    /// previously-absent sidecar); `Some(etag)` requests CAS
    /// against that exact version. Returns the new etag for the
    /// next CAS hop.
    ///
    /// `CasFailed` surfaces both true CAS-loss AND the sealed-
    /// sidecar case (a concurrent compactor sealed between our
    /// read and our PUT); callers distinguish by re-reading and
    /// inspecting `sidecar.seal` — a `Some(_)` seal post-CAS
    /// means the writer must back off and re-resolve against
    /// the merged target.
    pub async fn put_tombstones(
        &self,
        superfile_id: uuid::Uuid,
        expected_etag: Option<&Etag>,
        sidecar: &TombstonesSidecar,
    ) -> Result<Etag, WalStoreError> {
        let path = Self::tombstones_path(superfile_id);
        let bytes = tombstones_codec::encode_sidecar(sidecar).map_err(|source| {
            WalStoreError::SidecarCodec {
                path: path.clone(),
                source,
            }
        })?;
        let expected_opt =
            expected_etag.and_then(|e| if e.is_empty() { None } else { Some(e.as_str()) });
        match self
            .storage
            .put_if_match(&path, Bytes::from(bytes), expected_opt)
            .await
        {
            Ok(etag) => Ok(etag.unwrap_or_default()),
            Err(StorageError::PreconditionFailed { .. }) => Err(WalStoreError::CasFailed { path }),
            Err(other) => Err(WalStoreError::Storage {
                path,
                source: other,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use tempfile::TempDir;

    use super::*;
    use crate::{
        storage::LocalFsStorageProvider,
        supertable::wal::state_doc::{
            OpKind, RowId, SCHEMA_VERSION, TombstoneEntry, TombstoneOutcome, WalState,
        },
    };

    fn store() -> (TempDir, WalStore) {
        let dir = TempDir::new().expect("tempdir");
        let provider: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        (dir, WalStore::new(provider))
    }

    fn sample_state(wal_id: i128) -> WalStateDoc {
        WalStateDoc {
            wal_id: WalId(wal_id),
            schema_version: SCHEMA_VERSION,
            op_kind: OpKind::Delete,
            state: WalState::Intent,
            created_at: Utc::now(),
            lease: None,
            predicate_repr: "for tests".into(),
            target_ids: vec![RowId(1), RowId(2)],
            new_row_count: None,
            new_row_content_hash: None,
            preallocated_superfile_id: None,
            minted_id_spans: Vec::new(),
            tombstone_progress: vec![
                TombstoneEntry {
                    target_id: RowId(1),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                },
                TombstoneEntry {
                    target_id: RowId(2),
                    outcome: TombstoneOutcome::Pending,
                    tombstoned_in_superfile: None,
                },
            ],
        }
    }

    // ---- Happy-path round-trip --------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_then_read_roundtrips_state() {
        let (_dir, ws) = store();
        let state = sample_state(7);
        let etag = ws.create(&state).await.expect("create");
        let (read_state, read_etag) = ws.read(state.wal_id).await.expect("read");
        assert_eq!(read_state, state);
        // LocalFs returns a non-empty etag — accept whatever
        // shape it produces; we just need consistency.
        assert_eq!(read_etag, etag);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn read_missing_returns_not_found() {
        let (_dir, ws) = store();
        let err = ws.read(WalId(9999)).await.expect_err("must error");
        assert!(matches!(err, WalStoreError::NotFound { .. }), "{err:?}");
    }

    // ---- create twice fails ----------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn create_twice_fails_with_already_exists() {
        let (_dir, ws) = store();
        let state = sample_state(11);
        ws.create(&state).await.expect("first create");
        let err = ws.create(&state).await.expect_err("second must fail");
        assert!(
            matches!(err, WalStoreError::AlreadyExists { .. }),
            "{err:?}"
        );
    }

    // ---- update_with_etag CAS contract -----------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_with_correct_etag_advances_state() {
        let (_dir, ws) = store();
        let mut state = sample_state(13);
        let e1 = ws.create(&state).await.expect("create");
        state.state = WalState::Appended;
        let e2 = ws
            .update_with_etag(state.wal_id, &e1, &state)
            .await
            .expect("update");
        // Etag advanced — bytes (and so etag) differ.
        assert_ne!(e1, e2, "etag must advance after a real write");
        let (read_state, read_etag) = ws.read(state.wal_id).await.expect("read");
        assert_eq!(read_state.state, WalState::Appended);
        assert_eq!(read_etag, e2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn update_with_stale_etag_fails_cas() {
        let (_dir, ws) = store();
        let mut state = sample_state(17);
        let e1 = ws.create(&state).await.expect("create");
        // First update advances etag past e1.
        state.state = WalState::Appended;
        let _ = ws
            .update_with_etag(state.wal_id, &e1, &state)
            .await
            .expect("first update");
        // Second update against the stale e1 must lose CAS.
        state.state = WalState::Complete;
        let err = ws
            .update_with_etag(state.wal_id, &e1, &state)
            .await
            .expect_err("stale etag must lose");
        assert!(matches!(err, WalStoreError::CasFailed { .. }), "{err:?}");
    }

    // ---- N-way CAS race ---------------------------------------------------

    /// Note on `N`: the CAS contract — "exactly one writer wins
    /// against a shared expected_etag; all others see CasFailed"
    /// — is a *property* that's the same at N=2 as at N=100.
    /// This test runs at N=12 because the
    /// `LocalFsStorageProvider` we use in tests serializes every
    /// `put_if_match` through a single process-wide `flock` on
    /// `_supertable/.lock` to bracket the read-then-overwrite
    /// TOCTOU window. `flock` is acquired via a sync blocking
    /// call on a tokio worker thread; the lock holder then
    /// `.await`s I/O. If more contenders than worker threads
    /// pile up, all workers block on `lock_exclusive` while the
    /// holder needs a free worker to resume its own `.await`,
    /// deadlocking the runtime. Real S3 / GCS backends use
    /// native conditional PUT and don't have this scaling
    /// limit — when this test runs against an S3 mock instead
    /// of LocalFs, N can be bumped without changing the
    /// property check.
    ///
    /// `worker_threads = 16` + `N = 12` gives 4 idle workers for
    /// the runtime's blocking-pool dispatch + any awaited I/O,
    /// while keeping the test under a second.
    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn concurrent_updates_have_exactly_one_winner() {
        let (_dir, ws) = store();
        let state = sample_state(23);
        let initial_etag = ws.create(&state).await.expect("create");

        const N: usize = 12;
        let ws = Arc::new(ws);
        let mut handles = Vec::with_capacity(N);
        for i in 0..N {
            let ws = Arc::clone(&ws);
            let etag = initial_etag.clone();
            let mut s = state.clone();
            // Every task targets WalState::Appended with a
            // distinct predicate_repr so we can confirm
            // exactly one body landed.
            s.state = WalState::Appended;
            s.predicate_repr = format!("racer {i}");
            handles.push(tokio::spawn(async move {
                ws.update_with_etag(s.wal_id, &etag, &s).await
            }));
        }

        let mut ok_count = 0usize;
        let mut cas_failed = 0usize;
        let mut other_err = 0usize;
        for h in handles {
            match h.await.expect("join") {
                Ok(_) => ok_count += 1,
                Err(WalStoreError::CasFailed { .. }) => cas_failed += 1,
                Err(_) => other_err += 1,
            }
        }
        assert_eq!(ok_count, 1, "exactly one task must win CAS");
        assert_eq!(cas_failed, N - 1, "all losers must report CasFailed");
        assert_eq!(other_err, 0, "no spurious failures");
    }

    // ---- Arrow sidecar (overwrite-safe + content hash check) -------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn arrow_sidecar_round_trips_with_hash_verify() {
        let (_dir, ws) = store();
        let wal_id = WalId(29);
        let payload = Bytes::from_static(b"hello-payload");
        ws.put_arrow(wal_id, payload.clone()).await.expect("put");
        let hash = blake3::hash(&payload).to_hex().to_string();
        let got = ws
            .get_arrow(wal_id, Some(&hash))
            .await
            .expect("hashed read");
        assert_eq!(got, payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn arrow_sidecar_overwrite_is_legal() {
        let (_dir, ws) = store();
        let wal_id = WalId(31);
        ws.put_arrow(wal_id, Bytes::from_static(b"first"))
            .await
            .expect("first");
        // Bit-identical re-write must succeed (recovery replay).
        ws.put_arrow(wal_id, Bytes::from_static(b"first"))
            .await
            .expect("idempotent re-write");
        // Replacement bytes also succeed under our overwrite-
        // safe semantic — verified by the no-error result.
        ws.put_arrow(wal_id, Bytes::from_static(b"second"))
            .await
            .expect("replacement");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn arrow_sidecar_hash_mismatch_surfaces_typed_error() {
        let (_dir, ws) = store();
        let wal_id = WalId(37);
        ws.put_arrow(wal_id, Bytes::from_static(b"actual"))
            .await
            .expect("put");
        let bogus_hash = "00".repeat(32);
        let err = ws
            .get_arrow(wal_id, Some(&bogus_hash))
            .await
            .expect_err("hash check must fail");
        match err {
            WalStoreError::SidecarContentHashMismatch { expected, got, .. } => {
                assert_eq!(expected, bogus_hash);
                assert_ne!(got, bogus_hash);
            }
            other => panic!("expected SidecarContentHashMismatch; got {other:?}"),
        }
    }

    // ---- Tombstone sidecar (per-superfile RoaringBitmap) -----------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstones_sidecar_absent_returns_none() {
        let (_dir, ws) = store();
        let got = ws.get_tombstones(uuid::Uuid::nil()).await.expect("query");
        assert!(got.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstones_sidecar_roundtrips_through_storage() {
        let (_dir, ws) = store();
        let superfile_id = uuid::Uuid::from_u128(0xCAFE_BABE_DEAD_BEEF);
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(7);
        bitmap.insert(42);
        let sidecar = TombstonesSidecar {
            seal: None,
            bitmap: bitmap.clone(),
        };
        let etag1 = ws
            .put_tombstones(superfile_id, None, &sidecar)
            .await
            .expect("first put");
        let (got, etag_read) = ws
            .get_tombstones(superfile_id)
            .await
            .expect("get")
            .expect("present");
        assert!(got.seal.is_none());
        let got_ids: Vec<u32> = got.bitmap.iter().collect();
        let expected_ids: Vec<u32> = bitmap.iter().collect();
        assert_eq!(got_ids, expected_ids);
        assert_eq!(etag_read, etag1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tombstones_sidecar_stale_etag_fails_cas() {
        let (_dir, ws) = store();
        let superfile_id = uuid::Uuid::from_u128(0xFEED_FACE_BEEF_CAFE);
        let initial = TombstonesSidecar {
            seal: None,
            bitmap: roaring::RoaringBitmap::new(),
        };
        let etag1 = ws
            .put_tombstones(superfile_id, None, &initial)
            .await
            .expect("first put");
        let mut bumped_bitmap = roaring::RoaringBitmap::new();
        bumped_bitmap.insert(3);
        let bumped = TombstonesSidecar {
            seal: None,
            bitmap: bumped_bitmap,
        };
        // Bump under etag1 — succeeds.
        let _etag2 = ws
            .put_tombstones(superfile_id, Some(&etag1), &bumped)
            .await
            .expect("update");
        // Stale CAS — must fail.
        let err = ws
            .put_tombstones(superfile_id, Some(&etag1), &bumped)
            .await
            .expect_err("stale etag");
        assert!(matches!(err, WalStoreError::CasFailed { .. }), "{err:?}");
    }

    // ---- delete_state is idempotent --------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn delete_state_is_idempotent() {
        let (_dir, ws) = store();
        let state = sample_state(41);
        ws.create(&state).await.expect("create");
        ws.delete_state(state.wal_id).await.expect("first delete");
        // Second delete against absent path is Ok per storage
        // trait's idempotent-delete contract.
        ws.delete_state(state.wal_id).await.expect("second delete");
    }

    // ---- delete_arrow is idempotent --------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn delete_arrow_is_idempotent_and_removes_sidecar() {
        let (_dir, ws) = store();
        let wal_id = WalId(43);
        ws.put_arrow(wal_id, Bytes::from_static(b"payload"))
            .await
            .expect("put_arrow");
        // First delete removes it.
        ws.delete_arrow(wal_id).await.expect("first delete");
        // Re-fetch now fails (object gone) — confirms the delete landed.
        ws.get_arrow(wal_id, None)
            .await
            .expect_err("sidecar must be gone after delete");
        // Second delete against the absent path is still Ok.
        ws.delete_arrow(wal_id).await.expect("second delete");
    }

    // ---- list_wal_ids enumerates + sorts state docs ----------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn list_wal_ids_returns_only_state_docs_sorted_ascending() {
        let (_dir, ws) = store();
        // Create three state docs out of ascending order.
        for id in [50i128, 10, 30] {
            ws.create(&sample_state(id)).await.expect("create");
        }
        // Drop an `.arrow` sidecar under the same prefix — it must
        // NOT appear in the list (only `.json` state docs do).
        ws.put_arrow(WalId(99), Bytes::from_static(b"ignored"))
            .await
            .expect("put_arrow");

        let ids = ws.list_wal_ids().await.expect("list");
        let raw: Vec<i128> = ids.iter().map(|w| w.0).collect();
        assert_eq!(raw, vec![10, 30, 50], "ascending oldest-first order");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn list_wal_ids_on_empty_prefix_is_empty() {
        let (_dir, ws) = store();
        let ids = ws.list_wal_ids().await.expect("list");
        assert!(ids.is_empty());
    }

    // ---- list_tombstone_ids enumerates sidecars --------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn list_tombstone_ids_returns_superfiles_with_sidecars() {
        let (_dir, ws) = store();
        let a = uuid::Uuid::from_u128(0x1111);
        let b = uuid::Uuid::from_u128(0x2222);
        let empty = TombstonesSidecar {
            seal: None,
            bitmap: roaring::RoaringBitmap::new(),
        };
        ws.put_tombstones(a, None, &empty).await.expect("put a");
        ws.put_tombstones(b, None, &empty).await.expect("put b");

        let mut ids = ws.list_tombstone_ids().await.expect("list");
        ids.sort();
        let mut expected = vec![a, b];
        expected.sort();
        assert_eq!(ids, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn list_tombstone_ids_on_empty_prefix_is_empty() {
        let (_dir, ws) = store();
        let ids = ws.list_tombstone_ids().await.expect("list");
        assert!(ids.is_empty());
    }

    // ---- put_tombstones create-only via Some(empty etag) -----------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn put_tombstones_with_empty_expected_etag_is_create_only() {
        // `Some(empty)` collapses to `None` (create-only) inside
        // put_tombstones; a first write under it must land, and a
        // second create-only write against the now-present object
        // must lose CAS.
        let (_dir, ws) = store();
        let superfile_id = uuid::Uuid::from_u128(0xABCD);
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        let sidecar = TombstonesSidecar { seal: None, bitmap };
        let empty_etag: Etag = String::new();
        ws.put_tombstones(superfile_id, Some(&empty_etag), &sidecar)
            .await
            .expect("first create-only put lands");
        let err = ws
            .put_tombstones(superfile_id, Some(&empty_etag), &sidecar)
            .await
            .expect_err("second create-only put must lose CAS");
        assert!(matches!(err, WalStoreError::CasFailed { .. }), "{err:?}");
    }

    // ---- get_tombstones surfaces a decode error on garbage bytes ---------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn get_tombstones_surfaces_codec_error_on_corrupt_bytes() {
        let (dir, ws) = store();
        let superfile_id = uuid::Uuid::from_u128(0xDEAD);
        // Write raw garbage directly to the sidecar path, bypassing
        // the encoder, so the decode in `get_tombstones` fails.
        let provider: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        let path = WalStore::tombstones_path(superfile_id);
        provider
            .put_atomic(&path, Bytes::from_static(b"not a valid sidecar"))
            .await
            .expect("write garbage");
        let err = ws
            .get_tombstones(superfile_id)
            .await
            .expect_err("decode must fail");
        assert!(matches!(err, WalStoreError::SidecarCodec { .. }), "{err:?}");
    }
}
