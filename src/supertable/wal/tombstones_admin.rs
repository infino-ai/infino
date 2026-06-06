// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Infino Authors

//! Compaction-facing helpers for per-superfile tombstone
//! sidecars.
//!
//! These are the two operations a tombstone-aware compactor
//! needs:
//!
//! - [`seal`] — atomically stamp the seal flag on a sidecar so
//!   no further writers can land bits there. Used at the start
//!   of compaction's "freeze the sources" step.
//! - [`live_rows`] — read the (possibly sealed) sidecar and
//!   return the non-tombstoned doc-id set the compactor will
//!   include in the merged target.
//!
//! The seal is monotonic by construction: once `seal.is_some()`,
//! [`super::pipeline::cas_tombstone_bit`] (the writer's CAS
//! loop) detects it on the next GET and returns `Sealed` to its
//! caller, which then re-resolves against the manifest.
//! Compaction completes by publishing the merged superfile +
//! removing the source ids from the manifest in one CAS pass;
//! after that swap, the writer's re-resolve routes to the
//! merged target and lands the tombstone there.

use uuid::Uuid;

use crate::supertable::wal::persistence::{WalStore, WalStoreError};
use crate::supertable::wal::state_doc::SealRecord;
use crate::supertable::wal::tombstones_codec::TombstonesSidecar;

/// Typed failures from the compaction helpers.
#[derive(Debug, thiserror::Error)]
pub enum TombstonesAdminError {
    /// The sidecar already carried a `seal.is_some()` when we
    /// tried to seal it. Most often this means a previous,
    /// abandoned compaction left the seal in place; the
    /// compactor must drive the abandoned merge to completion
    /// (or unwind it) before sealing again.
    #[error(
        "tombstone sidecar for {superfile_id} is already sealed (compaction_id={existing_compaction_id})"
    )]
    AlreadySealed {
        superfile_id: Uuid,
        existing_compaction_id: Uuid,
    },

    /// CAS race lost between our GET + our PUT. A writer landed
    /// a tombstone bit between us reading the sidecar and us
    /// writing the sealed variant. Compaction should retry the
    /// seal after re-reading the manifest.
    #[error("CAS race lost while sealing sidecar for {superfile_id}")]
    CasLost { superfile_id: Uuid },

    /// Underlying WAL store I/O failure.
    #[error("wal store error: {0}")]
    WalStore(#[from] WalStoreError),
}

/// Atomically stamp the seal flag on a per-superfile tombstone
/// sidecar. Idempotent only on the same `compaction_id`:
/// repeating the seal call with a different `compaction_id`
/// surfaces `AlreadySealed` so the caller can pick up the
/// abandoned merge.
///
/// Behaviour:
///
/// - Sidecar absent (404) → create a fresh sealed sidecar with
///   an empty bitmap.
/// - Sidecar present + unsealed → preserve the existing bitmap,
///   stamp the seal, CAS-PUT.
/// - Sidecar present + sealed by **the same `compaction_id`** →
///   no-op success; return the existing sidecar. Lets the
///   compactor's recovery path replay seal calls safely.
/// - Sidecar present + sealed by a different compaction →
///   `AlreadySealed` so the compactor knows to recover that
///   work first.
/// - CAS-loss on the PUT → `CasLost` so the compactor can
///   re-read + decide whether to retry (typically yes, after
///   re-reading the manifest).
pub async fn seal(
    wal_store: &WalStore,
    superfile_id: Uuid,
    compaction_id: Uuid,
    sealed_at: chrono::DateTime<chrono::Utc>,
) -> Result<TombstonesSidecar, TombstonesAdminError> {
    let (existing, etag_opt) = match wal_store.get_tombstones(superfile_id).await? {
        Some((sc, etag)) => (Some(sc), Some(etag)),
        None => (None, None),
    };

    // Already sealed → idempotent on the same compaction id,
    // error otherwise. The mismatched-id case means a previous
    // compaction sealed this sidecar and didn't finish; the
    // caller has to drive that abandoned merge to completion
    // (or unwind it) before sealing again with a fresh id.
    if let Some(existing) = &existing
        && let Some(existing_seal) = existing.seal.as_ref()
    {
        if existing_seal.compaction_id == compaction_id {
            return Ok(existing.clone());
        }
        return Err(TombstonesAdminError::AlreadySealed {
            superfile_id,
            existing_compaction_id: existing_seal.compaction_id,
        });
    }

    let bitmap = existing
        .map(|sc| sc.bitmap)
        .unwrap_or_else(roaring::RoaringBitmap::new);
    let sealed = TombstonesSidecar {
        seal: Some(SealRecord {
            compaction_id,
            sealed_at,
        }),
        bitmap,
    };

    match wal_store
        .put_tombstones(superfile_id, etag_opt.as_ref(), &sealed)
        .await
    {
        Ok(_new_etag) => Ok(sealed),
        Err(WalStoreError::CasFailed { .. }) => Err(TombstonesAdminError::CasLost { superfile_id }),
        Err(other) => Err(other.into()),
    }
}

/// Return the local doc-ids of `superfile_id` that are NOT in
/// the sidecar's bitmap, scoped to `[0, n_docs)`. Used by the
/// compactor when building the merged target so tombstoned rows
/// are dropped on the floor.
///
/// Absent sidecar → every doc-id in `[0, n_docs)` is live.
/// Sealed-or-unsealed makes no difference here: the compactor
/// reads the bitmap and excludes its bits.
///
/// O(n_docs) allocation. Compactors that want to stream large
/// superfiles should call this once per source superfile and
/// iterate the returned `Vec` lazily; the bitmap inside the
/// sidecar is small (Roaring is sparse) so the wall-clock cost
/// is dominated by the iteration, not the GET.
pub async fn live_rows(
    wal_store: &WalStore,
    superfile_id: Uuid,
    n_docs: u32,
) -> Result<Vec<u32>, TombstonesAdminError> {
    let bitmap = match wal_store.get_tombstones(superfile_id).await? {
        Some((sc, _etag)) => sc.bitmap,
        None => roaring::RoaringBitmap::new(),
    };
    let mut out: Vec<u32> = Vec::with_capacity(n_docs as usize);
    for doc_id in 0..n_docs {
        if !bitmap.contains(doc_id) {
            out.push(doc_id);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};
    use chrono::Utc;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn fixture() -> (TempDir, WalStore) {
        let dir = TempDir::new().expect("tempdir");
        let storage: Arc<dyn StorageProvider> =
            Arc::new(LocalFsStorageProvider::new(dir.path()).expect("provider"));
        (dir, WalStore::new(storage))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_on_absent_sidecar_creates_sealed_empty() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x100);
        let cid = Uuid::from_u128(0xC0DE);
        let sealed = seal(&ws, sf, cid, Utc::now()).await.expect("seal");
        assert_eq!(sealed.seal.expect("set").compaction_id, cid);
        assert!(sealed.bitmap.is_empty());

        // Persisted on disk.
        let (post, _etag) = ws.get_tombstones(sf).await.expect("get").expect("present");
        assert_eq!(post.seal.expect("set").compaction_id, cid);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_preserves_existing_bitmap() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x200);
        // Pre-write an unsealed sidecar with 3 bits.
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(5);
        bitmap.insert(7);
        ws.put_tombstones(
            sf,
            None,
            &TombstonesSidecar {
                seal: None,
                bitmap: bitmap.clone(),
            },
        )
        .await
        .expect("seed");

        let cid = Uuid::from_u128(0xABCD);
        let sealed = seal(&ws, sf, cid, Utc::now()).await.expect("seal");
        assert_eq!(sealed.bitmap, bitmap, "seal must preserve the bitmap");
        assert_eq!(sealed.seal.expect("set").compaction_id, cid);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_is_idempotent_on_same_compaction_id() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x300);
        let cid = Uuid::from_u128(0xDEAD);
        let first = seal(&ws, sf, cid, Utc::now()).await.expect("seal-1");
        let again = seal(&ws, sf, cid, Utc::now()).await.expect("seal-2");
        // SealRecord's sealed_at goes through ms-precision
        // truncation on disk so we compare the
        // compaction-identifying fields only, not the timestamp.
        assert_eq!(
            first.seal.as_ref().expect("set").compaction_id,
            again.seal.as_ref().expect("set").compaction_id
        );
        assert_eq!(first.bitmap, again.bitmap);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_on_different_compaction_id_surfaces_already_sealed() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x400);
        let cid_a = Uuid::from_u128(0x1111);
        let cid_b = Uuid::from_u128(0x2222);
        let _ = seal(&ws, sf, cid_a, Utc::now()).await.expect("seal-a");
        let err = seal(&ws, sf, cid_b, Utc::now())
            .await
            .expect_err("must error");
        assert!(matches!(
            err,
            TombstonesAdminError::AlreadySealed {
                existing_compaction_id,
                ..
            } if existing_compaction_id == cid_a
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_rows_absent_sidecar_returns_full_range() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x500);
        let rows = live_rows(&ws, sf, 5).await.expect("live");
        assert_eq!(rows, vec![0u32, 1, 2, 3, 4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_rows_excludes_tombstoned_bits() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x600);
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(3);
        ws.put_tombstones(sf, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("seed");
        let rows = live_rows(&ws, sf, 5).await.expect("live");
        assert_eq!(rows, vec![0u32, 2, 4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_rows_works_on_sealed_sidecar() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x700);
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(2);
        ws.put_tombstones(sf, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("seed");
        let cid = Uuid::from_u128(0xC0DEC0DE);
        let _ = seal(&ws, sf, cid, Utc::now()).await.expect("seal");
        let rows = live_rows(&ws, sf, 4).await.expect("live");
        assert_eq!(rows, vec![0u32, 1, 3]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn race_writer_then_seal_landed_tombstone_visible_to_compactor() {
        // Race-window safety property: a writer's tombstone
        // bit lands BEFORE the compactor seals the sidecar. The
        // compactor's post-seal `live_rows` therefore excludes
        // the tombstoned row — the merged target won't carry it.
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x800);

        // Writer-side: land a tombstone at doc_id=3 via the
        // codec layer directly (mimicking what the WAL pipeline
        // does internally).
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(3);
        ws.put_tombstones(sf, None, &TombstonesSidecar { seal: None, bitmap })
            .await
            .expect("writer wrote");

        // Compactor-side: seal afterwards.
        let cid = Uuid::from_u128(0xC0DEFACE);
        let _ = seal(&ws, sf, cid, Utc::now()).await.expect("seal");

        // Live-rows excludes the tombstoned row.
        let rows = live_rows(&ws, sf, 5).await.expect("live");
        assert_eq!(rows, vec![0u32, 1, 2, 4]);
    }
}
