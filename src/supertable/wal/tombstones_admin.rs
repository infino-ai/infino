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

use std::time::Duration;

use uuid::Uuid;

pub use crate::config::DEFAULT_STALE_SEAL_TIMEOUT_MS;
use crate::supertable::wal::{
    persistence::{Etag, WalStore, WalStoreError},
    state_doc::SealRecord,
    tombstones_codec::TombstonesSidecar,
};

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

/// `true` if a seal placed at `sealed_at` is older than
/// `stale_timeout` as of `now`. Shared by [`seal`] (deciding whether
/// to steal it) and the compactor's candidate selection (deciding
/// whether a sealed superfile is even worth proposing again). Callers
/// pass [`DEFAULT_STALE_SEAL_TIMEOUT_MS`] (converted to a `Duration`)
/// unless a table's `CompactionSettings::stale_seal_timeout_ms`
/// overrides it.
pub fn is_seal_stale(
    sealed_at: chrono::DateTime<chrono::Utc>,
    now: chrono::DateTime<chrono::Utc>,
    stale_timeout: Duration,
) -> bool {
    let age = (now - sealed_at).to_std().unwrap_or(Duration::ZERO);
    age >= stale_timeout
}

/// Atomically stamp the seal flag on a per-superfile tombstone
/// sidecar. Returns the sealed sidecar plus the etag the PUT (or,
/// on the idempotent no-op branch, the GET) landed under -- callers
/// that later call [`unseal`] pass that etag straight through, no
/// re-read needed.
///
/// Behaviour:
///
/// - Sidecar absent (404) → create a fresh sealed sidecar with
///   an empty bitmap.
/// - Sidecar present + unsealed → preserve the existing bitmap,
///   stamp the seal, CAS-PUT.
/// - Sidecar present + sealed by **the same `compaction_id`** →
///   no-op, return the existing sidecar.
/// - Sidecar present + sealed by a different compaction, not
///   stale yet → `AlreadySealed`, someone else's live work, leave
///   it alone.
/// - Sidecar present + sealed by a different compaction, stale →
///   that compactor is presumed dead, take the seal over.
/// - CAS-loss on the PUT → `CasLost` so the caller can re-read +
///   retry.
pub async fn seal(
    wal_store: &WalStore,
    superfile_id: Uuid,
    compaction_id: Uuid,
    sealed_at: chrono::DateTime<chrono::Utc>,
    stale_timeout: Duration,
) -> Result<(TombstonesSidecar, Etag), TombstonesAdminError> {
    let (existing, etag_opt) = match wal_store.get_tombstones(superfile_id).await? {
        Some((sc, etag)) => (Some(sc), Some(etag)),
        None => (None, None),
    };

    if let Some(existing) = &existing
        && let Some(existing_seal) = existing.seal.as_ref()
    {
        if existing_seal.compaction_id == compaction_id {
            let etag = etag_opt.expect("sidecar present implies its GET returned an etag");
            return Ok((existing.clone(), etag));
        }
        if !is_seal_stale(existing_seal.sealed_at, sealed_at, stale_timeout) {
            return Err(TombstonesAdminError::AlreadySealed {
                superfile_id,
                existing_compaction_id: existing_seal.compaction_id,
            });
        }
        // Stale: that compactor is presumed dead, fall through and
        // take the seal over.
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
        Ok(new_etag) => Ok((sealed, new_etag)),
        Err(WalStoreError::CasFailed { .. }) => Err(TombstonesAdminError::CasLost { superfile_id }),
        Err(other) => Err(other.into()),
    }
}

/// Clear a seal previously placed by [`seal`], using the bitmap and
/// etag that call returned. No GET: one CAS-PUT. If the sidecar has
/// changed since (a writer bypassed our now-stale seal, or another
/// compactor stole it), the CAS fails and we no-op -- whatever
/// changed it already resolved the seal, there's nothing left for us
/// to clear.
pub async fn unseal(
    wal_store: &WalStore,
    superfile_id: Uuid,
    bitmap: roaring::RoaringBitmap,
    etag: &Etag,
) -> Result<(), TombstonesAdminError> {
    let unsealed = TombstonesSidecar { seal: None, bitmap };
    match wal_store
        .put_tombstones(superfile_id, Some(etag), &unsealed)
        .await
    {
        Ok(_) => Ok(()),
        Err(WalStoreError::CasFailed { .. }) => Ok(()),
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
    use std::sync::Arc;

    use chrono::Utc;
    use tempfile::TempDir;

    use super::*;
    use crate::storage::{LocalFsStorageProvider, StorageProvider};

    const DEFAULT_STALE_SEAL_TIMEOUT: Duration =
        Duration::from_millis(DEFAULT_STALE_SEAL_TIMEOUT_MS);

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
        let (sealed, _etag) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");
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
        let (sealed, _etag) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");
        assert_eq!(sealed.bitmap, bitmap, "seal must preserve the bitmap");
        assert_eq!(sealed.seal.expect("set").compaction_id, cid);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_is_idempotent_on_same_compaction_id() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x300);
        let cid = Uuid::from_u128(0xDEAD);
        let (first, _etag1) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-1");
        let (again, _etag2) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-2");
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
    async fn unseal_clears_seal_for_matching_compaction_id() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x300);
        let cid = Uuid::from_u128(0xBEEF);
        let (sidecar, etag) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");
        unseal(&ws, sf, sidecar.bitmap, &etag)
            .await
            .expect("unseal");
        let (post, _etag) = ws.get_tombstones(sf).await.expect("get").expect("present");
        assert!(post.seal.is_none(), "seal cleared");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_on_different_compaction_id_surfaces_already_sealed() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x400);
        let cid_a = Uuid::from_u128(0x1111);
        let cid_b = Uuid::from_u128(0x2222);
        let _ = seal(&ws, sf, cid_a, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-a");
        let err = seal(&ws, sf, cid_b, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
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
    async fn seal_steals_a_stale_seal_from_a_different_compaction() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x450);
        let cid_a = Uuid::from_u128(0x1111);
        let cid_b = Uuid::from_u128(0x2222);
        let old_time = Utc::now()
            - chrono::Duration::from_std(DEFAULT_STALE_SEAL_TIMEOUT).unwrap_or_default()
            - chrono::Duration::seconds(1);
        let _ = seal(&ws, sf, cid_a, old_time, DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-a");

        // cid_a's seal is older than the timeout, so it's presumed
        // dead and cid_b can take over.
        let (stolen, _etag) = seal(&ws, sf, cid_b, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-b should steal the stale seal");
        assert_eq!(stolen.seal.expect("set").compaction_id, cid_b);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn seal_does_not_steal_a_fresh_seal() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x460);
        let cid_a = Uuid::from_u128(0x1111);
        let cid_b = Uuid::from_u128(0x2222);
        let _ = seal(&ws, sf, cid_a, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-a");

        // cid_a's seal is fresh; cid_b must not be able to take over.
        let err = seal(&ws, sf, cid_b, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect_err("must not steal a fresh seal");
        assert!(matches!(err, TombstonesAdminError::AlreadySealed { .. }));
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
        let _ = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");
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
        let _ = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");

        // Live-rows excludes the tombstoned row.
        let rows = live_rows(&ws, sf, 5).await.expect("live");
        assert_eq!(rows, vec![0u32, 1, 2, 4]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unseal_clears_our_own_seal() {
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x900);
        let cid = Uuid::from_u128(0xAAAA);
        let (sealed, etag) = seal(&ws, sf, cid, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal");

        unseal(&ws, sf, sealed.bitmap, &etag).await.expect("unseal");

        let (after, _etag) = ws.get_tombstones(sf).await.expect("get").expect("present");
        assert!(after.seal.is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unseal_no_ops_when_the_seal_has_since_changed() {
        // Simulates: our seal went stale mid-merge and a different
        // compactor stole it (or a writer bypassed it) before we got
        // around to unsealing. Our stale etag must not be able to
        // clobber whatever is there now.
        let (_dir, ws) = fixture();
        let sf = Uuid::from_u128(0x901);
        let cid_a = Uuid::from_u128(0xAAAA);
        let cid_b = Uuid::from_u128(0xBBBB);
        let (sealed_a, etag_a) = seal(&ws, sf, cid_a, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-a");

        // Someone else re-seals under a different compaction_id,
        // moving the etag forward.
        let old_time = Utc::now()
            - chrono::Duration::from_std(DEFAULT_STALE_SEAL_TIMEOUT).unwrap_or_default()
            - chrono::Duration::seconds(1);
        // Force cid_a's seal to look stale so cid_b can steal it,
        // simulating the time-of-check having since moved on.
        ws.put_tombstones(
            sf,
            Some(&etag_a),
            &TombstonesSidecar {
                seal: Some(SealRecord {
                    compaction_id: cid_a,
                    sealed_at: old_time,
                }),
                bitmap: sealed_a.bitmap.clone(),
            },
        )
        .await
        .expect("backdate seal-a");
        let _ = seal(&ws, sf, cid_b, Utc::now(), DEFAULT_STALE_SEAL_TIMEOUT)
            .await
            .expect("seal-b steals the now-stale seal");

        // cid_a's unseal, using its now-stale etag, must no-op rather
        // than clobber cid_b's live seal.
        unseal(&ws, sf, sealed_a.bitmap, &etag_a)
            .await
            .expect("unseal must no-op, not error");

        let (after, _etag) = ws.get_tombstones(sf).await.expect("get").expect("present");
        assert_eq!(after.seal.expect("still sealed").compaction_id, cid_b);
    }
}
